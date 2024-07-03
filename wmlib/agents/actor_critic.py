import torch
import torch.nn as nn
from .. import core, nets


class ActorCritic(nn.Module):

    def __init__(self, config, act_space, tfstep, enable_fp16=False):
        super(ActorCritic, self).__init__()

        self.config = config
        self.act_space = act_space
        self.tfstep = tfstep
        self.enable_fp16 = enable_fp16
        discrete = hasattr(act_space, "n")
        if self.config.actor.dist == "auto":
            self.config = self.config.update({
                "actor.dist": "onehot" if discrete else "trunc_normal"})
        if self.config.actor_grad == "auto":
            self.config = self.config.update({
                "actor_grad": "reinforce" if discrete else "dynamics"})
        self.actor = nets.MLP(act_space.shape[0], **self.config.actor)
        self.critic = nets.MLP([], **self.config.critic)
        if self.config.slow_target:
            self._target_critic = nets.MLP([], **self.config.critic)
            self._updates = 0
        else:
            self._target_critic = self.critic
        self.actor_opt = core.EmptyOptimizer()
        self.critic_opt = core.EmptyOptimizer()
        self.rewnorm = core.StreamNorm(**self.config.reward_norm)

    def train(self, world_model, start, is_terminal, reward_fn):
        metrics = {}
        hor = self.config.imag_horizon
        # The weights are is_terminal flags for the imagination start states.
        # Technically, they should multiply the losses from the second trajectory
        # step onwards, which is the first imagined step. However, we are not
        # training the action that led into the first step anyway, so we can use
        # them to scale the whole sequence.
        with torch.cuda.amp.autocast(enabled=self.enable_fp16):
            # delete grads
            world_model.zero_grad(set_to_none=True)
            self.actor.zero_grad(set_to_none=True)
            self.critic.zero_grad(set_to_none=True)

            seq = world_model.imagine(self.actor, start, is_terminal, hor)
            if self.config.actor_grad == "reinforce":
                with torch.no_grad():
                    reward = reward_fn(seq)
            else:
                reward = reward_fn(seq)
            seq["reward"], mets1 = self.rewnorm(reward)
            mets1 = {f"reward_{k}": v for k, v in mets1.items()}
            target, mets2 = self.target(seq)
            actor_loss, mets3 = self.actor_loss(seq, target)
            critic_loss, mets4 = self.critic_loss(seq, target)

        # Backward passes under autocast are not recommended.
        self.actor_opt.backward(actor_loss, retain_graph=True)
        self.critic_opt.backward(critic_loss)

        metrics.update(self.actor_opt.step(actor_loss))
        metrics.update(self.critic_opt.step(critic_loss))
        metrics.update(**mets1, **mets2, **mets3, **mets4)
        self.update_slow_target()
        return metrics

    def actor_loss(self, seq, target):
        # Actions:      0   [a1]  [a2]   a3
        #                  ^  |  ^  |  ^  |
        #                 /   v /   v /   v
        # States:     [z0]->[z1]-> z2 -> z3
        # Targets:     t0   [t1]  [t2]
        # Baselines:  [v0]  [v1]   v2    v3
        # Entropies:        [e1]  [e2]
        # Weights:    [ 1]  [w1]   w2    w3
        # Loss:              l1    l2
        metrics = {}
        # Two states are lost at the end of the trajectory, one for the boostrap
        # value prediction and one because the corresponding action does not lead
        # anywhere anymore. One target is lost at the start of the trajectory
        # because the initial state comes from the replay buffer.
        policy = self.actor(seq["feat"][:-2].detach())
        if self.config.actor_grad == "dynamics":
            objective = target[1:]
        elif self.config.actor_grad == "reinforce":
            baseline = self._target_critic(seq["feat"][:-2]).mode
            advantage = (target[1:] - baseline).detach()
            action = (seq["action"][1:-1]).detach()
            objective = policy.log_prob(action) * advantage
        elif self.config.actor_grad == "both":
            baseline = self._target_critic(seq["feat"][:-2]).mode
            advantage = (target[1:] - baseline).detach()
            action = (seq["action"][1:-1]).detach()
            objective = policy.log_prob(action) * advantage
            mix = core.schedule(self.config.actor_grad_mix, self.tfstep)
            objective = mix * target[1:] + (1 - mix) * objective
            metrics["actor_grad_mix"] = mix
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = policy.entropy()
        ent_scale = core.schedule(self.config.actor_ent, self.tfstep)
        objective += ent_scale * ent
        weight = seq["weight"].detach()
        actor_loss = -(weight[:-2] * objective).mean()
        metrics["actor_entropy"] = ent.mean().item()
        metrics["actor_entropy_scale"] = ent_scale
        return actor_loss, metrics

    def critic_loss(self, seq, target):
        # States:     [z0]  [z1]  [z2]   z3
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]   v3
        # Weights:    [ 1]  [w1]  [w2]   w3
        # Targets:    [t0]  [t1]  [t2]
        # Loss:        l0    l1    l2
        dist = self.critic(seq["feat"][:-1].detach())
        target = target.detach()
        weight = seq["weight"].detach()
        critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
        metrics = {"critic": dist.mode.mean().item()}
        return critic_loss, metrics

    def target(self, seq):
        # States:     [z0]  [z1]  [z2]  [z3]
        # Rewards:    [r0]  [r1]  [r2]   r3
        # Values:     [v0]  [v1]  [v2]  [v3]
        # Discount:   [d0]  [d1]  [d2]   d3
        # Targets:     t0    t1    t2
        reward = seq["reward"]
        disc = seq["discount"]
        value = self._target_critic(seq["feat"]).mode
        # Skipping last time step because it is used for bootstrapping.
        target = core.lambda_return(
            reward[:-1], value[:-1], disc[:-1],
            bootstrap=value[-1],
            lambda_=self.config.discount_lambda,
            axis=0)
        metrics = {}
        metrics["critic_slow"] = value.mean().item()
        metrics["critic_target"] = target.mean().item()
        return target, metrics

    def update_slow_target(self):  # polyak update
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                    self.config.slow_target_fraction)
                for s, d in zip(self.critic.parameters(), self._target_critic.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
