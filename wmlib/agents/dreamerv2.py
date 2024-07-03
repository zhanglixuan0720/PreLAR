import torch

from . import expl
from .. import core, nets
from .actor_critic import ActorCritic
from .base import BaseAgent, BaseWorldModel
from einops import rearrange

class DreamerV2(BaseAgent):

    def __init__(self, config, obs_space, act_space, step):
        super(DreamerV2, self).__init__(config, obs_space, act_space, step)
        self.wm = WorldModel(config, obs_space, act_space,self.step,self.enable_fp16)
        self._task_behavior = ActorCritic(config, self.act_space, self.step,self.enable_fp16)

        self.init_expl_behavior()

    def policy(self, obs, state=None, mode="train"):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.enable_fp16):
                if state is None:
                    latent = self.wm.rssm.initial(len(obs["reward"]), obs["reward"].device)
                    action = torch.zeros((len(obs["reward"]),) + self.act_space.shape).to(obs["reward"].device)
                    state = latent, action
                latent, action = state
                embed = self.wm.encoder(self.wm.preprocess(obs))
                sample = (mode == "train") or not self.config.eval_state_mean
                latent, _ = self.wm.rssm.obs_step(
                    latent, action, embed, obs["is_first"], sample)
                feat = self.wm.rssm.get_feat(latent)
                action = self.get_action(feat, mode)
                outputs = {"action": action.cpu()}
                state = (latent, action)

        return outputs, state

    def report(self, data):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.enable_fp16):
                report = {}
                data = self.wm.preprocess(data)
                for key in self.wm.heads["decoder"].cnn_keys:
                    name = key.replace("/", "_")
                    report[f"openl_{name}"] = self.wm.video_pred(data, key).detach().cpu().numpy()
                return report

    def init_optimizers(self):
        wm_modules = [self.wm.encoder.parameters(), self.wm.rssm.parameters(),
                      *[head.parameters() for head in self.wm.heads.values()]]
        self.wm.model_opt = core.Optimizer("model", wm_modules, enable_fp16=self.enable_fp16,**self.config.model_opt)

        self._task_behavior.actor_opt = core.Optimizer("actor", self._task_behavior.actor.parameters(), enable_fp16=self.enable_fp16,
                                                       **self.config.actor_opt)
        self._task_behavior.critic_opt = core.Optimizer("critic", self._task_behavior.critic.parameters(), enable_fp16=self.enable_fp16,
                                                        **self.config.critic_opt)


class WorldModel(BaseWorldModel):

    def __init__(self, config, obs_space,act_space, step, enable_fp16 = False):
        super(WorldModel, self).__init__()

        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        self.config = config
        self.enable_fp16 = enable_fp16
        self.step = step
        self.rssm = nets.EnsembleRSSM(action_dim=act_space['action'].shape[0],**config.rssm)

        if self.config.encoder_type == 'plaincnn':
            self.encoder = nets.PlainCNNEncoder(shapes, **config.encoder)
        elif self.config.encoder_type == 'resnet':
            self.encoder = nets.ResNetEncoder(shapes, **config.encoder)
        else:
            raise NotImplementedError

        self.heads = torch.nn.ModuleDict()
        if self.config.decoder_type == 'plaincnn':
            self.heads["decoder"] = nets.PlainCNNDecoder(shapes, **config.decoder)
        elif self.config.decoder_type == 'resnet':
            self.heads["decoder"] = nets.ResNetDecoder(shapes, **config.decoder)
        else:
            raise NotImplementedError
        self.heads["reward"] = nets.MLP([], **config.reward_head)
        if config.pred_discount:
            self.heads["discount"] = nets.MLP([], **config.discount_head)
        for name in config.grad_heads:
            assert name in self.heads, name

        if self.config.beta != 0:
            self.intr_bonus = expl.VideoIntrBonus(
                config.beta, config.k, config.intr_seq_length,
                config.rssm.deter + config.rssm.stoch * config.rssm.discrete,
                config.queue_dim,
                config.queue_size,
                config.intr_reward_norm,
                config.beta_type,
            )

        self.model_opt = core.EmptyOptimizer()

    def train_iter(self, data, state=None):
        with torch.cuda.amp.autocast(enabled=self.enable_fp16):
            self.zero_grad(set_to_none=True)  # delete grads
            model_loss, state, outputs, metrics = self.loss(data, state)

        # Backward passes under autocast are not recommended.
        self.model_opt.backward(model_loss)
        metrics.update(self.model_opt.step(model_loss))
        metrics["model_loss"] = model_loss.item()
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm.observe(embed, data["action"], data["is_first"], state)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        assert len(kl_loss.shape) == 0
        likes = {}
        losses = {"kl": kl_loss}
        feat = self.rssm.get_feat(post)

        if self.config.beta != 0:
            data, intr_rew_len, int_rew_mets = self.intr_bonus.compute_bonus(data, feat)

        for name, head in self.heads.items():
            grad_head = (name in self.config.grad_heads)
            inp = feat if grad_head else feat.detach()
            if name == "reward" and self.config.beta != 0:
                inp = inp[:, :intr_rew_len]
            out = head(inp)
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                # NOTE: for bernoulli log_prob with float values (data["discount"]) means binary_cross_entropy_with_logits
                like = dist.log_prob(data[key])
                likes[key] = like
                losses[key] = -like.mean()
        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items()
        )
        outs = dict(
            embed=embed, feat=feat, post=post, prior=prior, likes=likes, kl=kl_value
        )
        metrics = {f"{name}_loss": value.detach().cpu().item() for name, value in losses.items()}
        metrics["model_kl"] = kl_value.mean().item()
        metrics["prior_entropy"] = self.rssm.get_dist(prior).entropy().mean().item()
        metrics["post_entropy"] = self.rssm.get_dist(post).entropy().mean().item()
        if self.config.beta != 0:
            metrics.update(**int_rew_mets)
        last_state = {k: v[:, -1] for k, v in post.items()}
        return model_loss, last_state, outs, metrics

    def video_pred(self, data, key):
        decoder = self.heads["decoder"]
        truth = data[key][:6] + 0.5
        embed = self.encoder(data)
        states, _ = self.rssm.observe(embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5])
        recon = decoder(self.rssm.get_feat(states))[key].mode[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data["action"][:6, 5:], init)
        openl = decoder(self.rssm.get_feat(prior))[key].mode
        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = torch.cat([truth, model, error], 3)
        # B, T, C, H, W = video.shape
        return rearrange(video, 'b t c h w -> t c h (b w) ') #video.permute((1, 3, 0, 4, 2)).reshape((T, H, B * W, C))
