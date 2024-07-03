import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from . import expl
from .. import core


class BaseAgent(nn.Module, ABC):
    def __init__(self, config, obs_space, act_space, step):
        super(BaseAgent, self).__init__()

        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.precision = config.precision
        self.enable_fp16 = self.precision == 16

    def init_expl_behavior(self):
        if self.config.expl_behavior == "greedy":
            self._expl_behavior = self._task_behavior
        else:
            self._expl_behavior = getattr(expl, self.config.expl_behavior)(
                self.config,
                self.act_space,
                self.wm,
                self.step,
                lambda seq: self.wm.heads["reward"](seq["feat"]).mode(),
            )

    def init_modules(self):
        # * Hacky: init modules without optimizers (once in opt)
        with torch.no_grad():
            # bs, sq = 4, max(8, self.config.intr_seq_length)
            bs, sq = 1, max(4, self.config.intr_seq_length)
            if "atari" in self.config.task:
                channels = 1 if self.config.atari_grayscale else 3
            elif "dmc" in self.config.task or "metaworld" in self.config.task or "carla" in self.config.task:
                channels = 3
            else:
                raise NotImplementedError
            actions = self.act_space.shape[0]
            dummy_data = {
                "image": torch.rand(bs, sq, channels, *self.config.render_size),
                "action": torch.rand(bs, sq, actions),
                "reward": torch.rand(bs, sq),
                "is_first": torch.rand(bs, sq),
                "is_last": torch.rand(bs, sq),
                "is_terminal": torch.rand(bs, sq),
            }
            dummy_data["is_first"] = torch.zeros_like(dummy_data["is_first"])
            dummy_data["is_first"][:, 0] = 1.0
            for key in self.obs_space:
                if key not in dummy_data:
                    dummy_data[key] = torch.rand(bs, sq, *self.obs_space[key].shape)
            # TODO: we should not update the model here
            self.train(dummy_data)

    @abstractmethod
    def init_optimizers(self):
        pass

    def train(self, data, state=None):
        metrics = {}
        self.wm.train()
        state, outputs, wm_metrics = self.wm.train_iter(data, state)
        self.wm.eval()
        metrics.update({f'wm/{k}':v for k, v in wm_metrics.items()})

        start = outputs["post"]
        start = core.dict_detach(start)
        reward = lambda seq: self.wm.heads["reward"](seq["feat"]).mode
        behavior_metrics = self._task_behavior.train(self.wm, start, data["is_terminal"], reward)
        metrics.update({f'behavior/{k}':v for k, v in behavior_metrics.items()})
        if self.config.expl_behavior != "greedy":
            behavior_metrics = self._expl_behavior.train(start, outputs, data)[-1]
            metrics.update({"expl/" + key: value for key, value in behavior_metrics.items()})
        return core.dict_detach(state), metrics

    def get_action(self, feat, mode):
        if mode == "eval":
            actor = self._task_behavior.actor(feat)
            action = actor.mode
            noise = self.config.eval_noise
        elif mode == "explore":
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
            noise = self.config.expl_noise
        elif mode == "train":
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
            noise = self.config.expl_noise
        action = core.action_noise(action, noise, self.act_space)
        return action

    def save_all(self, logdir):
        torch.save(self.state_dict(), logdir / "variables.pt")


class BaseWorldModel(nn.Module, ABC):

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_"):
                continue
            if value.dtype == torch.int32:
                value = value.float()
            if value.dtype == torch.uint8:
                # value = value.float() / 255.0 - 0.5
                value = value.float()
            obs[key] = value

        obs["image"] = obs["image"] / 255.0 - 0.5
        if self.config.clip_rewards in ["identity", "sign", "tanh"]:
            obs["reward"] = {
                "identity": (lambda x: x),
                "sign": torch.sign,
                "tanh": torch.tanh,
            }[self.config.clip_rewards](obs["reward"])
        else:
            obs["reward"] /= float(self.config.clip_rewards)
        obs["discount"] = 1.0 - obs["is_terminal"].float()
        obs["discount"] *= self.config.discount
        return obs

    def imagine(self, policy, start, is_terminal, horizon):
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        if self.config.imag_batch != -1:
            index = torch.randperm(len(start["deter"]), device=start["deter"].device)[:self.config.imag_batch]
            select = lambda x: torch.index_select(x, dim=0, index=index)
            start = {k: select(v) for k, v in start.items()}
        start["feat"] = self.rssm.get_feat(start)
        start["action"] = torch.zeros_like(policy(start["feat"]).mode)
        seq = {k: [v] for k, v in start.items()}
        for _ in range(horizon):
            action = policy(seq["feat"][-1].detach()).sample()
            state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self.rssm.get_feat(state)
            for key, value in {**state, "action": action, "feat": feat}.items():
                seq[key].append(value)
        seq = {k: torch.stack(v, 0) for k, v in seq.items()}
        if "discount" in self.heads:
            disc = self.heads["discount"](seq["feat"]).mean
            if is_terminal is not None:
                # Override discount prediction for the first step with the true
                # discount factor from the replay buffer.
                true_first = 1.0 - flatten(is_terminal).to(disc.dtype)
                true_first *= self.config.discount
                disc = torch.cat([true_first[None], disc[1:]], 0)
        else:
            disc = self.config.discount * torch.ones(seq["feat"].shape[:-1]).to(seq["feat"].device)
        seq["discount"] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        seq["weight"] = torch.cumprod(
            torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0
        )
        return seq
