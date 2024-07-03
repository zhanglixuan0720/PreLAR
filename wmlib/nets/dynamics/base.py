
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
from torchtyping import TensorType, patch_typeguard
from typing import Dict, Tuple, Union
from typeguard import typechecked
from collections import OrderedDict
from einops.layers.torch import Rearrange

from ... import core
from ...core import dists
from ..modules import get_act_module, NormLayer


State = Dict[str, torch.Tensor]  # FIXME to be more specified


class BaseDynamics(nn.Module, ABC):

    def __init__(
        self,
        action_free=False,
        fill_action=None,  # 50 in apv
        ensemble=5,
        stoch=30,
        deter=200,
        hidden=200,
        discrete=False,
        act="elu",
        norm="none",
        std_act="softplus",
        min_std=0.1,
    ):
        super().__init__()
        self._action_free = action_free
        self._fill_action = fill_action
        self._ensemble = ensemble
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        # self._act = get_act(act)
        self._act_module = get_act_module(act)
        self._norm = norm
        self._std_act = std_act
        self._min_std = min_std
        self._std_fn = {
                "softplus": lambda std: F.softplus(std),
                "sigmoid": lambda std: torch.sigmoid(std),
                "sigmoid2": lambda std: 2 * torch.sigmoid(std / 2),
            }[self._std_act]
        self._ensemble_out_layers = nn.ModuleList()
        for k in range(self._ensemble):
            self._ensemble_out_layers.append(nn.Sequential(OrderedDict([
                (f'img_out_{k}', nn.Linear(self._hidden, self._hidden)),
                (f'img_out_norm_{k}', NormLayer(self._norm, self._hidden)),
                (f'img_out_act_{k}', self._act_module()) 
            ])))
            if self._discrete:
                self._ensemble_out_layers[-1].add_module(f'img_out_head_{k}', nn.Linear(self._hidden, self._stoch * self._discrete))
                self._ensemble_out_layers[-1].add_module(f'rerange_{k}', Rearrange('... (s d) -> ... s d', s=self._stoch, d=self._discrete))
            else:
                self._ensemble_out_layers[-1].add_module(f'img_out_head_{k}', nn.Linear(self._hidden, 2 * self._stoch))
        


    @abstractmethod
    def initial(self, batch_size: int, device) -> State:
        pass

    def fill_action_with_zero(self, action):
        # action: [*B, action]
        B, D = action.shape[:-1], action.shape[-1]
        if self._action_free:
            return torch.zeros([*B, self._fill_action]).to(action.device)
        else:
            if self._fill_action is not None:
                zeros = torch.zeros([*B, self._fill_action - D]).to(action.device)
                return torch.cat([action, zeros], axis=1)
            else:
                # doing nothing
                return action

    @abstractmethod
    def observe(
        self,
        embed: TensorType["batch", "seq", "emb_dim"],
        action: TensorType["batch", "seq", "act_dim"],
        is_first,
        state: State = None
    ):
        pass

    def imagine(
        self,
        action: TensorType["batch", "seq", "act_dim"],
        state: State = None
    ):
        # a permute of (batch, sequence) to (sequence, batch)
        swap = lambda x: torch.permute(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(action.shape[0], action.device)
        assert isinstance(state, dict), state
        action = swap(action)
        prior = core.sequence_scan(self.img_step, state, action)[0]
        prior = {k: swap(v) for k, v in prior.items() if k != "mems"}
        return prior

    def get_feat(self, state):
        """
        gets stoch and deter as tensor
        """

        # FIXME verify shapes of this function
        stoch = state["stoch"]
        if self._discrete:
            stoch = torch.reshape(stoch, (*stoch.shape[:-2], self._stoch * self._discrete))
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state: State):
        """
        gets the stochastic state distribution
        """
        if self._discrete:
            logit = state["logit"]
            logit = logit.float()
            dist = dists.Independent(dists.OneHotDist(logit), 1)
        else:
            mean, std = state["mean"], state["std"]
            mean = mean.float()
            std = std.float()
            dist = dists.Independent(dists.Normal(mean, std), 1)
        return dist

    @abstractmethod
    def obs_step(
        self,
        prev_state: State,
        prev_action: TensorType["batch", "act_dim"],
        embed: TensorType["batch", "emb_dim"],
        is_first: TensorType["batch"],
        sample=True,
    ) -> Tuple[State, State]:
        pass

    @abstractmethod
    def img_step(
        self,
        prev_state: State,
        prev_action: TensorType["batch", "act_dim"],
        sample=True,
    ) -> State:
        pass

    def kl_loss(self, post: State, prior: State, forward: bool, balance: float, free: float, free_avg: bool):
        kld = tdist.kl_divergence
        sg = core.dict_detach
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)

        free = torch.tensor(free)
        if balance == 0.5:
            value = kld(self.get_dist(lhs), self.get_dist(rhs))
            loss = torch.maximum(value, free).mean()
        else:
            value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
            value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
            if free_avg:
                loss_lhs = torch.maximum(value_lhs.mean(), free)
                loss_rhs = torch.maximum(value_rhs.mean(), free)
            else:
                loss_lhs = torch.maximum(value_lhs, free).mean()
                loss_rhs = torch.maximum(value_rhs, free).mean()
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss, value

    def _suff_stats_ensemble(self, inp: TensorType["batch", "hidden"]):
        bs = list(inp.shape[:-1])
        assert len(bs) == 1, bs
        inp = inp.reshape([-1, inp.shape[-1]])
        stats = []
        for k in range(self._ensemble):
            x = self._ensemble_out_layers[k](inp)
            stats.append(self._suff_stats_layer(x))
        stats = {
            k: torch.stack([x[k] for x in stats], 0)
            for k in stats[0].keys()
        }
        return stats

    def _suff_stats_layer(self, x: TensorType["batch", "hidden"]):
        if self._discrete:
            return {"logit": x}
        else:
            mean, std = torch.chunk(x, 2, -1)
            std = self._std_fn(std)
            std = std + self._min_std
            return {"mean": mean, "std": std}
