import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange, unpack

from .base import BaseDecoder
from ..modules import get_act_module, NormLayer
from ... import core


class PlainCNNDecoder(BaseDecoder):

    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="elu",
        norm="none",
        cnn_depth=48,
        cnn_input_dim = 2048,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=[400, 400, 400, 400],
        **dummy_kwargs,
    ):
        super().__init__(shapes, cnn_keys, mlp_keys, mlp_layers)

        # self._act = get_act(act)
        self._act_module = get_act_module(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._cnn_input_dim = cnn_input_dim
        cnn_out_channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
        self.convin = nn.Sequential(nn.Linear(self._cnn_input_dim, 32 * self._cnn_depth),Rearrange('b t c -> (b t) c 1 1'))
        self._cnn_nn = nn.Sequential()
        h, w = 1, 1
        for i, kernel in enumerate(self._cnn_kernels):
            depth = int(2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth)
            h, w = (h - 1) * 2 + kernel, (2 - 1) * 2 + kernel # (h−1)∗s+k−2∗p
            input_channel = depth * 2 if i else 32 * self._cnn_depth
            act_module, norm = self._act_module, self._norm
            if i == len(self._cnn_kernels) - 1:
                depth, act_module, norm = sum(cnn_out_channels.values()), get_act_module('none'), 'none'
            self._cnn_nn.add_module(f"conv{i}", nn.ConvTranspose2d(input_channel, depth, kernel, 2))
            self._cnn_nn.add_module(f"convnorm{i}", NormLayer(norm, (depth, h, w)))
            self._cnn_nn.add_module(f"act{i}", act_module())
        self._cnn_out_ps = [[out_channel] for out_channel in cnn_out_channels.values()]
        



    def _cnn(self, features):
        x = self.convin(features).to(memory_format=torch.channels_last)
        x = self._cnn_nn(x)
        x = rearrange(x,'(b t) c h w -> b t c h w',b=features.shape[0])
        # means = torch.split(x, list(self._cnn_out_channels.values()), 2)
        means = unpack(x, self._cnn_out_ps, 'b t * h w ')

        dists = {
            key: core.dists.Independent(core.dists.MSE(mean), 3)
            for key, mean in zip(self.cnn_keys, means)
        }
        return dists




        # channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
        # ConvT = nn.ConvTranspose2d
        # x = self.get("convin", nn.Linear, features.shape[-1], 32 * self._cnn_depth)(features)
        # x = torch.reshape(x, [-1, 32 * self._cnn_depth, 1, 1]).to(memory_format=torch.channels_last)

        # for i, kernel in enumerate(self._cnn_kernels):
        #     depth = 2 ** (len(self._cnn_kernels) - i - 2) * self._cnn_depth
        #     act, norm = self._act, self._norm
        #     if i == len(self._cnn_kernels) - 1:
        #         depth, act, norm = sum(channels.values()), get_act("none"), "none"
        #     x = self.get(f"conv{i}", ConvT, x.shape[1], depth, kernel, 2)(x)
        #     x = self.get(f"convnorm{i}", NormLayer, norm, x.shape[-3:])(x)
        #     x = act(x)

        # x = x.reshape(features.shape[:-1] + x.shape[1:])
        # means = torch.split(x, list(channels.values()), 2)
        # dists = {
        #     key: core.dists.Independent(core.dists.MSE(mean), 3)
        #     for (key, shape), mean in zip(channels.items(), means)
        # }
        # return dists
