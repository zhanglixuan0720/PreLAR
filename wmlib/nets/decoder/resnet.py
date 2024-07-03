import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange, unpack

from .base import BaseDecoder
from ..modules import ResidualStack
from ... import core


class ResNetDecoder(BaseDecoder):

    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        cnn_depth=48,
        cnn_input_dim = 2048,
        mlp_layers=[400, 400, 400, 400],
        res_layers=2,
        res_depth=3,
        res_norm='none',
        **dummy_kwargs,
    ):
        super().__init__(shapes, cnn_keys, mlp_keys, mlp_layers)

        self._cnn_depth = cnn_depth
        self._res_layers = res_layers
        self._res_depth = res_depth
        self._res_norm = res_norm
        self._cnn_input_dim = cnn_input_dim

        # L = self._res_depth
        hw = 64 // 2**(self._res_depth + 1)
        cnn_out_channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
        self.convin = nn.Sequential(nn.Linear(self._cnn_input_dim, hw * hw * (2**(self._res_depth - 1)) * self._cnn_depth),Rearrange('b t (c h w)-> (b t) c h w',h=hw,w=hw)) 
        self._cnn_nn = nn.Sequential()
        for i in range(self._res_depth):
            depth = depth // 2 if i else int((2**(self._res_depth - 1)) * self._cnn_depth)
            self._cnn_nn.add_module(f"unpool{i}", nn.UpsamplingNearest2d(scale_factor=2))
            self._cnn_nn.add_module(f"res{i}", ResidualStack(depth, depth//2,
                                                             self._res_layers,
                                                             norm=self._res_norm, dec=True))
        self.convout = nn.ConvTranspose2d(depth//2, sum(cnn_out_channels.values()), 3, 2, 1, output_padding=1)
        self._cnn_out_ps = [[out_channel] for out_channel in cnn_out_channels.values()]


    def _cnn(self, features):
        x = self.convin(features).to(memory_format=torch.channels_last)
        x = self._cnn_nn(x)
        x = self.convout(x)
        x = rearrange(x,'(b t) c h w -> b t c h w',b=features.shape[0])
        # means = torch.split(x, list(self._cnn_out_channels.values()), 2)
        means = unpack(x, self._cnn_out_ps, 'b t * h w ')
        dists = {
            key: core.dists.Independent(core.dists.MSE(mean), 3)
            for key, mean in zip(self.cnn_keys, means)
        }
        return dists


        # channels = {k: self._shapes[k][-1] for k in self.cnn_keys}

        # L = self._res_depth
        # hw = 64 // 2**(self._res_depth + 1)
        # x = self.get("convin", nn.Linear, features.shape[-1], hw * hw * (2**(L - 1)) * self._cnn_depth)(features)
        # x = torch.reshape(x, [-1, (2**(L - 1)) * self._cnn_depth, hw, hw]).to(memory_format=torch.channels_last)
        # for i in range(L):
        #     x = self.get(f"unpool{i}", nn.UpsamplingNearest2d, scale_factor=2)(x)
        #     depth = x.shape[1]
        #     x = self.get(f"res{i}", ResidualStack, depth, depth // 2,
        #                  self._res_layers, norm=self._res_norm, dec=True)(x)

        # depth = sum(channels.values())
        # x = self.get(f"convout", nn.ConvTranspose2d, x.shape[1], depth, 3, 2, 1, output_padding=1)(x)

        # x = x.reshape(features.shape[:-1] + x.shape[1:])
        # means = torch.split(x, list(channels.values()), 2)
        # dists = {
        #     key: core.dists.Independent(core.dists.MSE(mean), 3)
        #     for (key, shape), mean in zip(channels.items(), means)
        # }
        # return dists
