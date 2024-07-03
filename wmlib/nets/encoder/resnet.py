import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .base import BaseEncoder
from ..modules import get_act_module, ResidualStack

class ResNetEncoder(BaseEncoder):

    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="elu",
        cnn_depth=48,
        mlp_layers=[400, 400, 400, 400],
        res_layers=2,
        res_depth=3,
        res_norm='none',
        **dummy_kwargs,
    ):
        super().__init__(shapes, cnn_keys, mlp_keys, mlp_layers)
        self._act_module = get_act_module(act)
        self._cnn_depth = cnn_depth

        self._res_layers = res_layers
        self._res_depth = res_depth
        self._res_norm = res_norm

        h, w, c = self.shapes[self.cnn_keys[0]] # raw image shape
        self._cnn_net = nn.Sequential()
        self._cnn_net.add_module('convin', nn.Conv2d(c, self._cnn_depth, 3, 2, 1))
        self._cnn_net.add_module('act', self._act_module())
        for i in range(self._res_depth):
            depth = 2 ** i * self._cnn_depth
            input_channels = depth // 2 if i else self._cnn_depth
            self._cnn_net.add_module(f"res{i}", ResidualStack(input_channels, depth,
                                                             self._res_layers,
                                                             norm=self._res_norm))
            self._cnn_net.add_module(f"pool{i}", nn.AvgPool2d(2, 2))

        self._cnn_net.add_module('flatten', Rearrange('b c h w -> b (c h w)'))
          


    def _cnn(self, data):
        x = torch.cat(list(data.values()), -1)
        x = x.to(memory_format=torch.channels_last)
        return self._cnn_net(x)
