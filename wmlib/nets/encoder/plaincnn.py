import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .base import BaseEncoder
from ..modules import get_act_module, NormLayer

class PlainCNNEncoder(BaseEncoder):

    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        act="elu",
        norm="none",
        cnn_depth=48,
        cnn_kernels=(4, 4, 4, 4),
        mlp_layers=[400, 400, 400, 400],
        **dummy_kwargs,
    ):
        super().__init__(shapes, cnn_keys, mlp_keys, mlp_layers)

        self._act_module = get_act_module(act)
        # self._act = get_act(act)
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels

        h, w, c = self.shapes[self.cnn_keys[0]] # raw image shape
        self._cnn_nn = nn.Sequential()
        for i, kernel in enumerate(self._cnn_kernels):
            depth = 2 ** i * self._cnn_depth
            input_channels = depth // 2 if i else c
            h, w = (h - kernel ) // 2 + 1, (w - kernel) // 2 + 1 # (h - k + 2p) // s + 1
            self._cnn_nn.add_module(f"conv{i}", nn.Conv2d(input_channels, depth, kernel, 2))
            self._cnn_nn.add_module(f"convnorm{i}", NormLayer(self._norm, (depth, h, w)))
            self._cnn_nn.add_module(f"act{i}", self._act_module())
            
        self._cnn_nn.add_module('flatten', Rearrange('b c h w -> b (c h w)'))

    def _cnn(self, data):
        x = torch.cat(list(data.values()), -1)
        x = x.to(memory_format=torch.channels_last)
        return self._cnn_nn(x)
