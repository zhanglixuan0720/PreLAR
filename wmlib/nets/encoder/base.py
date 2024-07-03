
import re
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..modules import  NormLayer


class BaseEncoder(nn.Module, ABC):

    def __init__(
        self,
        shapes,
        cnn_keys=r".*",
        mlp_keys=r".*",
        mlp_layers=[400, 400, 400, 400],
        mlp_input_dim = None,
    ):
        super().__init__()
        self.shapes = shapes
        self.cnn_keys = [
            k for k, v in shapes.items() if re.match(cnn_keys, k) and len(v) == 3
        ]
        self.mlp_keys = [
            k for k, v in shapes.items() if re.match(mlp_keys, k) and len(v) == 1
        ]
        print("Encoder CNN inputs:", list(self.cnn_keys))
        print("Encoder MLP inputs:", list(self.mlp_keys))
        self._mlp_layers = mlp_layers
        
        if self.mlp_keys:
            assert mlp_input_dim is not None
            inpup_dim = mlp_input_dim
            self._mlp_nn = nn.Sequential()
            for i, width in enumerate(self._mlp_layers):
                self._mlp_nn.add_module(f"dense{i}", nn.Linear(inpup_dim, width))
                self._mlp_nn.add_module(f"densenorm{i}", NormLayer(self._norm, width))
                self._mlp_nn.add_module(f"act{i}", self._act_module())
                inpup_dim = width

    def forward(self, data):
        key, shape = list(self.shapes.items())[0]
        batch_dims = data[key].shape[:-len(shape)]
        data = {
            k: torch.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
            for k, v in data.items()
        }
        outputs = []
        if self.cnn_keys:
            outputs.append(self._cnn({k: data[k] for k in self.cnn_keys}))
        if self.mlp_keys:
            outputs.append(self._mlp({k: data[k] for k in self.mlp_keys}))
        output = torch.cat(outputs, -1)
        return output.reshape(batch_dims + output.shape[1:])

    @abstractmethod
    def _cnn(self, data):
        pass

    def _mlp(self, data):
        x = torch.cat(list(data.values()), -1)
        x = self._mlp_nn(x)
        return x
