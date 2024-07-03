import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange, unpack

from .base import BaseDecoder
from ..modules import  ResidualStack
from ... import core

from einops import rearrange, repeat

class DecoupledResNetDecoder(BaseDecoder):

    # TODO: remame args
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
        deco_attmask=0.75,
        deco_attmaskwarmup=-1,
        **dummy_kwargs,
    ):
        super().__init__(shapes, cnn_keys, mlp_keys, mlp_layers)

        self._cnn_depth = cnn_depth

        self._res_layers = res_layers
        self._res_depth = res_depth
        self._res_norm = res_norm
        self._cnn_input_dim = cnn_input_dim

        self._deco_attmask = deco_attmask
        self._deco_attmaskwarmup = None if deco_attmaskwarmup == -1 else deco_attmaskwarmup

        self._training_step = 0
        self._current_attmask = None

        cnn_out_channels = {k: self._shapes[k][-1] for k in self.cnn_keys}
        hw = 64 // 2**(self._res_depth + 1)
        self.convin = nn.Sequential(nn.Linear(self._cnn_input_dim, hw * hw * (2**(self._res_depth - 1)) * self._cnn_depth),
                                                  Rearrange('b t (c h w)-> (b t) c h w',h=hw,w=hw))
        self._cnn_nn = nn.ModuleDict()
        self.ctx_shape = [hw*2**(i+1) for i in range(self._res_depth)]
        for i in range(self._res_depth):
            depth = depth // 2 if i else int((2**(self._res_depth - 1)) * self._cnn_depth)
            spitial_dim = self.ctx_shape[i]
            add_dim = hw * self._cnn_depth // (2**i) 
            self._cnn_nn.add_module(f"unpool{i}", nn.UpsamplingNearest2d(scale_factor=2))
            self._cnn_nn.add_module(f"res{i}", ResidualStack(depth, depth//2,
                                                             self._res_layers,
                                                             norm=self._res_norm, dec=True,
                                                             addin_dim=add_dim,
                                                             has_addin=(lambda x: x % 2 == 0) if spitial_dim < 32 else (lambda x: False),
                                                             cross_att=True,
                                                             mask=self._deco_attmask,
                                                             spatial_dim=(spitial_dim,spitial_dim)))
        self.convout = nn.ConvTranspose2d(depth//2, sum(cnn_out_channels.values()), 3, 2, 1, output_padding=1)
        self._cnn_out_ps = [[out_channel] for out_channel in cnn_out_channels.values()]


    def forward(self, features, shortcuts=None):
        outputs = {}
        if self.cnn_keys:
            outputs.update(self._cnn(features, shortcuts))
        if self.mlp_keys:
            outputs.update(self._mlp(features))
        return outputs

    # # TODO: clean up
    # def _cnn_old(self, features, shortcuts=None):
    #     if self.training:
    #         self._training_step += 1

    #     if self._deco_attmaskwarmup is not None:
    #         self._current_attmask = (1 - self._deco_attmask) * \
    #             (1 - min(1, self._training_step / self._deco_attmaskwarmup)) + self._deco_attmask
    #         if self._training_step % 100 == 0:
    #             print(f"Current attention mask: {self._current_attmask} {self._training_step}")
    #     else:
    #         self._current_attmask = None

    #     seq_len = features.shape[1]
    #     channels = {k: self._shapes[k][-1] for k in self.cnn_keys}

    #     L = self._res_depth
    #     hw = 64 // 2**(self._res_depth + 1)
    #     x = self.get("convin", nn.Linear, features.shape[-1], hw * hw * (2**(L - 1)) * self._cnn_depth)(features)
    #     # x = torch.reshape(x, [-1, (2**(L - 1)) * self._cnn_depth, hw, hw]).to(memory_format=torch.channels_last)
    #     x = rearrange(x, 'b t (c h w) -> (b t) c h w',h=hw,w=hw)
    #     for i in range(L):
    #         x = self.get(f"unpool{i}", nn.UpsamplingNearest2d, scale_factor=2)(x)
    #         depth = x.shape[1]

    #         ctx = shortcuts[x.shape[2]]
    #         addin = ctx
    #         # addin = rearrange(ctx, '(b t) c h w -> b t c h w',b=features.shape[0])
    #         # addin = ctx.reshape(features.shape[0], -1, *ctx.shape[-3:])  # [B, K, C, H, W]
    #         addin = repeat(addin, 'b c h w -> (b repeat) c h w', repeat=x.shape[0] // addin.shape[0]) # repeat_interleave
    #         # addin = addin.repeat_interleave(x.shape[0] // addin.shape[0], dim=0)  # [BT, K, C, H, W]
    #         # addin = addin.reshape(-1, *addin.shape[-3:])  # [BTK, C, H, W]

    #         x = self.get(f"res{i}", ResidualStack, x.shape[1], depth // 2,
    #                      self._res_layers, norm=self._res_norm, dec=True,
    #                      addin_dim=addin.shape[1],
    #                      has_addin=(lambda x: x % 2 == 0) if ctx.shape[-1] < 32 else (lambda x: False),
    #                      cross_att=True,
    #                      mask=self._deco_attmask,
    #                      spatial_dim=x.shape[-2:],
    #                      )(x, addin, attmask=self._current_attmask)

    #     depth = sum(channels.values())
    #     x = self.get(f"convout", nn.ConvTranspose2d, x.shape[1], depth, 3, 2, 1, output_padding=1)(x)

    #     x = x.reshape(features.shape[:-1] + x.shape[1:])
    #     means = torch.split(x, list(channels.values()), 2)
    #     dists = {
    #         key: core.dists.Independent(core.dists.MSE(mean), 3)
    #         for (key, shape), mean in zip(channels.items(), means)
    #     }
    #     return dists
    
    def _cnn(self, features, shortcuts=None):
        if self.training:
            self._training_step += 1

        if self._deco_attmaskwarmup is not None:
            self._current_attmask = (1 - self._deco_attmask) * \
                (1 - min(1, self._training_step / self._deco_attmaskwarmup)) + self._deco_attmask
            if self._training_step % 100 == 0:
                print(f"Current attention mask: {self._current_attmask} {self._training_step}")
        else:
            self._current_attmask = None

        x = self.convin(features).to(memory_format=torch.channels_last)
        for i in range(self._res_depth):
            ctx = shortcuts[self.ctx_shape[i]]
            addin = rearrange(ctx, 'b t c h w -> (b t) c h w')
            addin = repeat(addin, 'b c h w -> (b repeat) c h w', repeat=x.shape[0] // addin.shape[0])
            x = self._cnn_nn[f"unpool{i}"](x)
            x = self._cnn_nn[f"res{i}"](x, addin, attmask=self._current_attmask)
        x = self.convout(x)
        x = rearrange(x,'(b t) c h w -> b t c h w',b=features.shape[0])
        means = unpack(x, self._cnn_out_ps, 'b t * h w ')
        dists = {
            key: core.dists.Independent(core.dists.MSE(mean), 3)
            for key, mean in zip(self.cnn_keys, means)
        }
        return dists
            # addin = ctx
