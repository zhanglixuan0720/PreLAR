import torch
import torch.nn as nn
import kornia
import torchvision.transforms as T
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np

from .base import BaseEncoder
from ..modules import get_act_module, ResidualStack



class DecoupledResNetEncoder(BaseEncoder):
    '''
    Decoupled ResNet Encoder
    @Function: disentangle the visual context and the embedding
    '''
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
        deco_res_layers=2,
        deco_cnn_depth=48,
        deco_cond_choice='trand',
        deco_aug='none',
        **dummy_kwargs,
    ):
        super().__init__(shapes, cnn_keys, mlp_keys, mlp_layers)
        # self._act = get_act(act)
        self._act_module = get_act_module(act)
        self._cnn_depth = cnn_depth

        self._res_layers = res_layers
        self._res_depth = res_depth
        self._res_norm = res_norm

        self._deco_res_layers = deco_res_layers
        self._deco_cnn_depth = deco_cnn_depth
        self._deco_cond_choice = deco_cond_choice
        self._deco_aug = deco_aug

        self._cnn_net = nn.Sequential()
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

        self._deco_net = nn.ModuleDict()
        self._deco_net.add_module('cond_aug',get_augmentation(self._deco_aug, (h,w)))
        self._deco_net.add_module('convin', nn.Sequential(nn.Conv2d(c, self._deco_cnn_depth, 3, 2, 1),
                                                           self._act_module()))
        for i in range(self._res_depth):
            depth = 2 ** i * self._deco_cnn_depth
            input_channels = depth // 2 if i else self._deco_cnn_depth
            self._deco_net.add_module(f"res{i}", ResidualStack(input_channels, depth,
                                                               self._deco_res_layers,
                                                               norm=self._res_norm))
            self._deco_net.add_module(f"pool{i}", nn.AvgPool2d(2, 2))

    def forward(self, data, is_eval=False):
        key, shape = list(self.shapes.items())[0]
        batch_dims = data[key].shape[:-len(shape)]
        data = {
            k: torch.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
            for k, v in data.items()
        }

        output, shortcut = self._cnn({k: data[k] for k in self.cnn_keys}, batch_dims, is_eval)

        if is_eval:
            return output.reshape(batch_dims + output.shape[1:])
        else:
            return {
                'embed': output.reshape(batch_dims + output.shape[1:]),
                'shortcut': shortcut,
            }

    def _cnn(self, data, batch_dims=None, is_eval=False):
        x = torch.cat(list(data.values()), -1)
        x = x.to(memory_format=torch.channels_last)
        embed = self._cnn_net(x)
        shortcuts = {}
        if not is_eval:
            b, t = batch_dims
            with torch.no_grad():
                ctx = self.get_context(rearrange(x, '(b t) c h w -> b t c h w', b=b))#(x.reshape(batch_dims + x.shape[1:]))  # [B, T, C, H, W] => [B, C, H, W]
                ctx = self._deco_net['cond_aug'](ctx)  # [B, C, H, W]
            ctx = rearrange(ctx, 'b t c h w -> (b t) c h w')
            ctx = self._deco_net['convin'](ctx)
            for i in range(self._res_depth):
                ctx = self._deco_net[f"res{i}"](ctx)
                shortcuts[ctx.shape[2]] = rearrange(ctx, '(b t) c h w -> b t c h w', b=b)
                ctx = self._deco_net[f"pool{i}"](ctx)
        return embed, shortcuts
        
        
    # TODO: clean up or rename t0 tlast trand
    def get_context(self, frames):
        """
        frames: [B, T, C, H, W]
        """
        with torch.no_grad():
            if self._deco_cond_choice == 't0':
                # * initial frame
                context = frames[:, 0].unsqueeze(1)   # [B, C, H, W]
            elif self._deco_cond_choice == 'tlast':
                # * last frame
                context = frames[:, -1].unsqueeze(1)  # [B, C, H, W]
            elif self._deco_cond_choice == 'trand':
                # * timestep randomization
                idx = torch.from_numpy(np.random.choice(frames.shape[1], frames.shape[0])).to(frames.device)
                idx = idx.reshape(-1, 1, 1, 1, 1).repeat(1, 1, *frames.shape[-3:])  # [B, 1, C, H, W]
                context = frames.gather(1, idx) # .squeeze(1)  # [B, 1, C, H, W]# [B, C, H, W]
            elif self._deco_cond_choice == 'diff':
                idx = np.arange(frames.shape[1])
                idx[1:]-=1
                idx = torch.from_numpy(idx).to(frames.device)
                idx = idx.reshape(1,frames.shape[1], 1, 1, 1).repeat(frames.shape[0], 1, *frames.shape[-3:])
                context = frames.gather(1, idx)
            elif self._deco_cond_choice == 'self':
                idx = torch.from_numpy(np.arange(frames.shape[1])).to(frames.device)
                idx = idx.reshape(1,frames.shape[1], 1, 1, 1).repeat(frames.shape[0], 1, *frames.shape[-3:])
                context = frames.gather(1, idx)
            else:
                raise NotImplementedError
        return context


def get_augmentation(aug_type, shape):
    if aug_type == 'none':
        return nn.Identity()
    elif aug_type == 'shift':
        return nn.Sequential(
            nn.ReplicationPad2d(padding=8),
            kornia.augmentation.RandomCrop(shape[-2:])
        )
    elif aug_type == 'shift4':
        return nn.Sequential(
            nn.ReplicationPad2d(padding=4),
            kornia.augmentation.RandomCrop(shape[-2:])
        )
    elif aug_type == 'flip':
        return T.RandomHorizontalFlip(p=0.5)
    elif aug_type == 'scale':
        return T.RandomResizedCrop(
            size=shape[-2:], scale=[0.666667, 1.0], ratio=(0.75, 1.333333))
    elif aug_type == 'erasing':
        return kornia.augmentation.RandomErasing()
    else:
        raise NotImplementedError
