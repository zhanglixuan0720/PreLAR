import torch
import torch.nn as nn
import kornia
import torchvision.transforms as T

from .base import BaseEncoder
from ..modules import *


class ContextualizedResNetEncoder(BaseEncoder):

    # TODO: remame args
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
        ctx_res_layers=2,
        ctx_cnn_depth=48,
        ctx_cond_choice='trand',
        ctx_aug='none',
        **dummy_kwargs,
    ):
        super().__init__(shapes, cnn_keys, mlp_keys, mlp_layers)
        self._act = get_act(act)
        self._cnn_depth = cnn_depth

        self._res_layers = res_layers
        self._res_depth = res_depth
        self._res_norm = res_norm

        self._ctx_res_layers = ctx_res_layers
        self._ctx_cnn_depth = ctx_cnn_depth
        self._ctx_cond_choice = ctx_cond_choice
        self._ctx_aug = ctx_aug

    def __call__(self, data, eval=False):
        key, shape = list(self.shapes.items())[0]
        batch_dims = data[key].shape[:-len(shape)]
        data = {
            k: torch.reshape(v, (-1,) + tuple(v.shape)[len(batch_dims):])
            for k, v in data.items()
        }

        output, shortcut = self._cnn({k: data[k] for k in self.cnn_keys}, batch_dims, eval)
        # TODO: self._mlp
        if eval:
            return output.reshape(batch_dims + output.shape[1:])
        else:
            return {
                'embed': output.reshape(batch_dims + output.shape[1:]),
                'shortcut': shortcut,
            }

    def _cnn(self, data, batch_dims=None, eval=False):
        x = torch.cat(list(data.values()), -1)
        x = x.to(memory_format=torch.channels_last)

        shortcuts = {}
        if not eval:
            with torch.no_grad():
                ctx = self.get_context(x.reshape(batch_dims + x.shape[1:]))  # [B, T, C, H, W] => [B, C, H, W]

                module_name = f"cond_aug"
                if module_name not in self._modules:
                    self._modules[module_name] = get_augmentation(self._ctx_aug, ctx.shape)
                ctx = self.cond_aug(ctx)

        x = self.get(f"convin", nn.Conv2d, x.shape[1], self._cnn_depth, 3, 2, 1)(x)
        x = self._act(x)

        if not eval:
            ctx = self.get(f"cond_convin", nn.Conv2d, ctx.shape[1], self._ctx_cnn_depth, 3, 2, 1)(ctx)
            ctx = self._act(ctx)

        L = self._res_depth
        for i in range(L):
            depth = 2 ** i * self._cnn_depth
            x = self.get(f"res{i}", ResidualStack, x.shape[1], depth,
                         self._res_layers,
                         norm=self._res_norm,
                         spatial_dim=x.shape[-2:],
                         )(x)
            x = self.get(f"pool{i}", nn.AvgPool2d, 2, 2)(x)

            if not eval:
                ctx_depth = 2 ** i * self._ctx_cnn_depth
                ctx = self.get(f"cond_res{i}", ResidualStack, ctx.shape[1], ctx_depth,
                               self._ctx_res_layers,
                               norm=self._res_norm,
                               spatial_dim=ctx.shape[-2:],
                               )(ctx)
                shortcuts[ctx.shape[2]] = ctx  # [B, C, H, W]
                ctx = self.get(f"cond_pool{i}", nn.AvgPool2d, 2, 2)(ctx)

        return x.reshape(tuple(x.shape[:-3]) + (-1,)), shortcuts

    # TODO: clean up or rename t0 tlast trand
    def get_context(self, frames):
        """
        frames: [B, T, C, H, W]
        """
        with torch.no_grad():
            if self._ctx_cond_choice == 't0':
                # * initial frame
                context = frames[:, 0]  # [B, C, H, W]
            elif self._ctx_cond_choice == 'tlast':
                # * last frame
                context = frames[:, -1]  # [B, C, H, W]
            elif self._ctx_cond_choice == 'trand':
                # * timestep randomization
                idx = torch.from_numpy(np.random.choice(frames.shape[1], frames.shape[0])).to(frames.device)
                idx = idx.reshape(-1, 1, 1, 1, 1).repeat(1, 1, *frames.shape[-3:])  # [B, 1, C, H, W]
                context = frames.gather(1, idx).squeeze(1)  # [B, C, H, W]
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
