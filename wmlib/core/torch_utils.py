import itertools

import torch
import torch.nn as nn
import torch.optim as optim

from .. import DEBUG_METRICS


class Module(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

    def get(self, name, ctor, *args, **kwargs):
        # Create or get layer by name to avoid mentioning it in the constructor.
        if not hasattr(self, '_modules'):
            raise Exception("Pytorch Module already has this")
            # self._modules = {}
        if name not in self._modules:
            self._modules[name] = ctor(*args, **kwargs)

            # * To match tensorflow initialization
            # * For Dense/Linear: kernel_initializer='glorot_uniform', bias_initializer='zeros'
            if ctor not in [nn.LayerNorm] and hasattr(self._modules[name], "weight"):
                torch.nn.init.xavier_uniform_(self._modules[name].weight)
                if self._modules[name].bias is not None and self._modules[name].bias.data is not None:
                    torch.nn.init.zeros_(self._modules[name].bias)

            if ctor in [nn.Conv2d, nn.ConvTranspose2d]:
                print("setting memory format to channels last")
                self._modules[name] = self._modules[name].to(memory_format=torch.channels_last)  # FIXME testing

        return self._modules[name]


class EmptyOptimizer():

    def backward(self, loss, retain_graph=False):
        pass

    @property
    def opt(self):
        return None

    @property
    def scaler(self):
        return None

    def step(self, loss, external_scaler=None):
        return {}


class Optimizer():

    def __init__(
            self, name, modules, lr, eps=1e-4, clip=None, wd=None, opt='adam', enable_fp16=False):
        assert not wd or 0 <= wd < 1
        assert not clip or 1 <= clip
        if isinstance(modules, list):  # then is a list a parameters
            modules = itertools.chain(*modules)
        self._name = name
        self._clip = clip
        self._wd = wd
        self._enable_fp16 = enable_fp16
        # self._wd_pattern = wd_pattern # FIXME IGNORE FOR NOW PATTERNS, they are applied to all
        self._opt = {
            'adam': lambda p: optim.Adam(p, lr, eps=eps, weight_decay=wd),
            'adamw': lambda p: optim.AdamW(p, lr, eps=eps, weight_decay=wd),
            'adamax': lambda p: optim.Adamax(p, lr, eps=eps, weight_decay=wd),
            'sgd': lambda p: optim.SGD(p, lr, weight_decay=wd),
            'momentum': lambda p: optim.SGD(p, lr, momentum=0.9, weight_decay=wd),
            # 'adam_tf': lambda p: Adam_tf(p, lr, eps=eps, weight_decay=wd),
            # 'adamw_tf': lambda p: AdamW_tf(p, lr, eps=eps, weight_decay=wd),
        }[opt](modules)
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._enable_fp16)

        print(f'Init optimizer - {self._name}')

    @property
    def opt(self):
        return self._opt

    @property
    def scaler(self):
        return self._scaler

    def backward(self, loss, retain_graph=False):
        self._scaler.scale(loss).backward(retain_graph=retain_graph)

    def step(self, loss, external_scaler=None):
        # NOTE: when one loss applies to multiple optimizers, they share the same scaler
        current_scaler = external_scaler or self._scaler

        metrics = {}
        metrics[f'{self._name}_loss'] = loss.item()

        if self._clip:
            # gets unscaled gradients
            current_scaler.unscale_(self._opt)

            # Assuming only 1 optimizer group of tensors
            norm = torch.nn.utils.clip_grad_norm_(self._opt.param_groups[0]['params'], self._clip,
                                                  error_if_nonfinite=False)
            metrics[f'{self._name}_grad_norm'] = norm.item()  # implementation not equal to tf

        current_scaler.step(self._opt)
        if external_scaler is None:
            # NOTE: scaler.update should only be called once, after all optimizers used this iteration have been stepped:
            self._scaler.update()

        # opt.zero_grad()  # set_to_none=True here can modestly improve performance

        if self._enable_fp16 and DEBUG_METRICS:
            metrics[f'{self._name}_loss_scale'] = current_scaler.get_scale()  # incurs a CPU-GPU sync. TODO optimization

        return metrics


import warnings
from torch.optim.lr_scheduler import _LRScheduler


class ConstantLR(_LRScheduler):
    """Decays the learning rate of each parameter group by a small constant factor until the
    number of epoch reaches a pre-defined milestone: total_iters. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler decays the learning rate.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if epoch == 0
        >>> # lr = 0.025   if epoch == 1
        >>> # lr = 0.025   if epoch == 2
        >>> # lr = 0.025   if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = ConstantLR(self.opt, factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, factor=1.0 / 3, total_iters=5, last_epoch=-1, verbose=False):
        if factor > 1.0 or factor < 0:
            raise ValueError('Constant multiplicative factor expected to be between 0 and 1.')

        self.factor = factor
        self.total_iters = total_iters
        super(ConstantLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            self.normal_lr = [group['lr'] for group in self.optimizer.param_groups]
            return [group['lr'] * self.factor for group in self.optimizer.param_groups]

        if (self.last_epoch > self.total_iters or
                (self.last_epoch != self.total_iters)):
            return [group['lr'] for group in self.optimizer.param_groups]

        if (self.last_epoch == self.total_iters):
            # NOTE: Fix to error when self.factor == 0
            # return [group['lr'] * (1.0 / self.factor) for group in self.optimizer.param_groups]
            return self.normal_lr

    def _get_closed_form_lr(self):
        return [base_lr * (self.factor + (self.last_epoch >= self.total_iters) * (1 - self.factor))
                for base_lr in self.base_lrs]


def dict_to_device(d, device):
    return {k: v.to(device) for k, v in d.items()}


def dict_detach(d):
    return {k: v.detach() for k, v in d.items()}


def dict_apply(d, fn):
    return {k: fn(v) for k, v in d.items()}

def dict_cat(d1,d2, dim=0):
    return {k: torch.cat([d1[k], d2[k]], dim=dim) for k in d1.keys()}

def dict_chunk(d, chunk_num, dim):
    chunk_dicts = [dict() for i in range(chunk_num)]
    for k in d.keys():
        chunk_v = torch.chunk(d[k], chunk_num, dim=dim)
        for i in range(chunk_num):
            chunk_dicts[i][k] = chunk_v[i]
    return chunk_dicts

