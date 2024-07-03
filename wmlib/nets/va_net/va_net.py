import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdists
import numpy as np
from einops import rearrange

from ..encoder import VAResNetEncoder
from ..modules import get_act_module
from .. import core
from ...core import dists


class VANet(nn.Module):
    # def __init__(self,shape, config) -> None:
    def __init__(
            self,
            shape, 
            cnn_keys=r".*",
            mlp_keys=r".*",
            act="elu",
            cnn_depth=48,
            mlp_layers=[400, 400, 400, 400],
            res_layers=2,
            res_depth=3,
            res_norm='none',
            hidden_dim=1024,
            deter=0,
            stoch = 64,
            discrete=0,
            std_act='sigmoid2',
            va_method='concate', # concate | diff | flow | cat-diff 
            type_='stoch', # deter | stoch | mix
            **dummy_kwargs ) -> None:
        super().__init__()
        h,w,_ = shape['image']
        self._encode_dim = cnn_depth * 2** (res_depth - 1) * h//2**(res_depth+1) * w//2**(res_depth+1)
        self._hidden_dim = hidden_dim
        self._act_module = get_act_module(act)
        self._stoch = stoch
        self._discrete = discrete
        self._deter = deter
        self._std_act = std_act
        self._va_method = va_method
        self.type = type_
        self._patch_size = (h//16,w//16)
        self._patch_split = (16,16)
        self._patch_num = self._patch_split[0] * self._patch_split[1]

        action_dim = self._stoch * self._discrete if self._discrete else self._stoch * 2
        action_dim = self._deter if self.type =='deter' else action_dim
        action_dim = self._deter + action_dim if self.type == 'mix' else action_dim                                     
        self.va_encoder = VAResNetEncoder(shapes=shape,cnn_keys=cnn_keys,mlp_keys=mlp_keys,
                                        act=act,cnn_depth=cnn_depth,mlp_layers=mlp_layers,
                                        res_layers=res_layers,res_depth=res_depth,
                                        res_norm=res_norm,va_method=self._va_method) #nets.VAEncoder(shapes, **config.va_encoder)
        self.action_in = nn.Sequential(nn.Linear(self._encode_dim,self._hidden_dim),
                                       self._act_module(),
                                       nn.Linear(self._hidden_dim,action_dim))
        self._std_fn = {
                "softplus": lambda std: F.softplus(std),
                "sigmoid": lambda std: torch.sigmoid(std),
                "sigmoid2": lambda std: 2 * torch.sigmoid(std / 2),
            }[self._std_act]
    
    def forward(self,img0,img1,sample=True):
        if self._va_method in ['concate', 'mask']:
            x = torch.cat([img0,img1],dim=2) # b t c h w
        elif self._va_method == 'diff':
            x = img1 - img0
        elif self._va_method == 'flow':
            x = img1
        elif self._va_method == 'catdiff':
            x = torch.cat([img0,img1,img1-img0],dim=2)
        x = self.va_encoder({'image':x})
        x = self.action_in(x)
        action_code = {}
        if self.type == 'deter':
            return {'deter':x}
        elif self.type == 'stoch':
            x_stoch = x
        elif self.type == 'mix':
            x_deter = x[..., :self._deter]
            x_stoch = x[..., self._deter:]
            action_code.update({'deter':x_deter})
        else:
            raise NotImplementedError
        
        if self._discrete:
            x_stoch = rearrange(x_stoch,'... (s d) -> ... s d', s=self._stoch, d=self._discrete)
            dist  = self.get_dist({'logit':x_stoch})
            stoch = dist.sample() if sample else dist.mode
            stoch = rearrange(stoch,'... s d -> ... (s d)')
            action_code.update({'stoch':stoch,'logit':x_stoch})
        else:
            mean, std = torch.chunk(x_stoch, 2, dim=-1)
            dist = self.get_dist({'mean':mean,'std':std})
            stoch = dist.sample() if sample else dist.mode
            action_code.update({'stoch':stoch,'mean':mean,'std':std})
        return action_code
    
    def get_dist(self, state):
        """
        gets the stochastic state distribution
        """
        if self._discrete:
            logit = state["logit"]
            logit = logit.float()
            dist = dists.Independent(dists.OneHotDist(logit), 1)
        else:
            mean, std = state['mean'], state['std']
            std = self._std_fn(std)
            dist = dists.Independent(dists.Normal(mean, std), 1)
        return dist
    
    def append_action(self, action, ahead = False):
        new_action = torch.zeros_like(action[:,0,:],device=action.device).unsqueeze(dim=1)
        if ahead:
            return torch.cat([new_action, action], dim=1)
        return torch.cat([action, new_action], dim=1)
    
    def kl_loss(self, post, prior=None, forward: bool=False, balance: float=0.5, free: float=0.0, free_avg: bool=True):
        """
        computes the kl loss
        """
        if self.type == 'deter':
            value = torch.tensor(0,dtype=float,device=post['deter'].device)
            return value, value
        if self._discrete and prior is None:
            value = torch.log(torch.tensor(self._discrete,device=post['logit'].device))
            return value, value
        kld = tdists.kl_divergence
        sg = core.dict_detach
        _device = post['stoch'].device
        if prior is None:
            prior = {'mean':torch.zeros_like(post['mean'],device=_device),'std':torch.ones_like(post['std'],device=_device)}
        
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
    
    
    
