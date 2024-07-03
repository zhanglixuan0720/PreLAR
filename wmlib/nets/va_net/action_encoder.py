from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributions as dists
from ..modules import get_act_module
from einops.layers.torch import Rearrange
from einops import rearrange
from einops.layers.torch import Rearrange
from ...core import dists

class ActionEncoder(nn.Module):
    def __init__(self,action_dim,hidden_dim,act,deter=64,discrete=False,stoch=64,type_='deter', std_act='sigmoid2',twl=1,**dummy_kwargs):
        super().__init__()
        self._action_dim = action_dim
        self._hidden_dim = hidden_dim
        self.twl = int(twl)
        # self.twl = int(2)

        # self._embed_dim = embed_dim
        self._act_module = get_act_module(act)
        # self._dist_embed = dist_embed
        self._deter = deter
        self._discrete = discrete
        self._stoch = stoch
        self._std_act = std_act
        self.type = type_ # deter, stoch, mix
        if self.type == 'deter':
            self.embed_dim = self._deter
        elif self.type == 'stoch':
            self.embed_dim = self._stoch * self._discrete if self._discrete else self._stoch * 2
        elif self.type == 'mix':
            self.embed_dim = self._deter + (self._stoch * self._discrete if self._discrete else self._stoch * 2)
        else:
            raise NotImplementedError
        # if self._dist_embed:
            # self._embed_dim = self._stoch * self._discrete if self._discrete else self._stoch * 2
        # input_action_dim = self._action_dim if self.twl<=1 else self._action_dim * self.twl
        if self.twl <= 1:
            self._encoder = nn.Sequential(
                nn.Linear(self._action_dim,self._hidden_dim),
                self._act_module(),
                nn.Linear(self._hidden_dim,self.embed_dim)
            )
        else:
            self._encoder = nn.Sequential(
                Rearrange('... t d -> ... d t'),
                nn.Conv1d(self._action_dim,self._hidden_dim,self.twl,padding=self.twl-1),
                Rearrange('... d t -> ... t d'),
                self._act_module(),
                nn.Linear(self._hidden_dim,self.embed_dim)
            )
            self.action_buffer = ActionBuffer(self.twl)
        # if self._discrete:
        #     self._encoder.add_module('encoder_rerange',Rearrange('... (s d) -> ... s d', s=self._stoch, d=self._discrete))
        
        self._std_fn = {
                "softplus": lambda std: F.softplus(std),
                "sigmoid": lambda std: torch.sigmoid(std),
                "sigmoid2": lambda std: 2 * torch.sigmoid(std / 2),
            }[self._std_act]
        
    def forward(self,action,sample=True):
        x =  self._encoder(action)
        if self.twl > 1:
            x = x[:,:-self.twl+1]
            # x = x
        action_code = {}
        if self.type == 'deter':
            return {'deter':x}
        elif self.type == 'stoch':
            x_stoch = x
        elif self.type == 'mix':
            x_deter = x[..., :self._deter]
            x_stoch = x[..., self._deter:]
            action_code.update({'deter':x_deter}) 
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
        # mean = mean.float()
        # std = std.float()
            std = self._std_fn(std)
            dist = dists.Independent(dists.Normal(mean, std), 1)
        return dist

class ActionBuffer:
    def __init__(self,twl) -> None:
        self.actions = deque(maxlen=twl)

    def append(self,action):
        self.actions.append(action)
    
    def get_actions(self):
        return torch.stack(list(self.actions),dim=1)

    def clear(self):
        self.actions.clear()