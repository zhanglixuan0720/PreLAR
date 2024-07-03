import torch
import torch.distributions as tdist
from ..core import dists


class RandomAgent:

    def __init__(self, act_space, logprob=False):
        self.act_space = act_space['action']
        self.logprob = logprob
        if hasattr(self.act_space, 'n'):
            self._dist = dists.OneHotDist(torch.zeros(self.act_space.n))
        else:
            dist = tdist.Uniform(torch.tensor(self.act_space.low), torch.tensor(self.act_space.high))
            self._dist = tdist.Independent(dist, 1)

    def __call__(self, obs, state=None, mode=None):
        action = self._dist.sample((len(obs['is_first']),))
        output = {'action': action}
        if self.logprob:
            output['logprob'] = self._dist.log_prob(action)
        return output, None
