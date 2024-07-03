import re

import torch
import torch.distributions as tdist
import torch.nn as nn
from . import torch_utils, dists


def sequence_scan(fn, state, *inputs, reverse=False):
    # state -> (batch, state related)
    # inputs[N] -> (sequence, batch, units)
    # FIXME try to optimize this awful
    # FIXME IDEA FOR JIT BUT REQUIRES REMOVING *inputs
    # static scan also works with tensors....

    indices = range(inputs[0].shape[0])
    select_index = lambda inputs, i: [input[i] for input in inputs]
    last = (state,)
    outs = []
    if reverse:
        indices = reversed(indices)
    for index in indices:
        last = fn(last[0], *select_index(inputs, index))
        last = last if isinstance(last, tuple) else (last,)
        outs.append(last)
    if reverse:
        outs = outs[::-1]
    items, keys =  len(outs[0]), outs[0][0].keys()
    output = [{k: torch.stack([x[i][k] for x in outs], 0) for k in keys} for i in range(items)]
    return output

    # outs1 = {k: torch.stack([x[k] for x in outs], 0) for k in outs[0].keys()}
    # FIXME this is awfulllllllllll
    # create right structure
    if isinstance(outs[0][0], dict):
        # create dictionary structure
        output = list({} for _ in range(len(outs[0])))  # create lists
        for o in outs:
            for i_d, dictionary in enumerate(o):
                if isinstance(dictionary, dict):  # FIXME
                    for key in dictionary.keys():
                        if key not in output[i_d]:
                            output[i_d][key] = [dictionary[key]]
                        else:
                            output[i_d][key].append(dictionary[key])
                elif isinstance(dictionary, torch.Tensor):
                    # here we append elements to list
                    if not isinstance(output[i_d], list):
                        output[i_d] = []
                    output[i_d].append(dictionary)
                else:
                    raise NotImplementedError(f"sequence scan - creating structure - type {type(dictionary)}")

        # torch stack all entries
        for i_o, dictionary in enumerate(output):
            if isinstance(dictionary, dict):  # FIXME
                for key in dictionary.keys():
                    dictionary[key] = torch.stack(dictionary[key], 0)
            elif isinstance(dictionary, list):
                output[i_o] = torch.stack(dictionary, 0)

            else:
                raise NotImplementedError(f"sequence scan - stacking - type {type(dictionary)}")

    elif isinstance(outs[0][0], torch.Tensor):
        # create tensor structure
        # no output tuple, flatten all in same stack
        # and is very specific to the problem
        # and is awful
        output = []
        for o in outs:
            for tensor in o:  # flatten tuple
                output.append(tensor)

        output = torch.stack(output, 0)

    else:
        raise NotImplementedError(f"sequence scan - Not implemented type {type(outs[0])}")
    return output


def schedule(string, step):
    try:
        return float(string)
    except ValueError:
        # step = tf.cast(step, tf.float32) #Fixme cast
        match = re.match(r'linear\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clamp(step / duration, 0, 1)
            return (1 - mix) * initial + mix * final
        match = re.match(r'warmup\((.+),(.+)\)', string)
        if match:
            warmup, value = [float(group) for group in match.groups()]
            scale = torch.clamp(step / warmup, 0, 1)
            return scale * value
        match = re.match(r'exp\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, halflife = [float(group) for group in match.groups()]
            return (initial - final) * 0.5 ** (step / halflife) + final
        match = re.match(r'horizon\((.+),(.+),(.+)\)', string)
        if match:
            initial, final, duration = [float(group) for group in match.groups()]
            mix = torch.clamp(step / duration, 0, 1)
            horizon = (1 - mix) * initial + mix * final
            return 1 - 1 / horizon
        raise NotImplementedError(string)


def lambda_return(
        reward, value, pcont, bootstrap, lambda_, axis):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    dims = list(range(len(reward.shape)))
    dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
    if axis != 0:
        reward = reward.permute(dims)
        value = value.permute(dims)
        pcont = pcont.permute(dims)
    if bootstrap is None:
        bootstrap = torch.zeros_like(value[-1])
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = [bootstrap]
    for i in range(len(inputs) - 1, -1, -1):
        returns.append(inputs[i] + pcont[i]*lambda_ * returns[-1])
    returns = torch.stack(returns[::-1][:-1], 0)
    # returns = sequence_scan(
    #     lambda agg, input, pcont: input + pcont * lambda_ * agg,
    #     bootstrap, inputs, pcont, reverse=True)
    if axis != 0:
        returns = returns.permute(dims)
    return returns


def action_noise(action, amount, action_space):
    if amount == 0:
        return action
    # amount = tf.cast(amount, action.dtype) # FIXME cast
    if hasattr(action_space, 'n'):
        probs = amount / action.shape[-1] + (1 - amount) * action
        return dists.OneHotDist(probs=probs).sample()
    else:
        return torch.clamp(tdist.Normal(action, amount).sample(), -1, 1)


# class StreamNorm(torch_utils.Module):
class StreamNorm(nn.Module):
    def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8, init=1.0):
        super(StreamNorm, self).__init__()

        # Momentum of 0 normalizes only based on the current batch.
        # Momentum of 1 disables normalization.
        self._shape = tuple(shape)
        self._momentum = momentum
        self._scale = scale
        self._eps = eps
        self._init = init
        self.register_buffer('mag', torch.ones(shape, dtype=torch.float64) * init)

    def forward(self, inputs):
        metrics = {}
        self.update(inputs)
        metrics['mean'] = inputs.mean().item()
        metrics['std'] = inputs.std().item()
        outputs = self.transform(inputs)
        metrics['normed_mean'] = outputs.mean().item()
        metrics['normed_std'] = outputs.std().item()
        return outputs, metrics

    def reset(self):
        self.mag.fill_(self._init)

    def update(self, inputs):
        batch = inputs.reshape((-1,) + self._shape)
        mag = torch.abs(batch).mean(0).to(dtype=torch.float64)
        self.mag.data = (self._momentum * self.mag + (1 - self._momentum) * mag).data

    def transform(self, inputs):
        values = inputs.reshape((-1,) + self._shape)
        values /= self.mag.to(inputs.dtype)[None] + self._eps
        values *= self._scale
        return values.reshape(inputs.shape)


class CarryOverState:

    def __init__(self, fn):
        self._fn = fn
        self._state = None

    def __call__(self, *args):
        self._state, out = self._fn(*args, self._state)
        return out
