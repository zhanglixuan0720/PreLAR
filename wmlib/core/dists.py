import torch
import torch.nn.functional as F
import torch.distributions as tdist


class OneHotDist(tdist.OneHotCategorical):

    def __init__(self, logits=None, probs=None, dtype=None, validate_args=False):
        self._sample_dtype = dtype or torch.float32  # FIXME currently ignored
        super().__init__(probs=probs, logits=logits, validate_args=False)  # FIXME: validate_args=None? => Error

        # FIXME event_shape -1 for now, because I think could be empty
        # if so, tf uses logits or probs shape[-1]
        self._mode = F.one_hot(torch.argmax(self.logits, -1), self.event_shape[-1]).float()  # FIXME dtype

    @property
    def mode(self):
        return self._mode

    def sample(self, sample_shape=(), seed=None):
        # Straight through biased gradient estimator.
        # FIXME seed is not possible here

        sample = super().sample(sample_shape)  # .type(self._sample_dtype) # FIXME
        probs = super().probs
        while len(probs.shape) < len(sample.shape):  # adds dims on 0
            probs = probs[None]

        sample += (probs - probs.detach())  # .type(self._sample_dtype)
        return sample

    # custom log_prob more stable
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        value, logits = torch.broadcast_tensors(value, self.logits)
        indices = value.max(-1)[1]  # FIXME can we support soft label?
        ret = -F.cross_entropy(
            logits.reshape(-1, *self.event_shape),
            indices.reshape(-1).detach(),
            reduction='none'
        )
        return torch.reshape(ret, logits.shape[:-1])


# From https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py
import math
from numbers import Number

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1, ).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b -
                                 self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        # NOTE: additional to github.com/toshas/torch_truncnorm
        self._mode = torch.clamp(torch.zeros_like(self.a), self.a, self.b)
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def mode(self):
        return self._mode

    @property
    def variance(self):
        return self._variance

    # @property #In pytorch is a function
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        # icdf is numerically unstable; as a consequence, so is rsample.
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, scalar_a, scalar_b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, scalar_a, scalar_b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._mode = torch.clamp(self.loc, scalar_a, scalar_b)  # NOTE: additional to github.com/toshas/torch_truncnorm
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale


#
class TruncNormalDist(TruncatedNormal):

    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale, low, high)
        self._clip = clip
        self._mult = mult

        self.low = low
        self.high = high

    def sample(self, *args, **kwargs):
        event = super().rsample(*args, **kwargs)
        if self._clip:
            clipped = torch.clamp(
                event, self.low + self._clip, self.high - self._clip
            )
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


# Pytorch distributions extending with mode
class Independent(tdist.Independent):

    @property
    def mode(self):
        return self.base_dist.mode


class Normal(tdist.Normal):
    # FIXME support log_prob_without_constant
    @property
    def mode(self):
        return self.mean

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)


class MSE(tdist.Normal):

    def __init__(self, loc, validate_args=None):
        super(MSE, self).__init__(loc, 1.0, validate_args=validate_args)

    @property
    def mode(self):
        return self.mean

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # NOTE: dropped the constant term
        return -((value - self.loc) ** 2) / 2


class Bernoulli(tdist.Bernoulli):

    @property
    def mode(self):
        return torch.round(self.probs)  # >0.5
