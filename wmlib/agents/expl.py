import torch
import torch.nn as nn
import torch.distributions as tdist

from .. import core
from ..core import dists


class Random(nn.Module):

    def __init__(self, action_space):
        super(Random, self).__init__()
        self._action_space = action_space

    def actor(self, feat):
        shape = feat.shape[:-1] + [self._action_space.shape[-1]]
        if self.config.actor.dist == 'onehot':
            return dists.OneHotDist(torch.zeros(shape))
        else:
            dist = tdist.Uniform(-torch.ones(shape), torch.ones(shape))
            return tdist.Independent(dist, 1)

    def train(self, start, context, data):
        return None, {}


# TODO support plan2explore & model loss


class Plan2Explore(nn.Module):
    pass


class ModelLoss(nn.Module):
    pass


class VideoIntrBonus(nn.Module):
    def __init__(
        self,
        beta,
        k,
        intr_seq_length,
        feat_dim,
        queue_dim,
        queue_size,
        reward_norm,
        beta_type='abs',
    ) -> None:
        super().__init__()

        self.beta = beta
        self.k = k
        self.intr_seq_length = intr_seq_length
        self.tf_queue_step = 0
        self.tf_queue_size = queue_size
        shape = (feat_dim, queue_dim)
        self.random_projection_matrix = torch.nn.Parameter(
            torch.normal(mean=torch.zeros(shape), std=torch.ones(shape) / queue_dim),
            requires_grad=False,
        )
        self.register_buffer('queue', torch.zeros(queue_size, queue_dim))
        self.intr_rewnorm = core.StreamNorm(**reward_norm)

        self.beta_type = beta_type
        if self.beta_type == 'rel':
            self.plain_rewnorm = core.StreamNorm()

    def construct_queue(self, seq_feat):
        with torch.no_grad():
            seq_size = seq_feat.shape[0]
            self.queue.data[seq_size:] = self.queue.data[:-seq_size].clone()
            self.queue.data[:seq_size] = seq_feat.data

            self.tf_queue_step = self.tf_queue_step + seq_size
            self.tf_queue_step = min(self.tf_queue_step, self.tf_queue_size)
        return self.queue[: self.tf_queue_step]

    def compute_bonus(self, data, feat):
        with torch.no_grad():
            seq_feat = feat
            # NOTE: seq_feat [B, T, D], after unfold [B, T-S+1, D, S]
            seq_feat = seq_feat.unfold(dimension=1, size=self.intr_seq_length, step=1).mean(dim=-1)
            seq_feat = torch.matmul(seq_feat, self.random_projection_matrix)
            b, t, d = (seq_feat.shape[0], seq_feat.shape[1], seq_feat.shape[2])
            seq_feat = torch.reshape(seq_feat, (b * t, d))
            queue = self.construct_queue(seq_feat)
            dist = torch.norm(seq_feat[:, None, :] - queue[None, :, :], dim=-1)
            int_rew = -1.0 * torch.topk(
                -dist, k=min(self.k, queue.shape[0])
            ).values.mean(1)
            int_rew = int_rew.detach()
            int_rew, int_rew_mets = self.intr_rewnorm(int_rew)
            int_rew_mets = {f"intr_{k}": v for k, v in int_rew_mets.items()}
            int_rew = torch.reshape(int_rew, (b, t))

            plain_reward = data["reward"]

            if self.beta_type == 'abs':
                data["reward"] = data["reward"][:, :t] + self.beta * int_rew.detach()
            elif self.beta_type == 'rel':
                self.plain_rewnorm.update(data["reward"])
                beta = self.beta * self.plain_rewnorm.mag.item()
                data["reward"] = data["reward"][:, :t] + beta * int_rew.detach()
                int_rew_mets["abs_beta"] = beta
                int_rew_mets["plain_reward_mean"] = self.plain_rewnorm.mag.item()
            else:
                raise NotImplementedError

            if int_rew_mets['intr_mean'] < 1e-5:
                print("intr_rew too small:", int_rew_mets['intr_mean'])

            int_rew_mets["plain_reward_mean"] = plain_reward.mean().item()
            int_rew_mets["intr_mag"] = self.intr_rewnorm.mag.item()

        return data, t, int_rew_mets
