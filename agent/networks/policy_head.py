import einops

# import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

import utils

# from agent.networks.utils.diffusion_policy import DiffusionPolicy
# from agent.networks.utils.vqbet.pretrain_vqvae import init_vqvae, pretrain_vqvae
from agent.networks.mlp import MLP

######################################### Deterministic Head #########################################


class DeterministicHead(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        action_squash=True,
        loss_coef=1.0,
    ):
        super().__init__()
        self.loss_coef = loss_coef

        sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1])]

        if action_squash:
            layers += [nn.Tanh()]

        self.net = nn.Sequential(*layers)

    def forward(self, x, stddev=None, **kwargs):
        mu = self.net(x)
        std = stddev if stddev is not None else 0.1
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist

    def loss_fn(self, dist, target, reduction="mean", **kwargs):
        log_probs = dist.log_prob(target)
        loss = -log_probs

        if reduction == "mean":
            loss = loss.mean() * self.loss_coef
        elif reduction == "none":
            loss = loss * self.loss_coef
        elif reduction == "sum":
            loss = loss.sum() * self.loss_coef
        else:
            raise NotImplementedError

        return {
            "actor_loss": loss,
        }
