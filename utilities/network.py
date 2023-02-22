import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def swish(x):
    x = x * torch.sigmoid(x)
    return x


def group_norm(in_channels, num_groups=32):
    assert in_channels % num_groups == 0, \
           f"in_channels {in_channels} must be divisible by num_groups {num_groups}"
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def time_embed(t, embed_dim):
    half_dim = embed_dim // 2
    exp_factor = np.log(10000) / (half_dim - 1)
    embed = torch.exp(torch.arange(half_dim, dtype=torch.float32).mul(-exp_factor))
    embed = embed.to(t.device)  # Move embeddings to GPU (if possible)
    embed = t.to(torch.float32)[:, None] * embed[None, :]  # Pair-wise multiplication
    embed = torch.cat([torch.sin(embed), torch.cos(embed)], dim=1)
    if embed_dim % 2 == 1:  # Zero padding if embed_dim is odd
        embed = F.pad(embed, (0, 1, 0, 0))
    return embed
