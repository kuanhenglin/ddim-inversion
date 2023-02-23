import numpy as np
import torch
import torch.nn.functional as F


def beta_schedule(schedule, beta_start, beta_end, num_t):

    def sigmoid(x):
        x = (-x).exp().add(1.0).reciprocal()  # 1 / (e^-x + 1)
        return x

    dtype = torch.float32
    if schedule == "quad":
        betas = torch.linspace(np.sqrt(beta_start), np.sqrt(beta_end), num_t, dtype=dtype)
        betas.square_()
    elif schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_t, dtype=dtype)
    elif schedule == "constant":
        betas = torch.ones(num_t, dtype=dtype).mul(beta_end)
    elif schedule == "jsd":  # 1 / T, 1 / (T - 1), 1 / (T - 2), ..., 1
        betas = torch.linspace(num_t, 1, num_t, dtype=torch.float32).reciprocal()
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_t)  # Arbitrary values
        betas = sigmoid(betas).mul(beta_end - beta_start).add(beta_start)
    else:
        raise NotImplementedError(schedule)

    assert betas.shape == (num_t,)  # Sanity check that their shapes match
    return betas


def alpha(b, t):
    """
    Insert 0.0 to left of betas and select index with t + 1. We do this (instead of no insert and
    select with t) because t = -1 is possible, and we want a_t = 0.0 when that happens.
    """
    t = t.to(torch.int32)  # t needs to be int to be treated as indices
    b = F.pad(b, (1, 0), value=0.0)
    a_t = (1 - b).cumprod(dim=0).index_select(dim=0, index=t + 1)[:, None, None, None]
    return a_t


def criterion(output, target, name="l2"):
    if name == "l1":
        loss = (target - output).abs().mean()
    elif name == "l2":
        loss = (target - output).square().mean()
    elif name == "linf":
        loss = (target - output).max(dim=(1, 2, 3)).mean()
    else:
        raise NotImplementedError(name)
    return loss
