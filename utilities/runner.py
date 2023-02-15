import numpy as np
import torch


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
