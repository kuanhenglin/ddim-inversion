import numpy as np
import torch
from tqdm import tqdm


def slerp(z_1, z_2, alpha):
    theta = torch.acos((z_1 * z_2).sum() / (torch.norm(z_1) * torch.norm(z_2)))
    return (((1.0 - alpha) * theta).sin() / theta.sin()) * z_1 + \
        ((alpha * theta).sin() / theta.sin()) * z_2


def interpolation(z_1, z_2, diffusion, num_t_steps=10, num_alphas=10, show_progress=False):
    x_mixes = []
    alphas = np.linspace(0.0, 1.0, num=num_alphas)
    progress = tqdm(alphas) if show_progress else alphas
    for alpha in progress:
        z_mix = slerp(z_1, z_2, alpha=alpha)
        x_mix = diffusion.sample(x=z_mix, sequence=False, num_t_steps=num_t_steps)[0]
        x_mixes.append(x_mix.detach().cpu())
    return x_mixes


def proj_interpolation(z_1, z_2, diffusion, proj_fn_1=None, proj_fn_2=None,
                       num_t_steps=10, num_alphas=100, show_progress=False):
    if proj_fn_1 is not None:
        z_1 = proj_fn_1(z_1, sequence=False)
    if proj_fn_2 is not None:
        z_2 = proj_fn_2(z_2, sequence=False)
    x_mixes = interpolation(z_1, z_2, diffusion, num_t_steps=num_t_steps, num_alphas=num_alphas,
                            show_progress=show_progress)
    return x_mixes
