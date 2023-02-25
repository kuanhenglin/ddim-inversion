import torch
from torch import optim
from tqdm import tqdm

import utilities.runner as rutils
import utilities.utilities as utils


def gradient_inversion(diffusion, target, z=None, lr=0.01, num_i=100, num_t_steps=10,
                       criterion="l1", show_progress=False):
    device = diffusion.device

    z = utils.get_default(z, default=torch.randn(*target.shape, device=device))
    z.requires_grad_()

    optimizer = optim.Adam([z], lr=lr, betas=(0.9, 0.999), eps=1e-8)  # Use Adam by default
    target = target.to(device)

    progress = range(num_i)
    if show_progress:
        progress = tqdm(progress)
    for i in progress:
        y = diffusion.sample(x=z, sequence=False, ema=True, num_t_steps=num_t_steps)[0]
        loss = rutils.criterion(y, target, name=criterion)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if show_progress:
            progress.set_description(f"Loss: {loss.detach().cpu().numpy():.7}")
            progress.refresh()

    return z
