import torch
from torch import optim
from tqdm import tqdm

import utilities.runner as rutils
import utilities.utilities as utils


def gradient_inversion(z=None, target=None, diffusion=None, optimizer="adam", lr=0.01, num_i=300,
                       num_t_steps=10, criterion="l1", sequence=False, show_progress=False):
    device = diffusion.device

    z = utils.get_default(z, default=torch.randn(*target.shape, device=device))
    if target is None or diffusion is None:
        return z
    z.requires_grad_()

    maximize = criterion in ["psnr", "ssim"]
    if optimizer == "adam":  # Vanilla Adam
        optimizer = optim.Adam([z], lr=lr, betas=(0.9, 0.999), eps=1e-8, maximize=maximize)
        scheduler = None
    elif optimizer == "sgd":  # SGD w/ Nesterov momentum + piece-wise constant decay
        optimizer = optim.SGD([z], lr=lr, momentum=0.9, nesterov=True, maximize=maximize)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[num_i // 3, 2 * (num_i // 3)], gamma=0.1)
    else:
        raise NotImplementedError(optimizer)
    target = target.to(device)

    z_trained = []
    x_reconstructed = []

    progress = range(num_i)
    if show_progress:
        progress = tqdm(progress)
    for _ in progress:
        y = diffusion.sample(x=z, sequence=False, ema=True, num_t_steps=num_t_steps)[0]
        z_trained.append(z.detach().cpu())
        x_reconstructed.append(y.detach().cpu())

        loss = rutils.criterion(y, target, name=criterion)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        if show_progress:
            progress.set_description(f"Loss: {loss.detach().cpu().abs().numpy():.7}")
            progress.refresh()

    z_trained.append(z.detach().cpu())
    y = diffusion.sample(x=z, sequence=False, ema=True, num_t_steps=num_t_steps)[0]
    x_reconstructed.append(y.detach().cpu())

    if not sequence:
        z_trained = z_trained[-1]
        x_reconstructed = x_reconstructed[-1]
    return z_trained, x_reconstructed
