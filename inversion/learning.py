import os
from datetime import datetime
from functools import partial

import torch
from torch.utils import tensorboard
import torchvision.utils as vutils
from tqdm import tqdm

from networks.unet import UNet
import utilities.data as dutils
import utilities.optimizer as outils
import utilities.runner as rutils
import utilities.utilities as utils


class NoiseEncoder:

    def __init__(self, config, network_args, loss_type, diffusion, device=None):
        self.config = config

        if device is None:  # Use GPU on CPU depending on hardware
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        network = UNet(in_shape=config.data.shape, **network_args)
        network = network.to(device)
        self.network = network

        assert loss_type in ("direct", "reconstruction"),\
            f"Loss type {loss_type} must be either \"direct\" or \"reconstruction\""
        self.loss_type = loss_type

        self.diffusion = diffusion

        config_transform = {"zero_center": config.data.zero_center, "clamp": config.data.clamp}
        self.data_transform = partial(dutils.data_transform, **config_transform)
        self.inverse_data_transform = partial(dutils.inverse_data_transform, **config_transform)

        self.datetime = datetime.now().strftime("%y%m%d_%H%M%S")

    def train(self, diffusion_args, optimizer_args, batch_size=64, num_i=6000, z_criterion="l1",
              x_criterion="psnr", loader=None, show_progress=True, log_tensorboard=True,
              log_frequency=300):
        config = self.config

        writer = None
        log_path = f"logs/encoder_{self.datetime}"
        if log_tensorboard:
            writer = tensorboard.SummaryWriter(log_dir=log_path)
        else:
            os.mkdir(log_path)

        if loader is not None:
            loader = dutils.get_loader_samples(batch_size=batch_size, root=loader,
                                               stop_iteration=False)

        maximize = self.loss_type == "reconstruction" and x_criterion in ("psnr", "ssim")
        optimizer = outils.get_optimizer(parameters=self.network.parameters(), maximize=maximize,
                                         **optimizer_args)

        x_hat = None
        x_loss = None

        progress = range(num_i)
        if show_progress:
            progress = tqdm(progress)
        for i in progress:
            do_log = self.loss_type == "reconstruction" or (i + 1) % log_frequency == 0 or i == 0

            if loader is None:
                with torch.no_grad():
                    z = torch.randn(batch_size, *config.data.shape).to(self.device)
                    x = self.diffusion.sample(x=z, sequence=False, **diffusion_args)
            else:
                x, z = next(iter(loader))
                x = x.to(self.device)
                z = z.to(self.device)

            if self.loss_type == "direct":
                z_hat = self.network(x)
                z_loss = rutils.criterion(z_hat, z, name=z_criterion)
                z_loss.backward()
                if do_log:
                    with torch.no_grad():
                        x_hat = self.diffusion.sample(x=z_hat, sequence=False, ema=True,
                                                      **diffusion_args)
                        x_loss = rutils.criterion(x_hat, x, name=x_criterion)
            elif self.loss_type == "reconstruction":
                z_hat = self.network(x)
                with torch.no_grad():
                    z_loss = rutils.criterion(z_hat, z, name=z_criterion)
                x_hat = self.diffusion.sample(x=z_hat, sequence=False, ema=True, **diffusion_args)
                x_loss = rutils.criterion(x_hat, x, name=x_criterion)
                x_loss.backward()
            else:
                raise NotImplementedError(self.loss_type)

            optimizer.zero_grad()
            optimizer.step()

            if show_progress:
                description = f"Loss  |  z: {z_loss.detach().cpu().abs().numpy():.7}  "\
                              f"x: {x_loss.detach().cpu().abs().numpy():.7}"
                progress.set_description(description)
                progress.refresh()

            if log_tensorboard:
                writer.add_scalar("z_loss", z_loss.detach().cpu(), global_step=i + 1)
                if do_log:
                    writer.add_scalar("x_loss", x_loss.detach().cpu(), global_step=i + 1)
                    x_grid = vutils.make_grid(x[:64], pad_value=1.0)
                    writer.add_image("x", x_grid, global_step=i + 1)
                    x_hat_grid = vutils.make_grid(x_hat[:64], pad_value=1.0)
                    writer.add_image("x_hat", x_hat_grid, global_step=i + 1)

            i += 1

    def size(self):
        size = utils.get_size(self.network)
        return size
