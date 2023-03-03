import os
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.utils import data, tensorboard
import torchvision.utils as vutils
from tqdm import tqdm

from networks.unet import UNet
from networks.ema import EMA
import utilities.data as dutils
import utilities.optimizer as outils
import utilities.runner as rutils
import utilities.utilities as utils


class Diffusion:

    def __init__(self, config, device=None):

        self.config = config

        if device is None:  # Use GPU on CPU depending on hardware
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        betas = rutils.beta_schedule(
            schedule=config.diffusion.beta_schedule, beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end, num_t=config.diffusion.num_t)
        betas = betas.to(device)
        self.betas = betas

        network = UNet(
            in_shape=config.data.shape, hidden_channels=config.network.hidden_channels,
            num_blocks=config.network.num_blocks, channel_mults=config.network.channel_mults,
            attention_sizes=config.network.attention_sizes,
            time_embed_channels=config.network.embed_channels, dropout=config.network.dropout,
            num_groups=config.network.num_groups, do_conv_sample=config.network.do_conv_sample)
        network = network.to(device)
        self.network = network

        # Exponential moving average for weights (duplicate network)
        if config.network.ema > 0.0:
            self.ema = EMA(network, mu=config.network.ema, device=device)
        else:
            self.ema = None

        config_transform = {"zero_center": config.data.zero_center, "clamp": config.data.clamp}
        self.data_transform = partial(dutils.data_transform, **config_transform)
        self.inverse_data_transform = partial(dutils.inverse_data_transform, **config_transform)

        self.data_augmentation = dutils.Augmentation(flip_horizontal=config.data.flip_horizontal,
                                                     flip_vertical=config.data.flip_vertical)

        self.x_fixed = torch.randn(config.training.log_batch_size, *config.data.shape,
                                   device=self.device, dtype=torch.float32)

        self.datetime = datetime.now().strftime("%y%m%d_%H%M%S")

    def train(self, log_tensorboard=None):
        config = self.config

        log_tensorboard = utils.get_default(log_tensorboard, default=config.training.tensorboard)
        writer = None
        log_path = f"logs/run_{self.datetime}"
        if log_tensorboard:
            writer = tensorboard.SummaryWriter(log_dir=log_path)
        else:
            os.mkdir(log_path)

        torch.save(self.x_fixed.detach().cpu(), f"{log_path}/fixed_noise.pth")

        train_dataset = dutils.get_dataset(name=config.data.dataset, shape=config.data.shape,
                                           root=config.data.root, split="train",
                                           download=config.data.download)
        train_loader = data.DataLoader(train_dataset, batch_size=config.training.batch_size,
                                       shuffle=True, num_workers=config.data.num_workers)

        network = self.network
        ema = self.ema

        optimizer = outils.get_optimizer(
            name=config.optimizer.name, parameters=network.parameters(),
            learning_rate=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay, beta_1=config.optimizer.beta_1,
            amsgrad=config.optimizer.amsgrad, epsilon=config.optimizer.epsilon)

        progress = tqdm(total=config.training.num_i, position=0)
        i = 0
        while i < config.training.num_i:
            for _, (x_0, _) in enumerate(train_loader):

                n = x_0.shape[0]
                network.train()

                x_0 = x_0.to(self.device)
                x_0 = self.data_transform(x_0)
                e = torch.randn_like(x_0)  # Noise to mix with x_0 to create x_t

                # Arithmetic sampling
                t = torch.randint(low=0, high=config.diffusion.num_t, size=(n // 2 + 1,),
                                  device=self.device)
                t = torch.cat((t, config.diffusion.num_t - t - 1), dim=0)[:n]
                output = self.q_sample(x_0=x_0, t=t, e=e)  # Estimate noise added
                loss = rutils.criterion(output, e, name=config.training.criterion)

                optimizer.zero_grad()
                loss.backward()

                try:  # Perform gradient clipping only if defined in config
                    nn.utils.clip_grad_norm_(network.parameters(), config.optimizer.gradient_clip)
                except AttributeError:
                    pass
                optimizer.step()

                if ema is not None:
                    ema.update(network)

                if log_tensorboard:
                    writer.add_scalar("train_loss", loss.detach().cpu(), global_step=i + 1)
                    if (i + 1) % config.training.log_frequency == 0 or i == 0:
                        log_images_grid = self.log_grid(x=self.x_fixed, batch_size=64, ema=False)
                        writer.add_image("fixed_noise", log_images_grid, global_step=i + 1)
                        log_ema_images_grid = self.log_grid(x=self.x_fixed, batch_size=64, ema=True)
                        writer.add_image("fixed_noise_ema", log_ema_images_grid, global_step=i + 1)

                if (i + 1) % config.training.save_frequency == 0 or i + 1 == config.training.num_i:
                    i_file = str(i + 1).zfill(len(str(config.training.num_i)))
                    torch.save(self.network.state_dict(),
                               f"{log_path}/network_{i_file}.pth")
                    if ema is not None:
                        torch.save(self.ema.state_dict(),
                                   f"{log_path}/ema_{i_file}.pth")

                i += 1
                progress.update(1)
                if i >= config.training.num_i:  # Training for exactly num_i iterations
                    break

    def q_sample(self, x_0, t, e):
        """
        We express x_t as a linear combination of x_0 and noise e because
            q(x_t | x_0) = N(x_t; sqrt(a_t) x_0, (1 - a_t) I) .
        This is the key difference between DDPM and DDIM. The former tries to approximate e where
            x_(t - 1) + e = x_t ,
        whereas the latter mixes e and x_0 via a_t (see above). Because we are now using x_0, this
        is no longer a Markov chain, and during the p_sample process we can speed up the sampling
        process by skipping an arbitrary number of t each time just by parameterizing a_t.

        For more information: https://strikingloo.github.io/wiki/ddim
        """
        b = self.betas
        with torch.no_grad():
            a_t = (1.0 - b).cumprod(dim=0).index_select(dim=0, index=t)[:, None, None, None]
            x_t = a_t.sqrt() * x_0 + (1.0 - a_t).sqrt() * e  # DDIM Eq. 4
        output = self.network(x_t, t=t.to(torch.float32))  # Predicted e
        return output

    def p_sample(self, x, network=None, num_t=None, num_t_steps=None, skip_type="uniform", eta=None,
                 ema=True, sequence=False):
        config = self.config

        if ema and (network is None) and (self.ema is not None):
            network = self.ema

        network = utils.get_default(network, default=self.network)
        network.eval()

        # We can choose to start from a non-max num_t (e.g., for partial generation)
        num_t = utils.get_default(num_t, default=config.diffusion.num_t)
        num_t_steps = utils.get_default(num_t_steps, default=config.diffusion.num_t_steps)

        eta = utils.get_default(eta, default=config.diffusion.eta)

        if skip_type == "uniform":
            t_skip = num_t // num_t_steps
            t_sequence = list(range(0, num_t, t_skip))
        elif skip_type == "quadratic":
            t_sequence = np.square(np.linspace(0, np.sqrt(0.8 * num_t), num_t_steps))
            t_sequence = [int(t) for t in t_sequence]
        else:
            raise NotImplementedError(f"Time skip type {skip_type} not supported.")

        n = x.shape[0]
        b = self.betas
        t_sequence_next = [-1] + t_sequence[:-1]
        x_0_predictions = []
        x_t_predictions = [x]

        for i, j in zip(reversed(t_sequence), reversed(t_sequence_next)):
            t = (torch.ones(n) * i).to(self.device)  # Same time across batch
            t_next = (torch.ones(n) * j).to(self.device)

            a_t = rutils.alpha(b=b, t=t)
            a_t_next = rutils.alpha(b=b, t=t_next)

            x_t = x_t_predictions[-1]
            e_t = network(x_t, t=t)

            x_0_t = (x_t - (1.0 - a_t).sqrt() * e_t) / a_t.sqrt()  # DDIM Eq. 12, "predicted x_0"
            x_0_predictions.append(x_0_t.detach().cpu())

            # DDIM Eq. 16, s_t is constant for amount of random noise during generation.
            # If eta == 0, then we have DDIM; if eta == 1, then we have DDPM
            s_t = eta * (((1.0 - a_t_next) / (1.0 - a_t)) * ((1.0 - a_t) / a_t_next)).sqrt()
            e = s_t * torch.randn_like(x)  # DDIM Eq. 12, "random noise"

            # DDIM Eq. 12, "direction pointing to x_t"
            x_d = ((1.0 - a_t_next) - s_t.square()).sqrt() * e_t

            x_t_next = a_t_next.sqrt() * x_0_t + x_d + e  # DDIM Eq. 12
            x_t_predictions.append(x_t_next)  # Only keep gradients of final x_t prediction
            x_t_predictions[-2] = x_t_predictions[-2].detach().cpu()

        if not sequence:  # Only return final generated images
            return x_t_predictions[-1]
        return x_t_predictions, x_0_predictions  # Return entire generation process

    def sample(self, x=None, batch_size=None, sequence=False, **kwargs):
        config = self.config

        batch_size = utils.get_default(batch_size, default=config.training.batch_size)

        noise = torch.randn(batch_size, *config.data.shape, device=self.device, dtype=torch.float32)
        x = utils.get_default(x, default=noise)
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        outputs = self.p_sample(x, sequence=sequence, **kwargs)

        if not sequence:  # Only final generated images
            outputs = self.inverse_data_transform(outputs)
            return outputs

        x_t_predictions, x_0_predictions = outputs
        x_t_predictions = [self.inverse_data_transform(x_t) for x_t in x_t_predictions]
        x_0_predictions = [self.inverse_data_transform(x_0) for x_0 in x_0_predictions]
        return x_t_predictions, x_0_predictions

    def log_grid(self, x=None, batch_size=64, value=1.0, **kwargs):
        if x == "random":
            x = torch.randn(batch_size, *self.config.data.shape).to(self.device)
        x = utils.get_default(x, default=self.x_fixed)
        with torch.no_grad():
            log_images = self.sample(x=x, batch_size=batch_size, sequence=False, **kwargs)
        log_images_grid = vutils.make_grid(log_images, pad_value=value)
        return log_images_grid

    def size(self):
        size = utils.get_size(self.network)
        return size

    def load(self, path, name, ema=True):
        if ema:
            self.ema.load_state_dict(torch.load(f"{path}/{name}", map_location=self.device))
        else:
            self.network.load_state_dict(torch.load(f"{path}/{name}", map_location=self.device))
        self.x_fixed = torch.load(f"{path}/fixed_noise.pth", map_location=self.device)

    def freeze(self, ema=True):
        if ema:
            params = self.ema.ema.parameters()
        else:
            params = self.network.parameters()
        for param in params:  # Freeze all layers to save backpropagation memory
            param.requires_grad_(False)
