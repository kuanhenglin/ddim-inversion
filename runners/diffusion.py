from functools import partial
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils import data, tensorboard
import torchvision.utils as vutils
from tqdm.notebook import tqdm

from networks.unet import UNet
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
            attention_sizes=config.network.attention_sizes, dropout=config.network.dropout,
            group_norm=config.network.group_norm, do_conv_sample=config.network.do_conv_sample)
        network = network.to(device)
        self.network = network

        config_transform = {"zero_center": config.data.zero_center}
        self.data_transform = partial(dutils.data_transform, **config_transform)
        self.inverse_data_transform = partial(dutils.inverse_data_transform, **config_transform)

        self.x_fixed = torch.randn(config.training.log_batch_size, *config.data.shape,
                                   device=self.device, dtype=torch.float32)

        self.datetime = datetime.now().strftime("%y%m%d_%H%M%S")

    def train(self, log_tensorboard=None):
        config = self.config

        log_tensorboard = utils.get_default(log_tensorboard, default=config.training.tensorboard)
        writer = None
        if log_tensorboard:
            writer = tensorboard.SummaryWriter(log_dir=f"logs/run_{self.datetime}")

        train_dataset = dutils.get_dataset(name=config.data.dataset, shape=config.data.shape,
                                           shape_original=config.data.shape_original, split="train",
                                           download=config.data.download)
        train_loader = data.DataLoader(train_dataset, batch_size=config.training.batch_size,
                                       shuffle=True, num_workers=config.data.num_workers)

        network = self.network

        optimizer = outils.get_optimizer(
            name=config.optimizer.name, parameters=network.parameters(),
            learning_rate=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay, beta_1=config.optimizer.beta_1,
            amsgrad=config.optimizer.amsgrad, epsilon=config.optimizer.epsilon)

        i = 0
        epoch_total = int(np.ceil(config.data.num_train / config.training.batch_size))
        for _ in tqdm(range(config.training.epoch_max), position=0):
            for _, (x_0, _) in tqdm(enumerate(train_loader), total=epoch_total, leave=False):

                n = x_0.shape[0]
                network.train()

                x_0 = x_0.to(self.device)
                x_0 = self.data_transform(x_0, zero_center=config.data.zero_center)
                e = torch.randn_like(x_0)  # Noise to mix with x_0 to create x_t

                # Arithmetic sampling
                t = torch.randint(low=0, high=config.diffusion.num_t, size=(n // 2 + 1,),
                                  device=self.device)
                t = torch.cat((t, config.diffusion.num_t - t - 1), dim=0)[:n]
                output = self.q_sample(x_0=x_0, t=t, e=e)  # Estimate noise added
                loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)  # Sum of squares

                optimizer.zero_grad()
                loss.backward()

                try:  # Perform gradient clipping only if defined in config
                    nn.utils.clip_grad_norm_(network.parameters(), config.optimizer.gradient_clip)
                except AttributeError:
                    pass
                optimizer.step()

                if log_tensorboard:
                    writer.add_scalar("train_loss", loss.detach().cpu(), global_step=i + 1)
                    if (i + 1) % config.training.log_frequency == 0 or i == 0:
                        log_images = self.sample(x=self.x_fixed, batch_size=64, sequence=False)
                        log_images_grid = vutils.make_grid(log_images)
                        writer.add_image("fixed_noise", log_images_grid.cpu(), global_step=i + 1)

                i += 1

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

    @torch.no_grad()
    def p_sample(self, x, num_t=None, num_t_steps=None, skip_type="uniform", eta=None,
                 sequence=False):
        config = self.config

        self.network.eval()

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

            x_t = x_t_predictions[-1].to(self.device)
            e_t = self.network(x_t, t=t)

            x_0_t = (x_t - (1.0 - a_t).sqrt() * e_t) / a_t.sqrt()  # DDIM Eq. 12, "predicted x_0"
            x_0_predictions.append(x_0_t.detach().cpu())

            # DDIM Eq. 16, s_t is constant for amount of random noise during generation.
            # If eta == 0, then we have DDIM; if eta == 1, then we have DDPM
            s_t = eta * (((1.0 - a_t_next) / (1.0 - a_t)) * ((1.0 - a_t) / a_t_next)).sqrt()
            e = s_t * torch.randn_like(x)  # DDIM Eq. 12, "random noise"

            # DDIM Eq. 12, "direction pointing to x_t"
            x_d = ((1.0 - a_t_next) - s_t.square()).sqrt() * e_t

            x_t_next = a_t_next.sqrt() * x_0_t + x_d + e  # DDIM Eq. 12
            x_t_predictions.append(x_t_next.detach().cpu())

        if not sequence:  # Only return final generated images
            return x_t_predictions[-1]
        return x_t_predictions, x_0_predictions  # Return entire generation process

    @torch.no_grad()
    def sample(self, x=None, batch_size=None, sequence=False, **kwargs):
        config = self.config

        batch_size = utils.get_default(batch_size, default=config.training.batch_size)

        noise = torch.randn(batch_size, *config.data.shape, device=self.device, dtype=torch.float32)
        x = utils.get_default(x, default=noise)
        outputs = self.p_sample(x, sequence=sequence, **kwargs)

        if not sequence:  # Only final generated images
            outputs = self.inverse_data_transform(outputs)
            return outputs

        x_t_predictions, x_0_predictions = outputs
        x_t_predictions = [self.inverse_data_transform(x_t) for x_t in x_t_predictions]
        x_0_predictions = [self.inverse_data_transform(x_0) for x_0 in x_0_predictions]
        return x_t_predictions, x_0_predictions

    def size(self):
        size = utils.get_size(self.network)
        return size
