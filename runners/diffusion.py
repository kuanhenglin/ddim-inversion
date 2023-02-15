import numpy as np
import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm

from networks.unet import UNet
import utilities.data as dutils
import utilities.network as nutils
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
            attention_sizes=config.network.attenion_sizes, dropout=config.network.dropout,
            group_norm=config.network.group_norm, do_conv_sample=config.network.do_conv_sample)
        network = network.to(device)
        self.network = network

    def train(self):

        config = self.config

        train_dataset = None  # TODO: Implement get_dataset
        train_loader = data.DataLoader(train_dataset, batch_size=config.training.batch_size,
                                       shuffle=True, num_workers=4)

        network = self.network

        optimizer = outils.get_optimizer(
            name=config.optimizer.name, parameters=network.parameters(),
            learning_rate=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay, beta_1=config.optimizer.beta_1,
            amsgrad=config.optimizer.amsgrad, epsilon=config.optimizer.epsilon)

        i = 0
        for _ in tqdm(range(config.training.epoch_max)):
            for _, (x_0, _) in tqdm(enumerate(train_loader), leave=False):

                n = x_0.shape[0]
                network.train()

                x_0 = x_0.to(self.device)
                x_0 = dutils.data_transform(x_0, zero_center=config.data.zero_center)
                e = torch.randn_like(x_0)  # Noise to mix with x_0 to create x_t

                # Arithmetic sampling
                t = torch.randint(low=0, high=config.diffusion.num_t, size=(n // 2 + 1))
                t = torch.cat((t, config.diffusion.num_t - t - 1))[:n]
                output = self.q_sample(x_0=x_0, t=t, e=e)
                loss = nutils.get_loss(output, e)  # Compute sum of squares

                optimizer.zero_grad()
                loss.backward()

                try:  # Perform gradient clipping only if defined in config
                    nn.utils.clip_grad_norm_(network.parameters(), config.optimizer.gradient_clip)
                except AttributeError:
                    pass
                optimizer.step()

                if (i + 1) % config.training.log_frequency or i == 0:
                    tqdm.write(f"Loss: {loss.detach().cpu().numpy()}")  # TODO: Some evaluate metric

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
        a_t = (1 - b).cumprod(dim=0).index_select(dim=0, index=t)[:, None, None, None]
        x_t = a_t.sqrt() * x_0 + (1.0 - a_t).sqrt() * e  # DDIM paper, Eq. 4
        output = self.network(x_t, t=t.to(torch.float32))
        return output

    @torch.no_grad()
    def p_sample(self, x_t, num_t=None, num_t_steps=None, skip_type="uniform"):
        config = self.config

        # We can choose to start from a non-max num_t (e.g., for partial generation)
        num_t = utils.get_default(num_t, default=config.diffusion.num_t)
        num_t_steps = utils.get_default(num_t_steps, default=config.diffusion.num_t_steps)

        if skip_type == "uniform":
            t_skip = num_t // num_t_steps
            t_sequence = list(range(0, num_t, t_skip))
        elif skip_type == "quadratic":
            t_sequence = np.square(np.linspace(0, np.sqrt(0.8 * num_t), num_t_steps))
            t_sequence = [int(t) for t in t_sequence]
        else:
            raise NotImplementedError(f"Time skip type {skip_type} not supported.")

        n = x_t.shape[0]
        t_sequence_next = [-1] + t_sequence[:-1]
        x_0_predictions = []
        x_s = [x_t]
        for i, j in zip(reversed(t_sequence), reversed(t_sequence_next)):
            t = (torch.ones(n) * i).to(x_t.device)
            t_next = (torch.ones(n) * j).to(x_t.device)
            # TODO: Finish this thing

