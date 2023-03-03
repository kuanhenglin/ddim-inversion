import numpy as np
from scipy import linalg
import torch
from tqdm import tqdm

from evaluation.inception import InceptionV3
import utilities.utilities as utils


def matrix_sqrt(x):
    m = x.detach().cpu().numpy().astype(np.float32)
    x = torch.from_numpy(linalg.sqrtm(m, disp=False)[0].real).to(x)
    return x


def frechet_distance(mu_1, sigma_1, mu_2, sigma_2, epsilon=1e-7):
    assert mu_1.shape == mu_2.shape, "Mean vectors have different shapes, " \
                                     f"{mu_1.shape} and {mu_2.shape}"
    assert sigma_1.shape == sigma_2.shape, "Covariances have different shapes, " \
                                           f"{sigma_1.shape} and {sigma_2.shape}"

    sse = (mu_1 - mu_2).square().sum()
    covariance = matrix_sqrt(sigma_1 @ sigma_2)
    if not torch.isfinite(covariance).all():
        I = torch.eye(sigma_1.shape[0])
        covariance = matrix_sqrt(sigma_1 @ sigma_2 + epsilon * I)

    fid = sse + torch.trace(sigma_1) + torch.trace(sigma_2) - 2 * torch.trace(covariance)
    return fid


@torch.no_grad()
def feature_loader(network, loader, device, show_progress=True):
    f = []
    progress = tqdm(iter(loader)) if show_progress else iter(loader)
    for x, _ in progress:
        x = x.to(device)
        f_ = network(x)
        f.append(f_)
    f = torch.cat(f, dim=0)
    return f


@torch.no_grad()
def feature_sample(diffusion, network, batch_size, num_batches, device, noise=None,
                   show_progress=True):
    if noise is None:
        noise = [torch.randn(batch_size, *diffusion.network.in_shape) for _ in range(num_batches)]

    f = []
    progress = tqdm(noise) if show_progress else iter(noise)
    for z in progress:
        z = z.to(device)
        x = diffusion.sample(z, sequence=False)
        f_ = network(x)
        f.append(f_)
    f = torch.cat(f, dim=0)
    return f


@torch.no_grad()
def feature_statistics(f):
    mu = f.mean(dim=0)
    sigma = f.T.cov()  # Transpose equivalent to setting rowvar=False in np.cov()
    return mu, sigma


class FID:

    def __init__(self, train_loader, valid_loader, config, device, show_progress=True):
        self.config = config
        self.device = device

        self.inception = InceptionV3()
        self.inception.to(device)

        f_train = feature_loader(self.inception, train_loader, device=device,
                                 show_progress=show_progress)
        self.mu_train, self.sigma_train = feature_statistics(f_train)
        f_valid = feature_loader(self.inception, valid_loader, device=device,
                                 show_progress=show_progress)
        self.mu_valid, self.sigma_valid = feature_statistics(f_valid)

    @torch.no_grad()
    def __call__(self, diffusion, batch_size=None, num_batches=None, noise=None):
        config = self.config

        batch_size = utils.get_default(batch_size, default=config.evaluation.batch_size)
        num_batches = utils.get_default(num_batches, default=config.evaluation.num_batches)

        f = feature_sample(diffusion, self.inception, batch_size=batch_size,
                           num_batches=num_batches, noise=noise, device=self.device)
        mu_sample, sigma_sample = feature_statistics(f)
        fid_train = frechet_distance(mu_sample, sigma_sample, self.mu_train, self.sigma_train)
        fid_valid = frechet_distance(mu_sample, sigma_sample, self.mu_valid, self.sigma_valid)

        return fid_train, fid_valid
