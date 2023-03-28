import torch
import numpy as np
from torch import nn, optim
from tqdm.auto import tqdm

import utilities.optimizer as outils


class LogisticRegression(nn.Module):

    def __init__(self, in_shape):
        super().__init__()
        self.in_shape = in_shape
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(int(np.prod(in_shape)), 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x


class AttributeClassifier:

    def __init__(self, in_shape, weight_decay, diffusion, resnet, resnet_targets, target, device):
        super().__init__()
        self.device = device

        self.resnet = resnet
        self.resnet.eval()
        self.diffusion = diffusion
        self.target_index = resnet_targets.index(target)

        self.classifier = LogisticRegression(in_shape)
        self.classifier.to(device)
        self.weight_decay = weight_decay

    def __call__(self, x):
        x = self.classifier(x)
        return x

    def train(self, config, diffusion_args, loader=None, batch_size=128,
              num_i=10000, show_progress=True):

        optimizer = optim.SGD(self.classifier.parameters(), lr=0.1, momentum=0.9, nesterov=True,
                              weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[num_i // 2, (num_i // 4) * 3], gamma=0.1)

        criterion = nn.BCEWithLogitsLoss()

        progress = range(num_i)
        if show_progress:
            progress = tqdm(progress)
        for i in progress:
            if loader is None:  # Generate noise-image pair
                with torch.no_grad():
                    z = torch.randn(batch_size, *config.data.shape).to(self.device)
                    x = self.diffusion.sample(x=z, sequence=False, **diffusion_args)
            else:  # Get noise-image pair from pre-computed loader
                x, z = next(iter(loader))
                x = x.to(self.device)
                z = z.to(self.device)

            with torch.no_grad():  # Use ResNet to get binary attribute
                y = (self.resnet(x)[:, self.target_index] >= 0).to(torch.float32)

            y_hat = self.classifier(z).squeeze(dim=1)
            loss = criterion(y_hat, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            progress.set_description(
                f"Training loss: {loss.item():.7}   |   "
                f"Weight norm: {torch.norm(self.classifier.linear.weight).item():.7}")
            
    def eval(self):
        self.classifier.eval()

    def get_direction(self, normalize=True):
        direction = self.classifier.linear.weight.reshape(*self.classifier.in_shape).detach()
        if normalize:
            direction = direction / torch.norm(direction)
        return direction
