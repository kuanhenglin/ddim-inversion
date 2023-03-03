from torch import nn
import torch.nn.functional as F
from torchvision import models


class InceptionV3(nn.Module):

    def __init__(self, resize=True, normalize=True, requires_grad=False):
        super().__init__()
        self.resize = resize
        self.normalize = normalize

        # Note that the PyTorch implementation of Inception V3 is a little different than the
        # commonly-used implementation of Inception V3 for FID scores in literature, so there may be
        # some discrepancies. TODO: Implement option for literature-accurate Inception V3
        # See: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/inception.py
        self.inception = models.inception_v3(weights="IMAGENET1K_V1")
        self.inception.fc = nn.Identity()  # Omit last fully-connected layer to get features
        self.inception.aux_logits = False  # Turn off auxiliary logits (for training purposes)

        for param in self.parameters():
            param.requires_grad_(requires_grad)

    def forward(self, x):
        self.inception.eval()

        if self.resize:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        if self.normalize:
            x = 2.0 * x + 1.0

        f = self.inception(x)
        return f
