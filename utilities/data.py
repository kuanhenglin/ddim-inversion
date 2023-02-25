import os

import torch
from torch import jit, nn
from torch.utils import data
from torchvision import datasets, transforms
from PIL import Image


def get_dataset(name, shape, root="~/.torch/datasets", split="train",
                download=False):
    assert shape[-2] == shape[-1], "Last two dimensions of shape must match (i.e., square)."

    root = os.path.expanduser(root)  # Replace "~" with path to user home

    transform = transforms.Compose([transforms.Resize(shape[-2]), transforms.CenterCrop(shape[-2:]),
                                    transforms.ToTensor()])
    if name == "celeba":
        dataset = datasets.CelebA(root=root, split=split, transform=transform, download=download)
    elif name == "flowers102":
        # The split for Flowers102 is weird, as test has the highest number of images
        split_map = {"train": "test", "valid": "val", "test": "train"}
        dataset = datasets.Flowers102(root=root, split=split_map[split], transform=transform,
                                      download=download)
    elif name == "church":
        split_map = {"train": "train", "valid": "val", "test": "val"}
        dataset = datasets.LSUN(root=root, classes=[f"church_outdoor_{split_map[split]}"],
                                transform=transform)
    elif name == "miniplaces":
        dataset = MiniPlaces(root=root, split=split, transform=transform)
    elif name == "cifar10":
        split_map = {"train": True, "valid": False, "test": False}
        dataset = datasets.CIFAR10(root=root, train=split_map[split], transform=transform)
    elif name == "cifar100":
        split_map = {"train": True, "valid": False, "test": False}
        dataset = datasets.CIFAR100(root=root, train=split_map[split], transform=transform)
    else:
        raise NotImplementedError(f"Dataset name {name} not supported.")
    return dataset


def data_transform(x, zero_center=True, **kwargs):
    if zero_center:
        x = 2.0 * x - 1.0
    return x


def inverse_data_transform(x, zero_center=True, clamp=True, **kwargs):
    if zero_center:
        x = (x + 1.0) / 2.0
    if clamp:
        x = x.clamp(min=0.0, max=1.0)
    return x


class MiniPlaces(data.Dataset):
    def __init__(self, root, split, transform=None):
        self.root = f"{root}/miniplaces"
        self.split = split
        self.transform = transform
        self.filenames = []
        self.labels = []
        self.label_dict = {}

        if split == "train":
            label_path = f"{self.root}/train.txt"
        elif split == "valid":
            label_path = f"{self.root}/val.txt"
        else:
            raise NotImplementedError(split)

        with open(label_path) as f:
            lines = f.readlines()

        for line in lines:
            self.filenames.append(line.split(" ")[0])  # File name of image (.jpg)
            line_info = line.split("/")
            label = int(line_info[-1].split(" ")[1][:-1])  # [:-1] to get rid of trailing \n
            self.labels.append(label)
            if split == "train" and label not in self.label_dict:
                self.label_dict[label] = line_info[2]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        image = Image.open(f"{self.root}/images/{self.filenames[i]}")
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[i]
        return image, label


class Augmentation:

    def __init__(self, flip_horizontal=0.0, flip_vertical=0.0, **kwargs):
        augmentations = []
        if flip_horizontal > 0.0:
            augmentations.append(transforms.RandomHorizontalFlip(p=flip_horizontal))
        if flip_vertical > 0.0:
            augmentations.append(transforms.RandomVerticalFlip(p=flip_vertical))
        if len(augmentations) == 0:
            self.augmentation = None
        else:
            augmentation = nn.Sequential(*augmentations)
            self.augmentation = jit.script(augmentation)

    def __call__(self, x):
        if self.augmentation is not None:
            x = self.augmentation(x)
        return x
