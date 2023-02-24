from torch import jit, nn
from torchvision import datasets, transforms


def get_dataset(name, shape, root="~/.torch/datasets", split="train",
                download=False):
    assert shape[-2] == shape[-1], "Last two dimensions of shape must match (i.e., square)."

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
