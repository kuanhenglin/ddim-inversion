from torchvision import datasets, transforms


def get_dataset(name, shape, shape_original, split="train", download=False):
    assert shape[-2] == shape[-1], "Last two dimensions of shape must match (i.e., square)."

    shape_small = min(*shape_original[-2:])
    transform = transforms.Compose([transforms.CenterCrop((shape_small,) * 2),
                                    transforms.Resize(shape[-2:]), transforms.ToTensor()])
    if name == "celeba":
        dataset = datasets.CelebA(root="~/.torch/datasets", split=split, transform=transform,
                                  download=download)
    elif name == "flowers102":
        dataset = datasets.Flowers102(root="~./torch/datasets", split=split, transform=transform,
                                      download=download)
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
