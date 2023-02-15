import torch


def data_transform(x, zero_center=True):
    if zero_center:
        x = 2.0 * x - 1.0
    return x
