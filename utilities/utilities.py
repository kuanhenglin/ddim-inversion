import numpy as np


def get_size(network):
    size = 0
    for param in network.parameters():
        size += np.prod(param.shape)
    return size


def get_default(variable, default):
    if variable is None:
        return default
    else:
        return variable
