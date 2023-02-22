import argparse
import yaml

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


def get_namespace(config_dict):
    config = argparse.Namespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            value_new = get_namespace(value)  # Recursively convert to Namespace
        else:
            value_new = value
        setattr(config, key, value_new)
    return config


def get_yaml(path):
    with open("./configs/celeba.yml") as f:
        config = yaml.safe_load(f)
    config = get_namespace(config)
    return config
