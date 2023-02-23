from copy import deepcopy

import torch


class EMA:

    def __init__(self, network, mu=0.999, device=torch.device("cuda")):
        self.mu = mu
        self.ema = deepcopy(network).to(device)
        self.ema.eval()

    def __call__(self, *args, **kwargs):
        return self.ema(*args, **kwargs)

    @torch.no_grad()
    def update(self, network):
        state_network = network.state_dict()
        state_ema = self.ema.state_dict()
        for name in state_network.keys():
            state_ema[name].copy_(state_network[name].data.mul(1 - self.mu) +
                                  state_ema[name].data.mul(self.mu))

    def state_dict(self):
        return self.ema.state_dict()
