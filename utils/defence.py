import torch
import syft as sy
from torch import nn
from torch.optim import SGD


def DP_Adam(model, lr=0.01, noise_multiplier=1.0, max_grad_norm=1.0):
    return sy.optim.DPAdam(model.parameters(), lr=lr, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm)


def DP_SGD(model, lr=0.01, noise_multiplier=1.0, max_grad_norm=1.0):
    return sy.optim.DPSGD(model.parameters(), lr=lr, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm)