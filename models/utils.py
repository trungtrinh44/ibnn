from itertools import chain

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from prettytable import PrettyTable


def count_parameters(model, logger):
    table = PrettyTable(["Modules", "Parameters", "Trainable?"])
    total_params = 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        table.add_row([name, param, parameter.requires_grad])
        total_params += param
    logger.info(f"\n{table}")
    logger.info(f"Total Trainable Params: {total_params}")
    return total_params


def swish(x):
    return x*torch.sigmoid(x)


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


def get_activation(name: str):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=False)
    if name == 'softplus':
        return nn.Softplus()
    if name == 'swish':
        return Swish()
    if name == 'selu':
        return nn.SELU()
    if name == 'gelu':
        return nn.GELU()


def get_dimension_size_conv(input_size, padding, stride, kernel):
    return (input_size+2*padding-kernel)//stride + 1


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, init_method='normal', activation='relu'):
        super(Linear, self).__init__(in_features, out_features, bias)
        if init_method == 'orthogonal':
            nn.init.orthogonal_(
                self.weight, nn.init.calculate_gain(activation))
            if bias:
                nn.init.constant_(self.bias, 0.0)
        elif init_method == 'normal':
            nn.init.normal_(self.weight, mean=0.0, std=0.1)
            if bias:
                nn.init.constant_(self.bias, 0.1)
        elif init_method == 'wrn':
            nn.init.kaiming_normal_(self.weight)
            if bias:
                nn.init.constant_(self.bias, 0.0)


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', init_method='normal', activation='relu'):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        if init_method == 'orthogonal':
            nn.init.orthogonal_(
                self.weight, nn.init.calculate_gain(activation))
            if bias:
                nn.init.constant_(self.bias, 0.0)
        elif init_method == 'normal':
            nn.init.normal_(self.weight, mean=0.0, std=0.01)
            if bias:
                nn.init.constant_(self.bias, 0.01)
        # elif init_method == 'wrn':
        #     nn.init.kaiming_normal_(self.weight)
        #     if bias:
        #         nn.init.constant_(self.bias, 0.0)


class StoLayer(nn.Module):
    def __init__(self, in_features, n_components, prior_mean, prior_std):
        super(StoLayer, self).__init__()
        posterior_mean = torch.ones((n_components, *in_features))
        posterior_std = torch.ones((n_components, *in_features))
        # [1, In, 1, 1]
        self.posterior_mean = nn.Parameter(posterior_mean, requires_grad=True)
        self.posterior_std = nn.Parameter(posterior_std, requires_grad=True)
        nn.init.normal_(self.posterior_std, 0.05, 0.02)
        nn.init.normal_(self.posterior_mean, 1.0, 0.5)
        self.posterior_std.data.abs_().expm1_().log_()
        self.prior_mean = nn.Parameter(torch.tensor(prior_mean), requires_grad=False)
        self.prior_std = nn.Parameter(torch.tensor(prior_std), requires_grad=False)
    
    def get_input_sample(self, input, indices):
        mean = self.posterior_mean.repeat((1, 1, *input.shape[2:]))
        std = F.softplus(self.posterior_std).repeat((1, 1, *input.shape[2:]))
        components = D.Normal(mean[indices], std[indices])
        return components.rsample()

    def forward(self, x, indices):
        x = x * self.get_input_sample(x, indices)
        return x
    
    def kl(self, n_sample):
        mean = self.posterior_mean.mean(dim=0)
        std = F.softplus(self.posterior_std).pow(2.0).sum(dim=0).pow(0.5) / self.posterior_std.size(0)
        components = D.Normal(mean, std)
        prior = D.Normal(self.prior_mean, self.prior_std)
        return D.kl_divergence(components, prior).sum()

class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, probs, labels):
        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=probs.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * \
                confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin -
                                 accuracy_in_bin) * prop_in_bin

        return ece

def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Arguments:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)