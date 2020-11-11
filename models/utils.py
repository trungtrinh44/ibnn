from itertools import chain

import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from prettytable import PrettyTable
from torch.nn.modules.utils import _pair
from copy import deepcopy

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

def recursive_traverse(module, layers):
    children = list(module.children())
    if len(children) > 0:
        for child in children:
            recursive_traverse(child, layers)
    else:
        layers.append(module)

class StoLayer(object):
    @classmethod
    def convert_deterministic(cls, sto_model, index, det_model):
        param_tensors = []
        buffer_tensors = []
        layers = []
        recursive_traverse(sto_model, layers)
        for module in layers:
            if isinstance(module, StoLayer):
                module = module.to_det_module(index)
            param_tensors.extend(module.parameters())
            buffer_tensors.extend(module.buffers())
        for p1, p2 in zip(det_model.parameters(), param_tensors):
            p1.data = p2.data
        for p1, p2 in zip(det_model.buffers(), buffer_tensors):
            p1.data = p2.data
        return det_model
    
    @staticmethod
    def get_mask(mean, index):
        if index == 'ones':
            return torch.ones(mean.shape[1:], device=mean.device)
        if index == 'mean':
            return mean.mean(dim=0)
        return mean[index]
    
    def to_det_module(self, index):
        raise NotImplementedError()

    def sto_init(self, in_features, n_components, prior_mean, prior_std, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02)):
        # [1, In, 1, 1]
        self.posterior_U_mean = nn.Parameter(torch.ones((n_components, *in_features)), requires_grad=True)
        self.posterior_U_std = nn.Parameter(torch.ones((n_components, *in_features)), requires_grad=True)
        nn.init.normal_(self.posterior_U_std, posterior_std_init[0], posterior_std_init[1])
        nn.init.normal_(self.posterior_U_mean, posterior_mean_init[0], posterior_mean_init[1])
        self.posterior_U_std.data.abs_().expm1_().log_()

        if self.bias is not None:
            self.posterior_B_mean = nn.Parameter(torch.ones((n_components, 1, *in_features[1:])), requires_grad=True)
            self.posterior_B_std = nn.Parameter(torch.ones((n_components, 1, *in_features[1:])), requires_grad=True)
            nn.init.normal_(self.posterior_B_std, posterior_std_init[0], posterior_std_init[1])
            nn.init.normal_(self.posterior_B_mean, posterior_mean_init[0], posterior_mean_init[1])
            self.posterior_B_std.data.abs_().expm1_().log_()

        self.prior_mean = nn.Parameter(torch.tensor(prior_mean), requires_grad=False)
        self.prior_std = nn.Parameter(torch.tensor(prior_std), requires_grad=False)
        self.posterior_mean_init = posterior_mean_init
        self.posterior_std_init = posterior_std_init
        self.__aux_dim = in_features[1:]
    
    def get_mult_noise(self, input, indices):
        mean = self.posterior_U_mean
        std = F.softplus(self.posterior_U_std)
        components = D.Normal(mean[indices], std[indices])
        return components.rsample()
    
    def get_add_noise(self, input, indices):
        mean = self.posterior_B_mean
        std = F.softplus(self.posterior_B_std)
        components = D.Normal(mean[indices], std[indices])
        return components.rsample()

    def mult_noise(self, x, indices):
        x = x * self.get_mult_noise(x, indices)
        return x

    def add_bias(self, x, indices):
        x = x + self.bias.view(-1, *self.__aux_dim) * self.get_add_noise(x, indices)
        return x
    
    def kl(self):
        return self._kl(self.posterior_U_mean, self.posterior_U_std) + (0 if self.bias is None else self._kl(self.posterior_B_mean, self.posterior_B_std))
    
    def _kl(self, pos_mean, pos_std):
        mean = pos_mean.mean(dim=0)
        std = F.softplus(pos_std).pow(2.0).sum(0).pow(0.5) / pos_std.size(0)
        components = D.Normal(mean, std)
        prior = D.Normal(self.prior_mean, self.prior_std)
        return D.kl_divergence(components, prior).sum()
    
    def sto_extra_repr(self):
        return f"n_components={self.posterior_U_mean.size(0)}, prior_mean={self.prior_mean.data.item()}, prior_std={self.prior_std.data.item()}, posterior_mean_init={self.posterior_mean_init}, posterior_std_init={self.posterior_std_init}"

class StoConv2d(nn.Conv2d, StoLayer):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = 'zeros',
        n_components=8, prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02)
    ):
        super(StoConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.sto_init((in_channels, 1, 1), n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init)

    def _conv_forward(self, x):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            self.weight, None, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(x, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def forward(self, x, indices):
        x = self.mult_noise(x, indices)
        x = self._conv_forward(x)
        if self.bias is not None:
            x = self.add_bias(x, indices)
        return x
    
    def to_det_module(self, index):
        new_module = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias is not None, self.padding_mode)
        U_mask = StoLayer.get_mask(self.posterior_U_mean, index)
        new_module.weight.data = self.weight.data * U_mask
        if self.bias is not None:
            B_mask = StoLayer.get_mask(self.posterior_B_mean, index).squeeze()
            new_module.bias.data = self.bias.data * B_mask
        return new_module

    def extra_repr(self):
        return f"{super(StoConv2d, self).extra_repr()}, {self.sto_extra_repr()}"

class StoLinear(nn.Linear, StoLayer):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True,
        n_components=8, prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02)
    ):
        super(StoLinear, self).__init__(in_features, out_features, bias)
        self.sto_init((in_features, ), n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init)

    def forward(self, x, indices):
        x = self.mult_noise(x, indices)
        x = F.linear(x, self.weight, None)
        if self.bias is not None:
            x = self.add_bias(x, indices)
        return x

    def to_det_module(self, index):
        new_module = nn.Linear(self.in_features, self.out_features, self.bias is not None)
        U_mask = StoLayer.get_mask(self.posterior_U_mean, index)
        new_module.weight.data = self.weight.data * U_mask
        if self.bias is not None:
            B_mask = StoLayer.get_mask(self.posterior_B_mean, index).squeeze()
            new_module.bias.data = self.bias.data * B_mask
        return new_module

    def extra_repr(self):
        return f"{super(StoLinear, self).extra_repr()}, {self.sto_extra_repr()}"

class BayesianLayer(object):
    pass

class BayesianLinear(nn.Linear, BayesianLayer):
    def __init__(self, in_features, out_features, bias=True, prior_mean=0.0, prior_std=1.0):
        super(BayesianLinear, self).__init__(in_features, out_features, bias)
        self.weight_std = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        nn.init.normal_(self.weight_std, 0.015, 0.001)
        self.weight_std.data.abs_().expm1_().log_()
        if self.bias is not None:
            self.bias_std = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)
            nn.init.normal_(self.bias_std, 0.015, 0.001)
            self.bias_std.data.abs_().expm1_().log_()
        self.prior_mean = nn.Parameter(torch.tensor(prior_mean), requires_grad=False)
        self.prior_std = nn.Parameter(torch.tensor(prior_std), requires_grad=False)
    
    def posterior(self):
        if self.bias is not None:
            return D.Normal(self.weight, F.softplus(self.weight_std)), D.Normal(self.bias, F.softplus(self.bias_std))
        return D.Normal(self.weight, F.softplus(self.weight_std))

    def forward(self, x):
        if self.bias is not None:
            weight_sampler, bias_sampler = self.posterior()
            weight = weight_sampler.rsample()
            bias = bias_sampler.rsample()
        else:
            weight_sampler = self.posterior()
            weight = weight_sampler.rsample()
            bias = None
        return F.linear(x, weight, bias)
    
    def kl(self):
        prior = D.Normal(self.prior_mean, self.prior_std)
        if self.bias is not None:
            weight_sampler, bias_sampler = self.posterior()
            return D.kl_divergence(weight_sampler, prior).sum() + D.kl_divergence(bias_sampler, prior).sum()
        weight_sampler = self.posterior()
        return D.kl_divergence(weight_sampler, prior).sum()

class BayesianConv2d(nn.Conv2d, BayesianLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        prior_mean = 0.0,
        prior_std = 1.0,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True,
        padding_mode = 'zeros'
    ):
        super(BayesianConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.weight_std = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        nn.init.normal_(self.weight_std, 0.015, 0.001)
        self.weight_std.data.abs_().expm1_().log_()
        if self.bias is not None:
            self.bias_std = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)
            nn.init.normal_(self.bias_std, 0.015, 0.001)
            self.bias_std.data.abs_().expm1_().log_()
        self.prior_mean = nn.Parameter(torch.tensor(prior_mean), requires_grad=False)
        self.prior_std = nn.Parameter(torch.tensor(prior_std), requires_grad=False)
    
    def posterior(self):
        if self.bias is not None:
            return D.Normal(self.weight, F.softplus(self.weight_std)), D.Normal(self.bias, F.softplus(self.bias_std))
        return D.Normal(self.weight, F.softplus(self.weight_std))

    def forward(self, x):
        if self.bias is not None:
            weight_sampler, bias_sampler = self.posterior()
            weight = weight_sampler.rsample()
            bias = bias_sampler.rsample()
        else:
            weight_sampler = self.posterior()
            weight = weight_sampler.rsample()
            bias = None
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(x, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    def kl(self):
        prior = D.Normal(self.prior_mean, self.prior_std)
        if self.bias is not None:
            weight_sampler, bias_sampler = self.posterior()
            return D.kl_divergence(weight_sampler, prior).sum() + D.kl_divergence(bias_sampler, prior).sum()
        weight_sampler = self.posterior()
        return D.kl_divergence(weight_sampler, prior).sum()

class ECELoss(nn.Module):
    """
    Ported from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

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
