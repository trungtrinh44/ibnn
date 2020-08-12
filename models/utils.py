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


class MixtureGaussianWrapper(nn.Module):
    def __init__(self, layer, prior_mean=0.0, prior_std=1.0, posterior_p=0.5, posterior_std=1.0, train_posterior_std=False):
        super(MixtureGaussianWrapper, self).__init__()
        self.layer = layer
        self.posterior_params = nn.ParameterDict({
            'p': nn.Parameter(torch.tensor([posterior_p, 1-posterior_p]), requires_grad=False),
            'std': nn.Parameter(torch.tensor(posterior_std), requires_grad=train_posterior_std)
        })
        self.prior_params = nn.ParameterDict({
            'mean': nn.Parameter(torch.tensor(prior_mean), requires_grad=False),
            'std': nn.Parameter(torch.tensor(prior_std), requires_grad=False)
        })

    def draw_sample_from_x(self, x, L=1, return_log_prob=False):
        if L == 1:
            n_sample = ()
        else:
            n_sample = (L,)
        means = torch.cat([
            torch.zeros_like(x).unsqueeze_(-1),
            x.unsqueeze(-1)
        ], dim=-1)
        zero_mask = (x.detach() != 0.0).float()
        std = self.posterior_params['std']*x.abs().unsqueeze(-1)
        std = torch.max(std, torch.tensor(1e-9, device=std.device))
        normal = D.Normal(means, std)
        categorical = D.OneHotCategorical(probs=self.posterior_params['p'])

        p_sample = categorical.sample(n_sample + x.shape)
        x_sample = (p_sample*normal.rsample(n_sample)).sum(dim=-1)
        x_sample = x_sample * zero_mask
        if return_log_prob:
            posterior_log_prob = torch.logsumexp(normal.log_prob(x_sample.unsqueeze(-1)) + self.posterior_params['p'].log(), -1)
            return x_sample, posterior_log_prob*zero_mask
        return x_sample 

    def kl(self, n_sample):
        # Monte Carlo approximation for the weights KL
        x_sample, posterior_log_prob = self.draw_sample_from_x(self.layer.weight, L=n_sample, return_log_prob=True)
        prior_log_prob = self.prior().log_prob(x_sample)
        logdiff = (posterior_log_prob - prior_log_prob).mean(dim=0)
        return logdiff.sum()

    def prior(self):
        return D.Normal(self.prior_params['mean'], self.prior_params['std'])

    def forward(self, x):
        x_sample = self.draw_sample_from_x(x)
        output = self.layer(x_sample)
        return output

class GaussianWrapper(nn.Module):
    def __init__(self, layer, prior_mean=0.0, prior_std=1.0, posterior_std=1.0, train_posterior_std=False):
        super(GaussianWrapper, self).__init__()
        self.layer = layer
        self.posterior_params = nn.ParameterDict({
            'std': nn.Parameter(torch.tensor(posterior_std), requires_grad=train_posterior_std)
        })
        self.prior_params = nn.ParameterDict({
            'mean': nn.Parameter(torch.tensor(prior_mean), requires_grad=False),
            'std': nn.Parameter(torch.tensor(prior_std), requires_grad=False)
        })

    def draw_sample_from_x(self, x, L=1, return_log_prob=False):
        if L == 1:
            n_sample = ()
        else:
            n_sample = (L,)
        means = x
        zero_mask = (means.detach() != 0.0).float()
        std = self.posterior_params['std']*means.abs()
        std = torch.max(std, torch.tensor(1e-9, device=std.device))
        normal = D.Normal(means, std)

        x_sample = normal.rsample(n_sample)
        x_sample = x_sample * zero_mask
        if return_log_prob:
            posterior_log_prob = normal.log_prob(x_sample)
            return x_sample, posterior_log_prob*zero_mask
        return x_sample 

    def kl(self, n_sample):
        # Monte Carlo approximation for the weights KL
        means = self.layer.weight
        zero_mask = (means.detach() != 0.0).float()
        std = self.posterior_params['std']*means.abs()
        std = torch.max(std, torch.tensor(1e-9, device=std.device))
        normal = D.Normal(means, std)
        kl = D.kl_divergence(normal, self.prior())
        return kl.sum()

    def prior(self):
        return D.Normal(self.prior_params['mean'], self.prior_params['std'])

    def forward(self, x):
        x_sample = self.draw_sample_from_x(x)
        output = self.layer(x_sample)
        return output
