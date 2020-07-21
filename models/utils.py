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
        return nn.ReLU(inplace=True)
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


class StochasticLinear(nn.Module):
    def __init__(self, in_features: int, noise_features: int, out_features: int, bias: bool = True,
                 init_mean=0.0, init_log_std=0.0, init_method='normal', activation='relu',
                 freeze_prior_mean=True, freeze_prior_std=False):
        super(StochasticLinear, self).__init__()
        self.fx = Linear(in_features, out_features, bias,
                         init_method, activation)
        self.fz = Linear(noise_features, out_features,
                         False, init_method, activation)
        self.prior_params = nn.ParameterDict({
            'mean': nn.Parameter(torch.full((noise_features,), init_mean, dtype=torch.float32), requires_grad=not freeze_prior_mean),
            'logstd': nn.Parameter(torch.full((noise_features,), init_log_std, dtype=torch.float32), requires_grad=not freeze_prior_std)
        })
        self.posterior_params = nn.ParameterDict({
            'mean': nn.Parameter(torch.full((noise_features,), init_mean, dtype=torch.float32), requires_grad=True),
            'logstd': nn.Parameter(torch.full((noise_features,), init_log_std, dtype=torch.float32), requires_grad=True)
        })

    def parameters(self):
        return self.fx.parameters()

    def stochastic_params(self):
        return chain.from_iterable([
            self.fz.parameters(), self.prior_params.parameters(
            ), self.posterior_params.parameters()
        ])

    def prior(self):
        return D.Normal(self.prior_params['mean'], self.prior_params['logstd'].exp())

    def posterior(self):
        return D.Normal(self.posterior_params['mean'], self.posterior_params['logstd'].exp())

    def kl(self):
        prior = D.Normal(self.prior_params['mean'].detach(),
                         self.prior_params['logstd'].detach().exp())
        return D.kl_divergence(self.posterior(), prior).mean()

    def forward(self, x, L, sample_prior=False):
        if sample_prior:
            dist = self.prior()
        else:
            dist = self.posterior()
        z = dist.rsample((1, L))
        fz = F.linear(z, self.fz.weight.abs())
        x = self.fx(x).unsqueeze_(1) + fz
        return x, z


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


class StochasticConv2d(nn.Module):
    def __init__(self, width, height, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', init_method='normal', activation='relu',
                 init_mean=0.0, init_log_std=0.0, noise_type='full', noise_features=None, freeze_prior_mean=True, freeze_prior_std=False,
                 single_prior_std=False, single_prior_mean=False, use_abs=True):
        super(StochasticConv2d, self).__init__()
        out_width = get_dimension_size_conv(
            width, padding, stride, kernel_size)
        out_height = get_dimension_size_conv(
            height, padding, stride, kernel_size)
        self.fx = Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode, init_method, activation)
        if noise_type == 'full':
            self.fz = Conv2d(1, out_channels, kernel_size, stride, padding,
                             dilation, groups, bias, padding_mode, init_method, activation)
            self.prior_params = nn.ParameterDict({
                'mean': nn.Parameter(torch.full([1] if single_prior_mean else [height, width], init_mean, dtype=torch.float32), requires_grad=not freeze_prior_mean),
                'logstd': nn.Parameter(torch.full([1] if single_prior_std else [height, width], init_log_std, dtype=torch.float32), requires_grad=not freeze_prior_std)
            })
            self.posterior_params = nn.ParameterDict({
                'mean': nn.Parameter(torch.full([1] if single_prior_mean else [height, width], init_mean, dtype=torch.float32), requires_grad=True),
                'logstd': nn.Parameter(torch.full([1] if single_prior_mean else [height, width], init_log_std, dtype=torch.float32), requires_grad=True)
            })
            if use_abs:
                self.__noise_transform = lambda z: F.conv2d(
                    z, self.fz.weight.abs(), None, stride, padding, dilation, groups)
            else:
                self.__noise_transform = lambda z: self.fz(z)
        elif noise_type == 'partial':
            self.prior_params = nn.ParameterDict({
                'mean': nn.Parameter(torch.full([1] if single_prior_mean else [noise_features], init_mean, dtype=torch.float32), requires_grad=not freeze_prior_mean),
                'logstd': nn.Parameter(torch.full([1] if single_prior_mean else [noise_features], init_log_std, dtype=torch.float32), requires_grad=not freeze_prior_std)
            })
            self.posterior_params = nn.ParameterDict({
                'mean': nn.Parameter(torch.full([1] if single_prior_mean else [noise_features], init_mean, dtype=torch.float32), requires_grad=True),
                'logstd': nn.Parameter(torch.full([1] if single_prior_mean else [noise_features], init_log_std, dtype=torch.float32), requires_grad=True)
            })
            self.fz = Linear(noise_features, out_width *
                             out_height, bias, init_method, activation)
            if use_abs:
                self.__noise_transform = lambda z: F.linear(
                    z, self.fz.weight.abs()).reshape((-1, 1, out_height, out_width))
            else:
                self.__noise_transform = lambda z: self.fz(
                    z).reshape((-1, 1, out_height, out_width))
        else:
            raise NotImplementedError(
                "Currently only support full noise channel")

    def parameters(self):
        return self.fx.parameters()

    def stochastic_params(self):
        return chain.from_iterable([
            self.fz.parameters(), self.prior_params.parameters(
            ), self.posterior_params.parameters()
        ])

    def prior(self):
        return D.Normal(self.prior_params['mean'], self.prior_params['logstd'].exp())

    def posterior(self):
        return D.Normal(self.posterior_params['mean'], self.posterior_params['logstd'].exp())

    def kl(self):
        prior = D.Normal(self.prior_params['mean'].detach(),
                         self.prior_params['logstd'].detach().exp())
        return D.kl_divergence(self.posterior(), prior).mean()

    def forward(self, x, L, sample_prior=False):
        if sample_prior:
            dist = self.prior()
        else:
            dist = self.posterior()
        z = dist.rsample((L, 1))  # [L, 1, H, W] or [L, 1, n_z]
        fz = self.__noise_transform(z)
        x = self.fx(x).unsqueeze_(1) + fz
        return x, z
