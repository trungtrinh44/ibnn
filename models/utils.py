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


class StochasticLinear(nn.Module):
    def __init__(self, in_features: int, noise_features: int, out_features: int, bias: bool = True,
                 init_prior_mean=0.0, init_prior_std=0.0, init_method='normal', activation='relu',
                 freeze_posterior_mean=True, freeze_prior_std=False):
        super(StochasticLinear, self).__init__()
        self.fx = Linear(in_features, out_features, bias,
                         init_method, activation)
        self.fz = Linear(noise_features, out_features,
                         False, init_method, activation)
        self.prior_params = nn.ParameterDict({
            'mean': nn.Parameter(torch.full((noise_features,), init_prior_mean, dtype=torch.float32), requires_grad=not freeze_posterior_mean),
            'std': nn.Parameter(torch.full((noise_features,), init_prior_std, dtype=torch.float32), requires_grad=not freeze_prior_std)
        })
        self.posterior_params = nn.ParameterDict({
            'mean': nn.Parameter(torch.full((noise_features,), init_prior_mean, dtype=torch.float32), requires_grad=True),
            'std': nn.Parameter(torch.full((noise_features,), init_prior_std, dtype=torch.float32), requires_grad=True)
        })

    def parameters(self):
        return self.fx.parameters()

    def stochastic_params(self):
        return chain.from_iterable([
            self.fz.parameters(), self.prior_params.parameters(
            ), self.posterior_params.parameters()
        ])

    def prior(self):
        return D.Normal(self.prior_params['mean'], self.prior_params['std'].exp())

    def posterior(self):
        return D.Normal(self.posterior_params['mean'], self.posterior_params['std'].exp())

    def kl(self):
        prior = D.Normal(self.prior_params['mean'].detach(),
                         self.prior_params['std'].detach().exp())
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
                 prior_mean=0.0, prior_std=1.0, posterior_p=0.5, posterior_std=1.0):
        super(StochasticConv2d, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                           dilation, groups, bias, padding_mode, init_method, activation)
        self.posterior_params = nn.ParameterDict({
            'p': nn.Parameter(torch.tensor([posterior_p, 1-posterior_p]), requires_grad=False),
            'std': nn.Parameter(torch.tensor(posterior_std), requires_grad=False)
        })
        self.prior_params = nn.ParameterDict({
            'mean': nn.Parameter(torch.tensor(prior_mean), requires_grad=False),
            'std': nn.Parameter(torch.tensor(prior_std), requires_grad=False)
        })

    def draw_sample_from_x(self, x):
        means = torch.cat([
            torch.zeros_like(x).unsqueeze_(-1),
            x.unsqueeze(-1)
        ], dim=-1)
        zero_mask = x == 0.0
        x[zero_mask] = x[zero_mask] + 1e-8
        normal = D.Normal(means, self.posterior_params['std'].data*x.unsqueeze(-1))
        categorical = D.OneHotCategorical(probs=self.posterior_params['p'].data)
        
        p_sample = categorical.sample(x.shape)
        x_sample = (p_sample*normal.rsample()).sum(dim=-1)
        
        return x_sample
    
    def kl(self, n_sample):
        # Monte Carlo approximation for the weights KL
        x = self.conv.weight
        means = torch.cat([
            torch.zeros_like(x).unsqueeze_(-1),
            x.unsqueeze(-1)
        ], dim=-1)
        zero_mask = x == 0.0
        x[zero_mask] = x[zero_mask] + 1e-8
        normal = D.Normal(means, self.posterior_params['std'].data*x.unsqueeze(-1))
        categorical = D.OneHotCategorical(probs=self.posterior_params['p'].data)
        
        p_sample = categorical.sample((n_sample,) + x.shape)
        x_sample = (p_sample*normal.rsample((n_sample,))).sum(dim=-1)

        posterior_log_prob = torch.logsumexp(normal.log_prob(x_sample.unsqueeze(-1)) + self.posterior_params['p'].data.log(), -1)
        prior_log_prob = self.prior().log_prob(x_sample)
        logdiff = prior_log_prob - posterior_log_prob
        return logdiff.mean(dim=0).sum()

    def prior(self):
        return D.Normal(self.prior_params['mean'], self.prior_params['std'])
    
    def forward(self, x):
        x_sample = self.draw_sample_from_x(x)
        output = self.conv(x_sample)
        return output
