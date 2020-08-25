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
    def __init__(self, layer, prior_mean=0.0, prior_std=1.0, posterior_p=0.5, posterior_std=1.0, train_posterior_std=False, posterior_mean=[0.0, 1.0], train_posterior_mean=False):
        super(MixtureGaussianWrapper, self).__init__()
        self.layer = layer
        self.posterior_params = nn.ParameterDict({
            'p': nn.Parameter(torch.tensor([posterior_p, 1-posterior_p]), requires_grad=False),
            'std': nn.Parameter(torch.tensor(posterior_std), requires_grad=train_posterior_std),
            'mean': nn.Parameter(torch.tensor(posterior_mean), requires_grad=train_posterior_mean)
        })
        self.prior_params = nn.ParameterDict({
            'mean': nn.Parameter(torch.tensor(prior_mean), requires_grad=False),
            'std': nn.Parameter(torch.tensor(prior_std), requires_grad=False)
        })

    def kl(self, n_sample):
        # Monte Carlo approximation for the weights KL
        sample_shape = (n_sample, ) + self.layer.weight.shape
        sample = self.sample_noise(sample_shape)
        weight_sample = self.layer.weight.unsqueeze(0) * sample
        weight_unsqueeze = self.layer.weight.unsqueeze(-1)
        weight_mean = weight_unsqueeze * self.posterior_params['mean']
        weight_std = torch.max((weight_unsqueeze * self.posterior_params['std']).abs(), torch.tensor(1e-9, device=self.layer.weight.device))
        components = D.Normal(weight_mean, weight_std)
        prior = D.Normal(self.prior_params['mean'], self.prior_params['std'])
        posterior_log_prob = torch.logsumexp(components.log_prob(weight_sample.unsqueeze(-1)) + self.posterior_params['p'].log(), -1)
        prior_log_prob = prior.log_prob(weight_sample)
        logdiff = (posterior_log_prob - prior_log_prob).mean(dim=0)
        return logdiff.sum()

    def sample_noise(self, noise_shape):
        normal = D.Normal(self.posterior_params['mean'], self.posterior_params['std'])
        categorical = D.OneHotCategorical(probs=self.posterior_params['p'])
        sample = (categorical.sample(noise_shape) * normal.rsample(noise_shape)).sum(dim=-1)
        return sample

    def forward(self, x):
        sample = self.sample_noise(x.shape)
        mask = (x != 0.0).float()
        sample = sample * mask
        output = self.layer(x * sample)
        return output

class MixtureLaplaceWrapper(nn.Module):
    def __init__(self, layer, prior_mean=0.0, prior_std=1.0, posterior_p=0.5, posterior_std=1.0, train_posterior_std=False, posterior_mean=[0.0, 1.0], train_posterior_mean=False):
        super(MixtureLaplaceWrapper, self).__init__()
        self.layer = layer
        self.posterior_params = nn.ParameterDict({
            'p': nn.Parameter(torch.tensor([posterior_p, 1-posterior_p]), requires_grad=False),
            'std': nn.Parameter(torch.tensor(posterior_std), requires_grad=train_posterior_std),
            'mean': nn.Parameter(torch.tensor(posterior_mean), requires_grad=train_posterior_mean)
        })
        self.prior_params = nn.ParameterDict({
            'mean': nn.Parameter(torch.tensor(prior_mean), requires_grad=False),
            'std': nn.Parameter(torch.tensor(prior_std), requires_grad=False)
        })

    def kl(self, n_sample):
        # Monte Carlo approximation for the weights KL
        sample_shape = (n_sample, ) + self.layer.weight.shape
        sample = self.sample_noise(sample_shape)
        weight_sample = self.layer.weight.unsqueeze(0) * sample
        weight_unsqueeze = self.layer.weight.unsqueeze(-1)
        weight_mean = weight_unsqueeze * self.posterior_params['mean']
        weight_std = torch.max((weight_unsqueeze * self.posterior_params['std']).abs(), torch.tensor(1e-9, device=self.layer.weight.device))
        components = D.Laplace(weight_mean, weight_std)
        prior = D.Laplace(self.prior_params['mean'], self.prior_params['std'])
        posterior_log_prob = torch.logsumexp(components.log_prob(weight_sample.unsqueeze(-1)) + self.posterior_params['p'].log(), -1)
        prior_log_prob = prior.log_prob(weight_sample)
        logdiff = (posterior_log_prob - prior_log_prob).mean(dim=0)
        return logdiff.sum()

    def sample_noise(self, noise_shape):
        laplace = D.Laplace(self.posterior_params['mean'], self.posterior_params['std'])
        categorical = D.OneHotCategorical(probs=self.posterior_params['p'])
        sample = (categorical.sample(noise_shape) * laplace.rsample(noise_shape)).sum(dim=-1)
        return sample

    def forward(self, x):
        sample = self.sample_noise(x.shape)
        mask = (x != 0.0).float()
        sample = sample * mask
        output = self.layer(x * sample)
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
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece