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



class StoLayer(nn.Module):
    def __init__(self, in_features, n_components, prior_mean, prior_std, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02)):
        super(StoLayer, self).__init__()
        posterior_mean = torch.ones((n_components, *in_features))
        posterior_std = torch.ones((n_components, *in_features))
        # [1, In, 1, 1]
        self.posterior_mean = nn.Parameter(posterior_mean, requires_grad=True)
        self.posterior_std = nn.Parameter(posterior_std, requires_grad=True)
        nn.init.normal_(self.posterior_std, posterior_std_init[0], posterior_std_init[1])
        nn.init.normal_(self.posterior_mean, posterior_mean_init[0], posterior_mean_init[1])
        self.posterior_std.data.abs_().expm1_().log_()
        self.prior_mean = nn.Parameter(torch.tensor(prior_mean), requires_grad=False)
        self.prior_std = nn.Parameter(torch.tensor(prior_std), requires_grad=False)
        self.posterior_mean_init = posterior_mean_init
        self.posterior_std_init = posterior_std_init
    
    def get_input_sample(self, input, indices):
        mean = self.posterior_mean
        std = F.softplus(self.posterior_std)
        components = D.Normal(mean[indices], std[indices])
        return components.rsample()

    def forward(self, x, indices):
        x = x * self.get_input_sample(x, indices)
        return x
    
    def kl(self):
        mean = self.posterior_mean.mean(dim=0)
        std = F.softplus(self.posterior_std).pow(2.0).sum(0).pow(0.5) / self.posterior_std.size(0)
        components = D.Normal(mean, std)
        prior = D.Normal(self.prior_mean, self.prior_std)
        return D.kl_divergence(components, prior).sum()
    
    def extra_repr(self):
        return f"n_components={self.posterior_mean.size(0)}, prior_mean={self.prior_mean.data.item()}, prior_std={self.prior_std.data.item()}, posterior_mean={self.posterior_mean_init}, posterior_std={self.posterior_std_init}"

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
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        prior_mean = 0.0,
        prior_std = 1.0,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'
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
