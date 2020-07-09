import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, in_features: int, out_features: int, bias: bool = True, init_method='normal': bool = True, activation='relu'):
        super(Linear, self).__init__(in_features, out_features, bias)
        if init_method=='orthogonal':
            nn.init.orthogonal_(
                self.weight, nn.init.calculate_gain(activation))
            if bias:
                nn.init.constant_(self.bias, 0.0)
        elif init_method=='normal':
            nn.init.normal_(self.weight, mean=0.0, std=0.1)
            if bias:
                nn.init.constant_(self.bias, 0.1)


class StochasticLinear(nn.Module):
    def __init__(self, in_features: int, noise_features: int, out_features: int, bias: bool = True,
                 init_mean=0.0, init_log_std=0.0, p=3/4,
                 init_method='normal': bool = True, activation='relu'):
        super(StochasticLinear, self).__init__()
        self.fx = Linear(in_features, out_features, bias,
                         init_method, activation)
        self.fz = Linear(noise_features, out_features,
                         False, init_method, activation)
        self.prior_params = nn.ParameterDict({
            'mean': nn.Parameter(torch.full((noise_features,), init_mean, dtype=torch.float32), requires_grad=False),
            'logstd': nn.Parameter(torch.full((noise_features,), init_log_std, dtype=torch.float32), requires_grad=True)
        })
        self.posterior_params = nn.ParameterDict({
            'mean': nn.Parameter(torch.full((noise_features,), init_mean, dtype=torch.float32), requires_grad=True),
            'logstd': nn.Parameter(torch.full((noise_features,), init_log_std, dtype=torch.float32), requires_grad=True)
        })
        self._p = torch.tensor(p, dtype=torch.float32)
        self._fzws = np.prod(self.fz.weight.shape)

    def prior(self):
        return D.Normal(self.prior_params['mean'], self.prior_params['logstd'].exp())

    def posterior(self):
        return D.Normal(self.posterior_params['mean'], self.posterior_params['logstd'].exp())

    def kl(self):
        prior = D.Normal(self.prior_params['mean'].detach(), self.prior_params['logstd'].detach().exp())
        return D.kl_divergence(self.posterior(), prior).mean()

    def weight_norm(self):
        return torch.norm(self.fz.weight, p=self._p)/self._fzws

    def forward(self, x, L, sample_prior=False):
        if sample_prior:
            dist = self.prior()
        else:
            dist = self.posterior()
        z = dist.rsample((1, L))
        z = F.linear(z, self.fz.weight.abs())
        x = self.fx(x).unsqueeze_(1) + z
        return x


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', init_method='normal', activation='relu'):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=1,
                                     padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        if init_method=='orthogonal':
            nn.init.orthogonal_(
                self.weight, nn.init.calculate_gain(activation))
            if bias:
                nn.init.constant_(self.bias, 0.0)
        elif init_method=='normal':
            nn.init.normal_(self.weight, mean=0.0, std=0.1)
            if bias:
                nn.init.constant_(self.bias, 0.1)


class StochasticConv2d(nn.Conv2d):
    def __init__(self, width, height, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', init_method='normal', activation='relu',
                 init_mean=0.0, init_log_std=0.0, p=3/4, noise_type='full', noise_features=None):
        super(StochasticConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=1,
                                               padding=0, dilation=1, groups=1,
                                               bias=True, padding_mode='zeros')
        out_width = get_dimension_size_conv(
            width, padding, stride, kernel_size)
        out_height = get_dimension_size_conv(
            height, padding, stride, kernel_size)
        self.fx = Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode, init_method, activation)
        if noise_type == 'full':
            self.fz = Conv2d(1, out_channels, kernel_size, stride, padding,
                             dilation, groups, False, padding_mode, init_method, activation)
            self.prior_params = nn.ParameterDict({
                'mean': nn.Parameter(torch.full([height, width], init_mean, dtype=torch.float32), requires_grad=False),
                'logstd': nn.Parameter(torch.full([height, width], init_log_std, dtype=torch.float32), requires_grad=True)
            })
            self.posterior_params = nn.ParameterDict({
                'mean': nn.Parameter(torch.full([height, width], init_mean, dtype=torch.float32), requires_grad=True),
                'logstd': nn.Parameter(torch.full([height, width], init_log_std, dtype=torch.float32), requires_grad=True)
            })
            self.__noise_transform = lambda z: F.conv2d(z, self.fz.weight.abs(), None, stride, padding, dilation, groups)
        else:
            raise NotImplementedError(
                "Currently only support full noise channel")
        self._p = torch.tensor(p, dtype=torch.float32)
        self._fzws = np.prod(self.fz.weight.shape)
    
    def prior(self):
        return D.Normal(self.prior_params['mean'], self.prior_params['logstd'].exp())

    def posterior(self):
        return D.Normal(self.posterior_params['mean'], self.posterior_params['logstd'].exp())

    def kl(self):
        prior = D.Normal(self.prior_params['mean'].detach(), self.prior_params['logstd'].detach().exp())
        return D.kl_divergence(self.posterior(), prior).mean()
    
    def weight_norm(self):
        return torch.norm(self.fz.weight, p=self._p)/self._fzws

    def forward(self, x, L, sample_prior=False):
        if sample_prior:
            dist = self.prior()
        else:
            dist = self.posterior()
        z = dist.rsample((L, 1)) # [L, 1, H, W]
        z = self.__noise_transform(z)
        x = self.fx(x).unsqueeze_(1) + z
        return x
        