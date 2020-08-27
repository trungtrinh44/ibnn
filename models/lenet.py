import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from .utils import *
from itertools import chain


class DetDropoutLeNet(nn.Module):
    def __init__(self, width, height, in_channel, n_channels, n_hidden, n_output=10, init_method=False, activation='relu', dropout=0.5):
        super(DetDropoutLeNet, self).__init__()
        self.conv1 = Conv2d(in_channel, n_channels[0], kernel_size=5,
                            init_method=init_method, activation=activation)
        self.act1 = get_activation(activation)
        width = get_dimension_size_conv(width, 0, 1, 5)
        height = get_dimension_size_conv(height, 0, 1, 5)
        self.maxpool1 = nn.MaxPool2d(2)
        width = get_dimension_size_conv(width, 0, 2, 2)
        height = get_dimension_size_conv(height, 0, 2, 2)
        self.conv2 = Conv2d(n_channels[0], n_channels[1], kernel_size=5,
                            init_method=init_method, activation=activation)
        self.act2 = get_activation(activation)
        width = get_dimension_size_conv(width, 0, 1, 5)
        height = get_dimension_size_conv(height, 0, 1, 5)
        self.maxpool2 = nn.MaxPool2d(2)
        width = get_dimension_size_conv(width, 0, 2, 2)
        height = get_dimension_size_conv(height, 0, 2, 2)
        self.fc1 = Linear(n_channels[1]*width*height, n_hidden,
                          init_method=init_method, activation=activation)
        self.act3 = get_activation(activation)
        self.fc2 = Linear(n_hidden, n_output,
                          init_method=init_method, activation='linear')
        self.dropout = nn.Parameter(torch.tensor(dropout), requires_grad=False)

    def forward(self, x):
        bs = x.size(0)
        x = conv1 = self.conv1(x)
        x = F.dropout(x, p=self.dropout, training=False)
        x = self.act1(x)
        x = F.max_pool2d(x, 2)

        x = conv2 = self.conv2(x)
        x = F.dropout(x, p=self.dropout, training=False)
        x = self.act2(x)
        x = F.max_pool2d(x, 2)

        x = x.reshape((bs, -1))
        x = fc1 = self.fc1(x)
        x = F.dropout(x, p=self.dropout, training=False)
        x = self.act3(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class DropoutLeNet(nn.Module):
    def __init__(self, width, height, in_channel, n_channels, n_hidden, n_output=10, init_method=False, activation='relu', dropout=0.5):
        super(DropoutLeNet, self).__init__()
        self.conv1 = Conv2d(in_channel, n_channels[0], kernel_size=5,
                            init_method=init_method, activation=activation)
        self.act1 = get_activation(activation)
        width = get_dimension_size_conv(width, 0, 1, 5)
        height = get_dimension_size_conv(height, 0, 1, 5)
        self.maxpool1 = nn.MaxPool2d(2)
        width = get_dimension_size_conv(width, 0, 2, 2)
        height = get_dimension_size_conv(height, 0, 2, 2)
        self.conv2 = Conv2d(n_channels[0], n_channels[1], kernel_size=5,
                            init_method=init_method, activation=activation)
        self.act2 = get_activation(activation)
        width = get_dimension_size_conv(width, 0, 1, 5)
        height = get_dimension_size_conv(height, 0, 1, 5)
        self.maxpool2 = nn.MaxPool2d(2)
        width = get_dimension_size_conv(width, 0, 2, 2)
        height = get_dimension_size_conv(height, 0, 2, 2)
        self.fc1 = Linear(n_channels[1]*width*height, n_hidden,
                          init_method=init_method, activation=activation)
        self.act3 = get_activation(activation)
        self.fc2 = Linear(n_hidden, n_output,
                          init_method=init_method, activation='linear')
        self.dropout = nn.Parameter(torch.tensor(dropout), requires_grad=False)

    def __one_pass(self, x, return_conv=False):
        bs = x.size(0)
        x = self.conv1(x)
        x = conv1 = self.act1(x)
        x = F.max_pool2d(x, 2)

        x = F.dropout(x, p=self.dropout, training=True)
        x = self.conv2(x)
        x = conv2 = self.act2(x)
        x = F.max_pool2d(x, 2)

        x = x.reshape((bs, -1))
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.fc1(x)
        x = fc1 = self.act3(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.fc2(x)
        x = fc2 = F.log_softmax(x, dim=-1)
        if return_conv:
            return x, conv1, conv2, fc1, fc2
        return x

    def forward(self, x, L=1, return_conv=False):
        x = torch.repeat_interleave(x, repeats=L, dim=0)
        if return_conv:
            x, c1, c2, f1, f2 = self.__one_pass(x, True)
            x = x.reshape((-1, L) + x.shape[1:])
            c1 = c1.reshape((-1, L) + c1.shape[1:])
            c2 = c2.reshape((-1, L) + c2.shape[1:])
            f1 = f1.reshape((-1, L) + f1.shape[1:])
            f2 = f2.reshape((-1, L) + f2.shape[1:])
            return x, c1, c2, f1, f2
        else:
            x = self.__one_pass(x, False)
            x = x.reshape((-1, L) + x.shape[1:])
            return x

    def negative_loglikelihood(self, x, y, L, return_prob=False):
        y_pred = self.forward(x, L)
        y_target = y.unsqueeze(1).repeat(1, L)
        logp = D.Categorical(logits=y_pred).log_prob(y_target)
        logp = torch.logsumexp(
            logp, dim=1) - torch.log(torch.tensor(L, dtype=torch.float32, device=logp.device))
        if return_prob:
            return -logp.mean(), y_pred
        return -logp.mean()
    
    def train_loss(self, x, y, L):
        y_pred = self.forward(x, L)
        y_target = y.unsqueeze(1).repeat(1, L)
        logp = D.Categorical(logits=y_pred).log_prob(y_target)
        return -logp.mean()


class DeterministicLeNet(nn.Module):
    def __init__(self, width, height, in_channel, n_channels, n_hidden, n_output=10, init_method=False, activation='relu'):
        super(DeterministicLeNet, self).__init__()
        self.conv1 = Conv2d(in_channel, n_channels[0], kernel_size=5,
                            init_method=init_method, activation=activation)
        self.act1 = get_activation(activation)
        width = get_dimension_size_conv(width, 0, 1, 5)
        height = get_dimension_size_conv(height, 0, 1, 5)
        self.maxpool1 = nn.MaxPool2d(2)
        width = get_dimension_size_conv(width, 0, 2, 2)
        height = get_dimension_size_conv(height, 0, 2, 2)
        self.conv2 = Conv2d(n_channels[0], n_channels[1], kernel_size=5,
                            init_method=init_method, activation=activation)
        self.act2 = get_activation(activation)
        width = get_dimension_size_conv(width, 0, 1, 5)
        height = get_dimension_size_conv(height, 0, 1, 5)
        self.maxpool2 = nn.MaxPool2d(2)
        width = get_dimension_size_conv(width, 0, 2, 2)
        height = get_dimension_size_conv(height, 0, 2, 2)
        self.fc1 = Linear(n_channels[1]*width*height, n_hidden,
                          init_method=init_method, activation=activation)
        self.act3 = get_activation(activation)
        self.fc2 = Linear(n_hidden, n_output,
                          init_method=init_method, activation='linear')

    def forward(self, x, return_fc1=False):
        bs = x.size(0)
        x = self.conv1(x)
        x = self.act1(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = self.act2(x)
        x = F.max_pool2d(x, 2)

        x = x.reshape((bs, -1))
        x = fc1 = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        if return_fc1:
            return F.log_softmax(x, dim=-1), fc1
        return F.log_softmax(x, dim=-1)


class StochasticLeNet(nn.Module):
    def __init__(self, width, height, in_channel, n_channels, n_hidden, n_output=10, init_method='normal', activation='relu',
                 posterior_p=0.5, posterior_std=1.0, prior_mean=0.0, prior_std=1.0, train_posterior_std=False, posterior_mean=[0.0, 1.0], train_posterior_mean=False,
                 posterior_type='mixture_gaussian', **kargs):
        if posterior_type == 'gaussian':
            def wrapper(layer): return GaussianWrapper(layer, prior_mean, prior_std, posterior_mean, posterior_std, train_posterior_std, train_posterior_mean)
        elif posterior_type == 'laplace':
            def wrapper(layer): return LaplaceWrapper(layer, prior_mean, prior_std, posterior_mean, posterior_std, train_posterior_std, train_posterior_mean)
        elif posterior_type == 'mixture_gaussian':
            def wrapper(layer): return MixtureGaussianWrapper(layer, prior_mean=prior_mean, prior_std=prior_std, posterior_p=posterior_p,
                                                              posterior_std=posterior_std, train_posterior_std=train_posterior_std,
                                                              posterior_mean=posterior_mean, train_posterior_mean=train_posterior_mean)
        elif posterior_type == 'mixture_laplace':
            def wrapper(layer): return MixtureLaplaceWrapper(layer, prior_mean=prior_mean, prior_std=prior_std, posterior_p=posterior_p,
                                                              posterior_std=posterior_std, train_posterior_std=train_posterior_std,
                                                              posterior_mean=posterior_mean, train_posterior_mean=train_posterior_mean)
        super(StochasticLeNet, self).__init__()
        self.conv1 = Conv2d(in_channel, n_channels[0], kernel_size=5,
                            init_method=init_method, activation=activation)
        self.act1 = get_activation(activation)
        width = get_dimension_size_conv(width, 0, 1, 5)
        height = get_dimension_size_conv(height, 0, 1, 5)
        self.maxpool1 = nn.MaxPool2d(2)
        width = get_dimension_size_conv(width, 0, 2, 2)
        height = get_dimension_size_conv(height, 0, 2, 2)
        self.conv2 = wrapper(Conv2d(n_channels[0], n_channels[1], kernel_size=5,
                                    init_method=init_method, activation=activation))
        self.act2 = get_activation(activation)
        width = get_dimension_size_conv(width, 0, 1, 5)
        height = get_dimension_size_conv(height, 0, 1, 5)
        self.maxpool2 = nn.MaxPool2d(2)
        width = get_dimension_size_conv(width, 0, 2, 2)
        height = get_dimension_size_conv(height, 0, 2, 2)
        self.fc1 = wrapper(Linear(n_channels[1]*width*height, n_hidden,
                                  init_method=init_method, activation=activation))
        self.act3 = get_activation(activation)
        self.fc2 = wrapper(Linear(n_hidden, n_output,
                                  init_method=init_method, activation='linear'))

    def __one_pass(self, x, return_conv=False):
        bs = x.size(0)
        x = self.conv1(x)
        x = conv1 = self.act1(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = conv2 = self.act2(x)
        x = F.max_pool2d(x, 2)

        x = x.reshape((bs, -1))
        x = self.fc1(x)
        x = fc1 = self.act3(x)
        x = self.fc2(x)
        x = fc2 = F.log_softmax(x, dim=-1)
        if return_conv:
            return x, conv1, conv2, fc1, fc2
        return x

    def forward(self, x, L=1, return_conv=False):
        x = torch.repeat_interleave(x, repeats=L, dim=0)
        if return_conv:
            x, c1, c2, f1, f2 = self.__one_pass(x, True)
            x = x.reshape((-1, L) + x.shape[1:])
            c1 = c1.reshape((-1, L) + c1.shape[1:])
            c2 = c2.reshape((-1, L) + c2.shape[1:])
            f1 = f1.reshape((-1, L) + f1.shape[1:])
            f2 = f2.reshape((-1, L) + f2.shape[1:])
            return x, c1, c2, f1, f2
        else:
            x = self.__one_pass(x, False)
            x = x.reshape((-1, L) + x.shape[1:])
            return x

    def negative_loglikelihood(self, x, y, L, return_prob=False):
        y_pred = self.forward(x, L)
        y_target = y.unsqueeze(1).repeat(1, L)
        logp = D.Categorical(logits=y_pred).log_prob(y_target)
        logp = torch.logsumexp(
            logp, dim=1) - torch.log(torch.tensor(L, dtype=torch.float32, device=logp.device))
        if return_prob:
            return -logp.mean(), y_pred
        return -logp.mean()

    def vb_loss(self, x, y, L, no_kl=False):
        y_pred = self.forward(x, L)
        y_target = y.unsqueeze(1).repeat(1, L)
        logp = D.Categorical(logits=y_pred).log_prob(y_target)
        if no_kl:
            return -logp.mean(), torch.zeros(())
        kl = self.conv2.kl(L) + self.fc1.kl(L) + self.fc2.kl(L)
        return -logp.mean(), kl

def get_model_from_config(config, width, height, in_channels, n_classes):
    if config['model_type'] == 'deterministic':
        model = DeterministicLeNet(width, height, in_channels,
                                   config['conv_hiddens'], config['fc_hidden'], n_classes, config['init_method'], config['activation'])
    elif config['model_type'] == 'dropout':
        model = DropoutLeNet(width, height, in_channels,
                             config['conv_hiddens'], config['fc_hidden'], n_classes, config['init_method'], config['activation'], config['dropout'])
    else:
        model = StochasticLeNet(width, height, in_channels, config['conv_hiddens'], config['fc_hidden'], n_classes, config['init_method'], config['activation'],
                                config['posterior_p'], config['posterior_std'], config['init_prior_mean'], config['init_prior_std'],
                                train_posterior_std=config.get('train_posterior_std', False), posterior_type=config.get('posterior_type', 'mixture_gaussian'),
                                train_posterior_mean=config.get('train_posterior_mean', False), posterior_mean=config.get('posterior_mean', [0.0, 1.0]))
    return model