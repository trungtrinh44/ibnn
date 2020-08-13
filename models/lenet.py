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
        x = F.dropout(x, p=self.dropout, training=True)
        x = conv1 = self.act1(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = conv2 = self.act2(x)
        x = F.max_pool2d(x, 2)

        x = x.reshape((bs, -1))
        x = self.fc1(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = fc1 = self.act3(x)
        x = self.fc2(x)
        x = fc2 = F.log_softmax(x, dim=-1)
        if return_conv:
            return x, conv1, conv2, fc1, fc2
        return x

    def forward(self, x, L=1, return_conv=False):
        if L <= 1:
            return self.__one_pass(x, return_conv)
        else:
            if return_conv:
                outs, c1s, c2s, f1s, f2s = [], [], [], [], []
                for _ in range(L):
                    o, c1, c2, f1, f2 = self.__one_pass(x, return_conv)
                    outs.append(o.unsqueeze(1))
                    c1s.append(c1.unsqueeze(1))
                    c2s.append(c2.unsqueeze(1))
                    f1s.append(f1.unsqueeze(1))
                    f2s.append(f2.unsqueeze(1))
                return torch.cat(outs, dim=1), torch.cat(c1s, dim=1), torch.cat(c2s, dim=1), torch.cat(f1s, dim=1), torch.cat(f2s, dim=1)
            result = [
                self.__one_pass(x).unsqueeze(1) for _ in range(L)
            ]
            return torch.cat(result, dim=1)

    def negative_loglikelihood(self, x, y, L, return_prob=False):
        y_pred = self.forward(x, L)
        y_target = y.unsqueeze(1).repeat(1, L)
        logp = D.Categorical(logits=y_pred).log_prob(y_target)
        logp = torch.logsumexp(
            logp, dim=1) - torch.log(torch.tensor(L, dtype=torch.float32, device=logp.device))
        if return_prob:
            return -logp.mean(), y_pred
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
                 posterior_p=0.5, posterior_std=1.0, prior_mean=0.0, prior_std=1.0, train_posterior_std=False,
                 posterior_type='mixture_gaussian', **kargs):
        if posterior_type == 'gaussian':
            def wrapper(layer): return GaussianWrapper(
                layer, prior_mean, prior_std, posterior_std, train_posterior_std)
        else:
            def wrapper(layer): return MixtureGaussianWrapper(layer, prior_mean=prior_mean, prior_std=prior_std, posterior_p=posterior_p,
                                                              posterior_std=posterior_std, train_posterior_std=train_posterior_std)
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
        if L <= 1:
            return self.__one_pass(x, return_conv)
        else:
            if return_conv:
                outs, c1s, c2s, f1s, f2s = [], [], [], [], []
                for _ in range(L):
                    o, c1, c2, f1, f2 = self.__one_pass(x, return_conv)
                    outs.append(o.unsqueeze(1))
                    c1s.append(c1.unsqueeze(1))
                    c2s.append(c2.unsqueeze(1))
                    f1s.append(f1.unsqueeze(1))
                    f2s.append(f2.unsqueeze(1))
                return torch.cat(outs, dim=1), torch.cat(c1s, dim=1), torch.cat(c2s, dim=1), torch.cat(f1s, dim=1), torch.cat(f2s, dim=1)
            result = [
                self.__one_pass(x).unsqueeze(1) for _ in range(L)
            ]
            return torch.cat(result, dim=1)

    def negative_loglikelihood(self, x, y, L, return_prob=False):
        y_pred = self.forward(x, L)
        y_target = y.unsqueeze(1).repeat(1, L)
        logp = D.Categorical(logits=y_pred).log_prob(y_target)
        logp = torch.logsumexp(
            logp, dim=1) - torch.log(torch.tensor(L, dtype=torch.float32, device=logp.device))
        if return_prob:
            return -logp.mean(), y_pred
        return -logp.mean()

    def vb_loss(self, x, y, L):
        y_pred = self.forward(x)
        logp = D.Categorical(logits=y_pred).log_prob(y)
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
                                train_posterior_std=config.get('train_posterior_std', False), posterior_type=config.get('posterior_type', 'mixture_gaussian'))
    return model