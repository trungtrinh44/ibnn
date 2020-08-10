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
        x = conv1 = self.conv1(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.act1(x)
        x = F.max_pool2d(x, 2)

        x = conv2 = self.conv2(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.act2(x)
        x = F.max_pool2d(x, 2)

        x = x.reshape((bs, -1))
        x = fc1 = self.fc1(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.act3(x)
        x = self.fc2(x)
        if return_conv:
            return F.log_softmax(x, dim=-1), conv1, conv2, fc1
        return F.log_softmax(x, dim=-1)

    def forward(self, x, L=1, return_conv=False):
        if L <= 1:
            return self.__one_pass(x, return_conv)
        else:
            if return_conv:
                c1s, c2s, f1s, f2s = [], [], [], []
                for _ in range(L):
                    f2, c1, c2, f1 = self.__one_pass(x, return_conv)
                    c1s.append(c1.unsqueeze_(1))
                    c2s.append(c2.unsqueeze_(1))
                    f1s.append(f1.unsqueeze_(1))
                    f2s.append(f2.unsqueeze_(1))
                return torch.cat(f2s, dim=1), torch.cat(c1s, dim=1), torch.cat(c2s, dim=1), torch.cat(f1s, dim=1)
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
                 init_dist_mean=0.0, init_dist_std=1.0, init_prior_mean=0.0, init_prior_std=1.0, noise_type='full', noise_features=None, use_abs=False):
        super(StochasticLeNet, self).__init__()
        self.conv1 = StochasticConv2d(width, height, in_channel, n_channels[0], kernel_size=5,
                                      init_method=init_method, activation=activation,
                                      init_dist_mean=init_dist_mean, init_dist_std=init_dist_std,
                                      noise_type=noise_type, noise_features=noise_features, use_abs=use_abs)
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
        self.prior_params = nn.ParameterDict({
            'mean': nn.Parameter(torch.full((n_hidden,), init_prior_mean), requires_grad=False),
            'scaletril': nn.Parameter(torch.cholesky(torch.eye(n_hidden)*init_prior_std), requires_grad=False)
        })

    def prior(self):
        return D.MultivariateNormal(self.prior_params['mean'], scale_tril=self.prior_params['scaletril'])

    def negative_loglikelihood(self, x, y, L):
        y_pred = self.forward(x, L, False)
        y_target = y.unsqueeze(1).repeat(1, L)
        logp = D.Categorical(logits=y_pred).log_prob(y_target)
        logp = torch.logsumexp(
            logp, dim=1) - torch.log(torch.tensor(L, dtype=torch.float32, device=logp.device))
        return -logp.mean()

    def vb_loss(self, x, y, L):
        y_pred, z = self.forward(x, L, True)
        y_target = y.unsqueeze(1).repeat(1, L)
        logp = D.Categorical(logits=y_pred).log_prob(y_target)
        z_mean = z.mean(dim=1)

        z_cov = z - z_mean.unsqueeze(1)
        z_cov = (z_cov.transpose(2, 1) @ z_cov) / (L-1)
        z_cov = z_cov + torch.eye(z.size(-1), device=z.device)*1e-5
        
        posterior = D.MultivariateNormal(loc=z_mean, covariance_matrix=z_cov)
        kl = D.kl_divergence(posterior, self.prior()).mean()
        return -logp.mean(), kl

    def forward(self, x, L, return_noise=False, return_conv=False):
        bs = x.size(0)
        x, z = self.conv1(x, L)
        conv1_out = x
        x = self.act1(x)
        x = x.reshape(bs*L, x.size(2), x.size(3), x.size(4))
        x = F.max_pool2d(x, 2)

        x = conv2_out = self.conv2(x)
        x = self.act2(x)
        x = F.max_pool2d(x, 2)

        x = x.reshape((bs, L, -1))
        x = z = fc1_out = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        if return_noise:
            return F.log_softmax(x, dim=-1), z
        if return_conv:
            return F.log_softmax(x, dim=-1), conv1_out, conv2_out.reshape(bs, L, conv2_out.size(1), conv2_out.size(2), conv2_out.size(3)), fc1_out
        return F.log_softmax(x, dim=-1)
