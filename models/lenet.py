import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from .utils import *


class DeterministicLeNet(nn.Module):
    def __init__(self, width, height, in_channel, n_channels, n_hidden, n_output=10, orthogonal_init=False, activation='relu'):
        super(DeterministicLeNet, self).__init__()
        self.conv1 = Conv2d(in_channel, n_channels[0], kernel_size=5,
                            orthogonal_init=orthogonal_init, activation=activation)
        self.act1 = get_activation(activation)
        width = get_dimension_size_conv(width, 0, 1, 5)
        height = get_dimension_size_conv(height, 0, 1, 5)
        self.maxpool1 = nn.MaxPool2d(2)
        width = get_dimension_size_conv(width, 0, 2, 2)
        height = get_dimension_size_conv(height, 0, 2, 2)
        self.conv2 = Conv2d(n_channels[0], n_channels[1], kernel_size=5,
                            orthogonal_init=orthogonal_init, activation=activation)
        self.act2 = get_activation(activation)
        width = get_dimension_size_conv(width, 0, 1, 5)
        height = get_dimension_size_conv(height, 0, 1, 5)
        self.maxpool2 = nn.MaxPool2d(2)
        width = get_dimension_size_conv(width, 0, 2, 2)
        height = get_dimension_size_conv(height, 0, 2, 2)
        self.fc1 = Linear(n_channels[1]*width*height, n_hidden,
                          orthogonal_init=orthogonal_init, activation=activation)
        self.act3 = get_activation(activation)
        self.fc2 = Linear(n_hidden, n_output,
                          orthogonal_init=orthogonal_init, activation='linear')

    def forward(self, x):
        bs = x.size(0)
        x = self.conv1(x)
        x = self.act1(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = self.act2(x)
        x = F.max_pool2d(x, 2)

        x = x.reshape((bs, -1))
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class StochasticLeNet(nn.Module):
    def __init__(self, width, height, in_channel, n_channels, n_hidden, n_output=10, orthogonal_init=False, activation='relu',
                 init_mean=0.0, init_log_std=np.log(0.1), p=3/4, noise_type='full', noise_features=None):
        super(StochasticLeNet, self).__init__()
        self.conv1 = StochasticConv2d(width, height, in_channel, n_channels[0], kernel_size=5,
                                      orthogonal_init=orthogonal_init, activation=activation,
                                      init_mean=0.0, init_log_std=0.0, p=3/4, noise_type='full', noise_features=None)
        self.act1 = get_activation(activation)
        width = get_dimension_size_conv(width, 0, 1, 5)
        height = get_dimension_size_conv(height, 0, 1, 5)
        self.maxpool1 = nn.MaxPool2d(2)
        width = get_dimension_size_conv(width, 0, 2, 2)
        height = get_dimension_size_conv(height, 0, 2, 2)
        self.conv2 = Conv2d(n_channels[0], n_channels[1], kernel_size=5,
                            orthogonal_init=orthogonal_init, activation=activation)
        self.act2 = get_activation(activation)
        width = get_dimension_size_conv(width, 0, 1, 5)
        height = get_dimension_size_conv(height, 0, 1, 5)
        self.maxpool2 = nn.MaxPool2d(2)
        width = get_dimension_size_conv(width, 0, 2, 2)
        height = get_dimension_size_conv(height, 0, 2, 2)
        self.fc1 = Linear(n_channels[1]*width*height, n_hidden,
                          orthogonal_init=orthogonal_init, activation=activation)
        self.act3 = get_activation(activation)
        self.fc2 = Linear(n_hidden, n_output,
                          orthogonal_init=orthogonal_init, activation='linear')

    def prior(self):
        return self.conv1.prior()

    def posterior(self):
        return self.conv1.posterior()

    def kl(self):
        return self.conv1.kl()

    def __logsample(self, x, y, L, sample_prior):
        y_pred = self.forward(x, L, sample_prior)
        y_target = y.unsqueeze(1).repeat(1, L)
        return D.Categorical(logits=y_pred).log_prob(y_target)

    def __loglikelihood(self, x, y, L, sample_prior):
        logp = self.__logsample(x, y, L, sample_prior)
        logp = torch.logsumexp(
            logp, dim=1) - torch.log(torch.tensor(L, dtype=torch.float32, device=logp.device))
        return -logp.mean()

    def marginal_loglikelihood_loss(self, x, y, L):
        return self.__loglikelihood(x, y, L, True)

    def negative_loglikelihood(self, x, y, L):
        return self.__loglikelihood(x, y, L, False)

    def vb_loss(self, x, y, L):
        logp = self.__logsample(x, y, L, False)
        return -logp.mean(), self.kl(), self.conv1.weight_norm()

    def forward(self, x, L, sample_prior=False):
        bs = x.size(0)
        x = self.conv1(x, L, sample_prior)
        x = self.act1(x)
        x = x.reshape(bs*L, x.size(2), x.size(3), x.size(4))
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = self.act2(x)
        x = F.max_pool2d(x, 2)

        x = x.reshape((bs, L, -1))
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)