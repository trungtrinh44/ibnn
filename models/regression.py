import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from .utils import *
from itertools import chain


class RegressionMLP(nn.Module):
    def __init__(self, n_input, n_output, n_hiddens, n_z, activation='relu', init_mean=0.0, init_log_std=np.log(0.1),
                 init_method='normal', freeze_prior_mean=True, freeze_prior_std=False):
        super(RegressionMLP, self).__init__()
        self.first = StochasticLinear(
            n_input, n_z, n_hiddens[0], True, init_mean, init_log_std,
            init_method, activation, freeze_prior_mean, freeze_prior_std)
        self.act = get_activation(activation)
        self.layers = nn.Sequential(
            *(nn.Sequential(Linear(isize, osize, True, init_method, activation), get_activation(activation))
              for isize, osize in zip(n_hiddens[:-1], n_hiddens[1:])),
            Linear(n_hiddens[-1], n_output, True, init_method, 'linear')
        )
        self.likelihood_logstd = nn.Parameter(
            torch.zeros(()), requires_grad=False)

    def prior(self):
        return self.first.prior()

    def posterior(self):
        return self.first.posterior()

    def kl(self):
        return self.first.kl()

    def __logsample(self, x, y, L, sample_prior):
        y_pred = self.forward(x, L, sample_prior)
        y_target = y.unsqueeze(1).repeat(1, L)
        return D.Normal(y_pred, self.likelihood_logstd.exp()).log_prob(y_target)

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
        y_pred, z = self.forward(x, L, False, True)
        y_target = y.unsqueeze(1).repeat(1, L)
        logp = D.Normal(y_pred, self.likelihood_logstd.exp()
                        ).log_prob(y_target)
        return -logp.mean(), self.kl()

    def forward(self, x, L=1, sample_prior=False, return_noise=False):
        x, z = self.first(x, L, sample_prior)
        x = self.act(x)
        x = self.layers(x)
        if return_noise:
            return x.squeeze_(-1), z
        return x.squeeze_(-1)
