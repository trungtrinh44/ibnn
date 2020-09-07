import torch
import torch.distributions as D
import torch.nn as nn

class MixtureRsample(nn.Module):
    def __init__(self, location, scale, mixture_coefficient, distribution='normal', train_location=False, train_scale=False):
        super(MixtureRsample, self).__init__()
        self.location = nn.Parameter(torch.tensor(location), requires_grad=train_location)
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=train_scale)
        self.mixture_coefficient = nn.Parameter(torch.tensor(mixture_coefficient), requires_grad=False)
        if distribution == 'normal':
            self.dist = D.Normal
        elif distribution == 'laplace':
            self.dist = D.Laplace

    def rsample(self, n_sample):
        components = self.dist(self.location, self.scale)
        mixtures = D.Categorical(probs=self.mixture_coefficient)
        ms = mixtures.sample(n_sample).unsqueeze_(-1)
        cs = components.rsample(n_sample)
        sample = torch.gather(cs, -1, ms).squeeze(-1)
        return sample

    def log_prob(self, samples):
        components = self.dist(self.location, self.scale)
        log_prob = torch.logsumexp(components.log_prob(samples.unsqueeze(-1)) + self.mixture_coefficient.log(), dim=-1)
        return log_prob