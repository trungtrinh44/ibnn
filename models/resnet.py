import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, stride, init_method):
        super(Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_filters, eps=1e-5, momentum=0.1)
        nn.init.uniform_(self.bn1.weight)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = Conv2d(in_filters, out_filters, 3, stride=stride,
                            padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', init_method=init_method, activation='relu')
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_filters, eps=1e-5, momentum=0.1)
        nn.init.uniform_(self.bn2.weight)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_filters, out_filters, 3, stride=1,
                            padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', init_method=init_method, activation='relu')

        if stride != 1 or in_filters != out_filters:
            self.has_shortcut = True
            self.shortcut = Conv2d(in_filters, out_filters, 1, stride=stride,
                                   padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', init_method=init_method, activation='relu')
        else:
            self.has_shortcut = False

    def forward(self, x):
        o = self.relu1(self.bn1(x))
        out = self.conv1(o)
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        if self.has_shortcut:
            out = out + self.shortcut(o)
        else:
            out = out + x
        return out


class DetWideResNet(nn.Module):
    def __init__(self, size, in_channels, dropout, n_classes=10, n_per_block=4, k=2, init_method='normal'):
        super(DetWideResNet, self).__init__()
        self.conv1 = Conv2d(in_channels, 16, 3, stride=1,
                            padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', init_method=init_method, activation='relu')
        self.conv2 = nn.Sequential(
            Block(16, 16*k, dropout, 1, init_method),
            *(
                Block(16*k, 16*k, dropout, 1, init_method) for _ in range(n_per_block-1)
            )
        )
        self.conv3 = nn.Sequential(
            Block(16*k, 32*k, dropout, 2, init_method),
            *(
                Block(32*k, 32*k, dropout, 1, init_method) for _ in range(n_per_block-1)
            )
        )
        self.conv4 = nn.Sequential(
            Block(32*k, 64*k, dropout, 2, init_method),
            *(
                Block(64*k, 64*k, dropout, 1, init_method) for _ in range(n_per_block-1)
            )
        )
        self.bn = nn.BatchNorm2d(64*k, eps=1e-5, momentum=0.1)
        nn.init.uniform_(self.bn.weight)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(size//4)
        self.fc1 = Linear(64*k, n_classes, bias=True,
                          init_method=init_method, activation='linear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.relu(self.bn(x))
        x = self.avg_pool(x)
        x = self.fc1(x.flatten(start_dim=1, end_dim=-1))
        x = F.log_softmax(x, dim=-1)
        return x


class StoBlock(nn.Module):
    def __init__(self, in_filters, out_filters, sto_wrapper, stride, init_method):
        super(StoBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_filters, eps=1e-5, momentum=0.1)
        nn.init.uniform_(self.bn1.weight)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = Conv2d(in_filters, out_filters, 3, stride=stride,
                            padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', init_method=init_method, activation='relu')
        self.conv1 = sto_wrapper(self.conv1)
        self.bn2 = nn.BatchNorm2d(out_filters, eps=1e-5, momentum=0.1)
        nn.init.uniform_(self.bn2.weight)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_filters, out_filters, 3, stride=1,
                            padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', init_method=init_method, activation='relu')
        self.conv2 = sto_wrapper(self.conv2)
        if stride != 1 or in_filters != out_filters:
            self.has_shortcut = True
            self.shortcut = Conv2d(in_filters, out_filters, 1, stride=stride,
                                   padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', init_method=init_method, activation='relu')
        else:
            self.has_shortcut = False

    def kl(self, n_sample):
        return self.conv1.kl(n_sample) + self.conv2.kl(n_sample)

    def forward(self, x):
        o = self.relu1(self.bn1(x))
        out = self.conv1(o)
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.has_shortcut:
            out = out + self.shortcut(o)
        else:
            out = out + x
        return out


class StoWideResNet(nn.Module):
    def __init__(self, size, in_channels, posterior_type, prior_mean, prior_std, posterior_p, posterior_mean, posterior_std, train_posterior_std, train_posterior_mean,
                 n_classes=10, n_per_block=4, k=2, init_method='normal'):
        if posterior_type == 'gaussian':
            def wrapper(layer): return GaussianWrapper(layer, prior_mean, prior_std,
                                                       posterior_mean, posterior_std, train_posterior_std, train_posterior_mean)
        elif posterior_type == 'laplace':
            def wrapper(layer): return LaplaceWrapper(layer, prior_mean, prior_std,
                                                      posterior_mean, posterior_std, train_posterior_std, train_posterior_mean)
        elif posterior_type == 'mixture_gaussian':
            def wrapper(layer): return MixtureGaussianWrapper(layer, prior_mean=prior_mean, prior_std=prior_std, posterior_p=posterior_p,
                                                              posterior_std=posterior_std, train_posterior_std=train_posterior_std,
                                                              posterior_mean=posterior_mean, train_posterior_mean=train_posterior_mean)
        elif posterior_type == 'mixture_laplace':
            def wrapper(layer): return MixtureLaplaceWrapper(layer, prior_mean=prior_mean, prior_std=prior_std, posterior_p=posterior_p,
                                                             posterior_std=posterior_std, train_posterior_std=train_posterior_std,
                                                             posterior_mean=posterior_mean, train_posterior_mean=train_posterior_mean)
        super(StoWideResNet, self).__init__()
        self.conv1 = Conv2d(in_channels, 16, 3, stride=1,
                            padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', init_method=init_method, activation='relu')
        self.conv2 = nn.Sequential(
            StoBlock(16, 16*k, wrapper, 1, init_method),
            *(
                StoBlock(16*k, 16*k, wrapper, 1, init_method) for _ in range(n_per_block-1)
            )
        )
        self.conv3 = nn.Sequential(
            StoBlock(16*k, 32*k, wrapper, 2, init_method),
            *(
                StoBlock(32*k, 32*k, wrapper, 1, init_method) for _ in range(n_per_block-1)
            )
        )
        self.conv4 = nn.Sequential(
            StoBlock(32*k, 64*k, wrapper, 2, init_method),
            *(
                StoBlock(64*k, 64*k, wrapper, 1, init_method) for _ in range(n_per_block-1)
            )
        )
        self.bn = nn.BatchNorm2d(64*k, eps=1e-5, momentum=0.1)
        nn.init.uniform_(self.bn.weight)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(size//4)
        self.fc1 = Linear(64*k, n_classes, bias=True,
                          init_method=init_method, activation='linear')
        self.fc1 = wrapper(self.fc1)

    def forward(self, x, L=1):
        x = torch.repeat_interleave(x, repeats=L, dim=0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.relu(self.bn(x))
        x = self.avg_pool(x)
        x = self.fc1(x.flatten(start_dim=1, end_dim=-1))
        x = F.log_softmax(x, dim=-1)
        x = x.reshape((-1, L) + x.shape[1:])
        return x

    def kl(self, n_sample):
        return sum(l.kl(n_sample) for l in self.conv2)+sum(l.kl(n_sample) for l in self.conv3)+sum(l.kl(n_sample) for l in self.conv4)+self.fc1.kl(n_sample)

    def negative_loglikelihood(self, x, y, L, return_prob=False):
        y_pred = self.forward(x, L)
        y_target = y.unsqueeze(1).repeat(1, L)
        logp = D.Categorical(logits=y_pred).log_prob(y_target)
        logp = torch.logsumexp(
            logp, dim=1) - torch.log(torch.tensor(L, dtype=torch.float32, device=logp.device))
        if return_prob:
            return -logp.mean(), y_pred
        return -logp.mean()

    def vb_loss(self, x, y, loglike_sample, kl_sample, no_kl=False):
        y_pred = self.forward(x, loglike_sample)
        y_target = y.unsqueeze(1).repeat(1, loglike_sample)
        logp = D.Categorical(logits=y_pred).log_prob(y_target)
        if no_kl:
            return -logp.mean(), torch.zeros(())
        return -logp.mean(), self.kl(kl_sample)

class DropBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout, stride, init_method):
        super(DropBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_filters, eps=1e-5, momentum=0.1)
        nn.init.uniform_(self.bn1.weight)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = Conv2d(in_filters, out_filters, 3, stride=stride,
                            padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', init_method=init_method, activation='relu')
        self.dropout = dropout
        self.bn2 = nn.BatchNorm2d(out_filters, eps=1e-5, momentum=0.1)
        nn.init.uniform_(self.bn2.weight)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_filters, out_filters, 3, stride=1,
                            padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', init_method=init_method, activation='relu')
        if stride != 1 or in_filters != out_filters:
            self.has_shortcut = True
            self.shortcut = Conv2d(in_filters, out_filters, 1, stride=stride,
                                   padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', init_method=init_method, activation='relu')
        else:
            self.has_shortcut = False

    def kl(self, n_sample):
        return self.conv1.kl(n_sample) + self.conv2.kl(n_sample)

    def forward(self, x):
        o = self.relu1(self.bn1(x))
        out = self.conv1(F.dropout(o, p=self.dropout, training=True))
        out = self.conv2(F.dropout(self.relu2(self.bn2(out)), p=self.dropout, training=True))
        if self.has_shortcut:
            out = out + self.shortcut(o)
        else:
            out = out + x
        return out


class DropWideResNet(nn.Module):
    def __init__(self, size, in_channels, dropout, n_classes=10, n_per_block=4, k=2, init_method='normal'):
        super(DropWideResNet, self).__init__()
        self.conv1 = Conv2d(in_channels, 16, 3, stride=1,
                            padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', init_method=init_method, activation='relu')
        self.conv2 = nn.Sequential(
            DropBlock(16, 16*k, dropout, 1, init_method),
            *(
                DropBlock(16*k, 16*k, dropout, 1, init_method) for _ in range(n_per_block-1)
            )
        )
        self.conv3 = nn.Sequential(
            DropBlock(16*k, 32*k, dropout, 2, init_method),
            *(
                DropBlock(32*k, 32*k, dropout, 1, init_method) for _ in range(n_per_block-1)
            )
        )
        self.conv4 = nn.Sequential(
            DropBlock(32*k, 64*k, dropout, 2, init_method),
            *(
                DropBlock(64*k, 64*k, dropout, 1, init_method) for _ in range(n_per_block-1)
            )
        )
        self.bn = nn.BatchNorm2d(64*k, eps=1e-5, momentum=0.1)
        nn.init.uniform_(self.bn.weight)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(size//4)
        self.fc1 = Linear(64*k, n_classes, bias=True,
                          init_method=init_method, activation='linear')
        self.dropout = dropout

    def forward(self, x, L=1):
        x = torch.repeat_interleave(x, repeats=L, dim=0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.relu(self.bn(x))
        x = self.avg_pool(x)
        x = F.dropout(x, p=self.dropout, training=True)
        x = self.fc1(x.flatten(start_dim=1, end_dim=-1))
        x = F.log_softmax(x, dim=-1)
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

def get_wrn_model_from_config(config, width, height, in_channels, n_classes):
    if config['model_type'] == 'deterministic':
        model = DetWideResNet(width, in_channels, config['dropout'], n_classes, config['n_per_block'], config['k_factor'], config['init_method'])
    elif config['model_type'] == 'dropout':
        model = DropWideResNet(width, in_channels, config['dropout'], n_classes, config['n_per_block'], config['k_factor'], config['init_method'])
    else:
        model = StoWideResNet(width, in_channels, config['posterior_type'], config['init_prior_mean'], config['init_prior_std'], config['posterior_p'], 
                              config['posterior_mean'], config['posterior_std'], config['train_posterior_std'], config['train_posterior_mean'], n_classes, config['n_per_block'], config['k_factor'], config['init_method'])
    return model