"""
    DetWideResNet model definition
    ported from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.nn.init as init

from .utils import StoLayer, StoLinear, StoConv2d, BayesianLayer, BayesianConv2d, BayesianLinear

__all__ = ["DetWideResNet28x10", "StoWideResNet28x10", "BayesianWideResNet28x10"]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_uniform(m.weight, gain=math.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x), inplace=True)))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out += self.shortcut(x)

        return out

class BayesianWideBasic(nn.Module):
    def __init__(self, in_planes, planes, stride, prior_mean, prior_std):
        super(BayesianWideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = BayesianConv2d(in_planes, planes, kernel_size=3, padding=1, bias=True, prior_mean=prior_mean, prior_std=prior_std)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = BayesianConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, prior_mean=prior_mean, prior_std=prior_std
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = BayesianConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True, prior_mean=prior_mean, prior_std=prior_std)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out += self.shortcut(x)

        return out

class StoWideBasic(nn.Module):
    def __dummy_shortcut(self, x, i):
        return x

    def __init__(self, in_planes, planes, stride, n_components, prior_mean, prior_std, posterior_mean_init, posterior_std_init):
        super(StoWideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = StoConv2d(in_planes, planes, kernel_size=3, padding=1, bias=True, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = StoConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        if stride != 1 or in_planes != planes:
            self.shortcut = StoConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        else:
            self.shortcut = self.__dummy_shortcut

    def forward(self, x, indices):
        out = self.conv1(F.relu(self.bn1(x), inplace=True), indices)
        out = self.conv2(F.relu(self.bn2(out), inplace=True), indices)
        out += self.shortcut(x, indices)

        return out
    

class DetWideResNet(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen_factor=10, dropout_rate=0.0):
        super(DetWideResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        nstages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nstages[0])
        self.layer1 = self._wide_layer(WideBasic, nstages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nstages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nstages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nstages[3], momentum=0.9)
        self.linear = nn.Linear(nstages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out), inplace=True)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = F.log_softmax(self.linear(out), -1)

        return out

class StoWideResNet(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen_factor=10, n_components=2, prior_mean=1.0, prior_std=1.0, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02)):
        super(StoWideResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        nstages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = StoConv2d(3, nstages[0], kernel_size=3, stride=1, padding=1, bias=True, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.layer1 = self._wide_layer(StoWideBasic, nstages[1], n, n_components, prior_mean, prior_std, stride=1, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.layer2 = self._wide_layer(StoWideBasic, nstages[2], n, n_components, prior_mean, prior_std, stride=2, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.layer3 = self._wide_layer(StoWideBasic, nstages[3], n, n_components, prior_mean, prior_std, stride=2, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.bn1 = nn.BatchNorm2d(nstages[3], momentum=0.9)
        self.linear = StoLinear(nstages[3], num_classes, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.n_components = n_components
        self.sto_modules = [
            m for m in self.modules() if isinstance(m, StoLayer)
        ]

    def _wide_layer(self, block, planes, num_blocks, n_components, prior_mean, prior_std, stride, posterior_mean_init, posterior_std_init):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n_components, prior_mean, prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init))
            self.in_planes = planes

        return nn.ModuleList(layers)

    def forward(self, x, L=1, indices=None, return_kl=False):
        if L > 1:
            x = torch.repeat_interleave(x, L, dim=0)
        if indices is None:
            indices = torch.arange(x.size(0), dtype=torch.long, device=x.device) % self.n_components
        out = self.conv1(x, indices)
        for layer in self.layer1:
            out = layer(out, indices)
        for layer in self.layer2:
            out = layer(out, indices)
        for layer in self.layer3:
            out = layer(out, indices)
        out = F.relu(self.bn1(out), inplace=True)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = F.log_softmax(self.linear(out, indices), -1)
        out = out.view(-1, L, out.size(1))
        if return_kl:
            return out, self.kl()
        return out
    
    def kl(self):
        return sum(m.kl() for m in self.sto_modules)

    def vb_loss(self, x, y, n_sample):
        y = y.unsqueeze(1).expand(-1, n_sample)
        logp = D.Categorical(logits=self.forward(x, n_sample)).log_prob(y).mean()
        return -logp, self.kl()
    
    def nll(self, x, y, n_sample):
        indices = torch.empty(x.size(0)*n_sample, dtype=torch.long, device=x.device)
        prob = torch.cat([self.forward(x, n_sample, indices=torch.full((x.size(0)*n_sample,), idx, out=indices, device=x.device, dtype=torch.long)) for idx in range(self.n_components)], dim=1)
        logp = D.Categorical(logits=prob).log_prob(y.unsqueeze(1).expand(-1, self.n_components*n_sample))
        logp = torch.logsumexp(logp, 1) - torch.log(torch.tensor(self.n_components*n_sample, dtype=torch.float32, device=x.device))
        return -logp.mean(), prob

class BayesianWideResNet(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen_factor=10, prior_mean=0.0, prior_std=1.0):
        super(BayesianWideResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        nstages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nstages[0])
        self.layer1 = self._wide_layer(BayesianWideBasic, nstages[1], n, prior_mean, prior_std, stride=1)
        self.layer2 = self._wide_layer(BayesianWideBasic, nstages[2], n, prior_mean, prior_std, stride=2)
        self.layer3 = self._wide_layer(BayesianWideBasic, nstages[3], n, prior_mean, prior_std, stride=2)
        self.bn1 = nn.BatchNorm2d(nstages[3], momentum=0.9)
        self.linear = BayesianLinear(nstages[3], num_classes, bias=True, prior_mean=prior_mean, prior_std=prior_std)

    def _wide_layer(self, block, planes, num_blocks, prior_mean, prior_std, stride):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, prior_mean, prior_std))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, L=1):
        outs = []
        for _ in range(L):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.relu(self.bn1(out), inplace=True)
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            out = F.log_softmax(self.linear(out), -1)
            outs.append(out.unsqueeze(1))
        return torch.cat(outs, dim=1)
    
    def kl(self):
        return sum(m.kl() for m in self.modules() if isinstance(m, BayesianLayer))

    def vb_loss(self, x, y, n_sample):
        y = y.unsqueeze(1).expand(-1, n_sample)
        logp = D.Categorical(logits=self.forward(x, n_sample)).log_prob(y).mean()
        return -logp, self.kl()
    
    def nll(self, x, y, n_sample):
        y = y.unsqueeze(1).expand(-1, n_sample)
        prob = self.forward(x, n_sample)
        logp = D.Categorical(logits=prob).log_prob(y)
        logp = torch.logsumexp(logp, 1) - torch.log(torch.tensor(n_sample, dtype=torch.float32, device=x.device))
        return -logp.mean(), prob


class DetWideResNet28x10(DetWideResNet):
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super(DetWideResNet28x10, self).__init__(num_classes, depth=28, widen_factor=10, dropout_rate=dropout_rate)
    
class StoWideResNet28x10(StoWideResNet):
    def __init__(self, num_classes=10, n_components=2, prior_mean=1.0, prior_std=1.0, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02)):
        super(StoWideResNet28x10, self).__init__(num_classes, depth=28, widen_factor=10, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)

class BayesianWideResNet28x10(BayesianWideResNet):
    def __init__(self, num_classes=10, prior_mean=0.0, prior_std=1.0):
        super(BayesianWideResNet28x10, self).__init__(num_classes, depth=28, widen_factor=10, prior_mean=prior_mean, prior_std=prior_std)
