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

from .utils import StoLayer

__all__ = ["DetWideResNet28x10", "StoWideResNet28x10"]


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

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x), inplace=True)))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out += self.shortcut(x)

        return out

class StoWideBasic(nn.Module):
    def __init__(self, in_planes, planes, stride, n_components, prior_mean, prior_std):
        super(StoWideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.sl1 = StoLayer((in_planes, 1, 1), n_components, prior_mean, prior_std)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.sl2 = StoLayer((planes, 1, 1), n_components, prior_mean, prior_std)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.has_shortcut = True
            self.sl3 = StoLayer((in_planes, 1, 1), n_components, prior_mean, prior_std)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)
            )
        else:
            self.has_shortcut = False

    def forward(self, x, indices):
        out = self.conv1(
            self.sl1(F.relu(self.bn1(x), inplace=True), indices)
        )
        out = self.conv2(
            self.sl2(F.relu(self.bn2(out), inplace=True), indices)
        )
        if self.has_shortcut:
            x = self.sl3(x, indices)
        out += self.shortcut(x)

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
    def __init__(self, num_classes=10, depth=28, widen_factor=10, n_components=2, prior_mean=1.0, prior_std=1.0):
        super(StoWideResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        nstages = [16, 16 * k, 32 * k, 64 * k]

        self.sl1 = StoLayer((3, 1, 1), n_components, prior_mean, prior_std)
        self.conv1 = conv3x3(3, nstages[0])
        self.layer1 = self._wide_layer(StoWideBasic, nstages[1], n, n_components, prior_mean, prior_std, stride=1)
        self.layer2 = self._wide_layer(StoWideBasic, nstages[2], n, n_components, prior_mean, prior_std, stride=2)
        self.layer3 = self._wide_layer(StoWideBasic, nstages[3], n, n_components, prior_mean, prior_std, stride=2)
        self.bn1 = nn.BatchNorm2d(nstages[3], momentum=0.9)
        self.sl2 = StoLayer((nstages[3], ), n_components, prior_mean, prior_std)
        self.linear = nn.Linear(nstages[3], num_classes)
        self.n_components = n_components
        self.sto_modules = [
            m for m in self.modules() if isinstance(m, StoLayer)
        ]

    def _wide_layer(self, block, planes, num_blocks, n_components, prior_mean, prior_std, stride):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n_components, prior_mean, prior_std))
            self.in_planes = planes

        return nn.ModuleList(layers)

    def forward(self, x, L=1, indices=None):
        x = torch.repeat_interleave(x, L, dim=0)
        if indices is None:
            indices = torch.multinomial(torch.ones(self.n_components, device=x.device), x.size(0), True)
        out = self.sl1(x, indices)
        out = self.conv1(x)
        for layer in self.layer1:
            out = layer(out, indices)
        for layer in self.layer2:
            out = layer(out, indices)
        for layer in self.layer3:
            out = layer(out, indices)
        out = F.relu(self.bn1(out), inplace=True)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = F.log_softmax(self.linear(self.sl2(out, indices)), -1)
        out = out.view(-1, L, out.size(1))
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



class DetWideResNet28x10(DetWideResNet):
    def __init__(self, num_classes=10):
        super(DetWideResNet28x10, self).__init__(num_classes, depth=28, widen_factor=10)
    
class StoWideResNet28x10(StoWideResNet):
    def __init__(self, num_classes=10, n_components=2, prior_mean=1.0, prior_std=1.0):
        super(StoWideResNet28x10, self).__init__(num_classes, depth=28, widen_factor=10, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std)
