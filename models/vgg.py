"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from .utils import StoLayer

__all__ = ["DetVGG16", "DetVGG16BN", "DetVGG19", "DetVGG19BN", "StoVGG16", "StoVGG16BN", "StoVGG19", "StoVGG19BN"]


def make_layers(cfg, batch_norm=False):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_sto_layers(cfg, batch_norm=False, n_components=2, prior_mean=1.0, prior_std=1.0):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            sl = StoLayer((in_channels, 1, 1), n_components, prior_mean, prior_std)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [sl, conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [sl, conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.ModuleList(layers)

cfg = {
    16: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    19: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False):
        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], batch_norm)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=-1)
        return x

class StoVGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, n_components=2, prior_mean=1.0, prior_std=1.0, batch_norm=False):
        super(StoVGG, self).__init__()
        self.features = make_sto_layers(cfg[depth], batch_norm, n_components, prior_mean, prior_std)
        self.classifier = nn.ModuleList([
            StoLayer((512, ), n_components, prior_mean, prior_std),
            nn.Linear(512, 512),
            nn.ReLU(True),
            StoLayer((512, ), n_components, prior_mean, prior_std),
            nn.Linear(512, 512),
            nn.ReLU(True),
            StoLayer((512, ), n_components, prior_mean, prior_std),
            nn.Linear(512, num_classes)
        ])
        self.n_components = n_components
        self.sto_modules = [
            m for m in self.modules() if isinstance(m, StoLayer)
        ]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x, L=1, indices=None):
        if L > 1:
            x = torch.repeat_interleave(x, L, dim=0)
        if indices is None:
            indices = torch.multinomial(torch.ones(self.n_components, device=x.device), x.size(0), True)
        for layer in self.features:
            if isinstance(layer, StoLayer):
                x = layer(x, indices)
            else:
                x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.classifier:
            if isinstance(layer, StoLayer):
                x = layer(x, indices)
            else:
                x = layer(x)
        x = F.log_softmax(x, -1)
        x = x.view(-1, L, x.size(1))
        return x
    
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


class DetVGG16(VGG):
    def __init__(self, num_classes):
        super(DetVGG16, self).__init__(num_classes, 16, False)


class DetVGG16BN(VGG):
    def __init__(self, num_classes):
        super(DetVGG16BN, self).__init__(num_classes, 16, True)


class DetVGG19(VGG):
    def __init__(self, num_classes):
        super(DetVGG19, self).__init__(num_classes, 19, False)


class DetVGG19BN(VGG):
    def __init__(self, num_classes):
        super(DetVGG19BN, self).__init__(num_classes, 19, True)

class StoVGG16(StoVGG):
    def __init__(self, num_classes, n_components, prior_mean, prior_std):
        super(StoVGG16, self).__init__(num_classes, 16, n_components, prior_mean, prior_std, False)


class StoVGG16BN(StoVGG):
    def __init__(self, num_classes, n_components, prior_mean, prior_std):
        super(StoVGG16BN, self).__init__(num_classes, 16, n_components, prior_mean, prior_std, True)


class StoVGG19(StoVGG):
    def __init__(self, num_classes, n_components, prior_mean, prior_std):
        super(StoVGG19, self).__init__(num_classes, 19, n_components, prior_mean, prior_std, False)


class StoVGG19BN(StoVGG):
    def __init__(self, num_classes, n_components, prior_mean, prior_std):
        super(StoVGG19BN, self).__init__(num_classes, 19, n_components, prior_mean, prior_std, True)
