"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from .utils import StoLayer

__all__ = ["VGG16", "VGG16BN", "VGG19", "VGG19BN", "StoVGG16", "StoVGG16BN", "StoVGG19", "StoVGG19BN"]


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

def make_sto_layers(cfg, n_components, prior_mean, prior_std, batch_norm=False, stochastic_first_layer=False):
    layers = list()
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if i > 0 or stochastic_first_layer:
                sl = StoLayer((in_channels, 1, 1), n_components, prior_mean, prior_std)
            else:
                sl = nn.Identity()
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
    def __init__(self, num_classes, depth=16, batch_norm=False):
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
        return x

class StoVGG(nn.Module):
    def __init__(self, num_classes, n_components, prior_mean, prior_std, depth=16, batch_norm=False, stochastic_first_layer=False):
        super(StoVGG, self).__init__()
        self.features = make_sto_layers(cfg[depth], n_components, prior_mean, prior_std, batch_norm, stochastic_first_layer)
        self.classifier = nn.ModuleList((
            StoLayer((512, ), n_components, prior_mean, prior_std),
            nn.Linear(512, 512),
            nn.ReLU(True),
            StoLayer((512, ), n_components, prior_mean, prior_std),
            nn.Linear(512, 512),
            nn.ReLU(True),
            StoLayer((512, ), n_components, prior_mean, prior_std),
            nn.Linear(512, num_classes),
        ))
        self.n_components = n_components
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()
        self.sto_modules = [m for m in self.modules() if isinstance(m, StoLayer)]

    def kl(self, n_sample):
        return sum(m.kl(n_sample) for m in self.sto_modules)

    def forward(self, x, L=1, indices=None):
        x = torch.repeat_interleave(x, repeats=L, dim=0)
        if indices is None:
            indices = torch.multinomial(torch.ones(self.n_components, device=x.device), x.size(0), replacement=True)
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
        x = F.log_softmax(x, dim=-1)
        x = x.reshape((-1, L) + x.shape[1:])
        return x

class VGG16(VGG):
    def __init__(self, num_classes):
        super(VGG16, self).__init__(num_classes=num_classes, depth=16, batch_norm=False)


class VGG16BN(VGG):
    def __init__(self, num_classes):
        super(VGG16BN, self).__init__(num_classes=num_classes, depth=16, batch_norm=True)


class VGG19(VGG):
    def __init__(self, num_classes):
        super(VGG19, self).__init__(num_classes=num_classes, depth=19, batch_norm=False)


class VGG19BN(VGG):
    def __init__(self, num_classes):
        super(VGG19BN, self).__init__(num_classes=num_classes, depth=19, batch_norm=True)

class StoVGG16(StoVGG):
    def __init__(self, num_classes, n_components, prior_mean, prior_std, stochastic_first_layer):
        super(StoVGG16, self).__init__(num_classes=num_classes, depth=16, batch_norm=False, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std)


class StoVGG16BN(StoVGG):
    def __init__(self, num_classes, n_components, prior_mean, prior_std, stochastic_first_layer):
        super(StoVGG16BN, self).__init__(num_classes=num_classes, depth=16, batch_norm=True, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std)


class StoVGG19(StoVGG):
    def __init__(self, num_classes, n_components, prior_mean, prior_std, stochastic_first_layer):
        super(StoVGG19, self).__init__(num_classes=num_classes, depth=19, batch_norm=False, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std)


class StoVGG19BN(StoVGG):
    def __init__(self, num_classes, n_components, prior_mean, prior_std, stochastic_first_layer):
        super(StoVGG19BN, self).__init__(num_classes=num_classes, depth=19, batch_norm=True, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std)