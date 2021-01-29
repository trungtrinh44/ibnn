import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from typing import Type, Any, Callable, Union, List, Optional
from .utils import StoLayer, StoConv2d, StoLinear, EnsembleBatchNorm2d
from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, n_components=8, prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02)) -> StoConv2d:
    """3x3 convolution with padding"""
    return StoConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation,
                     n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, n_components=8, prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02)) -> StoConv2d:
    """1x1 convolution"""
    return StoConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                     n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)


class StoBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        n_components=8, prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02)
    ) -> None:
        super(StoBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.bn1 = norm_layer(inplanes) #, n_components)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.bn2 = norm_layer(planes) #, n_components)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out

class StoBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        n_components=8, prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02)
    ) -> None:
        super(StoBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.bn1 = norm_layer(inplanes) #, n_components)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.bn2 = norm_layer(width) #, n_components)
        self.conv3 = conv1x1(width, planes * self.expansion, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.bn3 = norm_layer(width) #, n_components)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)


        out += identity

        return out

class StoResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[StoBasicBlock, StoBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        n_components=8, prior_mean=1.0, prior_std=0.1, posterior_mean_init=(1.0, 0.5), posterior_std_init=(0.05, 0.02)
    ) -> None:
        super(StoResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.n_components = n_components

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = StoConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)
        self.bn1 = norm_layer(512 * block.expansion)
        self.relu = nn.ReLU(True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = StoLinear(512 * block.expansion, num_classes, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, StoBottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #         elif isinstance(m, StoBasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
        self.sto_modules = set(
            m for m in self.modules() if isinstance(m, StoLayer)
        )
    
    def kl(self):
        return sum(m.kl() for m in self.sto_modules)

    def vb_loss(self, x, y, n_sample):
        y = y.unsqueeze(1).expand(-1, n_sample)
        logp = D.Categorical(logits=self.forward(x, n_sample)).log_prob(y).mean()
        return -logp, self.kl()
    
    def nll(self, x, y, n_sample):
        log_prob = self.forward(x, n_sample*self.n_components, False)
        logp = D.Categorical(logits=log_prob).log_prob(y.unsqueeze(1).expand(-1, self.n_components*n_sample))
        logp = torch.logsumexp(logp, 1) - torch.log(torch.tensor(self.n_components*n_sample, dtype=torch.float32, device=x.device))
        return -logp.mean(), log_prob

    def _make_layer(self, block: Type[Union[StoBasicBlock, StoBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, n_components=None, prior_mean=None, prior_std=None, posterior_mean_init=None, posterior_std_init=None) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, n_components=n_components, prior_mean=prior_mean, prior_std=prior_std, posterior_mean_init=posterior_mean_init, posterior_std_init=posterior_std_init))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, L=1) -> Tensor:
        # See note [TorchScript super()]
        if L > 1:
            x = torch.repeat_interleave(x, L, dim=0)
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.log_softmax(self.fc(x), dim=-1)

        x = x.view(-1, L, x.size(1))
        return x

    def forward(self, x: Tensor, L=1, return_kl=False) -> Tensor:
        if return_kl:
            return self._forward_impl(x, L), self.kl()
        return self._forward_impl(x, L)

def _resnet(
    arch: str,
    block: Type[Union[StoBasicBlock, StoBottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> StoResNet:
    model = StoResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model

def resnet50(deterministic_pretrained: bool = False, progress: bool = True, **kwargs: Any) -> StoResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', StoBottleneck, [3, 4, 6, 3], deterministic_pretrained, progress,
                   **kwargs)
