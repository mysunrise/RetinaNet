import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
from layers import conv1x1, conv3x3

model_urls = {
    'resnet18': 'http://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'http://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'http://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'http://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'http://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # change channel
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,  # change size
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)  # change channel to planes * expansion
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FPN(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(FPN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])  # conv2  C2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # conv3  C3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # conv4  C4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # conv5  C5

        self.conv6 = conv3x3(512 * block.expansion, 256, stride=2,
                             padding=1)  # P6
        self.conv7 = conv3x3(256, 256, stride=2, padding=1)  # P7

        self.lateral_layer1 = conv1x1(512 * block.expansion, 256)
        self.lateral_layer2 = conv1x1(256 * block.expansion, 256)
        self.lateral_layer3 = conv1x1(128 * block.expansion, 256)

        self.corr_layer1 = conv3x3(256, 256, stride=1, padding=1)  # P4
        self.corr_layer2 = conv3x3(256, 256, stride=1, padding=1)  # P3


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, h, w = y.size()
        x_upsampled = F.upsample(x, [h, w], mode='bilinear')

        return x_upsampled + y

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Bottom-up
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        p5 = self.lateral_layer1(c5)
        lat2 = self.lateral_layer2(c4)
        p4 = self._upsample_add(p5, lat2)
        p4 = self.corr_layer1(p4)
        lat3 = self.lateral_layer3(c3)
        p3 = self._upsample_add(p4, lat3)
        p3 = self.corr_layer2(p3)

        return p3, p4, p5, p6, p7


def fpn50(pretrained=False, **kwargs):
    """Constructs a fpn-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FPN(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
        #print("model state:", model.state_dict().keys())
        #print("pretrain:", model_zoo.load_url(model_urls['resnet50']).keys())
    return model


def fpn101(pretrained=False, **kwargs):
    """Constructs a fpn-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FPN(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def fpn152(pretrained=False, **kwargs):
    """Constructs a fpn-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FPN(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model
