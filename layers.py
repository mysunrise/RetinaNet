import torch.nn as nn


def init_conv_weights(layer):
    nn.init.normal(layer.weight.data, std=0.01)
    nn.init.constant(layer.bias.data, val=0)
    return layer


def conv1x1(in_channel, out_channel, **kwargs):
    layer = nn.Conv2d(in_channel, out_channel, kernel_size=1, **kwargs)
    init_conv_weights(layer)
    return layer

def conv3x3(in_channel, out_channel, **kwargs):
    layer = nn.Conv2d(in_channel, out_channel, kernel_size=3, **kwargs)
    init_conv_weights(layer)
    return layer



