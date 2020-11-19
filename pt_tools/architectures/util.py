
import torch.nn as nn

import torch
from torchvision.models.densenet import _DenseLayer

from .arch_32x32 import Bottleneck, BasicBlock, \
    ShuffleBottleneck


def save_model(model, path):
    if path is None:
        return
    torch.save(model.state_dict(), path)



def count_parameters(model):
    return sum(layer.data.nelement() for layer in model.parameters())


def magnitude(number):
    s = str(number)
    for unit in "k", "M", "G", "T", "P":
        if number // 1000 > 1:
            number /= 1000.
            s = "~ {} {}".format(int(number+.5), unit)
    return s


def create_conv_from_sizes(input_size, output_size):
    def inv(s_i, s_o):
        if s_i < s_o:
            raise NotImplementedError('So far input size must be greater than output size')
        return s_i - s_o + 1
    c_i, h_i, w_i = input_size
    c_o, h_o, w_o = output_size

    k_h, k_w = inv(h_i, h_o), inv(w_i, w_o)
    return nn.Conv2d(c_i, c_o, (k_h, k_w), stride=1, padding=0)


def conv_output_size(input_size, kernel_size, stride=1, padding=0):
    return int((input_size - kernel_size + 2 * padding) / stride) + 1


def pooling_output_size(input_size, kernel_size, stride=2):
    return conv_output_size(input_size, kernel_size, stride, 0)


def compute_output_size(module, input_size):
    n_channels, height, width = input_size

    if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
        # Pooling/Convolution

        try:
            k_h, k_w = module.kernel_size
        except TypeError:
            k_h = k_w = module.kernel_size
        try:
            s_h, s_w = module.stride
        except TypeError:
            s_h= s_w = module.stride
        try:
            p_h, p_w = module.padding
        except TypeError:
            p_h = p_w = module.padding

        width = conv_output_size(width, k_w, s_w, p_w)
        height = conv_output_size(height, k_h, s_h, p_h)

    if isinstance(module, nn.Conv2d):
        n_channels = module.out_channels

    elif isinstance(module, nn.BatchNorm2d):
        n_channels = module.num_features

    elif isinstance(module, _DenseLayer):  # Must come before Sequential
        n_channels_bu = n_channels
        for m in module:
            n_channels, height, width = compute_output_size(m, (n_channels, height, width))
        n_channels += n_channels_bu

    elif isinstance(module, nn.Sequential):
        for m in module:
            n_channels, height, width = compute_output_size(m, (n_channels, height, width))

    elif isinstance(module, Bottleneck):
        for m in module.conv1, module.conv2, module.conv3:
            n_channels, height, width = compute_output_size(m, (n_channels, height, width))

    elif isinstance(module, BasicBlock):
        for m in module.conv1, module.conv2:
            n_channels, height, width = compute_output_size(m, (n_channels, height, width))

    elif isinstance(module, ShuffleBottleneck):
        # Behave either as a resnet-like layer or as a densenet-like
        n_channels_bu = n_channels
        for m in module.conv1, module.conv2, module.conv3:
            n_channels, height, width = compute_output_size(m, (n_channels, height, width))
        if module.stride == 2:
            n_channels += n_channels_bu

    return n_channels, height, width


