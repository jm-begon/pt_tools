"""
All the model for 32x32 were taken or adapted from
https://github.com/kuangliu/pytorch-cifar
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class Sizeable(nn.Module):
    def __init__(self, input_size, n_outputs):
        super().__init__()
        self.input_size = input_size
        self.n_outputs = n_outputs


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x, *args):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x, *args):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNet(Sizeable):
    def __init__(self, input_size, n_outputs, block, num_blocks, in_planes=64):
        super(ResNet, self).__init__(input_size, n_outputs)
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        if len(num_blocks) > 3:
            self.layer4 = self._make_layer(block, in_planes*8, num_blocks[3], stride=2)
        else:
            self.layer4 = lambda x: x
        self.linear = nn.Linear(512 * block.expansion, n_outputs)
        self.epoch = 0
        self.first_relu = nn.ReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, *args):
        out = self.first_relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# 11 173 962 parameters
def ResNet18(input_size=(3, 32, 32), n_outputs=10):
    return ResNet(input_size, n_outputs, BasicBlock, [2,2,2,2])


# 21 282 122 parameters
def ResNet34(input_size=(3, 32, 32), n_outputs=10):
    return ResNet(input_size, n_outputs, BasicBlock, [3,4,6,3])


# 23 520 842 parameters
def ResNet50(input_size=(3, 32, 32), n_outputs=10):
    return ResNet(input_size, n_outputs, Bottleneck, [3,4,6,3])


# 42 512 970 parameters
def ResNet101(input_size=(3, 32, 32), n_outputs=10):
    return ResNet(input_size, n_outputs, Bottleneck, [3,4,23,3])


# 58 156 618 parameters
def ResNet152(input_size=(3, 32, 32), n_outputs=10):
    return ResNet(input_size, n_outputs, Bottleneck, [3,8,36,3])


# 700 458 parameters (CIFAR 10)
def SmallResNet14(input_size=(3, 32, 32), n_outputs=10):
    return ResNet(input_size, n_outputs, BasicBlock, [2,2,2], in_planes=32)


class WideBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x, *args):
        out = self.dropout(self.conv1(self.relu1(self.bn1(x))))
        out = self.conv2(self.relu2(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(Sizeable):
    def __init__(self, input_size, n_outputs, depth, widen_factor, dropout_rate):
        super(WideResNet, self).__init__(input_size, n_outputs)
        self.in_planes = 16

        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        n_stages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, n_stages[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.layer1 = self._wide_layer(WideBlock, n_stages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBlock, n_stages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBlock, n_stages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(n_stages[3], momentum=0.9)
        self.linear = nn.Linear(n_stages[3], n_outputs)
        self.relu = nn.ReLU()

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, *args):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def WideResNet40_2(input_size=(3, 32, 32), n_outputs=10):
    return WideResNet(input_size, n_outputs, 40, 2, 0)


def WideResNet28_10(input_size=(3, 32, 32), n_outputs=10):
    return WideResNet(input_size, n_outputs, 28, 10, 0)


class DenseBottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x, *args):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, *args):
        out = self.conv(self.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(Sizeable):
    """
    Paper: ttps://arxiv.org/pdf/1608.06993.pdf
    """
    def __init__(self, input_size, n_outputs, block, nblocks, growth_rate=12, reduction=0.5):
        super(DenseNet, self).__init__(input_size, n_outputs)
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, n_outputs)

        self.relu = nn.ReLU()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, *args):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(self.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121(input_size=(3, 32, 32), n_outputs=10):
    return DenseNet(input_size, n_outputs, DenseBottleneck, [6,12,24,16],
                    growth_rate=32)




class ShuffleBlock(nn.Module):
    """from https://github.com/kuangliu/pytorch-cifar/blob/master/models/shufflenet.py"""
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x, *args):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        Cpg = int(C/g)
        return x.view(N,g,Cpg,H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)


class ShuffleBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(ShuffleBottleneck, self).__init__()
        self.stride = stride

        mid_planes = int(out_planes/4)
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x, *args):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out,res], 1)) if self.stride==2 else F.relu(out+res)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, n_input_channel, out_planes, num_blocks, groups,
                 n_outputs=10):
        super(ShuffleNet, self).__init__()

        self.conv1 = nn.Conv2d(n_input_channel, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[2], n_outputs)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(ShuffleBottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, *args):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ShuffleNetG2(input_size=(3, 32, 32), n_outputs=10):
    return ShuffleNet(n_input_channel=input_size[0], out_planes=[200, 400, 800],
                      num_blocks=[4, 8, 4], groups=2, n_outputs=n_outputs)
