import torch.nn as nn
import torch
import torch.nn.functional as F


class LearnableSerieslinearUnit(nn.ReLU):
    def __init__(self, dim, num_activations=3, deploy=False):
        super(LearnableSerieslinearUnit, self).__init__()
        self.num_activations = num_activations
        self.dim = dim
        self.deploy = deploy
        self.weight = torch.nn.Parameter(
            torch.randn(self.dim, 1, self.num_activations * 2 + 1, self.num_activations * 2 + 1))

        # 创建多个激活函数的系数（α）和偏置（b）作为可学习参数
        self.alphas = nn.Parameter(torch.ones(self.dim))
        self.biases = nn.Parameter(torch.zeros(self.dim))

        if self.deploy:
            self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        else:
            self.bias = None
            self.bn = nn.BatchNorm2d(self.dim, eps=1e-6)

        # 创建多个激活函数并放入 ModuleList 中
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(self.num_activations)])

        # 卷积核权重的通道数根据输入动态设置
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x):
        if self.deploy:
            # 部署模式下，使用卷积操作
            return F.conv2d(super(LearnableSerieslinearUnit, self).forward(x),
                            self.weight * self.alphas.view(-1, 1, 1, 1), self.bias, padding=self.num_activations,
                            groups=self.dim)
        else:
            # 训练模式下，逐一应用多个激活函数，并根据 α 和 b 进行缩放和偏置
            for i in range(self.num_activations):
                x = self.activations[i](x)  # 使用 ModuleList 中的激活函数
                x = x * self.alphas[i] + self.biases[i]

            return self.bn(F.conv2d(
                    super(LearnableSerieslinearUnit, self).forward(x),
                    self.weight, None, padding=self.num_activations, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        # 在这里实现与 BN 层的融合操作
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std

    def switch_to_deploy(self):
        # 在这里实现切换到 deploy 模式的操作
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, drop_rate=0.05, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.drop = nn.Dropout(drop_rate)
        

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.drop(out)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, drop_rate=0.05, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.drop = nn.Dropout(drop_rate)
        

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.drop(out)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=6,
                 drop_rate=0.05,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.act_learn = 1

        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.lslu = LearnableSerieslinearUnit(self.in_channel, 3, deploy=False)  # 原地计算不展开新的存储空间
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.drop = nn.Dropout(drop_rate)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # self.depth = nn.ModuleList()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.depth = len(blocks_num)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def change_act(self, m):
        self.act_learn = m

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lslu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = torch.nn.functional.leaky_relu(x, self.act_learn)
        x = self.layer2(x)
        x = torch.nn.functional.leaky_relu(x, self.act_learn)
        x = self.layer3(x)
        x = torch.nn.functional.leaky_relu(x, self.act_learn)
        x = self.layer4(x)
        x = torch.nn.functional.leaky_relu(x, self.act_learn)

        if self.include_top:
            x = self.avgpool(x)
            x = self.drop(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet18(num_classes=1000, drop_rate=0.05, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, drop_rate=0.05, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, drop_rate=0.1, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, drop_rate=0.2, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
