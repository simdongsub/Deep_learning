import torch.nn.functional as F
from torch import nn


def conv3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.channel_pad = (out_channels - in_channels) // 2

    def forward(self, x):
        shortcut = x
        if self.stride == 2:
            shortcut = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.channel_pad, self.channel_pad))
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)) + shortcut)


class ResNet20(nn.Module):
    channels = (16, 32, 64)

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv3(3, 16), nn.BatchNorm2d(16), nn.ReLU())
        self.stages = nn.ModuleList([
            self._stage(16, 16, 1),
            self._stage(16, 32, 2),
            self._stage(32, 64, 2),
        ])
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    @staticmethod
    def _stage(in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels),
            BasicBlock(out_channels, out_channels),
        )

    def forward(self, x):
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return tuple(features)
