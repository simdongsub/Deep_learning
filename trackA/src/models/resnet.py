"""CIFAR용 ResNet-20 백본 (He et al., 2016).

- Stage 1/2/3 출력(s1, s2, s3)을 모두 반환한다. 결합 모듈이 중간층을 쓰기 때문.
- Shortcut은 원 논문의 Option A(파라미터 없는 zero-padding)를 쓴다.
  → 파라미터 수가 논문 보고치(약 0.27M)와 맞는다.
- use_shortcut=False 로 두면 Plain-20 (위험 1 대응 실험용).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortcutA(nn.Module):
    """Option A: stride로 다운샘플 + 채널 zero-padding. 학습 파라미터 없음."""

    def __init__(self, stride: int, pad: int):
        super().__init__()
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        if self.stride > 1:
            x = x[:, :, :: self.stride, :: self.stride]
        if self.pad > 0:
            # (left, right, top, bottom, front, back) — 채널 축 앞뒤로 균등 패딩
            x = F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1, use_shortcut: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.use_shortcut = use_shortcut
        if use_shortcut and (stride != 1 or in_planes != planes):
            self.shortcut = ShortcutA(stride, planes - in_planes)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_shortcut:
            out = out + self.shortcut(x)
        return F.relu(out)


class ResNetCifar(nn.Module):
    """ResNet-20 백본. forward는 (s1, s2, s3) 특징맵을 반환한다."""

    # Stage별 출력 채널 — 결합 모듈이 사영층 크기를 정할 때 참조한다.
    stage_channels = (16, 32, 64)

    def __init__(self, n: int = 3, use_shortcut: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, n, 1, use_shortcut)
        self.layer2 = self._make_layer(16, 32, n, 2, use_shortcut)
        self.layer3 = self._make_layer(32, 64, n, 2, use_shortcut)
        self._init_weights()

    @staticmethod
    def _make_layer(in_planes, planes, blocks, stride, use_shortcut):
        layers = [BasicBlock(in_planes, planes, stride, use_shortcut)]
        for _ in range(blocks - 1):
            layers.append(BasicBlock(planes, planes, 1, use_shortcut))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        s1 = self.layer1(x)   # 16 x 32 x 32
        s2 = self.layer2(s1)  # 32 x 16 x 16
        s3 = self.layer3(s2)  # 64 x 8 x 8
        return s1, s2, s3


def resnet20(use_shortcut: bool = True) -> ResNetCifar:
    return ResNetCifar(n=3, use_shortcut=use_shortcut)


def plain20() -> ResNetCifar:
    """Skip 연결 없는 동일 깊이 백본 (슬라이드 17 위험 1 대응)."""
    return ResNetCifar(n=3, use_shortcut=False)
