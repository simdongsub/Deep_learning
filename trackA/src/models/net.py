"""백본 + 결합 Head 조립. 모든 설정이 같은 백본을 쓰고 Head만 바뀐다."""

import torch.nn as nn

from .fusion import build_head
from .resnet import ResNetCifar, plain20, resnet20


class Net(nn.Module):
    def __init__(self, backbone: ResNetCifar, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    @property
    def last_stats(self):
        """분석용 α / 게이트 값 (없으면 빈 dict)."""
        return getattr(self.head, "last_stats", {})


def build_model(cfg: dict, num_classes: int = 10) -> Net:
    arch = cfg.get("arch", "resnet20").lower()
    if arch == "resnet20":
        backbone = resnet20(use_shortcut=cfg.get("use_shortcut", True))
    elif arch == "plain20":
        backbone = plain20()
    else:
        raise ValueError(f"unknown arch: {arch}")

    head = build_head(cfg["head"], backbone.stage_channels, num_classes)
    return Net(backbone, head)
