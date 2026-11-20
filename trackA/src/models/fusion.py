"""결합(Fusion) 모듈 — 슬라이드 7·8의 B0 / B0+ / F1~F4.

모든 Head는 백본이 돌려준 (s1, s2, s3)를 받아 로짓을 낸다.
분석(슬라이드 13)을 위해 forward 중 계산한 α·게이트 값을 self.last_stats에 남긴다.
"""

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def gap(x: torch.Tensor) -> torch.Tensor:
    """Global Average Pooling: [B, C, H, W] -> [B, C]"""
    return x.mean(dim=(2, 3))


class LatentExtractor(nn.Module):
    """선택한 Stage들을 GAP → d차원으로 맞춘다 (슬라이드 7).

    proj='linear' : 선형 사영 (기본)
    proj='pad'    : 채널 zero-padding, 파라미터 없음 (Ablation A3)
    stop_grad=True: 중간층 경로에 stop-gradient (Ablation A4)
    """

    def __init__(
        self,
        stage_channels: Sequence[int],
        levels: Sequence[int],
        dim: int = 64,
        proj: str = "linear",
        stop_grad: bool = False,
    ):
        super().__init__()
        assert proj in ("linear", "pad"), f"unknown proj: {proj}"
        self.levels = list(levels)          # 1-based, 예: [1, 2, 3]
        self.dim = dim
        self.proj_type = proj
        self.stop_grad = stop_grad
        self.in_channels = [stage_channels[i - 1] for i in self.levels]

        if proj == "linear":
            self.projs = nn.ModuleList([nn.Linear(c, dim) for c in self.in_channels])
        else:
            self.projs = None
            for c in self.in_channels:
                if c > dim:
                    raise ValueError(f"proj='pad'는 채널 수({c}) <= dim({dim})일 때만 가능")

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        """반환: [B, k, d] — k = 선택한 Stage 수"""
        zs: List[torch.Tensor] = []
        last_level = max(self.levels)
        for i, lv in enumerate(self.levels):
            v = gap(feats[lv - 1])
            # A4: 중간층 경로로 흐르는 기울기만 끊는다 (마지막 Stage 경로는 유지)
            if self.stop_grad and lv != last_level:
                v = v.detach()
            if self.projs is not None:
                v = self.projs[i](v)
            elif v.shape[1] < self.dim:
                v = F.pad(v, (0, self.dim - v.shape[1]))
            zs.append(v)
        return torch.stack(zs, dim=1)


class _BaseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_stats: Dict[str, torch.Tensor] = {}


class B0Head(_BaseHead):
    """베이스라인: Stage 3 출력 → GAP → FC (슬라이드 6)."""

    def __init__(self, stage_channels: Sequence[int], num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(stage_channels[-1], num_classes)

    def forward(self, feats):
        return self.fc(gap(feats[-1]))


class B0PlusHead(_BaseHead):
    """파라미터 맞춤 통제군: B0에 여분의 FC를 하나 더 둔다 (슬라이드 8).

    hidden 값은 scripts/param_match.py 로 결정한다.
    """

    def __init__(self, stage_channels: Sequence[int], hidden: int, num_classes: int = 10):
        super().__init__()
        c = stage_channels[-1]
        self.fc1 = nn.Linear(c, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, feats):
        return self.fc2(F.relu(self.fc1(gap(feats[-1]))))


class AddHead(_BaseHead):
    """F1 · 더하기 (z1 + z2 + z3) → FC"""

    def __init__(self, extractor: LatentExtractor, num_classes: int = 10):
        super().__init__()
        self.extractor = extractor
        self.fc = nn.Linear(extractor.dim, num_classes)

    def forward(self, feats):
        z = self.extractor(feats)          # [B, k, d]
        return self.fc(z.sum(dim=1))


class ConcatHead(_BaseHead):
    """F2 · Concat → FC (정보 손실 없음, 파라미터 최다)"""

    def __init__(self, extractor: LatentExtractor, num_classes: int = 10):
        super().__init__()
        self.extractor = extractor
        self.fc = nn.Linear(extractor.dim * extractor.num_levels, num_classes)

    def forward(self, feats):
        z = self.extractor(feats)
        return self.fc(z.flatten(1))


class WeightedSumHead(_BaseHead):
    """F3 · 학습형 가중합. α는 Softmax 스칼라 — 층별 기여도를 그대로 읽을 수 있다.

    learn_alpha=False 면 균등 가중(1/k) 고정 — Ablation A2.
    """

    def __init__(self, extractor: LatentExtractor, learn_alpha: bool = True, num_classes: int = 10):
        super().__init__()
        self.extractor = extractor
        self.learn_alpha = learn_alpha
        k = extractor.num_levels
        if learn_alpha:
            self.alpha_logits = nn.Parameter(torch.zeros(k))
        else:
            self.register_buffer("alpha_logits", torch.zeros(k))
        self.fc = nn.Linear(extractor.dim, num_classes)

    def alphas(self) -> torch.Tensor:
        if self.learn_alpha:
            return torch.softmax(self.alpha_logits, dim=0)
        k = self.extractor.num_levels
        return torch.full((k,), 1.0 / k, device=self.alpha_logits.device)

    def forward(self, feats):
        z = self.extractor(feats)                      # [B, k, d]
        a = self.alphas()                              # [k]
        self.last_stats = {"alpha": a.detach()}
        return self.fc((z * a.view(1, -1, 1)).sum(dim=1))


class GateHead(_BaseHead):
    """F4 · Attention Gating (SE 모듈 응용) — 입력마다 다른 채널 게이트.

    context = concat(z) -> Linear -> ReLU -> Linear -> Sigmoid -> [B, k, d]
    """

    def __init__(self, extractor: LatentExtractor, reduction: int = 4, num_classes: int = 10):
        super().__init__()
        self.extractor = extractor
        k, d = extractor.num_levels, extractor.dim
        total = k * d
        hidden = max(total // reduction, 8)
        self.gate = nn.Sequential(
            nn.Linear(total, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, total),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(d, num_classes)

    def forward(self, feats):
        z = self.extractor(feats)                       # [B, k, d]
        b, k, d = z.shape
        g = self.gate(z.flatten(1)).view(b, k, d)       # [B, k, d]
        # 분석용: 샘플별 Stage 평균 게이트 세기
        self.last_stats = {"gate_level_mean": g.mean(dim=2).detach()}
        return self.fc((z * g).sum(dim=1))


def build_head(head_cfg: dict, stage_channels: Sequence[int], num_classes: int = 10) -> nn.Module:
    """config의 head 블록을 받아 Head를 만든다."""
    htype = head_cfg["type"].lower()

    if htype == "b0":
        return B0Head(stage_channels, num_classes)
    if htype == "b0plus":
        return B0PlusHead(stage_channels, int(head_cfg["hidden"]), num_classes)

    extractor = LatentExtractor(
        stage_channels=stage_channels,
        levels=head_cfg.get("levels", [1, 2, 3]),
        dim=int(head_cfg.get("dim", 64)),
        proj=head_cfg.get("proj", "linear"),
        stop_grad=bool(head_cfg.get("stop_grad", False)),
    )

    if htype in ("f1", "add"):
        return AddHead(extractor, num_classes)
    if htype in ("f2", "concat"):
        return ConcatHead(extractor, num_classes)
    if htype in ("f3", "weighted"):
        return WeightedSumHead(extractor, bool(head_cfg.get("learn_alpha", True)), num_classes)
    if htype in ("f4", "gate"):
        return GateHead(extractor, int(head_cfg.get("reduction", 4)), num_classes)

    raise ValueError(f"unknown head type: {htype}")
