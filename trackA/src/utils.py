"""시드 고정, 파라미터 집계, 설정 병합 등 공용 유틸."""

import copy
import os
import random
from typing import Any, Dict

import numpy as np
import torch

# 데이터 분할 전용 시드 — 실행 시드(0/1/2)와 분리해 분할을 한 번만 고정한다 (슬라이드 6).
SPLIT_SEED = 12345


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 속도 우선. 모든 설정에 동일하게 적용되므로 비교 공정성은 유지된다.
        torch.backends.cudnn.benchmark = True


def count_params(model: torch.nn.Module, trainable_only: bool = True) -> int:
    ps = model.parameters()
    return sum(p.numel() for p in ps if (p.requires_grad or not trainable_only))


def head_params(model) -> int:
    return sum(p.numel() for p in model.head.parameters() if p.requires_grad)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """중첩 dict 병합 (override가 이김)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def error_rate(correct: int, total: int) -> float:
    """오차율 (%) — 보고 지표는 정확도가 아니라 오차율이다 (슬라이드 12)."""
    return 100.0 * (1.0 - correct / max(total, 1))
