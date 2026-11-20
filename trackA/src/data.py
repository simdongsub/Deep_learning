"""CIFAR-10 데이터 파이프라인.

- 학습 45,000 / 검증 5,000 / 테스트 10,000, 고정 시드(SPLIT_SEED)로 한 번만 생성 (슬라이드 6)
- 데이터 양 변인: 100% / 25% / 10%, 클래스 균형 유지
- 부분집합은 중첩된다: 10% ⊂ 25% ⊂ 100% (클래스별 순열이 fraction과 무관하게 같음)
- 증강: RandomCrop(32, padding=4) + RandomHorizontalFlip (슬라이드 10)
"""

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .utils import SPLIT_SEED

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
NUM_TRAIN = 45_000
NUM_CLASSES = 10


def _transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def _split_indices(targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(targets))
    return perm[:NUM_TRAIN], perm[NUM_TRAIN:]


def _balanced_subset(targets: np.ndarray, train_idx: np.ndarray, fraction: float) -> np.ndarray:
    """클래스별로 같은 비율을 뽑는다. fraction이 달라도 순열은 같아 부분집합이 중첩된다."""
    if fraction >= 1.0:
        return train_idx
    rng = np.random.RandomState(SPLIT_SEED + 1)
    chosen = []
    for c in range(NUM_CLASSES):
        cls_idx = train_idx[targets[train_idx] == c]
        cls_idx = cls_idx[rng.permutation(len(cls_idx))]
        n = max(int(round(len(cls_idx) * fraction)), 1)
        chosen.append(cls_idx[:n])
    out = np.concatenate(chosen)
    return np.sort(out)


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)


def build_loaders(data_cfg: dict, batch_size: int, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    root = data_cfg.get("root", "./data")
    fraction = float(data_cfg.get("fraction", 1.0))
    num_workers = int(data_cfg.get("num_workers", 2))

    train_tf, eval_tf = _transforms()
    # 같은 원본에 transform만 다르게 두 번 올린다 (검증은 증강 없음)
    train_full = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    val_full = datasets.CIFAR10(root, train=True, download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)

    targets = np.asarray(train_full.targets)
    train_idx, val_idx = _split_indices(targets)
    train_idx = _balanced_subset(targets, train_idx, fraction)

    g = torch.Generator()
    g.manual_seed(seed)

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    train_loader = DataLoader(
        Subset(train_full, train_idx.tolist()), batch_size=batch_size, shuffle=True,
        drop_last=False, worker_init_fn=_worker_init, generator=g, **common,
    )
    val_loader = DataLoader(
        Subset(val_full, val_idx.tolist()), batch_size=512, shuffle=False, **common,
    )
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, **common)

    meta = {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_set)),
        "fraction": fraction,
        "per_class_train": int(len(train_idx) // NUM_CLASSES),
    }
    return train_loader, val_loader, test_loader, meta
