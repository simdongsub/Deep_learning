import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from utils import load_config

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs/default.yaml"


def split_indices(targets, ratio, split_seed):
    if ratio not in (10, 25, 100):
        raise ValueError("ratio must be 10, 25 or 100")
    rng = random.Random(split_seed)
    groups = [[] for _ in range(10)]
    for index, label in enumerate(targets):
        groups[label].append(index)
    train, validation = [], []
    for group in groups:
        if len(group) != 5000:
            raise ValueError("Expected 5,000 CIFAR-10 training samples per class")
        rng.shuffle(group)
        validation.extend(group[:500])
        train.extend(group[500:500 + 4500 * ratio // 100])
    return train, validation


def preprocess(training=False, data_config=None):
    config = data_config or load_config(DEFAULT_CONFIG)["data"]
    ops = []
    if training:
        ops.append(transforms.RandomCrop(config["crop_size"], padding=config["crop_padding"]))
        if config["horizontal_flip"]:
            ops.append(transforms.RandomHorizontalFlip())
    return transforms.Compose([
        *ops, transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"]),
    ])


def train_loaders(root, ratio, split_seed, seed, batch_size, workers, pin_memory=False,
                  data_config=None):
    training = datasets.CIFAR10(root, train=True, download=False,
                                transform=preprocess(True, data_config))
    validation = datasets.CIFAR10(root, train=True, download=False,
                                  transform=preprocess(False, data_config))
    train_ids, val_ids = split_indices(training.targets, ratio, split_seed)
    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=pin_memory)
    return (
        DataLoader(Subset(training, train_ids), shuffle=True, generator=generator, **common),
        DataLoader(Subset(validation, val_ids), **common),
    )


def test_loader(root, batch_size, workers, data_config=None):
    test = datasets.CIFAR10(root, train=False, download=False,
                            transform=preprocess(False, data_config))
    return DataLoader(test, batch_size=batch_size, num_workers=workers)


def download(root):
    # torchvision fetches the archive containing both train and test files.
    datasets.CIFAR10(root, train=True, download=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and verify CIFAR-10")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    root = parser.parse_args().data_dir
    download(root)
    training = datasets.CIFAR10(root, train=True, download=False)
    testing = datasets.CIFAR10(root, train=False, download=False)
    split_seed = load_config(DEFAULT_CONFIG)["split_seed"]
    for ratio in (100, 25, 10):
        train_ids, val_ids = split_indices(training.targets, ratio, split_seed)
        counts = [sum(training.targets[index] == label for index in train_ids)
                  for label in range(10)]
        assert not set(train_ids).intersection(val_ids)
        print(f"{ratio}%: train={len(train_ids)}, validation={len(val_ids)}, "
              f"per_class={counts}")
    print(f"official_test={len(testing)}")
