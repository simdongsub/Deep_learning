import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


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


def preprocess(training=False):
    ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if training else []
    return transforms.Compose([*ops, transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def train_loaders(root, ratio, split_seed, seed, batch_size, workers, pin_memory=False):
    training = datasets.CIFAR10(root, train=True, download=False, transform=preprocess(True))
    validation = datasets.CIFAR10(root, train=True, download=False, transform=preprocess())
    train_ids, val_ids = split_indices(training.targets, ratio, split_seed)
    generator = torch.Generator().manual_seed(seed)
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=pin_memory)
    return (
        DataLoader(Subset(training, train_ids), shuffle=True, generator=generator, **common),
        DataLoader(Subset(validation, val_ids), **common),
    )


def test_loader(root, batch_size, workers):
    test = datasets.CIFAR10(root, train=False, download=False, transform=preprocess())
    return DataLoader(test, batch_size=batch_size, num_workers=workers)


def download(root):
    # torchvision fetches the archive containing both train and test files.
    datasets.CIFAR10(root, train=True, download=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and verify CIFAR-10")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    download(parser.parse_args().data_dir)
