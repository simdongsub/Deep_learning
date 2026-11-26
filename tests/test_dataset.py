from collections import Counter

from torchvision import transforms

from dataset import split_indices, test_loader as make_test_loader, train_loaders


def test_split_balanced_nested_and_independent_of_run_seed():
    labels = [label for label in range(10) for _ in range(5000)]
    subsets = []
    for ratio, expected in ((10, 4500), (25, 11250), (100, 45000)):
        train, validation = split_indices(labels, ratio, 2026)
        assert len(train) == expected
        assert len(validation) == 5000
        assert len(set(train)) == expected
        assert not set(train) & set(validation)
        assert Counter(labels[i] for i in train) == {i: expected // 10 for i in range(10)}
        assert Counter(labels[i] for i in validation) == {i: 500 for i in range(10)}
        assert (train, validation) == split_indices(labels, ratio, 2026)
        subsets.append(set(train))
    assert subsets[0] < subsets[1] < subsets[2]


def test_loaders_keep_split_when_run_seed_changes(monkeypatch):
    class FakeCifar:
        def __init__(self, root, train, download, transform=None):
            self.targets = [label for label in range(10) for _ in range(5000 if train else 1000)]
            self.transform = transform

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return index, self.targets[index]

    monkeypatch.setattr("dataset.datasets.CIFAR10", FakeCifar)
    first_train, first_val = train_loaders("unused", 10, 2026, 0, 128, 0)
    second_train, second_val = train_loaders("unused", 10, 2026, 1, 128, 0)
    assert first_train.dataset.indices == second_train.dataset.indices
    assert first_val.dataset.indices == second_val.dataset.indices
    assert len(make_test_loader("unused", 128, 0).dataset) == 10000

    train_ops = first_train.dataset.dataset.transform.transforms
    val_ops = first_val.dataset.dataset.transform.transforms
    test_ops = make_test_loader("unused", 128, 0).dataset.transform.transforms
    assert any(isinstance(op, transforms.RandomCrop) for op in train_ops)
    assert any(isinstance(op, transforms.RandomHorizontalFlip) for op in train_ops)
    assert not any(isinstance(op, (transforms.RandomCrop, transforms.RandomHorizontalFlip))
                   for op in (*val_ops, *test_ops))
