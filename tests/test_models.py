import pytest
import torch
from torch.nn.functional import cross_entropy

from dataset import split_indices
from model import MODELS, build_model, parameter_count


@pytest.mark.parametrize("name", MODELS)
def test_forward_backward(name):
    torch.set_num_threads(1)
    model = build_model(name)
    images = torch.randn(2, 3, 32, 32)
    features = model.backbone(images)
    assert [tuple(item.shape[1:]) for item in features] == [(16, 32, 32), (32, 16, 16), (64, 8, 8)]
    logits = model(images)
    assert logits.shape == (2, 10)
    loss = cross_entropy(logits, torch.tensor([0, 9]))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.backbone.stem[0].weight.grad.abs().sum() > 0


def test_match_and_weights():
    assert parameter_count(build_model("F2")) - parameter_count(build_model("B0+")) == 8
    model = build_model("F3", uniform=True).eval()
    assert torch.allclose(model.fusion.weights(), torch.full((3,), 1 / 3))
    _, gates = build_model("F4").eval()(torch.randn(2, 3, 32, 32), return_gates=True)
    assert gates.shape == (2, 3, 64)
    assert torch.allclose(gates.sum(1), torch.ones(2, 64))


@pytest.mark.parametrize("name,options", [
    ("F2", {"stages": (3,)}), ("F4", {"stages": (2, 3)}),
    ("F1", {"stages": (1, 3)}), ("F3", {"uniform": True}),
    ("F4", {"projection": "pad"}), ("F3", {"stop_gradient": True}),
])
def test_ablation(name, options):
    logits = build_model(name, **options)(torch.randn(2, 3, 32, 32))
    cross_entropy(logits, torch.tensor([0, 9])).backward()
    assert logits.shape == (2, 10)


def test_split():
    labels = [label for label in range(10) for _ in range(5000)]
    previous = set()
    for ratio, size in [(10, 4500), (25, 11250), (100, 45000)]:
        train, val = split_indices(labels, ratio, 2026)
        assert (train, val) == split_indices(labels, ratio, 2026)
        assert len(set(train)) == size and len(set(val)) == 5000
        assert not (set(train) & set(val))
        assert previous <= set(train)
        assert [sum(labels[i] == label for i in train) for label in range(10)] == [size // 10] * 10
        previous = set(train)


def test_stop_gradient_only_removes_direct_intermediate_path():
    images = torch.randn(2, 3, 32, 32)
    for detach in (False, True):
        model = build_model("F3", stop_gradient=detach)
        middle, _, final = model.vectors(images)
        stem_weight = model.backbone.stem[0].weight
        projection_weight = model.project[0].weight
        stem_grad, head_grad = torch.autograd.grad(
            middle.sum(), (stem_weight, projection_weight), allow_unused=True, retain_graph=True,
        )
        assert (stem_grad is None) == detach
        assert head_grad is not None
        assert torch.autograd.grad(final.sum(), stem_weight)[0].abs().sum() > 0
