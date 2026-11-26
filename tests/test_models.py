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


def test_parameter_counts():
    expected = {"B0": 269722, "B0+": 285266, "F1": 272922,
                "F2": 285274, "F3": 272925, "F4": 275050}
    assert {name: parameter_count(build_model(name)) for name in MODELS} == expected


def test_fusion_formulas():
    torch.manual_seed(0)
    images = torch.randn(4, 3, 32, 32)
    for name in MODELS:
        model = build_model(name).eval()
        vectors = model.vectors(images)
        if name == "B0":
            expected = vectors[0]
        elif name == "B0+":
            expected = vectors[0] + model.extra(vectors[0])
        elif name == "F1":
            expected = sum(vectors)
        elif name == "F2":
            expected = model.fusion.reduce(torch.cat(vectors, dim=1))
        elif name == "F3":
            weights = model.fusion.weights()
            assert torch.allclose(weights.sum(), torch.tensor(1.0))
            expected = sum(w * v for w, v in zip(weights, vectors))
        else:
            _, gates = model(images, return_gates=True)
            assert not torch.allclose(gates[0], gates[1])
            expected = (torch.stack(vectors, dim=1) * gates).sum(dim=1)
        assert torch.allclose(model(images), model.classifier(expected), atol=1e-5)


def test_uniform_and_pad_options():
    uniform = build_model("F3", uniform=True)
    assert not any("scores" in key for key, _ in uniform.named_parameters())
    model = build_model("F1", projection="pad")
    assert sum(p.numel() for p in model.project.parameters()) == 0
    first, second, _ = model.vectors(torch.randn(2, 3, 32, 32))
    assert (first[:, 16:] == 0).all() and (second[:, 32:] == 0).all()


@pytest.mark.parametrize("options", [{"stages": (1, 2)}, {"stages": (3, 1)}, {"projection": "x"}])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        build_model("F1", **options)
