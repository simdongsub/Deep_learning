import torch
import torch.nn.functional as F
from torch import nn

from models.fusion import AddFusion, AttentionFusion, ConcatFusion, WeightedFusion
from models.resnet import ResNet20

MODELS = ("B0", "B0+", "F1", "F2", "F3", "F4")


class Classifier(nn.Module):
    def __init__(self, name, stages=(1, 2, 3), projection="linear", uniform=False,
                 stop_gradient=False):
        super().__init__()
        if name not in MODELS:
            raise ValueError(f"Unknown model: {name}")
        if not stages or tuple(sorted(set(stages))) != tuple(stages) or stages[-1] != 3:
            raise ValueError("Select unique stages in increasing order, including stage 3")
        if set(stages) - {1, 2, 3} or projection not in {"linear", "pad"}:
            raise ValueError("Invalid stage or projection")
        if uniform and name != "F3":
            raise ValueError("Uniform weights are an F3 control")
        self.backbone = ResNet20()
        self.stages = (3,) if name in ("B0", "B0+") else tuple(stages)
        self.stop_gradient = stop_gradient
        self.project = nn.ModuleList([
            nn.Linear(channels, 64) if projection == "linear" and channels < 64
            else nn.Identity()
            for channels in (ResNet20.channels[stage - 1] for stage in self.stages)
        ])
        if name == "B0+":
            # F2 adds (16+1)*64 + (32+1)*64 + (192+1)*64 = 15,552.
            # A 120-wide residual MLP adds (64+1)*120 + (120+1)*64 = 15,544.
            self.extra = nn.Sequential(nn.Linear(64, 120), nn.ReLU(), nn.Linear(120, 64))
        elif name == "F1":
            self.fusion = AddFusion()
        elif name == "F2":
            self.fusion = ConcatFusion(len(self.stages))
        elif name == "F3":
            self.fusion = WeightedFusion(len(self.stages), uniform)
        elif name == "F4":
            self.fusion = AttentionFusion()
        self.classifier = nn.Linear(64, 10)

    def vectors(self, images):
        features = self.backbone(images)
        vectors = []
        for stage, project in zip(self.stages, self.project):
            vector = features[stage - 1].mean(dim=(2, 3))
            if self.stop_gradient and stage < 3:
                vector = vector.detach()
            vector = project(vector)
            vectors.append(F.pad(vector, (0, 64 - vector.shape[1])))
        return vectors

    def forward(self, images, return_gates=False):
        vectors = self.vectors(images)
        if return_gates:
            if not isinstance(getattr(self, "fusion", None), AttentionFusion):
                raise ValueError("Gate diagnostics require F4")
            stacked = torch.stack(vectors, dim=1)
            gates = self.fusion.weights(stacked)
            return self.classifier((stacked * gates).sum(dim=1)), gates
        if hasattr(self, "fusion"):
            vector = self.fusion(vectors)
        else:
            vector = vectors[0]
            if hasattr(self, "extra"):
                vector = vector + self.extra(vector)
        return self.classifier(vector)


def build_model(name, **options):
    """Build one of MODELS for CIFAR-10.

    Input: float tensor (N, 3, 32, 32). Output: logits before softmax (N, 10).
    ``model.backbone(images)`` returns S1~S3 maps (N,16,32,32), (N,32,16,16), (N,64,8,8).
    Options: ``stages=(1, 2, 3)`` (increasing, must include 3; B0/B0+ always use S3),
    ``projection="linear"`` or ``"pad"``, ``uniform=False`` (F3 only),
    ``stop_gradient=False``. Invalid names or options raise ValueError.
    Move the model with ``.to(device)`` in the caller.
    """
    return Classifier(name, **options)


def parameter_count(model):
    """Return the number of trainable parameters (B0: 269,722)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
