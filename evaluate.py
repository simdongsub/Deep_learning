import argparse
import json
from pathlib import Path

import torch

from dataset import test_loader
from model import build_model
from train import epoch_pass
from utils import ROOT, device_for, write_csv


def evaluate(checkpoint, data_dir, results_dir, device, flops):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    settings = state["settings"]
    if len(state["history"]) != settings["epochs"]:
        raise ValueError("Complete the run before final test evaluation")
    model = build_model(settings["model"], **settings["options"]).to(device)
    model.load_state_dict(state["best"]["model"])
    loader = test_loader(data_dir, settings["config"]["batch_size"], settings["workers"])
    loss, accuracy = epoch_pass(model, loader, device)
    row = {key: settings[key] for key in ("run_id", "model", "ratio", "seed", "lr", "epochs", "params")}
    row.update(best_epoch=state["best"]["epoch"], best_val_acc=state["best"]["val_acc"],
               test_loss=loss, accuracy_pct=100 * accuracy, error_pct=100 * (1 - accuracy),
               train_seconds=sum(entry["seconds"] for entry in state["history"]),
               flops="", flops_unsupported="")
    if flops:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(model, torch.zeros(1, 3, 32, 32, device=device))
        row["flops"] = analysis.total()
        row["flops_unsupported"] = json.dumps(dict(analysis.unsupported_ops()), sort_keys=True)
    out = results_dir / settings["run_id"]
    write_csv(out / "test.csv", [row])
    if settings["model"] == "F4":
        sums = torch.zeros(10, len(model.stages), 64, device=device)
        counts = torch.zeros(10, device=device)
        with torch.no_grad():
            for images, labels in loader:
                labels = labels.to(device)
                _, gates = model(images.to(device), return_gates=True)
                sums.index_add_(0, labels, gates)
                counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))
        averages = (sums / counts[:, None, None]).cpu().tolist()
        write_csv(out / "gates.csv", [
            dict(label=label, stage=stage, channel=channel, gate=value)
            for label, class_gates in enumerate(averages)
            for stage, channels in zip(model.stages, class_gates)
            for channel, value in enumerate(channels)
        ])
    print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the validation-selected checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--flops", action="store_true")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.results_dir, device_for(args.device), args.flops)
