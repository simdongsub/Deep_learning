import argparse
import csv
import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from statistics import mean

from model import MODELS
from utils import ROOT, load_config, write_csv, write_json


def record(run_id, model, ratio, seed, lr, epochs, owner, results, checkpoints, extra=()):
    command = ["train.py", "--model", model, "--ratio", str(ratio), "--seed", str(seed),
               "--lr", str(lr), "--epochs", str(epochs), "--run-id", run_id,
               "--results-dir", str(results), "--checkpoints-dir", str(checkpoints), *extra]
    return dict(run_id=run_id, model=model, ratio=ratio, seed=seed, lr=lr, epochs=epochs,
                owner=owner, status="planned", estimated_minutes="20–30 (proposal)",
                actual_seconds="", result_path=str(results / run_id), argv=json.dumps(command))


def core_id(model, ratio, seed, lr):
    return f"core_{model}_r{ratio}_s{seed}_lr{lr:g}"


def validation_score(directory, row):
    with (directory / "settings.json").open() as file:
        settings = json.load(file)
    for field in ("run_id", "model", "ratio", "seed", "lr", "epochs"):
        if str(settings[field]) != str(row[field]):
            raise ValueError(f"Incompatible settings in {directory}: {field}")
    with (directory / "history.csv").open(newline="") as file:
        history = list(csv.DictReader(file))
    if [int(item["epoch"]) for item in history] != list(range(1, row["epochs"] + 1)):
        raise ValueError(f"Incomplete run: {directory}")
    return max(float(item["val_acc"]) for item in history)


def add_paths(parser):
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--checkpoints-dir", type=Path, default=ROOT / "checkpoints")


def execute(manifest, run_id, resume=False, options=()):
    with manifest.open(newline="", encoding="utf-8") as file:
        matches = [row for row in csv.DictReader(file) if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError("Select exactly one existing run ID")
    argv = json.loads(matches[0]["argv"])
    if not isinstance(argv, list) or not argv or argv[0] != "train.py":
        raise ValueError("Malformed training manifest")
    subprocess.run([sys.executable, *argv, *options, *(["--resume"] if resume else [])],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Plan runs, select shared LR, launch one run")
    add_paths(parser)
    parser.add_argument("--ratios", nargs="+", type=int, choices=(10, 25, 100), default=(100, 25, 10))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--choices", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("action", choices=("tune", "select", "core", "run"))
    args = parser.parse_args()
    results, checkpoints = args.results_dir.resolve(), args.checkpoints_dir.resolve()
    ratios = tuple(dict.fromkeys(args.ratios))
    config = load_config(ROOT / "configs/default.yaml")
    rates, seeds, epochs = (config[key] for key in ("learning_rates", "seeds", "epochs"))
    choices_path = args.choices or results / "selected_lr.json"
    if args.action == "run":
        if not args.run_id or not args.output:
            parser.error("run requires --output MANIFEST and --run-id ID")
        overrides = []
        if args.data_dir:
            overrides.extend(["--data-dir", str(args.data_dir.resolve())])
        if args.device:
            overrides.extend(["--device", args.device])
        execute(args.output, args.run_id, args.resume, overrides)
        return
    if args.action == "core":
        with choices_path.open() as file:
            choices = {int(ratio): value["lr"] for ratio, value in json.load(file).items()}
    rows = [
        record(core_id(model, ratio, seed, lr), model, ratio, seed, lr, epochs,
               {"B0": "A", "B0+": "B", "F1": "B", "F2": "C", "F3": "C", "F4": "C"}[model],
               results, checkpoints)
        for ratio, model, seed in product(ratios, MODELS, seeds if args.action == "core" else seeds[:1])
        for lr in ([choices[ratio]] if args.action == "core" else rates)
    ]
    if args.action == "select":
        scores = {}
        for ratio, lr in product(ratios, rates):
            scores[ratio, lr] = mean(
                validation_score(Path(row["result_path"]), row)
                for row in rows if (row["ratio"], row["lr"]) == (ratio, lr)
            )
        write_json(choices_path, {
            ratio: dict(lr=max(sorted(rates), key=lambda lr: scores[ratio, lr]),
                        mean_best_val={lr: scores[ratio, lr] for lr in rates})
            for ratio in ratios
        })
    else:
        if args.action == "core":
            for row in rows:
                if row["seed"] == seeds[0]:
                    row["status"] = "reuse_tuning_checkpoint"
        write_csv(args.output or results / f"{args.action}_plan.csv", rows)


if __name__ == "__main__":
    main()
