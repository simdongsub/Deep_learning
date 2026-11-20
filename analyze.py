import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import ROOT, write_csv


def read(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_chart(path, labels, values, deviations, ylabel):
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.55), 5))
    ax.bar(labels, values, yerr=deviations, capsize=3)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=40)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def analyze(manifests, output):
    plans = [row for path in manifests for row in read(path)]
    if len({row["run_id"] for row in plans}) != len(plans):
        raise ValueError("Use final core/ablation manifests with disjoint run IDs")
    output.mkdir(parents=True, exist_ok=True)
    by_id = {}
    groups = defaultdict(list)
    missing = []
    for plan in plans:
        directory = Path(plan["result_path"])
        history_path = directory / "history.csv"
        if history_path.exists():
            history = read(history_path)
            if history:
                fig, ax = plt.subplots()
                for key in ("train_loss", "val_loss"):
                    ax.plot([int(r["epoch"]) for r in history], [float(r[key]) for r in history], label=key)
                ax.set(xlabel="Epoch", ylabel="Loss")
                ax.legend()
                fig.savefig(output / f"{plan['run_id']}_loss.png")
                plt.close(fig)
                alphas = [key for key in history[0] if key.startswith("alpha_")]
                if alphas:
                    fig, ax = plt.subplots()
                    for key in alphas:
                        ax.plot([int(r["epoch"]) for r in history], [float(r[key]) for r in history], label=key)
                    ax.set(xlabel="Epoch", ylabel="Stage weight")
                    ax.legend()
                    fig.savefig(output / f"{plan['run_id']}_alpha.png")
                    plt.close(fig)
        path = directory / "test.csv"
        if not path.exists():
            missing.append(plan["run_id"])
            continue
        row, = read(path)
        for key in ("run_id", "model", "ratio", "seed", "lr", "epochs"):
            if row[key] != plan[key] and not (key == "lr" and float(row[key]) == float(plan[key])):
                raise ValueError(f"Unexpected {key} in {path}")
        by_id[plan["run_id"]] = row
        gates_path = directory / "gates.csv"
        if gates_path.exists():
            class_stage = defaultdict(list)
            for item in read(gates_path):
                class_stage[(int(item["label"]), int(item["stage"]))].append(float(item["gate"]))
            stages = sorted({stage for _, stage in class_stage})
            averages = [[mean(class_stage[label, stage]) for stage in stages] for label in range(10)]
            fig, ax = plt.subplots()
            picture = ax.imshow(averages, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(stages)), [f"S{stage}" for stage in stages])
            ax.set_yticks(range(10), ["airplane", "automobile", "bird", "cat", "deer",
                                      "dog", "frog", "horse", "ship", "truck"])
            fig.colorbar(picture, label="Mean channel gate")
            fig.tight_layout()
            fig.savefig(output / f"{plan['run_id']}_gates.png")
            plt.close(fig)
        if not plan.get("reference"):
            groups[(int(row["ratio"]), row["model"])].append(row)
    print(f"Missing final evaluations ({len(missing)}):", ", ".join(missing))
    if groups:
        summary = []
        for (ratio, model), runs in sorted(groups.items(), reverse=True):
            errors = [float(row["error_pct"]) for row in runs]
            summary.append(dict(ratio=ratio, model=model, n=len(runs),
                                seeds=" ".join(sorted(row["seed"] for row in runs)),
                                error_mean=mean(errors), error_sd=stdev(errors) if len(errors) > 1 else "",
                                params=runs[0]["params"],
                                seconds_mean=mean(float(row["train_seconds"]) for row in runs),
                                flops=runs[0]["flops"]))
        write_csv(output / "summary.csv", summary)
        save_chart(output / "core_error.png", [f"{r['model']} / {r['ratio']}% (n={r['n']})" for r in summary],
                   [r["error_mean"] for r in summary], [r["error_sd"] or 0 for r in summary], "Test error (%)")
    pairs = defaultdict(list)
    for plan in plans:
        run_id, reference = plan["run_id"], plan.get("reference")
        if reference and run_id in by_id and reference in by_id:
            label = run_id.split("_r", 1)[0]
            pairs[label].append(float(by_id[run_id]["error_pct"]) - float(by_id[reference]["error_pct"]))
    if pairs:
        rows = [dict(variant=label, n=len(values), delta_mean=mean(values),
                     delta_sd=stdev(values) if len(values) > 1 else "") for label, values in pairs.items()]
        write_csv(output / "ablation_summary.csv", rows)
        save_chart(output / "ablation_delta.png", [f"{r['variant']} (n={r['n']})" for r in rows],
                   [r["delta_mean"] for r in rows], [r["delta_sd"] or 0 for r in rows], "Error difference (pp)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize measured runs from final manifests")
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/analysis")
    args = parser.parse_args()
    analyze(args.manifests, args.output)
