import argparse
import json
from itertools import product

from experiments import add_paths, core_id, record, validation_score
from utils import ROOT, load_config, write_csv

ABLATION_OWNERS = {
    "s3": "동섭",
    "s23": "동섭",
    "s13": "환호",
    "pad": "환호",
    "uniform": "현성",
    "detach": "현성",
}


def main():
    parser = argparse.ArgumentParser(description="Plan 6 ablation variants × 3 seeds")
    add_paths(parser)
    parser.add_argument("--ratio", type=int, choices=(10, 25, 100), default=10)
    args = parser.parse_args()
    results = args.results_dir.resolve()
    checkpoints = args.checkpoints_dir.resolve()
    config = load_config(ROOT / "configs/default.yaml")
    with (results / "selected_lr.json").open() as file:
        lr = json.load(file)[str(args.ratio)]["lr"]
    seeds, epochs = config["seeds"], config["epochs"]

    def score(model):
        from statistics import mean
        return mean(
            validation_score(results / core_id(model, args.ratio, seed, lr),
                             dict(run_id=core_id(model, args.ratio, seed, lr), model=model,
                                  ratio=args.ratio, seed=seed, lr=lr, epochs=epochs))
            for seed in seeds
        )

    winner = max(("F1", "F2", "F3", "F4"), key=score)
    variants = [
        ("s3", winner, ["--stages", "3"]),
        ("s23", winner, ["--stages", "2", "3"]),
        ("s13", winner, ["--stages", "1", "3"]),
        ("uniform", "F3", ["--uniform"]),
        ("pad", winner, ["--projection", "pad"]),
        ("detach", winner, ["--stop-gradient"]),
    ]
    rows = []
    for (label, model, options), seed in product(variants, seeds):
        row = record(f"abl_{label}_{model}_r{args.ratio}_s{seed}_lr{lr:g}", model,
                     args.ratio, seed, lr, epochs, ABLATION_OWNERS[label], results, checkpoints, options)
        row["reference"] = core_id(model, args.ratio, seed, lr)
        rows.append(row)
    write_csv(results / "ablation_plan.csv", rows)
    print(f"Validation-selected fusion at {args.ratio}%: {winner}")


if __name__ == "__main__":
    main()
