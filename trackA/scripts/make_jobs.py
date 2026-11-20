"""실행 목록(슬라이드 11의 약 90회)을 명령어로 찍어준다.

사용:
    python -m scripts.make_jobs --stage tuning   > jobs_tuning.sh   # 18회
    python -m scripts.make_jobs --stage core     > jobs_core.sh     # 54회
    python -m scripts.make_jobs --stage ablation > jobs_ablation.sh # 약 18회
"""

import argparse

MAIN = ["configs/b0.yaml", "configs/b0plus.yaml", "configs/f1.yaml",
        "configs/f2.yaml", "configs/f3.yaml", "configs/f4.yaml"]
ABLATION = ["configs/ablation/a1_s2s3.yaml", "configs/ablation/a1_s1s3.yaml",
            "configs/ablation/a2_uniform_alpha.yaml", "configs/ablation/a3_pad.yaml",
            "configs/ablation/a4_stopgrad.yaml"]
LRS = [0.05, 0.1, 0.2]
FRACTIONS = [1.0, 0.25, 0.10]
SEEDS = [0, 1, 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["tuning", "core", "ablation"], required=True)
    p.add_argument("--out", default="runs")
    p.add_argument("--lr", type=float, default=0.1, help="core/ablation 단계에서 쓸 선택된 학습률")
    args = p.parse_args()

    print("#!/usr/bin/env bash\nset -e")
    if args.stage == "tuning":
        # 6 설정 × 학습률 3 × 시드 1 = 18회 (검증 세트로 학습률 선택)
        for cfg in MAIN:
            for lr in LRS:
                print(f"python -m src.train --config {cfg} --seed 0 --lr {lr} "
                      f"--fraction 1.0 --out {args.out} --tag tuning")
    elif args.stage == "core":
        # 6 설정 × 데이터 3종 × 시드 3 = 54회
        for cfg in MAIN:
            for frac in FRACTIONS:
                for seed in SEEDS:
                    print(f"python -m src.train --config {cfg} --seed {seed} --lr {args.lr} "
                          f"--fraction {frac} --out {args.out}")
    else:
        # Ablation: 100% 데이터, 시드 3
        for cfg in ABLATION:
            for seed in SEEDS:
                print(f"python -m src.train --config {cfg} --seed {seed} --lr {args.lr} "
                      f"--fraction 1.0 --out {args.out}")


if __name__ == "__main__":
    main()
