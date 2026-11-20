"""B0+ 통제군의 hidden 크기를 비교 대상에 맞춰 계산한다 (슬라이드 8).

사용:
    python -m scripts.param_match                      # 모든 설정의 파라미터 수 표
    python -m scripts.param_match --target configs/f4.yaml   # 그 설정에 맞는 B0+ hidden 제안
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.net import build_model          # noqa: E402
from src.train import load_config               # noqa: E402
from src.utils import count_params, head_params  # noqa: E402


def params_of(path: str):
    cfg = load_config(path)
    model = build_model(cfg)
    return cfg["name"], count_params(model), head_params(model)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default=None, help="B0+가 맞출 설정 (head 파라미터가 가장 많은 것)")
    p.add_argument("--glob", default="configs/*.yaml")
    args = p.parse_args()

    rows = []
    for path in sorted(glob.glob(args.glob)):
        try:
            rows.append((os.path.basename(path), *params_of(path)))
        except Exception as e:  # noqa: BLE001
            print(f"  (skip {path}: {e})")

    print(f"{'config':<16}{'name':<16}{'total':>12}{'head':>10}")
    for fname, name, total, head in rows:
        print(f"{fname:<16}{name:<16}{total:>12,}{head:>10,}")

    if args.target:
        _, target_total, target_head = params_of(args.target)
        # B0+ head = Linear(64,h) + Linear(h,10) = 64h + h + 10h + 10 = 75h + 10
        hidden = max(round((target_head - 10) / 75), 1)
        est_head = 75 * hidden + 10
        print(f"\n[B0+ 파라미터 맞춤] target={args.target} head={target_head:,}")
        print(f"  → configs/b0plus.yaml 의 hidden: {hidden}  (B0+ head ≈ {est_head:,}, "
              f"차이 {est_head - target_head:+,})")


if __name__ == "__main__":
    main()
