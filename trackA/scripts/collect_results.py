"""runs/ 아래 summary.json들을 모아 슬라이드 12 양식의 표로 만든다.

사용:
    python -m scripts.collect_results --runs runs --out results
출력:
    results/all_runs.csv       : 실행 하나당 한 줄
    results/summary_table.md   : 설정 × 데이터 양, 평균 ± 표준편차 (오차율 %)
"""

import argparse
import csv
import glob
import json
import os
import statistics as st
from collections import defaultdict

METRIC = "test_err_final"   # 보고 지표: 마지막 에폭 테스트 오차율


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    p.add_argument("--metric", default=METRIC, choices=["test_err_final", "test_err_best_val"])
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    records = []
    for path in sorted(glob.glob(os.path.join(args.runs, "**", "summary.json"), recursive=True)):
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))
    if not records:
        print(f"summary.json 을 찾지 못했습니다: {args.runs}")
        return

    csv_path = os.path.join(args.out, "all_runs.csv")
    fields = ["run", "name", "seed", "fraction", "lr", "params_total", "params_head",
              "val_err_best", "test_err_final", "test_err_best_val", "train_time_sec"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    # (설정, 데이터 양)별 시드 평균 ± 표준편차
    groups = defaultdict(list)
    for r in records:
        groups[(r["name"], r["fraction"])].append(r)

    fractions = sorted({f for _, f in groups}, reverse=True)
    names = sorted({n for n, _ in groups})

    lines = [f"# 결과 요약 ({args.metric}, 테스트 오차율 %)", "",
             "| 설정 | " + " | ".join(f"{int(f*100)}%" for f in fractions) + " | 파라미터 수 | 학습 시간(분) |",
             "|---|" + "---|" * (len(fractions) + 2)]
    for name in names:
        cells = []
        for frac in fractions:
            rs = groups.get((name, frac), [])
            if not rs:
                cells.append("—")
                continue
            vals = [r[args.metric] for r in rs]
            mean = st.mean(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            cells.append(f"{mean:.2f} ± {sd:.2f}" + ("" if len(vals) >= 3 else f" (n={len(vals)})"))
        same_name = [r for r in records if r["name"] == name]
        params = same_name[0]["params_total"]
        mins = st.mean([r["train_time_sec"] for r in same_name]) / 60
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {params:,} | {mins:.0f} |")

    lines += ["", "해석 규칙: |μA − μB| < σ 이면 '효과 없음' (슬라이드 5).",
              "시드 3개(0/1/2) 평균 ± 표준편차. Welch t-test p값은 참고치로만 병기한다."]

    md_path = os.path.join(args.out, "summary_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n저장: {csv_path}, {md_path}  (실행 {len(records)}건)")


if __name__ == "__main__":
    main()
