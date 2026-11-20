"""학습 · 평가 엔트리포인트.

실행 예:
    python -m src.train --config configs/b0.yaml --seed 0 --out runs
    python -m src.train --config configs/f4.yaml --seed 1 --fraction 0.25 --out /content/drive/MyDrive/trackA/runs

- 매 에폭 체크포인트를 저장하고, 같은 --out/--name 으로 다시 실행하면 이어서 학습한다.
  (Colab 세션이 끊겨도 run을 버리지 않기 위함)
- 결합 방식 외의 조건은 모두 config 기본값으로 고정된다 (슬라이드 10).
"""

import argparse
import csv
import json
import os
import time
from typing import Dict

import torch
import torch.nn as nn
import yaml

from .data import build_loaders
from .models.net import build_model
from .utils import AverageMeter, count_params, deep_update, error_rate, head_params, set_seed

DEFAULTS: Dict = {
    "name": "run",
    "arch": "resnet20",
    "use_shortcut": True,
    "head": {
        "type": "b0",
        "levels": [1, 2, 3],
        "dim": 64,
        "proj": "linear",        # linear | pad   (Ablation A3)
        "learn_alpha": True,     # F3 전용        (Ablation A2)
        "stop_grad": False,      #                (Ablation A4)
        "reduction": 4,          # F4 전용
        "hidden": 64,            # B0+ 전용
    },
    "data": {"root": "./data", "fraction": 1.0, "num_workers": 2},
    "optim": {
        "lr": 0.1,               # 튜닝 탐색 대상: {0.05, 0.1, 0.2}
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "batch_size": 128,
        "epochs": 100,
        "nesterov": False,
    },
    "amp": True,
    "deterministic": False,
}


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(DEFAULTS, user_cfg)


def make_amp(enabled: bool):
    """torch 버전 차이를 흡수한 AMP 헬퍼."""
    try:
        from torch.amp import GradScaler, autocast
        scaler = GradScaler("cuda", enabled=enabled)

        def ctx():
            return autocast("cuda", enabled=enabled)
    except (ImportError, TypeError):  # 구버전 torch
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler(enabled=enabled)

        def ctx():
            return autocast(enabled=enabled)
    return scaler, ctx


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_ctx) -> Dict[str, float]:
    model.eval()
    loss_meter, correct, total = AverageMeter(), 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp_ctx():
            out = model(x)
            loss = criterion(out, y)
        loss_meter.update(loss.item(), y.size(0))
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return {"loss": loss_meter.avg, "err": error_rate(correct, total)}


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp_ctx) -> Dict[str, float]:
    model.train()
    loss_meter, correct, total = AverageMeter(), 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx():
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), y.size(0))
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return {"loss": loss_meter.avg, "err": error_rate(correct, total)}


def current_alphas(model) -> list:
    """F3의 α 값 (없으면 빈 리스트) — 슬라이드 13 분석용."""
    head = model.head
    if hasattr(head, "alphas"):
        return [round(v, 6) for v in head.alphas().detach().cpu().tolist()]
    return []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    p.add_argument("--fraction", type=float, default=None, help="config의 data.fraction 덮어쓰기")
    p.add_argument("--lr", type=float, default=None, help="config의 optim.lr 덮어쓰기 (튜닝 탐색용)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", default="", help="실행 폴더 이름에 붙일 꼬리표 (예: ablation-A2)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.fraction is not None:
        cfg["data"]["fraction"] = args.fraction
    if args.lr is not None:
        cfg["optim"]["lr"] = args.lr
    if args.epochs is not None:
        cfg["optim"]["epochs"] = args.epochs

    frac = float(cfg["data"]["fraction"])
    run_name = f"{cfg['name']}_f{int(round(frac * 100))}_lr{cfg['optim']['lr']}_s{args.seed}"
    if args.tag:
        run_name = f"{run_name}_{args.tag}"
    run_dir = os.path.join(args.out, run_name)
    os.makedirs(run_dir, exist_ok=True)

    set_seed(args.seed, deterministic=cfg["deterministic"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, meta = build_loaders(
        cfg["data"], cfg["optim"]["batch_size"], args.seed
    )
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        momentum=cfg["optim"]["momentum"],
        weight_decay=cfg["optim"]["weight_decay"],
        nesterov=cfg["optim"]["nesterov"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["optim"]["epochs"])
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    scaler, amp_ctx = make_amp(use_amp)

    total_params, hparams = count_params(model), head_params(model)
    print(f"[{run_name}] device={device} params={total_params:,} (head {hparams:,}) "
          f"train={meta['n_train']} val={meta['n_val']} test={meta['n_test']}")

    # ---- 이어서 학습 (Colab 세션 끊김 대비) -------------------------------
    ckpt_path = os.path.join(run_dir, "ckpt_last.pt")
    best_path = os.path.join(run_dir, "ckpt_best.pt")
    start_epoch, best_val, elapsed = 0, float("inf"), 0.0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_val, elapsed = ck["epoch"] + 1, ck["best_val"], ck.get("elapsed", 0.0)
        print(f"  resumed from epoch {start_epoch} (best val err {best_val:.2f}%)")

    log_path = os.path.join(run_dir, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["epoch", "lr", "train_loss", "train_err", "val_loss", "val_err", "alpha", "sec"]
            )

    # ---- 학습 루프 ---------------------------------------------------------
    for epoch in range(start_epoch, cfg["optim"]["epochs"]):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, amp_ctx)
        va = evaluate(model, val_loader, criterion, device, amp_ctx)
        scheduler.step()
        dt = time.time() - t0
        elapsed += dt
        alphas = current_alphas(model)

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, f"{lr_now:.5f}", f"{tr['loss']:.4f}", f"{tr['err']:.2f}",
                f"{va['loss']:.4f}", f"{va['err']:.2f}",
                "|".join(f"{a:.4f}" for a in alphas), f"{dt:.1f}",
            ])
        if va["err"] < best_val:
            best_val = va["err"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_err": best_val}, best_path)
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_val": best_val, "elapsed": elapsed, "cfg": cfg,
        }, ckpt_path)

        if epoch % 5 == 0 or epoch == cfg["optim"]["epochs"] - 1:
            print(f"  ep{epoch:3d} lr {lr_now:.4f} | train err {tr['err']:.2f}% "
                  f"| val err {va['err']:.2f}% | {dt:.0f}s")

    # ---- 최종 평가 ---------------------------------------------------------
    final_test = evaluate(model, test_loader, criterion, device, amp_ctx)
    final_alphas = current_alphas(model)
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    best_test = evaluate(model, test_loader, criterion, device, amp_ctx)

    summary = {
        "run": run_name, "config": args.config, "name": cfg["name"], "seed": args.seed,
        "fraction": frac, "lr": cfg["optim"]["lr"], "epochs": cfg["optim"]["epochs"],
        "params_total": total_params, "params_head": hparams,
        "val_err_best": round(best_val, 3),
        "test_err_final": round(final_test["err"], 3),
        "test_err_best_val": round(best_test["err"], 3),
        "train_time_sec": round(elapsed, 1),
        "alpha_final": final_alphas,
        "data": meta, "device": str(device), "amp": use_amp,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[{run_name}] test err (final) {final_test['err']:.2f}% | "
          f"(best-val ckpt) {best_test['err']:.2f}% | {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
