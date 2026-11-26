import argparse
import time
from pathlib import Path

import torch
import torchvision
from torch import nn

from dataset import train_loaders
from model import MODELS, build_model, parameter_count
from utils import (ROOT, device_for, load_config, restore_rng, revision, rng_state,
                   run_name, save_checkpoint, set_seed, write_csv, write_json)


def epoch_pass(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.CrossEntropyLoss()
    totals = torch.zeros(2, device=device)
    count = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            totals[0] += loss.detach() * labels.numel()
            totals[1] += (logits.argmax(1) == labels).sum()
            count += labels.numel()
    return tuple((totals / count).tolist())


def train(args):
    config = load_config(args.config)
    if config["optimizer"] != "SGD" or config["scheduler"] != "CosineAnnealingLR":
        raise ValueError("This protocol supports SGD and CosineAnnealingLR")
    epochs = args.epochs if args.epochs is not None else config["epochs"]
    lr = args.lr if args.lr is not None else config["learning_rates"][1]
    workers = args.workers if args.workers is not None else config["workers"]
    if epochs < 1 or lr <= 0 or workers < 0:
        raise ValueError("Invalid epochs, learning rate or worker count")
    run_id = run_name(args.run_id or f"{args.model}_r{args.ratio}_s{args.seed}_lr{lr:g}_e{epochs}")
    out = args.results_dir / run_id
    checkpoint = args.checkpoints_dir / run_id / "last.pt"
    best_checkpoint = checkpoint.with_name("best.pt")
    if args.resume and not checkpoint.exists():
        raise ValueError(f"No checkpoint for run {run_id}: {checkpoint}")
    if not args.resume and (checkpoint.exists() or (out.exists() and any(out.iterdir()))):
        raise ValueError(f"Run {run_id} already exists; use --resume or a new --run-id")

    device = device_for(args.device)
    set_seed(args.seed)
    options = dict(stages=tuple(args.stages), projection=args.projection,
                   uniform=args.uniform, stop_gradient=args.stop_gradient)
    model = build_model(args.model, **options).to(device)
    training, validation = train_loaders(
        args.data_dir, args.ratio, config["split_seed"], args.seed,
        config["batch_size"], workers, device.type == "cuda", config.get("data"),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=config["momentum"],
                                weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    settings = dict(run_id=run_id, model=args.model, ratio=args.ratio, seed=args.seed, lr=lr,
                    epochs=epochs, options=options, config=config, workers=workers,
                    device=str(device), torch_version=str(torch.__version__),
                    cuda_version=torch.version.cuda, torchvision_version=torchvision.__version__,
                    cudnn_version=torch.backends.cudnn.version(),
                    hardware=(torch.cuda.get_device_name(device) if device.type == "cuda"
                              else "Apple MPS" if device.type == "mps" else "cpu"),
                    commit=revision(),
                    params=parameter_count(model))
    history, best = [], None
    if args.resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        previous = state["settings"]
        if {key: val for key, val in previous.items() if key != "hardware"} != {
            key: val for key, val in settings.items() if key != "hardware"
        }:
            raise ValueError("Resume requires identical resolved settings and software revision")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best = state["history"], state["best"]
        if [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
            raise ValueError(f"Checkpoint has invalid epoch history: {checkpoint}")
        restore_rng(state["rng"], training)
        write_csv(out / "history.csv", history)
    write_json(out / "settings.json", settings)
    print(f"{run_id} | device={device} | parameters={settings['params']}", flush=True)
    for epoch in range(len(history) + 1, epochs + 1):
        start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        try:
            train_loss, train_acc = epoch_pass(model, training, device, optimizer)
            val_loss, val_acc = epoch_pass(model, validation, device)
        except RuntimeError as error:
            if device.type == "mps":
                raise RuntimeError("MPS training failed; retry with --device cpu and a new run ID") from error
            raise
        scheduler.step()
        seconds = time.perf_counter() - start
        history.append(dict(epoch=epoch, lr=current_lr, train_loss=train_loss,
                            train_acc=train_acc, val_loss=val_loss, val_acc=val_acc,
                            seconds=seconds, hardware=settings["hardware"]))
        improved = best is None or val_acc > best["val_acc"]
        if improved:
            best = dict(epoch=epoch, val_acc=val_acc,
                        model={key: value.detach().cpu().clone()
                               for key, value in model.state_dict().items()})
        if args.model == "F3":
            history[-1].update({f"alpha_s{stage}": weight for stage, weight in
                                zip(model.stages, model.fusion.weights().detach().tolist())})
        state = dict(settings=settings, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     rng=rng_state(training), history=history, best=best,
                     epoch=epoch, training_seconds=sum(row["seconds"] for row in history))
        # last.pt is the resume source; best.pt records the epoch selected by validation.
        if improved:
            save_checkpoint(best_checkpoint, state)
        save_checkpoint(checkpoint, state)
        write_csv(out / "history.csv", history)
        print(f"{epoch}/{epochs} train={train_loss:.4f} val={val_loss:.4f} "
              f"val_acc={val_acc:.4f} ({seconds:.1f}s)", flush=True)


def parser():
    cli = argparse.ArgumentParser(description="Train and validate without reading test data")
    cli.add_argument("--model", choices=MODELS, default="B0")
    cli.add_argument("--ratio", type=int, choices=(10, 25, 100), default=100)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--lr", type=float)
    cli.add_argument("--epochs", type=int)
    cli.add_argument("--workers", type=int)
    cli.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    cli.add_argument("--stages", type=int, nargs="+", default=(1, 2, 3))
    cli.add_argument("--projection", choices=("linear", "pad"), default="linear")
    cli.add_argument("--uniform", action="store_true")
    cli.add_argument("--stop-gradient", action="store_true")
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--run-id")
    cli.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")
    cli.add_argument("--data-dir", type=Path, default=ROOT / "data")
    cli.add_argument("--results-dir", type=Path, default=ROOT / "results")
    cli.add_argument("--checkpoints-dir", type=Path, default=ROOT / "checkpoints")
    return cli


if __name__ == "__main__":
    train(parser().parse_args())
