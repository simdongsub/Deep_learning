import csv
import json
import os
import random
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name):
    device = torch.device("cuda" if name == "auto" and torch.cuda.is_available()
                          else "cpu" if name == "auto" else name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return device


def run_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", value):
        raise ValueError("Use a simple identifier for --run-id")
    return value


def revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("TRACK_A_ARCHIVE_COMMIT")


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path, rows):
    if not rows:
        raise ValueError("CSV requires at least one row")
    with atomic_path(path) as temporary:
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_json(path, data):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_checkpoint(path, state):
    with atomic_path(path) as temporary:
        torch.save(state, temporary)


def rng_state(loader):
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": loader.generator.get_state(),
    }


def restore_rng(state, loader):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    loader.generator.set_state(state["loader"])
