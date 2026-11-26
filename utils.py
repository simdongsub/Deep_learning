import csv
import json
import os
import random
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path

import torch
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name):
    if name == "auto":
        name = ("cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use --device cpu")
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
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        # Plain Python values keep torch.load(weights_only=True) compatible.
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        "loader": loader.generator.get_state(),
    }


def restore_rng(state, loader):
    random.setstate(state["python"])
    if "numpy" in state:
        name, keys, position, gaussian, cached = state["numpy"]
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32),
                             position, gaussian, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
    loader.generator.set_state(state["loader"])
