import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs(*dirs):
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_checkpoint(path, model, optimizer, epoch, history=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "history": history or {},
        "extra": extra or {},
    }, path)


def load_checkpoint(path, model, optimizer=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"], checkpoint.get("history", {}), checkpoint.get("extra", {})
