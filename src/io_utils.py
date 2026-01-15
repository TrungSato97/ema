"""
I/O helpers for saving outputs, making directories, saving CSV/npz and checkpoints.
"""

import os
from pathlib import Path
import json
from typing import Any, Dict
import numpy as np
import pandas as pd
import torch
import logging

logger = logging.getLogger(__name__)


def make_dirs(path: str) -> None:
    """
    Create directory if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)


def make_iter_dirs(exp_dir: str, iteration_idx: int) -> Dict[str, str]:
    """
    Make per-iteration directories and return dict of paths.
    """
    base = Path(exp_dir) / f"iteration_{iteration_idx}"
    paths = {
        "base": str(base),
        "checkpoints": str(base / "checkpoints"),
        "preds": str(base / "preds"),
        "preds_npz": str(base / "preds_npz"),
        "metrics": str(base / "metrics"),
        "eda": str(base / "eda"),
        "logs": str(base / "logs"),
    }
    for p in paths.values():
        make_dirs(p)
    return paths


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: Any, scheduler: Any,
                    epoch: int, extra: Dict[str, Any] = None) -> None:
    """
    Save checkpoint including optimizer and scheduler states.
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        try:
            state["scheduler_state_dict"] = scheduler.state_dict()
        except Exception:
            # Some schedulers may not have state
            pass
    if extra:
        state["extra"] = extra
    torch.save(state, path)
    logger.info("Saved checkpoint %s", path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Any = None,
                    scheduler: Any = None, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Load checkpoint and populate model/optimizer/scheduler states if provided.
    Returns the loaded dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception:
            logger.warning("Failed to load scheduler state")
    return checkpoint


def save_npz(path: str, **arrays) -> None:
    """
    Save compressed numpy arrays to npz.
    """
    dirname = os.path.dirname(path)
    if dirname:
        make_dirs(dirname)
    np.savez_compressed(path, **arrays)
    logger.info("Saved npz to %s", path)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """
    Load arrays from npz file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    return {k: data[k] for k in data.files}


def save_dataframe_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save a dataframe to csv, overwriting existing.
    """
    dirname = os.path.dirname(path)
    if dirname:
        make_dirs(dirname)
    df.to_csv(path, index=False)
    logger.info("Saved CSV to %s", path)


def append_row_to_csv(path: str, row: Dict[str, Any]) -> None:
    """
    Append a row to CSV file, creating file with header if doesn't exist.
    """
    dirname = os.path.dirname(path)
    if dirname:
        make_dirs(dirname)
    df = pd.DataFrame([row])
    if os.path.exists(path):
        df.to_csv(path, index=False, header=False, mode="a")
    else:
        df.to_csv(path, index=False, header=True)
    logger.info("Appended row to %s", path)
