"""
EMA utilities: update and save/load predictions as .npz
"""

from typing import Optional, Dict
import numpy as np
import logging
from .io_utils import save_npz, load_npz

logger = logging.getLogger(__name__)


def update_ema(z_prev: Optional[np.ndarray], z_hat: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute EMA update:
        z_new = alpha * z_prev + (1 - alpha) * z_hat
    If z_prev is None, return a copy of z_hat (initialization).
    Inputs:
        z_prev: None or (N, C) array
        z_hat: (N, C) array
        alpha: float in [0,1]
    Returns:
        z_new: (N, C) array
    """
    if z_prev is None:
        logger.info("Initializing EMA with z_hat")
        return z_hat.copy()
    if z_prev.shape != z_hat.shape:
        raise ValueError("z_prev and z_hat shapes must match")
    return alpha * z_prev + (1.0 - alpha) * z_hat


def save_preds_npz(path: str, indices: np.ndarray, z_hat: np.ndarray, z_ema: np.ndarray) -> None:
    """
    Save indices (N,), z_hat (N, C), z_ema (N, C) into compressed npz.
    """
    save_npz(path, indices=indices.astype(np.int32), z_hat=z_hat.astype(np.float32), z_ema=z_ema.astype(np.float32))
    logger.info("Saved preds npz: %s", path)


def load_preds_npz(path: str) -> Dict[str, np.ndarray]:
    """
    Load arrays from npz. Returns dict with keys 'indices', 'z_hat', 'z_ema' (if present).
    """
    data = load_npz(path)
    return data
