"""
Unit tests for EMA update.
Run: pytest tests/test_ema.py
"""

import numpy as np
from src.ema_utils import update_ema


def test_update_ema_basic():
    z_prev = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
    z_hat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    alpha = 0.5
    z_new = update_ema(z_prev, z_hat, alpha)
    expected = alpha * z_prev + (1 - alpha) * z_hat
    assert np.allclose(z_new, expected)


def test_update_ema_init():
    z_prev = None
    z_hat = np.array([[0.3, 0.7]], dtype=np.float32)
    alpha = 0.9
    z_new = update_ema(z_prev, z_hat, alpha)
    assert np.allclose(z_new, z_hat)
