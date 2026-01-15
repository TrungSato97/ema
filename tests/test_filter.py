"""
Unit tests for filter logic.
Run: pytest tests/test_filter.py
"""

import numpy as np
import pandas as pd
from src.filter_utils import filter_by_ema


def test_filter_promote_min_keep():
    # Create dummy CSV df with 10 samples across 2 classes (5 each)
    rows = []
    for i in range(10):
        cls = 0 if i < 5 else 1
        rows.append({
            "index": i,
            "image_path": f"/tmp/img_{i}.png",
            "label_noisy": cls,
            "label_orig": cls,
            "class_name": f"class_{cls}",
            "split": "train",
            "noise_flag": 0
        })
    df = pd.DataFrame(rows)

    # Make z_ema such that predicted classes are opposite of label_noisy except one
    # z_ema shape (10,2)
    z_ema = np.zeros((10, 2), dtype=float)
    for i in range(10):
        if i == 0:
            # correct for one sample
            true_cls = df.at[i, "label_noisy"]
            z_ema[i, true_cls] = 0.9
            z_ema[i, 1 - true_cls] = 0.1
        else:
            # wrong prediction
            true_cls = df.at[i, "label_noisy"]
            z_ema[i, true_cls] = 0.1
            z_ema[i, 1 - true_cls] = 0.9

    indices = df["index"].values
    # Set min_keep_ratio to 0.6 -> need ceil(5*0.6)=3 per class
    new_df, stats = filter_by_ema(indices, z_ema, df, min_keep_ratio=0.6)
    # Check that each class has at least 3 kept
    for cls_name in ["class_0", "class_1"]:
        info = stats[cls_name]
        assert info["kept"] >= 3
