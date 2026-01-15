"""
Filtering utilities: apply EMA-based filtering and enforce per-class minimum keep ratio.
Updated: support threshold `tau` such that we only REMOVE when
        (argmax_ema != label_noisy) AND (confidence >= tau).
If tau is None: original behavior (remove whenever argmax != label_noisy).
"""
from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import logging
import math

logger = logging.getLogger(__name__)

def filter_by_ema(indices: np.ndarray,
                  z_ema: np.ndarray,
                  train_csv_df: pd.DataFrame,
                  min_keep_ratio: float,
                  tau: Optional[float] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply filtering based on EMA predictions (z_ema) compared with label_noisy in train_csv_df.
    - indices: (N,) array of CSV 'index' values corresponding to rows in z_ema
    - z_ema: (N, C) array of EMA probabilities
    - train_csv_df: pd.DataFrame of train CSV (must contain 'index','label_noisy','class_name' ideally)
    - min_keep_ratio: float in (0,1)
    - tau: optional confidence threshold in [0,1]. If provided, removal only happens when
           (argmax != label_noisy) AND (confidence >= tau). If None -> removal when argmax != label_noisy.
    Returns:
      updated_df: original train_csv_df with additional column 'filter_flag' ('kept'/'removed')
      stats: dict containing per-class counts and kept ratios
    """
    if "index" not in train_csv_df.columns:
        raise ValueError("train_csv_df must contain 'index' column")

    # Build mapping from index -> row position in train_csv_df
    idx_values = train_csv_df["index"].values
    idx_to_row = {int(v): i for i, v in enumerate(idx_values)}
    N, C = z_ema.shape
    if N != len(indices):
        raise ValueError(f"indices length ({len(indices)}) must match first dim of z_ema ({N})")

    # initialize flags default kept -> will update below
    filter_flags = ["kept"] * len(train_csv_df)  # default keep, then set removed where rule says remove

    confidences = np.max(z_ema, axis=1)
    preds = np.argmax(z_ema, axis=1)

    # For each predicted sample, compare to label_noisy in CSV and decide
    for i_pos, idx in enumerate(indices):
        row_pos = idx_to_row.get(int(idx), None)
        if row_pos is None:
            # index not in CSV -> skip (shouldn't happen normally)
            logger.debug("Index %s from npz not found in train_csv_df; skipping", idx)
            continue
        label_noisy = int(train_csv_df.at[row_pos, "label_noisy"])
        pred_label = int(preds[i_pos])
        conf_val = float(confidences[i_pos])

        # Decision:
        # - If model predicts same as noisy label -> keep
        # - Else (model disagrees): remove only if tau is None (old behavior) OR conf >= tau
        if pred_label == label_noisy:
            filter_flags[row_pos] = "kept"
        else:
            if tau is None:
                # original behavior: remove all disagreements
                filter_flags[row_pos] = "removed"
            else:
                # new behavior: remove only high-confidence disagreements
                if conf_val >= float(tau):
                    filter_flags[row_pos] = "removed"
                else:
                    filter_flags[row_pos] = "kept"

    # Attach to copy
    train_csv_df = train_csv_df.copy()
    train_csv_df["filter_flag"] = filter_flags

    # Enforce min_keep_ratio per class by promoting top confident removed samples
    stats = {}
    if "class_name" in train_csv_df.columns:
        class_names = train_csv_df["class_name"].unique().tolist()
    else:
        # fallback to label_noisy classes
        class_names = sorted(train_csv_df["label_noisy"].unique().tolist())

    for cls in class_names:
        cls_mask = (train_csv_df["class_name"] == cls) if "class_name" in train_csv_df.columns else (train_csv_df["label_noisy"] == cls)
        cls_indices_df = train_csv_df[cls_mask]
        original_count = len(cls_indices_df)
        if original_count == 0:
            stats[cls] = {"original": 0, "kept": 0, "removed": 0, "kept_ratio": 0.0}
            continue
        kept_mask = (cls_indices_df["filter_flag"] == "kept")
        kept_count = int(kept_mask.sum())
        removed_count = int((~kept_mask).sum())
        required_min = math.ceil(min_keep_ratio * original_count)
        stats[cls] = {"original": original_count, "kept": kept_count, "removed": removed_count, "required_min": required_min}

        if kept_count < required_min:
            need = required_min - kept_count
            # Candidates: rows in this class currently removed
            removed_rows = cls_indices_df[~kept_mask]
            if removed_rows.empty:
                logger.warning("No removed rows available to promote for class %s", cls)
                continue
            # Build candidates with associated confidence on the sample's predicted prob for its current noisy label
            candidates = []
            for _, rr in removed_rows.iterrows():
                global_index = int(rr["index"])
                pos_list = np.where(indices == global_index)[0]
                if pos_list.size == 0:
                    # sample not present in indices (unlikely)
                    conf_val = 0.0
                else:
                    pos = pos_list[0]
                    # use max prob as a proxy (or prob of label_noisy)
                    try:
                        conf_val = float(np.max(z_ema[pos]))
                    except Exception:
                        conf_val = 0.0
                candidates.append((int(rr.name), conf_val))
            # sort candidates by conf val desc, promote top ones
            candidates.sort(key=lambda x: x[1], reverse=True)
            promote = candidates[:need]
            for rowpos, _ in promote:
                train_csv_df.at[rowpos, "filter_flag"] = "kept"
            logger.info("Promoted %d samples to 'kept' for class %s to reach min_keep_ratio", len(promote), cls)
            stats[cls]["kept"] += len(promote)
            stats[cls]["removed"] -= len(promote)
            stats[cls]["kept_ratio"] = stats[cls]["kept"] / original_count
        else:
            stats[cls]["kept_ratio"] = kept_count / original_count

    # Overall stats
    total_kept = int((train_csv_df["filter_flag"] == "kept").sum())
    total = len(train_csv_df)
    stats["_overall"] = {"total": total, "kept_total": total_kept, "kept_ratio_total": total_kept / total}
    return train_csv_df, stats
