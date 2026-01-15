"""
Filtering utilities: apply EMA-based filtering and enforce per-class minimum keep ratio.
"""
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import logging
import math
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import math

logger = logging.getLogger(__name__)


def filter_by_ema(indices: np.ndarray, z_ema: np.ndarray, train_csv_df: pd.DataFrame,
                  min_keep_ratio: float) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply filtering based on EMA predictions (z_ema) compared with label_noisy in train_csv_df.
    - indices: (N,) array of CSV 'index' values corresponding to rows in z_ema
    - z_ema: (N, C) array of EMA probabilities
    - train_csv_df: pd.DataFrame of train CSV
    - min_keep_ratio: float in (0,1)

    Returns:
      updated_df: original train_csv_df with additional column 'filter_flag' ('kept'/'removed')
      stats: dict containing per-class counts and kept ratios
    """

    if "index" not in train_csv_df.columns:
        raise ValueError("train_csv_df must contain 'index' column")

    # Build mapping from index -> row position in train_csv_df
    idx_to_row = {int(v): i for i, v in enumerate(train_csv_df["index"].values)}
    N, C = z_ema.shape
    # sanity check indices
    if N != len(indices):
        raise ValueError("indices length must match first dim of z_ema")
    # initialize flags default kept -> will update below
    filter_flags = ["removed"] * len(train_csv_df)
    confidences = np.max(z_ema, axis=1)
    preds = np.argmax(z_ema, axis=1)

    # For each predicted sample, compare to label_noisy in CSV and mark kept if match
    for i_pos, idx in enumerate(indices):
        row_pos = idx_to_row.get(int(idx), None)
        if row_pos is None:
            # index not in CSV -> skip (shouldn't happen)
            continue
        label_noisy = int(train_csv_df.at[row_pos, "label_noisy"])
        if int(preds[i_pos]) == label_noisy:
            filter_flags[row_pos] = "kept"
        else:
            filter_flags[row_pos] = "removed"

    # Build DataFrame columns to attach
    train_csv_df = train_csv_df.copy()
    train_csv_df["filter_flag"] = filter_flags

    # Enforce min_keep_ratio per class by promoting top confident removed samples
    stats = {}
    class_names = train_csv_df["class_name"].unique().tolist()
    for cls in class_names:
        cls_mask = train_csv_df["class_name"] == cls
        cls_indices = train_csv_df[cls_mask]
        original_count = len(cls_indices)
        if original_count == 0:
            stats[cls] = {"original": 0, "kept": 0, "removed": 0, "kept_ratio": 0.0}
            continue
        kept_mask = (cls_indices["filter_flag"] == "kept")
        kept_count = int(kept_mask.sum())
        removed_count = int((~kept_mask).sum())
        required_min = math.ceil(min_keep_ratio * original_count)
        stats[cls] = {"original": original_count, "kept": kept_count, "removed": removed_count, "required_min": required_min}

        if kept_count < required_min:
            need = required_min - kept_count
            # Candidates: rows in this class currently removed
            removed_rows = cls_indices[~kept_mask]
            if removed_rows.empty:
                logger.warning("No removed rows available to promote for class %s", cls)
                continue
            # For each removed sample, find its index in indices to get EMA confidence for that class
            # Build list of tuples (row_global_pos, conf_for_class)
            candidates = []
            for _, rr in removed_rows.iterrows():
                global_index = int(rr["index"])
                # find position in indices array
                pos_list = np.where(indices == global_index)[0]
                if pos_list.size == 0:
                    # sample not present in indices (unlikely)
                    conf_val = 0.0
                else:
                    pos = pos_list[0]
                    conf_val = float(z_ema[pos, int(rr["label_noisy"])])
                candidates.append((int(rr.name), conf_val))
            # sort candidates by conf val desc
            candidates.sort(key=lambda x: x[1], reverse=True)
            promote = candidates[:need]
            for rowpos, _ in promote:
                train_csv_df.at[rowpos, "filter_flag"] = "kept"
            logger.info("Promoted %d samples to 'kept' for class %s to reach min_keep_ratio", len(promote), cls)
            # update stats
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


def _build_index_maps(train_df: pd.DataFrame, indices: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Returns:
      idx_to_row: map index -> row position in train_df
      idx_to_pos: map index -> position in indices/z arrays
    """
    idx_to_row = {int(v): i for i, v in enumerate(train_df["index"].values)}
    idx_to_pos = {int(v): i for i, v in enumerate(indices.astype(np.int32))}
    return idx_to_row, idx_to_pos


def apply_keep_mask_with_min_keep(
    train_df: pd.DataFrame,
    indices: np.ndarray,
    kept_mask: np.ndarray,
    score_noisy: np.ndarray,
    min_keep_ratio: float,
    extra_cols_by_pos: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply kept_mask (aligned with indices) to train_df, then enforce per-class min_keep_ratio by promoting removed samples.
    Promotion uses score_noisy (aligned with indices): typically z_ema[pos, label_noisy].

    extra_cols_by_pos: dict name -> array aligned with indices to attach to df (e.g., vote_class, vote_agreement, pred_hat,...)
    """
    if "index" not in train_df.columns:
        raise ValueError("train_df must contain 'index'")
    if "label_noisy" not in train_df.columns:
        raise ValueError("train_df must contain 'label_noisy'")
    if len(indices) != len(kept_mask) or len(indices) != len(score_noisy):
        raise ValueError("indices, kept_mask, score_noisy must have same length")

    train_df = train_df.copy()
    idx_to_row, idx_to_pos = _build_index_maps(train_df, indices)

    # init removed
    train_df["filter_flag"] = "removed"
    train_df["promoted_flag"] = 0

    # set kept according to kept_mask
    for idx, is_kept in zip(indices.astype(np.int32), kept_mask.astype(bool)):
        row = idx_to_row.get(int(idx))
        if row is None:
            continue
        train_df.at[row, "filter_flag"] = "kept" if is_kept else "removed"

    # attach extra columns
    if extra_cols_by_pos:
        for col, arr in extra_cols_by_pos.items():
            if len(arr) != len(indices):
                raise ValueError(f"extra col '{col}' length mismatch with indices")
            # create with NaN first
            train_df[col] = np.nan
            for idx in indices.astype(np.int32):
                row = idx_to_row.get(int(idx))
                pos = idx_to_pos.get(int(idx))
                if row is None or pos is None:
                    continue
                train_df.at[row, col] = arr[pos]

    # enforce min_keep_ratio per class
    stats: Dict[str, Dict] = {}
    class_names = train_df["class_name"].unique().tolist()
    for cls in class_names:
        cls_mask = train_df["class_name"] == cls
        cls_df = train_df[cls_mask]
        n_total = len(cls_df)
        if n_total == 0:
            stats[cls] = {"original": 0, "kept": 0, "removed": 0, "required_min": 0, "kept_ratio": 0.0}
            continue

        kept_cnt = int((cls_df["filter_flag"] == "kept").sum())
        required_min = int(math.ceil(float(min_keep_ratio) * n_total))

        if kept_cnt < required_min:
            need = required_min - kept_cnt
            removed_rows = cls_df[cls_df["filter_flag"] == "removed"]
            if len(removed_rows) > 0:
                # rank removed by score_noisy descending
                candidates = []
                for row_idx, rr in removed_rows.iterrows():
                    idx_val = int(rr["index"])
                    pos = idx_to_pos.get(idx_val)
                    s = float(score_noisy[pos]) if pos is not None else 0.0
                    candidates.append((row_idx, s))
                candidates.sort(key=lambda x: x[1], reverse=True)
                promote = candidates[:need]
                for row_idx, _ in promote:
                    train_df.at[row_idx, "filter_flag"] = "kept"
                    train_df.at[row_idx, "promoted_flag"] = 1
                kept_cnt += len(promote)

        removed_cnt = n_total - kept_cnt
        stats[cls] = {
            "original": n_total,
            "kept": kept_cnt,
            "removed": removed_cnt,
            "required_min": required_min,
            "kept_ratio": kept_cnt / max(1, n_total),
        }

    total_kept = int((train_df["filter_flag"] == "kept").sum())
    total = int(len(train_df))
    stats["_overall"] = {
        "total": total,
        "kept_total": total_kept,
        "kept_ratio_total": total_kept / max(1, total),
    }
    return train_df, stats