# notebook/iterative_driver_voting.py
from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import logging

from src.config import TrainConfig
from src.dataset_utils import make_data_loaders
from src.train_loop import train_iteration
from src.ema_utils import update_ema
from src.filter_utils import apply_keep_mask_with_min_keep
from src.selection_utils import majority_vote, build_kept_and_labels

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -------------------------
# Small helpers
# -------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _save_preds_class_npz(
    path: Path,
    indices: np.ndarray,
    pred_hat: np.ndarray,
    pred_ema: np.ndarray,
    conf_hat: np.ndarray,
    conf_ema: np.ndarray,
) -> None:
    _ensure_dir(path.parent)
    np.savez_compressed(
        str(path),
        indices=indices.astype(np.int32),
        pred_hat=pred_hat.astype(np.int16),
        pred_ema=pred_ema.astype(np.int16),
        conf_hat=conf_hat.astype(np.float16),
        conf_ema=conf_ema.astype(np.float16),
    )


def _save_pred_history_npz(
    path: Path,
    indices: np.ndarray,
    pred_hat_hist: List[np.ndarray],
    pred_ema_hist: List[np.ndarray],
) -> None:
    """
    Save full history up to current iteration:
      pred_hat_hist_stack: (T, N)
      pred_ema_hist_stack: (T, N)
    This helps you verify voting correctness offline.
    """
    _ensure_dir(path.parent)

    # stack to (T, N)
    hat_stack = np.stack(pred_hat_hist, axis=0).astype(np.int16)
    ema_stack = np.stack(pred_ema_hist, axis=0).astype(np.int16)

    np.savez_compressed(
        str(path),
        indices=indices.astype(np.int32),
        pred_hat_hist=hat_stack,
        pred_ema_hist=ema_stack,
    )


def _compute_score_noisy(indices: np.ndarray, z_ema: np.ndarray, train_df: pd.DataFrame) -> np.ndarray:
    """
    score_noisy[pos] = z_ema[pos, label_noisy(sample)]
    Used for min_keep promotions.
    """
    idx_to_row = {int(v): i for i, v in enumerate(train_df["index"].values)}
    score = np.zeros(len(indices), dtype=np.float32)

    for pos, idx in enumerate(indices.astype(np.int32)):
        row = idx_to_row.get(int(idx))
        if row is None:
            continue
        y_noisy = int(train_df.at[row, "label_noisy"])
        score[pos] = float(z_ema[pos, y_noisy])

    return score


def _safe_float(x, default=-1.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


# -------------------------
# Core experiment runner
# -------------------------
def run_experiment_for_setting(
    config: TrainConfig,
    train_csv: str,
    val_csv: str,
    test_csv: str,
) -> None:
    """
    One full run for a fixed (noise_ratio, alpha, filter_mode, voting params...).
    Outputs (inside config.exp_dir):
      - manifest.json (run-level config + timing)
      - experiment_summary.csv (iteration-level summary + timing)
      - iteration_i/
          - preds_npz/preds_iter_i.npz            (z_hat, z_ema)
          - preds_npz/preds_class_iter_i.npz      (pred_hat/pred_ema + conf)
          - preds_npz/pred_history_upto_i.npz     (history stacks to verify voting)
          - selection/selection_iter_i.csv        (report-friendly selection table)
          - train_kept_i.csv                      (train set for next iteration)
    """
    config.validate()

    exp_dir = Path(config.exp_dir)
    _ensure_dir(exp_dir)

    # Run-level timers
    run_t0 = time.perf_counter()
    run_started_at = _now_str()

    # Load full train universe once
    original_train_df = pd.read_csv(train_csv)
    if "index" not in original_train_df.columns:
        raise ValueError("train_csv must contain 'index' column")
    if "label_noisy" not in original_train_df.columns:
        raise ValueError("train_csv must contain 'label_noisy' column")
    N_full = int(len(original_train_df))

    # Build fast mapping for label_noisy by index (avoid set_index every iteration)
    label_noisy_map = dict(zip(original_train_df["index"].astype(int).tolist(),
                               original_train_df["label_noisy"].astype(int).tolist()))

    # Manifest (run-level meta)
    manifest: Dict[str, Any] = {
        "run_started_at": run_started_at,
        "exp_dir": str(exp_dir),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "seed": int(config.seed),
        "noise_ratio": float(config.noise_ratio),
        "alpha": float(config.alpha),
        "min_keep_ratio": float(config.min_keep_ratio),

        "filter_mode": str(config.filter_mode),
        "voting_k": int(config.voting_k),
        "voting_agree_min": int(config.voting_agree_min),
        "voting_source": str(config.voting_source),
        "voting_start_iter": int(config.voting_start_iter),

        "max_iterations": int(config.max_iterations),
        "max_epochs_per_iter": int(config.max_epochs_per_iter),
        "patience_epoch": int(config.patience_epoch),
        "patience_iter": int(config.patience_iter),
        "early_stop_metric": str(getattr(config, "early_stop_metric", "val_acc_noisy")),
    }
    _save_json(exp_dir / "manifest.json", manifest)

    # Iteration summaries
    experiment_summary_rows: List[Dict[str, Any]] = []
    experiment_summary_path = exp_dir / "experiment_summary.csv"

    # EMA state
    z_ema_prev: Optional[np.ndarray] = None

    # Prediction history (aligned with indices order of train_full_loader)
    pred_hat_hist: List[np.ndarray] = []
    pred_ema_hist: List[np.ndarray] = []

    best_overall_val = -1.0
    iter_no_improve = 0

    for i in range(int(config.max_iterations)):
        iter_t0 = time.perf_counter()
        iter_started_at = _now_str()

        logger.info("=== Iteration %d | filter_mode=%s | alpha=%.4f | noise=%.2f ===",
                    i, config.filter_mode, float(config.alpha), float(config.noise_ratio))

        # Choose training csv (filtered kept from prev iter)
        if i == 0:
            current_train_csv = train_csv
        else:
            prev_kept_csv = exp_dir / f"iteration_{i-1}" / f"train_kept_{i-1}.csv"
            current_train_csv = str(prev_kept_csv) if prev_kept_csv.exists() else train_csv

        # Choose label col for train_loader (for vote_relabel after voting_start_iter)
        train_label_col = config.get_train_label_col(iter_idx=i)

        # Build dataloaders
        dls = make_data_loaders(
            train_csv=current_train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            config=config,
            train_full_csv=train_csv,          # CRITICAL: keep full-universe stable order
            train_label_col=train_label_col,
        )
        train_loader = dls["train"]
        val_loader = dls["val"]
        test_loader = dls["test"]
        train_full_loader = dls["train_full"]

        training_samples_used = int(len(train_loader.dataset))
        logger.info("Train samples used (iter %d): %d/%d (train_label_col=%s)",
                    i, training_samples_used, N_full, train_label_col)

        # -------- train iteration (timed) --------
        train_t0 = time.perf_counter()
        result = train_iteration(
            iter_idx=i,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_full_loader=train_full_loader,
            start_epoch=0,
        )
        train_t1 = time.perf_counter()
        iter_train_seconds = float(train_t1 - train_t0)

        indices = result["z_hat_indices"].astype(np.int32)
        z_hat = result["z_hat"].astype(np.float32)

        # Update EMA
        z_ema = update_ema(z_ema_prev, z_hat, float(config.alpha))
        z_ema_prev = z_ema

        # Compute classes/confidences
        pred_hat = np.argmax(z_hat, axis=1).astype(np.int32)
        pred_ema = np.argmax(z_ema, axis=1).astype(np.int32)
        conf_hat = np.max(z_hat, axis=1).astype(np.float32)
        conf_ema = np.max(z_ema, axis=1).astype(np.float32)

        pred_hat_hist.append(pred_hat)
        pred_ema_hist.append(pred_ema)

        # Iteration dirs
        iter_dir = exp_dir / f"iteration_{i}"
        _ensure_dir(iter_dir / "preds_npz")
        _ensure_dir(iter_dir / "selection")

        # Save z_hat/z_ema
        np.savez_compressed(
            str(iter_dir / "preds_npz" / f"preds_iter_{i}.npz"),
            indices=indices.astype(np.int32),
            z_hat=z_hat.astype(np.float32),
            z_ema=z_ema.astype(np.float32),
        )

        # Save class/conf
        _save_preds_class_npz(
            iter_dir / "preds_npz" / f"preds_class_iter_{i}.npz",
            indices, pred_hat, pred_ema, conf_hat, conf_ema
        )

        # Save history up to this iteration (for verifying voting correctness)
        _save_pred_history_npz(
            iter_dir / "preds_npz" / f"pred_history_upto_{i}.npz",
            indices, pred_hat_hist, pred_ema_hist
        )

        # -------- selection step (timed) --------
        sel_t0 = time.perf_counter()

        # label_noisy aligned with indices order
        label_noisy_full = np.array([label_noisy_map[int(idx)] for idx in indices.tolist()], dtype=np.int32)

        use_voting = (i >= int(config.voting_start_iter)) and (config.filter_mode in {"vote_match_noisy", "vote_relabel"})

        if not use_voting:
            # ema_hard fallback
            kept_mask = (pred_ema == label_noisy_full)
            vote_class = np.full_like(label_noisy_full, fill_value=-1, dtype=np.int32)
            vote_agreement = np.zeros_like(label_noisy_full, dtype=np.int32)
            label_train_arr = label_noisy_full.copy()
        else:
            # pick history source
            hist = pred_hat_hist if config.voting_source == "pred_hat" else pred_ema_hist

            k = int(config.voting_k)
            # Safety: if user sets weird start_iter < k-1, don't crash
            k_eff = min(k, len(hist))
            last_k = hist[-k_eff:]  # list length k_eff

            # vote_class, vote_agreement = majority_vote(last_k)
            # vote_class, vote_agreement = majority_vote(last_k, num_classes=config.num_classes)
            try:
                vote_class, vote_agreement = majority_vote(last_k, num_classes=config.num_classes)
            except TypeError:
                vote_class, vote_agreement = majority_vote(last_k)


            kept_mask, label_train_arr = build_kept_and_labels(
                vote_class=vote_class,
                vote_agreement=vote_agreement,
                label_noisy=label_noisy_full,
                agree_min=int(config.voting_agree_min),
                mode=str(config.filter_mode),
            )

        # score for promotions: z_ema prob of label_noisy
        score_noisy = _compute_score_noisy(indices, z_ema, original_train_df)

        # attach report columns aligned with indices
        extra_cols = {
            "pred_hat": pred_hat,
            "pred_ema": pred_ema,
            "conf_hat": conf_hat,
            "conf_ema": conf_ema,
            "vote_class": vote_class,
            "vote_agreement": vote_agreement,
            "score_noisy": score_noisy,
            "label_train": label_train_arr,
        }

        updated_df, stats = apply_keep_mask_with_min_keep(
            train_df=original_train_df,
            indices=indices,
            kept_mask=kept_mask,
            score_noisy=score_noisy,
            min_keep_ratio=float(config.min_keep_ratio),
            extra_cols_by_pos=extra_cols,
        )

        kept_samples = int((updated_df["filter_flag"] == "kept").sum())
        removed_samples = int((updated_df["filter_flag"] == "removed").sum())
        kept_ratio = float(stats["_overall"]["kept_ratio_total"])

        # Save selection table (report-friendly)
        sel_cols = [
            "index", "class_name", "label_noisy", "label_orig", "noise_flag",
            "pred_hat", "pred_ema", "vote_class", "vote_agreement",
            "score_noisy", "label_train", "filter_flag", "promoted_flag",
        ]
        sel_cols = [c for c in sel_cols if c in updated_df.columns]
        selection_df = updated_df[sel_cols].copy()
        selection_df.to_csv(iter_dir / "selection" / f"selection_iter_{i}.csv", index=False)

        # Save train_kept for next iteration
        train_kept_df = updated_df[updated_df["filter_flag"] == "kept"].copy()
        train_kept_path = iter_dir / f"train_kept_{i}.csv"
        train_kept_df.to_csv(train_kept_path, index=False)

        sel_t1 = time.perf_counter()
        iter_selection_seconds = float(sel_t1 - sel_t0)

        # -------- iteration timing --------
        iter_finished_at = _now_str()
        iter_t1 = time.perf_counter()
        iter_total_seconds = float(iter_t1 - iter_t0)

        # Stats for summary
        mean_agree = float(np.mean(vote_agreement)) if use_voting else 0.0
        num_relabelled = int(np.sum(label_train_arr != label_noisy_full)) if (use_voting and config.filter_mode == "vote_relabel") else 0

        val_score = _safe_float(result.get("val_acc_noisy", None), default=-1.0)

        summary_row = {
            # ids
            "noise_ratio": float(config.noise_ratio),
            "alpha": float(config.alpha),
            "iteration": int(i),

            # method
            "filter_mode": str(config.filter_mode),
            "voting_k": int(config.voting_k),
            "voting_agree_min": int(config.voting_agree_min),
            "voting_source": str(config.voting_source),
            "voting_start_iter": int(config.voting_start_iter),

            # selection stats
            "kept_ratio": float(kept_ratio),
            "samples_kept": int(kept_samples),
            "samples_removed": int(removed_samples),
            "samples_total": int(N_full),
            "training_samples_used": int(training_samples_used),

            "mean_vote_agreement": float(mean_agree),
            "num_relabelled": int(num_relabelled),

            # metrics
            "val_acc_reported": result.get("best_val_acc", None),
            "test_acc_reported": result.get("test_acc", None),
            "val_acc_orig": result.get("val_acc_orig", None),
            "test_acc_orig": result.get("test_acc_orig", None),
            "val_acc_noisy": result.get("val_acc_noisy", None),
            "test_acc_noisy": result.get("test_acc_noisy", None),
            "summary_train_full_acc_noisy": result.get("train_full_acc_noisy", None),
            "summary_train_full_acc_orig": result.get("train_full_acc_orig", None),

            # timing
            "iter_train_seconds": float(iter_train_seconds),
            "iter_selection_seconds": float(iter_selection_seconds),
            "iter_total_seconds": float(iter_total_seconds),
            "iter_started_at": iter_started_at,
            "iter_finished_at": iter_finished_at,
        }

        experiment_summary_rows.append(summary_row)
        pd.DataFrame(experiment_summary_rows).to_csv(experiment_summary_path, index=False)

        is_voting_mode = config.filter_mode in {"vote_match_noisy", "vote_relabel"}
        if is_voting_mode:
            # Before voting starts: do not early-stop across iterations
            if i < config.voting_start_iter:
                # (optional) you may log that early-stop is disabled here
                logger.info(
                    "Iter %d: voting not started yet (voting_start_iter=%d) -> skip early-stop across iterations.",
                    i, config.voting_start_iter
                )
            else:
                # First iteration where voting is active: initialize tracking baseline
                if i == config.voting_start_iter and best_overall_val < 0:
                    best_overall_val = val_score
                    iter_no_improve = 0
                else:
                    if val_score > best_overall_val:
                        best_overall_val = val_score
                        iter_no_improve = 0
                    else:
                        iter_no_improve += 1

                # allow stop only after we've had enough voting iterations
                if i >= (config.voting_start_iter + config.patience_iter) and iter_no_improve >= config.patience_iter:
                    logger.info(
                        "Stop across iterations due to no improvement (voting-mode). "
                        "i=%d start=%d patience=%d best=%.4f current=%.4f",
                        i, config.voting_start_iter, config.patience_iter, best_overall_val, val_score
                    )
                    break
        else:
            # ema_hard: original behavior
            if val_score > best_overall_val:
                best_overall_val = val_score
                iter_no_improve = 0
            else:
                iter_no_improve += 1
                if iter_no_improve >= config.patience_iter:
                    logger.info(
                        "Stop across iterations due to no improvement (ema_hard). "
                        "i=%d patience=%d best=%.4f current=%.4f",
                        i, config.patience_iter, best_overall_val, val_score
                    )
                    break

    # -------- finalize run-level timing --------
    run_t1 = time.perf_counter()
    run_finished_at = _now_str()
    run_total_seconds = float(run_t1 - run_t0)

    manifest["run_finished_at"] = run_finished_at
    manifest["run_total_seconds"] = run_total_seconds
    manifest["best_overall_val_acc_noisy"] = float(best_overall_val)

    _save_json(exp_dir / "manifest.json", manifest)

    logger.info(
        "RUN DONE: mode=%s noise=%.2f alpha=%.4f | best_val_noisy=%.4f | total_time=%.1fs | exp_dir=%s",
        str(config.filter_mode), float(config.noise_ratio), float(config.alpha),
        float(best_overall_val), float(run_total_seconds), str(exp_dir)
    )


def main() -> None:
    """
    Optional CLI-like entry.
    In notebook you typically call run_experiment_for_setting(...) directly.
    """
    config = TrainConfig()
    base_exp_dir = Path(config.make_exp_dir())

    noise_ratio = 0.8
    alpha = 0.3
    config.noise_ratio = float(noise_ratio)
    config.alpha = float(alpha)

    config.filter_mode = "vote_match_noisy"
    config.voting_k = 3
    config.voting_agree_min = 2
    config.voting_source = "pred_hat"
    config.voting_start_iter = 2

    csv_dir = Path(config.data_dir) / "csvs" / f"noise_{noise_ratio}"
    train_csv = str(csv_dir / "train.csv")
    val_csv = str(csv_dir / "val.csv")
    test_csv = str(csv_dir / "test.csv")

    config.exp_dir = str(base_exp_dir / f"noise_{noise_ratio}" / f"alpha_{alpha}" / f"mode_{config.filter_mode}")
    os.makedirs(config.exp_dir, exist_ok=True)

    run_experiment_for_setting(config, train_csv, val_csv, test_csv)


if __name__ == "__main__":
    main()
