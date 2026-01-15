# src/selection_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class VotingResult:
    vote_class: np.ndarray        # (N,)
    vote_agreement: np.ndarray    # (N,) number of votes for the majority class
    kept_mask: np.ndarray         # (N,) bool
    label_train: np.ndarray       # (N,) int - label to write into label_train (for kept samples)


def majority_vote(pred_history: List[np.ndarray], num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    pred_history: list of arrays (N,), length=k
    Returns:
      vote_class (N,), vote_agreement (N,)
    """
    if len(pred_history) == 0:
        raise ValueError("pred_history must be non-empty")

    k = len(pred_history)
    N = pred_history[0].shape[0]
    for p in pred_history:
        if p.shape[0] != N:
            raise ValueError("All pred arrays must have same length")

    # Stack (k, N)
    stack = np.stack(pred_history, axis=0)  # (k, N)
    # vote counts per sample (N, num_classes) via bincount in loop (small k, so OK)
    vote_class = np.zeros(N, dtype=np.int32)
    vote_agree = np.zeros(N, dtype=np.int32)

    for i in range(N):
        counts = np.bincount(stack[:, i], minlength=num_classes)  # safe upper bound
        cls = int(np.argmax(counts))
        agree = int(counts[cls])
        vote_class[i] = cls
        vote_agree[i] = agree

    return vote_class, vote_agree


def build_kept_and_labels(
    vote_class: np.ndarray,
    vote_agreement: np.ndarray,
    label_noisy: np.ndarray,
    agree_min: int,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    mode:
      - "vote_match_noisy": kept if agreement>=agree_min AND vote_class==label_noisy
        label_train = label_noisy
      - "vote_relabel": kept if agreement>=agree_min
        label_train = vote_class
    """
    if mode not in {"vote_match_noisy", "vote_relabel"}:
        raise ValueError(f"Unknown mode: {mode}")

    agree_ok = vote_agreement >= int(agree_min)

    if mode == "vote_match_noisy":
        kept = agree_ok & (vote_class == label_noisy)
        label_train = label_noisy.copy()
        return kept, label_train

    kept = agree_ok
    label_train = vote_class.copy()
    return kept, label_train
