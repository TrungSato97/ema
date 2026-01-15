"""
EDA plotting utilities for dataset and filtering reports.
"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
import seaborn as sns  # seaborn is optional but offers nicer heatmaps
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def plot_class_distribution(csv_paths: dict, save_path: str) -> None:
    """
    Plot class counts for train/val/test CSVs. csv_paths should be dict with keys 'train','val','test'
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    counts = {}
    for split in ["train", "val", "test"]:
        try:
            df = pd.read_csv(csv_paths[f"{split}_csv"])
        except Exception:
            df = pd.read_csv(csv_paths.get(split, ""))
        counts[split] = df["class_name"].value_counts().sort_index()
    classes = counts["train"].index.tolist()
    x = np.arange(len(classes))
    width = 0.25
    ax.bar(x - width, counts["train"].values, width=width, label="train")
    ax.bar(x, counts["val"].values, width=width, label="val")
    ax.bar(x + width, counts["test"].values, width=width, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Class distribution across splits")
    ax.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info("Saved class distribution plot to %s", save_path)


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str], save_path: str) -> None:
    """
    Save both raw and normalized confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax[0], xticklabels=class_names, yticklabels=class_names)
    ax[0].set_title("Confusion matrix (counts)")
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", ax=ax[1], xticklabels=class_names, yticklabels=class_names)
    ax[1].set_title("Confusion matrix (normalized)")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info("Saved confusion matrices to %s", save_path)


def plot_filter_ratios_over_iterations(summary_df: pd.DataFrame, save_path: str) -> None:
    """
    summary_df is expected to contain columns: iteration, class_name, kept_ratio
    We plot a heatmap (iterations x classes) of kept ratios.
    """
    pivot = summary_df.pivot(index="iteration", columns="class_name", values="kept_ratio")
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Kept ratio per class over iterations")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info("Saved filter ratios heatmap to %s", save_path)


def plot_confidence_histogram(confidences: np.ndarray, save_path: str) -> None:
    """
    confidences: 1D array of confidence values (0..1)
    """
    plt.figure(figsize=(6, 4))
    plt.hist(confidences, bins=50)
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Distribution of predicted confidences")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info("Saved confidence histogram to %s", save_path)
