"""
Configuration dataclass for the iterative EMA training pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch
import os
from typing import Literal, Optional

FilterMode = Literal["ema_hard", "vote_match_noisy", "vote_relabel"]
VotingSource = Literal["pred_hat", "pred_ema"]


@dataclass
class TrainConfig:
    
    # EMA & noise
    alpha: float = 0.9  # EMA factor
    noise_ratio: float = 0.8  # fraction of training labels to corrupt
    min_keep_ratio: float = 0.05  # per-class minimum keep fraction
    
    # ---------------- NEW: selection policy ----------------
    filter_mode: FilterMode = "ema_hard"

    
    voting_k: int = 3
    voting_agree_min: int = 2
    voting_source: VotingSource = "pred_hat"
    voting_start_iter: int = 2  # i < start_iter => fallback to ema_hard

    # For training labels
    train_label_col_default: str = "label_noisy"   # default training label
    train_label_col_relabel: str = "label_train"  # for vote_relabel after voting starts

    
    # Experiment identity
    exp_name: str = "cifar10_iter_ema_noise_ver_add_voting"
    seed: int = 42

    # Paths
    data_dir: str = f"data"
    output_dir: str = "outputs"
    output_eda: str = "eda_data"
    exp_dir: str = "cifar10_iter_ema_noise_ver_add_voting"  # will be set by make_exp_dir()

    # Data
    img_size: int = 224
    split_train: float = 0.9
    split_val: float = 0.1
    # split_test: float = 0.1

    # Training
    batch_size: int = 256
    num_workers: int = 4
    max_iterations: int = 20
    max_epochs_per_iter: int = 200
    patience_epoch: int = 5  # early stopping inside iteration
    patience_iter: int = 3  # early stopping across iterations
    save_every_n_epochs: int = 1000
    
    # Training selection metric (inside iteration)
    early_stop_metric: str = "val_acc_noisy"  # or "val_acc_orig"


    # Optimizer & LR
    optimizer: str = "sgd"  # 'sgd' or 'adamw'
    lr: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9



    # Device & precision
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_amp: bool = field(default_factory=lambda: torch.cuda.is_available())

    # Resume
    resume: bool = False

    # Misc
    num_classes: int = 10
    verbose: bool = True

    def validate(self) -> None:
        """
        Validate config values; raise ValueError on invalid combos.
        """
        if not (0.0 <= self.noise_ratio <= 1.0):
            raise ValueError("noise_ratio must be in [0,1]")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0,1]")
        if not (0.0 <= self.min_keep_ratio <= 1.0):
            raise ValueError("min_keep_ratio must be in [0,1]")
        if self.max_epochs_per_iter <= 0 or self.max_iterations <= 0:
            raise ValueError("max_epochs_per_iter and max_iterations must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.split_train + self.split_val != 1.0:
            # Allow small rounding diff
            total = self.split_train + self.split_val + self.split_test
            if abs(total - 1.0) > 1e-6:
                raise ValueError("train/val/test splits must sum to 1.0")

    def make_exp_dir(self) -> str:
        """
        Create and return experiment output directory string, set exp_dir attr.
        """
        base = os.path.join(self.output_dir, self.exp_name)
        os.makedirs(base, exist_ok=True)
        self.exp_dir = base
        return base

    def get_train_label_col(self, iter_idx: int) -> str:
        """
        Decide which column is used as training label for current iteration's train_loader.
        - ema_hard and vote_match_noisy: always label_noisy
        - vote_relabel: after voting_start_iter, use label_train
        """
        if self.filter_mode == "vote_relabel" and iter_idx >= self.voting_start_iter:
            return self.train_label_col_relabel
        return self.train_label_col_default