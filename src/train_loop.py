# train_loop.py (replace these functions)


from typing import Tuple, Dict, Optional
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
import logging
from .config import TrainConfig
from .io_utils import save_checkpoint, save_dataframe_csv, append_row_to_csv, make_dirs
from .model_utils import get_model, get_optimizer_scheduler
from pathlib import Path
from .seed_utils import set_global_seed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, device: str, use_amp: bool) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Run one training epoch.
    Returns:
        avg_loss, acc_noisy, acc_orig
        - avg_loss: float (0.0 if no samples)
        - acc_noisy: float or None (None if no samples)
        - acc_orig: float or None (None if no 'label_orig' present in any batch or no samples)
    """
    model.train()
    total_loss = 0.0
    correct_noisy = 0
    correct_orig = 0
    total = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.startswith("cuda") else None

    has_label_orig = False

    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch in pbar:
        inputs = batch["image"].to(device, non_blocking=True)
        labels_noisy = batch["label"].to(device, non_blocking=True)

        labels_orig = None
        if "label_orig" in batch:
            labels_orig = batch["label_orig"].to(device, non_blocking=True)
            has_label_orig = True

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels_noisy)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels_noisy)
            loss.backward()
            optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size

        preds = outputs.argmax(dim=1)
        correct_noisy += int((preds == labels_noisy).sum().item())
        if labels_orig is not None:
            correct_orig += int((preds == labels_orig).sum().item())

        total += batch_size

        # Show running stats (noisy acc)
        noisy_acc_running = (correct_noisy / total) if total > 0 else 0.0
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{noisy_acc_running:.4f}")

    # finalize metrics safely
    if total > 0:
        avg_loss = total_loss / total
        acc_noisy = correct_noisy / total
        acc_orig = (correct_orig / total) if has_label_orig else None
    else:
        # no samples in dataloader
        avg_loss = 0.0
        acc_noisy = None
        acc_orig = None

    return avg_loss, acc_noisy, acc_orig



def validate_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                   criterion: Optional[nn.Module], device: str) -> Tuple[float, float, Optional[float], np.ndarray, np.ndarray]:
    """
    Evaluate model on dataloader. Returns loss, acc_noisy, acc_orig, probs (N,C), preds (N,)
    acc_orig is None if no 'label_orig' available.
    """
    model.eval()
    total_loss = 0.0
    correct_noisy = 0
    correct_orig = 0
    total = 0
    all_probs = []
    all_preds = []
    has_label_orig = False

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validate", leave=False):
            inputs = batch["image"].to(device, non_blocking=True)
            labels_noisy = batch.get("label", None)
            if labels_noisy is not None:
                labels_noisy = labels_noisy.to(device, non_blocking=True)
            labels_orig = batch.get("label_orig", None)
            if labels_orig is not None:
                labels_orig = labels_orig.to(device, non_blocking=True)
                has_label_orig = True

            outputs = model(inputs)
            if criterion is not None and labels_noisy is not None:
                loss = criterion(outputs, labels_noisy)
                total_loss += loss.item() * inputs.size(0)

            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

            if labels_noisy is not None:
                correct_noisy += (preds == labels_noisy).sum().item()
            if labels_orig is not None:
                correct_orig += (preds == labels_orig).sum().item()
            total += inputs.size(0)

    avg_loss = total_loss / total if (criterion is not None and total > 0) else 0.0
    acc_noisy = (correct_noisy / total) if total > 0 and labels_noisy is not None else None
    acc_orig = (correct_orig / total) if total > 0 and has_label_orig else None
    probs_np = np.concatenate(all_probs, axis=0) if len(all_probs) > 0 else np.empty((0,))
    preds_np = np.concatenate(all_preds, axis=0) if len(all_preds) > 0 else np.empty((0,))
    return avg_loss, acc_noisy, acc_orig, probs_np, preds_np


def predict_on_loader(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                      device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict probabilities on entire dataloader (no labels required).
    Returns:
        indices: np.ndarray shape (N,) of sample indices (from dataset csv 'index' field)
        probs: np.ndarray shape (N, C)
    """
    model.eval()
    all_indices = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predict", leave=False):
            inputs = batch["image"].to(device, non_blocking=True)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_indices.append(np.array(batch["index"], dtype=np.int32))
    probs_np = np.concatenate(all_probs, axis=0)
    indices_np = np.concatenate(all_indices, axis=0)
    return indices_np, probs_np

def predict_with_acc(model, dataloader, device):
    model.eval()
    all_indices = []
    all_probs = []
    correct_noisy = 0
    correct_orig = 0
    total = 0
    has_label_orig = False

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="PredictFull", leave=False):
            inputs = batch["image"].to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            preds = probs.argmax(dim=1)
            if "label" in batch:
                labels_noisy = batch["label"].to(device)
                correct_noisy += (preds == labels_noisy).sum().item()
            if "label_orig" in batch:
                labels_orig = batch["label_orig"].to(device)
                correct_orig += (preds == labels_orig).sum().item()
                has_label_orig = True

            total += inputs.size(0)

            all_indices.append(batch["index"].cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    acc_noisy = correct_noisy / total if total > 0 else None
    acc_orig = correct_orig / total if total > 0 and has_label_orig else None

    return (
        np.concatenate(all_indices, axis=0),
        np.concatenate(all_probs, axis=0),
        acc_noisy,
        acc_orig
    )



def train_iteration(iter_idx: int, config: TrainConfig, train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader,
                    train_full_loader: torch.utils.data.DataLoader,
                    start_epoch: int = 0, resume_checkpoint: Optional[str] = None) -> Dict:
    """
    Train one iteration: initialize fresh pretrained model, train with early stopping (patience_epoch),
    save checkpoints (best, every N epochs, last), and return paths and metrics.
    """
    set_global_seed(config.seed, deterministic=True)

    device = config.device
    model = get_model(num_classes=config.num_classes, pretrained=True, device=device)
    optimizer, scheduler = get_optimizer_scheduler(model, config, total_epochs=config.max_epochs_per_iter)
    criterion = nn.CrossEntropyLoss()

    # Optionally resume
    epoch_start = max(0, start_epoch)
    if resume_checkpoint:
        from .io_utils import load_checkpoint
        ck = load_checkpoint(resume_checkpoint, model, optimizer, scheduler, map_location=device)
        epoch_start = int(ck.get("epoch", epoch_start))
        logger.info("Resumed iteration %d from epoch %d", iter_idx, epoch_start)

    best_val_acc_orig = -1.0
    best_val_acc_noisy = -1.0
    best_model_path = None
    metrics_rows = []
    early_stop_counter = 0

    # prepare iteration output paths
    from .io_utils import make_dirs, save_checkpoint
    iter_base = Path(config.exp_dir) / f"iteration_{iter_idx}"
    ck_dir = iter_base / "checkpoints"
    make_dirs(str(ck_dir / ".."))  # ensure parent exists
    make_dirs(str(ck_dir))

    for epoch in range(epoch_start, config.max_epochs_per_iter):
        t0 = time.time()
        train_loss, train_acc_noisy, train_acc_orig = train_epoch(model, train_loader, criterion, optimizer, device, config.use_amp)
        val_loss, val_acc_noisy, val_acc_orig, _, _ = validate_epoch(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]['lr'] if optimizer else config.lr
        t1 = time.time()

        metrics_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc_noisy": train_acc_noisy,
            "train_acc_orig": train_acc_orig,
            "val_loss": val_loss,
            "val_acc_noisy": val_acc_noisy,
            "val_acc_orig": val_acc_orig,
            "lr": lr,
            "epoch_time": t1 - t0
        })

        logger.info("Iter %d Epoch %d: train_acc_noisy=%.4f train_acc_orig=%s val_acc_orig=%s val_acc_noisy=%s ",
                    iter_idx, epoch,
                    train_acc_noisy if train_acc_noisy is not None else float('nan'),
                    f"{train_acc_orig:.4f}" if train_acc_orig is not None else "N/A",
                    f"{val_acc_orig:.4f}" if val_acc_orig is not None else "N/A",
                    f"{val_acc_noisy:.4f}" if val_acc_noisy is not None else "N/A"
                    )

        # Save every N epochs
        if (epoch + 1) % config.save_every_n_epochs == 0:
            path_epoch = str(ck_dir / f"model_iter{iter_idx}_epoch{epoch+1}.pth")
            save_checkpoint(path_epoch, model, optimizer, scheduler, epoch + 1)

        # Check best (prefer val_acc_orig if available, else val_acc_noisy)
        # current_val_score = val_acc_orig if val_acc_orig is not None else (val_acc_noisy if val_acc_noisy is not None else -1.0)
        if config.early_stop_metric == "val_acc_orig" and val_acc_orig is not None:
            current_val_score = val_acc_orig
        else:
            current_val_score = val_acc_noisy
            
        if current_val_score > best_val_acc_orig:
            best_val_acc_orig = current_val_score
            best_model_path = str(ck_dir / f"model_iter{iter_idx}_best.pth")
            save_checkpoint(best_model_path, model, optimizer, scheduler, epoch + 1, extra={"best_val_acc": best_val_acc_orig})
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # step scheduler after epoch
        try:
            scheduler.step()
        except Exception:
            pass

        # Early stopping in iteration
        if early_stop_counter >= config.patience_epoch:
            logger.info("Early stopping in iteration %d at epoch %d", iter_idx, epoch)
            break

    # Save last
    path_last = str(ck_dir / f"model_iter{iter_idx}_last.pth")
    save_checkpoint(path_last, model, optimizer, scheduler, epoch + 1)

    # Save metrics to csv
    import os
    metrics_dir = iter_base / "metrics"
    make_dirs(str(metrics_dir))
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = str(metrics_dir / "metrics_epoch.csv")
    save_dataframe_csv(metrics_df, metrics_path)

    # After training, load best model weights for evaluation/prediction
    if best_model_path is None:
        # fallback to last
        best_model_path = path_last
    # load best model state into a fresh model for evaluation to be safe
    best_model = get_model(num_classes=config.num_classes, pretrained=True, device=device)
    ck = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(ck["model_state_dict"])

    # Predict on train_full for z_hat
    # indices_np, z_hat = predict_on_loader(best_model, train_full_loader, device)
    indices_np, z_hat, train_full_acc_noisy, train_full_acc_orig = predict_with_acc(best_model, train_full_loader, device)

    # Evaluate on val and test sets for summary metrics (prefer original label metrics when available)
    _, val_acc_noisy, val_acc_orig, _, _ = validate_epoch(best_model, val_loader, None, device)
    _, test_acc_noisy, test_acc_orig, _, _ = validate_epoch(best_model, test_loader, None, device)

    # Prefer original label stats for best metrics if available
    if config.early_stop_metric == "val_acc_orig" and val_acc_orig is not None:
        reported_val_acc = val_acc_orig
    else:
        reported_val_acc = val_acc_noisy
        
    reported_test_acc = test_acc_orig if test_acc_orig is not None else test_acc_noisy

    logger.info("Iteration %d done. reported_val_acc=%.4f reported_test_acc=%.4f (val_orig=%s test_orig=%s train_full_noisy=%s train_full_orig=%s)",
                iter_idx,
                reported_val_acc if reported_val_acc is not None else float('nan'),
                reported_test_acc if reported_test_acc is not None else float('nan'),
                f"{val_acc_orig:.4f}" if val_acc_orig is not None else "N/A",
                f"{test_acc_orig:.4f}" if test_acc_orig is not None else "N/A",
                f"{train_full_acc_noisy:.4f}" if train_full_acc_noisy is not None else "N/A",
                f"{train_full_acc_orig:.4f}" if train_full_acc_orig is not None else "N/A",
                )   

    return {
        "best_model_path": best_model_path,
        "last_model_path": path_last,
        "metrics_df": metrics_df,
        "z_hat_indices": indices_np,
        "z_hat": z_hat,
        
         # ---- ADD TRAIN FULL ACC METRICS HERE ----
        "train_full_acc_noisy": float(train_full_acc_noisy) if train_full_acc_noisy is not None else None,
        "train_full_acc_orig": float(train_full_acc_orig) if train_full_acc_orig is not None else None,

        
        "best_val_acc": float(reported_val_acc) if reported_val_acc is not None else -1.0,
        "test_acc": float(reported_test_acc) if reported_test_acc is not None else -1.0,
        "val_acc_orig": float(val_acc_orig) if val_acc_orig is not None else None,
        "test_acc_orig": float(test_acc_orig) if test_acc_orig is not None else None,
        "val_acc_noisy": float(val_acc_noisy) if val_acc_noisy is not None else None,
        "test_acc_noisy": float(test_acc_noisy) if test_acc_noisy is not None else None
    }
