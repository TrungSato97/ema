"""
Dataset utilities: download CIFAR-10, save images out as PNG, create CSVs,
and provide PyTorch Dataset/Loader that reads the CSV.
"""

from typing import Dict, Tuple, List
from pathlib import Path
import os
import random
from PIL import Image
import numpy as np
import pandas as pd
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from .config import TrainConfig
import logging

import torch
import numpy as np
import random
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _worker_init_fn(worker_id):
    """
    Seed each worker with a derived seed from torch.initial_seed().
    This ensures torchvision transforms / numpy.random / random behave reproducibly per worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def inject_noise(labels: np.ndarray, noise_ratio: float, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject symmetric label noise: for a fraction `noise_ratio` of indices,
    replace label by a different random class uniformly.
    Returns:
        labels_noisy: np.ndarray (copy)
        noise_flags: np.ndarray of 0/1 flags
    """
    rng = np.random.RandomState(random_state)
    labels = labels.copy()
    N = len(labels)
    num_noisy = int(round(noise_ratio * N))
    indices = rng.choice(N, size=num_noisy, replace=False)
    noise_flags = np.zeros(N, dtype=np.int8)
    for idx in indices:
        orig = labels[idx]
        choices = list(range(10))
        choices.remove(int(orig))
        labels[idx] = rng.choice(choices)
        noise_flags[idx] = 1
    return labels, noise_flags


class CSVDataset(Dataset):
    """
    Dataset that loads samples from a CSV file with columns:
    index,image_path,label_noisy,label_orig,class_name,split,noise_flag
    """
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        # Sort by index to ensure deterministic order
        if "index" in self.df.columns:
            self.df = self.df.sort_values("index").reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label_noisy"])
        index_val = int(row["index"])
        return {
            "image": image,
            "label": label,
            "index": index_val,
            "image_path": img_path,
            "label_orig": int(row["label_orig"]),
            "noise_flag": int(row["noise_flag"])
        }


def prepare_cifar_data(config: TrainConfig, force_download: bool = False) -> Dict[str, str]:
    """
    Download CIFAR-10, split into train/val/test according to config splits,
    inject noise into TRAIN with config.noise_ratio, save images to disk resized to config.img_size,
    and write CSV files for train/val/test into data_dir/csvs/.
    Returns a dict with csv paths: {'train':..., 'val':..., 'test':...}
    """
    config.validate()
    data_root = Path(config.data_dir)
    images_root = data_root / "images"
    csv_root = data_root / "csvs"
    _ensure_dir(images_root)
    _ensure_dir(csv_root)

    # 1) Load CIFAR10: 50k train, 10k test
    dataset_trainval = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True)
    dataset_test = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True)

    X_trainval = dataset_trainval.data
    y_trainval = np.array(dataset_trainval.targets, dtype=np.int32)
    X_test = dataset_test.data
    y_test = np.array(dataset_test.targets, dtype=np.int32)
    class_names = dataset_trainval.classes

    # 2) Split 50k trainval → train/val theo config
    N = len(y_trainval)
    idxs = np.arange(N)
    train_ratio = config.split_train
    val_ratio = config.split_val

    idx_train, idx_val, y_train, y_val = train_test_split(
        idxs, y_trainval, train_size=train_ratio, random_state=config.seed, stratify=y_trainval
    )

    # Test giữ nguyên 10k gốc
    idx_test = np.arange(len(y_test))

    splits = {
        "train": (X_trainval, y_trainval, idx_train),
        "val":   (X_trainval, y_trainval, idx_val),
        "test":  (X_test, y_test, idx_test)
    }

    logger.info("Split sizes: train=%d, val=%d, test=%d", len(idx_train), len(idx_val), len(idx_test))

    # 3) Inject noise
    # 3.1) train labels
    train_labels_orig = y_trainval[idx_train]
    train_labels_noisy, train_noise_flags = inject_noise(train_labels_orig, config.noise_ratio, random_state=config.seed)
    
    # 3.2) val labels
    val_labels_orig = y_trainval[idx_val]
    val_labels_noisy, val_noise_flags = inject_noise(val_labels_orig, config.noise_ratio, random_state=(config.seed + 1))

    # Log noise summary
    train_noisy_count = int(train_noise_flags.sum())
    val_noisy_count = int(val_noise_flags.sum())
    logger.info(
        "Injected noise: train %d/%d (%.2f%%), val %d/%d (%.2f%%)",
        train_noisy_count, len(train_labels_orig), 100.0 * train_noisy_count / max(1, len(train_labels_orig)),
        val_noisy_count, len(val_labels_orig), 100.0 * val_noisy_count / max(1, len(val_labels_orig))
    )
    
    # 4) Save images + build CSV
    rows = []
    for split_name, (X_split, y_split, indices) in splits.items():
        for local_idx, global_idx in enumerate(indices):
            img = X_split[global_idx]
            cls_idx = int(y_split[global_idx])
            cls_name = class_names[cls_idx]

            out_dir = images_root / split_name / cls_name
            _ensure_dir(str(out_dir))

            img_id = f"{split_name}_{global_idx:06d}.png"
            img_path = out_dir / img_id
            pil = Image.fromarray(img)
            pil = pil.resize((config.img_size, config.img_size), Image.BILINEAR)
            pil.save(img_path)

            if split_name == "train":
                pos = np.where(idx_train == global_idx)[0][0]
                label_noisy = int(train_labels_noisy[pos])
                noise_flag = int(train_noise_flags[pos])
                label_orig = int(train_labels_orig[pos])
            
            elif split_name == "val":
                pos = np.where(idx_val == global_idx)[0][0]
                label_noisy = int(val_labels_noisy[pos])
                noise_flag = int(val_noise_flags[pos])
                label_orig = int(val_labels_orig[pos])
            
            else:
                label_noisy = cls_idx
                label_orig = cls_idx
                noise_flag = 0

            rows.append({
                "index": int(global_idx),
                "image_path": str(img_path),
                "label_noisy": int(label_noisy),
                "label_orig": int(label_orig),
                "class_name": str(class_names[label_orig]),
                "split": split_name,
                "noise_flag": int(noise_flag)
            })

    # 5) Xuất CSV
    df = pd.DataFrame(rows)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    train_csv = csv_root / "train.csv"
    val_csv   = csv_root / "val.csv"
    test_csv  = csv_root / "test.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    logger.info("Saved CSVs at %s", csv_root)

    return {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "class_names": class_names
    }


def get_transforms(img_size: int = 224, train: bool = True):
    """
    Return torchvision transforms for training/validation/testing that are compatible with ImageNet pretrained models.
    """
    if train:
        t = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        t = transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return t


def _safe_make_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory, generator: Optional[torch.Generator], worker_init_fn):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": worker_init_fn,
        "persistent_workers": False,
    }
    # only pass generator if supported
    try:
        if generator is not None:
            kwargs["generator"] = generator
    except Exception:
        pass
    return DataLoader(**kwargs)

def make_data_loaders(train_csv: str, val_csv: str, test_csv: str,
                      config: TrainConfig) -> Dict[str, DataLoader]:
    """
    Build DataLoader objects for train/val/test.
    Ensure reproducible shuffle by providing a torch.Generator seeded with config.seed.
    """
    t_train = get_transforms(config.img_size, train=True)
    t_eval = get_transforms(config.img_size, train=False)

    # Create generator for deterministic shuffling (PyTorch >=1.8 supports generator argument)
    try:
        g = torch.Generator()
        g.manual_seed(int(config.seed))
    except Exception:
        g = None

    train_dataset = CSVDataset(train_csv, transform=t_train)
    val_dataset = CSVDataset(val_csv, transform=t_eval)
    test_dataset = CSVDataset(test_csv, transform=t_eval)

    # Train loader: shuffle=True but deterministic via generator + worker_init_fn
    train_loader = _safe_make_dataloader(train_dataset, batch_size=config.batch_size,
                                         shuffle=True, num_workers=config.num_workers,
                                         pin_memory=True, generator=g, worker_init_fn=_worker_init_fn)

    val_loader = _safe_make_dataloader(val_dataset, batch_size=config.batch_size,
                                       shuffle=False, num_workers=config.num_workers,
                                       pin_memory=True, generator=None, worker_init_fn=_worker_init_fn)

    test_loader = _safe_make_dataloader(test_dataset, batch_size=config.batch_size,
                                        shuffle=False, num_workers=config.num_workers,
                                        pin_memory=True, generator=None, worker_init_fn=_worker_init_fn)

    # Full train loader for predictions: to be absolutely safe about order, use num_workers=0 OR SequentialSampler
    # Using num_workers=0 guarantees the returned order is the same as CSV order across runs.
    train_full_loader = DataLoader(CSVDataset(train_csv, transform=t_eval), batch_size=config.batch_size,
                                   shuffle=False, num_workers=0, pin_memory=True)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_full": train_full_loader,
        "train_dataset": train_dataset
    }