"""
Dataset utilities: download CIFAR-10, save images out as PNG, create CSVs,
and provide PyTorch Dataset/Loader that reads the CSV.

IMPORTANT: Images are saved once and shared across all noise ratios.
Only CSV files (containing label_noisy and noise_flag) are unique per noise ratio.

CSV FORMAT:
index,image_path,label_noisy,label_orig,class_name,split,noise_flag

- index: Global index from original CIFAR-10 dataset (0-49999 for train/val, 0-9999 for test)
- image_path: Path to the PNG image file (organized by label_orig, not label_noisy)
- label_noisy: The label used for training (may be corrupted by noise), integer 0-9
- label_orig: The ground truth label from CIFAR-10, integer 0-9
- class_name: String name of the class corresponding to label_orig
- split: 'train', 'val', or 'test'
- noise_flag: 1 if this sample has noisy label, 0 if clean
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
    
    Args:
        labels: np.ndarray of original labels (integers 0-9 for CIFAR-10)
        noise_ratio: Fraction of labels to corrupt (0.0 - 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        labels_noisy: np.ndarray (copy) with noise injected
        noise_flags: np.ndarray of 0/1 flags (1 = noisy, 0 = clean)
    """
    rng = np.random.RandomState(random_state)
    labels_noisy = labels.copy()
    N = len(labels)
    num_noisy = int(round(noise_ratio * N))
    
    # Randomly select indices to corrupt
    noisy_indices = rng.choice(N, size=num_noisy, replace=False)
    noise_flags = np.zeros(N, dtype=np.int8)
    
    # For each selected index, replace with a different random class
    for idx in noisy_indices:
        orig_label = int(labels_noisy[idx])
        # Choose from all classes except the original one
        other_classes = list(range(10))
        other_classes.remove(orig_label)
        labels_noisy[idx] = rng.choice(other_classes)
        noise_flags[idx] = 1
    
    return labels_noisy, noise_flags


def _prepare_images_once(data_root: Path, img_size: int, seed: int, 
                         split_train: float, split_val: float) -> Dict:
    """
    Chuẩn bị và lưu images một lần duy nhất.
    Images được lưu theo cấu trúc: images/split/class_name/image_id.png
    
    Hàm này chỉ chạy một lần khi images chưa tồn tại. Các lần sau sẽ skip.
    
    Returns:
        Dict chứa thông tin về data arrays, splits, và class names để tạo CSV files
    """
    images_root = data_root / "images"
    
    # Kiểm tra xem images đã được chuẩn bị chưa bằng cách check flag file
    images_ready_flag = data_root / ".images_prepared"
    
    if images_ready_flag.exists():
        logger.info("Images already prepared, loading metadata only...")
        # Load dataset để lấy thông tin cơ bản
        dataset_trainval = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True)
        dataset_test = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True)
        
        X_trainval = dataset_trainval.data
        y_trainval = np.array(dataset_trainval.targets, dtype=np.int32)
        X_test = dataset_test.data
        y_test = np.array(dataset_test.targets, dtype=np.int32)
        class_names = dataset_trainval.classes
        
        # Recreate splits với cùng seed để có indices nhất quán
        N = len(y_trainval)
        idxs = np.arange(N)
        idx_train, idx_val, _, _ = train_test_split(
            idxs, y_trainval, train_size=split_train, random_state=seed, stratify=y_trainval
        )
        idx_test = np.arange(len(y_test))
        
        logger.info("Metadata loaded: train=%d, val=%d, test=%d", 
                   len(idx_train), len(idx_val), len(idx_test))
        
        return {
            'X_trainval': X_trainval,
            'y_trainval': y_trainval,
            'X_test': X_test,
            'y_test': y_test,
            'idx_train': idx_train,
            'idx_val': idx_val,
            'idx_test': idx_test,
            'class_names': class_names,
            'images_root': images_root
        }
    
    logger.info("Preparing images for the first time...")
    _ensure_dir(images_root)
    
    # Load CIFAR-10: 50k trainval, 10k test
    dataset_trainval = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True)
    dataset_test = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True)

    X_trainval = dataset_trainval.data  # Shape: (50000, 32, 32, 3)
    y_trainval = np.array(dataset_trainval.targets, dtype=np.int32)  # Shape: (50000,), values 0-9
    X_test = dataset_test.data  # Shape: (10000, 32, 32, 3)
    y_test = np.array(dataset_test.targets, dtype=np.int32)  # Shape: (10000,), values 0-9
    class_names = dataset_trainval.classes  # ['airplane', 'automobile', ..., 'truck']

    logger.info("CIFAR-10 loaded: trainval=%d, test=%d, classes=%d", 
               len(y_trainval), len(y_test), len(class_names))
    logger.info("Class names: %s", class_names)

    # Split 50k trainval → train/val với stratification để maintain class balance
    N = len(y_trainval)
    idxs = np.arange(N)  # [0, 1, 2, ..., 49999]
    
    idx_train, idx_val, y_train_check, y_val_check = train_test_split(
        idxs, y_trainval, train_size=split_train, random_state=seed, stratify=y_trainval
    )
    
    # idx_train và idx_val là indices vào y_trainval
    # Ví dụ: idx_train[0] = 1234 nghĩa là sample đầu tiên của train set 
    # là sample thứ 1234 từ CIFAR-10 trainval set gốc
    
    idx_test = np.arange(len(y_test))  # [0, 1, 2, ..., 9999]

    logger.info("Split sizes: train=%d, val=%d, test=%d", len(idx_train), len(idx_val), len(idx_test))
    
    # Verify class distribution in splits
    logger.info("Verifying class distribution in splits...")
    for split_name, indices, y_array in [("train", idx_train, y_trainval), 
                                          ("val", idx_val, y_trainval),
                                          ("test", idx_test, y_test)]:
        labels = y_array[indices]
        unique, counts = np.unique(labels, return_counts=True)
        logger.info("%s class distribution: %s", split_name, dict(zip(unique, counts)))
    
    # Lưu images theo cấu trúc split/class_name/
    # QUAN TRỌNG: Images được organize theo label_orig (ground truth), KHÔNG phải label_noisy
    logger.info("Saving images to disk...")
    
    for split_name in ["train", "val", "test"]:
        if split_name == "train":
            indices = idx_train
            X_data = X_trainval
            y_data = y_trainval
        elif split_name == "val":
            indices = idx_val
            X_data = X_trainval
            y_data = y_trainval
        else:  # test
            indices = idx_test
            X_data = X_test
            y_data = y_test
        
        logger.info(f"Saving {split_name} images ({len(indices)} samples)...")
        
        for local_idx, global_idx in enumerate(indices):
            # global_idx là index vào dataset CIFAR-10 gốc
            # local_idx là vị trí trong split hiện tại (0, 1, 2, ...)
            
            img_array = X_data[global_idx]  # Shape: (32, 32, 3)
            label_orig = int(y_data[global_idx])  # Integer 0-9
            class_name = class_names[label_orig]  # String như 'cat', 'dog'

            # Tạo thư mục theo class_name
            out_dir = images_root / split_name / class_name
            _ensure_dir(str(out_dir))

            # Tên file bao gồm split và global_idx để đảm bảo unique và traceable
            img_filename = f"{split_name}_{global_idx:06d}.png"
            img_path = out_dir / img_filename
            
            # Chỉ lưu nếu file chưa tồn tại
            if not img_path.exists():
                pil_img = Image.fromarray(img_array)
                pil_img = pil_img.resize((img_size, img_size), Image.BILINEAR)
                pil_img.save(img_path)
            
            # Log progress mỗi 5000 images
            if (local_idx + 1) % 5000 == 0:
                logger.info(f"  Saved {local_idx + 1}/{len(indices)} {split_name} images")
    
    # Tạo flag file để đánh dấu images đã được chuẩn bị
    images_ready_flag.touch()
    logger.info("All images saved. Flag file created at %s", images_ready_flag)
    
    return {
        'X_trainval': X_trainval,
        'y_trainval': y_trainval,
        'X_test': X_test,
        'y_test': y_test,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test,
        'class_names': class_names,
        'images_root': images_root
    }


def prepare_cifar_data(config: TrainConfig, force_download: bool = False) -> Dict[str, str]:
    """
    Download CIFAR-10, split into train/val/test, inject noise, and create CSV files.
    
    IMPORTANT CHANGES:
    - Images are saved once in data_dir/images/ and reused across all noise ratios
    - CSV files are saved in data_dir/csvs/noise_{noise_ratio}/ for each noise ratio
    - This avoids duplicating 500MB+ of image files for each noise ratio
    
    CSV FORMAT:
    index,image_path,label_noisy,label_orig,class_name,split,noise_flag
    
    Returns a dict with csv paths: {'train_csv':..., 'val_csv':..., 'test_csv':..., 'class_names':...}
    """
    config.validate()
    
    # Sử dụng data_root làm thư mục gốc chung cho tất cả noise ratios
    data_root = Path(config.data_dir)
    
    # Nếu data_dir có chứa "noise_" trong path, extract ra root directory
    if "noise_" in str(data_root):
        # Tìm parent directory không chứa "noise_"
        while "noise_" in str(data_root):
            data_root = data_root.parent
        
    logger.info(f"Using shared data root: {data_root}")
    
    # Tạo thư mục CSV riêng cho noise ratio này
    csv_root = data_root / "csvs" / f"noise_{config.noise_ratio}"
    _ensure_dir(csv_root)
    
    logger.info(f"CSV files will be saved to: {csv_root}")
    
    # Chuẩn bị images (chỉ chạy một lần, sau đó reuse)
    image_data = _prepare_images_once(
        data_root=data_root,
        img_size=config.img_size,
        seed=config.seed,
        split_train=config.split_train,
        split_val=config.split_val
    )
    
    # Extract thông tin từ image_data
    X_trainval = image_data['X_trainval']
    y_trainval = image_data['y_trainval']
    X_test = image_data['X_test']
    y_test = image_data['y_test']
    idx_train = image_data['idx_train']
    idx_val = image_data['idx_val']
    idx_test = image_data['idx_test']
    class_names = image_data['class_names']
    images_root = image_data['images_root']
    
    logger.info("Injecting label noise with ratio=%.2f...", config.noise_ratio)
    
    # Inject noise vào labels
    # QUAN TRỌNG: Chúng ta inject noise vào labels của samples trong split, không phải toàn bộ dataset
    
    # 1. Train set: lấy labels gốc của các samples trong idx_train
    train_labels_orig = y_trainval[idx_train]  # Shape: (len(idx_train),)
    train_labels_noisy, train_noise_flags = inject_noise(
        train_labels_orig, config.noise_ratio, random_state=config.seed
    )
    
    # 2. Val set: lấy labels gốc của các samples trong idx_val
    val_labels_orig = y_trainval[idx_val]  # Shape: (len(idx_val),)
    val_labels_noisy, val_noise_flags = inject_noise(
        val_labels_orig, config.noise_ratio, random_state=(config.seed + 1)
    )
    
    # 3. Test set: KHÔNG inject noise (test set luôn clean)
    test_labels_orig = y_test[idx_test]
    test_labels_noisy = test_labels_orig.copy()
    test_noise_flags = np.zeros(len(idx_test), dtype=np.int8)

    # Log noise summary
    train_noisy_count = int(train_noise_flags.sum())
    val_noisy_count = int(val_noise_flags.sum())
    
    logger.info("Noise injection summary for noise_ratio=%.2f:", config.noise_ratio)
    logger.info("  Train: %d/%d samples noisy (%.2f%%)", 
               train_noisy_count, len(train_labels_orig), 
               100.0 * train_noisy_count / max(1, len(train_labels_orig)))
    logger.info("  Val: %d/%d samples noisy (%.2f%%)",
               val_noisy_count, len(val_labels_orig),
               100.0 * val_noisy_count / max(1, len(val_labels_orig)))
    logger.info("  Test: 0/%d samples noisy (0.00%% - test is always clean)", len(test_labels_orig))
    
    # Build CSV rows
    # Mỗi row phải có đầy đủ 7 columns: index, image_path, label_noisy, label_orig, class_name, split, noise_flag
    logger.info("Building CSV rows...")
    
    rows = []
    
    # Process train split
    for local_idx, global_idx in enumerate(idx_train):
        # global_idx: index trong CIFAR-10 trainval gốc (0-49999)
        # local_idx: vị trí trong train split (0, 1, 2, ...)
        
        label_orig = int(train_labels_orig[local_idx])  # Ground truth từ CIFAR-10
        label_noisy = int(train_labels_noisy[local_idx])  # Có thể bị corrupt bởi noise
        noise_flag = int(train_noise_flags[local_idx])  # 1 nếu noisy, 0 nếu clean
        class_name = class_names[label_orig]  # String name theo label_orig
        
        # Image path: images được organize theo label_orig
        img_filename = f"train_{global_idx:06d}.png"
        img_path = images_root / "train" / class_name / img_filename
        
        rows.append({
            "index": int(global_idx),
            "image_path": str(img_path),
            "label_noisy": label_noisy,
            "label_orig": label_orig,
            "class_name": class_name,
            "split": "train",
            "noise_flag": noise_flag
        })
    
    # Process val split
    for local_idx, global_idx in enumerate(idx_val):
        label_orig = int(val_labels_orig[local_idx])
        label_noisy = int(val_labels_noisy[local_idx])
        noise_flag = int(val_noise_flags[local_idx])
        class_name = class_names[label_orig]
        
        img_filename = f"val_{global_idx:06d}.png"
        img_path = images_root / "val" / class_name / img_filename
        
        rows.append({
            "index": int(global_idx),
            "image_path": str(img_path),
            "label_noisy": label_noisy,
            "label_orig": label_orig,
            "class_name": class_name,
            "split": "val",
            "noise_flag": noise_flag
        })
    
    # Process test split
    for local_idx, global_idx in enumerate(idx_test):
        label_orig = int(test_labels_orig[local_idx])
        label_noisy = int(test_labels_noisy[local_idx])  # Same as label_orig for test
        noise_flag = int(test_noise_flags[local_idx])  # Always 0 for test
        class_name = class_names[label_orig]
        
        img_filename = f"test_{global_idx:06d}.png"
        img_path = images_root / "test" / class_name / img_filename
        
        rows.append({
            "index": int(global_idx),
            "image_path": str(img_path),
            "label_noisy": label_noisy,
            "label_orig": label_orig,
            "class_name": class_name,
            "split": "test",
            "noise_flag": noise_flag
        })
    
    logger.info("Total CSV rows created: %d", len(rows))
    logger.info("  Train: %d, Val: %d, Test: %d", len(idx_train), len(idx_val), len(idx_test))

    # Xuất CSV vào thư mục riêng cho noise_ratio này
    df = pd.DataFrame(rows)
    
    # Verify data integrity
    logger.info("Verifying CSV data integrity...")
    assert len(df) == len(idx_train) + len(idx_val) + len(idx_test), "Total row count mismatch"
    assert df['label_orig'].min() >= 0 and df['label_orig'].max() <= 9, "label_orig out of range"
    assert df['label_noisy'].min() >= 0 and df['label_noisy'].max() <= 9, "label_noisy out of range"
    assert df['noise_flag'].isin([0, 1]).all(), "noise_flag must be 0 or 1"
    assert df[df['split'] == 'test']['noise_flag'].sum() == 0, "Test split should have no noise"
    logger.info("CSV data integrity check passed!")
    
    # Split dataframe và save
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_csv = csv_root / "train.csv"
    val_csv = csv_root / "val.csv"
    test_csv = csv_root / "test.csv"
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    logger.info("CSV files saved successfully!")
    logger.info("  Train CSV: %s (%d rows)", train_csv, len(train_df))
    logger.info("  Val CSV: %s (%d rows)", val_csv, len(val_df))
    logger.info("  Test CSV: %s (%d rows)", test_csv, len(test_df))

    return {
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "class_names": class_names
    }

class CSVDataset(Dataset):
    def __init__(self, csv_path: str, transform=None, label_col: str = "label_noisy"):
        self.df = pd.read_csv(csv_path)
        if "index" in self.df.columns:
            self.df = self.df.sort_values("index").reset_index(drop=True)
        self.transform = transform
        self.label_col = label_col

        if self.label_col not in self.df.columns:
            raise ValueError(f"label_col='{self.label_col}' not found in CSV columns: {list(self.df.columns)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = int(row[self.label_col])
        index_val = int(row["index"])

        out = {
            "image": image,
            "label": label,  # THIS is what training uses
            "index": index_val,
            "image_path": img_path,
        }

        # keep these for metrics/reporting
        if "label_noisy" in row:
            out["label_noisy"] = int(row["label_noisy"])
        if "label_orig" in row:
            out["label_orig"] = int(row["label_orig"])
        if "noise_flag" in row:
            out["noise_flag"] = int(row["noise_flag"])

        return out

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


def _safe_make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    generator: Optional[torch.Generator],
    worker_init_fn,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": worker_init_fn,
        "persistent_workers": False,
    }
    if generator is not None:
        kwargs["generator"] = generator
    return DataLoader(**kwargs)

def make_data_loaders(train_csv: str, val_csv: str, test_csv: str,
                      config: TrainConfig, train_full_csv: Optional[str] = None,
                      train_label_col: Optional[str] = None) -> Dict[str, DataLoader]:

    label_col = train_label_col or getattr(config, "train_label_col_default", "label_noisy")

    t_train = get_transforms(config.img_size, train=True)
    t_eval = get_transforms(config.img_size, train=False)

    # deterministic shuffle
    try:
        g = torch.Generator()
        g.manual_seed(int(config.seed))
    except Exception:
        g = None

    train_dataset = CSVDataset(train_csv, transform=t_train, label_col=label_col)
    val_dataset = CSVDataset(val_csv, transform=t_eval, label_col="label_noisy")
    test_dataset = CSVDataset(test_csv, transform=t_eval, label_col="label_noisy")

    train_loader = _safe_make_dataloader(train_dataset, config.batch_size, True,
                                         config.num_workers, True, g, _worker_init_fn)
    val_loader = _safe_make_dataloader(val_dataset, config.batch_size, False,
                                       config.num_workers, True, None, _worker_init_fn)
    test_loader = _safe_make_dataloader(test_dataset, config.batch_size, False,
                                        config.num_workers, True, None, _worker_init_fn)

    full_csv = train_full_csv if train_full_csv is not None else train_csv
    train_full_loader = DataLoader(
        CSVDataset(full_csv, transform=t_eval, label_col="label_noisy"),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader, "train_full": train_full_loader}
