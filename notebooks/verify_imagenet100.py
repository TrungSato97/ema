#!/usr/bin/env python3
"""
Check Imagenet100 folder structure against imagenet100.txt and
count images per class for train/val splits.

Expected layout:
<dataset_root>/
    train/
        class_000/
        class_001/
        ...
    val/
        class_000/
        class_001/
        ...
imagenet100.txt contains one class name per line (matching folder names).
"""

from pathlib import Path
import logging
from typing import List, Dict, Tuple
import pandas as pd


# Allowed image file extensions (case-insensitive)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def setup_logging(log_level: int = logging.INFO) -> None:
    """Configure basic logging."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def read_class_list(txt_path: Path) -> List[str]:
    """
    Read class names from a text file (one class per line).
    Strips whitespace and ignores empty lines.
    """
    if not txt_path.exists():
        raise FileNotFoundError(f"Class list file not found: {txt_path}")
    classes: List[str] = []
    with txt_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            name = line.strip()
            if name:
                classes.append(name)
    logging.info("Loaded %d classes from %s", len(classes), txt_path)
    return classes


def list_class_dirs(split_dir: Path) -> List[str]:
    """
    List immediate subdirectory names under split_dir.
    If split_dir does not exist, returns empty list.
    """
    if not split_dir.exists():
        logging.warning("Split directory not found: %s", split_dir)
        return []
    return [p.name for p in split_dir.iterdir() if p.is_dir()]


def check_structure(
    dataset_root: Path,
    class_list: List[str],
    splits: Tuple[str, ...] = ("train", "val"),
) -> Dict[str, Dict[str, List[str]]]:
    """
    Check structure for each split. Returns a nested dict:
    {
        "train": {"missing": [...], "extra": [...], "matched": [...]},
        "val":   {...}
    }
    """
    result: Dict[str, Dict[str, List[str]]] = {}
    class_set = set(class_list)
    for split in splits:
        split_dir = dataset_root / split
        present = set(list_class_dirs(split_dir))
        missing = sorted(list(class_set - present))
        extra = sorted(list(present - class_set))
        matched = sorted(list(class_set & present))
        logging.info(
            "[%s] present=%d, matched=%d, missing=%d, extra=%d",
            split,
            len(present),
            len(matched),
            len(missing),
            len(extra),
        )
        result[split] = {"missing": missing, "extra": extra, "matched": matched}
    return result


def count_images_in_dir(dir_path: Path) -> int:
    """Count image files in dir_path (non-recursive)."""
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    count = 0
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            count += 1
    return count


def build_count_dataframe(
    dataset_root: Path,
    class_list: List[str],
    splits: Tuple[str, ...] = ("train", "val"),
) -> pd.DataFrame:
    """
    Build a DataFrame with columns:
    ['split', 'class', 'class_dir_exists', 'image_count', 'dir_path']
    """
    rows = []
    for split in splits:
        split_dir = dataset_root / split
        for cls in class_list:
            class_dir = split_dir / cls
            exists = class_dir.exists() and class_dir.is_dir()
            img_count = count_images_in_dir(class_dir) if exists else 0
            rows.append(
                {
                    "split": split,
                    "class": cls,
                    "class_dir_exists": exists,
                    "image_count": img_count,
                    "dir_path": str(class_dir) if exists else "",
                }
            )
        # Also include "extra" classes present on disk but not in list
        present_dirs = set(list_class_dirs(split_dir))
        extras = sorted(present_dirs - set(class_list))
        for extra_cls in extras:
            extra_dir = split_dir / extra_cls
            rows.append(
                {
                    "split": split,
                    "class": extra_cls,
                    "class_dir_exists": True,
                    "image_count": count_images_in_dir(extra_dir),
                    "dir_path": str(extra_dir),
                }
            )
    df = pd.DataFrame(rows)
    # Order columns and sort for readability
    df = df[["split", "class", "class_dir_exists", "image_count", "dir_path"]]
    df.sort_values(by=["split", "class"], inplace=True, ignore_index=True)
    logging.info("Built dataframe with %d rows", len(df))
    return df


def summarize_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return aggregated stats per split:
    columns: ['split', 'num_classes', 'total_images', 'classes_with_zero_images']
    """
    summaries = []
    for split, group in df.groupby("split"):
        num_classes = group["class"].nunique()
        total_images = int(group["image_count"].sum())
        zero_classes = int((group["image_count"] == 0).sum())
        summaries.append(
            {
                "split": split,
                "num_classes": int(num_classes),
                "total_images": total_images,
                "classes_with_zero_images": zero_classes,
            }
        )
    return pd.DataFrame(summaries)


if __name__ == "__main__":
    # === User: set these paths before running ===
    dataset_root = Path("./data_imagenet100_cmc/imagenet100")  # <-- sửa path tại đây
    class_list_file = Path("./data_imagenet100_cmc/imagenet100.txt")  # <-- sửa path tại đây
    output_csv = Path("./data_imagenet100_cmc/imagenet100_class_counts.csv")  # optional: save csv
    # ===========================================

    setup_logging()

    try:
        classes = read_class_list(class_list_file)
    except FileNotFoundError as err:
        logging.error(err)
        raise SystemExit(1)

    # 1) Check structure
    structure_report = check_structure(dataset_root, classes, splits=("train", "val"))

    # Log structure differences
    for split, info in structure_report.items():
        logging.info("[%s] missing %d classes", split, len(info["missing"]))
        if info["missing"]:
            logging.debug("[%s] missing classes: %s", split, info["missing"])
        logging.info("[%s] extra %d classes", split, len(info["extra"]))
        if info["extra"]:
            logging.debug("[%s] extra classes: %s", split, info["extra"])

    # 2) Count images and build DataFrame
    df_counts = build_count_dataframe(dataset_root, classes, splits=("train", "val"))

    # Print top/bottom summary examples
    logging.info("Top 5 classes by image_count (train/val combined):")
    print(df_counts.sort_values("image_count", ascending=False).head(5).to_string(index=False))

    # Aggregated summary per split
    summary_df = summarize_counts(df_counts)
    print("/nSummary per split:")
    print(summary_df.to_string(index=False))

    # Save counts to CSV
    df_counts.to_csv(output_csv, index=False)
    logging.info("Saved counts to %s", output_csv)

    # If user wants, display DataFrame (in interactive session)
    try:
        import pandas as pd  # already imported, but safe to keep

        # Pretty print first rows
        print("/nSample rows from class counts:")
        print(df_counts.head(10).to_string(index=False))
    except Exception:
        pass
