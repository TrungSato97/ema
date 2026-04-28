"""
PIPELINE CHUẨN BỊ DỮ LIỆU CIFAR-100N (ROBUST VERSION)
- Giải quyết triệt để lỗi phân cấp Fine/Coarse labels.
- Xử lý lỗi Stratified Split khi có mẫu nhiễu quá hiếm (count = 1).
- Vẽ Transition Matrix tối ưu cho 100x100 classes.
"""

import os
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# PHẦN 1: EDA ANALYSIS
# ==========================================
def perform_eda(df: pd.DataFrame, output_dir: Path, noise_type: str, split_name: str, class_names: list):
    """Phân tích dữ liệu và vẽ Ma trận chuyển đổi nhiễu (Noise Transition Matrix)"""
    os.makedirs(output_dir, exist_ok=True)
    total = len(df)
    noisy_count = df['noise_flag'].sum()
    noise_ratio = (noisy_count / total) * 100 if total > 0 else 0
    
    # 1. Lưu report text
    report_path = output_dir / f"eda_report_{split_name}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"--- EDA REPORT: {noise_type.upper()} | SPLIT: {split_name.upper()} ---\n")
        f.write(f"Total images: {total}\n")
        f.write(f"Noisy images: {noisy_count} ({noise_ratio:.2f}%)\n\n")
        
        class_stats = df.groupby('class_name').agg(
            total=('index', 'count'),
            noisy=('noise_flag', 'sum')
        ).reset_index()
        class_stats['ratio'] = (class_stats['noisy'] / class_stats['total']) * 100
        
        f.write("Noise per class:\n")
        for _, row in class_stats.iterrows():
            f.write(f" - {row['class_name']:<20}: {row['noisy']}/{row['total']} ({row['ratio']:.2f}%)\n")
            
    logger.info(f"Đã lưu Text EDA Report cho {split_name} ({noise_ratio:.2f}% nhiễu).")

    # 2. Vẽ Noise Transition Matrix (100x100)
    if total > 0 and noisy_count > 0:
        categories = list(range(len(class_names)))
        orig_series = pd.Categorical(df['label_orig'], categories=categories)
        noisy_series = pd.Categorical(df['label_noisy'], categories=categories)
        
        cm = pd.crosstab(orig_series, noisy_series, normalize='index', dropna=False)
        
        plt.figure(figsize=(24, 20))
        # annot=False và gỡ labels để tránh đen đặc biểu đồ với 100 lớp
        sns.heatmap(cm, annot=False, cmap="Blues", cbar=True,
                    xticklabels=False, yticklabels=False)
        
        plt.title(f"Noise Transition Matrix - {noise_type} ({split_name.upper()})\nRow: Original GT | Col: Noisy Label", fontsize=16)
        plt.xlabel("Noisy Label (Human Assigned)", fontsize=14)
        plt.ylabel("Original Label (Ground Truth)", fontsize=14)
        plt.tight_layout()
        
        plot_path = output_dir / f"transition_matrix_{split_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Đã lưu Transition Matrix tại: {plot_path}")

# ==========================================
# PHẦN 2: LƯU ẢNH (SAVE ONCE ARCHITECTURE)
# ==========================================
def save_images_once(data_root: Path, img_size: int = 224) -> Path:
    images_dir = data_root / "images"
    flag_file = images_dir / ".images_ready"
    
    if flag_file.exists():
        logger.info(f"Ảnh đã tồn tại tại {images_dir}. Bỏ qua bước trích xuất.")
        return images_dir

    logger.info("Đang tải CIFAR-100 và giải nén ảnh ra đĩa...")
    os.makedirs(images_dir, exist_ok=True)
    
    ds_train = torchvision.datasets.CIFAR100(root=str(data_root), train=True, download=True)
    ds_test = torchvision.datasets.CIFAR100(root=str(data_root), train=False, download=True)
    class_names = ds_train.classes

    def save_split(dataset, split_folder):
        total = len(dataset.data)
        for i in range(total):
            img_arr = dataset.data[i]
            label = int(dataset.targets[i])
            c_name = class_names[label]
            
            out_dir = images_dir / split_folder / c_name
            os.makedirs(out_dir, exist_ok=True)
            img_path = out_dir / f"{i:06d}.png"
            
            pil_img = Image.fromarray(img_arr).resize((img_size, img_size), Image.Resampling.BILINEAR)
            pil_img.save(img_path)

    logger.info("Đang lưu tập Train...")
    save_split(ds_train, "train_pool")
    logger.info("Đang lưu tập Test...")
    save_split(ds_test, "test")
    
    flag_file.touch()
    logger.info("Hoàn tất lưu ảnh vật lý!")
    return images_dir

# ==========================================
# PHẦN 3: TẠO CSV & ĐỊNH TUYẾN DỮ LIỆU
# ==========================================
def build_cifar100n_pipeline(pt_file_path: str, val_split: float = 0.1, seed: int = 42):
    pt_path = Path(pt_file_path).resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file .pt tại: {pt_path}")
        
    data_root = pt_path.parent
    images_dir = save_images_once(data_root)
    
    # Load dataset chuẩn 100 lớp (Fine Labels)
    tmp_ds = torchvision.datasets.CIFAR100(root=str(data_root), train=True, download=False)
    y_train_orig = np.array(tmp_ds.targets, dtype=int)
    y_test_orig = np.array(torchvision.datasets.CIFAR100(root=str(data_root), train=False, download=False).targets, dtype=int)
    class_names = tmp_ds.classes

    try:
        noise_dict = torch.load(pt_path, weights_only=False)
    except TypeError:
        noise_dict = torch.load(pt_path)
    
    # CHÚ Ý: Cố tình loại bỏ 'coarse_label' để tránh lỗi train nhầm số lượng classes (20 thay vì 100)
    target_noise_types = ['noisy_label'] 
    
    for n_type in target_noise_types:
        if n_type not in noise_dict:
            logger.error(f"Không tìm thấy key '{n_type}' trong file .pt. Vui lòng kiểm tra lại file CIFAR-100_human.pt")
            continue
            
        logger.info(f"\n================ XỬ LÝ CẤU HÌNH: {n_type.upper()} (~40.2% NOISE) ================")
        csv_dir = data_root / "csvs" / f"noise_{n_type}"
        os.makedirs(csv_dir, exist_ok=True)
        
        y_train_noisy = np.array(noise_dict[n_type]).flatten().astype(int)
        
        # Bẫy lỗi 1: Đảm bảo số lượng nhãn nhiễu khớp với ảnh gốc
        assert len(y_train_noisy) == len(y_train_orig), "Lệch số lượng nhãn nhiễu và nhãn gốc!"
        
        # Bẫy lỗi 2: Đảm bảo nhãn nhiễu nằm trong khoảng 0-99
        assert y_train_noisy.max() < 100, f"Phát hiện nhãn > 99 ({y_train_noisy.max()}). Bạn đang dùng Coarse label?"

        # STRATIFICATION VỚI BẪY LỖI RARE KEYS
        noise_flags = (y_train_noisy != y_train_orig).astype(int)
        composite_keys = [f"{orig}_{flag}" for orig, flag in zip(y_train_orig, noise_flags)]
        
        value_counts = pd.Series(composite_keys).value_counts()
        rare_keys = value_counts[value_counts < 2].index
        
        if len(rare_keys) > 0:
            logger.warning(f"Phát hiện {len(rare_keys)} tổ hợp nhiễu hiếm (count < 2). Đã chuyển sang nhóm 'other' để tránh crash scikit-learn.")
            stratify_array = ["other" if k in rare_keys else k for k in composite_keys]
        else:
            stratify_array = composite_keys

        indices = np.arange(len(y_train_orig))
        idx_train, idx_val = train_test_split(indices, test_size=val_split, random_state=seed, stratify=stratify_array)
        idx_test = np.arange(len(y_test_orig))

        def make_rows(idxs, y_orig, split_name, folder_name, y_noisy=None, is_test=False):
            rows = []
            for global_idx in idxs:
                l_orig = int(y_orig[global_idx])
                c_name = class_names[l_orig]
                
                if is_test:
                    l_noisy = l_orig
                    n_flag = 0
                else:
                    l_noisy = int(y_noisy[global_idx])
                    n_flag = 1 if l_noisy != l_orig else 0
                    
                img_path = images_dir / folder_name / c_name / f"{global_idx:06d}.png"
                
                rows.append({
                    "index": int(global_idx),
                    "image_path": str(img_path),
                    "label_noisy": l_noisy,
                    "label_orig": l_orig,
                    "class_name": c_name,
                    "split": split_name,
                    "noise_flag": n_flag
                })
            return rows

        df_train = pd.DataFrame(make_rows(idx_train, y_train_orig, "train", "train_pool", y_train_noisy))
        df_val = pd.DataFrame(make_rows(idx_val, y_train_orig, "val", "train_pool", y_train_noisy))
        df_test = pd.DataFrame(make_rows(idx_test, y_test_orig, "test", "test", is_test=True))

        df_train.to_csv(csv_dir / "train.csv", index=False)
        df_val.to_csv(csv_dir / "val.csv", index=False)
        df_test.to_csv(csv_dir / "test.csv", index=False)
        logger.info(f"Đã lưu CSV vào: {csv_dir}")

        eda_dir = csv_dir / "eda_reports"
        perform_eda(df_train, eda_dir, n_type, "train", class_names)
        perform_eda(df_val, eda_dir, n_type, "val", class_names)

    logger.info("\n✅ PIPELINE CIFAR-100N HOÀN TẤT VÀ SẴN SÀNG HUẤN LUYỆN!")

if __name__ == "__main__":
    TARGET_PT_FILE = "../notebooks/data_cifar100N/CIFAR-100_human.pt"
    abs_path = os.path.abspath(TARGET_PT_FILE)
    print(f"Bắt đầu xử lý với file: {abs_path}")
    build_cifar100n_pipeline(pt_file_path=abs_path)