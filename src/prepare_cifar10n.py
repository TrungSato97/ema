"""
PIPELINE CHUẨN BỊ DỮ LIỆU CIFAR-10N & EDA TỰ ĐỘNG
- Lưu ảnh 1 lần vào train_pool và test.
- Tự động lấy gốc từ đường dẫn file .pt
- Sinh CSV chuẩn xác với Composite Key Stratification.
- Tự động vẽ biểu đồ EDA (Noise Transition Matrix).
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
            f.write(f" - {row['class_name']:<12}: {row['noisy']}/{row['total']} ({row['ratio']:.2f}%)\n")
            
    logger.info(f"Đã lưu Text EDA Report cho {split_name} ({noise_ratio:.2f}% nhiễu).")

    # 2. Vẽ Noise Transition Matrix (Sửa lại logic an toàn 10x10)
    if total > 0 and noisy_count > 0:
        # TẠO CATEGORICAL ĐỂ ÉP PANDAS GIỮ KÍCH THƯỚC MA TRẬN 10x10
        categories = list(range(len(class_names)))
        orig_series = pd.Categorical(df['label_orig'], categories=categories)
        noisy_series = pd.Categorical(df['label_noisy'], categories=categories)
        
        # crosstab bây giờ luôn đảm bảo trả về ma trận len(class_names) x len(class_names)
        cm = pd.crosstab(orig_series, noisy_series, normalize='index', dropna=False)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Noise Transition Matrix - {noise_type} ({split_name.upper()})\nRow: Original | Col: Noisy")
        plt.xlabel("Noisy Label (Human Assigned)")
        plt.ylabel("Original Label (Ground Truth)")
        plt.tight_layout()
        
        plot_path = output_dir / f"transition_matrix_{split_name}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info(f"Đã lưu Biểu đồ Transition Matrix tại: {plot_path}")

# ==========================================
# PHẦN 2: LƯU ẢNH (SAVE ONCE ARCHITECTURE)
# ==========================================
def save_images_once(data_root: Path, img_size: int = 224) -> Path:
    """Lưu toàn bộ 50k train vào 'train_pool' và 10k test vào 'test'."""
    images_dir = data_root / "images"
    flag_file = images_dir / ".images_ready"
    
    if flag_file.exists():
        logger.info(f"Ảnh đã tồn tại tại {images_dir}. Bỏ qua bước giải nén ảnh.")
        return images_dir

    logger.info("Lần đầu chạy: Đang tải CIFAR-10 và giải nén ảnh ra đĩa...")
    os.makedirs(images_dir, exist_ok=True)
    
    # Load dataset
    ds_train = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True)
    ds_test = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True)
    class_names = ds_train.classes

    def save_split(dataset, split_folder):
        total = len(dataset.data)
        logger.info(f"Đang lưu {total} ảnh vào thư mục {split_folder}...")
        for i in range(total):
            img_arr = dataset.data[i]
            label = int(dataset.targets[i])
            c_name = class_names[label]
            
            # Cấu trúc: images/train_pool/cat/00001.png
            out_dir = images_dir / split_folder / c_name
            os.makedirs(out_dir, exist_ok=True)
            img_path = out_dir / f"{i:06d}.png"
            
            pil_img = Image.fromarray(img_arr).resize((img_size, img_size), Image.Resampling.BILINEAR)
            pil_img.save(img_path)
            
            if (i + 1) % 10000 == 0:
                logger.info(f"  Đã lưu {i + 1}/{total} ảnh.")

    save_split(ds_train, "train_pool")
    save_split(ds_test, "test")
    
    flag_file.touch()
    logger.info("Hoàn tất lưu ảnh vật lý!")
    return images_dir

# ==========================================
# PHẦN 3: TẠO CSV & ĐỊNH TUYẾN DỮ LIỆU
# ==========================================
def build_cifar10n_pipeline(pt_file_path: str, val_split: float = 0.1, seed: int = 42):
    """Pipeline chính: Tự động định tuyến từ file .pt"""
    pt_path = Path(pt_file_path).resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file .pt tại: {pt_path}")
        
    # Tự động lấy thư mục chứa file .pt làm thư mục gốc (data_root)
    data_root = pt_path.parent
    logger.info(f"Data Root được xác định là: {data_root}")
    
    # 1. Đảm bảo ảnh đã được lưu 1 lần
    images_dir = save_images_once(data_root)
    
    # Lấy class_names từ torchvision để dùng cho EDA
    tmp_ds = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=False)
    y_train_orig = np.array(tmp_ds.targets, dtype=int)
    y_test_orig = np.array(torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=False).targets, dtype=int)
    class_names = tmp_ds.classes

    # 2. Đọc nhãn nhiễu
    logger.info(f"Đang đọc file nhãn nhiễu: {pt_path.name}")
    try:
        # Cập nhật an toàn cho PyTorch >= 2.0
        noise_dict = torch.load(pt_path, weights_only=True)
    except TypeError:
        # Fallback cho PyTorch bản quá cũ
        noise_dict = torch.load(pt_path)
    
    # Các loại nhiễu chuẩn trong CIFAR-10N
    target_noise_types = ['clean_label', 'aggre_label', 'worse_label', 'random_label1', 'random_label2', 'random_label3']
    
    for n_type in target_noise_types:
        if n_type not in noise_dict:
            continue
            
        logger.info(f"\n================ XỬ LÝ CẤU HÌNH: {n_type.upper()} ================")
        csv_dir = data_root / "csvs" / f"noise_{n_type}"
        os.makedirs(csv_dir, exist_ok=True)
        
        y_train_noisy = np.array(noise_dict[n_type]).flatten().astype(int)
        
        # 3. COMPOSITE KEY STRATIFICATION
        noise_flags = (y_train_noisy != y_train_orig).astype(int)
        composite_keys = [f"{orig}_{flag}" for orig, flag in zip(y_train_orig, noise_flags)]
        
        indices = np.arange(len(y_train_orig))
        idx_train, idx_val = train_test_split(indices, test_size=val_split, random_state=seed, stratify=composite_keys)
        idx_test = np.arange(len(y_test_orig))

        # 4. Tạo CSV Rows
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

        train_rows = make_rows(idx_train, y_train_orig, "train", "train_pool", y_train_noisy)
        val_rows = make_rows(idx_val, y_train_orig, "val", "train_pool", y_train_noisy)
        test_rows = make_rows(idx_test, y_test_orig, "test", "test", is_test=True)

        df_train = pd.DataFrame(train_rows)
        df_val = pd.DataFrame(val_rows)
        df_test = pd.DataFrame(test_rows)

        # 5. Lưu CSV
        df_train.to_csv(csv_dir / "train.csv", index=False)
        df_val.to_csv(csv_dir / "val.csv", index=False)
        df_test.to_csv(csv_dir / "test.csv", index=False)
        logger.info(f"Đã lưu CSV vào: {csv_dir}")

        # 6. Chạy EDA Phân tích
        eda_dir = csv_dir / "eda_reports"
        perform_eda(df_train, eda_dir, n_type, "train", class_names)
        perform_eda(df_val, eda_dir, n_type, "val", class_names)

    logger.info("\n✅ PIPELINE HOÀN TẤT!")

if __name__ == "__main__":
    # --- BẠN CHỈ CẦN THAY ĐỔI ĐƯỜNG DẪN NÀY ---
    # Giả định file .pt của bạn đang nằm ở: notebooks/data_cifar10N/CIFAR-10_human.pt
    # Bạn có thể dùng đường dẫn tương đối hoặc tuyệt đối tùy thuộc thư mục bạn đang đứng
    
    TARGET_PT_FILE = "../notebooks/data_cifar10N/CIFAR-10_human.pt"
    
    # Lấy đường dẫn tuyệt đối để chống lỗi thư mục working directory
    abs_path = os.path.abspath(TARGET_PT_FILE)
    
    print(f"Bắt đầu xử lý với file: {abs_path}")
    build_cifar10n_pipeline(pt_file_path=abs_path)