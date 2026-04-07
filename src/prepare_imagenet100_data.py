"""
Script để chuẩn bị dữ liệu cho bộ ImageNet-100.

Chức năng chính:
1. Quét các thư mục hình ảnh từ bộ dữ liệu ImageNet-100 đã có sẵn.
2. Phân chia dữ liệu:
   - Dữ liệu trong `train` gốc sẽ được chia thành tập `train` và `val` mới.
   - Dữ liệu trong `val` gốc sẽ được sử dụng làm tập `test`.
3. Thêm nhiễu vào nhãn (label noise) cho các tập `train` và `val` theo các tỷ lệ được chỉ định.
4. Xuất ra các file CSV (train.csv, val.csv, test.csv) cho mỗi tỷ lệ nhiễu,
   theo đúng định dạng mà project yêu cầu.

Cách sử dụng:
python scripts/prepare_imagenet100_data.py \
    --data_dir data_imagenet100_cmc/imagenet100 \
    --output_dir data/csvs_imagenet100 \
    --noise_ratios 0.2 0.4 0.6 0.8 \
    --val_split 0.1
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
import logging
from typing import Tuple, Dict, List

# --- Cấu hình Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def inject_noise(labels: np.ndarray, noise_ratio: float, num_classes: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thêm nhiễu đối xứng (symmetric label noise) vào một mảng các nhãn.
    """
    rng = np.random.RandomState(random_state)
    labels_noisy = labels.copy()
    N = len(labels)
    num_noisy = int(round(noise_ratio * N))
    
    # Chọn ngẫu nhiên các chỉ số để thêm nhiễu
    noisy_indices = rng.choice(N, size=num_noisy, replace=False)
    noise_flags = np.zeros(N, dtype=np.int8)
    
    # Với mỗi chỉ số được chọn, thay thế nhãn bằng một lớp ngẫu nhiên khác
    for idx in noisy_indices:
        orig_label = int(labels_noisy[idx])
        other_classes = list(range(num_classes))
        other_classes.remove(orig_label)
        labels_noisy[idx] = rng.choice(other_classes)
        noise_flags[idx] = 1
    
    return labels_noisy, noise_flags

def scan_images(data_path: Path, class_to_idx: Dict[str, int]) -> List[Dict]:
    """
    Quét thư mục hình ảnh và trả về một danh sách các dictionary chứa thông tin file.
    """
    image_info = []
    logging.info(f"Scanning directory: {data_path}...")
    
    # Giả định cấu trúc: data_path/class_name/image.JPEG
    for class_dir in data_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        if class_name not in class_to_idx:
            logging.warning(f"Class '{class_name}' not in class_to_idx map. Skipping.")
            continue
            
        label_orig = class_to_idx[class_name]
        
        for image_file in class_dir.glob('*.JPEG'):
            image_info.append({
                'image_path': str(image_file.resolve()),
                'class_name': class_name,
                'label_orig': label_orig
            })
            
    logging.info(f"Found {len(image_info)} images in {data_path}.")
    return image_info

def process_dataset(
    data_dir: Path, 
    output_dir: Path, 
    noise_ratios: List[float], 
    val_split_ratio: float, 
    seed: int
):
    """
    Hàm chính để xử lý toàn bộ dataset:
    1. Quét các file ảnh.
    2. Chia train/val/test.
    3. Thêm nhiễu cho các tỷ lệ được chỉ định.
    4. Lưu các file CSV.
    """
    
    # --- 1. Xác định các lớp và tạo mapping ---
    logging.info("--- Step 1: Identifying classes ---")
    train_path = data_dir / 'train'
    class_dirs = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    num_classes = len(class_dirs)
    class_to_idx = {name: i for i, name in enumerate(class_dirs)}
    
    if num_classes == 0:
        logging.error(f"No class directories found in {train_path}. Exiting.")
        return
        
    logging.info(f"Found {num_classes} classes. Example: {class_dirs[:5]}")

    # --- 2. Quét thư mục và tạo DataFrame ban đầu ---
    logging.info("\n--- Step 2: Scanning image directories ---")
    # Dữ liệu train gốc sẽ được chia thành train/val mới
    orig_train_info = scan_images(train_path, class_to_idx)
    # Dữ liệu val gốc sẽ được dùng làm test set
    orig_test_info = scan_images(data_dir / 'val', class_to_idx)
    
    orig_train_df = pd.DataFrame(orig_train_info)
    orig_test_df = pd.DataFrame(orig_test_info)
    
    # Gán split ban đầu
    orig_train_df['split'] = 'train_pool'
    orig_test_df['split'] = 'test'
    
    # Kết hợp lại để gán index duy nhất cho mỗi ảnh
    combined_df = pd.concat([orig_train_df, orig_test_df], ignore_index=True)
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'unique_id'}, inplace=True)
    
    logging.info(f"Total images found: {len(combined_df)}")
    logging.info(f"Original train pool size: {len(orig_train_df)}")
    logging.info(f"Original test set size: {len(orig_test_df)}")

    # --- 3. Chia tập train/val từ train_pool ---
    logging.info("\n--- Step 3: Splitting train/validation sets ---")
    train_pool_df = combined_df[combined_df['split'] == 'train_pool'].copy()
    
    # Sử dụng train_test_split để chia train_pool thành train và val mới
    # stratify để đảm bảo phân bố lớp đồng đều
    train_indices, val_indices = train_test_split(
        train_pool_df.index,
        test_size=val_split_ratio,
        random_state=seed,
        stratify=train_pool_df['label_orig']
    )
    
    # Cập nhật cột 'split' trong DataFrame tổng hợp
    combined_df.loc[train_indices, 'split'] = 'train'
    combined_df.loc[val_indices, 'split'] = 'val'
    
    logging.info(f"New train set size: {len(train_indices)}")
    logging.info(f"New validation set size: {len(val_indices)}")

    # --- 4. Lặp qua các tỷ lệ nhiễu và tạo file CSV ---
    for noise_ratio in noise_ratios:
        logging.info(f"\n--- Step 4: Processing for noise_ratio = {noise_ratio} ---")
        
        df_noisy = combined_df.copy()
        
        df_noisy['label_noisy'] = df_noisy['label_orig']
        df_noisy['noise_flag'] = 0
        
        # Thêm nhiễu vào tập train
        train_mask = df_noisy['split'] == 'train'
        train_labels_orig = df_noisy.loc[train_mask, 'label_orig'].to_numpy()
        train_labels_noisy, train_noise_flags = inject_noise(train_labels_orig, noise_ratio, num_classes, seed)
        df_noisy.loc[train_mask, 'label_noisy'] = train_labels_noisy
        df_noisy.loc[train_mask, 'noise_flag'] = train_noise_flags
        logging.info(f"  Train set: Injected noise into {train_noise_flags.sum()}/{len(train_labels_orig)} samples.")

        # Thêm nhiễu vào tập val
        val_mask = df_noisy['split'] == 'val'
        val_labels_orig = df_noisy.loc[val_mask, 'label_orig'].to_numpy()
        val_labels_noisy, val_noise_flags = inject_noise(val_labels_orig, noise_ratio, num_classes, seed + 1)
        df_noisy.loc[val_mask, 'label_noisy'] = val_labels_noisy
        df_noisy.loc[val_mask, 'noise_flag'] = val_noise_flags
        logging.info(f"  Validation set: Injected noise into {val_noise_flags.sum()}/{len(val_labels_orig)} samples.")
        
        # Chuẩn bị và lưu file CSV
        df_final = df_noisy.rename(columns={'unique_id': 'index'})
        final_columns = ['index', 'image_path', 'label_noisy', 'label_orig', 'class_name', 'split', 'noise_flag']
        df_final = df_final[final_columns]
        
        csv_output_dir = output_dir / f"noise_{noise_ratio}"
        csv_output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name in ['train', 'val', 'test']:
            split_df = df_final[df_final['split'] == split_name]
            output_path = csv_output_dir / f"{split_name}.csv"
            split_df.to_csv(output_path, index=False)
            logging.info(f"  Saved {split_name}.csv to {output_path} ({len(split_df)} rows)")
            
    logging.info("\n--- Processing complete! ---")

def main():
    parser = argparse.ArgumentParser(description="Prepare ImageNet-100 dataset with label noise.")
    parser.add_argument('--data_dir', type=str, default='../notebooks/data_imagenet100_cmc/imagenet100', help='Path to the root ImageNet-100 directory.')
    parser.add_argument('--output_dir', type=str, default='../notebooks/data_imagenet100_cmc/csvs', help='Directory to save the output CSV files.')
    parser.add_argument('--noise_ratios', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8], help='List of noise ratios to generate.')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of original training data for new validation set.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    args = parser.parse_args()
    
    process_dataset(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        noise_ratios=args.noise_ratios,
        val_split_ratio=args.val_split,
        seed=args.seed
    )

if __name__ == '__main__':
    main()
