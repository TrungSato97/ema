import os
import shutil
import pandas as pd
from tqdm import tqdm

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN (ĐÃ FIX LỖI PATH)
# ==========================================
BASE_DIR = "/mnt/d/data_noise_label/clothing1M"

# 1. Trỏ chính xác vào thư mục chứa file .txt
ANNOTATION_DIR = os.path.join(BASE_DIR, "raw_annotations") 

# 2. Thư mục chứa 1M ảnh ĐÃ GIẢI NÉN 
# (LƯU Ý: Đổi tên "extracted_images" thành tên thư mục thực tế bạn đã giải nén ảnh ra)
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "extracted_images") 

# 3. Thư mục đầu ra (Code sẽ tự tạo)
OUT_IMAGES_DIR = os.path.join(BASE_DIR, "images_37k")
OUT_CSVS_DIR = os.path.join(BASE_DIR, "csvs")

# Tạo thư mục nếu chưa có
os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUT_CSVS_DIR, exist_ok=True)

# ==========================================
# CÁC HÀM TIỆN ÍCH
# ==========================================
def load_kv_to_dict(file_path):
    """Đọc file format 'image_path label' thành dictionary"""
    d = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                d[parts[0]] = int(parts[1])
    return d

# ==========================================
# LUỒNG XỬ LÝ CHÍNH
# ==========================================
def main():
    print(f"1. Đang load Annotations từ: {ANNOTATION_DIR}")
    
    # Load Dict nhãn
    clean_labels = load_kv_to_dict(os.path.join(ANNOTATION_DIR, "clean_label_kv.txt"))
    noisy_labels = load_kv_to_dict(os.path.join(ANNOTATION_DIR, "noisy_label_kv.txt"))
    
    # Load Key lists
    with open(os.path.join(ANNOTATION_DIR, "clean_train_key_list.txt")) as f: 
        train_keys = f.read().splitlines()
    with open(os.path.join(ANNOTATION_DIR, "clean_val_key_list.txt")) as f: 
        val_keys = f.read().splitlines()
    with open(os.path.join(ANNOTATION_DIR, "clean_test_key_list.txt")) as f: 
        test_keys = f.read().splitlines()
    
    # Load category names
    category_names = {}
    with open(os.path.join(ANNOTATION_DIR, "category_names_eng.txt")) as f:
        for i, line in enumerate(f): 
            category_names[i] = line.strip()

    all_rows = []
    global_index = 0

    def process_split(keys, split_name, has_noise):
        nonlocal global_index
        missing_images = 0
        
        print(f"\nĐang xử lý tập {split_name} ({len(keys)} ảnh)...")
        for rel_path in tqdm(keys):
            # Đường dẫn ảnh gốc trong thư mục 1M ảnh
            src_image_path = os.path.join(RAW_IMAGES_DIR, rel_path)
            
            # Tạo tên file phẳng để lưu vào images_37k
            safe_filename = rel_path.replace("/", "_").replace("\\", "_")
            dst_image_path = os.path.join(OUT_IMAGES_DIR, safe_filename)

            # Copy file (Tối ưu I/O)
            if not os.path.exists(dst_image_path):
                if os.path.exists(src_image_path):
                    shutil.copy2(src_image_path, dst_image_path)
                else:
                    missing_images += 1
                    continue
            
            # DATA CONTRACT (Xử lý nhãn nhiễu/sạch)
            label_orig = clean_labels[rel_path]
            class_name = category_names.get(label_orig, str(label_orig))
            
            if has_noise:
                label_noisy = noisy_labels[rel_path]
                noise_flag = 1 if label_noisy != label_orig else 0
            else:
                label_noisy = label_orig
                noise_flag = 0
            
            all_rows.append({
                "index": global_index,
                "image_path": dst_image_path, 
                "label_noisy": label_noisy,
                "label_orig": label_orig,
                "class_name": class_name,
                "split": split_name,
                "noise_flag": noise_flag
            })
            global_index += 1
            
        if missing_images > 0:
            print(f"CẢNH BÁO: Bỏ qua {missing_images} ảnh do không tìm thấy trên ổ cứng (trong thư mục {RAW_IMAGES_DIR}).")

    # Thực thi tuần tự (Train/Val có nhiễu, Test sạch)
    process_split(train_keys, 'train', has_noise=True)
    process_split(val_keys, 'val', has_noise=True)
    process_split(test_keys, 'test', has_noise=False)

    # Xuất file CSV
    df = pd.DataFrame(all_rows)
    print(f"\n✅ Tổng hợp xong. Đã xử lý thành công {len(df)} ảnh.")
    
    for split in ['train', 'val', 'test']:
        df_split = df[df['split'] == split]
        csv_path = os.path.join(OUT_CSVS_DIR, f"{split}.csv")
        df_split.to_csv(csv_path, index=False)
        print(f"📁 Đã lưu {csv_path} ({len(df_split)} dòng)")

if __name__ == "__main__":
    main()