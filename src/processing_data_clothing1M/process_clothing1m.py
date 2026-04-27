import os
import shutil
import tarfile
import glob
import pandas as pd
from pathlib import Path

# --- CẤU HÌNH ĐƯỜNG DẪN ---
INPUT_DIR = Path("/mnt/d/data_noise_label/clothing1M")
OUTPUT_DIR = Path("/mnt/c/Users/truon/learning/ptit/research/trung/M_10_01_2025/code_v2/project/notebooks/data_clothing1M")

def load_kv(filepath):
    """Đọc file key-value (nhãn clean/noisy)"""
    kv = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                kv[parts[0]] = int(parts[1])
    return kv

def load_keys(filepath):
    """Đọc file danh sách (train/val/test key list)"""
    keys = set()
    with open(filepath, 'r') as f:
        for line in f:
            keys.add(line.strip().split()[0])
    return keys

def load_classes(filepath):
    """Đọc tên class"""
    classes = {}
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            classes[i] = line.strip()
    return classes

def main():
    # 1. KHỞI TẠO CÁC FOLDER ĐÍCH
    raw_zips_dir = INPUT_DIR / "raw_zips"
    ann_dir = INPUT_DIR / "raw_annotations"
    
    images_out_dir = OUTPUT_DIR / "images"
    meta_out_dir = OUTPUT_DIR / "meta_data"
    csv_out_dir = OUTPUT_DIR / "csvs"
    
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(meta_out_dir, exist_ok=True)
    os.makedirs(csv_out_dir, exist_ok=True)

    print("Bước 1: Đọc và tính toán tập giao (Intersection)...")
    clean_labels = load_kv(ann_dir / "clean_label_kv.txt")
    noisy_labels = load_kv(ann_dir / "noisy_label_kv.txt")
    class_names = load_classes(ann_dir / "category_names_eng.txt")

    train_keys = load_keys(ann_dir / "clean_train_key_list.txt")
    val_keys = load_keys(ann_dir / "clean_val_key_list.txt")
    test_keys = load_keys(ann_dir / "clean_test_key_list.txt")

    # Lọc ra những ảnh TỒN TẠI ở cả Clean Label và Noisy Label
    def get_overlap(split_keys):
        overlap = {}
        for k in split_keys:
            if k in clean_labels and k in noisy_labels:
                overlap[k] = {
                    "label_orig": clean_labels[k],
                    "label_noisy": noisy_labels[k]
                }
        return overlap

    target_train = get_overlap(train_keys)
    target_val = get_overlap(val_keys)
    target_test = get_overlap(test_keys)
    
    # Gom tất cả các key (dạng: "images/0/00/1278541879,867148000.jpg")
    all_target_keys = set(target_train.keys()) | set(target_val.keys()) | set(target_test.keys())
    print(f" -> Tổng số ảnh hợp lệ cần lấy (Overlap): {len(all_target_keys)}")

    # Tạo mapping để match với cấu trúc bên trong file .tar
    # Key là path trong file tar: "0/00/xxx.jpg", Value là path gốc: "images/0/00/xxx.jpg"
    tar_path_to_key = {k.replace("images/", "", 1): k for k in all_target_keys}

    print("\nBước 2: Quét file .tar và giải nén (Extract)...")
    tar_files = glob.glob(str(raw_zips_dir / "*.tar"))
    extracted_keys = set() # Set lưu lại những file đã thực sự giải nén thành công

    for tar_path in sorted(tar_files):
        print(f" -> Đang xử lý {os.path.basename(tar_path)}...")
        with tarfile.open(tar_path, 'r') as tar:
            members_to_extract = []
            keys_in_this_tar = []

            for m in tar.getmembers():
                if m.isfile() and m.name in tar_path_to_key:
                    members_to_extract.append(m)
                    keys_in_this_tar.append(tar_path_to_key[m.name])
            
            if members_to_extract:
                # Trích xuất thẳng vào OUTPUT_DIR/images
                # m.name là "0/00/xxx.jpg", nên ảnh sẽ lưu tại OUTPUT_DIR/images/0/00/xxx.jpg
                tar.extractall(path=images_out_dir, members=members_to_extract)
                extracted_keys.update(keys_in_this_tar)
                print(f"    + Giải nén {len(members_to_extract)} ảnh.")

    print(f" -> Hoàn tất giải nén: {len(extracted_keys)}/{len(all_target_keys)} ảnh hợp lệ.")

    print("\nBước 3: Tạo files CSV chuẩn cho Training...")
    # Hàm builder chỉ lấy những file CÓ THẬT (đã giải nén)
    def build_csv_rows(target_dict, split_name):
        rows = []
        for key, labels in target_dict.items():
            if key in extracted_keys:
                lbl_orig = labels["label_orig"]
                lbl_noisy = labels["label_noisy"]
                
                # Ở đây `key` là "images/0/00/...". 
                # Nối thẳng với OUTPUT_DIR ta được: ".../data_clothing1M/images/0/00/..." (Cực kỳ chuẩn xác)
                absolute_img_path = OUTPUT_DIR / key
                
                rows.append({
                    "image_path": str(absolute_img_path),
                    "label_noisy": lbl_noisy,
                    "label_orig": lbl_orig,
                    "class_name": class_names.get(lbl_orig, f"Class_{lbl_orig}"),
                    "split": split_name,
                    "noise_flag": 1 if lbl_noisy != lbl_orig else 0
                })
        return rows

    all_rows = []
    all_rows.extend(build_csv_rows(target_train, "train"))
    all_rows.extend(build_csv_rows(target_val, "val"))
    all_rows.extend(build_csv_rows(target_test, "test"))

    df_all = pd.DataFrame(all_rows)
    # Đánh index global đúng chuẩn như ImageNet100
    df_all = df_all.reset_index(drop=True).reset_index().rename(columns={"index": "global_index"})
    df_all.rename(columns={"global_index": "index"}, inplace=True)

    # Lưu CSV
    df_all[df_all["split"] == "train"].to_csv(csv_out_dir / "train.csv", index=False)
    df_all[df_all["split"] == "val"].to_csv(csv_out_dir / "val.csv", index=False)
    df_all[df_all["split"] == "test"].to_csv(csv_out_dir / "test.csv", index=False)

    print(f" -> Đã tạo xong train.csv ({len(df_all[df_all['split'] == 'train'])} mẫu), val.csv, test.csv")

    print("\nBước 4: Copy và backup Meta-data...")
    # Copy toàn bộ txt sang thư mục meta_data
    for txt_file in glob.glob(str(ann_dir / "*.txt")):
        shutil.copy(txt_file, meta_out_dir)
    
    # Save danh sách những file thực sự được dùng (Overlap keys)
    with open(meta_out_dir / "extracted_overlap_keys.txt", "w") as f:
        for k in sorted(extracted_keys):
            f.write(k + "\n")

    print(f"\n✅ HOÀN TẤT! Toàn bộ Data, Meta-data và CSV đã nằm sẵn sàng tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()