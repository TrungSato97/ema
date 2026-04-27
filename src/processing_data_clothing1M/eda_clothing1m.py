import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- CẤU HÌNH ĐƯỜNG DẪN TỚI THƯ MỤC PROJECT ---
BASE_DIR = "/mnt/c/Users/truon/learning/ptit/research/trung/M_10_01_2025/code_v2/project/notebooks/data_clothing1M"
CSV_DIR = os.path.join(BASE_DIR, "csvs")
RESULT_DIR = os.path.join(BASE_DIR, "eda_results")
os.makedirs(RESULT_DIR, exist_ok=True)

# Thiết lập theme y hệt file code cũ
sns.set_theme(style="whitegrid")

def plot_comprehensive_distribution(df_dict: dict, order: list, result_dir: str):
    """Vẽ biểu đồ phân bổ nhãn gốc (Ground Truth) cho 3 tập Train, Val, Test."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
    splits = ['train', 'val', 'test']
    
    for ax, split in zip(axes, splits):
        df = df_dict[split]
        total = len(df)
        
        sns.countplot(data=df, y='class_name', order=order, ax=ax, palette='viridis', hue='class_name', legend=False)
        ax.set_title(f'TẬP {split.upper()} ({total:,})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Số lượng', fontsize=12)
        ax.set_ylabel('Danh mục' if split == 'train' else '')
        
        # Hiển thị số lượng và %
        for container in ax.containers:
            labels = [f'{v.get_width():,.0f} ({100 * v.get_width() / total:.1f}%)' if total > 0 else '0 (0%)' for v in container]
            ax.bar_label(container, labels=labels, padding=5, fontsize=10)
            
        # Mở rộng trục X để chữ không bị lẹm
        max_width = max([p.get_width() for p in ax.patches]) if len(ax.patches) > 0 else 1
        ax.set_xlim(0, max_width * 1.35)
        
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'dist_clean_splits.png'), dpi=300)
    plt.close()

def plot_noisy_vs_clean(df_train: pd.DataFrame, order: list, result_dir: str):
    """Vẽ so sánh phân bổ giữa nhãn gốc (Clean) và nhãn nhiễu (Noisy) trên tập Train."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    total = len(df_train)
    
    # 1. Biểu đồ nhãn gốc
    sns.countplot(data=df_train, y='class_name', order=order, ax=axes[0], palette='viridis', hue='class_name', legend=False)
    axes[0].set_title(f'NHÃN GỐC (Clean) - Tập TRAIN ({total:,})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Số lượng', fontsize=12)
    axes[0].set_ylabel('Danh mục', fontsize=12)
    
    for container in axes[0].containers:
        labels = [f'{v.get_width():,.0f} ({100 * v.get_width() / total:.1f}%)' if total > 0 else '0 (0%)' for v in container]
        axes[0].bar_label(container, labels=labels, padding=5, fontsize=10)
        
    max_width_0 = max([p.get_width() for p in axes[0].patches]) if len(axes[0].patches) > 0 else 1
    axes[0].set_xlim(0, max_width_0 * 1.35)

    # 2. Tạo mapping tên cho nhãn nhiễu để hiển thị đồng bộ
    label_to_class = dict(zip(df_train['label_orig'], df_train['class_name']))
    df_train['class_name_noisy'] = df_train['label_noisy'].map(label_to_class)
    
    # 3. Biểu đồ nhãn nhiễu
    sns.countplot(data=df_train, y='class_name_noisy', order=order, ax=axes[1], palette='magma', hue='class_name_noisy', legend=False)
    axes[1].set_title(f'NHÃN NHIỄU (Noisy) - Tập TRAIN ({total:,})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Số lượng', fontsize=12)
    axes[1].set_ylabel('')
    
    for container in axes[1].containers:
        labels = [f'{v.get_width():,.0f} ({100 * v.get_width() / total:.1f}%)' if total > 0 else '0 (0%)' for v in container]
        axes[1].bar_label(container, labels=labels, padding=5, fontsize=10)
        
    max_width_1 = max([p.get_width() for p in axes[1].patches]) if len(axes[1].patches) > 0 else 1
    axes[1].set_xlim(0, max_width_1 * 1.35)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'dist_noisy_vs_clean_train.png'), dpi=300)
    plt.close()

def plot_noise_transition_matrix(df_train: pd.DataFrame, result_dir: str):
    """Vẽ Ma trận nhiễu (Noise Transition Matrix)."""
    # Lấy danh sách tên class theo đúng thứ tự ID của nhãn
    label_to_class = dict(zip(df_train['label_orig'], df_train['class_name']))
    labels_sorted = sorted(label_to_class.keys())
    class_names_sorted = [label_to_class[l] for l in labels_sorted]
    
    # Tính Confusion Matrix
    cm = confusion_matrix(df_train['label_orig'], df_train['label_noisy'], labels=labels_sorted)
    # Normalize theo hàng (100% nhãn thật phân bổ đi đâu)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', 
                xticklabels=class_names_sorted, yticklabels=class_names_sorted)
    plt.title('MA TRẬN CHUYỂN ĐỔI NHIỄU THỰC TẾ (Tập TRAIN)', fontsize=16, fontweight='bold')
    plt.xlabel('Nhãn Nhiễu (Dự đoán / Gán sai)', fontsize=12)
    plt.ylabel('Nhãn Gốc (Ground Truth Clean)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'noise_transition_matrix.png'), dpi=300)
    plt.close()

def main():
    print(f"Đang đọc dữ liệu CSV từ: {CSV_DIR}...")
    df_train = pd.read_csv(os.path.join(CSV_DIR, "train.csv"))
    df_val = pd.read_csv(os.path.join(CSV_DIR, "val.csv"))
    df_test = pd.read_csv(os.path.join(CSV_DIR, "test.csv"))
    
    df_dict = {'train': df_train, 'val': df_val, 'test': df_test}
    
    # Sắp xếp thứ tự danh mục theo tần suất xuất hiện (để biểu đồ đẹp và dễ nhìn)
    order = df_train['class_name'].value_counts().index.tolist()
    
    print("1. Đang vẽ biểu đồ phân phối nhãn gốc (Clean) cho cả 3 tập...")
    plot_comprehensive_distribution(df_dict, order, RESULT_DIR)
    
    print("2. Đang vẽ biểu đồ so sánh Nhãn Gốc và Nhãn Nhiễu (tập Train)...")
    plot_noisy_vs_clean(df_train, order, RESULT_DIR)
    
    print("3. Đang tính toán Tỷ lệ Nhiễu và vẽ Ma Trận Nhiễu...")
    noisy_count = df_train['noise_flag'].sum()
    noise_rate = noisy_count / len(df_train) * 100
    print(f"   -> Tỷ lệ nhiễu thực tế tập Train: {noise_rate:.2f}% ({noisy_count:,}/{len(df_train):,} ảnh)")
    
    plot_noise_transition_matrix(df_train, RESULT_DIR)
    
    print(f"✅ Hoàn tất EDA! Các biểu đồ chất lượng cao đã được lưu tại: {RESULT_DIR}")

if __name__ == "__main__":
    main()