"""
Enhanced EDA module for analyzing noisy label datasets.
Provides visualizations for class distribution, confusion matrices, and noise patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


def plot_class_distribution_per_noise(csv_paths: Dict[str, str], 
                                      noise_ratio: float,
                                      save_dir: str) -> None:
    """
    Vẽ biểu đồ phân bố số lượng mẫu của từng class theo các tập train/val/test.
    
    Args:
        csv_paths: Dictionary chứa đường dẫn đến train_csv, val_csv, test_csv
        noise_ratio: Tỷ lệ nhiễu được sử dụng (để đặt tên file)
        save_dir: Thư mục lưu kết quả
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Đọc dữ liệu từ các CSV
    train_df = pd.read_csv(csv_paths['train_csv'])
    val_df = pd.read_csv(csv_paths['val_csv'])
    test_df = pd.read_csv(csv_paths['test_csv'])
    
    # Tạo figure với 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Class Distribution (Noise Ratio: {noise_ratio})', 
                 fontsize=16, fontweight='bold')
    
    datasets = {
        'Train': train_df,
        'Validation': val_df,
        'Test': test_df
    }
    
    for idx, (dataset_name, df) in enumerate(datasets.items()):
        ax = axes[idx]
        
        # Đếm số lượng samples theo label_noisy (nhãn thực tế được sử dụng)
        class_counts = df['label_noisy'].value_counts().sort_index()
        
        # Vẽ bar chart
        bars = ax.bar(class_counts.index, class_counts.values, 
                      color=plt.cm.tab10(np.arange(len(class_counts)) % 10),
                      edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Class Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset_name} Set (n={len(df)})', 
                    fontsize=13, fontweight='bold')
        ax.set_xticks(range(10))
        ax.grid(axis='y', alpha=0.3)
        
        # Thêm số lượng lên mỗi cột
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = Path(save_dir) / f'class_distribution_noise_{noise_ratio}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved class distribution plot to {save_path}")


def plot_noise_confusion_matrix_counts(csv_path: str, 
                                       noise_ratio: float,
                                       save_dir: str,
                                       dataset_name: str = 'train') -> None:
    """
    Vẽ confusion matrix thể hiện số lượng samples giữa nhãn gốc và nhãn nhiễu.
    Mỗi ô (i,j) cho biết có bao nhiêu samples có nhãn gốc là i nhưng nhãn nhiễu là j.
    
    Args:
        csv_path: Đường dẫn đến file CSV
        noise_ratio: Tỷ lệ nhiễu
        save_dir: Thư mục lưu kết quả
        dataset_name: Tên dataset (train/val/test)
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # Tạo confusion matrix
    num_classes = 10
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for _, row in df.iterrows():
        orig = int(row['label_orig'])
        noisy = int(row['label_noisy'])
        confusion_matrix[orig, noisy] += 1
    
    # Vẽ heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlOrRd', 
                cbar_kws={'label': 'Sample Count'},
                linewidths=0.5, linecolor='gray', ax=ax)
    
    ax.set_xlabel('Noisy Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('Original Label', fontsize=13, fontweight='bold')
    ax.set_title(f'Noise Confusion Matrix - Counts\n'
                f'{dataset_name.capitalize()} Set (Noise Ratio: {noise_ratio})',
                fontsize=14, fontweight='bold', pad=20)
    
    # Thêm thống kê
    total_samples = confusion_matrix.sum()
    noisy_samples = total_samples - np.trace(confusion_matrix)
    actual_noise_ratio = noisy_samples / total_samples if total_samples > 0 else 0
    
    textstr = f'Total Samples: {total_samples}\n'
    textstr += f'Noisy Samples: {noisy_samples}\n'
    textstr += f'Actual Noise Ratio: {actual_noise_ratio:.2%}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    save_path = Path(save_dir) / f'confusion_matrix_counts_{dataset_name}_noise_{noise_ratio}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix (counts) to {save_path}")


def plot_noise_confusion_matrix_ratios(csv_path: str, 
                                       noise_ratio: float,
                                       save_dir: str,
                                       dataset_name: str = 'train') -> None:
    """
    Vẽ confusion matrix thể hiện tỷ lệ (%) samples giữa nhãn gốc và nhãn nhiễu.
    Mỗi hàng được chuẩn hóa về 100% để dễ quan sát tỷ lệ nhiễu của từng class.
    
    Args:
        csv_path: Đường dẫn đến file CSV
        noise_ratio: Tỷ lệ nhiễu
        save_dir: Thư mục lưu kết quả
        dataset_name: Tên dataset (train/val/test)
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # Tạo confusion matrix
    num_classes = 10
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=float)
    
    for _, row in df.iterrows():
        orig = int(row['label_orig'])
        noisy = int(row['label_noisy'])
        confusion_matrix[orig, noisy] += 1
    
    # Chuẩn hóa theo hàng (mỗi hàng = 100%)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Tránh chia cho 0
    confusion_matrix_normalized = (confusion_matrix / row_sums) * 100
    
    # Vẽ heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(confusion_matrix_normalized, annot=True, fmt='.1f', 
                cmap='RdYlGn_r', cbar_kws={'label': 'Percentage (%)'},
                linewidths=0.5, linecolor='gray', ax=ax,
                vmin=0, vmax=100)
    
    ax.set_xlabel('Noisy Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('Original Label', fontsize=13, fontweight='bold')
    ax.set_title(f'Noise Confusion Matrix - Percentages\n'
                f'{dataset_name.capitalize()} Set (Noise Ratio: {noise_ratio})',
                fontsize=14, fontweight='bold', pad=20)
    
    # Thêm chú thích
    textstr = 'Each row sums to 100%\n'
    textstr += 'Diagonal: Clean samples\n'
    textstr += 'Off-diagonal: Noisy samples'
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    save_path = Path(save_dir) / f'confusion_matrix_ratios_{dataset_name}_noise_{noise_ratio}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix (ratios) to {save_path}")


def generate_noise_analysis_report(csv_paths: Dict[str, str],
                                   noise_ratio: float,
                                   save_dir: str) -> None:
    """
    Tạo báo cáo phân tích chi tiết về noise pattern trong dataset.
    
    Args:
        csv_paths: Dictionary chứa đường dẫn đến train_csv, val_csv, test_csv
        noise_ratio: Tỷ lệ nhiễu được sử dụng
        save_dir: Thư mục lưu kết quả
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append(f"NOISE ANALYSIS REPORT - Noise Ratio: {noise_ratio}")
    report_lines.append("="*80)
    report_lines.append("")
    
    for dataset_name, csv_key in [('Train', 'train_csv'), 
                                   ('Validation', 'val_csv'), 
                                   ('Test', 'test_csv')]:
        df = pd.read_csv(csv_paths[csv_key])
        
        total = len(df)
        noisy = df['noise_flag'].sum()
        clean = total - noisy
        
        report_lines.append(f"{dataset_name} Dataset:")
        report_lines.append(f"  Total samples: {total}")
        report_lines.append(f"  Clean samples: {clean} ({clean/total*100:.2f}%)")
        report_lines.append(f"  Noisy samples: {noisy} ({noisy/total*100:.2f}%)")
        report_lines.append("")
        
        # Phân tích noise per class
        report_lines.append(f"  Noise distribution by original class:")
        for class_idx in range(10):
            class_mask = df['label_orig'] == class_idx
            class_total = class_mask.sum()
            class_noisy = (class_mask & (df['noise_flag'] == 1)).sum()
            if class_total > 0:
                noise_pct = class_noisy / class_total * 100
                report_lines.append(f"    Class {class_idx}: {class_noisy}/{class_total} "
                                  f"({noise_pct:.1f}%) noisy")
        report_lines.append("")
        report_lines.append("-"*80)
        report_lines.append("")
    
    # Lưu report
    report_path = Path(save_dir) / f'noise_analysis_report_noise_{noise_ratio}.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Saved noise analysis report to {report_path}")


def perform_complete_eda(csv_paths: Dict[str, str],
                        noise_ratio: float,
                        save_dir: str) -> None:
    """
    Thực hiện toàn bộ EDA analysis bao gồm:
    - Class distribution plots
    - Confusion matrices (counts và ratios)
    - Noise analysis report
    
    Args:
        csv_paths: Dictionary chứa đường dẫn đến train_csv, val_csv, test_csv
        noise_ratio: Tỷ lệ nhiễu được sử dụng
        save_dir: Thư mục lưu kết quả
    """
    logger.info(f"Starting complete EDA for noise_ratio={noise_ratio}")
    
    # 1. Class distribution
    plot_class_distribution_per_noise(csv_paths, noise_ratio, save_dir)
    
    # 2. Confusion matrices cho train và val (test không có noise)
    for dataset_name, csv_key in [('train', 'train_csv'), ('val', 'val_csv')]:
        plot_noise_confusion_matrix_counts(csv_paths[csv_key], noise_ratio, 
                                          save_dir, dataset_name)
        plot_noise_confusion_matrix_ratios(csv_paths[csv_key], noise_ratio, 
                                          save_dir, dataset_name)
    
    # 3. Generate text report
    generate_noise_analysis_report(csv_paths, noise_ratio, save_dir)
    
    logger.info(f"Completed EDA for noise_ratio={noise_ratio}")