# notebooks/generate_f1_report.py
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import logging
import json

# --- 1. Thiết lập môi trường để import từ thư mục `src` ---
try:
    # Lấy đường dẫn đến thư mục gốc của project
    # Giả định rằng script này nằm trong thư mục `notebooks`
    NOTEBOOK_DIR = Path(__file__).parent.resolve()
    PROJECT_ROOT = NOTEBOOK_DIR.parent
except NameError:
    # Fallback cho môi trường interactive (như Jupyter)
    NOTEBOOK_DIR = Path.cwd()
    PROJECT_ROOT = NOTEBOOK_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.dataset_utils import make_data_loaders
from src.model_utils import get_model

# --- 2. Cấu hình logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluate_f1_score(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: str,
    num_classes: int,
    report_per_class: bool = False
) -> tuple[float, dict[int, float] | None]:
    """
    Đánh giá model trên dataloader và trả về macro F1-score.
    Optionally returns per-class F1 scores.
    Sử dụng 'label_orig' làm nhãn thật.
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["image"].to(device, non_blocking=True)
            # Luôn sử dụng nhãn gốc để đánh giá
            targets = batch["label_orig"].cpu().numpy()

            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    if not all_targets:
        return 0.0, None

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Tính macro F1 score
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    per_class_f1_scores = None
    if report_per_class:
        # Lấy F1 cho từng class
        labels = list(range(num_classes))
        per_class_f1_array = f1_score(all_targets, all_preds, average=None, labels=labels, zero_division=0)
        per_class_f1_scores = {i: score for i, score in enumerate(per_class_f1_array)}

    return float(macro_f1), per_class_f1_scores


def create_main_f1_table(f1_df: pd.DataFrame) -> pd.DataFrame | None:
    """Tạo bảng tóm tắt cho Macro F1-score của các model tốt nhất."""
    logging.info("Generating Main Macro F1-Score Table...")
    
    results = []

    # Tìm iteration tốt nhất cho mỗi lần chạy dựa trên test_macro_f1_orig
    best_iter_df = f1_df.loc[f1_df.groupby(['exp_path'])['test_macro_f1_orig'].idxmax()].copy()
    best_iter_df.rename(columns={'iteration': 'best_iteration'}, inplace=True)

    # Tính delta so với baseline
    baseline_f1s = f1_df[f1_df['iteration'] == 0].drop_duplicates(subset=['noise_ratio']).set_index('noise_ratio')['test_macro_f1_orig'].to_dict()
    best_iter_df['baseline_f1'] = best_iter_df['noise_ratio'].map(baseline_f1s)
    best_iter_df['delta_vs_baseline'] = best_iter_df['test_macro_f1_orig'] - best_iter_df['baseline_f1']

    for noise_ratio, group in f1_df.groupby('noise_ratio'):
        # --- Baseline ---
        baseline_run = group[group['iteration'] == 0].iloc[0]
        results.append({
            'noise_ratio': noise_ratio,
            'method': 'Baseline',
            'test_macro_f1_orig': baseline_run['test_macro_f1_orig'],
            'best_iteration': 0,
            'alpha': '-',
            'delta_vs_baseline': 0.0,
            'is_best': False
        })

        # --- Filter Methods ---
        noise_group_best_iters = best_iter_df[best_iter_df['noise_ratio'] == noise_ratio]

        # Tìm alpha tốt nhất cho mỗi filter_mode
        best_per_mode = noise_group_best_iters.loc[
            noise_group_best_iters.groupby('filter_mode')['test_macro_f1_orig'].idxmax()
        ]

        # Tìm F1 score tốt nhất tổng thể
        if not best_per_mode.empty:
            overall_best_f1 = best_per_mode['test_macro_f1_orig'].max()
        else:
            overall_best_f1 = -1

        for _, row in best_per_mode.iterrows():
            is_best = row['test_macro_f1_orig'] == overall_best_f1
            results.append({
                'noise_ratio': noise_ratio,
                'method': row['filter_mode'],
                'test_macro_f1_orig': row['test_macro_f1_orig'],
                'best_iteration': row['best_iteration'],
                'alpha': row['alpha'],
                'delta_vs_baseline': row['delta_vs_baseline'],
                'is_best': is_best
            })

    if not results:
        return None

    final_table = pd.DataFrame(results)

    # Formatting
    final_table['is_best_marker'] = final_table.apply(
        lambda row: ' ⭐' if row['is_best'] and row['method'] != 'Baseline' else '', axis=1
    )
    final_table['method'] = final_table['method'] + final_table['is_best_marker']
    final_table['test_macro_f1_orig'] = (final_table['test_macro_f1_orig'] * 100).map('{:.2f}%'.format)
    final_table['delta_vs_baseline'] = (final_table['delta_vs_baseline'] * 100).map('{:+.2f}%'.format)

    # Reorder columns
    final_table = final_table[['noise_ratio', 'method', 'test_macro_f1_orig', 'delta_vs_baseline', 'best_iteration', 'alpha']]
    return final_table


def main():
    """
    Hàm chính để quét các thư mục thực nghiệm, tải model, tính toán F1 score và xuất báo cáo.
    """
    # --- CẤU HÌNH ---
    CALCULATE_PER_CLASS_F1 = True # Bật/tắt tính F1 cho từng class
    source_dir = Path("./outputs/cifar10_iter_ema_voting_sweep")
    report_path = Path("./reports/test_macro_f1_report.csv")
    main_report_path = Path("./reports/main_macro_f1_score.csv") # Bảng tóm tắt mới
    report_path.parent.mkdir(exist_ok=True, parents=True)

    # Sử dụng một config cơ bản để lấy các thiết lập chung
    config = TrainConfig()
    device = config.device

    # --- TÌM KIẾM CÁC THỰC NGHIỆM ---
    summary_files = list(source_dir.rglob("experiment_summary.csv"))
    if not summary_files:
        logging.error(f"Không tìm thấy file 'experiment_summary.csv' nào trong: {source_dir}")
        return

    logging.info(f"Tìm thấy {len(summary_files)} thực nghiệm để xử lý.")

    # --- XỬ LÝ TỪNG THỰC NGHIỆM ---
    all_f1_results = []

    for summary_file in tqdm(summary_files, desc="Đang xử lý các thực nghiệm"):
        exp_dir = summary_file.parent
        try:
            iter_df = pd.read_csv(summary_file)
        except Exception as e:
            logging.error(f"Lỗi khi đọc {summary_file}: {e}")
            continue

        # Lấy đường dẫn đến file test.csv từ manifest
        manifest_path = exp_dir / "manifest.json"
        if not manifest_path.exists():
            logging.warning(f"Không tìm thấy manifest.json trong {exp_dir}, bỏ qua.")
            continue
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        test_csv_path = manifest.get("test_csv")
        if not test_csv_path or not Path(test_csv_path).exists():
            logging.warning(f"Đường dẫn test_csv không hợp lệ trong manifest của {exp_dir}, bỏ qua.")
            continue

        # Lấy tên các lớp một lần cho mỗi thực nghiệm
        class_names = None
        if CALCULATE_PER_CLASS_F1:
            # Đọc file test.csv để lấy tên các lớp
            test_df_for_classes = pd.read_csv(test_csv_path)
            # Sắp xếp để đảm bảo thứ tự nhất quán
            class_names = sorted(test_df_for_classes['class_name'].unique())
            logging.info(f"Đã tìm thấy các lớp: {class_names}")

        # Tạo test dataloader một lần cho mỗi thực nghiệm
        dls = make_data_loaders(
            train_csv=test_csv_path,  # dummy, không sử dụng
            val_csv=test_csv_path,    # dummy, không sử dụng
            test_csv=test_csv_path,
            config=config,
            train_label_col='label_orig' # dummy
        )
        test_loader = dls['test']

        # Lặp qua từng iteration trong file summary
        for _, row in iter_df.iterrows():
            iteration = int(row['iteration'])

            # Xây dựng đường dẫn đến checkpoint
            checkpoint_path = exp_dir / f"iteration_{iteration}" / "checkpoints" / f"model_iter{iteration}_best.pth"

            if not checkpoint_path.exists():
                logging.warning(f"Không tìm thấy checkpoint, bỏ qua: {checkpoint_path}")
                continue

            # Tải model
            model = get_model(num_classes=config.num_classes, pretrained=False, device=device)
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            except Exception as e:
                logging.error(f"Lỗi khi tải checkpoint {checkpoint_path}: {e}")
                continue

            # Tính toán F1 score
            macro_f1, per_class_f1 = evaluate_f1_score(
                model, test_loader, device, 
                num_classes=config.num_classes, 
                report_per_class=CALCULATE_PER_CLASS_F1
            )

            # Lưu kết quả
            result_row = {
                'noise_ratio': row['noise_ratio'], 'alpha': row['alpha'],
                'filter_mode': row['filter_mode'], 'iteration': iteration,
                'test_macro_f1_orig': macro_f1, 'exp_path': str(exp_dir)
            }

            if per_class_f1 and class_names:
                for class_idx, f1_val in per_class_f1.items():
                    class_name = class_names[class_idx]
                    result_row[f'f1_{class_name}'] = f1_val
            all_f1_results.append(result_row)

    # --- LƯU BÁO CÁO ---
    if not all_f1_results:
        logging.info("Không có kết quả nào để lưu.")
        return

    f1_df = pd.DataFrame(all_f1_results)
    f1_df.to_csv(report_path, index=False)
    logging.info(f"Đã tạo thành công báo cáo Macro F1 chi tiết tại: {report_path}")
    print("\n--- Báo cáo Macro F1 chi tiết (Mẫu) ---")
    print(f1_df.head().to_markdown(index=False))
    print("---------------------------------------")

    # --- TẠO VÀ LƯU BẢNG TÓM TẮT ---
    main_f1_table = create_main_f1_table(f1_df)
    if main_f1_table is not None:
        main_f1_table.to_csv(main_report_path, index=False)
        logging.info(f"Đã tạo thành công bảng tóm tắt Macro F1 tại: {main_report_path}")
        print("\n--- Bảng tóm tắt Macro F1-Score cho Model tốt nhất ---")
        print(main_f1_table.to_markdown(index=False))
        print("------------------------------------------------------")


if __name__ == "__main__":
    main()
