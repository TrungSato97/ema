# scripts/reporting/generate_report_visuals.py
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100 # Chất lượng hình ảnh tốt hơn một chút


def aggregate_summary_results(source_dir: Path) -> Optional[pd.DataFrame]:
    """
    Recursively finds all 'experiment_summary.csv' files, reads them,
    and aggregates them into a single DataFrame.
    Extracts noise_ratio, alpha, and filter_mode from the folder structure.
    """
    summary_files = list(source_dir.rglob("experiment_summary.csv"))
    if not summary_files:
        logging.warning(f"No 'experiment_summary.csv' files found in {source_dir}")
        return None

    all_dfs = []
    for f in summary_files:
        try:
            df = pd.read_csv(f)
            # Extract params from path
            parts = f.parts
            # Example path: .../noise_0.8/alpha_0.3/mode_vote_relabel/experiment_summary.csv
            noise_str = next((p for p in parts if p.startswith("noise_")), None)
            alpha_str = next((p for p in parts if p.startswith("alpha_")), None)
            mode_str = next((p for p in parts if p.startswith("mode_")), None)

            if noise_str:
                df['noise_ratio'] = float(re.search(r'[\d.]+', noise_str).group())
            if alpha_str:
                df['alpha'] = float(re.search(r'[\d.]+', alpha_str).group())
            if mode_str:
                df['filter_mode'] = mode_str.replace("mode_", "")

            df['exp_path'] = str(f.parent)
            all_dfs.append(df)
        except Exception as e:
            logging.error(f"Could not process file {f}: {e}")

    if not all_dfs:
        return None

    full_df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Aggregated {len(full_df)} rows from {len(summary_files)} summary files.")
    return full_df


def create_main_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """(1) Creates the main results table comparing Baseline vs. methods."""
    logging.info("Generating Main Results Table...")

    results = []

    # Find best iteration for each run based on test_acc_orig
    best_iter_df = df.loc[df.groupby(['exp_path'])['test_acc_orig'].idxmax()].copy()
    best_iter_df.rename(columns={'iteration': 'best_iteration'}, inplace=True)

    # Tạo cột delta để so sánh với baseline
    baseline_accs = df[df['iteration'] == 0].set_index('noise_ratio')['test_acc_orig'].to_dict()
    best_iter_df['baseline_acc'] = best_iter_df['noise_ratio'].map(baseline_accs)
    best_iter_df['delta_vs_baseline'] = best_iter_df['test_acc_orig'] - best_iter_df['baseline_acc']

    for noise_ratio, group in df.groupby('noise_ratio'):
        # --- Baseline ---
        # Baseline is iteration 0. All filter modes/alphas are the same at this point.
        baseline_run = group[group['iteration'] == 0].iloc[0]
        results.append({
            'noise_ratio': noise_ratio,
            'method': 'Baseline',
            'test_acc_orig': baseline_run['test_acc_orig'],
            'val_acc_orig': baseline_run['val_acc_orig'],
            'best_iteration': 0,
            'kept_ratio': 1.0,
            'training_samples_used': baseline_run['training_samples_used'],
            'alpha': '-',
            'delta_vs_baseline': 0.0,
            'is_best': False
        })

        # --- Filter Methods ---
        noise_group_best_iters = best_iter_df[best_iter_df['noise_ratio'] == noise_ratio]

        # Find best run (best alpha) for each filter_mode
        best_per_mode = noise_group_best_iters.loc[
            noise_group_best_iters.groupby('filter_mode')['test_acc_orig'].idxmax()
        ]

        # Find the overall best test_acc among the methods for this noise_ratio
        if not best_per_mode.empty:
            overall_best_acc = best_per_mode['test_acc_orig'].max()
        else:
            overall_best_acc = -1

        for _, row in best_per_mode.iterrows():
            is_best = row['test_acc_orig'] == overall_best_acc
            results.append({
                'noise_ratio': noise_ratio,
                'method': row['filter_mode'],
                'test_acc_orig': row['test_acc_orig'],
                'val_acc_orig': row['val_acc_orig'],
                'best_iteration': row['best_iteration'],
                'kept_ratio': row['kept_ratio'],
                'training_samples_used': row['training_samples_used'],
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

    final_table['test_acc_orig'] = (final_table['test_acc_orig'] * 100).map('{:.2f}%'.format)
    final_table['val_acc_orig'] = (final_table['val_acc_orig'] * 100).map('{:.2f}%'.format)
    final_table['delta_vs_baseline'] = (final_table['delta_vs_baseline'] * 100).map('{:+.2f}%'.format)
    final_table['kept_ratio'] = (final_table['kept_ratio'] * 100).map('{:.2f}%'.format)

    # Reorder columns

    return final_table


def create_filter_statistics_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Creates a detailed table of filter performance statistics at the best iteration."""
    logging.info("Generating Filter Statistics Table...")

    stats_rows = []

    # Find the best run (best alpha) for each (noise_ratio, filter_mode) combination
    best_iter_df = df.loc[df.groupby('exp_path')['test_acc_orig'].idxmax()]
    best_runs = best_iter_df.loc[
        best_iter_df.groupby(['noise_ratio', 'filter_mode'])['test_acc_orig'].idxmax()
    ]

    for _, row in best_runs.iterrows():
        exp_path = Path(row['exp_path'])
        noise_ratio = row['noise_ratio']
        filter_mode = row['filter_mode']
        best_iter = int(row['iteration'])

        selection_file = exp_path / f"iteration_{best_iter}" / "selection" / f"selection_iter_{best_iter}.csv"

        if not selection_file.exists():
            logging.warning(f"Selection file not found for {noise_ratio}/{filter_mode}, skipping: {selection_file}")
            continue

        sel_df = pd.read_csv(selection_file)

        # --- Calculate Confusion Matrix components for the filter ---
        # Positive class: 'clean' (noise_flag == 0)
        # Filter's prediction: 'kept' is predicting 'clean'

        tp = len(sel_df[(sel_df['noise_flag'] == 0) & (sel_df['filter_flag'] == 'kept')])
        fp = len(sel_df[(sel_df['noise_flag'] == 1) & (sel_df['filter_flag'] == 'kept')])
        fn = len(sel_df[(sel_df['noise_flag'] == 0) & (sel_df['filter_flag'] == 'removed')])
        tn = len(sel_df[(sel_df['noise_flag'] == 1) & (sel_df['filter_flag'] == 'removed')])

        total_kept = tp + fp
        total_clean_in_dataset = tp + fn

        # --- Calculate Metrics ---
        precision = tp / total_kept if total_kept > 0 else 0.0
        recall = tp / total_clean_in_dataset if total_clean_in_dataset > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        stats_rows.append({
            'noise_ratio': noise_ratio,
            'filter_mode': filter_mode,
            'best_iteration': best_iter,
            'total_samples_kept': total_kept,
            'kept_were_clean (TP)': tp,
            'kept_were_noisy (FP)': fp,
            'removed_were_clean (FN)': fn,
            'removed_were_noisy (TN)': tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        })

    if not stats_rows:
        return None

    stats_table = pd.DataFrame(stats_rows).sort_values(by=['noise_ratio', 'filter_mode'])

    # Formatting
    for col in ['precision', 'recall', 'f1_score']:
        stats_table[col] = stats_table[col].map('{:.4f}'.format)

    return stats_table


def create_detailed_filter_statistics_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Creates a detailed table of filter performance statistics for EVERY iteration."""
    logging.info("Generating DETAILED Filter Statistics Table for every iteration...")

    all_stats_rows = []

    # Find the best run (alpha) for each (noise_ratio, filter_mode) combination
    best_iter_df = df.loc[df.groupby('exp_path')['test_acc_orig'].idxmax()]
    best_runs = best_iter_df.loc[
        best_iter_df.groupby(['noise_ratio', 'filter_mode'])['test_acc_orig'].idxmax()
    ]

    for _, best_run_row in best_runs.iterrows():
        exp_path = Path(best_run_row['exp_path'])
        noise_ratio = best_run_row['noise_ratio']
        filter_mode = best_run_row['filter_mode']
        alpha = best_run_row['alpha']

        iter_dirs = sorted(exp_path.glob("iteration_*"), key=lambda p: int(p.name.split('_')[-1]))

        for iter_dir in iter_dirs:
            iter_num = int(iter_dir.name.split('_')[-1])
            selection_file = iter_dir / "selection" / f"selection_iter_{iter_num}.csv"

            if not selection_file.exists():
                continue

            sel_df = pd.read_csv(selection_file)

            tp = len(sel_df[(sel_df['noise_flag'] == 0) & (sel_df['filter_flag'] == 'kept')])
            fp = len(sel_df[(sel_df['noise_flag'] == 1) & (sel_df['filter_flag'] == 'kept')])
            fn = len(sel_df[(sel_df['noise_flag'] == 0) & (sel_df['filter_flag'] == 'removed')])
            tn = len(sel_df[(sel_df['noise_flag'] == 1) & (sel_df['filter_flag'] == 'removed')])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            all_stats_rows.append({
                'noise_ratio': noise_ratio,
                'filter_mode': filter_mode,
                'alpha': alpha,
                'iteration': iter_num,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn,
            })

    if not all_stats_rows:
        return None

    stats_table = pd.DataFrame(all_stats_rows)
    return stats_table


def plot_ablation_alpha(df: pd.DataFrame, report_dir: Path):
    """(2) Plots ablation study for alpha."""
    logging.info("Plotting Ablation Study for Alpha...")

    best_test_acc = df.loc[df.groupby(['exp_path'])['test_acc_orig'].idxmax()]

    for noise_ratio, group in best_test_acc.groupby('noise_ratio'):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot for each filter_mode
        sns.lineplot(data=group, x='alpha', y='test_acc_orig', hue='filter_mode', marker='o', errorbar='sd')
        
        plt.title(f'Test Accuracy vs. Alpha (Noise Ratio: {noise_ratio})')
        plt.xlabel('Alpha (EMA coefficient)')
        plt.ylabel('Best Test Accuracy (on original labels)')
        plt.grid(True)
        plt.legend(title='Filter Mode')
        
        sns.lineplot(data=group, x='alpha', y='test_acc_orig', hue='filter_mode', marker='o', errorbar='sd', ax=ax)

        # Add annotations for max points
        for mode in group['filter_mode'].unique():
            mode_data = group[group['filter_mode'] == mode]
            if not mode_data.empty:
                best_point = mode_data.loc[mode_data['test_acc_orig'].idxmax()]
                ax.annotate(f"{best_point['test_acc_orig']:.3f}",
                             (best_point['alpha'], best_point['test_acc_orig']),
                             textcoords="offset points",
                             xytext=(0,10),
                             ha='center',
                             fontsize=9,
                             arrowprops=dict(arrowstyle="->", color='black'))

        ax.set_title(f'Test Accuracy vs. Alpha (Noise Ratio: {noise_ratio})')
        ax.set_xlabel('Alpha (EMA coefficient)')
        ax.set_ylabel('Best Test Accuracy (on original labels)')
        ax.grid(True)
        ax.legend(title='Filter Mode')

        plot_path = report_dir / f"ablation_alpha_noise_{noise_ratio}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved alpha ablation plot to {plot_path}")


def plot_kept_ratio_over_time(df: pd.DataFrame, report_dir: Path):
    """(3) Plots kept_ratio over iterations."""
    logging.info("Plotting Kept Ratio over Iterations...")
    
    # Find best alpha and mode for visualization
    best_settings = df.loc[df.groupby(['noise_ratio', 'iteration'])['test_acc_orig'].idxmax()]

    for noise_ratio, group in best_settings.groupby('noise_ratio'):
        plt.figure(figsize=(12, 7))
        
        sns.lineplot(data=group, x='iteration', y='kept_ratio', hue='filter_mode', marker='o')
        
    best_alpha_per_mode = df.loc[df.groupby(['noise_ratio', 'filter_mode'])['test_acc_orig'].idxmax()]

    for noise_ratio, group in best_alpha_per_mode.groupby('noise_ratio'):
        plot_data = pd.DataFrame()
        for _, setting in group.iterrows():
            run_data = df[df['exp_path'] == setting['exp_path']]
            plot_data = pd.concat([plot_data, run_data])

        if plot_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(data=plot_data, x='iteration', y='kept_ratio', hue='filter_mode', marker='o', ax=ax)

        # Expected clean ratio
        expected_clean = 1.0 - noise_ratio
        plt.axhline(y=expected_clean, color='r', linestyle='--', label=f'Expected Clean Ratio ({expected_clean:.2f})')
        
        plt.title(f'Kept Ratio vs. Iteration (Noise Ratio: {noise_ratio})')
        plt.xlabel('Iteration')
        plt.ylabel('Kept Ratio')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)
        
        ax.axhline(y=expected_clean, color='r', linestyle='--', label=f'Expected Clean Ratio ({expected_clean:.2f})')

        # Annotate final point for each line
        for line in ax.lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            if len(x_data) > 0:
                ax.annotate(f'{y_data[-1]:.3f}',
                            xy=(x_data[-1], y_data[-1]),
                            xytext=(5, 0),
                            textcoords='offset points',
                            fontsize=9, color=line.get_color())

        ax.set_title(f'Kept Ratio vs. Iteration (Noise Ratio: {noise_ratio})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Kept Ratio')
        ax.set_ylim(0, 1.05)
        ax.legend(title='Filter Mode')
        ax.grid(True)

        plot_path = report_dir / f"kept_ratio_vs_iteration_noise_{noise_ratio}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved kept ratio plot to {plot_path}")


def plot_test_acc_over_time(df: pd.DataFrame, report_dir: Path):
    """(4) Plots test accuracy over iterations."""
    logging.info("Plotting Test Accuracy over Iterations...")

    # Find the best run (alpha, mode) for each noise ratio
    best_runs = df.loc[df.groupby('noise_ratio')['test_acc_orig'].idxmax()]
    
    for _, row in best_runs.iterrows():
        noise_ratio = row['noise_ratio']
        alpha = row['alpha']
        filter_mode = row['filter_mode']
        
        # Get all iterations for this best run
        run_data = df[(df['noise_ratio'] == noise_ratio) & 
                      (df['alpha'] == alpha) & 
                      (df['filter_mode'] == filter_mode)]
    best_alpha_per_mode = df.loc[df.groupby(['noise_ratio', 'filter_mode'])['test_acc_orig'].idxmax()]

        plt.figure(figsize=(12, 7))
        sns.lineplot(data=run_data, x='iteration', y='test_acc_orig', marker='o', 
                     label=f'alpha={alpha}, mode={filter_mode}')
        
        plt.title(f'Test Accuracy vs. Iteration (Noise Ratio: {noise_ratio})')
        plt.xlabel('Iteration')
        plt.ylabel('Test Accuracy (on original labels)')
        plt.grid(True)
        plt.legend()
        
    for noise_ratio, group in best_alpha_per_mode.groupby('noise_ratio'):
        plot_data = pd.DataFrame()
        for _, setting in group.iterrows():
            run_data = df[df['exp_path'] == setting['exp_path']].copy()
            run_data['legend_label'] = f"{setting['filter_mode']} (α={setting['alpha']})"
            plot_data = pd.concat([plot_data, run_data])

        if plot_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(data=plot_data, x='iteration', y='test_acc_orig', hue='legend_label', marker='o', ax=ax)

        # Annotate baseline and peak for each line
        for line in ax.lines:
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            if len(x_data) > 0:
                # Baseline (iter 0)
                ax.annotate(f'{y_data[0]:.3f}', xy=(x_data[0], y_data[0]), xytext=(-15, -15), textcoords='offset points', fontsize=8, color=line.get_color())
                # Peak
                peak_idx = np.argmax(y_data)
                ax.annotate(f'{y_data[peak_idx]:.3f}', xy=(x_data[peak_idx], y_data[peak_idx]), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9, color=line.get_color(), arrowprops=dict(arrowstyle="->", color=line.get_color()))

        ax.set_title(f'Test Accuracy vs. Iteration (Noise Ratio: {noise_ratio})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Test Accuracy (on original labels)')
        ax.grid(True)
        ax.legend(title='Method (Best Alpha)')

        plot_path = report_dir / f"test_acc_vs_iteration_noise_{noise_ratio}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved test accuracy plot to {plot_path}")


def plot_val_acc_correlation(df: pd.DataFrame, report_dir: Path):
    """(5) Plots relationship between val_acc_noisy and val_acc_orig."""
    logging.info("Plotting Validation Accuracy Correlation...")
    
    noise_ratios_to_plot = df['noise_ratio'].unique()
    
    for noise_ratio in noise_ratios_to_plot:
        # Find the best run for this noise ratio
        best_run_idx = df[df['noise_ratio'] == noise_ratio]['test_acc_orig'].idxmax()
        best_run = df.loc[best_run_idx]
        
        run_data = df[df['exp_path'] == best_run['exp_path']]
        run_data = df[df['exp_path'] == best_run['exp_path']].copy()

        plt.figure(figsize=(12, 7))
        plt.plot(run_data['iteration'], run_data['val_acc_orig'], marker='o', linestyle='-', label='val_acc_orig (on clean labels)')
        plt.plot(run_data['iteration'], run_data['val_acc_noisy'], marker='x', linestyle='--', label='val_acc_noisy (on noisy labels)')
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(run_data['iteration'], run_data['val_acc_orig'], marker='o', linestyle='-', label='val_acc_orig (on clean labels)')
        ax.plot(run_data['iteration'], run_data['val_acc_noisy'], marker='x', linestyle='--', label='val_acc_noisy (on noisy labels)')
        
        plt.title(f'Validation Accuracy Correlation (Noise: {noise_ratio}, Alpha: {best_run["alpha"]}, Mode: {best_run["filter_mode"]})')
        plt.xlabel('Iteration')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        ax.set_title(f'Validation Accuracy Correlation (Noise: {noise_ratio})', fontsize=14)
        fig.suptitle(f'Best Run: alpha={best_run["alpha"]}, mode={best_run["filter_mode"]}', fontsize=10, y=0.92)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Validation Accuracy')
        ax.legend()
        ax.grid(True)

        # Add explanatory text
        explanation = (
            "Interpretation:\n"
            "- If lines move together: `val_acc_noisy` is a good proxy for real performance.\n"
            "- If lines diverge (especially `val_acc_noisy` ↑, `val_acc_orig` ↓):\n"
            "  The model is overfitting to noise in the validation set."
        )
        plt.text(0.05, 0.05, explanation, transform=ax.transAxes, fontsize=9,
                 verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plot_path = report_dir / f"val_acc_correlation_noise_{noise_ratio}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved validation accuracy correlation plot to {plot_path}")


def plot_filter_quality(df: pd.DataFrame, report_dir: Path):
def plot_filter_quality(detailed_stats_df: pd.DataFrame, report_dir: Path):
    """(6) Calculates and plots filter quality (Precision, Recall, F1)."""
    logging.info("Plotting Filter Quality (Precision/Recall/F1)...")

    # Find the best run for each noise ratio
    best_runs = df.loc[df.groupby('noise_ratio')['test_acc_orig'].idxmax()]
    for noise_ratio, group in detailed_stats_df.groupby('noise_ratio'):
        fig, ax = plt.subplots(figsize=(12, 7))

    for _, row in best_runs.iterrows():
        exp_path = Path(row['exp_path'])
        noise_ratio = row['noise_ratio']
        
        quality_data = []
        
        iter_dirs = sorted(exp_path.glob("iteration_*"), key=lambda p: int(p.name.split('_')[-1]))
        plot_df = group.melt(id_vars=['iteration', 'filter_mode'], value_vars=['precision', 'recall', 'f1_score'],
                             var_name='metric', value_name='score')

        for iter_dir in iter_dirs:
            iter_num = int(iter_dir.name.split('_')[-1])
            selection_file = iter_dir / "selection" / f"selection_iter_{iter_num}.csv"
            
            if not selection_file.exists():
                continue
        sns.lineplot(data=plot_df, x='iteration', y='score', hue='filter_mode', style='metric', marker='o', ax=ax)

            sel_df = pd.read_csv(selection_file)
            
            # noise_flag == 0 -> clean, noise_flag == 1 -> noisy
            # filter_flag == 'kept' -> positive prediction
            
            tp = len(sel_df[(sel_df['noise_flag'] == 0) & (sel_df['filter_flag'] == 'kept')])
            fp = len(sel_df[(sel_df['noise_flag'] == 1) & (sel_df['filter_flag'] == 'kept')])
            fn = len(sel_df[(sel_df['noise_flag'] == 0) & (sel_df['filter_flag'] == 'removed')])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            quality_data.append({'iteration': iter_num, 'precision': precision, 'recall': recall, 'f1_score': f1})
        ax.set_title(f'Filter Quality vs. Iteration (Noise: {noise_ratio})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.legend(title='Method & Metric')
        ax.grid(True)

        if not quality_data:
            continue
            
        quality_df = pd.DataFrame(quality_data)
        
        plt.figure(figsize=(12, 7))
        plt.plot(quality_df['iteration'], quality_df['precision'], marker='o', label='Precision')
        plt.plot(quality_df['iteration'], quality_df['recall'], marker='s', label='Recall')
        plt.plot(quality_df['iteration'], quality_df['f1_score'], marker='^', label='F1-Score', linewidth=2.5)
        
        plt.title(f'Filter Quality vs. Iteration (Noise: {noise_ratio})')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)
        
        plot_path = report_dir / f"filter_quality_noise_{noise_ratio}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved filter quality plot to {plot_path}")


def plot_class_balance(df: pd.DataFrame, report_dir: Path):
    """(7) Plots class distribution before and after filtering."""
    logging.info("Plotting Class Balance...")

    # Find the best run for each noise ratio
    best_runs = df.loc[df.groupby('noise_ratio')['test_acc_orig'].idxmax()]

    for _, row in best_runs.iterrows():
        exp_path = Path(row['exp_path'])
        noise_ratio = row['noise_ratio']
        best_iter = int(row['iteration'])

        # Path to the original training file (assuming it's in a standard location)
        # This might need adjustment based on your project structure
        original_train_csv = Path(str(exp_path).split('noise_')[0]) / f"csvs/noise_{noise_ratio}/train.csv"
        
        kept_csv = exp_path / f"iteration_{best_iter}" / f"train_kept_{best_iter}.csv"

        if not original_train_csv.exists() or not kept_csv.exists():
            logging.warning(f"Skipping class balance for noise {noise_ratio}: files not found.")
            continue

        df_orig = pd.read_csv(original_train_csv)
        df_kept = pd.read_csv(kept_csv)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        
        sns.countplot(ax=axes[0], data=df_orig, y='class_name', order=df_orig['class_name'].value_counts().index)
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        class_order = df_orig['class_name'].value_counts().index

        # Original
        sns.countplot(ax=axes[0], data=df_orig, y='class_name', order=class_order)
        axes[0].set_title(f'Original Distribution (N={len(df_orig)})')
        axes[0].set_xlabel('Sample Count')
        axes[0].set_ylabel('Class')
        for p in axes[0].patches:
            width = p.get_width()
            axes[0].text(width + 50, p.get_y() + p.get_height()/2., f'{width}', va='center')

        sns.countplot(ax=axes[1], data=df_kept, y='class_name', order=df_orig['class_name'].value_counts().index)
        # Kept
        sns.countplot(ax=axes[1], data=df_kept, y='class_name', order=class_order)
        axes[1].set_title(f'After Filtering at Best Iteration {best_iter} (N={len(df_kept)})')
        axes[1].set_xlabel('Sample Count')
        axes[1].set_ylabel('')

        for p in axes[1].patches:
            width = p.get_width()
            axes[1].text(width + 50, p.get_y() + p.get_height()/2., f'{width}', va='center')
            
        fig.suptitle(f'Class Balance Comparison (Noise: {noise_ratio})', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_path = report_dir / f"class_balance_noise_{noise_ratio}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved class balance plot to {plot_path}")


def main():
    """Main function to generate all reports and plots."""
    # --- CONFIGURATION ---
    # Directory where your experiment outputs are stored
    # (e.g., the 'outputs' folder you copied results into)
    source_dir = Path("./outputs/cifar10_iter_ema_voting_sweep")
    
    # Directory to save the generated reports and plots
    report_dir = Path("./reports")
    
    # --- EXECUTION ---
    report_dir.mkdir(exist_ok=True)
    
    # 1. Aggregate all summary data
    full_df = aggregate_summary_results(source_dir)
    
    if full_df is None:
        logging.error("Failed to aggregate summary data. Exiting.")
        return

    # Save the aggregated data for inspection
    full_df.to_csv(report_dir / "aggregated_summary_results.csv", index=False)
    logging.info(f"Saved aggregated data to {report_dir / 'aggregated_summary_results.csv'}")

    # 2. Generate Main Table
    main_table = create_main_table(full_df)
    if main_table is not None:
        main_table.to_csv(report_dir / "main_results_table.csv", index=False)
        logging.info(f"Saved main results table to {report_dir / 'main_results_table.csv'}")
        print("\n--- Main Results Table ---")
        print(main_table.to_markdown(index=False))
        print("--------------------------\n")

    # 3. Generate Filter Statistics Table
    # 3. Generate Filter Statistics Tables
    filter_stats_table = create_filter_statistics_table(full_df)
    if filter_stats_table is not None:
        filter_stats_table.to_csv(report_dir / "filter_statistics_table.csv", index=False)
        logging.info(f"Saved filter statistics table to {report_dir / 'filter_statistics_table.csv'}")
        print("\n--- Filter Performance Statistics at Best Iteration ---")
        print(filter_stats_table.to_markdown(index=False))
        print("-------------------------------------------------------\n")
        print("------------------------------------------------------\n")

    detailed_filter_stats_table = create_detailed_filter_statistics_table(full_df)
    if detailed_filter_stats_table is not None:
        detailed_filter_stats_table.to_csv(report_dir / "detailed_filter_statistics.csv", index=False)
        logging.info(f"Saved DETAILED filter statistics table to {report_dir / 'detailed_filter_statistics.csv'}")
        print("\n--- Detailed Filter Performance (Sample) ---")
        print(detailed_filter_stats_table.head().to_markdown(index=False))
        print("------------------------------------------\n")

    # 4. Generate Plots
    plot_ablation_alpha(full_df, report_dir)
    plot_kept_ratio_over_time(full_df, report_dir)
    plot_test_acc_over_time(full_df, report_dir)
    plot_val_acc_correlation(full_df, report_dir)
    
    # These plots need to find the best run from the summary first

    if detailed_filter_stats_table is not None:
        plot_filter_quality(detailed_filter_stats_table, report_dir)

    if not full_df.empty:
        best_runs_summary = full_df.loc[full_df.groupby('noise_ratio')['test_acc_orig'].idxmax()]
        plot_filter_quality(best_runs_summary, report_dir)
        plot_class_balance(best_runs_summary, report_dir)
    
        plot_class_balance(full_df, report_dir)

    logging.info(f"All reports and plots have been saved to: {report_dir.resolve()}")


if __name__ == "__main__":
    main()
