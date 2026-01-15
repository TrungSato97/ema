#!/usr/bin/env python3
"""
Create a Jupyter notebook 'notebooks/train_iterative_ema.ipynb' with
sections that explain and run the iterative EMA training pipeline.

Run: python scripts/create_notebook.py
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

intro = """# Iterative EMA Training Pipeline (CIFAR-10)

This notebook orchestrates the iterative training pipeline:
- Download CIFAR-10, split into train/val/test and inject label noise (configurable).
- Train ResNet-18 pre-trained on ImageNet for each iteration.
- After each iteration, compute predictions on **entire** training set,
  update EMA predictions (alpha configurable), and filter samples whose EMA argmax disagrees with `label_noisy`.
- Enforce per-class minimum kept ratio by promoting top-confident removed samples.
- Repeat for up to `max_iterations` or until early-stop across iterations.

**Sections**
1. Setup & imports
2. Load configuration
3. Prepare CIFAR dataset (images + CSV)
4. EDA (class distribution, sample images)
5. Dataloaders & transforms
6. Iterative training loop (demo run small config)
7. Visualizations and summary
8. How to resume
"""
cells.append(nbf.v4.new_markdown_cell(intro))

setup = """# 1) Setup & imports
from pathlib import Path
import logging
from src.config import TrainConfig
from src.dataset_utils import prepare_cifar_data, make_data_loaders
from src.train_loop import train_iteration
from src.ema_utils import update_ema, save_preds_npz, load_preds_npz
from src.filter_utils import filter_by_ema
from src.io_utils import make_dirs, save_dataframe_csv, save_npz
from src.eda import plot_class_distribution, plot_confusion_matrix, plot_filter_ratios_over_iterations, plot_confidence_histogram

logging.basicConfig(level=logging.INFO)
"""
cells.append(nbf.v4.new_code_cell(setup))

config_cell = """# 2) Config
config = TrainConfig()
# For a quick demo, override a few params (small epochs)
config.max_iterations = 2
config.max_epochs_per_iter = 3
config.batch_size = 256
config.exp_name = "demo_cifar_iter_ema"
config.make_exp_dir()
print(config)
"""
cells.append(nbf.v4.new_code_cell(config_cell))

prepare_data = """# 3) Prepare data (download, images, CSV)
paths = prepare_cifar_data(config, force_download=False)
print(paths)
"""
cells.append(nbf.v4.new_code_cell(prepare_data))

eda_cell = """# 4) Initial EDA (class distribution)
plot_class_distribution({'train_csv': paths['train_csv'], 'val_csv': paths['val_csv'], 'test_csv': paths['test_csv']},
                        save_path=f\"{config.exp_dir}/initial_class_distribution.png\")
print('Saved initial class distribution plot.')
"""
cells.append(nbf.v4.new_code_cell(eda_cell))

dataloaders_cell = """# 5) Build dataloaders
dls = make_data_loaders(paths['train_csv'], paths['val_csv'], paths['test_csv'], config)
train_loader = dls['train']
val_loader = dls['val']
test_loader = dls['test']
train_full_loader = dls['train_full']
"""
cells.append(nbf.v4.new_code_cell(dataloaders_cell))

iter_loop = """# 6) Iterative training driver (demo)
z_ema_prev = None
best_overall_val_acc = -1.0
iter_no_improve = 0
summary_rows = []

for i in range(config.max_iterations):
    print('Starting iteration', i)
    result = train_iteration(i, config, train_loader, val_loader, test_loader, train_full_loader, start_epoch=0)
    indices = result['z_hat_indices']
    z_hat = result['z_hat']
    if z_ema_prev is None:
        z_ema_prev = update_ema(None, z_hat, config.alpha)
    else:
        z_ema_prev = update_ema(z_ema_prev, z_hat, config.alpha)
    # Save preds npz
    npz_path = f\"{config.exp_dir}/iteration_{i}/preds_npz/preds_iter_{i}.npz\"
    save_npz(npz_path, indices=indices.astype('int32'), z_hat=z_hat.astype('float32'), z_ema=z_ema_prev.astype('float32'))
    # Apply filter
    import pandas as pd
    train_df = pd.read_csv(paths['train_csv'])
    updated_df, stats = filter_by_ema(indices, z_ema_prev, train_df, config.min_keep_ratio)
    updated_csv_path = f\"{config.exp_dir}/iteration_{i}/preds/preds_iter_{i}.csv\"
    updated_df.to_csv(updated_csv_path, index=False)
    print('Iteration', i, 'stats overall kept ratio:', stats['_overall']['kept_ratio_total'])
    summary_rows.append({
        'iteration': i,
        'kept_ratio': stats['_overall']['kept_ratio_total'],
        'val_acc': result['best_val_acc'],
        'test_acc': result['test_acc']
    })
    # Compare early-stop across iterations
    if result['best_val_acc'] > best_overall_val_acc:
        best_overall_val_acc = result['best_val_acc']
        iter_no_improve = 0
    else:
        iter_no_improve += 1
        if iter_no_improve > config.patience_iter:
            print('Stopping iterations due to no improvement across iterations.')
            break
"""
cells.append(nbf.v4.new_code_cell(iter_loop))

summary_cell = """# 7) Summary
import pandas as pd
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f\"{config.exp_dir}/summary_iterations.csv\", index=False)
print(summary_df)
"""
cells.append(nbf.v4.new_code_cell(summary_cell))

resume_note = """# 8) How to resume
"""
cells.append(nbf.v4.new_markdown_cell(resume_note))

nb['cells'] = cells
out_path = Path('notebooks') / 'train_iterative_ema.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Notebook written to', out_path)
