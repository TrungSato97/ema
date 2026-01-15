#!/usr/bin/env python3
"""
Create project structure for iterative EMA training pipeline.
Usage: python scripts/create_project_structure.py
"""
import os
from pathlib import Path
import textwrap

ROOT = Path.cwd()
PROJECT = ROOT
SRC = PROJECT / "src"
NOTEBOOKS = PROJECT / "notebooks"
OUTPUTS = PROJECT / "outputs"
SCRIPTS = PROJECT / "scripts"
TESTS = PROJECT / "tests"

dirs = [
    SRC, NOTEBOOKS, OUTPUTS, SCRIPTS, TESTS,
    SRC / "preds_npz",
    OUTPUTS / "example_exp",
    OUTPUTS / "example_exp" / "iteration_0",
    OUTPUTS / "example_exp" / "iteration_0" / "checkpoints",
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

skeleton_files = {
    SRC / "config.py": '"""\nConfiguration dataclasses for training pipeline\n"""\n\n# Implemented by user\n',
    SRC / "dataset_utils.py": '"""\nData utilities: download CIFAR-10, save images, CSVs, create dataloaders\n"""\n\n# Implemented by user\n',
    SRC / "model_utils.py": '"""\nModel utilities: construct resnet, optimizer, scheduler, checkpoint save/load\n"""\n\n# Implemented by user\n',
    SRC / "train_loop.py": '"""\nTraining loop for one iteration: train_epoch, validate_epoch, train_iteration\n"""\n\n# Implemented by user\n',
    SRC / "ema_utils.py": '"""\nEMA utils: update, save/load .npz\n"""\n\n# Implemented by user\n',
    SRC / "filter_utils.py": '"""\nFilter utilities based on EMA predictions\n"""\n\n# Implemented by user\n',
    SRC / "io_utils.py": '"""\nI/O helpers for saving outputs, making directories\n"""\n\n# Implemented by user\n',
    SRC / "eda.py": '"""\nEDA plotting utilities\n"""\n\n# Implemented by user\n',
    NOTEBOOKS / "train_iterative_ema.ipynb": '',  # will be written by create_notebook.py if used
    TESTS / "test_ema.py": '"""\nUnit tests for EMA\n"""\n\n# Implemented by user\n',
    TESTS / "test_filter.py": '"""\nUnit tests for filter logic\n"""\n\n# Implemented by user\n',
    SCRIPTS / "run_experiment.sh": '#!/bin/bash\n# Example run script\npython -m notebooks.train_iterative_ema\n',
    PROJECT / "README.md": textwrap.dedent("""\
    # Iterative EMA training project
    Structure created by scripts/create_project_structure.py
    """)
}

for path, content in skeleton_files.items():
    if not path.exists():
        path.write_text(content)
        print(f"Created {path}")

print("Project skeleton created. Edit files in src/ to implement functionality.")
