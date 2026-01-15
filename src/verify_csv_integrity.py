"""
Script để verify tính đúng đắn của CSV files đã được tạo ra.
Chạy script này sau khi prepare data để đảm bảo mọi thứ đều chính xác.

Usage:
    python verify_csv_integrity.py --csv_path data/csvs/noise_0.2/train.csv
    python verify_csv_integrity.py --csv_dir data/csvs/noise_0.2/
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
from collections import defaultdict

# CIFAR-10 class names theo thứ tự chính xác
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def verify_csv_structure(df: pd.DataFrame, csv_path: str) -> bool:
    """Verify rằng CSV có đúng cấu trúc columns."""
    required_columns = ['index', 'image_path', 'label_noisy', 'label_orig', 
                       'class_name', 'split', 'noise_flag']
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"❌ FAILED: Missing columns in {csv_path}: {missing_columns}")
        return False
    
    print(f"✓ CSV structure correct: all required columns present")
    return True


def verify_index_column(df: pd.DataFrame) -> bool:
    """Verify tính duy nhất và hợp lệ của index column."""
    print("\n--- Verifying Index Column ---")
    
    # Check uniqueness
    if df['index'].nunique() != len(df):
        duplicates = df[df.duplicated('index', keep=False)]
        print(f"❌ FAILED: Found duplicate indices:")
        print(duplicates[['index', 'label_orig', 'split']])
        return False
    print(f"✓ All indices are unique ({len(df)} rows)")
    
    # Check range
    min_idx = df['index'].min()
    max_idx = df['index'].max()
    print(f"✓ Index range: {min_idx} to {max_idx}")
    
    # Check type
    if not df['index'].dtype in [np.int32, np.int64, int]:
        print(f"❌ FAILED: Index column should be integer, got {df['index'].dtype}")
        return False
    print(f"✓ Index column has correct type (integer)")
    
    return True


def verify_label_columns(df: pd.DataFrame) -> bool:
    """Verify tính hợp lệ của label_orig và label_noisy."""
    print("\n--- Verifying Label Columns ---")
    
    # Check label_orig range
    if df['label_orig'].min() < 0 or df['label_orig'].max() > 9:
        print(f"❌ FAILED: label_orig out of range [0, 9]: min={df['label_orig'].min()}, max={df['label_orig'].max()}")
        return False
    print(f"✓ label_orig in valid range [0, 9]")
    
    # Check label_noisy range
    if df['label_noisy'].min() < 0 or df['label_noisy'].max() > 9:
        print(f"❌ FAILED: label_noisy out of range [0, 9]: min={df['label_noisy'].min()}, max={df['label_noisy'].max()}")
        return False
    print(f"✓ label_noisy in valid range [0, 9]")
    
    # Check consistency between labels and noise_flag
    clean_samples = df[df['noise_flag'] == 0]
    if len(clean_samples) > 0:
        mismatched = clean_samples[clean_samples['label_noisy'] != clean_samples['label_orig']]
        if len(mismatched) > 0:
            print(f"❌ FAILED: Found {len(mismatched)} clean samples (noise_flag=0) where label_noisy != label_orig")
            print(mismatched[['index', 'label_orig', 'label_noisy', 'noise_flag']].head())
            return False
        print(f"✓ All clean samples (noise_flag=0) have label_noisy == label_orig ({len(clean_samples)} samples)")
    
    noisy_samples = df[df['noise_flag'] == 1]
    if len(noisy_samples) > 0:
        same_labels = noisy_samples[noisy_samples['label_noisy'] == noisy_samples['label_orig']]
        if len(same_labels) > 0:
            print(f"❌ FAILED: Found {len(same_labels)} noisy samples (noise_flag=1) where label_noisy == label_orig")
            print("Note: Noisy samples should have different labels!")
            print(same_labels[['index', 'label_orig', 'label_noisy', 'noise_flag']].head())
            return False
        print(f"✓ All noisy samples (noise_flag=1) have label_noisy != label_orig ({len(noisy_samples)} samples)")
    
    return True


def verify_class_name_column(df: pd.DataFrame) -> bool:
    """Verify rằng class_name mapping đúng với label_orig."""
    print("\n--- Verifying Class Name Column ---")
    
    # Check that all class names are valid
    valid_classes = set(CIFAR10_CLASSES)
    invalid_classes = set(df['class_name'].unique()) - valid_classes
    if invalid_classes:
        print(f"❌ FAILED: Found invalid class names: {invalid_classes}")
        return False
    print(f"✓ All class names are valid CIFAR-10 classes")
    
    # Check mapping between label_orig and class_name
    for idx, row in df.iterrows():
        expected_class = CIFAR10_CLASSES[int(row['label_orig'])]
        if row['class_name'] != expected_class:
            print(f"❌ FAILED: Row {idx} has label_orig={row['label_orig']} but class_name='{row['class_name']}' (expected '{expected_class}')")
            return False
    
    print(f"✓ All class_name values correctly match label_orig")
    
    # Show class distribution
    class_counts = df['class_name'].value_counts().sort_index()
    print(f"\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} samples")
    
    return True


def verify_image_paths(df: pd.DataFrame, check_existence: bool = True) -> bool:
    """Verify tính hợp lệ của image paths."""
    print("\n--- Verifying Image Paths ---")
    
    # Check that paths are non-empty strings
    if df['image_path'].isna().any():
        print(f"❌ FAILED: Found {df['image_path'].isna().sum()} null image paths")
        return False
    print(f"✓ All image paths are non-null")
    
    # Check that paths follow expected pattern: .../images/split/class_name/filename.png
    for idx, row in df.sample(min(100, len(df))).iterrows():
        path_str = str(row['image_path'])
        
        # Extract components
        path_parts = path_str.split(os.sep)
        
        # Find 'images' in path
        if 'images' not in path_parts:
            print(f"❌ FAILED: Row {idx} path doesn't contain 'images' directory: {path_str}")
            return False
        
        images_idx = path_parts.index('images')
        
        # Check structure: .../images/split/class_name/filename.png
        if images_idx + 3 >= len(path_parts):
            print(f"❌ FAILED: Row {idx} path structure incorrect: {path_str}")
            return False
        
        split_from_path = path_parts[images_idx + 1]
        class_from_path = path_parts[images_idx + 2]
        filename = path_parts[images_idx + 3]
        
        # Verify split matches
        if split_from_path != row['split']:
            print(f"❌ FAILED: Row {idx} path split '{split_from_path}' doesn't match CSV split '{row['split']}'")
            return False
        
        # Verify class_name matches (based on label_orig)
        expected_class = CIFAR10_CLASSES[int(row['label_orig'])]
        if class_from_path != expected_class:
            print(f"❌ FAILED: Row {idx} path class '{class_from_path}' doesn't match expected '{expected_class}' (label_orig={row['label_orig']})")
            return False
        
        # Verify filename format
        if not filename.endswith('.png'):
            print(f"❌ FAILED: Row {idx} filename doesn't end with .png: {filename}")
            return False
        
        # Verify filename contains index
        expected_prefix = f"{row['split']}_{int(row['index']):06d}"
        if not filename.startswith(expected_prefix):
            print(f"❌ FAILED: Row {idx} filename '{filename}' doesn't match expected pattern '{expected_prefix}.png'")
            return False
    
    print(f"✓ Image path structure is correct (checked sample of {min(100, len(df))} rows)")
    
    # Check file existence if requested
    if check_existence:
        missing_files = []
        for idx, row in df.sample(min(50, len(df))).iterrows():
            if not os.path.exists(row['image_path']):
                missing_files.append((idx, row['image_path']))
        
        if missing_files:
            print(f"❌ FAILED: Found {len(missing_files)} missing image files (sample):")
            for idx, path in missing_files[:5]:
                print(f"  Row {idx}: {path}")
            return False
        print(f"✓ All sampled image files exist on disk (checked {min(50, len(df))} files)")
    
    return True


def verify_noise_flag(df: pd.DataFrame) -> bool:
    """Verify tính hợp lệ của noise_flag column."""
    print("\n--- Verifying Noise Flag Column ---")
    
    # Check values are only 0 or 1
    if not df['noise_flag'].isin([0, 1]).all():
        invalid = df[~df['noise_flag'].isin([0, 1])]
        print(f"❌ FAILED: Found {len(invalid)} rows with invalid noise_flag values:")
        print(invalid[['index', 'noise_flag']].head())
        return False
    print(f"✓ All noise_flag values are 0 or 1")
    
    # Calculate noise statistics
    total = len(df)
    noisy = (df['noise_flag'] == 1).sum()
    clean = (df['noise_flag'] == 0).sum()
    noise_ratio = noisy / total if total > 0 else 0
    
    print(f"\nNoise statistics:")
    print(f"  Total samples: {total}")
    print(f"  Clean samples: {clean} ({clean/total*100:.2f}%)")
    print(f"  Noisy samples: {noisy} ({noisy/total*100:.2f}%)")
    print(f"  Noise ratio: {noise_ratio:.4f}")
    
    return True


def verify_split_column(df: pd.DataFrame) -> bool:
    """Verify tính hợp lệ của split column."""
    print("\n--- Verifying Split Column ---")
    
    # Check values are only train/val/test
    valid_splits = {'train', 'val', 'test'}
    invalid = set(df['split'].unique()) - valid_splits
    if invalid:
        print(f"❌ FAILED: Found invalid split values: {invalid}")
        return False
    print(f"✓ All split values are valid (train/val/test)")
    
    # Show split distribution
    split_counts = df['split'].value_counts()
    print(f"\nSplit distribution:")
    for split, count in split_counts.items():
        print(f"  {split}: {count} samples")
    
    return True


def verify_test_set_cleanliness(df: pd.DataFrame) -> bool:
    """Verify rằng test set không có noise."""
    print("\n--- Verifying Test Set Cleanliness ---")
    
    test_df = df[df['split'] == 'test']
    if len(test_df) == 0:
        print("⚠ WARNING: No test samples found in this CSV")
        return True
    
    noisy_test = test_df[test_df['noise_flag'] == 1]
    if len(noisy_test) > 0:
        print(f"❌ FAILED: Found {len(noisy_test)} noisy samples in test set!")
        print("Test set should always be clean for fair evaluation.")
        print(noisy_test[['index', 'label_orig', 'label_noisy', 'noise_flag']].head())
        return False
    
    print(f"✓ Test set is completely clean (0 noisy samples out of {len(test_df)})")
    return True


def verify_csv_file(csv_path: str, check_file_existence: bool = True) -> bool:
    """
    Chạy tất cả verification checks trên một CSV file.
    
    Args:
        csv_path: Path to CSV file
        check_file_existence: Whether to check if image files actually exist
        
    Returns:
        bool: True if all checks pass, False otherwise
    """
    print("="*80)
    print(f"Verifying CSV file: {csv_path}")
    print("="*80)
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Successfully loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"❌ FAILED: Could not load CSV file: {e}")
        return False
    
    # Run all checks
    checks = [
        ("CSV Structure", lambda: verify_csv_structure(df, csv_path)),
        ("Index Column", lambda: verify_index_column(df)),
        ("Label Columns", lambda: verify_label_columns(df)),
        ("Class Name Column", lambda: verify_class_name_column(df)),
        ("Image Paths", lambda: verify_image_paths(df, check_file_existence)),
        ("Noise Flag", lambda: verify_noise_flag(df)),
        ("Split Column", lambda: verify_split_column(df)),
        ("Test Set Cleanliness", lambda: verify_test_set_cleanliness(df)),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            print(f"\n❌ FAILED: {check_name} check raised exception: {e}")
            import traceback
            traceback.print_exc()
            results[check_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All verification checks PASSED!")
        return True
    else:
        print(f"⚠ {total - passed} checks FAILED. Please review the output above.")
        return False


def verify_csv_directory(csv_dir: str, check_file_existence: bool = True) -> bool:
    """
    Verify all CSV files in a directory (train.csv, val.csv, test.csv).
    
    Args:
        csv_dir: Directory containing CSV files
        check_file_existence: Whether to check if image files actually exist
        
    Returns:
        bool: True if all CSVs pass verification, False otherwise
    """
    csv_files = ['train.csv', 'val.csv', 'test.csv']
    results = {}
    
    for csv_file in csv_files:
        csv_path = os.path.join(csv_dir, csv_file)
        if not os.path.exists(csv_path):
            print(f"⚠ WARNING: {csv_path} not found, skipping...")
            continue
        
        result = verify_csv_file(csv_path, check_file_existence)
        results[csv_file] = result
        print("\n")
    
    # Overall summary
    print("="*80)
    print("OVERALL DIRECTORY VERIFICATION SUMMARY")
    print("="*80)
    
    for csv_file, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {csv_file}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All CSV files in directory PASSED verification!")
    else:
        print("\n⚠ Some CSV files FAILED verification. Please review the output above.")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Verify integrity of CIFAR-10 CSV files")
    parser.add_argument('--csv_path', type=str, help='Path to a single CSV file to verify')
    parser.add_argument('--csv_dir', type=str, help='Directory containing CSV files to verify')
    parser.add_argument('--no_check_files', action='store_true', 
                       help='Skip checking if image files actually exist (faster)')
    
    args = parser.parse_args()
    
    if not args.csv_path and not args.csv_dir:
        print("Error: Must provide either --csv_path or --csv_dir")
        parser.print_help()
        return
    
    check_existence = not args.no_check_files
    
    if args.csv_path:
        success = verify_csv_file(args.csv_path, check_existence)
    else:
        success = verify_csv_directory(args.csv_dir, check_existence)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())