# scripts/reporting/clean_checkpoints.py
import os
from pathlib import Path
import logging
from typing import List

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clean_checkpoint_folders(root_dirs: List[Path]):
    """
    Iterates through experiment folders and deletes intermediate checkpoints,
    keeping only '_best.pth' and '_last.pth' files for each iteration.
    """
    files_to_delete = []
    
    for root_dir in root_dirs:
        if not root_dir.is_dir():
            logging.warning(f"Directory not found, skipping: {root_dir}")
            continue
            
        logging.info(f"Scanning for checkpoints in: {root_dir}")
        
        # Find all checkpoint files
        all_checkpoints = list(root_dir.rglob("*.pth"))
        
        # Group checkpoints by their iteration directory
        checkpoints_by_iter = {}
        for pth_file in all_checkpoints:
            iter_dir = pth_file.parent
            if iter_dir not in checkpoints_by_iter:
                checkpoints_by_iter[iter_dir] = []
            checkpoints_by_iter[iter_dir].append(pth_file)
            
        # Decide which files to keep or delete
        for iter_dir, pth_files in checkpoints_by_iter.items():
            files_to_keep = set()
            for pth_file in pth_files:
                if pth_file.name.endswith("_best.pth") or pth_file.name.endswith("_last.pth"):
                    files_to_keep.add(pth_file)
            
            for pth_file in pth_files:
                if pth_file not in files_to_keep:
                    files_to_delete.append(pth_file)

    if not files_to_delete:
        logging.info("No intermediate checkpoint files found to delete.")
        return

    logging.info(f"Found {len(files_to_delete)} intermediate checkpoint files to delete.")
    
    # Uncomment the following lines to actually delete the files
    # for f in files_to_delete:
    #     try:
    #         # os.remove(f)
    #         # logging.info(f"Deleted: {f}")
    #         pass # Safety: print first
    #     except OSError as e:
    #         logging.error(f"Error deleting file {f}: {e}")
    
    print("\n--- Files identified for deletion (run script again with deletion enabled) ---")
    for f in files_to_delete:
        print(f)
    print("--------------------------------------------------------------------------")
    logging.warning("Deletion is currently disabled for safety. Please review the file list and uncomment the 'os.remove(f)' line in the script to proceed.")


def main():
    """Main function to clean checkpoints."""
    # --- CONFIGURATION ---
    # List of root directories containing your experiment outputs
    ls_folder_output = [
        Path("./outputs/cifar10_ema_noise_val"),
        # Add other output folders if you have them
        # Path("./outputs/another_experiment"),
    ]
    
    # --- EXECUTION ---
    clean_checkpoint_folders(ls_folder_output)


if __name__ == "__main__":
    main()
