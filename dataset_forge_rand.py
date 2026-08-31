# dataset_forge.py
# The Sanitized Omnikon Dataset Architect (Strictly Blind & Randomized Splits)

import os
import shutil
import random
from pathlib import Path

# =====================================================================
# GLOBAL CONFIGURATION: THE COMMAND CENTER
# =====================================================================
# Temporary extraction zones
TEMP_REAL_DIR = "/home/CL502-19/Desktop/dmd/DATASET/temp_real"
TEMP_FAKE_GAN_DIR = "/home/CL502-19/Desktop/dmd/DATASET/temp_fake/images/images"
TEMP_FAKE_DIFF_DIR = "/home/CL502-19/Desktop/dmd/DATASET/temp_fake_diff/temp_fake_new"

# Final Migration Endpoints for Real Data
TRAIN_REAL_DIR = "/home/CL502-19/Desktop/dmd/DATASET/REAL/train"
CV_REAL_DIR = "/home/CL502-19/Desktop/dmd/DATASET/REAL/cv"
TEST_REAL_DIR = "/home/CL502-19/Desktop/dmd/DATASET/REAL/test"

# Final Migration Endpoints for Fake Data
TRAIN_FAKE_DIR = "/home/CL502-19/Desktop/dmd/DATASET/FAKE/train"
CV_FAKE_DIR = "/home/CL502-19/Desktop/dmd/DATASET/FAKE/cv"
TEST_FAKE_DIR = "/home/CL502-19/Desktop/dmd/DATASET/FAKE/test"

# Target Quotas
REAL_QUOTA = 48000
FAKE_GAN_QUOTA = 22000
FAKE_DIFF_QUOTA = 22000

# Data split ratios (60:20:20)
TRAIN_RATIO = 0.6
CV_RATIO = 0.2
TEST_RATIO = 0.2
# =====================================================================

def harvest_and_sample(source_dir, quota, seed=42):
    """Gathers image paths and extracts a randomized, unbiased sample."""
    print(f"\n[Architect] Scanning repository: {source_dir}...")
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    image_paths = [p for p in Path(source_dir).rglob("*") if p.suffix.lower() in valid_exts]
    
    total_found = len(image_paths)
    print(f"[Architect] Found {total_found} images in source.")
    if total_found < quota:
        raise ValueError(f"Insufficient images! Found {total_found}, but quota requires {quota}.")
    
    random.seed(seed)
    selected_paths = random.sample(image_paths, quota)
    print(f"[Architect] Randomly selected {len(selected_paths)} paths meeting the quota.")
    return selected_paths

def split_and_migrate(image_paths, train_dir, cv_dir, test_dir, seed=42):
    """Executes a blind 60/20/20 split and moves files into final destinations."""
    random.seed(seed)
    random.shuffle(image_paths)
    
    total = len(image_paths)
    train_end = int(total * TRAIN_RATIO)
    cv_end = train_end + int(total * CV_RATIO)
    
    splits = {
        train_dir: image_paths[:train_end],
        cv_dir: image_paths[train_end:cv_end],
        test_dir: image_paths[cv_end:]
    }
    
    for dest_dir, paths in splits.items():
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        print(f"  -> Migrating {len(paths)} files to {dest_path}...")
        for p in paths:
            shutil.copy2(str(p), dest_path / p.name)

if __name__ == "__main__":
    print("=== SANITIZED OMNIKON DATASET FORGE INITIATED ===")
    
    # 1. Harvest & Migrate Real Data
    print("\n[Phase 1] Processing Real Distribution...")
    real_paths = harvest_and_sample(TEMP_REAL_DIR, REAL_QUOTA, seed=42)
    split_and_migrate(real_paths, TRAIN_REAL_DIR, CV_REAL_DIR, TEST_REAL_DIR, seed=42)
    
    # 2. Harvest Synthetic Subsets
    print("\n[Phase 2] Harvesting Synthetic Pools...")
    gan_paths = harvest_and_sample(TEMP_FAKE_GAN_DIR, FAKE_GAN_QUOTA, seed=42)
    diff_paths = harvest_and_sample(TEMP_FAKE_DIFF_DIR, FAKE_DIFF_QUOTA, seed=42)
    
    # 3. Merge & Blind-Split Unified Fake Distribution (44k total: 22k GAN + 22k Diffusion)
    print("\n[Phase 3] Merging and Partitioning Synthetic Distribution...")
    unified_fakes = gan_paths + diff_paths
    split_and_migrate(unified_fakes, TRAIN_FAKE_DIR, CV_FAKE_DIR, TEST_FAKE_DIR, seed=42)
    
    print("\n=== FORGE COMPLETE. Sanitized splits are sealed without structural leakage. ===")