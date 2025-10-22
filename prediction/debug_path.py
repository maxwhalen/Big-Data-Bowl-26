#!/usr/bin/env python3
"""
Debug script to test the DATA_DIR path and file existence
"""

from pathlib import Path
import pandas as pd

# Test the current DATA_DIR path
DATA_DIR = Path("/Users/maxwhalen/Documents/GitHub/Big-Data-Bowl-26/prediction/../kaggle/input/nfl-big-data-bowl-2026-prediction/")

print(f"DATA_DIR: {DATA_DIR}")
print(f"DATA_DIR exists: {DATA_DIR.exists()}")
print(f"DATA_DIR is absolute: {DATA_DIR.is_absolute()}")

# Test train directory
train_dir = DATA_DIR / "train"
print(f"Train dir: {train_dir}")
print(f"Train dir exists: {train_dir.exists()}")

# Test specific files
train_input_files = [DATA_DIR / f"train/input_2023_w{w:02d}.csv" for w in range(1, 19)]
print(f"\nChecking train input files:")
for i, f in enumerate(train_input_files[:5]):  # Check first 5
    print(f"  {f}: exists={f.exists()}")

# Count existing files
existing_files = [f for f in train_input_files if f.exists()]
print(f"\nTotal existing input files: {len(existing_files)}")

# Test reading a file
if existing_files:
    print(f"\nTesting to read first file: {existing_files[0]}")
    try:
        df = pd.read_csv(existing_files[0])
        print(f"Successfully read file, shape: {df.shape}")
    except Exception as e:
        print(f"Error reading file: {e}")

# Test the exact same logic as in the notebook
print(f"\nTesting notebook logic:")
train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
print(f"Successfully concatenated, shape: {train_input.shape}")
