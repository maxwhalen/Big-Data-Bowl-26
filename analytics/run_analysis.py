#!/usr/bin/env python3
"""
Big Data Bowl Analysis - Standalone Script
Replicates test.ipynb notebook operations to diagnose memory issues
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
from tqdm.auto import tqdm
import psutil
import gc

def print_memory():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    vm = psutil.virtual_memory()
    
    print(f"\n{'='*60}")
    print(f"MEMORY: Process={mem_info.rss / 1024**3:.2f}GB | "
          f"System Used={vm.used / 1024**3:.2f}GB/{vm.total / 1024**3:.2f}GB ({vm.percent}%)")
    print(f"{'='*60}")

# Set up paths
data_dir = Path('/home/ubuntu/Big-Data-Bowl-26/data')
train_dir = data_dir / 'train'

print("="*80)
print("BIG DATA BOWL ANALYSIS - MEMORY DIAGNOSTIC")
print("="*80)
print_memory()

# ==============================================================================
# STEP 1: Load Input Files
# ==============================================================================
print("\n[1/7] Loading INPUT files...")
input_files = sorted(glob.glob(str(train_dir / 'input_2023_*.csv')))
input_dfs = []
for file in tqdm(input_files, desc="Loading input files"):
    week = Path(file).stem.split('_')[-1]
    df = pd.read_csv(file)
    df['week'] = week
    input_dfs.append(df)

input_data = pd.concat(input_dfs, ignore_index=True)
print(f"✓ Loaded {len(input_files)} input files: {len(input_data):,} rows")
print_memory()

# ==============================================================================
# STEP 2: Load Output Files
# ==============================================================================
print("\n[2/7] Loading OUTPUT files...")
output_files = sorted(glob.glob(str(train_dir / 'output_2023_*.csv')))
output_dfs = []
for file in tqdm(output_files, desc="Loading output files"):
    week = Path(file).stem.split('_')[-1]
    df = pd.read_csv(file)
    df['week'] = week
    output_dfs.append(df)

output_data = pd.concat(output_dfs, ignore_index=True)
print(f"✓ Loaded {len(output_files)} output files: {len(output_data):,} rows")
print_memory()

# ==============================================================================
# STEP 3: Create OUTPUT defender-offense dataframe (CRITICAL - HIGH MEMORY)
# ==============================================================================
print("\n[3/7] Creating OUTPUT defender-offense relationships...")
print("⚠️  This is a CRITICAL memory step - creating cartesian product")

# Get player info
player_info = input_data.groupby('nfl_id').agg({
    'player_name': 'first',
    'player_position': 'first',
    'player_role': 'first',
    'player_side': 'first',
    'o': 'first'
}).reset_index()

ball_info = input_data.groupby(['game_id', 'play_id']).agg({
    'ball_land_x': 'first',
    'ball_land_y': 'first'
}).reset_index()

# Merge output data with player info
output_with_info = output_data.merge(player_info, on='nfl_id', how='left')

# Separate into offense and defense
offense_output = output_with_info[output_with_info['player_side'] == 'Offense'].copy()
defense_output = output_with_info[output_with_info['player_side'] == 'Defense'].copy()

print(f"  Offense frames: {len(offense_output):,}")
print(f"  Defense frames: {len(defense_output):,}")
print_memory()

# Create cartesian product
print("  Creating defender-offense pairs (cartesian product)...")
defender_offense_pairs = defense_output.merge(
    offense_output,
    on=['game_id', 'play_id', 'frame_id', 'week'],
    suffixes=('_def', '_off')
)

print(f"  ✓ Total pairs: {len(defender_offense_pairs):,}")
print_memory()

# Add ball information
defender_offense_pairs = defender_offense_pairs.merge(ball_info, on=['game_id', 'play_id'], how='left')

# Calculate distances and orientations
print("  Calculating distances and orientations...")
dx_off = defender_offense_pairs['x_off'] - defender_offense_pairs['x_def']
dy_off = defender_offense_pairs['y_off'] - defender_offense_pairs['y_def']
defender_offense_pairs['distance_def_to_off'] = np.sqrt(dx_off**2 + dy_off**2)
defender_offense_pairs['angle_def_to_off'] = np.degrees(np.arctan2(dy_off, dx_off)) % 360

angle_diff = (defender_offense_pairs['angle_def_to_off'] - defender_offense_pairs['o_def'] + 180) % 360 - 180
defender_offense_pairs['orientation_diff_def_to_off'] = np.abs(angle_diff)

dx_ball = defender_offense_pairs['ball_land_x'] - defender_offense_pairs['x_def']
dy_ball = defender_offense_pairs['ball_land_y'] - defender_offense_pairs['y_def']
defender_offense_pairs['distance_def_to_ball'] = np.sqrt(dx_ball**2 + dy_ball**2)
defender_offense_pairs['angle_def_to_ball'] = np.degrees(np.arctan2(dy_ball, dx_ball)) % 360

angle_diff_ball = (defender_offense_pairs['angle_def_to_ball'] - defender_offense_pairs['o_def'] + 180) % 360 - 180
defender_offense_pairs['orientation_diff_def_to_ball'] = np.abs(angle_diff_ball)

print("✓ OUTPUT defender-offense dataframe created")
print_memory()

# ==============================================================================
# STEP 4: Create INPUT defender-offense dataframe (CRITICAL - HIGHEST MEMORY)
# ==============================================================================
print("\n[4/7] Creating INPUT defender-offense relationships...")
print("⚠️  This is the HIGHEST memory operation in the notebook")

# Get player info
player_info_input = input_data.groupby('nfl_id').agg({
    'player_name': 'first',
    'player_position': 'first',
    'player_role': 'first',
    'player_side': 'first'
}).reset_index()

ball_info_input = input_data.groupby(['game_id', 'play_id']).agg({
    'ball_land_x': 'first',
    'ball_land_y': 'first'
}).reset_index()

# Merge input data with player info
input_with_info = input_data.merge(player_info_input, on='nfl_id', how='left', suffixes=('', '_info'))
input_with_info['player_name'] = input_with_info['player_name'].fillna(input_with_info['player_name_info'])
input_with_info['player_position'] = input_with_info['player_position'].fillna(input_with_info['player_position_info'])
input_with_info['player_role'] = input_with_info['player_role'].fillna(input_with_info['player_role_info'])
input_with_info['player_side'] = input_with_info['player_side'].fillna(input_with_info['player_side_info'])

# Separate into offense and defense
offense_input = input_with_info[input_with_info['player_side'] == 'Offense'].copy()
defense_input = input_with_info[input_with_info['player_side'] == 'Defense'].copy()

print(f"  Offense frames: {len(offense_input):,}")
print(f"  Defense frames: {len(defense_input):,}")
print_memory()

# Create cartesian product - THIS IS THE MEMORY KILLER
print("  Creating defender-offense pairs (cartesian product)...")
print("  ⚠️  WARNING: This operation may exceed available memory")

try:
    input_defender_offense_pairs = defense_input.merge(
        offense_input,
        on=['game_id', 'play_id', 'frame_id', 'week'],
        suffixes=('_def', '_off')
    )
    print(f"  ✓ Total pairs: {len(input_defender_offense_pairs):,}")
    print_memory()
    
    # If we got here, calculate distances
    print("  Calculating distances and orientations...")
    dx_off = input_defender_offense_pairs['x_off'] - input_defender_offense_pairs['x_def']
    dy_off = input_defender_offense_pairs['y_off'] - input_defender_offense_pairs['y_def']
    input_defender_offense_pairs['distance_def_to_off'] = np.sqrt(dx_off**2 + dy_off**2)
    
    print("✓ INPUT defender-offense dataframe created successfully!")
    print_memory()
    
except MemoryError as e:
    print(f"\n❌ MEMORY ERROR: {e}")
    print("  The INPUT cartesian product exceeded available memory")
    print_memory()
    exit(1)

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE - MEMORY DIAGNOSTIC SUMMARY")
print("="*80)
print(f"\nDataframes created:")
print(f"  input_data: {len(input_data):,} rows")
print(f"  output_data: {len(output_data):,} rows")
print(f"  defender_offense_pairs (OUTPUT): {len(defender_offense_pairs):,} rows")
print(f"  input_defender_offense_pairs (INPUT): {len(input_defender_offense_pairs):,} rows")

print_memory()

print("\n✅ All operations completed successfully!")
print("   Your notebook should be able to run without memory issues now.")


