#!/usr/bin/env python3
"""
Fix the DATA_DIR path in the notebook to use a relative path that works from the notebook's working directory
"""

import json
import os

# Read the notebook
with open('final_submission.ipynb', 'r') as f:
    notebook = json.load(f)

# Use a relative path that should work from the prediction directory
relative_path = "../kaggle/input/nfl-big-data-bowl-2026-prediction/"

# Find and replace the DATA_DIR path
updated = False
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'source' in cell:
        for i, line in enumerate(cell['source']):
            if 'DATA_DIR = Path(' in line and 'kaggle/input/nfl-big-data-bowl-2026-prediction' in line:
                cell['source'][i] = f'    DATA_DIR = Path("{relative_path}")\n'
                print(f"Updated DATA_DIR path to: {relative_path}")
                updated = True

if updated:
    # Write the updated notebook
    with open('final_submission.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook updated successfully!")
else:
    print("DATA_DIR path not found in notebook")
