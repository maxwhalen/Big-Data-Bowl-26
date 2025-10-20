#!/usr/bin/env python3
"""
Quick test script for NFL Big Data Bowl 2026 solution
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all imports work"""
    print("🧪 Testing imports...")
    try:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import mean_squared_error
        from catboost import CatBoostRegressor, Pool as CatPool
        print("✅ Basic imports successful")
        
        import torch
        import torch.nn as nn
        print("✅ PyTorch imports successful")
        
        from nfl_complete_solution import NFLConfig, DataLoader, FeatureEngineer
        print("✅ Solution imports successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test data loading"""
    print("\n🧪 Testing data loading...")
    try:
        from nfl_complete_solution import NFLConfig, DataLoader
        
        config = NFLConfig(dev_mode=True)
        data_loader = DataLoader(config)
        
        # Check if data files exist
        if not config.BASE_DIR.exists():
            print(f"❌ Data directory not found: {config.BASE_DIR}")
            return False
        
        # Check for at least one week of data
        week1_input = config.BASE_DIR / "train/input_2023_w01.csv"
        week1_output = config.BASE_DIR / "train/output_2023_w01.csv"
        
        if not week1_input.exists():
            print(f"❌ Input data not found: {week1_input}")
            return False
        
        if not week1_output.exists():
            print(f"❌ Output data not found: {week1_output}")
            return False
        
        print("✅ Data files found")
        
        # Try loading one week
        input_df, output_df = data_loader.load_week_data(1)
        print(f"✅ Week 1 data loaded: Input {input_df.shape}, Output {output_df.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering"""
    print("\n🧪 Testing feature engineering...")
    try:
        from nfl_complete_solution import NFLConfig, DataLoader, FeatureEngineer
        
        config = NFLConfig(dev_mode=True)
        data_loader = DataLoader(config)
        feature_engineer = FeatureEngineer()
        
        # Load small sample
        input_df, output_df = data_loader.load_week_data(1)
        
        # Test physics features
        features = feature_engineer.engineer_physics_features(input_df.head(100))
        print(f"✅ Physics features: {features.shape}")
        
        # Test sequence features
        features = feature_engineer.add_sequence_features(features)
        print(f"✅ Sequence features: {features.shape}")
        
        # Test formation features
        features = feature_engineer.add_formation_features(features)
        print(f"✅ Formation features: {features.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Feature engineering error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 NFL Big Data Bowl 2026 - Solution Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_loading,
        test_feature_engineering
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print("❌ Test failed, stopping here")
            break
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready to run the solution.")
        print("\n📋 Commands to run:")
        print("  Dev mode (fast):  python nfl_complete_solution.py dev")
        print("  Production mode:  python nfl_complete_solution.py prod")
    else:
        print("❌ Some tests failed. Please fix issues before running the solution.")

if __name__ == "__main__":
    main()
