#!/usr/bin/env python3
"""
Quick Data Clear Script
Simple script to clear all data without confirmation prompts.
"""

import os
import shutil
import glob

def quick_clear():
    """Quickly clear all data"""
    print("🧹 Quick Data Clear")
    
    # Clear JSON files
    print("Clearing JSON files...")
    for json_file in glob.glob('**/*.json', recursive=True):
        try:
            os.remove(json_file)
            print(f"  Removed: {json_file}")
        except:
            pass
    
    # Clear training data
    print("Clearing training data...")
    for dir_name in ['training_data', 'training_images', 'training_labels', 'static/uploads']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            os.makedirs(dir_name, exist_ok=True)
            print(f"  Cleared: {dir_name}")
    
    # Clear model files (except best_model1.pth)
    print("Clearing model files...")
    model_files = ['best_model.pth', 'checkpoint.pt', 'checkpoint.pth', 'unified_model.pth']
    for model_file in model_files:
        if os.path.exists(model_file):
            os.remove(model_file)
            print(f"  Removed: {model_file}")
    
    # Clear RNN models
    if os.path.exists('models/rnn_models'):
        shutil.rmtree('models/rnn_models')
        os.makedirs('models/rnn_models', exist_ok=True)
        print("  Cleared RNN models")
    
    # Clear cache
    print("Clearing cache...")
    for pattern in ['**/__pycache__', '**/*.pyc', '**/*.log']:
        for item in glob.glob(pattern, recursive=True):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
            except:
                pass
    
    print("✅ Quick clear completed!")

if __name__ == "__main__":
    quick_clear()
