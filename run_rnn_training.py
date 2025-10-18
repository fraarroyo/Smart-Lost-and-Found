#!/usr/bin/env python3
"""
Quick script to generate datasets and train RNN models
"""

import subprocess
import sys
import os

def run_script(script_name):
    """Run a Python script and handle errors"""
    try:
        print(f"🔄 Running {script_name}...")
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {script_name} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}:")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main execution function"""
    
    print("🚀 BARYONYX RNN Training Pipeline")
    print("=" * 50)
    
    # Step 1: Generate datasets
    print("\n📊 Step 1: Generating training datasets...")
    if not run_script("rnn_training_datasets.py"):
        print("Failed to generate datasets. Exiting.")
        return
    
    # Check if datasets were created
    required_files = [
        "rnn_user_behavior_dataset.json",
        "rnn_description_dataset.csv", 
        "rnn_temporal_dataset.json"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return
    
    print("✅ All datasets generated successfully")
    
    # Step 2: Train models
    print("\n🧠 Step 2: Training RNN models...")
    if not run_script("train_rnn_models.py"):
        print("Failed to train models. Exiting.")
        return
    
    print("✅ All models trained successfully")
    
    # Step 3: Summary
    print("\n📋 Training Summary:")
    print("✅ User Behavior LSTM - Trained for predicting user actions")
    print("✅ Description Bidirectional RNN - Trained for text classification")
    print("✅ Temporal Pattern RNN - Trained for time-based predictions")
    print("\n🎯 Models are ready for use in the BARYONYX system!")

if __name__ == "__main__":
    main()
