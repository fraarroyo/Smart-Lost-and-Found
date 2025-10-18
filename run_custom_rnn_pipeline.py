#!/usr/bin/env python3
"""
Complete Custom RNN Pipeline Runner for BARYONYX Lost & Found System
Runs the entire pipeline: dataset generation → training → evaluation
"""

import subprocess
import sys
import os
import json
from datetime import datetime
import torch

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    try:
        print(f"🔄 {description}...")
        print(f"   Running: {script_name}")
        
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        print(f"✅ {description} completed successfully")
        if result.stdout:
            # Print only important output
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['error', 'warning', 'completed', 'saved', 'created']):
                    print(f"   {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}:")
        print(f"   Return code: {e.returncode}")
        if e.stderr:
            print(f"   Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {script_name}")
        return False

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'numpy', 'pandas', 'scikit-learn', 
        'matplotlib', 'seaborn', 'json'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_device():
    """Check available device"""
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return 'cuda'
    else:
        print("💻 Using CPU (CUDA not available)")
        return 'cpu'

def create_directories():
    """Create necessary directories"""
    directories = [
        'enhanced_datasets',
        'models/custom_rnn',
        'models/custom_rnn/checkpoints',
        'models/custom_rnn/plots',
        'models/custom_rnn/evaluation',
        'models/custom_rnn/evaluation/plots',
        'models/custom_rnn/evaluation/reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("📁 Created necessary directories")

def save_pipeline_info(device, start_time):
    """Save pipeline execution information"""
    info = {
        'start_time': start_time.isoformat(),
        'device': device,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'pipeline_version': '1.0.0'
    }
    
    with open('pipeline_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"📋 Pipeline info saved to pipeline_info.json")

def main():
    """Main pipeline execution"""
    start_time = datetime.now()
    
    print("🚀 BARYONYX Custom RNN Pipeline")
    print("=" * 50)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        print("❌ Requirements check failed. Please install missing packages.")
        return False
    
    # Check device
    device = check_device()
    print()
    
    # Create directories
    print("📁 Setting up directories...")
    create_directories()
    print()
    
    # Step 1: Generate enhanced datasets
    print("📊 Step 1: Generating Enhanced Datasets")
    print("-" * 40)
    if not run_script("enhanced_rnn_datasets.py", "Dataset generation"):
        print("❌ Dataset generation failed. Exiting.")
        return False
    
    # Check if datasets were created
    required_files = [
        "enhanced_datasets/enhanced_user_behavior.json",
        "enhanced_datasets/enhanced_item_matching.json", 
        "enhanced_datasets/enhanced_temporal.json",
        "enhanced_datasets/enhanced_text_descriptions.json"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All datasets generated successfully")
    print()
    
    # Step 2: Train custom RNN models
    print("🧠 Step 2: Training Custom RNN Models")
    print("-" * 40)
    if not run_script("custom_rnn_training.py", "Model training"):
        print("❌ Model training failed. Exiting.")
        return False
    
    # Check if models were created
    model_files = [
        "models/custom_rnn/item_matching_model.pth",
        "models/custom_rnn/behavior_predictor.pth",
        "models/custom_rnn/text_encoder.pth",
        "models/custom_rnn/vocab.pkl"
    ]
    
    missing_models = [f for f in model_files if not os.path.exists(f)]
    if missing_models:
        print(f"❌ Missing model files: {missing_models}")
        return False
    
    print("✅ All models trained successfully")
    print()
    
    # Step 3: Evaluate models
    print("🔍 Step 3: Evaluating Models")
    print("-" * 40)
    if not run_script("custom_rnn_evaluation.py", "Model evaluation"):
        print("❌ Model evaluation failed. Exiting.")
        return False
    
    print("✅ Model evaluation completed successfully")
    print()
    
    # Step 4: Generate summary
    print("📋 Step 4: Generating Summary")
    print("-" * 40)
    
    # Load evaluation results
    try:
        with open('models/custom_rnn/evaluation/evaluation_results.json', 'r') as f:
            eval_results = json.load(f)
        
        print("📊 Model Performance Summary:")
        print("=" * 30)
        for model_name, metrics in eval_results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        # Find best model
        best_model = max(eval_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
        
    except FileNotFoundError:
        print("⚠️ Evaluation results not found")
    
    # Save pipeline info
    save_pipeline_info(device, start_time)
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n🎉 Pipeline Completed Successfully!")
    print("=" * 40)
    print(f"Total time: {duration}")
    print(f"Device used: {device}")
    print(f"Models saved to: models/custom_rnn/")
    print(f"Evaluation results: models/custom_rnn/evaluation/")
    print(f"Integration guide: custom_rnn_integration_guide.md")
    print()
    print("🚀 Your custom RNN models are ready for integration!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
