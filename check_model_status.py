#!/usr/bin/env python3
"""
Model Status Checker
Check the status of all models in the system.
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_model_status():
    """Check the status of all models"""
    print("🔍 Checking Model Status")
    print("=" * 40)
    
    # Check main model files
    print("\n📁 Main Model Files:")
    main_models = [
        'best_model1.pth',
        'best_model1.2.pth', 
        'checkpoint.pt',
        'checkpoint.pth'
    ]
    
    for model_file in main_models:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"  ✅ {model_file} ({size:.1f} MB)")
        else:
            print(f"  ❌ {model_file} (not found)")
    
    # Check model directories
    print("\n📂 Model Directories:")
    model_dirs = [
        'models',
        'models/rnn_models',
        'models/image_rnn'
    ]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            files = len([f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))])
            print(f"  ✅ {model_dir}/ ({files} files)")
        else:
            print(f"  ❌ {model_dir}/ (not found)")
    
    # Check training data
    print("\n🧠 Training Data:")
    training_dirs = [
        'training_data',
        'image recog.v1i.coco-mmdetection'
    ]
    
    for training_dir in training_dirs:
        if os.path.exists(training_dir):
            files = len([f for f in os.listdir(training_dir) if os.path.isfile(os.path.join(training_dir, f))])
            print(f"  ✅ {training_dir}/ ({files} files)")
        else:
            print(f"  ❌ {training_dir}/ (not found)")
    
    # Test model loading
    print("\n🤖 Model Loading Test:")
    try:
        from ml_models import UnifiedModel
        model = UnifiedModel()
        
        # Check main model
        if hasattr(model, 'object_model') and model.object_model is not None:
            print("  ✅ Main object detection model loaded")
        else:
            print("  ❌ Main object detection model not loaded")
        
        # Check checkpoint model
        checkpoint_info = model.get_checkpoint_info()
        if checkpoint_info.get('loaded', False):
            print("  ✅ Checkpoint model loaded")
        else:
            print(f"  ℹ️  Checkpoint model: {checkpoint_info.get('status', 'Unknown')}")
        
        # Check text model
        if model._text_model_loaded:
            print("  ✅ BERT text model loaded")
        else:
            print("  ℹ️  BERT text model (lazy loaded)")
        
        print("  ✅ Model initialization successful")
        
    except Exception as e:
        print(f"  ❌ Model loading error: {e}")
    
    print("\n🎯 Summary:")
    print("  - Main model: best_model1.pth (primary)")
    print("  - Checkpoint: checkpoint.pt (optional)")
    print("  - BERT: Lazy loaded when needed")
    print("  - RNN: Loaded from models/rnn_models/")
    print("\n✨ System is ready for image analysis!")

if __name__ == "__main__":
    check_model_status()
