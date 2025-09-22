#!/usr/bin/env python3
"""
Training System Demo Script

This script demonstrates how the training system automatically processes uploaded items
and organizes them into a structured training dataset.
"""

import os
import json
import shutil
from datetime import datetime

def show_training_system_info():
    """Display information about the training system."""
    print("=" * 60)
    print("TRAINING SYSTEM OVERVIEW")
    print("=" * 60)
    
    print("\n📁 Directory Structure:")
    print("training_data/")
    print("├── images/")
    print("│   ├── electronics/")
    print("│   ├── clothing/")
    print("│   ├── accessories/")
    print("│   └── documents/")
    print("├── labels/")
    print("└── training_sample_*.json")
    print("\nmodels/")
    print("├── object_detector.pth")
    print("└── confidence_adjuster.pkl")
    
    print("\n🔄 Automatic Training Process:")
    print("1. User uploads item with image")
    print("2. Object detection runs on image")
    print("3. Item automatically added to training dataset")
    print("4. Image copied to category-specific folder")
    print("5. Training sample JSON created with metadata")
    print("6. Model confidence adjustments applied")
    
    print("\n📊 Training Data Management:")
    print("- All uploaded items become training samples")
    print("- Images organized by category")
    print("- Detected objects used as initial labels")
    print("- User feedback improves labels over time")
    print("- Admin can export complete dataset")

def check_training_directories():
    """Check if training directories exist and show their contents."""
    print("\n" + "=" * 60)
    print("TRAINING DIRECTORY STATUS")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check training data directory
    training_dir = os.path.join(base_dir, 'training_data')
    if os.path.exists(training_dir):
        print(f"✅ training_data/ directory exists")
        
        # Check images directory
        images_dir = os.path.join(training_dir, 'images')
        if os.path.exists(images_dir):
            print(f"✅ training_data/images/ directory exists")
            
            # List categories
            categories = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
            if categories:
                print(f"📁 Found categories: {', '.join(categories)}")
                for category in categories:
                    category_path = os.path.join(images_dir, category)
                    image_count = len([f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))])
                    print(f"   - {category}: {image_count} images")
            else:
                print("   - No category directories found yet")
        else:
            print("❌ training_data/images/ directory missing")
    else:
        print("❌ training_data/ directory missing")
    
    # Check models directory
    models_dir = os.path.join(base_dir, 'models')
    if os.path.exists(models_dir):
        print(f"✅ models/ directory exists")
        model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pth', '.pkl'))]
        if model_files:
            print(f"📁 Model files: {', '.join(model_files)}")
        else:
            print("   - No model files found yet")
    else:
        print("❌ models/ directory missing")

def show_training_samples():
    """Show recent training samples."""
    print("\n" + "=" * 60)
    print("RECENT TRAINING SAMPLES")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_dir = os.path.join(base_dir, 'training_data')
    
    if os.path.exists(training_dir):
        training_files = [f for f in os.listdir(training_dir) if f.endswith('.json')]
        training_files.sort(key=lambda x: os.path.getmtime(os.path.join(training_dir, x)), reverse=True)
        
        if training_files:
            print(f"Found {len(training_files)} training samples:")
            for i, file in enumerate(training_files[:5]):  # Show last 5
                file_path = os.path.join(training_dir, file)
                try:
                    with open(file_path, 'r') as f:
                        sample = json.load(f)
                    
                    print(f"\n📄 {file}:")
                    print(f"   Item ID: {sample.get('item_id', 'N/A')}")
                    print(f"   Category: {sample.get('category', 'N/A')}")
                    print(f"   Status: {sample.get('status', 'N/A')}")
                    print(f"   Detected Objects: {len(sample.get('detected_objects', []))}")
                    print(f"   Date: {sample.get('timestamp', 'N/A')[:10]}")
                    
                except Exception as e:
                    print(f"   Error reading {file}: {e}")
        else:
            print("No training samples found yet.")
    else:
        print("Training directory not found.")

def demonstrate_workflow():
    """Demonstrate the training workflow."""
    print("\n" + "=" * 60)
    print("TRAINING WORKFLOW DEMONSTRATION")
    print("=" * 60)
    
    print("\n🎯 How to test the training system:")
    print("\n1. Upload an item:")
    print("   - Go to /add_item")
    print("   - Upload an image with a phone, wallet, etc.")
    print("   - Fill in the item details")
    print("   - Submit the form")
    
    print("\n2. Check training dataset:")
    print("   - Go to /admin/training/dataset (as admin)")
    print("   - You'll see the item added to training data")
    print("   - Image copied to category folder")
    print("   - Training sample JSON created")
    
    print("\n3. Provide feedback:")
    print("   - Go to the item detail page")
    print("   - Click 'Provide Feedback'")
    print("   - Rate the detection accuracy")
    print("   - Submit feedback")
    
    print("\n4. Monitor improvements:")
    print("   - Upload similar images")
    print("   - Compare confidence scores")
    print("   - Check admin training dashboard")
    
    print("\n5. Export dataset:")
    print("   - Go to /admin/training/dataset")
    print("   - Click 'Export Dataset'")
    print("   - Download complete training dataset")

def main():
    """Main demo function."""
    print("🤖 LOST & FOUND TRAINING SYSTEM DEMO")
    print("=" * 60)
    
    show_training_system_info()
    check_training_directories()
    show_training_samples()
    demonstrate_workflow()
    
    print("\n" + "=" * 60)
    print("🎉 Training system is ready!")
    print("=" * 60)
    print("\nKey Features:")
    print("✅ Automatic training data collection")
    print("✅ Organized image storage by category")
    print("✅ User feedback integration")
    print("✅ Confidence adjustment system")
    print("✅ Admin training management")
    print("✅ Dataset export functionality")
    
    print("\nNext Steps:")
    print("1. Start the Flask application: python app.py")
    print("2. Upload some items to build the training dataset")
    print("3. Provide feedback to improve the model")
    print("4. Monitor training progress in admin dashboard")

if __name__ == "__main__":
    main() 