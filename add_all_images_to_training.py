#!/usr/bin/env python3
"""
Script to add all uploaded images to the training dataset.
This script processes all images in the uploads and static/img directories
and adds them to the training dataset for model improvement.
"""

import os
import sys
import json
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Flask app and models
from app import app, add_all_uploaded_images_to_training, add_all_existing_items_to_training, get_training_dataset_stats

def main():
    """Main function to add all images to training dataset."""
    print("🚀 Starting comprehensive training data addition process...")
    print("=" * 60)
    
    with app.app_context():
        # Get initial stats
        print("📊 Initial training dataset statistics:")
        initial_stats = get_training_dataset_stats()
        print(f"   - Training samples: {initial_stats['total_samples']}")
        print(f"   - Training images: {initial_stats['total_images']}")
        print(f"   - Categories: {len(initial_stats['categories'])}")
        print()
        
        # Add existing items to training
        print("1️⃣ Adding existing database items to training...")
        try:
            item_count = add_all_existing_items_to_training()
            print(f"   ✅ Added {item_count} existing items to training dataset")
        except Exception as e:
            print(f"   ❌ Error adding existing items: {e}")
        print()
        
        # Add all uploaded images to training
        print("2️⃣ Adding all uploaded images to training...")
        try:
            image_count, errors = add_all_uploaded_images_to_training()
            print(f"   ✅ Added {image_count} uploaded images to training dataset")
            if errors > 0:
                print(f"   ⚠️  {errors} errors occurred during processing")
        except Exception as e:
            print(f"   ❌ Error adding uploaded images: {e}")
        print()
        
        # Get final stats
        print("📊 Final training dataset statistics:")
        final_stats = get_training_dataset_stats()
        print(f"   - Training samples: {final_stats['total_samples']} (+{final_stats['total_samples'] - initial_stats['total_samples']})")
        print(f"   - Training images: {final_stats['total_images']} (+{final_stats['total_images'] - initial_stats['total_images']})")
        print(f"   - Categories: {len(final_stats['categories'])}")
        
        if final_stats['category_counts']:
            print("\n📁 Images by category:")
            for category, count in final_stats['category_counts'].items():
                print(f"   - {category.replace('_', ' ').title()}: {count} images")
        
        print("\n🎉 Training data addition process completed!")
        print("=" * 60)
        print("💡 Next steps:")
        print("   1. Go to /admin/training to view training statistics")
        print("   2. Go to /admin/training/dataset to manage the dataset")
        print("   3. Consider retraining the model with the new data")
        print("   4. Monitor model performance improvements")

if __name__ == "__main__":
    main()
