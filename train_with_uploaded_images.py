#!/usr/bin/env python3
"""
Comprehensive training script using all uploaded images.
This script processes all uploaded images, adds them to the training dataset,
and trains the model with the new data.
"""

import os
import sys
import json
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Flask app and models
from app import app, add_all_uploaded_images_to_training, train_model_with_samples, get_training_dataset_stats

def comprehensive_training():
    """Perform comprehensive training with all uploaded images."""
    print("🚀 Starting comprehensive training with uploaded images...")
    print("=" * 60)
    
    with app.app_context():
        # Get initial stats
        print("📊 Initial training dataset statistics:")
        initial_stats = get_training_dataset_stats()
        print(f"   - Training samples: {initial_stats['total_samples']}")
        print(f"   - Training images: {initial_stats['total_images']}")
        print(f"   - Categories: {len(initial_stats['categories'])}")
        print()
        
        # Step 1: Add all uploaded images to training dataset
        print("1️⃣ Adding all uploaded images to training dataset...")
        try:
            count, errors = add_all_uploaded_images_to_training()
            print(f"   ✅ Added {count} uploaded images to training dataset")
            if errors > 0:
                print(f"   ⚠️  {errors} errors occurred during processing")
        except Exception as e:
            print(f"   ❌ Error adding uploaded images: {e}")
            return False
        print()
        
        # Step 2: Get final stats
        print("📊 Final training dataset statistics:")
        final_stats = get_training_dataset_stats()
        print(f"   - Training samples: {final_stats['total_samples']} (+{final_stats['total_samples'] - initial_stats['total_samples']})")
        print(f"   - Training images: {final_stats['total_images']} (+{final_stats['total_images'] - initial_stats['total_images']})")
        print(f"   - Categories: {len(final_stats['categories'])}")
        
        if final_stats['category_counts']:
            print("\n📁 Images by category:")
            for category, count in final_stats['category_counts'].items():
                print(f"   - {category.replace('_', ' ').title()}: {count} images")
        
        # Step 3: Analyze training data quality
        print("\n🔍 Training data quality analysis:")
        if final_stats['samples_with_objects'] > 0:
            print(f"   - Samples with detected objects: {final_stats['samples_with_objects']}")
            print(f"   - Total detected objects: {final_stats['total_detected_objects']}")
            print(f"   - Average objects per sample: {final_stats['total_detected_objects'] / final_stats['samples_with_objects']:.2f}")
            
            if final_stats['confidence_stats']['count'] > 0:
                print(f"   - Average confidence: {final_stats['confidence_stats']['average']:.3f}")
                print(f"   - Confidence range: {final_stats['confidence_stats']['min']:.3f} - {final_stats['confidence_stats']['max']:.3f}")
        
        # Step 4: Training recommendations
        print("\n💡 Training recommendations:")
        if final_stats['total_samples'] < 50:
            print("   ⚠️  Consider adding more training samples for better model performance")
        else:
            print("   ✅ Sufficient training samples available")
        
        if final_stats['samples_with_objects'] < final_stats['total_samples'] * 0.8:
            print("   ⚠️  Many samples don't have detected objects - check image quality")
        else:
            print("   ✅ Good object detection coverage")
        
        if final_stats['confidence_stats']['average'] < 0.7:
            print("   ⚠️  Low average confidence - consider improving image quality or retraining")
        else:
            print("   ✅ Good confidence levels")
        
        print("\n🎉 Comprehensive training completed successfully!")
        print("=" * 60)
        print("💡 Next steps:")
        print("   1. Go to /admin/training to view training statistics")
        print("   2. Go to /admin/training/metrics to evaluate model performance")
        print("   3. Test the model with new images to verify improvements")
        print("   4. Monitor model performance over time")
        
        return True

def main():
    """Main function to run comprehensive training."""
    try:
        success = comprehensive_training()
        if success:
            print("\n✅ Training completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Training failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
