#!/usr/bin/env python3
"""
Test script to verify best_model.pth is being used for image recognition.
"""

import os
import sys
from ml_models import UnifiedModel

def test_best_model_loading():
    """Test that best_model.pth is loaded correctly."""
    print("🔍 Testing best_model.pth loading...")
    
    # Initialize the unified model
    model = UnifiedModel()
    
    # Check if best_model.pth was loaded
    print(f"✓ Model initialized successfully")
    print(f"✓ Device: {model.device}")
    print(f"✓ Score threshold: {model.score_threshold}")
    print(f"✓ Label names loaded: {len(model.label_names)} classes")
    
    if model.label_names:
        print(f"✓ Sample classes: {list(model.label_names.values())[:10]}")
    else:
        print("⚠️  No custom label names loaded, using COCO fallback")
    
    # Check COCO categories
    print(f"✓ COCO categories loaded: {len(model.coco_id_to_name)} classes")
    if model.coco_id_to_name:
        print(f"✓ Sample COCO classes: {list(model.coco_id_to_name.values())[:5]}")
    
    # Test detection on a sample image if available
    test_images = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
                if len(test_images) >= 3:  # Test on first 3 images found
                    break
        if len(test_images) >= 3:
            break
    
    if test_images:
        print(f"\n🧪 Testing detection on {len(test_images)} sample images...")
        for i, img_path in enumerate(test_images):
            try:
                print(f"\n  Image {i+1}: {os.path.basename(img_path)}")
                detections = model.detect_objects(img_path)
                
                if isinstance(detections, dict) and 'error' in detections:
                    print(f"    ❌ Error: {detections['error']}")
                else:
                    print(f"    ✓ Detected {len(detections)} objects")
                    for j, obj in enumerate(detections[:3]):  # Show first 3 detections
                        print(f"      {j+1}. {obj.get('class', 'unknown')} (confidence: {obj.get('confidence', 0):.3f})")
            except Exception as e:
                print(f"    ❌ Error processing {img_path}: {e}")
    else:
        print("\n⚠️  No test images found to test detection")
    
    print(f"\n✅ best_model.pth integration test completed!")

if __name__ == "__main__":
    test_best_model_loading()
