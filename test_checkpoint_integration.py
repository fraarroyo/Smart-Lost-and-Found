#!/usr/bin/env python3
"""
Test script to demonstrate checkpoint integration with the UnifiedModel system.
"""

import os
import sys
from ml_models import UnifiedModel
import numpy as np
from PIL import Image

def create_test_image(filename="test_image.jpg", size=(224, 224)):
    """Create a simple test image for testing."""
    # Create a simple colored image
    img_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(filename)
    print(f"Created test image: {filename}")
    return filename

def test_checkpoint_integration():
    """Test the checkpoint integration."""
    print("=== Testing Checkpoint Integration ===\n")
    
    # Initialize the unified model
    print("1. Initializing UnifiedModel...")
    model = UnifiedModel()
    print("✓ UnifiedModel initialized")
    
    # Check checkpoint info
    print("\n2. Checking checkpoint model info...")
    checkpoint_info = model.get_checkpoint_info()
    print("Checkpoint Info:")
    for key, value in checkpoint_info.items():
        print(f"  {key}: {value}")
    
    if not checkpoint_info.get('loaded', False):
        print("ℹ️  Checkpoint model not available (optional)")
        return False
    
    print("✓ Checkpoint model loaded successfully")
    
    # Create test images
    print("\n3. Creating test images...")
    test_image1 = create_test_image("test1.jpg")
    test_image2 = create_test_image("test2.jpg")
    print("✓ Test images created")
    
    # Test checkpoint prediction
    print("\n4. Testing checkpoint prediction...")
    try:
        prediction1 = model.predict_with_checkpoint(test_image1)
        if prediction1 is not None:
            print(f"✓ Prediction successful! Output shape: {prediction1.shape}")
        else:
            print("❌ Prediction failed")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False
    
    # Test checkpoint embedding
    print("\n5. Testing checkpoint embedding...")
    try:
        embedding1 = model.get_checkpoint_embedding(test_image1)
        embedding2 = model.get_checkpoint_embedding(test_image2)
        
        if embedding1 and embedding2:
            print(f"✓ Embeddings generated! Length: {len(embedding1)}")
        else:
            print("❌ Embedding generation failed")
            return False
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return False
    
    # Test image comparison
    print("\n6. Testing image comparison...")
    try:
        similarity = model.compare_with_checkpoint(test_image1, test_image2)
        print(f"✓ Image similarity: {similarity:.4f}")
    except Exception as e:
        print(f"❌ Comparison error: {e}")
        return False
    
    # Test with existing object detection
    print("\n7. Testing integration with object detection...")
    try:
        objects = model.detect_objects(test_image1)
        print(f"✓ Object detection completed! Found {len(objects)} objects")
        
        # Show detected objects
        for i, obj in enumerate(objects):
            if isinstance(obj, dict) and 'class' in obj:
                print(f"  Object {i+1}: {obj['class']} (confidence: {obj.get('confidence', 0):.3f})")
    except Exception as e:
        print(f"❌ Object detection error: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    print("✓ Checkpoint model successfully integrated with UnifiedModel")
    print("✓ All functionality working correctly")
    
    return True

def cleanup_test_files():
    """Clean up test files."""
    test_files = ["test1.jpg", "test2.jpg", "test_image.jpg"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up: {file}")

if __name__ == "__main__":
    try:
        success = test_checkpoint_integration()
        if success:
            print("\n🎉 Checkpoint integration test completed successfully!")
        else:
            print("\n❌ Some tests failed. Check the output above.")
    finally:
        cleanup_test_files()
