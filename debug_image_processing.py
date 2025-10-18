#!/usr/bin/env python3
"""Debug image processing workflow"""

import sys
import os
import traceback

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

def debug_image_processing():
    try:
        print("🔍 Debugging Image Processing Workflow")
        print("=" * 50)
        
        # Test with existing image
        test_image = 'temp_cropped_1028027548053176122.jpg'
        if not os.path.exists(test_image):
            print("❌ No test image found")
            return
        
        print(f"📸 Testing with: {test_image}")
        
        # Import the enhanced image analysis function
        from app import enhanced_image_analysis
        
        print("🔍 Running enhanced image analysis...")
        result = enhanced_image_analysis(test_image)
        
        print("✅ Image analysis completed!")
        print(f"📊 Detected objects: {len(result.get('detected_objects', []))}")
        print(f"🎨 Color info: {result.get('color_info', {})}")
        print(f"📏 Size info: {result.get('size_info', {})}")
        print(f"🏗️ Material info: {result.get('material_info', {})}")
        print(f"📈 Confidence score: {result.get('confidence_score', 0)}")
        
        # Check if objects were detected
        detected_objects = result.get('detected_objects', [])
        if detected_objects:
            print(f"\n🎯 Detected Objects:")
            for i, obj in enumerate(detected_objects, 1):
                print(f"  {i}. {obj.get('class', 'unknown')} (confidence: {obj.get('confidence', 0):.2f})")
        else:
            print("\n⚠️ No objects detected - this might be the issue!")
            
    except Exception as e:
        print(f"❌ Error in image processing: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_image_processing()
