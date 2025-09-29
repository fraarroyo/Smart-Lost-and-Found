#!/usr/bin/env python3
"""
Test script for Enhanced Color Detection System
Demonstrates the new color analysis capabilities.
"""

import os
import sys
from enhanced_color_detection import EnhancedColorDetector, enhance_existing_color_detection, get_enhanced_color_description

def test_enhanced_color_detection():
    """Test the enhanced color detection system."""
    print("🎨 Testing Enhanced Color Detection System")
    print("=" * 60)
    
    # Initialize detector
    detector = EnhancedColorDetector()
    
    # Find test images
    test_images = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
                if len(test_images) >= 3:
                    break
        if len(test_images) >= 3:
            break
    
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"Found {len(test_images)} test images")
    
    for i, img_path in enumerate(test_images, 1):
        print(f"\n🖼️  Image {i}: {os.path.basename(img_path)}")
        print("-" * 40)
        
        try:
            # Test full image analysis
            print("📊 Full Image Analysis:")
            analysis = detector.analyze_image_colors(img_path)
            
            if analysis.get('error'):
                print("  ❌ Analysis failed")
                continue
            
            # Display primary and secondary colors
            primary = analysis.get('primary_color')
            secondary = analysis.get('secondary_color')
            
            if primary:
                print(f"  🎨 Primary: {primary.get('name', 'unknown')} {primary.get('hex', '')}")
                print(f"     RGB: {primary.get('rgb', [0,0,0])}")
                print(f"     HSV: {[f'{x:.2f}' for x in primary.get('hsv', [0,0,0])]}")
            
            if secondary:
                print(f"  🎨 Secondary: {secondary.get('name', 'unknown')} {secondary.get('hex', '')}")
            
            # Display color features
            features = analysis.get('color_features', {})
            if features:
                print(f"  🌡️  Temperature: {'Warm' if features.get('is_warm') else 'Cool'}")
                print(f"  💎 Saturation: {'High' if features.get('is_saturated') else 'Low'} ({features.get('saturation', 0):.1f})")
                print(f"  ☀️  Brightness: {'High' if features.get('is_bright') else 'Low'} ({features.get('brightness', 0):.1f})")
            
            # Display color harmony
            harmony = analysis.get('color_harmony', {})
            if harmony:
                print(f"  🎭 Harmony: {harmony.get('harmony_type', 'unknown')} (score: {harmony.get('harmony_score', 0):.2f})")
            
            # Display color palette
            palette = analysis.get('dominant_colors', [])
            if palette:
                print(f"  🎨 Color Palette ({len(palette)} colors):")
                for j, color in enumerate(palette[:5]):  # Show first 5 colors
                    print(f"     {j+1}. {color.get('name', 'unknown')} {color.get('hex', '')} ({color.get('percentage', 0):.1f}%)")
            
            # Test object-specific analysis (simulate bounding box)
            print("\n📦 Object-Specific Analysis (simulated bbox):")
            try:
                # Create a simulated bounding box (center 50% of image)
                from PIL import Image
                with Image.open(img_path) as img:
                    w, h = img.size
                    bbox = [w//4, h//4, 3*w//4, 3*h//4]  # Center 50% of image
                
                obj_analysis = enhance_existing_color_detection(img_path, bbox, 'phone')
                if not obj_analysis.get('error'):
                    obj_primary = obj_analysis.get('primary_color')
                    if obj_primary:
                        print(f"  📱 Phone Color: {obj_primary.get('name', 'unknown')} {obj_primary.get('hex', '')}")
                    
                    # Test color description generation
                    description = get_enhanced_color_description(obj_analysis)
                    print(f"  📝 Description: {description}")
                
            except Exception as e:
                print(f"  ⚠️  Object analysis failed: {e}")
            
            # Test color comparison
            if len(palette) >= 2:
                print(f"\n🔍 Color Comparison:")
                color1 = palette[0]
                color2 = palette[1] if len(palette) > 1 else palette[0]
                
                euclidean_sim = detector.compare_colors(color1, color2, 'euclidean')
                cosine_sim = detector.compare_colors(color1, color2, 'cosine')
                
                print(f"  Euclidean similarity: {euclidean_sim:.3f}")
                print(f"  Cosine similarity: {cosine_sim:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error analyzing {img_path}: {e}")
    
    print(f"\n✅ Enhanced Color Detection Test Completed!")
    print("\n🎯 Key Features Demonstrated:")
    print("  • Multi-color space analysis (RGB, HSV, LAB)")
    print("  • Advanced color clustering and quantization")
    print("  • Color harmony detection")
    print("  • Object-specific color analysis")
    print("  • Color temperature and saturation analysis")
    print("  • Color comparison and matching")
    print("  • Enhanced color naming and description")

def test_color_matching():
    """Test color matching capabilities."""
    print("\n🔍 Testing Color Matching System")
    print("=" * 40)
    
    detector = EnhancedColorDetector()
    
    # Create sample colors for testing
    test_colors = [
        {'rgb': (255, 0, 0), 'name': 'red'},
        {'rgb': (0, 255, 0), 'name': 'green'},
        {'rgb': (0, 0, 255), 'name': 'blue'},
        {'rgb': (255, 255, 0), 'name': 'yellow'},
        {'rgb': (128, 128, 128), 'name': 'gray'}
    ]
    
    # Test color comparisons
    print("Color Similarity Matrix:")
    print("     ", end="")
    for color in test_colors:
        print(f"{color['name']:>8}", end="")
    print()
    
    for i, color1 in enumerate(test_colors):
        print(f"{color1['name']:>5}", end="")
        for j, color2 in enumerate(test_colors):
            similarity = detector.compare_colors(color1, color2, 'euclidean')
            print(f"{similarity:>8.2f}", end="")
        print()
    
    print("\n✅ Color Matching Test Completed!")

if __name__ == "__main__":
    try:
        test_enhanced_color_detection()
        test_color_matching()
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
