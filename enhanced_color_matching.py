#!/usr/bin/env python3
"""
Enhanced Color Matching System
Advanced color matching using ultra-enhanced color detection for better item matching.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ultra_enhanced_color_detection import UltraEnhancedColorDetector, compare_colors_ultra
import json

class EnhancedColorMatcher:
    """Enhanced color matching system for item comparison."""
    
    def __init__(self):
        self.color_detector = UltraEnhancedColorDetector()
        self.matching_weights = {
            'primary_color': 0.4,
            'secondary_color': 0.2,
            'color_harmony': 0.15,
            'material_match': 0.1,
            'perceptual_similarity': 0.1,
            'color_temperature': 0.05
        }
    
    def calculate_color_similarity(self, item1_colors: Dict, item2_colors: Dict) -> Dict:
        """Calculate comprehensive color similarity between two items."""
        try:
            # Extract color information
            item1_primary = item1_colors.get('primary_rgb', [0, 0, 0])
            item1_secondary = item1_colors.get('secondary_rgb', [])
            item1_enhanced = item1_colors.get('enhanced_analysis', {})
            
            item2_primary = item2_colors.get('primary_rgb', [0, 0, 0])
            item2_secondary = item2_colors.get('secondary_rgb', [])
            item2_enhanced = item2_colors.get('enhanced_analysis', {})
            
            # Calculate primary color similarity
            primary_similarity = self._calculate_primary_color_similarity(
                item1_primary, item2_primary
            )
            
            # Calculate secondary color similarity
            secondary_similarity = self._calculate_secondary_color_similarity(
                item1_secondary, item2_secondary
            )
            
            # Calculate enhanced similarity features
            enhanced_similarity = self._calculate_enhanced_similarity(
                item1_enhanced, item2_enhanced
            )
            
            # Calculate overall similarity score
            overall_similarity = self._calculate_overall_similarity(
                primary_similarity, secondary_similarity, enhanced_similarity
            )
            
            return {
                'overall_similarity': overall_similarity,
                'primary_similarity': primary_similarity,
                'secondary_similarity': secondary_similarity,
                'enhanced_similarity': enhanced_similarity,
                'match_quality': self._assess_match_quality(overall_similarity),
                'matching_details': self._generate_matching_details(
                    primary_similarity, secondary_similarity, enhanced_similarity
                )
            }
            
        except Exception as e:
            return {
                'overall_similarity': 0.0,
                'error': str(e),
                'match_quality': 'poor'
            }
    
    def _calculate_primary_color_similarity(self, color1: List, color2: List) -> Dict:
        """Calculate primary color similarity using multiple methods."""
        if not color1 or not color2 or len(color1) != 3 or len(color2) != 3:
            return {'similarity': 0.0, 'method': 'invalid_colors'}
        
        # Convert to tuples for comparison
        color1_tuple = tuple(color1)
        color2_tuple = tuple(color2)
        
        # Use ultra-enhanced color comparison
        comparison = compare_colors_ultra(color1_tuple, color2_tuple)
        
        return {
            'similarity': comparison['weighted_similarity'],
            'rgb_similarity': comparison['rgb_similarity'],
            'hsv_similarity': comparison['hsv_similarity'],
            'lab_similarity': comparison['lab_similarity'],
            'perceptual_match': comparison['perceptual_match'],
            'method': 'ultra_enhanced'
        }
    
    def _calculate_secondary_color_similarity(self, color1: List, color2: List) -> Dict:
        """Calculate secondary color similarity."""
        if not color1 or not color2:
            return {'similarity': 0.0, 'method': 'no_secondary_colors'}
        
        if len(color1) != 3 or len(color2) != 3:
            return {'similarity': 0.0, 'method': 'invalid_secondary_colors'}
        
        # Use ultra-enhanced color comparison
        comparison = compare_colors_ultra(tuple(color1), tuple(color2))
        
        return {
            'similarity': comparison['weighted_similarity'],
            'perceptual_match': comparison['perceptual_match'],
            'method': 'ultra_enhanced'
        }
    
    def _calculate_enhanced_similarity(self, enhanced1: Dict, enhanced2: Dict) -> Dict:
        """Calculate enhanced similarity features."""
        if not enhanced1 or not enhanced2:
            return {
                'color_harmony': 0.0,
                'material_match': 0.0,
                'perceptual_similarity': 0.0,
                'color_temperature': 0.0
            }
        
        # Color harmony similarity
        harmony1 = enhanced1.get('harmony_analysis', {})
        harmony2 = enhanced2.get('harmony_analysis', {})
        color_harmony = self._compare_color_harmony(harmony1, harmony2)
        
        # Material match
        material1 = enhanced1.get('material_analysis', {})
        material2 = enhanced2.get('material_analysis', {})
        material_match = self._compare_materials(material1, material2)
        
        # Perceptual similarity
        perceptual1 = enhanced1.get('perceptual_analysis', {})
        perceptual2 = enhanced2.get('perceptual_analysis', {})
        perceptual_similarity = self._compare_perceptual_properties(perceptual1, perceptual2)
        
        # Color temperature match
        color_temperature = self._compare_color_temperature(perceptual1, perceptual2)
        
        return {
            'color_harmony': color_harmony,
            'material_match': material_match,
            'perceptual_similarity': perceptual_similarity,
            'color_temperature': color_temperature
        }
    
    def _compare_color_harmony(self, harmony1: Dict, harmony2: Dict) -> float:
        """Compare color harmony between two items."""
        if not harmony1 or not harmony2:
            return 0.0
        
        # Compare harmony types
        type1 = harmony1.get('harmony_type', 'unknown')
        type2 = harmony2.get('harmony_type', 'unknown')
        
        if type1 == type2 and type1 != 'unknown':
            return 0.8  # High similarity for same harmony type
        elif type1 in ['complementary', 'triadic'] and type2 in ['complementary', 'triadic']:
            return 0.6  # Medium similarity for similar harmony types
        elif type1 == 'monochromatic' and type2 == 'monochromatic':
            return 0.9  # Very high similarity for monochromatic
        else:
            return 0.3  # Low similarity for different harmony types
    
    def _compare_materials(self, material1: Dict, material2: Dict) -> float:
        """Compare material types between two items."""
        if not material1 or not material2:
            return 0.0
        
        mat1 = material1.get('detected_material', 'unknown')
        mat2 = material2.get('detected_material', 'unknown')
        
        if mat1 == mat2 and mat1 != 'unknown':
            return 0.9  # High similarity for same material
        elif mat1 in ['metal', 'plastic'] and mat2 in ['metal', 'plastic']:
            return 0.6  # Medium similarity for similar material categories
        else:
            return 0.2  # Low similarity for different materials
    
    def _compare_perceptual_properties(self, perceptual1: Dict, perceptual2: Dict) -> float:
        """Compare perceptual properties between two items."""
        if not perceptual1 or not perceptual2:
            return 0.0
        
        # Compare color intensity
        intensity1 = perceptual1.get('color_intensity', 'unknown')
        intensity2 = perceptual2.get('color_intensity', 'unknown')
        intensity_match = 1.0 if intensity1 == intensity2 else 0.3
        
        # Compare saturation (normalized)
        sat1 = perceptual1.get('average_saturation', 0)
        sat2 = perceptual2.get('average_saturation', 0)
        saturation_similarity = 1.0 - abs(sat1 - sat2) / 100.0
        
        # Compare brightness (normalized)
        bright1 = perceptual1.get('average_brightness', 0)
        bright2 = perceptual2.get('average_brightness', 0)
        brightness_similarity = 1.0 - abs(bright1 - bright2) / 100.0
        
        # Average the similarities
        return (intensity_match + saturation_similarity + brightness_similarity) / 3.0
    
    def _compare_color_temperature(self, perceptual1: Dict, perceptual2: Dict) -> float:
        """Compare color temperature between two items."""
        if not perceptual1 or not perceptual2:
            return 0.0
        
        temp1 = perceptual1.get('color_temperature', 'unknown')
        temp2 = perceptual2.get('color_temperature', 'unknown')
        
        if temp1 == temp2 and temp1 != 'unknown':
            return 1.0  # Perfect match
        elif temp1 in ['warm', 'cool'] and temp2 in ['warm', 'cool']:
            return 0.5  # Partial match for temperature categories
        else:
            return 0.0  # No match
    
    def _calculate_overall_similarity(self, primary: Dict, secondary: Dict, enhanced: Dict) -> float:
        """Calculate overall similarity score."""
        weights = self.matching_weights
        
        # Primary color similarity
        primary_score = primary.get('similarity', 0.0)
        
        # Secondary color similarity
        secondary_score = secondary.get('similarity', 0.0)
        
        # Enhanced similarity components
        color_harmony = enhanced.get('color_harmony', 0.0)
        material_match = enhanced.get('material_match', 0.0)
        perceptual_similarity = enhanced.get('perceptual_similarity', 0.0)
        color_temperature = enhanced.get('color_temperature', 0.0)
        
        # Calculate weighted average
        overall = (
            weights['primary_color'] * primary_score +
            weights['secondary_color'] * secondary_score +
            weights['color_harmony'] * color_harmony +
            weights['material_match'] * material_match +
            weights['perceptual_similarity'] * perceptual_similarity +
            weights['color_temperature'] * color_temperature
        )
        
        return min(1.0, max(0.0, overall))
    
    def _assess_match_quality(self, similarity: float) -> str:
        """Assess the quality of color match."""
        if similarity >= 0.8:
            return 'excellent'
        elif similarity >= 0.6:
            return 'good'
        elif similarity >= 0.4:
            return 'fair'
        elif similarity >= 0.2:
            return 'poor'
        else:
            return 'very_poor'
    
    def _generate_matching_details(self, primary: Dict, secondary: Dict, enhanced: Dict) -> str:
        """Generate detailed matching description."""
        details = []
        
        # Primary color details
        primary_sim = primary.get('similarity', 0.0)
        if primary_sim > 0.7:
            details.append("Primary colors are very similar")
        elif primary_sim > 0.4:
            details.append("Primary colors are somewhat similar")
        else:
            details.append("Primary colors are different")
        
        # Secondary color details
        secondary_sim = secondary.get('similarity', 0.0)
        if secondary_sim > 0.7:
            details.append("Secondary colors match well")
        elif secondary_sim > 0.4:
            details.append("Secondary colors are somewhat similar")
        
        # Enhanced features
        harmony = enhanced.get('color_harmony', 0.0)
        if harmony > 0.6:
            details.append("Color harmony patterns match")
        
        material = enhanced.get('material_match', 0.0)
        if material > 0.6:
            details.append("Materials appear similar")
        
        perceptual = enhanced.get('perceptual_similarity', 0.0)
        if perceptual > 0.6:
            details.append("Visual appearance is similar")
        
        return "; ".join(details) if details else "No significant color similarities detected"

def enhance_item_color_matching(item1, item2) -> Dict:
    """Enhanced color matching for two items."""
    try:
        # Extract color information from items
        item1_colors = {
            'primary_rgb': item1.color.split(',')[0].strip() if item1.color else [0, 0, 0],
            'secondary_rgb': item1.color.split(',')[1].strip() if item1.color and ',' in item1.color else [],
            'enhanced_analysis': json.loads(item1.enhanced_analysis) if hasattr(item1, 'enhanced_analysis') and item1.enhanced_analysis else {}
        }
        
        item2_colors = {
            'primary_rgb': item2.color.split(',')[0].strip() if item2.color else [0, 0, 0],
            'secondary_rgb': item2.color.split(',')[1].strip() if item2.color and ',' in item2.color else [],
            'enhanced_analysis': json.loads(item2.enhanced_analysis) if hasattr(item2, 'enhanced_analysis') and item2.enhanced_analysis else {}
        }
        
        # Create matcher and calculate similarity
        matcher = EnhancedColorMatcher()
        similarity_result = matcher.calculate_color_similarity(item1_colors, item2_colors)
        
        return similarity_result
        
    except Exception as e:
        return {
            'overall_similarity': 0.0,
            'error': str(e),
            'match_quality': 'error'
        }

# Global matcher instance
enhanced_color_matcher = EnhancedColorMatcher()

def calculate_enhanced_color_similarity(item1_colors: Dict, item2_colors: Dict) -> Dict:
    """Calculate enhanced color similarity between two items."""
    return enhanced_color_matcher.calculate_color_similarity(item1_colors, item2_colors)

if __name__ == "__main__":
    # Test the enhanced color matching
    print("🎨 Testing Enhanced Color Matching System")
    print("=" * 50)
    
    # Test with sample color data
    test_colors1 = {
        'primary_rgb': [255, 0, 0],  # Red
        'secondary_rgb': [0, 0, 255],  # Blue
        'enhanced_analysis': {
            'harmony_analysis': {'harmony_type': 'complementary', 'harmony_score': 0.8},
            'material_analysis': {'detected_material': 'plastic', 'material_confidence': 0.7},
            'perceptual_analysis': {'color_temperature': 'warm', 'color_intensity': 'vibrant'}
        }
    }
    
    test_colors2 = {
        'primary_rgb': [200, 0, 0],  # Dark red
        'secondary_rgb': [0, 0, 200],  # Dark blue
        'enhanced_analysis': {
            'harmony_analysis': {'harmony_type': 'complementary', 'harmony_score': 0.7},
            'material_analysis': {'detected_material': 'plastic', 'material_confidence': 0.8},
            'perceptual_analysis': {'color_temperature': 'warm', 'color_intensity': 'moderate'}
        }
    }
    
    # Calculate similarity
    result = calculate_enhanced_color_similarity(test_colors1, test_colors2)
    
    print(f"Overall Similarity: {result['overall_similarity']:.3f}")
    print(f"Match Quality: {result['match_quality']}")
    print(f"Matching Details: {result['matching_details']}")
    
    print("\n✅ Enhanced Color Matching Test Completed!")
