#!/usr/bin/env python3
"""
Fast Color Detection System
Optimized for speed while maintaining good color accuracy.
"""

import cv2
import numpy as np
from PIL import Image
import webcolors
from sklearn.cluster import KMeans
import colorsys
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FastColorDetector:
    """Fast color detection optimized for speed."""
    
    def __init__(self):
        # Basic color database for fast lookup
        self.color_database = {
            'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 128, 0),
            'yellow': (255, 255, 0), 'orange': (255, 165, 0), 'purple': (128, 0, 128),
            'pink': (255, 192, 203), 'brown': (165, 42, 42), 'gray': (128, 128, 128),
            'black': (0, 0, 0), 'white': (255, 255, 255), 'silver': (192, 192, 192),
            'gold': (255, 215, 0), 'navy': (0, 0, 128), 'maroon': (128, 0, 0),
            'olive': (128, 128, 0), 'teal': (0, 128, 128), 'cyan': (0, 255, 255),
            'magenta': (255, 0, 255), 'lime': (0, 255, 0), 'indigo': (75, 0, 130)
        }
    
    def analyze_image_colors_fast(self, image_path: str, num_colors: int = 8) -> Dict:
        """Fast color analysis optimized for speed."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Resize for faster processing
            height, width = image_array.shape[:2]
            if height > 400 or width > 400:
                scale = min(400 / height, 400 / width)
                new_height = int(height * scale)
                new_width = int(width * scale)
                image_array = cv2.resize(image_array, (new_width, new_height))
            
            # Extract colors using optimized KMeans
            colors = self._extract_colors_fast_kmeans(image_array, num_colors)
            
            # Generate color names
            named_palette = self._generate_fast_named_palette(colors)
            
            # Basic analysis
            primary_color = colors[0] if colors else None
            secondary_color = colors[1] if len(colors) > 1 else None
            
            return {
                'success': True,
                'primary_colors': colors[:3],
                'all_colors': colors,
                'named_palette': named_palette,
                'primary_color': primary_color,
                'secondary_color': secondary_color,
                'confidence_score': 0.8,  # High confidence for fast method
                'method': 'fast_kmeans'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Fast color detection failed: {str(e)}",
                'primary_colors': [],
                'all_colors': [],
                'named_palette': [],
                'confidence_score': 0.0
            }
    
    def _extract_colors_fast_kmeans(self, image_array: np.ndarray, num_colors: int) -> List[Dict]:
        """Extract colors using fast KMeans clustering."""
        try:
            # Reshape image to list of pixels
            pixels = image_array.reshape(-1, 3)
            
            # Use optimized KMeans
            kmeans = KMeans(
                n_clusters=min(num_colors, 8),  # Limit clusters
                random_state=42,
                n_init=5,  # Reduced iterations
                max_iter=50  # Reduced max iterations
            )
            kmeans.fit(pixels)
            
            # Get cluster centers and labels
            centers = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # Calculate color frequencies
            label_counts = np.bincount(labels)
            
            colors = []
            for i, (center, count) in enumerate(zip(centers, label_counts)):
                if count > 0:
                    color_info = {
                        'rgb': tuple(center),
                        'frequency': int(count),
                        'percentage': float(count / len(pixels) * 100),
                        'hsv': self._rgb_to_hsv(center),
                        'method': 'fast_kmeans',
                        'cluster_id': i
                    }
                    colors.append(color_info)
            
            # Sort by frequency
            colors.sort(key=lambda x: x['frequency'], reverse=True)
            return colors
            
        except Exception as e:
            print(f"Fast KMeans error: {e}")
            return []
    
    def _rgb_to_hsv(self, rgb: np.ndarray) -> Tuple[float, float, float]:
        """Convert RGB to HSV quickly."""
        try:
            # Normalize RGB values
            rgb_normalized = rgb.astype(float) / 255.0
            
            # Convert to HSV
            hsv = colorsys.rgb_to_hsv(rgb_normalized[0], rgb_normalized[1], rgb_normalized[2])
            
            # Convert to 0-360, 0-100, 0-100 range
            h = hsv[0] * 360
            s = hsv[1] * 100
            v = hsv[2] * 100
            
            return (h, s, v)
            
        except Exception:
            return (0, 0, 0)
    
    def _generate_fast_named_palette(self, colors: List[Dict]) -> List[Dict]:
        """Generate named color palette quickly."""
        try:
            named_palette = []
            
            for color in colors[:6]:  # Top 6 colors
                color_name = self._get_closest_color_name_fast(color['rgb'])
                
                named_color = {
                    'name': color_name,
                    'rgb': color['rgb'],
                    'hsv': color['hsv'],
                    'percentage': color['percentage'],
                    'rank': len(named_palette) + 1
                }
                named_palette.append(named_color)
            
            return named_palette
            
        except Exception as e:
            print(f"Named palette generation error: {e}")
            return []
    
    def _get_closest_color_name_fast(self, rgb: Tuple[int, int, int]) -> str:
        """Get closest color name quickly."""
        try:
            # Try webcolors first
            try:
                return webcolors.rgb_to_name(rgb)
            except ValueError:
                pass
            
            # Find closest color in database
            min_distance = float('inf')
            closest_name = 'unknown'
            
            for name, color_rgb in self.color_database.items():
                distance = sum((a - b) ** 2 for a, b in zip(rgb, color_rgb))
                if distance < min_distance:
                    min_distance = distance
                    closest_name = name
            
            return closest_name
            
        except Exception:
            return 'unknown'


def enhance_existing_color_detection_fast(image_path: str) -> Dict:
    """Fast color detection function."""
    try:
        detector = FastColorDetector()
        result = detector.analyze_image_colors_fast(image_path)
        return result
    except Exception as e:
        return {
            'success': False,
            'error': f"Fast color detection failed: {str(e)}",
            'primary_colors': [],
            'all_colors': [],
            'named_palette': [],
            'confidence_score': 0.0
        }


if __name__ == "__main__":
    # Test the fast color detection
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = enhance_existing_color_detection_fast(image_path)
        print(f"Fast color detection result: {result}")
    else:
        print("Usage: python fast_color_detection.py <image_path>")
