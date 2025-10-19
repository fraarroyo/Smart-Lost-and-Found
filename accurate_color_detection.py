#!/usr/bin/env python3
"""
Accurate Color Detection System
Focused on detecting accurate colors from the main object in images.
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

class AccurateColorDetector:
    """Accurate color detection focused on main objects."""
    
    def __init__(self):
        # Enhanced color database with better color names
        self.color_database = {
            # Pink/Rose colors
            'pink': (255, 192, 203), 'rose': (255, 105, 180), 'rose_gold': (233, 150, 122),
            'light_pink': (255, 182, 193), 'hot_pink': (255, 20, 147), 'deep_pink': (255, 20, 147),
            'pale_pink': (250, 218, 221), 'dusty_pink': (217, 166, 169), 'blush': (222, 93, 131),
            
            # White/Off-white colors
            'white': (255, 255, 255), 'off_white': (248, 248, 255), 'ivory': (255, 255, 240),
            'cream': (255, 253, 208), 'pearl': (240, 248, 255), 'snow': (255, 250, 250),
            'alabaster': (237, 234, 229), 'eggshell': (240, 234, 214), 'vanilla': (243, 229, 171),
            
            # Blue colors
            'blue': (0, 0, 255), 'light_blue': (173, 216, 230), 'sky_blue': (135, 206, 235),
            'navy': (0, 0, 128), 'royal_blue': (65, 105, 225), 'steel_blue': (70, 130, 180),
            'powder_blue': (176, 224, 230), 'baby_blue': (135, 206, 235), 'azure': (240, 255, 255),
            
            # Basic colors
            'red': (255, 0, 0), 'green': (0, 128, 0), 'yellow': (255, 255, 0), 'orange': (255, 165, 0),
            'purple': (128, 0, 128), 'brown': (165, 42, 42), 'gray': (128, 128, 128), 'black': (0, 0, 0),
            'silver': (192, 192, 192), 'gold': (255, 215, 0)
        }
    
    def analyze_image_colors_accurate(self, image_path: str) -> Dict:
        """Analyze image colors with focus on main object."""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Focus on center region (main object is usually in center)
            center_region = self._extract_center_region(image_array)
            
            # Extract colors from center region
            colors = self._extract_colors_accurate(center_region)
            
            # Generate color names
            named_palette = self._generate_accurate_named_palette(colors)
            
            # Get primary and secondary colors
            primary_color = colors[0] if colors else None
            secondary_color = colors[1] if len(colors) > 1 else None
            
            # Get primary and secondary color names
            primary_color_name = named_palette[0]['name'] if named_palette else 'unknown'
            secondary_color_name = named_palette[1]['name'] if len(named_palette) > 1 else 'unknown'
            
            return {
                'success': True,
                'primary_colors': colors[:5],
                'all_colors': colors,
                'named_palette': named_palette,
                'primary_color': primary_color,
                'secondary_color': secondary_color,
                'primary_color_name': primary_color_name,
                'secondary_color_name': secondary_color_name,
                'color_names': [p.get('name', 'unknown') for p in named_palette],
                'confidence_score': 0.9,  # High confidence for accurate method
                'method': 'accurate_center_focus'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Accurate color detection failed: {str(e)}",
                'primary_colors': [],
                'all_colors': [],
                'named_palette': [],
                'confidence_score': 0.0
            }
    
    def _extract_center_region(self, image_array: np.ndarray) -> np.ndarray:
        """Extract center region of image where main object is likely located."""
        height, width = image_array.shape[:2]
        
        # Define center region (50% of image centered) - smaller region for better focus
        center_h_start = int(height * 0.25)
        center_h_end = int(height * 0.75)
        center_w_start = int(width * 0.25)
        center_w_end = int(width * 0.75)
        
        center_region = image_array[center_h_start:center_h_end, center_w_start:center_w_end]
        
        # Filter out very dark and very light pixels (likely background)
        filtered_region = self._filter_background_pixels(center_region)
        
        return filtered_region
    
    def _filter_background_pixels(self, image_array: np.ndarray) -> np.ndarray:
        """Filter out background pixels to focus on main object."""
        try:
            # Convert to HSV for better color filtering
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            
            # Create mask for non-background pixels
            # Exclude very dark pixels (V < 30) and very light pixels (V > 240)
            mask = (hsv[:, :, 2] > 30) & (hsv[:, :, 2] < 240)
            
            # Apply mask
            filtered_image = image_array.copy()
            filtered_image[~mask] = [255, 255, 255]  # Set background to white
            
            return filtered_image
            
        except Exception as e:
            print(f"Background filtering error: {e}")
            return image_array
    
    def _extract_colors_accurate(self, image_array: np.ndarray) -> List[Dict]:
        """Extract colors using accurate KMeans clustering."""
        try:
            # Reshape image to list of pixels
            pixels = image_array.reshape(-1, 3)
            
            # Use KMeans with optimal parameters for color detection
            kmeans = KMeans(
                n_clusters=8,  # Optimal number for color detection
                random_state=42,
                n_init=20,  # More iterations for accuracy
                max_iter=300  # More iterations for accuracy
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
                        'method': 'accurate_kmeans',
                        'cluster_id': i
                    }
                    colors.append(color_info)
            
            # Sort by frequency
            colors.sort(key=lambda x: x['frequency'], reverse=True)
            return colors[:8]  # Return top 8 colors
            
        except Exception as e:
            print(f"Accurate color extraction error: {e}")
            return []
    
    def _rgb_to_hsv(self, rgb: np.ndarray) -> Tuple[float, float, float]:
        """Convert RGB to HSV accurately."""
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
    
    def _generate_accurate_named_palette(self, colors: List[Dict]) -> List[Dict]:
        """Generate accurate named color palette with light color prioritization."""
        try:
            named_palette = []
            
            # First, identify and prioritize light colors
            light_colors = []
            other_colors = []
            
            for color in colors:
                if self._is_light_color(color['rgb']):
                    light_colors.append(color)
                else:
                    other_colors.append(color)
            
            # Sort light colors by frequency (descending)
            light_colors.sort(key=lambda x: x['frequency'], reverse=True)
            
            # Sort other colors by frequency (descending)
            other_colors.sort(key=lambda x: x['frequency'], reverse=True)
            
            # Combine: light colors first, then other colors
            prioritized_colors = light_colors + other_colors
            
            for i, color in enumerate(prioritized_colors[:6]):  # Top 6 colors
                color_name = self._get_accurate_color_name(color['rgb'])
                
                named_color = {
                    'name': color_name,
                    'rgb': color['rgb'],
                    'hsv': color['hsv'],
                    'percentage': color['percentage'],
                    'rank': i + 1
                }
                named_palette.append(named_color)
            
            return named_palette
            
        except Exception as e:
            print(f"Named palette generation error: {e}")
            return []
    
    def _get_accurate_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Get accurate color name using enhanced matching."""
        try:
            # Try webcolors first
            try:
                return webcolors.rgb_to_name(rgb)
            except ValueError:
                pass
            
            # Check for light colors first (common in phones, tumblers, etc.)
            if self._is_light_color(rgb):
                return self._get_light_color_name(rgb)
            
            # Find closest color in database
            min_distance = float('inf')
            closest_name = 'unknown'
            
            for name, color_rgb in self.color_database.items():
                # Use perceptual color distance (better than Euclidean)
                distance = self._perceptual_color_distance(rgb, color_rgb)
                if distance < min_distance:
                    min_distance = distance
                    closest_name = name
            
            return closest_name
            
        except Exception:
            return 'unknown'
    
    def _is_light_color(self, rgb: Tuple[int, int, int]) -> bool:
        """Check if color is light (high brightness)."""
        # Calculate brightness (V in HSV)
        r, g, b = rgb
        brightness = (r + g + b) / 3.0
        return brightness > 150  # Light threshold
    
    def _get_light_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Get name for light colors."""
        r, g, b = rgb
        
        # Check for specific light colors
        if r > 200 and g > 200 and b > 200:
            return 'white'
        elif r > g and r > b and r > 180:  # Light red/pink
            if g > 150 and b > 150:
                return 'light_pink'
            else:
                return 'pink'
        elif g > r and g > b and g > 180:  # Light green
            return 'light_green'
        elif b > r and b > g and b > 180:  # Light blue
            return 'light_blue'
        elif r > 180 and g > 180 and b < 150:  # Light yellow
            return 'light_yellow'
        elif r > 180 and g > 150 and b > 180:  # Light pink/rose
            return 'rose'
        else:
            return 'off_white'
    
    def _perceptual_color_distance(self, rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
        """Calculate perceptual color distance (better than Euclidean)."""
        try:
            # Convert to LAB color space for better perceptual distance
            lab1 = self._rgb_to_lab(rgb1)
            lab2 = self._rgb_to_lab(rgb2)
            
            # Calculate Euclidean distance in LAB space
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))
            
            return distance
            
        except Exception:
            # Fallback to simple Euclidean distance
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
    
    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to LAB color space."""
        try:
            # Convert RGB to LAB using OpenCV
            rgb_array = np.array([[rgb]], dtype=np.uint8)
            lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
            lab = lab_array[0, 0]
            return tuple(lab.astype(float))
        except Exception:
            return (0.0, 0.0, 0.0)


def enhance_existing_color_detection_accurate(image_path: str) -> Dict:
    """Accurate color detection function."""
    try:
        detector = AccurateColorDetector()
        result = detector.analyze_image_colors_accurate(image_path)
        return result
    except Exception as e:
        return {
            'success': False,
            'error': f"Accurate color detection failed: {str(e)}",
            'primary_colors': [],
            'all_colors': [],
            'named_palette': [],
            'confidence_score': 0.0
        }


if __name__ == "__main__":
    # Test the accurate color detection
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = enhance_existing_color_detection_accurate(image_path)
        print(f"Accurate color detection result: {result}")
    else:
        print("Usage: python accurate_color_detection.py <image_path>")
