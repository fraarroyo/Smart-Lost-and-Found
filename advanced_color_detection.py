#!/usr/bin/env python3
"""
Advanced Color Detection System
Enhanced color analysis with multiple algorithms, better accuracy, and comprehensive color matching.
"""

import cv2
import numpy as np
from PIL import Image
import webcolors
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import colorsys
from typing import Dict, List, Tuple, Optional, Union
import json
import os
from collections import Counter
import math

class AdvancedColorDetector:
    """Advanced color detection and analysis system with multiple algorithms."""
    
    def __init__(self):
        # Comprehensive color database with RGB, HSV, and LAB values
        self.color_database = self._initialize_color_database()
        
        # Color space weights for different analysis types
        self.color_space_weights = {
            'perceptual': {'L': 0.5, 'a': 0.3, 'b': 0.2},  # LAB space
            'hue_based': {'H': 0.6, 'S': 0.3, 'V': 0.1},   # HSV space
            'rgb_based': {'R': 0.33, 'G': 0.33, 'B': 0.34}  # RGB space
        }
        
        # Advanced clustering parameters
        self.clustering_methods = ['kmeans', 'dbscan', 'meanshift', 'hierarchical']
        
    def _initialize_color_database(self) -> Dict:
        """Initialize comprehensive color database with multiple color spaces."""
        colors = {
            # Primary Colors
            'red': {'rgb': (255, 0, 0), 'hsv': (0, 100, 100), 'lab': (53, 80, 67)},
            'green': {'rgb': (0, 128, 0), 'hsv': (120, 100, 50), 'lab': (46, -51, 50)},
            'blue': {'rgb': (0, 0, 255), 'hsv': (240, 100, 100), 'lab': (32, 79, -107)},
            'yellow': {'rgb': (255, 255, 0), 'hsv': (60, 100, 100), 'lab': (97, -21, 94)},
            'cyan': {'rgb': (0, 255, 255), 'hsv': (180, 100, 100), 'lab': (91, -48, -14)},
            'magenta': {'rgb': (255, 0, 255), 'hsv': (300, 100, 100), 'lab': (60, 98, -60)},
            
            # Extended Red Family
            'crimson': {'rgb': (220, 20, 60), 'hsv': (348, 91, 86), 'lab': (48, 75, 38)},
            'maroon': {'rgb': (128, 0, 0), 'hsv': (0, 100, 50), 'lab': (25, 48, 38)},
            'burgundy': {'rgb': (128, 0, 32), 'hsv': (345, 100, 50), 'lab': (25, 48, 20)},
            'scarlet': {'rgb': (255, 36, 0), 'hsv': (8, 100, 100), 'lab': (54, 75, 67)},
            'coral': {'rgb': (255, 127, 80), 'hsv': (16, 69, 100), 'lab': (71, 45, 50)},
            'salmon': {'rgb': (250, 128, 114), 'hsv': (6, 54, 98), 'lab': (70, 35, 25)},
            'rose': {'rgb': (255, 192, 203), 'hsv': (330, 25, 100), 'lab': (85, 20, 5)},
            'pink': {'rgb': (255, 192, 203), 'hsv': (330, 25, 100), 'lab': (85, 20, 5)},
            
            # Extended Orange Family
            'orange': {'rgb': (255, 165, 0), 'hsv': (39, 100, 100), 'lab': (74, 23, 78)},
            'amber': {'rgb': (255, 191, 0), 'hsv': (45, 100, 100), 'lab': (80, 10, 85)},
            'gold': {'rgb': (255, 215, 0), 'hsv': (51, 100, 100), 'lab': (87, -5, 90)},
            'peach': {'rgb': (255, 218, 185), 'hsv': (28, 27, 100), 'lab': (88, 15, 35)},
            'apricot': {'rgb': (251, 206, 177), 'hsv': (24, 29, 98), 'lab': (85, 12, 30)},
            
            # Extended Green Family
            'lime': {'rgb': (0, 255, 0), 'hsv': (120, 100, 100), 'lab': (88, -86, 83)},
            'emerald': {'rgb': (80, 200, 120), 'hsv': (140, 60, 78), 'lab': (75, -40, 35)},
            'mint': {'rgb': (152, 251, 152), 'hsv': (120, 39, 98), 'lab': (95, -35, 35)},
            'olive': {'rgb': (128, 128, 0), 'hsv': (60, 100, 50), 'lab': (51, -12, 56)},
            'forest': {'rgb': (34, 139, 34), 'hsv': (120, 76, 55), 'lab': (50, -50, 50)},
            'sage': {'rgb': (158, 183, 158), 'hsv': (120, 14, 72), 'lab': (73, -15, 15)},
            
            # Extended Blue Family
            'navy': {'rgb': (0, 0, 128), 'hsv': (240, 100, 50), 'lab': (12, 47, -64)},
            'sky': {'rgb': (135, 206, 235), 'hsv': (197, 43, 92), 'lab': (82, -20, -25)},
            'azure': {'rgb': (0, 127, 255), 'hsv': (210, 100, 100), 'lab': (55, 10, -100)},
            'teal': {'rgb': (0, 128, 128), 'hsv': (180, 100, 50), 'lab': (48, -28, -8)},
            'turquoise': {'rgb': (64, 224, 208), 'hsv': (174, 71, 88), 'lab': (83, -48, -5)},
            'royal': {'rgb': (65, 105, 225), 'hsv': (225, 71, 88), 'lab': (50, 25, -85)},
            'indigo': {'rgb': (75, 0, 130), 'hsv': (271, 100, 51), 'lab': (20, 58, -78)},
            
            # Extended Purple Family
            'purple': {'rgb': (128, 0, 128), 'hsv': (300, 100, 50), 'lab': (25, 58, -78)},
            'violet': {'rgb': (238, 130, 238), 'hsv': (300, 45, 93), 'lab': (70, 50, -50)},
            'lavender': {'rgb': (230, 230, 250), 'hsv': (240, 8, 98), 'lab': (92, 5, -15)},
            'plum': {'rgb': (221, 160, 221), 'hsv': (300, 28, 87), 'lab': (75, 25, -25)},
            'mauve': {'rgb': (224, 176, 255), 'hsv': (270, 31, 100), 'lab': (80, 25, -40)},
            
            # Extended Brown Family
            'brown': {'rgb': (165, 42, 42), 'hsv': (0, 75, 65), 'lab': (35, 50, 35)},
            'tan': {'rgb': (210, 180, 140), 'hsv': (34, 33, 82), 'lab': (75, 10, 35)},
            'beige': {'rgb': (245, 245, 220), 'hsv': (60, 10, 96), 'lab': (96, -5, 20)},
            'chocolate': {'rgb': (210, 105, 30), 'hsv': (25, 86, 82), 'lab': (55, 35, 65)},
            'coffee': {'rgb': (111, 78, 55), 'hsv': (25, 50, 44), 'lab': (40, 20, 30)},
            'caramel': {'rgb': (255, 213, 154), 'hsv': (36, 40, 100), 'lab': (88, 15, 50)},
            
            # Extended Gray Family
            'gray': {'rgb': (128, 128, 128), 'hsv': (0, 0, 50), 'lab': (53, 0, 0)},
            'silver': {'rgb': (192, 192, 192), 'hsv': (0, 0, 75), 'lab': (75, 0, 0)},
            'charcoal': {'rgb': (54, 69, 79), 'hsv': (200, 32, 31), 'lab': (30, -5, -10)},
            'slate': {'rgb': (112, 128, 144), 'hsv': (210, 22, 56), 'lab': (55, -5, -15)},
            'pewter': {'rgb': (161, 161, 170), 'hsv': (240, 5, 67), 'lab': (68, 0, -5)},
            
            # Black and White
            'black': {'rgb': (0, 0, 0), 'hsv': (0, 0, 0), 'lab': (0, 0, 0)},
            'white': {'rgb': (255, 255, 255), 'hsv': (0, 0, 100), 'lab': (100, 0, 0)},
            'ivory': {'rgb': (255, 255, 240), 'hsv': (60, 6, 100), 'lab': (98, -2, 10)},
            'cream': {'rgb': (255, 253, 208), 'hsv': (57, 18, 100), 'lab': (98, -8, 25)},
        }
        
        return colors
    
    def analyze_image_colors_advanced(self, image_path: str, num_colors: int = 10) -> Dict:
        """Advanced comprehensive color analysis using multiple algorithms."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Multiple color extraction methods
            kmeans_colors = self._extract_colors_kmeans(image_array, num_colors)
            meanshift_colors = self._extract_colors_meanshift(image_array, num_colors)
            histogram_colors = self._extract_colors_histogram(image_array, num_colors)
            
            # Combine and analyze results
            all_colors = kmeans_colors + meanshift_colors + histogram_colors
            dominant_colors = self._consolidate_color_results(all_colors, num_colors)
            
            # Advanced analysis
            color_harmony = self._analyze_advanced_harmony(dominant_colors)
            color_temperature = self._analyze_color_temperature_advanced(dominant_colors)
            color_contrast = self._analyze_color_contrast(dominant_colors)
            color_saturation = self._analyze_saturation_distribution(image_array)
            
            # Generate comprehensive description
            description = self._generate_advanced_color_description(dominant_colors, color_harmony, color_temperature)
            
            return {
                'primary_color': dominant_colors[0] if dominant_colors else None,
                'secondary_color': dominant_colors[1] if len(dominant_colors) > 1 else None,
                'tertiary_color': dominant_colors[2] if len(dominant_colors) > 2 else None,
                'all_colors': dominant_colors,
                'color_harmony': color_harmony,
                'color_temperature': color_temperature,
                'color_contrast': color_contrast,
                'saturation_analysis': color_saturation,
                'description': description,
                'confidence': self._calculate_color_confidence(dominant_colors),
                'analysis_method': 'advanced_multi_algorithm'
            }
            
        except Exception as e:
            print(f"Error in advanced color analysis: {e}")
            return self._get_default_color_analysis()
    
    def _extract_colors_kmeans(self, image_array: np.ndarray, num_colors: int) -> List[Dict]:
        """Extract colors using K-means clustering with multiple initializations."""
        try:
            # Reshape image to list of pixels
            pixels = image_array.reshape(-1, 3)
            
            # Remove very dark and very light pixels (likely noise)
            brightness = np.mean(pixels, axis=1)
            mask = (brightness > 20) & (brightness < 235)
            filtered_pixels = pixels[mask]
            
            if len(filtered_pixels) < num_colors:
                filtered_pixels = pixels
            
            # Multiple K-means runs with different initializations
            best_colors = []
            best_score = float('inf')
            
            for _ in range(5):  # 5 different initializations
                kmeans = KMeans(n_clusters=min(num_colors, len(filtered_pixels)), 
                              random_state=None, n_init=10)
                kmeans.fit(filtered_pixels)
                
                # Calculate inertia (lower is better)
                if kmeans.inertia_ < best_score:
                    best_score = kmeans.inertia_
                    best_colors = kmeans.cluster_centers_
            
            # Convert to color dictionaries
            colors = []
            for color in best_colors:
                rgb = tuple(map(int, color))
                color_info = self._create_color_info(rgb)
                colors.append(color_info)
            
            return colors
            
        except Exception as e:
            print(f"Error in K-means color extraction: {e}")
            return []
    
    def _extract_colors_meanshift(self, image_array: np.ndarray, num_colors: int) -> List[Dict]:
        """Extract colors using Mean Shift clustering."""
        try:
            pixels = image_array.reshape(-1, 3)
            
            # Remove extreme pixels
            brightness = np.mean(pixels, axis=1)
            mask = (brightness > 25) & (brightness < 230)
            filtered_pixels = pixels[mask]
            
            if len(filtered_pixels) < 10:
                filtered_pixels = pixels
            
            # Mean Shift clustering
            meanshift = MeanShift(bandwidth=30, max_iter=100)
            meanshift.fit(filtered_pixels)
            
            colors = []
            for color in meanshift.cluster_centers_:
                rgb = tuple(map(int, color))
                color_info = self._create_color_info(rgb)
                colors.append(color_info)
            
            return colors
            
        except Exception as e:
            print(f"Error in Mean Shift color extraction: {e}")
            return []
    
    def _extract_colors_histogram(self, image_array: np.ndarray, num_colors: int) -> List[Dict]:
        """Extract colors using histogram analysis."""
        try:
            # Calculate 3D histogram
            hist, bins = np.histogramdd(image_array.reshape(-1, 3), bins=32)
            
            # Find peaks in histogram
            peaks = []
            for i in range(hist.shape[0]):
                for j in range(hist.shape[1]):
                    for k in range(hist.shape[2]):
                        if hist[i, j, k] > 0:
                            # Convert bin indices back to RGB values
                            r = int((bins[0][i] + bins[0][i+1]) / 2)
                            g = int((bins[1][j] + bins[1][j+1]) / 2)
                            b = int((bins[2][k] + bins[2][k+1]) / 2)
                            peaks.append((r, g, b, hist[i, j, k]))
            
            # Sort by frequency and take top colors
            peaks.sort(key=lambda x: x[3], reverse=True)
            top_peaks = peaks[:num_colors]
            
            colors = []
            for r, g, b, freq in top_peaks:
                color_info = self._create_color_info((r, g, b))
                color_info['frequency'] = freq
                colors.append(color_info)
            
            return colors
            
        except Exception as e:
            print(f"Error in histogram color extraction: {e}")
            return []
    
    def _consolidate_color_results(self, all_colors: List[Dict], num_colors: int) -> List[Dict]:
        """Consolidate results from multiple algorithms and remove duplicates."""
        if not all_colors:
            return []
        
        # Group similar colors
        consolidated = []
        used_colors = set()
        
        for color in all_colors:
            rgb = color['rgb']
            is_duplicate = False
            
            for used_color in consolidated:
                if self._colors_are_similar(rgb, used_color['rgb'], threshold=25):
                    is_duplicate = True
                    # Merge frequencies if available
                    if 'frequency' in color and 'frequency' in used_color:
                        used_color['frequency'] += color['frequency']
                    break
            
            if not is_duplicate:
                consolidated.append(color)
                used_colors.add(rgb)
        
        # Sort by frequency or confidence and return top colors
        consolidated.sort(key=lambda x: x.get('frequency', x.get('confidence', 0)), reverse=True)
        return consolidated[:num_colors]
    
    def _create_color_info(self, rgb: Tuple[int, int, int]) -> Dict:
        """Create comprehensive color information dictionary."""
        r, g, b = rgb
        
        # Convert to different color spaces
        hsv = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
        lab = self._rgb_to_lab(r, g, b)
        
        # Find closest named color
        closest_name = self._find_closest_named_color(rgb)
        
        # Calculate color properties
        brightness = (r + g + b) / 3
        saturation = hsv[1] * 100
        hue = hsv[0] * 360
        
        return {
            'rgb': rgb,
            'hsv': (int(hsv[0] * 360), int(hsv[1] * 100), int(hsv[2] * 100)),
            'lab': lab,
            'name': closest_name,
            'brightness': brightness,
            'saturation': saturation,
            'hue': hue,
            'confidence': self._calculate_color_confidence_single(rgb)
        }
    
    def _rgb_to_lab(self, r: int, g: int, b: int) -> Tuple[float, float, float]:
        """Convert RGB to LAB color space."""
        # Normalize RGB values
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        # Convert to XYZ
        if r_norm > 0.04045:
            r_norm = ((r_norm + 0.055) / 1.055) ** 2.4
        else:
            r_norm = r_norm / 12.92
            
        if g_norm > 0.04045:
            g_norm = ((g_norm + 0.055) / 1.055) ** 2.4
        else:
            g_norm = g_norm / 12.92
            
        if b_norm > 0.04045:
            b_norm = ((b_norm + 0.055) / 1.055) ** 2.4
        else:
            b_norm = b_norm / 12.92
        
        # Apply transformation matrix
        x = r_norm * 0.4124564 + g_norm * 0.3575761 + b_norm * 0.1804375
        y = r_norm * 0.2126729 + g_norm * 0.7151522 + b_norm * 0.0721750
        z = r_norm * 0.0193339 + g_norm * 0.1191920 + b_norm * 0.9503041
        
        # Normalize by D65 illuminant
        x = x / 0.95047
        y = y / 1.00000
        z = z / 1.08883
        
        # Convert to LAB
        if y > 0.008856:
            fy = y ** (1/3)
        else:
            fy = (7.787 * y) + (16/116)
            
        if x > 0.008856:
            fx = x ** (1/3)
        else:
            fx = (7.787 * x) + (16/116)
            
        if z > 0.008856:
            fz = z ** (1/3)
        else:
            fz = (7.787 * z) + (16/116)
        
        L = (116 * fy) - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return (L, a, b)
    
    def _find_closest_named_color(self, rgb: Tuple[int, int, int]) -> str:
        """Find the closest named color from the database."""
        r, g, b = rgb
        min_distance = float('inf')
        closest_name = 'unknown'
        
        for name, color_data in self.color_database.items():
            db_rgb = color_data['rgb']
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, db_rgb)))
            
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        
        return closest_name
    
    def _colors_are_similar(self, rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int], threshold: int = 30) -> bool:
        """Check if two RGB colors are similar within threshold."""
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
        return distance < threshold
    
    def _analyze_advanced_harmony(self, colors: List[Dict]) -> Dict:
        """Analyze color harmony using advanced algorithms."""
        if len(colors) < 2:
            return {'type': 'single', 'harmony_score': 0.5}
        
        # Extract hue values
        hues = [color['hue'] for color in colors]
        
        # Analyze different harmony types
        harmony_scores = {}
        
        # Complementary harmony (180° apart)
        complementary_score = self._calculate_complementary_harmony(hues)
        harmony_scores['complementary'] = complementary_score
        
        # Triadic harmony (120° apart)
        triadic_score = self._calculate_triadic_harmony(hues)
        harmony_scores['triadic'] = triadic_score
        
        # Analogous harmony (30° apart)
        analogous_score = self._calculate_analogous_harmony(hues)
        harmony_scores['analogous'] = analogous_score
        
        # Find best harmony type
        best_harmony = max(harmony_scores.items(), key=lambda x: x[1])
        
        return {
            'type': best_harmony[0],
            'score': best_harmony[1],
            'all_scores': harmony_scores
        }
    
    def _calculate_complementary_harmony(self, hues: List[float]) -> float:
        """Calculate complementary harmony score."""
        if len(hues) < 2:
            return 0.0
        
        score = 0.0
        count = 0
        
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                diff = abs(hues[i] - hues[j])
                # Check if colors are approximately 180° apart
                if min(diff, 360 - diff) < 15:  # 15° tolerance
                    score += 1.0
                count += 1
        
        return score / count if count > 0 else 0.0
    
    def _calculate_triadic_harmony(self, hues: List[float]) -> float:
        """Calculate triadic harmony score."""
        if len(hues) < 3:
            return 0.0
        
        score = 0.0
        count = 0
        
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                for k in range(j + 1, len(hues)):
                    h1, h2, h3 = hues[i], hues[j], hues[k]
                    # Check if colors are approximately 120° apart
                    diff12 = min(abs(h1 - h2), 360 - abs(h1 - h2))
                    diff23 = min(abs(h2 - h3), 360 - abs(h2 - h3))
                    diff31 = min(abs(h3 - h1), 360 - abs(h3 - h1))
                    
                    if all(diff < 20 for diff in [diff12, diff23, diff31]):  # 20° tolerance
                        score += 1.0
                    count += 1
        
        return score / count if count > 0 else 0.0
    
    def _calculate_analogous_harmony(self, hues: List[float]) -> float:
        """Calculate analogous harmony score."""
        if len(hues) < 2:
            return 0.0
        
        score = 0.0
        count = 0
        
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                diff = min(abs(hues[i] - hues[j]), 360 - abs(hues[i] - hues[j]))
                # Check if colors are within 30° of each other
                if diff < 30:
                    score += 1.0
                count += 1
        
        return score / count if count > 0 else 0.0
    
    def _analyze_color_temperature_advanced(self, colors: List[Dict]) -> Dict:
        """Advanced color temperature analysis."""
        if not colors:
            return {'temperature': 'neutral', 'score': 0.5}
        
        # Calculate average hue
        avg_hue = sum(color['hue'] for color in colors) / len(colors)
        
        # Determine temperature based on hue
        if 0 <= avg_hue <= 60 or 300 <= avg_hue <= 360:
            temperature = 'warm'
            score = 0.8
        elif 180 <= avg_hue <= 240:
            temperature = 'cool'
            score = 0.8
        else:
            temperature = 'neutral'
            score = 0.5
        
        return {
            'temperature': temperature,
            'score': score,
            'average_hue': avg_hue
        }
    
    def _analyze_color_contrast(self, colors: List[Dict]) -> Dict:
        """Analyze color contrast between dominant colors."""
        if len(colors) < 2:
            return {'contrast': 'low', 'score': 0.3}
        
        # Calculate contrast between primary colors
        primary = colors[0]
        secondary = colors[1] if len(colors) > 1 else primary
        
        # Luminance contrast
        l1 = primary['lab'][0]
        l2 = secondary['lab'][0]
        contrast_ratio = (max(l1, l2) + 0.05) / (min(l1, l2) + 0.05)
        
        # Color contrast (Euclidean distance in LAB space)
        lab1 = primary['lab']
        lab2 = secondary['lab']
        color_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))
        
        # Determine contrast level
        if contrast_ratio > 7 or color_distance > 50:
            contrast = 'high'
            score = 0.9
        elif contrast_ratio > 3 or color_distance > 25:
            contrast = 'medium'
            score = 0.6
        else:
            contrast = 'low'
            score = 0.3
        
        return {
            'contrast': contrast,
            'score': score,
            'contrast_ratio': contrast_ratio,
            'color_distance': color_distance
        }
    
    def _analyze_saturation_distribution(self, image_array: np.ndarray) -> Dict:
        """Analyze saturation distribution across the image."""
        # Convert to HSV
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        
        # Calculate statistics
        mean_sat = np.mean(saturation)
        std_sat = np.std(saturation)
        max_sat = np.max(saturation)
        
        # Determine saturation level
        if mean_sat > 150:
            level = 'high'
        elif mean_sat > 100:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'level': level,
            'mean': float(mean_sat),
            'std': float(std_sat),
            'max': float(max_sat)
        }
    
    def _generate_advanced_color_description(self, colors: List[Dict], harmony: Dict, temperature: Dict) -> str:
        """Generate advanced color description."""
        if not colors:
            return "No colors detected"
        
        primary = colors[0]
        primary_name = primary['name']
        
        description_parts = [primary_name]
        
        # Add secondary color if present
        if len(colors) > 1:
            secondary = colors[1]
            if secondary['name'] != primary_name:
                description_parts.append(f"with {secondary['name']} accents")
        
        # Add harmony information
        if harmony['score'] > 0.6:
            harmony_type = harmony['type'].replace('_', ' ')
            description_parts.append(f"in {harmony_type} harmony")
        
        # Add temperature information
        if temperature['score'] > 0.7:
            description_parts.append(f"with {temperature['temperature']} undertones")
        
        # Add saturation information
        if primary['saturation'] > 80:
            description_parts.append("vibrant")
        elif primary['saturation'] < 30:
            description_parts.append("muted")
        
        return " ".join(description_parts)
    
    def _calculate_color_confidence(self, colors: List[Dict]) -> float:
        """Calculate overall confidence in color analysis."""
        if not colors:
            return 0.0
        
        # Average confidence of all colors
        avg_confidence = sum(color.get('confidence', 0.5) for color in colors) / len(colors)
        
        # Bonus for having multiple distinct colors
        if len(colors) > 1:
            avg_confidence += 0.1
        
        return min(avg_confidence, 1.0)
    
    def _calculate_color_confidence_single(self, rgb: Tuple[int, int, int]) -> float:
        """Calculate confidence for a single color."""
        r, g, b = rgb
        
        # Higher confidence for more saturated colors
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        saturation = (max_val - min_val) / max_val if max_val > 0 else 0
        
        # Higher confidence for colors that are not too dark or too light
        brightness = (r + g + b) / 3
        brightness_score = 1.0 - abs(brightness - 128) / 128
        
        # Combine scores
        confidence = (saturation * 0.6 + brightness_score * 0.4)
        return min(confidence, 1.0)
    
    def _get_default_color_analysis(self) -> Dict:
        """Return default color analysis when errors occur."""
        return {
            'primary_color': {'name': 'unknown', 'rgb': (128, 128, 128), 'confidence': 0.1},
            'secondary_color': None,
            'tertiary_color': None,
            'all_colors': [],
            'color_harmony': {'type': 'unknown', 'score': 0.0},
            'color_temperature': {'temperature': 'neutral', 'score': 0.5},
            'color_contrast': {'contrast': 'low', 'score': 0.3},
            'saturation_analysis': {'level': 'low', 'mean': 50.0},
            'description': 'Color analysis failed',
            'confidence': 0.1,
            'analysis_method': 'default_fallback'
        }

# Global instance for easy access
advanced_color_detector = AdvancedColorDetector()

def analyze_image_colors_advanced(image_path: str, num_colors: int = 10) -> Dict:
    """Convenience function for advanced color analysis."""
    return advanced_color_detector.analyze_image_colors_advanced(image_path, num_colors)

def enhance_existing_color_detection_advanced(image_path: str, bbox: List[int] = None, object_type: str = None) -> Dict:
    """Enhanced version of existing color detection with advanced algorithms."""
    try:
        if bbox:
            # Crop image to bounding box
            image = Image.open(image_path).convert('RGB')
            x1, y1, x2, y2 = bbox
            cropped = image.crop((x1, y1, x2, y2))
            
            # Save cropped image temporarily
            temp_path = f"temp_cropped_{hash(image_path)}.jpg"
            cropped.save(temp_path)
            
            try:
                result = advanced_color_detector.analyze_image_colors_advanced(temp_path, num_colors=8)
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            result = advanced_color_detector.analyze_image_colors_advanced(image_path, num_colors=8)
        
        return result
        
    except Exception as e:
        print(f"Error in advanced color detection: {e}")
        return advanced_color_detector._get_default_color_analysis()
