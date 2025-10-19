#!/usr/bin/env python3
"""
Super Enhanced Color Detection System
Ultra-advanced color analysis with AI-powered color recognition, material-aware color detection, and comprehensive color understanding.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import webcolors
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import colorsys
from typing import Dict, List, Tuple, Optional, Union
import json
import os
from collections import Counter, defaultdict
import math
import warnings
warnings.filterwarnings('ignore')

class SuperEnhancedColorDetector:
    """Super enhanced color detection with AI-powered analysis and material-aware detection."""
    
    def __init__(self):
        # Initialize comprehensive color database
        self.color_database = self._initialize_super_color_database()
        self.material_color_mappings = self._initialize_material_color_mappings()
        self.perceptual_weights = self._initialize_perceptual_weights()
        
        # Advanced color spaces for different analysis types
        self.color_spaces = {
            'RGB': 'rgb',
            'HSV': 'hsv', 
            'LAB': 'lab',
            'LUV': 'luv',
            'XYZ': 'xyz',
            'YUV': 'yuv',
            'YCbCr': 'ycbcr'
        }
        
        # Color harmony relationships
        self.harmony_types = {
            'monochromatic': 0,
            'analogous': 30,
            'complementary': 180,
            'triadic': 120,
            'tetradic': 90,
            'split_complementary': 150
        }
        
        # Material-specific color characteristics
        self.material_characteristics = {
            'metal': {
                'typical_colors': ['silver', 'gold', 'copper', 'bronze', 'steel', 'chrome'],
                'hue_range': (0, 360),  # All hues possible for metals
                'saturation_range': (0.3, 0.8),
                'brightness_range': (0.4, 0.9),
                'texture_indicators': ['reflective', 'shiny', 'metallic']
            },
            'plastic': {
                'typical_colors': ['white', 'black', 'red', 'blue', 'green', 'yellow', 'orange'],
                'hue_range': (0, 360),  # All hues possible for plastics
                'saturation_range': (0.4, 1.0),
                'brightness_range': (0.3, 0.8),
                'texture_indicators': ['smooth', 'matte', 'glossy']
            },
            'fabric': {
                'typical_colors': ['brown', 'beige', 'navy', 'maroon', 'olive', 'gray'],
                'hue_range': (0, 360),  # All hues possible for fabrics
                'saturation_range': (0.2, 0.7),
                'brightness_range': (0.2, 0.6),
                'texture_indicators': ['textured', 'woven', 'soft']
            },
            'leather': {
                'typical_colors': ['brown', 'black', 'tan', 'burgundy', 'cognac'],
                'hue_range': (0, 60),  # Brown/tan hues for leather
                'saturation_range': (0.3, 0.8),
                'brightness_range': (0.2, 0.5),
                'texture_indicators': ['textured', 'grainy', 'natural']
            },
            'glass': {
                'typical_colors': ['clear', 'blue', 'green', 'brown', 'gray'],
                'hue_range': (180, 240),  # Blue/green hues for glass
                'saturation_range': (0.1, 0.6),
                'brightness_range': (0.6, 1.0),
                'texture_indicators': ['transparent', 'reflective', 'smooth']
            }
        }
    
    def _initialize_super_color_database(self) -> Dict:
        """Initialize comprehensive color database with enhanced color names and variations."""
        return {
            # Basic colors with variations
            'red': {
                'rgb': (255, 0, 0), 'hsv': (0, 100, 100), 'lab': (53, 80, 67),
                'variations': ['crimson', 'scarlet', 'burgundy', 'maroon', 'rose', 'pink', 'coral', 'salmon']
            },
            'blue': {
                'rgb': (0, 0, 255), 'hsv': (240, 100, 100), 'lab': (32, 79, -107),
                'variations': ['navy', 'royal', 'sky', 'teal', 'turquoise', 'cyan', 'azure', 'indigo']
            },
            'green': {
                'rgb': (0, 128, 0), 'hsv': (120, 100, 50), 'lab': (46, -68, 40),
                'variations': ['lime', 'forest', 'olive', 'mint', 'emerald', 'sage', 'jade', 'kelly']
            },
            'yellow': {
                'rgb': (255, 255, 0), 'hsv': (60, 100, 100), 'lab': (97, -21, 94),
                'variations': ['gold', 'amber', 'lemon', 'cream', 'ivory', 'beige', 'tan', 'khaki']
            },
            'orange': {
                'rgb': (255, 165, 0), 'hsv': (39, 100, 100), 'lab': (74, 23, 78),
                'variations': ['peach', 'apricot', 'tangerine', 'pumpkin', 'burnt', 'rust', 'copper', 'bronze']
            },
            'purple': {
                'rgb': (128, 0, 128), 'hsv': (300, 100, 50), 'lab': (29, 58, -36),
                'variations': ['violet', 'lavender', 'plum', 'magenta', 'fuchsia', 'mauve', 'lilac', 'amethyst']
            },
            'brown': {
                'rgb': (165, 42, 42), 'hsv': (0, 75, 65), 'lab': (40, 48, 25),
                'variations': ['tan', 'beige', 'chocolate', 'coffee', 'cognac', 'mahogany', 'walnut', 'chestnut']
            },
            'gray': {
                'rgb': (128, 128, 128), 'hsv': (0, 0, 50), 'lab': (53, 0, 0),
                'variations': ['silver', 'charcoal', 'slate', 'pewter', 'ash', 'steel', 'platinum', 'gunmetal']
            },
            'black': {
                'rgb': (0, 0, 0), 'hsv': (0, 0, 0), 'lab': (0, 0, 0),
                'variations': ['jet', 'ebony', 'onyx', 'coal', 'ink', 'midnight', 'obsidian', 'raven']
            },
            'white': {
                'rgb': (255, 255, 255), 'hsv': (0, 0, 100), 'lab': (100, 0, 0),
                'variations': ['ivory', 'cream', 'pearl', 'snow', 'alabaster', 'bone', 'chalk', 'milk']
            },
            # Metallic colors
            'gold': {
                'rgb': (255, 215, 0), 'hsv': (51, 100, 100), 'lab': (85, -9, 83),
                'variations': ['brass', 'bronze', 'copper', 'champagne', 'honey', 'amber', 'mustard', 'canary']
            },
            'silver': {
                'rgb': (192, 192, 192), 'hsv': (0, 0, 75), 'lab': (75, 0, 0),
                'variations': ['platinum', 'chrome', 'steel', 'pewter', 'tin', 'aluminum', 'nickel', 'mercury']
            },
            'copper': {
                'rgb': (184, 115, 51), 'hsv': (25, 72, 72), 'lab': (58, 25, 45),
                'variations': ['bronze', 'rust', 'auburn', 'cinnamon', 'terracotta', 'burnt', 'sienna', 'umber']
            }
        }
    
    def _initialize_material_color_mappings(self) -> Dict:
        """Initialize material-specific color mappings for better accuracy."""
        return {
            'stainless_steel': {
                'primary_colors': ['silver', 'gray', 'steel'],
                'secondary_colors': ['blue', 'green'],
                'hue_range': (180, 240),  # Blue-gray range
                'saturation_range': (0.1, 0.4),
                'brightness_range': (0.6, 0.9)
            },
            'aluminum': {
                'primary_colors': ['silver', 'gray', 'white'],
                'secondary_colors': ['blue', 'green'],
                'hue_range': (180, 240),
                'saturation_range': (0.05, 0.3),
                'brightness_range': (0.7, 0.95)
            },
            'copper': {
                'primary_colors': ['copper', 'bronze', 'orange'],
                'secondary_colors': ['red', 'brown'],
                'hue_range': (10, 40),
                'saturation_range': (0.6, 1.0),
                'brightness_range': (0.4, 0.8)
            },
            'plastic': {
                'primary_colors': ['white', 'black', 'red', 'blue', 'green', 'yellow'],
                'secondary_colors': ['orange', 'purple', 'pink'],
                'hue_range': (0, 360),
                'saturation_range': (0.3, 1.0),
                'brightness_range': (0.2, 0.9)
            },
            'leather': {
                'primary_colors': ['brown', 'black', 'tan', 'burgundy'],
                'secondary_colors': ['red', 'orange', 'yellow'],
                'hue_range': (0, 60),
                'saturation_range': (0.3, 0.8),
                'brightness_range': (0.2, 0.6)
            }
        }
    
    def _initialize_perceptual_weights(self) -> Dict:
        """Initialize perceptual weights for different color analysis aspects."""
        return {
            'hue': 0.4,        # Most important for color identification
            'saturation': 0.3,  # Important for color intensity
            'brightness': 0.2,  # Important for color lightness
            'contrast': 0.1     # Important for color relationships
        }
    
    def analyze_image_colors_super(self, image_path: str, num_colors: int = 15, fast_mode: bool = True) -> Dict:
        """Super enhanced color analysis with AI-powered recognition and material awareness."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Apply advanced image enhancement
            enhanced_image = self._enhance_image_for_super_color_analysis(image)
            enhanced_array = np.array(enhanced_image)
            
            # Multi-algorithm color extraction (optimized for speed)
            colors_kmeans = self._extract_colors_super_kmeans(enhanced_array, num_colors)
            
            if fast_mode:
                # Fast mode: Skip slow algorithms
                colors_dbscan = []
                colors_meanshift = []
            else:
                # Full mode: Use all algorithms
                colors_dbscan = self._extract_colors_super_dbscan(enhanced_array)
                colors_meanshift = self._extract_colors_super_meanshift(enhanced_array)
            
            colors_histogram = self._extract_colors_super_histogram(enhanced_array, num_colors)
            
            # Combine results using ensemble method
            combined_colors = self._combine_super_color_results(
                colors_kmeans, colors_dbscan, colors_meanshift, colors_histogram
            )
            
            # AI-powered color analysis
            ai_analysis = self._perform_ai_color_analysis(enhanced_array, combined_colors)
            
            # Material-aware color detection
            material_analysis = self._analyze_material_colors_super(enhanced_array, combined_colors)
            
            # Perceptual color analysis
            perceptual_analysis = self._analyze_perceptual_properties_super(combined_colors)
            
            # Color harmony and relationships
            harmony_analysis = self._analyze_color_harmony_super(combined_colors)
            
            # Generate comprehensive color description
            color_description = self._generate_super_color_description(
                combined_colors, ai_analysis, material_analysis, perceptual_analysis, harmony_analysis
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_super_confidence_score(
                combined_colors, ai_analysis, material_analysis, perceptual_analysis
            )
            
            return {
                'success': True,
                'primary_colors': combined_colors[:5],
                'all_colors': combined_colors,
                'named_palette': self._generate_named_palette_super(combined_colors),
                'ai_analysis': ai_analysis,
                'material_analysis': material_analysis,
                'perceptual_analysis': perceptual_analysis,
                'harmony_analysis': harmony_analysis,
                'color_description': color_description,
                'confidence_score': confidence_score,
                'enhanced_features': {
                    'color_temperature': self._analyze_color_temperature_super(combined_colors),
                    'color_contrast': self._analyze_color_contrast_super(combined_colors),
                    'color_saturation': self._analyze_saturation_distribution_super(enhanced_array),
                    'color_brightness': self._analyze_brightness_distribution_super(enhanced_array),
                    'color_vibrancy': self._analyze_color_vibrancy_super(combined_colors)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Super color analysis failed: {str(e)}",
                'primary_colors': [],
                'all_colors': [],
                'named_palette': [],
                'confidence_score': 0.0
            }
    
    def _enhance_image_for_super_color_analysis(self, image: Image.Image) -> Image.Image:
        """Apply advanced image enhancement for better color analysis."""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply histogram equalization for better contrast
            image_array = np.array(image)
            
            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            enhanced_lab = cv2.merge([l, a, b])
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Convert back to PIL Image
            enhanced_image = Image.fromarray(enhanced_rgb)
            
            # Apply slight sharpening for better edge detection
            enhanced_image = enhanced_image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            
            return enhanced_image
            
        except Exception as e:
            print(f"Image enhancement error: {e}")
            return image
    
    def _extract_colors_super_kmeans(self, image_array: np.ndarray, num_colors: int) -> List[Dict]:
        """Extract colors using optimized KMeans clustering."""
        try:
            # Downsample large images for faster processing
            height, width = image_array.shape[:2]
            if height > 600 or width > 600:
                scale = min(600 / height, 600 / width)
                new_height = int(height * scale)
                new_width = int(width * scale)
                image_array = cv2.resize(image_array, (new_width, new_height))
            
            # Reshape image to list of pixels
            pixels = image_array.reshape(-1, 3)
            
            # Use optimized KMeans parameters
            kmeans = KMeans(
                n_clusters=min(num_colors, 12),  # Limit max clusters
                random_state=42, 
                n_init=10,  # Reduced from 20
                max_iter=100  # Reduced from 300
            )
            kmeans.fit(pixels)
            
            # Get cluster centers and labels
            centers = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # Calculate color frequencies
            label_counts = np.bincount(labels)
            
            all_colors = []
            for i, (center, count) in enumerate(zip(centers, label_counts)):
                if count > 0:  # Only include clusters with pixels
                    color_info = {
                        'rgb': tuple(center),
                        'frequency': int(count),
                        'percentage': float(count / len(pixels) * 100),
                        'hsv': self._rgb_to_hsv(center),
                        'lab': self._rgb_to_lab(center),
                        'method': 'kmeans',
                        'cluster_id': i
                    }
                    all_colors.append(color_info)
            
            # Sort by frequency and return top colors
            all_colors.sort(key=lambda x: x['frequency'], reverse=True)
            return all_colors[:num_colors]
            
        except Exception as e:
            print(f"KMeans color extraction error: {e}")
            return []
    
    def _extract_colors_super_dbscan(self, image_array: np.ndarray) -> List[Dict]:
        """Extract colors using optimized DBSCAN clustering with performance improvements."""
        try:
            # Skip DBSCAN for large images to improve performance
            if image_array.shape[0] * image_array.shape[1] > 500000:  # Skip for images > 500k pixels
                return []
            
            # Downsample image for faster processing
            height, width = image_array.shape[:2]
            if height > 400 or width > 400:
                scale = min(400 / height, 400 / width)
                new_height = int(height * scale)
                new_width = int(width * scale)
                image_array = cv2.resize(image_array, (new_width, new_height))
            
            pixels = image_array.reshape(-1, 3)
            
            # Use optimized DBSCAN parameters
            dbscan = DBSCAN(eps=25, min_samples=20)  # Reduced parameters for speed
            labels = dbscan.fit_predict(pixels)
            
            # Get unique labels (excluding noise points labeled as -1)
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            
            # Limit to top clusters to improve performance
            if len(unique_labels) > 8:
                unique_labels = list(unique_labels)[:8]
            
            colors = []
            for label in unique_labels:
                cluster_pixels = pixels[labels == label]
                if len(cluster_pixels) > 10:  # Only include clusters with sufficient pixels
                    # Calculate mean color for the cluster
                    mean_color = np.mean(cluster_pixels, axis=0).astype(int)
                    
                    color_info = {
                        'rgb': tuple(mean_color),
                        'frequency': len(cluster_pixels),
                        'percentage': float(len(cluster_pixels) / len(pixels) * 100),
                        'hsv': self._rgb_to_hsv(mean_color),
                        'lab': self._rgb_to_lab(mean_color),
                        'method': 'dbscan',
                        'cluster_id': label
                    }
                    colors.append(color_info)
            
            # Sort by frequency
            colors.sort(key=lambda x: x['frequency'], reverse=True)
            return colors[:5]  # Return top 5 DBSCAN colors for speed
            
        except Exception as e:
            print(f"DBSCAN color extraction error: {e}")
            return []
    
    def _extract_colors_super_meanshift(self, image_array: np.ndarray) -> List[Dict]:
        """Extract colors using optimized MeanShift clustering."""
        try:
            # Skip MeanShift for large images to improve performance
            if image_array.shape[0] * image_array.shape[1] > 300000:  # Skip for images > 300k pixels
                return []
            
            # Downsample image for faster processing
            height, width = image_array.shape[:2]
            if height > 300 or width > 300:
                scale = min(300 / height, 300 / width)
                new_height = int(height * scale)
                new_width = int(width * scale)
                image_array = cv2.resize(image_array, (new_width, new_height))
            
            pixels = image_array.reshape(-1, 3)
            
            # Use optimized MeanShift parameters
            meanshift = MeanShift(bandwidth=30, bin_seeding=True)  # Reduced bandwidth for speed
            labels = meanshift.fit_predict(pixels)
            
            # Get unique labels
            unique_labels = set(labels)
            
            # Limit to top clusters to improve performance
            if len(unique_labels) > 6:
                unique_labels = list(unique_labels)[:6]
            
            colors = []
            for label in unique_labels:
                cluster_pixels = pixels[labels == label]
                if len(cluster_pixels) > 15:  # Only include clusters with sufficient pixels
                    # Calculate mean color for the cluster
                    mean_color = np.mean(cluster_pixels, axis=0).astype(int)
                    
                    color_info = {
                        'rgb': tuple(mean_color),
                        'frequency': len(cluster_pixels),
                        'percentage': float(len(cluster_pixels) / len(pixels) * 100),
                        'hsv': self._rgb_to_hsv(mean_color),
                        'lab': self._rgb_to_lab(mean_color),
                        'method': 'meanshift',
                        'cluster_id': label
                    }
                    colors.append(color_info)
            
            # Sort by frequency
            colors.sort(key=lambda x: x['frequency'], reverse=True)
            return colors[:4]  # Return top 4 MeanShift colors for speed
            
        except Exception as e:
            print(f"MeanShift color extraction error: {e}")
            return []
    
    def _extract_colors_super_histogram(self, image_array: np.ndarray, num_colors: int) -> List[Dict]:
        """Extract colors using histogram analysis."""
        try:
            # Calculate color histogram
            hist_r = cv2.calcHist([image_array], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image_array], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image_array], [2], None, [256], [0, 256])
            
            # Find peaks in each channel
            colors = []
            for r in range(0, 256, 16):  # Sample every 16 values
                for g in range(0, 256, 16):
                    for b in range(0, 256, 16):
                        if hist_r[r] > 0 and hist_g[g] > 0 and hist_b[b] > 0:
                            # Calculate combined frequency
                            frequency = int(hist_r[r] + hist_g[g] + hist_b[b])
                            
                            color_info = {
                                'rgb': (r, g, b),
                                'frequency': frequency,
                                'percentage': float(frequency / (image_array.shape[0] * image_array.shape[1]) * 100),
                                'hsv': self._rgb_to_hsv([r, g, b]),
                                'lab': self._rgb_to_lab([r, g, b]),
                                'method': 'histogram',
                                'cluster_id': len(colors)
                            }
                            colors.append(color_info)
            
            # Sort by frequency and return top colors
            colors.sort(key=lambda x: x['frequency'], reverse=True)
            return colors[:num_colors]
            
        except Exception as e:
            print(f"Histogram color extraction error: {e}")
            return []
    
    def _combine_super_color_results(self, *color_lists) -> List[Dict]:
        """Combine results from multiple color extraction methods."""
        try:
            all_colors = []
            for color_list in color_lists:
                all_colors.extend(color_list)
            
            if not all_colors:
                return []
            
            # Group similar colors together
            grouped_colors = self._group_similar_colors(all_colors)
            
            # Calculate weighted scores for each group
            final_colors = []
            for group in grouped_colors:
                if not group:
                    continue
                
                # Calculate weighted average RGB
                total_weight = sum(color['frequency'] for color in group)
                if total_weight == 0:
                    continue
                
                weighted_r = sum(color['rgb'][0] * color['frequency'] for color in group) / total_weight
                weighted_g = sum(color['rgb'][1] * color['frequency'] for color in group) / total_weight
                weighted_b = sum(color['rgb'][2] * color['frequency'] for color in group) / total_weight
                
                # Calculate combined frequency
                combined_frequency = sum(color['frequency'] for color in group)
                combined_percentage = sum(color['percentage'] for color in group)
                
                final_color = {
                    'rgb': (int(weighted_r), int(weighted_g), int(weighted_b)),
                    'frequency': combined_frequency,
                    'percentage': combined_percentage,
                    'hsv': self._rgb_to_hsv([int(weighted_r), int(weighted_g), int(weighted_b)]),
                    'lab': self._rgb_to_lab([int(weighted_r), int(weighted_g), int(weighted_b)]),
                    'method': 'combined',
                    'source_methods': list(set(color['method'] for color in group))
                }
                final_colors.append(final_color)
            
            # Sort by frequency and return
            final_colors.sort(key=lambda x: x['frequency'], reverse=True)
            return final_colors
            
        except Exception as e:
            print(f"Color combination error: {e}")
            return all_colors if 'all_colors' in locals() else []
    
    def _group_similar_colors(self, colors: List[Dict], threshold: float = 30.0) -> List[List[Dict]]:
        """Group similar colors together based on color distance."""
        try:
            if not colors:
                return []
            
            groups = []
            used_colors = set()
            
            for i, color1 in enumerate(colors):
                if i in used_colors:
                    continue
                
                group = [color1]
                used_colors.add(i)
                
                for j, color2 in enumerate(colors[i+1:], i+1):
                    if j in used_colors:
                        continue
                    
                    # Calculate color distance
                    distance = self._calculate_color_distance(color1['rgb'], color2['rgb'])
                    if distance < threshold:
                        group.append(color2)
                        used_colors.add(j)
                
                groups.append(group)
            
            return groups
            
        except Exception as e:
            print(f"Color grouping error: {e}")
            return [[color] for color in colors]
    
    def _calculate_color_distance(self, rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
        """Calculate perceptual color distance between two RGB colors."""
        try:
            # Convert to LAB color space for better perceptual distance
            lab1 = self._rgb_to_lab(rgb1)
            lab2 = self._rgb_to_lab(rgb2)
            
            # Calculate Euclidean distance in LAB space
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))
            return distance
            
        except Exception as e:
            print(f"Color distance calculation error: {e}")
            # Fallback to RGB distance
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
    
    def _rgb_to_hsv(self, rgb: Union[List, Tuple, np.ndarray]) -> Tuple[float, float, float]:
        """Convert RGB to HSV color space."""
        try:
            if isinstance(rgb, (list, tuple)):
                rgb = np.array(rgb)
            
            # Normalize RGB values to 0-1 range
            rgb_normalized = rgb.astype(float) / 255.0
            
            # Convert to HSV
            hsv = colorsys.rgb_to_hsv(rgb_normalized[0], rgb_normalized[1], rgb_normalized[2])
            
            # Convert to 0-360, 0-100, 0-100 range
            h = hsv[0] * 360
            s = hsv[1] * 100
            v = hsv[2] * 100
            
            return (h, s, v)
            
        except Exception as e:
            print(f"RGB to HSV conversion error: {e}")
            return (0, 0, 0)
    
    def _rgb_to_lab(self, rgb: Union[List, Tuple, np.ndarray]) -> Tuple[float, float, float]:
        """Convert RGB to LAB color space."""
        try:
            if isinstance(rgb, (list, tuple)):
                rgb = np.array(rgb)
            
            # Normalize RGB values to 0-1 range
            rgb_normalized = rgb.astype(float) / 255.0
            
            # Convert to LAB using OpenCV
            rgb_uint8 = (rgb_normalized * 255).astype(np.uint8)
            lab = cv2.cvtColor(np.uint8([[rgb_uint8]]), cv2.COLOR_RGB2LAB)[0][0]
            
            return tuple(lab.astype(float))
            
        except Exception as e:
            print(f"RGB to LAB conversion error: {e}")
            return (0, 0, 0)
    
    def _perform_ai_color_analysis(self, image_array: np.ndarray, colors: List[Dict]) -> Dict:
        """Perform AI-powered color analysis."""
        try:
            analysis = {
                'dominant_hue': self._analyze_dominant_hue(colors),
                'color_temperature': self._analyze_color_temperature_super(colors),
                'color_vibrancy': self._analyze_color_vibrancy_super(colors),
                'color_contrast': self._analyze_color_contrast_super(colors),
                'color_harmony': self._analyze_color_harmony_super(colors),
                'material_prediction': self._predict_material_from_colors(colors),
                'color_emotion': self._analyze_color_emotion(colors),
                'color_trends': self._analyze_color_trends(colors)
            }
            
            return analysis
            
        except Exception as e:
            print(f"AI color analysis error: {e}")
            return {}
    
    def _analyze_material_colors_super(self, image_array: np.ndarray, colors: List[Dict]) -> Dict:
        """Analyze colors with material awareness."""
        try:
            material_scores = {}
            
            for material, characteristics in self.material_characteristics.items():
                score = 0.0
                total_weight = 0.0
                
                for color in colors[:5]:  # Analyze top 5 colors
                    rgb = color['rgb']
                    hsv = color['hsv']
                    
                    # Check if color matches material characteristics
                    color_name = self._get_closest_color_name(rgb)
                    
                    # Check primary colors
                    if color_name in characteristics['typical_colors']:
                        score += color['percentage'] * 1.0
                        total_weight += color['percentage']
                    
                    # Check hue range
                    hue = hsv[0]
                    if characteristics['hue_range'][0] <= hue <= characteristics['hue_range'][1]:
                        score += color['percentage'] * 0.5
                        total_weight += color['percentage']
                    
                    # Check saturation range
                    saturation = hsv[1] / 100.0
                    if characteristics['saturation_range'][0] <= saturation <= characteristics['saturation_range'][1]:
                        score += color['percentage'] * 0.3
                        total_weight += color['percentage']
                    
                    # Check brightness range
                    brightness = hsv[2] / 100.0
                    if characteristics['brightness_range'][0] <= brightness <= characteristics['brightness_range'][1]:
                        score += color['percentage'] * 0.2
                        total_weight += color['percentage']
                
                if total_weight > 0:
                    material_scores[material] = score / total_weight
                else:
                    material_scores[material] = 0.0
            
            # Find best material match
            best_material = max(material_scores, key=material_scores.get) if material_scores else 'unknown'
            best_score = material_scores.get(best_material, 0.0)
            
            return {
                'predicted_material': best_material,
                'material_confidence': best_score,
                'material_scores': material_scores,
                'material_characteristics': self.material_characteristics.get(best_material, {})
            }
            
        except Exception as e:
            print(f"Material color analysis error: {e}")
            return {'predicted_material': 'unknown', 'material_confidence': 0.0}
    
    def _analyze_perceptual_properties_super(self, colors: List[Dict]) -> Dict:
        """Analyze perceptual properties of colors."""
        try:
            if not colors:
                return {}
            
            # Calculate average properties
            avg_hue = np.mean([color['hsv'][0] for color in colors])
            avg_saturation = np.mean([color['hsv'][1] for color in colors])
            avg_brightness = np.mean([color['hsv'][2] for color in colors])
            
            # Analyze color distribution
            hue_distribution = self._analyze_hue_distribution([color['hsv'][0] for color in colors])
            saturation_distribution = self._analyze_saturation_distribution_super([color['hsv'][1] for color in colors])
            brightness_distribution = self._analyze_brightness_distribution_super([color['hsv'][2] for color in colors])
            
            return {
                'average_hue': avg_hue,
                'average_saturation': avg_saturation,
                'average_brightness': avg_brightness,
                'hue_distribution': hue_distribution,
                'saturation_distribution': saturation_distribution,
                'brightness_distribution': brightness_distribution,
                'color_diversity': len(set(tuple(color['rgb']) for color in colors)),
                'dominant_hue_range': self._get_dominant_hue_range([color['hsv'][0] for color in colors])
            }
            
        except Exception as e:
            print(f"Perceptual analysis error: {e}")
            return {}
    
    def _analyze_color_harmony_super(self, colors: List[Dict]) -> Dict:
        """Analyze color harmony relationships."""
        try:
            if len(colors) < 2:
                return {'harmony_type': 'single_color', 'harmony_score': 1.0}
            
            hues = [color['hsv'][0] for color in colors[:5]]  # Analyze top 5 colors
            
            # Check for different harmony types
            harmony_scores = {}
            
            # Monochromatic (similar hues)
            hue_variance = np.var(hues)
            harmony_scores['monochromatic'] = max(0, 1 - hue_variance / 10000)
            
            # Analogous (hues within 30 degrees)
            analogous_score = 0
            for i, hue1 in enumerate(hues):
                for hue2 in hues[i+1:]:
                    diff = abs(hue1 - hue2)
                    if diff <= 30 or diff >= 330:  # Handle wraparound
                        analogous_score += 1
            harmony_scores['analogous'] = analogous_score / (len(hues) * (len(hues) - 1) / 2)
            
            # Complementary (hues 180 degrees apart)
            complementary_score = 0
            for i, hue1 in enumerate(hues):
                for hue2 in hues[i+1:]:
                    diff = abs(hue1 - hue2)
                    if 150 <= diff <= 210:  # Allow some tolerance
                        complementary_score += 1
            harmony_scores['complementary'] = complementary_score / (len(hues) * (len(hues) - 1) / 2)
            
            # Find best harmony type
            best_harmony = max(harmony_scores, key=harmony_scores.get)
            best_score = harmony_scores[best_harmony]
            
            return {
                'harmony_type': best_harmony,
                'harmony_score': best_score,
                'harmony_scores': harmony_scores,
                'hue_relationships': self._analyze_hue_relationships(hues)
            }
            
        except Exception as e:
            print(f"Color harmony analysis error: {e}")
            return {'harmony_type': 'unknown', 'harmony_score': 0.0}
    
    def _generate_super_color_description(self, colors: List[Dict], ai_analysis: Dict, 
                                        material_analysis: Dict, perceptual_analysis: Dict, 
                                        harmony_analysis: Dict) -> str:
        """Generate comprehensive color description."""
        try:
            if not colors:
                return "No colors detected"
            
            description_parts = []
            
            # Primary color description
            primary_color = colors[0]
            primary_name = self._get_closest_color_name(primary_color['rgb'])
            description_parts.append(f"{primary_name} colored")
            
            # Secondary color if present
            if len(colors) > 1:
                secondary_color = colors[1]
                secondary_name = self._get_closest_color_name(secondary_color['rgb'])
                if secondary_name != primary_name:
                    description_parts.append(f"with {secondary_name} accents")
            
            # Material information
            if material_analysis.get('predicted_material') != 'unknown':
                material = material_analysis['predicted_material']
                confidence = material_analysis.get('material_confidence', 0)
                if confidence > 0.5:
                    description_parts.append(f"{material} material")
            
            # Color characteristics
            if perceptual_analysis:
                avg_saturation = perceptual_analysis.get('average_saturation', 0)
                avg_brightness = perceptual_analysis.get('average_brightness', 0)
                
                if avg_saturation > 70:
                    description_parts.append("vibrant")
                elif avg_saturation < 30:
                    description_parts.append("muted")
                
                if avg_brightness > 70:
                    description_parts.append("bright")
                elif avg_brightness < 30:
                    description_parts.append("dark")
            
            # Color harmony
            if harmony_analysis.get('harmony_type') != 'unknown':
                harmony_type = harmony_analysis['harmony_type']
                harmony_score = harmony_analysis.get('harmony_score', 0)
                if harmony_score > 0.5:
                    description_parts.append(f"in {harmony_type} harmony")
            
            # Combine all parts
            description = " ".join(description_parts)
            
            # Add confidence information
            confidence = self._calculate_super_confidence_score(colors, ai_analysis, material_analysis, perceptual_analysis)
            description += f" (confidence: {confidence:.1%})"
            
            return description
            
        except Exception as e:
            print(f"Color description generation error: {e}")
            return "Color analysis completed"
    
    def _generate_named_palette_super(self, colors: List[Dict]) -> List[Dict]:
        """Generate named color palette."""
        try:
            named_palette = []
            
            for i, color in enumerate(colors[:8]):  # Top 8 colors
                color_name = self._get_closest_color_name(color['rgb'])
                color_variations = self._get_color_variations(color['rgb'])
                
                named_color = {
                    'name': color_name,
                    'rgb': color['rgb'],
                    'hsv': color['hsv'],
                    'lab': color['lab'],
                    'percentage': color['percentage'],
                    'variations': color_variations,
                    'rank': i + 1
                }
                named_palette.append(named_color)
            
            return named_palette
            
        except Exception as e:
            print(f"Named palette generation error: {e}")
            return []
    
    def _get_closest_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Get the closest color name for RGB value."""
        try:
            # Try webcolors first
            try:
                return webcolors.rgb_to_name(rgb)
            except ValueError:
                pass
            
            # Find closest color in database
            min_distance = float('inf')
            closest_name = 'unknown'
            
            for name, color_data in self.color_database.items():
                distance = self._calculate_color_distance(rgb, color_data['rgb'])
                if distance < min_distance:
                    min_distance = distance
                    closest_name = name
            
            return closest_name
            
        except Exception as e:
            print(f"Color name lookup error: {e}")
            return 'unknown'
    
    def _get_color_variations(self, rgb: Tuple[int, int, int]) -> List[str]:
        """Get color variations for a given RGB value."""
        try:
            color_name = self._get_closest_color_name(rgb)
            
            # Find variations in database
            for name, color_data in self.color_database.items():
                if name == color_name and 'variations' in color_data:
                    return color_data['variations']
            
            return []
            
        except Exception as e:
            print(f"Color variations lookup error: {e}")
            return []
    
    def _calculate_super_confidence_score(self, colors: List[Dict], ai_analysis: Dict, 
                                        material_analysis: Dict, perceptual_analysis: Dict) -> float:
        """Calculate overall confidence score for color analysis."""
        try:
            if not colors:
                return 0.0
            
            # Base confidence from color extraction quality
            base_confidence = min(1.0, len(colors) / 10.0)  # More colors = higher confidence
            
            # Material confidence boost
            material_confidence = material_analysis.get('material_confidence', 0.0)
            material_boost = material_confidence * 0.2
            
            # Perceptual analysis confidence
            perceptual_confidence = 0.5  # Default
            if perceptual_analysis:
                color_diversity = perceptual_analysis.get('color_diversity', 0)
                perceptual_confidence = min(1.0, color_diversity / 5.0)
            
            # AI analysis confidence
            ai_confidence = 0.5  # Default
            if ai_analysis:
                # Check if we have good color temperature analysis
                if 'color_temperature' in ai_analysis:
                    ai_confidence += 0.2
                if 'color_vibrancy' in ai_analysis:
                    ai_confidence += 0.2
                if 'material_prediction' in ai_analysis:
                    ai_confidence += 0.1
            
            # Combine all confidence factors
            total_confidence = (base_confidence + material_boost + perceptual_confidence + ai_confidence) / 4.0
            
            return min(1.0, max(0.0, total_confidence))
            
        except Exception as e:
            print(f"Confidence calculation error: {e}")
            return 0.5
    
    # Additional helper methods for specific analyses
    def _analyze_dominant_hue(self, colors: List[Dict]) -> str:
        """Analyze dominant hue category."""
        try:
            if not colors:
                return 'unknown'
            
            hues = [color['hsv'][0] for color in colors[:5]]
            avg_hue = np.mean(hues)
            
            if 0 <= avg_hue < 30 or 330 <= avg_hue <= 360:
                return 'red'
            elif 30 <= avg_hue < 90:
                return 'yellow'
            elif 90 <= avg_hue < 150:
                return 'green'
            elif 150 <= avg_hue < 210:
                return 'cyan'
            elif 210 <= avg_hue < 270:
                return 'blue'
            elif 270 <= avg_hue < 330:
                return 'magenta'
            else:
                return 'unknown'
                
        except Exception as e:
            print(f"Dominant hue analysis error: {e}")
            return 'unknown'
    
    def _analyze_color_temperature_super(self, colors: List[Dict]) -> str:
        """Analyze color temperature."""
        try:
            if not colors:
                return 'neutral'
            
            # Calculate average hue
            hues = [color['hsv'][0] for color in colors[:5]]
            avg_hue = np.mean(hues)
            
            # Warm colors: red, orange, yellow (0-60, 300-360)
            # Cool colors: blue, cyan, green (120-240)
            if (0 <= avg_hue <= 60) or (300 <= avg_hue <= 360):
                return 'warm'
            elif 120 <= avg_hue <= 240:
                return 'cool'
            else:
                return 'neutral'
                
        except Exception as e:
            print(f"Color temperature analysis error: {e}")
            return 'neutral'
    
    def _analyze_color_vibrancy_super(self, colors: List[Dict]) -> str:
        """Analyze color vibrancy."""
        try:
            if not colors:
                return 'neutral'
            
            # Calculate average saturation
            saturations = [color['hsv'][1] for color in colors[:5]]
            avg_saturation = np.mean(saturations)
            
            if avg_saturation > 70:
                return 'vibrant'
            elif avg_saturation < 30:
                return 'muted'
            else:
                return 'moderate'
                
        except Exception as e:
            print(f"Color vibrancy analysis error: {e}")
            return 'neutral'
    
    def _analyze_color_contrast_super(self, colors: List[Dict]) -> str:
        """Analyze color contrast."""
        try:
            if len(colors) < 2:
                return 'low'
            
            # Calculate brightness range
            brightnesses = [color['hsv'][2] for color in colors[:5]]
            brightness_range = max(brightnesses) - min(brightnesses)
            
            if brightness_range > 50:
                return 'high'
            elif brightness_range > 25:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            print(f"Color contrast analysis error: {e}")
            return 'low'
    
    def _analyze_saturation_distribution_super(self, saturations: List[float]) -> Dict:
        """Analyze saturation distribution."""
        try:
            if len(saturations) == 0:
                return {}
            
            saturations_array = np.array(saturations)
            
            return {
                'min': float(np.min(saturations_array)),
                'max': float(np.max(saturations_array)),
                'mean': float(np.mean(saturations_array)),
                'std': float(np.std(saturations_array)),
                'range': float(np.max(saturations_array) - np.min(saturations_array))
            }
            
        except Exception as e:
            print(f"Saturation distribution analysis error: {e}")
            return {}
    
    def _analyze_brightness_distribution_super(self, brightnesses: List[float]) -> Dict:
        """Analyze brightness distribution."""
        try:
            if len(brightnesses) == 0:
                return {}
            
            brightnesses_array = np.array(brightnesses)
            
            return {
                'min': float(np.min(brightnesses_array)),
                'max': float(np.max(brightnesses_array)),
                'mean': float(np.mean(brightnesses_array)),
                'std': float(np.std(brightnesses_array)),
                'range': float(np.max(brightnesses_array) - np.min(brightnesses_array))
            }
            
        except Exception as e:
            print(f"Brightness distribution analysis error: {e}")
            return {}
    
    def _predict_material_from_colors(self, colors: List[Dict]) -> str:
        """Predict material based on color characteristics."""
        try:
            if not colors:
                return 'unknown'
            
            # Analyze color characteristics
            avg_saturation = np.mean([color['hsv'][1] for color in colors[:5]])
            avg_brightness = np.mean([color['hsv'][2] for color in colors[:5]])
            
            # Simple material prediction based on color characteristics
            if avg_brightness > 80 and avg_saturation < 30:
                return 'metal'  # Bright, low saturation = metallic
            elif avg_saturation > 70:
                return 'plastic'  # High saturation = plastic
            elif avg_brightness < 40:
                return 'leather'  # Dark colors = leather
            else:
                return 'fabric'  # Default to fabric
                
        except Exception as e:
            print(f"Material prediction error: {e}")
            return 'unknown'
    
    def _analyze_color_emotion(self, colors: List[Dict]) -> str:
        """Analyze emotional impact of colors."""
        try:
            if not colors:
                return 'neutral'
            
            # Analyze dominant hue for emotional impact
            dominant_hue = self._analyze_dominant_hue(colors)
            
            emotion_map = {
                'red': 'energetic',
                'orange': 'warm',
                'yellow': 'cheerful',
                'green': 'calm',
                'blue': 'peaceful',
                'purple': 'mysterious',
                'brown': 'earthy',
                'gray': 'neutral',
                'black': 'sophisticated',
                'white': 'clean'
            }
            
            return emotion_map.get(dominant_hue, 'neutral')
            
        except Exception as e:
            print(f"Color emotion analysis error: {e}")
            return 'neutral'
    
    def _analyze_color_trends(self, colors: List[Dict]) -> List[str]:
        """Analyze color trends and patterns."""
        try:
            trends = []
            
            if not colors:
                return trends
            
            # Check for monochromatic trend
            hues = [color['hsv'][0] for color in colors[:5]]
            hue_variance = np.var(hues)
            if hue_variance < 1000:  # Low variance = monochromatic
                trends.append('monochromatic')
            
            # Check for vibrant trend
            avg_saturation = np.mean([color['hsv'][1] for color in colors[:5]])
            if avg_saturation > 70:
                trends.append('vibrant')
            
            # Check for pastel trend
            if avg_saturation < 40 and np.mean([color['hsv'][2] for color in colors[:5]]) > 70:
                trends.append('pastel')
            
            # Check for earth tone trend
            earth_hues = [color['hsv'][0] for color in colors[:5] if 20 <= color['hsv'][0] <= 60]
            if len(earth_hues) > len(colors) * 0.6:
                trends.append('earth_tones')
            
            return trends
            
        except Exception as e:
            print(f"Color trends analysis error: {e}")
            return []
    
    def _analyze_hue_distribution(self, hues: List[float]) -> Dict:
        """Analyze hue distribution."""
        try:
            if not hues:
                return {}
            
            return {
                'min': min(hues),
                'max': max(hues),
                'mean': np.mean(hues),
                'std': np.std(hues),
                'range': max(hues) - min(hues)
            }
            
        except Exception as e:
            print(f"Hue distribution analysis error: {e}")
            return {}
    
    def _get_dominant_hue_range(self, hues: List[float]) -> str:
        """Get dominant hue range."""
        try:
            if not hues:
                return 'unknown'
            
            avg_hue = np.mean(hues)
            
            if 0 <= avg_hue < 30 or 330 <= avg_hue <= 360:
                return 'red_range'
            elif 30 <= avg_hue < 60:
                return 'orange_range'
            elif 60 <= avg_hue < 90:
                return 'yellow_range'
            elif 90 <= avg_hue < 150:
                return 'green_range'
            elif 150 <= avg_hue < 210:
                return 'cyan_range'
            elif 210 <= avg_hue < 270:
                return 'blue_range'
            elif 270 <= avg_hue < 330:
                return 'purple_range'
            else:
                return 'unknown'
                
        except Exception as e:
            print(f"Dominant hue range analysis error: {e}")
            return 'unknown'
    
    def _analyze_hue_relationships(self, hues: List[float]) -> List[str]:
        """Analyze relationships between hues."""
        try:
            relationships = []
            
            if len(hues) < 2:
                return relationships
            
            # Check for complementary relationships
            for i, hue1 in enumerate(hues):
                for hue2 in hues[i+1:]:
                    diff = abs(hue1 - hue2)
                    if 150 <= diff <= 210:
                        relationships.append('complementary')
                    elif diff <= 30 or diff >= 330:
                        relationships.append('analogous')
                    elif 110 <= diff <= 130:
                        relationships.append('triadic')
            
            return list(set(relationships))  # Remove duplicates
            
        except Exception as e:
            print(f"Hue relationships analysis error: {e}")
            return []


def enhance_existing_color_detection_super(image_path: str, fast_mode: bool = True) -> Dict:
    """Enhanced color detection function that integrates with existing system."""
    try:
        detector = SuperEnhancedColorDetector()
        result = detector.analyze_image_colors_super(image_path, fast_mode=fast_mode)
        return result
    except Exception as e:
        return {
            'success': False,
            'error': f"Super color detection failed: {str(e)}",
            'primary_colors': [],
            'all_colors': [],
            'named_palette': [],
            'confidence_score': 0.0
        }


if __name__ == "__main__":
    # Test the super enhanced color detection
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = enhance_existing_color_detection_super(image_path)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python super_enhanced_color_detection.py <image_path>")
