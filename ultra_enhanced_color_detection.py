#!/usr/bin/env python3
"""
Ultra Enhanced Color Detection System
Advanced color analysis with machine learning, perceptual color matching, and comprehensive color understanding.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
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

class UltraEnhancedColorDetector:
    """Ultra-enhanced color detection with machine learning and perceptual analysis."""
    
    def __init__(self):
        # Initialize comprehensive color database
        self.color_database = self._initialize_ultra_color_database()
        self.perceptual_weights = self._initialize_perceptual_weights()
        self.ml_models = {}
        
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
            'split_complementary': 150,
            'double_split_complementary': 30
        }
        
        # Material-specific color characteristics
        self.material_colors = {
            'metal': {'reflective': True, 'saturation_range': (0.3, 1.0), 'brightness_range': (0.4, 1.0)},
            'plastic': {'reflective': False, 'saturation_range': (0.2, 0.9), 'brightness_range': (0.3, 0.9)},
            'fabric': {'reflective': False, 'saturation_range': (0.1, 0.8), 'brightness_range': (0.2, 0.8)},
            'leather': {'reflective': False, 'saturation_range': (0.1, 0.7), 'brightness_range': (0.2, 0.7)},
            'glass': {'reflective': True, 'saturation_range': (0.0, 0.6), 'brightness_range': (0.6, 1.0)},
            'wood': {'reflective': False, 'saturation_range': (0.1, 0.6), 'brightness_range': (0.2, 0.7)}
        }
    
    def _initialize_ultra_color_database(self) -> Dict:
        """Initialize ultra-comprehensive color database with perceptual properties."""
        return {
            # Primary Colors with perceptual data
            'red': {
                'rgb': (255, 0, 0), 'hsv': (0, 100, 100), 'lab': (53, 80, 67),
                'perceptual': {'warmth': 1.0, 'intensity': 1.0, 'visibility': 1.0},
                'materials': ['plastic', 'fabric', 'metal', 'leather']
            },
            'green': {
                'rgb': (0, 128, 0), 'hsv': (120, 100, 50), 'lab': (46, -51, 50),
                'perceptual': {'warmth': -0.5, 'intensity': 0.8, 'visibility': 0.9},
                'materials': ['plastic', 'fabric', 'metal', 'leather']
            },
            'blue': {
                'rgb': (0, 0, 255), 'hsv': (240, 100, 100), 'lab': (32, 79, -107),
                'perceptual': {'warmth': -1.0, 'intensity': 1.0, 'visibility': 0.8},
                'materials': ['plastic', 'fabric', 'metal', 'glass']
            },
            
            # Extended color families with material associations
            'crimson': {
                'rgb': (220, 20, 60), 'hsv': (348, 91, 86), 'lab': (48, 75, 38),
                'perceptual': {'warmth': 0.9, 'intensity': 0.9, 'visibility': 0.95},
                'materials': ['fabric', 'leather', 'plastic']
            },
            'navy': {
                'rgb': (0, 0, 128), 'hsv': (240, 100, 50), 'lab': (20, 20, -50),
                'perceptual': {'warmth': -0.8, 'intensity': 0.6, 'visibility': 0.7},
                'materials': ['fabric', 'leather', 'metal']
            },
            'gold': {
                'rgb': (255, 215, 0), 'hsv': (51, 100, 100), 'lab': (88, -10, 85),
                'perceptual': {'warmth': 0.8, 'intensity': 0.9, 'visibility': 0.9},
                'materials': ['metal', 'plastic']
            },
            'silver': {
                'rgb': (192, 192, 192), 'hsv': (0, 0, 75), 'lab': (75, 0, 0),
                'perceptual': {'warmth': 0.0, 'intensity': 0.5, 'visibility': 0.8},
                'materials': ['metal', 'plastic']
            },
            'rose_gold': {
                'rgb': (233, 150, 122), 'hsv': (15, 48, 91), 'lab': (70, 25, 35),
                'perceptual': {'warmth': 0.7, 'intensity': 0.7, 'visibility': 0.85},
                'materials': ['metal', 'plastic']
            },
            'matte_black': {
                'rgb': (64, 64, 64), 'hsv': (0, 0, 25), 'lab': (25, 0, 0),
                'perceptual': {'warmth': 0.0, 'intensity': 0.2, 'visibility': 0.6},
                'materials': ['plastic', 'fabric', 'leather', 'metal']
            },
            'space_gray': {
                'rgb': (120, 120, 120), 'hsv': (0, 0, 47), 'lab': (47, 0, 0),
                'perceptual': {'warmth': 0.0, 'intensity': 0.3, 'visibility': 0.7},
                'materials': ['metal', 'plastic']
            },
            'midnight_green': {
                'rgb': (25, 25, 112), 'hsv': (240, 78, 44), 'lab': (15, 15, -45),
                'perceptual': {'warmth': -0.6, 'intensity': 0.5, 'visibility': 0.6},
                'materials': ['metal', 'plastic']
            }
        }
    
    def _initialize_perceptual_weights(self) -> Dict:
        """Initialize perceptual color matching weights."""
        return {
            'hue': 0.4,      # Most important for color identification
            'saturation': 0.3, # Important for color intensity
            'brightness': 0.2, # Important for color visibility
            'warmth': 0.1     # Important for color temperature
        }
    
    def analyze_image_colors_ultra(self, image_path: str, num_colors: int = 12) -> Dict:
        """Ultra-enhanced color analysis with machine learning and perceptual matching."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Apply image enhancement
            enhanced_image = self._enhance_image_for_color_analysis(image)
            enhanced_array = np.array(enhanced_image)
            
            # Foreground segmentation to reduce background influence
            try:
                valid_mask = self._compute_foreground_mask(enhanced_array)
            except Exception:
                valid_mask = None
            
            # Multi-algorithm color extraction
            colors_kmeans = self._extract_colors_kmeans(enhanced_array, num_colors, valid_mask)
            colors_dbscan = self._extract_colors_dbscan(enhanced_array, valid_mask)
            colors_meanshift = self._extract_colors_meanshift(enhanced_array, valid_mask)
            
            # Combine results using ensemble method
            combined_colors = self._combine_color_extraction_results(
                colors_kmeans, colors_dbscan, colors_meanshift
            )
            
            # Perceptual color analysis
            perceptual_analysis = self._analyze_perceptual_properties(combined_colors)
            
            # Material detection and color-material association
            material_analysis = self._analyze_material_colors(enhanced_array, combined_colors)
            
            # Color harmony analysis
            harmony_analysis = self._analyze_color_harmony(combined_colors)
            
            # Generate comprehensive color description
            color_description = self._generate_ultra_color_description(
                combined_colors, perceptual_analysis, material_analysis, harmony_analysis
            )
            
            return {
                'success': True,
                'primary_colors': combined_colors[:3],
                'all_colors': combined_colors,
                'perceptual_analysis': perceptual_analysis,
                'material_analysis': material_analysis,
                'harmony_analysis': harmony_analysis,
                'color_description': color_description,
                'named_palette': self._name_palette(combined_colors),
                'confidence_score': self._calculate_confidence_score(combined_colors, perceptual_analysis),
                'metadata': {
                    'num_colors_detected': len(combined_colors),
                    'image_size': image.size,
                    'analysis_method': 'ultra_enhanced'
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'fallback': True
            }
    
    def _enhance_image_for_color_analysis(self, image: Image.Image) -> Image.Image:
        """Enhance image for better color analysis."""
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(enhanced_rgb)
        
        # Apply slight saturation enhancement
        enhancer = ImageEnhance.Color(enhanced_image)
        enhanced_image = enhancer.enhance(1.1)
        
        return enhanced_image
    
    def _extract_colors_kmeans(self, image_array: np.ndarray, num_colors: int, valid_mask: np.ndarray | None = None) -> List[Dict]:
        """Extract colors using K-means clustering."""
        # Reshape image to list of pixels
        pixels = image_array.reshape(-1, 3)
        if valid_mask is not None:
            pixels = pixels[valid_mask.flatten()]
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers and labels
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        # Calculate color frequencies
        label_counts = Counter(labels)
        total_pixels = len(labels)
        
        color_data = []
        for i, color in enumerate(colors):
            frequency = label_counts[i] / total_pixels
            if frequency > 0.01:  # Only include colors with >1% frequency
                color_data.append({
                    'rgb': tuple(color),
                    'frequency': frequency,
                    'method': 'kmeans'
                })
        
        return sorted(color_data, key=lambda x: x['frequency'], reverse=True)
    
    def _extract_colors_dbscan(self, image_array: np.ndarray, valid_mask: np.ndarray | None = None) -> List[Dict]:
        """Extract colors using DBSCAN clustering."""
        # Reshape image to list of pixels
        pixels = image_array.reshape(-1, 3)
        if valid_mask is not None:
            pixels = pixels[valid_mask.flatten()]
        
        # Sample pixels for DBSCAN (too many pixels can be slow)
        sample_size = min(10000, len(pixels))
        sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[sample_indices]
        
        # Apply DBSCAN clustering with error handling
        try:
            dbscan = DBSCAN(eps=30, min_samples=50)
            labels = dbscan.fit_predict(sample_pixels)
        except Exception as e:
            # Fallback to K-means if DBSCAN fails
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(5, len(sample_pixels)//10), random_state=42, n_init=10)
            labels = kmeans.fit_predict(sample_pixels)
            # Convert to DBSCAN-like format
            labels = labels + 1  # DBSCAN uses -1 for noise, so shift labels
        
        # Get unique clusters (excluding noise points labeled as -1)
        unique_labels = set(labels) - {-1}
        
        color_data = []
        for label in unique_labels:
            cluster_pixels = sample_pixels[labels == label]
            if len(cluster_pixels) > 0:
                # Calculate mean color of cluster
                mean_color = np.mean(cluster_pixels, axis=0).astype(int)
                frequency = len(cluster_pixels) / len(sample_pixels)
                
                if frequency > 0.01:  # Only include colors with >1% frequency
                    color_data.append({
                        'rgb': tuple(mean_color),
                        'frequency': frequency,
                        'method': 'dbscan'
                    })
        
        return sorted(color_data, key=lambda x: x['frequency'], reverse=True)
    
    def _extract_colors_meanshift(self, image_array: np.ndarray, valid_mask: np.ndarray | None = None) -> List[Dict]:
        """Extract colors using Mean Shift clustering."""
        # Reshape image to list of pixels
        pixels = image_array.reshape(-1, 3)
        if valid_mask is not None:
            pixels = pixels[valid_mask.flatten()]
        
        # Sample pixels for Mean Shift (too many pixels can be slow)
        sample_size = min(5000, len(pixels))
        sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[sample_indices]
        
        # Apply Mean Shift clustering with error handling
        try:
            meanshift = MeanShift(bandwidth=30, bin_seeding=True)
            labels = meanshift.fit_predict(sample_pixels)
            cluster_centers = meanshift.cluster_centers_
        except Exception as e:
            # Fallback to K-means if Mean Shift fails
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(5, len(sample_pixels)//10), random_state=42, n_init=10)
            labels = kmeans.fit_predict(sample_pixels)
            cluster_centers = kmeans.cluster_centers_
        
        color_data = []
        for i, center in enumerate(cluster_centers):
            cluster_pixels = sample_pixels[labels == i]
            if len(cluster_pixels) > 0:
                frequency = len(cluster_pixels) / len(sample_pixels)
                
                if frequency > 0.01:  # Only include colors with >1% frequency
                    color_data.append({
                        'rgb': tuple(center.astype(int)),
                        'frequency': frequency,
                        'method': 'meanshift'
                    })
        
        return sorted(color_data, key=lambda x: x['frequency'], reverse=True)
    
    def _combine_color_extraction_results(self, *color_results) -> List[Dict]:
        """Combine results from multiple color extraction methods."""
        all_colors = []
        for color_list in color_results:
            all_colors.extend(color_list)
        
        # Group similar colors together
        grouped_colors = self._group_similar_colors(all_colors)
        
        # Calculate final color properties
        final_colors = []
        for group in grouped_colors:
            if group:
                # Calculate weighted average based on frequency and method
                # Increase weight for more saturated colors and penalize near-grays
                def sat_weight(c):
                    import colorsys as _cs
                    r, g, b = [v/255.0 for v in c['rgb']]
                    h, s, v = _cs.rgb_to_hsv(r, g, b)
                    gray_penalty = 0.7 if abs(r-g) < 0.03 and abs(g-b) < 0.03 else 1.0
                    return c['frequency'] * (0.6 + 0.8*s) * gray_penalty

                total_weight = sum(sat_weight(color) for color in group)
                weighted_rgb = np.average(
                    [color['rgb'] for color in group],
                    weights=[sat_weight(color) for color in group],
                    axis=0
                ).astype(int)
                
                final_colors.append({
                    'rgb': tuple(weighted_rgb),
                    'frequency': float(total_weight),
                    'methods': [color['method'] for color in group],
                    'confidence': len(group) / len(color_results)  # Higher if detected by multiple methods
                })
        
        return sorted(final_colors, key=lambda x: x['frequency'], reverse=True)

    def _compute_foreground_mask(self, image_array: np.ndarray) -> np.ndarray:
        """Compute a foreground mask using GrabCut with a slightly inset rectangle.
        Returns a boolean mask with True for foreground pixels.
        """
        h, w, _ = image_array.shape
        mask = np.zeros((h, w), np.uint8)
        # Inset rectangle to avoid borders
        x = int(w * 0.05); y = int(h * 0.05)
        rw = max(1, int(w * 0.90)); rh = max(1, int(h * 0.90))
        rect = (x, y, rw, rh)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(image_array, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), True, False)
        return fg_mask

    def _name_palette(self, colors: List[Dict]) -> List[Dict]:
        """Return top-3 color names with percentages."""
        total = sum(c.get('frequency', 0.0) for c in colors) or 1.0
        top = []
        for c in colors[:3]:
            name = self._get_closest_color_name(c['rgb'])
            top.append({
                'name': name,
                'rgb': c['rgb'],
                'percent': float(c.get('frequency', 0.0) / total)
            })
        return top
    
    def _group_similar_colors(self, colors: List[Dict], threshold: float = 30.0) -> List[List[Dict]]:
        """Group similar colors together based on Euclidean distance."""
        if not colors:
            return []
        
        groups = []
        used = set()
        
        for i, color1 in enumerate(colors):
            if i in used:
                continue
                
            group = [color1]
            used.add(i)
            
            for j, color2 in enumerate(colors[i+1:], i+1):
                if j in used:
                    continue
                    
                # Calculate Euclidean distance in RGB space
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(color1['rgb'], color2['rgb'])))
                
                if distance <= threshold:
                    group.append(color2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _analyze_perceptual_properties(self, colors: List[Dict]) -> Dict:
        """Analyze perceptual properties of detected colors."""
        if not colors:
            return {}
        
        # Convert RGB to HSV for perceptual analysis
        hsv_colors = []
        for color in colors:
            rgb = np.array(color['rgb']) / 255.0
            hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
            hsv_colors.append({
                'hue': hsv[0] * 360,  # Convert to degrees
                'saturation': hsv[1] * 100,  # Convert to percentage
                'value': hsv[2] * 100,  # Convert to percentage
                'rgb': color['rgb']
            })
        
        # Calculate overall color properties
        avg_hue = np.mean([c['hue'] for c in hsv_colors])
        avg_saturation = np.mean([c['saturation'] for c in hsv_colors])
        avg_value = np.mean([c['value'] for c in hsv_colors])
        
        # Determine color temperature
        color_temperature = self._calculate_color_temperature(avg_hue)
        
        # Determine color intensity
        color_intensity = self._calculate_color_intensity(avg_saturation, avg_value)
        
        return {
            'average_hue': avg_hue,
            'average_saturation': avg_saturation,
            'average_brightness': avg_value,
            'color_temperature': color_temperature,
            'color_intensity': color_intensity,
            'dominant_hue_range': self._get_dominant_hue_range(hsv_colors),
            'color_variety': len(set(c['hue'] // 30 for c in hsv_colors))  # Number of different hue ranges
        }
    
    def _calculate_color_temperature(self, hue: float) -> str:
        """Calculate color temperature based on hue."""
        if 0 <= hue <= 60 or 300 <= hue <= 360:
            return 'warm'
        elif 60 < hue < 180:
            return 'cool'
        else:
            return 'neutral'
    
    def _calculate_color_intensity(self, saturation: float, value: float) -> str:
        """Calculate color intensity based on saturation and value."""
        if saturation > 70 and value > 70:
            return 'vibrant'
        elif saturation > 50 and value > 50:
            return 'moderate'
        elif saturation > 30 and value > 30:
            return 'muted'
        else:
            return 'subtle'
    
    def _get_dominant_hue_range(self, hsv_colors: List[Dict]) -> str:
        """Get the dominant hue range."""
        hue_ranges = {
            'red': (0, 30),
            'orange': (30, 60),
            'yellow': (60, 90),
            'green': (90, 150),
            'cyan': (150, 180),
            'blue': (180, 240),
            'purple': (240, 300),
            'magenta': (300, 360)
        }
        
        range_counts = defaultdict(int)
        for color in hsv_colors:
            hue = color['hue']
            for range_name, (start, end) in hue_ranges.items():
                if start <= hue < end:
                    range_counts[range_name] += 1
                    break
        
        if range_counts:
            return max(range_counts, key=range_counts.get)
        return 'unknown'
    
    def _analyze_material_colors(self, image_array: np.ndarray, colors: List[Dict]) -> Dict:
        """Analyze material-specific color properties."""
        # Convert to different color spaces for material analysis
        lab_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        material_scores = {}
        
        for material, properties in self.material_colors.items():
            score = 0
            total_colors = len(colors)
            
            for color in colors:
                rgb = np.array(color['rgb']) / 255.0
                hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
                
                # Check saturation range
                sat = hsv[1] * 100
                if properties['saturation_range'][0] <= sat <= properties['saturation_range'][1]:
                    score += 1
                
                # Check brightness range
                val = hsv[2] * 100
                if properties['brightness_range'][0] <= val <= properties['brightness_range'][1]:
                    score += 1
            
            if total_colors > 0:
                material_scores[material] = score / (total_colors * 2)  # Normalize by max possible score
        
        # Find best matching material
        best_material = max(material_scores, key=material_scores.get) if material_scores else 'unknown'
        
        return {
            'detected_material': best_material,
            'material_confidence': material_scores.get(best_material, 0),
            'all_material_scores': material_scores
        }
    
    def _analyze_color_harmony(self, colors: List[Dict]) -> Dict:
        """Analyze color harmony relationships."""
        if len(colors) < 2:
            return {'harmony_type': 'single_color', 'harmony_score': 1.0}
        
        # Convert to HSV for harmony analysis
        hsv_colors = []
        for color in colors[:5]:  # Analyze top 5 colors
            rgb = np.array(color['rgb']) / 255.0
            hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
            hsv_colors.append(hsv[0] * 360)  # Hue in degrees
        
        # Calculate hue differences
        hue_diffs = []
        for i in range(len(hsv_colors)):
            for j in range(i + 1, len(hsv_colors)):
                diff = abs(hsv_colors[i] - hsv_colors[j])
                # Handle circular nature of hue
                diff = min(diff, 360 - diff)
                hue_diffs.append(diff)
        
        # Determine harmony type based on hue differences
        harmony_type = 'random'
        harmony_score = 0.0
        
        if hue_diffs:
            avg_diff = np.mean(hue_diffs)
            
            # Check for specific harmony types
            for harmony, target_diff in self.harmony_types.items():
                if harmony == 'monochromatic':
                    if avg_diff < 30:
                        harmony_type = harmony
                        harmony_score = 1.0 - (avg_diff / 30)
                        break
                else:
                    tolerance = 15
                    if abs(avg_diff - target_diff) <= tolerance:
                        harmony_type = harmony
                        harmony_score = 1.0 - (abs(avg_diff - target_diff) / tolerance)
                        break
        
        return {
            'harmony_type': harmony_type,
            'harmony_score': harmony_score,
            'average_hue_difference': np.mean(hue_diffs) if hue_diffs else 0,
            'color_count': len(colors)
        }
    
    def _generate_ultra_color_description(self, colors: List[Dict], perceptual: Dict, 
                                        material: Dict, harmony: Dict) -> str:
        """Generate comprehensive color description."""
        if not colors:
            return "No colors detected"
        
        primary_color = colors[0]
        primary_name = self._get_closest_color_name(primary_color['rgb'])
        
        # Build description components
        description_parts = []
        
        # Primary color
        description_parts.append(f"Primary color: {primary_name}")
        
        # Secondary colors
        if len(colors) > 1:
            secondary_colors = [self._get_closest_color_name(color['rgb']) for color in colors[1:3]]
            if secondary_colors:
                description_parts.append(f"Secondary colors: {', '.join(secondary_colors)}")
        
        # Color characteristics
        if perceptual.get('color_temperature'):
            description_parts.append(f"Color temperature: {perceptual['color_temperature']}")
        
        if perceptual.get('color_intensity'):
            description_parts.append(f"Intensity: {perceptual['color_intensity']}")
        
        # Material association
        if material.get('detected_material') and material.get('material_confidence', 0) > 0.3:
            description_parts.append(f"Material: {material['detected_material']}")
        
        # Harmony information
        if harmony.get('harmony_type') != 'random' and harmony.get('harmony_score', 0) > 0.5:
            description_parts.append(f"Color harmony: {harmony['harmony_type']}")
        
        return "; ".join(description_parts)
    
    def _get_closest_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Get the closest color name from the database."""
        min_distance = float('inf')
        closest_color = 'unknown'
        
        for color_name, color_data in self.color_database.items():
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, color_data['rgb'])))
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        
        return closest_color
    
    def _calculate_confidence_score(self, colors: List[Dict], perceptual: Dict) -> float:
        """Calculate overall confidence score for color detection."""
        if not colors:
            return 0.0
        
        # Base confidence from color frequency
        frequency_score = min(1.0, sum(color['frequency'] for color in colors[:3]))
        
        # Confidence from detection method agreement
        method_agreement = np.mean([color.get('confidence', 0.5) for color in colors[:3]])
        
        # Confidence from perceptual analysis
        perceptual_score = 0.5
        if perceptual.get('color_variety', 0) > 1:
            perceptual_score += 0.2
        if perceptual.get('color_intensity') in ['vibrant', 'moderate']:
            perceptual_score += 0.3
        
        # Combine scores
        overall_confidence = (frequency_score * 0.4 + method_agreement * 0.3 + perceptual_score * 0.3)
        
        return min(1.0, overall_confidence)
    
    def compare_colors_ultra(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> Dict:
        """Ultra-enhanced color comparison with perceptual analysis."""
        # Convert to different color spaces
        rgb1, rgb2 = np.array(color1), np.array(color2)
        
        # RGB distance
        rgb_distance = np.sqrt(np.sum((rgb1 - rgb2) ** 2))
        rgb_similarity = 1.0 - (rgb_distance / (255 * np.sqrt(3)))
        
        # HSV distance
        hsv1 = colorsys.rgb_to_hsv(rgb1[0]/255, rgb1[1]/255, rgb1[2]/255)
        hsv2 = colorsys.rgb_to_hsv(rgb2[0]/255, rgb2[1]/255, rgb2[2]/255)
        hsv_distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(hsv1, hsv2)))
        hsv_similarity = 1.0 - (hsv_distance / np.sqrt(3))
        
        # LAB distance (perceptual)
        lab1 = self._rgb_to_lab(rgb1)
        lab2 = self._rgb_to_lab(rgb2)
        lab_distance = np.sqrt(np.sum((lab1 - lab2) ** 2))
        lab_similarity = 1.0 - (lab_distance / 100)  # Approximate max LAB distance
        
        # Weighted similarity
        weights = self.perceptual_weights
        weighted_similarity = (
            weights['hue'] * hsv_similarity +
            weights['saturation'] * hsv_similarity +
            weights['brightness'] * hsv_similarity +
            weights['warmth'] * rgb_similarity
        )
        
        return {
            'rgb_similarity': rgb_similarity,
            'hsv_similarity': hsv_similarity,
            'lab_similarity': lab_similarity,
            'weighted_similarity': weighted_similarity,
            'perceptual_match': weighted_similarity > 0.7,
            'color_names': {
                'color1': self._get_closest_color_name(color1),
                'color2': self._get_closest_color_name(color2)
            }
        }
    
    def _rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to LAB color space."""
        # Normalize RGB
        rgb_norm = rgb / 255.0
        
        # Convert to XYZ
        rgb_linear = np.where(rgb_norm <= 0.04045, rgb_norm / 12.92, ((rgb_norm + 0.055) / 1.055) ** 2.4)
        
        # Apply transformation matrix
        xyz = np.dot(rgb_linear, [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        # Normalize by white point
        xyz = xyz / [0.95047, 1.00000, 1.08883]
        
        # Convert to LAB
        def f(t):
            return np.where(t > 0.008856, t ** (1/3), (7.787 * t) + (16/116))
        
        fx, fy, fz = f(xyz[0]), f(xyz[1]), f(xyz[2])
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return np.array([L, a, b])


# Global instance
ultra_color_detector = UltraEnhancedColorDetector()

def analyze_image_colors_ultra(image_path: str, num_colors: int = 12) -> Dict:
    """Convenience function for ultra-enhanced color analysis."""
    return ultra_color_detector.analyze_image_colors_ultra(image_path, num_colors)

def enhance_existing_color_detection_ultra(image_path: str, bbox: List[int] = None, object_type: str = None) -> Dict:
    """Ultra-enhanced version of existing color detection."""
    try:
        if bbox:
            # Crop image to bounding box
            image = Image.open(image_path).convert('RGB')
            cropped = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            
            # Save cropped image temporarily
            temp_path = f"temp_cropped_{os.getpid()}.jpg"
            cropped.save(temp_path)
            
            try:
                result = ultra_color_detector.analyze_image_colors_ultra(temp_path, num_colors=8)
                result['object_type'] = object_type
                result['bbox'] = bbox
                return result
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            result = ultra_color_detector.analyze_image_colors_ultra(image_path, num_colors=8)
            return result
            
    except Exception as e:
        print(f"Error in ultra color detection: {e}")
        return ultra_color_detector._get_default_color_analysis()

def compare_colors_ultra(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> Dict:
    """Ultra-enhanced color comparison."""
    return ultra_color_detector.compare_colors_ultra(color1, color2)

def get_ultra_color_description(colors: List[Dict], perceptual: Dict, material: Dict, harmony: Dict) -> str:
    """Generate ultra-enhanced color description."""
    return ultra_color_detector._generate_ultra_color_description(colors, perceptual, material, harmony)

if __name__ == "__main__":
    # Test the ultra-enhanced color detection
    print("🎨 Testing Ultra Enhanced Color Detection System")
    print("=" * 60)
    
    # Test with sample images if available
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        import glob
        test_images.extend(glob.glob(ext))
    
    if test_images:
        for img_path in test_images[:3]:  # Test first 3 images
            print(f"\n📸 Analyzing: {img_path}")
            result = analyze_image_colors_ultra(img_path)
            
            if result.get('success'):
                print(f"  ✅ Primary Color: {result['primary_colors'][0]['rgb'] if result['primary_colors'] else 'None'}")
                print(f"  📊 Confidence: {result.get('confidence_score', 0):.2f}")
                print(f"  🎨 Description: {result.get('color_description', 'N/A')}")
            else:
                print(f"  ❌ Error: {result.get('error', 'Unknown error')}")
    else:
        print("No test images found. Please add some images to test the system.")