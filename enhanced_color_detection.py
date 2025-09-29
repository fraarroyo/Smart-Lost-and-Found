#!/usr/bin/env python3
"""
Enhanced Color Detection System
Provides advanced color analysis, color space conversions, and color matching capabilities.
"""

import cv2
import numpy as np
from PIL import Image
import webcolors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import colorsys
from typing import Dict, List, Tuple, Optional, Union
import json
import os

class EnhancedColorDetector:
    """Advanced color detection and analysis system."""
    
    def __init__(self):
        self.color_spaces = ['RGB', 'HSV', 'LAB', 'LUV', 'XYZ']
        self.color_harmonies = {
            'complementary': 180,
            'triadic': 120,
            'tetradic': 90,
            'analogous': 30,
            'split_complementary': 150
        }
        
        # Extended color palette for better matching
        self.extended_colors = {
            'red': (255, 0, 0), 'crimson': (220, 20, 60), 'maroon': (128, 0, 0),
            'orange': (255, 165, 0), 'coral': (255, 127, 80), 'salmon': (250, 128, 114),
            'yellow': (255, 255, 0), 'gold': (255, 215, 0), 'amber': (255, 191, 0),
            'green': (0, 128, 0), 'lime': (0, 255, 0), 'emerald': (80, 200, 120),
            'blue': (0, 0, 255), 'navy': (0, 0, 128), 'sky': (135, 206, 235),
            'purple': (128, 0, 128), 'violet': (238, 130, 238), 'magenta': (255, 0, 255),
            'pink': (255, 192, 203), 'rose': (255, 228, 225), 'fuchsia': (255, 0, 255),
            'brown': (165, 42, 42), 'tan': (210, 180, 140), 'beige': (245, 245, 220),
            'gray': (128, 128, 128), 'silver': (192, 192, 192), 'charcoal': (54, 69, 79),
            'black': (0, 0, 0), 'white': (255, 255, 255), 'ivory': (255, 255, 240)
        }
    
    def analyze_image_colors(self, image_path: str, num_colors: int = 8) -> Dict:
        """Comprehensive color analysis of an image."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Resize for faster processing while maintaining quality
            if image_array.shape[0] > 1000 or image_array.shape[1] > 1000:
                scale = min(1000 / image_array.shape[0], 1000 / image_array.shape[1])
                new_size = (int(image_array.shape[1] * scale), int(image_array.shape[0] * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                image_array = np.array(image)
            
            # Extract color palette using multiple methods
            dominant_colors = self._extract_dominant_colors(image_array, num_colors)
            color_palette = self._extract_color_palette(image_array, num_colors)
            color_histogram = self._analyze_color_histogram(image_array)
            color_harmony = self._analyze_color_harmony(dominant_colors)
            
            # Analyze color distribution
            color_distribution = self._analyze_color_distribution(image_array)
            
            # Extract color features
            color_features = self._extract_color_features(image_array)
            
            return {
                'dominant_colors': dominant_colors,
                'color_palette': color_palette,
                'color_histogram': color_histogram,
                'color_harmony': color_harmony,
                'color_distribution': color_distribution,
                'color_features': color_features,
                'primary_color': dominant_colors[0] if dominant_colors else None,
                'secondary_color': dominant_colors[1] if len(dominant_colors) > 1 else None,
                'color_count': len(dominant_colors)
            }
            
        except Exception as e:
            print(f"Error in analyze_image_colors: {e}")
            return self._get_default_color_analysis()
    
    def analyze_object_colors(self, image_path: str, bbox: List[int], object_type: str = None) -> Dict:
        """Analyze colors of a specific object within a bounding box."""
        try:
            image = Image.open(image_path).convert('RGB')
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure bbox is within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                return self._get_default_color_analysis()
            
            # Crop object region
            object_region = image.crop((x1, y1, x2, y2))
            object_array = np.array(object_region)
            
            # Object-specific color analysis
            if object_type and object_type.lower() in ['phone', 'cell phone', 'smartphone']:
                return self._analyze_phone_colors(object_array, bbox)
            elif object_type and object_type.lower() in ['tumbler', 'cup', 'bottle']:
                return self._analyze_container_colors(object_array, bbox)
            else:
                return self.analyze_image_colors_from_array(object_array)
                
        except Exception as e:
            print(f"Error in analyze_object_colors: {e}")
            return self._get_default_color_analysis()
    
    def _analyze_phone_colors(self, image_array: np.ndarray, bbox: List[int]) -> Dict:
        """Specialized color analysis for phones/smartphones."""
        try:
            h, w = image_array.shape[:2]
            
            # Analyze different regions of the phone
            # Screen area (top 60% of the phone)
            screen_region = image_array[:int(h * 0.6), :]
            screen_colors = self._extract_dominant_colors(screen_region, 3)
            
            # Frame/body area (bottom 40% of the phone)
            frame_region = image_array[int(h * 0.6):, :]
            frame_colors = self._extract_dominant_colors(frame_region, 3)
            
            # Edge detection for frame color
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = image_array[edges > 0]
            edge_colors = self._extract_dominant_colors(edge_pixels, 2) if edge_pixels.size > 0 else []
            
            # Combine all color information
            all_colors = screen_colors + frame_colors + edge_colors
            unique_colors = self._remove_similar_colors(all_colors, threshold=30)
            
            return {
                'screen_colors': screen_colors,
                'frame_colors': frame_colors,
                'edge_colors': edge_colors,
                'primary_color': unique_colors[0] if unique_colors else None,
                'secondary_color': unique_colors[1] if len(unique_colors) > 1 else None,
                'all_colors': unique_colors,
                'object_type': 'phone'
            }
            
        except Exception as e:
            print(f"Error in _analyze_phone_colors: {e}")
            return self._get_default_color_analysis()
    
    def _analyze_container_colors(self, image_array: np.ndarray, bbox: List[int]) -> Dict:
        """Specialized color analysis for containers (tumblers, cups, bottles)."""
        try:
            h, w = image_array.shape[:2]
            
            # Analyze central region (main body)
            center_h_start, center_h_end = int(h * 0.2), int(h * 0.8)
            center_region = image_array[center_h_start:center_h_end, :]
            center_colors = self._extract_dominant_colors(center_region, 3)
            
            # Analyze top and bottom regions for accents
            top_region = image_array[:int(h * 0.3), :]
            bottom_region = image_array[int(h * 0.7):, :]
            
            top_colors = self._extract_dominant_colors(top_region, 2)
            bottom_colors = self._extract_dominant_colors(bottom_region, 2)
            
            # Combine colors
            all_colors = center_colors + top_colors + bottom_colors
            unique_colors = self._remove_similar_colors(all_colors, threshold=25)
            
            return {
                'center_colors': center_colors,
                'top_colors': top_colors,
                'bottom_colors': bottom_colors,
                'primary_color': unique_colors[0] if unique_colors else None,
                'secondary_color': unique_colors[1] if len(unique_colors) > 1 else None,
                'all_colors': unique_colors,
                'object_type': 'container'
            }
            
        except Exception as e:
            print(f"Error in _analyze_container_colors: {e}")
            return self._get_default_color_analysis()
    
    def _extract_dominant_colors(self, image_array: np.ndarray, num_colors: int) -> List[Dict]:
        """Extract dominant colors using K-means clustering."""
        try:
            # Reshape image to list of pixels
            pixels = image_array.reshape(-1, 3)
            
            # Remove very dark and very light pixels (likely background)
            brightness = np.mean(pixels, axis=1)
            mask = (brightness > 20) & (brightness < 235)
            filtered_pixels = pixels[mask]
            
            if len(filtered_pixels) < num_colors:
                filtered_pixels = pixels
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=min(num_colors, len(filtered_pixels)), 
                           random_state=42, n_init=10)
            kmeans.fit(filtered_pixels)
            
            # Get cluster centers and counts
            colors = kmeans.cluster_centers_.astype(int)
            counts = np.bincount(kmeans.labels_)
            
            # Sort by frequency
            sorted_indices = np.argsort(-counts)
            
            result = []
            for idx in sorted_indices:
                if idx < len(colors):
                    rgb = tuple(colors[idx])
                    color_info = {
                        'rgb': rgb,
                        'hex': self._rgb_to_hex(rgb),
                        'name': self._rgb_to_name(rgb),
                        'percentage': (counts[idx] / len(filtered_pixels)) * 100,
                        'hsv': self._rgb_to_hsv(rgb),
                        'lab': self._rgb_to_lab(rgb)
                    }
                    result.append(color_info)
            
            return result
            
        except Exception as e:
            print(f"Error in _extract_dominant_colors: {e}")
            return []
    
    def _extract_color_palette(self, image_array: np.ndarray, num_colors: int) -> List[Dict]:
        """Extract a harmonious color palette."""
        try:
            # Use median cut algorithm for better color quantization
            quantized = self._median_cut_quantization(image_array, num_colors)
            return self._extract_dominant_colors(quantized, num_colors)
        except Exception as e:
            print(f"Error in _extract_color_palette: {e}")
            return []
    
    def _median_cut_quantization(self, image_array: np.ndarray, num_colors: int) -> np.ndarray:
        """Implement median cut algorithm for color quantization."""
        try:
            pixels = image_array.reshape(-1, 3)
            
            # Start with all pixels in one bucket
            buckets = [pixels]
            
            while len(buckets) < num_colors and any(len(bucket) > 1 for bucket in buckets):
                # Find bucket with largest range
                largest_bucket_idx = 0
                largest_range = 0
                
                for i, bucket in enumerate(buckets):
                    if len(bucket) > 1:
                        ranges = np.max(bucket, axis=0) - np.min(bucket, axis=0)
                        max_range = np.max(ranges)
                        if max_range > largest_range:
                            largest_range = max_range
                            largest_bucket_idx = i
                
                # Split the bucket
                bucket = buckets[largest_bucket_idx]
                channel = np.argmax(np.max(bucket, axis=0) - np.min(bucket, axis=0))
                median = np.median(bucket[:, channel])
                
                left_bucket = bucket[bucket[:, channel] < median]
                right_bucket = bucket[bucket[:, channel] >= median]
                
                if left_bucket.size > 0 and right_bucket.size > 0:
                    buckets[largest_bucket_idx] = left_bucket
                    buckets.append(right_bucket)
                else:
                    break
            
            # Create quantized image
            quantized = np.zeros_like(image_array)
            pixels_flat = image_array.reshape(-1, 3)
            
            for bucket in buckets:
                if bucket.size > 0:
                    mean_color = np.mean(bucket, axis=0).astype(int)
                    
                    # Find closest pixels to this bucket's mean color
                    distances = np.sum((pixels_flat - mean_color) ** 2, axis=1)
                    closest_indices = np.argmin(distances)
                    
                    # Assign mean color to pixels that are closest to this bucket
                    quantized.reshape(-1, 3)[closest_indices] = mean_color
            
            return quantized
            
        except Exception as e:
            print(f"Error in median cut quantization: {e}")
            return image_array
    
    def _analyze_color_histogram(self, image_array: np.ndarray) -> Dict:
        """Analyze color distribution histogram."""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
            
            # Calculate histograms
            hist_rgb = cv2.calcHist([image_array], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            hist_hsv = cv2.calcHist([hsv], [0, 1, 2], None, [32, 32, 32], [0, 180, 0, 256, 0, 256])
            
            # Calculate color statistics
            mean_rgb = np.mean(image_array, axis=(0, 1))
            std_rgb = np.std(image_array, axis=(0, 1))
            
            return {
                'rgb_histogram': hist_rgb.tolist(),
                'hsv_histogram': hist_hsv.tolist(),
                'mean_rgb': mean_rgb.tolist(),
                'std_rgb': std_rgb.tolist(),
                'brightness': float(np.mean(mean_rgb)),
                'contrast': float(np.mean(std_rgb))
            }
            
        except Exception as e:
            print(f"Error in _analyze_color_histogram: {e}")
            return {}
    
    def _analyze_color_harmony(self, colors: List[Dict]) -> Dict:
        """Analyze color harmony relationships."""
        try:
            if len(colors) < 2:
                return {'harmony_type': 'single', 'harmony_score': 0.0}
            
            # Convert to HSV for hue analysis
            hues = [color['hsv'][0] for color in colors if 'hsv' in color]
            
            if len(hues) < 2:
                return {'harmony_type': 'unknown', 'harmony_score': 0.0}
            
            # Calculate hue differences
            hue_diffs = []
            for i in range(len(hues)):
                for j in range(i + 1, len(hues)):
                    diff = abs(hues[i] - hues[j])
                    hue_diffs.append(min(diff, 360 - diff))
            
            # Determine harmony type
            harmony_scores = {}
            for harmony_name, target_diff in self.color_harmonies.items():
                scores = [1.0 - abs(diff - target_diff) / 180.0 for diff in hue_diffs]
                harmony_scores[harmony_name] = max(scores) if scores else 0.0
            
            best_harmony = max(harmony_scores.items(), key=lambda x: x[1])
            
            return {
                'harmony_type': best_harmony[0],
                'harmony_score': best_harmony[1],
                'all_harmony_scores': harmony_scores,
                'hue_differences': hue_diffs
            }
            
        except Exception as e:
            print(f"Error in _analyze_color_harmony: {e}")
            return {'harmony_type': 'unknown', 'harmony_score': 0.0}
    
    def _analyze_color_distribution(self, image_array: np.ndarray) -> Dict:
        """Analyze spatial distribution of colors."""
        try:
            h, w = image_array.shape[:2]
            
            # Divide image into regions
            regions = {
                'top_left': image_array[:h//2, :w//2],
                'top_right': image_array[:h//2, w//2:],
                'bottom_left': image_array[h//2:, :w//2],
                'bottom_right': image_array[h//2:, w//2:],
                'center': image_array[h//4:3*h//4, w//4:3*w//4]
            }
            
            region_colors = {}
            for region_name, region in regions.items():
                region_colors[region_name] = self._extract_dominant_colors(region, 2)
            
            return region_colors
            
        except Exception as e:
            print(f"Error in _analyze_color_distribution: {e}")
            return {}
    
    def _extract_color_features(self, image_array: np.ndarray) -> Dict:
        """Extract advanced color features."""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
            
            # Calculate colorfulness (saturation)
            saturation = np.mean(hsv[:, :, 1])
            
            # Calculate brightness
            brightness = np.mean(hsv[:, :, 2])
            
            # Calculate color temperature (warm vs cool)
            r, g, b = np.mean(image_array, axis=(0, 1))
            color_temp = self._calculate_color_temperature(r, g, b)
            
            # Calculate color variance
            color_variance = np.var(image_array.reshape(-1, 3), axis=0)
            
            return {
                'saturation': float(saturation),
                'brightness': float(brightness),
                'color_temperature': color_temp,
                'color_variance': color_variance.tolist(),
                'is_warm': color_temp > 0.5,
                'is_saturated': saturation > 100,
                'is_bright': brightness > 150
            }
            
        except Exception as e:
            print(f"Error in _extract_color_features: {e}")
            return {}
    
    def _calculate_color_temperature(self, r: float, g: float, b: float) -> float:
        """Calculate color temperature (0 = cool, 1 = warm)."""
        try:
            # Simple warm/cool calculation based on red vs blue dominance
            total = r + g + b
            if total == 0:
                return 0.5
            
            red_ratio = r / total
            blue_ratio = b / total
            
            # Warm colors have more red, cool colors have more blue
            return red_ratio / (red_ratio + blue_ratio) if (red_ratio + blue_ratio) > 0 else 0.5
        except:
            return 0.5
    
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex color code."""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def _rgb_to_name(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to color name with improved matching."""
        try:
            # Try exact match first
            try:
                return webcolors.rgb_to_name(rgb)
            except ValueError:
                pass
            
            # Find closest color from extended palette
            min_distance = float('inf')
            closest_color = 'unknown'
            
            for name, color_rgb in self.extended_colors.items():
                distance = sum((a - b) ** 2 for a, b in zip(rgb, color_rgb))
                if distance < min_distance:
                    min_distance = distance
                    closest_color = name
            
            return closest_color
            
        except Exception as e:
            print(f"Error in _rgb_to_name: {e}")
            return 'unknown'
    
    def _rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to HSV."""
        try:
            r, g, b = [x / 255.0 for x in rgb]
            return colorsys.rgb_to_hsv(r, g, b)
        except:
            return (0.0, 0.0, 0.0)
    
    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to LAB color space."""
        try:
            # Convert RGB to XYZ first
            r, g, b = [x / 255.0 for x in rgb]
            
            # Apply gamma correction
            r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
            g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
            b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92
            
            # Convert to XYZ
            x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
            y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
            z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
            
            # Convert to LAB
            x /= 0.95047
            y /= 1.00000
            z /= 1.08883
            
            fx = x ** (1/3) if x > 0.008856 else (7.787 * x + 16/116)
            fy = y ** (1/3) if y > 0.008856 else (7.787 * y + 16/116)
            fz = z ** (1/3) if z > 0.008856 else (7.787 * z + 16/116)
            
            L = 116 * fy - 16
            a = 500 * (fx - fy)
            b_lab = 200 * (fy - fz)
            
            return (L, a, b_lab)
            
        except Exception as e:
            print(f"Error in _rgb_to_lab: {e}")
            return (0.0, 0.0, 0.0)
    
    def _remove_similar_colors(self, colors: List[Dict], threshold: int = 30) -> List[Dict]:
        """Remove similar colors from the list."""
        if not colors:
            return []
        
        unique_colors = [colors[0]]
        
        for color in colors[1:]:
            is_similar = False
            for unique_color in unique_colors:
                distance = sum((a - b) ** 2 for a, b in zip(color['rgb'], unique_color['rgb']))
                if distance < threshold ** 2:
                    is_similar = True
                    break
            
            if not is_similar:
                unique_colors.append(color)
        
        return unique_colors
    
    def _get_default_color_analysis(self) -> Dict:
        """Return default color analysis when errors occur."""
        return {
            'dominant_colors': [],
            'color_palette': [],
            'primary_color': None,
            'secondary_color': None,
            'color_count': 0,
            'error': True
        }
    
    def analyze_image_colors_from_array(self, image_array: np.ndarray) -> Dict:
        """Analyze colors from a numpy array instead of file path."""
        try:
            # Convert to PIL Image for consistency
            image = Image.fromarray(image_array)
            temp_path = 'temp_color_analysis.jpg'
            image.save(temp_path)
            
            result = self.analyze_image_colors(temp_path)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
            
        except Exception as e:
            print(f"Error in analyze_image_colors_from_array: {e}")
            return self._get_default_color_analysis()
    
    def compare_colors(self, color1: Dict, color2: Dict, method: str = 'euclidean') -> float:
        """Compare two colors and return similarity score (0-1)."""
        try:
            if 'rgb' not in color1 or 'rgb' not in color2:
                return 0.0
            
            rgb1 = np.array(color1['rgb'])
            rgb2 = np.array(color2['rgb'])
            
            if method == 'euclidean':
                distance = np.sqrt(np.sum((rgb1 - rgb2) ** 2))
                max_distance = np.sqrt(3 * 255 ** 2)
                return 1.0 - (distance / max_distance)
            
            elif method == 'cosine':
                rgb1_norm = rgb1 / (np.linalg.norm(rgb1) + 1e-8)
                rgb2_norm = rgb2 / (np.linalg.norm(rgb2) + 1e-8)
                return np.dot(rgb1_norm, rgb2_norm)
            
            elif method == 'lab':
                if 'lab' in color1 and 'lab' in color2:
                    lab1 = np.array(color1['lab'])
                    lab2 = np.array(color2['lab'])
                    distance = np.sqrt(np.sum((lab1 - lab2) ** 2))
                    max_distance = 100  # Approximate max LAB distance
                    return 1.0 - min(distance / max_distance, 1.0)
            
            return 0.0
            
        except Exception as e:
            print(f"Error in compare_colors: {e}")
            return 0.0
    
    def find_color_matches(self, target_colors: List[Dict], candidate_colors: List[Dict], 
                          threshold: float = 0.7) -> List[Dict]:
        """Find color matches between target and candidate color lists."""
        matches = []
        
        for target_color in target_colors:
            best_match = None
            best_score = 0.0
            
            for candidate_color in candidate_colors:
                score = self.compare_colors(target_color, candidate_color)
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = candidate_color
            
            if best_match:
                matches.append({
                    'target_color': target_color,
                    'matched_color': best_match,
                    'similarity_score': best_score
                })
        
        return matches


# Integration functions for existing system
def enhance_existing_color_detection(image_path: str, bbox: List[int] = None, object_type: str = None) -> Dict:
    """Enhanced color detection that integrates with existing system."""
    detector = EnhancedColorDetector()
    
    if bbox:
        return detector.analyze_object_colors(image_path, bbox, object_type)
    else:
        return detector.analyze_image_colors(image_path)

def get_enhanced_color_description(color_analysis: Dict) -> str:
    """Generate enhanced color description from analysis results."""
    try:
        if color_analysis.get('error'):
            return "Color analysis unavailable"
        
        primary = color_analysis.get('primary_color')
        secondary = color_analysis.get('secondary_color')
        
        if not primary:
            return "Color information not available"
        
        description_parts = []
        
        # Primary color
        if primary:
            color_name = primary.get('name', 'unknown')
            description_parts.append(f"Primary color: {color_name}")
        
        # Secondary color
        if secondary:
            color_name = secondary.get('name', 'unknown')
            description_parts.append(f"Secondary color: {color_name}")
        
        # Color features
        features = color_analysis.get('color_features', {})
        if features.get('is_warm'):
            description_parts.append("warm tones")
        elif features.get('color_temperature', 0.5) < 0.3:
            description_parts.append("cool tones")
        
        if features.get('is_saturated'):
            description_parts.append("vibrant colors")
        elif features.get('saturation', 0) < 50:
            description_parts.append("muted colors")
        
        # Color harmony
        harmony = color_analysis.get('color_harmony', {})
        if harmony.get('harmony_score', 0) > 0.7:
            description_parts.append(f"{harmony.get('harmony_type', 'harmonious')} color scheme")
        
        return ", ".join(description_parts)
        
    except Exception as e:
        print(f"Error generating color description: {e}")
        return "Color analysis error"


if __name__ == "__main__":
    # Test the enhanced color detection
    detector = EnhancedColorDetector()
    
    # Test with a sample image if available
    test_images = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
                if len(test_images) >= 2:
                    break
        if len(test_images) >= 2:
            break
    
    if test_images:
        print("🧪 Testing Enhanced Color Detection System")
        print("=" * 50)
        
        for i, img_path in enumerate(test_images):
            print(f"\nImage {i+1}: {os.path.basename(img_path)}")
            try:
                analysis = detector.analyze_image_colors(img_path)
                print(f"  Primary color: {analysis.get('primary_color', {}).get('name', 'unknown')}")
                print(f"  Secondary color: {analysis.get('secondary_color', {}).get('name', 'unknown')}")
                print(f"  Color count: {analysis.get('color_count', 0)}")
                
                features = analysis.get('color_features', {})
                print(f"  Warm/Cool: {'Warm' if features.get('is_warm') else 'Cool'}")
                print(f"  Saturation: {'High' if features.get('is_saturated') else 'Low'}")
                
                harmony = analysis.get('color_harmony', {})
                print(f"  Color harmony: {harmony.get('harmony_type', 'unknown')} (score: {harmony.get('harmony_score', 0):.2f})")
                
            except Exception as e:
                print(f"  Error: {e}")
    else:
        print("No test images found for color detection testing")
