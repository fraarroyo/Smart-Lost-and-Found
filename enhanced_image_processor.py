#!/usr/bin/env python3
"""
Enhanced Image Processing Pipeline
Integrates R-CNN, RNN, and BERT for comprehensive image analysis
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import BertTokenizer, BertModel
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

class EnhancedImageProcessor:
    """Comprehensive image processing using R-CNN, RNN, and BERT"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.logger = logging.getLogger(__name__)
        
        # Initialize BERT for text analysis
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        
        # Initialize R-CNN (using existing object detector)
        self.rcnn_available = True
        
        # Initialize RNN (using existing image RNN analyzer)
        self.rnn_available = True
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Enhanced analysis categories
        self.enhanced_categories = {
            'colors': ['black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple',
                      'pink', 'brown', 'gray', 'silver', 'gold', 'beige', 'tan', 'navy',
                      'maroon', 'olive', 'teal', 'coral'],
            'materials': ['plastic', 'metal', 'leather', 'fabric', 'wood', 'glass', 'ceramic',
                         'rubber', 'paper', 'cardboard', 'foam', 'silicone', 'carbon_fiber',
                         'aluminum', 'steel'],
            'conditions': ['excellent', 'good', 'fair', 'poor', 'damaged'],
            'sizes': ['extra_small', 'small', 'medium', 'large'],
            'brands': ['apple', 'samsung', 'nike', 'adidas', 'sony', 'lg', 'hp', 'dell',
                      'generic', 'unknown'],
            'styles': ['modern', 'vintage', 'sporty', 'elegant', 'casual', 'formal',
                      'minimalist', 'decorative']
        }
        
    def process_image_comprehensive(self, image_path: str, object_detector, rnn_analyzer, text_analyzer) -> Dict:
        """Process image using R-CNN, RNN, and optional BERT for comprehensive analysis"""
        try:
            print(f"[ENHANCED PROCESSING] Starting comprehensive analysis of {image_path}")
            
            # Step 1: R-CNN Object Detection
            print("   🔍 Step 1: R-CNN Object Detection...")
            rcnn_results = self._rcnn_analysis(image_path, object_detector)
            print(f"   ✅ R-CNN detected {len(rcnn_results.get('objects', []))} objects")
            
            # Step 2: RNN Image Detail Analysis
            print("   🧠 Step 2: RNN Image Detail Analysis...")
            rnn_results = self._rnn_analysis(image_path, rnn_analyzer)
            print(f"   ✅ RNN analysis completed with confidence {rnn_results.get('confidence', 0):.2f}")
            
            # Step 3: BERT Text Analysis (always run in comprehensive mode)
            print("   📝 Step 3: BERT Text Analysis...")
            bert_results = self._bert_analysis(image_path, text_analyzer)
            print(f"   ✅ BERT analysis completed with confidence {bert_results.get('text_confidence', 0):.2f}")
            
            # Step 4: Fusion and Integration
            print("   🔗 Step 4: Multi-Modal Fusion...")
            fused_results = self._fuse_analysis_results(rcnn_results, rnn_results, bert_results)
            print(f"   ✅ Multi-modal fusion completed")
            
            # Step 5: Enhanced Description Generation
            print("   ✍️  Step 5: Enhanced Description Generation...")
            enhanced_description = self._generate_enhanced_description(fused_results)
            print(f"   ✅ Enhanced description generated")
            
            return {
                'rcnn_analysis': rcnn_results,
                'rnn_analysis': rnn_results,
                'bert_analysis': bert_results,
                'fused_analysis': fused_results,
                'enhanced_description': enhanced_description,
                'processing_confidence': self._calculate_overall_confidence(fused_results),
                'processing_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive image processing: {e}")
            return {
                'error': str(e),
                'rcnn_analysis': {},
                'rnn_analysis': {},
                'bert_analysis': {},
                'fused_analysis': {},
                'enhanced_description': "Error in processing",
                'processing_confidence': 0.0
            }
    
    def process_image_fast(self, image_path: str, object_detector) -> Dict:
        """Fast image processing using only R-CNN for quick analysis"""
        try:
            print(f"[FAST PROCESSING] Starting fast analysis of {image_path}")
            
            # Only R-CNN Object Detection
            print("   🔍 R-CNN Object Detection...")
            rcnn_results = self._rcnn_analysis(image_path, object_detector)
            print(f"   ✅ R-CNN detected {len(rcnn_results.get('objects', []))} objects")
            
            # Generate basic description from R-CNN results only
            enhanced_description = self._generate_basic_description(rcnn_results)
            
            return {
                'rcnn_analysis': rcnn_results,
                'rnn_analysis': {'details': {}, 'confidence': 0.0},
                'bert_analysis': {'description': '', 'text_confidence': 0.0},
                'fused_analysis': {'object_identification': rcnn_results},
                'enhanced_description': enhanced_description,
                'processing_confidence': rcnn_results.get('object_confidence', 0.0),
                'processing_timestamp': datetime.utcnow().isoformat(),
                'processing_mode': 'fast'
            }
            
        except Exception as e:
            self.logger.error(f"Error in fast image processing: {e}")
            return {
                'error': str(e),
                'rcnn_analysis': {},
                'rnn_analysis': {},
                'bert_analysis': {},
                'fused_analysis': {},
                'enhanced_description': "Fast processing failed",
                'processing_confidence': 0.0,
                'processing_mode': 'fast'
            }
    
    def _rcnn_analysis(self, image_path: str, object_detector) -> Dict:
        """R-CNN object detection analysis"""
        try:
            # Use existing object detector
            detected_objects = object_detector.detect_objects(image_path)
            
            # Enhanced R-CNN analysis
            rcnn_analysis = {
                'objects': detected_objects,
                'object_count': len(detected_objects),
                'primary_object': None,
                'object_confidence': 0.0,
                'spatial_analysis': self._analyze_spatial_relationships(detected_objects),
                'size_analysis': self._analyze_object_sizes(detected_objects, image_path)
            }
            
            if detected_objects:
                # Find primary object (highest confidence)
                primary_obj = max(detected_objects, key=lambda x: x.get('confidence', 0))
                rcnn_analysis['primary_object'] = primary_obj
                rcnn_analysis['object_confidence'] = primary_obj.get('confidence', 0)
            
            return rcnn_analysis
            
        except Exception as e:
            self.logger.error(f"R-CNN analysis error: {e}")
            return {'objects': [], 'error': str(e)}
    
    def _rnn_analysis(self, image_path: str, rnn_analyzer) -> Dict:
        """RNN-based image detail analysis"""
        try:
            # Use existing RNN analyzer
            rnn_results = rnn_analyzer.analyze_image_details(image_path)
            
            # Enhanced RNN analysis
            enhanced_rnn = {
                'details': rnn_results.get('details', {}),
                'caption': rnn_results.get('caption', ''),
                'confidence': rnn_results.get('confidence', 0.0),
                'feature_analysis': self._analyze_rnn_features(rnn_results),
                'detail_confidence': self._calculate_detail_confidence(rnn_results)
            }
            
            return enhanced_rnn
            
        except Exception as e:
            self.logger.error(f"RNN analysis error: {e}")
            return {'details': {}, 'error': str(e)}
    
    def _bert_analysis(self, image_path: str, text_analyzer) -> Dict:
        """BERT-based text analysis of image context"""
        try:
            # Generate text description from image
            image_description = self._generate_image_description(image_path)
            
            # Use BERT to analyze the description
            bert_embedding = text_analyzer.analyze_text(image_description)
            
            # Enhanced BERT analysis
            bert_analysis = {
                'description': image_description,
                'embedding': bert_embedding.tolist() if hasattr(bert_embedding, 'tolist') else bert_embedding,
                'semantic_analysis': self._analyze_semantic_content(image_description),
                'text_confidence': self._calculate_text_confidence(image_description),
                'keyword_extraction': self._extract_keywords(image_description)
            }
            
            return bert_analysis
            
        except Exception as e:
            self.logger.error(f"BERT analysis error: {e}")
            return {'description': '', 'error': str(e)}
    
    def _fuse_analysis_results(self, rcnn_results: Dict, rnn_results: Dict, bert_results: Dict) -> Dict:
        """Fuse results from R-CNN, RNN, and BERT for comprehensive analysis"""
        try:
            fused_analysis = {
                'object_identification': self._fuse_object_identification(rcnn_results, rnn_results),
                'attribute_analysis': self._fuse_attribute_analysis(rcnn_results, rnn_results, bert_results),
                'confidence_scoring': self._fuse_confidence_scores(rcnn_results, rnn_results, bert_results),
                'semantic_understanding': self._fuse_semantic_understanding(rnn_results, bert_results),
                'recommendations': self._generate_recommendations(rcnn_results, rnn_results, bert_results)
            }
            
            return fused_analysis
            
        except Exception as e:
            self.logger.error(f"Fusion error: {e}")
            return {'error': str(e)}
    
    def _generate_enhanced_description(self, fused_results: Dict) -> str:
        """Generate enhanced description using all analysis results"""
        try:
            description_parts = []
            
            # Object identification
            obj_info = fused_results.get('object_identification', {})
            if obj_info.get('primary_object'):
                obj_class = obj_info['primary_object'].get('class', 'item')
                confidence = obj_info['primary_object'].get('confidence', 0)
                description_parts.append(f"{obj_class.title()} (confidence: {confidence:.1%})")
            
            # Attribute analysis
            attrs = fused_results.get('attribute_analysis', {})
            if attrs.get('colors'):
                color_str = ', '.join(attrs['colors'][:2])
                description_parts.append(f"{color_str} colored")
            
            if attrs.get('materials'):
                material_str = attrs['materials'][0]
                description_parts.append(f"{material_str} material")
            
            if attrs.get('condition'):
                condition_str = attrs['condition']
                description_parts.append(f"in {condition_str} condition")
            
            if attrs.get('size'):
                size_str = attrs['size'].replace('_', ' ')
                description_parts.append(f"{size_str} size")
            
            # Semantic understanding
            semantic = fused_results.get('semantic_understanding', {})
            if semantic.get('style'):
                style_str = semantic['style']
                description_parts.append(f"in {style_str} style")
            
            # Combine all parts
            if description_parts:
                enhanced_desc = " ".join(description_parts)
            else:
                enhanced_desc = "Comprehensive analysis completed"
            
            # Add confidence information
            confidence = fused_results.get('confidence_scoring', {}).get('overall_confidence', 0)
            enhanced_desc += f". Overall analysis confidence: {confidence:.1%}"
            
            return enhanced_desc
            
        except Exception as e:
            self.logger.error(f"Description generation error: {e}")
            return "Enhanced description generation failed"
    
    def _analyze_spatial_relationships(self, objects: List[Dict]) -> Dict:
        """Analyze spatial relationships between objects"""
        if len(objects) < 2:
            return {'relationship': 'single_object', 'complexity': 'low'}
        
        # Simple spatial analysis
        boxes = [obj.get('box', [0, 0, 0, 0]) for obj in objects if obj.get('box')]
        if len(boxes) < 2:
            return {'relationship': 'single_object', 'complexity': 'low'}
        
        # Calculate overlap and positioning
        overlaps = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if self._boxes_overlap(boxes[i], boxes[j]):
                    overlaps += 1
        
        return {
            'relationship': 'multiple_objects',
            'complexity': 'high' if overlaps > 0 else 'medium',
            'overlap_count': overlaps,
            'object_count': len(objects)
        }
    
    def _analyze_object_sizes(self, objects: List[Dict], image_path: str) -> Dict:
        """Analyze object sizes relative to image"""
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
            img.close()
            
            size_analysis = {
                'image_dimensions': (img_width, img_height),
                'object_sizes': [],
                'size_distribution': 'unknown'
            }
            
            for obj in objects:
                if obj.get('box'):
                    x1, y1, x2, y2 = map(int, obj['box'])
                    obj_width = x2 - x1
                    obj_height = y2 - y1
                    obj_area = obj_width * obj_height
                    img_area = img_width * img_height
                    relative_size = obj_area / img_area if img_area > 0 else 0
                    
                    size_analysis['object_sizes'].append({
                        'class': obj.get('class', 'unknown'),
                        'width': obj_width,
                        'height': obj_height,
                        'area': obj_area,
                        'relative_size': relative_size
                    })
            
            # Determine size distribution
            if size_analysis['object_sizes']:
                avg_relative_size = sum(s['relative_size'] for s in size_analysis['object_sizes']) / len(size_analysis['object_sizes'])
                if avg_relative_size > 0.5:
                    size_analysis['size_distribution'] = 'large_objects'
                elif avg_relative_size > 0.2:
                    size_analysis['size_distribution'] = 'medium_objects'
                else:
                    size_analysis['size_distribution'] = 'small_objects'
            
            return size_analysis
            
        except Exception as e:
            self.logger.error(f"Size analysis error: {e}")
            return {'error': str(e)}
    
    def _analyze_rnn_features(self, rnn_results: Dict) -> Dict:
        """Analyze RNN feature extraction quality"""
        details = rnn_results.get('details', {})
        
        feature_analysis = {
            'color_confidence': details.get('color_confidence', [0.0])[0] if details.get('color_confidence') else 0.0,
            'material_confidence': details.get('material_confidence', [0.0])[0] if details.get('material_confidence') else 0.0,
            'condition_confidence': details.get('condition_confidence', 0.0),
            'size_confidence': details.get('size_confidence', 0.0),
            'overall_feature_quality': 'unknown'
        }
        
        # Calculate overall feature quality
        confidences = [
            feature_analysis['color_confidence'],
            feature_analysis['material_confidence'],
            feature_analysis['condition_confidence'],
            feature_analysis['size_confidence']
        ]
        avg_confidence = sum(confidences) / len(confidences)
        
        if avg_confidence > 0.8:
            feature_analysis['overall_feature_quality'] = 'excellent'
        elif avg_confidence > 0.6:
            feature_analysis['overall_feature_quality'] = 'good'
        elif avg_confidence > 0.4:
            feature_analysis['overall_feature_quality'] = 'fair'
        else:
            feature_analysis['overall_feature_quality'] = 'poor'
        
        return feature_analysis
    
    def _calculate_detail_confidence(self, rnn_results: Dict) -> float:
        """Calculate confidence in RNN detail analysis"""
        details = rnn_results.get('details', {})
        confidences = []
        
        for key in ['color_confidence', 'material_confidence', 'condition_confidence', 'size_confidence']:
            if key in details:
                if isinstance(details[key], list):
                    confidences.extend(details[key])
                else:
                    confidences.append(details[key])
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _generate_image_description(self, image_path: str) -> str:
        """Generate text description from image for BERT analysis"""
        try:
            # Simple image description generation
            # In practice, you might use a more sophisticated image captioning model
            img = Image.open(image_path)
            width, height = img.size
            
            # Basic description based on image properties
            description = f"Image with dimensions {width}x{height} pixels"
            
            # Add color analysis
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                avg_color = np.mean(img_array, axis=(0, 1))
                if avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
                    description += ", predominantly red tones"
                elif avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
                    description += ", predominantly green tones"
                elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
                    description += ", predominantly blue tones"
                else:
                    description += ", mixed color tones"
            
            img.close()
            return description
            
        except Exception as e:
            self.logger.error(f"Image description generation error: {e}")
            return "Image analysis in progress"
    
    def _analyze_semantic_content(self, description: str) -> Dict:
        """Analyze semantic content using BERT"""
        try:
            # Simple semantic analysis
            words = description.lower().split()
            
            semantic_analysis = {
                'word_count': len(words),
                'complexity': 'low' if len(words) < 10 else 'medium' if len(words) < 20 else 'high',
                'color_mentions': sum(1 for word in words if word in self.enhanced_categories['colors']),
                'material_mentions': sum(1 for word in words if word in self.enhanced_categories['materials']),
                'size_mentions': sum(1 for word in words if word in self.enhanced_categories['sizes'])
            }
            
            return semantic_analysis
            
        except Exception as e:
            self.logger.error(f"Semantic analysis error: {e}")
            return {'error': str(e)}
    
    def _calculate_text_confidence(self, description: str) -> float:
        """Calculate confidence in text analysis"""
        if not description or description == "Image analysis in progress":
            return 0.0
        
        # Simple confidence based on description length and content
        word_count = len(description.split())
        if word_count < 5:
            return 0.3
        elif word_count < 15:
            return 0.6
        else:
            return 0.8
    
    def _extract_keywords(self, description: str) -> List[str]:
        """Extract keywords from description"""
        words = description.lower().split()
        keywords = []
        
        for word in words:
            if word in self.enhanced_categories['colors'] or \
               word in self.enhanced_categories['materials'] or \
               word in self.enhanced_categories['sizes']:
                keywords.append(word)
        
        return list(set(keywords))
    
    def _fuse_object_identification(self, rcnn_results: Dict, rnn_results: Dict) -> Dict:
        """Fuse object identification from R-CNN and RNN"""
        rcnn_objects = rcnn_results.get('objects', [])
        rnn_details = rnn_results.get('details', {})
        
        fused_identification = {
            'primary_object': rcnn_results.get('primary_object'),
            'object_count': len(rcnn_objects),
            'rnn_enhanced': bool(rnn_details),
            'confidence': rcnn_results.get('object_confidence', 0)
        }
        
        return fused_identification
    
    def _fuse_attribute_analysis(self, rcnn_results: Dict, rnn_results: Dict, bert_results: Dict) -> Dict:
        """Fuse attribute analysis from all models"""
        rnn_details = rnn_results.get('details', {})
        bert_keywords = bert_results.get('keyword_extraction', [])
        
        fused_attributes = {
            'colors': rnn_details.get('colors', []),
            'materials': rnn_details.get('materials', []),
            'condition': rnn_details.get('condition', 'unknown'),
            'size': rnn_details.get('size', 'unknown'),
            'brands': rnn_details.get('brands', []),
            'styles': rnn_details.get('styles', []),
            'keywords': bert_keywords
        }
        
        return fused_attributes
    
    def _fuse_confidence_scores(self, rcnn_results: Dict, rnn_results: Dict, bert_results: Dict) -> Dict:
        """Fuse confidence scores from all models"""
        rcnn_conf = rcnn_results.get('object_confidence', 0)
        rnn_conf = rnn_results.get('confidence', 0)
        bert_conf = bert_results.get('text_confidence', 0)
        
        # Weighted average (R-CNN: 40%, RNN: 40%, BERT: 20%)
        overall_confidence = (rcnn_conf * 0.4 + rnn_conf * 0.4 + bert_conf * 0.2)
        
        return {
            'rcnn_confidence': rcnn_conf,
            'rnn_confidence': rnn_conf,
            'bert_confidence': bert_conf,
            'overall_confidence': overall_confidence
        }
    
    def _fuse_semantic_understanding(self, rnn_results: Dict, bert_results: Dict) -> Dict:
        """Fuse semantic understanding from RNN and BERT"""
        rnn_details = rnn_results.get('details', {})
        bert_semantic = bert_results.get('semantic_analysis', {})
        
        return {
            'style': rnn_details.get('styles', ['unknown'])[0] if rnn_details.get('styles') else 'unknown',
            'complexity': bert_semantic.get('complexity', 'unknown'),
            'semantic_richness': bert_semantic.get('word_count', 0)
        }
    
    def _generate_recommendations(self, rcnn_results: Dict, rnn_results: Dict, bert_results: Dict) -> List[str]:
        """Generate recommendations based on all analysis results"""
        recommendations = []
        
        # R-CNN based recommendations
        if rcnn_results.get('object_count', 0) > 1:
            recommendations.append("Multiple objects detected - consider separate analysis")
        
        # RNN based recommendations
        rnn_details = rnn_results.get('details', {})
        if rnn_details.get('condition') == 'damaged':
            recommendations.append("Item appears damaged - handle with care")
        
        # BERT based recommendations
        bert_semantic = bert_results.get('semantic_analysis', {})
        if bert_semantic.get('complexity') == 'high':
            recommendations.append("Complex image - detailed analysis recommended")
        
        return recommendations
    
    def _calculate_overall_confidence(self, fused_results: Dict) -> float:
        """Calculate overall processing confidence"""
        confidence_scoring = fused_results.get('confidence_scoring', {})
        return confidence_scoring.get('overall_confidence', 0.0)
    
    def _boxes_overlap(self, box1: List[int], box2: List[int]) -> bool:
        """Check if two bounding boxes overlap"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
    
    def _generate_basic_description(self, rcnn_results: Dict) -> str:
        """Generate basic description from R-CNN results only"""
        try:
            objects = rcnn_results.get('objects', [])
            if not objects:
                return "No objects detected in the image"
            
            # Get the primary object (highest confidence)
            primary_obj = max(objects, key=lambda x: x.get('confidence', 0))
            obj_class = primary_obj.get('class', 'object')
            confidence = primary_obj.get('confidence', 0)
            
            # Generate simple description
            description = f"Detected {obj_class} with {confidence:.1%} confidence"
            
            # Add additional objects if present
            if len(objects) > 1:
                other_objects = [obj for obj in objects if obj != primary_obj]
                other_classes = [obj.get('class', 'object') for obj in other_objects[:2]]  # Limit to 2 additional
                if other_classes:
                    description += f". Also detected: {', '.join(other_classes)}"
            
            return description
            
        except Exception as e:
            self.logger.error(f"Error generating basic description: {e}")
            return "Basic description generation failed"
    
    def process_image_ultra_fast(self, image_path: str, object_detector) -> Dict:
        """Ultra-fast image processing using only basic R-CNN detection"""
        try:
            print(f"[ULTRA-FAST PROCESSING] Starting ultra-fast analysis of {image_path}")
            
            # Only basic R-CNN Object Detection (no enhanced color analysis)
            print("   🔍 Basic R-CNN Object Detection...")
            detected_objects = object_detector.detect_objects(image_path)
            
            # Skip enhanced color analysis for speed
            for obj in detected_objects:
                obj['enhanced_color'] = None
                obj['color_features'] = None
                obj['color_harmony'] = None
            
            # Generate minimal description
            if detected_objects:
                primary_obj = max(detected_objects, key=lambda x: x.get('confidence', 0))
                obj_class = primary_obj.get('class', 'object')
                confidence = primary_obj.get('confidence', 0)
                enhanced_description = f"Detected {obj_class} with {confidence:.1%} confidence"
            else:
                enhanced_description = "No objects detected in the image"
            
            return {
                'rcnn_analysis': {
                    'objects': detected_objects,
                    'object_count': len(detected_objects),
                    'primary_object': detected_objects[0] if detected_objects else None,
                    'object_confidence': detected_objects[0].get('confidence', 0) if detected_objects else 0
                },
                'rnn_analysis': {'details': {}, 'confidence': 0.0},
                'bert_analysis': {'description': '', 'text_confidence': 0.0},
                'fused_analysis': {'object_identification': {'objects': detected_objects}},
                'enhanced_description': enhanced_description,
                'processing_confidence': detected_objects[0].get('confidence', 0) if detected_objects else 0.0,
                'processing_timestamp': datetime.utcnow().isoformat(),
                'processing_mode': 'ultra_fast'
            }
            
        except Exception as e:
            self.logger.error(f"Error in ultra-fast image processing: {e}")
            return {
                'error': str(e),
                'rcnn_analysis': {'objects': []},
                'rnn_analysis': {},
                'bert_analysis': {},
                'fused_analysis': {},
                'enhanced_description': "Ultra-fast processing failed",
                'processing_confidence': 0.0,
                'processing_mode': 'ultra_fast'
            }

# Global enhanced image processor instance
enhanced_image_processor = EnhancedImageProcessor(device='cuda' if torch.cuda.is_available() else 'cpu')
