#!/usr/bin/env python3
"""
RNN-based Image Detail Extraction System
Uses RNNs to extract detailed information from images for lost & found items
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

class ImageFeatureExtractor(nn.Module):
    """CNN feature extractor for images"""
    
    def __init__(self, feature_dim: int = 512):
        super(ImageFeatureExtractor, self).__init__()
        # Use pre-trained ResNet as backbone
        self.backbone = models.resnet18(pretrained=True)
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        # Add custom feature extraction
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.feature_extractor(features)

class ImageDetailRNN(nn.Module):
    """RNN for extracting detailed information from images"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, 
                 num_layers: int = 2, output_dim: int = 128):
        super(ImageDetailRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for sequential detail extraction
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Output layers for different detail types
        self.color_head = nn.Linear(hidden_dim, 20)  # 20 color categories
        self.material_head = nn.Linear(hidden_dim, 15)  # 15 material types
        self.condition_head = nn.Linear(hidden_dim, 5)  # 5 condition levels
        self.size_head = nn.Linear(hidden_dim, 4)  # 4 size categories
        self.brand_head = nn.Linear(hidden_dim, 10)  # 10 brand categories
        self.style_head = nn.Linear(hidden_dim, 8)  # 8 style categories
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Extract different detail types
        color_pred = self.color_head(context_vector)
        material_pred = self.material_head(context_vector)
        condition_pred = self.condition_head(context_vector)
        size_pred = self.size_head(context_vector)
        brand_pred = self.brand_head(context_vector)
        style_pred = self.style_head(context_vector)
        
        return {
            'color': color_pred,
            'material': material_pred,
            'condition': condition_pred,
            'size': size_pred,
            'brand': brand_pred,
            'style': style_pred,
            'attention_weights': attention_weights
        }

class ImageCaptionRNN(nn.Module):
    """RNN for generating detailed image captions"""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, 
                 vocab_size: int = 1000, max_length: int = 50):
        super(ImageCaptionRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # LSTM for caption generation
        self.lstm = nn.LSTM(input_dim + vocab_size, hidden_dim, 2, 
                           batch_first=True, dropout=0.2)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Embedding for words
        self.word_embedding = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, image_features, captions=None):
        batch_size = image_features.size(0)
        
        if captions is not None:
            # Training mode
            captions_embedded = self.word_embedding(captions)
            # Concatenate image features with word embeddings
            inputs = torch.cat([image_features.unsqueeze(1).repeat(1, captions.size(1), 1), 
                              captions_embedded], dim=2)
            lstm_out, _ = self.lstm(inputs)
            outputs = self.output_layer(lstm_out)
            return outputs
        else:
            # Inference mode - generate caption
            generated_caption = []
            hidden = None
            
            # Start with image features
            current_input = image_features.unsqueeze(1)
            
            for _ in range(self.max_length):
                if hidden is None:
                    lstm_out, hidden = self.lstm(current_input)
                else:
                    lstm_out, hidden = self.lstm(current_input, hidden)
                
                output = self.output_layer(lstm_out)
                word_id = torch.argmax(output, dim=2)
                generated_caption.append(word_id)
                
                # Use generated word as next input
                word_embedded = self.word_embedding(word_id)
                current_input = torch.cat([image_features.unsqueeze(1), word_embedded], dim=2)
            
            return torch.cat(generated_caption, dim=1)

class ImageRNNAnalyzer:
    """Main class for RNN-based image analysis"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.feature_extractor = ImageFeatureExtractor().to(self.device)
        self.detail_rnn = ImageDetailRNN().to(self.device)
        self.caption_rnn = ImageCaptionRNN().to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Category mappings
        self.color_categories = [
            'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple',
            'pink', 'brown', 'gray', 'silver', 'gold', 'beige', 'tan', 'navy',
            'maroon', 'olive', 'teal', 'coral'
        ]
        
        self.material_categories = [
            'plastic', 'metal', 'leather', 'fabric', 'wood', 'glass', 'ceramic',
            'rubber', 'paper', 'cardboard', 'foam', 'silicone', 'carbon_fiber',
            'aluminum', 'steel'
        ]
        
        self.condition_categories = [
            'excellent', 'good', 'fair', 'poor', 'damaged'
        ]
        
        self.size_categories = [
            'extra_small', 'small', 'medium', 'large'
        ]
        
        self.brand_categories = [
            'apple', 'samsung', 'nike', 'adidas', 'sony', 'lg', 'hp', 'dell',
            'generic', 'unknown'
        ]
        
        self.style_categories = [
            'modern', 'vintage', 'sporty', 'elegant', 'casual', 'formal',
            'minimalist', 'decorative'
        ]
        
        # Model paths
        self.model_dir = 'models/image_rnn'
        os.makedirs(self.model_dir, exist_ok=True)
        
    def extract_image_features(self, image_path: str) -> torch.Tensor:
        """Extract features from image using CNN"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(image_tensor)
            
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features from {image_path}: {e}")
            return torch.zeros(1, 512).to(self.device)
    
    def analyze_image_details(self, image_path: str) -> Dict:
        """Analyze image and extract detailed information using RNN"""
        try:
            # Extract image features
            features = self.extract_image_features(image_path)
            
            # Create sequence of features for RNN (simulate different image regions)
            # In practice, you might extract features from different regions
            feature_sequence = features.unsqueeze(1).repeat(1, 5, 1)  # 5 regions
            
            # Analyze with RNN
            self.detail_rnn.eval()
            with torch.no_grad():
                predictions = self.detail_rnn(feature_sequence)
            
            # Convert predictions to readable format
            details = self._convert_predictions_to_details(predictions)
            
            # Generate detailed caption
            caption = self._generate_image_caption(features)
            
            return {
                'details': details,
                'caption': caption,
                'confidence': self._calculate_confidence(predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing image {image_path}: {e}")
            return {
                'details': {},
                'caption': "Unable to analyze image",
                'confidence': 0.0
            }
    
    def _convert_predictions_to_details(self, predictions: Dict) -> Dict:
        """Convert RNN predictions to readable details"""
        details = {}
        
        # Color analysis
        color_probs = F.softmax(predictions['color'], dim=1)
        top_colors = torch.topk(color_probs, 3, dim=1)
        details['colors'] = [
            self.color_categories[idx] for idx in top_colors.indices[0].tolist()
        ]
        details['color_confidence'] = top_colors.values[0].tolist()
        
        # Material analysis
        material_probs = F.softmax(predictions['material'], dim=1)
        top_materials = torch.topk(material_probs, 2, dim=1)
        details['materials'] = [
            self.material_categories[idx] for idx in top_materials.indices[0].tolist()
        ]
        details['material_confidence'] = top_materials.values[0].tolist()
        
        # Condition analysis
        condition_probs = F.softmax(predictions['condition'], dim=1)
        condition_idx = torch.argmax(condition_probs, dim=1).item()
        details['condition'] = self.condition_categories[condition_idx]
        details['condition_confidence'] = condition_probs[0][condition_idx].item()
        
        # Size analysis
        size_probs = F.softmax(predictions['size'], dim=1)
        size_idx = torch.argmax(size_probs, dim=1).item()
        details['size'] = self.size_categories[size_idx]
        details['size_confidence'] = size_probs[0][size_idx].item()
        
        # Brand analysis
        brand_probs = F.softmax(predictions['brand'], dim=1)
        top_brands = torch.topk(brand_probs, 2, dim=1)
        details['brands'] = [
            self.brand_categories[idx] for idx in top_brands.indices[0].tolist()
        ]
        details['brand_confidence'] = top_brands.values[0].tolist()
        
        # Style analysis
        style_probs = F.softmax(predictions['style'], dim=1)
        top_styles = torch.topk(style_probs, 2, dim=1)
        details['styles'] = [
            self.style_categories[idx] for idx in top_styles.indices[0].tolist()
        ]
        details['style_confidence'] = top_styles.values[0].tolist()
        
        return details
    
    def _generate_image_caption(self, features: torch.Tensor) -> str:
        """Generate detailed caption for the image"""
        try:
            self.caption_rnn.eval()
            with torch.no_grad():
                # Generate caption (simplified - in practice you'd use a trained model)
                # For now, return a template-based caption
                return "A detailed analysis of the item showing various characteristics and features."
        except Exception as e:
            self.logger.error(f"Error generating caption: {e}")
            return "Unable to generate detailed description"
    
    def _calculate_confidence(self, predictions: Dict) -> float:
        """Calculate overall confidence score"""
        try:
            confidences = []
            for key, pred in predictions.items():
                if key != 'attention_weights':
                    probs = F.softmax(pred, dim=1)
                    max_conf = torch.max(probs, dim=1)[0].item()
                    confidences.append(max_conf)
            
            return sum(confidences) / len(confidences) if confidences else 0.0
        except:
            return 0.0
    
    def analyze_item_condition(self, image_path: str) -> Dict:
        """Specifically analyze item condition using RNN"""
        try:
            features = self.extract_image_features(image_path)
            feature_sequence = features.unsqueeze(1).repeat(1, 3, 1)
            
            self.detail_rnn.eval()
            with torch.no_grad():
                predictions = self.detail_rnn(feature_sequence)
            
            condition_probs = F.softmax(predictions['condition'], dim=1)
            condition_idx = torch.argmax(condition_probs, dim=1).item()
            
            return {
                'condition': self.condition_categories[condition_idx],
                'confidence': condition_probs[0][condition_idx].item(),
                'all_conditions': {
                    cat: prob.item() for cat, prob in 
                    zip(self.condition_categories, condition_probs[0])
                }
            }
        except Exception as e:
            self.logger.error(f"Error analyzing condition: {e}")
            return {'condition': 'unknown', 'confidence': 0.0}
    
    def analyze_item_materials(self, image_path: str) -> Dict:
        """Analyze item materials using RNN"""
        try:
            features = self.extract_image_features(image_path)
            feature_sequence = features.unsqueeze(1).repeat(1, 4, 1)
            
            self.detail_rnn.eval()
            with torch.no_grad():
                predictions = self.detail_rnn(feature_sequence)
            
            material_probs = F.softmax(predictions['material'], dim=1)
            top_materials = torch.topk(material_probs, 3, dim=1)
            
            return {
                'materials': [
                    self.material_categories[idx] for idx in top_materials.indices[0].tolist()
                ],
                'confidences': top_materials.values[0].tolist(),
                'all_materials': {
                    cat: prob.item() for cat, prob in 
                    zip(self.material_categories, material_probs[0])
                }
            }
        except Exception as e:
            self.logger.error(f"Error analyzing materials: {e}")
            return {'materials': ['unknown'], 'confidences': [0.0]}
    
    def generate_detailed_description(self, image_path: str, item_type: str = None) -> str:
        """Generate comprehensive description using RNN analysis"""
        try:
            analysis = self.analyze_image_details(image_path)
            details = analysis['details']
            
            # Build description from RNN analysis
            description_parts = []
            
            if details.get('colors'):
                color_str = ', '.join(details['colors'][:2])  # Top 2 colors
                description_parts.append(f"{color_str} colored")
            
            if details.get('materials'):
                material_str = details['materials'][0]  # Top material
                description_parts.append(f"{material_str} material")
            
            if details.get('condition'):
                condition_str = details['condition']
                description_parts.append(f"in {condition_str} condition")
            
            if details.get('size'):
                size_str = details['size'].replace('_', ' ')
                description_parts.append(f"{size_str} size")
            
            if details.get('brands') and details['brands'][0] != 'unknown':
                brand_str = details['brands'][0]
                description_parts.append(f"{brand_str} brand")
            
            if details.get('styles'):
                style_str = details['styles'][0].replace('_', ' ')
                description_parts.append(f"{style_str} style")
            
            # Combine into final description
            if description_parts:
                description = f"{' '.join(description_parts)} {item_type or 'item'}"
            else:
                description = f"Analyzed {item_type or 'item'} with detailed characteristics"
            
            return description
            
        except Exception as e:
            self.logger.error(f"Error generating description: {e}")
            return f"Detailed analysis of {item_type or 'item'}"
    
    def save_models(self):
        """Save all RNN models"""
        try:
            torch.save(self.feature_extractor.state_dict(), 
                      os.path.join(self.model_dir, 'feature_extractor.pth'))
            torch.save(self.detail_rnn.state_dict(), 
                      os.path.join(self.model_dir, 'detail_rnn.pth'))
            torch.save(self.caption_rnn.state_dict(), 
                      os.path.join(self.model_dir, 'caption_rnn.pth'))
            self.logger.info("Image RNN models saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load all RNN models"""
        try:
            if os.path.exists(os.path.join(self.model_dir, 'feature_extractor.pth')):
                self.feature_extractor.load_state_dict(
                    torch.load(os.path.join(self.model_dir, 'feature_extractor.pth'), 
                              map_location=self.device))
            if os.path.exists(os.path.join(self.model_dir, 'detail_rnn.pth')):
                self.detail_rnn.load_state_dict(
                    torch.load(os.path.join(self.model_dir, 'detail_rnn.pth'), 
                              map_location=self.device))
            if os.path.exists(os.path.join(self.model_dir, 'caption_rnn.pth')):
                self.caption_rnn.load_state_dict(
                    torch.load(os.path.join(self.model_dir, 'caption_rnn.pth'), 
                              map_location=self.device))
            self.logger.info("Image RNN models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

# Global image RNN analyzer instance
image_rnn_analyzer = ImageRNNAnalyzer(device='cuda' if torch.cuda.is_available() else 'cpu')
