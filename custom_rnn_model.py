#!/usr/bin/env python3
"""
Custom RNN Model for BARYONYX Lost & Found System
Advanced RNN architecture specifically designed for item matching and user behavior prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict, deque
import pickle
import math
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, jaccard_score
)
import matplotlib.pyplot as plt
import seaborn as sns

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for better feature extraction"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ItemMatchingRNN(nn.Module):
    """
    Custom RNN model for lost & found item matching
    Combines LSTM, attention mechanisms, and transformer-like features
    """
    
    def __init__(self, 
                 input_size: int = 40,  # Changed from 20 to 40 for combined features
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 output_size: int = 2,  # Binary classification: match/no match
                 dropout: float = 0.2,
                 use_attention: bool = True,
                 use_positional_encoding: bool = True):
        super(ItemMatchingRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_positional_encoding = use_positional_encoding
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(hidden_size)
        
        # LSTM layers with residual connections
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=dropout if i < num_layers-1 else 0)
            for i in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Multi-head attention
        if use_attention:
            self.attention = MultiHeadAttention(hidden_size, num_heads=8, dropout=dropout)
            self.attention_norm = nn.LayerNorm(hidden_size)
        
        # Output layers with residual connections
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, 0, 0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size) or (batch_size, input_size)
            return_attention: Whether to return attention weights
        
        Returns:
            output: Model predictions
            attention_weights: Attention weights (if return_attention=True)
        """
        # Handle both 2D and 3D inputs
        if len(x.size()) == 2:
            # 2D input: (batch_size, input_size) -> (batch_size, 1, input_size)
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_projection(x)
        
        # Positional encoding
        if self.use_positional_encoding:
            x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_size)
        
        # LSTM layers with residual connections
        lstm_output = x
        for i, (lstm, layer_norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            residual = lstm_output
            lstm_out, _ = lstm(lstm_output)
            lstm_output = layer_norm(lstm_out + residual)  # Residual connection
        
        # Multi-head attention
        attention_weights = None
        if self.use_attention:
            attn_output, attention_weights = self.attention(
                lstm_output, lstm_output, lstm_output
            )
            lstm_output = self.attention_norm(lstm_output + attn_output)
        
        # Global average pooling
        pooled_output = torch.mean(lstm_output, dim=1)  # (batch_size, hidden_size)
        
        # Output layers
        output = self.output_layers(pooled_output)
        
        if return_attention:
            return output, attention_weights
        return output

class UserBehaviorPredictor(nn.Module):
    """
    Advanced user behavior prediction model
    Uses GRU with attention and temporal features
    """
    
    def __init__(self, 
                 feature_size: int = 15,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 5,
                 dropout: float = 0.3):
        super(UserBehaviorPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Feature embedding
        self.feature_embedding = nn.Linear(feature_size, hidden_size)
        
        # GRU layers
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x, return_attention=False):
        """
        Forward pass for user behavior prediction
        
        Args:
            x: Input features (batch_size, seq_len, feature_size) or (batch_size, feature_size)
            return_attention: Whether to return attention weights
        
        Returns:
            output: Behavior predictions
            attention_weights: Attention weights (if return_attention=True)
        """
        # Handle both 2D and 3D inputs
        if len(x.size()) == 2:
            # 2D input: (batch_size, feature_size) -> (batch_size, 1, feature_size)
            x = x.unsqueeze(1)
        
        # Feature embedding
        embedded = self.feature_embedding(x)
        
        # GRU processing
        gru_out, _ = self.gru(embedded)
        
        # Attention mechanism
        attention_scores = self.attention(gru_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        
        # Output prediction
        output = self.output_layers(context_vector)
        
        if return_attention:
            return output, attention_weights.squeeze(-1)
        return output

class TextDescriptionEncoder(nn.Module):
    """
    Advanced text description encoder using bidirectional LSTM with attention
    """
    
    def __init__(self, 
                 vocab_size: int = 5000,
                 embedding_dim: int = 128,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_size: int = 4,
                 dropout: float = 0.3):
        super(TextDescriptionEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                           batch_first=True, bidirectional=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x, return_attention=False):
        """
        Forward pass for text description encoding
        
        Args:
            x: Input text indices (batch_size, seq_len)
            return_attention: Whether to return attention weights
        
        Returns:
            output: Description classifications
            attention_weights: Attention weights (if return_attention=True)
        """
        # Embedding
        embedded = self.embedding(x)
        
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Attention mechanism
        attention_scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output classification
        output = self.output_layers(context_vector)
        
        if return_attention:
            return output, attention_weights.squeeze(-1)
        return output

class CustomRNNManager:
    """
    Manager class for custom RNN models
    Handles training, inference, and model management
    """
    
    def __init__(self, device: str = 'cpu', model_dir: str = 'models/custom_rnn/'):
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.logger = logging.getLogger(__name__)
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models
        self.item_matching_model = ItemMatchingRNN().to(self.device)
        self.behavior_predictor = UserBehaviorPredictor().to(self.device)
        self.text_encoder = TextDescriptionEncoder().to(self.device)
        
        # Data storage
        self.user_sequences = defaultdict(list)
        self.item_descriptions = []
        self.vocab = None
        self.vocab_size = 5000
        
        # Training history
        self.training_history = {
            'item_matching': {'losses': [], 'accuracies': []},
            'behavior': {'losses': [], 'accuracies': []},
            'text': {'losses': [], 'accuracies': []}
        }
    
    def create_item_features(self, item_data: Dict) -> List[float]:
        """
        Create comprehensive feature vector for item matching
        
        Args:
            item_data: Dictionary containing item information
        
        Returns:
            List of normalized features
        """
        features = [0.0] * 20  # 20-dimensional feature vector
        
        # Basic item features
        features[0] = item_data.get('item_type_encoding', 0.0)  # Normalized item type
        features[1] = item_data.get('color_encoding', 0.0)      # Normalized color
        features[2] = item_data.get('size_encoding', 0.0)       # Normalized size
        features[3] = item_data.get('condition_encoding', 0.0)   # Normalized condition
        
        # Location features
        features[4] = item_data.get('location_type_encoding', 0.0)  # Office, home, public, etc.
        features[5] = item_data.get('location_confidence', 0.0)     # Location confidence
        
        # Temporal features
        features[6] = item_data.get('hour_normalized', 0.0)         # Hour of day (0-1)
        features[7] = item_data.get('day_of_week_normalized', 0.0)  # Day of week (0-1)
        features[8] = item_data.get('month_normalized', 0.0)        # Month (0-1)
        features[9] = item_data.get('season_normalized', 0.0)       # Season (0-1)
        
        # User behavior features
        features[10] = item_data.get('user_activity_level', 0.0)    # User activity level
        features[11] = item_data.get('search_frequency', 0.0)       # Search frequency
        features[12] = item_data.get('upload_frequency', 0.0)       # Upload frequency
        
        # Item characteristics
        features[13] = item_data.get('has_image', 0.0)              # Has image (0 or 1)
        features[14] = item_data.get('description_length', 0.0)     # Description length
        features[15] = item_data.get('description_quality', 0.0)    # Description quality score
        
        # Matching features
        features[16] = item_data.get('similarity_score', 0.0)       # Similarity to other items
        features[17] = item_data.get('match_confidence', 0.0)       # Match confidence
        features[18] = item_data.get('category_consistency', 0.0)   # Category consistency
        features[19] = item_data.get('temporal_relevance', 0.0)     # Temporal relevance
        
        return features
    
    def create_user_behavior_features(self, user_data: Dict) -> List[float]:
        """
        Create comprehensive feature vector for user behavior prediction
        
        Args:
            user_data: Dictionary containing user behavior information
        
        Returns:
            List of normalized features
        """
        features = [0.0] * 15  # 15-dimensional feature vector
        
        # Time-based features
        features[0] = user_data.get('hour_normalized', 0.0)
        features[1] = user_data.get('day_of_week_normalized', 0.0)
        features[2] = user_data.get('time_since_last_action', 0.0)
        features[3] = user_data.get('session_duration', 0.0)
        
        # Action features
        features[4] = user_data.get('action_type_encoding', 0.0)
        features[5] = user_data.get('action_frequency', 0.0)
        features[6] = user_data.get('action_consistency', 0.0)
        
        # User profile features
        features[7] = user_data.get('user_experience_level', 0.0)
        features[8] = user_data.get('user_activity_level', 0.0)
        features[9] = user_data.get('user_preference_score', 0.0)
        
        # Context features
        features[10] = user_data.get('device_type_encoding', 0.0)
        features[11] = user_data.get('location_type_encoding', 0.0)
        features[12] = user_data.get('network_quality', 0.0)
        
        # Interaction features
        features[13] = user_data.get('interaction_success_rate', 0.0)
        features[14] = user_data.get('satisfaction_score', 0.0)
        
        return features
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 2) -> Dict[str, int]:
        """
        Build vocabulary from text descriptions
        
        Args:
            texts: List of text descriptions
            min_freq: Minimum frequency for word inclusion
        
        Returns:
            Vocabulary dictionary
        """
        from collections import Counter
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Create vocabulary
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, count in word_counts.items():
            if count >= min_freq and len(vocab) < self.vocab_size - 2:
                vocab[word] = len(vocab)
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        
        return vocab
    
    def text_to_sequence(self, text: str, max_length: int = 30) -> List[int]:
        """
        Convert text to sequence of word indices
        
        Args:
            text: Input text
            max_length: Maximum sequence length
        
        Returns:
            List of word indices
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        words = text.lower().split()
        word_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # Pad or truncate
        if len(word_indices) < max_length:
            word_indices.extend([self.vocab['<PAD>']] * (max_length - len(word_indices)))
        else:
            word_indices = word_indices[:max_length]
        
        return word_indices
    
    def predict_item_match(self, item_features: List[float]) -> Dict:
        """
        Predict item matching probability
        
        Args:
            item_features: Item feature vector
        
        Returns:
            Dictionary with predictions and confidence
        """
        self.item_matching_model.eval()
        
        # Convert to tensor
        x = torch.FloatTensor([item_features]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output, attention = self.item_matching_model(x, return_attention=True)
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        categories = {0: 'no_match', 1: 'match'}
        
        return {
            'prediction': prediction.item(),
            'category': categories[prediction.item()],
            'confidence': probabilities[0][prediction].item(),
            'probabilities': probabilities[0].cpu().numpy().tolist(),
            'attention_weights': attention[0].cpu().numpy().tolist()
        }
    
    def predict_user_behavior(self, behavior_sequence: List[List[float]]) -> Dict:
        """
        Predict next user behavior
        
        Args:
            behavior_sequence: Sequence of user behavior features
        
        Returns:
            Dictionary with behavior predictions
        """
        self.behavior_predictor.eval()
        
        # Convert to tensor
        x = torch.FloatTensor([behavior_sequence]).to(self.device)
        
        with torch.no_grad():
            output, attention = self.behavior_predictor(x, return_attention=True)
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        actions = {0: 'search', 1: 'upload', 2: 'view', 3: 'browse', 4: 'logout'}
        
        return {
            'prediction': prediction.item(),
            'action': actions[prediction.item()],
            'confidence': probabilities[0][prediction].item(),
            'probabilities': probabilities[0].cpu().numpy().tolist(),
            'attention_weights': attention[0].cpu().numpy().tolist()
        }
    
    def classify_description(self, text: str) -> Dict:
        """
        Classify item description
        
        Args:
            text: Item description text
        
        Returns:
            Dictionary with classification results
        """
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        self.text_encoder.eval()
        
        # Convert text to sequence
        sequence = self.text_to_sequence(text)
        x = torch.LongTensor([sequence]).to(self.device)
        
        with torch.no_grad():
            output, attention = self.text_encoder(x, return_attention=True)
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        categories = {0: 'phone', 1: 'mouse', 2: 'wallet', 3: 'tumbler'}
        
        return {
            'prediction': prediction.item(),
            'category': categories[prediction.item()],
            'confidence': probabilities[0][prediction].item(),
            'probabilities': probabilities[0].cpu().numpy().tolist(),
            'attention_weights': attention[0].cpu().numpy().tolist()
        }
    
    def save_models(self):
        """Save all custom models"""
        try:
            # Save model states
            torch.save(self.item_matching_model.state_dict(), 
                      os.path.join(self.model_dir, 'item_matching_model.pth'))
            torch.save(self.behavior_predictor.state_dict(), 
                      os.path.join(self.model_dir, 'behavior_predictor.pth'))
            torch.save(self.text_encoder.state_dict(), 
                      os.path.join(self.model_dir, 'text_encoder.pth'))
            
            # Save vocabulary
            if self.vocab is not None:
                with open(os.path.join(self.model_dir, 'vocab.pkl'), 'wb') as f:
                    pickle.dump(self.vocab, f)
            
            # Save training history
            with open(os.path.join(self.model_dir, 'training_history.json'), 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            self.logger.info("Custom RNN models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving custom RNN models: {e}")
    
    def load_models(self):
        """Load all custom models"""
        try:
            # Load model states
            self.item_matching_model.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'item_matching_model.pth'), 
                          map_location=self.device))
            self.behavior_predictor.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'behavior_predictor.pth'), 
                          map_location=self.device))
            self.text_encoder.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'text_encoder.pth'), 
                          map_location=self.device))
            
            # Load vocabulary
            if os.path.exists(os.path.join(self.model_dir, 'vocab.pkl')):
                with open(os.path.join(self.model_dir, 'vocab.pkl'), 'rb') as f:
                    self.vocab = pickle.load(f)
            
            # Load training history
            if os.path.exists(os.path.join(self.model_dir, 'training_history.json')):
                with open(os.path.join(self.model_dir, 'training_history.json'), 'r') as f:
                    self.training_history = json.load(f)
            
            self.logger.info("Custom RNN models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading custom RNN models: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about all models"""
        models_info = {}
        
        for name, model in [('item_matching', self.item_matching_model),
                           ('behavior_predictor', self.behavior_predictor),
                           ('text_encoder', self.text_encoder)]:
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            models_info[name] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
            }
        
        return {
            'device': str(self.device),
            'model_directory': self.model_dir,
            'vocab_size': self.vocab_size,
            'vocab_loaded': self.vocab is not None,
            'models': models_info
        }
    
    def evaluate_item_matching_model(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate item matching model with comprehensive metrics
        
        Args:
            test_data: List of test samples with 'lost_features', 'found_features', and 'is_match'
        
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        evaluator = ModelEvaluator(device=str(self.device))
        
        # Prepare test data
        y_true = []
        y_pred = []
        y_prob = []
        
        for item in test_data:
            # Ground truth
            y_true.append(1 if item['is_match'] else 0)
            
            # Create combined features
            combined_features = item['lost_features'] + item['found_features']
            
            # Get prediction
            result = self.predict_item_match(combined_features[:20])  # Use first 20 features
            y_pred.append(result['prediction'])
            y_prob.append(result['probabilities'])
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # Evaluate
        results = evaluator.evaluate_classification_model(
            y_true, y_pred, y_prob, 
            class_names=['No Match', 'Match']
        )
        
        # Add model-specific metrics
        results['model_type'] = 'item_matching'
        results['num_test_samples'] = len(test_data)
        
        return results
    
    def evaluate_behavior_model(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate user behavior prediction model with comprehensive metrics
        
        Args:
            test_data: List of test samples with user behavior sequences
        
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        evaluator = ModelEvaluator(device=str(self.device))
        
        # Prepare test data
        y_true = []
        y_pred = []
        y_prob = []
        
        # Group by user and create sequences
        user_sequences = defaultdict(list)
        for item in test_data:
            user_sequences[item['user_id']].append(item)
        
        for user_id, actions in user_sequences.items():
            if len(actions) >= 10:  # Minimum sequence length
                sequence = actions[:10]  # Take first 10 actions
                features = [action['features'] for action in sequence]
                
                # Get next action as label
                action_encoding = {'search': 0, 'view': 1, 'upload': 2, 'browse': 3, 'logout': 4}
                next_action = sequence[-1]['next_action']
                y_true.append(action_encoding.get(next_action, 0))
                
                # Get prediction
                result = self.predict_user_behavior(features)
                y_pred.append(result['prediction'])
                y_prob.append(result['probabilities'])
        
        # Check if we have any valid samples
        if len(y_true) == 0:
            print(f"Warning: No valid behavior sequences found for evaluation")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'jaccard_similarity': 0.0,
                'confusion_matrix': [[0, 0], [0, 0]],
                'iou_scores': {'class_0': 0.0, 'class_1': 0.0, 'mean_iou': 0.0},
                'per_class_metrics': {
                    'precision': [0.0, 0.0],
                    'recall': [0.0, 0.0],
                    'f1_score': [0.0, 0.0],
                    'support': [0, 0]
                },
                'classification_report': {},
                'model_type': 'behavior_prediction',
                'num_test_samples': 0
            }
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # Evaluate
        results = evaluator.evaluate_classification_model(
            y_true, y_pred, y_prob,
            class_names=['search', 'view', 'upload', 'browse', 'logout']
        )
        
        # Add model-specific metrics
        results['model_type'] = 'behavior_prediction'
        results['num_test_samples'] = len(y_true)
        
        return results
    
    def evaluate_text_model(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate text description classification model with comprehensive metrics
        
        Args:
            test_data: List of test samples with 'text' and 'label'
        
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        evaluator = ModelEvaluator(device=str(self.device))
        
        # Prepare test data
        y_true = []
        y_pred = []
        y_prob = []
        
        for item in test_data:
            y_true.append(item['label'])
            
            # Get prediction
            result = self.classify_description(item['text'])
            y_pred.append(result['prediction'])
            y_prob.append(result['probabilities'])
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # Evaluate
        results = evaluator.evaluate_classification_model(
            y_true, y_pred, y_prob,
            class_names=['phone', 'mouse', 'wallet', 'tumbler']
        )
        
        # Add model-specific metrics
        results['model_type'] = 'text_classification'
        results['num_test_samples'] = len(test_data)
        
        return results
    
    def comprehensive_evaluation(self, test_data: Dict) -> Dict:
        """
        Run comprehensive evaluation on all models
        
        Args:
            test_data: Dictionary with keys 'matching', 'behavior', 'text' containing test data
        
        Returns:
            Dictionary with evaluation results for all models
        """
        evaluation_results = {}
        
        # Evaluate each model
        if 'matching' in test_data:
            print("🔍 Evaluating item matching model...")
            evaluation_results['item_matching'] = self.evaluate_item_matching_model(test_data['matching'])
        
        if 'behavior' in test_data:
            print("👤 Evaluating behavior prediction model...")
            evaluation_results['behavior'] = self.evaluate_behavior_model(test_data['behavior'])
        
        if 'text' in test_data:
            print("📝 Evaluating text classification model...")
            evaluation_results['text'] = self.evaluate_text_model(test_data['text'])
        
        return evaluation_results
    
    def plot_evaluation_results(self, evaluation_results: Dict, save_dir: str = None):
        """
        Plot comprehensive evaluation results for all models
        
        Args:
            evaluation_results: Results from comprehensive_evaluation
            save_dir: Directory to save plots (optional)
        """
        evaluator = ModelEvaluator(device=str(self.device))
        
        for model_name, results in evaluation_results.items():
            print(f"📊 Plotting results for {model_name}...")
            
            # Plot confusion matrix
            cm = np.array(results['confusion_matrix'])
            class_names = results.get('class_names', [f'Class {i}' for i in range(cm.shape[0])])
            
            evaluator.plot_confusion_matrix(
                cm, class_names, 
                title=f"{model_name.replace('_', ' ').title()} - Confusion Matrix",
                save_path=f"{save_dir}/{model_name}_confusion_matrix.png" if save_dir else None
            )
            
            # Plot comprehensive metrics
            evaluator.plot_metrics_comparison(
                results, 
                model_name=model_name.replace('_', ' ').title(),
                save_path=f"{save_dir}/{model_name}_metrics_comparison.png" if save_dir else None
            )
    
    def generate_evaluation_reports(self, evaluation_results: Dict, output_dir: str = None):
        """
        Generate detailed evaluation reports for all models
        
        Args:
            evaluation_results: Results from comprehensive_evaluation
            output_dir: Directory to save reports (optional)
        """
        evaluator = ModelEvaluator(device=str(self.device))
        
        for model_name, results in evaluation_results.items():
            print(f"📋 Generating report for {model_name}...")
            
            # Generate detailed report
            report = evaluator.generate_evaluation_report(
                results, 
                model_name=model_name.replace('_', ' ').title()
            )
            
            # Save report
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                report_path = os.path.join(output_dir, f"{model_name}_evaluation_report.md")
                with open(report_path, 'w') as f:
                    f.write(report)
                print(f"   Report saved to: {report_path}")
            else:
                print(f"\n{report}")
    
    def save_evaluation_results(self, evaluation_results: Dict, output_dir: str = None):
        """
        Save evaluation results to files
        
        Args:
            evaluation_results: Results from comprehensive_evaluation
            output_dir: Directory to save results (optional)
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save raw results as JSON
            results_path = os.path.join(output_dir, 'evaluation_results.json')
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            
            # Save summary CSV
            summary_data = []
            for model_name, results in evaluation_results.items():
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1_Score': results['f1_score'],
                    'Jaccard_Similarity': results['jaccard_similarity'],
                    'Mean_IoU': results['iou_scores']['mean_iou'],
                    'ROC_AUC': results.get('roc_auc', 'N/A'),
                    'Mean_AP': results.get('map_scores', {}).get('mean_ap', 'N/A'),
                    'Num_Samples': results['num_test_samples']
                })
            
            import pandas as pd
            df = pd.DataFrame(summary_data)
            csv_path = os.path.join(output_dir, 'evaluation_summary.csv')
            df.to_csv(csv_path, index=False)
            
            print(f"📁 Evaluation results saved to: {output_dir}")
            print(f"   - JSON results: {results_path}")
            print(f"   - CSV summary: {csv_path}")

class ModelEvaluator:
    """
    Comprehensive evaluation class for custom RNN models
    Provides accuracy, F1 score, recall, IoU, mAP, and confusion matrix metrics
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.evaluation_results = {}
        
    def calculate_iou(self, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
        """
        Calculate Intersection over Union (IoU) for each class
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            num_classes: Number of classes
        
        Returns:
            Dictionary with IoU scores for each class
        """
        iou_scores = {}
        
        for class_id in range(num_classes):
            # Create binary masks for current class
            true_mask = (y_true == class_id)
            pred_mask = (y_pred == class_id)
            
            # Calculate intersection and union
            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            
            # Calculate IoU
            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection / union
            
            iou_scores[f'class_{class_id}'] = iou
        
        # Calculate mean IoU
        iou_scores['mean_iou'] = np.mean(list(iou_scores.values()))
        
        return iou_scores
    
    def calculate_map(self, y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> Dict[str, float]:
        """
        Calculate Mean Average Precision (mAP) for multi-class classification
        
        Args:
            y_true: Ground truth labels
            y_prob: Prediction probabilities
            num_classes: Number of classes
        
        Returns:
            Dictionary with mAP scores
        """
        map_scores = {}
        
        # Calculate AP for each class
        for class_id in range(num_classes):
            # Create binary labels for current class
            y_true_binary = (y_true == class_id).astype(int)
            y_prob_binary = y_prob[:, class_id]
            
            # Calculate Average Precision
            ap = average_precision_score(y_true_binary, y_prob_binary)
            map_scores[f'ap_class_{class_id}'] = ap
        
        # Calculate mean AP
        map_scores['mean_ap'] = np.mean(list(map_scores.values()))
        
        return map_scores
    
    def calculate_jaccard_similarity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Jaccard similarity coefficient
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
        
        Returns:
            Jaccard similarity score
        """
        return jaccard_score(y_true, y_pred, average='macro')
    
    def evaluate_classification_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_prob: np.ndarray = None, class_names: List[str] = None) -> Dict:
        """
        Comprehensive evaluation for classification models
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            class_names: List of class names (optional)
        
        Returns:
            Dictionary with all evaluation metrics
        """
        num_classes = len(np.unique(y_true))
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # IoU scores
        iou_scores = self.calculate_iou(y_true, y_pred, num_classes)
        
        # Jaccard similarity
        jaccard = self.calculate_jaccard_similarity(y_true, y_pred)
        
        # ROC AUC (if probabilities provided)
        roc_auc = None
        if y_prob is not None and len(y_true) > 0:
            try:
                if num_classes == 2:
                    roc_auc = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except ValueError:
                # Handle cases where ROC AUC cannot be calculated
                roc_auc = None
        
        # mAP (if probabilities provided)
        map_scores = None
        if y_prob is not None and len(y_true) > 0:
            try:
                map_scores = self.calculate_map(y_true, y_prob, num_classes)
            except ValueError:
                # Handle cases where mAP cannot be calculated
                map_scores = None
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Compile results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'jaccard_similarity': jaccard,
            'confusion_matrix': cm.tolist(),
            'iou_scores': iou_scores,
            'per_class_metrics': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1_score': f1_per_class.tolist(),
                'support': support.tolist()
            },
            'classification_report': class_report
        }
        
        if roc_auc is not None:
            results['roc_auc'] = roc_auc
        
        if map_scores is not None:
            results['map_scores'] = map_scores
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str] = None, 
                            title: str = "Confusion Matrix", save_path: str = None):
        """
        Plot confusion matrix with detailed visualization
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            title: Plot title
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.5, 0.02, f'Accuracy: {accuracy:.4f}', ha='center', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics_comparison(self, results: Dict, model_name: str = "Model", 
                              save_path: str = None):
        """
        Plot comprehensive metrics comparison
        
        Args:
            results: Evaluation results dictionary
            model_name: Name of the model
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_name} - Comprehensive Evaluation Metrics', fontsize=16, fontweight='bold')
        
        # 1. Overall Metrics Bar Chart
        ax1 = axes[0, 0]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Jaccard']
        values = [
            results['accuracy'],
            results['precision'],
            results['recall'],
            results['f1_score'],
            results['jaccard_similarity']
        ]
        
        bars = ax1.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax1.set_title('Overall Metrics', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Per-Class Precision
        ax2 = axes[0, 1]
        if 'per_class_metrics' in results:
            class_names = [f'Class {i}' for i in range(len(results['per_class_metrics']['precision']))]
            ax2.bar(class_names, results['per_class_metrics']['precision'], color='#ff7f0e')
            ax2.set_title('Per-Class Precision', fontweight='bold')
            ax2.set_ylabel('Precision')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Per-Class Recall
        ax3 = axes[0, 2]
        if 'per_class_metrics' in results:
            ax3.bar(class_names, results['per_class_metrics']['recall'], color='#2ca02c')
            ax3.set_title('Per-Class Recall', fontweight='bold')
            ax3.set_ylabel('Recall')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. IoU Scores
        ax4 = axes[1, 0]
        if 'iou_scores' in results:
            iou_classes = [k for k in results['iou_scores'].keys() if k != 'mean_iou']
            iou_values = [results['iou_scores'][k] for k in iou_classes]
            ax4.bar(iou_classes, iou_values, color='#d62728')
            ax4.set_title('IoU Scores by Class', fontweight='bold')
            ax4.set_ylabel('IoU Score')
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. mAP Scores (if available)
        ax5 = axes[1, 1]
        if 'map_scores' in results:
            map_classes = [k for k in results['map_scores'].keys() if k != 'mean_ap']
            map_values = [results['map_scores'][k] for k in map_classes]
            ax5.bar(map_classes, map_values, color='#9467bd')
            ax5.set_title('mAP Scores by Class', fontweight='bold')
            ax5.set_ylabel('mAP Score')
            ax5.tick_params(axis='x', rotation=45)
        else:
            ax5.text(0.5, 0.5, 'mAP not available\n(requires probabilities)', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('mAP Scores', fontweight='bold')
        
        # 6. ROC AUC (if available)
        ax6 = axes[1, 2]
        if 'roc_auc' in results:
            ax6.bar(['ROC AUC'], [results['roc_auc']], color='#8c564b')
            ax6.set_title('ROC AUC Score', fontweight='bold')
            ax6.set_ylabel('AUC Score')
            ax6.set_ylim(0, 1)
            ax6.text(0, results['roc_auc'] + 0.01, f'{results["roc_auc"]:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'ROC AUC not available\n(requires probabilities)', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('ROC AUC Score', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(self, results: Dict, model_name: str = "Model") -> str:
        """
        Generate detailed evaluation report
        
        Args:
            results: Evaluation results dictionary
            model_name: Name of the model
        
        Returns:
            Formatted evaluation report string
        """
        report = []
        report.append(f"# {model_name} - Comprehensive Evaluation Report")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall Metrics
        report.append("## Overall Metrics")
        report.append("-" * 20)
        report.append(f"Accuracy: {results['accuracy']:.4f}")
        report.append(f"Precision: {results['precision']:.4f}")
        report.append(f"Recall: {results['recall']:.4f}")
        report.append(f"F1-Score: {results['f1_score']:.4f}")
        report.append(f"Jaccard Similarity: {results['jaccard_similarity']:.4f}")
        
        if 'roc_auc' in results:
            report.append(f"ROC AUC: {results['roc_auc']:.4f}")
        
        if 'map_scores' in results:
            report.append(f"Mean Average Precision (mAP): {results['map_scores']['mean_ap']:.4f}")
        
        # IoU Scores
        report.append("\n## IoU Scores")
        report.append("-" * 15)
        for class_name, iou in results['iou_scores'].items():
            report.append(f"{class_name}: {iou:.4f}")
        
        # Per-Class Metrics
        if 'per_class_metrics' in results:
            report.append("\n## Per-Class Metrics")
            report.append("-" * 20)
            report.append("| Class | Precision | Recall | F1-Score | Support |")
            report.append("|-------|-----------|--------|----------|---------|")
            
            for i, (prec, rec, f1, sup) in enumerate(zip(
                results['per_class_metrics']['precision'],
                results['per_class_metrics']['recall'],
                results['per_class_metrics']['f1_score'],
                results['per_class_metrics']['support']
            )):
                report.append(f"| {i} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {int(sup)} |")
        
        # Confusion Matrix
        report.append("\n## Confusion Matrix")
        report.append("-" * 20)
        cm = np.array(results['confusion_matrix'])
        report.append("```")
        report.append(str(cm))
        report.append("```")
        
        # Classification Report
        report.append("\n## Detailed Classification Report")
        report.append("-" * 35)
        report.append("```")
        class_report = results['classification_report']
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict):
                report.append(f"\n{class_name}:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        report.append(f"  {metric}: {value:.4f}")
        report.append("```")
        
        return "\n".join(report)

# Global custom RNN manager instance
custom_rnn_manager = CustomRNNManager(device='cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # Test the custom RNN models
    print("🧪 Testing Custom RNN Models")
    print("=" * 50)
    
    # Test model info
    info = custom_rnn_manager.get_model_info()
    print(f"Device: {info['device']}")
    print(f"Model Directory: {info['model_directory']}")
    print(f"Vocabulary Size: {info['vocab_size']}")
    
    for model_name, model_info in info['models'].items():
        print(f"\n{model_name.upper()}:")
        print(f"  Total Parameters: {model_info['total_parameters']:,}")
        print(f"  Trainable Parameters: {model_info['trainable_parameters']:,}")
        print(f"  Model Size: {model_info['model_size_mb']:.2f} MB")
    
    # Test item matching
    print("\n🔍 Testing Item Matching:")
    sample_item_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                           0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 40 features
    match_result = custom_rnn_manager.predict_item_match(sample_item_features)
    print(f"Prediction: {match_result['category']} (confidence: {match_result['confidence']:.3f})")
    
    # Test behavior prediction
    print("\n👤 Testing User Behavior Prediction:")
    sample_behavior_sequence = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5]] * 10
    behavior_result = custom_rnn_manager.predict_user_behavior(sample_behavior_sequence)
    print(f"Prediction: {behavior_result['action']} (confidence: {behavior_result['confidence']:.3f})")
    
    # Test text classification (requires vocabulary)
    print("\n📝 Testing Text Classification:")
    sample_texts = [
        "Lost black iPhone 12 with cracked screen",
        "Found wireless mouse Logitech black",
        "Lost wallet brown leather with cards",
        "Found tumbler stainless steel silver"
    ]
    
    print("Building vocabulary...")
    custom_rnn_manager.build_vocabulary(sample_texts)
    
    for text in sample_texts:
        text_result = custom_rnn_manager.classify_description(text)
        print(f"Text: '{text}'")
        print(f"  Prediction: {text_result['category']} (confidence: {text_result['confidence']:.3f})")
    
    print("\n✅ Custom RNN Models testing completed!")
    
    # Test evaluation capabilities
    print("\n📊 Testing Evaluation Capabilities:")
    
    # Create sample test data for evaluation
    sample_test_data = {
        'matching': [
            {
                'lost_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 2,  # 20 features
                'found_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 2,  # 20 features
                'is_match': True
            },
            {
                'lost_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 2,  # 20 features
                'found_features': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] * 2,  # 20 features
                'is_match': False
            }
        ],
        'behavior': [
            # User 1 - 15 actions (enough for evaluation)
            {'user_id': 'user_1', 'features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5], 'next_action': 'search'},
            {'user_id': 'user_1', 'features': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 'next_action': 'view'},
            {'user_id': 'user_1', 'features': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 'next_action': 'upload'},
            {'user_id': 'user_1', 'features': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 'next_action': 'browse'},
            {'user_id': 'user_1', 'features': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'next_action': 'search'},
            {'user_id': 'user_1', 'features': [0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'next_action': 'view'},
            {'user_id': 'user_1', 'features': [0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1], 'next_action': 'upload'},
            {'user_id': 'user_1', 'features': [0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2], 'next_action': 'browse'},
            {'user_id': 'user_1', 'features': [0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3], 'next_action': 'search'},
            {'user_id': 'user_1', 'features': [1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4], 'next_action': 'view'},
            {'user_id': 'user_1', 'features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5], 'next_action': 'upload'},
            {'user_id': 'user_1', 'features': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 'next_action': 'browse'},
            {'user_id': 'user_1', 'features': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 'next_action': 'search'},
            {'user_id': 'user_1', 'features': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 'next_action': 'view'},
            {'user_id': 'user_1', 'features': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'next_action': 'upload'}
        ],
        'text': [
            {'text': 'Lost black iPhone 12', 'label': 0},
            {'text': 'Found wireless mouse', 'label': 1}
        ]
    }
    
    # Run comprehensive evaluation
    print("Running comprehensive evaluation...")
    evaluation_results = custom_rnn_manager.comprehensive_evaluation(sample_test_data)
    
    # Print summary results
    print("\n📈 Evaluation Summary:")
    for model_name, results in evaluation_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
        print(f"  Jaccard Similarity: {results['jaccard_similarity']:.4f}")
        print(f"  Mean IoU: {results['iou_scores']['mean_iou']:.4f}")
        if 'roc_auc' in results:
            print(f"  ROC AUC: {results['roc_auc']:.4f}")
        if 'map_scores' in results:
            print(f"  Mean AP: {results['map_scores']['mean_ap']:.4f}")
    
    print("\n✅ Comprehensive evaluation testing completed!")
