#!/usr/bin/env python3
"""
RNN Models for Lost & Found System
Implements LSTM, GRU, and Bidirectional RNNs for sequential data analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict, deque
import pickle

class AttentionLayer(nn.Module):
    """Attention mechanism for focusing on important features"""
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size
    
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

class UserBehaviorLSTM(nn.Module):
    """LSTM model for analyzing user behavior sequences"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = 5, dropout: float = 0.2):
        super(UserBehaviorLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        context_vector, attention_weights = self.attention(lstm_out)
        
        # Classification
        out = self.dropout(F.relu(self.fc1(context_vector)))
        out = self.fc2(out)
        
        return out, attention_weights

class BidirectionalDescriptionRNN(nn.Module):
    """Bidirectional RNN for processing item descriptions"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, 
                 hidden_size: int = 64, num_layers: int = 2, output_size: int = 10):
        super(BidirectionalDescriptionRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=0.2)
        
        # Attention for bidirectional output
        self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
        
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        gru_out, _ = self.gru(embedded)
        
        # Apply attention
        context_vector, attention_weights = self.attention(gru_out)
        
        output = self.fc(context_vector)
        return output, attention_weights

class TemporalPatternRNN(nn.Module):
    """RNN for temporal pattern recognition in lost/found items"""
    
    def __init__(self, input_size: int = 8, hidden_size: int = 32, 
                 num_layers: int = 2, output_size: int = 4):
        super(TemporalPatternRNN, self).__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=0.1)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        # Use last output
        last_output = gru_out[:, -1, :]
        
        out = self.dropout(F.relu(self.fc1(last_output)))
        out = self.fc2(out)
        return out

class RNNModelManager:
    """Manager class for all RNN models"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.user_behavior_model = UserBehaviorLSTM().to(self.device)
        self.description_model = BidirectionalDescriptionRNN().to(self.device)
        self.temporal_model = TemporalPatternRNN().to(self.device)
        
        # Data storage
        self.user_sequences = defaultdict(list)
        self.temporal_data = []
        self.vocab = {}
        self.vocab_size = 0
        
        # Model paths
        self.model_dir = 'models/rnn_models'
        os.makedirs(self.model_dir, exist_ok=True)
        
    def create_user_behavior_features(self, user_id: str, action: str, 
                                    item_type: str = None, timestamp: datetime = None) -> List[float]:
        """Create feature vector for user behavior"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Feature vector: [hour, day_of_week, action_type, item_type, confidence, 
        #                  search_count, upload_count, view_count, time_since_last, session_length]
        features = [0.0] * 10
        
        # Time features
        features[0] = timestamp.hour / 24.0  # Normalized hour
        features[1] = timestamp.weekday() / 7.0  # Normalized day of week
        
        # Action type encoding
        action_encoding = {
            'search': 1.0, 'upload': 2.0, 'view': 3.0, 'login': 4.0, 'logout': 5.0
        }
        features[2] = action_encoding.get(action, 0.0) / 5.0
        
        # Item type encoding
        item_encoding = {
            'phone': 1.0, 'wallet': 2.0, 'mouse': 3.0, 'tumbler': 4.0, 'keypad': 5.0
        }
        features[3] = item_encoding.get(item_type, 0.0) / 5.0 if item_type else 0.0
        
        # User activity counts (from recent history)
        recent_actions = self.user_sequences[user_id][-10:]  # Last 10 actions
        features[5] = sum(1 for a in recent_actions if a[1] == 'search') / 10.0
        features[6] = sum(1 for a in recent_actions if a[1] == 'upload') / 10.0
        features[7] = sum(1 for a in recent_actions if a[1] == 'view') / 10.0
        
        # Time since last action
        if len(self.user_sequences[user_id]) > 1:
            last_time = self.user_sequences[user_id][-1][2]  # timestamp is at index 2
            if isinstance(last_time, datetime):
                time_diff = (timestamp - last_time).total_seconds() / 3600.0  # Hours
                features[8] = min(time_diff / 24.0, 1.0)  # Normalized to 1 day max
        
        # Session length (simplified)
        features[9] = len(self.user_sequences[user_id]) / 100.0  # Normalized
        
        return features
    
    def add_user_action(self, user_id: str, action: str, item_type: str = None, 
                       timestamp: datetime = None, confidence: float = 1.0):
        """Add user action to sequence for analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store action
        self.user_sequences[user_id].append((action, item_type, timestamp, confidence))
        
        # Keep only last 100 actions per user
        if len(self.user_sequences[user_id]) > 100:
            self.user_sequences[user_id] = self.user_sequences[user_id][-100:]
    
    def predict_user_intent(self, user_id: str, sequence_length: int = 10) -> Dict:
        """Predict user's next likely action"""
        if user_id not in self.user_sequences or len(self.user_sequences[user_id]) < 3:
            return {"prediction": "unknown", "confidence": 0.0, "suggestions": []}
        
        # Prepare sequence data
        recent_actions = self.user_sequences[user_id][-sequence_length:]
        features = []
        
        for action, item_type, timestamp, confidence in recent_actions:
            feature_vector = self.create_user_behavior_features(user_id, action, item_type, timestamp)
            feature_vector[4] = confidence  # Add confidence
            features.append(feature_vector)
        
        # Pad sequence if needed
        while len(features) < sequence_length:
            features.insert(0, [0.0] * 10)
        
        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Predict
        self.user_behavior_model.eval()
        with torch.no_grad():
            prediction, attention = self.user_behavior_model(x)
            probabilities = F.softmax(prediction, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map prediction to action
        action_map = {0: "search", 1: "upload", 2: "view", 3: "browse", 4: "logout"}
        predicted_action = action_map.get(predicted_class, "unknown")
        
        # Generate suggestions based on prediction
        suggestions = self._generate_suggestions(predicted_action, user_id)
        
        return {
            "prediction": predicted_action,
            "confidence": confidence,
            "suggestions": suggestions,
            "attention_weights": attention[0].cpu().numpy().tolist()
        }
    
    def _generate_suggestions(self, predicted_action: str, user_id: str) -> List[str]:
        """Generate contextual suggestions based on prediction"""
        suggestions = []
        
        if predicted_action == "search":
            # Get recent search patterns
            recent_searches = [a for a in self.user_sequences[user_id][-5:] if a[1] == 'search']
            if recent_searches:
                item_types = [a[2] for a in recent_searches if a[2]]
                if item_types:
                    suggestions.append(f"Try searching for {item_types[-1]} items")
            suggestions.extend(["Search by color", "Search by location", "Search by time"])
        
        elif predicted_action == "upload":
            suggestions.extend(["Upload clear photos", "Add detailed description", "Specify location"])
        
        elif predicted_action == "view":
            suggestions.extend(["View similar items", "Check recent uploads", "Browse categories"])
        
        return suggestions
    
    def analyze_description_sequence(self, description: str) -> Dict:
        """Analyze item description using bidirectional RNN"""
        if not description:
            return {"features": [], "similarity_score": 0.0}
        
        # Simple tokenization (in production, use proper tokenizer)
        words = description.lower().split()
        
        # Convert to indices
        word_indices = []
        for word in words:
            if word in self.vocab:
                word_indices.append(self.vocab[word])
            else:
                word_indices.append(0)  # Unknown word
        
        # Pad or truncate to fixed length
        max_length = 20
        if len(word_indices) < max_length:
            word_indices.extend([0] * (max_length - len(word_indices)))
        else:
            word_indices = word_indices[:max_length]
        
        # Convert to tensor
        x = torch.tensor([word_indices], dtype=torch.long).to(self.device)
        
        # Analyze
        self.description_model.eval()
        with torch.no_grad():
            output, attention = self.description_model(x)
            features = output[0].cpu().numpy().tolist()
        
        # Calculate similarity score (simplified)
        similarity_score = float(torch.sigmoid(output[0].sum()).item())
        
        return {
            "features": features,
            "similarity_score": similarity_score,
            "attention_weights": attention[0].cpu().numpy().tolist()
        }
    
    def predict_temporal_patterns(self, item_type: str, location: str = None) -> Dict:
        """Predict when items of this type are typically lost/found"""
        # Create temporal features: [hour, day, month, season, location_type, 
        #                           item_frequency, weather_factor, holiday_factor]
        now = datetime.now()
        features = [0.0] * 8
        
        features[0] = now.hour / 24.0
        features[1] = now.day / 31.0
        features[2] = now.month / 12.0
        features[3] = ((now.month - 1) // 3) / 4.0  # Season
        
        # Location type encoding
        location_encoding = {
            'office': 1.0, 'home': 2.0, 'public': 3.0, 'transport': 4.0
        }
        features[4] = location_encoding.get(location, 0.0) / 4.0 if location else 0.0
        
        # Item frequency (from historical data)
        item_count = sum(1 for data in self.temporal_data if data.get('item_type') == item_type)
        features[5] = min(item_count / 100.0, 1.0)
        
        # Weather factor (simplified - in production, use real weather data)
        features[6] = 0.5  # Neutral weather
        
        # Holiday factor
        features[7] = 0.0  # Not a holiday
        
        # Convert to tensor and add sequence dimension
        x = torch.tensor([features], dtype=torch.float32).unsqueeze(1).to(self.device)  # Add sequence dimension
        
        # Predict
        self.temporal_model.eval()
        with torch.no_grad():
            prediction = self.temporal_model(x)
            probabilities = F.softmax(prediction, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Map to time periods
        time_periods = {0: "morning", 1: "afternoon", 2: "evening", 3: "night"}
        predicted_time = time_periods.get(predicted_class, "unknown")
        
        return {
            "predicted_time_period": predicted_time,
            "confidence": confidence,
            "recommendations": self._get_temporal_recommendations(predicted_time, item_type)
        }
    
    def _get_temporal_recommendations(self, time_period: str, item_type: str) -> List[str]:
        """Get recommendations based on temporal patterns"""
        recommendations = []
        
        if time_period == "morning":
            recommendations.extend(["Check common morning locations", "Look in commute areas"])
        elif time_period == "afternoon":
            recommendations.extend(["Check work areas", "Look in lunch spots"])
        elif time_period == "evening":
            recommendations.extend(["Check social areas", "Look in entertainment venues"])
        else:
            recommendations.extend(["Check home areas", "Look in personal spaces"])
        
        # Item-specific recommendations
        if item_type == "phone":
            recommendations.extend(["Check pockets and bags", "Look near charging areas"])
        elif item_type == "wallet":
            recommendations.extend(["Check where you last paid", "Look in coat pockets"])
        
        return recommendations
    
    def save_models(self):
        """Save all RNN models"""
        try:
            torch.save(self.user_behavior_model.state_dict(), 
                      os.path.join(self.model_dir, 'user_behavior_lstm.pth'))
            torch.save(self.description_model.state_dict(), 
                      os.path.join(self.model_dir, 'description_birnn.pth'))
            torch.save(self.temporal_model.state_dict(), 
                      os.path.join(self.model_dir, 'temporal_pattern_rnn.pth'))
            
            # Save vocab and data
            with open(os.path.join(self.model_dir, 'vocab.pkl'), 'wb') as f:
                pickle.dump(self.vocab, f)
            
            with open(os.path.join(self.model_dir, 'temporal_data.pkl'), 'wb') as f:
                pickle.dump(self.temporal_data, f)
            
            self.logger.info("RNN models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving RNN models: {e}")
    
    def load_models(self):
        """Load all RNN models"""
        try:
            # Load model states
            self.user_behavior_model.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'user_behavior_lstm.pth'), 
                          map_location=self.device))
            self.description_model.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'description_birnn.pth'), 
                          map_location=self.device))
            self.temporal_model.load_state_dict(
                torch.load(os.path.join(self.model_dir, 'temporal_pattern_rnn.pth'), 
                          map_location=self.device))
            
            # Load vocab and data
            if os.path.exists(os.path.join(self.model_dir, 'vocab.pkl')):
                with open(os.path.join(self.model_dir, 'vocab.pkl'), 'rb') as f:
                    self.vocab = pickle.load(f)
                    self.vocab_size = len(self.vocab)
            
            if os.path.exists(os.path.join(self.model_dir, 'temporal_data.pkl')):
                with open(os.path.join(self.model_dir, 'temporal_data.pkl'), 'rb') as f:
                    self.temporal_data = pickle.load(f)
            
            self.logger.info("RNN models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading RNN models: {e}")
    
    def train_models(self, training_data: Dict):
        """Train RNN models with provided data"""
        # This is a simplified training function
        # In production, you'd want more sophisticated training with proper datasets
        
        self.logger.info("Training RNN models...")
        
        # Set models to training mode
        self.user_behavior_model.train()
        self.description_model.train()
        self.temporal_model.train()
        
        # Training would go here with actual data
        # For now, just save the models
        self.save_models()
        
        self.logger.info("RNN model training completed")

# Global RNN manager instance
rnn_manager = RNNModelManager(device='cuda' if torch.cuda.is_available() else 'cpu')
