"""
Unified RNN Model for BARYONYX Lost & Found System

This module combines all RNN functionality into a single, efficient model:
- User Behavior Prediction
- Text Description Classification  
- Temporal Pattern Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
import json
import pickle
import os
from datetime import datetime, timedelta
import random

class AttentionLayer(nn.Module):
    """Attention mechanism for RNN outputs"""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, rnn_outputs):
        # rnn_outputs: (batch_size, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(rnn_outputs), dim=1)
        context_vector = torch.sum(attention_weights * rnn_outputs, dim=1)
        return context_vector, attention_weights.squeeze(-1)

class UnifiedRNNModel(nn.Module):
    """
    Unified RNN Model that handles all three tasks:
    1. User Behavior Prediction (LSTM)
    2. Text Description Classification (Bidirectional GRU)
    3. Temporal Pattern Recognition (RNN)
    """
    
    def __init__(self, 
                 vocab_size=1000,
                 embedding_dim=128,
                 hidden_size=64,
                 num_behavior_classes=5,
                 num_description_classes=4,
                 num_temporal_classes=5,
                 max_seq_length=20,
                 device='cpu'):
        super(UnifiedRNNModel, self).__init__()
        
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        
        # Shared embedding layer for text processing
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # User Behavior LSTM
        self.behavior_lstm = nn.LSTM(
            input_size=10,  # User behavior features
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.behavior_attention = AttentionLayer(hidden_size)
        self.behavior_fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_behavior_classes)
        )
        
        # Description Bidirectional GRU
        self.description_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.description_attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
        self.description_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_description_classes)
        )
        
        # Temporal Pattern RNN
        self.temporal_rnn = nn.RNN(
            input_size=10,  # Temporal features
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.temporal_fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_temporal_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward_behavior(self, x):
        """Forward pass for user behavior prediction"""
        # x: (batch_size, seq_len, 10)
        lstm_out, _ = self.behavior_lstm(x)
        context_vector, attention_weights = self.behavior_attention(lstm_out)
        output = self.behavior_fc(context_vector)
        return output, attention_weights
    
    def forward_description(self, x):
        """Forward pass for text description classification"""
        # x: (batch_size, seq_len) - word indices
        embedded = self.embedding(x)
        gru_out, _ = self.description_gru(embedded)
        context_vector, attention_weights = self.description_attention(gru_out)
        output = self.description_fc(context_vector)
        return output, attention_weights
    
    def forward_temporal(self, x):
        """Forward pass for temporal pattern recognition"""
        # x: (batch_size, seq_len, 10)
        rnn_out, _ = self.temporal_rnn(x)
        # Use the last output for temporal prediction
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        output = self.temporal_fc(last_output)
        return output, None
    
    def forward(self, x, task_type):
        """
        Unified forward pass
        
        Args:
            x: Input data
            task_type: 'behavior', 'description', or 'temporal'
        """
        if task_type == 'behavior':
            return self.forward_behavior(x)
        elif task_type == 'description':
            return self.forward_description(x)
        elif task_type == 'temporal':
            return self.forward_temporal(x)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

class UnifiedRNNManager:
    """
    Manager class for the unified RNN model
    Handles training, inference, and model management
    """
    
    def __init__(self, device='cpu', model_path='models/unified_rnn/'):
        self.device = device
        self.model_path = model_path
        self.vocab = None
        self.vocab_size = 1000
        self.model = None
        
        # Create model directory
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the unified model"""
        self.model = UnifiedRNNModel(
            vocab_size=self.vocab_size,
            device=self.device
        ).to(self.device)
    
    def build_vocabulary(self, texts, min_freq=2):
        """Build vocabulary from text data"""
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())
        
        # Create vocabulary
        self.vocab = {word: idx + 2 for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1
        self.vocab_size = len(self.vocab)
        
        # Update model with new vocab size
        self._init_model()
        
        return self.vocab
    
    def text_to_sequence(self, text, max_length=20):
        """Convert text to sequence of word indices"""
        words = text.lower().split()
        word_indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        if len(word_indices) < max_length:
            word_indices.extend([self.vocab['<PAD>']] * (max_length - len(word_indices)))
        else:
            word_indices = word_indices[:max_length]
        
        return word_indices
    
    def predict_behavior(self, user_sequence):
        """Predict next user action"""
        if not isinstance(user_sequence, torch.Tensor):
            user_sequence = torch.FloatTensor(user_sequence).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output, attention = self.model(user_sequence, 'behavior')
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        return prediction.item(), probabilities.cpu().numpy()[0], attention.cpu().numpy()[0]
    
    def classify_description(self, text):
        """Classify item description"""
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        sequence = self.text_to_sequence(text)
        sequence_tensor = torch.LongTensor(sequence).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output, attention = self.model(sequence_tensor, 'description')
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        categories = {0: 'phone', 1: 'mouse', 2: 'wallet', 3: 'tumbler'}
        return prediction.item(), probabilities.cpu().numpy()[0], attention.cpu().numpy()[0], categories
    
    def predict_temporal_pattern(self, temporal_sequence):
        """Predict optimal time period"""
        if not isinstance(temporal_sequence, torch.Tensor):
            temporal_sequence = torch.FloatTensor(temporal_sequence).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(temporal_sequence, 'temporal')
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        time_periods = {
            0: 'Early Morning (6-9 AM)',
            1: 'Morning (9-12 PM)',
            2: 'Afternoon (12-5 PM)',
            3: 'Evening (5-9 PM)',
            4: 'Night (9 PM-6 AM)'
        }
        
        return prediction.item(), probabilities.cpu().numpy()[0], time_periods
    
    def save_model(self):
        """Save the unified model and vocabulary"""
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(self.model_path, 'unified_rnn_model.pth'))
        
        # Save vocabulary
        if self.vocab is not None:
            with open(os.path.join(self.model_path, 'vocab.pkl'), 'wb') as f:
                pickle.dump(self.vocab, f)
        
        # Save metadata
        metadata = {
            'vocab_size': self.vocab_size,
            'device': self.device,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.model_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Unified RNN model saved to {self.model_path}")
    
    def load_model(self):
        """Load the unified model and vocabulary"""
        try:
            # Load model state
            model_path = os.path.join(self.model_path, 'unified_rnn_model.pth')
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("✅ Model weights loaded successfully")
            else:
                print("⚠️ Model weights not found, using initialized model")
            
            # Load vocabulary
            vocab_path = os.path.join(self.model_path, 'vocab.pkl')
            if os.path.exists(vocab_path):
                with open(vocab_path, 'rb') as f:
                    self.vocab = pickle.load(f)
                self.vocab_size = len(self.vocab)
                print(f"✅ Vocabulary loaded: {self.vocab_size} words")
            else:
                print("⚠️ Vocabulary not found, will need to build from data")
            
            # Load metadata
            metadata_path = os.path.join(self.model_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"✅ Model metadata loaded: {metadata}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def get_model_info(self):
        """Get model information and statistics"""
        if self.model is None:
            return "Model not initialized"
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'model_type': 'Unified RNN Model',
            'device': self.device,
            'vocab_size': self.vocab_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_path': self.model_path,
            'vocab_loaded': self.vocab is not None
        }
        
        return info

# Example usage and testing
if __name__ == "__main__":
    # Initialize the unified model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    manager = UnifiedRNNManager(device=device)
    
    # Test model info
    print("\n📊 Model Information:")
    info = manager.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with sample data
    print("\n🧪 Testing Model Components:")
    
    # Test behavior prediction
    sample_behavior = torch.randn(1, 10, 10).to(device)  # (batch, seq_len, features)
    behavior_pred, behavior_probs, behavior_attn = manager.predict_behavior(sample_behavior)
    print(f"Behavior Prediction: {behavior_pred}, Confidence: {behavior_probs.max():.3f}")
    
    # Test temporal prediction
    sample_temporal = torch.randn(1, 7, 10).to(device)  # (batch, seq_len, features)
    temporal_pred, temporal_probs, time_periods = manager.predict_temporal_pattern(sample_temporal)
    print(f"Temporal Prediction: {temporal_pred} ({time_periods[temporal_pred]}), Confidence: {temporal_probs.max():.3f}")
    
    # Test description classification (requires vocabulary)
    sample_texts = [
        "Lost black iPhone 12 with cracked screen",
        "Found wireless mouse Logitech black",
        "Lost wallet brown leather with cards",
        "Found tumbler stainless steel silver"
    ]
    
    print(f"\nBuilding vocabulary from {len(sample_texts)} sample texts...")
    manager.build_vocabulary(sample_texts)
    
    for text in sample_texts:
        desc_pred, desc_probs, desc_attn, categories = manager.classify_description(text)
        print(f"Text: '{text}'")
        print(f"  Prediction: {desc_pred} ({categories[desc_pred]}), Confidence: {desc_probs.max():.3f}")
    
    # Save the model
    manager.save_model()
    
    print("\n✅ Unified RNN Model testing completed!")

