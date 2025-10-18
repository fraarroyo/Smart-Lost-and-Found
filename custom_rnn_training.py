#!/usr/bin/env python3
"""
Custom RNN Training Pipeline for BARYONYX Lost & Found System
Comprehensive training system with data augmentation, validation, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import pickle

# Import custom models
from custom_rnn_model import CustomRNNManager, ItemMatchingRNN, UserBehaviorPredictor, TextDescriptionEncoder

class CustomRNNTrainer:
    """Comprehensive trainer for custom RNN models"""
    
    def __init__(self, device: str = 'cpu', model_dir: str = 'models/custom_rnn/'):
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.logger = self._setup_logging()
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'plots'), exist_ok=True)
        
        # Initialize models
        self.manager = CustomRNNManager(device=device, model_dir=model_dir)
        
        # Training history
        self.training_history = {
            'item_matching': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
            'behavior': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []},
            'text': {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for training"""
        logger = logging.getLogger('custom_rnn_trainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_enhanced_datasets(self, data_dir: str = 'enhanced_datasets/') -> Dict:
        """Load enhanced training datasets"""
        datasets = {}
        
        # Load user behavior dataset
        with open(os.path.join(data_dir, 'enhanced_user_behavior.json'), 'r') as f:
            datasets['behavior'] = json.load(f)
        
        # Load item matching dataset
        with open(os.path.join(data_dir, 'enhanced_item_matching.json'), 'r') as f:
            datasets['matching'] = json.load(f)
        
        # Load temporal dataset
        with open(os.path.join(data_dir, 'enhanced_temporal.json'), 'r') as f:
            datasets['temporal'] = json.load(f)
        
        # Load text descriptions dataset
        with open(os.path.join(data_dir, 'enhanced_text_descriptions.json'), 'r') as f:
            datasets['text'] = json.load(f)
        
        self.logger.info(f"Loaded datasets: {[(k, len(v)) for k, v in datasets.items()]}")
        return datasets
    
    def prepare_behavior_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare user behavior data for training"""
        X, y = [], []
        
        # Group by user and create sequences
        user_sequences = {}
        for item in data:
            user_id = item['user_id']
            if user_id not in user_sequences:
                user_sequences[user_id] = []
            user_sequences[user_id].append(item)
        
        # Create sequences with sliding window
        sequence_length = 10
        for user_id, actions in user_sequences.items():
            if len(actions) >= sequence_length:
                for i in range(len(actions) - sequence_length + 1):
                    sequence = actions[i:i + sequence_length]
                    features = [action['features'] for action in sequence]
                    
                    # Create label (next action)
                    action_encoding = {'search': 0, 'view': 1, 'upload': 2, 'browse': 3, 'logout': 4}
                    next_action = sequence[-1]['next_action']
                    label = action_encoding.get(next_action, 0)
                    
                    X.append(features)
                    y.append(label)
        
        return np.array(X), np.array(y)
    
    def prepare_matching_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare item matching data for training"""
        X, y = [], []
        
        for item in data:
            # Combine lost and found features
            combined_features = item['lost_features'] + item['found_features']
            X.append(combined_features)
            
            # Binary classification: match or no match
            y.append(1 if item['is_match'] else 0)
        
        return np.array(X), np.array(y)
    
    def prepare_text_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Prepare text description data for training"""
        # Build vocabulary
        texts = [item['text'] for item in data]
        vocab = self.manager.build_vocabulary(texts, min_freq=2)
        
        # Convert texts to sequences
        X, y = [], []
        max_length = 30
        
        for item in data:
            sequence = self.manager.text_to_sequence(item['text'], max_length)
            X.append(sequence)
            y.append(item['label'])
        
        return np.array(X), np.array(y), vocab
    
    def prepare_temporal_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare temporal data for training"""
        X, y = [], []
        
        for item in data:
            X.append(item['features'])
            y.append(item['label'])
        
        return np.array(X), np.array(y)
    
    def train_item_matching_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                                 X_val: np.ndarray, y_val: np.ndarray, 
                                 epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train the item matching model"""
        self.logger.info("Training item matching model...")
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Calculate class weights for imbalanced dataset
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train.cpu().numpy()), y=y_train.cpu().numpy())
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(self.manager.item_matching_model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        early_stopping_patience = 20
        
        for epoch in range(epochs):
            # Training
            self.manager.item_matching_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs, _ = self.manager.item_matching_model(batch_X, return_attention=True)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.manager.item_matching_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation
            self.manager.item_matching_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_outputs, _ = self.manager.item_matching_model(X_val, return_attention=True)
                val_loss = criterion(val_outputs, y_val).item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total = y_val.size(0)
                val_correct = (val_predicted == y_val).sum().item()
            
            # Calculate accuracies
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            # Update training history
            self.training_history['item_matching']['train_loss'].append(train_loss / len(train_loader))
            self.training_history['item_matching']['val_loss'].append(val_loss)
            self.training_history['item_matching']['train_acc'].append(train_acc)
            self.training_history['item_matching']['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.manager.item_matching_model.state_dict(), 
                          os.path.join(self.model_dir, 'checkpoints', 'best_item_matching.pth'))
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                               f"Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, "
                               f"Val Acc = {val_acc:.4f}")
            
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.manager.item_matching_model.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'checkpoints', 'best_item_matching.pth')))
        
        return {
            'best_val_acc': best_val_acc,
            'final_epoch': epoch,
            'train_losses': self.training_history['item_matching']['train_loss'],
            'val_losses': self.training_history['item_matching']['val_loss'],
            'train_accs': self.training_history['item_matching']['train_acc'],
            'val_accs': self.training_history['item_matching']['val_acc']
        }
    
    def train_behavior_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           epochs: int = 80, batch_size: int = 32) -> Dict:
        """Train the user behavior prediction model"""
        self.logger.info("Training behavior prediction model...")
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train.cpu().numpy()), y=y_train.cpu().numpy())
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(self.manager.behavior_predictor.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        early_stopping_patience = 15
        
        for epoch in range(epochs):
            # Training
            self.manager.behavior_predictor.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs, _ = self.manager.behavior_predictor(batch_X, return_attention=True)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.manager.behavior_predictor.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation
            self.manager.behavior_predictor.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_outputs, _ = self.manager.behavior_predictor(X_val, return_attention=True)
                val_loss = criterion(val_outputs, y_val).item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total = y_val.size(0)
                val_correct = (val_predicted == y_val).sum().item()
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            # Update training history
            self.training_history['behavior']['train_loss'].append(train_loss / len(train_loader))
            self.training_history['behavior']['val_loss'].append(val_loss)
            self.training_history['behavior']['train_acc'].append(train_acc)
            self.training_history['behavior']['val_acc'].append(val_acc)
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.manager.behavior_predictor.state_dict(),
                          os.path.join(self.model_dir, 'checkpoints', 'best_behavior.pth'))
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                               f"Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, "
                               f"Val Acc = {val_acc:.4f}")
            
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.manager.behavior_predictor.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'checkpoints', 'best_behavior.pth')))
        
        return {
            'best_val_acc': best_val_acc,
            'final_epoch': epoch,
            'train_losses': self.training_history['behavior']['train_loss'],
            'val_losses': self.training_history['behavior']['val_loss'],
            'train_accs': self.training_history['behavior']['train_acc'],
            'val_accs': self.training_history['behavior']['val_acc']
        }
    
    def train_text_model(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        epochs: int = 60, batch_size: int = 64) -> Dict:
        """Train the text description classification model"""
        self.logger.info("Training text classification model...")
        
        # Convert to tensors
        X_train = torch.LongTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.LongTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train.cpu().numpy()), y=y_train.cpu().numpy())
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(self.manager.text_encoder.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        early_stopping_patience = 15
        
        for epoch in range(epochs):
            # Training
            self.manager.text_encoder.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs, _ = self.manager.text_encoder(batch_X, return_attention=True)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.manager.text_encoder.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation
            self.manager.text_encoder.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_outputs, _ = self.manager.text_encoder(X_val, return_attention=True)
                val_loss = criterion(val_outputs, y_val).item()
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total = y_val.size(0)
                val_correct = (val_predicted == y_val).sum().item()
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            # Update training history
            self.training_history['text']['train_loss'].append(train_loss / len(train_loader))
            self.training_history['text']['val_loss'].append(val_loss)
            self.training_history['text']['train_acc'].append(train_acc)
            self.training_history['text']['val_acc'].append(val_acc)
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.manager.text_encoder.state_dict(),
                          os.path.join(self.model_dir, 'checkpoints', 'best_text.pth'))
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                               f"Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, "
                               f"Val Acc = {val_acc:.4f}")
            
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.manager.text_encoder.load_state_dict(
            torch.load(os.path.join(self.model_dir, 'checkpoints', 'best_text.pth')))
        
        return {
            'best_val_acc': best_val_acc,
            'final_epoch': epoch,
            'train_losses': self.training_history['text']['train_loss'],
            'val_losses': self.training_history['text']['val_loss'],
            'train_accs': self.training_history['text']['train_acc'],
            'val_accs': self.training_history['text']['val_acc']
        }
    
    def plot_training_results(self, results: Dict, model_name: str):
        """Plot training results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        ax1.plot(results['train_losses'], label='Training Loss', color='blue')
        ax1.plot(results['val_losses'], label='Validation Loss', color='red')
        ax1.set_title(f'{model_name} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(results['train_accs'], label='Training Accuracy', color='blue')
        ax2.plot(results['val_accs'], label='Validation Accuracy', color='red')
        ax2.set_title(f'{model_name} - Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Plot learning curves
        ax3.plot(results['train_losses'], label='Train Loss', color='blue', alpha=0.7)
        ax3.plot(results['val_losses'], label='Val Loss', color='red', alpha=0.7)
        ax3.set_title(f'{model_name} - Learning Curves')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True)
        
        # Plot accuracy comparison
        ax4.plot(results['train_accs'], label='Train Acc', color='blue', alpha=0.7)
        ax4.plot(results['val_accs'], label='Val Acc', color='red', alpha=0.7)
        ax4.set_title(f'{model_name} - Accuracy Comparison')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'plots', f'{model_name.lower().replace(" ", "_")}_training.png'))
        plt.show()
    
    def evaluate_models(self, X_test: Dict, y_test: Dict) -> Dict:
        """Evaluate all trained models"""
        self.logger.info("Evaluating models...")
        
        evaluation_results = {}
        
        # Evaluate item matching model
        if 'matching' in X_test:
            self.manager.item_matching_model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test['matching']).to(self.device)
                y_test_tensor = torch.LongTensor(y_test['matching']).to(self.device)
                
                outputs = self.manager.item_matching_model(X_test_tensor)
                predictions = torch.argmax(outputs, dim=1)
                
                accuracy = accuracy_score(y_test_tensor.cpu(), predictions.cpu())
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test_tensor.cpu(), predictions.cpu(), average='weighted')
                
                evaluation_results['item_matching'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
        
        # Evaluate behavior model
        if 'behavior' in X_test:
            self.manager.behavior_predictor.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test['behavior']).to(self.device)
                y_test_tensor = torch.LongTensor(y_test['behavior']).to(self.device)
                
                outputs = self.manager.behavior_predictor(X_test_tensor)
                predictions = torch.argmax(outputs, dim=1)
                
                accuracy = accuracy_score(y_test_tensor.cpu(), predictions.cpu())
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test_tensor.cpu(), predictions.cpu(), average='weighted')
                
                evaluation_results['behavior'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
        
        # Evaluate text model
        if 'text' in X_test:
            self.manager.text_encoder.eval()
            with torch.no_grad():
                X_test_tensor = torch.LongTensor(X_test['text']).to(self.device)
                y_test_tensor = torch.LongTensor(y_test['text']).to(self.device)
                
                outputs = self.manager.text_encoder(X_test_tensor)
                predictions = torch.argmax(outputs, dim=1)
                
                accuracy = accuracy_score(y_test_tensor.cpu(), predictions.cpu())
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test_tensor.cpu(), predictions.cpu(), average='weighted')
                
                evaluation_results['text'] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
        
        return evaluation_results
    
    def save_training_results(self, results: Dict):
        """Save training results and model info"""
        # Save training history
        with open(os.path.join(self.model_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save evaluation results
        with open(os.path.join(self.model_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save model info
        model_info = self.manager.get_model_info()
        with open(os.path.join(self.model_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        self.logger.info(f"Training results saved to {self.model_dir}")
    
    def train_all_models(self, data_dir: str = 'enhanced_datasets/'):
        """Train all custom RNN models"""
        self.logger.info("Starting comprehensive training of custom RNN models...")
        
        # Load datasets
        datasets = self.load_enhanced_datasets(data_dir)
        
        # Prepare data
        self.logger.info("Preparing training data...")
        
        # Behavior data
        X_behavior, y_behavior = self.prepare_behavior_data(datasets['behavior'])
        X_behavior_train, X_behavior_val, y_behavior_train, y_behavior_val = train_test_split(
            X_behavior, y_behavior, test_size=0.2, random_state=42)
        
        # Matching data
        X_matching, y_matching = self.prepare_matching_data(datasets['matching'])
        X_matching_train, X_matching_val, y_matching_train, y_matching_val = train_test_split(
            X_matching, y_matching, test_size=0.2, random_state=42)
        
        # Text data
        X_text, y_text, vocab = self.prepare_text_data(datasets['text'])
        X_text_train, X_text_val, y_text_train, y_text_val = train_test_split(
            X_text, y_text, test_size=0.2, random_state=42)
        
        # Update manager vocabulary
        self.manager.vocab = vocab
        self.manager.vocab_size = len(vocab)
        
        # Train models
        self.logger.info("Training item matching model...")
        matching_results = self.train_item_matching_model(
            X_matching_train, y_matching_train, X_matching_val, y_matching_val)
        
        self.logger.info("Training behavior prediction model...")
        behavior_results = self.train_behavior_model(
            X_behavior_train, y_behavior_train, X_behavior_val, y_behavior_val)
        
        self.logger.info("Training text classification model...")
        text_results = self.train_text_model(
            X_text_train, y_text_train, X_text_val, y_text_val)
        
        # Plot results
        self.plot_training_results(matching_results, "Item Matching")
        self.plot_training_results(behavior_results, "Behavior Prediction")
        self.plot_training_results(text_results, "Text Classification")
        
        # Evaluate models
        test_data = {
            'matching': X_matching_val,
            'behavior': X_behavior_val,
            'text': X_text_val
        }
        test_labels = {
            'matching': y_matching_val,
            'behavior': y_behavior_val,
            'text': y_text_val
        }
        
        evaluation_results = self.evaluate_models(test_data, test_labels)
        
        # Save results
        self.save_training_results(evaluation_results)
        
        # Save final models
        self.manager.save_models()
        
        self.logger.info("Training completed successfully!")
        return evaluation_results

def main():
    """Main training function"""
    print("🚀 Custom RNN Training Pipeline for BARYONYX")
    print("=" * 60)
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = CustomRNNTrainer(device=device)
    
    # Train all models
    results = trainer.train_all_models()
    
    # Print results
    print("\n📊 Training Results:")
    print("=" * 30)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n✅ Custom RNN training completed successfully!")

if __name__ == "__main__":
    main()
