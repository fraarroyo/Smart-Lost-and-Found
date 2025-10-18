"""
Training script for the Unified RNN Model

This script trains a single model that handles all three RNN tasks:
1. User Behavior Prediction
2. Text Description Classification
3. Temporal Pattern Recognition
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append('.')

from unified_rnn_model import UnifiedRNNManager
from rnn_training_datasets import create_user_behavior_dataset, create_description_dataset, create_temporal_dataset

def prepare_behavior_data(data, sequence_length=10):
    """Prepare user behavior data for training"""
    user_sequences = {}
    for i, sample in enumerate(data):
        user_id = f"user_{i % 50}"
        if user_id not in user_sequences:
            user_sequences[user_id] = []
        user_sequences[user_id].append(sample)
    
    X, y = [], []
    for user_id, actions in user_sequences.items():
        if len(actions) >= sequence_length:
            for i in range(len(actions) - sequence_length + 1):
                sequence = actions[i:i + sequence_length]
                features = [action['features'] for action in sequence]
                label = sequence[-1]['label']
                X.append(features)
                y.append(label)
    
    indices = np.random.permutation(len(X))
    X = np.array(X)[indices]
    y = np.array(y)[indices]
    
    print(f"Created {len(X)} user behavior sequences")
    return X, y

def prepare_description_data(data, manager):
    """Prepare description data for training"""
    texts = [sample['text'] for sample in data]
    labels = [sample['label'] for sample in data]
    
    # Build vocabulary
    manager.build_vocabulary(texts)
    
    # Convert texts to sequences
    sequences = []
    for text in texts:
        sequence = manager.text_to_sequence(text)
        sequences.append(sequence)
    
    return np.array(sequences), np.array(labels)

def prepare_temporal_data(data, sequence_length=7):
    """Prepare temporal data for training"""
    X, y = [], []
    
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i + sequence_length]
        features = [sample['features'] for sample in sequence]
        label = sequence[-1]['label']
        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y)

def train_unified_model(manager, behavior_data, description_data, temporal_data, epochs=50, batch_size=32):
    """Train the unified RNN model on all three tasks"""
    
    device = manager.device
    model = manager.model
    
    print("🚀 Starting Unified RNN Model Training")
    print("=" * 50)
    
    # Prepare data for all three tasks
    print("\n📊 Preparing training data...")
    
    # User Behavior Data
    X_behavior, y_behavior = prepare_behavior_data(behavior_data)
    X_behavior_train, X_behavior_val, y_behavior_train, y_behavior_val = train_test_split(
        X_behavior, y_behavior, test_size=0.2, random_state=42
    )
    
    # Description Data
    X_desc, y_desc = prepare_description_data(description_data, manager)
    X_desc_train, X_desc_val, y_desc_train, y_desc_val = train_test_split(
        X_desc, y_desc, test_size=0.2, random_state=42
    )
    
    # Temporal Data
    X_temp, y_temp = prepare_temporal_data(temporal_data)
    X_temp_train, X_temp_val, y_temp_train, y_temp_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )
    
    print(f"Behavior: {len(X_behavior_train)} train, {len(X_behavior_val)} val")
    print(f"Description: {len(X_desc_train)} train, {len(X_desc_val)} val")
    print(f"Temporal: {len(X_temp_train)} train, {len(X_temp_val)} val")
    
    # Convert to tensors
    X_behavior_train = torch.FloatTensor(X_behavior_train).to(device)
    y_behavior_train = torch.LongTensor(y_behavior_train).to(device)
    X_behavior_val = torch.FloatTensor(X_behavior_val).to(device)
    y_behavior_val = torch.LongTensor(y_behavior_val).to(device)
    
    X_desc_train = torch.LongTensor(X_desc_train).to(device)
    y_desc_train = torch.LongTensor(y_desc_train).to(device)
    X_desc_val = torch.LongTensor(X_desc_val).to(device)
    y_desc_val = torch.LongTensor(y_desc_val).to(device)
    
    X_temp_train = torch.FloatTensor(X_temp_train).to(device)
    y_temp_train = torch.LongTensor(y_temp_train).to(device)
    X_temp_val = torch.FloatTensor(X_temp_val).to(device)
    y_temp_val = torch.LongTensor(y_temp_val).to(device)
    
    # Create data loaders
    behavior_train_loader = DataLoader(
        TensorDataset(X_behavior_train, y_behavior_train),
        batch_size=batch_size, shuffle=True
    )
    behavior_val_loader = DataLoader(
        TensorDataset(X_behavior_val, y_behavior_val),
        batch_size=batch_size, shuffle=False
    )
    
    desc_train_loader = DataLoader(
        TensorDataset(X_desc_train, y_desc_train),
        batch_size=batch_size, shuffle=True
    )
    desc_val_loader = DataLoader(
        TensorDataset(X_desc_val, y_desc_val),
        batch_size=batch_size, shuffle=False
    )
    
    temp_train_loader = DataLoader(
        TensorDataset(X_temp_train, y_temp_train),
        batch_size=batch_size, shuffle=True
    )
    temp_val_loader = DataLoader(
        TensorDataset(X_temp_val, y_temp_val),
        batch_size=batch_size, shuffle=False
    )
    
    # Loss functions and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training history
    history = {
        'behavior': {'train_loss': [], 'val_acc': []},
        'description': {'train_loss': [], 'val_acc': []},
        'temporal': {'train_loss': [], 'val_acc': []}
    }
    
    best_val_acc = 0.0
    
    print(f"\n🎯 Training for {epochs} epochs...")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        
        # Training phase
        behavior_loss = 0.0
        desc_loss = 0.0
        temp_loss = 0.0
        
        # Train on behavior data
        for batch_X, batch_y in behavior_train_loader:
            optimizer.zero_grad()
            outputs, _ = model(batch_X, 'behavior')
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            behavior_loss += loss.item()
        
        # Train on description data
        for batch_X, batch_y in desc_train_loader:
            optimizer.zero_grad()
            outputs, _ = model(batch_X, 'description')
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            desc_loss += loss.item()
        
        # Train on temporal data
        for batch_X, batch_y in temp_train_loader:
            optimizer.zero_grad()
            outputs, _ = model(batch_X, 'temporal')
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            temp_loss += loss.item()
        
        # Validation phase
        model.eval()
        behavior_val_acc = 0.0
        desc_val_acc = 0.0
        temp_val_acc = 0.0
        
        with torch.no_grad():
            # Behavior validation
            behavior_preds = []
            for batch_X, batch_y in behavior_val_loader:
                outputs, _ = model(batch_X, 'behavior')
                preds = torch.argmax(outputs, dim=1)
                behavior_preds.extend(preds.cpu().numpy())
            behavior_val_acc = accuracy_score(y_behavior_val.cpu().numpy(), behavior_preds)
            
            # Description validation
            desc_preds = []
            for batch_X, batch_y in desc_val_loader:
                outputs, _ = model(batch_X, 'description')
                preds = torch.argmax(outputs, dim=1)
                desc_preds.extend(preds.cpu().numpy())
            desc_val_acc = accuracy_score(y_desc_val.cpu().numpy(), desc_preds)
            
            # Temporal validation
            temp_preds = []
            for batch_X, batch_y in temp_val_loader:
                outputs, _ = model(batch_X, 'temporal')
                preds = torch.argmax(outputs, dim=1)
                temp_preds.extend(preds.cpu().numpy())
            temp_val_acc = accuracy_score(y_temp_val.cpu().numpy(), temp_preds)
        
        # Update history
        history['behavior']['train_loss'].append(behavior_loss / len(behavior_train_loader))
        history['behavior']['val_acc'].append(behavior_val_acc)
        history['description']['train_loss'].append(desc_loss / len(desc_train_loader))
        history['description']['val_acc'].append(desc_val_acc)
        history['temporal']['train_loss'].append(temp_loss / len(temp_train_loader))
        history['temporal']['val_acc'].append(temp_val_acc)
        
        # Calculate average validation accuracy
        avg_val_acc = (behavior_val_acc + desc_val_acc + temp_val_acc) / 3
        
        # Learning rate scheduling
        scheduler.step(avg_val_acc)
        
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: "
                  f"Behavior Loss: {behavior_loss/len(behavior_train_loader):.4f}, Val Acc: {behavior_val_acc:.4f} | "
                  f"Desc Loss: {desc_loss/len(desc_train_loader):.4f}, Val Acc: {desc_val_acc:.4f} | "
                  f"Temp Loss: {temp_loss/len(temp_train_loader):.4f}, Val Acc: {temp_val_acc:.4f} | "
                  f"Avg Val Acc: {avg_val_acc:.4f}")
        
        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            manager.save_model()
    
    print(f"\n✅ Training completed!")
    print(f"Best average validation accuracy: {best_val_acc:.4f}")
    
    return history

def create_training_plots(history):
    """Create training visualization plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    tasks = ['behavior', 'description', 'temporal']
    task_names = ['User Behavior', 'Description', 'Temporal']
    
    for i, (task, name) in enumerate(zip(tasks, task_names)):
        # Loss plot
        axes[0, i].plot(history[task]['train_loss'], label='Training Loss')
        axes[0, i].set_title(f'{name} - Training Loss')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1, i].plot(history[task]['val_acc'], label='Validation Accuracy', color='green')
        axes[1, i].set_title(f'{name} - Validation Accuracy')
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Accuracy')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('unified_rnn_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    print("🚀 Unified RNN Model Training for BARYONYX System")
    print("=" * 60)
    
    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize unified model manager
    manager = UnifiedRNNManager(device=device)
    
    # Load training datasets
    print("\n📊 Loading training datasets...")
    behavior_data = create_user_behavior_dataset(1000)
    description_data = create_description_dataset(800)
    temporal_data = create_temporal_dataset(600)
    
    print(f"User Behavior Data: {len(behavior_data)} samples")
    print(f"Description Data: {len(description_data)} samples")
    print(f"Temporal Data: {len(temporal_data)} samples")
    
    # Train the unified model
    history = train_unified_model(
        manager, behavior_data, description_data, temporal_data,
        epochs=50, batch_size=32
    )
    
    # Create training plots
    print("\n📈 Creating training visualizations...")
    create_training_plots(history)
    
    # Test the trained model
    print("\n🧪 Testing trained model...")
    
    # Test behavior prediction
    sample_behavior = torch.randn(1, 10, 10).to(device)
    behavior_pred, behavior_probs, _ = manager.predict_behavior(sample_behavior)
    print(f"Behavior Prediction: {behavior_pred}, Confidence: {behavior_probs.max():.3f}")
    
    # Test description classification
    test_texts = [
        "Lost black iPhone 12 with cracked screen",
        "Found wireless mouse Logitech black",
        "Lost wallet brown leather with cards",
        "Found tumbler stainless steel silver"
    ]
    
    print("\nDescription Classification Tests:")
    for text in test_texts:
        desc_pred, desc_probs, _, categories = manager.classify_description(text)
        print(f"  '{text}' -> {categories[desc_pred]} (conf: {desc_probs.max():.3f})")
    
    # Test temporal prediction
    sample_temporal = torch.randn(1, 7, 10).to(device)
    temporal_pred, temporal_probs, time_periods = manager.predict_temporal_pattern(sample_temporal)
    print(f"\nTemporal Prediction: {temporal_pred} ({time_periods[temporal_pred]}), Confidence: {temporal_probs.max():.3f}")
    
    # Final model info
    print(f"\n📊 Final Model Information:")
    info = manager.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Unified RNN Model training completed successfully!")
    print(f"Model saved to: {manager.model_path}")

if __name__ == "__main__":
    main()

