
"""
Training Script for RNN Models in BARYONYX System
Trains all three RNN models using the generated datasets
"""

import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Import the RNN models
from rnn_models import RNNModelManager, UserBehaviorLSTM, BidirectionalDescriptionRNN, TemporalPatternRNN

def load_training_data():
    """Load the generated training datasets"""
    
    # Load user behavior dataset
    with open('rnn_user_behavior_dataset.json', 'r') as f:
        user_behavior_data = json.load(f)
    
    # Load description dataset
    description_data = []
    with open('rnn_description_dataset.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            description_data.append({
                'text': row['text'],
                'label': int(row['label']),
                'category': row['category']
            })
    
    # Load temporal dataset
    with open('rnn_temporal_dataset.json', 'r') as f:
        temporal_data = json.load(f)
    
    return user_behavior_data, description_data, temporal_data

def prepare_user_behavior_data(data, sequence_length=10):
    """Prepare user behavior data for training"""
    
    # Group data by user (simulate user sequences)
    user_sequences = {}
    for i, sample in enumerate(data):
        user_id = f"user_{i % 50}"  # Simulate 50 users
        if user_id not in user_sequences:
            user_sequences[user_id] = []
        user_sequences[user_id].append(sample)
    
    # Create sequences with better data augmentation
    X, y = [], []
    for user_id, actions in user_sequences.items():
        if len(actions) >= sequence_length:
            # Create multiple sequences with different starting points
            for i in range(len(actions) - sequence_length + 1):
                sequence = actions[i:i + sequence_length]
                features = [action['features'] for action in sequence]
                label = sequence[-1]['label']  # Next action
                X.append(features)
                y.append(label)
            
            # Add some overlapping sequences for more training data
            if len(actions) >= sequence_length + 5:
                for i in range(0, len(actions) - sequence_length, 3):  # Every 3rd position
                    sequence = actions[i:i + sequence_length]
                    features = [action['features'] for action in sequence]
                    label = sequence[-1]['label']
                    X.append(features)
                    y.append(label)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = np.array(X)[indices]
    y = np.array(y)[indices]
    
    print(f"Created {len(X)} user behavior sequences")
    return X, y

def prepare_description_data(data, vocab_size=1000):
    """Prepare description data for training"""
    
    # Build vocabulary
    vocab = {'<PAD>': 0, '<UNK>': 1}
    word_count = {}
    
    for sample in data:
        words = sample['text'].lower().split()
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
    
    # Add most frequent words to vocabulary
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_words[:vocab_size-2]:  # -2 for PAD and UNK
        vocab[word] = len(vocab)
    
    # Convert texts to sequences
    X, y = [], []
    max_length = 20
    
    for sample in data:
        words = sample['text'].lower().split()
        word_indices = [vocab.get(word, vocab['<UNK>']) for word in words]
        
        # Pad or truncate
        if len(word_indices) < max_length:
            word_indices.extend([vocab['<PAD>']] * (max_length - len(word_indices)))
        else:
            word_indices = word_indices[:max_length]
        
        X.append(word_indices)
        y.append(sample['label'])
    
    return np.array(X), np.array(y), vocab

def prepare_temporal_data(data):
    """Prepare temporal data for training"""
    
    X, y = [], []
    for sample in data:
        X.append(sample['features'])
        y.append(sample['label'])
    
    return np.array(X), np.array(y)

def train_user_behavior_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=16):
    """Train the user behavior LSTM model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Calculate class weights for imbalanced dataset
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train.cpu().numpy()), y=y_train.cpu().numpy())
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    # Loss and optimizer with improved settings
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # Use class weights
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)  # Higher learning rate, weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Training loop
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 20
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs, attention = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs, _ = model(X_val)
            val_pred = torch.argmax(val_outputs, dim=1)
            val_acc = accuracy_score(y_val.cpu(), val_pred.cpu())
        
        train_losses.append(total_loss / len(train_loader))
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {train_losses[-1]:.4f}, Val Acc = {val_acc:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (patience: {early_stopping_patience})")
            break
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return train_losses, val_accuracies

def train_description_model(model, X_train, y_train, X_val, y_val, vocab, epochs=50, batch_size=32):
    """Train the description bidirectional RNN model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Update model parameters
    model.vocab_size = len(vocab)
    model.embedding = nn.Embedding(len(vocab), model.embedding.embedding_dim).to(device)
    
    # Convert to tensors
    X_train = torch.LongTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    X_val = torch.LongTensor(X_val).to(device)
    y_val = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs, attention = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs, _ = model(X_val)
            val_pred = torch.argmax(val_outputs, dim=1)
            val_acc = accuracy_score(y_val.cpu(), val_pred.cpu())
        
        train_losses.append(total_loss / len(train_loader))
        val_accuracies.append(val_acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {train_losses[-1]:.4f}, Val Acc = {val_acc:.4f}")
    
    return train_losses, val_accuracies, vocab

def train_temporal_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train the temporal pattern RNN model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).unsqueeze(1).to(device)  # Add sequence dimension
    y_train = torch.LongTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).unsqueeze(1).to(device)
    y_val = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_pred = torch.argmax(val_outputs, dim=1)
            val_acc = accuracy_score(y_val.cpu(), val_pred.cpu())
        
        train_losses.append(total_loss / len(train_loader))
        val_accuracies.append(val_acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {train_losses[-1]:.4f}, Val Acc = {val_acc:.4f}")
    
    return train_losses, val_accuracies

def plot_training_results(results, title):
    """Plot training results"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(results['train_losses'])
    ax1.set_title(f'{title} - Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(results['val_accuracies'])
    ax2.set_title(f'{title} - Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main training function"""
    
    print("🚀 Starting RNN Model Training for BARYONYX System")
    print("=" * 60)
    
    # Load training data
    print("📊 Loading training datasets...")
    user_behavior_data, description_data, temporal_data = load_training_data()
    
    print(f"User Behavior Data: {len(user_behavior_data)} samples")
    print(f"Description Data: {len(description_data)} samples")
    print(f"Temporal Data: {len(temporal_data)} samples")
    
    # Initialize RNN manager
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    rnn_manager = RNNModelManager(device=device)
    
    # Update user behavior model to have 5 output classes (search, upload, view, browse, logout)
    # Also improve the model architecture
    rnn_manager.user_behavior_model.fc2 = nn.Linear(32, 5).to(device)
    
    # Add dropout and batch normalization for better training
    rnn_manager.user_behavior_model.dropout = nn.Dropout(0.3)
    rnn_manager.user_behavior_model.batch_norm = nn.BatchNorm1d(32)
    
    # Update description model to have 4 output classes (phone, mouse, wallet, tumbler)
    # The model uses hidden_size * 2 for bidirectional output, so 64 * 2 = 128
    rnn_manager.description_model.fc = nn.Linear(128, 4).to(device)
    
    # Train User Behavior Model
    print("\n🧠 Training User Behavior LSTM...")
    X_behavior, y_behavior = prepare_user_behavior_data(user_behavior_data)
    X_train, X_val, y_train, y_val = train_test_split(X_behavior, y_behavior, test_size=0.2, random_state=42)
    
    behavior_results = train_user_behavior_model(
        rnn_manager.user_behavior_model, X_train, y_train, X_val, y_val
    )
    
    # Train Description Model
    print("\n📝 Training Description Bidirectional RNN...")
    X_desc, y_desc, vocab = prepare_description_data(description_data)
    X_train, X_val, y_train, y_val = train_test_split(X_desc, y_desc, test_size=0.2, random_state=42)
    
    desc_results = train_description_model(
        rnn_manager.description_model, X_train, y_train, X_val, y_val, vocab
    )
    
    # Update vocabulary in manager
    rnn_manager.vocab = vocab
    rnn_manager.vocab_size = len(vocab)
    
    # Train Temporal Model
    print("\n⏰ Training Temporal Pattern RNN...")
    X_temp, y_temp = prepare_temporal_data(temporal_data)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    temp_results = train_temporal_model(
        rnn_manager.temporal_model, X_train, y_train, X_val, y_val
    )
    
    # Save trained models
    print("\n💾 Saving trained models...")
    rnn_manager.save_models()
    
    # Plot results
    print("\n📈 Training Results:")
    plot_training_results({
        'train_losses': behavior_results[0],
        'val_accuracies': behavior_results[1]
    }, "User Behavior LSTM")
    
    plot_training_results({
        'train_losses': desc_results[0],
        'val_accuracies': desc_results[1]
    }, "Description RNN")
    
    plot_training_results({
        'train_losses': temp_results[0],
        'val_accuracies': temp_results[1]
    }, "Temporal Pattern RNN")
    
    print("\n✅ Training completed successfully!")
    print("Models saved to models/rnn_models/ directory")

if __name__ == "__main__":
    main()
