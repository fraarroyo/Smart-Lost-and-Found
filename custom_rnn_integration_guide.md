# Custom RNN Integration Guide for BARYONYX Lost & Found System

This guide explains how to integrate the custom RNN models into the main BARYONYX application.

## Overview

The custom RNN system consists of three specialized models:

1. **Item Matching RNN** - Advanced item matching with attention mechanisms
2. **User Behavior Predictor** - Predicts user actions and preferences
3. **Text Description Encoder** - Classifies and processes item descriptions

## Quick Start

### 1. Generate Training Data

```bash
python enhanced_rnn_datasets.py
```

This creates realistic training datasets in the `enhanced_datasets/` directory.

### 2. Train the Models

```bash
python custom_rnn_training.py
```

This trains all three models with comprehensive validation and early stopping.

### 3. Evaluate Performance

```bash
python custom_rnn_evaluation.py
```

This generates detailed evaluation reports and visualizations.

## Integration with Main Application

### Basic Integration

```python
from custom_rnn_model import CustomRNNManager

# Initialize the manager
rnn_manager = CustomRNNManager(device='cuda' if torch.cuda.is_available() else 'cpu')

# Load trained models
rnn_manager.load_models()

# Use the models
```

### Item Matching Integration

```python
# In your item matching logic
def find_matching_items(lost_item_data, found_items_list):
    """Find matching items using custom RNN"""
    matches = []
    
    # Create features for lost item
    lost_features = rnn_manager.create_item_features(lost_item_data)
    
    for found_item in found_items_list:
        # Create features for found item
        found_features = rnn_manager.create_item_features(found_item)
        
        # Get matching prediction
        result = rnn_manager.predict_item_match(lost_features)
        
        if result['confidence'] > 0.7:  # Threshold for high confidence
            matches.append({
                'item': found_item,
                'confidence': result['confidence'],
                'category': result['category']
            })
    
    return sorted(matches, key=lambda x: x['confidence'], reverse=True)
```

### User Behavior Integration

```python
# In your user session handling
def track_user_behavior(user_id, action, item_type=None):
    """Track and predict user behavior"""
    # Add action to user sequence
    rnn_manager.add_user_action(user_id, action, item_type)
    
    # Get behavior prediction
    behavior_result = rnn_manager.predict_user_behavior(user_sequence)
    
    # Use prediction for recommendations
    if behavior_result['confidence'] > 0.6:
        return {
            'next_action': behavior_result['action'],
            'suggestions': behavior_result.get('suggestions', []),
            'confidence': behavior_result['confidence']
        }
    
    return None
```

### Text Classification Integration

```python
# In your item description processing
def classify_item_description(description):
    """Classify item description using custom RNN"""
    result = rnn_manager.classify_description(description)
    
    return {
        'category': result['category'],
        'confidence': result['confidence'],
        'probabilities': result['probabilities']
    }
```

## Advanced Integration

### Custom Feature Creation

```python
def create_custom_item_features(item_data):
    """Create custom features for specific use cases"""
    features = rnn_manager.create_item_features(item_data)
    
    # Add custom features
    features.extend([
        item_data.get('custom_feature_1', 0.0),
        item_data.get('custom_feature_2', 0.0),
        # ... more custom features
    ])
    
    return features
```

### Model Fine-tuning

```python
def fine_tune_model(model_name, new_data, epochs=10):
    """Fine-tune a specific model with new data"""
    if model_name == 'item_matching':
        model = rnn_manager.item_matching_model
    elif model_name == 'behavior':
        model = rnn_manager.behavior_predictor
    elif model_name == 'text':
        model = rnn_manager.text_encoder
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Fine-tuning logic here
    # ... (implement based on your needs)
    
    # Save updated model
    rnn_manager.save_models()
```

## API Endpoints Integration

### Flask Route Examples

```python
from flask import Flask, request, jsonify
from custom_rnn_model import CustomRNNManager

app = Flask(__name__)
rnn_manager = CustomRNNManager()
rnn_manager.load_models()

@app.route('/api/predict-match', methods=['POST'])
def predict_match():
    """API endpoint for item matching prediction"""
    data = request.json
    
    lost_item = data.get('lost_item')
    found_item = data.get('found_item')
    
    # Create features
    lost_features = rnn_manager.create_item_features(lost_item)
    found_features = rnn_manager.create_item_features(found_item)
    
    # Get prediction
    result = rnn_manager.predict_item_match(lost_features)
    
    return jsonify({
        'match': result['confidence'] > 0.7,
        'confidence': result['confidence'],
        'category': result['category']
    })

@app.route('/api/predict-behavior', methods=['POST'])
def predict_behavior():
    """API endpoint for user behavior prediction"""
    data = request.json
    
    user_id = data.get('user_id')
    behavior_sequence = data.get('behavior_sequence', [])
    
    # Get prediction
    result = rnn_manager.predict_user_behavior(behavior_sequence)
    
    return jsonify({
        'predicted_action': result['action'],
        'confidence': result['confidence'],
        'suggestions': result.get('suggestions', [])
    })

@app.route('/api/classify-description', methods=['POST'])
def classify_description():
    """API endpoint for text classification"""
    data = request.json
    
    description = data.get('description', '')
    
    # Get classification
    result = rnn_manager.classify_description(description)
    
    return jsonify({
        'category': result['category'],
        'confidence': result['confidence'],
        'probabilities': result['probabilities']
    })
```

## Performance Optimization

### Batch Processing

```python
def batch_predict_matches(lost_items, found_items):
    """Process multiple matches in batch for better performance"""
    results = []
    
    # Prepare batch data
    batch_features = []
    for lost_item, found_item in zip(lost_items, found_items):
        combined_features = (rnn_manager.create_item_features(lost_item) + 
                           rnn_manager.create_item_features(found_item))
        batch_features.append(combined_features)
    
    # Batch prediction
    batch_tensor = torch.FloatTensor(batch_features).to(rnn_manager.device)
    
    with torch.no_grad():
        rnn_manager.item_matching_model.eval()
        outputs = rnn_manager.item_matching_model(batch_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    # Process results
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        results.append({
            'prediction': pred.item(),
            'confidence': prob[pred].item(),
            'probabilities': prob.cpu().numpy().tolist()
        })
    
    return results
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_classify_description(description):
    """Cached version of description classification"""
    return rnn_manager.classify_description(description)
```

## Monitoring and Maintenance

### Model Performance Monitoring

```python
def monitor_model_performance():
    """Monitor model performance over time"""
    # Load recent evaluation results
    with open('models/custom_rnn/evaluation/evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    # Check for performance degradation
    for model_name, metrics in results.items():
        if metrics['accuracy'] < 0.8:  # Threshold
            print(f"Warning: {model_name} accuracy below threshold")
    
    return results
```

### Model Retraining

```python
def retrain_models_if_needed():
    """Retrain models if performance degrades"""
    current_performance = monitor_model_performance()
    
    for model_name, metrics in current_performance.items():
        if metrics['accuracy'] < 0.75:  # Retraining threshold
            print(f"Retraining {model_name}...")
            # Implement retraining logic
            # ... (call training pipeline)
```

## Configuration

### Model Configuration

```python
# config.py
RNN_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_dir': 'models/custom_rnn/',
    'batch_size': 32,
    'sequence_length': 10,
    'confidence_threshold': 0.7,
    'retrain_threshold': 0.75
}
```

### Environment Variables

```bash
# .env
RNN_DEVICE=cuda
RNN_MODEL_DIR=models/custom_rnn/
RNN_BATCH_SIZE=32
RNN_CONFIDENCE_THRESHOLD=0.7
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or use CPU
   rnn_manager = CustomRNNManager(device='cpu')
   ```

2. **Model Loading Errors**
   ```python
   # Check if models exist
   import os
   if not os.path.exists('models/custom_rnn/item_matching_model.pth'):
       print("Models not found. Run training first.")
   ```

3. **Vocabulary Not Found**
   ```python
   # Build vocabulary from your data
   texts = [item['description'] for item in your_data]
   rnn_manager.build_vocabulary(texts)
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debug mode in manager
rnn_manager = CustomRNNManager(device='cpu')
rnn_manager.logger.setLevel(logging.DEBUG)
```

## Best Practices

1. **Always validate input data** before passing to models
2. **Use appropriate confidence thresholds** for your use case
3. **Monitor model performance** regularly
4. **Retrain models** when performance degrades
5. **Use batch processing** for better performance
6. **Cache frequently used predictions**
7. **Handle edge cases** gracefully

## Support

For issues or questions:

1. Check the evaluation reports in `models/custom_rnn/evaluation/`
2. Review the training logs
3. Test with sample data using the evaluation scripts
4. Check model configuration and device compatibility

## Files Created

- `custom_rnn_model.py` - Main model definitions
- `enhanced_rnn_datasets.py` - Dataset generation
- `custom_rnn_training.py` - Training pipeline
- `custom_rnn_evaluation.py` - Evaluation system
- `custom_rnn_integration_guide.md` - This guide

## Next Steps

1. Run the complete pipeline: datasets → training → evaluation
2. Integrate models into your application
3. Monitor performance and retrain as needed
4. Customize features and thresholds for your specific use case
