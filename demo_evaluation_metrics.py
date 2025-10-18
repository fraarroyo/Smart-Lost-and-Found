#!/usr/bin/env python3
"""
Demonstration of Comprehensive Evaluation Metrics in Custom RNN Model
Shows how to use accuracy, F1 score, recall, IoU, mAP, and confusion matrix
"""

import numpy as np
import json
from custom_rnn_model import CustomRNNManager, ModelEvaluator

def create_sample_test_data():
    """Create sample test data for demonstration"""
    
    # Sample item matching data
    matching_data = []
    for i in range(100):
        # Create similar items (matches)
        if i < 50:
            lost_features = np.random.rand(20).tolist()
            found_features = lost_features + np.random.normal(0, 0.1, 20).tolist()  # Add noise
            is_match = True
        else:
            # Create different items (no matches)
            lost_features = np.random.rand(20).tolist()
            found_features = np.random.rand(20).tolist()
            is_match = False
        
        matching_data.append({
            'lost_features': lost_features,
            'found_features': found_features,
            'is_match': is_match
        })
    
    # Sample behavior data
    behavior_data = []
    action_encoding = {'search': 0, 'view': 1, 'upload': 2, 'browse': 3, 'logout': 4}
    actions = list(action_encoding.keys())
    
    for i in range(200):
        user_id = f"user_{i % 10}"  # 10 users
        features = np.random.rand(15).tolist()
        next_action = np.random.choice(actions)
        
        behavior_data.append({
            'user_id': user_id,
            'features': features,
            'next_action': next_action
        })
    
    # Sample text data
    text_data = []
    categories = ['phone', 'mouse', 'wallet', 'tumbler']
    descriptions = [
        "Lost black iPhone 12 with cracked screen",
        "Found wireless mouse Logitech black",
        "Lost wallet brown leather with cards",
        "Found tumbler stainless steel silver",
        "Lost Samsung Galaxy phone blue",
        "Found gaming mouse Razer red",
        "Lost leather wallet black",
        "Found water bottle blue plastic"
    ]
    
    for i in range(80):
        text_data.append({
            'text': np.random.choice(descriptions),
            'label': i % 4  # 4 categories
        })
    
    return {
        'matching': matching_data,
        'behavior': behavior_data,
        'text': text_data
    }

def demonstrate_evaluation_metrics():
    """Demonstrate comprehensive evaluation metrics"""
    
    print("🚀 Custom RNN Model - Comprehensive Evaluation Metrics Demo")
    print("=" * 70)
    
    # Initialize manager
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    manager = CustomRNNManager(device=device)
    
    # Create sample test data
    print("\n📊 Creating sample test data...")
    test_data = create_sample_test_data()
    
    print(f"  - Item matching samples: {len(test_data['matching'])}")
    print(f"  - Behavior samples: {len(test_data['behavior'])}")
    print(f"  - Text samples: {len(test_data['text'])}")
    
    # Build vocabulary for text model
    print("\n📝 Building vocabulary...")
    texts = [item['text'] for item in test_data['text']]
    manager.build_vocabulary(texts)
    
    # Run comprehensive evaluation
    print("\n🔍 Running comprehensive evaluation...")
    evaluation_results = manager.comprehensive_evaluation(test_data)
    
    # Display detailed results
    print("\n📈 Detailed Evaluation Results:")
    print("=" * 50)
    
    for model_name, results in evaluation_results.items():
        print(f"\n🔹 {model_name.upper().replace('_', ' ')} MODEL")
        print("-" * 40)
        
        # Basic metrics
        print(f"📊 Basic Metrics:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1-Score: {results['f1_score']:.4f}")
        print(f"  Jaccard Similarity: {results['jaccard_similarity']:.4f}")
        
        # IoU scores
        print(f"\n🎯 IoU Scores:")
        for class_name, iou in results['iou_scores'].items():
            print(f"  {class_name}: {iou:.4f}")
        
        # Advanced metrics (if available)
        if 'roc_auc' in results:
            print(f"\n📈 ROC AUC: {results['roc_auc']:.4f}")
        
        if 'map_scores' in results:
            print(f"\n🎯 mAP Scores:")
            for class_name, ap in results['map_scores'].items():
                print(f"  {class_name}: {ap:.4f}")
        
        # Confusion matrix
        print(f"\n📋 Confusion Matrix:")
        cm = np.array(results['confusion_matrix'])
        print(cm)
        
        # Per-class metrics
        if 'per_class_metrics' in results:
            print(f"\n📊 Per-Class Metrics:")
            print("  Class | Precision | Recall | F1-Score | Support")
            print("  ------|-----------|--------|----------|--------")
            for i, (prec, rec, f1, sup) in enumerate(zip(
                results['per_class_metrics']['precision'],
                results['per_class_metrics']['recall'],
                results['per_class_metrics']['f1_score'],
                results['per_class_metrics']['support']
            )):
                print(f"  {i:5d} | {prec:8.4f} | {rec:6.4f} | {f1:8.4f} | {int(sup):7d}")
    
    # Generate and save reports
    print("\n📋 Generating evaluation reports...")
    manager.generate_evaluation_reports(evaluation_results, output_dir='evaluation_demo')
    
    # Save results
    print("\n💾 Saving evaluation results...")
    manager.save_evaluation_results(evaluation_results, output_dir='evaluation_demo')
    
    # Demonstrate individual evaluator usage
    print("\n🔧 Individual ModelEvaluator Usage:")
    print("-" * 40)
    
    evaluator = ModelEvaluator(device=device)
    
    # Create sample predictions
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    y_prob = np.random.rand(10, 2)  # Random probabilities
    
    # Evaluate
    individual_results = evaluator.evaluate_classification_model(
        y_true, y_pred, y_prob, class_names=['Class A', 'Class B']
    )
    
    print("Sample individual evaluation:")
    print(f"  Accuracy: {individual_results['accuracy']:.4f}")
    print(f"  F1-Score: {individual_results['f1_score']:.4f}")
    print(f"  Mean IoU: {individual_results['iou_scores']['mean_iou']:.4f}")
    print(f"  Jaccard: {individual_results['jaccard_similarity']:.4f}")
    
    # Generate report
    report = evaluator.generate_evaluation_report(individual_results, "Sample Model")
    print(f"\n📄 Sample Report Preview:")
    print(report[:500] + "..." if len(report) > 500 else report)
    
    print("\n✅ Comprehensive evaluation metrics demonstration completed!")
    print("\n📁 Check 'evaluation_demo' directory for saved results and reports")

if __name__ == "__main__":
    import torch
    demonstrate_evaluation_metrics()
