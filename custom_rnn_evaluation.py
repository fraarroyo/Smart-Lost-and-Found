#!/usr/bin/env python3
"""
Custom RNN Model Evaluation System for BARYONYX Lost & Found System
Comprehensive evaluation with detailed metrics, visualizations, and performance analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

# Import custom models
from custom_rnn_model import CustomRNNManager

class CustomRNNEvaluator:
    """Comprehensive evaluator for custom RNN models"""
    
    def __init__(self, model_dir: str = 'models/custom_rnn/', device: str = 'cpu'):
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.logger = self._setup_logging()
        
        # Initialize manager
        self.manager = CustomRNNManager(device=device, model_dir=model_dir)
        self.manager.load_models()
        
        # Evaluation results storage
        self.evaluation_results = {}
        
        # Create output directories
        os.makedirs(os.path.join(model_dir, 'evaluation'), exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'evaluation', 'plots'), exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'evaluation', 'reports'), exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for evaluation"""
        logger = logging.getLogger('custom_rnn_evaluator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def evaluate_item_matching_model(self, test_data: List[Dict]) -> Dict:
        """Evaluate item matching model with comprehensive metrics"""
        self.logger.info("Evaluating item matching model...")
        
        # Prepare test data
        X_test = []
        y_test = []
        predictions = []
        probabilities = []
        
        for item in test_data:
            # Combine lost and found features
            combined_features = item['lost_features'] + item['found_features']
            X_test.append(combined_features)
            y_test.append(1 if item['is_match'] else 0)
            
            # Get prediction
            result = self.manager.predict_item_match(combined_features)  # Use all 40 features
            predictions.append(result['prediction'])
            probabilities.append(result['probabilities'])
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # ROC AUC (if binary classification)
        if len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0])
        else:
            roc_auc = None
        
        # Classification report
        class_report = classification_report(y_test, predictions, output_dict=True)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        self.evaluation_results['item_matching'] = results
        
        # Generate visualizations
        self._plot_confusion_matrix(cm, 'Item Matching', 'item_matching_cm')
        self._plot_roc_curve(y_test, probabilities, 'Item Matching', 'item_matching_roc')
        
        return results
    
    def evaluate_behavior_model(self, test_data: List[Dict]) -> Dict:
        """Evaluate user behavior prediction model"""
        self.logger.info("Evaluating behavior prediction model...")
        
        # Prepare test data
        X_test = []
        y_test = []
        predictions = []
        probabilities = []
        
        # Group by user and create sequences
        user_sequences = defaultdict(list)
        for item in test_data:
            user_sequences[item['user_id']].append(item)
        
        for user_id, actions in user_sequences.items():
            if len(actions) >= 10:  # Minimum sequence length
                sequence = actions[:10]  # Take first 10 actions
                features = [action['features'] for action in sequence]
                
                X_test.append(features)
                
                # Get next action as label
                action_encoding = {'search': 0, 'view': 1, 'upload': 2, 'browse': 3, 'logout': 4}
                next_action = sequence[-1]['next_action']
                y_test.append(action_encoding.get(next_action, 0))
                
                # Get prediction
                result = self.manager.predict_user_behavior(features)
                predictions.append(result['prediction'])
                probabilities.append(result['probabilities'])
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Classification report
        class_report = classification_report(y_test, predictions, output_dict=True)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        self.evaluation_results['behavior'] = results
        
        # Generate visualizations
        self._plot_confusion_matrix(cm, 'Behavior Prediction', 'behavior_cm')
        self._plot_class_distribution(y_test, predictions, 'Behavior Prediction', 'behavior_dist')
        
        return results
    
    def evaluate_text_model(self, test_data: List[Dict]) -> Dict:
        """Evaluate text description classification model"""
        self.logger.info("Evaluating text classification model...")
        
        # Prepare test data
        X_test = []
        y_test = []
        predictions = []
        probabilities = []
        
        for item in test_data:
            # Convert text to sequence
            sequence = self.manager.text_to_sequence(item['text'])
            X_test.append(sequence)
            y_test.append(item['label'])
            
            # Get prediction
            result = self.manager.classify_description(item['text'])
            predictions.append(result['prediction'])
            probabilities.append(result['probabilities'])
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Classification report
        class_report = classification_report(y_test, predictions, output_dict=True)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        self.evaluation_results['text'] = results
        
        # Generate visualizations
        self._plot_confusion_matrix(cm, 'Text Classification', 'text_cm')
        self._plot_class_distribution(y_test, predictions, 'Text Classification', 'text_dist')
        
        return results
    
    def _plot_confusion_matrix(self, cm: np.ndarray, title: str, filename: str):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(cm.shape[1]), 
                   yticklabels=range(cm.shape[0]))
        plt.title(f'{title} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'evaluation', 'plots', f'{filename}.png'))
        plt.show()
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, title: str, filename: str):
        """Plot ROC curve"""
        if len(np.unique(y_true)) == 2 and y_prob.shape[1] > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{title} - ROC Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, 'evaluation', 'plots', f'{filename}.png'))
            plt.show()
    
    def _plot_class_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str):
        """Plot class distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Actual distribution
        unique, counts = np.unique(y_true, return_counts=True)
        ax1.bar(unique, counts, alpha=0.7, label='Actual')
        ax1.set_title(f'{title} - Actual Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.legend()
        
        # Predicted distribution
        unique, counts = np.unique(y_pred, return_counts=True)
        ax2.bar(unique, counts, alpha=0.7, label='Predicted', color='orange')
        ax2.set_title(f'{title} - Predicted Class Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'evaluation', 'plots', f'{filename}.png'))
        plt.show()
    
    def plot_model_comparison(self):
        """Plot comparison of all models"""
        if not self.evaluation_results:
            self.logger.warning("No evaluation results available. Run evaluations first.")
            return
        
        # Extract metrics
        models = list(self.evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [self.evaluation_results[model][metric] for model in models]
            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric.title()} Comparison')
            axes[i].set_ylabel(metric.title())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'evaluation', 'plots', 'model_comparison.png'))
        plt.show()
    
    def generate_detailed_report(self) -> str:
        """Generate detailed evaluation report"""
        if not self.evaluation_results:
            return "No evaluation results available."
        
        report = []
        report.append("# Custom RNN Model Evaluation Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        # Model information
        model_info = self.manager.get_model_info()
        report.append("\n## Model Information")
        report.append(f"Device: {model_info['device']}")
        report.append(f"Vocabulary Size: {model_info['vocab_size']}")
        report.append(f"Vocabulary Loaded: {model_info['vocab_loaded']}")
        
        # Individual model results
        for model_name, results in self.evaluation_results.items():
            report.append(f"\n## {model_name.replace('_', ' ').title()} Model")
            report.append("-" * 40)
            report.append(f"Accuracy: {results['accuracy']:.4f}")
            report.append(f"Precision: {results['precision']:.4f}")
            report.append(f"Recall: {results['recall']:.4f}")
            report.append(f"F1-Score: {results['f1_score']:.4f}")
            
            if results.get('roc_auc'):
                report.append(f"ROC AUC: {results['roc_auc']:.4f}")
            
            # Confusion matrix
            report.append("\n### Confusion Matrix")
            cm = np.array(results['confusion_matrix'])
            report.append("```")
            report.append(str(cm))
            report.append("```")
            
            # Classification report
            report.append("\n### Detailed Classification Report")
            class_report = results['classification_report']
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict):
                    report.append(f"\n**Class {class_name}:**")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            report.append(f"  {metric}: {value:.4f}")
        
        # Overall summary
        report.append("\n## Overall Summary")
        report.append("-" * 20)
        
        avg_accuracy = np.mean([results['accuracy'] for results in self.evaluation_results.values()])
        avg_f1 = np.mean([results['f1_score'] for results in self.evaluation_results.values()])
        
        report.append(f"Average Accuracy: {avg_accuracy:.4f}")
        report.append(f"Average F1-Score: {avg_f1:.4f}")
        
        # Best performing model
        best_model = max(self.evaluation_results.items(), key=lambda x: x[1]['accuracy'])
        report.append(f"Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
        
        return "\n".join(report)
    
    def save_evaluation_results(self):
        """Save evaluation results to files"""
        # Save raw results
        with open(os.path.join(self.model_dir, 'evaluation', 'evaluation_results.json'), 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        # Save detailed report
        report = self.generate_detailed_report()
        with open(os.path.join(self.model_dir, 'evaluation', 'reports', 'detailed_report.md'), 'w') as f:
            f.write(report)
        
        # Save summary CSV
        summary_data = []
        for model_name, results in self.evaluation_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Score': results['f1_score'],
                'ROC_AUC': results.get('roc_auc', 'N/A')
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(self.model_dir, 'evaluation', 'summary.csv'), index=False)
        
        self.logger.info(f"Evaluation results saved to {self.model_dir}/evaluation/")
    
    def run_comprehensive_evaluation(self, test_data_dir: str = 'enhanced_datasets/'):
        """Run comprehensive evaluation on all models"""
        self.logger.info("Starting comprehensive evaluation...")
        
        # Load test data
        test_data = {}
        for filename in ['enhanced_item_matching.json', 'enhanced_user_behavior.json', 'enhanced_text_descriptions.json']:
            filepath = os.path.join(test_data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data_name = filename.replace('enhanced_', '').replace('.json', '')
                    test_data[data_name] = json.load(f)
        
        # Evaluate each model
        if 'item_matching' in test_data:
            self.evaluate_item_matching_model(test_data['item_matching'])
        
        if 'user_behavior' in test_data:
            self.evaluate_behavior_model(test_data['user_behavior'])
        
        if 'text_descriptions' in test_data:
            self.evaluate_text_model(test_data['text_descriptions'])
        
        # Generate comparison plots
        self.plot_model_comparison()
        
        # Save results
        self.save_evaluation_results()
        
        # Print summary
        print("\n📊 Evaluation Summary:")
        print("=" * 30)
        for model_name, results in self.evaluation_results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  F1-Score: {results['f1_score']:.4f}")
        
        self.logger.info("Comprehensive evaluation completed!")

def main():
    """Main evaluation function"""
    print("🔍 Custom RNN Model Evaluation System")
    print("=" * 50)
    
    # Initialize evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    evaluator = CustomRNNEvaluator(device=device)
    
    # Run comprehensive evaluation
    evaluator.run_comprehensive_evaluation()
    
    print("\n✅ Evaluation completed successfully!")

if __name__ == "__main__":
    main()
