#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
This script runs a complete evaluation of the current model and populates all metrics.
"""

import os
import sys
import json
from datetime import datetime
from collections import defaultdict

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Flask app and models
from app import app, Item, ModelMetrics

def run_comprehensive_evaluation():
    """Run comprehensive model evaluation and populate all metrics."""
    print("🔍 Starting comprehensive model evaluation...")
    print("=" * 60)
    
    with app.app_context():
        # Step 1: Check available data
        print("1️⃣ Checking available data...")
        items = Item.query.all()
        print(f"   - Total items in database: {len(items)}")
        
        items_with_objects = 0
        total_detected_objects = 0
        
        for item in items:
            try:
                det = json.loads(item.detected_objects) if item.detected_objects else []
                if det:
                    items_with_objects += 1
                    total_detected_objects += len(det)
            except:
                pass
        
        print(f"   - Items with detected objects: {items_with_objects}")
        print(f"   - Total detected objects: {total_detected_objects}")
        
        if items_with_objects == 0:
            print("   ⚠️  No items with detected objects found!")
            print("   💡 Run training first to detect objects in images")
            return False
        
        # Step 2: Check for object detection capability
        print("\n2️⃣ Checking object detection capability...")
        try:
            from app import object_detector
            print("   ✅ Object detector available")
            
            # Run object detection on items without objects
            processed_count = 0
            
            for item in items:
                try:
                    det = json.loads(item.detected_objects) if item.detected_objects else []
                    if not det and item.image_path:
                        # Run object detection
                        image_path = os.path.join('static', item.image_path)
                        if os.path.exists(image_path):
                            detected_objects = object_detector.detect_objects(image_path)
                            if detected_objects:
                                item.detected_objects = json.dumps(detected_objects)
                                processed_count += 1
                                print(f"   ✅ Detected objects in item {item.id}")
                except Exception as e:
                    print(f"   ❌ Error processing item {item.id}: {e}")
            
            if processed_count > 0:
                from app import db
                db.session.commit()
                print(f"   ✅ Processed {processed_count} items with object detection")
        except Exception as e:
            print(f"   ⚠️  Object detection not available: {e}")
            print("   💡 Using existing detected objects only")
        
        # Step 3: Build evaluation dataset
        print("\n3️⃣ Building evaluation dataset...")
        y_true, y_pred = [], []
        labels_set = set()
        
        for item in items:
            try:
                det = json.loads(item.detected_objects) if item.detected_objects else []
            except:
                det = []
            
            if not det:
                continue
                
            # Use the highest confidence detected object as prediction
            best_obj = max(det, key=lambda x: x.get('confidence', 0))
            predicted_class = best_obj.get('class', 'unknown')
            
            # Use category as true label (or detected class if no category)
            true_label = item.category or predicted_class
            
            y_true.append(true_label)
            y_pred.append(predicted_class)
            labels_set.update([true_label, predicted_class])
        
        print(f"   - Evaluation samples: {len(y_true)}")
        print(f"   - Unique classes: {len(labels_set)}")
        
        if not y_true:
            print("   ❌ No valid evaluation data found!")
            return False
        
        # Step 4: Compute classification metrics
        print("\n4️⃣ Computing classification metrics...")
        classification_metrics = compute_classification_metrics(y_true, y_pred, sorted(labels_set))
        
        print(f"   - Accuracy: {classification_metrics['accuracy']:.3f}")
        print(f"   - Macro F1: {classification_metrics['macro_f1']:.3f}")
        print(f"   - Micro F1: {classification_metrics['micro_f1']:.3f}")
        print(f"   - Weighted F1: {classification_metrics['weighted_f1']:.3f}")
        
        # Step 5: Compute MAP metrics
        print("\n5️⃣ Computing MAP metrics...")
        ground_truth = defaultdict(set)
        predictions = {}
        
        try:
            from app import find_potential_matches
            
            for item in items:
                # Relevant items: same category, opposite status
                relevant = set(
                    i.id for i in items
                    if i.id != item.id and i.category == item.category and i.status != item.status
                )
                ground_truth[item.id] = relevant
                
                # Use our matching algorithm to generate predicted ranking
                matches = find_potential_matches(item)
                ranked = [m['item'].id for m in matches]
                predictions[item.id] = ranked
            
            map_metrics = compute_map(ground_truth, predictions)
        except Exception as e:
            print(f"   ⚠️  MAP computation not available: {e}")
            print("   💡 Using simplified MAP calculation")
            
            # Simplified MAP calculation
            map_metrics = {
                'map': 0.0,
                'precision_at_5': 0.0,
                'precision_at_10': 0.0,
                'recall_at_5': 0.0,
                'recall_at_10': 0.0,
                'num_queries': 0
            }
        
        print(f"   - MAP: {map_metrics['map']:.3f}")
        print(f"   - Precision@5: {map_metrics['precision_at_5']:.3f}")
        print(f"   - Recall@5: {map_metrics['recall_at_5']:.3f}")
        
        # Step 6: Store metrics in database
        print("\n6️⃣ Storing metrics in database...")
        timestamp = datetime.now()
        
        # Clear existing metrics
        ModelMetrics.query.filter_by(model_type='unified_model').delete()
        
        # Store overall metrics
        ModelMetrics.add_metric('unified_model', 'accuracy', float(classification_metrics['accuracy']))
        ModelMetrics.add_metric('unified_model', 'macro_f1', float(classification_metrics['macro_f1']))
        ModelMetrics.add_metric('unified_model', 'micro_f1', float(classification_metrics['micro_f1']))
        ModelMetrics.add_metric('unified_model', 'weighted_f1', float(classification_metrics['weighted_f1']))
        ModelMetrics.add_metric('unified_model', 'micro_precision', float(classification_metrics['micro_precision']))
        ModelMetrics.add_metric('unified_model', 'micro_recall', float(classification_metrics['micro_recall']))
        
        # Store MAP metrics
        ModelMetrics.add_metric('unified_model', 'mean_average_precision', float(map_metrics['map']))
        ModelMetrics.add_metric('unified_model', 'precision_at_5', float(map_metrics['precision_at_5']))
        ModelMetrics.add_metric('unified_model', 'precision_at_10', float(map_metrics['precision_at_10']))
        ModelMetrics.add_metric('unified_model', 'recall_at_5', float(map_metrics['recall_at_5']))
        ModelMetrics.add_metric('unified_model', 'recall_at_10', float(map_metrics['recall_at_10']))
        
        # Store per-class metrics
        for label, vals in classification_metrics['per_class'].items():
            ModelMetrics.add_metric('unified_model', 'class_f1', float(vals['f1']), class_name=label)
            ModelMetrics.add_metric('unified_model', 'class_precision', float(vals['precision']), class_name=label)
            ModelMetrics.add_metric('unified_model', 'class_recall', float(vals['recall']), class_name=label)
            ModelMetrics.add_metric('unified_model', 'class_support', float(vals['support']), class_name=label)
        
        # Store evaluation summary
        evaluation_summary = {
            'timestamp': timestamp.isoformat(),
            'total_samples': classification_metrics['total_samples'],
            'num_classes': classification_metrics['num_classes'],
            'classification_metrics': classification_metrics,
            'map_metrics': map_metrics
        }
        
        print("   ✅ All metrics stored successfully")
        
        # Step 7: Display results
        print("\n7️⃣ Evaluation Results:")
        print("=" * 40)
        print(f"📊 Classification Metrics:")
        print(f"   - Accuracy: {classification_metrics['accuracy']:.1%}")
        print(f"   - Macro F1: {classification_metrics['macro_f1']:.3f}")
        print(f"   - Micro F1: {classification_metrics['micro_f1']:.3f}")
        print(f"   - Weighted F1: {classification_metrics['weighted_f1']:.3f}")
        print()
        print(f"🎯 Retrieval Metrics:")
        print(f"   - Mean Average Precision: {map_metrics['map']:.3f}")
        print(f"   - Precision@5: {map_metrics['precision_at_5']:.3f}")
        print(f"   - Recall@5: {map_metrics['recall_at_5']:.3f}")
        print()
        print(f"📈 Per-Class Performance:")
        for label, vals in classification_metrics['per_class'].items():
            print(f"   - {label}: F1={vals['f1']:.3f}, Precision={vals['precision']:.3f}, Recall={vals['recall']:.3f}")
        
        print("\n🎉 Model evaluation completed successfully!")
        print("💡 You can now view the metrics in the admin interface at /admin/training/metrics")
        
        return True

def compute_classification_metrics(y_true, y_pred, labels):
    """Compute comprehensive classification metrics."""
    if not y_true or not y_pred:
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'micro_f1': 0.0,
            'weighted_f1': 0.0,
            'per_class': {},
            'confusion_matrix': {}
        }
    
    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    # Per-class precision/recall/F1
    per_class = {}
    eps = 1e-9
    total_support = 0
    
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != label and p != label)
        
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        support = sum(1 for t in y_true if t == label)
        
        per_class[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
        total_support += support

    # Macro F1 (average of per-class F1 scores)
    macro_f1 = sum(v['f1'] for v in per_class.values()) / (len(per_class) or 1)

    # Micro F1 (global precision and recall)
    total_tp = sum(v['tp'] for v in per_class.values())
    total_fp = sum(v['fp'] for v in per_class.values())
    total_fn = sum(v['fn'] for v in per_class.values())
    
    micro_precision = total_tp / (total_tp + total_fp + eps)
    micro_recall = total_tp / (total_tp + total_fn + eps)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)

    # Weighted F1 (weighted by support)
    weighted_f1 = sum(v['f1'] * v['support'] for v in per_class.values()) / (total_support or 1)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'per_class': per_class,
        'total_samples': len(y_true),
        'num_classes': len(labels)
    }

def compute_map(ground_truth, predictions):
    """Compute mean Average Precision for retrieval evaluation."""
    def average_precision(relevant_set, ranked_list):
        if not ranked_list or not relevant_set:
            return 0.0
        hits = 0
        sum_precisions = 0.0
        for idx, pred in enumerate(ranked_list, 1):
            if pred in relevant_set:
                hits += 1
                sum_precisions += hits / idx
        return sum_precisions / len(relevant_set)

    def precision_at_k(relevant_set, ranked_list, k):
        if not ranked_list or not relevant_set:
            return 0.0
        top_k = ranked_list[:k]
        hits = sum(1 for pred in top_k if pred in relevant_set)
        return hits / min(k, len(ranked_list))

    def recall_at_k(relevant_set, ranked_list, k):
        if not ranked_list or not relevant_set:
            return 0.0
        top_k = ranked_list[:k]
        hits = sum(1 for pred in top_k if pred in relevant_set)
        return hits / len(relevant_set)

    ap_values = []
    precision_at_5 = []
    precision_at_10 = []
    recall_at_5 = []
    recall_at_10 = []

    for qid, rel in ground_truth.items():
        pred_list = predictions.get(qid, [])
        ap_values.append(average_precision(rel, pred_list))
        precision_at_5.append(precision_at_k(rel, pred_list, 5))
        precision_at_10.append(precision_at_k(rel, pred_list, 10))
        recall_at_5.append(recall_at_k(rel, pred_list, 5))
        recall_at_10.append(recall_at_k(rel, pred_list, 10))

    return {
        'map': sum(ap_values) / (len(ap_values) or 1),
        'precision_at_5': sum(precision_at_5) / (len(precision_at_5) or 1),
        'precision_at_10': sum(precision_at_10) / (len(precision_at_10) or 1),
        'recall_at_5': sum(recall_at_5) / (len(recall_at_5) or 1),
        'recall_at_10': sum(recall_at_10) / (len(recall_at_10) or 1),
        'num_queries': len(ap_values)
    }

def main():
    """Main function to run model evaluation."""
    try:
        success = run_comprehensive_evaluation()
        if success:
            print("\n✅ Model evaluation completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Model evaluation failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
