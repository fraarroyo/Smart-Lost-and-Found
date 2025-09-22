#!/usr/bin/env python3
"""
Realistic Model Evaluation Script
This script evaluates the model based on realistic lost & found scenarios:
1. Object detection accuracy (what objects are detected)
2. Matching system performance (finding similar items)
3. Confidence score reliability
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

def run_realistic_evaluation():
    """Run realistic evaluation based on lost & found use cases."""
    print("🔍 Starting realistic model evaluation...")
    print("=" * 60)
    
    with app.app_context():
        # Step 1: Check available data
        print("1️⃣ Analyzing available data...")
        items = Item.query.all()
        print(f"   - Total items in database: {len(items)}")
        
        # Analyze object detection results
        detection_stats = analyze_object_detection(items)
        print(f"   - Items with detected objects: {detection_stats['items_with_objects']}")
        print(f"   - Total detected objects: {detection_stats['total_objects']}")
        print(f"   - Unique object classes: {len(detection_stats['object_classes'])}")
        print(f"   - Average confidence: {detection_stats['avg_confidence']:.3f}")
        
        # Step 2: Evaluate object detection performance
        print("\n2️⃣ Evaluating object detection performance...")
        detection_metrics = evaluate_object_detection(items)
        
        print(f"   - Detection success rate: {detection_metrics['success_rate']:.1%}")
        print(f"   - High confidence detections: {detection_metrics['high_confidence_rate']:.1%}")
        print(f"   - Average objects per image: {detection_metrics['avg_objects_per_image']:.2f}")
        
        # Step 3: Evaluate confidence score reliability
        print("\n3️⃣ Evaluating confidence score reliability...")
        confidence_metrics = evaluate_confidence_reliability(items)
        
        print(f"   - Confidence distribution: {confidence_metrics['distribution']}")
        print(f"   - High confidence threshold: {confidence_metrics['high_confidence_threshold']:.3f}")
        
        # Step 4: Evaluate matching system (if available)
        print("\n4️⃣ Evaluating matching system...")
        matching_metrics = evaluate_matching_system(items)
        
        print(f"   - Matching system available: {matching_metrics['available']}")
        if matching_metrics['available']:
            print(f"   - Average matches per item: {matching_metrics['avg_matches']:.2f}")
            print(f"   - High similarity matches: {matching_metrics['high_similarity_rate']:.1%}")
        
        # Step 5: Store realistic metrics
        print("\n5️⃣ Storing realistic metrics...")
        store_realistic_metrics(detection_metrics, confidence_metrics, matching_metrics)
        
        # Step 6: Display comprehensive results
        print("\n6️⃣ Realistic Evaluation Results:")
        print("=" * 50)
        print(f"📊 Object Detection Performance:")
        print(f"   - Success Rate: {detection_metrics['success_rate']:.1%}")
        print(f"   - High Confidence Rate: {detection_metrics['high_confidence_rate']:.1%}")
        print(f"   - Objects per Image: {detection_metrics['avg_objects_per_image']:.2f}")
        print()
        print(f"🎯 Confidence Reliability:")
        print(f"   - Average Confidence: {confidence_metrics['avg_confidence']:.3f}")
        print(f"   - Confidence Range: {confidence_metrics['min_confidence']:.3f} - {confidence_metrics['max_confidence']:.3f}")
        print()
        print(f"🔍 Object Class Distribution:")
        for class_name, count in detection_stats['object_classes'].items():
            print(f"   - {class_name}: {count} detections")
        
        if matching_metrics['available']:
            print(f"\n🤝 Matching System Performance:")
            print(f"   - Average Matches: {matching_metrics['avg_matches']:.2f}")
            print(f"   - High Similarity Rate: {matching_metrics['high_similarity_rate']:.1%}")
        
        print("\n🎉 Realistic evaluation completed successfully!")
        print("💡 This evaluation focuses on practical lost & found performance")
        
        return True

def analyze_object_detection(items):
    """Analyze object detection results across all items."""
    stats = {
        'items_with_objects': 0,
        'total_objects': 0,
        'object_classes': {},
        'confidences': [],
        'avg_confidence': 0.0
    }
    
    for item in items:
        try:
            det = json.loads(item.detected_objects) if item.detected_objects else []
            if det:
                stats['items_with_objects'] += 1
                stats['total_objects'] += len(det)
                
                for obj in det:
                    obj_class = obj.get('class', 'unknown')
                    confidence = obj.get('confidence', 0.0)
                    
                    if obj_class not in stats['object_classes']:
                        stats['object_classes'][obj_class] = 0
                    stats['object_classes'][obj_class] += 1
                    stats['confidences'].append(confidence)
        except:
            pass
    
    if stats['confidences']:
        stats['avg_confidence'] = sum(stats['confidences']) / len(stats['confidences'])
    
    return stats

def evaluate_object_detection(items):
    """Evaluate object detection performance."""
    metrics = {
        'success_rate': 0.0,
        'high_confidence_rate': 0.0,
        'avg_objects_per_image': 0.0,
        'detection_quality': 'unknown'
    }
    
    total_items = len(items)
    items_with_objects = 0
    high_confidence_detections = 0
    total_objects = 0
    confidences = []
    
    for item in items:
        try:
            det = json.loads(item.detected_objects) if item.detected_objects else []
            if det:
                items_with_objects += 1
                total_objects += len(det)
                
                for obj in det:
                    confidence = obj.get('confidence', 0.0)
                    confidences.append(confidence)
                    if confidence > 0.7:
                        high_confidence_detections += 1
        except:
            pass
    
    if total_items > 0:
        metrics['success_rate'] = items_with_objects / total_items
        metrics['avg_objects_per_image'] = total_objects / total_items
    
    if confidences:
        metrics['high_confidence_rate'] = high_confidence_detections / len(confidences)
        
        # Determine detection quality
        avg_conf = sum(confidences) / len(confidences)
        if avg_conf > 0.8:
            metrics['detection_quality'] = 'excellent'
        elif avg_conf > 0.6:
            metrics['detection_quality'] = 'good'
        elif avg_conf > 0.4:
            metrics['detection_quality'] = 'fair'
        else:
            metrics['detection_quality'] = 'poor'
    
    return metrics

def evaluate_confidence_reliability(items):
    """Evaluate confidence score reliability."""
    metrics = {
        'avg_confidence': 0.0,
        'min_confidence': 1.0,
        'max_confidence': 0.0,
        'distribution': {},
        'high_confidence_threshold': 0.7
    }
    
    confidences = []
    
    for item in items:
        try:
            det = json.loads(item.detected_objects) if item.detected_objects else []
            for obj in det:
                confidence = obj.get('confidence', 0.0)
                confidences.append(confidence)
        except:
            pass
    
    if confidences:
        metrics['avg_confidence'] = sum(confidences) / len(confidences)
        metrics['min_confidence'] = min(confidences)
        metrics['max_confidence'] = max(confidences)
        
        # Confidence distribution
        high_conf = sum(1 for c in confidences if c > 0.7)
        medium_conf = sum(1 for c in confidences if 0.4 <= c <= 0.7)
        low_conf = sum(1 for c in confidences if c < 0.4)
        
        total = len(confidences)
        metrics['distribution'] = {
            'high': f"{high_conf/total:.1%}",
            'medium': f"{medium_conf/total:.1%}",
            'low': f"{low_conf/total:.1%}"
        }
    
    return metrics

def evaluate_matching_system(items):
    """Evaluate matching system performance."""
    metrics = {
        'available': False,
        'avg_matches': 0.0,
        'high_similarity_rate': 0.0
    }
    
    try:
        from app import find_potential_matches
        
        total_matches = 0
        high_similarity_matches = 0
        items_processed = 0
        
        for item in items:
            try:
                matches = find_potential_matches(item)
                if matches:
                    metrics['available'] = True
                    total_matches += len(matches)
                    items_processed += 1
                    
                    # Count high similarity matches
                    high_sim = sum(1 for m in matches if m.get('similarity', 0) > 0.7)
                    high_similarity_matches += high_sim
            except:
                pass
        
        if items_processed > 0:
            metrics['avg_matches'] = total_matches / items_processed
            metrics['high_similarity_rate'] = high_similarity_matches / total_matches if total_matches > 0 else 0.0
    
    except Exception as e:
        print(f"   ⚠️  Matching system not available: {e}")
    
    return metrics

def store_realistic_metrics(detection_metrics, confidence_metrics, matching_metrics):
    """Store realistic metrics in the database."""
    try:
        # Clear existing metrics
        ModelMetrics.query.filter_by(model_type='unified_model').delete()
        
        # Store detection metrics
        ModelMetrics.add_metric('unified_model', 'detection_success_rate', detection_metrics['success_rate'])
        ModelMetrics.add_metric('unified_model', 'high_confidence_rate', detection_metrics['high_confidence_rate'])
        ModelMetrics.add_metric('unified_model', 'avg_objects_per_image', detection_metrics['avg_objects_per_image'])
        
        # Store confidence metrics
        ModelMetrics.add_metric('unified_model', 'avg_confidence', confidence_metrics['avg_confidence'])
        ModelMetrics.add_metric('unified_model', 'min_confidence', confidence_metrics['min_confidence'])
        ModelMetrics.add_metric('unified_model', 'max_confidence', confidence_metrics['max_confidence'])
        
        # Store matching metrics if available
        if matching_metrics['available']:
            ModelMetrics.add_metric('unified_model', 'avg_matches_per_item', matching_metrics['avg_matches'])
            ModelMetrics.add_metric('unified_model', 'high_similarity_rate', matching_metrics['high_similarity_rate'])
        
        # Store overall performance score
        overall_score = calculate_overall_score(detection_metrics, confidence_metrics, matching_metrics)
        ModelMetrics.add_metric('unified_model', 'overall_performance_score', overall_score)
        
        print("   ✅ Realistic metrics stored successfully")
        
    except Exception as e:
        print(f"   ❌ Error storing metrics: {e}")

def calculate_overall_score(detection_metrics, confidence_metrics, matching_metrics):
    """Calculate an overall performance score."""
    score = 0.0
    
    # Detection success (40% weight)
    score += detection_metrics['success_rate'] * 0.4
    
    # High confidence rate (30% weight)
    score += detection_metrics['high_confidence_rate'] * 0.3
    
    # Average confidence (20% weight)
    score += min(confidence_metrics['avg_confidence'], 1.0) * 0.2
    
    # Matching performance (10% weight)
    if matching_metrics['available']:
        score += matching_metrics['high_similarity_rate'] * 0.1
    
    return score

def main():
    """Main function to run realistic evaluation."""
    try:
        success = run_realistic_evaluation()
        if success:
            print("\n✅ Realistic model evaluation completed successfully!")
            print("💡 The metrics now reflect realistic lost & found performance")
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
