#!/usr/bin/env python3
"""
Evaluate object detection on a COCO-style dataset using the current detector.

Inputs:
- --dataset: root folder containing train/valid/test with _annotations.coco.json
- --split: dataset split (train|valid|test)
- --images|--ann: optional explicit paths to images dir and annotation json

Outputs:
- Prints mAP (0.50:0.95), AP50, AP75, per-category AP
- Saves results to coco_eval_{split}_{timestamp}.json

This uses pycocotools; ensure requirements include pycocotools.
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ml_models import UnifiedModel


def load_categories(ann_file: str) -> Dict[int, str]:
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {int(cat['id']): str(cat['name']) for cat in data.get('categories', [])}


def build_name_to_id(categories: Dict[int, str]) -> Dict[str, int]:
    mapping = {}
    for cid, name in categories.items():
        mapping[name.lower()] = cid
    return mapping


def xyxy_to_xywh(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def main():
    parser = argparse.ArgumentParser(description='Evaluate detector on COCO dataset')
    parser.add_argument('--dataset', type=str, default='image recog.v1i.coco-mmdetection')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--images', type=str, default=None)
    parser.add_argument('--ann', type=str, default=None)
    parser.add_argument('--limit', type=int, default=0, help='Optional: limit number of images (0=all)')
    parser.add_argument('--score-thresh', type=float, default=None, help='Override score threshold')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.images and args.ann:
        img_dir = os.path.abspath(os.path.expanduser(args.images))
        ann_file = os.path.abspath(os.path.expanduser(args.ann))
    else:
        dataset_root = os.path.join(base_dir, args.dataset)
        img_dir = os.path.join(dataset_root, args.split)
        ann_file = os.path.join(dataset_root, args.split, '_annotations.coco.json')

    if not os.path.isdir(img_dir) or not os.path.isfile(ann_file):
        raise FileNotFoundError(f"Invalid dataset paths. images={img_dir}, ann={ann_file}")

    # Load COCO and categories
    coco = COCO(ann_file)
    categories = load_categories(ann_file)
    name_to_id = build_name_to_id(categories)

    # Build image id to file path
    img_id_to_path: Dict[int, str] = {}
    for img in coco.loadImgs(coco.getImgIds()):
        img_id_to_path[img['id']] = os.path.join(img_dir, img['file_name'])

    # Initialize detector
    detector = UnifiedModel()
    if args.score_thresh is not None:
        try:
            detector.score_threshold = float(args.score_thresh)
        except Exception:
            pass

    # Run inference and collect COCO-format detections
    results: List[Dict[str, Any]] = []
    img_ids = list(img_id_to_path.keys())
    if args.limit and args.limit > 0:
        img_ids = img_ids[:args.limit]

    for idx, img_id in enumerate(img_ids, 1):
        image_path = img_id_to_path[img_id]
        dets = detector.detect_objects(image_path)
        if isinstance(dets, dict) and 'error' in dets:
            continue
        for d in dets:
            cls = str(d.get('class', '')).lower()
            score = float(d.get('confidence', 0.0))
            box_xyxy = d.get('box', None)
            if not box_xyxy or len(box_xyxy) != 4:
                continue
            # Map class name to category_id; fallback to best guess if numeric
            category_id = name_to_id.get(cls)
            if category_id is None:
                try:
                    # If detector label matches COCO index in UnifiedModel.classes
                    # try direct numeric interpretation
                    as_int = int(d.get('label', -1))  # not present usually
                    if as_int in categories:
                        category_id = as_int
                except Exception:
                    category_id = None
            if category_id is None:
                # Skip classes not present in this dataset
                continue
            results.append({
                'image_id': int(img_id),
                'category_id': int(category_id),
                'bbox': xyxy_to_xywh(box_xyxy),
                'score': score,
            })
        if idx % 25 == 0:
            print(f"Processed {idx}/{len(img_ids)} images")

    if not results:
        print('No detections collected; aborting evaluation.')
        return

    # Evaluate using COCOeval
    coco_dt = coco.loadRes(results)
    coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract key metrics
    stats = coco_eval.stats  # 12-element vector
    summary = {
        'map': float(stats[0]),           # AP @[.5:.95]
        'ap50': float(stats[1]),          # AP @0.50
        'ap75': float(stats[2]),          # AP @0.75
        'ap_small': float(stats[3]),
        'ap_medium': float(stats[4]),
        'ap_large': float(stats[5]),
        'ar_1': float(stats[6]),
        'ar_10': float(stats[7]),
        'ar_100': float(stats[8]),
        'ar_small': float(stats[9]),
        'ar_medium': float(stats[10]),
        'ar_large': float(stats[11]),
    }

    # Optionally compute per-category AP
    per_category_ap: Dict[str, float] = {}
    try:
        precisions = coco_eval.eval['precision']  # [TxRxKxAxM]
        cat_ids = coco.getCatIds()
        for k, cat_id in enumerate(cat_ids):
            # average over IoU thresholds and recall, area all, max detections last index
            p = precisions[:, :, k, 0, -1]
            valid = p[p > -1]
            per_category_ap[categories[cat_id]] = float(np.mean(valid)) if valid.size else float('nan')
    except Exception:
        pass

    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = f'coco_eval_{args.split}_{timestamp}.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': os.path.abspath(img_dir),
            'annotations': os.path.abspath(ann_file),
            'split': args.split,
            'metrics': summary,
            'per_category_ap': per_category_ap,
            'num_images': len(img_ids),
            'num_detections': len(results),
        }, f, indent=2)
    print(f"Saved evaluation report to {out_file}")


if __name__ == '__main__':
    main()

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
