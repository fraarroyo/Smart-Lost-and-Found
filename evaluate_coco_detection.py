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

    # Initialize detector with best_model.pth
    detector = UnifiedModel()
    if args.score_thresh is not None:
        try:
            detector.score_threshold = float(args.score_thresh)
        except Exception:
            pass
    
    # Ensure we're using the best_model.pth checkpoint
    print(f"Using detector with checkpoint: {detector.checkpoint_path if hasattr(detector, 'checkpoint_path') else 'default'}")
    print(f"Label names loaded: {len(detector.label_names)} classes")
    if detector.label_names:
        print(f"Sample classes: {list(detector.label_names.values())[:5]}")

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


