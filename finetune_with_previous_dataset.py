#!/usr/bin/env python3
"""
Fine-tune the Faster R-CNN detector using the previous COCO-style dataset,
starting from the existing checkpoint in outputs/best_model.pth.

Usage (examples):
  python finetune_with_previous_dataset.py \
    --dataset "image recog.v1i.coco-mmdetection" \
    --split train \
    --epochs 5 \
    --output outputs/finetuned_model.pth

Notes:
  - Will fall back to COCO-pretrained weights if no outputs/best_model.pth is found
  - Saves a checkpoint containing: { 'model', 'num_classes', 'label_names' }
"""

import os
import json
import time
import argparse
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


def collate_fn(batch):
    return tuple(zip(*batch))


class CocoDetectionBBox(torchvision.datasets.CocoDetection):
    """COCO dataset wrapper that converts polygon/segmentation to boxes only."""

    def __init__(self, img_folder: str, ann_file: str, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # Convert COCO annotations to detection target
        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in target:
            bbox = ann.get('bbox')  # [x, y, w, h]
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann['category_id']))
            areas.append(float(ann.get('area', w * h)))
            iscrowd.append(int(ann.get('iscrowd', 0)))

        target_dict = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        if self._transforms is not None:
            img = self._transforms(img)
        else:
            img = F.to_tensor(img)

        return img, target_dict


def load_label_names_from_coco(ann_file: str) -> Dict[int, str]:
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    categories = data.get('categories', [])
    # COCO often has non-contiguous IDs; keep exact mapping
    return {int(cat['id']): str(cat['name']) for cat in categories}


def build_model(num_classes: int, score_thresh: float = 0.5):
    # Start with COCO backbone; threshold applied via post-filter during inference, not training
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def try_load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> bool:
    if not ckpt_path or not os.path.exists(ckpt_path):
        return False
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {ckpt_path}. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        return True
    except Exception as e:
        print(f"Warning: failed to load checkpoint from {ckpt_path}: {e}")
        return False


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    lr_scheduler = None
    if epoch == 0:
        # Warmup like in torchvision references
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0/1000, total_iters=min(1000, len(data_loader)))

    running_loss = 0.0
    last_print = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss += losses.item()
        if (i + 1) % print_freq == 0 or (time.time() - last_print) > 20:
            avg = running_loss / (i + 1)
            print(f"Epoch {epoch+1} [{i+1}/{len(data_loader)}] loss: {avg:.4f}")
            last_print = time.time()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune detector on previous dataset")
    parser.add_argument('--dataset', type=str, default="image recog.v1i.coco-mmdetection", help='Root folder containing train/test/valid with COCO jsons')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'valid', 'test'], help='Dataset split to use for training')
    parser.add_argument('--coco-images', type=str, default=None, help='Optional: path to COCO images directory (e.g., train2017)')
    parser.add_argument('--coco-ann', type=str, default=None, help='Optional: path to COCO annotation json (e.g., annotations/instances_train2017.json)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--ckpt', type=str, default='outputs/best_model.pth', help='Starting checkpoint path')
    parser.add_argument('--output', type=str, default='outputs/finetuned_model.pth')
    args = parser.parse_args()

    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if args.coco_images and args.coco_ann:
        img_dir = os.path.abspath(os.path.expanduser(args.coco_images))
        ann_file = os.path.abspath(os.path.expanduser(args.coco_ann))
        if not (os.path.isdir(img_dir) and os.path.isfile(ann_file)):
            raise FileNotFoundError(f"Invalid COCO paths. images={img_dir}, ann={ann_file}")
    else:
        dataset_root = os.path.join(base_dir, args.dataset)
        img_dir = os.path.join(dataset_root, args.split)
        ann_file = os.path.join(dataset_root, args.split, '_annotations.coco.json')
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Labels and num_classes
    label_names = load_label_names_from_coco(ann_file)
    # num_classes = max category id + 1 if contiguous; otherwise predictor will handle labels by id mapping
    # For torchvision detection, background is implicit at index 0 if labels start at 1.
    # We'll set num_classes to max ID + 1 to be safe.
    num_classes = (max(label_names.keys()) + 1) if label_names else 2
    print(f"Detected {len(label_names)} categories; num_classes set to {num_classes}")

    # Dataset & loader
    dataset = CocoDetectionBBox(img_dir, ann_file, transforms=None)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    # Model
    model = build_model(num_classes=num_classes)
    model.to(device)
    _ = try_load_checkpoint(model, os.path.join(base_dir, args.ckpt), device)

    # Optimizer & train
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, loader, device, epoch)

    # Save checkpoint
    os.makedirs(os.path.join(base_dir, 'outputs'), exist_ok=True)
    ckpt = {
        'model': model.state_dict(),
        'num_classes': num_classes,
        'label_names': label_names,
    }
    out_path = os.path.join(base_dir, args.output)
    torch.save(ckpt, out_path)
    print(f"Saved fine-tuned model to {out_path}")


if __name__ == '__main__':
    main()


