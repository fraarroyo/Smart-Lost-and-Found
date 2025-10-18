#!/usr/bin/env python3
"""
Evaluate BERT-based text embeddings for similarity on a pairs dataset.

Dataset formats:
- JSONL: one JSON per line with fields {"text1": str, "text2": str, "label": float|int}
         label is target similarity in [0,1] or a binary 0/1 match label.
- CSV: columns: text1,text2,label

Two evaluation modes:
- regression: compute Pearson/Spearman correlation and MSE between predicted cosine similarity and label
- classification: threshold predicted similarity and compute accuracy/F1

Examples:
  python evaluate_bert_text.py --data pairs.jsonl --format jsonl --mode regression
  python evaluate_bert_text.py --data pairs.csv --format csv --mode classification --threshold 0.5
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np

from ml_models import UnifiedModel


def _read_jsonl_pairs(path: str) -> List[Tuple[str, str, float]]:
    pairs: List[Tuple[str, str, float]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t1 = str(obj.get('text1', '')).strip()
                t2 = str(obj.get('text2', '')).strip()
                label_raw = obj.get('label', 0)
                try:
                    label = float(label_raw)
                except Exception:
                    label = 0.0
                if t1 and t2:
                    pairs.append((t1, t2, label))
            except Exception:
                continue
    return pairs


def _read_csv_pairs(path: str) -> List[Tuple[str, str, float]]:
    import csv
    pairs: List[Tuple[str, str, float]] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t1 = str(row.get('text1', '')).strip()
            t2 = str(row.get('text2', '')).strip()
            try:
                label = float(row.get('label', 0))
            except Exception:
                label = 0.0
            if t1 and t2:
                pairs.append((t1, t2, label))
    return pairs


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1.0
    return float(np.dot(v1, v2) / denom)


def evaluate_similarity(data_path: str, data_format: str, mode: str, threshold: float) -> None:
    if data_format not in {'jsonl', 'csv'}:
        raise ValueError('data format must be one of {jsonl,csv}')
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        sys.exit(1)

    if data_format == 'jsonl':
        pairs = _read_jsonl_pairs(data_path)
    else:
        pairs = _read_csv_pairs(data_path)

    if not pairs:
        print('❌ No valid pairs found in dataset')
        sys.exit(1)

    model = UnifiedModel()

    gold = []
    pred = []

    for t1, t2, y in pairs:
        emb1 = model.analyze_text(t1)
        emb2 = model.analyze_text(t2)
        sim = cosine_similarity(np.array(emb1, dtype=np.float32), np.array(emb2, dtype=np.float32))
        pred.append(sim)
        gold.append(float(y))

    gold_np = np.array(gold, dtype=np.float32)
    pred_np = np.array(pred, dtype=np.float32)

    print('\n=== BERT Text Similarity Evaluation ===')
    print(f"Samples: {len(pairs)}")

    if mode == 'regression':
        try:
            from scipy.stats import pearsonr, spearmanr
            pr, _ = pearsonr(gold_np, pred_np)
            sr, _ = spearmanr(gold_np, pred_np)
            mse = float(np.mean((gold_np - pred_np) ** 2))
            print(f"Pearson: {pr:.4f}")
            print(f"Spearman: {sr:.4f}")
            print(f"MSE: {mse:.6f}")
        except Exception:
            mse = float(np.mean((gold_np - pred_np) ** 2))
            print('scipy not available; showing MSE only')
            print(f"MSE: {mse:.6f}")
    else:
        # classification mode: threshold similarities and compute accuracy/F1
        y_true = (gold_np >= threshold).astype(np.int32)
        y_pred = (pred_np >= threshold).astype(np.int32)
        acc = float(np.mean(y_true == y_pred))
        print(f"Accuracy@{threshold:.2f}: {acc:.4f}")
        try:
            from sklearn.metrics import f1_score, precision_recall_fscore_support
            f1 = float(f1_score(y_true, y_pred))
            p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            print(f"Precision: {p:.4f}")
            print(f"Recall: {r:.4f}")
            print(f"F1: {f1:.4f}")
            print(f"(Support-weighted F1: {f:.4f})")
        except Exception:
            print('sklearn not available; only accuracy reported')


def main():
    parser = argparse.ArgumentParser(description='Evaluate BERT text similarity on pairs dataset')
    parser.add_argument('--data', required=True, help='Path to dataset file (JSONL or CSV)')
    parser.add_argument('--format', choices=['jsonl', 'csv'], required=True, help='Dataset format')
    parser.add_argument('--mode', choices=['regression', 'classification'], default='regression')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classification mode')
    args = parser.parse_args()

    evaluate_similarity(args.data, args.format, args.mode, args.threshold)


if __name__ == '__main__':
    main()


