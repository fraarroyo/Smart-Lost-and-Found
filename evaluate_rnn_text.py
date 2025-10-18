#!/usr/bin/env python3
"""
Evaluate the BidirectionalDescriptionRNN on a labeled text dataset.

Supported dataset formats:
- JSONL: one JSON per line with fields {"text": str, "label": int}
- CSV: columns: text,label (label as int)

Usage examples:
  python evaluate_rnn_text.py --data training_data/training_sample_7.json --format jsonl
  python evaluate_rnn_text.py --data dataset.csv --format csv --batch-size 64
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from rnn_models import RNNModelManager


def _read_jsonl(path: str) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = str(obj.get('text', '')).strip()
                label = int(obj.get('label', 0))
                if text:
                    samples.append((text, label))
            except Exception:
                continue
    return samples


def _read_csv(path: str) -> List[Tuple[str, int]]:
    import csv
    samples: List[Tuple[str, int]] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                text = str(row.get('text', '')).strip()
                label = int(row.get('label', 0))
                if text:
                    samples.append((text, label))
            except Exception:
                continue
    return samples


def _build_vocab_if_needed(manager: RNNModelManager, texts: List[str], min_freq: int = 1) -> None:
    if getattr(manager, 'vocab', None) and len(manager.vocab) > 0:
        return
    # Simple whitespace vocab
    from collections import Counter
    counter: Counter = Counter()
    for t in texts:
        counter.update(t.lower().split())
    vocab = {w: i + 1 for i, (w, c) in enumerate(counter.items()) if c >= min_freq}
    manager.vocab = vocab
    manager.vocab_size = len(vocab)


def _texts_to_tensor(manager: RNNModelManager, texts: List[str], max_length: int = 20) -> torch.Tensor:
    sequences: List[List[int]] = []
    for t in texts:
        words = t.lower().split()
        idxs = [manager.vocab.get(w, 0) for w in words][:max_length]
        if len(idxs) < max_length:
            idxs.extend([0] * (max_length - len(idxs)))
        sequences.append(idxs)
    return torch.tensor(sequences, dtype=torch.long)


@torch.no_grad()
def evaluate_description_model(data_path: str, data_format: str, batch_size: int, device: str) -> None:
    if data_format not in {'jsonl', 'csv'}:
        raise ValueError('data format must be one of {jsonl,csv}')

    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        sys.exit(1)

    # Load samples
    if data_format == 'jsonl':
        samples = _read_jsonl(data_path)
    else:
        samples = _read_csv(data_path)

    if not samples:
        print('❌ No valid samples found in the dataset')
        sys.exit(1)

    texts, labels = zip(*samples)
    labels_np = np.array(labels, dtype=np.int64)

    # Initialize model manager and load weights if available
    manager = RNNModelManager(device=device)
    try:
        manager.load_models()
    except Exception:
        pass

    # Ensure vocab exists
    _build_vocab_if_needed(manager, list(texts))

    # Prepare batches
    model = manager.description_model
    model.eval()
    device_t = torch.device(device)

    num_classes = getattr(model, 'fc', None).out_features if hasattr(model, 'fc') else int(np.max(labels_np) + 1)

    correct = 0
    total = 0
    all_preds: List[int] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_labels = torch.tensor(labels_np[start:start + batch_size], dtype=torch.long, device=device_t)

        inputs = _texts_to_tensor(manager, list(batch_texts)).to(device_t)
        outputs, _ = model(inputs)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        correct += int((preds == batch_labels).sum().item())
        total += int(batch_labels.shape[0])
        all_preds.extend([int(p.item()) for p in preds])

    accuracy = correct / max(1, total)

    # Per-class precision/recall/F1
    try:
        from sklearn.metrics import classification_report
        report = classification_report(labels_np, np.array(all_preds), digits=3)
    except Exception:
        report = 'sklearn not available; install scikit-learn for detailed metrics'

    print('\n=== Description RNN Evaluation ===')
    print(f"Samples: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    print('\nClassification report:')
    print(report)


def main():
    parser = argparse.ArgumentParser(description='Evaluate BidirectionalDescriptionRNN on labeled text data')
    parser.add_argument('--data', required=True, help='Path to dataset file (JSONL or CSV)')
    parser.add_argument('--format', choices=['jsonl', 'csv'], required=True, help='Dataset format')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    evaluate_description_model(args.data, args.format, args.batch_size, args.device)


if __name__ == '__main__':
    main()


