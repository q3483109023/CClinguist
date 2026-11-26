"""Reproducibility, metrics and artifact helpers for LATFF experiments."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable. Use --device cpu.")
    return device


def calculate_metrics(labels: Sequence[int], predictions: Sequence[int], label_names: Sequence[str]) -> Dict[str, object]:
    label_ids = list(range(len(label_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=label_ids, zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=label_ids)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            label_names[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in label_ids
        },
        "confusion_matrix": matrix.tolist(),
    }


def save_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def save_confusion_matrix_csv(path: Path, matrix, label_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *label_names])
        for name, row in zip(label_names, matrix):
            writer.writerow([name, *row])


def append_epoch_log(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

