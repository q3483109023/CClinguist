"""Train and directly evaluate one LATFF A/B/C classifier variant.

This script never calls the original RL code. It uses the separate real test
directory only once, after the best validation epoch has been selected.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from checkpoint import load_original_classifier_checkpoint
from config import ExperimentConfig, parse_config
from dataset import LABEL_NAMES, build_experiment_datasets
from models import MODEL_VARIANTS
from utils import (
    append_epoch_log,
    calculate_metrics,
    resolve_device,
    save_confusion_matrix_csv,
    save_json,
    set_seed,
)


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        generator=generator if shuffle else None,
    )


def evaluate(model, loader, device: torch.device) -> Tuple[float, Dict[str, object]]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    labels_all, predictions_all = [], []
    with torch.no_grad():
        for traces, lengths, environments, labels in loader:
            traces = traces.to(device)
            lengths = lengths.to(device)
            environments = environments.to(device)
            labels = labels.to(device)
            logits = model(traces, lengths, environments, traces.shape[1])
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
    if total_samples == 0:
        raise RuntimeError("Evaluation loader produced no samples.")
    return total_loss / total_samples, calculate_metrics(labels_all, predictions_all, LABEL_NAMES)


def train_one_epoch(model, loader, optimizer, device: torch.device) -> float:
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total_samples = 0
    for traces, lengths, environments, labels in loader:
        traces = traces.to(device)
        lengths = lengths.to(device)
        environments = environments.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(traces, lengths, environments, traces.shape[1])
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    if total_samples == 0:
        raise RuntimeError("Training loader produced no samples.")
    return total_loss / total_samples


def make_run_directory(config: ExperimentConfig) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = "quick" if config.quick else "full"
    run_dir = Path(config.output_root) / f"{prefix}_{config.model}_seed{config.seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_dataset_artifacts(run_dir: Path, datasets, split_info: Dict[str, object]) -> None:
    save_json(run_dir / "split.json", split_info)
    for name, dataset in datasets.items():
        save_json(run_dir / f"{name}_manifest.json", {"samples": dataset.manifest()})


def run_training(config: ExperimentConfig) -> Path:
    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = make_run_directory(config)
    save_json(run_dir / "config.json", config.to_dict())

    datasets, split_info = build_experiment_datasets(
        Path(config.train_data_path),
        Path(config.test_data_path),
        config.max_length,
        config.validation_fraction,
        config.seed,
        quick=config.quick,
        quick_train_envs=config.quick_train_envs,
        quick_test_envs=config.quick_test_envs,
        quick_samples_per_class=config.quick_samples_per_class,
    )
    save_dataset_artifacts(run_dir, datasets, split_info)
    loaders = {
        "train": make_loader(datasets["train"], config.batch_size, True, config.num_workers, config.seed),
        "validation": make_loader(datasets["validation"], config.batch_size, False, config.num_workers, config.seed),
        "test": make_loader(datasets["test"], config.batch_size, False, config.num_workers, config.seed),
    }

    model = MODEL_VARIANTS[config.model]().to(device)
    init_report = load_original_classifier_checkpoint(model, config.init_checkpoint)
    save_json(run_dir / "initial_checkpoint_load.json", init_report)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    epoch_count = 1 if config.quick else config.epochs
    best_macro_f1 = float("-inf")
    best_epoch = None
    best_checkpoint = run_dir / "best_model.pt"

    for epoch in range(1, epoch_count + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device)
        validation_loss, validation_metrics = evaluate(model, loaders["validation"], device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        append_epoch_log(run_dir / "epoch_metrics.csv", row)
        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            torch.save({
                "model_name": config.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "validation_metrics": validation_metrics,
                "config": config.to_dict(),
                "split_info": split_info,
            }, best_checkpoint)

    # Reload the saved best checkpoint before touching the independent test set.
    best_data = torch.load(best_checkpoint, map_location=device, weights_only=False)
    reloaded_model = MODEL_VARIANTS[config.model]().to(device)
    reloaded_model.load_state_dict(best_data["model_state_dict"], strict=True)
    test_loss, test_metrics = evaluate(reloaded_model, loaders["test"], device)
    save_json(run_dir / "test_metrics.json", {"loss": test_loss, **test_metrics})
    save_confusion_matrix_csv(run_dir / "test_confusion_matrix.csv", test_metrics["confusion_matrix"], LABEL_NAMES)
    save_json(run_dir / "summary.json", {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "test_loss": test_loss,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "checkpoint_reloaded": True,
        "checkpoint_path": str(best_checkpoint),
    })
    print(json.dumps({
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "checkpoint_reloaded": True,
    }, indent=2))
    return run_dir


if __name__ == "__main__":
    run_training(parse_config(description="Train one direct LATFF A/B/C classifier experiment"))

