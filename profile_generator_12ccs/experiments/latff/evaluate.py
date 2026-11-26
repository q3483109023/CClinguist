"""Re-evaluate an independently saved LATFF checkpoint on the held-out test root."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import torch

from config import ExperimentConfig
from dataset import LABEL_NAMES, build_experiment_datasets
from models import MODEL_VARIANTS
from train import evaluate, make_loader
from utils import resolve_device, save_confusion_matrix_csv, save_json, set_seed


def config_from_checkpoint(saved_config: dict) -> ExperimentConfig:
    allowed = {field.name for field in fields(ExperimentConfig)}
    return ExperimentConfig(**{key: value for key, value in saved_config.items() if key in allowed})


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved LATFF A/B/C checkpoint on the independent test set.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not {"model_name", "model_state_dict", "config"}.issubset(checkpoint):
        raise KeyError("This is not a LATFF experiment checkpoint.")
    config = config_from_checkpoint(checkpoint["config"])
    config.device = args.device
    set_seed(config.seed)
    device = resolve_device(config.device)
    datasets, split_info = build_experiment_datasets(
        Path(config.train_data_path), Path(config.test_data_path), config.max_length,
        config.validation_fraction, config.seed, quick=config.quick,
        quick_train_envs=config.quick_train_envs, quick_test_envs=config.quick_test_envs,
        quick_samples_per_class=config.quick_samples_per_class,
    )
    model_name = checkpoint["model_name"]
    model = MODEL_VARIANTS[model_name]().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loader = make_loader(datasets["test"], config.batch_size, False, config.num_workers, config.seed)
    loss, metrics = evaluate(model, loader, device)
    output_dir = args.output_dir or args.checkpoint.parent / "reevaluation"
    save_json(output_dir / "test_metrics.json", {"loss": loss, **metrics})
    save_json(output_dir / "split.json", split_info)
    save_confusion_matrix_csv(output_dir / "test_confusion_matrix.csv", metrics["confusion_matrix"], LABEL_NAMES)
    print({"loss": loss, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})


if __name__ == "__main__":
    main()

