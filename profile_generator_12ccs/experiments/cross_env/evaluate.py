"""Re-evaluate a saved B/D/E/F checkpoint on its independent test split."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "latff"))
from config import ExperimentConfig
from dataset import LABEL_NAMES, build_experiment_datasets
from train import evaluate, make_loader
from utils import resolve_device, save_confusion_matrix_csv, save_json, set_seed

from experiment_models import MODEL_VARIANTS
from run import ensure_cross_environment_quick_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    allowed = {field.name for field in fields(ExperimentConfig)}
    config = ExperimentConfig(**{key: value for key, value in saved["config"].items() if key in allowed})
    set_seed(config.seed)
    device = resolve_device(args.device)
    datasets, split_info = build_experiment_datasets(
        Path(config.train_data_path), Path(config.test_data_path), config.max_length,
        config.validation_fraction, config.seed, quick=config.quick,
        quick_train_envs=config.quick_train_envs, quick_test_envs=config.quick_test_envs,
        quick_samples_per_class=config.quick_samples_per_class,
    )
    ensure_cross_environment_quick_train(datasets, split_info, config)
    if split_info != saved["split_info"]:
        raise RuntimeError("Current data split differs from the saved checkpoint.")
    model = MODEL_VARIANTS[saved["model_name"]]().to(device)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    loader = make_loader(datasets["test"], config.batch_size, False, config.num_workers, config.seed)
    loss, metrics = evaluate(model, loader, device)
    output_dir = args.output_dir or args.checkpoint.parent / "reevaluation"
    save_json(output_dir / "test_metrics.json", {"loss": loss, **metrics})
    save_confusion_matrix_csv(output_dir / "test_confusion_matrix.csv", metrics["confusion_matrix"], LABEL_NAMES)
    print({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "output_dir": str(output_dir)})


if __name__ == "__main__":
    main()
