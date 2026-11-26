"""Configuration shared by the independent LATFF A/B/C experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = EXPERIMENT_DIR.parents[1]


@dataclass
class ExperimentConfig:
    model: str = "A"
    # Match classifier/train.py's original sweep configuration.
    seed: int = 3402
    device: str = "cuda"
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    validation_fraction: float = 0.2
    num_workers: int = 0
    quick: bool = False
    quick_train_envs: int = 4
    quick_test_envs: int = 2
    quick_samples_per_class: int = 2
    max_length: int = 512
    train_data_path: str = str(PROFILE_DIR / "data_12ccs" / "simulation_hk2jp_1201to1203")
    test_data_path: str = str(PROFILE_DIR / "data_12ccs" / "simulation_pk2hk_12_20")
    init_checkpoint: str = str(PROFILE_DIR / "models" / "best_model_accuracy.pth.tar")
    output_root: str = str(EXPERIMENT_DIR / "runs")

    def to_dict(self):
        return asdict(self)


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", choices=["A", "B", "C"], default="A")
    parser.add_argument("--seed", type=int, default=3402)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-train-envs", type=int, default=4)
    parser.add_argument("--quick-test-envs", type=int, default=2)
    parser.add_argument("--quick-samples-per-class", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--train-data-path", default=str(PROFILE_DIR / "data_12ccs" / "simulation_hk2jp_1201to1203"))
    parser.add_argument("--test-data-path", default=str(PROFILE_DIR / "data_12ccs" / "simulation_pk2hk_12_20"))
    parser.add_argument("--init-checkpoint", default=str(PROFILE_DIR / "models" / "best_model_accuracy.pth.tar"))
    parser.add_argument("--output-root", default=str(EXPERIMENT_DIR / "runs"))
    return parser


def parse_config(args: Sequence[str] | None = None, description: str = "LATFF experiment") -> ExperimentConfig:
    namespace = build_parser(description).parse_args(args)
    config = ExperimentConfig(**vars(namespace))
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not 0.0 < config.validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in the interval (0, 0.5).")
    if config.max_length < 1:
        raise ValueError("max_length must be at least 1.")
    return config

