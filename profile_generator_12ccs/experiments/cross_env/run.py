"""Train B/D/E/F on identical source splits, then evaluate a held-out route."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F

LATFF_DIR = Path(__file__).resolve().parent.parent / "latff"
sys.path.insert(0, str(LATFF_DIR))
from checkpoint import load_original_classifier_checkpoint
from config import ExperimentConfig, build_parser
from dataset import (CCA_LABELS, CCATraceDataset, LABEL_NAMES, build_experiment_datasets,
                     build_records, select_training_records)
from train import evaluate, make_loader, save_dataset_artifacts
from utils import append_epoch_log, resolve_device, save_confusion_matrix_csv, save_json, set_seed

from experiment_models import MODEL_VARIANTS


def cross_environment_loss(features, labels, environment_ids, temperature):
    """Same-CCA/different-environment positives and all other-CCA negatives.

    Same-environment negatives receive an additional logit margin of 0.1.
    Anchors without a cross-environment positive are excluded.
    """
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    same_label = labels[:, None] == labels[None, :]
    same_environment = (environment_ids[:, None, :] == environment_ids[None, :, :]).all(dim=-1)
    positive = same_label & ~same_environment
    negative = ~same_label
    candidates = positive | negative
    logits = logits + 0.1 * (negative & same_environment).to(logits.dtype)
    logits = logits.masked_fill(~candidates, torch.finfo(logits.dtype).min)
    valid = positive.any(dim=1)
    if not valid.any():
        return features.sum() * 0.0, 0
    log_prob = F.log_softmax(logits[valid], dim=1)
    positives = positive[valid]
    loss = -(log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positives.sum(dim=1)).mean()
    return loss, int(valid.sum().item())


def train_epoch(model, loader, optimizer, device, use_contrast, contrast_weight, temperature):
    model.train()
    total_loss = 0.0
    total_samples = 0
    active_anchors = 0
    for traces, lengths, environments, labels in loader:
        traces, lengths = traces.to(device), lengths.to(device)
        environments, labels = environments.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(traces, lengths, environments, traces.shape[1])
        loss = F.cross_entropy(logits, labels)
        if use_contrast:
            # RTT and bandwidth identify the source environment in this dataset.
            contrast_loss, count = cross_environment_loss(
                model.feature_encoded, labels, environments, temperature
            )
            loss = loss + contrast_weight * contrast_loss
            active_anchors += count
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total_samples += len(labels)
    return total_loss / total_samples, active_anchors


def ensure_cross_environment_quick_train(datasets, split_info, config):
    """Select quick training traces from distinct source environments per CCA."""
    if not config.quick:
        return
    train_environments = split_info["train_environments"]
    if len(train_environments) < 2 or config.quick_samples_per_class < 2:
        raise ValueError("Contrastive quick mode needs two training environments and two samples per class.")
    records = select_training_records(build_records(
        Path(config.train_data_path), "train", config.max_length, train_environments
    ))
    chosen = []
    for label in range(len(CCA_LABELS)):
        for sample_index in range(config.quick_samples_per_class):
            environment = train_environments[sample_index % len(train_environments)]
            candidates = [record for record in records if record.label == label and record.environment == environment]
            within_environment = sample_index // len(train_environments)
            if within_environment >= len(candidates):
                raise ValueError(f"Insufficient quick training samples for label {label} in {environment}")
            chosen.append(candidates[within_environment])
    datasets["train"] = CCATraceDataset(chosen)
    split_info["sample_counts"]["train"] = len(chosen)
    split_info["quick_train_sampling"] = (
        "one real trace per CCA from each of two training environments"
        if config.quick_samples_per_class == 2
        else "round-robin real traces across training environments per CCA"
    )


def main():
    parser = build_parser("B baseline and cross-environment/multiscale CCA experiments")
    for action in parser._actions:
        if action.dest == "model":
            action.choices = list(MODEL_VARIANTS)
            action.default = "D"
        if action.dest == "output_root":
            action.default = str(Path(__file__).resolve().parent / "runs")
    parser.add_argument("--contrast-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    if args.contrast_weight < 0 or args.temperature <= 0:
        parser.error("contrast-weight must be nonnegative and temperature must be positive")
    extra = {"contrast_weight": args.contrast_weight, "temperature": args.temperature}
    allowed = {field.name for field in fields(ExperimentConfig)}
    config = ExperimentConfig(**{key: value for key, value in vars(args).items() if key in allowed})
    if config.epochs < 1 or config.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    set_seed(config.seed)
    device = resolve_device(config.device)
    datasets, split_info = build_experiment_datasets(
        Path(config.train_data_path), Path(config.test_data_path), config.max_length,
        config.validation_fraction, config.seed, quick=config.quick,
        quick_train_envs=config.quick_train_envs, quick_test_envs=config.quick_test_envs,
        quick_samples_per_class=config.quick_samples_per_class,
    )
    ensure_cross_environment_quick_train(datasets, split_info, config)
    run_dir = Path(config.output_root) / f"{'quick' if config.quick else 'full'}_{config.model}_seed{config.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", {**config.to_dict(), **extra})
    save_dataset_artifacts(run_dir, datasets, split_info)
    loaders = {
        name: make_loader(data, config.batch_size, name == "train", config.num_workers, config.seed)
        for name, data in datasets.items()
    }
    model = MODEL_VARIANTS[config.model]().to(device)
    save_json(run_dir / "initial_checkpoint_load.json", load_original_classifier_checkpoint(model, config.init_checkpoint))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    use_contrast = config.model in ("D", "F")
    best_f1 = float("-inf")
    best_epoch = 0
    best_path = run_dir / "best_model.pt"
    for epoch in range(1, (1 if config.quick else config.epochs) + 1):
        train_loss, active = train_epoch(
            model, loaders["train"], optimizer, device, use_contrast,
            args.contrast_weight, args.temperature
        )
        validation_loss, validation = evaluate(model, loaders["validation"], device)
        append_epoch_log(run_dir / "epoch_metrics.csv", {
            "epoch": epoch, "train_loss": train_loss, "contrast_active_anchors": active,
            "validation_loss": validation_loss, "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
        })
        if validation["macro_f1"] > best_f1:
            best_f1, best_epoch = validation["macro_f1"], epoch
            torch.save({"model_name": config.model, "model_state_dict": model.state_dict(),
                        "config": {**config.to_dict(), **extra}, "epoch": epoch,
                        "validation_metrics": validation, "split_info": split_info}, best_path)
    saved = torch.load(best_path, map_location="cpu", weights_only=False)
    reloaded = MODEL_VARIANTS[config.model]().to(device)
    reloaded.load_state_dict(saved["model_state_dict"], strict=True)
    test_loss, test = evaluate(reloaded, loaders["test"], device)
    save_json(run_dir / "test_metrics.json", {"loss": test_loss, **test})
    save_confusion_matrix_csv(run_dir / "test_confusion_matrix.csv", test["confusion_matrix"], LABEL_NAMES)
    result = {"run_dir": str(run_dir), "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
              "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"],
              "contrast_active_anchors_last_epoch": active, "checkpoint_reloaded": True}
    save_json(run_dir / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
