"""Explicit checkpoint loading checks for the LATFF experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


def load_original_classifier_checkpoint(
    model: nn.Module, checkpoint_path: str | Path
) -> Dict[str, object]:
    """Load known public parameters and reject any unexpected key mismatch.

    C is permitted to miss only its newly introduced gate parameters. A and B
    must be exact state-dict matches with the supplied original checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint was not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise KeyError("Checkpoint must be a dictionary containing 'state_dict'.")
    state_dict = checkpoint["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint['state_dict'] must be a state-dict dictionary.")

    model_state = model.state_dict()
    checkpoint_keys = set(state_dict)
    model_keys = set(model_state)
    expected_missing = set(getattr(model, "expected_missing_checkpoint_keys", ()))
    missing = model_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_keys
    invalid_missing = missing - expected_missing

    if invalid_missing or unexpected:
        raise RuntimeError(
            "Checkpoint keys do not match the requested model. "
            f"Unexpected missing keys: {sorted(invalid_missing)}; "
            f"unexpected checkpoint keys: {sorted(unexpected)}."
        )
    if missing != expected_missing:
        raise RuntimeError(
            "Checkpoint missing-key set differs from this model's declared new parameters. "
            f"Expected {sorted(expected_missing)}, found {sorted(missing)}."
        )

    for key in checkpoint_keys:
        if state_dict[key].shape != model_state[key].shape:
            raise RuntimeError(
                f"Shape mismatch for '{key}': checkpoint {tuple(state_dict[key].shape)}, "
                f"model {tuple(model_state[key].shape)}."
            )

    if not missing:
        model.load_state_dict(state_dict, strict=True)
        strict = True
    else:
        result = model.load_state_dict(state_dict, strict=False)
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise RuntimeError(
                "Non-strict loading produced an unapproved mismatch: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}."
            )
        strict = False

    loaded_keys = sorted(checkpoint_keys & model_keys)
    loaded_parameters = sum(model_state[key].numel() for key in loaded_keys)
    total_parameters = sum(value.numel() for value in model_state.values())
    return {
        "checkpoint": str(checkpoint_path),
        "strict": strict,
        "loaded_key_count": len(loaded_keys),
        "loaded_parameter_count": loaded_parameters,
        "model_parameter_count": total_parameters,
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
    }

