"""Independent B, cross-environment contrastive, and multiscale variants."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "latff"))
from models import MaskedPoolingClassifier


def segment_mean(encoded: torch.Tensor, lengths: torch.Tensor, start: int, end: int, parts: int) -> torch.Tensor:
    """Mean over a relative segment of each valid trace; empty segments yield zero."""
    positions = torch.arange(encoded.size(1), device=encoded.device).unsqueeze(0)
    first = lengths * start // parts
    last = lengths * end // parts
    valid = (positions >= first.unsqueeze(1)) & (positions < last.unsqueeze(1))
    valid = valid & (positions < lengths.unsqueeze(1))
    counts = valid.sum(dim=1).clamp_min(1).unsqueeze(1)
    return encoded.masked_fill(~valid.unsqueeze(-1), 0.0).sum(dim=1) / counts


class MultiscaleClassifier(MaskedPoolingClassifier):
    """B plus changes between halves and between first/last quarters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.encoder_model.dim_input * self.encoder_model.num_heads
        self.temporal_projection = nn.Linear(2 * width, width, bias=False)
        nn.init.zeros_(self.temporal_projection.weight)

    @property
    def expected_missing_checkpoint_keys(self):
        return ("temporal_projection.weight",)

    def _make_feature(self, data_trace, lengths, x_env, max_length):
        encoded, env_feature, padding_mask = self.encoder_model.encode(
            data_trace, lengths, x_env, max_length
        )
        last_feature, mean_feature, max_feature = self._pool_masked(
            encoded, lengths, padding_mask
        )
        half_change = segment_mean(encoded, lengths, 1, 2, 2) - segment_mean(encoded, lengths, 0, 1, 2)
        quarter_change = segment_mean(encoded, lengths, 3, 4, 4) - segment_mean(encoded, lengths, 0, 1, 4)
        half_change = torch.where((lengths >= 2).unsqueeze(1), half_change, 0.0)
        quarter_change = torch.where((lengths >= 4).unsqueeze(1), quarter_change, 0.0)
        mean_feature = mean_feature + self.temporal_projection(torch.cat([half_change, quarter_change], dim=1))
        sequence_feature = torch.cat([last_feature, mean_feature, max_feature], dim=1)
        feature = torch.cat([sequence_feature, env_feature], dim=1)
        return feature.float(), {"encoded": encoded, "padding_mask": padding_mask, "sequence_feature": sequence_feature}


MODEL_VARIANTS = {
    "B": MaskedPoolingClassifier,
    "D": MaskedPoolingClassifier,  # B plus cross-environment contrastive training
    "E": MultiscaleClassifier,      # B plus multiscale temporal features
    "F": MultiscaleClassifier,      # both changes
}
