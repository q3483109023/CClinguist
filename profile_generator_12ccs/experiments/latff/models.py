"""A/B/C classifier variants for the LATFF experiment.

This module deliberately lives outside the original CClinguist implementation.
It preserves the original parameter names for the Encoder, environment branch and
classification head so the published classifier checkpoint can be reused.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrInt = Union[torch.Tensor, int]


def _as_max_length(max_length: TensorOrInt) -> int:
    """Convert the project's scalar max_length argument to a Python integer."""
    if isinstance(max_length, torch.Tensor):
        if max_length.numel() != 1:
            raise ValueError("max_length must be a scalar tensor or an integer.")
        max_length = int(max_length.item())
    if not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer.")
    return max_length


def build_padding_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """Return [B, L] bool mask where True means a Padding position.

    A length of zero is invalid. It would make masked max pooling undefined and
    normally indicates an invalid sample in the preprocessing pipeline.
    """
    if lengths.ndim != 1:
        raise ValueError(
            f"lengths must have shape [B], but received shape {tuple(lengths.shape)}."
        )
    if lengths.numel() == 0:
        raise ValueError("lengths cannot be empty.")
    if torch.is_floating_point(lengths):
        raise TypeError("lengths must contain integer sequence lengths, not floating values.")
    if torch.any(lengths < 1):
        raise ValueError("Every effective sequence length must be at least 1.")
    if torch.any(lengths > max_length):
        raise ValueError(
            f"An effective sequence length exceeds max_length={max_length}."
        )

    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)


def masked_pool(
    encoded: torch.Tensor,
    lengths: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute last, mean and max features using only valid sequence positions.

    Args:
        encoded: Transformer output with shape [B, L, D].
        lengths: Effective sequence lengths with shape [B].
        padding_mask: Optional bool tensor [B, L]. True means Padding. If given,
            it must exactly agree with lengths; this catches length/mask mistakes.
    """
    if encoded.ndim != 3:
        raise ValueError(
            f"encoded must have shape [B, L, D], but received {tuple(encoded.shape)}."
        )
    batch_size, max_length, _ = encoded.shape
    if lengths.shape != (batch_size,):
        raise ValueError(
            "lengths batch dimension must match encoded: "
            f"expected ({batch_size},), received {tuple(lengths.shape)}."
        )

    expected_padding_mask = build_padding_mask(lengths, max_length)
    if padding_mask is None:
        padding_mask = expected_padding_mask
    else:
        if padding_mask.dtype != torch.bool:
            raise TypeError("padding_mask must be a bool tensor where True means Padding.")
        if padding_mask.shape != (batch_size, max_length):
            raise ValueError(
                "padding_mask shape must match encoded [B, L]: "
                f"expected {(batch_size, max_length)}, received {tuple(padding_mask.shape)}."
            )
        if not torch.equal(padding_mask.to(lengths.device), expected_padding_mask):
            raise ValueError(
                "padding_mask does not match lengths. True must mark exactly the "
                "positions j >= effective_length for every sample."
            )
        padding_mask = padding_mask.to(encoded.device)

    # The final valid index is length - 1, never the fixed final padded index.
    last_index = (lengths.to(encoded.device) - 1).view(batch_size, 1, 1)
    last_index = last_index.expand(-1, 1, encoded.size(-1))
    last_feature = encoded.gather(dim=1, index=last_index).squeeze(1)

    denominator = lengths.to(encoded.device, dtype=encoded.dtype).clamp_min(1).unsqueeze(1)
    # masked_fill is required here rather than ``encoded * valid_mask``. If an
    # upstream padded position ever contains NaN, NaN * 0 would still be NaN.
    # Replacing padded values with zero excludes them from the mean safely.
    mean_feature = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0).sum(dim=1) / denominator

    # Padding values never participate in max. There is at least one valid value
    # because build_padding_mask rejects zero-length samples.
    fill_value = torch.finfo(encoded.dtype).min
    max_feature = encoded.masked_fill(padding_mask.unsqueeze(-1), fill_value).max(dim=1).values

    return last_feature, mean_feature, max_feature, padding_mask


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding matching the original model's formulation."""

    def __init__(self, d_model: int, max_length: int):
        super().__init__()
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(device=x.device, dtype=x.dtype)


class EncoderBackbone(nn.Module):
    """Original Transformer and environment branch, with reusable parameter names."""

    def __init__(
        self,
        dim_hidden: int,
        num_layers: int,
        num_heads: int,
        dim_input: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.input_encoder = nn.Linear(dim_input, num_heads * dim_input)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=dim_input * num_heads,
            nhead=num_heads,
            dim_feedforward=dim_hidden,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_layers
        )
        self.env_linear = nn.Linear(2, dim_hidden)

    def encode(
        self,
        data_trace: torch.Tensor,
        lengths: torch.Tensor,
        x_env: torch.Tensor,
        max_length: TensorOrInt,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if data_trace.ndim != 3:
            raise ValueError(
                "data_trace must have shape [B, L, dim_input], "
                f"but received {tuple(data_trace.shape)}."
            )
        batch_size, sequence_length, input_dim = data_trace.shape
        if input_dim != self.dim_input:
            raise ValueError(
                f"data_trace last dimension must be {self.dim_input}, received {input_dim}."
            )
        if x_env.shape != (batch_size, 2):
            raise ValueError(
                "x_env must have shape [B, 2], "
                f"but received {tuple(x_env.shape)}."
            )
        runtime_max_length = _as_max_length(max_length)
        if runtime_max_length != sequence_length:
            raise ValueError(
                "max_length must equal data_trace.shape[1]: "
                f"received {runtime_max_length}, expected {sequence_length}."
            )

        padding_mask = build_padding_mask(lengths, runtime_max_length).to(data_trace.device)
        encoded_input = self.input_encoder(data_trace)
        positional_encoding = PositionalEncoding(encoded_input.size(-1), runtime_max_length).to(
            data_trace.device
        )
        source = positional_encoding(encoded_input)
        encoded = self.transformer_encoder(
            src=source,
            src_key_padding_mask=padding_mask,
        )
        env_feature = self.env_linear(x_env)
        return encoded, env_feature, padding_mask


class _BaseClassifier(nn.Module):
    """Shared architecture and original classification head for all variants."""

    def __init__(
        self,
        num_classify: int = 12,
        dim_hidden: int = 128,
        num_layers: int = 8,
        num_heads: int = 8,
        dim_input: int = 2,
        dropout_rate: float = 0.2,
        margin: float = 1.2905981663049348,
        weight_verf: float = 0.00485668544506156,
    ):
        super().__init__()
        self.num_classify = num_classify
        self.encoder_model = EncoderBackbone(
            dim_hidden, num_layers, num_heads, dim_input, dropout_rate
        )
        self.fc = nn.Sequential(
            nn.Linear(3 * (dim_input * num_heads) + dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, int(dim_hidden / 2)),
            nn.ReLU(),
            nn.LayerNorm(int(dim_hidden / 2)),
            nn.Linear(int(dim_hidden / 2), num_classify),
        )
        self.margin = nn.Parameter(torch.tensor(float(margin)))
        self.weight_verf = weight_verf

    @property
    def expected_missing_checkpoint_keys(self) -> Tuple[str, ...]:
        return ()

    def _pool_original(self, encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact original pooling behavior; it intentionally includes padding."""
        return (
            encoded[:, -1, :],
            torch.mean(encoded, dim=1),
            torch.max(encoded, dim=1).values,
        )

    def _pool_masked(
        self, encoded: torch.Tensor, lengths: torch.Tensor, padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        last_feature, mean_feature, max_feature, _ = masked_pool(
            encoded, lengths, padding_mask
        )
        return last_feature, mean_feature, max_feature

    def _make_feature(
        self,
        data_trace: torch.Tensor,
        lengths: torch.Tensor,
        x_env: torch.Tensor,
        max_length: TensorOrInt,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def extract_features(
        self,
        data_trace: torch.Tensor,
        lengths: torch.Tensor,
        x_env: torch.Tensor,
        max_length: TensorOrInt,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Expose features for logic tests without changing the public forward API."""
        return self._make_feature(data_trace, lengths, x_env, max_length)

    def forward(
        self,
        data_trace: torch.Tensor,
        lengths: torch.Tensor,
        x_env: torch.Tensor,
        max_length: TensorOrInt,
    ) -> torch.Tensor:
        feature, _ = self._make_feature(data_trace, lengths, x_env, max_length)
        # Keep these attributes because the original get_loss implementation
        # consumes them after forward() during contrastive classifier training.
        self.feature_encoded = feature
        self.output_classify = self.fc(feature)
        return self.output_classify

    def get_loss(self, label: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keep the original classifier loss interface for future experiments."""
        label_input = label[0 : batch_size + self.num_classify]
        label_contrast = label[batch_size + self.num_classify :]
        same_label = label_input == label_contrast
        index_same = torch.where(same_label)
        index_different = torch.where(~same_label)

        output_input = self.output_classify[0 : batch_size + self.num_classify, :]
        output_contrast = self.output_classify[batch_size + self.num_classify :, :]
        feature_input = self.feature_encoded[0 : batch_size + self.num_classify, :]
        feature_contrast = self.feature_encoded[batch_size + self.num_classify :, :]

        distance_out = F.pairwise_distance(feature_input[index_different], feature_contrast[index_different], p=2)
        distance_in = F.pairwise_distance(feature_input[index_same], feature_contrast[index_same], p=2)
        zero = torch.zeros((), device=label.device)
        if distance_in.numel() == 0:
            verification_loss = 0.5 * torch.maximum(
                zero, (self.margin - torch.pow(distance_out, 2)).mean()
            )
        else:
            verification_loss = 0.5 * torch.pow(distance_in.mean(), 2) + 0.5 * torch.maximum(
                zero, (self.margin - torch.pow(distance_out, 2)).mean()
            )

        criterion = nn.CrossEntropyLoss()
        loss = (
            criterion(output_input, label_input)
            + criterion(output_contrast, label_contrast)
            + self.weight_verf * verification_loss
        )
        return loss, self.margin


class OriginalPoolingClassifier(_BaseClassifier):
    """A: original CClinguist pooling behavior, retained as the control model."""

    def _make_feature(self, data_trace, lengths, x_env, max_length):
        encoded, env_feature, padding_mask = self.encoder_model.encode(
            data_trace, lengths, x_env, max_length
        )
        last_feature, mean_feature, max_feature = self._pool_original(encoded)
        sequence_feature = torch.cat([last_feature, mean_feature, max_feature], dim=-1)
        feature = torch.cat([sequence_feature, env_feature], dim=-1)
        details = {
            "encoded": encoded,
            "padding_mask": padding_mask,
            "last": last_feature,
            "mean": mean_feature,
            "max": max_feature,
            "sequence_feature": sequence_feature,
            "feature": feature,
        }
        return feature.to(torch.float32), details


class MaskedPoolingClassifier(_BaseClassifier):
    """B: identical trainable layers, with length-correct three-way pooling."""

    def _make_feature(self, data_trace, lengths, x_env, max_length):
        encoded, env_feature, padding_mask = self.encoder_model.encode(
            data_trace, lengths, x_env, max_length
        )
        last_feature, mean_feature, max_feature = self._pool_masked(
            encoded, lengths, padding_mask
        )
        sequence_feature = torch.cat([last_feature, mean_feature, max_feature], dim=-1)
        feature = torch.cat([sequence_feature, env_feature], dim=-1)
        details = {
            "encoded": encoded,
            "padding_mask": padding_mask,
            "last": last_feature,
            "mean": mean_feature,
            "max": max_feature,
            "sequence_feature": sequence_feature,
            "feature": feature,
        }
        return feature.to(torch.float32), details


class LengthAwareAdaptiveFusionClassifier(MaskedPoolingClassifier):
    """C: masked pooling plus a length-only, three-branch Softmax gate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = nn.Linear(1, 3)
        # Softmax([0, 0, 0]) = [1/3, 1/3, 1/3]; multiplying by three
        # makes the initial gate [1, 1, 1] and preserves feature scale.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    @property
    def expected_missing_checkpoint_keys(self) -> Tuple[str, ...]:
        return ("gate.weight", "gate.bias")

    @staticmethod
    def normalized_length(lengths: torch.Tensor) -> torch.Tensor:
        if torch.any(lengths < 1):
            raise ValueError("Every effective sequence length must be at least 1.")
        return lengths.to(dtype=torch.float32).clamp(max=512).unsqueeze(1) / 512.0

    def _make_feature(self, data_trace, lengths, x_env, max_length):
        encoded, env_feature, padding_mask = self.encoder_model.encode(
            data_trace, lengths, x_env, max_length
        )
        last_feature, mean_feature, max_feature = self._pool_masked(
            encoded, lengths, padding_mask
        )
        length_feature = self.normalized_length(lengths).to(data_trace.device)
        gate_logits = self.gate(length_feature)
        gate_weights = 3.0 * torch.softmax(gate_logits, dim=1)

        sequence_feature = torch.cat(
            [
                gate_weights[:, 0:1] * last_feature,
                gate_weights[:, 1:2] * mean_feature,
                gate_weights[:, 2:3] * max_feature,
            ],
            dim=-1,
        )
        feature = torch.cat([sequence_feature, env_feature], dim=-1)
        details = {
            "encoded": encoded,
            "padding_mask": padding_mask,
            "last": last_feature,
            "mean": mean_feature,
            "max": max_feature,
            "length_feature": length_feature,
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
            "sequence_feature": sequence_feature,
            "feature": feature,
        }
        return feature.to(torch.float32), details


MODEL_VARIANTS = {
    "A": OriginalPoolingClassifier,
    "B": MaskedPoolingClassifier,
    "C": LengthAwareAdaptiveFusionClassifier,
}

