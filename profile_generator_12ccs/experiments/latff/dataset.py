"""Independent, read-only dataset loader for direct A/B/C classification.

The CSV matching, [Time_x, Time_y] features, 10-second window, per-sample
MinMaxScaler and 512-point cap match profile_generator_12ccs/problem.py.
This loader deliberately omits the RL-only environment index and action logic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# Exact 12-class mapping used in profile_generator_12ccs/problem.py.
CCA_LABELS: Dict[str, int] = {
    "htcp": 0,
    "bbr": 1,
    "vegas": 2,
    "westwood": 3,
    "scalable": 4,
    "highspeed": 5,
    "veno": 6,
    "reno": 7,
    "yeah": 8,
    "illinois": 9,
    "bic": 10,
    "cubic": 11,
}
LABEL_NAMES = [name for name, _ in sorted(CCA_LABELS.items(), key=lambda item: item[1])]
CSV_COLUMNS = [
    "_ws.col.Time", "frame.len", "ip.src", "tcp.srcport", "ip.dst", "tcp.dstport",
    "tcp.len", "tcp.seq_raw", "tcp.ack_raw", "tcp.options.timestamp.tsval",
    "tcp.options.timestamp.tsecr",
]
FILENAME_PATTERN = re.compile(r"^host_(\w+)_(\d+)\.csv$")


@dataclass(frozen=True)
class TraceRecord:
    split: str
    environment: str
    label_name: str
    label: int
    sample_index: int
    host_path: str
    capture_path: str
    trace: torch.Tensor
    env_feature: torch.Tensor
    length: int

    def manifest_entry(self) -> Dict[str, object]:
        entry = asdict(self)
        entry.pop("trace")
        entry["env_feature"] = self.env_feature.tolist()
        return entry


def _read_capture_pair(host_path: Path, capture_path: Path) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Match host/capture packets using the same keys as problem.Dataset."""
    host_df = pd.read_csv(host_path, sep=",", quotechar='"', names=CSV_COLUMNS, skiprows=1)
    if host_df.isna().any().any():
        host_df = host_df.dropna()
    host_df = host_df.rename(columns={
        "_ws.col.Time": "Time", "frame.len": "Length", "tcp.len": "DataLength",
        "tcp.seq_raw": "SeqNum", "tcp.options.timestamp.tsval": "Timestamp",
    })
    selected = ["Time", "Length", "DataLength", "SeqNum", "Timestamp"]
    host_df[["Length", "DataLength", "SeqNum", "Timestamp"]] = host_df[
        ["Length", "DataLength", "SeqNum", "Timestamp"]
    ].astype(int)
    host_df = host_df[selected]
    first_data = host_df.index[host_df["DataLength"] > 1000]
    if len(first_data) == 0:
        raise ValueError("host trace contains no packet with DataLength > 1000")
    host_df = host_df.iloc[first_data[0]:].reset_index(drop=True)
    first = host_df.loc[0, ["Length", "DataLength", "SeqNum", "Timestamp"]]

    capture_df = pd.read_csv(capture_path, sep=",", quotechar='"', names=CSV_COLUMNS, skiprows=1)
    capture_df = capture_df.rename(columns={
        "_ws.col.Time": "Time", "frame.len": "Length", "tcp.len": "DataLength",
        "tcp.seq_raw": "SeqNum", "tcp.options.timestamp.tsval": "Timestamp",
    }).reset_index(drop=True)
    # The original code removes the final capture row before integer conversion.
    capture_df = capture_df.iloc[:-1]
    capture_df[["Length", "DataLength", "SeqNum", "Timestamp"]] = capture_df[
        ["Length", "DataLength", "SeqNum", "Timestamp"]
    ].astype(int)
    capture_df = capture_df[selected]

    matches = capture_df[
        (capture_df["Length"] == first["Length"])
        & (capture_df["DataLength"] == first["DataLength"])
        & (capture_df["SeqNum"] == first["SeqNum"])
        & (capture_df["Timestamp"] == first["Timestamp"])
    ]
    if matches.empty:
        raise ValueError("the first host data packet has no matching capture packet")
    capture_df = capture_df.iloc[matches.index[0]:].reset_index(drop=True)

    merged = pd.merge(
        host_df,
        capture_df,
        on=["Length", "DataLength", "SeqNum", "Timestamp"],
        suffixes=("_host", "_seq"),
    )
    if merged.empty:
        raise ValueError("host/capture merge produced no matched packets")
    merged["Time_host"] = pd.to_datetime(merged["Time_host"], format="%H:%M:%S.%f")
    merged["Time_seq"] = pd.to_datetime(merged["Time_seq"], format="%H:%M:%S.%f")
    return merged, merged["Time_seq"].iloc[0]


def preprocess_pair(host_path: Path, capture_path: Path, max_length: int = 512) -> torch.Tensor:
    """Return the original project's normalized [Time_x, Time_y] trace."""
    merged, base_time = _read_capture_pair(host_path, capture_path)
    time_y = merged["Time_host"] - merged["Time_seq"]
    time_y = time_y.apply(lambda value: value if value >= pd.Timedelta(0) else value + pd.Timedelta(days=1))
    time_x = merged["Time_host"] - base_time
    time_x = time_x.apply(lambda value: value if value >= pd.Timedelta(0) else value + pd.Timedelta(days=1))
    trace = list(zip(time_x.dt.total_seconds(), time_y.dt.total_seconds()))
    trace = sorted(trace, key=lambda item: item[0])
    trace = [item for item in trace if item[0] <= 10]
    trace = trace[:max_length]
    if not trace:
        raise ValueError("no matched packets remain in the 10-second observation window")
    return torch.tensor(MinMaxScaler().fit_transform(trace), dtype=torch.float32)


def environment_feature(environment: str) -> torch.Tensor:
    numbers = re.findall(r"\d+", environment)
    if len(numbers) < 2:
        raise ValueError(f"environment name does not contain RTT and bandwidth: {environment}")
    return torch.tensor([int(numbers[0]), int(numbers[1])], dtype=torch.float32)


def list_environment_names(data_path: Path) -> List[str]:
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory was not found: {data_path}")
    environments = sorted(path.name for path in data_path.iterdir() if path.is_dir())
    if not environments:
        raise ValueError(f"No environment directories were found in {data_path}")
    return environments


def discover_file_pairs(data_path: Path, environments: Optional[Sequence[str]] = None):
    """Discover only known 12-CCA host/capture pairs; unknown CCA files are ignored."""
    selected_environments = list(environments) if environments is not None else list_environment_names(data_path)
    pairs = []
    for environment in selected_environments:
        environment_path = data_path / environment
        if not environment_path.is_dir():
            raise FileNotFoundError(f"Environment directory was not found: {environment_path}")
        for host_path in sorted(environment_path.glob("host_*.csv")):
            match = FILENAME_PATTERN.match(host_path.name)
            if match is None:
                continue
            label_name, sample_index_text = match.groups()
            if label_name not in CCA_LABELS:
                continue
            capture_path = environment_path / host_path.name.replace("host_", "capture_", 1)
            if not capture_path.is_file():
                raise FileNotFoundError(f"Missing capture pair for {host_path}: {capture_path}")
            pairs.append((environment, label_name, int(sample_index_text), host_path, capture_path))
    if not pairs:
        raise ValueError(f"No recognized 12-CCA host/capture pairs were found in {data_path}")
    return pairs


def build_records(
    data_path: Path,
    split: str,
    max_length: int,
    environments: Optional[Sequence[str]] = None,
) -> List[TraceRecord]:
    records = []
    failures = []
    for environment, label_name, sample_index, host_path, capture_path in discover_file_pairs(data_path, environments):
        try:
            trace = preprocess_pair(host_path, capture_path, max_length)
        except (ValueError, TypeError, pd.errors.ParserError) as error:
            failures.append(f"{host_path}: {error}")
            continue
        records.append(TraceRecord(
            split=split,
            environment=environment,
            label_name=label_name,
            label=CCA_LABELS[label_name],
            sample_index=sample_index,
            host_path=str(host_path.resolve()),
            capture_path=str(capture_path.resolve()),
            trace=trace,
            env_feature=environment_feature(environment),
            length=len(trace),
        ))
    if failures:
        preview = "\n".join(failures[:5])
        raise ValueError(f"{len(failures)} file pairs could not be preprocessed. First failures:\n{preview}")
    return records


def select_training_records(records: Sequence[TraceRecord], samples_per_group: int = 7) -> List[TraceRecord]:
    """Match the original training selection: central traces by final Time_x."""
    if samples_per_group < 1 or samples_per_group % 2 == 0:
        raise ValueError("samples_per_group must be a positive odd number.")
    grouped = defaultdict(list)
    for record in records:
        grouped[(record.environment, record.label_name)].append(record)
    selected = []
    for key, group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: float(item.trace[-1, 0]))
        if len(group) < samples_per_group:
            raise ValueError(f"{key} has {len(group)} valid samples; need {samples_per_group} for training.")
        middle = len(group) // 2
        radius = (samples_per_group - 1) // 2
        selected.extend(group[middle - radius: middle + radius + 1])
    return selected


def select_test_records(records: Sequence[TraceRecord], indices: Sequence[int] = tuple(range(9))) -> List[TraceRecord]:
    """Use the same numbered test captures exercised by test_from_params.sh."""
    wanted = set(indices)
    selected = [record for record in records if record.sample_index in wanted]
    grouped = Counter((record.environment, record.label_name) for record in selected)
    missing = [key for key, count in grouped.items() if count != len(wanted)]
    if missing:
        raise ValueError(
            "Some environment/class test groups do not contain every requested index "
            f"{sorted(wanted)}. Example: {missing[:3]}"
        )
    return selected


def limit_per_class(records: Sequence[TraceRecord], per_class: int) -> List[TraceRecord]:
    """Deterministically limit real samples for the one-epoch quick smoke test."""
    if per_class < 1:
        raise ValueError("per_class must be at least 1.")
    counts = Counter()
    selected = []
    for record in sorted(records, key=lambda item: (item.label, item.environment, item.sample_index)):
        if counts[record.label] < per_class:
            selected.append(record)
            counts[record.label] += 1
    missing = [LABEL_NAMES[label] for label in range(len(LABEL_NAMES)) if counts[label] < per_class]
    if missing:
        raise ValueError(f"Quick mode lacks enough real samples for: {missing}")
    return selected


class CCATraceDataset(Dataset):
    """Pre-padded direct-classification dataset retaining each sample's true length."""

    def __init__(self, records: Sequence[TraceRecord]):
        if not records:
            raise ValueError("CCATraceDataset requires at least one valid record.")
        self.records = list(records)
        self.max_length = max(record.length for record in self.records)
        traces = [record.trace for record in self.records]
        self.traces = pad_sequence(traces, batch_first=True, padding_value=0.0)
        self.lengths = torch.tensor([record.length for record in self.records], dtype=torch.long)
        self.environments = torch.stack([record.env_feature for record in self.records])
        self.labels = torch.tensor([record.label for record in self.records], dtype=torch.long)
        if self.traces.shape[1] != self.max_length:
            raise RuntimeError("Padding length does not match the maximum effective length.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.traces[index], self.lengths[index], self.environments[index], self.labels[index]

    def manifest(self):
        return [record.manifest_entry() for record in self.records]


def split_training_environments(environments: Sequence[str], validation_fraction: float, seed: int):
    environments = sorted(environments)
    if len(environments) < 2:
        raise ValueError("At least two training environments are required for environment-level validation.")
    shuffled = environments.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(environments) * validation_fraction))
    validation_count = min(validation_count, len(environments) - 1)
    validation = sorted(shuffled[:validation_count])
    train = sorted(shuffled[validation_count:])
    return train, validation


def build_experiment_datasets(
    train_data_path: Path,
    test_data_path: Path,
    max_length: int,
    validation_fraction: float,
    seed: int,
    quick: bool = False,
    quick_train_envs: int = 4,
    quick_test_envs: int = 2,
    quick_samples_per_class: int = 2,
):
    """Build train/validation/test datasets with environment-disjoint validation.

    The train and validation sets come from distinct network environments inside
    the original training directory. The test root is always a separate original
    directory and is never used for selecting the best epoch.
    """
    train_data_path = Path(train_data_path)
    test_data_path = Path(test_data_path)
    all_train_environments = list_environment_names(train_data_path)
    all_test_environments = list_environment_names(test_data_path)

    if quick:
        if quick_train_envs < 2:
            raise ValueError("quick_train_envs must be at least 2 for train/validation splitting.")
        all_train_environments = all_train_environments[:quick_train_envs]
        all_test_environments = all_test_environments[:quick_test_envs]

    train_environments, validation_environments = split_training_environments(
        all_train_environments, validation_fraction, seed
    )
    train_records = select_training_records(build_records(
        train_data_path, "train", max_length, train_environments
    ))
    validation_records = select_training_records(build_records(
        train_data_path, "validation", max_length, validation_environments
    ))
    test_records = select_test_records(build_records(
        test_data_path, "test", max_length, all_test_environments
    ))

    if quick:
        train_records = limit_per_class(train_records, quick_samples_per_class)
        validation_records = limit_per_class(validation_records, quick_samples_per_class)
        test_records = limit_per_class(test_records, quick_samples_per_class)

    datasets = {
        "train": CCATraceDataset(train_records),
        "validation": CCATraceDataset(validation_records),
        "test": CCATraceDataset(test_records),
    }
    source_overlap = set(record.host_path for record in train_records + validation_records) & set(
        record.host_path for record in test_records
    )
    if source_overlap:
        raise RuntimeError("Train/test source-file overlap detected: " + next(iter(source_overlap)))
    split_info = {
        "train_data_path": str(train_data_path.resolve()),
        "test_data_path": str(test_data_path.resolve()),
        "train_environments": train_environments,
        "validation_environments": validation_environments,
        "test_environments": all_test_environments,
        "quick": quick,
        "sample_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "source_file_overlap": len(source_overlap),
    }
    return datasets, split_info


def inspect_data_root(data_path: Path) -> Dict[str, object]:
    """Read-only file-level integrity report used before real experiments."""
    environments = list_environment_names(data_path)
    pairs = discover_file_pairs(data_path)
    counts = Counter(label for _, label, _, _, _ in pairs)
    pair_keys = {(environment, label, index) for environment, label, index, _, _ in pairs}
    expected_missing = []
    for environment in environments:
        for label in LABEL_NAMES:
            if not any(key[0] == environment and key[1] == label for key in pair_keys):
                expected_missing.append((environment, label))
    sample_host = pairs[0][3]
    sample_capture = pairs[0][4]
    sample_trace = preprocess_pair(sample_host, sample_capture)
    return {
        "data_path": str(data_path.resolve()),
        "environment_count": len(environments),
        "recognized_pair_count_by_class": {label: counts[label] for label in LABEL_NAMES},
        "missing_environment_class_groups": expected_missing,
        "sample_host": str(sample_host),
        "sample_capture": str(sample_capture),
        "sample_trace_length": len(sample_trace),
        "sample_trace_has_nan": bool(torch.isnan(sample_trace).any()),
        "sample_trace_min": float(sample_trace.min()),
        "sample_trace_max": float(sample_trace.max()),
    }


def _main():
    parser = argparse.ArgumentParser(description="Inspect real CClinguist 12-CCA CSV data without editing it.")
    parser.add_argument("data_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_data_root(args.data_path), indent=2))


if __name__ == "__main__":
    _main()

