"""Build train-only, class-indexed DTW tables for the controlled MEF experiment."""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from problem import Dataset
from dtw_cal import dtw_similarity

K = 12
N = 7
CLASS_NAMES = ["htcp", "bbr", "vegas", "westwood", "scalable", "highspeed",
               "veno", "reno", "yeah", "illinois", "bic", "cubic"]


def prepare(train_path, test_path, out):
    train_envs = os.listdir(train_path)
    test_envs = os.listdir(test_path)
    assert len(train_envs) == 43 and len(set(train_envs)) == 43, train_envs
    assert len(test_envs) == 32 and set(test_envs) <= set(train_envs), test_envs
    assert all((train_path / name).is_dir() for name in train_envs)
    assert all((test_path / name).is_dir() for name in test_envs)

    # Only training traces are loaded. Test directory names are metadata only.
    dataset = Dataset(36, str(train_path), None, [0])
    dataset.make_dataset_1209(36, train=True, train_data_num=N)
    assert os.listdir(train_path) == train_envs, "Training environment order changed"
    assert dataset.input_data_num == N
    assert len(dataset.x_trace) == len(dataset.y) == len(dataset.x_env_id) == len(train_envs)*K*N

    by_class = defaultdict(list)
    for trace, label, env in zip(dataset.x_trace, dataset.y, dataset.x_env_id):
        e, c = int(env[0]), int(label[0])
        values = np.asarray(trace, dtype=np.float64)[:, 1]
        assert len(values) and np.isfinite(values).all()
        by_class[e, c].append(values)

    # Original get_trace() uses positional offsets; check its layout explicitly.
    labels = np.asarray(dataset.y).reshape(len(train_envs), K*N)
    env_ids = np.asarray(dataset.x_env_id).reshape(len(train_envs), K*N)
    for e in range(1, len(train_envs)+1):
        assert (env_ids[e-1] == e).all(), (e, "environment block")
        assert np.array_equal(labels[e-1], labels[0]), (e, "class layout")
        for c in range(K):
            assert len(by_class[e, c]) == N, (e, c, len(by_class[e, c]))
    print("[MEF] Training trace layout PASS:", len(dataset.y), flush=True)

    # Class distances: mean of seven duration-rank-matched trace distances.
    # Retain original full DTW kernel and normalization; no band restriction.
    matrix = np.zeros((len(train_envs), K, K), dtype=np.float64)
    for e, name in enumerate(train_envs, 1):
        for i in range(K):
            for j in range(i+1, K):
                d = np.mean([dtw_similarity(a, b)
                             for a, b in zip(by_class[e, i], by_class[e, j])])
                matrix[e-1, i, j] = matrix[e-1, j, i] = d
        print(f"[MEF] DTW {e}/{len(train_envs)} {name}", flush=True)

    def save_table(path, data):
        assert np.isfinite(data).all() and (data >= 0).all()
        assert np.allclose(data, data.transpose(0, 2, 1))
        assert np.allclose(np.diagonal(data, axis1=1, axis2=2), 0)
        rows = [(e+1, i, j, m[i,j]) for e,m in enumerate(data)
                for i in range(K) for j in range(i,K)]
        np.savetxt(path, np.array(rows), delimiter="\t",
                   fmt=["%d", "%d", "%d", "%.17g"])
        print(f"[MEF] {path.name}: {data.shape}, {len(rows)} rows", flush=True)

    test_indices = [train_envs.index(name) for name in test_envs]
    assert os.listdir(test_path) == test_envs, "Test environment order changed"
    out.mkdir(parents=True, exist_ok=True)
    save_table(out / "train_dtw.txt", matrix)
    save_table(out / "test_dtw.txt", matrix[test_indices])
    (out / "manifest.json").write_text(json.dumps({
        "train_envs": train_envs, "test_envs": test_envs,
        "class_names_by_id": CLASS_NAMES,
        "samples_per_class": N,
        "aggregation": "mean of seven duration-rank-matched full DTW distances",
        "train_path": str(train_path.resolve()), "test_path": str(test_path.resolve()),
        "test_traces_used": False,
    }, indent=2) + "\n")
    print("[MEF] ALL PASS", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--test", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    prepare(args.train, args.test, args.out)
