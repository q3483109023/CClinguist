"""Persistent frozen-classifier observations for DTW-v2 15CCs."""

import argparse
import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def dataset_signature(path):
    root = Path(path)
    records = []

    # Preserve original filesystem iteration order,
    # because the upstream loader uses os.listdir().
    for env in os.listdir(root):
        folder = root / env
        assert folder.is_dir(), folder

        for name in os.listdir(folder):
            file = folder / name

            if file.is_file():
                stat = file.stat()

                records.append([
                    env, name,
                    stat.st_size,
                    stat.st_mtime_ns
                ])

    payload = json.dumps(records).encode()
    return hashlib.sha256(payload).hexdigest()


def metadata(args, envs, training, index):
    from dtw_experiment import K

    split = "train" if training else "test"
    path = args.train_data if training else args.test_data

    manifest = json.loads(
        (args.shared / "manifest.json").read_text()
    )

    assert manifest["test_traces_used"] is False
    assert manifest[split + "_envs"] == envs
    assert os.listdir(path) == envs

    source_root = Path(__file__).resolve().parent

    sources = [
        "dtw_experiment.py",
        "dtw_observation_cache.py",
        "problem_newcca.py",
        "model_classify_unknown.py",
        "dtw_cal.py",
    ]

    return {
        "version": 1,
        "split": split,
        "index": index,
        "num_classes": K,
        "samples_per_class": 7 if training else 1,
        "batch_size": args.batch_size,
        "envs": envs,
        "class_names": manifest["class_names_by_id"],
        "data_path": str(path.resolve()),
        "dataset_signature":
            dataset_signature(str(path.resolve())),
        "classifier_sha256": sha256(args.classifier),
        "source_sha256": {
            name: sha256(source_root / name)
            for name in sources
        },
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }


def cache_path(args, training, index):
    name = "train.pt" if training else f"test-{index}.pt"
    return args.cache_dir / name


def validate(probs, labels, meta):
    k = meta["num_classes"]
    n = meta["samples_per_class"]
    e = len(meta["envs"])

    assert probs.shape == (e, k*n, k)
    assert labels.shape == (k*n,)
    assert probs.dtype == torch.float32
    assert labels.dtype == torch.int64

    assert torch.isfinite(probs).all()
    assert (probs >= 0).all()

    assert torch.allclose(
        probs.sum(-1),
        torch.ones_like(probs[:, :, 0]),
        atol=1e-5
    )

    assert torch.bincount(
        labels, minlength=k
    ).tolist() == [n]*k


def read_cache(args, envs, device, training, index):
    start = time.perf_counter()
    path = cache_path(args, training, index)

    assert path.is_file(), path

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    expected = metadata(
        args, envs, training, index
    )

    assert data["metadata"] == expected, (
        f"Cache metadata mismatch: {path}"
    )

    probs = data["probs"]
    labels = data["labels"]

    validate(probs, labels, expected)

    print(
        f"[CACHE] HIT {path.name} "
        f"shape={tuple(probs.shape)} "
        f"elapsed={time.perf_counter()-start:.3f}s",
        flush=True
    )

    return probs.to(device), labels.to(device)


def build(args):
    from dtw_experiment import (
        seed_all,
        load_dtw,
        load_observations_uncached
    )

    assert torch.cuda.is_available()

    device = torch.device("cuda:0")
    seed_all(3402)

    args.cache_dir.mkdir(
        parents=True, exist_ok=False
    )

    jobs = [(True, 0)] + [
        (False, i) for i in range(10)
    ]

    timings = []

    for training, index in jobs:
        start = time.perf_counter()

        split = "train" if training else "test"
        data_path = (
            args.train_data if training else args.test_data
        )

        dtw, envs = load_dtw(
            args.shared, data_path, split, device
        )

        meta = metadata(
            args, envs, training, index
        )

        probs, labels = load_observations_uncached(
            args, dtw, envs, device, training, index
        )

        probs = probs.detach().cpu()
        labels = labels.detach().cpu()

        validate(probs, labels, meta)

        path = cache_path(args, training, index)
        tmp = path.with_suffix(".tmp")

        torch.save({
            "metadata": meta,
            "probs": probs,
            "labels": labels,
        }, tmp)

        os.replace(tmp, path)

        # Verify lossless serialization.
        loaded = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        assert torch.equal(probs, loaded["probs"])
        assert torch.equal(labels, loaded["labels"])
        assert loaded["metadata"] == meta

        elapsed = time.perf_counter() - start

        timings.append({
            "file": path.name,
            "shape": list(probs.shape),
            "elapsed_s": elapsed,
            "sha256": sha256(path),
        })

        print(
            f"[CACHE] BUILT {path.name} "
            f"shape={tuple(probs.shape)} "
            f"elapsed={elapsed:.2f}s",
            flush=True
        )

    (args.cache_dir / "build_summary.json").write_text(
        json.dumps(timings, indent=2) + "\n"
    )

    print("[CACHE] ALL 11 CACHES PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    for name in [
        "train-data", "test-data", "shared",
        "classifier", "cache-dir"
    ]:
        parser.add_argument(
            "--" + name,
            required=True,
            type=Path
        )

    parser.add_argument(
        "--batch-size", type=int, default=36
    )

    build(parser.parse_args())
