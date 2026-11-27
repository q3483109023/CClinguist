from pathlib import Path
from types import SimpleNamespace
import sys
import json
import torch

from mef_experiment import (
    seed_all,
    load_dtw,
    load_observations_uncached
)
from mef_observation_cache import read_cache, sha256

ROOT = Path.home() / "dl-workspace/experiment-results/CClinguist/mef-15ccs"

args = SimpleNamespace(
    batch_size=36,
    train_data=Path("data_15ccs/simulation_pk2hk_15CCs"),
    test_data=Path("data_15ccs/simulation_15CCs"),
    classifier=Path("models/checkpoint_9000.pth.tar"),
    shared=ROOT / "shared",
    cache_dir=ROOT / "cache",
)

device = torch.device("cuda:0")

seed_all(3404)

summary = json.loads(
    (args.cache_dir / "build_summary.json").read_text()
)

assert len(summary) == 11

for item in summary:
    assert sha256(
        args.cache_dir / item["file"]
    ) == item["sha256"]

print("ALL CACHE HASHES PASS", flush=True)

for job in sys.argv[1:]:
    training = job == "train"
    index = 0 if training else int(job)

    split = "train" if training else "test"
    path = args.train_data if training else args.test_data

    dtw, envs = load_dtw(
        args.shared, path, split, device
    )

    cached_p, cached_y = read_cache(
        args, envs, device, training, index
    )

    raw_p, raw_y = load_observations_uncached(
        args, dtw, envs, device, training, index
    )

    max_diff = (
        cached_p - raw_p
    ).abs().max().item()

    exact = torch.equal(cached_p, raw_p)

    print(
        f"VERIFY {job}: "
        f"exact={exact}, "
        f"max_abs_diff={max_diff:.10g}, "
        f"labels_equal={torch.equal(cached_y, raw_y)}",
        flush=True
    )

    assert torch.equal(cached_y, raw_y)
    assert exact, "Cached probabilities differ"

print("INDEPENDENT CACHE VERIFICATION PASS", flush=True)
