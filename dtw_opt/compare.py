"""对比原 DTW、弯曲窗口 DTW、z-score DTW 和导数 DTW。

没有现成的采集轨迹时，用六类固定形状的合成曲线。
每类多条样本只加了很小的噪声，最近邻标签是知道的，所以可以离线算准确率。

用法（在 dtw_opt 目录下）：
    python compare.py
"""

import os
import time

import numpy as np

from dtw_band import dtw_distance_band
from dtw_original import dtw_distance
from dtw_robust import dtw_derivative, dtw_zscore, dtw_zscore_band


SERIES_LENGTH = 160
SAMPLES_PER_CLASS = 3
RANDOM_SEED = 7
BAND_RATIO = 0.1


def _shapes(length):
    t = np.linspace(0, 1, length)
    return {
        "cubic": np.minimum(1.0, np.sqrt(t + 0.02)),
        "bbr": 0.55 + 0.30 * np.sin(2 * np.pi * 6 * t) + 0.15 * (t > 0.55),
        "reno": np.maximum(0.05, 1.0 - np.floor(t * 5) / 5),
        "vegas": 0.35 + 0.08 * np.sin(2 * np.pi * 2 * t),
        "yeah": 0.2 + 0.9 * t + 0.1 * np.sin(2 * np.pi * 12 * t),
        "illinois": np.where(t < 0.4, 2.2 * t, 0.88 - 0.35 * (t - 0.4)),
    }


def make_dataset(length=SERIES_LENGTH, samples_per_class=SAMPLES_PER_CLASS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    series = []
    labels = []
    for name, base in _shapes(length).items():
        for _ in range(samples_per_class):
            noise = rng.normal(0, 0.02, size=length)
            series.append(base + noise)
            labels.append(name)
    return series, labels


def nearest_label(query, gallery, labels, distance_fn, skip_index=None):
    best_distance = None
    best_label = None
    for index, sample in enumerate(gallery):
        if index == skip_index:
            continue
        distance = distance_fn(query, sample)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_label = labels[index]
    return best_label


def accuracy(queries, gallery, query_labels, gallery_labels, distance_fn, skip_same_index=False):
    correct = 0
    for index, query in enumerate(queries):
        skip = index if skip_same_index else None
        predicted = nearest_label(query, gallery, gallery_labels, distance_fn, skip_index=skip)
        correct += int(predicted == query_labels[index])
    return correct / len(queries)


def mean_pair_seconds(series, distance_fn, pair_count=40, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    count = len(series)
    started = time.perf_counter()
    for _ in range(pair_count):
        i, j = rng.integers(0, count, size=2)
        while j == i:
            j = int(rng.integers(0, count))
        distance_fn(series[i], series[j])
    return (time.perf_counter() - started) / pair_count


def offset_queries(series, offset=5.0):
    return [sample + offset for sample in series]


def scale_queries(series, scale=0.15):
    return [sample * scale for sample in series]


def main():
    series, labels = make_dataset()
    methods = [
        ("original", lambda a, b: dtw_distance(a, b)),
        ("band_10%", lambda a, b: dtw_distance_band(a, b, BAND_RATIO)),
        ("zscore", lambda a, b: dtw_zscore(a, b)),
        ("derivative", lambda a, b: dtw_derivative(a, b)),
        ("zscore+band_10%", lambda a, b: dtw_zscore_band(a, b, BAND_RATIO)),
    ]

    shifted = [np.roll(sample, 8) for sample in series]
    shifted_distance_full = dtw_distance(series[0], shifted[0])
    shifted_distance_band = dtw_distance_band(series[0], shifted[0], BAND_RATIO)

    header = (
        f"{'method':<18}{'ms/pair':>10}{'clean':>10}{'offset+5':>10}{'scale_0.15':>12}"
    )
    lines = [
        "合成曲线：6 类算法形状，每类 3 条，长度 160。",
        "clean：留一法最近邻，查询就是样本本身。",
        "offset+5：查询整体加 5，模拟曲线被抬高。",
        "scale_0.15：查询幅度乘 0.15，模拟低带宽下排队时延变小。",
        f"平移 8 个点后的距离  original={shifted_distance_full:.4f}  band={shifted_distance_band:.4f}",
        "",
        header,
    ]
    print("\n".join(lines))

    raised = offset_queries(series)
    shrunk = scale_queries(series)
    for name, distance_fn in methods:
        seconds = mean_pair_seconds(series, distance_fn)
        clean = accuracy(series, series, labels, labels, distance_fn, skip_same_index=True)
        under_offset = accuracy(raised, series, labels, labels, distance_fn, skip_same_index=False)
        under_scale = accuracy(shrunk, series, labels, labels, distance_fn, skip_same_index=False)
        row = f"{name:<18}{seconds * 1000:10.2f}{clean:10.3f}{under_offset:10.3f}{under_scale:12.3f}"
        print(row)
        lines.append(row)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "compare_results.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    main()
