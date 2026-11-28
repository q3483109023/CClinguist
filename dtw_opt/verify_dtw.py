"""核对三件事：

1. Python 整表与 dtw 库的 .distance / max(n, m) 一致（symmetric2）。
2. 12 类真实抓包曲线上，band 与 python 的距离、半径是否被撑开、格子数和耗时。
3. 距离表一致时，才可以用已训练策略做识别率对照。耗时只比较 python 与 band。

用法（在 dtw_opt 目录下）：
    python verify_dtw.py
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from dtw_band import dtw_distance_band
from dtw_original import (
    dtw_distance,
    radius_was_widened,
    sakoe_chiba_radius,
    window_cell_count,
)

BAND_RATIO = 0.1
ALGORITHMS = [
    "htcp", "bbr", "vegas", "westwood", "scalable", "highspeed",
    "veno", "reno", "yeah", "illinois", "bic", "cubic",
]
EXAMPLE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..",
    "SOTA",
    "CCanalyzer",
    "Data_example",
    "rtt_130ms_bdw_10000Kbps",
))


def library_distance(series_a, series_b, window_size=None):
    from dtw import dtw

    kwargs = {"dist_method": "euclidean", "distance_only": True}
    if window_size is not None:
        kwargs["window_type"] = "sakoechiba"
        kwargs["window_args"] = {"window_size": window_size}
    alignment = dtw(series_a, series_b, **kwargs)
    return alignment.distance / max(len(series_a), len(series_b))


def assert_close(name, got, expected):
    if not np.isclose(got, expected, rtol=1e-6, atol=1e-6):
        raise SystemExit(f"{name}: python={got} library={expected}")


def check_library():
    rng = np.random.default_rng(7)
    cases = {
        "常数偏移": (np.zeros(8), np.full(8, 3.0)),
        "不等长": (np.linspace(0, 1, 20), np.linspace(0, 1, 12) + 0.2),
        "随机": (rng.normal(size=30), rng.normal(size=30)),
        "平移8点": (np.sin(np.linspace(0, 4 * np.pi, 40)), np.roll(np.sin(np.linspace(0, 4 * np.pi, 40)), 8)),
    }
    lines = ["短序列对照 dtw 库（整表）"]
    for name, (series_a, series_b) in cases.items():
        got = dtw_distance(series_a, series_b, band_ratio=None)
        expected = library_distance(series_a, series_b)
        assert_close(name, got, expected)
        lines.append(f"  {name}: {got:.6f}")
    shifted = np.roll(np.sin(np.linspace(0, 6 * np.pi, 80)), 8)
    base = np.sin(np.linspace(0, 6 * np.pi, 80))
    full = dtw_distance(base, shifted)
    band = dtw_distance_band(base, shifted, BAND_RATIO)
    library = library_distance(base, shifted)
    assert_close("窗口内平移", full, library)
    assert_close("窗口内 band", band, full)
    lines.append(f"  长度 80 平移 8 点，整表与 10% 窗口都是 {full:.6f}")

    query = np.sin(np.linspace(0, 8 * np.pi, 50))
    reference = np.roll(query, 20)
    radius = sakoe_chiba_radius(len(query), len(reference), BAND_RATIO)
    got_band = dtw_distance(query, reference, band_ratio=BAND_RATIO)
    library_band = library_distance(query, reference, window_size=radius)
    assert_close("库的 Sakoe-Chiba", got_band, library_band)
    lines.append(
        f"  平移 20 点时，10% 窗口与库的 sakoechiba 都是 {got_band:.6f}，整表是 {dtw_distance(query, reference):.6f}"
    )
    print("\n".join(lines))
    return lines


def _read_table(path, columns):
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=columns, engine="python")
    if frame.iloc[0].astype(str).str.contains("Time|Type|Length").any():
        frame = frame.iloc[1:].reset_index(drop=True)
    return frame


def delay_series(host_path, capture_path, max_len=512):
    """按项目的方式抽出排队时延：发送时刻减捕获时刻，截到 10 秒和 512 点，再按列 MinMax。"""
    host = _read_table(host_path, ["Time", "Length", "DataLength", "SeqNum"])
    capture = _read_table(capture_path, ["Type", "Time", "Length", "DataLength", "SeqNum"])
    host["Length"] = pd.to_numeric(host["Length"], errors="coerce")
    host["DataLength"] = pd.to_numeric(host["DataLength"], errors="coerce")
    host["SeqNum"] = pd.to_numeric(host["SeqNum"], errors="coerce")
    host = host.dropna(subset=["Time", "Length", "DataLength", "SeqNum"])
    data_rows = host.index[host["DataLength"] > 1000]
    if len(data_rows) == 0:
        raise ValueError(f"没有数据包: {host_path}")
    host = host.iloc[data_rows[0]:].copy()
    capture = capture[capture["Type"].astype(str) == "seq"].copy()
    capture["Length"] = pd.to_numeric(capture["Length"], errors="coerce")
    capture["DataLength"] = pd.to_numeric(capture["DataLength"], errors="coerce")
    capture["SeqNum"] = pd.to_numeric(capture["SeqNum"], errors="coerce")
    capture = capture.dropna(subset=["Time", "Length", "DataLength", "SeqNum"])
    merged = host.merge(
        capture,
        on=["Length", "DataLength", "SeqNum"],
        suffixes=("_host", "_seq"),
    )
    if merged.empty:
        raise ValueError(f"host 与 capture 没有对齐: {host_path}")
    merged["Time_host"] = pd.to_datetime(merged["Time_host"], format="%H:%M:%S.%f")
    merged["Time_seq"] = pd.to_datetime(merged["Time_seq"], format="%H:%M:%S.%f")
    delay = (merged["Time_host"] - merged["Time_seq"]).dt.total_seconds()
    delay = delay.where(delay >= 0, delay + 24 * 3600)
    base = merged["Time_host"].iloc[0]
    clock = (merged["Time_host"] - base).dt.total_seconds()
    clock = clock.where(clock >= 0, clock + 24 * 3600)
    points = np.column_stack([clock.to_numpy(), delay.to_numpy()])
    points = points[points[:, 0] <= 10]
    if points.shape[0] > max_len:
        points = points[:max_len]
    if points.shape[0] < 2:
        raise ValueError(f"曲线太短: {host_path}")
    scaled = MinMaxScaler().fit_transform(points)
    return scaled[:, 1]


def load_twelve_curves(example_dir=EXAMPLE_DIR):
    """每个算法取重复实验编号的中间一条。"""
    series = []
    names = []
    for name in ALGORITHMS:
        paths = []
        for index in range(0, 9):
            host_path = os.path.join(example_dir, f"host_{name}_{index}.csv")
            capture_path = os.path.join(example_dir, f"capture_{name}_{index}.csv")
            if os.path.isfile(host_path) and os.path.isfile(capture_path):
                paths.append((host_path, capture_path))
        if not paths:
            raise SystemExit(f"缺少 {name} 的 host/capture: {example_dir}")
        host_path, capture_path = paths[len(paths) // 2]
        series.append(delay_series(host_path, capture_path))
        names.append(name)
    return names, series


def check_real_curves():
    names, series = load_twelve_curves()
    lengths = [len(item) for item in series]
    pair_count = 0
    mismatch = 0
    max_abs = 0.0
    widened = 0
    cell_ratios = []
    full_seconds = 0.0
    band_seconds = 0.0
    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            pair_count += 1
            radius = sakoe_chiba_radius(lengths[i], lengths[j], BAND_RATIO)
            nominal = int(np.ceil(BAND_RATIO * max(lengths[i], lengths[j])))
            if radius_was_widened(lengths[i], lengths[j], BAND_RATIO):
                widened += 1
            cells = window_cell_count(lengths[i], lengths[j], BAND_RATIO)
            cell_ratios.append(cells / (lengths[i] * lengths[j]))
            started = time.perf_counter()
            full = dtw_distance(series[i], series[j], band_ratio=None)
            full_seconds += time.perf_counter() - started
            started = time.perf_counter()
            band = dtw_distance_band(series[i], series[j], BAND_RATIO)
            band_seconds += time.perf_counter() - started
            diff = abs(full - band)
            max_abs = max(max_abs, diff)
            if diff > 1e-6:
                mismatch += 1
    same_table = mismatch == 0
    lines = [
        "12 类真实曲线：Data_example/rtt_130ms_bdw_10000Kbps，每类取中间一条重复实验。",
        "曲线是排队时延，截到 10 秒、最多 512 点，并按列做了 MinMax，与 problem.py 进 DTW 前的处理一致。",
        f"长度: {dict(zip(names, lengths))}",
        f"对数: {pair_count}",
        f"python 整表耗时 {full_seconds:.2f} s，band 耗时 {band_seconds:.2f} s",
        f"平均格子比例 {float(np.mean(cell_ratios)):.3f}，最小 {float(np.min(cell_ratios)):.3f}",
        f"半径被 |n-m| 撑大的对数: {widened}/{pair_count}",
        f"band 与 python 最大绝对差 {max_abs:.6e}，不一致对数 {mismatch}",
        "耗时结论只比较上面的 python 与 band。二者都是 Python，original 的 C 库不参与。",
    ]
    if same_table:
        lines.append("距离表数值没有改变。可以用已训练策略做识别率对照。")
    else:
        lines.append("距离表数值已经改变。先不要用已训练策略做识别率对照，profile 选择会跟着变。")
    print("\n".join(lines))
    return lines, same_table


def main():
    lines = check_library()
    lines.append("")
    real_lines, same_table = check_real_curves()
    lines.extend(real_lines)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "band_check_12ccs.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\n结果已写入 {out_path}")
    if not same_table:
        print("识别率对照未放行。")


if __name__ == "__main__":
    main()
