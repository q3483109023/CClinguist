"""对序列做归一化或差分之后再算 DTW。

排队时延曲线的幅度会随带宽变化，整体抬高一点也会把欧氏 DTW 带偏。
z-score 去掉均值和幅度；差分（导数 DTW）去掉整体抬高，更看转折形状。
两者都可以再套弯曲窗口。
"""

import numpy as np

from dtw_original import dtw_distance


def zscore(series):
    series = np.asarray(series, dtype=np.float64).ravel()
    std = series.std()
    if std < 1e-8:
        return np.zeros_like(series)
    return (series - series.mean()) / std


def derivative(series):
    series = np.asarray(series, dtype=np.float64).ravel()
    if series.size < 2:
        return series.copy()
    return np.diff(series)


def dtw_zscore(series_a, series_b, band_ratio=None):
    return dtw_distance(zscore(series_a), zscore(series_b), band_ratio=band_ratio)


def dtw_derivative(series_a, series_b, band_ratio=None):
    return dtw_distance(derivative(series_a), derivative(series_b), band_ratio=band_ratio)


def dtw_zscore_band(series_a, series_b, band_ratio=0.1):
    """先 z-score，再在弯曲窗口里算 DTW。速度和抗幅度变化放在同一次比较里。"""
    return dtw_zscore(series_a, series_b, band_ratio=band_ratio)
