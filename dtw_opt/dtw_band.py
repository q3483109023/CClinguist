"""带 Sakoe-Chiba 弯曲窗口的 DTW。

只计算时间轴附近的一条带子，用来降低计算量。
band_ratio=0.1 表示窗口半径约为序列长度的 10%。
"""

from dtw_original import dtw_distance


def dtw_distance_band(series_a, series_b, band_ratio=0.1):
    if not 0 < band_ratio <= 1:
        raise ValueError("band_ratio 需要落在 (0, 1] 内")
    return dtw_distance(series_a, series_b, band_ratio=band_ratio)
