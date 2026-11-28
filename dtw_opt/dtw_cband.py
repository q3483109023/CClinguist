"""用 dtw 库的 Sakoe-Chiba 窗口计算 10% 弯曲距离。

与 original 走同一个编译扩展。差别是传入 window_type="sakoechiba"，
半径与 Python band 的 sakoe_chiba_radius 相同。keep_internals 与
dtw_similarity_original 一致，耗时对照只比有没有这张窗口掩码。
"""

import numpy as np

from dtw_original import sakoe_chiba_radius


def dtw_distance_cband(series_a, series_b, band_ratio=0.1):
    from dtw import dtw

    if band_ratio is None or not 0 < band_ratio <= 1:
        raise ValueError("band_ratio 需要落在 (0, 1] 内")

    series_a = np.asarray(series_a, dtype=np.float64).ravel()
    series_b = np.asarray(series_b, dtype=np.float64).ravel()
    if series_a.size == 0 or series_b.size == 0:
        raise ValueError("序列不能为空")

    radius = sakoe_chiba_radius(series_a.size, series_b.size, band_ratio)
    alignment = dtw(
        series_a,
        series_b,
        dist_method="euclidean",
        keep_internals=True,
        window_type="sakoechiba",
        window_args={"window_size": radius},
    )
    return float(alignment.distance / max(series_a.size, series_b.size))
