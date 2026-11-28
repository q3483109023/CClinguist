"""与 dtw 库默认步态一致的 DTW，供 python 整表和 band 窗口共用。

dtw 库默认 step_pattern 是 symmetric2：斜向一步把当前点的代价算两次，
横着或竖着走只算一次。alignment.distance 是未归一化的累计代价。
这里再除以 max(n, m)，与 dtw_similarity_original 相同。
python 和 band 都走这份 Python 动态规划，只用来说明窗口少算了格子。
original 与 cband 都调用 dtw 库；库有编译扩展时是 C。判断窗口是否加速只比这两种。
"""

import numpy as np

INF = np.inf


def sakoe_chiba_radius(length_a, length_b, band_ratio):
    """返回窗口半径。band_ratio 为 None 或 >= 1 时表示整张表。"""
    if band_ratio is None or band_ratio >= 1:
        return None
    radius = int(np.ceil(float(band_ratio) * max(length_a, length_b)))
    return max(radius, abs(length_a - length_b), 1)


def radius_was_widened(length_a, length_b, band_ratio):
    """|n-m| 或最小半径 1 把 10% 半径撑大时返回 True。"""
    if band_ratio is None or band_ratio >= 1:
        return False
    nominal = int(np.ceil(float(band_ratio) * max(length_a, length_b)))
    return sakoe_chiba_radius(length_a, length_b, band_ratio) > nominal


def window_cell_count(length_a, length_b, band_ratio):
    """与 dtw_distance 实际填写的格子数一致。"""
    radius = sakoe_chiba_radius(length_a, length_b, band_ratio)
    if radius is None:
        return length_a * length_b
    cells = min(length_b, radius + 1)
    for i in range(1, length_a):
        j_start = max(0, i - radius)
        j_end = min(length_b - 1, i + radius)
        cells += j_end - j_start + 1
    return cells


def dtw_distance(series_a, series_b, band_ratio=None):
    """返回归一化 DTW 距离。band_ratio 为 None 时计算完整矩阵。

    band_ratio 取 (0, 1) 时只计算 Sakoe-Chiba 窗口内的格子。半径约为
    band_ratio * max(len(a), len(b))，并且至少为 |len(a)-len(b)|。
    """
    series_a = np.asarray(series_a, dtype=np.float64).ravel()
    series_b = np.asarray(series_b, dtype=np.float64).ravel()
    length_a = series_a.size
    length_b = series_b.size
    if length_a == 0 or length_b == 0:
        raise ValueError("序列不能为空")

    radius = sakoe_chiba_radius(length_a, length_b, band_ratio)

    previous = np.full(length_b, INF)
    previous[0] = abs(series_a[0] - series_b[0])
    first_row_end = length_b if radius is None else min(length_b, radius + 1)
    for j in range(1, first_row_end):
        previous[j] = previous[j - 1] + abs(series_a[0] - series_b[j])

    for i in range(1, length_a):
        current = np.full(length_b, INF)
        if radius is None:
            j_start, j_end = 0, length_b - 1
        else:
            j_start = max(0, i - radius)
            j_end = min(length_b - 1, i + radius)
        value_i = series_a[i]
        for j in range(j_start, j_end + 1):
            cost = abs(value_i - series_b[j])
            vertical = previous[j] + cost
            if j > 0:
                horizontal = current[j - 1] + cost
                diagonal = previous[j - 1] + 2.0 * cost
                current[j] = min(vertical, horizontal, diagonal)
            else:
                current[j] = vertical
        previous = current

    if not np.isfinite(previous[-1]):
        raise RuntimeError("窗口太窄，DTW 路径没有到达终点")
    return float(previous[-1] / max(length_a, length_b))


def comparison_notice(mode):
    """耗时和识别率对照各自该看什么。"""
    lines = [
        "判断窗口是否加速，只比较 DTW_MODE=original 与 DTW_MODE=cband，二者都是 dtw 库。",
        "python 与 band 是同一份 Python 动态规划，只说明窗口少算了格子。",
    ]
    if mode in ("band", "cband"):
        lines.append(
            "10% 窗口会改变一部分距离，已训练策略选出的 profile 可能变化。"
        )
    return "\n".join(lines)
