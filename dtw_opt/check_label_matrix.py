"""检查距离表下标用的是类别编号，而不是曲线在列表中的位置。"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dtw_original import dtw_distance
from labeled_matrix import distance_matrix_by_label


def curve(value):
    return np.array([[0.0, value], [1.0, value], [2.0, value], [3.0, value]], dtype=np.float64)


def full_distance(series_a, series_b):
    return dtw_distance(series_a, series_b, band_ratio=None)


def main():
    # 先放 7 条 bbr（标签 1），再放 1 条 htcp（标签 0）。
    # 7 条的中间一条是常数 10。选错曲线时，距离不会是 17.5。
    bbr_values = [0, 1, 2, 10, 4, 5, 6]
    traces = [curve(value) for value in bbr_values]
    labels = [[1] for _ in bbr_values]
    envs = [[1] for _ in bbr_values]
    traces.append(curve(0))
    labels.append([0])
    envs.append([1])

    matrix = distance_matrix_by_label(traces, labels, envs, 2, full_distance)
    got = float(matrix[0, 0, 1])
    # 等长常数曲线，symmetric2：代价 c + 2*c*(n-1)，再除以 n。n=4, c=10 时是 17.5。
    # 若误用列表里第一条 bbr（常数 0），距离应是常数差 0 的 0，而不是 17.5。
    expected = 10.0 * (2 * 4 - 1) / 4
    if abs(got - expected) > 1e-6:
        raise SystemExit(f"matrix[0, 0, 1]={got}，应是 htcp 与中间一条 bbr 的距离 {expected}")

    try:
        distance_matrix_by_label(traces[:7], labels[:7], envs[:7], 2, full_distance)
    except ValueError as exc:
        if "缺少类别" not in str(exc):
            raise
    else:
        raise SystemExit("环境里没有 htcp 时应当报错")

    print(f"label matrix check passed: matrix[0, 0, 1]={got}")


if __name__ == "__main__":
    main()
