"""按类别编号填写 DTW 距离表。

每个环境、每个类别可以有多条曲线。矩阵第 label 行用的是该类别曲线列表的中间一条，
下标与分类器的 alo_dic 编号一致，而不是曲线在列表里的先后位置。
"""

import numpy as np


def as_int(value):
    return int(np.asarray(value).reshape(-1)[0])


def distance_matrix_by_label(traces, labels, envs, num_alos, distance_fn):
    """返回形状 (环境数, num_alos, num_alos) 的距离表。环境编号从 1 开始。"""
    if num_alos < 1:
        raise ValueError("矩阵边长需要至少为 1")
    buckets = {}
    for index in range(len(traces)):
        env_id = as_int(envs[index])
        label = as_int(labels[index])
        if label < 0 or label >= num_alos:
            raise ValueError(f"类别编号 {label} 超出矩阵边长 {num_alos}")
        if env_id < 1:
            raise ValueError(f"环境编号 {env_id} 需要从 1 开始")
        buckets.setdefault(env_id, {}).setdefault(label, []).append(traces[index])
    if not buckets:
        raise ValueError("没有曲线可计算 DTW")

    max_env = max(buckets)
    missing_envs = [env_id for env_id in range(1, max_env + 1) if env_id not in buckets]
    if missing_envs:
        raise ValueError(f"环境编号不连续，缺少 {missing_envs}")

    representatives = {}
    for env_id, by_label in buckets.items():
        missing = [label for label in range(num_alos) if label not in by_label]
        if missing:
            raise ValueError(f"环境 {env_id} 缺少类别 {missing}")
        representatives[env_id] = {}
        for label in range(num_alos):
            curves = by_label[label]
            chosen = np.asarray(curves[len(curves) // 2], dtype=np.float64)
            if chosen.ndim != 2 or chosen.shape[1] < 2:
                raise ValueError("曲线需要至少两列，距离使用第 2 列")
            representatives[env_id][label] = chosen[:, 1]

    matrix = np.zeros((max_env, num_alos, num_alos))
    for env_id, series_by_label in representatives.items():
        row = env_id - 1
        for i in range(num_alos):
            for j in range(i, num_alos):
                distance = float(distance_fn(series_by_label[i], series_by_label[j]))
                matrix[row, i, j] = distance
                matrix[row, j, i] = distance
    return matrix


def write_labeled_matrix(path, matrix):
    """写出上三角。环境编号和类别编号都是矩阵下标对应的原始编号。"""
    with open(path, "w", encoding="utf-8") as handle:
        num_envs, num_alos, _ = matrix.shape
        for env_index in range(num_envs):
            env_id = env_index + 1
            for i in range(num_alos):
                for j in range(i, num_alos):
                    handle.write(f"{env_id}\t{i}\t{j}\t{matrix[env_index, i, j]}\n")
