from dtw import *
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import math
import re

from sklearn.preprocessing import MinMaxScaler

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'simulation_hk2jp_1201to1203')


def load_data_arrivetime(data):
    trace=data[0]
    label=data[1]
    env=data[2]
    dic_envs={}
    dic_alos={}
    envs=[]
    alos=[]
    for i in range(len(trace)):
        envname=env[i][0]
        if envname not in dic_envs:
            envs.append(envname)
            dic_envs[envname]=[]
        dic_envs[envname].append(trace[i])
        if label[i][0] not in dic_alos:
            alos.append(label[i][0])
            dic_alos[label[i][0]]=[]
        dic_alos[label[i][0]].append(trace[i])
    return dic_envs,dic_alos,envs,label
    
    
    

def load_data(path):
    dic_envs={}
    dic_alos={}
    envs = os.listdir(data_path)

    for env in envs:
        alo_path=os.path.join(data_path,env)
        alos=os.listdir(alo_path)

        for alo in alos:
            trace=np.loadtxt(os.path.join(alo_path,alo))
            if trace.shape[0]>10000:
                trace=trace[:10000,:]
            if env not in dic_envs:
                dic_envs[env]=[]
            dic_envs[env].append(trace)  
            if alo not in dic_alos:
                dic_alos[alo]=[]
            dic_alos[alo].append(trace) 
    return dic_envs,dic_alos,envs,alos  
            
import sys
import time
_opt_dir = os.path.abspath(os.path.join(base_dir, "..", "dtw_opt"))
if _opt_dir not in sys.path:
    sys.path.insert(0, _opt_dir)
from dtw_band import dtw_distance_band
from dtw_cband import dtw_distance_cband
from dtw_original import comparison_notice, dtw_distance
from labeled_matrix import as_int, distance_matrix_by_label, write_labeled_matrix

_MATRIX_FILES = {
    "original": "dtw_matrix_original.txt",
    "python": "dtw_matrix_python.txt",
    "band": "dtw_matrix_band.txt",
    "cband": "dtw_matrix_cband.txt",
}


def dtw_mode():
    """original 用 dtw 库整表，cband 用同一库的 10% 窗口，python 与 band 走 Python。未设置时为 band。"""
    mode = os.environ.get("DTW_MODE", "band").strip().lower()
    if mode not in _MATRIX_FILES:
        raise ValueError("DTW_MODE 只能是 original、python、band 或 cband")
    return mode


def dtw_similarity_original(data1, data2):
    """原论文的完整 DTW，保留不删。"""
    alignment = dtw(data1, data2, dist_method='euclidean', keep_internals=True)
    dis = alignment.distance / (max(len(data1), len(data2)))
    return dis


def dtw_similarity(data1, data2):
    mode = dtw_mode()
    if mode == "original":
        return dtw_similarity_original(data1, data2)
    if mode == "python":
        return dtw_distance(data1, data2, band_ratio=None)
    if mode == "cband":
        return dtw_distance_cband(data1, data2, band_ratio=0.1)
    return dtw_distance_band(data1, data2, band_ratio=0.1)

def resample(data1, data2, alpha=0.75):

    time1 = data1[:,0]
    time2 = data2[:,0]
    y1=data1[:,1]
    y2=data2[:,1]
    stride=min(np.min(time1[1:] - time1[:-1]),np.min(time2[1:] - time2[:-1]))
    stride = round(stride, 3)
    new_time1 = np.arange(0, np.max(time1), stride)    
    new_time2 = np.arange(0, np.max(time2), stride)
    new_y_1 = np.interp(new_time1, time1, y1)
    new_y_2 = np.interp(new_time2, time2, y2)
    return new_y_1, new_y_2
        

def _compute_dtw_from_directory():
    """目录里的曲线仍按读入顺序填表。run_newcca.py 不走这条路径。"""
    dic_envs, dic_alos, envs, alos = load_data(data_path)
    mode = dtw_mode()
    out_path = os.path.join(base_dir, _MATRIX_FILES[mode])
    print(f"DTW 模式: {mode}，距离表将写入 {out_path}")
    print(comparison_notice(mode))
    started = time.perf_counter()
    num_alos = len(dic_alos)
    num_envs = len(dic_envs)
    f = open(out_path, 'w')

    matrix_dtw = np.zeros((num_envs, num_alos, num_alos))
    for env in dic_envs:
        id_env = list(dic_envs.keys()).index(env)
        for i in range(len(dic_alos)):
            for j in range(i, len(dic_alos)):
                data1 = dic_envs[env][i]
                data2 = dic_envs[env][j]
                dis = dtw_similarity(data1[:, 1], data2[:, 1])
                matrix_dtw[id_env, i, j] = dis
                matrix_dtw[id_env, j, i] = dis
                f.write(f"{env}\t{i}\t{j}\t{dis}\n")
    f.close()
    print(f"DTW 距离表计算耗时 {time.perf_counter() - started:.2f} s")
    return matrix_dtw


def compute_dtw(data, num_cluster=12, data_path=None):
    if data_path is not None:
        return _compute_dtw_from_directory()

    traces, labels, envs = data
    for label in labels:
        if as_int(label) >= num_cluster:
            raise ValueError(f"类别编号 {as_int(label)} 超出矩阵边长 {num_cluster}")
    mode = dtw_mode()
    out_path = os.path.join(base_dir, _MATRIX_FILES[mode])
    print(f"DTW 模式: {mode}，距离表将写入 {out_path}")
    print(comparison_notice(mode))
    print(f"距离表按类别编号填写，边长 {num_cluster}，每个类别取中间那一条曲线")
    started = time.perf_counter()
    matrix_dtw = distance_matrix_by_label(traces, labels, envs, num_cluster, dtw_similarity)
    write_labeled_matrix(out_path, matrix_dtw)
    print(f"DTW 距离表计算耗时 {time.perf_counter() - started:.2f} s")
    return matrix_dtw
    
                
                
                
if __name__ == '__main__':
    compute_dtw(data_path)
    
            
