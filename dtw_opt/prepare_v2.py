"""Train-only DTW-v2 benchmark. The only changed factor is the window."""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import median
import numpy as np
from labeled_matrix import as_int, distance_matrix_by_label, write_labeled_matrix
from dtw_cband import dtw_distance_cband
from dtw_band import dtw_distance_band
from dtw_original import dtw_distance

def main(a):
    k = a.classes
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / f"profile_generator_{k}ccs"))
    from dtw_cal import dtw_similarity_original
    if k == 12:
        from problem import Dataset
    else:
        from problem_newcca import Dataset
    train_envs, test_envs = os.listdir(a.train), os.listdir(a.test)
    assert len(train_envs) == (43 if k == 12 else 16)
    assert len(test_envs) == (32 if k == 12 else 16)
    assert set(test_envs) <= set(train_envs)
    assert all((a.train / x).is_dir() for x in train_envs)
    assert all((a.test / x).is_dir() for x in test_envs)
    ds = Dataset(36, str(a.train), None, [0])
    kwargs = {} if k == 12 else dict(num_cluster=k, only_newcca=False)
    ds.make_dataset_1209(36, train=True, train_data_num=7, **kwargs)
    assert ds.input_data_num == 7
    traces, labels, envs = ds.x_trace, ds.y, ds.x_env_id
    assert len(traces) == len(labels) == len(envs) == len(train_envs)*k*7
    counts = Counter((as_int(e), as_int(y)) for e,y in zip(envs,labels))
    assert counts == Counter({(e,y):7 for e in range(1,len(train_envs)+1) for y in range(k)})
    assert os.listdir(a.train) == train_envs
    assert os.listdir(a.test) == test_envs
    for t in traces:
        t = np.asarray(t)
        assert t.ndim == 2 and len(t) and np.isfinite(t).all()
    print("[DTW-V2] Training layout PASS", len(traces), flush=True)

    # Small deterministic kernel checks: Python full vs library full,
    # Python band vs library band; unequal lengths included.
    rng = np.random.default_rng(3402)
    for n,m in [(8,8),(20,12),(30,30),(41,39),(12,20)]:
        x,y = rng.normal(size=n),rng.normal(size=m)
        full, band = dtw_similarity_original(x,y),dtw_distance_cband(x,y)
        assert np.isclose(full,dtw_distance(x,y),atol=1e-8,rtol=1e-8)
        assert np.isclose(band,dtw_distance_band(x,y),atol=1e-8,rtol=1e-8)
        assert band >= full - 1e-8
    print("[DTW-V2] Four kernels PASS",flush=True)
    a.out.mkdir(parents=True, exist_ok=False)
    funcs = {"original":dtw_similarity_original,"cband":dtw_distance_cband}
    matrices = {}
    timings = {m:[] for m in funcs}
    for r in range(3):
        order = list(funcs) if r%2==0 else list(reversed(funcs))
        for mode in order:
            start,cpu = time.perf_counter(),time.process_time()
            mat = distance_matrix_by_label(traces,labels,envs,k,funcs[mode])
            elapsed,cpus = time.perf_counter()-start,time.process_time()-cpu
            timings[mode].append({"wall_s":elapsed,"cpu_s":cpus})
            assert np.isfinite(mat).all() and (mat >= 0).all()
            assert np.allclose(mat,mat.transpose(0,2,1))
            assert np.allclose(np.diagonal(mat,axis1=1,axis2=2),0)
            if mode in matrices:
                assert np.array_equal(mat,matrices[mode])
            matrices[mode] = mat
            print(f"[DTW-V2] {mode} repeat={r+1} wall={elapsed:.3f}s cpu={cpus:.3f}s",flush=True)

    off = np.triu_indices(k,1)
    d = matrices["cband"] - matrices["original"]
    delta = d[:,off[0],off[1]]
    assert (delta >= -1e-8).all()
    ids = [train_envs.index(x) for x in test_envs]
    classes = ["htcp","bbr","vegas","westwood","scalable","highspeed","veno","reno","yeah","illinois","bic","cubic"]
    if k == 15:
        classes += ["pccLatency","astraea","pccLoss"]
    for mode,mat in matrices.items():
        out = a.out / mode
        out.mkdir()
        write_labeled_matrix(out / "train_dtw.txt",mat)
        write_labeled_matrix(out / "test_dtw.txt",mat[ids])
        (out / "manifest.json").write_text(json.dumps({
            "train_envs":train_envs,"test_envs":test_envs,"class_names_by_id":classes,
            "samples_per_class":7,"aggregation":"middle trace among seven duration-ranked training traces per class",
            "train_path":str(a.train.resolve()),"test_path":str(a.test.resolve()),
            "test_traces_used":False,"dtw_mode":mode,"band_ratio":0.1 if mode=="cband" else None
        },indent=2)+"\n")
    metrics={"classes":k,"pairs":int(delta.size),
             "equal_pairs":int(np.isclose(delta,0,atol=1e-8,rtol=0).sum()),
             "mean_distance_increase":float(delta.mean()),"max_distance_increase":float(delta.max()),
             "timings":timings,
             "median_wall_s":{m:median(x["wall_s"] for x in t) for m,t in timings.items()},
             "median_cpu_s":{m:median(x["cpu_s"] for x in t) for m,t in timings.items()}}
    metrics["wall_speedup"]=metrics["median_wall_s"]["original"]/metrics["median_wall_s"]["cband"]
    metrics["cpu_speedup"]=metrics["median_cpu_s"]["original"]/metrics["median_cpu_s"]["cband"]
    (a.out/"dtw_metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print("[DTW-V2] FINAL",json.dumps(metrics),flush=True)
    print("[DTW-V2] ALL PASS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--classes",type=int,choices=[12,15],required=True)
    for name in ("train","test","out"):
        p.add_argument("--"+name,type=Path,required=True)
    main(p.parse_args())
