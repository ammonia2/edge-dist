"""Automated benchmark of all trained models -> results/benchmark.csv

Per model: parameters, ONNX size, test accuracy (ONNX Runtime), batch-1 CPU
latency for PyTorch and ONNX Runtime, and ONNX Runtime throughput (batch 64).

    python scripts/benchmark.py [--runs 50]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgekd.data import load_raw  # noqa: E402
from edgekd.inference import MODEL_REGISTRY, Predictor  # noqa: E402
from edgekd.nas.hardware import profile_onnx  # noqa: E402
from edgekd.utils import CKPT_DIR, ONNX_DIR, RESULTS_DIR, count_params, load_json  # noqa: E402


def latency_ms(fn, runs, warmup=10):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1000)
    return float(np.median(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rows = []
    data = {d: load_raw(d, train=False) for d in ("cifar100", "cifar10")}
    teacher_acc = None
    tp = os.path.join(CKPT_DIR, "teacher_resnet50.pt")
    if os.path.exists(tp):
        teacher_acc = torch.load(tp, map_location="cpu", weights_only=False)["test_acc"]

    for name in MODEL_REGISTRY:
        if name == "teacher_resnet50" or not os.path.exists(os.path.join(ONNX_DIR, f"{name}.onnx")):
            continue
        ort_p = Predictor(name, "onnx", threads=args.threads)
        pt_p = Predictor(name, "torch", threads=args.threads)
        x, y = data[ort_p.dataset]
        correct = sum((ort_p.predict_batch(x[s:s + 500]) == y[s:s + 500].numpy()).sum()
                      for s in range(0, len(x), 500))
        one = ort_p.preprocess(x[:1]); batch = ort_p.preprocess(x[:64])
        ort_lat = latency_ms(lambda: ort_p.logits(one), args.runs)
        pt_lat = latency_ms(lambda: pt_p.logits(one), args.runs)
        thr = 64 / (latency_ms(lambda: ort_p.logits(batch), 10, 2) / 1000)
        mem = profile_onnx(os.path.join(ONNX_DIR, f"{name}.onnx"), ort_p.resolution, threads=args.threads,
                           runs=10)["peak_mem_mb"]
        mode = "nas" if name == "nas_best" else name.rsplit("_", 1)[1]
        rows.append({"model": name, "dataset": ort_p.dataset, "training": mode,
                     "params_M": count_params(pt_p.model) / 1e6,
                     "onnx_size_MB": os.path.getsize(os.path.join(ONNX_DIR, f"{name}.onnx")) / 2**20,
                     "test_acc_%": 100 * correct / len(x), "torch_latency_ms": pt_lat, "ort_latency_ms": ort_lat,
                     "ort_throughput_img_s": thr, "ort_peak_mem_MB": mem})
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)

    df = pd.DataFrame(rows)
    # KD gain table (CIFAR-100 students)
    gains = []
    for s in ("mobilenetv2", "squeezenet", "efficientnet_lite0"):
        b, k = df[df.model == f"{s}_baseline"], df[df.model == f"{s}_kd"]
        if len(b) and len(k):
            gains.append({"student": s, "baseline_acc_%": b["test_acc_%"].item(), "kd_acc_%": k["test_acc_%"].item(),
                          "kd_gain_pts": k["test_acc_%"].item() - b["test_acc_%"].item(),
                          "teacher_acc_%": teacher_acc})
    df.to_csv(os.path.join(RESULTS_DIR, "benchmark.csv"), index=False)
    pd.DataFrame(gains).to_csv(os.path.join(RESULTS_DIR, "kd_vs_baseline.csv"), index=False)
    print("\n" + df.round(3).to_string(index=False))
    if gains:
        print("\n" + pd.DataFrame(gains).round(2).to_string(index=False))
    nb = os.path.join(RESULTS_DIR, "nas_best_arch.json")
    if os.path.exists(nb):
        print("\nNAS constraints:", load_json(nb)["constraints"])


if __name__ == "__main__":
    main()
