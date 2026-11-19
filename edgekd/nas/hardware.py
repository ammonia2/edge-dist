"""Hardware profiling for the hardware-aware search.

Each candidate is exported to ONNX and profiled on CPU with ONNX Runtime
(batch 1, single intra-op thread to mimic a small edge core):

* latency      - median wall-clock over N runs after warm-up
* storage size - size of the exported .onnx file
* peak memory  - peak RSS increase (session creation + inference) measured in a
                 fresh subprocess: the kernel's peak-RSS counter (VmHWM) is reset
                 via /proc/self/clear_refs after imports (Linux only; `None` where
                 unsupported)
"""
import copy
import json
import os
import subprocess
import sys
import tempfile

import torch

WORKER = os.path.abspath(__file__)


def export_onnx(model, path, resolution, opset=17):
    """Export an fp32 CPU copy of `model` (the original may live on the GPU)."""
    model = copy.deepcopy(model).cpu().float().eval()
    model = model.to(memory_format=torch.contiguous_format)
    dummy = torch.randn(1, 3, resolution, resolution)
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset,
                      dynamo=False)
    return path


def profile_onnx(path, resolution, threads=1, warmup=10, runs=50):
    """Run the profiling worker in a clean subprocess and return its metrics."""
    out = subprocess.run([sys.executable, WORKER, path, str(resolution), str(threads), str(warmup), str(runs)],
                         capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    m = json.loads(out.stdout.strip().splitlines()[-1])
    m["size_mb"] = os.path.getsize(path) / 2 ** 20
    return m


def profile_model(model, resolution, keep_path=None, **kw):
    """Export `model` to ONNX (temp file unless keep_path) and profile it."""
    if keep_path:
        export_onnx(model, keep_path, resolution)
        return profile_onnx(keep_path, resolution, **kw)
    with tempfile.TemporaryDirectory() as d:
        p = export_onnx(model, os.path.join(d, "m.onnx"), resolution)
        return profile_onnx(p, resolution, **kw)


def check_constraints(metrics, constraints):
    """Return a list of violated constraints (empty -> deployable)."""
    v = []
    if metrics["latency_ms"] > constraints["max_latency_ms"]:
        v.append(f"latency {metrics['latency_ms']:.2f}ms > {constraints['max_latency_ms']}ms")
    if metrics["size_mb"] > constraints["max_size_mb"]:
        v.append(f"size {metrics['size_mb']:.2f}MB > {constraints['max_size_mb']}MB")
    pm = metrics.get("peak_mem_mb")
    if pm is not None and pm > constraints["max_peak_mem_mb"]:
        v.append(f"peak mem {pm:.1f}MB > {constraints['max_peak_mem_mb']}MB")
    return v


def _worker(path, resolution, threads, warmup, runs):
    import time

    import numpy as np
    import onnxruntime as ort

    def _status_kb(field):
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field):
                    return int(line.split()[1])

    try:  # Linux: reset the peak-RSS (VmHWM) counter so it reflects only the model
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")

        def rss_peak_mb():
            return _status_kb("VmHWM") / 1024
        base = _status_kb("VmRSS") / 1024
    except OSError:  # other OSes: peak memory unsupported
        rss_peak_mb, base = None, None

    x = np.random.randn(1, 3, resolution, resolution).astype(np.float32)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    for _ in range(warmup):
        sess.run(None, {"input": x})
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t) * 1000)
    peak = (rss_peak_mb() - base) if rss_peak_mb else None
    print(json.dumps({"latency_ms": float(np.median(times)), "latency_p90_ms": float(np.percentile(times, 90)),
                      "peak_mem_mb": peak}))


if __name__ == "__main__":
    a = sys.argv[1:]
    _worker(a[0], int(a[1]), int(a[2]), int(a[3]), int(a[4]))
