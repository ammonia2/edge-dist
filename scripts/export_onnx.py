"""Export every trained student (KD + baseline) and the NAS model to ONNX and
verify ONNX Runtime outputs against PyTorch.

    python scripts/export_onnx.py [--models mobilenetv2_kd nas_best ...] [--include-teacher]
"""
import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgekd.inference import MODEL_REGISTRY, load_torch_model  # noqa: E402
from edgekd.nas.hardware import export_onnx  # noqa: E402
from edgekd.utils import CKPT_DIR, ONNX_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--include-teacher", action="store_true")
    args = ap.parse_args()
    names = args.models or [n for n in MODEL_REGISTRY if n != "teacher_resnet50" or args.include_teacher]
    for name in names:
        if not os.path.exists(os.path.join(CKPT_DIR, MODEL_REGISTRY[name]["ckpt"])):
            print(f"skip {name}: no checkpoint"); continue
        model, res = load_torch_model(name)
        path = os.path.join(ONNX_DIR, f"{name}.onnx")
        export_onnx(model, path, res)
        onnx.checker.check_model(onnx.load(path))
        x = torch.randn(4, 3, res, res)
        with torch.no_grad():
            ref = model(x).numpy()
        out = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
        diff = float(np.abs(ref - out).max())
        print(f"{name:28s} -> {path} ({os.path.getsize(path) / 2**20:.2f} MB), max |torch-ort| = {diff:.2e}")
        assert diff < 1e-3, "ONNX output mismatch"


if __name__ == "__main__":
    main()
