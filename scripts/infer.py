"""Unified CPU inference CLI for all student models and the NAS model.

    # classify an image file with a distilled student via ONNX Runtime
    python scripts/infer.py --model mobilenetv2_kd --image path/to/img.png

    # evaluate test accuracy of one / all models (ONNX Runtime or PyTorch backend)
    python scripts/infer.py --model all --eval --backend onnx
    python scripts/infer.py --model nas_best --eval --backend torch

    # show predictions for a few CIFAR test images
    python scripts/infer.py --model squeezenet_kd --samples 5
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgekd.data import load_raw  # noqa: E402
from edgekd.inference import MODEL_REGISTRY, Predictor  # noqa: E402
from edgekd.utils import ONNX_DIR  # noqa: E402


def available(backend):
    names = [n for n in MODEL_REGISTRY if n != "teacher_resnet50"]
    if backend == "onnx":
        names = [n for n in names if os.path.exists(os.path.join(ONNX_DIR, f"{n}.onnx"))]
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", help=f"one of {sorted(MODEL_REGISTRY)} or 'all'")
    ap.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    ap.add_argument("--image")
    ap.add_argument("--eval", action="store_true", help="accuracy on the test set")
    ap.add_argument("--samples", type=int, default=0, help="print predictions for N test images")
    ap.add_argument("--threads", type=int)
    args = ap.parse_args()
    names = available(args.backend) if args.model == "all" else [args.model]

    for name in names:
        p = Predictor(name, args.backend, args.threads)
        print(f"== {name} [{args.backend}, {p.dataset}, input {p.resolution}px]")
        if args.image:
            for cls, prob in p.predict(args.image, topk=5):
                print(f"   {cls:15s} {prob:.3f}")
        if args.samples or args.eval:
            x, y = load_raw(p.dataset, train=False)
            if args.samples:
                pred = p.predict_batch(x[:args.samples])
                for i in range(args.samples):
                    print(f"   test[{i}] true={p.classes[y[i]]:12s} pred={p.classes[pred[i]]}")
            if args.eval:
                t0, correct = time.time(), 0
                for s in range(0, len(x), 500):
                    correct += (p.predict_batch(x[s:s + 500]) == y[s:s + 500].numpy()).sum()
                dt = time.time() - t0
                print(f"   test accuracy {100 * correct / len(x):.2f}%  ({len(x) / dt:.0f} img/s)")


if __name__ == "__main__":
    main()
