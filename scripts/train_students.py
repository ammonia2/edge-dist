"""Train the three student networks on CIFAR-100, either as plain baselines
(hard labels only) or with knowledge distillation from the ResNet-50 teacher.

    python scripts/train_students.py --mode baseline             # all three, no KD
    python scripts/train_students.py --mode kd                   # all three, with KD
    python scripts/train_students.py --mode kd --models squeezenet --alpha 0.9 --beta 0
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgekd.config import get_profile  # noqa: E402
from edgekd.data import get_loaders  # noqa: E402
from edgekd.kd import train_kd  # noqa: E402
from edgekd.models import STUDENTS, ResNet50Teacher, build_student  # noqa: E402
from edgekd.utils import (CKPT_DIR, RESULTS_DIR, count_params, save_checkpoint, save_json,  # noqa: E402
                          seed_all, set_threads, train_supervised)


def load_teacher():
    ck = torch.load(os.path.join(CKPT_DIR, "teacher_resnet50.pt"), map_location="cpu")
    t = ResNet50Teacher(100, input_res=ck["input_res"], pretrained=False)
    t.load_state_dict(ck["state_dict"])
    return t.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "kd"], required=True)
    ap.add_argument("--models", nargs="+", default=list(STUDENTS))
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--beta", type=float)
    args = ap.parse_args()
    P = get_profile(args.profile); S = P["student"]; K = dict(P["kd"])
    for k in ("temperature", "alpha", "beta"):
        if getattr(args, k) is not None:
            K[k] = getattr(args, k)
    epochs = args.epochs or S["epochs"]
    set_threads(P["threads"])
    teacher = load_teacher() if args.mode == "kd" else None

    for name in args.models:
        seed_all(0)  # same init + data order for baseline and KD runs
        train, test = get_loaders("cifar100", P["batch_size"], train_subset=S["train_subset"])
        model = build_student(name, 100)
        tag = f"{name}_{args.mode}"
        print(f"== {tag}: {count_params(model) / 1e6:.2f}M params, {epochs} epochs", flush=True)
        if args.mode == "baseline":
            hist = train_supervised(model, train, test, epochs, S["lr"], tag)
        else:
            hist = train_kd(model, teacher, train, test, epochs, S["lr"], log_name=tag, **K)
        save_checkpoint(model, os.path.join(CKPT_DIR, f"{tag}.pt"), arch=name, mode=args.mode,
                        test_acc=hist[-1]["test_acc"], kd=K if args.mode == "kd" else None)
        save_json({"model": name, "mode": args.mode, "params": count_params(model),
                   "kd": K if args.mode == "kd" else None, "history": hist},
                  os.path.join(RESULTS_DIR, f"{tag}_history.json"))


if __name__ == "__main__":
    main()
