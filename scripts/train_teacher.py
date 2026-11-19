"""Train (fine-tune) the ResNet-50 teacher on CIFAR-100.

    python scripts/train_teacher.py [--profile quick|full]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgekd.config import get_profile  # noqa: E402
from edgekd.data import get_loaders  # noqa: E402
from edgekd.models import ResNet50Teacher  # noqa: E402
from edgekd.utils import (CKPT_DIR, RESULTS_DIR, count_params, save_checkpoint, save_json,  # noqa: E402
                          seed_all, set_threads, train_supervised)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    P = get_profile(args.profile); T = P["teacher"]
    seed_all(0); set_threads(P["threads"])

    train, test = get_loaders("cifar100", P["batch_size"], train_subset=T["train_subset"])
    model = ResNet50Teacher(100, input_res=T["input_res"], pretrained=True)
    if T["freeze"]:
        model.freeze()
    print(f"ResNet-50 teacher: {count_params(model) / 1e6:.2f}M params, input {T['input_res']}px, "
          f"{len(train.x)} train images", flush=True)
    hist = train_supervised(model, train, test, args.epochs or T["epochs"], T["lr"], "teacher",
                            label_smoothing=0.1)
    save_checkpoint(model, os.path.join(CKPT_DIR, "teacher_resnet50.pt"), input_res=T["input_res"],
                    test_acc=hist[-1]["test_acc"])
    save_json({"model": "resnet50", "params": count_params(model), "history": hist},
              os.path.join(RESULTS_DIR, "teacher_history.json"))
    print(f"teacher final test acc {hist[-1]['test_acc']:.2f}%")


if __name__ == "__main__":
    main()
