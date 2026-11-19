"""Train the best discovered architecture from scratch on CIFAR-10.

    python scripts/nas_train_best.py [--profile quick|full] [--arch results/nas_best_arch.json]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgekd.config import get_profile  # noqa: E402
from edgekd.data import get_loaders  # noqa: E402
from edgekd.nas import build_model  # noqa: E402
from edgekd.utils import (CKPT_DIR, RESULTS_DIR, count_params, load_json, save_checkpoint,  # noqa: E402
                          save_json, seed_all, set_threads, train_supervised)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="quick")
    ap.add_argument("--arch", default=os.path.join(RESULTS_DIR, "nas_best_arch.json"))
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    P = get_profile(args.profile); N = P["nas"]
    seed_all(0); set_threads(P["threads"])
    arch = load_json(args.arch)["arch"]
    model = build_model(arch, 10)  # fresh weights: trained from scratch
    train, test = get_loaders("cifar10", P["batch_size"], train_subset=N["final_train_subset"])
    epochs = args.epochs or N["final_epochs"]
    print(f"NAS best: {count_params(model) / 1e3:.1f}K params, res {arch['resolution']}, {epochs} epochs", flush=True)
    hist = train_supervised(model, train, test, epochs, N["lr"] * 2, "nas_best")
    save_checkpoint(model, os.path.join(CKPT_DIR, "nas_best.pt"), arch=arch, test_acc=hist[-1]["test_acc"])
    save_json({"arch": arch, "params": count_params(model), "history": hist},
              os.path.join(RESULTS_DIR, "nas_best_train_history.json"))


if __name__ == "__main__":
    main()
