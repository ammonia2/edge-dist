"""KD ablation pilot behind Table 3b of the report: SqueezeNet, 10k CIFAR-100 images, 3 epochs,
same seed and data order, comparing the hard-label baseline with KD at beta=1 and beta=0.

    python scripts/kd_pilot.py            # needs checkpoints/teacher_resnet50.pt; ~30 min on a 4-core CPU
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgekd.data import get_loaders  # noqa: E402
from edgekd.kd import train_kd  # noqa: E402
from edgekd.models import build_student  # noqa: E402
from edgekd.utils import seed_all, set_threads, train_supervised  # noqa: E402
from train_students import load_teacher  # noqa: E402

CONFIGS = [("baseline", None), ("KD T=4 alpha=0.7 beta=1", (4, 0.7, 1.0)), ("KD T=4 alpha=0.7 beta=0", (4, 0.7, 0.0))]


def main():
    set_threads(None)
    teacher = load_teacher()
    for name, k in CONFIGS:
        seed_all(0)
        train, test = get_loaders("cifar100", 128, train_subset=10000, test_subset=2000)
        model, t0 = build_student("squeezenet", 100), time.time()
        if k is None:
            hist = train_supervised(model, train, test, 3, 0.1, name)
        else:
            hist = train_kd(model, teacher, train, test, 3, 0.1, temperature=k[0], alpha=k[1], beta=k[2],
                            log_name=name)
        print(f"RESULT {name}: {hist[-1]['test_acc']:.2f}% ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
