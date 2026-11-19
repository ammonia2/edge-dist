"""Training budgets.

QUICK is sized for a 4-core laptop CPU (whole pipeline ~1h). FULL is a
conventional schedule for a GPU machine. Select with `--profile full` on any script.
"""

PROFILES = {
    "quick": {
        "threads": 4,
        "batch_size": 128,
        # teacher: ImageNet-pretrained ResNet-50 fine-tuned on CIFAR-100
        "teacher": {"epochs": 2, "lr": 0.03, "train_subset": 25000, "input_res": 64, "freeze": True},
        # students (baseline and KD use identical budgets for a fair comparison)
        "student": {"epochs": 4, "lr": 0.1, "train_subset": 25000},
        # KD hyper-parameters
        "kd": {"temperature": 4.0, "alpha": 0.7, "beta": 1.0},
        # NAS (CIFAR-10)
        "nas": {"population": 8, "generations": 4, "proxy_epochs": 2, "proxy_train": 6000, "val": 2000,
                "lr": 0.05, "final_epochs": 5, "final_train_subset": None},
    },
    # Google Colab T4 GPU: whole pipeline ~30-35 min. Full CIFAR training sets, fine-tuned teacher at 96 px.
    "colab": {
        "threads": None,
        "batch_size": 256,
        "teacher": {"epochs": 3, "lr": 0.05, "train_subset": None, "input_res": 96, "freeze": False},
        "student": {"epochs": 10, "lr": 0.15, "train_subset": None},
        "kd": {"temperature": 4.0, "alpha": 0.7, "beta": 1.0},
        "nas": {"population": 8, "generations": 5, "proxy_epochs": 3, "proxy_train": 10000, "val": 5000,
                "lr": 0.1, "final_epochs": 15, "final_train_subset": None},
    },
    # few-minute end-to-end check that every stage runs (numbers are meaningless)
    "smoke": {
        "threads": None,
        "batch_size": 128,
        "teacher": {"epochs": 1, "lr": 0.03, "train_subset": 512, "input_res": 64, "freeze": True},
        "student": {"epochs": 1, "lr": 0.1, "train_subset": 512},
        "kd": {"temperature": 4.0, "alpha": 0.7, "beta": 1.0},
        "nas": {"population": 2, "generations": 1, "proxy_epochs": 1, "proxy_train": 512, "val": 256,
                "lr": 0.05, "final_epochs": 1, "final_train_subset": 1000},
    },
    "full": {
        "threads": None,
        "batch_size": 128,
        "teacher": {"epochs": 30, "lr": 0.05, "train_subset": None, "input_res": 224, "freeze": False},
        "student": {"epochs": 200, "lr": 0.1, "train_subset": None},
        "kd": {"temperature": 4.0, "alpha": 0.9, "beta": 1.0},
        "nas": {"population": 20, "generations": 15, "proxy_epochs": 5, "proxy_train": 20000, "val": 5000,
                "lr": 0.05, "final_epochs": 100, "final_train_subset": None},
    },
}

# Deployment constraints for hardware-aware NAS (batch-1 CPU inference, ONNX Runtime)
NAS_CONSTRAINTS = {"max_latency_ms": 3.0, "max_size_mb": 2.0, "max_peak_mem_mb": 25.0}


def get_profile(name):
    return PROFILES[name]
