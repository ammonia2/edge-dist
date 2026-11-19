import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "results")
ONNX_DIR = os.path.join(ROOT, "onnx_models")
for _d in (CKPT_DIR, RESULTS_DIR, ONNX_DIR):
    os.makedirs(_d, exist_ok=True)

# Training runs on CUDA when available (mixed precision + channels_last), else CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
if USE_AMP:
    torch.backends.cudnn.benchmark = True


def autocast():
    return torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP)


def grad_scaler():
    return torch.amp.GradScaler(DEVICE.type, enabled=USE_AMP)


def to_device(model):
    """Move a model to DEVICE (channels_last on GPU for faster convs)."""
    return model.to(DEVICE, memory_format=torch.channels_last) if USE_AMP else model.to(DEVICE)


def to_input(x):
    x = x.to(DEVICE, non_blocking=True)
    return x.contiguous(memory_format=torch.channels_last) if USE_AMP else x


def seed_all(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def set_threads(n=None):
    torch.set_num_threads(n or os.cpu_count())


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def logits_of(out):
    """Models may return (logits, features); return logits only."""
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def evaluate(model, loader, max_batches=None):
    model.eval()
    dev = next(model.parameters()).device
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(dev), y.to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            pred = logits_of(model(x)).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def make_optimizer(model, lr, epochs, steps_per_epoch, wd=5e-4, params=None):
    params = params if params is not None else [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, epochs * steps_per_epoch),
                                                pct_start=0.15)
    return opt, sched


def train_supervised(model, train_loader, test_loader, epochs, lr, log_name, label_smoothing=0.0,
                     params=None, eval_batches=None):
    """Plain cross-entropy training (teacher, baseline students, NAS candidates)."""
    to_device(model)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt, sched = make_optimizer(model, lr, epochs, len(train_loader), params=params)
    scaler = grad_scaler()
    history = []
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = to_input(x), y.to(DEVICE)
            with autocast():
                loss = crit(logits_of(model(x)).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * y.numel(); n += y.numel()
        acc = evaluate(model, test_loader, eval_batches) if test_loader is not None else float("nan")
        rec = {"epoch": ep + 1, "train_loss": tot / n, "test_acc": acc, "epoch_time_s": time.time() - t0}
        history.append(rec)
        if log_name:
            print(f"[{log_name}] epoch {ep + 1}/{epochs} loss={rec['train_loss']:.4f} "
                  f"acc={acc:.2f}% ({rec['epoch_time_s']:.0f}s)", flush=True)
    return history


def save_checkpoint(model, path, **meta):
    torch.save({"state_dict": model.state_dict(), **meta}, path)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
