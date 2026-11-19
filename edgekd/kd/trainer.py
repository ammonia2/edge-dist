import time

import torch

from ..utils import DEVICE, autocast, evaluate, grad_scaler, make_optimizer, to_device, to_input
from .losses import DistillationLoss, FeatureDistiller


def train_kd(student, teacher, train_loader, test_loader, epochs, lr, temperature=4.0, alpha=0.7,
             beta=1.0, log_name="kd"):
    """Distil a frozen teacher into a student with logit + feature distillation."""
    to_device(teacher).eval()
    to_device(student)
    for p in teacher.parameters():
        p.requires_grad = False
    distiller = FeatureDistiller(student.feature_channels, teacher.feature_channels) if beta > 0 else None
    crit = DistillationLoss(temperature, alpha, beta, distiller).to(DEVICE)
    params = list(student.parameters()) + (list(distiller.parameters()) if distiller else [])
    opt, sched = make_optimizer(student, lr, epochs, len(train_loader), params=params)
    scaler = grad_scaler()
    history = []
    for ep in range(epochs):
        student.train(); crit.train()
        t0, n, sums = time.time(), 0, {"loss": 0.0, "hard": 0.0, "soft": 0.0, "feat": 0.0}
        for x, y in train_loader:
            x, y = to_input(x), y.to(DEVICE)
            with autocast():
                with torch.no_grad():
                    t_logits, t_feat = teacher(x, return_features=True)
                s_logits, s_feat = student(x, return_features=True)
            # losses in fp32 for numerical stability of the temperature-scaled softmax
            loss, parts = crit(s_logits.float(), t_logits.float(), y, s_feat.float(), t_feat.float())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            b = y.numel(); n += b
            sums["loss"] += loss.item() * b
            for k, v in parts.items():
                sums[k] += v * b
        acc = evaluate(student, test_loader)
        rec = {"epoch": ep + 1, **{f"train_{k}": v / n for k, v in sums.items()}, "test_acc": acc,
               "epoch_time_s": time.time() - t0}
        history.append(rec)
        print(f"[{log_name}] epoch {ep + 1}/{epochs} loss={rec['train_loss']:.4f} "
              f"(hard={rec['train_hard']:.3f} soft={rec['train_soft']:.3f} feat={rec['train_feat']:.3f}) "
              f"acc={acc:.2f}% ({rec['epoch_time_s']:.0f}s)", flush=True)
    return history
