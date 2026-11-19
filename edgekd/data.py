"""Fast in-memory CIFAR loaders.

The whole dataset is held as a uint8 tensor and augmentation (random crop with
4px padding + horizontal flip) is applied batch-wise on tensors. This avoids the
per-image PIL overhead of torchvision transforms, which dominates on CPU.
"""
import hashlib
import os
import pickle
import tarfile
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

STATS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}
NUM_CLASSES = {"cifar100": 100, "cifar10": 10}


ARCHIVES = {  # name -> (url, md5, extracted dir)
    "cifar100": ("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
                 "eb9058c3a382ffc7106e4002c42a8d85", "cifar-100-python"),
    "cifar10": ("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                "c58f30108f718f92721af3b95e74349a", "cifar-10-batches-py"),
}
_REQUIRED = {
    "cifar100": ["train", "test"],
    "cifar10": [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"],
}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url, dst, timeout=30):
    """Stream url -> dst with a read timeout, progress output and HTTP Range resume of a partial file."""
    have = os.path.getsize(dst) if os.path.exists(dst) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # timeout also applies to every read()
        if have and r.status != 206:  # server ignored Range: start over
            have = 0
        total = have + int(r.headers.get("Content-Length", 0))
        done, last, t0 = have, 0.0, time.time()
        with open(dst, "ab" if have else "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 5:
                    last = time.time()
                    rate = (done - have) / max(last - t0, 1e-6) / 1e6
                    pct = f"{100 * done / total:5.1f}%" if total else ""
                    print(f"[data]   {done / 1e6:7.1f} / {total / 1e6:.1f} MB {pct}  {rate:.2f} MB/s", flush=True)
    if total and done < total:
        raise IOError(f"connection closed early ({done}/{total} bytes)")


def ensure_data(name, root=DATA_ROOT, retries=6):
    """Download + verify + extract a CIFAR archive if its files are missing. Raises loudly on failure."""
    url, md5, sub = ARCHIVES[name]
    base = os.path.join(root, sub)
    if all(os.path.exists(os.path.join(base, f)) for f in _REQUIRED[name]):
        return base
    os.makedirs(root, exist_ok=True)
    tgz = os.path.join(root, os.path.basename(url))
    for attempt in range(1, retries + 1):
        if os.path.exists(tgz) and _md5(tgz) == md5:
            break
        print(f"[data] downloading {name} (attempt {attempt}/{retries}) ...", flush=True)
        try:
            _download(url, tgz + ".part")  # keeps the .part on failure so the next attempt resumes
            os.replace(tgz + ".part", tgz)
        except Exception as e:  # network hiccup / stall: retry (resuming)
            print(f"[data] download failed: {e!r}", flush=True)
            time.sleep(5 * attempt)
        else:
            if _md5(tgz) != md5:  # corrupt: drop it so the next attempt starts clean
                print("[data] md5 mismatch, re-downloading", flush=True)
                os.remove(tgz)
    if not (os.path.exists(tgz) and _md5(tgz) == md5):
        raise RuntimeError(f"could not download a valid {name} archive from {url}")
    with tarfile.open(tgz, "r:gz") as t:
        t.extractall(root)
    missing = [f for f in _REQUIRED[name] if not os.path.exists(os.path.join(base, f))]
    if missing:
        raise RuntimeError(f"{name} extracted but files are missing: {missing}")
    print(f"[data] {name} ready at {base}", flush=True)
    return base


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_raw(name, train, root=DATA_ROOT):
    """Return (images uint8 [N,3,32,32], labels int64 [N])."""
    ensure_data(name, root)
    if name == "cifar100":
        d = _unpickle(os.path.join(root, "cifar-100-python", "train" if train else "test"))
        x, y = d["data"], d["fine_labels"]
    elif name == "cifar10":
        base = os.path.join(root, "cifar-10-batches-py")
        files = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        xs, ys = [], []
        for f in files:
            d = _unpickle(os.path.join(base, f))
            xs.append(d["data"]); ys += d["labels"]
        x, y = np.concatenate(xs), ys
    else:
        raise ValueError(name)
    x = torch.from_numpy(np.asarray(x, dtype=np.uint8).reshape(-1, 3, 32, 32))
    return x, torch.tensor(y, dtype=torch.long)


def normalize(x_uint8, name):
    mean, std = STATS[name]
    x = x_uint8.float().div_(255.0)
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x):
    """Random crop (pad 4) + horizontal flip on a float batch [B,3,H,W] (runs on x's device)."""
    b, _, h, w = x.shape
    d = x.device
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (b,), device=d)
    j = torch.randint(0, 9, (b,), device=d)
    # gather crops with unfold-free indexing
    rows = (i.view(b, 1) + torch.arange(h, device=d).view(1, h))  # [B,H]
    cols = (j.view(b, 1) + torch.arange(w, device=d).view(1, w))  # [B,W]
    xp = xp[torch.arange(b, device=d).view(b, 1, 1, 1), torch.arange(3, device=d).view(1, 3, 1, 1),
            rows.view(b, 1, h, 1), cols.view(b, 1, 1, w)]
    flip = torch.rand(b, device=d) < 0.5
    xp[flip] = xp[flip].flip(3)
    return xp


class TensorLoader:
    """Minimal iterable loader over in-memory tensors."""

    def __init__(self, x, y, name, batch_size=128, shuffle=False, train_aug=False, resolution=32, device=None):
        if device is not None:  # keep the whole (small) dataset resident on the GPU
            x, y = x.to(device), y.to(device)
        self.x, self.y, self.name = x, y, name
        self.bs, self.shuffle, self.aug, self.res = batch_size, shuffle, train_aug, resolution

    def __len__(self):
        return (len(self.x) + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.x)
        d = self.x.device
        idx = torch.randperm(n, device=d) if self.shuffle else torch.arange(n, device=d)
        for s in range(0, n, self.bs):
            b = idx[s:s + self.bs]
            xb = normalize(self.x[b], self.name)
            if self.aug:
                xb = augment(xb)
            if self.res != 32:
                xb = F.interpolate(xb, size=(self.res, self.res), mode="bilinear", align_corners=False)
            yield xb, self.y[b]


def get_loaders(name="cifar100", batch_size=128, train_subset=None, test_subset=None,
                val_split=0, resolution=32, seed=0, device="auto"):
    """Build train / (val) / test loaders.

    train_subset: use only this many (class-balanced-ish random) training images.
    val_split:    hold out this many training images as a validation set (used by NAS).
    device:       "auto" -> CUDA if available (data resident on GPU), else CPU.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else None
    g = torch.Generator().manual_seed(seed)
    xtr, ytr = load_raw(name, True)
    xte, yte = load_raw(name, False)
    perm = torch.randperm(len(xtr), generator=g)
    xtr, ytr = xtr[perm], ytr[perm]
    val = None
    if val_split:
        val = TensorLoader(xtr[:val_split], ytr[:val_split], name, 512, resolution=resolution, device=device)
        xtr, ytr = xtr[val_split:], ytr[val_split:]
    if train_subset:
        xtr, ytr = xtr[:train_subset], ytr[:train_subset]
    if test_subset:
        p = torch.randperm(len(xte), generator=g)[:test_subset]
        xte, yte = xte[p], yte[p]
    train = TensorLoader(xtr, ytr, name, batch_size, shuffle=True, train_aug=True, resolution=resolution,
                         device=device)
    test = TensorLoader(xte, yte, name, 512, resolution=resolution, device=device)
    return (train, val, test) if val_split else (train, test)
