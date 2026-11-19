"""Unified inference pipeline for every trained model (KD students, baseline
students, NAS model, teacher) with two interchangeable back-ends:

    backend="torch" - PyTorch checkpoint (checkpoints/*.pt)
    backend="onnx"  - ONNX Runtime CPU session (onnx_models/*.onnx)

    p = Predictor("mobilenetv2_kd", backend="onnx")
    p.predict("cat.png", topk=3)          # file path / PIL image / HxWx3 uint8 array
    p.predict_batch(uint8_tensor_Nx3x32x32)
"""
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from .data import DATA_ROOT, STATS
from .utils import CKPT_DIR, ONNX_DIR

STUDENT_NAMES = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
MODEL_REGISTRY = {f"{s}_{m}": {"dataset": "cifar100", "ckpt": f"{s}_{m}.pt", "arch": s}
                  for s in STUDENT_NAMES for m in ("kd", "baseline")}
MODEL_REGISTRY["nas_best"] = {"dataset": "cifar10", "ckpt": "nas_best.pt", "arch": "nas"}
MODEL_REGISTRY["teacher_resnet50"] = {"dataset": "cifar100", "ckpt": "teacher_resnet50.pt", "arch": "teacher"}


def class_names(dataset):
    if dataset == "cifar100":
        with open(os.path.join(DATA_ROOT, "cifar-100-python", "meta"), "rb") as f:
            return pickle.load(f, encoding="latin1")["fine_label_names"]
    with open(os.path.join(DATA_ROOT, "cifar-10-batches-py", "batches.meta"), "rb") as f:
        return pickle.load(f, encoding="latin1")["label_names"]


def load_torch_model(name):
    """Rebuild the nn.Module for a registry entry and load its checkpoint."""
    from .models import ResNet50Teacher, build_student
    from .nas import build_model
    info = MODEL_REGISTRY[name]
    ck = torch.load(os.path.join(CKPT_DIR, info["ckpt"]), map_location="cpu", weights_only=False)
    if info["arch"] == "nas":
        model, res = build_model(ck["arch"], 10), ck["arch"]["resolution"]
    elif info["arch"] == "teacher":
        model, res = ResNet50Teacher(100, input_res=ck["input_res"], pretrained=False), 32
    else:
        model, res = build_student(info["arch"], 100), 32
    model.load_state_dict(ck["state_dict"])
    return model.eval(), res


class Predictor:
    def __init__(self, name, backend="onnx", threads=None):
        if name not in MODEL_REGISTRY:
            raise KeyError(f"unknown model {name}; choose from {sorted(MODEL_REGISTRY)}")
        self.name, self.backend = name, backend
        self.dataset = MODEL_REGISTRY[name]["dataset"]
        self.classes = class_names(self.dataset)
        mean, std = STATS[self.dataset]
        self.mean = np.array(mean, np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(std, np.float32).reshape(1, 3, 1, 1)
        if backend == "torch":
            if threads:
                torch.set_num_threads(threads)
            self.model, self.resolution = load_torch_model(name)
        elif backend == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(os.path.join(ONNX_DIR, f"{name}.onnx"), so,
                                             providers=["CPUExecutionProvider"])
            shape = self.sess.get_inputs()[0].shape
            self.resolution = int(shape[-1])
            self.input_name = self.sess.get_inputs()[0].name
        else:
            raise ValueError(backend)

    # -------------------------------------------------------------- preprocessing
    def preprocess(self, images):
        """uint8 [N,3,H,W] (tensor or array) -> normalised float32 [N,3,R,R]."""
        x = torch.as_tensor(np.asarray(images)).float().div(255.0)
        if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
            x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        return ((x.numpy() - self.mean) / self.std).astype(np.float32)

    @staticmethod
    def load_image(img):
        """File path / PIL image / HxWx3 array -> uint8 [1,3,32,32]."""
        from PIL import Image
        if isinstance(img, str):
            img = Image.open(img)
        if isinstance(img, Image.Image):
            img = np.asarray(img.convert("RGB").resize((32, 32), Image.BILINEAR))
        arr = np.asarray(img, dtype=np.uint8)
        return arr.transpose(2, 0, 1)[None]

    # -------------------------------------------------------------- inference
    def logits(self, x):
        if self.backend == "torch":
            with torch.no_grad():
                return self.model(torch.from_numpy(x)).numpy()
        return self.sess.run(None, {self.input_name: x})[0]

    def predict_batch(self, images_uint8):
        return self.logits(self.preprocess(images_uint8)).argmax(1)

    def predict(self, img, topk=5):
        z = self.logits(self.preprocess(self.load_image(img)))[0]
        p = np.exp(z - z.max()); p /= p.sum()
        idx = p.argsort()[::-1][:topk]
        return [(self.classes[i], float(p[i])) for i in idx]
