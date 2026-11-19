"""Lightweight student architectures, implemented from scratch and adapted to
32x32 CIFAR inputs (the ImageNet stem stride-2 is replaced by stride-1).

All students implement `forward(x, return_features=False)`; with
`return_features=True` they return `(logits, last_feature_map)` for
feature-based distillation. `feature_channels` gives the channel count of that map.
"""
import torch.nn as nn


def _make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1, act=True):
        layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
        if act:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """MobileNetV2 / EfficientNet-Lite MBConv block (no squeeze-excite)."""

    def __init__(self, cin, cout, stride, expand, k=3):
        super().__init__()
        hidden = cin * expand
        self.use_res = stride == 1 and cin == cout
        layers = []
        if expand != 1:
            layers.append(ConvBNAct(cin, hidden, 1))
        layers += [ConvBNAct(hidden, hidden, k, stride, groups=hidden), ConvBNAct(hidden, cout, 1, act=False)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


class _Classifier(nn.Module):
    def _head(self, x, return_features):
        logits = self.classifier(self.pool(x).flatten(1))
        return (logits, x) if return_features else logits


# --------------------------------------------------------------------------- MobileNetV2
class MobileNetV2(_Classifier):
    # t, c, n, s
    CFG = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes=100, width=1.0, last_channels=1280, dropout=0.2):
        super().__init__()
        cin = _make_divisible(32 * width)
        layers = [ConvBNAct(3, cin, 3, 1)]  # CIFAR: stride 1 stem
        for t, c, n, s in self.CFG:
            cout = _make_divisible(c * width)
            for i in range(n):
                layers.append(InvertedResidual(cin, cout, s if i == 0 else 1, t))
                cin = cout
        self.feature_channels = _make_divisible(last_channels * max(1.0, width))
        layers.append(ConvBNAct(cin, self.feature_channels, 1))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_channels, num_classes))
        _init(self)

    def forward(self, x, return_features=False):
        return self._head(self.features(x), return_features)


# --------------------------------------------------------------------------- SqueezeNet 1.1
class Fire(nn.Module):
    def __init__(self, cin, squeeze, e1, e3):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(cin, squeeze, 1), nn.BatchNorm2d(squeeze), nn.ReLU(inplace=True))
        self.e1 = nn.Sequential(nn.Conv2d(squeeze, e1, 1), nn.BatchNorm2d(e1), nn.ReLU(inplace=True))
        self.e3 = nn.Sequential(nn.Conv2d(squeeze, e3, 3, padding=1), nn.BatchNorm2d(e3), nn.ReLU(inplace=True))

    def forward(self, x):
        import torch
        s = self.squeeze(x)
        return torch.cat([self.e1(s), self.e3(s)], 1)


class SqueezeNet(_Classifier):
    """SqueezeNet v1.1 (with BatchNorm for stable short training)."""
    feature_channels = 512

    def __init__(self, num_classes=100, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),   # 32x32
            nn.MaxPool2d(3, 2, 1),                                                     # 16x16
            Fire(64, 16, 64, 64), Fire(128, 16, 64, 64),
            nn.MaxPool2d(3, 2, 1),                                                     # 8x8
            Fire(128, 32, 128, 128), Fire(256, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1),                                                     # 4x4
            Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256), Fire(512, 64, 256, 256),
        )
        # SqueezeNet classifies with a 1x1 conv followed by global average pooling
        self.classifier_conv = nn.Sequential(nn.Dropout(dropout), nn.Conv2d(512, num_classes, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        _init(self)

    def forward(self, x, return_features=False):
        f = self.features(x)
        logits = self.pool(self.classifier_conv(f)).flatten(1)
        return (logits, f) if return_features else logits


# --------------------------------------------------------------------------- EfficientNet-Lite0
class EfficientNetLite0(_Classifier):
    """EfficientNet-Lite0: EfficientNet-B0 without squeeze-excite, ReLU6
    activations and a fixed (non-scaled) stem/head, as used for edge TPUs/mobile."""
    # expand, kernel, stride, out_channels, repeats
    CFG = [(1, 3, 1, 16, 1), (6, 3, 2, 24, 2), (6, 5, 2, 40, 2), (6, 3, 2, 80, 3),
           (6, 5, 1, 112, 3), (6, 5, 2, 192, 4), (6, 3, 1, 320, 1)]
    feature_channels = 1280

    def __init__(self, num_classes=100, dropout=0.2):
        super().__init__()
        layers = [ConvBNAct(3, 32, 3, 1)]  # CIFAR: stride 1 stem
        cin = 32
        for e, k, s, c, n in self.CFG:
            for i in range(n):
                layers.append(InvertedResidual(cin, c, s if i == 0 else 1, e, k))
                cin = c
        layers.append(ConvBNAct(cin, self.feature_channels, 1))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_channels, num_classes))
        _init(self)

    def forward(self, x, return_features=False):
        return self._head(self.features(x), return_features)


def _init(m):
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            nn.init.kaiming_normal_(mod.weight, mode="fan_out")
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.BatchNorm2d):
            nn.init.ones_(mod.weight); nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, 0, 0.01); nn.init.zeros_(mod.bias)


STUDENTS = {
    "mobilenetv2": MobileNetV2,
    "squeezenet": SqueezeNet,
    "efficientnet_lite0": EfficientNetLite0,
}


def build_student(name, num_classes=100):
    return STUDENTS[name](num_classes=num_classes)
