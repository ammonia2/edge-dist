"""Automated architecture generator: turns an architecture config (see
search_space.py) into a trainable `nn.Module`."""
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "relu6": lambda: nn.ReLU6(inplace=True),
    "hardswish": lambda: nn.Hardswish(inplace=True),
    "silu": lambda: nn.SiLU(inplace=True),
    "leaky_relu": lambda: nn.LeakyReLU(0.1, inplace=True),
}


def conv_bn(cin, cout, k, s, act=None, groups=1):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(ACTIVATIONS[act]())
    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    """Standard KxK convolution block with a residual connection when shapes match."""

    def __init__(self, cin, cout, k, s, act, **_):
        super().__init__()
        self.body = conv_bn(cin, cout, k, s, act)
        self.res = s == 1 and cin == cout

    def forward(self, x):
        return x + self.body(x) if self.res else self.body(x)


class DWSepBlock(nn.Module):
    """Depthwise-separable conv: KxK depthwise -> 1x1 pointwise."""

    def __init__(self, cin, cout, k, s, act, **_):
        super().__init__()
        self.body = nn.Sequential(conv_bn(cin, cin, k, s, act, groups=cin), conv_bn(cin, cout, 1, 1, act))
        self.res = s == 1 and cin == cout

    def forward(self, x):
        return x + self.body(x) if self.res else self.body(x)


class MBConvBlock(nn.Module):
    """Inverted residual: 1x1 expand -> KxK depthwise -> 1x1 linear projection."""

    def __init__(self, cin, cout, k, s, act, expand=4):
        super().__init__()
        h = cin * expand
        self.body = nn.Sequential(conv_bn(cin, h, 1, 1, act), conv_bn(h, h, k, s, act, groups=h),
                                  conv_bn(h, cout, 1, 1, None))
        self.res = s == 1 and cin == cout

    def forward(self, x):
        return x + self.body(x) if self.res else self.body(x)


BLOCKS = {"conv": ConvBlock, "dwsep": DWSepBlock, "mbconv": MBConvBlock}


class GeneratedNet(nn.Module):
    def __init__(self, arch, num_classes=10):
        super().__init__()
        self.arch = arch
        self.resolution = arch["resolution"]
        first_act = arch["stages"][0]["act"]
        layers = [conv_bn(3, arch["stem_channels"], 3, 1, first_act)]
        cin = arch["stem_channels"]
        for st in arch["stages"]:
            for i in range(st["blocks"]):
                layers.append(BLOCKS[st["block"]](cin, st["channels"], st["kernel"], 2 if i == 0 else 1,
                                                  st["act"], expand=st["expand"]))
                cin = st["channels"]
        self.features = nn.Sequential(*layers)
        self.feature_channels = cin
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(self, x, return_features=False):
        # inputs of a different size are resized to the searched resolution
        if x.shape[-1] != self.resolution:
            x = F.interpolate(x, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        f = self.features(x)
        logits = self.classifier(self.pool(f).flatten(1))
        return (logits, f) if return_features else logits


def build_model(arch, num_classes=10):
    return GeneratedNet(arch, num_classes)
