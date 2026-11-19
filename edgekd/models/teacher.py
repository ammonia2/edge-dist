"""ResNet-50 teacher for CIFAR-100.

Initialised from ImageNet weights (transfer learning keeps CPU training short) and
fine-tuned on CIFAR-100. Inputs are 32x32 CIFAR tensors; they are bilinearly
upsampled to `input_res` inside the model so teacher and students share one
data pipeline. `forward(x, return_features=True)` also returns the layer4 feature
map used for feature-based distillation.
"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Teacher(nn.Module):
    feature_channels = 2048

    def __init__(self, num_classes=100, input_res=64, pretrained=True):
        super().__init__()
        self.input_res = input_res
        self.freeze_prefix = False
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net

    def frozen_prefix(self):
        """Modules kept frozen during fast fine-tuning."""
        n = self.net
        return [n.conv1, n.bn1, n.layer1, n.layer2]

    def freeze(self):
        """Freeze stem..layer2 (weights and BN statistics) to cut CPU training cost."""
        self.freeze_prefix = True
        for m in self.frozen_prefix():
            for p in m.parameters():
                p.requires_grad = False
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_prefix:
            for m in self.frozen_prefix():
                m.eval()
        return self

    def forward(self, x, return_features=False):
        if x.shape[-1] != self.input_res:
            x = F.interpolate(x, size=(self.input_res, self.input_res), mode="bilinear", align_corners=False)
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        logits = n.fc(n.avgpool(x).flatten(1))
        return (logits, x) if return_features else logits
