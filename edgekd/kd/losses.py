"""Knowledge-distillation losses.

* soft-target KD with temperature-scaled softmax (Hinton et al., 2015)
* combined hard-label (CE) + soft-label (KL) objective
* feature-based distillation (FitNets-style hint loss) on intermediate feature maps
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_with_temperature(logits, T):
    """Temperature-scaled softmax: p_i = exp(z_i / T) / sum_j exp(z_j / T)."""
    return F.softmax(logits / T, dim=1)


def soft_target_loss(student_logits, teacher_logits, T):
    """KL(p_teacher^T || p_student^T) * T^2.

    The T^2 factor keeps gradient magnitudes comparable to the hard-label loss
    when the temperature changes.
    """
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = softmax_with_temperature(teacher_logits, T)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


class FeatureDistiller(nn.Module):
    """Aligns a student feature map to a teacher feature map.

    A learnable 1x1 conv + BN regressor maps student channels -> teacher channels;
    both maps are pooled to a common spatial size and the MSE between
    channel-normalised features is minimised.
    """

    def __init__(self, student_channels, teacher_channels, spatial=2):
        super().__init__()
        self.regressor = nn.Sequential(nn.Conv2d(student_channels, teacher_channels, 1, bias=False),
                                       nn.BatchNorm2d(teacher_channels))
        self.spatial = spatial

    def forward(self, f_s, f_t):
        f_s = F.adaptive_avg_pool2d(self.regressor(f_s), self.spatial)
        f_t = F.adaptive_avg_pool2d(f_t, self.spatial)
        f_s = F.normalize(f_s.flatten(1), dim=1)
        f_t = F.normalize(f_t.flatten(1), dim=1)
        return (f_s - f_t).pow(2).sum(1).mean()


class DistillationLoss(nn.Module):
    """L = (1 - alpha) * CE(y, s) + alpha * T^2 * KL(t^T || s^T) + beta * L_feat.

    alpha = 0 -> plain hard-label training; beta = 0 -> logit-only KD.
    """

    def __init__(self, temperature=4.0, alpha=0.7, beta=1.0, feature_distiller=None):
        super().__init__()
        self.T, self.alpha, self.beta = temperature, alpha, beta
        self.feat = feature_distiller

    def forward(self, s_logits, t_logits, labels, s_feat=None, t_feat=None):
        hard = F.cross_entropy(s_logits, labels)
        soft = soft_target_loss(s_logits, t_logits, self.T)
        loss = (1 - self.alpha) * hard + self.alpha * soft
        parts = {"hard": hard.item(), "soft": soft.item()}
        if self.feat is not None and self.beta > 0 and s_feat is not None:
            fl = self.feat(s_feat, t_feat)
            loss = loss + self.beta * fl
            parts["feat"] = fl.item()
        return loss, parts
