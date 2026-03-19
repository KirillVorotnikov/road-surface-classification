import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss — works better with class imbalances."""

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def create_criterion(config) -> nn.Module:
    loss_name = config.training.get("loss", "cross_entropy")
    label_smoothing = config.training.get("label_smoothing", 0.0)

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_name == "focal":
        gamma = config.training.get("focal_gamma", 2.0)
        return FocalLoss(gamma=gamma, label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
