"""
Custom Loss Functions for Class Imbalance in Medical Image Classification.
Implements Class-Weighted BCE Loss and Binary Focal Loss with logits support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for severe class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples and focuses gradient on hard minority samples.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for the rare/minority (Fractured) class (0 < alpha < 1).
            gamma: Focusing parameter (gamma >= 0). When gamma=0, reduces to standard BCE.
            reduction: 'mean' or 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits from model [N] or [N, 1].
            targets: Binary labels [N] or [N, 1] in {0, 1}.
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        # Compute standard binary cross entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Convert logits to probability
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        focal_loss = alpha_t * modulating_factor * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(
    method: str = "class_weighted",
    pos_weight: float = 3.55,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """
    Factory function returning the configured loss function.
    """
    if method == "class_weighted":
        weight_tensor = torch.tensor([pos_weight], device=device)
        return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    elif method == "focal_loss":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma).to(device)
    elif method == "standard_bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss method: {method}")
