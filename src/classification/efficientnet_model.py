"""
EfficientNetB3 Deep Learning Model for Bone Fracture Classification.
Provides transfer learning, multi-stage fine-tuning (frozen backbone,
partial fine-tuning, full fine-tuning), and optional multi-task capability.
"""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class BoneFractureEfficientNet(nn.Module):
    """
    EfficientNetB3 Fracture Classifier with customizable fine-tuning controls.
    """
    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.3,
        num_classes: int = 1,
        multi_task: bool = False,
        num_anatomy_classes: int = 5
    ):
        super().__init__()
        self.multi_task = multi_task
        self.num_classes = num_classes
        self.num_anatomy_classes = num_anatomy_classes

        # 1. Load EfficientNetB3 backbone
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        base_model = efficientnet_b3(weights=weights)

        # Retain feature extractor and pooling
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # In-features for EfficientNetB3 classifier is 1536
        in_features = base_model.classifier[1].in_features

        # 2. Custom Fracture Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

        # 3. Optional Multi-Task Anatomy Head
        if self.multi_task:
            self.anatomy_head = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_anatomy_classes)
            )
        else:
            self.anatomy_head = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Returns:
            fracture_logits: [N, 1] raw logits
            anatomy_logits: [N, 5] logits if multi_task else None
        """
        feats = self.features(x)
        pooled = self.avgpool(feats)
        flattened = torch.flatten(pooled, 1)

        fracture_logits = self.classifier(flattened).squeeze(-1)

        anatomy_logits = None
        if self.multi_task and self.anatomy_head is not None:
            anatomy_logits = self.anatomy_head(flattened)

        return fracture_logits, anatomy_logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes sigmoid probability for fracture detection."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            return torch.sigmoid(logits)

    def freeze_backbone(self) -> None:
        """Freezes all layers in the backbone for Phase 1 head training."""
        for param in self.features.parameters():
            param.requires_grad = False
        print("Model backbone frozen. Only classification head is trainable.")

    def unfreeze_upper_layers(self, num_blocks: int = 3, freeze_bn: bool = True) -> None:
        """
        Unfreezes the top N MBConv feature blocks for Phase 2 fine-tuning.
        Optionally keeps BatchNorm running stats frozen for small datasets.
        """
        total_layers = len(self.features)
        unfreeze_start = max(0, total_layers - num_blocks)

        for i, layer in enumerate(self.features):
            requires_grad = (i >= unfreeze_start)
            for param in layer.parameters():
                param.requires_grad = requires_grad

        if freeze_bn:
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

        print(f"Unfrozen top {num_blocks} feature layers (starting from block {unfreeze_start}/{total_layers}).")

    def unfreeze_all(self, freeze_bn: bool = True) -> None:
        """Unfreezes entire network for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

        if freeze_bn:
            for m in self.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
        print("Entire network unfrozen for end-to-end fine-tuning.")

    def count_trainable_parameters(self) -> Dict[str, int]:
        """Returns counts of trainable and total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable": trainable,
            "frozen": total - trainable,
            "total": total
        }


def build_fracture_classifier(
    pretrained: bool = True,
    dropout: float = 0.3,
    multi_task: bool = False,
    device: Optional[torch.device] = None
) -> BoneFractureEfficientNet:
    """Factory function for building and placing model on target device."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BoneFractureEfficientNet(pretrained=pretrained, dropout=dropout, multi_task=multi_task)
    model.to(device)
    return model
