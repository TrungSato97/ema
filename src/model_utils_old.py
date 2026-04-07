"""
Model utilities: construct resnet18 pre-trained, optimizer, scheduler, checkpoint load/save.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
from .config import TrainConfig
import logging

logger = logging.getLogger(__name__)


def get_model(num_classes: int, pretrained: bool = True, device: str = "cpu") -> nn.Module:
    """
    Load ResNet-18 pretrained on ImageNet and replace the final fc layer for num_classes.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) if pretrained else models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(device)
    logger.info("Loaded ResNet18 with %d output classes on device %s", num_classes, device)
    return model


def get_optimizer_scheduler(model: nn.Module, config: TrainConfig, total_epochs: int) -> Tuple[object, object]:
    """
    Return optimizer and scheduler according to config.
    Uses CosineAnnealingLR scheduler with T_max = total_epochs
    """
    if config.optimizer.lower() == "sgd":
        optimizer = SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))
    return optimizer, scheduler
