"""
Model utilities: construct resnet18 pre-trained, optimizer, scheduler, checkpoint load/save.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
from huggingface_hub import hf_hub_download
from .config import TrainConfig
import logging

logger = logging.getLogger(__name__)


def get_model(num_classes: int, pretrained: bool = True, device: str = "cpu") -> nn.Module:
    """
    Load ResNet-18 pretrained on ImageNet and replace the final fc layer for num_classes.
    """
    if num_classes == 14:
        logger.info("Initializing ResNet18 for Clothing1M (ImageNet1K Pretrained)...")
        # Load weights chuẩn cho ảnh 224x224
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Đổi lớp cuối cùng thành 14 output classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        logger.info("Loaded ResNet18 with %d output classes on device %s", num_classes, device)
        return model
    
    use_hf_for_imagenet100 = (num_classes == 100 and pretrained) 
    
    if use_hf_for_imagenet100:
        logger.info("Handling ImageNet-100 case: Downloading pretrained ResNet18 from Hugging Face.")
        model = models.resnet18(weights=None)
        
        try:
            repo_id = "edadaltocg/resnet18_cifar100"
            model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
            state_dict = torch.load(model_path, map_location='cpu')
            
            # --- PHẦN SỬA LỖI: Lọc bỏ các layer bị sai kích thước (mismatch shape) ---
            model_state = model.state_dict()
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k in model_state and v.shape == model_state[k].shape:
                    filtered_state_dict[k] = v
                else:
                    logger.warning(f"Bỏ qua layer '{k}' do khác kích thước. Checkpoint: {v.shape}, Model: {model_state[k].shape if k in model_state else 'Không có'}")
            
            # Load state dict đã được lọc
            model.load_state_dict(filtered_state_dict, strict=False)
            logger.info("Successfully loaded matching Hugging Face weights.")
            
        except Exception as e:
            logger.error(f"Error loading from HF: {e}. Falling back to default torchvision logic.")
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) if pretrained else models.resnet18(weights=None)

    # Thay thế lớp fc cuối cùng
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