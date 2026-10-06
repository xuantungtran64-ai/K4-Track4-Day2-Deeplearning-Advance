"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC."""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    model = timm.create_model(
        name,
        pretrained=(pretrained if init != "scratch" else False),
        num_classes=num_classes,
        drop_rate=drop_rate
    )
    if init == "frozen":
        freeze_backbone(model)
    return model


def freeze_backbone(model) -> None:
    for param in model.parameters():
        param.requires_grad = False
    
    if hasattr(model, 'get_classifier'):
        head = model.get_classifier()
        if head is not None:
            for param in head.parameters():
                param.requires_grad = True


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    head_params = []
    if hasattr(model, 'get_classifier'):
        head = model.get_classifier()
        if head is not None:
            head_params = list(head.parameters())
    head_param_ids = [id(p) for p in head_params]

    backbone_decay = []
    backbone_no_decay = []
    head_decay = []
    
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in head_param_ids:
            head_decay.append(param)
        else:
            if param.ndim <= 1:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
                
    return [
        {"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": head_decay, "lr": lr_head, "weight_decay": weight_decay}
    ]


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    try:
        from fvcore.nn import FlopCountAnalysis
        inputs = torch.randn(1, 3, img_size, img_size, device=next(model.parameters()).device)
        flops = FlopCountAnalysis(model, inputs).total()
        return flops / 1e9
    except:
        return 0.0
