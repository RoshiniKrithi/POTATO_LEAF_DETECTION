from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm
import timm


def _replace_classifier(model: nn.Module, num_features: int, num_classes: int) -> None:
    classifier = nn.Linear(num_features, num_classes)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        # e.g., mobilenet_v3_large
        model.classifier[-1] = classifier
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        model.classifier = classifier
    elif hasattr(model, "fc"):
        model.fc = classifier
    elif hasattr(model, "head"):
        model.head = classifier
    else:
        raise ValueError("Unknown model head to replace with classifier")


def create_model(name: str, num_classes: int, pretrained: bool = True) -> Tuple[nn.Module, int]:
    name = name.lower()
    if name == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        _replace_classifier(model, model.fc.in_features, num_classes)
        image_size = 224
    elif name.startswith("efficientnet_b"):
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        image_size = 300 if name in {"efficientnet_b0"} else 380
    elif name == "mobilenet_v3_large":
        model = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        _replace_classifier(model, model.classifier[-1].in_features, num_classes)
        image_size = 224
    else:
        # fallback to timm
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        image_size = 224
    return model, image_size


