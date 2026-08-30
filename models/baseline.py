import torch.nn as nn
from torchvision import models


def build_efficientnet_b0(num_classes=5, pretrained=True):
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model