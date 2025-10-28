import torch
import torch.nn as nn
from torchvision.models import resnet18

class FingertipResNet(nn.Module):
    def __init__(self, num_outputs=15, pretrained=True):
        super().__init__()
        self.backbone = resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_outputs)
        )

    def forward(self, x):
        return self.backbone(x)
