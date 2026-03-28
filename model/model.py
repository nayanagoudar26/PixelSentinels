import torch
import torch.nn as nn
from torchvision.models import vit_b_16


# Lightweight CNN classifier
class CNNModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# ViT feature extractor returning the embedding (without classifier)
class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # note: older torchvision used pretrained=True; newer versions use weights
        self.vit = vit_b_16(pretrained=True)
        # remove head/classifier to get feature vector
        if hasattr(self.vit, "heads"):
            self.vit.heads = nn.Identity()
        elif hasattr(self.vit, "head"):
            self.vit.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)


# CNN feature extractor producing a 512-d vector
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 28 * 28, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# Hybrid model that concatenates CNN and ViT features
class HybridModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.cnn = CNNFeatureExtractor()
        self.vit = ViTFeatureExtractor()

        self.fc = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)

        # ensure both features are 2D (batch, features)
        if vit_feat.ndim > 2:
            vit_feat = vit_feat.view(vit_feat.size(0), -1)

        combined = torch.cat((cnn_feat, vit_feat), dim=1)
        output = self.fc(combined)
        return output
