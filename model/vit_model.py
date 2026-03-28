import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class VisionTransformerModel(nn.Module):
    """
    Vision Transformer classifier that returns:
    1. logits
    2. feature embedding before the final classification layer
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        x = self.vit._process_input(x)

        class_token = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = self.vit.encoder(x)

        embedding = x[:, 0]
        logits = self.vit.heads(embedding)
        return logits, embedding
