"""
Training script for Spectre_Architecture (CNN + ViT + Hybrid).
"""

import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CNNModel, HybridModel
from vit_model import VisionTransformerModel

# ================= CONFIG =================

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "Architecture" / "data" / "dataset"
CLASSES_PATH = ROOT_DIR / "classes.json"
MODEL_PATH = ROOT_DIR / "model.pth"

BATCH_SIZE = 8
EPOCHS = 15
IMAGE_SIZE = (224, 224)

# 🔥 CHANGE THIS ONLY

MODEL_TYPE = "hybrid"   # "cnn", "vit", or "hybrid"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp")

# ==========================================

def is_valid_image_file(file_path: str) -> bool:
    return file_path.lower().endswith(VALID_EXTENSIONS)

def main():
    if MODEL_TYPE not in {"cnn", "vit", "hybrid"}:
        raise ValueError("MODEL_TYPE must be 'cnn', 'vit', or 'hybrid'.")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_DIR}")

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(
        root=str(DATASET_DIR),
        transform=transform,
        is_valid_file=is_valid_image_file
    )

    print("Dataset path:", DATASET_DIR)
    print("Total images:", len(dataset))
    print("Classes:", dataset.classes)

    if len(dataset) == 0:
        raise ValueError("No images found in dataset")

    # ===== CLASS BALANCING =====
    counts = Counter(label for _, label in dataset.samples)
    class_counts = [counts[i] for i in range(len(dataset.classes))]
    print("Class counts:", class_counts)

    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ===== MODEL SELECTION =====
    if MODEL_TYPE == "cnn":
        model = CNNModel(num_classes=len(dataset.classes))
    elif MODEL_TYPE == "vit":
        model = VisionTransformerModel(num_classes=len(dataset.classes))
    else:
        model = HybridModel(num_classes=len(dataset.classes))

    print("Model type:", MODEL_TYPE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ===== TRAINING =====
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            optimizer.zero_grad()

            outputs = model(images)

            # handle ViT output tuple
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.2f}%")

    # ===== SAVE =====
    with CLASSES_PATH.open("w") as f:
        json.dump(dataset.classes, f, indent=2)

    torch.save({
        "model_type": MODEL_TYPE,
        "state_dict": model.state_dict(),
        "classes": dataset.classes
    }, MODEL_PATH)

    print("Training complete!")
    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()
