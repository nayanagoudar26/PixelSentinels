"""
Prediction script for Spectre_Architecture.

Usage:
1. Install dependencies:
   pip install -r requirements.txt

2. Train the model first:
   python train.py

3. Run prediction:
   python predict.py --image sample.jpg
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import CNNModel, HybridModel
from vit_model import VisionTransformerModel


ROOT_DIR = Path(__file__).resolve().parent
CLASSES_PATH = ROOT_DIR / "classes.json"
MODEL_PATH = ROOT_DIR / "model.pth"
IMAGE_SIZE = (224, 224)
MODEL_TYPE = "cnn"  # Change to "vit" if you want to force the ViT branch.


def predict(image_path: str) -> None:
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Missing classes file: {CLASSES_PATH}. Run train.py first."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. Run train.py first."
        )

    with CLASSES_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)
    if not class_names:
        raise ValueError(
            f"{CLASSES_PATH} is empty. Run train.py to generate valid class names."
        )

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image not found: {image_file}")

    image = Image.open(image_file).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load trained weights from {MODEL_PATH}. "
            "Run train.py to generate a valid model.pth file."
        ) from exc

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        selected_model_type = checkpoint.get("model_type", MODEL_TYPE)
        state_dict = checkpoint["state_dict"]
        checkpoint_classes = checkpoint.get("class_names")
        if checkpoint_classes:
            class_names = checkpoint_classes
    else:
        selected_model_type = MODEL_TYPE
        state_dict = checkpoint

    if selected_model_type == "cnn":
        model = CNNModel(num_classes=len(class_names))
    elif selected_model_type == "vit":
        model = VisionTransformerModel(num_classes=len(class_names))
    elif selected_model_type == "hybrid":
        model = HybridModel(num_classes=len(class_names))
    else:
        raise ValueError(
            f"Unsupported model type '{selected_model_type}' found in model checkpoint."
        )

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        model_outputs = model(image_tensor)
        if selected_model_type == "vit":
            outputs, _ = model_outputs
        else:
            outputs = model_outputs
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probs, dim=1)

    predicted_class = class_names[predicted_index.item()]
    confidence_score = confidence.item() * 100

    print(f"Model type: {selected_model_type}")
    print(f"Predicted camera model: {predicted_class}")
    print(f"Confidence: {confidence_score:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the camera model used to capture an image."
    )
    parser.add_argument(
        "--image",
        default="sample.jpg",
        help="Path to the image to classify. Defaults to sample.jpg in the project folder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = ROOT_DIR / image_path
    predict(str(image_path))
