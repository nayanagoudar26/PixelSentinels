import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from prnu_module import extract_prnu_features
from signal_module import extract_signal_features


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "camera_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def extract_combined_features(image_path: str) -> np.ndarray | None:
    """Combine PRNU and signal features for one image."""
    prnu_features = extract_prnu_features(image_path)
    signal_features = extract_signal_features(image_path)

    if prnu_features is None or signal_features is None:
        return None

    return np.concatenate([signal_features, prnu_features]).astype(np.float32)


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load images from subfolders where each folder name is the camera label."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    feature_rows: list[np.ndarray] = []
    labels: list[str] = []
    processed_paths: list[str] = []

    subfolders = sorted([folder for folder in dataset_dir.iterdir() if folder.is_dir()])
    if not subfolders:
        raise ValueError(f"No class subfolders found inside: {dataset_dir}")

    for folder in subfolders:
        label = folder.name
        print(f"\n[INFO] Processing folder: {label}")

        image_files = sorted(
            [
                image_path
                for image_path in folder.iterdir()
                if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS
            ]
        )

        if not image_files:
            print(f"[WARN] No valid images found in folder: {folder}")
            continue

        for index, image_path in enumerate(image_files, start=1):
            if index == 1 or index % 25 == 0 or index == len(image_files):
                print(
                    f"[INFO] Processing image {index}/{len(image_files)}: {image_path.name}"
                )
            try:
                combined_features = extract_combined_features(str(image_path))
                if combined_features is None:
                    continue

                feature_rows.append(combined_features)
                labels.append(label)
                processed_paths.append(str(image_path))
            except Exception as exc:
                print(f"[WARN] Failed to process {image_path}: {exc}")

    if not feature_rows:
        raise ValueError("No valid features were extracted from the dataset.")

    return np.vstack(feature_rows), np.array(labels), processed_paths


def should_stratify(labels: np.ndarray) -> bool:
    unique_labels, counts = np.unique(labels, return_counts=True)
    return len(unique_labels) > 1 and np.min(counts) >= 2


def train_pipeline() -> tuple[RandomForestClassifier, StandardScaler, list[str]]:
    """Train, evaluate, and save the classifier and scaler."""
    print("[INFO] Loading dataset...")
    X, y, processed_paths = load_dataset(IMAGES_DIR)

    print(f"\n[INFO] Total processed images: {len(X)}")
    print(f"[INFO] Total classes: {len(np.unique(y))}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    stratify_labels = y if should_stratify(y) else None
    if stratify_labels is None:
        print("[WARN] Stratified split skipped because at least one class has fewer than 2 samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels,
    )

    print("\n[INFO] Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("[INFO] Saving model artifacts...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[INFO] Model saved to: {MODEL_PATH}")
    print("Scaler saved at models/scaler.pkl")

    return model, scaler, processed_paths


def classify_ai_placeholder(confidence: float) -> str:
    """
    Placeholder AI/Real flag.
    Replace with a dedicated detector if you later add one.
    """
    return "Real (placeholder)" if confidence >= 0.50 else "AI/Real uncertain (placeholder)"


def test_image(
    image_path: str,
    model: RandomForestClassifier | None = None,
    scaler: StandardScaler | None = None,
) -> dict | None:
    """
    Predict the camera model for a single image and return summary results.
    """
    try:
        image_file = Path(image_path)
        if not image_file.exists():
            print(f"[WARN] Test image not found: {image_path}")
            return None

        if model is None:
            if not MODEL_PATH.exists():
                print("[WARN] Trained model not found. Train the model first.")
                return None
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)

        if scaler is None:
            if not SCALER_PATH.exists():
                print("[WARN] Trained scaler not found. Train the model first.")
                return None
            with open(SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)

        prnu_features = extract_prnu_features(str(image_file))
        signal_features = extract_signal_features(str(image_file))
        if prnu_features is None or signal_features is None:
            print(f"[WARN] Could not extract features from test image: {image_path}")
            return None

        features = np.concatenate([signal_features, prnu_features]).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)[0]

        confidence = 0.0
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(features)))

        result = {
            "ai_or_real": classify_ai_placeholder(confidence),
            "predicted_camera_model": prediction,
            "confidence_score": confidence,
        }

        print("\nSample Test Result:")
        print(f"AI/Real: {result['ai_or_real']}")
        print(f"Predicted camera model: {result['predicted_camera_model']}")
        print(f"Confidence score: {result['confidence_score']:.4f}")

        return result
    except Exception as exc:
        print(f"[WARN] Test prediction failed for {image_path}: {exc}")
        return None


def find_sample_image() -> str | None:
    """Pick the first valid dataset image for a quick post-training test."""
    for folder in sorted(IMAGES_DIR.iterdir()):
        if not folder.is_dir():
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in VALID_EXTENSIONS:
                return str(image_path)
    return None


if __name__ == "__main__":
    trained_model, trained_scaler, _ = train_pipeline()

    sample_image = find_sample_image()
    if sample_image:
        print(f"\n[INFO] Running sample test on: {sample_image}")
        test_image(sample_image, trained_model, trained_scaler)
    else:
        print("[WARN] No sample image found for testing.")
