import cv2
import numpy as np


IMAGE_SIZE = (256, 256)


def _zero_mean_normalize(data: np.ndarray) -> np.ndarray:
    """Normalize the residual so it has zero mean and unit-scale when possible."""
    centered = data - np.mean(data)
    std = np.std(centered)
    if std > 1e-8:
        centered = centered / std
    return centered


def extract_prnu_features(image_path: str) -> np.ndarray | None:
    """
    Extract simple PRNU-inspired statistical features from an image.

    Steps:
    1. Read grayscale image
    2. Resize to 256x256
    3. Denoise with Gaussian blur
    4. Compute noise residual
    5. Apply zero-mean normalization
    6. Return summary statistics
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[WARN] Unable to read image: {image_path}")
            return None

        resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        original = resized.astype(np.float32)
        denoised = cv2.GaussianBlur(original, (5, 5), 0)

        noise_residual = original - denoised
        normalized_noise = _zero_mean_normalize(noise_residual)

        features = np.array(
            [
                np.mean(normalized_noise),
                np.std(normalized_noise),
                np.var(normalized_noise),
                np.mean(np.abs(normalized_noise)),
                np.max(normalized_noise),
                np.min(normalized_noise),
            ],
            dtype=np.float32,
        )
        return features
    except Exception as exc:
        print(f"[WARN] PRNU extraction failed for {image_path}: {exc}")
        return None
