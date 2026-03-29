import cv2
import numpy as np
import pywt


IMAGE_SIZE = (256, 256)


def extract_signal_features(image_path: str) -> np.ndarray | None:
    """
    Extract camera signal features using CFA-style differences, DCT, and DWT.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[WARN] Unable to read image: {image_path}")
            return None

        resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        gray = resized.astype(np.float32)

        horizontal_diff = np.mean(np.abs(gray[:, 1:] - gray[:, :-1]))
        vertical_diff = np.mean(np.abs(gray[1:, :] - gray[:-1, :]))

        dct_coefficients = cv2.dct(gray)
        dct_mean_abs = np.mean(np.abs(dct_coefficients))
        dct_std = np.std(dct_coefficients)

        _, (_, _, hh_band) = pywt.dwt2(gray, "haar")
        dwt_mean_abs = np.mean(np.abs(hh_band))
        dwt_std = np.std(hh_band)

        return np.array(
            [
                horizontal_diff,
                vertical_diff,
                dct_mean_abs,
                dct_std,
                dwt_mean_abs,
                dwt_std,
            ],
            dtype=np.float32,
        )
    except Exception as exc:
        print(f"[WARN] Signal extraction failed for {image_path}: {exc}")
        return None
