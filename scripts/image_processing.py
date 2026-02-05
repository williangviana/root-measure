import numpy as np
import cv2

from config import THRESHOLD_FRAC, SCALE_PX_PER_CM, gaussian_sigma, threshold_8bit


def preprocess(image, scale=SCALE_PX_PER_CM, sensitivity='thick', threshold=None):
    """Threshold, blur, re-threshold to produce a clean binary root mask.

    Automatically handles 8-bit and 16-bit images.
    Blur and threshold scale with DPI and sensitivity preset.

    Args:
        threshold: Manual threshold value in 8-bit range (0-255).
                   If None, uses Otsu's method for auto-detection.
    """
    is_16bit = image.dtype == np.uint16

    if threshold is None:
        # Auto-detect using Otsu's method
        if is_16bit:
            img8 = (image / 256).astype(np.uint8)
            otsu_thresh, _ = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            otsu_thresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Adjust for sensitivity: thin roots need higher threshold
        # Higher DPI (>=500, ~197 px/cm) needs less adjustment
        high_res = scale >= 197
        if sensitivity == 'thin':
            adj = 30 if high_res else 45
            otsu_thresh = min(255, otsu_thresh + adj)
        elif sensitivity == 'medium':
            adj = 15 if high_res else 25
            otsu_thresh = min(255, otsu_thresh + adj)
        threshold = int(otsu_thresh)

    # Scale threshold to image bit depth
    if is_16bit:
        threshold_scaled = threshold * 256
    else:
        threshold_scaled = threshold

    binary = (image < threshold_scaled).astype(np.uint8) * 255
    sigma = gaussian_sigma(scale, sensitivity)
    thresh = threshold_8bit(scale, sensitivity)
    blurred = cv2.GaussianBlur(binary, (0, 0), sigma)
    clean = blurred > thresh
    return clean
