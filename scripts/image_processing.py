import numpy as np
import cv2

from config import THRESHOLD_FRAC, SCALE_PX_PER_CM, gaussian_sigma, threshold_8bit


def preprocess(image, scale=SCALE_PX_PER_CM, sensitivity='thick', threshold=None):
    """Threshold, blur, re-threshold to produce a clean binary root mask.

    Automatically handles 8-bit and 16-bit images.
    Blur and threshold scale with DPI and sensitivity preset.

    Args:
        threshold: Manual threshold value (0-255 for 8-bit, 0-65535 for 16-bit).
                   If None, uses Otsu's method for auto-detection.
    """
    if threshold is None:
        # Auto-detect using Otsu's method
        if image.dtype == np.uint16:
            img8 = (image / 256).astype(np.uint8)
            otsu_thresh, _ = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Adjust for sensitivity: thin roots need higher threshold
            if sensitivity == 'thin':
                otsu_thresh = min(255, otsu_thresh + 30)
            elif sensitivity == 'medium':
                otsu_thresh = min(255, otsu_thresh + 15)
            threshold = int(otsu_thresh * 256)
        else:
            otsu_thresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Adjust for sensitivity: thin roots need higher threshold
            if sensitivity == 'thin':
                otsu_thresh = min(200, otsu_thresh + 30)
            elif sensitivity == 'medium':
                otsu_thresh = min(200, otsu_thresh + 15)
            threshold = int(otsu_thresh)
    binary = (image < threshold).astype(np.uint8) * 255
    sigma = gaussian_sigma(scale, sensitivity)
    thresh = threshold_8bit(scale, sensitivity)
    blurred = cv2.GaussianBlur(binary, (0, 0), sigma)
    clean = blurred > thresh
    return clean
