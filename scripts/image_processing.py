import numpy as np
import cv2

from config import THRESHOLD_FRAC, SCALE_PX_PER_CM, gaussian_sigma, threshold_8bit


def preprocess(image, scale=SCALE_PX_PER_CM, sensitivity='thick'):
    """Threshold, blur, re-threshold to produce a clean binary root mask.

    Automatically handles 8-bit and 16-bit images.
    Blur and threshold scale with DPI and sensitivity preset.
    """
    max_val = 65535 if image.dtype == np.uint16 else 255
    threshold = int(max_val * THRESHOLD_FRAC)
    binary = (image < threshold).astype(np.uint8) * 255
    sigma = gaussian_sigma(scale, sensitivity)
    thresh = threshold_8bit(scale, sensitivity)
    blurred = cv2.GaussianBlur(binary, (0, 0), sigma)
    clean = blurred > thresh
    return clean
