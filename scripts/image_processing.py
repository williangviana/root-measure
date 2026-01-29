import numpy as np
import cv2

from config import THRESHOLD_FRAC, GAUSSIAN_SIGMA, THRESHOLD_8BIT


def preprocess(image):
    """Threshold, blur, re-threshold to produce a clean binary root mask.

    Automatically handles 8-bit and 16-bit images.
    """
    max_val = 65535 if image.dtype == np.uint16 else 255
    threshold = int(max_val * THRESHOLD_FRAC)
    binary = (image < threshold).astype(np.uint8) * 255
    blurred = cv2.GaussianBlur(binary, (0, 0), GAUSSIAN_SIGMA)
    clean = blurred > THRESHOLD_8BIT
    return clean
