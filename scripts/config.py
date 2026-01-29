# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCALE_PX_PER_CM = 472         # 1200 DPI ≈ 472 pixels/cm
THRESHOLD_FRAC = 0.84         # Threshold as fraction of max (55000/65535 ≈ 0.84)
GAUSSIAN_SIGMA = 4.5          # Gaussian blur sigma (pixels)
THRESHOLD_8BIT = 60           # 8-bit threshold after blur
DISPLAY_DOWNSAMPLE = 2        # Downsample factor for cropped plate display (lower = sharper)
MIN_COMPONENT_SIZE = 100      # Ignore tiny skeleton fragments
MIN_PLATE_AREA_FRAC = 0.03    # Min plate area as fraction of image area
ROOT_CROP_PAD_FRAC = 0.05     # Padding around root area as fraction of plate size

# Physical ROI limits (in cm) — converted to pixels using DPI at runtime
ROI_HALF_WIDTH_CM = 0.85      # Half-width of ROI around root (~8.5mm, allows curved roots)
ROI_VERTICAL_CM = 10.6        # Max root length to search for tip (~10.6 cm, plate limit)
ROI_PAD_CM = 0.65             # Padding above/below click pair for tracing ROI
MAX_CLICK_DISTANCE_CM = 0.65  # Max distance from click to nearest skeleton pixel


def roi_half_width_px(scale):
    """ROI half-width in pixels, computed from scale (px/cm)."""
    return int(ROI_HALF_WIDTH_CM * scale)


def roi_vertical_px(scale):
    """ROI vertical extent in pixels, computed from scale (px/cm)."""
    return int(ROI_VERTICAL_CM * scale)


def roi_pad_px(scale):
    """ROI padding in pixels, computed from scale (px/cm)."""
    return int(ROI_PAD_CM * scale)


def max_click_distance_px(scale):
    """Max click-to-skeleton distance in pixels, computed from scale (px/cm)."""
    return int(MAX_CLICK_DISTANCE_CM * scale)
