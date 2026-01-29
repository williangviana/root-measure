# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCALE_PX_PER_CM = 472         # 1200 DPI ≈ 472 pixels/cm
THRESHOLD_FRAC = 0.84         # Threshold as fraction of max (55000/65535 ≈ 0.84)
DISPLAY_DOWNSAMPLE = 2        # Downsample factor for cropped plate display (lower = sharper)
MIN_PLATE_AREA_FRAC = 0.03    # Min plate area as fraction of image area
ROOT_CROP_PAD_FRAC = 0.05     # Padding around root area as fraction of plate size

# --- Baseline preprocessing params (tuned for thick roots at 1200 DPI / 472 px/cm) ---
_BASE_SCALE = 472             # Reference scale these values were tuned at
_BASE_SIGMA = 4.5             # Gaussian blur sigma at reference scale
_BASE_THRESH = 60             # 8-bit threshold after blur at reference scale
_BASE_MIN_COMP = 100          # Min skeleton component size at reference scale

# Sensitivity presets: (sigma_multiplier, threshold_multiplier)
# Lower sigma keeps thin roots; lower threshold keeps faint signal
SENSITIVITY_PRESETS = {
    'thick':  (1.0, 1.0),     # Setaria viridis, Brassica napus
    'normal': (0.7, 0.7),     # Schrenkiella parvula
    'thin':   (0.45, 0.45),   # Arabidopsis thaliana
}

# Physical ROI limits (in cm) — converted to pixels using DPI at runtime
ROI_HALF_WIDTH_CM = 0.85      # Half-width of ROI around root (~8.5mm, allows curved roots)
ROI_VERTICAL_CM = 10.6        # Max root length to search for tip (~10.6 cm, plate limit)
ROI_PAD_CM = 0.65             # Padding above/below click pair for tracing ROI
MAX_CLICK_DISTANCE_CM = 0.65  # Max distance from click to nearest skeleton pixel


def gaussian_sigma(scale, sensitivity='thick'):
    """Gaussian blur sigma controlled by sensitivity only (DPI-independent)."""
    s_mult, _ = SENSITIVITY_PRESETS[sensitivity]
    return _BASE_SIGMA * s_mult


def threshold_8bit(scale, sensitivity='thick'):
    """8-bit threshold scaled by sensitivity only (contrast is DPI-independent)."""
    _, t_mult = SENSITIVITY_PRESETS[sensitivity]
    return int(_BASE_THRESH * t_mult)


def min_component_size(scale):
    """Min skeleton component size (DPI-independent)."""
    return _BASE_MIN_COMP


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
