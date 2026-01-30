import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle

from config import MIN_PLATE_AREA_FRAC


def _to_uint8(image):
    """Convert any grayscale image to uint8 for display."""
    if image.dtype == np.uint8:
        return image
    max_val = 65535 if image.dtype == np.uint16 else image.max() or 1
    return ((image.astype(np.float32) / max_val) * 255).astype(np.uint8)


def detect_two_plates(image):
    """Auto-detect two vertically stacked plates in a scanned image.

    Returns list of 2 crop regions [(r1, r2, c1, c2), ...] sorted top to bottom.
    Each region is in full-image pixel coordinates.
    """
    # work on a small version for speed
    scale = 8
    small = image[::scale, ::scale]
    small_8 = _to_uint8(small)

    # plates are bright rectangles on a dark scanner background
    # use Otsu's method to find the optimal threshold automatically
    blur = cv2.GaussianBlur(small_8, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morphological close to fill gaps inside the plate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # erode slightly to separate plates that might be touching
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.erode(closed, erode_kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    min_area = small.shape[0] * small.shape[1] * MIN_PLATE_AREA_FRAC
    plates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # scale back to full image coords with a small margin
        margin = 5  # pixels in small-image space
        r1 = max(0, (y - margin)) * scale
        r2 = min(small.shape[0], (y + h + margin)) * scale
        c1 = max(0, (x - margin)) * scale
        c2 = min(small.shape[1], (x + w + margin)) * scale
        plates.append((r1, r2, c1, c2))

    # sort top to bottom
    plates.sort(key=lambda p: p[0])

    if len(plates) < 2:
        # fallback: try to split a single large region horizontally
        if len(plates) == 1:
            r1, r2, c1, c2 = plates[0]
            mid = (r1 + r2) // 2
            plates = [
                (r1, mid, c1, c2),
                (mid, r2, c1, c2),
            ]
            print("  Found one large region — splitting into two plates.")
        else:
            # last resort: split entire image in half
            mid = image.shape[0] // 2
            plates = [
                (0, mid, 0, image.shape[1]),
                (mid, image.shape[0], 0, image.shape[1]),
            ]
            print("  Could not auto-detect plates — splitting image in half.")
    elif len(plates) > 2:
        # keep the two largest
        plates.sort(key=lambda p: (p[1] - p[0]) * (p[3] - p[2]), reverse=True)
        plates = plates[:2]
        plates.sort(key=lambda p: p[0])

    return plates


def detect_single_plate(image):
    """Auto-detect a single plate in a scanned image.

    Returns list of 1 crop region [(r1, r2, c1, c2)] in full-image pixel coordinates.
    """
    # work on a small version for speed
    scale = 8
    small = image[::scale, ::scale]
    small_8 = _to_uint8(small)

    # use Otsu's method to find the plate
    blur = cv2.GaussianBlur(small_8, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morphological close to fill gaps inside the plate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    min_area = small.shape[0] * small.shape[1] * MIN_PLATE_AREA_FRAC
    plates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        margin = 5
        r1 = max(0, (y - margin)) * scale
        r2 = min(small.shape[0], (y + h + margin)) * scale
        c1 = max(0, (x - margin)) * scale
        c2 = min(small.shape[1], (x + w + margin)) * scale
        plates.append((r1, r2, c1, c2))

    if not plates:
        # fallback: use entire image
        plates = [(0, image.shape[0], 0, image.shape[1])]
        print("  Could not auto-detect plate — using entire image.")
    else:
        # keep the single largest region
        plates.sort(key=lambda p: (p[1] - p[0]) * (p[3] - p[2]), reverse=True)
        plates = [plates[0]]
        print("  Detected 1 plate.")

    return plates


def _find_plate_interior(plate_img):
    """Find the bright agar interior of a plate, excluding edges/ruler/tape.

    Returns (ir1, ir2, ic1, ic2) as row/col bounds relative to the plate crop,
    or None if detection fails.
    """
    plate_8 = _to_uint8(plate_img)

    # threshold to find bright plate surface
    _, bright = cv2.threshold(plate_8, 200, 255, cv2.THRESH_BINARY)

    # heavy morphological close to fill the entire plate interior
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k_close)

    # find the single largest bright contour — that's the plate
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # inset by a percentage to exclude the plate rim and rounded corners
    inset_r = int(h * 0.04)
    inset_c = int(w * 0.04)
    ir1 = y + inset_r
    ir2 = y + h - inset_r
    ic1 = x + inset_c
    ic2 = x + w - inset_c

    return ir1, ir2, ic1, ic2


def parse_plate_label(label):
    """Parse a plate label into genotype and optional condition.

    If label contains '_', splits into (genotype, condition).
    Otherwise returns (genotype, None).

    Examples:
        'WT' -> ('WT', None)
        'Col-0_Drought' -> ('Col-0', 'Drought')
        'crd-2_Salt' -> ('crd-2', 'Salt')
    """
    if '_' in label:
        parts = label.split('_', 1)  # split only on first underscore
        return (parts[0].strip(), parts[1].strip())
    else:
        return (label.strip(), None)


def prompt_plate_labels(num_plates, plate_offset=0, split_plate=False):
    """Prompt user to enter genotype (or genotype_condition) for each plate.

    Args:
        num_plates: number of plates in this image
        plate_offset: starting plate number (0-based) for continuous numbering (CSV only)
        split_plate: if True, ask for 2 genotypes per plate

    Returns:
        labels: list of tuples [(genotype, condition), ...] where condition may be None.
            Normal mode: one entry per plate.
            Split-plate mode: two entries per plate (genotype A, genotype B).
        In split-plate mode with 1 plate: returns 2 labels.
        In split-plate mode with 2 plates: returns 4 labels.
    """
    print("\n--- Enter plate labels ---")
    print("  Format: 'Genotype' or 'Genotype_Condition'")
    print("  Examples: WT, Col-0_Drought, crd-2_Salt")
    print("  (All plates share the same layout)\n")
    labels = []
    if split_plate:
        label_a = input(f"    Genotype A (red): ").strip()
        label_b = input(f"    Genotype B (blue): ").strip()
        parsed_a = parse_plate_label(label_a)
        parsed_b = parse_plate_label(label_b)
        for i in range(num_plates):
            labels.append(parsed_a)
            labels.append(parsed_b)
    else:
        label = input(f"  Genotype: ").strip()
        parsed = parse_plate_label(label)
        for i in range(num_plates):
            labels.append(parsed)
    return labels


def crop_plates_to_interior(plates, image):
    """Crop each plate to just the agar interior, excluding ruler and dark edges.

    Returns the full plate interior (not cropped to root bounding box).
    """
    cropped = []
    for r1, r2, c1, c2 in plates:
        plate_img = image[r1:r2, c1:c2]

        # find plate interior boundary
        interior = _find_plate_interior(plate_img)
        if interior is None:
            cropped.append((r1, r2, c1, c2))
            continue

        ir1, ir2, ic1, ic2 = interior

        # return the full plate interior in full-image coordinates
        cropped.append((r1 + ir1, r1 + ir2, c1 + ic1, c1 + ic2))

    return cropped


def select_plates_interactive(image, downsample=4):
    """Show the full scan and let the user draw rectangles around plates.

    Draw a rectangle by click-dragging. Press Enter to confirm it.
    Draw another rectangle and press Enter again, or press Enter with
    no new rectangle to finish.

    Returns list of (r1, r2, c1, c2) plate regions in full-image coordinates,
    sorted top to bottom.
    """
    img_8 = _to_uint8(image[::downsample, ::downsample])

    plates = []
    current_rect = [None]  # mutable container for the callback
    colors = ['red', 'blue']

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.imshow(img_8, cmap='gray', aspect='equal')
    fig.suptitle("Draw a rectangle around a plate, then press Enter.\n"
                 "Press Enter again with no new rectangle to finish.",
                 fontsize=11)

    def on_select(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        # store as (r1, r2, c1, c2) in display coords
        r1_d = int(min(y1, y2))
        r2_d = int(max(y1, y2))
        c1_d = int(min(x1, x2))
        c2_d = int(max(x1, x2))
        current_rect[0] = (r1_d, r2_d, c1_d, c2_d)

    selector = RectangleSelector(ax, on_select, useblit=True,
                                  button=[1], interactive=True,
                                  props=dict(facecolor='red', alpha=0.2,
                                             edgecolor='red', linewidth=2))

    def on_key(event):
        if event.key != 'enter':
            return
        if current_rect[0] is not None:
            r1_d, r2_d, c1_d, c2_d = current_rect[0]
            # convert display coords to full-image coords
            r1 = r1_d * downsample
            r2 = r2_d * downsample
            c1 = c1_d * downsample
            c2 = c2_d * downsample
            plates.append((r1, r2, c1, c2))

            # draw permanent overlay
            color = colors[len(plates) - 1] if len(plates) <= len(colors) else 'green'
            rect_patch = Rectangle((c1_d, r1_d), c2_d - c1_d, r2_d - r1_d,
                                    linewidth=2, edgecolor=color,
                                    facecolor=color, alpha=0.15)
            ax.add_patch(rect_patch)
            ax.text(c1_d + 5, r1_d + 20, f"Plate {len(plates)}",
                    color=color, fontsize=12, fontweight='bold')

            current_rect[0] = None
            selector.set_active(True)

            fig.suptitle(f"{len(plates)} plate(s) selected. "
                         f"Draw another or press Enter to finish.",
                         fontsize=11)
            fig.canvas.draw_idle()
        else:
            # no new rectangle — finish
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    if not plates:
        print("  No plates selected.")
        return []

    # sort top to bottom
    plates.sort(key=lambda p: p[0])
    for i, (r1, r2, c1, c2) in enumerate(plates):
        print(f"  Plate {i + 1}: rows {r1}-{r2}, cols {c1}-{c2}  "
              f"({c2 - c1}x{r2 - r1} px)")

    return plates
