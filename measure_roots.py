#!/usr/bin/env python3
"""
Root Length Measurement Tool
============================
Measures primary root lengths from scanned agar plate images (TIF, PNG, etc.).

Usage:
    python measure_roots.py                  # opens file dialog
    python measure_roots.py image.tif        # direct path
    python measure_roots.py image.tif 472    # with custom resolution (px/cm)

Workflow:
    1. Plates are auto-detected and shown side by side (zoomed in)
    2. Left-click TOP then BOTTOM of each root (all roots in one session)
    3. Right-click to undo last point
    4. Press Enter when done
    5. Script traces all roots and shows results overlaid on the image
    6. CSV saved next to the input image
"""

import sys
from pathlib import Path

import numpy as np
import cv2
import tifffile
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from skimage.graph import route_through_array

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


# ---------------------------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Plate Detection
# ---------------------------------------------------------------------------
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
    print("  Examples: WT, Col-0_Drought, crd-2_Salt\n")
    labels = []
    for i in range(num_plates):
        local_plate = i + 1
        global_plate = plate_offset + i + 1
        if split_plate:
            print(f"  Plate {local_plate} (CSV: Plate {global_plate}) has 2 genotypes:")
            label_a = input(f"    Genotype A (red): ").strip()
            label_b = input(f"    Genotype B (blue): ").strip()
            labels.append(parse_plate_label(label_a))
            labels.append(parse_plate_label(label_b))
        else:
            label = input(f"  Plate {local_plate} (CSV: Plate {global_plate}): ").strip()
            labels.append(parse_plate_label(label))
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


# ---------------------------------------------------------------------------
# Interactive Click Collector (side-by-side plates) — SINGLE CLICK MODE
# ---------------------------------------------------------------------------
class RootClickCollector:
    """Matplotlib event handler for collecting root top clicks and optional mark clicks.

    Supports 1 or 2 plates, and split-plate mode (2 genotypes per plate).

    Normal mode (num_marks=0):
        Single-click mode: just click the top of each root.
        For 2 plates: Enter advances plate, second Enter finishes.
        For 1 plate: Enter finishes immediately.

    Split-plate mode (split_plate=True):
        Each plate has 2 genotype groups (red and blue).
        Enter advances between groups, then to next plate.

    Multi-measurement mode (num_marks>0):
        Click tops → Enter → click marks → Enter, per group.
    """

    def __init__(self, fig, axes, plates, display_scale, plate_labels=None,
                 plate_offset=0, num_marks=0, split_plate=False):
        self.fig = fig
        self.axes = axes if isinstance(axes, list) else [axes]
        self.num_plates = len(self.axes)
        self.plates = plates
        self.scale = display_scale
        self.plate_offset = plate_offset
        self.num_marks = num_marks
        self.split_plate = split_plate
        self.points = []
        self.point_plates = []
        self.mark_points = []
        self.mark_plates = []
        self.artists = []
        self.mark_artists = []
        self.finished = False
        self.clicking_marks = False
        self.plate_colors = ['red', 'blue']

        # In split-plate mode, each physical plate has 2 genotype groups.
        # num_groups = total number of Enter-delimited clicking stages.
        # current_group cycles through all groups across all plates.
        if split_plate:
            self.num_groups = self.num_plates * 2
        else:
            self.num_groups = self.num_plates
        self.current_group = 0

        # plate_labels: one entry per group
        if plate_labels:
            self.plate_labels = plate_labels
        else:
            self.plate_labels = [("", None)] * self.num_groups

        self.cid_click = fig.canvas.mpl_connect('button_press_event',
                                                self._on_click)
        self.cid_key = fig.canvas.mpl_connect('key_press_event',
                                              self._on_key)

        self._update_title()
        self._highlight_current()

    def _group_to_plate(self, group):
        """Map a group index to physical plate index (axes index)."""
        if self.split_plate:
            return group // 2
        return group

    def _group_color(self, group):
        """Get color for a group. In split-plate mode, alternates red/blue per plate."""
        if self.split_plate:
            return self.plate_colors[group % 2]
        return self.plate_colors[group % len(self.plate_colors)]

    def _group_label_name(self, group):
        """Get formatted label for a group."""
        if group < len(self.plate_labels):
            return _format_plate_label(self.plate_labels[group])
        return ""

    def _current_ax(self):
        """Get the axes for the current group."""
        plate_idx = self._group_to_plate(self.current_group)
        return self.axes[plate_idx]

    def _display_to_full(self, ax, dcol, drow):
        """Convert display coordinates on a specific axes to full-image coords."""
        idx = self.axes.index(ax)
        r1, r2, c1, c2 = self.plates[idx]
        full_row = int(drow * self.scale) + r1
        full_col = int(dcol * self.scale) + c1
        return full_row, full_col

    def _highlight_current(self):
        """Highlight the current plate being clicked with the group color."""
        current_plate_idx = self._group_to_plate(self.current_group)
        color = self._group_color(self.current_group)
        for i, ax in enumerate(self.axes):
            if i == current_plate_idx:
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(5)
            else:
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(1)
        self.fig.canvas.draw_idle()

    def _count_tops_for_group(self, group):
        """Count how many top points belong to a specific group."""
        return sum(1 for p in self.point_plates if p == group)

    def _count_marks_for_group(self, group):
        """Count how many mark points belong to a specific group."""
        return sum(1 for p in self.mark_plates if p == group)

    def _expected_marks_for_group(self, group):
        """How many marks are expected for a group."""
        return self._count_tops_for_group(group) * self.num_marks

    # -- events --------------------------------------------------------------

    def _on_click(self, event):
        if event.inaxes not in self.axes:
            return

        # In split-plate mode, clicks go to the current group's plate
        # In normal mode, clicks must be on the active plate
        current_plate_idx = self._group_to_plate(self.current_group)
        clicked_plate_idx = self.axes.index(event.inaxes)
        if clicked_plate_idx != current_plate_idx:
            return

        if event.button == 1:  # left click — add point
            ax = event.inaxes
            dcol, drow = event.xdata, event.ydata
            full_row, full_col = self._display_to_full(ax, dcol, drow)

            if not self.clicking_marks:
                # --- clicking tops ---
                self.points.append((full_row, full_col))
                self.point_plates.append(self.current_group)

                group_count = self._count_tops_for_group(self.current_group)
                color = self._group_color(self.current_group)
                label = f"{group_count}"

                marker = ax.plot(dcol, drow, 'o', color=color,
                                 markersize=8, markeredgecolor='white',
                                 markeredgewidth=1)[0]
                text = ax.text(dcol + 8, drow - 8, label,
                               color=color, fontsize=9, fontweight='bold')
                self.artists.append([marker, text])
            else:
                # --- clicking marks ---
                expected = self._expected_marks_for_group(self.current_group)
                current_marks = self._count_marks_for_group(self.current_group)
                if current_marks >= expected:
                    return

                self.mark_points.append((full_row, full_col))
                self.mark_plates.append(self.current_group)

                color = self._group_color(self.current_group)
                marker = ax.plot(dcol, drow, 'x', color=color,
                                 markersize=10, markeredgewidth=2)[0]
                self.mark_artists.append([marker])

            self._update_title()
            self.fig.canvas.draw_idle()

        elif event.button == 3:  # right click — undo
            if self.clicking_marks:
                if self.mark_points and self.mark_plates and self.mark_plates[-1] == self.current_group:
                    self.mark_points.pop()
                    self.mark_plates.pop()
                    if self.mark_artists:
                        for a in self.mark_artists.pop():
                            a.remove()
                    self._update_title()
                    self.fig.canvas.draw_idle()
            else:
                if self.points and self.point_plates and self.point_plates[-1] == self.current_group:
                    self.points.pop()
                    self.point_plates.pop()
                    if self.artists:
                        for a in self.artists.pop():
                            a.remove()
                    self._update_title()
                    self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == 'enter':
            last_group = self.num_groups - 1
            if self.num_marks == 0:
                # --- normal mode (no marks) ---
                if self.current_group < last_group:
                    self.current_group += 1
                    self._highlight_current()
                    self._update_title()
                else:
                    self.finished = True
                    plt.close(self.fig)
            else:
                # --- multi-measurement mode ---
                if not self.clicking_marks:
                    self.clicking_marks = True
                    self._update_title()
                elif self.current_group < last_group:
                    self.clicking_marks = False
                    self.current_group += 1
                    self._highlight_current()
                    self._update_title()
                else:
                    self.finished = True
                    plt.close(self.fig)

    # -- helpers --------------------------------------------------------------

    def _update_title(self):
        n_tops = len(self.points)
        last_group = self.num_groups - 1
        group_label = self._group_label_name(self.current_group)

        if self.split_plate:
            plate_num = self._group_to_plate(self.current_group) + 1
            geno_num = (self.current_group % 2) + 1
            location = f"Plate {plate_num}, Genotype {geno_num}"
            if group_label:
                location += f" ({group_label})"
        else:
            plate_num = self.current_group + 1
            location = f"Plate {plate_num}"

        if self.num_marks == 0:
            if self.current_group < last_group:
                self.fig.suptitle(
                    f"{n_tops} root(s) marked.  Clicking {location}.  "
                    f"Press Enter when done.",
                    fontsize=11)
            else:
                self.fig.suptitle(
                    f"{n_tops} root(s) marked.  Clicking {location}.  "
                    f"Press Enter to finish and measure.",
                    fontsize=11)
        else:
            if not self.clicking_marks:
                group_tops = self._count_tops_for_group(self.current_group)
                self.fig.suptitle(
                    f"{location} — Click root TOPS ({group_tops} so far).  "
                    f"Press Enter when done clicking tops.",
                    fontsize=11)
            else:
                group_marks = self._count_marks_for_group(self.current_group)
                expected = self._expected_marks_for_group(self.current_group)
                self.fig.suptitle(
                    f"{location} — Click MARKS on roots "
                    f"({group_marks}/{expected}).  "
                    f"{self.num_marks} mark(s) per root, same order as tops.  "
                    f"Press Enter when done.",
                    fontsize=11)

    def get_top_points(self):
        """Return list of (row, col) top points."""
        return list(self.points)

    def get_point_plates(self):
        """Return list of group indices for each top point."""
        return list(self.point_plates)

    def get_mark_points(self):
        """Return list of (row, col) mark points."""
        return list(self.mark_points)

    def get_mark_plates(self):
        """Return list of plate indices (0 or 1) for each mark point."""
        return list(self.mark_plates)


def _format_plate_label(label_tuple):
    """Format a (genotype, condition) tuple for display."""
    if label_tuple is None:
        return ""
    genotype, condition = label_tuple
    if condition:
        return f"{genotype}_{condition}"
    return genotype


def show_image_for_clicking(image, plates, plate_labels=None, plate_offset=0,
                            downsample=DISPLAY_DOWNSAMPLE, num_marks=0,
                            split_plate=False):
    """Display cropped plates and collect top clicks (and optional marks).

    Handles 1 or 2 plates. Single plate shown alone; two plates shown side by side.
    In split-plate mode, each plate has 2 genotype groups (red/blue).

    Returns:
        (top_points, point_plates, mark_points, mark_plates)
    """
    num_plates = len(plates)
    crops_8 = []
    for r1, r2, c1, c2 in plates:
        crop = image[r1:r2, c1:c2]
        small = crop[::downsample, ::downsample]
        crops_8.append(_to_uint8(small))

    fig, axes = plt.subplots(1, num_plates, figsize=(9 * num_plates, 10))
    if num_plates == 1:
        axes = [axes]
    else:
        axes = axes.tolist()

    for i, (ax, img) in enumerate(zip(axes, crops_8)):
        ax.imshow(img, cmap='gray', aspect='equal')
        local_plate = i + 1
        if split_plate and plate_labels:
            # show both genotype labels for this plate
            label_a = _format_plate_label(plate_labels[i * 2])
            label_b = _format_plate_label(plate_labels[i * 2 + 1])
            title = f"Plate {local_plate}: {label_a} (red) / {label_b} (blue)"
        elif plate_labels:
            label = _format_plate_label(plate_labels[i])
            title = f"Plate {local_plate}: {label}" if label else f"Plate {local_plate}"
        else:
            title = f"Plate {local_plate}"
        ax.set_title(title, fontsize=10)

    collector = RootClickCollector(fig, axes, plates, downsample,
                                   plate_labels, plate_offset, num_marks,
                                   split_plate)
    plt.tight_layout()
    plt.show()

    return (collector.get_top_points(), collector.get_point_plates(),
            collector.get_mark_points(), collector.get_mark_plates())


# ---------------------------------------------------------------------------
# Root Tracing
# ---------------------------------------------------------------------------
def find_root_tip(binary_image, top_point, scale=SCALE_PX_PER_CM):
    """Find the root tip (bottom endpoint) starting from a top click.

    Traces the skeleton downward from the starting point to find the
    furthest endpoint (the root tip).

    Returns (tip_row, tip_col) in full image coordinates, or None if not found.
    """
    h, w = binary_image.shape

    # validate bounds
    if not (0 <= top_point[0] < h and 0 <= top_point[1] < w):
        return None

    # compute DPI-dynamic ROI limits
    half_w = roi_half_width_px(scale)
    vert = roi_vertical_px(scale)

    # extract ROI starting at the click point — downward only
    # no pixels above the click, so tracing can only go down
    rmin = max(0, top_point[0] - 10)    # tiny margin for click imprecision
    rmax = min(h, top_point[0] + vert)  # roots up to ~10 cm (plate limit)
    cmin = max(0, top_point[1] - half_w)
    cmax = min(w, top_point[1] + half_w)

    roi = binary_image[rmin:rmax, cmin:cmax]
    top_local = (top_point[0] - rmin, top_point[1] - cmin)

    # skeletonize the ROI
    skeleton = skeletonize(roi)
    if skeleton.sum() == 0:
        return None

    skel_points = np.argwhere(skeleton)
    if len(skel_points) == 0:
        return None

    # build graph
    coord_to_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(skel_points)}

    edges = []
    for i, (r, c) in enumerate(skel_points):
        r, c = int(r), int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                j = coord_to_idx.get((r + dr, c + dc))
                if j is not None and j > i:
                    dist = np.sqrt(2) if (dr != 0 and dc != 0) else 1.0
                    edges.append((i, j, dist))

    G = nx.Graph()
    G.add_nodes_from(range(len(skel_points)))
    G.add_weighted_edges_from(edges)

    # find component sizes
    node_comp_size = {}
    for comp in nx.connected_components(G):
        sz = len(comp)
        for n in comp:
            node_comp_size[n] = sz

    # snap click to nearest skeleton point in a large component
    tree = cKDTree(skel_points)
    dist, idx = tree.query(top_local, k=min(100, len(skel_points)))
    if np.isscalar(dist):
        dist, idx = [dist], [idx]

    start_idx = None
    max_dist = max_click_distance_px(scale)
    for d, i in zip(dist, idx):
        if d > max_dist:
            break
        if node_comp_size.get(int(i), 0) >= MIN_COMPONENT_SIZE:
            start_idx = int(i)
            break

    if start_idx is None:
        return None

    # find all endpoints (degree 1 nodes) in the same component
    component = nx.node_connected_component(G, start_idx)
    endpoints = [n for n in component if G.degree(n) == 1]

    if not endpoints:
        # no clear endpoints — use the furthest point from start
        lengths = nx.single_source_dijkstra_path_length(G, start_idx, weight='weight')
        furthest = max((n for n in component), key=lambda n: lengths.get(n, 0))
        tip_local = skel_points[furthest]
    else:
        # find the endpoint furthest down (max row) that's reachable from start
        # prefer endpoints that are below the start point
        start_row = skel_points[start_idx][0]
        below_endpoints = [n for n in endpoints if skel_points[n][0] > start_row]

        if below_endpoints:
            # pick the one furthest down
            tip_idx = max(below_endpoints, key=lambda n: skel_points[n][0])
        else:
            # fallback: furthest endpoint by path length
            lengths = nx.single_source_dijkstra_path_length(G, start_idx, weight='weight')
            tip_idx = max(endpoints, key=lambda n: lengths.get(n, 0))

        tip_local = skel_points[tip_idx]

    # convert back to full image coordinates
    tip_full = (tip_local[0] + rmin, tip_local[1] + cmin)
    return tip_full


def _snap_to_large_component(tree, point, node_comp_size, min_size,
                             scale=SCALE_PX_PER_CM, k=200):
    """Find nearest skeleton pixel belonging to a component >= min_size."""
    k = min(k, tree.n)
    dists, indices = tree.query(point, k=k)
    if k == 1:
        dists = [dists]
        indices = [indices]
    max_dist = max_click_distance_px(scale)
    for d, i in zip(dists, indices):
        if d > max_dist:
            return None
        if node_comp_size.get(int(i), 0) >= min_size:
            return int(i)
    return None


def _try_skeleton_graph(skeleton, start, end, scale=SCALE_PX_PER_CM):
    """Phase 1: shortest path on skeleton graph. Returns path array or None."""
    skel_points = np.argwhere(skeleton)
    if len(skel_points) == 0:
        return None

    coord_to_idx = {}
    for i, (r, c) in enumerate(skel_points):
        coord_to_idx[(int(r), int(c))] = i

    edges = []
    for i, (r, c) in enumerate(skel_points):
        r, c = int(r), int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                j = coord_to_idx.get((r + dr, c + dc))
                if j is not None and j > i:
                    dist = np.sqrt(2) if (dr != 0 and dc != 0) else 1.0
                    edges.append((i, j, dist))

    G = nx.Graph()
    G.add_nodes_from(range(len(skel_points)))
    G.add_weighted_edges_from(edges)

    # component sizes
    node_comp_size = {}
    for comp in nx.connected_components(G):
        sz = len(comp)
        for n in comp:
            node_comp_size[n] = sz

    tree = cKDTree(skel_points)

    idx_start = _snap_to_large_component(tree, start, node_comp_size,
                                         MIN_COMPONENT_SIZE, scale=scale)
    idx_end = _snap_to_large_component(tree, end, node_comp_size,
                                       MIN_COMPONENT_SIZE, scale=scale)

    if idx_start is None or idx_end is None:
        return None
    if not nx.has_path(G, idx_start, idx_end):
        return None

    path_idx = nx.dijkstra_path(G, idx_start, idx_end, weight='weight')
    return skel_points[path_idx]


def _hybrid_cost_path(binary_roi, skeleton, start, end):
    """Phase 2 fallback: cost-map path preferring skeleton > root > background."""
    cost = np.full(binary_roi.shape, 10000.0, dtype=np.float64)
    cost[binary_roi] = 5.0
    cost[skeleton] = 1.0

    path_list, _ = route_through_array(cost, start, end, fully_connected=True)
    return np.array(path_list)


def trace_root(binary_image, top_point, bottom_point, scale=SCALE_PX_PER_CM):
    """Trace a single root between two clicked points.

    Returns dict with length_cm, length_px, path, method, warning.
    """
    h, w = binary_image.shape

    # validate bounds
    for label, pt in [('Top', top_point), ('Bottom', bottom_point)]:
        if not (0 <= pt[0] < h and 0 <= pt[1] < w):
            return dict(length_cm=0, length_px=0, path=np.empty((0, 2)),
                        method='error', warning=f'{label} point outside image')

    # extract local ROI (DPI-dynamic)
    pad = roi_pad_px(scale)
    half_w = roi_half_width_px(scale)
    rmin = max(0, min(top_point[0], bottom_point[0]) - pad)
    rmax = min(h, max(top_point[0], bottom_point[0]) + pad)
    cmin = max(0, min(top_point[1], bottom_point[1]) - half_w)
    cmax = min(w, max(top_point[1], bottom_point[1]) + half_w)

    roi = binary_image[rmin:rmax, cmin:cmax]
    top_local = (top_point[0] - rmin, top_point[1] - cmin)
    bot_local = (bottom_point[0] - rmin, bottom_point[1] - cmin)

    # skeletonize the ROI
    skeleton = skeletonize(roi)

    if skeleton.sum() == 0:
        return dict(length_cm=0, length_px=0, path=np.empty((0, 2)),
                    method='error', warning='No root skeleton found near clicks')

    # Phase 1: skeleton graph
    path_local = _try_skeleton_graph(skeleton, top_local, bot_local, scale=scale)
    method = 'skeleton_graph'

    # Phase 2: fallback
    if path_local is None:
        path_local = _hybrid_cost_path(roi, skeleton, top_local, bot_local)
        method = 'hybrid_cost'

    # convert to full-image coords
    path_full = path_local + np.array([rmin, cmin])

    # path length
    diffs = np.diff(path_full.astype(np.float64), axis=0)
    length_px = np.sqrt((diffs ** 2).sum(axis=1)).sum()
    length_cm = length_px / scale

    return dict(length_cm=length_cm, length_px=length_px, path=path_full,
                method=method, warning=None)


# ---------------------------------------------------------------------------
# Results Display
# ---------------------------------------------------------------------------
def show_results(image, results, plates, point_plates,
                 downsample=DISPLAY_DOWNSAMPLE, num_marks=0, split_plate=False):
    """Show traced paths overlaid on cropped plates.

    Uses red/blue shades for groups. In split-plate mode, both colors appear
    on the same plate for different genotypes.
    Segments are drawn in alternating light/dark shades of the group color.
    No text labels on the overlay — just colored traces.
    Handles 1 or 2 plates.
    """
    # base colors and alternating shades for segments (indexed by group)
    group_base_colors = [
        [(0.9, 0.2, 0.2), (1.0, 0.5, 0.5)],   # Group 0: dark red, light red
        [(0.2, 0.2, 0.9), (0.5, 0.5, 1.0)],    # Group 1: dark blue, light blue
    ]

    num_plates = len(plates)
    crops_8 = []
    for r1, r2, c1, c2 in plates:
        crop = image[r1:r2, c1:c2]
        small = crop[::downsample, ::downsample]
        crops_8.append(_to_uint8(small))

    fig, axes = plt.subplots(1, num_plates, figsize=(9 * num_plates, 10))
    if num_plates == 1:
        axes = [axes]
    else:
        axes = axes.tolist()

    for i, (ax, img) in enumerate(zip(axes, crops_8)):
        ax.imshow(img, cmap='gray', aspect='equal')
        local_plate = i + 1
        ax.set_title(f"Plate {local_plate}", fontsize=10)

    for i, res in enumerate(results):
        if res['path'].size == 0:
            continue
        path = res['path']

        group_idx = point_plates[i] if i < len(point_plates) else 0
        # map group to physical plate for display
        if split_plate:
            plate_idx = group_idx // 2
            color_idx = group_idx % 2
        else:
            plate_idx = group_idx
            color_idx = group_idx % len(group_base_colors)
        shades = group_base_colors[color_idx]

        r1, r2, c1, c2 = plates[plate_idx]
        dp_row = (path[:, 0] - r1) / downsample
        dp_col = (path[:, 1] - c1) / downsample

        segments = res.get('segments', [])
        if segments and len(segments) > 1:
            # draw each segment in alternating shades
            # reconstruct segment boundaries from cumulative arc length
            mark_indices = []
            if 'mark_coords' in res:
                for mc in res['mark_coords']:
                    mark_indices.append(_find_nearest_path_index(path, mc))
                mark_indices.sort()
            else:
                # approximate: divide path proportionally by segment lengths
                total_len = sum(segments)
                cum = 0
                for seg_len in segments[:-1]:
                    cum += seg_len
                    frac = cum / total_len if total_len > 0 else 0
                    mark_indices.append(int(frac * (len(path) - 1)))

            boundaries = [0] + mark_indices + [len(path) - 1]
            for j in range(len(boundaries) - 1):
                start = boundaries[j]
                end = boundaries[j + 1] + 1
                color = shades[j % len(shades)]
                axes[plate_idx].plot(dp_col[start:end], dp_row[start:end], '-',
                                     color=color, linewidth=2.5, alpha=0.9)
        else:
            # single color for whole root
            axes[plate_idx].plot(dp_col, dp_row, '-', color=shades[0],
                                 linewidth=2, alpha=0.85)

    fig.suptitle(
        f"Traced {len(results)} root(s).  Close window to continue.",
        fontsize=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CSV Output
# ---------------------------------------------------------------------------
def append_results_to_csv(results, csv_path, plates, plate_labels, plate_offset,
                          root_offset, point_plates, num_marks=0,
                          split_plate=False):
    """Append measurements to a shared CSV file.

    Args:
        results: list of measurement dicts from trace_root
        csv_path: Path to the data.csv file
        plates: list of (r1, r2, c1, c2) plate regions
        plate_labels: list of (genotype, condition) tuples
        plate_offset: starting plate number (0-based, so first image = 0)
        root_offset: starting root number (0-based)
        point_plates: list indicating which group each root belongs to
        num_marks: number of marks per root (0 = normal mode)
        split_plate: if True, point_plates stores group indices (2 per plate)

    Returns:
        (new_plate_offset, new_root_offset) for the next image
    """
    # check if this is a factorial design (any plate has a condition)
    is_factorial = plate_labels and any(cond is not None for (geno, cond) in plate_labels)

    rows = []
    for i, r in enumerate(results):
        root_num = root_offset + i + 1
        row = {
            'Root_ID': root_num,
            'Length_cm': round(r['length_cm'], 3),
            'Length_px': round(r['length_px'], 1),
            'Warning': r['warning'] or '',
        }

        if plate_labels and i < len(point_plates):
            group_idx = point_plates[i]
            if split_plate:
                # group_idx 0,1 = plate 0 genotypes; 2,3 = plate 1 genotypes
                physical_plate = group_idx // 2
            else:
                physical_plate = group_idx
            row['Plate'] = plate_offset + physical_plate + 1
            genotype, condition = plate_labels[group_idx]
            row['Genotype'] = genotype
            if is_factorial:
                row['Condition'] = condition or ''

        # add segment columns if multi-measurement mode
        segments = r.get('segments', [])
        if num_marks > 0:
            # num_marks marks divide the root into (num_marks + 1) segments
            for seg_i in range(num_marks + 1):
                col_name = f'Segment_{seg_i + 1}_cm'
                if seg_i < len(segments):
                    row[col_name] = round(segments[seg_i], 3)
                else:
                    row[col_name] = ''

        rows.append(row)

    # set column order based on experiment type
    if is_factorial:
        col_order = ['Root_ID', 'Plate', 'Genotype', 'Condition', 'Length_cm']
    else:
        col_order = ['Root_ID', 'Plate', 'Genotype', 'Length_cm']

    # add segment columns
    if num_marks > 0:
        for seg_i in range(num_marks + 1):
            col_order.append(f'Segment_{seg_i + 1}_cm')

    col_order.extend(['Length_px', 'Warning'])

    df_new = pd.DataFrame(rows)

    # if file exists, append; otherwise create
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        # ensure columns match
        for col in col_order:
            if col not in df_existing.columns:
                df_existing[col] = ''
        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing[col] = ''
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    # only include columns that exist
    col_order = [c for c in col_order if c in df.columns]
    # add any extra columns from existing file that aren't in col_order
    for col in df.columns:
        if col not in col_order:
            col_order.append(col)
    df = df[col_order]
    df.to_csv(csv_path, index=False)
    print(f"\nResults appended to: {csv_path}")

    # return updated offsets
    new_plate_offset = plate_offset + len(plates)
    new_root_offset = root_offset + len(results)
    return new_plate_offset, new_root_offset


# ---------------------------------------------------------------------------
# Image Listing and Selection
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}


def list_images_in_folder(folder):
    """Find all image files in a folder and return sorted list."""
    folder = Path(folder)
    images = []
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(f)
    return sorted(images, key=lambda p: p.name.lower())


def select_image_from_list(images, processed=None):
    """Display numbered list of images and let user select one.

    Args:
        images: list of Path objects
        processed: set of Path objects that have been processed (shown with checkmark)

    Returns the selected Path, or None if user wants to quit.
    """
    if processed is None:
        processed = set()

    done_count = len(processed)
    total_count = len(images)

    print("\n" + "=" * 50)
    print(f"  Available images: ({done_count}/{total_count} done)")
    print("=" * 50)
    for i, img in enumerate(images, 1):
        status = "   DONE" if img in processed else ""
        print(f"  {i:2d}.  {img.name:<25}{status}")
    print("=" * 50)
    print("  Enter number to process, or 'q' to quit")
    print("=" * 50)

    while True:
        choice = input("\nSelect image: ").strip().lower()
        if choice == 'q':
            return None
        try:
            idx = int(choice)
            if 1 <= idx <= len(images):
                return images[idx - 1]
            else:
                print(f"  Please enter a number between 1 and {len(images)}")
        except ValueError:
            print("  Invalid input. Enter a number or 'q' to quit.")


# ---------------------------------------------------------------------------
# Process Single Image
# ---------------------------------------------------------------------------
def _find_nearest_path_index(path, point):
    """Find the index of the path point closest to a given (row, col) point."""
    dists = np.sqrt(((path - np.array(point)) ** 2).sum(axis=1))
    return int(np.argmin(dists))


def _compute_segments(path, mark_coords, scale):
    """Compute segment lengths along a traced path at mark positions.

    Args:
        path: (N, 2) array of (row, col) coordinates along the root
        mark_coords: list of (row, col) mark positions for this root
        scale: pixels per cm

    Returns:
        list of segment lengths in cm:
        [top_to_mark1, mark1_to_mark2, ..., last_mark_to_tip]
        Total = sum of all segments.
    """
    if path.size == 0 or not mark_coords:
        return []

    # snap each mark to nearest path index
    mark_indices = sorted([_find_nearest_path_index(path, m) for m in mark_coords])

    # compute cumulative arc length along the path
    diffs = np.diff(path.astype(np.float64), axis=0)
    step_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.zeros(len(path))
    cumulative[1:] = np.cumsum(step_lengths)

    # segment boundaries: start, mark1, mark2, ..., end
    boundaries = [0] + mark_indices + [len(path) - 1]
    segments = []
    for j in range(len(boundaries) - 1):
        seg_px = cumulative[boundaries[j + 1]] - cumulative[boundaries[j]]
        segments.append(seg_px / scale)

    return segments


def process_image(image_path, scale, csv_path, plate_offset=0, root_offset=0,
                  num_marks=0, plates_per_scan=2, split_plate=False):
    """Process a single image: load, detect plates, click roots, measure, save.

    Args:
        image_path: Path to the image file
        scale: pixels per cm
        csv_path: Path to the shared data.csv file
        plate_offset: starting plate number (0-based)
        root_offset: starting root number (0-based)
        num_marks: number of marks per root (0 = normal mode)
        plates_per_scan: 1 or 2 plates per image
        split_plate: if True, each plate has 2 genotypes

    Returns:
        (success, new_plate_offset, new_root_offset) or (False, plate_offset, root_offset) on error
    """
    print(f"\nLoading: {image_path}")
    try:
        ext = Path(image_path).suffix.lower()
        if ext in ('.tif', '.tiff'):
            image = tifffile.imread(image_path)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Error: could not read {image_path}")
                return False, plate_offset, root_offset
    except Exception as e:
        print(f"Error loading image: {e}")
        return False, plate_offset, root_offset

    # handle non-2D images (RGB, RGBA, etc.)
    if image.ndim == 3:
        print(f"Image has {image.shape[2]} channels — converting to grayscale.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        print(f"Unexpected image shape: {image.shape}")
        return False, plate_offset, root_offset

    print(f"Image: {image.shape[1]}x{image.shape[0]}, {image.dtype}")
    print(f"Scale: {scale:.1f} px/cm")

    # --- detect plates ---
    print("Detecting plates...")
    if plates_per_scan == 1:
        plates = detect_single_plate(image)
    else:
        plates = detect_two_plates(image)
    for i, (r1, r2, c1, c2) in enumerate(plates):
        local_plate = i + 1  # display as 1 or 2
        print(f"  Plate {local_plate}: rows {r1}-{r2}, cols {c1}-{c2}  "
              f"({(c2 - c1)}x{(r2 - r1)} px)")

    # --- get plate labels from user ---
    plate_labels = prompt_plate_labels(len(plates), plate_offset, split_plate)

    # --- preprocess ---
    print("Preprocessing...")
    binary = preprocess(image)
    print(f"Root pixels: {binary.sum():,} / {binary.size:,} "
          f"({100 * binary.sum() / binary.size:.1f}%)")

    # --- crop plates to interior (exclude ruler/edges) ---
    print("Cropping to plate interior...")
    plates = crop_plates_to_interior(plates, image)
    for i, (r1, r2, c1, c2) in enumerate(plates):
        local_plate = i + 1  # display as 1 or 2
        print(f"  Plate {local_plate} interior: rows {r1}-{r2}, cols {c1}-{c2}  "
              f"({(c2 - c1)}x{(r2 - r1)} px)")

    # --- interactive clicking ---
    print("\n--- Instructions ---")
    print("  Left-click:   mark TOP of each root (tip auto-detected)")
    if num_marks > 0:
        print(f"  Multi-measurement: {num_marks} mark(s) per root")
        print("  Workflow: click all tops → Enter → click marks (same order) → Enter")
    print("  Right-click:  undo last click")
    print("  Enter:        move to next stage / finish")
    print("  Scroll/pan:   use matplotlib toolbar to zoom\n")

    top_points, point_plates, mark_points, mark_plates = show_image_for_clicking(
        image, plates, plate_labels, plate_offset, num_marks=num_marks,
        split_plate=split_plate)

    if not top_points:
        print("No roots marked. Skipping this image.")
        # still increment plate offset even if no roots marked
        new_plate_offset = plate_offset + len(plates)
        return True, new_plate_offset, root_offset

    # --- organize marks per root (by click order within each group) ---
    # marks are clicked in the same order as tops, num_marks per root
    num_groups = len(plates) * 2 if split_plate else len(plates)
    root_marks = {}  # root_index -> list of (row, col) mark points
    if num_marks > 0 and mark_points:
        for group_idx in range(num_groups):
            # get tops for this group, in order
            group_top_indices = [i for i, p in enumerate(point_plates) if p == group_idx]
            # get marks for this group, in order
            group_mark_coords = [mark_points[i] for i, p in enumerate(mark_plates)
                                 if p == group_idx]
            # assign num_marks marks to each root
            for j, root_i in enumerate(group_top_indices):
                start = j * num_marks
                end = start + num_marks
                root_marks[root_i] = group_mark_coords[start:end]

    # --- find root tips and trace ---
    print(f"\nProcessing {len(top_points)} root(s)...")
    results = []
    for i, top in enumerate(top_points):
        # show root number per group (local numbering for display)
        group_idx = point_plates[i]
        group_root_num = sum(1 for j in range(i + 1) if point_plates[j] == group_idx)
        if split_plate:
            phys_plate = group_idx // 2 + 1
            geno_label = _format_plate_label(plate_labels[group_idx])
            print(f"  Plate {phys_plate} ({geno_label}), Root {group_root_num}...", end=" ", flush=True)
        else:
            print(f"  Plate {group_idx + 1}, Root {group_root_num}...", end=" ", flush=True)

        # auto-detect the root tip
        tip = find_root_tip(binary, top, scale=scale)
        if tip is None:
            print("WARNING: Could not find root tip")
            res = dict(length_cm=0, length_px=0, path=np.empty((0, 2)),
                       method='error', warning='Could not find root tip',
                       segments=[])
            results.append(res)
            continue

        # trace from top to tip
        res = trace_root(binary, top, tip, scale)

        # compute segments if marks exist for this root
        if i in root_marks and root_marks[i] and res['path'].size > 0:
            segments = _compute_segments(res['path'], root_marks[i], scale)
            res['segments'] = segments
            res['mark_coords'] = root_marks[i]  # store for display
            seg_str = " + ".join(f"{s:.2f}" for s in segments)
            print(f"{res['length_cm']:.2f} cm  (segments: {seg_str})")
        else:
            res['segments'] = []
            if res['warning']:
                print(f"WARNING: {res['warning']}")
            else:
                print(f"{res['length_cm']:.2f} cm")

        results.append(res)

    # --- show results ---
    show_results(image, results, plates, point_plates, num_marks=num_marks,
                 split_plate=split_plate)

    # --- save CSV ---
    new_plate_offset, new_root_offset = append_results_to_csv(
        results, csv_path, plates, plate_labels, plate_offset, root_offset,
        point_plates, num_marks=num_marks, split_plate=split_plate)

    # --- summary ---
    print(f"\n{'=' * 55}")
    print(f"  Summary: {len(results)} root(s) measured")
    print(f"{'=' * 55}")
    for i, r in enumerate(results):
        group_idx = point_plates[i]
        group_root_num = sum(1 for j in range(i + 1) if point_plates[j] == group_idx)
        if r['warning']:
            status = r['warning']
        elif r.get('segments'):
            seg_str = " + ".join(f"{s:.2f}" for s in r['segments'])
            status = f"{r['length_cm']:.2f} cm  (segments: {seg_str})"
        else:
            status = f"{r['length_cm']:.2f} cm"
        if split_plate:
            phys_plate = group_idx // 2 + 1
            geno_label = _format_plate_label(plate_labels[group_idx])
            print(f"  Plate {phys_plate} ({geno_label}), Root {group_root_num}: {status}")
        else:
            print(f"  Plate {group_idx + 1}, Root {group_root_num}: {status}")
    print(f"{'=' * 55}")

    return True, new_plate_offset, new_root_offset


def prompt_for_dpi():
    """Prompt user for scan DPI and return scale in px/cm."""
    print("\n" + "=" * 60)
    print("  Enter scan DPI (dots per inch)")
    print("  Common values: 600, 1200, 2400")
    print("  Press Enter for default (1200 DPI)")
    print("=" * 60)

    while True:
        response = input("\nDPI: ").strip()
        if response == "":
            dpi = 1200
            break
        try:
            dpi = int(response)
            if dpi > 0:
                break
            else:
                print("  Please enter a positive number.")
        except ValueError:
            print("  Invalid input. Enter a number like 1200.")

    # convert DPI to pixels per cm (1 inch = 2.54 cm)
    scale = dpi / 2.54
    print(f"  Using {dpi} DPI = {scale:.1f} pixels/cm")
    return scale


def prompt_for_plate_count():
    """Prompt user for number of plates per scan (1 or 2) and split-plate mode.

    Returns:
        (plates_per_scan, split_plate): int, bool
        split_plate means each plate has 2 genotypes side by side
    """
    print("\n" + "=" * 60)
    print("  How many plates per scan?")
    print("  1 = single plate per image")
    print("  2 = two plates stacked vertically (default)")
    print("=" * 60)

    while True:
        response = input("\n  Plates per scan (1 or 2, default: 2): ").strip()
        if response == "":
            plates_per_scan = 2
            break
        if response in ('1', '2'):
            plates_per_scan = int(response)
            break
        print("  Please enter 1 or 2.")

    print(f"  Using {plates_per_scan} plate(s) per scan.")

    # ask about split-plate (2 genotypes per plate)
    print("\n" + "=" * 60)
    print("  Split-plate mode")
    print("  Use this if each plate has 2 genotypes side by side")
    print("=" * 60)

    while True:
        response = input("\n  Two genotypes per plate? (y/n, default: n): ").strip().lower()
        if response in ('', 'n', 'no'):
            return plates_per_scan, False
        elif response in ('y', 'yes'):
            print("  Split-plate mode enabled: 2 genotypes per plate.")
            return plates_per_scan, True
        print("  Please enter y or n.")


def prompt_for_multi_measurement():
    """Prompt user for multi-measurement mode (segment marks on roots).

    Returns:
        num_marks: int, 0 = normal mode, >0 = number of marks per root
    """
    print("\n" + "=" * 60)
    print("  Multi-measurement mode")
    print("  Use this if roots have horizontal marks (e.g. transfer points)")
    print("  to measure segments between marks.")
    print("=" * 60)

    while True:
        response = input("\n  Multiple measurements per root? (y/n, default: n): ").strip().lower()
        if response in ('', 'n', 'no'):
            return 0
        elif response in ('y', 'yes'):
            break
        else:
            print("  Please enter y or n.")

    while True:
        response = input("  How many marks per root? (e.g. 1, 2, 3): ").strip()
        try:
            num_marks = int(response)
            if num_marks > 0:
                print(f"  Multi-measurement mode: {num_marks} mark(s) per root")
                print(f"  Workflow: click all root tops → Enter → click marks → Enter → next plate")
                return num_marks
            else:
                print("  Please enter a positive number.")
        except ValueError:
            print("  Invalid input. Enter a number like 1 or 2.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # determine working folder
    if len(sys.argv) >= 2:
        arg_path = Path(sys.argv[1])
        if arg_path.is_file():
            # single file mode — prompt for DPI and process
            scale = prompt_for_dpi()
            plates_per_scan, split_plate = prompt_for_plate_count()
            num_marks = prompt_for_multi_measurement()
            csv_path = arg_path.parent / 'data.csv'
            process_image(arg_path, scale, csv_path, 0, 0, num_marks,
                          plates_per_scan, split_plate)
            sys.exit(0)
        elif arg_path.is_dir():
            folder = arg_path
        else:
            print(f"Error: {sys.argv[1]} is not a valid file or directory")
            sys.exit(1)
    else:
        # default to script's directory
        folder = Path(__file__).parent

    # --- list images in folder ---
    images = list_images_in_folder(folder)
    if not images:
        print(f"No image files found in {folder}")
        sys.exit(0)

    print(f"\nFound {len(images)} image(s) in: {folder}")

    # --- prompt for DPI ---
    scale = prompt_for_dpi()

    # --- prompt for plate count ---
    plates_per_scan, split_plate = prompt_for_plate_count()

    # --- prompt for multi-measurement mode ---
    num_marks = prompt_for_multi_measurement()

    # --- setup shared CSV file ---
    csv_path = folder / 'data.csv'
    if csv_path.exists():
        print(f"\nNote: data.csv already exists. New data will be appended.")

    # --- main selection loop ---
    processed = set()
    plate_offset = 0
    root_offset = 0

    while True:
        # check if all images are done
        if len(processed) == len(images):
            print("\n" + "=" * 50)
            print("  All images have been processed!")
            print("=" * 50)
            print("  Enter number to re-process an image, or 'q' to quit")
            print("=" * 50)

        selected = select_image_from_list(images, processed)
        if selected is None:
            print(f"\nData saved to: {csv_path}")
            print("Goodbye!")
            break

        success, plate_offset, root_offset = process_image(
            selected, scale, csv_path, plate_offset, root_offset, num_marks,
            plates_per_scan, split_plate)
        processed.add(selected)


if __name__ == '__main__':
    main()
