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
    1. Draw rectangles around plates on the full scan
    2. Left-click TOP of each root (tip auto-detected)
    3. Right-click to undo last click
    4. Press Enter when done
    5. Script traces all roots and shows results overlaid on the image
    6. CSV saved to output/ folder
"""

import sys
import datetime
from pathlib import Path

import numpy as np
import cv2
import tifffile
import matplotlib
matplotlib.use('TkAgg')

from config import SCALE_PX_PER_CM
from image_processing import preprocess
from plate_detection import (select_plates_interactive,
                             prompt_plate_labels)
from click_collector import _format_plate_label, show_image_for_clicking
from root_tracing import find_root_tip, trace_root
from results_display import show_results
from csv_output import append_results_to_csv
from utils import list_images_in_folder, select_image_from_list, _compute_segments


def process_image(image_path, scale, csv_path, plate_offset=0, root_offset=0,
                  num_marks=0, split_plate=False, sensitivity='normal'):
    """Process a single image: select plates, click roots, measure, save.

    Args:
        image_path: Path to the image file
        scale: pixels per cm
        csv_path: Path to the shared data.csv file
        plate_offset: starting plate number (0-based)
        root_offset: starting root number (0-based)
        num_marks: number of marks per root (0 = normal mode)
        split_plate: if True, each plate has 2 genotypes
        sensitivity: 'thick', 'normal', or 'thin'

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

    # --- select plates interactively ---
    print("Select plates on the image...")
    plates = select_plates_interactive(image)
    if not plates:
        print("No plates selected. Skipping this image.")
        return True, plate_offset, root_offset

    # --- get plate labels from user ---
    plate_labels = prompt_plate_labels(len(plates), plate_offset, split_plate)

    # --- preprocess ---
    print(f"Preprocessing (sensitivity={sensitivity})...")
    binary = preprocess(image, scale=scale, sensitivity=sensitivity)
    print(f"Root pixels: {binary.sum():,} / {binary.size:,} "
          f"({100 * binary.sum() / binary.size:.1f}%)")

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


def prompt_for_split_plate():
    """Prompt user for split-plate mode (2 genotypes per plate).

    Returns:
        split_plate: bool
    """
    print("\n" + "=" * 60)
    print("  Split-plate mode")
    print("  Use this if each plate has 2 genotypes side by side")
    print("=" * 60)

    while True:
        response = input("\n  Two genotypes per plate? (y/n, default: n): ").strip().lower()
        if response in ('', 'n', 'no'):
            return False
        elif response in ('y', 'yes'):
            print("  Split-plate mode enabled: 2 genotypes per plate.")
            return True
        print("  Please enter y or n.")


def prompt_for_sensitivity():
    """Prompt user for root thickness / sensitivity preset.

    Returns:
        sensitivity: str ('thick', 'normal', or 'thin')
    """
    print("\n" + "=" * 60)
    print("  Root thickness")
    print("  1 = thick  (e.g. Setaria viridis, Brassica napus)")
    print("  2 = normal (e.g. Schrenkiella parvula, default)")
    print("  3 = thin   (e.g. Arabidopsis thaliana)")
    print("=" * 60)

    while True:
        response = input("\n  Root thickness (1/2/3, default: 2): ").strip()
        if response in ('', '2'):
            print("  Using normal sensitivity.")
            return 'normal'
        elif response == '1':
            print("  Using thick-root settings.")
            return 'thick'
        elif response == '3':
            print("  Using thin-root settings.")
            return 'thin'
        print("  Please enter 1, 2, or 3.")


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


def prompt_for_experiment():
    """Prompt user for experiment description, used in CSV filename."""
    print("\n" + "=" * 60)
    print("  Enter a short experiment description")
    print("  This will be used in the output filename")
    print("  Example: Salt Stress Col-0 vs crd-2")
    print("=" * 60)

    while True:
        desc = input("\nExperiment: ").strip()
        if desc:
            return desc
        print("  Please enter a description.")


def _build_csv_path(experiment_desc):
    """Build the output CSV path: output/YYYY-MM-DD - description.csv"""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'output'
    output_dir.mkdir(exist_ok=True)
    date_str = datetime.date.today().strftime('%Y-%m-%d')
    filename = f"{date_str} - {experiment_desc}.csv"
    return output_dir / filename


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _select_folder():
    """Open a folder picker dialog and return the selected Path."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askdirectory(title="Select folder containing scanned images")
    root.destroy()
    if not selected:
        return None
    return Path(selected)


def main():
    # --- gather experiment info first ---
    experiment_desc = prompt_for_experiment()
    scale = prompt_for_dpi()
    sensitivity = prompt_for_sensitivity()
    split_plate = prompt_for_split_plate()
    num_marks = prompt_for_multi_measurement()
    csv_path = _build_csv_path(experiment_desc)

    # --- determine working folder ---
    if len(sys.argv) >= 2:
        arg_path = Path(sys.argv[1])
        if arg_path.is_file():
            # single file mode
            process_image(arg_path, scale, csv_path, 0, 0, num_marks,
                          split_plate, sensitivity)
            sys.exit(0)
        elif arg_path.is_dir():
            folder = arg_path
        else:
            print(f"Error: {sys.argv[1]} is not a valid file or directory")
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("  Press Enter to select the folder with your scanned images")
        print("=" * 60)
        input()
        folder = _select_folder()
        if folder is None:
            print("No folder selected. Exiting.")
            sys.exit(0)

    # --- list images in folder ---
    images = list_images_in_folder(folder)
    if not images:
        print(f"No image files found in {folder}")
        sys.exit(0)

    print(f"\nFound {len(images)} image(s) in: {folder}")

    # --- setup shared CSV file ---
    csv_path = _build_csv_path(experiment_desc)
    if csv_path.exists():
        print(f"\nNote: {csv_path.name} already exists. New data will be appended.")

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
            split_plate, sensitivity)
        processed.add(selected)


if __name__ == '__main__':
    main()
