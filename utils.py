from pathlib import Path

import numpy as np


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
