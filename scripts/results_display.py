import numpy as np
import matplotlib.pyplot as plt

from config import DISPLAY_DOWNSAMPLE
from plate_detection import _to_uint8
from utils import _find_nearest_path_index


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
    plt.subplots_adjust(top=0.95)
    plt.show()
