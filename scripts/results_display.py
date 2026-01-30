import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from config import DISPLAY_DOWNSAMPLE
from plate_detection import _to_uint8
from utils import _find_nearest_path_index


def show_results(image, results, plates, point_plates,
                 downsample=DISPLAY_DOWNSAMPLE, num_marks=0, split_plate=False):
    """Show traced paths overlaid on cropped plates. Interactive retry selection.

    Click on a trace to mark it for retry (turns yellow).
    Click again to deselect. Press Enter to finish.

    Returns list of root indices selected for retry (empty = accept all).
    """
    # base colors and alternating shades for segments (indexed by group)
    group_base_colors = [
        [(0.9, 0.2, 0.2), (1.0, 0.5, 0.5)],   # Group 0: dark red, light red
        [(0.2, 0.2, 0.9), (0.5, 0.5, 1.0)],    # Group 1: dark blue, light blue
    ]
    SELECT_COLOR = (1.0, 0.85, 0.0)  # yellow for selected

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
        ax.set_xticks([])
        ax.set_yticks([])
        local_plate = i + 1
        ax.set_title(f"Plate {local_plate}", fontsize=10)

    # draw traces and build lookup structures
    trace_artists = {}   # root_index -> list of Line2D artists
    trace_plate = {}     # root_index -> plate_idx
    trace_colors = {}    # root_index -> original color
    path_trees = {}      # root_index -> (cKDTree of dp_col/dp_row, plate_idx)

    for i, res in enumerate(results):
        if res['path'].size == 0:
            continue
        path = res['path']

        group_idx = point_plates[i] if i < len(point_plates) else 0
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

        # build spatial index for click detection
        pts = np.column_stack([dp_col, dp_row])
        path_trees[i] = (cKDTree(pts), plate_idx)

        artists = []
        segments = res.get('segments', [])
        if segments and len(segments) > 1:
            mark_indices = []
            if 'mark_coords' in res:
                for mc in res['mark_coords']:
                    mark_indices.append(_find_nearest_path_index(path, mc))
                mark_indices.sort()
            else:
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
                line = axes[plate_idx].plot(dp_col[start:end], dp_row[start:end], '-',
                                            color=color, linewidth=2.5, alpha=0.9)[0]
                artists.append(line)
        else:
            line = axes[plate_idx].plot(dp_col, dp_row, '-', color=shades[0],
                                        linewidth=2, alpha=0.85)[0]
            artists.append(line)

        trace_artists[i] = artists
        trace_plate[i] = plate_idx
        trace_colors[i] = shades[0]

    # interactive selection state
    selected = set()
    CLICK_THRESHOLD = 15  # pixels in display coords

    def _find_nearest_root(ax, x, y):
        """Find root index whose trace is nearest to click (x, y)."""
        plate_idx = axes.index(ax) if ax in axes else -1
        best_dist = CLICK_THRESHOLD
        best_root = None
        for ri, (tree, pidx) in path_trees.items():
            if pidx != plate_idx:
                continue
            d, _ = tree.query([x, y])
            if d < best_dist:
                best_dist = d
                best_root = ri
        return best_root

    def _update_title():
        if selected:
            fig.suptitle(
                f"Traced {len(results)} root(s).  "
                f"{len(selected)} selected for retry (yellow).  "
                f"Click to toggle.  Press Enter to retry.",
                fontsize=11)
        else:
            fig.suptitle(
                f"Traced {len(results)} root(s).  "
                f"Click a bad trace to retry it, or press Enter to accept all.",
                fontsize=11)
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes not in axes or event.button != 1:
            return
        ri = _find_nearest_root(event.inaxes, event.xdata, event.ydata)
        if ri is None:
            return
        if ri in selected:
            # deselect — restore original color
            selected.discard(ri)
            for line in trace_artists[ri]:
                line.set_color(trace_colors[ri])
                line.set_linewidth(2)
        else:
            # select — turn yellow
            selected.add(ri)
            for line in trace_artists[ri]:
                line.set_color(SELECT_COLOR)
                line.set_linewidth(4)
        _update_title()

    def _on_key(event):
        if event.key == 'enter':
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('key_press_event', _on_key)
    _update_title()
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    return sorted(selected)
