import matplotlib.pyplot as plt

from config import DISPLAY_DOWNSAMPLE
from plate_detection import _to_uint8


def _format_plate_label(label_tuple):
    """Format a (genotype, condition) tuple for display."""
    if label_tuple is None:
        return ""
    genotype, condition = label_tuple
    if condition:
        return f"{genotype}_{condition}"
    return genotype


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
        self.point_flags = []      # None=normal, 'dead', 'touching'
        self.mark_points = []
        self.mark_plates = []
        self.artists = []
        self.mark_artists = []
        self.finished = False
        self.clicking_marks = False
        self._pending_flag = None   # 'dead' or 'touching' awaiting next click
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

        # store original view limits for home reset
        self._original_lims = {}
        for ax in self.axes:
            self._original_lims[id(ax)] = (ax.get_xlim(), ax.get_ylim())
        self._zoom_mode = False

        self.cid_click = fig.canvas.mpl_connect('button_press_event',
                                                self._on_click)
        self.cid_key = fig.canvas.mpl_connect('key_press_event',
                                              self._on_key)
        self.cid_scroll = fig.canvas.mpl_connect('scroll_event',
                                                  self._on_scroll)

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

    def _count_normal_tops_for_group(self, group):
        """Count how many normal (non-flagged) top points belong to a group."""
        return sum(1 for p, f in zip(self.point_plates, self.point_flags)
                   if p == group and f is None)

    def _expected_marks_for_group(self, group):
        """How many marks are expected for a group (only normal roots)."""
        return self._count_normal_tops_for_group(group) * self.num_marks

    # -- events --------------------------------------------------------------

    def _on_click(self, event):
        if self._zoom_mode:
            return
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

            if self._pending_flag is not None:
                # --- placing dead/touching marker at clicked location ---
                flag = self._pending_flag
                self._pending_flag = None
                label_text = 'DEAD' if flag == 'dead' else 'TOUCH'
                self.points.append((full_row, full_col))
                self.point_plates.append(self.current_group)
                self.point_flags.append(flag)

                group_count = self._count_tops_for_group(self.current_group)
                color = self._group_color(self.current_group)
                marker = ax.plot(dcol, drow, 'x', color=color,
                                 markersize=10, markeredgewidth=2)[0]
                text = ax.text(dcol + 8, drow - 8,
                               f"{group_count} {label_text}",
                               color=color, fontsize=9, fontweight='bold',
                               bbox=dict(facecolor='yellow', alpha=0.7))
                self.artists.append([marker, text])
            elif not self.clicking_marks:
                # --- clicking tops ---
                self.points.append((full_row, full_col))
                self.point_plates.append(self.current_group)
                self.point_flags.append(None)

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
            self._undo()

    def _undo(self):
        """Undo last click (top or mark) for the current group."""
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
                self.point_flags.pop()
                if self.artists:
                    for a in self.artists.pop():
                        a.remove()
                self._update_title()
                self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key in ('super+z', 'cmd+z', 'ctrl+z'):
            self._undo()
            return
        if event.key == 'z':
            # toggle zoom mode — left-click-drag to zoom rectangle
            self._zoom_mode = not self._zoom_mode
            if self._zoom_mode:
                self.fig.suptitle("ZOOM MODE: click and drag to zoom. Press Z to exit, H to reset.",
                                 fontsize=11)
                self.fig.canvas.toolbar.zoom()
            else:
                self.fig.canvas.toolbar.zoom()  # toggle off
                self._update_title()
            self.fig.canvas.draw_idle()
            return
        if event.key == 'h':
            # reset all axes to original view
            for ax in self.axes:
                xlim, ylim = self._original_lims[id(ax)]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            if self._zoom_mode:
                self._zoom_mode = False
                self.fig.canvas.toolbar.zoom()  # toggle off
            self._update_title()
            self.fig.canvas.draw_idle()
            return
        if event.key in ('d', 't') and not self.clicking_marks:
            # D = dead seedling, T = touching roots — next click places the marker
            self._pending_flag = 'dead' if event.key == 'd' else 'touching'
            label_text = 'DEAD' if self._pending_flag == 'dead' else 'TOUCH'
            self.fig.suptitle(
                f"Click on the {label_text} seedling to mark it.",
                fontsize=11)
            self.fig.canvas.draw_idle()
            return
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

    def _on_scroll(self, event):
        """Zoom in/out on scroll (trackpad or mouse wheel)."""
        if event.inaxes is None:
            return
        ax = event.inaxes
        scale_factor = 0.8 if event.button == 'up' else 1.25
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_w = (xlim[1] - xlim[0]) * scale_factor
        new_h = (ylim[1] - ylim[0]) * scale_factor
        ax.set_xlim([xdata - new_w * (xdata - xlim[0]) / (xlim[1] - xlim[0]),
                     xdata + new_w * (xlim[1] - xdata) / (xlim[1] - xlim[0])])
        ax.set_ylim([ydata - new_h * (ydata - ylim[0]) / (ylim[1] - ylim[0]),
                     ydata + new_h * (ylim[1] - ydata) / (ylim[1] - ylim[0])])
        self.fig.canvas.draw_idle()

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

    def get_point_flags(self):
        """Return list of flags: None=normal, 'dead', 'touching'."""
        return list(self.point_flags)

    def get_mark_points(self):
        """Return list of (row, col) mark points."""
        return list(self.mark_points)

    def get_mark_plates(self):
        """Return list of plate indices (0 or 1) for each mark point."""
        return list(self.mark_plates)

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
    plt.subplots_adjust(top=0.92)
    plt.show()

    return (collector.get_top_points(), collector.get_point_plates(),
            collector.get_point_flags(),
            collector.get_mark_points(), collector.get_mark_plates())


def show_manual_reclick(image, plates, root_labels, retry_indices=None,
                        results=None, point_plates=None, split_plate=False,
                        downsample=DISPLAY_DOWNSAMPLE, num_marks=0):
    """Open plate view for manual two-click re-tracing of specific roots.

    User clicks top then bottom of each root listed in root_labels.
    If num_marks > 0, also collects mark points for each root after top/bottom.
    Shows old bad traces in yellow so user knows which roots to fix.
    Returns (pairs, mark_dict) where pairs is list of (top, bottom) tuples
    and mark_dict maps root list index -> list of mark coords.
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
        ax.set_title(f"Plate {i + 1}", fontsize=10)

    # draw old bad traces in yellow so user can see which roots to re-click
    if results and retry_indices and point_plates is not None:
        for ri in retry_indices:
            res = results[ri]
            if res['path'].size == 0:
                continue
            path = res['path']
            group_idx = point_plates[ri]
            if split_plate:
                plate_idx = group_idx // 2
            else:
                plate_idx = group_idx
            r1, r2, c1, c2 = plates[plate_idx]
            dp_row = (path[:, 0] - r1) / downsample
            dp_col = (path[:, 1] - c1) / downsample
            axes[plate_idx].plot(dp_col, dp_row, '-',
                                 color=(1.0, 0.85, 0.0), linewidth=1.5, alpha=0.55)

    clicks = []         # list of (row, col) in full-image coords
    artists = []
    current_root = [0]  # mutable index into root_labels
    # phase: 'tops' = clicking top/bottom pairs, 'marks' = clicking mark points
    phase = ['tops']
    mark_clicks = []         # flat list of mark coords for current phase
    mark_dict = {}           # root list index -> list of mark coords
    marks_for_root = [0]     # which root index (in root_labels) we're collecting marks for

    def _to_full(ax, dcol, drow):
        idx = axes.index(ax)
        r1, r2, c1, c2 = plates[idx]
        return int(drow * downsample) + r1, int(dcol * downsample) + c1

    def _update():
        if phase[0] == 'tops':
            ri = current_root[0]
            if ri >= len(root_labels):
                if num_marks > 0:
                    fig.suptitle(
                        "All roots re-clicked.  Press Enter to continue to mark points.",
                        fontsize=11)
                else:
                    fig.suptitle("All roots re-clicked.  Press Enter to finish.",
                                 fontsize=11)
            elif len(clicks) % 2 == 0:
                fig.suptitle(
                    f"Root {root_labels[ri]}: click the TOP of the root.",
                    fontsize=11)
            else:
                fig.suptitle(
                    f"Root {root_labels[ri]}: click the BOTTOM (tip) of the root.",
                    fontsize=11)
        else:
            # marks phase
            mi = marks_for_root[0]
            if mi >= len(root_labels):
                fig.suptitle("All marks placed.  Press Enter to finish.",
                             fontsize=11)
            else:
                n_placed = len(mark_clicks)
                fig.suptitle(
                    f"Root {root_labels[mi]}: click mark {n_placed + 1} of "
                    f"{num_marks}.",
                    fontsize=11)
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes not in axes or event.button != 1:
            return
        ax = event.inaxes
        dcol, drow = event.xdata, event.ydata
        full_row, full_col = _to_full(ax, dcol, drow)

        if phase[0] == 'tops':
            if current_root[0] >= len(root_labels):
                return
            clicks.append((full_row, full_col))

            if len(clicks) % 2 == 1:
                # top click
                m = ax.plot(dcol, drow, 'o', color='green', markersize=8,
                            markeredgecolor='white', markeredgewidth=1)[0]
                artists.append([m])
            else:
                # bottom click — advance to next root
                m = ax.plot(dcol, drow, 's', color='green', markersize=8,
                            markeredgecolor='white', markeredgewidth=1)[0]
                artists[-1].append(m)
                current_root[0] += 1
        else:
            # marks phase
            mi = marks_for_root[0]
            if mi >= len(root_labels):
                return
            mark_clicks.append((full_row, full_col))
            m = ax.plot(dcol, drow, 'x', color='green', markersize=10,
                        markeredgewidth=2)[0]
            artists.append([m])
            if len(mark_clicks) >= num_marks:
                # done with this root's marks
                mark_dict[mi] = list(mark_clicks)
                mark_clicks.clear()
                marks_for_root[0] += 1

        _update()

    def _on_key(event):
        if event.key in ('super+z', 'cmd+z', 'ctrl+z'):
            if phase[0] == 'tops' and clicks:
                clicks.pop()
                if len(clicks) % 2 == 0 and current_root[0] > 0:
                    current_root[0] -= 1
                if artists:
                    last = artists[-1]
                    if len(clicks) % 2 == 0:
                        for a in artists.pop():
                            a.remove()
                    else:
                        last[-1].remove()
                        last.pop()
                _update()
            elif phase[0] == 'marks' and (mark_clicks or mark_dict):
                if mark_clicks:
                    mark_clicks.pop()
                    if artists:
                        for a in artists.pop():
                            a.remove()
                elif mark_dict and marks_for_root[0] > 0:
                    marks_for_root[0] -= 1
                    mi = marks_for_root[0]
                    restored = mark_dict.pop(mi)
                    mark_clicks.extend(restored)
                    mark_clicks.pop()
                    # remove last mark artist
                    if artists:
                        for a in artists.pop():
                            a.remove()
                _update()
        elif event.key == 'enter':
            if phase[0] == 'tops' and current_root[0] >= len(root_labels):
                if num_marks > 0:
                    phase[0] = 'marks'
                    marks_for_root[0] = 0
                    _update()
                else:
                    plt.close(fig)
            elif phase[0] == 'marks' and marks_for_root[0] >= len(root_labels):
                plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', _on_click)
    fig.canvas.mpl_connect('key_press_event', _on_key)
    _update()
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    # pair up clicks into (top, bottom) tuples
    pairs = []
    for j in range(0, len(clicks) - 1, 2):
        pairs.append((clicks[j], clicks[j + 1]))
    return pairs, mark_dict
