"""ImageCanvas — zoomable, pannable image canvas with overlay drawing."""

import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk


class ImageCanvas(ctk.CTkFrame):
    """Zoomable, pannable image canvas with overlay drawing."""

    # Interaction modes
    MODE_VIEW = "view"
    MODE_SELECT_PLATES = "select_plates"
    MODE_CLICK_ROOTS = "click_roots"
    MODE_CLICK_MARKS = "click_marks"
    MODE_REVIEW = "review"
    MODE_RECLICK = "reclick"

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = ctk.CTkCanvas(self, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._photo = None       # keep reference to prevent GC
        self._image_np = None    # original numpy array (uint8)
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._drag_start = None

        # interaction mode
        self._mode = self.MODE_VIEW
        self._on_click_callback = None
        self._on_done_callback = None

        # plate selection state
        self._rect_start = None  # image coords of rect drag start
        self._rect_drag_id = None  # canvas id of live drag rect
        self._plates = []        # list of (r1, r2, c1, c2) in image coords
        self._plate_rect_ids = []  # canvas ids of confirmed rects
        self._plates_count_at_enter = 0  # tracks plate count at last Enter

        # root clicking state
        self._root_points = []     # list of (row, col) in image coords
        self._root_flags = []      # None, 'dead', 'touching'
        self._root_groups = []     # group index per root (for split plate)
        self._root_marker_ids = [] # canvas ids of markers
        self._pending_flag = None  # 'dead' or 'touching'

        # mark clicking state (multi-measurement)
        self._mark_points = []     # list of (row, col) in image coords
        self._mark_marker_ids = [] # canvas ids of mark markers
        self._marks_expected = 0   # how many marks expected (stops clicks beyond this)
        self._all_marks = {}       # {root_index: [(row,col), ...]} all collected marks

        # trace overlay state
        self._traces = []          # list of (path_array, color_str)

        # review state (click to toggle retry selection)
        self._selected_for_retry = set()  # indices into _traces
        self._trace_original_colors = []  # original colors before selection

        # reclick state (top+bottom for retry roots)
        self._reclick_points = []  # list of (row, col) pairs
        self._reclick_expected = 0  # how many pairs expected
        self._reclick_marker_ids = []

        # bindings
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        # macOS trackpad: two-finger click = Button-2; physical right = Button-3
        self.canvas.bind("<ButtonPress-2>", self._on_right_click)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        # keyboard — bound at window level via bind_all in RootMeasureApp
        self.canvas.focus_set()

    # --- Mode management ---

    def set_mode(self, mode, on_done=None):
        """Set interaction mode."""
        self._mode = mode
        self._on_done_callback = on_done
        self.canvas.focus_set()

    def get_plates(self):
        return list(self._plates)

    def get_root_points(self):
        return list(self._root_points)

    def get_root_flags(self):
        return list(self._root_flags)

    def get_root_groups(self):
        return list(self._root_groups)

    def clear_plates(self):
        for rid in self._plate_rect_ids:
            self.canvas.delete(rid)
        self._plates.clear()
        self._plate_rect_ids.clear()

    def clear_roots(self):
        for rid in self._root_marker_ids:
            self.canvas.delete(rid)
        self._root_points.clear()
        self._root_flags.clear()
        self._root_groups.clear()
        self._root_marker_ids.clear()
        self._pending_flag = None

    def clear_marks(self):
        for rid in self._mark_marker_ids:
            self.canvas.delete(rid)
        self._mark_points.clear()
        self._mark_marker_ids.clear()

    def get_mark_points(self):
        return list(self._mark_points)

    def clear_traces(self):
        self._traces.clear()
        self._trace_original_colors.clear()
        self._selected_for_retry.clear()

    def add_trace(self, path, shades=None, mark_indices=None):
        """Add a traced path with optional segment coloring.

        Args:
            path: (N,2) array of (row, col) image coords
            shades: list of hex color strings to alternate between segments
            mark_indices: sorted list of path indices where marks divide segments
        """
        if shades is None:
            shades = ["#00ff88"]
        if mark_indices is None:
            mark_indices = []
        self._traces.append((path, shades, mark_indices))
        self._trace_original_colors.append(shades)

    def get_selected_for_retry(self):
        return sorted(self._selected_for_retry)

    def clear_review(self):
        self._selected_for_retry.clear()

    def clear_reclick(self):
        for rid in self._reclick_marker_ids:
            self.canvas.delete(rid)
        self._reclick_points.clear()
        self._reclick_marker_ids.clear()
        self._reclick_expected = 0

    def get_reclick_pairs(self):
        """Return list of (top, bottom) pairs from reclick points."""
        pairs = []
        for i in range(0, len(self._reclick_points) - 1, 2):
            pairs.append((self._reclick_points[i], self._reclick_points[i + 1]))
        return pairs

    # --- Image ---

    def set_image(self, img_np):
        """Set image from numpy array (grayscale or RGB uint8)."""
        self._image_np = img_np
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self.clear_plates()
        self.clear_roots()
        self.clear_traces()
        self._fit_image()
        self._redraw()

    def _fit_image(self):
        """Scale image to fit canvas."""
        if self._image_np is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        ih, iw = self._image_np.shape[:2]
        self._scale = min(cw / iw, ch / ih)
        self._offset_x = (cw - iw * self._scale) / 2
        self._offset_y = (ch - ih * self._scale) / 2

    def zoom_to_region(self, r1, r2, c1, c2, pad_frac=0.05):
        """Zoom canvas to show a specific image region with padding."""
        if self._image_np is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        rh = r2 - r1
        rw = c2 - c1
        pad_r = rh * pad_frac
        pad_c = rw * pad_frac
        r1 -= pad_r
        r2 += pad_r
        c1 -= pad_c
        c2 += pad_c
        rh = r2 - r1
        rw = c2 - c1
        self._scale = min(cw / rw, ch / rh)
        self._offset_x = (cw - rw * self._scale) / 2 - c1 * self._scale
        self._offset_y = (ch - rh * self._scale) / 2 - r1 * self._scale
        self._redraw()

    def _redraw(self):
        """Redraw image and all overlays."""
        self.canvas.delete("all")
        if self._image_np is None:
            return

        ih, iw = self._image_np.shape[:2]
        new_w = max(1, int(iw * self._scale))
        new_h = max(1, int(ih * self._scale))

        if len(self._image_np.shape) == 2:
            pil_img = Image.fromarray(self._image_np, mode='L')
        else:
            pil_img = Image.fromarray(self._image_np)

        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(
            int(self._offset_x), int(self._offset_y),
            image=self._photo, anchor="nw"
        )

        # redraw plate rectangles
        self._plate_rect_ids.clear()
        for i, (r1, r2, c1, c2) in enumerate(self._plates):
            cx1, cy1 = self.image_to_canvas(c1, r1)
            cx2, cy2 = self.image_to_canvas(c2, r2)
            rid = self.canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline="#4a9eff", width=2, dash=(6, 3))
            self._plate_rect_ids.append(rid)
            self.canvas.create_text(
                cx1 + 5, cy1 + 5, text=f"Plate {i + 1}",
                fill="#4a9eff", anchor="nw",
                font=("Helvetica", 12, "bold"))

        # redraw root markers (hide in review mode — only show traces)
        # group colors match CLI: red for group 0 (or non-split), blue for group 1
        _GROUP_MARKER_COLORS = ["#e63333", "#3333e6"]
        self._root_marker_ids.clear()
        if self._mode not in (self.MODE_REVIEW,):
            for i, ((row, col), flag) in enumerate(
                    zip(self._root_points, self._root_flags)):
                cx, cy = self.image_to_canvas(col, row)
                group = self._root_groups[i] if i < len(self._root_groups) else 0
                marker_color = _GROUP_MARKER_COLORS[group % len(_GROUP_MARKER_COLORS)]
                if flag is not None:
                    label = "DEAD" if flag == 'dead' else "TOUCH"
                    s = 6
                    id1 = self.canvas.create_line(
                        cx - s, cy - s, cx + s, cy + s,
                        fill=marker_color, width=2)
                    id2 = self.canvas.create_line(
                        cx - s, cy + s, cx + s, cy - s,
                        fill=marker_color, width=2)
                    id3 = self.canvas.create_text(
                        cx + 10, cy - 10, text=f"{i + 1} {label}",
                        fill=marker_color, anchor="w",
                        font=("Helvetica", 9, "bold"))
                    self._root_marker_ids.extend([id1, id2, id3])
                else:
                    r = 5
                    rid = self.canvas.create_oval(
                        cx - r, cy - r, cx + r, cy + r,
                        outline="white", fill=marker_color, width=1)
                    tid = self.canvas.create_text(
                        cx + 10, cy - 10, text=str(i + 1),
                        fill=marker_color, anchor="w",
                        font=("Helvetica", 9, "bold"))
                    self._root_marker_ids.extend([rid, tid])

        # redraw mark circles (hide in review mode)
        # marks use the light genotype shade to distinguish from root dots
        _GROUP_MARK_COLOR = ["#ff8080", "#8080ff"]  # light red, light blue
        self._mark_marker_ids.clear()
        if self._mode not in (self.MODE_REVIEW,):
            # draw saved marks from _all_marks (persisted across groups)
            for ri, marks in self._all_marks.items():
                group = self._root_groups[ri] if ri < len(self._root_groups) else 0
                color = _GROUP_MARK_COLOR[group % len(_GROUP_MARK_COLOR)]
                for mi, (row, col) in enumerate(marks):
                    cx, cy = self.image_to_canvas(col, row)
                    r = 4
                    self.canvas.create_oval(
                        cx - r, cy - r, cx + r, cy + r,
                        outline="white", fill=color, width=1)
            # draw current batch marks (not yet saved to _all_marks)
            if self._mode == self.MODE_CLICK_MARKS:
                current_group = getattr(self, '_current_root_group', 0)
                color = _GROUP_MARK_COLOR[current_group % len(_GROUP_MARK_COLOR)]
                for i, (row, col) in enumerate(self._mark_points):
                    cx, cy = self.image_to_canvas(col, row)
                    r = 5
                    rid = self.canvas.create_oval(
                        cx - r, cy - r, cx + r, cy + r,
                        outline="white", fill=color, width=1)
                    tid = self.canvas.create_text(
                        cx + 10, cy - 8, text=f"M{i + 1}",
                        fill=color, anchor="w",
                        font=("Helvetica", 8, "bold"))
                    self._mark_marker_ids.extend([rid, tid])

        # redraw traced paths with segment coloring
        for ti, (path, shades, mark_indices) in enumerate(self._traces):
            if len(path) < 2:
                continue
            is_selected = ti in self._selected_for_retry
            w = 4 if is_selected else 2

            if is_selected:
                # selected for retry: bright in review, dim in reclick
                sel_color = "#66580a" if self._mode == self.MODE_RECLICK else "#ffdd00"
                self._draw_path_segment(path, sel_color, w)
            elif mark_indices:
                # draw each segment in alternating shades
                boundaries = [0] + list(mark_indices) + [len(path) - 1]
                for j in range(len(boundaries) - 1):
                    start = boundaries[j]
                    end = boundaries[j + 1] + 1
                    color = shades[j % len(shades)]
                    self._draw_path_segment(path[start:end], color, w)
            else:
                # single color
                self._draw_path_segment(path, shades[0], w)

            # draw root number label at top of trace
            if self._mode == self.MODE_REVIEW and len(path) > 0:
                top_row, top_col = path[0]
                lx, ly = self.image_to_canvas(top_col, top_row)
                lbl_color = "#ffdd00" if is_selected else shades[0]
                self.canvas.create_text(
                    lx, ly - 10, text=str(ti + 1),
                    fill=lbl_color, anchor="s",
                    font=("Helvetica", 10, "bold"))

    def _draw_path_segment(self, path, color, width):
        """Draw a subsection of a path on the canvas."""
        if len(path) < 2:
            return
        step = max(1, len(path) // 500)
        coords = []
        for row, col in path[::step]:
            cx, cy = self.image_to_canvas(col, row)
            coords.extend([cx, cy])
        if len(coords) >= 4:
            self.canvas.create_line(
                *coords, fill=color, width=width, smooth=True)

    # --- Coordinate conversion ---

    def canvas_to_image(self, cx, cy):
        """Convert canvas coords to image pixel coords (col, row)."""
        ix = (cx - self._offset_x) / self._scale
        iy = (cy - self._offset_y) / self._scale
        return int(ix), int(iy)

    def image_to_canvas(self, ix, iy):
        """Convert image pixel coords (col, row) to canvas coords."""
        cx = ix * self._scale + self._offset_x
        cy = iy * self._scale + self._offset_y
        return cx, cy

    # --- Event handlers ---

    def _on_resize(self, event):
        if self._image_np is not None:
            self._fit_image()
            self._redraw()

    def _on_scroll(self, event):
        if self._image_np is None:
            return
        factor = 1.1 if event.delta > 0 else 0.9
        mx, my = event.x, event.y
        self._offset_x = mx - factor * (mx - self._offset_x)
        self._offset_y = my - factor * (my - self._offset_y)
        self._scale *= factor
        self._redraw()

    def _on_pan_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_pan_drag(self, event):
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._offset_x += dx
        self._offset_y += dy
        self._drag_start = (event.x, event.y)
        self._redraw()

    def _find_nearest_trace(self, img_col, img_row, threshold=30):
        """Find the trace index nearest to an image coordinate."""
        best_dist = threshold
        best_idx = None
        pt = np.array([img_row, img_col], dtype=float)
        for i, (path, _shades, _marks) in enumerate(self._traces):
            if len(path) < 2:
                continue
            step = max(1, len(path) // 200)
            dists = np.sqrt(((path[::step] - pt) ** 2).sum(axis=1))
            d = dists.min()
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _on_left_press(self, event):
        self.canvas.focus_set()
        if self._mode == self.MODE_SELECT_PLATES:
            col, row = self.canvas_to_image(event.x, event.y)
            self._rect_start = (row, col)
        elif self._mode == self.MODE_CLICK_ROOTS:
            col, row = self.canvas_to_image(event.x, event.y)
            flag = self._pending_flag
            self._pending_flag = None
            self._root_points.append((row, col))
            self._root_flags.append(flag)
            self._root_groups.append(getattr(self, '_current_root_group', 0))
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()
        elif self._mode == self.MODE_CLICK_MARKS:
            # stop accepting clicks once expected count reached
            if self._marks_expected > 0 and \
               len(self._mark_points) >= self._marks_expected:
                return
            col, row = self.canvas_to_image(event.x, event.y)
            self._mark_points.append((row, col))
            # draw mark as genotype-colored circle (light shade)
            cx, cy = event.x, event.y
            _GROUP_MARK_COLOR = ["#ff8080", "#8080ff"]
            current_group = getattr(self, '_current_root_group', 0)
            color = _GROUP_MARK_COLOR[current_group % len(_GROUP_MARK_COLOR)]
            n = len(self._mark_points)
            r = 5
            rid = self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline="white", fill=color, width=1)
            tid = self.canvas.create_text(
                cx + 10, cy - 8, text=f"M{n}",
                fill=color, anchor="w",
                font=("Helvetica", 8, "bold"))
            self._mark_marker_ids.extend([rid, tid])
            if self._on_click_callback:
                self._on_click_callback()
        elif self._mode == self.MODE_REVIEW:
            col, row = self.canvas_to_image(event.x, event.y)
            threshold = 30 / self._scale if self._scale > 0 else 30
            idx = self._find_nearest_trace(col, row, threshold=max(20, threshold))
            if idx is not None:
                if idx in self._selected_for_retry:
                    self._selected_for_retry.discard(idx)
                else:
                    self._selected_for_retry.add(idx)
                self._redraw()
                if self._on_click_callback:
                    self._on_click_callback()
        elif self._mode == self.MODE_RECLICK:
            col, row = self.canvas_to_image(event.x, event.y)
            self._reclick_points.append((row, col))
            self._redraw()
            # draw marker
            cx, cy = event.x, event.y
            r = 5
            rid = self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline="white", fill="#1a7a1a", width=1)
            self._reclick_marker_ids.append(rid)
            if self._on_click_callback:
                self._on_click_callback()

    def _on_left_drag(self, event):
        if self._mode == self.MODE_SELECT_PLATES and self._rect_start is not None:
            # draw live rubber-band rectangle
            if self._rect_drag_id is not None:
                self.canvas.delete(self._rect_drag_id)
            r1, c1 = self._rect_start
            cx1, cy1 = self.image_to_canvas(c1, r1)
            self._rect_drag_id = self.canvas.create_rectangle(
                cx1, cy1, event.x, event.y,
                outline="#4a9eff", width=2, dash=(4, 2))

    def _on_left_release(self, event):
        if self._mode == self.MODE_SELECT_PLATES and self._rect_start is not None:
            if self._rect_drag_id is not None:
                self.canvas.delete(self._rect_drag_id)
                self._rect_drag_id = None
            r1, c1 = self._rect_start
            c2, r2 = self.canvas_to_image(event.x, event.y)
            self._rect_start = None

            # normalize
            rmin, rmax = min(r1, r2), max(r1, r2)
            cmin, cmax = min(c1, c2), max(c1, c2)

            # ignore tiny rectangles (accidental clicks)
            if (rmax - rmin) < 20 or (cmax - cmin) < 20:
                return

            self._plates.append((rmin, rmax, cmin, cmax))
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()

    def _on_right_click(self, event):
        """Right-click / two-finger click: undo last action."""
        self._undo()

    def _undo(self):
        """Undo last action in current mode."""
        if self._mode == self.MODE_SELECT_PLATES and self._plates:
            self._plates.pop()
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()
            return True
        elif self._mode == self.MODE_CLICK_ROOTS and self._root_points:
            # only undo clicks from current group (matches CLI behavior)
            current_group = getattr(self, '_current_root_group', 0)
            if self._root_groups and self._root_groups[-1] != current_group:
                return False
            self._root_points.pop()
            self._root_flags.pop()
            self._root_groups.pop()
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()
            return True
        elif self._mode == self.MODE_CLICK_MARKS and self._mark_points:
            self._mark_points.pop()
            for _ in range(2):
                if self._mark_marker_ids:
                    self.canvas.delete(self._mark_marker_ids.pop())
            if self._on_click_callback:
                self._on_click_callback()
            return True
        elif self._mode == self.MODE_RECLICK and self._reclick_points:
            self._reclick_points.pop()
            if self._reclick_marker_ids:
                self.canvas.delete(self._reclick_marker_ids.pop())
            if self._on_click_callback:
                self._on_click_callback()
            return True
        return False

    def handle_key(self, event):
        """Handle keyboard events (called from app-level binding)."""
        # Cmd+Z / Ctrl+Z = undo
        if event.keysym.lower() == 'z' and (event.state & 0x8 or event.state & 0x4):
            return self._undo()
        if self._mode == self.MODE_CLICK_ROOTS:
            if event.keysym.lower() == 'd':
                self._pending_flag = 'dead'
                return True
            elif event.keysym.lower() == 't':
                self._pending_flag = 'touching'
                return True
        if event.keysym == 'Return':
            if self._mode == self.MODE_SELECT_PLATES:
                # Enter confirms drawn plate; second Enter with no new plate finishes
                if len(self._plates) > self._plates_count_at_enter:
                    self._plates_count_at_enter = len(self._plates)
                    return True
                # no new plate — finish selection
            if self._on_done_callback:
                self._on_done_callback()
                return True
        return False
