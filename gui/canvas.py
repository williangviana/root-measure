"""ImageCanvas — zoomable, pannable image canvas with overlay drawing."""

import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

# 30-color palettes (module-level so they're shared everywhere)
# Bright, saturated colors for root markers/traces
GROUP_MARKER_COLORS = [
    "#FF2020", "#2060FF", "#00CC44", "#FF9900", "#CC33CC",
    "#00CCCC", "#FF5500", "#8833FF", "#CCAA00", "#00DD77",
    "#FF2080", "#3399FF", "#88DD00", "#FFBB00", "#6622FF",
    "#00DDAA", "#FF3366", "#3366FF", "#66DD00", "#FF7700",
    "#BB22FF", "#00DD55", "#FF2299", "#2299FF", "#AADD00",
    "#FF8800", "#9922FF", "#00DD88", "#DD2222", "#2222DD",
]
# Pastel versions of the same hues for segment marks
GROUP_MARK_COLORS = [
    "#FFB0B0", "#B0C8FF", "#B0F0C4", "#FFD9A0", "#E8B0E8",
    "#A0E8E8", "#FFBFA0", "#C8B0FF", "#E8DCA0", "#A0F0C8",
    "#FFB0CC", "#A0C8FF", "#CCF0A0", "#FFE0A0", "#BAA0FF",
    "#A0F0DA", "#FFB0BB", "#A0BBFF", "#BBF0A0", "#FFC8A0",
    "#DCA0FF", "#A0F0B0", "#FFB0DD", "#A0DDFF", "#DAF0A0",
    "#FFD0A0", "#CCA0FF", "#A0F0CC", "#F0B0B0", "#B0B0F0",
]


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
        self._zoom_settle_id = None  # after() id for deferred hi-res redraw
        self._user_zoomed = False    # True after user scrolls/zooms/pans
        self._space_held = False     # True while spacebar held (pan mode)
        self._pil_base = None        # cached PIL Image from numpy
        self._img_id = None          # canvas id of the main image item
        self._rendered_scale = None  # scale at which _photo was last rendered

        # interaction mode
        self._mode = self.MODE_VIEW
        self._on_click_callback = None
        self._on_done_callback = None

        # plate selection state
        self._rect_start = None  # image coords of rect drag start
        self._rect_drag_id = None  # canvas id of live drag rect
        self._plates = []        # list of (r1, r2, c1, c2) in image coords
        self._pending_plate = None  # drawn but not yet confirmed
        self._plate_rect_ids = []  # canvas ids of confirmed rects

        # root clicking state
        self._root_points = []     # list of (row, col) in image coords
        self._root_flags = []      # None, 'dead', 'touching'
        self._root_groups = []     # group index per root (for split plate)
        self._root_plates = []     # plate index per root
        self._root_marker_ids = [] # canvas ids of markers
        self._pending_flag = None  # 'dead' or 'touching'

        # mark clicking state (multi-measurement)
        self._mark_points = []     # list of (row, col) in image coords
        self._mark_marker_ids = [] # canvas ids of mark markers
        self._marks_expected = 0   # how many marks expected (stops clicks beyond this)
        self._all_marks = {}       # {root_index: [(row,col), ...]} all collected marks

        # trace overlay state
        self._traces = []          # list of (path_array, color_str)
        self._trace_to_result = []  # maps trace index → root index

        # review state (click to toggle retry selection)
        self._selected_for_retry = set()  # indices into _traces
        self._trace_original_colors = []  # original colors before selection

        # reclick state (top+bottom for retry roots)
        self._reclick_points = []  # list of (row, col) pairs
        self._reclick_expected = 0  # how many pairs expected
        self._reclick_marker_ids = []

        # help overlay
        self._help_visible = False
        # set True after measurement is complete to hide dot markers
        self._measurement_done = False

        # bindings
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        # macOS trackpad: two-finger click = Button-2; physical right = Button-3
        self.canvas.bind("<ButtonPress-2>", self._on_right_click)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        self.canvas.bind("<Shift-MouseWheel>", self._on_scroll_h)
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
        self._pending_plate = None
        self._plate_rect_ids.clear()

    def clear_roots(self):
        for rid in self._root_marker_ids:
            self.canvas.delete(rid)
        self._root_points.clear()
        self._root_flags.clear()
        self._root_groups.clear()
        self._root_plates.clear()
        self._root_marker_ids.clear()
        self._pending_flag = None

    def clear_marks(self):
        for rid in self._mark_marker_ids:
            self.canvas.delete(rid)
        self._mark_points.clear()
        self._mark_marker_ids.clear()
        self._all_marks.clear()

    def get_mark_points(self):
        return list(self._mark_points)

    def set_plates(self, plates):
        """Restore plates from saved state."""
        self._plates = [tuple(p) for p in plates]

    def set_roots(self, points, flags, groups, plates):
        """Restore root points from saved state."""
        self._root_points = [tuple(p) for p in points]
        self._root_flags = list(flags)
        self._root_groups = list(groups)
        self._root_plates = list(plates)

    def set_marks(self, all_marks):
        """Restore marks from saved state."""
        self._all_marks = {int(k): [tuple(m) for m in v]
                           for k, v in all_marks.items()}

    def clear_traces(self):
        self._traces.clear()
        self._trace_to_result.clear()
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

    def set_traces(self, traces_data, trace_to_result=None):
        """Restore traces from saved session data."""
        self._traces.clear()
        self._trace_to_result.clear()
        self._trace_original_colors.clear()
        for t in traces_data:
            path = t['path']
            shades = t.get('shades', ['#00ff88'])
            mark_indices = t.get('mark_indices', [])
            self._traces.append((path, shades, mark_indices))
            self._trace_original_colors.append(shades)
        if trace_to_result:
            self._trace_to_result = list(trace_to_result)

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

    def get_reclick_groups(self, clicks_per_root):
        """Return list of click groups from reclick points.

        Each group has clicks_per_root points: [top, mark1, ..., markN, bottom].
        """
        groups = []
        for i in range(0, len(self._reclick_points), clicks_per_root):
            chunk = self._reclick_points[i:i + clicks_per_root]
            if len(chunk) == clicks_per_root:
                groups.append(chunk)
        return groups

    # --- Image ---

    def set_image(self, img_np):
        """Set image from numpy array (grayscale or RGB uint8)."""
        self._image_np = img_np
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._user_zoomed = False
        # cache PIL base image (avoids repeated fromarray)
        if img_np is not None:
            if len(img_np.shape) == 2:
                self._pil_base = Image.fromarray(img_np, mode='L')
            else:
                self._pil_base = Image.fromarray(img_np)
        else:
            self._pil_base = None
        self.clear_plates()
        self.clear_roots()
        self.clear_marks()
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

    def zoom_to_region(self, r1, r2, c1, c2, pad_frac=0.0):
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
        """Redraw image and all overlays at full LANCZOS quality."""
        self.canvas.delete("all")
        self._img_id = None
        if self._pil_base is None:
            return

        iw, ih = self._pil_base.size
        new_w = max(1, int(iw * self._scale))
        new_h = max(1, int(ih * self._scale))

        pil_img = self._pil_base.resize((new_w, new_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil_img)
        self._img_id = self.canvas.create_image(
            int(self._offset_x), int(self._offset_y),
            image=self._photo, anchor="nw"
        )
        self._rendered_scale = self._scale

        self._draw_overlays()

    def _draw_overlays(self):
        """Draw all overlays (plates, markers, traces, reclick)."""
        is_view = self._mode == self.MODE_VIEW

        # plate rectangles (confirmed)
        self._plate_rect_ids.clear()
        show_plate_rect = self._mode in (self.MODE_SELECT_PLATES, self.MODE_VIEW,
                                          self.MODE_REVIEW)
        for i, (r1, r2, c1, c2) in enumerate(self._plates):
            cx1, cy1 = self.image_to_canvas(c1, r1)
            cx2, cy2 = self.image_to_canvas(c2, r2)
            if show_plate_rect:
                rid = self.canvas.create_rectangle(
                    cx1, cy1, cx2, cy2,
                    outline="#9b59b6", width=2, dash=(6, 3))
                self._plate_rect_ids.append(rid)
                self.canvas.create_text(
                    cx1 + 5, cy2 + 5, text=f"Plate {i + 1}",
                    fill="#9b59b6", anchor="nw",
                    font=("Helvetica", 18, "bold"))
        # pending plate (not yet confirmed)
        if self._pending_plate is not None:
            r1, r2, c1, c2 = self._pending_plate
            cx1, cy1 = self.image_to_canvas(c1, r1)
            cx2, cy2 = self.image_to_canvas(c2, r2)
            n = len(self._plates) + 1
            self.canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline="#c39bd3", width=2)
            self.canvas.create_text(
                cx1 + 5, cy2 + 5, text=f"Plate {n} (press Enter)",
                fill="#c39bd3", anchor="nw",
                font=("Helvetica", 16, "bold"))

        # root markers (hide when measurement done or in review mode)
        self._root_marker_ids.clear()
        hide_markers = (self._mode == self.MODE_REVIEW
                        or self._measurement_done
                        or len(self._traces) > 0)
        if not hide_markers:
            group_counters = {}
            clicking_roots = self._mode in (self.MODE_CLICK_ROOTS,
                                             self.MODE_CLICK_MARKS)
            active_plate = getattr(self, '_current_plate_idx', None)
            dot_r = 3 if is_view else 5
            cross_s = 4 if is_view else 6
            for i, ((row, col), flag) in enumerate(
                    zip(self._root_points, self._root_flags)):
                group = self._root_groups[i] if i < len(self._root_groups) else 0
                plate = self._root_plates[i] if i < len(self._root_plates) else 0
                key = (plate, group)
                group_counters[key] = group_counters.get(key, 0) + 1
                if clicking_roots and active_plate is not None \
                        and plate != active_plate:
                    continue
                cx, cy = self.image_to_canvas(col, row)
                display_num = group_counters[key]
                marker_color = GROUP_MARKER_COLORS[group % len(GROUP_MARKER_COLORS)]
                if flag is not None:
                    label = "DEAD" if flag == 'dead' else "TOUCH"
                    id1 = self.canvas.create_line(
                        cx - cross_s, cy - cross_s, cx + cross_s, cy + cross_s,
                        fill=marker_color, width=2)
                    id2 = self.canvas.create_line(
                        cx - cross_s, cy + cross_s, cx + cross_s, cy - cross_s,
                        fill=marker_color, width=2)
                    id3 = self.canvas.create_text(
                        cx + 10, cy - 10, text=f"{display_num} {label}",
                        fill=marker_color, anchor="w",
                        font=("Helvetica", 9, "bold"))
                    self._root_marker_ids.extend([id1, id2, id3])
                else:
                    rid = self.canvas.create_oval(
                        cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r,
                        outline="white", fill=marker_color, width=1)
                    tid = self.canvas.create_text(
                        cx + 10, cy - 10, text=str(display_num),
                        fill=marker_color, anchor="w",
                        font=("Helvetica", 9, "bold"))
                    self._root_marker_ids.extend([rid, tid])

        # mark circles (hide in review mode or when measurement done)
        self._mark_marker_ids.clear()
        if self._mode not in (self.MODE_REVIEW,) and not self._measurement_done:
            mark_r = 3 if is_view else 4
            for ri, marks in self._all_marks.items():
                group = self._root_groups[ri] if ri < len(self._root_groups) else 0
                color = GROUP_MARK_COLORS[group % len(GROUP_MARK_COLORS)]
                for mi, (row, col) in enumerate(marks):
                    cx, cy = self.image_to_canvas(col, row)
                    self.canvas.create_oval(
                        cx - mark_r, cy - mark_r, cx + mark_r, cy + mark_r,
                        outline="white", fill=color, width=1)
            if self._mode == self.MODE_CLICK_MARKS:
                current_group = getattr(self, '_current_root_group', 0)
                color = GROUP_MARK_COLORS[current_group % len(GROUP_MARK_COLORS)]
                for i, (row, col) in enumerate(self._mark_points):
                    cx, cy = self.image_to_canvas(col, row)
                    r = 5
                    rid = self.canvas.create_oval(
                        cx - r, cy - r, cx + r, cy + r,
                        outline="white", fill=color, width=1)
                    tid = self.canvas.create_text(
                        cx + 10, cy - 8, text=str(i + 1),
                        fill=color, anchor="w",
                        font=("Helvetica", 8, "bold"))
                    self._mark_marker_ids.extend([rid, tid])

        # traced paths with segment coloring
        trace_plate_counters = {}
        for ti, (path, shades, mark_indices) in enumerate(self._traces):
            if len(path) < 2:
                continue
            is_selected = ti in self._selected_for_retry
            w = 4 if is_selected else 2

            if is_selected:
                if self._mode == self.MODE_RECLICK:
                    self._draw_path_segment(path, "#ff8c00", w, dash=(1, 2))
                else:
                    self._draw_path_segment(path, "#ff8c00", w)
            elif mark_indices:
                boundaries = [0] + list(mark_indices) + [len(path) - 1]
                for j in range(len(boundaries) - 1):
                    start = boundaries[j]
                    end = boundaries[j + 1] + 1
                    color = shades[j % len(shades)]
                    self._draw_path_segment(path[start:end], color, w)
            else:
                self._draw_path_segment(path, shades[0], w)

            if self._mode == self.MODE_REVIEW and len(path) > 0:
                ri = self._trace_to_result[ti] if ti < len(self._trace_to_result) else ti
                plate = (self._root_plates[ri]
                         if ri < len(self._root_plates) else 0)
                group = (self._root_groups[ri]
                         if ri < len(self._root_groups) else 0)
                key = (plate, group)
                trace_plate_counters[key] = \
                    trace_plate_counters.get(key, 0) + 1
                top_row, top_col = path[0]
                lx, ly = self.image_to_canvas(top_col, top_row)
                lbl_color = "#ff8c00" if is_selected else shades[0]
                self.canvas.create_text(
                    lx, ly - 10,
                    text=str(trace_plate_counters[key]),
                    fill=lbl_color, anchor="s",
                    font=("Helvetica", 10, "bold"))

        # reclick markers
        if self._mode == self.MODE_RECLICK and self._reclick_points:
            cpr = getattr(self, '_reclick_clicks_per_root', 2)
            self._reclick_marker_ids.clear()
            for i, (row, col) in enumerate(self._reclick_points):
                cx, cy = self.image_to_canvas(col, row)
                r = 5
                label = str((i % cpr) + 1)
                rid = self.canvas.create_oval(
                    cx - r, cy - r, cx + r, cy + r,
                    outline="white", fill="#1a7a1a", width=1)
                tid = self.canvas.create_text(
                    cx + 10, cy - 8, text=label,
                    fill="#1a7a1a", anchor="w",
                    font=("Helvetica", 8, "bold"))
                self._reclick_marker_ids.extend([rid, tid])

        # help overlay
        if self._help_visible:
            self._draw_help_overlay()

    def _draw_help_overlay(self):
        """Draw a keyboard shortcut reference card on the canvas."""
        _GLOBAL = [
            ("?", "Toggle this help"),
            ("Scroll", "Zoom in/out"),
            ("Space+Drag", "Pan image"),
            ("\u2318Z", "Undo last action"),
        ]
        _MODE_SHORTCUTS = {
            self.MODE_VIEW: [],
            self.MODE_SELECT_PLATES: [
                ("Drag", "Draw plate rectangle"),
                ("Enter", "Confirm plate / finish"),
                ("Right-click", "Undo last plate"),
            ],
            self.MODE_CLICK_ROOTS: [
                ("Click", "Place root top"),
                ("D + Click", "Mark as dead"),
                ("T + Click", "Mark as touching"),
                ("Enter", "Next genotype / plate"),
                ("Right-click", "Undo last click"),
            ],
            self.MODE_CLICK_MARKS: [
                ("Click", "Place mark point"),
                ("Enter", "Confirm marks"),
                ("Right-click", "Undo last mark"),
            ],
            self.MODE_REVIEW: [
                ("Click trace", "Select / deselect for retry"),
                ("Enter", "Accept all / retry selected"),
            ],
            self.MODE_RECLICK: [
                ("Click", "Place top / mark / bottom"),
                ("Enter", "Confirm re-click"),
                ("Right-click", "Undo last click"),
            ],
        }
        _MODE_NAMES = {
            self.MODE_VIEW: "View",
            self.MODE_SELECT_PLATES: "Select Plates",
            self.MODE_CLICK_ROOTS: "Click Roots",
            self.MODE_CLICK_MARKS: "Click Marks",
            self.MODE_REVIEW: "Review",
            self.MODE_RECLICK: "Re-click",
        }

        mode_shortcuts = _MODE_SHORTCUTS.get(self._mode, [])
        mode_name = _MODE_NAMES.get(self._mode, "")
        lines = []
        if mode_shortcuts:
            lines.append(("", f"--- {mode_name} ---"))
            lines.extend(mode_shortcuts)
            lines.append(("", ""))
        lines.append(("", "--- General ---"))
        lines.extend(_GLOBAL)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        font_key = ("Menlo", 11, "bold")
        font_desc = ("Helvetica", 11)
        line_h = 20
        pad = 16
        key_col_w = 120
        desc_col_w = 200
        card_w = key_col_w + desc_col_w + pad * 2
        card_h = len(lines) * line_h + pad * 2 + 24  # +24 for title

        x0 = (cw - card_w) // 2
        y0 = (ch - card_h) // 2

        # background
        self.canvas.create_rectangle(
            x0, y0, x0 + card_w, y0 + card_h,
            fill="#1a1a1a", outline="#4a9eff", width=2,
            stipple="gray75")
        # title
        self.canvas.create_text(
            x0 + card_w // 2, y0 + pad,
            text="Keyboard Shortcuts",
            fill="#4a9eff", anchor="n",
            font=("Helvetica", 13, "bold"))

        y = y0 + pad + 28
        for key, desc in lines:
            if key == "" and desc.startswith("---"):
                # section header
                self.canvas.create_text(
                    x0 + card_w // 2, y + line_h // 2,
                    text=desc.strip("- "),
                    fill="#4a9eff", anchor="center",
                    font=("Helvetica", 10, "bold"))
            elif key == "" and desc == "":
                pass  # spacer
            else:
                self.canvas.create_text(
                    x0 + pad + key_col_w, y + line_h // 2,
                    text=key, fill="white", anchor="e",
                    font=font_key)
                self.canvas.create_text(
                    x0 + pad + key_col_w + 12, y + line_h // 2,
                    text=desc, fill="#cccccc", anchor="w",
                    font=font_desc)
            y += line_h

    def _draw_path_segment(self, path, color, width, dash=None):
        """Draw a subsection of a path on the canvas."""
        if len(path) < 2:
            return
        step = max(1, len(path) // 500)
        coords = []
        for row, col in path[::step]:
            cx, cy = self.image_to_canvas(col, row)
            coords.extend([cx, cy])
        if len(coords) >= 4:
            kwargs = dict(fill=color, width=width, smooth=True)
            if dash:
                kwargs['dash'] = dash
            self.canvas.create_line(*coords, **kwargs)

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
            if not self._user_zoomed:
                self._fit_image()
            self._redraw()

    def _on_scroll(self, event):
        """Scroll/pinch = zoom centered on cursor."""
        if self._image_np is None:
            return
        self._do_zoom(event)

    def _on_scroll_h(self, event):
        """Horizontal scroll (Shift+MouseWheel on macOS trackpad)."""
        if self._image_np is None:
            return
        self._do_zoom(event)

    def _do_zoom(self, event):
        """Zoom centered on mouse position."""
        factor = 1.1 if event.delta > 0 else 0.9
        mx, my = event.x, event.y
        self._offset_x = mx - factor * (mx - self._offset_x)
        self._offset_y = my - factor * (my - self._offset_y)
        self._scale *= factor
        self._user_zoomed = True
        self._fast_redraw()

    def _fast_redraw(self):
        """Move existing image on canvas (instant); full re-render on settle."""
        if self._zoom_settle_id is not None:
            self.after_cancel(self._zoom_settle_id)
        if self._img_id is not None and self._rendered_scale == self._scale:
            # Pure pan — just move the image, no resize needed
            self.canvas.coords(self._img_id,
                               int(self._offset_x), int(self._offset_y))
        elif self._pil_base is not None:
            # Zoom changed — quick NEAREST resize
            self.canvas.delete("all")
            iw, ih = self._pil_base.size
            new_w = max(1, int(iw * self._scale))
            new_h = max(1, int(ih * self._scale))
            pil_img = self._pil_base.resize((new_w, new_h), Image.NEAREST)
            self._photo = ImageTk.PhotoImage(pil_img)
            self._img_id = self.canvas.create_image(
                int(self._offset_x), int(self._offset_y),
                image=self._photo, anchor="nw")
            self._rendered_scale = self._scale
        self._zoom_settle_id = self.after(150, self._settle_zoom)

    def _settle_zoom(self):
        """Redraw at full quality after gesture ends."""
        self._zoom_settle_id = None
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
        self._fast_redraw()

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
        if self._space_held:
            self._drag_start = (event.x, event.y)
            return
        if self._mode == self.MODE_SELECT_PLATES:
            # clear previous pending plate when starting a new drag
            if self._pending_plate is not None:
                self._pending_plate = None
                self._redraw()
            col, row = self.canvas_to_image(event.x, event.y)
            self._rect_start = (row, col)
        elif self._mode == self.MODE_CLICK_ROOTS:
            col, row = self.canvas_to_image(event.x, event.y)
            flag = self._pending_flag
            self._pending_flag = None
            self._root_points.append((row, col))
            self._root_flags.append(flag)
            self._root_groups.append(getattr(self, '_current_root_group', 0))
            self._root_plates.append(getattr(self, '_current_plate_idx', 0))
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
            current_group = getattr(self, '_current_root_group', 0)
            color = GROUP_MARK_COLORS[current_group % len(GROUP_MARK_COLORS)]
            n = len(self._mark_points)
            r = 5
            rid = self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline="white", fill=color, width=1)
            tid = self.canvas.create_text(
                cx + 10, cy - 8, text=str(n),
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
            if self._reclick_expected > 0 and \
               len(self._reclick_points) >= self._reclick_expected:
                return
            col, row = self.canvas_to_image(event.x, event.y)
            self._reclick_points.append((row, col))
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()

    def _on_left_drag(self, event):
        if self._space_held and self._drag_start is not None:
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            self._offset_x += dx
            self._offset_y += dy
            self._drag_start = (event.x, event.y)
            self._fast_redraw()
            return
        if self._mode == self.MODE_SELECT_PLATES and self._rect_start is not None:
            # draw live rubber-band rectangle
            if self._rect_drag_id is not None:
                self.canvas.delete(self._rect_drag_id)
            r1, c1 = self._rect_start
            cx1, cy1 = self.image_to_canvas(c1, r1)
            self._rect_drag_id = self.canvas.create_rectangle(
                cx1, cy1, event.x, event.y,
                outline="#9b59b6", width=2, dash=(4, 2))

    def _on_left_release(self, event):
        if self._space_held:
            self._drag_start = None
            return
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

            # store as pending — user can redraw before pressing Enter
            self._pending_plate = (rmin, rmax, cmin, cmax)
            self._redraw()

    def _on_right_click(self, event):
        """Right-click / two-finger click: undo last action."""
        self._undo()

    def _undo(self):
        """Undo last action in current mode."""
        if self._mode == self.MODE_SELECT_PLATES:
            if self._pending_plate is not None:
                self._pending_plate = None
                self._redraw()
                return True
            if self._plates:
                self._plates.pop()
                self._redraw()
                return True
            return False
        elif self._mode == self.MODE_CLICK_ROOTS and self._root_points:
            # only undo clicks from current group (matches CLI behavior)
            current_group = getattr(self, '_current_root_group', 0)
            if self._root_groups and self._root_groups[-1] != current_group:
                return False
            self._root_points.pop()
            self._root_flags.pop()
            self._root_groups.pop()
            self._root_plates.pop()
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
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()
            return True
        return False

    def handle_key(self, event):
        """Handle keyboard events (called from app-level binding)."""
        # ? key toggles help overlay
        if event.keysym == 'question' or \
                (event.keysym == 'slash' and event.state & 0x1):
            self._help_visible = not self._help_visible
            self._redraw()
            return True
        # dismiss help on any other key
        if self._help_visible:
            self._help_visible = False
            self._redraw()
            return True
        # spacebar hold for pan mode
        if event.keysym == 'space':
            self._space_held = True
            self.canvas.configure(cursor="fleur")
            return True
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
                if self._pending_plate is not None:
                    # confirm pending plate
                    self._plates.append(self._pending_plate)
                    self._pending_plate = None
                    self._redraw()
                    n = len(self._plates)
                    if getattr(self, '_app_status_callback', None):
                        self._app_status_callback(
                            f"{n} plate(s) confirmed.\n"
                            f"Draw another plate, or Enter to finish.")
                    return True
                # no pending plate — finish selection
            if self._on_done_callback:
                self._on_done_callback()
                return True
        return False

    def handle_key_release(self, event):
        """Handle key release events."""
        if event.keysym == 'space':
            self._space_held = False
            self._drag_start = None
            self.canvas.configure(cursor="")
