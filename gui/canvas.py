"""ImageCanvas — zoomable, pannable image canvas with overlay drawing."""

import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

# 30-color palettes (module-level so they're shared everywhere)
# First 8 are Okabe-Ito (colorblind-friendly), then extended hues
# Bright, saturated colors for root markers/traces
GROUP_MARKER_COLORS = [
    "#0072B2", "#E69F00", "#D55E00", "#56B4E9", "#009E73",  # OI 1-5
    "#F0E442", "#CC79A7", "#AA4400", "#882255", "#44AA99",  # OI 6-8 + ext
    "#AA4499", "#999933", "#6699CC", "#DD7788", "#117733",
    "#88CCEE", "#CC6677", "#DDCC77", "#332288", "#44BB99",
    "#EE8866", "#BBCC33", "#EEDD88", "#77AADD", "#EE6677",
    "#66CCEE", "#AA3377", "#BBBB33", "#4477AA", "#228833",
]
# Pastel versions of the same hues for segment marks
GROUP_MARK_COLORS = [
    "#A0D4F0", "#F5D89A", "#F0BFA0", "#B8DDF5", "#A0E8D0",  # OI 1-5
    "#F8F2B0", "#E8C4DA", "#E0C4A0", "#D4A0C0", "#B0DDD4",  # OI 6-8 + ext
    "#D8B0D8", "#D4D4A0", "#B8CCE0", "#F0C0CC", "#A0D4A0",
    "#C4E8F0", "#E0B8C4", "#F0E8B8", "#B0A8D8", "#B0DDD4",
    "#F5C8B0", "#D8E0A0", "#F5F0C0", "#C0D4F0", "#F0B8C4",
    "#B8E8F5", "#D8A0C4", "#D8D8A0", "#B0C4E0", "#A8D8A8",
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
    MODE_MANUAL_TRACE = "manual_trace"

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
        self._click_start = None     # deferred click position for click-or-pan
        self._is_panning = False     # True once drag exceeds pan threshold
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
        self._marks_display_numbers = []  # display number per mark-group (root position)

        # manual endpoints state (per-root top+bottom clicking)
        self._root_bottoms = {}    # {root_index: (row, col)} bottom click per root
        self._click_seq_pos = 0    # position in click sequence: 0=top, 1..N-2=marks, N-1=bottom
        self._clicks_per_root = 1  # 1=auto mode (top only), 2+=manual (top+marks+bottom)
        self._manual_endpoints = False  # True when manual endpoints mode is active

        # trace overlay state
        self._traces = []          # list of (path_array, color_str)
        self._trace_to_result = []  # maps trace index → root index

        # review state (click to toggle retry selection)
        self._selected_for_retry = set()  # indices into _traces
        self._trace_original_colors = []  # original colors before selection
        self._review_zoom_state = -1       # -1 = full view, 0..N-1 = zoomed plate index
        self._review_traces_visible = True # toggle trace overlay in review mode

        # reclick state (top+bottom for retry roots)
        self._reclick_points = []  # list of (row, col) pairs
        self._reclick_expected = 0  # how many pairs expected
        self._reclick_marker_ids = []

        # manual trace state
        self._manual_trace_points = []      # list of (row, col) — current root
        self._manual_trace_confirmed = []   # list of point-lists for completed roots
        self._manual_trace_marker_ids = []  # canvas oval ids
        self._manual_trace_line_ids = []    # canvas line ids
        self._manual_trace_submode = 'freehand'  # 'segmented' or 'freehand'
        self._manual_trace_strokes = []     # stroke start indices (freehand undo)
        self._manual_trace_drawing = False  # True during active freehand drag

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
        # macOS: pinch-to-zoom generates Control-MouseWheel
        self.canvas.bind("<Control-MouseWheel>", self._on_pinch_zoom)
        self.canvas.bind("<Option-MouseWheel>", self._on_pinch_zoom)
        # keyboard — bound at window level via bind_all in RootMeasureApp
        self.canvas.focus_set()

    # --- Mode management ---

    def set_mode(self, mode, on_done=None):
        """Set interaction mode."""
        # Reset transient state from previous mode
        self._manual_trace_drawing = False
        self._is_panning = False
        self._click_start = None
        self._drag_start = None
        self._mode = mode
        self._on_done_callback = on_done
        self.canvas.focus_set()

    def _trigger_done(self):
        """Trigger the current mode's done callback (for Done button)."""
        # Confirm pending plate — then let _on_plate_added decide next step
        if self._mode == self.MODE_SELECT_PLATES and self._pending_plate is not None:
            self._plates.append(self._pending_plate)
            self._pending_plate = None
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()
            return
        if self._on_done_callback:
            self._on_done_callback()

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
        self._root_bottoms.clear()
        self._click_seq_pos = 0

    def clear_marks(self):
        for rid in self._mark_marker_ids:
            self.canvas.delete(rid)
        self._mark_points.clear()
        self._mark_marker_ids.clear()
        self._all_marks.clear()
        self._marks_display_numbers.clear()

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

    def get_root_bottoms(self):
        """Return dict of root_index -> (row, col) bottom points."""
        return dict(self._root_bottoms)

    def set_root_bottoms(self, bottoms):
        """Restore root bottoms from saved state."""
        self._root_bottoms = {int(k): tuple(v) for k, v in bottoms.items()}

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
        for i, t in enumerate(traces_data):
            path = t['path']
            shades = t.get('shades', ['#00ff88'])
            mark_indices = t.get('mark_indices', [])
            self._traces.append((path, shades, mark_indices))
            self._trace_original_colors.append(shades)
        if trace_to_result:
            self._trace_to_result = list(trace_to_result)
        else:
            # Default: trace index maps to itself
            self._trace_to_result = list(range(len(traces_data)))

    def get_selected_for_retry(self):
        return sorted(self._selected_for_retry)

    def clear_review(self):
        self._selected_for_retry.clear()
        self._review_zoom_state = -1
        self._review_traces_visible = True

    def toggle_review_zoom(self):
        """Cycle zoom: full view → plate 0 → plate 1 → … → full view."""
        n = len(self._plates)
        if n == 0:
            return -1
        # cycle: -1 → 0 → 1 → … → n-1 → -1
        self._review_zoom_state += 1
        if self._review_zoom_state >= n:
            self._review_zoom_state = -1
        if self._review_zoom_state >= 0:
            self.zoom_to_region(*self._plates[self._review_zoom_state])
        else:
            self._fit_image()
            self._redraw()
        return self._review_zoom_state

    def toggle_review_traces(self):
        """Toggle trace overlay visibility in review mode."""
        self._review_traces_visible = not self._review_traces_visible
        self._redraw()
        return self._review_traces_visible

    def clear_reclick(self):
        for rid in self._reclick_marker_ids:
            self.canvas.delete(rid)
        self._reclick_points.clear()
        self._reclick_marker_ids.clear()
        self._reclick_expected = 0

    def confirm_manual_trace(self):
        """Save current manual trace points and start fresh for next root."""
        if self._manual_trace_points:
            self._manual_trace_confirmed.append(
                list(self._manual_trace_points))
        self._manual_trace_points.clear()
        self._manual_trace_marker_ids.clear()
        self._manual_trace_line_ids.clear()
        self._manual_trace_strokes.clear()
        self._manual_trace_drawing = False

    def clear_manual_trace(self):
        """Clear all manual trace state (current + confirmed)."""
        for rid in self._manual_trace_marker_ids:
            self.canvas.delete(rid)
        for rid in self._manual_trace_line_ids:
            self.canvas.delete(rid)
        self._manual_trace_points.clear()
        self._manual_trace_confirmed.clear()
        self._manual_trace_marker_ids.clear()
        self._manual_trace_line_ids.clear()
        self._manual_trace_strokes.clear()
        self._manual_trace_drawing = False

    def get_manual_trace_points(self):
        return list(self._manual_trace_points)

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

    def set_image_preview(self, img_np):
        """Set image for preview without clearing overlays or resetting zoom."""
        self._image_np = img_np
        if img_np is not None:
            if len(img_np.shape) == 2:
                self._pil_base = Image.fromarray(img_np, mode='L')
            else:
                self._pil_base = Image.fromarray(img_np)
        else:
            self._pil_base = None
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
        if self._review_zoom_state >= 0:
            show_plate_rect = False
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

        # bottom dots and connecting lines (manual endpoints mode)
        if not hide_markers and self._manual_endpoints and self._root_bottoms:
            bot_r = 3 if is_view else 4
            _clicking = self._mode in (self.MODE_CLICK_ROOTS,
                                        self.MODE_CLICK_MARKS)
            _active_pl = getattr(self, '_current_plate_idx', None)
            for ri, (brow, bcol) in self._root_bottoms.items():
                if ri >= len(self._root_points):
                    continue
                group = self._root_groups[ri] if ri < len(self._root_groups) else 0
                plate = self._root_plates[ri] if ri < len(self._root_plates) else 0
                if _clicking and _active_pl is not None \
                        and plate != _active_pl:
                    continue
                marker_color = GROUP_MARKER_COLORS[group % len(GROUP_MARKER_COLORS)]
                trow, tcol = self._root_points[ri]
                tcx, tcy = self.image_to_canvas(tcol, trow)
                bcx, bcy = self.image_to_canvas(bcol, brow)
                # bottom dot
                self.canvas.create_oval(
                    bcx - bot_r, bcy - bot_r, bcx + bot_r, bcy + bot_r,
                    outline="white", fill=marker_color, width=1)

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
                num_marks = max(1, self._marks_expected // max(1, len(self._marks_display_numbers))) if self._marks_display_numbers else 1
                for i, (row, col) in enumerate(self._mark_points):
                    cx, cy = self.image_to_canvas(col, row)
                    r = 5
                    rid = self.canvas.create_oval(
                        cx - r, cy - r, cx + r, cy + r,
                        outline="white", fill=color, width=1)
                    # show root display number instead of sequential index
                    group_idx = i // num_marks if num_marks > 0 else i
                    if group_idx < len(self._marks_display_numbers):
                        label = str(self._marks_display_numbers[group_idx])
                    else:
                        label = str(i + 1)
                    tid = self.canvas.create_text(
                        cx + 10, cy - 8, text=label,
                        fill=color, anchor="w",
                        font=("Helvetica", 8, "bold"))
                    self._mark_marker_ids.extend([rid, tid])

        # traced paths with segment coloring
        # In review mode, respect the traces visibility toggle
        _show_traces = (self._review_traces_visible
                        or self._mode not in (self.MODE_REVIEW,
                                              self.MODE_RECLICK,
                                              self.MODE_MANUAL_TRACE))
        trace_plate_counters = {}
        for ti, (path, shades, mark_indices) in enumerate(self._traces):
            if not _show_traces:
                continue
            if len(path) < 2:
                continue
            is_selected = ti in self._selected_for_retry
            w = 4 if is_selected else 2

            if is_selected:
                if self._mode == self.MODE_RECLICK:
                    self._draw_path_segment(path, "#ff6666", w, dash=(4, 12))
                else:
                    self._draw_path_segment(path, "#ff0000", w)
            elif mark_indices:
                boundaries = [0] + list(mark_indices) + [len(path) - 1]
                for j in range(len(boundaries) - 1):
                    start = boundaries[j]
                    end = boundaries[j + 1] + 1
                    color = shades[j % len(shades)]
                    self._draw_path_segment(path[start:end], color, w)
            else:
                self._draw_path_segment(path, shades[0], w)

            # Show number labels when traces are visible
            if len(path) > 0 and ti < len(self._trace_to_result):
                ri = self._trace_to_result[ti] if ti < len(self._trace_to_result) else ti
                plate = (self._root_plates[ri]
                         if ri < len(self._root_plates) else 0)
                group = (self._root_groups[ri]
                         if ri < len(self._root_groups) else 0)
                # Count all results 0..ri with same (plate, group), including dead/touching
                count = sum(1 for j in range(ri + 1)
                            if j < len(self._root_plates) and j < len(self._root_groups)
                            and self._root_plates[j] == plate
                            and self._root_groups[j] == group)
                top_row, top_col = path[0]
                lx, ly = self.image_to_canvas(top_col, top_row)
                lbl_color = "#ff0000" if is_selected else shades[0]
                self.canvas.create_text(
                    lx, ly - 10,
                    text=str(count),
                    fill=lbl_color, anchor="s",
                    font=("Helvetica", 10, "bold"))

        # root click dots when traces hidden in reclick/manual trace mode
        if (not _show_traces
                and self._mode in (self.MODE_RECLICK, self.MODE_MANUAL_TRACE)
                and self._root_points):
            dot_r = 5
            group_counters = {}
            for i, ((row, col), flag) in enumerate(
                    zip(self._root_points, self._root_flags)):
                group = self._root_groups[i] if i < len(self._root_groups) else 0
                plate = self._root_plates[i] if i < len(self._root_plates) else 0
                key = (plate, group)
                group_counters[key] = group_counters.get(key, 0) + 1
                cx, cy = self.image_to_canvas(col, row)
                marker_color = GROUP_MARKER_COLORS[group % len(GROUP_MARKER_COLORS)]
                if flag is not None:
                    label = "DEAD" if flag == 'dead' else "TOUCH"
                    cross_s = 6
                    self.canvas.create_line(
                        cx - cross_s, cy - cross_s, cx + cross_s, cy + cross_s,
                        fill=marker_color, width=2)
                    self.canvas.create_line(
                        cx - cross_s, cy + cross_s, cx + cross_s, cy - cross_s,
                        fill=marker_color, width=2)
                    self.canvas.create_text(
                        cx + 10, cy - 10, text=f"{group_counters[key]} {label}",
                        fill=marker_color, anchor="w",
                        font=("Helvetica", 9, "bold"))
                else:
                    self.canvas.create_oval(
                        cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r,
                        outline="white", fill=marker_color, width=1)
                    self.canvas.create_text(
                        cx + 10, cy - 10, text=str(group_counters[key]),
                        fill=marker_color, anchor="w",
                        font=("Helvetica", 9, "bold"))

        # dead/touching seedling markers when traces are visible
        if len(self._traces) > 0 and _show_traces:
            cross_s = 6
            for i, ((row, col), flag) in enumerate(
                    zip(self._root_points, self._root_flags)):
                if flag is None:
                    continue
                group = self._root_groups[i] if i < len(self._root_groups) else 0
                plate = self._root_plates[i] if i < len(self._root_plates) else 0
                # Count all results 0..i with same (plate, group)
                count = sum(1 for j in range(i + 1)
                            if j < len(self._root_plates) and j < len(self._root_groups)
                            and self._root_plates[j] == plate
                            and self._root_groups[j] == group)
                cx, cy = self.image_to_canvas(col, row)
                marker_color = GROUP_MARKER_COLORS[group % len(GROUP_MARKER_COLORS)]
                label = "DEAD" if flag == 'dead' else "TOUCH"
                self.canvas.create_line(
                    cx - cross_s, cy - cross_s, cx + cross_s, cy + cross_s,
                    fill=marker_color, width=2)
                self.canvas.create_line(
                    cx - cross_s, cy + cross_s, cx + cross_s, cy - cross_s,
                    fill=marker_color, width=2)
                self.canvas.create_text(
                    cx + 10, cy - 10, text=f"{count} {label}",
                    fill=marker_color, anchor="w",
                    font=("Helvetica", 9, "bold"))

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
            for rid in self._reclick_marker_ids:
                self.canvas.tag_raise(rid)

        # manual trace preview (confirmed + in-progress)
        if self._mode == self.MODE_MANUAL_TRACE:
            for rid in self._manual_trace_marker_ids:
                self.canvas.delete(rid)
            for rid in self._manual_trace_line_ids:
                self.canvas.delete(rid)
            self._manual_trace_marker_ids.clear()
            self._manual_trace_line_ids.clear()
            # draw confirmed traces from previous roots
            for confirmed in self._manual_trace_confirmed:
                for i, (row, col) in enumerate(confirmed):
                    cx, cy = self.image_to_canvas(col, row)
                    r = 4
                    rid = self.canvas.create_oval(
                        cx - r, cy - r, cx + r, cy + r,
                        outline="white", fill="#00cc99", width=1)
                    self._manual_trace_marker_ids.append(rid)
                    if i > 0:
                        prev_row, prev_col = confirmed[i - 1]
                        px, py = self.image_to_canvas(prev_col, prev_row)
                        lid = self.canvas.create_line(
                            px, py, cx, cy,
                            fill="#00cc99", width=2)
                        self._manual_trace_line_ids.append(lid)
            # draw current in-progress trace
            if self._manual_trace_submode == 'freehand':
                # Freehand: solid lines only, no dots
                for i in range(1, len(self._manual_trace_points)):
                    prev_row, prev_col = self._manual_trace_points[i - 1]
                    row, col = self._manual_trace_points[i]
                    px, py = self.image_to_canvas(prev_col, prev_row)
                    cx, cy = self.image_to_canvas(col, row)
                    lid = self.canvas.create_line(
                        px, py, cx, cy, fill="#00ffff", width=2)
                    self._manual_trace_line_ids.append(lid)
            else:
                # Segmented: dots + dashed lines
                for i, (row, col) in enumerate(self._manual_trace_points):
                    cx, cy = self.image_to_canvas(col, row)
                    r = 4
                    rid = self.canvas.create_oval(
                        cx - r, cy - r, cx + r, cy + r,
                        outline="white", fill="#00ffff", width=1)
                    self._manual_trace_marker_ids.append(rid)
                    if i > 0:
                        prev_row, prev_col = self._manual_trace_points[i - 1]
                        px, py = self.image_to_canvas(prev_col, prev_row)
                        lid = self.canvas.create_line(
                            px, py, cx, cy,
                            fill="#00ffff", width=2, dash=(6, 3))
                        self._manual_trace_line_ids.append(lid)
            all_ids = self._manual_trace_marker_ids + self._manual_trace_line_ids
            for rid in all_ids:
                self.canvas.tag_raise(rid)

        # plate info overlay (shown when zoomed into a plate)
        info = getattr(self, '_plate_info', None)
        if info and self._mode in (self.MODE_CLICK_ROOTS, self.MODE_CLICK_MARKS,
                                    self.MODE_REVIEW, self.MODE_RECLICK,
                                    self.MODE_MANUAL_TRACE):
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            y = ch - 6
            _fnt = ("Helvetica", 16, "bold")
            _bg = "#d8d8d8"
            _pad = 3
            # center: "Plate N"
            tid = self.canvas.create_text(
                cw / 2, y, text=info.get('center', ''),
                fill="black", anchor="s", font=_fnt)
            bb = self.canvas.bbox(tid)
            if bb:
                bg = self.canvas.create_rectangle(
                    bb[0] - _pad, bb[1] - _pad, bb[2] + _pad, bb[3] + _pad,
                    fill=_bg, outline=_bg)
                self.canvas.tag_lower(bg, tid)
            # left: genotype A + condition
            if info.get('left'):
                tid = self.canvas.create_text(
                    15, y, text=info['left'],
                    fill=info.get('left_color', '#cccccc'), anchor="sw",
                    font=_fnt)
                bb = self.canvas.bbox(tid)
                if bb:
                    bg = self.canvas.create_rectangle(
                        bb[0] - _pad, bb[1] - _pad, bb[2] + _pad, bb[3] + _pad,
                        fill=_bg, outline=_bg)
                    self.canvas.tag_lower(bg, tid)
            # right: genotype B + condition (split plates only)
            if info.get('right'):
                tid = self.canvas.create_text(
                    cw - 15, y, text=info['right'],
                    fill=info.get('right_color', '#cccccc'), anchor="se",
                    font=_fnt)
                bb = self.canvas.bbox(tid)
                if bb:
                    bg = self.canvas.create_rectangle(
                        bb[0] - _pad, bb[1] - _pad, bb[2] + _pad, bb[3] + _pad,
                        fill=_bg, outline=_bg)
                    self.canvas.tag_lower(bg, tid)

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
                ("Click", "Place root top / mark / bottom"
                          if self._manual_endpoints else "Place root top"),
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
            self.MODE_MANUAL_TRACE: [
                ("Click", "Add point along root"),
                ("Enter", "Confirm manual trace"),
                ("Right-click", "Undo last point"),
            ],
        }
        _MODE_NAMES = {
            self.MODE_VIEW: "View",
            self.MODE_SELECT_PLATES: "Select Plates",
            self.MODE_CLICK_ROOTS: "Click Roots",
            self.MODE_CLICK_MARKS: "Click Marks",
            self.MODE_REVIEW: "Review",
            self.MODE_RECLICK: "Re-click",
            self.MODE_MANUAL_TRACE: "Manual Trace",
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
        """Two-finger scroll = pan vertically."""
        if self._image_np is None or self._manual_trace_drawing:
            return
        # macOS delta is ±1..±N per tick; multiply for smooth pan speed
        dy = event.delta * 3
        self._offset_y += dy
        self._user_zoomed = True
        self._fast_redraw()

    def _on_scroll_h(self, event):
        """Shift + scroll = pan horizontally."""
        if self._image_np is None or self._manual_trace_drawing:
            return
        dx = event.delta * 3
        self._offset_x += dx
        self._user_zoomed = True
        self._fast_redraw()

    def _on_pinch_zoom(self, event):
        """Pinch-to-zoom (macOS: Ctrl+MouseWheel / Opt+MouseWheel)."""
        if self._image_np is None or self._manual_trace_drawing:
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
        elif self._mode == self.MODE_MANUAL_TRACE:
            if self._manual_trace_submode == 'freehand':
                # Start freehand drawing stroke
                col, row = self.canvas_to_image(event.x, event.y)
                self._manual_trace_strokes.append(len(self._manual_trace_points))
                self._manual_trace_points.append((row, col))
                self._manual_trace_drawing = True
                self._redraw()
            else:
                # Segmented: place point immediately (don't defer)
                col, row = self.canvas_to_image(event.x, event.y)
                self._manual_trace_points.append((row, col))
                self._redraw()
            if self._on_click_callback:
                self._on_click_callback()
        else:
            # defer click — may become a pan drag
            self._click_start = (event.x, event.y)
            self._is_panning = False
            self._drag_start = (event.x, event.y)

    def _execute_click(self, click_pos):
        """Execute the deferred click action at the press location."""
        sx, sy = click_pos
        if self._mode == self.MODE_CLICK_ROOTS:
            col, row = self.canvas_to_image(sx, sy)
            if not self._manual_endpoints:
                # Auto-detect mode: each click is a root top
                flag = self._pending_flag
                self._pending_flag = None
                self._root_points.append((row, col))
                self._root_flags.append(flag)
                self._root_groups.append(getattr(self, '_current_root_group', 0))
                self._root_plates.append(getattr(self, '_current_plate_idx', 0))
            else:
                # Manual endpoints: cycle through top / marks / bottom per root
                cpr = self._clicks_per_root
                if self._click_seq_pos == 0:
                    # TOP click for a new root
                    flag = self._pending_flag
                    self._pending_flag = None
                    self._root_points.append((row, col))
                    self._root_flags.append(flag)
                    self._root_groups.append(getattr(self, '_current_root_group', 0))
                    self._root_plates.append(getattr(self, '_current_plate_idx', 0))
                    if flag is not None:
                        # Dead/touching: skip remaining clicks for this root
                        pass
                    else:
                        self._click_seq_pos = 1
                elif self._click_seq_pos < cpr - 1:
                    # MARK click (intermediate)
                    root_idx = len(self._root_points) - 1
                    if root_idx not in self._all_marks:
                        self._all_marks[root_idx] = []
                    self._all_marks[root_idx].append((row, col))
                    self._click_seq_pos += 1
                else:
                    # BOTTOM click (last in sequence)
                    root_idx = len(self._root_points) - 1
                    self._root_bottoms[root_idx] = (row, col)
                    self._click_seq_pos = 0
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()
        elif self._mode == self.MODE_CLICK_MARKS:
            # stop accepting clicks once expected count reached
            if self._marks_expected > 0 and \
               len(self._mark_points) >= self._marks_expected:
                return
            col, row = self.canvas_to_image(sx, sy)
            self._mark_points.append((row, col))
            # draw mark as genotype-colored circle (light shade)
            cx, cy = sx, sy
            current_group = getattr(self, '_current_root_group', 0)
            color = GROUP_MARK_COLORS[current_group % len(GROUP_MARK_COLORS)]
            n = len(self._mark_points)
            # compute display label from root position numbers
            num_m = max(1, self._marks_expected // max(1, len(self._marks_display_numbers))) if self._marks_display_numbers else 1
            group_idx = (n - 1) // num_m
            if group_idx < len(self._marks_display_numbers):
                label = str(self._marks_display_numbers[group_idx])
            else:
                label = str(n)
            r = 5
            rid = self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline="white", fill=color, width=1)
            tid = self.canvas.create_text(
                cx + 10, cy - 8, text=label,
                fill=color, anchor="w",
                font=("Helvetica", 8, "bold"))
            self._mark_marker_ids.extend([rid, tid])
            if self._on_click_callback:
                self._on_click_callback()
        elif self._mode == self.MODE_REVIEW:
            col, row = self.canvas_to_image(sx, sy)
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
            col, row = self.canvas_to_image(sx, sy)
            self._reclick_points.append((row, col))
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()
        elif self._mode == self.MODE_MANUAL_TRACE:
            # segmented clicks are handled in _on_left_press directly;
            # this path only reached for edge cases
            pass

    _PAN_THRESHOLD = 20  # pixels of drag before switching to pan

    def _on_left_drag(self, event):
        # Freehand drawing: capture points instead of panning
        if (self._mode == self.MODE_MANUAL_TRACE
                and self._manual_trace_submode == 'freehand'
                and self._manual_trace_drawing
                and not self._space_held):
            col, row = self.canvas_to_image(event.x, event.y)
            if self._manual_trace_points:
                lr, lc = self._manual_trace_points[-1]
                if (row - lr) ** 2 + (col - lc) ** 2 < 36:  # 6px spacing
                    return
            self._manual_trace_points.append((row, col))
            # Incremental draw: add line segment without full redraw
            if len(self._manual_trace_points) >= 2:
                prev_row, prev_col = self._manual_trace_points[-2]
                px, py = self.image_to_canvas(prev_col, prev_row)
                cx, cy = self.image_to_canvas(col, row)
                lid = self.canvas.create_line(
                    px, py, cx, cy, fill="#00ffff", width=2)
                self._manual_trace_line_ids.append(lid)
                self.canvas.tag_raise(lid)
            return
        if self._drag_start is not None and (self._space_held or self._is_panning
                                             or self._mode != self.MODE_SELECT_PLATES):
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            if not self._is_panning and not self._space_held:
                # check if drag exceeds threshold before panning (euclidean distance)
                sx, sy = self._click_start or (event.x, event.y)
                dist = ((event.x - sx) ** 2 + (event.y - sy) ** 2) ** 0.5
                if dist < self._PAN_THRESHOLD:
                    return
                self._is_panning = True
                self.canvas.configure(cursor="fleur")
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
        # End freehand stroke on mouse release (check before space_held)
        if (self._mode == self.MODE_MANUAL_TRACE
                and self._manual_trace_submode == 'freehand'
                and self._manual_trace_drawing):
            self._manual_trace_drawing = False
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()
            return
        if self._space_held:
            self._drag_start = None
            return
        # finish deferred click-or-pan for non-plate modes
        if self._click_start is not None:
            was_panning = self._is_panning
            click_pos = self._click_start
            if was_panning:
                self.canvas.configure(cursor="")
            self._click_start = None
            self._is_panning = False
            self._drag_start = None
            if not was_panning:
                self._execute_click(click_pos)
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
            if self._on_click_callback:
                self._on_click_callback()

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
            if not self._manual_endpoints:
                # Auto mode: undo last root top click
                self._root_points.pop()
                self._root_flags.pop()
                self._root_groups.pop()
                self._root_plates.pop()
            else:
                # Manual endpoints: undo depends on position in sequence
                if self._click_seq_pos == 0:
                    # Completed root or flagged root — step back
                    root_idx = len(self._root_points) - 1
                    if self._root_flags[-1] is not None:
                        # Previous root was flagged (only had top) — remove it
                        self._root_points.pop()
                        self._root_flags.pop()
                        self._root_groups.pop()
                        self._root_plates.pop()
                    elif root_idx in self._root_bottoms:
                        # Undo bottom click
                        del self._root_bottoms[root_idx]
                        self._click_seq_pos = self._clicks_per_root - 1
                    else:
                        # Root only had top (shouldn't happen if seq completed)
                        self._root_points.pop()
                        self._root_flags.pop()
                        self._root_groups.pop()
                        self._root_plates.pop()
                elif self._click_seq_pos == 1:
                    # Undo top click of current root
                    self._root_points.pop()
                    self._root_flags.pop()
                    self._root_groups.pop()
                    self._root_plates.pop()
                    self._click_seq_pos = 0
                else:
                    # Undo a mark click
                    root_idx = len(self._root_points) - 1
                    if root_idx in self._all_marks and self._all_marks[root_idx]:
                        self._all_marks[root_idx].pop()
                        if not self._all_marks[root_idx]:
                            del self._all_marks[root_idx]
                    self._click_seq_pos -= 1
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()
            return True
        elif self._mode == self.MODE_CLICK_MARKS and self._mark_points:
            self._mark_points.pop()
            for _ in range(2):
                if self._mark_marker_ids:
                    self.canvas.delete(self._mark_marker_ids.pop())
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()
            return True
        elif self._mode == self.MODE_RECLICK and self._reclick_points:
            self._reclick_points.pop()
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()
            return True
        elif self._mode == self.MODE_MANUAL_TRACE and self._manual_trace_points:
            if (self._manual_trace_submode == 'freehand'
                    and self._manual_trace_strokes):
                stroke_start = self._manual_trace_strokes.pop()
                del self._manual_trace_points[stroke_start:]
            else:
                self._manual_trace_points.pop()
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
                    # confirm pending plate — let _on_plate_added decide next step
                    self._plates.append(self._pending_plate)
                    self._pending_plate = None
                    self._redraw()
                    if self._on_click_callback:
                        self._on_click_callback()
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
