"""Measurement workflow — tracing, review, retry, and CSV saving."""

import cv2
import numpy as np
from datetime import datetime

def _log(msg):
    """Print debug message with timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

from image_processing import preprocess
from root_tracing import find_root_tip, trace_root, build_plate_graph
from utils import _compute_segments, _find_nearest_path_index
from csv_output import append_results_to_csv, save_metadata
from plotting import plot_results

from canvas import ImageCanvas
from session import data_dir, traces_dir


def _interpolate_manual_path(points):
    """Interpolate between clicked points at ~1px spacing."""
    if len(points) < 2:
        return np.array(points)
    all_pts = []
    for i in range(len(points) - 1):
        r1, c1 = points[i]
        r2, c2 = points[i + 1]
        dist = np.sqrt((r2 - r1)**2 + (c2 - c1)**2)
        n_interp = max(2, int(np.ceil(dist)))
        rows = np.linspace(r1, r2, n_interp)
        cols = np.linspace(c1, c2, n_interp)
        if i > 0:
            rows = rows[1:]
            cols = cols[1:]
        all_pts.extend(zip(rows, cols))
    return np.array(all_pts)

# genotype color shades: [bright, pastel] for trace segments (30 entries)
# First 8 are Okabe-Ito (colorblind-friendly), then extended hues
GROUP_COLORS = [
    ["#0072B2", "#A0D4F0"],  # 0: blue (OI)
    ["#E69F00", "#F5D89A"],  # 1: orange (OI)
    ["#D55E00", "#F0BFA0"],  # 2: vermilion (OI)
    ["#56B4E9", "#B8DDF5"],  # 3: sky blue (OI)
    ["#009E73", "#A0E8D0"],  # 4: green (OI)
    ["#F0E442", "#F8F2B0"],  # 5: yellow (OI)
    ["#CC79A7", "#E8C4DA"],  # 6: pink (OI)
    ["#AA4400", "#E0C4A0"],  # 7: brown (OI)
    ["#882255", "#D4A0C0"],  # 8: wine
    ["#44AA99", "#B0DDD4"],  # 9: teal
    ["#AA4499", "#D8B0D8"],  # 10: magenta
    ["#999933", "#D4D4A0"],  # 11: olive
    ["#6699CC", "#B8CCE0"],  # 12: steel blue
    ["#DD7788", "#F0C0CC"],  # 13: rose
    ["#117733", "#A0D4A0"],  # 14: forest
    ["#88CCEE", "#C4E8F0"],  # 15: light blue
    ["#CC6677", "#E0B8C4"],  # 16: dusty rose
    ["#DDCC77", "#F0E8B8"],  # 17: sand
    ["#332288", "#B0A8D8"],  # 18: indigo
    ["#44BB99", "#B0DDD4"],  # 19: mint
    ["#EE8866", "#F5C8B0"],  # 20: salmon
    ["#BBCC33", "#D8E0A0"],  # 21: lime
    ["#EEDD88", "#F5F0C0"],  # 22: khaki
    ["#77AADD", "#C0D4F0"],  # 23: periwinkle
    ["#EE6677", "#F0B8C4"],  # 24: coral
    ["#66CCEE", "#B8E8F5"],  # 25: cyan
    ["#AA3377", "#D8A0C4"],  # 26: plum
    ["#BBBB33", "#D8D8A0"],  # 27: chartreuse
    ["#4477AA", "#B0C4E0"],  # 28: cobalt
    ["#228833", "#A8D8A8"],  # 29: emerald
]


class MeasurementMixin:
    """Mixin for RootMeasureApp that handles tracing, review, and saving.

    Expects the host class to provide:
        self.image, self.image_path, self.folder
        self.canvas (ImageCanvas), self.sidebar (Sidebar)
        self.lbl_bottom, self.update_idletasks()
        self._get_scale(), self._get_num_marks()
    """

    def _get_root_shades(self, root_idx):
        """Return color shades for a root based on its genotype group."""
        groups = self.canvas.get_root_groups()
        if root_idx < len(groups):
            color_idx = groups[root_idx] % len(GROUP_COLORS)
        else:
            color_idx = 0
        return GROUP_COLORS[color_idx]

    def _add_root_trace(self, root_idx, res):
        """Add a traced root to the canvas with genotype coloring."""
        path = res['path']
        if path.size == 0:
            return
        shades = self._get_root_shades(root_idx)
        mark_indices = []
        mark_coords = res.get('mark_coords', [])
        if mark_coords and len(res.get('segments', [])) > 1:
            for mc in mark_coords:
                mark_indices.append(_find_nearest_path_index(path, mc))
            mark_indices.sort()
        self.canvas.add_trace(path, shades=shades, mark_indices=mark_indices)

    def measure(self):
        """Run preprocessing, tracing, and show results for review."""
        _log("measure() called")
        self._exit_preview()
        points = self.canvas.get_root_points()
        flags = self.canvas.get_root_flags()
        plates = self.canvas.get_plates()
        _log(f"  {len(points)} root points, {len(plates)} plates")

        if not points or self.image is None:
            self.sidebar.set_status("Nothing to measure.")
            return

        self._scale_val = self._get_scale()
        self._sensitivity = self.sidebar.var_sensitivity.get()
        _log(f"  scale={self._scale_val}, sensitivity={self._sensitivity}")

        self._hide_action_buttons()
        self.sidebar.set_status("Preprocessing...")
        self.sidebar.btn_measure.configure(state="disabled")
        self.sidebar.btn_select_plates.configure(state="disabled")
        self.sidebar.btn_click_roots.configure(state="disabled")
        self.sidebar.btn_review.configure(state="disabled")
        self.sidebar.set_step(3)
        self.update()

        self._threshold = self.sidebar.get_threshold()
        self._binary = preprocess(self.image, scale=self._scale_val,
                                  sensitivity=self._sensitivity,
                                  threshold=self._threshold)

        self.canvas.clear_traces()
        self._results = []
        # map trace index -> result index (skip flagged roots with no trace)
        self._trace_to_result = []

        # pre-build skeleton + graph per plate (once each)
        self._plate_graphs = {}
        self.sidebar.show_progress(len(plates))
        self.update()
        for pi, bounds in enumerate(plates):
            self.sidebar.set_status(
                f"Building skeleton for plate {pi + 1}/{len(plates)}...")
            self.sidebar.update_progress(pi + 1)
            self.update()
            self._plate_graphs[pi] = build_plate_graph(
                self._binary, bounds, self._scale_val)
        self.sidebar.hide_progress()

        root_plates = self.canvas._root_plates

        num_marks = self._get_num_marks()
        total = len(points)
        self.sidebar.show_progress(total)
        self.update()

        for i, (top, flag) in enumerate(zip(points, flags)):
            self.sidebar.set_status(f"Tracing root {i + 1}/{total}...")
            self.sidebar.update_progress(i + 1)
            self.update()

            if flag is not None:
                warning = 'dead seedling' if flag == 'dead' else 'roots touching'
                res = dict(length_cm=None, length_px=None,
                           path=np.empty((0, 2)),
                           method='skip', warning=warning, segments=[])
                self._results.append(res)
                continue

            pi = root_plates[i] if i < len(root_plates) else 0
            pb = plates[pi] if pi < len(plates) else None
            pg = self._plate_graphs.get(pi)

            manual_bottoms = self.canvas.get_root_bottoms()
            if i in manual_bottoms:
                # Manual endpoints mode: use user-provided bottom
                tip = manual_bottoms[i]
            else:
                # Auto-detect mode: find tip automatically
                tip = find_root_tip(self._binary, top, scale=self._scale_val,
                                    plate_bounds=pb, plate_graph=pg)
            if tip is None:
                res = dict(length_cm=0, length_px=0,
                           path=np.empty((0, 2)),
                           method='error', warning='Could not find root tip',
                           segments=[])
                self._results.append(res)
                continue

            res = trace_root(self._binary, top, tip, self._scale_val,
                             plate_bounds=pb, plate_graph=pg)
            # compute segments if marks were collected for this root
            mark_coords = self.canvas._all_marks.get(i, [])
            if mark_coords and res['path'].size > 0:
                res['segments'] = _compute_segments(
                    res['path'], mark_coords, self._scale_val)
                res['mark_coords'] = mark_coords
                seg_str = " + ".join(f"{s:.2f}" for s in res['segments'])
                print(f"  Root {i + 1}: {res['length_cm']:.2f} cm "
                      f"(segments: {seg_str})")
            else:
                res['segments'] = []
                if res.get('length_cm'):
                    print(f"  Root {i + 1}: {res['length_cm']:.2f} cm")
            self._results.append(res)

            if res['path'].size > 0:
                self._add_root_trace(i, res)
                self._trace_to_result.append(i)

        # enter review mode
        self._show_review()

    def _show_review(self, skip_delay=False):
        """Show traced results and let user click bad traces to retry."""
        _log(f"_show_review(skip_delay={skip_delay})")
        _log(f"  {len(self._traces if hasattr(self, '_traces') else self.canvas._traces)} traces")
        self.sidebar.hide_progress()
        self.sidebar.set_step(4)
        # Re-enable workflow buttons so user can go back to previous steps
        self.sidebar.btn_select_plates.configure(state="normal")
        self.sidebar.btn_click_roots.configure(state="normal")
        self.sidebar.btn_measure.configure(state="normal")
        self.sidebar.btn_review.configure(state="normal")
        # sync trace-to-result mapping so canvas review numbering is correct
        self.canvas._trace_to_result = list(self._trace_to_result)
        # ensure traces are visible when entering review
        self.canvas._review_traces_visible = True
        self.canvas._review_zoom_state = -1
        # Block Enter for 500ms to prevent leftover keypresses from skipping review
        # (skip delay when returning from retrace to avoid double-enter)
        if skip_delay:
            self._review_ready = True
        else:
            self._review_ready = False
            self.after(500, self._enable_review)
        self.canvas.set_mode(
            ImageCanvas.MODE_REVIEW,
            on_done=self._review_done)
        # Full view — no single plate info to show
        self._clear_plate_info()
        self.canvas._fit_image()
        self.canvas._redraw()

        traced = [r for r in self._results
                  if r['method'] not in ('skip', 'error')]
        lengths = [r['length_cm'] for r in traced if r['length_cm']]
        n_with_segs = sum(1 for r in traced if r.get('segments'))
        msg = f"Traced {len(traced)} root(s)."
        if lengths:
            msg += f" Mean: {np.mean(lengths):.2f} cm"
        if n_with_segs:
            msg += f"\n{n_with_segs} root(s) with segments."
        msg += "\nClick a bad trace to select for retry."
        msg += "\nPress Enter to accept and continue."
        self.sidebar.set_status(msg)
        self._hide_action_buttons()
        self._show_action_frame()
        self.sidebar.btn_done.configure(text="Accept All")
        # Clear callback before setting new one
        self.canvas._on_click_callback = None
        self.canvas._on_click_callback = self._update_review_button
        self.sidebar.btn_done.pack(pady=(5, 0), padx=15, fill="x")
        self.btn_continue_later_mid.pack(pady=(10, 5), padx=15, fill="x")
        self.sidebar.show_review_toggles()
        self.lbl_bottom.configure(
            text="Click trace=select for retry (red)  |  Enter=accept  |  Scroll=zoom")


    def _enable_review(self):
        self._review_ready = True

    def _update_review_button(self):
        """Update review button text based on selection."""
        selected = self.canvas.get_selected_for_retry()
        if selected:
            n = len(selected)
            self.sidebar.btn_done.configure(
                text=f"Auto Retrace ({n})", fg_color="#2b5797")
            self.sidebar.btn_manual_trace.configure(
                text=f"Manual Trace ({n})", fg_color="#2b5797")
            if not self.sidebar.btn_manual_trace.winfo_ismapped():
                self.sidebar.btn_manual_trace.pack(pady=(3, 0), padx=15, fill="x")
            self.sidebar.show_manual_trace_modes()
        else:
            self.sidebar.btn_done.configure(
                text="Accept All", fg_color="#2b5797")
            self.sidebar.btn_manual_trace.pack_forget()
            self.sidebar.hide_manual_trace_modes()

    def _review_done(self):
        """Called when user presses Enter in review mode."""
        _log("_review_done() called")
        if not getattr(self, '_review_ready', True):
            _log("  ignoring - review not ready yet")
            return  # ignore leftover Enter from previous mode
        selected = self.canvas.get_selected_for_retry()
        _log(f"  selected for retry: {selected}")
        if not selected:
            # accept all — save and finish
            _log("  no selection → finishing measurement")
            self.canvas.set_mode(ImageCanvas.MODE_VIEW)
            self._finish_measurement()
            return
        # map trace indices to result indices
        self._retry_result_indices = [self._trace_to_result[s]
                                       for s in selected
                                       if s < len(self._trace_to_result)]
        _log(f"  retry_result_indices: {self._retry_result_indices}")
        if not self._retry_result_indices:
            _log("  no valid indices → finishing measurement")
            self.canvas.set_mode(ImageCanvas.MODE_VIEW)
            self._finish_measurement()
            return
        # enter reclick mode
        _log("  → entering reclick mode")
        self._start_reclick()

    def _start_reclick(self):
        """Enter reclick mode for bad traces."""
        _log("_start_reclick() called")
        self._hide_action_buttons()
        self.canvas.clear_reclick()
        self._reclick_idx = 0  # which retry root we're on
        num_marks = self._get_num_marks()
        self._reclick_clicks_per_root = 2 + num_marks
        self.canvas._reclick_clicks_per_root = self._reclick_clicks_per_root
        n = len(self._retry_result_indices)
        _log(f"  {n} roots to retry, {self._reclick_clicks_per_root} clicks per root")
        self.canvas._reclick_expected = n * self._reclick_clicks_per_root
        self.canvas.set_mode(
            ImageCanvas.MODE_RECLICK,
            on_done=self._reclick_enter)
        # zoom to first bad root's plate
        ri = self._retry_result_indices[0]
        points = self.canvas.get_root_points()
        if ri < len(points):
            top = points[ri]
            plates = self.canvas.get_plates()
            zoomed = False
            for r1, r2, c1, c2 in plates:
                if r1 <= top[0] <= r2 and c1 <= top[1] <= c2:
                    self.canvas.zoom_to_region(r1, r2, c1, c2)
                    zoomed = True
                    break
            if not zoomed and plates:
                self.canvas.zoom_to_region(*plates[0])
        elif self.canvas.get_plates():
            self.canvas.zoom_to_region(*self.canvas.get_plates()[0])
        self._show_reclick_status()
        self._hide_action_buttons()
        self._show_action_frame()
        # Update button text based on progress
        n = len(self._retry_result_indices)
        is_last = self._reclick_idx >= n - 1
        btn_text = "Finish" if is_last else "Next Root"
        self.sidebar.btn_done.configure(text=btn_text)
        self.sidebar.btn_done.pack(pady=(5, 0), padx=15, fill="x")
        self.btn_continue_later_mid.pack(pady=(10, 5), padx=15, fill="x")
        self.canvas._review_zoom_state = 0  # already zoomed to a plate
        self.sidebar.show_review_toggles()
        n = len(self.canvas._plates)
        if n > 1:
            self.sidebar.btn_toggle_zoom.configure(
                text="Zoom In Plate 2", fg_color="#2b5797")
        else:
            self.sidebar.btn_toggle_zoom.configure(
                text="Zoom Out", fg_color="#2b5797")

    def _show_reclick_status(self):
        """Show status for current reclick root."""
        n = len(self._retry_result_indices)
        pi = self._reclick_idx + 1
        cpr = self._reclick_clicks_per_root
        if cpr == 2:
            click_desc = "click 1 (top) then 2 (bottom)"
            bottom_text = "Click 1=top, 2=bottom"
        else:
            nums = ", ".join(str(i) for i in range(1, cpr + 1))
            click_desc = f"click {nums} (top to bottom)"
            bottom_text = f"Click {nums} (top to bottom)"
        self.sidebar.set_status(
            f"Re-click root {pi}/{n}: {click_desc}.\n"
            "Press Enter when done.")
        self.lbl_bottom.configure(
            text=f"{bottom_text}  |  Right-click=undo  |  Enter=done  |  Scroll=zoom")
        # Update button text
        is_last = self._reclick_idx >= n - 1
        btn_text = "Finish" if is_last else "Next Root"
        self.sidebar.btn_done.configure(text=btn_text)

    def _reclick_enter(self):
        """Called when user presses Enter during reclick."""
        _log("_reclick_enter() called")
        pts = self.canvas._reclick_points
        cpr = self._reclick_clicks_per_root
        n = len(self._retry_result_indices)
        _log(f"  {len(pts)} points clicked, need {(self._reclick_idx + 1) * cpr} for root {self._reclick_idx + 1}/{n}")
        # need at least enough points for the current root
        if len(pts) < (self._reclick_idx + 1) * cpr:
            self.sidebar.set_status(
                f"Need {cpr} clicks per root. "
                f"Click all points before pressing Enter.")
            return
        # advance past all roots that already have their points placed
        while self._reclick_idx < n - 1:
            self._reclick_idx += 1
            if len(pts) < (self._reclick_idx + 1) * cpr:
                # this root still needs clicks — stay here
                ri = self._retry_result_indices[self._reclick_idx]
                points = self.canvas.get_root_points()
                if ri < len(points):
                    top = points[ri]
                    plates = self.canvas.get_plates()
                    zoomed = False
                    for r1, r2, c1, c2 in plates:
                        if r1 <= top[0] <= r2 and c1 <= top[1] <= c2:
                            self.canvas.zoom_to_region(r1, r2, c1, c2)
                            zoomed = True
                            break
                    if not zoomed and plates:
                        self.canvas.zoom_to_region(*plates[0])
                self._show_reclick_status()
                return
        # all re-clicks done — re-trace
        _log("  all reclick points placed → retracing")
        self._do_retrace()

    def _do_retrace(self):
        """Re-trace roots with manually clicked points."""
        _log("_do_retrace() called")
        self._hide_action_buttons()
        # ensure binary mask exists (may be missing after session restore)
        if not hasattr(self, '_binary') or self._binary is None:
            self.sidebar.set_status("Preprocessing...")
            self.update()
            self._threshold = self.sidebar.get_threshold()
            self._binary = preprocess(self.image, scale=self._scale_val,
                                      sensitivity=self._sensitivity,
                                      threshold=self._threshold)
        cpr = self._reclick_clicks_per_root
        groups = self.canvas.get_reclick_groups(cpr)
        self.canvas.set_mode(ImageCanvas.MODE_VIEW)

        total = min(len(self._retry_result_indices), len(groups))
        self.sidebar.show_progress(total)
        self.update()

        plates = self.canvas.get_plates()
        plate_graphs = getattr(self, '_plate_graphs', {})

        for j, ri in enumerate(self._retry_result_indices):
            if j >= len(groups):
                break
            clicks = groups[j]
            top_manual = clicks[0]
            bot_manual = clicks[-1]
            _log(f"  retracing root {j + 1}/{total} (result idx {ri})")
            _log(f"    top={top_manual}, bot={bot_manual}")
            self.sidebar.set_status(f"Re-tracing root {j + 1}/{total}...")
            self.sidebar.update_progress(j + 1)
            self.update()
            # determine plate from click position
            pi = None
            for pidx, (r1, r2, c1, c2) in enumerate(plates):
                if r1 <= top_manual[0] <= r2 and c1 <= top_manual[1] <= c2:
                    pi = pidx
                    break
            pb = plates[pi] if pi is not None else None
            pg = plate_graphs.get(pi) if pi is not None else None
            res = trace_root(self._binary, top_manual, bot_manual,
                             self._scale_val, plate_bounds=pb, plate_graph=pg)
            _log(f"    traced: {res['length_cm']:.2f} cm, method={res['method']}")
            # use reclick marks if provided (clicks between top and bottom)
            if cpr > 2:
                mark_coords = list(clicks[1:-1])
                self.canvas._all_marks[ri] = mark_coords
            else:
                mark_coords = self.canvas._all_marks.get(ri, [])
            if mark_coords and res['path'].size > 0:
                res['segments'] = _compute_segments(
                    res['path'], mark_coords, self._scale_val)
                res['mark_coords'] = mark_coords
            else:
                res['segments'] = []
            self._results[ri] = res

        self._rebuild_all_traces()

        self.sidebar.hide_progress()

        # Go back to review mode so user can verify retrace worked
        _log("  → returning to review mode")
        self._show_review(skip_delay=True)

    def _rebuild_all_traces(self):
        """Rebuild all canvas traces from self._results."""
        self.canvas.clear_traces()
        self._trace_to_result.clear()
        for i, res in enumerate(self._results):
            if res['path'].size > 0 and res['method'] not in ('skip', 'error'):
                self._add_root_trace(i, res)
                self._trace_to_result.append(i)
        _log(f"  rebuilt {len(self.canvas._traces)} traces")
        self.canvas._redraw()

    # --- Manual trace flow ---

    def _start_manual_trace_with_mode(self, mode):
        """Enter manual trace with chosen sub-mode."""
        self.sidebar.hide_manual_trace_modes()
        self.canvas._manual_trace_submode = mode
        self._start_manual_trace()

    def _start_manual_trace(self):
        """Enter manual trace mode for selected bad traces."""
        selected = self.canvas.get_selected_for_retry()
        self._manual_trace_result_indices = [
            self._trace_to_result[s]
            for s in selected
            if s < len(self._trace_to_result)
        ]
        if not self._manual_trace_result_indices:
            self.canvas.set_mode(ImageCanvas.MODE_VIEW)
            self._finish_measurement()
            return
        self._manual_trace_idx = 0
        self.canvas.clear_manual_trace()
        self._enter_manual_trace_for_current()

    def _enter_manual_trace_for_current(self):
        """Set up manual trace mode for the current root."""
        ri = self._manual_trace_result_indices[self._manual_trace_idx]
        n = len(self._manual_trace_result_indices)
        pi_idx = self._manual_trace_idx + 1

        self._hide_action_buttons()
        self._show_action_frame()

        self.canvas.set_mode(
            ImageCanvas.MODE_MANUAL_TRACE,
            on_done=self._manual_trace_done)
        self.canvas._on_click_callback = self._update_manual_trace_status

        # zoom to the plate containing this root
        points = self.canvas.get_root_points()
        if ri < len(points):
            top = points[ri]
            plates = self.canvas.get_plates()
            for pidx, (r1, r2, c1, c2) in enumerate(plates):
                if r1 <= top[0] <= r2 and c1 <= top[1] <= c2:
                    self.canvas.zoom_to_region(r1, r2, c1, c2)
                    self._set_plate_info(pidx)
                    break

        is_last = self._manual_trace_idx >= n - 1
        btn_text = "Finish" if is_last else "Next Root"
        self.sidebar.btn_done.configure(text=btn_text)
        self.sidebar.btn_done.pack(pady=(5, 0), padx=15, fill="x")
        self.btn_continue_later_mid.pack(pady=(10, 5), padx=15, fill="x")
        self.sidebar.show_review_toggles()

        if self.canvas._manual_trace_submode == 'freehand':
            self.sidebar.set_status(
                f"Manual trace root {pi_idx}/{n} (Freehand):\n"
                f"Drag along the root to draw.\n"
                f"Right-click=undo stroke, Enter=confirm.")
            self.lbl_bottom.configure(
                text=f"Manual {pi_idx}/{n}  |  Drag=draw  |  "
                     f"Right-click=undo stroke  |  Enter=confirm  |  Scroll=zoom")
        else:
            self.sidebar.set_status(
                f"Manual trace root {pi_idx}/{n}:\n"
                f"Click points along the root (top→bottom).\n"
                f"Right-click=undo, Enter=confirm.")
            self.lbl_bottom.configure(
                text=f"Manual {pi_idx}/{n}  |  Click=add point  |  "
                     f"Right-click=undo  |  Enter=confirm  |  Scroll=zoom")

    def _manual_trace_done(self):
        """Called when user presses Enter to confirm a manual trace."""
        points = self.canvas.get_manual_trace_points()
        if len(points) < 2:
            self.sidebar.set_status("Need at least 2 points for a manual trace.")
            return

        path = _interpolate_manual_path(points)
        diffs = np.diff(path, axis=0)
        length_px = float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))
        length_cm = length_px / self._scale_val if self._scale_val else 0

        ri = self._manual_trace_result_indices[self._manual_trace_idx]

        mark_coords = self.canvas._all_marks.get(ri, [])
        segments = []
        if mark_coords and path.size > 0:
            segments = _compute_segments(path, mark_coords, self._scale_val)

        res = dict(
            length_cm=length_cm, length_px=length_px,
            path=path, method='manual', warning=None,
            segments=segments, mark_coords=mark_coords)
        self._results[ri] = res

        self._manual_trace_idx += 1
        self.canvas.confirm_manual_trace()

        if self._manual_trace_idx < len(self._manual_trace_result_indices):
            self._enter_manual_trace_for_current()
        else:
            self.canvas.clear_manual_trace()
            self._rebuild_all_traces()
            self._show_review(skip_delay=True)

    def _update_manual_trace_status(self):
        """Update status after each click/stroke during manual trace."""
        n_pts = len(self.canvas._manual_trace_points)
        n = len(self._manual_trace_result_indices)
        pi_idx = self._manual_trace_idx + 1
        if self.canvas._manual_trace_submode == 'freehand':
            n_strokes = len(self.canvas._manual_trace_strokes)
            self.sidebar.set_status(
                f"Manual trace root {pi_idx}/{n}: "
                f"{n_strokes} stroke(s), {n_pts} points.\n"
                f"Drag more or Enter=confirm, Right-click=undo stroke.")
        else:
            self.sidebar.set_status(
                f"Manual trace root {pi_idx}/{n}: {n_pts} point(s).\n"
                f"Click more points along the root.\n"
                f"Enter=confirm, Right-click=undo.")

    def _save_trace_screenshot(self, silent=False):
        """Save plate image with traced root overlays (no UI elements)."""
        folder = self.folder
        if not folder and self.image_path:
            folder = self.image_path.parent
        if not folder or self.canvas._image_np is None:
            return
        img = self.canvas._image_np.copy()
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img
        scale = self._scale_val
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.8, scale / 300)
        font_thick = max(2, int(scale / 200))
        root_plates = list(self.canvas._root_plates)
        root_groups = list(self.canvas._root_groups)
        trace_to_result = list(self._trace_to_result)

        for ti, (path, shades, mark_indices) in enumerate(self.canvas._traces):
            if len(path) < 2:
                continue
            if mark_indices:
                boundaries = [0] + list(mark_indices) + [len(path) - 1]
                for j in range(len(boundaries) - 1):
                    start = boundaries[j]
                    end = boundaries[j + 1] + 1
                    color_hex = shades[j % len(shades)]
                    bgr = self._hex_to_bgr(color_hex)
                    pts = path[start:end, [1, 0]].astype(np.int32)
                    cv2.polylines(img_bgr, [pts], False, bgr, thickness=3)
            else:
                bgr = self._hex_to_bgr(shades[0])
                pts = path[:, [1, 0]].astype(np.int32)
                cv2.polylines(img_bgr, [pts], False, bgr, thickness=3)
            # Add number + cm label (vertical) next to trace
            ri = trace_to_result[ti] if ti < len(trace_to_result) else ti
            plate = root_plates[ri] if ri < len(root_plates) else 0
            group = root_groups[ri] if ri < len(root_groups) else 0
            count = sum(1 for j in range(ri + 1)
                        if j < len(root_plates) and j < len(root_groups)
                        and root_plates[j] == plate and root_groups[j] == group)
            # number label at top of trace
            top_row, top_col = path[0]
            color_bgr = self._hex_to_bgr(shades[0])
            num_font_scale = font_scale * 1.8
            num_font_thick = max(1, int(font_thick * 1.5))
            self._draw_vertical_label(
                img_bgr, str(count), int(top_col), int(top_row),
                font, num_font_scale, num_font_thick, color_bgr)
            # cm label at bottom of trace (bigger font)
            res = self._results[ri] if ri < len(self._results) else None
            cm_val = res.get('length_cm') if res else None
            if cm_val:
                bot_row, bot_col = path[-1]
                cm_font_scale = font_scale * 1.8
                cm_font_thick = max(1, int(font_thick * 1.5))
                self._draw_vertical_label(
                    img_bgr, f"{cm_val:.2f}cm", int(bot_col), int(bot_row),
                    font, cm_font_scale, cm_font_thick, color_bgr,
                    anchor="top")

        # Draw dead/touching seedling markers
        from canvas import GROUP_MARKER_COLORS
        root_points = list(self.canvas._root_points)
        root_flags = list(self.canvas._root_flags)
        cross_size = max(8, int(scale / 50))
        for i, ((row, col), flag) in enumerate(zip(root_points, root_flags)):
            if flag is None:
                continue
            group = root_groups[i] if i < len(root_groups) else 0
            plate = root_plates[i] if i < len(root_plates) else 0
            count = sum(1 for j in range(i + 1)
                        if j < len(root_plates) and j < len(root_groups)
                        and root_plates[j] == plate and root_groups[j] == group)
            cx, cy = int(col), int(row)
            marker_color = GROUP_MARKER_COLORS[group % len(GROUP_MARKER_COLORS)]
            bgr = self._hex_to_bgr(marker_color)
            label = "DEAD" if flag == 'dead' else "TOUCH"
            # Draw X marker
            cv2.line(img_bgr, (cx - cross_size, cy - cross_size),
                     (cx + cross_size, cy + cross_size), bgr, 2)
            cv2.line(img_bgr, (cx - cross_size, cy + cross_size),
                     (cx + cross_size, cy - cross_size), bgr, 2)
            # Draw label (vertical)
            self._draw_vertical_label(
                img_bgr, f"{count} {label}", cx, cy - cross_size,
                font, font_scale, font_thick, bgr)

        # draw scale bar in bottom-right corner
        bar_cm = 1.0
        bar_px = int(bar_cm * scale)
        h, w = img_bgr.shape[:2]
        # if bar is wider than 1/3 of image, use 0.5 cm
        if bar_px > w // 3:
            bar_cm = 0.5
            bar_px = int(bar_cm * scale)
        margin = int(scale * 0.15)  # ~1.5 mm margin
        bar_h = max(4, int(scale * 0.03))
        x2 = w - margin
        x1 = x2 - bar_px
        y2 = h - margin
        y1 = y2 - bar_h
        # white bar with black outline
        cv2.rectangle(img_bgr, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1),
                       (0, 0, 0), -1)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 255, 255), -1)
        # label
        label = f"{bar_cm:g} cm"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, scale / 800)
        thickness = max(1, int(scale / 400))
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        tx = x1 + (bar_px - tw) // 2
        ty = y1 - int(scale * 0.02) - 2
        # black outline for readability
        cv2.putText(img_bgr, label, (tx, ty), font, font_scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img_bgr, label, (tx, ty), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

        # downscale for reference screenshot (keep under ~2 MB)
        h, w = img_bgr.shape[:2]
        max_dim = 8000
        if max(h, w) > max_dim:
            ratio = max_dim / max(h, w)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)

        exp = getattr(self, '_experiment_name', '')
        out_path = traces_dir(folder, exp) / f'{self.image_path.stem}_traces.jpg'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not silent:
            self.sidebar.set_status(
                self.sidebar.lbl_status.cget("text") +
                f"\nScreenshot: {out_path.name}")

    @staticmethod
    def _hex_to_bgr(hex_color):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)

    @staticmethod
    def _draw_vertical_label(img, text, x, y, font, font_scale, thickness,
                             color_bgr, anchor="bottom"):
        """Draw text rotated 90° CCW (reads bottom-to-top) at (x, y).

        anchor="bottom": label sits above (x, y) — bottom of label at y.
        anchor="top": label sits below (x, y) — top of label at y.
        """
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale,
                                             thickness + 2)
        pad = 4
        buf_h = th + baseline + pad * 2
        buf_w = tw + pad * 2
        buf = np.zeros((buf_h, buf_w, 3), dtype=np.uint8)
        # draw outline then colored text on buffer
        cv2.putText(buf, text, (pad, th + pad), font, font_scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(buf, text, (pad, th + pad), font, font_scale,
                    color_bgr, thickness, cv2.LINE_AA)
        # rotate 90° CCW — text reads bottom-to-top
        rotated = cv2.rotate(buf, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rh, rw = rotated.shape[:2]
        x0 = x - rw // 2
        if anchor == "top":
            y0 = y  # label starts at y, extends downward
        else:
            y0 = y - rh  # label ends at y, extends upward
        # clip to image bounds
        ih, iw = img.shape[:2]
        sx = max(0, -x0)
        sy = max(0, -y0)
        dx = max(0, x0)
        dy = max(0, y0)
        cw = min(rw - sx, iw - dx)
        ch = min(rh - sy, ih - dy)
        if cw <= 0 or ch <= 0:
            return
        roi = rotated[sy:sy + ch, sx:sx + cw]
        mask = np.any(roi > 0, axis=2)
        img[dy:dy + ch, dx:dx + cw][mask] = roi[mask]

    def show_review(self):
        """Re-enter review mode from the sidebar button."""
        if not hasattr(self, '_results') or not self._results:
            # try to rebuild from canvas traces (e.g. after session restore)
            if self.canvas._traces:
                self._rebuild_results_from_traces()
            else:
                self.sidebar.set_status("No traces to review.")
                return
        self.canvas._measurement_done = False
        self.canvas.clear_review()
        self._hide_action_buttons()
        self._show_review()

    def _rebuild_results_from_traces(self):
        """Reconstruct _results and _trace_to_result from canvas state."""
        points = self.canvas.get_root_points()
        flags = self.canvas.get_root_flags()
        self._results = []
        self._trace_to_result = []
        self._scale_val = self._get_scale()
        self._sensitivity = self.sidebar.var_sensitivity.get()
        trace_idx = 0
        for i, flag in enumerate(flags):
            if flag is not None:
                warning = 'dead seedling' if flag == 'dead' else 'roots touching'
                self._results.append(dict(
                    length_cm=None, length_px=None,
                    path=np.empty((0, 2)),
                    method='skip', warning=warning, segments=[]))
            elif trace_idx < len(self.canvas._traces):
                path_data = self.canvas._traces[trace_idx]
                path = np.array(path_data[0]) if not isinstance(
                    path_data[0], np.ndarray) else path_data[0]
                length_px = 0
                if len(path) >= 2:
                    diffs = np.diff(path, axis=0)
                    length_px = float(np.sum(np.sqrt(
                        (diffs ** 2).sum(axis=1))))
                length_cm = length_px / self._scale_val if self._scale_val else 0
                # recompute segments from saved marks
                mark_coords = self.canvas._all_marks.get(i, [])
                segments = []
                if mark_coords and len(path) >= 2:
                    segments = _compute_segments(
                        path, mark_coords, self._scale_val)
                self._results.append(dict(
                    length_cm=length_cm, length_px=length_px,
                    path=path, method='restored', warning=None,
                    segments=segments, mark_coords=mark_coords))
                self._trace_to_result.append(i)
                trace_idx += 1
            else:
                self._results.append(dict(
                    length_cm=0, length_px=0,
                    path=np.empty((0, 2)),
                    method='error', warning='no trace', segments=[]))

    def _finish_measurement(self):
        """Save results and show final summary."""
        _log("_finish_measurement() called")
        self.sidebar.hide_progress()
        self.sidebar.set_step(4)
        plates = self.canvas.get_plates()
        traced = [r for r in self._results
                  if r['method'] not in ('skip', 'error')]
        lengths = [r['length_cm'] for r in traced if r['length_cm']]
        _log(f"  {len(traced)} traced, {len(plates)} plates")
        msg = f"Done! {len(traced)} root(s) traced."
        if lengths:
            msg += f"\nMean: {np.mean(lengths):.2f} cm, "
            msg += f"Range: {min(lengths):.2f}–{max(lengths):.2f} cm"
        self.sidebar.set_status(msg)
        self.lbl_bottom.configure(text="Willian Viana — Dinneny Lab")


        self._save_results(self._results, plates, self._scale_val)
        self._save_trace_screenshot()
        self._save_metadata()

        self.sidebar.set_step(5)  # marks all 4 steps as done (green)
        self.canvas._measurement_done = True
        self.canvas._redraw()
        self.sidebar.btn_select_plates.configure(state="normal")
        self.sidebar.btn_click_roots.configure(state="normal")
        self.sidebar.btn_measure.configure(state="normal")
        self.sidebar.btn_review.configure(state="normal")

        if self.image_path:
            self._processed_images.add(self.image_path)

        self._auto_save()
        self._hide_action_buttons()
        self._show_action_frame()
        self.btn_next_image.pack(pady=(10, 3), padx=15, fill="x")
        self.btn_continue_later.pack(pady=3, padx=15, fill="x")
        self.btn_stop.pack(pady=3, padx=15, fill="x")

    def _save_metadata(self):
        """Save measurement metadata alongside CSV."""
        folder = self.folder
        if not folder and self.image_path:
            folder = self.image_path.parent
        if not folder:
            return
        exp = getattr(self, '_experiment_name', '')
        meta_path = data_dir(folder, exp) / 'metadata.csv'
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        dpi_text = self.sidebar.entry_dpi.get().strip() or "1200"
        experiment = self.sidebar.entry_experiment.get().strip() or ""
        genotypes = self.sidebar.entry_genotypes.get().strip() or ""
        conditions = self.sidebar.entry_condition.get().strip() or ""
        save_metadata(
            meta_path,
            image_name=self.image_path.name if self.image_path else "",
            dpi=dpi_text,
            sensitivity=self._sensitivity,
            experiment=experiment,
            genotypes=genotypes,
            conditions=conditions,
            csv_format='R',
            split_plate=self.sidebar.is_split_plate(),
            num_marks=self._get_num_marks(),
            timestamp=datetime.now().isoformat(timespec='seconds'),
        )

    def _save_results(self, results, plates, scale, silent=False):
        """Save measurement results to CSV. If silent=True, don't update status."""
        from csv_output import get_offsets_from_csv
        folder = self.folder
        if not folder and self.image_path:
            folder = self.image_path.parent
        if not folder:
            if not silent:
                self.sidebar.set_status(
                    self.sidebar.lbl_status.cget("text") +
                    "\nNo folder set — could not save CSV.")
            return

        exp = getattr(self, '_experiment_name', '')
        csv_path = data_dir(folder, exp) / 'raw_data.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # recalculate offsets excluding current image so re-saves are correct
        img_name = self.image_path.name if self.image_path else ''
        if csv_path.exists() and img_name:
            plate_off, root_off = get_offsets_from_csv(csv_path, exclude_image=img_name)
            self._plate_offset = plate_off
            self._root_offset = root_off

        # use stored group indices for each root
        split = self.sidebar.is_split_plate()

        # build plate_labels from sidebar genotype + condition entries
        geno_text = self.sidebar.entry_genotypes.get().strip()
        genotypes = [g.strip() for g in geno_text.split(",")
                     if g.strip()] if geno_text else ["genotype"]
        cond_text = self.sidebar.entry_condition.get().strip()
        conditions = [c.strip() for c in cond_text.split(",")
                      if c.strip()] if cond_text else ["Control"]

        # Build plate_labels, group_to_plate, and point_plates.
        # Keys must be unique per (plate, genotype) combination so that
        # different conditions on different plates don't overwrite each other.
        # Non-split: key = plate index (one genotype per plate).
        # Split:     key = (plate_index, geno_color_index) tuple.
        plate_labels = {}
        group_to_plate = {}
        root_plates_list = list(self.canvas._root_plates)
        root_groups_list = self.canvas.get_root_groups()
        if split:
            geno_a = genotypes[0]
            geno_b = genotypes[1] if len(genotypes) >= 2 else "genotype_B"
            for pi in range(len(plates)):
                cond = conditions[pi] if pi < len(conditions) else (
                    conditions[-1] if conditions else None)
                idx_a = self._genotype_colors.get(geno_a, pi * 2)
                idx_b = self._genotype_colors.get(geno_b, pi * 2 + 1)
                plate_labels[(pi, idx_a)] = (geno_a, cond)
                plate_labels[(pi, idx_b)] = (geno_b, cond)
                group_to_plate[(pi, idx_a)] = pi
                group_to_plate[(pi, idx_b)] = pi
            # build point_plates with matching (plate, group) tuple keys
            point_plates = []
            for i in range(len(root_groups_list)):
                pi = root_plates_list[i] if i < len(root_plates_list) else 0
                gi = root_groups_list[i]
                point_plates.append((pi, gi))
        else:
            point_plates = root_plates_list
            for pi in range(len(plates)):
                geno = genotypes[pi] if pi < len(genotypes) else genotypes[-1]
                cond = conditions[pi] if pi < len(conditions) else (
                    conditions[-1] if conditions else None)
                plate_labels[pi] = (geno, cond)
                group_to_plate[pi] = pi

        try:
            new_plate_offset, new_root_offset = append_results_to_csv(
                results, csv_path, plates, plate_labels,
                plate_offset=self._plate_offset,
                root_offset=self._root_offset,
                point_plates=point_plates,
                num_marks=self._get_num_marks(),
                split_plate=self.sidebar.is_split_plate(),
                image_name=img_name,
                group_to_plate=group_to_plate)
            self._plate_offset = new_plate_offset
            self._root_offset = new_root_offset
            if not silent:
                self.sidebar.set_status(
                    self.sidebar.lbl_status.cget("text") +
                    f"\nSaved to {csv_path}")
        except Exception as e:
            if not silent:
                self.sidebar.set_status(
                    self.sidebar.lbl_status.cget("text") +
                    f"\nCSV save error: {e}")
            import traceback
            traceback.print_exc()

    def _run_plot(self):
        """Generate tidy_data.csv and plot from raw_data.csv."""
        from csv_output import generate_tidy
        folder = self.folder
        if not folder and self.image_path:
            folder = self.image_path.parent
        if not folder:
            return
        exp = getattr(self, '_experiment_name', '')
        raw_path = data_dir(folder, exp) / 'raw_data.csv'
        if not raw_path.exists():
            return
        try:
            self.sidebar.set_status(
                self.sidebar.lbl_status.cget("text") +
                "\nGenerating tidy data and plot...")
            self.update_idletasks()
            tidy_path = data_dir(folder, exp) / 'tidy_data.csv'
            generate_tidy(raw_path, tidy_path, csv_format='R')
            # plot from raw data (always tall/R format)
            plot_results(raw_path,
                         value_col='Length_cm',
                         ylabel='Primary root length (cm)',
                         csv_format='R',
                         genotype_colors=getattr(self, '_genotype_colors', None))
            self.sidebar.set_status(
                self.sidebar.lbl_status.cget("text") +
                f"\nSaved tidy_data.csv and plot")
        except Exception as e:
            self.sidebar.set_status(
                self.sidebar.lbl_status.cget("text") +
                f"\nPlot error: {e}")
            import traceback
            traceback.print_exc()
