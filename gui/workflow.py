"""Measurement workflow — tracing, review, retry, and CSV saving."""

import numpy as np

from image_processing import preprocess
from root_tracing import find_root_tip, trace_root
from utils import _compute_segments, _find_nearest_path_index
from csv_output import append_results_to_csv

from canvas import ImageCanvas

# genotype color shades: [dark, light] for alternating segments
GROUP_COLORS = [
    ["#e63333", "#ff8080"],  # group 0: dark red, light red
    ["#3333e6", "#8080ff"],  # group 1: dark blue, light blue
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
        split = self.sidebar.var_split.get()
        if split and root_idx < len(groups):
            color_idx = groups[root_idx] % 2
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
        points = self.canvas.get_root_points()
        flags = self.canvas.get_root_flags()
        plates = self.canvas.get_plates()

        if not points or self.image is None:
            self.sidebar.set_status("Nothing to measure.")
            return

        self._scale_val = self._get_scale()
        self._sensitivity = self.sidebar.var_sensitivity.get()

        self.sidebar.set_status("Preprocessing...")
        self.sidebar.btn_measure.configure(state="disabled")
        self.sidebar.btn_select_plates.configure(state="disabled")
        self.sidebar.btn_click_roots.configure(state="disabled")
        self.update_idletasks()

        self._binary = preprocess(self.image, scale=self._scale_val,
                                  sensitivity=self._sensitivity)

        self.canvas.clear_traces()
        self._results = []
        # map trace index -> result index (skip flagged roots with no trace)
        self._trace_to_result = []

        num_marks = self._get_num_marks()
        if num_marks > 0:
            print(f"[marks] num_marks={num_marks}, "
                  f"all_marks keys={sorted(self.canvas._all_marks.keys())}")

        for i, (top, flag) in enumerate(zip(points, flags)):
            self.sidebar.set_status(f"Tracing root {i + 1}/{len(points)}...")
            self.update_idletasks()

            if flag is not None:
                warning = 'dead seedling' if flag == 'dead' else 'roots touching'
                res = dict(length_cm=None, length_px=None,
                           path=np.empty((0, 2)),
                           method='skip', warning=warning, segments=[])
                self._results.append(res)
                continue

            tip = find_root_tip(self._binary, top, scale=self._scale_val)
            if tip is None:
                res = dict(length_cm=0, length_px=0,
                           path=np.empty((0, 2)),
                           method='error', warning='Could not find root tip',
                           segments=[])
                self._results.append(res)
                continue

            res = trace_root(self._binary, top, tip, self._scale_val)
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

    def _show_review(self):
        """Show traced results and let user click bad traces to retry."""
        self.canvas.set_mode(
            ImageCanvas.MODE_REVIEW,
            on_done=self._review_done)
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
        msg += "\nEnter = accept / retry selected."
        self.sidebar.set_status(msg)
        self.lbl_bottom.configure(
            text="Click trace=select for retry (yellow)  |  Enter=accept / retry selected  |  Scroll=zoom")

    def _review_done(self):
        """Called when user presses Enter in review mode."""
        selected = self.canvas.get_selected_for_retry()
        if not selected:
            # accept all — save and finish
            self.canvas.set_mode(ImageCanvas.MODE_VIEW)
            self._finish_measurement()
            return
        # map trace indices to result indices
        self._retry_result_indices = [self._trace_to_result[s]
                                       for s in selected
                                       if s < len(self._trace_to_result)]
        if not self._retry_result_indices:
            self.canvas.set_mode(ImageCanvas.MODE_VIEW)
            self._finish_measurement()
            return
        # enter reclick mode
        self._start_reclick()

    def _start_reclick(self):
        """Enter reclick mode for bad traces."""
        self.canvas.clear_reclick()
        self._reclick_idx = 0  # which retry root we're on
        n = len(self._retry_result_indices)
        self.canvas.set_mode(
            ImageCanvas.MODE_RECLICK,
            on_done=self._reclick_enter)
        # zoom to first bad root
        ri = self._retry_result_indices[0]
        top = self.canvas.get_root_points()[ri]
        plates = self.canvas.get_plates()
        # find which plate this root is in
        for r1, r2, c1, c2 in plates:
            if r1 <= top[0] <= r2 and c1 <= top[1] <= c2:
                self.canvas.zoom_to_region(r1, r2, c1, c2)
                break
        self.sidebar.set_status(
            f"Re-click root 1/{n}: click TOP then BOTTOM.\n"
            "Right-click=undo. Enter=confirm pair.")
        self.lbl_bottom.configure(
            text="Click TOP then BOTTOM  |  Right-click=undo  |  Enter=confirm pair  |  Scroll=zoom")

    def _reclick_enter(self):
        """Called when user presses Enter during reclick."""
        pts = self.canvas._reclick_points
        # need exactly 2 points (top + bottom) for current root
        if len(pts) < (self._reclick_idx + 1) * 2:
            self.sidebar.set_status("Click both TOP and BOTTOM before pressing Enter.")
            return
        self._reclick_idx += 1
        n = len(self._retry_result_indices)
        if self._reclick_idx < n:
            # advance to next retry root
            ri = self._retry_result_indices[self._reclick_idx]
            top = self.canvas.get_root_points()[ri]
            plates = self.canvas.get_plates()
            for r1, r2, c1, c2 in plates:
                if r1 <= top[0] <= r2 and c1 <= top[1] <= c2:
                    self.canvas.zoom_to_region(r1, r2, c1, c2)
                    break
            pi = self._reclick_idx + 1
            self.sidebar.set_status(
                f"Re-click root {pi}/{n}: click TOP then BOTTOM.\n"
                "Right-click=undo. Enter=confirm pair.")
            return
        # all re-clicks done — re-trace
        self._do_retrace()

    def _do_retrace(self):
        """Re-trace roots with manually clicked top/bottom."""
        pairs = self.canvas.get_reclick_pairs()
        self.canvas.set_mode(ImageCanvas.MODE_VIEW)

        for j, ri in enumerate(self._retry_result_indices):
            if j >= len(pairs):
                break
            top_manual, bot_manual = pairs[j]
            self.sidebar.set_status(f"Re-tracing root {j + 1}/{len(pairs)}...")
            self.update_idletasks()
            res = trace_root(self._binary, top_manual, bot_manual, self._scale_val)
            # recompute segments if marks exist for this root
            mark_coords = self.canvas._all_marks.get(ri, [])
            if mark_coords and res['path'].size > 0:
                res['segments'] = _compute_segments(
                    res['path'], mark_coords, self._scale_val)
                res['mark_coords'] = mark_coords
            else:
                res['segments'] = []
            self._results[ri] = res

        # rebuild all traces
        self.canvas.clear_traces()
        self._trace_to_result.clear()
        for i, res in enumerate(self._results):
            if res['path'].size > 0 and res['method'] not in ('skip', 'error'):
                self._add_root_trace(i, res)
                self._trace_to_result.append(i)

        # back to review
        self._show_review()

    def _finish_measurement(self):
        """Save results and show final summary."""
        plates = self.canvas.get_plates()
        traced = [r for r in self._results
                  if r['method'] not in ('skip', 'error')]
        lengths = [r['length_cm'] for r in traced if r['length_cm']]
        msg = f"Done! {len(traced)} root(s) traced."
        if lengths:
            msg += f"\nMean: {np.mean(lengths):.2f} cm, "
            msg += f"Range: {min(lengths):.2f}–{max(lengths):.2f} cm"
        self.sidebar.set_status(msg)
        self.lbl_bottom.configure(
            text=f"Traced {len(traced)}/{len(self._results)} roots")

        self._save_results(self._results, plates, self._scale_val)
        self.sidebar.btn_measure.configure(state="normal")
        self.sidebar.btn_select_plates.configure(state="normal")
        self.sidebar.btn_click_roots.configure(state="normal")

    def _save_results(self, results, plates, scale):
        """Save measurement results to CSV."""
        folder = self.folder
        if not folder and self.image_path:
            folder = self.image_path.parent
        if not folder:
            self.sidebar.set_status(
                self.sidebar.lbl_status.cget("text") +
                "\nNo folder set — could not save CSV.")
            return

        csv_path = folder / 'output' / 'data.csv'
        csv_path.parent.mkdir(exist_ok=True)

        experiment = self.sidebar.entry_experiment.get().strip() or "experiment"

        # use stored group indices for each root
        split = self.sidebar.var_split.get()
        point_plates = self.canvas.get_root_groups()

        # plate_labels: one entry per group
        if split:
            # 2 groups per plate: (experiment, None) for each group
            plate_labels = [(experiment, None)] * (len(plates) * 2)
        else:
            plate_labels = [(experiment, None)] * len(plates)

        try:
            new_plate_offset, new_root_offset = append_results_to_csv(
                results, csv_path, plates, plate_labels,
                plate_offset=self._plate_offset,
                root_offset=self._root_offset,
                point_plates=point_plates,
                num_marks=self._get_num_marks(),
                split_plate=self.sidebar.var_split.get())
            self._plate_offset = new_plate_offset
            self._root_offset = new_root_offset
            self.sidebar.set_status(
                self.sidebar.lbl_status.cget("text") +
                f"\nSaved to {csv_path}")
        except Exception as e:
            self.sidebar.set_status(
                self.sidebar.lbl_status.cget("text") +
                f"\nCSV save error: {e}")
