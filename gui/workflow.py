"""Measurement workflow — tracing, review, retry, and CSV saving."""

import cv2
import numpy as np
from datetime import datetime

from image_processing import preprocess
from root_tracing import find_root_tip, trace_root
from utils import _compute_segments, _find_nearest_path_index
from csv_output import append_results_to_csv, save_metadata
from plotting import plot_results

from canvas import ImageCanvas
from session import data_dir, traces_dir

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

        self.sidebar.hide_action_buttons()
        self.sidebar.set_status("Preprocessing...")
        self.sidebar.btn_measure.configure(state="disabled")
        self.sidebar.btn_select_plates.configure(state="disabled")
        self.sidebar.btn_click_roots.configure(state="disabled")
        self.sidebar.btn_review.configure(state="disabled")
        self.sidebar.set_step(3)
        self.update()

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
        self.sidebar.hide_progress()
        self.sidebar.set_step(4)
        # Block Enter for 500ms to prevent leftover keypresses from skipping review
        self._review_ready = False
        self.canvas.set_mode(
            ImageCanvas.MODE_REVIEW,
            on_done=self._review_done)
        self.after(500, self._enable_review)
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
        self.sidebar.btn_continue_later_mid.pack(pady=(10, 5), padx=15, fill="x")
        self.lbl_bottom.configure(
            text="Click trace=select for retry (orange)  |  Enter=accept / retry selected  |  Scroll=zoom")

    def _enable_review(self):
        self._review_ready = True

    def _review_done(self):
        """Called when user presses Enter in review mode."""
        if not getattr(self, '_review_ready', True):
            return  # ignore leftover Enter from previous mode
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
        self.sidebar.hide_action_buttons()
        self.canvas.clear_reclick()
        self._reclick_idx = 0  # which retry root we're on
        num_marks = self._get_num_marks()
        self._reclick_clicks_per_root = 2 + num_marks
        self.canvas._reclick_clicks_per_root = self._reclick_clicks_per_root
        n = len(self._retry_result_indices)
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
        self.sidebar.btn_continue_later_mid.pack(pady=3, padx=15, fill="x")

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
            f"Re-click root {pi}/{n}: click {click_desc}.\n"
            "Right-click=undo. Enter=confirm.")
        self.lbl_bottom.configure(
            text=f"{bottom_text}  |  Right-click=undo  |  Enter=confirm  |  Scroll=zoom")

    def _reclick_enter(self):
        """Called when user presses Enter during reclick."""
        pts = self.canvas._reclick_points
        cpr = self._reclick_clicks_per_root
        n = len(self._retry_result_indices)
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
        self._do_retrace()

    def _do_retrace(self):
        """Re-trace roots with manually clicked points."""
        self.sidebar.hide_action_buttons()
        # ensure binary mask exists (may be missing after session restore)
        if not hasattr(self, '_binary') or self._binary is None:
            self.sidebar.set_status("Preprocessing...")
            self.update()
            self._binary = preprocess(self.image, scale=self._scale_val,
                                      sensitivity=self._sensitivity)
        cpr = self._reclick_clicks_per_root
        groups = self.canvas.get_reclick_groups(cpr)
        self.canvas.set_mode(ImageCanvas.MODE_VIEW)

        total = min(len(self._retry_result_indices), len(groups))
        self.sidebar.show_progress(total)
        self.update()

        for j, ri in enumerate(self._retry_result_indices):
            if j >= len(groups):
                break
            clicks = groups[j]
            top_manual = clicks[0]
            bot_manual = clicks[-1]
            self.sidebar.set_status(f"Re-tracing root {j + 1}/{total}...")
            self.sidebar.update_progress(j + 1)
            self.update()
            res = trace_root(self._binary, top_manual, bot_manual, self._scale_val)
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

        # rebuild all traces
        self.canvas.clear_traces()
        self._trace_to_result.clear()
        for i, res in enumerate(self._results):
            if res['path'].size > 0 and res['method'] not in ('skip', 'error'):
                self._add_root_trace(i, res)
                self._trace_to_result.append(i)

        # back to review
        self._show_review()

    def _save_trace_screenshot(self):
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
        for path, shades, mark_indices in self.canvas._traces:
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
        # draw scale bar in bottom-right corner
        scale = self._scale_val  # px/cm
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

        exp = getattr(self, '_experiment_name', '')
        out_path = traces_dir(folder, exp) / f'{self.image_path.stem}_traces.png'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img_bgr)
        self.sidebar.set_status(
            self.sidebar.lbl_status.cget("text") +
            f"\nScreenshot: {out_path.name}")

    @staticmethod
    def _hex_to_bgr(hex_color):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)

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
        self.sidebar.hide_action_buttons()
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
                self._results.append(dict(
                    length_cm=length_cm, length_px=length_px,
                    path=path, method='restored', warning=None, segments=[]))
                self._trace_to_result.append(i)
                trace_idx += 1
            else:
                self._results.append(dict(
                    length_cm=0, length_px=0,
                    path=np.empty((0, 2)),
                    method='error', warning='no trace', segments=[]))

    def _finish_measurement(self):
        """Save results and show final summary."""
        self.sidebar.set_step(4)
        plates = self.canvas.get_plates()
        traced = [r for r in self._results
                  if r['method'] not in ('skip', 'error')]
        lengths = [r['length_cm'] for r in traced if r['length_cm']]
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
        self.sidebar.btn_select_plates.configure(state="normal")
        self.sidebar.btn_click_roots.configure(state="normal")
        self.sidebar.btn_measure.configure(state="normal")
        self.sidebar.btn_review.configure(state="normal")

        if self.image_path:
            self._processed_images.add(self.image_path)

        self._auto_save()
        self.sidebar.hide_action_buttons()
        self.sidebar.btn_next_image.pack(pady=(10, 3), padx=15, fill="x")
        self.sidebar.btn_continue_later.pack(pady=3, padx=15, fill="x")
        self.sidebar.btn_stop.pack(pady=3, padx=15, fill="x")

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
            csv_format=self.sidebar.var_csv_format.get(),
            split_plate=self.sidebar.var_split.get(),
            num_marks=self._get_num_marks(),
            timestamp=datetime.now().isoformat(timespec='seconds'),
        )

    def _save_results(self, results, plates, scale):
        """Save measurement results to CSV."""
        from csv_output import get_offsets_from_csv
        folder = self.folder
        if not folder and self.image_path:
            folder = self.image_path.parent
        if not folder:
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
        split = self.sidebar.var_split.get()
        point_plates = self.canvas.get_root_groups()

        # build plate_labels from sidebar genotype + condition entries
        geno_text = self.sidebar.entry_genotypes.get().strip()
        genotypes = [g.strip() for g in geno_text.split(",")
                     if g.strip()] if geno_text else ["genotype"]
        cond_text = self.sidebar.entry_condition.get().strip()
        conditions = [c.strip() for c in cond_text.split(",")
                      if c.strip()] if cond_text else []

        if split:
            geno_a = genotypes[0]
            geno_b = genotypes[1] if len(genotypes) >= 2 else "genotype_B"
            plate_labels = []
            for pi in range(len(plates)):
                cond = conditions[pi] if pi < len(conditions) else (
                    conditions[0] if conditions else None)
                plate_labels.append((geno_a, cond))
                plate_labels.append((geno_b, cond))
        else:
            plate_labels = []
            for pi in range(len(plates)):
                geno = genotypes[pi] if pi < len(genotypes) else genotypes[-1]
                cond = conditions[pi] if pi < len(conditions) else (
                    conditions[0] if conditions else None)
                plate_labels.append((geno, cond))

        try:
            new_plate_offset, new_root_offset = append_results_to_csv(
                results, csv_path, plates, plate_labels,
                plate_offset=self._plate_offset,
                root_offset=self._root_offset,
                point_plates=point_plates,
                num_marks=self._get_num_marks(),
                split_plate=self.sidebar.var_split.get(),
                image_name=img_name)
            self._plate_offset = new_plate_offset
            self._root_offset = new_root_offset
            self.sidebar.set_status(
                self.sidebar.lbl_status.cget("text") +
                f"\nSaved to {csv_path}")
        except Exception as e:
            self.sidebar.set_status(
                self.sidebar.lbl_status.cget("text") +
                f"\nCSV save error: {e}")

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
            csv_format = self.sidebar.var_csv_format.get()
            tidy_path = data_dir(folder, exp) / 'tidy_data.csv'
            generate_tidy(raw_path, tidy_path, csv_format=csv_format)
            # plot from raw data (always tall/R format)
            plot_results(raw_path,
                         value_col='Length_cm',
                         ylabel='Primary root length (cm)',
                         csv_format='R')
            self.sidebar.set_status(
                self.sidebar.lbl_status.cget("text") +
                f"\nSaved tidy_data.csv and plot")
        except Exception as e:
            self.sidebar.set_status(
                self.sidebar.lbl_status.cget("text") +
                f"\nPlot error: {e}")
            import traceback
            traceback.print_exc()
