#!/usr/bin/env python3
"""Root Measure GUI — CustomTkinter app shell."""

import sys
from pathlib import Path

import customtkinter as ctk
import numpy as np
import cv2
import tifffile

# allow importing from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from config import DISPLAY_DOWNSAMPLE, SCALE_PX_PER_CM
from plate_detection import _to_uint8
from image_processing import preprocess
from root_tracing import find_root_tip, trace_root
from utils import _compute_segments
from csv_output import append_results_to_csv

from canvas import ImageCanvas
from sidebar import Sidebar

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def _detect_dpi(image_path):
    """Try to read DPI from image metadata. Returns int DPI or None."""
    ext = Path(image_path).suffix.lower()
    try:
        if ext in ('.tif', '.tiff'):
            with tifffile.TiffFile(str(image_path)) as tif:
                page = tif.pages[0]
                if 282 in page.tags and 283 in page.tags:
                    x_res = page.tags[282].value
                    if isinstance(x_res, tuple):
                        dpi = x_res[0] / x_res[1]
                    else:
                        dpi = float(x_res)
                    unit = page.tags.get(296)
                    if unit and unit.value == 3:
                        dpi = dpi * 2.54
                    dpi = int(round(dpi))
                    if dpi > 0:
                        return dpi
    except Exception:
        pass
    return None


class RootMeasureApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("Root Measure")
        self.geometry("1400x900")
        self.minsize(900, 600)

        # state
        self.image = None          # full grayscale numpy array
        self.image_path = None
        self.folder = None
        self.images = []           # list of Path objects

        # layout: sidebar + canvas
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = Sidebar(self, app=self)
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=0, pady=0)

        self.canvas = ImageCanvas(self)
        self.canvas.grid(row=0, column=1, sticky="nsew", padx=(2, 0), pady=0)

        # status bar at bottom
        self.status_bar = ctk.CTkFrame(self, height=30, fg_color="gray15")
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.status_bar.grid_propagate(False)
        self.lbl_bottom = ctk.CTkLabel(
            self.status_bar, text="Root Measure — Dinneny Lab",
            font=ctk.CTkFont(size=10), text_color="gray50")
        self.lbl_bottom.pack(side="left", padx=10)

        # global keyboard handler — works regardless of which widget has focus
        self.bind_all("<Key>", self._on_global_key)

    def _on_global_key(self, event):
        """Route keyboard events to canvas, skip if typing in an Entry."""
        # don't intercept keys when typing in entry fields
        widget_class = event.widget.winfo_class()
        if widget_class in ('Entry', 'TEntry', 'Text'):
            return
        self.canvas.handle_key(event)

    # --- Actions ---

    def load_folder(self):
        from tkinter import filedialog
        from utils import list_images_in_folder

        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return
        self.folder = Path(folder)
        self.images = list_images_in_folder(self.folder)

        if not self.images:
            self.sidebar.set_status(f"No images found in {self.folder.name}")
            return

        self.sidebar.set_status(f"{len(self.images)} image(s) in {self.folder.name}")
        self.sidebar.show_image_list(self.images)

    def load_image(self, path):
        """Load and display a single image."""
        self.image_path = path
        try:
            ext = path.suffix.lower()
            if ext in ('.tif', '.tiff'):
                img = tifffile.imread(str(path))
            else:
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    self.sidebar.set_status(f"Could not read {path.name}")
                    return

            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            self.image = img
            display = _to_uint8(img)
            self.canvas.set_image(display)

            self.sidebar.lbl_image_name.configure(text=path.name)
            self.sidebar.btn_select_plates.configure(state="normal")

            # auto-detect DPI and fill the field
            detected = _detect_dpi(path)
            dpi = detected or 1200
            self.sidebar.entry_dpi.delete(0, "end")
            self.sidebar.entry_dpi.insert(0, str(dpi))
            dpi_src = "detected" if detected else "default"

            self.sidebar.set_status(
                f"Loaded: {img.shape[1]}x{img.shape[0]}, {img.dtype}\n"
                f"DPI: {dpi} ({dpi_src})")
        except Exception as e:
            self.sidebar.set_status(f"Error: {e}")

    def _get_num_marks(self):
        """Get number of marks per root from sidebar (0 if multi-measurement off)."""
        if not self.sidebar.var_multi.get():
            return 0
        text = self.sidebar.entry_segments.get().strip()
        try:
            segments = int(text)
            if segments >= 2:
                return segments - 1  # N segments = N-1 marks
        except (ValueError, TypeError):
            pass
        return 1  # default: 2 segments = 1 mark

    def _get_scale(self):
        """Get scale (px/cm) from DPI entry or auto-detect."""
        dpi_text = self.sidebar.entry_dpi.get().strip()
        if dpi_text:
            try:
                dpi = int(dpi_text)
                if dpi > 0:
                    return dpi / 2.54
            except ValueError:
                pass
        # try auto-detect from image metadata
        if self.image_path:
            detected = _detect_dpi(self.image_path)
            if detected:
                self.sidebar.entry_dpi.delete(0, "end")
                self.sidebar.entry_dpi.insert(0, str(detected))
                return detected / 2.54
        # default 1200 DPI
        self.sidebar.entry_dpi.delete(0, "end")
        self.sidebar.entry_dpi.insert(0, "1200")
        return SCALE_PX_PER_CM

    def select_plates(self):
        """Enter plate selection mode on canvas."""
        self.canvas.clear_plates()
        self.canvas.clear_roots()
        self.canvas.clear_traces()
        self.canvas._plates_count_at_enter = 0
        self.canvas.set_mode(
            ImageCanvas.MODE_SELECT_PLATES,
            on_done=self._plates_done)
        self.sidebar.set_status(
            "Draw rectangle around plate, then Enter.\n"
            "Draw another + Enter, or Enter again to finish.")
        self.lbl_bottom.configure(
            text="Drag=draw plate  |  Right-click=undo  |  Enter=confirm  |  Enter again=done  |  Scroll=zoom")
        # disable downstream buttons while selecting
        self.sidebar.btn_click_roots.configure(state="disabled")
        self.sidebar.btn_measure.configure(state="disabled")

    def _plates_done(self):
        """Called when user presses Enter after selecting plates."""
        plates = self.canvas.get_plates()
        if not plates:
            self.sidebar.set_status("No plates selected. Draw at least one rectangle.")
            return
        self.canvas.set_mode(ImageCanvas.MODE_VIEW)
        self.sidebar.set_status(f"{len(plates)} plate(s) selected.")
        self.lbl_bottom.configure(text="Root Measure — Dinneny Lab")
        self.sidebar.btn_click_roots.configure(state="normal")

    def click_roots(self):
        """Enter root clicking mode on canvas."""
        plates = self.canvas.get_plates()
        if not plates:
            self.sidebar.set_status("Select plates first.")
            return
        self.canvas.clear_roots()
        self.canvas.clear_marks()
        self.canvas.clear_traces()
        self.canvas._all_marks = {}
        self._current_plate_idx = 0
        self.canvas.set_mode(
            ImageCanvas.MODE_CLICK_ROOTS,
            on_done=self._plate_roots_done)
        # zoom into first plate
        r1, r2, c1, c2 = plates[0]
        self.canvas.zoom_to_region(r1, r2, c1, c2)
        self.sidebar.set_status(
            f"Plate 1/{len(plates)} — Click root tops.\n"
            "D+Click=dead, T+Click=touching. Enter=next.")
        self.lbl_bottom.configure(
            text="Click=root top  |  D+Click=dead  |  T+Click=touching  |  Right-click=undo  |  Enter=next  |  Scroll=zoom")
        self.sidebar.btn_measure.configure(state="disabled")

    def _plate_roots_done(self):
        """Called when user presses Enter after clicking roots on a plate."""
        num_marks = self._get_num_marks()
        if num_marks > 0:
            # enter marks phase for this plate before advancing
            self._start_marks_phase()
            return
        self._advance_to_next_plate()

    def _advance_to_next_plate(self):
        """Advance to next plate for root clicking, or finish."""
        plates = self.canvas.get_plates()
        self._current_plate_idx += 1
        if self._current_plate_idx < len(plates):
            # advance to next plate
            r1, r2, c1, c2 = plates[self._current_plate_idx]
            self.canvas.zoom_to_region(r1, r2, c1, c2)
            pi = self._current_plate_idx + 1
            self.canvas.set_mode(
                ImageCanvas.MODE_CLICK_ROOTS,
                on_done=self._plate_roots_done)
            self.sidebar.set_status(
                f"Plate {pi}/{len(plates)} — Click root tops.\n"
                "D+Click=dead, T+Click=touching. Enter=next.")
            self.lbl_bottom.configure(
                text="Click=root top  |  D+Click=dead  |  T+Click=touching  |  Right-click=undo  |  Enter=next  |  Scroll=zoom")
            return
        # all plates done
        points = self.canvas.get_root_points()
        if not points:
            self.sidebar.set_status("No roots clicked. Click at least one root top.")
            self._current_plate_idx = 0
            return
        self.canvas.set_mode(ImageCanvas.MODE_VIEW)
        # zoom back to full image
        self.canvas._fit_image()
        self.canvas._redraw()
        n_normal = sum(1 for f in self.canvas.get_root_flags() if f is None)
        n_flagged = len(points) - n_normal
        msg = f"{len(points)} root(s) marked ({n_normal} to trace"
        if n_flagged:
            msg += f", {n_flagged} flagged"
        msg += ")."
        self.sidebar.set_status(msg)
        self.lbl_bottom.configure(text="Root Measure — Dinneny Lab")
        self.sidebar.btn_measure.configure(state="normal")

    def _start_marks_phase(self):
        """Enter mark clicking mode for normal roots on the current plate."""
        plates = self.canvas.get_plates()
        pi = self._current_plate_idx
        r1, r2, c1, c2 = plates[pi]
        # find normal (non-flagged) roots on this plate
        points = self.canvas.get_root_points()
        flags = self.canvas.get_root_flags()
        self._marks_plate_roots = []
        for i, ((row, col), flag) in enumerate(zip(points, flags)):
            if flag is None and r1 <= row <= r2 and c1 <= col <= c2:
                self._marks_plate_roots.append(i)
        if not self._marks_plate_roots:
            self._advance_to_next_plate()
            return
        num_marks = self._get_num_marks()
        self.canvas._marks_expected = len(self._marks_plate_roots) * num_marks
        self.canvas.clear_marks()
        self.canvas._on_click_callback = self._update_marks_status
        self.canvas.set_mode(
            ImageCanvas.MODE_CLICK_MARKS,
            on_done=self._plate_marks_done)
        self.canvas.zoom_to_region(r1, r2, c1, c2)
        self._update_marks_status()
        self.lbl_bottom.configure(
            text="Click=place mark  |  Right-click=undo  |  Enter=done  |  Scroll=zoom")

    def _update_marks_status(self):
        """Update status bar with marks progress."""
        pi = self._current_plate_idx
        num_marks = self._get_num_marks()
        n_roots = len(self._marks_plate_roots)
        expected = self.canvas._marks_expected
        placed = len(self.canvas.get_mark_points())
        if placed >= expected:
            self.sidebar.set_status(
                f"Plate {pi + 1} — All {expected} mark(s) placed.\n"
                "Press Enter to continue.")
        else:
            # show which root and which mark we're on
            current_root_idx = placed // num_marks + 1
            current_mark_in_root = placed % num_marks + 1
            self.sidebar.set_status(
                f"Plate {pi + 1} — Marks: {placed}/{expected}.\n"
                f"Root {current_root_idx}/{n_roots}, "
                f"mark {current_mark_in_root}/{num_marks}.\n"
                "Right-click=undo. Enter when all placed.")

    def _plate_marks_done(self):
        """Called when user presses Enter after clicking marks for a plate."""
        all_marks = self.canvas.get_mark_points()
        if len(all_marks) < self.canvas._marks_expected:
            self.sidebar.set_status(
                f"Need {self.canvas._marks_expected} mark(s), "
                f"only {len(all_marks)} placed. Click more marks.")
            return
        # assign num_marks marks to each normal root, in click order
        num_marks = self._get_num_marks()
        for j, ri in enumerate(self._marks_plate_roots):
            start = j * num_marks
            self.canvas._all_marks[ri] = all_marks[start:start + num_marks]
        self._advance_to_next_plate()

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
                color = "#00ff88" if res['warning'] is None else "#ffaa00"
                self.canvas.add_trace(res['path'], color)
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
                color = "#00ff88" if res['warning'] is None else "#ffaa00"
                self.canvas.add_trace(res['path'], color)
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
        if not self.folder:
            return

        csv_path = self.folder / 'output' / 'data.csv'
        csv_path.parent.mkdir(exist_ok=True)

        experiment = self.sidebar.entry_experiment.get().strip() or "experiment"

        # assign roots to plates based on position
        point_plates = []
        for row, col in self.canvas.get_root_points():
            assigned = 0
            for pi, (r1, r2, c1, c2) in enumerate(plates):
                if r1 <= row <= r2 and c1 <= col <= c2:
                    assigned = pi
                    break
            point_plates.append(assigned)

        # simple labels: experiment name for each plate
        plate_labels = [(experiment, None)] * len(plates)

        append_results_to_csv(
            results, csv_path, plates, plate_labels,
            plate_offset=0, root_offset=0,
            point_plates=point_plates,
            num_marks=self._get_num_marks(),
            split_plate=self.sidebar.var_split.get())

        self.sidebar.set_status(
            self.sidebar.lbl_status.cget("text") +
            f"\nSaved to {csv_path.name}")


def main():
    app = RootMeasureApp()
    app.mainloop()


if __name__ == '__main__':
    main()
