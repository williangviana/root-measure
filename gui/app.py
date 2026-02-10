#!/usr/bin/env python3
"""Root Measure GUI — CustomTkinter app shell."""

import sys
from pathlib import Path

import customtkinter as ctk
import cv2
import numpy as np
import tifffile

# allow importing from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from config import SCALE_PX_PER_CM
from plate_detection import _to_uint8

from canvas import ImageCanvas, GROUP_MARKER_COLORS
from sidebar import Sidebar
from workflow import MeasurementMixin
from session import save_session, load_session, restore_settings, \
    save_last_folder, get_last_folder, save_experiment_name, \
    get_experiment_name, save_persistent_settings, get_persistent_settings, \
    get_recent_folders, get_session_summaries, \
    session_dir, data_dir

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
        else:
            from PIL import Image as PILImage
            with PILImage.open(str(image_path)) as img:
                info_dpi = img.info.get('dpi')
                if info_dpi:
                    dpi = int(round(info_dpi[0]))
                    if dpi > 0:
                        return dpi
    except Exception:
        pass
    return None


class RootMeasureApp(MeasurementMixin, ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.title("Root Measuring Tool v1.0")
        self.geometry("1400x900")
        self.minsize(900, 600)

        # state
        self.image = None          # full grayscale numpy array
        self.image_path = None
        self.folder = None
        self.images = []           # list of Path objects
        self._plate_offset = 0     # accumulated plate offset across images
        self._root_offset = 0      # accumulated root offset across images
        self._processed_images = set()  # image paths already measured
        self._experiment_name = ''  # current experiment (scopes output dirs)
        self._image_canvas_data = {}  # per-image canvas snapshots {name: dict}
        self._genotype_colors = {}   # genotype name → color index (experiment-wide)

        # layout: sidebar + canvas
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left column: sidebar + action buttons at bottom
        self._left_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._left_frame.grid(row=0, column=0, sticky="nsw", padx=0, pady=0)
        self._left_frame.grid_rowconfigure(0, weight=1)

        self.sidebar = Sidebar(self._left_frame, app=self)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Action buttons frame - fixed at bottom of left column (hidden initially)
        # Match sidebar background color (CTkScrollableFrame default)
        self._action_frame = ctk.CTkFrame(self._left_frame, fg_color=("gray86", "gray17"))
        self._action_frame.grid(row=1, column=0, sticky="ew", padx=0, pady=0)
        self._action_frame.grid_remove()

        self.btn_next_image = ctk.CTkButton(
            self._action_frame, text="Next Image", fg_color="#2b5797",
            command=self.next_image)
        self.btn_continue_later = ctk.CTkButton(
            self._action_frame, text="Continue Later", fg_color="#2b5797",
            command=lambda: self.continue_later())
        self.btn_stop = ctk.CTkButton(
            self._action_frame, text="Finish & Plot", fg_color="#2b5797",
            command=lambda: self.finish_and_plot())
        self.btn_continue_later_mid = ctk.CTkButton(
            self._action_frame, text="Continue Later", fg_color="#2b5797",
            command=lambda: self.continue_later())

        self.canvas = ImageCanvas(self)
        self.canvas.grid(row=0, column=1, sticky="nsew", padx=(2, 0), pady=0)

        # controls bar at bottom
        self.status_bar = ctk.CTkFrame(self, height=32, fg_color="gray20")
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.status_bar.grid_propagate(False)
        self.lbl_bottom = ctk.CTkLabel(
            self.status_bar, text="Willian Viana — Dinneny Lab",
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#cccccc")
        self.lbl_bottom.pack(side="left", padx=10)

        # global keyboard handler — works regardless of which widget has focus
        self.bind_all("<Key>", self._on_global_key)
        self.bind_all("<KeyRelease>", self._on_global_key_release)
        # Prevent spacebar from activating focused buttons — override at
        # class level so space is always routed to canvas for pan mode.
        self.bind_class("TButton", "<space>", lambda e: "break")
        self.bind_class("TButton", "<KeyRelease-space>", lambda e: "break")
        self.bind_class("Button", "<space>", lambda e: "break")
        self.bind_class("Button", "<KeyRelease-space>", lambda e: "break")

        # try to auto-resume last session after window is drawn
        self.after(200, self._try_auto_resume)

    def _on_global_key(self, event):
        """Route keyboard events to canvas, skip if typing in an Entry."""
        widget_class = event.widget.winfo_class()
        if widget_class in ('Entry', 'TEntry', 'Text'):
            return
        # Spacebar: always capture for pan mode, prevent button activation
        if event.keysym == 'space':
            self.canvas.handle_key(event)
            return "break"
        handled = self.canvas.handle_key(event)
        if handled:
            return "break"

    def _on_global_key_release(self, event):
        """Route key release events to canvas."""
        if event.keysym == 'space':
            self.canvas.handle_key_release(event)
            return "break"
        self.canvas.handle_key_release(event)

    # --- Image loading ---

    def _try_auto_resume(self):
        """On startup, show up to 5 most recent sessions from any folder."""
        folders = get_recent_folders()
        sessions = []
        for f in folders:
            for s in get_session_summaries(f):
                sessions.append(s)
                if len(sessions) >= 5:
                    break
            if len(sessions) >= 5:
                break
        if sessions:
            self.sidebar.populate_sessions(sessions)

    def resume_session(self, folder, experiment=''):
        """Resume a saved session from the sessions list."""
        from utils import list_images_in_folder
        self.folder = folder
        self._experiment_name = experiment
        save_last_folder(folder)
        self.images = list_images_in_folder(self.folder)
        if not self.images:
            self.sidebar.set_status(f"No scans found in {folder.name}")
            return
        self.sidebar.sec_sessions.hide()
        self._try_resume()

    def _session_path(self):
        if self.folder:
            return session_dir(self.folder, self._experiment_name) / 'session.json'
        return None

    def _auto_save(self):
        sp = self._session_path()
        if sp:
            try:
                save_session(sp, self)
            except Exception as e:
                print(f"[auto_save] Error: {e}")
                import traceback
                traceback.print_exc()

    def _try_resume(self):
        """Check for session file and silently restore."""
        sp = self._session_path()
        if not sp:
            return False
        data = load_session(sp)
        if not data:
            return False
        # validate images match
        saved_names = set(data.get('images', []))
        current_names = {p.name for p in self.images}
        if not saved_names & current_names:
            return False
        # restore experiment name from session
        settings = data.get('settings', {})
        self._experiment_name = settings.get('experiment', '')
        exp = self._experiment_name
        # restore offsets and genotype color registry
        self._plate_offset = data.get('plate_offset', 0)
        self._root_offset = data.get('root_offset', 0)
        self._genotype_colors = data.get('genotype_colors', {})
        # restore processed images
        name_to_path = {p.name: p for p in self.images}
        self._processed_images = set()
        for name in data.get('processed_images', []):
            if name in name_to_path:
                self._processed_images.add(name_to_path[name])
        # restore settings
        restore_settings(self.sidebar, settings)
        # restore per-image canvas dict
        self._image_canvas_data = data.get('image_canvas_data', {})
        # restore current image and canvas state
        current = data.get('current_image')
        canvas_data = data.get('canvas', {})
        if current and current in name_to_path:
            current_path = name_to_path[current]
            # populate image list first (used in all branches)
            self.sidebar.btn_load_folder.configure(text="Change Folder")
            self.sidebar._populate_image_list(
                self.images, self._processed_images)
            self.sidebar.btn_finish_plot.pack_forget()

            # if current image is already done, go to image selection
            if current_path in self._processed_images:
                self.sidebar.sec_folder.collapse(summary=self.folder.name)
                self.sidebar.advance_to_images(
                    self.folder.name, self.images, self._processed_images)
                self.sidebar.set_status(
                    f"{len(self._processed_images)}/{len(self.images)} "
                    f"scan(s) done. Select next image.")
            else:
                # load the image (triggers advance_to_settings internally)
                self.load_image(current_path)
                # override DPI that load_image set
                self.sidebar.entry_dpi.delete(0, 'end')
                self.sidebar.entry_dpi.insert(0, settings.get('dpi', ''))
                # seed persistent settings from session if not yet saved
                if not get_persistent_settings(self.folder, exp):
                    save_persistent_settings(self.folder, {
                        'multi_measurement': settings.get('multi_measurement', False),
                        'segments': settings.get('segments', ''),
                        'split_plate': settings.get('split_plate', False),
                        'num_plates': settings.get('num_plates', 1),
                    }, exp)
                # advance sidebar to workflow
                self.sidebar.advance_to_experiment()
                self.sidebar.advance_to_workflow()
                # restore canvas state (set_image cleared plates/roots, re-add)
                if canvas_data.get('plates'):
                    self.canvas.set_plates(canvas_data['plates'])
                if canvas_data.get('root_points'):
                    self.canvas.set_roots(
                        canvas_data['root_points'],
                        canvas_data.get('root_flags', []),
                        canvas_data.get('root_groups', []),
                        canvas_data.get('root_plates', []))
                if canvas_data.get('all_marks'):
                    self.canvas.set_marks(canvas_data['all_marks'])
                if canvas_data.get('traces'):
                    self.canvas.set_traces(
                        canvas_data['traces'],
                        canvas_data.get('trace_to_result'))
                step = data.get('workflow_step', 1)
                self.canvas.set_mode(ImageCanvas.MODE_VIEW)
                self.canvas._redraw()
                self.sidebar.set_step(step)
                self.sidebar.sec_folder.collapse(summary=self.folder.name)
                plates = self.canvas.get_plates()
                points = self.canvas.get_root_points()
                # resume into root/marks clicking if saved mid-click
                if step == 2 and plates and points:
                    cs = data.get('click_state', {})
                    self._current_plate_idx = cs.get('plate_idx', 0)
                    self._split = cs.get('split', False)
                    self._current_group = cs.get('current_group', 0)
                    self._split_stage = cs.get('split_stage', 0)
                    self.canvas._current_root_group = self._current_group
                    # restore in-progress mark points
                    mark_pts = canvas_data.get('mark_points', [])
                    if mark_pts:
                        self.canvas._mark_points = [tuple(p) for p in mark_pts]
                        self._resume_marks_phase()
                    else:
                        self.click_roots(resume=True)
                    self.sidebar.set_status(
                        f"Session restored: {len(points)} root(s) on "
                        f"{len(plates)} plate(s).\n"
                        f"Continue where you left off.")
                elif step >= 4 and self.canvas._traces:
                    # measurement done or in review — show action buttons
                    self._split = self.sidebar.is_split_plate()
                    # rebuild workflow state so review/retrace/manual-trace work
                    self._rebuild_results_from_traces()
                    self.sidebar.set_step(5)
                    self.canvas._measurement_done = True
                    self.sidebar.btn_select_plates.configure(state="normal")
                    self.sidebar.btn_click_roots.configure(state="normal")
                    self.sidebar.btn_measure.configure(state="normal")
                    self.sidebar.btn_review.configure(state="normal")
                    self._hide_action_buttons()
                    self._show_action_frame()
                    self.btn_next_image.pack(
                        pady=(10, 3), padx=15, fill="x")
                    self.btn_continue_later.pack(
                        pady=3, padx=15, fill="x")
                    self.btn_stop.pack(pady=3, padx=15, fill="x")
                    self.sidebar.set_status(
                        f"Session restored: {len(plates)} plate(s), "
                        f"{len(points)} root(s).\n"
                        f"Click Review Traces to review.")
                else:
                    self.sidebar.set_status(
                        f"Session restored: {len(plates)} plate(s), "
                        f"{len(points)} root(s).\n"
                        f"Continue from where you left off.")
                    if plates:
                        self.sidebar.btn_click_roots.configure(state="normal")
                    if points:
                        self.sidebar.btn_measure.configure(state="normal")
        else:
            self.sidebar.advance_to_images(
                self.folder.name, self.images, self._processed_images)
        return True

    def load_folder(self):
        from tkinter import filedialog
        from utils import list_images_in_folder

        folder = filedialog.askdirectory(title="Select image folder")
        self.lift()
        self.focus_force()
        if not folder:
            return
        self.folder = Path(folder)
        save_last_folder(self.folder)
        self.images = list_images_in_folder(self.folder)

        if not self.images:
            self.sidebar.sec_sessions.hide()
            self.sidebar.btn_load_folder.configure(text="Open Folder")
            self.sidebar.set_status(f"No scans found in {self.folder.name}")
            return

        # Reset all experiment state for new folder
        self._experiment_name = ''
        self._genotype_colors = {}
        self._processed_images = set()
        self._plate_offset = 0
        self._root_offset = 0
        self._image_canvas_data = {}
        self.image = None
        self.image_path = None
        self.sidebar.advance_to_images(self.folder.name, self.images)
        # show this folder's sessions (if any) so user can click to resume
        sessions = get_session_summaries(self.folder)
        if sessions:
            self.sidebar.populate_sessions(sessions)
        else:
            self.sidebar.sec_sessions.hide()

    def load_image(self, path):
        """Load and display a single image."""
        # Hide action buttons from previous image
        self._hide_action_buttons()
        # Exit preview mode if active (clicking same image exits preview)
        if getattr(self, '_preview_active', False) and path == self.image_path:
            self._preview_active = False
            display = _to_uint8(self.image)
            self.canvas.set_image_preview(display)
            self.sidebar.set_status("Preview exited")
            return
        self._preview_active = False
        # If clicking the same image that's already done, restore completed view
        if (path == self.image_path and self.canvas._measurement_done
                and path in self._processed_images):
            self._restore_completed_view()
            return
        # Stash canvas from previous image before switching
        self._stash_canvas()
        self.image_path = path
        # Reset workflow state from previous image
        self._results = []
        self._trace_to_result = []
        self._binary = None
        self._retry_result_indices = []
        self._reclick_idx = 0
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
            self.canvas._measurement_done = False
            display = _to_uint8(img)
            self.canvas.set_image(display)  # clears plates/roots/marks/traces

            # Finished image — restore its saved canvas data
            if path in self._processed_images:
                self._restore_image_canvas(path.name)
                self.canvas._measurement_done = True
                self.canvas._redraw()
                self._restore_completed_view()
                self.sidebar.set_status(
                    f"Image already measured. "
                    f"Use workflow buttons to review.")
                return

            detected = _detect_dpi(path)
            dpi = detected or 1200
            self.sidebar.advance_to_settings(path.name, dpi)

            # Update auto-threshold display for new image
            if self.sidebar.var_auto_thresh.get():
                self._update_auto_threshold()

            # restore settings from previous scan
            if self.folder:
                ps = get_persistent_settings(self.folder, self._experiment_name)
                if ps:
                    # Restore all three fields with their saved values
                    num_plates = ps.get('num_plates', 1)
                    self.sidebar.entry_num_plates.delete(0, 'end')
                    self.sidebar.entry_num_plates.insert(0, str(num_plates))

                    geno_per_plate = ps.get('genotypes_per_plate', 1)
                    self.sidebar.entry_genotypes_per_plate.delete(0, 'end')
                    self.sidebar.entry_genotypes_per_plate.insert(0, str(geno_per_plate))

                    segs = ps.get('segments', '1') or '1'
                    self.sidebar.entry_segments.delete(0, 'end')
                    self.sidebar.entry_segments.insert(0, segs)

                    if ps.get('multi_measurement'):
                        self.sidebar.var_multi.set(True)
                        self.sidebar._toggle_segments()

            dpi_src = "detected" if detected else "default"
            self.sidebar.set_status(
                f"Loaded: {img.shape[1]}x{img.shape[0]}, {img.dtype}\n"
                f"DPI: {dpi} ({dpi_src})")
        except Exception as e:
            self.sidebar.set_status(f"Error: {e}")

    def _restore_completed_view(self):
        """Show completed measurement view for the current image."""
        self._split = self.sidebar.is_split_plate()
        if self.canvas._traces and not getattr(self, '_results', None):
            self._rebuild_results_from_traces()
        self._hide_action_buttons()
        self.sidebar.sec_sessions.hide()
        self.sidebar.sec_folder.collapse(summary=self.folder.name if self.folder else "")
        self.sidebar.sec_settings.show()
        self.sidebar.sec_settings.collapse(
            summary=f"{self.sidebar.entry_dpi.get()} DPI")
        self.sidebar.sec_experiment.show()
        self.sidebar.sec_experiment.collapse(
            summary=self.sidebar.entry_genotypes.get().strip() or "genotype")
        self.sidebar.sec_workflow.show()
        self.sidebar.sec_workflow.expand()
        self.sidebar.set_step(5)
        self.sidebar.btn_select_plates.configure(state="normal")
        self.sidebar.btn_click_roots.configure(state="normal")
        self.sidebar.btn_measure.configure(state="normal")
        self.sidebar.btn_review.configure(state="normal")
        self._hide_action_buttons()
        self._show_action_frame()
        self.btn_next_image.pack(pady=(10, 3), padx=15, fill="x")
        self.btn_continue_later.pack(pady=3, padx=15, fill="x")
        self.btn_stop.pack(pady=3, padx=15, fill="x")
        plates = self.canvas.get_plates()
        points = self.canvas.get_root_points()
        self.sidebar.set_status(
            f"Measurement complete: {len(plates)} plate(s), "
            f"{len(points)} root(s).\n"
            f"Click Review Traces to review.")
        self.lbl_bottom.configure(text="Willian Viana — Dinneny Lab")
        self._clear_plate_info()


    # --- Settings helpers ---

    def _ordinal(self, n):
        """Return ordinal string for n (1st, 2nd, 3rd, etc.)."""
        if 11 <= n % 100 <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def _get_num_marks(self):
        """Get number of marks per root from sidebar (0 if single segment)."""
        text = self.sidebar.entry_segments.get().strip()
        try:
            segments = int(text)
            if segments >= 2:
                return segments - 1  # N segments = N-1 marks
        except (ValueError, TypeError):
            pass
        return 0  # default: 1 segment = no marks needed

    def _register_genotype(self, name):
        """Return stable color index for a genotype, assigning next index if unseen."""
        if name not in self._genotype_colors:
            self._genotype_colors[name] = len(self._genotype_colors)
        return self._genotype_colors[name]

    def _set_plate_info(self, pi):
        """Set plate info overlay on the canvas for plate pi."""
        genotypes = [g.strip() for g in
                     self.sidebar.entry_genotypes.get().split(",")
                     if g.strip()]
        cond_text = self.sidebar.entry_condition.get().strip()
        conditions = [c.strip() for c in cond_text.split(",")
                      if c.strip()] if cond_text else []
        cond = conditions[pi] if pi < len(conditions) else (
            conditions[-1] if conditions else "")

        info = {'center': f"Plate {pi + 1}"}
        if self._split:
            geno_a = genotypes[0] if genotypes else "genotype_A"
            geno_b = genotypes[1] if len(genotypes) >= 2 else "genotype_B"
            info['left'] = f"{geno_a} {cond}".strip() if cond else geno_a
            info['right'] = f"{geno_b} {cond}".strip() if cond else geno_b
            idx_a = self._genotype_colors.get(geno_a, 0)
            idx_b = self._genotype_colors.get(geno_b, 1)
            info['left_color'] = GROUP_MARKER_COLORS[idx_a % len(GROUP_MARKER_COLORS)]
            info['right_color'] = GROUP_MARKER_COLORS[idx_b % len(GROUP_MARKER_COLORS)]
        else:
            geno = (genotypes[pi] if pi < len(genotypes)
                    else genotypes[-1] if genotypes else "genotype")
            info['left'] = f"{geno} {cond}".strip() if cond else geno
            idx = self._genotype_colors.get(geno, pi)
            info['left_color'] = GROUP_MARKER_COLORS[idx % len(GROUP_MARKER_COLORS)]
        self.canvas._plate_info = info

    def _clear_plate_info(self):
        """Clear plate info overlay from the canvas."""
        self.canvas._plate_info = None

    def _hide_action_buttons(self):
        """Hide all action buttons and the frame."""
        self.btn_next_image.pack_forget()
        self.btn_continue_later.pack_forget()
        self.btn_stop.pack_forget()
        self.btn_continue_later_mid.pack_forget()
        self._action_frame.grid_remove()
        # Hide sidebar Done button and manual trace button too
        self.sidebar.btn_done.pack_forget()
        self.sidebar.btn_manual_trace.pack_forget()
        self.sidebar.hide_review_toggles()

    def _show_action_frame(self):
        """Ensure action button frame is visible (after hiding all buttons first)."""
        # Clear any stale buttons before showing frame
        self.btn_next_image.pack_forget()
        self.btn_continue_later.pack_forget()
        self.btn_stop.pack_forget()
        self.btn_continue_later_mid.pack_forget()
        self._action_frame.grid()

    def _get_scale(self):
        """Get scale (px/cm) from DPI entry or auto-detect."""
        dpi_text = self.sidebar.entry_dpi.get().strip()
        if dpi_text:
            # Handle "800 dpi" format - extract just the number
            dpi_num = dpi_text.lower().replace("dpi", "").strip()
            try:
                dpi = int(dpi_num)
                if dpi > 0:
                    return dpi / 2.54
            except ValueError:
                pass
        if self.image_path:
            detected = _detect_dpi(self.image_path)
            if detected:
                self.sidebar.entry_dpi.delete(0, "end")
                self.sidebar.entry_dpi.insert(0, f"{detected} dpi")
                return detected / 2.54
        self.sidebar.entry_dpi.delete(0, "end")
        self.sidebar.entry_dpi.insert(0, "1200 dpi")
        return SCALE_PX_PER_CM

    def _preview_preprocessing(self, force_show=False):
        """Show preprocessed binary image for current settings, or toggle back to original."""
        if self.image is None:
            self.sidebar.set_status("Load an image first")
            return
        # Toggle off if already showing preview (unless force_show for settings updates)
        if getattr(self, '_preview_active', False) and not force_show:
            self._preview_active = False
            display = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB) if len(self.image.shape) == 2 else self.image
            if display.dtype == np.uint16:
                display = (display / 256).astype(np.uint8)
            self.canvas.set_image_preview(display)
            self.sidebar.set_status("Ready")
            return
        try:
            from image_processing import preprocess
            scale = self._get_scale()
            sensitivity = self.sidebar.var_sensitivity.get()
            threshold = self.sidebar.get_threshold()
            binary = preprocess(self.image, scale=scale, sensitivity=sensitivity,
                                threshold=threshold)
            # Convert to displayable format (white roots on black)
            display = (binary.astype(np.uint8) * 255)
            # Use preview method to preserve overlays and zoom
            self.canvas.set_image_preview(display)
            self._preview_active = True
            self.sidebar.set_status("Preview - click Preview again to exit")
        except Exception as e:
            self.sidebar.set_status(f"Preview error: {e}")

    def _update_auto_threshold(self):
        """Calculate and display auto-detected threshold value."""
        if self.image is None:
            return
        try:
            # Get base Otsu threshold (always in 8-bit range)
            if self.image.dtype == np.uint16:
                img8 = (self.image / 256).astype(np.uint8)
                otsu_thresh, _ = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                otsu_thresh, _ = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Adjust based on sensitivity: thin roots need higher threshold
            # Higher DPI (>=500, ~197 px/cm) needs less adjustment
            scale = self._get_scale()
            high_res = scale >= 197
            sensitivity = self.sidebar.var_sensitivity.get()
            if sensitivity == 'thin':
                adj = 30 if high_res else 45
                otsu_thresh = min(255, otsu_thresh + adj)
            elif sensitivity == 'medium':
                adj = 15 if high_res else 25
                otsu_thresh = min(255, otsu_thresh + adj)
            # thick: use Otsu value as-is

            self.sidebar.set_auto_threshold_value(int(otsu_thresh))
        except Exception:
            pass

    def _exit_preview(self):
        """Exit preview mode and restore original image."""
        if getattr(self, '_preview_active', False) and self.image is not None:
            self._preview_active = False
            display = _to_uint8(self.image)
            # Use preview method to preserve overlays and zoom
            self.canvas.set_image_preview(display)

    # --- Sidebar phase callbacks ---

    def _on_next_settings(self):
        """Called when user clicks Next on image settings."""
        self._exit_preview()
        self.sidebar.advance_to_experiment()

    def _on_start_workflow(self):
        """Called when user clicks Start Workflow."""
        self._exit_preview()
        from csv_output import get_offsets_from_csv
        self._experiment_name = self.sidebar.entry_experiment.get().strip()
        exp = self._experiment_name
        if self.folder:
            csv_path = data_dir(self.folder, exp) / 'raw_data.csv'
            # seed offsets from existing CSV so new data continues correctly
            if csv_path.exists():
                csv_plate_off, csv_root_off = get_offsets_from_csv(csv_path)
                self._plate_offset = max(self._plate_offset, csv_plate_off)
                self._root_offset = max(self._root_offset, csv_root_off)
            save_experiment_name(self.folder, exp, exp)
            try:
                num_plates = int(self.sidebar.entry_num_plates.get().strip() or "1")
            except (ValueError, TypeError):
                num_plates = 1
            save_persistent_settings(self.folder, {
                'multi_measurement': self.sidebar.var_multi.get(),
                'segments': self.sidebar.entry_segments.get().strip(),
                'split_plate': self.sidebar.is_split_plate(),
                'genotypes_per_plate': self.sidebar.get_genotypes_per_plate(),
                'num_plates': num_plates,
            }, exp)
        self.sidebar.advance_to_workflow()
        self.select_plates()

    def _stash_canvas(self):
        """Snapshot current canvas state into per-image dict."""
        if self.image_path:
            from session import _collect_canvas
            self._image_canvas_data[self.image_path.name] = _collect_canvas(self.canvas)

    def _restore_image_canvas(self, image_name):
        """Restore canvas state from per-image dict."""
        cd = self._image_canvas_data.get(image_name, {})
        if cd.get('plates'):
            self.canvas.set_plates(cd['plates'])
        if cd.get('root_points'):
            self.canvas.set_roots(
                cd['root_points'],
                cd.get('root_flags', []),
                cd.get('root_groups', []),
                cd.get('root_plates', []))
        if cd.get('all_marks'):
            self.canvas.set_marks(cd['all_marks'])
        if cd.get('traces'):
            self.canvas.set_traces(
                cd['traces'], cd.get('trace_to_result'))

    def next_image(self):
        """Return to image selection after finishing measurement."""
        # re-save CSV/screenshot/metadata in case retrace changed results (silent to avoid duplicate status)
        if getattr(self, '_results', None):
            plates = self.canvas.get_plates()
            self._save_results(self._results, plates, self._scale_val, silent=True)
            self._save_trace_screenshot(silent=True)
            self._save_metadata()
        if self.image_path:
            self._processed_images.add(self.image_path)
        self._stash_canvas()
        self._auto_save()
        self.sidebar.advance_to_images(
            self.folder.name, self.images, self._processed_images)

    def continue_later(self):
        """Save session and quit the app."""
        self._stash_canvas()
        self._auto_save()
        self.destroy()

    def finish_and_plot(self):
        """All images done — generate final plot."""
        if self.sidebar.var_plot.get():
            self._run_plot()
        # save final session state (keeps session visible on next launch)
        self._auto_save()
        self.sidebar.set_status(
            self.sidebar.lbl_status.cget("text") +
            "\nAll done!")
        self.destroy()

    # --- Plate & root clicking flow ---

    def select_plates(self):
        """Enter plate selection mode on canvas."""
        self._exit_preview()
        self._hide_action_buttons()
        self.canvas._measurement_done = False
        self.canvas.clear_plates()
        self.canvas.clear_roots()
        self.canvas.clear_marks()
        self.canvas.clear_traces()
        self.canvas.clear_review()  # clear any review selection highlights
        self._clear_plate_info()
        self.canvas._fit_image()
        self.canvas._redraw()
        # Reset all workflow state
        self._results = []
        self._trace_to_result = []
        self._binary = None
        self._retry_result_indices = []
        self._reclick_idx = 0
        self._current_plate_idx = 0
        self._split_stage = 0
        # Clear callbacks before setting new ones
        self.canvas._on_click_callback = None
        self.canvas._app_status_callback = self.sidebar.set_status
        self.canvas._on_click_callback = self._on_plate_added
        self.canvas.set_mode(
            ImageCanvas.MODE_SELECT_PLATES,
            on_done=self._plates_done)
        self.sidebar.set_status(
            "Draw rectangle around plate.\n"
            "Adjust by redrawing.")
        self.lbl_bottom.configure(
            text="Drag=draw plate  |  Redraw=adjust  |  Right-click=undo  |  Scroll=zoom")

        self.sidebar.btn_click_roots.configure(state="disabled")
        self.sidebar.btn_measure.configure(state="disabled")
        self.sidebar.btn_review.configure(state="disabled")
        self._hide_action_buttons()
        self._show_action_frame()
        # Done button shown after first plate is drawn (via _on_plate_added)
        self.btn_continue_later_mid.pack(pady=(10, 5), padx=15, fill="x")
        self.sidebar.set_step(1)

    def _on_plate_added(self):
        """Show Done button when first plate is drawn, auto-advance if expected count reached."""
        confirmed_plates = self.canvas.get_plates()
        num_confirmed = len(confirmed_plates)
        num_with_pending = num_confirmed + (1 if self.canvas._pending_plate else 0)

        if num_with_pending > 0:
            # Check expected number of plates
            try:
                expected = int(self.sidebar.entry_num_plates.get().strip() or "1")
            except (ValueError, TypeError):
                expected = 1
            expected = max(1, expected)

            if expected == 1:
                # Single plate - show confirm button
                btn_text = "Confirm Plate"
            elif num_with_pending < expected:
                # Multiple plates - show progress
                btn_text = f"Plate {num_with_pending}/{expected}"
            elif num_confirmed >= expected:
                # All plates CONFIRMED (not just drawn) - auto-advance
                self.after(100, self._plates_done)
                return
            else:
                # All plates drawn but pending not yet confirmed
                btn_text = f"Confirm Plate {num_with_pending}/{expected}"

            self.sidebar.btn_done.configure(text=btn_text)
            if not self.sidebar.btn_done.winfo_ismapped():
                self.sidebar.btn_done.pack(pady=(5, 0), padx=15, fill="x")

    def _on_root_clicked(self):
        """Callback when a root is clicked (button already visible)."""
        pass

    def _plates_done(self):
        """Called when user presses Enter after selecting plates."""
        plates = self.canvas.get_plates()
        if not plates:
            self.sidebar.set_status("No plates selected. Draw at least one rectangle.")
            return
        self.canvas._app_status_callback = None
        self.canvas.set_mode(ImageCanvas.MODE_VIEW)
        self.sidebar.set_status(f"{len(plates)} plate(s) selected.")
        self.lbl_bottom.configure(text="Willian Viana — Dinneny Lab")
        self._clear_plate_info()
        self.sidebar.btn_click_roots.configure(state="normal")
        self.sidebar.set_step(2)
        self._auto_save()
        self.click_roots()

    def click_roots(self, resume=False):
        """Enter root clicking mode on canvas."""
        self._exit_preview()
        self._hide_action_buttons()
        plates = self.canvas.get_plates()
        if not plates:
            self.sidebar.set_status("Select plates first.")
            return
        # Clear callbacks before setting new ones
        self.canvas._on_click_callback = None
        if not resume:
            self.canvas.clear_roots()
            self.canvas.clear_marks()
            self.canvas.clear_traces()
            self.canvas.clear_review()  # clear any review selection highlights
            self._current_plate_idx = 0
            self._split = self.sidebar.is_split_plate()
            self._split_stage = 0
            # _current_group set per-plate in _enter_root_click_stage via registry
        self.sidebar.btn_measure.configure(state="disabled")
        self.sidebar.btn_review.configure(state="disabled")
        self._hide_action_buttons()
        self._show_action_frame()
        self.btn_continue_later_mid.pack(pady=(10, 5), padx=15, fill="x")
        self.sidebar.set_step(2)
        # _enter_root_click_stage shows Done button
        self._enter_root_click_stage()

    def _enter_root_click_stage(self):
        """Set up the canvas for the current root clicking stage."""
        plates = self.canvas.get_plates()
        pi = self._current_plate_idx
        r1, r2, c1, c2 = plates[pi]

        # resolve genotype name and register for stable color index
        genotypes = [g.strip() for g in
                     self.sidebar.entry_genotypes.get().split(",")
                     if g.strip()]
        cond_text = self.sidebar.entry_condition.get().strip()
        conditions = [c.strip() for c in cond_text.split(",")
                      if c.strip()] if cond_text else []
        if self._split:
            geno_name = (genotypes[self._split_stage]
                         if self._split_stage < len(genotypes)
                         else f"group_{self._split_stage}")
        else:
            geno_name = (genotypes[pi] if pi < len(genotypes)
                         else genotypes[-1] if genotypes else "genotype")
        self._current_group = self._register_genotype(geno_name)
        self.canvas._current_root_group = self._current_group
        self.canvas._current_plate_idx = self._current_plate_idx
        self._set_plate_info(pi)
        self.canvas._on_click_callback = self._on_root_clicked
        self.canvas.set_mode(
            ImageCanvas.MODE_CLICK_ROOTS,
            on_done=self._plate_roots_done)
        self.canvas.zoom_to_region(r1, r2, c1, c2)

        # build status label
        label_parts = [geno_name]
        if conditions:
            cond = conditions[pi] if pi < len(conditions) else conditions[-1]
            label_parts.append(cond)
        label = " / ".join(label_parts)
        self.sidebar.set_status(
            f"Plate {pi + 1}/{len(plates)} — {label}\n"
            "Click root tops. D+Click=dead, T+Click=touching.")
        self.lbl_bottom.configure(
            text="Click=root top  |  D+Click=dead  |  T+Click=touching  |  Right-click=undo  |  Scroll=zoom")
        # Determine button text based on what's next
        is_last = (pi == len(plates) - 1) and (not self._split or self._split_stage == 1)
        num_marks = self._get_num_marks()
        if num_marks > 0:
            btn_text = "Click 1st Segments"
        elif is_last:
            btn_text = "Start Trace"
        else:
            btn_text = "Next Plate"
        self.sidebar.btn_done.configure(text=btn_text)
        if not self.sidebar.btn_done.winfo_ismapped():
            self.sidebar.btn_done.pack(pady=(5, 0), padx=15, fill="x")

    def _plate_roots_done(self):
        """Called when user presses Enter after clicking roots on a plate."""
        self._auto_save()
        num_marks = self._get_num_marks()
        if num_marks > 0:
            self._start_marks_phase()
            return
        self._advance_to_next_stage()

    def _advance_to_next_stage(self):
        """Advance to next genotype group or next plate."""
        plates = self.canvas.get_plates()
        if self._split and self._split_stage == 0:
            # stay on same plate, switch to genotype B
            self._split_stage = 1
            self._enter_root_click_stage()
            return
        # advance to next plate
        self._split_stage = 0
        self._current_plate_idx += 1
        if self._current_plate_idx < len(plates):
            self._enter_root_click_stage()
            return
        # all plates done — auto-measure
        points = self.canvas.get_root_points()
        if not points:
            self.sidebar.set_status("No roots clicked. Click at least one root top.")
            self._current_plate_idx = 0
            return
        self.canvas.set_mode(ImageCanvas.MODE_VIEW)
        self.sidebar.btn_measure.configure(state="normal")
        self.measure()

    # --- Marks phase ---

    def _resume_marks_phase(self):
        """Resume mark clicking with saved mark points."""
        plates = self.canvas.get_plates()
        pi = self._current_plate_idx
        r1, r2, c1, c2 = plates[pi]
        points = self.canvas.get_root_points()
        flags = self.canvas.get_root_flags()
        groups = self.canvas.get_root_groups()
        current_group = self._current_group
        self._marks_plate_roots = []
        display_numbers = []
        position = 0
        for i, ((row, col), flag) in enumerate(zip(points, flags)):
            if not (r1 <= row <= r2 and c1 <= col <= c2):
                continue
            if self._split and i < len(groups) and groups[i] != current_group:
                continue
            position += 1
            if flag is not None:
                continue
            self._marks_plate_roots.append(i)
            display_numbers.append(position)
        num_marks = self._get_num_marks()
        self.canvas._marks_expected = len(self._marks_plate_roots) * num_marks
        self.canvas._marks_display_numbers = display_numbers
        # Clear callback before setting new one
        self.canvas._on_click_callback = None
        self.canvas._on_click_callback = self._update_marks_status
        self._set_plate_info(pi)
        self.canvas.set_mode(
            ImageCanvas.MODE_CLICK_MARKS,
            on_done=self._plate_marks_done)
        self.canvas.zoom_to_region(r1, r2, c1, c2)
        self._hide_action_buttons()
        self._show_action_frame()
        self.sidebar.btn_done.pack(pady=(5, 0), padx=15, fill="x")
        self.btn_continue_later_mid.pack(pady=(10, 5), padx=15, fill="x")
        self.sidebar.set_step(2)
        self._update_marks_status()
        self.lbl_bottom.configure(
            text="Click=place mark  |  Right-click=undo  |  Scroll=zoom")

    def _start_marks_phase(self):
        """Enter mark clicking mode for normal roots on the current plate/group."""
        plates = self.canvas.get_plates()
        pi = self._current_plate_idx
        r1, r2, c1, c2 = plates[pi]
        points = self.canvas.get_root_points()
        flags = self.canvas.get_root_flags()
        groups = self.canvas.get_root_groups()
        current_group = self._current_group
        self._marks_plate_roots = []
        display_numbers = []
        position = 0
        for i, ((row, col), flag) in enumerate(zip(points, flags)):
            if not (r1 <= row <= r2 and c1 <= col <= c2):
                continue
            if self._split and i < len(groups) and groups[i] != current_group:
                continue
            position += 1
            if flag is not None:
                continue
            self._marks_plate_roots.append(i)
            display_numbers.append(position)
        if not self._marks_plate_roots:
            self._advance_to_next_stage()
            return
        num_marks = self._get_num_marks()
        self.canvas._marks_expected = len(self._marks_plate_roots) * num_marks
        self.canvas._marks_display_numbers = display_numbers
        # Only clear temporary mark points, not _all_marks (preserves other plates)
        for rid in self.canvas._mark_marker_ids:
            self.canvas.canvas.delete(rid)
        self.canvas._mark_points.clear()
        self.canvas._mark_marker_ids.clear()
        # Clear callback before setting new one
        self.canvas._on_click_callback = None
        self.canvas._on_click_callback = self._update_marks_status
        self._set_plate_info(pi)
        self.canvas.set_mode(
            ImageCanvas.MODE_CLICK_MARKS,
            on_done=self._plate_marks_done)
        self.canvas.zoom_to_region(r1, r2, c1, c2)
        if not self.sidebar.btn_done.winfo_ismapped():
            self.sidebar.btn_done.pack(pady=(5, 0), padx=15, fill="x")
        self._update_marks_status()
        self.lbl_bottom.configure(
            text="Click=place mark  |  Right-click=undo  |  Scroll=zoom")

    def _update_marks_status(self):
        """Update status bar and button with marks progress."""
        pi = self._current_plate_idx
        plates = self.canvas.get_plates()
        num_marks = self._get_num_marks()
        n_roots = len(self._marks_plate_roots)
        expected = self.canvas._marks_expected
        placed = len(self.canvas.get_mark_points())
        is_last_plate = (pi == len(plates) - 1) and (not self._split or self._split_stage == 1)
        if placed >= expected:
            self.sidebar.set_status(
                f"Plate {pi + 1} — All {expected} mark(s) placed.\n"
                "Press Enter to continue.")
            btn_text = "Start Trace" if is_last_plate else "Next Plate"
        else:
            current_root_seq = placed // num_marks
            current_mark_in_root = placed % num_marks + 1
            display_nums = self.canvas._marks_display_numbers
            if current_root_seq < len(display_nums):
                root_label = str(display_nums[current_root_seq])
            else:
                root_label = str(current_root_seq + 1)
            self.sidebar.set_status(
                f"Plate {pi + 1} — Marks: {placed}/{expected}.\n"
                f"Root {root_label}/{n_roots}, "
                f"mark {current_mark_in_root}/{num_marks}.\n"
                "Right-click=undo. Enter when all placed.")
            # Mark number + 1 because roots are "1st segments"
            ordinal = self._ordinal(current_mark_in_root + 1)
            btn_text = f"Click {ordinal} Segments"
        self.sidebar.btn_done.configure(text=btn_text)

    def _plate_marks_done(self):
        """Called when user presses Enter after clicking marks for a plate."""
        all_marks = self.canvas.get_mark_points()
        if len(all_marks) < self.canvas._marks_expected:
            self.sidebar.set_status(
                f"Need {self.canvas._marks_expected} mark(s), "
                f"only {len(all_marks)} placed. Click more marks.")
            return
        num_marks = self._get_num_marks()
        for j, ri in enumerate(self._marks_plate_roots):
            start = j * num_marks
            self.canvas._all_marks[ri] = all_marks[start:start + num_marks]
        self._auto_save()
        self._advance_to_next_stage()


def main():
    app = RootMeasureApp()
    app.mainloop()


if __name__ == '__main__':
    main()
