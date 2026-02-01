#!/usr/bin/env python3
"""Root Measure GUI — CustomTkinter app shell."""

import sys
from pathlib import Path

import customtkinter as ctk
import cv2
import tifffile

# allow importing from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from config import SCALE_PX_PER_CM
from plate_detection import _to_uint8

from canvas import ImageCanvas
from sidebar import Sidebar
from workflow import MeasurementMixin
from session import save_session, load_session, restore_settings, \
    save_last_folder, get_last_folder

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

        # layout: sidebar + canvas
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = Sidebar(self, app=self)
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=0, pady=0)

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
        self.canvas.handle_key(event)

    def _on_global_key_release(self, event):
        """Route key release events to canvas."""
        if event.keysym == 'space':
            self.canvas.handle_key_release(event)
            return "break"
        self.canvas.handle_key_release(event)

    # --- Image loading ---

    def _try_auto_resume(self):
        """On startup, check if the last folder has a session to resume."""
        from utils import list_images_in_folder
        last = get_last_folder()
        if not last:
            return
        sp = last / 'output' / 'session.json'
        data = load_session(sp)
        if not data:
            return
        self.folder = last
        self.images = list_images_in_folder(self.folder)
        if not self.images:
            return
        if self._try_resume():
            return
        # session existed but user declined resume — show image list
        self.sidebar.advance_to_images(self.folder.name, self.images)

    def _session_path(self):
        if self.folder:
            return self.folder / 'output' / 'session.json'
        return None

    def _auto_save(self):
        sp = self._session_path()
        if sp:
            save_session(sp, self)

    def _try_resume(self):
        """Check for session file and offer to resume."""
        from tkinter import messagebox
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
        n_done = len(data.get('processed_images', []))
        n_total = len(data.get('images', []))
        if not messagebox.askyesno(
                "Resume Session?",
                f"Found a previous session with {n_done}/{n_total} "
                f"image(s) done.\nResume where you left off?"):
            sp.unlink(missing_ok=True)
            return False
        # restore offsets
        self._plate_offset = data.get('plate_offset', 0)
        self._root_offset = data.get('root_offset', 0)
        # restore processed images
        name_to_path = {p.name: p for p in self.images}
        self._processed_images = set()
        for name in data.get('processed_images', []):
            if name in name_to_path:
                self._processed_images.add(name_to_path[name])
        # restore settings
        settings = data.get('settings', {})
        restore_settings(self.sidebar, settings)
        # restore current image and canvas state
        current = data.get('current_image')
        canvas_data = data.get('canvas', {})
        if current and current in name_to_path:
            # collapse folder and populate image list first
            self.sidebar.advance_to_images(
                self.folder.name, self.images, self._processed_images)
            self.load_image(name_to_path[current])
            # override DPI that load_image set
            self.sidebar.entry_dpi.delete(0, 'end')
            self.sidebar.entry_dpi.insert(0, settings.get('dpi', ''))
            # advance sidebar to workflow
            self.sidebar.advance_to_experiment()
            self.sidebar.advance_to_workflow()
            # restore canvas state
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
            self.canvas.set_mode(ImageCanvas.MODE_VIEW)
            self.canvas._redraw()
            step = data.get('workflow_step', 1)
            self.sidebar.set_step(step)
            plates = self.canvas.get_plates()
            points = self.canvas.get_root_points()
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
        if not folder:
            return
        self.folder = Path(folder)
        save_last_folder(self.folder)
        self.images = list_images_in_folder(self.folder)

        if not self.images:
            self.sidebar.set_status(f"No scans found in {self.folder.name}")
            return

        if self._try_resume():
            return

        self.sidebar.advance_to_images(self.folder.name, self.images)

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

            detected = _detect_dpi(path)
            dpi = detected or 1200
            self.sidebar.advance_to_settings(path.name, dpi)

            dpi_src = "detected" if detected else "default"
            self.sidebar.set_status(
                f"Loaded: {img.shape[1]}x{img.shape[0]}, {img.dtype}\n"
                f"DPI: {dpi} ({dpi_src})")
        except Exception as e:
            self.sidebar.set_status(f"Error: {e}")

    # --- Settings helpers ---

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
        if self.image_path:
            detected = _detect_dpi(self.image_path)
            if detected:
                self.sidebar.entry_dpi.delete(0, "end")
                self.sidebar.entry_dpi.insert(0, str(detected))
                return detected / 2.54
        self.sidebar.entry_dpi.delete(0, "end")
        self.sidebar.entry_dpi.insert(0, "1200")
        return SCALE_PX_PER_CM

    # --- Sidebar phase callbacks ---

    def _on_next_settings(self):
        """Called when user clicks Next on image settings."""
        self.sidebar.advance_to_experiment()

    def _on_start_workflow(self):
        """Called when user clicks Start Workflow."""
        self.sidebar.advance_to_workflow()
        self.select_plates()

    def next_image(self):
        """Return to image selection after finishing measurement."""
        if self.image_path:
            self._processed_images.add(self.image_path)
        self._auto_save()
        self.sidebar.advance_to_images(
            self.folder.name, self.images, self._processed_images)

    def continue_later(self):
        """Save session and quit the app."""
        if self.image_path:
            self._processed_images.add(self.image_path)
        self._auto_save()
        self.destroy()

    def finish_and_plot(self):
        """All images done — generate final plot."""
        if self.sidebar.var_plot.get():
            self._run_plot()
        # clean up session file
        sp = self._session_path()
        if sp and sp.exists():
            sp.unlink()
        self.sidebar.set_status(
            self.sidebar.lbl_status.cget("text") +
            "\nAll done!")
        self.destroy()

    # --- Plate & root clicking flow ---

    def select_plates(self):
        """Enter plate selection mode on canvas."""
        self.canvas.clear_plates()
        self.canvas.clear_roots()
        self.canvas.clear_traces()
        self.canvas._app_status_callback = self.sidebar.set_status
        self.canvas.set_mode(
            ImageCanvas.MODE_SELECT_PLATES,
            on_done=self._plates_done)
        self.sidebar.set_status(
            "Draw rectangle around plate.\n"
            "Adjust by redrawing. Enter=confirm.")
        self.lbl_bottom.configure(
            text="Drag=draw plate  |  Redraw=adjust  |  Right-click=undo  |  Enter=confirm  |  Scroll=zoom")
        self.sidebar.btn_click_roots.configure(state="disabled")
        self.sidebar.btn_measure.configure(state="disabled")
        self.sidebar.set_step(1)

    def _plates_done(self):
        """Called when user presses Enter after selecting plates."""
        plates = self.canvas.get_plates()
        if not plates:
            self.sidebar.set_status("No plates selected. Draw at least one rectangle.")
            return
        self.canvas.set_mode(ImageCanvas.MODE_VIEW)
        self.sidebar.set_status(f"{len(plates)} plate(s) selected.")
        self.lbl_bottom.configure(text="Willian Viana — Dinneny Lab")
        self.sidebar.btn_click_roots.configure(state="normal")
        self.sidebar.set_step(2)
        self._auto_save()
        self.click_roots()

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
        self._split = self.sidebar.var_split.get()
        # In split mode, each plate has 2 genotype groups (A=red, B=blue).
        # _current_group tracks the global group index across all plates.
        self._current_group = 0
        # _split_stage: 0 = first genotype, 1 = second genotype (split only)
        self._split_stage = 0
        self.canvas._current_root_group = 0
        self._enter_root_click_stage()
        self.sidebar.btn_measure.configure(state="disabled")
        self.sidebar.set_step(2)

    def _enter_root_click_stage(self):
        """Set up the canvas for the current root clicking stage."""
        plates = self.canvas.get_plates()
        pi = self._current_plate_idx
        r1, r2, c1, c2 = plates[pi]
        self.canvas._current_root_group = self._current_group
        self.canvas._current_plate_idx = self._current_plate_idx
        self.canvas.set_mode(
            ImageCanvas.MODE_CLICK_ROOTS,
            on_done=self._plate_roots_done)
        self.canvas.zoom_to_region(r1, r2, c1, c2)
        if self._split:
            genotypes = [g.strip() for g in
                         self.sidebar.entry_genotypes.get().split(",")
                         if g.strip()]
            if self._split_stage == 0:
                geno_name = genotypes[0] if genotypes else "A"
                geno_label = f"{geno_name} (red)"
            else:
                geno_name = genotypes[1] if len(genotypes) >= 2 else "B"
                geno_label = f"{geno_name} (blue)"
            self.sidebar.set_status(
                f"Plate {pi + 1}/{len(plates)} — {geno_label}\n"
                "Click root tops. Enter=next.")
        else:
            self.sidebar.set_status(
                f"Plate {pi + 1}/{len(plates)} — Click root tops.\n"
                "D+Click=dead, T+Click=touching. Enter=next.")
        self.lbl_bottom.configure(
            text="Click=root top  |  D+Click=dead  |  T+Click=touching  |  Right-click=undo  |  Enter=next  |  Scroll=zoom")

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
            self._current_group += 1
            self._enter_root_click_stage()
            return
        # advance to next plate
        self._split_stage = 0
        if self._split:
            self._current_group += 1
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
        for i, ((row, col), flag) in enumerate(zip(points, flags)):
            if flag is not None:
                continue
            if not (r1 <= row <= r2 and c1 <= col <= c2):
                continue
            # in split mode, only include roots from current group
            if self._split and i < len(groups) and groups[i] != current_group:
                continue
            self._marks_plate_roots.append(i)
        if not self._marks_plate_roots:
            self._advance_to_next_stage()
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
