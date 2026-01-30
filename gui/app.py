#!/usr/bin/env python3
"""Root Measure GUI — CustomTkinter app shell."""

import sys
from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageTk
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

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ImageCanvas(ctk.CTkFrame):
    """Zoomable, pannable image canvas with overlay drawing."""

    # Interaction modes
    MODE_VIEW = "view"
    MODE_SELECT_PLATES = "select_plates"
    MODE_CLICK_ROOTS = "click_roots"

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
        self._root_marker_ids = [] # canvas ids of markers
        self._pending_flag = None  # 'dead' or 'touching'

        # trace overlay state
        self._traces = []          # list of (path_array, color_str)

        # bindings
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<MouseWheel>", self._on_scroll)
        # keyboard (canvas needs focus)
        self.canvas.bind("<Key>", self._on_key)
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
        self._root_marker_ids.clear()
        self._pending_flag = None

    def clear_traces(self):
        self._traces.clear()

    def add_trace(self, path, color="#00ff88"):
        """Add a traced path (N,2 array of row,col) to draw on canvas."""
        self._traces.append((path, color))

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

        # redraw root markers
        self._root_marker_ids.clear()
        for i, ((row, col), flag) in enumerate(
                zip(self._root_points, self._root_flags)):
            cx, cy = self.image_to_canvas(col, row)
            if flag is not None:
                # dead/touching: X marker
                color = "#ff6b6b" if flag == 'dead' else "#ffa500"
                label = "DEAD" if flag == 'dead' else "TOUCH"
                s = 6
                id1 = self.canvas.create_line(
                    cx - s, cy - s, cx + s, cy + s,
                    fill=color, width=2)
                id2 = self.canvas.create_line(
                    cx - s, cy + s, cx + s, cy - s,
                    fill=color, width=2)
                id3 = self.canvas.create_text(
                    cx + 10, cy - 10, text=f"{i + 1} {label}",
                    fill=color, anchor="w",
                    font=("Helvetica", 9, "bold"))
                self._root_marker_ids.extend([id1, id2, id3])
            else:
                # normal: circle
                r = 5
                rid = self.canvas.create_oval(
                    cx - r, cy - r, cx + r, cy + r,
                    outline="white", fill="#ff3b3b", width=1)
                tid = self.canvas.create_text(
                    cx + 10, cy - 10, text=str(i + 1),
                    fill="#ff3b3b", anchor="w",
                    font=("Helvetica", 9, "bold"))
                self._root_marker_ids.extend([rid, tid])

        # redraw traced paths
        for path, color in self._traces:
            if len(path) < 2:
                continue
            # subsample for performance (draw every Nth point)
            step = max(1, len(path) // 500)
            coords = []
            for row, col in path[::step]:
                cx, cy = self.image_to_canvas(col, row)
                coords.extend([cx, cy])
            if len(coords) >= 4:
                self.canvas.create_line(
                    *coords, fill=color, width=2, smooth=True)

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
            self._redraw()
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
        """Right-click: undo last action in current mode, or start pan."""
        if self._mode == self.MODE_SELECT_PLATES and self._plates:
            self._plates.pop()
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()
        elif self._mode == self.MODE_CLICK_ROOTS and self._root_points:
            self._root_points.pop()
            self._root_flags.pop()
            self._redraw()
            if self._on_click_callback:
                self._on_click_callback()
        else:
            self._on_pan_start(event)

    def _on_key(self, event):
        if self._mode == self.MODE_CLICK_ROOTS:
            if event.keysym.lower() == 'd':
                self._pending_flag = 'dead'
                return
            elif event.keysym.lower() == 't':
                self._pending_flag = 'touching'
                return
        if event.keysym == 'Return':
            if self._mode == self.MODE_SELECT_PLATES:
                # Enter confirms drawn plate; second Enter with no new plate finishes
                if len(self._plates) > self._plates_count_at_enter:
                    # new plate(s) since last Enter — confirm and keep going
                    self._plates_count_at_enter = len(self._plates)
                    return
                # no new plate — finish selection
            if self._on_done_callback:
                self._on_done_callback()


class Sidebar(ctk.CTkScrollableFrame):
    """Left sidebar with settings and controls."""

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, width=280, **kwargs)
        self.app = app

        # --- Header ---
        ctk.CTkLabel(self, text="Root Measure",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(
            pady=(15, 2), padx=15, anchor="w")
        ctk.CTkLabel(self, text="Dinneny Lab",
                     font=ctk.CTkFont(size=12),
                     text_color="gray").pack(padx=15, anchor="w")

        self._add_separator()

        # --- Image Loading ---
        ctk.CTkLabel(self, text="IMAGE",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#4a9eff").pack(pady=(10, 5), padx=15, anchor="w")

        self.btn_load_folder = ctk.CTkButton(
            self, text="Open Folder", command=app.load_folder)
        self.btn_load_folder.pack(pady=5, padx=15, fill="x")

        self.lbl_image_name = ctk.CTkLabel(self, text="No image loaded",
                                            text_color="gray",
                                            font=ctk.CTkFont(size=11))
        self.lbl_image_name.pack(padx=15, anchor="w")

        # image list (populated after folder load)
        self.image_listbox = None

        self._add_separator()

        # --- Settings ---
        ctk.CTkLabel(self, text="SETTINGS",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#4a9eff").pack(pady=(10, 5), padx=15, anchor="w")

        # Experiment name
        ctk.CTkLabel(self, text="Experiment:",
                     font=ctk.CTkFont(size=11)).pack(padx=15, anchor="w")
        self.entry_experiment = ctk.CTkEntry(self, placeholder_text="e.g. WT vs crd-1")
        self.entry_experiment.pack(pady=(2, 8), padx=15, fill="x")

        # DPI
        ctk.CTkLabel(self, text="DPI:",
                     font=ctk.CTkFont(size=11)).pack(padx=15, anchor="w")
        self.entry_dpi = ctk.CTkEntry(self, placeholder_text="auto-detect")
        self.entry_dpi.pack(pady=(2, 8), padx=15, fill="x")

        # Sensitivity
        ctk.CTkLabel(self, text="Root thickness:",
                     font=ctk.CTkFont(size=11)).pack(padx=15, anchor="w")
        self.var_sensitivity = ctk.StringVar(value="medium")
        self.menu_sensitivity = ctk.CTkSegmentedButton(
            self, values=["thick", "medium", "thin"],
            variable=self.var_sensitivity)
        self.menu_sensitivity.pack(pady=(2, 8), padx=15, fill="x")

        # Split plate
        self.var_split = ctk.BooleanVar(value=False)
        self.chk_split = ctk.CTkCheckBox(
            self, text="Split plate (2 genotypes)",
            variable=self.var_split,
            font=ctk.CTkFont(size=11))
        self.chk_split.pack(pady=5, padx=15, anchor="w")

        # Multi-measurement
        self.var_multi = ctk.BooleanVar(value=False)
        self.chk_multi = ctk.CTkCheckBox(
            self, text="Multi-measurement",
            variable=self.var_multi,
            command=self._toggle_segments,
            font=ctk.CTkFont(size=11))
        self.chk_multi.pack(pady=5, padx=15, anchor="w")

        self.frame_segments = ctk.CTkFrame(self, fg_color="transparent")
        ctk.CTkLabel(self.frame_segments, text="Segments per root:",
                     font=ctk.CTkFont(size=11)).pack(side="left", padx=(15, 5))
        self.entry_segments = ctk.CTkEntry(self.frame_segments, width=50,
                                            placeholder_text="2")
        self.entry_segments.pack(side="left")
        # hidden by default
        self.frame_segments.pack_forget()

        self._add_separator()

        # --- Actions ---
        ctk.CTkLabel(self, text="WORKFLOW",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#4a9eff").pack(pady=(10, 5), padx=15, anchor="w")

        self.btn_select_plates = ctk.CTkButton(
            self, text="1. Select Plates", command=app.select_plates,
            state="disabled", fg_color="#2b5797")
        self.btn_select_plates.pack(pady=3, padx=15, fill="x")

        self.btn_click_roots = ctk.CTkButton(
            self, text="2. Click Root Tops", command=app.click_roots,
            state="disabled", fg_color="#2b5797")
        self.btn_click_roots.pack(pady=3, padx=15, fill="x")

        self.btn_measure = ctk.CTkButton(
            self, text="3. Measure & Save", command=app.measure,
            state="disabled", fg_color="#217346")
        self.btn_measure.pack(pady=3, padx=15, fill="x")

        self._add_separator()

        # --- Status ---
        self.lbl_status = ctk.CTkLabel(
            self, text="Ready",
            font=ctk.CTkFont(size=11),
            text_color="gray", wraplength=250)
        self.lbl_status.pack(pady=10, padx=15, anchor="w")

    def _add_separator(self):
        sep = ctk.CTkFrame(self, height=1, fg_color="gray30")
        sep.pack(fill="x", padx=15, pady=8)

    def _toggle_segments(self):
        if self.var_multi.get():
            self.frame_segments.pack(pady=(0, 8), fill="x")
        else:
            self.frame_segments.pack_forget()

    def set_status(self, text):
        self.lbl_status.configure(text=text)

    def show_image_list(self, images):
        """Show scrollable list of images in sidebar."""
        if self.image_listbox is not None:
            self.image_listbox.destroy()

        self.image_listbox = ctk.CTkFrame(self, fg_color="transparent")
        self.image_listbox.pack(fill="x", padx=10, pady=5)

        for i, img_path in enumerate(images):
            btn = ctk.CTkButton(
                self.image_listbox,
                text=img_path.name,
                font=ctk.CTkFont(size=11),
                height=28,
                fg_color="transparent",
                text_color="white",
                hover_color="gray30",
                anchor="w",
                command=lambda p=img_path: self.app.load_image(p))
            btn.pack(fill="x", pady=1)


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
            self.sidebar.set_status(
                f"Loaded: {img.shape[1]}x{img.shape[0]}, {img.dtype}")
        except Exception as e:
            self.sidebar.set_status(f"Error: {e}")

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
            from measure_roots import _detect_dpi
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
            text="PLATE SELECTION — drag to draw, Enter=confirm, Enter again=done")
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
        self.canvas.clear_traces()
        self._current_plate_idx = 0
        self.canvas.set_mode(
            ImageCanvas.MODE_CLICK_ROOTS,
            on_done=self._plate_roots_done)
        # zoom into first plate
        r1, r2, c1, c2 = plates[0]
        self.canvas.zoom_to_region(r1, r2, c1, c2)
        self.sidebar.set_status(
            f"Plate 1/{len(plates)} — Click root tops.\n"
            "D+Click=dead, T+Click=touching. Enter=next plate.")
        self.lbl_bottom.configure(
            text="ROOT CLICKING — Click=root top, D+Click=dead, "
                 "T+Click=touching, Right-click=undo, Enter=next")
        self.sidebar.btn_measure.configure(state="disabled")

    def _plate_roots_done(self):
        """Called when user presses Enter after clicking roots on a plate."""
        plates = self.canvas.get_plates()
        self._current_plate_idx += 1
        if self._current_plate_idx < len(plates):
            # advance to next plate
            r1, r2, c1, c2 = plates[self._current_plate_idx]
            self.canvas.zoom_to_region(r1, r2, c1, c2)
            pi = self._current_plate_idx + 1
            self.sidebar.set_status(
                f"Plate {pi}/{len(plates)} — Click root tops.\n"
                "D+Click=dead, T+Click=touching. Enter=next.")
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

    def measure(self):
        """Run preprocessing, tracing, and display results."""
        points = self.canvas.get_root_points()
        flags = self.canvas.get_root_flags()
        plates = self.canvas.get_plates()

        if not points or self.image is None:
            self.sidebar.set_status("Nothing to measure.")
            return

        scale = self._get_scale()
        sensitivity = self.sidebar.var_sensitivity.get()

        self.sidebar.set_status("Preprocessing...")
        self.sidebar.btn_measure.configure(state="disabled")
        self.update_idletasks()

        binary = preprocess(self.image, scale=scale, sensitivity=sensitivity)

        self.canvas.clear_traces()
        results = []
        for i, (top, flag) in enumerate(zip(points, flags)):
            self.sidebar.set_status(f"Tracing root {i + 1}/{len(points)}...")
            self.update_idletasks()

            if flag is not None:
                warning = 'dead seedling' if flag == 'dead' else 'roots touching'
                res = dict(length_cm=None, length_px=None,
                           path=np.empty((0, 2)),
                           method='skip', warning=warning, segments=[])
                results.append(res)
                continue

            tip = find_root_tip(binary, top, scale=scale)
            if tip is None:
                res = dict(length_cm=0, length_px=0,
                           path=np.empty((0, 2)),
                           method='error', warning='Could not find root tip',
                           segments=[])
                results.append(res)
                continue

            res = trace_root(binary, top, tip, scale)
            res['segments'] = []
            results.append(res)

            # draw traced path on canvas
            if res['path'].size > 0:
                color = "#00ff88" if res['warning'] is None else "#ffaa00"
                self.canvas.add_trace(res['path'], color)

        self.canvas._redraw()

        # summary
        traced = [r for r in results if r['method'] not in ('skip', 'error')]
        lengths = [r['length_cm'] for r in traced if r['length_cm']]
        msg = f"Done! {len(traced)} root(s) traced."
        if lengths:
            msg += f"\nMean: {np.mean(lengths):.2f} cm, "
            msg += f"Range: {min(lengths):.2f}–{max(lengths):.2f} cm"
        self.sidebar.set_status(msg)
        self.lbl_bottom.configure(
            text=f"Traced {len(traced)}/{len(points)} roots")

        # save CSV
        self._save_results(results, plates, scale)
        self.sidebar.btn_measure.configure(state="normal")

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
            num_marks=0,
            split_plate=self.sidebar.var_split.get())

        self.sidebar.set_status(
            self.sidebar.lbl_status.cget("text") +
            f"\nSaved to {csv_path.name}")


def main():
    app = RootMeasureApp()
    app.mainloop()


if __name__ == '__main__':
    main()
