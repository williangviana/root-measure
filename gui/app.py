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

from config import DISPLAY_DOWNSAMPLE
from plate_detection import _to_uint8

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ImageCanvas(ctk.CTkFrame):
    """Zoomable, pannable image canvas using tkinter Canvas."""

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

        # bindings
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<MouseWheel>", self._on_scroll)

    def set_image(self, img_np):
        """Set image from numpy array (grayscale or RGB uint8)."""
        self._image_np = img_np
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
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

    def _redraw(self):
        """Redraw image on canvas at current zoom/pan."""
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

    def canvas_to_image(self, cx, cy):
        """Convert canvas coords to image pixel coords."""
        ix = (cx - self._offset_x) / self._scale
        iy = (cy - self._offset_y) / self._scale
        return int(ix), int(iy)

    def image_to_canvas(self, ix, iy):
        """Convert image pixel coords to canvas coords."""
        cx = ix * self._scale + self._offset_x
        cy = iy * self._scale + self._offset_y
        return cx, cy


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

    def select_plates(self):
        self.sidebar.set_status("Plate selection — coming soon")

    def click_roots(self):
        self.sidebar.set_status("Root clicking — coming soon")

    def measure(self):
        self.sidebar.set_status("Measurement — coming soon")


def main():
    app = RootMeasureApp()
    app.mainloop()


if __name__ == '__main__':
    main()
