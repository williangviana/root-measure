"""Sidebar — left panel with settings, workflow buttons, and image list."""

import customtkinter as ctk


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
            self, text="2. Select Roots", command=app.click_roots,
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
