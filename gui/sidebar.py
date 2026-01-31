"""Sidebar — progressive, collapsible left panel."""

import customtkinter as ctk


class _Section:
    """Collapsible section: clickable header + toggle-able body."""

    def __init__(self, parent, title):
        self.frame = ctk.CTkFrame(parent, fg_color="transparent")
        # header row
        self._header = ctk.CTkFrame(self.frame, fg_color="transparent",
                                    cursor="hand2")
        self._header.pack(fill="x", padx=15, pady=(8, 0))
        self._arrow = ctk.CTkLabel(self._header, text="\u25bc",
                                   font=ctk.CTkFont(size=10),
                                   text_color="#4a9eff", width=14)
        self._arrow.pack(side="left")
        self._title = ctk.CTkLabel(self._header, text=title,
                                   font=ctk.CTkFont(size=12, weight="bold"),
                                   text_color="#4a9eff")
        self._title.pack(side="left", padx=(2, 0))
        self._summary = ctk.CTkLabel(self._header, text="",
                                     font=ctk.CTkFont(size=10),
                                     text_color="gray50")
        self._summary.pack(side="left", padx=(8, 0))
        # body
        self.body = ctk.CTkFrame(self.frame, fg_color="transparent")
        self.body.pack(fill="x")
        self._expanded = True
        # click header to toggle
        for w in (self._header, self._arrow, self._title, self._summary):
            w.bind("<Button-1>", lambda e: self.toggle())

    def toggle(self):
        if self._expanded:
            self.collapse()
        else:
            self.expand()

    def expand(self):
        self._expanded = True
        self.body.pack(fill="x")
        self._arrow.configure(text="\u25bc")
        self._summary.configure(text="")

    def collapse(self, summary=""):
        self._expanded = False
        self.body.pack_forget()
        self._arrow.configure(text="\u25b6")
        if summary:
            self._summary.configure(text=summary)

    def show(self):
        self.frame.pack(fill="x")

    def hide(self):
        self.frame.pack_forget()


class Sidebar(ctk.CTkScrollableFrame):
    """Left sidebar with progressive, collapsible sections."""

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, width=280, **kwargs)
        self.app = app

        # --- Header ---
        ctk.CTkLabel(self, text="Root Measuring Tool",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(
            pady=(15, 0), padx=15, anchor="w")
        ctk.CTkLabel(self, text="Willian Viana — Dinneny Lab",
                     font=ctk.CTkFont(size=11),
                     text_color="gray").pack(padx=15, pady=(2, 0), anchor="w")
        ctk.CTkLabel(self, text="williangviana@outlook.com",
                     font=ctk.CTkFont(size=11),
                     text_color="gray50").pack(padx=15, anchor="w")

        self._add_separator()

        # ===== SECTION: FOLDER =====
        self.sec_folder = _Section(self, "FOLDER")
        self.sec_folder.show()
        b = self.sec_folder.body
        self.btn_load_folder = ctk.CTkButton(
            b, text="Open Folder", command=app.load_folder)
        self.btn_load_folder.pack(pady=5, padx=15, fill="x")

        # ===== SECTION: IMAGES =====
        self.sec_images = _Section(self, "SCANNED PLATES")
        # hidden until folder loaded
        self._image_list_frame = None
        self.btn_finish_plot = ctk.CTkButton(
            self.sec_images.body, text="Finish & Plot",
            fg_color="#217346",
            command=lambda: app.finish_and_plot())
        # hidden until at least one image is processed

        # ===== SECTION: IMAGE SETTINGS =====
        self.sec_settings = _Section(self, "SCAN SETTINGS")
        b = self.sec_settings.body
        ctk.CTkLabel(b, text="DPI:",
                     font=ctk.CTkFont(size=11)).pack(padx=15, anchor="w")
        self.entry_dpi = ctk.CTkEntry(b, placeholder_text="auto-detect")
        self.entry_dpi.pack(pady=(2, 8), padx=15, fill="x")

        ctk.CTkLabel(b, text="Root thickness:",
                     font=ctk.CTkFont(size=11)).pack(padx=15, anchor="w")
        self.var_sensitivity = ctk.StringVar(value="medium")
        self.menu_sensitivity = ctk.CTkSegmentedButton(
            b, values=["thick", "medium", "thin"],
            variable=self.var_sensitivity)
        self.menu_sensitivity.pack(pady=(2, 8), padx=15, fill="x")

        self.var_multi = ctk.BooleanVar(value=False)
        self.chk_multi = ctk.CTkCheckBox(
            b, text="Multi-measurement",
            variable=self.var_multi,
            command=self._toggle_segments,
            font=ctk.CTkFont(size=11))
        self.chk_multi.pack(pady=5, padx=15, anchor="w")

        self.frame_segments = ctk.CTkFrame(b, fg_color="transparent")
        ctk.CTkLabel(self.frame_segments, text="Segments per root:",
                     font=ctk.CTkFont(size=11)).pack(side="left", padx=(15, 5))
        self.entry_segments = ctk.CTkEntry(self.frame_segments, width=50,
                                            placeholder_text="2")
        self.entry_segments.pack(side="left")

        self.var_split = ctk.BooleanVar(value=False)
        self.chk_split = ctk.CTkCheckBox(
            b, text="Split plate (2 genotypes)",
            variable=self.var_split,
            font=ctk.CTkFont(size=11))
        self.chk_split.pack(pady=5, padx=15, anchor="w")

        self.btn_next_settings = ctk.CTkButton(
            b, text="Next \u00bb", fg_color="#2b5797",
            command=lambda: app._on_next_settings())
        self.btn_next_settings.pack(pady=(10, 5), padx=15, fill="x")

        # ===== SECTION: EXPERIMENT =====
        self.sec_experiment = _Section(self, "EXPERIMENT")
        b = self.sec_experiment.body

        ctk.CTkLabel(b, text="Experiment:",
                     font=ctk.CTkFont(size=11)).pack(padx=15, anchor="w")
        self.entry_experiment = ctk.CTkEntry(
            b, placeholder_text="e.g. salt_screen_1")
        self.entry_experiment.pack(pady=(2, 8), padx=15, fill="x")

        ctk.CTkLabel(b, text="Genotypes:",
                     font=ctk.CTkFont(size=11)).pack(padx=15, anchor="w")
        self.entry_genotypes = ctk.CTkEntry(
            b, placeholder_text="e.g. WT, crd-1")
        self.entry_genotypes.pack(pady=(2, 4), padx=15, fill="x")
        ctk.CTkLabel(b, text="Comma-separated if multiple",
                     font=ctk.CTkFont(size=9),
                     text_color="gray50").pack(padx=15, anchor="w")

        ctk.CTkLabel(b, text="Condition:",
                     font=ctk.CTkFont(size=11)).pack(
            padx=15, pady=(6, 0), anchor="w")
        self.entry_condition = ctk.CTkEntry(
            b, placeholder_text="e.g. Control, PEG")
        self.entry_condition.pack(pady=(2, 4), padx=15, fill="x")
        ctk.CTkLabel(b, text="Comma-separated, maps to plates in order",
                     font=ctk.CTkFont(size=9),
                     text_color="gray50").pack(padx=15, anchor="w")

        ctk.CTkLabel(b, text="CSV format:",
                     font=ctk.CTkFont(size=11)).pack(
            padx=15, pady=(8, 0), anchor="w")
        self.var_csv_format = ctk.StringVar(value="R")
        self.menu_csv_format = ctk.CTkSegmentedButton(
            b, values=["R", "Prism"],
            variable=self.var_csv_format)
        self.menu_csv_format.pack(pady=(2, 8), padx=15, fill="x")

        self.var_plot = ctk.BooleanVar(value=True)
        self.chk_plot = ctk.CTkCheckBox(
            b, text="Plot with statistics when done",
            variable=self.var_plot,
            font=ctk.CTkFont(size=11))
        self.chk_plot.pack(pady=(8, 2), padx=15, anchor="w")

        self.btn_start_workflow = ctk.CTkButton(
            b, text="Start Workflow \u00bb", fg_color="#2b5797",
            command=lambda: app._on_start_workflow())
        self.btn_start_workflow.pack(pady=(10, 5), padx=15, fill="x")

        # ===== SECTION: WORKFLOW =====
        self.sec_workflow = _Section(self, "WORKFLOW")
        b = self.sec_workflow.body

        # step button style constants
        self._step_color_idle = "#3a3a3a"
        self._step_color_active = "#2b5797"
        self._step_color_done = "#217346"

        self.btn_select_plates = ctk.CTkButton(
            b, text="1. Select Plates", command=app.select_plates,
            state="disabled", fg_color=self._step_color_idle)
        self.btn_select_plates.pack(pady=3, padx=15, fill="x")

        self.btn_click_roots = ctk.CTkButton(
            b, text="2. Click Roots", command=app.click_roots,
            state="disabled", fg_color=self._step_color_idle)
        self.btn_click_roots.pack(pady=3, padx=15, fill="x")

        self.btn_measure = ctk.CTkButton(
            b, text="3. Trace", command=app.measure,
            state="disabled", fg_color=self._step_color_idle)
        self.btn_measure.pack(pady=3, padx=15, fill="x")

        self.btn_review = ctk.CTkButton(
            b, text="4. Review", command=None,
            state="disabled", fg_color=self._step_color_idle)
        self.btn_review.pack(pady=3, padx=15, fill="x")

        self.btn_save = ctk.CTkButton(
            b, text="5. Save & Plot", command=None,
            state="disabled", fg_color=self._step_color_idle)
        self.btn_save.pack(pady=3, padx=15, fill="x")

        self.btn_next_image = ctk.CTkButton(
            b, text="Next Image \u00bb", fg_color="#2b5797",
            command=app.next_image)
        # hidden until measurement finishes

        # --- Status area (always visible, below workflow) ---
        self._status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._status_frame.pack(fill="x", pady=(15, 5))

        self.lbl_status = ctk.CTkLabel(
            self._status_frame, text="Open a folder containing the scanned plates.",
            font=ctk.CTkFont(size=11),
            text_color="gray", wraplength=250)
        self.lbl_status.pack(pady=(0, 5), padx=15, anchor="w")

        # progress bar inside status area
        self._progress_frame = ctk.CTkFrame(self._status_frame,
                                             fg_color="transparent")
        self.progress_bar = ctk.CTkProgressBar(self._progress_frame, width=250)
        self.progress_bar.set(0)
        self.progress_bar.pack(padx=15, pady=(0, 2), fill="x")
        self.lbl_progress = ctk.CTkLabel(
            self._progress_frame, text="",
            font=ctk.CTkFont(size=10), text_color="gray50")
        self.lbl_progress.pack(padx=15, anchor="w")
        self._progress_frame.pack(fill="x")
        self._progress_frame.pack_forget()

    # --- helpers ---

    def _add_separator(self):
        ctk.CTkFrame(self, height=1, fg_color="gray30").pack(
            fill="x", padx=15, pady=8)

    def _toggle_segments(self):
        if self.var_multi.get():
            self.frame_segments.pack(pady=(0, 8), fill="x")
        else:
            self.frame_segments.pack_forget()

    def set_status(self, text):
        self.lbl_status.configure(text=text)
        self._status_frame.pack_forget()
        self._status_frame.pack(fill="x", pady=(15, 5))

    def set_step(self, step):
        """Highlight the current workflow step (1-5). Previous steps turn green."""
        buttons = [
            self.btn_select_plates, self.btn_click_roots,
            self.btn_measure, self.btn_review, self.btn_save,
        ]
        for i, btn in enumerate(buttons):
            num = i + 1
            if num < step:
                btn.configure(fg_color=self._step_color_done)
            elif num == step:
                btn.configure(fg_color=self._step_color_active)
            else:
                btn.configure(fg_color=self._step_color_idle)

    def show_progress(self, total):
        self._progress_total = total
        self.progress_bar.set(0)
        self.lbl_progress.configure(text=f"0 / {total}")
        self._progress_frame.pack(fill="x")

    def update_progress(self, current):
        frac = current / self._progress_total
        self.progress_bar.set(frac)
        self.lbl_progress.configure(
            text=f"{current} / {self._progress_total}")

    def hide_progress(self):
        self._progress_frame.pack_forget()

    # --- phase transitions ---

    def advance_to_images(self, folder_name, images, processed=None):
        """Phase 1: folder loaded — show images, collapse folder."""
        if processed is None:
            processed = set()
        self.sec_folder.collapse(summary=folder_name)
        # populate image list
        if self._image_list_frame is not None:
            self._image_list_frame.destroy()
        self._image_list_frame = ctk.CTkFrame(
            self.sec_images.body, fg_color="transparent")
        self._image_list_frame.pack(fill="x", padx=10, pady=5)
        for img_path in images:
            done = img_path in processed
            label = f"\u2713  {img_path.name}" if done else img_path.name
            text_color = "#217346" if done else "white"
            btn = ctk.CTkButton(
                self._image_list_frame,
                text=label,
                font=ctk.CTkFont(size=11),
                height=28,
                fg_color="transparent",
                text_color=text_color,
                hover_color="gray30",
                anchor="w",
                command=lambda p=img_path: self.app.load_image(p))
            btn.pack(fill="x", pady=1)
        self.btn_finish_plot.pack_forget()
        if len(processed) > 0:
            self.btn_finish_plot.pack(pady=(8, 5), padx=10, fill="x")
        self.sec_images.show()
        self.sec_images.expand()
        # hide later sections
        self.sec_settings.hide()
        self.sec_experiment.hide()
        self.sec_workflow.hide()
        self.btn_next_image.pack_forget()
        n_done = len(processed)
        n_total = len(images)
        if n_done > 0:
            self.set_status(f"{n_done}/{n_total} scan(s) done. Select next image.")
        else:
            self.set_status(f"{n_total} scan(s) found.")

    def advance_to_settings(self, image_name, dpi):
        """Phase 2: image selected — show settings, collapse images."""
        self.sec_images.collapse(summary=image_name)
        self.entry_dpi.delete(0, "end")
        self.entry_dpi.insert(0, str(dpi))
        self.sec_settings.show()
        self.sec_settings.expand()
        # hide later sections
        self.sec_experiment.hide()
        self.sec_workflow.hide()

    def advance_to_experiment(self):
        """Phase 3: settings confirmed — show experiment, collapse settings."""
        dpi = self.entry_dpi.get().strip() or "auto"
        sens = self.var_sensitivity.get()
        parts = [f"{dpi} DPI", sens]
        if self.var_split.get():
            parts.append("split")
        if self.var_multi.get():
            segs = self.entry_segments.get().strip() or "2"
            parts.append(f"{segs} seg")
        self.sec_settings.collapse(summary=", ".join(parts))
        self.sec_experiment.show()
        self.sec_experiment.expand()
        # hide workflow
        self.sec_workflow.hide()

    def advance_to_workflow(self):
        """Phase 4: experiment configured — show workflow, collapse experiment."""
        genos = self.entry_genotypes.get().strip() or "genotype"
        cond = self.entry_condition.get().strip()
        summary = genos
        if cond:
            summary += f" | {cond}"
        self.sec_experiment.collapse(summary=summary)
        self.sec_workflow.show()
        self.sec_workflow.expand()
        self.btn_select_plates.configure(state="normal")
        self.btn_click_roots.configure(state="disabled")
        self.btn_measure.configure(state="disabled")
        self.btn_review.configure(state="disabled")
        self.btn_save.configure(state="disabled")
        self.set_step(1)
        self.set_status("Ready. Select plates to begin.")
