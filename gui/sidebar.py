"""Sidebar — progressive, collapsible left panel."""

import customtkinter as ctk
import tkinter as tk


class _Tooltip:
    """Hover tooltip for any widget."""

    def __init__(self, widget, text):
        self._widget = widget
        self._text = text
        self._tw = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        x = self._widget.winfo_rootx() + self._widget.winfo_width() + 4
        y = self._widget.winfo_rooty()
        self._tw = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self._text, justify="left",
                         background="#333333", foreground="#e0e0e0",
                         relief="solid", borderwidth=1,
                         font=("Helvetica", 11), wraplength=250,
                         padx=8, pady=6)
        label.pack()

    def _hide(self, event=None):
        if self._tw:
            self._tw.destroy()
            self._tw = None


def _label_with_tip(parent, text, tip, **kwargs):
    """Create a row with a label and a (?) tooltip icon. Returns the frame."""
    row = ctk.CTkFrame(parent, fg_color="transparent")
    ctk.CTkLabel(row, text=text, **kwargs).pack(side="left")
    q = ctk.CTkLabel(row, text="(?)", font=ctk.CTkFont(size=10),
                     text_color="gray50", cursor="hand2")
    q.pack(side="left", padx=(4, 0))
    _Tooltip(q, tip)
    return row


class _Section:
    """Collapsible section: clickable header + toggle-able body."""

    def __init__(self, parent, title):
        self._sidebar = parent  # reference to Sidebar for ordered packing
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
        if self.frame.winfo_manager():
            return  # already packed
        sidebar = self._sidebar
        order = getattr(sidebar, '_section_order', None)
        if order is None:
            # During __init__, sections are packed in order — plain pack is fine
            self.frame.pack(fill="x")
            return
        # Find the correct 'before' widget to maintain section order
        my_idx = next((i for i, s in enumerate(order) if s is self), -1)
        if my_idx == -1:
            # Not in ordered list (e.g. sessions) — pack at the very end
            self.frame.pack(fill="x")
            return
        before = sidebar._status_frame
        for s in order[my_idx + 1:]:
            if s.frame.winfo_manager():
                before = s.frame
                break
        self.frame.pack(fill="x", before=before)

    def hide(self):
        self.frame.pack_forget()


class Sidebar(ctk.CTkScrollableFrame):
    """Left sidebar with progressive, collapsible sections."""

    def __init__(self, parent, app, **kwargs):
        # Make scrollbar subtle - blend with background until hovered
        super().__init__(parent, width=280,
                         scrollbar_button_color="gray20",
                         scrollbar_button_hover_color="gray40",
                         **kwargs)
        self.app = app

        # --- Header ---
        ctk.CTkLabel(self, text="Root Measuring Tool",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(
            pady=(15, 0), padx=15, anchor="w")
        ctk.CTkLabel(self, text="Willian Viana — Dinneny Lab",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="gray").pack(padx=15, pady=(2, 0), anchor="w")
        ctk.CTkLabel(self, text="Contact: williangviana@outlook.com",
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

        # ===== SECTION: SAVED SESSIONS =====
        self.sec_sessions = _Section(self, "SAVED SESSIONS")
        # hidden until populated
        self._session_buttons = []

        # ===== SECTION: IMAGES =====
        self.sec_images = _Section(self, "SCANNED PLATES")
        # hidden until folder loaded
        self._image_list_frame = None
        self.btn_finish_plot = ctk.CTkButton(
            self.sec_folder.body, text="Finish & Plot",
            fg_color="#2b5797",
            command=lambda: app.finish_and_plot())
        # hidden until at least one image is processed

        # ===== SECTION: IMAGE SETTINGS =====
        self.sec_settings = _Section(self, "SCAN SETTINGS")
        b = self.sec_settings.body
        _label_with_tip(b, "DPI:",
                        "Scanner resolution (dots per inch).\n"
                        "Used to convert pixels to millimeters.\n"
                        "Auto-detected from image. Edit if incorrect.",
                        font=ctk.CTkFont(size=11)).pack(padx=15, anchor="w")
        self.entry_dpi = ctk.CTkEntry(b, placeholder_text="auto-detect")
        self.entry_dpi.pack(pady=(2, 8), padx=15, fill="x")

        _label_with_tip(b, "Root thickness:",
                        "Match this to your root age/type:\n"
                        "• thick: older, thicker roots\n"
                        "• medium: typical roots (default)\n"
                        "• thin: young seedlings, fine roots",
                        font=ctk.CTkFont(size=11)).pack(padx=15, anchor="w")
        self.var_sensitivity = ctk.StringVar(value="medium")
        self.menu_sensitivity = ctk.CTkSegmentedButton(
            b, values=["thick", "medium", "thin"],
            variable=self.var_sensitivity,
            command=self._on_sensitivity_change)
        self.menu_sensitivity.pack(pady=(2, 8), padx=15, fill="x")

        _label_with_tip(b, "Root detection:",
                        "Click Preview to see detected roots.\n"
                        "Adjust slider until the full root is visible.\n"
                        "Lower = detect more (may add noise).\n"
                        "Higher = detect less (may miss thin roots).",
                        font=ctk.CTkFont(size=11)).pack(padx=15, anchor="w")
        self._thresh_frame = ctk.CTkFrame(b, fg_color="transparent")
        self._thresh_frame.pack(pady=(2, 8), padx=15, fill="x")
        self.var_auto_thresh = ctk.BooleanVar(value=True)
        self.chk_auto_thresh = ctk.CTkCheckBox(
            self._thresh_frame, text="Auto", width=60,
            variable=self.var_auto_thresh,
            command=self._toggle_threshold,
            font=ctk.CTkFont(size=11))
        self.chk_auto_thresh.pack(side="left")
        self.slider_thresh = ctk.CTkSlider(
            self._thresh_frame, from_=30, to=230, number_of_steps=200,
            width=90)
        self.slider_thresh.set(140)
        self.slider_thresh.pack(side="left", padx=(5, 5))
        self.lbl_thresh_val = ctk.CTkLabel(
            self._thresh_frame, text="140", width=30,
            font=ctk.CTkFont(size=11))
        self.lbl_thresh_val.pack(side="left")
        self.btn_preview = ctk.CTkButton(
            self._thresh_frame, text="Preview", width=60, height=24,
            fg_color="#555555",
            command=lambda: app._preview_preprocessing())
        self.btn_preview.pack(side="right")
        self.slider_thresh.configure(command=self._on_thresh_change)
        # Bind click on slider to disable auto mode
        self.slider_thresh.bind("<Button-1>", self._on_slider_click)

        self.var_multi = ctk.BooleanVar(value=False)
        _row_multi = ctk.CTkFrame(b, fg_color="transparent")
        _row_multi.pack(pady=5, padx=15, anchor="w")
        self.chk_multi = ctk.CTkCheckBox(
            _row_multi, text="Segment mode",
            variable=self.var_multi,
            command=self._toggle_segments,
            font=ctk.CTkFont(size=11))
        self.chk_multi.pack(side="left")
        _q = ctk.CTkLabel(_row_multi, text="(?)", font=ctk.CTkFont(size=10),
                          text_color="gray50", cursor="hand2")
        _q.pack(side="left", padx=(4, 0))
        _Tooltip(_q, "Measure root in segments (e.g. tip, middle, base).\n"
                     "After clicking root tops, you'll click points\n"
                     "along each root to divide it into parts.")

        self.frame_segments = ctk.CTkFrame(b, fg_color="transparent")
        _label_with_tip(self.frame_segments, "Number of segments:",
                        "How many parts to divide each root into.\n"
                        "Example: 2 segments = 1 click per root,\n"
                        "3 segments = 2 clicks per root.",
                        font=ctk.CTkFont(size=11)).pack(side="left", padx=(15, 5))
        self.entry_segments = ctk.CTkEntry(self.frame_segments, width=50,
                                            placeholder_text="2")
        self.entry_segments.pack(side="left")

        self.var_split = ctk.BooleanVar(value=False)
        _row_split = ctk.CTkFrame(b, fg_color="transparent")
        _row_split.pack(pady=5, padx=15, anchor="w")
        self.chk_split = ctk.CTkCheckBox(
            _row_split, text="Split plate",
            variable=self.var_split,
            font=ctk.CTkFont(size=11),
            width=0)
        self.chk_split.pack(side="left")
        _q2 = ctk.CTkLabel(_row_split, text="(?)", font=ctk.CTkFont(size=10),
                           text_color="gray50", cursor="hand2")
        _q2.pack(side="left", padx=(4, 0))
        _Tooltip(_q2, "Two genotypes per plate (left/right).\n"
                      "You'll click roots for each genotype\n"
                      "separately. Genotypes assigned in pairs.")

        self.btn_next_settings = ctk.CTkButton(
            b, text="Next", fg_color="#2b5797",
            command=lambda: app._on_next_settings())
        self.btn_next_settings.pack(pady=(10, 5), padx=15, fill="x")

        # ===== SECTION: EXPERIMENT =====
        self.sec_experiment = _Section(self, "EXPERIMENT")
        b = self.sec_experiment.body

        _label_with_tip(b, "Experiment name:",
                        "Name for your experiment (e.g. 'salt_stress_01').\n"
                        "Used to organize output files and sessions.",
                        font=ctk.CTkFont(size=11)).pack(padx=15, anchor="w")
        self.entry_experiment = ctk.CTkEntry(
            b, placeholder_text="e.g. salt_screen_1")
        self.entry_experiment.pack(pady=(2, 8), padx=15, fill="x")

        _label_with_tip(b, "Genotypes:",
                        "Comma-separated list (e.g. 'WT, mutant1, mutant2').\n"
                        "Assigned to plates in order (plate 1 = first genotype).\n"
                        "With split plates, list pairs: 'WT, mutant' for each plate.",
                        font=ctk.CTkFont(size=11)).pack(padx=15, anchor="w")
        self.entry_genotypes = ctk.CTkEntry(
            b, placeholder_text="e.g. WT, crd-1")
        self.entry_genotypes.pack(pady=(2, 4), padx=15, fill="x")
        ctk.CTkLabel(b, text="Comma-separated if multiple",
                     font=ctk.CTkFont(size=9),
                     text_color="gray50").pack(padx=15, anchor="w")

        _label_with_tip(b, "Conditions:",
                        "Treatment labels (e.g. 'control, salt, drought').\n"
                        "Assigned to plates in order.\n"
                        "If empty, 'Control' is used for all plates.",
                        font=ctk.CTkFont(size=11)).pack(
            padx=15, pady=(6, 0), anchor="w")
        self.entry_condition = ctk.CTkEntry(
            b, placeholder_text="e.g. Control, PEG")
        self.entry_condition.pack(pady=(2, 4), padx=15, fill="x")
        ctk.CTkLabel(b, text="Comma-separated, maps to plates in order",
                     font=ctk.CTkFont(size=9),
                     text_color="gray50").pack(padx=15, anchor="w")

        self.var_plot = ctk.BooleanVar(value=True)

        self.btn_start_workflow = ctk.CTkButton(
            b, text="Start Workflow", fg_color="#2b5797",
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
            b, text="4. Review Traces", command=app.show_review,
            state="disabled", fg_color=self._step_color_idle)
        self.btn_review.pack(pady=3, padx=15, fill="x")

        # --- Ordered section list (for pack-order preservation) ---
        # sec_sessions is excluded — it always packs after _status_frame
        self._section_order = [
            self.sec_folder, self.sec_images,
            self.sec_settings, self.sec_experiment, self.sec_workflow,
        ]

        # --- Status area (always visible at bottom of sidebar) ---
        self._status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._status_frame.pack(fill="x", pady=(15, 5))

        self.lbl_status = ctk.CTkLabel(
            self._status_frame, text="Open a folder containing the scanned plates.",
            font=ctk.CTkFont(size=11),
            text_color="gray", wraplength=250)
        self.lbl_status.pack(pady=(0, 5), padx=15, anchor="w")

        # Done button (hidden by default, shown during interactive steps)
        self.btn_done = ctk.CTkButton(
            self._status_frame, text="Done", fg_color="#2b5797",
            command=lambda: app.canvas._trigger_done())
        # starts hidden

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

    def _on_sensitivity_change(self, val):
        """Update auto threshold when sensitivity changes."""
        if self.var_auto_thresh.get():
            self.app._update_auto_threshold()
        # Update preview if active
        if getattr(self.app, '_preview_active', False):
            self.app._preview_preprocessing(force_show=True)

    def _toggle_threshold(self):
        if self.var_auto_thresh.get():
            self.lbl_thresh_val.configure(text_color="gray50")
            # Show auto-detected value
            self.app._update_auto_threshold()
            # Update preview if active
            if getattr(self.app, '_preview_active', False):
                self.app._preview_preprocessing(force_show=True)
        else:
            self.lbl_thresh_val.configure(text_color=("gray10", "gray90"))

    def _on_slider_click(self, event):
        """Disable auto mode when user clicks on slider."""
        if self.var_auto_thresh.get():
            self.var_auto_thresh.set(False)
            self.lbl_thresh_val.configure(text_color=("gray10", "gray90"))

    def _on_thresh_change(self, val):
        self.lbl_thresh_val.configure(text=str(int(val)))
        # Auto-update preview when slider changes
        if getattr(self.app, '_preview_active', False):
            self.app._preview_preprocessing(force_show=True)

    def get_threshold(self):
        """Return threshold value or None for auto-detect."""
        if self.var_auto_thresh.get():
            return None
        return int(self.slider_thresh.get())

    def set_auto_threshold_value(self, val):
        """Update slider and label to show auto-detected threshold."""
        self.slider_thresh.set(val)
        self.lbl_thresh_val.configure(text=str(int(val)))

    def set_status(self, text):
        self.lbl_status.configure(text=text)

    def set_step(self, step):
        """Highlight the current workflow step (1-4). Previous steps turn green."""
        buttons = [
            self.btn_select_plates, self.btn_click_roots,
            self.btn_measure, self.btn_review,
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

    def populate_sessions(self, sessions):
        """Show saved session entries. sessions = list of summary dicts."""
        # clear old buttons
        for btn in self._session_buttons:
            btn.destroy()
        self._session_buttons.clear()

        if not sessions:
            self.sec_sessions.hide()
            return

        b = self.sec_sessions.body
        for s in sessions:
            title = s['experiment'] or s['folder_name']
            progress = f"{s['n_done']}/{s['n_total']} scans done"
            if s.get('current_image'):
                progress += f"  •  {s['current_image']}"
            frame = ctk.CTkFrame(b, fg_color="gray20", corner_radius=6,
                                 cursor="hand2")
            frame.pack(fill="x", padx=15, pady=3)
            ctk.CTkLabel(frame, text=title,
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color="white").pack(padx=10, pady=(6, 0),
                                                   anchor="w")
            ctk.CTkLabel(frame, text=progress,
                         font=ctk.CTkFont(size=10),
                         text_color="gray60").pack(padx=10, pady=(0, 6),
                                                    anchor="w")
            folder = s['folder']
            exp = s.get('experiment', '')
            for w in (frame, *frame.winfo_children()):
                w.bind("<Button-1>",
                       lambda e, f=folder, x=exp: self.app.resume_session(f, x))
            self._session_buttons.append(frame)

        self.sec_sessions.show()
        self.sec_sessions.expand()

    # --- phase transitions ---

    def _populate_image_list(self, images, processed=None):
        """Build the image list inside the folder section body."""
        if processed is None:
            processed = set()
        if self._image_list_frame is not None:
            self._image_list_frame.destroy()
        self._image_list_frame = ctk.CTkFrame(
            self.sec_folder.body, fg_color="transparent")
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

    def advance_to_images(self, folder_name, images, processed=None):
        """Phase 1: folder loaded — show images in folder section."""
        if processed is None:
            processed = set()
        self.btn_load_folder.pack_forget()
        self._populate_image_list(images, processed)
        self.sec_folder.expand()
        self.sec_folder._summary.configure(text=folder_name)
        # hide later sections
        self.sec_images.hide()
        self.sec_settings.hide()
        self.sec_experiment.hide()
        self.sec_workflow.hide()
        self.hide_action_buttons()
        n_done = len(processed)
        n_total = len(images)
        if n_done > 0:
            self.set_status(f"{n_done}/{n_total} scan(s) done. Select next image.")
        else:
            self.set_status(f"{n_total} scan(s) found.")

    def advance_to_settings(self, image_name, dpi):
        """Phase 2: image selected — show settings, collapse folder."""
        self.btn_finish_plot.pack_forget()
        self.sec_sessions.hide()
        self.sec_folder.collapse(summary=image_name)
        self.entry_dpi.delete(0, "end")
        self.entry_dpi.insert(0, str(dpi))
        self.sec_settings.show()
        self.sec_settings.expand()
        # hide later sections
        self.sec_images.hide()
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
        self.set_step(1)
        self.set_status("Ready. Select plates to begin.")
