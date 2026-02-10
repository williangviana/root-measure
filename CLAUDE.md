# Project: Root Measure

Mac desktop app for measuring root lengths from scanned agar plate images.

**Stack:** Python, CustomTkinter, OpenCV, scikit-image, networkx, pandas, scipy, matplotlib (plotting only)

## Token Efficiency Rules

1. **Only read files relevant to the task.** Never read the entire codebase.
2. **One task at a time.** Don't touch unrelated code.
3. **Targeted edits only.** Use the Edit tool on specific lines. Never rewrite whole files or rewrite functions that aren't changing.
4. **No unsolicited changes.** No extra comments, docstrings, type hints, refactors, or error handling beyond what was asked.
5. **Don't echo code back.** State what changed and where (file:line_number), don't paste the modified code.
6. **Keep responses short.** No lengthy explanations unless asked.

## Git Workflow

- Auto-commit and push after completing each task.
- Use short, descriptive commit messages.
- One logical change per commit — don't bundle unrelated changes.
- Never copy files as backups — git is the backup system.

## Pipeline-First Thinking (CRITICAL)

**Before writing any code, trace the full data flow end-to-end.** Every change touches a pipeline:

```
User input → sidebar fields → app state → canvas drawing → tracing →
results list → CSV build → raw_data.csv → tidy_data.csv → plot →
session save → session restore → review/re-click → re-save CSV
```

For ANY change, answer these before coding:
1. **Where does this data originate?** (sidebar field, click event, config, session file)
2. **Where is it stored?** (app attr, canvas dict, _image_canvas_data, session JSON)
3. **Where is it consumed?** (tracing, CSV rows, plot labels, status messages, screenshots)
4. **What happens on re-save/review?** (rebuilt from traces? from session? from CSV?)
5. **What happens on session restore?** (is this field saved and restored?)
6. **Does the indexing stay consistent?** (registry indices vs sequential indices vs plate numbers)

**If a change affects step N, check steps N+1 through the end.** Don't just fix the immediate symptom — follow the data all the way to CSV output, plotting, session save/restore, and review/re-click.

## Bug Fix Discipline (CRITICAL)

**Never fix just the one thing that broke. Always audit the surrounding code for the same class of bug.**

1. **Same-class audit**: If attribute X was missing, check if attributes Y and Z are also missing in the same code path. If a button text wasn't updating, check ALL buttons in the same flow.
2. **Session resume is the #1 trap**: `_try_resume()` in app.py has multiple branches (step==2, step>=4, image-done). Each branch must initialize the same `self._*` attributes that the normal workflow sets. When fixing a missing attribute in one branch, audit ALL attributes that the normal flow sets (select_plates → click_roots → measure → review) and confirm each one is either (a) initialized in the resume branch, (b) initialized on-demand with a safety check, or (c) not needed.
3. **tkinter swallows exceptions**: Button callbacks in CustomTkinter silently eat errors. If something "works but the UI doesn't update," the most likely cause is an exception after the working part but before the UI update. Add print/traceback when debugging.
4. **Test the full scenario, not just the symptom**: If the bug is "X doesn't work after session resume," also verify that retrace, manual trace, CSV save, and review all still work after session resume — not just X.

## Code Discipline

- Match existing patterns and style in the codebase.
- Don't introduce new dependencies without asking.
- Don't change function signatures without asking.
- Don't rename variables or functions unless that's the task.

## Project Structure

```
root_measure/
├── gui/                           # ← PRIMARY: CustomTkinter desktop app
│   ├── app.py                     # Entry point: RootMeasureApp (CTk window, layout, image loading)
│   ├── sidebar.py                 # Sidebar: collapsible sections, settings, progress bar
│   ├── canvas.py                  # ImageCanvas: plate selection, root clicking, review, zoom
│   └── workflow.py                # MeasurementMixin: tracing, retry, CSV saving, plotting
├── scripts/                       # Shared backend (tracing, CSV, plotting)
│   ├── config.py                  # Constants and ROI helper functions
│   ├── image_processing.py        # preprocess() — binary root mask generation
│   ├── plate_detection.py         # Plate detection, interior cropping
│   ├── root_tracing.py            # Root tip detection, skeleton graph, path tracing
│   ├── csv_output.py              # CSV append logic
│   ├── plotting.py                # Box plots with statistics (ANOVA, t-test, Tukey, CLD)
│   └── utils.py                   # Image listing, path helpers, segment computation
├── output/                        # CSV results (gitignored)
├── setup.py                       # cx_Freeze build config (standalone .app)
├── RootMeasure.command            # macOS launcher
├── requirements.txt
├── .gitignore
└── CLAUDE.md                      # This file
```

## GUI Architecture

- **app.py**: Main window. Sidebar (left) + ImageCanvas (right) + status bar (bottom). Loads images, routes keyboard events.
- **sidebar.py**: Progressive workflow — collapsible sections (Folder → Images → Settings → Experiment → Workflow). Has progress bar for tracing.
- **canvas.py**: CustomTkinter canvas with modes: VIEW, SELECT_PLATES, CLICK_ROOTS, CLICK_MARKS, REVIEW, RECLICK. Handles zoom, pan, drawing.
- **workflow.py**: MeasurementMixin added to RootMeasureApp. Runs preprocessing, tracing loop, review/retry, CSV save, and plotting.

All new features should target the **gui/** folder. The `scripts/` folder contains shared backend logic (tracing, CSV, plotting) imported by the GUI.
