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
├── scripts/                       # Shared backend + legacy CLI
│   ├── config.py                  # Constants and ROI helper functions
│   ├── image_processing.py        # preprocess() — binary root mask generation
│   ├── plate_detection.py         # Plate detection, interior cropping, label prompting
│   ├── root_tracing.py            # Root tip detection, skeleton graph, path tracing
│   ├── csv_output.py              # CSV append logic
│   ├── plotting.py                # Box plots with statistics (ANOVA, t-test, Tukey, CLD)
│   ├── utils.py                   # Image listing, path helpers, segment computation
│   ├── measure_roots.py           # Legacy CLI entry point (not primary)
│   ├── click_collector.py         # Legacy CLI matplotlib click handler
│   └── results_display.py         # Legacy CLI traced root overlay
├── output/                        # CSV results (gitignored)
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

All new features should target the **gui/** folder. The `scripts/` folder contains shared backend logic (tracing, CSV, plotting) imported by both GUI and legacy CLI.
