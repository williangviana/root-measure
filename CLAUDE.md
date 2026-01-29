# Project: Root Measure

Interactive Python tool for measuring root lengths from scanned agar plate images.

**Stack:** Python, OpenCV, matplotlib, scikit-image, networkx, pandas

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
├── scripts/
│   ├── measure_roots.py       # Entry point: CLI, image loading, processing pipeline
│   ├── config.py              # Constants and ROI helper functions
│   ├── image_processing.py    # preprocess() — binary root mask generation
│   ├── plate_detection.py     # Plate detection, interior cropping, label prompting
│   ├── click_collector.py     # Interactive matplotlib click handler and display
│   ├── root_tracing.py        # Root tip detection, skeleton graph, path tracing
│   ├── results_display.py     # Traced root overlay visualization
│   ├── csv_output.py          # CSV append logic
│   └── utils.py               # Image listing, path helpers, segment computation
├── output/                    # CSV results (gitignored)
├── .gitignore
├── CLAUDE.md                  # This file
└── GITHUB_SETUP.md            # Git reference
```
