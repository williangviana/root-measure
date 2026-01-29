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

Currently a single file — will be updated after module split:

```
root_measure/
├── measure_roots.py   # 1,575 lines — monolith (to be split)
├── CLAUDE.md          # this file
└── GITHUB_SETUP.md    # git reference
```
