#!/bin/bash
# ──────────────────────────────────────────────────────────────
# RootMeasure Launcher
# Double-click this file to start Root Measure.
# On first run it will install Python and dependencies for you.
# ──────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
REQ="$SCRIPT_DIR/requirements.txt"

# ── Helpers ───────────────────────────────────────────────────

info()  { printf '\n  ✅  %s\n' "$1"; }
step()  { printf '\n  ⏳  %s …\n' "$1"; }
fail()  { printf '\n  ❌  %s\n' "$1"; exit 1; }

# ── 1. Ensure Python 3 ───────────────────────────────────────

if command -v python3 &>/dev/null; then
    PY="$(command -v python3)"
else
    # Try Homebrew install
    if ! command -v brew &>/dev/null; then
        step "Installing Homebrew (needed for Python)"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # Add brew to PATH for Apple Silicon and Intel
        if [ -f /opt/homebrew/bin/brew ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [ -f /usr/local/bin/brew ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi
    step "Installing Python 3 via Homebrew"
    brew install python@3
    PY="$(command -v python3)" || fail "Python 3 installation failed"
fi

PY_VER="$("$PY" --version 2>&1)"
info "Python found: $PY_VER"

# ── 2. Create virtual environment (first run only) ───────────

if [ ! -d "$VENV" ]; then
    step "Creating virtual environment (one-time setup)"
    "$PY" -m venv "$VENV"
    info "Virtual environment created"
fi

# Activate
source "$VENV/bin/activate"

# ── 3. Install / update dependencies ─────────────────────────

STAMP="$VENV/.req_stamp"
if [ ! -f "$STAMP" ] || [ "$REQ" -nt "$STAMP" ]; then
    step "Installing dependencies (this may take a minute)"
    pip install --upgrade pip -q
    pip install -r "$REQ" -q
    touch "$STAMP"
    info "Dependencies installed"
else
    info "Dependencies up to date"
fi

# ── 4. Launch ─────────────────────────────────────────────────

info "Starting Root Measure …"
cd "$SCRIPT_DIR"
python gui/app.py
