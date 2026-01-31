#!/bin/bash
# Root Measure — Build Script
# Creates a standalone macOS .app bundle using cx_Freeze.
# Run this on the target architecture (Intel or Apple Silicon).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

APP_NAME="Root Measure"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "============================================"
echo "  Root Measure — Build"
echo "============================================"
echo ""

# --- 1. Check for Python 3 ---
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found."
    echo "Install Python 3 from https://www.python.org/downloads/"
    echo "or via Homebrew:  brew install python"
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[1/6] Found Python $PY_VERSION"

# --- 2. Create virtual environment ---
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/6] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "[2/6] Virtual environment exists"
fi

source "$VENV_DIR/bin/activate"

# --- 3. Install dependencies ---
echo "[3/6] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install cx_Freeze -q

# --- 4. Build .app bundle ---
echo "[4/6] Building $APP_NAME.app..."
rm -rf build/
python setup.py build 2>&1 | tail -5

# Find the built app directory (name varies by platform)
BUILD_DIR=$(ls -d build/exe.* 2>/dev/null | head -1)
if [ -z "$BUILD_DIR" ]; then
    echo "ERROR: Build failed — no output directory found."
    exit 1
fi

# Rename to .app bundle
APP_PATH="$SCRIPT_DIR/$APP_NAME.app"
rm -rf "$APP_PATH"
mv "$BUILD_DIR" "$APP_PATH"

# --- 5. Install to /Applications ---
echo "[5/6] Installing to /Applications..."
INSTALL_PATH="/Applications/$APP_NAME.app"
rm -rf "$INSTALL_PATH"
cp -R "$APP_PATH" "$INSTALL_PATH"

# --- 6. Strip quarantine ---
echo "[6/6] Stripping quarantine flags..."
xattr -cr "$INSTALL_PATH"

echo ""
echo "============================================"
echo "  Build complete!"
echo "  Installed to: $INSTALL_PATH"
echo ""
echo "  You can now launch Root Measure from"
echo "  your Applications folder or Launchpad."
echo "============================================"
