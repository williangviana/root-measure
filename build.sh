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

# --- 1. Ensure Python 3 is installed ---
if ! command -v python3 &>/dev/null; then
    echo "[1/6] Python 3 not found — installing via Homebrew..."

    # Install Homebrew if missing
    if ! command -v brew &>/dev/null; then
        echo "       Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Add Homebrew to PATH for this session
        if [ -f /opt/homebrew/bin/brew ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [ -f /usr/local/bin/brew ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi

    brew install python
    hash -r  # refresh command cache
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[1/6] Using Python $PY_VERSION"

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

# --- 6. Ad-hoc code sign and strip quarantine ---
echo "[6/6] Signing and stripping quarantine..."
codesign --force --deep --sign - "$INSTALL_PATH"
xattr -cr "$INSTALL_PATH"

echo ""
echo "============================================"
echo "  Build complete!"
echo "  Installed to: $INSTALL_PATH"
echo ""
echo "  You can now launch Root Measure from"
echo "  your Applications folder or Launchpad."
echo "============================================"
