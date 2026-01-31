#!/bin/bash
# Double-click this file to install Root Measure.
# Everything is automatic — no Terminal knowledge needed.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

APP_NAME="Root Measure"
VENV_DIR="$SCRIPT_DIR/.venv"

clear
echo ""
echo "  ============================================"
echo "    Root Measure — Installer"
echo "  ============================================"
echo ""
echo "  This will install Root Measure to your"
echo "  Applications folder. Please wait..."
echo ""

# --- 1. Ensure Python 3 is installed ---
if ! command -v python3 &>/dev/null; then
    echo "  [1/6] Installing Python (this may take a few minutes)..."

    if ! command -v brew &>/dev/null; then
        echo "         Installing Homebrew first..."
        NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        if [ -f /opt/homebrew/bin/brew ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [ -f /usr/local/bin/brew ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi

    brew install python
    hash -r
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  [1/6] Python $PY_VERSION ✓"

# --- 2. Create virtual environment ---
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "  [2/6] Virtual environment ✓"

# --- 3. Install dependencies ---
echo "  [3/6] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install cx_Freeze -q
echo "  [3/6] Dependencies ✓"

# --- 4. Build .app bundle ---
echo "  [4/6] Building app..."
rm -rf build/
python setup.py build 2>&1 | tail -3

BUILD_DIR=$(ls -d build/exe.* 2>/dev/null | head -1)
if [ -z "$BUILD_DIR" ]; then
    echo ""
    echo "  ERROR: Build failed."
    echo "  Press any key to close."
    read -n 1
    exit 1
fi

APP_PATH="$SCRIPT_DIR/$APP_NAME.app"
rm -rf "$APP_PATH"
mv "$BUILD_DIR" "$APP_PATH"
echo "  [4/6] Build ✓"

# --- 5. Install to /Applications ---
INSTALL_PATH="/Applications/$APP_NAME.app"
rm -rf "$INSTALL_PATH"
cp -R "$APP_PATH" "$INSTALL_PATH"
echo "  [5/6] Installed to Applications ✓"

# --- 6. Strip quarantine ---
xattr -cr "$INSTALL_PATH"
echo "  [6/6] Ready ✓"

# Clean up local build
rm -rf "$APP_PATH" build/

echo ""
echo "  ============================================"
echo "    Installation complete!"
echo ""
echo "    Open Root Measure from your"
echo "    Applications folder or Launchpad."
echo "  ============================================"
echo ""
echo "  Press any key to close this window."
read -n 1
