#!/bin/bash
# Root Measure — One-line installer
# Usage: curl -sL https://raw.githubusercontent.com/williangviana/root-measure/master/install.sh | bash

set -e

APP_NAME="Root Measure"
REPO="williangviana/root-measure"
WORK_DIR="$HOME/.root-measure-install"

echo ""
echo "============================================"
echo "  Root Measure — Installer"
echo "============================================"
echo ""

# --- 1. Ensure Python 3 is installed ---
if ! command -v python3 &>/dev/null; then
    echo "[1/7] Installing Python..."

    if ! command -v brew &>/dev/null; then
        echo "       Installing Homebrew first..."
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
echo "[1/7] Python $PY_VERSION ✓"

# --- 2. Download project from GitHub ---
echo "[2/7] Downloading Root Measure..."
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
curl -sL "https://github.com/$REPO/archive/refs/heads/master.tar.gz" | tar xz -C "$WORK_DIR" --strip-components=1
cd "$WORK_DIR"
echo "[2/7] Downloaded ✓"

# --- 3. Create virtual environment ---
python3 -m venv .venv
source .venv/bin/activate
echo "[3/7] Virtual environment ✓"

# --- 4. Install dependencies ---
echo "[4/7] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install cx_Freeze -q
echo "[4/7] Dependencies ✓"

# --- 5. Build .app bundle ---
echo "[5/7] Building app..."
python setup.py build 2>&1 | tail -3

BUILD_DIR=$(ls -d build/exe.* 2>/dev/null | head -1)
if [ -z "$BUILD_DIR" ]; then
    echo "ERROR: Build failed."
    exit 1
fi
echo "[5/7] Build ✓"

# --- 6. Install to /Applications ---
INSTALL_PATH="/Applications/$APP_NAME.app"
rm -rf "$INSTALL_PATH"
mv "$BUILD_DIR" "$INSTALL_PATH"
echo "[6/7] Installed to Applications ✓"

# --- 7. Strip quarantine ---
xattr -cr "$INSTALL_PATH"
echo "[7/7] Ready ✓"

# Clean up
rm -rf "$WORK_DIR"

echo ""
echo "============================================"
echo "  Installation complete!"
echo ""
echo "  Open Root Measure from your"
echo "  Applications folder or Launchpad."
echo "============================================"
