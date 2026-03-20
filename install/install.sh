#!/bin/bash
# Root Measure — One-line installer
# Usage: sudo curl -sL https://raw.githubusercontent.com/williangviana/root-measure/stable/install/install.sh | bash

set -e

APP_NAME="Root Measure"
REPO="williangviana/root-measure"
WORK_DIR="$HOME/.root-measure-install"

# Show errors on failure
trap 'echo ""; echo "ERROR: Installation failed at the step above."; echo "Please screenshot this output and send it for help."; exit 1' ERR

echo ""
echo "============================================"
echo "  Root Measure — Installer"
echo "============================================"
echo ""

# --- 0. Ensure Xcode Command Line Tools are installed ---
# Required for compiling cx_Freeze and other C extensions
if ! xcode-select -p &>/dev/null; then
    echo "[0/7] Installing Xcode Command Line Tools..."
    echo "     A system dialog may appear — click Install and wait."
    xcode-select --install
    # Wait for the installation to complete
    until xcode-select -p &>/dev/null; do
        sleep 5
    done
    echo "[0/7] Xcode CLT ✓"
fi

# --- 1. Ensure Homebrew Python 3 is installed ---
# macOS system Python (3.9) has a slow Tcl/Tk — always use Homebrew Python
if ! command -v brew &>/dev/null; then
    echo "[1/7] Installing Homebrew..."
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Ensure Homebrew bin is in PATH (curl|bash doesn't load shell profile)
if [ -f /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
elif [ -f /usr/local/bin/brew ]; then
    eval "$(/usr/local/bin/brew shellenv)"
fi

if ! brew list python@3.12 &>/dev/null; then
    echo "[1/7] Installing Python 3.12..."
    brew install python@3.12
    hash -r
fi

# Prefer Homebrew python3.12 over system python3
if command -v python3.12 &>/dev/null; then
    PY=python3.12
elif [ -x /opt/homebrew/bin/python3.12 ]; then
    PY=/opt/homebrew/bin/python3.12
elif [ -x /usr/local/bin/python3.12 ]; then
    PY=/usr/local/bin/python3.12
else
    PY=python3
fi

PY_VERSION=$($PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[1/7] Python $PY_VERSION ✓"

# --- 2. Download project from GitHub ---
echo "[2/7] Downloading Root Measure..."
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
curl -sL "https://github.com/$REPO/archive/refs/heads/stable.tar.gz" | tar xz -C "$WORK_DIR" --strip-components=1
cd "$WORK_DIR"
echo "[2/7] Downloaded ✓"

# --- 3. Create virtual environment ---
$PY -m venv .venv
source .venv/bin/activate
echo "[3/7] Virtual environment ✓"

# --- 4. Install dependencies ---
echo "[4/7] Installing dependencies..."
pip install --upgrade pip -q
pip install -r install/requirements.txt -q
echo "[4/7] Installing cx_Freeze (this may take a few minutes)..."
pip install cx_Freeze
echo "[4/7] Dependencies ✓"

# --- 5. Build .app bundle ---
echo "[5/7] Building app..."
python install/setup.py bdist_mac 2>&1 | tail -5

BUILT_APP=$(ls -d build/*.app 2>/dev/null | head -1)
if [ -z "$BUILT_APP" ]; then
    echo "ERROR: Build failed."
    exit 1
fi
echo "[5/7] Build ✓"

# --- 6. Install to /Applications ---
INSTALL_PATH="/Applications/$APP_NAME.app"
rm -rf "$INSTALL_PATH"
mv "$BUILT_APP" "$INSTALL_PATH"
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
