#!/bin/bash
# Root Measure — One-line installer
# Usage: curl -sL https://raw.githubusercontent.com/williangviana/root-measure/stable/install/install.sh | bash

set -e

APP_NAME="Root Measure"
REPO="williangviana/root-measure"
WORK_DIR="$HOME/.root-measure-install"

echo ""
echo "============================================"
echo "  Root Measure — Installer"
echo "============================================"
echo ""

# --- Helper: check if user has admin (sudo) access ---
has_admin() {
    # Quick check without prompting for password
    sudo -n true 2>/dev/null
}

# --- 1. Ensure Python 3 is available ---
# Prefer Homebrew Python 3.12 (faster Tcl/Tk) but fall back to system Python
PY=""

# Try existing Homebrew first (no install needed)
if command -v brew &>/dev/null; then
    if brew list python@3.12 &>/dev/null; then
        if command -v python3.12 &>/dev/null; then
            PY=python3.12
        elif [ -x /opt/homebrew/bin/python3.12 ]; then
            PY=/opt/homebrew/bin/python3.12
        elif [ -x /usr/local/bin/python3.12 ]; then
            PY=/usr/local/bin/python3.12
        fi
    fi
fi

# If no Homebrew Python, try installing Homebrew + Python (needs admin)
if [ -z "$PY" ] && has_admin; then
    if ! command -v brew &>/dev/null; then
        echo "[1/7] Installing Homebrew..."
        NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        if [ -f /opt/homebrew/bin/brew ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [ -f /usr/local/bin/brew ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi
    if command -v brew &>/dev/null && ! brew list python@3.12 &>/dev/null; then
        echo "[1/7] Installing Python 3.12..."
        brew install python@3.12
        hash -r
    fi
    if command -v python3.12 &>/dev/null; then
        PY=python3.12
    elif [ -x /opt/homebrew/bin/python3.12 ]; then
        PY=/opt/homebrew/bin/python3.12
    elif [ -x /usr/local/bin/python3.12 ]; then
        PY=/usr/local/bin/python3.12
    fi
fi

# Fall back to system Python 3 (ships with macOS / Xcode CLI tools)
if [ -z "$PY" ]; then
    if command -v python3 &>/dev/null; then
        PY=python3
        echo "[1/7] Using system Python (no admin access for Homebrew)"
    else
        echo "ERROR: Python 3 not found. Please install Python from https://www.python.org/downloads/"
        exit 1
    fi
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
pip install cx_Freeze -q
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

# --- 6. Install to Applications ---
# Use /Applications if writable, otherwise ~/Applications
if [ -w /Applications ]; then
    INSTALL_DIR="/Applications"
else
    INSTALL_DIR="$HOME/Applications"
    mkdir -p "$INSTALL_DIR"
fi

INSTALL_PATH="$INSTALL_DIR/$APP_NAME.app"
rm -rf "$INSTALL_PATH"
mv "$BUILT_APP" "$INSTALL_PATH"
echo "[6/7] Installed to $INSTALL_DIR ✓"

# --- 7. Strip quarantine ---
xattr -cr "$INSTALL_PATH" 2>/dev/null || true
echo "[7/7] Ready ✓"

# Clean up
rm -rf "$WORK_DIR"

echo ""
echo "============================================"
echo "  Installation complete!"
echo ""
echo "  Open Root Measure from:"
echo "  $INSTALL_DIR"
echo "============================================"
