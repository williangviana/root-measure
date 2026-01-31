#!/bin/bash
# Build RootMeasure.app standalone macOS bundle
set -e
cd "$(dirname "$0")"

echo "=== Installing PyInstaller (if needed) ==="
pip install pyinstaller

echo "=== Building RootMeasure.app ==="
pyinstaller --clean --noconfirm RootMeasure.spec

echo ""
echo "Done!  →  dist/RootMeasure.app"
echo "You can double-click it or run:  open dist/RootMeasure.app"
