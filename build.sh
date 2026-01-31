#!/bin/bash
# Build RootMeasure.app standalone macOS bundle
# Outputs to local disk (not Google Drive) for fast launch times.
set -e
cd "$(dirname "$0")"

LOCAL_DIST="$HOME/RootMeasure-build"
LOCAL_WORK="$LOCAL_DIST/work"

echo "=== Installing PyInstaller (if needed) ==="
pip install pyinstaller

echo "=== Building RootMeasure.app ==="
pyinstaller --clean --noconfirm \
    --distpath "$LOCAL_DIST/dist" \
    --workpath "$LOCAL_WORK" \
    RootMeasure.spec

rm -rf "$LOCAL_WORK"

APP="$LOCAL_DIST/dist/RootMeasure.app"

echo "=== Stripping macOS quarantine flags ==="
xattr -cr "$APP"

echo ""
echo "Done!  →  $APP"
echo "You can double-click it or run:  open \"$APP\""
