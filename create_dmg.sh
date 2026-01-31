#!/bin/bash
# Create a distributable DMG from RootMeasure.app
set -e
cd "$(dirname "$0")"

APP="dist/RootMeasure.app"
DMG="dist/RootMeasure.dmg"
VOLUME="RootMeasure"
STAGING="dist/dmg_staging"

if [ ! -d "$APP" ]; then
    echo "Error: $APP not found. Run build.sh first."
    exit 1
fi

echo "=== Creating DMG ==="
rm -rf "$STAGING" "$DMG"
mkdir -p "$STAGING"
cp -R "$APP" "$STAGING/"
ln -s /Applications "$STAGING/Applications"

hdiutil create -volname "$VOLUME" \
    -srcfolder "$STAGING" \
    -ov -format UDZO \
    "$DMG"

rm -rf "$STAGING"
echo ""
echo "Done!  →  $DMG"
echo "Share this file — recipients drag RootMeasure.app to Applications."
