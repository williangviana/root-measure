#!/bin/bash
# Create a distributable DMG from RootMeasure.app
set -e

LOCAL_DIST="$HOME/RootMeasure-build/dist"
APP="$LOCAL_DIST/RootMeasure.app"
DMG="$LOCAL_DIST/RootMeasure.dmg"
VOLUME="RootMeasure"
STAGING="$LOCAL_DIST/dmg_staging"

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
