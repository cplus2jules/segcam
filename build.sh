#!/bin/bash
# Builds a universal (Apple Silicon + Intel) SegCam.app and SegCam.zip next to this script.
# Then double-click SegCam.app, or drag it into /Applications.
set -euo pipefail
cd "$(dirname "$0")"

APP=SegCam.app
BIN="$APP/Contents/MacOS/SegCam"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

rm -rf "$APP" SegCam.zip
mkdir -p "$APP/Contents/MacOS"
for arch in arm64 x86_64; do
  swiftc -O -swift-version 5 -target "$arch-apple-macos14" -o "$TMP/SegCam-$arch" main.swift
done
lipo -create "$TMP/SegCam-arm64" "$TMP/SegCam-x86_64" -output "$BIN"
cp Info.plist "$APP/Contents/Info.plist"
codesign --force --sign - "$APP"
ditto -c -k --keepParent "$APP" SegCam.zip
echo "Built $APP and SegCam.zip"
