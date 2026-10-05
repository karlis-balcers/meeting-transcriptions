#!/bin/bash
# Installs (or updates) Meeting Transcriptions on macOS from the latest GitHub release.
#   curl -fsSL https://raw.githubusercontent.com/karlis-balcers/meeting-transcriptions/main/install.sh | bash
set -euo pipefail

REPO="karlis-balcers/meeting-transcriptions"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "This installer is for macOS. On Windows use install.ps1, on Linux run from source with ./run.sh."
    exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
    echo "Release builds are for Apple Silicon (M1 and newer). On an Intel Mac run from source with ./run.sh."
    exit 1
fi

echo "Looking up the latest release..."
url=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep -o '"browser_download_url": *"[^"]*macos-arm64[^"]*\.zip"' \
    | head -1 | sed 's/.*"\(https[^"]*\)"/\1/')
if [ -z "$url" ]; then
    echo "No macOS build found in the latest release."
    exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
echo "Downloading $(basename "$url")..."
curl -fL --progress-bar -o "$tmp/app.zip" "$url"
ditto -x -k "$tmp/app.zip" "$tmp/unpacked"

dest="/Applications"
[ -w "$dest" ] || dest="$HOME/Applications"
mkdir -p "$dest"
pkill -f "MeetingTranscriptions.app" 2>/dev/null || true
rm -rf "$dest/MeetingTranscriptions.app"
mv "$tmp/unpacked/"*.app "$dest/MeetingTranscriptions.app"
# Not notarized, so remove the download quarantine flag or Gatekeeper blocks it.
xattr -dr com.apple.quarantine "$dest/MeetingTranscriptions.app" 2>/dev/null || true

echo "Installed to $dest/MeetingTranscriptions.app"
echo
echo "To capture the other side of a call install BlackHole (brew install blackhole-2ch),"
echo "make a Multi-Output Device in Audio MIDI Setup and pick BlackHole as the output capture device."
open "$dest/MeetingTranscriptions.app"
