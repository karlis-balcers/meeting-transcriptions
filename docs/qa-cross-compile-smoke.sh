#!/usr/bin/env bash
# QA cross-compile smoke gate (Basher, 2026-07-03).
# Builds ./cmd/transcribe for all six targets with CGO_ENABLED=0.
# Run from transcribe/ via WSL. Not part of the shipped product.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 3

OUT="${TMPDIR:-/tmp}"
rm -f "$OUT"/tl-* "$OUT"/td-* "$OUT"/tw-* 2>/dev/null

pass=0
fail=0
results=()

build_one() {
    local goos="$1" goarch="$2" name="$3"
    if GOOS="$goos" GOARCH="$goarch" CGO_ENABLED=0 go build -o "$OUT/$name" ./cmd/transcribe 2>/tmp/cc-err-"$name"; then
        results+=("OK   GOOS=$goos GOARCH=$goarch -> $name")
        pass=$((pass+1))
    else
        results+=("FAIL GOOS=$goos GOARCH=$goarch -> $name ($(cat /tmp/cc-err-"$name" | tr '\n' ' '))")
        fail=$((fail+1))
    fi
}

build_one linux   amd64 tl-amd64
build_one linux   arm64 tl-arm64
build_one darwin  amd64 td-amd64
build_one darwin  arm64 td-arm64
build_one windows amd64 tw-amd64.exe
build_one windows arm64 tw-arm64.exe

echo "=== CROSS-COMPILE RESULTS (pass=$pass fail=$fail) ==="
for r in "${results[@]}"; do echo "$r"; done
echo "=== artifacts ==="
ls -la "$OUT"/tl-amd64 "$OUT"/tl-arm64 "$OUT"/td-amd64 "$OUT"/td-arm64 "$OUT"/tw-amd64.exe "$OUT"/tw-arm64.exe 2>&1
echo "QA_SMOKERC=$fail"
