#!/usr/bin/env bash
# Download the Kodak lossless true color image suite (24 images).
# Source: http://r0k.us/graphics/kodak/

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
OUT="$ROOT/data/kodak"

mkdir -p "$OUT"
echo "Downloading Kodak dataset to $OUT ..."

for i in $(seq 1 24); do
    FNAME=$(printf "kodim%02d.png" $i)
    DEST="$OUT/$FNAME"
    if [ -f "$DEST" ]; then
        echo "  $FNAME already exists, skipping."
    else
        echo "  Downloading $FNAME ..."
        curl -f -o "$DEST" "http://r0k.us/graphics/kodak/kodak/$FNAME" \
            || wget -O "$DEST" "http://r0k.us/graphics/kodak/kodak/$FNAME"
    fi
done

echo "Done. $(ls "$OUT"/*.png | wc -l | tr -d ' ') images in $OUT"
