#!/usr/bin/env bash
# Fetches the YOLO weights solution.py needs, since weights/*.pt is
# gitignored (too large to commit). Run once before an offline eval run.
set -euo pipefail
cd "$(dirname "$0")"

BASE_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0"

for f in yolo11s.pt; do
    if [ -f "$f" ]; then
        echo "already have $f, skipping"
        continue
    fi
    echo "fetching $f..."
    curl -fL -o "$f" "$BASE_URL/$f"
done
