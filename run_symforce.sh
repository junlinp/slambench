#!/usr/bin/env bash
set -euo pipefail

# Workspace root (assumes script is run from repo root inside the devcontainer)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer workspace symforce; fall back to image layout (/opt/slambench from Dockerfile)
if [[ -d "$ROOT_DIR/symforce" ]]; then
  SYMFORCE_DIR="$ROOT_DIR/symforce"
elif [[ -d "/opt/slambench/symforce" ]]; then
  SYMFORCE_DIR="/opt/slambench/symforce"
else
  echo "[symforce] symforce not found at $ROOT_DIR/symforce or /opt/slambench/symforce"
  exit 1
fi
BUILD_DIR="$SYMFORCE_DIR/build"
DATASET="${1:-problem-16-22106-pre.txt}"
# Expect dataset to be checked into the repo under data/dubrovnik/
DATASET_PATH="$ROOT_DIR/data/dubrovnik/$DATASET"
LOG_FILE="${2:-$ROOT_DIR/symforce.log}"

echo "[symforce] root: $ROOT_DIR"
echo "[symforce] dataset: $DATASET_PATH"

SYM_EXE="$BUILD_DIR/bin/examples/bundle_adjustment_in_the_large_example"
if [[ ! -x "$SYM_EXE" ]]; then
  echo "[symforce] executable missing: $SYM_EXE"
  echo "[symforce] build with: (cd \"$SYMFORCE_DIR\" && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j bundle_adjustment_in_the_large_example)"
  exit 1
fi

echo "[symforce] running example..."
echo "[symforce] command: $SYM_EXE $DATASET_PATH"
"$SYM_EXE" "$DATASET_PATH" > "$LOG_FILE" 2>&1

echo "[symforce] done. Log: $LOG_FILE"
