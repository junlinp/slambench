#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-problem-16-22106-pre.txt}"
DATASET_PATH="$ROOT_DIR/data/dubrovnik/$DATASET"
BUILD_DIR="$ROOT_DIR/msckf_c/build"
LOG_FILE="${2:-$ROOT_DIR/shur.log}"

echo "[shur] root: $ROOT_DIR"
echo "[shur] dataset: $DATASET"

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "[shur] dataset not found at $DATASET_PATH"
  exit 1
fi

# Build shur_runner (CPU) from workspace, or use image binary (/opt/slambench from Dockerfile)
if [[ -d "$ROOT_DIR/msckf_c" ]]; then
  cmake -S "$ROOT_DIR/msckf_c" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
  cmake --build "$BUILD_DIR" --config Release -j >/dev/null 2>&1
fi

WORKSPACE_BINARY="$BUILD_DIR/shur_runner"
IMAGE_BINARY="/opt/slambench/msckf_c/build/shur_runner"
if [[ -x "$WORKSPACE_BINARY" ]]; then
  SHUR_BIN="$WORKSPACE_BINARY"
elif [[ -x "$IMAGE_BINARY" ]]; then
  SHUR_BIN="$IMAGE_BINARY"
else
  echo "[shur] CPU binary not found:"
  echo "  - $WORKSPACE_BINARY (build from workspace msckf_c)"
  echo "  - $IMAGE_BINARY (devcontainer image)"
  exit 1
fi

echo "[shur] command: $SHUR_BIN $DATASET_PATH"

TIMEFORMAT="TIME %R"
{ time "$SHUR_BIN" "$DATASET_PATH"; } > "$LOG_FILE" 2>&1

echo "[shur] done. Log: $LOG_FILE"
