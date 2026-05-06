#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-problem-16-22106-pre.txt}"
DATASET_PATH="$ROOT_DIR/data/dubrovnik/$DATASET"
BUILD_DIR="$ROOT_DIR/sam/build"
LOG_FILE="${2:-$ROOT_DIR/sam.log}"

echo "[sam] root: $ROOT_DIR"
echo "[sam] dataset: $DATASET"

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "[sam] dataset not found at $DATASET_PATH"
  exit 1
fi

if [[ -d "$ROOT_DIR/sam" ]]; then
  rm -rf "$BUILD_DIR"
  cmake -S "$ROOT_DIR/sam" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR" --config Release -j
fi

WORKSPACE_BINARY="$BUILD_DIR/sam_runner"
if [[ ! -x "$WORKSPACE_BINARY" ]]; then
  echo "[sam] binary not found: $WORKSPACE_BINARY"
  exit 1
fi

CMD="$WORKSPACE_BINARY \"$DATASET_PATH\""
echo "[sam] command: $CMD"

{ eval "$CMD"; } > "$LOG_FILE" 2>&1

echo "[sam] done. Log: $LOG_FILE"
