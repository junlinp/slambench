#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-problem-16-22106-pre.txt}"
DATASET_PATH="$ROOT_DIR/data/dubrovnik/$DATASET"
BUILD_DIR="$ROOT_DIR/msckf_c/build"
WORKSPACE_BINARY="$BUILD_DIR/shur_runner"
LOG_FILE="${2:-$ROOT_DIR/shur.log}"

echo "[shur] root: $ROOT_DIR"

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "[shur] dataset not found at $DATASET_PATH"
  exit 1
fi

if [[ -d "$ROOT_DIR/msckf_c" ]]; then
  cmake -S "$ROOT_DIR/msckf_c" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$BUILD_DIR" --config Release -j >/dev/null
else
  echo "[shur] msckf_c source directory not found at $ROOT_DIR/msckf_c"
  exit 1
fi

if [[ ! -x "$WORKSPACE_BINARY" ]]; then
  echo "[shur] binary not found or not executable: $WORKSPACE_BINARY"
  exit 1
fi

CMD="$WORKSPACE_BINARY \"$DATASET_PATH\""
echo "[shur] command: $CMD"

TIMEFORMAT="TIME %R"
{ time eval "$CMD"; } > "$LOG_FILE" 2>&1

echo "[shur] done. Log: $LOG_FILE"

