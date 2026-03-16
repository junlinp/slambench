#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-problem-16-22106-pre.txt}"
DATASET_PATH="$ROOT_DIR/data/dubrovnik/$DATASET"
LOG_FILE="${2:-$ROOT_DIR/gtsam.log}"
BINARY_BAL="/opt/slambench/gtsam/build/examples/SFMExample_bal"
BINARY_SMART="/opt/slambench/gtsam/build/examples/SFMExample_SmartFactor_bal"
LOG_BASE="${LOG_FILE%.log}"
if [[ "$LOG_BASE" == "$LOG_FILE" ]]; then
  LOG_BASE="$LOG_FILE"
fi
LOG_BAL="${LOG_BASE}_bal.log"
LOG_SMART="${LOG_BASE}_smartfactor.log"

echo "[gtsam] root: $ROOT_DIR"

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "[gtsam] dataset not found at $DATASET_PATH"
  exit 1
fi

if [[ ! -x "$BINARY_BAL" || ! -x "$BINARY_SMART" ]]; then
  echo "[gtsam] required binaries not found:"
  echo "  - $BINARY_BAL"
  echo "  - $BINARY_SMART"
  echo "Ensure the devcontainer image was built (it builds gtsam examples)."
  exit 1
fi

parse_total_time() {
  local log="$1"
  awk '/^TIME[[:space:]]/ {print $2}' "$log" | head -n1
}

echo "[gtsam] running SFMExample_bal..."
CMD="$BINARY_BAL $DATASET_PATH"
echo "[gtsam] command: $CMD"
TIMEFORMAT="TIME %R"
{ time eval "$CMD"; } > "$LOG_BAL" 2>&1

echo "[gtsam] running SFMExample_SmartFactor_bal..."
CMD="$BINARY_SMART $DATASET_PATH"
echo "[gtsam] command: $CMD"
TIMEFORMAT="TIME %R"
{ time eval "$CMD"; } > "$LOG_SMART" 2>&1

cp "$LOG_BAL" "$LOG_FILE"

echo
echo "[gtsam] dual-binary comparison (seconds):"
printf "%-24s %12s\n" "mode" "TIME(s)"
printf "%-24s %12s\n" "------------------------" "------------"
printf "%-24s %12s\n" "SFMExample_bal" "$(parse_total_time "$LOG_BAL")"
printf "%-24s %12s\n" "SFMExample_SmartFactor_bal" "$(parse_total_time "$LOG_SMART")"

echo "[gtsam] done. Log: $LOG_FILE"
echo "[gtsam] logs:"
echo "  bal         -> $LOG_BAL"
echo "  smartfactor -> $LOG_SMART"
echo "  default     -> $LOG_FILE (bal, for compatibility)"
