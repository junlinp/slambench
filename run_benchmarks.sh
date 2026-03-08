#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="$(uname -m)"
CPU_MODEL="$(awk -F: '/model name/ {print $2; exit}' /proc/cpuinfo | sed 's/^ //')"
CPU_CORES="$(nproc)"
CPU_CACHE="$(awk -F: '/cache size/ {print $2; exit}' /proc/cpuinfo | sed 's/^ //')"
SCHED_POLICY="$(chrt -p $$ 2>/dev/null | awk -F: '/scheduling policy/ {gsub(/^ +/,"",$2); print $2; exit}')"
SCHED_PRIORITY="$(chrt -p $$ 2>/dev/null | awk -F: '/scheduling priority/ {gsub(/^ +/,"",$2); print $2; exit}')"
CPU_AFFINITY="$(taskset -pc $$ 2>/dev/null | awk -F: '{gsub(/^ +/,"",$2); print $2; exit}')"
CPU_GOVERNOR="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true)"

print_section() {
  local title="$1"
  echo
  echo "========== $title =========="
  echo
}

sanitize_name() {
  local s="$1"
  s="${s//\//_}"
  s="${s// /_}"
  s="${s//=/_}"
  echo "$s"
}

parse_symforce_total() {
  local log="$1"
  awk -F'|' '/Optimizer<sym::Optimize>::Optimize/ {gsub(/ /,"",$3); print $3}' "$log" | head -n1
}

parse_symforce_iters() {
  local log="$1"
  local finished
  finished=$(awk '/Finished in [0-9]+ iterations/ {for(i=1;i<=NF;i++){if($i ~ /^[0-9]+$/){print $i; exit}}}' "$log")
  if [[ -n "${finished:-}" ]]; then
    echo "$finished"
    return
  fi
  local count
  count=$(grep -c 'LM<sym::Optimize> \[iter' "$log" || true)
  echo "${count:-0}"
}

parse_symforce_initial_cost() {
  local log="$1"
  awk -F'error prev/linear/new: ' '
    /error prev\/linear\/new:/ {
      split($2, a, ",");
      gsub(/ /, "", a[1]);
      split(a[1], b, "/");
      print b[1];
      exit
    }' "$log"
}

parse_symforce_final_cost() {
  local log="$1"
  awk -F'error prev/linear/new: ' '
    /error prev\/linear\/new:/ {
      split($2, a, ",");
      gsub(/ /, "", a[1]);
      split(a[1], b, "/");
      v=b[3];
    }
    END {print v}' "$log"
}

parse_ceres_total() {
  local log="$1"
  awk '/^Total[[:space:]]/ {print $2}' "$log" | head -n1
}

parse_ceres_iters() {
  local log="$1"
  awk '/^[[:space:]]*[0-9]+[[:space:]]/ {print}' "$log" | wc -l
}

parse_ceres_initial_cost() {
  local log="$1"
  awk '/^Initial[[:space:]]/ {print $2}' "$log" | head -n1
}

parse_ceres_final_cost() {
  local log="$1"
  awk '/^Final[[:space:]]/ {print $2}' "$log" | head -n1
}

parse_gtsam_total() {
  local log="$1"
  awk '/^TIME[[:space:]]/ {print $2}' "$log" | head -n1
}

parse_gtsam_iters() {
  local log="$1"
  local iter
  iter=$(awk '/iterations:/ {for(i=1;i<=NF;i++){if($i ~ /^[0-9]+$/){print $i; exit}}}' "$log")
  if [[ -n "${iter:-}" ]]; then
    echo "$iter"
    return
  fi
  echo 1
}

parse_gtsam_initial_cost() {
  local log="$1"
  awk '/^Initial error:/ {print $3}' "$log" | head -n1
}

parse_gtsam_final_cost() {
  local log="$1"
  awk '/^final error:/ {print $3}' "$log" | head -n1
}

parse_msckf_total() {
  local log="$1"
  awk '/^TIME[[:space:]]/ {print $2}' "$log" | head -n1
}

parse_msckf_iters() {
  local log="$1"
  awk '/^iterations:/ {print $2}' "$log" | head -n1
}

parse_msckf_initial_cost() {
  local log="$1"
  awk '/^iter 0 loss/ {print $4; exit}' "$log"
}

parse_msckf_final_cost() {
  local log="$1"
  local v
  v=$(awk -F'loss=' '/step alpha=/ {print $2}' "$log" | tail -n1)
  if [[ -n "${v:-}" ]]; then
    echo "$v"
    return
  fi
  awk '/^iter [0-9]+ loss/ {v=$4} END {print v}' "$log"
}

parse_shur_total() {
  local log="$1"
  awk '/^TIME[[:space:]]/ {print $2}' "$log" | head -n1
}

parse_shur_iters() {
  local log="$1"
  # Count "iter N loss" lines (same format as msckf); last iter index + 1 = iters
  awk '/^iter [0-9]+ loss/ {k=$2} END {print k+1}' "$log"
}

parse_shur_initial_cost() {
  local log="$1"
  awk '/^iter 0 loss/ {print $4; exit}' "$log"
}

parse_shur_final_cost() {
  local log="$1"
  awk '/^iter [0-9]+ loss/ {v=$4} END {print v}' "$log"
}

run_one_dataset() {
  local dataset="$1"
  local suffix="$2"
  local dataset_path="$ROOT_DIR/data/dubrovnik/$dataset"

  if [[ ! -f "$dataset_path" ]]; then
    echo "[bench] dataset not found: $dataset_path"
    exit 1
  fi

  local sym_log ceres_log ceres_sparse_log gtsam_log msckf_log shur_log
  if [[ -n "$suffix" ]]; then
    sym_log="$ROOT_DIR/symforce_${suffix}.log"
    ceres_log="$ROOT_DIR/ceres_${suffix}.log"
    ceres_sparse_log="$ROOT_DIR/ceres_sparse_${suffix}.log"
    gtsam_log="$ROOT_DIR/gtsam_${suffix}.log"
    msckf_log="$ROOT_DIR/msckf_${suffix}.log"
    shur_log="$ROOT_DIR/shur_${suffix}.log"
  else
    sym_log="$ROOT_DIR/symforce.log"
    ceres_log="$ROOT_DIR/ceres.log"
    ceres_sparse_log="$ROOT_DIR/ceres_sparse.log"
    gtsam_log="$ROOT_DIR/gtsam.log"
    msckf_log="$ROOT_DIR/msckf.log"
    shur_log="$ROOT_DIR/shur.log"
  fi

  local gtsam_base gtsam_bal_log gtsam_smart_log
  gtsam_base="${gtsam_log%.log}"
  if [[ "$gtsam_base" == "$gtsam_log" ]]; then
    gtsam_base="$gtsam_log"
  fi
  gtsam_bal_log="${gtsam_base}_bal.log"
  gtsam_smart_log="${gtsam_base}_smartfactor.log"

  local msckf_base msckf_rr_log msckf_rc_log msckf_cr_log msckf_cc_log msckf_openblas_log
  msckf_base="${msckf_log%.log}"
  if [[ "$msckf_base" == "$msckf_log" ]]; then
    msckf_base="$msckf_log"
  fi
  msckf_rr_log="${msckf_base}_rr.log"
  msckf_rc_log="${msckf_base}_rc.log"
  msckf_cr_log="${msckf_base}_cr.log"
  msckf_cc_log="${msckf_base}_cc.log"
  msckf_openblas_log="${msckf_base}_openblas.log"

  print_section "Run SymForce [$dataset]"
  bash "$ROOT_DIR/run_symforce.sh" "$dataset" "$sym_log"

  print_section "Run Ceres Dense [$dataset]"
  bash "$ROOT_DIR/run_ceres.sh" dense_schur "$ceres_log" "$dataset"

  print_section "Run Ceres Sparse [$dataset]"
  bash "$ROOT_DIR/run_ceres.sh" sparse_schur "$ceres_sparse_log" "$dataset"

  print_section "Run GTSAM [$dataset]"
  bash "$ROOT_DIR/run_gtsam.sh" "$dataset" "$gtsam_log"

  print_section "Run MSCKF [$dataset]"
  bash "$ROOT_DIR/run_msckf.sh" "$dataset" "$msckf_log"

  print_section "Run shur [$dataset]"
  bash "$ROOT_DIR/run_shur.sh" "$dataset" "$shur_log"

  local sym_t ceres_t ceres_sparse_t gtsam_bal_t gtsam_smart_t
  local sym_iters ceres_iters ceres_sparse_iters gtsam_bal_iters gtsam_smart_iters
  local msckf_rr_t msckf_rc_t msckf_cr_t msckf_cc_t msckf_openblas_t
  local msckf_rr_iters msckf_rc_iters msckf_cr_iters msckf_cc_iters msckf_openblas_iters
  local shur_t shur_iters
  local sym_ci sym_cf ceres_ci ceres_cf ceres_sparse_ci ceres_sparse_cf
  local gtsam_bal_ci gtsam_bal_cf gtsam_smart_ci gtsam_smart_cf
  local msckf_rr_ci msckf_rr_cf msckf_rc_ci msckf_rc_cf msckf_cr_ci msckf_cr_cf msckf_cc_ci msckf_cc_cf
  local msckf_openblas_ci msckf_openblas_cf

  sym_t=$(parse_symforce_total "$sym_log")
  ceres_t=$(parse_ceres_total "$ceres_log")
  ceres_sparse_t=$(parse_ceres_total "$ceres_sparse_log")
  gtsam_bal_t=$(parse_gtsam_total "$gtsam_bal_log")
  gtsam_smart_t=$(parse_gtsam_total "$gtsam_smart_log")

  sym_iters=$(parse_symforce_iters "$sym_log")
  ceres_iters=$(parse_ceres_iters "$ceres_log")
  ceres_sparse_iters=$(parse_ceres_iters "$ceres_sparse_log")
  gtsam_bal_iters=$(parse_gtsam_iters "$gtsam_bal_log")
  gtsam_smart_iters=$(parse_gtsam_iters "$gtsam_smart_log")

  sym_ci=$(parse_symforce_initial_cost "$sym_log")
  sym_cf=$(parse_symforce_final_cost "$sym_log")
  ceres_ci=$(parse_ceres_initial_cost "$ceres_log")
  ceres_cf=$(parse_ceres_final_cost "$ceres_log")
  ceres_sparse_ci=$(parse_ceres_initial_cost "$ceres_sparse_log")
  ceres_sparse_cf=$(parse_ceres_final_cost "$ceres_sparse_log")
  gtsam_bal_ci=$(parse_gtsam_initial_cost "$gtsam_bal_log")
  gtsam_bal_cf=$(parse_gtsam_final_cost "$gtsam_bal_log")
  gtsam_smart_ci=$(parse_gtsam_initial_cost "$gtsam_smart_log")
  gtsam_smart_cf=$(parse_gtsam_final_cost "$gtsam_smart_log")

  msckf_rr_t=$(parse_msckf_total "$msckf_rr_log")
  msckf_rc_t=$(parse_msckf_total "$msckf_rc_log")
  msckf_cr_t=$(parse_msckf_total "$msckf_cr_log")
  msckf_cc_t=$(parse_msckf_total "$msckf_cc_log")
  msckf_rr_iters=$(parse_msckf_iters "$msckf_rr_log")
  msckf_rc_iters=$(parse_msckf_iters "$msckf_rc_log")
  msckf_cr_iters=$(parse_msckf_iters "$msckf_cr_log")
  msckf_cc_iters=$(parse_msckf_iters "$msckf_cc_log")
  msckf_rr_ci=$(parse_msckf_initial_cost "$msckf_rr_log")
  msckf_rr_cf=$(parse_msckf_final_cost "$msckf_rr_log")
  msckf_rc_ci=$(parse_msckf_initial_cost "$msckf_rc_log")
  msckf_rc_cf=$(parse_msckf_final_cost "$msckf_rc_log")
  msckf_cr_ci=$(parse_msckf_initial_cost "$msckf_cr_log")
  msckf_cr_cf=$(parse_msckf_final_cost "$msckf_cr_log")
  msckf_cc_ci=$(parse_msckf_initial_cost "$msckf_cc_log")
  msckf_cc_cf=$(parse_msckf_final_cost "$msckf_cc_log")
  if [[ -f "$msckf_openblas_log" ]]; then
    msckf_openblas_t=$(parse_msckf_total "$msckf_openblas_log")
    msckf_openblas_iters=$(parse_msckf_iters "$msckf_openblas_log")
    msckf_openblas_ci=$(parse_msckf_initial_cost "$msckf_openblas_log")
    msckf_openblas_cf=$(parse_msckf_final_cost "$msckf_openblas_log")
  else
    msckf_openblas_t=""
    msckf_openblas_iters=""
    msckf_openblas_ci=""
    msckf_openblas_cf=""
  fi
  shur_t=$(parse_shur_total "$shur_log")
  shur_iters=$(parse_shur_iters "$shur_log")
  shur_ci=$(parse_shur_initial_cost "$shur_log")
  shur_cf=$(parse_shur_final_cost "$shur_log")

  if [[ -z "$sym_t" || -z "$ceres_t" || -z "$ceres_sparse_t" || -z "$gtsam_bal_t" || -z "$gtsam_smart_t" || -z "$msckf_rr_t" || -z "$msckf_rc_t" || -z "$msckf_cr_t" || -z "$msckf_cc_t" || -z "$shur_t" || -z "$sym_iters" || -z "$ceres_iters" || -z "$ceres_sparse_iters" || -z "$gtsam_bal_iters" || -z "$gtsam_smart_iters" || -z "$msckf_rr_iters" || -z "$msckf_rc_iters" || -z "$msckf_cr_iters" || -z "$msckf_cc_iters" || -z "$shur_iters" || "$sym_iters" -eq 0 || "$ceres_iters" -eq 0 || "$ceres_sparse_iters" -eq 0 || "$gtsam_bal_iters" -eq 0 || "$gtsam_smart_iters" -eq 0 || "$msckf_rr_iters" -eq 0 || "$msckf_rc_iters" -eq 0 || "$msckf_cr_iters" -eq 0 || "$msckf_cc_iters" -eq 0 || "$shur_iters" -eq 0 ]]; then
    echo "[bench] failed to parse totals/iters for dataset '$dataset'"
    exit 1
  fi

  export ARCH CPU_MODEL CPU_CORES CPU_CACHE
  export SCHED_POLICY SCHED_PRIORITY CPU_AFFINITY CPU_GOVERNOR
  export DATASET_NAME="$dataset"

  python3 - <<PY
import os
sym_total = float("$sym_t")
ceres_total = float("$ceres_t")
ceres_sparse_total = float("$ceres_sparse_t")
gtsam_bal_total = float("$gtsam_bal_t")
gtsam_smart_total = float("$gtsam_smart_t")
msckf_rr_total = float("$msckf_rr_t")
msckf_rc_total = float("$msckf_rc_t")
msckf_cr_total = float("$msckf_cr_t")
msckf_cc_total = float("$msckf_cc_t")
msckf_openblas_total_str = "$msckf_openblas_t"
sym_iters = int("$sym_iters")
ceres_iters = int("$ceres_iters")
ceres_sparse_iters = int("$ceres_sparse_iters")
gtsam_bal_iters = int("$gtsam_bal_iters")
gtsam_smart_iters = int("$gtsam_smart_iters")
msckf_rr_iters = int("$msckf_rr_iters")
msckf_rc_iters = int("$msckf_rc_iters")
msckf_cr_iters = int("$msckf_cr_iters")
msckf_cc_iters = int("$msckf_cc_iters")
msckf_openblas_iters_str = "$msckf_openblas_iters"
sym_ci_str = "$sym_ci"
sym_cf_str = "$sym_cf"
ceres_ci_str = "$ceres_ci"
ceres_cf_str = "$ceres_cf"
ceres_sparse_ci_str = "$ceres_sparse_ci"
ceres_sparse_cf_str = "$ceres_sparse_cf"
gtsam_bal_ci_str = "$gtsam_bal_ci"
gtsam_bal_cf_str = "$gtsam_bal_cf"
gtsam_smart_ci_str = "$gtsam_smart_ci"
gtsam_smart_cf_str = "$gtsam_smart_cf"
msckf_rr_ci_str = "$msckf_rr_ci"
msckf_rr_cf_str = "$msckf_rr_cf"
msckf_rc_ci_str = "$msckf_rc_ci"
msckf_rc_cf_str = "$msckf_rc_cf"
msckf_cr_ci_str = "$msckf_cr_ci"
msckf_cr_cf_str = "$msckf_cr_cf"
msckf_cc_ci_str = "$msckf_cc_ci"
msckf_cc_cf_str = "$msckf_cc_cf"
msckf_openblas_ci_str = "$msckf_openblas_ci"
msckf_openblas_cf_str = "$msckf_openblas_cf"
shur_total = float("$shur_t")
shur_iters = int("$shur_iters")
shur_ci_str = "$shur_ci"
shur_cf_str = "$shur_cf"

dataset_name = os.environ.get("DATASET_NAME", "?")
arch = os.environ.get("ARCH", "?")
cpu_model = os.environ.get("CPU_MODEL", "?")
cpu_cores = os.environ.get("CPU_CORES", "?")
cpu_cache = os.environ.get("CPU_CACHE", "?")
sched_policy = os.environ.get("SCHED_POLICY", "?")
sched_priority = os.environ.get("SCHED_PRIORITY", "?")
cpu_affinity = os.environ.get("CPU_AFFINITY", "?")
cpu_governor = os.environ.get("CPU_GOVERNOR", "?")

def make_table(headers, rows):
    widths = [max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) for i in range(len(headers))]
    align = ["<"] + [">"] * (len(headers) - 1)
    def sep(char="-"):
        return "+" + "+".join(char * (w + 2) for w in widths) + "+"
    def fmt_row(row):
        return "|" + "|".join(f" {str(row[i]):{align[i]}{widths[i]}} " for i in range(len(headers))) + "|"
    return "\n".join([sep("-"), fmt_row(headers), sep("=")] + [fmt_row(r) for r in rows] + [sep("-")])

results = [
    ("symforce", sym_total, sym_iters, sym_ci_str, sym_cf_str),
    ("ceres_dense", ceres_total, ceres_iters, ceres_ci_str, ceres_cf_str),
    ("ceres_sparse", ceres_sparse_total, ceres_sparse_iters, ceres_sparse_ci_str, ceres_sparse_cf_str),
    ("gtsam_bal", gtsam_bal_total, gtsam_bal_iters, gtsam_bal_ci_str, gtsam_bal_cf_str),
    ("gtsam_smartfactor", gtsam_smart_total, gtsam_smart_iters, gtsam_smart_ci_str, gtsam_smart_cf_str),
    ("msckf_rr", msckf_rr_total, msckf_rr_iters, msckf_rr_ci_str, msckf_rr_cf_str),
    ("msckf_rc", msckf_rc_total, msckf_rc_iters, msckf_rc_ci_str, msckf_rc_cf_str),
    ("msckf_cr", msckf_cr_total, msckf_cr_iters, msckf_cr_ci_str, msckf_cr_cf_str),
    ("msckf_cc", msckf_cc_total, msckf_cc_iters, msckf_cc_ci_str, msckf_cc_cf_str),
    ("shur_solver", shur_total, shur_iters, shur_ci_str, shur_cf_str),
]
if msckf_openblas_total_str and msckf_openblas_iters_str:
    results.append(("msckf_openblas", float(msckf_openblas_total_str), int(msckf_openblas_iters_str), msckf_openblas_ci_str, msckf_openblas_cf_str))
else:
    results.append(("msckf_openblas", None, None, "", ""))
base_per_iter = ceres_total / ceres_iters if ceres_iters > 0 else None
rows = []
for name, total, iters, cost_i_str, cost_f_str in results:
    if total is None or iters is None:
        rows.append([name, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
        continue
    per_iter = total / iters if iters > 0 else float("nan")
    rate = iters / total if total > 0 else float("nan")
    ratio = (per_iter / base_per_iter) if (base_per_iter and base_per_iter > 0) else float("inf")
    cost_i = f"{float(cost_i_str):.3e}" if cost_i_str else "n/a"
    cost_f = f"{float(cost_f_str):.3e}" if cost_f_str else "n/a"
    rows.append([name, f"{total:.3f}", f"{iters:d}", f"{per_iter:.3f}", f"{rate:.2f}", f"{ratio:.3f}x" if base_per_iter and base_per_iter > 0 else "n/a", cost_i, cost_f])

print("\n========== Summary ==========\n")
print(f"[bench] dataset        : {dataset_name}")
print(f"[bench] arch           : {arch}")
print(f"[bench] cpu model      : {cpu_model}")
print(f"[bench] cpu cores      : {cpu_cores}")
print(f"[bench] cpu cache      : {cpu_cache}")
print(f"[bench] sched policy   : {sched_policy}")
print(f"[bench] sched priority : {sched_priority}")
print(f"[bench] cpu affinity   : {cpu_affinity}")
print(f"[bench] cpu governor   : {cpu_governor}")
print(f"\n[bench] summary [{dataset_name}]:")
print(make_table(["engine", "total (s)", "iters", "per_iter (s)", "iter/s", "per_iter ratio", "initial cost", "final cost"], rows))
PY
}

DATASETS=("$@")
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=(
    "problem-16-22106-pre.txt"
    "problem-16-22106-pre_stride=10.txt"
    "problem-16-22106-pre_stride=20.txt"
  )
fi

SCHED_POLICY="${SCHED_POLICY:-unknown}"
SCHED_PRIORITY="${SCHED_PRIORITY:-unknown}"
CPU_AFFINITY="${CPU_AFFINITY:-unknown}"
CPU_GOVERNOR="${CPU_GOVERNOR:-unknown}"

print_section "Environment"
echo "[bench] arch           : $ARCH"
echo "[bench] cpu model      : $CPU_MODEL"
echo "[bench] cpu cores      : $CPU_CORES"
echo "[bench] cpu cache      : $CPU_CACHE"
echo "[bench] sched policy   : $SCHED_POLICY"
echo "[bench] sched priority : $SCHED_PRIORITY"
echo "[bench] cpu affinity   : $CPU_AFFINITY"
echo "[bench] cpu governor   : $CPU_GOVERNOR"
echo "[bench] datasets       : ${DATASETS[*]}"

for dataset in "${DATASETS[@]}"; do
  if [[ ${#DATASETS[@]} -gt 1 ]]; then
    run_one_dataset "$dataset" "$(sanitize_name "$dataset")"
  else
    run_one_dataset "$dataset" ""
  fi
done
