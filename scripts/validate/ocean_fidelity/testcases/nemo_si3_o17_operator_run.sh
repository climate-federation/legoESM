#!/usr/bin/env bash
# Operator runner for SI3 O17 composition scorer (DECISION 210):
#   (A) K parallel step ranges on the real 100-step NEMO record followed by merge
#   (B) 100-step carried-state free run from step 1's recorded entry state
# All outputs and scratch are written under the operator-supplied output directory;
# the evidence directory is read-only and never modified.
set -euo pipefail

readonly HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO="$(cd "$HERE/../../../.." && pwd)"
readonly SCORER="$HERE/nemo_si3_o17_composition_scorer.py"
readonly PY="/home/dbalwada/legoESM/.venv/bin/python"
readonly DEFAULT_EVIDENCE="/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step"

OUT_DIR=""
K_PARALLEL=20
N_STEPS=100
EVIDENCE="$DEFAULT_EVIDENCE"
MIN_AVAIL_GB=250
MAX_PROC_RSS_KIB=104857600  # 100 GiB in KiB

usage() {
  cat <<EOF
Usage:
  $0 <OUT_DIR> [K_PARALLEL=20] [N_STEPS=100] [EVIDENCE_DIR]
  $0 --out <OUT_DIR> [--parallel K] [--steps N] [--evidence DIR]
EOF
}

positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    --parallel|-k)
      K_PARALLEL="$2"
      shift 2
      ;;
    --steps|-n)
      N_STEPS="$2"
      shift 2
      ;;
    --evidence)
      EVIDENCE="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "REFUSE: unknown option $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done

if [[ ${#positional[@]} -ge 1 && -z "$OUT_DIR" ]]; then
  OUT_DIR="${positional[0]}"
fi
if [[ ${#positional[@]} -ge 2 ]]; then
  K_PARALLEL="${positional[1]}"
fi
if [[ ${#positional[@]} -ge 3 ]]; then
  N_STEPS="${positional[2]}"
fi
if [[ ${#positional[@]} -ge 4 ]]; then
  EVIDENCE="${positional[3]}"
fi

if [[ -z "$OUT_DIR" ]]; then
  echo "REFUSE: output directory is required" >&2
  usage >&2
  exit 2
fi
if [[ ! -d "$EVIDENCE/record_100step/o17_master_records" ]]; then
  echo "REFUSE: missing O17 record directory under $EVIDENCE" >&2
  exit 2
fi
if [[ "$K_PARALLEL" -lt 1 || "$N_STEPS" -lt 1 || "$K_PARALLEL" -gt "$N_STEPS" ]]; then
  echo "REFUSE: require 1 <= K_PARALLEL ($K_PARALLEL) <= N_STEPS ($N_STEPS)" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
readonly ABS_OUT="$(cd "$OUT_DIR" && pwd)"
readonly ABS_EVIDENCE="$(cd "$EVIDENCE" && pwd)"
if [[ "$ABS_OUT" == "$ABS_EVIDENCE" || "$ABS_OUT" == "$ABS_EVIDENCE/"* ]]; then
  echo "REFUSE: output directory $ABS_OUT must not be inside read-only evidence $ABS_EVIDENCE" >&2
  exit 2
fi

check_memory_gate() {
  local avail_gb
  avail_gb="$(free -g | awk '/^Mem:/ {print $7}')"
  if [[ -z "$avail_gb" || "$avail_gb" -lt "$MIN_AVAIL_GB" ]]; then
    echo "REFUSE: available memory ${avail_gb:-0} GiB < required ${MIN_AVAIL_GB} GiB" >&2
    exit 2
  fi
  local heavy
  heavy="$(ps -eo pid=,rss=,comm= | awk -v lim="$MAX_PROC_RSS_KIB" '$2 > lim {printf "pid=%s rss_kib=%s comm=%s; ", $1, $2, $3}')"
  if [[ -n "$heavy" ]]; then
    echo "REFUSE: process above 100 GiB RSS detected ($heavy)" >&2
    exit 2
  fi
}

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export TMPDIR="$ABS_OUT/tmp"
export PYTHONPATH="$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src"

mkdir -p "$TMPDIR" "$ABS_OUT/ranges"

# (A) Parallel step ranges + merge
check_memory_gate

pids=()
range_jsons=()
for ((i = 0; i < K_PARALLEL; i++)); do
  start=$(( i * N_STEPS / K_PARALLEL + 1 ))
  stop=$(( (i + 1) * N_STEPS / K_PARALLEL ))
  tag="$(printf 'range_%04d_%04d' "$start" "$stop")"
  rjson="$ABS_OUT/ranges/${tag}.json"
  rlog="$ABS_OUT/ranges/${tag}.log"
  range_jsons+=("$rjson")
  "$PY" "$SCORER" \
    --evidence "$ABS_EVIDENCE" \
    --execution eager \
    --expect-steps "$N_STEPS" \
    --steps "${start}..${stop}" \
    --out "$rjson" >"$rlog" 2>&1 &
  pids+=("$!")
done

range_rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    range_rc=2
  fi
done

merged_rc=0
"$PY" "$SCORER" \
  --expect-steps "$N_STEPS" \
  --merge "${range_jsons[@]}" \
  --out "$ABS_OUT/parallel_merged_report.json" >"$ABS_OUT/parallel_merged.log" 2>&1 || merged_rc=$?

# (B) Carried-state free run for N_STEPS
check_memory_gate

free_rc=0
"$PY" "$SCORER" \
  --evidence "$ABS_EVIDENCE" \
  --execution eager \
  --expect-steps "$N_STEPS" \
  --free-run \
  --n-steps "$N_STEPS" \
  --out "$ABS_OUT/free_run_report.json" >"$ABS_OUT/free_run.log" 2>&1 || free_rc=$?

if [[ "$range_rc" -ne 0 || "$merged_rc" -ne 0 || "$free_rc" -ne 0 ]]; then
  exit 2
fi
