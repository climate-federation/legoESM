#!/usr/bin/env bash
# O17 acquisition tool. Default is read-only; only explicit --run invokes NEMO.
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly repo=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly py=${PYTHON_BIN:-/home/dbalwada/legoESM/.venv/bin/python3}
readonly master=$here/../nemo_si3_o8_o16_master_acquisition/run.sh
readonly scorer=$here/../nemo_si3_o17_composition_scorer.py
readonly default_root=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step
readonly default_fixture=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round18/oracle_o8_o16_master
export PYTHONPATH="$repo:$repo/packages/core:$repo/packages/ocean:$repo/packages/ice:$repo/packages/atmosphere:$repo/packages/land:$repo/packages/coupler:$repo/packages/driver"
read -r o17_record_bytes o17_disk_kib <<< \
  "$("$py" "$here/../nemo_si3_o17_record_budget.py" --shell)"
[[ "$o17_record_bytes" =~ ^[0-9]+$ && "$o17_disk_kib" =~ ^[0-9]+$ ]] \
  || { printf 'REFUSE: O17 record budget calculator failed\n' >&2; exit 64; }
readonly o17_record_bytes o17_disk_kib

refuse() { printf 'REFUSE: %s\n' "$*" >&2; return 1; }
expect_refusal() { if "$@" >/dev/null 2>&1; then refuse "plant survived: $*"; fi; }

preflight() {
  local scratch
  scratch=$(mktemp -d "${TMPDIR:-/tmp}/si3_o17_preflight.XXXXXX")
  trap 'rm -rf -- "$scratch"' RETURN
  "$py" "$here/preflight.py" --repo "$repo" > "$scratch/structure.json"
  expect_refusal "$py" "$here/preflight.py" --repo "$repo" --plant swap-rheology-advection
  expect_refusal "$py" "$here/preflight.py" --repo "$repo" --plant drop-redistribution
  "$py" "$scorer" --schema > "$scratch/schema.json"
  # Real scorer on a writer-shaped ORCA2 record built from the model (2 steps):
  # identical -> PASS; one-ULP t_si and signed-zero plants -> FAIL.
  "$py" "$here/../nemo_si3_o17_synthetic_record.py" --out "$scratch/synthetic" --steps 2 \
    > "$scratch/synthetic.json"
  grep -q '"status": "PASS_O17_SYNTHETIC_RECORD_SCORER"' "$scratch/synthetic.json"
  "$py" "$here/preflight.py" --repo "$repo" --numerical --master "$default_fixture" \
    > "$scratch/numerical.json"
  "$py" "$here/preflight.py" --repo "$repo" --numerical --master "$default_fixture" \
    --numerical-plant swap-rheology-advection > "$scratch/swap.json"
  "$py" "$here/preflight.py" --repo "$repo" --numerical --master "$default_fixture" \
    --numerical-plant drop-deformation-fold > "$scratch/fold.json"
  O17_STUB_SCORE_JSON="$scratch/synthetic/identical.json" \
    O17_STUB_FAIL_SCORE_JSON="$scratch/synthetic/one_ulp_t_si.json" \
    O17_EXTENDED=1 EVIDENCE_ROOT="$scratch/root" EVIDENCE="$scratch/evidence" \
    REFUSED_DIR="$scratch/refused" REF_CFG="O17_PREFLIGHT_${BASHPID}_REF" \
    REC_CFG="O17_PREFLIGHT_${BASHPID}_REC" "$master" --preflight-only
  printf 'PASS_O17_UNIFIED_MASTER_PREFLIGHT_AND_NUMERICAL_PLANTS\n'
}

mode=${1:---preflight-only}
evidence=${2:-$default_root}
case "$mode" in
  --preflight-only)
    preflight
    ;;
  --run)
    [[ ${O17_OPERATOR_ACK:-} == RUN_UNMODIFIED_ORCA2_SAS_ICE_100X10800 ]] \
      || refuse 'set O17_OPERATOR_ACK=RUN_UNMODIFIED_ORCA2_SAS_ICE_100X10800'
    [[ "$evidence" =~ /si3_rounds/round[0-9]+/ ]] \
      || refuse 'evidence must be under si3_rounds/round<k>/'
    O17_EXTENDED=1 EVIDENCE="$evidence" \
      REF_CFG=${O17_REF_CFG:-ORCA2_SAS_ICE_O17_REF} \
      REC_CFG=${O17_REC_CFG:-ORCA2_SAS_ICE_O17_REC} \
      RUN_TIMEOUT_SEC=${O17_TIMEOUT_SECONDS:-21600} \
      MIN_FREE_RAM_GB=${O17_MIN_FREE_RAM_GB:-180} \
      MIN_FREE_DISK_KB=${O17_MIN_FREE_DISK_KB:-$o17_disk_kib} \
      MAX_EVIDENCE_DISK_KB=${O17_MAX_EVIDENCE_DISK_KB:-$o17_disk_kib} \
      O17_RECORD_BYTES_GATE=$o17_record_bytes \
      "$master" --run "$evidence"
    ;;
  --admit-existing)
    O17_EXTENDED=1 EVIDENCE="$evidence" "$master" --admit-existing "$evidence"
    ;;
  --move-aside-builds)
    O17_EXTENDED=1 REF_CFG=${O17_REF_CFG:-ORCA2_SAS_ICE_O17_REF} \
      REC_CFG=${O17_REC_CFG:-ORCA2_SAS_ICE_O17_REC} "$master" --move-aside-builds
    ;;
  *)
    refuse "usage: $0 [--preflight-only|--run|--admit-existing|--move-aside-builds] [path]"
    ;;
esac
