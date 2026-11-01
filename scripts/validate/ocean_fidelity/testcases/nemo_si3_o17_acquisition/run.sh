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

refuse() { printf 'REFUSE: %s\n' "$*" >&2; return 1; }
expect_refusal() { if "$@" >/dev/null 2>&1; then refuse "plant survived: $*"; fi; }

preflight() {
  local scratch
  scratch=$(mktemp -d /tmp/si3_o17_preflight.XXXXXX)
  trap 'rm -rf -- "$scratch"' RETURN
  "$py" "$here/preflight.py" --repo "$repo" > "$scratch/structure.json"
  expect_refusal "$py" "$here/preflight.py" --repo "$repo" --plant swap-rheology-advection
  expect_refusal "$py" "$here/preflight.py" --repo "$repo" --plant drop-redistribution
  "$py" "$scorer" --schema > "$scratch/schema.json"
  "$py" "$here/preflight.py" --repo "$repo" --numerical --master "$default_fixture" \
    > "$scratch/numerical.json"
  "$py" "$here/preflight.py" --repo "$repo" --numerical --master "$default_fixture" \
    --numerical-plant swap-rheology-advection > "$scratch/swap.json"
  "$py" "$here/preflight.py" --repo "$repo" --numerical --master "$default_fixture" \
    --numerical-plant drop-deformation-fold > "$scratch/fold.json"
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
      MIN_FREE_DISK_KB=${O17_MIN_FREE_DISK_KB:-300000000} \
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
