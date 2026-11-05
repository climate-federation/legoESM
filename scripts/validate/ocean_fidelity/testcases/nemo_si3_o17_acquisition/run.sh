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

synthetic_scores() {
  # The master's stub driver (its preflight step 10) scores with these reports:
  # a writer-shaped ORCA2 record built from the model and scored by the real
  # scorer (identical -> PASS; one-ULP t_si and signed-zero plants -> FAIL).
  local dir=$1
  "$py" "$here/../nemo_si3_o17_synthetic_record.py" --out "$dir/synthetic" --steps 2 \
    > "$dir/synthetic.json"
  grep -q '"status": "PASS_O17_SYNTHETIC_RECORD_SCORER"' "$dir/synthetic.json" \
    || refuse 'synthetic writer-shaped record did not score PASS/FAIL as planted'
  export O17_STUB_SCORE_JSON="$dir/synthetic/identical.json"
  export O17_STUB_FAIL_SCORE_JSON="$dir/synthetic/one_ulp_t_si.json"
}

run_master() {
  # The one --run entry, shared by the operator and the preflight dry run.
  O17_EXTENDED=1 EVIDENCE="$1" \
    REF_CFG=${O17_REF_CFG:-ORCA2_SAS_ICE_O17_REF} \
    REC_CFG=${O17_REC_CFG:-ORCA2_SAS_ICE_O17_REC} \
    RUN_TIMEOUT_SEC=${O17_TIMEOUT_SECONDS:-21600} \
    MIN_FREE_RAM_GB=${O17_MIN_FREE_RAM_GB:-180} \
    MIN_FREE_DISK_KB=${O17_MIN_FREE_DISK_KB:-$o17_disk_kib} \
    MAX_EVIDENCE_DISK_KB=${O17_MAX_EVIDENCE_DISK_KB:-$o17_disk_kib} \
    O17_RECORD_BYTES_GATE=$o17_record_bytes \
    "$master" --run "$1"
}

preflight() {
  local scratch
  scratch=$(mktemp -d "${TMPDIR:-/tmp}/si3_o17_preflight.XXXXXX")
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
  local -a dry=(O17_OPERATOR_ACK=RUN_UNMODIFIED_ORCA2_SAS_ICE_100X10800
    O8O16_STOP_BEFORE_BUILD=1 EVIDENCE_ROOT="$scratch/root" REFUSED_DIR="$scratch/refused")
  # Plant: 2b1921abc4's --run was run_master without the stub scores; it must refuse.
  if (unset O17_STUB_SCORE_JSON O17_STUB_FAIL_SCORE_JSON; export "${dry[@]}"
      run_master "$scratch/si3_rounds/round0/plant") > "$scratch/plant.log" 2>&1; then
    refuse 'master --run without the stub score reports was not refused'
  fi
  grep -q 'stub run needs the real scorer report O17_STUB_SCORE_JSON' "$scratch/plant.log" \
    || { cat "$scratch/plant.log" >&2; refuse 'stub-score plant was refused for another reason'; }
  # The exact operator --run branch (same env, branch and master entry), stopped
  # by the master after its full preflight and resource gates, before any build.
  env "${dry[@]}" "$0" --run "$scratch/si3_rounds/round0/oracle_o17_dry" > "$scratch/dry_run.log" 2>&1 \
    || { tail -20 "$scratch/dry_run.log" >&2; refuse 'dry --run failed'; }
  grep -E '^(PASS_|PREFLIGHT_OK|DRY_RUN_)' "$scratch/dry_run.log"
  grep -q '^DRY_RUN_STOPPED_BEFORE_BUILD_AND_NEMO: ' "$scratch/dry_run.log" \
    || refuse 'dry --run did not stop before build and NEMO'
  printf 'PASS_O17_UNIFIED_MASTER_PREFLIGHT_AND_NUMERICAL_PLANTS\n'
}

with_synthetic_scores() {
  local work rc=0
  work=$(mktemp -d "${TMPDIR:-/tmp}/si3_o17_scores.XXXXXX")
  synthetic_scores "$work"
  "$@" || rc=$?
  if [[ "$rc" -eq 0 ]]; then rm -rf -- "$work"; else printf 'O17_SCORES_RETAINED: %s\n' "$work" >&2; fi
  return "$rc"
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
    with_synthetic_scores run_master "$evidence"
    ;;
  --admit-existing)
    with_synthetic_scores env O17_EXTENDED=1 EVIDENCE="$evidence" "$master" --admit-existing "$evidence"
    ;;
  --move-aside-builds)
    O17_EXTENDED=1 REF_CFG=${O17_REF_CFG:-ORCA2_SAS_ICE_O17_REF} \
      REC_CFG=${O17_REC_CFG:-ORCA2_SAS_ICE_O17_REC} "$master" --move-aside-builds
    ;;
  *)
    refuse "usage: $0 [--preflight-only|--run|--admit-existing|--move-aside-builds] [path]"
    ;;
esac
