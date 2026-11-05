#!/usr/bin/env bash
# Decision-independent C-ladder foundation. This script cannot build or run NEMO.
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly repo=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly py=/home/dbalwada/legoESM/.venv/bin/python
readonly scratch_root=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_c0_scratch
readonly master=$here/../nemo_si3_o8_o16_master_acquisition/run.sh
export TMPDIR=${TMPDIR:-$scratch_root}
export PYTHONPATH="$repo:$repo/packages/core:$repo/packages/ocean:$repo/packages/atmosphere:$repo/packages/coupler:$repo/packages/ice:$repo/packages/land:$repo/packages/ml:$repo/packages/tools:$repo/src"

refuse() { printf 'REFUSE: %s\n' "$*" >&2; return 1; }
expect_refusal() { if timeout 1500 "$@" >/dev/null 2>&1; then refuse "plant survived: $*"; fi; }

[[ ${1:---preflight-only} == --preflight-only && $# -le 1 ]] \
  || { refuse "usage: $0 [--preflight-only]"; exit 64; }
[[ "$TMPDIR" == "$scratch_root" || "$TMPDIR" == "$scratch_root/"* ]] \
  || { refuse "TMPDIR must be under $scratch_root"; exit 64; }
[[ -x "$py" ]] || { refuse "required Python is missing: $py"; exit 64; }

scratch=$(mktemp -d "$TMPDIR/c_ladder_preflight.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT

timeout 1500 "$py" "$here/preflight.py" > "$scratch/source_contract.json"
grep -q '"status": "PASS_COUPLED_C_SOURCE_CONTRACT"' "$scratch/source_contract.json" \
  || refuse "coupled source contract did not pass"
for plant in drop-sbc-ssm swap-flux-update pretend-mlf; do
  expect_refusal "$py" "$here/preflight.py" --plant "$plant"
done
printf 'PASS_COUPLED_C_SOURCE_CONTRACT_AND_PLANTS\n'

# Reuse the admitted master framework for real-ppsrc observer placement,
# same-program passivity, stub-NEMO execution, retention, and its own plants.
# PYTHON_BIN keeps the operator's required interpreter through the delegation.
timeout 1500 env PYTHON_BIN="$py" O17_EXTENDED=0 "$master" --preflight-only \
  > "$scratch/si3_master.log"
grep -q '^PASS_100_STEP_STUB_DRIVER_RETENTION$' "$scratch/si3_master.log" \
  || { tail -40 "$scratch/si3_master.log" >&2; refuse "delegated SI3 master preflight failed"; }
printf 'PASS_REUSED_REAL_PPSRC_OBSERVER_AND_STUB_NEMO_PREFLIGHT\n'
printf 'PREFLIGHT_OK: no build and no NEMO integration were run\n'
