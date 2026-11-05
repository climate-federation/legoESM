#!/usr/bin/env bash
# Coupled C-ladder profile; real execution requires the exact operator acknowledgement.
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly repo=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly py=/home/dbalwada/legoESM/.venv/bin/python
readonly scratch_root=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_c0_scratch
readonly default_evidence_root=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round30
readonly master=$here/../nemo_si3_o8_o16_master_acquisition/run.sh
export TMPDIR=${TMPDIR:-$scratch_root}
export PYTHONPATH="$repo:$repo/packages/core:$repo/packages/ocean:$repo/packages/atmosphere:$repo/packages/coupler:$repo/packages/ice:$repo/packages/land:$repo/packages/ml:$repo/packages/tools:$repo/src"

refuse() { printf 'REFUSE: %s\n' "$*" >&2; return 1; }

[[ $# -le 2 ]] || { printf 'REFUSE: usage: %s [--preflight-only|--run|--move-aside-builds] [evidence]\n' "$0" >&2; exit 64; }
[[ "$TMPDIR" == /* && -d "$TMPDIR" && -w "$TMPDIR" ]] \
  || { refuse "TMPDIR must be an existing writable absolute directory: $TMPDIR" || :; exit 64; }
[[ "$TMPDIR" != /tmp && "$TMPDIR" != /tmp/* ]] \
  || { refuse 'TMPDIR must not be under /tmp' || :; exit 64; }
[[ -x "$py" ]] || { refuse "required Python is missing: $py" || :; exit 64; }

mode=${1:---preflight-only}
evidence=${2:-}
if [[ "$mode" == --run ]]; then
  evidence_root=${EVIDENCE_ROOT:-$default_evidence_root}
  evidence=${evidence:-$evidence_root/oracle_coupled_c_ladder}
  if [[ ${C_LADDER_STUB:-0} == 1 ]]; then
    [[ "$evidence" =~ ^(.*/si3_rounds/round[0-9]+)(/[^/].*)$ ]] \
      || { refuse 'stub evidence must be under si3_rounds/round<k>/' || :; exit 64; }
  else
    [[ "$evidence" =~ ^(/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round[0-9]+)(/[^/].*)$ ]] \
      || { refuse 'evidence must be under /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round<k>/' || :; exit 64; }
  fi
  evidence_root=${BASH_REMATCH[1]}
else
  evidence_root=${EVIDENCE_ROOT:-$TMPDIR/c_ladder_preflight_root}
fi

runner=(env \
  PYTHON_BIN="$py" O17_EXTENDED=1 O8O16_PROFILE_SCRIPT="$here/c_profile.sh" \
  SOURCE_CFG=ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY SOURCE_EXP=EXP00 \
  NEMO_ROOT="${NEMO_ROOT:-$scratch_root/c_ladder_builds}" \
  NEMO_SOURCE_ROOT="${NEMO_SOURCE_ROOT:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2}" \
  MIN_FREE_RAM_GB="${MIN_FREE_RAM_GB:-150}" MIN_FREE_DISK_KB="${MIN_FREE_DISK_KB:-157286400}" \
  MAX_EVIDENCE_DISK_KB="${MAX_EVIDENCE_DISK_KB:-157286400}" RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-21600}" \
  BASE_BLD="${BASE_BLD:-/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD}" \
  REF_CFG="${REF_CFG:-ORCA2_ORCA1ICE_C_LADDER_REF}" REC_CFG="${REC_CFG:-ORCA2_ORCA1ICE_C_LADDER_REC}" \
  EVIDENCE_ROOT="$evidence_root" \
  "$master" "$mode" "$evidence")
if [[ "$mode" == --preflight-only ]]; then exec timeout 1500 "${runner[@]}"; else exec "${runner[@]}"; fi
