#!/usr/bin/env bash
# Operator-only kt=2 extension of the reviewed round-331/332 TKE recorder.
set -Eeuo pipefail

readonly MODE=${1:---preflight-only}
case "$MODE" in
  --preflight-only|--run|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--preflight-only|--run|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly here
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PARENT=$here/../nemo_testcase_l4_orca2_round332_source_order_acquisition/run.sh
readonly GATE=$here/../nemo_testcase_l4_orca2_round336_tke_gate.py
readonly TEST=$REPO/tests/ocean/fidelity/test_orca2_round336_tke_gate.py
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/acquisition
readonly TARGET_RUN=$EVIDENCE/orca1ice_tke_internal_kt2_np2
readonly PHASE_REFERENCE=$EVIDENCE/orca1ice_trazdf_internal_kt2_np2
readonly PY=/home/dbalwada/legoESM/.venv/bin/python

refuse() { printf 'REFUSE: %s\n' "$*" >&2; exit 65; }
for path in "$PARENT" "$GATE" "$TEST" "$PHASE_REFERENCE/producer_commit.txt"; do
  [[ -f "$path" ]] || refuse "missing round-336 TKE acquisition input $path"
done
bash -n "$PARENT" || refuse 'inherited round-332 driver has invalid shell syntax'
bash -n "$0" || refuse 'round-336 TKE wrapper has invalid shell syntax'

export R328_ACQ_TARGET=tke332
export R328_TARGET_CFG_OVERRIDE=ORCA2_ORCA1ICE_OMIP_L4_R336TKE
export R328_EVIDENCE_OVERRIDE=$EVIDENCE
export R328_TARGET_RUN_NAME_OVERRIDE=orca1ice_tke_internal_kt2_np2

if [[ "$MODE" == --preflight-only ]]; then
  export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
  "$PY" -m pytest -q -p no:cacheprovider "$TEST"
fi

"$PARENT" "$MODE"

if [[ "$MODE" == --preflight-only ]]; then
  printf 'ORCA2_ROUND336_TKE_KT2_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

[[ -d "$TARGET_RUN" ]] || refuse 'inherited driver produced no canonical kt=2 run'
producer=$(cat "$TARGET_RUN/producer_commit.txt")
[[ $producer =~ ^[0-9a-f]{40}$ ]] || refuse 'kt=2 producer manifest is malformed'
if [[ "$MODE" == --admit-existing ]]; then
  [[ -f "$TARGET_RUN/round336_tke_extension_toolchain.sha256" ]] \
    || refuse 'retained kt=2 run lacks its extension toolchain manifest'
  ( cd "$REPO" && sha256sum -c "$TARGET_RUN/round336_tke_extension_toolchain.sha256" >/dev/null ) \
    || refuse 'retained kt=2 extension toolchain changed'
else
  ( cd "$REPO" && sha256sum \
      "${0#$REPO/}" "${GATE#$REPO/}" "${TEST#$REPO/}" ) \
    >"$TARGET_RUN/round336_tke_extension_toolchain.sha256"
fi
for plant in header field-order truncation checksum stamp final mxl pdlr time-level; do
  log=$TARGET_RUN/round336_tke_${plant}_plant.log
  if "$PY" "$GATE" --record-dir "$TARGET_RUN" \
      --phase-reference-dir "$PHASE_REFERENCE" --expect-commit "$producer" \
      --plant "$plant" >"$log" 2>&1; then
    refuse "$plant round-336 kt=2 TKE plant stayed green"
  fi
  grep 'REFUSE:' "$log" >/dev/null \
    || refuse "$plant round-336 kt=2 TKE plant missed its refusal"
done
"$PY" "$GATE" --record-dir "$TARGET_RUN" \
  --phase-reference-dir "$PHASE_REFERENCE" --expect-commit "$producer" \
  --output "$TARGET_RUN/round336_tke_kt2_admission.json"
( cd "$TARGET_RUN" && sha256sum oracle_tke331_kt00000002_r*.bin \
    oracle_r336_tke_mxl_kt00000002_r*.bin round336_tke_kt2_admission.json \
    round336_tke_*_plant.log >round336_tke_kt2_outputs.sha256 )
printf 'ORCA2_ROUND336_TKE_KT2_ACQUISITION_PASS %s\n' "$TARGET_RUN"
