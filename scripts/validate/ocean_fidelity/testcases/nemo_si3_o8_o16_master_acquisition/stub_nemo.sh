#!/usr/bin/env bash
# Checklist-15 executable: exercises run.sh control flow without running NEMO.
set -Eeuo pipefail
if [[ ${O8O16_STUB_FAIL:-0} == 1 ]]; then
  printf 'STUB_REQUESTED_FAILURE\n' >&2
  exit 7
fi
readonly REAL_SAS=${O8O16_REAL_SAS_FIXTURE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round13/refused/oracle_o7_orca2_sas_ice_20261025T172428Z_rc2_3061284}
readonly REAL_RUN=$REAL_SAS/record_2step
[[ -f "$REAL_SAS/o7_admission.json" && -f "$REAL_SAS/RUN_COMPLETE" ]] \
  || { printf 'missing admitted real SAS fixture: %s\n' "$REAL_SAS" >&2; exit 8; }
grep -q '"status": "PASS_O7_ADMISSION"' "$REAL_SAS/o7_admission.json" \
  || { printf 'real SAS fixture is not admitted: %s\n' "$REAL_SAS" >&2; exit 8; }
printf '%s  %s\n' \
  '2aebbdf45ccf3b5d468254ee02166d9af2b8635cf999b8ee35472486c44b5099' "$REAL_RUN/ocean.output" \
  '7c5faa410f726eb6977e2434917f75db681ca1460dd6697373867ed1f0a31dee' "$REAL_RUN/output.namelist.dyn" \
  '2ccc2d012159bcc1d182dc48bf34ff2abdfb09aecb45ff227cd30ebadd9a8603' "$REAL_RUN/output.namelist.ice" \
  | sha256sum -c - >/dev/null
cp "$REAL_RUN/ocean.output" ocean.output
cp "$REAL_RUN/output.namelist.dyn" output.namelist.dyn
cp "$REAL_RUN/output.namelist.ice" output.namelist.ice
# The stub exercises the preregistered 100-step driver.  NEMO's real output is
# retained verbatim above; this final resolved-log override is the only staged value.
printf '\n nn_itend = 100\n' >> ocean.output
printf 'stub restart, observer independent\n' > ORCA2_00000100_restart.nc
if [[ -d o8_o16_master_records ]]; then
  : "${O8O16_GATE:?O8O16_GATE is required by stub NEMO}"
  : "${O8O16_PY:?O8O16_PY is required by stub NEMO}"
  "$O8O16_PY" "$O8O16_GATE" --write-stub-records o8_o16_master_records >/dev/null
fi
printf 'STOP 0\n'
