#!/usr/bin/env bash
# Checklist 15/16 executable: only emits names produced by the real O4 build.
set -Eeuo pipefail
if [[ ${O4_STUB_FAIL:-0} == 1 ]]; then
  printf 'STUB_REQUESTED_FAILURE\n' >&2
  exit 7
fi
: "${O4_REAL_SAS_FIXTURE:?retained real SAS fixture is required}"
: "${O4_STUB_WRITER:?independent O4 record writer is required}"
: "${O4_PY:?Python interpreter is required}"
readonly REAL_RUN=$O4_REAL_SAS_FIXTURE/record_2step
[[ -f "$O4_REAL_SAS_FIXTURE/o7_admission.json" && -f "$O4_REAL_SAS_FIXTURE/RUN_COMPLETE" ]] \
  || { printf 'missing admitted real SAS fixture: %s\n' "$O4_REAL_SAS_FIXTURE" >&2; exit 8; }
grep -q '"status": "PASS_O7_ADMISSION"' "$O4_REAL_SAS_FIXTURE/o7_admission.json" \
  || { printf 'real SAS fixture is not admitted\n' >&2; exit 8; }
cp "$REAL_RUN/ocean.output" ocean.output
cp "$REAL_RUN/output.namelist.dyn" output.namelist.dyn
cp "$REAL_RUN/output.namelist.ice" output.namelist.ice
printf '\n ln_dynALL = F\n ln_dynRHGADV = F\n ln_dynADV1D = F\n ln_dynADV2D = T\n ln_adv_Pra = T\n ln_adv_UMx = F\n nn_itend = 2\n nn_stock = 2\n' >> ocean.output
# Both names are observed in the admitted two-step SAS run and are the exact
# names NEMO produces at nn_stock=nn_itend=2.
printf 'stub ocean restart derived from retained SAS run\n' > ORCA2_SAS_00000002_restart.nc
printf 'stub ice restart derived from retained SAS run\n' > ORCA2_SAS_00000002_restart_ice.nc
if [[ -d o4_prather_records ]]; then
  "$O4_PY" "$O4_STUB_WRITER" o4_prather_records --source-real "$REAL_RUN/ocean.output"
fi
printf 'STOP 0\n' >&2
