#!/usr/bin/env bash
# Grid ladder G1, round 4 acquisition: the round-3 acquisition (same deck,
# domain file, stage frames at kt = 1..50) plus a read-only record of the
# split-explicit solver's substeps at kt = 1, 2, 17, 31, 41, built under NEW
# target names.  Round 4 found that both open questions sit inside the
# solver's substep loop: (a) the first row that is not bit-identical to NEMO
# (kt = 1, the external handoff, with a bit-identical right-hand side), whose
# last-place differences are what the tracer update and the equation of state
# amplify into the step-17 interior events; (b) the remaining fold residual,
# first beyond rounding in the handoff at kt = 31 and 41.
# The writer is the campaign's existing one (VORTEX round 196) under G1 names,
# opened only at the listed steps; the patch is that round's patch on NEMO's
# shipped dynspg_ts.F90 plus eight added writes of the solver's Coriolis
# coefficients at loop entry.  Additive lines only (run.sh checks).
# Usage: run_r4.sh [--run|--preflight-only|--admit-existing]   (operator only)
set -Eeuo pipefail
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
export REF_NAME=ORCA2G1_OMIP_L1_R4REF
export RUN_NAME=ORCA2G1_OMIP_L1_P5
export STAGE_STEPS=50
export STEP_PATCH=$here/stprk3_g1_step_record_stage50.patch
export SPGTS_MODULE=$here/g1_spgts_record.F90
export SPGTS_PATCH=$here/dynspg_ts_g1_substep_record.patch
export SPGTS_STEPS=1,2,17,31,41
export EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round4/oracle_g1_r4}
exec "$here/run.sh" "$@"
