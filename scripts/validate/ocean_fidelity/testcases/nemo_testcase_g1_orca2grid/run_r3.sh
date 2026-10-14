#!/usr/bin/env bash
# Grid ladder G1, round 3 acquisition: the SAME deck, domain file and patches
# as run.sh, with stage frames at kt = 1..50 instead of 1..10, built under a
# NEW target name.  Round 3 needs NEMO's stage frames (a) at kt 15..24, where
# single open-ocean sea-level cells leave the 1e-15 bar inside one step, and
# (b) at kt 31..41, where the fold band's amplitude is large enough for the
# remaining fold statement to stand above rounding.  The only source change is
# the writer's gate (kstp <= nit000 + 9 -> + 49, five places); the patch is
# stprk3_g1_step_record_stage50.patch, derived by that one substitution.
# Usage: run_r3.sh [--run|--preflight-only|--admit-existing]   (operator only)
set -Eeuo pipefail
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
if [[ "$(sed 's/kstp <= nit000 + 9 )/kstp <= nit000 + 49 )/' "$here/stprk3_g1_step_record.patch" | sha256sum)" \
      != "$(sha256sum <"$here/stprk3_g1_step_record_stage50.patch")" ]]; then
  printf 'REFUSE: the stage-50 patch is not the round-1 patch with its gate moved\n' >&2
  exit 68
fi
# run.sh refuses a target that already exists, so the reference binary is
# rebuilt under its own new name too (same sources as round 1's reference).
export REF_NAME=ORCA2G1_OMIP_L1_R3REF
export RUN_NAME=ORCA2G1_OMIP_L1_P4
export STAGE_STEPS=50
export STEP_PATCH=$here/stprk3_g1_step_record_stage50.patch
export EVIDENCE=${EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round3/oracle_g1_r3}
exec "$here/run.sh" "$@"
