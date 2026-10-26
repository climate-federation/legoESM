#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100"
#SBATCH --job-name=ls_restart
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/ls_restart_%j.out
set +e
# Does the momentum restart actually rescue a stalled fit, and does the
# instrument behave on an arm where it must NOT fire?
#
# PRE-REGISTERED, both arms, before the job ran:
#
#   mynn25       stalled at step 3 of 40 in job 9414301 after two accepted
#                steps, so its Adam momentum buffer holds two steps of history
#                by then.  CONFIRMS the momentum diagnosis if the printed
#                dL at that step is POSITIVE (uphill) and the retry then
#                accepts.  REFUTES it if dL is NEGATIVE -- the direction was
#                downhill and the ladder simply could not find a reducing
#                length, which is a curvature/resolution problem that this
#                change does not address and must not be claimed to.
#
#   smagorinsky  stalled at step 1, where Adam's bias-corrected momentum IS
#                the current gradient.  Its dL is therefore negative by
#                construction and the retry MUST NOT fire.  This is the
#                control: a restart reported here means the sign convention in
#                the dot product is wrong, and the mynn25 number is then worth
#                nothing either.
#
# Cheapest form of the question: 2 schemes, 6 steps, the same 7 cases and the
# same flags as the campaign, so the only difference from job 9414301 is the
# driver commit.  Everything else -- nlev, aggregation, surface handover,
# tier, lr, optimizer -- is copied from tune_all_cases_joint.sbatch.
: "${E_WT:?set E_WT to the pinned worktree}"
WT=$E_WT
cd "$WT" || exit 1
PIN=$(git rev-parse HEAD)
echo "PINNED sha: $PIN"; echo "host: $(hostname)"; nvidia-smi -L
if [ -n "$EXPECT_SHA" ] && [ "$PIN" != "$EXPECT_SHA" ]; then
  echo "worktree is at $PIN, expected $EXPECT_SHA -- refusing"; exit 4
fi
export PYTHONPATH=$WT/src:$(ls -d "$WT"/packages/*/ | tr '\n' ':')$PYTHONPATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results
OUT=${OUT:-$R/scm_les_turbulence/_ls_restart_check}

echo "=== unit: the verdict table and its premise ==="
$PY -m pytest -q \
    tests/atmosphere/hydrostatic/test_run_scm_les_turbulence_tuning.py \
    -k "line_search_verdict or stale_momentum" 2>&1 | tail -12
echo "  pytest rc=$?"

CASE_LIST="ekman gabls1 cbl bomex rico astex dycoms"
CASES=""
for c in $CASE_LIST; do
  [ -d "$R/les_ref/$c" ] || { echo "MISSING $c"; exit 2; }
  CASES="${CASES:+$CASES,}$c:$R/les_ref/$c"
done

rm -rf "$OUT"
t0=$SECONDS
$PY scripts/run/run_scm_les_turbulence_tuning.py \
    --cases "$CASES" --outdir "$OUT" --nlev 48 --steps "${STEPS:-6}" \
    --schemes mynn25,smagorinsky \
    --joint-aggregation default_relative --surface-flux-to-closure
echo "driver rc=$? wall=$((SECONDS - t0))s"
echo "=== ranking ==="; cat "$OUT/ranking.csv"
echo "DONE $(date +%T)"
