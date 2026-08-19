#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=mynncbl
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/mynncbl_%j.out
set +e
# mynn25 is the ONE closure the campaign cannot rank. Measured by
# diag_scheme_blowup at the campaign's own defaults: finite through 2000 steps
# on all seven other cases, NON-FINITE at step 10 on cbl, all 48 levels at
# once. Every other closure survives cbl. So the pair (mynn25, cbl) is the
# defect.
#
# ROUND 1 (job 9408248) established, with one variable per arm:
#   * it is EXPLOSIVE GROWTH, not a 0/0 -- T max runs 299.87 -> 1.0e41 (step 4)
#     -> 4.1e62 -> NaN (step 10);
#   * dt 10 s -> 1 s does NOT fix it (blows at t = 21 s instead of t = 40 s),
#     so it is not a timestep-CFL violation;
#   * --surface-flux-to-closure delays it to t ~ 1300 s but does not fix it.
# and it exposed a defect in the PROBE: it watched PhysicsState.tke, while
# mynn25's carry is the separate `qke` slot, so the carry column read a
# constant zero for reasons that had nothing to do with the scheme.
#
# ROUND 2 asks the question round 1 could not: does the CARRY (qke) run away
# before the state does -- a closure feedback -- or does the state go first?
# And WHERE in the column, which separates a surface-layer feedback from a lid
# one. cbl is the only case with u_geo = v_geo = 0 and f_c = 0, so the resolved
# shear is exactly zero.
: "${M_WT:?set M_WT to the pinned worktree}"
WT=$M_WT
cd "$WT" || exit 1
echo "PINNED sha: $(git rev-parse HEAD)"; echo "host: $(hostname)"
export PYTHONPATH=$WT/src:$(ls -d "$WT"/packages/*/ | tr '\n' ':')$PYTHONPATH
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
P=scripts/validate/diag_scheme_blowup.py

echo "=============== A. mynn25 / cbl, every step, full columns ==============="
$PY $P --case cbl --scheme mynn25 --nlev 48 --steps 12 --print-every 1 \
    --dump-profiles 2>&1 | tail -40
echo "rc=${PIPESTATUS[0]}"

echo "=============== B. CONTROL tke / cbl, same window ==============="
# The closest sibling that SURVIVES cbl, at the same steps and the same dump,
# so the mynn25 columns can be read against a healthy one rather than against
# an expectation.
$PY $P --case cbl --scheme tke --nlev 48 --steps 12 --print-every 6 \
    --dump-profiles 2>&1 | tail -22
echo "rc=${PIPESTATUS[0]}"

echo "=============== C. CONTROL mynn25 / ekman, same window ==============="
# The case mynn25 SURVIVES that is closest to cbl: also neutral-ish and dry,
# but with a geostrophic wind and rotation, so the shear is NOT zero. If the
# qke column here looks nothing like cbl's at the same step, zero shear is the
# variable.
$PY $P --case ekman --scheme mynn25 --nlev 48 --steps 12 --print-every 6 \
    --dump-profiles 2>&1 | tail -22
echo "rc=${PIPESTATUS[0]}"
echo "DONE $(date +%T)"
