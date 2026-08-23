#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3dyngate
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/fv3_dyngate_%j.log
# Module 6's gate needs far more memory than the phase gates: one full
# fv_dynamics step unrolls six faces of remap per k_split iteration, and
# the NH arm at k_split=2 aborted XLA compilation (SIGABRT, rc=134) at
# the 32G the shared phase3d_gate.sbatch allocates. That was a resource
# limit, not a defect -- three of the four cases had already passed.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3jax
[ -e "$REPO/.git" ] || { echo "PREFLIGHT FAILED: REPO=$REPO"; exit 5; }
DIRTY=$(git -C "$REPO" status --porcelain | wc -l)
[ "$DIRTY" -eq 0 ] || { echo "PREFLIGHT FAILED: dirty ($DIRTY)"; exit 3; }
echo "host: $(hostname)"
echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
cd "$REPO"
BT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/_pytest_${SLURM_JOB_ID:-0}
# UNQUOTED on purpose: TESTS may carry several files (word-split is the
# point; a quoted expansion turned two paths into one bogus argument
# and pytest exited 4 with "file or directory not found").
$PY -m pytest -q --basetemp="$BT" ${TESTS:-tests/grids/test_fv3_dynamics.py} 2>&1 | tail -30
echo "    pytest rc=${PIPESTATUS[0]}"
rm -rf "$BT"
echo DYNGATE_DONE
