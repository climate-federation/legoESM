#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3p3dgate
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/fv3_phase3d_gate_%j.log
# The 3-D layer's own gates, without the 30-minute whole-lane measurement.
#
# Runs the phase modules' test files plus the shared-helper one, so a change to
# fv3_phase3d_common or to a phase module can be checked in minutes. The full
# jax_lane_measure job stays the release gate; this is the edit loop.
#
# TESTS may be overridden:
#   sbatch --export=ALL,TESTS="tests/grids/test_fv3_dsw_tail_3d.py" phase3d_gate.sbatch
set -uo pipefail
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
  _SELF=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
          | tr ' ' '\n' | sed -n 's/^Command=//p' | head -1)
fi
[ -n "${_SELF:-}" ] || _SELF="${BASH_SOURCE[0]}"
REPO=$(cd "$(dirname "$_SELF")/../../.." && pwd)
[ -e "$REPO/.git" ] || { echo "PREFLIGHT FAILED: REPO=$REPO"; exit 5; }
DIRTY=$(git -C "$REPO" status --porcelain | wc -l)
[ "$DIRTY" -eq 0 ] || { echo "PREFLIGHT FAILED: $REPO dirty ($DIRTY)"; exit 3; }
echo "host: $(hostname)"
echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_ENABLE_X64=1
cd "$REPO"
# TRAP (earned): SLURM sets TMPDIR=/local and it gets reaped mid-job, so
# --basetemp is forced onto a burg path.
BT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/_pytest_${SLURM_JOB_ID:-0}
DEFAULT_TESTS="tests/grids/test_fv3_phase3d_common.py \
tests/grids/test_fv3_cgrid_phase_3d.py \
tests/grids/test_fv3_dsw_phase_3d.py \
tests/grids/test_fv3_gradient_gate_triage.py"
for f in ${TESTS:-$DEFAULT_TESTS}; do
  echo "--- $f"
  $PY -m pytest -x -q --basetemp="$BT" "$f" 2>&1 | tail -25
  echo "    pytest rc=${PIPESTATUS[0]}"
done
echo PHASE3D_GATE_DONE
