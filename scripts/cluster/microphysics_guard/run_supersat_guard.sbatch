#!/bin/bash
#SBATCH --job-name=ss_guard
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM-ssguard-runs/ss_guard_%j.log
# Uniform micro-physics super-saturation guard: calibration probe + test suites.
#
# ANTI-CONFOUND: runs from a DETACHED worktree pinned to an explicit SHA, so an
# edit in any other checkout during the run cannot change what was measured.
# The pinned SHA is echoed first.
set -uo pipefail

PINNED_SHA="${PINNED_SHA:?set PINNED_SHA=<sha> before sbatch}"
SRC=/burg-archive/glab/users/pg2328/legoESM
WT="/burg-archive/glab/users/pg2328/legoESM-ssguard-runs/wt_${SLURM_JOB_ID}"
PROBE_SRC="${PROBE_SRC:-/burg-archive/glab/users/pg2328/legoESM-ssguard/scripts/tmp/_probe_supersat_guard.py}"
PY="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"

mkdir -p "$(dirname "$WT")"
git -C "$SRC" worktree add --detach "$WT" "$PINNED_SHA" >/dev/null 2>&1 || {
  echo "WORKTREE ADD FAILED"; exit 1; }
echo "PINNED sha=$(git -C "$WT" rev-parse HEAD)"
echo "PINNED describe=$(git -C "$WT" log --oneline -1)"

cd "$WT"
_pp="$WT"
for _d in "$WT"/packages/*/; do
  [ -d "${_d}legoesm" ] && _pp="${_pp}:${_d}"
done
export PYTHONPATH="${_pp}:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

# The probe is gitignored (scripts/tmp/_probe_*.py); copy it in explicitly.
mkdir -p "$WT/scripts/tmp"
cp "$PROBE_SRC" "$WT/scripts/tmp/_probe_supersat_guard.py"

echo "===== PROBE: RH decay from 1.4, hard OFF vs ON ====="
"$PY" "$WT/scripts/tmp/_probe_supersat_guard.py"
echo "PROBE_RC=$?"

echo "===== RATCHET: tests/test_microphysics_supersaturation_guard.py ====="
"$PY" -m pytest -q "$WT/tests/test_microphysics_supersaturation_guard.py" 2>&1 | tail -40
echo "RATCHET_RC=${PIPESTATUS[0]}"

echo "===== EXISTING hard-sat suites (must stay green) ====="
"$PY" -m pytest -q \
  "$WT/tests/unit/test_hard_saturation_adjustment.py" \
  "$WT/tests/unit/test_hard_sat_ice_curve.py" 2>&1 | tail -30
echo "HARDSAT_RC=${PIPESTATUS[0]}"

echo "===== Contract / param-spec / dispatch ratchets ====="
"$PY" -m pytest -q \
  "$WT/tests/test_param_specs.py" \
  "$WT/tests/test_physics_contracts.py" \
  "$WT/tests/test_dispatch_hardening.py" \
  "$WT/tests/test_validate_strict_coverage.py" 2>&1 | tail -30
echo "RATCHETS_RC=${PIPESTATUS[0]}"

echo "===== Microphysics + sundqvist unit tests ====="
"$PY" -m pytest -q \
  "$WT/tests/unit/test_physics_microphysics.py" \
  "$WT/tests/atmosphere/hydrostatic/unit/test_sundqvist_faithful.py" 2>&1 | tail -30
echo "MICRO_RC=${PIPESTATUS[0]}"

echo "ALL DONE job=${SLURM_JOB_ID}"
