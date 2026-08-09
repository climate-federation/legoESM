#!/bin/bash
#SBATCH --job-name=ss_rcecli
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM-ssguard-runs/ss_rcecli_%j.log
# The new RCE/CRM-lane guard CLI: its own suite plus the RCEMIP driver tests
# that could regress from touching _build_microphysics_config (run_lba_plane
# and run_gate_plane both call it).
set -uo pipefail

PINNED_SHA="${PINNED_SHA:?set PINNED_SHA}"
SRC=/burg-archive/glab/users/pg2328/legoESM
RUNS=/burg-archive/glab/users/pg2328/legoESM-ssguard-runs
WT="$RUNS/wtrce_${SLURM_JOB_ID}"
PY="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"

mkdir -p "$RUNS"
git -C "$SRC" worktree add --detach "$WT" "$PINNED_SHA" >/dev/null 2>&1 || { echo "WT ADD FAILED"; exit 1; }
echo "PINNED sha=$(git -C "$WT" rev-parse HEAD)"
cd "$WT"
_pp="$WT"; for d in "$WT"/packages/*/; do [ -d "${d}legoesm" ] && _pp="${_pp}:${d}"; done
export PYTHONPATH="$_pp"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

echo "===== new RCEMIP guard-CLI suite ====="
"$PY" -m pytest -q "$WT/tests/unit/test_run_rcemip_plane_hard_sat_cli.py" 2>&1 | tail -30
echo "RCECLI_RC=${PIPESTATUS[0]}"

echo "===== RCEMIP / plane-driver regression (shared _build_microphysics_config) ====="
"$PY" -m pytest -q \
  "$WT/tests/unit/test_run_rcemip_plane_positivity.py" \
  "$WT/tests/unit/test_rcemip_wing_ic.py" \
  "$WT/tests/unit/test_rcemip_land_slab.py" 2>&1 | tail -20
echo "RCEREG_RC=${PIPESTATUS[0]}"

echo "ALL DONE job=${SLURM_JOB_ID}"
