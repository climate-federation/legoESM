#!/bin/bash
#SBATCH --job-name=ss_guard3
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM-ssguard-runs/ss_guard3_%j.log
# Round 3: reachability audits, CLI round-trips, guardrail ratchets, lint.
set -uo pipefail

PINNED_SHA="${PINNED_SHA:?set PINNED_SHA}"
SRC=/burg-archive/glab/users/pg2328/legoESM
RUNS=/burg-archive/glab/users/pg2328/legoESM-ssguard-runs
WT="$RUNS/wt3_${SLURM_JOB_ID}"
PY="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"

mkdir -p "$RUNS"
git -C "$SRC" worktree add --detach "$WT" "$PINNED_SHA" >/dev/null 2>&1 || { echo "WT ADD FAILED"; exit 1; }
echo "PINNED sha=$(git -C "$WT" rev-parse HEAD)"
cd "$WT"
_pp="$WT"; for d in "$WT"/packages/*/; do [ -d "${d}legoesm" ] && _pp="${_pp}:${d}"; done
export PYTHONPATH="$_pp"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

echo "===== Reachability audits (params / flags / schemes) ====="
"$PY" -m pytest -q \
  "$WT/tests/unit/test_params_reachability_audit.py" \
  "$WT/tests/unit/test_flag_reachability_audit.py" \
  "$WT/tests/unit/test_scheme_reachability_audit.py" \
  "$WT/tests/unit/test_driver_scheme_reachability.py" 2>&1 | tail -30
echo "REACH_RC=${PIPESTATUS[0]}"

echo "===== CLI round-trips touching microphysics config ====="
"$PY" -m pytest -q \
  "$WT/tests/unit/test_run_amip_cli.py" \
  "$WT/tests/unit/test_run_les_plane_cli.py" 2>&1 | tail -20
echo "CLI_RC=${PIPESTATUS[0]}"

echo "===== Guardrail ratchets (constants / saturation / private imports) ====="
"$PY" -m pytest -q \
  "$WT/tests/test_no_hardcoded_constants.py" \
  "$WT/tests/test_no_saturation_reimpl.py" \
  "$WT/tests/test_no_private_cross_imports.py" \
  "$WT/tests/test_no_inline_physics_coeffs.py" 2>&1 | tail -25
echo "GUARD_RC=${PIPESTATUS[0]}"

echo "===== Full super-saturation guard suites (re-confirm at this SHA) ====="
"$PY" -m pytest -q \
  "$WT/tests/test_microphysics_supersaturation_guard.py" \
  "$WT/tests/unit/test_supersaturation_guard_behaviour.py" \
  "$WT/tests/unit/test_hard_saturation_adjustment.py" \
  "$WT/tests/unit/test_hard_sat_ice_curve.py" \
  "$WT/tests/unit/test_physics_microphysics.py" 2>&1 | tail -20
echo "SUITE_RC=${PIPESTATUS[0]}"

echo "===== ruff on the changed files ====="
"$PY" -m ruff check \
  "$WT/packages/atmosphere/legoesm/atmosphere/physics/microphysics/sundqvist.py" \
  "$WT/packages/atmosphere/legoesm/atmosphere/physics/microphysics/ml_emulator.py" \
  "$WT/packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py" \
  "$WT/packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py" \
  "$WT/packages/coupler/legoesm/driver/config.py" \
  "$WT/packages/coupler/legoesm/driver/run_config_yaml.py" \
  "$WT/tests/test_microphysics_supersaturation_guard.py" \
  "$WT/tests/unit/test_supersaturation_guard_behaviour.py" 2>&1 | tail -25
echo "RUFF_RC=${PIPESTATUS[0]}"

echo "ALL DONE job=${SLURM_JOB_ID}"
