#!/bin/bash
#SBATCH --job-name=ss_guard2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM-ssguard-runs/ss_guard2_%j.log
# Round 2: behavioural suite + BYTE-IDENTITY control (branch vs origin/main) +
# pre-existing-failure control for the two ocean gm_bvp ratchet failures.
#
# ANTI-CONFOUND: two DETACHED worktrees, both pinned to explicit SHAs.
set -uo pipefail

PINNED_SHA="${PINNED_SHA:?set PINNED_SHA}"
BASE_SHA="${BASE_SHA:?set BASE_SHA}"
SRC=/burg-archive/glab/users/pg2328/legoESM
RUNS=/burg-archive/glab/users/pg2328/legoESM-ssguard-runs
WT="$RUNS/wt2_${SLURM_JOB_ID}"
WTB="$RUNS/wtbase_${SLURM_JOB_ID}"
PROBES=/burg-archive/glab/users/pg2328/legoESM-ssguard/scripts/tmp
PY="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"

mkdir -p "$RUNS"
git -C "$SRC" worktree add --detach "$WT" "$PINNED_SHA" >/dev/null 2>&1 || { echo "WT ADD FAILED"; exit 1; }
git -C "$SRC" worktree add --detach "$WTB" "$BASE_SHA" >/dev/null 2>&1 || { echo "WTB ADD FAILED"; exit 1; }
echo "PINNED branch=$(git -C "$WT" rev-parse HEAD)"
echo "PINNED base  =$(git -C "$WTB" rev-parse HEAD)"

_pp_of() { local w="$1"; local p="$w"; for d in "$w"/packages/*/; do [ -d "${d}legoesm" ] && p="${p}:${d}"; done; echo "$p"; }
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

echo "===== BYTE-IDENTITY: default-config sundqvist + ml_emulator, base vs branch ====="
for w in "$WTB" "$WT"; do
  cp "$PROBES/_probe_ssguard_byteid.py" "$w/scripts/tmp/" 2>/dev/null || { mkdir -p "$w/scripts/tmp"; cp "$PROBES/_probe_ssguard_byteid.py" "$w/scripts/tmp/"; }
done
PYTHONPATH="$(_pp_of "$WTB")" "$PY" "$WTB/scripts/tmp/_probe_ssguard_byteid.py" > "$RUNS/byteid_base_${SLURM_JOB_ID}.txt" 2>"$RUNS/byteid_base_err.txt"
echo "BASE_PROBE_RC=$?"
PYTHONPATH="$(_pp_of "$WT")" "$PY" "$WT/scripts/tmp/_probe_ssguard_byteid.py" > "$RUNS/byteid_branch_${SLURM_JOB_ID}.txt" 2>"$RUNS/byteid_branch_err.txt"
echo "BRANCH_PROBE_RC=$?"
if diff -u "$RUNS/byteid_base_${SLURM_JOB_ID}.txt" "$RUNS/byteid_branch_${SLURM_JOB_ID}.txt" > "$RUNS/byteid_diff_${SLURM_JOB_ID}.txt"; then
  echo "BYTE_IDENTITY=PASS ($(wc -l < "$RUNS/byteid_branch_${SLURM_JOB_ID}.txt") lines compared, 0 differing)"
else
  echo "BYTE_IDENTITY=FAIL"; head -40 "$RUNS/byteid_diff_${SLURM_JOB_ID}.txt"
fi

echo "===== CONTROL: are the gm_bvp ratchet failures PRE-EXISTING on base? ====="
PYTHONPATH="$(_pp_of "$WTB")" "$PY" -m pytest -q \
  "$WTB/tests/test_param_specs.py::test_config_module_has_valid_param_spec" \
  "$WTB/tests/test_physics_contracts.py::test_physics_scheme_has_valid_contract" 2>&1 | tail -8
echo "BASE_RATCHETS_RC=${PIPESTATUS[0]}"

cd "$WT"
export PYTHONPATH="$(_pp_of "$WT")"

echo "===== BEHAVIOURAL: tests/unit/test_supersaturation_guard_behaviour.py ====="
"$PY" -m pytest -q "$WT/tests/unit/test_supersaturation_guard_behaviour.py" 2>&1 | tail -50
echo "BEHAV_RC=${PIPESTATUS[0]}"

echo "===== STRUCTURAL ratchet + existing hard-sat suites ====="
"$PY" -m pytest -q \
  "$WT/tests/test_microphysics_supersaturation_guard.py" \
  "$WT/tests/unit/test_hard_saturation_adjustment.py" \
  "$WT/tests/unit/test_hard_sat_ice_curve.py" 2>&1 | tail -30
echo "STRUCT_RC=${PIPESTATUS[0]}"

echo "===== Micro + config + CLI regression ====="
"$PY" -m pytest -q \
  "$WT/tests/unit/test_physics_microphysics.py" \
  "$WT/tests/atmosphere/hydrostatic/unit/test_sundqvist_faithful.py" \
  "$WT/tests/test_dispatch_hardening.py" \
  "$WT/tests/test_validate_strict_coverage.py" 2>&1 | tail -30
echo "MICRO_RC=${PIPESTATUS[0]}"

echo "===== param_specs + contracts on BRANCH (expect same 2 gm_bvp fails as base) ====="
"$PY" -m pytest -q \
  "$WT/tests/test_param_specs.py" "$WT/tests/test_physics_contracts.py" 2>&1 | tail -12
echo "BRANCH_RATCHETS_RC=${PIPESTATUS[0]}"

echo "ALL DONE job=${SLURM_JOB_ID}"
