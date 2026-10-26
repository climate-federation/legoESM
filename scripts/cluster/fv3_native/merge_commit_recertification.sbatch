#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=pr1769merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/pr1769_merge/recert_%j.out
#
# PR #1769: RE-CERTIFY ON THE MERGE COMMIT, not on the branch tip.
#
# WHY. The PR's receipts -- 8 steps bitwise at C24, 6 at C48, the 288-step
# driver parity -- were all collected on the branch tip. Merging main into the
# branch brought 1693 files with it, and that set INCLUDES the gate's own code
# path: fv3_native_acoustic_3d.py, fv3_native_dsw_tail_3d.py and
# fv3_native_physics_coupling.py all moved. Review called this out and it is
# right: a bitwise certificate collected before those files changed does not
# certify the merge.
#
# WHAT MAIN ACTUALLY CHANGED in those three, read rather than assumed:
#   * `_ext_scalar_planes_6` -> `ext_scalar_planes_6` (private -> public, the
#     no-private-cross-imports ratchet). A pure rename; the remaining hits on
#     the old name are a DIFFERENT function in fv3_dsw_tail_3d.py.
#   * `_HS_REF_RADIUS = 6.371e6` -> `constants.R_earth = 6.371229e6` in the
#     Held-Suarez time-scale ratio. Those two numbers DIFFER by 3.6e-5
#     relative, so this is not obviously inert -- but `rad_ratio = radius /
#     constants.R_earth` with `radius` defaulting to `constants.R_earth` gives
#     exactly 1.0 either way, and NO caller in the tree passes `radius=`. So it
#     is inert on every current path, and both the NumPy authority and the PR's
#     new JAX twin read the same constant.
# Neither is a numerics change at the configurations below. This job is what
# turns that reading into a measurement.
#
# THE GATE, on the merge commit:
#   A  C24 kt=2 pad=5, 8 steps, nq=1  -- the PR's own headline depth
#   B  C24 kt=2 pad=5, 8 steps, nq=2  -- two tracers
#   C  C24 kt=2 pad=5, 3 steps, nq=1, dt=30000 -- drives nsplt>=3, the corner
#      the second tracer defect was found in
# Exit 0 = bitwise, 1 = differs, 2 = refused. Anything but 0 on A or B BLOCKS
# the merge.
#
# Plus the PR's whole CPU test surface, because the same merge absorbed 1005
# changed lines of model_driver.py and 689 of config.py.
#
# NOT COVERED HERE and stated so it is not mistaken for covered: the 24-rank
# multi-process driver parity (needs a real multi-process allocation) and the
# wall-clock cost of the new per-sub-iteration seam refresh, which fires on
# every masked scan slot and has never been timed. Both are named on the PR.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_pr1769_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$WT" || exit 2
export PYTHONPATH="$WT/packages/core:$WT/packages/atmosphere:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ocean:$WT/packages/ml:$WT/packages/tools:$WT/src:$WT:$WT/scripts/validate"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export LEGOESM_MESH_CACHE_DIR="$WT/.meshcache"
export LEGOESM_ALLOW_BIG_MESH_BUILD=1
mkdir -p "$LEGOESM_MESH_CACHE_DIR"

echo "sha=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD) host=$(hostname) $(date -u)"
echo "merge parents: $(git log -1 --format=%p HEAD)"

FAILED=0

echo "=== PR test surface on the MERGE commit $(date -u) ==="
$PY -m pytest \
    tests/grids/test_fv3_duo_window_spmd.py \
    tests/grids/test_fv3_physics_coupling.py \
    tests/parallel/test_early_init_hardening.py \
    tests/atmosphere/hydrostatic/unit/test_fv3_duo_dynamics.py \
    tests/unit/test_run_amip_cli.py \
    tests/unit/test_config_positional_abi.py \
    -q 2>&1 | tail -30
rc=${PIPESTATUS[0]}; echo "=== tests exit=$rc ==="
# NB main is ALREADY red on 11 of test_run_amip_cli.py (measured on plain
# origin/main: 227 passed / 11 failed). A nonzero here is only a blocker if
# the failure set is bigger than those 11 -- the log above shows which.

run_gate () {  # $1 label, rest = gate args
  local label="$1"; shift
  echo "=== gate $label $(date -u) ==="
  # 24 host devices: 6*kt*kt = 24 windows at kt=2, and the six-face reference
  # arm needs 6. One process, forced host devices -- no jax.distributed.
  XLA_FLAGS="--xla_force_host_platform_device_count=24" \
    timeout 5400 $PY -u scripts/validate/fv3_native/tiled_m6_model_gate.py "$@" 2>&1 | tail -25
  local grc=${PIPESTATUS[0]}
  echo "=== gate $label exit=$grc (0=bitwise 1=differs 2=refused) ==="
  [ "$grc" -ne 0 ] && FAILED=1
  return 0
}

run_gate A_nq1      --n 24 --km 5 --kt 2 --pad 5 --steps 8 --nq 1 --n-split 8 --dt 900
run_gate B_nq2      --n 24 --km 5 --kt 2 --pad 5 --steps 8 --nq 2 --n-split 8 --dt 900
run_gate C_nsplt3   --n 24 --km 5 --kt 2 --pad 5 --steps 3 --nq 1 --n-split 8 --dt 30000

echo "PR1769 MERGE RECERTIFICATION DONE $(date -Is) gates=$([ $FAILED -eq 0 ] && echo PASS || echo FAIL)"
exit "$FAILED"
