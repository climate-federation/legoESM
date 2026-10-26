#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=prec1675
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/precision_1675/suite_%j.out
#
# #1675 -- the mixed-precision LIFT, validated on a compute node.
#
# What this proves, and what it cannot.  The lift removes the #1665 interim
# refusal of `--precision mixed` and adds a step-boundary re-cast of the eager
# dycore state back to the STORAGE dtype (`core.precision.finalize_to_storage`).
# The claim being gated is narrow and mechanical:
#
#   * in `mixed`, after one step of a dycore whose mass fixer runs, EVERY float
#     leaf carries the storage dtype (float32) EXCEPT `p_s`, which stays at the
#     accumulate dtype (float64) on purpose -- the mass fixer's exact
#     correction is what delivers the machine-precision fixing, and rounding it
#     back to fp32 destroys that (owner decision recorded on #1675);
#   * `fp64` and strict `fp32` are UNCHANGED -- the re-cast is a no-op whenever
#     storage and accumulate share a dtype, which is every mode except mixed.
#
# The MPAS arm is here rather than on the login node because it builds a
# subdivision-5 Voronoi mesh (10242 cells); the lat-lon arms are cheap enough
# to run anywhere and are included so one log carries the whole verdict.
set -uo pipefail
export LEGOESM_REPO=/burg-archive/glab/users/pg2328/_issues_wt
source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/scaling_ginsburg/_env.sh
cd "$REPO"

OUT=/burg-archive/glab/users/pg2328/legoESM/results/precision_1675
mkdir -p "$OUT" "$OUT/meshcache"
export LEGOESM_MESH_CACHE_DIR="$OUT/meshcache"
export LEGOESM_ALLOW_BIG_MESH_BUILD=1
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "sha=$(git rev-parse --short HEAD) host=$(hostname) $(date -u)"
$PY -c "import jax; print('jax', jax.__version__)"

run () {  # $1 = label, rest = pytest args
  local label="$1"; shift
  echo "=== $label $(date -u) ==="
  $PY -m pytest "$@" -q 2>&1 | tail -25
  echo "=== $label exit=${PIPESTATUS[0]} ==="
}

run leaf_dtypes      tests/unit/test_precision_leaf_dtypes.py
run precision_modes  tests/unit/test_precision_modes.py
run bootstrap        tests/unit/test_runtime_bootstrap.py
run blockers         tests/unit/test_production_blockers.py
run dtype_contracts  tests/unit/test_precision_dtype_contracts.py
run grid_capability  tests/unit/test_grid_capability.py
run omip_precision   tests/validation/test_precision_omip.py \
                     tests/validation/test_validate_omip_precision.py
run amip_precision   tests/validation/test_precision_amip.py -m "slow or not slow"

# NON-VACUITY: the leaf-dtype gate must go RED when the re-cast is removed.
# Done by patching a throwaway copy of the tree, never the tree under test.
echo "=== non-vacuity (re-cast removed) $(date -u) ==="
NV="$OUT/nonvacuity_tree"
rm -rf "$NV" && mkdir -p "$NV"
cp -a "$REPO"/. "$NV"/ 2>/dev/null
for TGT in \
    "$NV/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py" \
    "$NV/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_cdgrid.py" \
    "$NV/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py"; do
  python - "$TGT" <<'PYNV'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = s.replace("        state = finalize_to_storage(state)\n", "        pass  # re-cast removed\n")
s = s.replace("state_out = finalize_to_storage(\n            cast_pytree(state_new, None, \"storage\"))",
              "state_out = cast_pytree(state_new, None, \"storage\")")
s = s.replace("state_out = finalize_to_storage(cast_pytree(state_new, None, \"storage\"))",
              "state_out = cast_pytree(state_new, None, \"storage\")")
s = s.replace("return (finalize_to_storage(\n                    cast_pytree(state_new, None, \"storage\")), phys_state_out,",
              "return (cast_pytree(state_new, None, \"storage\"), phys_state_out,")
open(p, "w").write(s)
print("stripped", p.split("/")[-1], "remaining finalize calls:", s.count("finalize_to_storage("))
PYNV
done
NVPP="$NV:$NV/src"
for p in atmosphere core coupler ice land ml ocean tools; do NVPP="$NVPP:$NV/packages/$p"; done
(cd "$NV" && PYTHONPATH="$NVPP" $PY -m pytest tests/unit/test_precision_leaf_dtypes.py -q -k "latlon or cube" 2>&1 | tail -12)
echo "=== non-vacuity exit=$? (EXPECT a failure above: mixed must go red) ==="

echo "PRECISION 1675 SUITE DONE $(date -Is)"
