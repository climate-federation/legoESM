#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=land1321
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/mpas_land_mpi_1321/item4_%j.out
#
# #1321 work item 4 -- EXECUTE the single-rank vs multi-rank ladder on the MPAS
# full-physics lane. Items 1-3 landed (PR #1546 published the downwelling
# surface fluxes the land marshaller needs under MPI; PR #1357 wired the
# distributed q_v del2 halo), and a driver-level test for item 4 exists
# (tests/distributed/test_mpas_multilayer_land_mpi.py) -- but the issue's own
# closing condition is an EXECUTED run, and the last state check on the issue
# says exactly that: "the code exists" is not "the lane runs".
#
# The ladder, smallest distributed first, per the repo rule:
#   np=1  -- the serial reference; also proves the test is not MPI-only green
#   np=2  -- the smallest real partition
#   np=4  -- a partition where a rank owns no land at all is possible
#
# What each tier asserts is documented in the test module; the two BIT-EXACT
# tiers (land advance on the Voronoi partition, checkpoint round trip over
# owned cells) are the ones that close this item. The end-to-end driver tier is
# deliberately loose because the ATMOSPHERE feeding the land is itself
# partition-dependent (~1.3 K over one day on this configuration).
#
# The q_v del2 distributed halo -- the half of this issue the last state check
# explicitly did NOT verify -- is run here too, under the same ranks.
set -uo pipefail
module load openmpi/gcc/64/4.1.7a1
export LEGOESM_REPO=/burg-archive/glab/users/pg2328/_issues_wt
source /burg-archive/glab/users/pg2328/legoESM/scripts/cluster/scaling_ginsburg/_env.sh
cd "$REPO"

OUT=/burg-archive/glab/users/pg2328/legoESM/results/mpas_land_mpi_1321
mkdir -p "$OUT" "$OUT/meshcache"
export LEGOESM_MESH_CACHE_DIR="$OUT/meshcache"
export LEGOESM_ALLOW_BIG_MESH_BUILD=1
# Both ranks must see the same directory; /tmp is node-local and this is a
# single-node srun, but naming it explicitly keeps the multi-node case honest.
export LEGOESM_TEST_SHARED_TMP="$OUT/shared_tmp"
mkdir -p "$LEGOESM_TEST_SHARED_TMP"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Optional: covers the two-leaf canopy leaves as well as the soil columns.
# Without it the canopy rows SKIP (they refuse to run on generic constants).
: "${LEGOESM_TEST_SURFDATA:=/burg-archive/glab/users/pg2328/legoESM/data/legoesm_surfdata_c260716.nc}"
export LEGOESM_TEST_SURFDATA

echo "sha=$(git rev-parse --short HEAD) host=$(hostname) $(date -u)"
"$PY_MPI" -c "import jax, mpi4py; print('jax', jax.__version__, 'mpi4py', mpi4py.__version__)"

TESTS="tests/distributed/test_mpas_multilayer_land_mpi.py tests/unit/test_mpas_qv_smoothing.py"

FAILED=0
for np in 1 2 4; do
  echo "=== np=$np $(date -u) ==="
  srun --mpi=pmix --cpu-bind=cores -n "$np" \
      "$PY_MPI" -m pytest $TESTS -q -p no:randomly 2>&1 | tail -30
  rc=${PIPESTATUS[0]}
  echo "=== np=$np exit=$rc ==="
  [ "$rc" -ne 0 ] && FAILED=1
done

# The job's OWN exit code has to carry the verdict. Printing a failure and
# exiting 0 is how a red ladder gets read as a green one (review finding).
echo "MPAS LAND MPI 1321 ITEM 4 DONE $(date -Is) overall=$([ $FAILED -eq 0 ] && echo PASS || echo FAIL)"
exit "$FAILED"
