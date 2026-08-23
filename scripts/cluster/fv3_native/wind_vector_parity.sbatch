#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=windvec
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --exclude=g055,g083,g210,g161,g203,g013,g132,g086
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/windvec_%j.log
# Physics-coupling wind-vector parity: port compute_fv3_native_wind_vectors vs
# the pinned-Fortran gridstruct vlon/vlat/es/ew dumped by the extchain oracle
# (run 9470130).  Reuses the existing dumps; no oracle rebuild.
set -uo pipefail
REPO=${WINDVEC_REPO:-/burg-archive/glab/users/pg2328/legoESM_fv3jax}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/extchain
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
cd $REPO
echo "REPO=$REPO  SHA=$(git rev-parse HEAD)"
# reproducibility: refuse to certify from a dirty tree.  The grid-config match
# (n, ng, grid_type 0, do_schmidt identity) is enforced inside the comparison
# by the face-map coordinate floor (<1e-12): a different grid gives different
# A-grid coordinates and the bijection control fails loudly.
DIRTY=$(git status --porcelain | wc -l)
if [ "$DIRTY" != "0" ]; then echo "REFUSE: tree dirty ($DIRTY files)"; exit 2; fi

rc=0
echo "=== C12 ==="
$PY $REPO/scripts/validate/fv3_native/compare_wind_vectors.py \
    --oracle-dir $OUT/run_c12 --n 12 --ng 3 || rc=1
echo "=== C48 ==="
$PY $REPO/scripts/validate/fv3_native/compare_wind_vectors.py \
    --oracle-dir $OUT/run_c48 --n 48 --ng 3 || rc=1

echo "WINDVEC_RC=$rc"
# single unambiguous verdict: only emit the pass token if BOTH resolutions
# passed (each comparator prints its own tagged WIND_VECTOR_PARITY_OK n=<N>;
# a grep consumer must key on WINDVEC_BOTH_OK, not the per-resolution token).
if [ "$rc" = "0" ]; then echo "WINDVEC_BOTH_OK"; else echo "WINDVEC_BOTH_FAIL"; fi
exit $rc
