#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=tri353
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/scripts/_test_tripole_mpi_353_%j.out
# Issue #353 validation: tripolar-aware MPI halo.
#   1. serial (single-rank) bit-exactness vs serial _fold_row  (no MPI)
#   2. np=2 / np=4 MPI halo == serial fold (skips if mpi4jax absent)
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date; git log --oneline -1

echo; echo "=== [1] SERIAL tripolar tests (no MPI) ==="
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -m pytest -q \
  tests/parallel/test_latlon_mpi_tripole_serial.py
echo "serial_rc=$?"

echo; echo "=== regression: existing serial halo tests still pass ==="
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -m pytest -q \
  tests/parallel/test_latlon_mpi_halo_serial.py
echo "serial_regression_rc=$?"

echo; echo "=== mpi4jax availability ==="
JAX_PLATFORMS=cpu $PY -c "import mpi4jax, mpi4py; print('mpi4jax OK', mpi4jax.__version__)" 2>&1 | tail -2
HAVE_MPI=$?

if [ "$HAVE_MPI" -eq 0 ]; then
  echo; echo "=== [2] np=2 MPI tripolar halo ==="
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 mpirun -np 2 $PY -m pytest -q -p no:cacheprovider \
    tests/distributed/test_latlon_mpi_tripole.py
  echo "mpi2_rc=$?"
  echo; echo "=== [3] np=4 MPI tripolar halo ==="
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 mpirun -np 4 $PY -m pytest -q -p no:cacheprovider \
    tests/distributed/test_latlon_mpi_tripole.py
  echo "mpi4_rc=$?"
else
  echo "mpi4jax NOT available — skipping multi-rank tests (serial proof stands)."
fi
echo DONE; date
