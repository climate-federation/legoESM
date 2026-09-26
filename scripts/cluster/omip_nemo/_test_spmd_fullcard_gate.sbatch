#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=spmdfc
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/spmdfc_%j.out
# Stage-3 gate: the full-card SPMD banding (z_coord + iwm) must (a) make the
# partial-cell config equivalent between SPMD fake devices and single device,
# and (b) leave the pure z-star SPMD path (existing gates) bit-identical.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
echo "HEAD+dirty: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export XLA_FLAGS=--xla_force_host_platform_device_count=4
echo "=== NEW full-card gate (partial-cell equivalence + z-star no-op) ==="
$PY -m pytest -q tests/parallel/test_latlon_ocean_spmd_fullcard.py 2>&1 | tail -30
rc1=$?
echo "=== EXISTING SPMD gates (must stay green: z-star equivalence + forcing) ==="
$PY -m pytest -q tests/parallel/test_latlon_ocean_spmd_step.py 2>&1 | tail -6
rc2=$?
echo "=== ruff F-class on the two edited modules ==="
$PY -m ruff check --select F,E9 \
  packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py \
  packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py 2>&1 | tail -4
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] || { echo "FATAL rc1=$rc1 rc2=$rc2"; exit 1; }
echo "ALL GATES PASS"
echo DONE
