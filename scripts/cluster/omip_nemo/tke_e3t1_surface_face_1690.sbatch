#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=tke1690
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/tke1690_%j.out
#
# Issue #1690 -- the nemo_z0 virtual-surface face distance is NEMO's e3t(1)
# (live top-cell thickness dz_ref[0]*J), not the top cell's midpoint depth
# (-z_full_ref[0]*J = half e3t(1)), which doubled the surface coupling.
# Runs every TKE suite that touches the surface BC, plus the DINO recipe
# identity tests, on CPU with x64.
set -euo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -m pytest -q \
  tests/ocean/unit/test_tke_nemo_terms.py \
  tests/ocean/unit/test_tke_nemo_identity.py \
  tests/ocean/unit/test_tke_carried_coefficients.py \
  tests/ocean/unit/test_tke_veros_dz_slots.py \
  tests/unit/test_run_omip_core2_kappa_convention_cli.py
echo "EXIT=$?"
