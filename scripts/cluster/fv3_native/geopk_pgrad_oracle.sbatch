#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=geopkpg
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --exclude=g161
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/geopk_pgrad_%j.log
# Brick geopk_pgrad: multilevel hydrostatic pressure-chain oracle
#   geopk(CG=T) -> p_grad_c -> geopk(CG=F,computehalo=T) -> one_grad_p
# Two fixtures (km=2 and km=3) from ONE km-agnostic binary — km arrives
# in the input header, so the compile chain runs ONCE before the loop.
set -euo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
S=$REPO/scripts/validate/fv3_native
W=/burg-archive/glab/users/pg2328/fv3_recon/geopk_pgrad_work
rm -rf "$W" && mkdir -p "$W" && cd "$W"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "REPO SHA $(git -C "$REPO" rev-parse HEAD)"

FF="-O2 -fdefault-real-8 -fdefault-double-8 -cpp -ffree-line-length-none"
# NOTE: neither -DSW_DYNAMICS nor -DUSE_COND — this is the first brick
# of the campaign outside the SW lane.
gfortran $FF -J "$W" -c $S/fv3_swcore_shim.F90         -o "$W/shim.o"
gfortran $FF -J "$W" -c $S/fv3_geopk_pgrad_shim.F90    -o "$W/gpshim.o"
gfortran $FF -J "$W" -c $S/fv3_tpcore_duo_extract.F90  -o "$W/tpc.o"
# compiled ONLY for the CERTIFIED a2b_edge_duo_mod (a2b_ord4);
# dsw5_duo_extract_mod is never called from this driver
gfortran $FF -J "$W" -c $S/fv3_dsw5_duo_extract.F90    -o "$W/dsw5.o"
gfortran $FF -J "$W" -c $S/fv3_geopk_pgrad_extract.F90 -o "$W/gpx.o"
gfortran $FF -J "$W" $S/fv3_geopk_pgrad_driver.F90 \
  "$W/shim.o" "$W/gpshim.o" "$W/tpc.o" "$W/dsw5.o" "$W/gpx.o" -o "$W/drv"

PP=$REPO/packages/core:$REPO/src
for KM in 2 3; do
  PYTHONPATH=$PP $PY $S/gen_geopk_pgrad_oracle.py "$W" --km "$KM"
  cd "$W" && ./drv
  PYTHONPATH=$PP $PY $S/gen_geopk_pgrad_oracle.py "$W" --km "$KM" --pack
done

# the pytest gate SKIPS when a fixture is missing — make that loud here
for KM in 2 3; do
  test -s "$REPO/tests/grids/fixtures/geopk_pgrad_oracle_c12_km${KM}.npz"
done

PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  $PY -m pytest \
    "$REPO/tests/grids/test_fv3_native_geopk_pgrad.py" \
    "$REPO/tests/grids/test_fv3_native_pgrad.py" \
    "$REPO/tests/grids/test_fv3_native_duo_stepper.py" \
    -q -rs
echo GEOPK_PGRAD_DONE
