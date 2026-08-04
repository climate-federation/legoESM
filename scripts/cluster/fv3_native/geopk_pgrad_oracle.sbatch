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

# ---- build metadata for the RUN MANIFEST (codex r20 blocker 2) ----
# --pack REQUIRES this manifest and re-verifies every source hash against
# the repo, so a changed extract/shim/driver/a2b dependency, compiler or
# preprocessor can no longer be stamped onto stale Fortran output.
SOURCES="fv3_swcore_shim.F90 fv3_geopk_pgrad_shim.F90 fv3_tpcore_duo_extract.F90 fv3_dsw5_duo_extract.F90 fv3_geopk_pgrad_extract.F90 fv3_geopk_pgrad_driver.F90"
CC_VER=$(gfortran --version | head -1)
CC_TGT=$(gfortran -dumpmachine)
EXE_SHA=$(sha256sum "$W/drv" | cut -d' ' -f1)
REPO_SHA=$(git -C "$REPO" rev-parse HEAD)

PP=$REPO/packages/core:$REPO/src
for KM in 2 3; do
  PYTHONPATH=$PP $PY $S/gen_geopk_pgrad_oracle.py "$W" --km "$KM"
  cd "$W" && ./drv

  MAN="$W/run_manifest_km${KM}.txt"
  {
    echo "schema=geopk_pgrad_run_manifest_v1"
    echo "km=${KM}"
    echo "repo_sha=${REPO_SHA}"
    echo "host=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID:-none}"
    echo "compiler=gfortran"
    echo "compiler_version=${CC_VER}"
    echo "compiler_target=${CC_TGT}"
    echo "flags=${FF}"
    # the preprocess lane is part of the certificate: NEITHER macro is
    # defined, which is what puts this brick outside the SW lane
    echo "defines=NONE(no -DSW_DYNAMICS, no -DUSE_COND)"
    echo "executable_sha256=${EXE_SHA}"
    for f in $SOURCES; do
      echo "source.${f}=$(sha256sum "$S/$f" | cut -d' ' -f1)"
    done
    echo "input_sha256=$(sha256sum "$W/geopk_pgrad_input.txt" | cut -d' ' -f1)"
    echo "output_sha256=$(sha256sum "$W/geopk_pgrad_output_km${KM}.txt" | cut -d' ' -f1)"
  } > "$MAN"
  cat "$MAN"

  PYTHONPATH=$PP $PY $S/gen_geopk_pgrad_oracle.py "$W" --km "$KM" --pack
done

for KM in 2 3; do
  test -s "$REPO/tests/grids/fixtures/geopk_pgrad_oracle_c12_km${KM}.npz"
done
# The fixtures ARE the certificate and MUST be committed (codex r20
# blocker 1 — they were untracked at e2b06d364, so the commit carried no
# reproducible certificate).  test_fixtures_are_committed enforces it;
# this is the operator-facing reminder.
echo "=== git add these before committing ==="
for KM in 2 3; do
  echo "  tests/grids/fixtures/geopk_pgrad_oracle_c12_km${KM}.npz"
done
git -C "$REPO" status --short -- tests/grids/fixtures/ || true

PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  $PY -m pytest \
    "$REPO/tests/grids/test_fv3_native_geopk_pgrad.py" \
    "$REPO/tests/grids/test_fv3_native_pgrad.py" \
    "$REPO/tests/grids/test_fv3_native_duo_stepper.py" \
    -q -rs
echo GEOPK_PGRAD_DONE
