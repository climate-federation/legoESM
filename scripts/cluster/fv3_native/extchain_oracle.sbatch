#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=extchain
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --exclude=g055,g083,g210,g161,g203,g013,g132,g086
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/extchain_%j.log
# Composed duo halo-exchange oracle: build the extchain driver against the
# PINNED tree's build objects + the REAL libFMS_r8 (no shims), run it at
# C48 and C12 on 6 ranks (layout 1,1 per tile), then run the port
# comparison and the consuming pytest.
#
# The pinned tree is READ-ONLY: we compile only our two new files
# (fv3_extchain_extract.F90 verbatim-copy module + the driver) in a
# scratch dir and link the existing $PIN/build_hydro objects.
set -uo pipefail
MPI=/burg/opt/openmpi-4.1.1_ucx_cuda_11.0.3_support
export PATH=$MPI/bin:$PATH
export LD_LIBRARY_PATH=$MPI/lib:/burg/opt/netcdf-fortran-4.6.2/lib:/burg/opt/netcdf-c-4.9.3/lib:${LD_LIBRARY_PATH:-}
NCF=/burg/opt/netcdf-fortran-4.6.2
PIN=/burg-archive/glab/users/pg2328/fv3_oracle_pinned
SRC=$PIN/atmos_cubed_sphere-symmetryclean
B=$PIN/build_hydro
FMSB=/burg-archive/glab/users/pg2328/fv3_recon/fms_sentinel_work/build
FMSSRC=/burg-archive/glab/users/pg2328/fv3_recon/fms_sentinel_work/fms-src
FMSLIB=$(find $FMSB -name "libfms_r8.a" | head -1)
FMSINC=$FMSB/include_r8

REPO=${EXTCHAIN_REPO:-/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/agent-a0099839a1b861d35}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/extchain
DECK=$PIN/run_hydro_1step_gfs

echo "REPO=$REPO"
echo "REPO_SHA=$(git -C $REPO rev-parse HEAD 2>/dev/null || echo untracked)"
echo "FMSLIB=$FMSLIB"
sha256sum $SRC/tools/fv_duogrid.F90
[ -f "$B/duogrid_mod.mod" ] || { echo "NO PINNED BUILD MODS"; exit 1; }
[ -f "$B/lib_grid_mod.mod" ] || { echo "NO lib_grid_mod.mod"; exit 1; }
[ -f "$DECK/input.nml" ] || { echo "NO PARITY DECK"; exit 1; }

# EXTCHAIN_SKIP_RUNS=1: reuse existing oracle dumps, run only the
# comparison + pytest (compare-side iteration without re-running Fortran)
SKIP=${EXTCHAIN_SKIP_RUNS:-0}

# ---- build our two files in scratch, link pinned objects -------------
W=$OUT/build
if [ "$SKIP" = "1" ]; then
  echo "SKIP_RUNS=1: reusing $OUT/run_c48 + run_c12"
  [ -f $OUT/run_c48/extchain_t1.dat ] || { echo "NO EXISTING DUMPS"; exit 1; }
else
rm -rf $W && mkdir -p $W && cd $W
FF="-O2 -fdefault-real-8 -fdefault-double-8 -fcray-pointer -ffree-line-length-none -fno-range-check -fdec -cpp"
DEFS="-DSPMD -Duse_libMPI -Duse_netCDF -DINTERNAL_FILE_NML"
INC="-I$FMSINC -I$NCF/include -I$B -I$FMSSRC/include"
mpif90 $FF $DEFS $INC -c $REPO/scripts/validate/fv3_native/fv3_extchain_extract.F90 -o extchain_extract.o \
  || { echo "EXTRACT_COMPILE_FAILED"; exit 1; }
mpif90 $FF $DEFS $INC -I$W -c $REPO/scripts/validate/fv3_native/fv3_extchain_oracle_driver.F90 -o extchain_driver.o \
  || { echo "DRIVER_COMPILE_FAILED"; exit 1; }
LINKOBJS=$(ls $B/*.o | grep -v "/atmos_model.o$")
mpif90 $FF -o extchain.exe extchain_extract.o extchain_driver.o $LINKOBJS \
  $FMSLIB -L$NCF/lib -lnetcdff -L/burg/opt/netcdf-c-4.9.3/lib -lnetcdf \
  || { echo "LINK_FAILED"; exit 1; }
ls -la extchain.exe

# ---- run: C48 (parity deck, npz=5 hydro) and C12 ---------------------
run_one () {
  local tag=$1 npx=$2
  local R=$OUT/run_$tag
  rm -rf $R && mkdir -p $R/INPUT $R/RESTART
  for f in input.nml field_table diag_table; do cp -f $DECK/$f $R/ 2>/dev/null || true; done
  cd $R
  $PY - "$npx" <<'PY'
import re, sys
npx = int(sys.argv[1])
s = open('input.nml').read()
s = re.sub(r'\bnpx\s*=\s*\d+', f'npx      = {npx}', s)
s = re.sub(r'\bnpy\s*=\s*\d+', f'npy      = {npx}', s)
open('input.nml','w').write(s)
print("resolved npx:", re.findall(r'\bnpx\s*=\s*\d+', s))
PY
  echo "=== extchain run $tag (npx=$npx) ==="
  mpirun -np 6 $W/extchain.exe > run_out.txt 2>&1 \
    || { echo "--- tail ---"; tail -40 run_out.txt; echo "EXTCHAIN_RUN_${tag}_FAILED"; return 1; }
  grep -q EXTCHAIN_DRIVER_DONE run_out.txt \
    || { echo "--- tail ---"; tail -40 run_out.txt; echo "EXTCHAIN_RUN_${tag}_NO_DONE"; return 1; }
  ls -la extchain_t*.dat | head -3
  echo "EXTCHAIN_RUN_${tag}_OK"
}
run_one c48 49 || exit 1
run_one c12 13 || exit 1
fi   # SKIP

# ---- comparison + fixture -------------------------------------------
mkdir -p $REPO/tests/grids/fixtures
echo "=== compare C12 ==="
$PY $REPO/scripts/validate/fv3_native/compare_extchain_oracle.py \
    --oracle-dir $OUT/run_c12 --n 12 --ng 3 \
    --fixture-out $REPO/tests/grids/fixtures/fv3_extchain_oracle_c12.npz \
    --summary-out $OUT/extchain_summary_c12.npz \
  || { echo "COMPARE_C12_FAILED"; exit 1; }
echo "=== mutation control C12 (port k2e_nord forced to 4; finals MUST diverge) ==="
$PY $REPO/scripts/validate/fv3_native/compare_extchain_oracle.py \
    --oracle-dir $OUT/run_c12 --n 12 --ng 3 --port-k2e-nord 4 \
  || { echo "MUTATION_CONTROL_ERRORED"; exit 1; }
echo "=== compare C48 ==="
$PY $REPO/scripts/validate/fv3_native/compare_extchain_oracle.py \
    --oracle-dir $OUT/run_c48 --n 48 --ng 3 \
    --summary-out $OUT/extchain_summary_c48.npz \
  || { echo "COMPARE_C48_FAILED"; exit 1; }

echo "=== pytest ==="
cd $REPO
$PY -m pytest tests/grids/test_fv3_extchain_oracle.py -q 2>&1 | tail -15

echo EXTCHAIN_ALL_DONE
