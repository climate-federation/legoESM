#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=dynstage
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --exclude=g055,g083,g210,g161,g203,g013,g132,g086
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/dynstage_%j.log
# ORACLE STAGE-STATE instrument: generate the staged verbatim
# dyn_core/fv_dynamics copies, build them against the pinned GFS-constants
# objects + the real GFS libFMS_r8, run one REAL fv_dynamics step twice
# (real arm / staged arm) on the C48 hydro parity deck, certify bitwise,
# then run the port stage comparison (clean + two mutation controls).
set -uo pipefail
MPI=/burg/opt/openmpi-4.1.1_ucx_cuda_11.0.3_support
export PATH=$MPI/bin:$PATH
export LD_LIBRARY_PATH=$MPI/lib:/burg/opt/netcdf-fortran-4.6.2/lib:/burg/opt/netcdf-c-4.9.3/lib:${LD_LIBRARY_PATH:-}
NCF=/burg/opt/netcdf-fortran-4.6.2
PIN=/burg-archive/glab/users/pg2328/fv3_oracle_pinned
SRC=$PIN/atmos_cubed_sphere-symmetryclean
B=$PIN/build_hydro_gfsconst
FMSB=/burg-archive/glab/users/pg2328/fv3_recon/fms_sentinel_work/build_gfs
FMSSRC=/burg-archive/glab/users/pg2328/fv3_recon/fms_sentinel_work/fms-src
FMSLIB=$(find $FMSB -name "libfms_r8.a" | head -1)
FMSINC=$FMSB/include_r8

REPO=${DYNSTAGE_REPO:-/burg-archive/glab/users/pg2328/legoESM/.claude/worktrees/agent-ab6d05255171d25cb}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/dynstage
DECK=$PIN/run_hydro_1step_gfs

echo "REPO=$REPO"
echo "REPO_SHA=$(git -C $REPO rev-parse HEAD 2>/dev/null || echo untracked)"
echo "FMSLIB=$FMSLIB"
sha256sum $SRC/model/dyn_core.F90 $SRC/model/fv_dynamics.F90
[ -f "$B/dyn_core_mod.mod" ] || { echo "NO PINNED BUILD MODS"; exit 1; }
[ -f "$DECK/input.nml" ] || { echo "NO PARITY DECK"; exit 1; }

# DYNSTAGE_SKIP_RUN=1: reuse existing oracle dumps; comparison only.
SKIP=${DYNSTAGE_SKIP_RUN:-0}

W=$OUT/build
R=$OUT/run_c48
if [ "$SKIP" = "1" ]; then
  echo "SKIP_RUN=1: reusing $R"
  [ -f $R/dyncore_stage_t1.dat ] || { echo "NO EXISTING DUMPS"; exit 1; }
else
rm -rf $W && mkdir -p $W && cd $W

echo "=== generate staged copies (verbatim round-trip enforced) ==="
$PY $REPO/scripts/validate/fv3_native/gen_dyncore_stage_copy.py \
    --outdir $W || { echo "GEN_FAILED"; exit 1; }

FF="-O2 -fdefault-real-8 -fdefault-double-8 -fcray-pointer -ffree-line-length-none -fno-range-check -fdec -cpp"
DEFS="-DSPMD -Duse_libMPI -Duse_netCDF -DINTERNAL_FILE_NML"
INC="-I$FMSINC -I$NCF/include -I$B -I$FMSSRC/include"
mpif90 $FF $DEFS $INC -c $REPO/scripts/validate/fv3_native/fv3_dyncore_stage_dump.F90 -o stage_dump.o \
  || { echo "DUMP_COMPILE_FAILED"; exit 1; }
mpif90 $FF $DEFS $INC -I$W -c $W/staged_dyn_core.F90 -o staged_dyn_core.o \
  || { echo "STAGED_DYNCORE_COMPILE_FAILED"; exit 1; }
mpif90 $FF $DEFS $INC -I$W -c $W/staged_fv_dynamics.F90 -o staged_fv_dynamics.o \
  || { echo "STAGED_FVDYN_COMPILE_FAILED"; exit 1; }
mpif90 $FF $DEFS $INC -I$W -c $REPO/scripts/validate/fv3_native/fv3_dyncore_stage_driver.F90 -o stage_driver.o \
  || { echo "DRIVER_COMPILE_FAILED"; exit 1; }
LINKOBJS=$(ls $B/*.o | grep -v "/atmos_model.o$")
mpif90 $FF -o dynstage.exe stage_dump.o staged_dyn_core.o staged_fv_dynamics.o stage_driver.o $LINKOBJS \
  $FMSLIB -L$NCF/lib -lnetcdff -L/burg/opt/netcdf-c-4.9.3/lib -lnetcdf \
  || { echo "LINK_FAILED"; exit 1; }
ls -la dynstage.exe

echo "=== run: C48 parity deck, real arm + staged arm ==="
rm -rf $R && mkdir -p $R/INPUT $R/RESTART
for f in input.nml field_table diag_table; do cp -f $DECK/$f $R/ 2>/dev/null || true; done
cd $R
mpirun -np 6 $W/dynstage.exe > run_out.txt 2>&1 \
  || { echo "--- tail ---"; tail -60 run_out.txt; echo "STAGE_RUN_FAILED"; exit 1; }
grep -q STAGE_DRIVER_DONE run_out.txt \
  || { echo "--- tail ---"; tail -60 run_out.txt; echo "STAGE_RUN_NO_DONE"; exit 1; }
grep "FMSConstants" logfile.000000.out | head -1
grep "^CERT" dyncore_stage_t1.mf | head -25
ls -la dyncore_stage_t*.dat
fi   # SKIP

echo "=== stage comparison: clean run ==="
cd $REPO
$PY scripts/validate/fv3_native/compare_dyncore_stages.py \
    --dump-dir $R --json $OUT/stage_diffs.json
RC_CLEAN=$?
echo "COMPARE_CLEAN_RC=$RC_CLEAN"

echo "=== mutation control 1: fluxavg (expect S09 flagged first) ==="
$PY scripts/validate/fv3_native/compare_dyncore_stages.py \
    --dump-dir $R --mutate fluxavg | tail -8
echo "MUTATE_FLUXAVG_RC=$?"

echo "=== pytest: instrument unit tests ==="
$PY -m pytest tests/grids/test_fv3_dyncore_stage_instrument.py -q \
    --basetemp=/burg-archive/glab/users/pg2328/fv3_duo_gaps/pyt_${SLURM_JOB_ID} 2>&1 | tail -6

echo "=== mutation control 2: b2 (expect S11 flagged first) ==="
$PY scripts/validate/fv3_native/compare_dyncore_stages.py \
    --dump-dir $R --mutate b2 | tail -8
echo "MUTATE_B2_RC=$?"

echo DYNSTAGE_ALL_DONE
