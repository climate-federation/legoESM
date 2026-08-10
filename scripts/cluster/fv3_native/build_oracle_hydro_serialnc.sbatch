#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=duohydro
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=g083,g210,g161,g203,g013,g132,g086
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/build_hydro_nc474_%j.log
# A HYDROSTATIC 3-D duo oracle, built from the PINNED, checksum-verified,
# unmodified Zenodo tree.
#
# Why this build has to exist: every shipped 3-D deck in the release runs
# `hydrostatic = .F.`, so a hydrostatic port has NOTHING in the release to
# bit-gate against. `hydrostatic` is a NAMELIST flag, not a compile flag,
# so the same source built WITHOUT -DSW_DYNAMICS can produce the missing
# hydrostatic reference for our own benchmark configurations.
#
# Differences from duo_model_build.sbatch, both deliberate:
#   1. SRC is the pinned tree, not the instrumented fv3_recon copy
#      (+102/+66 lines and a delp perturbation probe -- never a reference).
#   2. DEFS drops -DSW_DYNAMICS so the 3-D branches compile in.
set -uo pipefail
MPI=/burg/opt/openmpi-4.1.1_ucx_cuda_11.0.3_support
export PATH=$MPI/bin:$PATH
export LD_LIBRARY_PATH=$MPI/lib:/burg/opt/netcdf-fortran-4.5.3/lib:/burg/opt/netcdf-c-4.7.4/lib:/cm/shared/apps/hdf5/1.10.1/lib:${LD_LIBRARY_PATH:-}
NCF=/burg/opt/netcdf-fortran-4.5.3
PIN=/burg-archive/glab/users/pg2328/fv3_oracle_pinned
SRC=$PIN/atmos_cubed_sphere-symmetryclean
DRV=/burg-archive/glab/users/pg2328/fv3_recon/duo_model/atmos_drivers
FMSB=/burg-archive/glab/users/pg2328/fv3_recon/fms_sentinel_work/build
FMSSRC=/burg-archive/glab/users/pg2328/fv3_recon/fms_sentinel_work/fms-src
FMSLIB=$(find $FMSB -name "libfms_r8.a" | head -1)
FMSINC=$FMSB/include_r8
echo "PIN=$SRC"; echo "FMSLIB=$FMSLIB"
[ -f "$SRC/model/fv_dynamics.F90" ] || { echo "PINNED TREE MISSING"; exit 1; }
echo "pinned dyn_core lines = $(wc -l < $SRC/model/dyn_core.F90)  (must be 3128, unmodified)"

B=$PIN/build_hydro_nc474
rm -rf $B && mkdir -p $B && cd $B
FF="-O2 -fdefault-real-8 -fdefault-double-8 -fcray-pointer -ffree-line-length-none -fno-range-check -fdec -cpp"
DEFS="-DSPMD -Duse_libMPI -Duse_netCDF -DINTERNAL_FILE_NML"     # no -DSW_DYNAMICS
INC="-I$FMSINC -I$NCF/include -I$SRC/tools -I$FMSSRC/include"
echo "DEFS=$DEFS"

FILES=$(ls $SRC/tools/global_grid/*.F90 $SRC/tools/*.F90 $SRC/model/*.F90 \
          $SRC/driver/solo/*.F90 $DRV/solo/atmos_model.F90)
echo "compiling $(echo "$FILES" | wc -w) files (fixpoint passes)"
prev=999
for pass in $(seq 1 12); do
  remaining=""; n_ok=0
  for f in $FILES; do
    o=$(basename ${f%.F90}).o
    [ -f $o ] && { n_ok=$((n_ok+1)); continue; }
    if mpif90 $FF $DEFS $INC -I$B -c $f -o $o 2> ${o%.o}.err; then n_ok=$((n_ok+1))
    else remaining="$remaining $f"; fi
  done
  cur=$(echo $remaining | wc -w)
  echo "pass $pass: ok=$n_ok remaining=$cur"
  [ -z "$remaining" ] && break
  if [ "$cur" = "$prev" ]; then
    echo "FIXPOINT STALL"
    for f in $remaining; do echo "== $(basename $f)"; head -12 $(basename ${f%.F90}).err; done
    break
  fi
  prev=$cur
done
[ -n "${remaining:-}" ] && { echo "BUILD_INCOMPLETE"; exit 1; }
mpif90 $FF -o fv3_solo_hydro.exe *.o $FMSLIB -L$NCF/lib -lnetcdff \
  -L/burg/opt/netcdf-c-4.7.4/lib -lnetcdf || { echo "LINK_FAILED"; exit 1; }
ls -la fv3_solo_hydro.exe
echo BUILD_HYDRO_ORACLE_DONE

# --- chained run: same coherent stack (netcdf-fortran-4.5.3 + netcdf-c-4.7.4
# --- + hdf5 1.10.1, all SERIAL so no libmpi ABI clash) ---
PIN=/burg-archive/glab/users/pg2328/fv3_oracle_pinned
SRCRUN=/burg-archive/glab/users/pg2328/fv3_recon/duo_model/run_c48_nh13
W=$PIN/run_hydro_npz5_nc474
rm -rf $W && mkdir -p $W/RESTART $W/INPUT
cp -r $SRCRUN/INPUT $W/ 2>/dev/null || true
for f in input.nml diag_table field_table; do cp $SRCRUN/$f $W/; done
cd $W
python3 - <<'PY'
import re
s=open('input.nml').read()
s=re.sub(r'hydrostatic\s*=\s*\.F\.','hydrostatic = .T.',s)
s=re.sub(r'phys_hydrostatic\s*=\s*\.F\.','phys_hydrostatic = .T.',s)
s=re.sub(r'\bnpz\s*=\s*\d+','npz    = 5',s)
s=re.sub(r'\bdays\s*=\s*\d+','days = 1',s)
open('input.nml','w').write(s)
PY
echo "=== RUN ==="
mpirun -np 6 $PIN/build_hydro_nc474/fv3_solo_hydro.exe > run_out.txt 2>&1
echo "run rc=$?"
grep -iE "HDF5 library version|MPI_ERR|FATAL" run_out.txt | head -5
echo "=== nc files written ==="
ls -la *.nc RESTART/*.nc 2>/dev/null | head -20
tail -4 run_out.txt
echo NC474_DONE
