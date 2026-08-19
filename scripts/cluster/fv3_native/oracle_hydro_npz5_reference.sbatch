#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=orcnc474
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --mem=24G
#SBATCH --exclude=g055,g083,g210,g161,g203,g013,g132,g086
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/run_nc474_%j.log
set -uo pipefail
MPI=/burg/opt/openmpi-4.1.1_ucx_cuda_11.0.3_support
export PATH=$MPI/bin:$PATH
# COHERENT SERIAL STACK: netcdf-fortran-4.5.3 -> netcdf-c-4.7.4 -> hdf5 1.10.1.
# All three are serial (no libmpi), which is what removes the
# MPI_Comm_create_keyval/MPI_ERR_ARG abort that netcdf-c-4.9.3 caused by
# linking libmpi.so.12 against our libmpi.so.40.
export LD_LIBRARY_PATH=$MPI/lib:/burg/opt/netcdf-fortran-4.5.3/lib:/burg/opt/netcdf-c-4.7.4/lib:/cm/shared/apps/hdf5/1.10.1/lib:${LD_LIBRARY_PATH:-}
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
print("npz/hydro/days ->", re.findall(r'(?:hydrostatic\s*=\s*\.\w\.|\bnpz\s*=\s*\d+|\bdays\s*=\s*\d+)', s))
PY
echo "=== RUN (np=6) ==="
mpirun -np 6 $PIN/build_hydro_nc474/fv3_solo_hydro.exe > run_out.txt 2>&1
echo "run rc=$?"
grep -iE "HDF5 library version|MPI_ERR|FATAL|SIGABRT" run_out.txt | head -5
echo "=== RESOLVED NAMELIST ==="
grep -iE "^ *(HYDROSTATIC|NPZ|K_SPLIT|N_SPLIT|DUOGRID|D_EXT|NORD|HORD_MT) " logfile.000000.out 2>/dev/null | head -12
echo "=== NetCDF OUTPUT ==="
ls -la *.nc RESTART/* 2>/dev/null | head -25
echo "=== last physics ==="
grep -E "PS\*|U  max|V  max|T  max" run_out.txt | tail -8
echo RUNNC474_DONE
