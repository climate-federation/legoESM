#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=nemo_1h
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nemo1h_%j.out

# NEMO ORACLE, 2 days from the same rest+IC start as RUN_GATEWAY, HOURLY window-mean
# avt/avm/bn2 (W), to/so/tos/mldkz5/mldr10_1 (T), uo (U): the oracle's own first-day
# tendency, hour by hour, for the 19 m vs 36 m turbulent-layer question (2026-09-26).
# 32 ranks inside the morays container, to validate the run pipeline + produce
# a first monthly-mean grid_T and the eORCA1 mesh_mask. XIOS attached mode
# (using_server=false) → plain mpirun ./nemo.exe.
set +e
WS=/burg-archive/glab/users/pg2328/nemo_orca1
SIF=$WS/morays_env_amd64.sif
RUNREL=nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_HOURLY
echo "=== HOST ==="; hostname; date; echo "ntasks=$SLURM_NTASKS"
module load singularity/3.7.1 2>&1 | tail -1
# g188 (2026-09-26): module loaded silently, binary absent -> fail loudly so
# afterok chains do not fire on a zero-second "COMPLETED".
SING=$(command -v singularity || command -v apptainer) || { echo "no singularity/apptainer on $(hostname)"; exit 3; }
echo "container runtime: $SING"

echo "=== RUN NEMO (mpirun -np 32) ==="
time $SING exec --writable-tmpfs --bind "$WS":/home/jdoe/to_host "$SIF" \
  bash -lc "cd /home/jdoe/to_host/$RUNREL && ulimit -s unlimited && mpirun -np 32 ./nemo.exe"
echo "nemo_rc=$?"

RUNDIR=$WS/$RUNREL
cd "$RUNDIR"
echo "=== ocean.output tail (errors + end) ==="
ls ocean.output* 2>/dev/null | head
tail -25 ocean.output 2>/dev/null || tail -25 ocean.output_0000 2>/dev/null
echo "=== ANY E R R O R ? ==="; grep -lE "E R R O R|Fortran runtime error|forrtl|Segmentation" ocean.output* *.err 2>/dev/null | head
echo "=== output .nc files ==="; ls -lah *grid_T*.nc *grid_U*.nc *icemod*.nc mesh_mask*.nc 2>/dev/null | head -30
echo "=== tile suffixes? (multiple_file → _NNNN) ==="; ls *_0000.nc 2>/dev/null | head; ls *grid_T*.nc 2>/dev/null | grep -cE "_[0-9]{4}\.nc$" | sed 's/^/tiled_grid_T_count=/'
echo "=== time.step (last model step) ==="; cat time.step 2>/dev/null
echo "=== DONE ==="
