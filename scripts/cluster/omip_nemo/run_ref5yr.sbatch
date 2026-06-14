#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_nref5
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --time=11:55:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nref_%j.out

# OMIP-faithful step C (REFERENCE): NEMO ORCA1 5-yr COREv2-NYF spinup
# (nn_itend=43800) on 32 ranks in-container -> first climatology reference.
# On `short` (12h, schedules faster than scavenger `burst`); yearly restarts
# (nn_stock=8760) allow resuming for more years later.
set +e
WS=/burg-archive/glab/users/pg2328/nemo_orca1
SIF=$WS/morays_env_amd64.sif
RUNREL=nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
echo "=== HOST ==="; hostname; date; echo "ntasks=$SLURM_NTASKS"
module load singularity/3.7.1 2>&1 | tail -1

echo "=== RUN NEMO 5yr (mpirun -np 32) ==="
time singularity exec --writable-tmpfs --bind "$WS":/home/jdoe/to_host "$SIF" \
  bash -lc "cd /home/jdoe/to_host/$RUNREL && ulimit -s unlimited && mpirun -np 32 ./nemo.exe"
echo "nemo_rc=$?"

RUNDIR=$WS/$RUNREL; cd "$RUNDIR"
echo "=== ocean.output tail ==="; tail -15 ocean.output 2>/dev/null
echo "=== ERRORS? ==="; grep -lE "E R R O R|Fortran runtime|forrtl|Segmentation" ocean.output* *.err 2>/dev/null | head
echo "=== last model step ==="; cat time.step 2>/dev/null
echo "=== annual + last monthly grid_T ==="; ls -lah ORCA1_1y_*grid_T*.nc 2>/dev/null | tail; ls ORCA1_1m_*grid_T*.nc 2>/dev/null | tail -3
echo "=== output size ==="; du -sh "$RUNDIR" 2>/dev/null
echo "=== DONE ==="
