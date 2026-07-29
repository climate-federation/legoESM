#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=nemo_trd
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nemo_trd_%j.out
# NEMO ORCA1 tracer-trend re-run (tendency-matching step b): restart from the
# RUN_REF step-8760 tiles (32-rank decomp -> 32 ranks REQUIRED), 24 steps
# (1 day) with &namtrd ln_tra_trd=.true., writing hourly state+vmix trends
# and a daily full trend budget (file_def 1h_trd/1d_trd groups).  Same
# Singularity container + attached-XIOS (iodef using_server=false) pattern
# that ran RUN_REF and the DINO twins (_run_dino_nemo.sbatch).
set +e
WS=/burg-archive/glab/users/pg2328/nemo_orca1; SIF=$WS/morays_env_amd64.sif
EXPC=/home/jdoe/to_host/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD
EXPH=$WS/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD
echo "=== HOST ==="; hostname; date
module load singularity/3.7.1 2>&1 | tail -1
ulimit -s unlimited
echo "=== NEMO ORCA1 trend re-run (steps 8761-8784, 32 ranks, attached XIOS) ==="
time singularity exec --writable-tmpfs --bind "$WS":/home/jdoe/to_host "$SIF" bash -lc "
  cd $EXPC
  rm -f ocean.output time.step *.stat 2>/dev/null
  mpirun -np 32 ./nemo
" 2>&1 | tail -30
echo "rc=$?"
echo "=== ocean.output tail ==="; tail -25 $EXPH/ocean.output 2>/dev/null
echo "=== E R R O R scan ==="; grep -A 4 "E R R O R" $EXPH/ocean.output 2>/dev/null | head -20
echo "=== trend outputs ==="; ls -la $EXPH/ORCA1_*trd* 2>/dev/null
echo "=== DONE ==="; date
