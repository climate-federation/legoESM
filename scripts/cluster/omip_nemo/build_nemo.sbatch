#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=omip_build
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/build_%j.out

# OMIP-faithful step B (BUILD): run ORCA1 deploy.sh INSIDE the morays container
# (download COREv2+Zenodo inputs + compile NEMO via makenemo -m ORCA1_GCC -j3).
# Verbose deploy output → $WS/deploy_<jid>.log; SLURM .out stays concise.
set +e
WS=/burg-archive/glab/users/pg2328/nemo_orca1
SIF=$WS/morays_env_amd64.sif
ORCA1=$WS/nemo_5.0.1/cfgs/ORCA1
echo "=== HOST ==="; hostname; date
module load singularity/3.7.1 2>&1 | tail -1

echo "=== ENV DIAGNOSTIC (inside container) ==="
singularity exec "$SIF" bash -lc '
  echo "gcc:      $(command -v gcc)"
  echo "gfortran: $(command -v gfortran)"
  echo "mpif90:   $(command -v mpif90)"
  echo "-- nf-config --"; nf-config --flibs 2>&1 | head -1; nf-config --prefix 2>&1 | head -1
  echo "-- nc-config --"; nc-config --libs 2>&1 | head -1
  echo "-- libnetcdff location --"; find /usr /opt /home /lib -name "libnetcdff*" 2>/dev/null | head
  echo "-- arch NCDF path (/usr/lib/x86_64-linux-gnu) --"; ls /usr/lib/x86_64-linux-gnu/ 2>/dev/null | grep -iE "netcdf" | head
' 2>&1 | head -30

echo "=== RUN deploy.sh (download + compile) ==="
LOG=$WS/deploy_${SLURM_JOB_ID}.log
ulimit -s unlimited
time singularity exec --writable-tmpfs --bind "$WS":/home/jdoe/to_host "$SIF" \
  bash -lc 'cd /home/jdoe/to_host/nemo_5.0.1/cfgs/ORCA1 && ./deploy.sh' > "$LOG" 2>&1
echo "deploy_rc=$?"
echo "=== deploy log tail (50) ==="; tail -50 "$LOG"
echo "=== build products ==="
ls -la $ORCA1/EXP00/nemo $ORCA1/EXP00/RUN_DEFAULT/nemo.exe 2>&1
echo "=== INPUTS staged ==="; ls -la $ORCA1/INPUTS/ 2>&1 | head; du -sh $ORCA1/INPUTS 2>/dev/null
echo "=== mesh_mask / eORCA1 in inputs ==="; find $ORCA1/INPUTS -iname "*mesh*mask*" -o -iname "*eORCA1*" -o -iname "*domain_cfg*" 2>/dev/null | head
echo "=== DONE ==="
