#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=nemo_ldf1ts
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nemo_ldf1tsB_%j.out
#
# NEMO ORCA1 per-step lateral momentum viscosity trend (utrd_ldf/vtrd_ldf),
# 2 steps from the RUN_DT150_TRD step-288 restart, to certify the tripole
# port of dynldf_lev_lap (rn_shlat=2, eddy_viscosity_3D) one step at a time.
# RUN DIR: EXP00/RUN_LDF1TS_B, inputs symlinked relative to RUN_DT150_TRD.
set +e
WS=/burg-archive/glab/users/pg2328/nemo_orca1; SIF=$WS/morays_env_amd64.sif
EXPC=/home/jdoe/to_host/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_LDF1TS_B
EXPH=$WS/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_LDF1TS_B
echo "=== HOST ==="; hostname; date
module load singularity/3.7.1 2>&1 | tail -1
ulimit -s unlimited

echo "=== GUARD: refuse to relaunch into a used run dir ==="
if [ -e "$EXPH/time.step" ] || [ -e "$EXPH/ocean.output" ]; then
  echo "FATAL: $EXPH already ran -- clean it or use a fresh dir"; exit 3
fi
echo "=== CONFIG PROVENANCE (md5) ==="
md5sum "$EXPH"/namelist_cfg "$EXPH"/file_def_nemo-oce.xml "$EXPH"/iodef.xml
echo "=== window + trend groups ==="
grep -E "nn_it000|nn_itend|ln_tra_trd" "$EXPH/namelist_cfg" | head -3
grep -E '1h_trd|1d_trd' "$EXPH/file_def_nemo-oce.xml" | grep enabled

echo "=== NEMO ORCA1 ldf 1ts rerun (steps 289-290, 32 ranks) ==="
time singularity exec --writable-tmpfs --bind "$WS":/home/jdoe/to_host "$SIF" bash -lc "
  cd $EXPC
  mpirun -np 32 ./nemo
" 2>&1 | tail -30
echo "rc=$?"
echo "=== ocean.output tail ==="; tail -25 $EXPH/ocean.output 2>/dev/null
echo "=== E R R O R scan ==="; grep -A 4 "E R R O R" $EXPH/ocean.output 2>/dev/null | head -20
echo "=== trend outputs (NEW files only) ==="; ls -la $EXPH/ORCA1_*ldf1ts* 2>/dev/null
echo "=== DONE ==="; date
