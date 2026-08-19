#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=nemo_trd2
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nemo_trd2_%j.out
#
# NEMO ORCA1 trend re-run #2: the per-process tracer trends BEYOND vertical
# mixing, at the HOURLY cadence the operator test needs.
#
# WHY.  Stage B2 of the tendency campaign verified our implicit vertical
# diffusion against NEMO's ttrd_zdf/strd_zdf (T corr 0.98-0.999, S 0.94-0.99).
# That is ONE operator.  The user's ask is per-process tendency fidelity in
# 2-D and 3-D, which needs advection (ttrd_totad) and lateral diffusion
# (ttrd_ldf) too.  Those fields were emitted LIVE in the 1d group of the
# earlier reruns, but a DAILY-MEAN trend cannot be compared against an
# operator applied to an instantaneous state (a nonlinear operator on a mean
# is not the mean of the operator), so they must be re-emitted hourly.
#
# WHAT CHANGED vs RUN_TRD (config diff, one direction only): the 1h_trd file
# group is ENABLED and gains ttrd/strd totad, ldf, iso, qsr, qns, bbl, cdt;
# the 1d_trd group is disabled; nn_itend 10920 -> 8784 (24 steps = 24 h, the
# window the existing hourly comparator uses).  Same restart tiles (step
# 8760, 32-rank decomposition -> 32 ranks REQUIRED), same executable, same
# namelist physics.
#
# RUN DIR: EXP00/RUN_TRD2 -- a clone of RUN_TRD with the NetCDF inputs
# SYMLINKED (50 GB not copied) and the config files copied, so the existing
# keep_run* outputs cannot be overwritten.
set +e
WS=/burg-archive/glab/users/pg2328/nemo_orca1; SIF=$WS/morays_env_amd64.sif
EXPC=/home/jdoe/to_host/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
EXPH=$WS/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
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

echo "=== NEMO ORCA1 trend re-run #2 (steps 8761-8784, 32 ranks, attached XIOS) ==="
time singularity exec --writable-tmpfs --bind "$WS":/home/jdoe/to_host "$SIF" bash -lc "
  cd $EXPC
  mpirun -np 32 ./nemo
" 2>&1 | tail -30
echo "rc=$?"
echo "=== ocean.output tail ==="; tail -25 $EXPH/ocean.output 2>/dev/null
echo "=== E R R O R scan ==="; grep -A 4 "E R R O R" $EXPH/ocean.output 2>/dev/null | head -20
echo "=== trend outputs (NEW files only) ==="; ls -la $EXPH/ORCA1_1h_*trd1h* 2>/dev/null
echo "=== DONE ==="; date
