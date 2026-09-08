#!/bin/bash
#SBATCH --job-name=mpasscore
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mpasscore_%j.out
#
# Score the MPAS cross-grid pair that ran to completion and then declined to
# score itself: a run ending exactly on a snapshot day writes snapshot_final,
# not snapshot_day0030, and the arm asserted the latter.  The day stamp is
# checked here from the run's own diagnostic table rather than trusted from
# the file name.
set -uo pipefail
R=/burg-archive/glab/users/pg2328/legoESM; cd $R || exit 1
export PYTHONPATH="$(ls -d $R/packages/*/ | sed 's:/$::' | tr '\n' ':')$R:$R/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
for ARM in card nocard; do
  OUT=$R/results/omip_nemo/nemolev_mpas_tke_${ARM}_d30
  SNAP=$OUT/snapshot_day0030.npz
  [ -f "$SNAP" ] || SNAP=$OUT/snapshot_final.npz
  echo; echo "############ MPAS $ARM ############"
  [ -f "$SNAP" ] || { echo "SKIP: no snapshot yet"; continue; }
  LASTDAY=$(tail -1 $OUT/diag_timeseries.csv | cut -d, -f2)
  case "$LASTDAY" in
    30.000|30.0|30) echo "day check OK: last row day $LASTDAY  ($SNAP)" ;;
    *) echo "SKIP: last row is day $LASTDAY, not 30"; continue ;;
  esac
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GW_T" --nemo-time-idx 5 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label mpas_${ARM}_d30 \
    --output-dir $R/results/omip_nemo/compare_gw_mpas_${ARM}_d30 2>&1 | tail -36
  echo "### $ARM rc=$?"
done
echo DONE
