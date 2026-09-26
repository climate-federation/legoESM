#!/bin/bash
#SBATCH --job-name=score3c
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score3c_%j.out
#
# Pass C: re-score with the corrected ocean mask, plus the instrument control
# and the NEMO circulation reference.
#
#  1+2. The three-way scorecard under BOTH mask modes.  'nearest' classifies a
#       target cell wet iff the nearest cell of every source is wet; 'coverage'
#       is the regridder's distance-to-wet-data flag, which admits land cells
#       within 2.5 deg of ocean and fills them by extrapolation.  Running both
#       is the instrument control: it quantifies how much of the near-land
#       signal was mask contamination rather than model error.  Only the
#       'nearest' numbers are reportable.
#  3.   NEMO's own AMOC@26N and ACC@Drake.  NOTE THE WINDOW: the only NEMO
#       grid_U/grid_V archived here are ANNUAL means (ORCA1_1y_*), while our
#       arms are day-30/day-90 snapshots, so this is a reference magnitude and
#       NOT a matched-window comparison.  Do not quote a difference from it.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
RUN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
INP=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
[ -r "$GT" ] || { echo "FATAL: NEMO grid_T unreadable"; exit 1; }

three () {  # $1 tripole snap  $2 mpas snap  $3 month  $4 outdir  $5 mask-mode  $6 lab_t  $7 lab_m
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole "$1" --mpas "$2" --nemo-gridt "$GT" --nemo-month $3 \
    --mask-mode $5 --label-tripole "$6" --label-mpas "$7" --out-dir "$4"
  echo "=== $(basename $4) [$5] rc=$?"
}

echo; echo "########## 1. MATCHED cross-grid pair, day 30 vs NEMO month 1"
three $R/xgrid_trp_d30/snapshot_final.npz $R/xgrid_mpas_d30/snapshot_final.npz \
      1 $R/xgrid3_d30_m01_v2 nearest "tripole eORCA1" "MPAS ico7"
echo; echo "--- instrument control: same inputs, old coverage mask ---"
three $R/xgrid_trp_d30/snapshot_final.npz $R/xgrid_mpas_d30/snapshot_final.npz \
      1 $R/xgrid3_d30_m01_covmask coverage "tripole eORCA1" "MPAS ico7"

echo; echo "########## 2. BEST-CONFIG per grid, day 90 vs NEMO month 3"
three $R/nemolev_trp_fwreal_d90/snapshot_day0090.npz \
      $R/nemolev_mpas15_tke_d90/snapshot_day0090.npz \
      3 $R/best3_d90_m03_v2 nearest "tripole fwreal" "MPAS ico7 tke"
echo; echo "--- freshwater A/B on the SAME figure: VSF control vs real FW ---"
three $R/nemolev_trp_gwcorr_d90/snapshot_day0090.npz \
      $R/nemolev_trp_fwreal_d90/snapshot_day0090.npz \
      3 $R/fw_ab_d90_m03 nearest "tripole VSF (control)" "tripole real-FW"

echo; echo "########## 3. NEMO reference transports (ANNUAL mean -- window NOT matched)"
$PY -u scripts/validate/nemo_transports.py \
  --grid-v $RUN/ORCA1_1y_20000101_20041231_grid_V.nc \
  --grid-u $RUN/ORCA1_1y_20000101_20041231_grid_U.nc \
  --domain-cfg $INP/orca1_inputs/data_repository/input_fields/domain_cfg.nc \
  --target-lat 26.5
echo "=== nemo transports rc=$?"
echo "=== ALL DONE ==="
