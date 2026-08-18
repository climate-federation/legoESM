#!/bin/bash
#SBATCH --job-name=score3b
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score3b_%j.out
#
# Pass B of the NEMO-fidelity scoring sweep.
#   1. gwcorr_d90 = the CONTROL for fwreal_d90.  Verified from the manifests:
#      the two command lines differ by exactly one token pair,
#      '--freshwater-closure real_freshwater'.  Without this the fwreal SSS
#      improvement is confounded (its nearest previously-scored neighbour,
#      trpsbc_win_d90, also differs in --ice-ocean-heat-coeff and ran dirty).
#   2. the day-30 TKE mixing-length arms (mxl4 vs its mxl3 control) and the
#      ice-melt / runoff-placement arms, all vs NEMO month 1.
#   3. a BEST-CONFIG three-way (tripole fwreal d90 + MPAS ico7 d90 vs NEMO m3).
#      NOT a matched cross-grid pair -- the two arms run different flag sets --
#      so it answers "how far is each grid's best configuration from NEMO",
#      never "how much do the grids differ".  The matched-pair answer is
#      xgrid3_d30_m01.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
[ -r "$GT" ] || { echo "FATAL: NEMO grid_T unreadable"; exit 1; }

score () {  # $1 run dir  $2 snapshot  $3 nemo month  $4 label
  local s=$R/$1/$2
  if [ ! -r "$s" ]; then echo "SKIP $1: $s not readable"; return; fi
  $PY -u scripts/validate/compare_omip_nemo.py \
    --legoesm-snapshot "$s" --nemo-gridt "$GT" --nemo-month $3 \
    --freeze-clamp-C -1.8 --sss-faithful --grid-label "$4" \
    --output-dir $R/compare_$1_m0$3
  echo "=== $1 rc=$?"
}

echo; echo "########## 1. fwreal CONTROL (day 90 vs NEMO month 3)"
score nemolev_trp_gwcorr_d90  snapshot_day0090.npz 3 "tripole gwcorr d90 (fwreal control)"

echo; echo "########## 2. day-30 arms vs NEMO month 1"
score nemolev_trp_mxl4_d30      snapshot_final.npz   1 "tripole tke-mxl-choice 4 d30"
score nemolev_trp_mxl3ctl_d30   snapshot_final.npz   1 "tripole tke-mxl-choice 3 d30 (control)"
score nemolev_trp_nnmxl2_d30    snapshot_day0030.npz 1 "tripole nn_mxl2 d30"
score nemolev_trp_icemelt70_d30 snapshot_final.npz   1 "tripole ice-melt-70 d30"

echo; echo "########## 3. BEST-CONFIG three-way, day 90 vs NEMO month 3"
$PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
  --tripole $R/nemolev_trp_fwreal_d90/snapshot_day0090.npz \
  --mpas    $R/nemolev_mpas15_tke_d90/snapshot_day0090.npz \
  --nemo-gridt "$GT" --nemo-month 3 \
  --label-tripole "tripole fwreal" --label-mpas "MPAS ico7 tke" \
  --out-dir $R/best3_d90_m03
echo "=== best three-way rc=$?"
echo "=== ALL DONE ==="
