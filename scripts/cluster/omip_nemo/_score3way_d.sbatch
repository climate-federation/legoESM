#!/bin/bash
#SBATCH --job-name=score3d
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score3d_%j.out
#
# Pass D: re-score with the post-review scorer (nearest-source ocean mask,
# 8-neighbour coastal halo with no polar land padding, per-field finite
# intersection, MLD tail statistics, SSH demoted to opt-in).
#
#  0. the scorer's own unit tests FIRST -- an instrument that fails its tests
#     does not get to produce numbers.
#  1. matched cross-grid pair, day 30 vs NEMO month 1, plus the SAME inputs
#     under the OLD coverage mask as the instrument control: the difference
#     between the two is how much of the near-land signal was mask
#     contamination rather than model error.
#  2. a 2-degree run of the same pair.  The IDW stencil (k=4) smooths a
#     ~111 km eORCA/NEMO cell and a ~60 km MPAS cell unequally at 1 degree,
#     which flatters tripole-vs-NEMO (both ORCA1) against MPAS-vs-NEMO.  If the
#     ranking survives a target coarser than every source, it is not a regrid
#     artifact.
#  3. best-config per grid, day 90 vs NEMO month 3.
#  4. the freshwater A/B (virtual salt flux control vs real freshwater) drawn
#     on one figure.
#  5. NEMO reference transports.  WINDOW NOT MATCHED: the only archived NEMO
#     grid_U/grid_V are ANNUAL means while our arms are day-30/90 snapshots, so
#     this is a magnitude reference and no difference from it may be quoted.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
RUN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
INP=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
[ -r "$GT" ] || { echo "FATAL: NEMO grid_T unreadable"; exit 1; }

echo; echo "########## 0. SCORER UNIT TESTS (gate)"
$PY -m pytest tests/validate/test_compare_three_way_nemo.py -q
trc=$?; echo "=== scorer tests rc=$trc"
[ $trc -eq 0 ] || { echo "FATAL: scorer tests failed -- refusing to produce numbers"; exit 1; }

NFAIL=0
three () {  # $1 trp  $2 mpas  $3 month  $4 outdir  $5 mask  $6 res  $7 lab_t  $8 lab_m
  local tag=$(basename $4)
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole "$1" --mpas "$2" --nemo-gridt "$GT" --nemo-month $3 \
    --mask-mode $5 --res-deg $6 --label-tripole "$7" --label-mpas "$8" \
    --out-dir "$4"
  # Capture $? on the line AFTER the command and BEFORE any other expansion:
  # `rc=$?` inside an echo whose argument also contains $(basename ...) reports
  # the exit status of basename, so a crashed scorer logged rc=0.
  local rc=$?
  [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
  echo "=== $tag [mask=$5 res=$6] rc=$rc"
}

XT=$R/xgrid_trp_d30/snapshot_final.npz
XM=$R/xgrid_mpas_d30/snapshot_final.npz

echo; echo "########## 1. MATCHED cross-grid pair, day 30 vs NEMO month 1"
three $XT $XM 1 $R/xgrid3_d30_m01_v2      nearest  1.0 "tripole eORCA1" "MPAS ico7"
echo; echo "--- instrument control: identical inputs, OLD coverage mask ---"
three $XT $XM 1 $R/xgrid3_d30_m01_covmask coverage 1.0 "tripole eORCA1" "MPAS ico7"

echo; echo "########## 2. resolution control: same pair on a 2-degree target"
three $XT $XM 1 $R/xgrid3_d30_m01_res2    nearest  2.0 "tripole eORCA1" "MPAS ico7"

echo; echo "########## 3. BEST-CONFIG per grid, day 90 vs NEMO month 3"
three $R/nemolev_trp_fwreal_d90/snapshot_day0090.npz \
      $R/nemolev_mpas15_tke_d90/snapshot_day0090.npz \
      3 $R/best3_d90_m03_v2 nearest 1.0 "tripole fwreal" "MPAS ico7 tke"

echo; echo "########## 4. FRESHWATER A/B: virtual salt flux control vs real FW"
three $R/nemolev_trp_gwcorr_d90/snapshot_day0090.npz \
      $R/nemolev_trp_fwreal_d90/snapshot_day0090.npz \
      3 $R/fw_ab_d90_m03 nearest 1.0 "tripole VSF (control)" "tripole real-FW"

echo; echo "########## 5. NEMO reference transports (ANNUAL -- window NOT matched)"
$PY -u scripts/validate/nemo_transports.py \
  --grid-v $RUN/ORCA1_1y_20000101_20041231_grid_V.nc \
  --grid-u $RUN/ORCA1_1y_20000101_20041231_grid_U.nc \
  --domain-cfg $INP/orca1_inputs/data_repository/input_fields/domain_cfg.nc \
  --target-lat 26.5
rc=$?; [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
echo "=== nemo transports rc=$rc"
echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
