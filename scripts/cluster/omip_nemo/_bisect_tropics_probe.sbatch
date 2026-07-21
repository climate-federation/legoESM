#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=bisect_trop
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/bisect_trop_%j.out
# TROPICS TREE-REGRESSION BISECT, probe pair 1.
# ll_ship1head CONFIRMED the +0.64 C tropical warm bias is PURELY the tree
# (ship1 config byte-identical: f5925dcfd +0.073 vs HEAD +0.713 at d90).
# ship1 = KPP latlon -> TKE-only commits are inert; PRIME suspect =
# 0be115614's "model-EOS-consistent KPP/CATKE mixing" half (changes the KPP
# density/Ri diagnostics on WRIGHT runs too; the S-EOS itself is DINO-only).
# Probe: 0be115614~1 vs 0be115614, ship1 config, 30 days, metric = the
# tropics_23S_23N SST row of the day-30 scorecard vs NEMO m1 (the +0.64
# builds roughly linearly -> expect ~+0.2-0.3 signal at d30 if this commit
# is the driver; parent should sit near the f5925 trajectory).
set +e
MAIN=/burg-archive/glab/users/pg2328/legoESM
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=$MAIN
RUN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
ICEIC=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/Ice_initialization.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
for SHA in 0be115614~1 0be115614; do
  TAG=$(git -C $MAIN rev-parse --short "$SHA")
  WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/bisect_trop_$TAG
  echo; echo "########## TROPICS PROBE $SHA ($TAG) ##########"
  git -C $MAIN worktree add --detach "$WT" "$SHA" 2>&1 | tail -1
  cd "$WT" || continue
  git log --oneline -1
  export PYTHONPATH="$(ls -d $WT/packages/*/ 2>/dev/null | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
  GA="--grid latlon_bathy --latlon-res 180x360 --mesh $ROOT/data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 --mask-marginal-seas"
  COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --years 0.085 --snapshot-every-days 30 --diag-every-days 5"
  P45="--bottom-drag-scheme nemo_quadratic --iwm --iwm-forcing-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc --dm2dc --isf --isf-forcing-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc --bbl-adv"
  OUT=$ROOT/results/omip_nemo/nemolev_bisect_trop_$TAG
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
    $GA $COMMON $P45 --woa-init --woa-t $WTIC --woa-s $WSIC \
    --tracer-advection superbee --kpp-ri-crit 0.15 --ice-init $ICEIC \
    --sw-rgb-chl --output $OUT 2>&1 | tail -8
  echo "run_rc=${PIPESTATUS[0]}"
  SNAP=$OUT/snapshot_day0030.npz
  if [ -f "$SNAP" ]; then
    cd "$WT"
    $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
      --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
      --grid-label "trop_$TAG" \
      --output-dir $ROOT/results/omip_nemo/compare_trop_${TAG}_m01 2>&1 \
      | grep -aE "tropics_23S_23N|SST\]" | head -4
    echo "[trop-probe] $TAG done — tropics row above (SST band = first tropics line)"
  else
    echo "[trop-probe] $TAG NO day-30 snapshot (blowup?)"
  fi
done
echo; echo "=== TROPICS PAIR DONE ==="; date