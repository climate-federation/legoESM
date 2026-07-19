#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=trp5_eice90
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trp5_eice90_%j.out
# TRP4 30-DAY PROBE: the trp3_tkeice ship config + the two 2026-07-18 Arctic
# fixes, on the vfs-tke-fix branch:
#   F1  --freshwater-salinity local   (NEMO tra_sbc sfx = emp*sss — kills the
#       fixed-35 fresh-shelf over-brining, ~1.35x on the Siberian shelf)
#   F2  TKEConfig.eice=3 (card default; NEMO nn_eice=3) — under-ice
#       attenuation of the lc/etau wave-TKE sources (max(0,1-4*fi))
# 30 days only: the halocline-erosion signature is fully formed by day 30
# (trp3 Siberian +2.62 at d30, flat after), so a 30-day A/B against trp3's
# day-30 regional table is the cheapest decisive test.  Compare vs NEMO m1.
# CAVEAT (documented, not a confound for the Arctic question): the branch is
# current-main-based, NOT the 38483caa8 tree trp3 ran on — a residual
# tree-delta rides along; interpret the SIBERIAN/ARCTIC lines (dominated by
# the two fixes), not global scores.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/vfs-tke-fix; cd $WT
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=/burg-archive/glab/users/pg2328/legoESM
ROOT2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1
RUN=$ROOT2/cfgs/ORCA1/EXP00/RUN_REF
INP=$ROOT2/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields
INITD=$ROOT2/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/initial_conditions
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
SSSCLIM=$INP/sss_climatology_for_restoring.nc
ICEIC=$ROOT2/cfgs/ORCA1/INPUTS/Ice_initialization.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
echo "=== worktree (fix branch) ==="; git -C $WT log --oneline -1
# --mesh must be the RELATIVE path (resolved via the worktree's data symlink):
# build_tripole's mesh-consistency guard compares path STRINGS and the grid
# side uses the relative default (absolute $ROOT/... form fails the guard).
GA="--grid tripole --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 \
  --min-levels 2 --C-smag-lap 3.0 --smag-cfl-safety 0.125 --ew-cyclic-overlap"
COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal \
  --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.3 --snapshot-every-days 30 --diag-every-days 5"
LEVERS="--tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate"
P45="--bottom-drag-scheme nemo_quadratic \
  --iwm --iwm-forcing-file $INP/zdfiwm_forcing_TRA.nc \
  --dm2dc \
  --isf --isf-forcing-file $INP/runoff-icb_DaiTrenberth_Depoorter.nc \
  --bbl-adv"
IC="--woa-init --woa-t $WTIC --woa-s $WSIC \
  --nemo-monthly-init $INITD/woce_temp_monthly_init_4p2.nc $INITD/woce_salt_monthly_init_4p2.nc \
  --sss-restore-file $SSSCLIM \
  --tripole-vmix tke --ice-init $ICEIC"
# local-S REQUIRES the normalization opt-out (the model rejects
# local+normalize: global-salt covariance).  Caveat: without normalization
# the real CORE-II +0.65 Sv imbalance drifts the GLOBAL mean ~-0.17 PSU over
# 30 d (near-uniform) — read the Siberian-vs-other-region CONTRAST, and the
# companion trp4a probe (eice-only, normalize kept) isolates F2.
FIXES=""   # trp5 = 90-day tripole ship: eice=3 via the ORCA1 card (F2), s_ref+normalize kept (protocol-clean vs trp3); local-S joins once the joint volume+salt normalization exists.
LABEL=trp5_eice90
OUT=$ROOT/results/omip_nemo/nemolev_${LABEL}
echo "=== RUN $LABEL (trp3 config + F1 local-S + F2 eice=3 card default) ==="
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  $GA $COMMON $LEVERS $P45 $IC $FIXES --sw-rgb-chl --output $OUT 2>&1 | tail -80
echo "run_rc=${PIPESTATUS[0]}"; date
SNAP=$OUT/snapshot_day0090.npz
if [ -f "$SNAP" ]; then
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GT" --nemo-month 3 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label "trp5_eice90" \
    --output-dir $ROOT/results/omip_nemo/compare_trp5_eice90_m03 2>&1 | tail -30
  $PY $ROOT/scripts/tmp/_diag_arctic_sss_regions.py "$SNAP" "$GT" "trp5_eice90 d90" 3
else
  echo "[run] no day-90 snapshot -> check $OUT for blowup."
fi
echo "=== DONE ==="; date
