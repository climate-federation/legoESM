#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=faith_trp2b
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/faith_trp2b_%j.out

# MAXIMALLY-NEMO-FAITHFUL tripole day-90 run, v2 (P4/P5 campaign): the
# trp1_faithful_woce_rgbchl configuration (2026-06-23) PLUS the newly
# ported NEMO schemes, closing every remaining audited namelist gap
# except the vertical-mixing closure (KPP vs TKE, still isolated):
#   drag    = zdfdrg np_non_lin (--bottom-drag-scheme nemo_quadratic;
#             rn_Cd0=1e-3, rn_ke0=2.5e-3 — was MOM6 quadratic-with-floor)
#   zdfiwm  = de Lavergne 2020 internal-wave mixing, ORCA1 maps
#             (ln_zdfiwm; backgrounds forced molecular per zdfiwm_init)
#   dm2dc   = Bernie 2007 diurnal SW on the daily CORE-II radiation
#   ISF     = 'spe' prescribed Depoorter melt over [zmin,zmax] (ln_isf)
#   BBL     = Campin-Goosse advective bottom boundary layer (ln_trabbl)
# Score at day-90 vs NEMO month-3 exactly like trp1 -> A/B of the levers.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/nemo-gaps-p4p5; cd $WT
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=/burg-archive/glab/users/pg2328/legoESM
RUN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
INP=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
ROOT2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
echo "=== worktree ==="; git -C $WT log --oneline -1

GA="--grid tripole --mesh $ROOT/data/grids/eORCA1.2_mesh_mask.nc --dt 150 \
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
INITD=$ROOT2/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/initial_conditions
SSSCLIM=$ROOT2/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc
# P46: NEMO-native monthly IC (overrides the annual woa build) + sn_sss
# restoring target + ln_rnf_depth_ini runoff depths (Arctic salt levers)
IC="--woa-init --woa-t $WTIC --woa-s $WSIC \
  --nemo-monthly-init $INITD/woce_temp_monthly_init_4p2.nc $INITD/woce_salt_monthly_init_4p2.nc \
  --sss-restore-file $SSSCLIM"
LABEL=trp2_p46_full
OUT=$ROOT/results/omip_nemo/nemolev_${LABEL}
echo "=== RUN $LABEL (trp1 + nemo drag + zdfiwm + dm2dc + ISF spe + BBL) ==="
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  $GA $COMMON $LEVERS $P45 $IC --sw-rgb-chl --output $OUT 2>&1 | tail -250
echo "run_rc=${PIPESTATUS[0]}"; date

SNAP=$OUT/snapshot_day0090.npz
if [ -f "$SNAP" ]; then
  echo "=== SCORE + BATTERY (month-3, vs NEMO) ==="
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GT" --nemo-month 3 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label "nemolev_${LABEL}" \
    --output-dir $ROOT/results/omip_nemo/compare_nemolev_${LABEL}_m03 2>&1 | tail -18
  $PY scripts/validate/diag_omip_nemo_battery.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GT" --nemo-month 3 --domain-cfg $INP/domain_cfg.nc \
    --nemo-gridv $RUN/ORCA1_1y_20000101_20041231_grid_V.nc \
    --nemo-gridu $RUN/ORCA1_1y_20000101_20041231_grid_U.nc \
    --grid-label "nemolev_${LABEL}" \
    --output-dir $ROOT/results/omip_nemo/battery_nemolev_${LABEL}_m03 2>&1 | tail -18
else
  echo "[run] no day-90 snapshot -> check $OUT for blowup."
fi
echo "=== DONE ==="; date
