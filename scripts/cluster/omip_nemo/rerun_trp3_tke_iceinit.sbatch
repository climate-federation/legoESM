#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --job-name=trp3_tkeice
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trp3_tkeice_%j.out

# trp2_p46 + --tripole-vmix tke --ice-init: the FULL faithful tripole stack (drag/zdfiwm/dm2dc/
# ISF/BBL + NEMO monthly IC + sn_sss restoring) with the prognostic ice
# INITIALISED FROM NEMO's OWN Jan-1 ice state (Ice_initialization.nc:
# at_i/ht_i/ht_s/sm_i/tmsu) instead of the zero-ice cold start.
# ROOT CAUSE A/B (audit, job 8900201): zero-ice start freezes ~1-1.5 m of NEW
# ice in 90 d over 10-20 m Siberian shelf columns -> +1.2..+3.5 PSU brine =
# the +2.6..+3.8 Siberian salt residual; over-iced Barents (NEMO conc 0.37)
# gates restoring/evap -> the -1.28 fresh bias.  NEMO starts WITH that ice
# (Siberian conc 0.994, h 1.55 m) and never pays the brine bill.
# Score at day-90 vs NEMO month-3 + Arctic region breakdown -> A/B vs trp2_p46.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=/burg-archive/glab/users/pg2328/legoESM
RUN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
INP=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
ROOT2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
ICEIC=$ROOT2/cfgs/ORCA1/INPUTS/Ice_initialization.nc
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
# P46 IC + THE LEVER UNDER TEST: NEMO SI3 ice initial state (--tripole-vmix tke --ice-init)
IC="--woa-init --woa-t $WTIC --woa-s $WSIC \
  --nemo-monthly-init $INITD/woce_temp_monthly_init_4p2.nc $INITD/woce_salt_monthly_init_4p2.nc \
  --sss-restore-file $SSSCLIM \
  --tripole-vmix tke --ice-init $ICEIC"
LABEL=trp3_tke_iceinit
OUT=$ROOT/results/omip_nemo/nemolev_${LABEL}
echo "=== RUN $LABEL (trp2_p46 stack, post-fix rerun + NEMO SI3 ice IC) ==="
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
  echo "=== Arctic SSS regions (vs NEMO m3) ==="
  $PY $ROOT/scripts/tmp/_diag_arctic_sss_regions.py "$SNAP" "$GT" \
    "tripole trp2_iceinit (NEMO SI3 ice IC)" 3 2>/dev/null
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
