#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=faith_ll2base
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/faith_ll2base_%j.out

# Lat-lon (180x360, NEMO L75) day-90 companion to run_faithful_trp2_p45:
# the same P4/P5 lever set on the regular-grid path (build_latlon_bathy;
# NEMO eORCA1 bathy/mask regridded; marginal seas masked per the L75
# cold-start recipe).  Second-grid coverage for the campaign rerun.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/nemo-gaps-p4p5; cd $WT
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=/burg-archive/glab/users/pg2328/legoESM
RUN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
INP=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
echo "=== worktree ==="; git -C $WT log --oneline -1

GA="--grid latlon_bathy --latlon-res 180x360 --mesh $ROOT/data/grids/eORCA1.2_mesh_mask.nc \
  --dt 150 --min-levels 2 --mask-marginal-seas"
COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal \
  --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.0822 --snapshot-every-days 30 --diag-every-days 5"
LEVERS="--tracer-advection superbee --runoff-depth-spread-m 150 --river-mouth-restoring-gate"
P45=""
IC="--woa-init --woa-t $WTIC --woa-s $WSIC"
LABEL=ll2_base_ab
OUT=$ROOT/results/omip_nemo/nemolev_${LABEL}
echo "=== RUN $LABEL (latlon 180x360 + nemo drag + zdfiwm + dm2dc + ISF + BBL) ==="
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
