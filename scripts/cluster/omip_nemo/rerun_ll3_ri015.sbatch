#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=ll3_ri015
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ll3_ri015_%j.out
# POST-FIX rerun of ll2_ri015 (identical command; code = the one variable): ll2_p45_full config + the PROVEN --kpp-ri-crit 0.15
# (day-30 A/B: NH-midlat Jan MLD 145->90 m = NEMO 93; SST cold bias halved; day-90
# reduced-config: NH-midlat -0.76 -> -0.02). This is the FULL-config validation the
# campaign was missing (runs from the main checkout post-merge: iwm/dm2dc/isf/drag
# + the PR#809 KPP-latlon wiring all present).
set +e
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=$WT
RUN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
INP=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
echo "=== HOST ==="; hostname; nvidia-smi -L | head -1; date; git log --oneline -1
GA="--grid latlon_bathy --latlon-res 180x360 --mesh $ROOT/data/grids/eORCA1.2_mesh_mask.nc \
  --dt 150 --min-levels 2 --mask-marginal-seas"
COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv \
  --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
  --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice \
  --prognostic-ice-dynamics free_drift --years 0.25 --snapshot-every-days 30 --diag-every-days 5"
P45="--bottom-drag-scheme nemo_quadratic --iwm --iwm-forcing-file $INP/zdfiwm_forcing_TRA.nc \
  --dm2dc --isf --isf-forcing-file $INP/runoff-icb_DaiTrenberth_Depoorter.nc --bbl-adv"
IC="--woa-init --woa-t $WTIC --woa-s $WSIC"
LABEL=ll3_ri015
OUT=$ROOT/results/omip_nemo/nemolev_${LABEL}
echo "=== RUN $LABEL (full P45 + kpp-ri-crit 0.15) ==="
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  $GA $COMMON $P45 $IC --tracer-advection superbee --kpp-ri-crit 0.15 --sw-rgb-chl \
  --output $OUT 2>&1 | tail -60
echo "run_rc=${PIPESTATUS[0]}"; date
SNAP=$OUT/snapshot_day0090.npz
if [ -f "$SNAP" ]; then
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" --nemo-gridt "$GT" \
    --nemo-month 3 --freeze-clamp-C -1.8 --sss-faithful --grid-label $LABEL \
    --output-dir $ROOT/results/omip_nemo/compare_${LABEL}_m03 2>&1 | tail -30
fi
echo DONE; date
