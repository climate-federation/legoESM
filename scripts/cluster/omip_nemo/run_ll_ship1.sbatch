#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=ll_ship1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ll_ship1_%j.out
# LATLON SHIP CANDIDATE: full P45 + --kpp-ri-crit 0.15 (SST 2.12->0.98 proven) +
# --ice-init (Arctic brine fix) + the corrected ice/KPP/ISF/normalization forcing.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/nemo-gaps-p4p5; cd $WT
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=/burg-archive/glab/users/pg2328/legoESM
RUN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
echo HOST $(hostname); date; git -C $WT log --oneline -1
GA="--grid latlon_bathy --latlon-res 180x360 --mesh $ROOT/data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 --mask-marginal-seas"
COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --years 0.25 --snapshot-every-days 30 --diag-every-days 5"
P45="--bottom-drag-scheme nemo_quadratic --iwm --iwm-forcing-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc --dm2dc --isf --isf-forcing-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc --bbl-adv"
LABEL=ll_ship1
OUT=$ROOT/results/omip_nemo/nemolev_${LABEL}
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py $GA $COMMON $P45   --woa-init --woa-t $WTIC --woa-s $WSIC --tracer-advection superbee   --kpp-ri-crit 0.15 --ice-init /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/Ice_initialization.nc --sw-rgb-chl --output $OUT 2>&1 | tail -50
echo run_rc=${PIPESTATUS[0]}; date
SNAP=$OUT/snapshot_day0090.npz
[ -f "$SNAP" ] && $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" --nemo-gridt "$GT"   --nemo-month 3 --freeze-clamp-C -1.8 --sss-faithful --grid-label $LABEL   --output-dir $ROOT/results/omip_nemo/compare_${LABEL}_m03 2>&1 | tail -30
echo DONE; date
