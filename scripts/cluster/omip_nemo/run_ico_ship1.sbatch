#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=ico_ship1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ico_ship1_%j.out
# MPAS SHIP CANDIDATE.  == the proven mpas8_corr config (job 9067424: SST 0.953,
# SSS 0.736 best-of-3, sn_sss restoring).  The MPAS SHIP RESULT IS mpas8_corr;
# this file exists so the campaign carries a VALID MPAS ship sbatch -- do not
# relaunch unless the config drifts (a fresh run just reproduces 9067424).
#
# FIX HISTORY: the original ico_ship1 (job 9076126) died at 2 min because it
# copied latlon's P45 verbatim, passing 4 levers MPAS has no bridge for:
#   --iwm      (run_omip_core2:1267 -- not wired on the MPAS vmix bridge)
#   --bbl-adv  (run_omip_core2:3580 -- raises "tripole/latlon only")
#   --isf / --dm2dc (host-loop-only; MPASOceanModel.step has no t_seconds window)
# and also FORGOT 3 levers mpas8 needs: --sss-restore-file (sn_sss target),
# --runoff-depth-nemo-ini, --river-mouth-restoring-gate.  --sw-rgb-chl is also a
# latlon-cgrid-only physics (RGB-chl SW penetration) -- dropped here too.
# These are GENUINE MPAS capability gaps, not config choices; the MPAS ship
# candidate is the largest MPAS-LEGAL subset of the faithful stack.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/nemo-gaps-p4p5; cd $WT
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=/burg-archive/glab/users/pg2328/legoESM
ROOT2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1
RUN=$ROOT2/cfgs/ORCA1/EXP00/RUN_REF
INP=$ROOT2/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
SSSCLIM=$INP/sss_climatology_for_restoring.nc
ICEIC=$ROOT2/cfgs/ORCA1/INPUTS/Ice_initialization.nc
echo HOST $(hostname); date; git -C $WT log --oneline -1
# GA/COMMON/LEVERS/IC identical to the ll_ship2 / trp2_ship1 protocol EXCEPT the
# grid and the unavoidable MPAS gaps (no iwm/bbl/isf/dm2dc/drag-scheme/rgb-chl).
GA="--grid mpas --mpas-level 7 --mpas-lloyd 20 --dt 150"
COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --years 0.25 --snapshot-every-days 30 --diag-every-days 5"
LEVERS="--tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate"
IC="--woa-init --woa-t $WTIC --woa-s $WSIC --sss-restore-file $SSSCLIM --ice-init $ICEIC"
LABEL=ico_ship1
OUT=$ROOT/results/omip_nemo/nemolev_${LABEL}
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py $GA $COMMON $LEVERS $IC --kpp-ri-crit 0.15 --output $OUT 2>&1 | tail -50
echo run_rc=${PIPESTATUS[0]}; date
SNAP=$OUT/snapshot_day0090.npz
[ -f "$SNAP" ] && $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" --nemo-gridt "$GT"   --nemo-month 3 --freeze-clamp-C -1.8 --sss-faithful --grid-label $LABEL   --output-dir $ROOT/results/omip_nemo/compare_${LABEL}_m03 2>&1 | tail -30
echo DONE; date
