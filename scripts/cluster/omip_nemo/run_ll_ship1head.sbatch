#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=ll_ship1head
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ll_ship1head_%j.out
# TROPICS REGRESSION CONFIRM (controlled, ONE variable = the tree).
# run_ll_ship1.sbatch's config is BYTE-IDENTICAL below (same GA/COMMON/P45/IC/
# scoring) — the ONLY change is WT: nemo-gaps-p4p5 (tree f5925dcfd, where
# ll_ship1 scored tropics_23S_23N SST bias +0.073) -> vfs-tke-fix (HEAD).
# If HEAD+ship1-config tropics ~ +0.7, the ~+0.6 tropical warm regression is in
# the tree (f5925dcfd..HEAD) and a bisect is justified.  If ~ +0.07, there is
# NO tree regression in this config and the ll7/trp5/mpas9 +0.7-1.0 came from
# THEIR config drift (e.g. --freshwater-salinity local), not the tree — so the
# ll7-vs-ship1 "+0.07->+0.71" was a CONFOUND, not a result.  Score day-90 vs
# NEMO m3 (same protocol as the +0.073 baseline).  DO NOT change any flag.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/vfs-tke-fix; cd $WT
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
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
LABEL=ll_ship1head
OUT=$ROOT/results/omip_nemo/nemolev_${LABEL}
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py $GA $COMMON $P45   --woa-init --woa-t $WTIC --woa-s $WSIC --tracer-advection superbee   --kpp-ri-crit 0.15 --ice-init /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/Ice_initialization.nc --sw-rgb-chl --output $OUT 2>&1 | tail -50
echo run_rc=${PIPESTATUS[0]}; date
SNAP=$OUT/snapshot_day0090.npz
if [ -f "$SNAP" ]; then
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" --nemo-gridt "$GT"   --nemo-month 3 --freeze-clamp-C -1.8 --sss-faithful --grid-label $LABEL   --output-dir $ROOT/results/omip_nemo/compare_${LABEL}_m03 2>&1 | tail -30
  echo "=== TROPICS TREE-REGRESSION VERDICT (HEAD ship1-config vs f5925dcfd +0.073) ==="
  $PY -c "import json;r=json.load(open('$ROOT/results/omip_nemo/compare_${LABEL}_m03/report.json'));t=r['SST_bands']['tropics_23S_23N'];print(f'HEAD ship1-config tropics_23S_23N SST bias = {t[\"bias\"]:+.3f} (baseline f5925dcfd = +0.073).  delta = {t[\"bias\"]-0.073:+.3f}');print('  -> TREE REGRESSION CONFIRMED, bisect justified' if abs(t['bias'])>0.35 else '  -> NO tree regression in ship1-config; ll7 +0.71 was config drift')"
else
  echo "[run] no day-90 snapshot -> check $OUT for blowup."
fi
echo DONE; date
