#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=bisect_ice
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/bisect_ice_%j.out
# LATLON ICE-RUNAWAY BISECT PROBE (tree regression f5925dcfd..38483caa8).
# Fact base: ship1 config, day-5 ice_max_thick — OLD tree f5925 stays ~15-25 m
# (bounded, d91 33 m); trees 38483caa8 (ll4/ll5) + HEAD (ll_ship1head) hit
# 286-330 m by day-5 (ll7 reached 5407 m).  Threshold: day-5 thick_max > 50 m
# = RUNAWAY.  This job probes SHAs on the BRANCH MAINLINE with the ship1
# config (all flags exist at f5925, so every later mainline SHA accepts them;
# latlon data paths are ABSOLUTE -> a detached worktree needs no data symlink).
# Probe pair (highest-prior suspect first): 4c32e98a2 = the sea-ice forcing-
# limitations commit (ice-EB SW debit, thermo-time partition level, transport)
# vs its parent.  Verdict logic:
#   parent bounded + 4c32e runaway  -> culprit = 4c32e98a2 (fix directly)
#   both runaway                    -> culprit earlier (16a494bc2 main-merge
#                                      cryo-audit batch: 075f19914 ridging/
#                                      drag/tilt, 74d1f, 213a3)
#   both bounded                    -> culprit later (38483caa8 final merge)
set +e
MAIN=/burg-archive/glab/users/pg2328/legoESM
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=$MAIN
RUN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
ICEIC=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/Ice_initialization.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
for SHA in 4c32e98a2~1 4c32e98a2; do
  TAG=$(git -C $MAIN rev-parse --short "$SHA")
  WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/bisect_ice_$TAG
  echo; echo "########## PROBE $SHA ($TAG) ##########"
  git -C $MAIN worktree add --detach "$WT" "$SHA" 2>&1 | tail -1
  cd "$WT" || { echo "[probe] worktree FAILED for $SHA"; continue; }
  git log --oneline -1
  export PYTHONPATH="$(ls -d $WT/packages/*/ 2>/dev/null | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
  # legacy layout fallback (older SHAs may keep src/ only)
  [ -d "$WT/packages" ] || export PYTHONPATH="$WT:$WT/src"
  GA="--grid latlon_bathy --latlon-res 180x360 --mesh $ROOT/data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 --mask-marginal-seas"
  COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --years 0.0137 --snapshot-every-days 30 --diag-every-days 1"
  P45="--bottom-drag-scheme nemo_quadratic --iwm --iwm-forcing-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc --dm2dc --isf --isf-forcing-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc --bbl-adv"
  OUT=$ROOT/results/omip_nemo/nemolev_bisect_ice_$TAG
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
    $GA $COMMON $P45 --woa-init --woa-t $WTIC --woa-s $WSIC \
    --tracer-advection superbee --kpp-ri-crit 0.15 --ice-init $ICEIC \
    --sw-rgb-chl --output $OUT 2>&1 | tail -12
  echo "run_rc=${PIPESTATUS[0]}"
  echo "--- $TAG ice trajectory (day, thick_max) ---"
  awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next}{printf "%6.1fd thick_max=%8.1f\n",$h["day"],$h["ice_max_thick_m"]}' "$OUT/diag_timeseries.csv" 2>/dev/null
  V=$(awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next}{v=$h["ice_max_thick_m"]} END{print (v>50)?"RUNAWAY":"BOUNDED"}' "$OUT/diag_timeseries.csv" 2>/dev/null)
  echo "[probe-verdict] $TAG day-5 = $V"
done
echo; echo "=== BISECT PAIR DONE ==="; date