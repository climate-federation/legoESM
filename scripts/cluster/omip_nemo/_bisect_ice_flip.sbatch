#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=bisect_flip
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/bisect_flip_%j.out
# LATLON ICE-RUNAWAY BISECT, ROUND 2 — file-level flip at 38483caa8.
# Round 1: 2c875755a BOUNDED, 4c32e98a2 BOUNDED (both day-5 <50 m) -> the
# culprit rides the 38483caa8 main-merge.  Static reads exonerate the obvious
# ice-file changes for OUR config (transport hunk = Voronoi-only; dynamics =
# EVP/mEVP-only; free_drift unchanged; closing-rate/ridging gated OFF), so
# this arm decides ICE-PACKAGE vs DRIVER/COUPLING:
#   ARM A: tree = 38483caa8, but packages/ice/ REVERTED to 4c32e98a2.
#     BOUNDED  -> culprit IS in packages/ice (split files next).
#     RUNAWAY  -> culprit in the merge's driver/ocean side (run_omip_core2
#                 ice-ocean blend / stress partition / currents / F_ocean).
set +e
MAIN=/burg-archive/glab/users/pg2328/legoESM
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=$MAIN
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
ICEIC=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/Ice_initialization.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/bisect_flipA
git -C $MAIN worktree add --detach "$WT" 38483caa8 2>&1 | tail -1
cd "$WT" || exit 1
git checkout 4c32e98a2 -- packages/ice/
echo "=== tree: 38483caa8 + packages/ice reverted to 4c32e98a2 ==="
git status --short | head -8
export PYTHONPATH="$(ls -d $WT/packages/*/ 2>/dev/null | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
GA="--grid latlon_bathy --latlon-res 180x360 --mesh $ROOT/data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 --mask-marginal-seas"
COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --years 0.0137 --snapshot-every-days 30 --diag-every-days 1"
P45="--bottom-drag-scheme nemo_quadratic --iwm --iwm-forcing-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc --dm2dc --isf --isf-forcing-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc --bbl-adv"
OUT=$ROOT/results/omip_nemo/nemolev_bisect_flipA
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  $GA $COMMON $P45 --woa-init --woa-t $WTIC --woa-s $WSIC \
  --tracer-advection superbee --kpp-ri-crit 0.15 --ice-init $ICEIC \
  --sw-rgb-chl --output $OUT 2>&1 | tail -12
echo "run_rc=${PIPESTATUS[0]}"
echo "--- flipA ice trajectory ---"
awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next}{printf "%6.1fd thick_max=%8.1f\n",$h["day"],$h["ice_max_thick_m"]}' "$OUT/diag_timeseries.csv" 2>/dev/null
V=$(awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next}{v=$h["ice_max_thick_m"]} END{print (v>50)?"RUNAWAY":"BOUNDED"}' "$OUT/diag_timeseries.csv" 2>/dev/null)
echo "[flip-verdict] 384+ice@4c32e day-5 = $V  (BOUNDED->ice-package culprit; RUNAWAY->driver/coupling culprit)"
echo "=== DONE ==="; date