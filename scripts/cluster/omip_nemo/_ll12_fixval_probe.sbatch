#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=ll12_fixval
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ll12_fixval_%j.out
# LATLON ICE-RUNAWAY: FIX-VALIDATION PROBE at HEAD (lead_freeze_latent=credit).
# flip2 arm-1 pinned sea_ice.py; the hunk read pins the lead-freeze SST gate
# (audit #2) as the only volume-affecting live change; the working mechanism
# is the supercooling latch (gate x L_f extraction on an un-floored ocean).
# --freeze-floor is surrogate-exclusive with prognostic ice (ll10 rejected),
# so this probe reverts ONLY the two gate jnp.where blocks in a detached
# HEAD worktree: BOUNDED day-5 -> gate confirmed as the enabler (then build
# the conserving frazil-style fix: floor + ice credit); RUNAWAY -> the
# mechanism is elsewhere in sea_ice.py (back to hunk flips).
set +e
MAIN=/burg-archive/glab/users/pg2328/legoESM
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=$MAIN
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
ICEIC=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/Ice_initialization.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/ll12_fixval
git -C $MAIN worktree add --detach "$WT" omip-arctic-vfs-tke-fix 2>&1 | tail -1
cd "$WT" || exit 1
git log --oneline -1
export PYTHONPATH="$(ls -d $WT/packages/*/ 2>/dev/null | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
GA="--grid latlon_bathy --latlon-res 180x360 --mesh $ROOT/data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 --mask-marginal-seas"
COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --years 0.0137 --snapshot-every-days 30 --diag-every-days 1"
P45="--bottom-drag-scheme nemo_quadratic --iwm --iwm-forcing-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc --dm2dc --isf --isf-forcing-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc --bbl-adv"
OUT=$ROOT/results/omip_nemo/nemolev_ll12_fixval
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  $GA $COMMON $P45 --woa-init --woa-t $WTIC --woa-s $WSIC \
  --tracer-advection superbee --kpp-ri-crit 0.15 --ice-init $ICEIC \
  --sw-rgb-chl --output $OUT 2>&1 | tail -10
echo "run_rc=${PIPESTATUS[0]}"
awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next}{printf "%6.1fd thick_max=%8.1f\n",$h["day"],$h["ice_max_thick_m"]}' "$OUT/diag_timeseries.csv" 2>/dev/null
V=$(awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next}{v=$h["ice_max_thick_m"]} END{print (v>50)?"RUNAWAY":"BOUNDED"}' "$OUT/diag_timeseries.csv" 2>/dev/null)
echo "[ll12-verdict] HEAD + lead_freeze_latent=credit day-5 = $V  (BOUNDED = runaway fix validated end-to-end)"
echo "=== DONE ==="; date