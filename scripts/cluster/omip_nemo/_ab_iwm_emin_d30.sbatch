#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=ab_emin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ab_emin_%j.out
#
# One-variable A/B for the IWM TKE-floor fix (rn_emin 1e-6->1e-10, rmxl_min->1e-3
# under ln_zdfiwm).  Runs the SAME 30-day tripole+IWM card at a pinned SHA in an
# isolated worktree, scores nino3 bias at d10/d20/d30 vs NEMO GATEWAY recs 1/3/5.
# Arm FIX = 134cff43 (floor 1e-10); arm BASE = fd1f6be (floor 1e-6, parent).
# The ONLY code difference between the two SHAs is the floor in orca1_zdftke_config.
#   usage: sbatch _ab_iwm_emin_d30.sbatch <sha> <label>
set -uo pipefail
PIN=${1:?usage: sbatch _ab_iwm_emin_d30.sbatch <sha> <label>}
LABEL=${2:?need a label (fix|base)}
ROOT=/burg-archive/glab/users/pg2328/legoESM
WT=/burg-archive/glab/users/pg2328/legoESM_wt_emin_$LABEL
# fresh worktree pinned to the SHA (isolated tree so concurrent arms don't collide)
git -C "$ROOT" worktree prune
[ -d "$WT" ] && git -C "$ROOT" worktree remove --force "$WT" 2>/dev/null || true
git -C "$ROOT" worktree add --detach "$WT" "$PIN" || { echo "FATAL: worktree add failed"; exit 1; }
# gitignored runtime data (mesh etc.) is absent in the worktree -> symlink from main
mkdir -p "$WT/data"
ln -sfn "$ROOT/data/grids" "$WT/data/grids"
cd "$WT" || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$(git -C "$ROOT" rev-parse "$PIN")" ] || { echo "FATAL: HEAD $HEAD != pin"; exit 1; }
echo "ARM $LABEL at $(git log --oneline -1)"
# confirm the floor this arm will use
grep -nE "rn_emin, rmxl_min = 1.0e-1?0" scripts/run/run_omip_core2.py | head -2

export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate:$WT/scripts/run"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/ab_emin_$LABEL

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 \
  --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.08214 --snapshot-every-days 10 --diag-every-days 5 \
  --tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate \
  --bottom-drag-scheme nemo_quadratic \
  --iwm --iwm-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc \
  --dm2dc --isf --isf-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc \
  --bbl-adv --woa-init \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
    $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --tripole-vmix tke --ice-init $NEMOIN/Ice_initialization.nc \
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic \
  --tke-kappa-convention veros_sqrte --sw-rgb-chl --ice-ocean-heat-coeff 70 \
  --gateway-transports --mle --mle-ce 0.06 \
  --output $OUT 2>&1 | tail -25
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }

export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
echo "=== nino3 bias trajectory (arm $LABEL) vs matched GATEWAY recs ==="
for pair in "0010 1" "0020 3" "0030 5"; do
  set -- $pair
  SNAP=$OUT/snapshot_day$1.npz
  [ -f "$SNAP" ] || { echo "no day $1 snapshot"; continue; }
  echo "### day $1 vs GATEWAY rec $2 ###"
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GW_T" --nemo-time-idx "$2" --freeze-clamp-C -1.8 \
    --sss-faithful --grid-label ab_emin_${LABEL}_d$1 \
    --output-dir $ROOT/results/omip_nemo/compare_ab_emin_${LABEL}_d$1 2>&1 \
    | grep -E "nino3_5S5N|\[SST\].*rmse|\[SSS\].*rmse" | head -4
done
echo "=== DONE arm $LABEL ==="
