#!/bin/bash
#SBATCH --job-name=fesombase2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --constraint="a40|rtx8000"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesombase2_%j.out
# NEW PRODUCTION BASELINE, FESOM (user decision 8, 2026-09-06): NEMO's 75
# levels with NEMO's 30 m minimum column (mesh_nemo75_30m: floor at the
# 28.65 m interface, min 15 levels) instead of FESOM's 5th interface (4.5 m
# on this ladder).  Everything else = fesom_nemo75v_d30.
# PRE-REGISTERED (day 30 vs GATEWAY rec 5, vs fesom_nemo75v_d30): SST rmse
#   within 0.05 of 0.65; nino3 within 0.2 of +1.39; SSS rmse within 0.05 of
#   0.88; no blow-up (the 4.5 m columns were the CFL worry).
set -uo pipefail
PIN=${1:?usage: sbatch _fesom_base2_d30.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
echo "fesom_jax: $(git -C /burg-archive/glab/users/pg2328/fesom_jax_repo log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:/burg-archive/glab/users/pg2328/fesom_jax_repo:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
MI=$ROOT/data/fesom2_core2/mesh_ic/core2_mesh_ic
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid fesom \
  --fesom-mesh-dir "$ROOT/data/fesom2_core2/mesh_nemo75_30m" \
  --woa-init \
  --nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
    $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
  --dm2dc --sw-rgb-chl \
  --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --ice-lead-freeze-source nemo_qlead \
  --fesom-vmix legoesm_tke --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 \
  --tke-prognostic --tke-kappa-convention veros_sqrte \
  --ice-init $NEMOIN/Ice_initialization.nc --ice-ocean-heat-coeff 70 \
  --sss-restore --sss-restore-channel water_flux \
  --sss-restore-normalization live_s \
  --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo \
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --runoff --river-mouth-restoring-gate \
  --iwm --iwm-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc \
  --forcing-ramp-days 5 \
  --dt 1800 --years 0.0822 --snapshot-every-days 15 \
  --output $ROOT/results/omip_nemo/fesom_base2_d30 2>&1 | tail -14
run_rc=${PIPESTATUS[0]}
echo "=== run rc=$run_rc"
[ "$run_rc" -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }
echo "=== run rc=$? ==="
export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
for pair in "30 5"; do
  set -- $pair; d=$1; rec=$2
  SNAP=$ROOT/results/omip_nemo/fesom_base2_d30/snapshot_day00$d.npz
  [ -f "$SNAP" ] || SNAP=$ROOT/results/omip_nemo/fesom_base2_d30/snapshot_final.npz
  echo "=== day $d vs GATEWAY rec $rec ==="
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GW_T" --nemo-time-idx $rec --freeze-clamp-C -1.8 \
    --sss-faithful --grid-label fesom_base2_d$d \
    --output-dir $ROOT/results/omip_nemo/compare_fesombase2_d00$d 2>&1 | tail -38
done
echo DONE
