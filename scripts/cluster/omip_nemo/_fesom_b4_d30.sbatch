#!/bin/bash
#SBATCH --job-name=fesom_b4_d30
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesomb4_%j.out
#
# FIRST SCOREABLE FESOM FAITHFULNESS ARM (stage B4 complete): the full
# harmonized surface card the tripole/MPAS lanes run, within fesom-lane
# support — shared CORE-II bulk + dm2dc + RGB chl + legoESM prognostic ice
# (free-drift, 1 cat) + SSS restore (water_flux channel, NEMO nn_sssr=2) +
# Dai-Trenberth runoff (node-adjacency spread) + NEMO monthly-init IC.
# Not yet on fesom (ledger): isf/iwm/bbl (B6), geothermal, mle; closure =
# fesom CVMix-TKE (single-column twin pending); EOS-80; VCPW 4.2e6.
# PRE-REGISTERED vs GATEWAY rec 5 (d26-30): CONFIRM harmonization progress
# if global SST rmse < 1.0 (old own-forcing FESOM arm: 1.016 with -0.65
# bias) AND Arctic SSS |bias| < 0.6 (restore+runoff+ice now active);
# REFUTE/flag any band diverging > 1.5 C.
set -uo pipefail
PIN=${1:?usage: sbatch _fesom_b4_d30.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
echo "fesom_jax: $(git -C /burg-archive/glab/users/pg2328/fesom_jax_repo log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:/burg-archive/glab/users/pg2328/fesom_jax_repo:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
MI=$ROOT/data/fesom2_core2/mesh_ic/core2_mesh_ic
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid fesom \
  --fesom-mesh-dir "$MI/mesh_core2" --fesom-ic-dir "$MI/ic_core2" \
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
  --forcing-ramp-days 5 \
  --dt 1800 --years 0.0823288 --snapshot-every-days 30 \
  --output $ROOT/results/omip_nemo/fesom_b4_d30 2>&1 | tail -14
echo "=== run rc=$? ==="
export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
SNAP=$ROOT/results/omip_nemo/fesom_b4_d30/snapshot_day0030.npz
[ -f "$SNAP" ] || { echo "FATAL: day-30 snapshot missing"; exit 1; }
echo "=== day 30 vs GATEWAY rec 5 ==="
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GW_T" --nemo-time-idx 5 --freeze-clamp-C -1.8 \
  --sss-faithful --grid-label fesom_b4_d30 \
  --output-dir $ROOT/results/omip_nemo/compare_gw_fesom_b4_d30 2>&1 | tail -38
echo DONE
