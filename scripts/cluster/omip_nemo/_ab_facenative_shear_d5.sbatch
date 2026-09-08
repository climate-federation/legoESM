#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=shearab
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx8000|a40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/shearab_%j.out
#
# ============================================================================
# NEMO's FACE-NATIVE SHEAR PRODUCTION — the last card gap, and the only one
# that cannot be tested offline.  Two 5-day tripole arms, ONE variable.
# ============================================================================
#
# WHY A GRID RUN.  tke_vertical_mixing needs u_face_now/v_face_now/
# u_face_before/v_face_before AND face_masks_3d -- raw C-grid face state plus
# per-level wumask/wvmask and NEMO's coastal masks (zdfsh2.F90:78-94).  The
# offline column probe has flattened the face topology away, so it raises
# rather than reporting a number.  Every other gap on this card was settled
# offline for minutes; this one costs GPU because there is no alternative.
#
# WHY IT IS THE REMAINING SUSPECT.  With the card's config space exhausted
# (a4f36f385), our diffusivity still falls 18% over an hour from NEMO's own
# state while NEMO's rises 5%.  Shear production is a SOURCE, so it acts
# during the step and not at zero steps -- which is exactly where the error
# is.  And averaging velocities to cell centres before differencing SMOOTHS,
# making our source systematically weaker than NEMO's face-native form.  The
# coastal doubling makes the effect largest near land, which is also where
# the standing three-grid ask puts the most weight.
#
# COST GATE. d90 measures ~30 h, so d5 is ~1.7 h per arm; two arms in
# parallel is ~1.7 h wall on one GPU each. The cheaper offline test does not
# exist for this term -- that is the justification, not a preference.
#
# PRE-REGISTERED, scored with compare_three_way_nemo.py so the near_land /
# near_land_arctic / near_land_nonarctic subdomains are reported (the coastal
# doubling is the distinctive part of this term):
#   CONFIRMED  : near_land MLD deepens by >= 5 m vs the control AND the global
#                or Southern Ocean MLD bias moves TOWARD NEMO.
#   REFUTED    : every subdomain moves < 1 m -- the term is negligible at 1
#                degree and the step error is elsewhere.
#   MIXED      : near-land moves but the basin-scale bias moves AWAY from
#                NEMO -- report both, pin nothing.
# The mechanism predicts MORE mixing, so a deeper mixed layer. Our day-90
# Southern Ocean is already too deep, so "helps the step, hurts day 90" is a
# live outcome and gets reported as a trade, not a win.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM
cd $ROOT || exit 1
echo "HEAD:    $(git -C $ROOT rev-parse HEAD)"
echo "SUBJECT: $(git -C $ROOT log --oneline -1)"
git -C $ROOT status --porcelain -- scripts/run/run_omip_core2.py \
    packages/ocean/legoesm/ocean/physics/vertical_mixing/ > /tmp/shearab_dirty.$$
if [ -s /tmp/shearab_dirty.$$ ]; then
  echo "FATAL: model/driver paths dirty at launch:"; cat /tmp/shearab_dirty.$$
  rm -f /tmp/shearab_dirty.$$; exit 1
fi
rm -f /tmp/shearab_dirty.$$
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free \
  --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
IN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
W=/burg/glab/users/pg2328/legoESM/data/woce_gouretski

run_arm () {
  local out="$1"; shift
  XLA_PYTHON_CLIENT_PREALLOCATE=false LEGOESM_JIT_CACHE_DIR="" \
  JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
    --grid tripole --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
    --C-smag-lap 3.0 --smag-cfl-safety 0.125 --ew-cyclic-overlap --nemo-vertical \
    --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv --momentum-rk3 \
    --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
    --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice \
    --prognostic-ice-dynamics free_drift --years 0.0137 --snapshot-every-days 30 \
    --diag-every-days 5 --tracer-advection superbee --runoff-depth-nemo-ini \
    --river-mouth-restoring-gate --bottom-drag-scheme nemo_quadratic \
    --iwm --iwm-forcing-file $IN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc \
    --dm2dc --isf --isf-forcing-file $IN/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc \
    --bbl-adv --woa-init --woa-t $W/woce_t_woa57_annual.nc --woa-s $W/woce_s_woa57_annual.nc \
    --nemo-monthly-init \
      $IN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
      $IN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
    --sss-restore-file $IN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
    --tripole-vmix tke --ice-init $IN/Ice_initialization.nc \
    --tke-surface-bc nemo_dirichlet --tke-prognostic --sw-rgb-chl \
    --ice-ocean-heat-coeff 70 "$@" \
    --output $ROOT/results/omip_nemo/$out 2>&1 | tail -25
  echo "=== arm $out rc=$? ==="
}

run_arm nemolev_trp_shearctl_d5
run_arm nemolev_trp_shearface_d5 --tke-shear-production nemo_face_native

echo; echo "########## THREE-WAY SCORE (near-land subdomains are the point)"
JAX_PLATFORMS=cpu $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
  --tripole $ROOT/results/omip_nemo/nemolev_trp_shearctl_d5/snapshot_final.npz \
  --mpas    $ROOT/results/omip_nemo/nemolev_trp_shearface_d5/snapshot_final.npz \
  --label-tripole "squared_centered (control)" \
  --label-mpas "nemo_face_native" \
  --nemo-gridt /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc \
  --nemo-month 1 --res-deg 1.0 \
  --out-dir $ROOT/results/omip_nemo/shear_facenative_d5
echo "=== score rc=$? ==="
echo "=== ALL DONE ==="
