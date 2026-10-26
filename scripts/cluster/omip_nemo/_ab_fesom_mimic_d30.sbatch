#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fmimic30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fmimic30_%j.out
#
# ============================================================================
# FESOM-MIMIC CARD — why does the third grid disagree?
#
# THE QUESTION (user, 2026-08-18): tripole and MPAS agree almost exactly;
# FESOM2 does not, even on matched NYF forcing (SST 1.56 vs 0.85; a growing
# warm-SH / cold-NH dipole, Antarctic MLD ~19 m shallow).  The premise "all
# three share the same physics" is FALSE in code: fesom-jax runs its own
# CVMix classical-Gaspar TKE (Neumann surface BC, no Langmuir, no etau
# penetration, no IWM, mxl BD-2) while tripole/MPAS share the ORCA1 zdftke
# card.  This arm runs OUR tripole with those branches switched to the
# fesom-jax settings — a COMPOSITE treatment, deliberately labelled as such:
# attribution is to the card-as-a-unit; a positive result gets a per-variable
# decomposition next (the Wangara method).
#
# CARD DELTA vs control nemolev_trp_signedn2_d30 (everything else verbatim):
#   --tke-surface-bc veros_flux   (fesom: Neumann flux BC, not Dirichlet)
#   --tke-lc off                  (fesom: no Langmuir)
#   --tke-etau none               (fesom: no penetration term)
#   --tke-mxl-choice 4            (fesom BD choice-2 ~ our 4; APPROXIMATE)
#   NO --iwm                      (fesom: no wave mixing)
#
# PRE-REGISTERED, day 30 vs NEMO m1 (control values bracketed):
#   SH_midlat SST bias [-0.4]: mimic reproduces FESOM2's +1.5-class warm
#     shift => card explains the dipole (CONFIRM); moves < +0.5 => the gap is
#     in fesom-jax's IMPLEMENTATION or dycore, not the closure card (REFUTE).
#   Antarctic MLD bias [~+1]: shallows by >= 10 m toward FESOM2's -19 =>
#     Langmuir/etau/IWM are the MLD lever (CONFIRM).
#   WATCH nino3 as a tripwire only (this arm does not touch viscosity).
# ============================================================================
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
PIN=${1:?usage: sbatch _ab_fesom_mimic_d30.sbatch <pinned-sha>}
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/nemolev_trp_fesommimic_d30

echo "=== gate: card-knob tests ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/unit/test_run_omip_core2_eq_visc_shaping.py \
  -p no:cacheprovider 2>&1 | tail -3
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate failed rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 \
  --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.0822 --snapshot-every-days 10 --diag-every-days 5 \
  --tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate \
  --bottom-drag-scheme nemo_quadratic \
  --dm2dc --isf --isf-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc \
  --bbl-adv --woa-init \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
    $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --tripole-vmix tke --ice-init $NEMOIN/Ice_initialization.nc \
  --tke-surface-bc veros_flux --tke-mxl-choice 4 --tke-prognostic \
  --tke-lc off --tke-etau none \
  --tke-kappa-convention veros_sqrte --sw-rgb-chl --ice-ocean-heat-coeff 70 \
  --gateway-transports --mle --mle-ce 0.06 \
  --output $OUT 2>&1 | tail -25
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed -- not scoring"; exit 1; }

GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
SNAP=$OUT/snapshot_day0030.npz
[ -f "$SNAP" ] || { echo "FATAL: no day-30 snapshot"; exit 1; }
export JAX_PLATFORMS=cpu
echo "=== score vs NEMO m1 ==="
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label fesommimic_d30 \
  --output-dir $ROOT/results/omip_nemo/compare_fesommimic_d30_m01 2>&1 | tail -45
echo "=== compare rc=$?"
