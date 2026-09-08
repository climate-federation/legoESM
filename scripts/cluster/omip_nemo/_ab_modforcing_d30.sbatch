#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=modfrc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=23:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/modfrc_%j.out
#
# ============================================================================
# CORRECTED CORE-II FORCING: the one test.  ONE VARIABLE vs signedn2_d30.
#
# WHAT CHANGED. Our forcing cache was built from the RAW CORE-II fields
# (U_10, V_10, T_10, Q_10, SWDN, LWDN, RAIN+SNOW). NEMO ORCA1's namelist reads
# the Large & Yeager bias-corrected _MOD fields. This arm points
# --forcing-path at a cache rebuilt from those; every other flag matches the
# control byte for byte.
#
# EVIDENCE, both reviewers (codex 9450473, GLM):
#   paired speed sqrt(u^2+v^2), this file's own grid, no remapping:
#     nino3  base 4.736  _MOD 6.279  ratio 1.326, turning angle 9.8 deg
#     global base 6.282  _MOD 6.838  ratio 1.089, turning angle 3.1 deg
#   Not a rotation -- a SPEED correction, the QuikSCAT tropical adjustment for
#   NCEP-1's well-known 20-30 percent weak equatorial trades. Observed TAO and
#   scatterometer climatologies put nino3 near 6-7 m/s. Stress goes as the
#   square: 6.28 m/s implies 0.054 N/m2 against NEMO's measured taum of
#   0.0533; the raw 4.74 implies 0.027, about half.
#
# WHY IT MIGHT FIX THE COLD TONGUE. nino3 is our largest error (+3.03 C, three
# times any other SST band), the EUC is 29 percent weak, and cutting lateral
# viscosity 6.8x moved nino3 by 0.09 C -- i.e. the momentum SOURCE was short,
# not the dissipation.
#
# PRE-REGISTERED, day 30 vs signedn2_d30 (nino3 +3.192, EUC 200E 0.336 vs
# NEMO 0.496, tropics SST bias +0.635):
#   nino3 : CONFIRMED if it falls below +2.0; PARTIAL +2.0 to +2.7; REFUTED if
#           it stays above +2.9, which would mean the wind deficit was not the
#           cold tongue's cause and the search moves on.
#   EUC 200E core: CONFIRMED >= 0.42 (halfway to NEMO). The undercurrent is
#           the mechanism, so it must respond or the story is wrong even if
#           SST moves for another reason.
#   SH_midlat / NH_midlat SST: REPORT either way. A 9 percent global wind
#           increase changes the whole ocean, so a tropical win bought with a
#           midlatitude loss is a trade, not a fix.
#   max|u| : stability check, must stay under 2.5. Stronger winds mean more
#           momentum in, and the card has only ever been run at the weaker
#           forcing.
#
# NOT A TUNING KNOB. This is a correctness fix -- the oracle reads these
# fields. If it degrades the model, that is a finding about the model, not a
# reason to go back to forcing it with winds NEMO does not use.
# ============================================================================
set -uo pipefail
ARM=rfw
VISC_FLAGS="--C-smag-lap 3.0"
PIN=${1:?usage: sbatch _ab_modforcing_d30.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
# Codex 9442707 MAJOR: every coefficient arm wrote the SAME directory, so a
# second arm silently overwrote the first one's snapshots and the comparison
# would have been scored against whichever ran last. Key the output on the
# coefficient under test.
OUT=$ROOT/results/omip_nemo/nemolev_trp_modfrc_d30

echo "=== gate: unit tests for the reduction semantics + CLI wiring ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_polar_cap_boost.py \
  tests/unit/test_run_omip_core2_eq_visc_shaping.py 2>&1 | tail -3
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate tests failed rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  $VISC_FLAGS --smag-cfl-safety 0.125 \
  --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.0822 --snapshot-every-days 1 --diag-every-days 5 \
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
  --gateway-transports --mle --mle-ce 0.06 --forcing-path /burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/core2_nyf_mod \
  --output $OUT 2>&1 | grep --line-buffered -E "fwbudget|\[setup\]|\[diag\]|Error|error" | cat
run_rc=$?
echo "=== run rc=$run_rc"
# codex 9430935 MAJOR-4: never score a failed or snapshot-less arm -- a stale
# or missing snapshot scored silently looks like a completed arm.
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc -- not scoring"; exit 1; }

GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
GU=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_U.nc
EXP_GATEWAY_V=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_V.nc
SNAP=$OUT/snapshot_day0030.npz
[ -f "$SNAP" ] || { echo "FATAL: $SNAP missing -- not scoring"; exit 1; }

echo "=== score: Southern-Ocean ice volume tendency (the quantity under test) ==="
JAX_PLATFORMS=cpu $PY scripts/validate/ocean_fidelity/so_ice_volume_tendency.py \
  --run-dir "$OUT" --days 6 --label "modfrc" 2>&1 | tail -14
export JAX_PLATFORMS=cpu

echo "=== score: Z20 + upwelling + EUC (matched recs 5:6, days 26-30) ==="
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline.py \
  --legoesm-snapshot "$SNAP" --nemo-gridt "$GT" --nemo-month 1 \
  --label modfrc_d30 --nemo-wfile "$GW" --nemo-w-var wo --nemo-w-recs 5:6 \
  --nemo-ufile "$GU" --meridional-lon 240,260 \
  --nemo-vfile "$EXP_GATEWAY_V" 2>&1 | tail -140
echo "=== eqtherm rc=$?"

echo "=== score: full skill vs NEMO m1 (ENSO boxes) ==="
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
  --grid-label modfrc_d30 \
  --output-dir $ROOT/results/omip_nemo/compare_modfrc_d30_m01 2>&1 | tail -40
echo "=== compare rc=$?"
