#!/bin/bash
#SBATCH --job-name=orca1_z0
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/orca1_z0_%j.out
#
# ONE-VARIABLE ARM: THE SURFACE TKE HELD AT NEMO'S z=0, NOT THE FIRST
# INTERIOR INTERFACE (#1690).
#
# Built from orca1_faithful_card.sbatch with cp -f; the printed diff against it
# has exactly four OPERATIONAL differences -- job name, log name, output
# directory, and the one flag below (this header also differs; comments do
# not run). The code is the SAME tree the baseline ran from (3c37fea09), in
# which --tke-surface-bc-level already exists, so nothing but the placement
# differs.
#
# WHY. The baseline trp_orca1faith_d30 resolved tke_surface_bc_level =
# interior_pinned, which pins the surface TKE at the first interior interface
# near 10 m and leaves the top layer with no surface source at all. Measured
# consequence at 220E (d332d51a1): our K_M is 5-6% of NEMO's avm through 3-15 m
# -- ours collapses from 5.99e-03 at 1 m to 1.02e-03 at 3.3 m while NEMO's
# peaks at 10.8 m at 3.18e-02 -- and the shear is surface-trapped (ours
# 2.37e-02 vs 4.36e-03 at 3.3 m, flat with depth where NEMO's grows to 1.6e-02
# near 60 m). GLM showed the shear PRODUCTION is nearly identical in both
# models (~6e-7 m2/s3), so NEMO's near-surface turbulence is imported from the
# surface condition, not generated locally; ours sits at shear equilibrium.
# Codex ranked the same cause first. Every other coefficient already matches
# the oracle (surface_bc nemo_dirichlet, lc_coeff 0.25, etau_frac 0.08,
# etau_mode below_ml). The forcing is fully exonerated by measurement: same
# wind (0.05% at 220E), relative wind off both sides, drag within 1.3%.
#
# PRE-REGISTERED, on the day-30 snapshot against GATEWAY record 5, using the
# committed instruments _probe_vertical_momentum.sbatch and
# _probe_euc_profiles.sbatch, which live in /burg-archive/glab/users/pg2328/
# _omip_merge/scripts/cluster/omip_nemo/ (NOT in this checkout -- codex) and
# are run from there, pointed at this OUT. Revised after GLM review
# closed two gaps and separated the fast from the slow response:
#
# THE DECISION IS GATED ON THE FAST TERMS ONLY. K_M and shear adjust in days;
# the undercurrent needs the ~2-month first-baroclinic Kelvin crossing of the
# Pacific (~2.5 m/s), so at day 30 it is mid-transient by construction and
# cannot carry a confirm/refute verdict.
#   CONFIRM  -- K_M through 3-15 m at 220E >= 80% of NEMO's avm (the
#               mechanism's actual prediction: a missing top-layer source with
#               the pinned value equal to NEMO's predicts near-parity), AND
#               surface shear at 3.3 m falls from 2.37e-02 to below 1.0e-02.
#   PARTIAL  -- K_M in 15-80%, OR K_M >= 80% with shear still above 1.0e-02:
#               ANY proper subset of the fast criteria firing. Placement is a
#               real lever but not the whole story; the surface VALUE (rn_ebb)
#               or the dissipation length is next. Report, no fix.
#   REFUTE   -- K_M through 3-15 m stays below 15% of avm.
#   There is now no outcome that satisfies none of these.
#
# THE UNDERCURRENT IS SCORED AS A TREND, NOT A THRESHOLD: the 220E core speed
# and the 0-400 m net zonal transport from the day-20 snapshot to day 30. A
# rising slope is consistent with the mechanism; a flat or falling slope
# despite recovered K_M is the "second defect downstream" signal. A 39%->45%
# reading at day 30 alone triggers NOTHING.
#
# TRIPWIRE -- K_M overshooting NEMO's by >2x in the top 10 m means the
# surface VALUE was calibrated under the old placement (the #1690 review
# warned value and placement may be one transferred unit) and must be
# re-examined before anything is adopted.
#
# ATTRIBUTION CONFOUND, NAMED BY GLM AND NOT CLOSED BY THIS ARM: moving the
# Dirichlet from 10 m to 0 m does two things at once -- the top layer gains a
# surface source, AND the old 10 m pin (which was also a CLAMP on that
# interface, sitting where shear is ~5x larger) is removed. The effective
# source magnitude is therefore not held fixed. A recovery could come from the
# clamp's removal rather than from the new source. The post-run scoring MUST
# include the TKE production budget K*S^2 through 1-10 m to separate the two;
# without it, CONFIRM means "the placement change works", not "the mechanism
# is the top-layer source".
#
# INSTANTANEOUS vs 5-DAY-MEAN caveat applies as everywhere on this thread:
# read structure and factors, not percent-level differences.
#
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
# ATTRIBUTION GATE (codex): a mutable checkout that merely prints HEAD does
# not guarantee the baseline's code ran. Refuse to start on any other commit.
# The gate is on CODE IDENTITY with the baseline commit, not on HEAD itself:
# committing this card advances HEAD without touching any runtime path, and
# a HEAD check would then refuse its own card.
BASE=3c37fea09
git diff --quiet "$BASE" -- scripts/run packages src || { echo "REFUSING: runtime code differs from baseline commit $BASE"; git diff --stat "$BASE" -- scripts/run packages src | tail -5; exit 2; }
[ -z "$(git status --porcelain -- scripts/run packages src 2>/dev/null)" ] || { echo "REFUSING: uncommitted changes under scripts/run, packages or src"; exit 2; }
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
# TWO LEGS, as the baseline did: ~0.70 h per simulated day means 30 days is
# ~21 h, well past the 12 h partition cap. --years is the ABSOLUTE target, so
# leg 2 asks for the FULL 30 days and resumes from leg 1's restart.
DAYS=${DAYS:-30}
LEG=${LEG:-1}
HALF=$($PY -c "print(${DAYS}/2)")
TARGET=$([ "$LEG" = 1 ] && echo $HALF || echo $DAYS)
YEARS=$($PY -c "print(${TARGET}/365.0)")
OUT=${OUT:-/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1z0_d${DAYS}}
echo "DAYS=$DAYS LEG=$LEG TARGET=$TARGET YEARS=$YEARS OUT=$OUT"

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 \
  --C-smag-lap 3.0 --smag-cfl-safety 0.125 \
  --A-h 2e4 --A-h-profile-file $NEMOIN/orca1_inputs/data_repository/input_fields/eddy_viscosity_3D.nc \
  --visc-schedule '0:1e4:3.0,8:1e4:1.5,16:1e4:0.5' \
  --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff \
  --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --ice-lead-freeze-source nemo_qlead \
  --years $YEARS --snapshot-every-days 5 --diag-every-days 1 \
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
  --gateway-transports --mle --mle-ce 0.06 --kprofile-snapshots \
  `# ---- THE FOUR ADOPTED SETS, and nothing else, appended ----` \
  --ice-freeze-scheme unesco \
  --gm-treguier --gm-kappa-min 0 \
  --gm-slope-scheme nemo_iso_lap --gm-slope-positions nemo_native \
  --gm-bolus-advection through_fct --gm-msc-stabilize \
  --redi-coefficient nemo21 --redi-aht0 900 \
  --ice-exchange nemo_si3 --ice-snow bulk --ice-snow-flooding on \
  --forcing-time-interp linear --chl-remap nemo_scrip \
  `# ---- THE ONE VARIABLE UNDER TEST: surface TKE held at NEMO z=0, not the first interior interface (#1690) ---- ` \
  --tke-surface-bc-level nemo_z0 \
  `# EOS stays wright: --eos nemo_teos10 is REFUSED because NEMO's TEOS-10` \
  `# takes Conservative Temperature and ABSOLUTE salinity while our tracers` \
  `# are potential temperature and PRACTICAL salinity.` \
  --restart-save $OUT/restart_leg$LEG.npz \
  $([ "$LEG" = 2 ] && echo "--restart-from $OUT/restart_leg1.npz") \
  --output $OUT 2>&1 | tail -40
echo "=== card exit: $? ==="
echo "OUT=$OUT"
echo DONE
