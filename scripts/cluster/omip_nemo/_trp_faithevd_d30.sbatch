#!/bin/bash
#SBATCH --job-name=trpfaithevd
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:45:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trpfaithevd_%j.out
#
# ONE VARIABLE OFF THE ORCA1-FAITHFUL BASELINE: NEMO's convective adjustment.
#
# WHY, and it is measured rather than argued.  NEMO's archived heat
# diffusivity in the equatorial Pacific at day 30 reaches 37.5 m2/s at the
# top interface, 16.7 at 2.1 m and 3.34 at 3.3 m.  Its viscosity at that same
# top interface is 7.6e-03, and the TKE closure ties the two through a
# Prandtl number clamped to [1, 10], so the closure could not put avt above
# 7.6e-03 there.  The archived value implies a Prandtl number of 2e-04.  The
# ONLY term in ORCA1 that writes a tracer diffusivity outside that band is
# enhanced vertical diffusion (rn_evd = 100, nn_evdm = 0, tracers only), and
# a five-day mean of a field that is 100 where it fires gives 37.5 at an
# occupancy near 0.37.  Our runs carry convection.scheme = 'none'.
#
# PRE-REGISTERED, on the nino3 layer dipole the baseline shows -- +1.104 C
# over 0-2 m, +0.423 over 2-10 m, -0.647 over 10-50 m, with the column
# integral matching NEMO to 0.003 C:
#   SURFACE (0-2 m) is the primary.  The warm bias must FALL.  Anything at or
#     below +0.80 counts as the branch doing what the measurement says it
#     does; no movement, or warming, REFUTES the mechanism at this depth and
#     that is a real possible outcome of this run.
#   10-50 m is the SECONDARY and is expected to improve only PARTLY.  NEMO's
#     diffusivity between 15 and 65 m is 1.3e-03 to 2.1e-03, which is
#     ordinary closure turbulence and NOT this branch -- the branch would
#     read near 100.  So most of the cold half is the separate energy
#     deficit, and if the cold half barely moves that CONFIRMS the split
#     rather than failing the arm.
#   COLUMN INTEGRAL must stay within 0.05 C of the baseline.  This branch
#     redistributes heat vertically and creates none; a drifting column
#     integral means something other than redistribution happened and the
#     run is not interpretable.
#   BLOWUP: any NaN, or global mean SST moving more than 0.5 C, stops the
#     reading entirely.
#
# RESIDUAL RISK, stated not discovered later: the baseline's manifest records
# git_dirty = true with an EMPTY patch list, so its tree carried uncommitted
# changes whose content was never captured.  The pin below reproduces the
# COMMIT, not that dirt.  _omip_wt's untracked files today are results,
# scratch cards and a text note, none of which enter the model, so the
# likeliest reading is that the dirt was untracked files -- but it is a
# residual risk on this comparison and not a proven equivalence.
#
# THE BASELINE CARD THIS WAS COPIED FROM, VERBATIM, FOLLOWS.
#
# THE ORCA1-FAITHFUL PRODUCTION CARD, REBUILT.
#
# The first version of this card was written FROM SCRATCH with about twenty
# flags and it went 100% NaN in velocity within a day. The cause was not the
# adopted physics: the card simply was not a working configuration. The
# known-good baseline (_trp_base2_d30.sbatch) carries about forty flags --
# the equatorial viscosity shape and profile, the Smagorinsky settings, the
# NEMO vertical coordinate, partial cells, the SMC03 pressure gradient,
# min-levels, TKE vertical mixing, bottom drag, the restoring stack, the
# geothermal and internal-wave forcing, the monthly IC -- and omitting any of
# that is not "a simpler run", it is a different and unstable model.
#
# So this card is now the BASELINE'S OWN FLAG BLOCK, verbatim, PLUS the four
# adopted sets appended at the end. That is the only honest way to claim the
# only difference is the adoption.
#
# THIS IS A NEW BASELINE, NOT AN ARM: the adopted sets move together by user
# authorisation, so the result is NOT attributable term by term against the
# pre-adoption baseline, and no single lever may be credited for a difference.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_evdarm_wt; cd $ROOT || exit 1
PIN=26acc418edb6c44ea9d44893cbe69bbf613ff6ce
HAVE=$(git -C $ROOT rev-parse HEAD)
# The baseline this arm is compared against ran from THIS commit.  The two
# trees have since diverged by thousands of lines of ocean physics, so running
# the arm anywhere else would change the model as well as the convection and
# the comparison would mean nothing.  Refuse rather than produce an
# uninterpretable number.
if [ "$HAVE" != "$PIN" ]; then
  echo "REFUSING: tree is at $HAVE, the baseline ran from $PIN"; exit 2
fi
echo "PIN OK: $HAVE"
echo "HEAD: $(git log --oneline -1)"
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
# trp_orca1faith_d30 is the pre-mask baseline every 2026-09-13/14 measurement
# references; the masked card writes beside it, never over it.
# trp_orca1faithmask_d30 = mask only (leg 1 to day 15, 9769155); this card now
# also carries NEMO's initial-condition time blend, so it writes beside both.
OUT=${OUT:-/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_faithevd_d${DAYS}}
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
  `# ---- initial condition = what NEMO starts from: fldread ln_tint puts the` \
  `# 1 January state at 0.5*December + 0.5*January of the monthly climatology ----` \
  --nemo-init-tint \
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
  `# ---- wet domain = the oracle's: the Great Lakes, Caspian and Lake Victoria` \
  `# (207 cells wet in mesh_mask.nc, land in NEMO's domain_cfg) become land ----` \
  --nemo-domain-mask \
  `# EOS stays wright: --eos nemo_teos10 is REFUSED because NEMO's TEOS-10` \
  `# takes Conservative Temperature and ABSOLUTE salinity while our tracers` \
  `# are potential temperature and PRACTICAL salinity.` \
  `# ---- THE ONE VARIABLE: NEMO's enhanced vertical diffusion ----` \
  `# ORCA1 &namzdf: ln_zdfevd=.true., rn_evd=100, nn_evdm=0.  zdfevd.F90:93` \
  `# OVERWRITES the tracer diffusivity with rn_evd wherever the static` \
  `# stability falls below -1e-12, and leaves the viscosity untouched` \
  `# because nn_evdm=0.  K-bg is 0 because NEMO contributes NOTHING where` \
  `# the branch does not fire, and this card's composition is additive, so` \
  `# any non-zero K-bg would raise the diffusivity in every stable column` \
  `# too and stop this being one variable.` \
  --convection enhanced_diffusion --convection-K-conv 100 --convection-K-bg 0 \
  --convection-n2-mode nemo_bn2 --convection-n2-eos teos10 \
  --convection-trigger hard --convection-n2-threshold=-1e-12 \
  `# NOT passed: --convection-two-level.  NEMO uses MIN(rn2, rn2b), which` \
  `# needs the BEFORE tracers a leap-frog outer step carries; this driver` \
  `# builds an rk3 outer step and REFUSES the flag rather than silently` \
  `# dropping the second arm.  NEMO's own key_RK3 build is now-only for the` \
  `# same reason, so this is a stated deviation that follows the oracle's` \
  `# RK3 configuration, and it can only make the branch fire LESS.` \
  --restart-save $OUT/restart_leg$LEG.npz \
  $([ "$LEG" = 2 ] && echo "--restart-from $OUT/restart_leg1.npz") \
  --output $OUT 2>&1 | tail -40
echo "=== card exit: $? ==="
echo "OUT=$OUT"
echo DONE
