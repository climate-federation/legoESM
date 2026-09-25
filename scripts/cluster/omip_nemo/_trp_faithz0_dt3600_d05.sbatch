#!/bin/bash
#SBATCH --job-name=trpdt3600
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trpdt3600_%j.out
#
# DOES THE ORACLE'S TIMESTEP SELECT THE TURBULENT BRANCH?
#
# At day 90 our cold-tongue turbulent energy is 0.11 of the oracle's at the
# first metre and dead below 7 m, while the oracle carries a subsurface
# turbulent layer at 15-21 m that our column never forms (e2c11683b). The
# closure is verified faithful given the energy (the oracle's own state
# through our closure returns its own viscosity at x0.985), and the lens is
# already five times the oracle's by day 5, so whatever separates the two
# columns acts in the first days. Of the candidate entry points the dual
# review ranked, the cheapest to test is the timestep: the control ran at
# 150 s and the oracle at 3600 s, and in a column whose turbulent state is
# decided early, a 24x coarser implicit step changes how much of the first
# days' heating and wind energy is mixed at each step.
#
# ONE VARIABLE. This card is the production control card with --dt 3600 in
# place of --dt 150 and nothing else in the physics flag block changed. It
# runs from the SAME pinned commit as the day-30 control (3b5bdfb2a, the
# tree _evdarm_wt is detached there) so the comparison is code-identical.
# Five days, one leg, 120 steps.
#
# PRE-REGISTERED, on the cold-tongue lens T(0-2 m) - T(10-50 m) at day 5 in
# nino3 (control 1.48 K, oracle 0.31 K), read with global_tracer_content.py
# exactly as nino3_dipole_faithz0.json was:
#   * <= 0.6 K  -> the timestep SELECTS THE BRANCH: at the oracle's step our
#     column stays turbulent, our physics is faithful within numerics, and
#     the remaining question is which of the two steps is the more accurate
#     representation of the ocean -- not which model is wrong.
#   * >= 1.2 K  -> the timestep is REFUTED as the entry point; the lens
#     forms regardless, and the thermal path (the first-cell stratification
#     cutoff, what heats the top metre) or the surface value is next.
#   * between   -> PARTIAL; report the number and the first-metre energy
#     beside it, claim neither.
#   * the run fails (NaN, refusal, blow-up) -> a STABILITY LIMIT of our
#     model at the oracle's step, reported as such; no physics reading.
# Secondary readout, not gated: the 5-65 m viscosity ratio against the
# oracle's days-0-5 mean (equatorial_diffusivity_oracle.py --rec 0), which
# read x0.54 for the control at day 15 and x0.003 at day 30.
#
# WHAT ELSE A 24x LONGER STEP TOUCHES, NAMED SO IT IS NOT MISTAKEN FOR A
# SECOND VARIABLE: the Smagorinsky CFL cap (--smag-cfl-safety 0.125) binds
# harder, the adaptive implicit vertical advection engages more often, and
# the diurnal solar cycle (--dm2dc) is sampled 24 times a day instead of
# 576. Those are what "the oracle's timestep" means for this model; they are
# part of the one variable, not additions to it.
#
# BUILT FROM THE WORKING CARD (_trp_faithz0_d30.sbatch), not from scratch.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_evdarm_wt; cd $ROOT || exit 1
# WHY THIS CARD EXISTS, and why it points at a DIFFERENT tree than the card it
# was copied from.  The physics flag block below is byte-identical to
# orca1_faithful_card.sbatch -- only the job name, the log path, ROOT and OUT
# differ.  The one thing that changes is the CODE the flags run against.
#
# The tripole baseline (trp_orca1faithtint_d30) was integrated from _omip_wt,
# whose orca1_zdftke_config predates #1690 and therefore resolves the TKE
# surface boundary condition to 'interior_pinned'.  The harmonized MPAS run
# runs from _omip_merge, where that same builder sets 'nemo_z0' -- NEMO's own
# placement, restored once #1704 fixed the surface-metric defect that had
# justified the earlier revert.  Neither card passes --tke-surface-bc-level, so
# neither run RECORDS the choice as a flag: it is a lane default, and the two
# lanes sat on opposite sides of it.  The surface TKE boundary condition is a
# known SST lever here, so a cross-grid SST or MLD comparison between those two
# runs is confounded before it starts.
#
# Running the tripole from _omip_merge puts BOTH grids on one code tree, so the
# boundary condition matches and the remaining cross-grid differences are the
# six GM/Redi settings and the grid itself -- which is the comparison the
# harmonization campaign is actually trying to make.
#
# PIN: the tree must be the exact commit the MPAS run integrated, or this is
# not a controlled comparison and the run refuses rather than quietly
# producing a number nobody can pair.
PIN=3b5bdfb2a88c0dadaf0e8277ed422e1d1dd27fbe
HAVE=$(git -C $ROOT rev-parse HEAD)
if [ "$HAVE" != "$PIN" ]; then
  echo "REFUSING: $ROOT is at $HAVE, expected $PIN (the commit the harmonized"
  echo "          MPAS run 9891588 integrated). Re-pin deliberately or rebuild."
  exit 2
fi
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
# TWO LEGS, as the baseline did: ~0.70 h per simulated day means 30 days is
# ~21 h, well past the 12 h partition cap. --years is the ABSOLUTE target, so
# leg 2 asks for the FULL 30 days and resumes from leg 1's restart.
# ONE LEG: five days at the oracle's timestep is 120 steps, minutes of GPU
# after compilation, so there is nothing to split.
DAYS=${DAYS:-5}
LEG=1
TARGET=$DAYS
YEARS=$($PY -c "print(${TARGET}/365.0)")
# THE ONE VARIABLE. The control ran --dt 150; the oracle runs rn_Dt = 3600
# (RUN_GATEWAY/namelist_cfg:58, and the restart's own rdt). Nothing else in
# the flag block below differs from the control card.
DT=${DT:-3600}
# trp_orca1faith_d30 is the pre-mask baseline every 2026-09-13/14 measurement
# references; the masked card writes beside it, never over it.
# trp_orca1faithmask_d30 = mask only (leg 1 to day 15, 9769155); this card now
# also carries NEMO's initial-condition time blend, so it writes beside both.
OUT=${OUT:-/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_faithz0_dt${DT}_d$(printf %02d $DAYS)}
echo "DAYS=$DAYS LEG=$LEG TARGET=$TARGET YEARS=$YEARS DT=$DT OUT=$OUT"
# The run's stdout used to go straight into "tail -40", which buffers the whole
# leg and then discards all but the last forty lines.  Everything the driver
# says about what it RESOLVED -- the A_h profile it built, the viscosity each
# schedule segment applies, the forcing paths -- fires in the first minutes and
# was therefore unreadable, so the half-viscosity defect ran a whole campaign
# with its own log line printing the correct diagnosis into a pipe nobody could
# read.  Keep the full log on disk beside the output; tail still goes to the
# SLURM file so the job summary is unchanged.
mkdir -p $OUT
# NEMO runs no Smagorinsky at all -- its lateral momentum viscosity is read
# entirely from eddy_viscosity_3D.nc (nn_ahm_ijk_t = -30) -- while this card
# adds one and ramps it down mid-run.  The two knobs below let an arm turn
# that term off without touching any other flag; the DEFAULTS ARE THE CARD'S
# OWN VALUES, so an unparameterised submission runs exactly what it always
# ran.  Both columns must move together: --C-smag-lap sets the value the
# schedule's first segment inherits before day 0.
CSMAG=${CSMAG:-3.0}
VSCHED=${VSCHED:-0:1e4:3.0,8:1e4:1.5,16:1e4:0.5}
echo "CSMAG=$CSMAG VSCHED=$VSCHED"

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid tripole \
  --mesh data/grids/eORCA1.2_mesh_mask.nc --dt $DT --min-levels 2 \
  --C-smag-lap $CSMAG --smag-cfl-safety 0.125 \
  --A-h 2e4 --A-h-profile-file $NEMOIN/orca1_inputs/data_repository/input_fields/eddy_viscosity_3D.nc \
  --visc-schedule "$VSCHED" \
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
  --restart-save $OUT/restart_leg$LEG.npz \
  --output $OUT 2>&1 | tee $OUT/run.log | tail -40
echo "=== card exit: $? ==="
echo "OUT=$OUT"
echo DONE
