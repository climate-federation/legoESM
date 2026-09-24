#!/bin/bash
#SBATCH --job-name=eqavm
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eqavm_%j.out
#
# HAS OUR VERTICAL MOMENTUM VISCOSITY EVER BEEN SCORED AGAINST NEMO'S? NO.
#
# The tracer side is settled: at matched state, on NEMO's own hourly column,
# our diffusivity is 7.69e-03 against NEMO's 7.25e-03 in the tropics --
# correlation 0.80 over 482026 interfaces -- and five closure settings move
# that not at all. What was never scored is the MOMENTUM side. Every saved
# artifact from that comparator contains the tracer rows and nothing else:
# the six tendency_match_rec1.json files carry no avm, no Prandtl and no
# momentum entry, and the card that produced them piped its output through
# grep -E "lon-boxed|nino|eqpac|cfg/stage_a]", which reduced a full Stage-A
# report to 76 lines. The probe COMPUTES the momentum and Prandtl split -- it
# has a function for exactly that, including depth cuts -- and nobody has ever
# read it. This card changes nothing but the grep.
#
# WHY MOMENTUM, AND WHY NOW. The undercurrent measurement already on disk
# (job 9921094, day 25, the normal window rather than the wind-relaxation one)
# says our equatorial circulation is wrong in a specific, structured way:
#
#   at 220E   core speed  ours 0.357 m/s   NEMO 0.725 m/s   (49%)
#             core depth  ours 120 m       NEMO 97 m
#             flow at 10m ours -0.650 m/s  NEMO -0.234 m/s  (2.8x)
#
# and at 240E our column has no coherent undercurrent at all -- our maximum
# sits at 371 m where NEMO's is at 54 m. So the trades' momentum is staying
# at the surface as an over-strong westward flow instead of appearing at
# depth as an undercurrent. That is a vertical momentum transfer signature,
# and vertical momentum transfer is the one field in this closure nobody has
# compared against the oracle.
#
# IT IS ALSO THE ONLY SURVIVING CANDIDATE OF ITS CLASS. Lateral viscosity was
# refuted by experiment -- cutting the Smagorinsky closure 6.8x moved the
# cold tongue 0.09 C and the undercurrent 5%. Upwelling is measured and
# uninformative: across the six matched windows on disk the ascent ratio runs
# 6.04, 0.66, 0.71, 2.06, 1.00, 1.54, which is scatter around one, exactly as
# predicted for a quantity whose forcing is locked to the oracle's by
# construction. Convection is closed by a refuted arm. The tracer closure is
# exonerated above.
#
# PRE-REGISTERED, on our K_M against NEMO's avm in the equatorial box above
# about 150 m, at matched state:
#   * ours LARGER than NEMO's -> too much momentum diffusion spreads the
#     undercurrent downward and drags the surface along with it. This is the
#     branch the undercurrent measurement already predicts, because it is the
#     only one that produces a DEEPER, WEAKER core and a STRONGER surface
#     westward flow at the same time.
#   * ours SMALLER -> the trades' momentum never reaches the undercurrent.
#     That also gives a weak core, but it gives a WEAKER surface flow, not the
#     2.8x excess measured. This branch would therefore mean the undercurrent
#     signature has a different cause and the search moves to the pressure
#     gradient and the advection of momentum.
#   * ours MATCHES (say within 30%, the same tolerance the tracer side was
#     judged at) -> vertical viscosity is exonerated too, and with lateral
#     viscosity, upwelling, convection and tracer mixing all eliminated the
#     remaining terms are the zonal pressure gradient and momentum advection.
# All three are reachable and none is favoured by the instrument.
#
# CONFIGURED LIKE THE RUN IT EXPLAINS: --mxl-choice 3, the production setting.
# ONE ARM ONLY. Adding suspects here would turn a measurement into a sweep
# before the measurement has said anything.
#
# THE ONE LIMITATION, STATED RATHER THAN BURIED: this is a single hourly
# record, so its diurnal phase is unknown, and it is the same record the
# tracer comparison used. It is a strong statement about one hour of NEMO's
# equatorial ocean and not about the diurnal cycle.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
echo "HEAD: $(git -C $M log --oneline -1)"
echo "dirty: $(git -C $M status --porcelain | wc -l)"
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

D=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
for f in T U V; do
  [ -r "$D/ORCA1_1h_20010101_20010101_trd1h_$f.nc" ] || {
    echo "FATAL: missing trd1h_$f"; exit 1; }
done

# Extra arguments are passed straight through, so the same card serves as its
# own one-variable arm: with no arguments it is the production configuration,
# and `sbatch <card> --prandtl-mode nemo_ri` differs from it in that flag and
# nothing else. TAG names the output directory so an arm cannot overwrite the
# control it is compared against.
TAG="${TAG:-prod_mxl3}"
OUT=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/eq_avm_$TAG
mkdir -p "$OUT"
echo "ARGS: --mxl-choice 3 $*"
$PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
  --tfile $D/ORCA1_1h_20010101_20010101_trd1h_T.nc \
  --ufile $D/ORCA1_1h_20010101_20010101_trd1h_U.nc \
  --vfile $D/ORCA1_1h_20010101_20010101_trd1h_V.nc \
  --rec 1 --skip-maps --mxl-choice 3 "$@" \
  --output-dir "$OUT"
rc=$?
echo "=== card exit: $rc ==="
exit $rc
