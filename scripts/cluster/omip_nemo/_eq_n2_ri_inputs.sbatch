#!/bin/bash
#SBATCH --job-name=eqn2ri
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eqn2ri_%j.out
#
# WHICH OF THE RICHARDSON NUMBER'S INPUTS DRIVES THE PRANDTL EXCESS?
#
# The cold-tongue chain is closed and its last link is a PARTITION defect: in
# the entrainment band 65-105 m our heat diffusivity is a quarter of the
# oracle's while our momentum viscosity is HALF AGAIN LARGER, so our Prandtl
# number is 5.75 times the oracle's. That is not a starvation of mixing; it is
# heat and momentum being separated by the closure.
#
# THE MAPPING ITSELF IS FAITHFUL, which is what makes this measurable. NEMO
# forms Pr = min(10, max(1, zri/ri_cri)) with ri_cri = 2/(2+rn_ediss/rn_ediff)
# = 0.2222, a slope of 4.5, and our prandtl_ri_coeff IS 4.5. A faithful
# mapping fed a wrong Richardson number gives a wrong Prandtl, so the fault
# must lie in Ri's inputs -- the stratification, the shear, or the viscosity
# that enters the shear production.
#
# ONE INPUT IS ALREADY MEASURED AND IT REFUTES MY OWN EXPECTATION. I expected
# our half-strength undercurrent to leave the entrainment band with about half
# the oracle's shear, which alone would have quadrupled Ri. It does not. From
# the twelve archived force-balance dumps of this same run, the root-mean-
# square vertical shear at 220E over 65-105 m is 0.896 of the oracle's, spread
# 0.176, so shear-squared is 0.803 and it can inflate Ri by only about a
# quarter. That expectation is RETRACTED. By elimination the stratification
# would have to carry almost all of the excess, and this card measures whether
# it does.
#
# PRE-REGISTERED, on the ratio of our N2 to the oracle's in the entrainment
# band 65-105 m, averaged over the five marks, with disjoint thresholds:
#   * at or above 2.0 -> STRATIFICATION IS THE DRIVER. The over-sharp front
#     saturates the Prandtl mapping directly, and the self-reinforcing loop
#     the chain predicts is confirmed rather than merely argued.
#   * at or below 1.2 -> STRATIFICATION IS NOT THE DRIVER. With shear already
#     measured at 0.896 the two inputs together cannot produce a 5.75-fold
#     Prandtl excess, so the fault is in how the mapping is applied or in the
#     viscosity that enters the shear production, and the self-check rows
#     below become the finding rather than a control.
#   * strictly between -> PARTIAL. Report the number, combine it with the
#     measured shear-squared 0.803 into a predicted Ri ratio, compare that
#     against the observed Prandtl ratio, and name no single driver.
#
# THE SELF-CHECK IS THE REASON FOR --prandtl-check, AND IT RUNS WHATEVER THE
# ABOVE SAYS. It scores each model's own archived viscosity-to-diffusivity
# ratio against the mapping, on that model's own state, so it asks three
# questions that need no cross-model comparison at all: whether our archived
# ratio equals what our own formula predicts from our own stratification and
# shear, whether the same formula reproduces NEMO's ratio from NEMO's state,
# and therefore whether a disagreement lives in the formula or in its inputs.
# An implementation defect fails the first and passes the second.
#
# WHY THIS IS NOT THE ARCHIVED d30 REPORT, which already exists and appears to
# say the opposite. results/omip_nemo/eq_n2_tke_d30.json reports our N2 at
# 0.573 of the oracle's in this band and our Ri BELOW the oracle's, which if
# spliced onto the diffusivity result would contradict it outright. It differs
# from that result in THREE ways at once -- a different run
# (trp_orca1faithtint_d30), a different day (30, not the 50-90 equilibrium
# window), and a different box (the probe's 200-260E default rather than the
# campaign's 220-240E cold tongue) -- so it is not a matched comparison and is
# NOT quoted against the d90 measurement. This card removes all three
# differences instead of arguing about them.
#
# REUSED, NOT REBUILT: equatorial_n2_vs_nemo.py already reduces both models in
# the same box over the same three depth bands, and those bands are the same
# 15-65 and 65-105 m the diffusivity card reported, so the two results line up
# row for row without any reinterpretation.
#
# THE SAMPLING CAVEAT, NAMED BEFORE THE READING AND WITH ITS DIRECTION. Ours
# is an instantaneous snapshot at a fixed clock hour, the oracle's a five-day
# mean. The surface band was shown to be uninterpretable for exactly this
# reason -- its diffusivity ratios ran 0.00 to 1.97 across the five marks --
# and N2 is smoothed by averaging, so a five-day mean is biased toward WEAKER
# stratification. That makes our ratio read HIGH, which favours the
# stratification branch, so a reading at or below 1.2 is the safer of the two
# and a reading above 2.0 must clear that bias to be believed. Five marks are
# taken so the statistic is a mean over independent samples.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
echo "HEAD: $(git -C $M log --oneline -1)"
echo "dirty: $(git -C $M status --porcelain | wc -l)"
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

MAIN=/burg-archive/glab/users/pg2328/legoESM
R=$MAIN/results/omip_nemo/trp_faithz0_d90
G=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
STEM=$G/ORCA1_5d_20000101_20000331
MESH=$MAIN/data/grids/eORCA1.2_mesh_mask.nc
[ -r "$MESH" ] || MESH=/burg-archive/glab/users/pg2328/_omip_merge/data/grids/eORCA1.2_mesh_mask.nc
for f in ${STEM}_grid_T.nc ${STEM}_grid_W.nc ${STEM}_grid_U.nc ${STEM}_grid_V.nc "$MESH"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

OUT=$MAIN/results/omip_nemo/eq_n2_ri_d90
mkdir -p "$OUT"

# The campaign's cold-tongue box, the same 220-240E every other measurement in
# this chain used, passed EXPLICITLY rather than left to the probe's wider
# default -- which is one of the three reasons the archived d30 report is not
# comparable to this one.
rc=0
for D in 50 60 70 80 90; do
  SNAP=$R/snapshot_day$(printf '%04d' $D).npz
  [ -r "$SNAP" ] || SNAP=$R/snapshot_final.npz
  [ -r "$SNAP" ] || { echo "FATAL: no snapshot for day $D"; rc=1; continue; }
  $PY - "$SNAP" "$D" <<'STAMP' || { echo "FATAL: day $D snapshot is not at day $D"; exit 1; }
import sys
import numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
want = float(sys.argv[2])
if "time_days" not in d.files:
    print(f"  day {want:g}: NO STAMP (unverified)")
else:
    got = float(np.asarray(d["time_days"]).ravel()[0])
    print(f"  day {want:g}: stamp {got:g}")
    if abs(got - want) > 1e-6:
        sys.exit(1)
STAMP
  echo "######## day $D ########"
  $PY -u scripts/validate/ocean_fidelity/equatorial_n2_vs_nemo.py \
    --snapshot "$SNAP" \
    --nemo-gridt ${STEM}_grid_T.nc --nemo-gridw ${STEM}_grid_W.nc \
    --nemo-gridu ${STEM}_grid_U.nc --nemo-gridv ${STEM}_grid_V.nc \
    --nemo-day "$D" --mesh-mask "$MESH" \
    --lat-halfwidth 2.0 --lon-west 220.0 --lon-east 240.0 \
    --prandtl-check --tke-budget \
    --out-json "$OUT/n2_ri_d$D.json" 2>&1 | tail -60
  r=${PIPESTATUS[0]}; [ "$r" -eq 0 ] || { echo "### day $D rc=$r"; rc=$r; }
done
echo "=== card exit: $rc ==="
exit $rc
