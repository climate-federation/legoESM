#!/bin/bash
#SBATCH --job-name=ctkh
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ctkh_%j.out
#
# THE LAST CANDIDATE, MEASURED FROM MODEL OUTPUT ON BOTH SIDES.
#
# The cold-tongue chain is closed except for one link. Heat arrives at our
# surface and is not carried down -- between day 5 and day 90 at 220-240E our
# surface warms 2.085 K while our upper fifty metres COOLS 1.807 K, where the
# oracle warms 1.646 and 1.103 -- so our front steepens sixty times faster
# than the oracle's, which over-strengthens the zonal density gradient, which
# lets the baroclinic pressure gradient cancel the barotropic push, which
# starves the undercurrent and warms the tongue.
#
# EVERY OTHER CANDIDATE IS ELIMINATED. Advection, by the oracle's own trend
# ranking: in the top fifty metres it moves heat down by diffusion, beating
# advection 2.4-fold at levels 1-9 and 23-fold at the surface. The
# vertical-diffusion OPERATOR, by the correctly-anchored matched-state stage,
# which reproduces the oracle's own tendency at correlation 0.94 when fed the
# oracle's own diffusivity. Shortwave penetration, by six configuration rows
# matched against the oracle's namelist and its own input files. Lateral
# viscosity, convection, the equation of state, sea level, and the initial
# condition, each earlier and each on its own evidence.
#
# WHAT IS LEFT IS THE VALUE OF THE DIFFUSIVITY ITSELF, and the only estimate
# of it is 0.68 of the oracle's from a probe whose velocity averaging I
# downgraded this session. This card measures the same quantity WITHOUT that
# instrument: our own archived diffusivity against the oracle's own archived
# one, both written by the models rather than reconstructed by a diagnostic.
#
# PRE-REGISTERED, disjoint and exhaustive, on the ratio of our diffusivity to
# the oracle's in the cold-tongue box over the five marks below:
#   * at or below 0.5  -> DEFICIENT. The closure's diffusivity is the cause of
#     the over-sharp front, agreeing with both the matched-state estimate and
#     the free-running column, and the chain is closed end to end.
#   * at or above 0.7  -> NOT DEFICIENT. The last candidate falls, the column
#     signature has no surviving explanation, and that is a finding rather
#     than a failure -- it would mean the missing transfer is not the
#     diffusivity and the search reopens.
#   * strictly between -> INCONCLUSIVE. Report the number, claim no
#     direction, and do not round toward the convenient branch.
#
# THE SAMPLING CAVEAT, NAMED BEFORE THE READING AND WITH ITS DIRECTION. Our
# diffusivity is an instantaneous snapshot at a fixed clock hour; the oracle's
# is a five-day mean. Averaging a quantity that switches on at night and off
# by day LOWERS the mean, so the oracle's number is if anything biased DOWN
# relative to a fair comparison -- which makes a "deficient" reading harder to
# obtain, not easier, and a null reading weaker. Five marks are taken so the
# statistic is a mean over independent samples rather than one draw.
#
# REUSED, NOT REBUILT: equatorial_diffusivity_oracle.py already reduces both
# sides in the same box over the same depth window with the same per-column
# median, which is the whole point -- the only thing differing between its two
# tables is the model.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
echo "HEAD: $(git -C $M log --oneline -1)"
echo "dirty: $(git -C $M status --porcelain | wc -l)"
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_faithz0_d90
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
[ -r "$GW" ] || { echo "FATAL: no oracle grid_W"; exit 1; }

# The campaign's cold-tongue box, the same 220-240E used for every other
# measurement in this chain, so this number is comparable to them. Depth
# window to 100 m because the front and the warm lens both live above that;
# the probe prints per-depth rows inside it.
rc=0
for pair in "50:9" "60:11" "70:13" "80:15" "90:17"; do
  D=${pair%%:*}; REC=${pair#*:}
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
  echo "######## day $D (oracle rec $REC) ########"
  $PY -u scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py \
    --nemo-gridw "$GW" --rec "$REC" \
    --lat-halfwidth 2.0 --lon-west 220.0 --lon-east 240.0 --depth-max 100.0 \
    --legoesm-snapshot "$SNAP" 2>&1 | tail -40
  r=${PIPESTATUS[0]}; [ "$r" -eq 0 ] || { echo "### day $D rc=$r"; rc=$r; }
done
echo "=== card exit: $rc ==="
exit $rc
