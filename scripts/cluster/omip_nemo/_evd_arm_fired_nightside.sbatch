#!/bin/bash
#SBATCH --job-name=evdfired
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/evdfired_%j.out
#
# DID THE ENHANCED-DIFFUSION ARM EVER FIRE?
#
# The NEMO-exact enhanced-diffusion arm (trp_faithevd_d30: rn_evd = 100,
# tracers only, NEMO's own bn2 trigger at -1e-12) has ALREADY RUN to day 30,
# and its cold-tongue lens is 3.79 K against 3.96 K without it -- a four
# percent move where the oracle's lens is 0.76 K. Taken at face value that
# refutes enhanced diffusion as the cause of the lens. It cannot be taken at
# face value, because the arm's own pre-registration (2026-09-20) said so in
# advance: REFUTE only if the lens persists WHILE the trigger fires at night
# at NEMO's rate; NO-OP, do not score, if it never fires. The arm was launched
# without the runtime firing-fraction logger its registration required, so
# nothing on disk says which of those two it is. A term that never fires and a
# term that fires and does not matter need OPPOSITE next steps.
#
# THE SNAPSHOT IS GLOBAL, AND SOMEWHERE IT IS ALWAYS NIGHT. Every snapshot is
# written at 00:00 UTC. In the Pacific cold tongue (220-240E) that is
# mid-afternoon, peak stratification, the one hour a nightly convective
# trigger is guaranteed NOT to fire -- which is why a Pacific reading of zero
# proves nothing. In the equatorial Indian Ocean (40-100E) the same instant is
# 02:40 to 06:40 local, the pre-dawn hours when nightly cooling is deepest and
# NEMO's own occupancy peaks. So the question "does our trigger fire at
# night" can be read from the arm's EXISTING snapshots at no GPU cost, by
# reading the night sector instead of relaunching the run.
#
# PRE-REGISTERED, on the fraction of top-interface cells whose heat
# diffusivity exceeds 1 m2/s in the Indian Ocean box, |lat| <= 2, 40-100E,
# mean over the three marks:
#   * >= 0.20 -> THE TRIGGER FIRES at night in our columns. The day-30 Pacific
#     null is then a genuine REFUTATION of enhanced diffusion as the lens
#     cause, with the one caveat that the Pacific night rate itself remains
#     unmeasured. The missing heat mixing is elsewhere.
#   * <= 0.02 -> THE TRIGGER DOES NOT FIRE. The arm was a NO-OP by its own
#     registration and its lens number is not a result. The question becomes
#     why our columns never go statically unstable at night when the oracle's
#     do -- surface cooling, the diurnal solar cycle, or the lens itself being
#     too stable to overturn.
#   * strictly between -> PARTIAL: report the number, claim neither.
# Both outer branches are reachable: our surface heat loss is applied, the
# diurnal cycle is on, and nothing forbids an inversion; equally nothing
# guarantees one.
#
# TWO CONTROLS IN THE SAME SNAPSHOT. The Pacific box (220-240E, afternoon)
# must read near zero in the arm -- if it does not, the trigger fires in
# daylight too and the process is not the nightly one assumed, which is its
# own finding. The Atlantic box (330-360E, 22:00 to midnight local) brackets
# the night from the other side. The unconvecting control run is NOT read: a
# scheme set to 'none' gives exactly zero by construction and proves nothing.
#
# THE ORACLE COLUMN IS A FIVE-DAY MEAN and is printed for location only: where
# NEMO's mean diffusivity exceeds 1 m2/s the sentinel fired at some hour in
# those five days. It is not a night-time rate and is not compared to ours.
#
# REUSED, NOT REBUILT: equatorial_diffusivity_oracle.py already prints the
# per-depth convecting fraction ("conv-frac") for any longitude box on both
# sides from the models' own archived coefficients.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
echo "HEAD: $(git -C $M log --oneline -1)"
echo "dirty: $(git -C $M status --porcelain | wc -l)"
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_faithevd_d30
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_W.nc
[ -r "$GW" ] || { echo "FATAL: no oracle grid_W"; exit 1; }

# The arm carries the convection scheme it is being asked about; refuse to
# measure anything else.
$PY - <<'CHK' || { echo "FATAL: arm manifest does not carry enhanced_diffusion"; exit 1; }
import json
m = json.load(open("/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_faithevd_d30/run_manifest.json"))
c = m["config"]["resolved_config"]["runtime_config"]["physics"]["convection"]
ed = c["enhanced_diffusion"]
print(f"  arm: scheme={c['scheme']} K_conv={ed['K_conv']} K_bg={ed['K_bg']} "
      f"n2_mode={ed['n2_mode']} thr={ed['n2_threshold']} smooth={ed['smooth_transition']}")
assert c["scheme"] == "enhanced_diffusion" and ed["K_conv"] == 100.0
CHK

rc=0
for pair in "20:3" "25:4" "30:5"; do
  D=${pair%%:*}; REC=${pair#*:}
  SNAP=$R/snapshot_day$(printf '%04d' $D).npz
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
  for box in "indian_night:40:100" "atlantic_evening:330:360" "pacific_afternoon:220:240"; do
    NAME=${box%%:*}; rest=${box#*:}; W=${rest%%:*}; E=${rest#*:}
    echo "######## day $D  $NAME ${W}-${E}E  (oracle rec $REC, 5-day mean, location only) ########"
    $PY -u scripts/validate/ocean_fidelity/equatorial_diffusivity_oracle.py \
      --nemo-gridw "$GW" --rec "$REC" \
      --lat-halfwidth 2.0 --lon-west "$W" --lon-east "$E" --depth-max 30.0 \
      --legoesm-snapshot "$SNAP" 2>&1 | grep -E "^\s+[0-9.]+\s|closure|zone|conv-frac|vs NEMO|surface|FATAL|Error|Traceback" | head -30
    r=${PIPESTATUS[0]}; [ "$r" -eq 0 ] || { echo "### day $D $NAME rc=$r"; rc=$r; }
  done
done
echo "=== card exit: $rc ==="
exit $rc
