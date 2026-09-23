#!/bin/bash
#SBATCH --job-name=fwchscore
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fwchscore_%j.out
#
# THE VERDICT CARD FOR THE FRESHWATER-TREATMENT ARM (job 9939797).
#
# Scores the arm against the SAME oracle record and the SAME edge rings that
# produced the finding it is testing, so the two numbers are comparable by
# construction rather than by assertion. Written and committed BEFORE the arm
# reaches day 30 so the threshold cannot be chosen after seeing the result.
#
# WHAT THE ARM IS. `trp_fwchan_d30` is `trp_pinned_d30`'s own command line plus
# `--sss-restore-channel water_flux --sss-restore-normalization live_s
# --freshwater-closure real_freshwater` -- FESOM's salinity restoring
# treatment, which the driver will not let the tripole take one flag at a time.
#
# WHY IT WAS RUN. FESOM's day-30 surface salinity scores 0.465 against the
# oracle where the tripole scores 0.258, 86% of that gap sits in the single
# ring of cells touching the edge of the scored ocean, and four measurements
# removed the scoring explanation: smoothing to a common 330 km left the ratio
# at 1.80-1.85 while SST's fell 1.18 -> 1.07; the oracle's own field carried
# through FESOM's mesh accounts for 6%; narrowing the stencil to one neighbour
# removed NONE of FESOM's coastal error while costing the tripole 9.6%; and the
# excess survives where the nearest node sits within 9 km of the target-cell
# centre, 1.5383 against 0.9358. A coastal MODEL difference carries the weight,
# and the restoring treatment is the one lane difference that acts on salinity
# and diverges hardest at coasts.
#
# PRE-REGISTERED VERDICT, on RING-1 salinity rmse. The global figure is printed
# but is NOT the primary: this error lives in one ring, and a global number
# averages it against 24315 interior cells that carry 0.9% of the gap.
#   arm ring-1 <= 0.55  -> the freshwater treatment is a LEADING TERM in the
#                          coastal salinity error
#   arm ring-1 >= 0.75  -> REFUTED; it does nothing, or makes the tripole
#                          worse, and the FESOM excess needs another lane
#                          difference
#   0.55 to 0.75        -> NO VERDICT
# The control sits at 0.6605 and FESOM at 1.3917, so the window brackets the
# control by roughly +/- 0.1 -- wide enough that run-to-run spread cannot
# manufacture a verdict on its own. See the null thread (4cdbc7899): two runs
# from the SAME restart with configs agreeing on 626 of 626 leaves produced
# different oceans, so a small move is not a result.
#
# AND IF IT MOVES, SAY WHAT CANNOT BE CONCLUDED. The driver refuses the channel
# without the closure, so a move is attributable to the PAIR and not to either
# member. Separating them needs a second arm with the closure changed alone.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
G=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
STEM=$G/ORCA1_5d_20000101_20000331
WORK=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/_fwchan_score
DUMP=$WORK/cells_arm_d30.npz

CTL=$R/trp_pinned_d30/snapshot_final.npz
ARM=$R/trp_fwchan_d30/snapshot_day0030.npz
# A COMPLETED run writes its target day as snapshot_final.npz, not
# snapshot_dayNNNN.npz, and both the scorer and the figure card refuse the
# wrong name. Accept either rather than failing on a naming convention.
[ -r "$ARM" ] || ARM=$R/trp_fwchan_d30/snapshot_final.npz
for f in "$CTL" "$ARM" ${STEM}_grid_T.nc; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
mkdir -p "$WORK" || exit 1

# --expect-day 30 makes a snapshot from the wrong day a HARD ERROR rather than
# a number in a log nobody reads, and --nemo-time-idx 5 pins the oracle record
# that ENDS on day 30 (records are 5-day means, so day N is N/5 - 1). Omitting
# it once defaulted to the LAST record and scored day 30 against NEMO's day 90.
$PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
  --tripole "$CTL" --mpas "$ARM" \
  --nemo-gridt ${STEM}_grid_T.nc --nemo-time-idx 5 --expect-day 30 \
  --label-tripole control --label-mpas fwchan \
  --res-deg 1.0 --coast-cells 2 --mask-mode nearest \
  --out-dir "$WORK/score" --dump-cells "$DUMP" 2>&1 | tail -60
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "FATAL: scorer failed rc=$rc"; exit $rc; }

$PY -u - "$DUMP" <<'EOF'
import sys
import numpy as np

d = np.load(sys.argv[1], allow_pickle=True)
for key in ("scored", "area", "edge_ring", "SSS_a", "SSS_b", "SSS_nemo"):
    if key not in d.files:
        raise SystemExit(f"FATAL: dump lacks {key!r}: {sorted(d.files)}")
scored = np.asarray(d["scored"], dtype=bool)
area = np.asarray(d["area"], dtype=np.float64)
ring = np.asarray(d["edge_ring"], dtype=int)
e_ctl = np.asarray(d["SSS_a"], dtype=np.float64) - np.asarray(d["SSS_nemo"])
e_arm = np.asarray(d["SSS_b"], dtype=np.float64) - np.asarray(d["SSS_nemo"])
use = scored & np.isfinite(e_ctl) & np.isfinite(e_arm)
if int(use.sum()) < 1000:
    raise SystemExit(f"VACUOUS: only {int(use.sum())} usable cells")

def wrms(e, m):
    w = area[m]
    return float(np.sqrt((w * e[m] ** 2).sum() / w.sum())) if w.sum() > 0 else float("nan")

print(f"\n{'edge ring':>12} {'cells':>7} {'control':>9} {'fwchan':>9} "
      f"{'arm-ctl':>9}")
rows = [(k, str(k)) for k in range(1, int(ring.max()) + 1)]
rows += [(0, "interior"), (None, "ALL")]
ring1 = None
for val, lab in rows:
    m = use if val is None else (use & (ring == val))
    n = int(m.sum())
    if n == 0:
        continue
    rc_, ra_ = wrms(e_ctl, m), wrms(e_arm, m)
    if val == 1:
        ring1 = ra_
    print(f"{lab:>12} {n:7d} {rc_:9.4f} {ra_:9.4f} {ra_ - rc_:+9.4f}")

if ring1 is None:
    raise SystemExit("VACUOUS: ring 1 is empty; the edge rings did not resolve")
# The thresholds are the card header's, restated here only so the arithmetic is
# done by the machine. The card does NOT print a verdict sentence: a probe that
# writes its own conclusion gets that conclusion quoted back as evidence, which
# has already happened once on this campaign.
print(f"\nring-1 salinity rmse: control 0.6605 (published), arm {ring1:.4f}")
print(f"pre-registered window: <= 0.55 LEADING TERM, >= 0.75 REFUTED, "
      f"else NO VERDICT")
print("If the arm moved, it moved on the CHANNEL AND THE CLOSURE TOGETHER -- "
      "the driver refuses one without the other, so neither member is "
      "separately implicated by this run.")
EOF
rc2=$?
echo "=== card exit: $rc2 ==="
exit $rc2
