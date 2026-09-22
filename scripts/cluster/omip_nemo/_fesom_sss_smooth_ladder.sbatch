#!/bin/bash
#SBATCH --job-name=ssssmooth
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ssssmooth_%j.out
#
# AT WHAT SPATIAL SCALE DOES THE FESOM SALINITY GAP LIVE?
#
# FESOM's day-30 surface salinity scores rmse 0.465 against the NEMO oracle
# where the tripole scores 0.258, and the two arms' mean offsets agree to
# 7e-4, so the gap is SCATTER. Two passes have localised it without naming it:
# binning by FESOM's own node spacing put 48% of the excess variance in the
# DENSEST bin and 39% in the SPARSEST, with 14% across the four middle bins
# that cover 83% of the area.
#
# WHY THE OBVIOUS NEXT TEST WAS ABANDONED. The plan was to stratify the dense
# bin by distance to the edge of the scored domain, on the theory that the
# tripole's home-geometry advantage -- it runs on the oracle's OWN grid and
# land mask, FESOM does not -- shows up as FESOM error wherever the two
# disagree about where the ocean ends. GLM refuted the DESIGN before it ran:
# the scorer pairs sources by inverse-distance over their four nearest
# neighbours, and coastal salinity gradients are the sharpest in the field, so
# "the interpolation stencil aliases a sharp gradient" predicts EXACTLY the
# same edge concentration as "the masks disagree". A positive reading would
# have been unattributable. That table still runs, but as description only.
#
# WHAT THIS CARD DOES INSTEAD, and it needed no new code. The scorer already
# ships --smooth-radius-deg, whose whole purpose is this question: after
# regridding it replaces every field by its top-hat mean over a fixed
# great-circle radius, applied IDENTICALLY to all three sources (verified at
# the call site, not read off the help text -- compare_three_way_nemo.py
# applies it to the tripole, FESOM and NEMO fields in one comprehension). The
# interpolation stencil's four neighbours span ~76 km on FESOM's 38 km mesh
# and ~220 km on ORCA1's ~111 km cells, so the two arms do NOT resolve the
# same scales even after landing on a common target. Smoothing all three to
# one PHYSICAL radius removes that asymmetry.
#
# A LADDER, NOT A CHOSEN RADIUS. Picking one radius would be choosing the
# answer; the reading here is the TREND across baseline / 1.0 / 1.5 / 2.0 /
# 3.0 degrees, and every rung is reported.
#
# PRE-REGISTERED READING, written before the numbers exist:
#   * the GAP (excess rms, sqrt(rmse_FESOM^2 - rmse_tripole^2)) collapses
#     toward zero as the radius grows -> the disagreement lives BELOW the
#     common smoothing scale. That is a representativeness/stencil result, not
#     a resolved model difference, and the FESOM salinity "miss" is then an
#     artifact of scoring a fine mesh against a coarse oracle.
#   * the gap SURVIVES at 2-3 degrees -> the arms disagree at scales both
#     remaps resolve. The stencil is then exonerated as the leading term and a
#     model or mask difference stands.
#   * both arms' rmse fall together with the RATIO roughly held -> smoothing
#     is removing noise from both and this test is uninformative. Say so; do
#     not read the shrinking absolute numbers as progress.
# SST is carried alongside as the reference field: its gap is small already
# (0.3665 vs 0.4325), so if SST's gap collapses in the same way, the collapse
# is generic to smoothing and says nothing about salinity.
#
# NON-VACUITY GATE. A control that changes nothing has not been run. At the
# widest radius BOTH arms' salinity rmse must move by more than 5%; if neither
# does, the smoothing is not reaching the field and every rung is meaningless.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
G=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
STEM=$G/ORCA1_5d_20000101_20000331
WORK=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/_smoothladder_work

TRP=$R/trp_pinned_d30/snapshot_final.npz
FES=$R/fesom_orca1faithtint_d30/snapshot_day0030.npz
MPS=$R/mpas_orca1faithtint_d30/snapshot_day0030.npz
for f in "$TRP" "$FES" "$MPS" ${STEM}_grid_T.nc; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
mkdir -p "$WORK" || exit 1

# Identical inputs, labels and options to the committed day-30 scorecard, so
# the baseline rung reproduces its published numbers and the ladder is a
# one-variable sweep off it.
run_rung () {   # $1 = tag, $2.. = extra flags
  local tag="$1"; shift
  echo "### rung $tag"
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole "$TRP" --mpas "$FES" --also-mask "$MPS" \
    --nemo-gridt ${STEM}_grid_T.nc --nemo-time-idx 5 \
    --label-tripole tripole --label-mpas FESOM2 \
    --res-deg 1.0 --coast-cells 2 --mask-mode nearest \
    --out-dir "$WORK/$tag" "$@" > "$WORK/$tag.log" 2>&1
  local rc=$?
  echo "### rung $tag rc=$rc"
  [ "$rc" -eq 0 ] || tail -25 "$WORK/$tag.log"
  return $rc
}

run_rung baseline               || exit 1
run_rung r1p0 --smooth-radius-deg 1.0 || exit 1
run_rung r1p5 --smooth-radius-deg 1.5 || exit 1
run_rung r2p0 --smooth-radius-deg 2.0 || exit 1
run_rung r3p0 --smooth-radius-deg 3.0 || exit 1

$PY -u - "$WORK" <<'EOF'
import json
import sys
from pathlib import Path

import numpy as np

work = Path(sys.argv[1])
rungs = [("baseline", None), ("r1p0", 1.0), ("r1p5", 1.5),
         ("r2p0", 2.0), ("r3p0", 3.0)]

rows = []
for tag, rad in rungs:
    p = work / tag / "report.json"
    if not p.is_file():
        raise SystemExit(f"FATAL: {p} missing; the rung did not complete")
    r = json.loads(p.read_text())
    got = r.get("smooth_radius_deg")
    # The report stamps the radius it actually applied.  Trusting the flag we
    # passed instead would let a silently ignored option read as a null result
    # -- the exact shape of a control that cannot fail.
    if (rad is None) != (got is None) or (rad is not None
                                          and abs(got - rad) > 1e-9):
        raise SystemExit(
            f"FATAL: rung {tag} was asked for radius {rad} and the report "
            f"stamps {got}; the control did not take.")
    ent = {"tag": tag, "rad": rad}
    for fld in ("SST", "SSS"):
        f = r["fields"][fld]
        ent[f"{fld}_a"] = f["tripole_vs_nemo"]["global"]["rmse"]
        ent[f"{fld}_b"] = f["FESOM2_vs_nemo"]["global"]["rmse"]
    rows.append(ent)

def excess(b, a):
    """Excess rms: the part of B's variance that A's does not account for."""
    d = b ** 2 - a ** 2
    return float(np.sqrt(d)) if d > 0 else -float(np.sqrt(-d))

print(f"\n{'rung':>9} {'radius':>7} | {'SSS trip':>9} {'SSS FES':>9} "
      f"{'excess':>8} {'ratio':>6} | {'SST trip':>9} {'SST FES':>9} "
      f"{'excess':>8} {'ratio':>6}")
for e in rows:
    rad = "none" if e["rad"] is None else f"{e['rad']:.1f}"
    print(f"{e['tag']:>9} {rad:>7} | {e['SSS_a']:9.4f} {e['SSS_b']:9.4f} "
          f"{excess(e['SSS_b'], e['SSS_a']):8.4f} "
          f"{e['SSS_b'] / e['SSS_a']:6.2f} | "
          f"{e['SST_a']:9.4f} {e['SST_b']:9.4f} "
          f"{excess(e['SST_b'], e['SST_a']):8.4f} "
          f"{e['SST_b'] / e['SST_a']:6.2f}")

base, wide = rows[0], rows[-1]
mv_a = abs(wide["SSS_a"] - base["SSS_a"]) / base["SSS_a"]
mv_b = abs(wide["SSS_b"] - base["SSS_b"]) / base["SSS_b"]
print(f"\nwidest rung moved salinity rmse by {mv_a * 100:.1f}% (tripole) and "
      f"{mv_b * 100:.1f}% (FESOM2)")
if mv_a < 0.05 and mv_b < 0.05:
    raise SystemExit(
        "VACUOUS: the widest smoothing radius moved neither arm by 5%; the "
        "control is not reaching the field and no rung above is readable.")

e0 = excess(base["SSS_b"], base["SSS_a"])
e1 = excess(wide["SSS_b"], wide["SSS_a"])
s0 = excess(base["SST_b"], base["SST_a"])
s1 = excess(wide["SST_b"], wide["SST_a"])
print(f"salinity excess rms {e0:.4f} -> {e1:.4f} "
      f"({(1 - e1 / e0) * 100:.0f}% removed)")
print(f"     SST excess rms {s0:.4f} -> {s1:.4f} "
      f"({(1 - s1 / s0) * 100:.0f}% removed)")
print("\nRead against the PRE-REGISTERED trichotomy in the card header. The "
      "salinity number is only interesting insofar as it differs from the SST "
      "one: a collapse both fields share is a fact about smoothing, not about "
      "salinity.")
EOF
rc2=$?
echo "=== card exit: $rc2 ==="
exit $rc2
