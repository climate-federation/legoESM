#!/bin/bash
#SBATCH --job-name=nodedist
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nodedist_%j.out
#
# IS THE FESOM SURFACE-SALINITY EXCESS A REMAP ARTIFACT OR A MODEL DIFFERENCE?
#
# FESOM's day-30 surface salinity scores rmse 0.465 against NEMO where the
# tripole scores 0.258, and the excess is northern. The open objection is that
# FESOM enters the scorer as an unstructured NODE CLOUD -- median spacing
# 37.8 km, 99th percentile nearest-node distance 64 km, maximum 270 km --
# while NEMO and the tripole share both source geometry and remap path. Nearest
# -centre pairing of a cloud against a structured target inflates error exactly
# where the cloud is sparse, and that inflation is indistinguishable from model
# error in the scorecard.
#
# A PREVIOUS ATTEMPT TO SETTLE THIS BY SPLITTING THE ARCTIC INTO COASTAL AND
# INTERIOR WAS RETRACTED (3558d23d2) FOR TWO REASONS, BOTH OF WHICH THIS CARD
# IS BUILT TO AVOID. First, it recombined area-weighted sums of squares using
# CELL COUNTS; everything below carries the scorer's own cos-latitude weights
# through to the last line. Second, and more fundamental, "interior" is not a
# control for a node-cloud remap at all: the sparsest hole in the mesh is four
# times the 99th percentile and plausibly sits in the Canada Basin, which IS
# interior. The only honest stratifier is the NODE DISTANCE ITSELF.
#
# THE CONTROL, WHICH IS WHAT MAKES THIS NON-VACUOUS. Node distance correlates
# with geography: the sparse parts of the mesh are polar and coastal, and those
# are also where the ocean is hardest to get right. So "FESOM's error rises
# with node distance" ON ITS OWN proves nothing -- any arm's error would.
# Therefore BOTH arms are scored in the SAME bins, binned by FESOM's node
# distance, and the quantity that discriminates is the RATIO between them:
#   * ratio roughly FLAT across bins -> the sparse-mesh cells are not special;
#     the excess is not explained by the remap, and a model difference stands.
#   * ratio RISES steeply with node distance -> the excess is concentrated
#     exactly where the interpolation is worst, and the remap explanation
#     carries it.
#   * tripole error itself climbing steeply while the ratio stays flat -> the
#     bins are selecting hard ocean, not a remap artifact, and neither arm is
#     implicated by this test.
# Pre-registered before the numbers exist, so the verdict cannot be fitted.
#
# SECOND PASS, ADDED AFTER THE FIRST RAN. The node-distance table came back
# ELEVATED AT BOTH ENDS with a flat middle: of the excess variance (the gap,
# not either arm's own error) 48% sits in the DENSEST bin and 39% in the
# SPARSEST, leaving 14% across the four middle bins that cover 83% of the area.
# Two ends means two mechanisms, and the dichotomy pre-registered above was too
# narrow to name either. The sparse end is what a sparse remap predicts. The
# DENSE end is not, and that is what this pass goes after.
#
# THE HYPOTHESIS, which is GLM's: the dense bin is where FESOM refines -- on
# coasts -- and it is also where the tripole enjoys a HOME-GEOMETRY ADVANTAGE,
# running on NEMO's own eORCA1 grid and land mask while FESOM does not. A
# disagreement about where the ocean ENDS would therefore read as FESOM error
# against a NEMO-anchored metric, and would concentrate on the boundary of the
# scored domain. So: stratify by distance to that boundary AT FIXED NODE
# DISTANCE, which is what the first pass could not do.
#   * excess variance in bin 1 concentrated in rings 1-2 -> the domain edge
#     carries bin 1's half of the gap, and the remaining question is per-arm
#     MASK AGREEMENT, which needs a dump field that does not exist yet.
#   * excess flat across rings inside bin 1 -> the edge is NOT it; a dense-mesh
#     representativeness effect or a real model difference survives.
#   * both arms rising toward the edge with a FLAT RATIO -> the rings select
#     hard ocean and neither arm is implicated, exactly as for node distance.
# The ALL-CELLS table is the control for the bin-1 one: a ring effect that is
# just as strong everywhere is not a fact about the dense bin.
#
# WHAT "EDGE" MEANS HERE, stated because the name would otherwise overclaim:
# the complement of the scored mask is real land UNION every cell some arm
# masked away, so ring 1 is "touching something not scored", NOT "touching a
# coastline". That is the right surface for a mask-mismatch question and the
# wrong one for a coastal-physics question; do not read it as the latter.
#
# THE GATE THAT THE RETRACTED VERSION (3558d23d2) WOULD HAVE FAILED: each
# table recombines its rings through the area-fraction identity and ABORTS if
# the result misses the stratum total by more than 5e-4. The retraction was
# caused by recombining with CELL COUNTS instead of area; a decomposition that
# cannot reproduce the number it decomposes is arithmetic about nothing.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
G=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
STEM=$G/ORCA1_5d_20000101_20000331
WORK=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/_nodedist_work

# DAY selects the mark. It defaults to 30, which is the comparison every
# published number on this thread comes from, so an unset call is unchanged.
#
# DAY=0005 IS THE CONTROL THE THREAD NEVER RAN. Every finding so far --
# 86% of the gap in one ring, interpolation eliminated, the restoring
# treatment eliminated -- compares DAY-30 states and then reasons about which
# model process built the difference. Nobody checked whether the difference
# was there at the START. Five days is the earliest mark both lanes wrote and
# it is far too short for a coastal salinity signal to be built by the physics
# under test, so if the ring-1 ratio at day 5 is already near its day-30
# value, no model process is responsible and the gap is INITIALISATION, not a
# coastal model difference. If instead it is near 1.0 and grows, the model
# really is building it and the thread's framing is right.
DAY=${DAY:-0030}
REC=$(( 10#$DAY / 5 - 1 ))
# The day-30 call keeps its ORIGINAL dump and output paths so an unset DAY
# reproduces the published comparison byte-for-byte, including where its
# figures land; other marks get their own directories and cannot overwrite it.
if [ "$DAY" = "0030" ]; then
  DUMP=$WORK/cells_d30.npz; OUT=$WORK/score
else
  DUMP=$WORK/cells_d${DAY}.npz; OUT=$WORK/score_d${DAY}
fi

# A run that REACHES its target writes that day as snapshot_final.npz, so the
# day-30 tripole is only findable under that name while its day-5 mark is not.
TRP=$R/trp_pinned_d30/snapshot_day${DAY}.npz
[ -r "$TRP" ] || TRP=$R/trp_pinned_d30/snapshot_final.npz
FES=$R/fesom_orca1faithtint_d30/snapshot_day${DAY}.npz
MPS=$R/mpas_orca1faithtint_d30/snapshot_day${DAY}.npz
for f in "$TRP" "$FES" "$MPS" ${STEM}_grid_T.nc; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
mkdir -p "$WORK" || exit 1

# The tripole fallback above can silently hand back day 30 when a mid-series
# mark is missing, and the scorer pins the oracle record from DAY without ever
# looking inside the snapshot -- so a wrong-day fallback would score day 30
# against NEMO's day 5 and exit clean. Check the stamp inside each file.
#
# THE FESOM LANE WRITES NO TIME METADATA AT ALL. Its snapshots carry only the
# fields (H_bathy, S, T, eta, ice, masks, coordinates), so the ONLY evidence of
# when a FESOM snapshot was taken is its FILE NAME -- there is nothing to check
# it against. That is reported here rather than either failing the comparison
# or letting silence read as a pass: a disagreeing stamp is fatal, a missing
# one is announced. The hazard this gate exists for is the tripole's
# snapshot_final fallback, and the tripole does stamp, so the protection holds
# where it is needed.
for f in "$TRP" "$FES" "$MPS"; do
  $PY - "$f" "$DAY" <<'GATE' || { echo "FATAL: $f is not day $DAY"; exit 1; }
import sys
import numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
tag = "/".join(sys.argv[1].split("/")[-2:])
if "time_days" not in d.files:
    print(f"  {tag}: NO TIME STAMP -- day taken from the filename, unverified")
    sys.exit(0)
got = float(np.asarray(d["time_days"]).ravel()[0])
want = float(sys.argv[2])
print(f"  {tag}: day stamp {got:g} (want {want:g})")
sys.exit(0 if abs(got - want) < 1e-6 else 1)
GATE
done

# Same inputs, labels and options as the committed day-30 scorecard, recovered
# from its own report.json, so the per-cell dump describes THAT comparison and
# not a new one. --dump-cells changes no number the scorer prints.
$PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
  --tripole "$TRP" --mpas "$FES" --also-mask "$MPS" \
  --nemo-gridt ${STEM}_grid_T.nc --nemo-time-idx "$REC" \
  --label-tripole tripole --label-mpas FESOM2 \
  --res-deg 1.0 --coast-cells 2 --mask-mode nearest \
  --out-dir "$OUT" --dump-cells "$DUMP" 2>&1 | tail -80
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "FATAL: scorer failed rc=$rc"; exit $rc; }

$PY -u - "$DUMP" <<'EOF'
import sys
import numpy as np

d = np.load(sys.argv[1], allow_pickle=True)
lab_b = str(d["label_b"])
key = f"node_km_{lab_b}"
if key not in d.files:
    raise SystemExit(
        f"VACUOUS: no node-distance grid for {lab_b!r}; the arm carries land "
        f"so it is not a node cloud, and this test does not apply. "
        f"Keys: {sorted(d.files)}")

scored = np.asarray(d["scored"], dtype=bool)
area = np.asarray(d["area"], dtype=np.float64)
nd = np.asarray(d[key], dtype=np.float64)
sa, sb, sn = (np.asarray(d[k], dtype=np.float64)
              for k in ("SSS_a", "SSS_b", "SSS_nemo"))

ea = sa - sn          # tripole minus NEMO
eb = sb - sn          # FESOM minus NEMO
finite = np.isfinite(ea) & np.isfinite(eb) & np.isfinite(nd)
use = scored & finite
n_use = int(use.sum())
print(f"scored {int(scored.sum())} cells, {n_use} with finite errors and "
      f"node distance")
if n_use < 1000:
    raise SystemExit(f"VACUOUS: only {n_use} usable cells")
if not np.any(np.abs(eb[use]) > 0):
    raise SystemExit("VACUOUS: every FESOM error is exactly zero")

def wrms(e, w, m):
    ww = w[m]
    tot = ww.sum()
    if tot <= 0:
        return float("nan")
    return float(np.sqrt((ww * e[m] ** 2).sum() / tot))

print(f"\nwhole scored domain, area-weighted SSS rmse: "
      f"tripole {wrms(ea, area, use):.4f}  {lab_b} {wrms(eb, area, use):.4f}")

# Bin edges are the node cloud's OWN geometry, not round numbers: its median
# spacing and its own percentiles, so "far" means far FOR THIS MESH.
q = np.percentile(nd[use], [25, 50, 75, 90, 99])
edges = np.concatenate([[0.0], q, [np.inf]])
print(f"\nbins from {lab_b}'s own nearest-node percentiles "
      f"(25/50/75/90/99 = " + "/".join(f"{x:.0f}" for x in q) + " km)")
print(f"{'node dist [km]':>18} {'cells':>7} {'area frac':>9} "
      f"{'tripole':>9} {lab_b:>9} {'ratio':>7}")
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = use & (nd >= lo) & (nd < hi)
    n = int(m.sum())
    if n == 0:
        continue
    ra, rb = wrms(ea, area, m), wrms(eb, area, m)
    frac = float(area[m].sum() / area[use].sum())
    rows.append((lo, hi, n, frac, ra, rb, rb / ra if ra > 0 else float("nan")))
    hi_s = "inf" if not np.isfinite(hi) else f"{hi:.0f}"
    print(f"{lo:8.0f} - {hi_s:>7} {n:7d} {frac:9.3f} "
          f"{ra:9.4f} {rb:9.4f} {rb / ra if ra > 0 else float('nan'):7.2f}")

if len(rows) < 3:
    raise SystemExit(f"VACUOUS: only {len(rows)} non-empty bins")

ratios = [r[6] for r in rows if np.isfinite(r[6])]
print(f"\nratio {lab_b}/tripole across {len(ratios)} bins: "
      f"min {min(ratios):.2f}, max {max(ratios):.2f}, "
      f"spread {max(ratios) / min(ratios):.2f}x")
print("Read this against the PRE-REGISTERED reading in the card header: a flat "
      "ratio leaves the remap explanation unsupported, a steeply rising one "
      "carries it. The ratio, not either arm's own trend, is the discriminator "
      "-- both arms see the same bins, so anything the bins select about ocean "
      "difficulty divides out.")

# ---------------------------------------------------------------------------
# SECOND STRATIFIER: DISTANCE TO THE EDGE OF THE SCORED DOMAIN, AT FIXED NODE
# DISTANCE.  See the card header for the pre-registered reading.
# ---------------------------------------------------------------------------
if "edge_ring" not in d.files:
    raise SystemExit(
        "VACUOUS: this dump predates --dump-cells writing edge_ring; "
        "re-run the scorer above rather than reading a stale npz.")
ring = np.asarray(d["edge_ring"], dtype=int)

# ORIENTATION GATE. The recombination gate below proves the rings PARTITION the
# stratum, which an inverted labelling would also satisfy -- and reading the
# table upside down is exactly the kind of error that survives a passing check.
# So verify independently, without reusing the dilation that built the labels:
# every ring-1 cell must touch a non-scored cell among its eight neighbours,
# and no interior cell may.
touch = np.zeros(scored.shape, dtype=bool)
for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
        if dy == 0 and dx == 0:
            continue
        s = np.roll(~scored, dx, axis=1)         # longitude periodic
        sh = np.zeros_like(s)
        if dy == 0:
            sh = s
        elif dy > 0:
            sh[dy:] = s[:-dy]
        else:
            sh[:dy] = s[-dy:]
        touch |= sh
bad1 = int((scored & (ring == 1) & ~touch).sum())
bad0 = int((scored & (ring == 0) & touch).sum())
if bad1 or bad0:
    raise SystemExit(
        f"FATAL: edge_ring is not oriented as documented -- {bad1} ring-1 "
        f"cells touch no unscored neighbour and {bad0} interior cells do. "
        f"Every ring number below would be read backwards.")
print(f"[gate] edge_ring orientation verified on {int((ring == 1).sum())} "
      f"ring-1 cells against an independent neighbour test")

# The excess variance is the quantity that decomposes the GAP rather than
# describing either arm: with frac_s the area fraction of stratum s, the
# identity total_rms^2 = sum_s frac_s * rms_s^2 holds for EACH arm, so the
# difference of the two totals splits as sum_s frac_s * (rms_b^2 - rms_a^2).
def excess_table(mask_all, title):
    tot = area[mask_all].sum()
    if tot <= 0:
        print(f"\n{title}: EMPTY")
        return
    ra_all, rb_all = wrms(ea, area, mask_all), wrms(eb, area, mask_all)
    gap = rb_all ** 2 - ra_all ** 2
    print(f"\n{title}")
    print(f"  whole stratum: tripole {ra_all:.4f}  {lab_b} {rb_all:.4f}  "
          f"n={int(mask_all.sum())}")
    print(f"{'edge ring':>12} {'cells':>7} {'area frac':>9} {'tripole':>9} "
          f"{lab_b:>9} {'ratio':>7} {'% of gap':>9}")
    # Rings run 1 (touching a non-scored cell) outward; 0 is everything
    # deeper inside than the dump labelled, i.e. the interior.
    groups = [(k, f"{k}") for k in range(1, int(ring.max()) + 1)]
    groups.append((0, "interior"))
    acc_a = acc_b = 0.0
    n_rows = 0
    for val, lab in groups:
        m = mask_all & (ring == val)
        n = int(m.sum())
        if n == 0:
            continue
        n_rows += 1
        frac = float(area[m].sum() / tot)
        ra, rb = wrms(ea, area, m), wrms(eb, area, m)
        acc_a += frac * ra ** 2
        acc_b += frac * rb ** 2
        share = (frac * (rb ** 2 - ra ** 2) / gap * 100.0) if gap != 0 else float("nan")
        print(f"{lab:>12} {n:7d} {frac:9.3f} {ra:9.4f} {rb:9.4f} "
              f"{rb / ra if ra > 0 else float('nan'):7.2f} {share:8.1f}%")
    if n_rows < 3:
        raise SystemExit(f"VACUOUS: only {n_rows} non-empty rings in {title}")
    # The recombination gate: if these do not reproduce the stratum totals the
    # decomposition is not the identity it claims to be, and every share above
    # is arithmetic about nothing.  This is the check the RETRACTED
    # count-weighted version (3558d23d2) would have failed.
    print(f"  recombine: tripole {np.sqrt(acc_a):.4f} vs {ra_all:.4f}, "
          f"{lab_b} {np.sqrt(acc_b):.4f} vs {rb_all:.4f}")
    for got, want, who in ((np.sqrt(acc_a), ra_all, "tripole"),
                           (np.sqrt(acc_b), rb_all, lab_b)):
        if abs(got - want) > 5e-4:
            raise SystemExit(
                f"FATAL: {who} ring recombination {got:.6f} != stratum total "
                f"{want:.6f}; the area-fraction identity does not hold and "
                f"nothing above is readable.")

lo1, hi1 = edges[0], edges[1]
bin1 = use & (nd >= lo1) & (nd < hi1)
excess_table(bin1, f"BIN 1 ONLY (node distance {lo1:.0f}-{hi1:.0f} km), "
                   f"by distance to the scored domain's edge")
excess_table(use, "ALL SCORED CELLS, by distance to the scored domain's edge "
                  "(the control: is any ring effect specific to bin 1?)")
EOF
rc2=$?
echo "=== card exit: $rc2 ==="
exit $rc2
