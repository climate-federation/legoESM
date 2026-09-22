#!/bin/bash
#SBATCH --job-name=regnull
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/regnull_%j.out
#
# WHERE DOES THE SAME-CONFIGURATION DIVERGENCE ACTUALLY LIVE?
#
# Two runs of one configuration, from one restart, differ at day 20 by up to
# 2.614 K in temperature over roughly 42% of wet cells. In the equatorial
# Pacific that divergence is invisible: the undercurrent core speed and net
# transport reproduce to the printed digit, and the sea-level gradient to two
# parts in a thousand (measured, job 9923264). So the noise is large somewhere
# and negligible there, and until it is located, every regional number this
# campaign quotes outside the equator is a difference with no error bar.
#
# Those numbers are not incidental -- they are the campaign directive's own
# list: Pacific SST in the ENSO region, southern ocean SST and mixed-layer
# depth, the coast of North America, polar salinity. The current scorecard
# reads southern ocean -0.107, off North America -0.012, Arctic +0.010 against
# NEMO. If two identical runs differ by more than that in the same region, the
# corresponding scorecard row is not yet a measurement.
#
# REGIONS ARE REUSED, NOT INVENTED. The boxes come from _NAMED_BOXES in
# compare_three_way_nemo.py -- nino3, eq_pacific, southern_ocean,
# off_north_america -- plus that module's Arctic definition, latitude at or
# north of 60. Defining fresh boxes here would make these numbers
# incomparable with every scorecard the campaign has already produced, which
# is the entire reason for not doing it.
#
# AREA WEIGHTING IS NOT OPTIONAL on a tripole mesh: cell area varies by more
# than an order of magnitude between the equator and the northern fold, so an
# unweighted regional mean is not a regional mean. The snapshots carry
# cell_area and it is used for every statistic below. Both the weighted RMS
# and the raw maximum are reported, because a small RMS with a large maximum
# is a localized flip and means something different from a broad shift.
#
# PRE-REGISTERED: for each region, compare the same-configuration spread to
# that region's standing ours-minus-NEMO bias. Spread far below the bias means
# the scorecard row is a measurement. Spread comparable to the bias means the
# row needs an ensemble before it can be read as a model difference at all.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
# DAY is settable so the same null can be re-read as the pair diverges
# further. That matters: a floor sampled after five days of divergence
# understates the floor after fifteen, and the scorecard rows it is being
# compared against are quoted at day 30.
DAY=${DAY:-0020}
A=$R/trp_faithz0_d30/snapshot_day${DAY}.npz
B=$R/trp_faithz0_d90/snapshot_day${DAY}.npz
echo "### same-configuration null at day ${DAY}"
for f in "$A" "$B"; do [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }; done

$PY -u - "$A" "$B" <<'EOF'
import sys
import numpy as np

# Boxes copied from _NAMED_BOXES in compare_three_way_nemo.py, 0..360 lon.
BOXES = (
    ("nino3", -5.0, 5.0, 210.0, 270.0),
    ("eq_pacific", -2.0, 2.0, 180.0, 280.0),
    ("southern_ocean", -70.0, -45.0, 0.0, 360.0),
    ("off_north_america", 30.0, 45.0, 280.0, 310.0),
)
ARCTIC_LAT_N = 60.0

A = np.load(sys.argv[1], allow_pickle=True)
B = np.load(sys.argv[2], allow_pickle=True)
lat = np.asarray(A["lat_T"], dtype=np.float64)
lon = np.asarray(A["lon_T"], dtype=np.float64) % 360.0
area = np.asarray(A["cell_area"], dtype=np.float64)
# The field named land_mask holds 1 for OCEAN in these snapshots, not for
# land. The first version of this card guessed the other way and every
# statistic came out exactly zero -- because it was averaging over land fill
# values, which are identical in both files. The guess is no longer made:
# the polarity is DERIVED from where the ocean actually has water, and the
# derivation is asserted.
mask_raw = np.asarray(A["land_mask"], dtype=np.float64)
Tsurf_a = np.take(np.asarray(A["T"], dtype=np.float64), 0, axis=2)
hi = mask_raw > 0.5
lo = ~hi
# Ocean cells carry a spread of temperatures; land fill does not.
spread_hi = float(np.nanstd(Tsurf_a[hi])) if hi.any() else 0.0
spread_lo = float(np.nanstd(Tsurf_a[lo])) if lo.any() else 0.0
wet = hi if spread_hi > spread_lo else lo
polarity = "1=ocean" if wet is hi else "0=ocean"
print(f"land_mask polarity DERIVED as {polarity}: surface-T spread is "
      f"{spread_hi:.3f} K where mask>0.5 and {spread_lo:.3f} K where mask<=0.5")
print(f"wet cells: {int(wet.sum())} of {wet.size} "
      f"({100.0 * wet.sum() / wet.size:.1f}%)")
assert wet.sum() > 0.5 * wet.size, "wet fraction implausibly small for a global ocean mesh"

def region_masks():
    for name, s, n, w, e in BOXES:
        inlat = (lat >= s) & (lat <= n)
        inlon = (lon >= w) & (lon <= e) if w <= e else ((lon >= w) | (lon <= e))
        yield name, wet & inlat & inlon
    yield "arctic", wet & (lat >= ARCTIC_LAT_N)
    yield "global", wet

def stats(diff2d, mask):
    m = mask & np.isfinite(diff2d)
    if not m.any():
        return None
    w = area[m]
    d = diff2d[m]
    rms = float(np.sqrt(np.sum(w * d * d) / np.sum(w)))
    mean = float(np.sum(w * d) / np.sum(w))
    # How many cells differ AT ALL, bitwise. This is the discriminator a
    # rounded print cannot give: a region that is bitwise identical across
    # thousands of active cells after days of divergence is masked or
    # clamped, not merely quiet, whereas a region where most cells differ
    # by a tiny amount is simply one the divergence has not reached yet.
    return rms, mean, float(np.abs(d).max()), int(m.sum()), int(np.sum(d != 0.0))

for var in ("T", "S", "eta"):
    x = np.asarray(A[var], dtype=np.float64)
    y = np.asarray(B[var], dtype=np.float64)
    # Surface level is the scored quantity (SST / SSS); axis order is
    # reported so a wrong guess about it cannot pass silently. eta is
    # already two-dimensional and is included because it discriminates
    # between the two explanations for a region that does not move: a
    # masking bug freezes EVERY field there, whereas sea level responding
    # while temperature does not points at a surface clamp.
    print(f"\n=== {var}  shape {x.shape} ===")
    if x.ndim == 2:
        surf = x - y
        print(f"two-dimensional field; shape {surf.shape}, grid shape {lat.shape}")
    else:
        lvl_axis = int(np.argmin(x.shape))
        surf = np.take(x, 0, axis=lvl_axis) - np.take(y, 0, axis=lvl_axis)
        print(f"level axis taken as {lvl_axis} (size {x.shape[lvl_axis]}); "
              f"surface field shape {surf.shape}, grid shape {lat.shape}")
    if surf.shape != lat.shape:
        print("  !! surface field does not match the grid shape -- axis guess wrong, skipping")
        continue
    # The weighted MEAN is the column to read against a scorecard bias. A
    # root-mean-square is the wrong floor for a bias statistic: it cannot
    # cancel, so it is systematically larger and flatters nothing.
    print(f"  {'region':20s} {'wRMS':>12s} {'wmean(bias)':>14s} "
          f"{'max|d|':>12s} {'cells':>8s} {'differ':>8s}")
    seen = {}
    for name, mask in region_masks():
        s = stats(surf, mask)
        if s is None:
            print(f"  {name:20s} {'no cells':>12s}")
            continue
        rms, mean, mx, n, ndiff = s
        seen[name] = (rms, n)
        print(f"  {name:20s} {rms:12.3e} {mean:+14.3e} {mx:12.3e} "
              f"{n:8d} {ndiff:8d}")

    # NON-VACUITY GATES. The reproducibility job already proved these two
    # files differ -- 2.614 K in T over roughly 42% of cells. So a table of
    # zeros here does not mean the runs agree, it means this card is
    # measuring the wrong cells, which is exactly what happened on its first
    # run. Both gates below would have caught that immediately.
    if seen.get("global", (0.0, 0))[0] <= 0.0:
        raise SystemExit(
            f"VACUOUS: global {var} difference is exactly zero, but the two "
            f"files are known to differ. The mask or the level axis is wrong."
        )
    n_nino3 = seen.get("nino3", (0.0, 0))[1]
    if n_nino3 < 500:
        raise SystemExit(
            f"VACUOUS: nino3 resolves only {n_nino3} cells. That box spans 10 "
            f"degrees of latitude and 60 of longitude in open ocean and must "
            f"hold thousands. Check the units and layout of lat_T / lon_T."
        )

print("\nREAD THIS AGAINST the standing day-30 scorecard, ours minus NEMO: "
      "southern ocean -0.107, off North America -0.012, Arctic +0.010, "
      "global SST +0.029 with rmse 0.368. A same-configuration spread "
      "comparable to one of those rows means that row is not yet a "
      "measurement of anything about the model.")
EOF
rc=$?
echo "=== card exit: $rc ==="
exit $rc
