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
A=$R/trp_faithz0_d30/snapshot_day0020.npz
B=$R/trp_faithz0_d90/snapshot_day0020.npz
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
wet = np.asarray(A["land_mask"]) < 0.5 if A["land_mask"].max() <= 1.0 else np.asarray(A["land_mask"]) > 0.5
print(f"wet cells: {int(wet.sum())} of {wet.size}   (land_mask min {A['land_mask'].min()}, max {A['land_mask'].max()})")

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
    return rms, mean, float(np.abs(d).max()), int(m.sum())

for var in ("T", "S"):
    x = np.asarray(A[var], dtype=np.float64)
    y = np.asarray(B[var], dtype=np.float64)
    # Surface level is the scored quantity (SST / SSS); axis order is
    # reported so a wrong guess about it cannot pass silently.
    print(f"\n=== {var}  shape {x.shape} ===")
    lvl_axis = int(np.argmin(x.shape))
    surf = np.take(x, 0, axis=lvl_axis) - np.take(y, 0, axis=lvl_axis)
    print(f"level axis taken as {lvl_axis} (size {x.shape[lvl_axis]}); "
          f"surface field shape {surf.shape}, grid shape {lat.shape}")
    if surf.shape != lat.shape:
        print("  !! surface field does not match the grid shape -- axis guess wrong, skipping")
        continue
    print(f"  {'region':20s} {'wRMS':>10s} {'wmean':>10s} {'max|d|':>10s} {'cells':>9s}")
    for name, mask in region_masks():
        s = stats(surf, mask)
        if s is None:
            print(f"  {name:20s} {'no cells':>10s}")
            continue
        rms, mean, mx, n = s
        print(f"  {name:20s} {rms:10.4f} {mean:+10.4f} {mx:10.4f} {n:9d}")

print("\nREAD THIS AGAINST the standing day-30 scorecard, ours minus NEMO: "
      "southern ocean -0.107, off North America -0.012, Arctic +0.010, "
      "global SST +0.029 with rmse 0.368. A same-configuration spread "
      "comparable to one of those rows means that row is not yet a "
      "measurement of anything about the model.")
EOF
rc=$?
echo "=== card exit: $rc ==="
exit $rc
