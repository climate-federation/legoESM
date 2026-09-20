#!/bin/bash
#SBATCH --job-name=nino3bins
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nino3bins_%j.out
#
# WHERE, vertically, does our nino3 extra heat sit?
#
# The nino3 warm bias is the leading error (+0.842 C at day 25, +1.147 at day
# 30) and the surface heat flux has already been eliminated as its cause: we
# put about 30 W/m2 LESS heat into the box than NEMO at both days and are warm
# anyway. So the heat either arrives by advection or fails to leave downward.
#
# This separates those by depth band on OUR OWN run at two matched days,
# using the committed tracer-content probe (locked conventions, fp64, mesh
# metrics as the area weights, z-star dilation so a free-surface volume change
# is not misread as a content change).
#
# ATTRIBUTION IS ALREADY SETTLED for the sign: over day 25 to day 30 NEMO's own
# nino3 SST rises 25.24 -> 25.39 (+0.15) while ours rises 26.08 -> 26.54
# (+0.46), so the growing bias is OUR warming, not the oracle cooling.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/_omip_merge
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
echo "=== dirty files (provenance): $(git -C $REPO status --porcelain | wc -l)"
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

RUN=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1faithtint_d30
MESH=/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc
OUTJ=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/nino3_heat_bins.json

for f in "$RUN/snapshot_day0025.npz" "$RUN/snapshot_day0030.npz" "$MESH"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

echo "=== nino3 box (5S-5N, 150W-90W), control run, days 25 and 30 ==="
$PY scripts/validate/ocean_fidelity/global_tracer_content.py \
    --snapshot "$RUN/snapshot_day0025.npz" "$RUN/snapshot_day0030.npz" \
    --mesh-mask "$MESH" \
    --region=-5,5,-150,-90 \
    --json-out "$OUTJ"
rc=$?
echo "=== rc=$rc ==="

echo "=== per-bin change, day 25 -> day 30 ==="
$PY - "$OUTJ" <<'PYEOF'
import json, sys
rep = json.load(open(sys.argv[1]))
snaps = list(rep["snapshots"].items())
if len(snaps) != 2:
    raise SystemExit(f"expected 2 snapshots, got {len(snaps)}")
(_, a), (_, b) = snaps
print(f"region: {rep['region']}")
print(f"box mean T: {a['mean_T_C']:.4f} -> {b['mean_T_C']:.4f} C "
      f"({b['mean_T_C']-a['mean_T_C']:+.4f})")
print(f"box volume: {a['volume_m3']:.6e} -> {b['volume_m3']:.6e} m3")
ha, hb = a["heat_by_depth_bin_C_m3"], b["heat_by_depth_bin_C_m3"]
va, vb = a["volume_by_depth_bin_m3"], b["volume_by_depth_bin_m3"]
print(f"{'bin':>12} {'T d25 [C]':>11} {'T d30 [C]':>11} {'dT':>9} {'vol frac':>9}")
vtot = sum(va.values())
for k in ha:
    # Each bin's content divided by its OWN volume, so the bands are read in
    # degrees instead of in an extensive quantity whose size is set by the
    # bin's thickness.
    ta = ha[k] / va[k] if va[k] > 0 else float("nan")
    tb = hb[k] / vb[k] if vb[k] > 0 else float("nan")
    print(f"{k:>12} {ta:11.4f} {tb:11.4f} {tb-ta:+9.4f} {va[k]/vtot:9.3f}")
PYEOF
