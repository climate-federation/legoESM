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
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
export PYTHONPATH="$M/src:$(ls -d $M/packages/*/ | sed 's:/$::' | tr '\n' ':')$M:$M/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
G=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY
STEM=$G/ORCA1_5d_20000101_20000331
WORK=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/_nodedist_work
DUMP=$WORK/cells_d30.npz

TRP=$R/trp_pinned_d30/snapshot_final.npz
FES=$R/fesom_orca1faithtint_d30/snapshot_day0030.npz
MPS=$R/mpas_orca1faithtint_d30/snapshot_day0030.npz
for f in "$TRP" "$FES" "$MPS" ${STEM}_grid_T.nc; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done
mkdir -p "$WORK" || exit 1

# Same inputs, labels and options as the committed day-30 scorecard, recovered
# from its own report.json, so the per-cell dump describes THAT comparison and
# not a new one. --dump-cells changes no number the scorer prints.
$PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
  --tripole "$TRP" --mpas "$FES" --also-mask "$MPS" \
  --nemo-gridt ${STEM}_grid_T.nc --nemo-time-idx 5 \
  --label-tripole tripole --label-mpas FESOM2 \
  --res-deg 1.0 --coast-cells 2 --mask-mode nearest \
  --out-dir "$WORK/score" --dump-cells "$DUMP" 2>&1 | tail -80
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
EOF
rc2=$?
echo "=== card exit: $rc2 ==="
exit $rc2
