#!/usr/bin/env bash
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/sso_regen/%x-%j.out
#
# #1712 step 1: regenerate the subgrid-orography field as a RESIDUAL variance
# cut at the model's own effective resolution, instead of the block standard
# deviation of four samples that production reads today.
#
# WHY THIS RUNS ON LEVANTE AND NOT ON GINSBURG: the source is
# ``extpar_sso_latlon_0p25deg.nc`` under /work/bd1083, which is a Levante path.
# The only elevation data on the Ginsburg filesystem is a 1-degree ETOPO, and a
# 1-degree source cannot resolve sub-2-degree variance at all -- using it would
# reproduce, at a coarser scale, precisely the "stddev of four samples" defect
# this issue is about.
#
# THE CUTOFF IS DERIVED, NOT PICKED.  That is the whole point of #1712: the
# current field's scale decomposition is two hand-chosen numbers with a 33x
# lever on Southern-Ocean drag and no tie to the model grid.  Here the upper
# cutoff is the model's EFFECTIVE resolution, taken from the constant the
# loader's own scale check already uses (_EFFECTIVE_RESOLUTION_DX = 3.5 cells,
# grids/topography.py), and the block is the model cell:
#
#     MPAS level 5  ~2.2 deg cell  ->  block 2.0 deg, resolved cutoff 7.5 deg
#     MPAS level 6  ~1.1 deg cell  ->  block 1.0 deg, resolved cutoff 4.0 deg
#
# (block and cutoff are both snapped to a multiple of the 0.25 deg source, as
# ``subgrid_orography_residual_stddev`` requires; it also refuses a cutoff
# below the block, which would measure variance the block stddev already has.)
#
# WHAT THIS DOES NOT DO: it does not point production at the result.  Selecting
# one of these files changes ``sgh``, and the launch stress goes as ``sgh^2``
# -- up to 33x in the Southern Ocean on the sweep in #1712.  That is a new
# baseline, not a comparison, and it needs the drag efficiency retuned against
# it plus a validation arm before any campaign reads it.  Stage 2 below only
# REPORTS how far the new field sits from the one in use.
set -uo pipefail

ROOT=${ROOT:-/work/bd1083/b309178/diffESM/legoesm_pg}
SRC=${SRC:-/work/bd1083/b309178/diffESM/land_data/extpar_sso_latlon_0p25deg.nc}
OUT=${OUT:-$ROOT/sso_regen}
PY=${PY:-python}
mkdir -p "$OUT"
cd "$ROOT" || exit 2
export PYTHONPATH="$(ls -d $ROOT/packages/*/ 2>/dev/null | sed 's:/$::' | tr '\n' ':')$ROOT"

echo "sha=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)  host=$(hostname)  $(date -u)"
[ -f "$SRC" ] || { echo "FATAL: source elevation file missing: $SRC"; exit 2; }

gen () {   # $1 = label, $2 = block_deg, $3 = resolved_cutoff_deg
    echo "=== $1: block=$2 deg, resolved cutoff=$3 deg $(date -u) ==="
    "$PY" -u scripts/data/prep_subgrid_orography.py \
        --input "$SRC" \
        --out "$OUT/sso_residual_$1.nc" \
        --fine-res-deg 0.25 \
        --block-deg "$2" \
        --resolved-cutoff-deg "$3"
}

echo "############ STAGE 1: generate ############"
gen mpas_l5 2.0 7.5
gen mpas_l6 1.0 4.0

echo "############ STAGE 2: how far is this from what production reads? ############"
# Reported, not applied.  The drag goes as sgh^2, so the ratio of the means of
# the SQUARE is the number that matters, not the ratio of the stddevs.
"$PY" - "$SRC" "$OUT/sso_residual_mpas_l5.nc" "$OUT/sso_residual_mpas_l6.nc" <<'PYEOF'
import sys
import numpy as np
import xarray as xr

cur = xr.open_dataset(sys.argv[1])
name = "SSO_STDH" if "SSO_STDH" in cur else list(cur.data_vars)[0]
a = np.asarray(cur[name].values, dtype=np.float64)
lat = None
for c in cur[name].dims:
    if "lat" in c.lower():
        lat = np.asarray(cur[c].values, dtype=np.float64)
print(f"IN USE   {sys.argv[1]}")
print(f"  {name}: mean {np.nanmean(a):.2f} m, max {np.nanmax(a):.1f} m")

for path in sys.argv[2:]:
    ds = xr.open_dataset(path)
    b = np.asarray(ds["SSO_STDH"].values, dtype=np.float64)
    print(f"\nNEW      {path}")
    print(f"  attrs: construction={ds.attrs.get('construction')!r} "
          f"block_deg={ds.attrs.get('block_deg')} "
          f"resolved_cutoff_deg={ds.attrs.get('resolved_cutoff_deg')}")
    print(f"  SSO_STDH: mean {np.nanmean(b):.2f} m, max {np.nanmax(b):.1f} m")
    if b.shape == a.shape:
        # Drag ~ sgh^2, so compare the means of the SQUARES, area weighted.
        w = (np.cos(np.deg2rad(lat))[:, None] * np.ones((1, a.shape[1]))
             if lat is not None and lat.size == a.shape[0] else np.ones_like(a))
        num = np.nansum(w * b ** 2)
        den = np.nansum(w * a ** 2)
        print(f"  area-weighted <sgh^2> ratio vs the file in use: "
              f"{num / den:.3f}x  (this is the drag factor, not the sgh ratio)")
        so = (lat >= -60) & (lat <= -40) if lat is not None else None
        if so is not None and so.any():
            ws = w[so]
            print(f"  same ratio over 40-60S only: "
                  f"{np.nansum(ws * b[so] ** 2) / np.nansum(ws * a[so] ** 2):.3f}x")
    else:
        print(f"  shapes differ ({b.shape} vs {a.shape}); regrid before comparing")
PYEOF
echo DONE $(date -u)
