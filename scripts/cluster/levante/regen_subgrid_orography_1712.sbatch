#!/usr/bin/env bash
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/work/bd1083/b309178/diffESM/legoesm_pg/sso_regen_1712/%x-%j.out
#
# #1712: build subgrid-orography (SSO_STDH) files whose scale decomposition is
# ANCHORED to the model grid, and report how far they sit from the file
# production reads.  Submit from the repo root (ROOT defaults to it).
#
# Anchor (prep_subgrid_orography.anchored_sso_scales): block = model cell,
# resolved cutoff = EFFECTIVE_RESOLUTION_DX (3.5) cells, the cell taken from
# legoesm.grids.topography.voronoi_cell_spacing_deg -- the same numbers the
# loader's scale check uses, so every file built here passes it on its mesh.
#
# Source: ETOPO 2022 at 1 arc-minute (elevation ``z``).  The previous version
# of this script read an ICON extpar SSO file, which holds subgrid STATISTICS,
# not elevation, compared against that file instead of the one production
# reads, used cutoffs (4.0 / 7.5 deg) above the guard's tolerance, and ran from
# a directory that is not the repo.
#
# (mkdir -p /work/bd1083/b309178/diffESM/legoesm_pg/sso_regen_1712 before
# submitting; PY must name a Python with legoesm's dependencies.)
#
# Both residual estimators are built (block_stddev, the default, and rms); see
# subgrid_orography_residual_stddev for what each keeps and drops.  This script
# selects nothing: pointing production at one of these files is a decision.
set -euo pipefail

ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
SRC=${SRC:-/work/bd1083/b309178/diffESM/legoesm_ggn/ETOPO_2022_60s.nc}
SRC1DEG=${SRC1DEG:-/work/bd1083/b309178/diffESM/legoesm_ap/data/bathymetry/etopo_1deg_clean.nc}
PROD=${PROD:-/work/bd1083/b309178/diffESM/legoesm_ap/data/orography/sso_stdh_2deg_from1deg.nc}
OUT=${OUT:-/work/bd1083/b309178/diffESM/legoesm_pg/sso_regen_1712}
: "${PY:?set PY to a Python with the legoesm dependencies}"
for f in "$SRC" "$SRC1DEG" "$PROD" "$PY"; do
    [ -e "$f" ] || { echo "FATAL: missing $f"; exit 2; }
done
[ -f "$ROOT/scripts/data/prep_subgrid_orography.py" ] || { echo "FATAL: ROOT=$ROOT is not the repo"; exit 2; }
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$(ls -d "$ROOT"/packages/*/ | tr '\n' ':')"
FINE=$("$PY" -c 'print(repr(1/60))')
cell () { "$PY" -c "from legoesm.grids.topography import voronoi_cell_spacing_deg as f; print(repr(f($1)))"; }
L6=$(cell 40962); L5=$(cell 10242)
echo "sha=$(git rev-parse --short HEAD) host=$(hostname) $(date -u) fine=$FINE cell_l6=$L6 cell_l5=$L5"

echo "### control C1: rebuild the production file from its own recipe"
"$PY" scripts/data/prep_subgrid_orography.py --input "$SRC1DEG" --out "$OUT/c1_rebuild.nc" \
    --fine-res-deg 1.0 --block-deg 2.0
"$PY" - "$PROD" "$OUT/c1_rebuild.nc" <<'EOF'
import sys, numpy as np, xarray as xr
a = xr.open_dataset(sys.argv[1])["SSO_STDH"].values
b = xr.open_dataset(sys.argv[2])["SSO_STDH"].values
d = float(np.max(np.abs(a - b)))
print(f"C1 max|rebuild - production| = {d:.3e} m")
assert a.shape == b.shape and d == 0.0, "C1 FAILED: the instrument does not reproduce the production file"
print("C1 PASS")
EOF

echo "### control C3: regrid identity at the source resolution"
"$PY" - "$SRC" "$FINE" <<'EOF'
import sys, numpy as np, xarray as xr
sys.path.insert(0, "scripts/data")
from prep_etopo_topography import regrid_elevation_to_latlon
ds = xr.open_dataset(sys.argv[1])
fine = regrid_elevation_to_latlon(ds, target_res_deg=float(sys.argv[2]))["elevation"].values
raw = ds["z"].assign_coords(lon=ds["lon"].values % 360).sortby("lon").sortby("lat").values
assert fine.shape == raw.shape, (fine.shape, raw.shape)
d = float(np.abs(fine - raw).max())
print(f"C3 max|regrid(1') - source| = {d:.3e} m")
# identical coordinates up to the lon % 360 rounding: the interpolation weights
# are 1 +- 1e-12, so anything above a millimetre is a real resampling.
assert d < 1e-3, "C3 FAILED: the fine grid is not the source grid"
print("C3 PASS")
EOF

gen () {   # label cell fine estimator
    "$PY" scripts/data/prep_subgrid_orography.py --input "$SRC" --elev-var z \
        --out "$OUT/sso_$1.nc" --fine-res-deg "$3" --model-cell-deg "$2" \
        --residual-estimator "$4"
}
echo "### stage 1: anchored files from the 1' source"
for est in block_stddev rms; do
    gen "l6_${est}" "$L6" "$FINE" "$est"
    gen "l5_${est}" "$L5" "$FINE" "$est"
    # same source point-sampled (bilinear) at 0.25 deg: separates the source
    # resolution from the band definition in the ratio to production
    gen "l6_${est}_fine025" "$L6" 0.25 "$est"
done

echo "### stage 2: against the file production reads (input-field statistics)"
"$PY" scripts/validate/sso_anchor_report.py --reference "$PROD" --mesh-level 6 \
    --json "$OUT/report_l6.json" --files "$OUT"/sso_l6_*.nc
"$PY" scripts/validate/sso_anchor_report.py --reference "$PROD" --mesh-level 5 \
    --json "$OUT/report_l5.json" --files "$OUT"/sso_l5_*.nc
echo "DONE $(date -u)"
