#!/bin/bash
#SBATCH --job-name=fesomnyf
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesomnyf_%j.out
#
# FESOM2 (fesom-jax) 30-day arm on CORE-II NORMAL-YEAR forcing (fesom
# c74311d + legoESM 6b93b0b04).  ONE variable vs the fesom_core2_d30 arm
# 9369143: the atmospheric forcing (JRA55-do 1958 -> the same nyf.zarr the
# tripole/MPAS arms use).
#
# WHAT THIS ARM IS NOT.  It is not a matched comparison against the NEMO
# reference, and the pre-registered numbers below must not be read as one.
# Four inputs still differ on the FESOM side: the initial temperature and
# salinity climatology, the surface-salinity restoring, the river runoff and
# the chlorophyll field.  A fifth was found in review: the wind speeds.  The
# cache read here carries the RAW CORE-II winds, mean 4.74 m/s over the
# equatorial Pacific, while the NEMO reference reads the scatterometer-
# corrected winds at 6.28 m/s -- roughly 1.8x the surface stress.  The
# corrected cache exists (core2_nyf_mod, built by _build_nyf_mod.sbatch) but
# three arms already ran against the raw one and stay reproducible on it, so
# THIS arm deliberately keeps the raw winds and is comparable to the other
# legoESM arms, not to NEMO.  The wind variant is asserted below so a log can
# never be read without knowing which one ran.
#
# PRE-REGISTERED, day-30 vs NEMO m1 (protocol confound test):
#   FESOM2_vs_nemo SST rmse [was 1.49 vs tripole 0.85]:
#     CONFIRMED protocol-dominated : <= 1.1
#     PARTIAL                      : 1.1 - 1.35
#     LARGE RESIDUAL               : >= 1.35
#   A LARGE RESIDUAL here does NOT establish "model, not forcing": with the
#   winds, the initial state, the restoring, the runoff and the chlorophyll
#   all still differing, forcing has not been eliminated. Reading it that way
#   would attribute to the ocean an error the inputs can produce on their own.
#   Watch also the hemispheric dipole (SH_midlat +1.47 / NH_midlat -1.94):
#   a JRA-year seasonal-phase artifact should COLLAPSE under NYF.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
FJX=/burg-archive/glab/users/pg2328/fesom_jax_repo
D=$REPO/data/fesom2_core2
NYF=/burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/core2_nyf/nyf.zarr
cd $REPO || exit 1
PIN=${1:?usage: sbatch _fesom2_nyf_d30.sbatch <legoESM-sha> <fesom-sha>}
FPIN=${2:?usage: sbatch _fesom2_nyf_d30.sbatch <legoESM-sha> <fesom-sha>}
[ "$(git -C $REPO rev-parse HEAD)" = "$PIN" ] || { echo "FATAL: legoESM HEAD != $PIN"; exit 1; }
[ "$(git -C $FJX rev-parse HEAD)" = "$FPIN" ] || { echo "FATAL: fesom HEAD != $FPIN"; exit 1; }
echo "=== HEAD $(git -C $REPO log --oneline -1)"
echo "=== fesom_jax $(git -C $FJX log --oneline -1)"
[ -d "$NYF" ] || { echo "FATAL: no nyf.zarr at $NYF"; exit 1; }

MI=$D/mesh_ic/core2_mesh_ic
export FESOM_SSS_PATH="$MI/PHC2_salx.nc"
export FESOM_RUNOFF_PATH="$MI/CORE2_runoff.nc"
export FESOM_CHL_PATH="$MI/Sweeney_2005.nc"
export FESOM_PHC_PATH="$MI/phc3.0_winter.nc"
for f in "$FESOM_SSS_PATH" "$FESOM_RUNOFF_PATH" "$FESOM_CHL_PATH"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

export PYTHONPATH="$FJX:$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_ENABLE_X64=0     # fesom-jax published hindcast runs f32; keep ITS convention
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

echo "=== gate: NYF reader + driver CLI tests ==="
( cd $FJX && JAX_PLATFORMS=cpu PYTHONPATH=$FJX $PY -m pytest -q \
    fesom_jax/tests/test_nyf_forcing.py -p no:cacheprovider 2>&1 | tail -2 )
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate failed rc=$rc"; exit 1; }

echo "=== assert which winds this cache carries (raw 4.74 vs corrected 6.28) ==="
JAX_PLATFORMS=cpu $PY - "$NYF" <<'PYEOF'
import sys
from pathlib import Path

import numpy as np
from legoesm.ocean.forcing.core2 import load_core2_nyf

f = load_core2_nyf(cache_dir=Path(sys.argv[1]).parent, allow_synthetic=False)
lat = np.asarray(f.lat, dtype=np.float64)
lon = np.asarray(f.lon, dtype=np.float64) % 360
jy = np.where((lat >= -5) & (lat <= 5))[0]
ix = np.where((lon >= 210) & (lon <= 270))[0]
w = np.cos(np.deg2rad(lat[jy]))[:, None] * np.ones(len(ix))[None, :]
# Sample the WHOLE record axis, not one window: a cache assembled from two
# sources -- corrected for part of the year, raw for the rest -- would pass a
# 20-record probe while the run consumed the other variant for most of its
# integration.  Every sampled window has to land in the raw band.
n_rec = int(np.asarray(f.u10).shape[0])
stride = max(1, n_rec // 40)
windows = list(range(0, n_rec, stride))
sp = []
for k in windows:
    u = np.asarray(f.u10[k], dtype=np.float64)[np.ix_(jy, ix)]
    v = np.asarray(f.v10[k], dtype=np.float64)[np.ix_(jy, ix)]
    sp.append(float(np.nansum(np.hypot(u, v) * w) / np.nansum(w)))
sp = np.asarray(sp)
got = float(sp.mean())


def _variant(x):
    if 4.2 <= x <= 5.3:
        return "raw"
    if 5.9 <= x <= 6.7:
        return "corrected"
    return "unrecognised"


variants = sorted({_variant(x) for x in sp})
print(f"equatorial Pacific mean 10 m wind over {len(windows)} records spanning "
      f"the whole cache = {got:.3f} m/s (min {sp.min():.3f}, max {sp.max():.3f}) "
      f"-> {variants}")
if variants != ["raw"]:
    raise SystemExit(
        f"this arm is pre-registered on the RAW CORE-II winds, but the cache "
        f"reads as {variants} across its record axis (mean {got:.3f} m/s, "
        f"range {sp.min():.3f}-{sp.max():.3f}); the numbers below were "
        f"written for a wholly raw arm and are not comparable")
PYEOF
[ $? -eq 0 ] || { echo "FATAL: wind-variant assertion failed"; exit 1; }

$PY -u scripts/run/run_fesom_core2.py \
  --mesh-dir "$MI/mesh_core2" \
  --ic-dir "$MI/ic_core2" \
  --days 30 --dt 1800 --year 1958 \
  --forcing core2_nyf --nyf-zarr "$NYF" \
  --snapshot-every-days 10 \
  --output $REPO/results/omip_nemo/fesom_nyf_d30
rc=$?
echo "=== run rc=$rc ==="
[ $rc -eq 0 ] || exit $rc

echo "=== score: 3-way vs tripole + NEMO m1 (same protocol as xgrid_trp_fesom_d30) ==="
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
SNAP=$REPO/results/omip_nemo/fesom_nyf_d30/snapshot_day0030.npz
[ -f "$SNAP" ] || { echo "FATAL: no day-30 snapshot"; exit 1; }
$PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
  --tripole $REPO/results/omip_nemo/xgrid_trp_d30/snapshot_final.npz \
  --mpas "$SNAP" --label-mpas FESOM2_NYF \
  --nemo-gridt "$GT" --nemo-month 1 --res-deg 1.0 \
  --out-dir $REPO/results/omip_nemo/xgrid_trp_fesomnyf_d30 2>&1 | tail -40
echo "=== score rc=$?"
