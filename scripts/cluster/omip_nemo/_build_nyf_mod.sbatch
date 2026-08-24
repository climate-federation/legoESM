#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=nyfmod
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nyfmod_%j.out

# Rebuild the CORE-II cache from the scatterometer-corrected winds NEMO reads.
# Writes to a NEW path: three arms are running against the existing cache and
# must stay reproducible, so the old nyf.zarr is left untouched.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
OUT=/burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/core2_nyf_mod/nyf.zarr
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS

echo "=== old cache left in place: $(du -sh /burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing/core2_nyf 2>/dev/null | cut -f1) ==="
$PY -u scripts/data/build_core2_nyf_zarr.py --inputs-dir "$NEMOIN" \
    --out "$OUT" --wind-variant mod 2>&1 | tail -30
rc=$?
echo "=== build rc=$rc"
[ $rc -eq 0 ] || exit 1

echo "=== verify: the new cache must carry the CORRECTED winds ==="
$PY - <<'PYEOF'
import numpy as np
from pathlib import Path
from legoesm.ocean.forcing.core2 import load_core2_nyf
f = load_core2_nyf(cache_dir=Path('/burg-archive/home/pg2328/.cache/legoesm/'
                                  'ocean_fidelity/forcing/core2_nyf_mod'),
                   allow_synthetic=False)
lat = np.asarray(f.lat, dtype=np.float64); lon = np.asarray(f.lon, dtype=np.float64) % 360
jy = np.where((lat >= -5) & (lat <= 5))[0]; ix = np.where((lon >= 210) & (lon <= 270))[0]
w = np.cos(np.deg2rad(lat[jy]))[:, None] * np.ones(len(ix))[None, :]
sp = []
for k in range(100, 120):
    u = np.asarray(f.u10[k], dtype=np.float64)[np.ix_(jy, ix)]
    v = np.asarray(f.v10[k], dtype=np.float64)[np.ix_(jy, ix)]
    sp.append(float(np.nansum(np.hypot(u, v) * w) / np.nansum(w)))
got = float(np.mean(sp))
print(f"nino3 mean |U10| in the NEW cache = {got:.3f} m/s")
print("  expected ~6.28 (corrected)   old cache was 4.736 (raw)")
if not (6.0 <= got <= 6.6):
    raise SystemExit(f"FAILED: {got:.3f} is not the corrected wind; the swap "
                     "did not take effect")
print("OK: the cache carries the corrected winds")
PYEOF
echo "=== verify rc=$? ==="
