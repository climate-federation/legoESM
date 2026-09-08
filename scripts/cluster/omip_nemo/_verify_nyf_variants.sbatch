#!/bin/bash
#SBATCH --job-name=nyfver
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nyfver_%j.out
#
# THE SAME STATISTIC ON ALL THREE CACHES.
#
# The winds-only build's gate compared its numbers against values measured
# elsewhere with a DIFFERENT reduction -- the campaign's 6.28 m/s came from
# CORE-II records 100-119 on the source grid, and 165.4 W/m2 was an unweighted
# mean over 365 daily records -- so it flagged a cache that may be correct.
# Comparing across reductions is the error the gate was supposed to prevent.
#
# This computes ONE statistic per channel, the same way, on all three caches.
# The winds-only cache is correct if and only if its wind matches `mod` and
# every other channel matches `base`.
set -uo pipefail
R=/burg-archive/glab/users/pg2328/legoESM; cd $R || exit 1
export PYTHONPATH="$(ls -d $R/packages/*/ | sed 's:/$::' | tr '\n' ':')$R"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY - <<'PYEOF'
import sys
import numpy as np
import xarray as xr

ROOT = "/burg-archive/home/pg2328/.cache/legoesm/ocean_fidelity/forcing"
CACHES = {"base": "core2_nyf", "mod": "core2_nyf_mod",
          "windsonly": "core2_nyf_windsonly"}
CH = ("u10", "v10", "T_air", "q_air", "sw_down", "lw_down", "precip", "snow")

def stats(path):
    ds = xr.open_zarr(f"{ROOT}/{path}/nyf.zarr")
    lat = np.asarray(ds.lat.values, dtype=np.float64)
    w = np.cos(np.deg2rad(lat))[None, :, None]
    out = {}
    for c in CH:
        if c not in ds:
            out[c] = float("nan"); continue
        a = np.asarray(ds[c].values, dtype=np.float64)
        out[c] = float((a * w).sum() / (np.ones_like(a) * w).sum())
    a = np.asarray(ds.u10.values, dtype=np.float64)
    b = np.asarray(ds.v10.values, dtype=np.float64)
    out["speed"] = float((np.hypot(a, b) * w).sum()
                         / (np.ones_like(a) * w).sum())
    return out

S = {k: stats(v) for k, v in CACHES.items()}
print(f"{'channel':10s}{'base':>14}{'mod':>14}{'windsonly':>14}")
for c in (*CH, "speed"):
    print(f"{c:10s}{S['base'][c]:>14.6g}{S['mod'][c]:>14.6g}"
          f"{S['windsonly'][c]:>14.6g}")

def close(a, b):
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))

bad = []
for c in ("u10", "v10", "speed"):
    if not close(S["windsonly"][c], S["mod"][c]):
        bad.append(f"{c}: windsonly {S['windsonly'][c]:.6g} != mod "
                   f"{S['mod'][c]:.6g} -- the wind is not the corrected one")
for c in ("T_air", "q_air", "sw_down", "lw_down", "precip", "snow"):
    if not close(S["windsonly"][c], S["base"][c]):
        bad.append(f"{c}: windsonly {S['windsonly'][c]:.6g} != base "
                   f"{S['base'][c]:.6g} -- a non-wind channel was corrected")
# A cache identical to `base` everywhere would pass the second loop and prove
# nothing, so require the wind to actually differ from base.
if close(S["windsonly"]["speed"], S["base"]["speed"]):
    bad.append("wind speed equals base: this cache changes nothing")

if bad:
    print("\nFATAL:")
    for b in bad:
        print("  " + b)
    sys.exit(1)
print("\nOK: winds-only cache = mod winds + base everything else")
PYEOF
echo "=== verify rc=$? ==="
echo DONE
