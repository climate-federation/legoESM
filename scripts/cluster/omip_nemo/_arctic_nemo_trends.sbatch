#!/bin/bash
#SBATCH --job-name=arctrd
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/arctrd_%j.out
#
# WHAT NEMO'S OWN OPERATORS DO TO THE ARCTIC HALOCLINE.
# Five switch-level candidates were refuted tonight (mixing-length choice,
# advection limiter, surface TKE BC, convection, IC), each on a plausible
# mechanism.  Stop guessing levers: read the oracle's per-process tendencies.
#
# OUR MEASURED EROSION RATE, for comparison: the Arctic surface salinifies
# +0.42 psu in 5 days = 9.7e-7 psu/s.  Whatever process erodes our halocline
# must run at about that rate.  NEMO's trends say which of ITS operators is
# active there, and at what magnitude.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -u - <<'PYEOF'
import sys
from pathlib import Path
import numpy as np
import netCDF4 as nc

sys.path.insert(0, "scripts/validate/ocean_fidelity")
from global_tracer_content import (load_mesh_metrics, load_mesh_latitude,
                                   load_mesh_depth_1d)

MESH = "data/grids/eORCA1.2_mesh_mask.nc"
F = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/"
     "RUN_TRD2/ORCA1_1h_20010101_20010101_trd1h_T.nc")

e1t, e2t, e3t, tmask = load_mesh_metrics(MESH)
lat = load_mesh_latitude(MESH)
gdept = load_mesh_depth_1d(MESH)
wet = tmask > 0.5
arctic = (lat >= 60.0) & wet.any(axis=0)
dV = e1t[None] * e2t[None] * e3t * (wet & arctic[None])

ds = nc.Dataset(F)
terms = [v for v in ("strd_zdf", "strd_evd", "strd_totad", "strd_ldf",
                     "strd_bbl", "strd_tot") if v in ds.variables]
print(f"NEMO Arctic (>60N) mean salinity tendency [psu/s], 24-h mean, "
      f"top 20 m vs 20-60 m -- RMS and mean|.| (signed means CANCEL)")
print(f"{'term':10s} {'top20_rms':>12s} {'top20_abs':>12s} "
      f"{'mid_rms':>12s} {'mid_abs':>12s}")
top = (gdept <= 20.0)
mid = (gdept > 20.0) & (gdept <= 60.0)
for t in terms:
    a = np.ma.filled(np.ma.masked_invalid(ds.variables[t][:]), 0.0)
    a = a.mean(axis=0).astype(np.float64) * 1.0e-3     # units 1e-3/s -> psu/s
    if np.abs(a).max() > 1.0:
        print(f"{t:10s} SKIP (fill contamination, max {np.abs(a).max():.2e})")
        continue
    # RMS and mean|.|, NOT the signed mean: tendencies alternate sign across
    # the basin, so a volume-weighted signed mean over 6225 columns measures
    # CANCELLATION, not magnitude (the first version of this probe reported
    # 1e-11 for terms that are locally ~1e-6).
    out = []
    for sel in (top, mid):
        w = dV * sel[:, None, None]
        if w.sum() <= 0:
            out += [np.nan, np.nan]
            continue
        out.append(float(np.sqrt((a * a * w).sum() / w.sum())))
        out.append(float((np.abs(a) * w).sum() / w.sum()))
    print(f"{t:10s} {out[0]:12.3e} {out[1]:12.3e} {out[2]:12.3e} {out[3]:12.3e}")
ds.close()
print()
print("OUR measured erosion: surface +0.42 psu / 5 d = +9.7e-07 psu/s")
PYEOF
echo "=== rc=$? ==="
