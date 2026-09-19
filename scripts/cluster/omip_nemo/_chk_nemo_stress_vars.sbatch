#!/bin/bash
#SBATCH --job-name=chkSBC
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/chkSBC_%j.out
#
# WHERE DOES NEMO KEEP ITS SURFACE STRESS, AND WHAT ARE THE UNITS REALLY?
#
# Identical CORE-II forcing is the PREMISE of this whole comparison, so a
# stress difference would be a harmonization defect rather than a physics
# result. Before comparing anything, find out what exists.
#
# Units are printed but NOT trusted. This repo's NEMO output has already lied
# twice: grid_U's uoce_e3u is labelled m/s while its long_name says
# thickness-weighted, and grid_T's to/so hold Conservative Temperature and
# Absolute Salinity while claiming potential and practical. Magnitude is the
# check that actually discriminates -- a wind stress is O(0.1) Pa and nothing
# else in these files looks like that.
#
# SBC and both velocity files are scanned, because NEMO writes utau/vtau to
# grid_U/grid_V in some configurations and to SBC in others, and guessing
# which would be exactly the "infer what data exists from the one file a
# script happens to open" mistake.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge
cd $ROOT || exit 1
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cpu
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
G=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY

$PY -u - "$G" <<'PY'
import sys
from pathlib import Path
import numpy as np
import netCDF4 as nc

g = Path(sys.argv[1])
for fn in ("ORCA1_5d_20000101_20000331_SBC.nc",
           "ORCA1_5d_20000101_20000331_grid_U.nc",
           "ORCA1_5d_20000101_20000331_grid_V.nc"):
    p = g / fn
    if not p.is_file():
        print(f"=== {fn}: ABSENT")
        continue
    ds = nc.Dataset(p)
    print(f"\n=== {fn}  ({len(ds.dimensions['time_counter'])} records)")
    for n, v in ds.variables.items():
        if v.ndim != 3:          # 2-D field plus time
            continue
        a = np.ma.masked_invalid(v[5] if v.shape[0] > 5 else v[0])
        cnt = int(a.count())
        mx = float(np.abs(a).max()) if cnt else float("nan")
        mean = float(np.abs(a).mean()) if cnt else float("nan")
        flag = "  <-- stress-like magnitude" if 1e-3 < mean < 2.0 else ""
        print(f"  {n:16s} units={getattr(v, 'units', '?'):10s} "
              f"|max|={mx:11.4g} mean|x|={mean:11.4g}  "
              f"{getattr(v, 'long_name', '?')[:40]}{flag}")
    ds.close()
PY
echo "=== rc=$? ==="
