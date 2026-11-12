#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=tattrs
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/tattrs_%j.out
#
# WHICH TEMPERATURE IS NEMO'S `to`?
#
# The new cross-section plotter differences our 3-D temperature against NEMO's
# `to`. Our ocean runs TEOS-10, i.e. CONSERVATIVE temperature. If `to` is
# POTENTIAL temperature the difference panel carries an offset that is a
# convention mismatch, not a model bias -- and a smooth, stationary interior
# difference shaped like the field itself is exactly what that looks like.
# Same question for salinity: practical (psu) vs absolute (g/kg).
#
# This reads ATTRIBUTES only -- standard_name, long_name, units -- so it is
# seconds of work and settles the question before any section is interpreted.
set -uo pipefail
cd /burg-archive/glab/users/pg2328/_omip_merge || exit 1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
$PY -u - "$GW" <<'PY'
import sys, xarray as xr
ds = xr.open_dataset(sys.argv[1], decode_times=False)
for v in ("to", "tos", "tob", "so", "sos", "sob", "deptht"):
    if v not in ds.variables:
        print(f"{v:8s} ABSENT"); continue
    a = ds[v].attrs
    print(f"{v:8s} dims={ds[v].dims} units={a.get('units')!r} "
          f"standard_name={a.get('standard_name')!r} long_name={a.get('long_name')!r}")
PY
