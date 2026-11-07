#!/bin/bash
#SBATCH --job-name=nemodiur
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nemodiur_%j.out
#
# THE DISCRIMINATOR the whole lens claim has been missing: NEMO's OWN diurnal
# cycle of the cold-tongue near-surface stratification, from its hourly output
# (RUN_TRD2, 24 hourly records of votemper).  Our snapshots are taken at 00Z =
# mid-afternoon local time in the cold tongue, while every NEMO number quoted
# so far is a 5-day MEAN.  If NEMO's own mid-afternoon T(0.5 m) - T(20 m) is
# close to its daily mean (~0.3 K), the lens we carry is real and the EVD arm's
# null refutes convection as its cause.  If NEMO's afternoon value is itself a
# few K, then a large part of "the lens" was a sampling artefact all along and
# the EVD null is exactly what a faithful port should produce.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY - <<'PY2'
import numpy as np, netCDF4 as nc
P = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/"
     "RUN_TRD2/ORCA1_1h_20010101_20010101_trd1h_T.nc")
d = nc.Dataset(P)
lat = d.variables["nav_lat_grid_T"][:]; lon = d.variables["nav_lon_grid_T"][:] % 360.0
dep = d.variables["deptht"][:]
T = d.variables["votemper"]
k05 = int(np.argmin(np.abs(dep - 0.5))); k20 = int(np.argmin(np.abs(dep - 20.0)))
print(f"{P.split('/')[-1]}  {T.shape[0]} hourly records; using depths "
      f"{dep[k05]:.2f} m and {dep[k20]:.2f} m")
for lo, hi in ((220, 240), (240, 260)):
    box = (np.abs(lat) <= 2.0) & (lon >= lo) & (lon < hi)
    rows = []
    for t in range(T.shape[0]):
        a = np.ma.filled(T[t, k05], np.nan)[box]
        b = np.ma.filled(T[t, k20], np.nan)[box]
        ok = np.isfinite(a) & np.isfinite(b) & (a > -5)
        rows.append(np.nanmean(a[ok]) - np.nanmean(b[ok]))
    rows = np.array(rows)
    utc = np.arange(T.shape[0]) + 0.5
    loc = (utc + (lo + hi) / 2.0 / 15.0) % 24.0
    order = np.argsort(loc)
    print(f"\n=== NEMO {lo}-{hi}E |lat|<=2: T(0.5 m) - T(20 m) through the day ===")
    print("  local hour: " + " ".join(f"{h:5.1f}" for h in loc[order]))
    print("  lens   [K]: " + " ".join(f"{v:5.2f}" for v in rows[order]))
    aft = (loc >= 14) & (loc < 18)
    print(f"  daily mean {rows.mean():.2f} K;  mid-afternoon (14-18 local) "
          f"{rows[aft].mean():.2f} K;  min {rows.min():.2f}  max {rows.max():.2f}")
    print("  OUR 00Z snapshots sample the mid-afternoon value; the 5-day-mean "
          "numbers quoted so far (0.31 at d15, 0.76 at d30) are the daily mean.")
PY2
echo DONE
