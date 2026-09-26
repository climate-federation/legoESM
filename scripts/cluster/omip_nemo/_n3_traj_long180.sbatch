#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=n3traj
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:15:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/n3traj180_%j.out
# nino3 SST trajectory from long180's existing snapshots (plateau-vs-drift).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
$PY - <<'PY'
import glob, numpy as np
out = "/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/nemolev_trp_long180"
prev = None
for d in range(15, 181, 15):
    snap = f"{out}/snapshot_day{d:04d}.npz"
    if not glob.os.path.exists(snap):
        continue
    z = np.load(snap)
    lat = z["lat_T"]; lon = z["lon_T"] % 360
    T0 = z["T"][..., 0]; wet = z["land_mask"] > 0.5
    box = (np.abs(lat) <= 5) & (lon >= 210) & (lon <= 270) & wet
    area = z["cell_area"] if "cell_area" in z.files else np.cos(np.deg2rad(lat))
    w = area[box]; v = T0[box]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    sst = float((v[ok] * w[ok]).sum() / w[ok].sum())
    dd = "" if prev is None else f"  d{sst - prev:+.3f}"
    print(f"day {d:3d}  nino3 SST {sst:.3f} C{dd}")
    prev = sst
PY
echo DONE
