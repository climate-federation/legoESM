#!/bin/bash
#SBATCH --job-name=lakes
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/lakes_%j.out
#
# WHAT DO THE TWO MODELS DO WITH THE GREAT LAKES?
#
# User, 2026-09-14: "we always have an issue in the US great lakes SSS". On
# our side the lakes are 69 wet cells (41.5-48.9N, 269-283E) carrying OCEAN
# salinity, 30.3-32.8 psu at day 20 of the z0 arm (measured from the snapshot
# on the login node). This probe asks the oracle side, which needs netCDF4
# and therefore a compute node:
#   * are those cells wet in NEMO (finite `so` in grid_T at the scored record)?
#   * if wet, what salinity does NEMO carry there, and does its domain_cfg
#     flag them as a DEFINED closed sea (ORCA1 runs the namclo defaults:
#     ln_maskcs=F, ln_mask_csundef=T -> defined closed seas stay wet with
#     their net E-P spread to the ocean, undefined ones are masked)?
#   * what does the SSS-restoring climatology hold over the lakes -- that is
#     what our restoring pulls them toward.
# INVENTORY FIRST: every candidate mask name in domain_cfg.nc is listed
# before any is read, because guessing names has failed on this thread.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY - <<'PY'
import numpy as np
import netCDF4 as nc
E = "/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00"
GT = f"{E}/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc"
# RUN_GATEWAY/domain_cfg.nc and RUN_DEFAULT/domain_cfg.nc are DANGLING symlinks
# (they point at ../RUN_REF and at another user's home); the real file is the
# one in the input repository.
DC = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/"
      "orca1_inputs/data_repository/input_fields/domain_cfg.nc")
SSSR = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/"
        "orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc")
SNAPS = {
    "z0_arm day20": "/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1z0_d30/snapshot_day0020.npz",
    "baseline day30": "/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1faith_d30/snapshot_final.npz",
}
def box(lat, lon):
    lon = lon % 360.0
    return (lat >= 41) & (lat <= 49.5) & (lon >= 267) & (lon <= 285)

d = nc.Dataset(GT)
lat = np.asarray(d.variables["nav_lat"][:], np.float64)
lon = np.asarray(d.variables["nav_lon"][:], np.float64)
sname = "so" if "so" in d.variables else next(v for v in d.variables if v.startswith("so"))
so = np.asarray(np.ma.filled(np.ma.masked_invalid(d.variables[sname][4, 0]), np.nan), np.float64)
so = np.where(np.abs(so) > 1e10, np.nan, so)
d.close()
b = box(lat, lon)
wet = b & np.isfinite(so) & (so != 0.0)
print(f"NEMO grid_T '{sname}' record 4 (day 20): Great Lakes box has {int(wet.sum())} wet cells")
if wet.any():
    print(f"  NEMO SSS there: min {np.nanmin(so[wet]):.3f} median {np.nanmedian(so[wet]):.3f} max {np.nanmax(so[wet]):.3f}")
    print(f"  lat {lat[wet].min():.1f}-{lat[wet].max():.1f} lon {(lon[wet]%360).min():.1f}-{(lon[wet]%360).max():.1f}")

dc = nc.Dataset(DC)
print("\ndomain_cfg.nc variables (inventory):", sorted(dc.variables))
for v in sorted(dc.variables):
    if "mask" in v.lower() or "clo" in v.lower() or "cs" in v.lower():
        a = np.asarray(dc.variables[v][:]).squeeze()
        if a.ndim == 2 and a.shape == b.shape:
            print(f"  {v:18s} unique {np.unique(a)[:8]}  in-box nonzero {int((a[b] != 0).sum())}")
if "top_level" in dc.variables:
    tl = np.asarray(dc.variables["top_level"][:]).squeeze()
    print(f"  top_level in box: wet(>0) {int((tl[b] > 0).sum())}")
if "bottom_level" in dc.variables:
    bl = np.asarray(dc.variables["bottom_level"][:]).squeeze()
    print(f"  bottom_level in box: wet(>0) {int((bl[b] > 0).sum())}, max {int(bl[b].max())}")
# GLOBAL wet-domain mismatch between our mask and the oracle's domain_cfg:
# where are ALL the cells that are wet for one model and dry for the other?
tl = np.asarray(dc.variables["top_level"][:]).squeeze()
nemo_wet = tl > 0
snap = np.load(SNAPS["z0_arm day20"])
ours_wet = np.asarray(snap["land_mask"]) > 0.5
slat = np.asarray(snap["lat_T"], np.float64); slon = np.asarray(snap["lon_T"], np.float64) % 360.0
if ours_wet.shape != nemo_wet.shape:
    # Our mesh frame (332, 362) carries the periodic halo columns i=0 and
    # i=361 and the north-fold row j=331; NEMO's domain_cfg / XIOS frame
    # (331, 360) is the inner domain. Map ours -> inner and VERIFY the map by
    # the coordinates themselves before comparing masks.
    gl = np.asarray(dc.variables["gphit"][:]).squeeze(); gm = np.asarray(dc.variables["glamt"][:]).squeeze() % 360.0
    sl, sm = slat[:331, 1:361], slon[:331, 1:361]
    dlat = float(np.nanmax(np.abs(sl - gl))); dlon = float(np.nanmax(np.abs(((sm - gm) + 180.0) % 360.0 - 180.0)))
    print(f"\nframe map ours[:331,1:361] -> domain_cfg: max |dlat| {dlat:.2e}, max |dlon| {dlon:.2e} (must be ~0)")
    if dlat < 1e-3 and dlon < 1e-3:
        ours_wet = ours_wet[:331, 1:361]; slat = sl; slon = sm
if ours_wet.shape == nemo_wet.shape:
    a = ours_wet & ~nemo_wet
    c = ~ours_wet & nemo_wet
    print(f"\nWET-DOMAIN MISMATCH vs domain_cfg top_level: wet-ours/dry-NEMO {int(a.sum())}, dry-ours/wet-NEMO {int(c.sum())}")
    for name, m in (("wet-ours/dry-NEMO", a), ("dry-ours/wet-NEMO", c)):
        if not m.any():
            continue
        la, lo = slat[m], slon[m]
        # crude clustering by 10-degree boxes, largest first
        keys = {}
        for y, x in zip(la, lo):
            k = (int(np.floor(y / 10.0)) * 10, int(np.floor(x / 10.0)) * 10)
            keys[k] = keys.get(k, 0) + 1
        top = sorted(keys.items(), key=lambda kv: -kv[1])[:8]
        print(f"  {name}: 10-degree boxes (lat0, lon0): count -> " + ", ".join(f"({k[0]},{k[1]}):{n}" for k, n in top))
else:
    print(f"\nshape mismatch ours {ours_wet.shape} vs domain_cfg {nemo_wet.shape}; no global comparison")
dc.close()

# The mesh OUR tripole grid is built from (the card's --mesh-mask): the lakes
# are wet for us because they are wet HERE.
MM = "/burg-archive/glab/users/pg2328/legoESM/data/grids/eORCA1.2_mesh_mask.nc"
mm = nc.Dataset(MM)
print("\nour mesh_mask variables (inventory):", sorted(mm.variables))
for v in ("tmask", "tmaskutil", "top_level", "bottom_level"):
    if v in mm.variables:
        a = np.asarray(mm.variables[v][:]).squeeze()
        a2 = a[0] if a.ndim == 3 else a
        if a2.shape == b.shape:
            print(f"  {v:12s} in-box wet(>0): {int((a2[b] > 0).sum())}")
mm.close()

try:
    r = nc.Dataset(SSSR)
    print("\nSSS restoring file variables:", sorted(r.variables))
    vn = next(v for v in r.variables if r.variables[v].ndim >= 3)
    a = np.asarray(np.ma.filled(np.ma.masked_invalid(r.variables[vn][0]), np.nan), np.float64).squeeze()
    if a.ndim == 3:
        a = a[0]
    rl = np.asarray(r.variables["nav_lat"][:] if "nav_lat" in r.variables else r.variables["lat"][:], np.float64)
    ro = np.asarray(r.variables["nav_lon"][:] if "nav_lon" in r.variables else r.variables["lon"][:], np.float64)
    if rl.ndim == 1:
        rl, ro = np.meshgrid(rl, ro, indexing="ij")
    rb = box(rl, ro)
    v = a[rb]
    v = v[np.isfinite(v) & (v != 0)]
    print(f"  '{vn}' over the box: n={v.size} min {v.min():.3f} median {np.median(v):.3f} max {v.max():.3f}")
    r.close()
except Exception as e:  # inventory-first: report, do not guess
    print("SSS restoring file not readable:", e)

for lab, p in SNAPS.items():
    s = np.load(p)
    sb = box(np.asarray(s["lat_T"], np.float64), np.asarray(s["lon_T"], np.float64))
    w = sb & (np.asarray(s["land_mask"]) > 0.5)
    S = np.asarray(s["S"], np.float64)[..., 0]
    print(f"\nours {lab}: {int(w.sum())} wet cells in the box, SSS min {S[w].min():.3f} median {np.median(S[w]):.3f} max {S[w].max():.3f}")
PY
echo DONE
