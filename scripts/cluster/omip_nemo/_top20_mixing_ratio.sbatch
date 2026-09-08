#!/bin/bash
#SBATCH --job-name=top20
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/top20_%j.out
#
# THE EMPTY CELL (GLM MAJOR-5): every diffusivity ratio measured so far was
# either momentum (5-65 m) or the entrainment band (65-105 m).  The claim that
# the cold-tongue bias is a top-20 m TRACER mixing deficit rests on a cell
# never measured.  This measures it, and two things that discriminate the
# alternatives in the same read:
#   * 0-20 m box-mean tracer diffusivity, ours vs NEMO, at the MATCHED local
#     hour and as a daily mean (NEMO's is hourly, ours is a 00Z snapshot);
#   (the 0-20 m heat-content comparison this card once printed was DELETED on
#    2026-09-08: misaligned layer thicknesses made its number meaningless);
#   * the shortwave absorbed within 0-20 m implied by each model's optics
#     (GLM CRITICAL-1: the warm-shallow/cold-deep signature fits too-shallow
#     absorption just as well as too-weak mixing).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
$PY - <<'PY2'
import numpy as np, netCDF4 as nc
OURS = ("/burg-archive/glab/users/pg2328/_evd_wt/results/omip_nemo/"
        "trp_evd_d30/snapshot_day0015.npz")
NEMO = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/"
        "EXP00/RUN_TRD2/ORCA1_1h_20010101_20010101_trd1h_T.nc")
LO, HI, HW = 220.0, 240.0, 2.0

z = np.load(OURS)
lat = np.asarray(z["lat_T"]); lon = np.asarray(z["lon_T"]) % 360.0
wet = np.asarray(z["land_mask"]) > 0.5
box = wet & (np.abs(lat) <= HW) & (lon >= LO) & (lon < HI)
zi = np.abs(np.asarray(z["z_interface_ref"]))          # interior interfaces
K = np.asarray(z["K_H_diag"])[box]                     # (ncol, nlev-1)
sel_i = zi <= 20.0
print(f"ours: {box.sum()} columns; {sel_i.sum()} interfaces within 20 m "
      f"({zi[sel_i][0]:.2f}..{zi[sel_i][-1]:.2f} m)")
k_ours = np.nanmedian(K[:, sel_i])
k_ours_mean = np.nanmean(K[:, sel_i])

d = nc.Dataset(NEMO)
nlat = d.variables["nav_lat_grid_T"][:]; nlon = d.variables["nav_lon_grid_T"][:] % 360.0
nbox = (np.abs(nlat) <= HW) & (nlon >= LO) & (nlon < HI)
depw = d.variables["depthw"][:]
avt = d.variables["avt"]
selw = depw <= 20.0
nt = avt.shape[0]
utc = np.arange(nt) + 0.5
loc = (utc + (LO + HI) / 2.0 / 15.0) % 24.0
k_nemo_h = []
for t in range(nt):
    a = np.ma.filled(avt[t][selw], np.nan)[:, nbox]
    k_nemo_h.append(np.nanmedian(a))
k_nemo_h = np.array(k_nemo_h)
aft = (loc >= 14) & (loc < 18)

print("\n=== 0-20 m TRACER diffusivity, cold tongue ===")
print(f"  ours   (00Z snapshot = mid-afternoon local): median {k_ours:.3e}  mean {k_ours_mean:.3e} m2/s")
print(f"  NEMO   daily mean {np.nanmean(k_nemo_h):.3e}   mid-afternoon {np.nanmean(k_nemo_h[aft]):.3e}"
      f"   night(18-06) {np.nanmean(k_nemo_h[(loc>=18)|(loc<6)]):.3e} m2/s")
print(f"  RATIO ours/NEMO  vs daily mean {k_ours/np.nanmean(k_nemo_h):.3f}"
      f"   vs matched afternoon {k_ours/np.nanmean(k_nemo_h[aft]):.3f}")
# 0-20 m column heat content DELETED 2026-09-08: the two sides integrated to
# different actual depths (our cell thicknesses vs NEMO's depthw ladder, sliced
# by two different counts) and both were divided by a nominal 20 m, so the
# printed +5.16 K was an instrument artefact. Never quote it. The diffusivity
# ratio above is what carried the measurement.
print("\n=== shortwave absorbed within 0-20 m, from each optics (GLM CRITICAL-1) ===")
for tag, frac_ir, e_ir, e_vis in (("ours/NEMO RGB", 0.58, 0.35, 23.0),):
    f20 = frac_ir * (1 - np.exp(-20.0 / e_ir)) + (1 - frac_ir) * (1 - np.exp(-20.0 / e_vis))
    f1 = frac_ir * (1 - np.exp(-1.08 / e_ir)) + (1 - frac_ir) * (1 - np.exp(-1.08 / e_vis))
    print(f"  {tag}: {100*f20:.1f}% absorbed above 20 m, {100*f1:.1f}% in the top 1.08 m cell")
print("  (identical optics on both sides means this cannot be the discriminator;")
print("   the ratio above is.)")
PY2
echo DONE
