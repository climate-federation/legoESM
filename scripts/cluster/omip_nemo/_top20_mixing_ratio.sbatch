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
export OURS_SNAP NEMO_FILE NEMO_RECS
$PY - <<'PY2'
import numpy as np, netCDF4 as nc
import os
# codex CRITICAL 2026-09-08: this was hard-coded to the EVD ARM's day-15 state
# while its output was being read as the BASELINE at day 30. Both the snapshot
# and the oracle file are now inputs, and the run prints which it used.
OURS = os.environ.get("OURS_SNAP", "/burg-archive/glab/users/pg2328/legoESM/"
                      "results/omip_nemo/trp_base2_d30/snapshot_day0030.npz")
NEMO = os.environ.get("NEMO_FILE", "/burg-archive/glab/users/pg2328/nemo_orca1/"
                      "nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2/"
                      "ORCA1_1h_20010101_20010101_trd1h_T.nc")
NEMO_RECS = os.environ.get("NEMO_RECS", "")   # "" = every record
print(f"[inputs] ours  {OURS}")
print(f"[inputs] NEMO  {NEMO}  records {NEMO_RECS or 'all'}")
LO, HI, HW = 220.0, 240.0, 2.0
# GLM's mechanism (2026-09-08) says the momentum-mixing gap that matters for the
# undercurrent is the DEEP one, not the 0-20 m one; measure both bands.
BANDS = ((0.0, 20.0), (20.0, 60.0), (60.0, 160.0))

z = np.load(OURS)
lat = np.asarray(z["lat_T"]); lon = np.asarray(z["lon_T"]) % 360.0
wet = np.asarray(z["land_mask"]) > 0.5
box = wet & (np.abs(lat) <= HW) & (lon >= LO) & (lon < HI)
zi = np.abs(np.asarray(z["z_interface_ref"]))          # interior interfaces
K = np.asarray(z["K_H_diag"])[box]                     # (ncol, nlev-1)
KM = np.asarray(z["K_M_diag"])[box]                    # momentum, same shape
sel_i = zi <= 20.0
print(f"ours: {box.sum()} columns; {sel_i.sum()} interfaces within 20 m "
      f"({zi[sel_i][0]:.2f}..{zi[sel_i][-1]:.2f} m)")
k_ours = np.nanmedian(K[:, sel_i])
k_ours_mean = np.nanmean(K[:, sel_i])
m_ours = np.nanmedian(KM[:, sel_i])

d = nc.Dataset(NEMO)
_la = "nav_lat_grid_T" if "nav_lat_grid_T" in d.variables else "nav_lat"
_lo = "nav_lon_grid_T" if "nav_lon_grid_T" in d.variables else "nav_lon"
nlat = d.variables[_la][:]; nlon = d.variables[_lo][:] % 360.0
nbox = (np.abs(nlat) <= HW) & (nlon >= LO) & (nlon < HI)
depw = d.variables["depthw"][:]
avt = d.variables["avt"]
avm = d.variables["avm"] if "avm" in d.variables else None
selw = depw <= 20.0
_recs = ([int(x) for x in NEMO_RECS.split(",")] if NEMO_RECS
         else list(range(avt.shape[0])))
nt = len(_recs)
utc = np.arange(nt) + 0.5
loc = (utc + (LO + HI) / 2.0 / 15.0) % 24.0
k_nemo_h, m_nemo_h = [], []
for _i, t in enumerate(_recs):
    a = np.ma.filled(avt[t][selw], np.nan)[:, nbox]
    k_nemo_h.append(np.nanmedian(a))
    if avm is not None:
        m_nemo_h.append(np.nanmedian(np.ma.filled(avm[t][selw], np.nan)[:, nbox]))
k_nemo_h = np.array(k_nemo_h); m_nemo_h = np.array(m_nemo_h)
# The local-hour split is meaningless unless the file actually resolves a day.
DIURNAL = nt >= 12
aft = (loc >= 14) & (loc < 18) if DIURNAL else np.ones(nt, bool)
if not DIURNAL:
    print(f"[note] {nt} record(s): no diurnal resolution, so every 'matched hour' "
          "column below repeats the record mean -- read the daily column only.")

print("\n=== 0-20 m TRACER diffusivity, cold tongue ===")
print(f"  ours   (00Z snapshot = mid-afternoon local): median {k_ours:.3e}  mean {k_ours_mean:.3e} m2/s")
print(f"  NEMO   daily mean {np.nanmean(k_nemo_h):.3e}   mid-afternoon {np.nanmean(k_nemo_h[aft]):.3e}"
      f"   night(18-06) {np.nanmean(k_nemo_h[(loc>=18)|(loc<6)]):.3e} m2/s")
print(f"  RATIO ours/NEMO  vs daily mean {k_ours/np.nanmean(k_nemo_h):.3f}"
      f"   vs matched afternoon {k_ours/np.nanmean(k_nemo_h[aft]):.3f}")
# codex CRITICAL-2 (2026-09-08): a matched TRACER diffusivity says nothing about
# the MOMENTUM one, which is what sets the undercurrent. Same box, same hours.
print("\n=== 0-20 m MOMENTUM diffusivity, cold tongue (codex CRITICAL-2) ===")
if m_nemo_h.size:
    print(f"  ours median {m_ours:.3e} m2/s")
    print(f"  NEMO daily mean {np.nanmean(m_nemo_h):.3e}   mid-afternoon {np.nanmean(m_nemo_h[aft]):.3e} m2/s")
    print(f"  RATIO ours/NEMO  vs daily mean {m_ours/np.nanmean(m_nemo_h):.3f}"
          f"   vs matched afternoon {m_ours/np.nanmean(m_nemo_h[aft]):.3f}")
    print(f"  Prandtl (momentum/tracer)  ours {m_ours/k_ours:.3f}   NEMO {np.nanmean(m_nemo_h)/np.nanmean(k_nemo_h):.3f}")
else:
    print("  avm ABSENT from the NEMO hourly file -- momentum mixing UNTESTED, say so")

# 0-20 m column heat content DELETED 2026-09-08: the two sides integrated to
# different actual depths (our cell thicknesses vs NEMO's depthw ladder, sliced
# by two different counts) and both were divided by a nominal 20 m, so the
# printed +5.16 K was an instrument artefact. Never quote it. The diffusivity
# ratio above is what carried the measurement.
print("\n=== band-by-band diffusivity, ours vs NEMO daily mean (GLM depth test) ===")
for b0, b1 in BANDS:
    si = (zi >= b0) & (zi < b1)
    sn = (depw >= b0) & (depw < b1)
    if not si.any() or not sn.any():
        continue
    ko = np.nanmedian(K[:, si]); mo = np.nanmedian(KM[:, si])
    kn = np.nanmedian([np.nanmedian(np.ma.filled(avt[t][sn], np.nan)[:, nbox]) for t in range(nt)])
    mn = (np.nanmedian([np.nanmedian(np.ma.filled(avm[t][sn], np.nan)[:, nbox]) for t in range(nt)])
          if avm is not None else np.nan)
    print(f"  {b0:5.0f}-{b1:5.0f} m : tracer ours {ko:.3e} NEMO {kn:.3e} ratio {ko/kn:6.3f} | "
          f"momentum ours {mo:.3e} NEMO {mn:.3e} ratio {mo/mn:6.3f} | "
          f"Prandtl ours {mo/ko:5.2f} NEMO {mn/kn:5.2f}")

# If our turbulence simply STOPS shallower than the oracle's, that shows up as a
# turbulent-layer depth, and as a mixed-layer depth from temperature alone --
# the second is independent of any diffusivity convention.
KTURB = 1.0e-4
def _depth_of_last(mask, zlev):
    out = np.full(mask.shape[0], np.nan)
    for i in range(mask.shape[0]):
        idx = np.flatnonzero(mask[i])
        if idx.size:
            out[i] = zlev[idx[-1]]
    return out
turb_o = _depth_of_last(K > KTURB, zi)
Tcol = np.asarray(z["T"])[box]
zc_o = np.abs(np.asarray(z["z_center_ref"]))
def _mld(Tc, zlev, dT=0.2):
    out = np.full(Tc.shape[0], np.nan)
    for i in range(Tc.shape[0]):
        ref = Tc[i, 0]
        idx = np.flatnonzero(np.abs(Tc[i] - ref) > dT)
        out[i] = zlev[idx[0]] if idx.size else zlev[-1]
    return out
mld_o = _mld(Tcol, zc_o)
turb_n, mld_n = [], []
for _i, t in enumerate(_recs):
    a_ = np.ma.filled(avt[t], np.nan)[:, nbox].T
    turb_n.append(np.nanmedian(_depth_of_last(a_ > KTURB, depw)))
_tv = next(v for v in ("votemper", "to", "thetao") if v in d.variables)
tt = np.ma.filled(d.variables[_tv][_recs[0]], np.nan)[:, nbox].T
mld_n = _mld(tt, np.asarray(d.variables["deptht"][:]))
print("\n=== how deep does the turbulence reach? (cold tongue) ===")
print(f"  turbulent layer (K > {KTURB:g} m2/s): ours median {np.nanmedian(turb_o):6.1f} m"
      f"   NEMO median {np.nanmedian(turb_n):6.1f} m")
print(f"  mixed layer (|T - T_surf| > 0.2 K):   ours median {np.nanmedian(mld_o):6.1f} m"
      f"   NEMO median {np.nanmedian(mld_n):6.1f} m")
print("  (the second line uses temperature only, so it does not depend on any"
      " diffusivity convention)")

print("\n=== shortwave absorbed within 0-20 m, from each optics (GLM CRITICAL-1) ===")
for tag, frac_ir, e_ir, e_vis in (("ours/NEMO RGB", 0.58, 0.35, 23.0),):
    f20 = frac_ir * (1 - np.exp(-20.0 / e_ir)) + (1 - frac_ir) * (1 - np.exp(-20.0 / e_vis))
    f1 = frac_ir * (1 - np.exp(-1.08 / e_ir)) + (1 - frac_ir) * (1 - np.exp(-1.08 / e_vis))
    print(f"  {tag}: {100*f20:.1f}% absorbed above 20 m, {100*f1:.1f}% in the top 1.08 m cell")
print("  (identical optics on both sides means this cannot be the discriminator;")
print("   the ratio above is.)")
PY2
echo DONE
