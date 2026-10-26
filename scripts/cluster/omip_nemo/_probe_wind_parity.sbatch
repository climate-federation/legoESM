#!/bin/bash
#SBATCH --job-name=windpar
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=0:25:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/windpar_%j.out
#
# DO BOTH MODELS SEE THE SAME EQUATORIAL WIND?
#
# The applied surface stress is the top-ranked remaining candidate for our
# equatorial defect, and it is the one input never verified: correct forcing
# PROVENANCE establishes the wind SOURCE, not the stress the ocean actually
# receives. GLM's cheapest discriminator was ours-vs-NEMO tau_x -- but the
# oracle's vector stress turned out to be declared and never written
# (utau_ao/vtau_ao finite fraction 0.0000, measured, c62fd5497), so that test
# is not runnable.
#
# This is the test one step EARLIER in the chain, and it needs no model run at
# all -- both sides are files already on disk:
#   * NEMO's own `windsp` from the GATEWAY SBC stream, which IS written
#     (finite fraction 0.5462 = the ocean fraction, range 1.16-23.42 m/s);
#   * the CORE-II wind we both read, sqrt(U_10_MOD^2 + V_10_MOD^2).
#
# WHAT IT SETTLES. Our card remaps forcing with NEMO's OWN SCRIP weights
# (--forcing-remap nemo_scrip), so if NEMO's applied wind matches the raw
# CORE-II field, then both models see the same wind and any stress difference
# is purely the DRAG COEFFICIENT -- which narrows the top candidate to one line
# of physics without a GPU-second. If NEMO's applied wind is materially
# different from the source, that difference is itself the forcing defect and
# points at relative-wind (rn_vfac) or the remap.
#
# PRE-REGISTERED, before looking, because a small difference is EXPECTED and
# would otherwise be over-read: a previously recorded measurement on this
# thread put NEMO's nino3 windsp at 6.064 against U_10_MOD's 6.279, a 3.4%
# gap, attributed to bicubic remapping plus relative wind. So:
#   * |gap| <= 5%  -> the wind is COMMON. Stress differences are the drag
#     coefficient or the relative-wind term, not the wind field.
#   * |gap| >= 15% -> a real forcing difference; the wind itself is the defect
#     and the stress ranking is confirmed at its source.
#   * 5-15%        -> report the number, claim no direction.
#
# MATCHED WINDOW, and it is the thing most likely to go wrong here: SBC record
# 5 is the 5-day mean over days 26-30, so the CORE-II side must average its
# 6-hourly records 100-119 inclusive (4 per day), NOT a single instant. An
# unmatched window would manufacture a gap out of weather.
#
# CAVEAT STATED UP FRONT: NEMO's `windsp` is the modulus of whatever wind NEMO
# applied, which may already include the ocean-current feedback. The CORE-II
# side is the absolute wind. If rn_vfac is 0 in ORCA1 they are comparable; if
# not, part of any gap is that term by construction, not an error.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

$PY - <<'PY'
import numpy as np
import netCDF4 as nc

IN = "/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS"
SBC = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/"
       "EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_SBC.nc")

# ---- NEMO's applied wind speed, record 5 = the day 26-30 mean -------------
d = nc.Dataset(SBC)
w = np.asarray(np.ma.filled(np.ma.masked_invalid(
    d.variables["windsp"][5]), np.nan), dtype=np.float64)
w = np.where(np.abs(w) > 1e10, np.nan, w)
nlat = np.asarray(d.variables["nav_lat"][:], dtype=np.float64)
nlon = np.asarray(d.variables["nav_lon"][:], dtype=np.float64) % 360.0
d.close()
print(f"NEMO windsp record 5: finite {np.mean(np.isfinite(w)):.4f}, "
      f"range {np.nanmin(w):.3f}..{np.nanmax(w):.3f} m/s")

# ---- the CORE-II wind we both read ---------------------------------------
du = nc.Dataset(f"{IN}/u_10.15JUNE2009.nc")
dv = nc.Dataset(f"{IN}/v_10.15JUNE2009.nc")
uname = "U_10_MOD" if "U_10_MOD" in du.variables else "U_10"
vname = "V_10_MOD" if "V_10_MOD" in dv.variables else "V_10"
print(f"CORE-II variables: {uname}, {vname}  "
      f"(available: {sorted(v for v in du.variables if v.startswith('U_'))})")

# SBC record 5 is the 5-day mean over days 26-30; CORE-II is 6-hourly, so the
# matched slice is records 100-119 inclusive. Averaging the SPEED, not the
# components, because the modulus of a mean wind is not the mean of the speed
# and NEMO's windsp is a mean OF the applied speed.
R0, R1 = 100, 120
u = np.asarray(du.variables[uname][R0:R1], dtype=np.float64)
v = np.asarray(dv.variables[vname][R0:R1], dtype=np.float64)
clat = np.asarray(du.variables["LAT"][:] if "LAT" in du.variables
                  else du.variables["lat"][:], dtype=np.float64)
clon = np.asarray(du.variables["LON"][:] if "LON" in du.variables
                  else du.variables["lon"][:], dtype=np.float64) % 360.0
du.close(); dv.close()
spd = np.sqrt(u * u + v * v).mean(axis=0)
print(f"CORE-II speed, records {R0}-{R1 - 1} (days 26-30): "
      f"shape {spd.shape}, range {spd.min():.3f}..{spd.max():.3f} m/s")

if clat.ndim == 1:
    clat2 = clat[:, None] * np.ones((1, clon.size))
    clon2 = np.ones((clat.size, 1)) * clon[None, :]
else:
    clat2, clon2 = clat, clon

def box(f, la, lo, lon0):
    m = ((np.abs(la) <= 1.0)
         & (np.abs((lo - lon0 + 180.0) % 360.0 - 180.0) <= 2.0))
    return float(np.nanmean(f[m])) if m.any() else float("nan")

print("\nEQUATORIAL WIND SPEED, |lat|<=1 box mean [m/s], matched days 26-30")
print(f"{'lon':>6} {'NEMO':>9} {'CORE-II':>9} {'gap %':>8}")
gaps = []
for lon0 in (160, 180, 200, 220, 240, 260, 280):
    a = box(w, nlat, nlon, lon0)
    b = box(spd, clat2, clon2, lon0)
    g = 100.0 * (a - b) / b if np.isfinite(b) and b != 0 else float("nan")
    gaps.append(g)
    print(f"{lon0:6d} {a:9.3f} {b:9.3f} {g:+8.2f}")

gm = float(np.nanmean(np.abs(np.asarray(gaps, dtype=np.float64))))
print(f"\n[wind] mean |gap| across 160-280E: {gm:.2f}%")
print("[wind] PRE-REGISTERED: <=5% means the wind is COMMON to both models, "
      "so a stress difference would be the DRAG COEFFICIENT or the "
      "relative-wind term, not the wind field. >=15% means the wind itself "
      "is the forcing defect. Between, report and claim no direction.")
print("[wind] CAVEAT: NEMO's windsp is the modulus of the wind NEMO APPLIED "
      "and may already carry the ocean-current feedback; the CORE-II side is "
      "the absolute wind. Part of any gap is that term by construction if "
      "ORCA1 runs rn_vfac /= 0.")
PY
echo DONE
