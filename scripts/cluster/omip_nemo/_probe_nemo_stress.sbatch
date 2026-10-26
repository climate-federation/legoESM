#!/bin/bash
#SBATCH --job-name=nemotau
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/nemotau_%j.out
#
# WHAT STRESS DOES THE ORACLE ACTUALLY PUBLISH, AND WHAT IS ITS EQUATORIAL
# ZONAL PROFILE?
#
# GLM ranked the applied surface stress as the top remaining candidate for the
# equatorial defect and named the cheapest discriminator: equatorial tauu(x),
# 160-260E, ours against NEMO's SBC. Before designing anything around that, the
# oracle side has to exist: our only committed SBC reader
# (frozen_column_tke_twin.py:load_sbc_taum) reads the MODULUS taum/taum_oce,
# never a zonal component, so whether NEMO writes utau/sozotaux/tauuo here is
# an open question and netCDF4 is not importable on the login node.
#
# This lists every 2-D field in the SBC file and, if a zonal stress is present,
# prints its equatorial profile. It measures ONLY the oracle side -- our own
# applied stress is not stored in any snapshot and needs a driver diagnostic,
# which is a code change and will be dual-reviewed separately. Half a
# comparison is worth having first because it costs minutes and decides
# whether the other half is worth building.
#
# NO PRE-REGISTERED THRESHOLD HERE ON PURPOSE: this is an inventory plus one
# profile, not a test. GLM's registered thresholds (CONFIRM a bug at a tau_x
# ratio <= 0.80 or >= 1.25 over >= 8 continuous degrees within 180-280E, i.e.
# |delta tau| >= 0.015 N/m2 on ~0.08 N/m2 trades; REFUTE at a pointwise ratio
# inside [0.88, 1.18]) apply to the COMPARISON, once our side exists. The band
# is deliberately wide because a relative-wind difference alone forces ~12%.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

$PY - <<'PY'
import numpy as np
import netCDF4 as nc

SBC = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/"
       "EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_SBC.nc")
d = nc.Dataset(SBC)
print("SBC 2-D time-varying fields (name: units -- long_name):")
for v, o in sorted(d.variables.items()):
    if o.ndim == 3:
        print(f"  {v:16s} {getattr(o, 'units', '?'):12s} "
              f"{getattr(o, 'long_name', '')[:60]}")

# utau_ao is the one this stream actually carries -- "surface_downward_x_stress"
# over air-ocean. The first version of this list guessed six other spellings and
# found none of them, which is exactly why the inventory above is printed before
# the lookup: the file names the variable, not my memory.
# WHICH STRESS FIELDS ACTUALLY CARRY DATA. utau_ao is declared with the right
# long_name and is ENTIRELY EMPTY at record 5 -- finite fraction 0.0000 while
# the coordinates are perfectly good (12888 cells inside |lat|<=1). A variable
# existing in the header is not a variable carrying data, so every candidate is
# screened for finiteness before any of them is believed.
print("\nfiniteness of every stress-like field at record 5:")
for v in ("utau_ao", "vtau_ao", "taum", "taum_oce", "windsp"):
    if v in d.variables:
        _a = np.asarray(np.ma.filled(np.ma.masked_invalid(
            d.variables[v][5]), np.nan), dtype=np.float64)
        _a = np.where(np.abs(_a) > 1e10, np.nan, _a)
        _ff = float(np.mean(np.isfinite(_a)))
        _rng = (f"{np.nanmin(_a):+.4f}..{np.nanmax(_a):+.4f}"
                if _ff > 0 else "empty")
        print(f"  {v:10s} finite {_ff:.4f}  {_rng}")

CANDS = ("utau_ao", "utau", "sozotaux", "tauuo", "tauu", "utau_oce", "taux")
name = next((v for v in CANDS if v in d.variables), None)
print(f"\nzonal-stress candidates {CANDS}")
print(f"found: {name!r}")

if name is None:
    print("\nNo zonal stress in this file. GLM's tauu(x) design needs either a")
    print("different NEMO output stream or a reconstruction from taum and the")
    print("wind direction -- do NOT silently substitute the MODULUS taum, which")
    print("is sign-blind and cannot show an excess WESTWARD stress.")
else:
    a = np.ma.filled(np.ma.masked_invalid(d.variables[name][:]), np.nan)
    a = np.asarray(a, dtype=np.float64)
    print(f"shape {a.shape}; record 5 of {a.shape[0]} is day 30")
    f = np.where(np.abs(a[5]) > 1e10, np.nan, a[5])
    lat = np.asarray(d.variables["nav_lat"][:], dtype=np.float64)
    lon = np.asarray(d.variables["nav_lon"][:], dtype=np.float64) % 360.0
    # The first run of this returned NaN at every longitude. Diagnose rather
    # than guess: this stream is XIOS output with the halo dropped (331,360)
    # against the mesh's (332,362), and nav_lat is known on this thread to
    # carry -1 fill in fully-masked rows, which silently destroys a
    # coordinate-box selection.
    print(f"  field finite fraction: {np.mean(np.isfinite(f)):.4f}"
          f"   min {np.nanmin(f):+.5f}  max {np.nanmax(f):+.5f}")
    print(f"  nav_lat range {np.nanmin(lat):+.2f}..{np.nanmax(lat):+.2f}"
          f"   nav_lon range {np.nanmin(lon):.2f}..{np.nanmax(lon):.2f}")
    _band = np.abs(lat) <= 1.0
    print(f"  cells with |lat|<=1: {int(_band.sum())}"
          f"   of which finite stress: {int(np.isfinite(f[_band]).sum())}")
    print(f"\nNEMO equatorial zonal wind stress, |lat|<=1 box mean [N/m2];"
          f" negative = WESTWARD (trades).")
    print(f"{'lon':>6} {'tau_x':>10}")
    for lon0 in (160, 180, 200, 220, 240, 260, 280):
        m = ((np.abs(lat - 0.0) <= 1.0)
             & (np.abs((lon - lon0 + 180.0) % 360.0 - 180.0) <= 2.0))
        print(f"{lon0:6d} {np.nanmean(f[m]) if m.any() else np.nan:10.5f}")
    print("\nOURS IS NOT MEASURED HERE. The driver computes sf.tau_x from")
    print("compute_omip2_surface_forcing and never prints it, and no snapshot")
    print("stores it, so the comparison needs a one-step instrumented run.")
d.close()
PY
echo DONE
