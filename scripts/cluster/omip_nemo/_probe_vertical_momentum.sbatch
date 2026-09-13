#!/bin/bash
#SBATCH --job-name=vmom
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=0:25:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/vmom_%j.out
#
# DOES THE WIND STRESS REACH THE THERMOCLINE?
#
# GLM's replacement for a continuity budget I was about to build, and its
# reasoning is the part worth keeping: a budget both models satisfy BY
# CONSTRUCTION, assembled from the velocity field already known to be wrong,
# has zero discriminating power -- it says where the water goes, never why.
# Only relations tying INDEPENDENT fields to the velocity can break in one
# model alone.
#
# The forcing is now fully exonerated by measurement: same wind source, same
# wind field (0.05% at 220E), relative wind off on both sides, and drag
# matching to 1.3% -- with our drag slightly LOWER, so we apply slightly LESS
# stress while the surface current runs 97% too strong. GLM reads that tetrad
# -- over-westward surface jet, weak or absent undercurrent, steeper sea-level
# tilt, 1.6x upwelling -- as textbook STRESS FAILING TO REACH THE THERMOCLINE.
#
# So this compares the two ingredients of the vertical stress divergence at the
# defect peak, ours against the oracle's:
#   * the vertical viscosity: our K_M_diag against NEMO's own avm;
#   * the shear du/dz, each model on its own levels.
# Comparing OUR K_M against NEMO'S avm is not circular -- the oracle's
# viscosity is an independent field, not something derived from our velocity.
# (Deriving d/dz(K_M du/dz) from our own K_M to argue about our own K_M would
# be, which is GLM's earlier objection and is respected here.)
#
# INVENTORY FIRST, then the lookup. Guessing variable names has failed twice on
# this thread -- six spellings of the wind stress found nothing while the
# file's own field list named utau_ao immediately, and utau_ao then turned out
# to be declared and entirely empty. A variable in the header is not a variable
# carrying data, so every candidate is screened for finiteness before use.
#
# PRE-REGISTERED, before looking. The surface current is +97% too strong and
# the undercurrent is 39% of NEMO's at this longitude:
#   * if our K_M through 20-80 m is a factor >=3 BELOW NEMO's avm, the stress
#     is not being carried down and "stress fails to reach the thermocline" is
#     CONFIRMED as the mechanism;
#   * if our K_M matches NEMO's avm within a factor of 1.5 through that range,
#     the mechanism is REFUTED and the shear column becomes the question;
#   * between, report the numbers and claim no direction.
# Read the PROFILE, not a single level: the last two probes were decided by
# structure while a scalar landed indeterminate.
#
# NOTE, against GLM's own advice: it suggested comparing our K_M floor to a
# NEMO background of "O(1e-4)". That is the NAMELIST value. We MEASURED the
# resolved floors as matched -- ln_zdfiwm=.true. makes zdfiwm.F90:378 override
# rn_avm0 with avmb = rnu = 1.4e-6, which is our value too. The measured floor
# is used here, not the recalled one.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1

$PY - <<'PY'
import numpy as np
import netCDF4 as nc

GW = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/"
      "EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331")
SNAP = ("/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/"
        "trp_orca1faith_d30/snapshot_final.npz")
REC, LON0 = 5, 220.0

dw = nc.Dataset(f"{GW}_grid_W.nc")
print("grid_W 3-D fields (name: units -- long_name):")
for v, o in sorted(dw.variables.items()):
    if o.ndim == 4:
        print(f"  {v:14s} {getattr(o, 'units', '?'):10s} "
              f"{getattr(o, 'long_name', '')[:55]}")

def finite_screen(ds, names):
    """A variable in the header is not a variable carrying data."""
    out = {}
    for v in names:
        if v not in ds.variables:
            continue
        a = np.asarray(np.ma.filled(np.ma.masked_invalid(
            ds.variables[v][REC]), np.nan), np.float64)
        a = np.where(np.abs(a) > 1e10, np.nan, a)
        ff = float(np.mean(np.isfinite(a)))
        print(f"  {v:14s} finite {ff:.4f}"
              + (f"  range {np.nanmin(a):.3e}..{np.nanmax(a):.3e}"
                 if ff > 0 else "  EMPTY"))
        if ff > 0:
            out[v] = a
    return out

print("\nfiniteness of viscosity/diffusivity candidates at record 5:")
cand = finite_screen(dw, ("avm", "avt", "avm_k", "avt_k", "difvmo", "difvho",
                          "avm_oce", "avt_oce"))
zw = np.asarray(dw.variables["depthw"][:], np.float64)
wlat = np.asarray(dw.variables["nav_lat"][:], np.float64)
wlon = np.asarray(dw.variables["nav_lon"][:], np.float64) % 360.0
dw.close()

du = nc.Dataset(f"{GW}_grid_U.nc")
uo = np.asarray(np.ma.filled(np.ma.masked_invalid(
    du.variables["uo"][REC] if "uo" in du.variables
    else du.variables["uocetr_eff"][REC]), np.nan), np.float64)
uo = np.where(np.abs(uo) > 1e10, np.nan, uo)
zu = np.asarray(du.variables["depthu"][:], np.float64)
ulat = np.asarray(du.variables["nav_lat"][:], np.float64)
ulon = np.asarray(du.variables["nav_lon"][:], np.float64) % 360.0
du.close()

s = np.load(SNAP)
KM = np.asarray(s["K_M_diag"], np.float64)
u_o = np.asarray(s["u"], np.float64)
if u_o.shape[1] == np.asarray(s["lat_T"]).shape[1] + 1:
    u_o = 0.5 * (u_o[:, :-1] + u_o[:, 1:])      # C-grid faces -> T points
slat = np.asarray(s["lat_T"], np.float64)
slon = np.asarray(s["lon_T"], np.float64) % 360.0
zc = np.abs(np.asarray(s["z_center_ref"], np.float64))
zi = np.abs(np.asarray(s["z_interface_ref"], np.float64))

# AXIS ORDER IS PART OF THE API AND THE TWO SIDES DISAGREE. Our snapshot is
# (ny, nx, nz) -- depth LAST. NEMO's 4-D fields at one record are (nz, ny, nx)
# -- depth FIRST. The first version used one helper for both and died with
# "size of axis is 75 but size of corresponding boolean axis is 331".
def _box(la, lo):
    return ((np.abs(la) <= 1.0)
            & (np.abs((lo - LON0 + 180.0) % 360.0 - 180.0) <= 2.0))

def prof_ours(f3, la, lo):                      # (ny, nx, nz)
    m = _box(la, lo)
    if not m.any():
        return np.full(f3.shape[-1], np.nan)
    return np.array([np.nanmean(f3[..., k][m]) for k in range(f3.shape[-1])])

def prof_nemo(f3, la, lo):                      # (nz, ny, nx)
    m = _box(la, lo)
    if not m.any():
        return np.full(f3.shape[0], np.nan)
    return np.array([np.nanmean(f3[k][m]) for k in range(f3.shape[0])])

KM_p = prof_ours(KM, slat, slon)
u_p = prof_ours(u_o, slat, slon)
un_p = prof_nemo(uo, ulat, ulon)
avm_name = next((v for v in ("avm", "avm_k", "difvmo", "avm_oce")
                 if v in cand), None)
avm_p = prof_nemo(cand[avm_name], wlat, wlon) if avm_name else None
print(f"\nusing NEMO viscosity variable: {avm_name!r}")

dudz_o = np.diff(u_p) / np.diff(zc)
dudz_n = np.diff(un_p) / np.diff(zu)

print(f"\nVERTICAL MOMENTUM INGREDIENTS at {LON0:.0f}E, |lat|<=1 box mean.")
print("Each model on ITS OWN levels -- read the profile shape, not row pairs.")
print(f"{'z_ours':>8} {'K_M':>10} {'du/dz':>10}   "
      f"{'z_NEMO':>8} {'avm':>10} {'du/dz':>10}")
for i in range(min(22, dudz_o.size)):
    lhs = f"{zi[i] if i < zi.size else np.nan:8.1f} {KM_p[i]:10.3e} {dudz_o[i]:10.3e}"
    if avm_p is not None and i < avm_p.size - 1:
        rhs = f"{zw[i]:8.1f} {avm_p[i]:10.3e} {dudz_n[i]:10.3e}"
    else:
        rhs = f"{zw[i] if i < zw.size else np.nan:8.1f} {'--':>10} {dudz_n[i] if i < dudz_n.size else np.nan:10.3e}"
    print(f"{lhs}   {rhs}")

if avm_p is not None:
    band_o = (zi[:KM_p.size] >= 20.0) & (zi[:KM_p.size] <= 80.0)
    band_n = (zw[:avm_p.size] >= 20.0) & (zw[:avm_p.size] <= 80.0)
    a = float(np.nanmedian(KM_p[:len(band_o)][band_o]))
    b = float(np.nanmedian(avm_p[:len(band_n)][band_n]))
    print(f"\n[vmom] 20-80 m median: ours K_M {a:.3e}, NEMO avm {b:.3e}, "
          f"ratio ours/NEMO {a / b if b else float('nan'):.3f}")
    print("[vmom] PRE-REGISTERED: ours a factor >=3 BELOW NEMO CONFIRMS "
          "'stress fails to reach the thermocline'; within a factor of 1.5 "
          "REFUTES it and the shear column becomes the question; between, "
          "report and claim no direction.")
else:
    print("\n[vmom] NEMO publishes no usable viscosity in grid_W -- the "
          "oracle side of this comparison does not exist in this stream, "
          "exactly as its vector wind stress did not. Do NOT substitute a "
          "diffusivity for a viscosity; they differ by the Prandtl ratio.")
PY
echo DONE
