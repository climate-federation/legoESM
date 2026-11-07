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

# Override with `sbatch --export=ALL,SNAP=<path>` to score an arm's snapshot;
# the default is the baseline every number in the header was measured on.
export SNAP=${SNAP:-/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_orca1faith_d30/snapshot_final.npz}
# Oracle record = the 5-day window ENDING at the snapshot day (0-based k covers
# days 5k..5k+5): snapshot_dayNNNN -> NNNN/5-1; snapshot_final is day 30 -> 5.
# Until 2026-09-15 this probe pinned record 5 whatever SNAP it was handed, so
# day-20/25 snapshots were paired with NEMO days 25-30.  Override: REC=<k>.
if [ -z "${REC:-}" ]; then
  case "$(basename "$SNAP")" in
    snapshot_day*.npz) d=$(basename "$SNAP" .npz); d=$((10#${d#snapshot_day}))
      [ $((d % 5)) -eq 0 ] && [ "$d" -ge 5 ] || { echo "FATAL: day $d is not a positive multiple of 5; no 5-day record ends there"; exit 1; }
      REC=$((d / 5 - 1)) ;;
    *) REC=5; echo "WARNING: no day in $(basename "$SNAP"); assuming day 30 = record 5 (pass REC=<k> otherwise)" ;;
  esac
fi
export REC
echo "SNAP: $SNAP  (oracle record $REC = days $((5*REC))-$((5*REC+5)))"

$PY - <<'PY'
import os
import numpy as np
import netCDF4 as nc

GW = ("/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/"
      "EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331")
SNAP = os.environ["SNAP"]
REC, LON0 = int(os.environ["REC"]), 220.0

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

# ---- GLM's unchecked warning: "reversed w smells like VERTICAL INDEXING" ---
# The 240E vertical-velocity reversal has been carried as a finding since
# 5e7161f66 (ours -14.8e-6 m/s against NEMO +18.9e-6). grid_W publishes `wo`
# in m/s -- the SAME units as our mass_flux_w -- so the two profiles can be
# compared directly on their own levels without any unit bridge. If our
# profile is NEMO's shifted by one level, or sign-flipped throughout, the
# "reversal" is an artifact of this probe family and the finding must be
# retracted rather than built on.
w_o = np.asarray(s["mass_flux_w"], np.float64)
dw2 = nc.Dataset(f"{GW}_grid_W.nc")
wn = np.asarray(np.ma.filled(np.ma.masked_invalid(
    dw2.variables["wo"][REC]), np.nan), np.float64)
wn = np.where(np.abs(wn) > 1e10, np.nan, wn)
dw2.close()
for _lon in (220.0, 240.0):
    LON_SAVE, LON0 = LON0, _lon
    wp_o = prof_ours(w_o, slat, slon)
    wp_n = prof_nemo(wn, wlat, wlon)
    print(f"\nEQUATORIAL w at {_lon:.0f}E, |lat|<=1 box mean [1e-6 m/s]; "
          f"+ = UPWARD. Same units on both sides (our mass_flux_w and NEMO's "
          f"wo are both m/s), each on its own levels.")
    print(f"{'z_ours':>8} {'w_ours':>10}   {'z_NEMO':>8} {'w_NEMO':>10}")
    for i in range(min(16, wp_o.size, wp_n.size)):
        zo_i = zi[i] if i < zi.size else np.nan
        print(f"{zo_i:8.1f} {1e6 * wp_o[i]:10.3f}   "
              f"{zw[i]:8.1f} {1e6 * wp_n[i]:10.3f}")
    _so = np.sign(wp_o[:16][np.isfinite(wp_o[:16])])
    _sn = np.sign(wp_n[:16][np.isfinite(wp_n[:16])])
    _n = min(_so.size, _sn.size)
    if _n:
        agree = float(np.mean(_so[:_n] == _sn[:_n]))
        print(f"[w] sign agreement over the top {_n} levels: {100 * agree:.0f}%"
              "  -- a systematic flip or a one-level offset here means the "
              "reversal is an INDEXING artifact, not physics.")
    LON0 = LON_SAVE
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

# ---- ARM SCORING BLOCK (a811a3e2f pre-registration, surface TKE at z=0) ----
# Scores whatever SNAP points at against the SAME oracle record. The numbers
# it prints are the ones the arm card registered on: K_M through 3-15 m as a
# fraction of NEMO's avm (CONFIRM >= 0.80, REFUTE < 0.15, PARTIAL between),
# OUR shear at 3.3 m (CONFIRM needs < 1.0e-02, from 2.37e-02), the TRIPWIRE
# (K_M > 2x avm anywhere in the top 10 m), and GLM's attribution confound --
# the shear PRODUCTION K*S^2 through 1-10 m on both sides, which separates
# "top-layer source added" from "10 m clamp removed". Verdict text is NOT
# printed by the probe; the rule is quoted and the reader applies it.
#
# ALIGNMENT, stated because the table above is offset on the NEMO side:
# avm_p[k] sits at zw[k] (zw[0] = 0 is the surface W-point), while
# dudz_n[i] = (u[i+1]-u[i])/(zu[i+1]-zu[i]) sits at zw[i+1]. The table prints
# avm_p[i] beside dudz_n[i], one W-level apart -- harmless for shape reading,
# wrong for a product. Here NEMO's production pairs avm_p[i+1] with
# dudz_n[i], both at zw[i+1]. Ours pairs KM_p[i] with dudz_o[i], both at
# zi[i] (z_interface_ref holds the 74 interior interfaces, zi[0] ~ 1 m).
if avm_p is not None:
    n_o = min(KM_p.size, dudz_o.size, zi.size)
    zo, K_o, S_o = zi[:n_o], KM_p[:n_o], dudz_o[:n_o]
    n_n = min(avm_p.size - 1, dudz_n.size)
    zn, K_n, S_n = zw[1:n_n + 1], avm_p[1:n_n + 1], dudz_n[:n_n]
    avm_at_zo = np.interp(zo, zw[:avm_p.size], avm_p)
    P_o, P_n = K_o * S_o ** 2, K_n * S_n ** 2
    P_n_at_zo = np.interp(zo, zn, P_n)

    print(f"\n[arm] SNAP = {SNAP}")
    print(f"[arm] TOP 15 m at {LON0:.0f}E, ours on zi vs NEMO interpolated "
          f"to the same depths; production P = K*(du/dz)^2 [m2/s3]")
    print(f"{'z':>6} {'K_M':>10} {'avm@z':>10} {'ratio':>7} {'S_ours':>10} "
          f"{'P_ours':>10} {'P_NEMO@z':>10}")
    top = zo <= 15.0
    for i in np.flatnonzero(top):
        r = K_o[i] / avm_at_zo[i] if avm_at_zo[i] > 0 else np.nan
        print(f"{zo[i]:6.1f} {K_o[i]:10.3e} {avm_at_zo[i]:10.3e} {r:7.3f} "
              f"{S_o[i]:10.3e} {P_o[i]:10.3e} {P_n_at_zo[i]:10.3e}")

    # Codex: np.interp carries a NaN into both neighbouring intervals and
    # nanmedian/nanmax would then score a band with holes as if it were
    # complete. Every scoring quantity is required to be finite over its
    # whole band, and the scoring is refused otherwise.
    score = zo <= 15.0
    need = (K_o[score], S_o[score], avm_at_zo[score])
    if not all(np.isfinite(x).all() for x in need):
        raise SystemExit("[arm] REFUSED: non-finite K_M, shear or "
                         "interpolated avm inside the 0-15 m scoring band")
    band = (zo >= 3.0) & (zo <= 15.0)
    ratio_med = float(np.median(K_o[band] / avm_at_zo[band]))
    i33 = int(np.argmin(np.abs(zo - 3.3)))
    s33_n = float(np.interp(zo[i33], zn, S_n))
    trip = float(np.max((K_o / avm_at_zo)[zo <= 10.0]))
    # Codex: integrating each side over the samples it happens to have inside
    # 1-10 m compares different intervals (NEMO's W-levels enclose 1.02-8.93 m).
    # Both profiles are interpolated onto ONE common grid and integrated there.
    zz = np.linspace(1.0, 10.0, 19)
    Pint_o = float(np.trapezoid(np.interp(zz, zo, P_o), zz))
    Pint_n = float(np.trapezoid(np.interp(zz, zn, P_n), zz))
    if not (np.isfinite(Pint_o) and np.isfinite(Pint_n)):
        raise SystemExit("[arm] REFUSED: non-finite production inside 1-10 m")
    print(f"\n[arm] K_M / avm, 3-15 m, median of per-depth ratios: "
          f"{ratio_med:.3f}   (baseline 0.05-0.06)")
    print(f"[arm] shear at {zo[i33]:.1f} m: ours {S_o[i33]:.3e}, NEMO at the "
          f"same depth {s33_n:.3e}   (baseline ours 2.371e-02)")
    print(f"[arm] TRIPWIRE max K_M/avm in the top 10 m: {trip:.3f}   "
          f"(> 2 means the surface VALUE was calibrated under the old "
          f"placement -- re-examine before adopting anything)")
    print(f"[arm] production K*S^2 integrated 1-10 m [m3/s3]: ours "
          f"{Pint_o:.3e}, NEMO {Pint_n:.3e}, ratio {Pint_o / Pint_n:.3f}")
    print("[arm] PRE-REGISTERED (a811a3e2f): CONFIRM = ratio >= 0.80 AND our "
          "3.3 m shear < 1.0e-02; REFUTE = ratio < 0.15; PARTIAL = anything "
          "else. The undercurrent is scored as a day-20 -> day-30 TREND by "
          "the EUC probe, never by a threshold here.")
    # GLM, on review: the production ratio CANNOT separate "clamp removed"
    # from "top-layer source added" -- K*S^2 was already equal on both sides
    # in the baseline by shear-equilibrium compensation, so ~1 is expected
    # under either reading and under no change at all. It is kept as a
    # disequilibrium indicator only. What discriminates is the SHAPE of the
    # TKE profile itself: a z=0 Dirichlet source decays monotonically from
    # the surface; a removed 10 m clamp erases an interior feature there.
    print("[arm] the production ratio does NOT discriminate the two readings "
          "(both predict ~1); the TKE profile shape below does.")
    if "tke" in s.files:
        e_p = prof_ours(np.asarray(s["tke"], np.float64), slat, slon)
        ze = zi if e_p.size == zi.size else (zc if e_p.size == zc.size else None)
        print(f"\n[arm] TKE profile, top 16 levels at {LON0:.0f}E [m2/s2] "
              f"(NEMO does not publish en in this stream; compare the arm "
              f"against the baseline run of this same probe)")
        for i in range(min(16, e_p.size)):
            zz_i = ze[i] if ze is not None else np.nan
            print(f"{zz_i:8.1f} {e_p[i]:10.3e}")
        top = e_p[:16]
        mono = bool(np.all(np.diff(top[np.isfinite(top)]) <= 0))
        print(f"[arm] TKE monotonically decreasing over the top 16 levels: "
              f"{mono}  -- an interior maximum near 10 m is the clamp's "
              f"signature, a surface maximum decaying downward is the "
              f"z=0 source's.")
    else:
        print("[arm] snapshot carries no 'tke' field; profile-shape check "
              "unavailable")
PY
echo DONE
