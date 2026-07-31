"""Is the DINO ACC deficit a BUOYANCY problem? -- thermal-wind decomposition.

The ACC is thermal-wind: a weaker meridional density gradient supports a weaker
baroclinic transport.  This script measures, for legoESM and NEMO on the SAME
grid with the SAME metric and the SAME window:

  1. the actual ACC transport (compare_fullframe.py metric, verbatim) and the
     thermal-wind-implied BAROCLINIC transport from the density field alone
     (bottom-referenced), plus the barotropic residual -> the split;
  2. the channel density/T/S meridional contrast vs depth + isopycnal slope;
  3. the year-by-year contrast, split above/below 1400 m;
  4. the T-vs-S decomposition of the density gradient (alpha/beta split).

Diagnostic only: reads recorded fields, imports the production EOS, writes
nothing to packages/ or src/.

WEIGHTING / PROTOCOL (identical for both models -- nothing below branches on
which model is being evaluated):
  * frame           : full 199(y) x 52(x) x 36(z) DINO frame, no halo strip
                      (haloless NEMO 5 output; matches compare_fullframe.py).
  * geometry        : mesh_mask REFERENCE geometry (e3t_0, gdept_0 -- partial
                      cells) for the thermal wind and all vertical weighting;
                      e3t_1d for the recorded ACC metric, because that is what
                      the recorded 65.5 / 91.1 Sv numbers used.  legoESM is
                      z-star and NEMO's diagnostic thicknesses are time-mean,
                      but BOTH models are evaluated on the same reference
                      geometry, so the ratio is a controlled comparison.
  * wet mask        : tmask(mesh_mask) AND legoESM land_mask -- the SAME 3-D
                      mask applied to both models.
  * channel band    : T-rows where all 52 longitudes are wet, i.e. the
                      re-entrant (zonally unblocked) latitudes.  Measured from
                      the mask, not hardcoded: j = 14..48, 64.4 S .. 45.4 S.
                      Everything north/south of it has land walls, so it is
                      the only band that can carry a net zonal throughflow.
  * horizontal avg  : per-longitude sections, then MEDIAN over i=2..-2 --
                      the recorded ACC metric's reducer, reused for every
                      transport number so all of them are comparable.
  * density         : in-situ, production EOS legoesm.ocean.eos.nemo_seos_eos
                      with the DINO S-EOS coefficients from RUN_5Y/namelist_cfg
                      (a0=0.165, b0=0.76554, lambda1=0.06, mu1=1.497e-4,
                      lambda2=mu2=nu=0), depth = gdept_0.  Same call for both.
  * NEMO T,S        : native-iom artifact -- votemper/vosaline are toce*e3t
                      UNDIVIDED; divided by the co-written vovvle3t, as in
                      dino_year5_compare.py.

Run:  .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/acc_thermal_wind.py
"""
import os
import sys

import netCDF4 as nc
import numpy as np

from legoesm import constants
from legoesm.ocean.eos import NemoSEOSConfig, nemo_seos_alpha_beta, nemo_seos_eos

CFG = NemoSEOSConfig()          # defaults ARE the DINO/Kamm set (verified vs namelist_cfg)
RHO0 = CFG.rho0
DINO = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO"
LEGO_DIR = os.environ.get(
    "DINO_TW_LEGO_DIR",
    "/tmp/claude-10257/-home-dbalwada-legoESM/853ee94c-2651-44cc-ba12-f55bf3ed1979/scratchpad",
)
DEEP_M = 1400.0                 # the reported upper/deep split depth

# ---------------------------------------------------------------- geometry ---
mm = nc.Dataset(f"{DINO}/RUN_TRAJ/mesh_mask.nc")


def llz(a):                     # (lev,y,x) -> (y,x,lev)
    return np.moveaxis(np.asarray(a).squeeze(), 0, -1)


tmask = llz(mm["tmask"][0]) > 0.5
umask = llz(mm["umask"][0]) > 0.5
e3t0 = llz(mm["e3t_0"][0])                        # (y,x,z) partial-cell thickness [m]
gdept0 = llz(mm["gdept_0"][0])                    # (y,x,z) T-point depth [m, +down]
e3t1d = np.asarray(mm["e3t_1d"][:]).squeeze()     # (z,) reference thickness [m]
gdept1d = np.asarray(mm["gdept_1d"][:]).squeeze()
e2u_col = np.asarray(mm["e2u"][0]).squeeze()[:, 25]   # (y,) -- compare_fullframe.py's choice
gphit = np.asarray(mm["gphit"][0]).squeeze()
gphiv = np.asarray(mm["gphiv"][0]).squeeze()
e2v = np.asarray(mm["e2v"][0]).squeeze()
NY, NX, NZ = tmask.shape

# f at v-points [1/s].  Band is 45-64 S, nowhere near the equator, so no guard.
f_v = 2.0 * constants.Omega * np.sin(np.deg2rad(gphiv))

# Channel band = T-rows with all NX longitudes wet at the surface (re-entrant).
_wetcols = tmask[:, :, 0].sum(axis=1)
_band = np.where(_wetcols == NX)[0]
J0, J1 = int(_band.min()), int(_band.max())       # inclusive T-row band
assert np.all(np.diff(_band) == 1), "channel band is not contiguous"
L_BAND = float(e2v[J0:J1, 25].sum())              # band meridional extent [m]


# ------------------------------------------------------------------ loaders ---
def load_nemo(gridT, tidx=0, gridU=None):
    """Physical T,S (+u) on the full frame.  Divides out the iom e3t artifact."""
    g = nc.Dataset(gridT)
    e3d = llz(g["vovvle3t"][tidx])
    inv = 1.0 / np.maximum(e3d, 1e-6)
    out = {"T": llz(g["votemper"][tidx]) * inv, "S": llz(g["vosaline"][tidx]) * inv}
    if gridU is not None:
        out["u"] = llz(nc.Dataset(gridU)["vozocrtx"][tidx])
    return out


def load_lego(path):
    d = np.load(path)
    lU = d["u"]                                    # (199,53,36) u-faces
    u = lU[:, 1:53, :].copy()                      # faces 1..52 <-> NEMO u-cols
    u[:, 47, :] = lU[:, 48, :]                     # (compare_fullframe.py, verbatim)
    return {"T": d["T"], "S": d["S"], "u": u, "land_mask": d["land_mask"]}


# ------------------------------------------------------------------ metrics ---
def acc_full(u, wet_u):
    """Recorded ACC metric (compare_fullframe.py): full-section zonal transport,
    e3t_1d weighting, median over longitudes 2..-2 [Sv]."""
    return float(np.median(np.einsum("jik,k,j->i", np.where(wet_u, u, 0.0), e3t1d, e2u_col)[2:-2]) / 1e6)


def acc_band(u, wet_u, e3=None):
    """Same integral restricted to the channel band [Sv per longitude].  Default
    weighting is the real partial-cell e3t_0 (the geometry the thermal wind uses);
    pass e3=e3t_1d to isolate the recorded metric's reference-thickness weighting."""
    e3 = e3t0[J0:J1 + 1] if e3 is None else np.broadcast_to(e3, u[J0:J1 + 1].shape)
    w = np.where(wet_u[J0:J1 + 1], u[J0:J1 + 1] * e3, 0.0)
    return np.einsum("jik,j->i", w, e2u_col[J0:J1 + 1]) / 1e6


def bc_bt_band(u, wet_u):
    """EXACT split of the band transport into a bottom-referenced BAROCLINIC part
    and a BAROTROPIC (reference-level) part [Sv per longitude, each].

    u(z) = u_bot + (u(z) - u_bot), with u_bot the deepest wet cell-centre value.
    The second term is what bottom-referenced thermal wind predicts; the first is
    the reference-level transport it cannot see.  bc + bt == acc_band identically.
    """
    us, ws, e3s = u[J0:J1 + 1], wet_u[J0:J1 + 1], e3t0[J0:J1 + 1]
    col = ws.any(axis=2)
    kbot = np.where(col, ws.sum(axis=2) - 1, 0)                     # deepest wet level
    ubot = np.where(col, np.take_along_axis(us, kbot[:, :, None], axis=2)[:, :, 0], 0.0)
    H = np.sum(np.where(ws, e3s, 0.0), axis=2)
    e2 = e2u_col[J0:J1 + 1]
    bt = np.einsum("ji,ji,j->i", ubot, H, e2) / 1e6
    bc = np.einsum("jik,j->i", np.where(ws, (us - ubot[:, :, None]) * e3s, 0.0), e2) / 1e6
    return bc, bt


def _shear_integrand(drho, wet_pair, e3v, dpv, fv):
    """Bottom-referenced baroclinic transport integrand, per (v-row, lon, lev).

    u_tw(z) = int_{-H}^{z} (g/(rho0 f)) drho/dy dz'  (z UP, f<0 in the SH, rho
    decreasing northward => u_tw > 0 eastward at the surface: the ACC sign).
    Transport = int dy int dz u_tw = int dy int dz' shear(z')*|z'|, so the
    depth |z| = gdept is the vertical weight and the meridional integral of
    d/dy telescopes -- e2v cancels exactly against the 1/e2v in the gradient.
    """
    return np.where(wet_pair, (constants.g / (RHO0 * fv)) * drho * dpv * e3v, 0.0)


def per_lev(integ):
    """Per-level Sv per longitude: meridional sum, then the ACC metric's median
    over longitudes 2..-2 (NOT a zonal sum -- that would be NX times too large)."""
    return np.median(integ.sum(axis=0)[2:-2], axis=0) / 1e6


def thermal_wind(rho, wet):
    """Bottom-referenced baroclinic transport [Sv per longitude] + the per-level
    integrand (for the depth-attribution report).

    Note the half-cell x-offset: rho gradients sit on T-columns, u on u-faces.
    Immaterial to a median over 48 longitudes, and identical for both models.
    """
    a, b = slice(J0, J1), slice(J0 + 1, J1 + 1)     # v-point pairs (j, j+1)
    integ = _shear_integrand(
        rho[b] - rho[a],
        wet[a] & wet[b],
        0.5 * (e3t0[a] + e3t0[b]),
        0.5 * (gdept0[a] + gdept0[b]),
        f_v[a, :][:, :, None],
    )
    return integ.sum(axis=(0, 2)) / 1e6, integ


def ts_split(T, S, wet):
    """Thermal wind from the T-only and S-only parts of the density gradient.

    drho = drho/dT * dT + drho/dS * dS, derivatives at the v-point midpoint via
    the production alpha/beta (drho/dT = -rho0*alpha, drho/dS = +rho0*beta).
    """
    a, b = slice(J0, J1), slice(J0 + 1, J1 + 1)
    al, be = nemo_seos_alpha_beta(
        0.5 * (T[a] + T[b]), 0.5 * (S[a] + S[b]), 0.5 * (gdept0[a] + gdept0[b]), CFG
    )
    al, be = np.asarray(al), np.asarray(be)
    common = (wet[a] & wet[b], 0.5 * (e3t0[a] + e3t0[b]),
              0.5 * (gdept0[a] + gdept0[b]), f_v[a, :][:, :, None])
    tw_T = _shear_integrand(-RHO0 * al * (T[b] - T[a]), *common).sum(axis=(0, 2)) / 1e6
    tw_S = _shear_integrand(+RHO0 * be * (S[b] - S[a]), *common).sum(axis=(0, 2)) / 1e6
    return tw_T, tw_S


def contrast_profile(fld, wet):
    """Band-accumulated meridional contrast per level, zonally averaged over the
    longitudes where the pair is wet: sum_j (f[j+1]-f[j]) -> f(45S) - f(64S)."""
    a, b = slice(J0, J1), slice(J0 + 1, J1 + 1)
    pair = wet[a] & wet[b]
    acc = np.sum(np.where(pair, fld[b] - fld[a], 0.0), axis=0)      # (x,z)
    n = pair.any(axis=0).sum(axis=0)                               # wet lons per level
    return np.where(n > 0, acc.sum(axis=0) / np.maximum(n, 1), np.nan)


def depth_split(prof, wet):
    """Thickness-weighted mean of a per-level profile above / below DEEP_M [same
    units as prof].  Thickness = band-mean e3t_0 over wet cells."""
    a, b = slice(J0, J1), slice(J0 + 1, J1 + 1)
    pair = wet[a] & wet[b]
    thick = np.sum(np.where(pair, 0.5 * (e3t0[a] + e3t0[b]), 0.0), axis=(0, 1))
    up = gdept1d < DEEP_M
    out = []
    for sel in (up, ~up):
        w = np.where(sel & np.isfinite(prof), thick, 0.0)
        out.append(float(np.sum(np.nan_to_num(prof) * w) / max(np.sum(w), 1e-30)))
    return out


def rho_of(st, wet):
    p = gdept0 * RHO0 * constants.g          # nemo_seos_eos takes Pa; zh = p/(rho0 g)
    r = np.asarray(nemo_seos_eos(st["T"], st["S"], p, CFG))
    return np.where(wet, r, np.nan)


# --------------------------------------------------------------------- main ---
def main():
    lego5 = load_lego(f"{LEGO_DIR}/year_seamfix_y5.npz")
    nemo5 = load_nemo(f"{DINO}/RUN_5Y/DINO_1y_00050101_00051230_grid_T.nc", 0,
                      f"{DINO}/RUN_5Y/DINO_1y_00050101_00051230_grid_U.nc")
    wet = tmask & (lego5["land_mask"][:, :, None] > 0.5)     # SAME mask for both
    wet_u = umask & (lego5["land_mask"][:, :, None] > 0.5)

    print(f"frame {NY}x{NX}x{NZ} | channel band T-rows {J0}..{J1} "
          f"({gphit[J0, 25]:.1f}..{gphit[J1, 25]:.1f} lat, {L_BAND / 1e3:.0f} km, "
          f"all {NX} lons wet) | deep split {DEEP_M:.0f} m")
    print(f"mask: tmask&land_mask, {int(wet.sum())} wet T-cells "
          f"(tmask alone {int(tmask.sum())}) -- identical for both models")
    print(f"EOS: nemo_seos_eos a0={CFG.a0} b0={CFG.b0} lambda1={CFG.lambda1} "
          f"mu1={CFG.mu1} rho0={RHO0}\n")

    # ---------------- 1. THE THERMAL-WIND TEST -------------------------------
    print("=" * 78)
    print("1. THERMAL-WIND TEST (year 5, median over longitudes 2..-2)")
    print("=" * 78)
    res = {}
    med = lambda a: float(np.median(a[2:-2]))          # noqa: E731 -- the ACC metric's reducer
    # The band decomposition uses the MEAN over the same longitudes 2..-2: the
    # median is not additive, so B+C would not sum to A under it.  Both reducers
    # are printed for A to show the choice is immaterial (the section transport is
    # nearly longitude-independent, which is why the recorded metric medians).
    avg = lambda a: float(np.mean(a[2:-2]))            # noqa: E731
    for name, st in (("legoESM", lego5), ("NEMO", nemo5)):
        rho = rho_of(st, wet)
        tw, integ = thermal_wind(rho, wet)
        bc, bt = bc_bt_band(st["u"], wet_u)
        ab = acc_band(st["u"], wet_u)
        # Exact by construction; tolerance is float32 roundoff (NEMO output is
        # float32, legoESM float64 -- a data property, not a protocol difference).
        assert np.allclose(bc + bt, ab, rtol=1e-6, atol=1e-5), \
            f"{name}: per-longitude bc+bt != acc_band (indexing bug)"
        res[name] = dict(
            acc_full=acc_full(st["u"], wet_u),
            acc_full_e30=med(np.einsum("jik,j->i", np.where(wet_u, st["u"] * e3t0, 0.0), e2u_col) / 1e6),
            acc_band_med=med(ab),
            acc_band=avg(ab),
            acc_band_e31d=avg(acc_band(st["u"], wet_u, e3t1d)),
            bc=avg(bc), bt=avg(bt), tw=avg(tw), tw_med=med(tw), rho=rho, integ=integ,
        )
    lg, nm = res["legoESM"], res["NEMO"]
    print(f"{'':30s}{'legoESM':>12s}{'NEMO':>12s}{'ratio':>10s}{'diff':>9s}")
    for key, lab in (("acc_full", "ACC actual full-sect, e3t_1d*"),
                     ("acc_full_e30", "ACC actual full-sect, e3t_0*"),
                     ("acc_band_med", "ACC actual band, e3t_0*"),
                     ("acc_band_e31d", "ACC actual band, e3t_1d"),
                     ("acc_band", "ACC actual band, e3t_0  [A]"),
                     ("bc", "  baroclinic, measured [B]"),
                     ("bt", "  barotropic, measured  [C]"),
                     ("tw", "THERMAL WIND predicts B [D]")):
        print(f"{lab:30s}{lg[key]:12.1f}{nm[key]:12.1f}"
              f"{lg[key] / nm[key]:10.3f}{lg[key] - nm[key]:+9.1f}")
    print("  (* = median over lons 2..-2, the recorded metric's reducer -- row 1 IS the")
    print("   recorded 65.5/91.1, reproduced.  Unstarred = mean over the same lons, so")
    print("   B+C==A exactly; median-vs-mean on A differs by "
          f"{abs(lg['acc_band_med'] - lg['acc_band']):.1f}/{abs(nm['acc_band_med'] - nm['acc_band']):.1f} Sv"
          " lego/NEMO.)")
    # Where in latitude does the FULL-section deficit sit?  The recorded metric
    # integrates over ALL 199 rows, not just the re-entrant band; only the band
    # can carry a net throughflow, so the rest is blocked-latitude recirculation
    # that does not cancel in a single-longitude section.  e3t_1d, as recorded.
    print("  full-section (e3t_1d) transport by latitude group [Sv]:")
    for lab, sl in (("south of band  ", slice(0, J0)), ("BAND (re-entrant)", slice(J0, J1 + 1)),
                    ("north of band  ", slice(J1 + 1, NY))):
        v = []
        for st in (lego5, nemo5):
            w = np.where(wet_u[sl], st["u"][sl], 0.0)
            v.append(med(np.einsum("jik,k,j->i", w, e3t1d, e2u_col[sl]) / 1e6))
        print(f"    {lab} rows {sl.start:3d}..{sl.stop - 1:3d}: lego {v[0]:+8.1f}  "
              f"NEMO {v[1]:+8.1f}  diff {v[0] - v[1]:+7.1f}")
    for n, d in (("legoESM", lg), ("NEMO", nm)):
        print(f"  geostrophy check {n:8s}: D/B = {d['tw'] / d['bc']:.2f} "
              f"({d['tw']:.1f} vs {d['bc']:.1f} Sv)")
    print("\n  ADDITIVE deficit decomposition of the band transport [Sv, lego - NEMO]:")
    print(f"    total  A: {lg['acc_band'] - nm['acc_band']:+7.1f}"
          f"  =  baroclinic B: {lg['bc'] - nm['bc']:+7.1f}"
          f"  +  barotropic C: {lg['bt'] - nm['bt']:+7.1f}")
    print(f"    density-predicted baroclinic deficit D: {lg['tw'] - nm['tw']:+7.1f} Sv"
          f"  ({100 * (lg['tw'] - nm['tw']) / (lg['acc_band'] - nm['acc_band']):.0f}% of total A)")
    r_act, r_bc, r_tw = (lg['acc_band'] / nm['acc_band'], lg['bc'] / nm['bc'],
                         lg['tw'] / nm['tw'])
    print(f"    ratios: actual band {r_act:.3f} | measured baroclinic {r_bc:.3f} | "
          f"thermal-wind {r_tw:.3f} | recorded full-section {lg['acc_full'] / nm['acc_full']:.3f}")
    if r_tw <= r_act + 0.05:
        print("  => VERDICT: the density deficit accounts for AT LEAST the whole transport")
        print("     deficit (thermal-wind ratio <= actual ratio). The momentum side is not")
        print("     losing transport -- it is if anything compensating. BUOYANCY PROBLEM.")
    elif r_tw > 0.95:
        print("  => VERDICT: density is fine; the deficit is barotropic/momentum.")
    else:
        print("  => VERDICT: SPLIT -- see the additive decomposition above.")

    # ---------------- 2. DENSITY STRUCTURE ----------------------------------
    print("\n" + "=" * 78)
    print("2. DENSITY STRUCTURE vs DEPTH (band contrast = value@45S - value@64S,")
    print("   zonal mean over wet lons; slope = -(drho/dy)/(drho/dz), band mean)")
    print("=" * 78)
    prof = {}
    for n, st in (("legoESM", lego5), ("NEMO", nemo5)):
        rc = contrast_profile(res[n]["rho"], wet)
        Tc = contrast_profile(np.where(wet, st["T"], np.nan), wet)
        Sc = contrast_profile(np.where(wet, st["S"], np.nan), wet)
        _rb = np.where(wet, res[n]["rho"], np.nan)[J0:J1 + 1]
        nk = np.sum(np.isfinite(_rb), axis=(0, 1))
        rbar = np.where(nk > 0, np.nansum(np.nan_to_num(_rb), axis=(0, 1)) / np.maximum(nk, 1), np.nan)
        drdz = np.gradient(rbar, -gdept1d)                   # z UP => drho/dz < 0
        slope = -(rc / L_BAND) / drdz
        twk = per_lev(res[n]["integ"])                       # per-level Sv per longitude
        prof[n] = dict(rc=rc, Tc=Tc, Sc=Sc, slope=slope, twk=twk)
    print(f"{'k':>3s}{'depth':>8s} | {'drho lego':>10s}{'drho nemo':>10s}{'ratio':>7s}"
          f" | {'dT lego':>8s}{'dT nemo':>8s} | {'dS lego':>8s}{'dS nemo':>8s}"
          f" | {'slp lego':>9s}{'slp nemo':>9s} | {'twSv lego':>10s}{'twSv nemo':>10s}")
    for k in range(NZ):
        a, b = prof["legoESM"], prof["NEMO"]
        rr = a["rc"][k] / b["rc"][k] if abs(b["rc"][k]) > 1e-9 else np.nan
        print(f"{k:3d}{gdept1d[k]:8.0f} | {a['rc'][k]:10.4f}{b['rc'][k]:10.4f}{rr:7.2f}"
              f" | {a['Tc'][k]:8.3f}{b['Tc'][k]:8.3f} | {a['Sc'][k]:8.4f}{b['Sc'][k]:8.4f}"
              f" | {a['slope'][k]:9.2e}{b['slope'][k]:9.2e}"
              f" | {a['twk'][k]:10.2f}{b['twk'][k]:10.2f}")
    for n in ("legoESM", "NEMO"):
        u_, d_ = depth_split(prof[n]["rc"], wet)
        tu = float(prof[n]["twk"][gdept1d < DEEP_M].sum())
        td = float(prof[n]["twk"][gdept1d >= DEEP_M].sum())
        print(f"  {n:8s} thickness-wtd drho contrast: upper {u_:+.4f}  deep {d_:+.4f} kg/m3"
              f" | tw contribution: upper {tu:6.1f}  deep {td:6.1f} Sv")

    # Which END of the band is wrong?  A too-weak deep contrast is either "no
    # cold water made at the poleward edge" or "the equatorward deep is not warm
    # enough" -- different budget terms.  5-row, all-longitude means at the edges.
    print("  deep end-member T [degC] (5-row x all-lon mean at the band edges):")
    print(f"{'k':>5s}{'depth':>7s}{'  S-edge lego  S-edge NEMO  N-edge lego  N-edge NEMO':s}")
    for k in range(25, NZ - 1):
        vals = []
        for st in (lego5, nemo5):
            for sl in (slice(J0, J0 + 5), slice(J1 - 4, J1 + 1)):
                m = wet[sl, :, k]
                vals.append(float(st["T"][sl, :, k][m].mean()) if m.any() else np.nan)
        print(f"{k:5d}{gdept1d[k]:7.0f}{vals[0]:13.3f}{vals[2]:13.3f}{vals[1]:13.3f}{vals[3]:13.3f}")

    # ---------------- 3. TIME EVOLUTION -------------------------------------
    print("\n" + "=" * 78)
    print(f"3. TIME EVOLUTION -- band drho contrast (thickness-wtd, kg/m3) split at"
          f" {DEEP_M:.0f} m + thermal-wind and actual transport")
    print("   (all medians over lons 2..-2; tw_up+tw_deep misses tw_tot by ~2% because")
    print("    a sum of per-level medians is not the median of the sum)")
    print("=" * 78)
    rows = []
    for y in range(1, 6):
        p = f"{LEGO_DIR}/year_seamfix{'' if y == 1 else f'_y{y}'}.npz"
        rows.append(("legoESM", y, load_lego(p), True))
    N20T = f"{DINO}/RUN_20Y_REBUILD/DINO_1y_00060101_00201230_grid_T.nc"
    N20U = f"{DINO}/RUN_20Y_REBUILD/DINO_1y_00060101_00201230_grid_U.nc"
    rows.append(("NEMO", 1, load_nemo(f"{DINO}/RUN_1Y/DINO_1y_00010101_00011230_grid_T.nc", 0,
                                      f"{DINO}/RUN_1Y/DINO_1y_00010101_00011230_grid_U.nc"), True))
    rows.append(("NEMO", 5, nemo5, True))
    for i in range(15):
        rows.append(("NEMO", 6 + i, load_nemo(N20T, i, N20U), True))
    print(f"{'model':>8s}{'yr':>4s}{'drho_up':>10s}{'drho_deep':>11s}"
          f"{'tw_up':>8s}{'tw_deep':>9s}{'tw_tot':>8s}{'ACCband':>9s}{'ACCfull':>9s}")
    for n, y, st, has_u in rows:
        rho = rho_of(st, wet)
        tw, integ = thermal_wind(rho, wet)
        twk = per_lev(integ)
        u_, d_ = depth_split(contrast_profile(rho, wet), wet)
        ab = np.median(acc_band(st["u"], wet_u)[2:-2]) if has_u else np.nan
        af = acc_full(st["u"], wet_u) if has_u else np.nan
        print(f"{n:>8s}{y:4d}{u_:10.4f}{d_:11.4f}"
              f"{twk[gdept1d < DEEP_M].sum():8.1f}{twk[gdept1d >= DEEP_M].sum():9.1f}"
              f"{np.median(tw[2:-2]):8.1f}{ab:9.1f}{af:9.1f}")

    # ---------------- 4. T vs S ---------------------------------------------
    print("\n" + "=" * 78)
    print("4. T vs S CONTRIBUTION to the density gradient (year 5)")
    print("   drho = -rho0*alpha*dT + rho0*beta*dS, alpha/beta at the v-point midpoint")
    print("=" * 78)
    print(f"{'':22s}{'legoESM':>12s}{'NEMO':>12s}{'ratio':>10s}")
    tws = {}
    for n, st in (("legoESM", lego5), ("NEMO", nemo5)):
        tws[n] = ts_split(st["T"], st["S"], wet)
    for idx, lab in ((0, "tw from T grad [Sv]"), (1, "tw from S grad [Sv]")):
        a = float(np.median(tws["legoESM"][idx][2:-2]))
        b = float(np.median(tws["NEMO"][idx][2:-2]))
        print(f"{lab:22s}{a:12.1f}{b:12.1f}{a / b if abs(b) > 1e-9 else np.nan:10.3f}")
    for n in ("legoESM", "NEMO"):
        s = float(np.median(tws[n][0][2:-2])), float(np.median(tws[n][1][2:-2]))
        tot = res[n]["tw_med"]      # medians throughout this section
        print(f"  {n:8s}: T {s[0]:+7.1f} + S {s[1]:+7.1f} = {sum(s):7.1f} Sv vs full tw "
              f"{tot:7.1f} (linearisation err {sum(s) - tot:+.2f}) | "
              f"|S|/(|T|+|S|) = {abs(s[1]) / (abs(s[0]) + abs(s[1])):.2f}")
        # the linearised split must reconstruct the full functional
        assert abs(sum(s) - tot) < 0.03 * max(abs(tot), 1.0), "alpha/beta split does not close"
    for n, d in (("legoESM", lg), ("NEMO", nm)):
        assert abs(d["bc"] + d["bt"] - d["acc_band"]) < 1e-6 * max(abs(d["acc_band"]), 1.0), \
            f"{n}: measured bc+bt != actual band transport"
    print("\nself-checks passed: per-longitude bc+bt == acc_band (exact, both models), "
          "alpha/beta T/S split closes (<3%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
