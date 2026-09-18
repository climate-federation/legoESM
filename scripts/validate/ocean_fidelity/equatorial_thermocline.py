#!/usr/bin/env python
"""Equatorial Pacific thermocline: ours vs NEMO, along 2S-2N.

WHY.  The cold tongue is +3.19 C too warm in Nino3 and the bias GROWS
(+2.09/+2.50/+3.19 at days 10/20/30).  Two families of cause produce that same
SST signature and need different fixes:

  A. THERMOCLINE / UPWELLING.  If the equatorial thermocline is too DEEP or
     too DIFFUSE, the water upwelled at the equator comes from a warmer level
     and the cold tongue cannot form.  Z20 (depth of the 20 C isotherm) is the
     standard index for this, and its EAST-WEST TILT is the standard index for
     whether the equatorial circulation is right at all.

  B. SURFACE FLUX.  A heat-flux or wind-speed error warms the surface without
     touching the subsurface structure.

They are distinguished by the SUBSURFACE: cause A shifts Z20 and flattens the
tilt, cause B leaves Z20 alone and warms only the top few tens of metres.  So
this probe reports Z20(lon) for both models plus the temperature profile at a
few longitudes, and nothing else -- it is a discriminator, not a scorecard.

READ THE OUTPUT AS A DISCRIMINATOR, NOT A VERDICT.  A matching Z20 with a warm
surface points at B.  A too-deep or too-flat Z20 points at A.  Anything else
means neither family is clean and the probe has done its job by saying so.

Instrument caveats, stated up front:
  * ours is an INSTANTANEOUS snapshot, NEMO is a climatological month mean --
    the same mismatch every scorecard in this campaign carries;
  * Z20 is undefined where the whole column is below 20 C; those columns are
    reported as NaN and COUNTED rather than silently dropped;
  * we interpolate Z20 linearly in depth between the bracketing levels, on
    both sides, through the same function -- no per-model convention.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1]))


def z20_from_column(T, z, iso=20.0):
    """Depth [m, >0] where T first crosses `iso` downward; NaN if never.

    Linear interpolation between the bracketing levels. Applied identically to
    both models so the convention cannot become a difference.
    """
    T = np.asarray(T, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    good = np.isfinite(T)
    if good.sum() < 2:
        return np.nan
    T, z = T[good], z[good]
    if T[0] < iso:          # surface already colder than the isotherm
        return np.nan
    below = np.nonzero(T < iso)[0]
    if below.size == 0:     # never reaches it within the column
        return np.nan
    k = int(below[0])
    T1, T0 = T[k], T[k - 1]
    z1, z0 = z[k], z[k - 1]
    if not np.isfinite(T1 - T0) or (T1 - T0) == 0.0:
        return float(z1)
    return float(z0 + (z1 - z0) * (T0 - iso) / (T0 - T1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legoesm-snapshot", required=True)
    ap.add_argument("--nemo-gridt", required=True)
    ap.add_argument("--nemo-month", type=int, default=1)
    ap.add_argument("--nemo-t-recs", default=None,
                    help="Explicit record slice 'A:B' into --nemo-gridt, "
                         "averaged, INSTEAD of selecting a calendar month. "
                         "Use it with RUN_GATEWAY's 5-day grid_T so the "
                         "thermocline is compared against NEMO's MATCHED cold "
                         "start rather than a multi-year monthly climatology "
                         "(rec 5 = days 26-30, rec 17 = days 86-90) -- the "
                         "same records --nemo-w-recs already selects for the "
                         "velocity side, which was the only half of this "
                         "instrument using the matched reference.")
    ap.add_argument("--lat-halfwidth", type=float, default=2.0)
    ap.add_argument("--label", default="legoESM")
    ap.add_argument("--nemo-wfile", default=None,
                    help="NEMO file carrying `wocetr_eff` (the hourly trd1h_T; "
                         "the monthly grid_T does NOT have it). Enables the "
                         "UPWELLING block, which asks the question Z20 cannot: "
                         "if the thermocline is too deep, is it because the "
                         "equatorial ascent is too weak?")
    ap.add_argument("--mesh-mask", default=None,
                    help="eORCA1 mesh_mask, required with --nemo-wfile: "
                         "`wocetr_eff` is a TRANSPORT [m3/s] and our "
                         "`mass_flux_w` is a VELOCITY [m/s], so one side has "
                         "to be divided by the cell area e1t*e2t before they "
                         "are the same quantity.")
    ap.add_argument("--nemo-w-var", default="wocetr_eff",
                    choices=("wocetr_eff", "wo"),
                    help="which NEMO vertical field the wfile carries. "
                         "`wocetr_eff` is a TRANSPORT [m3/s] (hourly trd1h_T; "
                         "needs --mesh-mask for the area divide). `wo` is a "
                         "VELOCITY [m/s] (the 5-day grid_W files) and is "
                         "compared directly.")
    ap.add_argument("--box-budget", default=None,
                    help="'lo,hi' longitude window: volume budget of the box "
                         "|lat|<=--budget-lat-halfwidth, --budget-layer, ours "
                         "(stored mass fluxes) vs NEMO (uocetr_eff/vocetr_eff/"
                         "wo; needs --nemo-ufile/--nemo-vfile/--nemo-wfile/"
                         "--mesh-mask/--nemo-w-recs)")
    ap.add_argument("--budget-layer", default="50,150")
    ap.add_argument("--heat-budget", action="store_true",
                    help="LIMITATION FIRST (measured 2026-09-08, do not quote "
                         "these terms as a closed budget): neither side can be "
                         "closed from the archived fields. Ours stores "
                         "INSTANTANEOUS fluxes, so the terms cannot match a "
                         "15-day mean tendency; the oracle publishes 5-day "
                         "MEANS, and a mean diffusivity times a mean gradient "
                         "is not a mean flux -- where its mixing is "
                         "intermittent the two are anticorrelated (avt at 50 m "
                         "in the cold tongue: median 4.1e-04, max 3.8e+01 "
                         "m2/s), which is what produced a +852 K/month "
                         "diffusive term at 50-150 m. The terms are usable "
                         "only for ORDER and SIGN in a layer with no "
                         "intermittent mixing (15-25 m), never as a closure. "
                         "With --box-budget: split the layer's HEAT "
                         "budget into horizontal advection, vertical "
                         "advection and vertical diffusion on both sides, in "
                         "K/month, using each side's own upwind temperature "
                         "on its own faces. Codex+GLM discriminator "
                         "(2026-09-08) for whether the cold 20 m is advective."
                         " Needs `to` in --nemo-gridt and `avt` in "
                         "--nemo-wfile.")
    ap.add_argument("--heat-budget-average", default=None,
                    help="comma-separated legoESM snapshots to AVERAGE before "
                         "computing the budget, so our instantaneous fluxes "
                         "become a time mean over the oracle's own window. "
                         "Without this the budget cannot close (2026-09-08); "
                         "with it, pass the matching --nemo-w-recs.")
    ap.add_argument("--heat-budget-prev", default=None,
                    help="earlier legoESM snapshot: gives the OBSERVED dT/dt "
                         "the three terms are checked against (the control -- "
                         "a residual comparable to the terms means the "
                         "instrument is being read, not the ocean).")
    ap.add_argument("--budget-lat-halfwidth", default="2")
    ap.add_argument("--nemo-w-recs", default=None,
                    help="record selection in the wfile, `K` or `A:B` "
                         "(python slice, stop-exclusive). Default: all "
                         "records, averaged. Use this to MATCH the window of "
                         "the snapshot (e.g. `17:18` = days 86-90 of a 5-day "
                         "file against a day-90 snapshot).")
    ap.add_argument("--nemo-ufile", default=None,
                    help="NEMO 5-day grid_U file (uo). Enables the EUC block: "
                         "zonal-velocity profile at the equator, ours vs NEMO "
                         "-- an absent undercurrent and an absent ascent need "
                         "different fixes. Record selection follows "
                         "--nemo-w-recs.")
    ap.add_argument("--nemo-vfile", default=None,
                    help="NEMO 5-day grid_V file (vo). With --meridional-lon, "
                         "also prints the MERIDIONAL surface current v(lat) at "
                         "each longitude — the Ekman-divergence carrier. With "
                         "the wind stress matched (same NCAR bulk, vfac=0 both "
                         "sides), a too-weak or displaced poleward v at "
                         "+-1..3 deg implicates the vertical-viscosity "
                         "profile, not the forcing.")
    ap.add_argument("--v-depth-m", type=float, default=10.0,
                    help="depth of the v(lat) comparison [m].")
    ap.add_argument("--meridional-lon", default=None,
                    help="comma list of longitudes (deg E). For each, print "
                         "the MERIDIONAL profile of w at --w-depth-m over "
                         "|lat|<=8, ours vs NEMO: a DISPLACED ascent (upwelling "
                         "off-equator) and an ABSENT ascent look identical in "
                         "the 2S-2N band mean and need different fixes.")
    ap.add_argument("--w-depth-m", type=float, default=50.0,
                    help="depth at which the equatorial vertical velocity is "
                         "compared. 50 m sits inside the upwelling core and "
                         "above the Z20 the bias is measured at.")
    a = ap.parse_args()

    import xarray as xr
    from compare_omip_nemo import _load_legoesm, _load_nemo, regrid_curv_to_latlon

    L = _load_legoesm(a.legoesm_snapshot)
    if L.get("T3d") is None or L.get("z_center_ref") is None:
        raise SystemExit("snapshot lacks T3d / z_center_ref -- cannot do Z20")
    N = _load_nemo(a.nemo_gridt, 0, month=a.nemo_month)

    ds = xr.open_dataset(a.nemo_gridt, decode_times=False)
    # ORCA1's grid_T calls 3-D temperature "to"; other NEMO configs and CMOR
    # output use thetao/votemper/toce. Checked in that order and named in the
    # error so a miss is a one-line fix rather than a hunt.
    tvar = next((v for v in ("to", "thetao", "votemper", "toce")
                 if v in ds and ds[v].ndim >= 3), None)
    if tvar is None:
        raise SystemExit(f"{a.nemo_gridt}: no 3-D temperature (looked for "
                         f"to/thetao/votemper/toce; present: "
                         f"{sorted(ds.data_vars)[:12]})")
    print(f"[nemo] 3-D temperature variable: {tvar!r}")

    zc = np.asarray(L["z_center_ref"], dtype=np.float64)
    zc = np.abs(zc)
    nlev = zc.size
    print(f"[lego] {nlev} levels, {zc[0]:.1f} -> {zc[-1]:.0f} m")

    # Regrid our columns level by level, through the SAME helper the scorecard
    # uses, so the horizontal treatment is not a new convention either.
    tgt_lat = np.arange(-89.5, 90.0, 1.0)
    tgt_lon = np.arange(0.5, 360.0, 1.0)
    lev = []
    for k in range(nlev):
        f, _ = regrid_curv_to_latlon(np.asarray(L["T3d"])[..., k], L["lat"],
                                     L["lon"], L["mask"], tgt_lat, tgt_lon)
        lev.append(f)
    Lg = np.stack(lev, axis=-1)                      # (nlat, nlon, nlev)

    band = np.abs(tgt_lat) <= a.lat_halfwidth
    print(f"[band] |lat| <= {a.lat_halfwidth} -> {int(band.sum())} rows")

    # NEMO on its own grid, band-averaged after regridding to the same target.
    nT = np.asarray(ds[tvar].values)
    if nT.ndim == 4:
        if a.nemo_t_recs:
            # Explicit records: the matched cold-start window, same convention
            # as --nemo-w-recs on the velocity side.
            sel = _select_recs(nT, a.nemo_t_recs)
            print(f"[nemo] temperature from records {a.nemo_t_recs} "
                  f"({sel.shape[0]} of {nT.shape[0]}) -- MATCHED window")
            nT = np.nanmean(sel, axis=0)
        else:
            from compare_omip_nemo import _nemo_record_months
            tdim = ds[tvar].dims[0]
            nt = int(ds.sizes[tdim])
            months = _nemo_record_months(ds, tdim, nt)
            idx = ([i for i in range(nt) if months[i] == a.nemo_month]
                   if months is not None else [a.nemo_month - 1])
            print(f"[nemo] temperature from calendar month {a.nemo_month} "
                  f"({len(idx)} records averaged) -- pass --nemo-t-recs for "
                  f"the matched cold-start window instead")
            nT = np.nanmean(nT[idx], axis=0)
    nT = np.where(np.abs(nT) > 1e10, np.nan, nT)
    nT = np.where(nT == 0.0, np.nan, nT)
    zn = np.asarray(ds["deptht"].values, dtype=np.float64)
    print(f"[nemo] {zn.size} levels, {zn[0]:.1f} -> {zn[-1]:.0f} m")

    nlat_n, nlon_n = nT.shape[1], nT.shape[2]
    nav_lat = np.asarray(ds["nav_lat"].values)
    nav_lon = np.asarray(ds["nav_lon"].values)
    mask2d = np.isfinite(nT[0])
    Ng = []
    for k in range(zn.size):
        f, _ = regrid_curv_to_latlon(nT[k], nav_lat, nav_lon,
                                     mask2d.astype(np.float64), tgt_lat, tgt_lon)
        Ng.append(f)
    Ng = np.stack(Ng, axis=-1)

    print()
    print("Z20 [m] along the equator, band mean. NEMO minus ours is the gap;")
    print("a DEEPER ours (positive gap) means a more diffuse thermocline.")
    print(f"{'lon':>6} {'ours':>9} {'NEMO':>9} {'ours-NEMO':>10}")
    lons = [160, 180, 200, 220, 240, 260, 270, 280]
    gaps = []
    for lon in lons:
        j = int(np.argmin(np.abs(tgt_lon - lon)))
        zl = np.nanmean([z20_from_column(Lg[i, j, :], zc)
                         for i in np.nonzero(band)[0]])
        zn_ = np.nanmean([z20_from_column(Ng[i, j, :], zn)
                          for i in np.nonzero(band)[0]])
        d = zl - zn_
        if np.isfinite(d):
            gaps.append(d)
        f = lambda x: f"{x:9.1f}" if np.isfinite(x) else "      nan"
        print(f"{lon:6d} {f(zl)} {f(zn_)} {d:10.1f}" if np.isfinite(d)
              else f"{lon:6d} {f(zl)} {f(zn_)}        nan")
    if gaps:
        print(f"\n[Z20] mean gap {np.mean(gaps):+.1f} m over "
              f"{len(gaps)}/{len(lons)} longitudes")

    # East-west tilt: the single number that says whether the equatorial
    # circulation exists at all. NEMO's Pacific thermocline shoals eastward.
    def tilt(Z):
        jw = int(np.argmin(np.abs(tgt_lon - 160)))
        je = int(np.argmin(np.abs(tgt_lon - 260)))
        w = np.nanmean([z20_from_column(Z[i, jw, :], zc if Z is Lg else zn)
                        for i in np.nonzero(band)[0]])
        e = np.nanmean([z20_from_column(Z[i, je, :], zc if Z is Lg else zn)
                        for i in np.nonzero(band)[0]])
        return w - e
    tl, tn = tilt(Lg), tilt(Ng)
    print(f"[tilt] west(160E)-east(100W) Z20: ours {tl:+.1f} m, "
          f"NEMO {tn:+.1f} m  -> ours is {100.0 * tl / tn:.0f}% of NEMO's"
          if np.isfinite(tn) and tn != 0 else
          f"[tilt] ours {tl:+.1f} m, NEMO {tn:+.1f} m")

    if a.nemo_wfile:
        if a.nemo_w_var == "wocetr_eff" and not a.mesh_mask:
            raise SystemExit("--nemo-wfile requires --mesh-mask: wocetr_eff is "
                             "m3/s and our mass_flux_w is m/s, and without the "
                             "cell area the two are not the same quantity.")
        _upwelling_block(a, L, tgt_lat, tgt_lon, band, zc,
                         regrid_curv_to_latlon)
    if a.meridional_lon or a.nemo_ufile:
        _euc_merid_block(a, L, zc)
    if a.box_budget:
        _box_budget_block(a, L, zc)
    return 0


_SEC_PER_MONTH = 30.0 * 86400.0


class _MeanSnap(dict):
    """Time-averaged snapshot: same access surface as an npz for the budget."""

    @property
    def files(self):
        return list(self.keys())



def _upwind(flux, lo_side, hi_side):
    """Temperature carried by ``flux`` on a face: the LO-side cell when the
    flux is positive (flowing lo -> hi), the HI-side cell otherwise."""
    return np.where(flux > 0.0, lo_side, hi_side)


def _heat_terms(Fw, Fs, Fup, T, K, area, zc, z_if, kl, kt, kb, box, wet):
    """Layer heat budget, one side, in K/month.

    Sign convention, stated once and used everywhere: z is UP, every returned
    term is a TENDENCY of the layer mean temperature (positive = warming).
      Fw   (ny, nx+1, nz)  volume flux [m3/s] through the WEST face of each
                           cell, positive EASTWARD
      Fs   (ny+1, nx, nz)  through the SOUTH face, positive NORTHWARD
      Fup  (ny, nx, nz+1)  through each interface, positive UPWARD
      K    (ny, nx, nz-1)  tracer diffusivity on the INTERIOR interfaces
                           (interface index k maps to K[..., k-1])
      wet  (ny, nx, nz)    True where the cell is ocean.  Cells below the
                           bathymetry carry a FILL temperature, and a fill
                           value next to real water is a 20 K gradient across
                           one interface -- that produced a 852 K/month
                           diffusive term on the oracle's side before this
                           argument existed, so a dry neighbour zeroes the
                           interface rather than being read as a gradient.
    Returns (horizontal advection, vertical advection, vertical diffusion).
    """
    Tl = np.concatenate([T[:, :1], T], axis=1)      # cell west of each face
    Tr = np.concatenate([T, T[:, -1:]], axis=1)     # cell east of each face
    Wl = np.concatenate([wet[:, :1], wet], axis=1)
    Wr = np.concatenate([wet, wet[:, -1:]], axis=1)
    Hx = np.where(Wl & Wr, Fw * _upwind(Fw, Tl, Tr), 0.0)
    Td = np.concatenate([T[:1], T], axis=0)         # cell south of each face
    Tu = np.concatenate([T, T[-1:]], axis=0)        # cell north of each face
    Wd = np.concatenate([wet[:1], wet], axis=0)
    Wu = np.concatenate([wet, wet[-1:]], axis=0)
    Hy = np.where(Wd & Wu, Fs * _upwind(Fs, Td, Tu), 0.0)
    Ta = np.concatenate([T[..., :1], T], axis=-1)   # cell ABOVE each interface
    Tb = np.concatenate([T, T[..., -1:]], axis=-1)  # cell BELOW each interface
    Wa = np.concatenate([wet[..., :1], wet], axis=-1)
    Wb = np.concatenate([wet, wet[..., -1:]], axis=-1)
    Hz = np.where(Wa & Wb, Fup * _upwind(Fup, Tb, Ta), 0.0)

    dz = np.diff(z_if)
    vol = float((area[..., None] * dz[None, None, :] * wet)[..., kl][box].sum())
    # heat leaving the box through the east/north faces minus that entering
    out_x = float((Hx[:, 1:, :] - Hx[:, :-1, :])[..., kl].sum(-1)[box].sum())
    out_y = float((Hy[1:, :, :] - Hy[:-1, :, :])[..., kl].sum(-1)[box].sum())
    out_z = float((Hz[..., kt] - Hz[..., kb])[box].sum())
    # downgradient diffusive flux, positive UP:  F = -K dT/dz = K (T_below - T_above)/dz
    def _fd(k):
        if k < 1 or k > K.shape[-1]:
            return np.zeros_like(area)
        both = wet[..., k] & wet[..., k - 1]
        return np.where(both,
                        K[..., k - 1] * (T[..., k] - T[..., k - 1])
                        / (zc[k] - zc[k - 1]) * area, 0.0)
    fd_cell = (_fd(kt) - _fd(kb))
    out_d = float(fd_cell[box].sum())
    f = _SEC_PER_MONTH / vol
    # An instrument that can be dominated by ONE cell is not measuring the box:
    # report the worst single-cell share so an outlier cannot hide in the mean.
    worst = float(np.abs(fd_cell[box]).max()) if box.any() else 0.0
    share = worst / max(abs(out_d), 1e-30)
    if share > 0.25:
        print(f"  [warn] one cell carries {100 * share:.0f}% of the diffusive term "
              f"({worst * f:+.3f} of {-out_d * f:+.3f} K/month) -- treat it as an "
              "instrument outlier, not the ocean")
    return -out_x * f - out_y * f, -out_z * f, -out_d * f


def _box_budget_block(a, L, zc):
    """Volume budget of a lon/lat/depth box, ours vs NEMO, split by wall
    direction: which velocity component carries the convergence that heaves
    the lower thermocline (isopycnal probe, 2026-09-05).

    OURS: the STORED tracer-advecting fluxes -- mass_flux_u/v [m^2/s] x face
    length (dy_u/dx_v) = m^3/s, mass_flux_w [m/s] x cell_area.  NEMO:
    uocetr_eff/vocetr_eff [m^3/s] (the effective tracer transport, same
    quantity class as ours) and wo [m/s] x e1t*e2t.  Both sides are summed as
    CELL divergences over the box (east face minus west face per T cell), so
    the zonal / meridional split is exact on each side's own C-grid and no
    wall has to be located on a curvilinear mesh.

    CONTROL: the horizontal net outflow of the layer must be balanced by the
    vertical net outflow (w_top - w_bot)*area, up to the slow z-star thickness
    change; the residual is PRINTED on both sides and the vertical sign is
    the one that closes it (stated, not assumed).  A residual comparable to
    the terms means the instrument, not the ocean, is being read.
    """
    import netCDF4 as nc
    lo, hi = (float(x) for x in a.box_budget.split(","))
    z0, z1 = (float(x) for x in a.budget_layer.split(","))
    half = float(a.budget_lat_halfwidth)
    if a.heat_budget_average:
        _files = [f.strip() for f in a.heat_budget_average.split(",") if f.strip()]
        _mems = [np.load(f) for f in _files]
        _keys = set(_mems[0].files)
        snap = {k: (np.mean([np.asarray(m[k], dtype=np.float64) for m in _mems], axis=0)
                    if np.asarray(_mems[0][k]).dtype.kind == "f"
                    and np.asarray(_mems[0][k]).ndim > 0 else _mems[0][k])
                for k in _keys}
        snap = _MeanSnap(snap)
        print(f"[budget] averaging {len(_files)} snapshots: "
              f"{_files[0].split('/')[-1]} .. {_files[-1].split('/')[-1]}")
    else:
        snap = np.load(a.legoesm_snapshot)
    need = ("mass_flux_u", "mass_flux_v", "mass_flux_w", "dy_u", "dx_v",
            "cell_area")
    miss = [k for k in need if k not in snap.files]
    if miss:
        raise SystemExit(f"--box-budget needs {miss} in the snapshot")
    lat = np.asarray(L["lat"], dtype=np.float64)
    lon = np.asarray(L["lon"], dtype=np.float64) % 360.0
    box = (np.abs(lat) <= half) & (lon >= lo) & (lon < hi) & (
        np.asarray(snap["land_mask"]) > 0.5)
    kl = (zc >= z0) & (zc < z1)
    z_if = np.concatenate([[0.0], 0.5 * (zc[:-1] + zc[1:]), [2.0 * zc[-1] - zc[-2]]])
    kt = int(np.argmin(np.abs(z_if - z0)))
    kb = int(np.argmin(np.abs(z_if - z1)))
    mfu = np.asarray(snap["mass_flux_u"], dtype=np.float64) * np.asarray(snap["dy_u"])[..., None]
    mfv = np.asarray(snap["mass_flux_v"], dtype=np.float64) * np.asarray(snap["dx_v"])[..., None]
    mfw = np.asarray(snap["mass_flux_w"], dtype=np.float64) * np.asarray(snap["cell_area"])[..., None]
    mfu = np.where(np.isfinite(mfu), mfu, 0.0)
    mfv = np.where(np.isfinite(mfv), mfv, 0.0)
    mfw = np.where(np.isfinite(mfw), mfw, 0.0)
    ny, nx = lat.shape
    # C-grid faces: u at (ny, nx+1), v at (ny+1, nx); east face of cell i is u[i+1]
    div_x = (mfu[:, 1:, :] - mfu[:, :-1, :])[..., kl].sum(-1)
    div_y = (mfv[1:, :, :] - mfv[:-1, :, :])[..., kl].sum(-1)
    w_top = mfw[..., kt]
    w_bot = mfw[..., kb]
    ox = float(div_x[box].sum()) / 1e6
    oy = float(div_y[box].sum()) / 1e6
    wt = float(w_top[box].sum()) / 1e6
    wb = float(w_bot[box].sum()) / 1e6
    # vertical sign: the convention that closes continuity is reported
    res_up = ox + oy + (wt - wb)        # w positive UP: outflow through top = +w_top
    res_dn = ox + oy - (wt - wb)
    sign = "up" if abs(res_up) <= abs(res_dn) else "down"
    res = res_up if sign == "up" else res_dn
    wt_up = wt if sign == "up" else -wt
    wb_up = wb if sign == "up" else -wb
    print(f"\n=== BOX VOLUME BUDGET {lo:.0f}-{hi:.0f}E |lat|<={half:g}, "
          f"{z0:.0f}-{z1:.0f} m (levels {kl.sum()}), Sv (1e6 m3/s); + = OUT of the box ===")
    print(f"ours ({a.legoesm_snapshot}): zonal out {ox:+.3f}  meridional out {oy:+.3f}  "
          f"horizontal net out {ox+oy:+.3f} | w_top(up) {wt_up:+.3f} at {z_if[kt]:.0f} m, "
          f"w_bot(up) {wb_up:+.3f} at {z_if[kb]:.0f} m => vertical net out {wt_up-wb_up:+.3f} | "
          f"residual {res:+.3f} (w sign '{sign}' closes it; {int(box.sum())} cells)")
    hb_ours = None
    if a.heat_budget:
        Tsnap = np.asarray(snap["T"], dtype=np.float64)
        Ksnap = np.asarray(snap["K_H_diag"], dtype=np.float64)
        area_o = np.asarray(snap["cell_area"], dtype=np.float64)
        mfw_up = mfw if sign == "up" else -mfw
        wet_o = np.isfinite(Tsnap) & (np.abs(Tsnap) > 1e-6)
        hb_ours = _heat_terms(mfu, mfv, mfw_up, np.nan_to_num(Tsnap), Ksnap, area_o,
                              zc, z_if, kl, kt, kb, box, wet_o)
        dzl = np.diff(z_if)[kl]
        wl = (wet_o[..., kl] * dzl)
        tbar = float((np.nan_to_num(Tsnap)[..., kl] * wl).sum(-1)[box].sum()
                     / wl.sum(-1)[box].sum())
        print(f"\n=== BOX HEAT BUDGET {z0:.0f}-{z1:.0f} m, K/month (+ = warms the layer) ===")
        print(f"ours: horizontal adv {hb_ours[0]:+.3f}   vertical adv {hb_ours[1]:+.3f}   "
              f"vertical diff {hb_ours[2]:+.3f}   sum {sum(hb_ours):+.3f}   "
              f"(layer mean T {tbar:.3f} C)")
        if a.heat_budget_prev:
            prev = np.load(a.heat_budget_prev)
            dt_days = float(snap["time_days"]) - float(prev["time_days"])
            Tp = np.nan_to_num(np.asarray(prev["T"], dtype=np.float64))
            tprev = float((Tp[..., kl] * wl).sum(-1)[box].sum() / wl.sum(-1)[box].sum())
            obs = (tbar - tprev) / dt_days * 30.0
            print(f"  CONTROL observed dT/dt over the previous {dt_days:.0f} days: {obs:+.3f} K/month;"
                  f" residual (sum - observed) {sum(hb_ours) - obs:+.3f}."
                  " A residual comparable to the terms means the instrument is being read,"
                  " not the ocean -- ours is one instantaneous snapshot, so some residual is"
                  " expected; a residual LARGER than the vertical term voids the comparison.")
    # NEMO
    def _load(fn, var):
        ds = nc.Dataset(fn)
        try:
            x = np.ma.filled(np.ma.masked_invalid(ds.variables[var][:]), np.nan).astype(np.float64)
            la = np.asarray(ds.variables["nav_lat"][:]); lo_ = np.asarray(ds.variables["nav_lon"][:]) % 360.0
            zz = np.asarray(ds.variables[[v for v in ds.variables if v.startswith("depth")][0]][:])
        finally:
            ds.close()
        return np.nanmean(_select_recs(x, a.nemo_w_recs), axis=0), la, lo_, zz
    ue, lat_u, lon_u, zu = _load(a.nemo_ufile, "uocetr_eff")
    ve, lat_v, lon_v, zv = _load(a.nemo_vfile, "vocetr_eff")
    wo, lat_w, lon_w, zw = _load(a.nemo_wfile, "wo")
    dsm = nc.Dataset(a.mesh_mask)
    try:
        e1t = np.squeeze(np.asarray(dsm.variables["e1t"][:], dtype=np.float64))
        e2t = np.squeeze(np.asarray(dsm.variables["e2t"][:], dtype=np.float64))
        lat_t = np.squeeze(np.asarray(dsm.variables["gphit"][:], dtype=np.float64))
        lon_t = np.squeeze(np.asarray(dsm.variables["glamt"][:], dtype=np.float64)) % 360.0
    finally:
        dsm.close()
    ue = np.where(np.isfinite(ue), ue, 0.0); ve = np.where(np.isfinite(ve), ve, 0.0)
    wo = np.where(np.isfinite(wo), wo, 0.0)
    nzn, nyn, nxn = ue.shape
    # NEMO output frames (331,360) vs mesh_mask (332,362): the file drops the
    # cyclic columns and the fold row.  The offset is CHECKED against grid_T's
    # own nav_lat/nav_lon, not assumed.
    dst = nc.Dataset(a.nemo_gridt)
    try:
        lat_f = np.asarray(dst.variables["nav_lat"][:]); lon_f = np.asarray(dst.variables["nav_lon"][:]) % 360.0
        tvar = next(v for v in ("to", "thetao", "votemper") if v in dst.variables)
        wet_f = np.isfinite(np.ma.filled(np.ma.masked_invalid(dst.variables[tvar][0, 0]), np.nan))
    finally:
        dst.close()
    best = None
    for r0 in (0, 1):
        for c0 in (0, 1, 2):
            sl_lat = lat_t[r0:r0 + nyn, c0:c0 + nxn]
            if sl_lat.shape != lat_f.shape:
                continue
            # land cells carry fill/zero coordinates in the output files:
            # compare on wet cells only
            d = float(np.nanmax(np.abs(sl_lat - lat_f)[wet_f]))
            if best is None or d < best[0]:
                best = (d, r0, c0)
    d, r0, c0 = best
    if d > 1e-3:
        raise SystemExit(f"NEMO frame offset not found (best |dlat| {d:.3e}); refusing")
    lat_t = lat_f; lon_t = lon_f
    e1t = e1t[r0:r0 + nyn, c0:c0 + nxn]; e2t = e2t[r0:r0 + nyn, c0:c0 + nxn]
    print(f"[frame] NEMO file = mesh_mask[{r0}:{r0+nyn}, {c0}:{c0+nxn}] (|dlat| max {d:.1e})")
    kln = (zu >= z0) & (zu < z1)
    z_ifn = np.concatenate([[0.0], 0.5 * (zu[:-1] + zu[1:])])
    ktn = int(np.argmin(np.abs(zw - z0))); kbn = int(np.argmin(np.abs(zw - z1)))
    # U(i) is the east face of T(i): div_x = U(i) - U(i-1); V(j) north face of T(j)
    dxn = ue[kln].sum(0); dxn = dxn - np.roll(dxn, 1, axis=1)
    dyn = ve[kln].sum(0); dyn = dyn - np.roll(dyn, 1, axis=0)
    boxn = (np.abs(lat_t) <= half) & (lon_t >= lo) & (lon_t < hi) & wet_f
    nx_ = float(dxn[boxn].sum()) / 1e6; ny_ = float(dyn[boxn].sum()) / 1e6
    area = e1t * e2t
    nwt = float((wo[ktn] * area)[boxn].sum()) / 1e6
    nwb = float((wo[kbn] * area)[boxn].sum()) / 1e6
    resn = nx_ + ny_ + (nwt - nwb)
    print(f"NEMO (uocetr_eff/vocetr_eff/wo, records {a.nemo_w_recs}): zonal out {nx_:+.3f}  "
          f"meridional out {ny_:+.3f}  horizontal net out {nx_+ny_:+.3f} | w_top(up) {nwt:+.3f} at "
          f"{zw[ktn]:.0f} m, w_bot(up) {nwb:+.3f} at {zw[kbn]:.0f} m => vertical net out {nwt-nwb:+.3f} | "
          f"residual {resn:+.3f} (wo positive up; {int(boxn.sum())} cells)")
    if a.heat_budget and hb_ours is not None:
        dst2 = nc.Dataset(a.nemo_gridt)
        try:
            tv = next(v for v in ("to", "thetao", "votemper") if v in dst2.variables)
            ton = np.ma.filled(np.ma.masked_invalid(dst2.variables[tv][:]), np.nan).astype(np.float64)
            ton = np.nanmean(_select_recs(ton, a.nemo_t_recs or a.nemo_w_recs), axis=0)
        finally:
            dst2.close()
        dsw2 = nc.Dataset(a.nemo_wfile)
        try:
            if "avt" not in dsw2.variables:
                raise SystemExit("--heat-budget needs `avt` in --nemo-wfile")
            avtn = np.ma.filled(np.ma.masked_invalid(dsw2.variables["avt"][:]), np.nan).astype(np.float64)
            avtn = np.nanmean(_select_recs(avtn, a.nemo_w_recs), axis=0)
        finally:
            dsw2.close()
        wet_n = (np.isfinite(ton) & (np.abs(ton) > 1e-6)).transpose(1, 2, 0)
        ton = np.where(np.isfinite(ton), ton, 0.0).transpose(1, 2, 0)
        avtn = np.where(np.isfinite(avtn), avtn, 0.0).transpose(1, 2, 0)
        uet = ue.transpose(1, 2, 0); vet = ve.transpose(1, 2, 0)
        wot = (wo * area[None]).transpose(1, 2, 0)
        Fw_n = np.concatenate([np.zeros_like(uet[:, :1]), uet], axis=1)
        Fs_n = np.concatenate([np.zeros_like(vet[:1]), vet], axis=0)
        Fup_n = np.concatenate([wot, np.zeros_like(wot[..., :1])], axis=-1)
        z_ifn2 = np.concatenate([zw, [2.0 * zw[-1] - zw[-2]]])
        hb_nemo = _heat_terms(Fw_n, Fs_n, Fup_n, ton, avtn[..., 1:], area,
                              zu, z_ifn2, kln, ktn, kbn, boxn, wet_n)
        dzn2 = np.diff(z_ifn2)[kln]
        wln = (wet_n[..., kln] * dzn2)
        tbn = float((ton[..., kln] * wln).sum(-1)[boxn].sum() / wln.sum(-1)[boxn].sum())
        print(f"NEMO: horizontal adv {hb_nemo[0]:+.3f}   vertical adv {hb_nemo[1]:+.3f}   "
              f"vertical diff {hb_nemo[2]:+.3f}   sum {sum(hb_nemo):+.3f}   "
              f"(layer mean T {tbn:.3f} C)")
        def _r(x, y):
            return float("nan") if abs(y) < 1e-12 else x / y
        print(f"RATIO ours/NEMO  horizontal {_r(hb_ours[0], hb_nemo[0]):+.3f}   "
              f"vertical {_r(hb_ours[1], hb_nemo[1]):+.3f}   "
              f"diffusive {_r(hb_ours[2], hb_nemo[2]):+.3f}")
        print("READ (pre-registered 2026-09-08): the claim that the cold 20 m is ADVECTIVE "
              "needs the vertical term to be both LEADING on our side and off NEMO's by more "
              "than the other two. A leading horizontal term, or a vertical ratio near 1, "
              "refutes it and points back at the mixing closure.")

    print("READ: a layer that is being HEAVED DOWN has horizontal CONVERGENCE (net out < 0) "
          "balanced by descent below it; compare the zonal and meridional columns to see which "
          "component differs from NEMO.  Ours is one snapshot; NEMO a 5-day mean.")


def _select_recs(x, spec):
    """Apply the --nemo-w-recs selection (int or A:B slice) to axis 0."""
    if spec is None:
        return x
    if ":" in spec:
        lo, hi = (int(s) for s in spec.split(":"))
        out = x[lo:hi]
    else:
        k = int(spec)
        out = x[k:k + 1]
    if out.shape[0] == 0:
        raise SystemExit(f"--nemo-w-recs {spec} selects no records")
    return out


def _euc_merid_block(a, L, zc):
    """Meridional w profile and EUC u(z): displaced-vs-absent ascent, and
    whether the undercurrent under it is intact.

    NATIVE-TO-NATIVE.  Our tripole runs NEMO's own eORCA1 mesh, so columns are
    selected by a coordinate box on each side's own coordinates -- no regrid,
    no interpolation convention.  Our u sits on T-point coords vs NEMO's
    U-point uo (half a cell east); at 1 degree that is far inside any structure
    read here.  NEMO's 5-day frame is (331,360) vs our (332,362): coordinate
    selection makes the frames irrelevant.
    """
    import netCDF4 as nc

    snap = np.load(a.legoesm_snapshot)
    lat_o = np.asarray(L["lat"], dtype=np.float64)
    lon_o = np.asarray(L["lon"], dtype=np.float64) % 360.0

    def box_mean(field2d, lat2, lon2, lat0, lon0, dlat, dlon):
        m = ((np.abs(lat2 - lat0) <= dlat)
             & (np.abs((lon2 - lon0 + 180.0) % 360.0 - 180.0) <= dlon))
        return np.nanmean(field2d[m]) if m.any() else np.nan

    # ---- SEA LEVEL ALONG THE EQUATOR: the pressure-gradient test ----------
    # An undercurrent is maintained by the zonal pressure gradient the wind
    # sets up by piling water in the west. Our eta and NEMO's ssh are both
    # already here, so this settles the gradient question without a momentum
    # budget. ONLY THE ZONAL GRADIENT OF THE DIFFERENCE IS MEANINGFUL -- the
    # two models carry different reference pressures, so the offset itself
    # says nothing (GLM).
    #
    # PRE-REGISTERED (GLM, before running): the difference rising eastward by
    # >= 5 cm between 220E and 260E means a missing eastward force
    # g*d(delta eta)/dx >= 1.1e-7 m/s^2, about 0.28 m/s per month, which is
    # commensurate with the measured 0.33 m/s core deficit at 220E -> the
    # pressure gradient IS the cause. An eastward trend <= 1 cm over 200-260E
    # implies under 0.04 m/s/month -> gradient EXONERATED.
    #
    # GUARD, also GLM's: eta is only the BAROTROPIC part. A flat difference
    # does not exonerate the gradient until the steric contribution from the
    # T(x,z) structure at 100-150 m is checked, which is why the thermocline
    # depths are printed alongside.
    if a.nemo_gridt:
        _ssh_o = np.asarray(snap["eta"], dtype=np.float64)
        _dsg = nc.Dataset(a.nemo_gridt)
        try:
            _cands = ("sshn", "ssh", "ssh_m", "zos", "sossheig")
            _nm = next((v for v in _cands if v in _dsg.variables), None)
            if _nm is None:
                print("\n[sea level] SKIPPED: none of "
                      f"{_cands} in {a.nemo_gridt}; available 2-D: "
                      + ", ".join(sorted(
                          v for v, o in _dsg.variables.items()
                          if o.ndim == 3))[:200])
            else:
                _sn = np.ma.filled(np.ma.masked_invalid(
                    _dsg.variables[_nm][:]), np.nan).astype(np.float64)
                _sn = _select_recs(_sn, a.nemo_w_recs)
                _sn = np.nanmean(_sn, axis=0) if _sn.ndim == 3 else _sn
                _latn = np.asarray(_dsg.variables["nav_lat"][:], np.float64)
                _lonn = np.asarray(_dsg.variables["nav_lon"][:],
                                   np.float64) % 360.0
                print(f"\nEQUATORIAL SEA LEVEL, |lat|<=1 box mean [m]. NEMO "
                      f"variable {_nm!r}. Read the GRADIENT of the "
                      f"difference, never its offset.")
                print(f"{'lon':>6} {'ours':>9} {'NEMO':>9} {'diff':>9}")
                _lons = (180, 200, 220, 240, 260, 280)
                _d = {}
                for lon0 in _lons:
                    o = box_mean(_ssh_o, lat_o, lon_o, 0.0, lon0, 1.0, 2.0)
                    n = box_mean(_sn, _latn, _lonn, 0.0, lon0, 1.0, 2.0)
                    _d[lon0] = o - n
                    print(f"{lon0:6d} {o:9.4f} {n:9.4f} {o - n:9.4f}")
                _r = _d.get(260, np.nan) - _d.get(220, np.nan)
                _r2 = _d.get(260, np.nan) - _d.get(200, np.nan)
                print(f"\n[sea level] difference change 220E->260E: "
                      f"{100.0 * _r:+.2f} cm; 200E->260E: {100.0 * _r2:+.2f} cm")
                if np.isfinite(_r):
                    _acc = 9.80665 * _r / (40.0 * 111e3 * np.cos(0.0))
                    print(f"[sea level] implied zonal acceleration difference "
                          f"{_acc:+.3e} m/s^2 = {_acc * 2.592e6:+.3f} m/s per "
                          f"30 days, against a measured core deficit of "
                          f"0.33 m/s at 220E.")
                print("[sea level] PRE-REGISTERED: >= +5 cm rise 220E->260E "
                      "CONFIRMS the pressure gradient as the cause; <= +1 cm "
                      "over 200E->260E REFUTES it. Between the two, report "
                      "the number and claim no direction. eta is BAROTROPIC "
                      "only -- do not exonerate without the steric term.")
        finally:
            _dsg.close()

    if a.meridional_lon and a.nemo_wfile:
        w_o = np.asarray(snap["mass_flux_w"], dtype=np.float64)
        z_if = np.concatenate([[0.0], 0.5 * (zc[:-1] + zc[1:])])
        kl = int(np.argmin(np.abs(z_if - a.w_depth_m)))
        ds = nc.Dataset(a.nemo_wfile)
        try:
            wn = np.ma.filled(np.ma.masked_invalid(
                ds.variables[a.nemo_w_var][:]), np.nan).astype(np.float64)
            zw = np.asarray(ds.variables["depthw"][:], dtype=np.float64)
            ln = "nav_lat" if "nav_lat" in ds.variables else "nav_lat_grid_T"
            lat_n = np.asarray(ds.variables[ln][:])
            lon_n = np.asarray(ds.variables[ln.replace("lat", "lon")][:]) % 360.0
        finally:
            ds.close()
        wn = np.nanmean(_select_recs(wn, a.nemo_w_recs), axis=0)
        kn = int(np.argmin(np.abs(zw - a.w_depth_m)))
        for lon0 in (float(s) for s in a.meridional_lon.split(",")):
            print(f"\nMeridional w at {lon0:.0f}E, {z_if[kl]:.1f} m, "
                  "1e-6 m/s (ours snapshot vs NEMO record mean):")
            print(f"{'lat':>6} {'ours':>10} {'NEMO':>10}")
            for lat0 in range(-8, 9):
                o = box_mean(w_o[..., kl], lat_o, lon_o, lat0, lon0, 0.5, 1.0)
                n = box_mean(wn[kn], lat_n, lon_n, lat0, lon0, 0.5, 1.0)
                print(f"{lat0:6d} {1e6 * o:10.3f} {1e6 * n:10.3f}")

    if a.nemo_vfile and a.meridional_lon:
        v_o = np.asarray(snap["v"], dtype=np.float64)
        if v_o.shape[0] == lat_o.shape[0] + 1:
            # C-grid v faces (ny+1, nx, nz): average to T cells.
            v_o = 0.5 * (v_o[:-1, :, :] + v_o[1:, :, :])
        if v_o.shape[:2] != lat_o.shape:
            raise SystemExit(f"v {v_o.shape} does not align with T coords "
                             f"{lat_o.shape} -- refusing to index.")
        ds = nc.Dataset(a.nemo_vfile)
        try:
            vn = np.ma.filled(np.ma.masked_invalid(
                ds.variables["vo"][:]), np.nan).astype(np.float64)
            zv = np.asarray(ds.variables["depthv"][:], dtype=np.float64)
            lat_nv = np.asarray(ds.variables["nav_lat"][:])
            lon_nv = np.asarray(ds.variables["nav_lon"][:]) % 360.0
        finally:
            ds.close()
        vn = np.nanmean(_select_recs(vn, a.nemo_w_recs), axis=0)
        kv_o = int(np.argmin(np.abs(zc - a.v_depth_m)))
        kv_n = int(np.argmin(np.abs(zv - a.v_depth_m)))
        for lon0 in (float(s) for s in a.meridional_lon.split(",")):
            print(f"\nMeridional surface current v at {lon0:.0f}E, "
                  f"{zc[kv_o]:.1f} m, m/s (+ = northward; the Ekman "
                  "divergence carrier):")
            print(f"{'lat':>6} {'ours':>8} {'NEMO':>8}")
            for lat0 in range(-6, 7):
                o = box_mean(v_o[..., kv_o], lat_o, lon_o, lat0, lon0, 0.5, 1.0)
                n = box_mean(vn[kv_n], lat_nv, lon_nv, lat0, lon0, 0.5, 1.0)
                print(f"{lat0:6d} {o:8.3f} {n:8.3f}")

    if a.nemo_ufile:
        if a.nemo_w_recs is None:
            # Without a record window the U block silently averaged all 18
            # GATEWAY records (days 1-90) against a single-day snapshot --
            # an unmatched comparison that read as "FESOM EUC 78% of NEMO".
            raise SystemExit("--nemo-ufile needs --nemo-w-recs (matched window); "
                             "refusing to average every record.")
        if lat_o.ndim == 1:
            # Unstructured grids (MPAS cells / FESOM nodes): the snapshot
            # carries the GEOGRAPHIC cell-centred zonal velocity written by
            # the runner (Perot / inverse-rotation); the raw ``u`` there is
            # edge-normal (MPAS) or absent (FESOM) and cannot be boxed.
            if "u_east" not in snap.files:
                raise SystemExit("unstructured snapshot without u_east -- "
                                 "rerun with a runner that writes it.")
            u_o = np.asarray(snap["u_east"], dtype=np.float64)
        else:
            u_o = np.asarray(snap["u"], dtype=np.float64)
            if u_o.shape[1] == lat_o.shape[1] + 1:
                # C-grid u faces (ny, nx+1, nz): average the two faces of each
                # T cell so the box selection below can use T coordinates.
                u_o = 0.5 * (u_o[:, :-1, :] + u_o[:, 1:, :])
        if u_o.shape[:lat_o.ndim] != lat_o.shape:
            raise SystemExit(f"u {u_o.shape} does not align with T coords "
                             f"{lat_o.shape} -- refusing to index.")
        ds = nc.Dataset(a.nemo_ufile)
        try:
            un = np.ma.filled(np.ma.masked_invalid(
                ds.variables["uo"][:]), np.nan).astype(np.float64)
            zu = np.asarray(ds.variables["depthu"][:], dtype=np.float64)
            lat_n = np.asarray(ds.variables["nav_lat"][:])
            lon_n = np.asarray(ds.variables["nav_lon"][:]) % 360.0
        finally:
            ds.close()
        un = np.nanmean(_select_recs(un, a.nemo_w_recs), axis=0)
        # EUC WIDTH (GLM review): 100x lateral viscosity predicts a
        # diffusively SMEARED jet (half-width >= 3-4 deg vs NEMO ~1-2); a
        # NARROW and weak jet would falsify the viscosity attribution.
        for lon0 in (200, 220, 240):
            # core depth from NEMO's own profile at this lon (same on both
            # sides so width is compared at one physical level)
            prof_n = np.array([box_mean(un[k], lat_n, lon_n, 0.0, lon0,
                                        1.0, 1.0) for k in range(zu.size)
                               if zu[k] <= 400.0])
            if not np.isfinite(prof_n).any():
                continue
            kc_n = int(np.nanargmax(prof_n))
            zcore = zu[kc_n]
            kc_o = int(np.argmin(np.abs(zc - zcore)))
            print(f"\nEUC meridional profile at {lon0}E, core depth "
                  f"{zcore:.0f} m (NEMO's core level), m/s:")
            print(f"{'lat':>6} {'ours':>8} {'NEMO':>8}")
            for lat0 in range(-6, 7):
                o = box_mean(u_o[..., kc_o], lat_o, lon_o, lat0, lon0, 0.5, 1.0)
                n = box_mean(un[kc_n], lat_n, lon_n, lat0, lon0, 0.5, 1.0)
                print(f"{lat0:6d} {o:8.3f} {n:8.3f}")

        print(f"\nEUC: equatorial zonal velocity, |lat|<=1 box mean, m/s "
              f"(NEMO records {a.nemo_w_recs}).")
        print("max over 0-400 m (core speed) and its depth; + = eastward.")
        print(f"{'lon':>6} {'ours_max':>9} {'@m':>5} {'nemo_max':>9} {'@m':>5} "
              f"{'ours_10m':>9} {'nemo_10m':>9}")
        _profiles = {}
        k400_o = zc <= 400.0
        k400_n = zu <= 400.0
        k10_o = int(np.argmin(np.abs(zc - 10.0)))
        k10_n = int(np.argmin(np.abs(zu - 10.0)))
        for lon0 in (160, 180, 200, 220, 240, 260):
            prof_o = np.array([box_mean(u_o[..., k], lat_o, lon_o, 0.0, lon0,
                                        1.0, 1.0) for k in np.nonzero(k400_o)[0]])
            prof_n = np.array([box_mean(un[k], lat_n, lon_n, 0.0, lon0,
                                        1.0, 1.0) for k in np.nonzero(k400_n)[0]])
            def mx(p, z):
                if not np.isfinite(p).any():
                    return np.nan, np.nan
                k = int(np.nanargmax(p))
                return p[k], z[k]
            mo, zo = mx(prof_o, zc[k400_o])
            mn, zn_ = mx(prof_n, zu[k400_n])
            s10_o = box_mean(u_o[..., k10_o], lat_o, lon_o, 0.0, lon0, 1.0, 1.0)
            s10_n = box_mean(un[k10_n], lat_n, lon_n, 0.0, lon0, 1.0, 1.0)
            print(f"{lon0:6d} {mo:9.3f} {zo:5.0f} {mn:9.3f} {zn_:5.0f} "
                  f"{s10_o:9.3f} {s10_n:9.3f}")
            _profiles[lon0] = (prof_o, zc[k400_o], prof_n, zu[k400_n])

        # THE MAX AND ITS DEPTH ARE NOT ENOUGH.  A single (speed, depth) pair
        # cannot say whether a deep maximum is a genuine subsurface JET or the
        # top of a broad deep drift that merely happens to be the largest
        # eastward value in the column -- and that distinction decides whether
        # a core-depth mismatch is a displaced jet or a missing one.  Both
        # reviewers asked for the full profiles, and the loop above already
        # computes them, so printing costs nothing.
        print("\nEQUATORIAL u(z), |lat|<=1 box mean, m/s; + = eastward.")
        print("Each side on ITS OWN levels -- the levels differ, so read the "
              "SHAPE (is there a subsurface maximum, and how sharp) rather "
              "than pairing rows.")
        # DEPTH-INTEGRATED TRANSPORT: does the column carry the same NET zonal
        # momentum, merely distributed differently, or is it actually missing?
        # That distinction is not visible in a core speed, and it is the one
        # that separates a vertical-transfer defect from a momentum SINK.
        # Independent of both closures -- it is just u integrated over depth.
        print("\nNET ZONAL TRANSPORT 0-400 m, |lat|<=1 box mean [m2/s]; "
              "+ = eastward.")
        print(f"{'lon':>6} {'ours':>9} {'NEMO':>9} {'ours-NEMO':>10}")
        for lon0 in sorted(_profiles):
            po, zo_, pn, zn2 = _profiles[lon0]
            _mo, _mn = np.isfinite(po), np.isfinite(pn)
            _io = float(np.trapezoid(po[_mo], zo_[_mo])) if _mo.sum() > 1 else np.nan
            _in = float(np.trapezoid(pn[_mn], zn2[_mn])) if _mn.sum() > 1 else np.nan
            print(f"{lon0:6d} {_io:9.2f} {_in:9.2f} {_io - _in:10.2f}")

        for lon0 in sorted(_profiles):
            po, zo_, pn, zn2 = _profiles[lon0]
            print(f"\n  {lon0}E   ours (depth m: u)          NEMO (depth m: u)")
            for i in range(max(po.size, pn.size)):
                lhs = (f"{zo_[i]:8.0f}:{po[i]:7.3f}"
                       if i < po.size and np.isfinite(po[i]) else " " * 16)
                rhs = (f"{zn2[i]:8.0f}:{pn[i]:7.3f}"
                       if i < pn.size and np.isfinite(pn[i]) else "")
                if lhs.strip() or rhs.strip():
                    print(f"    {lhs}      {rhs}")


def _upwelling_block(a, L, tgt_lat, tgt_lon, band, zc, regrid):
    """Equatorial vertical velocity, ours vs NEMO, at a single depth.

    WHY IT IS HERE.  Z20 says the thermocline is 26 m too deep; it cannot say
    whether that is because the water is not being lifted.  This is the cheapest
    measurement that can, and it runs on files that already exist.

    UNITS ARE NOT THE SAME ON THE TWO SIDES and that is the whole trap: NEMO's
    ``wocetr_eff`` is an effective vertical TRANSPORT [m3/s] on the W points,
    ours is ``mass_flux_w`` [m/s] (MASS_FLUX_W_UNITS, ocean_model_latlon_cgrid
    :120).  NEMO's is divided by the T-cell area e1t*e2t here so both sides are
    velocities before anything is compared.

    SIGN IS DERIVED, NOT ASSUMED.  Equatorial upwelling is a known answer: the
    Pacific cold tongue exists because water rises there.  So NEMO's band mean
    at this depth TELLS us NEMO's sign convention, and ours is then reported
    with the same test rather than flipped to match.  If the two disagree the
    block says so and stops short of a ratio -- a sign difference is either a
    convention difference or a reversed circulation, and this probe cannot tell
    those apart.

    WHAT IT CANNOT SUPPORT.  The two sides are at DIFFERENT TIMES: ours is an
    instantaneous snapshot from a 30/90-day run, NEMO's is a 24-hour mean at the
    end of its first year, because the monthly grid_T carries no vertical
    velocity at all.  A 10-20% difference is inside that mismatch and must not
    be read.  A factor of ~2 is not, and the Z20 gap (26 m on 55 m) is a
    factor-of-2-sized question.
    """
    import netCDF4 as nc
    from legoesm import constants  # noqa: F401  (unit conventions live there)

    w_l = np.load(a.legoesm_snapshot).get("mass_flux_w")
    if w_l is None:
        raise SystemExit("snapshot has no `mass_flux_w`; rerun with the mass "
                         "flux carry stored, or drop --nemo-wfile.")
    w_l = np.asarray(w_l, dtype=np.float64)
    # Interface depths: the snapshot stores CENTRES, so interfaces are their
    # midpoints with the surface at 0. Good to a few metres, which is far
    # inside the level spacing at 50 m and is only used to PICK a level.
    z_if = np.concatenate([[0.0], 0.5 * (zc[:-1] + zc[1:])])
    kl = int(np.argmin(np.abs(z_if - a.w_depth_m)))

    ds = nc.Dataset(a.nemo_wfile)
    try:
        wn = np.ma.filled(np.ma.masked_invalid(
            ds.variables[a.nemo_w_var][:]), np.nan).astype(np.float64)
        zw = np.asarray(ds.variables["depthw"][:], dtype=np.float64)
        # trd1h_T names its coords nav_lat_grid_T; the 5-day grid_W nav_lat.
        latname = ("nav_lat_grid_T" if "nav_lat_grid_T" in ds.variables
                   else "nav_lat")
        nav_lat = np.asarray(ds.variables[latname][:])
        nav_lon = np.asarray(ds.variables[latname.replace("lat", "lon")][:])
    finally:
        ds.close()
    wn = _select_recs(wn, a.nemo_w_recs)
    n_rec = wn.shape[0]
    wn = np.nanmean(wn, axis=0)
    kn = int(np.argmin(np.abs(zw - a.w_depth_m)))

    if a.nemo_w_var == "wocetr_eff":
        dsm = nc.Dataset(a.mesh_mask)
        try:
            e1t = np.asarray(dsm.variables["e1t"][:],
                             dtype=np.float64).squeeze()
            e2t = np.asarray(dsm.variables["e2t"][:],
                             dtype=np.float64).squeeze()
        finally:
            dsm.close()
        # wocetr_eff is written on the (331,360) inner frame; the mesh_mask is
        # the full (332,362). Slice with the SAME native window the rest of
        # this campaign uses rather than guessing an offset here.
        from global_tracer_content import _NATIVE_J, _NATIVE_I
        area = (e1t * e2t)[_NATIVE_J, _NATIVE_I]
        if area.shape != wn.shape[1:]:
            raise SystemExit(f"area {area.shape} vs wocetr_eff {wn.shape[1:]}"
                             " -- frames do not match; do not divide.")
        wn_ms = wn[kn] / area                        # m3/s -> m/s
    else:
        wn_ms = wn[kn]                               # wo is already m/s

    lm = np.isfinite(np.asarray(L["mask"], dtype=np.float64))
    ours, _ = regrid(w_l[..., kl], L["lat"], L["lon"],
                     np.asarray(L["mask"], dtype=np.float64),
                     tgt_lat, tgt_lon)
    theirs, _ = regrid(wn_ms, nav_lat, nav_lon,
                       np.isfinite(wn_ms).astype(np.float64),
                       tgt_lat, tgt_lon)
    del lm

    print()
    print(f"Equatorial vertical velocity at {z_if[kl]:.1f} m (ours) / "
          f"{zw[kn]:.1f} m (NEMO), band mean, m/s x 1e6.")
    src = ("wocetr_eff [m3/s] divided by e1t*e2t"
           if a.nemo_w_var == "wocetr_eff" else "wo [m/s]")
    print(f"  NEMO side is a {n_rec}-record mean of {src}; ours is "
          f"mass_flux_w [m/s] from the snapshot (instantaneous).")
    rows = np.nonzero(band)[0]
    box = (tgt_lon >= 180.0) & (tgt_lon <= 280.0)    # 180-80W, the cold tongue
    bo = np.nanmean(ours[np.ix_(rows, np.nonzero(box)[0])])
    bt = np.nanmean(theirs[np.ix_(rows, np.nonzero(box)[0])])
    print(f"{'lon':>6} {'ours':>11} {'NEMO':>11}")
    for lon in (160, 180, 200, 220, 240, 260, 280):
        j = int(np.argmin(np.abs(tgt_lon - lon)))
        print(f"{lon:6d} {1e6 * np.nanmean(ours[rows, j]):11.3f} "
              f"{1e6 * np.nanmean(theirs[rows, j]):11.3f}")
    print(f"[box 180-80W] ours {1e6 * bo:+.3f}  NEMO {1e6 * bt:+.3f}  (1e-6 m/s)")
    if not (np.isfinite(bo) and np.isfinite(bt)):
        print("[upwelling] one side is NaN in the box — no ratio.")
    elif np.sign(bo) != np.sign(bt):
        print("[upwelling] THE TWO SIGNS DISAGREE. NEMO's sign here is the "
              "known-answer anchor (the cold tongue requires ascent), so this "
              "is either an opposite w convention in our snapshot or a "
              "reversed equatorial cell. NO RATIO is reported: this probe "
              "cannot tell those apart, and the next step is to check the "
              "convention before reading anything into the magnitude.")
    else:
        print(f"[upwelling] ours / NEMO = {bo / bt:.2f} in the 180-80W box. "
              "Read only a LARGE departure from 1: ours is an instantaneous "
              "snapshot against a NEMO time mean, which is worth more than "
              "10-20%.")


if __name__ == "__main__":
    raise SystemExit(main())
