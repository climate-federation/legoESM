"""NEMO ORCA1 reference transports (AMOC@26N + ACC@Drake) for OMIP faithfulness.

Offline read of the NEMO ORCA1 ``grid_V`` (vo, e3v) / ``grid_U`` (uo, e3u) +
``domain_cfg`` (e1v/e2u, gphi*, glam*) — no mesh_mask needed.

AMOC: the Atlantic Meridional Overturning at a target latitude with the SAME
convention as the legoESM model-side diagnostic (``ocean.spinup.
compute_amoc_from_state`` / ``diagnostics_streamfunction.moc_streamfunction``):
ψ(z) = −∫_{surface}^{z} V_zonal dz' (cumsum top→down, negated), Atlantic-masked
by a longitude band, reported as the surface-referenced upper-mid peak.  RAPID
array obs ~17 Sv at 26.5 N.

ACC: net eastward volume transport through a single Drake-Passage meridian
section (per-row nearest-longitude U-column, depth- and latitude-integrated).
Physically equivalent to the model-side ``diagnostics_climate.acc_transport``
(ψ_bt max−min in the Drake band).  Obs ~137 Sv.  Units: Sv.

Usage:
    python scripts/validate/nemo_transports.py \
        --grid-v .../ORCA1_1y_..._grid_V.nc \
        --domain-cfg .../domain_cfg.nc [--target-lat 26.5] \
        [--grid-u .../ORCA1_1y_..._grid_U.nc]   # adds ACC@Drake
"""
from __future__ import annotations

import argparse

import numpy as np

_SV = 1.0e6   # m^3/s per Sverdrup
_PW = 1.0e15  # W per Petawatt
_RHO0 = 1025.0  # reference seawater density [kg/m^3]  (legoesm.constants.rho_ocean)
_CP = 3994.0    # specific heat of seawater [J/(kg*K)] (legoesm.constants.c_sw)


def _sq2(a):
    a = np.asarray(a)
    while a.ndim > 2:
        a = a[0]
    return a


def _sq3(a):
    a = np.asarray(a)
    while a.ndim > 3:
        a = a[0]
    return a


def _tsel(da, time_idx):
    """Select one time_counter record (matched-window comparisons against a
    single-month spin-up state) or the mean over all records (time_idx None,
    the prior behaviour)."""
    if "time_counter" not in da.dims:
        return da
    if time_idx is None:
        return da.mean("time_counter")
    return da.isel(time_counter=int(time_idx))


def amoc_core(voe3, e1v, gphiv, glamv, depthv, *, target_lat=26.5,
              lon_min=-75.0, lon_max=15.0):
    """Atlantic MOC strength [Sv] at ``target_lat`` from PURE arrays (no I/O).

    Parameters
    ----------
    voe3 : (z, y, x)  time-mean meridional volume-flux density vo*e3v [m/s·m].
    e1v : (y, x)      v-face zonal width [m].
    gphiv, glamv : (y, x)  v-face latitude / longitude [deg].
    depthv : (z,)     level depths [m], positive down.

    Returns a dict with ``amoc_Sv`` (sign-agnostic surface-referenced peak
    overturning in the upper-mid column), ``row_lat_deg``, ``j``,
    ``depth_of_max_m``.  Split out from the NetCDF reader so it is unit-testable
    on a synthetic analytic overturning.
    """
    voe3 = np.nan_to_num(np.asarray(voe3, dtype=np.float64), nan=0.0)
    e1v = np.asarray(e1v, dtype=np.float64)
    gphiv = np.asarray(gphiv, dtype=np.float64)
    glamv = np.asarray(glamv, dtype=np.float64)
    depthv = np.asarray(depthv, dtype=np.float64).ravel()

    lon_w = ((glamv + 180.0) % 360.0) - 180.0
    if lon_min <= lon_max:
        atl = (lon_w >= lon_min) & (lon_w <= lon_max)
    else:                                                  # wrap-around band
        atl = (lon_w >= lon_min) | (lon_w <= lon_max)
    atl = atl.astype(np.float64)                          # (y, x)

    # Zonally-integrated thickness-weighted meridional volume flux per (z, y).
    flux = voe3 * (e1v * atl)[None, :, :]                 # (z, y, x)
    Vx = flux.sum(axis=2)                                 # (z, y)
    psi = -np.cumsum(Vx, axis=0) / _SV                    # (z, y) [Sv], top->down

    # Atlantic-mean latitude per y-row; pick the row nearest the target.
    with np.errstate(invalid="ignore"):
        lat_y = np.nansum(gphiv * atl, axis=1) / np.maximum(atl.sum(axis=1), 1)
    j = int(np.argmin(np.abs(lat_y - target_lat)))
    # SIGN-AGNOSTIC peak overturning: reference ψ to the surface (removes any net
    # barotropic throughflow offset so ψ_surface = 0), then take the
    # largest-magnitude excursion in the UPPER-MID column (depth < 3000 m, to
    # exclude the deep AABW cell).  Reports physical AMOC strength regardless of
    # ψ sign convention — NEMO's ψ here is positive-peaked whereas the legoESM
    # moc_streamfunction is negative-peaked (same formula, opposite vo/cumsum
    # orientation), so a fixed -min/+max would disagree.
    prof = psi[:, j] - psi[0, j]
    upper = depthv < 3000.0
    seg = prof[upper] if np.any(upper) else prof
    kmax = int(np.nanargmax(np.abs(seg)))
    return {"amoc_Sv": float(abs(seg[kmax])), "row_lat_deg": float(lat_y[j]),
            "j": j,
            "depth_of_max_m": float((depthv[upper] if np.any(upper) else depthv)[kmax])}


def mht_core(voe3, theta_v, e1v, gphiv, *, rho0=_RHO0, cp=_CP):
    """Global meridional ocean heat transport MHT(lat) [PW] from PURE arrays.

    MHT(y) = rho0 * cp * Σ_x Σ_z (voe3 · theta_v · e1v)   [W]

    with ``voe3 = vo*e3v`` the meridional volume-flux density at v-points,
    ``theta_v`` the potential temperature [degC] averaged onto the SAME v-points,
    and ``e1v`` the v-face zonal width.  The FULL zonal integral at a latitude has
    ~zero net mass flux (mass conservation), so the transport is essentially
    reference-independent and degC is the conventional choice.  Split out from the
    NetCDF reader so it is unit-testable on a synthetic flow.  All inputs must be
    pre-aligned onto the v-grid (same z,y,x shape; e1v/gphiv on the v-rows).

    Returns a dict: ``mht_PW``/``lat_deg`` (the full curve), and the diagnostic
    scalars ``nh_peak_PW``/``nh_peak_lat`` (max over lat>0) and ``sh_min_PW``/
    ``sh_min_lat`` (min over lat<0).  NH peak obs ~1.8 PW (poleward), SH min
    ~−1 PW (poleward = southward).
    """
    voe3 = np.nan_to_num(np.asarray(voe3, dtype=np.float64), nan=0.0)
    theta_v = np.nan_to_num(np.asarray(theta_v, dtype=np.float64), nan=0.0)
    e1v = np.asarray(e1v, dtype=np.float64)
    gphiv = np.asarray(gphiv, dtype=np.float64)

    # Per-(z,y,x) heat flux [W]; sum over depth and longitude -> per-y curve.
    hf = rho0 * cp * voe3 * theta_v * e1v[None, :, :]      # (z, y, x) [W]
    mht_PW = hf.sum(axis=(0, 2)) / _PW                     # (y,) [PW]
    with np.errstate(invalid="ignore"):
        lat_y = np.nanmean(gphiv, axis=1)                 # mean lat per v-row
    fin = np.isfinite(mht_PW) & np.isfinite(lat_y)        # finite-aware peak
    nh = (lat_y > 0.0) & fin
    sh = (lat_y < 0.0) & fin
    if np.any(nh):
        nh_i = int(np.argmax(np.where(nh, mht_PW, -np.inf)))
        nh_peak, nh_lat = float(mht_PW[nh_i]), float(lat_y[nh_i])
    else:
        nh_peak = nh_lat = float("nan")
    if np.any(sh):
        sh_i = int(np.argmin(np.where(sh, mht_PW, np.inf)))
        sh_min, sh_lat = float(mht_PW[sh_i]), float(lat_y[sh_i])
    else:
        sh_min = sh_lat = float("nan")
    return {"mht_PW": mht_PW, "lat_deg": lat_y,
            "nh_peak_PW": nh_peak, "nh_peak_lat": nh_lat,
            "sh_min_PW": sh_min, "sh_min_lat": sh_lat}


def nemo_mht(grid_v_path, grid_t_path, domain_cfg_path, *, time_idx=None):
    """Global meridional ocean heat transport [PW] from NEMO grid_V + grid_T."""
    import xarray as xr
    dV = xr.open_dataset(grid_v_path, decode_times=False)
    dT = xr.open_dataset(grid_t_path, decode_times=False)
    dc = xr.open_dataset(domain_cfg_path, decode_times=False)

    vname = "vo" if "vo" in dV else ("voce" if "voce" in dV else None)
    if vname is None:
        raise KeyError("grid_V has neither 'vo' nor 'voce'")
    vo_da = dV[vname]
    if "e3v" in dV:
        voe3_da = vo_da * dV["e3v"]
        voe3_da = _tsel(voe3_da, time_idx)
        voe3 = _sq3(voe3_da.values)                        # (z, y, x) v-points
    else:
        vo_da = _tsel(vo_da, time_idx)
        voe3 = _sq3(vo_da.values) * _sq3(dc["e3v_0"].values)
    tname = next((n for n in ("thetao", "toce", "votemper", "to") if n in dT), None)
    if tname is None:
        raise KeyError("grid_T has none of 'thetao'/'toce'/'votemper'/'to'")
    t_da = dT[tname]
    t_da = _tsel(t_da, time_idx)
    theta_t = _sq3(t_da.values)                            # (z, y, x) T-points
    e1v = _sq2(dc["e1v"].values)
    gphiv = _sq2(dc["gphiv"].values)

    # Average T -> the v-row BETWEEN T(j) and T(j+1): NEMO vo[j] sits north of
    # T(j), so theta_v[j] = 0.5*(theta[j] + theta[j+1]).  Align every field onto
    # the ny-1 interior v-rows.
    theta_v = 0.5 * (theta_t[:, :-1, :] + theta_t[:, 1:, :])   # (z, ny-1, x)
    voe3_a = voe3[:, :-1, :]
    e1v_a = e1v[:-1, :]
    gphiv_a = gphiv[:-1, :]
    return mht_core(voe3_a, theta_v, e1v_a, gphiv_a)


def acc_drake_core(uoe3, e2u, gphiu, glamu, *, drake_lon=-68.0,
                   lat_south=-65.0, lat_north=-45.0, max_lon_dev_deg=5.0):
    """Drake-passage ACC zonal volume transport [Sv] from PURE arrays (no I/O).

    Net EASTWARD volume flux (SIGNED, eastward positive) through a FIXED model
    i-column (a meridian) at the Drake-Passage longitude, summed over depth and
    the passage latitude band.

    Why a fixed column (not a per-row nearest-longitude pick): a single i-column
    is a CONTIGUOUS section with NO staircase gap, so it measures the net zonal
    flux exactly.  This is valid because ORCA1 is a REGULAR lat-lon grid in the
    Southern Ocean (the tripole fold is north of ~50 N), where a constant i IS a
    meridian.  That regularity is ASSERTED, not assumed: every band row's
    circular longitude at the chosen column must stay within ``max_lon_dev_deg``
    of ``drake_lon`` — otherwise the grid is curvilinear there and a fixed-i
    section would mis-sample, so the function RAISES rather than silently report
    a plausible-but-wrong number.

    Parameters
    ----------
    uoe3 : (z, y, x)  time-mean zonal volume-flux density uo*e3u [m/s·m].
    e2u : (y, x)      u-face meridional width [m].
    gphiu, glamu : (y, x)  u-face latitude / longitude [deg].

    Returns a dict with ``acc_Sv`` (SIGNED net section transport — eastward
    positive, so a reversed U convention or wrong section surfaces as a sign
    flip / implausible value rather than being hidden by abs), ``n_rows``,
    ``section_lon_deg`` (mean section longitude), ``i_col`` (the fixed column),
    ``max_lon_dev_deg`` (the regularity-check residual).  Split out from the
    NetCDF reader so it is unit-testable on a synthetic analytic flow.
    Physically equivalent to the model-side ``acc_transport`` (ψ_bt max−min over
    the same Drake lat band) for the total ACC throughflow; the band default
    ([-65,-45]) matches ``diagnostics_climate.acc_transport`` for apples-to-
    apples comparison.  Drake obs ~137 Sv.
    """
    uoe3 = np.nan_to_num(np.asarray(uoe3, dtype=np.float64), nan=0.0)
    e2u = np.asarray(e2u, dtype=np.float64)
    gphiu = np.asarray(gphiu, dtype=np.float64)
    glamu = np.asarray(glamu, dtype=np.float64)

    # Circular longitude distance to a normalized drake_lon (so e.g. 292 == -68).
    drake_w = ((float(drake_lon) + 180.0) % 360.0) - 180.0
    lon_w = ((glamu + 180.0) % 360.0) - 180.0                  # (y, x) [-180,180)
    cdist = np.abs(((lon_w - drake_w + 180.0) % 360.0) - 180.0)  # (y, x) circular

    band = (gphiu >= lat_south) & (gphiu <= lat_north)         # (y, x)
    band_rows = np.where(band.any(axis=1))[0]
    if band_rows.size == 0:
        return {"acc_Sv": float("nan"), "n_rows": 0, "i_col": -1,
                "section_lon_deg": float("nan"), "max_lon_dev_deg": float("nan")}

    # Fixed meridian column from the band-centre row (nearest the Drake longitude
    # among that row's in-band cells), then the actual band rows AT that column.
    j_mid = int(band_rows[band_rows.size // 2])
    cd_mid = np.where(band[j_mid], cdist[j_mid], np.inf)
    i_col = int(np.argmin(cd_mid))
    rows = np.where(band[:, i_col])[0]
    if rows.size == 0:                                         # degenerate column
        return {"acc_Sv": float("nan"), "n_rows": 0, "i_col": i_col,
                "section_lon_deg": float("nan"), "max_lon_dev_deg": float("nan")}

    # Regularity assertion: the fixed column must stay near the Drake meridian
    # over the whole band, else a constant-i section is not a meridian here.
    max_dev = float(np.max(cdist[rows, i_col]))
    if max_dev > max_lon_dev_deg:
        raise ValueError(
            f"Drake section column i={i_col} deviates {max_dev:.1f} deg from "
            f"drake_lon={drake_w:.1f} over lat [{lat_south},{lat_north}] "
            f"(grid not regular here) -> a fixed-i meridian section is invalid; "
            f"a curvilinear section path with V-face connectors is required.")

    # Depth-integrated zonal transport along the section: Σ_z uo*e3u * e2u.
    Udz = uoe3[:, rows, i_col].sum(axis=0) * e2u[rows, i_col]  # (n_rows,) [m^3/s]
    transport = float(Udz.sum())
    return {"acc_Sv": transport / _SV, "n_rows": int(rows.size), "i_col": i_col,
            "section_lon_deg": float(np.mean(lon_w[rows, i_col])),
            "max_lon_dev_deg": max_dev}


def nemo_acc_drake(grid_u_path, domain_cfg_path, *, time_idx=None, drake_lon=-68.0,
                   lat_south=-65.0, lat_north=-45.0):
    """Drake-passage ACC transport [Sv] from NEMO grid_U + domain_cfg."""
    import xarray as xr
    dU = xr.open_dataset(grid_u_path, decode_times=False)
    dc = xr.open_dataset(domain_cfg_path, decode_times=False)

    uname = "uo" if "uo" in dU else ("uoce" if "uoce" in dU else None)
    if uname is None:
        raise KeyError("grid_U has neither 'uo' nor 'uoce'")
    uo_da = dU[uname]
    # Time-mean the TRANSPORT (uo*e3u), not uo and e3u separately (free surface).
    if "e3u" in dU:
        uoe3_da = uo_da * dU["e3u"]
        uoe3_da = _tsel(uoe3_da, time_idx)
        uoe3 = _sq3(uoe3_da.values)
    else:                                                      # static e3u_0
        uo_da = _tsel(uo_da, time_idx)
        uoe3 = _sq3(uo_da.values) * _sq3(dc["e3u_0"].values)
    e2u = _sq2(dc["e2u"].values)
    gphiu = _sq2(dc["gphiu"].values)
    glamu = _sq2(dc["glamu"].values)
    return acc_drake_core(uoe3, e2u, gphiu, glamu, drake_lon=drake_lon,
                          lat_south=lat_south, lat_north=lat_north)


def nemo_amoc_at_latitude(grid_v_path, domain_cfg_path, *, time_idx=None, target_lat=26.5,
                          lon_min=-75.0, lon_max=15.0):
    """Atlantic MOC strength [Sv] at ``target_lat`` from NEMO grid_V + domain_cfg."""
    import xarray as xr
    dV = xr.open_dataset(grid_v_path, decode_times=False)
    dc = xr.open_dataset(domain_cfg_path, decode_times=False)

    vname = "vo" if "vo" in dV else ("voce" if "voce" in dV else None)
    if vname is None:
        raise KeyError("grid_V has neither 'vo' nor 'voce'")
    vo_da = dV[vname]
    # Time-mean the TRANSPORT (vo*e3v), not vo and e3v separately — under a
    # free surface e3v varies in time, so mean(vo)*mean(e3v) != mean(vo*e3v).
    if "e3v" in dV:
        voe3_da = vo_da * dV["e3v"]                        # aligned (t,z,y,x)
        voe3_da = _tsel(voe3_da, time_idx)
        voe3 = _sq3(voe3_da.values)                        # (z, y, x) = mean(vo*e3v)
    else:                                                  # static e3v_0
        vo_da = _tsel(vo_da, time_idx)
        voe3 = _sq3(vo_da.values) * _sq3(dc["e3v_0"].values)
    e1v = _sq2(dc["e1v"].values)                          # (y, x)
    gphiv = _sq2(dc["gphiv"].values)                      # (y, x)
    glamv = _sq2(dc["glamv"].values)                      # (y, x)
    depthv = np.asarray(dV["depthv"].values).ravel()      # (z,) positive down
    return amoc_core(voe3, e1v, gphiv, glamv, depthv,
                     target_lat=target_lat, lon_min=lon_min, lon_max=lon_max)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-v", required=True)
    p.add_argument("--grid-u", default=None,
                   help="ORCA1 grid_U (uo,e3u) — enables ACC@Drake.")
    p.add_argument("--grid-t", default=None,
                   help="ORCA1 grid_T (thetao) — enables global MHT(lat).")
    p.add_argument("--domain-cfg", required=True)
    p.add_argument("--target-lat", type=float, default=26.5)
    p.add_argument("--lon-min", type=float, default=-75.0)
    p.add_argument("--lon-max", type=float, default=15.0)
    p.add_argument("--drake-lon", type=float, default=-68.0)
    p.add_argument("--drake-lat-south", type=float, default=-65.0)
    p.add_argument("--drake-lat-north", type=float, default=-45.0)
    p.add_argument("--time-idx", type=int, default=None,
                   help="Single time_counter record to use (e.g. 2 = March of "
                        "year 1 in a monthly Jan-first file) for matched-"
                        "window spin-up comparisons; default = mean over all "
                        "records (the prior behaviour).")
    a = p.parse_args()
    r = nemo_amoc_at_latitude(a.grid_v, a.domain_cfg, time_idx=a.time_idx,
                              target_lat=a.target_lat,
                              lon_min=a.lon_min, lon_max=a.lon_max)
    print(f"[NEMO] AMOC@{a.target_lat}N (row lat {r['row_lat_deg']:.2f}, "
          f"j={r['j']}) = {r['amoc_Sv']:.2f} Sv  (RAPID obs ~17)")
    if a.grid_u is not None:
        ac = nemo_acc_drake(a.grid_u, a.domain_cfg, time_idx=a.time_idx,
                            drake_lon=a.drake_lon,
                            lat_south=a.drake_lat_south,
                            lat_north=a.drake_lat_north)
        print(f"[NEMO] ACC@Drake (section lon {ac['section_lon_deg']:.1f}, "
              f"i={ac['i_col']}, {ac['n_rows']} rows, "
              f"lon-dev {ac['max_lon_dev_deg']:.2f} deg) = {ac['acc_Sv']:.2f} Sv "
              f"(eastward +; obs ~137)")
    if a.grid_t is not None:
        mh = nemo_mht(a.grid_v, a.grid_t, a.domain_cfg, time_idx=a.time_idx)
        print(f"[NEMO] MHT NH peak = {mh['nh_peak_PW']:.2f} PW @ "
              f"{mh['nh_peak_lat']:.1f}N, SH min = {mh['sh_min_PW']:.2f} PW @ "
              f"{mh['sh_min_lat']:.1f}N  (NH obs ~1.8 PW)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
