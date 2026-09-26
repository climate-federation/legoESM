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


def arctic_gateways_core(uoe3, voe3, e2u, e1v, gphit, glamt, tmask_top,
                         umask_3d=None, vmask_3d=None):
    """Per-gateway volume transport [Sv] across the lat >= 66N Arctic boundary.

    Scored with the MODEL's OWN section machinery
    (``legoesm.ocean.diagnostics_sections``), so the oracle and the model are
    measured on an identical section and no numerics are duplicated here.  The
    only thing this function does is translate the C-grid INDEX convention.

    NEMO -> legoESM face indexing.  NEMO's ``u(j,i)`` is the face EAST of
    ``T(j,i)`` and ``v(j,i)`` the face NORTH of it; ``region_boundary_faces``
    puts our u-face ``i`` WEST of cell ``i`` and our v-face ``j`` SOUTH of cell
    ``j``.  So our index is NEMO's plus one.  MEASURED off the mesh, not read
    from documentation: ``glamu - glamt = +0.5000`` deg at four consecutive
    equatorial columns, and a great-circle test puts ``e1u[i]`` on
    dist(T_i, T_i+1) rather than dist(T_i-1, T_i) in every band where the two
    are distinguishable, INCLUDING across the tripole fold.  This is the same
    convention the grid loader now uses (fixed in 15c6670ae).

    The section is NOT a constant-j row: eORCA1 is a tripole, so a j-row is not
    a latitude circle.  The region mask handles that.

    Returns ``(gateways_Sv, diag)``.  Sign: POSITIVE = INTO the Arctic.
    """
    import jax.numpy as jnp

    from legoesm.ocean.diagnostics_sections import (
        arctic_region_mask, gateway_face_masks, region_boundary_faces,
        section_transport)

    uoe3 = np.asarray(uoe3, dtype=np.float64)
    voe3 = np.asarray(voe3, dtype=np.float64)
    # Land arrives as a fill value that xarray decodes to NaN.  section_transport
    # multiplies UNSELECTED faces by a zero weight, and 0 * NaN is NaN, so a
    # single land cell anywhere would turn every gateway into NaN while any
    # spot-check on a different field still looked fine (codex).  Zero them
    # here, where "not wet" genuinely means "no transport".
    # "Wet" must be the 3-D FACE mask, not the surface one.  Below the seafloor
    # NEMO writes fill values on columns that are wet AT THE SURFACE, so a
    # surface mask broadcast over depth calls ~1e6 sub-bathymetry cells "wet
    # with a missing value" and aborts a perfectly good run -- which is exactly
    # what happened on the first real-data attempt.
    if umask_3d is None or vmask_3d is None:
        dry_u = np.broadcast_to(~(np.asarray(tmask_top) > 0.5),
                                np.asarray(uoe3).shape)
        dry_v = np.broadcast_to(~(np.asarray(tmask_top) > 0.5),
                                np.asarray(voe3).shape)
        print("[NEMO] WARNING: no 3-D face masks supplied; the missing-value "
              "check uses the SURFACE mask and will be over-strict at depth.")
    else:
        dry_u = ~(np.asarray(umask_3d) > 0.5)
        dry_v = ~(np.asarray(vmask_3d) > 0.5)
    bad_u = ~np.isfinite(uoe3)
    bad_v = ~np.isfinite(voe3)
    # Zero ONLY where the cell is genuinely dry.  Blanket zeroing would turn a
    # MISSING WET transport into a plausible finite answer that the finiteness
    # check below could never catch (codex), so a non-finite value on a wet
    # cell is treated as a defect and refused.
    wet_bad_u = int(np.count_nonzero(bad_u & ~dry_u))
    wet_bad_v = int(np.count_nonzero(bad_v & ~dry_v))
    # WHERE the residue sits decides how to treat it.  Measured on the ORCA1
    # gateway output: 0.054% of u and 0.062% of v, spread over latitudes
    # -55..+76 but concentrated in the UPPER 30 levels and peaking near k=12,
    # which is the signature of a mesh-vintage bathymetry difference (shelf
    # edges masked in the run but wet in mesh_mask), not of missing ocean data
    # -- a genuine data fault would not track depth level at all.
    #
    # So the residue is tolerated GLOBALLY but refused where it actually
    # matters: if any face this section SELECTS carries a missing value, the
    # transport through it would silently vanish, so that is fatal.  The check
    # therefore has to happen after the faces are known, not before.
    if wet_bad_u or wet_bad_v:
        tot = np.asarray(uoe3).size
        for tag, bad, dry in (("u", bad_u, dry_u), ("v", bad_v, dry_v)):
            sel_bad = bad & ~dry
            n = int(np.count_nonzero(sel_bad))
            if not n:
                continue
            k, j, _i = np.nonzero(sel_bad)
            lat = np.asarray(gphit)[j, _i]
            per_k = np.bincount(k, minlength=sel_bad.shape[0])
            worst = np.argsort(per_k)[::-1][:5]
            print(f"[NEMO] {tag}: {n} cells the mesh calls wet carry no value "
                  f"({100.0*n/tot:.4f}% of the field); levels {k.min()}-"
                  f"{k.max()}, lat {lat.min():.1f}..{lat.max():.1f}; worst " +
                  ", ".join(f"k={int(w)}:{int(per_k[w])}" for w in worst))
        print("[NEMO] treating these as a mesh-vintage bathymetry difference; "
              "the section's OWN faces are checked separately below.")

    n_nan_u = int(np.count_nonzero(bad_u))
    n_nan_v = int(np.count_nonzero(bad_v))
    uoe3 = np.where(bad_u, 0.0, uoe3)
    voe3 = np.where(bad_v, 0.0, voe3)

    nz, ny, nx = voe3.shape
    mfu = np.zeros((ny, nx + 1, nz))
    mfu[:, 1:, :] = np.moveaxis(uoe3, 0, -1)
    # our u-face 0 lies between cell nx-1 and cell 0 = NEMO's u at i = nx-1,
    # which after the shift is stored at our index nx (never selected itself).
    mfu[:, 0, :] = mfu[:, nx, :]
    mfv = np.zeros((ny + 1, nx, nz))
    mfv[1:, :, :] = np.moveaxis(voe3, 0, -1)

    dy_u = np.zeros((ny, nx + 1))
    dy_u[:, 1:] = np.asarray(e2u)
    dy_u[:, 0] = np.asarray(e2u)[:, -1]
    dx_v = np.zeros((ny + 1, nx))
    dx_v[1:, :] = np.asarray(e1v)

    region = arctic_region_mask(jnp.asarray(gphit), jnp.asarray(tmask_top))
    faces = region_boundary_faces(region)
    gates = gateway_face_masks(faces, jnp.asarray(glamt))

    # THE CHECK THAT PROTECTS THE NUMBER: no face this section selects may
    # carry a missing value, or its transport silently vanishes from a gateway.
    _nx = nx
    sel_u = np.asarray(faces.u_sel)[:, 1:_nx + 1]      # our face i <- NEMO i-1
    sel_v = np.asarray(faces.v_sel)[1:, :]             # our face j <- NEMO j-1
    # non-finite AND WET.  Using bad_u alone would flag essentially every
    # ocean column, because every column has fill values below its own
    # seafloor -- those are legitimately dry, not missing.
    hit_u = int(np.count_nonzero((bad_u & ~dry_u).any(axis=0) & sel_u))
    hit_v = int(np.count_nonzero((bad_v & ~dry_v).any(axis=0) & sel_v))
    n_sel_u, n_sel_v = int(sel_u.sum()), int(sel_v.sum())
    if hit_u or hit_v:
        # These faces are wet in mesh_mask but masked in the run, so the run
        # carried NO transport through them and this sum does the same.  That
        # is an ASSUMPTION, not a fact -- their true value is unknowable from
        # this file -- so it is stated, and bounded below, rather than hidden
        # behind a tolerance knob.
        print(f"[NEMO] {hit_u} of {n_sel_u} u-faces and {hit_v} of {n_sel_v} "
              "v-faces on the section are wet in the mesh but masked in the "
              "run; counted as ZERO transport, and bounded per gateway below.")
    else:
        print(f"[NEMO] section-face check PASSED: none of the {n_sel_u} "
              f"u-faces and {n_sel_v} v-faces this section selects carries a "
              "missing value")
    missing_face = ((bad_u & ~dry_u).any(axis=0) & sel_u,
                    (bad_v & ~dry_v).any(axis=0) & sel_v)

    out = {}
    for name, gf in gates.items():
        st = section_transport(jnp.asarray(mfu), jnp.asarray(mfv),
                               jnp.asarray(dy_u), jnp.asarray(dx_v), gf)
        vol = float(st.volume)
        if not np.isfinite(vol):
            raise SystemExit(
                f"FATAL: gateway {name!r} returned a non-finite transport. "
                "Do not quote any number from this run.")
        out[name] = vol / _SV

    # Bound what the masked faces could have carried, per gateway: their count
    # times the MEDIAN magnitude of that gateway's other faces.  A bound of a
    # few percent means the sector number stands; a large one means it does
    # not, and the reader can see which without any threshold being chosen here.
    miss_u, miss_v = missing_face
    for name, gf in gates.items():
        gu = np.asarray(gf.u_sel)[:, 1:nx + 1] & miss_u
        gv = np.asarray(gf.v_sel)[1:, :] & miss_v
        n_miss = int(gu.sum() + gv.sum())
        if not n_miss:
            continue
        wv = (np.asarray(gf.v_sel)[1:, :].astype(float)
              * np.asarray(dx_v)[1:, :])
        per_face = np.abs(np.moveaxis(voe3, 0, -1).sum(-1) * wv)
        typical = float(np.median(per_face[per_face > 0])) if (per_face > 0).any() else 0.0
        print(f"[NEMO]   {name}: {n_miss} masked face(s), bound on their "
              f"possible contribution ~{n_miss * typical / _SV:.4f} Sv")

    # SHIFT SENSITIVITY, reported not gated.  An earlier version of this probe
    # "validated" the staggering by demanding that no selected face carry an
    # all-zero NEMO column.  That check was WRONG (codex): the boundary of a wet
    # region legitimately includes wet-Arctic/LAND coastline faces, whose NEMO
    # transport is correctly zero, so it would have aborted a perfectly good
    # run.  What is informative instead is how much the answer MOVES if the
    # index shift is dropped -- if that is negligible the section is insensitive
    # to the convention and the result is robust either way; if it is large, the
    # convention is load-bearing and the reader should know.
    mfu_uns = np.zeros_like(mfu)
    mfu_uns[:, :nx, :] = np.moveaxis(uoe3, 0, -1)
    mfv_uns = np.zeros_like(mfv)
    mfv_uns[:ny, :, :] = np.moveaxis(voe3, 0, -1)
    # The counterfactual must drop the WHOLE conversion, metrics included --
    # pairing unshifted transports with shifted metrics would measure only
    # transport misregistration against fixed geometry, a narrower thing than
    # the label claims (codex).
    dy_u_uns = np.zeros_like(dy_u)
    dy_u_uns[:, :nx] = np.asarray(e2u)
    dx_v_uns = np.zeros_like(dx_v)
    dx_v_uns[:ny, :] = np.asarray(e1v)
    unshifted = {}
    for name, gf in gates.items():
        st = section_transport(jnp.asarray(mfu_uns), jnp.asarray(mfv_uns),
                               jnp.asarray(dy_u_uns), jnp.asarray(dx_v_uns), gf)
        unshifted[name] = float(st.volume) / _SV

    v_sel = np.asarray(faces.v_sel)[1:ny, :]
    diag = {
        "n_masked_section_u": hit_u,
        "n_masked_section_v": hit_v,
        "n_v_faces": int(v_sel.sum()),
        "n_region_cells": int(np.asarray(region).sum()),
        "net_Sv": float(sum(out.values())),
        "n_nonfinite_u": n_nan_u,
        "n_nonfinite_v": n_nan_v,
        "unshifted_Sv": unshifted,
    }
    return out, diag


def nemo_arctic_gateways(grid_u_path, grid_v_path, mesh_path, *, time_idx=None,
                         mesh_slice=None):
    """Arctic 66N gateway transports [Sv] from NEMO grid_U/grid_V + mesh_mask.

    ``mesh_path`` is the eORCA1 ``mesh_mask`` (``domain_cfg`` is absent from the
    ORCA1 EXP00 tree).  It is also the mesh the legoESM tripole runs, so both
    sides use identical metrics -- the controlled choice, not merely the
    available one.

    The mesh carries a halo the model output does not: mesh_mask is
    ``(332, 362)`` while grid_U/grid_V are ``(331, 360)``.  Rather than assume a
    slice, the interior is DERIVED by matching the output shape and then
    verified against the coordinate duplication, and a mismatch raises.
    """
    import xarray as xr

    dU = xr.open_dataset(grid_u_path, decode_times=False)
    dV = xr.open_dataset(grid_v_path, decode_times=False)
    dm = xr.open_dataset(mesh_path, decode_times=False)

    def _pick(ds, names, what):
        for n in names:
            if n in ds:
                return n
        raise KeyError(f"{what}: none of {names} in {list(ds.variables)[:40]}")

    uname = _pick(dU, ("uo", "uoce"), "grid_U zonal velocity")
    vname = _pick(dV, ("vo", "voce"), "grid_V meridional velocity")
    # Time-mean the TRANSPORT (u*e3u), never u and e3u separately: under a free
    # surface e3 varies in time and mean(u)*mean(e3) != mean(u*e3).  A missing
    # e3 is a DIFFERENT QUANTITY (reference rather than live thickness), so it
    # raises instead of silently substituting (codex).
    if "e3u" not in dU:
        raise SystemExit(f"FATAL: {grid_u_path} has no e3u. Using the reference "
                         "thickness e3u_0 would measure a different quantity.")
    if "e3v" not in dV:
        raise SystemExit(f"FATAL: {grid_v_path} has no e3v. Using the reference "
                         "thickness e3v_0 would measure a different quantity.")
    uoe3 = _sq3(_tsel(dU[uname] * dU["e3u"], time_idx).values)
    voe3 = _sq3(_tsel(dV[vname] * dV["e3v"], time_idx).values)
    ny_out, nx_out = voe3.shape[-2:]

    glamt_full = _sq2(dm["glamt"].values)
    my, mx = glamt_full.shape
    if my < ny_out or mx < nx_out:
        raise SystemExit(f"FATAL: mesh {(my, mx)} is smaller than the output "
                         f"{(ny_out, nx_out)}.")

    # WHICH WINDOW of the mesh is the model output?  Do not derive this from
    # shape arithmetic: on eORCA1 the first attempt (drop the south row, take
    # columns 1..360) disagreed with the output's own coordinates by up to
    # 145 deg, and only a coordinate check caught it.  So SEARCH the few
    # candidate offsets and pick the one that actually matches nav_lon, and
    # fail loudly listing what was tried if none does.
    nav = _sq2(dV["nav_lon"].values) if "nav_lon" in dV else None
    if nav is None:
        raise SystemExit("FATAL: the output has no nav_lon, so the mesh window "
                         "cannot be verified; refusing to guess.")
    row_varies = np.nanstd(nav, axis=1) > 1.0e-3      # row 0 is a constant -1
    if int(row_varies.sum()) < 2:
        raise SystemExit("FATAL: nav_lon has too few varying rows to identify "
                         "the mesh window.")
    # Score on the MEDIAN over LOW-LATITUDE rows, never the maximum.  Near the
    # tripole fold longitude is degenerate -- meridians converge and the branch
    # cut is arbitrary -- so a max is dominated by cells where any longitude
    # comparison is meaningless.  A first version of this check used the max
    # and rejected EVERY candidate window at ~144 deg, which was the degenerate
    # cells talking, not a real mismatch.  Same trap as the relative-metric
    # blow-up at near-zero cell widths.
    nav_lat = _sq2(dV["nav_lat"].values) if "nav_lat" in dV else None
    usable = row_varies[:, None] & np.isfinite(nav)
    if nav_lat is not None:
        usable = usable & (np.abs(nav_lat) < 60.0)
    if not usable.any():
        raise SystemExit("FATAL: no usable low-latitude cells to identify the "
                         "mesh window.")

    def _mismatch(j0, i0):
        w = glamt_full[j0:j0 + ny_out, i0:i0 + nx_out]
        if w.shape != (ny_out, nx_out):
            return np.inf
        d = ((w - nav + 180.0) % 360.0) - 180.0
        sel = usable & np.isfinite(d)
        return float(np.nanmedian(np.abs(d[sel]))) if sel.any() else np.inf

    if mesh_slice is None:
        cands = [(j0, i0) for j0 in range(my - ny_out + 1)
                 for i0 in range(mx - nx_out + 1)]
        scored = sorted(((_mismatch(j0, i0), j0, i0) for j0, i0 in cands))
        best, j0, i0 = scored[0]
        if not np.isfinite(best) or best > 1.0e-3:
            tried = ", ".join(f"({a},{b})->{m:.3e}" for m, a, b in scored[:6])
            raise SystemExit(
                "FATAL: no mesh window matches the output's nav_lon. Tried "
                f"{tried}. The mesh and the output are not the same domain, "
                "or nav_lon is unusable; refusing to quote any transport.")
        mesh_slice = (slice(j0, j0 + ny_out), slice(i0, i0 + nx_out))
        print(f"[NEMO] mesh window {mesh_slice} identified from nav_lon "
              f"(median |dlon| = {best:.2e} deg over the "
              f"low-latitude rows)")
    chk = glamt_full[mesh_slice]
    if chk.shape != (ny_out, nx_out):
        raise SystemExit(f"FATAL: mesh slice gives {chk.shape}, output is "
                         f"{(ny_out, nx_out)}; align them explicitly.")

    tmask_top = _sq3(dm[_pick(dm, ("tmask",), "mesh tmask")].values)[0][mesh_slice]
    e2u = _sq2(dm["e2u"].values)[mesh_slice]
    e1v = _sq2(dm["e1v"].values)[mesh_slice]
    gphit = _sq2(dm["gphit"].values)[mesh_slice]
    def _mask3(name):
        if name not in dm:
            return None
        m = np.asarray(dm[name].values)
        while m.ndim > 3:
            m = m[0]
        return m[(slice(None),) + mesh_slice]

    umask_3d, vmask_3d = _mask3("umask"), _mask3("vmask")
    if umask_3d is None or vmask_3d is None:
        print("[NEMO] WARNING: mesh has no umask/vmask; falling back to the "
              "surface tmask for the missing-value check.")
    gw, diag = arctic_gateways_core(uoe3, voe3, e2u, e1v, gphit, chk, tmask_top,
                                    umask_3d=umask_3d, vmask_3d=vmask_3d)
    diag["bruteforce"] = arctic_gateways_bruteforce(
        voe3, e1v, gphit, chk, tmask_top)
    return gw, diag


def arctic_gateways_bruteforce(voe3, e1v, gphit, glamt, tmask_top,
                               lat_min=66.0):
    """INDEPENDENT recomputation of the same crossing, no model machinery.

    GLM asked for the one check that would most raise confidence in the oracle
    number, and this is it: a deliberately dumb sum over the v-faces where the
    ``lat_min`` line is crossed, in NEMO's OWN indexing, with NEMO's own
    metrics -- no index translation, no region mask, no face-selection code,
    nothing shared with :func:`arctic_gateways_core`.  Agreement between the
    two validates the translation, the masking, the sign and the thickness
    handling all at once; disagreement means one of them is wrong and NEITHER
    may be quoted.

    NEMO's ``v(j)`` is the face NORTH of ``T(j)``, so the crossing faces are
    exactly those with ``gphit[j] < lat_min <= gphit[j+1]``.  A meridional
    transport through a face is ``vo * e3v`` (already depth-weighted) times the
    face's ZONAL width ``e1v``.  Sign: NEMO's v is positive northward, and
    north of the line is inside the region, so positive = INTO the region --
    the same convention the other path reports.
    """
    voe3 = np.nan_to_num(np.asarray(voe3, dtype=np.float64), nan=0.0,
                         posinf=0.0, neginf=0.0)
    e1v = np.asarray(e1v, dtype=np.float64)
    gphit = np.asarray(gphit, dtype=np.float64)
    lon180 = ((np.asarray(glamt) + 180.0) % 360.0) - 180.0
    wet = np.asarray(tmask_top) > 0.5

    south_out = gphit[:-1, :] < lat_min          # T(j) outside
    north_in = gphit[1:, :] >= lat_min           # T(j+1) inside
    crossing = south_out & north_in & wet[:-1, :] & wet[1:, :]

    trans = voe3[:, :-1, :].sum(axis=0) * e1v[:-1, :]    # (ny-1, nx) m^3/s
    # attribute each face to the bin of the cell INSIDE the region, matching
    # the other path's rule
    lon_inside = lon180[1:, :]

    from legoesm.ocean.diagnostics_sections import ARCTIC_GATEWAYS

    out = {}
    for name, bins in ARCTIC_GATEWAYS:
        sel = np.zeros_like(crossing)
        for lo, hi in bins:
            sel |= crossing & (lon_inside >= lo) & (lon_inside < hi)
        out[name] = float(np.where(sel, trans, 0.0).sum()) / _SV
    return out, int(crossing.sum())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-v", required=True)
    p.add_argument("--grid-u", default=None,
                   help="ORCA1 grid_U (uo,e3u) — enables ACC@Drake.")
    p.add_argument("--grid-t", default=None,
                   help="ORCA1 grid_T (thetao) — enables global MHT(lat).")
    p.add_argument("--domain-cfg", default=None,
                   help="ORCA1 domain_cfg — required for AMOC/ACC/MHT.")
    p.add_argument("--mesh", default=None,
                   help="eORCA1 mesh_mask — required by --arctic-gateways.")
    p.add_argument("--arctic-gateways", action="store_true",
                   help="Volume transport per gateway across the 66N Arctic "
                        "boundary, on the SAME section the model's own online "
                        "diagnostic uses. Positive = INTO the Arctic. Needs "
                        "--grid-u and --mesh; skips the other diagnostics.")
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
    if a.arctic_gateways:
        if a.grid_u is None or a.mesh is None:
            p.error("--arctic-gateways needs --grid-u and --mesh")
        gw, diag = nemo_arctic_gateways(a.grid_u, a.grid_v, a.mesh,
                                        time_idx=a.time_idx)
        print(f"[NEMO] Arctic 66N gateways, + = INTO the Arctic "
              f"({diag['n_region_cells']} region cells, "
              f"{diag['n_v_faces']} v-faces)")
        for name, sv in gw.items():
            print(f"[NEMO]   {name:<18s} {sv:+8.4f} Sv")
        print(f"[NEMO]   {'NET':<18s} {diag['net_Sv']:+8.4f} Sv")
        if diag["n_nonfinite_u"] or diag["n_nonfinite_v"]:
            print(f"[NEMO] land fill values zeroed before summing: "
                  f"{diag['n_nonfinite_u']} in u, {diag['n_nonfinite_v']} in v")
        # Sensitivity, REPORTED not gated.  If dropping the NEMO->legoESM index
        # shift barely moves a gateway, that gateway is insensitive to the
        # convention; if it moves it a lot, the convention is load-bearing and
        # the reader should weigh the result accordingly.
        print("[NEMO] sensitivity to the index shift (same section, shift "
              "dropped):")
        for name, sv in gw.items():
            u = diag["unshifted_Sv"][name]
            print(f"[NEMO]   {name:<18s} shifted {sv:+8.4f}  unshifted "
                  f"{u:+8.4f}  delta {sv - u:+8.4f} Sv")
        # INDEPENDENT CROSS-CHECK (GLM's "single check that would most raise
        # confidence"): the same crossing summed in NEMO's own indexing with
        # NEMO's own metrics, sharing no code with the path above.  Agreement
        # validates the index translation, the masking, the sign and the
        # thickness handling at once.  Disagreement means one of them is wrong
        # and NEITHER may be quoted.
        bf, n_faces = diag["bruteforce"]
        print(f"[NEMO] independent recomputation ({n_faces} crossing v-faces, "
              "no model machinery, no index translation):")
        worst = 0.0
        for name, sv in gw.items():
            b = bf[name]
            worst = max(worst, abs(sv - b))
            print(f"[NEMO]   {name:<18s} sections {sv:+8.4f}  brute force "
                  f"{b:+8.4f}  delta {sv - b:+8.4f} Sv")
        print(f"[NEMO] worst disagreement {worst:.4f} Sv")
        return 0
    if a.domain_cfg is None:
        p.error("--domain-cfg is required unless --arctic-gateways is given")
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
