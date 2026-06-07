"""NEMO ORCA1 reference transports (AMOC@26N) for the OMIP-faithful comparison.

Offline read of the NEMO ORCA1 ``grid_V`` (vo, e3v) + ``domain_cfg`` (e1v, gphiv,
glamv) — no mesh_mask needed.  Computes the Atlantic Meridional Overturning at a
target latitude with the SAME convention as the legoESM model-side diagnostic
(``ocean.spinup.compute_amoc_from_state`` / ``diagnostics_streamfunction.
moc_streamfunction``): ψ(z) = −∫_{surface}^{z} V_zonal dz' (cumsum top→down,
negated), AMOC = −min_z ψ at the latitude row nearest the target, Atlantic-masked
by a longitude band.  Units: Sv.  RAPID array obs ~17 Sv at 26.5 N.

Usage:
    python scripts/validate/nemo_transports.py \
        --grid-v .../ORCA1_1y_..._grid_V.nc \
        --domain-cfg .../domain_cfg.nc [--target-lat 26.5]
"""
from __future__ import annotations

import argparse

import numpy as np

_SV = 1.0e6  # m^3/s per Sverdrup


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


def nemo_amoc_at_latitude(grid_v_path, domain_cfg_path, *, target_lat=26.5,
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
        if "time_counter" in voe3_da.dims:
            voe3_da = voe3_da.mean("time_counter")
        voe3 = _sq3(voe3_da.values)                        # (z, y, x) = mean(vo*e3v)
    else:                                                  # static e3v_0
        if "time_counter" in vo_da.dims:
            vo_da = vo_da.mean("time_counter")
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
    p.add_argument("--domain-cfg", required=True)
    p.add_argument("--target-lat", type=float, default=26.5)
    p.add_argument("--lon-min", type=float, default=-75.0)
    p.add_argument("--lon-max", type=float, default=15.0)
    a = p.parse_args()
    r = nemo_amoc_at_latitude(a.grid_v, a.domain_cfg, target_lat=a.target_lat,
                              lon_min=a.lon_min, lon_max=a.lon_max)
    print(f"[NEMO] AMOC@{a.target_lat}N (row lat {r['row_lat_deg']:.2f}, "
          f"j={r['j']}) = {r['amoc_Sv']:.2f} Sv  (RAPID obs ~17)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
