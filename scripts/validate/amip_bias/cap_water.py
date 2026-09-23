#!/usr/bin/env python3
"""Polar-cap water and isolation numbers from MPAS checkpoints against ERA5.

Two views of the same question -- is the Arctic dry because water is removed
or because it never arrives -- read straight from checkpoints (full 3D state)
so that every day with a checkpoint can be scored, not only published months:

  profile   q_v, ERA5 q, their ratio, RH w.r.t. liquid (model and ERA5) and the
            T bias at reference pressures, area-weighted over lat >= --lat-lo.
  loop      the self-isolation loop's numbers per checkpoint: the
            surface-pressure dome (85-90N minus 70-80N, and minus 60-70N as
            "edge"; flat ocean cells in both datasets, model minus ERA5), the
            925 hPa water ratio and RH_liq over >= 75N, the 925 hPa T bias over
            >= 75N, and the 850 hPa zonal wind over 70-90N next to ERA5's.

ERA5 = monthly climatology 1979-2014 of the run's calendar month (ta, hus,
ps, ua; the month comes from the run's start date plus the day count on a
365-day calendar), hus converted to mixing ratio, sampled at the model's
cells.  The model is interpolated in log-pressure per column to each
reference pressure; a level below a column's lowest mid-level (or above its
top) is left out, and the SAME cells are left out of the ERA5 mean at that
level, so both sides of every ratio and bias use one cell set.  The dome uses
ocean cells whose model surface height is below --dome-max-z (default 5 m),
so a surface-height difference cannot pose as a pressure difference.  Cell
order is proven by the checkpoint's own column index; edge winds are used
only when that index is the identity.

Usage:
  cap_water.py profile <run>:<day> [...] [--lat-lo 75]
  cap_water.py loop    <run>:<day> [...]
  cap_water.py transport <run>:<day> [...] [--bins --mass --era5-q]
  cap_water.py era5 --era5-dir DIR --dates 1979-01-07,1979-01-16   (same bins from ERA5 daily analyses)
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import pathlib
import sys

import numpy as np

from legoesm.diagnostics.column_integrals import column_water_vapor

_VAL = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_VAL))
_spec = importlib.util.spec_from_file_location("cloud_layers", _VAL / "cloud_layers.py")
cl = importlib.util.module_from_spec(_spec)
sys.modules["cloud_layers"] = cl
_spec.loader.exec_module(cl)
rb = cl.rb

PLEV = np.array([1000, 925, 850, 700, 600, 500, 400, 300], float) * 100.0
ERA5_YEARS = ("1979-01-01", "2014-12-31")
ERA5_MIN_MONTHS = 30


_NOLEAP_MONTH_ENDS = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def run_month(exp, day):
    """Calendar month of model ``day``: the driver maps day 0 to Jan 1 of a
    365-day (noleap) year and ``start_day`` is a day offset (0.0 in AMIP)."""
    doy = int(np.floor(float(exp.get("start_day") or 0.0) + day)) % 365     # 0-based day of year
    return int(np.searchsorted(_NOLEAP_MONTH_ENDS, doy, side="right")) + 1


def paired_area_mean(model, ref, area, mask):
    """Area means of ``model`` and ``ref`` over the SAME cells: those inside
    ``mask`` where BOTH values are finite.  Returns (model, ref, retained
    area fraction of the mask)."""
    model, ref = np.asarray(model, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    ok = mask & np.isfinite(model) & np.isfinite(ref)
    if not ok.any():
        raise SystemExit("FATAL: no supported column in the mask")
    w = area[ok]
    return (float((model[ok] * w).sum() / w.sum()), float((ref[ok] * w).sum() / w.sum()),
            float(w.sum() / area[mask].sum()))


def columns_to_plev(p_full, field, plev):
    """Interpolate a top-down (ncol, nlev) field in log-p to ``plev`` per
    column; NaN where the target lies below that column's lowest level."""
    out = np.full((p_full.shape[0], np.size(plev)), np.nan)
    lt = np.log(np.atleast_1d(plev))
    for i in range(p_full.shape[0]):
        ok = (lt <= np.log(p_full[i, -1])) & (lt >= np.log(p_full[i, 0]))   # inside the column's mid-levels
        out[i, ok] = np.interp(lt[ok], np.log(p_full[i]), field[i])
    return out


def area_mean(x, area, mask):
    """Area-weighted mean of the finite entries of ``x`` inside ``mask``
    (x may be (ncol,) or (ncol, k); returns a scalar or (k,))."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        ok = mask & np.isfinite(x)
        if not ok.any():
            raise SystemExit("FATAL: empty area mean")
        return float((x[ok] * area[ok]).sum() / area[ok].sum())
    return np.array([area_mean(x[:, k], area, mask) for k in range(x.shape[1])])


def era5_month(var, month):
    """ERA5 monthly climatology of ``var`` for ``month``, lat ascending,
    lon in [0, 360)."""
    import xarray as xr
    from heating_budget import ERA5
    files = sorted(glob.glob(f"{ERA5}/{var}/*.nc"))
    if not files:
        raise SystemExit(f"FATAL: no ERA5 monthly {var} under {ERA5}")
    if len(files) != 1:
        raise SystemExit(f"FATAL: expected one ERA5 monthly file for {var}, found {len(files)}")
    d = xr.open_dataset(files[0])[var].sel(time=slice(*ERA5_YEARS))
    d = d[d.time.dt.month == month]
    if d.time.size < ERA5_MIN_MONTHS:
        raise SystemExit(f"FATAL: ERA5 {var}: only {d.time.size} samples of month {month}")
    d = d.mean("time")
    d = d.rename({("lat" if "lat" in d.dims else "latitude"): "lat",
                  ("lon" if "lon" in d.dims else "longitude"): "lon"})
    if float(d.lon.max()) <= 180.0:
        d = d.assign_coords(lon=(d.lon % 360)).sortby("lon")
    return d.sortby("lat")


def on_cells(d, lat, lon):
    """Nearest ERA5 grid value at every cell; the longitude axis is padded
    cyclically first so a cell near 360 can pick the 0-degree column."""
    import xarray as xr
    first, last = d.isel(lon=0), d.isel(lon=-1)
    d = xr.concat([last.assign_coords(lon=float(last.lon) - 360.0), d,
                   first.assign_coords(lon=float(first.lon) + 360.0)], "lon")
    return d.sel(lat=xr.DataArray(lat, dims="c"), lon=xr.DataArray(lon, dims="c"),
                 method="nearest", tolerance=3.0).values



def load_state(run, day):
    z = np.load(f"{rb.ROOT}/{run}/checkpoint_day_{day:04d}.npz", allow_pickle=True)
    exp = json.load(open(f"{rb.ROOT}/{run}/experiment_config.json"))
    lat, lon, area = cl.mesh_coords(exp)
    order = cl.cell_order(z, lat.size)
    from legoesm import constants
    if "day" not in z.files or abs(float(z["day"]) - day) > 1e-6:
        raise SystemExit(f"FATAL: {run} day {day}: checkpoint day stamp mismatch")
    st = {k: np.asarray(z[k], dtype=np.float64)[order] for k in ("T", "p_s", "trc_q_v")}
    vg = np.asarray(z["meta_vgrid"], dtype=np.float64)
    p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * st["p_s"][:, None]
    if not np.all(np.diff(p_half, axis=1) > 0):
        raise SystemExit(f"FATAL: {run} day {day}: pressure not increasing top-down")
    st["p_full"] = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    st["dp"] = np.diff(p_half, axis=1)
    for k in ("trc_q_c", "trc_q_i"):
        st[k] = np.asarray(z[k], dtype=np.float64)[order] if k in z.files else None
    for k in ("T", "trc_q_v"):
        if not np.isfinite(st[k]).all() or st[k].shape != st["p_full"].shape:
            raise SystemExit(f"FATAL: {run} day {day}: bad {k}")
    st["u_edge"] = np.asarray(z["u"]) if "u" in z.files else None
    st["order_is_identity"] = bool(np.array_equal(order, np.arange(order.size)))
    st["z_sfc"] = np.asarray(z["phis"], dtype=np.float64)[order] / constants.g if "phis" in z.files else None
    st["month"] = run_month(exp, day)
    st["exp"] = exp
    return st, lat, lon, area, order


def profile(args):
    from legoesm.thermo import saturation_mixing_ratio
    specs = [(s.split(":")[0], int(s.split(":")[1])) for s in args.specs]
    states = [load_state(r, d) for r, d in specs]
    lat, lon, area = states[0][1:4]
    if len({int(s[0]["exp"]["grid"]["resolution"]) for s in states}) != 1:
        raise SystemExit("FATAL: checkpoints are on different meshes; one table per mesh")
    months = {s[0]["month"] for s in states}
    if len(months) != 1:
        raise SystemExit(f"FATAL: checkpoints span different months {months}; one table per month")
    month = months.pop()
    ta, hus = era5_month("ta", month), era5_month("hus", month)
    plev_e = ta.plev.values
    Te = np.stack([on_cells(ta.sel(plev=p), lat, lon) for p in plev_e], 1)     # (ncol, nplev_e)
    qe = np.stack([on_cells(hus.sel(plev=p), lat, lon) for p in plev_e], 1)
    qe = qe / (1.0 - qe)                                                        # specific -> mixing ratio
    o = np.argsort(plev_e)
    Te = columns_to_plev(np.broadcast_to(plev_e[o], (lat.size, o.size)), Te[:, o], PLEV)
    qe = columns_to_plev(np.broadcast_to(plev_e[o], (lat.size, o.size)), qe[:, o], PLEV)
    mask = (lat >= args.lat_lo) & (lat < args.lat_hi)
    rhe_full = qe / np.asarray(saturation_mixing_ratio(np.where(np.isfinite(Te), Te, 250.0), PLEV[None, :]))
    print(f"band {args.lat_lo:g}-{args.lat_hi:g}N vs ERA5 month {month} clim; every ratio/bias over the SAME cells "
          f"(model interpolated in log-p per column, cells without that level dropped on BOTH sides; "
          f"ERA5 columns shown are the mean over the FIRST run's supported cells; 'kept' = area fraction)")
    print(f"{'hPa':>5s} {'kept':>5s} {'T_ERA5':>7s} {'q_ERA5':>7s} {'RH_ERA5':>7s} | "
          + " ".join(f"{s:>30s}" for s in args.specs))
    rows = []
    for st, *_ in states:
        T = columns_to_plev(st["p_full"], st["T"], PLEV)
        q = columns_to_plev(st["p_full"], st["trc_q_v"], PLEV)
        rh = q / np.asarray(saturation_mixing_ratio(np.where(np.isfinite(T), T, 250.0), PLEV[None, :]))
        per_level = []
        for k in range(PLEV.size):
            ok = mask & np.isfinite(q[:, k]) & np.isfinite(qe[:, k]) & np.isfinite(T[:, k]) & np.isfinite(Te[:, k])
            if not ok.any():
                per_level.append(None)                        # level unsupported in this cap
                continue
            tm, te, kept = paired_area_mean(np.where(ok, T[:, k], np.nan), Te[:, k], area, mask)
            qmk, qek, _ = paired_area_mean(np.where(ok, q[:, k], np.nan), qe[:, k], area, mask)
            rhm, rhk, _ = paired_area_mean(np.where(ok, rh[:, k], np.nan), rhe_full[:, k], area, mask)
            per_level.append((tm - te, qmk, qmk / qek, rhm, te, qek, rhk, kept))
        rows.append(per_level)
    for k, p in enumerate(PLEV):
        r0 = next((r[k] for r in rows if r[k] is not None), None)   # ERA5 columns from the first run that has the level
        if r0 is None:
            print(f"{p/100:5.0f} {0.0:5.2f}  (no supported column in the cap at this level, any run)")
            continue
        line = f"{p/100:5.0f} {r0[7]:5.2f} {r0[4]:7.1f} {r0[5]*1e3:7.3f} {r0[6]:7.2f} | "
        line += " ".join((f"dT{r[k][0]:+6.1f} q{r[k][1]*1e3:6.3f} ({r[k][2]:4.2f}x) RH{r[k][3]:5.2f} k{r[k][7]:4.2f}"
                          if r[k] is not None else f"{'unsupported':>30s}") for r in rows)
        print(line)
    return 0


def loop(args):
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm.grids.voronoi import reconstruct_cell_velocity
    from legoesm.grids.factory import create_grid
    import jax.numpy as jnp
    import xarray as xr
    print("dome = p_s anomaly (model - ERA5, flat ocean cells) 85-90N minus 70-80N; "
          "dome_edge = minus 60-70N (few flat ocean cells there: Nordic/Barents seas only)")
    print(f"{'run:day':>14s} {'dome[hPa]':>10s} {'edge[hPa]':>10s} {'q925/ERA5':>10s} {'RH925':>6s} "
          f"{'dT925[K]':>10s} {'u850 70-90N':>12s} {'ERA5 u850':>10s} {'prw>=72.5N':>11s} {'prw>=75N':>9s}")
    cache, meshes = {}, {}
    for spec in args.specs:
        run, day = spec.split(":")
        st, lat, lon, area, order = load_state(run, int(day))
        m, res = st["month"], int(st["exp"]["grid"]["resolution"])
        if (run, m) not in cache:
            sf = xr.open_dataset(glob.glob(f"{rb.ROOT}/{run}/cmor/fx/sftlf_*.nc")[0])["sftlf"]
            if float(sf.lon.max()) <= 180.0:
                sf = sf.assign_coords(lon=(sf.lon % 360)).sortby("lon")
            ocean = on_cells(sf.sortby("lat"), lat, lon) < 50
            pse = on_cells(era5_month("ps", m), lat, lon)
            qe = on_cells(era5_month("hus", m).sel(plev=92500.0), lat, lon); qe = qe / (1.0 - qe)
            Te925 = on_cells(era5_month("ta", m).sel(plev=92500.0), lat, lon)
            ue850 = on_cells(era5_month("ua", m).sel(plev=85000.0), lat, lon)
            cache[(run, m)] = (ocean, pse, qe, Te925, ue850)
        if res not in meshes:
            meshes[res] = create_grid("mpas", resolution=res)
        mesh = meshes[res]
        ocean, pse, qe, Te925, ue850 = cache[(run, m)]
        if st["z_sfc"] is None:
            raise SystemExit(f"FATAL: {spec}: checkpoint carries no surface geopotential")
        flat = ocean & (np.abs(st["z_sfc"]) < args.dome_max_z) & np.isfinite(pse) & (pse > 95000.0)
        # pse > 950 hPa screens ERA5 cells that sit on terrain (no ERA5 orography on disk)
        pole = area_mean(st["p_s"] - pse, area, (lat >= 85) & flat) / 100.0
        dome = pole - area_mean(st["p_s"] - pse, area, (lat >= 70) & (lat < 80) & flat) / 100.0
        dome_edge = pole - area_mean(st["p_s"] - pse, area, (lat >= 60) & (lat < 70) & flat) / 100.0
        q925 = columns_to_plev(st["p_full"], st["trc_q_v"], 92500.0)[:, 0]
        T925 = columns_to_plev(st["p_full"], st["T"], 92500.0)[:, 0]
        cap = lat >= 75
        qm, qek, kept = paired_area_mean(q925, qe, area, cap)
        rh = area_mean(q925 / np.asarray(saturation_mixing_ratio(np.where(np.isfinite(T925), T925, 250.0), 92500.0)), area, cap)
        tm, tek, _ = paired_area_mean(T925, Te925, area, cap)
        if st["u_edge"] is None:
            raise SystemExit(f"FATAL: {spec}: checkpoint carries no edge wind")
        if not st["order_is_identity"]:
            raise SystemExit(f"FATAL: {spec}: columns are permuted; edge winds cannot be mapped")
        if st["u_edge"].shape[0] != np.asarray(mesh.dvEdge).shape[0]:
            raise SystemExit(f"FATAL: {spec}: edge count {st['u_edge'].shape[0]} != mesh")
        u_east, _v_north = reconstruct_cell_velocity(jnp.asarray(st["u_edge"]), mesh)   # documented order
        u850 = columns_to_plev(st["p_full"], np.asarray(u_east, dtype=np.float64), 85000.0)[:, 0]
        u, uref, kept_u = paired_area_mean(u850, ue850, area, lat >= 70)
        dk = tuple(area[(lat >= lo) & (lat < hi) & flat].sum() / area[(lat >= lo) & (lat < hi)].sum()
                   for lo, hi in ((85, 90), (70, 80), (60, 70)))
        prw = np.asarray(column_water_vapor(st["trc_q_v"], st["p_s"], None, dp=st["dp"]), dtype=np.float64)
        prw725, prw75 = area_mean(prw, area, lat >= 72.5), area_mean(prw, area, cap)
        print(f"{spec:>14s} {dome:+10.1f} {dome_edge:+10.1f} {qm/qek:10.2f} {rh:6.2f} {tm-tek:+10.1f} {u:+12.1f} "
              f"{uref:+10.1f} {prw725:11.2f} {prw75:9.2f} (area kept: 925 hPa {kept:.2f}, wind {kept_u:.2f}, dome cells "
              f"{dk[0]:.2f}/{dk[1]:.2f}/{dk[2]:.2f})")
    return 0


def cap_moisture_transport(q, u_edge, dp, mesh, cap):
    """Column-integrated moisture transport across the boundary of the cell set
    ``cap`` [kg/s]: (gross inflow, gross outflow, net) plus the same net from
    the model's own divergence operator.

    The edge flux is the CENTRED tracer flux (``cell_to_edge_avg`` of q times
    the edge-normal wind, as ``tracer_horizontal_advection`` forms it),
    mass-weighted with the edge-averaged layer thickness.  Gross inflow and
    outflow are split PER EDGE AND LEVEL before the vertical sum (a column
    with inflow below and outflow aloft counts in both).  ``u_edge`` is
    positive from ``cellsOnEdge[0]`` to ``cellsOnEdge[1]`` (``edgeSignOnCell``
    is +1 for that first cell), so an edge with only its second cell inside
    the cap carries ``+F`` inward and one with only its first cell inside
    carries ``-F`` inward; the sign is checked against the model's operator
    only, not independently (net import > 0 with P - E > 0 over the cap is
    the physical sanity check).  Interior edges cancel in the area-integrated
    divergence, so ``-sum_cap(area * div)`` must equal the boundary sum: that
    identity validates the GEOMETRY (edge selection, orientation, weights),
    not the water budget - a wrong q or wind would still close.  All inputs
    must be finite (NaN would pass the identity).
    """
    import jax.numpy as jnp
    from legoesm.core.operators_voronoi import cell_to_edge_avg_3d, divergence_cell_3d
    from legoesm import constants
    for name, a in (("q", q), ("u_edge", u_edge), ("dp", dp)):
        if not np.isfinite(np.asarray(a)).all():
            raise SystemExit(f"FATAL: non-finite {name} in the transport input")
    q, u_edge, dp = jnp.asarray(q), jnp.asarray(u_edge), jnp.asarray(dp)
    flux = cell_to_edge_avg_3d(q, mesh) * u_edge * cell_to_edge_avg_3d(dp, mesh) / constants.g
    F = np.asarray(flux * mesh.dvEdge[:, None], dtype=np.float64)          # (nEdges, nlev) kg/s
    c0, c1 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
    orient = np.where(cap[c1] & ~cap[c0], 1.0, 0.0) + np.where(cap[c0] & ~cap[c1], -1.0, 0.0)
    inward = orient[:, None] * F                                            # (nEdges, nlev), + = into the cap
    div = np.asarray(jnp.sum(divergence_cell_3d(flux, mesh), axis=1), dtype=np.float64)
    net_div = -float((np.asarray(mesh.areaCell, dtype=np.float64) * div)[cap].sum())
    return float(inward[inward > 0].sum()), float(inward[inward < 0].sum()), float(inward.sum()), net_div


def era5_q_columns(month, lat, lon):
    """ERA5 monthly mixing ratio on the ERA5 pressure levels at every cell
    (nearest ERA5 column) and the ERA5 surface pressure: (plev, (ncol, nplev), ps)."""
    hus = era5_month("hus", month).sortby("plev")
    plev = np.asarray(hus.plev, dtype=np.float64)
    cols = np.stack([on_cells(hus.isel(plev=k), lat, lon) for k in range(plev.size)], axis=1)   # (ncol, nplev)
    return plev, cols / (1.0 - cols), on_cells(era5_month("ps", month), lat, lon)


def era5_q_on_levels(plev, cols, ps, p_full):
    """Log-p interpolation of the ERA5 columns to THIS checkpoint's model levels;
    only ERA5 levels above the ERA5 surface are used as brackets, so a model
    level is NaN where it lies below the ERA5 surface, above the ERA5 top, or
    would be bracketed by a below-surface ERA5 level."""
    out = np.full(p_full.shape, np.nan)
    lp = np.log(plev)
    for i in range(p_full.shape[0]):
        valid = plev <= ps[i]
        if valid.sum() < 2:
            continue
        top, bot = plev[valid][0], plev[valid][-1]
        ok = (p_full[i] >= top) & (p_full[i] <= bot)
        out[i, ok] = np.interp(np.log(p_full[i, ok]), lp[valid], cols[i][valid])
    return out


def inflow_humidity(q_model, q_ref, u_edge, dp, mesh, cap):
    """Inflow-mass-weighted mixing ratio of the air ENTERING the cap, model vs a
    reference field on the same edge-levels and the same weights
    (w = inward u * dp * dv / g > 0), restricted to edge-levels where the
    reference is defined; the extra gross import obtained by replacing only q
    with the reference on the INFLOW [kg/s]; the signed NET change when q is
    replaced on inflow AND outflow [kg/s]; and the inflow and outflow mass
    fractions where the reference is defined (elsewhere the model humidity is
    RETAINED, so the replacement is partial by those fractions).  Both humidities are the centred two-cell edge
    average (the flux reconstruction), i.e. the boundary humidity, not the
    upwind cell's.  A fixed-flow sensitivity, not a budget partition."""
    import jax.numpy as jnp
    from legoesm.core.operators_voronoi import cell_to_edge_avg_3d
    from legoesm import constants
    c0, c1 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
    orient = np.where(cap[c1] & ~cap[c0], 1.0, 0.0) + np.where(cap[c0] & ~cap[c1], -1.0, 0.0)
    w = orient[:, None] * np.asarray(u_edge) * np.asarray(cell_to_edge_avg_3d(jnp.asarray(dp), mesh)) \
        * np.asarray(mesh.dvEdge)[:, None] / constants.g
    qm = np.asarray(cell_to_edge_avg_3d(jnp.asarray(q_model), mesh))
    qr = np.asarray(cell_to_edge_avg_3d(jnp.asarray(q_ref), mesh))   # NaN if either cell is undefined there
    fin = np.isfinite(qr)
    use = (w > 0) & fin
    if not use.any() or not (w > 0).any():
        raise SystemExit("FATAL: no inflow edge-level with a defined reference humidity")
    covered = float(w[use].sum() / w[w > 0].sum())
    out_cov = float(w[(w < 0) & fin].sum() / w[w < 0].sum()) if (w < 0).any() else 1.0
    ww = w[use]
    both = (w != 0) & fin
    return float((qm[use] * ww).sum() / ww.sum()), float((qr[use] * ww).sum() / ww.sum()), \
        float(((qr[use] - qm[use]) * ww).sum()), float(((qr[both] - qm[both]) * w[both]).sum()), covered, out_cov


LAYER_BOUNDS_PA = (60000.0, 80000.0)                  # above 600 / 600-800 / below 800 hPa (edge-level pressure)
SECTORS = (("Atl 330-30", 330.0), ("Bar 30-90", 30.0), ("Sib 90-150", 90.0), ("Pac 150-210", 150.0),
           ("Ala 210-270", 210.0), ("CAA 270-330", 270.0))      # 60-degree sectors, start longitude [deg E]


def cap_inflow_air_state(q, T, u_edge, dp, p_full, mesh, cap):
    """``inflow_air_state`` on the model's cap boundary: edge mass weights as in
    ``cap_transport_bins``, edge-averaged mixing ratio converted to specific
    humidity (the ERA5 convention), edge-averaged T, q_sat from the shared
    saturation curve at the edge-level pressure."""
    import jax.numpy as jnp
    from legoesm.core.operators_voronoi import cell_to_edge_avg_3d
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm import constants
    c0, c1 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
    orient = np.where(cap[c1] & ~cap[c0], 1.0, 0.0) + np.where(cap[c0] & ~cap[c1], -1.0, 0.0)
    w = orient[:, None] * np.asarray(u_edge) * np.asarray(cell_to_edge_avg_3d(jnp.asarray(dp), mesh)) \
        * np.asarray(mesh.dvEdge)[:, None] / constants.g
    r = np.asarray(cell_to_edge_avg_3d(jnp.asarray(q), mesh)); qs = r / (1.0 + r)
    Te = np.asarray(cell_to_edge_avg_3d(jnp.asarray(T), mesh))
    p_edge = np.asarray(cell_to_edge_avg_3d(jnp.asarray(p_full), mesh))
    rs = np.asarray(saturation_mixing_ratio(jnp.asarray(Te), jnp.asarray(p_edge))); q_sat = rs / (1.0 + rs)
    layer = np.digitize(p_edge, LAYER_BOUNDS_PA)
    sector = sector_index(np.degrees(np.asarray(mesh.lonEdge)))[:, None] * np.ones_like(layer)
    return inflow_air_state(w, qs, Te, q_sat, layer, sector, orient[:, None] != 0)


def cap_transport_bins(q, u_edge, dp, p_full, mesh, cap, q_ref=None):
    """Per (layer, sector) gross inflow, gross outflow and net transport across the
    cap boundary [kg/s], the signed net change from replacing q by ``q_ref``
    where it is defined (0 where not), and that change over the inflowing
    elements alone; bins recover the totals of ``cap_moisture_transport``.  Layers by the edge-averaged level pressure,
    sectors by the edge longitude."""
    import jax.numpy as jnp
    from legoesm.core.operators_voronoi import cell_to_edge_avg_3d
    from legoesm import constants
    c0, c1 = np.asarray(mesh.cellsOnEdge[0]), np.asarray(mesh.cellsOnEdge[1])
    orient = np.where(cap[c1] & ~cap[c0], 1.0, 0.0) + np.where(cap[c0] & ~cap[c1], -1.0, 0.0)
    w = orient[:, None] * np.asarray(u_edge) * np.asarray(cell_to_edge_avg_3d(jnp.asarray(dp), mesh)) \
        * np.asarray(mesh.dvEdge)[:, None] / constants.g
    qm = np.asarray(cell_to_edge_avg_3d(jnp.asarray(q), mesh))
    F = qm * w                                                        # (nEdges, nlev), + = into the cap
    chg = np.zeros_like(F)
    if q_ref is not None:
        qr = np.asarray(cell_to_edge_avg_3d(jnp.asarray(q_ref), mesh))
        fin = np.isfinite(qr)
        chg[fin] = (qr[fin] - qm[fin]) * w[fin]
    p_edge = np.asarray(cell_to_edge_avg_3d(jnp.asarray(p_full), mesh))
    layer = np.digitize(p_edge, LAYER_BOUNDS_PA)                      # 0 above 600, 1 600-800, 2 below 800
    sector = sector_index(np.degrees(np.asarray(mesh.lonEdge)))
    return bin_sums(F, chg, layer, sector[:, None] * np.ones_like(layer), orient[:, None] != 0, air_in=w > 0)


def sector_index(lon_deg):
    """Index into ``SECTORS`` of each longitude [deg, any range]."""
    lon = np.asarray(lon_deg) % 360.0
    sector = np.full(lon.shape, -1)
    for k, (_, lo) in enumerate(SECTORS):
        sector[((lon - lo) % 360.0) < 60.0] = k
    assert (sector >= 0).all()
    return sector


def bin_sums(F, chg, layer, sector, active, air_in=None):
    """(3 layers, len(SECTORS), 5): gross inflow, gross outflow, net, the summed
    ``chg`` of the ``active`` elements of ``F`` in each (layer, sector) bin, and
    that sum over the elements whose AIR enters the cap (``air_in``, the sign of
    the mass transport; defaults to ``F > 0``, which differs where q is 0)."""
    if air_in is None:
        air_in = F > 0
    out = np.zeros((3, len(SECTORS), 5))
    for li in range(3):
        for si in range(len(SECTORS)):
            m = (layer == li) & (sector == si) & active
            Fm, cm, am = F[m], chg[m], air_in[m]
            out[li, si] = (Fm[Fm > 0].sum(), Fm[Fm < 0].sum(), Fm.sum(), cm.sum(), cm[am].sum())
    return out


def inflow_air_state(w, q, T, q_sat, layer, sector, active):
    """(3 layers, len(SECTORS), 4) over the elements whose AIR enters the cap
    (``w > 0``): mass transport [kg/s], mass-weighted humidity [kg/kg], mass-
    weighted temperature [K] and the mass-weighted relative humidity
    sum(w q)/sum(w q_sat) (NaN where no air enters).  Both boundary
    instruments call it with their own mass weights ``w``."""
    out = np.full((3, len(SECTORS), 4), np.nan)
    for li in range(3):
        for si in range(len(SECTORS)):
            m = (layer == li) & (sector == si) & active & (w > 0)
            if not m.any():
                continue
            wm = w[m]; M = wm.sum()
            out[li, si] = (M, (wm * q[m]).sum() / M, (wm * T[m]).sum() / M, (wm * q[m]).sum() / (wm * q_sat[m]).sum())
    return out


def print_inflow_air(label, S, to_hpa):
    """Per-sector (all levels) and per-layer inflowing-air state from ``inflow_air_state``."""
    print(f"{'':>14s}  inflowing air, {label}: mass in [hPa/day] / q [g/kg] / T [K] / RH")
    print(f"{'':>26s}" + "".join(f"{nm:>30s}" for nm, _ in SECTORS) + f"{'ALL':>30s}")
    rows = [(LAYER_NAMES[li], S[li:li + 1]) for li in range(3)] + [("ALL", S)]
    for name, B in rows:
        def cell(sub):
            M = np.nansum(sub[..., 0])
            if M <= 0:
                return f"  {'-':>28s}"
            q = np.nansum(sub[..., 0] * sub[..., 1]) / M; T = np.nansum(sub[..., 0] * sub[..., 2]) / M
            rh = np.nansum(sub[..., 0] * sub[..., 1]) / np.nansum(sub[..., 0] * sub[..., 1] / np.where(np.isfinite(sub[..., 3]) & (sub[..., 3] > 0), sub[..., 3], np.nan))
            return f"  {M * to_hpa:6.1f}/{q * 1e3:5.3f}/{T:6.1f}/{rh:4.2f}"
        print(f"{'':>14s}{name:>12s}" + "".join(cell(B[:, si]) for si in range(len(SECTORS))) + cell(B))


def era5_boundary_transport(v, q, ps, lon_deg, plev, lat_b, T=None):
    """Moisture transport across the latitude circle ``lat_b`` [deg N] from ERA5
    pressure-level fields already interpolated to that circle: ``v`` and ``q``
    (nlev, nlon) [m/s northward; kg per kg of the air that ``dp/g`` weighs,
    i.e. SPECIFIC humidity for ERA5's total-air pressure], ``ps`` (nlon) [Pa],
    ``plev`` (nlev) [Pa] increasing downward.  Returns the (layer, sector)
    bins [kg/s] as ``bin_sums`` (chg column = 0), + = northward = into the cap.

    Same construction as the model's ``cap_transport_bins``: the transport of
    each (level, longitude) element is v * q * dp / g * (R cos(lat) dlon) and
    the in/out split is made per element before any sum.  ``dp`` is the
    level's pressure slab between the arithmetic mid-points to its
    neighbours; the top slab reaches up to plev[0] minus half the first
    spacing (floored at 0) and the LOWEST level's slab reaches down to the
    surface pressure, so the column is complete from ~0 to ps.  Every slab
    is cut at ps: a level whose centre is below ground keeps only the part
    of its slab above the surface (ERA5's below-ground values are
    extrapolations; that retained part is the price of a complete column).
    Layers by the level pressure, as the model uses the edge-level pressure:
    a slab straddling 600 or 800 hPa is counted whole on the side its centre
    lies (totals are exact, per-layer rows carry a half-slab caveat on both
    sides of the comparison).  With ``T`` (nlev, nlon) [K] the INFLOWING-air
    state of ``inflow_air_state`` is returned instead (q_sat from the shared
    saturation curve at the level pressure, as specific humidity).
    """
    from legoesm import constants
    v, q, ps, plev = (np.asarray(a, dtype=np.float64) for a in (v, q, ps, plev))
    if not (np.diff(plev) > 0).all() or plev[-1] < 5000.0:
        raise SystemExit("FATAL: ERA5 plev must increase downward and be in Pa")
    for name, a in (("v", v), ("q", q), ("ps", ps)):
        if not np.isfinite(a).all():
            raise SystemExit(f"FATAL: non-finite ERA5 {name} on the boundary circle")
    mid = 0.5 * (plev[1:] + plev[:-1])
    p_hi = np.concatenate([[max(plev[0] - 0.5 * (plev[1] - plev[0]), 0.0)], mid])
    p_lo = np.concatenate([mid, [np.inf]])                                  # lowest slab: down to ps
    dp = np.clip(np.minimum(p_lo[:, None], ps[None, :]) - p_hi[:, None], 0.0, None)   # (nlev, nlon)
    lon = np.asarray(lon_deg, dtype=np.float64) % 360.0
    dlon = np.diff(np.concatenate([lon, [lon[0] + 360.0]]))
    if not np.allclose(dlon, dlon[0]):
        raise SystemExit("FATAL: ERA5 longitudes not equally spaced")
    w = dp / constants.g * constants.R_earth * np.cos(np.radians(lat_b)) * np.radians(dlon)[None, :]
    F = v * q * w
    layer = np.digitize(plev, LAYER_BOUNDS_PA)[:, None] * np.ones_like(F, dtype=int)
    sector = sector_index(lon)[None, :] * np.ones_like(F, dtype=int)
    if T is not None:
        from legoesm.thermo import saturation_mixing_ratio
        q_sat = np.asarray(saturation_mixing_ratio(np.asarray(T, dtype=np.float64), plev[:, None] * np.ones_like(F)))
        q_sat = q_sat / (1.0 + q_sat)                                          # mixing ratio -> specific, like q
        return inflow_air_state(v * w, q, np.asarray(T, dtype=np.float64), q_sat, layer, sector, np.ones(F.shape, bool))
    return bin_sums(F, np.zeros_like(F), layer, sector, np.ones(F.shape, bool))


LAYER_NAMES = ("above 600", "600-800 hPa", "below 800")


def print_bin_tables(tables, with_chg):
    """The layer x sector tables of ``bin_sums`` output: in / out / net per bin,
    row and column totals; ``tables`` = [(label, B, units), ...]."""
    for label, B, units in tables:
        print(f"{'':>14s}  layer x sector, {label} [{units}]: in / out / net"
              + (" / ERA5-q net change / ERA5-q inflow change" if with_chg else ""))
        print(f"{'':>26s}" + "".join(f"{nm:>26s}" for nm, _ in SECTORS) + f"{'ALL':>26s}")
        for li in range(3):
            row = "".join(f"  {B[li, si, 0]:5.3f}/{B[li, si, 1]:6.3f}/{B[li, si, 2]:6.3f}"
                          + (f"/{B[li, si, 3]:6.3f}/{B[li, si, 4]:6.3f}" if with_chg else "      ") for si in range(len(SECTORS)))
            T = B[li].sum(axis=0)
            row += f"  {T[0]:5.3f}/{T[1]:6.3f}/{T[2]:6.3f}" + (f"/{T[3]:6.3f}/{T[4]:6.3f}" if with_chg else "")
            print(f"{'':>14s}{LAYER_NAMES[li]:>12s}{row}")
        T = B.sum(axis=(0, 1)); Ts = B.sum(axis=0)
        row = "".join(f"  {Ts[si, 0]:5.3f}/{Ts[si, 1]:6.3f}/{Ts[si, 2]:6.3f}"
                      + (f"/{Ts[si, 3]:6.3f}/{Ts[si, 4]:6.3f}" if with_chg else "      ") for si in range(len(SECTORS)))
        print(f"{'':>14s}{'ALL':>12s}{row}  {T[0]:5.3f}/{T[1]:6.3f}/{T[2]:6.3f}" + (f"/{T[3]:6.3f}/{T[4]:6.3f}" if with_chg else ""))


def trapezoid(v):
    """Trapezoid time mean of equally spaced samples (one sample = itself)."""
    return (0.5 * (v[0] + v[-1]) + sum(v[1:-1])) / (len(v) - 1) if len(v) > 1 else v[0]


def era5_daily(path, var):
    """One variable of a cdo-converted ERA5 daily file with a datetime time axis
    (cdo leaves the pressure-level files' axis as the float 'YYYYMMDD.f' it
    read from GRIB; the surface file decodes)."""
    import pandas as pd
    import xarray as xr
    d = xr.open_dataset(path)[var]
    if not np.issubdtype(d.time.dtype, np.datetime64):
        t = np.asarray(d.time, dtype=np.float64)
        days = pd.to_datetime([f"{int(x):08d}" for x in t], format="%Y%m%d")
        d = d.assign_coords(time=days + pd.to_timedelta(np.round((t - np.floor(t)) * 86400.0), unit="s"))
    if not (np.diff(d.time.values).astype("timedelta64[s]") > np.timedelta64(0, "s")).all():
        raise SystemExit(f"FATAL: {path}: time axis not increasing")
    return d


def era5_transport(args):
    """The same layer x sector boundary transport from ERA5 pressure-level
    analyses (the DKRZ pool extracted to netCDF: v = var132, specific humidity =
    var133, surface pressure = var134; file stems from ``--stems``), for the
    samples in ``--dates a,b`` inclusive: per sample and the plain mean over the
    window, moisture [mm/day over the cap] and air mass [hPa/day of cap-mean
    surface pressure].  The per-sample products are averaged (in/out split per
    sample and element), so with the pool's hourly files the mean carries the
    sub-daily v'q' covariance AND the sub-daily sign reversals; the DAILY-MEAN
    files (stamped 11:30) lose both (products of daily means, split after
    averaging) - measured Jan 7-16 1979: gross in 1.094 hourly vs 1.004 daily.  A plain
    mean over the window: for daily-mean files each sample already represents
    its day; hourly analyses are instantaneous samples, and ``--hour H`` keeps
    only those at hour H (00 UTC = the model's checkpoint protocol; the mean
    of 10 daily 00 UTC samples vs the 240-hour mean measures the snapshot
    alias).  The model side uses the trapezoid of its instantaneous snapshots."""
    from legoesm import constants
    if len(args.dates) != 2:
        raise SystemExit(f"FATAL: --dates needs first,last; got {args.dates}")
    stems = dict(kv.split("=") for kv in args.stems.split(","))
    d = {k: era5_daily(f"{args.era5_dir}/{stems[k]}.nc", v).sel(time=slice(*args.dates))
         for k, v in (("v", "var132"), ("q", "var133"), ("ps", "var134"))}
    if "T" in stems:
        d["T"] = era5_daily(f"{args.era5_dir}/{stems['T']}.nc", "var130").sel(time=slice(*args.dates))
    n = d["v"].time.size
    if n == 0 or any(x.time.size != n for x in d.values()):
        raise SystemExit(f"FATAL: ERA5 dates {args.dates}: {[int(x.time.size) for x in d.values()]} samples")
    for k in d:
        if k != "v" and (not np.array_equal(d[k].time.values, d["v"].time.values) or not np.array_equal(d[k].lon.values, d["v"].lon.values)):
            raise SystemExit(f"FATAL: ERA5 {k}: time/lon axis differs from v")
        if k in ("q", "T") and not np.array_equal(d[k].plev.values, d["v"].plev.values):
            raise SystemExit(f"FATAL: ERA5 {k}: plev axis differs from v")
    if args.hour is not None:
        d = {k: x.sel(time=x.time.dt.hour == args.hour) for k, x in d.items()}
        n = d["v"].time.size
        if n == 0:
            raise SystemExit(f"FATAL: no ERA5 samples at hour {args.hour}")
    step = np.diff(d["v"].time.values).astype("timedelta64[s]")
    if n > 1 and not (step == step[0]).all():
        raise SystemExit(f"FATAL: ERA5 samples not equally spaced: {np.unique(step)}")
    for k in d:
        d[k] = d[k].sortby("lat").interp(lat=args.lat).transpose("time", ..., "lon")
    plev = np.asarray(d["v"].plev, dtype=np.float64)
    lon = np.asarray(d["v"].lon, dtype=np.float64)
    A = 2.0 * np.pi * constants.R_earth ** 2 * (1.0 - np.sin(np.radians(args.lat)))
    to_mm, to_hpa = 86400.0 / A, constants.g / A * 86400.0 / 100.0
    import subprocess
    sha = subprocess.run(["git", "-C", str(_VAL), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "-C", str(_VAL), "status", "--porcelain", "--", str(pathlib.Path(__file__).name)],
                      capture_output=True, text=True).stdout.strip():
        sha += "+dirty"
    print(f"ERA5 moisture transport across {args.lat:g}N [mm/day over the cap area]; {n} samples "
          f"{str(d['v'].time.values[0])[:16]}..{str(d['v'].time.values[-1])[:16]} step {step[0] if n > 1 else 'n/a'}; "
          f"files {args.era5_dir}/{{{args.stems}}}; plev {plev[0]/100:g}-{plev[-1]/100:g} hPa ({plev.size}); "
          f"cap_water @ {sha or 'unknown'}")
    print(f"{'time':>16s} {'in':>8s} {'out':>8s} {'net':>8s} {'mass net':>9s}")
    bins, mass, air = [], [], []
    for t in range(n):
        q = np.asarray(d["q"].isel(time=t), dtype=np.float64)                 # specific humidity on total-air dp
        v, ps = np.asarray(d["v"].isel(time=t)), np.asarray(d["ps"].isel(time=t))
        b = era5_boundary_transport(v, q, ps, lon, plev, args.lat) * to_mm
        m = era5_boundary_transport(v, np.ones_like(q), ps, lon, plev, args.lat) * to_hpa
        bins.append(b); mass.append(m)
        if "T" in d:
            air.append(era5_boundary_transport(v, q, ps, lon, plev, args.lat, T=np.asarray(d["T"].isel(time=t))))
        T, M = b.sum(axis=(0, 1)), m.sum(axis=(0, 1))
        if n <= 31:
            print(f"{str(d['v'].time.values[t])[:16]:>16s} {T[0]:8.3f} {T[1]:8.3f} {T[2]:8.3f} {M[2]:9.1f}")
    label = f"mean of {n} samples {args.dates[0]}..{args.dates[1]}"
    B, M = np.mean(bins, axis=0), np.mean(mass, axis=0)
    print(f"{'ERA5':>14s}: net import {label} {B.sum(axis=(0, 1))[2]:+.3f} mm/day (mass net {M.sum(axis=(0, 1))[2]:+.1f} hPa/day)")
    print_bin_tables([(label, B, "mm/day over the cap"), (f"AIR MASS, {label}", M, "hPa/day of cap-mean p_s")], False)
    if air:
        # time mean of the inflow state = inflow-mass-weighted over samples (mass-weighted q, T, RH)
        A = np.asarray(air); Mw = np.nan_to_num(A[..., 0])
        S = np.full(A.shape[1:], np.nan); Msum = Mw.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            S[..., 0] = Msum
            for j in (1, 2):
                S[..., j] = np.nansum(Mw * A[..., j], axis=0) / Msum
            S[..., 3] = np.nansum(Mw * A[..., 1], axis=0) / np.nansum(Mw * A[..., 1] / A[..., 3], axis=0)
        S[..., 0] /= n
        print_inflow_air(label, S, to_hpa)
    else:
        S = None
    if args.json:
        json.dump({"dates": list(args.dates), "n": int(n), "lat": args.lat, "stems": args.stems, "cap_water": sha,
                   "sectors": [nm for nm, _ in SECTORS], "layers": list(LAYER_NAMES),
                   "moisture_mm_day": B.tolist(), "mass_hpa_day": M.tolist(),
                   "times": [str(t)[:16] for t in d["v"].time.values],
                   "inflow_air": (S.tolist() if air else None),
                   "inflow_air_columns": ["mass_in_kg_s_per_sample", "q_kg_kg", "T_K", "RH"],
                   "samples_moisture_mm_day": np.asarray(bins).tolist(), "samples_mass_hpa_day": np.asarray(mass).tolist()},
                  open(args.json, "w"))
    return 0


def transport(args):
    """Moisture import into the cap (lat >= --lat) per checkpoint, mm/day over the cap area:
    gross in, gross out, net, and the time means of the per-snapshot products (transient
    included) vs the product of the time-mean fields (mean-flow part)."""
    from legoesm.grids.factory import create_grid
    from legoesm import constants
    print(f"moisture transport across lat >= {args.lat:g} [mm/day over the cap area]; vapour and condensate separately")
    print(f"{'run:day':>14s} {'in':>8s} {'out':>8s} {'net':>8s} {'closure':>9s} {'cond net':>9s} {'prw':>7s}"
          + (f" {'q_in model':>11s} {'q_in ERA5':>10s} {'ratio':>6s} {'extra in':>9s} {'net chg':>8s} {'cover':>6s}" if args.era5_q else ""))
    meshes, acc, era5q, era5acc = {}, {}, {}, {}
    for spec in args.specs:
        run, day = spec.split(":")
        st, lat, lon, area, order = load_state(run, int(day))
        if st["u_edge"] is None or not st["order_is_identity"]:
            raise SystemExit(f"FATAL: {spec}: needs edge winds and identity column order")
        res = int(st["exp"]["grid"]["resolution"])
        if res not in meshes:
            meshes[res] = create_grid("mpas", resolution=res)
        mesh = meshes[res]
        if st["u_edge"].shape[0] != np.asarray(mesh.dvEdge).shape[0]:
            raise SystemExit(f"FATAL: {spec}: edge count {st['u_edge'].shape[0]} != mesh")
        cap = np.asarray(lat >= args.lat)
        A = float(area[cap].sum())
        to_mm = 86400.0 / A
        gin, gout, net, net_div = cap_moisture_transport(st["trc_q_v"], st["u_edge"], st["dp"], mesh, cap)
        if abs(net - net_div) > 1e-6 * max(abs(gin), abs(gout), 1.0):
            raise SystemExit(f"FATAL: {spec}: boundary sum {net:.4g} != -area*div {net_div:.4g}")
        cnet = (cap_moisture_transport(st["trc_q_c"] + st["trc_q_i"], st["u_edge"], st["dp"], mesh, cap)[2]
                if st["trc_q_c"] is not None and st["trc_q_i"] is not None else float("nan"))
        prw = float(area_mean(np.asarray(column_water_vapor(st["trc_q_v"], st["p_s"], None, dp=st["dp"])), area, cap))
        line = f"{spec:>14s} {gin*to_mm:8.3f} {gout*to_mm:8.3f} {net*to_mm:8.3f} {(net-net_div)*to_mm:9.1e} {cnet*to_mm:9.4f} {prw:7.2f}"
        if args.era5_q:
            key = (res, st["month"])
            if key not in era5q:
                era5q[key] = era5_q_columns(st["month"], lat, lon)
            qref = era5_q_on_levels(*era5q[key], st["p_full"])
            qm, qr, extra, netchg, cov, ocov = inflow_humidity(st["trc_q_v"], qref, st["u_edge"], st["dp"], mesh, cap)
            line += f" {qm*1e3:11.3f} {qr*1e3:10.3f} {qm/qr:6.2f} {extra*to_mm:9.3f} {netchg*to_mm:8.3f} {cov:6.2f}/{ocov:4.2f}"
            e_acc = era5acc.setdefault(run, {"extra": [], "netchg": []})
            e_acc["extra"].append(extra * to_mm); e_acc["netchg"].append(netchg * to_mm)
        print(line)
        a = acc.setdefault(run, {"n": 0, "net": 0.0, "nets": [], "q": 0.0, "u": 0.0, "dp": 0.0, "mesh": mesh, "cap": cap, "A": A})
        a["n"] += 1; a["net"] += net * to_mm; a["nets"].append(net * to_mm)
        a.setdefault("snaps", []).append((st["trc_q_v"], st["u_edge"], st["dp"]))
        a.setdefault("days", []).append(float(day))
        if args.bins:
            if args.mass:
                # AIR-mass transport (q = 1): hPa/day of cap-mean surface pressure equivalent
                bm = cap_transport_bins(np.ones_like(st["trc_q_v"]), st["u_edge"], st["dp"], st["p_full"], mesh, cap)
                a.setdefault("mass_bins", []).append(bm * constants.g / A * 86400.0 / 100.0)
            b = cap_transport_bins(st["trc_q_v"], st["u_edge"], st["dp"], st["p_full"], mesh, cap,
                                   qref if args.era5_q else None)
            a.setdefault("air", []).append(cap_inflow_air_state(st["trc_q_v"], st["T"], st["u_edge"], st["dp"], st["p_full"], mesh, cap))
            tot = b.sum(axis=(0, 1))
            if not np.allclose(tot[:3], (gin, gout, net), rtol=1e-9, atol=1e-6):
                raise SystemExit(f"FATAL: {spec}: bins {tot[:3]} do not recover the totals {(gin, gout, net)}")
            a.setdefault("bins", []).append(b * to_mm)
        a["q"] = a["q"] + st["trc_q_v"]; a["u"] = a["u"] + st["u_edge"]; a["dp"] = a["dp"] + st["dp"]
    for run, a in acc.items():
        n = a["n"]
        mean_net = cap_moisture_transport(a["q"] / n, a["u"] / n, a["dp"] / n, a["mesh"], a["cap"])[2] * 86400.0 / a["A"]
        nets = a["nets"]
        if n > 1 and not np.allclose(np.diff(a["days"]), a["days"][1] - a["days"][0]):
            raise SystemExit(f"FATAL: {run}: snapshots not equally spaced in time ({a['days']})")
        trap = trapezoid(nets)
        # humidity-weather covariance: each snapshot's winds carrying the run's TIME-MEAN humidity
        q_mean = a["q"] / n
        nets_qmean = [cap_moisture_transport(q_mean, u_s, dp_s, a["mesh"], a["cap"])[2] * 86400.0 / a["A"]
                      for _q, u_s, dp_s in a["snaps"]]
        tz = trapezoid
        C = tz(nets) - tz(nets_qmean)
        print(f"{run:>14s}: mean of {n} snapshot products {a['net']/n:+.3f} mm/day (trapezoid over the span {trap:+.3f}); "
              f"product of time-mean fields {mean_net:+.3f}; remainder (transients + dp covariance) {a['net']/n - mean_net:+.3f}")
        print(f"{'':>14s}  snapshot winds carrying the run's time-mean humidity, trapezoid {tz(nets_qmean):+.3f}; "
              f"humidity-weather covariance C = original - that = {C:+.3f} mm/day")
        if args.bins:
            bins = a["bins"]
            tzb = trapezoid(bins)
            tables = [(f"trapezoid days {a['days'][0]:g}-{a['days'][-1]:g}", tzb, "mm/day over the cap"),
                      (f"day {a['days'][-1]:g} alone", bins[-1], "mm/day over the cap")]
            if args.mass:
                tables.append((f"AIR MASS, trapezoid days {a['days'][0]:g}-{a['days'][-1]:g}", trapezoid(a["mass_bins"]),
                               "hPa/day of cap-mean p_s"))
            print_bin_tables(tables, args.era5_q)
            A = np.asarray(a["air"]); Mw = np.nan_to_num(A[..., 0]); Msum = Mw.sum(0)
            S = np.full(A.shape[1:], np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                S[..., 0] = Msum / n
                for j in (1, 2):
                    S[..., j] = np.nansum(Mw * A[..., j], axis=0) / Msum
                S[..., 3] = np.nansum(Mw * A[..., 1], axis=0) / np.nansum(Mw * A[..., 1] / A[..., 3], axis=0)
            print_inflow_air(f"mean of {n} snapshots days {a['days'][0]:g}-{a['days'][-1]:g}", S, constants.g / a["A"] * 86400.0 / 100.0)
        if run in era5acc:
            e = era5acc[run]
            print(f"{'':>14s}  ERA5-humidity replacement, trapezoid: extra gross inflow {tz(e['extra']):+.3f} mm/day; "
                  f"signed NET change (inflow and outflow replaced) {tz(e['netchg']):+.3f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile"); p.add_argument("specs", nargs="+"); p.add_argument("--lat-lo", type=float, default=75.0)
    p.add_argument("--lat-hi", type=float, default=90.01, help="band upper latitude (exclusive)")
    p.set_defaults(fn=profile)
    l = sub.add_parser("loop"); l.add_argument("specs", nargs="+"); l.set_defaults(fn=loop)
    t = sub.add_parser("transport"); t.add_argument("specs", nargs="+"); t.set_defaults(fn=transport)
    t.add_argument("--lat", type=float, default=72.5, help="cap boundary latitude [deg N]")
    e = sub.add_parser("era5", help=era5_transport.__doc__.split("\n")[0]); e.set_defaults(fn=era5_transport)
    e.add_argument("--era5-dir", required=True, help="directory with pl_132.nc, pl_133.nc, sf_134.nc")
    e.add_argument("--dates", type=lambda x: tuple(x.split(",")), required=True, help="first,last date (inclusive)")
    e.add_argument("--lat", type=float, default=72.5, help="cap boundary latitude [deg N]")
    e.add_argument("--stems", default="v=pl_132,q=pl_133,ps=sf_134", help="file stems per variable (hourly: v=pl1h_132,...)")
    e.add_argument("--json", default=None, help="also write the window-mean bins (layer x sector x in/out/net) to this file")
    e.add_argument("--hour", type=int, default=None, help="keep only samples at this UTC hour (hourly files)")
    t.add_argument("--mass", action="store_true", help="with --bins: also the AIR-mass transport per bin (q = 1), "
                   "in hPa/day of cap-mean surface-pressure equivalent")
    t.add_argument("--bins", action="store_true", help="layer (600/800 hPa) x 60-degree longitude sector "
                   "decomposition of in/out/net (and the ERA5-q net change), trapezoid mean and last day alone")
    t.add_argument("--era5-q", action="store_true", help="inflow-weighted humidity of the entering air, "
                   "model vs ERA5 monthly climatology on the same weights [g/kg], and the extra gross import "
                   "from replacing q by ERA5 [mm/day]; 'cover' = inflow mass fraction where ERA5 is defined")
    l.add_argument("--dome-max-z", type=float, default=5.0, help="max model surface height [m] of a dome cell")
    args = ap.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
