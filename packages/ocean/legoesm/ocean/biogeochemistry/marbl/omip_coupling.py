"""Couple the 32-tracer MARBL port to scripts/run_omip.py (lat-lon C-grid, JRA55-do).

Used by run_omip.py when ``--biogeo marbl``:

* ``setup_marbl``  -- load the initial state (``--marbl-ic``), the monthly
  deposition / sea-ice forcing (``--marbl-forcing``, optional) and build the
  jitted MARBL source/sink step.  ``--marbl-write-grid PATH`` instead writes the
  model grid (for scripts/build_marbl_ic.py) and exits.
* ``marbl_jra55_step`` -- called at the end of ``_jra55_step``: MARBL sources and
  sinks every ``--marbl-every`` steps, using JRA55 shortwave, 10-m wind, sea-level
  pressure and xCO2.  Transport is done inside the ocean model.
* writes mean 2-D diagnostics every ``--marbl-diag-days`` days to
  ``<output>/latlon/<resolution>/marbl_diag_dayNNNNNN.npz`` and prints global NPP,
  export, air-sea CO2 flux and the total-P inventory (conservation check).

Place at: src/legoesm/ocean/biogeochemistry/marbl/omip_coupling.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.ocean.biogeochemistry.marbl.config import TRACER_NAMES, MarblConfig, MarblTracers
from legoesm.ocean.biogeochemistry.marbl.driver3d import Marbl2DForcing, make_marbl_stepper

FORCING_VARS = ("dust_flux", "iron_flux", "nox_flux", "nhy_flux", "ifrac")
R_EARTH = 6.371e6
SEC_PER_YEAR = 365.0 * 86400.0
_M: dict = {}


def active() -> bool:
    return _M.get("step") is not None


def _cell_area(lat_deg, lon_deg):
    lat = np.asarray(lat_deg, float)
    lon = np.asarray(lon_deg, float)
    if np.abs(lat).max() > np.pi / 2 + 1e-6:          # degrees -> radians (model grid is radians)
        lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    dlat = np.abs(np.gradient(lat))
    dlon = np.abs(np.gradient(lon))
    band = np.sin(np.minimum(lat + dlat / 2, np.pi / 2)) - np.sin(np.maximum(lat - dlat / 2, -np.pi / 2))
    return R_EARTH ** 2 * band[:, None] * dlon[None, :]


def _write_grid(path, grid, z_coord, state):
    np.savez(path,
             lat=np.asarray(grid.lat), lon=np.asarray(grid.lon),
             z=-np.asarray(z_coord.z_full_ref), z_half=-np.asarray(z_coord.z_half_ref),
             dz=np.asarray(z_coord.dz_ref), mask=np.asarray(state.land_mask.data))
    print(f"  MARBL: model grid written to {path} "
          f"({np.asarray(grid.lat).size} x {np.asarray(grid.lon).size} x {np.asarray(z_coord.dz_ref).size})")


def _load_ic(path, shape, mask, dtype):
    import xarray as xr
    ds = xr.open_dataset(path)
    out = {}
    for k in TRACER_NAMES:
        if k not in ds:
            raise KeyError(f"{path}: missing tracer {k!r}")
        a = np.asarray(ds[k].values, np.float64)
        if a.shape != shape:
            raise ValueError(f"{path}: {k} has shape {a.shape}, model grid is {shape}")
        a = np.where(np.isfinite(a), a, 0.0) * mask[..., None]
        out[k] = jnp.asarray(a, dtype)
    return MarblTracers(**out)


def _load_forcing(path, shape2d):
    if path is None:
        print("  MARBL forcing: no --marbl-forcing file -> no dust/Fe/N deposition, "
              "sea ice where SST < -1 C")
        return None
    import xarray as xr
    ds = xr.open_dataset(path)
    out = {}
    for k in FORCING_VARS:
        if k not in ds:
            if k == "ifrac":
                print("  MARBL forcing: no ifrac in file -> sea ice where SST < -1 C")
                continue
            raise KeyError(f"{path}: missing {k!r}")
        a = np.nan_to_num(np.asarray(ds[k].values, np.float64))
        if a.shape != (12,) + tuple(shape2d):
            raise ValueError(f"{path}: {k} has shape {a.shape}, expected (12,) + {tuple(shape2d)}")
        out[k] = jnp.asarray(a)
    print(f"  MARBL forcing: monthly {', '.join(out)} from {path}")
    return out


def setup_marbl(args, state, grid, z_coord, model):
    """Attach the MARBL initial state to ``state`` and build the source/sink step."""
    if getattr(args, "marbl_write_grid", None):
        _write_grid(args.marbl_write_grid, grid, z_coord, state)
        sys.exit(0)
    if args.forcing_mode != "jra55_do_tropical":
        raise ValueError("--biogeo marbl is only wired for --forcing-mode jra55_do_tropical")
    if not args.marbl_ic:
        raise ValueError("--biogeo marbl needs --marbl-ic (build it with scripts/build_marbl_ic.py)")
    if not model.config.implicit_vertical_mixing:
        raise ValueError("--biogeo marbl needs --implicit-vertical-mixing")
    if not jax.config.jax_enable_x64:
        print("  WARNING: JAX_ENABLE_X64 is off; MARBL runs in float32 (export JAX_ENABLE_X64=1)")

    shape = tuple(state.T.data.shape)
    mask = np.asarray(state.land_mask.data)
    dtype = jnp.float64 if jax.config.jax_enable_x64 else state.T.data.dtype
    bgc = _load_ic(args.marbl_ic, shape, mask, dtype)

    every = max(1, int(getattr(args, "marbl_every", 1)))
    dt = float(args.dt)
    zt = -np.asarray(z_coord.z_full_ref)
    zw = -np.asarray(z_coord.z_half_ref)[1:]
    dz = np.asarray(z_coord.dz_ref)
    _M.clear()
    _M.update(
        step=make_marbl_stepper(MarblConfig(), dt * every, zt, zw, dz,
                                batch_size=int(getattr(args, "marbl_batch_size", 4096))),
        every=every,
        dt_bgc_days=dt * every / 86400.0,
        forcing=_load_forcing(getattr(args, "marbl_forcing", None), shape[:2]),
        z_coord=getattr(model, "z_coord", z_coord),    # the model's own (partial-cell) coordinate
        min_wc=float(model.config.min_water_column_m),
        area=jnp.asarray(_cell_area(grid.lat, grid.lon)),
        mask=jnp.asarray(mask),
        outdir=Path(args.output) / "latlon" / str(args.resolution),
        diag_days=float(getattr(args, "marbl_diag_days", 30.0)),
        acc=None, n_acc=0, acc_day0=None,
        no_sources=os.environ.get("MARBL_NO_SOURCES", "0") == "1",
    )
    _M["outdir"].mkdir(parents=True, exist_ok=True)
    wet = mask > 0.5
    print(f"  MARBL: layer thickness from {type(_M['z_coord']).__name__} (run_omip passed {type(z_coord).__name__})")
    print(f"  MARBL: 32 tracers from {args.marbl_ic}; source/sink every {every} step(s) "
          f"(dt_bgc = {dt * every:.0f} s); surface PO4 mean "
          f"{float(np.asarray(bgc.PO4)[..., 0][wet].mean()):.3f} mmol/m3, "
          f"NO3 {float(np.asarray(bgc.NO3)[..., 0][wet].mean()):.2f} mmol/m3")
    return state._replace(biogeo=bgc)


def _layer_thickness(state):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import compute_layer_thickness
    return compute_layer_thickness(state.eta.data, state.H_bathy.data, _M["z_coord"],
                                   min_water_column_m=_M["min_wc"])


def _total_P(bgc, h_k):
    P = (bgc.PO4 + bgc.DOP + bgc.DOPr + bgc.spP + bgc.diatP + bgc.diazP + bgc.zooC / 117.0)
    vol = h_k * _M["area"][..., None] * _M["mask"][..., None]
    return float(jnp.sum(P * vol))          # mmol


def _inventories(bgc, h_k, S=None):
    """Inventories [mmol] with the model's layer thicknesses, ocean cells only."""
    vol = h_k * _M["area"][..., None] * _M["mask"][..., None]
    tr = {"P": bgc.PO4 + bgc.DOP + bgc.DOPr + bgc.spP + bgc.diatP + bgc.diazP + bgc.zooC / 117.0,
          "N": bgc.NO3 + bgc.NH4 + bgc.DON + bgc.DONr
               + (bgc.spC + bgc.diatC + bgc.diazC + bgc.zooC) * (16.0 / 117.0),
          "ALK": bgc.ALK, "DIC": bgc.DIC, "Si": bgc.SiO3 + bgc.diatSi, "Fe": bgc.Fe,
          "vol": jnp.ones_like(bgc.DIC)}
    if S is not None:
        tr["salt"] = S
    return {k: float(jnp.sum(v * vol)) for k, v in tr.items()}


def _print_inventory(label, day, h_k, bgc):
    inv, inv0 = _inventories(bgc, h_k, _M.get("S_now")), _M["INV0"]
    print(f"    [MARBL{label}] day {day:.2f} inventory change since start: "
          + ", ".join(f"{k} {(inv[k] - inv0[k]) / inv0[k]:+.2e}" for k in inv), flush=True)


def _inventory_only(day, h_k, bgc):
    if _M.get("last_inv_day") is None:
        _M["last_inv_day"] = day - _M["dt_bgc_days"]
    if day - _M["last_inv_day"] + 1e-9 < _M["diag_days"]:
        return
    _M["last_inv_day"] = day
    _print_inventory(" transport only", day, h_k, bgc)


def _accumulate(diag, day, h_k, bgc):
    if _M["acc"] is None:
        _M["acc"] = jax.tree_util.tree_map(jnp.zeros_like, diag)
        _M["acc_day0"] = day - _M["dt_bgc_days"]      # window starts one BGC step earlier
        _M["n_acc"] = 0
    _M["acc"] = jax.tree_util.tree_map(jnp.add, _M["acc"], diag)
    _M["n_acc"] += 1
    if day - _M["acc_day0"] + 1e-9 < _M["diag_days"]:
        return
    n = _M["n_acc"]
    mean = {k: np.asarray(v) / n for k, v in _M["acc"]._asdict().items()}
    fname = _M["outdir"] / f"marbl_diag_day{int(round(day)):06d}.npz"
    np.savez(fname, day_start=_M["acc_day0"], day_end=day, **mean)
    area = np.asarray(_M["area"]) * np.asarray(_M["mask"])     # ocean cells only
    npp = np.nansum(mean["NPP"].sum(0) * area) * 12e-3 * SEC_PER_YEAR * 1e-15     # PgC/yr
    exp = np.nansum(mean["POC_flux_100m"] * area) * 12e-3 * SEC_PER_YEAR * 1e-15
    fco2 = np.nansum(mean["flux_co2"] * area) * 12e-3 * SEC_PER_YEAR * 1e-15
    p_tot = _total_P(bgc, h_k)
    p0 = _M.setdefault("P0", p_tot)
    print(f"    [MARBL] days {_M['acc_day0']:.0f}-{day:.0f}: NPP {npp:.1f} PgC/yr, "
          f"export(100m) {exp:.2f} PgC/yr, air-sea CO2 {fco2:+.2f} PgC/yr (+ into ocean), "
          f"total P drift {(p_tot - p0) / p0:+.2e}  -> {fname.name}", flush=True)
    _print_inventory("", day, h_k, bgc)
    _M["acc"] = None


def marbl_jra55_step(state, step_idx, dt, model, jra55_state, atm, tile_resp):
    """MARBL sources and sinks after the ocean step (every ``--marbl-every`` steps)."""
    if not active() or (step_idx + 1) % _M["every"] != 0:
        return state
    day = (step_idx + 1) * dt / 86400.0
    sst = state.T.data[..., 0]
    ones = jnp.ones_like(sst)

    sw_net = atm.sw_down * (1.0 - tile_resp.albedo)
    u10_sqr = atm.u_lowest ** 2 + atm.v_lowest ** 2
    p_sfc = getattr(atm, "p_surface", None)
    atm_press = (p_sfc / 101325.0) if p_sfc is not None else ones

    f = _M["forcing"]
    if f is None:
        zero = jnp.zeros_like(sst)
        dep = dict(dust_flux=zero, iron_flux=zero, nox_flux=zero, nhy_flux=zero)
        ifrac = jnp.clip((-1.0 - sst) / 0.8, 0.0, 1.0)
    else:
        m = int((day % 365.0) / 365.0 * 12) % 12
        dep = {k: f[k][m] * _M["mask"] for k in ("dust_flux", "iron_flux", "nox_flux", "nhy_flux")}
        ifrac = f["ifrac"][m] if "ifrac" in f else jnp.clip((-1.0 - sst) / 0.8, 0.0, 1.0)

    forcing = Marbl2DForcing(
        sw_net=sw_net, u10_sqr=u10_sqr, atm_press=atm_press * ones, ifrac=ifrac,
        xco2=float(jra55_state["co2_ppmv"]) * ones, **dep,
    )
    h_k = _layer_thickness(state)
    _M["S_now"] = state.S.data
    if "INV0" not in _M:
        _M["INV0"] = _inventories(state.biogeo, h_k, state.S.data)
        print(f"    [MARBL] budget diagnostics on{' (sources/sinks OFF: transport only)' if _M['no_sources'] else ''}",
              flush=True)
    if _M["no_sources"]:
        _inventory_only(day, h_k, state.biogeo)
        return state
    bgc_new, diag = _M["step"](state.biogeo, state.T.data, state.S.data, h_k, forcing)
    # sediment + hydrothermal Fe (MARBL_FE_SOURCES, build_marbl_fe_sources.py) [patch_marbl_fe_sources]
    if "fe_src" not in _M:
        _p = __import__("os").environ.get("MARBL_FE_SOURCES")
        _M["fe_src"] = None
        if _p:
            import xarray as _xr
            _d = _xr.open_dataset(_p)
            _src = (_d["fe_sed"].values + _d["fe_vent"].values) * __import__("numpy").asarray(_M["mask"])[..., None]
            _M["fe_src"] = jnp.asarray(_src)
            _tot = float((_src.sum(-1) * __import__("numpy").asarray(_M["area"])).sum()) * 365 * 86400 * 1e-12
            print(f"  MARBL: sediment + vent Fe from {_p}: {_tot:.2f} Gmol/yr", flush=True)
    if _M["fe_src"] is not None:
        _act = h_k > 1e-3
        _dFe = jnp.where(_act, _M["fe_src"] * (_M["dt_bgc_days"] * 86400.0) / jnp.maximum(h_k, 1e-3), 0.0)
        bgc_new = bgc_new._replace(Fe=bgc_new.Fe + _dFe.astype(bgc_new.Fe.dtype))
    _wet = _M["mask"] > 0.5                       # land columns: tracers and diagnostics = 0
    bgc_new = jax.tree_util.tree_map(lambda x: jnp.where(_wet[..., None], x, 0.0), bgc_new)
    diag = jax.tree_util.tree_map(lambda x: jnp.where(_wet, x, 0.0), diag)
    _accumulate(diag, day, h_k, bgc_new)
    return state._replace(biogeo=bgc_new)


def is_marbl(bgc) -> bool:
    return isinstance(bgc, MarblTracers)


def surface_scalars(bgc, mask_2d) -> dict:
    """Ocean-mean surface values for run_omip's progress lines and diagnostics."""
    wet = np.asarray(mask_2d) > 0.5
    def m(a, scale=1.0):
        return float(np.asarray(a)[..., 0][wet].mean()) * scale
    return {
        "NO3_surf": m(bgc.NO3), "PO4_surf": m(bgc.PO4), "SiO3_surf": m(bgc.SiO3),
        "Fe_surf_nM": m(bgc.Fe, 1e3), "O2_surf": m(bgc.O2),
        "DIC_surf": m(bgc.DIC), "ALK_surf": m(bgc.ALK),
        "Chl_surf": m(bgc.spChl) + m(bgc.diatChl) + m(bgc.diazChl),
    }
