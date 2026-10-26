#!/usr/bin/env python
"""Demo: one backward pass -> global maps of land-flux sensitivity to parameters.

Runs the multilayer land + two-leaf canopy on every land column of a lat-lon
grid.  Soil texture (Cosby pedotransfer), dominant PFT, leaf area and canopy
height come from the harmonised surface dataset; the atmospheric forcing is the
idealised latitude-dependent synthetic LMIP forcing (with an idealised zonal
rainfall profile), so the maps show the model's response to its boundary data,
not a real climate.

After a forward spin-up, the window-integrated evapotranspiration and gross
primary production, each summed over ALL land columns, are differentiated in
ONE Jacobian evaluation with respect to per-column multiplicative factors on
four inputs — leaf area index, Vcmax25, root depth and rainfall rate.  Land
columns do not interact, so that returns ``d target_i / d ln(x_i)`` for every
cell, every input and both targets at once: eight global sensitivity maps from
one forward and two reverse sweeps (a perturbation experiment needs one forward
run per input per perturbation size).  ``--et-inputs`` / ``--gpp-inputs``
choose which of them are plotted.

The AD gradient is checked against a central finite difference at the wettest
cell (two extra forward runs); a failed check aborts the run.

Usage
-----
    JAX_ENABLE_X64=1 python scripts/run/land_sensitivity/demo_global_param_map.py \
        --surfdata data/legoesm_surfdata_c260716.nc --resolution 24 \
        --out-dir diagnostics/land_sensitivity/global_param_map
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import NamedTuple

_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.grids.latlon import create_latlon_grid
from legoesm.land.boundary_data.builders import (
    build_soil_hydraulics,
    glacier_mask,
    surface_data_to_land_params,
)
from legoesm.land.boundary_data.gap_fill import fill_land_param_gaps
from legoesm.land.canopy.config import CanopyConfig
from legoesm.land.climate_forcing import make_climatological_forcing
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.global_surface_data import get_surfdata_preset, load_global_surface_data
from legoesm.land.lmip_forcing import latitude_daily_mean_sw_w
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    root_zone_moisture_stress,
    step_multilayer_land_with_diagnostics,
)
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.soil_hydraulics import theta_from_psi

from legoesm import constants
from scripts.run.land_sensitivity import column_setup as cs

INPUTS = ("lai", "vcmax", "root_depth", "precip")
TARGETS = ("et", "gpp")            # window totals that are differentiated (one Jacobian)
TARGET_LABELS = {"et": "ET", "gpp": "GPP"}
TARGET_UNITS = {"et": "kg/m2", "gpp": "gC/m2"}
DEFAULT_ET_INPUTS = ("vcmax", "precip")
DEFAULT_GPP_INPUTS = INPUTS
INPUT_LABELS = {"lai": "leaf area index", "vcmax": "Vcmax25",
                "root_depth": "root depth", "precip": "rainfall rate"}
# Per-(col, layer) hydraulic fields the Cosby builder fills (subset per column).
HYDRAULIC_ARRAY_FIELDS = ("theta_sat", "psi_sat", "b_ch", "K_sat")

# --- idealised zonal rainfall shape (dimensionless multiplier on --precip-rate) ---
# An equatorial (ITCZ) peak, mid-latitude storm-track peaks, dry subtropics in
# between and a dry-polar floor.  A demo choice for spatial contrast, not a
# climatology.
ZONAL_ITCZ_WIDTH_DEG = 10.0
ZONAL_ITCZ_WEIGHT = 2.0
ZONAL_STORM_TRACK_LAT_DEG = 45.0
ZONAL_STORM_TRACK_WIDTH_DEG = 15.0
ZONAL_STORM_TRACK_WEIGHT = 1.2
ZONAL_BACKGROUND = 0.25
DEFAULT_PRECIP_KG_M2_S = 3.0e-5   # 2.6 mm/day base rate before the zonal shape

# --- idealised latitudinal thermal climate for the climate-feature generator ---
# Mean annual air temperature T_eq - dT sin^2(lat); seasonal half-amplitude
# A_pole sin(lat), NH-phased by the generator, so the SOUTHERN hemisphere's
# negative amplitude flips the phase (its summer in January).  Warmer than the
# LMIP latitude preset (which puts 40 deg near freezing in June); a demo
# choice for a growing-season window, not a climatology.
CLIMATE_T_EQUATOR_K = 300.0
CLIMATE_EQUATOR_TO_POLE_DROP_K = 35.0
CLIMATE_SEASONAL_AMP_POLE_K = 15.0
FD_EPS_FACTOR = 1e-3               # [-] on the multiplicative factor
FD_ATOL = 1e-6                     # [kg/m2 per unit factor]
ET_MIN_FOR_PERCENT = 1.0           # [kg/m2 over the window] below which % maps are masked
GPP_MIN_FOR_PERCENT = 1.0          # [gC/m2 over the window] below which % maps are masked
TARGET_MIN_FOR_PERCENT = {"et": ET_MIN_FOR_PERCENT, "gpp": GPP_MIN_FOR_PERCENT}


class GlobalSetup(NamedTuple):
    lat_rad: jnp.ndarray       # (n_land,)
    lon_rad: jnp.ndarray       # (n_land,)
    idx: np.ndarray            # (n_land,) column index into the full grid
    grid_shape: tuple          # (n_lat, n_lon)
    config: MultiLayerLandConfig  # per-column hydraulics, subset to land
    params: object             # CanopyLandParams (n_land,) with per-column root fields
    theta_wp: np.ndarray       # (n_land,) root-zone plant wilting point
    theta_fc: np.ndarray       # (n_land,) root-zone field capacity
    t_init_k: np.ndarray       # (n_land,)
    n_land_total: int          # land columns before any --max-cells subsample


def zonal_precip_shape(lat_rad) -> jnp.ndarray:
    lat_deg = jnp.rad2deg(jnp.asarray(lat_rad))
    itcz = ZONAL_ITCZ_WEIGHT * jnp.exp(-(lat_deg / ZONAL_ITCZ_WIDTH_DEG) ** 2)
    storm = ZONAL_STORM_TRACK_WEIGHT * jnp.exp(
        -((jnp.abs(lat_deg) - ZONAL_STORM_TRACK_LAT_DEG) / ZONAL_STORM_TRACK_WIDTH_DEG) ** 2)
    return itcz + storm + ZONAL_BACKGROUND


def precip_per_column(lat_rad, base_rate: float, profile: str) -> jnp.ndarray:
    if profile == "zonal":
        return base_rate * zonal_precip_shape(lat_rad)
    if profile == "constant":
        return jnp.full(jnp.asarray(lat_rad).shape, base_rate)
    raise ValueError(f"Unknown precip profile {profile!r}; use 'zonal' or 'constant'")


def base_config() -> MultiLayerLandConfig:
    return MultiLayerLandConfig(
        surface_scheme=CanopyConfig(max_iters=cs.NEWTON_MAX_ITERS, tol=cs.NEWTON_TOL))


def load_surface_data(surfdata_path: str, grid, config: MultiLayerLandConfig):
    """Read + regrid the harmonised surfdata onto ``grid`` (the config's soil grid)."""
    cfg_sd = get_surfdata_preset("legoesm_surfdata")._replace(surf_path=surfdata_path)
    return load_global_surface_data(cfg_sd, grid, soil_grid=make_soil_grid(config.soil_grid))


def _cover(a) -> np.ndarray:
    a = np.asarray(a)
    return a[0] if a.ndim == 2 else a


def land_mask(gsd, land_frac_min: float) -> np.ndarray:
    """Land columns: surfdata land fraction >= ``land_frac_min`` and not glacier-dominant."""
    land_fraction = _cover(gsd.f_land) + _cover(gsd.f_lake) + _cover(gsd.f_glacier)
    return (land_fraction >= land_frac_min) & ~glacier_mask(gsd), land_fraction


def setup_from_surface_data(gsd, lat_rad_all, lon_rad_all, config: MultiLayerLandConfig,
                            day_of_year: float, *, land_frac_min: float,
                            grid_shape: tuple, max_cells: int | None = None,
                            seed: int = 0) -> GlobalSetup:
    """Per-column land inputs from a regridded surfdata, subset to land columns.

    Hydraulics come from the Cosby pedotransfer per (column, layer); the plant
    wilting point / field capacity of each column are read off its own retention
    curve per layer and root-density-weighted (the model's root profile); LAI,
    canopy height, PFT physiology and soil-colour albedo come from
    ``surface_data_to_land_params`` at ``day_of_year`` and are then held fixed
    over the run.
    """
    lat_all = np.asarray(lat_rad_all, dtype=np.float64).ravel()
    lon_all = np.asarray(lon_rad_all, dtype=np.float64).ravel()
    ncol = lat_all.shape[0]
    config = config._replace(hydraulics=build_soil_hydraulics(gsd, base=config.hydraulics))
    hyd = config.hydraulics
    wp_layer = theta_from_psi(jnp.asarray(cs.PSI_WILTING_M), hyd)        # (ncol, n_layers)
    fc_layer = theta_from_psi(jnp.asarray(cs.PSI_FIELD_CAPACITY_M), hyd)
    z_node = make_soil_grid(config.soil_grid).z_node
    _, root_frac, _, _ = root_zone_moisture_stress(
        wp_layer, config.beta_min, config.root_depth, 0.0, 1.0, z_node, ncol)
    wp_col = np.asarray(jnp.sum(root_frac * wp_layer, axis=-1))
    fc_col = np.asarray(jnp.sum(root_frac * fc_layer, axis=-1))
    if not np.all(fc_col > wp_col):
        raise ValueError("field capacity <= wilting point in some column; texture inputs invalid")
    theta_top = jnp.asarray(0.5 * (wp_col + fc_col))
    params = surface_data_to_land_params(gsd, config.surface_scheme, day_of_year, theta_top)
    mask, land_fraction = land_mask(gsd, land_frac_min)
    params = fill_land_param_gaps(params, gsd, f_land=land_fraction)
    idx = np.where(mask)[0]
    n_land_total = int(idx.size)
    if n_land_total == 0:
        raise ValueError("no land columns selected; lower --land-frac-min")
    if max_cells is not None and max_cells < n_land_total:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_cells, replace=False))
    n = int(idx.size)
    sub = {}
    for name in HYDRAULIC_ARRAY_FIELDS:
        v = getattr(hyd, name)
        if isinstance(v, (jnp.ndarray, np.ndarray)) and np.ndim(v) >= 1:
            sub[name] = jnp.asarray(v)[idx]
    config = config._replace(hydraulics=hyd._replace(**sub))
    params = jax.tree_util.tree_map(lambda a: jnp.asarray(a)[idx], params)
    params = params._replace(
        theta_wp=jnp.asarray(wp_col[idx]), theta_fc=jnp.asarray(fc_col[idx]),
        root_depth=jnp.full(n, float(config.root_depth)))
    t_init = np.asarray(latitudinal_climate(jnp.asarray(lat_all[idx]))[0])
    return GlobalSetup(
        lat_rad=jnp.asarray(lat_all[idx]), lon_rad=jnp.asarray(lon_all[idx]), idx=idx,
        grid_shape=tuple(grid_shape), config=config, params=params,
        theta_wp=wp_col[idx], theta_fc=fc_col[idx], t_init_k=t_init,
        n_land_total=n_land_total)


def initial_state(setup: GlobalSetup):
    n = int(setup.idx.size)
    theta0 = 0.5 * (setup.theta_wp + setup.theta_fc)
    state = init_multilayer_land_state(
        n, setup.config, T_init=jnp.asarray(setup.t_init_k),
        theta_init=jnp.asarray(theta0)[:, None])
    return state._replace(TgC=jnp.asarray(setup.t_init_k) - constants.T_freeze)


def latitudinal_climate(lat_rad):
    """``(mean_annual_t_k, t_seasonal_amp_k)`` of the idealised latitudinal climate."""
    s = jnp.sin(jnp.asarray(lat_rad))
    return (CLIMATE_T_EQUATOR_K - CLIMATE_EQUATOR_TO_POLE_DROP_K * s ** 2,
            CLIMATE_SEASONAL_AMP_POLE_K * s)


def global_forcing(lat_rad, lon_rad, doy, hour, precip_col):
    """Per-column forcing: ``vmap`` of the package's climate-feature generator.

    Thermal climate from :func:`latitudinal_climate`, the day's daily-mean
    insolation from each latitude's solar geometry, diurnal cycles at the local
    solar hour.
    """
    mat_k, amp_k = latitudinal_climate(lat_rad)

    def one(la, lo, m, a, p):
        return make_climatological_forcing(
            m, a, latitude_daily_mean_sw_w(la, doy), p, doy,
            hour + jnp.rad2deg(lo) / cs.DEG_PER_HOUR)

    f = jax.vmap(one)(lat_rad, lon_rad, mat_k, amp_k, precip_col)
    return jax.tree_util.tree_map(
        lambda a: a.reshape(a.shape[0]) if (a.ndim == 2 and a.shape[1] == 1) else a, f)


def unit_factors(n: int) -> dict:
    return {k: jnp.ones(n) for k in INPUTS}


def apply_factors(setup: GlobalSetup, precip_col, factors: dict):
    """Multiply the per-column inputs by ``factors`` (all ones at the baseline).

    Bare columns (LAI = 0) keep LAI = 0 through a ``where`` rather than
    ``0 * factor``: the canopy's derivative with respect to LAI is undefined at
    zero leaf area, and ``0 * NaN`` would poison that column's gradient.  The
    selected branch carries a zero cotangent instead, so bare columns report a
    zero LAI sensitivity (masked to NaN in the maps).
    """
    p = setup.params
    params = p._replace(
        LAI=jnp.where(p.LAI > 0.0, p.LAI * factors["lai"], p.LAI),
        Vcmax25_C3_leaf=p.Vcmax25_C3_leaf * factors["vcmax"],
        Vcmax25_C4_leaf=p.Vcmax25_C4_leaf * factors["vcmax"],
        root_depth=p.root_depth * factors["root_depth"])
    return params, precip_col * factors["precip"]


def make_window_fn(setup: GlobalSetup, time_axis: cs.TimeAxis, dt_s: float, precip_col,
                   *, remat: bool = True):
    """``window(factors, state0) -> (et, (gpp, final_state, n_iters_max_per_col))``.

    ``et`` [kg/m2] is the latent heat flux integrated over the window and divided
    by the latent heat of vaporisation (so any sublimation is counted at L_v).
    ``n_iters_max_per_col`` ``(n,)`` is each column's largest canopy Newton
    iteration count over the window (``== max_iters`` means a step did not converge).
    """
    n = int(setup.idx.size)
    lat, lon, config = setup.lat_rad, setup.lon_rad, setup.config
    doy_arr, hour_arr = jnp.asarray(time_axis.doy), jnp.asarray(time_axis.hour)

    def window(factors, state0):
        params, p_col = apply_factors(setup, precip_col, factors)

        def body(carry, xs):
            state, et, gpp = carry
            doy, hour = xs
            f = global_forcing(lat, lon, doy, hour, p_col)
            new_state, resp, _, out = step_multilayer_land_with_diagnostics(
                state, f, config, cs.DEFAULT_U_MIN_M_S, dt_s, doy=doy, land_params=params)
            et = et + resp.lhflx / constants.L_v * dt_s
            gpp = gpp + out.gpp * dt_s
            return (new_state, et, gpp), out.n_iters

        if remat:
            body = jax.checkpoint(body)
        (final, et, gpp), n_iters = jax.lax.scan(
            body, (state0, jnp.zeros(n), jnp.zeros(n)), (doy_arr, hour_arr))
        return et, (gpp, final, jnp.max(n_iters, axis=0))

    def totals(factors, state0):
        """``(sum ET, sum GPP)`` over the columns, plus the per-column aux; one
        ``jacrev`` of this pair is one forward and two reverse sweeps."""
        et, (gpp, final, n_iters) = window(factors, state0)
        return jnp.stack([jnp.sum(et), jnp.sum(gpp)]), (et, gpp, final, n_iters)

    return window, totals


class SensitivityMaps(NamedTuple):
    et: np.ndarray                 # (n_land,) window ET [kg/m2]
    gpp: np.ndarray                # (n_land,) window GPP [gC/m2]
    grad: dict                     # target -> input -> (n_land,) d target_i / d ln x_i
    percent_per_10pct: dict        # target -> input -> (n_land,) % change per +10 % of x
    converged: np.ndarray          # (n_land,) bool: every window step's canopy solve converged

    def take(self, sel: np.ndarray) -> SensitivityMaps:
        """The maps restricted to the local column indices ``sel``."""
        return SensitivityMaps(
            et=self.et[sel], gpp=self.gpp[sel],
            grad={t: {k: v[sel] for k, v in g.items()} for t, g in self.grad.items()},
            percent_per_10pct={t: {k: v[sel] for k, v in g.items()}
                               for t, g in self.percent_per_10pct.items()},
            converged=self.converged[sel])


def unconverged_columns(n_iters_max_per_col, max_iters: int) -> np.ndarray:
    """Columns with at least one step whose canopy Newton solve hit ``max_iters``.

    The solver's implicit-function-theorem gradient is ZEROED on such a step,
    so those columns' sensitivities are partial: they are masked (NaN), never
    reported as small.
    """
    return np.asarray(n_iters_max_per_col) >= max_iters


def save_spun_state(path: pathlib.Path, state, n_iters_spin, key: dict) -> None:
    """Cache the spun-up land state (flattened leaves) with the run key that produced it."""
    leaves = jax.tree_util.tree_leaves(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, key=json.dumps(key, sort_keys=True), n_iters_spin=np.asarray(n_iters_spin),
             **{f"leaf_{i}": np.asarray(a) for i, a in enumerate(leaves)})


def load_spun_state(path: pathlib.Path, template, key: dict):
    """Load a cached spun-up state; raise (never silently reuse) on a key or shape mismatch."""
    z = np.load(path)
    if str(z["key"]) != json.dumps(key, sort_keys=True):
        raise ValueError(f"spun-state cache {path} was made for a different run: "
                         f"{z['key']} != {json.dumps(key, sort_keys=True)}")
    leaves, treedef = jax.tree_util.tree_flatten(template)
    if f"leaf_{len(leaves)}" in z.files:
        raise ValueError(f"spun-state cache {path} has more leaves than the state template")
    saved = []
    for i, tmpl in enumerate(leaves):
        a = z[f"leaf_{i}"]
        if a.shape != np.shape(tmpl):
            raise ValueError(f"spun-state cache leaf {i}: shape {a.shape} != {np.shape(tmpl)}")
        saved.append(jnp.asarray(a))
    return treedef.unflatten(saved), z["n_iters_spin"]


def _slice_state(state, sel):
    return jax.tree_util.tree_map(lambda a: a[sel], state)


def _concat_states(states):
    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *states)


def compute_maps(setup: GlobalSetup, *, dt_s: float, spinup_days: int, window_days: int,
                 start_doy: float, precip_col, remat: bool = True,
                 mask_unconverged: bool = False, column_chunks: int = 1,
                 spun_state_cache: pathlib.Path | None = None, cache_key: dict | None = None):
    """Forward spin-up, then ONE reverse pass over the scored window.

    Columns where a window step's canopy Newton solve hit ``max_iters`` carry a
    PARTIAL gradient (the solver zeroes that step's contribution).  They are
    counted and flagged in ``SensitivityMaps.converged``; with
    ``mask_unconverged`` their sensitivities are set to NaN instead of plotted.

    ``column_chunks`` splits the columns into that many independent batches for
    the spin-up and the reverse pass (columns do not interact, so the gradient is
    unchanged; peak memory scales with the chunk size — the 1-degree run was
    killed for memory in a single 14,705-column reverse pass).  With
    ``spun_state_cache`` the spun-up state is written there and, if the file
    already exists for the same ``cache_key``, loaded instead of recomputed.
    """
    steps_per_day = int(round(cs.SECONDS_PER_DAY / dt_s))
    n = int(setup.idx.size)
    if not 1 <= column_chunks <= n:
        raise ValueError(f"column_chunks must be in [1, {n}], got {column_chunks}")
    chunks = np.array_split(np.arange(n), column_chunks)
    timings = {}
    state0 = initial_state(setup)
    max_iters = setup.config.surface_scheme.max_iters
    spin_axis = cs.make_time_axis(spinup_days * steps_per_day, dt_s,
                                  start_doy=start_doy - spinup_days)
    t0 = time.perf_counter()
    if spun_state_cache is not None and spun_state_cache.exists():
        state_spun, n_iters_spin = load_spun_state(spun_state_cache, state0, cache_key or {})
        print(f"spin-up state loaded from {spun_state_cache}", flush=True)
    else:
        states, n_its = [], []
        for sel in chunks:
            sub = setup if column_chunks == 1 else subset_setup(setup, sel)
            spin_window, _ = make_window_fn(sub, spin_axis, dt_s, precip_col[sel], remat=False)
            _, (_, st, n_it) = jax.jit(spin_window)(unit_factors(sel.size),
                                                    _slice_state(state0, sel))
            jax.block_until_ready(st)
            states.append(st)
            n_its.append(np.asarray(n_it))
        state_spun = states[0] if column_chunks == 1 else _concat_states(states)
        n_iters_spin = np.concatenate(n_its)
        if spun_state_cache is not None:
            save_spun_state(spun_state_cache, state_spun, n_iters_spin, cache_key or {})
            print(f"spin-up state cached at {spun_state_cache}", flush=True)
    timings["spinup_s"] = time.perf_counter() - t0
    bad_spin = unconverged_columns(n_iters_spin, max_iters)
    if bad_spin.any():
        # Forward-only: the state of those columns is the solver's last iterate
        # (finite), so the run continues; they are reported and, if they also
        # fail inside the scored window, masked below.
        print(f"spin-up: canopy Newton hit max_iters in {int(bad_spin.sum())} of {n} "
              f"columns (state kept; masked only if unconverged in the window)", flush=True)

    win_axis = cs.make_time_axis(window_days * steps_per_day, dt_s, start_doy=start_doy)
    window, totals = make_window_fn(setup, win_axis, dt_s, precip_col, remat=remat)
    t0 = time.perf_counter()
    et, gpp = np.empty(n), np.empty(n)
    n_iters = np.empty(n, dtype=np.int64)
    grads = {t: {k: np.empty(n) for k in INPUTS} for t in TARGETS}
    total = np.zeros(len(TARGETS))
    for sel in chunks:
        sub = setup if column_chunks == 1 else subset_setup(setup, sel)
        _, totals_c = (window, totals) if column_chunks == 1 else make_window_fn(
            sub, win_axis, dt_s, precip_col[sel], remat=remat)
        # one Jacobian of the (ET, GPP) pair: (2, n_chunk) per input
        g_c, (et_c, gpp_c, _final, n_it_c) = jax.jit(
            jax.jacrev(totals_c, has_aux=True))(unit_factors(sel.size),
                                                _slice_state(state_spun, sel))
        jax.block_until_ready(g_c)
        total += np.array([float(np.sum(np.asarray(et_c))), float(np.sum(np.asarray(gpp_c)))])
        et[sel], gpp[sel], n_iters[sel] = np.asarray(et_c), np.asarray(gpp_c), np.asarray(n_it_c)
        for k in INPUTS:
            for ti, t in enumerate(TARGETS):
                grads[t][k][sel] = np.asarray(g_c[k])[ti]
        if column_chunks > 1:
            print(f"reverse pass: chunk of {sel.size} columns done "
                  f"({time.perf_counter() - t0:.0f}s so far)", flush=True)
    timings["reverse_pass_s"] = time.perf_counter() - t0
    bad = unconverged_columns(n_iters, max_iters)
    if bad.all():
        raise RuntimeError("canopy Newton solve hit max_iters in EVERY column of the window")
    if bad.any():
        print(f"window: canopy Newton hit max_iters in {int(bad.sum())} of {n} columns; "
              f"their gradients are partial and "
              f"{'masked (NaN)' if mask_unconverged else 'plotted anyway'}", flush=True)
    grad = {t: {k: (np.where(bad, np.nan, v) if mask_unconverged else v) for k, v in g.items()}
            for t, g in grads.items()}
    if not (np.all(np.isfinite(total)) and np.all(np.isfinite(et)) and np.all(np.isfinite(gpp))
            and all(np.all(np.isfinite(v[~bad])) for g in grad.values() for v in g.values())):
        raise FloatingPointError("non-finite ET/GPP or gradient in the global reverse pass")
    # No vegetation -> no leaf-area / Vcmax sensitivity: report NaN, not zero.
    vegetated = np.asarray(setup.params.LAI) > 0.0
    base = {"et": et, "gpp": gpp}
    pct = {}
    for t in TARGETS:
        for k in ("lai", "vcmax"):
            grad[t][k] = np.where(vegetated, grad[t][k], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct[t] = {k: np.where(base[t] >= TARGET_MIN_FOR_PERCENT[t], 10.0 * g / base[t], np.nan)
                      for k, g in grad[t].items()}
    maps = SensitivityMaps(et=et, gpp=gpp, grad=grad, percent_per_10pct=pct, converged=~bad)
    return maps, state_spun, totals, timings


def subset_setup(setup: GlobalSetup, sel: np.ndarray) -> GlobalSetup:
    """The columns ``sel`` (local indices into ``setup``) as their own ``GlobalSetup``."""
    sel = np.asarray(sel)
    hyd = setup.config.hydraulics
    sub = {name: jnp.asarray(getattr(hyd, name))[sel] for name in HYDRAULIC_ARRAY_FIELDS
           if np.ndim(getattr(hyd, name)) >= 1}
    return setup._replace(
        lat_rad=setup.lat_rad[sel], lon_rad=setup.lon_rad[sel], idx=setup.idx[sel],
        config=setup.config._replace(hydraulics=hyd._replace(**sub)),
        params=jax.tree_util.tree_map(lambda a: jnp.asarray(a)[sel], setup.params),
        theta_wp=setup.theta_wp[sel], theta_fc=setup.theta_fc[sel], t_init_k=setup.t_init_k[sel])


def finite_difference_check(maps: SensitivityMaps, totals, state_spun, *, inputs, rtol: float,
                            cell: int | None = None) -> list[cs.GradientCheck]:
    """Central differences of BOTH window totals on one cell's factor(s).

    Default cell: the wettest VEGETATED column (bare columns have no leaf-area
    sensitivity to check).  ``totals``/``state_spun`` may describe a column
    SUBSET of ``maps`` (see :func:`finite_difference_check_subset`).
    """
    n = maps.et.shape[0]
    if cell is None:
        vegetated = np.isfinite(maps.grad["et"]["lai"])
        if not np.any(vegetated):
            raise ValueError("no vegetated column to run the finite-difference check on")
        j = int(np.argmax(np.where(vegetated, maps.et, -np.inf)))
    else:
        j = int(cell)
    fwd = jax.jit(lambda fac: totals(fac, state_spun)[0])
    checks = []
    for name in inputs:
        for ti, t in enumerate(TARGETS):
            def total_of(x, name=name, ti=ti):
                fac = unit_factors(n)
                fac[name] = x
                return fwd(fac)[ti]
            fd, smooth = cs.finite_difference(total_of, jnp.ones(n), j, FD_EPS_FACTOR,
                                              atol=FD_ATOL)
            checks.append(cs.gradient_check(
                f"cell {j} d{TARGET_LABELS[t]}_total/d ln {name}", maps.grad[t][name][j], fd,
                rtol=rtol, atol=FD_ATOL, fd_consistent=smooth))
    return checks


SUBSET_MISMATCH_FAIL_RTOL = 1e-2     # [-] a real interaction / indexing defect
SUBSET_MISMATCH_REPORT_RTOL = 1e-6   # [-] round-off amplified by the solver: printed only


def _report_subset_mismatch(name: str, sub: np.ndarray, full: np.ndarray, sel: np.ndarray) -> None:
    """Fail on an O(1) subset-vs-full mismatch; print the round-off-level ones."""
    scale = np.maximum(np.abs(full), np.max(np.abs(full)) * 1e-6 + 1e-30)
    rel = np.abs(sub - full) / scale
    worst = int(np.argmax(rel))
    if rel[worst] > SUBSET_MISMATCH_FAIL_RTOL:
        raise RuntimeError(f"subset {name} != full-run value at column {int(sel[worst])}: "
                           f"{sub[worst]:.6g} vs {full[worst]:.6g} (rel {rel[worst]:.2e}); "
                           "columns interact or the subset is mis-indexed")
    noisy = np.where(rel > SUBSET_MISMATCH_REPORT_RTOL)[0]
    if noisy.size:
        print(f"subset check: {name} differs from the full run by "
              f"> {SUBSET_MISMATCH_REPORT_RTOL:g} relative in {noisy.size} of {sel.size} columns "
              f"(max {rel[worst]:.2e} at column {int(sel[worst])}); batch-layout round-off, "
              "not interaction", flush=True)


def finite_difference_check_subset(maps: SensitivityMaps, setup: GlobalSetup, state_spun, *,
                                   time_axis: cs.TimeAxis, dt_s: float, precip_col,
                                   max_cells: int, seed: int, inputs, rtol: float,
                                   remat: bool = True) -> list[cs.GradientCheck]:
    """The finite-difference check on a random subset of ``max_cells`` columns.

    Land columns do not interact, so the window function of the subset (built
    from the same spun-up state, sliced) reproduces the full run's per-column ET
    and its gradient exactly; the check costs two forward runs of ``max_cells``
    columns instead of two of the whole grid.  The subset's AD gradient is first
    required to match the full run's entries for those columns.
    """
    n = int(setup.idx.size)
    if max_cells >= n:
        raise ValueError(f"--fd-max-cells {max_cells} must be < the {n} columns run")
    sel = np.sort(np.random.default_rng(seed).choice(n, max_cells, replace=False))
    sub = subset_setup(setup, sel)
    state_sub = jax.tree_util.tree_map(lambda a: a[sel], state_spun)
    _, totals_sub = make_window_fn(sub, time_axis, dt_s, precip_col[sel], remat=remat)
    grads_sub, (et_sub, gpp_sub, _, _) = jax.jit(jax.jacrev(totals_sub, has_aux=True))(
        unit_factors(max_cells), state_sub)
    # The batched forward is not bit-reproducible across batch layouts: XLA
    # fuses/vectorises differently and the canopy Newton early-stop turns
    # round-off into ~1e-5 relative ET differences in a few columns (2 of 256 at
    # 1 degree).  Column INTERACTION or a mis-indexed subset gives O(1)
    # mismatches, so that is what fails; smaller differences are reported.
    _report_subset_mismatch("window ET", np.asarray(et_sub), maps.et[sel], sel)
    _report_subset_mismatch("window GPP", np.asarray(gpp_sub), maps.gpp[sel], sel)
    for k in inputs:
        for ti, t in enumerate(TARGETS):
            full = maps.grad[t][k][sel]
            ok = np.isfinite(full)
            _report_subset_mismatch(f"d {TARGET_LABELS[t]} / d ln {k}",
                                    np.asarray(grads_sub[k])[ti][ok], full[ok], sel[ok])
    maps_sub = maps.take(sel)
    checks = finite_difference_check(maps_sub, totals_sub, state_sub, inputs=inputs, rtol=rtol)
    return [c._replace(label=f"[subset of {max_cells}] {c.label}") for c in checks]


def _write_arrays(out: pathlib.Path, setup: GlobalSetup, maps: SensitivityMaps) -> None:
    np.savez(out / "global_param_map.npz", idx=setup.idx, lat_rad=np.asarray(setup.lat_rad),
             lon_rad=np.asarray(setup.lon_rad), grid_shape=np.asarray(setup.grid_shape),
             et_kg_m2=maps.et, gpp_gC_m2=maps.gpp, converged=maps.converged,
             **{f"grad_{t}_{k}": v for t, g in maps.grad.items() for k, v in g.items()},
             **{f"pct_per_10pct_{t}_{k}": v
                for t, g in maps.percent_per_10pct.items() for k, v in g.items()})


def load_maps(path: pathlib.Path, setup: GlobalSetup) -> SensitivityMaps:
    """Inverse of :func:`_write_arrays`; refuses arrays from a different column set."""
    z = np.load(path)
    if not np.array_equal(z["idx"], setup.idx):
        raise ValueError(f"{path} holds {z['idx'].size} columns that differ from this run's "
                         f"{setup.idx.size}; resume only with identical grid/mask settings")
    return SensitivityMaps(
        et=z["et_kg_m2"], gpp=z["gpp_gC_m2"], converged=z["converged"],
        grad={t: {k: z[f"grad_{t}_{k}"] for k in INPUTS} for t in TARGETS},
        percent_per_10pct={t: {k: z[f"pct_per_10pct_{t}_{k}"] for k in INPUTS} for t in TARGETS})


def to_grid(values: np.ndarray, idx: np.ndarray, shape: tuple) -> np.ndarray:
    full = np.full(int(np.prod(shape)), np.nan)
    full[idx] = values
    return full.reshape(shape)


def plot(maps: SensitivityMaps, setup: GlobalSetup, lat_deg_1d, lon_deg_1d,
         path: pathlib.Path, *, window_days: int,
         panels: dict | None = None) -> None:
    """Baseline ET and GPP on the top row, then one row per input with the ET
    sensitivity (left) and the GPP sensitivity (right); ``panels`` maps each
    target to the inputs shown for it (default ``DEFAULT_ET_INPUTS`` /
    ``DEFAULT_GPP_INPUTS``), and a panel that is not requested is left blank."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = panels or {"et": DEFAULT_ET_INPUTS, "gpp": DEFAULT_GPP_INPUTS}
    for t, names in panels.items():
        bad = [k for k in names if k not in INPUTS]
        if t not in TARGETS or bad:
            raise ValueError(f"unknown panel selection {t}: {bad or names}")
    shown = [k for k in INPUTS if any(k in names for names in panels.values())]

    def edges(x):
        d = np.diff(x)
        return np.concatenate([[x[0] - d[0] / 2], x[:-1] + d / 2, [x[-1] + d[-1] / 2]])

    lon_e, lat_e = edges(lon_deg_1d), edges(lat_deg_1d)
    n_rows = 1 + len(shown)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.9 * n_rows), squeeze=False)
    base = {"et": maps.et, "gpp": maps.gpp}
    long = {"et": "evapotranspiration", "gpp": "gross primary production"}
    for ci, t in enumerate(TARGETS):
        ax = axes[0, ci]
        m = ax.pcolormesh(lon_e, lat_e, to_grid(base[t], setup.idx, setup.grid_shape),
                          cmap=cs.CMAP_SEQUENTIAL)
        fig.colorbar(m, ax=ax, label=f"{TARGET_LABELS[t]} over {window_days} days "
                                     f"[{TARGET_UNITS[t]}]")
        ax.set_title(f"{long[t]} (baseline)", loc="left")
    n_maps = 0
    for ri, name in enumerate(shown, start=1):
        for ci, t in enumerate(TARGETS):
            ax = axes[ri, ci]
            if name not in panels.get(t, ()):
                ax.set_axis_off()
                continue
            z = to_grid(maps.percent_per_10pct[t][name], setup.idx, setup.grid_shape)
            v = float(np.nanpercentile(np.abs(z), 98)) if np.any(np.isfinite(z)) else 1.0
            m = ax.pcolormesh(lon_e, lat_e, z, cmap=cs.CMAP_DIVERGING, vmin=-v, vmax=v)
            fig.colorbar(m, ax=ax, label=f"% change in {TARGET_LABELS[t]} per +10 % "
                                         f"{INPUT_LABELS[name]}")
            ax.set_title(f"sensitivity of {TARGET_LABELS[t]} to {INPUT_LABELS[name]}", loc="left")
            n_maps += 1
    for ax in axes.ravel():
        if ax.axison:
            ax.set_xlabel("longitude [deg]")
            ax.set_ylabel("latitude [deg]")
            ax.set_facecolor("0.94")
    fig.suptitle(f"{n_maps} sensitivity maps from ONE backward pass (ET and GPP differentiated "
                 "together) — synthetic forcing, surfdata soils/vegetation", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surfdata", default="data/legoesm_surfdata_c260716.nc")
    ap.add_argument("--resolution", type=int, default=24, help="lat-lon grid N x 2N")
    ap.add_argument("--dt-s", type=float, default=3.0 * cs.SECONDS_PER_HOUR)
    ap.add_argument("--start-doy", type=float, default=152.0, help="first scored day (June 1)")
    ap.add_argument("--spinup-days", type=int, default=60)
    ap.add_argument("--window-days", type=int, default=60)
    ap.add_argument("--land-frac-min", type=float, default=0.5)
    ap.add_argument("--precip-profile", default="zonal", choices=["zonal", "constant"])
    ap.add_argument("--precip-rate", type=float, default=DEFAULT_PRECIP_KG_M2_S,
                    help="base rainfall rate [kg/m2/s] before the zonal shape")
    ap.add_argument("--max-cells", type=int, default=None,
                    help="random subsample of land columns (smoke runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fd-inputs", nargs="*", default=["lai"], choices=INPUTS,
                    help="inputs checked against finite differences at the wettest cell")
    ap.add_argument("--fd-rtol", type=float, default=2e-3)
    ap.add_argument("--et-inputs", nargs="*", default=list(DEFAULT_ET_INPUTS), choices=INPUTS,
                    help="inputs whose ET sensitivity map is plotted")
    ap.add_argument("--gpp-inputs", nargs="*", default=list(DEFAULT_GPP_INPUTS), choices=INPUTS,
                    help="inputs whose GPP sensitivity map is plotted")
    ap.add_argument("--fd-max-cells", type=int, default=None,
                    help="run the finite-difference check on a random subset of this many "
                         "columns (two forward runs of the subset instead of the whole grid)")
    ap.add_argument("--no-remat", dest="remat", action="store_false")
    ap.add_argument("--column-chunks", type=int, default=1,
                    help="split the columns into this many independent batches for the spin-up "
                         "and reverse pass (same gradient, peak memory / chunks)")
    ap.add_argument("--spun-state-cache", default=None,
                    help="npz path: write the spun-up state there, or load it if it exists "
                         "for the same run settings (skips the spin-up after a killed run)")
    ap.add_argument("--resume-maps", action="store_true",
                    help="skip the spin-up and reverse pass: load the arrays this run already "
                         "wrote to --out-dir and the spun state from --spun-state-cache, then "
                         "run the finite-difference check and the figure (a run killed after "
                         "its reverse pass was saved)")
    ap.add_argument("--mask-unconverged", action="store_true",
                    help="NaN-mask columns whose canopy Newton solve hit max_iters in the window "
                         "(default: plot them; their gradients are partial)")
    ap.add_argument("--out-dir", default="diagnostics/land_sensitivity/global_param_map")
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    grid = create_latlon_grid(args.resolution)
    lat2d = np.asarray(grid.lat2d, dtype=np.float64)
    lon2d = np.asarray(grid.lon2d, dtype=np.float64)
    config = base_config()
    gsd = load_surface_data(args.surfdata, grid, config)
    doy_mid = args.start_doy + 0.5 * args.window_days
    setup = setup_from_surface_data(
        gsd, lat2d, lon2d, config, doy_mid, land_frac_min=args.land_frac_min,
        grid_shape=lat2d.shape, max_cells=args.max_cells, seed=args.seed)
    precip_col = precip_per_column(setup.lat_rad, args.precip_rate, args.precip_profile)
    t_setup = time.perf_counter() - t0
    print(f"grid {lat2d.shape} -> {setup.n_land_total} land columns "
          f"({setup.idx.size} run); setup {t_setup:.1f}s", flush=True)
    cache = pathlib.Path(args.spun_state_cache) if args.spun_state_cache else None
    cache_key = {k: getattr(args, k) for k in (
        "surfdata", "resolution", "dt_s", "start_doy", "spinup_days", "land_frac_min",
        "precip_profile", "precip_rate", "max_cells", "seed")}
    steps_per_day = int(round(cs.SECONDS_PER_DAY / args.dt_s))
    win_axis = cs.make_time_axis(args.window_days * steps_per_day, args.dt_s,
                                 start_doy=args.start_doy)
    if args.resume_maps:
        if cache is None:
            raise ValueError("--resume-maps needs --spun-state-cache (the finite-difference "
                             "check restarts the window from the spun state)")
        maps = load_maps(out / "global_param_map.npz", setup)
        state_spun, _ = load_spun_state(cache, initial_state(setup), cache_key)
        _, totals = make_window_fn(setup, win_axis, args.dt_s, precip_col, remat=args.remat)
        prev = json.loads((out / "global_param_map.json").read_text())
        timings = dict(prev.get("timings", {}), resumed_from=str(out / "global_param_map.npz"))
        print(f"resumed: maps from {out / 'global_param_map.npz'}, state from {cache}", flush=True)
    else:
        maps, state_spun, totals, timings = compute_maps(
            setup, dt_s=args.dt_s, spinup_days=args.spinup_days, window_days=args.window_days,
            start_doy=args.start_doy, precip_col=precip_col, remat=args.remat,
            mask_unconverged=args.mask_unconverged, column_chunks=args.column_chunks,
            spun_state_cache=cache, cache_key=cache_key)
        print(f"spin-up {timings['spinup_s']:.1f}s, reverse pass "
              f"{timings['reverse_pass_s']:.1f}s", flush=True)
        # Hours of compute are on disk before any check can abort the run; the
        # gradient checks are appended to the JSON below and the figure is written
        # only once they pass.
        _write_arrays(out, setup, maps)
        cs.write_json(out / "global_param_map.json", cs.provenance(
            vars(args), timings=timings, n_land_total=setup.n_land_total,
            n_run=int(setup.idx.size), n_unconverged=int((~maps.converged).sum()),
            gradient_checks="NOT YET RUN"))
    timings["setup_s"] = t_setup
    t0 = time.perf_counter()
    if args.fd_max_cells is None:
        checks = finite_difference_check(maps, totals, state_spun, inputs=args.fd_inputs,
                                         rtol=args.fd_rtol)
    else:
        checks = finite_difference_check_subset(
            maps, setup, state_spun, time_axis=win_axis, dt_s=args.dt_s, precip_col=precip_col,
            max_cells=args.fd_max_cells, seed=args.seed, inputs=args.fd_inputs,
            rtol=args.fd_rtol, remat=args.remat)
    timings["fd_checks_s"] = time.perf_counter() - t0
    print(cs.format_checks(checks))
    (out / "gradient_checks.txt").write_text(cs.format_checks(checks) + "\n")
    summary = {t: {k: {"median_pct_per_10pct": float(np.nanmedian(v)),
                       "p10": float(np.nanpercentile(v, 10)),
                       "p90": float(np.nanpercentile(v, 90))}
                   for k, v in g.items()}
               for t, g in maps.percent_per_10pct.items()}
    cs.write_json(out / "global_param_map.json", cs.provenance(
        vars(args), timings=timings, n_land_total=setup.n_land_total, n_run=int(setup.idx.size),
        n_unconverged=int((~maps.converged).sum()), masked=bool(args.mask_unconverged),
        summary=summary, gradient_checks=[c._asdict() for c in checks]))
    cs.require_all_pass(checks)
    if args.plot:
        plot(maps, setup, np.rad2deg(np.asarray(grid.lat)), np.rad2deg(np.asarray(grid.lon)),
             out / "global_param_map.png", window_days=args.window_days,
             panels={"et": tuple(args.et_inputs), "gpp": tuple(args.gpp_inputs)})
    for t, g in summary.items():
        for k, s in g.items():
            print(f"{INPUT_LABELS[k]:16s}: % {TARGET_LABELS[t]} per +10 %  "
                  f"median {s['median_pct_per_10pct']:+.2f}  "
                  f"[p10 {s['p10']:+.2f}, p90 {s['p90']:+.2f}]")
    print(f"timings: {', '.join(f'{k}={v:.1f}s' for k, v in timings.items())}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
