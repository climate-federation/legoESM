"""Shared single-column setup for the differentiable land sensitivity demos.

The three demos (``demo_process_jacobian``, ``demo_adjoint_memory``,
``demo_global_param_map``) run the SAME multilayer-land + two-leaf-canopy column
the LMIP driver runs, under the package's synthetic LMIP forcing.  This module
holds only what they share:

* a column config from a named soil texture (Cosby pedotransfer, the plant
  wilting point / field capacity read off that column's own retention curve),
* canopy parameters for a named biome (the canopy PFT tables),
* the per-step forcing with the two perturbation knobs the demos differentiate
  through (air-temperature offset, air-humidity scale),
* the ``lax.scan`` rollout returning the flux history (and optionally the state
  trajectory),
* a central-finite-difference gradient check and a provenance stamp.

No land physics is re-implemented here: every number comes from
``step_multilayer_land_with_diagnostics`` and the package's own helpers.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import subprocess
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.boundary_data.gap_fill import bare_canopy_params
from legoesm.land.canopy.config import (
    PFT_AERO_PARAMS,
    PFT_CANOPY_HEIGHT,
    PFT_VCMAX25_C3,
    CanopyConfig,
    CanopyLandParams,
)
from legoesm.land.climate_forcing import make_climatological_forcing
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.lmip_forcing import latitude_daily_mean_sw_w
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    step_multilayer_land_with_diagnostics,
)
from legoesm.land.pedotransfer import soil_hydraulics_config_from_texture
from legoesm.land.soil_grid import make_soil_grid
from legoesm.land.soil_hydraulics import psi_from_theta, theta_from_psi

from legoesm import constants

# --- soil-physics conventions for the plant water thresholds (matric potential) ---
PSI_WILTING_M = -150.0        # permanent wilting point, -15 bar [m of water]
PSI_FIELD_CAPACITY_M = -3.3   # field capacity, -1/3 bar [m of water]

# --- representative USDA textures (sand %, clay %) fed to the Cosby pedotransfer ---
TEXTURES: dict[str, tuple[float, float]] = {
    "sand": (85.0, 5.0),
    "loam": (40.0, 20.0),
    "clay": (20.0, 45.0),
}

# --- canopy defaults ---
DEFAULT_BIOME = "DBF"          # deciduous broadleaf forest (canopy PFT tables)
CLIMATE_ZONE_TEMPERATE = 1     # column index of the [boreal, temperate, tropical] PFT tables
DEFAULT_LAI = 4.0              # leaf area index [m2/m2]

# --- solver / run defaults (same values run_lmip_smoke uses) ---
NEWTON_MAX_ITERS = 50
NEWTON_TOL = 1e-2
DEFAULT_U_MIN_M_S = 1.0        # wind-speed floor inside the land step [m/s]
DEFAULT_PRECIP_KG_M2_S = 2.0e-5  # 1.73 mm/day, the synthetic-LMIP default rate
DEFAULT_DT_S = 3600.0

# --- column climate, fed to the package's climate-feature forcing generator ---
# A temperate-continental site: mean annual 285 K with a 10 K seasonal
# half-amplitude gives ~295 K July afternoons; insolation follows the real solar
# geometry of the column's latitude.  The LMIP latitude preset
# (288 K - 30 K |lat|/90 deg) is deliberately not used: it puts 40 N near
# freezing in June, so the growing-season demos would run a snowy regime.
DEFAULT_MEAN_ANNUAL_T_K = 285.0
DEFAULT_T_SEASONAL_AMP_K = 10.0
DEG_PER_HOUR = 15.0            # Earth rotation, 360 deg / 24 h (local solar hour offset)

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0
HOURS_PER_DAY = 24.0
DAYS_PER_YEAR = 365.0

# --- figure palette (dataviz reference instance; categorical order is fixed) ---
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")   # blue, orange, aqua
CMAP_SEQUENTIAL = "Blues"
CMAP_DIVERGING = "RdBu_r"


# ===========================================================================
# Column configuration
# ===========================================================================
def texture_thresholds(hydraulics) -> tuple[float, float]:
    """Plant wilting point and field capacity [m3/m3] from the column's own
    retention curve (``theta_from_psi`` at the conventional matric potentials)."""
    wp = float(theta_from_psi(jnp.asarray(PSI_WILTING_M), hydraulics))
    fc = float(theta_from_psi(jnp.asarray(PSI_FIELD_CAPACITY_M), hydraulics))
    return wp, fc


def make_column_config(texture: str) -> MultiLayerLandConfig:
    """Multilayer-land + two-leaf-canopy config for a named soil texture.

    Raises ``ValueError`` on an unknown texture (no silent default).
    """
    if texture not in TEXTURES:
        raise ValueError(
            f"Unknown texture {texture!r}; choose one of {sorted(TEXTURES)}")
    sand_pct, clay_pct = TEXTURES[texture]
    hydraulics = soil_hydraulics_config_from_texture(sand_pct, clay_pct)
    wp, fc = texture_thresholds(hydraulics)
    return MultiLayerLandConfig(
        surface_scheme=CanopyConfig(max_iters=NEWTON_MAX_ITERS, tol=NEWTON_TOL),
        hydraulics=hydraulics,
        theta_wp=wp,
        theta_fc=fc,
    )


def make_canopy_params(ncol: int, *, biome: str = DEFAULT_BIOME,
                       lai: float = DEFAULT_LAI) -> CanopyLandParams:
    """Vegetated :class:`CanopyLandParams` for ``ncol`` identical columns.

    Starts from the package's bare-ground template and fills the vegetation
    fields from the canopy PFT tables at the temperate climate zone.  Raises
    ``ValueError`` on a biome missing from any of the tables.
    """
    for table, name in ((PFT_CANOPY_HEIGHT, "PFT_CANOPY_HEIGHT"),
                        (PFT_VCMAX25_C3, "PFT_VCMAX25_C3"),
                        (PFT_AERO_PARAMS, "PFT_AERO_PARAMS")):
        if biome not in table:
            raise ValueError(f"Unknown biome {biome!r}: not in {name}")
    base = bare_canopy_params(ncol)

    def full(v):
        return jnp.full(ncol, float(v))

    return base._replace(
        LAI=full(lai),
        hc=full(PFT_CANOPY_HEIGHT[biome]),
        FNonVeg=full(0.0),
        Vcmax25_C3_leaf=full(PFT_VCMAX25_C3[biome][CLIMATE_ZONE_TEMPERATE]),
        rz0m=full(PFT_AERO_PARAMS[biome]["rz0m"]),
        rd=full(PFT_AERO_PARAMS[biome]["rd"]),
    )


def initial_state(config: MultiLayerLandConfig, *, t_init_k: float, ncol: int = 1):
    """Column state at uniform temperature, soil water midway between the
    plant wilting point and field capacity, growth-temperature EMA seeded."""
    theta0 = 0.5 * (config.theta_wp + config.theta_fc)
    return init_multilayer_land_state(
        ncol, config, T_init=t_init_k, theta_init=theta0,
        TgC_init=t_init_k - constants.T_freeze)


def soil_layers(config: MultiLayerLandConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(dz, z_node, z_interface)`` [m] of the config's soil grid."""
    g = make_soil_grid(config.soil_grid)
    return np.asarray(g.dz), np.asarray(g.z_node), np.asarray(g.z_interface)


def column_water_kg_m2(theta: jnp.ndarray, dz) -> jnp.ndarray:
    """Soil water of the whole column ``sum_k theta_k dz_k rho_w`` [kg/m2] (last axis)."""
    return jnp.sum(theta * jnp.asarray(dz), axis=-1) * constants.rho_water


def column_storage_kg_m2(state, dz) -> jnp.ndarray:
    """All liquid and frozen water a single column holds [kg/m2]: soil water, surface
    ponding, snow and the intercepted canopy store (stores the state does not carry
    count as zero)."""
    total = column_water_kg_m2(state.theta_soil[0], dz) + state.snow_depth[0]
    if state.surface_water is not None:
        total = total + state.surface_water[0] * constants.rho_water
    if state.W_canopy is not None:
        total = total + state.W_canopy[0]
    return total


# ===========================================================================
# Forcing and time axis
# ===========================================================================
class TimeAxis(NamedTuple):
    doy: np.ndarray    # (n,) day of year in [0, 365)
    hour: np.ndarray   # (n,) UTC hour in [0, 24)
    t_s: np.ndarray    # (n,) elapsed seconds since the first step


def make_time_axis(n_steps: int, dt_s: float, *, start_doy: float,
                   start_hour: float = 0.0) -> TimeAxis:
    """Per-step ``(doy, hour)`` for ``n_steps`` of ``dt_s`` from ``start_doy``."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    t_s = np.arange(n_steps, dtype=np.float64) * dt_s
    doy = (start_doy + (start_hour * SECONDS_PER_HOUR + t_s) / SECONDS_PER_DAY) % DAYS_PER_YEAR
    hour = (start_hour + t_s / SECONDS_PER_HOUR) % HOURS_PER_DAY
    return TimeAxis(doy=doy, hour=hour, t_s=t_s)


class ColumnClimate(NamedTuple):
    """Thermal climate of a column for the climate-feature forcing generator."""
    mean_annual_t_k: float
    t_seasonal_amp_k: float   # NH-phased seasonal half-amplitude (peak ~July)


DEFAULT_CLIMATE = ColumnClimate(DEFAULT_MEAN_ANNUAL_T_K, DEFAULT_T_SEASONAL_AMP_K)


def column_forcing(lat_rad: float, lon_rad: float, doy, hour, precip_rate, *,
                   climate: ColumnClimate = DEFAULT_CLIMATE,
                   t_air_offset_k=0.0, q_air_scale=1.0) -> AtmToSurface:
    """Synthetic forcing for one column plus two perturbation knobs.

    The thermal climate comes from ``climate``; the day's daily-mean insolation
    from the latitude's solar geometry (``latitude_daily_mean_sw_w``); the
    seasonal / diurnal cycles, humidity, longwave and the rain/snow split from
    ``make_climatological_forcing`` at the local solar hour.

    ``t_air_offset_k`` is added to the lowest-level air temperature ONLY (the
    downward longwave and the rain/snow split keep their unperturbed values), so
    its derivative is the sensitivity to near-surface air temperature at fixed
    radiation.  ``q_air_scale`` multiplies the lowest-level specific humidity
    (a humidity / vapour-pressure-deficit perturbation at fixed temperature).
    """
    sw_mean_w = latitude_daily_mean_sw_w(lat_rad, doy)
    local_hour = hour + jnp.rad2deg(jnp.asarray(lon_rad)) / DEG_PER_HOUR
    f = make_climatological_forcing(
        climate.mean_annual_t_k, climate.t_seasonal_amp_k, sw_mean_w, precip_rate,
        doy, local_hour)
    return f._replace(T_lowest=f.T_lowest + t_air_offset_k,
                      q_lowest=f.q_lowest * q_air_scale)


# ===========================================================================
# Rollout
# ===========================================================================
class ColumnHistory(NamedTuple):
    """Per-step diagnostics of a single-column rollout (each leaf ``(n,)``
    except ``theta`` ``(n, n_layers)``)."""
    lhflx: jnp.ndarray       # actual latent heat flux, +up [W/m2]
    shflx: jnp.ndarray       # sensible heat flux, +up [W/m2]
    le_canopy: jnp.ndarray   # canopy (transpiration) latent heat demand [W/m2]
    le_soil: jnp.ndarray     # below-canopy soil evaporation [W/m2]
    gpp: jnp.ndarray         # gross primary production [gC/m2/s]
    t_sfc: jnp.ndarray       # surface skin temperature [K]
    runoff: jnp.ndarray      # surface + subsurface runoff [kg/m2/s]
    theta: jnp.ndarray       # soil water content after the step [m3/m3]
    n_iters: jnp.ndarray     # canopy Newton iterations


def step_history(state, forcing, config, params, dt_s, doy, *, u_min=DEFAULT_U_MIN_M_S):
    """One land step -> ``(new_state, ColumnHistory)`` for a single column."""
    new_state, resp, _carbon, out = step_multilayer_land_with_diagnostics(
        state, forcing, config, u_min, dt_s, doy=doy, land_params=params)
    hist = ColumnHistory(
        lhflx=resp.lhflx[0], shflx=resp.shflx[0],
        le_canopy=out.LE_canopy[0], le_soil=out.LE_soil[0], gpp=out.gpp[0],
        t_sfc=resp.T_sfc[0],
        runoff=new_state.runoff_surface[0] + new_state.runoff_subsurface[0],
        theta=new_state.theta_soil[0], n_iters=out.n_iters[0],
    )
    return new_state, hist


def rollout(state0, config, params, *, lat_rad: float, lon_rad: float,
            time_axis: TimeAxis, dt_s: float, precip,
            climate: ColumnClimate = DEFAULT_CLIMATE,
            t_air_offset=None, q_air_scale=None,
            u_min: float = DEFAULT_U_MIN_M_S, remat: bool = False,
            return_states: bool = False):
    """``lax.scan`` the column over ``time_axis``.

    ``precip`` is a scalar or ``(n,)`` rate [kg/m2/s]; ``t_air_offset`` [K] and
    ``q_air_scale`` [-] are optional ``(n,)`` perturbation series (the demos
    differentiate with respect to these).  ``remat`` re-materialises each step
    under reverse-mode AD (bounded memory for long windows).  With
    ``return_states`` the state trajectory after every step is returned as a
    third element.
    """
    n = int(time_axis.doy.shape[0])
    precip = jnp.broadcast_to(jnp.asarray(precip, dtype=jnp.float64), (n,))
    t_off = (jnp.zeros(n) if t_air_offset is None
             else jnp.broadcast_to(jnp.asarray(t_air_offset), (n,)))
    q_sc = (jnp.ones(n) if q_air_scale is None
            else jnp.broadcast_to(jnp.asarray(q_air_scale), (n,)))

    def body(state, xs):
        doy, hour, p, toff, qsc = xs
        f = column_forcing(lat_rad, lon_rad, doy, hour, p, climate=climate,
                           t_air_offset_k=toff, q_air_scale=qsc)
        new_state, hist = step_history(state, f, config, params, dt_s, doy, u_min=u_min)
        return new_state, ((hist, new_state) if return_states else hist)

    if remat:
        body = jax.checkpoint(body)
    xs = (jnp.asarray(time_axis.doy), jnp.asarray(time_axis.hour), precip, t_off, q_sc)
    final, ys = jax.lax.scan(body, state0, xs)
    if return_states:
        hist, states = ys
        return final, hist, states
    return final, ys


def state_at(states, i: int):
    """Slice a stacked state trajectory (leading time axis) at step ``i``."""
    return jax.tree_util.tree_map(lambda a: a[i], states)


# ===========================================================================
# Gradient checks and provenance
# ===========================================================================
class GradientCheck(NamedTuple):
    label: str
    ad: float
    fd: float
    passed: bool
    fd_consistent: bool = True   # FD at eps and eps/2 agree -> the forward is smooth here


# Two-step-size agreement required before a finite difference is trusted as a
# slope.  A smooth response gives O(eps^2) differences; a jump in the forward
# (a branch switch, a bistable canopy) makes the difference quotient scale like
# 1/eps, so halving eps doubles it.
FD_CONSISTENCY_RTOL = 0.02


def central_difference(fn: Callable, x, index, eps: float, *, one_sided: bool = False) -> float:
    """Difference quotient of a scalar ``fn`` along ``e_index``.

    Central ``(fn(x + eps e) - fn(x - eps e)) / (2 eps)`` by default; ``one_sided``
    uses ``(fn(x + eps e) - fn(x)) / eps`` for inputs that must not go negative
    (rain at zero).
    """
    e = jnp.zeros_like(x).at[index].set(eps)
    if one_sided:
        return float((fn(x + e) - fn(x)) / eps)
    return float((fn(x + e) - fn(x - e)) / (2.0 * eps))


def finite_difference(fn: Callable, x, index, eps: float, *, atol: float,
                      one_sided: bool = False,
                      consistency_rtol: float = FD_CONSISTENCY_RTOL) -> tuple[float, bool]:
    """``(fd, consistent)``: the difference quotient at ``eps`` and whether the one
    at ``eps/2`` agrees with it (``|a - b| <= atol + consistency_rtol max(|a|, |b|)``).

    An inconsistent pair means the forward model is not smooth at this point on
    this scale, so NO finite difference there can arbitrate an AD value.
    """
    a = central_difference(fn, x, index, eps, one_sided=one_sided)
    b = central_difference(fn, x, index, 0.5 * eps, one_sided=one_sided)
    consistent = bool(np.isfinite(a) and np.isfinite(b)
                      and abs(a - b) <= atol + consistency_rtol * max(abs(a), abs(b)))
    return a, consistent


def gradient_check(label: str, ad: float, fd: float, *, rtol: float, atol: float,
                   fd_consistent: bool = True) -> GradientCheck:
    """AD-vs-FD agreement test ``|ad - fd| <= atol + rtol |fd|``.

    With ``fd_consistent=False`` the check is recorded as NONSMOOTH: the finite
    difference is not a slope there, so it neither passes nor fails the AD value.
    """
    ad, fd = float(ad), float(fd)
    ok = bool(fd_consistent and np.isfinite(ad) and np.isfinite(fd)
              and abs(ad - fd) <= atol + rtol * abs(fd))
    return GradientCheck(label=label, ad=ad, fd=fd, passed=ok, fd_consistent=bool(fd_consistent))


def format_checks(checks: list[GradientCheck]) -> str:
    lines = [f"{'check':<44s} {'AD':>14s} {'FD':>14s}  status"]
    for c in checks:
        status = ("PASS" if c.passed else "FAIL") if c.fd_consistent else "NONSMOOTH"
        lines.append(f"{c.label:<44s} {c.ad:>14.6g} {c.fd:>14.6g}  {status}")
    return "\n".join(lines)


def require_all_pass(checks: list[GradientCheck]) -> None:
    """Raise if any smooth-point AD-vs-FD check failed, or if none could be run.

    NONSMOOTH checks (the finite difference is not a slope there) are reported
    but cannot arbitrate; at least one smooth check must have passed, otherwise
    the instrument verified nothing and the gradients are not a result.
    """
    smooth = [c for c in checks if c.fd_consistent]
    bad = [c for c in smooth if not c.passed]
    if bad:
        raise RuntimeError("gradient check FAILED:\n" + format_checks(bad))
    if not smooth:
        raise RuntimeError("no smooth point for a finite-difference check; "
                           "the AD values are unverified:\n" + format_checks(checks))


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 — provenance only; never fail a run on git
        return "unknown"


def provenance(args: dict, **extra) -> dict:
    """Machine-readable stamp: git SHA, JAX version, x64 flag, UTC time, CLI args."""
    return {
        "git_sha": git_sha(),
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                 for k, v in args.items()},
        **extra,
    }


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


__all__ = [
    "CMAP_DIVERGING", "CMAP_SEQUENTIAL", "DEFAULT_BIOME", "DEFAULT_CLIMATE",
    "DEFAULT_DT_S", "DEFAULT_LAI", "DEFAULT_MEAN_ANNUAL_T_K",
    "DEFAULT_PRECIP_KG_M2_S", "DEFAULT_T_SEASONAL_AMP_K", "DEFAULT_U_MIN_M_S",
    "DEG_PER_HOUR", "PSI_FIELD_CAPACITY_M", "PSI_WILTING_M", "SECONDS_PER_DAY",
    "SECONDS_PER_HOUR", "SERIES_COLORS", "TEXTURES", "ColumnClimate",
    "ColumnHistory", "GradientCheck", "TimeAxis", "central_difference", "column_forcing",
    "format_checks", "gradient_check", "initial_state", "make_canopy_params",
    "make_column_config", "make_time_axis", "provenance", "psi_from_theta",
    "require_all_pass", "rollout", "column_storage_kg_m2", "column_water_kg_m2",
    "soil_layers",
    "state_at", "step_history", "texture_thresholds", "write_json",
]
