"""Single-column ocean model with KPP mixing and MARBL biogeochemistry.

One time step (operator split, all pure JAX):

1. KPP (``kpp_vertical_mixing``) diagnoses K_v, A_v, h_bl and the non-local
   T/S transport from (u, v, T, S), the wind stress and the surface buoyancy
   flux.  The buoyancy flux uses the non-solar heat flux plus the shortwave
   absorbed above the previous h_bl (LMD94 definition of B_f).
2. T, S: Jerlov shortwave penetration, surface non-solar heat and
   freshwater (virtual salt) fluxes, KPP non-local transport and relaxation
   to a climatology, then backward-Euler diffusion with K_v.
3. u, v: exact Coriolis rotation over dt, surface stress, linear (Rayleigh)
   damping standing in for inertial-wave radiation that a column cannot
   represent, then backward-Euler viscosity with A_v.
4. MARBL: interior source/sink and surface fluxes from the state at the
   start of the step, forward Euler (as MARBL in POP), then backward-Euler
   diffusion of all tracers with K_v.  No clipping is applied.

Forcing is cyclic over a 365-day year and linearly interpolated in time;
surface fields and the T/S climatology may have different sampling (e.g.
daily and monthly).  Daily-mean shortwave can be redistributed over an
analytic diurnal cycle.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.radiation.solar import (
    cos_zenith_angle,
    daily_mean_insolation,
)
from legoesm.ocean.biogeochemistry.marbl.config import (
    MarblConfig,
    MarblInteriorForcing,
    MarblSurfaceForcing,
    MarblTracers,
)
from legoesm.ocean.biogeochemistry.marbl.interface import marbl_interior_tendencies
from legoesm.ocean.biogeochemistry.marbl.surface import marbl_surface_fluxes
from legoesm.ocean.eos import c_sw, rho_0
from legoesm.ocean.physics.shortwave_penetration import (
    JERLOV_TYPES,
    ShortwavePenetrationConfig,
    shortwave_penetration_tendency,
)
from legoesm.ocean.physics.vertical_mixing.config import KPPConfig
from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
    build_dz_half,
    implicit_vertical_diffusion_ocean,
)
from legoesm.ocean.physics.vertical_mixing.kpp import (
    kpp_surface_fluxes,
    kpp_vertical_mixing,
)
from legoesm.ocean.vertical import OceanZStarCoordinate, create_ocean_z_star

_SECONDS_PER_YEAR = constants.seconds_per_day * constants.days_per_year
_PA_PER_BAR = 1.0e5


class ColumnGrid(NamedTuple):
    z_coord: OceanZStarCoordinate
    dz: jnp.ndarray        # (nlev,) [m]
    dz_half: jnp.ndarray   # (nlev-1,) [m]
    zt: jnp.ndarray        # (nlev,) layer-centre depth [m, > 0]
    zw: jnp.ndarray        # (nlev,) layer-bottom depth [m, > 0]
    p_bar: jnp.ndarray     # (nlev,) sea pressure at layer centres [bar]
    lat: float             # [deg N]
    lon: float             # [deg E]
    f_coriolis: float      # [1/s]


def make_column_grid(lat: float, lon: float, depth: float, n_levels: int = 60,
                     dz_surface: float = 5.0, dz_deep: float = 250.0) -> ColumnGrid:
    """Stretched column grid down to ``depth`` [m] at (lat, lon) [deg]."""
    z = create_ocean_z_star(n_levels=n_levels, H_max=depth,
                            dz_surface=dz_surface, dz_deep=dz_deep)
    zt = -z.z_full_ref
    return ColumnGrid(
        z_coord=z, dz=z.dz_ref, dz_half=build_dz_half(z.dz_ref),
        zt=zt, zw=-z.z_half_ref[1:],
        p_bar=rho_0 * constants.g * zt / _PA_PER_BAR,
        lat=lat, lon=lon,
        f_coriolis=2.0 * constants.Omega * jnp.sin(lat * constants.DEG_TO_RAD),
    )


class ColumnConfig(NamedTuple):
    """Static column-physics settings."""
    dt: float = 3600.0                      # [s]
    tau_nudge_T_days: float = 30.0          # T relaxation timescale [day]
    tau_nudge_S_days: float = 30.0          # S relaxation timescale [day]
    u_damping_days: float = 5.0             # inertial-current e-folding [day]; <= 0 disables
    diurnal_shortwave: bool = True
    kpp: KPPConfig = KPPConfig()
    shortwave: ShortwavePenetrationConfig = ShortwavePenetrationConfig()
    carbonate_n_iter: int = 50


class ColumnForcing(NamedTuple):
    """Cyclic forcing; surface fields (n_sfc,), climatology (n_clim, nlev).

    Samples are interval means: sample i covers [i, i+1) / n of the year.
    """
    tau_x: jnp.ndarray         # wind stress on the ocean [Pa]
    tau_y: jnp.ndarray
    u10_sqr: jnp.ndarray       # mean squared 10-m wind [m^2/s^2]
    sw_net: jnp.ndarray        # net shortwave into the ocean, interval mean [W/m^2]
    q_nonsolar: jnp.ndarray    # longwave + sensible + latent into the ocean [W/m^2]
    freshwater: jnp.ndarray    # P - E (+ R) into the ocean [kg/m^2/s]
    atm_press: jnp.ndarray     # [atm]
    ifrac: jnp.ndarray         # sea-ice fraction
    xco2: jnp.ndarray          # [ppmv]
    dust_flux: jnp.ndarray     # [kg/m^2/s]
    iron_flux: jnp.ndarray     # [mmol/m^2/s]
    nox_flux: jnp.ndarray      # [mmol N/m^2/s]
    nhy_flux: jnp.ndarray      # [mmol N/m^2/s]
    T_clim: jnp.ndarray        # (n_clim, nlev) potential temperature [degC]
    S_clim: jnp.ndarray        # (n_clim, nlev) [PSU]


class ColumnState(NamedTuple):
    T: jnp.ndarray             # (nlev,) potential temperature [degC]
    S: jnp.ndarray             # (nlev,) [PSU]
    u: jnp.ndarray             # (nlev,) [m/s]
    v: jnp.ndarray
    bgc: MarblTracers          # each (nlev,)
    h_bl: jnp.ndarray          # () KPP boundary-layer depth [m]
    time: jnp.ndarray          # () seconds since 1 Jan 00:00 UTC of year 0


class ColumnStepDiagnostics(NamedTuple):
    h_bl: jnp.ndarray
    sst: jnp.ndarray
    NPP: jnp.ndarray           # (3,) vertically integrated C fixation per PFT [mmol C/m^2/s]
    POC_flux_100m: jnp.ndarray # [mmol C/m^2/s]
    CaCO3_flux_100m: jnp.ndarray
    flux_co2: jnp.ndarray      # [mmol/m^2/s], + into ocean
    flux_o2: jnp.ndarray
    pCO2surf: jnp.ndarray      # [ppmv]
    pH_surf: jnp.ndarray
    POC_sed_loss: jnp.ndarray  # [mmol/m^2/s]
    CaCO3_sed_loss: jnp.ndarray


def cyclic_interp(samples: jnp.ndarray, year_frac: jnp.ndarray) -> jnp.ndarray:
    """Linear interpolation of interval-mean samples (n, ...) at year fraction in [0, 1)."""
    n = samples.shape[0]
    pos = year_frac * n - 0.5
    i0 = jnp.floor(pos)
    w = pos - i0
    i0 = i0.astype(jnp.int32) % n
    i1 = (i0 + 1) % n
    return (1.0 - w) * samples[i0] + w * samples[i1]


def diurnal_factor(time: jnp.ndarray, lat: float, lon: float) -> jnp.ndarray:
    """Instantaneous / daily-mean top-of-atmosphere insolation (0 at night).

    Multiplying a daily-mean shortwave flux by this factor imposes the
    diurnal cycle while preserving the daily mean.
    """
    day_of_year = jnp.floor((time % _SECONDS_PER_YEAR) / constants.seconds_per_day) + 1.0
    hour = (time % constants.seconds_per_day) / 3600.0
    lat_r = lat * constants.DEG_TO_RAD
    lon_r = lon * constants.DEG_TO_RAD
    inst = constants.S_0 * jnp.maximum(cos_zenith_angle(lat_r, lon_r, day_of_year, hour), 0.0)
    mean = daily_mean_insolation(lat_r, day_of_year)
    return jnp.where(mean > 0.0, inst / jnp.where(mean > 0.0, mean, 1.0), 0.0)


def _stack(tr: MarblTracers) -> jnp.ndarray:
    return jnp.stack(list(tr))


def _unstack(arr: jnp.ndarray) -> MarblTracers:
    return MarblTracers(*[arr[i] for i in range(arr.shape[0])])


def _flux_at_depth(flux_out: jnp.ndarray, zw: jnp.ndarray, depth: float) -> jnp.ndarray:
    """Interpolate a flux defined at layer bottoms (zero at the surface) to ``depth``."""
    z = jnp.concatenate([jnp.zeros(1, zw.dtype), zw])
    f = jnp.concatenate([jnp.zeros(1, flux_out.dtype), flux_out])
    return jnp.interp(depth, z, f)


def column_step(
    state: ColumnState,
    forcing: ColumnForcing,
    grid: ColumnGrid,
    cfg: ColumnConfig,
    marbl_cfg: MarblConfig,
) -> tuple[ColumnState, ColumnStepDiagnostics]:
    """Advance the column by ``cfg.dt``."""
    dt = cfg.dt
    year_frac = (state.time % _SECONDS_PER_YEAR) / _SECONDS_PER_YEAR
    sfc = lambda a: cyclic_interp(a, year_frac)  # noqa: E731
    tau_x, tau_y = sfc(forcing.tau_x), sfc(forcing.tau_y)
    sw = sfc(forcing.sw_net)
    if cfg.diurnal_shortwave:
        sw = sw * diurnal_factor(state.time, grid.lat, grid.lon)
    q_ns = sfc(forcing.q_nonsolar)
    fw = sfc(forcing.freshwater)
    T_clim = cyclic_interp(forcing.T_clim, year_frac)
    S_clim = cyclic_interp(forcing.S_clim, year_frac)
    dz0 = grid.dz[0]
    one = jnp.ones(())

    # ---- 1. KPP ----------------------------------------------------------------------
    jerlov = JERLOV_TYPES[cfg.shortwave.water_type]
    transmitted = (jerlov.R * jnp.exp(-state.h_bl / jerlov.zeta1)
                   + (1.0 - jerlov.R) * jnp.exp(-state.h_bl / jerlov.zeta2))
    q_kpp = q_ns + sw * (1.0 - transmitted)
    B_f, Q_T, Q_S = kpp_surface_fluxes(state.T[0], state.S[0], q_net=q_kpp, freshwater=fw)
    mix = kpp_vertical_mixing(
        state.u, state.v, state.T, state.S, jnp.zeros(()), grid.z_coord, one, cfg.kpp,
        tau_x=tau_x, tau_y=tau_y, B_f=B_f, Q_sfc_T=Q_T, Q_sfc_S=Q_S, apply_diffusion=False,
    )

    # ---- 2. temperature and salinity -------------------------------------------------
    dT = shortwave_penetration_tendency(sw, grid.dz, grid.z_coord.z_half_ref, one, cfg.shortwave)
    dT = dT.at[0].add(q_ns / (rho_0 * c_sw * dz0))
    dT = dT + mix.dT_dt + (T_clim - state.T) / (cfg.tau_nudge_T_days * constants.seconds_per_day)
    dS = mix.dS_dt + (S_clim - state.S) / (cfg.tau_nudge_S_days * constants.seconds_per_day)
    dS = dS.at[0].add(-state.S[0] * fw / (rho_0 * dz0))
    TS = implicit_vertical_diffusion_ocean(
        jnp.stack([state.T + dt * dT, state.S + dt * dS]), mix.K_v, grid.dz, grid.dz_half, dt)

    # ---- 3. momentum -------------------------------------------------------------------
    c, s = jnp.cos(grid.f_coriolis * dt), jnp.sin(grid.f_coriolis * dt)
    u = c * state.u + s * state.v
    v = -s * state.u + c * state.v
    u = u.at[0].add(dt * tau_x / (rho_0 * dz0))
    v = v.at[0].add(dt * tau_y / (rho_0 * dz0))
    if cfg.u_damping_days > 0.0:
        damp = jnp.exp(-dt / (cfg.u_damping_days * constants.seconds_per_day))
        u, v = u * damp, v * damp
    uv = implicit_vertical_diffusion_ocean(jnp.stack([u, v]), mix.A_v, grid.dz, grid.dz_half, dt)

    # ---- 4. MARBL ------------------------------------------------------------------------
    nlev = grid.dz.shape[0]
    interior_forcing = MarblInteriorForcing(
        potemp=state.T, salinity=state.S, pressure_bar=grid.p_bar,
        surf_shortwave=(sw * (1.0 - sfc(forcing.ifrac)))[None],
        PAR_col_frac=jnp.ones((1,)),
        dust_flux=sfc(forcing.dust_flux),
        fesedflux=jnp.zeros(nlev), feventflux=jnp.zeros(nlev),
    )
    tend, idiag = marbl_interior_tendencies(
        state.bgc, interior_forcing, grid.dz, grid.zt, grid.zw, jnp.asarray(nlev),
        marbl_cfg, n_iter=cfg.carbonate_n_iter)
    surface_forcing = MarblSurfaceForcing(
        sst=state.T[0], sss=state.S[0], u10_sqr=sfc(forcing.u10_sqr),
        atm_press=sfc(forcing.atm_press), ifrac=sfc(forcing.ifrac),
        xco2=sfc(forcing.xco2), xco2_alt_co2=sfc(forcing.xco2),
        dust_flux=sfc(forcing.dust_flux), iron_flux=sfc(forcing.iron_flux),
        nox_flux=sfc(forcing.nox_flux), nhy_flux=sfc(forcing.nhy_flux),
    )
    surf_top = jax.tree_util.tree_map(lambda a: a[0], state.bgc)
    sflux, sdiag = marbl_surface_fluxes(surf_top, surface_forcing, marbl_cfg,
                                        n_iter=cfg.carbonate_n_iter)
    bgc = _stack(state.bgc) + dt * _stack(tend)
    bgc = bgc.at[:, 0].add(dt * _stack(sflux) / dz0)
    bgc = implicit_vertical_diffusion_ocean(bgc, mix.K_v, grid.dz, grid.dz_half, dt)

    new_state = ColumnState(T=TS[0], S=TS[1], u=uv[0], v=uv[1], bgc=_unstack(bgc),
                            h_bl=mix.h_bl, time=state.time + dt)
    diag = ColumnStepDiagnostics(
        h_bl=mix.h_bl, sst=state.T[0],
        NPP=jnp.sum(idiag.photoC * grid.dz, axis=-1),
        POC_flux_100m=_flux_at_depth(idiag.POC_flux_out, grid.zw, 100.0),
        CaCO3_flux_100m=_flux_at_depth(idiag.CaCO3_flux_out, grid.zw, 100.0),
        flux_co2=sdiag.flux_co2, flux_o2=sdiag.flux_o2,
        pCO2surf=sdiag.pCO2surf, pH_surf=sdiag.pH_surf,
        POC_sed_loss=idiag.POC_sed_loss, CaCO3_sed_loss=idiag.CaCO3_sed_loss,
    )
    return new_state, diag


def run_column_raw(
    state0: ColumnState,
    forcing: ColumnForcing,
    grid: ColumnGrid,
    cfg: ColumnConfig,
    marbl_cfg: MarblConfig,
    n_days: int,
    steps_per_day: int,
) -> tuple[ColumnState, ColumnState, ColumnStepDiagnostics]:
    """Integrate ``n_days`` days (non-jitted; use inside jax.grad).

    Returns (final state, end-of-day states, daily-mean diagnostics); the
    daily outputs have a leading axis of length ``n_days``.  Each day is
    rematerialised in the backward pass (``jax.checkpoint``).
    """
    if steps_per_day * cfg.dt != constants.seconds_per_day:
        raise ValueError(f"steps_per_day * dt must equal one day, got {steps_per_day} * {cfg.dt}")

    def one_step(st, _):
        return column_step(st, forcing, grid, cfg, marbl_cfg)

    @jax.checkpoint
    def one_day(st, _):
        st, diags = jax.lax.scan(one_step, st, None, length=steps_per_day)
        mean = jax.tree_util.tree_map(lambda a: jnp.mean(a, axis=0), diags)
        return st, (st, mean)

    final, (daily_states, daily_diags) = jax.lax.scan(one_day, state0, None, length=n_days)
    return final, daily_states, daily_diags


run_column = jax.jit(run_column_raw, static_argnames=("cfg", "n_days", "steps_per_day"))
run_column.__doc__ = "Jitted ``run_column_raw`` (static: cfg, n_days, steps_per_day)."


def initial_state(T: jnp.ndarray, S: jnp.ndarray, bgc: MarblTracers,
                  h_bl0: float = 50.0, time0: float = 0.0) -> ColumnState:
    """Column at rest with given profiles."""
    zero = jnp.zeros_like(T)
    return ColumnState(T=T, S=S, u=zero, v=zero, bgc=bgc,
                       h_bl=jnp.asarray(h_bl0, T.dtype), time=jnp.asarray(time0, T.dtype))
