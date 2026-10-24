"""MARBL sources and sinks on a 3-D (lat, lon, level) ocean state.

Applies the column MARBL port (``marbl_interior_tendencies`` +
``marbl_surface_fluxes``) to every water column -- the same calls the 1-D
column model makes (``ocean/column/model.py``, step 4) -- and returns the
forward-Euler update (no clipping, as MARBL in POP) plus 2-D diagnostics.

Transport (advection, horizontal diffusion, vertical mixing) is NOT done here:
the ocean model moves ``MarblTracers`` fields with the same operators as T and S.

Columns are processed with ``jax.lax.map(..., batch_size=...)`` so memory stays
bounded at high resolution.  MARBL arithmetic runs in float64 when
``JAX_ENABLE_X64=1`` (recommended: the carbonate solver and particle scans need
consistent dtypes); results are cast back to the state's dtype.

Place at: src/legoesm/ocean/biogeochemistry/marbl/driver3d.py
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.biogeochemistry.marbl.config import (
    TRACER_NAMES,
    MarblConfig,
    MarblInteriorForcing,
    MarblSurfaceForcing,
    MarblTracers,
)
from legoesm.ocean.biogeochemistry.marbl.interface import marbl_interior_tendencies
from legoesm.ocean.biogeochemistry.marbl.surface import marbl_surface_fluxes

RHO_SW = 1026.0          # [kg/m^3], for sea pressure only (MARBL rho_sw)
PA_PER_BAR = 1.0e5
MIN_DZ = 1.0e-3          # [m]; thinner layers are treated as inactive


class Marbl2DForcing(NamedTuple):
    """Surface forcing on the model grid, each (n_lat, n_lon) (xco2 may be scalar)."""
    sw_net: jnp.ndarray      # net shortwave into the ocean, before sea ice [W/m^2]
    u10_sqr: jnp.ndarray     # squared 10-m wind speed [m^2/s^2]
    atm_press: jnp.ndarray   # [atm]
    ifrac: jnp.ndarray       # sea-ice fraction [0-1]
    xco2: jnp.ndarray        # [ppmv]
    dust_flux: jnp.ndarray   # [kg/m^2/s]
    iron_flux: jnp.ndarray   # soluble Fe deposition [mmol/m^2/s]
    nox_flux: jnp.ndarray    # [mmol N/m^2/s]
    nhy_flux: jnp.ndarray    # [mmol N/m^2/s]


class Marbl2DDiagnostics(NamedTuple):
    NPP: jnp.ndarray            # (3, n_lat, n_lon) vertically integrated C fixation [mmol C/m^2/s]
    POC_flux_100m: jnp.ndarray  # [mmol C/m^2/s]
    flux_co2: jnp.ndarray       # [mmol/m^2/s], + into ocean
    flux_o2: jnp.ndarray        # [mmol/m^2/s], + into ocean
    pCO2surf: jnp.ndarray       # [ppmv]
    pH_surf: jnp.ndarray
    Chl_surf: jnp.ndarray       # total surface Chl [mg/m^3]


def _column(bgc, T, S, dz, kmt, f, zt, zw, cfg, n_iter, dt):
    """One column: forward-Euler MARBL update and diagnostics (all arrays 1-D)."""
    nlev = dz.shape[0]
    p_bar = RHO_SW * constants.g * zt / PA_PER_BAR
    interior = MarblInteriorForcing(
        potemp=T, salinity=S, pressure_bar=p_bar,
        surf_shortwave=(f.sw_net * (1.0 - f.ifrac))[None],
        PAR_col_frac=jnp.ones((1,), dz.dtype),
        dust_flux=f.dust_flux,
        fesedflux=jnp.zeros(nlev, dz.dtype),
        feventflux=jnp.zeros(nlev, dz.dtype),
    )
    tend, idiag = marbl_interior_tendencies(bgc, interior, dz, zt, zw, kmt, cfg, n_iter=n_iter)
    sforc = MarblSurfaceForcing(
        sst=T[0], sss=S[0], u10_sqr=f.u10_sqr, atm_press=f.atm_press, ifrac=f.ifrac,
        xco2=f.xco2, xco2_alt_co2=f.xco2, dust_flux=f.dust_flux, iron_flux=f.iron_flux,
        nox_flux=f.nox_flux, nhy_flux=f.nhy_flux,
    )
    top = MarblTracers(**{k: getattr(bgc, k)[0] for k in TRACER_NAMES})
    sflux, sdiag = marbl_surface_fluxes(top, sforc, cfg, n_iter=n_iter)

    new = MarblTracers(**{
        k: (getattr(bgc, k) + dt * getattr(tend, k)).at[0].add(dt * getattr(sflux, k) / dz[0])
        for k in TRACER_NAMES
    })
    z_i = jnp.concatenate([jnp.zeros(1, zw.dtype), zw])
    f_i = jnp.concatenate([jnp.zeros(1, dz.dtype), idiag.POC_flux_out])
    diag = (
        jnp.sum(idiag.photoC * dz, axis=-1),                 # (3,)
        jnp.interp(100.0, z_i, f_i),
        sdiag.flux_co2, sdiag.flux_o2, sdiag.pCO2surf, sdiag.pH_surf,
        bgc.spChl[0] + bgc.diatChl[0] + bgc.diazChl[0],
    )
    return new, diag


def marbl_step_3d(
    bgc: MarblTracers,
    T: jnp.ndarray,
    S: jnp.ndarray,
    h_k: jnp.ndarray,
    forcing: Marbl2DForcing,
    zt_ref: jnp.ndarray,
    zw_ref: jnp.ndarray,
    dz_ref: jnp.ndarray,
    cfg: MarblConfig,
    dt: float,
    n_iter: int = 50,
    batch_size: int = 4096,
) -> tuple[MarblTracers, Marbl2DDiagnostics]:
    """Advance MARBL tracers by one source/sink step on a (n_lat, n_lon, nlev) grid.

    Parameters
    ----------
    bgc : MarblTracers, each (n_lat, n_lon, nlev) [mmol/m^3]
    T, S : (n_lat, n_lon, nlev) potential temperature [degC], salinity [PSU]
    h_k : (n_lat, n_lon, nlev) actual layer thickness [m]; ~0 below the seafloor / on land
    forcing : Marbl2DForcing on (n_lat, n_lon)
    zt_ref, zw_ref, dz_ref : (nlev,) reference layer-centre depth, layer-bottom depth,
        thickness [m], positive down (zt = -z_full_ref, zw = -z_half_ref[1:]).
    """
    n_lat, n_lon, nlev = T.shape
    ncol = n_lat * n_lon
    f64 = lambda a: jnp.asarray(a, jnp.float64)  # noqa: E731  (float32 if x64 is off)
    out_dtype = {k: getattr(bgc, k).dtype for k in TRACER_NAMES}

    active = h_k > MIN_DZ                                       # (n_lat, n_lon, nlev)
    kmt = jnp.sum(active, axis=-1).astype(jnp.int32)            # wet levels per column
    dz = jnp.where(active, f64(h_k), f64(dz_ref))               # dummy thickness below seafloor

    col = lambda a: f64(a).reshape(ncol, nlev)                  # noqa: E731
    sfc = lambda a: jnp.broadcast_to(f64(a), (n_lat, n_lon)).reshape(ncol)  # noqa: E731
    xs = (
        MarblTracers(**{k: col(getattr(bgc, k)) for k in TRACER_NAMES}),
        col(T), col(S), dz.reshape(ncol, nlev), jnp.maximum(kmt, 1).reshape(ncol),
        Marbl2DForcing(*[sfc(a) for a in forcing]),
    )
    zt, zw = f64(zt_ref), f64(zw_ref)

    def one(x):
        b, t, s, d, k, f = x
        return _column(b, t, s, d, k, f, zt, zw, cfg, n_iter, dt)

    new_cols, diag_cols = jax.lax.map(one, xs, batch_size=batch_size)

    new = MarblTracers(**{
        k: jnp.where(active, getattr(new_cols, k).reshape(n_lat, n_lon, nlev),
                     getattr(bgc, k)).astype(out_dtype[k])
        for k in TRACER_NAMES
    })
    wet = kmt > 0
    npp, poc100, fco2, fo2, pco2, ph, chl = diag_cols
    diag = Marbl2DDiagnostics(
        NPP=jnp.where(wet, npp.T.reshape(3, n_lat, n_lon), 0.0),
        POC_flux_100m=jnp.where(wet, poc100.reshape(n_lat, n_lon), 0.0),
        flux_co2=jnp.where(wet, fco2.reshape(n_lat, n_lon), 0.0),
        flux_o2=jnp.where(wet, fo2.reshape(n_lat, n_lon), 0.0),
        pCO2surf=jnp.where(wet, pco2.reshape(n_lat, n_lon), jnp.nan),
        pH_surf=jnp.where(wet, ph.reshape(n_lat, n_lon), jnp.nan),
        Chl_surf=jnp.where(wet, chl.reshape(n_lat, n_lon), jnp.nan),
    )
    return new, diag


def make_marbl_stepper(cfg: MarblConfig, dt: float, zt_ref, zw_ref, dz_ref,
                       n_iter: int = 50, batch_size: int = 4096):
    """Jitted ``(bgc, T, S, h_k, forcing) -> (bgc_new, diag)`` with config baked in."""
    zt_ref, zw_ref, dz_ref = (jnp.asarray(a, jnp.float64) for a in (zt_ref, zw_ref, dz_ref))

    @jax.jit
    def step(bgc, T, S, h_k, forcing):
        return marbl_step_3d(bgc, T, S, h_k, forcing, zt_ref, zw_ref, dz_ref,
                             cfg, dt, n_iter=n_iter, batch_size=batch_size)

    return step
