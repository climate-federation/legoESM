"""External (coupler-provided) surface forcing.

Applies an ``OceanSurfaceForcing`` struct's tau / q_net / freshwater / salt to
the top ocean layer through the PHYSICS-function path.  The cubed-sphere
(``OceanModel``) dynamics route surface forcing through ``physics_fn`` (the
lat-lon C-grid applies ``surface_forcing`` directly in its dynamics), so this
scheme is how a prognostic cubed-sphere ocean closes two-way coupling (e.g. the
sea-ice -> ocean exchange via ``coupler.ocean_forcing``).

Scope: the cubed-sphere ``OceanModel`` (``make_ocean_physics`` ->
``make_surface_forcing_physics``).  MPAS builds physics through
``make_mpas_ocean_physics`` (a SEPARATE factory) which has its OWN edge-projected
external block (it also accepts ``scheme="external"``); MPAS does NOT route
through this module.

Sign / unit conventions match the lat-lon C-grid direct application
(``ocean_pe_latlon_cgrid.py``) so the SAME ``OceanSurfaceForcing`` produces the
same physics on every grid that consumes it:
  - momentum: ocean reaction = -tau (ATMOSPHERE convention), so an on-ocean
    stress passed as ``-stress`` (as the F11 helper does) accelerates the ocean
    in the stress direction.  NOTE this is the OPPOSITE tau sign of the
    ``prescribed`` scheme, which applies +tau as an on-ocean stress directly.
    The ``external`` scheme is for coupler/atmosphere-convention forcing; do not
    feed it an on-ocean stress without negating.
  - heat: dT/dt = q_net / (rho_0 c_sw dz_0), q_net > 0 INTO ocean.  The FULL
    ``q_net`` is deposited in the surface layer; this scheme does NOT separately
    penetrate ``sw_down`` through the column.  If the ocean physics also enables
    shortwave penetration (``ShortwavePenetrationConfig``) it adds ``sw_down``
    heating on TOP of this — so with ``external`` fold solar into ``q_net`` and
    leave ``sw_down=None`` (the F11 helper does) to avoid double counting.
  - salinity: virtual salt from freshwater (-S*fw/(rho_0 dz_0), fw>0 freshens)
    PLUS the real salt-mass flux (+salt*1e3/(rho_0 dz_0), salt>0 salinifies).

FRESHWATER CLOSURE: the ``freshwater`` channel is applied here as a VIRTUAL
salt flux (salinity dilution at fixed ocean volume) — the standard z-coord /
rigid-lid closure.  It does NOT add freshwater mass to ``eta``: the cubed-sphere
``OceanModel`` updates ``eta`` only through the barotropic continuity solver and
has no surface mass source, and the physics-function route
(``SurfaceForcingOutput``) carries no eta tendency.  So melt/freeze freshens /
salinifies correctly (the dominant ice->ocean stratification feedback) but the
small sea-level/volume response is omitted.  This differs from the lat-lon
``freshwater=`` path, which ALSO adds ``deta/dt = F_fw/rho_0`` outside the
physics function; wiring a real-freshwater eta source into the cubed-sphere
barotropic solver is deferred (tracked with MPAS two-way).

The top-layer thickness ``dz_0`` is the ACTUAL partial-cell-aware thickness
(``compute_layer_thickness(...)[..., 0]``), not ``dz_ref[0]*J`` — so flux-to-
tendency conversion is conservative on shallow/partial top cells.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.ocean.eos import rho_0 as rho_0_ref, c_sw
from legoesm.ocean.physics.surface_forcing._shared import (
    WindStressConvention,
    surface_tendency_factors,
    wind_stress_sign,
)
from legoesm.ocean.physics.surface_forcing.output import SurfaceForcingOutput

__physics_contract__ = {
    "summary": (
        "External (coupler-provided) surface forcing: apply an "
        "OceanSurfaceForcing struct's wind stress, net heat, freshwater and "
        "salt fluxes to the top ocean layer through the physics-function path "
        "(cubed-sphere two-way coupling)."
    ),
    "inputs": {
        "tau_x": "N/m^2", "tau_y": "N/m^2", "q_net": "W/m^2",
        "freshwater": "kg/m^2/s", "salt_flux": "kg/m^2/s", "S": "psu",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "degC/s", "dS_dt": "psu/s",
        "Q_net": "W/m^2", "tau_x": "N/m^2", "tau_y": "N/m^2",
    },
    "sign_convention": (
        "Top-layer boundary sources (NOT interior-conservative); ATMOSPHERE "
        "stress convention — ocean reaction = -tau, so a stress passed as "
        "-stress accelerates the ocean in the stress direction (OPPOSITE the "
        "prescribed scheme's +tau); q_net > 0 goes INTO the ocean (warming); "
        "freshwater > 0 freshens via a virtual salt flux at fixed volume (no eta "
        "source); real salt > 0 salinifies; z positive up. Unknown scheme "
        "raises ValueError."
    ),
    # Coupler-provided surface boundary source/sink; not interior-conservative.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Standard z-coordinate / rigid-lid surface flux boundary conditions "
        "(virtual-salt freshwater closure); Griffies (2004)"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_surface_forcing_sign_convention.py + "
        "tests/ocean/unit/test_surface_forcing_flux_feedback.py — a positive "
        "q_net warms the surface layer; a -tau input accelerates the ocean "
        "along the stress; freshwater freshens; an unknown scheme raises."
    ),
}


def external_surface_forcing(
    u: jnp.ndarray,
    v: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    dz_0: jnp.ndarray,
    surface_forcing,
    min_wet_cell_thickness_m: float = 1.0e-3,  # coeff-ok: wet-cell thickness floor [m]
) -> SurfaceForcingOutput:
    """Apply a coupler ``OceanSurfaceForcing`` to the top layer.  Returns zero
    tendencies for any channel that is ``None``.

    Parameters
    ----------
    u, v, T, S : arrays, shape (..., nlev)
        Prognostic fields (u/v may be C-grid-staggered relative to T).
    dz_0 : array, shape (...)
        ACTUAL top-layer thickness [m] at T points (partial-cell aware), e.g.
        ``compute_layer_thickness(eta, H_bathy, z_coord)[..., 0]``.  Zero on
        land columns (used as the wet/dry mask).
    surface_forcing : OceanSurfaceForcing
        Coupler-provided tau_x / tau_y / q_net / freshwater / salt_flux.
    """
    nlev = u.shape[-1]
    dtype = u.dtype

    is_ocean = dz_0 > min_wet_cell_thickness_m
    # Wet-cell flux→tendency reciprocals (#518: shared helper; eos rho_0/c_sw).
    # inv_rho_dz serves BOTH the momentum and the salt (virtual+real) paths.
    inv_rho_dz, inv_rho_csw_dz = surface_tendency_factors(
        is_ocean, dz_0, rho_0_ref, c_sw)
    zT = jnp.zeros_like(dz_0)

    tau_x = getattr(surface_forcing, "tau_x", None) if surface_forcing else None
    tau_y = getattr(surface_forcing, "tau_y", None) if surface_forcing else None
    q_net = getattr(surface_forcing, "q_net", None) if surface_forcing else None
    fw = getattr(surface_forcing, "freshwater", None) if surface_forcing else None
    salt = getattr(surface_forcing, "salt_flux", None) if surface_forcing else None

    # --- Momentum: ocean reaction = -tau (atmosphere convention) ---
    # ATMOSPHERE_REACTION convention, named for self-documentation (#518 item
    # 11); sign is a static -1.0 — behaviour unchanged (-1.0*x == -x).
    _tau_sign = wind_stress_sign(WindStressConvention.ATMOSPHERE_REACTION)
    if tau_x is not None:
        du_dt_T = _tau_sign * jnp.asarray(tau_x, dtype) * inv_rho_dz
    else:
        du_dt_T = zT
    if tau_y is not None:
        dv_dt_T = _tau_sign * jnp.asarray(tau_y, dtype) * inv_rho_dz
    else:
        dv_dt_T = zT

    # C-grid staggering detection (mirror prescribed_surface_forcing): lat-lon
    # C-grid has u at (n_lat, n_lon+1) / v at (n_lat+1, n_lon); cubed-sphere /
    # A-grid share T's 2-D shape (no interpolation).
    T_2d, u_2d, v_2d = inv_rho_dz.shape, u.shape[:-1], v.shape[:-1]
    is_cgrid_u = (len(u_2d) == 2 and len(T_2d) == 2
                  and u_2d[0] == T_2d[0] and u_2d[1] == T_2d[1] + 1)
    is_cgrid_v = (len(v_2d) == 2 and len(T_2d) == 2
                  and v_2d[0] == T_2d[0] + 1 and v_2d[1] == T_2d[1])
    if is_cgrid_u:
        du_dt_uf = 0.5 * (du_dt_T + jnp.roll(du_dt_T, 1, axis=1))
        du_dt_uf = jnp.concatenate([du_dt_uf, du_dt_uf[:, 0:1]], axis=1)
    else:
        du_dt_uf = du_dt_T
    if is_cgrid_v:
        dv_dt_int = 0.5 * (dv_dt_T[:-1] + dv_dt_T[1:])
        dv_dt_vf = jnp.pad(dv_dt_int, ((1, 1), (0, 0)))
    else:
        dv_dt_vf = dv_dt_T

    pad_u = ((0, 0),) * (len(u.shape) - 1)
    pad_v = ((0, 0),) * (len(v.shape) - 1)
    pad_T = ((0, 0),) * (len(T.shape) - 1)
    du_dt = jnp.pad(du_dt_uf[..., None], (*pad_u, (0, nlev - 1)))
    dv_dt = jnp.pad(dv_dt_vf[..., None], (*pad_v, (0, nlev - 1)))

    # --- Heat: dT/dt = q_net / (rho_0 c_sw dz_0) ---
    if q_net is not None:
        dT_top = jnp.asarray(q_net, dtype) * inv_rho_csw_dz
        Q_net = jnp.asarray(q_net, dtype)
    else:
        dT_top = zT
        Q_net = zT
    dT_dt = jnp.pad(dT_top[..., None], (*pad_T, (0, nlev - 1)))

    # --- Salinity: virtual salt (freshwater) + real salt mass ---
    # Reuses inv_rho_dz (same 1/(rho_0*dz) factor as momentum).
    dS_top = zT
    if fw is not None:
        dS_top = dS_top - S[..., 0] * jnp.asarray(fw, dtype) * inv_rho_dz
    if salt is not None:
        dS_top = dS_top + jnp.asarray(salt, dtype) * 1.0e3 * inv_rho_dz
    dS_dt = jnp.pad(dS_top[..., None], (*pad_T, (0, nlev - 1)))

    return SurfaceForcingOutput(
        du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, dS_dt=dS_dt,
        Q_net=Q_net, tau_x=(tau_x if tau_x is not None else zT),
        tau_y=(tau_y if tau_y is not None else zT),
    )
