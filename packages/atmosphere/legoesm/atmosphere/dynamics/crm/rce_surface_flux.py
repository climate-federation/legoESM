"""RCE surface flux composer — layout-agnostic across NH states.

Provides a scalar-only (heat + moisture) bulk-aerodynamic surface
flux tendency that injects sensible + latent fluxes at the lowest
model level of any non-hydrostatic state pytree:

* :class:`legoesm.core.state.PlaneNonHydrostaticState` (3D arrays)
* :class:`legoesm.core.state.NonHydrostaticState` (cubed-sphere, 4D)
* :class:`legoesm.core.state.MPASNonHydrostaticState` (Voronoi, 2D)

All three share the convention that ``state.theta_prime`` and
``state.rho_prime`` have ``nlev`` as the LAST axis, and
``state.tracers`` carries ``(..., nlev, n_tracers)`` — so the surface
slab extraction ``arr[..., k_sfc=-1]`` is identical across layouts.

Bulk formulas reused from :mod:`legoesm.core.bulk_flux`
(``simple_bulk_fluxes``) — no re-derivation.

Scope
-----
* SCALAR fluxes only (sensible heat → θ', latent heat → q_v).
* Momentum stress NOT applied here: would require edge→cell wind
  reconstruction on MPAS, which is layout-specific. Wind speed is
  passed as an explicit array argument — caller computes it however
  matches their grid (use :func:`wind_speed_at_lowest_level_plane` /
  ``..._cs`` helpers for the structured-grid layouts).
* No iteration over MO stability — uses constant ``C_h`` (Wing 2018
  RCEMIP1 convention with C_h ≈ 1.5e-3, gustiness floor 5 m/s).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.bulk_flux import simple_bulk_fluxes, apply_gustiness


# --- RCEMIP1 surface-flux closure constants (Wing et al. 2018, RCEMIP) ---
# Empirical closure coefficients shared by the RCE surface-flux composers below;
# named here (not inline signature literals) so a tuning change is one edit.
_RCEMIP1_C_H = 1.5e-3               # bulk heat/moisture exchange coefficient [-]
_RCEMIP1_GUSTINESS_FLOOR_MS = 5.0  # minimum surface wind speed (gustiness) [m/s]


def _validate_nh_state_shape(state, height_coord) -> None:
    """Cross-layout shape contract (Codex iter-2 strengthened).

    Requires the FULL horizontal + vertical contract:
    * ``rho_prime.shape == theta_prime.shape`` (no silent broadcast
      across mismatched horizontal layouts that would update the
      wrong density column);
    * ``tracers.shape[:-1] == theta_prime.shape`` (tracers cell-
      centred + vertically aligned with the scalar fields);
    * height-coord 1D arrays all match ``nlev``.
    """
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data
    tracers = state.tracers.data
    if rho_p.shape != theta_p.shape:
        raise ValueError(
            f"rho_prime shape {rho_p.shape} != theta_prime shape "
            f"{theta_p.shape} — silent broadcast would corrupt the "
            f"surface flux density column."
        )
    if tracers.shape[:-1] != theta_p.shape:
        raise ValueError(
            f"tracers shape {tracers.shape} must align with "
            f"theta_prime shape {theta_p.shape} on all axes except "
            f"the trailing tracer axis."
        )
    nlev = theta_p.shape[-1]
    for name in ("rho_ref", "theta_ref", "exner_ref", "dz"):
        arr = getattr(height_coord, name)
        if arr.shape != (nlev,):
            raise ValueError(
                f"height_coord.{name} shape {arr.shape} != (nlev={nlev},)."
            )


def _validate_qv_slot(qv_slot: int, n_tracers: int) -> None:
    if not isinstance(qv_slot, int):
        raise ValueError(
            f"qv_slot must be int; got {qv_slot!r} of type "
            f"{type(qv_slot).__name__}."
        )
    if qv_slot < 0 or qv_slot >= n_tracers:
        raise ValueError(
            f"qv_slot={qv_slot} out of range [0, {n_tracers}); "
            f"negative indices are not allowed (would silently wrap)."
        )


def compose_rce_surface_scalar_tendencies(
    state, height_coord,
    T_sfc, q_sfc, wind_speed,
    C_h: float = _RCEMIP1_C_H,
    gustiness_floor: float = _RCEMIP1_GUSTINESS_FLOOR_MS,
    qv_slot: int = 0,
):
    """Scalar (heat + moisture) surface flux tendencies at the lowest
    model level (k_sfc = -1).

    Layout-agnostic — works for plane (3D), cubed-sphere (4D),
    MPAS (2D) NH states because the surface-slab indexing is
    identical across them.

    Wind-speed convention (Codex iter-2)
    -----------------------------------
    Caller passes RAW lowest-level wind speed (no gustiness baked in).
    Composer applies the Wing 2018 RCEMIP1 gustiness floor here:
    ``|U|_eff = sqrt(|U|² + u_gust²)``. Setting ``gustiness_floor=0``
    disables it. The structured-grid helpers
    :func:`wind_speed_at_lowest_level_plane` /
    :func:`wind_speed_at_lowest_level_cs` return RAW wind speed
    (also Codex iter-2 — no double-counting).

    Parameters
    ----------
    state : NonHydrostaticState | MPASNonHydrostaticState |
        PlaneNonHydrostaticState
        NH prognostic state.
    height_coord : HeightCoordinate
        Provides ``rho_ref``, ``theta_ref``, ``exner_ref``, ``dz``.
    T_sfc, q_sfc : array
        Surface temperature [K] + saturation mixing ratio [kg/kg].
        Must broadcast to the horizontal-layout shape (i.e., shape
        of ``state.theta_prime.data[..., -1]``).
    wind_speed : array
        RAW wind speed at lowest model level [m/s]. Same horizontal
        shape as ``T_sfc``. Gustiness floor applied by composer.
    C_h : float
        Bulk scalar transfer coefficient (default 1.5e-3 per
        Wing 2018 RCEMIP1).
    gustiness_floor : float
        Gustiness floor ``u_gust`` [m/s]. Default 5.0 (Wing 2018).
        Set to 0 to disable.
    qv_slot : int
        Tracer-axis index of water vapor (default 0).

    Returns
    -------
    dtheta_prime_dt_sfc : array
        Tendency on theta_prime at k_sfc [K/s]. Same shape as
        ``state.theta_prime.data[..., -1]``.
    dq_v_dt_sfc : array
        Tendency on q_v at k_sfc [kg/kg/s]. Same shape.
    """
    _validate_nh_state_shape(state, height_coord)
    _validate_qv_slot(qv_slot, state.tracers.data.shape[-1])
    if gustiness_floor < 0.0:
        raise ValueError(
            f"gustiness_floor must be non-negative; got {gustiness_floor}."
        )
    k_sfc = -1
    theta_total_sfc = (
        height_coord.theta_ref[k_sfc] + state.theta_prime.data[..., k_sfc]
    )
    exner_sfc = height_coord.exner_ref[k_sfc]
    T_atm_sfc = theta_total_sfc * exner_sfc
    q_atm_sfc = state.tracers.data[..., k_sfc, qv_slot]
    rho_sfc = (
        height_coord.rho_ref[k_sfc] + state.rho_prime.data[..., k_sfc]
    )
    # Apply gustiness floor: |U|_eff = sqrt(|U|² + u_gust²) (Wing 2018).
    # Shared with the coupled bulk-flux paths (core.bulk_flux.apply_gustiness).
    # Pass the precomputed magnitude as the first component (v=0): the floor is
    # inside the single sqrt, so it stays AD-safe at calm wind.
    wind_speed_eff = apply_gustiness(wind_speed, 0.0, gustiness_floor)
    # Reuse coupler/bulk_flux.simple_bulk_fluxes for shflx + lhflx.
    # u/v not consumed for scalar-only fluxes (Cd=0 zeros the stress).
    zero_field = jnp.zeros_like(T_atm_sfc)
    _, _, shflx, lhflx = simple_bulk_fluxes(
        u_lowest=zero_field, v_lowest=zero_field,
        T_lowest=T_atm_sfc, q_lowest=q_atm_sfc,
        T_sfc=T_sfc, q_sfc=q_sfc,
        rho=rho_sfc, wind_speed=wind_speed_eff,
        Cd=0.0, Ch=C_h,
    )
    dz_sfc = height_coord.dz[k_sfc]
    # Sensible heat → θ' tendency: dθ/dt = Q_h / (ρ·c_pd·Δz·Π).
    dtheta_dt_sfc = shflx / (
        rho_sfc * constants.c_pd * dz_sfc * exner_sfc
    )
    # Latent heat → q_v tendency: dq_v/dt = E / (ρ·Δz), with E = LH / L, L the
    # latent heat the constant-coefficient law charged (Kirchhoff L_v(T_sfc)).
    from legoesm.thermo import charged_latent_heat
    dq_v_dt_sfc = lhflx / (charged_latent_heat("constant", T_sfc) * rho_sfc * dz_sfc)
    return dtheta_dt_sfc, dq_v_dt_sfc


def apply_rce_surface_fluxes(
    state, height_coord, dt,
    T_sfc, q_sfc, wind_speed,
    C_h: float = _RCEMIP1_C_H,
    gustiness_floor: float = _RCEMIP1_GUSTINESS_FLOOR_MS,
    qv_slot: int = 0,
):
    """Forward-Euler step of the scalar RCE surface flux tendencies.

    ``state ← state + dt · (dθ', dq_v) at k_sfc``. Other prognostic
    fields (u, v, w, rho_prime, phis, other tracers) untouched.

    Layout-agnostic — see :func:`compose_rce_surface_scalar_tendencies`.
    Caller passes RAW wind speed; gustiness floor applied by composer.
    """
    dtheta_dt_sfc, dq_v_dt_sfc = compose_rce_surface_scalar_tendencies(
        state, height_coord, T_sfc, q_sfc, wind_speed,
        C_h=C_h, gustiness_floor=gustiness_floor, qv_slot=qv_slot,
    )
    k_sfc = -1
    new_theta_prime = state.theta_prime.data.at[..., k_sfc].add(
        dt * dtheta_dt_sfc,
    )
    new_tracers = state.tracers.data.at[..., k_sfc, qv_slot].add(
        dt * dq_v_dt_sfc,
    )
    return state._replace(
        theta_prime=state.theta_prime.replace(data=new_theta_prime),
        tracers=state.tracers.replace(data=new_tracers),
    )


def wind_speed_at_lowest_level_plane(state) -> jax.Array:
    """RAW cell-centred wind speed at lowest model level (plane state).

    ``|U| = sqrt(u² + v²)`` — gustiness floor NOT applied here
    (Codex iter-2). The composer
    :func:`compose_rce_surface_scalar_tendencies` applies the
    Wing 2018 RCEMIP1 gustiness floor when combining this value
    with the bulk-flux coefficients, so callers MUST pass the raw
    value through to the composer; double-applying the floor here
    would inflate the scalar exchange.

    Assumes Arakawa-C u, v already approximately cell-centred for
    diagnostic purposes (acceptable since RCE has no mean wind after
    the Galilean filter — relative error ~O(dx·du/dx)).
    """
    if state.u.data.ndim != 3 or state.v.data.ndim != 3:
        raise ValueError(
            f"wind_speed_at_lowest_level_plane requires plane (3D) "
            f"u, v arrays; got u.ndim={state.u.data.ndim}, "
            f"v.ndim={state.v.data.ndim}."
        )
    u_sfc = state.u.data[..., -1]
    v_sfc = state.v.data[..., -1]
    return jnp.sqrt(u_sfc ** 2 + v_sfc ** 2)


def wind_speed_at_lowest_level_cs(state) -> jax.Array:
    """RAW cell-centred wind speed at lowest model level (cubed-sphere).

    Returns shape ``(6, n, n)``. Gustiness floor NOT applied here —
    see :func:`wind_speed_at_lowest_level_plane`.
    """
    if state.u.data.ndim != 4 or state.v.data.ndim != 4:
        raise ValueError(
            f"wind_speed_at_lowest_level_cs requires cubed-sphere (4D) "
            f"u, v arrays; got u.ndim={state.u.data.ndim}, "
            f"v.ndim={state.v.data.ndim}."
        )
    u_sfc = state.u.data[..., -1]
    v_sfc = state.v.data[..., -1]
    return jnp.sqrt(u_sfc ** 2 + v_sfc ** 2)
