"""FV3_3D iter 583: atmospheric angular momentum diagnostic.

Faithful port of FV3's ``compute_aam`` (fv_dynamics.F90:1264-1307).

Computes per-column mass-integrated atmospheric angular momentum
(AAM) and the total AAM (area-weighted).  In FV3 this drives
``consv_am`` correction and the ``id_aam``/``id_amdt``
diagnostics — for legoESM we expose it as a pure diagnostic.

Reference
---------
FV3 fv_dynamics.F90 compute_aam:

    aam(i,j) = sum_k ( (r²·Ω + r·ua) · dm )
    where r = R · cos(lat); dm = delp / g

For NH (legoESM): dm = rho_full · dz · area  (full mass per
cell).

For PE (legoESM): dm = delp · area / g  (mass per cell from
hybrid pressure).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants


def compute_atmospheric_angular_momentum(
    u_center: jax.Array,
    rho_full: jax.Array,
    grid,
    hc,
) -> tuple[jax.Array, float]:
    """Compute per-column AAM and total mass-weighted AAM.

    Parameters
    ----------
    u_center : jax.Array, shape (6, n, n, nlev)
        Cell-centered zonal wind component in face-local x.
        For NH state this is ``state.u.data`` directly.
        Note: FV3 uses *geographic* u (ua = u_east).  This
        function takes face-local u; rotation to u_east is
        the caller's responsibility for true AAM.  For drift
        monitoring without rotation, the face-local AAM is
        still a useful conserved-ish quantity.
    rho_full : jax.Array, shape (6, n, n, nlev)
        Full density (rho_0 + rho_prime).  In legoESM NH:
        broadcast ``hc.rho_ref[None, None, None, :]`` and add
        ``state.rho_prime.data``.
    grid : CubedSphereGrid
        Provides ``lat``, ``area``, ``radius``.
    hc : HeightCoordinate
        Provides ``dz`` (layer thickness, shape (nlev,)).

    Returns
    -------
    aam_column : jax.Array, shape (6, n, n)
        Per-column mass-integrated AAM [kg·m²/s].
    aam_total : float
        Globally-summed AAM [kg·m²/s].
    """
    R = float(grid.radius)
    cos_lat = jnp.cos(grid.lat)               # (6, n, n)
    r1 = R * cos_lat                          # (6, n, n)
    r2 = r1 * r1
    omega = constants.Omega

    # Per-cell mass: rho * dz * area (kg)
    dz_b = jnp.asarray(hc.dz)[None, None, None, :]  # (1, 1, 1, nlev)
    area_b = grid.area[..., None]                    # (6, n, n, 1)
    dm = rho_full * dz_b * area_b                    # (6, n, n, nlev)

    # AAM per cell: (r²·Ω + r·u) · dm
    r1_b = r1[..., None]                              # (6, n, n, 1)
    r2_b = r2[..., None]
    aam_cell = (r2_b * omega + r1_b * u_center) * dm
    # Column integral
    aam_column = jnp.sum(aam_cell, axis=-1)           # (6, n, n)
    aam_total = float(jnp.sum(aam_column))
    return aam_column, aam_total


def aam_from_nh_state(state, grid, hc) -> tuple[jax.Array, float]:
    """FV3_3D iter 587: AAM from a NonHydrostaticState, using geographic
    u_east (not face-local u).

    Faithful to FV3's compute_aam which uses ua (geographic east wind).
    Rotates the state's face-local (u, v) to (u_east, v_north) via
    grid.angle before computing AAM.

    Parameters
    ----------
    state : NonHydrostaticState
    grid : CubedSphereGrid
    hc : HeightCoordinate

    Returns
    -------
    aam_column, aam_total : as ``compute_atmospheric_angular_momentum``.
    """
    # Rotate face-local (u_face, v_face) to (u_east, v_north).
    # Inverse of rotate_winds_geo_to_grid:
    #   u_east = cos(angle)*u_face - sin(angle)*v_face
    angle = grid.angle[..., None]  # (6, n, n, 1)
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    u_east = cos_a * state.u.data - sin_a * state.v.data
    rho_ref_b = jnp.asarray(hc.rho_ref)[None, None, None, :]
    rho_full = rho_ref_b + state.rho_prime.data
    return compute_atmospheric_angular_momentum(
        u_east, rho_full, grid, hc,
    )


def aam_drift_nh(state_old, state_new, grid, hc) -> float:
    """FV3_3D iter 587: AAM tendency between two NH states.

    Returns amdt = AAM(state_new) - AAM(state_old), faithful to FV3
    fv_dynamics.F90:768 ``amdt = g_sum(te_2d, ...)``.

    A positive value means AM was injected by the dycore.  In
    adiabatic flat-surface runs amdt should be ~0 (the dycore
    should conserve AM exactly if flux-form & no friction).
    Mountain torque (FV3 ``zxg`` term) is NOT included here — for
    runs with terrain, the physical AM tendency includes
    dt·sum(ps·zxg·area) and this function returns only the
    dycore-internal drift.

    Returns
    -------
    amdt : float
        AAM tendency [kg·m²/s].
    """
    _, aam_old = aam_from_nh_state(state_old, grid, hc)
    _, aam_new = aam_from_nh_state(state_new, grid, hc)
    return aam_new - aam_old
