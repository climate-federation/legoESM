"""Newtonian (Haney 1971) surface-tracer relaxation toward a climatology.

A pure, JAX-traceable, AD-safe array kernel that relaxes the ocean
surface-layer temperature and salinity toward a fixed climatological
target on a finite timescale ``tau``:

    dq/dt|_restore = - (q_surf - q_target) / tau          (Haney 1971)

discretised as the per-step fraction ``alpha = dt / tau``:

    q_surf_new = q_surf - alpha * (q_surf - q_target) * mask

This is the canonical home for the relaxation math shared by

  * the OMIP standalone driver (``run_omip._apply_restoring`` FV branch — the
    grid-agnostic SST/SSS restoring toward WOA), and
  * the coupled 3D-ocean spin-up (``apply_surface_relaxation_step`` below,
    used by ``CoupledESMDriver._step_ocean`` to anchor the WOA-initialised 3D
    ocean against the cold-start drift while the atmosphere equilibrates).

It is deliberately a TRIVIAL Newtonian relaxation (no region masks, ice
gating, or virtual-salt-flux conversion).  The richer OMIP-2 SSS restoring
with region masks / ice gating / flux caps lives in
:mod:`legoesm.ocean.forcing.sss_restoring` + :mod:`legoesm.ocean.coupler.sss_apply`
(a NumPy host-loop path); this module is the JAX hot-loop counterpart.
"""

from __future__ import annotations

import jax.numpy as jnp


def relax_surface_tracers(
    T_top, S_top, T_target, S_target, alpha_T, alpha_S, mask,
):
    """Relax surface-layer ``T``/``S`` toward a target by per-step fraction.

    Parameters
    ----------
    T_top, S_top : array
        Current surface-layer temperature / salinity (any grid layout).
    T_target, S_target : array
        Climatological surface target, broadcastable to ``T_top``/``S_top``.
    alpha_T, alpha_S : float or scalar array
        Per-step relaxation fraction ``dt / tau`` (0 => no relaxation).
    mask : array
        Wet-cell mask (1 = ocean, 0 = land); dry cells are left untouched.

    Returns
    -------
    (T_top_new, S_top_new) : tuple of arrays
    """
    T_new = T_top - alpha_T * (T_top - T_target) * mask
    S_new = S_top - alpha_S * (S_top - S_target) * mask
    return T_new, S_new


def apply_surface_relaxation_step(
    state,
    *,
    T_target,
    S_target,
    dt: float,
    tau_T_s: float,
    tau_S_s: float,
):
    """Apply one step of surface T/S relaxation to a lat-lon C-grid ocean state.

    Updates only the top model level (index 0), land-masked via
    ``state.land_mask``.  A non-positive ``tau`` disables that tracer's
    relaxation (``alpha = 0`` => unchanged).  Pure / AD-safe.

    Parameters
    ----------
    state : LatLonCGridOceanState
        ``state.T`` / ``state.S`` are ``Field`` s of shape
        ``(n_lat, n_lon, nlev)``; ``state.land_mask`` is ``(n_lat, n_lon)``.
    T_target, S_target : array ``(n_lat, n_lon)``
        Climatological surface target (T in the state's units [degC], S [PSU]).
    dt : float
        Coupling step [s].
    tau_T_s, tau_S_s : float
        Relaxation timescales [s] (<= 0 => that tracer is not relaxed).

    Returns
    -------
    new_state : LatLonCGridOceanState
    """
    mask = state.land_mask.data
    alpha_T = 0.0 if tau_T_s <= 0.0 else dt / tau_T_s
    alpha_S = 0.0 if tau_S_s <= 0.0 else dt / tau_S_s
    if alpha_T == 0.0 and alpha_S == 0.0:
        return state
    T = state.T.data
    S = state.S.data
    T_new_top, S_new_top = relax_surface_tracers(
        T[..., 0], S[..., 0], T_target, S_target, alpha_T, alpha_S, mask,
    )
    return state._replace(
        T=state.T.replace(data=T.at[..., 0].set(jnp.asarray(T_new_top, T.dtype))),
        S=state.S.replace(data=S.at[..., 0].set(jnp.asarray(S_new_top, S.dtype))),
    )


__all__ = ["relax_surface_tracers", "apply_surface_relaxation_step"]
