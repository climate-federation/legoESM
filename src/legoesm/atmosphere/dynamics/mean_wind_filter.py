"""Domain-mean wind removal — plane (uniform Cartesian) CRM only.

SCOPE (Codex iter-1): this helper is ONLY valid for the doubly-
periodic plane CRM where horizontal cell area is uniform and the
prognostic state is vertical-last ``(ny, nx, nlev)``. Cubed-sphere,
lat-lon, MPAS, and any density/area-weighted setup require a
weighted-mean variant that this module does NOT provide.

On the plane, an unweighted arithmetic mean equals the proper
area-weighted mean (uniform area), so this implementation is
exact for the plane and INTENTIONALLY narrow.

Physics
-------
On a small periodic plane, no large-scale forcing dissipates the
horizontal-mean momentum that accumulates from net surface stress
/ radiative acceleration / spin-up transient. Standard remedy
(Held et al. 2007, Bretherton & Khairoutdinov 2015): periodically
subtract the per-level horizontal-mean ``u`` and ``v``. Equivalent
to a Galilean transformation cancelling the spurious mean drift.

The operation on a uniform plane is mass-neutral (mean wind has
zero divergence under periodic BC) and strictly REDUCES THE
UNWEIGHTED HORIZONTAL VELOCITY VARIANCE (the physical
``∫ ρ |u|²/2 dV`` invariant requires density weighting and is
not the quantity being monotonically reduced here — Codex iter-1).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _validate_plane_state(state, fn_name: str) -> None:
    """Plane-state contract enforcement (Codex iter-1).

    Hard-fail when:
    * State doesn't have the Field+NamedTuple protocol → TypeError.
    * State.u.data is not 3D ``(ny, nx, nlev)`` → ValueError. Any
      cubed-sphere ``(6, n, n, nlev)``, MPAS ``(nEdges, nlev)``, or
      lat-lon C-grid layout fails here, since the unweighted
      arithmetic mean is not the correct domain mean for those
      grids.
    """
    if not (
        hasattr(state, "u") and hasattr(state.u, "data")
        and hasattr(state.u, "replace")
        and hasattr(state, "v") and hasattr(state.v, "data")
        and hasattr(state.v, "replace")
        and hasattr(state, "_replace")
    ):
        raise TypeError(
            f"{fn_name} requires a state with `u` and `v` Fields "
            f"exposing .data + .replace, plus NamedTuple _replace; "
            f"got {type(state).__name__}."
        )
    if state.u.data.ndim != 3:
        raise ValueError(
            f"{fn_name} is plane-only: expects state.u.data with "
            f"shape (ny, nx, nlev); got ndim={state.u.data.ndim} "
            f"shape={state.u.data.shape}. Cubed-sphere / MPAS / "
            f"lat-lon states need area-weighted variants not "
            f"provided by this module."
        )


def remove_horizontal_mean_wind(state):
    """Subtract per-level horizontal means of ``u`` and ``v``.

    PLANE ONLY (vertical-last ``(ny, nx, nlev)``). Validates layout
    + raises on non-plane state classes. See module docstring for
    the scope rationale.

    Parameters
    ----------
    state : PlaneNonHydrostaticState
        Must expose ``u`` and ``v`` Fields whose ``.data`` has shape
        ``(ny, nx, nlev)``.

    Returns
    -------
    PlaneNonHydrostaticState
        Same state with ``u``, ``v`` shifted by their per-level
        horizontal means; every other field untouched.
    """
    _validate_plane_state(state, "remove_horizontal_mean_wind")
    u = state.u.data
    v = state.v.data
    # Mean over (ny, nx) axes for each vertical level (axis -1).
    # On a uniform plane, the unweighted arithmetic mean equals
    # the proper area-weighted mean (uniform area = dx·dy).
    u_mean = jnp.mean(u, axis=(0, 1), keepdims=True)
    v_mean = jnp.mean(v, axis=(0, 1), keepdims=True)
    u_new = u - u_mean
    v_new = v - v_mean
    return state._replace(
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )


def compute_horizontal_mean_wind(state) -> tuple[jax.Array, jax.Array]:
    """Diagnostic: per-level unweighted horizontal means of ``u``,
    ``v``.

    PLANE ONLY (same layout contract as
    :func:`remove_horizontal_mean_wind`; raises on non-plane
    states). The returned values are arithmetic means — equal to
    the area-weighted means on the uniform plane.
    """
    _validate_plane_state(state, "compute_horizontal_mean_wind")
    u_mean = jnp.mean(state.u.data, axis=(0, 1))   # (nlev,)
    v_mean = jnp.mean(state.v.data, axis=(0, 1))   # (nlev,)
    return u_mean, v_mean
