"""Build steady SCM/LES large-scale forcing from a flagged GCM column.

Stage 4 of ``docs/COMPARE_REANALYSIS.md`` (gap #3, **assembly** half): turn the
large-scale state extracted from a worst-performing GCM column into a *steady*
(snapshot) :class:`~legoesm.atmosphere.forcing.scm.scm_forcing.SCMForcing` that drives the
standalone LES exactly like that column's environment, so any LES↔GCM
discrepancy attributes to the physics/closure given the same large-scale state.

This module is the **assembler**: given the column-local large-scale quantities
(subsidence or ``ω``, geostrophic wind, horizontal-advective tendencies, surface
forcing, latitude) it returns a validated ``SCMForcing`` whose channels are
constant-in-time callables (a single GCM snapshot ⇒ steady forcing).  The
grid-metric *extraction* of those quantities from a column's neighbourhood
(``ω`` from ``∇·v_h`` via continuity, ``∇θ`` / ``∇q`` advection, geostrophic
wind from ``∇Φ``) is the companion grid-side piece — it reuses the existing
``grids.vertical`` continuity operators and is tracked separately.

Reuse (CLAUDE.md — no re-derived numerics):

* ``ω → w`` via :func:`legoesm.atmosphere.physics._shared.diagnose_grid_w_from_omega`
  (hydrostatic ``w = -ω/(ρg)`` with virtual-temperature density).
* Coriolis ``f_c = 2Ω sinφ`` via
  :func:`legoesm.grids.cubed_sphere.coriolis_parameter_fv3`.
* ``SCMForcing`` schema + :func:`legoesm.atmosphere.forcing.scm.scm_forcing.validate_forcing`.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics._shared import diagnose_grid_w_from_omega
from legoesm.atmosphere.forcing.scm.scm_forcing import (
    ProfileFn,
    ScalarFn,
    SCMForcing,
    validate_forcing,
)


class ColumnLargeScaleState(NamedTuple):
    """Steady large-scale quantities extracted from one flagged GCM column.

    Profiles are ``(nlev,)`` surface-last (matching the model column).  Provide
    **either** ``subsidence_w`` (already ``w = dz/dt`` [m/s], positive up) **or**
    ``omega`` (``ω = dp/dt`` [Pa/s], positive = sinking) — not both.  ``T`` /
    ``p_full`` / ``q_v`` are the column profiles needed for the ``ω → w``
    conversion and are otherwise unused here.  Surface forcing is selected by
    ``prescribe`` (``"none"`` / ``"T_s"`` / ``"fluxes"``).
    """

    lat_rad: float
    T: jax.Array
    p_full: jax.Array
    q_v: jax.Array
    omega: Optional[jax.Array] = None
    subsidence_w: Optional[jax.Array] = None
    u_geo: Optional[jax.Array] = None
    v_geo: Optional[jax.Array] = None
    theta_adv: Optional[jax.Array] = None
    qv_adv: Optional[jax.Array] = None
    prescribe: str = "none"
    T_s: Optional[float] = None
    w_th_s: Optional[float] = None
    w_qv_s: Optional[float] = None


def subsidence_w_from_omega(
    omega: jax.Array,
    T: jax.Array,
    p_full: jax.Array,
    q_v: jax.Array | None = None,
) -> jax.Array:
    """Single-column ``ω → w`` [m/s, positive up], reusing the shared diagnostic.

    Wraps :func:`diagnose_grid_w_from_omega` (which expects ``(ncol, nlev)``)
    for a single ``(nlev,)`` column.
    """
    omega = jnp.asarray(omega)
    # Promote to a common floating dtype so a low-precision input never
    # downcasts the thermodynamic inputs feeding the density conversion.
    dtype_args = [omega, T, p_full, jnp.float32]
    if q_v is not None:
        dtype_args.insert(3, jnp.asarray(q_v))
    dtype = jnp.result_type(*dtype_args)
    omega = omega.astype(dtype)
    T = jnp.asarray(T, dtype=dtype)
    p_full = jnp.asarray(p_full, dtype=dtype)
    nlev = omega.shape[0]
    checks = [("omega", omega), ("T", T), ("p_full", p_full)]
    qv = None
    if q_v is not None:
        qv_arr = jnp.asarray(q_v, dtype=dtype)
        checks.append(("q_v", qv_arr))
        qv = qv_arr[None, :]
    for name, arr in checks:
        if arr.ndim != 1 or arr.shape[0] != nlev:
            raise ValueError(
                f"subsidence_w_from_omega: {name} must be rank-1 length "
                f"{nlev}; got shape {arr.shape}."
            )
    w = diagnose_grid_w_from_omega(omega[None, :], T[None, :], p_full[None, :], qv)
    return w[0]


def coriolis_f_c(lat_rad: float) -> float:
    """Coriolis parameter ``f_c = 2Ω sinφ`` [1/s] (reuses the shared formula)."""
    from legoesm.grids.cubed_sphere import coriolis_parameter_fv3

    return float(coriolis_parameter_fv3(jnp.asarray(lat_rad), units="rad"))


def _steady_profile(profile: jax.Array) -> ProfileFn:
    """Wrap a ``(nlev,)`` profile as a constant-in-time forcing callable."""
    arr = jnp.asarray(profile)
    return lambda _t: arr


def _steady_scalar(value: float) -> ScalarFn:
    """Wrap a scalar as a constant-in-time forcing callable."""
    arr = jnp.asarray(value)
    return lambda _t: arr


def _check_profile(name: str, profile: jax.Array, nlev: int) -> None:
    arr = jnp.asarray(profile)
    if arr.ndim != 1 or arr.shape[0] != nlev:
        raise ValueError(
            f"build_column_scm_forcing: {name} must be a rank-1 profile of "
            f"length nlev={nlev}; got shape {arr.shape}."
        )


def _check_surface_exclusivity(ls: ColumnLargeScaleState) -> None:
    """Reject surface channels inconsistent with ``prescribe`` (extras).

    ``validate_forcing`` checks the *required* channel is present; here we also
    reject *conflicting* extras so a mis-set field fails loudly rather than
    being silently ignored by the surface bypass.  Unknown ``prescribe`` values
    fall through to ``validate_forcing`` (which raises).
    """
    if ls.prescribe == "none":
        extras = [n for n in ("T_s", "w_th_s", "w_qv_s") if getattr(ls, n) is not None]
        if extras:
            raise ValueError(
                f"prescribe='none' but surface field(s) set: {extras}."
            )
    elif ls.prescribe == "T_s":
        extras = [n for n in ("w_th_s", "w_qv_s") if getattr(ls, n) is not None]
        if extras:
            raise ValueError(f"prescribe='T_s' conflicts with flux field(s): {extras}.")
    elif ls.prescribe == "fluxes":
        if ls.T_s is not None:
            raise ValueError("prescribe='fluxes' conflicts with T_s.")


def build_column_scm_forcing(
    ls: ColumnLargeScaleState, *, allow_no_subsidence: bool = False
) -> SCMForcing:
    """Assemble a validated steady :class:`SCMForcing` from a column's state.

    Resolves subsidence (explicit ``subsidence_w`` or converted from ``omega``),
    Coriolis from latitude, wraps each available large-scale profile/scalar as a
    constant-in-time callable, and validates the surface-prescription dispatch
    (raising on an unknown ``prescribe`` or an inconsistent / conflicting surface
    channel).

    Raises if both ``omega`` and ``subsidence_w`` are supplied (ambiguous), and
    — because silently dropping the large-scale subsidence would bias a
    forced-column LES toward the wrong equilibrium — if **neither** is supplied
    unless ``allow_no_subsidence=True`` is passed explicitly.
    """
    if ls.omega is not None and ls.subsidence_w is not None:
        raise ValueError(
            "ColumnLargeScaleState: provide either omega or subsidence_w, "
            "not both (ambiguous subsidence specification)."
        )

    nlev = jnp.asarray(ls.T).shape[0]
    for name in ("u_geo", "v_geo", "theta_adv", "qv_adv", "subsidence_w", "omega"):
        val = getattr(ls, name)
        if val is not None:
            _check_profile(name, val, nlev)
    _check_surface_exclusivity(ls)

    if ls.subsidence_w is not None:
        w = jnp.asarray(ls.subsidence_w)
    elif ls.omega is not None:
        w = subsidence_w_from_omega(ls.omega, ls.T, ls.p_full, ls.q_v)
    elif allow_no_subsidence:
        w = None
    else:
        raise ValueError(
            "ColumnLargeScaleState: neither omega nor subsidence_w supplied; "
            "pass allow_no_subsidence=True to intentionally force the LES with "
            "zero large-scale subsidence."
        )

    forcing = SCMForcing(
        f_c=coriolis_f_c(ls.lat_rad),
        u_geo=None if ls.u_geo is None else _steady_profile(ls.u_geo),
        v_geo=None if ls.v_geo is None else _steady_profile(ls.v_geo),
        subsidence_w=None if w is None else _steady_profile(w),
        theta_adv=None if ls.theta_adv is None else _steady_profile(ls.theta_adv),
        qv_adv=None if ls.qv_adv is None else _steady_profile(ls.qv_adv),
        prescribe=ls.prescribe,
        T_s=None if ls.T_s is None else _steady_scalar(ls.T_s),
        w_th_s=None if ls.w_th_s is None else _steady_scalar(ls.w_th_s),
        w_qv_s=None if ls.w_qv_s is None else _steady_scalar(ls.w_qv_s),
    )
    validate_forcing(forcing)
    return forcing
