"""Mass-conserving flux-form tracer transport on the cubed sphere (#771).

The advective-form horizontal transport ``-(u·∇q)`` (see
:func:`atmosphere.dynamics.tracer_transport.advective_tracer_tendency`) does
NOT conserve column water ``∫ q·δp·dA`` under divergent flow — it drifts,
systematically in convergence zones, and pairs with a post-step ``max(q, 0)``
clip that is a one-signed moisture source.  This is the #771 day-150 blow-up
wall.

This module transports the tracer MASS ``δp·q`` — and the layer mass ``δp`` —
with the SAME validated FV3 finite-volume operator the cube shallow-water core
uses for mass (:func:`legoesm.core.fv_tp_2d.transport_step`: PPM reconstruction
+ monotone flux limiter + fp64 flux closure + a mass-conserving rescale).  The
mixing ratio is then recovered as ``q★ = (δp·q)★ / δp★``.  Because tracer mass
and layer mass ride the identical operator with the identical winds:

* **Free-stream preserving** — ``q ≡ const`` ⟹ ``(δp·q)★ = const·δp★`` ⟹
  ``q★ = const`` (no spurious source/sink from the transport of a uniform
  tracer through a divergent flow).
* **Mass conserving** — ``∑ area·(δp·q)★ = ∑ area·δp·q`` exactly (the
  ``mass_target`` rescale), so column water is conserved on the transported
  ``δp★`` grid.
* **Monotone / positive-definite** — ``transport_step`` clips negatives before
  the conserving rescale, so dispersive under/overshoots near moisture fronts
  cannot inject a one-signed moisture source.

Consistency with the dynamics' own ``δp`` (which the hybrid PE core carries via
the surface pressure ``p_s``, not via this flux operator) is left to the
caller: ``δp★`` and the dynamics ``δp`` differ only by the flux-form vs
``div_v`` continuity discretisation and both conserve total dry mass; the
caller may adopt ``δp★`` for moisture or reconcile with a mass fixer.  See
``docs`` / issue #771.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.conservation import conservation_accumulator
from legoesm.core.fv_tp_2d import transport_step


def flux_form_tracer_step(
    q: jax.Array,
    delp: jax.Array,
    ut: jax.Array,
    vt: jax.Array,
    dt: float,
    cdgrid,
    *,
    hord: int = 8,
    nord=None,
    damp_c=None,
) -> tuple[jax.Array, jax.Array]:
    """Advance cube tracers one step by mass-conserving flux-form transport.

    Parameters
    ----------
    q : jax.Array, shape ``(6, n, n, nlev, n_tracers)``
        Tracer mixing ratios [kg/kg], packed along a trailing tracer axis.
    delp : jax.Array, shape ``(6, n, n, nlev)``
        Layer pressure thickness ``δp`` [Pa] at the start of the step.
    ut, vt : jax.Array, shape ``(6, n, n, nlev)``
        Contravariant transporting winds — the same winds the mass solver uses,
        e.g. ``d2a2c_vect(u_d, v_d, cdgrid)[4:6]``.
    dt : float
        Transport timestep [s].
    cdgrid : CubedSphereCDGrid
        Cubed-sphere C/D-grid geometry (``cdgrid.base.area`` = cell areas).
    hord : int, optional
        PPM order forwarded to the underlying ``fv_tp_2d`` (8 = monotone).
    nord, damp_c : optional
        del-n damping parameters forwarded to :func:`transport_step`.

    Returns
    -------
    q_new : jax.Array, shape ``(6, n, n, nlev, n_tracers)``
        ``(δp·q)★ / δp★`` — the flux-form transported mixing ratios.
    delp_new : jax.Array, shape ``(6, n, n, nlev)``
        The co-transported layer mass ``δp★`` (the divisor above), returned so
        the caller can reconcile it with the dynamics' ``δp``.
    """
    if q.ndim != 5:
        raise ValueError(
            f"flux_form_tracer_step expects q with shape "
            f"(6, n, n, nlev, n_tracers); got {q.shape}")
    if delp.shape != q.shape[:-1]:
        raise ValueError(
            f"delp shape {delp.shape} must match q[..., 0] shape {q.shape[:-1]}")

    acc = conservation_accumulator()
    area_e = cdgrid.base.area.astype(acc)             # (6, n, n)

    def _mass2d(field2d: jax.Array) -> jax.Array:     # ∑ area·field on a level
        return jnp.sum(field2d.astype(acc) * area_e)

    def _transport_conserving(field2d, ut_k, vt_k):
        return transport_step(
            field2d, ut_k, vt_k, dt, cdgrid,
            mass_target=_mass2d(field2d),
            nord=nord, damp_c=damp_c, hord=hord)

    # --- Co-transport the layer mass δp, one level at a time. ---
    delp_new = jax.vmap(
        _transport_conserving, in_axes=(-1, -1, -1), out_axes=-1)(delp, ut, vt)

    # --- Co-transport the tracer mass δp·q, per (level, tracer). ---
    # δp·q for EVERY tracer rides the SAME per-level winds; vmap the tracer axis
    # inside the level vmap so the winds broadcast correctly and are not tiled.
    qmass = delp[..., None] * q                       # (6, n, n, nlev, ntr)

    def _transport_level_tracers(qm_k, ut_k, vt_k):   # qm_k: (6, n, n, ntr)
        return jax.vmap(
            lambda qm1: _transport_conserving(qm1, ut_k, vt_k),
            in_axes=-1, out_axes=-1)(qm_k)

    qmass_new = jax.vmap(
        _transport_level_tracers, in_axes=(3, -1, -1), out_axes=3)(
            qmass, ut, vt)                            # (6, n, n, nlev, ntr)

    # Recover the mixing ratio; guard the divisor (δp★ is positive by
    # construction — transport_step clips negatives — but a rest column can be
    # ~0 at the model top).
    delp_safe = jnp.maximum(delp_new, jnp.asarray(1e-30, delp_new.dtype))
    q_new = qmass_new / delp_safe[..., None]
    return q_new, delp_new
