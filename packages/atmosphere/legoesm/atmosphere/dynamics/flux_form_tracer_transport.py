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

* **Free-stream preserving** — ``q ≡ const`` ⟹ ``(δp·q)★ ≈ const·δp★`` ⟹
  ``q★ ≈ const`` to FLOATING-POINT ROUNDOFF (not bit-exact): ``transport_step``
  is positively homogeneous — its PPM fluxes are linear and the monotone
  limiter + ``mass_target`` rescale (``scale = mt / max(∑h⁺, 1)``) are
  scale-invariant for ``c > 0`` while ``∑h⁺ > 1`` — but the per-field mass
  targets ``_mass2d(δp)`` and ``_mass2d(δp·c)`` are summed independently, so
  ``c`` factors out only up to summation roundoff.  The truth-test gates this
  at ``< 1e-9`` (fp64), i.e. roundoff, not zero.
* **Mass conserving** — ``∑ area·(δp·q)★ = ∑ area·δp·q`` to the accumulator's
  fp64 precision (the ``mass_target`` rescale), so column water is conserved on
  the transported ``δp★`` grid — see the HARD CONTRACT below.
* **Monotone / positive-definite** — ``transport_step`` clips negatives before
  the conserving rescale, so dispersive under/overshoots near moisture fronts
  cannot inject a one-signed moisture source.

HARD CONTRACT — the returned ``q_new`` conserves ``∫δp·q`` and is free-stream
preserving **only when paired with the returned co-transported ``δp★``**.  A
caller that stores ``q_new`` against a DIFFERENT layer mass — e.g. the hybrid
PE core's own ``δp`` derived from the RK3-updated ``p_s`` (which this flux
operator does NOT transport) — FORFEITS both guarantees: ``∫δp_dyn·q_new`` is
no longer the conserved moisture mass, and a uniform tracer no longer stays
uniform.  The caller MUST either adopt ``δp★`` for the moisture columns or
reconcile ``δp★`` with the dynamics ``δp`` via a mass fixer.  ``δp★`` and the
dynamics ``δp`` differ only by the flux-form vs ``div_v`` continuity
discretisation (both conserve total dry mass), so the reconciliation is a small
correction — but it is NOT optional.  See issue #771.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.conservation import (
    conservation_accumulator,
    global_face_sum_if_scattered,
)
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
        The co-transported layer mass ``δp★`` (the divisor above).

    HARD CONTRACT (see the module docstring): ``q_new`` conserves ``∫δp·q`` and
    preserves a uniform tracer **only when paired with the returned ``δp★``**.
    Storing ``q_new`` against a different layer mass (e.g. the PE core's own
    ``p_s``-derived ``δp``) forfeits BOTH guarantees; the caller MUST adopt
    ``δp★`` for the moisture columns or reconcile it with the dynamics ``δp``
    via a mass fixer.
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

    def _transport_raw(field2d, ut_k, vt_k):
        # Raw flux-form transport (PPM + fp64 flux closure) WITHOUT
        # ``transport_step``'s internal per-level clip + conserving rescale
        # (``mass_target=None``).  The rescale — and its (optional) cross-face
        # allreduce — is hoisted OUT of the level/tracer vmap into
        # ``_conserving_rescale`` below so the collective runs ONCE on a batched
        # vector, never inside vmap (the repo has no precedent for a collective
        # under vmap; this mirrors ``conservation.fix_mass_hydrostatic``).
        return transport_step(
            field2d, ut_k, vt_k, dt, cdgrid,
            mass_target=None, nord=nord, damp_c=damp_c, hord=hord)

    def _conserving_rescale(field_in, field_raw, red_area):
        # Mirror ``_finalize_transport``'s clip + mass-conserving rescale
        # (``h_pos = max(h,0); scale = mass_in / max(mass_pos, 1); h_pos*scale``)
        # but BATCHED over the leading level (and tracer) axes and face-scatter
        # aware.  ``red_area`` = ``area_e`` broadcast to ``field_in``'s trailing
        # axes; reduce the spatial (face, i, j) axes, keeping level[, tracer].
        # BIT-IDENTICAL to the per-level internal rescale on single-rank /
        # replicated (``global_face_sum_if_scattered`` is then identity); under
        # MPI face-scatter it allreduces the owned-face partials to the true
        # global masses so ``scale`` is uniform across ranks (else each rank
        # rescales to its own partial and silently breaks conservation — #811).
        pos = jnp.maximum(field_raw, 0.0)
        mass_in = jnp.sum(field_in.astype(acc) * red_area, axis=(0, 1, 2))
        mass_pos = jnp.sum(pos.astype(acc) * red_area, axis=(0, 1, 2))
        # ONE allreduce for the (numerator, denominator) pair — never divide a
        # pre-reduced ratio (Σ of a ratio ≠ ratio of Σ); clamp the REDUCED
        # denominator, not the local partial.
        mass_in, mass_pos = global_face_sum_if_scattered(
            jnp.stack([mass_in, mass_pos], axis=0), area_e)
        scale = mass_in / jnp.maximum(mass_pos, 1.0)  # fp64, per level[, tracer]
        return pos * scale.astype(pos.dtype)

    # --- Co-transport the layer mass δp, one level at a time. ---
    delp_raw = jax.vmap(
        _transport_raw, in_axes=(-1, -1, -1), out_axes=-1)(delp, ut, vt)
    delp_new = _conserving_rescale(delp, delp_raw, area_e[..., None])

    # --- Co-transport the tracer mass δp·q, per (level, tracer). ---
    # δp·q for EVERY tracer rides the SAME per-level winds; vmap the tracer axis
    # inside the level vmap so the winds broadcast correctly and are not tiled.
    qmass = delp[..., None] * q                       # (6, n, n, nlev, ntr)

    def _transport_level_tracers(qm_k, ut_k, vt_k):   # qm_k: (6, n, n, ntr)
        return jax.vmap(
            lambda qm1: _transport_raw(qm1, ut_k, vt_k),
            in_axes=-1, out_axes=-1)(qm_k)

    qmass_raw = jax.vmap(
        _transport_level_tracers, in_axes=(3, -1, -1), out_axes=3)(
            qmass, ut, vt)                            # (6, n, n, nlev, ntr)
    qmass_new = _conserving_rescale(qmass, qmass_raw, area_e[..., None, None])

    # Recover the mixing ratio.  PRECONDITION: δp is a physical layer thickness
    # [Pa], so δp★ ≫ the 1e-30 floor everywhere in any real atmosphere — the
    # floor is a pure NaN-guard, never an expected path.  It can only bite a
    # degenerate all-zero column; there the co-transported tracer mass is also
    # zero (transport_step conserves it from a zero input), so q_new = 0/1e-30
    # = 0 and the conservation identity ∫δp★·q_new = ∫δp·q holds trivially
    # (both sides 0 in that column).  For any δp★ ≥ 1e-30, delp_safe == δp★ so
    # δp★·q_new == qmass★ exactly and conservation is exact.
    delp_safe = jnp.maximum(delp_new, jnp.asarray(1e-30, delp_new.dtype))
    q_new = qmass_new / delp_safe[..., None]
    return q_new, delp_new
