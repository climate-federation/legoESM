"""Large-scale forcing primitives shared by the SCM and the plane CRM.

SAM imposes large-scale forcing on a CRM/SCM column through three channels
(``forcing.f90`` + ``subsidence.f90``):

1. **Large-scale subsidence** — vertical advection of every prognostic scalar
   (and the horizontal momentum) by a horizontally-uniform large-scale vertical
   velocity ``w_ls(z)``::

       dphi/dt|_subs = - w_ls * d(phi)/dz          (first-order UPWIND in z)

   With ``w_ls`` positive upward, subsiding air (``w_ls < 0``) advects the
   reference stratification downward, warming/drying the column.
2. **Prescribed horizontal-advection tendencies** ``dθ/dt|_ls`` and
   ``dq_v/dt|_ls`` — added directly to the thermodynamic / moisture tendency.
3. **Nudging / geostrophic relaxation** — relax the horizontal wind (and
   optionally θ, q) toward a target profile on a timescale ``τ``.

This module holds the *array-level* primitives (no model-state types) so both
:mod:`legoesm.atmosphere.forcing.scm.scm_forcing` (single column, ``HydrostaticState``) and
the plane CRM (:mod:`legoesm.atmosphere.forcing.plane_large_scale_forcing`,
``PlaneNonHydrostaticState``) build on the SAME upwind operator rather than
re-deriving it.  Both models store columns **top-to-bottom** (index 0 = model
top, index ``nlev-1`` = surface), so ``z_full`` decreases with index.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def upwind_dphi_dz_top2bottom(
    phi: jax.Array,
    z_full: jax.Array,
    w: jax.Array,
) -> jax.Array:
    """First-order upwind ``d(phi)/dz`` for a top-to-bottom vertical layout.

    Index convention: ``k=0`` = model top, ``k=nlev-1`` = surface, so
    ``z_full[..., k-1] > z_full[..., k]`` everywhere.  The returned gradient
    uses the **positive-z-upward** convention (positive when ``phi`` increases
    with height).

    Donor-side selection by the sign of ``w`` (positive upward):
      * ``w > 0`` (rising air): donor is *below*  → ``(phi[k] − phi[k+1]) /
        (z[k] − z[k+1])``.
      * ``w < 0`` (subsiding): donor is *above*  → ``(phi[k-1] − phi[k]) /
        (z[k-1] − z[k])``.
      * ``w == 0`` falls into the "from-above" branch but is multiplied by
        zero downstream, so the choice is moot.

    Endpoints with no upstream donor (top with ``w < 0`` = downward inflow,
    surface with ``w > 0`` = upward inflow) return a duplicated interior
    stencil; the caller masks those tendencies to zero with
    :func:`mask_inflow_endpoint_tendency` rather than fabricating an inflow
    gradient.  ``z`` differences are floored at 1 m to keep AD finite.
    """
    diff = phi[..., :-1] - phi[..., 1:]          # phi[k] - phi[k+1], k=0..nlev-2
    dz = z_full[..., :-1] - z_full[..., 1:]      # z[k] - z[k+1], all > 0
    grad = diff / jnp.clip(dz, 1.0, None)        # shape (..., nlev-1)

    # "From-above" gradient at level k for k=1..nlev-1 (uses diff[k-1]); k=0
    # endpoint falls back to the "from-below" gradient (duplicate grad[0]).
    grad_from_above = jnp.concatenate([grad[..., :1], grad], axis=-1)
    # "From-below" gradient at level k for k=0..nlev-2 (uses diff[k]); k=nlev-1
    # endpoint falls back to the "from-above" gradient (duplicate grad[-1]).
    grad_from_below = jnp.concatenate([grad, grad[..., -1:]], axis=-1)

    return jnp.where(w > 0.0, grad_from_below, grad_from_above)


def mask_inflow_endpoint_tendency(
    tend: jax.Array, w: jax.Array,
) -> jax.Array:
    """Zero the advective tendency at endpoints with no upstream donor.

    The cell above the model top and below the surface lie outside the
    simulated column.  When ``w_ls`` carries air *into* the domain there —
    top with ``w < 0`` (downward inflow) or surface with ``w > 0`` (upward
    inflow) — the upwind stencil has no real donor and
    :func:`upwind_dphi_dz_top2bottom` returned a duplicated interior value.
    Masking those tendencies to zero is the only boundary-condition-free
    option that does not fabricate an inflow profile; a non-trivial top
    boundary should instead set ``w_ls[0] = 0`` (closed) or drive inflow
    through the prescribed advective tendency.

    Operates on the trailing vertical axis for any leading broadcast shape
    (shared ``nlev``).  Requires ``nlev >= 2``.
    """
    nlev = tend.shape[-1]
    idx = jnp.arange(nlev)
    is_top = (idx == 0)
    is_surf = (idx == nlev - 1)
    bcast_shape = (1,) * (tend.ndim - 1) + (nlev,)
    is_top = is_top.reshape(bcast_shape)
    is_surf = is_surf.reshape(bcast_shape)
    tend = jnp.where(is_top & (w < 0.0), jnp.zeros_like(tend), tend)
    tend = jnp.where(is_surf & (w > 0.0), jnp.zeros_like(tend), tend)
    return tend


def zero_vertical_endpoints(tend: jax.Array) -> jax.Array:
    """Zero the tendency at both vertical endpoints (top + surface).

    Operates on the trailing ``nlev`` axis for any leading broadcast shape.
    """
    nlev = tend.shape[-1]
    idx = jnp.arange(nlev)
    interior = (idx > 0) & (idx < nlev - 1)
    return tend * interior.reshape((1,) * (tend.ndim - 1) + (nlev,))


def subsidence_tendency_top2bottom(
    phi: jax.Array,
    z_full: jax.Array,
    w_ls: jax.Array,
) -> jax.Array:
    """Large-scale subsidence tendency ``-w_ls · d(phi)/dz`` (upwind), zeroed
    at both vertical endpoints to match SAM.

    SAM's ``subsidence.f90`` computes the vertical advective tendency only on
    the interior ``k = 2 .. nzm-1`` — the top and surface subsidence tendency
    are left at zero (those boundaries are handled by the rigid ``w=0`` BC and
    the sounding/sponge).  We therefore zero BOTH endpoints unconditionally
    (codex iter-31 C), rather than only the no-donor inflow endpoints, so the
    plane CRM does not inject heat/mass at a boundary where SAM injects none.
    ``phi``, ``z_full`` and ``w_ls`` broadcast on the trailing ``nlev`` axis.
    """
    dphi_dz = upwind_dphi_dz_top2bottom(phi, z_full, w_ls)
    return zero_vertical_endpoints(-w_ls * dphi_dz)
