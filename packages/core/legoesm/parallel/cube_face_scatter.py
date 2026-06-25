"""Cubed-sphere MPI face-scatter: rank-local grid/metrics + state.

Background
----------
The cubed-sphere MPI path historically kept the FULL ``(6, n, n, ...)`` state
*and* grid metrics on every rank and only wrote owned faces via halo exchange
(`run_levante_gpu_scaling.py`'s "replicated-dynamics" mode).  Multi-rank
wall-clock then measured replicated compute, not real domain decomposition.

This module provides the missing pieces for a TRUE face decomposition where
each rank owns a contiguous subset of the 6 faces (face-only mode, ranks in
{1, 2, 3, 6}):

* :func:`slice_cubed_sphere_grid` / :func:`slice_cubed_sphere_cdgrid` — slice
  the per-face geometric metric arrays to the rank's owned faces, so operators
  that multiply ``state * metric`` broadcast correctly on the local
  ``(n_local, ...)`` leading dim.
* :func:`make_rank_local_cube_model` — swap a model's full grid/cdgrid for the
  sliced versions (the model is a plain mutable class).
* State is scattered with the existing :func:`legoesm.parallel.layout.scatter_pytree`.

Why this is correct (validated by the design probes + the equivalence test)
---------------------------------------------------------------------------
1. **Metrics are only ever applied at interior shape.** Every cube operator
   halo-pads the STATE fields and applies LOCAL-face metrics (indexed at their
   interior positions) to the haloed state.  No operator reads a metric at a
   halo cell, so metrics need slicing, NOT halo exchange.
2. **The scattered face-only halo already works.** ``_pad_halo_mpi_face_only``
   keys off ``data.shape[0]`` and supports ``shape[0] == len(local_face_ids)``
   for scalar / 4D / vector / interp_offsets / AD (custom_vjp sendrecv).
3. **interp_offsets and duogrid stay FULL ``(6, ...)``.** The halo machinery
   indexes them by the GLOBAL face id (``interp_offsets[global_face, edge]``),
   so they must NOT be sliced — see ``_GLOBAL_FACE_INDEXED_FIELDS``.  This is
   the one field-selectivity subtlety that a naive "slice anything with
   ``shape[0]==6``" would get wrong.
4. **Global reductions are MPI-aware.** ``global_integral`` (used by the mass
   fixer / ``anchor_mass_to_initial``) does ``allreduce(SUM)`` when the MPI halo
   backend is active, so summing owned-face metrics across ranks recovers the
   true global integral.  ``zero_mean_ps_tendency`` is NOT supported under
   scatter (its per-stage denominator ``jnp.sum(area)`` is the LOCAL owned-face
   area, not allreduced) — :func:`make_rank_local_cube_model` raises rather than
   silently running the wrong correction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Fields whose leading axis is the 6 faces but which the HALO machinery indexes
# by GLOBAL face id, so they must be kept full (6, ...) even when the state and
# the geometric metrics are scattered to owned faces only.  Slicing these would
# silently corrupt cross-face interpolation/remap at face seams.
_GLOBAL_FACE_INDEXED_FIELDS = frozenset(
    {
        "halo_interp_offsets",
        "halo_interp_offsets_h2",
        "halo_interp_offsets_h3",
        "duogrid",
    }
)

_N_FACES = 6


def _slice_namedtuple_faces(obj, idx, skip_fields):
    """Return ``obj`` with every per-face ``(6, ...)`` array field sliced to the
    owned faces ``idx``, recursing into nested NamedTuple fields.

    A field is sliced iff it is a JAX array with ``ndim >= 3`` and
    ``shape[0] == 6`` and its name is not in ``skip_fields``.  Nested
    NamedTuples (e.g. ``CubedSphereCDGrid.base``) are recursed.  Everything
    else (scalars, ``None``, connectivity tables kept by name) is returned
    unchanged.
    """
    fields = obj._fields  # NamedTuple
    updates = {}
    for name in fields:
        val = getattr(obj, name)
        if name in skip_fields:
            continue  # keep full (global-face-indexed or topology)
        if hasattr(val, "_fields"):
            # Nested NamedTuple (e.g. the base CubedSphereGrid inside a CDGrid).
            updates[name] = _slice_namedtuple_faces(val, idx, skip_fields)
            continue
        if isinstance(val, (jax.Array, jnp.ndarray)) and val.ndim >= 3 and val.shape[0] == _N_FACES:
            updates[name] = val[idx]
    return obj._replace(**updates)


def _owned_face_index(owned_face_ids) -> jax.Array:
    """Concrete index array for the rank's owned global face ids."""
    ids = tuple(int(f) for f in owned_face_ids)
    if not ids:
        raise ValueError("owned_face_ids is empty — a rank must own >=1 face")
    if any(f < 0 or f >= _N_FACES for f in ids):
        raise ValueError(
            f"owned_face_ids {ids} out of range [0, {_N_FACES})"
        )
    return jnp.asarray(ids, dtype=jnp.int32)


def slice_cubed_sphere_grid(grid, owned_face_ids):
    """Slice a :class:`CubedSphereGrid` to the rank's owned faces.

    Geometric metric arrays (lat/lon, area, dx/dy, angles, padded-angle/half-
    width helpers, ...) are sliced on their leading face axis.  The cross-face
    halo interpolation tables (``halo_interp_offsets*``) and ``duogrid`` are
    kept FULL because the halo machinery indexes them by global face id.
    """
    idx = _owned_face_index(owned_face_ids)
    return _slice_namedtuple_faces(grid, idx, _GLOBAL_FACE_INDEXED_FIELDS)


def slice_cubed_sphere_cdgrid(cdgrid, owned_face_ids):
    """Slice a :class:`CubedSphereCDGrid` (C/D-grid metrics + base grid) to the
    rank's owned faces.

    All C/D metric arrays are face-local and sliced; the nested ``base``
    :class:`CubedSphereGrid` is sliced via :func:`slice_cubed_sphere_grid`
    (which keeps the interp/duogrid tables full).
    """
    idx = _owned_face_index(owned_face_ids)
    return _slice_namedtuple_faces(cdgrid, idx, _GLOBAL_FACE_INDEXED_FIELDS)


def make_rank_local_cube_model(model, owned_face_ids):
    """Mutate ``model`` in place so its grid + cdgrid are sliced to the owned
    faces, and return it.

    The cubed-sphere PE/SW models are plain mutable classes that capture
    ``self.grid`` and ``self.cdgrid`` (built once from the full grid).  We slice
    both and clear any cached mass target so it is recomputed from the
    (scattered) initial state via the MPI-aware ``global_integral``.

    Raises
    ------
    NotImplementedError
        If the model config requests ``zero_mean_ps_tendency`` — its per-stage
        zero-mean correction divides by the LOCAL owned-face area (not
        allreduced across ranks), so it is incorrect under face-scatter.  Use
        ``fix_mass`` / ``anchor_mass_to_initial`` (both MPI-aware via
        ``global_integral``) instead.
    """
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "zero_mean_ps_tendency", False):
        raise NotImplementedError(
            "cube face-scatter does not support zero_mean_ps_tendency=True: "
            "the per-stage zero-mean correction's denominator jnp.sum(area) is "
            "the rank-LOCAL owned-face area, not allreduced across ranks, so "
            "the correction would be wrong under face decomposition.  Set "
            "zero_mean_ps_tendency=False and rely on fix_mass / "
            "anchor_mass_to_initial (MPI-aware via global_integral)."
        )
    if hasattr(model, "cdgrid") and model.cdgrid is not None:
        model.cdgrid = slice_cubed_sphere_cdgrid(model.cdgrid, owned_face_ids)
    if hasattr(model, "grid") and model.grid is not None:
        model.grid = slice_cubed_sphere_grid(model.grid, owned_face_ids)
    # Force the anchored-mass target to be recomputed from the scattered IC
    # (its global_integral allreduces owned-face sums to the true global mass).
    if hasattr(model, "_target_mass"):
        model._target_mass = None
    return model
