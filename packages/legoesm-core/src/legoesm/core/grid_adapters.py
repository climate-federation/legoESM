"""Grid-agnostic column adapters for the physics pipeline.

Provides a uniform interface for flattening and unflattening atmospheric
fields between their native grid layout and the ``(ncol, nlev)`` column
format expected by all column-physics kernels.

Three concrete adapters cover the supported grid topologies:

* **GridProtocolAdapter** — wraps any grid satisfying ``GridProtocol``
  (cubed-sphere, lat-lon, Gaussian, Voronoi, …).
* **SingleColumnAdapter** — trivial adapter for SCM / single-column tests
  where the horizontal dimension is already a single column.
* **make_adapter** — factory that inspects the grid and returns the
  appropriate adapter.

Design goals:

1. Column kernels never see grid topology — they always receive
   ``(ncol, nlev)`` or ``(ncol,)`` arrays.
2. The adapter is a plain NamedTuple (JAX-pytree friendly, zero overhead
   in a traced context).
3. No ``jnp.prod`` on dynamic shapes — ``ncol`` is resolved at build
   time from the grid.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Column adapter protocol — any adapter must expose these methods/attrs
# ---------------------------------------------------------------------------

class ColumnAdapter(NamedTuple):
    """Grid-agnostic adapter for column physics.

    Stores pre-resolved metadata so that flatten/unflatten are pure
    reshape operations with no dynamic shape computation.

    Attributes
    ----------
    ncol : int
        Total number of horizontal columns.
    shape_2d : tuple[int, ...]
        Native horizontal shape, e.g. ``(6, 48, 48)`` for cubed-sphere.
    """
    ncol: int
    shape_2d: tuple

    # -- 3-D fields: (..., nlev) <-> (ncol, nlev) --------------------

    def flatten_3d(self, field: jax.Array) -> jax.Array:
        """Reshape ``(*shape_2d, nlev)`` → ``(ncol, nlev)``."""
        nlev = field.shape[-1]
        return field.reshape(self.ncol, nlev)

    def unflatten_3d(self, cols: jax.Array) -> jax.Array:
        """Reshape ``(ncol, nlev)`` → ``(*shape_2d, nlev)``."""
        nlev = cols.shape[-1]
        return cols.reshape(*self.shape_2d, nlev)

    # -- 2-D (surface) fields: (...) <-> (ncol,) --------------------

    def flatten_2d(self, field: jax.Array) -> jax.Array:
        """Reshape ``(*shape_2d,)`` → ``(ncol,)``."""
        return field.reshape(self.ncol)

    def unflatten_2d(self, cols: jax.Array) -> jax.Array:
        """Reshape ``(ncol,)`` → ``(*shape_2d,)``."""
        return cols.reshape(self.shape_2d)


# ---------------------------------------------------------------------------
# Single-column adapter (SCM)
# ---------------------------------------------------------------------------

class SingleColumnGrid(NamedTuple):
    """Minimal grid for single-column model (SCM) experiments.

    Satisfies the subset of ``GridProtocol`` needed by the physics
    pipeline: lat/lon scalars and trivial to_columns / from_columns.
    """
    lat: jax.Array   # scalar or (1,)
    lon: jax.Array   # scalar or (1,)

    @property
    def grid_lat(self):
        return self.lat.reshape(1)

    @property
    def grid_lon(self):
        return self.lon.reshape(1)

    @property
    def grid_n_columns(self) -> int:
        return 1

    def to_columns(self, field):
        """``(1, ...) or (nlev,)`` → ``(1, ...)``."""
        if field.ndim == 1:
            return field[None, :]          # (nlev,) → (1, nlev)
        return field.reshape(1, *field.shape[1:])

    def from_columns(self, cols):
        """``(1, ...)`` → ``(1, ...)``."""
        return cols


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_adapter(grid) -> ColumnAdapter:
    """Build a ColumnAdapter from any supported grid object.

    Parameters
    ----------
    grid
        A grid implementing ``grid_n_columns`` (from ``GridProtocol``),
        or a ``SingleColumnGrid``.

    Returns
    -------
    ColumnAdapter
    """
    ncol = grid.grid_n_columns

    # Infer native 2-D shape from the grid's lat array
    if isinstance(grid, SingleColumnGrid):
        shape_2d = (1,)
    else:
        lat = grid.grid_lat
        shape_2d = tuple(int(s) for s in lat.shape)

    return ColumnAdapter(ncol=ncol, shape_2d=shape_2d)
