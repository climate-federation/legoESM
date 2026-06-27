"""Observation containers and operators for 4D-Var data assimilation.

All observation operators H(x) are differentiable (jax.grad-compatible).
Interpolation weights are precomputed at init time; only weighted sums
are inside the AD tape.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

from legoesm.core.field import Field


# ---------------------------------------------------------------------------
# Observation container
# ---------------------------------------------------------------------------

class Observation(NamedTuple):
    """A batch of observations at one time."""
    values: jax.Array       # shape (n_obs,)
    errors: jax.Array       # observation error std devs, shape (n_obs,)
    time_index: int         # which time step in the assimilation window
    operator: object        # ObsOperator (callable state -> jax.Array)
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# Observation error covariance (R matrix)
# ---------------------------------------------------------------------------

class DiagonalR(NamedTuple):
    """Diagonal observation error covariance.

    ``inv_multiply`` divides by ``sigma**2``; a zero observation error std would
    make that ``inf``/``NaN`` (and a non-finite gradient).  ``DiagonalR`` is a
    pytree ``NamedTuple`` (``typing.NamedTuple`` forbids a custom ``__new__``),
    so the divide is floored at the smallest positive normal of ``sigma``'s
    dtype — a no-op for any physical observation error, but it keeps a
    degenerate zero std finite instead of poisoning the cost-gradient.  Validate
    ``sigma > 0`` up front with :meth:`validate` when you control construction.
    """
    sigma: jax.Array  # shape (n_obs,), strictly positive

    def validate(self) -> "DiagonalR":
        """Raise if any concrete ``sigma`` entry is non-positive / non-finite.

        Returns ``self`` for chaining.  Skipped (no-op) when ``sigma`` is a
        traced abstract value, since a Python-truthy check is impossible there.
        """
        sigma = jnp.asarray(self.sigma)
        if not isinstance(sigma, jax.core.Tracer):
            if not bool(jnp.all(jnp.isfinite(sigma) & (sigma > 0))):
                raise ValueError(
                    "DiagonalR.sigma must be strictly positive and finite "
                    "(observation error std); a zero/negative entry makes "
                    "R^{-1} = 1/sigma**2 non-finite."
                )
        return self

    def inv_multiply(self, d: jax.Array) -> jax.Array:
        """R^{-1} @ d = d / sigma^2 (floored against a zero std).

        Uses a double-``where`` so a zero-std entry yields a finite VALUE *and* a
        finite GRADIENT: the differentiated branch divides by a safe ``1`` where
        sigma**2 underflows the floor (otherwise the VJP of ``1/sigma**2`` is
        ``-1/sigma**4`` and ``floor**2`` underflows to 0, giving ``-inf*0=NaN``).
        """
        sigma2 = self.sigma ** 2
        floor = jnp.asarray(jnp.finfo(sigma2.dtype).tiny, dtype=sigma2.dtype)
        ok = sigma2 > floor
        safe = jnp.where(ok, sigma2, jnp.ones_like(sigma2))
        return jnp.where(ok, d / safe, d / floor)


# ---------------------------------------------------------------------------
# Observation operator protocol
# ---------------------------------------------------------------------------

class ObsOperator(Protocol):
    """Observation operator: state -> observation space."""
    def __call__(self, state) -> jax.Array: ...


# ---------------------------------------------------------------------------
# Direct observation operator
# ---------------------------------------------------------------------------

class DirectObsOperator:
    """Direct observation of a state field at grid points.

    H(x) = x[field_name].data[indices]

    Parameters
    ----------
    field_name : str
        Name of the state field to observe.
    indices : tuple of jax.Array
        Multi-index into field.data.
    grid : GridProtocol, optional
        Grid (unused, for interface consistency).
    """

    def __init__(self, field_name: str, indices: tuple,
                 grid=None):
        self.field_name = field_name
        self.indices = indices

    def __call__(self, state) -> jax.Array:
        val = getattr(state, self.field_name)
        arr = val.data if isinstance(val, Field) else val
        return arr[self.indices]


# ---------------------------------------------------------------------------
# Interpolating observation operator
# ---------------------------------------------------------------------------

class InterpolatingObsOperator:
    """Observation at arbitrary (lat, lon) via nearest-neighbor lookup.

    The nearest grid column is found by great-circle distance at init
    time; only the index gather is inside the AD tape.

    Parameters
    ----------
    field_name : str
        State field to observe.
    obs_lat : jax.Array
        Observation latitudes [radians], shape (n_obs,).
    obs_lon : jax.Array
        Observation longitudes [radians], shape (n_obs,).
    obs_level : jax.Array or None
        Level indices (integer), shape (n_obs,). None for 2D fields.
    grid : GridProtocol
        Model grid.
    vertical_coord : optional
        Vertical coordinate (unused for level-index interpolation).
    """

    def __init__(self, field_name: str, obs_lat: jax.Array,
                 obs_lon: jax.Array, obs_level=None,
                 grid=None, vertical_coord=None):
        self.field_name = field_name
        self.obs_level = obs_level
        self.n_obs = obs_lat.shape[0]

        # Precompute nearest grid-point indices using great-circle distance
        grid_lat = grid.to_columns(grid.grid_lat)  # (ncol,)
        grid_lon = grid.to_columns(grid.grid_lon)  # (ncol,)

        # Compute distances from each obs to all grid points
        # Use vectorized great-circle distance
        # For memory efficiency, find nearest neighbor per obs
        self.grid = grid

        # Find nearest grid column for each obs point
        # Great-circle distance: cos(d) = sin(lat1)sin(lat2) + cos(lat1)cos(lat2)cos(dlon)
        sin_olat = jnp.sin(obs_lat)
        cos_olat = jnp.cos(obs_lat)
        sin_glat = jnp.sin(grid_lat)
        cos_glat = jnp.cos(grid_lat)

        # Compute in batches to avoid memory blowup
        # cos_dist[i,j] = sin(olat_i)*sin(glat_j) + cos(olat_i)*cos(glat_j)*cos(olon_i - glon_j)
        cos_dist = (sin_olat[:, None] * sin_glat[None, :]
                    + cos_olat[:, None] * cos_glat[None, :]
                    * jnp.cos(obs_lon[:, None] - grid_lon[None, :]))
        cos_dist = jnp.clip(cos_dist, -1.0, 1.0)

        # Nearest neighbor index
        self.nearest_idx = jnp.argmax(cos_dist, axis=1)  # (n_obs,)

    def __call__(self, state) -> jax.Array:
        val = getattr(state, self.field_name)
        arr = val.data if isinstance(val, Field) else val
        # Flatten to columns
        flat = self.grid.to_columns(arr)

        if self.obs_level is not None and flat.ndim > 1:
            # 3D field: index by (column, level)
            return flat[self.nearest_idx, self.obs_level]
        else:
            # 2D field
            if flat.ndim > 1:
                flat = flat[:, 0] if flat.shape[1] == 1 else flat
            return flat[self.nearest_idx]


# ---------------------------------------------------------------------------
# Column integral observation operator
# ---------------------------------------------------------------------------

class ColumnIntegralObsOperator:
    """Observe a column-integrated quantity (equal-weight vertical sum).

    H(x) = sum over levels of x[field].data at selected columns.

    Currently uses equal weights per level (simple sum). For
    mass-weighted integration (dp/g weighting), pass a vertical_coord
    that provides layer thicknesses (not yet implemented).

    Parameters
    ----------
    field_name : str
        State field to integrate.
    grid : GridProtocol
        Model grid.
    vertical_coord : object, optional
        Reserved for future mass-weighted integration.
    obs_columns : jax.Array
        Column indices to observe, shape (n_obs,).
    """

    def __init__(self, field_name: str, grid=None,
                 vertical_coord=None, obs_columns: jax.Array = None):
        self.field_name = field_name
        self.grid = grid
        self.obs_columns = obs_columns
        self.vertical_coord = vertical_coord

    def __call__(self, state) -> jax.Array:
        val = getattr(state, self.field_name)
        arr = val.data if isinstance(val, Field) else val
        flat = self.grid.to_columns(arr)  # (ncol, nlev)

        # Simple vertical sum (equal weight per level)
        col_integral = jnp.sum(flat[self.obs_columns], axis=-1)
        return col_integral


# ---------------------------------------------------------------------------
# Composite observation operator
# ---------------------------------------------------------------------------

class CompositeObsOperator:
    """Combine multiple operators for multi-variable observations.

    Parameters
    ----------
    operators : tuple of ObsOperator
        Sub-operators.
    """

    def __init__(self, operators: tuple):
        self.operators = operators

    def __call__(self, state) -> jax.Array:
        parts = [op(state) for op in self.operators]
        return jnp.concatenate(parts)


# ---------------------------------------------------------------------------
# Synthetic observation generation
# ---------------------------------------------------------------------------

def generate_synthetic_obs(
    truth_trajectory,
    operators: tuple,
    time_indices: tuple[int, ...],
    error_stds: tuple[float, ...],
    key: jax.Array,
) -> tuple[Observation, ...]:
    """Generate synthetic observations from a truth run.

    y = H(x_true) + epsilon, epsilon ~ N(0, R)

    Parameters
    ----------
    truth_trajectory : pytree with leading time axis
        Stacked trajectory from jax.lax.scan (each leaf has shape (n_steps, ...)).
    operators : tuple of ObsOperator
        One operator per observation batch.
    time_indices : tuple of int
        Time step index for each observation batch.
    error_stds : tuple of float
        Observation error standard deviation for each batch.
    key : jax.Array
        PRNG key.

    Returns
    -------
    tuple of Observation
    """
    observations = []
    for i, (op, tidx, sigma) in enumerate(zip(operators, time_indices, error_stds)):
        key, subkey = jax.random.split(key)
        # Extract state at time tidx from trajectory
        state_t = jax.tree.map(lambda arr: arr[tidx], truth_trajectory)
        h_x = op(state_t)
        noise = jax.random.normal(subkey, shape=h_x.shape) * sigma
        obs = Observation(
            values=h_x + noise,
            errors=jnp.full_like(h_x, sigma),
            time_index=tidx,
            operator=op,
        )
        observations.append(obs)
    return tuple(observations)
