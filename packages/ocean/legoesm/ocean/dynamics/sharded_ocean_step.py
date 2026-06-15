"""Multi-GPU SPMD step for the lat-lon C-grid ocean (latlon / tripole).

Wraps ``LatLonCGridOceanModel.step`` in ``jax.shard_map`` over a 1-D ``"lat"``
device mesh, so the ocean state is partitioned by latitude band across devices
(eORCA025 ¼° needs ~141 GiB — fits 32 GiB GPUs at N>=5 bands). The dynamics
operators are element-wise / local-stencil and the in-step halo exchange routes
through the SPMD band body once ``activate_latlon_spmd_halo(mesh)`` is armed; the
barotropic PCG reductions already dispatch to ``batch_psum_spmd`` on ``"lat"``
(``barotropic_common._global_dot_batch``). This is the ocean analogue of the
cubed-sphere ``parallel.sharded_dynamics.make_sharded_step``.

Correctness is gated by ``tests/parallel/test_latlon_ocean_spmd_step.py`` (N-step
tol-match vs the single-device step). Pure-dynamics (no host-loop coupling): the
OMIP host post-step updates (ice / SSS-restore / geothermal) are NOT inside this
wrapper — they need separate gather/scatter or jax-porting (see
[[omip-multinode-spmd-scope]]).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

try:                                   # JAX >= 0.8 exposes shard_map at top level
    from jax import shard_map
except ImportError:                    # pragma: no cover - JAX < 0.8 fallback
    from jax.experimental.shard_map import shard_map

from legoesm.parallel.latlon_spmd import activate_latlon_spmd_halo

# --- staggered band-slice classification of LatLonCGridGeometry fields -------
# The C-grid staggers metric arrays by stagger point.  For a latitude band
# ``rows [a:b)`` (b-a cell rows), a cell/u-row array (leading dim n_lat) is
# sliced ``[a:b]`` while a v/q-row array (leading dim n_lat+1) is sliced
# ``[a:b+1]`` so the band keeps its BOUNDING v-faces (the +1 shared face).  This
# is the crux of the staggered lat-band decomposition (memory
# ``omip-multinode-spmd-scope``): v cannot share u's plain NamedSharding because
# its length n_lat+1 is coprime with the device count.
# Two grid classes flow through the ocean model: the simpler ``LatLonGrid``
# (regular lat-lon, e.g. the ½° matrix cell) and the curvilinear
# ``LatLonCGridGeometry`` (tripole / eORCA025 ¼°).  The slicer handles BOTH; the
# sets are the union of their fields, grouped by stagger row.  Shared names
# (cos_lat/sin_lat/lat = cell-row; n_lon/radius/lon/dlon/dlat = keep; n_lat +
# total_area = special) are consistent across both classes.
_BAND_CELL_ROW_FIELDS = frozenset({       # leading dim n_lat  -> [a:b]
    # shared
    "cos_lat", "sin_lat", "lat",
    # LatLonGrid (regular)
    "lat2d", "lon2d", "f", "dx", "dy", "area",
    # LatLonCGridGeometry (curvilinear C-grid)
    "lat_T", "lon_T", "dx_T", "dy_T", "area_T", "dx_u", "dy_u",
    "f_T", "f_u", "cos_alpha_u", "sin_alpha_u",
})
_BAND_V_ROW_FIELDS = frozenset({          # leading dim n_lat+1 -> [a:b+1]
    # LatLonGrid (regular) v-face coords
    "lat_v", "cos_lat_v",
    # LatLonCGridGeometry (curvilinear C-grid)
    "dx_v", "dy_v", "area_q", "f_v", "cos_alpha_v", "sin_alpha_v",
})
_BAND_KEEP_FIELDS = frozenset({           # n_lon-dim / scalar / descriptor
    "n_lon", "radius", "lon", "dlon", "dlat", "fold",
})


def build_local_latlon_band_grid(grid, a: int, b: int):
    """Slice a ``LatLonCGridGeometry`` into the band-local grid for cell rows
    ``[a, b)`` (a device's lat band in the SPMD decomposition).

    Cell/u-row metrics (leading dim ``n_lat``) are sliced ``[a:b]``; v/q-row
    metrics (leading dim ``n_lat+1``) are sliced ``[a:b+1]`` so the band carries
    its two bounding v-faces (the shared boundary face).  ``n_lat`` becomes
    ``b-a`` and ``total_area`` is recomputed over the sliced cells.  Every field
    of the NamedTuple must be classified (cell-row / v-row / keep / special) or
    this RAISES — so a new geometry field forces an explicit decision here
    rather than being silently mis-sliced.

    Parameters
    ----------
    grid : LatLonCGridGeometry
        The GLOBAL grid.
    a, b : int
        Cell-row band bounds (Python ints, ``0 <= a < b <= grid.n_lat``).

    Returns
    -------
    LatLonCGridGeometry
        Band-local grid with ``n_lat == b - a``.

    Notes
    -----
    ``fold`` is kept as-is; for a bipolar (eORCA) north band the fold seam is
    handled by the SPMD halo body, not the metric slice.  This is a pure,
    differentiable index slice (no metric recomputation beyond ``total_area``).
    """
    n_lat = int(grid.n_lat)
    if not (0 <= a < b <= n_lat):
        raise ValueError(
            f"band [{a}:{b}) out of range for n_lat={n_lat}")
    updates = {"n_lat": b - a}
    for name in grid._fields:
        if name in ("n_lat", "total_area"):
            continue                       # handled specially
        val = getattr(grid, name)
        if name in _BAND_CELL_ROW_FIELDS:
            updates[name] = val[a:b]
        elif name in _BAND_V_ROW_FIELDS:
            updates[name] = val[a:b + 1]
        elif name in _BAND_KEEP_FIELDS:
            pass                           # replicated / unaffected by lat band
        else:
            raise KeyError(
                f"LatLonCGridGeometry field {name!r} not classified for band "
                f"slicing — add it to _BAND_CELL_ROW_FIELDS / _BAND_V_ROW_FIELDS"
                f" / _BAND_KEEP_FIELDS in sharded_ocean_step.py")
    # recompute the scalar total over the band's cells (area field name differs:
    # LatLonGrid -> "area", LatLonCGridGeometry -> "area_T")
    if "total_area" in grid._fields:
        _area = updates.get("area_T", updates.get("area"))
        if _area is None:
            raise KeyError("no cell-area field (area / area_T) to recompute "
                           "total_area")
        updates["total_area"] = jnp.sum(_area)
    return grid._replace(**updates)


def _lat_spec(x):
    """Lat-band PartitionSpec for an array leaf: shard axis 0, replicate rest."""
    nd = int(getattr(x, "ndim", np.ndim(x)))
    if nd < 1:
        return P()                     # scalar -> replicated
    return P("lat", *((None,) * (nd - 1)))


def make_sharded_ocean_step(model, mesh):
    """Return ``step(state, dt) -> state`` running ``model.step`` lat-band-SPMD.

    Parameters
    ----------
    model
        ``LatLonCGridOceanModel`` (latlon or tripole geometry).
    mesh
        A 1-D ``jax.sharding.Mesh`` with axis name ``"lat"`` (from
        ``legoesm.parallel.mesh.create_latlon_mesh(...).mesh``).

    Notes
    -----
    The ``in_specs``/``out_specs`` are derived from the state pytree by
    ``jax.tree.map`` (every array leaf -> ``P("lat", None, ...)``; static
    Field metadata is not a leaf). ``dt`` is captured in the closure (not a
    sharded argument). ``check_rep=False`` because the band halo intentionally
    reads neighbour-rank data (replication-unaware).
    """
    if mesh is None:                   # single-device: plain step
        return lambda state, dt: model.step(state, dt)

    activate_latlon_spmd_halo(mesh)

    def sharded_step(state, dt):
        in_spec = jax.tree.map(_lat_spec, state)
        fn = shard_map(
            lambda s: model.step(s, dt),
            mesh=mesh, in_specs=(in_spec,), out_specs=in_spec,
            check_rep=False,
        )
        return fn(state)

    return sharded_step
