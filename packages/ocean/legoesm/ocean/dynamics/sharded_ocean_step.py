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
import numpy as np
from jax.sharding import PartitionSpec as P

try:                                   # JAX >= 0.8 exposes shard_map at top level
    from jax import shard_map
except ImportError:                    # pragma: no cover - JAX < 0.8 fallback
    from jax.experimental.shard_map import shard_map

from legoesm.parallel.latlon_spmd import activate_latlon_spmd_halo

# Per-device band grids: the SPMD wrapper REUSES the tested band slicers from
# ``legoesm.parallel.latlon_mpi`` rather than a bespoke slice — they already
# handle the staggered v-row (n_lat+1), the bipolar-fold localization (the
# northernmost band keeps the active ``fold``; interior bands get
# ``fold_j=cap_j=-1`` so the pad_ns_* operators fall through to the band halo
# instead of folding their own last row), AND keep ``total_area`` GLOBAL (the
# area-weighted-mean denominator must not become band-local).  Build a band
# layout with ``make_latlon_band_layout(rank, n_dev, n_lat, n_lon, fold)`` then
# call ``slice_cgrid_geometry_to_band(geom, layout)`` (curvilinear tripole /
# eORCA025) or ``slice_latlon_grid_to_band(grid, layout)`` (regular lat-lon ½°).
# latlon_mpi defers its ``from mpi4py import MPI`` to function scope, so these
# pure slicers import without requiring mpi4py.  See omip-multinode-spmd-scope.


def _lat_spec(x):
    """Lat-band PartitionSpec for an array leaf: shard axis 0, replicate rest."""
    nd = int(getattr(x, "ndim", np.ndim(x)))
    if nd < 1:
        return P()                     # scalar -> replicated
    return P("lat", *((None,) * (nd - 1)))


def build_band_grids(grid, n_devices: int):
    """Build the ``n_devices`` UNIFORM lat-band geometries for the SPMD step,
    reusing the tested MPI band slicers (no bespoke re-derivation).

    Requires ``grid.n_lat % n_devices == 0`` so every band has the same leading
    shape — a ``shard_map`` body is ONE program, so the per-band grids must be
    structurally identical (only the values + the north band's active fold
    differ). ``make_latlon_band_layout`` would otherwise hand the first
    ``n_lat % n_devices`` ranks one extra row.

    Returns a list of ``n_devices`` band-local ``LatLonCGridGeometry`` (rank 0 =
    south band ... rank ``n_devices-1`` = north band). The slicer keeps
    ``total_area`` GLOBAL on every band (area-weighted-mean denominator), slices
    v/q rows ``[s:e+1]`` (the shared staggered boundary face), and localizes the
    bipolar fold — active only on the north band, ``fold_j=cap_j=-1`` sentinel on
    interior bands so ``fold_is_local()`` is False there and the pad_ns_*
    operators fall through to the SPMD band halo. See omip-multinode-spmd-scope.

    ``latlon_mpi`` defers ``from mpi4py import MPI`` to function scope, so these
    pure slicers import without requiring mpi4py.
    """
    from legoesm.parallel.latlon_mpi import (
        make_latlon_band_layout,
        slice_cgrid_geometry_to_band,
    )
    n_lat = int(grid.n_lat)
    if n_devices < 1:
        raise ValueError(f"n_devices must be >= 1, got {n_devices}")
    if n_lat % n_devices != 0:
        raise ValueError(
            f"SPMD lat-band requires n_lat ({n_lat}) divisible by n_devices "
            f"({n_devices}) so every band is uniform (one shard_map program). "
            f"Pick n_devices among the divisors of {n_lat}.")
    fold = getattr(grid, "fold", None)
    return [
        slice_cgrid_geometry_to_band(
            grid,
            make_latlon_band_layout(r, n_devices, n_lat, int(grid.n_lon), fold))
        for r in range(n_devices)
    ]


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
