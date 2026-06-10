"""Device detection and mesh setup — thin delegation layer.

The actual mesh creation logic lives in ``parallel.mesh`` and
``parallel.device_config``.  This module provides a unified API that
:func:`runtime.bootstrap` can call without the caller needing to know
which sub-module to import.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Re-export HardwareConfig from parallel.device_config for convenience.
from legoesm.parallel.device_config import (        # noqa: F401
    HardwareConfig,
    detect_devices as detect_hardware,
    MixedPrecisionPolicy,
    mixed_precision_policy,
    get_optimal_dtype,
)

# Re-export DeviceConfig and mesh helpers from parallel.mesh.
from legoesm.parallel.mesh import (                  # noqa: F401
    DeviceConfig,
    get_active_config,
    set_active_config,
    create_device_mesh,
    create_latlon_mesh,
    create_level_mesh,
    shard_pytree,
    replicate_pytree,
)


def setup_devices(
    *,
    n_devices: int | str = "auto",
    backend: str | None = None,
    distributed: bool = False,
    grid_type: str = "cubed_sphere",
    allow_level_fallback: bool = False,
    grid_n: int | None = None,
) -> DeviceConfig:
    """One-shot device setup.

    Selects the appropriate mesh constructor based on *grid_type* and
    *distributed* flag.

    Returns a :class:`DeviceConfig` (which is also set as the active
    singleton via ``parallel.mesh.set_active_config``).

    Parameters
    ----------
    allow_level_fallback : bool
        Issue #273 follow-up.  When True and ``grid_type='cubed_sphere'``
        and ``n_devices`` fails face-sharding divisibility (e.g. 4,
        which is not in ``{1, 2, 3, 6, 24, ...}``), the cubed-sphere
        mesh constructor routes to ``create_cubed_sphere_level_mesh``
        instead of clamping down to 3.  Without this flag, a 4-GPU
        cubed-sphere boot raises (when ``n_devices=4`` explicit) or
        silently clamps to 3 (when ``n_devices='auto'``).
    """
    # Late imports so that ``unittest.mock.patch`` on the canonical
    # ``legoesm.parallel.mesh`` module works in tests.
    from legoesm.parallel import mesh as _mesh

    if distributed:
        if grid_type == "latlon":
            # Lat-lon band MPI: build a LatLonBandLayout and activate
            # set_halo_backend("mpi", layout).  See
            # legoesm.parallel.distributed.initialize_distributed_latlon.
            # Returns a *layout* (not a DeviceConfig) — we wrap it
            # in a minimal DeviceConfig for ModelDriver's
            # ``_device_config`` slot.  CPU MPI doesn't need a JAX
            # device mesh — each rank has 1 CPU device and mpi4jax
            # handles inter-rank comm.
            if grid_n is None or grid_n <= 0:
                raise ValueError(
                    "Lat-lon MPI initialisation requires grid_n "
                    "(=n_lat) > 0.  Pass it through "
                    "bootstrap(..., grid_n=config.grid.resolution)."
                )
            from legoesm.parallel.distributed import (
                initialize_distributed_latlon,
            )
            initialize_distributed_latlon(global_n_lat=grid_n)
            # Build a minimal DeviceConfig.  ``is_distributed=True``
            # is what ModelDriver checks; the cubed-sphere fields
            # (face_sharding, replicated_sharding, mesh) stay None
            # — operators consume them only via the cubed-sphere
            # SPMD code paths that lat-lon never reaches.
            import jax
            return DeviceConfig(
                mesh=None,
                face_sharding=None,
                replicated_sharding=None,
                n_devices=1,
                backend=jax.default_backend().upper(),
                is_distributed=True,
                tiling=(1, 1),
                grid_type=grid_type,
            )
        if grid_type == "mpas":
            # MPAS / Voronoi cell-partition MPI.  Unlike the lat-lon band
            # (a layout computable from ``n_lat`` alone), the Voronoi
            # partition needs the *actual global mesh*, which ModelDriver
            # builds in ``_create_grid``.  So the partition + halo-backend
            # activation are deferred to that point
            # (``ModelDriver._create_grid`` calls ``initialize_voronoi_mpi``
            # and sets ``self._voronoi_layout``).  Here we only return a
            # minimal DeviceConfig so the driver takes the MPAS distributed
            # code paths; CPU single-node MPI needs no JAX device mesh
            # (mpi4jax handles inter-rank comm).  Validation of the MPI
            # stack happens when the partition's ``VoronoiHaloExchange`` is
            # constructed.
            import jax
            return DeviceConfig(
                mesh=None,
                face_sharding=None,
                replicated_sharding=None,
                n_devices=1,
                backend=jax.default_backend().upper(),
                is_distributed=True,
                tiling=(1, 1),
                grid_type=grid_type,
            )
        from legoesm.parallel.distributed import initialize_distributed
        return initialize_distributed(grid_type=grid_type)

    if grid_type == "latlon":
        return _mesh.create_latlon_mesh(n_devices=n_devices, backend=backend)
    elif grid_type == "spectral":
        return _mesh.create_level_mesh(n_devices=n_devices, backend=backend)
    else:
        return _mesh.create_device_mesh(
            n_devices=n_devices, backend=backend,
            allow_level_fallback=allow_level_fallback,
        )
