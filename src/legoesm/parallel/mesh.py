"""Device mesh creation and array sharding for parallelism.

Supports three sharding strategies:

1. **Face sharding** (cubed-sphere): shard the 6-face dimension across devices.
   Supports 1–6 devices (one or more faces per device).

2. **Sub-face tiling** (cubed-sphere): split each face into a (tx × ty) tile
   grid, enabling up to 6 × tx × ty devices.  Each device owns one tile of
   one face, with halo exchange between tiles on the same face AND between
   faces.  Total devices must be 6 × tx × ty where tx, ty ≥ 1.

3. **Lat-lon domain decomposition**: shard the latitude dimension across
   devices for lat-lon grids.

4. **Level sharding** (spectral): distribute vertical levels across devices
   for embarrassingly parallel SH transforms.

For multi-node MPI, see :mod:`legoesm.parallel.distributed`.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

logger = logging.getLogger(__name__)

# Number of cubed-sphere faces.
_N_FACES = 6


# ==============================================================================
# Device configuration
# ==============================================================================

class DeviceConfig(NamedTuple):
    """Resolved device configuration.

    Attributes
    ----------
    mesh : Mesh or None
        JAX device mesh.  ``None`` for single-device execution.
    face_sharding : NamedSharding or None
        Sharding specification for face-first arrays ``(6, n, n, ...)``.
        For voronoi grids, this is the primary (cell/edge) sharding.
    replicated_sharding : NamedSharding or None
        Sharding that replicates data across all devices.
    n_devices : int
        Number of devices in use.
    backend : str
        Backend name (``"CPU"``, ``"GPU"``, ``"TPU"``, ``"METAL"``).
    is_distributed : bool
        ``True`` if using multi-node MPI.
    tiling : tuple[int, int]
        Sub-face tile grid ``(tx, ty)``.  ``(1, 1)`` means face-only
        sharding (no sub-face tiling).
    grid_type : str
        ``"cubed_sphere"``, ``"latlon"``, ``"spectral"``, or ``"voronoi"``.
    voronoi_dims : tuple[int, int, int] or None
        ``(nCells, nEdges, nVertices)`` for voronoi grids.  ``None`` for
        other grid types.  Used by :func:`shard_pytree` to identify which
        arrays to shard along axis 0.
    """
    mesh: Mesh | None
    face_sharding: NamedSharding | None
    replicated_sharding: NamedSharding | None
    n_devices: int
    backend: str
    is_distributed: bool
    tiling: tuple[int, int] = (1, 1)
    grid_type: str = "cubed_sphere"
    voronoi_dims: tuple[int, int, int] | None = None


# Singleton — set once at startup, queried by the rest of the code.
_active_config: DeviceConfig | None = None


def get_active_config() -> DeviceConfig | None:
    """Return the active device configuration, or ``None``."""
    return _active_config


def set_active_config(config: DeviceConfig | None) -> None:
    """Set the active device configuration singleton."""
    global _active_config
    _active_config = config


# ==============================================================================
# Helper: factor a device count into face × tile decomposition
# ==============================================================================

def _best_tile_factorization(n_devices: int) -> tuple[int, int, int]:
    """Find the exact (n_face_groups, tx, ty) for n_devices.

    We want 6 * tx * ty = n_devices.  Square tiles (tx == ty) are required
    because inter-face boundaries with axis swaps (e.g., face 1 SOUTH →
    face 5 EAST) need matching strip lengths on both sides.  With non-square
    tiles the i-edge and j-edge have different cell counts, causing mismatches
    at swapped boundaries.

    Raises ``ValueError`` if n_devices is not an exact supported count.
    Use :func:`legoesm.parallel.runtime.validate_device_count` to check
    before calling this function.

    Returns (n_face_groups, tx, ty) where total = 6 * tx * ty.
    """
    if n_devices <= 6:
        if n_devices < 1 or 6 % n_devices != 0:
            raise ValueError(
                f"For ≤6 devices, count must divide 6. Got {n_devices}. "
                f"Supported: 1, 2, 3, 6."
            )
        return (n_devices, 1, 1)

    # n_devices > 6: need sub-face tiling
    if n_devices % 6 != 0:
        raise ValueError(
            f"For >6 devices, count must be 6*k² (multiple of 6 with "
            f"square tiles). Got {n_devices}. "
            f"Next valid counts: "
            + ", ".join(str(6 * k * k) for k in range(2, 12) if 6 * k * k >= n_devices)[:5]
        )

    tiles_per_face = n_devices // 6
    t = int(math.isqrt(tiles_per_face))
    if t * t != tiles_per_face:
        raise ValueError(
            f"For >6 devices, tiles_per_face ({tiles_per_face}) must be "
            f"a perfect square. Got {n_devices} = 6 × {tiles_per_face}. "
            f"Nearest valid: {6 * t * t} ({t}×{t} tiles) "
            f"or {6 * (t+1) * (t+1)} ({t+1}×{t+1} tiles)."
        )

    return (6, t, t)


# ==============================================================================
# Mesh creation — cubed-sphere
# ==============================================================================

def create_device_mesh(
    n_devices: int | str = "auto",
    backend: str | None = None,
    devices: Sequence | None = None,
) -> DeviceConfig:
    """Create a JAX device mesh for cubed-sphere parallelism.

    Supports both face-only sharding (1–6 devices) and sub-face tiling
    (>6 devices, in multiples of 6).

    Parameters
    ----------
    n_devices : int or ``"auto"``
        Number of devices to use.  ``"auto"`` uses all available.
        For ≤6 devices, must divide 6.  For >6 devices, must be a
        multiple of 6 (sub-face tiling is used).
    backend : str or None
        JAX backend (``"cpu"``, ``"gpu"``, ``"tpu"``).  ``None`` = auto.
    devices : sequence or None
        Optional explicit device list.

    Returns
    -------
    DeviceConfig
    """
    global _active_config

    # Resolve devices, with fallback for unavailable backends.
    if devices is not None:
        if backend is not None:
            warnings.warn(
                "create_device_mesh received both explicit devices and a backend; "
                "ignoring backend and using provided devices.",
                RuntimeWarning,
                stacklevel=2,
            )
        devices = list(devices)
        if not devices:
            raise ValueError("devices must contain at least one JAX device")
    else:
        if backend is None and jax.process_count() > 1:
            devices = jax.local_devices()
        elif backend is not None:
            try:
                devices = jax.devices(backend)
            except RuntimeError:
                devices = []
            if not devices:
                warnings.warn(
                    f"Requested backend '{backend}' has no devices. "
                    f"Falling back to default backend.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                devices = jax.devices()
        else:
            devices = jax.devices()

    all_count = len(devices)
    first_platform = str(getattr(devices[0], "platform", "")) if devices else ""
    backend_name = (first_platform or jax.default_backend()).upper()

    if all_count == 0:
        raise RuntimeError(
            "No JAX devices found. Check your JAX installation and hardware."
        )

    # Resolve n_devices.
    if n_devices == "auto":
        n_dev = all_count
    else:
        n_dev = int(n_devices)

    if n_dev < 1:
        raise ValueError(f"n_devices must be >= 1, got {n_dev!r}.")

    n_dev = min(n_dev, all_count)

    # Single-device fast path.
    if n_dev == 1:
        config = DeviceConfig(
            mesh=None,
            face_sharding=None,
            replicated_sharding=None,
            n_devices=1,
            backend=backend_name,
            is_distributed=False,
            tiling=(1, 1),
            grid_type="cubed_sphere",
        )
        _active_config = config
        logger.info("legoESM: single-device mode (%s)", backend_name)
        return config

    # Determine face/tile decomposition (no silent round-down).
    n_face_devices, tx, ty = _best_tile_factorization(n_dev)
    selected = devices[:n_dev]

    if tx == 1 and ty == 1:
        # Face-only sharding (original behavior).
        mesh = Mesh(selected, axis_names=("face",))
        face_sharding = NamedSharding(mesh, P("face"))
        replicated_sharding = NamedSharding(mesh, P())
        tiling = (1, 1)
        logger.info(
            "legoESM: %d-device face mesh on %s (faces per device: %d)",
            n_dev, backend_name, _N_FACES // n_dev,
        )
    else:
        # Sub-face tiling: reshape devices into a 3D mesh
        # (face, tile_i, tile_j) so arrays (6, n, n, ...) are sharded
        # over both horizontal directions.
        import numpy as np
        dev_array = np.array(selected).reshape(6, tx, ty)
        mesh = Mesh(dev_array, axis_names=("face", "tile_i", "tile_j"))
        face_sharding = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
        replicated_sharding = NamedSharding(mesh, P())
        tiling = (tx, ty)
        logger.info(
            "legoESM: %d-device sub-face mesh on %s "
            "(6 faces × %d×%d tiles = %d tiles/face)",
            n_dev, backend_name, tx, ty, tx * ty,
        )

    config = DeviceConfig(
        mesh=mesh,
        face_sharding=face_sharding,
        replicated_sharding=replicated_sharding,
        n_devices=n_dev,
        backend=backend_name,
        is_distributed=False,
        tiling=tiling,
        grid_type="cubed_sphere",
    )
    _active_config = config
    return config


# ==============================================================================
# Lat-lon domain decomposition
# ==============================================================================

def create_latlon_mesh(
    n_devices: int | str = "auto",
    backend: str | None = None,
    devices: Sequence | None = None,
) -> DeviceConfig:
    """Create a JAX device mesh for lat-lon grid parallelism.

    Shards the latitude dimension across devices.  Each device gets
    a contiguous band of latitudes.

    Parameters
    ----------
    n_devices : int or ``"auto"``
        Number of devices.  ``"auto"`` uses all available.
    backend : str or None
        JAX backend.  ``None`` = auto.
    devices : sequence or None
        Optional explicit device list.

    Returns
    -------
    DeviceConfig
    """
    global _active_config

    if devices is not None:
        devices = list(devices)
    elif backend is not None:
        try:
            devices = jax.devices(backend)
        except RuntimeError:
            devices = jax.devices()
    else:
        devices = jax.devices()

    all_count = len(devices)
    first_platform = str(getattr(devices[0], "platform", "")) if devices else ""
    backend_name = (first_platform or jax.default_backend()).upper()

    if n_devices == "auto":
        n_dev = all_count
    else:
        n_dev = min(int(n_devices), all_count)

    if n_dev <= 1:
        config = DeviceConfig(
            mesh=None,
            face_sharding=None,
            replicated_sharding=None,
            n_devices=1,
            backend=backend_name,
            is_distributed=False,
            tiling=(1, 1),
            grid_type="latlon",
        )
        _active_config = config
        return config

    selected = devices[:n_dev]
    mesh = Mesh(selected, axis_names=("lat",))
    lat_sharding = NamedSharding(mesh, P("lat"))
    replicated_sharding = NamedSharding(mesh, P())

    config = DeviceConfig(
        mesh=mesh,
        face_sharding=lat_sharding,  # reuse field name for primary sharding
        replicated_sharding=replicated_sharding,
        n_devices=n_dev,
        backend=backend_name,
        is_distributed=False,
        tiling=(1, 1),
        grid_type="latlon",
    )
    _active_config = config
    logger.info(
        "legoESM: %d-device lat-lon mesh on %s (lat-parallel)",
        n_dev, backend_name,
    )
    return config


# ==============================================================================
# Level-parallel mesh (for spectral models)
# ==============================================================================

def create_level_mesh(
    n_devices: int | str = "auto",
    backend: str | None = None,
    devices: Sequence | None = None,
) -> DeviceConfig:
    """Create a JAX device mesh for level-parallel spectral computation.

    Distributes vertical levels across devices.  Each level's SH transform
    is independent, making this embarrassingly parallel.

    Parameters
    ----------
    n_devices : int or ``"auto"``
        Number of devices.
    backend : str or None
        JAX backend.
    devices : sequence or None
        Optional explicit device list.

    Returns
    -------
    DeviceConfig
    """
    global _active_config

    if devices is not None:
        devices = list(devices)
    elif backend is not None:
        try:
            devices = jax.devices(backend)
        except RuntimeError:
            devices = jax.devices()
    else:
        devices = jax.devices()

    all_count = len(devices)
    first_platform = str(getattr(devices[0], "platform", "")) if devices else ""
    backend_name = (first_platform or jax.default_backend()).upper()

    if n_devices == "auto":
        n_dev = all_count
    else:
        n_dev = min(int(n_devices), all_count)

    if n_dev <= 1:
        config = DeviceConfig(
            mesh=None,
            face_sharding=None,
            replicated_sharding=None,
            n_devices=1,
            backend=backend_name,
            is_distributed=False,
            tiling=(1, 1),
            grid_type="spectral",
        )
        _active_config = config
        return config

    selected = devices[:n_dev]
    mesh = Mesh(selected, axis_names=("level",))
    level_sharding = NamedSharding(mesh, P("level"))
    replicated_sharding = NamedSharding(mesh, P())

    config = DeviceConfig(
        mesh=mesh,
        face_sharding=level_sharding,  # reuse field name
        replicated_sharding=replicated_sharding,
        n_devices=n_dev,
        backend=backend_name,
        is_distributed=False,
        tiling=(1, 1),
        grid_type="spectral",
    )
    _active_config = config
    logger.info(
        "legoESM: %d-device level-parallel mesh on %s (spectral)",
        n_dev, backend_name,
    )
    return config


# ==============================================================================
# Voronoi (icosahedral) mesh parallelism
# ==============================================================================

def create_voronoi_device_mesh(
    nCells: int,
    nEdges: int,
    nVertices: int,
    n_devices: int | str = "auto",
    backend: str | None = None,
    devices: Sequence | None = None,
) -> DeviceConfig:
    """Create a JAX device mesh for Voronoi (MPAS) mesh parallelism.

    Shards cell- and edge-centered arrays along axis 0 across devices.
    For best performance, reorder the mesh with
    :func:`legoesm.parallel.voronoi_partition.reorder_voronoi_for_sharding`
    before sharding so that each device gets a spatially contiguous cell
    cluster.

    Parameters
    ----------
    nCells : int
        Number of Voronoi cells in the mesh.
    nEdges : int
        Number of edges in the mesh.
    nVertices : int
        Number of vertices (Delaunay triangle centers) in the mesh.
    n_devices : int or ``"auto"``
        Number of devices.  ``"auto"`` uses all available.
    backend : str or None
        JAX backend.  ``None`` = auto.
    devices : sequence or None
        Optional explicit device list.

    Returns
    -------
    DeviceConfig
        Config with ``grid_type="voronoi"`` and ``voronoi_dims`` set.
    """
    global _active_config

    if devices is not None:
        devices = list(devices)
    elif backend is not None:
        try:
            devices = jax.devices(backend)
        except RuntimeError:
            devices = jax.devices()
    else:
        devices = jax.devices()

    all_count = len(devices)
    first_platform = str(getattr(devices[0], "platform", "")) if devices else ""
    backend_name = (first_platform or jax.default_backend()).upper()

    if n_devices == "auto":
        n_dev = all_count
    else:
        n_dev = min(int(n_devices), all_count)

    if n_dev <= 1:
        config = DeviceConfig(
            mesh=None,
            face_sharding=None,
            replicated_sharding=None,
            n_devices=1,
            backend=backend_name,
            is_distributed=False,
            tiling=(1, 1),
            grid_type="voronoi",
            voronoi_dims=(nCells, nEdges, nVertices),
        )
        _active_config = config
        return config

    selected = devices[:n_dev]
    mesh = Mesh(selected, axis_names=("device",))
    primary_sharding = NamedSharding(mesh, P("device"))
    replicated_sharding = NamedSharding(mesh, P())

    config = DeviceConfig(
        mesh=mesh,
        face_sharding=primary_sharding,
        replicated_sharding=replicated_sharding,
        n_devices=n_dev,
        backend=backend_name,
        is_distributed=False,
        tiling=(1, 1),
        grid_type="voronoi",
        voronoi_dims=(nCells, nEdges, nVertices),
    )
    _active_config = config
    logger.info(
        "legoESM: %d-device voronoi mesh on %s "
        "(nCells=%d, nEdges=%d, nVertices=%d)",
        n_dev, backend_name, nCells, nEdges, nVertices,
    )
    return config


# ==============================================================================
# Pytree sharding utilities
# ==============================================================================

def shard_pytree(pytree, config: DeviceConfig):
    """Shard a pytree across devices according to the grid type.

    - **cubed_sphere**: face-first arrays ``(6, n, n, ...)`` sharded on face.
      With sub-face tiling, a 3D mesh ``(face, tile_i, tile_j)`` is used so
      both horizontal dimensions are sharded directly.
    - **latlon**: arrays ``(n_lat, n_lon, ...)`` sharded on lat dimension.
    - **spectral**: arrays ``(nlev, ...)`` sharded on level dimension.

    Parameters
    ----------
    pytree
        Any JAX pytree.
    config : DeviceConfig
        Output of a ``create_*_mesh`` function.

    Returns
    -------
    Sharded pytree with the same structure.
    """
    if config.face_sharding is None:
        return pytree  # single-device: nothing to do

    tiled_face_only = None
    if (
        config.grid_type == "cubed_sphere"
        and config.mesh is not None
        and config.tiling != (1, 1)
    ):
        tiled_face_only = NamedSharding(config.mesh, P("face"))

    def _shard_leaf(leaf):
        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
            return leaf

        if config.grid_type == "cubed_sphere":
            if leaf.ndim >= 1 and leaf.shape[0] == _N_FACES:
                # In tiled cubed-sphere mode, use both tile axes only for
                # true face-plane arrays; lower-rank face-leading fields keep
                # face-only sharding.
                if config.tiling != (1, 1) and leaf.ndim < 3:
                    sharding = tiled_face_only or config.face_sharding
                    return jax.device_put(leaf, sharding)
                return jax.device_put(leaf, config.face_sharding)
            return jax.device_put(leaf, config.replicated_sharding)

        elif config.grid_type == "latlon":
            if leaf.ndim >= 2:
                return jax.device_put(leaf, config.face_sharding)
            return jax.device_put(leaf, config.replicated_sharding)

        elif config.grid_type == "spectral":
            # Spectral state arrays have shape (n_sh, nlev).
            # Shard on the level axis (axis 1) for independent SH transforms.
            if leaf.ndim == 2:
                level_axis_sharding = NamedSharding(
                    config.mesh, P(None, "level"),
                )
                return jax.device_put(leaf, level_axis_sharding)
            if leaf.ndim == 1:
                # 1D arrays (e.g., lnps_hat): replicate across devices.
                return jax.device_put(leaf, config.replicated_sharding)
            return jax.device_put(leaf, config.replicated_sharding)

        elif config.grid_type == "voronoi":
            # Voronoi state arrays: shard cell- and edge-centered arrays
            # along axis 0.  Vertex arrays and small arrays are replicated.
            if config.voronoi_dims is not None and leaf.ndim >= 1:
                nCells, nEdges, _nVerts = config.voronoi_dims
                if leaf.shape[0] in (nCells, nEdges):
                    return jax.device_put(leaf, config.face_sharding)
            return jax.device_put(leaf, config.replicated_sharding)

        # Default: replicate
        return jax.device_put(leaf, config.replicated_sharding)

    return jax.tree.map(_shard_leaf, pytree)


def shard_latlon(pytree, config: DeviceConfig):
    """Shard a lat-lon pytree by latitude dimension.

    Arrays with shape ``(n_lat, ...)`` are sharded along axis 0.
    """
    if config.face_sharding is None:
        return pytree
    return shard_pytree(pytree, config)


def shard_levels(pytree, config: DeviceConfig, nlev: int):
    """Shard a pytree by vertical level dimension.

    Arrays whose first axis matches ``nlev`` are sharded across devices.
    Other arrays are replicated.

    Parameters
    ----------
    pytree
        Any JAX pytree.
    config : DeviceConfig
        Level-parallel mesh config.
    nlev : int
        Number of vertical levels (to identify level-first arrays).

    Returns
    -------
    Sharded pytree.
    """
    if config.face_sharding is None:
        return pytree

    def _shard_leaf(leaf):
        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
            return leaf
        if leaf.ndim >= 1 and leaf.shape[0] == nlev:
            return jax.device_put(leaf, config.face_sharding)
        return jax.device_put(leaf, config.replicated_sharding)

    return jax.tree.map(_shard_leaf, pytree)


def replicate_pytree(pytree, config: DeviceConfig):
    """Replicate a pytree across all devices.

    Use this for data that every device needs (e.g., grid metrics).

    Parameters
    ----------
    pytree
        Any JAX pytree.
    config : DeviceConfig
        Output of a ``create_*_mesh`` function.

    Returns
    -------
    Replicated pytree.
    """
    if config.replicated_sharding is None:
        return pytree

    def _replicate_leaf(leaf):
        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
            return leaf
        return jax.device_put(leaf, config.replicated_sharding)

    return jax.tree.map(_replicate_leaf, pytree)
