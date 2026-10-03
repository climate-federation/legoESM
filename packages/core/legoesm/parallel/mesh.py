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
   for level-tiled SH transforms (memory scaffolding; spectral scales as
   single-device — see ``create_cubed_sphere_level_mesh`` docstring).

For multi-node MPI, see :mod:`legoesm.parallel.distributed`.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

logger = logging.getLogger(__name__)

# Number of cubed-sphere faces.
N_FACES = 6


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
        Backend name (``"cpu"``, ``"gpu"``, ``"tpu"``, ``"mps"``).
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

def _largest_valid_cubed_sphere_count(n: int) -> int:
    """Return the largest valid cubed-sphere device count that is <= n.

    Valid counts: 1, 2, 3, 6 (face-level), then 6*k^2 for k=2,3,…
    """
    candidates = [c for c in (1, 2, 3) if c <= n]
    k = 1
    while True:
        c = 6 * k * k
        if c > n:
            break
        candidates.append(c)
        k += 1
    return max(candidates) if candidates else 1


def create_device_mesh(
    n_devices: int | str = "auto",
    backend: str | None = None,
    devices: Sequence | None = None,
    allow_level_fallback: bool = False,
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
        n_dev = _largest_valid_cubed_sphere_count(all_count)
        if n_dev != all_count:
            # Issue #273: when n_devices fails face-sharding divisibility
            # (e.g. 4 on a 4×A100 node), opt-in level-parallel fallback
            # keeps every device busy via column-wise sharding instead of
            # clamping to the nearest face-compatible count.
            if allow_level_fallback:
                logger.info(
                    "create_device_mesh: %d devices fail face-sharding "
                    "divisibility; allow_level_fallback=True → routing to "
                    "create_cubed_sphere_level_mesh.",
                    all_count,
                )
                return create_cubed_sphere_level_mesh(
                    n_devices=all_count, backend=backend, devices=devices,
                )
            logger.warning(
                "create_device_mesh: %d device(s) available but %d is not a "
                "valid cubed-sphere count (must divide 6 for ≤6 devices, or be "
                "6·k² for >6 devices).  Using %d device(s).  Pass n_devices "
                "explicitly to suppress this warning, or "
                "allow_level_fallback=True to keep all %d devices busy on a "
                "level-parallel mesh (issue #273).",
                all_count, all_count, n_dev, all_count,
            )
    else:
        n_dev = int(n_devices)
        if allow_level_fallback:
            try:
                _best_tile_factorization(n_dev)
            except ValueError:
                logger.info(
                    "create_device_mesh: n_devices=%d fails face-sharding "
                    "divisibility; allow_level_fallback=True → routing to "
                    "create_cubed_sphere_level_mesh.",
                    n_dev,
                )
                return create_cubed_sphere_level_mesh(
                    n_devices=n_dev, backend=backend, devices=devices,
                )

    if n_dev < 1:
        raise ValueError(f"n_devices must be >= 1, got {n_dev!r}.")

    if n_dev > all_count:
        logger.warning(
            "create_device_mesh: requested %d devices but only %d available "
            "locally (clamping).  Under MPI, pass per-rank device count, "
            "not the total across all ranks.",
            n_dev, all_count,
        )
        n_dev = all_count

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
            n_dev, backend_name, N_FACES // n_dev,
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


def _create_1d_mesh(
    n_devices: int | str,
    backend: str | None,
    devices: Sequence | None,
    *,
    axis: str,
    grid_type: str,
    log_fmt: str,
) -> DeviceConfig:
    """Shared body of the 1-D meshes (level / lat): one named ``axis``.

    ``face_sharding`` carries the ``P(axis)`` spec (field name reused so
    call sites that read it as "the active partition" work unchanged);
    ``grid_type`` tags the path.  Sets the module-global active config.
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
            grid_type=grid_type,
        )
        _active_config = config
        return config

    selected = devices[:n_dev]
    mesh = Mesh(selected, axis_names=(axis,))
    axis_sharding = NamedSharding(mesh, P(axis))
    replicated_sharding = NamedSharding(mesh, P())

    config = DeviceConfig(
        mesh=mesh,
        face_sharding=axis_sharding,
        replicated_sharding=replicated_sharding,
        n_devices=n_dev,
        backend=backend_name,
        is_distributed=False,
        tiling=(1, 1),
        grid_type=grid_type,
    )
    _active_config = config
    logger.info(log_fmt, n_dev, backend_name)
    return config


# ==============================================================================
# Cubed-sphere level-parallel fallback (issue #273, 4-GPU unblock)
# ==============================================================================

def create_cubed_sphere_level_mesh(
    n_devices: int | str = "auto",
    backend: str | None = None,
    devices: Sequence | None = None,
) -> DeviceConfig:
    """Create a level-parallel mesh for cubed-sphere on device counts
    that fail face-sharding divisibility (e.g. 4, 5, 7, 9).

    Replicates the horizontal cubed-sphere stencil across all devices
    and shards the vertical level axis.  Column-wise physics (radiation,
    convection, turbulence) parallelizes naturally; the dycore runs
    replicated on each device (no horizontal halo exchange needed
    across devices since each holds the full ``(6, n, n)`` field).

    Use case
    --------
    Issue #273: 4×A100 GPU node where 4 does not divide 6 and is not
    ``6·k²``.  Face-only sharding would clamp to 3 GPUs (33 % of the
    node idle).  Level fallback keeps all 4 GPUs busy on the
    radiation-dominated workload that drives the throughput gap.

    Status
    ------
    *Scaffolding only*.  The mesh is constructed and returned with
    ``grid_type='cubed_sphere_level'`` so callers can opt in by
    inspecting the tag.  Full driver integration — replicated-dycore
    code path, level-axis-aware radiation sharding, vertical halo
    for level-coupled operators (vertical advection, hydrostatic
    integration) — is downstream operator work flagged in the
    issue #273 follow-up plan.

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
        ``grid_type='cubed_sphere_level'`` to distinguish from the
        face-sharded ``cubed_sphere`` path.
    """
    return _create_1d_mesh(
        n_devices, backend, devices, axis="level",
        grid_type="cubed_sphere_level",
        log_fmt=(
            "legoESM: %d-device level-parallel cubed-sphere mesh on %s "
            "(face-sharding divisibility failed; dycore runs replicated, "
            "physics columns shard over level axis)"
        ),
    )


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
    return _create_1d_mesh(
        n_devices, backend, devices, axis="lat", grid_type="latlon",
        log_fmt="legoESM: %d-device lat-lon mesh on %s (lat-parallel)",
    )


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
    is independent in principle, but this path is NOT wired for real
    distributed throughput today: the SH transforms (``sh_analysis_3d`` /
    ``sh_synthesis_3d``) batch/chunk the level axis for MEMORY only and emit
    no per-shard collectives, ``DeviceConfig.is_distributed`` is hardcoded
    ``False`` here, and the semi-implicit spectral solve couples vertical
    levels (an ``(nlev, nlev)`` matrix per wavenumber) so true level
    sharding would force a level all-gather.  Spectral therefore scales as
    single-device (``_valid_gpu_counts`` returns ``[1]`` in the scaling
    bench) — a genuine cliff, not a missing-flag.  Treat this mesh as
    memory-tiling scaffolding pending a real latitude-decomposed Legendre
    transform.

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
    return _create_1d_mesh(
        n_devices, backend, devices, axis="level", grid_type="spectral",
        log_fmt="legoESM: %d-device level-parallel mesh on %s (spectral)",
    )


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

def multiprocess_safe_device_put(leaf, sharding):
    """``jax.device_put`` that is safe under multi-controller SPMD.

    ``jax.device_put(x, sharding)`` with a sharding that spans processes
    ASSERTS the value is bit-identical on every process.  Per-process XLA
    autotuning can legitimately pick different kernels on different nodes,
    producing last-bit differences in host-precomputed inputs (first hit:
    the external-forcing leaves on the 2-node Levante cs_spmd receipt run,
    #693 job 26030677 — values identical to 8 significant digits, assert
    still trips).  Under >1 process, build the global array from each
    process's LOCAL copy via ``jax.make_array_from_callback`` instead —
    each process materializes only its addressable shards, no cross-process
    equality requirement, no communication.

    Single-process (and non-array leaves) delegate to plain
    ``jax.device_put`` — byte-identical behavior to before.
    Already-global (non-fully-addressable) leaves pass through unchanged.
    """
    # NUMPY ARRAYS MUST TAKE THE LOCAL PATH TOO. ``jnp.ndarray`` IS
    # ``jax.Array``, so the original pair named one type, and a NumPy leaf —
    # which is what the mesh builders produce — failed the check and fell
    # straight through to the asserting placement below. That assert gathers
    # the whole field onto every process, so per-process memory grew with the
    # process count and an allocation of tens of gigabytes ended
    # the ten-million-cell ladder at 192 devices. Which buffer exactly, the
    # gathered result or a temporary of the gather, was not established.
    if not isinstance(leaf, (jax.Array, np.ndarray)):
        return jax.device_put(leaf, sharding)
    if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable:
        return leaf  # already a global sharded array; nothing to place
    if jax.process_count() > 1:
        # Refuse a masked array rather than convert it. Placing one directly
        # raises, but converting first would strip the mask and place the fill
        # values as if they were data, so taking the local path here must not
        # quietly turn a rejection into silently wrong numbers.
        if isinstance(leaf, np.ma.MaskedArray):
            raise ValueError(
                "masked arrays cannot be placed across devices: the mask "
                "would be dropped and the fill values placed as data. "
                "Resolve the mask before sharding.")
        host = np.asarray(leaf)
        return jax.make_array_from_callback(
            host.shape, sharding, lambda idx: host[idx])
    return jax.device_put(leaf, sharding)


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
            if leaf.ndim >= 1 and leaf.shape[0] == N_FACES:
                # In tiled cubed-sphere mode, use both tile axes only for
                # true face-plane arrays; lower-rank face-leading fields keep
                # face-only sharding.
                if config.tiling != (1, 1) and leaf.ndim < 3:
                    sharding = tiled_face_only or config.face_sharding
                    return multiprocess_safe_device_put(leaf, sharding)
                if config.tiling != (1, 1) and leaf.ndim >= 3:
                    # STAGGERED face-plane leaves — D-grid winds
                    # (6, n+1, n, ...) / (6, n, n+1, ...) — cannot
                    # shard over the tile axes (IndivisibleError, P4
                    # discovery probe job 8464703).  Phase-1: face-only
                    # sharding (spatial replicated across a face's kt^2
                    # tile devices); the tiled shard_map stage slices
                    # its LOCAL duplicated-shared-row block body-side
                    # (Pace layout) via staggered_face_to_tile_blocks.
                    tx, ty = config.tiling
                    if leaf.shape[1] % tx != 0 or leaf.shape[2] % ty != 0:
                        sharding = tiled_face_only or config.face_sharding
                        return multiprocess_safe_device_put(leaf, sharding)
                return multiprocess_safe_device_put(leaf, config.face_sharding)
            return multiprocess_safe_device_put(leaf, config.replicated_sharding)

        elif config.grid_type == "cubed_sphere_level":
            # Issue #273 follow-up: level-parallel cubed-sphere mesh.
            # The dycore runs fully replicated horizontally; the mesh
            # has only a ``'level'`` axis, so every leaf is placed on
            # the replicated sharding.  Downstream column-wise physics
            # builds its own column mesh on the same device set and
            # shards there (see ``build_physics_pipeline`` for the
            # column-mesh construction).
            return multiprocess_safe_device_put(leaf, config.replicated_sharding)

        elif config.grid_type == "latlon":
            if leaf.ndim >= 2:
                return multiprocess_safe_device_put(leaf, config.face_sharding)
            return multiprocess_safe_device_put(leaf, config.replicated_sharding)

        elif config.grid_type == "spectral":
            # Spectral state arrays have shape (n_sh, nlev).
            # Shard on the level axis (axis 1) for independent SH transforms.
            if leaf.ndim == 2:
                level_axis_sharding = NamedSharding(
                    config.mesh, P(None, "level"),
                )
                return multiprocess_safe_device_put(leaf, level_axis_sharding)
            if leaf.ndim == 1:
                # 1D arrays (e.g., lnps_hat): replicate across devices.
                return multiprocess_safe_device_put(leaf, config.replicated_sharding)
            return multiprocess_safe_device_put(leaf, config.replicated_sharding)

        elif config.grid_type == "voronoi":
            # Voronoi state arrays: shard cell- and edge-centered arrays
            # along axis 0.  Vertex arrays and small arrays are replicated.
            if config.voronoi_dims is not None and leaf.ndim >= 1:
                nCells, nEdges, _nVerts = config.voronoi_dims
                if leaf.shape[0] in (nCells, nEdges):
                    return multiprocess_safe_device_put(leaf, config.face_sharding)
            return multiprocess_safe_device_put(leaf, config.replicated_sharding)

        # Default: replicate
        return multiprocess_safe_device_put(leaf, config.replicated_sharding)

    return jax.tree.map(_shard_leaf, pytree)


# ==============================================================================
# Tiled staggered-leaf layout (P4 phase-1) — Pace-style duplicated rows
# ==============================================================================
# D-grid staggered face arrays (n+1 cells on one horizontal axis) cannot
# shard evenly over a (face, tile_i, tile_j) mesh.  The tiled shard_map
# stage instead works on PER-TILE BLOCKS that DUPLICATE the shared
# staggered row/col between neighbouring tiles (each tile holds nl+1
# entries on the staggered axis), exactly the FV3/Pace rank layout.
# These pure helpers define that layout once — body-side slicing and
# the canonical-owner inverse — so shard/gather and the parity tests
# share a single source of truth.  Canonical ownership: the LOWER tile
# owns the shared boundary entry (tile ti contributes its rows
# 1..nl for ti > 0; tile 0 contributes rows 0..nl).


def classify_face_metric(arr, n: int):
    """Classify a cubed-sphere metric array for tiled (6*kt^2) layout.

    The tiled shard_map tendency stage (P4 phase-1b) needs every metric
    as a PER-TILE block.  How a field becomes one depends on its
    horizontal extents relative to the face resolution ``n``:

    Returns one of
      ("sliceable", None)  both horiz axes in {n, n+1} — host-slice
          per tile with :func:`tiled_face_block` (centered cell /
          staggered corner / edge metrics).
      ("padded", h)        a horiz axis is n+2h (h>=1) — the global
          array holds only the outer FACE ring; an interior tile's
          local ring is NOT a slice, so it must be produced by running
          the tiled halo exchange ONCE at setup on the unpadded metric
          (codex P4 metric design).
      ("table", None)      exchange-helper offset table (6, 4, m) — an
          exchange internal, not a per-tile spatial metric.
      ("scalar", None)     0-d / non-face-leading array.
      ("other", None)      face-leading but unrecognised extent — must
          be classified before it can ride the tiled stage (ratchet).

    Pure / shape-only; no device or mesh needed.
    """
    import numpy as _np

    if not hasattr(arr, "shape") or arr.ndim < 1 or arr.shape[0] != N_FACES:
        return ("scalar", None)
    if arr.ndim < 3:
        # (6, m) face-leading vector — e.g. nothing spatial in 2 dims.
        return ("scalar", None)
    # Exchange-helper offset tables are (6, 4, ...) — the size-4 edge
    # axis is the unambiguous signature (h1 (6,4,n); h2/h3 (6,4,2,n)),
    # distinct from any spatial metric at a tiled scale (n >= 12).
    # Checked BEFORE the horizontal-extent logic: an h1 table is
    # (6, 4, n), whose b == n would otherwise read as a spatial axis.
    if arr.shape[1] == 4:
        return ("table", None)
    a, b = arr.shape[1], arr.shape[2]
    horiz = {n, n + 1}

    def _pad_h(x):
        d = x - n
        return (d // 2) if (d > 0 and d % 2 == 0) else None

    if a in horiz and b in horiz:
        return ("sliceable", None)
    ha, hb = _pad_h(a), _pad_h(b)
    hs = [h for h in (ha if a not in horiz else None,
                      hb if b not in horiz else None) if h]
    if (a in horiz or ha) and (b in horiz or hb) and hs:
        return ("padded", max(hs))
    return ("other", None)


def tiled_face_block(face_arr, ti: int, tj: int, nl: int, kt: int):
    """Slice tile (ti, tj)'s block from ANY single-face metric array,
    inferring per-axis staggering from the shape.

    A cdgrid carries cell (n, n), corner (n+1, n+1), and edge
    (n, n+1)/(n+1, n) face arrays.  Each horizontal axis is either
    CENTERED (size kt*nl) -> nl local cells, or STAGGERED (size
    kt*nl+1) -> nl+1 local cells DUPLICATING the shared boundary entry
    between neighbouring tiles (Pace layout).  The single source of
    truth for slicing every metric the tiled shard_map stage needs.

    face_arr : (A, B, ...) with A, B in {kt*nl, kt*nl+1}.
    Returns the (a, b, ...) tile block (a = nl[+1], b = nl[+1]).
    """
    a, b = face_arr.shape[0], face_arr.shape[1]
    if a == kt * nl:
        i0, i1 = ti * nl, (ti + 1) * nl
    elif a == kt * nl + 1:
        i0, i1 = ti * nl, ti * nl + nl + 1
    else:
        raise ValueError(
            f"axis 0 size {a} is neither kt*nl={kt * nl} (centered) nor "
            f"kt*nl+1={kt * nl + 1} (staggered) for kt={kt}, nl={nl}.")
    if b == kt * nl:
        j0, j1 = tj * nl, (tj + 1) * nl
    elif b == kt * nl + 1:
        j0, j1 = tj * nl, tj * nl + nl + 1
    else:
        raise ValueError(
            f"axis 1 size {b} is neither kt*nl={kt * nl} (centered) nor "
            f"kt*nl+1={kt * nl + 1} (staggered) for kt={kt}, nl={nl}.")
    return face_arr[i0:i1, j0:j1]


def tiled_padded_block(face_arr, ti: int, tj: int, nl: int, kt: int):
    """Slice tile (ti, tj)'s nl+2h padded block from a PRECOMPUTED
    padded face metric (n+2h on each horizontal axis).

    Key fact (codex P4 padded-metric design): a cubed-sphere padded
    angle metric (``cos_angle_padded`` etc.) already encodes the
    RECEIVER-face extended-gnomonic-geometry halo for the whole face
    (``angle = angle_padded[:, 1:-1, 1:-1]``; cubed_sphere.py:360).  A
    tile's local nl+2h ring is therefore a contiguous SUB-WINDOW of
    that global padded array — interior tiles pick up neighbouring
    same-face interior cells, edge tiles pick up the receiver-geometry
    outer ring — so the per-tile padded metric is a plain strided
    slice, NOT a halo exchange (a scalar exchange of the UNpadded
    cos_angle would carry the neighbour basis = wrong at face seams).

    face_arr : (n+2h, n+2h, ...) precomputed padded metric (single
        face).  h is inferred per axis from the extent.
    Returns the (nl+2h, nl+2h, ...) tile block.
    """
    ha = face_arr.shape[0] - kt * nl
    hb = face_arr.shape[1] - kt * nl
    if ha <= 0 or ha % 2 or hb <= 0 or hb % 2:
        raise ValueError(
            f"axes {face_arr.shape[:2]} are not kt*nl+2h padded for "
            f"kt={kt}, nl={nl}.")
    ha, hb = ha // 2, hb // 2
    return face_arr[ti * nl: ti * nl + nl + 2 * ha,
                    tj * nl: tj * nl + nl + 2 * hb]


def staggered_tile_block(face_arr, ti: int, tj: int, nl: int,
                         stag_axis: int):
    """Slice tile (ti, tj)'s duplicated-row staggered block.

    face_arr : (n+1, n, ...) when ``stag_axis == 0``, (n, n+1, ...)
        when ``stag_axis == 1`` (single face, no leading face axis).
    Returns (nl+1, nl, ...) / (nl, nl+1, ...): the shared boundary
    entry appears in BOTH adjacent tiles' blocks.
    """
    if stag_axis == 0:
        return face_arr[ti * nl: ti * nl + nl + 1,
                        tj * nl: (tj + 1) * nl]
    return face_arr[ti * nl: (ti + 1) * nl,
                    tj * nl: tj * nl + nl + 1]


def staggered_blocks_to_face(blocks, kt: int, stag_axis: int):
    """Inverse of per-tile slicing: canonical (n+1, ...) face array.

    blocks : nested list ``blocks[ti][tj]`` of (nl+1, nl, ...) /
        (nl, nl+1, ...) arrays in tile-row-major order.
    Shared entries must agree between neighbouring blocks (the tiled
    exchange maintains this); the LOWER tile's copy is taken
    (canonical owner), so a disagreement is silently resolved — the
    parity tests assert agreement separately.
    """
    import jax.numpy as jnp

    if stag_axis == 0:
        rows = []
        for ti in range(kt):
            row = jnp.concatenate([blocks[ti][tj] for tj in range(kt)],
                                  axis=1)
            rows.append(row if ti == 0 else row[1:])
        return jnp.concatenate(rows, axis=0)
    cols = []
    for tj in range(kt):
        col = jnp.concatenate([blocks[ti][tj] for ti in range(kt)],
                              axis=0)
        cols.append(col if tj == 0 else col[:, 1:])
    return jnp.concatenate(cols, axis=1)


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

    # Multicontroller (2026-08-04, the MPAS s10 162 GB wall): a plain
    # replicated device_put of a NUMPY/host leaf runs jax's whole-array
    # cross-process assert_equal ([n_proc, leaf] on one device) and the
    # replicated logical size is n_dev x leaf — for the s10 global mesh
    # (1.27 GB of arrays) that is 128 x 1.27 = 162.5 GB, matching the
    # gpu_hlo_schedule failure to 0.1 %. Route through the shared
    # assert-free put (owned/full shards per process, no consistency
    # collective) with ONE cheap exact-hash contract gate for the whole
    # pytree — the same PR #1457 pattern as the ocean lane. The
    # single-process path inside the helper is the historical
    # device_put, byte-unchanged.
    import jax as _jax

    if _jax.process_count() > 1:
        from legoesm.parallel.geometry_consistency import (
            addressable_shard_put, assert_pytree_bytes_equal)

        assert_pytree_bytes_equal(pytree, "replicate_pytree")

        def _replicate_leaf(leaf):
            if not isinstance(leaf, (jax.Array, jnp.ndarray)):
                return leaf
            return addressable_shard_put(leaf, config.replicated_sharding)

        return jax.tree.map(_replicate_leaf, pytree)

    def _replicate_leaf(leaf):
        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
            return leaf
        return jax.device_put(leaf, config.replicated_sharding)

    return jax.tree.map(_replicate_leaf, pytree)


# ==============================================================================
# Tiled cdgrid sliceable-metric assembly (P4 phase-1b)
# ==============================================================================

def stack_tiled_sliceable_metrics(cdgrid, kt: int):
    """Stack per-tile blocks of every per-tile-SLICEABLE cdgrid metric
    over the (face, tile_i, tile_j) device order.

    Two metric classes are pure per-tile slices and both are stacked:
      * ``"sliceable"`` (axes in {n, n+1} — cell / corner / edge) via
        :func:`tiled_face_block`;
      * ``"padded"`` (precomputed n+2h angle metrics) via
        :func:`tiled_padded_block` — the global padded array already
        holds the receiver-geometry halo for the whole face, so a
        tile's nl+2h ring is a strided slice, NOT a halo exchange
        (codex P4 padded design).
    The result is ``(6*kt^2, *block)`` per metric where device
    ``d = (f*kt + ti)*kt + tj`` holds its own tile block — the
    cubed-sphere analogue of the voronoi ``stacked_meshes`` pattern the
    tiled ``shard_map`` tendency stage indexes by ``axis_index``.

    Returns ``(stacks, deferred)``: ``deferred`` is now ONLY the
    non-spatial fields — exchange offset ``table`` arrays (consumed by
    the tiled state exchange, not per-tile metrics) and ``scalar``
    fields (``n`` becomes ``nl`` for the local grid).  Pure host-side
    slicing; no device placement.
    """
    import jax.numpy as jnp

    n = cdgrid.base.n
    if n % kt != 0:
        raise ValueError(
            f"face resolution n={n} not divisible by kt={kt}")
    nl = n // kt

    def _walk(obj, prefix=""):
        fields = getattr(obj, "_fields", None)
        if fields is None:
            return
        for fname in fields:
            val = getattr(obj, fname)
            if hasattr(val, "_fields"):
                yield from _walk(val, prefix=f"{prefix}{fname}.")
            elif hasattr(val, "shape"):
                yield (f"{prefix}{fname}", val)

    stacks = {}
    deferred = []
    for name, arr in _walk(cdgrid):
        kind, h = classify_face_metric(arr, n)
        if kind == "sliceable":
            _blk = tiled_face_block
        elif kind == "padded":
            # Padded metrics are ALSO a pure per-tile slice (the global
            # padded array already holds the receiver-geometry halo for
            # the whole face — codex P4 padded design); h2 padded angle
            # metrics ride a strided nl+2h window, no exchange.
            _blk = tiled_padded_block
        else:
            deferred.append((name, kind if h is None else f"{kind}:h{h}"))
            continue
        blocks = []
        for f in range(6):
            for ti in range(kt):
                for tj in range(kt):
                    blocks.append(_blk(arr[f], ti, tj, nl, kt))
        stacks[name] = jnp.stack(blocks, axis=0)
    deferred.sort()
    return stacks, deferred
