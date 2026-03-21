"""Canonical unified parallel runtime for legoESM.

This module provides ``ParallelRuntime`` — the single entry point for
all parallelism in legoESM, whether single-device, multi-GPU, MPI,
or hybrid MPI × multi-device.

Execution modes
---------------

+---------+---------+-----------------------------------------------+
| Ranks   | Devices | Description                                   |
+=========+=========+===============================================+
| 1       | 1       | Serial (all sharding/comms are identity ops)  |
+---------+---------+-----------------------------------------------+
| 1       | N       | JAX SPMD — data sharded across local devices  |
+---------+---------+-----------------------------------------------+
| N       | 1/rank  | MPI — one process per device, MPI halos       |
+---------+---------+-----------------------------------------------+
| N       | M/rank  | Hybrid — MPI between ranks, SPMD within rank  |
+---------+---------+-----------------------------------------------+

All other parallel modules (``mesh``, ``distributed``, ``layout``,
``comm``, ``sharded_dynamics``, ``halo_exchange``) are implementation
details consumed by this module.  User code should create a
``ParallelRuntime`` and pass it to the model driver.

Usage
-----
::

    from legoesm.parallel.runtime import ParallelRuntime

    # Auto-detect (single-process: uses all local devices)
    rt = ParallelRuntime.create()

    # Explicit: 6 MPI ranks, 1 GPU each
    rt = ParallelRuntime.create(grid_type="cubed_sphere", grid_n=48)

    # Query
    print(rt.mode)              # "serial" | "multi_device" | "mpi" | "hybrid"
    print(rt.global_n_devices)  # total devices across all ranks
    print(rt.layout)            # rank-local ownership
"""

from __future__ import annotations

import logging
import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


# ======================================================================
# Supported decomposition table
# ======================================================================

_VALID_FACE_ONLY = {1, 2, 3, 6}


def _supported_device_counts(max_n: int = 1024) -> set[int]:
    """Return the set of supported cubed-sphere device counts up to *max_n*.

    Supported counts:
    - 1, 2, 3, 6 (face-only)
    - 6 * k² for k = 2, 3, ... (sub-face square tiling)
    """
    counts = set(_VALID_FACE_ONLY)
    k = 2
    while 6 * k * k <= max_n:
        counts.add(6 * k * k)
        k += 1
    return counts


def validate_device_count(n: int, grid_type: str = "cubed_sphere") -> None:
    """Raise ``ValueError`` if *n* is not a supported device count.

    Unlike the old ``_best_tile_factorization`` which silently rounded
    down, this function fails fast with a precise error message listing
    nearby valid counts.
    """
    if grid_type != "cubed_sphere":
        if n < 1:
            raise ValueError(f"Device count must be >= 1, got {n}")
        return  # lat-lon / spectral accept any count

    if n < 1:
        raise ValueError(f"Device count must be >= 1, got {n}")

    valid = _supported_device_counts(max(n * 2, 128))
    if n in valid:
        return

    # Build a helpful suggestion
    lower = sorted(c for c in valid if c <= n)
    upper = sorted(c for c in valid if c > n)
    suggestions = []
    if lower:
        suggestions.append(f"next lower: {lower[-1]}")
    if upper:
        suggestions.append(f"next higher: {upper[0]}")
    suggestion_str = "; ".join(suggestions) if suggestions else "use 1, 2, 3, or 6"

    raise ValueError(
        f"Unsupported device count {n} for cubed_sphere. "
        f"Supported counts: 1, 2, 3, 6, 24, 54, 96, 150, 216, 294, 384, ... "
        f"(1/2/3/6 for face-only, or 6*k² for sub-face tiling). "
        f"{suggestion_str}."
    )


# ======================================================================
# Halo backend enumeration
# ======================================================================

class HaloBackend:
    """Halo exchange backend identifiers."""
    LOCAL = "local"       # pad_halo from grids.halo (single process)
    JAX_SPMD = "jax"      # implicit via XLA when data is sharded
    MPI = "mpi"           # mpi4jax sendrecv / allgather
    HYBRID = "hybrid"     # MPI between ranks + JAX within rank


class ReductionBackend:
    """Global reduction backend identifiers."""
    LOCAL = "local"       # jnp.sum / jnp.max (single process)
    JAX_PSUM = "jax"      # jax.lax.psum (multi-device, single process)
    MPI = "mpi"           # mpi4jax allreduce
    HYBRID = "hybrid"     # JAX psum within rank, MPI between ranks


# ======================================================================
# ParallelRuntime
# ======================================================================

class ParallelRuntime:
    """Canonical unified parallel execution context.

    Attributes
    ----------
    mode : str
        One of ``"serial"``, ``"multi_device"``, ``"mpi"``, ``"hybrid"``.
    rank : int
        MPI rank (0 for non-MPI).
    world_size : int
        MPI world size (1 for non-MPI).
    local_device_count : int
        Number of JAX devices on this rank.
    global_n_devices : int
        Total devices across all ranks.
    device_config : DeviceConfig
        JAX device mesh configuration for this rank's devices.
    layout : DistributedLayout | SingleRankLayout
        Rank-local data ownership descriptor.
    topology : CommTopology | None
        MPI communication topology (None for non-MPI).
    halo_backend : str
        Active halo exchange backend.
    reduction_backend : str
        Active global reduction backend.
    grid_type : str
        Grid type (``"cubed_sphere"``, ``"latlon"``, ``"voronoi"``).
    """

    def __init__(
        self,
        mode: str,
        rank: int,
        world_size: int,
        local_device_count: int,
        device_config,
        layout,
        topology=None,
        halo_backend: str = HaloBackend.LOCAL,
        reduction_backend: str = ReductionBackend.LOCAL,
        grid_type: str = "cubed_sphere",
    ):
        self.mode = mode
        self.rank = rank
        self.world_size = world_size
        self.local_device_count = local_device_count
        self.global_n_devices = world_size * local_device_count
        self.device_config = device_config
        self.layout = layout
        self.topology = topology
        self.halo_backend = halo_backend
        self.reduction_backend = reduction_backend
        self.grid_type = grid_type

    @classmethod
    def create(
        cls,
        *,
        grid_type: str = "cubed_sphere",
        grid_n: int | None = None,
        n_devices: int | str = "auto",
        backend: str | None = None,
    ) -> ParallelRuntime:
        """Create the canonical parallel runtime.

        Detects the execution environment (MPI ranks, local devices)
        and configures the appropriate mode.

        Parameters
        ----------
        grid_type : str
            ``"cubed_sphere"``, ``"latlon"``, ``"voronoi"``, ``"spectral"``.
        grid_n : int or None
            Grid resolution (needed for MPI layout construction).
        n_devices : int or ``"auto"``
            Devices per rank.  ``"auto"`` uses all local devices.
        backend : str or None
            JAX backend override.

        Returns
        -------
        ParallelRuntime
        """
        # Detect MPI
        is_mpi = False
        rank = 0
        world_size = 1
        try:
            if jax.process_count() > 1:
                is_mpi = True
                rank = jax.process_index()
                world_size = jax.process_count()
        except Exception:
            pass

        # If not detected via JAX, check mpi4py directly
        if not is_mpi:
            try:
                import importlib.util
                if importlib.util.find_spec("mpi4py") is not None:
                    from mpi4py import MPI
                    comm = MPI.COMM_WORLD
                    if comm.Get_size() > 1:
                        is_mpi = True
                        rank = comm.Get_rank()
                        world_size = comm.Get_size()
            except Exception:
                pass

        # Resolve local devices
        local_devices = jax.local_devices() if is_mpi else jax.devices()
        if backend is not None and not is_mpi:
            try:
                local_devices = jax.devices(backend)
            except RuntimeError:
                local_devices = jax.devices()

        if n_devices == "auto":
            n_local = len(local_devices)
        else:
            n_local = min(int(n_devices), len(local_devices))

        total_devices = world_size * n_local

        # Validate decomposition for cubed-sphere
        if grid_type == "cubed_sphere":
            validate_device_count(total_devices, grid_type)

        # Create device mesh for local devices
        from legoesm.parallel.mesh import create_device_mesh, DeviceConfig
        from legoesm.parallel.layout import SingleRankLayout, make_layout

        _gn = grid_n or 1  # default grid_n for layout construction

        if n_local <= 1 and not is_mpi:
            # Serial mode
            device_config = DeviceConfig(
                mesh=None,
                face_sharding=None,
                replicated_sharding=None,
                n_devices=1,
                backend=jax.default_backend().upper(),
                is_distributed=False,
                tiling=(1, 1),
                grid_type=grid_type,
            )
            layout = make_layout(0, 1, _gn)
            mode = "serial"
            halo = HaloBackend.LOCAL
            reduction = ReductionBackend.LOCAL

        elif not is_mpi and n_local > 1:
            # Multi-device single process
            device_config = create_device_mesh(
                n_devices=n_local, backend=backend,
            )
            layout = make_layout(0, 1, _gn)
            mode = "multi_device"
            halo = HaloBackend.JAX_SPMD
            reduction = ReductionBackend.LOCAL

        elif is_mpi and n_local <= 1:
            # MPI, one device per rank
            device_config = DeviceConfig(
                mesh=None,
                face_sharding=None,
                replicated_sharding=None,
                n_devices=1,
                backend=jax.default_backend().upper(),
                is_distributed=True,
                tiling=(1, 1),
                grid_type=grid_type,
            )
            layout = make_layout(rank, world_size, _gn)
            mode = "mpi"
            halo = HaloBackend.MPI
            reduction = ReductionBackend.MPI

        else:
            # Hybrid: MPI + multi-device per rank
            device_config = create_device_mesh(
                n_devices=n_local, backend=backend,
            )
            device_config = device_config._replace(is_distributed=True)
            layout = make_layout(rank, world_size, _gn)
            mode = "hybrid"
            halo = HaloBackend.HYBRID
            reduction = ReductionBackend.HYBRID

        # Build topology for MPI modes
        topology = None
        if is_mpi and grid_type == "cubed_sphere":
            from legoesm.parallel.comm import build_comm_topology
            topology = build_comm_topology(rank, world_size)

        rt = cls(
            mode=mode,
            rank=rank,
            world_size=world_size,
            local_device_count=n_local,
            device_config=device_config,
            layout=layout,
            topology=topology,
            halo_backend=halo,
            reduction_backend=reduction,
            grid_type=grid_type,
        )

        logger.info(
            "ParallelRuntime: mode=%s, rank=%d/%d, local_devices=%d, "
            "total_devices=%d, halo=%s, reduction=%s",
            mode, rank, world_size, n_local, total_devices, halo, reduction,
        )
        return rt

    # ------------------------------------------------------------------
    # Halo dispatch
    # ------------------------------------------------------------------

    def halo_exchange(self, data, grid, halo: int = 1):
        """Dispatch halo exchange to the correct backend.

        Parameters
        ----------
        data : jax.Array
            Field to exchange halos for.
        grid : CubedSphereGrid
            Grid with connectivity metadata.
        halo : int
            Halo width.

        Returns
        -------
        jax.Array
            Halo-padded field.
        """
        if self.halo_backend == HaloBackend.LOCAL:
            from legoesm.grids.halo import pad_halo
            return pad_halo(data, halo=halo)

        if self.halo_backend == HaloBackend.JAX_SPMD:
            # On multi-device single-process, pad_halo works because
            # XLA automatically inserts collectives when reading
            # across sharded dimensions.
            from legoesm.grids.halo import pad_halo
            return pad_halo(data, halo=halo)

        if self.halo_backend == HaloBackend.MPI:
            from legoesm.parallel.halo_exchange import pad_halo_mpi
            return pad_halo_mpi(data, self.topology, halo=halo)

        if self.halo_backend == HaloBackend.HYBRID:
            # MPI for inter-rank, JAX SPMD for intra-rank.
            # For now, delegate to MPI (conservative; XLA handles
            # intra-rank collectives implicitly).
            from legoesm.parallel.halo_exchange import pad_halo_mpi
            return pad_halo_mpi(data, self.topology, halo=halo)

        raise ValueError(f"Unknown halo backend: {self.halo_backend}")

    # ------------------------------------------------------------------
    # Reduction dispatch
    # ------------------------------------------------------------------

    def global_sum(self, local_value):
        """Global sum across all ranks/devices."""
        if self.reduction_backend in (ReductionBackend.LOCAL, ReductionBackend.JAX_PSUM):
            return local_value

        from legoesm.parallel.reductions import global_sum_mpi
        return global_sum_mpi(local_value)

    def global_max(self, local_value):
        """Global max across all ranks/devices."""
        if self.reduction_backend in (ReductionBackend.LOCAL, ReductionBackend.JAX_PSUM):
            return local_value

        from legoesm.parallel.reductions import global_max_mpi
        return global_max_mpi(local_value)

    # ------------------------------------------------------------------
    # Data movement
    # ------------------------------------------------------------------

    def scatter(self, global_data):
        """Extract this rank's portion from a global array."""
        from legoesm.parallel.layout import scatter, SingleRankLayout
        if isinstance(self.layout, SingleRankLayout):
            return global_data
        return scatter(global_data, self.layout)

    def gather(self, local_data, root_only: bool = False):
        """Reconstruct global array from rank-local data."""
        from legoesm.parallel.layout import gather, SingleRankLayout
        if isinstance(self.layout, SingleRankLayout):
            return local_data
        return gather(local_data, self.layout, root_only=root_only)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def describe(self) -> dict:
        """Return metadata dict for benchmark output."""
        return {
            "mode": self.mode,
            "rank": self.rank,
            "world_size": self.world_size,
            "local_device_count": self.local_device_count,
            "global_n_devices": self.global_n_devices,
            "backend": self.device_config.backend,
            "grid_type": self.grid_type,
            "tiling": self.device_config.tiling,
            "halo_backend": self.halo_backend,
            "reduction_backend": self.reduction_backend,
            "jax_version": jax.__version__,
        }

    def __repr__(self) -> str:
        return (
            f"ParallelRuntime(mode={self.mode!r}, rank={self.rank}/{self.world_size}, "
            f"local_devices={self.local_device_count}, "
            f"halo={self.halo_backend}, reduction={self.reduction_backend})"
        )
