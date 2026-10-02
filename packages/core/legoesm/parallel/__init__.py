"""Parallelism utilities for legoESM.

Provides device mesh setup, array sharding, and distributed
communication for multi-GPU, multi-CPU, and multi-node execution.

Parallelism strategies
----------------------
1. **Cubed-sphere face sharding** (1–6 devices):
   ``create_device_mesh()`` shards the 6-face dimension.

2. **Sub-face tiling** (>6 devices, multiples of 6):
   ``create_device_mesh()`` splits each face into tiles,
   enabling up to 6 × tx × ty devices.

3. **Lat-lon domain decomposition**:
   ``create_latlon_mesh()`` shards by latitude.

4. **Level-parallel spectral**:
   ``create_level_mesh()`` distributes SH transforms across devices.

5. **Multi-node MPI**:
   ``initialize_distributed()`` for mpi4jax-based halo exchange
   and reductions, with face-only or sub-face tiling decomposition.

6. **Metal (Apple Silicon)**:
   ``get_metal_config()`` for hybrid Metal/CPU routing.

7. **Voronoi mesh decomposition**:
   ``partition_voronoi_mesh()`` partitions unstructured MPAS/Voronoi
   meshes via a capability-aware ``method="auto"`` default (METIS graph
   partition when ``pymetis`` is present — the ``[mesh]`` extra — else
   geometric RCB), or an explicit ``"sfc"`` Hilbert space-filling-curve
   partitioner; local arrays use owned-first indexing. Halo exchange via
   ``VoronoiHaloExchange``.

8. **Ensemble parallelism**:
   ``make_ensemble_step()`` vmaps the model step over an ensemble
   dimension; ``ensemble_integrate()`` combines scan + vmap for
   efficient time-integration of many members.  Multi-device
   sharding via ``shard_ensemble()``.

9. **Sharded dynamics** (shard_map-based):
   ``make_sharded_step()`` wraps a dynamics model for SPMD execution
   across devices with explicit sharding constraints and optional
   halo exchange.  ``shard_state()`` / ``gather_state()`` move data
   between single-device and multi-device layouts.

10. **Async halo exchange (compute/communication overlap)**:
   ``overlapped_halo_compute()`` splits stencil evaluation into
   interior (halo-independent) and boundary (halo-dependent) parts,
   enabling the XLA compiler to overlap interior computation with
   halo communication scheduling.

11. **Device configuration and hardware-aware optimization**:
    ``detect_devices()`` auto-detects hardware (CPU/GPU/TPU/Metal)
    and ``mixed_precision_policy()`` returns optimal dtype policies.
"""

from legoesm.parallel.async_halo import (
    InteriorBoundaryMasks,
    OverlapContext,
    boundary_slices,
    create_interior_boundary_masks,
    create_overlap_context,
    extract_interior_padded,
    interior_slice,
    merge_interior_boundary,
    overlapped_halo_compute,
    split_interior_boundary,
    start_halo_exchange,
)
from legoesm.parallel.device_config import (
    HardwareConfig,
    MixedPrecisionPolicy,
    get_optimal_dtype,
    mixed_precision_policy,
)
from legoesm.parallel.device_config import (
    detect_devices as detect_hardware,
)
from legoesm.parallel.ensemble import (
    create_ensemble_mesh,
    ensemble_integrate,
    ensemble_integrate_with_forcing,
    ensemble_mean,
    ensemble_percentile,
    ensemble_spread,
    ensemble_std,
    gather_ensemble,
    make_ensemble_step,
    make_ensemble_step_jit,
    perturb_initial_conditions,
    perturb_parameters,
    shard_ensemble,
    stack_states,
    unstack_states,
)
from legoesm.parallel.halo_exchange_voronoi import (
    VoronoiHaloExchange,
    exchange_local_simulated,
)
from legoesm.parallel.layout import (
    DistributedLayout,
    FaceOwnership,
    SingleRankLayout,
    make_layout,
)
from legoesm.parallel.layout import (
    gather as layout_gather,
)
from legoesm.parallel.layout import (
    gather_pytree as layout_gather_pytree,
)
from legoesm.parallel.layout import (
    global_reduce as layout_global_reduce,
)
from legoesm.parallel.layout import (
    local_max as layout_local_max,
)
from legoesm.parallel.layout import (
    local_min as layout_local_min,
)
from legoesm.parallel.layout import (
    local_sum as layout_local_sum,
)
from legoesm.parallel.layout import (
    scatter as layout_scatter,
)
from legoesm.parallel.layout import (
    scatter_pytree as layout_scatter_pytree,
)
from legoesm.parallel.mesh import (
    DeviceConfig,
    create_device_mesh,
    create_latlon_mesh,
    create_level_mesh,
    create_voronoi_device_mesh,
    get_active_config,
    replicate_pytree,
    shard_pytree,
)
from legoesm.parallel.profiling import (
    get_stats as get_mpi_profile_stats,
)
from legoesm.parallel.profiling import (
    is_profiling_enabled,
    mpi_timer,
    print_mpi_profile,
)
from legoesm.parallel.profiling import (
    reset_stats as reset_mpi_profile_stats,
)
from legoesm.parallel.reductions import (
    batch_allreduce_mpi,
)
from legoesm.parallel.runtime import (
    HaloBackend,
    ParallelRuntime,
    ReductionBackend,
    validate_device_count,
)
from legoesm.parallel.sharded_dynamics import (
    CompiledShardedStep,
    StepCacheKey,
    check_sharding,
    create_output_shardings,
    gather_voronoi_state_spmd,
    greedy_edge_coloring,
    make_sharded_step,
    make_voronoi_sharded_step,
    multi_ordering_edge_coloring,
    sharded_integrate,
    sharded_integrate_scan,
    sharded_step_with_halo,
)
from legoesm.parallel.sharded_dynamics import (
    gather_state as gather_state_from_devices,
)
from legoesm.parallel.sharded_dynamics import (
    shard_state as shard_state_to_devices,
)
from legoesm.parallel.voronoi_partition import (
    HaloCommSchedule,
    VoronoiPartition,
    build_local_mesh,
    hilbert_cell_keys,
    partition_cells_geometric,
    partition_cells_sfc,
    partition_voronoi_mesh,
    reorder_voronoi_for_sharding,
    resolve_partition_method,
    scatter_to_local,
)

__all__ = [
    # Canonical runtime (preferred entry point)
    "ParallelRuntime",
    "HaloBackend",
    "ReductionBackend",
    "validate_device_count",
    # Device mesh
    "DeviceConfig",
    "create_device_mesh",
    "create_latlon_mesh",
    "create_level_mesh",
    "create_voronoi_device_mesh",
    "get_active_config",
    "replicate_pytree",
    "shard_pytree",
    # Voronoi mesh decomposition
    "HaloCommSchedule",
    "VoronoiPartition",
    "partition_cells_geometric",
    "partition_cells_sfc",
    "hilbert_cell_keys",
    "partition_voronoi_mesh",
    "build_local_mesh",
    "scatter_to_local",
    "reorder_voronoi_for_sharding",
    "resolve_partition_method",
    "VoronoiHaloExchange",
    "exchange_local_simulated",
    # Ensemble parallelism
    "stack_states",
    "unstack_states",
    "perturb_initial_conditions",
    "perturb_parameters",
    "make_ensemble_step",
    "make_ensemble_step_jit",
    "ensemble_integrate",
    "ensemble_integrate_with_forcing",
    "ensemble_mean",
    "ensemble_std",
    "ensemble_percentile",
    "ensemble_spread",
    "create_ensemble_mesh",
    "shard_ensemble",
    "gather_ensemble",
    # Async halo exchange (compute/communication overlap)
    "InteriorBoundaryMasks",
    "OverlapContext",
    "create_interior_boundary_masks",
    "split_interior_boundary",
    "merge_interior_boundary",
    "boundary_slices",
    "interior_slice",
    "extract_interior_padded",
    "overlapped_halo_compute",
    "create_overlap_context",
    "start_halo_exchange",
    # Sharded dynamics (shard_map-based SPMD)
    "StepCacheKey",
    "CompiledShardedStep",
    "make_sharded_step",
    "shard_state_to_devices",
    "gather_state_from_devices",
    "gather_voronoi_state_spmd",
    "create_output_shardings",
    "sharded_step_with_halo",
    "sharded_integrate",
    "sharded_integrate_scan",
    "check_sharding",
    # Batch MPI reductions
    "batch_allreduce_mpi",
    # MPI profiling
    "is_profiling_enabled",
    "mpi_timer",
    "get_mpi_profile_stats",
    "reset_mpi_profile_stats",
    "print_mpi_profile",
    # Device configuration and hardware-aware optimization
    "HardwareConfig",
    "detect_hardware",
    "get_optimal_dtype",
    "MixedPrecisionPolicy",
    "mixed_precision_policy",
]
