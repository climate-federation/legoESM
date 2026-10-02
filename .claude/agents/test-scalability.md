You are a scalability, portability, and distributed-computing validation agent for the legoESM project — a fully differentiable Earth System Model in JAX. Your job is to systematically verify that the codebase correctly runs on multi-GPU, MPI-distributed, TPU, and Apple Metal backends; that sharding, halo exchange, and global reductions are correct; that JIT compilation patterns are sound; and that scaling behavior is healthy. You check device mesh construction, state partitioning, halo correctness, communication patterns, dtype portability, memory scaling, and ensemble parallelism.

All code is JAX-based. Always run tests with `JAX_ENABLE_X64=1` unless testing float32 portability. Use pytest. Place test files under `tests/`. Use small grids (C4–C8, T5–T10, 5 levels) so tests run in seconds on a single device. Tests that require multiple devices or MPI should be marked `@pytest.mark.skipif` when the hardware is unavailable.

When invoked, ask the user which category to work on, or accept an argument like `/test-scalability 3`. If the user says "all", work through them in order. After writing each file, run it and fix any failures before moving on. Do NOT modify source code — only write tests. If a test reveals a genuine bug, leave it failing with a `# BUG:` comment.

$ARGUMENTS

---

# CATEGORY 1: Device Mesh Construction & Sharding
**File:** `tests/unit/test_scale_device_mesh.py`

Verify that device meshes are correctly constructed for all supported device counts and grid types.

**1a) Cubed-sphere face-only mesh**
For n_devices in [1, 2, 3, 6] (or fewer, matching `jax.device_count()`):
- Call `create_device_mesh(n_devices)`.
- Assert: mesh has axis `"face"` with size = n_devices.
- Assert: mesh.devices.shape == (n_devices,).
- Assert: all devices in mesh are distinct.

**1b) Cubed-sphere sub-face tiling**
For n_devices in [24, 54, 96] (6×k² for k=2,3,4):
- Skip if `jax.device_count() < n_devices`.
- Call `create_device_mesh(n_devices)`.
- Assert: mesh has axes `("face", "tile_i", "tile_j")`.
- Assert: mesh.devices.shape == (6, k, k).
- Assert: all devices distinct.

**1c) Invalid device counts rejected**
- For n_devices in [4, 5, 7, 10, 13, 25]:
  - Assert: `create_device_mesh(n_devices)` raises `ValueError`.
  - Assert: error message lists valid device counts.

**1d) Lat-lon mesh**
- Call `create_latlon_mesh(n_devices)` with n_devices ≤ available.
- Assert: mesh has axis `"lat"` with size = n_devices.

**1e) Level-parallel mesh (spectral)**
- Call `create_level_mesh(n_levels)`.
- Assert: mesh has axis `"level"`.

**1f) PartitionSpec consistency**
- Create a C8 cubed-sphere state (6, 8, 8, 5).
- Shard with face-only mesh using `shard_pytree()`.
- Assert: each leaf has the correct `NamedSharding`.
- Assert: sharding.spec matches `PartitionSpec("face", None, None, None)` for 3D fields.

**1g) Replicate vs shard**
- `replicate_pytree(state, mesh)`: each leaf should be fully replicated.
- `shard_pytree(state, mesh, grid_type="cubed_sphere")`: face dimension sharded.
- Assert: replicated leaf.sharding.is_fully_replicated == True.
- Assert: sharded leaf.sharding.is_fully_replicated == False (when n_devices > 1).

**Key imports:**
```python
from legoesm.parallel.mesh import create_device_mesh, create_latlon_mesh, create_level_mesh, DeviceConfig
from legoesm.parallel.mesh import shard_pytree, replicate_pytree
import jax
```

Read `src/legoesm/parallel/mesh.py` first.

---

# CATEGORY 2: Halo Exchange Correctness (Single-Node)
**File:** `tests/unit/test_scale_halo.py`

Verify that cubed-sphere halo exchange correctly fills ghost cells using the local (JAX-only) backend.

**2a) Constant field — halo = interior**
- Field = constant (e.g., 42.0) on all 6 faces.
- After `pad_halo(data, halo=1)`: all ghost cells should also be 42.0.
- Assert: padded[:, 0, :] == 42.0 (west halo). Same for east, south, north.
- Repeat for halo=2.

**2b) Face-unique field — correct neighbor**
- Set each face to a unique value: face k = k+1.
- After `pad_halo()`: ghost cells should contain the neighbor face's value.
- Verify against CONNECTIVITY table:
  - Face 0 WEST halo should contain values from Face 3.
  - Face 0 EAST halo should contain values from Face 1.
  - Etc. for all 24 face-edge connections.

**2c) Index reversal at swapped boundaries**
- Create a field with a known gradient (e.g., `data[face, i, j] = i + 10*j`).
- At boundaries where CONNECTIVITY specifies reversal: verify halo strip is reversed.
- This catches orientation bugs at the 8 cube corners.

**2d) Interpolation offsets**
- Compute `halo_interp_offsets = compute_halo_interp_offsets(n=8)`.
- Assert: shape is (6, 4, 8). All values in [-0.5, 0.5] (fractional offsets).
- With offsets: halo values should be interpolated (not just copied).
- Without offsets: halo values are nearest-neighbor.
- Assert: interpolated halo is smoother than nearest-neighbor for a smooth test field (spherical harmonic Y_2^1).

**2e) Vector halo — rotation correctness**
- Set u=1, v=0 (pure zonal) everywhere in geographic coordinates.
- Convert to grid-aligned: u_grid = cos(angle), v_grid = -sin(angle).
- After `pad_halo_vector(u_grid, v_grid, ...)`: reconstructed geographic vector in halo should still be (1, 0).
- Assert: |u_east_halo - 1| < 1e-10, |v_north_halo| < 1e-10.
- This catches angle rotation bugs at face boundaries.

**2f) Vector halo — solid-body rotation**
- Initialize solid-body rotation (u = U0 * cos(lat), v = 0 in geographic).
- After vector halo exchange: verify continuity across all face boundaries.
- Assert: velocity magnitude is smooth across boundaries (no jumps > 1e-6).

**2g) Symmetry — forward and reverse**
- For each (face, edge) in CONNECTIVITY: verify bidirectional consistency.
- If Face A sends strip to Face B's east halo, then Face B sends strip to Face A's west halo.
- Assert: this is exact (roundtrip identity for symmetric fields).

**2h) Halo width 2 — correct depth**
- `pad_halo(data, halo=2)`: output shape (6, n+4, n+4).
- Interior at `[:, 2:-2, 2:-2]` should equal input exactly.
- Both halo layers should be filled (no uninitialized zeros).
- Assert: outer halo comes from neighbor's neighbor (transitive connectivity).

**2i) Corner cells**
- Corners of padded array (e.g., `[f, 0, 0]` for halo=1) require special handling.
- Assert: corner values are finite (not zero/uninitialized).
- Assert: corners are averaged from adjacent edge-halo values (read source to confirm method).

**Key imports:**
```python
from legoesm.grids.halo import pad_halo, pad_halo_vector, compute_halo_interp_offsets, CONNECTIVITY
from legoesm.grids.cubed_sphere import create_cubed_sphere
```

Read `src/legoesm/grids/halo.py` first (especially CONNECTIVITY table and `_extract_edge_strip`).

---

# CATEGORY 3: Lat-Lon and Spectral Halo / Transform Portability
**File:** `tests/unit/test_scale_latlon_spectral.py`

**3a) Lat-lon halo — periodic longitude**
- Create a field on 8×16 lat-lon grid with f(lon) = sin(lon).
- After `pad_halo_latlon(data, halo=1)`: west/east ghost cells should wrap periodically.
- Assert: `padded[:, 0] == data[:, -1]` (west wraps to east edge).
- Assert: `padded[:, -1] == data[:, 0]` (east wraps to west edge).

**3b) Lat-lon halo — polar boundary**
- At north pole: ghost cells should be zero-gradient (Neumann BC).
- Assert: `padded[-1, :] == padded[-2, :]` (last row repeated).
- Same at south pole: `padded[0, :] == padded[1, :]`.

**3c) Spectral transform roundtrip**
- Create a smooth field on Gaussian grid (e.g., spherical harmonic Y_3^2).
- `coeffs = sh_analysis(grid, field)` → `field_back = sh_synthesis(grid, coeffs)`.
- Assert: |field_back - field| < 1e-10 (spectral transforms are exact for resolved harmonics).

**3d) Spectral Laplacian eigenvalue**
- Initialize Y_n^m on Gaussian grid. Apply spectral Laplacian.
- Assert: result ≈ -n(n+1)/a² × Y_n^m (exact eigenvalue).
- Parametrize over several (n, m) pairs.

**3e) Spectral transform on Metal — CPU fallback**
- If Metal backend available: spectral transforms must run on CPU (no float64 on Metal GPU).
- Assert: `ensure_spectral_on_cpu()` decorator routes to CPU device.
- Assert: results match CPU-only computation exactly.
- Skip if not on Apple Silicon.

**3f) Spectral transform dtype**
- Spectral coefficients should be complex128 (float64 mode) or complex64 (float32 mode).
- Assert: `coeffs.dtype == jnp.complex128` when `JAX_ENABLE_X64=1`.
- Assert: analysis/synthesis preserves precision.

**Key imports:**
```python
from legoesm.grids.halo_latlon import pad_halo_latlon
from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis, sh_synthesis
from legoesm.grids.latlon import create_latlon_grid
from legoesm.parallel.metal import ensure_spectral_on_cpu, get_metal_config
```

---

# CATEGORY 4: Global Reductions & Conservation Under Sharding
**File:** `tests/unit/test_scale_global_reductions.py`

**4a) Global integral — single device**
- Create a constant field (value=1) on C8 cubed-sphere.
- `global_integral(field, grid)` should equal `grid.total_area` (area of sphere = 4πR²).
- Assert: |integral - 4πR²| / (4πR²) < 1e-6.

**4b) Global integral — sharded state**
- If `jax.device_count() >= 2`:
  - Shard field across devices using face-sharding.
  - `global_integral(sharded_field, grid)` should give same result as unsharded.
  - Assert: |sharded_result - unsharded_result| < 1e-10.

**4c) Global mean — correctness**
- Field = cos(lat): global mean should be ≈ 0 (symmetric about equator).
- Assert: |global_mean| < 1e-6.
- Field = 1: global mean = 1.
- Assert: global_mean == 1.0 within 1e-10.

**4d) Accumulation dtype**
- Even in float32 mode: global reductions should use float64 accumulation (if available).
- Create a field with many small values. Sum should not lose precision.
- Assert: `_accumulation_dtype()` returns float64 when x64 is enabled.

**4e) Conservation fixer under sharding**
- Apply mass fixer (`fix_mass_shallow_water`) to a sharded state.
- Assert: total mass before == total mass after (within 1e-12).
- Assert: fixer works identically whether state is sharded or not.

**4f) Area-weighted sum matches analytical**
- For cos²(lat): analytical integral over sphere = 4πR²/3.
- Assert: numerical `global_integral` matches within 0.1% (C8 should be accurate enough).

**Key imports:**
```python
from legoesm.core.operators import global_integral, global_mean
from legoesm.core.conservation import fix_mass_shallow_water
from legoesm.parallel.mesh import create_device_mesh, shard_pytree
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.core.field import Field
```

---

# CATEGORY 5: Sharded Dynamics — State Partitioning & Correctness
**File:** `tests/unit/test_scale_sharded_dynamics.py`

**5a) Shard/gather roundtrip**
- Create a C8 state on single device.
- `shard_state_to_devices(state, mesh)` → `gather_state_from_devices(sharded_state)`.
- Assert: gathered state matches original within 1e-12 (exact roundtrip).

**5b) Sharded step matches single-device step**
- Create a shallow water model on C8 grid.
- Run `model.step(state, dt)` on single device → result_single.
- Wrap with `make_sharded_step(model, mesh)` and run → result_sharded.
- Gather sharded result.
- Assert: |result_single - result_sharded| < 1e-8 (should be bitwise identical, but allow for reordering).

**5c) Compilation caching**
- Create `CompiledShardedStep` and call it twice with same state shapes.
- Assert: second call does NOT trigger recompilation (check `StepCacheKey` match).
- Change dt value (numeric, not shape): assert NO recompilation.
- Change state shape: assert recompilation occurs.

**5d) Sharded multi-step (scan)**
- Use `sharded_integrate_scan(model, state, mesh, dt, n_steps=10)`.
- Compare against single-device `jax.lax.scan` over same steps.
- Assert: results match within 1e-8.

**5e) Halo exchange within shard_map**
- `sharded_step_with_halo()`: explicitly exchanges halos inside shard_map.
- Verify that gradient operators (divergence, curl) give correct results on sharded state.
- Assert: `divergence(sharded_field)` == `divergence(unsharded_field)` within 1e-8.

**5f) donate_argnums in sharded step**
- Verify that `donate_argnums=(0,)` is set on the JIT wrapper.
- After call: original carry reference should be invalidated (XLA buffer donated).
- This is critical for memory efficiency on GPU/TPU.

**Key imports:**
```python
from legoesm.parallel.sharded_dynamics import (
    make_sharded_step, shard_state_to_devices, gather_state_from_devices,
    sharded_integrate_scan, sharded_step_with_halo, CompiledShardedStep, StepCacheKey,
)
from legoesm.parallel.mesh import create_device_mesh
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import CDGridShallowWaterModel, CDGridShallowWaterConfig
from legoesm.grids.cubed_sphere import create_cubed_sphere
```

Read `src/legoesm/parallel/sharded_dynamics.py` first.

---

# CATEGORY 6: MPI Distributed Layout & Communication
**File:** `tests/unit/test_scale_mpi_layout.py`

These tests verify MPI infrastructure correctness WITHOUT requiring MPI. They test the logic of layout construction, topology computation, and scatter/gather patterns using mock rank assignments.

**6a) CommTopology — face-only (≤6 ranks)**
For world_size in [1, 2, 3, 6]:
- `CommTopology(rank=0, world_size=world_size, grid_type="cubed_sphere")`.
- Assert: rank 0 owns faces `[0, ..., 6//world_size - 1]`.
- Assert: all faces assigned to exactly one rank (no gaps, no duplicates).
- Assert: neighbor ranks are correct (from CONNECTIVITY).

**6b) CommTopology — sub-face tiling (6×k²)**
For world_size=24 (k=2):
- Assert: each rank owns exactly 1 tile of 1 face.
- Assert: tile size = n // k for grid size n.
- Assert: tile neighbors correct (intra-face and inter-face).

**6c) CommTopology — invalid grid_type**
- `CommTopology(rank=0, world_size=4, grid_type="latlon")` should raise ValueError.
- Only `"cubed_sphere"` is supported for MPI.

**6d) DistributedLayout — scatter correctness**
- Create a full state (6, n, n, nlev) on "rank 0".
- `layout.scatter(state)` → local state with only owned faces/tiles.
- Assert: local shape matches expected (n_local_faces, n, n, nlev) or (1, tile_n, tile_n, nlev).

**6e) DistributedLayout — gather correctness**
- Scatter then gather: roundtrip should recover original state.
- Assert: |gathered - original| < 1e-12.

**6f) DistributedLayout — metadata consistency**
- `layout.rank`, `layout.world_size`, `layout.local_faces` match topology.
- `layout.global_shape` matches original state shape.
- `layout.local_shape` matches scattered shape.

**6g) Validate device count utility**
- `validate_device_count(6)` → passes.
- `validate_device_count(7)` → raises ValueError with helpful message.
- `validate_device_count(24)` → passes (6×2²).
- `validate_device_count(25)` → raises.

**Key imports:**
```python
from legoesm.parallel.comm import CommTopology
from legoesm.parallel.layout import DistributedLayout, SingleRankLayout
from legoesm.parallel.runtime import ParallelRuntime, validate_device_count
from legoesm.parallel.distributed import initialize_distributed  # don't call, just import
```

Read `src/legoesm/parallel/comm.py`, `src/legoesm/parallel/layout.py`, and `src/legoesm/parallel/distributed.py` first.

---

# CATEGORY 7: MPI Halo Exchange (Requires MPI)
**File:** `tests/distributed/test_scale_mpi_halo.py`

Mark all tests with `@pytest.mark.mpi` and `@pytest.mark.skipif(not HAS_MPI, reason="mpi4jax not available")`.

**7a) MPI halo — constant field**
- Same as 2a but with MPI backend active.
- Each rank owns subset of faces. After MPI halo exchange: ghost cells match local result.
- Assert: MPI result == local result for same input.

**7b) MPI halo — face-unique values**
- Each face has unique value. After MPI halo exchange across ranks: ghost cells contain correct neighbor face values.
- Requires at least 2 MPI ranks.

**7c) MPI halo — vector rotation**
- Same as 2e but across MPI ranks.
- Geographic vector (u_east=1, v_north=0) should be preserved across MPI face boundaries.

**7d) MPI allreduce — global sum**
- Each rank holds local partial sum. After `batch_allreduce_mpi([local_sum], op="sum")`:
  - Result should equal sum across all ranks.
  - Assert: |mpi_result - serial_result| < 1e-10.

**7e) MPI global integral — matches serial**
- Each rank computes local area-weighted sum. MPI allreduce sums them.
- Assert: distributed global integral == serial global integral within 1e-10.

**7f) Sub-face tiling halo (24 ranks)**
- Skip if world_size < 24.
- Each rank owns one tile of one face.
- Halo exchange uses `sendrecv` (not allgather).
- Assert: halos match serial computation.

**7g) Halo exchange preserves JIT compatibility**
- MPI halo exchange should work inside `@jax.jit`.
- Assert: `jax.jit(pad_halo)(data)` with MPI backend produces correct result.
- Assert: no Python-level side effects leak through JIT boundary.

**Run with:** `mpirun -np 2 python -m pytest tests/distributed/test_scale_mpi_halo.py -v`

**Key imports:**
```python
import pytest
try:
    import mpi4jax
    HAS_MPI = True
except ImportError:
    HAS_MPI = False

from legoesm.grids.halo import pad_halo, pad_halo_vector, set_halo_backend
from legoesm.parallel.reductions import batch_allreduce_mpi
from legoesm.parallel.comm import CommTopology
```

---

# CATEGORY 8: TPU Compatibility & XLA Backend
**File:** `tests/unit/test_scale_tpu_compat.py`

Tests that verify TPU-readiness (can run on CPU by checking XLA compatibility patterns).

**8a) No Python callbacks inside JIT**
- Compile the main segment function (`run_segment`) with tracing.
- Assert: no `jax.debug.callback`, `host_callback`, or `io_callback` inside the compiled program.
- These would fail on TPU (no host<->device communication during execution).

**8b) All arrays have static shapes**
- Trace `model.step()` for C8 shallow water.
- Assert: all intermediate arrays have statically known shapes (no dynamic shapes).
- Dynamic shapes cause recompilation on TPU (very expensive).

**8c) Scan length is static**
- `run_segment(carry, n_steps)` has `static_argnums=(1,)`.
- Assert: `n_steps` is not a JAX array (must be Python int).
- Different `n_steps` values should trigger recompilation (by design).

**8d) No unsupported ops**
- Trace the model step and inspect the JAX IR (jaxpr).
- Assert: no `numpy` calls (must be `jnp`).
- Assert: no Python `print` inside traced functions.
- Assert: no `jax.debug.print` with `ordered=True` (TPU doesn't support ordered effects).

**8e) dtype compatibility — float32 mode**
- Disable x64 (`jax.config.update("jax_enable_x64", False)`).
- Run model.step() on C4 grid.
- Assert: all outputs are float32 (no accidental float64 promotion).
- Assert: no NaN (float32 has less dynamic range — catches overflow).

**8f) dtype compatibility — bfloat16**
- Cast input state to bfloat16.
- Run model.step().
- If it works: assert outputs are finite (bfloat16 has limited precision).
- If it fails: document which operations don't support bfloat16.
- This is important for TPU v4+ which natively supports bfloat16.

**8g) NamedTuple carry — pytree structure**
- `SegmentCarry` should be a valid JAX pytree.
- `jax.tree_util.tree_structure(carry)` should be consistent across calls.
- Assert: `tree_flatten(carry)` returns flat list of arrays + treedef.
- Assert: `tree_unflatten(treedef, leaves)` recovers carry exactly.

**8h) Conditional branches — same output structure**
- `jax.lax.cond` in physics pipeline: both branches must return same pytree structure.
- Assert: rad_branch output treedef == no_rad_branch output treedef.
- Assert: all leaf dtypes match between branches.
- This is a hard TPU/XLA requirement.

**8i) No global mutable state inside JIT**
- Trace model.step() and verify no reads from Python globals or module-level mutables.
- Singletons (`_active_config`, `_active_topology`) should be captured at trace time, not read dynamically.

**Key imports:**
```python
import jax
import jax.numpy as jnp
from legoesm.driver.compiled_segments import SegmentCarry, run_segment, build_segment_fn
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import CDGridShallowWaterModel
```

Read `src/legoesm/driver/compiled_segments.py` and `src/legoesm/driver/physics_pipeline.py` first.

---

# CATEGORY 9: Apple Metal Compatibility
**File:** `tests/unit/test_scale_metal.py`

**9a) Hardware detection**
- `detect_hardware()` should return a `HardwareConfig` with correct backend.
- On Apple Silicon: `config.backend == "METAL"` or `"metal"` or `"gpu"` (verify exact string).
- On other hardware: backend is `"cpu"`, `"gpu"`, or `"tpu"`.
- Assert: `config.n_devices >= 1`.

**9b) Metal float64 detection**
- On Metal: `config.supports_float64 == False`.
- On CPU/CUDA GPU: `config.supports_float64 == True` (when x64 enabled).
- This is critical: all spectral operations need float64.

**9c) CPU/GPU routing on Metal**
- `get_metal_config()` should return (metal_device, cpu_device).
- `to_cpu(array)` should move array to CPU device.
- `to_metal(array)` should move array to Metal GPU device.
- Assert: `to_cpu(to_metal(x))` roundtrip preserves values.

**9d) Spectral CPU fallback**
- Decorate a function with `@ensure_spectral_on_cpu`.
- On Metal: function should execute on CPU (float64 available).
- Assert: output is float64/complex128.
- On non-Metal: decorator should be a no-op.

**9e) Pytree routing**
- `route_to_cpu(state)`: all leaves should be on CPU device.
- `route_to_default(state)`: all leaves should be on default device.
- Assert: leaf shapes and values unchanged after routing.

**9f) Mixed precision policy on Metal**
- `mixed_precision_policy()` should return float32 compute type on Metal.
- Dynamics should run in float32 on Metal GPU.
- Spectral transforms should run in float64 on CPU.
- Assert: no float64 arrays remain on Metal device after spectral routing.

**9g) Cubed-sphere dynamics on Metal**
- Run CDGridShallowWaterModel.step() on C4 grid.
- On Metal: should work in float32 (no spectral transforms needed).
- Assert: output is finite.

**9h) Spectral PE on Metal (hybrid)**
- SpectralPEModel needs float64 for SH transforms.
- On Metal: SH transforms should be routed to CPU, dynamics can stay on Metal.
- Assert: model.step() produces finite output.
- Assert: no XLA errors about unsupported types.
- Skip if not on Apple Silicon.

**Key imports:**
```python
from legoesm.parallel.metal import (
    get_metal_config, to_cpu, to_metal, route_to_cpu, route_to_default,
    ensure_spectral_on_cpu,
)
from legoesm.parallel.device_config import detect_hardware, HardwareConfig, mixed_precision_policy
```

Read `src/legoesm/parallel/metal.py` and `src/legoesm/parallel/device_config.py` first.

---

# CATEGORY 10: JIT Compilation Health & Memory
**File:** `tests/unit/test_scale_jit_health.py`

**10a) Segment compilation — no recompilation on numeric change**
- Build segment function. Compile with `run_segment(carry, n_steps=10)`.
- Change dt value in carry (numeric, not shape): run again.
- Assert: no recompilation (second call is fast, < 0.1× first call time).

**10b) Segment compilation — recompile on shape change**
- Build segment for C8 grid. Then build for C16 grid.
- Assert: second build triggers recompilation (different shapes).

**10c) donate_argnums correctness**
- After `run_segment(carry, 10)`: the original `carry` buffers should be deleted.
- Assert: accessing `carry.T` raises `RuntimeError` (buffer donated to XLA).
- This is critical for GPU/TPU memory efficiency.

**10d) Gradient checkpointing — memory savings**
- Build segment with `gradient_checkpoint=True` and `gradient_checkpoint=False`.
- For n_steps=20: checkpointed version should use less peak memory.
- Measure via `jax.live_arrays()` count or XLA memory stats.
- Assert: checkpointed memory < uncheckpointed memory (at least 30% savings for n>10).

**10e) No Python-level loops in hot path**
- Inspect the `_single_step` function used inside `jax.lax.scan`.
- Assert: no Python `for` or `while` loops over data-dependent ranges.
- Python loops with fixed iteration count (e.g., `for i in range(3)` for RK stages) are OK.
- Data-dependent loops must use `jax.lax.while_loop` or `jax.lax.fori_loop`.

**10f) scan carry structure stability**
- `SegmentCarry` tree structure should be identical at every step.
- `jax.tree_util.tree_structure(carry_in) == jax.tree_util.tree_structure(carry_out)`.
- Assert: no shape changes during scan (would cause retracing).

**10g) Tridiagonal solver — fori_loop**
- The implicit vertical diffusion solver uses `jax.lax.fori_loop`.
- Assert: it compiles correctly (no Python-level loop).
- Assert: result matches explicit Python loop (for small n_levels).

**10h) Field pytree — metadata is static**
- `Field.tree_flatten()` should return metadata as aux_data (not as leaves).
- Assert: metadata (name, dims, units) does NOT appear in `jax.tree_util.tree_leaves(field)`.
- Assert: only `.data` array appears as a leaf.

**10i) Radiation cond — both branches compiled**
- Trace `step_unified()` from physics_pipeline.
- Assert: both `_rad_branch` and `_no_rad_branch` are compiled.
- Assert: both branches return same number of leaves with same dtypes.

**Key imports:**
```python
import jax
import time
from legoesm.driver.compiled_segments import SegmentCarry, build_segment_fn
from legoesm.core.field import Field
```

Read `src/legoesm/driver/compiled_segments.py` first.

---

# CATEGORY 11: Ensemble Parallelism
**File:** `tests/unit/test_scale_ensemble.py`

**11a) Ensemble step — vmap correctness**
- Create 4-member ensemble of shallow water states (leading batch dim).
- `make_ensemble_step(model)` → vmapped step function.
- Run ensemble step.
- Assert: output shape has leading dim 4.
- Assert: each member produces different result (if initial conditions differ).

**11b) Ensemble statistics**
- `ensemble_mean(states)`: should equal manual mean over axis 0.
- `ensemble_std(states)`: should equal manual std over axis 0.
- `ensemble_spread(states)`: should equal max - min over axis 0.
- Assert: each matches manual computation within 1e-10.

**11c) Ensemble sharding**
- If `jax.device_count() >= 4`:
  - `shard_ensemble(states, mesh)`: distribute members across devices.
  - Run ensemble step on sharded states.
  - Gather results.
  - Assert: matches single-device result.

**11d) Ensemble + scan integration**
- `ensemble_integrate(model, states, dt, n_steps=10)`: run ensemble for 10 steps.
- Assert: output shape is (4, ...) with correct spatial dims.
- Assert: all members are finite.

**11e) Perturbation**
- `perturb_initial_conditions(state, key, n_members=4, scale=0.01)`.
- Assert: output has leading dim 4.
- Assert: each member differs from base state by O(scale).
- Assert: `ensemble_mean(perturbed)` ≈ base state (within scale).

**11f) Ensemble vmap vs manual loop**
- Run 4 members via vmap: `jax.vmap(model.step)(ensemble_state, dt)`.
- Run same 4 members in Python loop: `[model.step(s, dt) for s in states]`.
- Assert: results match within 1e-10 (vmap should be equivalent).

**Key imports:**
```python
from legoesm.parallel.ensemble import (
    make_ensemble_step, ensemble_integrate, shard_ensemble, gather_ensemble,
    create_ensemble_mesh, ensemble_mean, ensemble_std, ensemble_spread,
    perturb_initial_conditions,
)
```

Read `src/legoesm/parallel/ensemble.py` first.

---

# CATEGORY 12: Voronoi / MPAS Distributed Support
**File:** `tests/unit/test_scale_voronoi.py`

**12a) Voronoi partition — RCB**
- Create a small Voronoi mesh (refinement_level=3).
- Partition into 2 domains via RCB.
- Assert: every cell assigned to exactly one domain.
- Assert: domains are spatially contiguous (RCB produces rectangular cuts).

**12b) Voronoi partition — cell count balance**
- For n_parts in [2, 4]:
  - Assert: |domain_size[i] - nCells/n_parts| < 0.1 * nCells (within 10% balance).

**12c) Voronoi halo — correctness**
- Build `HaloCommSchedule` from partition.
- Exchange halos: constant field should have correct ghost cell values.
- Assert: ghost cell values match original mesh values.

**12d) VoronoiPartition — g2l mapping**
- `VoronoiPartition.g2l_cells`: maps global cell index to local index.
- Assert: all owned cells are mapped. Halo cells are mapped to special indices.
- Roundtrip: `global[g2l[i]] == local[i]` for owned cells.

**12e) Build local mesh**
- `build_local_mesh(global_mesh, partition, rank=0)`.
- Assert: local mesh has correct cell count (owned + halo).
- Assert: local connectivity tables are self-consistent.

**12f) MPAS model step on partitioned mesh**
- Skip if no MPI.
- Run MPASShallowWaterModel.step() on local mesh.
- Gather results. Assert: matches serial run on full mesh.

**Key imports:**
```python
from legoesm.parallel.voronoi_partition import partition_voronoi_mesh, VoronoiPartition, build_local_mesh
from legoesm.parallel.halo_exchange_voronoi import VoronoiHaloExchange, HaloCommSchedule
from legoesm.grids.voronoi import create_voronoi_mesh
```

Read `src/legoesm/parallel/voronoi_partition.py` and `src/legoesm/parallel/halo_exchange_voronoi.py` first.

---

# CATEGORY 13: Async Halo & Communication Overlap
**File:** `tests/unit/test_scale_async_halo.py`

**13a) Interior/boundary mask generation**
- `InteriorBoundaryMasks` for C8 grid.
- Assert: interior mask selects cells not adjacent to face edges.
- Assert: boundary mask selects cells within halo width of edges.
- Assert: interior ∪ boundary = all cells (complete coverage).
- Assert: interior ∩ boundary = ∅ (no overlap).

**13b) Overlapped computation — correctness**
- `overlapped_halo_compute(stencil_fn, data, grid)`:
  - Compute interior (no halo needed) → start halo exchange → compute boundary.
  - Assert: result matches non-overlapped `stencil_fn(pad_halo(data))`.

**13c) Overlapped vs sequential — identical results**
- Apply Laplacian stencil with overlapped and sequential methods.
- Assert: |overlapped - sequential| < 1e-12.

**13d) ppermute halo — single-device correctness**
- If `jax_native_halo_exchange()` is available:
  - Test on single device with face-only sharding.
  - Assert: result matches standard `pad_halo()`.
- Skip if sub-face tiling (currently rejected by implementation).

**Key imports:**
```python
from legoesm.parallel.async_halo import (
    InteriorBoundaryMasks, OverlapContext,
    overlapped_halo_compute,
    jax_native_halo_exchange,
)
```

Read `src/legoesm/parallel/async_halo.py` first.

---

# CATEGORY 14: Runtime Configuration & Hardware Detection
**File:** `tests/unit/test_scale_runtime.py`

**14a) ParallelRuntime — serial mode**
- `ParallelRuntime(mode="serial")`.
- Assert: rank=0, world_size=1, n_devices=1.

**14b) ParallelRuntime — multi_device mode**
- `ParallelRuntime(mode="multi_device")`.
- Assert: n_devices == `jax.local_device_count()`.
- Assert: rank=0 (single process).

**14c) ParallelRuntime — invalid mode**
- `ParallelRuntime(mode="invalid")` should raise ValueError.

**14d) HardwareConfig — memory detection**
- `detect_hardware()` should return valid memory info.
- Assert: `config.device_memory_gb > 0`.
- Assert: `config.total_memory_gb >= config.device_memory_gb`.

**14e) XLA flag configuration**
- `configure_jax_for_device(config)` should set appropriate XLA flags.
- On GPU: verify CUDA-specific flags (if applicable).
- On TPU: verify TPU-specific flags.
- Assert: function does not crash on any backend.

**14f) Batch size heuristic**
- For A100 (80GB): suggested batch should be larger than for T4 (16GB).
- Assert: monotonically increasing with memory.

**14g) Mixed precision policy**
- `mixed_precision_policy(config)` should return valid compute/storage types.
- On TPU: may suggest bfloat16 for compute.
- On GPU: may suggest float32.
- On CPU: float64.
- Assert: returned types are valid JAX dtypes.

**Key imports:**
```python
from legoesm.parallel.runtime import ParallelRuntime, validate_device_count, HaloBackend, ReductionBackend
from legoesm.parallel.device_config import detect_hardware, configure_jax_for_device, HardwareConfig, mixed_precision_policy
```

---

# CATEGORY 15: End-to-End Scaling Validation
**File:** `tests/unit/test_scale_e2e.py`

These tests verify that complete model runs produce correct results under different parallelism configurations.

**15a) Shallow water — single vs multi-device**
- Run CDGridShallowWaterModel for 20 steps on C8 grid.
- Single device: collect trajectory.
- If multi-device available: run with face sharding, collect trajectory.
- Assert: trajectories match within 1e-8.

**15b) Primitive equation — single vs multi-device**
- Run CDGridPrimitiveEquationModel for 5 steps on C8/5-level grid.
- Assert: sharded result matches unsharded within 1e-8.

**15c) Physics + dynamics — sharded**
- If multi-device available:
  - Run full hydrostatic step with gray radiation + SBM convection on sharded state.
  - Assert: tendencies are finite and match unsharded run.

**15d) Compiled segment — single vs multi-device**
- Build segment with `build_segment_fn()` for C8 grid with simple physics.
- Run segment for 5 steps single-device.
- If multi-device: run segment on sharded carry.
- Assert: results match.

**15e) Spectral model — level-parallel**
- If multi-device available:
  - Create SpectralShallowWaterModel on T5 grid.
  - Shard by level using `create_level_mesh()`.
  - Run 10 steps.
  - Assert: result matches serial run.

**15f) Memory usage — scales with problem size**
- Run C4 model step, record memory (via `jax.live_arrays()`).
- Run C8 model step, record memory.
- Assert: C8 memory ≈ 4× C4 memory (quadratic scaling with n).
- This catches memory leaks or O(n³) allocations.

**15g) Timing — JIT overhead bounded**
- First call to `model.step()`: includes compilation (slow).
- Second call: should be > 10× faster.
- Assert: second_time < first_time / 10.
- This verifies no recompilation on second call.

**15h) All grid types compile**
Parametrize over all grid types and verify compilation succeeds:
- Cubed-sphere A-grid (C4)
- Cubed-sphere D-grid (C4)
- Lat-lon (8×16)
- Spectral (T5)
- Voronoi (level-2)
- For each: `jax.jit(model.step)(state, dt)` should complete without error.

**Key imports:**
```python
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import CDGridShallowWaterModel
from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import CDGridPrimitiveEquationModel
from legoesm.atmosphere.dynamics.spectral_sw import SpectralShallowWaterModel
from legoesm.atmosphere.dynamics.shallow_water_mpas import MPASShallowWaterModel
from legoesm.parallel.mesh import create_device_mesh, create_level_mesh
from legoesm.parallel.sharded_dynamics import make_sharded_step, shard_state_to_devices, gather_state_from_devices
from legoesm.driver.compiled_segments import build_segment_fn
```

---

# CATEGORY 16: Portability Blockers & Anti-Patterns
**File:** `tests/unit/test_scale_portability.py`

Static analysis tests that scan the codebase for patterns that break scalability.

**16a) No hardcoded device placement**
- Grep for `jax.devices()[0]` or `.device()` in production code (not tests).
- Assert: no explicit device placement outside of `parallel/` module.
- Hardcoded device selection prevents multi-device execution.

**16b) No numpy in hot path**
- Grep for `import numpy` or `np.` in dynamics and physics modules.
- Assert: all numerical operations use `jnp`, not `np`.
- numpy operations are not traceable by JAX (break JIT).

**16c) No Python mutation of arrays**
- Grep for `.at[`, `[...] =`, `__setitem__` in dynamics modules.
- `jnp.ndarray.__setitem__` is not supported in JAX.
- Assert: only `jnp.where`, `.at[].set()`, `.at[].add()` patterns used.

**16d) No global mutable state in physics**
- Grep for `global ` keyword in physics modules.
- Assert: no `global` declarations in hot path code.
- Mutable closures (`_time` dict) are OK if captured at trace time.

**16e) No os.environ reads inside JIT**
- `os.environ` reads inside traced functions cause recompilation.
- Assert: no `os.environ` or `os.getenv` in dynamics/physics modules.

**16f) All lax.cond branches return same structure**
- For each `jax.lax.cond` usage: verify both branches return pytrees with identical treedef.
- This is required for all XLA backends (especially TPU).
- Scan the codebase for `lax.cond` calls and verify.

**16g) No string formatting inside JIT**
- f-strings and `.format()` cause tracing errors.
- Assert: no f-strings in functions called within `jax.jit`.

**16h) Singleton state — documented**
- Count global singletons in `parallel/` module.
- Assert: each singleton has a `set_*()` and `get_*()` accessor.
- Document: these prevent multiple independent simulations in same process.

**Key approach:**
```python
import ast
import pathlib

def scan_module_for_pattern(module_path, pattern_fn):
    """Walk AST of module and check for anti-pattern."""
    source = pathlib.Path(module_path).read_text()
    tree = ast.parse(source)
    violations = []
    for node in ast.walk(tree):
        if pattern_fn(node):
            violations.append((node.lineno, ast.dump(node)))
    return violations
```

---

# Implementation Notes

- **Multi-device tests**: Use `@pytest.mark.skipif(jax.device_count() < N, reason=f"Need {N} devices")` to skip tests that require more devices than available.

- **MPI tests**: Use `@pytest.mark.mpi` and provide instructions to run with `mpirun -np N pytest ...`. Place in `tests/distributed/` directory.

- **Metal tests**: Use `@pytest.mark.skipif(jax.default_backend() != "METAL", reason="Metal only")`.

- **TPU tests**: Most TPU compatibility tests can run on CPU (they check XLA patterns, not TPU hardware). Actual TPU tests need `@pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU only")`.

- **Timing tests**: Use generous thresholds (10×) to avoid flakiness. Mark with `@pytest.mark.slow`.

- **Static analysis tests** (Category 16): Use Python `ast` module or simple `grep` patterns, not runtime tracing.

- **Parametrize** across grid types where applicable:
  ```python
  @pytest.mark.parametrize("n_devices", [1, 2, 3, 6])
  def test_face_mesh(n_devices):
      if jax.device_count() < n_devices:
          pytest.skip(f"Need {n_devices} devices")
      ...
  ```

- **Tolerances**:
  - Single-device vs sharded: 1e-8 (should be exact, but allow for operation reordering).
  - MPI vs serial: 1e-10 (floating-point summation order may differ).
  - Float32 vs float64: 1e-4 (float32 precision).
  - Timing: relative (second call < first / 10).

- Every test must be JAX-compatible: no numpy mutation, use `jnp` throughout.

After writing each file, run it with:
```
JAX_ENABLE_X64=1 python -m pytest tests/unit/test_scale_<name>.py -v --tb=short
```
For MPI tests:
```
JAX_ENABLE_X64=1 mpirun -np 2 python -m pytest tests/distributed/test_scale_<name>.py -v --tb=short
```
Fix import/shape errors. If a test reveals a genuine bug (wrong halo, broken sharding, dtype mismatch), leave it failing with a `# BUG:` comment explaining the suspected issue.
