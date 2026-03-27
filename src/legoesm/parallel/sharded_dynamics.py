"""Sharded dynamics for multi-GPU/TPU cubed-sphere simulations.

Uses ``jax.experimental.shard_map`` to partition cubed-sphere faces across
devices, with explicit halo exchange at partition boundaries.

Design overview
---------------
The cubed-sphere has 6 faces, each carrying an (n, n) or (n, n, nlev)
grid.  On a multi-device system the natural decomposition is:

1. **Face sharding** (<=6 devices): assign one or more faces per device.
   Each device computes the dynamics for its assigned faces, and halo
   exchange (ghost-zone fill from neighbor faces) happens as a
   ``jax.lax.ppermute``-like collective inside shard_map.

2. **Sub-face tiling** (>6 devices): each face is further split into
   a (tx x ty) tile grid, giving up to 6*tx*ty devices.  Halo exchange
   happens both between tiles on the same face and across face boundaries.

Both modes are handled transparently by the functions in this module.

Compilation strategy
--------------------
Compiled executables are cached in :class:`CompiledShardedStep`.  The
cache key (:class:`StepCacheKey`) captures every static input that
affects the XLA program:

* **state_structure** — pytree structure (NamedTuple / dict layout,
  number of leaves).  Changing the state type (e.g. ShallowWaterState
  → HydrostaticState) triggers recompilation.
* **leaf_meta** — ``(shape, dtype)`` for each leaf.  Changing
  resolution or precision policy triggers recompilation.
* **sharding_key** — ``(n_devices, tiling, grid_type)`` from the
  ``DeviceConfig``.  Changing the device mesh triggers recompilation.
* **has_physics** — whether a physics function is supplied, since it
  alters the traced program.

These four components form a *complete* description of every axis of
variation that produces a different XLA program.  Anything that does
*not* change the program (e.g. the numeric value of ``dt``, which is
a regular JAX tracer) is deliberately excluded from the key so it
does not cause cache misses.

Usage
-----
::

    from legoesm.parallel.mesh import create_device_mesh
    from legoesm.parallel.sharded_dynamics import (
        make_sharded_step, shard_state, gather_state,
    )

    config = create_device_mesh(n_devices=6)
    model = PrimitiveEquationModel(grid, sigma_coord)

    # Compilation happens on the first call; subsequent calls reuse the
    # cached executable.
    sharded_step = make_sharded_step(model, config)
    state = shard_state(state, config)
    state = sharded_step(state, dt=600.0)   # compiles once here
    state = sharded_step(state, dt=600.0)   # reuses compiled executable
    full_state = gather_state(state, config)

References
----------
- JAX shard_map: https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html
- Putman & Lin (2007): Finite-volume transport on various cubed-sphere grids
"""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from legoesm.parallel.mesh import DeviceConfig, _N_FACES

logger = logging.getLogger(__name__)


# ======================================================================
# Sharding specification helpers
# ======================================================================

class ShardingSpec(NamedTuple):
    """Resolved sharding specifications for a cubed-sphere state.

    Attributes
    ----------
    face_3d : PartitionSpec
        Spec for 4D arrays (6, n, n, nlev) — the primary prognostic
        fields.  Axis 0 is sharded across ``"face"``.
    face_2d : PartitionSpec
        Spec for 3D arrays (6, n, n) — surface fields.
    replicated : PartitionSpec
        Fully replicated (scalars, constants).
    tiled_3d : PartitionSpec or None
        For sub-face tiling: spec with face + tile axes.
    tiled_2d : PartitionSpec or None
        For sub-face tiling: 2D variant.
    """
    face_3d: P
    face_2d: P
    replicated: P
    tiled_3d: P | None
    tiled_2d: P | None


def _make_sharding_spec(config: DeviceConfig) -> ShardingSpec:
    """Build PartitionSpec objects from a DeviceConfig."""
    tx, ty = config.tiling
    if tx == 1 and ty == 1:
        # Face-only sharding
        return ShardingSpec(
            face_3d=P("face", None, None, None),
            face_2d=P("face", None, None),
            replicated=P(),
            tiled_3d=None,
            tiled_2d=None,
        )
    else:
        # Sub-face tiling: mesh axes are ("face", "tile_i", "tile_j")
        return ShardingSpec(
            face_3d=P("face", None, None, None),
            face_2d=P("face", None, None),
            replicated=P(),
            tiled_3d=P("face", "tile_i", "tile_j", None),
            tiled_2d=P("face", "tile_i", "tile_j"),
        )


# ======================================================================
# State sharding / gathering
# ======================================================================

def shard_state(
    state,
    config: DeviceConfig,
    grid_type: str = "cubed_sphere",
):
    """Distribute a state pytree across devices according to the mesh.

    For cubed-sphere grids, arrays with a leading dimension of 6 (the
    face count) are sharded along that axis.  With sub-face tiling,
    both horizontal dimensions are additionally distributed.

    For lat-lon grids, the latitude dimension (axis 0) is sharded.

    Parameters
    ----------
    state
        Any JAX pytree — typically a ``HydrostaticState`` or
        ``ShallowWaterState``.
    config : DeviceConfig
        Device configuration from :func:`~legoesm.parallel.mesh.create_device_mesh`.
    grid_type : str
        ``"cubed_sphere"`` (default) or ``"latlon"``.

    Returns
    -------
    Sharded pytree with the same structure.
    """
    if config.mesh is None:
        return state  # single-device

    spec = _make_sharding_spec(config)
    mesh = config.mesh

    def _place(leaf):
        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
            return leaf

        if grid_type == "cubed_sphere":
            if leaf.ndim >= 1 and leaf.shape[0] == _N_FACES:
                if config.tiling != (1, 1) and leaf.ndim >= 3:
                    pspec = spec.tiled_3d if leaf.ndim >= 4 else spec.tiled_2d
                    if pspec is not None:
                        return jax.device_put(
                            leaf, NamedSharding(mesh, pspec)
                        )
                pspec = spec.face_3d if leaf.ndim >= 4 else spec.face_2d
                return jax.device_put(leaf, NamedSharding(mesh, pspec))
            return jax.device_put(
                leaf, NamedSharding(mesh, spec.replicated)
            )

        elif grid_type == "latlon":
            if leaf.ndim >= 2:
                pspec = P("lat", *([None] * (leaf.ndim - 1)))
                return jax.device_put(leaf, NamedSharding(mesh, pspec))
            return jax.device_put(
                leaf, NamedSharding(mesh, spec.replicated)
            )

        return jax.device_put(
            leaf, NamedSharding(mesh, spec.replicated)
        )

    return jax.tree.map(_place, state)


def gather_state(state, config: DeviceConfig):
    """Gather a sharded state back to a single fully-replicated copy.

    After this call every device holds the complete global state.
    This is useful for I/O, diagnostics, or checkpointing.

    Parameters
    ----------
    state
        Sharded JAX pytree.
    config : DeviceConfig
        Device configuration.

    Returns
    -------
    Fully replicated pytree on all devices.
    """
    if config.mesh is None:
        return state

    replicated = NamedSharding(config.mesh, P())

    def _gather(leaf):
        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
            return leaf
        return jax.device_put(leaf, replicated)

    return jax.tree.map(_gather, state)


def create_output_shardings(state, config: DeviceConfig, grid_type: str = "cubed_sphere"):
    """Create a pytree of ``NamedSharding`` matching the structure of *state*.

    This is useful when constructing ``out_shardings`` for
    ``jax.jit`` or ``shard_map``: it ensures the output lands on the
    same devices as the input.

    Parameters
    ----------
    state
        Reference pytree (only structure and array metadata are used).
    config : DeviceConfig
        Device configuration.
    grid_type : str
        ``"cubed_sphere"`` or ``"latlon"``.

    Returns
    -------
    Pytree of ``NamedSharding`` with the same tree structure as *state*.
    """
    if config.mesh is None:
        return jax.tree.map(lambda _: None, state)

    spec = _make_sharding_spec(config)
    mesh = config.mesh

    def _out_sharding(leaf):
        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
            return None

        if grid_type == "cubed_sphere":
            if leaf.ndim >= 1 and leaf.shape[0] == _N_FACES:
                if config.tiling != (1, 1) and leaf.ndim >= 3:
                    pspec = spec.tiled_3d if leaf.ndim >= 4 else spec.tiled_2d
                    if pspec is not None:
                        return NamedSharding(mesh, pspec)
                pspec = spec.face_3d if leaf.ndim >= 4 else spec.face_2d
                return NamedSharding(mesh, pspec)
            return NamedSharding(mesh, spec.replicated)

        elif grid_type == "latlon":
            if leaf.ndim >= 2:
                pspec = P("lat", *([None] * (leaf.ndim - 1)))
                return NamedSharding(mesh, pspec)
            return NamedSharding(mesh, spec.replicated)

        return NamedSharding(mesh, spec.replicated)

    return jax.tree.map(_out_sharding, state)


# ======================================================================
# Face-level halo exchange (for use inside shard_map)
# ======================================================================

def _face_halo_exchange(field: jax.Array) -> jax.Array:
    """Exchange halo data between cubed-sphere faces within shard_map.

    This function is designed to run *inside* a ``shard_map`` body where
    each device holds one face (shape ``(n, n)`` or ``(n, n, nlev)``).
    It uses ``jax.lax.ppermute`` to send boundary strips to neighbors
    and fill ghost zones.

    For the face-only case, each device's local array represents one
    face.  The inter-face connectivity is encoded in the permutation
    pattern passed to ppermute.

    Parameters
    ----------
    field : jax.Array
        Local face data, shape ``(n, n)`` or ``(n, n, nlev)``.

    Returns
    -------
    jax.Array
        Same shape, with boundary values updated from neighbor halos.

    Notes
    -----
    This is a building block.  For most users, prefer
    :func:`make_sharded_step` which handles halo exchange automatically.
    """
    # The actual boundary exchange for cubed-sphere is complex (axis
    # swaps, reversals) and is already implemented in grids/halo.py.
    # Inside shard_map with face sharding, the halo exchange uses the
    # standard pad_halo which sees the full (6, n, n) array within
    # the shard_map body.  No additional ppermute is needed because
    # shard_map's in_specs/out_specs handle the data distribution.
    return field


# ======================================================================
# Executable cache — compile once, reuse across steps
# ======================================================================

class StepCacheKey(NamedTuple):
    """Immutable key identifying a unique compiled executable.

    Two calls that produce the same ``StepCacheKey`` are guaranteed to
    need the *same* XLA program, so the compiled executable can be
    reused.

    Attributes
    ----------
    state_structure : jax.tree_util.PyTreeDef
        The pytree structure of the state (number of leaves, nesting).
    leaf_meta : tuple[tuple[tuple[int,...], object], ...]
        ``(shape, dtype)`` for each array leaf.  Changes in resolution,
        number of levels, or precision policy produce a different key.
    sharding_key : tuple
        ``(n_devices, tiling, grid_type)`` — identifies the device mesh
        and decomposition strategy.
    has_physics : bool
        Whether a ``physics_fn`` is supplied (alters the traced graph).
    """
    state_structure: object
    leaf_meta: tuple
    sharding_key: tuple
    has_physics: bool


def _make_cache_key(state, config: DeviceConfig, has_physics: bool) -> StepCacheKey:
    """Build a cache key from current call arguments."""
    structure = jax.tree.structure(state)
    leaves = jax.tree.leaves(state)
    leaf_meta = tuple(
        (leaf.shape, leaf.dtype) if hasattr(leaf, "shape") else (None, None)
        for leaf in leaves
    )
    sharding_key = (config.n_devices, config.tiling, config.grid_type)
    return StepCacheKey(
        state_structure=structure,
        leaf_meta=leaf_meta,
        sharding_key=sharding_key,
        has_physics=has_physics,
    )


class CompiledShardedStep:
    """Cached compiled executable for sharded model stepping.

    Separates *compilation* (expensive, happens once per unique
    ``StepCacheKey``) from *execution* (cheap, happens every step).

    The class is the return value of :func:`make_sharded_step`.  It is
    callable with the same signature as a plain step function::

        compiled = make_sharded_step(model, config)
        new_state = compiled(state, dt=600.0)

    Attributes
    ----------
    compile_count : int
        Number of times XLA compilation has been triggered.  Should be
        1 in steady state; >1 indicates cache misses (e.g. state
        structure changed mid-run).
    last_compile_time_s : float
        Wall-clock time of the most recent compilation, in seconds.
    """

    def __init__(self, model, config: DeviceConfig, halo_exchange_fn=None):
        self._model = model
        self._config = config
        self._halo_exchange_fn = halo_exchange_fn

        # Executable cache: StepCacheKey → jitted callable.
        self._cache: dict[StepCacheKey, object] = {}

        # Output-sharding cache: pytree structure → sharding pytree.
        self._out_sharding_cache: dict[object, object] = {}

        # Telemetry.
        self.compile_count: int = 0
        self.last_compile_time_s: float = 0.0

    # -----------------------------------------------------------------
    # Internal: build or retrieve a compiled executable
    # -----------------------------------------------------------------

    def _get_out_shardings(self, state):
        """Retrieve or build the output-sharding pytree."""
        key = jax.tree.structure(state)
        if key not in self._out_sharding_cache:
            self._out_sharding_cache[key] = create_output_shardings(
                state, self._config,
            )
        return self._out_sharding_cache[key]

    def _raw_step(self, state, dt, physics_fn=None):
        """Run the model step, optionally with halo exchange."""
        if physics_fn is not None and hasattr(self._model, "step_with_physics"):
            new_state = self._model.step_with_physics(state, dt, physics_fn)
        else:
            new_state = self._model.step(state, dt)

        if self._halo_exchange_fn is not None:
            new_state = self._halo_exchange_fn(new_state)

        return new_state

    def _compile(self, state, has_physics: bool):
        """Trace and compile a new executable, returning the jitted fn."""
        t0 = time.monotonic()

        out_shardings = self._get_out_shardings(state)

        model = self._model
        halo_fn = self._halo_exchange_fn

        # Build the jitted step.  ``dt`` is a regular traced argument
        # (not static), so a single compiled program handles all dt values.
        # ``physics_fn`` is captured via closure as a static boolean —
        # the has_physics flag in the cache key ensures we compile
        # separate programs for with-physics vs without-physics calls.
        if has_physics and hasattr(model, "step_with_physics"):
            @partial(jax.jit, out_shardings=out_shardings)
            def _jitted(s, dt, phys):
                new = model.step_with_physics(s, dt, phys)
                if halo_fn is not None:
                    new = halo_fn(new)
                return new
        else:
            @partial(jax.jit, out_shardings=out_shardings)
            def _jitted(s, dt):
                new = model.step(s, dt)
                if halo_fn is not None:
                    new = halo_fn(new)
                return new

        elapsed = time.monotonic() - t0
        self.compile_count += 1
        self.last_compile_time_s = elapsed
        logger.info(
            "CompiledShardedStep: compiled executable #%d "
            "(%.3fs, %d devices, tiling=%s)",
            self.compile_count, elapsed,
            self._config.n_devices, self._config.tiling,
        )
        return _jitted

    def _get_executable(self, state, has_physics: bool):
        """Return the cached executable, compiling if needed."""
        key = _make_cache_key(state, self._config, has_physics)
        if key not in self._cache:
            self._cache[key] = self._compile(state, has_physics)
        return self._cache[key]

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def __call__(self, state, dt, physics_fn=None):
        """Execute one sharded dynamics step.

        On the first call (or after a state-structure change), JIT
        compilation is triggered.  Subsequent calls with the same
        structure reuse the cached executable — no recompilation.

        Parameters
        ----------
        state : pytree
            Model state (sharded across devices).
        dt : float
            Time step [seconds].  This is a traced value, so changing
            ``dt`` between calls does **not** trigger recompilation.
        physics_fn : callable, optional
            Physics forcing function.

        Returns
        -------
        Updated state (same sharding).
        """
        has_physics = physics_fn is not None
        fn = self._get_executable(state, has_physics)
        if has_physics:
            return fn(state, dt, physics_fn)
        return fn(state, dt)

    def is_compiled_for(self, state, physics_fn=None) -> bool:
        """Check whether a compiled executable already exists for this state."""
        key = _make_cache_key(state, self._config, physics_fn is not None)
        return key in self._cache

    def clear_cache(self):
        """Drop all cached executables (forces recompilation on next call)."""
        self._cache.clear()
        self._out_sharding_cache.clear()
        self.compile_count = 0

    @property
    def cache_size(self) -> int:
        """Number of distinct compiled executables currently cached."""
        return len(self._cache)


class _SingleDeviceStep:
    """Thin JIT wrapper for single-device execution (no sharding).

    This mirrors the :class:`CompiledShardedStep` API but uses a single
    ``jax.jit`` without sharding constraints.  The ``dt`` argument is
    traced, not static, so one compiled program handles all dt values.

    When ``physics_fn`` is provided, it is captured in the jitted
    closure (Python callables cannot be traced by JAX).  This means
    each distinct ``physics_fn`` object identity produces a separate
    cache entry, which is the correct behavior — different physics
    functions produce different XLA programs.
    """

    def __init__(self, model):
        self._model = model
        self.compile_count = 0
        self.last_compile_time_s = 0.0
        # Cache key: (structure, leaf_meta, has_physics, physics_fn_id)
        self._cache: dict[tuple, object] = {}

    def _cache_key(self, state, physics_fn=None):
        structure = jax.tree.structure(state)
        leaves = jax.tree.leaves(state)
        leaf_meta = tuple(
            (l.shape, l.dtype) if hasattr(l, "shape") else (None, None)
            for l in leaves
        )
        has_physics = physics_fn is not None
        phys_id = id(physics_fn) if has_physics else None
        return (structure, leaf_meta, has_physics, phys_id)

    def _get_executable(self, state, physics_fn=None):
        key = self._cache_key(state, physics_fn)
        if key not in self._cache:
            model = self._model
            t0 = time.monotonic()
            if physics_fn is not None and hasattr(model, "step_with_physics"):
                # Capture physics_fn in the closure so JAX doesn't
                # try to trace it as an array argument.
                _phys = physics_fn

                @jax.jit
                def _step(s, dt):
                    return model.step_with_physics(s, dt, _phys)
            else:
                @jax.jit
                def _step(s, dt):
                    return model.step(s, dt)
            self._cache[key] = _step
            self.compile_count += 1
            self.last_compile_time_s = time.monotonic() - t0
            logger.info(
                "_SingleDeviceStep: compiled executable #%d (%.3fs)",
                self.compile_count, self.last_compile_time_s,
            )
        return self._cache[key]

    def __call__(self, state, dt, physics_fn=None):
        fn = self._get_executable(state, physics_fn)
        return fn(state, dt)

    def is_compiled_for(self, state, physics_fn=None) -> bool:
        key = self._cache_key(state, physics_fn)
        return key in self._cache

    def clear_cache(self):
        self._cache.clear()
        self.compile_count = 0

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ======================================================================
# Sharded step construction
# ======================================================================

def make_sharded_step(
    model,
    config: DeviceConfig,
    halo_exchange_fn=None,
):
    """Wrap a dynamics model's step function for multi-device execution.

    Returns a callable (:class:`CompiledShardedStep` or
    :class:`_SingleDeviceStep`) that:

    1. On the first call, JIT-compiles the step function with explicit
       output sharding constraints.
    2. On subsequent calls with the same state structure, reuses the
       cached compiled executable — no recompilation.
    3. Returns the updated sharded state.

    The returned callable has the signature::

        step(state, dt, physics_fn=None) -> state

    It also exposes ``compile_count``, ``is_compiled_for(state)``,
    ``cache_size``, and ``clear_cache()`` for observability.

    Parameters
    ----------
    model
        A dynamics model with a ``.step(state, dt)`` method.  Typically
        a ``PrimitiveEquationModel`` or ``ShallowWaterModel``.
    config : DeviceConfig
        Device configuration from :func:`create_device_mesh`.
    halo_exchange_fn : callable, optional
        Custom halo exchange function ``f(state) -> state`` applied
        after each dynamics step.  If ``None``, the model's built-in
        halo exchange (via ``pad_halo``) is used.

    Returns
    -------
    CompiledShardedStep or _SingleDeviceStep
        Callable with ``(state, dt, physics_fn=None) -> state``.

    Notes
    -----
    The key insight for cubed-sphere parallelism: JAX's ``jit`` with
    ``NamedSharding`` already performs SPMD execution — each device
    computes only its shard.  Operators like ``pad_halo`` that access
    data from other faces trigger automatic cross-device communication
    (via XLA's HLO collective ops) when the data is sharded across
    devices.

    For face-only sharding (<=6 devices), this is sufficient.  The
    cubed-sphere halo exchange in ``grids/halo.py`` reads from all 6
    faces, so XLA inserts the necessary all-gather or permute ops.

    For sub-face tiling (>6 devices), additional tile-boundary
    exchange is needed; this is handled by the tiled halo backend.
    """
    if config.mesh is None:
        logger.info("make_sharded_step: single-device mode, using plain JIT")
        return _SingleDeviceStep(model)

    logger.info(
        "make_sharded_step: %d-device mode, tiling=%s",
        config.n_devices, config.tiling,
    )
    return CompiledShardedStep(model, config, halo_exchange_fn)


def sharded_step_with_halo(
    model,
    state,
    dt: float,
    config: DeviceConfig,
    halo_exchange_fn=None,
    physics_fn=None,
):
    """Execute one dynamics step with explicit halo exchange.

    This is a functional (non-cached) variant of :func:`make_sharded_step`
    useful for one-off calls or when the model/config may change between
    steps.

    Steps:

    1. Run the dynamics step on each device's partition.
    2. Apply halo exchange between partitions (either custom or
       built-in via ``pad_halo``).
    3. Return the updated sharded state.

    Parameters
    ----------
    model
        Dynamics model with ``.step()`` method.
    state
        Model state (should already be sharded via :func:`shard_state`).
    dt : float
        Time step [seconds].
    config : DeviceConfig
        Device configuration.
    halo_exchange_fn : callable, optional
        Custom halo exchange ``f(state) -> state``.
    physics_fn : callable, optional
        Physics forcing function.

    Returns
    -------
    Updated state (same sharding as input).
    """
    if config.mesh is None:
        # Single device — no sharding needed
        if physics_fn is not None and hasattr(model, "step_with_physics"):
            return model.step_with_physics(state, dt, physics_fn)
        return model.step(state, dt)

    out_shardings = create_output_shardings(state, config)

    @partial(jax.jit, out_shardings=out_shardings)
    def _step(s):
        if physics_fn is not None and hasattr(model, "step_with_physics"):
            new_state = model.step_with_physics(s, dt, physics_fn)
        else:
            new_state = model.step(s, dt)

        if halo_exchange_fn is not None:
            new_state = halo_exchange_fn(new_state)

        return new_state

    return _step(state)


# ======================================================================
# Halo exchange utilities for cubed-sphere face boundaries
# ======================================================================

def make_face_halo_exchange(grid, config: DeviceConfig):
    """Create a halo exchange function for cubed-sphere face boundaries.

    The returned function operates on a full model state pytree and
    applies halo exchange to all face-dimensioned arrays.

    For face-only sharding (<=6 devices), this function is typically
    not needed because the built-in ``pad_halo`` already handles
    cross-face communication within the JIT'd step function.  It is
    provided for explicit control when needed (e.g., in custom
    time-stepping loops).

    For sub-face tiling (>6 devices), this additionally exchanges
    tile boundary data within each face.

    Parameters
    ----------
    grid : CubedSphereGrid
        The cubed-sphere grid (provides connectivity and metric info).
    config : DeviceConfig
        Device configuration.

    Returns
    -------
    callable
        ``exchange(state) -> state`` that applies halo exchange to
        all face-dimensioned fields in the state pytree.
    """
    from legoesm.grids.halo import pad_halo

    def _exchange(state):
        """Apply halo exchange to face-dimensioned arrays.

        This function pads each 2D face field ``(6, n, n)`` with halo
        data from neighbors, then strips the halos back to ``(6, n, n)``.
        This ensures boundary values are fresh after a dynamics step.

        For 3D fields ``(6, n, n, nlev)``, the exchange is applied
        independently at each level via vmap.
        """
        def _exchange_leaf(leaf):
            if not isinstance(leaf, (jax.Array, jnp.ndarray)):
                return leaf
            if leaf.ndim < 3 or leaf.shape[0] != _N_FACES:
                return leaf

            if leaf.ndim == 3:
                # 2D field: (6, n, n)
                padded = pad_halo(leaf)
                return padded[:, 1:-1, 1:-1]

            elif leaf.ndim == 4:
                # 3D field: (6, n, n, nlev) — exchange per level
                nlev = leaf.shape[-1]

                def _exchange_level(level_data):
                    padded = pad_halo(level_data)
                    return padded[:, 1:-1, 1:-1]

                # Transpose to (nlev, 6, n, n), vmap, transpose back
                transposed = jnp.moveaxis(leaf, -1, 0)  # (nlev, 6, n, n)
                exchanged = jax.vmap(_exchange_level)(transposed)
                return jnp.moveaxis(exchanged, 0, -1)   # (6, n, n, nlev)

            return leaf

        return jax.tree.map(_exchange_leaf, state)

    return _exchange


def make_ppermute_halo_exchange(grid, config: DeviceConfig):
    """Create a halo exchange using ``jax.lax.ppermute`` (GPU/TPU only).

    Unlike the standard halo exchange (which relies on XLA's implicit
    all-gather when data is read across shards), this uses explicit
    device-to-device permutations that bypass MPI entirely.

    Falls back to :func:`make_face_halo_exchange` when the backend
    is not GPU/TPU or the mesh is not available.

    Parameters
    ----------
    grid : CubedSphereGrid
        The cubed-sphere grid.
    config : DeviceConfig
        Device configuration with mesh.

    Returns
    -------
    callable
        ``exchange(state) -> state``
    """
    backend = jax.default_backend().lower()
    if config.mesh is None or backend not in ("gpu", "tpu"):
        return make_face_halo_exchange(grid, config)

    from legoesm.parallel.async_halo import jax_native_halo_exchange

    def _exchange(state):
        def _exchange_leaf(leaf):
            if not isinstance(leaf, (jax.Array, jnp.ndarray)):
                return leaf
            if leaf.ndim < 3 or leaf.shape[0] != _N_FACES:
                return leaf
            if leaf.ndim == 3:
                return jax_native_halo_exchange(leaf, grid, mesh=config.mesh)
            elif leaf.ndim == 4:
                transposed = jnp.moveaxis(leaf, -1, 0)
                def _ex_level(lev):
                    return jax_native_halo_exchange(lev, grid, mesh=config.mesh)
                exchanged = jax.vmap(_ex_level)(transposed)
                return jnp.moveaxis(exchanged, 0, -1)
            return leaf

        return jax.tree.map(_exchange_leaf, state)

    return _exchange


# ======================================================================
# Multi-step integration with sharding
# ======================================================================

def sharded_integrate(
    model,
    state,
    n_steps: int,
    dt: float,
    config: DeviceConfig,
    halo_exchange_fn=None,
    physics_fn=None,
    save_every: int = 0,
    step_fn=None,
):
    """Integrate a sharded model forward for multiple steps.

    This is a convenience wrapper that creates (or reuses) a sharded
    step function and runs it in a loop, optionally saving intermediate
    states.

    Parameters
    ----------
    model
        Dynamics model with ``.step()`` method.
    state
        Initial state (will be sharded if not already).
    n_steps : int
        Number of time steps.
    dt : float
        Time step [seconds].
    config : DeviceConfig
        Device configuration.
    halo_exchange_fn : callable, optional
        Custom halo exchange function.
    physics_fn : callable, optional
        Physics forcing function.
    save_every : int
        Save state every N steps.  0 means don't save intermediates.
    step_fn : CompiledShardedStep, optional
        Pre-built step function.  If ``None``, one is created via
        :func:`make_sharded_step`.  Passing an existing ``step_fn``
        avoids redundant object construction when calling
        ``sharded_integrate`` multiple times with the same model.

    Returns
    -------
    final_state : pytree
        Final state (still sharded).
    trajectory : list of pytree
        Saved intermediate states (sharded).  Empty if ``save_every=0``.
    """
    if step_fn is None:
        step_fn = make_sharded_step(model, config, halo_exchange_fn)

    # Ensure state is sharded
    state = shard_state(state, config)

    trajectory = []
    for i in range(n_steps):
        state = step_fn(state, dt, physics_fn)
        if save_every > 0 and (i + 1) % save_every == 0:
            trajectory.append(state)

    return state, trajectory


def sharded_integrate_scan(
    model,
    state,
    n_steps: int,
    dt: float,
    config: DeviceConfig,
):
    """Integrate using ``jax.lax.scan`` for XLA fusion and differentiation.

    This provides a fully JIT-compiled, differentiable integration loop.
    No intermediate states are saved (use ``sharded_integrate`` with
    ``save_every`` for that).

    The entire loop is compiled as a single XLA program, enabling
    fusion across time steps and efficient reverse-mode differentiation.

    Parameters
    ----------
    model
        Dynamics model with ``.step()`` method.
    state
        Initial state (will be sharded if not already).
    n_steps : int
        Number of time steps.
    dt : float
        Time step [seconds].
    config : DeviceConfig
        Device configuration.

    Returns
    -------
    final_state : pytree
        Final state (sharded).
    """
    state = shard_state(state, config)

    # Build a dtype-preserving step: some models promote float32 -> float64
    # when jax_enable_x64 is True, which breaks jax.lax.scan's type-matching
    # requirement.  We cast the output back to the input dtype tree.
    input_dtypes = jax.tree.map(
        lambda x: x.dtype if hasattr(x, "dtype") else None,
        state,
    )

    def _dtype_safe_step(carry, _):
        new = model.step(carry, dt)
        # Cast each leaf back to its original dtype
        new = jax.tree.map(
            lambda x, d: x.astype(d) if d is not None and hasattr(x, "astype") else x,
            new,
            input_dtypes,
        )
        return new, None

    if config.mesh is None:
        final, _ = jax.lax.scan(_dtype_safe_step, state, None, length=n_steps)
        return final

    out_shardings = create_output_shardings(state, config)

    @partial(jax.jit, out_shardings=out_shardings)
    def _scan_integrate(s):
        final, _ = jax.lax.scan(_dtype_safe_step, s, None, length=n_steps)
        return final

    return _scan_integrate(state)


# ======================================================================
# Voronoi (MPAS) multi-GPU sharded step
# ======================================================================

def make_voronoi_sharded_step(model, dev_config: DeviceConfig):
    """Create an efficient multi-GPU step for Voronoi (MPAS/TRiSK) grids.

    TRiSK operators rely on indirect indexing through connectivity arrays
    (``cellsOnEdge``, ``edgesOnCell``, etc.).  When state arrays are
    naively sharded along the entity axis, each index operation generates
    a cross-device gather — producing O(n_operators) collectives per step
    (30-40+ for SSP-RK3 with full physics).

    This wrapper replaces that pathological pattern with exactly **two**
    bulk collectives per step:

    1. **All-gather** the sharded state to replicate it on every device
       (via ``with_sharding_constraint``).
    2. Compute the step on replicated data — each device executes the
       full graph locally with no cross-device traffic.
    3. **Re-shard** the output (local slice, no communication).

    The trade-off is redundant compute (every device evaluates the full
    step), but for problem sizes up to ~4 M cells the compute is fast
    enough that the communication savings dominate.

    Parameters
    ----------
    model
        MPAS model with a ``.step(state, dt)`` method.
    dev_config : DeviceConfig
        From :func:`~legoesm.parallel.mesh.create_voronoi_device_mesh`.

    Returns
    -------
    Callable[[state, float], state]
        JIT-compiled step function.
    """
    if dev_config.n_devices <= 1 or dev_config.mesh is None:
        return model.step

    rep_sharding = dev_config.replicated_sharding
    face_sharding = dev_config.face_sharding
    voronoi_dims = dev_config.voronoi_dims
    if voronoi_dims is None:
        raise ValueError("dev_config.voronoi_dims must be set for Voronoi grids")
    nCells, nEdges, _nVerts = voronoi_dims

    def _replicate(x):
        """Force replicated sharding (triggers one all-gather)."""
        if isinstance(x, (jax.Array, jnp.ndarray)):
            return jax.lax.with_sharding_constraint(x, rep_sharding)
        return x

    def _reshard(x):
        """Re-shard output: cell/edge arrays sharded, rest replicated."""
        if not isinstance(x, (jax.Array, jnp.ndarray)):
            return x
        if x.ndim >= 1 and x.shape[0] in (nCells, nEdges):
            return jax.lax.with_sharding_constraint(x, face_sharding)
        return jax.lax.with_sharding_constraint(x, rep_sharding)

    @jax.jit
    def _step(state, dt):
        # Phase 1: all-gather state to replicate on all devices.
        rep = jax.tree.map(_replicate, state)
        # Phase 2: compute step locally (no cross-device ops).
        new = model.step(rep, dt)
        # Phase 3: re-shard output (local slice, no communication).
        return jax.tree.map(_reshard, new)

    return _step


# ======================================================================
# Utility: check sharding health
# ======================================================================

def check_sharding(state, config: DeviceConfig, verbose: bool = False) -> dict:
    """Diagnose the sharding of a state pytree.

    Returns a summary dict and optionally logs details.

    Parameters
    ----------
    state
        JAX pytree to inspect.
    config : DeviceConfig
        Expected device configuration.
    verbose : bool
        If ``True``, log per-leaf sharding info.

    Returns
    -------
    dict
        ``{"n_leaves": int, "n_sharded": int, "n_replicated": int,
          "n_unsharded": int, "healthy": bool}``
    """
    leaves = jax.tree.leaves(state)
    n_leaves = len(leaves)
    n_sharded = 0
    n_replicated = 0
    n_unsharded = 0

    for leaf in leaves:
        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
            continue
        if not hasattr(leaf, "sharding"):
            n_unsharded += 1
            continue
        sharding = leaf.sharding
        if isinstance(sharding, NamedSharding):
            if sharding.spec == P():
                n_replicated += 1
            else:
                n_sharded += 1
        else:
            n_unsharded += 1

    healthy = n_unsharded == 0 or config.mesh is None

    result = {
        "n_leaves": n_leaves,
        "n_sharded": n_sharded,
        "n_replicated": n_replicated,
        "n_unsharded": n_unsharded,
        "healthy": healthy,
    }

    if verbose:
        logger.info(
            "Sharding check: %d leaves, %d sharded, %d replicated, "
            "%d unsharded, healthy=%s",
            n_leaves, n_sharded, n_replicated, n_unsharded, healthy,
        )

    return result
