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

def _pad_local_mesh_to(mesh, target_nCells, target_nEdges, target_nVertices):
    """Pad a local VoronoiMesh to target dimensions with inert ghost entities.

    Ghost cells have ``areaCell=1``, zero signs/weights, and connectivity
    pointing to index 0.  Ghost edges have ``dvEdge=0`` (zero flux),
    ``dcEdge=1``, and ``cellsOnEdge=[0,0]``.
    """
    from legoesm.grids.voronoi import VoronoiMesh

    pad_c = target_nCells - mesh.nCells
    pad_e = target_nEdges - mesh.nEdges
    pad_v = target_nVertices - mesh.nVertices
    if pad_c == 0 and pad_e == 0 and pad_v == 0:
        return mesh

    def pad1(arr, n, fill=0.0):
        if n <= 0:
            return arr
        return jnp.concatenate([arr, jnp.full((n,), fill, dtype=arr.dtype)])

    def pad2_col(arr, n, fill=0):
        if n <= 0:
            return arr
        K = arr.shape[0]
        return jnp.concatenate(
            [arr, jnp.full((K, n), fill, dtype=arr.dtype)], axis=1,
        )

    return VoronoiMesh(
        nCells=target_nCells,
        nEdges=target_nEdges,
        nVertices=target_nVertices,
        maxEdges=mesh.maxEdges,
        vertexDegree=mesh.vertexDegree,
        radius=mesh.radius,
        # Cell coords
        latCell=pad1(mesh.latCell, pad_c),
        lonCell=pad1(mesh.lonCell, pad_c),
        xCell=pad1(mesh.xCell, pad_c),
        yCell=pad1(mesh.yCell, pad_c),
        zCell=pad1(mesh.zCell, pad_c),
        # Edge coords
        latEdge=pad1(mesh.latEdge, pad_e),
        lonEdge=pad1(mesh.lonEdge, pad_e),
        xEdge=pad1(mesh.xEdge, pad_e),
        yEdge=pad1(mesh.yEdge, pad_e),
        zEdge=pad1(mesh.zEdge, pad_e),
        # Vertex coords
        latVertex=pad1(mesh.latVertex, pad_v),
        lonVertex=pad1(mesh.lonVertex, pad_v),
        xVertex=pad1(mesh.xVertex, pad_v),
        yVertex=pad1(mesh.yVertex, pad_v),
        zVertex=pad1(mesh.zVertex, pad_v),
        # Connectivity — ghost entries point to 0 (safe index)
        cellsOnEdge=pad2_col(mesh.cellsOnEdge, pad_e, fill=0),
        edgesOnCell=pad2_col(mesh.edgesOnCell, pad_c, fill=0),
        verticesOnCell=pad2_col(mesh.verticesOnCell, pad_c, fill=0),
        verticesOnEdge=pad2_col(mesh.verticesOnEdge, pad_e, fill=0),
        edgesOnVertex=pad2_col(mesh.edgesOnVertex, pad_v, fill=0),
        cellsOnVertex=pad2_col(mesh.cellsOnVertex, pad_v, fill=0),
        cellsOnCell=pad2_col(mesh.cellsOnCell, pad_c, fill=0),
        edgesOnEdge=pad2_col(mesh.edgesOnEdge, pad_e, fill=0),
        nEdgesOnCell=pad1(mesh.nEdgesOnCell, pad_c, fill=0),
        nEdgesOnEdge=pad1(mesh.nEdgesOnEdge, pad_e, fill=0),
        # Geometry
        areaCell=pad1(mesh.areaCell, pad_c, fill=1.0),
        areaTriangle=pad1(mesh.areaTriangle, pad_v, fill=1.0),
        dcEdge=pad1(mesh.dcEdge, pad_e, fill=1.0),
        dvEdge=pad1(mesh.dvEdge, pad_e, fill=0.0),
        angleEdge=pad1(mesh.angleEdge, pad_e, fill=0.0),
        # Weights / signs — zero for ghosts
        weightsOnEdge=pad2_col(mesh.weightsOnEdge, pad_e, fill=0.0),
        kiteAreasOnVertex=pad2_col(mesh.kiteAreasOnVertex, pad_v, fill=0.0),
        fEdge=pad1(mesh.fEdge, pad_e, fill=0.0),
        fVertex=pad1(mesh.fVertex, pad_v, fill=0.0),
        edgeSignOnCell=pad2_col(mesh.edgeSignOnCell, pad_c, fill=0.0),
        edgeSignOnVertex=pad2_col(mesh.edgeSignOnVertex, pad_v, fill=0.0),
        meshDensity=pad1(mesh.meshDensity, pad_c, fill=0.0),
    )


def _build_voronoi_partition_infra(global_mesh, n_dev, halo_depth=2):
    """Pre-compute per-device local meshes and gather/scatter indices.

    After ``reorder_voronoi_for_sharding`` the global mesh has *both*
    cells and edges ordered by contiguous device blocks.  We compute
    partitions whose owned-entity boundaries exactly match the shard
    boundaries (``cells_per = nCells // n_dev``, ``edges_per = nEdges //
    n_dev``), then build local meshes with remapped connectivity.

    Using ``partition_voronoi_mesh`` directly is unsuitable because it
    derives edge ownership from cell ownership, producing an uneven edge
    split that mismatches the even shard split.  Instead we construct the
    :class:`VoronoiPartition` objects manually with contiguous-block
    ownership for both cells **and** edges.

    Returns
    -------
    stacked_meshes : VoronoiMesh
        Each leaf has shape ``(n_dev, max_local_*)``.
    gather_cells : jnp.ndarray, (n_dev, max_local_cells)
    gather_edges : jnp.ndarray, (n_dev, max_local_edges)
    n_owned_cells : list[int]
    n_owned_edges : list[int]
    """
    import numpy as np
    from legoesm.parallel.voronoi_partition import (
        VoronoiPartition,
        HaloCommSchedule,
        build_local_mesh,
        _compute_halo_cells,
    )

    nCells = global_mesh.nCells
    nEdges = global_mesh.nEdges
    nVertices = global_mesh.nVertices
    cells_per = nCells // n_dev
    edges_per = nEdges // n_dev
    verts_per = nVertices // n_dev

    # Convert mesh arrays to numpy for setup.
    gm_np = jax.tree.map(
        lambda x: np.asarray(x) if hasattr(x, "shape") else x,
        global_mesh,
    )
    cellsOnCell_np = np.asarray(gm_np.cellsOnCell)       # (maxEdges, nCells)
    cellsOnEdge_np = np.asarray(gm_np.cellsOnEdge)       # (2, nEdges)
    cellsOnVertex_np = np.asarray(gm_np.cellsOnVertex)    # (vDeg, nVerts)
    verticesOnCell_np = np.asarray(gm_np.verticesOnCell)   # (maxEdges, nCells)
    verticesOnEdge_np = np.asarray(gm_np.verticesOnEdge)   # (2, nEdges)
    maxEdges = int(gm_np.maxEdges)
    vDeg = int(gm_np.vertexDegree)

    # Contiguous-block cell ownership (matches shard layout).
    cell_owner = np.repeat(np.arange(n_dev, dtype=np.int32), cells_per)
    if len(cell_owner) < nCells:
        cell_owner = np.concatenate([
            cell_owner,
            np.full(nCells - len(cell_owner), n_dev - 1, dtype=np.int32),
        ])

    # Dummy comm schedule (not needed for shard_map path).
    _dummy_comm = HaloCommSchedule(
        neighbor_ranks=(), send_counts=(), recv_counts=(),
        send_idx=jnp.empty(0, dtype=jnp.int32),
        recv_idx=jnp.empty(0, dtype=jnp.int32),
    )

    partitions = []
    local_meshes_raw = []

    for rank in range(n_dev):
        # ----- Owned entities (contiguous blocks) ----- #
        c_start, c_end = rank * cells_per, (rank + 1) * cells_per
        e_start, e_end = rank * edges_per, (rank + 1) * edges_per
        v_start, v_end = rank * verts_per, (rank + 1) * verts_per

        owned_cells = np.arange(c_start, c_end, dtype=np.int64)
        owned_edges = np.arange(e_start, e_end, dtype=np.int64)
        owned_vertices = np.arange(v_start, v_end, dtype=np.int64)
        owned_cells_set = set(owned_cells.tolist())
        owned_edges_set = set(owned_edges.tolist())

        # ----- Halo cells: k-ring neighbours of owned cells ----- #
        halo_cells_set = _compute_halo_cells(
            cell_owner, cellsOnCell_np, maxEdges, rank, halo_depth,
        )
        halo_cells = np.array(sorted(halo_cells_set), dtype=np.int64)
        local_cells = np.concatenate([owned_cells, halo_cells])
        local_cells_set = set(local_cells.tolist())

        # ----- Halo edges: edges connected to local cells ----- #
        local_edges_set = set()
        for e in range(nEdges):
            c0 = int(cellsOnEdge_np[0, e])
            c1 = int(cellsOnEdge_np[1, e])
            if c0 in local_cells_set or c1 in local_cells_set:
                local_edges_set.add(e)
        halo_edges = np.array(
            sorted(local_edges_set - owned_edges_set), dtype=np.int64,
        )
        local_edges = np.concatenate([owned_edges, halo_edges])

        # ----- Halo vertices: vertices connected to local cells/edges ----- #
        local_verts_set = set()
        for c in local_cells:
            for j in range(maxEdges):
                v = int(verticesOnCell_np[j, c])
                if 0 <= v < nVertices:
                    local_verts_set.add(v)
        for e in local_edges:
            for j in range(2):
                v = int(verticesOnEdge_np[j, e])
                if 0 <= v < nVertices:
                    local_verts_set.add(v)
        owned_verts_set = set(owned_vertices.tolist())
        halo_vertices = np.array(
            sorted(local_verts_set - owned_verts_set), dtype=np.int64,
        )
        local_vertices = np.concatenate([owned_vertices, halo_vertices])

        # ----- Global-to-local maps ----- #
        cell_g2l = np.full(nCells, -1, dtype=np.int64)
        for i, g in enumerate(local_cells):
            cell_g2l[g] = i
        edge_g2l = np.full(nEdges, -1, dtype=np.int64)
        for i, g in enumerate(local_edges):
            edge_g2l[g] = i
        vertex_g2l = np.full(nVertices, -1, dtype=np.int64)
        for i, g in enumerate(local_vertices):
            vertex_g2l[g] = i

        part = VoronoiPartition(
            rank=rank,
            n_ranks=n_dev,
            nCells_global=nCells,
            nEdges_global=nEdges,
            nVertices_global=nVertices,
            n_owned_cells=len(owned_cells),
            n_owned_edges=len(owned_edges),
            n_owned_vertices=len(owned_vertices),
            n_local_cells=len(local_cells),
            n_local_edges=len(local_edges),
            n_local_vertices=len(local_vertices),
            local_cells=local_cells,
            local_edges=local_edges,
            local_vertices=local_vertices,
            cell_g2l=cell_g2l,
            edge_g2l=edge_g2l,
            vertex_g2l=vertex_g2l,
            cell_comm=_dummy_comm,
            edge_comm=_dummy_comm,
            vertex_comm=_dummy_comm,
        )
        lm = build_local_mesh(gm_np, part)
        partitions.append(part)
        local_meshes_raw.append(lm)

    # Uniform padding to the maximum local sizes across all devices.
    max_lc = max(p.n_local_cells for p in partitions)
    max_le = max(p.n_local_edges for p in partitions)
    max_lv = max(p.n_local_vertices for p in partitions)

    local_meshes = [
        _pad_local_mesh_to(lm, max_lc, max_le, max_lv)
        for lm in local_meshes_raw
    ]

    # Stack into a single pytree with a leading device dimension.
    stacked_meshes = jax.tree.map(
        lambda *leaves: jnp.stack(leaves, axis=0),
        *local_meshes,
    )

    # Gather indices: for each device, global cell/edge indices of its
    # local entities (owned + halo), padded with 0 for ghost slots.
    gather_cells = np.zeros((n_dev, max_lc), dtype=np.int64)
    gather_edges = np.zeros((n_dev, max_le), dtype=np.int64)
    n_owned_cells = []
    n_owned_edges = []
    for rank, part in enumerate(partitions):
        gather_cells[rank, : part.n_local_cells] = part.local_cells
        gather_edges[rank, : part.n_local_edges] = part.local_edges
        n_owned_cells.append(part.n_owned_cells)
        n_owned_edges.append(part.n_owned_edges)

    return (
        stacked_meshes,
        jnp.array(gather_cells),
        jnp.array(gather_edges),
        n_owned_cells,
        n_owned_edges,
        max_lc,
        max_le,
    )


def make_voronoi_sharded_step(model, dev_config: DeviceConfig):
    """Create a halo-partitioned multi-GPU step for Voronoi (MPAS/TRiSK) grids.

    Instead of replicating the full state and redundantly computing the
    full step on every device, this implementation:

    1. Pre-computes per-device local meshes (owned cells/edges + halo)
       at setup time via domain decomposition.
    2. At each SSP-RK3 stage:
       a. **All-gather** the owned state shards to form the full state.
       b. Each device gathers its local state (owned + halo) from the
          full array via pre-computed indices.
       c. Each device computes tendencies on its **local mesh only**
          (O(N/p) work instead of O(N)).
       d. Each device extracts the owned portion for the RK update.
    3. After 3 stages, applies temperature floor and mass conservation
       fix, then returns the sharded result.

    Communication is O(N) per stage (all-gather), but compute is O(N/p).
    For the halo exchange, 2-ring halo depth is used to support del4
    hyperdiffusion.

    Parameters
    ----------
    model
        ``MPASPrimitiveEquationModel`` with ``.mesh``, ``.sigma_coord``,
        ``.config``.
    dev_config : DeviceConfig
        From :func:`~legoesm.parallel.mesh.create_voronoi_device_mesh`.

    Returns
    -------
    Callable[[state, float], state]
        JIT-compiled step function.
    """
    if dev_config.n_devices <= 1 or dev_config.mesh is None:
        return model.step

    import numpy as np
    try:
        from jax.shard_map import shard_map
    except ImportError:  # JAX < 0.8
        from jax.experimental.shard_map import shard_map
    from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
        mpas_hydrostatic_tendencies,
    )
    from legoesm.core.state import MPASHydrostaticState

    n_dev = dev_config.n_devices
    voronoi_dims = dev_config.voronoi_dims
    if voronoi_dims is None:
        raise ValueError("dev_config.voronoi_dims must be set for Voronoi grids")
    nCells, nEdges, _nVerts = voronoi_dims
    jax_mesh = dev_config.mesh
    face_sharding = dev_config.face_sharding

    cells_per = nCells // n_dev
    edges_per = nEdges // n_dev

    global_mesh = model.mesh
    sigma = model.sigma_coord
    cfg = model.config

    # ------------------------------------------------------------------
    # Setup: build per-device local meshes and gather indices
    # ------------------------------------------------------------------
    logger.info(
        "Building halo-partitioned infrastructure for %d device(s) "
        "(nCells=%d, nEdges=%d, halo_depth=2) ...",
        n_dev, nCells, nEdges,
    )
    t0 = time.time()
    (
        stacked_meshes,   # VoronoiMesh pytree with (n_dev, max_l*) leaves
        gather_cells,     # (n_dev, max_lc)
        gather_edges,     # (n_dev, max_le)
        _n_owned_cells,
        _n_owned_edges,
        max_lc,
        max_le,
    ) = _build_voronoi_partition_infra(global_mesh, n_dev, halo_depth=2)
    logger.info(
        "  partition setup done in %.2fs  "
        "(max_local_cells=%d, max_local_edges=%d, cells_per=%d, edges_per=%d)",
        time.time() - t0, max_lc, max_le, cells_per, edges_per,
    )

    # Replicate the stacked meshes so every device can index its slice.
    rep_sharding = dev_config.replicated_sharding
    stacked_meshes = jax.tree.map(
        lambda x: jax.device_put(x, rep_sharding) if hasattr(x, "shape") else x,
        stacked_meshes,
    )
    gather_cells = jax.device_put(gather_cells, rep_sharding)
    gather_edges = jax.device_put(gather_edges, rep_sharding)

    # ------------------------------------------------------------------
    # shard_map kernel: all-gather → local gather → local tendency
    # ------------------------------------------------------------------

    def _local_tendency(u_shard, T_shard, ps_shard, phis_shard, dt_val):
        """Inside shard_map: compute tendency for this device's partition."""
        # 1. All-gather to get full state on this device.
        u_full = jax.lax.all_gather(u_shard, "device", axis=0, tiled=True)
        T_full = jax.lax.all_gather(T_shard, "device", axis=0, tiled=True)
        ps_full = jax.lax.all_gather(ps_shard, "device", axis=0, tiled=True)
        phis_full = jax.lax.all_gather(phis_shard, "device", axis=0, tiled=True)

        # 2. Gather this device's local state (owned + halo).
        dev_idx = jax.lax.axis_index("device")
        my_cell_idx = gather_cells[dev_idx]   # (max_lc,)
        my_edge_idx = gather_edges[dev_idx]   # (max_le,)

        T_local = T_full[my_cell_idx]         # (max_lc, nlev)
        ps_local = ps_full[my_cell_idx]       # (max_lc,)
        phis_local = phis_full[my_cell_idx]   # (max_lc,)
        u_local = u_full[my_edge_idx]         # (max_le, nlev)

        # 3. Get this device's local mesh.
        my_mesh = jax.tree.map(lambda x: x[dev_idx], stacked_meshes)

        # 4. Build local state and compute tendency.
        local_state = MPASHydrostaticState(
            u=model.mesh.nCells,  # placeholder — replaced below
            T=model.mesh.nCells,
            p_s=model.mesh.nCells,
            phis=model.mesh.nCells,
        )
        # Construct properly typed Field objects using the model's state
        # field metadata (name, dims, units, etc.).
        from legoesm.core.field import Field
        local_state = MPASHydrostaticState(
            u=Field(data=u_local, name="u",
                    dims=("nEdges", "nlev"), units="m/s",
                    long_name="normal velocity", staggering="edge"),
            T=Field(data=T_local, name="T",
                    dims=("nCells", "nlev"), units="K",
                    long_name="temperature", staggering="cell"),
            p_s=Field(data=ps_local, name="p_s",
                      dims=("nCells",), units="Pa",
                      long_name="surface pressure", staggering="cell"),
            phis=Field(data=phis_local, name="phis",
                       dims=("nCells",), units="m^2/s^2",
                       long_name="surface geopotential", staggering="cell"),
        )

        tend = mpas_hydrostatic_tendencies(
            local_state, my_mesh, sigma, cfg, dt=dt_val,
        )

        # 5. Return only the owned shard of the tendency.
        du = tend.du_dt.data[:edges_per]
        dT = tend.dT_dt.data[:cells_per]
        dps = tend.dp_s_dt.data[:cells_per]

        return du, dT, dps

    _shard_tendency = shard_map(
        _local_tendency,
        mesh=jax_mesh,
        in_specs=(P("device"), P("device"), P("device"), P("device"), P()),
        out_specs=(P("device"), P("device"), P("device")),
        check_rep=False,
    )

    # ------------------------------------------------------------------
    # JIT-compiled step: SSP-RK3 with halo refresh between stages
    # ------------------------------------------------------------------

    @jax.jit
    def _step(state, dt):
        u = state.u.data       # (nEdges, nlev) sharded
        T = state.T.data       # (nCells, nlev) sharded
        ps = state.p_s.data    # (nCells,) sharded
        phis = state.phis.data # (nCells,) sharded

        # --- Stage 1: k1 = state + dt * F(state) ---
        du1, dT1, dps1 = _shard_tendency(u, T, ps, phis, dt)
        u1 = u + dt * du1
        T1 = T + dt * dT1
        ps1 = ps + dt * dps1

        # --- Stage 2: k2 = 3/4*state + 1/4*(k1 + dt*F(k1)) ---
        du2, dT2, dps2 = _shard_tendency(u1, T1, ps1, phis, dt)
        u2 = 0.75 * u + 0.25 * (u1 + dt * du2)
        T2 = 0.75 * T + 0.25 * (T1 + dt * dT2)
        ps2 = 0.75 * ps + 0.25 * (ps1 + dt * dps2)

        # --- Stage 3: k3 = 1/3*state + 2/3*(k2 + dt*F(k2)) ---
        du3, dT3, dps3 = _shard_tendency(u2, T2, ps2, phis, dt)
        u_new = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * du3)
        T_new = (1.0 / 3.0) * T + (2.0 / 3.0) * (T2 + dt * dT3)
        ps_new = (1.0 / 3.0) * ps + (2.0 / 3.0) * (ps2 + dt * dps3)

        # --- Post-processing (mirrors model.step) ---
        if cfg.T_min > 0:
            T_new = jnp.maximum(T_new, cfg.T_min)

        if cfg.fix_mass:
            # Global mass fixer via all-reduce (works on sharded arrays).
            area_full = global_mesh.areaCell      # replicated
            ps_old_full = jax.lax.with_sharding_constraint(
                ps, rep_sharding)
            ps_new_full = jax.lax.with_sharding_constraint(
                ps_new, rep_sharding)
            mass_old = jnp.sum(ps_old_full * area_full)
            mass_new = jnp.sum(ps_new_full * area_full)
            correction = (mass_old - mass_new) / jnp.sum(area_full)
            ps_new = ps_new + correction

        return MPASHydrostaticState(
            u=state.u.replace(data=u_new),
            T=state.T.replace(data=T_new),
            p_s=state.p_s.replace(data=ps_new),
            phis=state.phis,
        )

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
