"""Sharded dynamics for multi-GPU/TPU cubed-sphere simulations.

Uses ``jax.shard_map`` to partition cubed-sphere faces across
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
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from legoesm.parallel.mesh import DeviceConfig, N_FACES
from legoesm.core.field import Field
from legoesm.grids.halo import pad_halo, pad_halo_4d

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
    """Build PartitionSpec objects from a DeviceConfig.

    Three regimes:

    * **Face-only sharding** (1, 2, 3, 6 devices): leading face axis
      is sharded.
    * **Sub-face tiling** (6·k² devices): face axis + two tile axes.
    * **Level-parallel cubed-sphere** (issue #273 fallback for
      device counts that fail face-divisibility, e.g. 4): mesh
      axis is ``'level'``, NOT ``'face'``.  The dycore runs
      fully replicated horizontally — every device computes the
      complete cubed-sphere stencil independently — and the
      level axis is reserved for downstream column-wise physics
      to shard.  Returning ``P()`` for both 3D and 4D specs
      makes ``shard_state`` produce a fully-replicated dycore
      state, which is the correct behavior on the level mesh.
    """
    if getattr(config, "grid_type", None) == "cubed_sphere_level":
        # Replicated-dycore path.  Mesh has only a ``'level'`` axis;
        # any attempt to address ``'face'`` would crash with
        # ``unmatched mesh axis`` from JAX.
        return ShardingSpec(
            face_3d=P(),
            face_2d=P(),
            replicated=P(),
            tiled_3d=None,
            tiled_2d=None,
        )

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
            if leaf.ndim >= 1 and leaf.shape[0] == N_FACES:
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
            if leaf.ndim >= 1 and leaf.shape[0] == N_FACES:
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
    phys_id : int or None
        ``id(physics_fn)`` — distinct callables produce different XLA
        programs and must not share a cache entry.
    """
    state_structure: object
    leaf_meta: tuple
    sharding_key: tuple
    has_physics: bool
    phys_id: object = None


def _make_cache_key(state, config: DeviceConfig, physics_fn=None) -> StepCacheKey:
    """Build a cache key from current call arguments."""
    structure = jax.tree.structure(state)
    leaves = jax.tree.leaves(state)
    leaf_meta = tuple(
        (leaf.shape, leaf.dtype) if hasattr(leaf, "shape") else (None, None)
        for leaf in leaves
    )
    sharding_key = (config.n_devices, config.tiling, config.grid_type)
    has_physics = physics_fn is not None
    phys_id = id(physics_fn) if has_physics else None
    return StepCacheKey(
        state_structure=structure,
        leaf_meta=leaf_meta,
        sharding_key=sharding_key,
        has_physics=has_physics,
        phys_id=phys_id,
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

    def _get_in_shardings(self, state):
        """Input-sharding pytree for the ``(state, dt)`` call signature.

        Mirrors the output leaf policy (:func:`create_output_shardings`)
        for the state argument so the compiled executable pins its inputs
        to the face-sharded layout instead of silently accepting — and
        then redundantly computing the full globe from — a replicated
        copy on every device.  ``dt`` is a scalar and stays
        unconstrained (replicated).  On a single-device config
        (``mesh is None``) the policy pytree is all-``None``, i.e. no
        constraint — identical to the previous behavior.
        """
        return (self._get_out_shardings(state), None)

    def _compile(self, state, physics_fn=None):
        """Trace and compile a new executable, returning the jitted fn."""
        t0 = time.monotonic()

        out_shardings = self._get_out_shardings(state)
        in_shardings = self._get_in_shardings(state)

        model = self._model
        halo_fn = self._halo_exchange_fn

        # Build the jitted step.  ``dt`` is a regular traced argument
        # (not static), so a single compiled program handles all dt values.
        # ``physics_fn`` is captured in the closure so JAX doesn't try
        # to trace it as an array argument.  The phys_id in the cache
        # key ensures we compile separate programs for distinct callables.
        if physics_fn is not None and hasattr(model, "step_with_physics"):
            _phys = physics_fn

            @partial(jax.jit, in_shardings=in_shardings,
                     out_shardings=out_shardings)
            def _jitted(s, dt):
                new = model.step_with_physics(s, dt, _phys)
                if halo_fn is not None:
                    new = halo_fn(new)
                return new
        else:
            @partial(jax.jit, in_shardings=in_shardings,
                     out_shardings=out_shardings)
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

    def _get_executable(self, state, physics_fn=None):
        """Return the cached executable, compiling if needed."""
        key = _make_cache_key(state, self._config, physics_fn)
        if key not in self._cache:
            self._cache[key] = self._compile(state, physics_fn)
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
            Physics forcing function.  Captured in the compiled
            closure — not passed as a traced argument.

        Returns
        -------
        Updated state (same sharding).
        """
        fn = self._get_executable(state, physics_fn)
        return fn(state, dt)

    def is_compiled_for(self, state, physics_fn=None) -> bool:
        """Check whether a compiled executable already exists for this state."""
        key = _make_cache_key(state, self._config, physics_fn)
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
    *,
    n: int = 0,
    nlev: int = 1,
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
    n : int, optional
        Per-face resolution — logging/prewarm metadata forwarded to
        ``activate_spmd_halo_backend``.  The old ppermute-vs-all_gather
        volume auto-selection is RETIRED: ppermute is always selected;
        all_gather only via explicit ``LEGOESM_SPMD_FORCE_ALLGATHER=1``.
    nlev : int, optional
        Number of vertical levels (logging only, see ``n``).

    Returns
    -------
    CompiledShardedStep or _SingleDeviceStep
        Callable with ``(state, dt, physics_fn=None) -> state``.

    Notes
    -----
    For face-only sharding (1/2/3/6 devices) this function ACTIVATES
    the explicit SPMD halo backend
    (``cubesphere_exchange.activate_spmd_halo_backend``): ``pad_halo``
    then routes through shard_map ppermute kernels (multiface; one-face
    at halo=1 with 6 devices) instead of relying on XLA's implicit
    cross-shard reads.  Letting GSPMD auto-insert collectives for the
    cross-face reads — the pre-activation behavior this Notes section
    used to describe — replicates ALL compute per device (HLO probe job
    8456476); the all_gather kernels survive only as the explicit
    ``LEGOESM_SPMD_FORCE_ALLGATHER=1`` diagnostic.

    For sub-face tiling (>6 devices), additional tile-boundary
    exchange is needed; the SPMD halo backend is NOT activated there
    (the tiled path is unvalidated — bench guards exclude it).
    """
    if config.mesh is None:
        logger.info("make_sharded_step: single-device mode, using plain JIT")
        return _SingleDeviceStep(model)

    # Activate explicit SPMD halo exchange for face-sharded cubed-sphere.
    # This replaces implicit cross-shard reads with explicit
    # collective-permute rounds, producing much better XLA communication
    # patterns.
    #
    # Iter-49 generalised activation from "exactly 6 devices" to "any
    # divisor of 6" (1, 2, 3, 6) on the allgather kernels; the
    # ppermute-multiface refit then made ppermute the DEFAULT exchange
    # for every face-sharded count and at halo=2 — the allgather
    # variant provably replicated ALL compute per device (HLO probe job
    # 8456476, per-device FLOPs ratio 1.00 at 2 devices) and is now an
    # explicit diagnostic opt-in only (LEGOESM_SPMD_FORCE_ALLGATHER=1).
    _n = config.n_devices
    _tiling = getattr(config, 'tiling', (1, 1))
    _face_ok = (_n in (1, 2, 3, 6) and _tiling == (1, 1))
    # 6*kt^2 sub-face tiling: the tiled ppermute EXCHANGE is serial-
    # exact (h1+h2, offsets+raw, corners — probe job 8464648), but the
    # DYCORE is not yet tile-aware: consumers slice padded arrays with
    # full-face (n+2h) indexing (operators_cdgrid.py:555/638/766,
    # operators_3d.py:85, fv_tp_2d.py:1024, fv3_sw_core.py:1358 —
    # codex review) and staggered (n+1) leaves cannot shard over tile
    # axes (IndivisibleError, probe job 8464703).  Activation is
    # therefore EXPERIMENTAL and opt-in only; the P4 milestone
    # (tile-aware consumers + staggered-leaf ownership layout) flips
    # the default.
    import os as _os
    _tiled_ok = (
        _os.environ.get("LEGOESM_TILED_SPMD", "0") == "1"
        and _tiling[0] == _tiling[1] and _tiling[0] >= 2
        and _n == 6 * _tiling[0] * _tiling[1]
    )
    if ((_face_ok or _tiled_ok)
            and config.mesh is not None
            and "face" in getattr(config.mesh, 'axis_names', ())):
        from legoesm.parallel.cubesphere_exchange import (
            activate_spmd_halo_backend,
        )
        activate_spmd_halo_backend(config.mesh, n=n, nlev=nlev)
        logger.info(
            "make_sharded_step: activated SPMD halo backend "
            "(%d devices, %s, n=%d, nlev=%d)",
            config.n_devices,
            "face-sharded" if _face_ok else f"tiled {_tiling}",
            n, nlev,
        )

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
    def _exchange(state):
        """Apply halo exchange to face-dimensioned arrays.

        This function pads each 2D face field ``(6, n, n)`` with halo
        data from neighbors, then strips the halos back to ``(6, n, n)``.
        This ensures boundary values are fresh after a dynamics step.

        For 3D fields ``(6, n, n, nlev)`` the exchange uses the native
        4D halo path (``pad_halo_4d``) which fetches halos for every
        level in one MPI message — see CLAUDE.md ``Parallel and HPC
        Rules``.  The previous ``vmap(pad_halo)`` per level pattern is
        forbidden because it issues ``nlev`` separate messages.
        """
        def _exchange_leaf(leaf):
            if not isinstance(leaf, (jax.Array, jnp.ndarray)):
                return leaf
            if leaf.ndim < 3 or leaf.shape[0] != N_FACES:
                return leaf

            if leaf.ndim == 3:
                # 2D field: (6, n, n)
                padded = pad_halo(leaf)
                return padded[:, 1:-1, 1:-1]

            elif leaf.ndim == 4:
                # 3D field: (6, n, n, nlev) — single 4D halo exchange.
                padded = pad_halo_4d(leaf)
                return padded[:, 1:-1, 1:-1, :]

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
            if leaf.ndim < 3 or leaf.shape[0] != N_FACES:
                return leaf
            if leaf.ndim == 3:
                return jax_native_halo_exchange(leaf, grid, mesh=config.mesh)
            elif leaf.ndim == 4:
                # NOTE: jax_native_halo_exchange / _ppermute_halo_exchange
                # are documented as a 2D-only legacy path.  When the
                # SPMD backend in cubesphere_exchange.py is active the
                # production code does not enter this branch — it goes
                # through the native 4D ``packed_pad_halo_4d``.  Keep
                # the per-level vmap here as a documented fallback;
                # extending the legacy ppermute kernel to 4D is tracked
                # separately.
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


def _close_halo_under_cellsOnEdge(
    owned_cells_arr,
    halo_cells_set,
    cellsOnEdge_np,
    n_passes,
):
    """Iteratively extend ``halo_cells_set`` so the halo is "closed
    under cellsOnEdge".

    A halo set is *closed* when every edge whose one ``cellsOnEdge``
    entry is in ``owned ∪ halo`` also has its OTHER entry in the halo.
    Closure ensures that downstream ``cellsOnEdge``-based operators
    (e.g., ``gradient_edge``, ``divergence_cell``) at every owned cell
    AND at every halo cell whose value is read by an owned-cell
    operator can be evaluated locally without ``-1`` indices.

    Iter-23 surfaced the *unclosed* failure mode: a single
    ``cellsOnEdge`` entry of ``-1`` (the global→local sentinel)
    caused ``gradient_edge`` to silently dereference Python's
    last-element index → 1e76 garbage after one SSP-RK3 step.

    Iter-25/27/37 settled on 2 passes as the minimum sufficient depth
    for the MPAS dycore's longest operator chain (∇⁴ in
    hyperdiffusion: owned → halo-1 → halo-2).

    Modifies ``halo_cells_set`` in place.

    Parameters
    ----------
    owned_cells_arr : np.ndarray, shape (n_owned,), int64
        Owned cell global indices for this rank.
    halo_cells_set : set[int]
        Initial halo (cellsOnCell ring + iter-23 owned-edge other-cells).
    cellsOnEdge_np : np.ndarray, shape (2, nEdges), int
        Global cellsOnEdge connectivity.
    n_passes : int
        Number of augmentation iterations.  2 is the minimum that
        closes the dycore's depth-2 operator chain; use a higher
        value only if a future operator extends the chain depth.
    """
    for _ in range(n_passes):
        cur_local_arr = np.concatenate([
            owned_cells_arr,
            np.fromiter(
                halo_cells_set,
                dtype=np.int64,
                count=len(halo_cells_set),
            ),
        ])
        # Edges where AT LEAST one ``cellsOnEdge`` is currently local.
        edge_one_in = (
            np.isin(cellsOnEdge_np[0], cur_local_arr)
            | np.isin(cellsOnEdge_np[1], cur_local_arr)
        )
        cand_edges = np.flatnonzero(edge_one_in)
        cand_cells = cellsOnEdge_np[:, cand_edges].reshape(-1)
        cand_cells = np.unique(cand_cells[cand_cells >= 0])
        # Set difference: cells not yet in local set.
        new_cells = cand_cells[~np.isin(cand_cells, cur_local_arr)]
        if new_cells.size == 0:
            return
        halo_cells_set.update(new_cells.tolist())


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
    max_lc : int
    max_le : int
    partitions : list[VoronoiPartition]
        Per-device partition descriptors (for ppermute schedule building).
    cell_owner : np.ndarray, (nCells,)
        Cell ownership array.
    """
    import numpy as np
    from legoesm.parallel.voronoi_partition import (
        VoronoiPartition,
        HaloCommSchedule,
        build_local_mesh,
        compute_halo_cells,
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
    np.asarray(gm_np.cellsOnVertex)    # (vDeg, nVerts)
    verticesOnCell_np = np.asarray(gm_np.verticesOnCell)   # (maxEdges, nCells)
    verticesOnEdge_np = np.asarray(gm_np.verticesOnEdge)   # (2, nEdges)
    maxEdges = int(gm_np.maxEdges)
    int(gm_np.vertexDegree)

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
        set(owned_edges.tolist())

        # ----- Halo cells: k-ring neighbours of owned cells ----- #
        halo_cells_set = compute_halo_cells(
            cell_owner, cellsOnCell_np, maxEdges, rank, halo_depth,
        )
        # Augment with the OTHER cell of every owned edge: on Voronoi
        # SCVT meshes the cellsOnCell adjacency *should* match the
        # cellsOnEdge connectivity, but the k-ring construction in
        # ``compute_halo_cells`` can miss a handful of cells at the
        # mesh boundary or near pentagons (owned edges whose far cell
        # is reachable via cellsOnEdge but whose hop chain through
        # cellsOnCell at depth <= halo_depth is broken by a -1 slot
        # or pentagon irregularity).  Without this augmentation, the
        # local mesh's cellsOnEdge has -1 entries for those edges
        # after remap → ``gradient_edge`` does ``phi[-1]`` (Python
        # last-element indexing!) and produces 1e76 garbage
        # within one SSP-RK3 step.  Iter-23 root-cause analysis.
        owned_edge_cells = cellsOnEdge_np[:, owned_edges].reshape(-1)
        owned_edge_cells = owned_edge_cells[owned_edge_cells >= 0]
        for c in owned_edge_cells:
            ic = int(c)
            if ic not in owned_cells_set:
                halo_cells_set.add(ic)
        # Iter-25/27/37: extend the halo so every halo cell's adjacent
        # edges have BOTH cells in local_cells.  Without this, halo
        # cells have some of their adjacent edges silently excluded by
        # the AND filter below; operator quantities at halo cells then
        # differ slightly from single-device, and owned-cell tendencies
        # that read those halo quantities inherit ~1% drift per step.
        # Two passes are sufficient for the dycore's longest operator
        # chain (depth 2: cell → edge → cell, twice for ∇⁴ in
        # hyperdiffusion) — see iter-37 docstring.
        owned_cells_arr = np.asarray(owned_cells, dtype=np.int64)
        _close_halo_under_cellsOnEdge(
            owned_cells_arr, halo_cells_set, cellsOnEdge_np, n_passes=2,
        )
        halo_cells = np.array(sorted(halo_cells_set), dtype=np.int64)
        local_cells = np.concatenate([owned_cells, halo_cells])
        set(local_cells.tolist())

        # ----- Halo edges: edges where BOTH cellsOnEdge are in local_cells ----- #
        # Original Python loop over nEdges scaled poorly at MPAS resolutions
        # (1M+ cells); replace with a single ``np.isin`` on the cellsOnEdge
        # neighbour arrays so the whole partition setup is O(nEdges) numpy.
        #
        # CRITICAL: use AND, not OR.  Including an edge whose only one
        # neighbour is in local_cells leaves the other neighbour at -1
        # after the global→local remap (``cell_g2l[non_local] == -1``);
        # downstream operators like ``gradient_edge(phi, mesh)`` then
        # do ``phi[c1=-1]`` which is Python's last-element indexing
        # and produces garbage, blowing the SSP-RK3 step into 1e76
        # territory after a single iteration (iter-21 / iter-23 finding).
        # Using AND keeps the cellsOnEdge connectivity fully valid in the
        # local mesh; halo cells that lack some of their edges due to
        # the cut do not break correctness because the dycore only
        # uses tendencies on OWNED cells (halo tendencies are discarded
        # by the ``return`` in ``_local_tendency``).
        local_cells_arr = local_cells
        edge_in_local = np.isin(cellsOnEdge_np[0], local_cells_arr) & np.isin(
            cellsOnEdge_np[1], local_cells_arr,
        )
        local_edges_arr = np.flatnonzero(edge_in_local).astype(np.int64)
        # Sanity: every owned edge must satisfy the both-sides-local test
        # (an owned edge has at least one cell in owned_cells; the cell-
        # halo of depth >=1 covers the other side).  Surface a clear
        # error if a future mesh ordering change breaks that invariant.
        if not np.isin(owned_edges, local_edges_arr).all():
            missing = np.setdiff1d(owned_edges, local_edges_arr)
            raise RuntimeError(
                f"Voronoi partition rank={rank}: {len(missing)} owned "
                f"edges have a neighbour cell outside the halo-depth="
                f"{halo_depth} cell halo.  Increase halo_depth or check "
                f"the mesh ordering produced by reorder_voronoi_for_sharding."
            )
        halo_edges = np.setdiff1d(
            local_edges_arr, owned_edges, assume_unique=True,
        )
        local_edges = np.concatenate([owned_edges, halo_edges])

        # ----- Halo vertices: vertices connected to local cells/edges ----- #
        # Vectorised gather: collect verticesOnCell over local cells and
        # verticesOnEdge over local edges in one pass each, then dedupe.
        verts_from_cells = verticesOnCell_np[:, local_cells].reshape(-1)
        verts_from_edges = verticesOnEdge_np[:, local_edges].reshape(-1)
        candidate_verts = np.concatenate([verts_from_cells, verts_from_edges])
        valid = (candidate_verts >= 0) & (candidate_verts < nVertices)
        local_verts_arr = np.unique(candidate_verts[valid]).astype(np.int64)
        halo_vertices = np.setdiff1d(
            local_verts_arr, owned_vertices, assume_unique=True,
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
        partitions,
        cell_owner,
    )


def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
                             edges_per, max_lc, max_le):
    """Build a ppermute-based halo exchange schedule.

    Instead of all-gathering the full state (O(N) communication),
    this schedule uses ``jax.lax.ppermute`` to exchange only halo data
    between neighboring devices.  The communication graph is edge-colored
    so that each round of ppermute moves data between non-conflicting
    pairs simultaneously.

    Parameters
    ----------
    partitions : list[VoronoiPartition]
    cell_owner : np.ndarray, (nCells,)
    n_dev, cells_per, edges_per : int
    max_lc, max_le : int
        Maximum local cell/edge counts (owned + halo) across devices.

    Returns
    -------
    dict with keys:
        n_rounds, ppermute_perms, send_cell_idx, recv_cell_pos,
        send_edge_idx, recv_edge_pos, halo_cells_per_round,
        halo_edges_per_round.
    """
    import numpy as np
    from collections import defaultdict

    # ------------------------------------------------------------------
    # 1. For each device pair, find which cells/edges cross the boundary
    # ------------------------------------------------------------------
    # halo_cells_from[d][d'] = global indices of d's halo cells owned by d'
    halo_cells_from: dict[int, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list))
    halo_edges_from: dict[int, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list))

    for d, part in enumerate(partitions):
        for h_idx in range(part.n_owned_cells, part.n_local_cells):
            g = int(part.local_cells[h_idx])
            owner = int(cell_owner[g])
            halo_cells_from[d][owner].append(g)

        for h_idx in range(part.n_owned_edges, part.n_local_edges):
            g = int(part.local_edges[h_idx])
            owner = min(g // edges_per, n_dev - 1)
            halo_edges_from[d][owner].append(g)

    # ------------------------------------------------------------------
    # 2. Build undirected communication graph
    # ------------------------------------------------------------------
    comm_pairs: set[tuple[int, int]] = set()
    for d in range(n_dev):
        for d_prime in halo_cells_from[d]:
            if d != d_prime:
                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
        for d_prime in halo_edges_from[d]:
            if d != d_prime:
                comm_pairs.add((min(d, d_prime), max(d, d_prime)))

    if not comm_pairs:
        return {
            'n_rounds': 0,
            'ppermute_perms': [],
            'send_cell_idx': [],
            'recv_cell_pos': [],
            'send_edge_idx': [],
            'recv_edge_pos': [],
            'halo_cells_per_round': [],
            'halo_edges_per_round': [],
        }

    # ------------------------------------------------------------------
    # 3. Edge-color the graph (greedy)
    # ------------------------------------------------------------------
    vertex_colors: dict[int, set[int]] = defaultdict(set)
    edge_colors: dict[tuple[int, int], int] = {}
    for u, v in sorted(comm_pairs):
        used = vertex_colors[u] | vertex_colors[v]
        color = 0
        while color in used:
            color += 1
        edge_colors[(u, v)] = color
        vertex_colors[u].add(color)
        vertex_colors[v].add(color)

    n_rounds = max(edge_colors.values()) + 1
    rounds: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (u, v), color in edge_colors.items():
        rounds[color].append((u, v))

    # ------------------------------------------------------------------
    # 4. Build directed send/recv maps for each device pair
    # ------------------------------------------------------------------
    # cell_send_map[(src, dst)] = list of owned-local indices in src to send
    # cell_recv_map[(dst, src)] = list of local positions in dst to place data
    cell_send_map: dict[tuple[int, int], list[int]] = {}
    cell_recv_map: dict[tuple[int, int], list[int]] = {}
    edge_send_map: dict[tuple[int, int], list[int]] = {}
    edge_recv_map: dict[tuple[int, int], list[int]] = {}

    for d in range(n_dev):
        for d_prime, cells_g in halo_cells_from[d].items():
            if d_prime == d:
                continue
            # d_prime sends its owned cells that d needs as halo
            cell_send_map[(d_prime, d)] = [
                g - d_prime * cells_per for g in cells_g]
            cell_recv_map[(d, d_prime)] = [
                int(partitions[d].cell_g2l[g]) for g in cells_g]

        for d_prime, edges_g in halo_edges_from[d].items():
            if d_prime == d:
                continue
            edge_send_map[(d_prime, d)] = [
                g - d_prime * edges_per for g in edges_g]
            edge_recv_map[(d, d_prime)] = [
                int(partitions[d].edge_g2l[g]) for g in edges_g]

    # ------------------------------------------------------------------
    # 5. Assemble per-round ppermute patterns and index arrays
    # ------------------------------------------------------------------
    ppermute_perms_out: list[list[tuple[int, int]]] = []
    send_cell_idx_out: list[jnp.ndarray] = []
    recv_cell_pos_out: list[jnp.ndarray] = []
    send_edge_idx_out: list[jnp.ndarray] = []
    recv_edge_pos_out: list[jnp.ndarray] = []
    halo_cells_per_round: list[int] = []
    halo_edges_per_round: list[int] = []

    for r in range(n_rounds):
        # Max halo size across all pairs in this round
        max_c = 0
        max_e = 0
        for u, v in rounds[r]:
            for src, dst in [(u, v), (v, u)]:
                max_c = max(max_c, len(cell_send_map.get((src, dst), [])))
                max_e = max(max_e, len(edge_send_map.get((src, dst), [])))
        max_c = max(max_c, 1)  # at least 1 for array shape
        max_e = max(max_e, 1)
        halo_cells_per_round.append(max_c)
        halo_edges_per_round.append(max_e)

        # Bidirectional ppermute pattern
        perm: list[tuple[int, int]] = []
        partner: dict[int, int] = {}
        for u, v in rounds[r]:
            perm.append((u, v))
            perm.append((v, u))
            partner[u] = v
            partner[v] = u
        ppermute_perms_out.append(perm)

        # Per-device index arrays (padded with safe defaults)
        sc = np.zeros((n_dev, max_c), dtype=np.int64)
        # Garbage slot: writes go to max_lc (trimmed off later)
        rc = np.full((n_dev, max_c), max_lc, dtype=np.int64)
        se = np.zeros((n_dev, max_e), dtype=np.int64)
        re = np.full((n_dev, max_e), max_le, dtype=np.int64)

        for d in range(n_dev):
            if d not in partner:
                continue
            dp = partner[d]

            cs = cell_send_map.get((d, dp), [])
            for j, idx in enumerate(cs):
                sc[d, j] = idx

            cr = cell_recv_map.get((d, dp), [])
            for j, pos in enumerate(cr):
                rc[d, j] = pos

            es = edge_send_map.get((d, dp), [])
            for j, idx in enumerate(es):
                se[d, j] = idx

            er = edge_recv_map.get((d, dp), [])
            for j, pos in enumerate(er):
                re[d, j] = pos

        send_cell_idx_out.append(jnp.array(sc))
        recv_cell_pos_out.append(jnp.array(rc))
        send_edge_idx_out.append(jnp.array(se))
        recv_edge_pos_out.append(jnp.array(re))

    return {
        'n_rounds': n_rounds,
        'ppermute_perms': ppermute_perms_out,
        'send_cell_idx': send_cell_idx_out,
        'recv_cell_pos': recv_cell_pos_out,
        'send_edge_idx': send_edge_idx_out,
        'recv_edge_pos': recv_edge_pos_out,
        'halo_cells_per_round': halo_cells_per_round,
        'halo_edges_per_round': halo_edges_per_round,
    }


def make_voronoi_sharded_step(
    model,
    dev_config: DeviceConfig,
    *,
    halo_strategy: str = "auto",
    ppermute_cells_per_device_threshold: int = 2_000,
):
    """Create a halo-partitioned multi-GPU step for Voronoi (MPAS/TRiSK) grids.

    Instead of replicating the full state and redundantly computing the
    full step on every device, this implementation:

    1. Pre-computes per-device local meshes (owned cells/edges + halo)
       at setup time via domain decomposition.
    2. At each SSP-RK3 stage, exchanges only halo data between
       neighboring devices (not the full state), then computes
       tendencies on the local mesh.
    3. After 3 stages, applies temperature floor and mass conservation
       fix, then returns the sharded result.

    Parameters
    ----------
    model
        ``MPASPrimitiveEquationModel`` with ``.mesh``, ``.sigma_coord``,
        ``.config``.
    dev_config : DeviceConfig
        From :func:`~legoesm.parallel.mesh.create_voronoi_device_mesh`.
    halo_strategy : str
        ``"auto"`` (default) selects ``"ppermute"`` for large grids and
        ``"allgather"`` for small ones based on
        *ppermute_cells_per_device_threshold*.
        ``"ppermute"`` forces neighbor-only exchange via
        ``jax.lax.ppermute`` — O(halo) communication.
        ``"allgather"`` forces the full-state all-gather —
        O(N) communication.
    ppermute_cells_per_device_threshold : int
        When ``halo_strategy="auto"``, use ppermute only if each device
        owns at least this many cells.  Below this threshold the
        per-round packing/scatter overhead of ppermute exceeds the
        communication savings over allgather.  Default: 2 000.
        (Lowered from 25 000 to avoid the O(N) allgather bottleneck
        on moderate icosahedral grids like I5 with 2–4 GPUs.)

    Returns
    -------
    callable
        ``step(state, dt, physics_fn=None) -> state``.  ``physics_fn``
        follows the MPAS operator-split convention
        (``physics_fn(state, mesh, sigma_coord, *, phys_state, forcing)``
        returning bare ``MPASHydrostaticTendencies`` — e.g.
        ``held_suarez_forcing_mpas``) and is captured in the jitted
        closure, never traced as an argument (same convention as
        :class:`CompiledShardedStep`).  Each distinct ``physics_fn``
        identity compiles a separate executable; ``physics_fn=None``
        compiles exactly the dynamics-only graph.  On a single-device
        config this returns ``model.step``, whose signature is
        call-compatible.
    """
    if dev_config.n_devices <= 1 or dev_config.mesh is None:
        return model.step

    try:
        from jax import shard_map  # JAX >= 0.8 exposes it at top level
    except ImportError:  # JAX < 0.8 fallback
        from jax.experimental.shard_map import shard_map
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

    # The MPAS RHS comes from the passed-in model instance, not an atmosphere
    # import — this substrate ``parallel`` module must not depend UP on the
    # atmosphere component (federation: legoesm-core stays standalone-installable).
    # The free function is needed (not ``.tendencies``) because each device runs
    # it on its own rank-local, traced mesh.
    mpas_hydrostatic_tendencies = getattr(model, "sharded_tendency_fn", None)
    if mpas_hydrostatic_tendencies is None:
        raise TypeError(
            f"{type(model).__name__} does not expose a 'sharded_tendency_fn' "
            f"staticmethod; the multi-device Voronoi sharder needs the free "
            f"tendency RHS (state, mesh, sigma_coord, config, *, dt=...) to run "
            f"on a rank-local mesh without importing the dycore's component."
        )

    # ------------------------------------------------------------------
    # Auto-select halo strategy based on grid size per device
    # ------------------------------------------------------------------
    if halo_strategy == "auto":
        if cells_per < ppermute_cells_per_device_threshold:
            halo_strategy = "allgather"
            logger.info(
                "Auto-selected allgather strategy: cells_per_device=%d < "
                "threshold=%d — ppermute packing overhead would dominate.",
                cells_per, ppermute_cells_per_device_threshold,
            )
        else:
            halo_strategy = "ppermute"
            logger.info(
                "Auto-selected ppermute strategy: cells_per_device=%d >= "
                "threshold=%d.",
                cells_per, ppermute_cells_per_device_threshold,
            )

    # ------------------------------------------------------------------
    # Setup: build per-device local meshes and gather indices
    # ------------------------------------------------------------------
    logger.info(
        "Building halo-partitioned infrastructure for %d device(s) "
        "(nCells=%d, nEdges=%d, halo_depth=3, strategy=%s) ...",
        n_dev, nCells, nEdges, halo_strategy,
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
        partitions_out,   # list[VoronoiPartition] (for ppermute schedule)
        cell_owner_out,   # np.ndarray (nCells,) cell ownership
    ) = _build_voronoi_partition_infra(global_mesh, n_dev, halo_depth=3)
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

    nlev = model.sigma_coord.n_levels

    # ------------------------------------------------------------------
    # Strategy dispatch: ppermute (O(halo)) vs allgather (O(N))
    # ------------------------------------------------------------------

    use_ppermute = halo_strategy == "ppermute"

    if use_ppermute:
        # Build ppermute schedule: neighbor-only halo exchange
        t1 = time.time()
        pp_sched = _build_ppermute_schedule(
            partitions_out, cell_owner_out, n_dev,
            cells_per, edges_per, max_lc, max_le,
        )
        n_rounds = pp_sched['n_rounds']
        ppermute_perms = pp_sched['ppermute_perms']

        # Replicate index arrays on all devices
        send_cell_idx = [
            jax.device_put(a, rep_sharding) for a in pp_sched['send_cell_idx']]
        recv_cell_pos = [
            jax.device_put(a, rep_sharding) for a in pp_sched['recv_cell_pos']]
        send_edge_idx = [
            jax.device_put(a, rep_sharding) for a in pp_sched['send_edge_idx']]
        recv_edge_pos = [
            jax.device_put(a, rep_sharding) for a in pp_sched['recv_edge_pos']]

        # Log halo exchange statistics
        sum(pp_sched['halo_cells_per_round'])
        sum(pp_sched['halo_edges_per_round'])
        total_pp_bytes = sum(
            hc * (nlev + 2) + he * nlev
            for hc, he in zip(pp_sched['halo_cells_per_round'],
                              pp_sched['halo_edges_per_round'])
        ) * 4  # float32
        ag_bytes = (nCells * (nlev + 2) + nEdges * nlev) * 4
        logger.info(
            "  ppermute schedule: %d rounds, max halo cells/edges per round: %s / %s",
            n_rounds,
            pp_sched['halo_cells_per_round'],
            pp_sched['halo_edges_per_round'],
        )
        logger.info(
            "  comm volume per stage: ppermute ~%.1f KB vs allgather ~%.1f KB (%.1fx reduction)",
            total_pp_bytes / 1024,
            ag_bytes / 1024,
            ag_bytes / max(total_pp_bytes, 1),
        )
        logger.info("  ppermute schedule built in %.3fs", time.time() - t1)

        # ---- ppermute-based shard_map kernel ----

        def _local_tendency(u_shard, T_shard, ps_shard, phis_shard, dt_val):
            """Inside shard_map: ppermute halo exchange → local tendency."""
            dev_idx = jax.lax.axis_index("device")

            # Pack cell fields into a single buffer: (cells_per, nlev+2)
            cell_pack = jnp.concatenate([
                T_shard,                           # (cells_per, nlev)
                ps_shard[:, jnp.newaxis],          # (cells_per, 1)
                phis_shard[:, jnp.newaxis],        # (cells_per, 1)
            ], axis=-1)

            # Initialize local arrays with +1 garbage slot for safe
            # padding.  Single Pad HLO op replaces alloc-zeros +
            # scatter; subsequent halo scatters write into the zeroed
            # tail slots.
            cell_local = jnp.pad(
                cell_pack, ((0, max_lc + 1 - cells_per), (0, 0)),
            )
            u_local = jnp.pad(
                u_shard, ((0, max_le + 1 - edges_per), (0, 0)),
            )

            # Exchange halos via ppermute rounds (one per edge-color).
            # Cell and edge data are packed into a single flat buffer per
            # round so that each round issues ONE ppermute instead of two,
            # halving NCCL collective overhead.
            for r in range(n_rounds):
                # Gather send buffers
                sc_idx = send_cell_idx[r][dev_idx]   # (halo_c_r,)
                se_idx = send_edge_idx[r][dev_idx]   # (halo_e_r,)
                send_c = cell_pack[sc_idx]            # (hc, nlev+2)
                send_e = u_shard[se_idx]              # (he, nlev)

                # Pack into single flat buffer for one ppermute
                send_c_flat = send_c.ravel()
                send_e_flat = send_e.ravel()
                send_packed = jnp.concatenate([send_c_flat, send_e_flat])

                recv_packed = jax.lax.ppermute(
                    send_packed, "device", perm=ppermute_perms[r])

                # Unpack: split at the cell/edge boundary and reshape
                split_at = send_c_flat.shape[0]  # hc * (nlev+2), static
                recv_c = recv_packed[:split_at].reshape(send_c.shape)
                recv_e = recv_packed[split_at:].reshape(send_e.shape)

                # Scatter received data into halo positions
                # (padding entries target the garbage slot at max_lc/max_le)
                rc_pos = recv_cell_pos[r][dev_idx]   # (halo_c_r,)
                re_pos = recv_edge_pos[r][dev_idx]   # (halo_e_r,)
                cell_local = cell_local.at[rc_pos].set(recv_c)
                u_local = u_local.at[re_pos].set(recv_e)

            # Trim garbage slot
            cell_local = cell_local[:max_lc]
            u_local = u_local[:max_le]

            # Unpack cell fields
            T_local = cell_local[:, :nlev]
            ps_local = cell_local[:, nlev]
            phis_local = cell_local[:, nlev + 1]

            # Get this device's local mesh
            my_mesh = jax.tree.map(lambda x: x[dev_idx], stacked_meshes)

            # Build local state and compute tendency
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
                           long_name="surface geopotential",
                           staggering="cell"),
            )
            tend = mpas_hydrostatic_tendencies(
                local_state, my_mesh, sigma, cfg, dt=dt_val,
            )

            # Return only the owned shard of the tendency
            return (tend.du_dt.data[:edges_per],
                    tend.dT_dt.data[:cells_per],
                    tend.dp_s_dt.data[:cells_per])

    else:
        # ---- Legacy all-gather shard_map kernel ----
        gather_cells_rep = jax.device_put(gather_cells, rep_sharding)
        gather_edges_rep = jax.device_put(gather_edges, rep_sharding)

        def _local_tendency(u_shard, T_shard, ps_shard, phis_shard, dt_val):
            """Inside shard_map: all-gather → local gather → tendency."""
            cell_pack = jnp.concatenate([
                T_shard,
                ps_shard[:, jnp.newaxis],
                phis_shard[:, jnp.newaxis],
            ], axis=-1)
            cell_full = jax.lax.all_gather(
                cell_pack, "device", axis=0, tiled=True)
            u_full = jax.lax.all_gather(
                u_shard, "device", axis=0, tiled=True)

            dev_idx = jax.lax.axis_index("device")
            cell_local = cell_full[gather_cells_rep[dev_idx]]
            u_local = u_full[gather_edges_rep[dev_idx]]

            T_local = cell_local[:, :nlev]
            ps_local = cell_local[:, nlev]
            phis_local = cell_local[:, nlev + 1]

            my_mesh = jax.tree.map(lambda x: x[dev_idx], stacked_meshes)

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
                           long_name="surface geopotential",
                           staggering="cell"),
            )
            tend = mpas_hydrostatic_tendencies(
                local_state, my_mesh, sigma, cfg, dt=dt_val,
            )
            return (tend.du_dt.data[:edges_per],
                    tend.dT_dt.data[:cells_per],
                    tend.dp_s_dt.data[:cells_per])

    _shard_tendency = shard_map(
        _local_tendency,
        mesh=jax_mesh,
        in_specs=(P("device"), P("device"), P("device"), P("device"), P()),
        out_specs=(P("device"), P("device"), P("device")),
        check_vma=False,
    )

    # ------------------------------------------------------------------
    # Pre-compute mass conservation constants (avoid per-step allreduce)
    # ------------------------------------------------------------------
    if cfg.fix_mass:
        _area_for_mass = jax.device_put(global_mesh.areaCell, face_sharding)
        _total_area = float(jnp.sum(global_mesh.areaCell))

    # ------------------------------------------------------------------
    # JIT-compiled step: SSP-RK3 with halo refresh between stages
    # ------------------------------------------------------------------

    def _build_step(physics_fn=None):
        """Build one jitted step executable.

        ``physics_fn`` is captured in the closure — JAX cannot trace a
        Python callable as an array argument (same convention as
        ``CompiledShardedStep._compile`` / ``_SingleDeviceStep``).
        ``physics_fn=None`` produces exactly the dynamics-only graph.
        """
        _phys = physics_fn

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

            # --- Operator-split physics (mirrors
            #     MPASPrimitiveEquationModel._step_jit): evaluate ONCE on
            #     the post-dynamics state and apply forward over dt,
            #     BEFORE the temperature floor and the mass fix.  Column
            #     physics is cell/edge-local, so it runs on the sharded
            #     global arrays OUTSIDE the shard_map kernel — GSPMD
            #     partitions the pointwise work per device with no halo
            #     traffic. ---
            if _phys is not None:
                post_dyn = MPASHydrostaticState(
                    u=state.u.replace(data=u_new),
                    T=state.T.replace(data=T_new),
                    p_s=state.p_s.replace(data=ps_new),
                    phis=state.phis,
                )
                _pt = _phys(post_dyn, global_mesh, sigma,
                            phys_state=None, forcing=None)
                if type(_pt) is tuple:
                    # (tendencies, phys_state_out) is the stateful-physics
                    # convention; this step has no physics-state carry
                    # channel and silently dropping the carry would
                    # corrupt stateful schemes (TKE etc.).  Trace-time
                    # Python check → loud failure, never a wrong answer.
                    raise TypeError(
                        "make_voronoi_sharded_step: physics_fn returned a "
                        "(tendencies, phys_state) tuple, but the sharded "
                        "Voronoi step has no physics-state carry channel. "
                        "Use a stateless physics_fn returning bare "
                        "tendencies (e.g. held_suarez_forcing_mpas)."
                    )
                u_new = u_new + dt * _pt.du_dt.data
                T_new = T_new + dt * _pt.dT_dt.data
                ps_new = ps_new + dt * _pt.dp_s_dt.data

            # --- Post-processing (mirrors model.step) ---
            if cfg.T_min > 0:
                T_new = jnp.maximum(T_new, cfg.T_min)

            if cfg.fix_mass:
                # Compute both masses inside a single reduction.  Stacking
                # the two ps fields and reducing once lets XLA fuse the
                # two cross-device sums into a single allreduce HLO instead
                # of emitting two sequentially-dependent allreduces (the
                # second cannot start until the first materialises).
                ps_pair = jnp.stack([ps, ps_new], axis=0)
                masses = jnp.sum(ps_pair * _area_for_mass[None], axis=tuple(
                    range(1, ps_pair.ndim)
                ))  # shape (2,)
                correction = (masses[0] - masses[1]) / _total_area
                ps_new = ps_new + correction

            return MPASHydrostaticState(
                u=state.u.replace(data=u_new),
                T=state.T.replace(data=T_new),
                p_s=state.p_s.replace(data=ps_new),
                phis=state.phis,
            )

        return _step

    # Executable cache keyed by physics_fn identity (same convention as
    # ``_make_cache_key``: distinct callables ⇒ distinct XLA programs; a
    # stable callable ⇒ exactly one compile).  Each cached executable's
    # closure holds a strong reference to its physics_fn, so an id()
    # cannot be recycled while its cache entry is alive.
    _step_cache: dict = {}

    def _voronoi_step(state, dt, physics_fn=None):
        """Sharded Voronoi step.  ``physics_fn`` is closure-captured into
        the jitted executable (selected by object identity) — it is never
        passed to ``jax.jit`` as a traced argument."""
        key = None if physics_fn is None else id(physics_fn)
        fn = _step_cache.get(key)
        if fn is None:
            fn = _build_step(physics_fn)
            _step_cache[key] = fn
        return fn(state, dt)

    return _voronoi_step


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
