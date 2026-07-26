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
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from legoesm.core.field import Field
from legoesm.grids.halo import pad_halo, pad_halo_4d
from legoesm.parallel.mesh import N_FACES, DeviceConfig

logger = logging.getLogger(__name__)


def _refuse_stateful_physics_unthreaded_wrapper(physics_fn) -> None:
    """Refuse stateful physics on step wrappers that drop the carry.

    Issue #405/#413: these generic sharded wrappers call
    ``model.step_with_physics(s, dt, physics_fn)`` with no
    ``phys_state`` — a stateful physics_fn (prognostic TKE-family /
    MYNN-2.5 turbulence, mass_flux/edmf/bechtold convection,
    prognostic-spectral GWD) would silently reseed its prognostic
    fields every step.  ``combined.make_physics`` tags its output with
    ``_requires_phys_state``; refuse loudly when the tag is set.  Use
    the shared wrapper-aware predicate so a ``functools.partial`` /
    ``__wrapped__`` wrapper that hides the tag cannot slip through.
    """
    from legoesm.timestepping.integration import (
        physics_requires_phys_state,
    )
    if physics_requires_phys_state(physics_fn):
        raise NotImplementedError(
            "This sharded step wrapper does not thread the PhysicsState "
            "carry, so the configured stateful physics would silently "
            "reseed every step (issue #405/#413).  Use a diagnostic "
            "scheme, the ModelDriver loops / MPAS step, or "
            "make_voronoi_sharded_step(return_phys_state=True), which "
            "thread the carry."
        )


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
                    # Staggered D-grid leaves (n+1 on a horizontal
                    # axis) cannot take tile specs (IndivisibleError;
                    # P4 phase-1 policy matches mesh.shard_pytree):
                    # face-only sharding, body-side block slicing via
                    # staggered_tile_block.
                    tx, ty = config.tiling
                    if (leaf.shape[1] % tx != 0
                            or leaf.shape[2] % ty != 0):
                        pspec = P("face",
                                  *([None] * (leaf.ndim - 1)))
                        return NamedSharding(mesh, pspec)
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
            _refuse_stateful_physics_unthreaded_wrapper(physics_fn)
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
                _refuse_stateful_physics_unthreaded_wrapper(physics_fn)
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
    # UPDATE (2026-06-14): the tile-aware CONSUMERS now EXIST and are
    # bit-identity-validated standalone — the full tiled production SW
    # tendency ``make_tiled_fv3_sw_tendencies_stage_2d`` (momentum + mass-PPM,
    # in-stage scalar/vector halos + deep-h pre-pad, staggered-leaf
    # lower-owns-shared reassembly) passes np24 (kt=2) + np54 (kt=3)
    # bit-identity vs the global op AND a 2-node multi-controller run
    # (``scripts/validate/validate_tiled_fv3_sw_multinode.py``, rel=0.0).  What
    # remains for the default-flip is WIRING that stage into THIS step (this
    # function still calls the full-face operators); the 3D
    # ``fv3_hydrostatic_tendencies`` tiling is in progress (dgrid_vorticity
    # 4D-tiled).  See ``tiled_production_cdgrid.py`` +
    # ``docs/performance/scaling/cube_production_tiling_design.md``.  NOT Ginsburg-benchable
    # (np>6 anti-scales on Gloo-TCP/PCIe) — future-HW capability.
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
    if physics_fn is not None:
        _refuse_stateful_physics_unthreaded_wrapper(physics_fn)

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
        HaloCommSchedule,
        VoronoiPartition,
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


def _greedy_edge_coloring_ordered(comm_pairs, order):
    """First-fit edge coloring visiting ``order`` (a list of normalized
    ``(min,max)`` pairs). Always a PROPER coloring; the color count depends
    on the visitation order.
    """
    from collections import defaultdict

    vertex_colors: dict[int, set[int]] = defaultdict(set)
    edge_colors: dict[tuple[int, int], int] = {}
    for u, v in order:
        used = vertex_colors[u] | vertex_colors[v]
        color = 0
        while color in used:
            color += 1
        edge_colors[(u, v)] = color
        vertex_colors[u].add(color)
        vertex_colors[v].add(color)
    return edge_colors


def _greedy_edge_coloring(comm_pairs):
    """Legacy first-fit coloring on sorted pairs (the reference/never-regress
    baseline for :func:`_multi_ordering_edge_coloring`). Worst case
    ``2*max_degree - 1`` colors — each color is one ppermute ROUND, and the
    route-B MPAS lane is round-latency-bound (#1113), so excess colors are
    pure wall-clock.
    """
    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
    return _greedy_edge_coloring_ordered(comm_pairs, edges)


def _check_proper_edge_coloring(edge_colors, comm_pairs):
    """Every pair colored, and no vertex sees a color twice."""
    from collections import defaultdict

    if set(edge_colors) != {tuple(sorted(p)) for p in comm_pairs}:
        return False
    seen: dict[int, set[int]] = defaultdict(set)
    for (u, v), c in edge_colors.items():
        if c in seen[u] or c in seen[v]:
            return False
        seen[u].add(c)
        seen[v].add(c)
    return True


# Fixed shuffle seeds for the multi-start greedy edge coloring below —
# a constant so every MPI rank / process builds the byte-identical
# schedule (the coloring must agree across ranks or the ppermute pattern
# desynchronises). NOT Math.random / device randomness: this is host-side
# schedule construction, deterministic by seed.
_COLORING_SHUFFLE_SEEDS = tuple(range(16))


def _multi_ordering_edge_coloring(comm_pairs):
    """Proper edge coloring via multi-start first-fit; returns the coloring
    using the FEWEST colors (= ppermute rounds) across several deterministic
    visitation orders.

    First-fit greedy is order-sensitive: on the reordered MPAS comm graphs
    the sorted order can overshoot the chromatic index by up to 3 rounds at
    16 devices, while a degree-descending or shuffled order reaches the
    ``max_degree`` lower bound (verified optimal on ico subdivisions 3–5 ×
    {4,8,16} devices, auto/sfc partitions). Every candidate is a proper
    coloring by construction, so taking the min can NEVER produce an
    invalid schedule and can never regress below the legacy sorted greedy.

    Deterministic across ranks (sorted + degree orders + fixed-seed
    shuffles). Returns ``(edge_colors, max_degree)``.
    """
    import random
    from collections import defaultdict

    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
    deg: dict[int, int] = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    max_degree = max(deg.values(), default=0)

    orders = [
        edges,                                                   # sorted
        sorted(edges, key=lambda e: -(deg[e[0]] + deg[e[1]])),   # sum-deg desc
        sorted(edges, key=lambda e: -max(deg[e[0]], deg[e[1]])),  # max-deg desc
    ]
    for seed in _COLORING_SHUFFLE_SEEDS:
        shuffled = edges[:]
        random.Random(seed).shuffle(shuffled)
        orders.append(shuffled)

    best_colors: dict[tuple[int, int], int] | None = None
    best_rounds = None
    for order in orders:
        ec = _greedy_edge_coloring_ordered(comm_pairs, order)
        rounds = max(ec.values(), default=-1) + 1
        if best_rounds is None or rounds < best_rounds:
            best_rounds, best_colors = rounds, ec
            if best_rounds <= max_degree:
                break            # hit the chromatic-index floor — optimal
    return best_colors, max_degree


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
        n_rounds, n_rounds_greedy, max_degree, coloring_method,
        ppermute_perms, send_cell_idx, recv_cell_pos,
        send_edge_idx, recv_edge_pos, halo_cells_per_round,
        halo_edges_per_round.
    """
    from collections import defaultdict

    import numpy as np

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
            'n_rounds_greedy': 0,
            'max_degree': 0,
            'coloring_method': 'none',
            'ppermute_perms': [],
            'send_cell_idx': [],
            'recv_cell_pos': [],
            'send_edge_idx': [],
            'recv_edge_pos': [],
            'halo_cells_per_round': [],
            'halo_edges_per_round': [],
        }

    # ------------------------------------------------------------------
    # 3. Edge-color the graph: each color = one bidirectional ppermute
    #    ROUND, and the route-B lane is round-latency-bound (#1113), so
    #    fewer colors = directly less wall-clock. First-fit greedy is
    #    order-sensitive; the multi-start coloring reaches the
    #    chromatic-index floor (= max_degree) on every probed MPAS config
    #    where the legacy sorted greedy overshoots (up to 3 rounds at 16
    #    devices). It can never regress: the legacy sorted order is one of
    #    its candidates and it takes the min. Both are verified proper.
    # ------------------------------------------------------------------
    greedy_colors = _greedy_edge_coloring(comm_pairs)
    n_rounds_greedy = max(greedy_colors.values()) + 1
    multi_colors, max_degree = _multi_ordering_edge_coloring(comm_pairs)
    n_rounds_multi = max(multi_colors.values()) + 1
    # Adopt the multi-start coloring ONLY when it STRICTLY reduces rounds;
    # on a tie keep the exact legacy sorted-greedy coloring so the produced
    # schedule is byte-identical to before wherever there is no round win
    # (the win only appears at high device counts — >=16 on the probed
    # MPAS meshes). Both colorings are proper.
    if n_rounds_multi < n_rounds_greedy:
        edge_colors, n_rounds, coloring_method = (
            multi_colors, n_rounds_multi, "multi_greedy")
    else:
        edge_colors, n_rounds, coloring_method = (
            greedy_colors, n_rounds_greedy, "greedy")
    assert _check_proper_edge_coloring(edge_colors, comm_pairs), (
        "improper ppermute edge coloring — two same-round exchanges "
        "would collide at a device")
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
        'n_rounds_greedy': n_rounds_greedy,
        'max_degree': max_degree,
        'coloring_method': coloring_method,
        'ppermute_perms': ppermute_perms_out,
        'send_cell_idx': send_cell_idx_out,
        'recv_cell_pos': recv_cell_pos_out,
        'send_edge_idx': send_edge_idx_out,
        'recv_edge_pos': recv_edge_pos_out,
        'halo_cells_per_round': halo_cells_per_round,
        'halo_edges_per_round': halo_edges_per_round,
    }


# Schema-drift tripwire (mirrors the M3d ocean twin
# ``voronoi_mpi.exchange_state_mpas_ocean``): a NEW HydrostaticState field
# would silently ride through the packed SPMD halo exchange UNEXCHANGED
# (stale halos on every RK stage) — fail loudly so the cell-pack layout,
# the physics application and this set are extended deliberately.
# Workload signatures — (n_devices, global edge rows, global cell rows,
# nlev), all trace-time-static — where a tendency-output
# optimization_barrier is measured to pay. OBSERVED CORRELATION, not a
# proven XLA cost-model account: at ico-L8 np4 (and only there among
# np2/4/8) the compiled step carries three once-per-step 3.3-3.6 ms
# serialized loop-fusion kernels that the HLO frame table resolves to the
# RK pytree_axpy (pytree_ops.py), ~10.9 ms/step in total (nsys 26479922,
# HLO 26480096), and the barrier removes most of that: same-day ladder
# np4 17.78 -> 14.10 ms (-20.7%) with np2 +1.2% (noise) and np8 0.0%
# (jobs 26486123 dead-gate vs 26486163). An UNgated barrier regressed np2
# by 10.7% in the earlier experiment (26480310), hence the gate. The
# signature includes cell rows + nlev because L8's edge count
# (1,966,080, unpadded — padding pads CELLS, e.g. 655,362 -> 655,376 for
# 16) is divisible several ways and edge rows alone would fire on
# unmeasured workloads (codex round-12). Grow ONLY with a measured
# receipt for the exact signature.
_FUSION_BARRIER_WORKLOADS = frozenset({(4, 1_966_080, 655_376, 26)})

_VORONOI_SPMD_STATE_FIELDS = frozenset(
    {"u", "T", "p_s", "phis", "v", "tracers"})


def check_voronoi_spmd_state_schema(state) -> tuple:
    """Validate the MPAS state schema for the sharded SPMD step.

    Raises on (a) a ``HydrostaticState`` field-set drift (a new field
    must be threaded through the packed exchange deliberately) and
    (b) a non-None ``v`` (MPAS carries the wind as edge-normal ``u``
    only; a ``v`` would be silently dropped by the pack).

    Returns the canonical tracer WIRE order (``sorted(keys)``) — every
    device packs/unpacks tracers in the same order even if dict insertion
    order ever diverged, so same-dtype tracers can never swap silently
    (same guard as ``make_voronoi_mpi_step``'s ``_exchange_mpas_state``).
    """
    if set(state._fields) != set(_VORONOI_SPMD_STATE_FIELDS):
        raise ValueError(
            "make_voronoi_sharded_step: HydrostaticState schema changed "
            f"({sorted(set(state._fields) ^ set(_VORONOI_SPMD_STATE_FIELDS))}"
            "); extend the packed SPMD halo exchange (cell-pack layout), "
            "the physics application, and _VORONOI_SPMD_STATE_FIELDS "
            "deliberately."
        )
    if state.v is not None:
        raise ValueError(
            "make_voronoi_sharded_step: state.v must be None on MPAS "
            "(the wind is edge-normal u); a non-None v would be silently "
            "dropped by the packed halo exchange."
        )
    return tuple(sorted(state.tracers)) if state.tracers is not None else ()


def _pack_cell_state(T, p_s, phis, q_flat):
    """Production cell-pack WIRE layout: ``T | p_s | phis | tracers``.

    ``q_flat`` is the tracer block ``(n, nlev * n_q)`` concatenated in
    the canonical SORTED-key order (width 0 for a dry run).  The single
    source of truth for the packed exchange layout — the shard_map
    kernel and the sentinel routing test both go through here, so an
    omitted field or a swapped slot cannot hide in a hand-rolled copy.
    """
    return jnp.concatenate(
        [T, p_s[:, jnp.newaxis], phis[:, jnp.newaxis], q_flat], axis=-1)


def _unpack_cell_state(cell_buf, nlev):
    """Inverse of :func:`_pack_cell_state`: ``(T, p_s, phis, q_flat)``."""
    return (cell_buf[:, :nlev], cell_buf[:, nlev],
            cell_buf[:, nlev + 1], cell_buf[:, nlev + 2:])


def _ppermute_halo_fill(cell_pack, u_shard, halo_sl, ppermute_perms,
                        max_lc, max_le):
    """Fill (owned + halo) local buffers from owned shards via ppermute.

    Runs INSIDE ``shard_map``.  ``cell_pack`` ``(cells_per, W)`` is the
    packed owned-cell buffer — ALL cell-centred prognostics (T | p_s |
    phis | tracers) concatenated on the trailing axis; ``u_shard``
    ``(edges_per, nlev)`` the owned-edge buffer.  Each edge-colored round
    posts ONE flat ppermute carrying BOTH entity classes for ALL packed
    fields — the SPMD mirror of route-A's batched union-neighbor exchange
    (one message per neighbor per dtype group; the compute-precision cast
    upstream guarantees a single dtype group here).

    ``halo_sl`` is a tuple of per-round ``(send_cell_idx, recv_cell_pos,
    send_edge_idx, recv_edge_pos)`` tuples whose arrays are ALREADY
    device-local ``(1, n_round)`` shard_map arguments (``P("device")``
    specs) — per-rank LOCAL metadata; no device materializes the global
    schedule.  ``ppermute_perms`` is the static per-round permutation.

    Returns ``(cell_local, u_local)`` of shapes ``(max_lc, W)`` /
    ``(max_le, nlev)``; ghost tail rows stay zero.
    """
    cells_per = cell_pack.shape[0]
    edges_per = u_shard.shape[0]
    # +1 garbage slot for padded scatter targets (trimmed at the end):
    # schedule rows are padded to the round's max halo count, and padding
    # entries target position max_lc / max_le.
    cell_local = jnp.pad(cell_pack, ((0, max_lc + 1 - cells_per), (0, 0)))
    u_local = jnp.pad(u_shard, ((0, max_le + 1 - edges_per), (0, 0)))

    for r, (sc, rc, se, re) in enumerate(halo_sl):
        send_c = cell_pack[sc[0]]             # (hc_r, W)
        send_e = u_shard[se[0]]               # (he_r, nlev)
        send_c_flat = send_c.ravel()
        send_packed = jnp.concatenate([send_c_flat, send_e.ravel()])
        recv_packed = jax.lax.ppermute(
            send_packed, "device", perm=ppermute_perms[r])
        split_at = send_c_flat.shape[0]       # static
        recv_c = recv_packed[:split_at].reshape(send_c.shape)
        recv_e = recv_packed[split_at:].reshape(send_e.shape)
        cell_local = cell_local.at[rc[0]].set(recv_c)
        u_local = u_local.at[re[0]].set(recv_e)

    return cell_local[:max_lc], u_local[:max_le]


def make_voronoi_sharded_step(
    model,
    dev_config: DeviceConfig,
    *,
    halo_strategy: str = "auto",
    ppermute_cells_per_device_threshold: int = 2_000,
    return_phys_state: bool = False,
):
    """Create a halo-partitioned multi-GPU step for Voronoi (MPAS/TRiSK) grids.

    Instead of replicating the full state and redundantly computing the
    full step on every device, this implementation:

    1. Pre-computes per-device local meshes (owned cells/edges + halo)
       at setup time via domain decomposition.
    2. At each RK stage (``config.time_integrator`` via
       ``dispatch_integrator`` — same integrator code as the serial
       ``_step_jit``), exchanges only halo data between neighboring
       devices (not the full state), then computes tendencies on the
       local mesh.  The packed exchange carries the FULL prognostic
       state: u (edge) plus T, p_s, phis and every tracer (cell) in one
       flat ppermute payload per neighbor round.
    3. Applies operator-split physics ONCE on the post-dynamics state
       (traced ``forcing`` + prognostic ``phys_state`` carry threaded
       through), then the temperature/tracer floors and the global mass
       fix — mirroring the serial ``MPASPrimitiveEquationModel._step_jit``
       operator ordering exactly.

    Local-only metadata: the per-device local meshes (stacked with a
    leading device axis), the ppermute schedule index arrays, and the
    mass-fix ``areaCell`` are ``P("device")``-sharded and passed as
    ARGUMENTS into the jitted step (multi-controller-safe: sharded jit
    args are legal where sharded closure constants raise at trace time)
    — each device holds ONLY its own local mesh + schedule rows, never
    the global connectivity.  The one remaining NON-local metadata is
    the global-mesh closure handed to the operator-split physics term:
    column-local physics runs OUTSIDE shard_map on the GSPMD-sharded
    global arrays, reading only replicated 1-D cell fields (latCell
    etc. — O(nCells) scalars, not the 2-D connectivity).  See the
    physics block below and
    ``docs/performance/scaling/mpas_atm_native_step_audit.md``.

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
    return_phys_state : bool
        ``False`` (default, backward-compatible): the returned step is
        ``step(state, dt, physics_fn=None, forcing=None, phys_state=None)
        -> state`` — the physics carry is dropped, so a STATEFUL
        physics_fn is refused loudly (issue #405/#413).  ``True``: the
        step returns ``(state, phys_state_out)`` — full operator-split
        production parity with ``make_voronoi_mpi_step(
        return_phys_state=True)``; the prognostic physics carry (TKE /
        convection state) and the traced per-step ``forcing`` (e.g.
        prescribed ``T_sfc``) are threaded through.

    Returns
    -------
    callable
        ``step(state, dt, physics_fn=None, forcing=None, phys_state=None)``
        returning ``state`` (``return_phys_state=False``) or
        ``(state, phys_state_out)`` (``return_phys_state=True``).
        ``physics_fn`` follows the MPAS operator-split convention
        (``physics_fn(state, mesh, sigma_coord, *, phys_state, forcing)``
        returning ``MPASHydrostaticTendencies`` or a ``(tendencies,
        phys_state_out)`` tuple) and is captured in the jitted closure,
        never traced as an argument (same convention as
        :class:`CompiledShardedStep`).  Each distinct ``physics_fn``
        identity compiles a separate executable; ``physics_fn=None``
        compiles exactly the dynamics-only graph.  ``forcing`` /
        ``phys_state`` are jit arguments (NOT static) so new values each
        step do not retrace (SegmentForcing doctrine); their pytree
        STRUCTURE must stay stable across steps.  On a single-device
        config this returns ``model.step``, whose signature is
        call-compatible (state-only contract).
    """
    if dev_config.n_devices <= 1 or dev_config.mesh is None:
        if return_phys_state:
            # model.step returns only the state and stashes the carry on
            # the model EAGERLY (skipped under an outer trace, gh-417) —
            # returning it here would silently drop/reseed the carry
            # inside scan-driven callers.  Refuse loudly; the serial
            # carry contract is the model/driver's own.
            raise ValueError(
                "make_voronoi_sharded_step(return_phys_state=True) needs "
                "a multi-device config; on a single device use "
                "model.step (eager, carry stashed on the model) or the "
                "ModelDriver loop, which threads the carry."
            )
        return model.step

    from legoesm.core.precision import cast_pytree
    from legoesm.core.state import MPASHydrostaticState
    from legoesm.parallel.mesh import multiprocess_safe_device_put
    from legoesm.parallel.shard_map_compat import shard_map
    from legoesm.timestepping.dispatch import dispatch_integrator
    from legoesm.timestepping.integration import (
        refuse_unthreaded_stateful_physics,
    )

    n_dev = dev_config.n_devices
    voronoi_dims = dev_config.voronoi_dims
    if voronoi_dims is None:
        raise ValueError("dev_config.voronoi_dims must be set for Voronoi grids")
    nCells, nEdges, _nVerts = voronoi_dims
    jax_mesh = dev_config.mesh

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

    # LOCAL-ONLY metadata: shard the stacked local meshes on the leading
    # device axis — device i holds ONLY its own local mesh (leaf slice
    # [i]), never the other devices' connectivity.  The mesh rides into
    # the jitted step as an ARGUMENT with P("device") shard_map in_specs
    # (multi-controller-safe: a sharded jit ARG is legal where a sharded
    # CLOSURE constant raises at trace time under jax.distributed;
    # ``multiprocess_safe_device_put`` builds the global array from each
    # process's local copy).  All leaves are arrays after the jnp.stack
    # in _build_voronoi_partition_infra (ints become (n_dev,) arrays).
    dev_sharding = dev_config.face_sharding  # P("device") on axis 0
    stacked_meshes = jax.tree.map(
        lambda x: multiprocess_safe_device_put(x, dev_sharding),
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

        # LOCAL-ONLY metadata: shard the per-round index arrays on the
        # leading device axis (each device holds only its own schedule
        # rows) and thread them as shard_map ARGUMENTS — see the stacked
        # meshes above for why args, not closures.
        halo_args = tuple(
            (
                multiprocess_safe_device_put(
                    pp_sched['send_cell_idx'][r], dev_sharding),
                multiprocess_safe_device_put(
                    pp_sched['recv_cell_pos'][r], dev_sharding),
                multiprocess_safe_device_put(
                    pp_sched['send_edge_idx'][r], dev_sharding),
                multiprocess_safe_device_put(
                    pp_sched['recv_edge_pos'][r], dev_sharding),
            )
            for r in range(n_rounds)
        )

        # Log halo exchange statistics (dry-state estimate: tracers add
        # nlev*n_tracers further cell channels to both strategies).
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

    else:
        # ---- Legacy all-gather strategy: local gather indices ----
        halo_args = (
            multiprocess_safe_device_put(gather_cells, dev_sharding),
            multiprocess_safe_device_put(gather_edges, dev_sharding),
        )

    # ------------------------------------------------------------------
    # shard_map kernel: packed full-state halo fill → local tendency
    # ------------------------------------------------------------------
    # The kernel is built per canonical tracer-key tuple (the keys are
    # part of the traced program: cell-pack width and the tracer dict
    # rebuilt on the local mesh).  Memoized so a stable state structure
    # reuses one shard_map object → one jit executable (no retrace).

    mesh_in_specs = jax.tree.map(lambda _: P("device"), stacked_meshes)
    halo_in_specs = jax.tree.map(lambda _: P("device"), halo_args)

    def _make_local_tendency(tkeys: tuple):

        def _local_tendency(u_shard, T_shard, ps_shard, phis_shard,
                            q_shard, dt_val, mesh_sl, halo_sl):
            """Inside shard_map: full-state halo fill → local tendency.

            ``q_shard`` is the tracer block ``(cells_per, nlev * n_q)``
            — tracers concatenated on the trailing axis in the canonical
            sorted-key WIRE order (width 0 for a dry run).  ``mesh_sl``
            / ``halo_sl`` are this device's P("device") slices of the
            stacked local meshes and the halo schedule (leading axis 1).
            """
            # Pack ALL cell-centred prognostics into a single buffer
            # (cells_per, nlev + 2 + nlev*n_q) via the shared wire-layout
            # helper (also driven directly by the sentinel routing test).
            cell_pack = _pack_cell_state(T_shard, ps_shard, phis_shard,
                                         q_shard)

            if use_ppermute:
                cell_local, u_local = _ppermute_halo_fill(
                    cell_pack, u_shard, halo_sl, ppermute_perms,
                    max_lc, max_le,
                )
            else:
                cell_full = jax.lax.all_gather(
                    cell_pack, "device", axis=0, tiled=True)
                u_full = jax.lax.all_gather(
                    u_shard, "device", axis=0, tiled=True)
                gc, ge = halo_sl
                cell_local = cell_full[gc[0]]
                u_local = u_full[ge[0]]

            # Unpack cell fields (inverse of the shared pack helper)
            T_local, ps_local, phis_local, q_local = _unpack_cell_state(
                cell_local, nlev)

            # This device's local mesh (leading axis is the length-1
            # device slice of the stacked meshes).
            my_mesh = jax.tree.map(lambda x: x[0], mesh_sl)

            tracers_local = None
            if tkeys:
                tracers_local = {
                    k: Field(data=q_local[:, i * nlev:(i + 1) * nlev],
                             name=k, dims=("nCells", "nlev"),
                             units="kg/kg", staggering="cell")
                    for i, k in enumerate(tkeys)
                }

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
                tracers=tracers_local,
            )
            tend = mpas_hydrostatic_tendencies(
                local_state, my_mesh, sigma, cfg, dt=dt_val,
            )

            # Owned shards only.  Tracer ADVECTION tendencies ride back
            # in the same packed wire order (mpas_hydrostatic_tendencies
            # always returns tracer_tendencies for a tracered state).
            if tkeys:
                dq_owned = jnp.concatenate(
                    [tend.tracer_tendencies[k].data for k in tkeys],
                    axis=-1)[:cells_per]
            else:
                dq_owned = jnp.zeros((cells_per, 0), dtype=T_shard.dtype)
            return (tend.du_dt.data[:edges_per],
                    tend.dT_dt.data[:cells_per],
                    tend.dp_s_dt.data[:cells_per],
                    dq_owned)

        return _local_tendency

    _shard_tendency_cache: dict = {}

    def _get_shard_tendency(tkeys: tuple):
        fn = _shard_tendency_cache.get(tkeys)
        if fn is None:
            fn = shard_map(
                _make_local_tendency(tkeys),
                mesh=jax_mesh,
                in_specs=(P("device"), P("device"), P("device"),
                          P("device"), P("device"), P(),
                          mesh_in_specs, halo_in_specs),
                out_specs=(P("device"), P("device"), P("device"),
                           P("device")),
                check_vma=False,
            )
            _shard_tendency_cache[tkeys] = fn
        return fn

    # ------------------------------------------------------------------
    # Pre-compute mass conservation constants (avoid per-step allreduce)
    # ------------------------------------------------------------------
    if cfg.fix_mass:
        # areaCell rides as a P("device")-sharded jit ARGUMENT aligned
        # with the p_s cell shards (elementwise product stays local;
        # GSPMD emits one allreduce for the sum) — local-only, and
        # multi-controller-safe because it is an argument, not a closure
        # constant.  Take a HOST copy first (codex M3c-1 MAJOR): the
        # caller's mesh may arrive REPLICATED (bench replicate_pytree),
        # and multiprocess_safe_device_put passes non-fully-addressable
        # arrays through UNCHANGED — a replicated leaf would silently
        # stay replicated under multi-controller.  A host array is
        # always fully addressable, so the P("device") shard is
        # guaranteed on both controllers.  total_area is a host float.
        _area_for_mass = multiprocess_safe_device_put(
            np.asarray(global_mesh.areaCell), dev_sharding)
        # fp64 area sum to match the fp64 mass-budget accumulator below
        # (mirrors make_voronoi_mpi_step; identical under x64).
        _total_area = float(jnp.sum(
            global_mesh.areaCell.astype(jnp.float64)))
    else:
        _area_for_mass = jnp.zeros((0,))  # unused placeholder arg
        _total_area = 1.0

    # ------------------------------------------------------------------
    # JIT-compiled step: dispatch_integrator dynamics (tracer advection
    # included) → operator-split physics → floors → global mass fix.
    # Mirrors the serial MPASPrimitiveEquationModel._step_jit ordering.
    # ------------------------------------------------------------------

    def _build_step(physics_fn=None):
        """Build one jitted step executable.

        ``physics_fn`` is captured in the closure — JAX cannot trace a
        Python callable as an array argument (same convention as
        ``CompiledShardedStep._compile`` / ``_SingleDeviceStep``).
        ``physics_fn=None`` produces exactly the dynamics-only graph.
        ``forcing`` / ``phys_state`` are traced jit arguments (NOT
        static) so per-step values do not retrace (SegmentForcing
        doctrine); the local-mesh / halo-schedule / area constants are
        threaded as sharded arguments (see the factory docstring).
        """
        _phys = physics_fn

        @jax.jit
        def _step(state, dt, forcing, phys_state,
                  mesh_arg, halo_arg, area_arg):
            # Canonical tracer wire order — static at trace time (part
            # of the state's pytree structure).
            tkeys = (tuple(sorted(state.tracers))
                     if state.tracers is not None else ())

            # Precision parity with the serial ``_step_jit``: integrate
            # in compute precision, store in storage precision.
            state = cast_pytree(state, None, "compute")
            shard_tendency = _get_shard_tendency(tkeys)

            def _pack_tracers(s):
                if not tkeys:
                    return jnp.zeros(
                        s.T.data.shape[:-1] + (0,), dtype=s.T.data.dtype)
                return jnp.concatenate(
                    [s.tracers[k].data for k in tkeys], axis=-1)

            def dyn_tendency_fn(s):
                """Dynamics tendencies as a state-shaped pytree (tracer
                ADVECTION rides under the same tracer keys) so the pytree
                RK integrator advances moisture mass-consistently with
                u/T/p_s — mirroring the serial ``dyn_tendency_fn``."""
                du, dT, dps, dq = shard_tendency(
                    s.u.data, s.T.data, s.p_s.data, s.phis.data,
                    _pack_tracers(s), dt, mesh_arg, halo_arg,
                )
                # Static workload-gated fusion barrier (codex rounds
                # 11-12; see _FUSION_BARRIER_WORKLOADS). Every gate
                # operand is trace-time static (closure int + aval
                # shapes) — no retrace; the barrier is an identity for
                # numerics and AD.
                _sig = (n_dev, s.u.data.shape[0],
                        s.T.data.shape[0], s.T.data.shape[1])
                if _sig in _FUSION_BARRIER_WORKLOADS:
                    du, dT, dps, dq = jax.lax.optimization_barrier(
                        (du, dT, dps, dq))
                tr_tend = None
                if s.tracers is not None:
                    tr_tend = {
                        k: s.tracers[k].replace(
                            data=dq[..., i * nlev:(i + 1) * nlev])
                        for i, k in enumerate(tkeys)
                    }
                return MPASHydrostaticState(
                    u=s.u.replace(data=du),
                    T=s.T.replace(data=dT),
                    p_s=s.p_s.replace(data=dps),
                    phis=s.phis.replace(
                        data=jnp.zeros_like(s.phis.data)),
                    v=s.v,
                    tracers=tr_tend,
                )

            # --- 1. Dynamics: RK-integrate (physics OFF; tracer
            #        advection stays in the dynamics).  Honors
            #        config.time_integrator via the SAME dispatch the
            #        serial step uses (the previous hard-coded SSP-RK3
            #        silently overrode e.g. the ssp_rk54_scan default).
            state_new = dispatch_integrator(
                state, dyn_tendency_fn, dt, cfg.time_integrator,
            )

            # --- 2. Operator-split physics (mirrors _step_jit): evaluate
            #     ONCE on the post-dynamics state and apply forward over
            #     dt, BEFORE the floors and the mass fix.  Column physics
            #     is cell/edge-local, so it runs on the sharded global
            #     arrays OUTSIDE the shard_map kernel — GSPMD partitions
            #     the pointwise work per device with no halo traffic (the
            #     global-mesh closure contributes only replicated 1-D
            #     cell fields such as latCell). ---
            phys_state_out = phys_state
            if _phys is not None:
                _pr = _phys(state_new, global_mesh, sigma,
                            phys_state=phys_state, forcing=forcing)
                # NB ``type(...) is tuple`` (not isinstance): tendencies
                # are themselves a NamedTuple — mirror the serial guard.
                if type(_pr) is tuple:
                    if not return_phys_state:
                        # Stateful-physics convention with no carry
                        # channel armed: silently dropping the carry
                        # would corrupt stateful schemes (TKE etc.).
                        # Trace-time Python check → loud failure.
                        raise TypeError(
                            "make_voronoi_sharded_step: physics_fn "
                            "returned a (tendencies, phys_state) tuple, "
                            "but the step was built with "
                            "return_phys_state=False (no carry channel). "
                            "Rebuild with return_phys_state=True and "
                            "thread the returned carry, or use a "
                            "stateless physics_fn returning bare "
                            "tendencies (e.g. held_suarez_forcing_mpas)."
                        )
                    _pt, phys_state_out = _pr[0], _pr[1]
                else:
                    _pt = _pr
                state_new = MPASHydrostaticState(
                    u=state_new.u.replace(
                        data=state_new.u.data + dt * _pt.du_dt.data),
                    T=state_new.T.replace(
                        data=state_new.T.data + dt * _pt.dT_dt.data),
                    p_s=state_new.p_s.replace(
                        data=state_new.p_s.data + dt * _pt.dp_s_dt.data),
                    phis=state_new.phis,
                    v=state_new.v,
                    tracers=state_new.tracers,
                )
                if (state_new.tracers is not None
                        and _pt.tracer_tendencies is not None):
                    state_new = state_new._replace(tracers={
                        k: (state_new.tracers[k].replace(
                                data=state_new.tracers[k].data
                                + dt * _pt.tracer_tendencies[k].data)
                            if k in _pt.tracer_tendencies
                            else state_new.tracers[k])
                        for k in state_new.tracers
                    })

            # --- 3. Floors (mirrors _step_jit): temperature and tracer
            #        non-negativity (advection is not positive-definite;
            #        clamp before tracers feed saturation). ---
            if cfg.T_min > 0:
                state_new = state_new._replace(
                    T=state_new.T.replace(
                        data=jnp.maximum(state_new.T.data, cfg.T_min)))
            if state_new.tracers is not None:
                state_new = state_new._replace(tracers={
                    k: f.replace(data=jnp.maximum(f.data, 0.0))
                    for k, f in state_new.tracers.items()
                })

            # --- 4. Global mass fixer ---
            if cfg.fix_mass:
                # Promote to the fp64 budget accumulator (mirrors the
                # serial fixer — plain fp32 reductions over 1e4-1e5 cells
                # leak N·eps noise), and compute both masses inside a
                # single fused reduction: stacking the two weighted ps
                # fields lets XLA emit ONE allreduce instead of two
                # sequentially-dependent ones.
                acc = jnp.float64
                area_acc = area_arg.astype(acc)
                ps_pair = jnp.stack([
                    state.p_s.data.astype(acc),
                    state_new.p_s.data.astype(acc),
                ], axis=0) * area_acc[None]
                masses = jnp.sum(ps_pair, axis=1)  # shape (2,)
                correction = (masses[0] - masses[1]) / _total_area
                # Apply in fp64, then return p_s to its pre-fix carry
                # dtype (codex M3c-1 MAJOR): under an fp32 compute state
                # with x64 enabled the fp64 correction would otherwise
                # promote the carry — the downcast-skipping storage cast
                # below cannot undo it, breaking the lax.scan carry-dtype
                # contract and re-tracing host loops.  No-op (bit
                # identical) whenever the compute state is already fp64.
                # NB the serial _fix_mass_mpas_hydro deliberately leaves
                # the promoted add (iter-11); parity in that corner mode
                # differs only by the rounding of the correction add.
                _ps = state_new.p_s.data
                state_new = state_new._replace(
                    p_s=state_new.p_s.replace(
                        data=(_ps + correction).astype(_ps.dtype)))

            return cast_pytree(state_new, None, "storage"), phys_state_out

        return _step

    # Executable cache keyed by physics_fn identity (same convention as
    # ``_make_cache_key``: distinct callables ⇒ distinct XLA programs; a
    # stable callable ⇒ exactly one compile).  Each cached executable's
    # closure holds a strong reference to its physics_fn, so an id()
    # cannot be recycled while its cache entry is alive.
    _step_cache: dict = {}

    def _voronoi_step(state, dt, physics_fn=None, forcing=None,
                      phys_state=None):
        """Sharded Voronoi step.  ``physics_fn`` is closure-captured into
        the jitted executable (selected by object identity) — it is never
        passed to ``jax.jit`` as a traced argument.  ``forcing`` /
        ``phys_state`` ARE traced jit arguments."""
        check_voronoi_spmd_state_schema(state)
        # Issue #405/#413: never silently run stateful physics without
        # its carry.  The predicates also see a partial/__wrapped__
        # wrapper that hides the tag.
        if return_phys_state:
            refuse_unthreaded_stateful_physics(
                physics_fn, phys_state,
                where="make_voronoi_sharded_step(return_phys_state=True)")
        else:
            _refuse_stateful_physics_unthreaded_wrapper(physics_fn)
        key = None if physics_fn is None else id(physics_fn)
        fn = _step_cache.get(key)
        if fn is None:
            fn = _build_step(physics_fn)
            _step_cache[key] = fn
        state_new, phys_state_out = fn(
            state, dt, forcing, phys_state,
            stacked_meshes, halo_args, _area_for_mass,
        )
        if return_phys_state:
            return state_new, phys_state_out
        return state_new

    # Effective (post-"auto") strategy, carried on the returned callable
    # so benches/tests can RECORD what actually ran instead of the
    # requested flag (codex M3c-2 MINOR; same pattern as kessler's
    # ``_bound_dt``).  Only multi-device steps carry it — the
    # single-device early return above hands back ``model.step``.
    _voronoi_step._halo_strategy_effective = halo_strategy
    return _voronoi_step


def gather_voronoi_state_spmd(state, dev_config: DeviceConfig):
    """Gather a device-sharded Voronoi state to fully-replicated arrays.

    The multi-controller counterpart of a plain ``jax.device_get``: under
    ``jax.distributed`` each process only holds its addressable shards, so
    host reads of a ``P("device")``-sharded leaf raise.  Re-laying every
    array leaf onto ``dev_config.replicated_sharding`` (via
    :func:`legoesm.parallel.latlon_spmd.replicate_leaf` — a jitted identity
    with replicated ``out_shardings``, an all-gather under GSPMD) makes the
    full global value addressable on every process for I/O / gates.

    Single-DEVICE configs (``dev_config.mesh is None``, the
    ``create_voronoi_device_mesh(n_devices=1)`` shape) pass through
    unchanged.  Multi-device single-PROCESS configs take the plain
    ``device_put`` branch of ``replicate_leaf`` (cheap, no collective).
    """
    if dev_config.mesh is None or dev_config.replicated_sharding is None:
        return state
    from legoesm.parallel.latlon_spmd import replicate_leaf

    rep = dev_config.replicated_sharding
    multi = jax.process_count() > 1

    def _gather_leaf(leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
        return replicate_leaf(leaf, rep, multiprocess=multi)

    return jax.tree.map(_gather_leaf, state)


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
