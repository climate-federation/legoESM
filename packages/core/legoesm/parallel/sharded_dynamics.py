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
import re
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

# Sub-face tile factors whose tiled step is bit-identity-validated against the
# global op. kt=2 (24 devices) and kt=3 (54) were validated standalone AND in a
# 2-node multi-controller run (rel=0.0). Anything else replicates the global
# state instead of sharding it, so it is REFUSED rather than silently run
# (#1360). Grow this set only together with the validation evidence.
VALIDATED_TILE_FACTORS = frozenset({2, 3})


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
    _tiled_requested = (
        _os.environ.get("LEGOESM_TILED_SPMD", "0") == "1"
        and _tiling[0] == _tiling[1] and _tiling[0] >= 2
        and _n == 6 * _tiling[0] * _tiling[1]
    )
    # #1360: an UNVALIDATED kt used to fall through this branch silently, which
    # left the step replicating the GLOBAL state on every device. The user then
    # saw an opaque XLA argument-size error --
    #   "The byte size of input/output arguments (83247045120) exceeds the base
    #    limit (63820333056)"  (job 26495955, C768/L60 f32 at kt=4/96 devices)
    # -- which reads as an OOM, not as "this tiling is not supported". That is
    # the silent-fallback pattern the dispatch-hardening rule exists to kill:
    # refuse loudly instead, naming what IS validated.
    if _tiled_requested and _tiling[0] not in VALIDATED_TILE_FACTORS:
        raise ValueError(
            f"tiled cube SPMD is bit-identity-validated only at kt in "
            f"{sorted(VALIDATED_TILE_FACTORS)} (6*kt^2 = "
            f"{[6 * k * k for k in sorted(VALIDATED_TILE_FACTORS)]} devices); "
            f"got kt={_tiling[0]} ({_n} devices). Running it would NOT shard: "
            f"the step falls back to replicating the global state on every "
            f"device and dies with an XLA argument-size error that looks like "
            f"an OOM (#1360). Validate that kt the way kt=2/3 were "
            f"(tiled-vs-global bit identity + a multi-controller run, "
            f"scripts/validate/validate_tiled_fv3_sw_multinode.py) and add it "
            f"to VALIDATED_TILE_FACTORS, or use a validated device count.")
    _tiled_ok = _tiled_requested
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
        subgrid_topo_stddev=(None if mesh.subgrid_topo_stddev is None
                             else pad1(mesh.subgrid_topo_stddev,
                                       pad_c, fill=0.0)),
        land_frac=(None if mesh.land_frac is None
                   else pad1(mesh.land_frac, pad_c, fill=0.0)),
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


#: Halo depth the SPMD Voronoi partition infra is built at.  ONE definition
#: consumed by both the production step factory and ``spmd_schedule_cost``:
#: a score computed at a different depth describes a different comm graph, and
#: two independently hardcoded 3s let production drift unnoticed.
SPMD_HALO_DEPTH = 3


def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
                       ppermute_cells_per_device_threshold=2_000,
                       round_profile_for_device=None):
    """How much halo communication one ownership choice costs, computed offline.

    Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
    ROUNDS one halo exchange needs -- the sequential collective launches that
    dominate MPAS strong scaling above ~64 devices.  Runs on a laptop: no GPU,
    no MPI, no benchmark job, so a split can be compared before it costs an
    allocation.

    It calls the SAME builders production calls
    (:func:`_build_voronoi_partition_infra` then
    :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
    different question: a 1-ring ``cellsOnEdge`` adjacency graph reports 8
    rounds where the real depth-3-plus-closure graph reports 12-14.

    WHAT THE NUMBER IS NOT
    ----------------------
    * ``n_rounds`` is per HALO FILL, not per model step.  A step costs
      ``n_rounds`` x (tendency evaluations per step), which depends on the
      configured integrator -- SSP-RK3 evaluates 3 times, but the MPAS default
      is ``ssp_rk54_scan``.  Multiply with the integrator you actually run.
    * ``n_rounds`` is NOT proven equal to the comm graph's ``max_degree``.
      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
      keeps the best; equality is MEASURED (compare the returned
      ``max_degree``), never assumed.  Do not claim "the colouring is already
      optimal so only ownership can help" from this function.
    * It scores the ppermute strategy.  Production auto-selects ALLGATHER when
      cells/device is below ``ppermute_cells_per_device_threshold``, in which
      case there is no ppermute schedule and this number is counterfactual --
      see the returned ``production_strategy``.

    MESH STATE -- the one thing that silently invalidates the score
    --------------------------------------------------------------
    Production does NOT reorder inside ``make_voronoi_sharded_step``; it
    consumes an already-reordered ``model.mesh``.  The scaling bench reorders
    ONCE for a ``reorder_target`` device count and then runs at a possibly
    DIFFERENT device count.  So pass what you actually have:

    * raw mesh, scoring a run at ``n_dev``: defaults are right.
    * raw mesh, but the run reorders for a different target: pass
      ``reorder_target=<that target>``; the split is built for the target and
      scored at ``n_dev``.
    * already-reordered mesh (what production holds): pass
      ``already_reordered=True``; ``method`` is then ignored and reported as
      ``"pre-reordered"``, because the ownership is already baked in.

    Parameters
    ----------
    mesh : VoronoiMesh
    n_dev : int
        Device count the run uses.  Must be >= 1.
    method : str
        Ownership for the reorder; ignored when *already_reordered*.
    reorder_target : int | None
        Device count the reorder targets, when it differs from *n_dev*.
    already_reordered : bool
    halo_depth : int
        Must match production (3) or the graph is a different graph.
    ppermute_cells_per_device_threshold : int
        Mirror of the production auto-select threshold, only used to report
        ``production_strategy``.
    round_profile_for_device : int | None
        When set to a device id, also return ``round_profile``: the per-round
        payload THAT DEVICE exchanges, in schedule order, restricted to the
        rounds it actually participates in.

        This exists to make a profiler trace interpretable.  An ``nsys``
        capture is per RANK, and a rank appears only in the colour classes
        that touch it -- at s9/np64 rank 0 shows 8 ``SendRecv`` per halo fill
        while the graph's ``max_degree`` is 11 -- so the k-th observed
        collective is the k-th round CONTAINING THAT DEVICE, not the k-th
        round.  Pairing measured durations against all rounds would silently
        misalign them.

        Each entry gives ``round`` (index in the full schedule), ``partner``
        and ``halo_cells``/``halo_edges`` -- the schedule's per-round PADDED
        extents, which are what goes on the wire: the index arrays are
        ``(n_dev, max_c)`` and ``ppermute`` moves the whole padded buffer, so
        a pair's own send count does not set its cost.  Sizes are ENTITY
        COUNTS, not bytes; converting needs the packed cell width from
        :func:`_pack_cell_state` (``nlev*(1+n_tracers)+2``) for cells and
        ``nlev`` for edges.

    Returns
    -------
    dict
        ``n_rounds`` (the cost), ``max_degree`` (the lower bound to compare
        it against), ``n_rounds_greedy``, ``coloring_method``,
        ``resolved_method`` (concrete, never ``"auto"``),
        ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
        ``max_local_cells``, ``max_local_edges``, and the echoed inputs.

    Reference census on the unrelaxed mesh, which any change here must still
    reproduce: subdiv-8 sfc 12/14 rounds at 64/128 devices, metis 13/19,
    geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18.
    """
    from legoesm.parallel.voronoi_partition import (
        reorder_voronoi_for_sharding, resolve_sharding_partition_method,
    )

    if int(n_dev) != n_dev or int(n_dev) < 1:
        # int() would silently truncate 3.9 -> 3 and score the wrong split.
        raise ValueError(
            f"spmd_schedule_cost: n_dev must be an integer >= 1, got {n_dev!r}")
    n_dev = int(n_dev)

    if already_reordered:
        if reorder_target is not None:
            raise ValueError(
                "spmd_schedule_cost: reorder_target is meaningless with "
                "already_reordered=True — the ownership is already baked into "
                "the mesh.")
        prepared, resolved = mesh, "pre-reordered"
    else:
        target = n_dev if reorder_target is None else int(reorder_target)
        prepared = reorder_voronoi_for_sharding(mesh, target, method=method)
        # Report the CONCRETE ownership: "auto" hides which partitioner ran.
        # Uses the SAME resolver the reorder used, so the label cannot drift
        # from the policy.
        resolved = resolve_sharding_partition_method(method)

    # The builder assigns residual entities to the LAST owner but excludes them
    # from every owned contiguous block, so schedule send indices can exceed a
    # device's shard length -- a number that looks fine and is not.  Reachable
    # via reorder_target: a mesh padded for 3 devices is not divisible by 4.
    # The scaling bench rejects that pairing; so does this.
    n_cells, n_edges = int(prepared.nCells), int(prepared.nEdges)
    if n_cells % n_dev or n_edges % n_dev:
        raise ValueError(
            f"spmd_schedule_cost: prepared mesh has nCells={n_cells}, "
            f"nEdges={n_edges}, neither divisible by n_dev={n_dev}. The mesh "
            f"is padded for its reorder target"
            f"{'' if already_reordered else f' ({target})'}, so scoring it at "
            f"a device count that does not divide it silently mis-slices the "
            f"owned blocks. Score at a device count that divides the prepared "
            f"mesh.")
    (
        _stacked, _gc, _ge, _noc, _noe, max_lc, max_le, partitions, cell_owner,
    ) = _build_voronoi_partition_infra(prepared, n_dev, halo_depth=halo_depth)
    cells_per = n_cells // n_dev
    edges_per = n_edges // n_dev
    sched = _build_ppermute_schedule(
        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
    )
    return {
        "method": method,
        "resolved_method": resolved,
        "n_dev": n_dev,
        # Unknown for a pre-reordered mesh: the ownership is baked in and the
        # target that produced it is not recoverable from the mesh. Reporting
        # n_dev there would assert something we did not verify.
        "reorder_target": (None if already_reordered else
                           (n_dev if reorder_target is None
                            else int(reorder_target))),
        "already_reordered": bool(already_reordered),
        "halo_depth": halo_depth,
        "n_rounds": int(sched["n_rounds"]),
        "n_rounds_greedy": int(sched["n_rounds_greedy"]),
        "max_degree": int(sched.get("max_degree", -1)),
        "coloring_method": sched["coloring_method"],
        # Production returns before selecting a strategy at n_dev==1, and a
        # caller may force halo_strategy; this reports what AUTO would pick.
        "production_strategy": (
            None if n_dev == 1 else
            ("allgather" if cells_per < ppermute_cells_per_device_threshold
             else "ppermute")),
        "cells_per_device": cells_per,
        "max_local_cells": int(max_lc),
        "max_local_edges": int(max_le),
        **(
            {} if round_profile_for_device is None else
            {"round_profile": _round_profile(sched, round_profile_for_device,
                                             n_dev)}
        ),
    }


def _round_profile(sched, device, n_dev):
    """Per-round payload for ONE device, in the order it observes them.

    See ``spmd_schedule_cost``'s ``round_profile_for_device``.  Only rounds
    whose colour class touches *device* are returned, because those are the
    only ones on which it issues a collective.
    """
    # Strict, like the n_dev check: int() would coerce 0.9 to 0 and silently
    # profile a different device than the caller named.
    if int(device) != device or not 0 <= device < n_dev:
        raise ValueError(
            f"round_profile_for_device must be an integer in [0, {n_dev}), "
            f"got {device!r}")
    device = int(device)
    out = []
    for r, perm in enumerate(sched["ppermute_perms"]):
        partner = next((dst for src, dst in perm if src == device), None)
        if partner is None:
            continue
        # halo_cells_per_round is the round's PADDED extent, and that is the
        # right payload measure rather than a per-pair count: the index
        # arrays are (n_dev, max_c) and ppermute moves the padded buffer, so
        # every pair in the round puts max_c entities on the wire.  (The
        # per-pair send maps cannot be recovered from those arrays anyway --
        # they are zero-padded and 0 is a valid index.)
        out.append({
            "round": r,
            "partner": int(partner),
            "halo_cells": int(sched["halo_cells_per_round"][r]),
            "halo_edges": int(sched["halo_edges_per_round"][r]),
        })
    return out


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
        # int32, not int64: every rank of the setup holds ALL n_dev
        # partitions, and these three full-global arrays dominate the
        # setup's HOST memory — at subdiv-10 / 192 devices the int64
        # version is ~100 GB per rank, and 4 ranks/node OOM-killed a
        # 512 GB node (job 26996571). int32 halves it. Guarded: a mesh
        # at or beyond 2^31-1 entities fails loudly, not wraps (the
        # >= keeps one entity of headroom on purpose; codex/GLM r1).
        if max(nCells, nEdges, nVertices) >= np.iinfo(np.int32).max:
            raise ValueError(
                f"global-to-local maps use int32; mesh has "
                f"{max(nCells, nEdges, nVertices)} entities >= 2^31-1")
        cell_g2l = np.full(nCells, -1, dtype=np.int32)
        cell_g2l[local_cells] = np.arange(len(local_cells), dtype=np.int32)
        edge_g2l = np.full(nEdges, -1, dtype=np.int32)
        edge_g2l[local_edges] = np.arange(len(local_edges), dtype=np.int32)
        vertex_g2l = np.full(nVertices, -1, dtype=np.int32)
        vertex_g2l[local_vertices] = np.arange(len(local_vertices),
                                               dtype=np.int32)

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


def greedy_edge_coloring_ordered(comm_pairs, order):
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


def greedy_edge_coloring(comm_pairs):
    """Legacy first-fit coloring on sorted pairs (the reference/never-regress
    baseline for :func:`multi_ordering_edge_coloring`). Worst case
    ``2*max_degree - 1`` colors — each color is one ppermute ROUND, and the
    route-B MPAS lane is round-latency-bound (#1113), so excess colors are
    pure wall-clock.
    """
    edges = sorted({(min(u, v), max(u, v)) for u, v in comm_pairs})
    return greedy_edge_coloring_ordered(comm_pairs, edges)


def check_proper_edge_coloring(edge_colors, comm_pairs):
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


def multi_ordering_edge_coloring(comm_pairs):
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
        ec = greedy_edge_coloring_ordered(comm_pairs, order)
        rounds = max(ec.values(), default=-1) + 1
        if best_rounds is None or rounds < best_rounds:
            best_rounds, best_colors = rounds, ec
            if best_rounds <= max_degree:
                break            # hit the chromatic-index floor — optimal
    return best_colors, max_degree


def _build_halo_send_maps(partitions, cell_owner, n_dev, cells_per,
                          edges_per):
    """Per-pair halo send/recv maps + the undirected comm-pair graph.

    Shared by :func:`_build_ppermute_schedule` (coloured rounds) and
    :func:`_build_ragged_halo_schedule` (one grouped collective) so the
    two strategies exchange EXACTLY the same rows — the schedules differ
    only in how the transfers are grouped into collectives.

    Returns ``(comm_pairs, cell_send_map, cell_recv_map, edge_send_map,
    edge_recv_map)`` where ``cell_send_map[(src, dst)]`` lists owned-
    local indices in ``src`` to send and ``cell_recv_map[(dst, src)]``
    the matching local positions in ``dst`` (same order), likewise for
    edges.
    """
    from collections import defaultdict

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

    comm_pairs: set[tuple[int, int]] = set()
    for d in range(n_dev):
        for d_prime in halo_cells_from[d]:
            if d != d_prime:
                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
        for d_prime in halo_edges_from[d]:
            if d != d_prime:
                comm_pairs.add((min(d, d_prime), max(d, d_prime)))

    cell_send_map: dict[tuple[int, int], list[int]] = {}
    cell_recv_map: dict[tuple[int, int], list[int]] = {}
    edge_send_map: dict[tuple[int, int], list[int]] = {}
    edge_recv_map: dict[tuple[int, int], list[int]] = {}

    for d in range(n_dev):
        for d_prime, cells_g in halo_cells_from[d].items():
            if d_prime == d:
                continue
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

    return comm_pairs, cell_send_map, cell_recv_map, edge_send_map, \
        edge_recv_map


def _resolve_size_coloring(env_value: str) -> bool:
    """Resolve LEGOESM_MPAS_SIZE_COLORING: '1'/'' on (DEFAULT), '0' off.

    DEFAULT ON since the s9@64 production A/B/A2 (job 26857404, drift
    1.7%): ratio 0.803 — 10.2 -> 8.11 ms/step from the padded-byte cut
    alone, matching the node-NIC-saturation model's prediction. Results
    are bit-identical to the legacy colouring (transfers/scatters are
    row-disjoint; only wire grouping changes), so the escape hatch '0'
    exists for schedule-reproduction runs, not for numerics.

    Size-aware colouring keeps the SAME round count but groups
    similar-payload pairs into the same round, cutting the padded/actual
    byte inflation (measured 2.926x at s9@64, job 26855933 — every pair
    in a round ships the round MAXIMUM because the ppermute index
    arrays are shape-uniform across devices). Transfers and results are
    bit-identical (unpack scatters write disjoint rows); only the wire
    grouping changes. Unknown values raise (dispatch hardening)."""
    if env_value == "0":
        return False
    if env_value in ("1", ""):
        return True
    raise ValueError(
        f"LEGOESM_MPAS_SIZE_COLORING={env_value!r}: must be '0' or '1' "
        f"(empty = ON — the receipted default)")


def padded_weight(edge_colors, pair_w, floor=0):
    """Total padded wire weight of a colouring: per round, every pair
    ships the round max (cells and edges tracked with equal weight —
    their per-entity widths are nlev+2 vs nlev, near-equal).

    ``floor`` is the smallest extent the schedule arrays can have. The built
    schedule floors each entity's per-round extent at one row, so a round that
    exchanges only cells still ships one edge row per active pair, and scoring
    it at zero underprices it (codex review). Left at zero by default: the
    colouring that ships today was chosen against the unfloored score, and
    changing the score would change that colouring in every run without
    anyone asking for it. The annealed search passes ``floor=1`` and compares
    against a floored baseline, so both sides of ITS decision agree.
    """
    from collections import defaultdict
    rounds_c = defaultdict(lambda: floor)
    rounds_e = defaultdict(lambda: floor)
    counts = defaultdict(int)
    for pair, color in edge_colors.items():
        wc, we = pair_w[pair]
        rounds_c[color] = max(rounds_c[color], wc)
        rounds_e[color] = max(rounds_e[color], we)
        counts[color] += 1
    return sum(counts[r] * (rounds_c[r] + rounds_e[r]) for r in counts)




def _resolve_anneal_coloring(env_value: str) -> int:
    """Resolve ``LEGOESM_MPAS_ANNEAL_COLORING``: how many moves of annealed
    search to spend refining the halo round schedule. ``''``/``'0'`` = off
    (default), otherwise a positive count in MILLIONS of moves.

    Off by default because it changes the schedule, and whether a schedule
    that ships fewer bytes over one more round is faster on the machine is a
    measurement to take rather than a default to move. Measured offline at 64
    devices: two million moves take the padded share of the wire from 34.4% to
    25.9%, at about five seconds of host time; a hundred million only reach
    25.6%, so a small budget is the whole win.

    Unknown values raise (dispatch hardening: a typo must not silently pick a
    different halo schedule).
    """
    if env_value in ("", "0"):
        return 0
    if not _CANONICAL_INT_RE.fullmatch(env_value):
        raise ValueError(
            f"LEGOESM_MPAS_ANNEAL_COLORING={env_value!r}: must be a positive "
            f"whole number of millions of search moves, or '0'/empty for off.")
    millions = int(env_value)
    if millions > 200:
        raise ValueError(
            f"LEGOESM_MPAS_ANNEAL_COLORING={millions}: out of range 1..200. "
            f"The search flattens by two million moves, so a larger budget "
            f"spends setup time it cannot convert into bytes.")
    return millions * 1_000_000


def _exp_neg(x):
    """``exp(-x)`` for ``x >= 0``, from addition, multiplication and division
    only, so that two machines agree on every bit.

    The maths library's ``exp`` is free to differ in its last bits between
    builds, and the halo schedule is decided by comparing against it. Two ranks
    that disagree by one bit build DIFFERENT schedules, and a schedule
    disagreement does not raise: the exchange collides and the model computes
    the wrong physics quietly. IEEE 754 pins +, -, * and /, so a series built
    only from those is reproducible where a library call is not. Raised by
    GLM-5.2 in review.

    Argument reduction by halving down to |x| <= 1/32, then eleven Taylor
    terms, then squaring back up. Accurate to well under a part in a billion
    over the range this is used on, which is far finer than the decision it
    feeds.
    """
    if x <= 0.0:
        return 1.0
    if x > 64.0:                      # exp(-64) is below 2e-28; call it zero
        return 0.0
    halvings = 0
    while x > 0.03125:
        x *= 0.5
        halvings += 1
    term = 1.0
    total = 1.0
    for k in range(1, 12):
        term *= -x / k
        total += term
    for _ in range(halvings):
        total *= total
    return total


def anneal_coloring(pairs, pair_w, colors, n_colors, n_moves, seed=0):
    """How much room is left in the colouring? Search harder and see.

    Same move as the model's descent -- give one exchange a different round --
    but a move that makes things slightly worse is sometimes accepted, with a
    probability that falls as the search proceeds, so it can leave the basin a
    strictly-downhill version stops in. Allowed to use more rounds than the
    descent settled on, because an extra round costs about fifteen
    microseconds on this lane while a round of padding costs far more.

    Measured at 64 devices, subdivision 9, depth 9: the shipped colouring
    ships 34.4% padding, and two million moves take that to 25.9% -- about a
    tenth off the wire, for about five seconds of host time at setup. A
    hundred million moves only reach 25.6%, so the budget is small on purpose.

    Off unless LEGOESM_MPAS_ANNEAL_COLORING asks for it.

    Driven by a MOVE COUNT, not by a clock, and by an exponential built from
    arithmetic this machine and the next one agree on (:func:`_exp_neg`).
    Every process of a multi-controller run derives its own schedule
    independently and they must agree exactly; a wall-clock budget would hand
    different ranks different schedules, and so would the maths library's
    ``exp``. A disagreement there does not raise -- it collides in the
    exchange and computes the wrong physics quietly. Both raised by GLM-5.2 in
    review.

    A plain threshold rule was tried in place of the exponential and measured
    WORSE at both sizes -- 27.3% against 25.9% at 64 devices, 41.9% against
    39.1% at 128, over a sweep of six threshold scales -- so the smooth
    acceptance is doing real work and stays.
    """
    import random
    from collections import defaultdict

    if not pairs:
        return dict(colors)
    missing = [p for p in pairs if p not in colors]
    if missing:
        raise ValueError(
            f"anneal_coloring: {len(missing)} exchanges have no starting "
            f"round, e.g. {missing[:3]}; the search refines a colouring, it "
            f"does not build one.")
    for pair, (w_c, w_e) in pair_w.items():
        if int(w_c) != w_c or int(w_e) != w_e:
            raise ValueError(
                f"anneal_coloring: weight for {pair} is not a whole number of "
                f"rows ({w_c}, {w_e}). The running cost is accumulated, so "
                f"fractional weights would let it drift from the cost of the "
                f"colouring it returns.")

    rng = random.Random(seed)
    colors = dict(colors)
    pair_list = sorted(pairs)

    members = defaultdict(list)
    endpoints = defaultdict(lambda: defaultdict(int))
    for pair, color in colors.items():
        members[color].append(pair)
        endpoints[color][pair[0]] += 1
        endpoints[color][pair[1]] += 1

    def class_cost(color):
        # Floored at one row per entity, because the schedule arrays are: a
        # round that exchanges only cells still ships one edge row per active
        # pair. Scoring it at zero would let the search "win" by emptying an
        # entity out of a round that still pays for it (codex review).
        group = members[color]
        if not group:
            return 0
        return len(group) * (max([pair_w[q][0] for q in group] + [1])
                             + max([pair_w[q][1] for q in group] + [1]))

    total = sum(class_cost(c) for c in list(members))
    best_total, best_colors = total, dict(colors)

    # A temperature on the scale of one exchange's weight: warm enough early
    # to cross a round boundary, cold enough at the end to stop wandering.
    # Scaling it to a whole round's cost instead was measured worse.
    heaviest = max(w[0] + w[1] for w in pair_w.values())
    start = heaviest * 0.25
    n_moves = int(n_moves)
    moves = accepted = 0
    for move in range(n_moves):
        temperature = start * (n_moves - move) / n_moves + 1.0
        moves += 1
        pair = pair_list[rng.randrange(len(pair_list))]
        old = colors[pair]
        new_color = rng.randrange(n_colors)
        if new_color == old:
            continue
        if endpoints[new_color][pair[0]] or endpoints[new_color][pair[1]]:
            continue          # would collide at a device
        before = class_cost(old) + class_cost(new_color)
        members[old].remove(pair)
        members[new_color].append(pair)
        after = class_cost(old) + class_cost(new_color)
        delta = after - before
        if delta <= 0 or rng.random() < _exp_neg(delta / temperature):
            accepted += 1
            colors[pair] = new_color
            for node in pair:
                endpoints[old][node] -= 1
                endpoints[new_color][node] += 1
            total += delta
            if total < best_total:
                best_total, best_colors = total, dict(colors)
        else:
            members[new_color].remove(pair)
            members[old].append(pair)
    logger.debug("  anneal: %d moves, %d accepted", moves, accepted)
    return best_colors


def size_aware_edge_coloring(comm_pairs, edge_colors, pair_w, n_rounds,
                             coloring_method="greedy"):
    """Regroup an edge colouring so similar-sized exchanges share a round.

    Every pair in a round ships that round's LARGEST halo, because the sharded
    step needs one shape across devices. So the wire cost of a colouring is
    :func:`padded_weight`, and a heavy pair landing in a crowded round is
    expensive while grouping similar payloads together is cheap. This searches
    for a lighter colouring at the same round count, and accepts one extra
    round only when it buys at least a tenth of the weight -- an extra
    sequential exchange prices at about 15 microseconds on this lane, which is
    a good trade while the lane is bytes-bound.

    Returns ``(colours, rounds, method)``; the inputs are returned unchanged
    when nothing better is found, so the caller does not have to check.

    Deterministic: every process in a multi-controller run derives its own
    schedule and they must agree, so the orderings tie-break on the pair
    itself rather than on set iteration order, and the jitter is seeded.

    Extracted from the schedule builder so that the probes measuring how much
    padding is left (scripts/validate/mpas_halo_padding_census.py) score the
    colouring the model actually ships rather than a re-derivation of it.
    """
    import random as _random_sc
    base_w = padded_weight(edge_colors, pair_w)
    # Seed candidates: payload-descending first-fit + jitters. These
    # often overshoot the round budget on dense graphs (s9@64: every
    # reorder blew past 11 rounds and the guard rejected them all,
    # job 26856688) — so ALWAYS follow with a round-PRESERVING local
    # search that moves pairs between existing rounds.
    # FULL tie-break key (weight, then the pair itself): every
    # multicontroller process must derive the IDENTICAL colouring
    # independently, and a weight-only key leaves equal-weight order
    # to set-iteration order.
    edges_by_size = sorted(
        comm_pairs,
        key=lambda p: (-(pair_w[p][0] + pair_w[p][1]), p))
    candidates = [edges_by_size]
    for seed in (1, 2, 3):
        jit = edges_by_size[:]
        rng = _random_sc.Random(seed)
        for i in range(0, len(jit) - 1, 2):
            if rng.random() < 0.5:
                jit[i], jit[i + 1] = jit[i + 1], jit[i]
        candidates.append(jit)
    # +1-ROUND CANDIDATES admitted (2026-08-11): the cross-lane law
    # (wide-halo 0.970, mixed-pad 0.983) prices an extra sequential
    # collective at ~15 us marginal while the lane is BYTES-bound —
    # so a colouring that spends one extra round to cut padded
    # weight is a good trade. Admission bar at adoption below:
    # equal rounds need ANY strict weight win; rounds+1 needs
    # >= 10% below the best equal-rounds weight.
    seeds = [dict(edge_colors)]
    for order in candidates:
        ec = greedy_edge_coloring_ordered(comm_pairs, order)
        if max(ec.values(), default=-1) + 1 <= n_rounds + 1:
            seeds.append(ec)

    def _local_search(ec):
        """Move pairs between existing rounds (endpoint-conflict
        free) while the padded weight strictly drops."""
        from collections import defaultdict
        colors = dict(ec)
        n_r = max(colors.values()) + 1
        occupied = defaultdict(set)   # round -> endpoint set
        members = defaultdict(list)
        for p, c in colors.items():
            occupied[c].update(p)
            members[c].append(p)
        improved = True
        while improved:
            improved = False
            w_now = padded_weight(colors, pair_w)
            for p in sorted(colors, key=lambda q:
                            (-(pair_w[q][0] + pair_w[q][1]), q)):
                c0 = colors[p]
                for c1 in range(n_r):
                    if c1 == c0 or (occupied[c1] & set(p)):
                        continue
                    colors[p] = c1
                    w_try = padded_weight(colors, pair_w)
                    if w_try < w_now:
                        occupied[c0] = set(
                            x for q in members[c0] if q != p for x in q)
                        members[c0].remove(p)
                        members[c1].append(p)
                        occupied[c1].update(p)
                        w_now = w_try
                        improved = True
                        break
                    colors[p] = c0
        return colors

    best_w, best_ec = base_w, None          # equal-rounds champion
    plus_w, plus_ec = None, None            # rounds+1 champion
    for seed_ec in seeds:
        ec = _local_search(seed_ec)
        if not check_proper_edge_coloring(ec, comm_pairs):
            continue
        r = max(ec.values(), default=-1) + 1
        if r > n_rounds + 1:
            continue
        w = padded_weight(ec, pair_w)
        if r <= n_rounds:
            if w < best_w:
                best_w, best_ec = w, ec
        else:
            if plus_w is None or w < plus_w:
                plus_w, plus_ec = w, ec
    if plus_ec is not None and plus_w < 0.90 * best_w:
        best_w, best_ec = plus_w, plus_ec
    if best_ec is None:
        logger.info(
            "  size-aware colouring found no improvement "
            "(padded weight %d)", base_w)
        return _maybe_anneal(comm_pairs, edge_colors, pair_w, base_w,
                             coloring_method)
    logger.info(
        "  size-aware colouring adopted: padded weight %d -> %d "
        "(-%.0f%%), rounds %d", base_w, best_w,
        100 * (1 - best_w / max(base_w, 1)),
        max(best_ec.values()) + 1)
    return _maybe_anneal(comm_pairs, best_ec, pair_w, best_w, "size_aware")


def _maybe_anneal(comm_pairs, edge_colors, pair_w, weight, method):
    """Refine a colouring by annealed search, when asked, and only when it
    strictly wins. Returns ``(colours, rounds, method)`` either way.

    Kept separate from the descent above because it answers a different
    question: the descent finds the bottom of the basin its seeds land in,
    while this one is allowed to climb out. It is also allowed ONE more round
    than the descent settled on -- an extra sequential exchange prices at
    about fifteen microseconds here, and a round of padding costs far more.
    """
    import os as _os_anneal

    n_moves = _resolve_anneal_coloring(
        _os_anneal.environ.get("LEGOESM_MPAS_ANNEAL_COLORING", ""))
    rounds = max(edge_colors.values()) + 1
    if not n_moves:
        return edge_colors, rounds, method

    searched = anneal_coloring(comm_pairs, pair_w, edge_colors, rounds + 1,
                               n_moves)
    if not check_proper_edge_coloring(searched, comm_pairs):
        # Cannot happen by construction -- the move rejects any colour already
        # holding an endpoint -- which is exactly why it is checked: a silent
        # improper schedule collides at a device and corrupts the halo.
        raise AssertionError(
            "annealed colouring is improper; two exchanges in one round "
            "would collide at a device")
    searched_w = padded_weight(searched, pair_w, floor=1)
    weight = padded_weight(edge_colors, pair_w, floor=1)
    if searched_w >= weight:
        logger.info(
            "  annealed colouring found no improvement (padded weight %d)",
            weight)
        return edge_colors, rounds, method
    searched_rounds = max(searched.values()) + 1
    logger.info(
        "  annealed colouring adopted: padded weight %d -> %d (-%.0f%%), "
        "rounds %d -> %d", weight, searched_w,
        100 * (1 - searched_w / max(weight, 1)), rounds, searched_rounds)
    return searched, searched_rounds, "size_aware_annealed"

def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
                             edges_per, max_lc, max_le,
                             cell_width: int = 1, edge_width: int = 1):
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

    # Steps 1+2+4 (boundary discovery, comm graph, directed send/recv
    # maps) live in the shared helper so the ragged schedule moves
    # EXACTLY the same rows.
    (comm_pairs, cell_send_map, cell_recv_map, edge_send_map,
     edge_recv_map) = _build_halo_send_maps(
        partitions, cell_owner, n_dev, cells_per, edges_per)

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
    greedy_colors = greedy_edge_coloring(comm_pairs)
    n_rounds_greedy = max(greedy_colors.values()) + 1
    multi_colors, max_degree = multi_ordering_edge_coloring(comm_pairs)
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

    # SIZE-AWARE colouring (opt-in): same round count, pairs grouped by
    # payload size so the per-round max padding shrinks. Candidates =
    # weight-descending first-fit + a few weight-jittered restarts;
    # adopted only when the round count DOES NOT regress and the padded
    # wire weight strictly improves.
    import os as _os_sc
    if not _resolve_size_coloring(
            _os_sc.environ.get("LEGOESM_MPAS_SIZE_COLORING", "")):
        # The annealed search is a separate switch and must not be silently
        # inert because this one is off (codex review).
        pair_w = {}
        for (u, v) in comm_pairs:
            wc = max(len(cell_send_map.get((u, v), [])),
                     len(cell_send_map.get((v, u), [])))
            we = max(len(edge_send_map.get((u, v), [])),
                     len(edge_send_map.get((v, u), [])))
            pair_w[(u, v)] = (wc * cell_width, we * edge_width)
        edge_colors, n_rounds, coloring_method = _maybe_anneal(
            comm_pairs, edge_colors, pair_w,
            padded_weight(edge_colors, pair_w), coloring_method)
    else:
        import random as _random_sc
        # Pair weights in TRUE relative units (codex review: an
        # equal-weight proxy can rate a cell/edge trade as improving
        # while actual bytes worsen — the packed widths are nlev+2 vs
        # nlev). The factory threads the real widths; the default 1:1
        # is the estimator's unit-free score.
        pair_w = {}
        for (u, v) in comm_pairs:
            wc = max(len(cell_send_map.get((u, v), [])),
                     len(cell_send_map.get((v, u), [])))
            we = max(len(edge_send_map.get((u, v), [])),
                     len(edge_send_map.get((v, u), [])))
            pair_w[(u, v)] = (wc * cell_width, we * edge_width)
        edge_colors, n_rounds, coloring_method = size_aware_edge_coloring(
            comm_pairs, edge_colors, pair_w, n_rounds, coloring_method)
    assert check_proper_edge_coloring(edge_colors, comm_pairs), (
        "improper ppermute edge coloring — two same-round exchanges "
        "would collide at a device")
    rounds: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (u, v), color in edge_colors.items():
        rounds[color].append((u, v))

    # ------------------------------------------------------------------
    # 5. Assemble per-round ppermute patterns and index arrays
    #    (the directed send/recv maps come from _build_halo_send_maps)
    # ------------------------------------------------------------------
    ppermute_perms_out: list[list[tuple[int, int]]] = []
    send_cell_idx_out: list[jnp.ndarray] = []
    recv_cell_pos_out: list[jnp.ndarray] = []
    send_edge_idx_out: list[jnp.ndarray] = []
    recv_edge_pos_out: list[jnp.ndarray] = []
    halo_cells_per_round: list[int] = []
    halo_edges_per_round: list[int] = []
    recv_cell_pos_np: list[np.ndarray] = []
    recv_edge_pos_np: list[np.ndarray] = []

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
        recv_cell_pos_np.append(rc)
        recv_edge_pos_np.append(re)

    assert_recv_positions_unique(
        recv_cell_pos_np, recv_edge_pos_np, n_dev, max_lc, max_le)

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
# unmeasured workloads (codex round-12). The dtype is part of the
# signature for the same reason — fusion decisions depend on element
# type, and the receipt is f32-only (f64 unmeasured as of 2026-07-27).
# Grow ONLY with a measured receipt for the exact signature.
# PROVISIONAL f64 np8 entry under test (job 26493638: f64 np4 is HEALTHY
# at eff 0.95 while np8 ANTI-scales 20.10 -> 21.42 — the candidate
# pathological shape shifts one rung with the doubled element size).
# Receipt job decides whether this entry stays.
_FUSION_BARRIER_WORKLOADS = frozenset({
    (4, 1_966_080, 655_376, 26, "float32"),
    (8, 1_966_080, 655_376, 26, "float64"),
})

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



def assert_recv_positions_unique(recv_cell_pos, recv_edge_pos, n_dev,
                                 cell_garbage, edge_garbage):
    """Raise if any device receives the same halo position in two rounds.

    Every halo row has exactly one owner, so a device receives each of its
    halo positions in exactly one round. The per-round fill does not need
    that -- later rounds simply overwrite earlier ones -- but the merged fill
    (``LEGOESM_MPAS_HALO_MERGE_SCATTER``) writes all rounds in a single
    scatter, where a repeated position has no defined winner and a GPU may
    resolve it differently from a CPU.

    So if a future schedule ever relays a row through an intermediate value,
    or sends a shared corner twice, it is caught here, on the host, before
    any of it reaches a device.

    ``recv_cell_pos`` / ``recv_edge_pos`` are per-round ``(n_dev, max_*)``
    integer arrays; entries equal to the garbage position are the padding
    that every round shares and are excluded.
    """
    for device in range(n_dev):
        for name, per_round, garbage in (
                ("cell", recv_cell_pos, cell_garbage),
                ("edge", recv_edge_pos, edge_garbage)):
            written = np.concatenate([r[device] for r in per_round])
            real = written[written != garbage]
            positions, counts = np.unique(real, return_counts=True)
            repeated = positions[counts > 1]
            if repeated.size:
                raise ValueError(
                    f"halo schedule writes the same {name} position more "
                    f"than once on device {device}: "
                    f"{repeated[:8].tolist()}"
                    + (f" and {repeated.size - 8} more"
                       if repeated.size > 8 else "")
                    + ". The merged halo scatter has no defined winner for "
                      "a repeated position.")


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
    import os as _os_ballast

    cells_per = cell_pack.shape[0]
    edges_per = u_shard.shape[0]
    # ONE scatter per entity class instead of one per round, when asked.
    merge_scatter = _resolve_halo_merge_scatter(
        _os_ballast.environ.get("LEGOESM_MPAS_HALO_MERGE_SCATTER", ""))

    # Garbage slots for padded scatter targets, trimmed at the end: schedule
    # rows are padded to the round's max halo count, and padding entries
    # target position max_lc / max_le.
    #
    # The merged path gives EVERY ROUND ITS OWN garbage row rather than
    # sharing one. Sharing works forward -- the row is trimmed off, so an
    # unspecified winner never reaches the answer -- but it makes the single
    # scatter's index list non-unique, which (a) is undefined behaviour that
    # a GPU can resolve differently from a CPU, and (b) has a reverse-mode
    # transpose that gathers that row's cotangent back to the padding entry
    # of every round at once. One row per round makes the whole index list
    # unique, which is both correct by construction and cheaper: the scatter
    # can then be lowered with the no-duplicates promise.
    n_rounds = len(halo_sl)
    _pad_slots = n_rounds if merge_scatter else 1
    cell_local = jnp.pad(
        cell_pack, ((0, max_lc + _pad_slots - cells_per), (0, 0)))
    u_local = jnp.pad(
        u_shard, ((0, max_le + _pad_slots - edges_per), (0, 0)))

    ballast = _resolve_halo_ballast(
        _os_ballast.environ.get("LEGOESM_MPAS_HALO_BALLAST", ""))
    nocomm = _resolve_halo_nocomm(
        _os_ballast.environ.get("LEGOESM_MPAS_HALO_NOCOMM", ""))
    nostage = _resolve_halo_nostage(
        _os_ballast.environ.get("LEGOESM_MPAS_HALO_NOSTAGE", ""))

    # Each halo row has exactly ONE owner, so the real receive positions are
    # disjoint across rounds; that is what makes the writes safe to defer and
    # issue together, and it is asserted on the schedule where the schedule
    # is built, not trusted here.
    _pending_c, _pending_cp, _pending_e, _pending_ep = [], [], [], []

    if merge_scatter and nostage:
        raise ValueError(
            "LEGOESM_MPAS_HALO_MERGE_SCATTER=1 together with "
            "LEGOESM_MPAS_HALO_NOSTAGE=1: the no-stage arm performs no "
            "scatters at all, so the merge switch cannot act, and a receipt "
            "recording both would name a knob that did nothing. Unset one.")

    if nostage:
        # MEASUREMENT ONLY, WRONG ANSWERS: skip the whole per-round loop,
        # leaving the halo rows at their padded initial values. The
        # kernel, the local region, the masking and every other line of
        # the step are unchanged, so (nocomm - nostage) is the cost of
        # the on-device halo staging -- gather, concatenate, scatter --
        # measured against the SAME program.
        #
        # Why this is needed: the obvious control, the same per-device
        # load on ONE device, is NOT the same program.
        # make_voronoi_sharded_step returns the plain serial model.step
        # at n_devices == 1, so that arm cannot be subtracted from a
        # sharded one. Found by codex review after exactly that
        # subtraction had been reported.
        return cell_local[:max_lc], u_local[:max_le]


    for r, (sc, rc, se, re) in enumerate(halo_sl):
        send_c = cell_pack[sc[0]]             # (hc_r, W)
        send_e = u_shard[se[0]]               # (he_r, nlev)
        send_c_flat = send_c.ravel()
        send_packed = jnp.concatenate([send_c_flat, send_e.ravel()])
        if ballast > 1:
            # MEASUREMENT ONLY: multiply the bytes on the wire by
            # `ballast` while holding the round count, the schedule and
            # every arithmetic operation fixed, then discard the copies
            # on receipt. This is the one-variable experiment for the
            # bandwidth term of the exchange -- fitting it out of three
            # A/B receipts leaves it resting on a compute time imported
            # from an older trace. Result is bit-identical: the kept
            # slice is the same buffer that would have been sent.
            send_packed = jnp.concatenate([send_packed] * ballast)
        if nocomm:
            # MEASUREMENT ONLY, and it produces WRONG ANSWERS: drop the
            # collective and let each device scatter its OWN gathered
            # rows into its halo slots. Everything else -- the gather,
            # the concatenate, the scatter, the schedule shape, the
            # kernel count -- is unchanged, so the step's change is the
            # wire time plus whatever waiting for the slowest peer
            # costs. Splitting that pair off is the only way to see how
            # much of the step is on-device halo staging rather than
            # communication.
            # optimization_barrier keeps the send-side pack alive.
            # Without it `recv_packed[:split]` is a slice of the
            # concatenate, XLA folds it back to the operand, and the arm
            # silently drops one concatenate and one fusion per round --
            # measured 21 -> 18 concatenates, 110 -> 107 fusions at 4
            # devices. That made the arm time a slightly different
            # program, so its two terms were bounds rather than
            # estimates.
            recv_packed = jax.lax.optimization_barrier(send_packed)
        else:
            recv_packed = jax.lax.ppermute(
                send_packed, "device", perm=ppermute_perms[r])
        split_at = send_c_flat.shape[0]       # static
        recv_c = recv_packed[:split_at].reshape(send_c.shape)
        recv_e = recv_packed[split_at:split_at + send_e.size].reshape(
            send_e.shape)
        if merge_scatter:
            _pending_c.append(recv_c)
            _pending_cp.append(jnp.where(rc[0] == max_lc, max_lc + r, rc[0]))
            _pending_e.append(recv_e)
            _pending_ep.append(jnp.where(re[0] == max_le, max_le + r, re[0]))
        else:
            cell_local = cell_local.at[rc[0]].set(recv_c)
            u_local = u_local.at[re[0]].set(recv_e)

    if merge_scatter and _pending_c:
        cell_local = cell_local.at[jnp.concatenate(_pending_cp)].set(
            jnp.concatenate(_pending_c, axis=0), unique_indices=True)
        u_local = u_local.at[jnp.concatenate(_pending_ep)].set(
            jnp.concatenate(_pending_e, axis=0), unique_indices=True)

    return cell_local[:max_lc], u_local[:max_le]


#: Canonical decimal integer, no sign / whitespace / underscores /
#: leading zeros — the spellings ``int()`` would silently accept.
_CANONICAL_INT_RE = re.compile(r"[1-9][0-9]*")


def _resolve_halo_merge_scatter(env_value: str) -> bool:
    """Resolve ``LEGOESM_MPAS_HALO_MERGE_SCATTER``: write the received halo
    rows into the local buffers ONCE instead of once per coloured round;
    ``''``/``'0'`` = off (default).

    The on-device staging -- gather, concatenate, scatter -- costs 0.310 ms of
    a 5.760 ms step at 64 GPUs, measured by an arm that skips it entirely. The
    fill scatters thirteen times, once per round. Each halo row has exactly
    one owner, so the receive positions are disjoint across rounds and the
    writes can be deferred and issued together.

    Bit-identical: every real position is written once with the same value
    either way. The one repeated index is the padding's garbage slot, which
    every round targets and which is trimmed off the return, so an unspecified
    winner there cannot reach the answer.

    Off by default because it changes the compiled program, and that is a
    measurement to take rather than a default to move.

    Unknown values raise (dispatch hardening).
    """
    if env_value in ("", "0"):
        return False
    if env_value == "1":
        return True
    raise ValueError(
        f"LEGOESM_MPAS_HALO_MERGE_SCATTER={env_value!r}: must be '0' or '1' "
        f"(empty = off).")


def _resolve_halo_nocomm(env_value: str) -> bool:
    """Resolve LEGOESM_MPAS_HALO_NOCOMM: ``'1'`` replaces every halo
    ``ppermute`` with the identity; ``''``/``'0'`` off (default).

    A MEASUREMENT knob that DELIBERATELY BREAKS THE ANSWER -- each
    device scatters its own rows into its halo slots, so the halo is
    garbage and the run is meaningless as physics. It exists to time
    the on-device halo staging (gather, concatenate, scatter) and the
    enlarged local region SEPARATELY from the wire time and the wait
    for the slowest peer, which is otherwise unsplittable: the profiler
    on this stack does not record the halo collectives at all.

    Never valid in production. Unknown values raise (dispatch
    hardening).
    """
    if env_value in ("", "0"):
        return False
    if env_value == "1":
        return True
    raise ValueError(
        f"LEGOESM_MPAS_HALO_NOCOMM={env_value!r}: must be '0' or '1' "
        f"(empty = off). It is a timing knob that BREAKS the answer; "
        f"a typo must not silently enable it.")


def _resolve_halo_nostage(env_value: str) -> bool:
    """Resolve LEGOESM_MPAS_HALO_NOSTAGE: ``'1'`` skips the entire
    per-round halo staging loop; ``''``/``'0'`` off (default).

    A MEASUREMENT knob that DELIBERATELY BREAKS THE ANSWER -- the halo
    rows keep their padded initial values. It exists so the staging cost
    (gather, concatenate, scatter) can be measured against the SAME
    program: the kernel, the local region and the masking are unchanged.
    The obvious alternative, the same per-device load on ONE device,
    is NOT the same program -- ``make_voronoi_sharded_step`` returns the
    plain serial ``model.step`` at ``n_devices == 1``.

    Its own resolver rather than sharing LEGOESM_MPAS_HALO_NOCOMM's, so
    a typo raises an error naming the variable the user actually set.
    """
    if env_value in ("", "0"):
        return False
    if env_value == "1":
        return True
    raise ValueError(
        f"LEGOESM_MPAS_HALO_NOSTAGE={env_value!r}: must be '0' or '1' "
        f"(empty = off). It is a timing knob that BREAKS the answer; "
        f"a typo must not silently enable it.")


def _resolve_halo_ballast(env_value: str) -> int:
    """Resolve LEGOESM_MPAS_HALO_BALLAST: wire-payload multiplier for
    the exchange, ``''``/``'1'`` = off (default).

    A MEASUREMENT knob, never a production one. It repeats the packed
    send buffer N times so the collective moves N x the bytes with the
    SAME round count, the SAME schedule and the SAME arithmetic, and
    throws the copies away on receipt. The step's change is then the
    bandwidth term of the exchange, measured with one variable moved
    instead of fitted out of three separate A/B receipts against a
    compute time taken from an older trace.

    Unknown or non-canonical values raise (dispatch hardening): a typo
    must not silently run a different payload multiple.
    """
    if env_value in ("", "1"):
        return 1
    if _CANONICAL_INT_RE.fullmatch(env_value) is None:
        raise ValueError(
            f"LEGOESM_MPAS_HALO_BALLAST={env_value!r}: must be a "
            f"canonical decimal integer >= 1 (empty or '1' = off)")
    n = int(env_value)
    if not 1 <= n <= 8:
        raise ValueError(
            f"LEGOESM_MPAS_HALO_BALLAST={n}: out of range 1..8. It "
            f"multiplies every halo message, so a large value runs the "
            f"node out of memory rather than measuring anything.")
    return n


# Device-count ceiling for LEGOESM_MPAS_RAGGED_HALO=auto. Production A/B
# receipts (drift-controlled): ratio ragged/coloured 0.686 @16 devices
# (jobs 26822138/26824483), 0.735 @32 (26825520), 1.220 @64 (26824688);
# the inversion is the unpruned zero-size-slice cost (~12 us/slice,
# confirmed at fixed degree+payload by job 26825475). Raise only with a
# new production A/B receipt above the current edge.
_RAGGED_AUTO_MAX_NDEV = 32


def _resolve_ragged_halo(env_value: str, n_dev: int) -> bool:
    """Resolve LEGOESM_MPAS_RAGGED_HALO: '1' force-on, '0'/'' off,
    'auto' = on iff ``n_dev <= _RAGGED_AUTO_MAX_NDEV`` (the receipted
    win band). Unknown values raise (dispatch-hardening: a typo must
    not silently pick a halo strategy)."""
    if env_value in ("0", ""):
        return False
    if env_value == "1":
        return True
    if env_value == "auto":
        return n_dev <= _RAGGED_AUTO_MAX_NDEV
    raise ValueError(
        f"LEGOESM_MPAS_RAGGED_HALO={env_value!r}: must be one of "
        f"'0', '1', 'auto' (empty = off)")


#: Tendency evaluations per step for each ``dispatch_integrator`` name.
#: Consumed by the wide-halo (communication-avoiding) step: halo depth =
#: evals x SPMD_HALO_DEPTH, because SSP/RK stage validity shrinks by one
#: tendency reach per evaluation (Shu-Osher shrinking-region argument).
#: Grow-only alongside ``timestepping.dispatch._INTEGRATORS``; a name
#: missing here refuses wide mode rather than guessing a depth.
_INTEGRATOR_TENDENCY_EVALS = {
    "ssp_rk3": 3, "ssp3": 3, "rk3": 3,
    "ssp_rk3_scan": 3, "ssp3_scan": 3, "rk3_scan": 3,
    "ssp_rk34": 4, "ssp34": 4, "rk34": 4,
    "ssp_rk54": 5, "ssp54": 5, "ssp45": 5, "rk54": 5,
    "ssp_rk54_scan": 5, "ssp54_scan": 5, "rk54_scan": 5,
    "rk4": 4, "runge_kutta_4": 4,
}


#: Sentinel ring distance for local rows that are padding or outside the
#: wide region — always beyond any mask threshold.
_WIDE_RING_FAR = np.iinfo(np.int32).max


def _build_wide_halo_rings(global_mesh, partitions, max_lc, max_le,
                           halo_depth):
    """Per-device ring distances for the wide-halo shrinking masks.

    Returns ``(cell_ring, edge_ring)`` int32 arrays of shape
    ``(n_dev, max_lc)`` / ``(n_dev, max_le)``:

    * ``cell_ring[d, i]`` — BFS ring of device *d*'s i-th local cell
      from its owned block (0 = owned), ``_WIDE_RING_FAR`` for padding.
    * ``edge_ring[d, j]`` — max of the two adjacent cells' rings
      (an edge is in the depth-``r`` region iff BOTH its cells are —
      the same AND filter ``_build_voronoi_partition_infra`` applies),
      ``_WIDE_RING_FAR`` when a cell is absent or the row is padding.
      Owned edges are overridden inside the kernel (first
      ``edges_per`` rows), not here.

    The eval-k mask keeps entities with ring <= ``(evals-k) *
    SPMD_HALO_DEPTH`` (plus all owned rows): outside that region the
    stage values are frozen (zero tendency) so every primal stays
    finite — an unmasked wide step lets garbage outer-ring values turn
    zero cotangents into NaN through the chain rule (0 * NaN), which
    the fill transpose then scatter-adds into owned gradients.
    """
    coc = np.asarray(global_mesh.cellsOnCell)     # (maxEdges, nCells)
    coe = np.asarray(global_mesh.cellsOnEdge)     # (2, nEdges)
    n_dev = len(partitions)
    nCells = coc.shape[1]
    cell_ring = np.full((n_dev, max_lc), _WIDE_RING_FAR, dtype=np.int32)
    edge_ring = np.full((n_dev, max_le), _WIDE_RING_FAR, dtype=np.int32)

    # cellsOnCell BFS from the owned cell block — the metric the GPU
    # parity gate certified (job 26846337, s6@4, u atol 1e-6).  A
    # union-graph metric seeded from owned cells+edges was tried for
    # codex P1 (closure cells left FAR) and REVERTED: shrinking the
    # ring distances enlarges every keep-set, and the s6@4 GPU parity
    # gate FAILED on u at 4e-2 (job 26849483) — the extra kept cells
    # compute tendencies on locally-incomplete connectivity that the
    # freeze was protecting against.  Residual (documented, empirically
    # bounded by the parity gate): cells the partition closure adds
    # beyond the cellsOnCell k-ring stay FAR and their stage values
    # frozen; the count is logged per rank below.
    for d, part in enumerate(partitions):
        g2l = part.cell_g2l
        n_owned = part.n_owned_cells
        n_local = part.n_local_cells
        ring_l = np.full(max_lc, _WIDE_RING_FAR, dtype=np.int64)
        ring_l[:n_owned] = 0
        frontier = np.asarray(part.local_cells[:n_owned])
        seen = np.zeros(nCells, dtype=bool)
        seen[frontier] = True
        for r in range(1, halo_depth + 1):
            if frontier.size == 0:
                break
            nb = coc[:, frontier].ravel()
            nb = nb[nb >= 0]
            nb = np.unique(nb)
            nb = nb[~seen[nb]]
            seen[nb] = True
            lidx = g2l[nb]
            nb_local = lidx[lidx >= 0]
            ring_l[nb_local] = r
            frontier = nb
        n_unlabelled = int((ring_l[:n_local] == _WIDE_RING_FAR).sum())
        if n_unlabelled:
            logger.info(
                "wide halo: rank %d keeps %d closure-added local cells "
                "FROZEN (unlabelled by the cellsOnCell ring BFS); the "
                "s6@4 GPU parity gate is the guard that this freeze "
                "does not reach owned results.", d, n_unlabelled)
        cell_ring[d] = ring_l.astype(np.int32)

        le = np.asarray(part.local_edges)
        c12 = coe[:, le]                          # (2, n_local_edges)
        r12 = np.full_like(c12, _WIDE_RING_FAR, dtype=np.int64)
        for side in range(2):
            cs = c12[side]
            valid = cs >= 0
            lidx = np.full(cs.shape, -1, dtype=np.int64)
            lidx[valid] = g2l[cs[valid]]
            present = lidx >= 0
            r12[side, present] = ring_l[lidx[present]]
        edge_ring[d, :le.shape[0]] = np.max(
            r12, axis=0).astype(np.int32)

    return cell_ring, edge_ring


def _build_rim_rings(global_mesh, partitions, max_lc, max_le,
                     max_width):
    """Per-device INWARD ring distances — the rim complement of
    :func:`_build_wide_halo_rings`.

    Returns ``(cell_rim, edge_rim)`` int32 arrays of shape
    ``(n_dev, max_lc)`` / ``(n_dev, max_le)``:

    * ``cell_rim[d, i]`` — BFS hops (over ``cellsOnCell``) from device
      *d*'s i-th local cell to the nearest cell NOT owned by *d*;
      non-owned (halo) rows are the seed and get 0; owned cells farther
      than ``max_width`` hops (the interior) and padding rows get
      ``_WIDE_RING_FAR``. The width-``w`` RIM — the owned cells whose
      radius-``w`` stencil can see a ghost value, i.e. the rows an
      interior/rim split must recompute after the halo fill — is
      ``1 <= cell_rim <= w``.
    * ``edge_rim[d, j]`` — min of the two adjacent cells' rim
      distances (an edge is ghost-affected iff EITHER cell is —
      dual to the outward builder's AND/max), ``_WIDE_RING_FAR`` when
      padding.

    The BFS seeds from every non-owned LOCAL cell; owned boundary cells
    adjacent to a cell of another device that is absent from the local
    halo cannot occur for ``max_width <= halo_depth`` (depth-1 closure
    contains every neighbour of an owned cell), which the caller must
    hold — asserted below.
    """
    coc = np.asarray(global_mesh.cellsOnCell)     # (maxEdges, nCells)
    coe = np.asarray(global_mesh.cellsOnEdge)     # (2, nEdges)
    n_dev = len(partitions)
    cell_rim = np.full((n_dev, max_lc), _WIDE_RING_FAR, dtype=np.int32)
    edge_rim = np.full((n_dev, max_le), _WIDE_RING_FAR, dtype=np.int32)

    for d, part in enumerate(partitions):
        g2l = part.cell_g2l
        n_owned = part.n_owned_cells
        n_local = part.n_local_cells
        assert max_width >= 1, "rim width must be >= 1"
        rim_l = np.full(max_lc, _WIDE_RING_FAR, dtype=np.int64)
        # Seed: every local non-owned (halo) cell at distance 0.
        rim_l[n_owned:n_local] = 0
        frontier = np.asarray(part.local_cells[n_owned:n_local])
        # done marks GLOBAL cells already labelled (seed + visited owned).
        done = np.zeros(coc.shape[1], dtype=bool)
        done[frontier] = True
        for r in range(1, max_width + 1):
            if frontier.size == 0:
                break
            nb = coc[:, frontier].ravel()
            nb = nb[nb >= 0]
            nb = np.unique(nb)
            nb = nb[~done[nb]]
            done[nb] = True
            lidx = g2l[nb]
            # keep OWNED rows only — the rim lives in the owned block.
            nb_owned = lidx[(lidx >= 0) & (lidx < n_owned)]
            rim_l[nb_owned] = r
            frontier = nb
        cell_rim[d] = rim_l.astype(np.int32)

        le = np.asarray(part.local_edges)
        c12 = coe[:, le]                          # (2, n_local_edges)
        r12 = np.full_like(c12, _WIDE_RING_FAR, dtype=np.int64)
        for side in range(2):
            cs = c12[side]
            valid = cs >= 0
            lidx = np.full(cs.shape, -1, dtype=np.int64)
            lidx[valid] = g2l[cs[valid]]
            present = lidx >= 0
            r12[side, present] = rim_l[lidx[present]]
        edge_rim[d, :le.shape[0]] = np.min(
            r12, axis=0).astype(np.int32)

    return cell_rim, edge_rim


def _build_rim_rings_structural(global_mesh, partitions, max_lc, max_le,
                                *, n_rounds=3):
    """Rim membership by STRUCTURAL stencil closure -- provably safe.

    Returns ``(cell_rim, edge_rim)`` with the ``_build_rim_rings`` contract
    (``1`` = rim, ``_WIDE_RING_FAR`` = interior), consumed by
    ``_build_rim_plan`` at ``rim_width=1``.

    An owned entity is RIM iff any non-owned (halo) entity lies within
    ``n_rounds`` hops of it over the mesh incidence graph -- the union of
    every cell/edge/vertex adjacency the RHS could read. Unlike the cheap
    ``min(cell_rim)`` measure this cannot be fooled by edge-ownership and
    cell-ownership being independent blocks (a halo EDGE between two owned
    cells taints them directly), and unlike a random-fill DEPENDENCY probe it
    is STATE-INDEPENDENT: the default MPAS RHS has genuine state-dependent
    switches (donor-cell vertical advection ``where(sigma_dot>0)``, pressure
    ``maximum``/``clip`` floors, division by ``p_s``) that can zero a halo
    contribution at one state and admit it at another, so a probe can mark an
    entity interior that is halo-dependent at other states. The structural
    closure includes every entity the operator can reach regardless of which
    branch is live, so it over-includes (safe: a slightly larger rim pass) and
    never under-includes (which would be a silent wrong answer). The switches
    do not enlarge the horizontal reach -- they only gate a dependency inside
    the operator's existing connectivity -- so a closure over that
    connectivity is conservative. ``_build_rim_rings_by_dependency`` is kept as
    a VALIDATOR: every entity it flags must lie inside this structural set.

    ``n_rounds`` is the composed-stage count; 3 covers the measured 2-edge-hop
    momentum stencil with a margin. Setup-time numpy only.
    """
    from legoesm.parallel.voronoi_partition import build_local_mesh

    n_dev = len(partitions)
    cell_rim = np.full((n_dev, max_lc), _WIDE_RING_FAR, dtype=np.int32)
    edge_rim = np.full((n_dev, max_le), _WIDE_RING_FAR, dtype=np.int32)

    def _spread(taint_tgt, adj):
        # adj: (deg, N) neighbour indices into taint_tgt; return per-source
        # OR of the target taint over valid neighbours. Shape (N,).
        if adj is None:
            return None
        a = np.asarray(adj)
        if a.ndim != 2:
            return None
        valid = a >= 0
        gathered = np.where(valid, taint_tgt[np.where(valid, a, 0)], False)
        return gathered.any(axis=0)

    for d, part in enumerate(partitions):
        lm = build_local_mesh(global_mesh, part)
        n_oc = part.n_owned_cells
        n_oe = part.n_owned_edges
        n_lc = part.n_local_cells
        n_le = part.n_local_edges
        n_lv = int(getattr(lm, "nVertices", 0))

        def _get(name):
            return getattr(lm, name, None)

        coc = _get("cellsOnCell")       # (deg, nCells) cell->cell
        coe = _get("cellsOnEdge")       # (2, nEdges)   edge->cell
        eoc = _get("edgesOnCell")       # (deg, nCells) cell->edge
        eoe = _get("edgesOnEdge")       # (deg2, nEdges) edge->edge
        voe = _get("verticesOnEdge")    # (2, nEdges)   edge->vertex
        eov = _get("edgesOnVertex")     # (vd, nVertices) vertex->edge
        cov = _get("cellsOnVertex")     # (vd, nVertices) vertex->cell
        voc = _get("verticesOnCell")    # (deg, nCells) cell->vertex

        taint_c = np.zeros(n_lc, dtype=bool); taint_c[n_oc:] = True
        taint_e = np.zeros(n_le, dtype=bool); taint_e[n_oe:] = True
        taint_v = np.zeros(max(n_lv, 1), dtype=bool)

        for _ in range(n_rounds):
            nc = taint_c.copy(); ne = taint_e.copy(); nv = taint_v.copy()
            # cell <- cell
            r = _spread(taint_c, coc);            nc |= r if r is not None else False
            # cell <- edge (edgesOnCell) ; edge <- cell (cellsOnEdge)
            r = _spread(taint_e, eoc);            nc |= r if r is not None else False
            r = _spread(taint_c, coe);            ne |= r if r is not None else False
            # edge <- edge
            r = _spread(taint_e, eoe);            ne |= r if r is not None else False
            # edge <- vertex (verticesOnEdge) ; vertex <- edge (edgesOnVertex)
            if n_lv:
                r = _spread(taint_v, voe);        ne |= r if r is not None else False
                r = _spread(taint_e, eov);        nv |= r if r is not None else False
                # cell <- vertex (verticesOnCell) ; vertex <- cell (cellsOnVertex)
                r = _spread(taint_v, voc);        nc |= r if r is not None else False
                r = _spread(taint_c, cov);        nv |= r if r is not None else False
            taint_c, taint_e, taint_v = nc, ne, nv

        cell_rim[d, :n_oc] = np.where(taint_c[:n_oc], 1, _WIDE_RING_FAR)
        edge_rim[d, :n_oe] = np.where(taint_e[:n_oe], 1, _WIDE_RING_FAR)
    return cell_rim, edge_rim


def _build_rim_rings_by_dependency(
    tendency_fn, sigma, cfg, dt,
    global_mesh, partitions, max_lc, max_le,
    *, n_probe_nlev=4, n_draws=3, seed=0,
):
    """Rim membership by DATA DEPENDENCY, not by a graph-hop heuristic.

    Returns ``(cell_rim, edge_rim)`` with the SAME shape/sentinel contract as
    :func:`_build_rim_rings` (``1`` = rim, ``_WIDE_RING_FAR`` = interior), so
    ``_build_rim_plan`` consumes it unchanged at ``rim_width=1``.

    WHY this exists: the cheap ``min(cell_rim)`` edge measure is WRONG for the
    momentum tendency because edge-ownership and cell-ownership are independent
    blocks — a non-owned (halo) EDGE can sit between two owned, deep-interior
    cells, and an owned edge that reads it via ``edgesOnEdge`` is then
    misclassified interior and silently computed from the pad (measured: ~24
    owned edges wrong by 5e-3, and 2 owned cells wrong, on a metis s6 fixture).
    Rather than re-derive the exact stencil closure over every connectivity
    array the RHS touches (cellsOnEdge, verticesOnEdge, edgesOnVertex, …) and
    risk missing one, ask the OPERATOR itself: run the tendency twice with the
    owned rows held fixed and the HALO rows filled with two different random
    draws. Any owned entity whose tendency changes reads the halo, so it is
    rim. Exact by construction and immune to future operator changes.

    Setup-time only (numpy/CPU-ish; runs the traced RHS on each device's local
    mesh once per draw). ``n_draws`` random fills are unioned so a single
    coincidental cancellation cannot mark a genuinely halo-dependent entity as
    interior.
    """
    from legoesm.core.state import MPASHydrostaticState
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.parallel.voronoi_partition import build_local_mesh

    n_dev = len(partitions)
    cell_rim = np.full((n_dev, max_lc), _WIDE_RING_FAR, dtype=np.int32)
    edge_rim = np.full((n_dev, max_le), _WIDE_RING_FAR, dtype=np.int32)
    # The rim is a TOPOLOGY property, independent of the vertical resolution,
    # so probe with a cheap column and its OWN matching sigma rather than the
    # run's (whose level count is unrelated and would mismatch the buffers).
    nlev = int(n_probe_nlev)
    sigma = create_sigma_coordinate(nlev)
    rng = np.random.default_rng(seed)

    def _state(u, T, ps, phis):
        return MPASHydrostaticState(
            u=Field(data=jnp.asarray(u), name="u", dims=("nEdges", "nlev"),
                    units="m/s", long_name="", staggering="edge"),
            T=Field(data=jnp.asarray(T), name="T", dims=("nCells", "nlev"),
                    units="K", long_name="", staggering="cell"),
            p_s=Field(data=jnp.asarray(ps), name="p_s", dims=("nCells",),
                      units="Pa", long_name="", staggering="cell"),
            phis=Field(data=jnp.asarray(phis), name="phis", dims=("nCells",),
                       units="m^2/s^2", long_name="", staggering="cell"),
            tracers=None)

    for d, part in enumerate(partitions):
        lm = build_local_mesh(global_mesh, part)
        n_lc = part.n_local_cells
        n_le = part.n_local_edges
        n_oc = part.n_owned_cells
        n_oe = part.n_owned_edges
        n_hc = n_lc - n_oc
        n_he = n_le - n_oe
        rim_c = np.zeros(n_oc, dtype=bool)
        rim_e = np.zeros(n_oe, dtype=bool)
        # For each OWNED state draw, run TWO different halo fills and flag any
        # owned entity whose tendency differs between them. The owned state is
        # VARIED across draws (GLM review): a fixed owned state can mask a
        # halo dependence that a state-dependent coefficient zeroes at that one
        # state (a limiter in its inactive plateau, an upwind selector, an
        # owned factor that happens to be 0), which would mark the entity
        # interior and silently compute it wrong at other states. Varying the
        # owned state -- including near-zero and large magnitudes to trip any
        # threshold/branch -- probes those. Union the rim over all draws.
        state_scales = [(5.0, 5.0, 1.0e3, 100.0),
                        (0.05, 0.5, 10.0, 5.0),      # near-flat: gradients ~0
                        (50.0, 60.0, 2.0e4, 3.0e3)]  # large: trip any cap
        draws = max(int(n_draws), len(state_scales))
        for i in range(draws):
            us, Ts, ps_s, phs = state_scales[i % len(state_scales)]
            u_own = rng.normal(scale=us, size=(n_oe, nlev))
            T_own = 250.0 + rng.normal(scale=Ts, size=(n_oc, nlev))
            ps_own = 1.0e5 + rng.normal(scale=ps_s, size=(n_oc,))
            phis_own = rng.normal(scale=phs, size=(n_oc,))
            cur = []
            for _ in range(2):   # two halo fills at THIS owned state
                u = np.empty((n_le, nlev)); u[:n_oe] = u_own
                T = np.empty((n_lc, nlev)); T[:n_oc] = T_own
                ps = np.empty((n_lc,)); ps[:n_oc] = ps_own
                phis = np.empty((n_lc,)); phis[:n_oc] = phis_own
                u[n_oe:] = rng.normal(scale=us, size=(n_he, nlev))
                T[n_oc:] = 250.0 + rng.normal(scale=4 * Ts, size=(n_hc, nlev))
                ps[n_oc:] = 1.0e5 + rng.normal(scale=5 * ps_s, size=(n_hc,))
                phis[n_oc:] = rng.normal(scale=3 * phs, size=(n_hc,))
                t = tendency_fn(_state(u, T, ps, phis), lm, sigma, cfg, dt=dt)
                cur.append((np.asarray(t.du_dt.data[:n_oe]),
                            np.asarray(t.dT_dt.data[:n_oc]),
                            np.asarray(t.dp_s_dt.data[:n_oc])))
            rim_e |= np.any(cur[0][0] != cur[1][0], axis=1)
            rim_c |= np.any(cur[0][1] != cur[1][1], axis=1)
            rim_c |= (cur[0][2] != cur[1][2])
        cell_rim[d, :n_oc] = np.where(rim_c, 1, _WIDE_RING_FAR)
        edge_rim[d, :n_oe] = np.where(rim_e, 1, _WIDE_RING_FAR)
    return cell_rim, edge_rim


def _build_rim_plan(global_mesh, partitions, cell_rim, edge_rim,
                    rim_width, stencil_depth):
    """Per-device compact RIM SUBMESH plan for the interior/rim split.

    v2 after codex review of v1 (two blockers fixed):

    * EDGE SCATTER comes from ``edge_rim`` on DEVICE-owned rows (the
      contiguous shard blocks that production returns), predicate
      ``0 <= edge_rim <= rim_width`` — cut edges carry 0 under the
      min-of-cells rule. v1 adopted the synthetic partition's
      lower-cell edge-ownership, an unrelated set that could miss,
      mis-scatter into halo rows, or double-patch.
    * The closure is built with VECTORIZED numpy (frontier BFS over
      ``cellsOnCell``, mask reductions for edges/vertices) — no
      per-device call into the Python-loop generic partitioner, whose
      cost at production scale (s9/64) is billions of interpreter
      iterations.

    Per device the compact submesh rows are ordered: rim cells first
    (sorted global), then closure cells; scatter-target edges first,
    then remaining closure edges. ``build_local_mesh`` consumes only
    the index/g2l fields of the partition descriptor, so the comm
    schedules are ``None``.

    Returns a list (one entry per device) of dicts with ``sub_mesh``,
    ``n_rim_cells``/``n_rim_edges``, ``cell_gather``/``edge_gather``
    (device-LOCAL rows supplying each submesh entity, submesh order)
    and ``cell_scatter``/``edge_scatter`` (device-LOCAL owned rows
    receiving submesh tendency rows ``[0:n_rim_*)``). Raises if any
    closure entity leaves the device-local region. Setup-time only.
    """
    from legoesm.parallel.voronoi_partition import (
        VoronoiPartition, build_local_mesh,
    )

    coc = np.asarray(global_mesh.cellsOnCell)     # (maxEdges, nCells)
    coe = np.asarray(global_mesh.cellsOnEdge)     # (2, nEdges)
    cov = np.asarray(global_mesh.cellsOnVertex)   # (vDeg, nVertices)
    nCells = int(global_mesh.nCells)
    nEdges = int(global_mesh.nEdges)
    nVertices = int(global_mesh.nVertices)

    plans = []
    for d, part in enumerate(partitions):
        n_owned_c = part.n_owned_cells
        n_owned_e = part.n_owned_edges
        rim_mask = ((cell_rim[d, :n_owned_c] >= 1)
                    & (cell_rim[d, :n_owned_c] <= rim_width))
        rim_global = np.sort(
            np.asarray(part.local_cells)[np.where(rim_mask)[0]])

        # scatter-target edges FIRST (needed to seed the closure):
        # DEVICE-owned rows with rim distance in [0, rim_width] — cut
        # edges are 0 and their far cell lives in the device HALO, so
        # the closure must be seeded from the edges' cells too, not
        # from rim cells alone (v2 fix: 77 scatter edges escaped a
        # rim-only closure on the s3@6 fixture).
        _own_rim = edge_rim[d, :n_owned_e]
        # EXACT-PARTITION tripwire: _build_rim_rings initialises owned
        # rim distance to _WIDE_RING_FAR and its BFS never assigns a
        # finite value above rim_width, so every owned edge is either
        # scatter (0..rim_width) or deep-interior (FAR). Anything else
        # means the ring arrays are inconsistent with this rim_width.
        own_e_rows = np.where(
            (_own_rim >= 0) & (_own_rim <= rim_width))[0]
        interior_e_rows = np.where(_own_rim == _WIDE_RING_FAR)[0]
        if own_e_rows.size + interior_e_rows.size != n_owned_e:
            _n_bad = int(n_owned_e
                         - own_e_rows.size - interior_e_rows.size)
            raise ValueError(
                f"rim plan device {d}: {_n_bad} owned edges have a "
                f"finite rim distance > {rim_width} — inconsistent "
                f"with _build_rim_rings output for this rim_width")
        # A FAR owned edge is deep interior by construction: its whole
        # rim_width stencil lies inside the device-local region, so BOTH
        # adjacent cells must be device-local. If not, the partition is
        # genuinely broken (an adjacent cell absent from the device-local
        # region) and the interior pass would compute from garbage;
        # refuse to build.
        if interior_e_rows.size:
            _int_e = np.asarray(part.local_edges)[interior_e_rows]
            _int_cells = coe[:, _int_e]
            _cell_g2l = np.asarray(part.cell_g2l)
            _bad = (_int_cells < 0) | (_cell_g2l[_int_cells] < 0)
            if _bad.any():
                _n_bad = int(_bad.sum())
                raise ValueError(
                    f"rim plan device {d}: {_n_bad} cells adjacent to "
                    f"interior owned edges are not device-local — "
                    f"partition inconsistent with its local region")
        scatter_e_global = np.sort(
            np.asarray(part.local_edges)[own_e_rows])
        seed_cells = coe[:, scatter_e_global].ravel()
        seed_cells = np.unique(np.concatenate(
            [rim_global, seed_cells[seed_cells >= 0]]))

        # --- vectorized cell closure: BFS depth stencil_depth ---
        in_local = np.zeros(nCells, dtype=bool)
        in_local[seed_cells] = True
        frontier = seed_cells
        for _ in range(stencil_depth):
            if frontier.size == 0:
                break
            nb = coc[:, frontier].ravel()
            nb = nb[nb >= 0]
            nb = np.unique(nb)
            nb = nb[~in_local[nb]]
            in_local[nb] = True
            frontier = nb
        not_rim = in_local.copy()
        not_rim[rim_global] = False
        closure_cells = np.where(not_rim)[0]
        local_cells = np.concatenate([rim_global, closure_cells])

        # --- edges: any adjacent cell local ---
        e_c1, e_c2 = coe[0], coe[1]
        c_loc = np.zeros(nCells + 1, dtype=bool)
        c_loc[:nCells] = in_local
        e_local_mask = c_loc[np.where(e_c1 >= 0, e_c1, nCells)] | \
            c_loc[np.where(e_c2 >= 0, e_c2, nCells)]
        # Restrict to edges the DEVICE partition carries: the production
        # builder's closed cellsOnEdge construction (AND-filter, mesh
        # corners) drops a handful of outer-boundary edges the generic
        # OR-rule would keep. Those sit at maximum distance from every
        # rim entity; the full-tendency closure gate
        # (test_rim_plan_closure) is the arbiter that dropping them
        # never reaches a rim value — scatter edges stay strict below.
        dev_has_edge = part.edge_g2l >= 0
        e_local_mask &= dev_has_edge
        # every scatter edge is in the closure by construction now
        # (its cells seeded the BFS) — keep the assert as a tripwire.
        if not e_local_mask[scatter_e_global].all():
            raise ValueError(
                f"rim plan device {d}: {int((~e_local_mask[scatter_e_global]).sum())} "
                f"scatter-target edges outside the closure — seeding bug")
        e_local_mask_rest = e_local_mask.copy()
        e_local_mask_rest[scatter_e_global] = False
        local_edges = np.concatenate(
            [scatter_e_global, np.where(e_local_mask_rest)[0]])

        # --- vertices: any incident cell local ---
        v_c = cov
        v_local_mask = np.zeros(nVertices, dtype=bool)
        for k in range(v_c.shape[0]):
            ck = v_c[k]
            valid = ck >= 0
            v_local_mask[valid] |= in_local[ck[valid]]
        # boolean-or over incident cells per vertex needs vertex-major
        # reduction; the loop above is over vertexDegree (3), not N.
        local_vertices = np.where(v_local_mask)[0]

        def g2l_of(local_ids, n_global):
            m = np.full(n_global, -1, dtype=np.int32)
            m[local_ids] = np.arange(len(local_ids), dtype=np.int32)
            return m

        rim_part = VoronoiPartition(
            rank=0, n_ranks=2,
            nCells_global=nCells, nEdges_global=nEdges,
            nVertices_global=nVertices,
            n_owned_cells=len(rim_global),
            n_owned_edges=len(scatter_e_global),
            n_owned_vertices=0,
            n_local_cells=len(local_cells),
            n_local_edges=len(local_edges),
            n_local_vertices=len(local_vertices),
            local_cells=local_cells, local_edges=local_edges,
            local_vertices=local_vertices,
            cell_g2l=g2l_of(local_cells, nCells),
            edge_g2l=g2l_of(local_edges, nEdges),
            vertex_g2l=g2l_of(local_vertices, nVertices),
            cell_comm=None, edge_comm=None, vertex_comm=None,
        )
        sub_mesh = build_local_mesh(global_mesh, rim_part)

        cell_gather = part.cell_g2l[local_cells]
        edge_gather = part.edge_g2l[local_edges]
        if (cell_gather < 0).any() or (edge_gather < 0).any():
            raise ValueError(
                f"rim plan device {d}: closure leaves the device-local "
                f"region ({int((cell_gather < 0).sum())} cells, "
                f"{int((edge_gather < 0).sum())} edges) — rim_width="
                f"{rim_width} + stencil_depth={stencil_depth} exceeds "
                f"the partition halo depth")

        plans.append({
            "sub_mesh": sub_mesh,
            "n_rim_cells": int(len(rim_global)),
            "n_rim_edges": int(len(scatter_e_global)),
            "cell_gather": cell_gather.astype(np.int64),
            "edge_gather": edge_gather.astype(np.int64),
            "cell_scatter": cell_gather[:len(rim_global)].astype(np.int64),
            "edge_scatter": edge_gather[:len(scatter_e_global)].astype(np.int64),
        })
    return plans


def _stack_rim_plans(plans):
    """Stack per-device rim plans into shard_map-able arrays.

    Uses the SAME padding machinery as the device meshes
    (:func:`_pad_local_mesh_to`): every submesh is padded to the
    across-device maxima and stacked on a leading device axis;
    gather/scatter index vectors are padded with a trailing GARBAGE
    slot index (the padded buffers carry one sacrificial row, mirroring
    the halo fill's ``max_lc + 1`` convention) so padded lanes read and
    write only garbage. Returns a dict of stacked arrays plus the
    per-device true sizes (int32 vectors) the kernel masks with.
    Setup-time only.
    """
    n_dev = len(plans)
    max_rc = max(p["sub_mesh"].nCells for p in plans)
    max_re = max(p["sub_mesh"].nEdges for p in plans)
    max_rv = max(p["sub_mesh"].nVertices for p in plans)

    padded = [_pad_local_mesh_to(p["sub_mesh"], max_rc, max_re, max_rv)
              for p in plans]
    stacked_sub = jax.tree.map(
        lambda *leaves: jnp.stack(leaves, axis=0), *padded)

    def pad_idx(vecs, width, garbage):
        out = np.full((n_dev, width), garbage, dtype=np.int64)
        for d, v in enumerate(vecs):
            out[d, :len(v)] = v
        return out

    # gather indices point into the device-local (owned+halo+garbage)
    # buffers; scatter indices point into owned+garbage tendency rows.
    # The garbage slot index is the buffer's LAST row, appended by the
    # consumer before the gather/scatter (max_lc / cells_per etc. + 0).
    cg = pad_idx([p["cell_gather"] for p in plans], max_rc, -1)
    eg = pad_idx([p["edge_gather"] for p in plans], max_re, -1)
    max_sc = max(len(p["cell_scatter"]) for p in plans)
    max_se = max(len(p["edge_scatter"]) for p in plans)
    cs = pad_idx([p["cell_scatter"] for p in plans], max_sc, -1)
    es = pad_idx([p["edge_scatter"] for p in plans], max_se, -1)

    return {
        "sub_mesh": stacked_sub,
        "cell_gather": cg, "edge_gather": eg,
        "cell_scatter": cs, "edge_scatter": es,
        "n_rim_cells": np.array([p["n_rim_cells"] for p in plans],
                                dtype=np.int32),
        "n_rim_edges": np.array([p["n_rim_edges"] for p in plans],
                                dtype=np.int32),
        "n_sub_cells": np.array([p["sub_mesh"].nCells for p in plans],
                                dtype=np.int32),
        "n_sub_edges": np.array([p["sub_mesh"].nEdges for p in plans],
                                dtype=np.int32),
    }


class InteriorPad(NamedTuple):
    """Finite fill for the halo rows the interior pass reads before the
    exchange lands. Any finite value is legal: an interior cell's stencil is
    all-owned by construction, so its tendency never depends on these rows;
    they exist only so the operators run without NaN/Inf (a zero surface
    pressure would divide to Inf and could escape a reduction). Rim cells DO
    read them and are wrong here, which is exactly why the rim pass recomputes
    and overwrites them."""
    u: float = 0.0
    T: float = 250.0
    p_s: float = 1.0e5
    phis: float = 0.0


def interior_rim_tendency(
    tendency_fn, sigma, cfg, dt,
    u_owned, T_owned, ps_owned, phis_owned,
    u_full, T_full, ps_full, phis_full,
    local_mesh, sub_mesh,
    cell_gather, edge_gather, cell_scatter, edge_scatter,
    *, pad=InteriorPad(),
):
    """Two-pass interior/rim tendency, equal to a single full-mesh pass.

    The INTERIOR pass runs the RHS on a buffer built from the OWNED shard
    plus a finite pad for the halo rows — it reads nothing the halo exchange
    produces, so a scheduler can run it while the exchange is in flight. Its
    result is correct for owned entities whose radius-R stencil is all-owned
    (rim distance > rim_width) and WRONG for the rim, which read the pad.

    The RIM pass runs the same RHS on the compact ``sub_mesh``, whose fields
    are gathered from the FILLED buffers (``*_full``, owned+halo after the
    exchange). Its tendency for submesh rows ``[0:len(*_scatter))`` — the rim
    entities, in scatter order — is correct, and is scattered back over the
    interior result's owned rim rows.

    Returns ``(du, dT, dps)`` for OWNED rows only. Equal to a single pass on
    the fully-filled local mesh up to floating-point re-association ONLY: the
    rim pass sums each stencil in ``sub_mesh`` order while the full pass sums
    it in ``local_mesh`` order, so identical inputs reassociate. Dry only
    (no tracers); the caller asserts tracers are absent.

    All operations are jnp and index with statically-shaped gather/scatter
    vectors, so the function is jit-able and differentiable. ``*_scatter``
    indices are unique per device (``_build_rim_plan`` guarantees it), so the
    reverse-mode transpose of the scatter is a well-defined gather — no row
    receives two cotangents.
    """
    from legoesm.core.state import MPASHydrostaticState
    n_oe = u_owned.shape[0]
    n_oc = T_owned.shape[0]
    max_le = u_full.shape[0]
    max_lc = T_full.shape[0]

    def _pad(owned, n_full, fill):
        n_pad = n_full - owned.shape[0]
        tail_shape = (n_pad,) + owned.shape[1:]
        return jnp.concatenate(
            [owned, jnp.full(tail_shape, fill, dtype=owned.dtype)], axis=0)

    # Interior pass: owned rows are real, halo rows are the finite pad, and
    # NOTHING here reads *_full — so this pass carries no data dependence on
    # the halo exchange.
    u_i = _pad(u_owned, max_le, pad.u)
    T_i = _pad(T_owned, max_lc, pad.T)
    ps_i = _pad(ps_owned, max_lc, pad.p_s)
    phis_i = _pad(phis_owned, max_lc, pad.phis)
    interior_state = MPASHydrostaticState(
        u=Field(data=u_i, name="u", dims=("nEdges", "nlev"), units="m/s",
                long_name="normal velocity", staggering="edge"),
        T=Field(data=T_i, name="T", dims=("nCells", "nlev"), units="K",
                long_name="temperature", staggering="cell"),
        p_s=Field(data=ps_i, name="p_s", dims=("nCells",), units="Pa",
                  long_name="surface pressure", staggering="cell"),
        phis=Field(data=phis_i, name="phis", dims=("nCells",),
                   units="m^2/s^2", long_name="surface geopotential",
                   staggering="cell"),
        tracers=None,
    )
    tend_i = tendency_fn(interior_state, local_mesh, sigma, cfg, dt=dt)
    du = tend_i.du_dt.data[:n_oe]
    dT = tend_i.dT_dt.data[:n_oc]
    dps = tend_i.dp_s_dt.data[:n_oc]

    # Rim pass: gather the compact submesh from the FILLED buffers. Padded
    # gather/scatter lanes carry -1 (the _stack_rim_plans convention). Append
    # ONE garbage row to every buffer so a -1 index addresses that garbage row
    # (jnp: index -1 = last row) instead of aliasing the last REAL row — codex
    # blocker on the sharded path. For unpadded plans there are no -1 indices,
    # so the extra row is never touched and this is a no-op.
    def _pad1_2d(x):
        return jnp.concatenate([x, jnp.zeros((1,) + x.shape[1:], x.dtype)], 0)

    u_g = _pad1_2d(u_full)
    T_g = _pad1_2d(T_full)
    ps_g = _pad1_2d(ps_full)
    phis_g = _pad1_2d(phis_full)
    rim_state = MPASHydrostaticState(
        u=Field(data=u_g[edge_gather], name="u", dims=("nEdges", "nlev"),
                units="m/s", long_name="normal velocity", staggering="edge"),
        T=Field(data=T_g[cell_gather], name="T", dims=("nCells", "nlev"),
                units="K", long_name="temperature", staggering="cell"),
        p_s=Field(data=ps_g[cell_gather], name="p_s", dims=("nCells",),
                  units="Pa", long_name="surface pressure", staggering="cell"),
        phis=Field(data=phis_g[cell_gather], name="phis",
                   dims=("nCells",), units="m^2/s^2",
                   long_name="surface geopotential", staggering="cell"),
        tracers=None,
    )
    tend_r = tendency_fn(rim_state, sub_mesh, sigma, cfg, dt=dt)
    n_re = edge_scatter.shape[0]
    n_rc = cell_scatter.shape[0]
    # Scatter into a buffer with ONE appended garbage row so a -1 scatter index
    # writes garbage, not a real owned row; drop the garbage row afterward.
    du_p = jnp.concatenate([du, jnp.zeros((1, du.shape[1]), du.dtype)], 0)
    dT_p = jnp.concatenate([dT, jnp.zeros((1, dT.shape[1]), dT.dtype)], 0)
    dps_p = jnp.concatenate([dps, jnp.zeros((1,), dps.dtype)], 0)
    du = du_p.at[edge_scatter].set(tend_r.du_dt.data[:n_re])[:du.shape[0]]
    dT = dT_p.at[cell_scatter].set(tend_r.dT_dt.data[:n_rc])[:dT.shape[0]]
    dps = dps_p.at[cell_scatter].set(tend_r.dp_s_dt.data[:n_rc])[:dps.shape[0]]
    return du, dT, dps


def _resolve_halo_overlap(env_value: str) -> bool:
    """Resolve LEGOESM_MPAS_HALO_OVERLAP: '1' on, '0'/'' off (default).

    On: the ppermute step computes the tendency for owned cells/edges whose
    stencil is entirely device-owned WHILE the halo exchange is in flight, then
    recomputes the boundary rim on a compact submesh once the halo lands. Off
    (default): every existing configuration is byte-identical. Unknown values
    raise (dispatch-hardening, same contract as LEGOESM_MPAS_WIDE_HALO)."""
    if env_value in ("0", ""):
        return False
    if env_value == "1":
        return True
    raise ValueError(
        f"LEGOESM_MPAS_HALO_OVERLAP={env_value!r}: must be '0' or '1' "
        f"(empty = off)")


def _overlap_rim_n_rounds(cfg) -> int:
    """Structural-closure hop count for the interior/rim split, sized to the
    RHS the config selects. 3 covers the bare dry 2-hop stencil; hyperdiffusion
    (del4 = two composed del2s) or APVM reach ~4 incidence hops, so 5 with a
    margin. Over-sizing only grows the rim pass; under-sizing is a silent wrong
    answer, so this rounds UP."""
    wide = (float(getattr(cfg, "nu_del4", 0.0)) > 0.0
            or float(getattr(cfg, "nu_del4_ps", 0.0)) > 0.0
            or float(getattr(cfg, "apvm_scale", 0.0)) > 0.0)
    return 5 if wide else 3


def _resolve_wide_halo(env_value: str) -> bool:
    """Resolve LEGOESM_MPAS_WIDE_HALO: '1' on, '0'/'' off (default).

    Wide halo = communication-avoiding step: ONE halo fill per model
    step at depth ``evals x SPMD_HALO_DEPTH`` instead of one depth-3
    fill per tendency evaluation.  Unknown values raise
    (dispatch-hardening, same contract as LEGOESM_MPAS_RAGGED_HALO)."""
    if env_value in ("0", ""):
        return False
    if env_value == "1":
        return True
    raise ValueError(
        f"LEGOESM_MPAS_WIDE_HALO={env_value!r}: must be '0' or '1' "
        f"(empty = off)")


def _build_ragged_halo_schedule(partitions, cell_owner, n_dev, cells_per,
                                edges_per, max_lc, max_le):
    """One-collective halo schedule for ``jax.lax.ragged_all_to_all``.

    Consumes the SAME directed send/recv maps as the coloured ppermute
    schedule (:func:`_build_halo_send_maps`), so both strategies move
    identical rows; this one groups every neighbour transfer into ONE
    ``ragged_all_to_all`` per entity class (cells, edges) instead of
    ``max_degree`` sequential rounds. Receipt: the microbench
    (bench_halo_collectives, job 26818265) measured 0.475x the coloured
    schedule's per-fill time at 16 GPUs, with the advantage growing
    with device count.

    Layout per device ``d`` (all arrays stacked on a leading device
    axis so they shard as ``P("device")`` args):

    * send buffer: per-destination blocks, destinations ascending;
      ``c_send_idx[d]`` gathers owned-local rows into that order
      (padded with 0 — pad rows are never referenced by the offsets).
    * ``c_in_off[d, i]`` / ``c_send_sz[d, i]``: slice of MY send buffer
      going to device ``i`` (zero-size for non-neighbours).
    * receive staging: per-SOURCE blocks, sources ascending;
      ``c_out_off[d, i]`` is where MY slice lands on receiver ``i``
      (ragged_all_to_all's sender-chosen receiver offset), and
      ``c_recv_sz[d, i]`` the rows I receive from ``i``.
    * ``c_recv_pos[d]``: staging row -> local halo position, padded
      with the ``max_lc`` garbage slot (trimmed by the caller), so the
      staging scatter mirrors the ppermute path's pad-target pattern.

    Offsets/sizes are int32 (the GPU custom call rejects int64) and
    row counts are padded to the global maxima so the stacked arrays
    are rectangular.
    """
    import numpy as np

    (comm_pairs, cell_send_map, cell_recv_map, edge_send_map,
     edge_recv_map) = _build_halo_send_maps(
        partitions, cell_owner, n_dev, cells_per, edges_per)

    def build_entity(send_map, recv_map, garbage_slot):
        send_counts = np.zeros((n_dev, n_dev), np.int64)
        for (src, dst), rows in send_map.items():
            send_counts[src, dst] = len(rows)
        s_tot = send_counts.sum(axis=1)
        r_tot = send_counts.sum(axis=0)
        s_max = max(int(s_tot.max()), 1)
        r_max = max(int(r_tot.max()), 1)

        send_idx = np.zeros((n_dev, s_max), np.int64)
        in_off = np.zeros((n_dev, n_dev), np.int32)
        send_sz = np.zeros((n_dev, n_dev), np.int32)
        out_off = np.zeros((n_dev, n_dev), np.int32)
        recv_sz = np.zeros((n_dev, n_dev), np.int32)
        recv_pos = np.full((n_dev, r_max), garbage_slot, np.int64)

        # Receiver staging offsets: per-source blocks, sources ascending
        stage_off = np.zeros((n_dev, n_dev), np.int64)
        for d in range(n_dev):
            off = 0
            for src in range(n_dev):
                if send_counts[src, d]:
                    stage_off[d, src] = off
                    off += send_counts[src, d]

        for d in range(n_dev):
            off = 0
            for dst in range(n_dev):
                n = int(send_counts[d, dst])
                if n == 0:
                    continue
                send_idx[d, off:off + n] = send_map[(d, dst)]
                in_off[d, dst] = off
                send_sz[d, dst] = n
                out_off[d, dst] = stage_off[dst, d]
                off += n
            for src in range(n_dev):
                n = int(send_counts[src, d])
                if n == 0:
                    continue
                recv_sz[d, src] = n
                so = int(stage_off[d, src])
                recv_pos[d, so:so + n] = recv_map[(d, src)]

        # ragged_all_to_all contract: what I send to i == what i
        # receives from me.
        assert (send_sz == recv_sz.T).all(), "ragged size contract broken"
        return {
            "send_idx": send_idx, "in_off": in_off, "send_sz": send_sz,
            "out_off": out_off, "recv_sz": recv_sz, "recv_pos": recv_pos,
            "s_max": s_max, "r_max": r_max,
        }

    cells = build_entity(cell_send_map, cell_recv_map, max_lc)
    edges = build_entity(edge_send_map, edge_recv_map, max_le)
    return {"cells": cells, "edges": edges,
            "n_pairs": len(comm_pairs)}


def _ragged_halo_fill(cell_pack, u_shard, ragged_sl, max_lc, max_le):
    """Fill (owned + halo) local buffers via ONE ragged_all_to_all per
    entity class (cells, edges) — the grouped-P2P replacement for the
    sequential coloured rounds of :func:`_ppermute_halo_fill`.

    Runs INSIDE ``shard_map``. ``ragged_sl`` is the per-device
    ``P("device")`` slice (leading axis 1) of the stacked
    :func:`_build_ragged_halo_schedule` arrays, ordered
    ``(c_send_idx, c_in_off, c_send_sz, c_out_off, c_recv_sz,
    c_recv_pos, e_send_idx, e_in_off, e_send_sz, e_out_off, e_recv_sz,
    e_recv_pos)``. Same garbage-slot contract as the ppermute path:
    padded staging rows scatter to row ``max_lc`` / ``max_le`` and are
    trimmed on return.
    """
    (c_send_idx, c_in_off, c_send_sz, c_out_off, c_recv_sz, c_recv_pos,
     e_send_idx, e_in_off, e_send_sz, e_out_off, e_recv_sz,
     e_recv_pos) = ragged_sl

    cells_per = cell_pack.shape[0]
    edges_per = u_shard.shape[0]
    cell_local = jnp.pad(cell_pack, ((0, max_lc + 1 - cells_per), (0, 0)))
    u_local = jnp.pad(u_shard, ((0, max_le + 1 - edges_per), (0, 0)))

    send_c = cell_pack[c_send_idx[0]]
    stag_c = jnp.zeros((c_recv_pos.shape[1], cell_pack.shape[1]),
                       cell_pack.dtype)
    recv_c = jax.lax.ragged_all_to_all(
        send_c, stag_c, c_in_off[0], c_send_sz[0], c_out_off[0],
        c_recv_sz[0], axis_name="device")
    cell_local = cell_local.at[c_recv_pos[0]].set(recv_c)

    send_e = u_shard[e_send_idx[0]]
    stag_e = jnp.zeros((e_recv_pos.shape[1], u_shard.shape[1]),
                       u_shard.dtype)
    recv_e = jax.lax.ragged_all_to_all(
        send_e, stag_e, e_in_off[0], e_send_sz[0], e_out_off[0],
        e_recv_sz[0], axis_name="device")
    u_local = u_local.at[e_recv_pos[0]].set(recv_e)

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
    # Wide-halo (communication-avoiding) mode: ONE fill per step at
    # depth evals x SPMD_HALO_DEPTH, whole RK body inside shard_map.
    # Opt-in via LEGOESM_MPAS_WIDE_HALO=1; default OFF = every existing
    # configuration byte-identical.  Motivation (campaign 2026-08-10):
    # the lane is bound by sequential-collective count x ~155-310 us
    # latency floor; s9@64 runs 11 coloured rounds x 3 RK3 fills = 33
    # collectives/step, and the depth-9 comm graph colours to the SAME
    # 11 rounds (job 26845329) with only +17% local extents — so one
    # wide fill cuts 33 -> 11 sequential collectives at ~ +17% payload.
    # ------------------------------------------------------------------
    import os as _os_wide
    # Validate the ballast knob HERE, not only where it is consumed: it
    # is read inside the ppermute fill, so with an allgather or ragged
    # halo both a typo and a deliberate N>1 would be silently ignored
    # and the run would quietly measure nothing (codex).
    import os as _os_bal
    _bal = _resolve_halo_ballast(
        _os_bal.environ.get("LEGOESM_MPAS_HALO_BALLAST", ""))
    if _bal > 1 and halo_strategy not in ("auto", "ppermute"):
        raise ValueError(
            f"LEGOESM_MPAS_HALO_BALLAST={_bal} is only implemented for "
            f"the coloured ppermute exchange, but halo_strategy="
            f"{halo_strategy!r} was requested. The run would move the "
            f"unmodified payload and the measurement would be null.")

    use_wide_halo = _resolve_wide_halo(
        _os_wide.environ.get("LEGOESM_MPAS_WIDE_HALO", "0"))
    if use_wide_halo:
        _integ_name = str(cfg.time_integrator).lower()
        if _integ_name.endswith("_scan"):
            raise ValueError(
                f"LEGOESM_MPAS_WIDE_HALO=1: time_integrator="
                f"{cfg.time_integrator!r} folds its stages into one "
                f"lax.scan body, so the per-evaluation shrinking masks "
                f"cannot be threaded by trace-time call order. Use the "
                f"unrolled spelling (e.g. 'ssp_rk3') with wide halo.")
        _wide_evals = _INTEGRATOR_TENDENCY_EVALS.get(_integ_name)
        if _wide_evals is None:
            raise ValueError(
                f"LEGOESM_MPAS_WIDE_HALO=1: time_integrator="
                f"{cfg.time_integrator!r} has no entry in "
                f"_INTEGRATOR_TENDENCY_EVALS, so the required halo depth "
                f"is unknown. Add the evals count (and its shrinking-"
                f"region justification) before enabling wide halo.")
        _halo_depth_eff = SPMD_HALO_DEPTH * _wide_evals
    else:
        _wide_evals = None
        _halo_depth_eff = SPMD_HALO_DEPTH

    # ------------------------------------------------------------------
    # Setup: build per-device local meshes and gather indices
    # ------------------------------------------------------------------
    pp_sched = None   # set below when the coloured schedule is built
    logger.info(
        "Building halo-partitioned infrastructure for %d device(s) "
        "(nCells=%d, nEdges=%d, halo_depth=%d, strategy=%s%s) ...",
        n_dev, nCells, nEdges, _halo_depth_eff, halo_strategy,
        ", WIDE HALO (1 fill/step)" if use_wide_halo else "",
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
    ) = _build_voronoi_partition_infra(global_mesh, n_dev,
                                       halo_depth=_halo_depth_eff)
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

    # Wide-halo shrinking-mask ring distances (P("device")-sharded jit
    # ARGUMENTS like the meshes/schedules — sharded closure constants
    # raise under multi-controller).  Empty tuple when off.
    if use_wide_halo:
        _cr_np, _er_np = _build_wide_halo_rings(
            global_mesh, partitions_out, max_lc, max_le, _halo_depth_eff)
        wide_args = (
            multiprocess_safe_device_put(_cr_np, dev_sharding),
            multiprocess_safe_device_put(_er_np, dev_sharding),
        )
    else:
        wide_args = ()

    nlev = model.sigma_coord.n_levels

    # ------------------------------------------------------------------
    # Strategy dispatch: ppermute (O(halo)) vs allgather (O(N))
    # ------------------------------------------------------------------

    use_ppermute = halo_strategy == "ppermute"
    # Grouped-P2P variant of the ppermute strategy: ONE ragged_all_to_all
    # per entity class instead of max_degree sequential coloured rounds.
    # SCALE-BANDED (production A/B receipts, campaign doc 2026-08-09):
    # ratio ragged/coloured 0.686 @16 devices, 0.735 @32, 1.220 @64 —
    # the ragged collective pays ~12 us per ZERO-SIZE slice (unpruned
    # no-op sends grow with device count at fixed traffic). Hence
    # "auto" = ragged only up to _RAGGED_AUTO_MAX_NDEV.
    # GPU-only: XLA:CPU has no ragged-all-to-all thunk.
    import os as _os_ragged
    # VALIDATION IS DELIBERATELY UNCONDITIONAL (breaking contract,
    # accepted 2026-08-09, env var is one day old): an invalid value
    # raises even when the resolved strategy is allgather and ragged is
    # unreachable — a typo must never silently pick a halo strategy on
    # the NEXT run where ppermute IS selected. (Single-device runs
    # early-return above and never see this; they have no halo.)
    use_ragged = _resolve_ragged_halo(
        _os_ragged.environ.get("LEGOESM_MPAS_RAGGED_HALO", "0"),
        n_dev) and use_ppermute

    # Interior/rim halo-compute overlap (opt-in, default off). Computes the
    # tendency for owned entities whose stencil is entirely device-owned while
    # the ppermute fill is in flight, then recomputes the rim on a submesh once
    # the halo lands. Requires the ppermute strategy; the standalone core is
    # dry-only, so tracers on this path are refused at the kernel below.
    import os as _os_overlap
    use_overlap = _resolve_halo_overlap(
        _os_overlap.environ.get("LEGOESM_MPAS_HALO_OVERLAP", ""))
    if use_overlap and not use_ppermute:
        raise ValueError(
            "LEGOESM_MPAS_HALO_OVERLAP=1 requires the ppermute halo strategy "
            "(the overlap hides the ppermute fill); this run selected "
            f"halo_strategy={halo_strategy!r}")
    if use_overlap and use_ragged:
        raise ValueError(
            "LEGOESM_MPAS_HALO_OVERLAP=1 and LEGOESM_MPAS_RAGGED_HALO=1 are "
            "mutually exclusive halo strategies")

    if use_ragged:
        t1 = time.time()
        rg_sched = _build_ragged_halo_schedule(
            partitions_out, cell_owner_out, n_dev,
            cells_per, edges_per, max_lc, max_le,
        )
        halo_args = tuple(
            multiprocess_safe_device_put(rg_sched[ent][key], dev_sharding)
            for ent in ("cells", "edges")
            for key in ("send_idx", "in_off", "send_sz", "out_off",
                        "recv_sz", "recv_pos")
        )
        logger.info(
            "  ragged halo schedule: 2 collectives/fill over %d comm "
            "pairs (send rows/dev max: cells %d, edges %d) — "
            "LEGOESM_MPAS_RAGGED_HALO=1",
            rg_sched["n_pairs"], rg_sched["cells"]["s_max"],
            rg_sched["edges"]["s_max"])
        logger.info("  ragged halo schedule built in %.3fs",
                    time.time() - t1)
    elif use_ppermute:
        # Build ppermute schedule: neighbor-only halo exchange
        t1 = time.time()
        pp_sched = _build_ppermute_schedule(
            partitions_out, cell_owner_out, n_dev,
            cells_per, edges_per, max_lc, max_le,
            cell_width=nlev + 2, edge_width=nlev,
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

    # Interior/rim overlap plan (setup-time, gated). The structural rim is
    # sized to the RHS the config selects (dry 3 hops, hyperdiffusion 5); the
    # plan's compact submesh + gather/scatter arrays ride as P("device")
    # shard_map ARGUMENTS, like the stacked meshes and the halo schedule.
    rim_arg = ()
    rim_in_specs = ()
    if use_overlap:
        t1 = time.time()
        _cr, _er = _build_rim_rings_structural(
            global_mesh, partitions_out, max_lc, max_le,
            n_rounds=_overlap_rim_n_rounds(cfg))
        _rim_plans = _build_rim_plan(
            global_mesh, partitions_out, _cr, _er,
            rim_width=1, stencil_depth=2)
        _stacked_rim = _stack_rim_plans(_rim_plans)
        # Host arrays first (multiprocess_safe_device_put passes a replicated
        # leaf through unchanged; a host array is always fully addressable, so
        # the P("device") shard is guaranteed on every controller — same
        # reasoning as areaCell below).
        _sub_dev = jax.tree.map(
            lambda a: multiprocess_safe_device_put(
                np.asarray(a), dev_sharding),
            _stacked_rim["sub_mesh"])
        rim_arg = (
            _sub_dev,
            multiprocess_safe_device_put(
                np.asarray(_stacked_rim["cell_gather"]), dev_sharding),
            multiprocess_safe_device_put(
                np.asarray(_stacked_rim["edge_gather"]), dev_sharding),
            multiprocess_safe_device_put(
                np.asarray(_stacked_rim["cell_scatter"]), dev_sharding),
            multiprocess_safe_device_put(
                np.asarray(_stacked_rim["edge_scatter"]), dev_sharding),
        )
        rim_in_specs = jax.tree.map(lambda _: P("device"), rim_arg)
        logger.info(
            "  interior/rim overlap plan built in %.3fs (n_rounds=%d, "
            "submesh gather width cells/edges %d/%d)", time.time() - t1,
            _overlap_rim_n_rounds(cfg),
            int(_stacked_rim["cell_gather"].shape[-1]),
            int(_stacked_rim["edge_gather"].shape[-1]))

    def _make_local_tendency(tkeys: tuple):

        def _local_tendency(u_shard, T_shard, ps_shard, phis_shard,
                            q_shard, dt_val, mesh_sl, halo_sl, rim_sl=None):
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

            if use_ragged:
                cell_local, u_local = _ragged_halo_fill(
                    cell_pack, u_shard, halo_sl, max_lc, max_le,
                )
            elif use_ppermute:
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

            if use_overlap:
                # Interior pass runs on the OWNED shards (pre-fill, so it is
                # independent of the collective and the scheduler can run it
                # while the halo is in flight); the rim pass reads the FILLED
                # buffers. Dry-only core, so tracers are refused here.
                if tkeys:
                    raise ValueError(
                        "LEGOESM_MPAS_HALO_OVERLAP=1 does not support tracers "
                        "yet; the interior/rim core is dry-only")
                _sub_mesh = jax.tree.map(lambda x: x[0], rim_sl[0])
                du_o, dT_o, dps_o = interior_rim_tendency(
                    mpas_hydrostatic_tendencies, sigma, cfg, dt_val,
                    u_shard, T_shard, ps_shard, phis_shard,
                    u_local, T_local, ps_local, phis_local,
                    my_mesh, _sub_mesh,
                    rim_sl[1][0], rim_sl[2][0], rim_sl[3][0], rim_sl[4][0])
                return (du_o, dT_o, dps_o,
                        jnp.zeros((cells_per, 0), dtype=T_shard.dtype))

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
                          mesh_in_specs, halo_in_specs)
                         + ((rim_in_specs,) if use_overlap else ()),
                out_specs=(P("device"), P("device"), P("device"),
                           P("device")),
                check_vma=False,
            )
            _shard_tendency_cache[tkeys] = fn
        return fn

    # ------------------------------------------------------------------
    # WIDE-HALO kernel: ONE packed fill at depth evals x SPMD_HALO_DEPTH,
    # then the whole RK body on the local region with no further
    # exchange.  Validity shrinks by one tendency reach
    # (SPMD_HALO_DEPTH rings) per evaluation — the Shu-Osher
    # shrinking-region argument — so after the last of N evaluations the
    # state is valid exactly on the owned cells this kernel returns.
    # Outer rings hold progressively stale/garbage values that are
    # sliced away; they cannot reach an owned cell because one tendency
    # reads at most SPMD_HALO_DEPTH rings.  The stage arithmetic on
    # owned cells is the SAME dispatch_integrator arithmetic the
    # per-fill path runs outside shard_map, on bitwise-identical inputs
    # (halo copies of the previous step's owner values).
    # ------------------------------------------------------------------

    def _make_local_wide_step(tkeys: tuple):

        def _local_wide_step(u_shard, T_shard, ps_shard, phis_shard,
                             q_shard, dt_val, mesh_sl, halo_sl, wide_sl):
            cell_ring = wide_sl[0][0]     # (max_lc,) int32
            edge_ring = wide_sl[1][0]     # (max_le,) int32
            _owned_c = jnp.arange(max_lc) < cells_per
            _owned_e = jnp.arange(max_le) < edges_per
            cell_pack = _pack_cell_state(T_shard, ps_shard, phis_shard,
                                         q_shard)
            if use_ragged:
                cell_local, u_local = _ragged_halo_fill(
                    cell_pack, u_shard, halo_sl, max_lc, max_le,
                )
            elif use_ppermute:
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

            T_local, ps_local, phis_local, q_local = _unpack_cell_state(
                cell_local, nlev)
            my_mesh = jax.tree.map(lambda x: x[0], mesh_sl)

            tracers_local = None
            if tkeys:
                tracers_local = {
                    k: Field(data=q_local[:, i * nlev:(i + 1) * nlev],
                             name=k, dims=("nCells", "nlev"),
                             units="kg/kg", staggering="cell")
                    for i, k in enumerate(tkeys)
                }
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

            # Trace-time evaluation counter: the unrolled integrators
            # call the tendency N times SEQUENTIALLY in Python during
            # one trace, so the k-th call gets the k-th shrinking mask
            # (the factory refuses *_scan integrators for exactly this
            # reason).  Fresh per trace: the dict lives in this
            # function's scope.
            _eval_i = {"k": 0}

            def _wide_tendency(s):
                """Full-local-region tendencies, state-shaped (phis
                rides a zero tendency; tracer ADVECTION under the same
                keys) — the local-mesh mirror of ``dyn_tendency_fn``,
                MASKED to the eval's shrinking valid region: entities
                outside ring ``(evals - k) * SPMD_HALO_DEPTH`` get a
                ZERO tendency, freezing their stage values at finite
                fill values.  Their values are never read by a later
                evaluation whose result reaches an owned cell (the
                Shu-Osher shrinking-region argument), and the freeze
                keeps every primal finite — unmasked garbage rings turn
                zero cotangents into NaN (0 * NaN) which the fill
                transpose scatter-adds into owned gradients."""
                _eval_i["k"] += 1
                thr = SPMD_HALO_DEPTH * (_wide_evals - _eval_i["k"])
                keep_c = (_owned_c | (cell_ring <= thr))[:, None]
                keep_e = (_owned_e | (edge_ring <= thr))[:, None]
                tend = mpas_hydrostatic_tendencies(
                    s, my_mesh, sigma, cfg, dt=dt_val,
                )
                tr_tend = None
                if tkeys:
                    tr_tend = {
                        k: s.tracers[k].replace(
                            data=jnp.where(
                                keep_c,
                                tend.tracer_tendencies[k].data, 0.0))
                        for k in tkeys
                    }
                return MPASHydrostaticState(
                    u=s.u.replace(
                        data=jnp.where(keep_e, tend.du_dt.data, 0.0)),
                    T=s.T.replace(
                        data=jnp.where(keep_c, tend.dT_dt.data, 0.0)),
                    p_s=s.p_s.replace(
                        data=jnp.where(keep_c[:, 0],
                                       tend.dp_s_dt.data, 0.0)),
                    phis=s.phis.replace(
                        data=jnp.zeros_like(s.phis.data)),
                    v=s.v,
                    tracers=tr_tend,
                )

            final = dispatch_integrator(
                local_state, _wide_tendency, dt_val, cfg.time_integrator,
            )
            if _eval_i["k"] != _wide_evals:
                raise AssertionError(
                    f"wide halo: integrator {cfg.time_integrator!r} made "
                    f"{_eval_i['k']} tendency evaluations, table says "
                    f"{_wide_evals} — _INTEGRATOR_TENDENCY_EVALS is wrong "
                    f"and the halo depth/masks with it.")

            if tkeys:
                q_owned = jnp.concatenate(
                    [final.tracers[k].data for k in tkeys],
                    axis=-1)[:cells_per]
            else:
                q_owned = jnp.zeros((cells_per, 0), dtype=T_shard.dtype)
            return (final.u.data[:edges_per],
                    final.T.data[:cells_per],
                    final.p_s.data[:cells_per],
                    q_owned)

        return _local_wide_step

    _shard_wide_cache: dict = {}

    def _get_shard_wide_step(tkeys: tuple):
        fn = _shard_wide_cache.get(tkeys)
        if fn is None:
            fn = shard_map(
                _make_local_wide_step(tkeys),
                mesh=jax_mesh,
                in_specs=(P("device"), P("device"), P("device"),
                          P("device"), P("device"), P(),
                          mesh_in_specs, halo_in_specs,
                          jax.tree.map(lambda _: P("device"), wide_args)),
                out_specs=(P("device"), P("device"), P("device"),
                           P("device")),
                check_vma=False,
            )
            _shard_wide_cache[tkeys] = fn
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
                  mesh_arg, halo_arg, area_arg, wide_arg, rim_arg):
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
                    *((rim_arg,) if use_overlap else ()),
                )
                # Static workload-gated fusion barrier (codex rounds
                # 11-12; see _FUSION_BARRIER_WORKLOADS). Every gate
                # operand is trace-time static (closure int + aval
                # shapes) — no retrace; the barrier is an identity for
                # numerics and AD.
                _sig = (n_dev, s.u.data.shape[0],
                        s.T.data.shape[0], s.T.data.shape[1],
                        str(s.u.data.dtype))
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
            #        Wide-halo mode runs the SAME dispatch INSIDE
            #        shard_map after one deep fill (see
            #        _make_local_wide_step); default path unchanged.
            if use_wide_halo:
                u_new, T_new, ps_new, q_new = _get_shard_wide_step(tkeys)(
                    state.u.data, state.T.data, state.p_s.data,
                    state.phis.data, _pack_tracers(state), dt,
                    mesh_arg, halo_arg, wide_arg,
                )
                tr_new = None
                if state.tracers is not None:
                    tr_new = {
                        k: state.tracers[k].replace(
                            data=q_new[..., i * nlev:(i + 1) * nlev])
                        for i, k in enumerate(tkeys)
                    }
                state_new = MPASHydrostaticState(
                    u=state.u.replace(data=u_new),
                    T=state.T.replace(data=T_new),
                    p_s=state.p_s.replace(data=ps_new),
                    phis=state.phis,
                    v=state.v,
                    tracers=tr_new,
                )
            else:
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
            stacked_meshes, halo_args, _area_for_mass, wide_args, rim_arg,
        )
        if return_phys_state:
            return state_new, phys_state_out
        return state_new

    # Effective (post-"auto") strategy, carried on the returned callable
    # so benches/tests can RECORD what actually ran instead of the
    # requested flag (codex M3c-2 MINOR; same pattern as kessler's
    # ``_bound_dt``).  Only multi-device steps carry it — the
    # single-device early return above hands back ``model.step``.
    _voronoi_step._halo_strategy_effective = (
        "ppermute_ragged" if use_ragged else halo_strategy)
    _voronoi_step._wide_halo_effective = use_wide_halo
    _voronoi_step._halo_depth_effective = _halo_depth_eff
    # WHICH SCHEDULE this step actually got. An A/B on the halo schedule that
    # does not record the schedule cannot say whether the knob acted, and one
    # was run that way: the arms differed by less than their own noise and
    # there was no way to tell a null result from a switch that never armed.
    _voronoi_step._coloring_method_effective = (
        pp_sched.get("coloring_method") if pp_sched else None)
    _voronoi_step._n_rounds_effective = (
        pp_sched.get("n_rounds") if pp_sched else None)
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
