"""Single-process lat-band SPMD step for the lat-lon C-grid hydrostatic atm.

The atmosphere analogue of ``ocean.dynamics.sharded_ocean_step``: wrap
``CGridLatLonPrimitiveEquationModel`` in a ``jax.shard_map`` over a 1-D ``"lat"``
device mesh so the dycore runs multi-GPU/TPU with pure ``ppermute``/``psum``
collectives (no mpi4jax). It REUSES the shared grid-agnostic primitives
(``latlon_spmd`` band perms / halo body / pole masks, ``reconstruct_vface_lower``
/ ``to_vface_lower`` for the staggered v, ``batch_psum_spmd`` /
``_spmd_lat_psum_or_none`` for global reductions, ``latlon_mpi`` band-geometry
slicers) and writes NEW only the atm-specific state layout + geometry-band glue.

Built in stages (each independently sbatch-validated on CPU host devices):
  * Stage 2 (THIS): ``shard_state_atm_latlon`` / ``gather_state_atm_latlon`` —
    the 6-field C-grid state layout with the staggered-v ``v_lower`` round-trip.
  * Stage 3-4: SPMD-aware Coriolis + v-face interp operators; the un-jitted
    band body with ``grid=`` threading.
  * Stage 5: ``make_sharded_atm_latlon_step`` + the serial-vs-SPMD equivalence
    gate.

The staggered meridional velocity ``v`` has leading dim ``n_lat+1`` (coprime
with ``n_lat`` for ``N>1``), so it is carried sharded as ``v_lower = v[:n_lat]``
and reconstructed to the full faces inside the shard_map (see
:func:`legoesm.parallel.latlon_spmd.reconstruct_vface_lower`). All other leaves
have leading dim ``n_lat`` and shard ``P("lat")`` directly; longitude is kept
local (periodic). No land/u/v masks (the atm domain is global).
"""
from __future__ import annotations

from contextlib import contextmanager

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from legoesm.parallel.geometry_consistency import (
    assert_schema_agrees, broadcast_checked)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
)


def lat_spec(arr) -> P:
    """``P("lat", None, ...)`` for an array sharded on its leading (lat) axis."""
    return P("lat", *((None,) * (arr.ndim - 1)))


def shard_state_atm_latlon(
    state: CGridLatLonHydrostaticState, mesh,
) -> CGridLatLonHydrostaticState:
    """Lay out a C-grid hydrostatic atm state for the lat-band shard_map.

    Cell/u leaves (``u, T, p_s, phis`` + every tracer; all leading-dim
    ``n_lat``) shard ``P("lat", None, ...)``. The staggered ``v`` (leading dim
    ``n_lat+1``) drops its top pole-wall face -> ``v_lower = v[:n_lat]``
    (``n_lat`` rows, divisible by the device count) sharded the same way; the
    dropped face is the north pole wall (``v[n_lat] == 0`` after any step) and
    is reconstructed inside the body. Mirrors ``ocean.shard_state_latlon`` but
    walks the 6-field atm pytree (bare arrays + a tracers dict, no masks).
    """
    from legoesm.parallel.latlon_spmd import shard_leaf

    _mp = jax.process_count() > 1

    def _put(arr):
        # shard_leaf: single-process -> device_put (byte-unchanged); multi-
        # controller -> per-process local band via make_array_from_process_local_data
        # (no all-gather, no transient global replica — issue #1100).
        return shard_leaf(arr, NamedSharding(mesh, lat_spec(arr)), multiprocess=_mp)

    n_lat = state.T.shape[0]
    v_lower = state.v[:n_lat]
    return state._replace(
        u=_put(state.u),
        v=_put(v_lower),
        T=_put(state.T),
        p_s=_put(state.p_s),
        phis=_put(state.phis),
        tracers={k: _put(val) for k, val in state.tracers.items()},
    )


def build_sharded_held_suarez_state_atm_latlon(
    grid, sigma_coord, mesh, *,
    T_init: float = 300.0,
    p_s_init: float | None = None,
    perturbation_amplitude: float = 1.0,
    seed: int = 42,
) -> CGridLatLonHydrostaticState:
    """Band-local Held-Suarez C-grid state, directly in the sharded layout.

    The #1100 invariant for multi-process runs: **neither global builds nor
    ``device_put`` replication** — every global-shaped leaf is created with
    ``jax.make_array_from_callback``, whose callback is invoked only for the
    row slices owned by THIS process's addressable devices (documented JAX
    semantics: per-addressable-shard callbacks with GLOBAL index slices).  No
    full global array is ever handed to ``device_put`` — the path measured to
    detonate under many-process packing in #1100 (its cross-process
    consistency check amplified even small replicated objects; an
    implementation behaviour we cite as measured, not as API contract).  This
    removes the per-process global-state BUILD that made the route-B lat-lon
    bench OOM under full-node CPU packing (``held_suarez_init_latlon`` +
    ``hydrostatic_to_cgrid`` materialised the full ``(n_lat, n_lon, nlev)``
    state on every process before sharding).

    Bit-identical to
    ``shard_state_atm_latlon(hydrostatic_to_cgrid(held_suarez_init_latlon(
    grid, sigma), grid), mesh)`` for the flat-terrain case — gated by
    ``tests/parallel/test_atm_latlon_bandlocal_build.py``.  The staggered
    ``v`` is created directly as its sharded ``v_lower`` layout (the dropped
    north pole-wall face is identically zero in this at-rest IC, exactly what
    ``gather_state_atm_latlon`` re-appends).

    One deliberate exception: the 2-D lowest-level temperature perturbation
    (``jax.random.normal`` over ``(n_lat, n_lon)``) is evaluated in full on
    every process — identical threefry streams cannot be row-sliced without
    evaluating the whole field, and at bench scale it is O(10 MB) vs the
    O(GB) 3-D leaves the callback path avoids.

    Flat terrain only (the bench IC): there is deliberately no ``phis``
    parameter — the topography variant of ``held_suarez_init_latlon`` would
    need its own band-local surface-pressure callback; extend explicitly
    rather than reuse this builder.
    """
    from legoesm import constants
    from legoesm.core.precision import get_policy

    _dtype = get_policy().storage
    if p_s_init is None:
        p_s_init = constants.p_ref

    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = sigma_coord.n_levels

    # 2-D seed field, exact expression of held_suarez_init_latlon
    key = jax.random.PRNGKey(seed)
    pert2d = jax.random.normal(key, (n_lat, n_lon), dtype=_dtype) * jnp.asarray(
        perturbation_amplitude, dtype=_dtype
    )

    def _slice_shape(gshape, idx):
        return tuple(len(range(*sl.indices(n))) for sl, n in zip(idx, gshape))

    def _make(gshape, cb):
        sharding = NamedSharding(mesh, P("lat", *((None,) * (len(gshape) - 1))))
        return jax.make_array_from_callback(gshape, sharding, cb)

    def _zeros_cb(gshape):
        return lambda idx: jnp.zeros(_slice_shape(gshape, idx), dtype=_dtype)

    def _T_cb(idx):
        shape = _slice_shape((n_lat, n_lon, nlev), idx)
        # EXACT expression of held_suarez_init_latlon (``ones * T_init`` then
        # ``.at[:, :, -1].add(pert)``) so promotion semantics match for
        # strongly-typed ``T_init`` too, not just Python floats.
        block = jnp.ones(shape, dtype=_dtype) * T_init
        return block.at[:, :, -1].add(pert2d[idx[0], idx[1]])

    def _ps_cb(idx):
        shape = _slice_shape((n_lat, n_lon), idx)
        # held_suarez_init_latlon with phis=None: p_s_init * exp(-0/(R_d T))
        phis_block = jnp.zeros(shape, dtype=_dtype)
        return (p_s_init * jnp.exp(
            -phis_block / (constants.R_d * T_init))).astype(_dtype)

    sh_u = (n_lat, n_lon + 1, nlev)
    sh_vlow = (n_lat, n_lon, nlev)   # sharded layout: pole-wall face dropped
    sh_T = (n_lat, n_lon, nlev)
    sh_2d = (n_lat, n_lon)
    return CGridLatLonHydrostaticState(
        u=_make(sh_u, _zeros_cb(sh_u)),
        v=_make(sh_vlow, _zeros_cb(sh_vlow)),
        T=_make(sh_T, _T_cb),
        p_s=_make(sh_2d, _ps_cb),
        phis=_make(sh_2d, _zeros_cb(sh_2d)),
        tracers={},
    )


def gather_state_atm_latlon(
    state: CGridLatLonHydrostaticState, mesh,
) -> CGridLatLonHydrostaticState:
    """Inverse of :func:`shard_state_atm_latlon`: replicate every leaf and
    rebuild the full ``(n_lat+1, ...)`` ``v`` by re-appending the zero north
    pole-wall face. Bit-comparable to the single-device state (whose top v-face
    is the pole wall == 0).

    Multi-controller (route-B ``jax.distributed``, mesh spanning processes):
    replication routes through a jit-compiled identity instead of
    ``device_put`` (see :func:`legoesm.parallel.latlon_spmd.replicate_leaf`,
    the primitive shared with the ocean gather); the single-process path is
    byte-unchanged."""
    from legoesm.parallel.latlon_spmd import replicate_leaf

    rep = NamedSharding(mesh, P())
    _mp = jax.process_count() > 1

    def _get(arr):
        return replicate_leaf(arr, rep, multiprocess=_mp)

    v_lower = _get(state.v)
    v_full = jnp.concatenate([v_lower, jnp.zeros_like(v_lower[:1])], axis=0)
    return state._replace(
        u=_get(state.u),
        v=v_full,
        T=_get(state.T),
        p_s=_get(state.p_s),
        phis=_get(state.phis),
        tracers={k: _get(val) for k, val in state.tracers.items()},
    )


# ==============================================================================
# Stage 7 — cell-centered HydrostaticState <-> sharded C-grid state BRIDGE
# ==============================================================================
# The SPMD step (``make_sharded_atm_latlon_step``) consumes/produces a
# ``CGridLatLonHydrostaticState`` (raw face-staggered arrays, laid out as
# ``v_lower``).  The rest of the system — IC builders, the driver, output,
# restart I/O — speaks the cell-centered, Field-wrapped ``HydrostaticState``.
# These two thin host-boundary adapters reconcile the two by composing the
# EXISTING serial converters (``hydrostatic_to_cgrid`` / ``cgrid_to_hydrostatic``,
# the same ones the serial ``model.step(HydrostaticState)`` uses) with the
# shard/gather above.  The conversion runs on the FULL (un-sharded on entry,
# gathered on exit) arrays — it never executes inside a ``shard_map``, so it adds
# ZERO new SPMD-correctness surface; all sharded numerics stay in the validated
# C-grid step.


def shard_hydrostatic_to_atm_latlon(hs, grid, mesh):
    """Bridge a cell-centered ``HydrostaticState`` to a lat-band-SHARDED C-grid
    state: convert (cell winds -> C-grid faces) THEN shard. ``mesh=None`` returns
    the un-sharded C-grid state (single-device fallback). Inverse of
    :func:`gather_atm_latlon_to_hydrostatic`."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        hydrostatic_to_cgrid)
    c_state = hydrostatic_to_cgrid(hs, grid)
    return c_state if mesh is None else shard_state_atm_latlon(c_state, mesh)


def gather_atm_latlon_to_hydrostatic(c_state, grid, mesh):
    """Inverse of :func:`shard_hydrostatic_to_atm_latlon`: gather the sharded
    C-grid state THEN convert (C-grid faces -> cell winds) to the cell-centered,
    Field-wrapped ``HydrostaticState`` the driver/output/restart contract
    expects. ``mesh=None`` converts the already-full C-grid state directly."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        cgrid_to_hydrostatic)
    full = c_state if mesh is None else gather_state_atm_latlon(c_state, mesh)
    return cgrid_to_hydrostatic(full, grid)


# LatLonGrid scalar fields that stay STATIC per band (uniform bands -> one
# shard_map program); every OTHER LatLonGrid field is a jax.Array and is
# stacked over the band axis + indexed by axis_index in the body.
_ATM_GRID_STATIC_FIELDS = frozenset({"n_lat", "n_lon", "radius", "dlon", "dlat"})


def atm_grid_array_field_names(grid) -> list[str]:
    """Order-stable list of the ``jax.Array`` fields of a ``LatLonGrid`` (all
    except the static scalars). NamedTuple field order, so the host stack and
    the in-body ``[r]`` index agree."""
    import numpy as np
    names = []
    for name in grid._fields:
        if name in _ATM_GRID_STATIC_FIELDS:
            continue
        val = getattr(grid, name)
        if isinstance(val, (jax.Array, np.ndarray)):
            names.append(name)
    return names


def build_band_grids_atm(grid, n_devices: int):
    """Build the ``n_devices`` UNIFORM lat-band ``LatLonGrid`` geometries via the
    tested MPI slicer (no bespoke metric re-derivation).

    ``skip_total_area_reduce=True`` keeps ``total_area`` the GLOBAL full-sphere
    sum on EVERY band (the mass-fixer denominator must stay global; the band
    slicer would otherwise ``global_sum_mpi`` a band-local area — wrong / errors
    without mpi4py). Requires ``n_lat % n_devices == 0`` so all bands share the
    leading shape (one shard_map program). rank 0 = south band, N-1 = north.
    """
    from legoesm.parallel.latlon_mpi import (
        make_latlon_band_layout, slice_latlon_grid_to_band)
    n_lat = int(grid.n_lat)
    if n_devices < 1:
        raise ValueError(f"n_devices must be >= 1, got {n_devices}")
    if n_lat % n_devices != 0:
        raise ValueError(
            f"atm lat-band SPMD requires n_lat ({n_lat}) divisible by "
            f"n_devices ({n_devices}) so every band is uniform (one shard_map "
            f"program). Pick n_devices among the divisors of {n_lat}.")
    fold = getattr(grid, "fold", None)
    return [
        slice_latlon_grid_to_band(
            grid,
            make_latlon_band_layout(r, n_devices, n_lat, int(grid.n_lon), fold),
            skip_total_area_reduce=True)
        for r in range(n_devices)
    ]


def state_finite_scalar(state, axis: str | tuple | None = None):
    """Traced SCALAR bool: True iff EVERY array leaf of ``state`` is finite.

    The M2b in-graph blowup guard: inside a ``shard_map`` pass ``axis`` (the
    mesh axis name, or a TUPLE of axis names for the 2-D ``("lat", "lon")``
    tile mesh) so the per-band non-finite presence is ``psum``-reduced —
    every device then returns the SAME replicated scalar (safe for
    ``out_specs=P()``), and the host reads ONE scalar per segment instead of
    gathering + host-syncing the full state.  Covers ALL state leaves (u,
    v_lower, T, p_s, phis, every tracer) — a superset of the historical
    gathered p_s/T host check, so any blowup the old guard caught this one
    catches at the same segment boundary.  Uses per-leaf ``jnp.any`` (0/1
    presence, no count) so the reduction cannot overflow.
    """
    bad = jnp.zeros((), dtype=jnp.int32)
    for leaf in jax.tree.leaves(state):
        bad = bad | jnp.any(~jnp.isfinite(leaf)).astype(jnp.int32)
    if axis is not None:
        bad = jax.lax.psum(bad, axis)   # 0 iff every band is clean
    return bad == 0


def atm_latlon_geometry_bytes(grid, n_devices: int) -> dict:
    """Per-device geometry residency of the lat-band SPMD step, computed from
    the ACTUAL band-grid array shapes (no fabricated numbers).

    ``replicated_per_device_bytes``: the historical layout — every device
    holds the full all-band stack of every ``LatLonGrid`` array field
    (:func:`atm_grid_array_field_names`; dominated by the five
    ``(n_lat, n_lon)`` 2-D fields ``lat2d, lon2d, f, dx, area``).
    ``sharded_per_device_bytes``: the ``shard_geometry=True`` layout — each
    device holds only its own band's slice (exactly ``replicated /
    n_devices``; the stack leading dim is ``n_devices`` and bands are
    uniform).  Excludes the optional polar-filter mask stacks (two
    ``(n_lat,)``-scale vectors when the filter is on) — negligible next to
    the 2-D fields and absent in the default configs.
    """
    band_grids = build_band_grids_atm(grid, n_devices)
    template = band_grids[0]
    names = atm_grid_array_field_names(template)
    band_bytes = sum(int(jnp.asarray(getattr(template, n)).nbytes)
                     for n in names)
    total = band_bytes * n_devices          # the (n_dev, band...) stacks
    return {
        "n_geometry_fields": len(names),
        "replicated_per_device_bytes": int(total),
        "sharded_per_device_bytes": int(total // n_devices),
    }


def _build_geometry_stacks(model, mesh, n_dev: int, shard_geometry: bool):
    """Stack every band's ``LatLonGrid`` array fields (+ the optional
    polar-filter masks) over a leading band axis and lay them out on ``mesh``.

    ``shard_geometry=False`` (the historical layout): every stack is
    ``device_put`` REPLICATED (``P()``) — each device holds ALL bands'
    geometry and the body indexes its own band at ``axis_index``.

    ``shard_geometry=True`` (M2b): every stack is sharded ``P("lat", ...)``
    on the leading band axis — each device holds ONLY its own band's slice
    (leading extent 1 inside the shard_map body, static index ``[0]``).  The
    VALUES the body consumes are identical either way (the same band slice),
    so the step numerics are bit-unchanged; only the residency changes
    (per-device geometry bytes drop by ``n_dev`` —
    :func:`atm_latlon_geometry_bytes`).

    Returns ``(template, array_field_names, stacks, stacks_spec)``.
    """
    grid = model.grid
    band_grids = build_band_grids_atm(grid, n_dev)
    template = band_grids[0]
    array_field_names = atm_grid_array_field_names(template)
    raw = {
        name: jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
                        axis=0)
        for name in array_field_names
    }
    # Per-band polar-filter masks (only when the filter is on): slice the
    # global masks [s:e] (cell) / [s:e+1] (v-face stagger) and stack.  Absent
    # otherwise -> the body passes None -> _step_cgrid_impl resolves to
    # self._polar_mask (also None), no filter.
    nl = int(grid.n_lat) // n_dev
    if model._polar_mask is not None:
        raw["__polar_mask"] = jnp.stack(
            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(n_dev)],
            axis=0)
        raw["__polar_mask_v"] = jnp.stack(
            [model._polar_mask_v[r * nl:r * nl + nl + 1] for r in range(n_dev)],
            axis=0)
    # #1362: the band geometry above is RECOMPUTED per process from the same
    # config, and per-process XLA autotuning on device-derived grid fields
    # makes the last ULPs differ at larger sizes -- which trips the
    # bit-identical assert inside a replicated device_put (the ocean lane hit
    # exactly this at LL576 np=4; this lane hit it at LL768).  Verify
    # cross-process agreement, then broadcast process 0's bytes.  Guarded, not
    # blind: a REAL divergence (different polar masks = different filtering =
    # different physics) RAISES instead of being masked by process 0.
    #
    # The schema gate matters more here than in the ocean lane: the polar-mask
    # entries are CONDITIONAL on ``model._polar_mask``, so a per-process
    # difference in that one setting changes the field LIST itself, which
    # would desynchronize the per-field gathers rather than fail cleanly.
    ordered_names = list(raw)
    assert_schema_agrees(ordered_names, n_dev,
                         context="make_sharded_atm_latlon_step")
    raw = {
        name: jnp.asarray(broadcast_checked(
            raw[name], name, context="make_sharded_atm_latlon_step"))
        for name in ordered_names
    }
    spec_of = lat_spec if shard_geometry else (lambda _arr: P())
    stacks = {
        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
        for name, arr in raw.items()
    }
    stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}
    return template, array_field_names, stacks, stacks_spec


def _make_band_step_body(model, template, array_field_names, axis,
                         perm_north, physics_fn, shard_geometry: bool):
    """One band's un-jitted C-grid step body — shared by the per-step
    shard_map (:func:`make_sharded_atm_latlon_step`) and the compiled segment
    scan (:func:`make_sharded_atm_latlon_segment`) so the band numerics are
    written ONCE.

    Returns ``band_step(state_local, stacks_local, dt, ps_local) ->
    (state_out_local, ps_out)`` operating on the band-local ``v_lower``
    layout.  ``shard_geometry`` selects the geometry index (STATIC Python
    bool, feature-gating exception): sharded stacks arrive with a leading
    extent of 1 (static index 0); replicated stacks carry all bands
    (dynamic index at ``axis_index``).  Same band values either way.
    """
    from legoesm.parallel.latlon_spmd import (
        reconstruct_vface_lower, to_vface_lower, spmd_pole_end_masks)

    def band_step(state_local, stacks_local, dt, ps_local):
        gi = 0 if shard_geometry else jax.lax.axis_index(axis)
        band_geom = template._replace(
            **{name: stacks_local[name][gi] for name in array_field_names})
        pmask = (stacks_local["__polar_mask"][gi]
                 if "__polar_mask" in stacks_local else None)
        pmaskv = (stacks_local["__polar_mask_v"][gi]
                  if "__polar_mask_v" in stacks_local else None)
        # Reconstruct the band's nl+1 v-faces from v_lower (shared interface
        # row via ppermute), run the un-jitted band step, convert v back.
        v_full = reconstruct_vface_lower(state_local.v, axis, perm_north)
        state_band = state_local._replace(v=v_full)
        out, ps_out = model._step_cgrid_impl(
            state_band, dt,
            physics_fn=physics_fn, phys_state=ps_local,
            grid=band_geom, sigma_coord=model.sigma_coord,
            polar_mask=pmask, polar_mask_v=pmaskv,
            pole_v_bc_masks=spmd_pole_end_masks(),
        )
        return out._replace(v=to_vface_lower(out.v)), ps_out

    return band_step


def _dtype_sig(tree) -> tuple:
    """Trace-time carry signature of a pytree: the pytree STRUCTURE plus
    every array leaf's ``(shape, dtype, weak_type)``.

    ``lax.scan`` requires carry-in == carry-out in ALL of structure, shape,
    dtype and weak type — dtype strings alone would mark a step that flips
    weak typing (or shape) while preserving dtypes as "stable" and then
    fail inside the scan lowering instead of being absorbed by the unroll
    (codex M3b review).  Non-array leaves contribute through the treedef.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return (str(treedef),
            tuple((tuple(leaf.shape), str(leaf.dtype),
                   bool(getattr(leaf, "weak_type", False)))
                  for leaf in leaves if hasattr(leaf, "dtype")))


def unroll_to_dtype_fixed_point(step1, state, n_left: int):
    """Unroll ``step1`` applications until the state's dtype signature is a
    FIXED POINT of the step (bounded by ``n_left``); returns
    ``(state, n_left_remaining)``.

    PUBLIC (M3b): shared by this module's lat-band segments and the tiled
    cube segment (``tiled_step_adapter.scan_tiled_cc_steps``) — any
    ``lax.scan``-of-a-step needs carry-in == carry-out dtypes, and this is
    the ONE place that trace-time unroll lives (no re-derivation).

    Why: ``lax.scan`` needs carry-in == carry-out dtypes.  A mixed-precision
    IC (f32 grid/init-derived leaves beside the step's f64 sources under
    x64 — the f64 sigma-coordinate arrays and the strong-f64
    ``jnp.asarray(dt)`` operand) promotes over the first stepS exactly as
    the per-step Python loop absorbs silently: observed on the bench IC
    (jobs 8916406/8916740), ``p_s`` promotes f32->f64 in step 1 and
    ``u/v/T`` follow in step 2 by mixing the now-f64 ``p_s`` — the fixed
    point can take MORE than one application, and no role-based cast can
    express it (the default f32 policy no-ops while the promotion is
    mixed-LEAF arithmetic).

    ``jax.eval_shape`` probes the step's output dtypes ABSTRACTLY at trace
    time (zero FLOPs), so exactly the needed number of steps is unrolled —
    NONE for an already dtype-stable state (the parity-gate f64 states scan
    all ``n_steps``).  The unroll count is a trace-time constant baked into
    the compiled program: every segment call executes the same
    ``k unrolled + scan(n_left)`` schedule with ``k + n_left == n_steps``.
    Zero extra casts, zero numerical difference vs the per-step lane;
    strictly monotone leaf promotion over a finite dtype lattice guarantees
    termination, and the ``n_left`` bound caps the unroll at the segment
    length (a 1-step segment simply runs its single step unrolled).
    """
    while (n_left > 0
           and _dtype_sig(jax.eval_shape(step1, state)) != _dtype_sig(state)):
        state = step1(state)
        n_left -= 1
    return state, n_left


def _refuse_unsupported_spmd_config(model) -> None:
    """Dispatch-hardening shared by the step + segment factories: only a
    non-fold lat-lon grid with an SPMD-safe mass path is supported.  Fail
    LOUD rather than silently mis-fold / band-local-sum."""
    fold = getattr(model.grid, "fold", None)
    if fold is not None and bool(getattr(fold, "is_active", False)):
        raise NotImplementedError(
            "atm lat-band SPMD: tripole north-fold is a follow-up.")
    if getattr(model.config, "anchor_mass_to_initial", False):
        raise NotImplementedError(
            "atm lat-band SPMD: anchor_mass_to_initial uses a band-local "
            "jnp.sum(p_s*area) target that is not yet SPMD-routed; disable it "
            "or use fix_mass with the pre-state (psum'd) path.")


@contextmanager
def _latlon_spmd_armed(mesh):
    """Arm the lat-band SPMD halo backend around a (possibly tracing) call,
    restoring the FULL previous backend state (backend + MPI topology + SPMD
    mesh) after — a later serial/full-domain call must not take SPMD-only
    branches outside a shard_map.  The first call through a jitted fn traces
    with the backend armed (baking the band halo); later calls reuse the
    cached compile and the arm/restore keeps interleaved serial paths
    untouched."""
    from legoesm.grids.halo import (
        get_halo_backend, get_mpi_topology, get_spmd_mesh,
        set_halo_backend, set_spmd_mesh)
    from legoesm.parallel.latlon_spmd import activate_latlon_spmd_halo
    prev_backend = get_halo_backend()
    prev_topo = get_mpi_topology()
    prev_mesh = get_spmd_mesh()
    activate_latlon_spmd_halo(mesh)
    try:
        yield
    finally:
        set_spmd_mesh(prev_mesh)
        set_halo_backend(prev_backend, prev_topo)


def make_sharded_atm_latlon_step(model, mesh, physics_fn=None, *,
                                 shard_geometry: bool = False):
    """Return ``step(c_state, dt) -> c_state`` running the C-grid hydrostatic atm
    step lat-band-SPMD over the 1-D ``"lat"`` mesh.

    ``c_state`` is a ``CGridLatLonHydrostaticState`` laid out with
    :func:`shard_state_atm_latlon` (``v`` carried as the ``n_lat``-row
    ``v_lower``). The body reconstructs each band's ``nl+1`` v-faces, runs the
    UN-jitted ``model._step_cgrid_impl`` on the band geometry + band polar masks
    + per-band pole masks, and converts the result ``v`` back to ``v_lower``.
    REUSES the shared primitives (band perms, the v-face round-trip, the band
    halo via the swapped backend, ``spmd_pole_end_masks``, the band slicer);
    atm-NEW is only the 6-field state/geometry walk. ``check_vma=False`` (the
    band halo reads neighbour-rank data). Mirrors ``make_sharded_ocean_step``.

    ``physics_fn`` (optional): a STATELESS, COLUMN-LOCAL physics closure
    (``physics_fn(hs, grid, sigma_coord) -> HydrostaticTendencies``, e.g.
    Held-Suarez or any per-column parameterization). It is evaluated inside each
    RK stage on the BAND geometry (``_step_cgrid_impl`` routes the band grid into
    ``_call_physics`` so a lat-dependent forcing sees the band's latitudes), and
    its wind tendencies couple cell->face through the SPMD-aware
    ``interp_cell_to_vface_halo`` — so column-local physics is decomposition-
    invariant with NO collectives. The only global reductions in the step
    (``zero_mean_tendency``, the mass fixer) are already "lat"-psum-routed.

    STATEFUL physics: a DETERMINISTIC prognostic carry (tke/qke, convection
    profiles, GWD spectrum) IS SPMD-routed — pass it per call:
    ``step(c_state, dt, phys_state=ps) -> (c_state, ps)``.  Every
    ``(ncol=n_lat*n_lon, ...)`` ``PhysicsState`` leaf band-splits on dim 0
    (the ColumnAdapter flatten is a C-order lat-major reshape, so a
    contiguous dim-0 shard is exactly the band's own columns); the
    ``prng_key (2,)`` and other non-column leaves replicate.  STOCHASTIC
    schemes are decomposition-INVARIANT since increment 2: the Bechtold AR1
    innovation folds the per-step sub-key with each column's GLOBAL id
    (``PhysicsState.col_index``, band-split with the carry), so a physical
    column draws the same variate under any decomposition.

    ``shard_geometry`` (M2b, default ``False`` = historical layout,
    byte-identical): ``True`` lays the band-geometry stacks out SHARDED
    ``P("lat")`` on the band axis — each device holds only its own band's
    geometry slice instead of a replicated all-band copy (1/n_dev the
    bytes, :func:`atm_latlon_geometry_bytes`).  The body consumes the SAME
    band values either way, so the step is bit-identical (gated by
    ``tests/parallel/test_atm_latlon_segment.py``).
    """
    from legoesm.parallel.latlon_spmd import latlon_band_perms
    from legoesm.parallel.shard_map_compat import shard_map

    # Stochastic physics is SPMD-safe since increment 2: the Bechtold AR1
    # innovation folds the per-step sub-key with each column's GLOBAL id
    # (``PhysicsState.col_index`` — band-split with the carry, so every
    # shard holds its own global ids) and the replicated master key splits
    # identically on every band — the draw is decomposition-INVARIANT.
    # Deterministic prognostic carries (tke/qke, conv profiles, GWD
    # spectrum) thread exactly: the ColumnAdapter flatten is a C-order
    # (lat-major) reshape, so a contiguous dim-0 shard of every
    # ``(ncol, ...)`` PhysicsState leaf IS the band's own columns.

    # A stateful (tagged) physics_fn with NO carry would silently reseed
    # its PhysicsState every step (issue #405/#413) — model.step()'s
    # guard is bypassed here, so the returned step re-checks per call.
    from legoesm.timestepping.integration import (
        refuse_unthreaded_stateful_physics)

    if mesh is None:                       # single-device: plain C-grid step
        def _serial_step(c_state, dt, phys_state=None):
            refuse_unthreaded_stateful_physics(
                physics_fn, phys_state, where="atm lat-band SPMD step")
            out, ps_out = model._step_cgrid(
                c_state, dt, physics_fn=physics_fn, phys_state=phys_state)
            return out if phys_state is None else (out, ps_out)
        return _serial_step

    n_dev = mesh.devices.size
    axis = mesh.axis_names[0]
    grid = model.grid

    _refuse_unsupported_spmd_config(model)

    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
        model, mesh, n_dev, shard_geometry)

    perm_north, _perm_south = latlon_band_perms(n_dev)
    band_step = _make_band_step_body(
        model, template, array_field_names, axis, perm_north, physics_fn,
        shard_geometry)

    def _body(state_local, stacks_local, dt):
        out, _ = band_step(state_local, stacks_local, dt, None)
        return out

    def _body_with_carry(state_local, stacks_local, dt, ps_local):
        # Stateful variant: the band's PhysicsState chunk (ncol_band =
        # nl*n_lon leading dim — the C-order lat-major flatten makes a
        # contiguous dim-0 shard exactly the band's own columns) is
        # threaded into every RK stage and the carry-out returned.
        return band_step(state_local, stacks_local, dt, ps_local)

    _ncol_global = int(grid.n_lat) * int(grid.n_lon)

    def _ps_spec_leaf(leaf):
        # (ncol, ...) leaves band-split on dim 0 (lat-major flatten);
        # everything else (prng_key (2,), scalars) replicated.
        if (hasattr(leaf, "ndim") and leaf.ndim >= 1
                and leaf.shape[0] == _ncol_global):
            return P("lat")
        return P()

    # Build the JITTED shard_map ONCE and cache it. ``jax.jit`` is LOAD-BEARING:
    # a bare shard_map is NOT compilation-cached, so calling it re-traces +
    # recompiles the (large, un-jitted) band step EVERY call — a 32x64x10 nd=2
    # step took ~142 s/step (bench 8560671), and the whole equivalence gate ran
    # ~65 min. Wrapping in jit caches the compile: probe 8561202 measured
    # [3079, 1.4, 1.2, 1.1, 1.1] ms — first call compiles, the rest hit the
    # cache. ``dt`` is a TRACED operand (not a closure constant) so a changing dt
    # does not retrigger compilation. The grid-tracer concern that kept
    # _step_cgrid_impl un-jitted does NOT bite here: band_geom's STATIC scalar
    # fields stay concrete (template._replace only swaps the array fields), and
    # the SPMD operator retrofits removed the trace-time static-bool checks on
    # the cut/pole branches.
    _cache = {}

    def sharded_step(c_state, dt, phys_state=None):
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="atm lat-band SPMD step")
        # Cache key = (state pytree STRUCTURE, phys_state pytree structure):
        # in_specs/out_specs derive from both, so a structure change (optional
        # field None <-> Field, or the phys carry appearing/disappearing) must
        # rebuild the shard_map rather than reuse stale specs (codex finding,
        # ocean-twin parity). ``phys_state`` threading is the AIMIP-branch
        # feature main lacks (main rejects a non-None carry here).
        key = (jax.tree.structure(c_state),
               None if phys_state is None
               else jax.tree_util.tree_structure(phys_state))
        fn = _cache.get(key)
        if fn is None:
            in_spec = jax.tree.map(lat_spec, c_state)
            if phys_state is None:
                fn = jax.jit(shard_map(
                    _body, mesh=mesh, in_specs=(in_spec, stacks_spec, P()),
                    out_specs=in_spec, check_vma=False))
            else:
                ps_spec = jax.tree.map(_ps_spec_leaf, phys_state)
                fn = jax.jit(shard_map(
                    _body_with_carry, mesh=mesh,
                    in_specs=(in_spec, stacks_spec, P(), ps_spec),
                    out_specs=(in_spec, ps_spec), check_vma=False))
            _cache[key] = fn
        # Arm the SPMD band halo around the call ONLY (see _latlon_spmd_armed).
        with _latlon_spmd_armed(mesh):
            if phys_state is None:
                return fn(c_state, stacks, jnp.asarray(dt))
            return fn(c_state, stacks, jnp.asarray(dt), phys_state)

    sharded_step._geom_stacks = stacks   # test/introspection only
    return sharded_step


def make_sharded_atm_latlon_segment(model, mesh, n_steps: int,
                                    physics_fn=None, *,
                                    shard_geometry: bool = True):
    """Return ``segment(c_state, dt) -> (c_state, all_finite)`` advancing
    ``n_steps`` C-grid steps in ONE compiled program — a ``lax.scan`` of the
    band step inside a single jitted ``shard_map``, built once and reused
    (the M2b lever: "compile atmosphere lat-lon segments instead of
    launching one step at a time").

    Contrast with driving :func:`make_sharded_atm_latlon_step` in a Python
    loop: ONE host dispatch (+ halo-backend arm/restore + cache-key hash) per
    SEGMENT instead of per STEP, and no per-step host round-trip between
    device launches.  The scanned band body is the SAME
    ``_make_band_step_body`` the per-step path runs, so the trajectory
    matches the sequential sharded steps to compilation-order roundoff
    (gated at 1e-12 by ``tests/parallel/test_atm_latlon_segment.py``).

    ``all_finite`` is a REPLICATED traced scalar bool from
    :func:`state_finite_scalar` — the in-graph blowup guard (``psum`` of
    per-band non-finite presence over ALL state leaves).  The host reads
    this ONE scalar per segment instead of gathering the full state.

    ``shard_geometry=True`` (default — a NEW API, no historical layout to
    preserve): each device holds ONLY its own band's geometry slice
    (``P("lat")`` stacks) instead of a replicated all-band copy —
    bit-identical numerics, 1/n_dev the geometry bytes
    (:func:`atm_latlon_geometry_bytes`).

    STATELESS physics only (``None`` / Held-Suarez / column-local closures,
    the production ``run_atm_latlon_spmd`` envelope): a stateful
    ``PhysicsState`` carry is refused loudly — thread it through the
    per-step :func:`make_sharded_atm_latlon_step` until the segment lane
    routes the carry through the scan.

    ``mesh=None``: the single-device twin — ``jit(lax.scan)`` over the serial
    C-grid step with the model's own geometry, same ``(state, all_finite)``
    contract.

    ``n_steps`` is STATIC (the compiled scan length): one compiled program
    per distinct segment length (``run_atm_latlon_spmd`` caches per length —
    at most two: the regular segment and the final remainder).

    Carry dtype: leading steps are UNROLLED outside the ``lax.scan`` until
    the state's dtype signature is a fixed point of the step
    (:func:`unroll_to_dtype_fixed_point` — ``jax.eval_shape`` probe, zero
    FLOPs, trace-time constant).  A mixed-precision IC promotes over the
    first stepS (``p_s`` first, ``u/v/T`` next via the promoted ``p_s`` —
    observed jobs 8916406/8916740) exactly as the per-step Python loop
    absorbs silently; an already-stable state unrolls NOTHING and scans all
    ``n_steps``.  Zero extra casts, zero numerical difference vs the
    per-step lane — never a silent precision change.
    """
    from legoesm.parallel.latlon_spmd import latlon_band_perms
    from legoesm.parallel.shard_map_compat import shard_map
    from legoesm.timestepping.integration import (
        refuse_unthreaded_stateful_physics)

    if int(n_steps) < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    n_steps = int(n_steps)

    def _refuse_carry(phys_state):
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="atm lat-lon compiled segment")
        if phys_state is not None:
            raise NotImplementedError(
                "make_sharded_atm_latlon_segment: a stateful PhysicsState "
                "carry is not yet segment-routed — use the per-step "
                "make_sharded_atm_latlon_step(phys_state=...) path.")

    if mesh is None:                       # single-device compiled segment
        def _serial_seg(c_state, dt):
            def _step1(s):
                out, _ps = model._step_cgrid_impl(
                    s, dt, physics_fn=physics_fn, phys_state=None)
                return out
            # Unroll to the scan-carry dtype fixed point (helper docstring).
            out, n_left = unroll_to_dtype_fixed_point(
                _step1, c_state, n_steps)
            if n_left > 0:
                out, _ = jax.lax.scan(lambda s, _x: (_step1(s), None),
                                      out, xs=None, length=n_left)
            return out, state_finite_scalar(out)

        fn_serial = jax.jit(_serial_seg)

        def serial_segment(c_state, dt, phys_state=None):
            _refuse_carry(phys_state)
            return fn_serial(c_state, jnp.asarray(dt))

        return serial_segment

    n_dev = mesh.devices.size
    axis = mesh.axis_names[0]

    _refuse_unsupported_spmd_config(model)

    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
        model, mesh, n_dev, shard_geometry)
    perm_north, _perm_south = latlon_band_perms(n_dev)
    band_step = _make_band_step_body(
        model, template, array_field_names, axis, perm_north, physics_fn,
        shard_geometry)

    def _seg_body(state_local, stacks_local, dt):
        def _step1(s):
            out, _ps = band_step(s, stacks_local, dt, None)
            return out
        # Unroll to the scan-carry dtype fixed point (helper docstring).
        out, n_left = unroll_to_dtype_fixed_point(
            _step1, state_local, n_steps)
        if n_left > 0:
            out, _ = jax.lax.scan(lambda s, _x: (_step1(s), None),
                                  out, xs=None, length=n_left)
        return out, state_finite_scalar(out, axis=axis)

    _cache = {}

    def segment(c_state, dt, phys_state=None):
        _refuse_carry(phys_state)
        # Cache key = state pytree STRUCTURE (in/out specs derive from it) —
        # same doctrine as the per-step factory.
        key = jax.tree.structure(c_state)
        fn = _cache.get(key)
        if fn is None:
            in_spec = jax.tree.map(lat_spec, c_state)
            fn = jax.jit(shard_map(
                _seg_body, mesh=mesh,
                in_specs=(in_spec, stacks_spec, P()),
                out_specs=(in_spec, P()), check_vma=False))
            _cache[key] = fn
        with _latlon_spmd_armed(mesh):
            return fn(c_state, stacks, jnp.asarray(dt))

    segment._geom_stacks = stacks   # test/introspection only
    return segment


def run_atm_latlon_spmd_segment(model, mesh, hs_init, dt, n_steps,
                                physics_fn=None, phys_state=None):
    """Run ``n_steps`` of the lat-band-SPMD C-grid hydrostatic step from a
    cell-centered ``HydrostaticState``, returning a ``HydrostaticState``.

    The Stage-7 driver seam: it bridges the cell-centered driver/output/restart
    contract to the sharded C-grid step. Convert+shard ONCE on entry, run the
    whole ``n_steps`` purely in the sharded C-grid layout, gather+convert ONCE on
    exit. This MATCHES the serial ``for _ in range(n): hs = model.step(hs, dt,
    physics_fn=physics_fn)`` loop, which — via the model's ``id()``-keyed
    ``_cgrid_cache`` — likewise stays in the C-grid layout across the loop and
    only converts cell<->face at the segment boundaries. So the (lossy)
    cell->face->cell round-trip happens ONCE on both paths, not per step, and the
    two trajectories reduce to the Stage-5 C-grid equivalence (tendency bit-exact
    for dynamics + column-local physics; the limited-FV-PPM cut truncation
    bounded).

    ``physics_fn`` (optional): a COLUMN-LOCAL physics closure (e.g.
    Held-Suarez / any per-column parameterization). Evaluated per RK stage on
    the band geometry; decomposition-invariant with no collectives.
    ``mesh=None`` runs the single-device step. A stateful physics threads
    its ``PhysicsState`` via ``phys_state=`` (band-split carry; see
    :func:`make_sharded_atm_latlon_step`) — deterministic AND stochastic
    (the per-global-column draw is decomposition-invariant).

    Parameters
    ----------
    model : CGridLatLonPrimitiveEquationModel
    mesh : jax.sharding.Mesh | None   1-D ``"lat"`` mesh (n_lat % n_dev == 0).
    hs_init : HydrostaticState        cell-centered, Field-wrapped.
    dt : float
    n_steps : int
    physics_fn : callable | None      column-local physics (stateless, or a
                                      DETERMINISTIC stateful scheme with its
                                      carry in ``phys_state``).
    phys_state : PhysicsState | None  deterministic prognostic carry.  Its
                                      ``(ncol, ...)`` leaves band-split on
                                      dim 0 (the C-order lat-major flatten
                                      makes a contiguous shard the band's
                                      own columns); stochastic schemes are
                                      refused (replicated-key draws are
                                      decomposition-variant).

    Returns
    -------
    HydrostaticState                       (``phys_state is None``), or
    (HydrostaticState, PhysicsState)       with the threaded carry-out.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    step = make_sharded_atm_latlon_step(model, mesh, physics_fn=physics_fn)
    c_state = shard_hydrostatic_to_atm_latlon(hs_init, model.grid, mesh)
    ps = phys_state
    for _ in range(n_steps):
        if ps is None:
            c_state = step(c_state, dt)
        else:
            c_state, ps = step(c_state, dt, phys_state=ps)
    hs_out = gather_atm_latlon_to_hydrostatic(c_state, model.grid, mesh)
    return hs_out if phys_state is None else (hs_out, ps)


def run_atm_latlon_spmd(model, mesh, hs_init, dt, n_steps, *,
                        segment_steps=None, physics_fn=None, on_segment=None,
                        compiled_segments=False):
    """Production lat-band-SPMD run driver: integrate ``n_steps`` of the C-grid
    hydrostatic atm in SEGMENTS, returning ``(hs_final, status)``.

    Each segment runs ``run_atm_latlon_spmd_segment`` (shard once -> sharded
    steps -> gather once), so a cell-centered ``HydrostaticState`` is available
    at every segment boundary for output / coupling (the ``on_segment(hs,
    step_done)`` callback) WITHOUT gathering every step. A NaN/Inf in the
    gathered surface pressure or temperature ends the run as
    ``"BLOWUP at step N"`` (the host-side equivalent of the compiled driver's
    finite-check), so a diverging run stops cleanly instead of integrating
    garbage. ``segment_steps=None`` uses one segment of ``n_steps``.

    DYNAMICS ONLY or STATELESS ``physics_fn`` (Held-Suarez / per-column
    parameterizations); a stateful ``PhysicsState`` carry is not yet SPMD-routed
    (see :func:`run_atm_latlon_spmd_segment`). ``mesh=None`` runs single-device.

    This is the run-loop the production driver dispatches to for a single-process
    multi-device lat-lon grid; it composes only the Stage-5/7-validated step +
    bridge, so it adds no new SPMD-correctness surface.

    The integration STAYS in the sharded C-grid layout for the WHOLE run
    (convert+shard ONCE on entry, convert+gather ONCE on exit). The per-segment
    gather is an OUTPUT-ONLY copy that is NOT fed back into the next segment — so
    the (lossy) cell<->face re-projection happens only at the run boundaries, NOT
    at every segment, and the trajectory is segmentation-invariant + matches the
    serial ``model.step`` loop (whose ``_cgrid_cache`` likewise stays C-grid).

    Parameters
    ----------
    model : CGridLatLonPrimitiveEquationModel
    mesh : jax.sharding.Mesh | None
    hs_init : HydrostaticState
    dt : float
    n_steps : int                     total steps to integrate.
    segment_steps : int | None        steps per gathered segment (output cadence).
    physics_fn : callable | None      stateless column-local physics.
    on_segment : callable | None      ``on_segment(hs_global, step_done)`` at each
                                      segment boundary (post-gather, for I/O); the
                                      gathered state is a COPY, not fed back.
    compiled_segments : bool          M2b opt-in (default ``False`` =
                                      historical per-step path, byte-identical).
                                      ``True``: each segment runs as ONE
                                      compiled ``lax.scan``
                                      (:func:`make_sharded_atm_latlon_segment`;
                                      band-SHARDED geometry) — one host
                                      dispatch per segment — and the blowup
                                      guard reads the segment's IN-GRAPH
                                      finite scalar (all state leaves; a
                                      superset of the historical gathered
                                      p_s/T check, so any old-path blowup is
                                      caught at the same boundary).  The
                                      full state is gathered ONLY for
                                      ``on_segment`` / the final return /
                                      a blowup report — a callback-free run
                                      never gathers mid-run.

    Returns
    -------
    (HydrostaticState, str)           final state + ``"COMPLETED"`` / ``"BLOWUP
                                      at step N"``.
    """
    import jax.numpy as jnp
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    seg = n_steps if segment_steps is None else int(segment_steps)
    if seg < 1:
        raise ValueError(f"segment_steps must be >= 1, got {segment_steps}")

    if compiled_segments:
        # One compiled scan per DISTINCT segment length (at most two: the
        # regular length and the final remainder), built once and reused.
        seg_fns: dict = {}

        def _seg_fn(n):
            fn = seg_fns.get(n)
            if fn is None:
                fn = make_sharded_atm_latlon_segment(
                    model, mesh, n, physics_fn=physics_fn)
                seg_fns[n] = fn
            return fn

        c_state = shard_hydrostatic_to_atm_latlon(hs_init, model.grid, mesh)
        done = 0
        while done < n_steps:
            this = min(seg, n_steps - done)
            c_state, ok = _seg_fn(this)(c_state, dt)
            done += this
            # ONE scalar host read per segment (the M2b finite check); the
            # full state is gathered only on blowup / for the callback.
            if not bool(ok):
                hs_out = gather_atm_latlon_to_hydrostatic(
                    c_state, model.grid, mesh)
                return hs_out, f"BLOWUP at step {done}"
            if on_segment is not None:
                on_segment(gather_atm_latlon_to_hydrostatic(
                    c_state, model.grid, mesh), done)
        return (gather_atm_latlon_to_hydrostatic(c_state, model.grid, mesh),
                "COMPLETED")

    step = make_sharded_atm_latlon_step(model, mesh, physics_fn=physics_fn)
    # Convert + shard ONCE; the run stays in the sharded C-grid layout.
    c_state = shard_hydrostatic_to_atm_latlon(hs_init, model.grid, mesh)
    done = 0
    status = "COMPLETED"
    while done < n_steps:
        this = min(seg, n_steps - done)
        for _ in range(this):
            c_state = step(c_state, dt)
        done += this
        # Gather a cell-centered COPY for output / blowup-check ONLY — the
        # integration continues from c_state (sharded C-grid), so the lossy
        # cell<->face round-trip is NOT fed back into the dynamics.
        hs_out = gather_atm_latlon_to_hydrostatic(c_state, model.grid, mesh)
        finite = bool(jnp.isfinite(hs_out.p_s.data).all()
                      & jnp.isfinite(hs_out.T.data).all())
        if not finite:
            return hs_out, f"BLOWUP at step {done}"
        if on_segment is not None:
            on_segment(hs_out, done)
    return gather_atm_latlon_to_hydrostatic(c_state, model.grid, mesh), status


# ==============================================================================
# M3a — native 2-D ("lat", "lon") tiling for the atmosphere SPMD step
# ==============================================================================
# The 1-D lat-band decomposition's halo perimeter is the CONSTANT n_lon per
# cut (independent of the device count) — the term that caps band scaling.
# The 2-D tiling shards latitude AND longitude: lat stays the pole-terminated
# line (ppermute at cuts, the serial 180-deg fold at the pole tiles), lon
# becomes a periodic ring (cyclic ppermute — the wrap IS the roll
# permutation).  Staggered ownership mirrors the 1-D v convention:
#
#   * v (n_lat+1 rows)   -> v_lower = v[:n_lat]; each tile's north boundary
#     face is the lat-neighbour's v_lower[0] (reconstruct_vface_lower).
#   * u (n_lon+1 columns) -> u_left = u[:, :n_lon]; each tile's east seam
#     face is the lon-neighbour's u_left[:, 0] (reconstruct_uface_left) —
#     the +1 seam column is OWNED by the tile whose slice starts there and
#     reconstructed on the periodic wrap (u[:, n_lon] == u[:, 0] identity).
#
# The step body is the SAME un-jitted ``model._step_cgrid_impl`` the band
# path runs: all lon-direction neighbour access inside the operators already
# routes through the backend-dispatched ``pad_lon_cgrid`` / ``pad_halo_latlon*``
# (which this lane arms with the 2-D mesh), so no operator numerics are
# duplicated here.  Corner (diagonal) dependencies compose through the
# sequential lat-then-lon exchanges inside ``make_latlon_2d_pad_body``; the
# C-grid chain has no explicit-diagonal stencil (vertex circulations combine
# lat-padded u with lon-padded v).
#
# The 1-D band lane above is UNTOUCHED and remains the default production
# path (choose_latlon_2d_topology returns (N, 1) whenever the band is
# FEASIBLE — the 2-D pad's pole-fold lon-all_gathers outweigh its perimeter
# advantage until the partner-ppermute fold lands; the 2-D lane is for the
# beyond-band regime n_devices > n_lat/min_tile or indivisible n_lat.  The
# (N, 1) 2-D mesh degenerates bit-identically anyway).


def tile_spec(arr) -> P:
    """``P("lat", "lon", None, ...)`` for an array tiled on its two leading
    (lat, lon) axes — the 2-D twin of :func:`lat_spec`."""
    return P("lat", "lon", *((None,) * (arr.ndim - 2)))


def shard_state_atm_latlon_2d(
    state: CGridLatLonHydrostaticState, mesh,
) -> CGridLatLonHydrostaticState:
    """Lay out a C-grid hydrostatic atm state for the 2-D ("lat", "lon")
    tile shard_map.

    Cell leaves (``T, p_s, phis`` + every tracer) tile ``P("lat", "lon",
    ...)`` directly.  The staggered ``v`` (leading dim ``n_lat+1``) drops its
    north pole-wall face -> ``v_lower = v[:n_lat]`` exactly as the 1-D layout
    (:func:`shard_state_atm_latlon`).  The staggered ``u`` (lon dim
    ``n_lon+1``) drops its east periodic-seam column -> ``u_left =
    u[:, :n_lon]``; the dropped column is the periodic closure
    (``u[:, n_lon] == u[:, 0]`` — the identity ``interp_cell_to_uface``
    constructs and every C-grid tendency preserves) and is reconstructed
    inside the body from the east neighbour via the lon ring.
    """
    def _put(arr):
        return jax.device_put(arr, NamedSharding(mesh, tile_spec(arr)))

    n_lat, n_lon = state.T.shape[0], state.T.shape[1]
    return state._replace(
        u=_put(state.u[:, :n_lon]),
        v=_put(state.v[:n_lat]),
        T=_put(state.T),
        p_s=_put(state.p_s),
        phis=_put(state.phis),
        tracers={k: _put(val) for k, val in state.tracers.items()},
    )


def gather_state_atm_latlon_2d(
    state: CGridLatLonHydrostaticState, mesh,
) -> CGridLatLonHydrostaticState:
    """Inverse of :func:`shard_state_atm_latlon_2d`: replicate every leaf,
    re-append the zero north pole-wall v face and the periodic u seam column
    (``u[:, n_lon] = u[:, 0]``).  Bit-comparable to the single-device state,
    whose top v-face is the pole wall (== 0) and whose last u column is the
    periodic closure (== column 0)."""
    from legoesm.parallel.latlon_spmd import replicate_leaf

    rep = NamedSharding(mesh, P())
    _mp = jax.process_count() > 1

    def _get(arr):
        return replicate_leaf(arr, rep, multiprocess=_mp)

    v_lower = _get(state.v)
    v_full = jnp.concatenate([v_lower, jnp.zeros_like(v_lower[:1])], axis=0)
    u_left = _get(state.u)
    u_full = jnp.concatenate([u_left, u_left[:, 0:1]], axis=1)
    return state._replace(
        u=u_full,
        v=v_full,
        T=_get(state.T),
        p_s=_get(state.p_s),
        phis=_get(state.phis),
        tracers={k: _get(val) for k, val in state.tracers.items()},
    )


def build_tile_grids_atm_2d(grid, p_lat: int, p_lon: int):
    """Build the ``p_lat x p_lon`` UNIFORM tile ``LatLonGrid`` geometries via
    the tested 2-D MPI slicer (no bespoke metric re-derivation) — the 2-D
    twin of :func:`build_band_grids_atm`.

    Returns a ``[p_lat][p_lon]`` nested list (row r = lat band, col c = lon
    sector).  ``skip_total_area_reduce=True`` keeps ``total_area`` the GLOBAL
    full-sphere sum on EVERY tile (the mass-fixer denominator stays global).
    Requires ``n_lat % p_lat == 0`` and ``n_lon % p_lon == 0`` (uniform tiles
    -> one shard_map program) and at least 2 cells per SPLIT dimension per
    tile (the widest production halo — the PPM ``halo=2`` exchange — moves
    edge blocks of that depth in one ppermute hop).
    """
    from legoesm.parallel.latlon_mpi import (
        make_latlon_2d_layout, slice_latlon_grid_to_block_2d)
    n_lat, n_lon = int(grid.n_lat), int(grid.n_lon)
    if p_lat < 1 or p_lon < 1:
        raise ValueError(
            f"p_lat/p_lon must be >= 1, got ({p_lat}, {p_lon})")
    if n_lat % p_lat != 0 or n_lon % p_lon != 0:
        raise ValueError(
            f"atm 2-D SPMD tiling requires n_lat ({n_lat}) % p_lat "
            f"({p_lat}) == 0 and n_lon ({n_lon}) % p_lon ({p_lon}) == 0 so "
            f"every tile is uniform (one shard_map program).")
    nl, w = n_lat // p_lat, n_lon // p_lon
    if (p_lat > 1 and nl < 2) or (p_lon > 1 and w < 2):
        raise ValueError(
            f"atm 2-D SPMD tiling: tiles must keep >= 2 cells per split "
            f"dimension (PPM halo=2 single-hop exchange); got "
            f"{nl}x{w} tiles from ({p_lat}, {p_lon}) on {n_lat}x{n_lon}.")
    fold = getattr(grid, "fold", None)
    return [
        [
            slice_latlon_grid_to_block_2d(
                grid,
                make_latlon_2d_layout(
                    r * p_lon + c, p_lat, p_lon, n_lat, n_lon, fold),
                skip_total_area_reduce=True)
            for c in range(p_lon)
        ]
        for r in range(p_lat)
    ]


def _build_geometry_stacks_2d(model, mesh, p_lat: int, p_lon: int,
                              shard_geometry: bool):
    """Stack every tile's ``LatLonGrid`` array fields (+ the optional
    polar-filter masks at ``p_lon == 1``) over LEADING ``(p_lat, p_lon)``
    tile axes and lay them out on ``mesh`` — the 2-D twin of
    :func:`_build_geometry_stacks`.

    ``shard_geometry=True``: stacks are sharded ``P("lat", "lon", ...)`` on
    the tile axes — each device holds ONLY its own tile's slice (leading
    extents ``(1, 1)`` inside the body, static index ``[0, 0]``).
    ``shard_geometry=False``: replicated (``P()``) stacks, indexed at
    ``(axis_index("lat"), axis_index("lon"))``.  Same tile VALUES either way.

    Returns ``(template, array_field_names, stacks, stacks_spec)``.
    """
    grid = model.grid
    tile_grids = build_tile_grids_atm_2d(grid, p_lat, p_lon)
    template = tile_grids[0][0]
    array_field_names = atm_grid_array_field_names(template)
    raw = {
        name: jnp.stack([
            jnp.stack([jnp.asarray(getattr(tile_grids[r][c], name))
                       for c in range(p_lon)], axis=0)
            for r in range(p_lat)
        ], axis=0)
        for name in array_field_names
    }
    # Per-tile polar-filter masks: p_lon > 1 is refused by the factories
    # (the filter rfft's the full lon circle); at p_lon == 1 the stacks
    # mirror the band layout with a singleton lon-tile axis.
    nl = int(grid.n_lat) // p_lat
    if model._polar_mask is not None:
        if p_lon > 1:
            raise NotImplementedError(
                "atm 2-D SPMD tiling: use_polar_filter=True with p_lon > 1 "
                "is not wired — the polar filter FFTs the full longitude "
                "circle (needs a lon-gather FFT).  Use p_lon == 1 or "
                "disable the filter.")
        raw["__polar_mask"] = jnp.stack(
            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(p_lat)],
            axis=0)[:, None]
        raw["__polar_mask_v"] = jnp.stack(
            [model._polar_mask_v[r * nl:r * nl + nl + 1]
             for r in range(p_lat)],
            axis=0)[:, None]
    # #1362, 2-D twin of the guard in _build_geometry_stacks -- same
    # per-process recompute, same replicated-device_put bit-identity assert.
    # n_dev is the FULL device count here (p_lat * p_lon): the schema
    # fingerprint must describe this process's whole mesh, not one axis.
    ordered_names = list(raw)
    assert_schema_agrees(ordered_names, p_lat * p_lon,
                         context="make_sharded_atm_latlon_step_2d")
    raw = {
        name: jnp.asarray(broadcast_checked(
            raw[name], name, context="make_sharded_atm_latlon_step_2d"))
        for name in ordered_names
    }
    spec_of = tile_spec if shard_geometry else (lambda _arr: P())
    stacks = {
        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
        for name, arr in raw.items()
    }
    stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}
    return template, array_field_names, stacks, stacks_spec


def _make_tile_step_body_2d(model, template, array_field_names,
                            perm_north, p_lon: int, physics_fn,
                            shard_geometry: bool):
    """One tile's un-jitted C-grid step body — the 2-D twin of
    :func:`_make_band_step_body`, shared by the per-step and segment 2-D
    factories so the tile numerics are written ONCE.

    Returns ``tile_step(state_local, stacks_local, dt, ps_local) ->
    (state_out_local, ps_out)`` operating on the tile-local
    ``(u_left, v_lower)`` layout: reconstruct the tile's ``nl+1`` v-faces
    (lat ppermute) AND ``w+1`` u-faces (lon ring ppermute), run the un-jitted
    band/tile step on the tile geometry, convert both staggers back.
    """
    from legoesm.parallel.latlon_spmd import (
        reconstruct_uface_left, reconstruct_vface_lower,
        spmd_pole_end_masks, to_uface_left, to_vface_lower)

    def tile_step(state_local, stacks_local, dt, ps_local):
        if shard_geometry:
            gi, gj = 0, 0
        else:
            gi = jax.lax.axis_index("lat")
            gj = jax.lax.axis_index("lon")
        tile_geom = template._replace(
            **{name: stacks_local[name][gi, gj]
               for name in array_field_names})
        pmask = (stacks_local["__polar_mask"][gi, gj]
                 if "__polar_mask" in stacks_local else None)
        pmaskv = (stacks_local["__polar_mask_v"][gi, gj]
                  if "__polar_mask_v" in stacks_local else None)
        # Reconstruct the tile's nl+1 v-faces (shared interface row via the
        # lat ppermute) and w+1 u-faces (periodic seam column via the lon
        # ring), run the un-jitted step, convert both staggers back.
        v_full = reconstruct_vface_lower(state_local.v, "lat", perm_north)
        u_full = reconstruct_uface_left(state_local.u, "lon", p_lon)
        state_tile = state_local._replace(u=u_full, v=v_full)
        out, ps_out = model._step_cgrid_impl(
            state_tile, dt,
            physics_fn=physics_fn, phys_state=ps_local,
            grid=tile_geom, sigma_coord=model.sigma_coord,
            polar_mask=pmask, polar_mask_v=pmaskv,
            pole_v_bc_masks=spmd_pole_end_masks(),
        )
        return (out._replace(u=to_uface_left(out.u),
                             v=to_vface_lower(out.v)), ps_out)

    return tile_step


def _check_2d_mesh(mesh) -> tuple[int, int]:
    """Validate the 2-D tile mesh axes and return ``(p_lat, p_lon)``."""
    names = tuple(mesh.axis_names)
    if names != ("lat", "lon"):
        raise ValueError(
            f"atm 2-D SPMD tiling: mesh axes must be ('lat', 'lon'); got "
            f"{names}.  Build it as Mesh(devices.reshape(p_lat, p_lon), "
            f"axis_names=('lat', 'lon')) — choose_latlon_2d_topology picks "
            f"(p_lat, p_lon).")
    return int(mesh.shape["lat"]), int(mesh.shape["lon"])


def _refuse_unsupported_spmd_config_2d(model, p_lon: int) -> None:
    """2-D-specific dispatch-hardening on top of the shared band refusals."""
    _refuse_unsupported_spmd_config(model)
    if p_lon > 1 and bool(getattr(model.config, "use_polar_filter", False)):
        raise NotImplementedError(
            "atm 2-D SPMD tiling: use_polar_filter=True with p_lon > 1 is "
            "not wired — the polar filter FFTs the full longitude circle "
            "and needs a lon-gather FFT.  (The route-A MPI path "
            "make_latlon_2d_mpi_step DOES wire this via the AD-safe "
            "lat-pencil transpose; the SPMD ppermute equivalent is a "
            "follow-up.)  Use p_lon == 1 or disable the filter.")


def make_sharded_atm_latlon_step_2d(model, mesh, physics_fn=None, *,
                                    shard_geometry: bool = True):
    """Return ``step(c_state, dt) -> c_state`` running the C-grid hydrostatic
    atm step 2-D-tile-SPMD over a ``("lat", "lon")`` mesh — the M3a native
    2-D tiling twin of :func:`make_sharded_atm_latlon_step`.

    ``c_state`` is a ``CGridLatLonHydrostaticState`` laid out with
    :func:`shard_state_atm_latlon_2d` (``v`` as ``v_lower``, ``u`` as
    ``u_left``).  The body reconstructs each tile's staggered faces (v via
    the lat ppermute, u via the periodic lon ring), runs the UN-jitted
    ``model._step_cgrid_impl`` on the tile geometry + per-tile pole masks,
    and converts both staggers back.  Halos: the armed 2-D SPMD backend
    routes ``pad_halo_latlon*`` through ``make_latlon_2d_pad_body`` (lat
    ppermute + lon ring + the EXACT serial 180-deg pole fold via a lon-ring
    all_gather at the pole tiles), ``pad_with_pole_bc_lat`` through the
    lat-only wall body, and ``pad_lon_cgrid`` through the lon ring — all
    shared machinery, no operator numerics duplicated.  Global reductions
    (the mass fixer's ``batch_global_area_sums``) psum over BOTH mesh axes.

    A degenerate ``(N, 1)`` mesh is bit-identical to the 1-D band step
    (every lon-ring op takes its static local branch; gated by
    ``tests/parallel/test_atm_latlon_2d_tiling.py``).  The 1-D band factory
    remains the default production lane.

    ``physics_fn``: STATELESS column-local closures only (Held-Suarez etc.),
    evaluated per RK stage on the TILE geometry — decomposition-invariant
    with no collectives.  A stateful ``PhysicsState`` carry is REFUSED: its
    ``(ncol, ...)`` leaves flatten lat-major over the GLOBAL grid, so a
    contiguous dim-0 shard is a lat BAND's columns, not a 2-D tile's —
    thread carries through the 1-D :func:`make_sharded_atm_latlon_step`.

    ``shard_geometry=True`` (default — new API, no historical layout):
    per-device tile geometry slices (``P("lat", "lon")`` stacks);
    ``False`` replicates the all-tile stacks (indexed at the axis indices).
    Same tile values either way (bit-identical numerics).
    """
    from legoesm.parallel.latlon_spmd import latlon_band_perms
    from legoesm.parallel.shard_map_compat import shard_map
    from legoesm.timestepping.integration import (
        refuse_unthreaded_stateful_physics)

    if mesh is None:                       # single-device: plain C-grid step
        return make_sharded_atm_latlon_step(model, None,
                                            physics_fn=physics_fn)

    p_lat, p_lon = _check_2d_mesh(mesh)
    _refuse_unsupported_spmd_config_2d(model, p_lon)

    template, array_field_names, stacks, stacks_spec = (
        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
    perm_north, _perm_south = latlon_band_perms(p_lat)
    tile_step = _make_tile_step_body_2d(
        model, template, array_field_names, perm_north, p_lon, physics_fn,
        shard_geometry)

    def _body(state_local, stacks_local, dt):
        out, _ = tile_step(state_local, stacks_local, dt, None)
        return out

    _cache = {}

    def sharded_step(c_state, dt, phys_state=None):
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="atm lat-lon 2-D SPMD step")
        if phys_state is not None:
            raise NotImplementedError(
                "make_sharded_atm_latlon_step_2d: a stateful PhysicsState "
                "carry is not 2-D-tile-routed (its (ncol, ...) leaves "
                "flatten lat-major over the GLOBAL grid — a contiguous "
                "dim-0 shard is a lat band, not a 2-D tile).  Thread the "
                "carry through the 1-D make_sharded_atm_latlon_step.")
        key = jax.tree.structure(c_state)
        fn = _cache.get(key)
        if fn is None:
            in_spec = jax.tree.map(tile_spec, c_state)
            fn = jax.jit(shard_map(
                _body, mesh=mesh, in_specs=(in_spec, stacks_spec, P()),
                out_specs=in_spec, check_vma=False))
            _cache[key] = fn
        with _latlon_spmd_armed(mesh):
            return fn(c_state, stacks, jnp.asarray(dt))

    sharded_step._geom_stacks = stacks   # test/introspection only
    return sharded_step


def make_sharded_atm_latlon_segment_2d(model, mesh, n_steps: int,
                                       physics_fn=None, *,
                                       shard_geometry: bool = True):
    """Return ``segment(c_state, dt) -> (c_state, all_finite)`` advancing
    ``n_steps`` C-grid steps in ONE compiled ``lax.scan`` over the 2-D
    ``("lat", "lon")`` tile mesh — the M3a twin of
    :func:`make_sharded_atm_latlon_segment` (same contract: static
    ``n_steps``, replicated in-graph finite scalar psum'd over BOTH mesh
    axes, leading steps unrolled to the scan-carry dtype fixed point via
    :func:`unroll_to_dtype_fixed_point`, STATELESS physics only,
    ``mesh=None`` -> the single-device compiled twin)."""
    from legoesm.parallel.latlon_spmd import latlon_band_perms
    from legoesm.parallel.shard_map_compat import shard_map
    from legoesm.timestepping.integration import (
        refuse_unthreaded_stateful_physics)

    if int(n_steps) < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    n_steps = int(n_steps)

    if mesh is None:                       # single-device compiled segment
        return make_sharded_atm_latlon_segment(model, None, n_steps,
                                               physics_fn=physics_fn)

    p_lat, p_lon = _check_2d_mesh(mesh)
    _refuse_unsupported_spmd_config_2d(model, p_lon)

    def _refuse_carry(phys_state):
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="atm lat-lon 2-D compiled segment")
        if phys_state is not None:
            raise NotImplementedError(
                "make_sharded_atm_latlon_segment_2d: a stateful PhysicsState "
                "carry is not 2-D-tile-routed — use the 1-D per-step "
                "make_sharded_atm_latlon_step(phys_state=...) path.")

    template, array_field_names, stacks, stacks_spec = (
        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
    perm_north, _perm_south = latlon_band_perms(p_lat)
    tile_step = _make_tile_step_body_2d(
        model, template, array_field_names, perm_north, p_lon, physics_fn,
        shard_geometry)

    def _seg_body(state_local, stacks_local, dt):
        def _step1(s):
            out, _ps = tile_step(s, stacks_local, dt, None)
            return out
        # Unroll to the scan-carry dtype fixed point (helper docstring).
        # Public name (the _-prefixed original was promoted; the 2-D path
        # kept the stale private reference — NameError on first segment
        # trace, caught by test_2d_segment_matches_sequential_and_serial).
        out, n_left = unroll_to_dtype_fixed_point(
            _step1, state_local, n_steps)
        if n_left > 0:
            out, _ = jax.lax.scan(lambda s, _x: (_step1(s), None),
                                  out, xs=None, length=n_left)
        return out, state_finite_scalar(out, axis=("lat", "lon"))

    _cache = {}

    def segment(c_state, dt, phys_state=None):
        _refuse_carry(phys_state)
        key = jax.tree.structure(c_state)
        fn = _cache.get(key)
        if fn is None:
            in_spec = jax.tree.map(tile_spec, c_state)
            fn = jax.jit(shard_map(
                _seg_body, mesh=mesh,
                in_specs=(in_spec, stacks_spec, P()),
                out_specs=(in_spec, P()), check_vma=False))
            _cache[key] = fn
        with _latlon_spmd_armed(mesh):
            return fn(c_state, stacks, jnp.asarray(dt))

    segment._geom_stacks = stacks   # test/introspection only
    return segment
