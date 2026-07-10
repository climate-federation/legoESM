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

from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
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
    def _put(arr):
        return jax.device_put(arr, NamedSharding(mesh, lat_spec(arr)))

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
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
        hydrostatic_to_cgrid)
    c_state = hydrostatic_to_cgrid(hs, grid)
    return c_state if mesh is None else shard_state_atm_latlon(c_state, mesh)


def gather_atm_latlon_to_hydrostatic(c_state, grid, mesh):
    """Inverse of :func:`shard_hydrostatic_to_atm_latlon`: gather the sharded
    C-grid state THEN convert (C-grid faces -> cell winds) to the cell-centered,
    Field-wrapped ``HydrostaticState`` the driver/output/restart contract
    expects. ``mesh=None`` converts the already-full C-grid state directly."""
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
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


def state_finite_scalar(state, axis: str | None = None):
    """Traced SCALAR bool: True iff EVERY array leaf of ``state`` is finite.

    The M2b in-graph blowup guard: inside a ``shard_map`` pass ``axis`` (the
    mesh axis name) so the per-band non-finite presence is ``psum``-reduced —
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

    Carry dtype: the init carry is cast to the STORAGE dtype once at segment
    entry.  ``_step_cgrid_impl`` returns storage-dtype arrays, so a
    lower-precision IC (e.g. an f32 grid-derived state under an f64 storage
    policy) would otherwise trip ``lax.scan``'s carry-type check on the very
    promotion the per-step Python loop absorbs silently after its first
    step.  The cast is UPCAST-ONLY and value-exact (f32 embeds exactly in
    f64), so the segment trajectory stays bit-identical to the per-step
    path; no precision is ever dropped.
    """
    from legoesm.core.precision import cast_pytree
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
            # Storage-dtype carry (upcast-only, value-exact): the scan carry
            # must match the step's storage-dtype output — see the docstring.
            c_state = cast_pytree(c_state, None, "storage")

            def _one(s, _):
                out, _ps = model._step_cgrid_impl(
                    s, dt, physics_fn=physics_fn, phys_state=None)
                return out, None
            out, _ = jax.lax.scan(_one, c_state, xs=None, length=n_steps)
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
        # Storage-dtype carry (upcast-only, value-exact): the scan carry must
        # match the band step's storage-dtype output — see the docstring.
        state_local = cast_pytree(state_local, None, "storage")

        def _one(s, _):
            out, _ps = band_step(s, stacks_local, dt, None)
            return out, None
        out, _ = jax.lax.scan(_one, state_local, xs=None, length=n_steps)
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
