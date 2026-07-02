r"""Multi-GPU SPMD step for the lat-lon C-grid ocean (latlon / tripole).

Wraps ``LatLonCGridOceanModel.step`` in ``jax.shard_map`` over a 1-D ``"lat"``
device mesh, so the ocean state is partitioned by latitude band across devices
(eORCA025 ¼° needs ~141 GiB — fits 32 GiB GPUs at N>=5 bands). The dynamics
operators are element-wise / local-stencil and the in-step halo exchange routes
through the SPMD band body once ``activate_latlon_spmd_halo(mesh)`` is armed; the
barotropic PCG reductions already dispatch to ``batch_psum_spmd`` on ``"lat"``
(``barotropic_common._global_dot_batch``). This is the ocean analogue of the
cubed-sphere ``parallel.sharded_dynamics.make_sharded_step``.

Architecture (the two non-trivial pieces — see ``omip-multinode-spmd-scope``):

* GRID = **replicated-stacked, indexed** (Structure A). The ``N`` band
  geometries are built host-side (``build_band_grids``), their ARRAY fields are
  ``jnp.stack``-ed into a replicated pytree, and the in-``shard_map`` body picks
  its own band by ``jax.lax.axis_index("lat")``. The grid is 2-D/small so
  replication is cheap, and this AVOIDS staggered-sharding the grid (the v-row
  is ``n_lat+1``, coprime with ``n_lat`` for ``N>1``). The
  ``LatLonCGridGeometry`` SCALAR fields (``n_lat``, ``n_lon``, ``radius``,
  ``dlon``, ``dlat``, ``fold``) stay STATIC python values — they gate trace-time
  ``if``\ s and ``jnp.zeros((n_lat, ...))`` shape builds, so a traced ``n_lat``
  would break the step. Only the ``jax.Array`` fields are stacked/indexed; each
  band geometry is rebuilt as ``template._replace(**indexed_arrays)`` so the
  scalars come from a band template (all bands share ``n_lat_local = n_lat/N``).

* STATE = cell fields shard ``P("lat")``; the STAGGERED ``v`` / ``v_mask``
  (shape ``(n_lat+1, n_lon, ...)``) use the **band-local v-faces** layout
  (Structure B). The sharded state carries ``v_lower = v[0:n_lat]`` (``n_lat``
  rows, ``P("lat")`` — the top pole row ``v[n_lat]`` is a 0 wall on the regular
  grid and is dropped). In-body, band ``r`` reconstructs its ``nl+1`` v-faces by
  ``ppermute``-ing band ``r+1``'s first ``v_lower`` row (= global ``v[e]``) up as
  its north boundary row; the north band's non-target receives the pole-wall 0.
  After ``model.step`` the result's ``nl+1`` v is converted back to the ``nl``-row
  ``v_lower`` representation (drop the boundary row, owned by band ``r+1``).

Correctness is gated by ``tests/parallel/test_latlon_ocean_spmd_step.py`` (N-step
tol-match vs the single-device step). Pure-dynamics (no host-loop coupling): the
OMIP host post-step updates (ice / SSS-restore / geothermal) are NOT inside this
wrapper — they need separate gather/scatter or jax-porting (see
[[omip-multinode-spmd-scope]]).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

try:                                   # JAX >= 0.8 exposes shard_map at top level
    from jax import shard_map
except ImportError:                    # pragma: no cover - JAX < 0.8 fallback
    from jax.experimental.shard_map import shard_map

from legoesm.parallel.latlon_spmd import (
    activate_latlon_spmd_halo,
    latlon_band_perms,
)

# Per-device band grids: the SPMD wrapper REUSES the tested band slicers from
# ``legoesm.parallel.latlon_mpi`` rather than a bespoke slice — they already
# handle the staggered v-row (n_lat+1), the bipolar-fold localization (the
# northernmost band keeps the active ``fold``; interior bands get
# ``fold_j=cap_j=-1`` so the pad_ns_* operators fall through to the band halo
# instead of folding their own last row), AND keep ``total_area`` GLOBAL (the
# area-weighted-mean denominator must not become band-local).  Build a band
# layout with ``make_latlon_band_layout(rank, n_dev, n_lat, n_lon, fold)`` then
# call ``slice_cgrid_geometry_to_band(geom, layout)`` (curvilinear tripole /
# eORCA025) or ``slice_latlon_grid_to_band(grid, layout)`` (regular lat-lon ½°).
# latlon_mpi defers its ``from mpi4py import MPI`` to function scope, so these
# pure slicers import without requiring mpi4py.  See omip-multinode-spmd-scope.

# The LatLonCGridGeometry fields that MUST stay static python scalars (they gate
# trace-time ``if``-s + ``jnp.zeros`` shape builds): never stacked/indexed.  The
# ``fold`` is a FoldDescriptor whose ``is_active``/``fold_j``/``cap_j`` are python
# ints/bools used at trace time; on a regular grid the fold is inactive and
# identical on every band, so taking it from the band template is exact.  Every
# OTHER LatLonCGridGeometry field is a ``jax.Array`` and IS stacked/indexed.
_GEOM_STATIC_FIELDS = frozenset(
    {"n_lat", "n_lon", "radius", "dlon", "dlat", "fold"}
)

# State fields that are staggered on the v-grid (leading dim n_lat+1): carried in
# the sharded state as ``v_lower = field[0:n_lat]`` (P("lat")) and reconstructed
# to the band's nl+1 faces in-body.  All OTHER array state fields are cell/u
# leading-dim n_lat and shard P("lat") directly.
_V_STAGGERED_STATE_FIELDS = ("v", "v_mask")


def _geom_array_field_names(geom):
    """The ``jax.Array`` field names of a ``LatLonCGridGeometry`` (everything
    except the static scalars + the ``fold`` descriptor).  Order-stable
    (the NamedTuple field order) so the host stack and the in-body index agree."""
    names = []
    for name in geom._fields:
        if name in _GEOM_STATIC_FIELDS:
            continue
        val = getattr(geom, name)
        if isinstance(val, (jax.Array, np.ndarray)):
            names.append(name)
    return names


def _lat_spec(x):
    """Lat-band PartitionSpec for an array leaf: shard axis 0, replicate rest."""
    nd = int(getattr(x, "ndim", np.ndim(x)))
    if nd < 1:
        return P()                     # scalar -> replicated
    return P("lat", *((None,) * (nd - 1)))


def _step_body(model, state, dt, *, grid, vertex_mask,
               surface_forcing=None, freshwater=None):
    """Run ONE ocean step via the model's NON-jitted body (``_step_impl`` /
    ``_ab2_step`` + the static-gated post-steps), the un-jitted twin of
    ``LatLonCGridOceanModel._step_jitted``.

    The shard_map body must call THIS, not ``model.step`` — ``model.step`` goes
    through ``_step_jitted`` (``@jax.jit``), a NESTED jit that takes ``grid`` as a
    TRACED argument, so the band geometry's STATIC scalars (``fold.is_active``,
    ``fold.fold_j``, ``n_lat`` …) become tracers and the operators' trace-time
    ``if fold.is_active`` / ``jnp.zeros((n_lat, …))`` blow up
    (``TracerBoolConversionError``).  shard_map already provides the JIT boundary,
    so running the un-jitted body keeps ``grid`` a concrete Python-scalar-carrying
    pytree inside the single trace — the model docstring's own "call ``_step_impl``
    directly inside an outer JIT context" guidance.

    ``surface_forcing`` / ``freshwater`` are the BAND-LOCAL (already
    ``P("lat")``-sliced) external-forcing pytrees (``OceanSurfaceForcing`` /
    ``FreshwaterForcing``); they are forwarded verbatim into ``_step_impl`` /
    ``_ab2_step`` (which apply the wind-stress / heat / virtual-salt / eta terms
    in-core).  ``None`` ⇒ the un-forced step (byte-identical to the original
    no-forcing body).

    Mirrors ``_step_jitted``'s dispatch verbatim (outer_integrator branch +
    polar-filter / freeze-floor / ew-cyclic static gates) so a config that
    enables any post-step op runs it identically under SPMD.  ``_step_jitted``
    stays the production single-device shim; this is its body without the wrapper.
    """
    _oi = getattr(model.config, "outer_integrator", "forward_euler")
    if _oi not in ("forward_euler", "ab2"):
        raise ValueError(
            "config.outer_integrator must be 'forward_euler' or 'ab2', "
            f"got {_oi!r}")
    if _oi == "ab2":
        if model.config.tracer_time_integrator == "ab2":
            raise ValueError(
                "outer_integrator='ab2' double-counts with "
                "tracer_time_integrator='ab2'; set the inner one to 'euler'.")
        new_state = model._ab2_step(
            state, dt, freshwater=freshwater, surface_forcing=surface_forcing,
            grid=grid, vertex_mask=vertex_mask)
    else:
        new_state = model._step_impl(
            state, dt, freshwater=freshwater, surface_forcing=surface_forcing,
            grid=grid, vertex_mask=vertex_mask)
    if model.config.polar_filter.use_polar_filter:
        new_state = model._apply_polar_filter(new_state, dt, grid=grid)
    if model.config.freeze_floor:
        new_state = model._apply_freeze_floor(new_state)
    if getattr(model.config, "ew_cyclic_overlap", False):
        new_state = model._apply_ew_cyclic_overlap(new_state)
    return new_state


def build_band_grids(grid, n_devices: int):
    """Build the ``n_devices`` UNIFORM lat-band geometries for the SPMD step,
    reusing the tested MPI band slicers (no bespoke re-derivation).

    Requires ``grid.n_lat % n_devices == 0`` so every band has the same leading
    shape — a ``shard_map`` body is ONE program, so the per-band grids must be
    structurally identical (only the values + the north band's active fold
    differ). ``make_latlon_band_layout`` would otherwise hand the first
    ``n_lat % n_devices`` ranks one extra row.

    Returns a list of ``n_devices`` band-local ``LatLonCGridGeometry`` (rank 0 =
    south band ... rank ``n_devices-1`` = north band). The slicer keeps
    ``total_area`` GLOBAL on every band (area-weighted-mean denominator), slices
    v/q rows ``[s:e+1]`` (the shared staggered boundary face), and localizes the
    bipolar fold — active only on the north band, ``fold_j=cap_j=-1`` sentinel on
    interior bands so ``fold_is_local()`` is False there and the pad_ns_*
    operators fall through to the SPMD band halo. See omip-multinode-spmd-scope.

    ``latlon_mpi`` defers ``from mpi4py import MPI`` to function scope, so these
    pure slicers import without requiring mpi4py.
    """
    from legoesm.parallel.latlon_mpi import (
        make_latlon_band_layout,
        slice_cgrid_geometry_to_band,
    )
    n_lat = int(grid.n_lat)
    if n_devices < 1:
        raise ValueError(f"n_devices must be >= 1, got {n_devices}")
    if n_lat % n_devices != 0:
        raise ValueError(
            f"SPMD lat-band requires n_lat ({n_lat}) divisible by n_devices "
            f"({n_devices}) so every band is uniform (one shard_map program). "
            f"Pick n_devices among the divisors of {n_lat}.")
    fold = getattr(grid, "fold", None)
    return [
        slice_cgrid_geometry_to_band(
            grid,
            make_latlon_band_layout(r, n_devices, n_lat, int(grid.n_lon), fold))
        for r in range(n_devices)
    ]


def _build_band_vertex_masks(model, n_dev):
    """Per-band vertex masks (n_lat/N+1, n_lon+1): SLICE the model's primed GLOBAL
    vertex mask ``[s : e+1]`` per band (the v/q-row stagger ``slice_cgrid_geometry_
    to_band`` uses for ``area_q``).

    The slice — NOT a per-band ``compute_vertex_mask`` — is what makes the
    band-cut rows correct.  The global ``model._vertex_mask`` already encoded the
    four-cell product at EVERY vertex row including the interior cut rows (it saw
    the neighbour-band cells), so band ``r``'s vertex rows ``[r*nl : r*nl+nl+1]``
    are exact.  A per-band ``compute_vertex_mask`` would instead pad the band's
    cell mask with a POLE WALL at the cut (``pad_with_pole_bc_lat`` has no SPMD
    branch — it falls to the local constant pad), delivering a wrong (walled)
    north/south boundary vertex row at interior cuts — the "one row off at the
    cut" failure the ``compute_vertex_mask`` docstring warns about.

    Requires the model's vertex mask to be primed (``model._ensure_vertex_mask``
    on the concrete initial state; ``step`` does this on the first call).
    """
    vmask = model._vertex_mask
    if vmask is None:
        raise RuntimeError(
            "make_sharded_ocean_step: the model vertex-mask cache is not "
            "primed.  Call model._ensure_vertex_mask(state) (or model.step) "
            "on the concrete initial state before building the sharded step.")
    vmask = np.asarray(vmask)                   # (n_lat+1, n_lon+1)
    n_lat = vmask.shape[0] - 1
    nl = n_lat // n_dev
    # v/q-row stagger: band r owns global vertex rows [r*nl : r*nl+nl+1] (the
    # shared interior boundary vertex row appears in band r AND band r+1) — the
    # same [s:e+1] slice slice_cgrid_geometry_to_band applies to area_q.
    return [jnp.asarray(vmask[r * nl: r * nl + nl + 1]) for r in range(n_dev)]


def shard_state_latlon(state, mesh):
    """Lay out a ``LatLonCGridOceanState`` for the lat-band SPMD step.

    Cell / u-grid array fields (leading dim ``n_lat``) shard ``P("lat")``.  The
    staggered ``v`` / ``v_mask`` (leading dim ``n_lat+1``) are carried as
    ``v_lower = field[0:n_lat]`` (``n_lat`` rows, ``P("lat")``) — the top pole row
    ``field[n_lat]`` is a 0 wall on the regular grid and is reconstructed in-body
    by the band halo.  ``None`` fields pass through.  The inverse is
    :func:`gather_state_latlon`.

    This is the layout the in-``shard_map`` body of :func:`make_sharded_ocean_step`
    expects; the test uses it instead of a uniform ``tree.map(P("lat"))`` (which
    fails on ``v`` because ``n_lat+1`` is not divisible by ``N``).
    """
    def _shard_cell(field):
        if field is None:
            return None
        sh = NamedSharding(mesh, _lat_spec(field.data))
        return field.replace(data=jax.device_put(field.data, sh))

    def _shard_v(field):
        if field is None:
            return None
        # Drop the TOP v-row (the north pole wall, v==0 on the regular grid) so
        # the leading dim becomes n_lat (divisible by N).  ROUND-TRIP INVARIANT
        # (codex finding): shard_state_latlon -> gather_state_latlon re-appends a
        # ZERO top row, so the round-trip is a bitwise identity ONLY when the
        # input's top v-row is already zero (a valid masked regular-grid state;
        # the pole-wall BC enforces v[n_lat]=0 every step).  An arbitrary nonzero
        # top row (e.g. a raw IC perturbation) is dropped -> reconstructed as 0:
        # CORRECT for the dynamics (the pole wall zeros it at step 1) but not a
        # bit round-trip of that one row.  NOT for a tripole north fold (raises
        # in make_sharded_ocean_step) where the top row is a live fold partner.
        nlat1 = field.data.shape[0]
        v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
        sh = NamedSharding(mesh, _lat_spec(v_lower))
        return field.replace(data=jax.device_put(v_lower, sh))

    updates = {}
    for name in _V_STAGGERED_STATE_FIELDS:
        updates[name] = _shard_v(getattr(state, name))
    # Every other (array) leaf: shard P("lat").  NamedTuple fields not in the
    # v-staggered set and not None get the cell sharding; None stays None.
    for name in state._fields:
        if name in _V_STAGGERED_STATE_FIELDS:
            continue
        val = getattr(state, name)
        if val is None:
            updates[name] = None
        elif hasattr(val, "data"):                 # a Field
            updates[name] = _shard_cell(val)
        else:
            # Non-Field, non-None leaf (e.g. a raw array carry like psi).
            # Shard 2-D+ on lat, replicate lower-rank — matches shard_pytree.
            arr = jnp.asarray(val)
            spec = _lat_spec(arr) if arr.ndim >= 1 else P()
            updates[name] = jax.device_put(arr, NamedSharding(mesh, spec))
    return state._replace(**updates)


def gather_state_latlon(state, mesh):
    """Inverse of :func:`shard_state_latlon`: gather every leaf to a single device
    and rebuild the full ``(n_lat+1, ...)`` ``v`` / ``v_mask`` by appending the
    pole-wall row (zeros) the layout dropped.

    The appended row is the north pole wall (``v == 0`` there on the regular
    grid), which is exactly what the single-device rest/forced state carries, so
    the gathered state is bit-comparable to the single-device reference.
    """
    rep = NamedSharding(mesh, P())

    def _gather_arr(a):
        return jax.device_put(a, rep)

    updates = {}
    for name in _V_STAGGERED_STATE_FIELDS:
        field = getattr(state, name)
        if field is None:
            updates[name] = None
            continue
        v_lower = _gather_arr(field.data)
        pole = jnp.zeros_like(v_lower[:1])          # north pole-wall row
        full = jnp.concatenate([v_lower, pole], axis=0)
        updates[name] = field.replace(data=full)
    for name in state._fields:
        if name in _V_STAGGERED_STATE_FIELDS:
            continue
        val = getattr(state, name)
        if val is None:
            updates[name] = None
        elif hasattr(val, "data"):
            updates[name] = val.replace(data=_gather_arr(val.data))
        else:
            updates[name] = _gather_arr(jnp.asarray(val))
    return state._replace(**updates)


def _shard_forcing(forcing, mesh):
    """Lay out an external-forcing pytree (``OceanSurfaceForcing`` /
    ``FreshwaterForcing`` / ``SurfaceTracerForcing`` …) for the lat-band SPMD step.

    Every populated leaf of these NamedTuples is a CELL-CENTRED array of leading
    dim ``n_lat`` (``OceanSurfaceForcing`` carries the seven channels as
    ``(n_lat, n_lon)`` T-point arrays — ``omip2_applicator`` interpolates
    ``tau_x``/``tau_y`` to the u/v faces INSIDE the model dynamics, NOT before, so
    the forcing is never v-staggered).  They therefore shard ``P("lat")`` exactly
    like a cell state field; ``None`` leaves pass through untouched.

    ``forcing is None`` ⇒ ``None`` (the un-forced step, byte-identical).  The
    inverse is unnecessary — the forcing is a per-step INPUT, never gathered back.
    """
    if forcing is None:
        return None
    rep_or_lat = []
    for val in forcing:
        if val is None:
            rep_or_lat.append(None)
            continue
        arr = jnp.asarray(val)
        spec = _lat_spec(arr) if arr.ndim >= 1 else P()
        rep_or_lat.append(jax.device_put(arr, NamedSharding(mesh, spec)))
    # Rebuild the SAME NamedTuple subclass (preserves field names / pytree def).
    return type(forcing)(*rep_or_lat)


def make_sharded_ocean_step(model, mesh):
    """Return ``step(state, dt, surface_forcing=None, freshwater=None) -> state``
    running ``model.step`` lat-band-SPMD.

    Parameters
    ----------
    model
        ``LatLonCGridOceanModel`` (regular lat-lon OR tripole: the north
        fold is applied data-dependently on the north band via
        ``north_fold_mask``/``apply_north_fold`` — gated by the tripole
        SPMD equivalence suite).
    mesh
        A 1-D ``jax.sharding.Mesh`` with axis name ``"lat"`` (from
        ``legoesm.parallel.mesh.create_latlon_mesh(...).mesh``).

    Notes
    -----
    Build the ``N`` band geometries + band vertex masks host-side, stack their
    ARRAY fields into a replicated pytree, and index by
    ``jax.lax.axis_index("lat")`` in the ``shard_map`` body (the geometry SCALAR
    fields stay static — see the module docstring).  ``dt`` is a TRACED
    replicated operand (so the jit-cached compile survives a dt change).
    ``check_vma=False`` (the JAX >= 0.8
    replication check) because the band halo intentionally reads neighbour-rank
    data (replication-unaware).

    The state must be laid out with :func:`shard_state_latlon` (``v`` /
    ``v_mask`` carried as the ``n_lat``-row ``v_lower``).  The body reconstructs
    each band's ``nl+1`` v-faces, runs the step on the band geometry, and
    converts the result back to the ``v_lower`` representation.

    ``surface_forcing`` / ``freshwater`` (per-step, optional): cell-shaped
    ``OceanSurfaceForcing`` / ``FreshwaterForcing`` pytrees, sharded ``P("lat")``
    by :func:`_shard_forcing` and forwarded into ``model._step_impl`` /
    ``_ab2_step`` (the in-core wind-stress / heat / virtual-salt terms).  ``None``
    (the default) is byte-identical to the original no-forcing step.
    """
    if mesh is None:                   # single-device: plain step
        return lambda state, dt, surface_forcing=None, freshwater=None: (
            model.step(state, dt, freshwater=freshwater,
                       surface_forcing=surface_forcing))

    n_dev = mesh.devices.size
    axis = mesh.axis_names[0]

    # Tripole north fold: the lat-lon C-grid operators select the fold on the
    # north band data-dependently (``north_fold_mask`` / ``apply_north_fold`` +
    # the ``where(north_mask)`` neumann-fill / TVD-clamp branches) under the
    # armed SPMD backend, since the single shard_map trace cannot use the serial
    # ``if fold_is_local`` (fold_j=-1 on every band).  The build_band_grids
    # slicer keeps the active fold + perms on each band.  Correctness is gated by
    # the full-step SPMD-vs-serial equivalence test on a synthetic tripole
    # (tests/parallel/test_latlon_ocean_spmd_tripole.py).

    # --- host-side band geometries + vertex masks (replicated, indexed in-body) ---
    band_grids = build_band_grids(model.grid, n_dev)
    band_vmasks = _build_band_vertex_masks(model, n_dev)
    template = band_grids[0]           # static-scalar source (uniform bands)
    array_field_names = _geom_array_field_names(template)

    # Stack each geometry ARRAY field over the band axis (rank 0..N-1) -> a
    # replicated pytree the body indexes by axis_index.  Replicate (P()) so every
    # device holds the whole small grid stack.
    rep = NamedSharding(mesh, P())
    geom_stacks = {
        name: jax.device_put(
            jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
                      axis=0),
            rep)
        for name in array_field_names
    }
    vmask_stack = jax.device_put(
        jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0), rep)

    # Static perms for the v north-boundary-row ppermute (band r receives band
    # r+1's v_lower[0] = global v[e]; north band non-target receives 0).
    perm_north, _perm_south = latlon_band_perms(n_dev)

    def _body(state_local, geom_stacks_local, vmask_stack_local, dt,
              surface_forcing_local, freshwater_local):
        r = jax.lax.axis_index(axis)

        # Rebuild this band's geometry: index the stacked arrays at r, keep the
        # static scalars (n_lat_local, n_lon, radius, dlon, dlat, fold) from the
        # band template.  The fold is inactive + identical on every band here.
        geom_arrays = {name: geom_stacks_local[name][r]
                       for name in array_field_names}
        band_geom = template._replace(**geom_arrays)
        band_vmask = vmask_stack_local[r]

        # Reconstruct the band's nl+1 v-faces from the nl-row v_lower.  band r's
        # north boundary row is band r+1's v_lower[0] (= global v[e]); the north
        # band (no r+1) receives the pole-wall 0 via the ppermute non-target.
        def _reconstruct_v(field):
            if field is None:
                return None
            v_lower = field.data                      # (nl, n_lon[, nlev])
            boundary = jax.lax.ppermute(
                v_lower[0:1], axis, perm_north)        # band r+1's first row -> 0 at top
            v_band = jnp.concatenate([v_lower, boundary], axis=0)  # nl+1 rows
            return field.replace(data=v_band)

        # Convert a returned nl+1 v back to the nl-row v_lower (drop the boundary
        # row, owned by band r+1).
        def _to_v_lower(field):
            if field is None:
                return None
            return field.replace(data=field.data[:-1])

        v_updates = {name: _reconstruct_v(getattr(state_local, name))
                     for name in _V_STAGGERED_STATE_FIELDS}
        state_band = state_local._replace(**v_updates)

        # The forcing pytrees are CELL-shaped (P("lat")), already band-sliced by
        # shard_map's in_specs to this band's n_lat rows — forwarded verbatim to
        # the in-core wind-stress / heat / virtual-salt application (no halo / v
        # reconstruction needed; tau is interp_cell_to_{u,v}face INSIDE the step,
        # which routes through the armed SPMD band halo).
        result = _step_body(model, state_band, dt,
                            grid=band_geom, vertex_mask=band_vmask,
                            surface_forcing=surface_forcing_local,
                            freshwater=freshwater_local)

        out_updates = {name: _to_v_lower(getattr(result, name))
                       for name in _V_STAGGERED_STATE_FIELDS}
        return result._replace(**out_updates)

    # Build the JITTED shard_map ONCE per input STRUCTURE and cache it.
    # ``jax.jit`` is LOAD-BEARING: a bare shard_map is NOT compilation-cached,
    # so calling it re-traces + recompiles the (large, un-jitted) band step
    # EVERY call — the exact 142 s/step pathology the atm lat-band step hit
    # (fixed in fa2ce32b6); production ``--n-gpus`` would recompile per model
    # step.  ``dt`` is a TRACED operand (not a closure constant) so a changing
    # dt does not retrigger compilation.  The cache is keyed by the (state,
    # forcing) treedefs because the shard_map in/out specs are built from the
    # pytree structure (a run flips forcing None->pytree at most once).
    _cache = {}

    def sharded_step(state, dt, surface_forcing=None, freshwater=None):
        # Lay out the forcing pytrees the same way the state cell fields are laid
        # out (P("lat")); None passes through as None (the un-forced step).
        sf_sharded = _shard_forcing(surface_forcing, mesh)
        fw_sharded = _shard_forcing(freshwater, mesh)
        key = (jax.tree.structure(state),
               jax.tree.structure(sf_sharded),
               jax.tree.structure(fw_sharded))
        fn = _cache.get(key)
        if fn is None:
            in_spec = jax.tree.map(_lat_spec, state)
            geom_spec = jax.tree.map(lambda _x: P(), geom_stacks)
            vmask_spec = P()
            # Cell-shaped forcing leaves shard P("lat"); a None forcing is a None
            # operand with a None spec (tree.map over None yields None — shard_map
            # accepts a None operand/spec pair).  Build the specs FROM the (already
            # device_put) sharded pytrees so the leaf structure matches per call.
            sf_spec = jax.tree.map(_lat_spec, sf_sharded)
            fw_spec = jax.tree.map(_lat_spec, fw_sharded)
            # JAX >= 0.8 top-level shard_map takes ``check_vma`` (the replication
            # check); the band halo reads neighbour-rank data so disable it (same
            # as the validated PCG / halo-parity shard_maps).  dt is a replicated
            # scalar operand (P()).
            fn = jax.jit(shard_map(
                _body,
                mesh=mesh,
                in_specs=(in_spec, geom_spec, vmask_spec, P(),
                          sf_spec, fw_spec),
                out_specs=in_spec,
                check_vma=False,
            ))
            _cache[key] = fn
        # Arm the SPMD halo backend ONLY around the call, then RESTORE the
        # previous backend (codex finding): leaving it globally armed makes a
        # later serial/full-domain ocean call take SPMD-only branches (axis_index /
        # ppermute / psum in pad_with_pole_bc_lat, conservation, eta_floor) OUTSIDE
        # a shard_map -> crash.  The FIRST call traces WITH the backend armed
        # (baking the SPMD halo/reduction ops into the compiled program); later
        # calls reuse the cached compile, for which the arming is a harmless
        # no-op — same contract as the atm lat-band step.
        # Save+restore the FULL backend state (backend + MPI topology + SPMD mesh)
        # so a prior "mpi"/"spmd" backend is restored intact: activate_* clears the
        # MPI topology, and set_halo_backend("mpi") REQUIRES a topology (codex).
        from legoesm.grids.halo import (
            get_halo_backend, get_mpi_topology, get_spmd_mesh,
            set_halo_backend, set_spmd_mesh,
        )
        _prev_backend = get_halo_backend()
        _prev_topo = get_mpi_topology()
        _prev_mesh = get_spmd_mesh()
        activate_latlon_spmd_halo(mesh)
        try:
            return fn(state, geom_stacks, vmask_stack, jnp.asarray(dt),
                      sf_sharded, fw_sharded)
        finally:
            set_spmd_mesh(_prev_mesh)
            set_halo_backend(_prev_backend, _prev_topo)

    return sharded_step


def make_sharded_ocean_step_global(model, mesh):
    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
    that takes a GLOBAL (single-device-layout) state + forcing and returns a
    GLOBAL state — the minimal-diff driver entry point.

    Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
    (:func:`shard_state_latlon` + :func:`_shard_forcing`), runs the lat-band SPMD
    step, then gathers the state OUT (:func:`gather_state_latlon`).  This lets the
    OMIP host loop keep operating on a normal full-domain state — the per-step
    host BCs (SSS restore, prognostic ice, geothermal, BBL, nudge) see the
    gathered global state UNCHANGED — at the cost of a per-step gather/scatter
    (acceptable for the host-coupled OMIP driver; the pure-dynamics inner loop
    should use :func:`make_sharded_ocean_step` directly to stay sharded).

    ``mesh is None`` ⇒ the plain single-device ``model.step`` (no scatter/gather).
    """
    if mesh is None:                   # single-device: plain step
        return lambda state, dt, surface_forcing=None, freshwater=None: (
            model.step(state, dt, freshwater=freshwater,
                       surface_forcing=surface_forcing))

    inner = make_sharded_ocean_step(model, mesh)

    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
        # Scatter the global state to the band layout; the forcing is sharded
        # INSIDE ``inner`` (it calls _shard_forcing), so pass it through global.
        ss = shard_state_latlon(state, mesh)
        ss = inner(ss, dt, surface_forcing=surface_forcing,
                   freshwater=freshwater)
        return gather_state_latlon(ss, mesh)

    return sharded_step_global
