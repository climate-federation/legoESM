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

* GRID = **band-stacked, P("lat")-SHARDED** (Structure A, sharded since
  #1370-iii). The ``N`` band geometries are built host-side
  (``build_band_grids``), their ARRAY fields are ``jnp.stack``-ed over a
  leading band axis, sharded ``P("lat")`` so each device holds ONLY its own
  band's slab, and the in-``shard_map`` body reads its local slab at
  ``[0]``. This AVOIDS staggered-sharding the grid itself (the v-row is
  ``n_lat+1``, coprime with ``n_lat`` for ``N>1``). The
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
    reconstruct_vface_lower_multi,
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
               freshwater=None, surface_forcing=None, sponge=None,
               t_seconds=None):
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
            state, dt, freshwater=freshwater,
            surface_forcing=surface_forcing, sponge=sponge,
            grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds)
    else:
        new_state = model._step_impl(
            state, dt, freshwater=freshwater,
            surface_forcing=surface_forcing, sponge=sponge,
            grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds)
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


def _content_hash48(arr) -> float:
    """48-bit content digest of ``arr``'s bytes, exactly representable in f64.

    Used to compare EXACT-dtype arrays (masks, index tables) across
    processes: unlike moment fingerprints, a byte digest is positional, so a
    permutation or a two-cell flip cannot cancel. 48 bits keeps the value
    under 2**53 so it survives the float64 ``process_allgather`` payload
    exactly. Not cryptographic — collision-resistance at 2**-48 is far
    beyond the ~10 setup-time comparisons this guard makes.
    """
    import hashlib

    a = np.ascontiguousarray(arr)
    h = hashlib.blake2b(a.tobytes(), digest_size=6)
    return float(int.from_bytes(h.digest(), "big"))

def geom_band_fingerprint(host, n_bands):
    """Low-memory cross-process fingerprint of a band-STACKED geometry field.

    ``host`` has leading axis ``n_bands`` (the per-device band stack). The
    fingerprint is PER BAND (codex 2026-08-03 r14: whole-array moments let a
    band-local drift hide in the global sum once each process's own bytes
    become live computation inputs):

    * exact dtypes (int/bool/uint — masks, index tables): one positional
      48-bit byte digest per band (a permutation or two-cell flip within a
      band cannot cancel);
    * float dtypes: per-band ``[sum, sum_of_squares, absmax]`` of the finite
      entries in float64, plus per-band non-finite counts in ``struct``.
      DOCUMENTED RESIDUALS: a within-band float change preserving all
      three moments to rtol is not detected, and non-finite entries that
      change POSITION or kind (nan vs inf) with an unchanged per-band
      count also pass; band grids are analytic in lat/lon, so any real
      inconsistency moves the moments.

    Returns ``(struct, vals, is_exact)`` as float64 arrays safe for
    ``process_allgather``.
    """
    import numpy as _np

    host = _np.asarray(host)
    assert host.shape[0] == n_bands, (host.shape, n_bands)
    is_exact = host.dtype.kind in "biu"
    struct = [float(host.ndim), *map(float, host.shape),
              float(_np.dtype(host.dtype).num)]
    if is_exact:
        vals = _np.array([_content_hash48(host[b]) for b in range(n_bands)],
                         dtype=_np.float64)
    else:
        per_band = []
        for b in range(n_bands):
            flat = host[b].ravel()
            finite = flat[_np.isfinite(flat)]
            f64 = finite.astype(_np.float64)
            struct.append(float(flat.size - finite.size))
            per_band.extend([
                float(f64.sum()) if f64.size else 0.0,
                float((f64 * f64).sum()) if f64.size else 0.0,
                float(_np.abs(f64).max()) if f64.size else 0.0,
            ])
        vals = _np.array(per_band, dtype=_np.float64)
    return _np.array(struct, dtype=_np.float64), vals, is_exact


def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=1e-5):
    """True iff every process's fingerprint matches process 0's.

    ``g_struct``/``g_vals`` are the ``process_allgather``-ed outputs of
    :func:`geom_band_fingerprint` (leading axis = process). Structure and
    exact-dtype digests compare EXACTLY; float moments to ``rtol``.
    """
    import numpy as _np

    struct_ok = bool(_np.all(g_struct == g_struct[0]))
    if is_exact:
        vals_ok = bool(_np.all(g_vals == g_vals[0]))
    else:
        vals_ok = bool(_np.allclose(g_vals, g_vals[0], rtol=rtol, atol=0.0))
    return struct_ok and vals_ok



def _schema_fingerprint(names, n_dev) -> np.ndarray:
    """Fixed-shape schema digest: field-name list, count, x64 flag, n_dev.

    Gathered ONCE before the per-field loop so a process-dependent field
    selection is caught by a collective every process reaches, instead of
    desynchronizing the per-field gathers (codex round-5 findings 3/4).
    """
    import hashlib

    joined = ",".join(names).encode()
    digest = float(int.from_bytes(
        hashlib.blake2b(joined, digest_size=6).digest(), "big"))
    return np.array(
        [float(len(names)), digest, float(bool(jax.config.jax_enable_x64)),
         float(n_dev)], dtype=np.float64)


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


def _addressable_shard_put(arr, sharding):
    """Put an array onto a (possibly multi-process) sharding WITHOUT jax's
    whole-array cross-process ``assert_equal``.

    Under multicontroller, ``jax.device_put(numpy_array, sharding)`` calls
    ``multihost_utils.assert_equal`` on the FULL array
    (jax _src/dispatch.py::_device_put_sharding_impl) — a
    ``process_allgather`` whose output is ``[n_processes, *shape]`` on one
    device: at LL2304 L20 one 3-D field is 849 MB, so 96 processes fetch
    81.5 GB > an 80 GB A100 (the wall that killed oc @96/@128, jobs
    26644681/26644682, AFTER the geometry-broadcast fix; @64 = 54.3 GB
    just fit, which is why smaller ladders never saw it).

    Single-process: the historical ``jax.device_put``, byte-unchanged and
    with no host round-trip (codex r17 — the input may already be a jax
    device array). Multicontroller: ``jax.make_array_from_callback``
    supplies each process's addressable shards directly — no consistency
    collective. The ``device_put`` bit-identity CONTRACT is preserved by
    the callers' cheap exact-hash gate (:func:`assert_pytree_bytes_equal`)
    instead of jax's full-array allgather.
    """
    if jax.process_count() <= 1:
        return jax.device_put(arr, sharding)
    host = np.asarray(arr)
    return jax.make_array_from_callback(
        host.shape, sharding, lambda idx: host[idx])


def assert_pytree_bytes_equal(tree, what):
    """Cheap multi-process replacement for the per-leaf ``assert_equal``
    that :func:`_addressable_shard_put` bypasses (codex r17 HIGH-1).

    Hashes every array leaf's BYTES (48-bit positional digest — the same
    exactness as ``device_put``'s contract) into one small vector,
    allgathers it, and refuses on any cross-process mismatch. Cost is one
    tiny collective + a host-side hash pass, independent of process count
    — vs jax's [n_processes, full_array] allgather.

    No-op single-process. Symmetric: every process hashes the same leaves
    in the same order, so all raise or none.
    """
    if jax.process_count() <= 1:
        return
    from jax.experimental import multihost_utils

    leaves = [x for x in jax.tree_util.tree_leaves(tree)
              if hasattr(x, "ndim")]
    vals = np.array([_content_hash48(np.asarray(x)) for x in leaves],
                    dtype=np.float64)
    g = multihost_utils.process_allgather(vals)
    if not bool(np.all(g == g[0])):
        bad = [i for i in range(len(leaves))
               if not bool(np.all(g[:, i] == g[0, i]))]
        raise RuntimeError(
            f"{what}: array leaves {bad} differ across processes (48-bit "
            f"byte digests disagree) — the inputs each process built are "
            f"NOT identical, which the removed jax device_put assert "
            f"would have refused. Fix the per-process build before "
            f"sharding.")


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
    # v-carrier contract (see make_sharded_ocean_step's fold note): the TOP
    # v-face row (regular pole wall OR tripole seam/cap row) must be
    # wall-masked — the carrier drops it and reconstructs it as zero, which
    # would silently delete a LIVE seam row.  Host-side check on the
    # concrete state (this fn runs outside jit).
    assert_pytree_bytes_equal(state, "shard_state_latlon")
    vm = getattr(state, "v_mask", None)
    if vm is not None:
        import numpy as _np

        if _np.asarray(vm.data)[-1].any():
            raise ValueError(
                "shard_state_latlon: the state's TOP v-face row is LIVE "
                "(v_mask[-1] has ocean faces) — the lat-band v-carrier "
                "drops that row and reconstructs it as the pole/cap wall "
                "zero, which would silently delete seam velocities. "
                "Mask the cap row (the tripole cap convention) or extend "
                "the carrier before sharding this state.")

    def _shard_cell(field):
        if field is None:
            return None
        sh = NamedSharding(mesh, _lat_spec(field.data))
        return field.replace(data=_addressable_shard_put(field.data, sh))

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
        return field.replace(data=_addressable_shard_put(v_lower, sh))

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
            # ndim>=1 lat-shards via _lat_spec (1-D included); only true
            # scalars replicate (codex r17: the old "replicate lower-rank"
            # wording did not match _lat_spec's behaviour).
            arr = jnp.asarray(val)
            spec = _lat_spec(arr) if arr.ndim >= 1 else P()
            updates[name] = _addressable_shard_put(arr, NamedSharding(mesh, spec))
    return state._replace(**updates)


def shard_forcing_latlon(forcing, mesh):
    """Lay out a forcing pytree (``FreshwaterForcing`` / ``OceanSurfaceForcing``
    / ``SpongeForcing`` — or any nesting of them) for the lat-band SPMD step.

    Every OMIP forcing field is CELL-CENTERED ``(n_lat, n_lon[, nlev])`` (the
    cell->face wind-stress interpolation happens INSIDE the step through the
    SPMD-aware halo pads), so array leaves shard ``P("lat", None, ...)`` with
    no ``v_lower`` handling; scalars replicate; ``None`` fields pass through
    untouched (they vanish from the pytree structure, matching the specs the
    step derives).  ``forcing=None`` returns ``None``.
    """
    if forcing is None or mesh is None:
        return forcing
    assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")

    def _put(leaf):
        if leaf is None:
            return None
        arr = jnp.asarray(leaf)
        return _addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))

    return jax.tree.map(_put, forcing)


def shard_forcing_stack_latlon(stack, mesh):
    """Lay out a STACKED per-block forcing pytree for the lat-band SPMD
    block-scan (the ``run_omip`` JRA55 lanes; see
    ``_build_jra55_block_fn`` / ``_build_jra55_block_fn_interp``).

    Unlike :func:`shard_forcing_latlon` (per-step, lat at axis 0), the
    block builders stack ``N`` steps / raw records along a LEADING axis,
    so the lat axis sits at position 1.  A ``(n_rec, n_lat, n_lon[, ...])``
    leaf therefore shards ``P(None, "lat", ...)`` (records replicated, the
    time index stays shard-local so the in-scan interpolation needs no
    cross-band comm); a bare ``(n_lat, n_lon)`` leaf shards ``P("lat", None)``;
    1-D metadata / scalars replicate; ``None`` and non-array leaves pass
    through.  ``mesh=None`` returns ``stack`` unchanged (serial lane).

    CONTRACT: rank is the ONLY signal used, so a rank-2 leaf is assumed to
    be a ``(n_lat, n_lon)`` field and is lat-sharded on axis 0.  Any future
    metadata that is genuinely rank-2 but NOT lat-major (e.g. a
    ``(n_rec, n_meta)`` table) would be silently mis-sharded — keep such
    metadata 1-D (or replicate it explicitly) before it reaches this helper.

    Keeping this next to :func:`shard_state_latlon` means the driver and
    the parity tests share ONE layout definition — the block-scan forcing
    stack must be laid out consistently with the state the sharded step
    carries, and a second copy would drift.
    """
    if mesh is None:
        return stack

    assert_pytree_bytes_equal(stack, "shard_forcing_stack_latlon")

    def _put(leaf):
        if leaf is None or not hasattr(leaf, "ndim"):
            return leaf
        arr = jnp.asarray(leaf)
        if arr.ndim >= 3:
            spec = P(None, "lat", *((None,) * (arr.ndim - 2)))
        elif arr.ndim == 2:
            spec = P("lat", None)
        else:
            spec = P()
        return _addressable_shard_put(arr, NamedSharding(mesh, spec))

    return jax.tree.map(_put, stack)


def append_vface_wall_row(v_lower):
    """Rebuild the full ``(n_lat+1, ...)`` staggered v-array from the lat-band
    carrier ``v_lower`` by appending the TOP wall row (zeros).

    This is THE reconstruction of the row the carrier drops: the regular-grid
    north pole wall / tripole cap row, identically zero under the v-carrier
    contract (:func:`shard_state_latlon` REFUSES a state whose top v-face row
    is live), so the result is bit-identical to the original staggered array.
    Shared by :func:`gather_state_latlon` (full-state gather) and the
    persistent-lane DEVICE-SIDE staggered reads (the OMIP driver's
    surface-current consumers, ``run_omip_core2._surface_uv_faces``) so the
    row reconstruction is written once.  Works on any trailing shape (3-D
    leaves or 2-D surface slices) and preserves sharding under GSPMD.
    """
    wall = jnp.zeros_like(v_lower[:1])
    return jnp.concatenate([v_lower, wall], axis=0)


def gather_state_latlon(state, mesh):
    """Inverse of :func:`shard_state_latlon`: gather every leaf to a single device
    and rebuild the full ``(n_lat+1, ...)`` ``v`` / ``v_mask`` by appending the
    pole-wall row (zeros) the layout dropped.

    The appended row is the north pole wall (``v == 0`` there on the regular
    grid), which is exactly what the single-device rest/forced state carries, so
    the gathered state is bit-comparable to the single-device reference.

    Multi-controller (route-B ``jax.distributed``, mesh spanning processes):
    replication routes through a jit-compiled identity instead of
    ``device_put`` (see :func:`legoesm.parallel.latlon_spmd.replicate_leaf`,
    the primitive shared with the atm gather); the single-process path is
    byte-unchanged.
    """
    from legoesm.parallel.latlon_spmd import replicate_leaf

    rep = NamedSharding(mesh, P())
    _mp = jax.process_count() > 1

    def _gather_arr(a):
        return replicate_leaf(a, rep, multiprocess=_mp)

    updates = {}
    for name in _V_STAGGERED_STATE_FIELDS:
        field = getattr(state, name)
        if field is None:
            updates[name] = None
            continue
        v_lower = _gather_arr(field.data)
        # north pole-wall / cap row (the shared reconstruction helper)
        updates[name] = field.replace(data=append_vface_wall_row(v_lower))
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


def make_sharded_ocean_step(model, mesh):
    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
    sponge=None, t_seconds=None) -> state`` running ``model.step``
    lat-band-SPMD.

    The forcing channels mirror ``model.step``'s keyword surface: pass
    pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
    ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
    is a replicated scalar like ``dt``).  ``None`` forcing keeps the
    dynamics-only program; each distinct None<->populated combination
    compiles (and caches) its own executable.

    Parameters
    ----------
    model
        ``LatLonCGridOceanModel`` (latlon geometry; tripole north-fold is a
        follow-up — raises ``NotImplementedError`` if ``model.grid.fold`` is
        active).
    mesh
        A 1-D ``jax.sharding.Mesh`` with axis name ``"lat"`` (from
        ``legoesm.parallel.mesh.create_latlon_mesh(...).mesh``).

    Notes
    -----
    Build the ``N`` band geometries + band vertex masks host-side, stack their
    ARRAY fields over a leading band axis SHARDED ``P("lat")`` (each device
    holds only its own slab; the ``shard_map`` body reads it at ``[0]``; the
    geometry SCALAR fields stay static — see the module docstring).  ``dt`` is a TRACED,
    replicated operand and the jitted ``shard_map`` is built once and cached
    (a per-call rebuild re-traced the whole ocean step every call — see the
    ``_cache`` note below).  ``check_vma=False`` (the JAX >= 0.8
    replication check) because the band halo intentionally reads neighbour-rank
    data (replication-unaware).

    The state must be laid out with :func:`shard_state_latlon` (``v`` /
    ``v_mask`` carried as the ``n_lat``-row ``v_lower``).  The body reconstructs
    each band's ``nl+1`` v-faces, runs the step on the band geometry, and
    converts the result back to the ``v_lower`` representation.
    """
    if mesh is None:                   # single-device: plain step
        return lambda state, dt, **forcing_kwargs: model.step(
            state, dt, **forcing_kwargs)

    n_dev = mesh.devices.size
    axis = mesh.axis_names[0]

    # Tripole north-fold (scaling-audit item 4): SUPPORTED under the same
    # v-carrier contract as the regular grid.  Every fold-touching operator
    # is already uniform-program fold-capable (the data-dependent
    # ``north_fold_mask``/``apply_north_fold`` selection on
    # ``axis_index == N-1`` — gated by test_latlon_spmd_northfold), and
    # ``build_band_grids``' slicer keeps ``is_active`` rank-consistent with
    # the ``fold_j=-1`` sentinel off the north band.  The one structural
    # assumption is the v-carrier's: the TOP v-face row (the seam/cap row,
    # ``v[n_lat]``) must be WALL-MASKED so the in-body reconstruction's
    # zero row is exact — true for the cap-row convention of
    # ``create_synthetic_tripole`` and the eORCA masks (``v_mask[-1] == 0``;
    # the serial step keeps ``v[-1] == 0`` identically).  That contract is
    # asserted on the CONCRETE state in :func:`shard_state_latlon` — a live
    # (unmasked) seam v-row refuses loudly there instead of silently
    # reconstructing zeros here.

    # --- host-side band geometries + vertex masks (band-stacked, P("lat")-sharded) ---
    band_grids = build_band_grids(model.grid, n_dev)
    band_vmasks = _build_band_vertex_masks(model, n_dev)
    template = band_grids[0]           # static-scalar source (uniform bands)
    array_field_names = _geom_array_field_names(template)

    # Stack each geometry ARRAY field over the band axis (rank 0..N-1) and
    # SHARD along that axis (#1370 stage (iii), codex round-18): each device
    # holds ONLY its own band's slab instead of the whole global stack —
    # this was one of the residual ~1.4-1.7 global-field-equivalents of
    # per-device residency left after the host-side-build fix (probe
    # 26524423). The leading axis has length n_dev, so P("lat") divides it
    # exactly; the body indexes its local slab at [0]. Values are unchanged
    # — same stack, different placement; the cross-process divergence
    # guard below runs on HOST values and is placement-blind.
    rep = NamedSharding(mesh, P("lat"))

    def _replicated_put(arr, name):
        # (Name kept for history; since 2026-08-03 this is a SHARDED stack
        # put.) The band-geometry arrays are (re)computed per process and
        # can differ in their last ULPs (per-process XLA autotuning on
        # device-derived grid fields) — the fully-replicated-put era
        # broadcast process 0's bytes to sidestep the P() bit-identity
        # assert (job 26450848). With the #1370-iii P("lat") sharding each
        # process's devices consume ONLY its own band rows, so the
        # broadcast became both unnecessary and, at nd>=96, fatal (its
        # psum program is nd x the stack — see the note at the put below).
        # GUARD (codex round-3): process 0 must not silently mask REAL
        # cross-process divergence. Compare an allgathered fingerprint:
        # structural entries exactly; value entries EXACTLY for integer/bool
        # arrays (masks are comparison results — bit-reproducible, and an
        # exact compare is the only way to catch a two-cell flip that cancels
        # in the sum, codex round-4) and to rtol 1e-5 for float arrays (only
        # ULP autotune drift is expected there; quantize-then-assert-equal
        # false-positived on a rounding boundary, job 26453240).
        # NO DEADLOCK RISK: every process fingerprints the same fields in the
        # same order and derives the verdict from the SAME gathered array, so
        # the refusal is symmetric — all raise or none.
        # Residual (documented): a float-geometry divergence preserving sum,
        # sum-of-squares AND absmax to 1e-5 is not detected; band grids are
        # analytic in lat/lon, so any real inconsistency moves those moments.
        host = np.asarray(arr)
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils

            # PER-BAND fingerprints (module-level, unit-tested): exact
            # dtypes hash positionally per band; floats compare per-band
            # moments to rtol 1e-5 — bounds each band's drift instead of
            # letting it hide in a whole-array sum, since each process's
            # own bytes are now the live inputs for the bands it owns
            # (codex r14). A mask that genuinely differs across processes
            # means different wet domains = different physics: refusing is
            # correct, not a false alarm.
            struct, vals, is_exact = geom_band_fingerprint(
                host, host.shape[0])
            g_struct = multihost_utils.process_allgather(struct)
            g_vals = multihost_utils.process_allgather(vals)
            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
                raise RuntimeError(
                    f"make_sharded_ocean_step: band-geometry field {name!r} "
                    f"DIVERGES across processes (exact_dtype={is_exact}, "
                    f"gathered={g_vals.tolist()}) — a real config/grid "
                    f"inconsistency, not autotune noise; refusing to "
                    f"shard it.")
            # NO broadcast_one_to_all here (removed 2026-08-03): its psum
            # program is [n_processes, stack] in / P() fully-replicated out,
            # so its logical arg bytes are nd x the global stack — 82.4 GB
            # at nd=96 and 109.6 GB at nd=128 for one 3-D field, the
            # near-linear-in-nd wall that killed oc @96/@128 (jobs
            # 26642771/26636762) while @64 sat just under XLA's 63.8 GB
            # limit.  The target sharding is P("lat"): each process's
            # devices consume ONLY its own band rows, so cross-process
            # byte-identity of non-owned rows is irrelevant, and REAL
            # divergence is already refused by the fingerprint gate above.
            # make_array_from_callback hands each process exactly its
            # addressable slabs — the same #1100 pattern as the state
            # build — with no global-sized collective program at all.
        host_np = np.asarray(host)
        return jax.make_array_from_callback(
            host_np.shape, rep, lambda idx: host_np[idx])

    if jax.process_count() > 1:
        # Schema gate FIRST (one fixed-shape collective every process
        # reaches): a process-dependent field list or a mixed
        # jax_enable_x64 setting would otherwise desynchronize the
        # per-field gathers below instead of failing with a clear message.
        from jax.experimental import multihost_utils as _mhu

        _g = _mhu.process_allgather(
            _schema_fingerprint(list(array_field_names), n_dev))
        if not bool(np.all(_g == _g[0])):
            raise RuntimeError(
                "make_sharded_ocean_step: the band-geometry SCHEMA differs "
                "across processes (field list / x64 setting / device count "
                f"— gathered {_g.tolist()}). Fix the per-process config "
                "before sharding; the per-field checks below assume one "
                "schema.")

    geom_stacks = {
        name: _replicated_put(
            jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
                      axis=0), name)
        for name in array_field_names
    }
    vmask_stack = _replicated_put(
        jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0),
        "vertex_mask")

    # Static perms for the v north-boundary-row ppermute (band r receives band
    # r+1's v_lower[0] = global v[e]; north band non-target receives 0).
    perm_north, _perm_south = latlon_band_perms(n_dev)

    def _body(state_local, forcing_local, geom_stacks_local,
              vmask_stack_local, dt):
        fw_local, sf_local, sponge_local, t_s_local = forcing_local
        r = jax.lax.axis_index(axis)

        # Rebuild this band's geometry. The stacks are SHARDED P("lat") on
        # their leading band axis, so inside shard_map each device's local
        # view is its own (1, ...) slab — index [0], NOT [r] (stage (iii);
        # [r] was the replicated-stack indexing). Static scalars
        # (n_lat_local, n_lon, radius, dlon, dlat, fold) come from the band
        # template. The fold is inactive + identical on every band here.
        geom_arrays = {name: geom_stacks_local[name][0]
                       for name in array_field_names}
        band_geom = template._replace(**geom_arrays)
        band_vmask = vmask_stack_local[0]

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

        # Fused v-carrier reconstruction (scaling-M4): under the SAME
        # trace-time switch as the pad aggregation, pack the boundary-row
        # ppermutes of ALL staggered carriers (v + v_mask) into one
        # collective per dtype group — value-identical (a bit-copy
        # exchange; the flag flip re-keys the sharded_step cache below so
        # a reused step object rebuilds).  Default OFF = the historical
        # per-field ppermutes, byte-identical.
        import os as _os
        _fused_v = _os.environ.get(
            "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
        present = [name for name in _V_STAGGERED_STATE_FIELDS
                   if getattr(state_local, name) is not None]
        if _fused_v and len(present) > 1:
            fulls = reconstruct_vface_lower_multi(
                tuple(getattr(state_local, n).data for n in present),
                axis, perm_north)
            v_updates = {n: getattr(state_local, n).replace(data=f)
                         for n, f in zip(present, fulls)}
            for name in _V_STAGGERED_STATE_FIELDS:
                v_updates.setdefault(name, None)
        else:
            v_updates = {name: _reconstruct_v(getattr(state_local, name))
                         for name in _V_STAGGERED_STATE_FIELDS}
        state_band = state_local._replace(**v_updates)

        result = _step_body(model, state_band, dt,
                            grid=band_geom, vertex_mask=band_vmask,
                            freshwater=fw_local, surface_forcing=sf_local,
                            sponge=sponge_local, t_seconds=t_s_local)

        out_updates = {name: _to_v_lower(getattr(result, name))
                       for name in _V_STAGGERED_STATE_FIELDS}
        return result._replace(**out_updates)

    # Build the JITTED shard_map ONCE and cache it — the atm sibling's fix
    # (make_sharded_atm_latlon_step, probe 8561202): a bare shard_map is NOT
    # compilation-cached, so rebuilding it per call re-traced + recompiled the
    # (large, un-jitted) ocean band step EVERY call (~80 s/step at 16x32x3 nd=4
    # on CPU — host tracing, not compute; the bench's scaling number was
    # meaningless). ``dt`` is a TRACED operand (was a mutated closure cell,
    # which also forced the per-call rebuild) so a changing dt does not
    # retrigger compilation.
    _cache = {}
    n_lat_global = int(model.grid.n_lat)

    def _validate_forcing_layout(forcing):
        """Loudly refuse forcing leaves the lat-band shard cannot split.

        Every OMIP forcing field is CELL-CENTERED ``(n_lat, n_lon[, nlev])``
        — there are no v-face ``(n_lat+1, …)`` forcing arrays, so no
        ``v_lower`` handling.  A wrong-leading-dim leaf would otherwise die
        inside shard_map with an opaque divisibility error.
        """
        leaves, _ = jax.tree_util.tree_flatten_with_path(forcing)
        for path, leaf in leaves:
            nd = int(getattr(leaf, "ndim", np.ndim(leaf)))
            if nd == 1:
                # No OMIP forcing field is 1-D; _lat_spec would shard a
                # profile/staggered vector over "lat" silently-wrong
                # (codex r1 #2).
                raise ValueError(
                    f"sharded ocean step: forcing leaf "
                    f"{jax.tree_util.keystr(path)} is 1-D "
                    f"(shape {tuple(leaf.shape)}); forcing must be "
                    f"cell-centered (n_lat, n_lon[, nlev]) arrays or "
                    f"scalars.")
            if nd >= 2 and int(leaf.shape[0]) != n_lat_global:
                raise ValueError(
                    f"sharded ocean step: forcing leaf {jax.tree_util.keystr(path)} "
                    f"has leading dim {leaf.shape[0]} != n_lat "
                    f"({n_lat_global}); forcing must be cell-centered "
                    f"(n_lat, n_lon[, nlev]) to shard on the lat axis.")

    def sharded_step(state, dt, freshwater=None, surface_forcing=None,
                     sponge=None, t_seconds=None, aux=None):
        # ``aux`` (codex/2-proc repro 2026-08-03): the geometry + vmask
        # stacks are SHARDED global arrays; when this wrapper runs INSIDE
        # an outer trace (a bench/driver ``jit``/``scan`` over the step —
        # jit-of-jit inlines the inner call), concrete closure arrays
        # become OUTER-TRACE CONSTANTS and jax's MLIR constant handler
        # tries to fetch their value — impossible for non-addressable
        # arrays (RuntimeError: 'Fetching value ... non-addressable'; the
        # multicontroller lane has been broken this way since the
        # #1370-iii stack sharding). Callers that wrap the step in their
        # own jit MUST thread ``step.aux`` through their jit boundary as
        # an ARGUMENT and pass it back here.
        # ONE forcing operand: None fields drop out of the pytree structure,
        # so specs derived by tree.map skip them automatically and the
        # structure key below distinguishes every None<->array combination.
        forcing = (freshwater, surface_forcing, sponge, t_seconds)
        _validate_forcing_layout((freshwater, surface_forcing, sponge))
        # Cache key = the state's AND forcing's pytree STRUCTURE, plus the
        # forcing leaves' RANKS: in_specs/out_specs are derived from them,
        # so a later call with a different structure (an optional field
        # flipping None <-> Field, a sea-ice lane populating sf.salt_flux,
        # the restoring lane passing no forcing at all) OR a same-field
        # rank change (SpongeForcing.gamma is legitimately 2-D horizontal
        # OR 3-D full-rank — same structure, different _lat_spec; codex r1
        # #1) must rebuild the shard_map rather than reuse stale specs.
        forcing_ndims = tuple(
            int(getattr(leaf, "ndim", np.ndim(leaf)))
            for leaf in jax.tree.leaves(forcing))
        # The SPMD fused-halo switch is read at TRACE time inside the pad
        # dispatch — flipping LEGOESM_LATLON_SPMD_FUSED_HALO on a reused
        # step object must rebuild the shard_map, not reuse a stale jaxpr
        # (codex, audit item 7).
        import os as _os

        _fused_halo = _os.environ.get(
            "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
        key = (jax.tree.structure(state), jax.tree.structure(forcing),
               forcing_ndims, _fused_halo)
        fn = _cache.get(key)
        if fn is None:
            in_spec = jax.tree.map(_lat_spec, state)
            # Forcing leaves are cell-centered -> plain lat-band specs;
            # scalars (t_seconds) replicate, exactly like dt.
            forcing_spec = jax.tree.map(_lat_spec, forcing)
            # Stage (iii): the stacks are banded on their leading axis, so
            # the shard_map spec matches their P("lat") placement (the body
            # indexes its local (1, ...) slab at [0]).
            geom_spec = jax.tree.map(lambda _x: P("lat"), geom_stacks)
            vmask_spec = P("lat")
            # JAX >= 0.8 top-level shard_map takes ``check_vma`` (the
            # replication check); the band halo reads neighbour-rank data so
            # disable it (same as the validated PCG / halo-parity shard_maps).
            fn = jax.jit(shard_map(
                _body,
                mesh=mesh,
                in_specs=(in_spec, forcing_spec, geom_spec, vmask_spec, P()),
                out_specs=in_spec,
                check_vma=False,
            ))
            _cache[key] = fn
        # Arm the SPMD halo backend ONLY around the call, then RESTORE the
        # previous backend (codex finding): leaving it globally armed makes a
        # later serial/full-domain ocean call take SPMD-only branches
        # (axis_index / ppermute / psum in pad_with_pole_bc_lat, conservation,
        # eta_floor) OUTSIDE a shard_map -> crash. The FIRST call traces with
        # the backend armed (baking the SPMD halo/reduction ops into the
        # compiled program); later calls reuse the cached compile, and the
        # arm/restore keeps any interleaved serial path untouched.
        # Save+restore the FULL backend state (backend + MPI topology + SPMD
        # mesh) so a prior "mpi"/"spmd" backend is restored intact: activate_*
        # clears the MPI topology, and set_halo_backend("mpi") REQUIRES a
        # topology (codex).
        from legoesm.grids.halo import (
            get_halo_backend, get_mpi_topology, get_spmd_mesh,
            set_halo_backend, set_spmd_mesh,
        )
        _prev_backend = get_halo_backend()
        _prev_topo = get_mpi_topology()
        _prev_mesh = get_spmd_mesh()
        activate_latlon_spmd_halo(mesh)
        try:
            _geom, _vmask = aux if aux is not None else (geom_stacks,
                                                        vmask_stack)
            return fn(state, forcing, _geom, _vmask, jnp.asarray(dt))
        finally:
            set_spmd_mesh(_prev_mesh)
            set_halo_backend(_prev_backend, _prev_topo)

    # Expose the stacks so outer-jit callers can pass them as arguments
    # (see the ``aux`` note in the signature).
    sharded_step.aux = (geom_stacks, vmask_stack)
    return sharded_step


def make_sharded_ocean_step_global(model, mesh):
    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
    that takes a GLOBAL (single-device-layout) state + forcing and returns a
    GLOBAL state — the minimal-diff driver entry point.

    Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
    (:func:`shard_state_latlon` + :func:`shard_forcing_latlon`), runs the lat-band
    SPMD step, then gathers the state OUT (:func:`gather_state_latlon`).  This lets
    the OMIP host loop keep operating on a normal full-domain state — the per-step
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
        # Scatter the global state AND forcing to the band layout explicitly
        # (codex r17 item 3: the old comment claimed inner sharded the
        # forcing; it forwarded it global and relied on implicit JIT input
        # placement — which under multicontroller pays jax's whole-array
        # device_put assert, the nd-linear wall this module removes).
        ss = shard_state_latlon(state, mesh)
        ss = inner(ss, dt,
                   surface_forcing=shard_forcing_latlon(surface_forcing, mesh),
                   freshwater=shard_forcing_latlon(freshwater, mesh))
        return gather_state_latlon(ss, mesh)

    return sharded_step_global
