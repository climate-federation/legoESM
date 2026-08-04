Reading additional input from stdin...
OpenAI Codex v0.145.0
--------
workdir: /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
model: gpt-5.6-terra
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019fc622-7b1f-7bd2-927b-41363c142a1f
--------
user
Round-16 final: r15's 4 items applied — (1) discrimination test now uses 3e-5 band-local drift with absmax in an unaffected band and ASSERTS the old global gate accepts while the per-band gate refuses; (2) ULP test uses np.nextafter; (3) two stale comment lines rewritten; (4) residual doc covers nonfinite position/kind changes. All 8 tests pass. Read the diff (git diff HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py). End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
codex
I’ll review the requested diff for correctness and regressions, with particular attention to the revised discrimination and ULP tests.
exec
/usr/bin/bash -lc 'git diff --check HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py && git diff --unified=80 HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 274ms:
diff --git a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
index 0ea2c649a..406f4a76e 100644
--- a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
+++ b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
@@ -1,99 +1,100 @@
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
 
-* GRID = **replicated-stacked, indexed** (Structure A). The ``N`` band
-  geometries are built host-side (``build_band_grids``), their ARRAY fields are
-  ``jnp.stack``-ed into a replicated pytree, and the in-``shard_map`` body picks
-  its own band by ``jax.lax.axis_index("lat")``. The grid is 2-D/small so
-  replication is cheap, and this AVOIDS staggered-sharding the grid (the v-row
-  is ``n_lat+1``, coprime with ``n_lat`` for ``N>1``). The
+* GRID = **band-stacked, P("lat")-SHARDED** (Structure A, sharded since
+  #1370-iii). The ``N`` band geometries are built host-side
+  (``build_band_grids``), their ARRAY fields are ``jnp.stack``-ed over a
+  leading band axis, sharded ``P("lat")`` so each device holds ONLY its own
+  band's slab, and the in-``shard_map`` body reads its local slab at
+  ``[0]``. This AVOIDS staggered-sharding the grid itself (the v-row is
+  ``n_lat+1``, coprime with ``n_lat`` for ``N>1``). The
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
@@ -141,160 +142,225 @@ def _step_body(model, state, dt, *, grid, vertex_mask,
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
 
+def geom_band_fingerprint(host, n_bands):
+    """Low-memory cross-process fingerprint of a band-STACKED geometry field.
+
+    ``host`` has leading axis ``n_bands`` (the per-device band stack). The
+    fingerprint is PER BAND (codex 2026-08-03 r14: whole-array moments let a
+    band-local drift hide in the global sum once each process's own bytes
+    become live computation inputs):
+
+    * exact dtypes (int/bool/uint — masks, index tables): one positional
+      48-bit byte digest per band (a permutation or two-cell flip within a
+      band cannot cancel);
+    * float dtypes: per-band ``[sum, sum_of_squares, absmax]`` of the finite
+      entries in float64, plus per-band non-finite counts in ``struct``.
+      DOCUMENTED RESIDUALS: a within-band float change preserving all
+      three moments to rtol is not detected, and non-finite entries that
+      change POSITION or kind (nan vs inf) with an unchanged per-band
+      count also pass; band grids are analytic in lat/lon, so any real
+      inconsistency moves the moments.
+
+    Returns ``(struct, vals, is_exact)`` as float64 arrays safe for
+    ``process_allgather``.
+    """
+    import numpy as _np
+
+    host = _np.asarray(host)
+    assert host.shape[0] == n_bands, (host.shape, n_bands)
+    is_exact = host.dtype.kind in "biu"
+    struct = [float(host.ndim), *map(float, host.shape),
+              float(_np.dtype(host.dtype).num)]
+    if is_exact:
+        vals = _np.array([_content_hash48(host[b]) for b in range(n_bands)],
+                         dtype=_np.float64)
+    else:
+        per_band = []
+        for b in range(n_bands):
+            flat = host[b].ravel()
+            finite = flat[_np.isfinite(flat)]
+            f64 = finite.astype(_np.float64)
+            struct.append(float(flat.size - finite.size))
+            per_band.extend([
+                float(f64.sum()) if f64.size else 0.0,
+                float((f64 * f64).sum()) if f64.size else 0.0,
+                float(_np.abs(f64).max()) if f64.size else 0.0,
+            ])
+        vals = _np.array(per_band, dtype=_np.float64)
+    return _np.array(struct, dtype=_np.float64), vals, is_exact
+
+
+def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=1e-5):
+    """True iff every process's fingerprint matches process 0's.
+
+    ``g_struct``/``g_vals`` are the ``process_allgather``-ed outputs of
+    :func:`geom_band_fingerprint` (leading axis = process). Structure and
+    exact-dtype digests compare EXACTLY; float moments to ``rtol``.
+    """
+    import numpy as _np
+
+    struct_ok = bool(_np.all(g_struct == g_struct[0]))
+    if is_exact:
+        vals_ok = bool(_np.all(g_vals == g_vals[0]))
+    else:
+        vals_ok = bool(_np.allclose(g_vals, g_vals[0], rtol=rtol, atol=0.0))
+    return struct_ok and vals_ok
+
+
 
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
     vm = getattr(state, "v_mask", None)
     if vm is not None:
         import numpy as _np
 
         if _np.asarray(vm.data)[-1].any():
             raise ValueError(
                 "shard_state_latlon: the state's TOP v-face row is LIVE "
                 "(v_mask[-1] has ocean faces) — the lat-band v-carrier "
                 "drops that row and reconstructs it as the pole/cap wall "
                 "zero, which would silently delete seam velocities. "
@@ -424,283 +490,277 @@ def append_vface_wall_row(v_lower):
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
-    ARRAY fields into a replicated pytree, and index by
-    ``jax.lax.axis_index("lat")`` in the ``shard_map`` body (the geometry SCALAR
-    fields stay static — see the module docstring).  ``dt`` is a TRACED,
+    ARRAY fields over a leading band axis SHARDED ``P("lat")`` (each device
+    holds only its own slab; the ``shard_map`` body reads it at ``[0]``; the
+    geometry SCALAR fields stay static — see the module docstring).  ``dt`` is a TRACED,
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
 
-    # --- host-side band geometries + vertex masks (replicated, indexed in-body) ---
+    # --- host-side band geometries + vertex masks (band-stacked, P("lat")-sharded) ---
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
-    # — same stack, different placement; the process-0 broadcast +
-    # divergence guard below runs on HOST values and is placement-blind.
+    # — same stack, different placement; the cross-process divergence
+    # guard below runs on HOST values and is placement-blind.
     rep = NamedSharding(mesh, P("lat"))
 
     def _replicated_put(arr, name):
-        # Multicontroller: a P() (fully-replicated) device_put ASSERTS the
-        # value is bit-identical on every process. The band-geometry arrays
-        # are (re)computed per process and can differ in their last ULPs
-        # (per-process XLA autotuning on device-derived grid fields), which
-        # trips that assert at larger sizes (job 26450848: LL576 np=4,
-        # area-scale fields differing at 1e-7 relative). Broadcast process
-        # 0's bytes so every controller puts the SAME replicated value.
+        # (Name kept for history; since 2026-08-03 this is a SHARDED stack
+        # put.) The band-geometry arrays are (re)computed per process and
+        # can differ in their last ULPs (per-process XLA autotuning on
+        # device-derived grid fields) — the fully-replicated-put era
+        # broadcast process 0's bytes to sidestep the P() bit-identity
+        # assert (job 26450848). With the #1370-iii P("lat") sharding each
+        # process's devices consume ONLY its own band rows, so the
+        # broadcast became both unnecessary and, at nd>=96, fatal (its
+        # psum program is nd x the stack — see the note at the put below).
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
 
-            flat = host.ravel()
-            # Integer/bool arrays (masks, index tables) are exact data, not
-            # autotuned arithmetic: fingerprint their BYTES so a positional
-            # difference is caught. Moment-only compares are blind to a
-            # permutation — a bool mask's (sum, sumsq, absmax) is identical
-            # for every arrangement with the same true-count (codex round-5).
-            # A mask that genuinely differs across processes means different
-            # wet domains = different physics: refusing is the correct
-            # outcome, not a false alarm.
-            is_exact = host.dtype.kind in "biu"
-            struct = np.array(
-                [float(host.ndim), *map(float, host.shape),
-                 float(np.dtype(host.dtype).num)], dtype=np.float64)
-            if is_exact:
-                vals = np.array([_content_hash48(host)], dtype=np.float64)
-            else:
-                finite = flat[np.isfinite(flat)]
-                f64 = finite.astype(np.float64)
-                struct = np.concatenate(
-                    [struct, [float(flat.size - finite.size)]])
-                vals = np.array(
-                    [float(f64.sum()) if f64.size else 0.0,
-                     float((f64 * f64).sum()) if f64.size else 0.0,
-                     float(np.abs(f64).max()) if f64.size else 0.0],
-                    dtype=np.float64)
+            # PER-BAND fingerprints (module-level, unit-tested): exact
+            # dtypes hash positionally per band; floats compare per-band
+            # moments to rtol 1e-5 — bounds each band's drift instead of
+            # letting it hide in a whole-array sum, since each process's
+            # own bytes are now the live inputs for the bands it owns
+            # (codex r14). A mask that genuinely differs across processes
+            # means different wet domains = different physics: refusing is
+            # correct, not a false alarm.
+            struct, vals, is_exact = geom_band_fingerprint(
+                host, host.shape[0])
             g_struct = multihost_utils.process_allgather(struct)
             g_vals = multihost_utils.process_allgather(vals)
-            struct_ok = bool(np.all(g_struct == g_struct[0]))
-            if is_exact:
-                vals_ok = bool(np.all(g_vals == g_vals[0]))
-            else:
-                vals_ok = bool(np.allclose(g_vals, g_vals[0],
-                                           rtol=1e-5, atol=0.0))
-            if not (struct_ok and vals_ok):
+            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
                 raise RuntimeError(
                     f"make_sharded_ocean_step: band-geometry field {name!r} "
-                    f"DIVERGES across processes (struct_ok={struct_ok}, "
-                    f"vals_ok={vals_ok}, exact_dtype={is_exact}, "
+                    f"DIVERGES across processes (exact_dtype={is_exact}, "
                     f"gathered={g_vals.tolist()}) — a real config/grid "
                     f"inconsistency, not autotune noise; refusing to "
-                    f"broadcast process 0 over it.")
-            host = multihost_utils.broadcast_one_to_all(host)
-        return jax.device_put(jnp.asarray(host), rep)
+                    f"shard it.")
+            # NO broadcast_one_to_all here (removed 2026-08-03): its psum
+            # program is [n_processes, stack] in / P() fully-replicated out,
+            # so its logical arg bytes are nd x the global stack — 82.4 GB
+            # at nd=96 and 109.6 GB at nd=128 for one 3-D field, the
+            # near-linear-in-nd wall that killed oc @96/@128 (jobs
+            # 26642771/26636762) while @64 sat just under XLA's 63.8 GB
+            # limit.  The target sharding is P("lat"): each process's
+            # devices consume ONLY its own band rows, so cross-process
+            # byte-identity of non-owned rows is irrelevant, and REAL
+            # divergence is already refused by the fingerprint gate above.
+            # make_array_from_callback hands each process exactly its
+            # addressable slabs — the same #1100 pattern as the state
+            # build — with no global-sized collective program at all.
+        host_np = np.asarray(host)
+        return jax.make_array_from_callback(
+            host_np.shape, rep, lambda idx: host_np[idx])
 
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

exec
/usr/bin/bash -lc "git diff --stat HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py
git diff --unified=100 HEAD -- tests/ocean/unit/test_sharded_geom_fingerprint.py
git diff --unified=35 HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py | rg -n -C 10 'geom_band_fingerprint|band_fingerprints_agree|_replicated_put|make_array_from_callback|DIVERGES'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 272ms:
 .../legoesm/ocean/dynamics/sharded_ocean_step.py   | 172 ++++++++++++++-------
 1 file changed, 116 insertions(+), 56 deletions(-)
93-     under 2**53 so it survives the float64 ``process_allgather`` payload
94-     exactly. Not cryptographic — collision-resistance at 2**-48 is far
95-     beyond the ~10 setup-time comparisons this guard makes.
96-     """
97-     import hashlib
98- 
99-     a = np.ascontiguousarray(arr)
100-     h = hashlib.blake2b(a.tobytes(), digest_size=6)
101-     return float(int.from_bytes(h.digest(), "big"))
102- 
103:+def geom_band_fingerprint(host, n_bands):
104-+    """Low-memory cross-process fingerprint of a band-STACKED geometry field.
105-+
106-+    ``host`` has leading axis ``n_bands`` (the per-device band stack). The
107-+    fingerprint is PER BAND (codex 2026-08-03 r14: whole-array moments let a
108-+    band-local drift hide in the global sum once each process's own bytes
109-+    become live computation inputs):
110-+
111-+    * exact dtypes (int/bool/uint — masks, index tables): one positional
112-+      48-bit byte digest per band (a permutation or two-cell flip within a
113-+      band cannot cancel);
--
141-+            struct.append(float(flat.size - finite.size))
142-+            per_band.extend([
143-+                float(f64.sum()) if f64.size else 0.0,
144-+                float((f64 * f64).sum()) if f64.size else 0.0,
145-+                float(_np.abs(f64).max()) if f64.size else 0.0,
146-+            ])
147-+        vals = _np.array(per_band, dtype=_np.float64)
148-+    return _np.array(struct, dtype=_np.float64), vals, is_exact
149-+
150-+
151:+def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=1e-5):
152-+    """True iff every process's fingerprint matches process 0's.
153-+
154-+    ``g_struct``/``g_vals`` are the ``process_allgather``-ed outputs of
155:+    :func:`geom_band_fingerprint` (leading axis = process). Structure and
156-+    exact-dtype digests compare EXACTLY; float moments to ``rtol``.
157-+    """
158-+    import numpy as _np
159-+
160-+    struct_ok = bool(_np.all(g_struct == g_struct[0]))
161-+    if is_exact:
162-+        vals_ok = bool(_np.all(g_vals == g_vals[0]))
163-+    else:
164-+        vals_ok = bool(_np.allclose(g_vals, g_vals[0], rtol=rtol, atol=0.0))
165-+    return struct_ok and vals_ok
--
289-     # this was one of the residual ~1.4-1.7 global-field-equivalents of
290-     # per-device residency left after the host-side-build fix (probe
291-     # 26524423). The leading axis has length n_dev, so P("lat") divides it
292-     # exactly; the body indexes its local slab at [0]. Values are unchanged
293--    # — same stack, different placement; the process-0 broadcast +
294--    # divergence guard below runs on HOST values and is placement-blind.
295-+    # — same stack, different placement; the cross-process divergence
296-+    # guard below runs on HOST values and is placement-blind.
297-     rep = NamedSharding(mesh, P("lat"))
298- 
299:     def _replicated_put(arr, name):
300--        # Multicontroller: a P() (fully-replicated) device_put ASSERTS the
301--        # value is bit-identical on every process. The band-geometry arrays
302--        # are (re)computed per process and can differ in their last ULPs
303--        # (per-process XLA autotuning on device-derived grid fields), which
304--        # trips that assert at larger sizes (job 26450848: LL576 np=4,
305--        # area-scale fields differing at 1e-7 relative). Broadcast process
306--        # 0's bytes so every controller puts the SAME replicated value.
307-+        # (Name kept for history; since 2026-08-03 this is a SHARDED stack
308-+        # put.) The band-geometry arrays are (re)computed per process and
309-+        # can differ in their last ULPs (per-process XLA autotuning on
--
357--                     float(np.abs(f64).max()) if f64.size else 0.0],
358--                    dtype=np.float64)
359-+            # PER-BAND fingerprints (module-level, unit-tested): exact
360-+            # dtypes hash positionally per band; floats compare per-band
361-+            # moments to rtol 1e-5 — bounds each band's drift instead of
362-+            # letting it hide in a whole-array sum, since each process's
363-+            # own bytes are now the live inputs for the bands it owns
364-+            # (codex r14). A mask that genuinely differs across processes
365-+            # means different wet domains = different physics: refusing is
366-+            # correct, not a false alarm.
367:+            struct, vals, is_exact = geom_band_fingerprint(
368-+                host, host.shape[0])
369-             g_struct = multihost_utils.process_allgather(struct)
370-             g_vals = multihost_utils.process_allgather(vals)
371--            struct_ok = bool(np.all(g_struct == g_struct[0]))
372--            if is_exact:
373--                vals_ok = bool(np.all(g_vals == g_vals[0]))
374--            else:
375--                vals_ok = bool(np.allclose(g_vals, g_vals[0],
376--                                           rtol=1e-5, atol=0.0))
377--            if not (struct_ok and vals_ok):
378:+            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
379-                 raise RuntimeError(
380-                     f"make_sharded_ocean_step: band-geometry field {name!r} "
381:-                    f"DIVERGES across processes (struct_ok={struct_ok}, "
382--                    f"vals_ok={vals_ok}, exact_dtype={is_exact}, "
383:+                    f"DIVERGES across processes (exact_dtype={is_exact}, "
384-                     f"gathered={g_vals.tolist()}) — a real config/grid "
385-                     f"inconsistency, not autotune noise; refusing to "
386--                    f"broadcast process 0 over it.")
387--            host = multihost_utils.broadcast_one_to_all(host)
388--        return jax.device_put(jnp.asarray(host), rep)
389-+                    f"shard it.")
390-+            # NO broadcast_one_to_all here (removed 2026-08-03): its psum
391-+            # program is [n_processes, stack] in / P() fully-replicated out,
392-+            # so its logical arg bytes are nd x the global stack — 82.4 GB
393-+            # at nd=96 and 109.6 GB at nd=128 for one 3-D field, the
394-+            # near-linear-in-nd wall that killed oc @96/@128 (jobs
395-+            # 26642771/26636762) while @64 sat just under XLA's 63.8 GB
396-+            # limit.  The target sharding is P("lat"): each process's
397-+            # devices consume ONLY its own band rows, so cross-process
398-+            # byte-identity of non-owned rows is irrelevant, and REAL
399-+            # divergence is already refused by the fingerprint gate above.
400:+            # make_array_from_callback hands each process exactly its
401-+            # addressable slabs — the same #1100 pattern as the state
402-+            # build — with no global-sized collective program at all.
403-+        host_np = np.asarray(host)
404:+        return jax.make_array_from_callback(
405-+            host_np.shape, rep, lambda idx: host_np[idx])
406- 
407-     if jax.process_count() > 1:
408-         # Schema gate FIRST (one fixed-shape collective every process
409-         # reaches): a process-dependent field list or a mixed
410-         # jax_enable_x64 setting would otherwise desynchronize the
411-         # per-field gathers below instead of failing with a clear message.
412-         from jax.experimental import multihost_utils as _mhu
413- 
414-         _g = _mhu.process_allgather(
415-             _schema_fingerprint(list(array_field_names), n_dev))
416-         if not bool(np.all(_g == _g[0])):
417-             raise RuntimeError(
418-                 "make_sharded_ocean_step: the band-geometry SCHEMA differs "
419-                 "across processes (field list / x64 setting / device count "
420-                 f"— gathered {_g.tolist()}). Fix the per-process config "
421-                 "before sharding; the per-field checks below assume one "
422-                 "schema.")
423- 
424-     geom_stacks = {
425:         name: _replicated_put(
426-             jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
427-                       axis=0), name)
428-         for name in array_field_names
429-     }
430:     vmask_stack = _replicated_put(
431-         jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0),
432-         "vertex_mask")
433- 
434-     # Static perms for the v north-boundary-row ppermute (band r receives band
435-     # r+1's v_lower[0] = global v[e]; north band non-target receives 0).
436-     perm_north, _perm_south = latlon_band_perms(n_dev)
437- 
438-     def _body(state_local, forcing_local, geom_stacks_local,
439-               vmask_stack_local, dt):
440-         fw_local, sf_local, sponge_local, t_s_local = forcing_local

exec
/usr/bin/bash -lc "git status --short
git ls-files --error-unmatch tests/ocean/unit/test_sharded_geom_fingerprint.py 2>&1 || true
if [ -f tests/ocean/unit/test_sharded_geom_fingerprint.py ]; then nl -ba tests/ocean/unit/test_sharded_geom_fingerprint.py; fi
git show HEAD:tests/ocean/unit/test_sharded_geom_fingerprint.py >/dev/null 2>&1; printf 'head_test_exit=%s\\n' \""'$?"' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 456ms:
 M packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
 M scripts/plot/plot_scaling_paper_figure.py
?? .physics-validator/scaling_campaign/codex_bcast_fix_r14_2026-08-03.md
?? .physics-validator/scaling_campaign/codex_bcast_fix_r15_2026-08-03.md
?? .physics-validator/scaling_campaign/codex_bcast_fix_r16_2026-08-03.md
?? tests/ocean/unit/test_sharded_geom_fingerprint.py
error: pathspec 'tests/ocean/unit/test_sharded_geom_fingerprint.py' did not match any file(s) known to git
Did you forget to 'git add'?
     1	"""Unit tests for the band-geometry cross-process fingerprint gate.
     2	
     3	The gate decides whether ``make_sharded_ocean_step`` accepts per-process
     4	band-geometry stacks without the (removed, nd-linear-cost) process-0
     5	broadcast — see the 2026-08-03 fix note at the sharded put. These tests
     6	pin the gate's discrimination properties single-process (the
     7	multicontroller allgather wiring is exercised by the distributed suite).
     8	"""
     9	import numpy as np
    10	import pytest
    11	
    12	from legoesm.ocean.dynamics.sharded_ocean_step import (
    13	    band_fingerprints_agree,
    14	    geom_band_fingerprint,
    15	)
    16	
    17	N_BANDS = 4
    18	SHAPE = (N_BANDS, 6, 8)
    19	
    20	
    21	def _gather(*hosts):
    22	    """Simulate process_allgather over per-process fingerprints."""
    23	    fps = [geom_band_fingerprint(h, N_BANDS) for h in hosts]
    24	    exacts = {fp[2] for fp in fps}
    25	    assert len(exacts) == 1
    26	    g_struct = np.stack([fp[0] for fp in fps])
    27	    g_vals = np.stack([fp[1] for fp in fps])
    28	    return g_struct, g_vals, fps[0][2]
    29	
    30	
    31	def test_identical_float_stacks_agree():
    32	    rng = np.random.default_rng(0)
    33	    a = rng.normal(size=SHAPE).astype(np.float32)
    34	    assert band_fingerprints_agree(*_gather(a, a.copy()))
    35	
    36	
    37	def test_ulp_scale_drift_agrees():
    38	    rng = np.random.default_rng(1)
    39	    a = rng.normal(size=SHAPE).astype(np.float64) + 10.0
    40	    b = np.nextafter(a, np.inf)  # a TRUE 1-ULP elementwise drift
    41	    assert band_fingerprints_agree(*_gather(a, b))
    42	
    43	
    44	def test_band_local_drift_refused_where_global_gate_passed():
    45	    # THE r14 discrimination case: a band-local drift SMALL enough that the
    46	    # old whole-array gate (sum/sumsq/absmax at rtol 1e-5) accepts it, with
    47	    # the global absmax held by an UNAFFECTED band — the per-band gate must
    48	    # still refuse.
    49	    rng = np.random.default_rng(2)
    50	    a = rng.normal(size=SHAPE).astype(np.float64) + 10.0
    51	    a[0, 0, 0] = 500.0          # absmax lives in band 0
    52	    b = a.copy()
    53	    b[2] *= 1.0 + 3e-5          # one band drifts; global sums move ~7e-6 rel
    54	
    55	    def _global_moments(x):
    56	        f = x.ravel().astype(np.float64)
    57	        return np.array([f.sum(), (f * f).sum(), np.abs(f).max()])
    58	
    59	    # the OLD global gate would have ACCEPTED this pair...
    60	    assert np.allclose(_global_moments(a), _global_moments(b),
    61	                       rtol=1e-5, atol=0.0)
    62	    # ...the per-band gate refuses it.
    63	    assert not band_fingerprints_agree(*_gather(a, b))
    64	
    65	
    66	def test_exact_dtype_permutation_refused():
    67	    # Moment fingerprints are blind to permutations; the positional
    68	    # per-band byte digest must not be.
    69	    a = np.zeros(SHAPE, dtype=np.int32)
    70	    a[1, 2, 3] = 1
    71	    b = np.zeros_like(a)
    72	    b[1, 3, 2] = 1  # same count, different position, same band
    73	    assert not band_fingerprints_agree(*_gather(a, b))
    74	
    75	
    76	def test_bool_mask_two_cell_flip_refused():
    77	    a = np.zeros(SHAPE, dtype=bool)
    78	    a[0, 0, 0] = True
    79	    b = a.copy()
    80	    b[0, 0, 0] = False
    81	    b[0, 5, 7] = True  # true-count preserved
    82	    assert not band_fingerprints_agree(*_gather(a, b))
    83	
    84	
    85	def test_nonfinite_count_mismatch_refused():
    86	    a = np.ones(SHAPE, dtype=np.float32)
    87	    b = a.copy()
    88	    b[3, 0, 0] = np.nan  # struct carries per-band non-finite counts
    89	    assert not band_fingerprints_agree(*_gather(a, b))
    90	
    91	
    92	def test_shape_mismatch_refused():
    93	    a = np.ones(SHAPE, dtype=np.float32)
    94	    b = np.ones((N_BANDS, 6, 9), dtype=np.float32)
    95	    fa = geom_band_fingerprint(a, N_BANDS)
    96	    fb = geom_band_fingerprint(b, N_BANDS)
    97	    # Different shapes -> different struct lengths; the agree helper is
    98	    # only called on stackable gathers, so assert the structs differ.
    99	    assert fa[0].shape != fb[0].shape or not np.array_equal(fa[0], fb[0])
   100	
   101	
   102	def test_wrong_leading_axis_asserts():
   103	    with pytest.raises(AssertionError):
   104	        geom_band_fingerprint(np.ones((3, 2)), N_BANDS)
head_test_exit=128

exec
/usr/bin/bash -lc "nl -ba packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py | sed -n '205,315p'
nl -ba packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py | sed -n '625,760p'
rg -n \"make_array_from_callback|geom_band_fingerprint|band_fingerprints_agree|_replicated_put\" packages tests -g '*.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 351ms:
   205	
   206	def _content_hash48(arr) -> float:
   207	    """48-bit content digest of ``arr``'s bytes, exactly representable in f64.
   208	
   209	    Used to compare EXACT-dtype arrays (masks, index tables) across
   210	    processes: unlike moment fingerprints, a byte digest is positional, so a
   211	    permutation or a two-cell flip cannot cancel. 48 bits keeps the value
   212	    under 2**53 so it survives the float64 ``process_allgather`` payload
   213	    exactly. Not cryptographic — collision-resistance at 2**-48 is far
   214	    beyond the ~10 setup-time comparisons this guard makes.
   215	    """
   216	    import hashlib
   217	
   218	    a = np.ascontiguousarray(arr)
   219	    h = hashlib.blake2b(a.tobytes(), digest_size=6)
   220	    return float(int.from_bytes(h.digest(), "big"))
   221	
   222	def geom_band_fingerprint(host, n_bands):
   223	    """Low-memory cross-process fingerprint of a band-STACKED geometry field.
   224	
   225	    ``host`` has leading axis ``n_bands`` (the per-device band stack). The
   226	    fingerprint is PER BAND (codex 2026-08-03 r14: whole-array moments let a
   227	    band-local drift hide in the global sum once each process's own bytes
   228	    become live computation inputs):
   229	
   230	    * exact dtypes (int/bool/uint — masks, index tables): one positional
   231	      48-bit byte digest per band (a permutation or two-cell flip within a
   232	      band cannot cancel);
   233	    * float dtypes: per-band ``[sum, sum_of_squares, absmax]`` of the finite
   234	      entries in float64, plus per-band non-finite counts in ``struct``.
   235	      DOCUMENTED RESIDUALS: a within-band float change preserving all
   236	      three moments to rtol is not detected, and non-finite entries that
   237	      change POSITION or kind (nan vs inf) with an unchanged per-band
   238	      count also pass; band grids are analytic in lat/lon, so any real
   239	      inconsistency moves the moments.
   240	
   241	    Returns ``(struct, vals, is_exact)`` as float64 arrays safe for
   242	    ``process_allgather``.
   243	    """
   244	    import numpy as _np
   245	
   246	    host = _np.asarray(host)
   247	    assert host.shape[0] == n_bands, (host.shape, n_bands)
   248	    is_exact = host.dtype.kind in "biu"
   249	    struct = [float(host.ndim), *map(float, host.shape),
   250	              float(_np.dtype(host.dtype).num)]
   251	    if is_exact:
   252	        vals = _np.array([_content_hash48(host[b]) for b in range(n_bands)],
   253	                         dtype=_np.float64)
   254	    else:
   255	        per_band = []
   256	        for b in range(n_bands):
   257	            flat = host[b].ravel()
   258	            finite = flat[_np.isfinite(flat)]
   259	            f64 = finite.astype(_np.float64)
   260	            struct.append(float(flat.size - finite.size))
   261	            per_band.extend([
   262	                float(f64.sum()) if f64.size else 0.0,
   263	                float((f64 * f64).sum()) if f64.size else 0.0,
   264	                float(_np.abs(f64).max()) if f64.size else 0.0,
   265	            ])
   266	        vals = _np.array(per_band, dtype=_np.float64)
   267	    return _np.array(struct, dtype=_np.float64), vals, is_exact
   268	
   269	
   270	def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=1e-5):
   271	    """True iff every process's fingerprint matches process 0's.
   272	
   273	    ``g_struct``/``g_vals`` are the ``process_allgather``-ed outputs of
   274	    :func:`geom_band_fingerprint` (leading axis = process). Structure and
   275	    exact-dtype digests compare EXACTLY; float moments to ``rtol``.
   276	    """
   277	    import numpy as _np
   278	
   279	    struct_ok = bool(_np.all(g_struct == g_struct[0]))
   280	    if is_exact:
   281	        vals_ok = bool(_np.all(g_vals == g_vals[0]))
   282	    else:
   283	        vals_ok = bool(_np.allclose(g_vals, g_vals[0], rtol=rtol, atol=0.0))
   284	    return struct_ok and vals_ok
   285	
   286	
   287	
   288	def _schema_fingerprint(names, n_dev) -> np.ndarray:
   289	    """Fixed-shape schema digest: field-name list, count, x64 flag, n_dev.
   290	
   291	    Gathered ONCE before the per-field loop so a process-dependent field
   292	    selection is caught by a collective every process reaches, instead of
   293	    desynchronizing the per-field gathers (codex round-5 findings 3/4).
   294	    """
   295	    import hashlib
   296	
   297	    joined = ",".join(names).encode()
   298	    digest = float(int.from_bytes(
   299	        hashlib.blake2b(joined, digest_size=6).digest(), "big"))
   300	    return np.array(
   301	        [float(len(names)), digest, float(bool(jax.config.jax_enable_x64)),
   302	         float(n_dev)], dtype=np.float64)
   303	
   304	
   305	def _build_band_vertex_masks(model, n_dev):
   306	    """Per-band vertex masks (n_lat/N+1, n_lon+1): SLICE the model's primed GLOBAL
   307	    vertex mask ``[s : e+1]`` per band (the v/q-row stagger ``slice_cgrid_geometry_
   308	    to_band`` uses for ``area_q``).
   309	
   310	    The slice — NOT a per-band ``compute_vertex_mask`` — is what makes the
   311	    band-cut rows correct.  The global ``model._vertex_mask`` already encoded the
   312	    four-cell product at EVERY vertex row including the interior cut rows (it saw
   313	    the neighbour-band cells), so band ``r``'s vertex rows ``[r*nl : r*nl+nl+1]``
   314	    are exact.  A per-band ``compute_vertex_mask`` would instead pad the band's
   315	    cell mask with a POLE WALL at the cut (``pad_with_pole_bc_lat`` has no SPMD
   625	        # (Name kept for history; since 2026-08-03 this is a SHARDED stack
   626	        # put.) The band-geometry arrays are (re)computed per process and
   627	        # can differ in their last ULPs (per-process XLA autotuning on
   628	        # device-derived grid fields) — the fully-replicated-put era
   629	        # broadcast process 0's bytes to sidestep the P() bit-identity
   630	        # assert (job 26450848). With the #1370-iii P("lat") sharding each
   631	        # process's devices consume ONLY its own band rows, so the
   632	        # broadcast became both unnecessary and, at nd>=96, fatal (its
   633	        # psum program is nd x the stack — see the note at the put below).
   634	        # GUARD (codex round-3): process 0 must not silently mask REAL
   635	        # cross-process divergence. Compare an allgathered fingerprint:
   636	        # structural entries exactly; value entries EXACTLY for integer/bool
   637	        # arrays (masks are comparison results — bit-reproducible, and an
   638	        # exact compare is the only way to catch a two-cell flip that cancels
   639	        # in the sum, codex round-4) and to rtol 1e-5 for float arrays (only
   640	        # ULP autotune drift is expected there; quantize-then-assert-equal
   641	        # false-positived on a rounding boundary, job 26453240).
   642	        # NO DEADLOCK RISK: every process fingerprints the same fields in the
   643	        # same order and derives the verdict from the SAME gathered array, so
   644	        # the refusal is symmetric — all raise or none.
   645	        # Residual (documented): a float-geometry divergence preserving sum,
   646	        # sum-of-squares AND absmax to 1e-5 is not detected; band grids are
   647	        # analytic in lat/lon, so any real inconsistency moves those moments.
   648	        host = np.asarray(arr)
   649	        if jax.process_count() > 1:
   650	            from jax.experimental import multihost_utils
   651	
   652	            # PER-BAND fingerprints (module-level, unit-tested): exact
   653	            # dtypes hash positionally per band; floats compare per-band
   654	            # moments to rtol 1e-5 — bounds each band's drift instead of
   655	            # letting it hide in a whole-array sum, since each process's
   656	            # own bytes are now the live inputs for the bands it owns
   657	            # (codex r14). A mask that genuinely differs across processes
   658	            # means different wet domains = different physics: refusing is
   659	            # correct, not a false alarm.
   660	            struct, vals, is_exact = geom_band_fingerprint(
   661	                host, host.shape[0])
   662	            g_struct = multihost_utils.process_allgather(struct)
   663	            g_vals = multihost_utils.process_allgather(vals)
   664	            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
   665	                raise RuntimeError(
   666	                    f"make_sharded_ocean_step: band-geometry field {name!r} "
   667	                    f"DIVERGES across processes (exact_dtype={is_exact}, "
   668	                    f"gathered={g_vals.tolist()}) — a real config/grid "
   669	                    f"inconsistency, not autotune noise; refusing to "
   670	                    f"shard it.")
   671	            # NO broadcast_one_to_all here (removed 2026-08-03): its psum
   672	            # program is [n_processes, stack] in / P() fully-replicated out,
   673	            # so its logical arg bytes are nd x the global stack — 82.4 GB
   674	            # at nd=96 and 109.6 GB at nd=128 for one 3-D field, the
   675	            # near-linear-in-nd wall that killed oc @96/@128 (jobs
   676	            # 26642771/26636762) while @64 sat just under XLA's 63.8 GB
   677	            # limit.  The target sharding is P("lat"): each process's
   678	            # devices consume ONLY its own band rows, so cross-process
   679	            # byte-identity of non-owned rows is irrelevant, and REAL
   680	            # divergence is already refused by the fingerprint gate above.
   681	            # make_array_from_callback hands each process exactly its
   682	            # addressable slabs — the same #1100 pattern as the state
   683	            # build — with no global-sized collective program at all.
   684	        host_np = np.asarray(host)
   685	        return jax.make_array_from_callback(
   686	            host_np.shape, rep, lambda idx: host_np[idx])
   687	
   688	    if jax.process_count() > 1:
   689	        # Schema gate FIRST (one fixed-shape collective every process
   690	        # reaches): a process-dependent field list or a mixed
   691	        # jax_enable_x64 setting would otherwise desynchronize the
   692	        # per-field gathers below instead of failing with a clear message.
   693	        from jax.experimental import multihost_utils as _mhu
   694	
   695	        _g = _mhu.process_allgather(
   696	            _schema_fingerprint(list(array_field_names), n_dev))
   697	        if not bool(np.all(_g == _g[0])):
   698	            raise RuntimeError(
   699	                "make_sharded_ocean_step: the band-geometry SCHEMA differs "
   700	                "across processes (field list / x64 setting / device count "
   701	                f"— gathered {_g.tolist()}). Fix the per-process config "
   702	                "before sharding; the per-field checks below assume one "
   703	                "schema.")
   704	
   705	    geom_stacks = {
   706	        name: _replicated_put(
   707	            jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
   708	                      axis=0), name)
   709	        for name in array_field_names
   710	    }
   711	    vmask_stack = _replicated_put(
   712	        jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0),
   713	        "vertex_mask")
   714	
   715	    # Static perms for the v north-boundary-row ppermute (band r receives band
   716	    # r+1's v_lower[0] = global v[e]; north band non-target receives 0).
   717	    perm_north, _perm_south = latlon_band_perms(n_dev)
   718	
   719	    def _body(state_local, forcing_local, geom_stacks_local,
   720	              vmask_stack_local, dt):
   721	        fw_local, sf_local, sponge_local, t_s_local = forcing_local
   722	        r = jax.lax.axis_index(axis)
   723	
   724	        # Rebuild this band's geometry. The stacks are SHARDED P("lat") on
   725	        # their leading band axis, so inside shard_map each device's local
   726	        # view is its own (1, ...) slab — index [0], NOT [r] (stage (iii);
   727	        # [r] was the replicated-stack indexing). Static scalars
   728	        # (n_lat_local, n_lon, radius, dlon, dlat, fold) come from the band
   729	        # template. The fold is inactive + identical on every band here.
   730	        geom_arrays = {name: geom_stacks_local[name][0]
   731	                       for name in array_field_names}
   732	        band_geom = template._replace(**geom_arrays)
   733	        band_vmask = vmask_stack_local[0]
   734	
   735	        # Reconstruct the band's nl+1 v-faces from the nl-row v_lower.  band r's
   736	        # north boundary row is band r+1's v_lower[0] (= global v[e]); the north
   737	        # band (no r+1) receives the pole-wall 0 via the ppermute non-target.
   738	        def _reconstruct_v(field):
   739	            if field is None:
   740	                return None
   741	            v_lower = field.data                      # (nl, n_lon[, nlev])
   742	            boundary = jax.lax.ppermute(
   743	                v_lower[0:1], axis, perm_north)        # band r+1's first row -> 0 at top
   744	            v_band = jnp.concatenate([v_lower, boundary], axis=0)  # nl+1 rows
   745	            return field.replace(data=v_band)
   746	
   747	        # Convert a returned nl+1 v back to the nl-row v_lower (drop the boundary
   748	        # row, owned by band r+1).
   749	        def _to_v_lower(field):
   750	            if field is None:
   751	                return None
   752	            return field.replace(data=field.data[:-1])
   753	
   754	        # Fused v-carrier reconstruction (scaling-M4): under the SAME
   755	        # trace-time switch as the pad aggregation, pack the boundary-row
   756	        # ppermutes of ALL staggered carriers (v + v_mask) into one
   757	        # collective per dtype group — value-identical (a bit-copy
   758	        # exchange; the flag flip re-keys the sharded_step cache below so
   759	        # a reused step object rebuilds).  Default OFF = the historical
   760	        # per-field ppermutes, byte-identical.
tests/test_cases/baroclinic_wave.py:781:    ``jax.make_array_from_callback``, whose callback runs only for the
tests/test_cases/baroclinic_wave.py:873:        return jax.make_array_from_callback(gshape, shard, cb)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:222:def geom_band_fingerprint(host, n_bands):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:270:def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=1e-5):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:274:    :func:`geom_band_fingerprint` (leading axis = process). Structure and
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:624:    def _replicated_put(arr, name):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:660:            struct, vals, is_exact = geom_band_fingerprint(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:664:            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:681:            # make_array_from_callback hands each process exactly its
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:685:        return jax.make_array_from_callback(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:706:        name: _replicated_put(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:711:    vmask_stack = _replicated_put(
tests/unit/test_parallel.py:595:    process; routes through make_array_from_callback under >1 process so the
tests/parallel/test_atm_latlon_bandlocal_build.py:11:(``make_array_from_callback`` per-process rows) leg of #1100 needs a real
packages/core/legoesm/parallel/mesh.py:759:    process's LOCAL copy via ``jax.make_array_from_callback`` instead —
packages/core/legoesm/parallel/mesh.py:774:        return jax.make_array_from_callback(
packages/core/legoesm/parallel/latlon_spmd.py:309:    each, rc=137 OOM). Instead ``jax.make_array_from_callback`` invokes the
packages/core/legoesm/parallel/latlon_spmd.py:316:    NOT differentiable: ``make_array_from_callback`` is a host construction API,
packages/core/legoesm/parallel/latlon_spmd.py:332:    return jax.make_array_from_callback(arr.shape, sharding, lambda idx: arr[idx])
tests/ocean/unit/test_sharded_geom_fingerprint.py:13:    band_fingerprints_agree,
tests/ocean/unit/test_sharded_geom_fingerprint.py:14:    geom_band_fingerprint,
tests/ocean/unit/test_sharded_geom_fingerprint.py:23:    fps = [geom_band_fingerprint(h, N_BANDS) for h in hosts]
tests/ocean/unit/test_sharded_geom_fingerprint.py:34:    assert band_fingerprints_agree(*_gather(a, a.copy()))
tests/ocean/unit/test_sharded_geom_fingerprint.py:41:    assert band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:63:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:73:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:82:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:89:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:95:    fa = geom_band_fingerprint(a, N_BANDS)
tests/ocean/unit/test_sharded_geom_fingerprint.py:96:    fb = geom_band_fingerprint(b, N_BANDS)
tests/ocean/unit/test_sharded_geom_fingerprint.py:104:        geom_band_fingerprint(np.ones((3, 2)), N_BANDS)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:91:    ``jax.make_array_from_callback``, whose callback is invoked only for the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:143:        return jax.make_array_from_callback(gshape, sharding, cb)

exec
/usr/bin/bash -lc "rg --files tests/ocean | head -80
rg -n \"testpaths|python_files|addopts|ruff|flake8\" pyproject.toml setup.cfg tox.ini pytest.ini .pre-commit-config.yaml 2>/dev/null || true
sed -n '730,790p' packages/core/legoesm/parallel/mesh.py
sed -n '295,342p' packages/core/legoesm/parallel/latlon_spmd.py
git diff --check HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 277ms:
tests/ocean/__init__.py
tests/ocean/validation/test_differentiability_ocean.py
tests/ocean/validation/test_step_fd_vs_ad.py
tests/ocean/run_ocean_all_grids_matrix.py
tests/ocean/validation/test_step_conservation.py
tests/ocean/validation/__init__.py
tests/ocean/validation/test_step_gradient_matrix.py
tests/ocean/distributed/test_ocean_mpi_tvd_parity.py
tests/ocean/distributed/conftest.py
tests/ocean/distributed/test_barotropic_pcg_mpi_parity.py
tests/ocean/distributed/test_mpas_ocean_stage_halo.py
tests/ocean/distributed/test_mpas_ocean_scatter.py
tests/ocean/distributed/test_barotropic_pcg_mpas_mpi.py
tests/ocean/distributed/test_ocean_mpi_tripole_step_parity.py
tests/ocean/distributed/test_ocean_mpi_wide_halo_parity.py
tests/ocean/distributed/__init__.py
tests/ocean/distributed/test_ocean_mpi_conservation.py
tests/ocean/fidelity/test_oceananigans_geostrophic_adjustment.py
tests/ocean/fidelity/test_mitgcm_advection_gyre_recipe.py
tests/ocean/fidelity/test_oceananigans_tendency_align.py
tests/ocean/fidelity/test_references.py
tests/ocean/unit/test_tke_advection.py
tests/ocean/unit/test_init_woa_interp.py
tests/ocean/unit/test_advection_weno.py
tests/ocean/unit/test_tke_prognostic.py
tests/ocean/unit/test_bathymetry.py
tests/ocean/unit/test_pgf_smc03_phase1.py
tests/ocean/unit/test_baroclinic_f32_lever.py
tests/ocean/unit/test_veros_global_flexible_recipe.py
tests/ocean/unit/test_rigid_lid.py
tests/ocean/unit/test_linear_free_surface.py
tests/ocean/unit/test_omip_spmd_south_pad.py
tests/ocean/unit/test_surface_forcing_implicit.py
tests/ocean/unit/test_tripole_vmix_cli.py
tests/ocean/unit/test_silvestri_jet_driver.py
tests/ocean/unit/test_mle.py
tests/ocean/unit/test_z_star_from_thicknesses.py
tests/ocean/unit/test_ab2_integrator.py
tests/ocean/unit/test_diag_omip_nemo_battery.py
tests/ocean/unit/test_rigid_lid_cache_jit.py
tests/ocean/unit/test_partial_cells_phase5.py
tests/ocean/unit/test_ab2_blend_helper.py
tests/ocean/unit/test_catke_integration.py
tests/ocean/unit/test_polar_cap_boost.py
tests/ocean/unit/test_ddm_merryfield_faithful.py
tests/ocean/unit/test_momentum_diagnostics_closure.py
tests/ocean/unit/test_veros_layout.py
tests/ocean/unit/test_nemo_state_bridge.py
tests/ocean/unit/test_partial_cells_phase2.py
tests/ocean/unit/test_tidal_simmons_faithful.py
tests/ocean/unit/test_inertia_gravity_wave.py
tests/ocean/unit/test_barotropic_local_clamp.py
tests/ocean/unit/test_tke_veros_dz_slots.py
tests/ocean/unit/test_silvestri_turbulence_2d.py
tests/ocean/unit/test_baroclinic_rk3.py
tests/ocean/unit/test_veros_acc_recipe.py
tests/ocean/unit/test_freesurface_helmholtz_adjoint_mpas.py
tests/ocean/unit/test_barotropic_slow_forcing_ab2.py
tests/ocean/unit/test_ocean_scm.py
tests/ocean/unit/test_latlon_regional_channel.py
tests/ocean/unit/test_leith.py
tests/ocean/unit/test_momentum_only_tendency.py
tests/ocean/unit/test_eke_signed_sources.py
tests/ocean/unit/test_latlon_cgrid_ocean.py
tests/ocean/unit/test_baroclinic_decomposition.py
tests/ocean/unit/test_freesurface_helmholtz_adjoint.py
tests/ocean/unit/test_dino_experiment.py
tests/ocean/unit/test_gm_bolus_through_fct.py
tests/ocean/unit/test_tke_integration.py
tests/ocean/unit/test_implicit_vmix_dzw_slot.py
tests/ocean/unit/test_surface_buoyancy_flux_shared.py
tests/ocean/unit/test_gm_resolution_function.py
tests/ocean/unit/test_flux_divergence_zero_flux.py
tests/ocean/unit/test_veros_acc_basic_recipe.py
tests/ocean/unit/test_static_rho_ref_mpas.py
tests/ocean/unit/test_concept_registry.py
tests/ocean/unit/test_mle_faithful.py
tests/ocean/unit/test_partial_cells_phase3a.py
tests/ocean/unit/test_barotropic_continuity_and_drag.py
tests/ocean/unit/test_eady_uniform_model_config.py
pyproject.toml:112:    "ruff",
pyproject.toml:161:[tool.ruff]
pyproject.toml:165:# conventions; audited upstream, excluded from ruff/audits here.
pyproject.toml:168:[tool.ruff.lint]
pyproject.toml:171:[tool.ruff.lint.per-file-ignores]
pyproject.toml:305:testpaths = ["tests"]
pyproject.toml:306:addopts = "-v --tb=short -m 'not slow'"
pyproject.toml:310:    # with e.g. `-m 'tier1 and not slow'`. The default addopts stays `not slow`
        backend=backend_name,
        is_distributed=False,
        tiling=(1, 1),
        grid_type="voronoi",
        voronoi_dims=(nCells, nEdges, nVertices),
    )
    _active_config = config
    logger.info(
        "legoESM: %d-device voronoi mesh on %s "
        "(nCells=%d, nEdges=%d, nVertices=%d)",
        n_dev, backend_name, nCells, nEdges, nVertices,
    )
    return config


# ==============================================================================
# Pytree sharding utilities
# ==============================================================================

def multiprocess_safe_device_put(leaf, sharding):
    """``jax.device_put`` that is safe under multi-controller SPMD.

    ``jax.device_put(x, sharding)`` with a sharding that spans processes
    ASSERTS the value is bit-identical on every process.  Per-process XLA
    autotuning can legitimately pick different kernels on different nodes,
    producing last-bit differences in host-precomputed inputs (first hit:
    the external-forcing leaves on the 2-node Levante cs_spmd receipt run,
    #693 job 26030677 — values identical to 8 significant digits, assert
    still trips).  Under >1 process, build the global array from each
    process's LOCAL copy via ``jax.make_array_from_callback`` instead —
    each process materializes only its addressable shards, no cross-process
    equality requirement, no communication.

    Single-process (and non-array leaves) delegate to plain
    ``jax.device_put`` — byte-identical behavior to before.
    Already-global (non-fully-addressable) leaves pass through unchanged.
    """
    if not isinstance(leaf, (jax.Array, jnp.ndarray)):
        return jax.device_put(leaf, sharding)
    if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable:
        return leaf  # already a global sharded array; nothing to place
    if jax.process_count() > 1:
        import numpy as np
        host = np.asarray(leaf)
        return jax.make_array_from_callback(
            host.shape, sharding, lambda idx: host[idx])
    return jax.device_put(leaf, sharding)


def shard_pytree(pytree, config: DeviceConfig):
    """Shard a pytree across devices according to the grid type.

    - **cubed_sphere**: face-first arrays ``(6, n, n, ...)`` sharded on face.
      With sub-face tiling, a 3D mesh ``(face, tile_i, tile_j)`` is used so
      both horizontal dimensions are sharded directly.
    - **latlon**: arrays ``(n_lat, n_lon, ...)`` sharded on lat dimension.
    - **spectral**: arrays ``(nlev, ...)`` sharded on level dimension.

    Parameters
    ----------
    pytree
def shard_leaf(arr, sharding, *, multiprocess: bool):
    """Scatter one full-global leaf onto ``sharding``'s mesh — the SCATTER
    primitive symmetric to :func:`replicate_leaf`, shared by the atm and ocean
    lat-band SPMD steps (``shard_state_atm_latlon`` / ``shard_state_latlon``).

    Single-process: plain ``jax.device_put`` (the historical path, unchanged and
    byte-identical).

    Multi-controller (``jax.process_count() > 1``, route-B ``jax.distributed``,
    a mesh spanning processes): a top-level ``jax.device_put`` of the FULL global
    array to a cross-process ``NamedSharding`` cannot place shards on peer
    processes' devices, so XLA falls back to an all-gather that (a) transiently
    materialises a second global copy per process and (b) is the collective seen
    to crash under full-node CPU packing (issue #1100: 128 procs × global-state
    each, rc=137 OOM). Instead ``jax.make_array_from_callback`` invokes the
    callback ONCE PER ADDRESSABLE SHARD with that shard's global index, and each
    process reads only its own shards out of the global array it already holds —
    no all-gather, no transient global replica. Using the per-shard index
    callback (not an enclosing [min,max) span) makes it correct for ANY
    device→process placement, including a non-contiguous/interleaved mesh order.

    NOT differentiable: ``make_array_from_callback`` is a host construction API,
    so (unlike the historical ``device_put``) a ``jax.grad``/``vjp`` cannot be
    taken THROUGH the multiprocess scatter. This is fine — the scatter is an
    init-time boundary (``scatter_to_local`` before the step loop, per the MPI
    pattern), never inside a differentiated loss; gradients w.r.t. params flow
    through the already-sharded state, not the scatter itself.

    Parameters
    ----------
    arr : the FULL global array, present on every process (host or device).
    sharding : NamedSharding — the lat-band ``P("lat", None, ...)`` target.
    multiprocess : pass ``jax.process_count() > 1`` (keyword-only so the branch
        is explicit at every call site, mirroring :func:`replicate_leaf`).
    """
    if not multiprocess:
        return jax.device_put(arr, sharding)
    return jax.make_array_from_callback(arr.shape, sharding, lambda idx: arr[idx])


def _pole_fold(rows, negate: bool):
    """Serial pole fold of ``rows`` (lat-mirror + 180 deg lon roll [+ sign]).

    ``rows`` is ``(halo, n_lon_padded[, nlev])`` already lon-wrapped; matches
    :func:`legoesm.grids.halo_latlon.fold_pole_rows` (``half = n_lon_pad // 2``,
    ``roll(rows[::-1], half, axis=lon)``)."""
    half = rows.shape[1] // 2
    sign = -1.0 if negate else 1.0

exec
/usr/bin/bash -lc "sed -n '150,190p' pyproject.toml
sed -n '295,325p' pyproject.toml
rg -n \"E302|D.*blank|\\[tool.ruff.lint\\]|select\\s*=\" pyproject.toml
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests/ocean/unit/test_sharded_geom_fingerprint.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 127 in 205ms:
# legoesm-atmosphere~=0.1.0 from an index that does not publish it yet.
[tool.uv.sources]
legoesm-core = { workspace = true }
legoesm-atmosphere = { workspace = true }
legoesm-ocean = { workspace = true }
legoesm-land = { workspace = true }
legoesm-ice = { workspace = true }
legoesm-coupler = { workspace = true }
legoesm-ml = { workspace = true }
legoesm-tools = { workspace = true }

[tool.ruff]
line-length = 100
target-version = "py311"
# Vendored 3rd-party CLM-ML-JAX backend (BSD-3) — not written to legoESM's lint
# conventions; audited upstream, excluded from ruff/audits here.
extend-exclude = ["packages/land/legoesm/land/canopy/clm_ml_backend"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.ruff.lint.per-file-ignores]
# FV3-native dycore ports (docs/.../fv3_native campaign): these modules are
# loop-faithful, INDEX-EXACT transcriptions of GFDL_atmos_cubed_sphere Fortran
# (sw_core/tp_core/a2b_edge/fv_grid_*).  The Fortran-metric locals are held in
# UPPER_SNAKE fort views (DXA, DEL6_V, SIN_SG, CSG, ...) to mirror the source
# 1:1 — renaming to snake_case would destroy the line-by-line cross-reference
# to the oracle (the higher-priority faithfulness goal).  `fort`/`fort1` are
# deliberately lower-case (they read as array *views*, not classes).  N803/N806
# = kernel args/vars, N801 = the fort view classes.
"packages/core/legoesm/core/fv3_native_sw_core.py" = ["N803", "N806", "N801"]
"packages/core/legoesm/core/fv3_native_d_sw.py" = ["N803", "N806", "N801"]
"packages/core/legoesm/core/fv3_native_duo_sw_core.py" = ["N803", "N806", "N801"]
"packages/core/legoesm/grids/fv3_native_gridstruct.py" = ["N801", "N803", "N806"]
# FV3-native oracle validators/tests: same verbatim-Fortran UPPER_SNAKE
# metric locals (SIN/COS/DXC/RC, ...) mirroring the source 1:1.
"scripts/validate/fv3_native/gen_divduo_oracle.py" = ["N806", "N816"]
"tests/grids/test_fv3_native_duo_phase4c.py" = ["N806"]
# Fuller CLUBB port (PORT_CLUBB.md): the closure coefficients, model flags and
# intermediate variables mirror CLUBB's canonical symbol names 1:1 (c_K1,
# iiPDF_type, l_use_C7_Richardson, C2rt, Skw, wpthlp, Lscale, ...) as in
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
# Vendored 3rd-party CLM-ML-JAX backend (BSD-3) — not type-annotated to legoESM's
# standard; excluded from type-checking here (audited upstream).
exclude = ['packages/land/legoesm/land/canopy/clm_ml_backend/']

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short -m 'not slow'"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    # Complexity-tier ladder (docs/validation/TESTING.md). Opt-in selectors: run a rung
    # with e.g. `-m 'tier1 and not slow'`. The default addopts stays `not slow`
    # (does NOT tier-gate) so untagged tests still run until the suite is tagged.
    "tier0: unit/operator tests — kernels, no model integration (numerical invariants)",
    "tier1: research tier — idealized/column/shallow-water; mass+energy+AAM gates, analytic benchmarks",
    "tier2: intermediate tier — hydrostatic 3D + slab; mass+energy+moisture gates (often slow)",
    "tier3: operational tier — full complexity + real forcing (AMIP/OMIP/ERA5); budget closure (slow)",
]

# ---------------------------------------------------------------------------
# Import-boundary enforcement (the path to independent self-running packages).
# These contracts encode the TARGET layered architecture (master plan D7):
#   core (substrate)  <  {atmosphere, ocean, land, ice}  <  coupler  <  driver
# Each Earth-system component must depend only on `core` (+ truly-shared bricks),
# never on another component or on the coupler/driver — that one-way dependency
# is exactly what lets `pip install legoesm-ocean` run standalone (Stage D carve).
#
168:[tool.ruff.lint]
169:select = ["E", "F", "I", "N", "W", "UP"]
/usr/bin/bash: line 3: python: command not found

exec
/usr/bin/bash -lc "command -v ruff || true
command -v uv || true
command -v pytest || true
awk 'NR==221 || NR==222 || NR==223 || NR==267 || NR==268 || NR==269 || NR==270 { printf \"%d:[%s]\\n\", NR, "'$0 }'"' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
git diff --name-status HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py
git ls-files --others --exclude-standard -- tests/ocean/unit/test_sharded_geom_fingerprint.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 273ms:
/home/b/b381103/.local/bin/uv
221:[]
222:[def geom_band_fingerprint(host, n_bands):]
223:[    """Low-memory cross-process fingerprint of a band-STACKED geometry field.]
267:[    return _np.array(struct, dtype=_np.float64), vals, is_exact]
268:[]
269:[]
270:[def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=1e-5):]
M	packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
tests/ocean/unit/test_sharded_geom_fingerprint.py

codex
I found two ship blockers in the review: the new regression test is still untracked, and the new top-level helper is one blank line short of the repository’s configured Ruff spacing rule.
codex
Issues:

- [test_sharded_geom_fingerprint.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/tests/ocean/unit/test_sharded_geom_fingerprint.py:1) is untracked, so it is absent from the requested `git diff HEAD` and will not ship unless added.
- [sharded_ocean_step.py:222](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:222) has only one blank line before a top-level function; configured Ruff `E` rules require two.

VERDICT: FIX-FIRST
tokens used
40,528
Issues:

- [test_sharded_geom_fingerprint.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/tests/ocean/unit/test_sharded_geom_fingerprint.py:1) is untracked, so it is absent from the requested `git diff HEAD` and will not ship unless added.
- [sharded_ocean_step.py:222](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:222) has only one blank line before a top-level function; configured Ruff `E` rules require two.

VERDICT: FIX-FIRST
