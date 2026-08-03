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
session id: 019fc632-767c-7021-830c-6ea0c1bfff62
--------
user
Round-17: review the EXTENSION of the assert-free put fix (git diff HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py). New evidence since r16: the post-geometry-fix @96/@128 reruns failed with the byte-identical 82.39 GB signature; the OOM traceback names jax's OWN device_put -> multihost_utils.assert_equal -> process_allgather of the FULL array (dispatch.py:514: any numpy-array put to an all-process sharding pays a whole-array allgather landing [n_proc,*shape] on one device — 96 x 849 MB = 81.5 GiB > 80 GB A100; 64 x = 54.3 GiB fit, explaining every earlier pass). Fix: new module helper _addressable_shard_put (docstringed with the mechanism) replacing jax.device_put at 5 setup-path sites: _shard_cell, _shard_v, the non-Field state leaf, shard_forcing_latlon._put, shard_forcing_stack_latlon._put; single-process path unchanged (plain device_put). 23-test SPMD parity suite green post-change (single-process virtual devices — exercises the OLD path; the multicontroller path's test IS the @96/@128 falsification pair now in queue, stated honestly).
Attack: (1) semantics differences device_put-vs-make_array_from_callback for P() replicated scalars and 1-D leaves under multicontroller (callback idx for a replicated sharding = full-array index per device — correct? memory duplication concerns?); (2) any OTHER setup-path device_put of host globals left in this module or the bench (grep); (3) the helper's np.asarray on jnp inputs (device arrays passed in? _shard_cell receives field.data possibly already on CPU device from the build context — np.asarray forces host copy, fine?); (4) losing jax's assert entirely — is any call site relying on it for a REAL cross-process consistency need not covered by the geometry fingerprint gate (state ICs? forcing?); if so what's the cheap replacement (e.g. per-band moments on state too)? (5) docstring accuracy. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + numbered issues.
codex
I’ll inspect the extension narrowly for multicontroller semantics, remaining host-global puts, consistency checks, and the helper’s stated mechanism.
exec
/usr/bin/bash -lc "pwd && rg --files -g 'AGENTS.md' -g 'agEnts.md' -g 'agents.md' .. . 2>/dev/null && git diff --check HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py && git diff --unified=100 HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 1 in 387ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign

exec
/usr/bin/bash -lc 'find .. -name AGENTS.md -print 2>/dev/null; git diff --check HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py; git diff --unified=100 HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 2125ms:
diff --git a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
index 406f4a76e..4991a56da 100644
--- a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
+++ b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
@@ -238,340 +238,367 @@ def geom_band_fingerprint(host, n_bands):
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
 
 
+def _addressable_shard_put(arr, sharding):
+    """Put a host array onto a (possibly multi-process) sharding WITHOUT
+    jax's whole-array cross-process ``assert_equal``.
+
+    Under multicontroller, ``jax.device_put(numpy_array, sharding)`` calls
+    ``multihost_utils.assert_equal`` on the FULL array
+    (jax _src/dispatch.py::_device_put_sharding_impl) — a
+    ``process_allgather`` whose output is ``[n_processes, *shape]`` on one
+    device: at LL2304 L20 one 3-D field is 849 MB, so 96 processes fetch
+    81.5 GiB > an 80 GB A100 (the wall that killed oc @96/@128, jobs
+    26644681/26644682, AFTER the geometry-broadcast fix; @64 = 54.3 GiB
+    just fit, which is why smaller ladders never saw it).
+
+    ``jax.make_array_from_callback`` supplies each process's addressable
+    shards directly — no consistency collective. The layouts here are
+    band-sharded or replicated stacks of DETERMINISTICALLY-built host
+    values; cross-process identity of non-owned rows is not required, and
+    real geometry divergence is refused by the fingerprint gate at the
+    geometry puts.
+    """
+    host = np.asarray(arr)
+    if jax.process_count() <= 1:
+        return jax.device_put(jnp.asarray(host), sharding)
+    return jax.make_array_from_callback(
+        host.shape, sharding, lambda idx: host[idx])
+
+
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
                 "Mask the cap row (the tripole cap convention) or extend "
                 "the carrier before sharding this state.")
 
     def _shard_cell(field):
         if field is None:
             return None
         sh = NamedSharding(mesh, _lat_spec(field.data))
-        return field.replace(data=jax.device_put(field.data, sh))
+        return field.replace(data=_addressable_shard_put(field.data, sh))
 
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
-        return field.replace(data=jax.device_put(v_lower, sh))
+        return field.replace(data=_addressable_shard_put(v_lower, sh))
 
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
-            updates[name] = jax.device_put(arr, NamedSharding(mesh, spec))
+            updates[name] = _addressable_shard_put(arr, NamedSharding(mesh, spec))
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
 
     def _put(leaf):
         if leaf is None:
             return None
         arr = jnp.asarray(leaf)
-        return jax.device_put(arr, NamedSharding(mesh, _lat_spec(arr)))
+        return _addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
 
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
-        return jax.device_put(arr, NamedSharding(mesh, spec))
+        return _addressable_shard_put(arr, NamedSharding(mesh, spec))
 
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

exec
/usr/bin/bash -lc "rg -n -C 5 \"device_put|make_array_from_callback|process_allgather|geom_band_fingerprint|fingerprint|shard_(state|forcing)|_addressable_shard_put\" packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py; rg -n -i -C 2 \"device_put|shard_state_latlon|shard_forcing_(stack_)?latlon|make_sharded_ocean_step\" --glob '*.{py,sh,slurm}' . | head -n 1200; rg -n -i \"@96|@128|multicontroller|multicontroller|route-B|route_b|multihost|scaling.*campaign|scaling-campaign\" .github packages tests scripts benchmarks 2>/dev/null | head -n 800" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 2328ms:
205-
206-def _content_hash48(arr) -> float:
207-    """48-bit content digest of ``arr``'s bytes, exactly representable in f64.
208-
209-    Used to compare EXACT-dtype arrays (masks, index tables) across
210:    processes: unlike moment fingerprints, a byte digest is positional, so a
211-    permutation or a two-cell flip cannot cancel. 48 bits keeps the value
212:    under 2**53 so it survives the float64 ``process_allgather`` payload
213-    exactly. Not cryptographic — collision-resistance at 2**-48 is far
214-    beyond the ~10 setup-time comparisons this guard makes.
215-    """
216-    import hashlib
217-
218-    a = np.ascontiguousarray(arr)
219-    h = hashlib.blake2b(a.tobytes(), digest_size=6)
220-    return float(int.from_bytes(h.digest(), "big"))
221-
222:def geom_band_fingerprint(host, n_bands):
223:    """Low-memory cross-process fingerprint of a band-STACKED geometry field.
224-
225-    ``host`` has leading axis ``n_bands`` (the per-device band stack). The
226:    fingerprint is PER BAND (codex 2026-08-03 r14: whole-array moments let a
227-    band-local drift hide in the global sum once each process's own bytes
228-    become live computation inputs):
229-
230-    * exact dtypes (int/bool/uint — masks, index tables): one positional
231-      48-bit byte digest per band (a permutation or two-cell flip within a
--
237-      change POSITION or kind (nan vs inf) with an unchanged per-band
238-      count also pass; band grids are analytic in lat/lon, so any real
239-      inconsistency moves the moments.
240-
241-    Returns ``(struct, vals, is_exact)`` as float64 arrays safe for
242:    ``process_allgather``.
243-    """
244-    import numpy as _np
245-
246-    host = _np.asarray(host)
247-    assert host.shape[0] == n_bands, (host.shape, n_bands)
--
265-            ])
266-        vals = _np.array(per_band, dtype=_np.float64)
267-    return _np.array(struct, dtype=_np.float64), vals, is_exact
268-
269-
270:def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=1e-5):
271:    """True iff every process's fingerprint matches process 0's.
272-
273:    ``g_struct``/``g_vals`` are the ``process_allgather``-ed outputs of
274:    :func:`geom_band_fingerprint` (leading axis = process). Structure and
275-    exact-dtype digests compare EXACTLY; float moments to ``rtol``.
276-    """
277-    import numpy as _np
278-
279-    struct_ok = bool(_np.all(g_struct == g_struct[0]))
--
283-        vals_ok = bool(_np.allclose(g_vals, g_vals[0], rtol=rtol, atol=0.0))
284-    return struct_ok and vals_ok
285-
286-
287-
288:def _schema_fingerprint(names, n_dev) -> np.ndarray:
289-    """Fixed-shape schema digest: field-name list, count, x64 flag, n_dev.
290-
291-    Gathered ONCE before the per-field loop so a process-dependent field
292-    selection is caught by a collective every process reaches, instead of
293-    desynchronizing the per-field gathers (codex round-5 findings 3/4).
--
333-    # shared interior boundary vertex row appears in band r AND band r+1) — the
334-    # same [s:e+1] slice slice_cgrid_geometry_to_band applies to area_q.
335-    return [jnp.asarray(vmask[r * nl: r * nl + nl + 1]) for r in range(n_dev)]
336-
337-
338:def _addressable_shard_put(arr, sharding):
339-    """Put a host array onto a (possibly multi-process) sharding WITHOUT
340-    jax's whole-array cross-process ``assert_equal``.
341-
342:    Under multicontroller, ``jax.device_put(numpy_array, sharding)`` calls
343-    ``multihost_utils.assert_equal`` on the FULL array
344:    (jax _src/dispatch.py::_device_put_sharding_impl) — a
345:    ``process_allgather`` whose output is ``[n_processes, *shape]`` on one
346-    device: at LL2304 L20 one 3-D field is 849 MB, so 96 processes fetch
347-    81.5 GiB > an 80 GB A100 (the wall that killed oc @96/@128, jobs
348-    26644681/26644682, AFTER the geometry-broadcast fix; @64 = 54.3 GiB
349-    just fit, which is why smaller ladders never saw it).
350-
351:    ``jax.make_array_from_callback`` supplies each process's addressable
352-    shards directly — no consistency collective. The layouts here are
353-    band-sharded or replicated stacks of DETERMINISTICALLY-built host
354-    values; cross-process identity of non-owned rows is not required, and
355:    real geometry divergence is refused by the fingerprint gate at the
356-    geometry puts.
357-    """
358-    host = np.asarray(arr)
359-    if jax.process_count() <= 1:
360:        return jax.device_put(jnp.asarray(host), sharding)
361:    return jax.make_array_from_callback(
362-        host.shape, sharding, lambda idx: host[idx])
363-
364-
365:def shard_state_latlon(state, mesh):
366-    """Lay out a ``LatLonCGridOceanState`` for the lat-band SPMD step.
367-
368-    Cell / u-grid array fields (leading dim ``n_lat``) shard ``P("lat")``.  The
369-    staggered ``v`` / ``v_mask`` (leading dim ``n_lat+1``) are carried as
370-    ``v_lower = field[0:n_lat]`` (``n_lat`` rows, ``P("lat")``) — the top pole row
--
385-    if vm is not None:
386-        import numpy as _np
387-
388-        if _np.asarray(vm.data)[-1].any():
389-            raise ValueError(
390:                "shard_state_latlon: the state's TOP v-face row is LIVE "
391-                "(v_mask[-1] has ocean faces) — the lat-band v-carrier "
392-                "drops that row and reconstructs it as the pole/cap wall "
393-                "zero, which would silently delete seam velocities. "
394-                "Mask the cap row (the tripole cap convention) or extend "
395-                "the carrier before sharding this state.")
396-
397-    def _shard_cell(field):
398-        if field is None:
399-            return None
400-        sh = NamedSharding(mesh, _lat_spec(field.data))
401:        return field.replace(data=_addressable_shard_put(field.data, sh))
402-
403-    def _shard_v(field):
404-        if field is None:
405-            return None
406-        # Drop the TOP v-row (the north pole wall, v==0 on the regular grid) so
407-        # the leading dim becomes n_lat (divisible by N).  ROUND-TRIP INVARIANT
408:        # (codex finding): shard_state_latlon -> gather_state_latlon re-appends a
409-        # ZERO top row, so the round-trip is a bitwise identity ONLY when the
410-        # input's top v-row is already zero (a valid masked regular-grid state;
411-        # the pole-wall BC enforces v[n_lat]=0 every step).  An arbitrary nonzero
412-        # top row (e.g. a raw IC perturbation) is dropped -> reconstructed as 0:
413-        # CORRECT for the dynamics (the pole wall zeros it at step 1) but not a
414-        # bit round-trip of that one row.  NOT for a tripole north fold (raises
415-        # in make_sharded_ocean_step) where the top row is a live fold partner.
416-        nlat1 = field.data.shape[0]
417-        v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
418-        sh = NamedSharding(mesh, _lat_spec(v_lower))
419:        return field.replace(data=_addressable_shard_put(v_lower, sh))
420-
421-    updates = {}
422-    for name in _V_STAGGERED_STATE_FIELDS:
423-        updates[name] = _shard_v(getattr(state, name))
424-    # Every other (array) leaf: shard P("lat").  NamedTuple fields not in the
--
434-        else:
435-            # Non-Field, non-None leaf (e.g. a raw array carry like psi).
436-            # Shard 2-D+ on lat, replicate lower-rank — matches shard_pytree.
437-            arr = jnp.asarray(val)
438-            spec = _lat_spec(arr) if arr.ndim >= 1 else P()
439:            updates[name] = _addressable_shard_put(arr, NamedSharding(mesh, spec))
440-    return state._replace(**updates)
441-
442-
443:def shard_forcing_latlon(forcing, mesh):
444-    """Lay out a forcing pytree (``FreshwaterForcing`` / ``OceanSurfaceForcing``
445-    / ``SpongeForcing`` — or any nesting of them) for the lat-band SPMD step.
446-
447-    Every OMIP forcing field is CELL-CENTERED ``(n_lat, n_lon[, nlev])`` (the
448-    cell->face wind-stress interpolation happens INSIDE the step through the
--
456-
457-    def _put(leaf):
458-        if leaf is None:
459-            return None
460-        arr = jnp.asarray(leaf)
461:        return _addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
462-
463-    return jax.tree.map(_put, forcing)
464-
465-
466:def shard_forcing_stack_latlon(stack, mesh):
467-    """Lay out a STACKED per-block forcing pytree for the lat-band SPMD
468-    block-scan (the ``run_omip`` JRA55 lanes; see
469-    ``_build_jra55_block_fn`` / ``_build_jra55_block_fn_interp``).
470-
471:    Unlike :func:`shard_forcing_latlon` (per-step, lat at axis 0), the
472-    block builders stack ``N`` steps / raw records along a LEADING axis,
473-    so the lat axis sits at position 1.  A ``(n_rec, n_lat, n_lon[, ...])``
474-    leaf therefore shards ``P(None, "lat", ...)`` (records replicated, the
475-    time index stays shard-local so the in-scan interpolation needs no
476-    cross-band comm); a bare ``(n_lat, n_lon)`` leaf shards ``P("lat", None)``;
--
481-    be a ``(n_lat, n_lon)`` field and is lat-sharded on axis 0.  Any future
482-    metadata that is genuinely rank-2 but NOT lat-major (e.g. a
483-    ``(n_rec, n_meta)`` table) would be silently mis-sharded — keep such
484-    metadata 1-D (or replicate it explicitly) before it reaches this helper.
485-
486:    Keeping this next to :func:`shard_state_latlon` means the driver and
487-    the parity tests share ONE layout definition — the block-scan forcing
488-    stack must be laid out consistently with the state the sharded step
489-    carries, and a second copy would drift.
490-    """
491-    if mesh is None:
--
499-            spec = P(None, "lat", *((None,) * (arr.ndim - 2)))
500-        elif arr.ndim == 2:
501-            spec = P("lat", None)
502-        else:
503-            spec = P()
504:        return _addressable_shard_put(arr, NamedSharding(mesh, spec))
505-
506-    return jax.tree.map(_put, stack)
507-
508-
509-def append_vface_wall_row(v_lower):
510-    """Rebuild the full ``(n_lat+1, ...)`` staggered v-array from the lat-band
511-    carrier ``v_lower`` by appending the TOP wall row (zeros).
512-
513-    This is THE reconstruction of the row the carrier drops: the regular-grid
514-    north pole wall / tripole cap row, identically zero under the v-carrier
515:    contract (:func:`shard_state_latlon` REFUSES a state whose top v-face row
516-    is live), so the result is bit-identical to the original staggered array.
517-    Shared by :func:`gather_state_latlon` (full-state gather) and the
518-    persistent-lane DEVICE-SIDE staggered reads (the OMIP driver's
519-    surface-current consumers, ``run_omip_core2._surface_uv_faces``) so the
520-    row reconstruction is written once.  Works on any trailing shape (3-D
--
523-    wall = jnp.zeros_like(v_lower[:1])
524-    return jnp.concatenate([v_lower, wall], axis=0)
525-
526-
527-def gather_state_latlon(state, mesh):
528:    """Inverse of :func:`shard_state_latlon`: gather every leaf to a single device
529-    and rebuild the full ``(n_lat+1, ...)`` ``v`` / ``v_mask`` by appending the
530-    pole-wall row (zeros) the layout dropped.
531-
532-    The appended row is the north pole wall (``v == 0`` there on the regular
533-    grid), which is exactly what the single-device rest/forced state carries, so
534-    the gathered state is bit-comparable to the single-device reference.
535-
536-    Multi-controller (route-B ``jax.distributed``, mesh spanning processes):
537-    replication routes through a jit-compiled identity instead of
538:    ``device_put`` (see :func:`legoesm.parallel.latlon_spmd.replicate_leaf`,
539-    the primitive shared with the atm gather); the single-process path is
540-    byte-unchanged.
541-    """
542-    from legoesm.parallel.latlon_spmd import replicate_leaf
543-
--
573-    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
574-    sponge=None, t_seconds=None) -> state`` running ``model.step``
575-    lat-band-SPMD.
576-
577-    The forcing channels mirror ``model.step``'s keyword surface: pass
578:    pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
579-    ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
580-    is a replicated scalar like ``dt``).  ``None`` forcing keeps the
581-    dynamics-only program; each distinct None<->populated combination
582-    compiles (and caches) its own executable.
583-
--
601-    (a per-call rebuild re-traced the whole ocean step every call — see the
602-    ``_cache`` note below).  ``check_vma=False`` (the JAX >= 0.8
603-    replication check) because the band halo intentionally reads neighbour-rank
604-    data (replication-unaware).
605-
606:    The state must be laid out with :func:`shard_state_latlon` (``v`` /
607-    ``v_mask`` carried as the ``n_lat``-row ``v_lower``).  The body reconstructs
608-    each band's ``nl+1`` v-faces, runs the step on the band geometry, and
609-    converts the result back to the ``v_lower`` representation.
610-    """
611-    if mesh is None:                   # single-device: plain step
--
625-    # assumption is the v-carrier's: the TOP v-face row (the seam/cap row,
626-    # ``v[n_lat]``) must be WALL-MASKED so the in-body reconstruction's
627-    # zero row is exact — true for the cap-row convention of
628-    # ``create_synthetic_tripole`` and the eORCA masks (``v_mask[-1] == 0``;
629-    # the serial step keeps ``v[-1] == 0`` identically).  That contract is
630:    # asserted on the CONCRETE state in :func:`shard_state_latlon` — a live
631-    # (unmasked) seam v-row refuses loudly there instead of silently
632-    # reconstructing zeros here.
633-
634-    # --- host-side band geometries + vertex masks (band-stacked, P("lat")-sharded) ---
635-    band_grids = build_band_grids(model.grid, n_dev)
--
657-        # assert (job 26450848). With the #1370-iii P("lat") sharding each
658-        # process's devices consume ONLY its own band rows, so the
659-        # broadcast became both unnecessary and, at nd>=96, fatal (its
660-        # psum program is nd x the stack — see the note at the put below).
661-        # GUARD (codex round-3): process 0 must not silently mask REAL
662:        # cross-process divergence. Compare an allgathered fingerprint:
663-        # structural entries exactly; value entries EXACTLY for integer/bool
664-        # arrays (masks are comparison results — bit-reproducible, and an
665-        # exact compare is the only way to catch a two-cell flip that cancels
666-        # in the sum, codex round-4) and to rtol 1e-5 for float arrays (only
667-        # ULP autotune drift is expected there; quantize-then-assert-equal
668-        # false-positived on a rounding boundary, job 26453240).
669:        # NO DEADLOCK RISK: every process fingerprints the same fields in the
670-        # same order and derives the verdict from the SAME gathered array, so
671-        # the refusal is symmetric — all raise or none.
672-        # Residual (documented): a float-geometry divergence preserving sum,
673-        # sum-of-squares AND absmax to 1e-5 is not detected; band grids are
674-        # analytic in lat/lon, so any real inconsistency moves those moments.
675-        host = np.asarray(arr)
676-        if jax.process_count() > 1:
677-            from jax.experimental import multihost_utils
678-
679:            # PER-BAND fingerprints (module-level, unit-tested): exact
680-            # dtypes hash positionally per band; floats compare per-band
681-            # moments to rtol 1e-5 — bounds each band's drift instead of
682-            # letting it hide in a whole-array sum, since each process's
683-            # own bytes are now the live inputs for the bands it owns
684-            # (codex r14). A mask that genuinely differs across processes
685-            # means different wet domains = different physics: refusing is
686-            # correct, not a false alarm.
687:            struct, vals, is_exact = geom_band_fingerprint(
688-                host, host.shape[0])
689:            g_struct = multihost_utils.process_allgather(struct)
690:            g_vals = multihost_utils.process_allgather(vals)
691:            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
692-                raise RuntimeError(
693-                    f"make_sharded_ocean_step: band-geometry field {name!r} "
694-                    f"DIVERGES across processes (exact_dtype={is_exact}, "
695-                    f"gathered={g_vals.tolist()}) — a real config/grid "
696-                    f"inconsistency, not autotune noise; refusing to "
--
702-            # near-linear-in-nd wall that killed oc @96/@128 (jobs
703-            # 26642771/26636762) while @64 sat just under XLA's 63.8 GB
704-            # limit.  The target sharding is P("lat"): each process's
705-            # devices consume ONLY its own band rows, so cross-process
706-            # byte-identity of non-owned rows is irrelevant, and REAL
707:            # divergence is already refused by the fingerprint gate above.
708:            # make_array_from_callback hands each process exactly its
709-            # addressable slabs — the same #1100 pattern as the state
710-            # build — with no global-sized collective program at all.
711-        host_np = np.asarray(host)
712:        return jax.make_array_from_callback(
713-            host_np.shape, rep, lambda idx: host_np[idx])
714-
715-    if jax.process_count() > 1:
716-        # Schema gate FIRST (one fixed-shape collective every process
717-        # reaches): a process-dependent field list or a mixed
718-        # jax_enable_x64 setting would otherwise desynchronize the
719-        # per-field gathers below instead of failing with a clear message.
720-        from jax.experimental import multihost_utils as _mhu
721-
722:        _g = _mhu.process_allgather(
723:            _schema_fingerprint(list(array_field_names), n_dev))
724-        if not bool(np.all(_g == _g[0])):
725-            raise RuntimeError(
726-                "make_sharded_ocean_step: the band-geometry SCHEMA differs "
727-                "across processes (field list / x64 setting / device count "
728-                f"— gathered {_g.tolist()}). Fix the per-process config "
--
935-    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
936-    that takes a GLOBAL (single-device-layout) state + forcing and returns a
937-    GLOBAL state — the minimal-diff driver entry point.
938-
939-    Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
940:    (:func:`shard_state_latlon` + :func:`shard_forcing_latlon`), runs the lat-band
941-    SPMD step, then gathers the state OUT (:func:`gather_state_latlon`).  This lets
942-    the OMIP host loop keep operating on a normal full-domain state — the per-step
943-    host BCs (SSS restore, prognostic ice, geothermal, BBL, nudge) see the
944-    gathered global state UNCHANGED — at the cost of a per-step gather/scatter
945-    (acceptable for the host-coupled OMIP driver; the pure-dynamics inner loop
--
956-
957-    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
958-        # Scatter the global state to the band layout; the forcing is sharded
959-        # INSIDE ``inner`` (make_sharded_ocean_step lays it out), so pass it
960-        # through global.
961:        ss = shard_state_latlon(state, mesh)
962-        ss = inner(ss, dt, surface_forcing=surface_forcing,
963-                   freshwater=freshwater)
964-        return gather_state_latlon(ss, mesh)
965-
966-    return sharded_step_global
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-787-
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-788-        if self._use_cpu_for_spectral:
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py:789:            state_cpu = jax.device_put(state, self._cpu_device)
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-790-            result_cpu = dispatch_integrator(state_cpu, tendency_fn, dt, self.config.time_integrator)
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-791-            if self.config.use_conservation_fixer:
--
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-793-                    result_cpu, state_cpu, self.grid, self.z_coord, self.config,
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-794-                )
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py:795:            return jax.device_put(result_cpu, self._default_device)
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-796-
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-797-        result = dispatch_integrator(state, tendency_fn, dt, self.config.time_integrator)
--
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-855-    def _integrate_on_cpu(self, state, n_steps, dt, save_every):
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-856-        """Batch integration on CPU: transfer once, not per step."""
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py:857:        state_cpu = jax.device_put(state, self._cpu_device)
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-858-        trajectory_cpu = [state_cpu]
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-859-
--
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-863-                trajectory_cpu.append(state_cpu)
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-864-
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py:865:        state_out = jax.device_put(state_cpu, self._default_device)
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-866-        trajectory_out = [
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py:867:            jax.device_put(s, self._default_device) for s in trajectory_cpu
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-868-        ]
./packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-869-        return state_out, trajectory_out
--
./scripts/bench/bench_atm_latlon_spmd_scaling.py-230-        # the global (n_lat, n_lon, nlev) state per process — each leaf is
./scripts/bench/bench_atm_latlon_spmd_scaling.py-231-        # created via make_array_from_callback for the rows this process's
./scripts/bench/bench_atm_latlon_spmd_scaling.py:232:        # devices own (no global build, no device_put replication, no
./scripts/bench/bench_atm_latlon_spmd_scaling.py-233-        # assert_equal all-gather). This is what lets full-node-packed CPU
./scripts/bench/bench_atm_latlon_spmd_scaling.py-234-        # rungs (128 procs/node) survive at large n_lat.
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-324-    if vmask is None:
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-325-        raise RuntimeError(
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:326:            "make_sharded_ocean_step: the model vertex-mask cache is not "
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-327-            "primed.  Call model._ensure_vertex_mask(state) (or model.step) "
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-328-            "on the concrete initial state before building the sharded step.")
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-340-    jax's whole-array cross-process ``assert_equal``.
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-341-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:342:    Under multicontroller, ``jax.device_put(numpy_array, sharding)`` calls
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-343-    ``multihost_utils.assert_equal`` on the FULL array
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:344:    (jax _src/dispatch.py::_device_put_sharding_impl) — a
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-345-    ``process_allgather`` whose output is ``[n_processes, *shape]`` on one
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-346-    device: at LL2304 L20 one 3-D field is 849 MB, so 96 processes fetch
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-358-    host = np.asarray(arr)
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-359-    if jax.process_count() <= 1:
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:360:        return jax.device_put(jnp.asarray(host), sharding)
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-361-    return jax.make_array_from_callback(
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-362-        host.shape, sharding, lambda idx: host[idx])
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-363-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-364-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:365:def shard_state_latlon(state, mesh):
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-366-    """Lay out a ``LatLonCGridOceanState`` for the lat-band SPMD step.
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-367-
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-373-    :func:`gather_state_latlon`.
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-374-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:375:    This is the layout the in-``shard_map`` body of :func:`make_sharded_ocean_step`
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-376-    expects; the test uses it instead of a uniform ``tree.map(P("lat"))`` (which
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-377-    fails on ``v`` because ``n_lat+1`` is not divisible by ``N``).
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-378-    """
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:379:    # v-carrier contract (see make_sharded_ocean_step's fold note): the TOP
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-380-    # v-face row (regular pole wall OR tripole seam/cap row) must be
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-381-    # wall-masked — the carrier drops it and reconstructs it as zero, which
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-388-        if _np.asarray(vm.data)[-1].any():
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-389-            raise ValueError(
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:390:                "shard_state_latlon: the state's TOP v-face row is LIVE "
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-391-                "(v_mask[-1] has ocean faces) — the lat-band v-carrier "
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-392-                "drops that row and reconstructs it as the pole/cap wall "
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-406-        # Drop the TOP v-row (the north pole wall, v==0 on the regular grid) so
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-407-        # the leading dim becomes n_lat (divisible by N).  ROUND-TRIP INVARIANT
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:408:        # (codex finding): shard_state_latlon -> gather_state_latlon re-appends a
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-409-        # ZERO top row, so the round-trip is a bitwise identity ONLY when the
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-410-        # input's top v-row is already zero (a valid masked regular-grid state;
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-413-        # CORRECT for the dynamics (the pole wall zeros it at step 1) but not a
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-414-        # bit round-trip of that one row.  NOT for a tripole north fold (raises
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:415:        # in make_sharded_ocean_step) where the top row is a live fold partner.
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-416-        nlat1 = field.data.shape[0]
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-417-        v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-441-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-442-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:443:def shard_forcing_latlon(forcing, mesh):
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-444-    """Lay out a forcing pytree (``FreshwaterForcing`` / ``OceanSurfaceForcing``
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-445-    / ``SpongeForcing`` — or any nesting of them) for the lat-band SPMD step.
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-464-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-465-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:466:def shard_forcing_stack_latlon(stack, mesh):
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-467-    """Lay out a STACKED per-block forcing pytree for the lat-band SPMD
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-468-    block-scan (the ``run_omip`` JRA55 lanes; see
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-469-    ``_build_jra55_block_fn`` / ``_build_jra55_block_fn_interp``).
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-470-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:471:    Unlike :func:`shard_forcing_latlon` (per-step, lat at axis 0), the
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-472-    block builders stack ``N`` steps / raw records along a LEADING axis,
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-473-    so the lat axis sits at position 1.  A ``(n_rec, n_lat, n_lon[, ...])``
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-484-    metadata 1-D (or replicate it explicitly) before it reaches this helper.
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-485-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:486:    Keeping this next to :func:`shard_state_latlon` means the driver and
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-487-    the parity tests share ONE layout definition — the block-scan forcing
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-488-    stack must be laid out consistently with the state the sharded step
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-513-    This is THE reconstruction of the row the carrier drops: the regular-grid
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-514-    north pole wall / tripole cap row, identically zero under the v-carrier
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:515:    contract (:func:`shard_state_latlon` REFUSES a state whose top v-face row
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-516-    is live), so the result is bit-identical to the original staggered array.
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-517-    Shared by :func:`gather_state_latlon` (full-state gather) and the
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-526-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-527-def gather_state_latlon(state, mesh):
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:528:    """Inverse of :func:`shard_state_latlon`: gather every leaf to a single device
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-529-    and rebuild the full ``(n_lat+1, ...)`` ``v`` / ``v_mask`` by appending the
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-530-    pole-wall row (zeros) the layout dropped.
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-536-    Multi-controller (route-B ``jax.distributed``, mesh spanning processes):
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-537-    replication routes through a jit-compiled identity instead of
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:538:    ``device_put`` (see :func:`legoesm.parallel.latlon_spmd.replicate_leaf`,
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-539-    the primitive shared with the atm gather); the single-process path is
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-540-    byte-unchanged.
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-570-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-571-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:572:def make_sharded_ocean_step(model, mesh):
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-573-    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-574-    sponge=None, t_seconds=None) -> state`` running ``model.step``
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-576-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-577-    The forcing channels mirror ``model.step``'s keyword surface: pass
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:578:    pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-579-    ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-580-    is a replicated scalar like ``dt``).  ``None`` forcing keeps the
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-604-    data (replication-unaware).
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-605-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:606:    The state must be laid out with :func:`shard_state_latlon` (``v`` /
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-607-    ``v_mask`` carried as the ``n_lat``-row ``v_lower``).  The body reconstructs
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-608-    each band's ``nl+1`` v-faces, runs the step on the band geometry, and
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-628-    # ``create_synthetic_tripole`` and the eORCA masks (``v_mask[-1] == 0``;
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-629-    # the serial step keeps ``v[-1] == 0`` identically).  That contract is
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:630:    # asserted on the CONCRETE state in :func:`shard_state_latlon` — a live
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-631-    # (unmasked) seam v-row refuses loudly there instead of silently
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-632-    # reconstructing zeros here.
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-691-            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-692-                raise RuntimeError(
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:693:                    f"make_sharded_ocean_step: band-geometry field {name!r} "
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-694-                    f"DIVERGES across processes (exact_dtype={is_exact}, "
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-695-                    f"gathered={g_vals.tolist()}) — a real config/grid "
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-724-        if not bool(np.all(_g == _g[0])):
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-725-            raise RuntimeError(
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:726:                "make_sharded_ocean_step: the band-geometry SCHEMA differs "
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-727-                "across processes (field list / x64 setting / device count "
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-728-                f"— gathered {_g.tolist()}). Fix the per-process config "
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-932-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-933-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:934:def make_sharded_ocean_step_global(model, mesh):
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-935-    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-936-    that takes a GLOBAL (single-device-layout) state + forcing and returns a
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-937-    GLOBAL state — the minimal-diff driver entry point.
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-938-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:939:    Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:940:    (:func:`shard_state_latlon` + :func:`shard_forcing_latlon`), runs the lat-band
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-941-    SPMD step, then gathers the state OUT (:func:`gather_state_latlon`).  This lets
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-942-    the OMIP host loop keep operating on a normal full-domain state — the per-step
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-944-    gathered global state UNCHANGED — at the cost of a per-step gather/scatter
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-945-    (acceptable for the host-coupled OMIP driver; the pure-dynamics inner loop
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:946:    should use :func:`make_sharded_ocean_step` directly to stay sharded).
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-947-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-948-    ``mesh is None`` ⇒ the plain single-device ``model.step`` (no scatter/gather).
--
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-953-                       surface_forcing=surface_forcing))
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-954-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:955:    inner = make_sharded_ocean_step(model, mesh)
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-956-
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-957-    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-958-        # Scatter the global state to the band layout; the forcing is sharded
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:959:        # INSIDE ``inner`` (make_sharded_ocean_step lays it out), so pass it
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-960-        # through global.
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:961:        ss = shard_state_latlon(state, mesh)
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-962-        ss = inner(ss, dt, surface_forcing=surface_forcing,
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-963-                   freshwater=freshwater)
--
./scripts/bench/roofline_probe.py-389-        # Place on device 0 explicitly and block so allocation/H2D is done
./scripts/bench/roofline_probe.py-390-        # before timing.
./scripts/bench/roofline_probe.py:391:        x = jax.device_put(x, dev0)
./scripts/bench/roofline_probe.py:392:        y = jax.device_put(y, dev0)
./scripts/bench/roofline_probe.py-393-        jax.block_until_ready((x, y))
./scripts/bench/roofline_probe.py-394-
--
./scripts/bench/roofline_probe.py-518-        # mesh so each device owns a (1, n) shard → a length-n payload.
./scripts/bench/roofline_probe.py-519-        x_global = jnp.arange(2 * n, dtype=dtype).reshape(2, n)
./scripts/bench/roofline_probe.py:520:        x_global = jax.device_put(
./scripts/bench/roofline_probe.py-521-            x_global, jax.sharding.NamedSharding(mesh, P("face")))
./scripts/bench/roofline_probe.py-522-        jax.block_until_ready(x_global)
--
./scripts/bench/roofline_probe.py-704-    for b in batch_sizes:
./scripts/bench/roofline_probe.py-705-        x_global = jnp.ones((2, b), dtype=dtype)
./scripts/bench/roofline_probe.py:706:        x_global = jax.device_put(
./scripts/bench/roofline_probe.py-707-            x_global, jax.sharding.NamedSharding(mesh, P("r")))
./scripts/bench/roofline_probe.py-708-        jax.block_until_ready(x_global)
--
./scripts/bench/roofline_probe.py-810-    scanned time loop."""
./scripts/bench/roofline_probe.py-811-    x = jnp.ones((8,), dtype=dtype)
./scripts/bench/roofline_probe.py:812:    x = jax.device_put(x)
./scripts/bench/roofline_probe.py-813-    jax.block_until_ready(x)
./scripts/bench/roofline_probe.py-814-
--
./scripts/bench/run_cpu_mpi_scaling.py-766-    cz5 = NamedSharding(mesh, P("face", "tile_i", "tile_j", None, None))
./scripts/bench/run_cpu_mpi_scaling.py-767-    state = {
./scripts/bench/run_cpu_mpi_scaling.py:768:        "u_d": jax.device_put(
./scripts/bench/run_cpu_mpi_scaling.py-769-            expand_corners_to_blocks(fv3.u_d.data, kt, nl), cz),
./scripts/bench/run_cpu_mpi_scaling.py:770:        "v_d": jax.device_put(
./scripts/bench/run_cpu_mpi_scaling.py-771-            expand_corners_to_blocks(fv3.v_d.data, kt, nl), cz),
./scripts/bench/run_cpu_mpi_scaling.py:772:        "T": jax.device_put(fv3.T.data, cz),
./scripts/bench/run_cpu_mpi_scaling.py:773:        "p_s": jax.device_put(fv3.p_s.data, co),
./scripts/bench/run_cpu_mpi_scaling.py:774:        "phis": jax.device_put(fv3.phis.data, co),
./scripts/bench/run_cpu_mpi_scaling.py-775-    }
./scripts/bench/run_cpu_mpi_scaling.py-776-    if _moist:
--
./scripts/bench/run_cpu_mpi_scaling.py-780-        q_pack = jnp.stack(
./scripts/bench/run_cpu_mpi_scaling.py-781-            [fv3.tracers[nm].data for nm in ("q_v", "q_c", "q_r")], axis=-1)
./scripts/bench/run_cpu_mpi_scaling.py:782:        state["q_pack"] = jax.device_put(q_pack, cz5)
./scripts/bench/run_cpu_mpi_scaling.py-783-
./scripts/bench/run_cpu_mpi_scaling.py-784-    _dt_built = float(dt)
--
./tests/test_cases/baroclinic_wave.py-781-    ``jax.make_array_from_callback``, whose callback runs only for the
./tests/test_cases/baroclinic_wave.py-782-    cell/edge ranges owned by THIS process's addressable devices — no
./tests/test_cases/baroclinic_wave.py:783:    per-process global state build, no global ``device_put``.  The mesh
./tests/test_cases/baroclinic_wave.py-784-    itself remains global on every process (its partition-local
./tests/test_cases/baroclinic_wave.py-785-    construction through the SFC machinery is the OPEN remainder of
--
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-6-split bench and ``bench_ocean_latlon_spmd_pcg.py`` times the barotropic PCG
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-7-KERNEL only — neither times the composed production step
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:8:(``make_sharded_ocean_step``: baroclinic + barotropic [implicit-CN
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-9-free-surface by default, split-explicit via ``--baro-solver``] + implicit
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-10-vmix + tracers) under the lat-band SPMD backend.
--
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-23-atm bench — every process calls ``jax.distributed.initialize`` BEFORE any
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-24-other JAX use, the ("lat",) mesh is built over the GLOBAL ``jax.devices()``,
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:25:and the existing ``make_sharded_ocean_step`` band-ppermute halo + psum
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-26-reductions (incl. the barotropic ``_global_sum_pair``) run unchanged across
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-27-processes (NCCL on GPU / gloo on CPU). NO mpi4jax is armed in this mode (the
--
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-344-
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-345-    from legoesm.ocean.dynamics.sharded_ocean_step import (
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:346:        make_sharded_ocean_step,
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:347:        shard_state_latlon,
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-348-    )
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-349-
--
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-381-    # LL2304@64 (probe 26523157: the compiled STEP is clean; the residency
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-382-    # is setup-time). Host-side globals are RAM, and only the per-band
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:383:    # shards reach the accelerator via shard_state_latlon's device_put.
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-384-    # Identical values on every process (deterministic init + the step
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-385-    # factory's existing process-0 broadcast + content-hash guard).
--
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-448-    if nd == 1:
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-449-        mesh = None
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:450:        step = make_sharded_ocean_step(model, None)
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-451-        s = s0
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-452-    else:
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-453-        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-454-                                 axis_names=("lat",))
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:455:        step = make_sharded_ocean_step(model, mesh)
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:456:        s = shard_state_latlon(s0, mesh)
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-457-
./scripts/bench/bench_ocean_latlon_spmd_scaling.py-458-    # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
--
./tests/unit/test_run_omip_latlon_spmd.py-54-    from legoesm.ocean.dynamics.sharded_ocean_step import (
./tests/unit/test_run_omip_latlon_spmd.py-55-        gather_state_latlon,
./tests/unit/test_run_omip_latlon_spmd.py:56:        make_sharded_ocean_step,
./tests/unit/test_run_omip_latlon_spmd.py:57:        shard_state_latlon,
./tests/unit/test_run_omip_latlon_spmd.py-58-    )
./tests/unit/test_run_omip_latlon_spmd.py-59-    from legoesm.parallel.mesh import create_latlon_mesh
--
./tests/unit/test_run_omip_latlon_spmd.py-80-    model.prime_step_caches(state0)
./tests/unit/test_run_omip_latlon_spmd.py-81-    dev = create_latlon_mesh(n_devices=2)
./tests/unit/test_run_omip_latlon_spmd.py:82:    spmd_step = make_sharded_ocean_step(model, dev.mesh)
./tests/unit/test_run_omip_latlon_spmd.py:83:    ss0 = shard_state_latlon(state0, dev.mesh)
./tests/unit/test_run_omip_latlon_spmd.py-84-    ckpt_dir = tmp_path / "spmd"
./tests/unit/test_run_omip_latlon_spmd.py-85-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
--
./tests/unit/test_run_omip_latlon_spmd.py-161-
./tests/unit/test_run_omip_latlon_spmd.py-162-    ``interp_mode`` is ``"cpu"`` (host-interp block) or ``"gpu"`` (in-scan
./tests/unit/test_run_omip_latlon_spmd.py:163:    GPU-interp block).  Uses the SAME shard_forcing_stack_latlon layout the
./tests/unit/test_run_omip_latlon_spmd.py-164-    driver uses (imported, not re-derived) so the test can't drift from it.
./tests/unit/test_run_omip_latlon_spmd.py-165-    """
--
./tests/unit/test_run_omip_latlon_spmd.py-178-    from legoesm.ocean.dynamics.sharded_ocean_step import (
./tests/unit/test_run_omip_latlon_spmd.py-179-        gather_state_latlon,
./tests/unit/test_run_omip_latlon_spmd.py:180:        make_sharded_ocean_step,
./tests/unit/test_run_omip_latlon_spmd.py:181:        shard_forcing_stack_latlon,
./tests/unit/test_run_omip_latlon_spmd.py:182:        shard_state_latlon,
./tests/unit/test_run_omip_latlon_spmd.py-183-    )
./tests/unit/test_run_omip_latlon_spmd.py-184-    from legoesm.parallel.mesh import create_latlon_mesh
--
./tests/unit/test_run_omip_latlon_spmd.py-217-
./tests/unit/test_run_omip_latlon_spmd.py-218-    dev = create_latlon_mesh(n_devices=2)
./tests/unit/test_run_omip_latlon_spmd.py:219:    spmd_step = make_sharded_ocean_step(model, dev.mesh)
./tests/unit/test_run_omip_latlon_spmd.py-220-    model.prime_step_caches(state0)
./tests/unit/test_run_omip_latlon_spmd.py:221:    ss0 = shard_state_latlon(state0, dev.mesh)
./tests/unit/test_run_omip_latlon_spmd.py-222-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
./tests/unit/test_run_omip_latlon_spmd.py-223-        model, ss0, jra55_state=_fresh_js(),
./tests/unit/test_run_omip_latlon_spmd.py-224-        spmd_step=spmd_step,
./tests/unit/test_run_omip_latlon_spmd.py-225-        spmd_gather=lambda st: gather_state_latlon(st, dev.mesh),
./tests/unit/test_run_omip_latlon_spmd.py:226:        spmd_shard_stack=partial(shard_forcing_stack_latlon, mesh=dev.mesh),
./tests/unit/test_run_omip_latlon_spmd.py-227-        **common)
./tests/unit/test_run_omip_latlon_spmd.py-228-    assert ok_spmd, "SPMD JRA55 loop reported not-ok"
--
./scripts/bench/bench_ocean_latlon_spmd_pcg.py-173-        mesh = Mesh(np.array(devs[:n_dev]), axis_names=("lat",))
./scripts/bench/bench_ocean_latlon_spmd_pcg.py-174-        isp = P("lat", None)
./scripts/bench/bench_ocean_latlon_spmd_pcg.py:175:        b_sh = jax.device_put(b_host, NamedSharding(mesh, isp))
./scripts/bench/bench_ocean_latlon_spmd_pcg.py:176:        x0_sh = jax.device_put(x0_host, NamedSharding(mesh, isp))
./scripts/bench/bench_ocean_latlon_spmd_pcg.py-177-
./scripts/bench/bench_ocean_latlon_spmd_pcg.py-178-        activate_latlon_spmd_halo(mesh)
--
./scripts/bench/bench_cube_shardmap_halo.py-391-            has_collective = _hlo_has_ppermute(_f, x_sh)
./scripts/bench/bench_cube_shardmap_halo.py-392-            y_sh, vjp_sh = jax.vjp(_f, x_sh)
./scripts/bench/bench_cube_shardmap_halo.py:393:            w_sh = jax.device_put(w, dev_config.face_sharding)
./scripts/bench/bench_cube_shardmap_halo.py-394-            g_spmd = np.asarray(jax.device_get(vjp_sh(w_sh)[0]))
./scripts/bench/bench_cube_shardmap_halo.py-395-    finally:
--
./scripts/cluster/scaling_derecho/mc_nccl_probe.py-83-    mesh = Mesh(np.array(gdev), ("lat",))
./scripts/cluster/scaling_derecho/mc_nccl_probe.py-84-    x = jnp.arange(n * 8, dtype=jnp.float32).reshape(n, 8)
./scripts/cluster/scaling_derecho/mc_nccl_probe.py:85:    xs = jax.device_put(x, NamedSharding(mesh, P("lat", None)))
./scripts/cluster/scaling_derecho/mc_nccl_probe.py-86-    perm = [(i, (i + 1) % n) for i in range(n)]
./scripts/cluster/scaling_derecho/mc_nccl_probe.py-87-
--
./scripts/cluster/scaling_derecho/mc_nccl_probe.py-114-
./scripts/cluster/scaling_derecho/mc_nccl_probe.py-115-    # Stage 5: ppermute latency (small message — the halo latency regime).
./scripts/cluster/scaling_derecho/mc_nccl_probe.py:116:    small = jax.device_put(
./scripts/cluster/scaling_derecho/mc_nccl_probe.py-117-        jnp.ones((n, 64), jnp.float32), NamedSharding(mesh, P("lat", None)))
./scripts/cluster/scaling_derecho/mc_nccl_probe.py-118-    g = jax.jit(shard_map(lambda a: jax.lax.ppermute(a, "lat", perm),
--
./packages/core/legoesm/parallel/mesh.py-747-# ==============================================================================
./packages/core/legoesm/parallel/mesh.py-748-
./packages/core/legoesm/parallel/mesh.py:749:def multiprocess_safe_device_put(leaf, sharding):
./packages/core/legoesm/parallel/mesh.py:750:    """``jax.device_put`` that is safe under multi-controller SPMD.
./packages/core/legoesm/parallel/mesh.py-751-
./packages/core/legoesm/parallel/mesh.py:752:    ``jax.device_put(x, sharding)`` with a sharding that spans processes
./packages/core/legoesm/parallel/mesh.py-753-    ASSERTS the value is bit-identical on every process.  Per-process XLA
./packages/core/legoesm/parallel/mesh.py-754-    autotuning can legitimately pick different kernels on different nodes,
--
./packages/core/legoesm/parallel/mesh.py-762-
./packages/core/legoesm/parallel/mesh.py-763-    Single-process (and non-array leaves) delegate to plain
./packages/core/legoesm/parallel/mesh.py:764:    ``jax.device_put`` — byte-identical behavior to before.
./packages/core/legoesm/parallel/mesh.py-765-    Already-global (non-fully-addressable) leaves pass through unchanged.
./packages/core/legoesm/parallel/mesh.py-766-    """
./packages/core/legoesm/parallel/mesh.py-767-    if not isinstance(leaf, (jax.Array, jnp.ndarray)):
./packages/core/legoesm/parallel/mesh.py:768:        return jax.device_put(leaf, sharding)
./packages/core/legoesm/parallel/mesh.py-769-    if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable:
./packages/core/legoesm/parallel/mesh.py-770-        return leaf  # already a global sharded array; nothing to place
--
./packages/core/legoesm/parallel/mesh.py-774-        return jax.make_array_from_callback(
./packages/core/legoesm/parallel/mesh.py-775-            host.shape, sharding, lambda idx: host[idx])
./packages/core/legoesm/parallel/mesh.py:776:    return jax.device_put(leaf, sharding)
./packages/core/legoesm/parallel/mesh.py-777-
./packages/core/legoesm/parallel/mesh.py-778-
--
./packages/core/legoesm/parallel/mesh.py-819-                if config.tiling != (1, 1) and leaf.ndim < 3:
./packages/core/legoesm/parallel/mesh.py-820-                    sharding = tiled_face_only or config.face_sharding
./packages/core/legoesm/parallel/mesh.py:821:                    return multiprocess_safe_device_put(leaf, sharding)
./packages/core/legoesm/parallel/mesh.py-822-                if config.tiling != (1, 1) and leaf.ndim >= 3:
./packages/core/legoesm/parallel/mesh.py-823-                    # STAGGERED face-plane leaves — D-grid winds
--
./packages/core/legoesm/parallel/mesh.py-832-                    if leaf.shape[1] % tx != 0 or leaf.shape[2] % ty != 0:
./packages/core/legoesm/parallel/mesh.py-833-                        sharding = tiled_face_only or config.face_sharding
./packages/core/legoesm/parallel/mesh.py:834:                        return multiprocess_safe_device_put(leaf, sharding)
./packages/core/legoesm/parallel/mesh.py:835:                return multiprocess_safe_device_put(leaf, config.face_sharding)
./packages/core/legoesm/parallel/mesh.py:836:            return multiprocess_safe_device_put(leaf, config.replicated_sharding)
./packages/core/legoesm/parallel/mesh.py-837-
./packages/core/legoesm/parallel/mesh.py-838-        elif config.grid_type == "cubed_sphere_level":
--
./packages/core/legoesm/parallel/mesh.py-844-            # shards there (see ``build_physics_pipeline`` for the
./packages/core/legoesm/parallel/mesh.py-845-            # column-mesh construction).
./packages/core/legoesm/parallel/mesh.py:846:            return multiprocess_safe_device_put(leaf, config.replicated_sharding)
./packages/core/legoesm/parallel/mesh.py-847-
./packages/core/legoesm/parallel/mesh.py-848-        elif config.grid_type == "latlon":
./packages/core/legoesm/parallel/mesh.py-849-            if leaf.ndim >= 2:
./packages/core/legoesm/parallel/mesh.py:850:                return multiprocess_safe_device_put(leaf, config.face_sharding)
./packages/core/legoesm/parallel/mesh.py:851:            return multiprocess_safe_device_put(leaf, config.replicated_sharding)
./packages/core/legoesm/parallel/mesh.py-852-
./packages/core/legoesm/parallel/mesh.py-853-        elif config.grid_type == "spectral":
--
./packages/core/legoesm/parallel/mesh.py-858-                    config.mesh, P(None, "level"),
./packages/core/legoesm/parallel/mesh.py-859-                )
./packages/core/legoesm/parallel/mesh.py:860:                return multiprocess_safe_device_put(leaf, level_axis_sharding)
./packages/core/legoesm/parallel/mesh.py-861-            if leaf.ndim == 1:
./packages/core/legoesm/parallel/mesh.py-862-                # 1D arrays (e.g., lnps_hat): replicate across devices.
./packages/core/legoesm/parallel/mesh.py:863:                return multiprocess_safe_device_put(leaf, config.replicated_sharding)
./packages/core/legoesm/parallel/mesh.py:864:            return multiprocess_safe_device_put(leaf, config.replicated_sharding)
./packages/core/legoesm/parallel/mesh.py-865-
./packages/core/legoesm/parallel/mesh.py-866-        elif config.grid_type == "voronoi":
--
./packages/core/legoesm/parallel/mesh.py-870-                nCells, nEdges, _nVerts = config.voronoi_dims
./packages/core/legoesm/parallel/mesh.py-871-                if leaf.shape[0] in (nCells, nEdges):
./packages/core/legoesm/parallel/mesh.py:872:                    return multiprocess_safe_device_put(leaf, config.face_sharding)
./packages/core/legoesm/parallel/mesh.py:873:            return multiprocess_safe_device_put(leaf, config.replicated_sharding)
./packages/core/legoesm/parallel/mesh.py-874-
./packages/core/legoesm/parallel/mesh.py-875-        # Default: replicate
./packages/core/legoesm/parallel/mesh.py:876:        return multiprocess_safe_device_put(leaf, config.replicated_sharding)
./packages/core/legoesm/parallel/mesh.py-877-
./packages/core/legoesm/parallel/mesh.py-878-    return jax.tree.map(_shard_leaf, pytree)
--
./packages/core/legoesm/parallel/mesh.py-1092-            return leaf
./packages/core/legoesm/parallel/mesh.py-1093-        if leaf.ndim >= 1 and leaf.shape[0] == nlev:
./packages/core/legoesm/parallel/mesh.py:1094:            return jax.device_put(leaf, config.face_sharding)
./packages/core/legoesm/parallel/mesh.py:1095:        return jax.device_put(leaf, config.replicated_sharding)
./packages/core/legoesm/parallel/mesh.py-1096-
./packages/core/legoesm/parallel/mesh.py-1097-    return jax.tree.map(_shard_leaf, pytree)
--
./packages/core/legoesm/parallel/mesh.py-1120-        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
./packages/core/legoesm/parallel/mesh.py-1121-            return leaf
./packages/core/legoesm/parallel/mesh.py:1122:        return jax.device_put(leaf, config.replicated_sharding)
./packages/core/legoesm/parallel/mesh.py-1123-
./packages/core/legoesm/parallel/mesh.py-1124-    return jax.tree.map(_replicate_leaf, pytree)
--
./packages/core/legoesm/parallel/metal.py-106-        cpu_device = jax.devices("cpu")[0]
./packages/core/legoesm/parallel/metal.py-107-        return SpectralDevicePlacement(
./packages/core/legoesm/parallel/metal.py:108:            grid=jax.device_put(grid, cpu_device),
./packages/core/legoesm/parallel/metal.py-109-            use_cpu_for_spectral=True,
./packages/core/legoesm/parallel/metal.py-110-            cpu_device=cpu_device,
--
./packages/core/legoesm/parallel/metal.py-133-    """
./packages/core/legoesm/parallel/metal.py-134-    cpu = jax.devices("cpu")[0]
./packages/core/legoesm/parallel/metal.py:135:    return jax.device_put(array, cpu)
./packages/core/legoesm/parallel/metal.py-136-
./packages/core/legoesm/parallel/metal.py-137-
--
./packages/core/legoesm/parallel/metal.py-154-        return array
./packages/core/legoesm/parallel/metal.py-155-    metal = jax.devices()[0]
./packages/core/legoesm/parallel/metal.py:156:    return jax.device_put(array, metal)
./packages/core/legoesm/parallel/metal.py-157-
./packages/core/legoesm/parallel/metal.py-158-
--
./packages/core/legoesm/parallel/metal.py-170-    """
./packages/core/legoesm/parallel/metal.py-171-    cpu = jax.devices("cpu")[0]
./packages/core/legoesm/parallel/metal.py:172:    return jax.device_put(pytree, cpu)
./packages/core/legoesm/parallel/metal.py-173-
./packages/core/legoesm/parallel/metal.py-174-
--
./packages/core/legoesm/parallel/metal.py-190-    if default is None:
./packages/core/legoesm/parallel/metal.py-191-        default = jax.devices()[0]
./packages/core/legoesm/parallel/metal.py:192:    return jax.device_put(pytree, default)
./packages/core/legoesm/parallel/metal.py-193-
./packages/core/legoesm/parallel/metal.py-194-
--
./packages/core/legoesm/parallel/metal.py-215-    if _is_on_device(x, device):
./packages/core/legoesm/parallel/metal.py-216-        return x
./packages/core/legoesm/parallel/metal.py:217:    return jax.device_put(x, device)
./packages/core/legoesm/parallel/metal.py-218-
./packages/core/legoesm/parallel/metal.py-219-
--
./packages/core/legoesm/parallel/column_shard.py-131-    """
./packages/core/legoesm/parallel/column_shard.py-132-    if field.ndim == 0:
./packages/core/legoesm/parallel/column_shard.py:133:        return jax.device_put(field, NamedSharding(mesh, P()))
./packages/core/legoesm/parallel/column_shard.py-134-    trailing = (None,) * (field.ndim - 1)
./packages/core/legoesm/parallel/column_shard.py:135:    return jax.device_put(field, NamedSharding(mesh, P("col", *trailing)))
./packages/core/legoesm/parallel/column_shard.py-136-
./packages/core/legoesm/parallel/column_shard.py-137-
./packages/core/legoesm/parallel/column_shard.py-138-def replicate(field: jax.Array, mesh: Mesh) -> jax.Array:
./packages/core/legoesm/parallel/column_shard.py-139-    """Place ``field`` replicated across every device in ``mesh``."""
./packages/core/legoesm/parallel/column_shard.py:140:    return jax.device_put(field, NamedSharding(mesh, P()))
--
./packages/core/legoesm/parallel/tiled_production_cdgrid.py-2520-
./packages/core/legoesm/parallel/tiled_production_cdgrid.py-2521-    sh = NamedSharding(mesh, P(*AXES))
./packages/core/legoesm/parallel/tiled_production_cdgrid.py:2522:    dummy = jax.device_put(jnp.zeros((6, kt, kt), dtype=jnp.float32), sh)
./packages/core/legoesm/parallel/tiled_production_cdgrid.py-2523-
./packages/core/legoesm/parallel/tiled_production_cdgrid.py-2524-    @partial(shard_map, mesh=mesh, in_specs=P(*AXES), out_specs=P(*AXES),
--
./packages/core/legoesm/parallel/sharded_dynamics.py-239-                    pspec = spec.tiled_3d if leaf.ndim >= 4 else spec.tiled_2d
./packages/core/legoesm/parallel/sharded_dynamics.py-240-                    if pspec is not None:
./packages/core/legoesm/parallel/sharded_dynamics.py:241:                        return jax.device_put(
./packages/core/legoesm/parallel/sharded_dynamics.py-242-                            leaf, NamedSharding(mesh, pspec)
./packages/core/legoesm/parallel/sharded_dynamics.py-243-                        )
./packages/core/legoesm/parallel/sharded_dynamics.py-244-                pspec = spec.face_3d if leaf.ndim >= 4 else spec.face_2d
./packages/core/legoesm/parallel/sharded_dynamics.py:245:                return jax.device_put(leaf, NamedSharding(mesh, pspec))
./packages/core/legoesm/parallel/sharded_dynamics.py:246:            return jax.device_put(
./packages/core/legoesm/parallel/sharded_dynamics.py-247-                leaf, NamedSharding(mesh, spec.replicated)
./packages/core/legoesm/parallel/sharded_dynamics.py-248-            )
--
./packages/core/legoesm/parallel/sharded_dynamics.py-251-            if leaf.ndim >= 2:
./packages/core/legoesm/parallel/sharded_dynamics.py-252-                pspec = P("lat", *([None] * (leaf.ndim - 1)))
./packages/core/legoesm/parallel/sharded_dynamics.py:253:                return jax.device_put(leaf, NamedSharding(mesh, pspec))
./packages/core/legoesm/parallel/sharded_dynamics.py:254:            return jax.device_put(
./packages/core/legoesm/parallel/sharded_dynamics.py-255-                leaf, NamedSharding(mesh, spec.replicated)
./packages/core/legoesm/parallel/sharded_dynamics.py-256-            )
./packages/core/legoesm/parallel/sharded_dynamics.py-257-
./packages/core/legoesm/parallel/sharded_dynamics.py:258:        return jax.device_put(
./packages/core/legoesm/parallel/sharded_dynamics.py-259-            leaf, NamedSharding(mesh, spec.replicated)
./packages/core/legoesm/parallel/sharded_dynamics.py-260-        )
--
./packages/core/legoesm/parallel/sharded_dynamics.py-288-        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
./packages/core/legoesm/parallel/sharded_dynamics.py-289-            return leaf
./packages/core/legoesm/parallel/sharded_dynamics.py:290:        return jax.device_put(leaf, replicated)
./packages/core/legoesm/parallel/sharded_dynamics.py-291-
./packages/core/legoesm/parallel/sharded_dynamics.py-292-    return jax.tree.map(_gather, state)
--
./packages/core/legoesm/parallel/sharded_dynamics.py-2028-    from legoesm.core.precision import cast_pytree
./packages/core/legoesm/parallel/sharded_dynamics.py-2029-    from legoesm.core.state import MPASHydrostaticState
./packages/core/legoesm/parallel/sharded_dynamics.py:2030:    from legoesm.parallel.mesh import multiprocess_safe_device_put
./packages/core/legoesm/parallel/sharded_dynamics.py-2031-    from legoesm.parallel.shard_map_compat import shard_map
./packages/core/legoesm/parallel/sharded_dynamics.py-2032-    from legoesm.timestepping.dispatch import dispatch_integrator
--
./packages/core/legoesm/parallel/sharded_dynamics.py-2114-    # (multi-controller-safe: a sharded jit ARG is legal where a sharded
./packages/core/legoesm/parallel/sharded_dynamics.py-2115-    # CLOSURE constant raises at trace time under jax.distributed;
./packages/core/legoesm/parallel/sharded_dynamics.py:2116:    # ``multiprocess_safe_device_put`` builds the global array from each
./packages/core/legoesm/parallel/sharded_dynamics.py-2117-    # process's local copy).  All leaves are arrays after the jnp.stack
./packages/core/legoesm/parallel/sharded_dynamics.py-2118-    # in _build_voronoi_partition_infra (ints become (n_dev,) arrays).
./packages/core/legoesm/parallel/sharded_dynamics.py-2119-    dev_sharding = dev_config.face_sharding  # P("device") on axis 0
./packages/core/legoesm/parallel/sharded_dynamics.py-2120-    stacked_meshes = jax.tree.map(
./packages/core/legoesm/parallel/sharded_dynamics.py:2121:        lambda x: multiprocess_safe_device_put(x, dev_sharding),
./packages/core/legoesm/parallel/sharded_dynamics.py-2122-        stacked_meshes,
./packages/core/legoesm/parallel/sharded_dynamics.py-2123-    )
--
./packages/core/legoesm/parallel/sharded_dynamics.py-2147-        halo_args = tuple(
./packages/core/legoesm/parallel/sharded_dynamics.py-2148-            (
./packages/core/legoesm/parallel/sharded_dynamics.py:2149:                multiprocess_safe_device_put(
./packages/core/legoesm/parallel/sharded_dynamics.py-2150-                    pp_sched['send_cell_idx'][r], dev_sharding),
./packages/core/legoesm/parallel/sharded_dynamics.py:2151:                multiprocess_safe_device_put(
./packages/core/legoesm/parallel/sharded_dynamics.py-2152-                    pp_sched['recv_cell_pos'][r], dev_sharding),
./packages/core/legoesm/parallel/sharded_dynamics.py:2153:                multiprocess_safe_device_put(
./packages/core/legoesm/parallel/sharded_dynamics.py-2154-                    pp_sched['send_edge_idx'][r], dev_sharding),
./packages/core/legoesm/parallel/sharded_dynamics.py:2155:                multiprocess_safe_device_put(
./packages/core/legoesm/parallel/sharded_dynamics.py-2156-                    pp_sched['recv_edge_pos'][r], dev_sharding),
./packages/core/legoesm/parallel/sharded_dynamics.py-2157-            )
--
./packages/core/legoesm/parallel/sharded_dynamics.py-2184-        # ---- Legacy all-gather strategy: local gather indices ----
./packages/core/legoesm/parallel/sharded_dynamics.py-2185-        halo_args = (
./packages/core/legoesm/parallel/sharded_dynamics.py:2186:            multiprocess_safe_device_put(gather_cells, dev_sharding),
./packages/core/legoesm/parallel/sharded_dynamics.py:2187:            multiprocess_safe_device_put(gather_edges, dev_sharding),
./packages/core/legoesm/parallel/sharded_dynamics.py-2188-        )
./packages/core/legoesm/parallel/sharded_dynamics.py-2189-
--
./packages/core/legoesm/parallel/sharded_dynamics.py-2313-        # constant.  Take a HOST copy first (codex M3c-1 MAJOR): the
./packages/core/legoesm/parallel/sharded_dynamics.py-2314-        # caller's mesh may arrive REPLICATED (bench replicate_pytree),
./packages/core/legoesm/parallel/sharded_dynamics.py:2315:        # and multiprocess_safe_device_put passes non-fully-addressable
./packages/core/legoesm/parallel/sharded_dynamics.py-2316-        # arrays through UNCHANGED — a replicated leaf would silently
./packages/core/legoesm/parallel/sharded_dynamics.py-2317-        # stay replicated under multi-controller.  A host array is
./packages/core/legoesm/parallel/sharded_dynamics.py-2318-        # always fully addressable, so the P("device") shard is
./packages/core/legoesm/parallel/sharded_dynamics.py-2319-        # guaranteed on both controllers.  total_area is a host float.
./packages/core/legoesm/parallel/sharded_dynamics.py:2320:        _area_for_mass = multiprocess_safe_device_put(
./packages/core/legoesm/parallel/sharded_dynamics.py-2321-            np.asarray(global_mesh.areaCell), dev_sharding)
./packages/core/legoesm/parallel/sharded_dynamics.py-2322-        # fp64 area sum to match the fp64 mass-budget accumulator below
--
./packages/core/legoesm/parallel/sharded_dynamics.py-2576-    ``create_voronoi_device_mesh(n_devices=1)`` shape) pass through
./packages/core/legoesm/parallel/sharded_dynamics.py-2577-    unchanged.  Multi-device single-PROCESS configs take the plain
./packages/core/legoesm/parallel/sharded_dynamics.py:2578:    ``device_put`` branch of ``replicate_leaf`` (cheap, no collective).
./packages/core/legoesm/parallel/sharded_dynamics.py-2579-    """
./packages/core/legoesm/parallel/sharded_dynamics.py-2580-    if dev_config.mesh is None or dev_config.replicated_sharding is None:
--
./packages/core/legoesm/parallel/ensemble.py-496-    def shard_leaf(x):
./packages/core/legoesm/parallel/ensemble.py-497-        if x.ndim == 0:
./packages/core/legoesm/parallel/ensemble.py:498:            return jax.device_put(x, replicated)
./packages/core/legoesm/parallel/ensemble.py:499:        return jax.device_put(x, sharding)
./packages/core/legoesm/parallel/ensemble.py-500-
./packages/core/legoesm/parallel/ensemble.py-501-    return jax.tree.map(shard_leaf, batched_state)
--
./packages/core/legoesm/parallel/ensemble.py-520-    # measurable overhead at each gather call.
./packages/core/legoesm/parallel/ensemble.py-521-    dev0 = jax.devices()[0]
./packages/core/legoesm/parallel/ensemble.py:522:    return jax.tree.map(lambda x: jax.device_put(x, dev0), sharded_state)
./packages/core/legoesm/parallel/ensemble.py-523-
./packages/core/legoesm/parallel/ensemble.py-524-
--
./packages/core/legoesm/parallel/latlon_spmd.py-271-    (``gather_state_atm_latlon`` / ``gather_state_latlon``).
./packages/core/legoesm/parallel/latlon_spmd.py-272-
./packages/core/legoesm/parallel/latlon_spmd.py:273:    Single-process: plain ``jax.device_put`` (the historical path, unchanged).
./packages/core/legoesm/parallel/latlon_spmd.py-274-    Multi-controller (``jax.process_count() > 1``, route-B ``jax.distributed``):
./packages/core/legoesm/parallel/latlon_spmd.py:275:    a top-level ``device_put`` cannot reshard an array whose shards live on
./packages/core/legoesm/parallel/latlon_spmd.py-276-    other processes' devices, so the replication runs as a jit-compiled
./packages/core/legoesm/parallel/latlon_spmd.py-277-    identity with replicated ``out_shardings`` — the supported cross-process
--
./packages/core/legoesm/parallel/latlon_spmd.py-290-    if multiprocess:
./packages/core/legoesm/parallel/latlon_spmd.py-291-        return jax.jit(lambda a: a, out_shardings=rep)(arr)
./packages/core/legoesm/parallel/latlon_spmd.py:292:    return jax.device_put(arr, rep)
./packages/core/legoesm/parallel/latlon_spmd.py-293-
./packages/core/legoesm/parallel/latlon_spmd.py-294-
--
./packages/core/legoesm/parallel/latlon_spmd.py-296-    """Scatter one full-global leaf onto ``sharding``'s mesh — the SCATTER
./packages/core/legoesm/parallel/latlon_spmd.py-297-    primitive symmetric to :func:`replicate_leaf`, shared by the atm and ocean
./packages/core/legoesm/parallel/latlon_spmd.py:298:    lat-band SPMD steps (``shard_state_atm_latlon`` / ``shard_state_latlon``).
./packages/core/legoesm/parallel/latlon_spmd.py-299-
./packages/core/legoesm/parallel/latlon_spmd.py:300:    Single-process: plain ``jax.device_put`` (the historical path, unchanged and
./packages/core/legoesm/parallel/latlon_spmd.py-301-    byte-identical).
./packages/core/legoesm/parallel/latlon_spmd.py-302-
./packages/core/legoesm/parallel/latlon_spmd.py-303-    Multi-controller (``jax.process_count() > 1``, route-B ``jax.distributed``,
./packages/core/legoesm/parallel/latlon_spmd.py:304:    a mesh spanning processes): a top-level ``jax.device_put`` of the FULL global
./packages/core/legoesm/parallel/latlon_spmd.py-305-    array to a cross-process ``NamedSharding`` cannot place shards on peer
./packages/core/legoesm/parallel/latlon_spmd.py-306-    processes' devices, so XLA falls back to an all-gather that (a) transiently
--
./packages/core/legoesm/parallel/latlon_spmd.py-315-
./packages/core/legoesm/parallel/latlon_spmd.py-316-    NOT differentiable: ``make_array_from_callback`` is a host construction API,
./packages/core/legoesm/parallel/latlon_spmd.py:317:    so (unlike the historical ``device_put``) a ``jax.grad``/``vjp`` cannot be
./packages/core/legoesm/parallel/latlon_spmd.py-318-    taken THROUGH the multiprocess scatter. This is fine — the scatter is an
./packages/core/legoesm/parallel/latlon_spmd.py-319-    init-time boundary (``scatter_to_local`` before the step loop, per the MPI
--
./packages/core/legoesm/parallel/latlon_spmd.py-329-    """
./packages/core/legoesm/parallel/latlon_spmd.py-330-    if not multiprocess:
./packages/core/legoesm/parallel/latlon_spmd.py:331:        return jax.device_put(arr, sharding)
./packages/core/legoesm/parallel/latlon_spmd.py-332-    return jax.make_array_from_callback(arr.shape, sharding, lambda idx: arr[idx])
./packages/core/legoesm/parallel/latlon_spmd.py-333-
--
./packages/ml/legoesm/training/scale_build.py-138-    #1286 fix A: the per-run ``samples`` list is kept host-resident so the whole
./packages/ml/legoesm/training/scale_build.py-139-    training set is NOT parked in device memory; the training loop
./packages/ml/legoesm/training/scale_build.py:140:    ``device_put``s one sample at a time.  ``jax.device_get`` converts every
./packages/ml/legoesm/training/scale_build.py-141-    device-array leaf to numpy and passes non-array leaves through unchanged;
./packages/ml/legoesm/training/scale_build.py-142-    the build's transient device allocation is freed once this returns.
--
./packages/ml/legoesm/training/scale_build.py-535-    #1286: with ``nproc > 1`` this builds ONLY ``rank``'s contiguous shard (fix
./packages/ml/legoesm/training/scale_build.py-536-    B — no global materialization); with ``host_resident`` each built sample is
./packages/ml/legoesm/training/scale_build.py:537:    moved off-device to host numpy (fix A — the loop ``device_put``s per batch).
./packages/ml/legoesm/training/scale_build.py-538-    """
./packages/ml/legoesm/training/scale_build.py-539-    import jax.numpy as jnp
--
./tests/parallel/test_tiled_fix_ps_mass.py-59-    stage = make_tiled_fix_ps_mass_stage_2d(mesh, grid, N, KT)
./tests/parallel/test_tiled_fix_ps_mass.py-60-    fo = NamedSharding(mesh, P("face", None, None))
./tests/parallel/test_tiled_fix_ps_mass.py:61:    t = np.asarray(stage(jax.device_put(p_s_new, fo), jax.device_put(p_s_old, fo)))
./tests/parallel/test_tiled_fix_ps_mass.py-62-
./tests/parallel/test_tiled_fix_ps_mass.py-63-    # (a) match the global op (psum reorders the global scalars -> ULP).
--
./tests/parallel/test_conservation_spmd_lat_reduction.py-63-
./tests/parallel/test_conservation_spmd_lat_reduction.py-64-    isp = P("lat", None)
./tests/parallel/test_conservation_spmd_lat_reduction.py:65:    a_sh = jax.device_put(jnp.asarray(area), NamedSharding(mesh, isp))
./tests/parallel/test_conservation_spmd_lat_reduction.py:66:    f1_sh = jax.device_put(jnp.asarray(f1), NamedSharding(mesh, isp))
./tests/parallel/test_conservation_spmd_lat_reduction.py:67:    f2_sh = jax.device_put(jnp.asarray(f2), NamedSharding(mesh, isp))
./tests/parallel/test_conservation_spmd_lat_reduction.py-68-
./tests/parallel/test_conservation_spmd_lat_reduction.py-69-    activate_latlon_spmd_halo(mesh)  # backend -> 'spmd', mesh -> this 'lat' mesh
--
./tests/parallel/test_conservation_spmd_lat_reduction.py-109-
./tests/parallel/test_conservation_spmd_lat_reduction.py-110-    isp = P("lat", None)
./tests/parallel/test_conservation_spmd_lat_reduction.py:111:    a_sh = jax.device_put(jnp.asarray(area), NamedSharding(mesh, isp))
./tests/parallel/test_conservation_spmd_lat_reduction.py:112:    f_sh = jax.device_put(jnp.asarray(field), NamedSharding(mesh, isp))
./tests/parallel/test_conservation_spmd_lat_reduction.py-113-
./tests/parallel/test_conservation_spmd_lat_reduction.py-114-    activate_latlon_spmd_halo(mesh)
--
./tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py-180-    fw5 = NamedSharding(mesh, P("face", None, None, None, None))
./tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py-181-    ut, vt, Tt, pst, qt = step(
./tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py:182:        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
./tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py:183:        jax.device_put(T, fw), jax.device_put(p_s, fo),
./tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py:184:        jax.device_put(phis, fo), jax.device_put(q_pack, fw5))
./tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py-185-
./tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py-186-    nl = N // KT
--
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-164-
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-165-    # Multi-controller idiom: the SAME host-global array is built identically
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:166:    # on every process (deterministic np.arange), and device_put slices out
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-167-    # each process's addressable shards (matches test_atm_latlon_spmd_step.py
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-168-    # and shard_state_atm_latlon; no cross-process transfer of the global
--
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-170-    g0 = (1.0 + np.arange(N_GLOBAL_DEVICES * NH * NH, dtype=np.float64)
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-171-          ).reshape(*MESH_SHAPE, NH, NH)
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:172:    x0 = jax.device_put(g0, NamedSharding(mesh, P(*AXES)))
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-173-    # ``run`` returns a fully-REPLICATED (out_specs=P()) global scalar; read
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-174-    # THIS process's local replica directly (unambiguously addressable) rather
--
./tests/parallel/test_cubesphere_exchange.py-30-    spec = P("face", *((None,) * (data.ndim - 1)))
./tests/parallel/test_cubesphere_exchange.py-31-    sharding = jax.sharding.NamedSharding(mesh, spec)
./tests/parallel/test_cubesphere_exchange.py:32:    return jax.device_put(data, sharding)
./tests/parallel/test_cubesphere_exchange.py-33-
./tests/parallel/test_cubesphere_exchange.py-34-
--
./tests/parallel/test_tiled_cgrid_gradient.py-64-    co = P("face", "tile_i", "tile_j")
./tests/parallel/test_tiled_cgrid_gradient.py-65-    fo = P("face", None, None)
./tests/parallel/test_tiled_cgrid_gradient.py:66:    eta_sh = jax.device_put(eta, NamedSharding(mesh, co))   # tile-sharded
./tests/parallel/test_tiled_cgrid_gradient.py:67:    rdxc_sh = jax.device_put(cdg.rdxc, NamedSharding(mesh, fo))
./tests/parallel/test_tiled_cgrid_gradient.py:68:    rdyc_sh = jax.device_put(cdg.rdyc, NamedSharding(mesh, fo))
./tests/parallel/test_tiled_cgrid_gradient.py-69-    offs_j = jnp.asarray(offs)
./tests/parallel/test_tiled_cgrid_gradient.py-70-
--
./packages/ml/legoesm/training/neural_gcm_spectral.py-1671-    the explicit-placement hygiene: it makes each sample's single H2D copy
./packages/ml/legoesm/training/neural_gcm_spectral.py-1672-    visible at the call site, keeps behavior identical if the dataset ever
./packages/ml/legoesm/training/neural_gcm_spectral.py:1673:    arrives COMMITTED (e.g. an explicit ``device_put(cpu)`` producer), and
./packages/ml/legoesm/training/neural_gcm_spectral.py-1674-    bounds peak device footprint to one sample. Wraps the #985
./packages/ml/legoesm/training/neural_gcm_spectral.py-1675-    ``_stage_tree``. Uses ``jax.local_devices()`` (not ``jax.devices()``)
--
./packages/ml/legoesm/training/neural_gcm_spectral.py-3091-
./packages/ml/legoesm/training/neural_gcm_spectral.py-3092-def _stage_tree(tree, device):
./packages/ml/legoesm/training/neural_gcm_spectral.py:3093:    """``device_put`` only the ARRAY leaves of ``tree`` onto ``device``, leaving
./packages/ml/legoesm/training/neural_gcm_spectral.py-3094-    None / metadata leaves untouched.
./packages/ml/legoesm/training/neural_gcm_spectral.py-3095-
./packages/ml/legoesm/training/neural_gcm_spectral.py-3096-    A chunk is ``(ic_states, targets, forcings)`` of pytrees whose leaves are
./packages/ml/legoesm/training/neural_gcm_spectral.py-3097-    mostly arrays but not exclusively (``forcings`` may be ``None``); a bare
./packages/ml/legoesm/training/neural_gcm_spectral.py:3098:    ``jax.device_put(tree, device)`` chokes on a non-array leaf, so filter to
./packages/ml/legoesm/training/neural_gcm_spectral.py-3099-    arrays via ``eqx.is_array``. Used to host-stage prefetched chunks (#985).
./packages/ml/legoesm/training/neural_gcm_spectral.py-3100-    """
./packages/ml/legoesm/training/neural_gcm_spectral.py-3101-    return jax.tree_util.tree_map(
./packages/ml/legoesm/training/neural_gcm_spectral.py:3102:        lambda x: jax.device_put(x, device) if eqx.is_array(x) else x, tree
./packages/ml/legoesm/training/neural_gcm_spectral.py-3103-    )
./packages/ml/legoesm/training/neural_gcm_spectral.py-3104-
--
./packages/ml/legoesm/training/neural_gcm_spectral.py-3140-    # distinction is the fix: a T106 4xA100 run OOM'd because the prefetched chunk
./packages/ml/legoesm/training/neural_gcm_spectral.py-3141-    # rode GPU memory during the current chunk's training peak, and a post-hoc
./packages/ml/legoesm/training/neural_gcm_spectral.py:3142:    # device_put(cpu) does NOT help (the arrays are constructed on the GPU first).
./packages/ml/legoesm/training/neural_gcm_spectral.py-3143-    # ``jax.default_device`` is thread-local, so the producer thread builds on the
./packages/ml/legoesm/training/neural_gcm_spectral.py-3144-    # CPU while the main thread trains on the GPU. Each SAMPLE is then moved onto
--
./packages/ml/legoesm/training/neural_gcm_spectral.py-3187-
./packages/ml/legoesm/training/neural_gcm_spectral.py-3188-    # True when chunks are built host-staged (prefetch + CPU backend): the
./packages/ml/legoesm/training/neural_gcm_spectral.py:3189:    # training loop must then device_put each SAMPLE onto the compute device at
./packages/ml/legoesm/training/neural_gcm_spectral.py-3190-    # the point of use (#985).
./packages/ml/legoesm/training/neural_gcm_spectral.py-3191-    _chunks.host_staged = _host_dev is not None
--
./tests/parallel/test_latlon_spmd_halo.py-59-
./tests/parallel/test_latlon_spmd_halo.py-60-    isp = P("lat", *((None,) * (field.ndim - 1)))
./tests/parallel/test_latlon_spmd_halo.py:61:    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
./tests/parallel/test_latlon_spmd_halo.py-62-    out = np.asarray(pad_halo_latlon_band_spmd(mesh, halo=1, negate=negate)(field_sh))
./tests/parallel/test_latlon_spmd_halo.py-63-
--
./tests/parallel/test_latlon_spmd_halo.py-95-
./tests/parallel/test_latlon_spmd_halo.py-96-    isp = P("lat", None)
./tests/parallel/test_latlon_spmd_halo.py:97:    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
./tests/parallel/test_latlon_spmd_halo.py-98-    activate_latlon_spmd_halo(mesh)
./tests/parallel/test_latlon_spmd_halo.py-99-    try:
--
./tests/parallel/test_latlon_spmd_halo.py-145-
./tests/parallel/test_latlon_spmd_halo.py-146-    isp = P("lat", *((None,) * (field.ndim - 1)))
./tests/parallel/test_latlon_spmd_halo.py:147:    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
./tests/parallel/test_latlon_spmd_halo.py-148-    body = make_latlon_band_wall_pad_body(
./tests/parallel/test_latlon_spmd_halo.py-149-        mesh, halo=1, south_value=south_value, north_value=north_value)
--
./tests/parallel/test_latlon_spmd_halo.py-194-
./tests/parallel/test_latlon_spmd_halo.py-195-    isp = P("lat", None)
./tests/parallel/test_latlon_spmd_halo.py:196:    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
./tests/parallel/test_latlon_spmd_halo.py-197-    activate_latlon_spmd_halo(mesh)
./tests/parallel/test_latlon_spmd_halo.py-198-    try:
--
./tests/parallel/test_latlon_spmd_halo.py-241-
./tests/parallel/test_latlon_spmd_halo.py-242-    isp = P("lat", None)
./tests/parallel/test_latlon_spmd_halo.py:243:    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
./tests/parallel/test_latlon_spmd_halo.py-244-    activate_latlon_spmd_halo(mesh)
./tests/parallel/test_latlon_spmd_halo.py-245-    try:
--
./tests/parallel/test_tiled_cgrid_divergence.py-100-    face_only = NamedSharding(mesh, P("face", None, None))
./tests/parallel/test_tiled_cgrid_divergence.py-101-    centered = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
./tests/parallel/test_tiled_cgrid_divergence.py:102:    u_sh = jax.device_put(u_c, face_only)
./tests/parallel/test_tiled_cgrid_divergence.py:103:    v_sh = jax.device_put(v_c, face_only)
./tests/parallel/test_tiled_cgrid_divergence.py:104:    dyx_sh = jax.device_put(cdg.dy_edge_x, face_only)
./tests/parallel/test_tiled_cgrid_divergence.py:105:    dxy_sh = jax.device_put(cdg.dx_edge_y, face_only)
./tests/parallel/test_tiled_cgrid_divergence.py:106:    area_sh = jax.device_put(cdg.base.area, centered)
./tests/parallel/test_tiled_cgrid_divergence.py-107-
./tests/parallel/test_tiled_cgrid_divergence.py-108-    fo = P("face", None, None)
--
./tests/parallel/test_latlon_vface_reconstruct.py-54-
./tests/parallel/test_latlon_vface_reconstruct.py-55-    isp = P("lat", *((None,) * (v_lower.ndim - 1)))
./tests/parallel/test_latlon_vface_reconstruct.py:56:    vl_sh = jax.device_put(v_lower, NamedSharding(mesh, isp))
./tests/parallel/test_latlon_vface_reconstruct.py-57-
./tests/parallel/test_latlon_vface_reconstruct.py-58-    @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp, check_vma=False)
--
./tests/parallel/test_latlon_vface_reconstruct.py-85-    v_lower = jnp.asarray(v_full[:N_LAT])
./tests/parallel/test_latlon_vface_reconstruct.py-86-    isp = P("lat", *((None,) * (v_lower.ndim - 1)))
./tests/parallel/test_latlon_vface_reconstruct.py:87:    vl_sh = jax.device_put(v_lower, NamedSharding(mesh, isp))
./tests/parallel/test_latlon_vface_reconstruct.py-88-
./tests/parallel/test_latlon_vface_reconstruct.py-89-    @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp, check_vma=False)
--
./tests/parallel/test_latlon_vface_reconstruct.py-119-
./tests/parallel/test_latlon_vface_reconstruct.py-120-    isp = P("lat", *((None,) * (u_cell.ndim - 1)))
./tests/parallel/test_latlon_vface_reconstruct.py:121:    u_sh = jax.device_put(jnp.asarray(u_cell), NamedSharding(mesh, isp))
./tests/parallel/test_latlon_vface_reconstruct.py:122:    v_sh = jax.device_put(jnp.asarray(v_cell), NamedSharding(mesh, isp))
./tests/parallel/test_latlon_vface_reconstruct.py-123-
./tests/parallel/test_latlon_vface_reconstruct.py-124-    activate_latlon_spmd_halo(mesh)   # arm the spmd backend for interp_cell_to_vface_halo
--
./tests/parallel/test_tiled_fv3_sw_full.py-69-    sh = NamedSharding(mesh, fo)
./tests/parallel/test_tiled_fv3_sw_full.py-70-    dh_t, du_t, dv_t = stage(
./tests/parallel/test_tiled_fv3_sw_full.py:71:        jax.device_put(h, sh), jax.device_put(u_d, sh),
./tests/parallel/test_tiled_fv3_sw_full.py:72:        jax.device_put(v_d, sh), jax.device_put(h_s, sh))
./tests/parallel/test_tiled_fv3_sw_full.py-73-    dh_t, du_t, dv_t = np.asarray(dh_t), np.asarray(du_t), np.asarray(dv_t)
./tests/parallel/test_tiled_fv3_sw_full.py-74-
--
./tests/parallel/test_tiled_fv3_sw_full.py-121-    stage = make_tiled_fv3_sw_tendencies_stage_2d(mesh, cdg, N, 2)
./tests/parallel/test_tiled_fv3_sw_full.py-122-    sh = NamedSharding(mesh, P("face", None, None))
./tests/parallel/test_tiled_fv3_sw_full.py:123:    h_sh = jax.device_put(h, sh)
./tests/parallel/test_tiled_fv3_sw_full.py:124:    u_sh = jax.device_put(u_d, sh)
./tests/parallel/test_tiled_fv3_sw_full.py:125:    v_sh = jax.device_put(v_d, sh)
./tests/parallel/test_tiled_fv3_sw_full.py:126:    hs_sh = jax.device_put(h_s, sh)
./tests/parallel/test_tiled_fv3_sw_full.py-127-
./tests/parallel/test_tiled_fv3_sw_full.py-128-    def loss(hh, uu, vv, hs):
--
./packages/ml/legoesm/training/data_parallel.py-262-            # A no-op (cheap) when the sample is already device-resident (the
./packages/ml/legoesm/training/data_parallel.py-263-            # legacy eager path), so this is safe for both.
./packages/ml/legoesm/training/data_parallel.py:264:            sample = jax.device_put(sample)
./packages/ml/legoesm/training/data_parallel.py-265-            params, opt_state, loss = mpi_data_parallel_train_step(
./packages/ml/legoesm/training/data_parallel.py-266-                loss_fn, params, opt_state, optimizer, sample, num_processes, comm=comm)
--
./tests/parallel/test_latlon_ocean_spmd_step.py-1-"""SPMD equivalence gate for the lat-lon C-grid ocean step (multi-node OMIP).
./tests/parallel/test_latlon_ocean_spmd_step.py-2-
./tests/parallel/test_latlon_ocean_spmd_step.py:3:This is the CORRECTNESS GATE for ``make_sharded_ocean_step`` (lat-band SPMD over
./tests/parallel/test_latlon_ocean_spmd_step.py-4-the ``"lat"`` axis): N steps under ``shard_map`` on 4 (CPU) devices must match
./tests/parallel/test_latlon_ocean_spmd_step.py-5-the single-device step to tolerance. It is the eORCA025 ¼° enabler (the ¼° grid
--
./tests/parallel/test_latlon_ocean_spmd_step.py-7-
./tests/parallel/test_latlon_ocean_spmd_step.py-8-STATUS (2026-06-16): the WRAPPER + halo layer are DONE and validated — the
./tests/parallel/test_latlon_ocean_spmd_step.py:9:staggered-v band decomposition (``shard_state_latlon`` ↔ ``v_lower`` ↔ in-body
./tests/parallel/test_latlon_ocean_spmd_step.py-10-ppermute reconstruction), the replicated-stacked grid indexed by
./tests/parallel/test_latlon_ocean_spmd_step.py-11-``axis_index``, and the SPMD branches of ``pad_with_pole_bc_lat`` /
--
./tests/parallel/test_latlon_ocean_spmd_step.py-83-    try:
./tests/parallel/test_latlon_ocean_spmd_step.py-84-        from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: F401
./tests/parallel/test_latlon_ocean_spmd_step.py:85:            make_sharded_ocean_step,
./tests/parallel/test_latlon_ocean_spmd_step.py-86-        )
./tests/parallel/test_latlon_ocean_spmd_step.py-87-        from legoesm.parallel.mesh import create_latlon_mesh  # noqa: F401
--
./tests/parallel/test_latlon_ocean_spmd_step.py-98-    from legoesm.parallel.mesh import create_latlon_mesh
./tests/parallel/test_latlon_ocean_spmd_step.py-99-    from legoesm.ocean.dynamics.sharded_ocean_step import (
./tests/parallel/test_latlon_ocean_spmd_step.py:100:        make_sharded_ocean_step,
./tests/parallel/test_latlon_ocean_spmd_step.py:101:        shard_state_latlon,
./tests/parallel/test_latlon_ocean_spmd_step.py-102-        gather_state_latlon,
./tests/parallel/test_latlon_ocean_spmd_step.py-103-    )
--
./tests/parallel/test_latlon_ocean_spmd_step.py-122-    model._ensure_vertex_mask(state0)
./tests/parallel/test_latlon_ocean_spmd_step.py-123-
./tests/parallel/test_latlon_ocean_spmd_step.py:124:    # lat-band SPMD on 4 devices.  The state is laid out with shard_state_latlon
./tests/parallel/test_latlon_ocean_spmd_step.py-125-    # (cell fields P("lat"); the staggered v / v_mask carried as v_lower, the
./tests/parallel/test_latlon_ocean_spmd_step.py-126-    # n_lat-row block that DOES divide N — a uniform tree.map(P("lat")) would
--
./tests/parallel/test_latlon_ocean_spmd_step.py-128-    # row reappended) for the bit-comparison vs the single-device reference.
./tests/parallel/test_latlon_ocean_spmd_step.py-129-    dev = create_latlon_mesh(n_devices=4)
./tests/parallel/test_latlon_ocean_spmd_step.py:130:    step = make_sharded_ocean_step(model, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py:131:    ss = shard_state_latlon(state0, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py-132-    for _ in range(n_steps):
./tests/parallel/test_latlon_ocean_spmd_step.py-133-        ss = step(ss, dt)
--
./tests/parallel/test_latlon_ocean_spmd_step.py-222-    from legoesm.ocean.dynamics.sharded_ocean_step import (
./tests/parallel/test_latlon_ocean_spmd_step.py-223-        gather_state_latlon,
./tests/parallel/test_latlon_ocean_spmd_step.py:224:        make_sharded_ocean_step,
./tests/parallel/test_latlon_ocean_spmd_step.py:225:        shard_forcing_latlon,
./tests/parallel/test_latlon_ocean_spmd_step.py:226:        shard_state_latlon,
./tests/parallel/test_latlon_ocean_spmd_step.py-227-    )
./tests/parallel/test_latlon_ocean_spmd_step.py-228-
--
./tests/parallel/test_latlon_ocean_spmd_step.py-245-    model._ensure_vertex_mask(state0)
./tests/parallel/test_latlon_ocean_spmd_step.py-246-    dev = create_latlon_mesh(n_devices=4)
./tests/parallel/test_latlon_ocean_spmd_step.py:247:    step = make_sharded_ocean_step(model, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py:248:    ss = shard_state_latlon(state0, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py:249:    fws = shard_forcing_latlon(fw, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py:250:    sfs = shard_forcing_latlon(sf, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py:251:    sponges = shard_forcing_latlon(sponge, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py-252-
./tests/parallel/test_latlon_ocean_spmd_step.py-253-    # Cache-key flip smoke: dynamics-only compile first, then the forcing
--
./tests/parallel/test_latlon_ocean_spmd_step.py-271-@pytest.mark.skipif(jax.device_count() < 4,
./tests/parallel/test_latlon_ocean_spmd_step.py-272-                    reason="needs >=4 devices (XLA_FLAGS host device count)")
./tests/parallel/test_latlon_ocean_spmd_step.py:273:def test_shard_forcing_stack_latlon_layout():
./tests/parallel/test_latlon_ocean_spmd_step.py-274-    """The block-scan stack sharder (run_omip JRA55 lanes) puts the lat axis
./tests/parallel/test_latlon_ocean_spmd_step.py-275-    on the ``"lat"`` mesh axis whether the leaf is a stacked
--
./tests/parallel/test_latlon_ocean_spmd_step.py-281-    from legoesm.parallel.mesh import create_latlon_mesh
./tests/parallel/test_latlon_ocean_spmd_step.py-282-    from legoesm.ocean.dynamics.sharded_ocean_step import (
./tests/parallel/test_latlon_ocean_spmd_step.py:283:        shard_forcing_stack_latlon,
./tests/parallel/test_latlon_ocean_spmd_step.py-284-    )
./tests/parallel/test_latlon_ocean_spmd_step.py-285-
--
./tests/parallel/test_latlon_ocean_spmd_step.py-299-        "meta": "1958-01-01",                            # non-array passthrough
./tests/parallel/test_latlon_ocean_spmd_step.py-300-    }
./tests/parallel/test_latlon_ocean_spmd_step.py:301:    out = shard_forcing_stack_latlon(stack, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py-302-
./tests/parallel/test_latlon_ocean_spmd_step.py-303-    assert _lat_axis(out["rec3d"]) == 1
--
./tests/parallel/test_latlon_ocean_spmd_step.py-310-
./tests/parallel/test_latlon_ocean_spmd_step.py-311-    # mesh=None is the serial-lane passthrough (identity).
./tests/parallel/test_latlon_ocean_spmd_step.py:312:    assert shard_forcing_stack_latlon(stack, None) is stack
./tests/parallel/test_latlon_ocean_spmd_step.py-313-
./tests/parallel/test_latlon_ocean_spmd_step.py-314-
--
./tests/parallel/test_latlon_ocean_spmd_step.py-352-    activate_latlon_spmd_halo(mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py-353-    try:
./tests/parallel/test_latlon_ocean_spmd_step.py:354:        sharded = body(jax.device_put(F, sh), jax.device_put(area, sh),
./tests/parallel/test_latlon_ocean_spmd_step.py:355:                       jax.device_put(mask, sh))
./tests/parallel/test_latlon_ocean_spmd_step.py-356-        # Non-vacuity: at least one band's local mean differs from the
./tests/parallel/test_latlon_ocean_spmd_step.py-357-        # global mean, so a missing psum WOULD change the answer.
--
./tests/parallel/test_latlon_ocean_spmd_step.py-385-    from legoesm.parallel.mesh import create_latlon_mesh
./tests/parallel/test_latlon_ocean_spmd_step.py-386-    from legoesm.ocean.dynamics.sharded_ocean_step import (
./tests/parallel/test_latlon_ocean_spmd_step.py:387:        make_sharded_ocean_step,
./tests/parallel/test_latlon_ocean_spmd_step.py:388:        shard_state_latlon,
./tests/parallel/test_latlon_ocean_spmd_step.py-389-    )
./tests/parallel/test_latlon_ocean_spmd_step.py-390-    from legoesm.ocean.state import OceanSurfaceForcing
--
./tests/parallel/test_latlon_ocean_spmd_step.py-397-    model._ensure_vertex_mask(state0)
./tests/parallel/test_latlon_ocean_spmd_step.py-398-    dev = create_latlon_mesh(n_devices=4)
./tests/parallel/test_latlon_ocean_spmd_step.py:399:    step = make_sharded_ocean_step(model, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py:400:    ss = shard_state_latlon(state0, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py-401-    bad = OceanSurfaceForcing(
./tests/parallel/test_latlon_ocean_spmd_step.py-402-        tau_y=jnp.zeros((grid.n_lat + 1, grid.n_lon)))
--
./tests/parallel/test_latlon_ocean_spmd_step.py-410-                    reason="sharded_ocean_step module not present")
./tests/parallel/test_latlon_ocean_spmd_step.py-411-def test_sharded_ocean_step_global_matches_explicit_scatter_gather():
./tests/parallel/test_latlon_ocean_spmd_step.py:412:    """``make_sharded_ocean_step_global`` (global-in/global-out, the minimal
./tests/parallel/test_latlon_ocean_spmd_step.py:413:    driver entry) must equal the explicit ``shard_state_latlon`` -> inner step ->
./tests/parallel/test_latlon_ocean_spmd_step.py-414-    ``gather_state_latlon`` path BIT-FOR-BIT (it is literally that composition),
./tests/parallel/test_latlon_ocean_spmd_step.py-415-    AND match the single-device reference to the same re-association floor.
--
./tests/parallel/test_latlon_ocean_spmd_step.py-420-    from legoesm.grids.latlon import ensure_geometry
./tests/parallel/test_latlon_ocean_spmd_step.py-421-    from legoesm.ocean.dynamics.sharded_ocean_step import (
./tests/parallel/test_latlon_ocean_spmd_step.py:422:        make_sharded_ocean_step,
./tests/parallel/test_latlon_ocean_spmd_step.py:423:        make_sharded_ocean_step_global,
./tests/parallel/test_latlon_ocean_spmd_step.py:424:        shard_state_latlon,
./tests/parallel/test_latlon_ocean_spmd_step.py-425-        gather_state_latlon,
./tests/parallel/test_latlon_ocean_spmd_step.py-426-    )
--
./tests/parallel/test_latlon_ocean_spmd_step.py-452-
./tests/parallel/test_latlon_ocean_spmd_step.py-453-    # explicit scatter -> inner sharded step -> gather
./tests/parallel/test_latlon_ocean_spmd_step.py:454:    inner = make_sharded_ocean_step(model, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py:455:    ss = shard_state_latlon(state0, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py-456-    for _ in range(n_steps):
./tests/parallel/test_latlon_ocean_spmd_step.py-457-        ss = inner(ss, dt, surface_forcing=sf)
--
./tests/parallel/test_latlon_ocean_spmd_step.py-459-
./tests/parallel/test_latlon_ocean_spmd_step.py-460-    # global-in/global-out wrapper (scatter + gather PER STEP)
./tests/parallel/test_latlon_ocean_spmd_step.py:461:    glob = make_sharded_ocean_step_global(model, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py-462-    sg = state0
./tests/parallel/test_latlon_ocean_spmd_step.py-463-    for _ in range(n_steps):
--
./tests/parallel/test_tiled_center_to_dgrid_vector.py-97-
./tests/parallel/test_tiled_center_to_dgrid_vector.py-98-    fw = NamedSharding(mesh, P("face", None, None, None))
./tests/parallel/test_tiled_center_to_dgrid_vector.py:99:    u_t, v_t = stage(jax.device_put(u_cc, fw), jax.device_put(v_cc, fw))
./tests/parallel/test_tiled_center_to_dgrid_vector.py-100-
./tests/parallel/test_tiled_center_to_dgrid_vector.py-101-    ru, rv = _compare_corner(np.asarray(u_t), np.asarray(v_t), u_g, v_g, KT, nl)
--
./tests/parallel/test_atm_latlon_2d_tiling.py-186-        in_specs=P("lat", "lon", None),
./tests/parallel/test_atm_latlon_2d_tiling.py-187-        out_specs=P("lat", "lon", None), check_vma=False))
./tests/parallel/test_atm_latlon_2d_tiling.py:188:    sharded = jax.device_put(
./tests/parallel/test_atm_latlon_2d_tiling.py-189-        field, NamedSharding(mesh, P("lat", "lon", None)))
./tests/parallel/test_atm_latlon_2d_tiling.py:190:    out = np.asarray(jax.device_put(fn(sharded), NamedSharding(mesh, P())))
./tests/parallel/test_atm_latlon_2d_tiling.py-191-
./tests/parallel/test_atm_latlon_2d_tiling.py-192-    nl, w = N_LAT // p_lat, N_LON // p_lon
--
./tests/parallel/test_atm_latlon_2d_tiling.py-214-        body, mesh=mesh, in_specs=P("lat", "lon", None),
./tests/parallel/test_atm_latlon_2d_tiling.py-215-        out_specs=P("lat", "lon", None), check_vma=False))
./tests/parallel/test_atm_latlon_2d_tiling.py:216:    sharded = jax.device_put(
./tests/parallel/test_atm_latlon_2d_tiling.py-217-        field, NamedSharding(mesh, P("lat", "lon", None)))
./tests/parallel/test_atm_latlon_2d_tiling.py-218-    hlo = fn.lower(sharded).compile().as_text()
--
./tests/parallel/test_atm_latlon_2d_tiling.py-238-        "(partner ppermute lever)")
./tests/parallel/test_atm_latlon_2d_tiling.py-239-    assert "collective-permute" in hlo
./tests/parallel/test_atm_latlon_2d_tiling.py:240:    out = np.asarray(jax.device_put(fn(sharded), NamedSharding(mesh, P())))
./tests/parallel/test_atm_latlon_2d_tiling.py-241-    serial = np.asarray(pad_halo_latlon_3d_local(field, halo))
./tests/parallel/test_atm_latlon_2d_tiling.py-242-    nl, w = N_LAT // p_lat, N_LON // p_lon
--
./tests/parallel/test_atm_latlon_2d_tiling.py-264-    fn, sharded, hlo = _lowered_pad(mesh, field, halo)
./tests/parallel/test_atm_latlon_2d_tiling.py-265-    assert "all-gather" in hlo
./tests/parallel/test_atm_latlon_2d_tiling.py:266:    out = np.asarray(jax.device_put(fn(sharded), NamedSharding(mesh, P())))
./tests/parallel/test_atm_latlon_2d_tiling.py-267-    serial = np.asarray(pad_halo_latlon_3d_local(field, halo))
./tests/parallel/test_atm_latlon_2d_tiling.py-268-    w = n_lon // p_lon
--
./tests/parallel/test_atm_latlon_2d_tiling.py-375-    finally:
./tests/parallel/test_atm_latlon_2d_tiling.py-376-        deactivate_latlon_spmd_halo()
./tests/parallel/test_atm_latlon_2d_tiling.py:377:    g = lambda x: np.asarray(jax.device_put(x, NamedSharding(mesh, P())))
./tests/parallel/test_atm_latlon_2d_tiling.py-378-    dul_b, dvl_b, dT_b, dps_b = (g(x) for x in out)
./tests/parallel/test_atm_latlon_2d_tiling.py-379-    return serial, (dul_b, dvl_b, dT_b, dps_b), (p_lat, p_lon)
--
./tests/ocean/unit/test_ocean_compatibility.py-130-        step_source = inspect.getsource(SpectralOceanModel.step)
./tests/ocean/unit/test_ocean_compatibility.py-131-        assert "_use_cpu_for_spectral" in step_source
./tests/ocean/unit/test_ocean_compatibility.py:132:        assert "device_put" in step_source
./tests/ocean/unit/test_ocean_compatibility.py-133-
./tests/ocean/unit/test_ocean_compatibility.py-134-        # Verify batched CPU integration method exists
--
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-2-
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-3-Twin of ``test_latlon_ocean_spmd_step.py`` with
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py:4:``barotropic_wide_halo=True``: N steps of ``make_sharded_ocean_step`` across
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-5-4 (CPU) devices must match the single-device wide-halo step at the sharded
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-6-split-explicit re-association floor.  A staggered off-by-one in the wide
--
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-74-    from legoesm.parallel.mesh import create_latlon_mesh
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-75-    from legoesm.ocean.dynamics.sharded_ocean_step import (
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py:76:        make_sharded_ocean_step,
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py:77:        shard_state_latlon,
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-78-        gather_state_latlon,
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-79-    )
--
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-90-
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-91-    dev = create_latlon_mesh(n_devices=4)
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py:92:    step = make_sharded_ocean_step(model, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py:93:    ss = shard_state_latlon(state0, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-94-    for _ in range(n_steps):
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-95-        ss = step(ss, dt)
--
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-113-    from legoesm.parallel.mesh import create_latlon_mesh
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-114-    from legoesm.ocean.dynamics.sharded_ocean_step import (
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py:115:        make_sharded_ocean_step,
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py:116:        shard_state_latlon,
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-117-    )
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-118-
--
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-123-    model._ensure_vertex_mask(state0)
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-124-    dev = create_latlon_mesh(n_devices=4)
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py:125:    step = make_sharded_ocean_step(model, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py:126:    ss = shard_state_latlon(state0, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-127-    lowered = jax.jit(step).lower(ss, 600.0)
./tests/parallel/test_latlon_ocean_spmd_wide_halo.py-128-    return lowered.compile().as_text()
--
./tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py-8-end-to-end with the ported parity + conservation gates armed — pinning the
./tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py-9-multi-controller pieces a Derecho/Levante multi-node NCCL ocean run
./tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:10:exercises: ``shard_state_latlon`` onto a mesh spanning non-addressable
./tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py-11-devices, cross-process ppermute/psum inside the jitted step, the replication
./tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py-12-gather, rank-0-gated output, and the ``sync_global_devices`` barriers.
--
./tests/parallel/test_atm_latlon_spmd_step.py-151-    afn = atm_grid_array_field_names(template)
./tests/parallel/test_atm_latlon_spmd_step.py-152-    rep = NamedSharding(mesh, P())
./tests/parallel/test_atm_latlon_spmd_step.py:153:    stacks = {n: jax.device_put(jnp.stack([jnp.asarray(getattr(g, n))
./tests/parallel/test_atm_latlon_spmd_step.py-154-              for g in band_grids], 0), rep) for n in afn}
./tests/parallel/test_atm_latlon_spmd_step.py-155-    perm_north, _ = latlon_band_perms(N_DEV)
--
./tests/parallel/test_atm_latlon_spmd_step.py-176-    finally:
./tests/parallel/test_atm_latlon_spmd_step.py-177-        deactivate_latlon_spmd_halo()
./tests/parallel/test_atm_latlon_spmd_step.py:178:    g = lambda x: np.asarray(jax.device_put(x, NamedSharding(mesh, P())))
./tests/parallel/test_atm_latlon_spmd_step.py-179-    du_b, dvl_b, dT_b, dps_b = (g(x) for x in out)
./tests/parallel/test_atm_latlon_spmd_step.py-180-    # band returns v_lower (N_LAT rows); re-cap the north pole row (=0) so the
--
./tests/parallel/test_atm_latlon_spmd_step.py-273-
./tests/parallel/test_atm_latlon_spmd_step.py-274-
./tests/parallel/test_atm_latlon_spmd_step.py:275:def test_replicate_leaf_multiprocess_branch_matches_device_put():
./tests/parallel/test_atm_latlon_spmd_step.py-276-    """The multi-controller gather branch (jit-compiled identity with
./tests/parallel/test_atm_latlon_spmd_step.py-277-    replicated out_shardings) must produce the SAME replicated array as the
./tests/parallel/test_atm_latlon_spmd_step.py:278:    single-process device_put branch — on values, sharding, and for both a
./tests/parallel/test_atm_latlon_spmd_step.py-279-    lat-sharded and an already-replicated input. This exercises the
./tests/parallel/test_atm_latlon_spmd_step.py-280-    ``multiprocess=True`` code path for real on a single process (where both
--
./tests/parallel/test_atm_latlon_spmd_step.py-287-    rng = np.random.default_rng(7)
./tests/parallel/test_atm_latlon_spmd_step.py-288-    full = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))
./tests/parallel/test_atm_latlon_spmd_step.py:289:    sharded = jax.device_put(full, NamedSharding(mesh, P("lat", None, None)))
./tests/parallel/test_atm_latlon_spmd_step.py-290-
./tests/parallel/test_atm_latlon_spmd_step.py:291:    for arr in (sharded, jax.device_put(full, rep)):
./tests/parallel/test_atm_latlon_spmd_step.py-292-        a = replicate_leaf(arr, rep, multiprocess=False)
./tests/parallel/test_atm_latlon_spmd_step.py-293-        b = replicate_leaf(arr, rep, multiprocess=True)
--
./tests/parallel/test_tiled_fv3_hydrostatic_step.py-136-    fo = NamedSharding(mesh, P("face", None, None))
./tests/parallel/test_tiled_fv3_hydrostatic_step.py-137-    ut, vt, Tt, pst = step(
./tests/parallel/test_tiled_fv3_hydrostatic_step.py:138:        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
./tests/parallel/test_tiled_fv3_hydrostatic_step.py:139:        jax.device_put(T, fw), jax.device_put(p_s, fo),
./tests/parallel/test_tiled_fv3_hydrostatic_step.py:140:        jax.device_put(phis, fo))
./tests/parallel/test_tiled_fv3_hydrostatic_step.py-141-
./tests/parallel/test_tiled_fv3_hydrostatic_step.py-142-    r_u = _rel_corner(np.asarray(ut), ug, KT, nl)
--
./tests/parallel/test_latlon_spmd_fused_halo.py-322-    from legoesm.ocean.dynamics.sharded_ocean_step import (
./tests/parallel/test_latlon_spmd_fused_halo.py-323-        gather_state_latlon,
./tests/parallel/test_latlon_spmd_fused_halo.py:324:        make_sharded_ocean_step,
./tests/parallel/test_latlon_spmd_fused_halo.py:325:        shard_state_latlon,
./tests/parallel/test_latlon_spmd_fused_halo.py-326-    )
./tests/parallel/test_latlon_spmd_fused_halo.py-327-    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
--
./tests/parallel/test_latlon_spmd_fused_halo.py-347-    for flag in ("0", "1"):
./tests/parallel/test_latlon_spmd_fused_halo.py-348-        _env_flag(monkeypatch, flag)
./tests/parallel/test_latlon_spmd_fused_halo.py:349:        step = make_sharded_ocean_step(model, dev.mesh)
./tests/parallel/test_latlon_spmd_fused_halo.py:350:        ss = shard_state_latlon(state0, dev.mesh)
./tests/parallel/test_latlon_spmd_fused_halo.py-351-        hlo_counts[flag] = _count_ppermutes(
./tests/parallel/test_latlon_spmd_fused_halo.py-352-            jax.jit(step).lower(ss, 600.0).compile().as_text())
--
./tests/parallel/test_latlon_spmd_fused_halo.py-370-    # driver calls step() directly each step).
./tests/parallel/test_latlon_spmd_fused_halo.py-371-    _env_flag(monkeypatch, "0")
./tests/parallel/test_latlon_spmd_fused_halo.py:372:    step = make_sharded_ocean_step(model, dev.mesh)
./tests/parallel/test_latlon_spmd_fused_halo.py:373:    ss = shard_state_latlon(state0, dev.mesh)
./tests/parallel/test_latlon_spmd_fused_halo.py-374-    n_off = _count_ppermutes(
./tests/parallel/test_latlon_spmd_fused_halo.py-375-        jax.jit(lambda s, d: step(s, d)).lower(ss, 600.0)
--
./tests/parallel/test_mpas_atm_native_step.py-602-        dev_sharding = NamedSharding(jmesh, P("device"))
./tests/parallel/test_mpas_atm_native_step.py-603-        halo_args = tuple(
./tests/parallel/test_mpas_atm_native_step.py:604:            tuple(jax.device_put(a, dev_sharding) for a in (
./tests/parallel/test_mpas_atm_native_step.py-605-                sched["send_cell_idx"][r], sched["recv_cell_pos"][r],
./tests/parallel/test_mpas_atm_native_step.py-606-                sched["send_edge_idx"][r], sched["recv_edge_pos"][r]))
--
./tests/parallel/test_tiled_mass_divergence.py-79-    fo = P("face", None, None)
./tests/parallel/test_tiled_mass_divergence.py-80-    sh_fo = NamedSharding(mesh, fo)
./tests/parallel/test_tiled_mass_divergence.py:81:    h_sh = jax.device_put(h, sh_fo)
./tests/parallel/test_tiled_mass_divergence.py:82:    u_sh = jax.device_put(u_c, sh_fo)
./tests/parallel/test_tiled_mass_divergence.py:83:    v_sh = jax.device_put(v_c, sh_fo)
./tests/parallel/test_tiled_mass_divergence.py-84-
./tests/parallel/test_tiled_mass_divergence.py-85-    dh_t = np.asarray(stage(h_sh, u_sh, v_sh))   # (6, n, n), cc exact partition
--
./tests/parallel/test_tiled_pad_body.py-60-
./tests/parallel/test_tiled_pad_body.py-61-    sharding = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
./tests/parallel/test_tiled_pad_body.py:62:    ref_sh = jax.device_put(ref, sharding)
./tests/parallel/test_tiled_pad_body.py-63-    fn = _make_exchange_ppermute_tiled(mesh, ndim=3, halo=1,
./tests/parallel/test_tiled_pad_body.py-64-                                       with_offsets=True)
--
./tests/parallel/test_tiled_pad_body.py-81-    offs = jnp.asarray(compute_halo_interp_offsets(N))
./tests/parallel/test_tiled_pad_body.py-82-    sharding = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
./tests/parallel/test_tiled_pad_body.py:83:    ref_sh = jax.device_put(ref, sharding)
./tests/parallel/test_tiled_pad_body.py-84-
./tests/parallel/test_tiled_pad_body.py-85-    wrapped = _make_exchange_ppermute_tiled(
--
./tests/parallel/test_tiled_pad_body.py-127-    sh4 = NamedSharding(mesh, P("face", "tile_i", "tile_j", None))
./tests/parallel/test_tiled_pad_body.py-128-    sh3 = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
./tests/parallel/test_tiled_pad_body.py:129:    ref4_sh = jax.device_put(ref4, sh4)
./tests/parallel/test_tiled_pad_body.py-130-
./tests/parallel/test_tiled_pad_body.py-131-    body4 = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)
--
./tests/parallel/test_tiled_pad_body.py-148-    assert out4.shape == (6, KT * out_blk, KT * out_blk, C)
./tests/parallel/test_tiled_pad_body.py-149-    for k in range(C):
./tests/parallel/test_tiled_pad_body.py:150:        ref3_sh = jax.device_put(ref4[..., k], sh3)
./tests/parallel/test_tiled_pad_body.py-151-        out3 = _stage3(ref3_sh, offs)
./tests/parallel/test_tiled_pad_body.py-152-        np.testing.assert_array_equal(
--
./tests/parallel/test_tiled_pad_body.py-217-    sh3 = NamedSharding(mesh, fw3)
./tests/parallel/test_tiled_pad_body.py-218-    sho = NamedSharding(mesh, fo)
./tests/parallel/test_tiled_pad_body.py:219:    ca_d, sa_d = jax.device_put(ca, sho), jax.device_put(sa, sho)
./tests/parallel/test_tiled_pad_body.py:220:    cap_d, sap_d = jax.device_put(cap, sho), jax.device_put(sap, sho)
./tests/parallel/test_tiled_pad_body.py:221:    out4u, out4v = _stage4(jax.device_put(u4, sh4), jax.device_put(v4, sh4),
./tests/parallel/test_tiled_pad_body.py-222-                           ca_d, sa_d, cap_d, sap_d, offs)
./tests/parallel/test_tiled_pad_body.py-223-    out_blk = NL + 2
./tests/parallel/test_tiled_pad_body.py-224-    assert np.asarray(out4u).shape == (6, KT * out_blk, KT * out_blk, C)
./tests/parallel/test_tiled_pad_body.py-225-    for k in range(C):
./tests/parallel/test_tiled_pad_body.py:226:        o3u, o3v = _stage3(jax.device_put(u4[..., k], sh3),
./tests/parallel/test_tiled_pad_body.py:227:                           jax.device_put(v4[..., k], sh3),
./tests/parallel/test_tiled_pad_body.py-228-                           ca_d, sa_d, cap_d, sap_d, offs)
./tests/parallel/test_tiled_pad_body.py-229-        np.testing.assert_array_equal(
--
./tests/parallel/test_sharded_step_sharding_tripwire.py-197-
./tests/parallel/test_sharded_step_sharding_tripwire.py-198-        good = {
./tests/parallel/test_sharded_step_sharding_tripwire.py:199:            "T": jax.device_put(arr, NamedSharding(mesh, P("face"))),
./tests/parallel/test_sharded_step_sharding_tripwire.py:200:            "scalar": jax.device_put(
./tests/parallel/test_sharded_step_sharding_tripwire.py-201-                jnp.asarray(1.0), NamedSharding(mesh, P()),
./tests/parallel/test_sharded_step_sharding_tripwire.py-202-            ),
--
./tests/parallel/test_sharded_step_sharding_tripwire.py-206-
./tests/parallel/test_sharded_step_sharding_tripwire.py-207-        bad = dict(good)
./tests/parallel/test_sharded_step_sharding_tripwire.py:208:        bad["T"] = jax.device_put(arr, NamedSharding(mesh, P()))  # replicated
./tests/parallel/test_sharded_step_sharding_tripwire.py-209-        with pytest.raises(RuntimeError, match="does not match expected"):
./tests/parallel/test_sharded_step_sharding_tripwire.py-210-            _assert_expected_sharding(bad, dev_config, where="synthetic-bad")
--
./tests/parallel/test_persistent_sharded_ocean_loop.py-3-
./tests/parallel/test_persistent_sharded_ocean_loop.py-4-The production multi-GPU OMIP host loop used to call
./tests/parallel/test_persistent_sharded_ocean_loop.py:5:``make_sharded_ocean_step_global`` EVERY step — a full-state scatter
./tests/parallel/test_persistent_sharded_ocean_loop.py:6:(``shard_state_latlon``) + gather (``gather_state_latlon``) per step, i.e.
./tests/parallel/test_persistent_sharded_ocean_loop.py-7-2 full-state transfers/step.  The persistent lane instead keeps the state
./tests/parallel/test_persistent_sharded_ocean_loop.py:8:lat-band SHARDED across steps via ``make_sharded_ocean_step`` and gathers
./tests/parallel/test_persistent_sharded_ocean_loop.py-9-ONLY at real output boundaries.  This gate asserts, over a 10-step loop with
./tests/parallel/test_persistent_sharded_ocean_loop.py-10-production-shaped surface + freshwater forcing (``normalize_freshwater=True``
--
./tests/parallel/test_persistent_sharded_ocean_loop.py-75-    try:
./tests/parallel/test_persistent_sharded_ocean_loop.py-76-        from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: F401
./tests/parallel/test_persistent_sharded_ocean_loop.py:77:            make_sharded_ocean_step,
./tests/parallel/test_persistent_sharded_ocean_loop.py-78-        )
./tests/parallel/test_persistent_sharded_ocean_loop.py-79-        from legoesm.parallel.mesh import create_latlon_mesh  # noqa: F401
--
./tests/parallel/test_persistent_sharded_ocean_loop.py-129-    # ---- count every full-state transfer through the module entry points ----
./tests/parallel/test_persistent_sharded_ocean_loop.py-130-    calls = {"shard": 0, "gather": 0}
./tests/parallel/test_persistent_sharded_ocean_loop.py:131:    _shard0, _gather0 = sos.shard_state_latlon, sos.gather_state_latlon
./tests/parallel/test_persistent_sharded_ocean_loop.py-132-
./tests/parallel/test_persistent_sharded_ocean_loop.py-133-    def _counting_shard(st, m):
--
./tests/parallel/test_persistent_sharded_ocean_loop.py-139-        return _gather0(st, m)
./tests/parallel/test_persistent_sharded_ocean_loop.py-140-
./tests/parallel/test_persistent_sharded_ocean_loop.py:141:    monkeypatch.setattr(sos, "shard_state_latlon", _counting_shard)
./tests/parallel/test_persistent_sharded_ocean_loop.py-142-    monkeypatch.setattr(sos, "gather_state_latlon", _counting_gather)
./tests/parallel/test_persistent_sharded_ocean_loop.py-143-
--
./tests/parallel/test_persistent_sharded_ocean_loop.py-171-
./tests/parallel/test_persistent_sharded_ocean_loop.py-172-    # ---------------- OLD lane: per-step global-in/global-out wrapper --------
./tests/parallel/test_persistent_sharded_ocean_loop.py:173:    glob = sos.make_sharded_ocean_step_global(model, mesh)
./tests/parallel/test_persistent_sharded_ocean_loop.py-174-    sg = state0
./tests/parallel/test_persistent_sharded_ocean_loop.py-175-    cur_probe = {}
--
./tests/parallel/test_persistent_sharded_ocean_loop.py-199-    # ---------------- NEW lane: persistent sharded loop ----------------------
./tests/parallel/test_persistent_sharded_ocean_loop.py-200-    calls["shard"] = calls["gather"] = 0
./tests/parallel/test_persistent_sharded_ocean_loop.py:201:    inner = sos.make_sharded_ocean_step(model, mesh)
./tests/parallel/test_persistent_sharded_ocean_loop.py:202:    ss = sos.shard_state_latlon(state0, mesh)          # ONE initial shard
./tests/parallel/test_persistent_sharded_ocean_loop.py-203-    for k in range(1, n_steps + 1):
./tests/parallel/test_persistent_sharded_ocean_loop.py-204-        # surface-current consumer on the SHARDED state: v is the n_lat-row
--
./tests/parallel/test_persistent_sharded_ocean_loop.py-234-            v_out = np.asarray(ss.v.data)
./tests/parallel/test_persistent_sharded_ocean_loop.py-235-            assert v_out.shape[0] == n_lat + 1
./tests/parallel/test_persistent_sharded_ocean_loop.py:236:            ss = sos.shard_state_latlon(ss, mesh)      # lazy re-shard
./tests/parallel/test_persistent_sharded_ocean_loop.py-237-        if k == ice_bc_step:
./tests/parallel/test_persistent_sharded_ocean_loop.py-238-            # ice-thermo-style BC on the SHARDED state (2-D slice pull +
--
./tests/parallel/test_latlon_spmd_northfold.py-89-
./tests/parallel/test_latlon_spmd_northfold.py-90-    isp = P("lat", *((None,) * (field.ndim - 1)))
./tests/parallel/test_latlon_spmd_northfold.py:91:    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
./tests/parallel/test_latlon_spmd_northfold.py-92-    activate_latlon_spmd_halo(mesh)
./tests/parallel/test_latlon_spmd_northfold.py-93-    try:
--
./tests/parallel/test_atm_latlon_operator_split_spmd.py-191-        c_b = step_dev(c_b, f_b)
./tests/parallel/test_atm_latlon_operator_split_spmd.py-192-    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
./tests/parallel/test_atm_latlon_operator_split_spmd.py:193:    c_b = jax.tree.map(lambda x: np.asarray(jax.device_put(x, rep)), c_b)
./tests/parallel/test_atm_latlon_operator_split_spmd.py-194-
./tests/parallel/test_atm_latlon_operator_split_spmd.py-195-    # Non-vacuity: tke genuinely evolved off its 1e-4 floor (AR1 -> ~0.0037).
--
./packages/core/legoesm/grids/gaussian.py-496-        if target_device is None:
./packages/core/legoesm/grids/gaussian.py-497-            return jnp.array(np_arr, dtype=dtype)
./packages/core/legoesm/grids/gaussian.py:498:        return jax.device_put(np_arr, target_device)
./packages/core/legoesm/grids/gaussian.py-499-
./packages/core/legoesm/grids/gaussian.py-500-    # Precompute weighted Legendre matrices (avoid recomputing every SH analysis)
--
./tests/parallel/test_tiled_d2a2c_ua_va.py-126-    co = P("face", "tile_i", "tile_j")
./tests/parallel/test_tiled_d2a2c_ua_va.py-127-    sh = NamedSharding(mesh, co)
./tests/parallel/test_tiled_d2a2c_ua_va.py:128:    args = [jax.device_put(x, sh) for x in (utmp_int, vtmp_int, cos_sg5, rsin2)]
./tests/parallel/test_tiled_d2a2c_ua_va.py-129-
./tests/parallel/test_tiled_d2a2c_ua_va.py-130-    @partial(shard_map, mesh=mesh, in_specs=(co, co, co, co),
--
./tests/parallel/test_tiled_fv3_hydrostatic_thermo.py-148-    fo = NamedSharding(mesh, P("face", None, None))
tests/bench/test_scaling_metadata.py:247:def test_transport_route_b_nccl_on_gpu_gloo_on_cpu():
tests/bench/test_scaling_metadata.py:294:    # A route-B NCCL (or intra-process xla-local) GPU row has NO mpi4jax halo
scripts/plot/plot_scaling_dashboard.py:2:"""Current-state scaling dashboard for the Ginsburg campaign (2x2 PNG).
scripts/plot/plot_scaling_dashboard.py:136:    fig.suptitle("legoESM scaling campaign — Ginsburg, 2026-06-10 (interim)", y=0.995)
scripts/bench/bench_ocean_latlon_spmd_scaling.py:22:Multi-controller (route-B, ``--multicontroller``): identical contract to the
scripts/bench/bench_ocean_latlon_spmd_scaling.py:31:  srun -n 8 python bench_ocean_latlon_spmd_scaling.py --multicontroller \
scripts/bench/bench_ocean_latlon_spmd_scaling.py:33:  mpiexec -n 8 python ... --multicontroller --coordinator host0:9876
scripts/bench/bench_ocean_latlon_spmd_scaling.py:303:    p.add_argument("--multicontroller", action="store_true",
scripts/bench/bench_ocean_latlon_spmd_scaling.py:304:                   help="Route-B multi-controller: jax.distributed.initialize "
scripts/bench/bench_ocean_latlon_spmd_scaling.py:338:    if args.multicontroller:
scripts/bench/bench_ocean_latlon_spmd_scaling.py:342:        from legoesm.parallel.early_init import init_multicontroller_distributed
scripts/bench/bench_ocean_latlon_spmd_scaling.py:343:        init_multicontroller_distributed(args.coordinator)
scripts/bench/bench_ocean_latlon_spmd_scaling.py:355:    if args.multicontroller and nd != avail:
scripts/bench/bench_ocean_latlon_spmd_scaling.py:357:        # of the program (non-addressable participation hazard). Route-B uses
scripts/bench/bench_ocean_latlon_spmd_scaling.py:360:            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
scripts/bench/bench_ocean_latlon_spmd_scaling.py:392:            # local_devices, NOT devices: under multicontroller jax.devices()
scripts/bench/bench_ocean_latlon_spmd_scaling.py:480:    # needs the gathered global final state on ONE process; multicontroller
scripts/bench/bench_ocean_latlon_spmd_scaling.py:573:        residual_reason = ("multicontroller run: the residual probe needs "
scripts/bench/bench_ocean_latlon_spmd_scaling.py:700:        multicontroller=bool(args.multicontroller),
scripts/bench/bench_ocean_latlon_spmd_scaling.py:763:            "multicontroller": bool(args.multicontroller),
scripts/bench/bench_ocean_latlon_spmd_scaling.py:766:            # Route-B transport facts (socket-fallback flag): a
scripts/bench/bench_ocean_latlon_spmd_scaling.py:769:            "nccl": (_nccl_report if args.multicontroller
tests/bench/test_aggregate_bcw_scaling.py:181:    Models the REAL route-B node-fill: each rung fills a 128-core node with
tests/bench/test_aggregate_bcw_scaling.py:201:    """#764 item 1: the cube route-B throughput lane.  The flat cube
scripts/bench/run_cpu_mpi_scaling.py:96:    # SLURM sets SLURM_CPUS_PER_TASK; PBS/PALS (Derecho route-B) does NOT — it
scripts/bench/run_cpu_mpi_scaling.py:99:    # n_cores accounting) so the cube route-B lane does not silently single-
scripts/bench/run_cpu_mpi_scaling.py:1428:    # wrong for the route-B cube lane on PBS).
scripts/bench/run_cpu_mpi_scaling.py:1454:        # cs-spmd (route-B) keeps auto-resolution.
scripts/bench/run_cpu_mpi_scaling.py:1646:        # subset omitted PALS_LOCAL_SIZE, so a Derecho mpiexec route-B
scripts/bench/run_cpu_mpi_scaling.py:1678:        # Route-B: jax.distributed is already federated (initialized above,
scripts/bench/bench_mpas_spmd_scaling.py:31:Multi-controller (route-B, ``--multicontroller``): identical contract to the
scripts/bench/bench_mpas_spmd_scaling.py:38:``--multicontroller`` the partition checksum is asserted equal across
scripts/bench/bench_mpas_spmd_scaling.py:44:  srun -n 6 python bench_mpas_spmd_scaling.py --multicontroller \
scripts/bench/bench_mpas_spmd_scaling.py:46:  mpiexec -n 6 python ... --multicontroller --coordinator host0:9876
scripts/bench/bench_mpas_spmd_scaling.py:212:                        "meshes (the multicontroller selfspawn tests "
scripts/bench/bench_mpas_spmd_scaling.py:235:    p.add_argument("--multicontroller", action="store_true",
scripts/bench/bench_mpas_spmd_scaling.py:236:                   help="Route-B multi-controller: jax.distributed.initialize "
scripts/bench/bench_mpas_spmd_scaling.py:261:    if args.multicontroller:
scripts/bench/bench_mpas_spmd_scaling.py:296:    if args.multicontroller and nd != avail:
scripts/bench/bench_mpas_spmd_scaling.py:298:        # of the program (non-addressable participation hazard). Route-B uses
scripts/bench/bench_mpas_spmd_scaling.py:301:            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
scripts/bench/bench_mpas_spmd_scaling.py:318:    if args.multicontroller:
scripts/bench/bench_mpas_spmd_scaling.py:326:        from jax.experimental import multihost_utils
scripts/bench/bench_mpas_spmd_scaling.py:338:        multihost_utils.assert_equal(
scripts/bench/bench_mpas_spmd_scaling.py:399:        from jax.experimental import multihost_utils
scripts/bench/bench_mpas_spmd_scaling.py:400:        multihost_utils.sync_global_devices("mpas_spmd_bench_start")
scripts/bench/bench_mpas_spmd_scaling.py:415:        from jax.experimental import multihost_utils
scripts/bench/bench_mpas_spmd_scaling.py:416:        multihost_utils.sync_global_devices("mpas_spmd_bench_end")
scripts/bench/bench_mpas_spmd_scaling.py:513:        multicontroller=bool(args.multicontroller),
scripts/bench/bench_mpas_spmd_scaling.py:561:            "multicontroller": bool(args.multicontroller),
scripts/bench/bench_cube_shardmap_halo.py:564:def _multihost_barrier(tag: str) -> None:
scripts/bench/bench_cube_shardmap_halo.py:568:        from jax.experimental import multihost_utils
scripts/bench/bench_cube_shardmap_halo.py:569:        multihost_utils.sync_global_devices(tag)
scripts/bench/bench_cube_shardmap_halo.py:582:    from jax.experimental import multihost_utils
scripts/bench/bench_cube_shardmap_halo.py:583:    gathered = multihost_utils.process_allgather(np.asarray(ms, dtype=np.float64))
scripts/bench/bench_cube_shardmap_halo.py:623:    _multihost_barrier("blk1_point_start")
scripts/bench/bench_atm_latlon_spmd_scaling.py:39:Multi-controller (route-B, ``--multicontroller``): the lat-lon analogue of the
scripts/bench/bench_atm_latlon_spmd_scaling.py:53:  srun -n 8 python bench_atm_latlon_spmd_scaling.py --multicontroller \
scripts/bench/bench_atm_latlon_spmd_scaling.py:55:  mpiexec -n 8 python ... --multicontroller --coordinator host0:9876
scripts/bench/bench_atm_latlon_spmd_scaling.py:165:    p.add_argument("--multicontroller", action="store_true",
scripts/bench/bench_atm_latlon_spmd_scaling.py:166:                   help="Route-B multi-controller: jax.distributed.initialize "
scripts/bench/bench_atm_latlon_spmd_scaling.py:184:    if args.multicontroller:
scripts/bench/bench_atm_latlon_spmd_scaling.py:191:            init_multicontroller_distributed,
scripts/bench/bench_atm_latlon_spmd_scaling.py:193:        init_multicontroller_distributed(args.coordinator)
scripts/bench/bench_atm_latlon_spmd_scaling.py:213:    if args.multicontroller and nd != avail:
scripts/bench/bench_atm_latlon_spmd_scaling.py:215:        # of the program (non-addressable participation hazard). Route-B uses
scripts/bench/bench_atm_latlon_spmd_scaling.py:218:            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
scripts/bench/bench_atm_latlon_spmd_scaling.py:257:            from jax.experimental import multihost_utils
scripts/bench/bench_atm_latlon_spmd_scaling.py:258:            multihost_utils.sync_global_devices("atm_latlon_spmd_bench_start")
scripts/bench/bench_atm_latlon_spmd_scaling.py:285:            from jax.experimental import multihost_utils
scripts/bench/bench_atm_latlon_spmd_scaling.py:286:            multihost_utils.sync_global_devices("atm_latlon_spmd_bench_end")
scripts/bench/bench_atm_latlon_spmd_scaling.py:370:        multicontroller=bool(args.multicontroller),
scripts/bench/bench_atm_latlon_spmd_scaling.py:446:            "multicontroller": bool(args.multicontroller),
scripts/bench/bench_atm_latlon_spmd_scaling.py:456:            # Route-B transport facts (socket-fallback flag): a
scripts/bench/bench_atm_latlon_spmd_scaling.py:459:            "nccl": (_nccl_report if args.multicontroller
scripts/plot/plot_scaling_paper_figure.py:35:                  "LL2048@64 26502539, LL2048@128 26534060, LL2304@96 26628072",
scripts/plot/plot_scaling_paper_figure.py:54:        scatter=[("LL1536 @64", 64, 4.97), ("LL2048 f64 @128", 128, 9.60),
scripts/plot/plot_scaling_paper_figure.py:55:                 ("LL2304 @96", 96, 7.87)],
scripts/plot/plot_scaling_paper_figure.py:56:        note="LL2048@128 = 39.1 GC/s",
scripts/bench/bench_ocean_latlon_spmd_pcg.py:11:mpi4jax — runs on route-B (cuda-jax, the RTX8000 PCIe pair) where mpi4jax
scripts/bench/bench_ocean_latlon_spmd_pcg.py:31:  GPU (route-B, the real target):
scripts/bench/bench_ocean_latlon_spmd_pcg.py:51:    Route-B is single-controller: ONE process drives every GPU on the node
scripts/bench/aggregate_bcw_scaling.py:248:    # (rank/face count).  n_devices is required for the cube route-B node-fill
packages/ocean/legoesm/ocean/dynamics/eta_floor.py:28:    SPMD (single-controller lat-band shard_map, route-B multi-GPU — no mpi4jax):
scripts/bench/bcw_scaling_ledger.py:1:"""Scaling-progress ledger for the baroclinic-wave campaign.
scripts/bench/bcw_scaling_ledger.py:3:Tracks headline scaling metrics *as a function of campaign iteration* so each
scripts/bench/bcw_scaling_ledger.py:204:    ax_e.set_title(f"{grid}: scaling vs campaign iteration", fontsize=10, loc="left")
tests/bench/test_latlon_mpi_wiring.py:14:* single-process lat-lon is still capped at 1 device (route-B SPMD is
tests/bench/test_latlon_mpi_wiring.py:73:    """Route-B SPMD is not wired: single-process lat-lon stays at 1 device.
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:456:    # SPMD (single-controller shard_map, route-B multi-GPU — no mpi4jax)
scripts/bench/metadata.py:34:#: scaling audit) — a route-A mpi4jax row, a route-B NCCL row, a gloo/TCP
scripts/bench/metadata.py:357:    - ``process_count > 1``: route-B multi-controller ``jax.distributed`` →
scripts/bench/metadata.py:394:    ``transport`` (route-B NCCL, gloo, intra-process ``xla-local``, serial)
scripts/bench/metadata.py:455:        that pass ``n_ranks`` explicitly resolve correctly, route-B /
scripts/bench/metadata.py:469:    # route-B NCCL / xla-local / serial row (codex finding 2).
scripts/bench/metadata.py:978:        from jax.experimental import multihost_utils
scripts/bench/metadata.py:979:        _sched = np.asarray(multihost_utils.process_allgather(
scripts/bench/metadata.py:1005:            from jax.experimental import multihost_utils
scripts/bench/metadata.py:1006:            multihost_utils.sync_global_devices(f"{sync_label}_{tag}")
scripts/bench/metadata.py:1070:        from jax.experimental import multihost_utils
scripts/bench/metadata.py:1072:            multihost_utils.process_allgather(my_blocks))
scripts/bench/bench_ppermute_microbench.py:19:Single-process multi-device (NVLink within a node) or multicontroller
scripts/bench/bench_ppermute_microbench.py:20:(``--multicontroller``, NCCL over the fabric) — the two give different
scripts/bench/bench_ppermute_microbench.py:32:        --multicontroller --n-devices 8
scripts/bench/bench_ppermute_microbench.py:130:    p.add_argument("--multicontroller", action="store_true",
scripts/bench/bench_ppermute_microbench.py:144:    if args.multicontroller:
scripts/bench/bench_ppermute_microbench.py:145:        from legoesm.parallel.early_init import init_multicontroller_distributed
scripts/bench/bench_ppermute_microbench.py:147:        init_multicontroller_distributed(args.coordinator)
scripts/bench/bench_ppermute_microbench.py:192:        "multicontroller": bool(args.multicontroller),
scripts/bench/bench_cube_tiled_step_scaling.py:37:        --kt 2 --resolution 192 --nlev 60 --steps 12 --multicontroller
scripts/bench/bench_cube_tiled_step_scaling.py:45:        --kt 2 --resolution 192 --nlev 60 --steps 12 --multicontroller
scripts/bench/bench_cube_tiled_step_scaling.py:105:    p.add_argument("--multicontroller", action="store_true",
scripts/bench/bench_cube_tiled_step_scaling.py:106:                   help="Route-B: one process per GPU; shared hardened "
scripts/bench/bench_cube_tiled_step_scaling.py:123:    if args.multicontroller:
scripts/bench/bench_cube_tiled_step_scaling.py:125:            init_multicontroller_distributed,
scripts/bench/bench_cube_tiled_step_scaling.py:127:        init_multicontroller_distributed(args.coordinator)
scripts/bench/bench_cube_tiled_step_scaling.py:137:    if args.multicontroller and n_devices != avail:
scripts/bench/bench_cube_tiled_step_scaling.py:139:            f"--multicontroller: 6*kt^2 ({n_devices}) must equal the "
scripts/bench/bench_cube_tiled_step_scaling.py:264:        from jax.experimental import multihost_utils
scripts/bench/bench_cube_tiled_step_scaling.py:266:        multihost_utils.sync_global_devices("cube_tiled_bench_start")
scripts/bench/bench_cube_tiled_step_scaling.py:299:        from jax.experimental import multihost_utils
scripts/bench/bench_cube_tiled_step_scaling.py:301:        multihost_utils.sync_global_devices("cube_tiled_bench_end")
scripts/bench/bench_cube_tiled_step_scaling.py:306:        # Reduce ON DEVICE: under multicontroller the state shards span
scripts/bench/bench_cube_tiled_step_scaling.py:332:        multicontroller=bool(args.multicontroller),
scripts/bench/bench_cube_tiled_step_scaling.py:374:            # Route-B transport facts (the PBS wrapper's contract): a
scripts/bench/bench_cube_tiled_step_scaling.py:376:            "nccl": (_nccl_report if args.multicontroller else None),
scripts/bench/bench_cube_tiled_step_scaling.py:380:            "multicontroller": bool(args.multicontroller),
tests/bench/test_bench_mpas_spmd_gates.py:17:tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py.
tests/bench/test_bench_mpas_spmd_gates.py:58:                 "--multicontroller", "--coordinator", "--partition-method",
tests/bench/test_bench_ocean_latlon_spmd_gates.py:15:tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py (and the
tests/bench/test_bench_ocean_latlon_spmd_gates.py:16:launcher-gated test_latlon_ocean_spmd_multicontroller.py).
tests/bench/test_bench_ocean_latlon_spmd_gates.py:54:                 "--multicontroller", "--coordinator"):
scripts/bench/run_levante_gpu_scaling.py:283:#       single-process multi-GPU (route-B SPMD) latlon step.
scripts/bench/run_levante_gpu_scaling.py:1634:                    "Single-process multi-device SPMD (route-B) is not yet "
scripts/cluster/scaling_levante/README.md:68:## Route-B (jax.distributed + NCCL) + ocean + CPU additions (2026-07)
scripts/cluster/scaling_levante/README.md:75:| `gpu_multinode_scaling.sbatch` | MULTI-NODE route-B lanes over NCCL/IB (SLURM auto-detected `jax.distributed`): A = cube `--cs-spmd` (6 GPU / 2 nodes), C = atm lat-lon `--multicontroller` (8 GPU), D = ocean `--multicontroller` (8 GPU). |
scripts/cluster/scaling_levante/README.md:79:`_env.sh` now also carries the NCCL-over-IB defaults for the route-B lanes
scripts/cluster/scaling_levante/README.md:90:## 2026-07 lane E: icosahedral/MPAS multicontroller
scripts/cluster/scaling_levante/README.md:98:`tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py`.
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:48:      --multicontroller --n-devices "$NP" \
scripts/cluster/scaling_levante/_env.sh:71:# --- NCCL over InfiniBand (route-B: jax.distributed multi-node lanes) --------
scripts/cluster/scaling_levante/mpas_s10_128.sbatch:35:    --multicontroller --n-devices 128 \
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:342:    Under multicontroller, ``jax.device_put(numpy_array, sharding)`` calls
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:343:    ``multihost_utils.assert_equal`` on the FULL array
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:347:    81.5 GiB > an 80 GB A100 (the wall that killed oc @96/@128, jobs
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:536:    Multi-controller (route-B ``jax.distributed``, mesh spanning processes):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:677:            from jax.experimental import multihost_utils
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:689:            g_struct = multihost_utils.process_allgather(struct)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:690:            g_vals = multihost_utils.process_allgather(vals)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:702:            # near-linear-in-nd wall that killed oc @96/@128 (jobs
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:720:        from jax.experimental import multihost_utils as _mhu
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:19:#   Lane C (RUN_LATLON=1) atm lat-lon lat-band multihost, 8 procs — the
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:22:#   Lane D (RUN_OCEAN=1)  ocean lat-band multihost, 8 procs
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:23:#   Lane E (RUN_MPAS=1)   icosahedral MPAS multicontroller, 6 procs
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:160:# mpi4jax >=2-node ceiling. Gate: tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:162:    echo "=== LANE C: atm latlon multihost np=$NP_LL (LL$LL_RES) ==="
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:166:        --multicontroller --n-devices "$NP_LL" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:170:        --out "$OUTDIR/latlon_multihost/spmd_scaling.jsonl"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:175:# Gate: tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:177:    echo "=== LANE D: ocean latlon multihost np=$NP_LL (LL$OC_RES) $OC_EXTRA ==="
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:184:        --multicontroller --n-devices "$NP_LL" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:188:        --out "$OUTDIR/ocean_multicontroller/ocean_spmd_scaling.jsonl"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:193:# Lane E: icosahedral/MPAS multicontroller (6 procs = 2 nodes x 3 GPUs) —
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:197:# Gate: tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:199:    echo "=== LANE E: icosahedral MPAS multicontroller np=$NP_MPAS (L$ICO_LEVEL) ==="
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:208:        --multicontroller --n-devices "$NP_MPAS" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:212:        --out "$OUTDIR/mpas_multicontroller/smoke.jsonl" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:215:        --multicontroller --n-devices "$NP_MPAS" \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:219:        --out "$OUTDIR/mpas_multicontroller/mpas_spmd_scaling.jsonl"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:247:                --multicontroller --n-devices 8 \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:262:                --multicontroller --n-devices 8 \
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:585:    # on the route-B multi-GPU path).  Checked FIRST (is_multi_process() is False
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:855:    # route-B single-process lat-lon SPMD (halo backend == "spmd",
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1672:    # with jax.lax.psum (NOT mpi4jax, not even required on the route-B multi-GPU
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:13:# Stage A: CPU-virtual parity gate; Stage B: multicontroller timed run.
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:71:echo "== Stage B: 24-GPU multicontroller timed runs =="
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:82:    --steps 12 --warmup 2 --multicontroller --out "${OUT}"
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:88:    --steps 12 --warmup 2 --multicontroller --closed-loop --out "${OUT}"
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:70:      --multicontroller --n-devices 32 \
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:22:# Context anchors (same bench, steps 12/warmup 3): LL2048@128 f32
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:39:      --multicontroller --n-devices "$3" --mode strong \
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:44:# @96 receipt banked in job 26628072 (7.872 ms); rerunning only the
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:12:# CP-COMBINING A/B at LL2048@128 (the measured/bound ~2.9 gap).
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:46:  echo "=== LL2048@128 f32 arm=$1 XLA_EXTRA='$2' ==="
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:50:      --multicontroller --n-devices 128 --mode strong \
scripts/cluster/scaling_levante/mpas_s10_192.sbatch:30:    --multicontroller --n-devices 192 \
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch:14:# measured/bound < 1, the known tell).  Tiles: 16x4096 (the @128 band),
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch:50:      --multicontroller --n-devices 64 --mode strong \
scripts/plot/plot_scaling_indicators.py:1:"""Plot the EVOLUTION of the main scaling indicators across campaign
scripts/plot/plot_scaling_indicators.py:116:        "legoESM scaling-indicator evolution across campaign iterations "
scripts/plot/plot_levante_gpu_scaling_comparison.py:2:"""Combine per-GPU-count JSON results from run_levante_gpu_scaling.sh campaigns
scripts/plot/plot_levante_gpu_scaling_comparison.py:52:    """Load all scaling JSON files from a campaign directory tree."""
scripts/run/run_amip.py:28:# Route-B multicontroller (--multicontroller) initializes jax.distributed in
scripts/run/run_amip.py:29:# main() via init_multicontroller_distributed's explicit --coordinator path (a
scripts/run/run_amip.py:35:if "--multicontroller" not in sys.argv:
scripts/run/run_amip.py:1541:              "band-local. Single-process by default; add --multicontroller for "
scripts/run/run_amip.py:1542:              "the multi-node route-B lane. Distinct from --distributed (MPI)."))
scripts/run/run_amip.py:1552:        "--multicontroller", action="store_true", default=False,
scripts/run/run_amip.py:1553:        help=("Promote --enable-latlon-spmd to ROUTE-B (jax.distributed, "
scripts/run/run_amip.py:1561:              "--multicontroller under mpiexec (reads Open MPI OMPI_* / Cray "
scripts/run/run_amip.py:1989:    # Route-B multicontroller (--multicontroller) also launches under mpiexec/srun
scripts/run/run_amip.py:1994:    if not args.distributed and not getattr(args, "multicontroller", False) and (
scripts/run/run_amip.py:2586:    # Route-B multicontroller: initialize jax.distributed BEFORE any device work
scripts/run/run_amip.py:2591:    # falls back to SLURM/OMPI auto-detect. No-op unless --multicontroller.
scripts/run/run_amip.py:2592:    if getattr(args, "multicontroller", False):
scripts/run/run_amip.py:2594:            parser.error("--multicontroller requires --enable-latlon-spmd (it "
scripts/run/run_amip.py:2595:                         "is the route-B transport for the lat-band SPMD lane).")
scripts/run/run_amip.py:2596:        from legoesm.parallel.early_init import init_multicontroller_distributed
scripts/run/run_amip.py:2597:        init_multicontroller_distributed(getattr(args, "coordinator", None))
scripts/plot/plot_bcw_scaling.py:1:"""Publication scaling figures for the baroclinic-wave campaign.
scripts/validate/validate_driver_cs_spmd_parity.py:11:  final state (``multihost_utils.process_allgather``) and compares it to
scripts/validate/validate_driver_cs_spmd_parity.py:71:    from jax.experimental import multihost_utils
scripts/validate/validate_driver_cs_spmd_parity.py:88:            data = multihost_utils.process_allgather(data, tiled=True)
scripts/validate/validate_driver_cs_spmd_parity.py:248:        from jax.experimental import multihost_utils
scripts/validate/validate_driver_cs_spmd_parity.py:249:        rc_arr = multihost_utils.broadcast_one_to_all(rc_arr)
scripts/validate/validate_tiled_fv3_sw_multinode.py:163:    from jax.experimental import multihost_utils
scripts/validate/validate_tiled_fv3_sw_multinode.py:164:    worst_all = multihost_utils.process_allgather(
scripts/run/run_omip.py:130:    # Route-B multicontroller (jax.distributed cross-process NCCL): promote the
scripts/run/run_omip.py:132:    multicontroller: bool = False
scripts/run/run_omip.py:221:        multicontroller=getattr(args, "multicontroller", False),
scripts/run/run_omip.py:716:                       "bench_ocean_latlon_spmd_scaling --multicontroller); "
scripts/run/run_omip.py:726:    p.add_argument("--multicontroller", action="store_true", default=False,
scripts/run/run_omip.py:728:                       "Promote --enable-latlon-spmd to ROUTE-B "
scripts/run/run_omip.py:740:                       "--multicontroller under mpiexec (Open MPI OMPI_* / "
scripts/run/run_omip.py:3254:    # Route-B multicontroller: only rank 0 writes restart/snapshot files, but
scripts/run/run_omip.py:3278:                # Route-B: a rank-0 write error (e.g. ENOSPC) must NOT raise —
scripts/run/run_omip.py:3303:            # Route-B: the exit decision MUST be an all-rank consensus. Ranks
scripts/run/run_omip.py:3309:            from jax.experimental import multihost_utils
scripts/run/run_omip.py:3311:                multihost_utils.process_allgather(
scripts/run/run_omip.py:3325:        # fname is None on route-B non-root ranks (gather ran, no write).
scripts/run/run_omip.py:3380:        # under route-B (shards span processes).  Skip it when spmd_step is set.
scripts/run/run_omip.py:3522:            # state (_st_diag): under route-B the raw ``state.eta`` is sharded
scripts/run/run_omip.py:3583:                # fname is None on route-B non-root ranks (gather ran, no write).
scripts/run/run_omip.py:3771:                # fname is None on route-B non-root ranks (gather ran, no write).
scripts/run/run_omip.py:3810:    ``write=False`` (route-B non-root ranks) skips every file write but still
scripts/run/run_omip.py:3819:    # consistency); route-B non-root ranks return it here WITHOUT writing any
scripts/run/run_omip.py:3968:    # --multicontroller is ONLY the route-B transport for the lat-band ocean
scripts/run/run_omip.py:3974:    if run_config.multicontroller and not run_config.enable_latlon_spmd:
scripts/run/run_omip.py:3976:            "--multicontroller requires --enable-latlon-spmd (it is the "
scripts/run/run_omip.py:3977:            "route-B transport for the lat-band ocean SPMD step). Without the "
scripts/run/run_omip.py:4785:        _multi = run_config.multicontroller
scripts/run/run_omip.py:4789:                "--multicontroller is set; multi-node ocean scaling needs "
scripts/run/run_omip.py:4790:                "the route-B lane (jax.distributed cross-process NCCL).")
scripts/run/run_omip.py:4792:            # Route-B: the mesh spans ALL global devices (one band per device
scripts/run/run_omip.py:4799:                    f"--multicontroller uses ALL global devices ({_nd} across "
scripts/run/run_omip.py:4832:                _lane = "route-B multicontroller" if _multi else "single-controller"
scripts/run/run_omip.py:4873:    # spmd_gather collective above: under route-B multiproc only rank 0
scripts/run/run_omip.py:4878:    # Route-B: only rank 0 writes output files (concurrent writes to the same
scripts/run/run_omip.py:4897:    # Save output.  Route-B: a rank-0 write failure must NOT raise past this
scripts/run/run_omip.py:4989:    # Route-B multicontroller (jax.distributed cross-process NCCL): initialize
scripts/run/run_omip.py:4992:    # calls that initialise the XLA backend".  No-op unless --multicontroller.
scripts/run/run_omip.py:4993:    if getattr(args, "multicontroller", False):
scripts/run/run_omip.py:4994:        from legoesm.parallel.early_init import init_multicontroller_distributed
scripts/run/run_omip.py:4995:        init_multicontroller_distributed(getattr(args, "coordinator", None))
scripts/run/run_omip.py:5007:    # Route-B: every process runs main() to a consistent exit code, but only
scripts/cluster/scaling_derecho/finalize_scaling.sh:3:# Finalize a CPU-vs-A100 scaling campaign: aggregate every job under <outdir>
scripts/cluster/scaling_derecho/README.md:486:| `gpu_multinode_scaling.pbs` | MULTI-NODE GPU lanes over jax.distributed + NCCL: A = cube `--cs-spmd` (6 GPU / 2 nodes), C = atm lat-lon `--multicontroller` (8 GPU), D = ocean `--multicontroller` (8 GPU); plus the optional route-A CUDA-aware mpi4jax lane (`RUN_ROUTEA=1`, needs the overlay env) and the comm-tuning A/B ladder (`RUN_TUNE=1`, lane T below). |
scripts/cluster/scaling_derecho/README.md:507:## 2026-07 lane E: icosahedral/MPAS multicontroller
scripts/cluster/scaling_derecho/README.md:518:`tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py`.
scripts/cluster/scaling_derecho/README.md:523:Once the route-B lanes are green on this machine, the remaining strong-
scripts/cluster/scaling_derecho/README.md:539:   route-B latlon lanes.**  Stays per-run opt-in (recompiles after the
scripts/cluster/scaling_derecho/mc_nccl_probe.py:3:Answers ONE infra question on Derecho: does route-B ``jax.distributed``
scripts/cluster/scaling_derecho/mc_nccl_probe.py:15:     route-B is the multi-node halo transport (compare the ~0.27 ms mpi4jax
scripts/cluster/scaling_derecho/submit_scaling.sh:172:        # Opt-in route-B cube CPU lane (#764): jax.distributed/gloo cs-spmd,
scripts/cluster/scaling_derecho/submit_scaling.sh:180:            submit "cpu ${GRID} res=${R} (route-B cs-spmd)" \
scripts/cluster/scaling_derecho/mc_nccl_canary.sh:11:# Route-B multi-controller NCCL CANARY (2 nodes x 4 A100 = 8 GPUs).
scripts/cluster/scaling_derecho/mc_nccl_canary.sh:17:# If YES, route-B is the multi-node GPU halo transport (all-NCCL, no mpi4jax,
scripts/cluster/scaling_derecho/mc_nccl_canary.sh:31:#                               --multicontroller, 8 GPU
scripts/cluster/scaling_derecho/mc_nccl_canary.sh:94:    tests/parallel/test_atm_latlon_spmd_multicontroller.py
scripts/cluster/scaling_derecho/mc_nccl_canary.sh:96:    tests/parallel/test_latlon_ocean_spmd_multicontroller.py
scripts/cluster/scaling_derecho/mc_nccl_canary.sh:101:    --multicontroller --coordinator "${COORD_HOST}:${PORT_C1}" \
scripts/cluster/scaling_derecho/mc_nccl_canary.sh:105:    --multicontroller --coordinator "${COORD_HOST}:${PORT_C2}" \
packages/ocean/legoesm/ocean/conservation.py:57:    SPMD (single-controller lat-band shard_map, route-B multi-GPU — no mpi4jax):
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:72:RUN_LATLON="${RUN_LATLON:-1}"       # lane C: atm latlon lat-band multihost
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:73:RUN_OCEAN="${RUN_OCEAN:-1}"         # lane D: ocean lat-band multihost
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:74:RUN_MPAS="${RUN_MPAS:-1}"           # lane E: icosahedral MPAS multicontroller
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:89:# XLA collective-permute combining + pipelined p2p. #1113 found the route-B MPAS
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:165:# Lane C: ATM lat-lon lat-band multihost (8 procs = 2 nodes x 4 GPUs, NCCL)
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:166:# — native ppermute band halos (route-B) replace the route-A mpi4jax leg
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:168:# Gate: tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:171:    echo "=== LANE C: atm latlon multihost, np=8 over 2 nodes (LL$LL_RES) ==="
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:176:            --multicontroller --n-devices 8 \
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:180:            --out "$OUTDIR/latlon_multihost/spmd_scaling.jsonl" < /dev/null
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:187:# Lane D: OCEAN lat-band multihost (8 procs = 2 nodes x 4 GPUs, NCCL) via
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:188:# the bench's --transport spmd --multihost lane (parity/conservation gates
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:190:# Gate: tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:193:    echo "=== LANE D: ocean latlon multihost, np=8 over 2 nodes (LL$OC_RES) ==="
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:198:            --multicontroller --n-devices 8 \
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:202:            --out "$OUTDIR/ocean_multicontroller/ocean_spmd_scaling.jsonl" < /dev/null
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:209:# Lane E: icosahedral/MPAS multicontroller (6 procs = 2 nodes x 3 GPUs, NCCL)
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:214:# Gate: tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py.
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:217:    echo "=== LANE E: icosahedral MPAS multicontroller, np=6 over 2 nodes (L$ICO_LEVEL) ==="
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:229:            --multicontroller --n-devices 6 \
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:233:            --out "$OUTDIR/mpas_multicontroller/smoke.jsonl" < /dev/null \
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:236:            --multicontroller --n-devices 6 \
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:240:            --out "$OUTDIR/mpas_multicontroller/mpas_spmd_scaling.jsonl" < /dev/null
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:282:# next rungs now that the route-B lanes are green (see
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:314:                --multicontroller --n-devices 8 \
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:351:                --multicontroller --n-devices 8 \
scripts/cluster/scaling_derecho/cube_tiled_step.pbs:17:# Stage B (multicontroller, one process per GPU, route-B NCCL): the timed
scripts/cluster/scaling_derecho/cube_tiled_step.pbs:68:echo "== Stage B: 24-GPU multicontroller timed runs =="
scripts/cluster/scaling_derecho/cube_tiled_step.pbs:104:    --steps 12 --warmup 2 --multicontroller --out "${OUT}"
scripts/cluster/scaling_derecho/cube_tiled_step.pbs:113:    --steps 12 --warmup 2 --multicontroller --closed-loop --out "${OUT}"
scripts/cluster/scaling_derecho/cube_scaling_cpu_routeb.sh:13:# route-B stack the latlon sweep uses (bench_atm_latlon_spmd_scaling.py),
scripts/cluster/scaling_derecho/cube_scaling_cpu_routeb.sh:73:echo "=== cube CPU route-B strong scaling: ranks=[$CUBE_SPMD_RANKS] on ${_CORES} cores  res=[$STRONG_RES] ==="
scripts/cluster/scaling_derecho/cube_scaling_cpu_routeb.sh:96:    echo "--- cube route-B ranks=$N x ${THREADS} threads  res=$RES ---"
tests/unit/test_early_init.py:258:# init_multicontroller_distributed: the shared --multicontroller entry point
tests/unit/test_early_init.py:259:# (ocean/atm SPMD benches + the run_omip route-B driver).
tests/unit/test_early_init.py:262:def test_multicontroller_coordinator_uses_launcher_env(monkeypatch):
tests/unit/test_early_init.py:270:    early_init.init_multicontroller_distributed("host:5000")
tests/unit/test_early_init.py:281:def test_multicontroller_coordinator_missing_env_raises(monkeypatch):
tests/unit/test_early_init.py:293:        early_init.init_multicontroller_distributed("host:5000")
tests/unit/test_early_init.py:297:def test_multicontroller_no_coordinator_delegates_to_fallback(monkeypatch):
tests/unit/test_early_init.py:303:    early_init.init_multicontroller_distributed(None)
tests/unit/test_run_omip_cli.py:73:def test_multicontroller_flags_round_trip():
tests/unit/test_run_omip_cli.py:74:    """--multicontroller / --coordinator parse and reach OMIPRunConfig
tests/unit/test_run_omip_cli.py:75:    (the route-B cross-process lane, part 2c of the ocean-SPMD promotion)."""
tests/unit/test_run_omip_cli.py:77:    assert args.multicontroller is False
tests/unit/test_run_omip_cli.py:80:    assert cfg.multicontroller is False
tests/unit/test_run_omip_cli.py:84:        "--grid", "latlon", "--enable-latlon-spmd", "--multicontroller",
tests/unit/test_run_omip_cli.py:87:    assert cfg.multicontroller is True
tests/unit/test_run_omip_cli.py:91:def test_multicontroller_without_spmd_refused():
tests/unit/test_run_omip_cli.py:92:    """--multicontroller alone (no --enable-latlon-spmd) must hard-fail BEFORE
tests/unit/test_run_omip_cli.py:98:    args = parse_args(["--grid", "latlon", "--multicontroller"])
tests/unit/test_device_config.py:343:    def test_configure_tpu_multihost_does_not_crash_on_jax_0_9(self):
tests/unit/test_bench_ocean_latlon_spmd_scaling.py:6:tests/parallel/test_latlon_ocean_spmd_step.py; the --multicontroller path is
tests/unit/test_bench_ocean_latlon_spmd_scaling.py:56:    assert rec["multicontroller"] is False
tests/unit/test_run_amip_cli.py:2217:def test_multicontroller_coordinator_flags_parse():
tests/unit/test_run_amip_cli.py:2218:    """Route-B flags round-trip through the parser (they are RUN args consumed
tests/unit/test_run_amip_cli.py:2221:    a = parser.parse_args(["--enable-latlon-spmd", "--multicontroller",
tests/unit/test_run_amip_cli.py:2223:    assert a.multicontroller is True
tests/unit/test_run_amip_cli.py:2227:    assert d.multicontroller is False
tests/unit/test_run_amip_cli.py:2231:def test_multicontroller_requires_enable_latlon_spmd(capsys):
tests/unit/test_run_amip_cli.py:2232:    """--multicontroller without --enable-latlon-spmd is refused in main()
tests/unit/test_run_amip_cli.py:2233:    BEFORE any device work (it is only the route-B transport for that lane)."""
tests/unit/test_run_amip_cli.py:2237:              "--multicontroller"])
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:74:    repo's existing ``*_multicontroller_selfspawn.py`` gates hit this too).
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:190:    from multihost_harness import run_federated  # reuse retry/timeout/kill
tests/parallel/test_ppermute_edge_coloring.py:1:"""Edge-coloring correctness + round-count for the route-B MPAS ppermute
tests/parallel/test_ppermute_edge_coloring.py:4:Each color is one bidirectional ppermute ROUND and the route-B lane is
tests/parallel/test_atm_latlon_spmd_step.py:281:    mechanisms are legal), so the route-B gather cannot silently diverge.
tests/parallel/test_early_init_hardening.py:1:"""Route-B hardening gates for ``legoesm.parallel.early_init`` (audit item 6).
tests/parallel/test_early_init_hardening.py:5:covered by the multicontroller self-spawn suites.
tests/ocean/unit/test_sharded_geom_fingerprint.py:7:multicontroller allgather wiring is exercised by the distributed suite).
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:1:"""SELF-SPAWNING route-B multicontroller gate for the ``run_omip`` DRIVER.
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:4:(``test_latlon_ocean_spmd_multicontroller_selfspawn.py``, which drives
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:7:(``scripts/run/run_omip.py --enable-latlon-spmd --multicontroller``)
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:11:* the early ``init_multicontroller_distributed`` bootstrap firing BEFORE any
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:52:def test_run_omip_two_process_selfspawn_multicontroller(tmp_path):
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:53:    from multihost_harness import run_federated
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:56:    # read it (read-only) — this drives the richest route-B path: the
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:88:            "--enable-latlon-spmd", "--multicontroller",
tests/parallel/test_run_omip_latlon_spmd_multicontroller_selfspawn.py:129:            f"({n_lat + 1}) — route-B save missed the gather")
tests/parallel/test_latlon_spmd_shard_leaf.py:11:``*_multicontroller_selfspawn`` tests (which spawn real processes); here
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:1:"""Route-B multi-controller gate for the lat-band SPMD ocean step.
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:3:The ocean twin of ``test_atm_latlon_spmd_multicontroller.py``: N processes
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:22:      tests/parallel/test_latlon_ocean_spmd_multicontroller.py -x -q
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:97:def test_multicontroller_ocean_step_matches_serial():
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:3:Companion to tests/parallel/test_latlon_ocean_spmd_multicontroller.py (the
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:7:production bench (``bench_ocean_latlon_spmd_scaling.py --multicontroller``)
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:15:the bench reads for rank/size; port races retry via multihost_harness.
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:37:    from multihost_harness import run_federated
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:48:            "--multicontroller", "--coordinator", f"localhost:{port}",
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:74:    # Rank 0 owns the JSONL; the record carries the multicontroller fields.
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:80:    assert rec["multicontroller"] is True
tests/parallel/test_mpas_atm_native_step.py:28:``test_mpas_spmd_multicontroller_selfspawn.py``; the route-A exchange
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:1:"""SELF-SPAWNING route-B multicontroller gate for the ``run_amip`` DRIVER.
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:3:Companion to the run_omip route-B gate
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:4:(``test_run_omip_latlon_spmd_multicontroller_selfspawn.py``): THIS variant
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:6:(``scripts/run/run_amip.py --enable-latlon-spmd --multicontroller``) end-to-end
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:10:* the ``main()`` ``init_multicontroller_distributed`` bootstrap firing AFTER
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:12:  correctly SKIPPED for ``--multicontroller`` (it would try to load libmpi
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:46:def test_run_amip_two_process_selfspawn_multicontroller(tmp_path):
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:47:    from multihost_harness import run_federated
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:66:            # disable them so the route-B run reaches COMPLETED (default diag=5).
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:78:            "--enable-latlon-spmd", "--multicontroller",
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:97:    assert "route-B multicontroller" in root, (
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:98:        f"no route-B banner on rank 0 — the multicontroller lane did not "
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:108:    assert 100.0 < max_T < 1e3, f"max|T|={max_T} unphysical (route-B blowup?)"
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:109:    assert 0.0 < max_u < 1e3, f"max|u|={max_u} unphysical (route-B blowup?)"
tests/parallel/test_spmd_distributed_mode.py:151:    from jax.experimental import multihost_utils
tests/parallel/test_spmd_distributed_mode.py:159:    monkeypatch.setattr(multihost_utils, "process_allgather", _fake_allgather)
tests/parallel/test_spmd_distributed_mode.py:169:    from jax.experimental import multihost_utils
tests/parallel/test_spmd_distributed_mode.py:177:    monkeypatch.setattr(multihost_utils, "process_allgather", _fake_allgather)
tests/ocean/unit/test_multigrid_preconditioner.py:177:    """route-B guard gap: lat-lon SPMD arms the 'spmd' halo backend with
tests/parallel/test_atm_latlon_bandlocal_build.py:6:lat mesh — this is the correctness contract that lets the route-B bench skip
packages/core/legoesm/parallel/latlon_spmd.py:274:    Multi-controller (``jax.process_count() > 1``, route-B ``jax.distributed``):
packages/core/legoesm/parallel/latlon_spmd.py:303:    Multi-controller (``jax.process_count() > 1``, route-B ``jax.distributed``,
tests/parallel/multihost_harness.py:1:"""Shared launcher for the 2-process jax.distributed (multihost) gates.
packages/core/legoesm/parallel/early_init.py:87:    The route-B hazard this guards: ``jax.distributed.initialize`` (or an
packages/core/legoesm/parallel/early_init.py:140:    """Per-process ``local_device_ids`` for a multicontroller launch.
packages/core/legoesm/parallel/early_init.py:171:    """Best-effort NCCL transport facts for run metadata (route-B analog of
packages/core/legoesm/parallel/early_init.py:180:    sockets (git cba9715b2: 'route-B NCCL works cross-node but
packages/core/legoesm/parallel/early_init.py:330:def init_multicontroller_distributed(coordinator: str | None = None) -> None:
packages/core/legoesm/parallel/early_init.py:331:    """Initialize ``jax.distributed`` for a route-B multicontroller launch.
packages/core/legoesm/parallel/early_init.py:333:    Shared by every ``--multicontroller`` entry point (the ocean/atm SPMD
packages/core/legoesm/parallel/early_init.py:334:    benches and the ``run_omip`` route-B driver) so the launcher-env contract
packages/core/legoesm/parallel/sharded_dynamics.py:1490:    route-B MPAS lane is round-latency-bound (#1113), so excess colors are
packages/core/legoesm/parallel/sharded_dynamics.py:1647:    #    ROUND, and the route-B lane is round-latency-bound (#1113), so
packages/core/legoesm/parallel/latlon_mpi.py:1689:# Fused multi-field halo exchange (scaling campaign, audit lever O4)
packages/core/legoesm/parallel/reductions.py:701:    (route-B, pure-jax multi-GPU — no mpi4jax), where the barotropic PCG's
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2485:    — the route-B one-process-per-GPU lane): single-device / single-process
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2544:            from jax.experimental import multihost_utils
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2545:            multihost_utils.sync_global_devices(tag)
tests/parallel/test_atm_latlon_spmd_multicontroller.py:1:"""Route-B multi-controller gate for the lat-band SPMD atm step.
tests/parallel/test_atm_latlon_spmd_multicontroller.py:25:      tests/parallel/test_atm_latlon_spmd_multicontroller.py -x -q
tests/parallel/test_atm_latlon_spmd_multicontroller.py:97:def test_multicontroller_step_matches_serial():
packages/core/legoesm/grids/halo_latlon.py:723:    phase (scaling campaign audit lever O4).
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:4:tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py: spawns
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:6:(``bench_mpas_spmd_scaling.py --multicontroller``) end-to-end with the
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:17:the bench reads for rank/size; port races retry via multihost_harness.
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:39:    from multihost_harness import run_federated
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:50:            "--multicontroller", "--coordinator", f"localhost:{port}",
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:72:    # Rank 0 owns the JSONL; the record carries the multicontroller fields.
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:78:    assert rec["multicontroller"] is True
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:94:    from multihost_harness import run_federated
tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:105:            "--multicontroller", "--coordinator", f"localhost:{port}",
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py:3:Companion to tests/parallel/test_atm_latlon_spmd_multicontroller.py (the
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py:7:port-race retries via multihost_harness.
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py:122:        print(f"rank{rank} MULTIHOST PARITY FAIL: " + "; ".join(failures),
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py:125:    print(f"rank{rank} multihost parity OK", flush=True)
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py:130:    from multihost_harness import run_federated
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py:145:        assert "multihost parity OK" in out
packages/coupler/legoesm/driver/model_driver.py:5139:        from jax.experimental import multihost_utils as _mhu
packages/coupler/legoesm/driver/model_driver.py:5172:        from jax.experimental import multihost_utils as _mhu
packages/coupler/legoesm/driver/model_driver.py:8949:        Under route-B multicontroller (``jax.process_count() > 1``, after
packages/coupler/legoesm/driver/model_driver.py:8950:        ``init_multicontroller_distributed``) ``jax.devices()`` is the GLOBAL
packages/coupler/legoesm/driver/model_driver.py:8961:                f"--multicontroller (route-B) uses ALL {len(devs)} global "
packages/coupler/legoesm/driver/model_driver.py:9233:                "operator-split tiled cube: multicontroller (route-B "
packages/coupler/legoesm/driver/model_driver.py:9352:        # multi-process (route-B) comm-init cannot deadlock.  No-op
packages/coupler/legoesm/driver/model_driver.py:9614:        # (route-B one-process-per-GPU) comm-init cannot deadlock.  No-op
packages/coupler/legoesm/driver/model_driver.py:9744:        # Route-B multicontroller (jax.distributed cross-process NCCL): the mesh
packages/coupler/legoesm/driver/model_driver.py:9879:            _lane = "route-B multicontroller" if _mp else "single-controller"
packages/coupler/legoesm/driver/model_driver.py:9947:            # route-B (``_mp``) -> a jit-identity with replicated out_shardings
packages/coupler/legoesm/driver/model_driver.py:10016:            # rank; the selfspawn route-B gate greps this line + the finite
packages/coupler/legoesm/driver/distributed_checkpoint.py:18:``jax.experimental.multihost_utils`` for efficient multi-host array
packages/coupler/legoesm/driver/distributed_checkpoint.py:247:    Uses ``jax.experimental.multihost_utils`` for efficient multi-host
packages/coupler/legoesm/driver/distributed_checkpoint.py:269:    from jax.experimental import multihost_utils
packages/coupler/legoesm/driver/distributed_checkpoint.py:290:    multihost_utils.sync_global_devices("save_checkpoint_sharded_pre")
packages/coupler/legoesm/driver/distributed_checkpoint.py:310:    multihost_utils.sync_global_devices("save_checkpoint_sharded_post")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py:1126:        # (route-A MPI).  Under multi-controller SPMD (route-B, federated
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:98:    removes the per-process global-state BUILD that made the route-B lat-lon
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:185:    Multi-controller (route-B ``jax.distributed``, mesh spanning processes):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:35:argument (SegmentForcing doctrine) before it can be scanned; multicontroller
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:36:route-B for either tiled lane; segment-scanning the bench

exec
/usr/bin/bash -lc "rg -n \"jax\\.device_put|device_put\\(|make_array_from_callback\" packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py; sed -n '1,245p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py; sed -n '610,755p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py; rg -n -C 4 \"def _lat_spec|_lat_spec\\(\" packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 266ms:
342:    Under multicontroller, ``jax.device_put(numpy_array, sharding)`` calls
351:    ``jax.make_array_from_callback`` supplies each process's addressable
360:        return jax.device_put(jnp.asarray(host), sharding)
361:    return jax.make_array_from_callback(
708:            # make_array_from_callback hands each process exactly its
712:        return jax.make_array_from_callback(
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
102-            names.append(name)
103-    return names
104-
105-
106:def _lat_spec(x):
107-    """Lat-band PartitionSpec for an array leaf: shard axis 0, replicate rest."""
108-    nd = int(getattr(x, "ndim", np.ndim(x)))
109-    if nd < 1:
110-        return P()                     # scalar -> replicated
--
396-
397-    def _shard_cell(field):
398-        if field is None:
399-            return None
400:        sh = NamedSharding(mesh, _lat_spec(field.data))
401-        return field.replace(data=_addressable_shard_put(field.data, sh))
402-
403-    def _shard_v(field):
404-        if field is None:
--
414-        # bit round-trip of that one row.  NOT for a tripole north fold (raises
415-        # in make_sharded_ocean_step) where the top row is a live fold partner.
416-        nlat1 = field.data.shape[0]
417-        v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
418:        sh = NamedSharding(mesh, _lat_spec(v_lower))
419-        return field.replace(data=_addressable_shard_put(v_lower, sh))
420-
421-    updates = {}
422-    for name in _V_STAGGERED_STATE_FIELDS:
--
434-        else:
435-            # Non-Field, non-None leaf (e.g. a raw array carry like psi).
436-            # Shard 2-D+ on lat, replicate lower-rank — matches shard_pytree.
437-            arr = jnp.asarray(val)
438:            spec = _lat_spec(arr) if arr.ndim >= 1 else P()
439-            updates[name] = _addressable_shard_put(arr, NamedSharding(mesh, spec))
440-    return state._replace(**updates)
441-
442-
--
457-    def _put(leaf):
458-        if leaf is None:
459-            return None
460-        arr = jnp.asarray(leaf)
461:        return _addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
462-
463-    return jax.tree.map(_put, forcing)
464-
465-

exec
/usr/bin/bash -lc "rg --files | rg '("'^|/)(bench_ocean_latlon_spmd_scaling|.*ocean.*scaling.*)'"\\.(py|sh|pbs)"'$|test_latlon_ocean_spmd'"' ; rg -n -C 3 \"device_put|make_array_from_callback|shard_state_latlon|shard_forcing_(stack_)?latlon|make_sharded_ocean_step\" benchmarks scripts packages tests --glob '*.{py,sh,pbs}' | rg -C 3 'bench_ocean|sharded_ocean_step|device_put|make_array'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 1832ms:
tests/bench/test_bench_ocean_mpas_scaling.py
scripts/bench/bench_ocean_mpas_scaling.py
scripts/bench/bench_ocean_mpi_scaling.py
scripts/bench/bench_ocean_gpu_scaling.py
scripts/bench/bench_ocean_latlon_spmd_scaling.py
tests/ocean/unit/test_ah_lat_scaling.py
tests/unit/test_bench_ocean_latlon_spmd_scaling.py
tests/parallel/test_latlon_ocean_spmd_step.py
tests/parallel/test_latlon_ocean_spmd_multicontroller.py
tests/parallel/test_latlon_ocean_spmd_tripole.py
tests/parallel/test_latlon_ocean_spmd_multiprocess.py
tests/parallel/test_latlon_ocean_spmd_wide_halo.py
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs
scripts/cluster/scaling_derecho/ocean_cpu_scaling.pbs
rg: benchmarks: No such file or directory (os error 2)
tests/unit/test_run_omip_latlon_spmd.py-53-
tests/unit/test_run_omip_latlon_spmd.py-54-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/unit/test_run_omip_latlon_spmd.py-55-        gather_state_latlon,
tests/unit/test_run_omip_latlon_spmd.py:56:        make_sharded_ocean_step,
tests/unit/test_run_omip_latlon_spmd.py:57:        shard_state_latlon,
tests/unit/test_run_omip_latlon_spmd.py-58-    )
tests/unit/test_run_omip_latlon_spmd.py-59-    from legoesm.parallel.mesh import create_latlon_mesh
--
tests/unit/test_run_omip_latlon_spmd.py-79-
tests/unit/test_run_omip_latlon_spmd.py-80-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py-81-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py:82:    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py:83:    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-84-    ckpt_dir = tmp_path / "spmd"
tests/unit/test_run_omip_latlon_spmd.py-85-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
--
tests/unit/test_run_omip_latlon_spmd.py-166-    import jax
--
tests/unit/test_run_omip_latlon_spmd.py-177-
tests/unit/test_run_omip_latlon_spmd.py-178-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/unit/test_run_omip_latlon_spmd.py-179-        gather_state_latlon,
tests/unit/test_run_omip_latlon_spmd.py:180:        make_sharded_ocean_step,
tests/unit/test_run_omip_latlon_spmd.py:181:        shard_forcing_stack_latlon,
tests/unit/test_run_omip_latlon_spmd.py:182:        shard_state_latlon,
tests/unit/test_run_omip_latlon_spmd.py-183-    )
--
tests/unit/test_run_omip_latlon_spmd.py-216-    assert ok_serial, "serial JRA55 loop reported not-ok"
tests/unit/test_run_omip_latlon_spmd.py-217-
tests/unit/test_run_omip_latlon_spmd.py-218-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py:219:    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-220-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py:221:    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-222-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
--
tests/parallel/test_tiled_fix_ps_mass.py-58-    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
tests/parallel/test_tiled_fix_ps_mass.py-59-    stage = make_tiled_fix_ps_mass_stage_2d(mesh, grid, N, KT)
tests/parallel/test_tiled_fix_ps_mass.py-60-    fo = NamedSharding(mesh, P("face", None, None))
tests/parallel/test_tiled_fix_ps_mass.py:61:    t = np.asarray(stage(jax.device_put(p_s_new, fo), jax.device_put(p_s_old, fo)))
tests/parallel/test_tiled_fix_ps_mass.py-62-
tests/parallel/test_tiled_fix_ps_mass.py-63-    # (a) match the global op (psum reorders the global scalars -> ULP).
tests/parallel/test_tiled_fix_ps_mass.py-64-    rel = float(np.max(np.abs(t - g))) / (float(np.max(np.abs(g))) + 1e-300)
--
tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py-179-    fo = NamedSharding(mesh, P("face", None, None))
tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py-180-    fw5 = NamedSharding(mesh, P("face", None, None, None, None))
tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py-181-    ut, vt, Tt, pst, qt = step(
tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py:182:        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py:183:        jax.device_put(T, fw), jax.device_put(p_s, fo),
tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py:184:        jax.device_put(phis, fo), jax.device_put(q_pack, fw5))
tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py-185-
tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py-186-    nl = N // KT
tests/parallel/test_tiled_fv3_hydrostatic_moist_step.py-187-    # D-grid corner fields (u_d, v_d) -> block-wise corner compare; cc fields
--
scripts/bench/bench_mpas_spmd_scaling.py-146-    model = MPASPrimitiveEquationModel(mesh_model, sigma, cfg)
scripts/bench/bench_mpas_spmd_scaling.py-147-    # #1100 MPAS twin: the timed path never global-builds the state.
scripts/bench/bench_mpas_spmd_scaling.py-148-    # build_sharded_baroclinic_wave_state_mpas creates every leaf via
scripts/bench/bench_mpas_spmd_scaling.py:149:    # jax.make_array_from_callback (only THIS process's shard rows are
scripts/bench/bench_mpas_spmd_scaling.py-150-    # ever materialised; value-identical (few-ULP contract, measured
scripts/bench/bench_mpas_spmd_scaling.py-151-    # exact on the pinned CPU stack) to global-build + shard_pytree —
scripts/bench/bench_mpas_spmd_scaling.py-152-    # tests/parallel/test_mpas_partitionlocal_build.py).  The GLOBAL
--
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-163-        return sm_norm(x_final)
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-164-
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-165-    # Multi-controller idiom: the SAME host-global array is built identically
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:166:    # on every process (deterministic np.arange), and device_put slices out
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-167-    # each process's addressable shards (matches test_atm_latlon_spmd_step.py
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-168-    # and shard_state_atm_latlon; no cross-process transfer of the global
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-169-    # array). The sharding spans all 24 (incl. non-addressable) mesh devices.
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-170-    g0 = (1.0 + np.arange(N_GLOBAL_DEVICES * NH * NH, dtype=np.float64)
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-171-          ).reshape(*MESH_SHAPE, NH, NH)
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:172:    x0 = jax.device_put(g0, NamedSharding(mesh, P(*AXES)))
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-173-    # ``run`` returns a fully-REPLICATED (out_specs=P()) global scalar; read
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-174-    # THIS process's local replica directly (unambiguously addressable) rather
tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py-175-    # than relying on np.asarray of a multi-process global array.
--
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-786-            )
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-787-
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-788-        if self._use_cpu_for_spectral:
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py:789:            state_cpu = jax.device_put(state, self._cpu_device)
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-790-            result_cpu = dispatch_integrator(state_cpu, tendency_fn, dt, self.config.time_integrator)
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-791-            if self.config.use_conservation_fixer:
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-792-                result_cpu = _spectral_conservation_fixer(
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-793-                    result_cpu, state_cpu, self.grid, self.z_coord, self.config,
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-794-                )
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py:795:            return jax.device_put(result_cpu, self._default_device)
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-796-
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-797-        result = dispatch_integrator(state, tendency_fn, dt, self.config.time_integrator)
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-798-        if self.config.use_conservation_fixer:
--
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-854-
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-855-    def _integrate_on_cpu(self, state, n_steps, dt, save_every):
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-856-        """Batch integration on CPU: transfer once, not per step."""
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py:857:        state_cpu = jax.device_put(state, self._cpu_device)
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-858-        trajectory_cpu = [state_cpu]
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-859-
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-860-        for i in range(n_steps):
--
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-862-            if (i + 1) % save_every == 0:
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-863-                trajectory_cpu.append(state_cpu)
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-864-
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py:865:        state_out = jax.device_put(state_cpu, self._default_device)
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-866-        trajectory_out = [
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py:867:            jax.device_put(s, self._default_device) for s in trajectory_cpu
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-868-        ]
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-869-        return state_out, trajectory_out
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py-870-
--
tests/parallel/test_tiled_cgrid_gradient.py-63-
tests/parallel/test_tiled_cgrid_gradient.py-64-    co = P("face", "tile_i", "tile_j")
tests/parallel/test_tiled_cgrid_gradient.py-65-    fo = P("face", None, None)
tests/parallel/test_tiled_cgrid_gradient.py:66:    eta_sh = jax.device_put(eta, NamedSharding(mesh, co))   # tile-sharded
tests/parallel/test_tiled_cgrid_gradient.py:67:    rdxc_sh = jax.device_put(cdg.rdxc, NamedSharding(mesh, fo))
tests/parallel/test_tiled_cgrid_gradient.py:68:    rdyc_sh = jax.device_put(cdg.rdyc, NamedSharding(mesh, fo))
tests/parallel/test_tiled_cgrid_gradient.py-69-    offs_j = jnp.asarray(offs)
tests/parallel/test_tiled_cgrid_gradient.py-70-
tests/parallel/test_tiled_cgrid_gradient.py-71-    @partial(shard_map, mesh=mesh,
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-228-    else:
scripts/bench/bench_atm_latlon_spmd_scaling.py-229-        # #1100: band-local IC construction. The nd>1 lanes never materialise
scripts/bench/bench_atm_latlon_spmd_scaling.py-230-        # the global (n_lat, n_lon, nlev) state per process — each leaf is
scripts/bench/bench_atm_latlon_spmd_scaling.py:231:        # created via make_array_from_callback for the rows this process's
scripts/bench/bench_atm_latlon_spmd_scaling.py:232:        # devices own (no global build, no device_put replication, no
scripts/bench/bench_atm_latlon_spmd_scaling.py-233-        # assert_equal all-gather). This is what lets full-node-packed CPU
scripts/bench/bench_atm_latlon_spmd_scaling.py-234-        # rungs (128 procs/node) survive at large n_lat.
scripts/bench/bench_atm_latlon_spmd_scaling.py-235-        model = _build_model(n_lat, args.n_lon, args.nlev)
--
tests/parallel/test_tiled_cgrid_divergence.py-99-    # phase-1a policy): each device holds its whole face slab.
tests/parallel/test_tiled_cgrid_divergence.py-100-    face_only = NamedSharding(mesh, P("face", None, None))
tests/parallel/test_tiled_cgrid_divergence.py-101-    centered = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
tests/parallel/test_tiled_cgrid_divergence.py:102:    u_sh = jax.device_put(u_c, face_only)
tests/parallel/test_tiled_cgrid_divergence.py:103:    v_sh = jax.device_put(v_c, face_only)
tests/parallel/test_tiled_cgrid_divergence.py:104:    dyx_sh = jax.device_put(cdg.dy_edge_x, face_only)
tests/parallel/test_tiled_cgrid_divergence.py:105:    dxy_sh = jax.device_put(cdg.dx_edge_y, face_only)
tests/parallel/test_tiled_cgrid_divergence.py:106:    area_sh = jax.device_put(cdg.base.area, centered)
tests/parallel/test_tiled_cgrid_divergence.py-107-
tests/parallel/test_tiled_cgrid_divergence.py-108-    fo = P("face", None, None)
tests/parallel/test_tiled_cgrid_divergence.py-109-    co = P("face", "tile_i", "tile_j")
--
tests/parallel/test_latlon_ocean_spmd_step.py-1-"""SPMD equivalence gate for the lat-lon C-grid ocean step (multi-node OMIP).
tests/parallel/test_latlon_ocean_spmd_step.py-2-
tests/parallel/test_latlon_ocean_spmd_step.py:3:This is the CORRECTNESS GATE for ``make_sharded_ocean_step`` (lat-band SPMD over
tests/parallel/test_latlon_ocean_spmd_step.py-4-the ``"lat"`` axis): N steps under ``shard_map`` on 4 (CPU) devices must match
tests/parallel/test_latlon_ocean_spmd_step.py-5-the single-device step to tolerance. It is the eORCA025 ¼° enabler (the ¼° grid
tests/parallel/test_latlon_ocean_spmd_step.py-6-OOMs on one 32 GiB GPU; lat-band sharding fits it at N>=5).
--
--
tests/parallel/test_latlon_ocean_spmd_step.py-82-def _have_sharded_step():
tests/parallel/test_latlon_ocean_spmd_step.py-83-    try:
tests/parallel/test_latlon_ocean_spmd_step.py-84-        from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: F401
tests/parallel/test_latlon_ocean_spmd_step.py:85:            make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py-86-        )
tests/parallel/test_latlon_ocean_spmd_step.py-87-        from legoesm.parallel.mesh import create_latlon_mesh  # noqa: F401
tests/parallel/test_latlon_ocean_spmd_step.py-88-        return True
--
tests/parallel/test_latlon_ocean_spmd_step.py-97-def test_latlon_ocean_spmd_matches_single_device():
tests/parallel/test_latlon_ocean_spmd_step.py-98-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-99-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py:100:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py:101:        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-102-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-103-    )
--
tests/parallel/test_latlon_ocean_spmd_step.py-127-    # fail on the n_lat+1 v rows).  The result is gathered (and the dropped pole
tests/parallel/test_latlon_ocean_spmd_step.py-128-    # row reappended) for the bit-comparison vs the single-device reference.
tests/parallel/test_latlon_ocean_spmd_step.py-129-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:130:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:131:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-132-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-133-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_step.py-134-    ss = gather_state_latlon(ss, dev.mesh)
--
tests/parallel/test_latlon_ocean_spmd_step.py-221-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-222-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py-223-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py:224:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py:225:        shard_forcing_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py:226:        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-227-    )
--
tests/parallel/test_latlon_ocean_spmd_step.py-244-
tests/parallel/test_latlon_ocean_spmd_step.py-245-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-246-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:247:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:248:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:249:    fws = shard_forcing_latlon(fw, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:250:    sfs = shard_forcing_latlon(sf, dev.mesh)
--
--
tests/parallel/test_latlon_ocean_spmd_step.py-280-    every in-scan forcing op."""
tests/parallel/test_latlon_ocean_spmd_step.py-281-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-282-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py:283:        shard_forcing_stack_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-284-    )
tests/parallel/test_latlon_ocean_spmd_step.py-285-
--
tests/parallel/test_latlon_ocean_spmd_step.py-351-    _prev = (get_halo_backend(), get_mpi_topology(), get_spmd_mesh())
tests/parallel/test_latlon_ocean_spmd_step.py-352-    activate_latlon_spmd_halo(mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-353-    try:
tests/parallel/test_latlon_ocean_spmd_step.py:354:        sharded = body(jax.device_put(F, sh), jax.device_put(area, sh),
tests/parallel/test_latlon_ocean_spmd_step.py:355:                       jax.device_put(mask, sh))
tests/parallel/test_latlon_ocean_spmd_step.py-356-        # Non-vacuity: at least one band's local mean differs from the
tests/parallel/test_latlon_ocean_spmd_step.py-357-        # global mean, so a missing psum WOULD change the answer.
tests/parallel/test_latlon_ocean_spmd_step.py-358-        w = np.asarray(area) * np.asarray(mask)
--
tests/parallel/test_latlon_ocean_spmd_step.py-384-    with an opaque shard_map divisibility error."""
tests/parallel/test_latlon_ocean_spmd_step.py-385-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-386-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py:387:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py:388:        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-389-    )
tests/parallel/test_latlon_ocean_spmd_step.py-390-    from legoesm.ocean.state import OceanSurfaceForcing
--
tests/parallel/test_latlon_ocean_spmd_step.py-396-    state0 = _perturbed_state(grid, z_coord)
tests/parallel/test_latlon_ocean_spmd_step.py-397-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-398-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:399:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:400:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-401-    bad = OceanSurfaceForcing(
tests/parallel/test_latlon_ocean_spmd_step.py-402-        tau_y=jnp.zeros((grid.n_lat + 1, grid.n_lon)))
tests/parallel/test_latlon_ocean_spmd_step.py-403-    with pytest.raises(ValueError, match="leading dim"):
--
tests/parallel/test_latlon_ocean_spmd_step.py-409-@pytest.mark.skipif(not _have_sharded_step(),
tests/parallel/test_latlon_ocean_spmd_step.py-410-                    reason="sharded_ocean_step module not present")
tests/parallel/test_latlon_ocean_spmd_step.py-411-def test_sharded_ocean_step_global_matches_explicit_scatter_gather():
tests/parallel/test_latlon_ocean_spmd_step.py:412:    """``make_sharded_ocean_step_global`` (global-in/global-out, the minimal
tests/parallel/test_latlon_ocean_spmd_step.py:413:    driver entry) must equal the explicit ``shard_state_latlon`` -> inner step ->
tests/parallel/test_latlon_ocean_spmd_step.py-414-    ``gather_state_latlon`` path BIT-FOR-BIT (it is literally that composition),
tests/parallel/test_latlon_ocean_spmd_step.py-415-    AND match the single-device reference to the same re-association floor.
--
--
tests/parallel/test_latlon_ocean_spmd_step.py-419-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-420-    from legoesm.grids.latlon import ensure_geometry
tests/parallel/test_latlon_ocean_spmd_step.py-421-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py:422:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py:423:        make_sharded_ocean_step_global,
tests/parallel/test_latlon_ocean_spmd_step.py:424:        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-425-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-426-    )
--
tests/parallel/test_latlon_ocean_spmd_step.py-451-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py-452-
tests/parallel/test_latlon_ocean_spmd_step.py-453-    # explicit scatter -> inner sharded step -> gather
tests/parallel/test_latlon_ocean_spmd_step.py:454:    inner = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:455:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-456-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-457-        ss = inner(ss, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_step.py-458-    ss = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-459-
tests/parallel/test_latlon_ocean_spmd_step.py-460-    # global-in/global-out wrapper (scatter + gather PER STEP)
tests/parallel/test_latlon_ocean_spmd_step.py:461:    glob = make_sharded_ocean_step_global(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-462-    sg = state0
tests/parallel/test_latlon_ocean_spmd_step.py-463-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-464-        sg = glob(sg, dt, surface_forcing=sf)
--
tests/parallel/test_tiled_fv3_sw_full.py-68-    fo = P("face", None, None)
tests/parallel/test_tiled_fv3_sw_full.py-69-    sh = NamedSharding(mesh, fo)
tests/parallel/test_tiled_fv3_sw_full.py-70-    dh_t, du_t, dv_t = stage(
tests/parallel/test_tiled_fv3_sw_full.py:71:        jax.device_put(h, sh), jax.device_put(u_d, sh),
tests/parallel/test_tiled_fv3_sw_full.py:72:        jax.device_put(v_d, sh), jax.device_put(h_s, sh))
tests/parallel/test_tiled_fv3_sw_full.py-73-    dh_t, du_t, dv_t = np.asarray(dh_t), np.asarray(du_t), np.asarray(dv_t)
tests/parallel/test_tiled_fv3_sw_full.py-74-
tests/parallel/test_tiled_fv3_sw_full.py-75-    blk_u_j = NL + 1
--
tests/parallel/test_tiled_fv3_sw_full.py-120-    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
tests/parallel/test_tiled_fv3_sw_full.py-121-    stage = make_tiled_fv3_sw_tendencies_stage_2d(mesh, cdg, N, 2)
tests/parallel/test_tiled_fv3_sw_full.py-122-    sh = NamedSharding(mesh, P("face", None, None))
tests/parallel/test_tiled_fv3_sw_full.py:123:    h_sh = jax.device_put(h, sh)
tests/parallel/test_tiled_fv3_sw_full.py:124:    u_sh = jax.device_put(u_d, sh)
tests/parallel/test_tiled_fv3_sw_full.py:125:    v_sh = jax.device_put(v_d, sh)
tests/parallel/test_tiled_fv3_sw_full.py:126:    hs_sh = jax.device_put(h_s, sh)
tests/parallel/test_tiled_fv3_sw_full.py-127-
tests/parallel/test_tiled_fv3_sw_full.py-128-    def loss(hh, uu, vv, hs):
tests/parallel/test_tiled_fv3_sw_full.py-129-        dh, du, dv = stage(hh, uu, vv, hs)
--
tests/parallel/test_tiled_fv3_hydrostatic_step.py-135-    fw = NamedSharding(mesh, P("face", None, None, None))
tests/parallel/test_tiled_fv3_hydrostatic_step.py-136-    fo = NamedSharding(mesh, P("face", None, None))
tests/parallel/test_tiled_fv3_hydrostatic_step.py-137-    ut, vt, Tt, pst = step(
tests/parallel/test_tiled_fv3_hydrostatic_step.py:138:        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
tests/parallel/test_tiled_fv3_hydrostatic_step.py:139:        jax.device_put(T, fw), jax.device_put(p_s, fo),
tests/parallel/test_tiled_fv3_hydrostatic_step.py:140:        jax.device_put(phis, fo))
tests/parallel/test_tiled_fv3_hydrostatic_step.py-141-
tests/parallel/test_tiled_fv3_hydrostatic_step.py-142-    r_u = _rel_corner(np.asarray(ut), ug, KT, nl)
tests/parallel/test_tiled_fv3_hydrostatic_step.py-143-    r_v = _rel_corner(np.asarray(vt), vg, KT, nl)
--
tests/parallel/test_atm_latlon_2d_tiling.py-185-        body, mesh=mesh,
tests/parallel/test_atm_latlon_2d_tiling.py-186-        in_specs=P("lat", "lon", None),
tests/parallel/test_atm_latlon_2d_tiling.py-187-        out_specs=P("lat", "lon", None), check_vma=False))
tests/parallel/test_atm_latlon_2d_tiling.py:188:    sharded = jax.device_put(
tests/parallel/test_atm_latlon_2d_tiling.py-189-        field, NamedSharding(mesh, P("lat", "lon", None)))
tests/parallel/test_atm_latlon_2d_tiling.py:190:    out = np.asarray(jax.device_put(fn(sharded), NamedSharding(mesh, P())))
tests/parallel/test_atm_latlon_2d_tiling.py-191-
tests/parallel/test_atm_latlon_2d_tiling.py-192-    nl, w = N_LAT // p_lat, N_LON // p_lon
tests/parallel/test_atm_latlon_2d_tiling.py-193-    hl, hw = nl + 2 * halo, w + 2 * halo
--
tests/parallel/test_atm_latlon_2d_tiling.py-213-    fn = jax.jit(shard_map(
tests/parallel/test_atm_latlon_2d_tiling.py-214-        body, mesh=mesh, in_specs=P("lat", "lon", None),
tests/parallel/test_atm_latlon_2d_tiling.py-215-        out_specs=P("lat", "lon", None), check_vma=False))
tests/parallel/test_atm_latlon_2d_tiling.py:216:    sharded = jax.device_put(
tests/parallel/test_atm_latlon_2d_tiling.py-217-        field, NamedSharding(mesh, P("lat", "lon", None)))
tests/parallel/test_atm_latlon_2d_tiling.py-218-    hlo = fn.lower(sharded).compile().as_text()
tests/parallel/test_atm_latlon_2d_tiling.py-219-    return fn, sharded, hlo
--
tests/parallel/test_atm_latlon_2d_tiling.py-237-        "even-p_lon pole fold must not emit all-gather "
tests/parallel/test_atm_latlon_2d_tiling.py-238-        "(partner ppermute lever)")
tests/parallel/test_atm_latlon_2d_tiling.py-239-    assert "collective-permute" in hlo
tests/parallel/test_atm_latlon_2d_tiling.py:240:    out = np.asarray(jax.device_put(fn(sharded), NamedSharding(mesh, P())))
tests/parallel/test_atm_latlon_2d_tiling.py-241-    serial = np.asarray(pad_halo_latlon_3d_local(field, halo))
tests/parallel/test_atm_latlon_2d_tiling.py-242-    nl, w = N_LAT // p_lat, N_LON // p_lon
tests/parallel/test_atm_latlon_2d_tiling.py-243-    hl, hw = nl + 2 * halo, w + 2 * halo
--
tests/parallel/test_atm_latlon_2d_tiling.py-263-    field = jnp.asarray(rng.standard_normal((N_LAT, n_lon, NLEV)))
tests/parallel/test_atm_latlon_2d_tiling.py-264-    fn, sharded, hlo = _lowered_pad(mesh, field, halo)
tests/parallel/test_atm_latlon_2d_tiling.py-265-    assert "all-gather" in hlo
tests/parallel/test_atm_latlon_2d_tiling.py:266:    out = np.asarray(jax.device_put(fn(sharded), NamedSharding(mesh, P())))
tests/parallel/test_atm_latlon_2d_tiling.py-267-    serial = np.asarray(pad_halo_latlon_3d_local(field, halo))
tests/parallel/test_atm_latlon_2d_tiling.py-268-    w = n_lon // p_lon
tests/parallel/test_atm_latlon_2d_tiling.py-269-    hw = w + 2 * halo
--
tests/parallel/test_atm_latlon_2d_tiling.py-374-        out = fn(sc, stacks)
tests/parallel/test_atm_latlon_2d_tiling.py-375-    finally:
tests/parallel/test_atm_latlon_2d_tiling.py-376-        deactivate_latlon_spmd_halo()
tests/parallel/test_atm_latlon_2d_tiling.py:377:    g = lambda x: np.asarray(jax.device_put(x, NamedSharding(mesh, P())))
tests/parallel/test_atm_latlon_2d_tiling.py-378-    dul_b, dvl_b, dT_b, dps_b = (g(x) for x in out)
tests/parallel/test_atm_latlon_2d_tiling.py-379-    return serial, (dul_b, dvl_b, dT_b, dps_b), (p_lat, p_lon)
tests/parallel/test_atm_latlon_2d_tiling.py-380-
--
tests/parallel/test_latlon_spmd_fused_halo.py-321-    )
tests/parallel/test_latlon_spmd_fused_halo.py-322-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_spmd_fused_halo.py-323-        gather_state_latlon,
tests/parallel/test_latlon_spmd_fused_halo.py:324:        make_sharded_ocean_step,
tests/parallel/test_latlon_spmd_fused_halo.py:325:        shard_state_latlon,
tests/parallel/test_latlon_spmd_fused_halo.py-326-    )
tests/parallel/test_latlon_spmd_fused_halo.py-327-    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
--
tests/parallel/test_latlon_spmd_fused_halo.py-346-    hlo_counts = {}
tests/parallel/test_latlon_spmd_fused_halo.py-347-    for flag in ("0", "1"):
tests/parallel/test_latlon_spmd_fused_halo.py-348-        _env_flag(monkeypatch, flag)
tests/parallel/test_latlon_spmd_fused_halo.py:349:        step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py:350:        ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-351-        hlo_counts[flag] = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-352-            jax.jit(step).lower(ss, 600.0).compile().as_text())
--
tests/parallel/test_latlon_spmd_fused_halo.py-369-    # (an outer-jit artifact, not the production call pattern — the
tests/parallel/test_latlon_spmd_fused_halo.py-370-    # driver calls step() directly each step).
tests/parallel/test_latlon_spmd_fused_halo.py-371-    _env_flag(monkeypatch, "0")
tests/parallel/test_latlon_spmd_fused_halo.py:372:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py:373:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-374-    n_off = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-375-        jax.jit(lambda s, d: step(s, d)).lower(ss, 600.0)
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-1-"""SPMD equivalence + collective-count gate for the WIDE-HALO barotropic.
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-2-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-3-Twin of ``test_latlon_ocean_spmd_step.py`` with
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:4:``barotropic_wide_halo=True``: N steps of ``make_sharded_ocean_step`` across
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-5-4 (CPU) devices must match the single-device wide-halo step at the sharded
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-6-split-explicit re-association floor.  A staggered off-by-one in the wide
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-7-v-exchange, a reach under-budget at a band cut, or a pole-flag error in the
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-73-def _run_pair(cfg, n_lat=48, n_lon=96, nlev=10, dt=600.0, n_steps=3):
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-74-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-75-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:76:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:77:        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-78-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-79-    )
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-89-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-90-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-91-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:92:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:93:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-94-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-95-        ss = step(ss, dt)
--
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-112-def _sharded_step_hlo(cfg, n_lat=48, n_lon=96, nlev=10):
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-113-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-114-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:115:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:116:        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-117-    )
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-118-
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-122-    state0 = _perturbed_state(grid, z_coord)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-123-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-124-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:125:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:126:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-127-    lowered = jax.jit(step).lower(ss, 600.0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-128-    return lowered.compile().as_text()
--
tests/parallel/test_atm_latlon_spmd_step.py-150-    template = band_grids[0]
tests/parallel/test_atm_latlon_spmd_step.py-151-    afn = atm_grid_array_field_names(template)
tests/parallel/test_atm_latlon_spmd_step.py-152-    rep = NamedSharding(mesh, P())
tests/parallel/test_atm_latlon_spmd_step.py:153:    stacks = {n: jax.device_put(jnp.stack([jnp.asarray(getattr(g, n))
tests/parallel/test_atm_latlon_spmd_step.py-154-              for g in band_grids], 0), rep) for n in afn}
tests/parallel/test_atm_latlon_spmd_step.py-155-    perm_north, _ = latlon_band_perms(N_DEV)
tests/parallel/test_atm_latlon_spmd_step.py-156-
--
tests/parallel/test_atm_latlon_spmd_step.py-175-        out = fn(sc, stacks)
tests/parallel/test_atm_latlon_spmd_step.py-176-    finally:
tests/parallel/test_atm_latlon_spmd_step.py-177-        deactivate_latlon_spmd_halo()
tests/parallel/test_atm_latlon_spmd_step.py:178:    g = lambda x: np.asarray(jax.device_put(x, NamedSharding(mesh, P())))
tests/parallel/test_atm_latlon_spmd_step.py-179-    du_b, dvl_b, dT_b, dps_b = (g(x) for x in out)
tests/parallel/test_atm_latlon_spmd_step.py-180-    # band returns v_lower (N_LAT rows); re-cap the north pole row (=0) so the
tests/parallel/test_atm_latlon_spmd_step.py-181-    # shape matches the serial (N_LAT+1)-row v-face array for comparison.
--
tests/parallel/test_atm_latlon_spmd_step.py-272-                f"denominator, or v-face reconstruction)."))
tests/parallel/test_atm_latlon_spmd_step.py-273-
tests/parallel/test_atm_latlon_spmd_step.py-274-
tests/parallel/test_atm_latlon_spmd_step.py:275:def test_replicate_leaf_multiprocess_branch_matches_device_put():
tests/parallel/test_atm_latlon_spmd_step.py-276-    """The multi-controller gather branch (jit-compiled identity with
tests/parallel/test_atm_latlon_spmd_step.py-277-    replicated out_shardings) must produce the SAME replicated array as the
tests/parallel/test_atm_latlon_spmd_step.py:278:    single-process device_put branch — on values, sharding, and for both a
tests/parallel/test_atm_latlon_spmd_step.py-279-    lat-sharded and an already-replicated input. This exercises the
tests/parallel/test_atm_latlon_spmd_step.py-280-    ``multiprocess=True`` code path for real on a single process (where both
tests/parallel/test_atm_latlon_spmd_step.py-281-    mechanisms are legal), so the route-B gather cannot silently diverge.
--
tests/parallel/test_atm_latlon_spmd_step.py-286-    rep = NamedSharding(mesh, P())
tests/parallel/test_atm_latlon_spmd_step.py-287-    rng = np.random.default_rng(7)
tests/parallel/test_atm_latlon_spmd_step.py-288-    full = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))
tests/parallel/test_atm_latlon_spmd_step.py:289:    sharded = jax.device_put(full, NamedSharding(mesh, P("lat", None, None)))
tests/parallel/test_atm_latlon_spmd_step.py-290-
tests/parallel/test_atm_latlon_spmd_step.py:291:    for arr in (sharded, jax.device_put(full, rep)):
tests/parallel/test_atm_latlon_spmd_step.py-292-        a = replicate_leaf(arr, rep, multiprocess=False)
tests/parallel/test_atm_latlon_spmd_step.py-293-        b = replicate_leaf(arr, rep, multiprocess=True)
tests/parallel/test_atm_latlon_spmd_step.py-294-        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
--
tests/parallel/test_tiled_pad_body.py-59-    serial = np.asarray(pad_halo_local(ref, interp_offsets=offs))
tests/parallel/test_tiled_pad_body.py-60-
tests/parallel/test_tiled_pad_body.py-61-    sharding = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
tests/parallel/test_tiled_pad_body.py:62:    ref_sh = jax.device_put(ref, sharding)
tests/parallel/test_tiled_pad_body.py-63-    fn = _make_exchange_ppermute_tiled(mesh, ndim=3, halo=1,
tests/parallel/test_tiled_pad_body.py-64-                                       with_offsets=True)
tests/parallel/test_tiled_pad_body.py-65-    out = fn(ref_sh, jnp.asarray(offs))
--
tests/parallel/test_tiled_pad_body.py-80-    ref = jnp.asarray(rng.standard_normal((6, N, N)))
tests/parallel/test_tiled_pad_body.py-81-    offs = jnp.asarray(compute_halo_interp_offsets(N))
tests/parallel/test_tiled_pad_body.py-82-    sharding = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
tests/parallel/test_tiled_pad_body.py:83:    ref_sh = jax.device_put(ref, sharding)
tests/parallel/test_tiled_pad_body.py-84-
tests/parallel/test_tiled_pad_body.py-85-    wrapped = _make_exchange_ppermute_tiled(
tests/parallel/test_tiled_pad_body.py-86-        mesh, ndim=3, halo=1, with_offsets=True)
--
tests/parallel/test_tiled_pad_body.py-126-    offs = jnp.asarray(compute_halo_interp_offsets(N))
tests/parallel/test_tiled_pad_body.py-127-    sh4 = NamedSharding(mesh, P("face", "tile_i", "tile_j", None))
tests/parallel/test_tiled_pad_body.py-128-    sh3 = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
tests/parallel/test_tiled_pad_body.py:129:    ref4_sh = jax.device_put(ref4, sh4)
tests/parallel/test_tiled_pad_body.py-130-
tests/parallel/test_tiled_pad_body.py-131-    body4 = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)
tests/parallel/test_tiled_pad_body.py-132-    body3 = make_tiled_pad_body(mesh, ndim=3, halo=1, with_offsets=True)
--
tests/parallel/test_tiled_pad_body.py-147-    out4 = _stage4(ref4_sh, offs)              # (6, KT*out_blk, KT*out_blk, C)
tests/parallel/test_tiled_pad_body.py-148-    assert out4.shape == (6, KT * out_blk, KT * out_blk, C)
tests/parallel/test_tiled_pad_body.py-149-    for k in range(C):
tests/parallel/test_tiled_pad_body.py:150:        ref3_sh = jax.device_put(ref4[..., k], sh3)
tests/parallel/test_tiled_pad_body.py-151-        out3 = _stage3(ref3_sh, offs)
tests/parallel/test_tiled_pad_body.py-152-        np.testing.assert_array_equal(
tests/parallel/test_tiled_pad_body.py-153-            np.asarray(out4)[..., k], np.asarray(out3),
--
tests/parallel/test_tiled_pad_body.py-216-    sh4 = NamedSharding(mesh, fw4)
tests/parallel/test_tiled_pad_body.py-217-    sh3 = NamedSharding(mesh, fw3)
tests/parallel/test_tiled_pad_body.py-218-    sho = NamedSharding(mesh, fo)
tests/parallel/test_tiled_pad_body.py:219:    ca_d, sa_d = jax.device_put(ca, sho), jax.device_put(sa, sho)
tests/parallel/test_tiled_pad_body.py:220:    cap_d, sap_d = jax.device_put(cap, sho), jax.device_put(sap, sho)
tests/parallel/test_tiled_pad_body.py:221:    out4u, out4v = _stage4(jax.device_put(u4, sh4), jax.device_put(v4, sh4),
tests/parallel/test_tiled_pad_body.py-222-                           ca_d, sa_d, cap_d, sap_d, offs)
tests/parallel/test_tiled_pad_body.py-223-    out_blk = NL + 2
tests/parallel/test_tiled_pad_body.py-224-    assert np.asarray(out4u).shape == (6, KT * out_blk, KT * out_blk, C)
tests/parallel/test_tiled_pad_body.py-225-    for k in range(C):
tests/parallel/test_tiled_pad_body.py:226:        o3u, o3v = _stage3(jax.device_put(u4[..., k], sh3),
tests/parallel/test_tiled_pad_body.py:227:                           jax.device_put(v4[..., k], sh3),
tests/parallel/test_tiled_pad_body.py-228-                           ca_d, sa_d, cap_d, sap_d, offs)
tests/parallel/test_tiled_pad_body.py-229-        np.testing.assert_array_equal(
tests/parallel/test_tiled_pad_body.py-230-            np.asarray(out4u)[..., k], np.asarray(o3u),
--
tests/parallel/test_persistent_sharded_ocean_loop.py-2-(scaling-M2 increment 1, ``run_omip_core2 --spmd-persistent-state``).
tests/parallel/test_persistent_sharded_ocean_loop.py-3-
tests/parallel/test_persistent_sharded_ocean_loop.py-4-The production multi-GPU OMIP host loop used to call
tests/parallel/test_persistent_sharded_ocean_loop.py:5:``make_sharded_ocean_step_global`` EVERY step — a full-state scatter
tests/parallel/test_persistent_sharded_ocean_loop.py:6:(``shard_state_latlon``) + gather (``gather_state_latlon``) per step, i.e.
tests/parallel/test_persistent_sharded_ocean_loop.py-7-2 full-state transfers/step.  The persistent lane instead keeps the state
tests/parallel/test_persistent_sharded_ocean_loop.py:8:lat-band SHARDED across steps via ``make_sharded_ocean_step`` and gathers
tests/parallel/test_persistent_sharded_ocean_loop.py-9-ONLY at real output boundaries.  This gate asserts, over a 10-step loop with
tests/parallel/test_persistent_sharded_ocean_loop.py-10-production-shaped surface + freshwater forcing (``normalize_freshwater=True``
tests/parallel/test_persistent_sharded_ocean_loop.py-11-so the in-step psum reduction is load-bearing):
--
tests/parallel/test_persistent_sharded_ocean_loop.py-74-def _have_sharded_step():
tests/parallel/test_persistent_sharded_ocean_loop.py-75-    try:
tests/parallel/test_persistent_sharded_ocean_loop.py-76-        from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: F401
tests/parallel/test_persistent_sharded_ocean_loop.py:77:            make_sharded_ocean_step,
tests/parallel/test_persistent_sharded_ocean_loop.py-78-        )
tests/parallel/test_persistent_sharded_ocean_loop.py-79-        from legoesm.parallel.mesh import create_latlon_mesh  # noqa: F401
tests/parallel/test_persistent_sharded_ocean_loop.py-80-        return True
--
tests/parallel/test_persistent_sharded_ocean_loop.py-170-                u_sfc, v_sfc)
tests/parallel/test_persistent_sharded_ocean_loop.py-171-
tests/parallel/test_persistent_sharded_ocean_loop.py-172-    # ---------------- OLD lane: per-step global-in/global-out wrapper --------
tests/parallel/test_persistent_sharded_ocean_loop.py:173:    glob = sos.make_sharded_ocean_step_global(model, mesh)
tests/parallel/test_persistent_sharded_ocean_loop.py-174-    sg = state0
tests/parallel/test_persistent_sharded_ocean_loop.py-175-    cur_probe = {}
tests/parallel/test_persistent_sharded_ocean_loop.py-176-    for k in range(1, n_steps + 1):
--
tests/parallel/test_persistent_sharded_ocean_loop.py-198-
tests/parallel/test_persistent_sharded_ocean_loop.py-199-    # ---------------- NEW lane: persistent sharded loop ----------------------
tests/parallel/test_persistent_sharded_ocean_loop.py-200-    calls["shard"] = calls["gather"] = 0
tests/parallel/test_persistent_sharded_ocean_loop.py:201:    inner = sos.make_sharded_ocean_step(model, mesh)
tests/parallel/test_persistent_sharded_ocean_loop.py:202:    ss = sos.shard_state_latlon(state0, mesh)          # ONE initial shard
tests/parallel/test_persistent_sharded_ocean_loop.py-203-    for k in range(1, n_steps + 1):
tests/parallel/test_persistent_sharded_ocean_loop.py-204-        # surface-current consumer on the SHARDED state: v is the n_lat-row
--
scripts/bench/roofline_probe.py-388-        y = jnp.zeros((n,), dtype=dtype)
scripts/bench/roofline_probe.py-389-        # Place on device 0 explicitly and block so allocation/H2D is done
scripts/bench/roofline_probe.py-390-        # before timing.
scripts/bench/roofline_probe.py:391:        x = jax.device_put(x, dev0)
scripts/bench/roofline_probe.py:392:        y = jax.device_put(y, dev0)
scripts/bench/roofline_probe.py-393-        jax.block_until_ready((x, y))
scripts/bench/roofline_probe.py-394-
scripts/bench/roofline_probe.py-395-        # Latency mode (block each call): guarantees every triad actually
--
scripts/bench/roofline_probe.py-517-        # Global array shape (2, n): axis 0 sharded over the 2-device face
scripts/bench/roofline_probe.py-518-        # mesh so each device owns a (1, n) shard → a length-n payload.
scripts/bench/roofline_probe.py-519-        x_global = jnp.arange(2 * n, dtype=dtype).reshape(2, n)
scripts/bench/roofline_probe.py:520:        x_global = jax.device_put(
scripts/bench/roofline_probe.py-521-            x_global, jax.sharding.NamedSharding(mesh, P("face")))
scripts/bench/roofline_probe.py-522-        jax.block_until_ready(x_global)
scripts/bench/roofline_probe.py-523-
--
scripts/bench/roofline_probe.py-703-
scripts/bench/roofline_probe.py-704-    for b in batch_sizes:
scripts/bench/roofline_probe.py-705-        x_global = jnp.ones((2, b), dtype=dtype)
scripts/bench/roofline_probe.py:706:        x_global = jax.device_put(
scripts/bench/roofline_probe.py-707-            x_global, jax.sharding.NamedSharding(mesh, P("r")))
scripts/bench/roofline_probe.py-708-        jax.block_until_ready(x_global)
scripts/bench/roofline_probe.py-709-
--
scripts/bench/roofline_probe.py-809-    NO per-step host round-trip — the relevant floor for the model's
scripts/bench/roofline_probe.py-810-    scanned time loop."""
scripts/bench/roofline_probe.py-811-    x = jnp.ones((8,), dtype=dtype)
scripts/bench/roofline_probe.py:812:    x = jax.device_put(x)
scripts/bench/roofline_probe.py-813-    jax.block_until_ready(x)
scripts/bench/roofline_probe.py-814-
scripts/bench/roofline_probe.py-815-    @jax.jit
--
tests/parallel/test_latlon_spmd_northfold.py-88-    ref = np.asarray(pad_fn(field, grid_serial))           # serial north fold
tests/parallel/test_latlon_spmd_northfold.py-89-
tests/parallel/test_latlon_spmd_northfold.py-90-    isp = P("lat", *((None,) * (field.ndim - 1)))
tests/parallel/test_latlon_spmd_northfold.py:91:    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
tests/parallel/test_latlon_spmd_northfold.py-92-    activate_latlon_spmd_halo(mesh)
tests/parallel/test_latlon_spmd_northfold.py-93-    try:
tests/parallel/test_latlon_spmd_northfold.py-94-        @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp,
--
tests/parallel/test_tiled_mass_divergence.py-78-
tests/parallel/test_tiled_mass_divergence.py-79-    fo = P("face", None, None)
tests/parallel/test_tiled_mass_divergence.py-80-    sh_fo = NamedSharding(mesh, fo)
tests/parallel/test_tiled_mass_divergence.py:81:    h_sh = jax.device_put(h, sh_fo)
tests/parallel/test_tiled_mass_divergence.py:82:    u_sh = jax.device_put(u_c, sh_fo)
tests/parallel/test_tiled_mass_divergence.py:83:    v_sh = jax.device_put(v_c, sh_fo)
tests/parallel/test_tiled_mass_divergence.py-84-
tests/parallel/test_tiled_mass_divergence.py-85-    dh_t = np.asarray(stage(h_sh, u_sh, v_sh))   # (6, n, n), cc exact partition
tests/parallel/test_tiled_mass_divergence.py-86-
--
tests/parallel/test_sharded_step_sharding_tripwire.py-196-        arr = jnp.ones((6, 4, 4, 3))
tests/parallel/test_sharded_step_sharding_tripwire.py-197-
tests/parallel/test_sharded_step_sharding_tripwire.py-198-        good = {
tests/parallel/test_sharded_step_sharding_tripwire.py:199:            "T": jax.device_put(arr, NamedSharding(mesh, P("face"))),
tests/parallel/test_sharded_step_sharding_tripwire.py:200:            "scalar": jax.device_put(
tests/parallel/test_sharded_step_sharding_tripwire.py-201-                jnp.asarray(1.0), NamedSharding(mesh, P()),
tests/parallel/test_sharded_step_sharding_tripwire.py-202-            ),
tests/parallel/test_sharded_step_sharding_tripwire.py-203-        }
--
tests/parallel/test_sharded_step_sharding_tripwire.py-205-        _assert_expected_sharding(good, dev_config, where="synthetic-good")
tests/parallel/test_sharded_step_sharding_tripwire.py-206-
tests/parallel/test_sharded_step_sharding_tripwire.py-207-        bad = dict(good)
tests/parallel/test_sharded_step_sharding_tripwire.py:208:        bad["T"] = jax.device_put(arr, NamedSharding(mesh, P()))  # replicated
tests/parallel/test_sharded_step_sharding_tripwire.py-209-        with pytest.raises(RuntimeError, match="does not match expected"):
tests/parallel/test_sharded_step_sharding_tripwire.py-210-            _assert_expected_sharding(bad, dev_config, where="synthetic-bad")
tests/parallel/test_sharded_step_sharding_tripwire.py-211-
--
tests/parallel/test_cubesphere_exchange.py-29-    P = jax.sharding.PartitionSpec
tests/parallel/test_cubesphere_exchange.py-30-    spec = P("face", *((None,) * (data.ndim - 1)))
tests/parallel/test_cubesphere_exchange.py-31-    sharding = jax.sharding.NamedSharding(mesh, spec)
tests/parallel/test_cubesphere_exchange.py:32:    return jax.device_put(data, sharding)
tests/parallel/test_cubesphere_exchange.py-33-
tests/parallel/test_cubesphere_exchange.py-34-
tests/parallel/test_cubesphere_exchange.py-35-@pytest.fixture
--
tests/parallel/test_ppermute_multiface.py-430-            return _stencil_step(carry, offsets), None
tests/parallel/test_ppermute_multiface.py-431-        return jax.lax.scan(body, x0, None, length=steps)[0]
tests/parallel/test_ppermute_multiface.py-432-
tests/parallel/test_ppermute_multiface.py:433:    x = jax.device_put(_ramp_4d(n, c) / 1000.0, sharding)
tests/parallel/test_ppermute_multiface.py-434-    runner = jax.jit(run, in_shardings=(sharding,), out_shardings=sharding)
tests/parallel/test_ppermute_multiface.py-435-    return runner.lower(x).compile().as_text()
tests/parallel/test_ppermute_multiface.py-436-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-323-    vmask = model._vertex_mask
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-324-    if vmask is None:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-325-        raise RuntimeError(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:326:            "make_sharded_ocean_step: the model vertex-mask cache is not "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-327-            "primed.  Call model._ensure_vertex_mask(state) (or model.step) "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-328-            "on the concrete initial state before building the sharded step.")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-329-    vmask = np.asarray(vmask)                   # (n_lat+1, n_lon+1)
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-339-    """Put a host array onto a (possibly multi-process) sharding WITHOUT
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-340-    jax's whole-array cross-process ``assert_equal``.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-341-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:342:    Under multicontroller, ``jax.device_put(numpy_array, sharding)`` calls
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-343-    ``multihost_utils.assert_equal`` on the FULL array
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:344:    (jax _src/dispatch.py::_device_put_sharding_impl) — a
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-345-    ``process_allgather`` whose output is ``[n_processes, *shape]`` on one
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-346-    device: at LL2304 L20 one 3-D field is 849 MB, so 96 processes fetch
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-347-    81.5 GiB > an 80 GB A100 (the wall that killed oc @96/@128, jobs
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-348-    26644681/26644682, AFTER the geometry-broadcast fix; @64 = 54.3 GiB
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-349-    just fit, which is why smaller ladders never saw it).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-350-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:351:    ``jax.make_array_from_callback`` supplies each process's addressable
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-352-    shards directly — no consistency collective. The layouts here are
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-353-    band-sharded or replicated stacks of DETERMINISTICALLY-built host
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-354-    values; cross-process identity of non-owned rows is not required, and
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-357-    """
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-358-    host = np.asarray(arr)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-359-    if jax.process_count() <= 1:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:360:        return jax.device_put(jnp.asarray(host), sharding)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:361:    return jax.make_array_from_callback(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-362-        host.shape, sharding, lambda idx: host[idx])
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-363-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-364-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:365:def shard_state_latlon(state, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-366-    """Lay out a ``LatLonCGridOceanState`` for the lat-band SPMD step.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-367-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-368-    Cell / u-grid array fields (leading dim ``n_lat``) shard ``P("lat")``.  The
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-372-    by the band halo.  ``None`` fields pass through.  The inverse is
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-373-    :func:`gather_state_latlon`.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-374-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:375:    This is the layout the in-``shard_map`` body of :func:`make_sharded_ocean_step`
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-376-    expects; the test uses it instead of a uniform ``tree.map(P("lat"))`` (which
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-377-    fails on ``v`` because ``n_lat+1`` is not divisible by ``N``).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-378-    """
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:379:    # v-carrier contract (see make_sharded_ocean_step's fold note): the TOP
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-380-    # v-face row (regular pole wall OR tripole seam/cap row) must be
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-381-    # wall-masked — the carrier drops it and reconstructs it as zero, which
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-382-    # would silently delete a LIVE seam row.  Host-side check on the
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-387-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-388-        if _np.asarray(vm.data)[-1].any():
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-389-            raise ValueError(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:390:                "shard_state_latlon: the state's TOP v-face row is LIVE "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-391-                "(v_mask[-1] has ocean faces) — the lat-band v-carrier "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-392-                "drops that row and reconstructs it as the pole/cap wall "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-393-                "zero, which would silently delete seam velocities. "
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-405-            return None
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-406-        # Drop the TOP v-row (the north pole wall, v==0 on the regular grid) so
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-407-        # the leading dim becomes n_lat (divisible by N).  ROUND-TRIP INVARIANT
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:408:        # (codex finding): shard_state_latlon -> gather_state_latlon re-appends a
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-409-        # ZERO top row, so the round-trip is a bitwise identity ONLY when the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-410-        # input's top v-row is already zero (a valid masked regular-grid state;
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-411-        # the pole-wall BC enforces v[n_lat]=0 every step).  An arbitrary nonzero
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-412-        # top row (e.g. a raw IC perturbation) is dropped -> reconstructed as 0:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-413-        # CORRECT for the dynamics (the pole wall zeros it at step 1) but not a
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-414-        # bit round-trip of that one row.  NOT for a tripole north fold (raises
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:415:        # in make_sharded_ocean_step) where the top row is a live fold partner.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-416-        nlat1 = field.data.shape[0]
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-417-        v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-418-        sh = NamedSharding(mesh, _lat_spec(v_lower))
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-440-    return state._replace(**updates)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-441-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-442-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:443:def shard_forcing_latlon(forcing, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-444-    """Lay out a forcing pytree (``FreshwaterForcing`` / ``OceanSurfaceForcing``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-445-    / ``SpongeForcing`` — or any nesting of them) for the lat-band SPMD step.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-446-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-463-    return jax.tree.map(_put, forcing)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-464-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-465-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:466:def shard_forcing_stack_latlon(stack, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-467-    """Lay out a STACKED per-block forcing pytree for the lat-band SPMD
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-468-    block-scan (the ``run_omip`` JRA55 lanes; see
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-469-    ``_build_jra55_block_fn`` / ``_build_jra55_block_fn_interp``).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-470-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:471:    Unlike :func:`shard_forcing_latlon` (per-step, lat at axis 0), the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-472-    block builders stack ``N`` steps / raw records along a LEADING axis,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-473-    so the lat axis sits at position 1.  A ``(n_rec, n_lat, n_lon[, ...])``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-474-    leaf therefore shards ``P(None, "lat", ...)`` (records replicated, the
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-483-    ``(n_rec, n_meta)`` table) would be silently mis-sharded — keep such
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-484-    metadata 1-D (or replicate it explicitly) before it reaches this helper.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-485-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:486:    Keeping this next to :func:`shard_state_latlon` means the driver and
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-487-    the parity tests share ONE layout definition — the block-scan forcing
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-488-    stack must be laid out consistently with the state the sharded step
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-489-    carries, and a second copy would drift.
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-512-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-513-    This is THE reconstruction of the row the carrier drops: the regular-grid
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-514-    north pole wall / tripole cap row, identically zero under the v-carrier
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:515:    contract (:func:`shard_state_latlon` REFUSES a state whose top v-face row
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-516-    is live), so the result is bit-identical to the original staggered array.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-517-    Shared by :func:`gather_state_latlon` (full-state gather) and the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-518-    persistent-lane DEVICE-SIDE staggered reads (the OMIP driver's
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-525-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-526-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-527-def gather_state_latlon(state, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:528:    """Inverse of :func:`shard_state_latlon`: gather every leaf to a single device
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-529-    and rebuild the full ``(n_lat+1, ...)`` ``v`` / ``v_mask`` by appending the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-530-    pole-wall row (zeros) the layout dropped.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-531-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-535-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-536-    Multi-controller (route-B ``jax.distributed``, mesh spanning processes):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-537-    replication routes through a jit-compiled identity instead of
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:538:    ``device_put`` (see :func:`legoesm.parallel.latlon_spmd.replicate_leaf`,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-539-    the primitive shared with the atm gather); the single-process path is
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-540-    byte-unchanged.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-541-    """
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-569-    return state._replace(**updates)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-570-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-571-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:572:def make_sharded_ocean_step(model, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-573-    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-574-    sponge=None, t_seconds=None) -> state`` running ``model.step``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-575-    lat-band-SPMD.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-576-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-577-    The forcing channels mirror ``model.step``'s keyword surface: pass
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:578:    pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-579-    ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-580-    is a replicated scalar like ``dt``).  ``None`` forcing keeps the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-581-    dynamics-only program; each distinct None<->populated combination
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-603-    replication check) because the band halo intentionally reads neighbour-rank
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-604-    data (replication-unaware).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-605-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:606:    The state must be laid out with :func:`shard_state_latlon` (``v`` /
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-607-    ``v_mask`` carried as the ``n_lat``-row ``v_lower``).  The body reconstructs
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-608-    each band's ``nl+1`` v-faces, runs the step on the band geometry, and
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-609-    converts the result back to the ``v_lower`` representation.
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-627-    # zero row is exact — true for the cap-row convention of
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-628-    # ``create_synthetic_tripole`` and the eORCA masks (``v_mask[-1] == 0``;
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-629-    # the serial step keeps ``v[-1] == 0`` identically).  That contract is
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:630:    # asserted on the CONCRETE state in :func:`shard_state_latlon` — a live
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-631-    # (unmasked) seam v-row refuses loudly there instead of silently
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-632-    # reconstructing zeros here.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-633-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-690-            g_vals = multihost_utils.process_allgather(vals)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-691-            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-692-                raise RuntimeError(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:693:                    f"make_sharded_ocean_step: band-geometry field {name!r} "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-694-                    f"DIVERGES across processes (exact_dtype={is_exact}, "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-695-                    f"gathered={g_vals.tolist()}) — a real config/grid "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-696-                    f"inconsistency, not autotune noise; refusing to "
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-705-            # devices consume ONLY its own band rows, so cross-process
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-706-            # byte-identity of non-owned rows is irrelevant, and REAL
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-707-            # divergence is already refused by the fingerprint gate above.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:708:            # make_array_from_callback hands each process exactly its
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-709-            # addressable slabs — the same #1100 pattern as the state
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-710-            # build — with no global-sized collective program at all.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-711-        host_np = np.asarray(host)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:712:        return jax.make_array_from_callback(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-713-            host_np.shape, rep, lambda idx: host_np[idx])
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-714-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-715-    if jax.process_count() > 1:
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-723-            _schema_fingerprint(list(array_field_names), n_dev))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-724-        if not bool(np.all(_g == _g[0])):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-725-            raise RuntimeError(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:726:                "make_sharded_ocean_step: the band-geometry SCHEMA differs "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-727-                "across processes (field list / x64 setting / device count "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-728-                f"— gathered {_g.tolist()}). Fix the per-process config "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-729-                "before sharding; the per-field checks below assume one "
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-931-    return sharded_step
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-932-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-933-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:934:def make_sharded_ocean_step_global(model, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-935-    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-936-    that takes a GLOBAL (single-device-layout) state + forcing and returns a
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-937-    GLOBAL state — the minimal-diff driver entry point.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-938-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:939:    Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:940:    (:func:`shard_state_latlon` + :func:`shard_forcing_latlon`), runs the lat-band
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-941-    SPMD step, then gathers the state OUT (:func:`gather_state_latlon`).  This lets
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-942-    the OMIP host loop keep operating on a normal full-domain state — the per-step
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-943-    host BCs (SSS restore, prognostic ice, geothermal, BBL, nudge) see the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-944-    gathered global state UNCHANGED — at the cost of a per-step gather/scatter
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-945-    (acceptable for the host-coupled OMIP driver; the pure-dynamics inner loop
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:946:    should use :func:`make_sharded_ocean_step` directly to stay sharded).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-947-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-948-    ``mesh is None`` ⇒ the plain single-device ``model.step`` (no scatter/gather).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-949-    """
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-952-            model.step(state, dt, freshwater=freshwater,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-953-                       surface_forcing=surface_forcing))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-954-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:955:    inner = make_sharded_ocean_step(model, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-956-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-957-    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-958-        # Scatter the global state to the band layout; the forcing is sharded
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:959:        # INSIDE ``inner`` (make_sharded_ocean_step lays it out), so pass it
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-960-        # through global.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:961:        ss = shard_state_latlon(state, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-962-        ss = inner(ss, dt, surface_forcing=surface_forcing,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-963-                   freshwater=freshwater)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-964-        return gather_state_latlon(ss, mesh)
--
tests/parallel/test_latlon_spmd_shard_leaf.py-1-"""Direct tests for the lat-band SCATTER primitive ``shard_leaf`` (issue #1100).
tests/parallel/test_latlon_spmd_shard_leaf.py-2-
tests/parallel/test_latlon_spmd_shard_leaf.py:3:``shard_leaf`` replaces the ``jax.device_put(full_global, NamedSharding)`` scatter
tests/parallel/test_latlon_spmd_shard_leaf.py-4-that all-gathered a transient second global copy per process (rc=137 OOM under
tests/parallel/test_latlon_spmd_shard_leaf.py-5-128-proc CPU packing). The multiprocess branch instead contributes each process's
tests/parallel/test_latlon_spmd_shard_leaf.py-6-addressable lat-band via ``make_array_from_process_local_data`` — no all-gather.
--
tests/parallel/test_latlon_spmd_shard_leaf.py-10-The true cross-PROCESS collective elision is covered by the
tests/parallel/test_latlon_spmd_shard_leaf.py-11-``*_multicontroller_selfspawn`` tests (which spawn real processes); here
tests/parallel/test_latlon_spmd_shard_leaf.py-12-``process_count()==1`` so the multiprocess branch is exercised for its index
tests/parallel/test_latlon_spmd_shard_leaf.py:13:math + assembly and must match ``device_put`` byte-for-byte.
tests/parallel/test_latlon_spmd_shard_leaf.py-14-"""
tests/parallel/test_latlon_spmd_shard_leaf.py-15-import numpy as np
tests/parallel/test_latlon_spmd_shard_leaf.py-16-import pytest
--
tests/parallel/test_latlon_spmd_shard_leaf.py-27-    return Mesh(np.array(jax.devices()[:4]), axis_names=("lat",))
tests/parallel/test_latlon_spmd_shard_leaf.py-28-
tests/parallel/test_latlon_spmd_shard_leaf.py-29-
tests/parallel/test_latlon_spmd_shard_leaf.py:30:def test_single_process_branch_is_device_put():
tests/parallel/test_latlon_spmd_shard_leaf.py-31-    mesh = _mesh4()
tests/parallel/test_latlon_spmd_shard_leaf.py-32-    arr = np.arange(8 * 3 * 2, dtype=np.float64).reshape(8, 3, 2)
tests/parallel/test_latlon_spmd_shard_leaf.py-33-    sh = NamedSharding(mesh, P("lat", None, None))
--
tests/parallel/test_latlon_spmd_shard_leaf.py-36-    assert np.array_equal(np.asarray(out), arr)
tests/parallel/test_latlon_spmd_shard_leaf.py-37-
tests/parallel/test_latlon_spmd_shard_leaf.py-38-
tests/parallel/test_latlon_spmd_shard_leaf.py:39:def test_multiprocess_branch_matches_device_put():
tests/parallel/test_latlon_spmd_shard_leaf.py-40-    # process_count()==1 here: the make_array_from_process_local_data path must
tests/parallel/test_latlon_spmd_shard_leaf.py:41:    # reassemble the SAME global array (and same sharding) as a plain device_put.
tests/parallel/test_latlon_spmd_shard_leaf.py-42-    mesh = _mesh4()
tests/parallel/test_latlon_spmd_shard_leaf.py-43-    arr = np.arange(8 * 3 * 2, dtype=np.float64).reshape(8, 3, 2)
tests/parallel/test_latlon_spmd_shard_leaf.py-44-    sh = NamedSharding(mesh, P("lat", None, None))
tests/parallel/test_latlon_spmd_shard_leaf.py:45:    ref = jax.device_put(arr, sh)
tests/parallel/test_latlon_spmd_shard_leaf.py-46-    out = shard_leaf(arr, sh, multiprocess=True)
tests/parallel/test_latlon_spmd_shard_leaf.py-47-    assert out.sharding == ref.sharding
tests/parallel/test_latlon_spmd_shard_leaf.py-48-    assert np.array_equal(np.asarray(out), np.asarray(ref))
--
tests/parallel/test_tiled_fv3_hydrostatic_momentum.py-169-    fw = NamedSharding(mesh, P("face", None, None, None))
tests/parallel/test_tiled_fv3_hydrostatic_momentum.py-170-    fo = NamedSharding(mesh, P("face", None, None))
tests/parallel/test_tiled_fv3_hydrostatic_momentum.py-171-    du_t, dv_t = stage(
tests/parallel/test_tiled_fv3_hydrostatic_momentum.py:172:        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
tests/parallel/test_tiled_fv3_hydrostatic_momentum.py:173:        jax.device_put(T, fw), jax.device_put(p_s, fo),
tests/parallel/test_tiled_fv3_hydrostatic_momentum.py:174:        jax.device_put(phis, fo))
tests/parallel/test_tiled_fv3_hydrostatic_momentum.py-175-
tests/parallel/test_tiled_fv3_hydrostatic_momentum.py-176-    def _re(arr):
tests/parallel/test_tiled_fv3_hydrostatic_momentum.py-177-        # gathered (6, KT*(nl+1), KT*(nl+1), nlev): keep per-tile blocks intact
--
tests/parallel/test_latlon_spmd_halo.py-58-    ref = np.asarray(serial_fn(field, halo=1))            # (N_LAT+2, N_LON+2[,lev])
tests/parallel/test_latlon_spmd_halo.py-59-
tests/parallel/test_latlon_spmd_halo.py-60-    isp = P("lat", *((None,) * (field.ndim - 1)))
tests/parallel/test_latlon_spmd_halo.py:61:    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
tests/parallel/test_latlon_spmd_halo.py-62-    out = np.asarray(pad_halo_latlon_band_spmd(mesh, halo=1, negate=negate)(field_sh))
tests/parallel/test_latlon_spmd_halo.py-63-
tests/parallel/test_latlon_spmd_halo.py-64-    blk = NL + 2                                          # per-band padded lat
--
tests/parallel/test_latlon_spmd_halo.py-94-    ref = np.asarray(pad_halo_latlon(field, halo=1))      # local backend
tests/parallel/test_latlon_spmd_halo.py-95-
tests/parallel/test_latlon_spmd_halo.py-96-    isp = P("lat", None)
tests/parallel/test_latlon_spmd_halo.py:97:    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
tests/parallel/test_latlon_spmd_halo.py-98-    activate_latlon_spmd_halo(mesh)
tests/parallel/test_latlon_spmd_halo.py-99-    try:
tests/parallel/test_latlon_spmd_halo.py-100-        @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp,
--
tests/parallel/test_latlon_spmd_halo.py-144-        field, halo=1, south_value=south_value, north_value=north_value))
tests/parallel/test_latlon_spmd_halo.py-145-
tests/parallel/test_latlon_spmd_halo.py-146-    isp = P("lat", *((None,) * (field.ndim - 1)))
tests/parallel/test_latlon_spmd_halo.py:147:    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
tests/parallel/test_latlon_spmd_halo.py-148-    body = make_latlon_band_wall_pad_body(
tests/parallel/test_latlon_spmd_halo.py-149-        mesh, halo=1, south_value=south_value, north_value=north_value)
tests/parallel/test_latlon_spmd_halo.py-150-    from functools import partial
--
tests/parallel/test_latlon_spmd_halo.py-193-    ref = np.asarray(pad_with_pole_bc_lat(field, halo=1))   # local backend
tests/parallel/test_latlon_spmd_halo.py-194-
tests/parallel/test_latlon_spmd_halo.py-195-    isp = P("lat", None)
tests/parallel/test_latlon_spmd_halo.py:196:    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
tests/parallel/test_latlon_spmd_halo.py-197-    activate_latlon_spmd_halo(mesh)
tests/parallel/test_latlon_spmd_halo.py-198-    try:
tests/parallel/test_latlon_spmd_halo.py-199-        @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp,
--
tests/parallel/test_latlon_spmd_halo.py-240-    ref[-1] = 0.0
tests/parallel/test_latlon_spmd_halo.py-241-
tests/parallel/test_latlon_spmd_halo.py-242-    isp = P("lat", None)
tests/parallel/test_latlon_spmd_halo.py:243:    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
tests/parallel/test_latlon_spmd_halo.py-244-    activate_latlon_spmd_halo(mesh)
tests/parallel/test_latlon_spmd_halo.py-245-    try:
tests/parallel/test_latlon_spmd_halo.py-246-        @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp,
--
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-2-
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-3-The ocean twin of ``test_atm_latlon_spmd_multicontroller.py``: N processes
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-4-federate via ``jax.distributed`` into ONE multi-controller program, the
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:5:("lat",) mesh spans the GLOBAL device set, and ``make_sharded_ocean_step`` +
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-6-the band ppermute/psum halo (incl. the barotropic ``_global_sum_pair`` psum)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-7-run UNCHANGED — collectives cross processes via the distributed runtime
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-8-(NCCL/gloo). NO mpi4jax anywhere in this file (mixed-stack deadlock hazard),
--
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-69-)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-70-from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: E402
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-71-    gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:72:    make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:73:    shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-74-)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-75-from legoesm.ocean.state import LatLonCGridOceanConfig  # noqa: E402
--
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-122-    # serial run above already did; belt-and-braces for wrapper band masks).
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-123-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-124-
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:125:    step = make_sharded_ocean_step(model, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:126:    ss = shard_state_latlon(state0, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-127-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-128-        ss = step(ss, dt)
--
tests/parallel/test_latlon_vface_reconstruct.py-53-    v_lower = jnp.asarray(v_full[:N_LAT])  # n_lat rows, divisible by N_DEV
tests/parallel/test_latlon_vface_reconstruct.py-54-
tests/parallel/test_latlon_vface_reconstruct.py-55-    isp = P("lat", *((None,) * (v_lower.ndim - 1)))
tests/parallel/test_latlon_vface_reconstruct.py:56:    vl_sh = jax.device_put(v_lower, NamedSharding(mesh, isp))
tests/parallel/test_latlon_vface_reconstruct.py-57-
tests/parallel/test_latlon_vface_reconstruct.py-58-    @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp, check_vma=False)
tests/parallel/test_latlon_vface_reconstruct.py-59-    def f(vl):
--
tests/parallel/test_latlon_vface_reconstruct.py-84-    v_full[-1] = 0.0
tests/parallel/test_latlon_vface_reconstruct.py-85-    v_lower = jnp.asarray(v_full[:N_LAT])
tests/parallel/test_latlon_vface_reconstruct.py-86-    isp = P("lat", *((None,) * (v_lower.ndim - 1)))
tests/parallel/test_latlon_vface_reconstruct.py:87:    vl_sh = jax.device_put(v_lower, NamedSharding(mesh, isp))
tests/parallel/test_latlon_vface_reconstruct.py-88-
tests/parallel/test_latlon_vface_reconstruct.py-89-    @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp, check_vma=False)
tests/parallel/test_latlon_vface_reconstruct.py-90-    def rt(vl):
--
tests/parallel/test_latlon_vface_reconstruct.py-118-    u_s, v_s = np.asarray(u_s), np.asarray(v_s)   # (N_LAT,N_LON+1), (N_LAT+1,N_LON)
tests/parallel/test_latlon_vface_reconstruct.py-119-
tests/parallel/test_latlon_vface_reconstruct.py-120-    isp = P("lat", *((None,) * (u_cell.ndim - 1)))
tests/parallel/test_latlon_vface_reconstruct.py:121:    u_sh = jax.device_put(jnp.asarray(u_cell), NamedSharding(mesh, isp))
tests/parallel/test_latlon_vface_reconstruct.py:122:    v_sh = jax.device_put(jnp.asarray(v_cell), NamedSharding(mesh, isp))
tests/parallel/test_latlon_vface_reconstruct.py-123-
tests/parallel/test_latlon_vface_reconstruct.py-124-    activate_latlon_spmd_halo(mesh)   # arm the spmd backend for interp_cell_to_vface_halo
tests/parallel/test_latlon_vface_reconstruct.py-125-    try:
--
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-94-
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-95-    fw = NamedSharding(mesh, P("face", None, None, None))
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-96-    dq_t = stage(
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py:97:        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py:98:        jax.device_put(q, fw), jax.device_put(vert_adv_q, fw))
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-99-
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-100-    rel = _rel_cc(dq_t, dq_g)
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-101-    assert rel < 1e-10, f"dq_dt rel {rel:.3e} (kt={KT})"
--
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-139-    fw = NamedSharding(mesh, P("face", None, None, None))
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-140-    fw5 = NamedSharding(mesh, P("face", None, None, None, None))
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-141-    dq_t = stage(
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py:142:        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py:143:        jax.device_put(q, fw5), jax.device_put(vert_adv_q, fw5))
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-144-    rel = _rel_cc(dq_t, dq_g)
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-145-    assert rel < 1e-10, f"packed dq_dt rel {rel:.3e} (kt={KT})"
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-146-
--
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-219-    fo = NamedSharding(mesh, P("face", None, None))
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-220-    fw5 = NamedSharding(mesh, P("face", None, None, None, None))
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-221-    dq_t = stage(
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py:222:        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py:223:        jax.device_put(T, fw), jax.device_put(p_s, fo),
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py:224:        jax.device_put(q_pack, fw5), jax.device_put(vert_adv_q, fw5))
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-225-    rel = _rel_cc(dq_t, dq_g)
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-226-    assert rel < 1e-10, f"moist tracer tendency rel {rel:.3e} (kt={KT})"
tests/parallel/test_tiled_fv3_hydrostatic_tracer.py-227-
--
tests/parallel/test_tiled_center_to_dgrid_vector.py-96-    stage = make_tiled_center_to_dgrid_vector_stage_2d(mesh, cdg, N, KT, NLEV)
tests/parallel/test_tiled_center_to_dgrid_vector.py-97-
tests/parallel/test_tiled_center_to_dgrid_vector.py-98-    fw = NamedSharding(mesh, P("face", None, None, None))
tests/parallel/test_tiled_center_to_dgrid_vector.py:99:    u_t, v_t = stage(jax.device_put(u_cc, fw), jax.device_put(v_cc, fw))
tests/parallel/test_tiled_center_to_dgrid_vector.py-100-
tests/parallel/test_tiled_center_to_dgrid_vector.py-101-    ru, rv = _compare_corner(np.asarray(u_t), np.asarray(v_t), u_g, v_g, KT, nl)
tests/parallel/test_tiled_center_to_dgrid_vector.py-102-    assert ru < 1e-10, f"u_d rel {ru:.3e} (kt={KT})"
--
scripts/bench/run_cpu_mpi_scaling.py-765-    co = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
scripts/bench/run_cpu_mpi_scaling.py-766-    cz5 = NamedSharding(mesh, P("face", "tile_i", "tile_j", None, None))
scripts/bench/run_cpu_mpi_scaling.py-767-    state = {
scripts/bench/run_cpu_mpi_scaling.py:768:        "u_d": jax.device_put(
scripts/bench/run_cpu_mpi_scaling.py-769-            expand_corners_to_blocks(fv3.u_d.data, kt, nl), cz),
scripts/bench/run_cpu_mpi_scaling.py:770:        "v_d": jax.device_put(
scripts/bench/run_cpu_mpi_scaling.py-771-            expand_corners_to_blocks(fv3.v_d.data, kt, nl), cz),
scripts/bench/run_cpu_mpi_scaling.py:772:        "T": jax.device_put(fv3.T.data, cz),
scripts/bench/run_cpu_mpi_scaling.py:773:        "p_s": jax.device_put(fv3.p_s.data, co),
scripts/bench/run_cpu_mpi_scaling.py:774:        "phis": jax.device_put(fv3.phis.data, co),
scripts/bench/run_cpu_mpi_scaling.py-775-    }
scripts/bench/run_cpu_mpi_scaling.py-776-    if _moist:
scripts/bench/run_cpu_mpi_scaling.py-777-        import jax.numpy as jnp
--
scripts/bench/run_cpu_mpi_scaling.py-779-        # q_pack order [q_v, q_c, q_r] — the tiled moist contract.
scripts/bench/run_cpu_mpi_scaling.py-780-        q_pack = jnp.stack(
scripts/bench/run_cpu_mpi_scaling.py-781-            [fv3.tracers[nm].data for nm in ("q_v", "q_c", "q_r")], axis=-1)
scripts/bench/run_cpu_mpi_scaling.py:782:        state["q_pack"] = jax.device_put(q_pack, cz5)
scripts/bench/run_cpu_mpi_scaling.py-783-
scripts/bench/run_cpu_mpi_scaling.py-784-    _dt_built = float(dt)
scripts/bench/run_cpu_mpi_scaling.py-785-
--
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py-7-production bench (``bench_ocean_latlon_spmd_scaling.py --multicontroller``)
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py-8-end-to-end with the ported parity + conservation gates armed — pinning the
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py-9-multi-controller pieces a Derecho/Levante multi-node NCCL ocean run
tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:10:exercises: ``shard_state_latlon`` onto a mesh spanning non-addressable
--
tests/parallel/test_mpas_atm_native_step.py-601-        jmesh = Mesh(np.array(devices), axis_names=("device",))
tests/parallel/test_mpas_atm_native_step.py-602-        dev_sharding = NamedSharding(jmesh, P("device"))
tests/parallel/test_mpas_atm_native_step.py-603-        halo_args = tuple(
tests/parallel/test_mpas_atm_native_step.py:604:            tuple(jax.device_put(a, dev_sharding) for a in (
tests/parallel/test_mpas_atm_native_step.py-605-                sched["send_cell_idx"][r], sched["recv_cell_pos"][r],
tests/parallel/test_mpas_atm_native_step.py-606-                sched["send_edge_idx"][r], sched["recv_edge_pos"][r]))
tests/parallel/test_mpas_atm_native_step.py-607-            for r in range(sched["n_rounds"])
--
scripts/bench/bench_ocean_latlon_spmd_pcg.py-172-            continue
scripts/bench/bench_ocean_latlon_spmd_pcg.py-173-        mesh = Mesh(np.array(devs[:n_dev]), axis_names=("lat",))
scripts/bench/bench_ocean_latlon_spmd_pcg.py-174-        isp = P("lat", None)
scripts/bench/bench_ocean_latlon_spmd_pcg.py:175:        b_sh = jax.device_put(b_host, NamedSharding(mesh, isp))
scripts/bench/bench_ocean_latlon_spmd_pcg.py:176:        x0_sh = jax.device_put(x0_host, NamedSharding(mesh, isp))
scripts/bench/bench_ocean_latlon_spmd_pcg.py-177-
scripts/bench/bench_ocean_latlon_spmd_pcg.py-178-        activate_latlon_spmd_halo(mesh)
scripts/bench/bench_ocean_latlon_spmd_pcg.py-179-        try:
--
tests/test_cases/baroclinic_wave.py-778-
tests/test_cases/baroclinic_wave.py-779-    The multi-device invariant of ``build_sharded_held_suarez_state_atm_latlon``
tests/test_cases/baroclinic_wave.py-780-    for the voronoi lane: every state leaf is created with
tests/test_cases/baroclinic_wave.py:781:    ``jax.make_array_from_callback``, whose callback runs only for the
tests/test_cases/baroclinic_wave.py-782-    cell/edge ranges owned by THIS process's addressable devices — no
tests/test_cases/baroclinic_wave.py:783:    per-process global state build, no global ``device_put``.  The mesh
tests/test_cases/baroclinic_wave.py-784-    itself remains global on every process (its partition-local
tests/test_cases/baroclinic_wave.py-785-    construction through the SFC machinery is the OPEN remainder of
tests/test_cases/baroclinic_wave.py-786-    #1100 — connectivity, not analytic IC).
--
tests/test_cases/baroclinic_wave.py-870-        return jnp.zeros(lat_c[idx[0]].shape)
tests/test_cases/baroclinic_wave.py-871-
tests/test_cases/baroclinic_wave.py-872-    def _make(gshape, cb):
tests/test_cases/baroclinic_wave.py:873:        return jax.make_array_from_callback(gshape, shard, cb)
tests/test_cases/baroclinic_wave.py-874-
tests/test_cases/baroclinic_wave.py-875-    T_arr = _make((nCells, nlev), _T_cb)
tests/test_cases/baroclinic_wave.py-876-    u_arr = _make((nEdges, nlev), _u_cb)
--
tests/parallel/test_tiled_fv3_hydrostatic_thermo.py-147-    fw = NamedSharding(mesh, P("face", None, None, None))
tests/parallel/test_tiled_fv3_hydrostatic_thermo.py-148-    fo = NamedSharding(mesh, P("face", None, None))
tests/parallel/test_tiled_fv3_hydrostatic_thermo.py-149-    dT_t = stage(
tests/parallel/test_tiled_fv3_hydrostatic_thermo.py:150:        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
tests/parallel/test_tiled_fv3_hydrostatic_thermo.py:151:        jax.device_put(T, fw), jax.device_put(p_s, fo),
tests/parallel/test_tiled_fv3_hydrostatic_thermo.py:152:        jax.device_put(omega, fw), jax.device_put(vert_adv_T, fw))
tests/parallel/test_tiled_fv3_hydrostatic_thermo.py-153-
tests/parallel/test_tiled_fv3_hydrostatic_thermo.py-154-    rel = _rel_cc(dT_t, dT_g)
tests/parallel/test_tiled_fv3_hydrostatic_thermo.py-155-    assert rel < 1e-10, f"dT_dt rel {rel:.3e} (kt={KT}, hybrid={hybrid})"
--
tests/parallel/test_tiled_fv3_sw_momentum.py-81-
tests/parallel/test_tiled_fv3_sw_momentum.py-82-    fo = P("face", None, None)
tests/parallel/test_tiled_fv3_sw_momentum.py-83-    sh_fo = NamedSharding(mesh, fo)
tests/parallel/test_tiled_fv3_sw_momentum.py:84:    h_sh = jax.device_put(h, sh_fo)
tests/parallel/test_tiled_fv3_sw_momentum.py:85:    u_sh = jax.device_put(u_d, sh_fo)
tests/parallel/test_tiled_fv3_sw_momentum.py:86:    v_sh = jax.device_put(v_d, sh_fo)
tests/parallel/test_tiled_fv3_sw_momentum.py:87:    hs_sh = jax.device_put(h_s, sh_fo)
tests/parallel/test_tiled_fv3_sw_momentum.py-88-
tests/parallel/test_tiled_fv3_sw_momentum.py-89-    du_t, dv_t = stage(h_sh, u_sh, v_sh, hs_sh)
tests/parallel/test_tiled_fv3_sw_momentum.py-90-    du_t = np.asarray(du_t)   # (6, KT*NL, KT*(NL+1))
--
tests/parallel/test_tiled_vector_pad.py-76-    co = P("face", "tile_i", "tile_j")
tests/parallel/test_tiled_vector_pad.py-77-    sh_fo = NamedSharding(mesh, fo)
tests/parallel/test_tiled_vector_pad.py-78-
tests/parallel/test_tiled_vector_pad.py:79:    u_sh = jax.device_put(u, sh_fo)
tests/parallel/test_tiled_vector_pad.py:80:    v_sh = jax.device_put(v, sh_fo)
tests/parallel/test_tiled_vector_pad.py:81:    ca_sh = jax.device_put(ca, sh_fo)
tests/parallel/test_tiled_vector_pad.py:82:    sa_sh = jax.device_put(sa, sh_fo)
tests/parallel/test_tiled_vector_pad.py:83:    cap_sh = jax.device_put(cap, sh_fo)
tests/parallel/test_tiled_vector_pad.py:84:    sap_sh = jax.device_put(sap, sh_fo)
tests/parallel/test_tiled_vector_pad.py-85-    offs_j = jnp.asarray(offs)
tests/parallel/test_tiled_vector_pad.py-86-
tests/parallel/test_tiled_vector_pad.py-87-    @partial(shard_map, mesh=mesh,
--
tests/parallel/test_latlon_ocean_spmd_tripole.py-1-"""SPMD equivalence gate for the lat-lon C-grid ocean step on a TRIPOLE grid.
tests/parallel/test_latlon_ocean_spmd_tripole.py-2-
tests/parallel/test_latlon_ocean_spmd_tripole.py-3-The tripole twin of ``test_latlon_ocean_spmd_step.py`` (regular grid): N steps of
tests/parallel/test_latlon_ocean_spmd_tripole.py:4:``make_sharded_ocean_step`` on a synthetic tripole (active bipolar north fold)
tests/parallel/test_latlon_ocean_spmd_tripole.py-5-across 4 (CPU) devices must match the single-device step to the same
tests/parallel/test_latlon_ocean_spmd_tripole.py-6-floating-point re-association tolerance.  This is the CORRECTNESS GATE for the
tests/parallel/test_latlon_ocean_spmd_tripole.py-7-tripole north fold under SPMD (the data-dependent ``north_fold_mask`` /
--
tests/parallel/test_latlon_ocean_spmd_tripole.py-85-def _have_sharded_step():
tests/parallel/test_latlon_ocean_spmd_tripole.py-86-    try:
tests/parallel/test_latlon_ocean_spmd_tripole.py-87-        from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: F401
tests/parallel/test_latlon_ocean_spmd_tripole.py:88:            make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_tripole.py-89-        )
tests/parallel/test_latlon_ocean_spmd_tripole.py-90-        from legoesm.parallel.mesh import create_latlon_mesh  # noqa: F401
tests/parallel/test_latlon_ocean_spmd_tripole.py-91-        return True
--
tests/parallel/test_latlon_ocean_spmd_tripole.py-112-    # step on a tripole (north fold + barotropic solve).
tests/parallel/test_latlon_ocean_spmd_tripole.py-113-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_tripole.py-114-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_tripole.py:115:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_tripole.py:116:        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_tripole.py-117-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_tripole.py-118-    )
--
tests/parallel/test_latlon_ocean_spmd_tripole.py-149-    model._ensure_vertex_mask(state0)        # prime the build-once vmask cache
tests/parallel/test_latlon_ocean_spmd_tripole.py-150-
tests/parallel/test_latlon_ocean_spmd_tripole.py-151-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_tripole.py:152:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py:153:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-154-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_tripole.py-155-        ss = step(ss, dt, surface_forcing=sf)
--
scripts/bench/bench_cube_shardmap_halo.py-390-
scripts/bench/bench_cube_shardmap_halo.py-391-            has_collective = _hlo_has_ppermute(_f, x_sh)
scripts/bench/bench_cube_shardmap_halo.py-392-            y_sh, vjp_sh = jax.vjp(_f, x_sh)
scripts/bench/bench_cube_shardmap_halo.py:393:            w_sh = jax.device_put(w, dev_config.face_sharding)
scripts/bench/bench_cube_shardmap_halo.py-394-            g_spmd = np.asarray(jax.device_get(vjp_sh(w_sh)[0]))
scripts/bench/bench_cube_shardmap_halo.py-395-    finally:
scripts/bench/bench_cube_shardmap_halo.py-396-        _restore_halo(snap)
--
tests/parallel/test_segment_sharding_device_config.py-329-        respective arg ... float32[..,nlev] ... spec=P() vs spec=P('face',)".
tests/parallel/test_segment_sharding_device_config.py-330-
tests/parallel/test_segment_sharding_device_config.py-331-        Here we put exactly such a leaf — a ``[6*N*N, NLEV]`` array
tests/parallel/test_segment_sharding_device_config.py:332:        ``device_put`` onto ``P("face")`` (sharding its cell axis) — into
tests/parallel/test_segment_sharding_device_config.py-333-        the carry's ``conv_prog`` slot (the mock step passes ``conv_prog``
tests/parallel/test_segment_sharding_device_config.py-334-        through unchanged, so its shape is unconstrained) and assert the
tests/parallel/test_segment_sharding_device_config.py-335-        sharded segment MATCHES it: it runs without the mismatch crash and
--
tests/parallel/test_segment_sharding_device_config.py-356-        conv_prog_flat = jnp.arange(
tests/parallel/test_segment_sharding_device_config.py-357-            n_cells * NLEV, dtype=jnp.float32,
tests/parallel/test_segment_sharding_device_config.py-358-        ).reshape(n_cells, NLEV)
tests/parallel/test_segment_sharding_device_config.py:359:        conv_prog_sharded = jax.device_put(
tests/parallel/test_segment_sharding_device_config.py-360-            conv_prog_flat, NamedSharding(dev_config.mesh, PartitionSpec("face")),
tests/parallel/test_segment_sharding_device_config.py-361-        )
tests/parallel/test_segment_sharding_device_config.py-362-        # Sanity: the input genuinely shards the FLAT cell axis (the
--
tests/parallel/test_atm_latlon_bandlocal_build.py-8-
tests/parallel/test_atm_latlon_bandlocal_build.py-9-Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=4``);
tests/parallel/test_atm_latlon_bandlocal_build.py-10-skips when fewer devices are available.  The multi-process
tests/parallel/test_atm_latlon_bandlocal_build.py:11:(``make_array_from_callback`` per-process rows) leg of #1100 needs a real
tests/parallel/test_atm_latlon_bandlocal_build.py-12-multi-controller launch and is validated by the bench itself on cluster —
tests/parallel/test_atm_latlon_bandlocal_build.py-13-this gate pins the single-process semantics both legs share.
tests/parallel/test_atm_latlon_bandlocal_build.py-14-"""
--
tests/parallel/test_atm_latlon_operator_split_spmd.py-190-    for _ in range(n_steps):
tests/parallel/test_atm_latlon_operator_split_spmd.py-191-        c_b = step_dev(c_b, f_b)
tests/parallel/test_atm_latlon_operator_split_spmd.py-192-    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
tests/parallel/test_atm_latlon_operator_split_spmd.py:193:    c_b = jax.tree.map(lambda x: np.asarray(jax.device_put(x, rep)), c_b)
tests/parallel/test_atm_latlon_operator_split_spmd.py-194-
tests/parallel/test_atm_latlon_operator_split_spmd.py-195-    # Non-vacuity: tke genuinely evolved off its 1e-4 floor (AR1 -> ~0.0037).
tests/parallel/test_atm_latlon_operator_split_spmd.py-196-    tke_grew = float(np.max(np.abs(np.asarray(c_s.tke) - 1e-4)))
--
tests/parallel/test_tiled_zero_mean_tendency.py-57-    dev = np.array(jax.devices()[:ndev]).reshape(6, KT, KT)
tests/parallel/test_tiled_zero_mean_tendency.py-58-    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
tests/parallel/test_tiled_zero_mean_tendency.py-59-    stage = make_tiled_zero_mean_tendency_stage_2d(mesh, grid, N, KT)
tests/parallel/test_tiled_zero_mean_tendency.py:60:    t = np.asarray(stage(jax.device_put(
tests/parallel/test_tiled_zero_mean_tendency.py-61-        tend, NamedSharding(mesh, P("face", None, None)))))
tests/parallel/test_tiled_zero_mean_tendency.py-62-
tests/parallel/test_tiled_zero_mean_tendency.py-63-    # (a) match the global op (psum reorders the global scalar -> ULP drift).
--
tests/parallel/test_tiled_d2a2c_ua_va.py-125-    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
tests/parallel/test_tiled_d2a2c_ua_va.py-126-    co = P("face", "tile_i", "tile_j")
tests/parallel/test_tiled_d2a2c_ua_va.py-127-    sh = NamedSharding(mesh, co)
tests/parallel/test_tiled_d2a2c_ua_va.py:128:    args = [jax.device_put(x, sh) for x in (utmp_int, vtmp_int, cos_sg5, rsin2)]
tests/parallel/test_tiled_d2a2c_ua_va.py-129-
tests/parallel/test_tiled_d2a2c_ua_va.py-130-    @partial(shard_map, mesh=mesh, in_specs=(co, co, co, co),
tests/parallel/test_tiled_d2a2c_ua_va.py-131-             out_specs=(co, co), check_vma=False)
--
tests/parallel/test_latlon_spmd_pcg.py-105-    ref_sq = float(jnp.sum(a * a))
tests/parallel/test_latlon_spmd_pcg.py-106-
tests/parallel/test_latlon_spmd_pcg.py-107-    isp = P("lat", None)
tests/parallel/test_latlon_spmd_pcg.py:108:    a_sh = jax.device_put(a, NamedSharding(mesh, isp))
tests/parallel/test_latlon_spmd_pcg.py:109:    b_sh = jax.device_put(b, NamedSharding(mesh, isp))
tests/parallel/test_latlon_spmd_pcg.py-110-
tests/parallel/test_latlon_spmd_pcg.py-111-    @partial(shard_map, mesh=mesh, in_specs=(isp, isp),
tests/parallel/test_latlon_spmd_pcg.py-112-             out_specs=P(), check_vma=False)
--
tests/parallel/test_latlon_spmd_pcg.py-150-    # Sharded solve: arm the spmd backend so (a) A_op's pad routes to the
tests/parallel/test_latlon_spmd_pcg.py-151-    # band halo body and (b) _global_dot_batch routes to psum.
tests/parallel/test_latlon_spmd_pcg.py-152-    isp = P("lat", None)
tests/parallel/test_latlon_spmd_pcg.py:153:    b_sh = jax.device_put(b, NamedSharding(mesh, isp))
tests/parallel/test_latlon_spmd_pcg.py:154:    x0_sh = jax.device_put(x0, NamedSharding(mesh, isp))
tests/parallel/test_latlon_spmd_pcg.py-155-    activate_latlon_spmd_halo(mesh)
tests/parallel/test_latlon_spmd_pcg.py-156-    try:
tests/parallel/test_latlon_spmd_pcg.py-157-        @partial(shard_map, mesh=mesh, in_specs=(isp, isp),
--
tests/parallel/test_latlon_spmd_pcg.py-209-                        lambda local, axis_name: local)
tests/parallel/test_latlon_spmd_pcg.py-210-
tests/parallel/test_latlon_spmd_pcg.py-211-    isp = P("lat", None)
tests/parallel/test_latlon_spmd_pcg.py:212:    b_sh = jax.device_put(b, NamedSharding(mesh, isp))
tests/parallel/test_latlon_spmd_pcg.py:213:    x0_sh = jax.device_put(x0, NamedSharding(mesh, isp))
tests/parallel/test_latlon_spmd_pcg.py-214-    activate_latlon_spmd_halo(mesh)
tests/parallel/test_latlon_spmd_pcg.py-215-    try:
tests/parallel/test_latlon_spmd_pcg.py-216-        @partial(shard_map, mesh=mesh, in_specs=(isp, isp),
--
tests/parallel/test_tiled_dp_s_dt.py-109-
tests/parallel/test_tiled_dp_s_dt.py-110-    fw = NamedSharding(mesh, P("face", None, None, None))
tests/parallel/test_tiled_dp_s_dt.py-111-    fo = NamedSharding(mesh, P("face", None, None))
tests/parallel/test_tiled_dp_s_dt.py:112:    out = stage(jax.device_put(u_d, fw), jax.device_put(v_d, fw),
tests/parallel/test_tiled_dp_s_dt.py:113:                jax.device_put(p_s, fo))
tests/parallel/test_tiled_dp_s_dt.py-114-    a = np.asarray(out).reshape(6, KT, nl, KT, nl)
tests/parallel/test_tiled_dp_s_dt.py-115-    t = _reassemble_cc(lambda ti, tj: a[:, ti, :, tj, :], KT)
tests/parallel/test_tiled_dp_s_dt.py-116-    np.testing.assert_array_equal(
--
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py-14-* the ALL-global-device mesh (``jax.devices()`` spans both processes) with the
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py-15-  strict-subset guard;
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py-16-* the cross-process carry gather (``replicate_leaf(multiprocess=True)`` — the
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py:17:  jit-identity all-gather; a raw ``device_put`` cannot reshard shards on the
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py-18-  other process's device);
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py-19-* an all-rank-consistent exit (the finite/NaN guard reads the REPLICATED
tests/parallel/test_run_amip_latlon_spmd_multicontroller_selfspawn.py-20-  gathered state, so both ranks return the same status in lockstep).
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-5-harness" gap: ``bench_ocean_mpi_scaling.py`` is the route-A (mpi4jax) phase-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-6-split bench and ``bench_ocean_latlon_spmd_pcg.py`` times the barotropic PCG
scripts/bench/bench_ocean_latlon_spmd_scaling.py-7-KERNEL only — neither times the composed production step
scripts/bench/bench_ocean_latlon_spmd_scaling.py:8:(``make_sharded_ocean_step``: baroclinic + barotropic [implicit-CN
scripts/bench/bench_ocean_latlon_spmd_scaling.py-9-free-surface by default, split-explicit via ``--baro-solver``] + implicit
scripts/bench/bench_ocean_latlon_spmd_scaling.py-10-vmix + tracers) under the lat-band SPMD backend.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-11-
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-22-Multi-controller (route-B, ``--multicontroller``): identical contract to the
scripts/bench/bench_ocean_latlon_spmd_scaling.py-23-atm bench — every process calls ``jax.distributed.initialize`` BEFORE any
scripts/bench/bench_ocean_latlon_spmd_scaling.py-24-other JAX use, the ("lat",) mesh is built over the GLOBAL ``jax.devices()``,
scripts/bench/bench_ocean_latlon_spmd_scaling.py:25:and the existing ``make_sharded_ocean_step`` band-ppermute halo + psum
scripts/bench/bench_ocean_latlon_spmd_scaling.py-26-reductions (incl. the barotropic ``_global_sum_pair``) run unchanged across
scripts/bench/bench_ocean_latlon_spmd_scaling.py-27-processes (NCCL on GPU / gloo on CPU). NO mpi4jax is armed in this mode (the
scripts/bench/bench_ocean_latlon_spmd_scaling.py-28-documented mixed-stack deadlock hazard).
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-343-        init_multicontroller_distributed(args.coordinator)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-344-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-345-    from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/bench/bench_ocean_latlon_spmd_scaling.py:346:        make_sharded_ocean_step,
scripts/bench/bench_ocean_latlon_spmd_scaling.py:347:        shard_state_latlon,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-348-    )
scripts/bench/bench_ocean_latlon_spmd_scaling.py-349-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-350-    nd = args.n_devices
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-380-    # compiles a global-sized init program — the 102 GB wall that killed
scripts/bench/bench_ocean_latlon_spmd_scaling.py-381-    # LL2304@64 (probe 26523157: the compiled STEP is clean; the residency
scripts/bench/bench_ocean_latlon_spmd_scaling.py-382-    # is setup-time). Host-side globals are RAM, and only the per-band
scripts/bench/bench_ocean_latlon_spmd_scaling.py:383:    # shards reach the accelerator via shard_state_latlon's device_put.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-384-    # Identical values on every process (deterministic init + the step
scripts/bench/bench_ocean_latlon_spmd_scaling.py-385-    # factory's existing process-0 broadcast + content-hash guard).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-386-    # nd==1 keeps the old on-device build: that lane TIMES the
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-447-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-448-    if nd == 1:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-449-        mesh = None
scripts/bench/bench_ocean_latlon_spmd_scaling.py:450:        step = make_sharded_ocean_step(model, None)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-451-        s = s0
scripts/bench/bench_ocean_latlon_spmd_scaling.py-452-    else:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-453-        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-454-                                 axis_names=("lat",))
scripts/bench/bench_ocean_latlon_spmd_scaling.py:455:        step = make_sharded_ocean_step(model, mesh)
scripts/bench/bench_ocean_latlon_spmd_scaling.py:456:        s = shard_state_latlon(s0, mesh)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-457-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-458-    # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
scripts/bench/bench_ocean_latlon_spmd_scaling.py-459-    # blocks with sync only AROUND the block — the previous per-step
--
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py-14-real decomposition bug is O(1e-3)).
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py-15-
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py-16-What this pins beyond the single-process gate: the multi-controller
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py:17:construction path — ``shard_state_atm_latlon``'s ``device_put`` onto a mesh
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py-18-spanning NON-addressable devices, cross-process ppermute/psum inside the
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py-19-jitted shard_map, and ``gather_state_atm_latlon``'s replication gather — i.e.
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py-20-exactly the pieces a Derecho/Levante multi-node NCCL run exercises.
--
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py-68-    assert len(jax.devices()) == N_PROC, jax.devices()
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py-69-
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py-70-    # Deterministic identical build on every process (the multi-controller
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py:71:    # contract: device_put slices the same host-global array per process).
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py-72-    grid = create_latlon_grid(
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py-73-        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py-74-        omega=constants.Omega)
--
tests/parallel/test_conservation_spmd_lat_reduction.py-62-    ref2 = float(np.sum(f2 * area))
tests/parallel/test_conservation_spmd_lat_reduction.py-63-
tests/parallel/test_conservation_spmd_lat_reduction.py-64-    isp = P("lat", None)
tests/parallel/test_conservation_spmd_lat_reduction.py:65:    a_sh = jax.device_put(jnp.asarray(area), NamedSharding(mesh, isp))
tests/parallel/test_conservation_spmd_lat_reduction.py:66:    f1_sh = jax.device_put(jnp.asarray(f1), NamedSharding(mesh, isp))
tests/parallel/test_conservation_spmd_lat_reduction.py:67:    f2_sh = jax.device_put(jnp.asarray(f2), NamedSharding(mesh, isp))
tests/parallel/test_conservation_spmd_lat_reduction.py-68-
tests/parallel/test_conservation_spmd_lat_reduction.py-69-    activate_latlon_spmd_halo(mesh)  # backend -> 'spmd', mesh -> this 'lat' mesh
tests/parallel/test_conservation_spmd_lat_reduction.py-70-    try:
--
tests/parallel/test_conservation_spmd_lat_reduction.py-108-    ref = float(np.sum(field * area))          # true global area-weighted sum
tests/parallel/test_conservation_spmd_lat_reduction.py-109-
tests/parallel/test_conservation_spmd_lat_reduction.py-110-    isp = P("lat", None)
tests/parallel/test_conservation_spmd_lat_reduction.py:111:    a_sh = jax.device_put(jnp.asarray(area), NamedSharding(mesh, isp))
tests/parallel/test_conservation_spmd_lat_reduction.py:112:    f_sh = jax.device_put(jnp.asarray(field), NamedSharding(mesh, isp))
tests/parallel/test_conservation_spmd_lat_reduction.py-113-
tests/parallel/test_conservation_spmd_lat_reduction.py-114-    activate_latlon_spmd_halo(mesh)
tests/parallel/test_conservation_spmd_lat_reduction.py-115-    try:
--
tests/parallel/test_tiled_fv3_hydrostatic_tendencies.py-137-    fw = NamedSharding(mesh, P("face", None, None, None))
tests/parallel/test_tiled_fv3_hydrostatic_tendencies.py-138-    fo = NamedSharding(mesh, P("face", None, None))
tests/parallel/test_tiled_fv3_hydrostatic_tendencies.py-139-    du_t, dv_t, dT_t, dps_t = stage(
tests/parallel/test_tiled_fv3_hydrostatic_tendencies.py:140:        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
tests/parallel/test_tiled_fv3_hydrostatic_tendencies.py:141:        jax.device_put(T, fw), jax.device_put(p_s, fo),
tests/parallel/test_tiled_fv3_hydrostatic_tendencies.py:142:        jax.device_put(phis, fo))
tests/parallel/test_tiled_fv3_hydrostatic_tendencies.py-143-
tests/parallel/test_tiled_fv3_hydrostatic_tendencies.py-144-    r_du = _rel_corner(np.asarray(du_t), du_g, KT, nl)
tests/parallel/test_tiled_fv3_hydrostatic_tendencies.py-145-    r_dv = _rel_corner(np.asarray(dv_t), dv_g, KT, nl)
--
tests/parallel/test_warmup_tiled_cube_comms.py-74-    mesh = _mesh()
tests/parallel/test_warmup_tiled_cube_comms.py-75-    perms = (list(get_tiled_tables(KT).perms)
tests/parallel/test_warmup_tiled_cube_comms.py-76-             + list(tiled_guard_perms(KT)) + list(tiled_diag_perms(KT)))
tests/parallel/test_warmup_tiled_cube_comms.py:77:    dummy = jax.device_put(np.zeros((6, KT, KT), np.float32),
tests/parallel/test_warmup_tiled_cube_comms.py-78-                           NamedSharding(mesh, P(*AXES)))
tests/parallel/test_warmup_tiled_cube_comms.py-79-
tests/parallel/test_warmup_tiled_cube_comms.py-80-    @partial(shard_map, mesh=mesh, in_specs=P(*AXES), out_specs=P(*AXES),
--
tests/ocean/unit/test_sharded_geom_fingerprint.py-1-"""Unit tests for the band-geometry cross-process fingerprint gate.
tests/ocean/unit/test_sharded_geom_fingerprint.py-2-
tests/ocean/unit/test_sharded_geom_fingerprint.py:3:The gate decides whether ``make_sharded_ocean_step`` accepts per-process
tests/ocean/unit/test_sharded_geom_fingerprint.py-4-band-geometry stacks without the (removed, nd-linear-cost) process-0
tests/ocean/unit/test_sharded_geom_fingerprint.py-5-broadcast — see the 2026-08-03 fix note at the sharded put. These tests
tests/ocean/unit/test_sharded_geom_fingerprint.py-6-pin the gate's discrimination properties single-process (the
--
scripts/cluster/scaling_derecho/mc_nccl_probe.py-82-
scripts/cluster/scaling_derecho/mc_nccl_probe.py-83-    mesh = Mesh(np.array(gdev), ("lat",))
scripts/cluster/scaling_derecho/mc_nccl_probe.py-84-    x = jnp.arange(n * 8, dtype=jnp.float32).reshape(n, 8)
scripts/cluster/scaling_derecho/mc_nccl_probe.py:85:    xs = jax.device_put(x, NamedSharding(mesh, P("lat", None)))
scripts/cluster/scaling_derecho/mc_nccl_probe.py-86-    perm = [(i, (i + 1) % n) for i in range(n)]
scripts/cluster/scaling_derecho/mc_nccl_probe.py-87-
scripts/cluster/scaling_derecho/mc_nccl_probe.py-88-    from jax.experimental.shard_map import shard_map
--
scripts/cluster/scaling_derecho/mc_nccl_probe.py-113-        return 1
scripts/cluster/scaling_derecho/mc_nccl_probe.py-114-
scripts/cluster/scaling_derecho/mc_nccl_probe.py-115-    # Stage 5: ppermute latency (small message — the halo latency regime).
scripts/cluster/scaling_derecho/mc_nccl_probe.py:116:    small = jax.device_put(
scripts/cluster/scaling_derecho/mc_nccl_probe.py-117-        jnp.ones((n, 64), jnp.float32), NamedSharding(mesh, P("lat", None)))
scripts/cluster/scaling_derecho/mc_nccl_probe.py-118-    g = jax.jit(shard_map(lambda a: jax.lax.ppermute(a, "lat", perm),
scripts/cluster/scaling_derecho/mc_nccl_probe.py-119-                          mesh=mesh, in_specs=P("lat", None),
--
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-13-# Runs the SINGLE-PROCESS multi-device SPMD lane of
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-14-# scripts/bench/bench_ocean_latlon_spmd_scaling.py: the FULL
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-15-# lat-lon C-grid ocean step sharded over 1/2/4 A100s via
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs:16:# make_sharded_ocean_step (shard_map ppermute band halos + psum reductions
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-17-# over NVLink/NCCL; NO mpi4jax — uses the plain `legoesm-gpu` env from
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-18-# README Step 1, no route-A overlay needed).
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-19-#
--
packages/core/legoesm/parallel/mesh.py-746-# Pytree sharding utilities
packages/core/legoesm/parallel/mesh.py-747-# ==============================================================================
packages/core/legoesm/parallel/mesh.py-748-
packages/core/legoesm/parallel/mesh.py:749:def multiprocess_safe_device_put(leaf, sharding):
packages/core/legoesm/parallel/mesh.py:750:    """``jax.device_put`` that is safe under multi-controller SPMD.
packages/core/legoesm/parallel/mesh.py-751-
packages/core/legoesm/parallel/mesh.py:752:    ``jax.device_put(x, sharding)`` with a sharding that spans processes
packages/core/legoesm/parallel/mesh.py-753-    ASSERTS the value is bit-identical on every process.  Per-process XLA
packages/core/legoesm/parallel/mesh.py-754-    autotuning can legitimately pick different kernels on different nodes,
packages/core/legoesm/parallel/mesh.py-755-    producing last-bit differences in host-precomputed inputs (first hit:
packages/core/legoesm/parallel/mesh.py-756-    the external-forcing leaves on the 2-node Levante cs_spmd receipt run,
packages/core/legoesm/parallel/mesh.py-757-    #693 job 26030677 — values identical to 8 significant digits, assert
packages/core/legoesm/parallel/mesh.py-758-    still trips).  Under >1 process, build the global array from each
packages/core/legoesm/parallel/mesh.py:759:    process's LOCAL copy via ``jax.make_array_from_callback`` instead —
packages/core/legoesm/parallel/mesh.py-760-    each process materializes only its addressable shards, no cross-process
packages/core/legoesm/parallel/mesh.py-761-    equality requirement, no communication.
packages/core/legoesm/parallel/mesh.py-762-
packages/core/legoesm/parallel/mesh.py-763-    Single-process (and non-array leaves) delegate to plain
packages/core/legoesm/parallel/mesh.py:764:    ``jax.device_put`` — byte-identical behavior to before.
packages/core/legoesm/parallel/mesh.py-765-    Already-global (non-fully-addressable) leaves pass through unchanged.
packages/core/legoesm/parallel/mesh.py-766-    """
packages/core/legoesm/parallel/mesh.py-767-    if not isinstance(leaf, (jax.Array, jnp.ndarray)):
packages/core/legoesm/parallel/mesh.py:768:        return jax.device_put(leaf, sharding)
packages/core/legoesm/parallel/mesh.py-769-    if isinstance(leaf, jax.Array) and not leaf.is_fully_addressable:
packages/core/legoesm/parallel/mesh.py-770-        return leaf  # already a global sharded array; nothing to place
packages/core/legoesm/parallel/mesh.py-771-    if jax.process_count() > 1:
packages/core/legoesm/parallel/mesh.py-772-        import numpy as np
packages/core/legoesm/parallel/mesh.py-773-        host = np.asarray(leaf)
packages/core/legoesm/parallel/mesh.py:774:        return jax.make_array_from_callback(
packages/core/legoesm/parallel/mesh.py-775-            host.shape, sharding, lambda idx: host[idx])
packages/core/legoesm/parallel/mesh.py:776:    return jax.device_put(leaf, sharding)
packages/core/legoesm/parallel/mesh.py-777-
packages/core/legoesm/parallel/mesh.py-778-
packages/core/legoesm/parallel/mesh.py-779-def shard_pytree(pytree, config: DeviceConfig):
--
packages/core/legoesm/parallel/mesh.py-818-                # face-only sharding.
packages/core/legoesm/parallel/mesh.py-819-                if config.tiling != (1, 1) and leaf.ndim < 3:
packages/core/legoesm/parallel/mesh.py-820-                    sharding = tiled_face_only or config.face_sharding
packages/core/legoesm/parallel/mesh.py:821:                    return multiprocess_safe_device_put(leaf, sharding)
packages/core/legoesm/parallel/mesh.py-822-                if config.tiling != (1, 1) and leaf.ndim >= 3:
packages/core/legoesm/parallel/mesh.py-823-                    # STAGGERED face-plane leaves — D-grid winds
packages/core/legoesm/parallel/mesh.py-824-                    # (6, n+1, n, ...) / (6, n, n+1, ...) — cannot
--
packages/core/legoesm/parallel/mesh.py-831-                    tx, ty = config.tiling
packages/core/legoesm/parallel/mesh.py-832-                    if leaf.shape[1] % tx != 0 or leaf.shape[2] % ty != 0:
packages/core/legoesm/parallel/mesh.py-833-                        sharding = tiled_face_only or config.face_sharding
packages/core/legoesm/parallel/mesh.py:834:                        return multiprocess_safe_device_put(leaf, sharding)
packages/core/legoesm/parallel/mesh.py:835:                return multiprocess_safe_device_put(leaf, config.face_sharding)
packages/core/legoesm/parallel/mesh.py:836:            return multiprocess_safe_device_put(leaf, config.replicated_sharding)
packages/core/legoesm/parallel/mesh.py-837-
packages/core/legoesm/parallel/mesh.py-838-        elif config.grid_type == "cubed_sphere_level":
packages/core/legoesm/parallel/mesh.py-839-            # Issue #273 follow-up: level-parallel cubed-sphere mesh.
--
packages/core/legoesm/parallel/mesh.py-843-            # builds its own column mesh on the same device set and
packages/core/legoesm/parallel/mesh.py-844-            # shards there (see ``build_physics_pipeline`` for the
packages/core/legoesm/parallel/mesh.py-845-            # column-mesh construction).
packages/core/legoesm/parallel/mesh.py:846:            return multiprocess_safe_device_put(leaf, config.replicated_sharding)
packages/core/legoesm/parallel/mesh.py-847-
packages/core/legoesm/parallel/mesh.py-848-        elif config.grid_type == "latlon":
packages/core/legoesm/parallel/mesh.py-849-            if leaf.ndim >= 2:
packages/core/legoesm/parallel/mesh.py:850:                return multiprocess_safe_device_put(leaf, config.face_sharding)
packages/core/legoesm/parallel/mesh.py:851:            return multiprocess_safe_device_put(leaf, config.replicated_sharding)
packages/core/legoesm/parallel/mesh.py-852-
packages/core/legoesm/parallel/mesh.py-853-        elif config.grid_type == "spectral":
packages/core/legoesm/parallel/mesh.py-854-            # Spectral state arrays have shape (n_sh, nlev).
--
packages/core/legoesm/parallel/mesh.py-857-                level_axis_sharding = NamedSharding(
packages/core/legoesm/parallel/mesh.py-858-                    config.mesh, P(None, "level"),
packages/core/legoesm/parallel/mesh.py-859-                )
packages/core/legoesm/parallel/mesh.py:860:                return multiprocess_safe_device_put(leaf, level_axis_sharding)
packages/core/legoesm/parallel/mesh.py-861-            if leaf.ndim == 1:
packages/core/legoesm/parallel/mesh.py-862-                # 1D arrays (e.g., lnps_hat): replicate across devices.
packages/core/legoesm/parallel/mesh.py:863:                return multiprocess_safe_device_put(leaf, config.replicated_sharding)
packages/core/legoesm/parallel/mesh.py:864:            return multiprocess_safe_device_put(leaf, config.replicated_sharding)
packages/core/legoesm/parallel/mesh.py-865-
packages/core/legoesm/parallel/mesh.py-866-        elif config.grid_type == "voronoi":
packages/core/legoesm/parallel/mesh.py-867-            # Voronoi state arrays: shard cell- and edge-centered arrays
--
packages/core/legoesm/parallel/mesh.py-869-            if config.voronoi_dims is not None and leaf.ndim >= 1:
packages/core/legoesm/parallel/mesh.py-870-                nCells, nEdges, _nVerts = config.voronoi_dims
packages/core/legoesm/parallel/mesh.py-871-                if leaf.shape[0] in (nCells, nEdges):
packages/core/legoesm/parallel/mesh.py:872:                    return multiprocess_safe_device_put(leaf, config.face_sharding)
packages/core/legoesm/parallel/mesh.py:873:            return multiprocess_safe_device_put(leaf, config.replicated_sharding)
packages/core/legoesm/parallel/mesh.py-874-
packages/core/legoesm/parallel/mesh.py-875-        # Default: replicate
packages/core/legoesm/parallel/mesh.py:876:        return multiprocess_safe_device_put(leaf, config.replicated_sharding)
packages/core/legoesm/parallel/mesh.py-877-
packages/core/legoesm/parallel/mesh.py-878-    return jax.tree.map(_shard_leaf, pytree)
packages/core/legoesm/parallel/mesh.py-879-
--
packages/core/legoesm/parallel/mesh.py-1091-        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
packages/core/legoesm/parallel/mesh.py-1092-            return leaf
packages/core/legoesm/parallel/mesh.py-1093-        if leaf.ndim >= 1 and leaf.shape[0] == nlev:
packages/core/legoesm/parallel/mesh.py:1094:            return jax.device_put(leaf, config.face_sharding)
packages/core/legoesm/parallel/mesh.py:1095:        return jax.device_put(leaf, config.replicated_sharding)
packages/core/legoesm/parallel/mesh.py-1096-
packages/core/legoesm/parallel/mesh.py-1097-    return jax.tree.map(_shard_leaf, pytree)
packages/core/legoesm/parallel/mesh.py-1098-
--
packages/core/legoesm/parallel/mesh.py-1119-    def _replicate_leaf(leaf):
packages/core/legoesm/parallel/mesh.py-1120-        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
packages/core/legoesm/parallel/mesh.py-1121-            return leaf
packages/core/legoesm/parallel/mesh.py:1122:        return jax.device_put(leaf, config.replicated_sharding)
packages/core/legoesm/parallel/mesh.py-1123-
packages/core/legoesm/parallel/mesh.py-1124-    return jax.tree.map(_replicate_leaf, pytree)
packages/core/legoesm/parallel/mesh.py-1125-
--
packages/core/legoesm/parallel/metal.py-105-    if backend == "mps":
packages/core/legoesm/parallel/metal.py-106-        cpu_device = jax.devices("cpu")[0]
packages/core/legoesm/parallel/metal.py-107-        return SpectralDevicePlacement(
packages/core/legoesm/parallel/metal.py:108:            grid=jax.device_put(grid, cpu_device),
packages/core/legoesm/parallel/metal.py-109-            use_cpu_for_spectral=True,
packages/core/legoesm/parallel/metal.py-110-            cpu_device=cpu_device,
packages/core/legoesm/parallel/metal.py-111-            default_device=jax.devices()[0],
--
packages/core/legoesm/parallel/metal.py-132-    jax.Array on CPU.
packages/core/legoesm/parallel/metal.py-133-    """
packages/core/legoesm/parallel/metal.py-134-    cpu = jax.devices("cpu")[0]
packages/core/legoesm/parallel/metal.py:135:    return jax.device_put(array, cpu)
packages/core/legoesm/parallel/metal.py-136-
packages/core/legoesm/parallel/metal.py-137-
packages/core/legoesm/parallel/metal.py-138-def to_metal(array: jax.Array) -> jax.Array:
--
packages/core/legoesm/parallel/metal.py-153-    if backend != "mps":
packages/core/legoesm/parallel/metal.py-154-        return array
packages/core/legoesm/parallel/metal.py-155-    metal = jax.devices()[0]
packages/core/legoesm/parallel/metal.py:156:    return jax.device_put(array, metal)
packages/core/legoesm/parallel/metal.py-157-
packages/core/legoesm/parallel/metal.py-158-
packages/core/legoesm/parallel/metal.py-159-def route_to_cpu(pytree):
--
packages/core/legoesm/parallel/metal.py-169-    Pytree with all array leaves on CPU.
packages/core/legoesm/parallel/metal.py-170-    """
packages/core/legoesm/parallel/metal.py-171-    cpu = jax.devices("cpu")[0]
packages/core/legoesm/parallel/metal.py:172:    return jax.device_put(pytree, cpu)
packages/core/legoesm/parallel/metal.py-173-
packages/core/legoesm/parallel/metal.py-174-
packages/core/legoesm/parallel/metal.py-175-def route_to_default(pytree):
--
packages/core/legoesm/parallel/metal.py-189-    default = getattr(jax.config, "jax_default_device", None)
packages/core/legoesm/parallel/metal.py-190-    if default is None:
packages/core/legoesm/parallel/metal.py-191-        default = jax.devices()[0]
packages/core/legoesm/parallel/metal.py:192:    return jax.device_put(pytree, default)
packages/core/legoesm/parallel/metal.py-193-
packages/core/legoesm/parallel/metal.py-194-
packages/core/legoesm/parallel/metal.py-195-def is_metal_backend() -> bool:
--
packages/core/legoesm/parallel/metal.py-214-        return x
packages/core/legoesm/parallel/metal.py-215-    if _is_on_device(x, device):
packages/core/legoesm/parallel/metal.py-216-        return x
packages/core/legoesm/parallel/metal.py:217:    return jax.device_put(x, device)
packages/core/legoesm/parallel/metal.py-218-
packages/core/legoesm/parallel/metal.py-219-
packages/core/legoesm/parallel/metal.py-220-def ensure_spectral_on_cpu(fn):
--
packages/core/legoesm/parallel/column_shard.py-130-    ``mesh.shape['col']`` — use ``pad_to_shardable`` first if not.
packages/core/legoesm/parallel/column_shard.py-131-    """
packages/core/legoesm/parallel/column_shard.py-132-    if field.ndim == 0:
packages/core/legoesm/parallel/column_shard.py:133:        return jax.device_put(field, NamedSharding(mesh, P()))
packages/core/legoesm/parallel/column_shard.py-134-    trailing = (None,) * (field.ndim - 1)
packages/core/legoesm/parallel/column_shard.py:135:    return jax.device_put(field, NamedSharding(mesh, P("col", *trailing)))
packages/core/legoesm/parallel/column_shard.py-136-
packages/core/legoesm/parallel/column_shard.py-137-
packages/core/legoesm/parallel/column_shard.py-138-def replicate(field: jax.Array, mesh: Mesh) -> jax.Array:
packages/core/legoesm/parallel/column_shard.py-139-    """Place ``field`` replicated across every device in ``mesh``."""
packages/core/legoesm/parallel/column_shard.py:140:    return jax.device_put(field, NamedSharding(mesh, P()))
--
tests/ocean/unit/test_ocean_compatibility.py-129-
tests/ocean/unit/test_ocean_compatibility.py-130-        step_source = inspect.getsource(SpectralOceanModel.step)
tests/ocean/unit/test_ocean_compatibility.py-131-        assert "_use_cpu_for_spectral" in step_source
tests/ocean/unit/test_ocean_compatibility.py:132:        assert "device_put" in step_source
tests/ocean/unit/test_ocean_compatibility.py-133-
tests/ocean/unit/test_ocean_compatibility.py-134-        # Verify batched CPU integration method exists
tests/ocean/unit/test_ocean_compatibility.py-135-        assert hasattr(SpectralOceanModel, '_step_on_cpu')
--
packages/core/legoesm/parallel/sharded_dynamics.py-238-                if config.tiling != (1, 1) and leaf.ndim >= 3:
packages/core/legoesm/parallel/sharded_dynamics.py-239-                    pspec = spec.tiled_3d if leaf.ndim >= 4 else spec.tiled_2d
packages/core/legoesm/parallel/sharded_dynamics.py-240-                    if pspec is not None:
packages/core/legoesm/parallel/sharded_dynamics.py:241:                        return jax.device_put(
packages/core/legoesm/parallel/sharded_dynamics.py-242-                            leaf, NamedSharding(mesh, pspec)
packages/core/legoesm/parallel/sharded_dynamics.py-243-                        )
packages/core/legoesm/parallel/sharded_dynamics.py-244-                pspec = spec.face_3d if leaf.ndim >= 4 else spec.face_2d
packages/core/legoesm/parallel/sharded_dynamics.py:245:                return jax.device_put(leaf, NamedSharding(mesh, pspec))
packages/core/legoesm/parallel/sharded_dynamics.py:246:            return jax.device_put(
packages/core/legoesm/parallel/sharded_dynamics.py-247-                leaf, NamedSharding(mesh, spec.replicated)
packages/core/legoesm/parallel/sharded_dynamics.py-248-            )
packages/core/legoesm/parallel/sharded_dynamics.py-249-
packages/core/legoesm/parallel/sharded_dynamics.py-250-        elif grid_type == "latlon":
packages/core/legoesm/parallel/sharded_dynamics.py-251-            if leaf.ndim >= 2:
packages/core/legoesm/parallel/sharded_dynamics.py-252-                pspec = P("lat", *([None] * (leaf.ndim - 1)))
packages/core/legoesm/parallel/sharded_dynamics.py:253:                return jax.device_put(leaf, NamedSharding(mesh, pspec))
packages/core/legoesm/parallel/sharded_dynamics.py:254:            return jax.device_put(
packages/core/legoesm/parallel/sharded_dynamics.py-255-                leaf, NamedSharding(mesh, spec.replicated)
packages/core/legoesm/parallel/sharded_dynamics.py-256-            )
packages/core/legoesm/parallel/sharded_dynamics.py-257-
packages/core/legoesm/parallel/sharded_dynamics.py:258:        return jax.device_put(
packages/core/legoesm/parallel/sharded_dynamics.py-259-            leaf, NamedSharding(mesh, spec.replicated)
packages/core/legoesm/parallel/sharded_dynamics.py-260-        )
packages/core/legoesm/parallel/sharded_dynamics.py-261-
--
packages/core/legoesm/parallel/sharded_dynamics.py-287-    def _gather(leaf):
packages/core/legoesm/parallel/sharded_dynamics.py-288-        if not isinstance(leaf, (jax.Array, jnp.ndarray)):
packages/core/legoesm/parallel/sharded_dynamics.py-289-            return leaf
packages/core/legoesm/parallel/sharded_dynamics.py:290:        return jax.device_put(leaf, replicated)
packages/core/legoesm/parallel/sharded_dynamics.py-291-
packages/core/legoesm/parallel/sharded_dynamics.py-292-    return jax.tree.map(_gather, state)
packages/core/legoesm/parallel/sharded_dynamics.py-293-
--
packages/core/legoesm/parallel/sharded_dynamics.py-2027-
packages/core/legoesm/parallel/sharded_dynamics.py-2028-    from legoesm.core.precision import cast_pytree
packages/core/legoesm/parallel/sharded_dynamics.py-2029-    from legoesm.core.state import MPASHydrostaticState
packages/core/legoesm/parallel/sharded_dynamics.py:2030:    from legoesm.parallel.mesh import multiprocess_safe_device_put
packages/core/legoesm/parallel/sharded_dynamics.py-2031-    from legoesm.parallel.shard_map_compat import shard_map
packages/core/legoesm/parallel/sharded_dynamics.py-2032-    from legoesm.timestepping.dispatch import dispatch_integrator
packages/core/legoesm/parallel/sharded_dynamics.py-2033-    from legoesm.timestepping.integration import (
--
packages/core/legoesm/parallel/sharded_dynamics.py-2113-    # the jitted step as an ARGUMENT with P("device") shard_map in_specs
packages/core/legoesm/parallel/sharded_dynamics.py-2114-    # (multi-controller-safe: a sharded jit ARG is legal where a sharded
packages/core/legoesm/parallel/sharded_dynamics.py-2115-    # CLOSURE constant raises at trace time under jax.distributed;
packages/core/legoesm/parallel/sharded_dynamics.py:2116:    # ``multiprocess_safe_device_put`` builds the global array from each
packages/core/legoesm/parallel/sharded_dynamics.py-2117-    # process's local copy).  All leaves are arrays after the jnp.stack
packages/core/legoesm/parallel/sharded_dynamics.py-2118-    # in _build_voronoi_partition_infra (ints become (n_dev,) arrays).
packages/core/legoesm/parallel/sharded_dynamics.py-2119-    dev_sharding = dev_config.face_sharding  # P("device") on axis 0
packages/core/legoesm/parallel/sharded_dynamics.py-2120-    stacked_meshes = jax.tree.map(
packages/core/legoesm/parallel/sharded_dynamics.py:2121:        lambda x: multiprocess_safe_device_put(x, dev_sharding),
packages/core/legoesm/parallel/sharded_dynamics.py-2122-        stacked_meshes,
packages/core/legoesm/parallel/sharded_dynamics.py-2123-    )
packages/core/legoesm/parallel/sharded_dynamics.py-2124-
--
packages/core/legoesm/parallel/sharded_dynamics.py-2146-        # meshes above for why args, not closures.
packages/core/legoesm/parallel/sharded_dynamics.py-2147-        halo_args = tuple(
packages/core/legoesm/parallel/sharded_dynamics.py-2148-            (
packages/core/legoesm/parallel/sharded_dynamics.py:2149:                multiprocess_safe_device_put(
packages/core/legoesm/parallel/sharded_dynamics.py-2150-                    pp_sched['send_cell_idx'][r], dev_sharding),
packages/core/legoesm/parallel/sharded_dynamics.py:2151:                multiprocess_safe_device_put(
packages/core/legoesm/parallel/sharded_dynamics.py-2152-                    pp_sched['recv_cell_pos'][r], dev_sharding),
packages/core/legoesm/parallel/sharded_dynamics.py:2153:                multiprocess_safe_device_put(
packages/core/legoesm/parallel/sharded_dynamics.py-2154-                    pp_sched['send_edge_idx'][r], dev_sharding),
packages/core/legoesm/parallel/sharded_dynamics.py:2155:                multiprocess_safe_device_put(
packages/core/legoesm/parallel/sharded_dynamics.py-2156-                    pp_sched['recv_edge_pos'][r], dev_sharding),
packages/core/legoesm/parallel/sharded_dynamics.py-2157-            )
packages/core/legoesm/parallel/sharded_dynamics.py-2158-            for r in range(n_rounds)
--
packages/core/legoesm/parallel/sharded_dynamics.py-2183-    else:
packages/core/legoesm/parallel/sharded_dynamics.py-2184-        # ---- Legacy all-gather strategy: local gather indices ----
packages/core/legoesm/parallel/sharded_dynamics.py-2185-        halo_args = (
packages/core/legoesm/parallel/sharded_dynamics.py:2186:            multiprocess_safe_device_put(gather_cells, dev_sharding),
packages/core/legoesm/parallel/sharded_dynamics.py:2187:            multiprocess_safe_device_put(gather_edges, dev_sharding),
packages/core/legoesm/parallel/sharded_dynamics.py-2188-        )
packages/core/legoesm/parallel/sharded_dynamics.py-2189-
packages/core/legoesm/parallel/sharded_dynamics.py-2190-    # ------------------------------------------------------------------
--
packages/core/legoesm/parallel/sharded_dynamics.py-2312-        # multi-controller-safe because it is an argument, not a closure
packages/core/legoesm/parallel/sharded_dynamics.py-2313-        # constant.  Take a HOST copy first (codex M3c-1 MAJOR): the
packages/core/legoesm/parallel/sharded_dynamics.py-2314-        # caller's mesh may arrive REPLICATED (bench replicate_pytree),
packages/core/legoesm/parallel/sharded_dynamics.py:2315:        # and multiprocess_safe_device_put passes non-fully-addressable
packages/core/legoesm/parallel/sharded_dynamics.py-2316-        # arrays through UNCHANGED — a replicated leaf would silently
packages/core/legoesm/parallel/sharded_dynamics.py-2317-        # stay replicated under multi-controller.  A host array is
packages/core/legoesm/parallel/sharded_dynamics.py-2318-        # always fully addressable, so the P("device") shard is
packages/core/legoesm/parallel/sharded_dynamics.py-2319-        # guaranteed on both controllers.  total_area is a host float.
packages/core/legoesm/parallel/sharded_dynamics.py:2320:        _area_for_mass = multiprocess_safe_device_put(
packages/core/legoesm/parallel/sharded_dynamics.py-2321-            np.asarray(global_mesh.areaCell), dev_sharding)
packages/core/legoesm/parallel/sharded_dynamics.py-2322-        # fp64 area sum to match the fp64 mass-budget accumulator below
packages/core/legoesm/parallel/sharded_dynamics.py-2323-        # (mirrors make_voronoi_mpi_step; identical under x64).
--
packages/core/legoesm/parallel/sharded_dynamics.py-2575-    Single-DEVICE configs (``dev_config.mesh is None``, the
packages/core/legoesm/parallel/sharded_dynamics.py-2576-    ``create_voronoi_device_mesh(n_devices=1)`` shape) pass through
packages/core/legoesm/parallel/sharded_dynamics.py-2577-    unchanged.  Multi-device single-PROCESS configs take the plain
packages/core/legoesm/parallel/sharded_dynamics.py:2578:    ``device_put`` branch of ``replicate_leaf`` (cheap, no collective).
packages/core/legoesm/parallel/sharded_dynamics.py-2579-    """
packages/core/legoesm/parallel/sharded_dynamics.py-2580-    if dev_config.mesh is None or dev_config.replicated_sharding is None:
packages/core/legoesm/parallel/sharded_dynamics.py-2581-        return state
--
tests/unit/test_rce_metal_backend.py-139-
tests/unit/test_rce_metal_backend.py-140-
tests/unit/test_rce_metal_backend.py-141-def _put_on_accelerator(pytree, dev):
tests/unit/test_rce_metal_backend.py:142:    """device_put every leaf of pytree onto dev. Mirrors to_metal /
tests/unit/test_rce_metal_backend.py-143-    to_cpu pattern from legoesm.parallel.metal."""
tests/unit/test_rce_metal_backend.py-144-    return jax.tree_util.tree_map(
tests/unit/test_rce_metal_backend.py-145-        lambda x: (
tests/unit/test_rce_metal_backend.py:146:            jax.device_put(x, dev) if hasattr(x, "dtype") else x
tests/unit/test_rce_metal_backend.py-147-        ),
tests/unit/test_rce_metal_backend.py-148-        pytree,
tests/unit/test_rce_metal_backend.py-149-    )
--
tests/unit/test_rce_metal_backend.py-186-
tests/unit/test_rce_metal_backend.py-187-@metal_required
tests/unit/test_rce_metal_backend.py-188-def test_cwv_runs_on_metal():
tests/unit/test_rce_metal_backend.py:189:    """Codex iter-2: explicit device_put + output-device assertion."""
tests/unit/test_rce_metal_backend.py-190-    dev = _accelerator_device()
tests/unit/test_rce_metal_backend.py-191-    state, grid, hc, _ = _setup_f32()
tests/unit/test_rce_metal_backend.py-192-    acc_state = _put_on_accelerator(state, dev)
--
tests/unit/test_rce_metal_backend.py-246-    acc_hc = _put_on_accelerator(hc, dev)
tests/unit/test_rce_metal_backend.py-247-    new_state = apply_rce_surface_fluxes(
tests/unit/test_rce_metal_backend.py-248-        acc_state, acc_hc, dt=1.0,
tests/unit/test_rce_metal_backend.py:249:        T_sfc=jax.device_put(T_sfc, dev),
tests/unit/test_rce_metal_backend.py:250:        q_sfc=jax.device_put(q_sfc, dev),
tests/unit/test_rce_metal_backend.py:251:        wind_speed=jax.device_put(wspd, dev),
tests/unit/test_rce_metal_backend.py-252-    )
tests/unit/test_rce_metal_backend.py-253-    _assert_on_accelerator(new_state.theta_prime.data, dev)
tests/unit/test_rce_metal_backend.py-254-    assert bool(jnp.all(jnp.isfinite(new_state.theta_prime.data)))
--
tests/unit/test_rce_metal_backend.py-296-@metal_required
tests/unit/test_rce_metal_backend.py-297-def test_tracer_positivity_runs_on_metal():
tests/unit/test_rce_metal_backend.py-298-    dev = _accelerator_device()
tests/unit/test_rce_metal_backend.py:299:    q_neg = jax.device_put(
tests/unit/test_rce_metal_backend.py-300-        jnp.array([-1.0, 0.0, 2.0], dtype=jnp.float32), dev,
tests/unit/test_rce_metal_backend.py-301-    )
tests/unit/test_rce_metal_backend.py-302-    out = clip_positive(q_neg)
--
tests/unit/test_rce_metal_backend.py-313-@metal_required
tests/unit/test_rce_metal_backend.py-314-def test_cwv_cpu_vs_metal_agreement():
tests/unit/test_rce_metal_backend.py-315-    """Cross-backend agreement within float32 round-off (Codex iter-2:
tests/unit/test_rce_metal_backend.py:316:    explicit device_put + output-device assertion to prevent
tests/unit/test_rce_metal_backend.py-317-    false-pass on a CPU-default-with-visible-accelerator runner)."""
tests/unit/test_rce_metal_backend.py-318-    dev = _accelerator_device()
tests/unit/test_rce_metal_backend.py-319-    state, _, hc, _ = _setup_f32()
--
tests/unit/test_rce_metal_backend.py-334-@metal_required
tests/unit/test_rce_metal_backend.py-335-def test_surface_flux_cpu_vs_metal_agreement():
tests/unit/test_rce_metal_backend.py-336-    """Surface flux update must agree across backends within float32.
tests/unit/test_rce_metal_backend.py:337:    Codex iter-2: explicit device_put + output-device assertion."""
tests/unit/test_rce_metal_backend.py-338-    dev = _accelerator_device()
tests/unit/test_rce_metal_backend.py-339-    state, _, hc, _ = _setup_f32()
tests/unit/test_rce_metal_backend.py-340-    wspd = wind_speed_at_lowest_level_plane(state)
--
tests/unit/test_rce_metal_backend.py-342-    q_sfc = jnp.full((NY, NX), 0.022, dtype=jnp.float32)
tests/unit/test_rce_metal_backend.py-343-    acc_state = _put_on_accelerator(state, dev)
tests/unit/test_rce_metal_backend.py-344-    acc_hc = _put_on_accelerator(hc, dev)
tests/unit/test_rce_metal_backend.py:345:    acc_wspd = jax.device_put(wspd, dev)
tests/unit/test_rce_metal_backend.py:346:    acc_T = jax.device_put(T_sfc, dev)
tests/unit/test_rce_metal_backend.py:347:    acc_q = jax.device_put(q_sfc, dev)
tests/unit/test_rce_metal_backend.py-348-    acc_out = apply_rce_surface_fluxes(
tests/unit/test_rce_metal_backend.py-349-        acc_state, acc_hc, dt=1.0,
tests/unit/test_rce_metal_backend.py-350-        T_sfc=acc_T, q_sfc=acc_q, wind_speed=acc_wspd,
--
tests/unit/test_rce_metal_backend.py-379-    T_sfc = jnp.full((NY, NX), 300.0, dtype=jnp.float32)
tests/unit/test_rce_metal_backend.py-380-    q_sfc = jnp.full((NY, NX), 0.022, dtype=jnp.float32)
tests/unit/test_rce_metal_backend.py-381-    acc_state = _put_on_accelerator(state, dev)
tests/unit/test_rce_metal_backend.py:382:    acc_wspd = jax.device_put(wspd, dev)
tests/unit/test_rce_metal_backend.py:383:    acc_T = jax.device_put(T_sfc, dev)
tests/unit/test_rce_metal_backend.py:384:    acc_q = jax.device_put(q_sfc, dev)
tests/unit/test_rce_metal_backend.py-385-    fn = jax.jit(lambda s, T, q, w: apply_rce_surface_fluxes(
tests/unit/test_rce_metal_backend.py-386-        s, hc, dt=1.0, T_sfc=T, q_sfc=q, wind_speed=w,
tests/unit/test_rce_metal_backend.py-387-    ))
--
packages/core/legoesm/parallel/ensemble.py-495-
packages/core/legoesm/parallel/ensemble.py-496-    def shard_leaf(x):
packages/core/legoesm/parallel/ensemble.py-497-        if x.ndim == 0:
packages/core/legoesm/parallel/ensemble.py:498:            return jax.device_put(x, replicated)
packages/core/legoesm/parallel/ensemble.py:499:        return jax.device_put(x, sharding)
packages/core/legoesm/parallel/ensemble.py-500-
packages/core/legoesm/parallel/ensemble.py-501-    return jax.tree.map(shard_leaf, batched_state)
packages/core/legoesm/parallel/ensemble.py-502-
--
packages/core/legoesm/parallel/ensemble.py-519-    # state has 30+ leaves; the lookup is cheap individually but adds
packages/core/legoesm/parallel/ensemble.py-520-    # measurable overhead at each gather call.
packages/core/legoesm/parallel/ensemble.py-521-    dev0 = jax.devices()[0]
packages/core/legoesm/parallel/ensemble.py:522:    return jax.tree.map(lambda x: jax.device_put(x, dev0), sharded_state)
packages/core/legoesm/parallel/ensemble.py-523-
packages/core/legoesm/parallel/ensemble.py-524-
packages/core/legoesm/parallel/ensemble.py-525-# ============================================================================
--
packages/core/legoesm/parallel/latlon_spmd.py-270-    mesh — the gather primitive shared by the atm and ocean lat-band SPMD steps
packages/core/legoesm/parallel/latlon_spmd.py-271-    (``gather_state_atm_latlon`` / ``gather_state_latlon``).
packages/core/legoesm/parallel/latlon_spmd.py-272-
packages/core/legoesm/parallel/latlon_spmd.py:273:    Single-process: plain ``jax.device_put`` (the historical path, unchanged).
packages/core/legoesm/parallel/latlon_spmd.py-274-    Multi-controller (``jax.process_count() > 1``, route-B ``jax.distributed``):
packages/core/legoesm/parallel/latlon_spmd.py:275:    a top-level ``device_put`` cannot reshard an array whose shards live on
packages/core/legoesm/parallel/latlon_spmd.py-276-    other processes' devices, so the replication runs as a jit-compiled
packages/core/legoesm/parallel/latlon_spmd.py-277-    identity with replicated ``out_shardings`` — the supported cross-process
packages/core/legoesm/parallel/latlon_spmd.py-278-    collective path (every process executes the same program; XLA inserts the
--
packages/core/legoesm/parallel/latlon_spmd.py-289-    """
packages/core/legoesm/parallel/latlon_spmd.py-290-    if multiprocess:
packages/core/legoesm/parallel/latlon_spmd.py-291-        return jax.jit(lambda a: a, out_shardings=rep)(arr)
packages/core/legoesm/parallel/latlon_spmd.py:292:    return jax.device_put(arr, rep)
packages/core/legoesm/parallel/latlon_spmd.py-293-
packages/core/legoesm/parallel/latlon_spmd.py-294-
packages/core/legoesm/parallel/latlon_spmd.py-295-def shard_leaf(arr, sharding, *, multiprocess: bool):
--
packages/core/legoesm/parallel/latlon_spmd.py-297-    primitive symmetric to :func:`replicate_leaf`, shared by the atm and ocean
packages/core/legoesm/parallel/latlon_spmd.py:298:    lat-band SPMD steps (``shard_state_atm_latlon`` / ``shard_state_latlon``).
packages/core/legoesm/parallel/latlon_spmd.py-299-
packages/core/legoesm/parallel/latlon_spmd.py:300:    Single-process: plain ``jax.device_put`` (the historical path, unchanged and
packages/core/legoesm/parallel/latlon_spmd.py-301-    byte-identical).
packages/core/legoesm/parallel/latlon_spmd.py-302-
packages/core/legoesm/parallel/latlon_spmd.py-303-    Multi-controller (``jax.process_count() > 1``, route-B ``jax.distributed``,
packages/core/legoesm/parallel/latlon_spmd.py:304:    a mesh spanning processes): a top-level ``jax.device_put`` of the FULL global
packages/core/legoesm/parallel/latlon_spmd.py-305-    array to a cross-process ``NamedSharding`` cannot place shards on peer
packages/core/legoesm/parallel/latlon_spmd.py-306-    processes' devices, so XLA falls back to an all-gather that (a) transiently
packages/core/legoesm/parallel/latlon_spmd.py-307-    materialises a second global copy per process and (b) is the collective seen
packages/core/legoesm/parallel/latlon_spmd.py-308-    to crash under full-node CPU packing (issue #1100: 128 procs × global-state
packages/core/legoesm/parallel/latlon_spmd.py:309:    each, rc=137 OOM). Instead ``jax.make_array_from_callback`` invokes the
packages/core/legoesm/parallel/latlon_spmd.py-310-    callback ONCE PER ADDRESSABLE SHARD with that shard's global index, and each
packages/core/legoesm/parallel/latlon_spmd.py-311-    process reads only its own shards out of the global array it already holds —
packages/core/legoesm/parallel/latlon_spmd.py-312-    no all-gather, no transient global replica. Using the per-shard index
packages/core/legoesm/parallel/latlon_spmd.py-313-    callback (not an enclosing [min,max) span) makes it correct for ANY
packages/core/legoesm/parallel/latlon_spmd.py-314-    device→process placement, including a non-contiguous/interleaved mesh order.
packages/core/legoesm/parallel/latlon_spmd.py-315-
packages/core/legoesm/parallel/latlon_spmd.py:316:    NOT differentiable: ``make_array_from_callback`` is a host construction API,
packages/core/legoesm/parallel/latlon_spmd.py:317:    so (unlike the historical ``device_put``) a ``jax.grad``/``vjp`` cannot be
packages/core/legoesm/parallel/latlon_spmd.py-318-    taken THROUGH the multiprocess scatter. This is fine — the scatter is an
packages/core/legoesm/parallel/latlon_spmd.py-319-    init-time boundary (``scatter_to_local`` before the step loop, per the MPI
packages/core/legoesm/parallel/latlon_spmd.py-320-    pattern), never inside a differentiated loss; gradients w.r.t. params flow
--
packages/core/legoesm/parallel/latlon_spmd.py-328-        is explicit at every call site, mirroring :func:`replicate_leaf`).
packages/core/legoesm/parallel/latlon_spmd.py-329-    """
packages/core/legoesm/parallel/latlon_spmd.py-330-    if not multiprocess:
packages/core/legoesm/parallel/latlon_spmd.py:331:        return jax.device_put(arr, sharding)
packages/core/legoesm/parallel/latlon_spmd.py:332:    return jax.make_array_from_callback(arr.shape, sharding, lambda idx: arr[idx])
packages/core/legoesm/parallel/latlon_spmd.py-333-
packages/core/legoesm/parallel/latlon_spmd.py-334-
packages/core/legoesm/parallel/latlon_spmd.py-335-def _pole_fold(rows, negate: bool):
--
tests/unit/test_no_module_top_jax_alloc.py-26-  individual JAX numpy constructors.  No longer assumes the alias
tests/unit/test_no_module_top_jax_alloc.py-27-  is literally ``jnp``.
tests/unit/test_no_module_top_jax_alloc.py-28-  (Any of these at module top would hit the same eager-dispatch crash.)
tests/unit/test_no_module_top_jax_alloc.py:29:* ``jax.device_put(...)`` detected (also triggers eager dispatch).
tests/unit/test_no_module_top_jax_alloc.py-30-* Protected-module list now scans the entire ``src/legoesm/grids/``
tests/unit/test_no_module_top_jax_alloc.py-31-  subtree (per Codex MEDIUM — re-exports from ``__init__.py``
tests/unit/test_no_module_top_jax_alloc.py-32-  pull in the whole package early).
--
tests/unit/test_no_module_top_jax_alloc.py-70-
tests/unit/test_no_module_top_jax_alloc.py-71-# Top-level JAX functions that move data to device (also trigger
tests/unit/test_no_module_top_jax_alloc.py-72-# eager dispatch on the default platform).
tests/unit/test_no_module_top_jax_alloc.py:73:_JAX_DISPATCH_FUNCS = frozenset({"device_put"})
tests/unit/test_no_module_top_jax_alloc.py-74-
tests/unit/test_no_module_top_jax_alloc.py-75-
tests/unit/test_no_module_top_jax_alloc.py-76-def _collect_jax_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
--
tests/unit/test_no_module_top_jax_alloc.py-79-    Tracks BOTH:
tests/unit/test_no_module_top_jax_alloc.py-80-    * Module aliases pointing at ``jax.numpy`` (e.g., ``jnp``,
tests/unit/test_no_module_top_jax_alloc.py-81-      ``jax_np``, ``np2`` after ``import jax.numpy as np2``).
tests/unit/test_no_module_top_jax_alloc.py:82:    * Module aliases pointing at ``jax`` itself (for ``jax.device_put``
tests/unit/test_no_module_top_jax_alloc.py-83-      detection).
tests/unit/test_no_module_top_jax_alloc.py-84-
tests/unit/test_no_module_top_jax_alloc.py-85-    Does NOT track ``from jax.numpy import asarray`` — direct-name
--
tests/unit/test_no_module_top_jax_alloc.py-140-    Matches:
tests/unit/test_no_module_top_jax_alloc.py-141-    * ``<jnp_alias>.<array_ctor>(...)``  (e.g., ``jnp.asarray(...)``)
tests/unit/test_no_module_top_jax_alloc.py-142-    * ``<jax_alias>.numpy.<array_ctor>(...)``
tests/unit/test_no_module_top_jax_alloc.py:143:    * ``<jax_alias>.device_put(...)``
tests/unit/test_no_module_top_jax_alloc.py-144-    * ``<direct_ctor>(...)`` for ``from jax.numpy import <ctor>``
tests/unit/test_no_module_top_jax_alloc.py-145-    """
tests/unit/test_no_module_top_jax_alloc.py-146-    if not isinstance(node, ast.Call):
--
tests/unit/test_no_module_top_jax_alloc.py-181-    """Return ``(line, code_snippet)`` for each top-level JAX alloc.
tests/unit/test_no_module_top_jax_alloc.py-182-
tests/unit/test_no_module_top_jax_alloc.py-183-    Scans ``ast.Assign``, ``ast.AnnAssign``, AND bare ``ast.Expr``
tests/unit/test_no_module_top_jax_alloc.py:184:    statements at module scope (Codex LOW: a bare ``jax.device_put(...)``
tests/unit/test_no_module_top_jax_alloc.py-185-    at module top is a statement, not an assignment, and the initial
tests/unit/test_no_module_top_jax_alloc.py-186-    walker missed it).
tests/unit/test_no_module_top_jax_alloc.py-187-    """
--
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2519-    perms += list(tiled_diag_perms(kt))
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2520-
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2521-    sh = NamedSharding(mesh, P(*AXES))
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2522:    dummy = jax.device_put(jnp.zeros((6, kt, kt), dtype=jnp.float32), sh)
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2523-
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2524-    @partial(shard_map, mesh=mesh, in_specs=P(*AXES), out_specs=P(*AXES),
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2525-             check_vma=False)
--
packages/core/legoesm/grids/gaussian.py-495-        np_arr = np.asarray(array, dtype=dtype)
packages/core/legoesm/grids/gaussian.py-496-        if target_device is None:
packages/core/legoesm/grids/gaussian.py-497-            return jnp.array(np_arr, dtype=dtype)
packages/core/legoesm/grids/gaussian.py:498:        return jax.device_put(np_arr, target_device)
packages/core/legoesm/grids/gaussian.py-499-
packages/core/legoesm/grids/gaussian.py-500-    # Precompute weighted Legendre matrices (avoid recomputing every SH analysis)
packages/core/legoesm/grids/gaussian.py-501-    w_col = w_gauss[:, None]  # (n_lat, 1)
--
tests/unit/test_parallel.py-591-
tests/unit/test_parallel.py-592-
tests/unit/test_parallel.py-593-class TestMultiprocessSafeDevicePut:
tests/unit/test_parallel.py:594:    """multiprocess_safe_device_put (#693): identical to device_put single-
tests/unit/test_parallel.py:595:    process; routes through make_array_from_callback under >1 process so the
tests/unit/test_parallel.py:596:    cross-process bit-equality assert in device_put cannot trip on host-
tests/unit/test_parallel.py-597-    precomputed forcing (Levante 2-node cs_spmd receipt, job 26030677)."""
tests/unit/test_parallel.py-598-
tests/unit/test_parallel.py-599-    def _sharding(self):
tests/unit/test_parallel.py:600:        from legoesm.parallel.mesh import multiprocess_safe_device_put  # noqa: F401
tests/unit/test_parallel.py-601-        dev = jax.devices()[0]
tests/unit/test_parallel.py-602-        mesh = Mesh(np.array([dev]), ("face",))
tests/unit/test_parallel.py-603-        return NamedSharding(mesh, P())
tests/unit/test_parallel.py-604-
tests/unit/test_parallel.py:605:    def test_single_process_matches_device_put(self):
tests/unit/test_parallel.py:606:        from legoesm.parallel.mesh import multiprocess_safe_device_put
tests/unit/test_parallel.py-607-        x = jnp.arange(12.0).reshape(6, 2)
tests/unit/test_parallel.py-608-        s = self._sharding()
tests/unit/test_parallel.py:609:        out = multiprocess_safe_device_put(x, s)
tests/unit/test_parallel.py:610:        ref = jax.device_put(x, s)
tests/unit/test_parallel.py-611-        assert np.array_equal(np.asarray(out), np.asarray(ref))
tests/unit/test_parallel.py-612-        assert out.sharding == ref.sharding
tests/unit/test_parallel.py-613-
tests/unit/test_parallel.py-614-    def test_non_array_delegates(self):
tests/unit/test_parallel.py:615:        from legoesm.parallel.mesh import multiprocess_safe_device_put
tests/unit/test_parallel.py:616:        out = multiprocess_safe_device_put(3.5, self._sharding())
tests/unit/test_parallel.py-617-        assert float(out) == 3.5
tests/unit/test_parallel.py-618-
tests/unit/test_parallel.py-619-    def test_multiprocess_branch_uses_callback_and_preserves_values(self):
tests/unit/test_parallel.py:620:        from legoesm.parallel.mesh import multiprocess_safe_device_put
tests/unit/test_parallel.py-621-        x = jnp.arange(24.0).reshape(6, 4)
tests/unit/test_parallel.py-622-        s = self._sharding()
tests/unit/test_parallel.py-623-        with patch("jax.process_count", return_value=2):
tests/unit/test_parallel.py:624:            out = multiprocess_safe_device_put(x, s)
tests/unit/test_parallel.py-625-        # values intact and placed on the requested sharding, with NO
tests/unit/test_parallel.py-626-        # cross-process equality assertion involved
tests/unit/test_parallel.py-627-        assert np.array_equal(np.asarray(out), np.asarray(x))
--
packages/core/legoesm/runtime/config.py-125-    # 0. Multi-controller federation (spmd mode) ------------------------------
packages/core/legoesm/runtime/config.py-126-    # ``jax.distributed.initialize()`` must run before the BACKEND CLIENT
packages/core/legoesm/runtime/config.py-127-    # is instantiated — i.e. before the first ``jax.devices()`` /
packages/core/legoesm/runtime/config.py:128:    # ``device_put`` / trace — so every launched process joins one
packages/core/legoesm/runtime/config.py-129-    # program (the --cs-spmd bench contract).  Plain ``import jax`` does
packages/core/legoesm/runtime/config.py-130-    # NOT create the client, so module-level jax imports elsewhere are
packages/core/legoesm/runtime/config.py-131-    # fine; the hazards are device queries before this point.  Steps 1-4
--
tests/unit/test_wb_scale_lazy_loader_1286.py-66-
tests/unit/test_wb_scale_lazy_loader_1286.py-67-
tests/unit/test_wb_scale_lazy_loader_1286.py-68-# ---------------------------------------------------------------------------
tests/unit/test_wb_scale_lazy_loader_1286.py:69:# Fix A: _sample_to_host yields host numpy that round-trips through device_put
tests/unit/test_wb_scale_lazy_loader_1286.py-70-# ---------------------------------------------------------------------------
tests/unit/test_wb_scale_lazy_loader_1286.py-71-
tests/unit/test_wb_scale_lazy_loader_1286.py-72-def test_sample_to_host_is_numpy_and_roundtrips():
--
tests/unit/test_wb_scale_lazy_loader_1286.py-88-        assert isinstance(leaf, np.ndarray), type(leaf)
tests/unit/test_wb_scale_lazy_loader_1286.py-89-        assert not isinstance(leaf, jax.Array)
tests/unit/test_wb_scale_lazy_loader_1286.py-90-
tests/unit/test_wb_scale_lazy_loader_1286.py:91:    # device_put (what the training loop does per batch) restores the values.
tests/unit/test_wb_scale_lazy_loader_1286.py:92:    back = jax.device_put(host)
tests/unit/test_wb_scale_lazy_loader_1286.py-93-    for a, b in zip(jax.tree_util.tree_leaves(sample),
tests/unit/test_wb_scale_lazy_loader_1286.py-94-                    jax.tree_util.tree_leaves(back)):
tests/unit/test_wb_scale_lazy_loader_1286.py-95-        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
tests/unit/test_wb_scale_lazy_loader_1286.py-96-
tests/unit/test_wb_scale_lazy_loader_1286.py-97-
tests/unit/test_wb_scale_lazy_loader_1286.py:98:def test_loop_device_puts_host_samples():
tests/unit/test_wb_scale_lazy_loader_1286.py:99:    """The training loop must accept host-numpy samples (device_put per batch)
tests/unit/test_wb_scale_lazy_loader_1286.py-100-    and produce finite losses — a smoke of the fix-A consume path."""
tests/unit/test_wb_scale_lazy_loader_1286.py-101-    import jax
tests/unit/test_wb_scale_lazy_loader_1286.py-102-    import jax.numpy as jnp
--
scripts/run/train_weatherbench_scale.py-212-    # --- ERA5 IC/target/forcing samples, sharded across ranks ---
scripts/run/train_weatherbench_scale.py-213-    # #1286: the loader builds ONLY this rank's contiguous shard (fix B — never
scripts/run/train_weatherbench_scale.py-214-    # the full global list) and keeps it HOST-resident (fix A — the training
scripts/run/train_weatherbench_scale.py:215:    # loop device_puts one sample at a time).  The rank's slice is byte-for-byte
scripts/run/train_weatherbench_scale.py-216-    # the old ``shard_samples(build_all(), rank, nproc)`` partition, so gradient
scripts/run/train_weatherbench_scale.py-217-    # semantics are unchanged; we no longer materialize the global GPU-resident
scripts/run/train_weatherbench_scale.py-218-    # set that OOM'd at T106.
--
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-1-"""Multi-PROCESS jax.distributed equivalence VALIDATION for the lat-band SPMD ocean
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py:2:step — the multi-node correctness gate for ``make_sharded_ocean_step_global``.
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-3-
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-4-Runs STANDALONE (not pytest): the root ``tests/conftest.py`` initializes the XLA
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-5-backend (``ensure_metal_or_fallback`` -> ``jax.default_backend()``) at collection,
--
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-117-    """Run N SPMD steps over an ``n_dev``-band lat mesh (global devices), gather."""
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-118-    from legoesm.parallel.mesh import create_latlon_mesh
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-119-    from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py:120:        make_sharded_ocean_step_global,
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-121-    )
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-122-    dev = create_latlon_mesh(n_devices=n_dev)
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-123-    assert dev.mesh.devices.size == n_dev, dev.mesh.devices.size
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-124-    model._ensure_vertex_mask(state0)
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py:125:    step = make_sharded_ocean_step_global(model, dev.mesh)
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-126-    ss = state0
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-127-    for _ in range(N_STEPS):
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-128-        ss = step(ss, DT)
--
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-217-    # MULTI-PROCESS SPMD step: the lat mesh spans the GLOBAL device set (all 4
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-218-    # across both processes); the band halo + barotropic reductions cross the
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-219-    # process boundary via ppermute/psum (the "spmd" backend armed per-call inside
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py:220:    # make_sharded_ocean_step).  Exactly the run_omip_core2 --distributed step.
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-221-    ss = _spmd_step_result(model, state0, 4)
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-222-
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-223-    local_ok = True
--
scripts/run/run_omip_core2.py-3326-    p.add_argument("--n-gpus", type=int, default=1,
scripts/run/run_omip_core2.py-3327-                   help="Multi-GPU lat-band SPMD ocean step (latlon_bathy / tripole "
scripts/run/run_omip_core2.py-3328-                        "only): partition the ocean state by latitude band across N "
scripts/run/run_omip_core2.py:3329:                        "local devices via make_sharded_ocean_step_global. n_lat is "
scripts/run/run_omip_core2.py-3330-                        "padded with LAND rows at the SOUTH to a multiple of N (the "
scripts/run/run_omip_core2.py-3331-                        "tripole north fold stays at the north). The host post-step "
scripts/run/run_omip_core2.py-3332-                        "BCs (SSS restore / prognostic ice / geothermal / BBL / nudge) "
--
scripts/run/run_omip_core2.py-3348-                        "--distributed) is byte-unchanged.")
scripts/run/run_omip_core2.py-3349-    p.add_argument("--spmd-persistent-state", action="store_true",
scripts/run/run_omip_core2.py-3350-                   help="With --n-gpus > 1: keep the ocean state lat-band "
scripts/run/run_omip_core2.py:3351:                        "SHARDED across steps (make_sharded_ocean_step) instead "
scripts/run/run_omip_core2.py-3352-                        "of the global-in/global-out wrapper's full-state "
scripts/run/run_omip_core2.py-3353-                        "scatter+gather EVERY step (scaling-M2). Host post-step "
scripts/run/run_omip_core2.py-3354-                        "BCs run UNCHANGED: the leaf-wise host updates (SSS "
--
scripts/run/run_omip_core2.py-5541-                f"n_gpus ({args.n_gpus}) — the south-pad failed.")
scripts/run/run_omip_core2.py-5542-        from legoesm.parallel.mesh import create_latlon_mesh
scripts/run/run_omip_core2.py-5543-        from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip_core2.py:5544:            make_sharded_ocean_step_global,
scripts/run/run_omip_core2.py-5545-        )
scripts/run/run_omip_core2.py-5546-        # Prime the build-once vertex-mask cache from the concrete state BEFORE
scripts/run/run_omip_core2.py-5547-        # building the sharded step (the wrapper slices the primed global vmask
--
scripts/run/run_omip_core2.py-5563-            # helpers below manage (initial shard, snapshot/abort/final
scripts/run/run_omip_core2.py-5564-            # gathers, and the counted per-step gathers forced by host-global
scripts/run/run_omip_core2.py-5565-            # consumers).  t_sec is always None here (tide fail-fasts above).
scripts/run/run_omip_core2.py-5566-            from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip_core2.py-5567-                gather_state_latlon,
scripts/run/run_omip_core2.py:5568:                make_sharded_ocean_step,
scripts/run/run_omip_core2.py:5569:                shard_state_latlon,
scripts/run/run_omip_core2.py-5570-            )
scripts/run/run_omip_core2.py:5571:            _spmd_inner = make_sharded_ocean_step(model, _spmd_mesh)
scripts/run/run_omip_core2.py-5572-            _ocean_step = (lambda st, sf, fw, t_sec=None:
scripts/run/run_omip_core2.py-5573-                           _spmd_inner(st, dt, surface_forcing=sf,
scripts/run/run_omip_core2.py-5574-                                       freshwater=fw))
--
scripts/run/run_omip_core2.py-5586-                  f"(--spmd-persistent-state): full-state gathers only at "
scripts/run/run_omip_core2.py-5587-                  f"snapshot/abort/final + counted per-step forcings.")
scripts/run/run_omip_core2.py-5588-        else:
scripts/run/run_omip_core2.py:5589:            _spmd_step = make_sharded_ocean_step_global(model, _spmd_mesh)
scripts/run/run_omip_core2.py-5590-            # t_sec is always None here (tide-enabled fail-fasts above).
scripts/run/run_omip_core2.py-5591-            _ocean_step = (lambda st, sf, fw, t_sec=None:
scripts/run/run_omip_core2.py-5592-                           _spmd_step(st, dt, surface_forcing=sf,
--
scripts/run/run_omip.py-123-    # GMRediConfig / VisbeckConfig / TreguierConfig (#691/#724).
scripts/run/run_omip.py-124-    gm_redi: GMRediConfig = _DEFAULT_BATHY_GM_REDI
scripts/run/run_omip.py-125-    # Lat-band SPMD (single-controller multi-GPU) for the lat-lon restoring
scripts/run/run_omip.py:126:    # lane: wraps the loop's dynamics step in ``make_sharded_ocean_step``
scripts/run/run_omip.py-127-    # (the #751/#758-validated lane-D step).  0 devices = all local.
scripts/run/run_omip.py-128-    enable_latlon_spmd: bool = False
scripts/run/run_omip.py-129-    spmd_n_devices: int = 0
--
scripts/run/run_omip.py-707-    p.add_argument("--enable-latlon-spmd", action="store_true", default=False,
scripts/run/run_omip.py-708-                   help=(
scripts/run/run_omip.py-709-                       "Run the lat-lon lane's dynamics step lat-band-SPMD "
scripts/run/run_omip.py:710:                       "across the local devices (make_sharded_ocean_step — "
scripts/run/run_omip.py-711-                       "the validated multi-GPU ocean lane). Supports the "
scripts/run/run_omip.py-712-                       "restoring lane AND the JRA55 block-scan lanes "
scripts/run/run_omip.py-713-                       "(forcing stacks are lat-band-sharded; the in-scan "
--
scripts/run/run_omip.py-4811-
scripts/run/run_omip.py-4812-            from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip.py-4813-                gather_state_latlon,
scripts/run/run_omip.py:4814:                make_sharded_ocean_step,
scripts/run/run_omip.py:4815:                shard_forcing_stack_latlon,
scripts/run/run_omip.py:4816:                shard_state_latlon,
scripts/run/run_omip.py-4817-            )
--
scripts/run/run_omip.py-4820-            # state so the wrapper can build per-band masks host-side.
scripts/run/run_omip.py-4821-            model.prime_step_caches(state)
scripts/run/run_omip.py-4822-            _dev = create_latlon_mesh(n_devices=_nd)
scripts/run/run_omip.py:4823:            spmd_step = make_sharded_ocean_step(model, _dev.mesh)
scripts/run/run_omip.py-4824-            spmd_gather = partial(gather_state_latlon, mesh=_dev.mesh)
scripts/run/run_omip.py:4825:            state = shard_state_latlon(state, _dev.mesh)
scripts/run/run_omip.py-4826-            # Lay per-block forcing stacks out lat-band-sharded so the
--
scripts/validate/validate_tiled_fv3_sw_multinode.py-111-    def to_global(arr):
scripts/validate/validate_tiled_fv3_sw_multinode.py-112-        # Build the face-sharded/tile-replicated global array from the
scripts/validate/validate_tiled_fv3_sw_multinode.py-113-        # process-local (full, replicated) data — each device fills its shard.
scripts/validate/validate_tiled_fv3_sw_multinode.py:114:        return jax.make_array_from_callback(
scripts/validate/validate_tiled_fv3_sw_multinode.py-115-            arr.shape, sh, lambda idx: arr[idx])
scripts/validate/validate_tiled_fv3_sw_multinode.py-116-
scripts/validate/validate_tiled_fv3_sw_multinode.py-117-    try:
--
packages/coupler/legoesm/driver/compiled_segments.py-917-        if (_flat_face is not None and leaf.ndim >= 1
packages/coupler/legoesm/driver/compiled_segments.py-918-                and leaf.shape[0] != 6 and leaf.shape[0] > 0
packages/coupler/legoesm/driver/compiled_segments.py-919-                and leaf.shape[0] % 6 == 0):
packages/coupler/legoesm/driver/compiled_segments.py:920:            # multiprocess_safe_device_put, NOT jax.device_put: under
packages/coupler/legoesm/driver/compiled_segments.py-921-            # multi-controller SPMD the cross-process bit-equality assert
packages/coupler/legoesm/driver/compiled_segments.py:922:            # in device_put trips on host-precomputed forcing (#693,
packages/coupler/legoesm/driver/compiled_segments.py-923-            # Levante 2-node receipt job 26030677).
packages/coupler/legoesm/driver/compiled_segments.py:924:            from legoesm.parallel.mesh import multiprocess_safe_device_put
packages/coupler/legoesm/driver/compiled_segments.py:925:            updates[name] = multiprocess_safe_device_put(leaf, _flat_face)
packages/coupler/legoesm/driver/compiled_segments.py-926-        else:
packages/coupler/legoesm/driver/compiled_segments.py-927-            updates[name] = shard_pytree(leaf, device_config)
packages/coupler/legoesm/driver/compiled_segments.py-928-    return forcing._replace(**updates) if updates else forcing
--
tests/unit/test_neural_gcm_spectral.py-1213-
tests/unit/test_neural_gcm_spectral.py-1214-
tests/unit/test_neural_gcm_spectral.py-1215-def test_stage_tree_moves_arrays_skips_non_arrays():
tests/unit/test_neural_gcm_spectral.py:1216:    """``_stage_tree`` device_puts array leaves and leaves None / non-array leaves
tests/unit/test_neural_gcm_spectral.py-1217-    untouched, so a (ics, tgts, forcings=None) chunk stages without choking."""
tests/unit/test_neural_gcm_spectral.py-1218-    from legoesm.training.neural_gcm_spectral import _stage_tree
tests/unit/test_neural_gcm_spectral.py-1219-
--
tests/unit/test_neural_gcm_spectral.py-1311-        """The non-chunked loop with host_staged=True must produce weights
tests/unit/test_neural_gcm_spectral.py-1312-        BIT-IDENTICAL to host_staged=False on the same data — staging is
tests/unit/test_neural_gcm_spectral.py-1313-        placement-only, never numerics. (On the CPU test backend the staging
tests/unit/test_neural_gcm_spectral.py:1314:        device_put is a same-device move; the test guards the plumbing:
tests/unit/test_neural_gcm_spectral.py-1315-        tuple-target handling, forcings=None, rebind lifetime.)"""
tests/unit/test_neural_gcm_spectral.py-1316-        from legoesm.training.neural_gcm_spectral import (
tests/unit/test_neural_gcm_spectral.py-1317-            _train_spectral_loop, make_sfno_spectral_physics,
--
packages/coupler/legoesm/driver/model_driver.py-9429-                # carry (the lat-band lane's contract — _run_compiled
packages/coupler/legoesm/driver/model_driver.py-9430-                # re-packs from self.* every segment; codex).  The
packages/coupler/legoesm/driver/model_driver.py-9431-                # driver-visible fields are grid-shaped carry leaves
packages/coupler/legoesm/driver/model_driver.py:9432:                # (exact tile partition), so a per-leaf device_put onto
packages/coupler/legoesm/driver/model_driver.py-9433-                # the carry's existing sharding suffices; a no-op
packages/coupler/legoesm/driver/model_driver.py-9434-                # resharding when the callback does not mutate.  The
packages/coupler/legoesm/driver/model_driver.py-9435-                # threaded physics carry (held radiation / tke /
packages/coupler/legoesm/driver/model_driver.py-9436-                # conv_prog) keeps its sharded values.
packages/coupler/legoesm/driver/model_driver.py-9437-                _fold = dict(
packages/coupler/legoesm/driver/model_driver.py:9438:                    u=jax.device_put(self.state.u.data, carry.u.sharding),
packages/coupler/legoesm/driver/model_driver.py:9439:                    v=jax.device_put(self.state.v.data, carry.v.sharding),
packages/coupler/legoesm/driver/model_driver.py:9440:                    T=jax.device_put(self.state.T.data, carry.T.sharding),
packages/coupler/legoesm/driver/model_driver.py:9441:                    p_s=jax.device_put(self.state.p_s.data,
packages/coupler/legoesm/driver/model_driver.py-9442-                                       carry.p_s.sharding),
packages/coupler/legoesm/driver/model_driver.py:9443:                    q_v=jax.device_put(self.q_v, carry.q_v.sharding),
packages/coupler/legoesm/driver/model_driver.py:9444:                    q_c=jax.device_put(self.q_c, carry.q_c.sharding),
packages/coupler/legoesm/driver/model_driver.py:9445:                    q_r=jax.device_put(self.q_r, carry.q_r.sharding))
packages/coupler/legoesm/driver/model_driver.py-9446-                if isinstance(self.tracers, dict):
packages/coupler/legoesm/driver/model_driver.py-9447-                    for _nm in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
packages/coupler/legoesm/driver/model_driver.py-9448-                        _cv = getattr(carry, _nm)
packages/coupler/legoesm/driver/model_driver.py-9449-                        _sv = self.tracers.get(_nm)
packages/coupler/legoesm/driver/model_driver.py-9450-                        if _cv is not None and _sv is not None:
packages/coupler/legoesm/driver/model_driver.py:9451:                            _fold[_nm] = jax.device_put(_sv, _cv.sharding)
packages/coupler/legoesm/driver/model_driver.py-9452-                carry = carry._replace(**_fold)
packages/coupler/legoesm/driver/model_driver.py-9453-        logger.info("operator-split tiled cube: %s (%.1fs)", status,
packages/coupler/legoesm/driver/model_driver.py-9454-                    time.time() - t0)
--
packages/coupler/legoesm/driver/model_driver.py-9744-        # Route-B multicontroller (jax.distributed cross-process NCCL): the mesh
packages/coupler/legoesm/driver/model_driver.py-9745-        # spans devices across processes. ``_mp`` gates the cross-process gather
packages/coupler/legoesm/driver/model_driver.py-9746-        # (replicate_leaf's jit-identity all-gather vs the single-process
packages/coupler/legoesm/driver/model_driver.py:9747:        # device_put) and the rank-0 log gating; both are no-ops when
packages/coupler/legoesm/driver/model_driver.py-9748-        # process_count()==1 (single-controller / serial), so never-regress.
packages/coupler/legoesm/driver/model_driver.py-9749-        _mp = jax.process_count() > 1
packages/coupler/legoesm/driver/model_driver.py-9750-        _io_rank = jax.process_index() == 0
--
packages/coupler/legoesm/driver/model_driver.py-9943-
packages/coupler/legoesm/driver/model_driver.py-9944-            # Gather the sharded carry to a replicated cell-centered copy for
packages/coupler/legoesm/driver/model_driver.py-9945-            # the callback + NaN guard.  ``replicate_leaf`` is the shared gather
packages/coupler/legoesm/driver/model_driver.py:9946:            # primitive: single-process -> jax.device_put (byte-identical);
packages/coupler/legoesm/driver/model_driver.py-9947-            # route-B (``_mp``) -> a jit-identity with replicated out_shardings
packages/coupler/legoesm/driver/model_driver.py:9948:            # (the supported cross-process all-gather — a top-level device_put
packages/coupler/legoesm/driver/model_driver.py-9949-            # cannot reshard shards living on other processes' devices).
packages/coupler/legoesm/driver/model_driver.py-9950-            carry_full = jax.tree.map(
packages/coupler/legoesm/driver/model_driver.py-9951-                lambda x: replicate_leaf(x, rep, multiprocess=_mp), carry)
--
packages/coupler/legoesm/driver/model_driver.py-9991-                # not mutate.  The non-driver carry fields (held radiation, tke,
packages/coupler/legoesm/driver/model_driver.py-9992-                # conv_prog) keep their threaded sharded values.
packages/coupler/legoesm/driver/model_driver.py-9993-                _fold = dict(
packages/coupler/legoesm/driver/model_driver.py:9994:                    u=jax.device_put(self.state.u.data, carry.u.sharding),
packages/coupler/legoesm/driver/model_driver.py:9995:                    v=jax.device_put(self.state.v.data, carry.v.sharding),
packages/coupler/legoesm/driver/model_driver.py:9996:                    T=jax.device_put(self.state.T.data, carry.T.sharding),
packages/coupler/legoesm/driver/model_driver.py:9997:                    p_s=jax.device_put(self.state.p_s.data, carry.p_s.sharding),
packages/coupler/legoesm/driver/model_driver.py:9998:                    q_v=jax.device_put(self.q_v, carry.q_v.sharding),
packages/coupler/legoesm/driver/model_driver.py:9999:                    q_c=jax.device_put(self.q_c, carry.q_c.sharding),
packages/coupler/legoesm/driver/model_driver.py:10000:                    q_r=jax.device_put(self.q_r, carry.q_r.sharding))
packages/coupler/legoesm/driver/model_driver.py-10001-                # Double-moment tracers: a DA/coupling callback may nudge
packages/coupler/legoesm/driver/model_driver.py-10002-                # self.tracers[q_i…N_i]; fold those back too (same repack-from-self
packages/coupler/legoesm/driver/model_driver.py-10003-                # _run_compiled does), else a double-moment run diverges after a
--
packages/coupler/legoesm/driver/model_driver.py-10007-                    for _nm in ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i"):
packages/coupler/legoesm/driver/model_driver.py-10008-                        _cv, _sv = getattr(carry, _nm), self.tracers.get(_nm)
packages/coupler/legoesm/driver/model_driver.py-10009-                        if _cv is not None and _sv is not None:
packages/coupler/legoesm/driver/model_driver.py:10010:                            _fold[_nm] = jax.device_put(_sv, _cv.sharding)
packages/coupler/legoesm/driver/model_driver.py-10011-                carry = carry._replace(**_fold)
packages/coupler/legoesm/driver/model_driver.py-10012-
packages/coupler/legoesm/driver/model_driver.py-10013-        if _io_rank:
--
packages/coupler/legoesm/driver/sharded_operator_split_step.py-134-    vl = _carry_valid_leads(carry)
packages/coupler/legoesm/driver/sharded_operator_split_step.py-135-    return SegmentCarry(**{
packages/coupler/legoesm/driver/sharded_operator_split_step.py-136-        name: (None if (v := getattr(carry, name)) is None
packages/coupler/legoesm/driver/sharded_operator_split_step.py:137:               else jax.device_put(v, NamedSharding(
packages/coupler/legoesm/driver/sharded_operator_split_step.py-138-                   mesh, _carry_leaf_spec(name, v, n_dev, vl))))
packages/coupler/legoesm/driver/sharded_operator_split_step.py-139-        for name in SegmentCarry._fields
packages/coupler/legoesm/driver/sharded_operator_split_step.py-140-    })
--
packages/coupler/legoesm/driver/sharded_operator_split_step.py-146-    n_dev = mesh.devices.size
packages/coupler/legoesm/driver/sharded_operator_split_step.py-147-    return SegmentForcing(**{
packages/coupler/legoesm/driver/sharded_operator_split_step.py-148-        name: (None if (v := getattr(forcing, name)) is None
packages/coupler/legoesm/driver/sharded_operator_split_step.py:149:               else jax.device_put(v, NamedSharding(
packages/coupler/legoesm/driver/sharded_operator_split_step.py-150-                   mesh, _forcing_leaf_spec(name, v, n_dev))))
packages/coupler/legoesm/driver/sharded_operator_split_step.py-151-        for name in SegmentForcing._fields
packages/coupler/legoesm/driver/sharded_operator_split_step.py-152-    })
--
packages/coupler/legoesm/driver/sharded_operator_split_step.py-280-    afn = atm_grid_array_field_names(template)
packages/coupler/legoesm/driver/sharded_operator_split_step.py-281-    rep = NamedSharding(mesh, P())
packages/coupler/legoesm/driver/sharded_operator_split_step.py-282-    stacks = {
packages/coupler/legoesm/driver/sharded_operator_split_step.py:283:        name: jax.device_put(
packages/coupler/legoesm/driver/sharded_operator_split_step.py-284-            jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids], 0),
packages/coupler/legoesm/driver/sharded_operator_split_step.py-285-            rep)
packages/coupler/legoesm/driver/sharded_operator_split_step.py-286-        for name in afn
--
packages/coupler/legoesm/driver/sharded_operator_split_step.py-291-    # passes None -> _step_cgrid_impl resolves self._polar_mask (also None).
packages/coupler/legoesm/driver/sharded_operator_split_step.py-292-    nl = int(grid.n_lat) // n_dev
packages/coupler/legoesm/driver/sharded_operator_split_step.py-293-    if model._polar_mask is not None:
packages/coupler/legoesm/driver/sharded_operator_split_step.py:294:        stacks["__polar_mask"] = jax.device_put(jnp.stack(
packages/coupler/legoesm/driver/sharded_operator_split_step.py-295-            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(n_dev)],
packages/coupler/legoesm/driver/sharded_operator_split_step.py-296-            0), rep)
packages/coupler/legoesm/driver/sharded_operator_split_step.py:297:        stacks["__polar_mask_v"] = jax.device_put(jnp.stack(
packages/coupler/legoesm/driver/sharded_operator_split_step.py-298-            [model._polar_mask_v[r * nl:r * nl + nl + 1] for r in range(n_dev)],
packages/coupler/legoesm/driver/sharded_operator_split_step.py-299-            0), rep)
packages/coupler/legoesm/driver/sharded_operator_split_step.py-300-    _cache = {}
--
packages/coupler/legoesm/driver/tiled_operator_split_step.py-213-    packed = pack_carry_tiled(carry, n)
packages/coupler/legoesm/driver/tiled_operator_split_step.py-214-    return SegmentCarry(**{
packages/coupler/legoesm/driver/tiled_operator_split_step.py-215-        name: (None if (v := getattr(packed, name)) is None
packages/coupler/legoesm/driver/tiled_operator_split_step.py:216:               else jax.device_put(
packages/coupler/legoesm/driver/tiled_operator_split_step.py-217-                   v, NamedSharding(mesh, _tile_carry_spec(name, v, n))))
packages/coupler/legoesm/driver/tiled_operator_split_step.py-218-        for name in SegmentCarry._fields
packages/coupler/legoesm/driver/tiled_operator_split_step.py-219-    })
--
tests/unit/test_column_shard.py-185-
tests/unit/test_column_shard.py-186-        # Sharded path.  On a single test device the mesh is just
tests/unit/test_column_shard.py-187-        # ``n_dev=1`` but the helper API still routes through
tests/unit/test_column_shard.py:188:        # ``device_put`` so the test catches any sharding-spec bugs
tests/unit/test_column_shard.py-189-        # (wrong axis name, mis-shaped P, etc.).
tests/unit/test_column_shard.py-190-        mesh = create_column_mesh(n_devices=1)
tests/unit/test_column_shard.py-191-        n_dev = mesh.shape["col"]
--
tests/unit/test_scale_metal.py-196-        cpu = jax.devices("cpu")[0]
tests/unit/test_scale_metal.py-197-        # Create via numpy to avoid the default device (may be mps, no f64)
tests/unit/test_scale_metal.py-198-        import numpy as _np
tests/unit/test_scale_metal.py:199:        x = jax.device_put(_np.float64(1.0), cpu)
tests/unit/test_scale_metal.py-200-        assert x.dtype == jnp.float64
tests/unit/test_scale_metal.py-201-
tests/unit/test_scale_metal.py-202-    def test_cpu_supports_complex128(self):
--
tests/unit/test_scale_metal.py-205-            pytest.skip("JAX_ENABLE_X64 not set")
tests/unit/test_scale_metal.py-206-        cpu = jax.devices("cpu")[0]
tests/unit/test_scale_metal.py-207-        import numpy as _np
tests/unit/test_scale_metal.py:208:        x = jax.device_put(_np.complex128(1.0 + 2.0j), cpu)
tests/unit/test_scale_metal.py-209-        assert x.dtype == jnp.complex128
--
tests/atmosphere/hydrostatic/unit/test_radiation.py-40-from legoesm.runtime.backend import get_backend
tests/atmosphere/hydrostatic/unit/test_radiation.py-41-
tests/atmosphere/hydrostatic/unit/test_radiation.py-42-
tests/atmosphere/hydrostatic/unit/test_radiation.py:43:# Skip marker for tests that exercise jax.device_put with a shard Mesh.
tests/atmosphere/hydrostatic/unit/test_radiation.py-44-# The Apple GPU backend ``mps`` (jax-mps / MLX) exposes a single device, so
tests/atmosphere/hydrostatic/unit/test_radiation.py-45-# ``batched_copy_array_to_devices_with_sharding`` cannot build a multi-device
tests/atmosphere/hydrostatic/unit/test_radiation.py-46-# shard Mesh there.  Skip on that backend; the tests still run on CI
--
tests/atmosphere/hydrostatic/unit/test_radiation.py-48-_skip_if_metal_broken = pytest.mark.skipif(
tests/atmosphere/hydrostatic/unit/test_radiation.py-49-    get_backend() == "mps",
tests/atmosphere/hydrostatic/unit/test_radiation.py-50-    reason=(
tests/atmosphere/hydrostatic/unit/test_radiation.py:51:        "Apple GPU (mps) is single-device: device_put with a shard Mesh is "
tests/atmosphere/hydrostatic/unit/test_radiation.py-52-        "unsupported.  CI Linux/CUDA runs this."
tests/atmosphere/hydrostatic/unit/test_radiation.py-53-    ),
tests/atmosphere/hydrostatic/unit/test_radiation.py-54-)
--
tests/unit/test_cdgrid.py-314-        if not getattr(jax.config, 'x64_enabled', False):
tests/unit/test_cdgrid.py-315-            self.skipTest("x64 not enabled")
tests/unit/test_cdgrid.py-316-        try:
tests/unit/test_cdgrid.py:317:            jax.device_put(jnp.array(1.0, dtype=jnp.float64))
tests/unit/test_cdgrid.py-318-        except Exception:
tests/unit/test_cdgrid.py-319-            self.skipTest("backend does not support float64")
tests/unit/test_cdgrid.py-320-        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
--
packages/ml/legoesm/training/scale_build.py-137-
packages/ml/legoesm/training/scale_build.py-138-    #1286 fix A: the per-run ``samples`` list is kept host-resident so the whole
packages/ml/legoesm/training/scale_build.py-139-    training set is NOT parked in device memory; the training loop
packages/ml/legoesm/training/scale_build.py:140:    ``device_put``s one sample at a time.  ``jax.device_get`` converts every
packages/ml/legoesm/training/scale_build.py-141-    device-array leaf to numpy and passes non-array leaves through unchanged;
packages/ml/legoesm/training/scale_build.py-142-    the build's transient device allocation is freed once this returns.
packages/ml/legoesm/training/scale_build.py-143-    """
--
packages/ml/legoesm/training/scale_build.py-534-
packages/ml/legoesm/training/scale_build.py-535-    #1286: with ``nproc > 1`` this builds ONLY ``rank``'s contiguous shard (fix
packages/ml/legoesm/training/scale_build.py-536-    B — no global materialization); with ``host_resident`` each built sample is
packages/ml/legoesm/training/scale_build.py:537:    moved off-device to host numpy (fix A — the loop ``device_put``s per batch).
packages/ml/legoesm/training/scale_build.py-538-    """
packages/ml/legoesm/training/scale_build.py-539-    import jax.numpy as jnp
packages/ml/legoesm/training/scale_build.py-540-
--
packages/ml/legoesm/training/neural_gcm_spectral.py-1670-    transfer them per call anyway, so staging is not a crash guard. It IS
packages/ml/legoesm/training/neural_gcm_spectral.py-1671-    the explicit-placement hygiene: it makes each sample's single H2D copy
packages/ml/legoesm/training/neural_gcm_spectral.py-1672-    visible at the call site, keeps behavior identical if the dataset ever
packages/ml/legoesm/training/neural_gcm_spectral.py:1673:    arrives COMMITTED (e.g. an explicit ``device_put(cpu)`` producer), and
packages/ml/legoesm/training/neural_gcm_spectral.py-1674-    bounds peak device footprint to one sample. Wraps the #985
packages/ml/legoesm/training/neural_gcm_spectral.py-1675-    ``_stage_tree``. Uses ``jax.local_devices()`` (not ``jax.devices()``)
packages/ml/legoesm/training/neural_gcm_spectral.py-1676-    so a multi-process runtime never targets a non-addressable device.
--
packages/ml/legoesm/training/neural_gcm_spectral.py-3090-
packages/ml/legoesm/training/neural_gcm_spectral.py-3091-
packages/ml/legoesm/training/neural_gcm_spectral.py-3092-def _stage_tree(tree, device):
packages/ml/legoesm/training/neural_gcm_spectral.py:3093:    """``device_put`` only the ARRAY leaves of ``tree`` onto ``device``, leaving
packages/ml/legoesm/training/neural_gcm_spectral.py-3094-    None / metadata leaves untouched.
packages/ml/legoesm/training/neural_gcm_spectral.py-3095-
packages/ml/legoesm/training/neural_gcm_spectral.py-3096-    A chunk is ``(ic_states, targets, forcings)`` of pytrees whose leaves are
packages/ml/legoesm/training/neural_gcm_spectral.py-3097-    mostly arrays but not exclusively (``forcings`` may be ``None``); a bare
packages/ml/legoesm/training/neural_gcm_spectral.py:3098:    ``jax.device_put(tree, device)`` chokes on a non-array leaf, so filter to
packages/ml/legoesm/training/neural_gcm_spectral.py-3099-    arrays via ``eqx.is_array``. Used to host-stage prefetched chunks (#985).
packages/ml/legoesm/training/neural_gcm_spectral.py-3100-    """
packages/ml/legoesm/training/neural_gcm_spectral.py-3101-    return jax.tree_util.tree_map(
packages/ml/legoesm/training/neural_gcm_spectral.py:3102:        lambda x: jax.device_put(x, device) if eqx.is_array(x) else x, tree
packages/ml/legoesm/training/neural_gcm_spectral.py-3103-    )
packages/ml/legoesm/training/neural_gcm_spectral.py-3104-
packages/ml/legoesm/training/neural_gcm_spectral.py-3105-
--
packages/ml/legoesm/training/neural_gcm_spectral.py-3139-    # below -- instead of building it on the GPU and copying it down. That
packages/ml/legoesm/training/neural_gcm_spectral.py-3140-    # distinction is the fix: a T106 4xA100 run OOM'd because the prefetched chunk
packages/ml/legoesm/training/neural_gcm_spectral.py-3141-    # rode GPU memory during the current chunk's training peak, and a post-hoc
packages/ml/legoesm/training/neural_gcm_spectral.py:3142:    # device_put(cpu) does NOT help (the arrays are constructed on the GPU first).
packages/ml/legoesm/training/neural_gcm_spectral.py-3143-    # ``jax.default_device`` is thread-local, so the producer thread builds on the
packages/ml/legoesm/training/neural_gcm_spectral.py-3144-    # CPU while the main thread trains on the GPU. Each SAMPLE is then moved onto
packages/ml/legoesm/training/neural_gcm_spectral.py-3145-    # the compute device just before its step (the training loop), so a prefetched
--
packages/ml/legoesm/training/neural_gcm_spectral.py-3186-        yield from (_prefetch_iter(_serial()) if prefetch else _serial())
packages/ml/legoesm/training/neural_gcm_spectral.py-3187-
packages/ml/legoesm/training/neural_gcm_spectral.py-3188-    # True when chunks are built host-staged (prefetch + CPU backend): the
packages/ml/legoesm/training/neural_gcm_spectral.py:3189:    # training loop must then device_put each SAMPLE onto the compute device at
packages/ml/legoesm/training/neural_gcm_spectral.py-3190-    # the point of use (#985).
packages/ml/legoesm/training/neural_gcm_spectral.py-3191-    _chunks.host_staged = _host_dev is not None
packages/ml/legoesm/training/neural_gcm_spectral.py-3192-    # Number of chunks per epoch (constant): the mid-epoch checkpoint reads
--
packages/ml/legoesm/training/data_parallel.py-261-            # sample onto the device here and let it free at the next iteration.
packages/ml/legoesm/training/data_parallel.py-262-            # A no-op (cheap) when the sample is already device-resident (the
packages/ml/legoesm/training/data_parallel.py-263-            # legacy eager path), so this is safe for both.
packages/ml/legoesm/training/data_parallel.py:264:            sample = jax.device_put(sample)
packages/ml/legoesm/training/data_parallel.py-265-            params, opt_state, loss = mpi_data_parallel_train_step(
packages/ml/legoesm/training/data_parallel.py-266-                loss_fn, params, opt_state, optimizer, sample, num_processes, comm=comm)
packages/ml/legoesm/training/data_parallel.py-267-            losses.append(float(loss))
--
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-2-
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-3-Exercises the builder + the nd=1 single-device main() end-to-end on a tiny
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-4-grid (JSONL record shape) and the argument guards. The multi-device timing
tests/unit/test_bench_ocean_latlon_spmd_scaling.py:5:path reuses make_sharded_ocean_step, whose correctness gate is
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-6-tests/parallel/test_latlon_ocean_spmd_step.py; the --multicontroller path is
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-7-cluster-gated (see the script docstring).
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-8-"""
--
tests/unit/test_sharded_dynamics.py-585-        serial = np.asarray(jax.jit(fn)(*args))
tests/unit/test_sharded_dynamics.py-586-        for nd in (2, 3):
tests/unit/test_sharded_dynamics.py-587-            dev = create_device_mesh(n_devices=nd)
tests/unit/test_sharded_dynamics.py:588:            args_sh = tuple(jax.device_put(a, dev.face_sharding) for a in args)
tests/unit/test_sharded_dynamics.py-589-            sharded = np.asarray(jax.jit(fn)(*args_sh))
tests/unit/test_sharded_dynamics.py-590-            # BIT-exact: a global conserved integral is decomposition-independent.
tests/unit/test_sharded_dynamics.py-591-            assert np.array_equal(sharded, serial), (
--
tests/unit/test_scale_sharded_dynamics.py-128-        """shard_state should work inside JIT."""
tests/unit/test_scale_sharded_dynamics.py-129-        config = create_device_mesh(n_devices=1)
tests/unit/test_scale_sharded_dynamics.py-130-        state = {"field": jnp.ones((6, N, N), dtype=jnp.float64)}
tests/unit/test_scale_sharded_dynamics.py:131:        # shard_state itself uses jax.device_put, which is a host-side
tests/unit/test_scale_sharded_dynamics.py-132-        # operation — but the result should be a valid JAX array
tests/unit/test_scale_sharded_dynamics.py-133-        out = shard_state(state, config)
tests/unit/test_scale_sharded_dynamics.py-134-        assert isinstance(out["field"], jax.Array)
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2022-        tendency_fn = self._make_tendency_fn(physics_fn)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2023-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2024-        if self._use_cpu_for_spectral:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:2025:            state_cpu = jax.device_put(state, self._cpu_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2026-            result_cpu = self._do_step(state_cpu, dt, tendency_fn,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2027-                                       target_mass=target_mass)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:2028:            return jax.device_put(result_cpu, self._default_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2029-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2030-        return self._do_step(state, dt, tendency_fn, target_mass=target_mass)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2031-
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2059-        tendency_fn = self._make_tendency_fn(physics_fn, forcing_data)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2060-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2061-        if self._use_cpu_for_spectral:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:2062:            state_cpu = jax.device_put(state, self._cpu_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2063-            result_cpu = self._do_step(state_cpu, dt, tendency_fn,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2064-                                       target_mass=target_mass)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:2065:            return jax.device_put(result_cpu, self._default_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2066-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2067-        return self._do_step(state, dt, tendency_fn, target_mass=target_mass)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2068-
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2143-                "leapfrog integrators; use per-step step() (which runs the "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2144-                "leapfrog state machine) or an SSP/SI time_integrator."
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2145-            )
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:2146:        state_cpu = jax.device_put(state, self._cpu_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2147-        trajectory_cpu = [state_cpu]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2148-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2149-        # Refresh dt-dependent matrices on the host once before stepping.
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2162-                trajectory_cpu.append(state_cpu)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2163-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2164-        # Transfer back to Metal
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:2165:        state_out = jax.device_put(state_cpu, self._default_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2166-        trajectory_out = [
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py:2167:            jax.device_put(s, self._default_device) for s in trajectory_cpu
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2168-        ]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2169-        return state_out, trajectory_out
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py-2170-
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-367-        integrator = self.config.time_integrator
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-368-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-369-        if self._use_cpu_for_spectral:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:370:            state_cpu = jax.device_put(state, self._cpu_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-371-            result_cpu = dispatch_integrator(state_cpu, tendency_fn, dt, integrator)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-372-            result_cpu = self._apply_filter(result_cpu)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:373:            return jax.device_put(result_cpu, self._default_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-374-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-375-        result = dispatch_integrator(state, tendency_fn, dt, integrator)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-376-        return self._apply_filter(result)
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-487-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-488-    def _integrate_on_cpu(self, state, n_steps, dt, save_every):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-489-        """Batch integration on CPU: transfer once, not per step."""
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:490:        state_cpu = jax.device_put(state, self._cpu_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-491-        trajectory_cpu = [state_cpu]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-492-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-493-        for i in range(n_steps):
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-496-                trajectory_cpu.append(state_cpu)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-497-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-498-        # Transfer back to Metal
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:499:        state_out = jax.device_put(state_cpu, self._default_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-500-        trajectory_out = [
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py:501:            jax.device_put(s, self._default_device) for s in trajectory_cpu
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-502-        ]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-503-        return state_out, trajectory_out
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py-504-
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-925-        slow_tendency_fn, acoustic_update_fn = self._build_se_functions()
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-926-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-927-        if self._use_cpu_for_spectral:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:928:            state_cpu = jax.device_put(state, self._cpu_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-929-            result_cpu = split_explicit_step(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-930-                state_cpu, slow_tendency_fn, acoustic_update_fn,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-931-                dt, se_config,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-932-            )
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:933:            state_new = jax.device_put(result_cpu, self._default_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-934-        else:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-935-            state_new = split_explicit_step(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-936-                state, slow_tendency_fn, acoustic_update_fn, dt, se_config,
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-1000-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-1001-    def _integrate_on_cpu(self, state, n_steps, dt, save_every):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-1002-        """Batch integration on CPU: transfer once, not per step."""
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1003:        state_cpu = jax.device_put(state, self._cpu_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-1004-        trajectory_cpu = [state_cpu]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-1005-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-1006-        # Anchored-mass snapshot (mirrors step(); the batched path calls
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-1018-                trajectory_cpu.append(state_cpu)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-1019-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-1020-        # Transfer back to Metal
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1021:        state_out = jax.device_put(state_cpu, self._default_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-1022-        trajectory_out = [
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py:1023:            jax.device_put(s, self._default_device) for s in trajectory_cpu
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-1024-        ]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-1025-        return state_out, trajectory_out
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py-1026-
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-364-            (u_d, v_d, state.T.data, state.p_s.data, state.phis.data),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-365-            None, "compute")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-366-        blocked = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:367:            "u_d": jax.device_put(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-368-                expand_corners_to_blocks(u_d, kt, nl), cz),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:369:            "v_d": jax.device_put(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-370-                expand_corners_to_blocks(v_d, kt, nl), cz),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:371:            "T": jax.device_put(T_in, cz),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:372:            "p_s": jax.device_put(ps_in, co),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:373:            "phis": jax.device_put(phis_in, co),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-374-        }
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-375-        if _moist:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-376-            q_pack = jnp.stack(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-377-                [state.tracers[nm].data for nm in _TILED_TRACERS], axis=-1)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-378-            (q_pack,) = cast_pytree((q_pack,), None, "compute")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:379:            blocked["q_pack"] = jax.device_put(q_pack, cz5)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-380-        return blocked
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-381-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py-382-    def step(blocked):
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_spectral.py-316-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_spectral.py-317-        integrator = self.config.time_integrator
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_spectral.py-318-        if self._use_cpu_for_spectral:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_spectral.py:319:            state_cpu = jax.device_put(state, self._cpu_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_spectral.py-320-            result = dispatch_integrator(state_cpu, tendency_fn, dt, integrator)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_spectral.py-321-            result = self._truncate(result)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_spectral.py:322:            return jax.device_put(result, self._default_device)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_spectral.py-323-        result = dispatch_integrator(state, tendency_fn, dt, integrator)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_spectral.py-324-        return self._truncate(result)
--
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-60-    _mp = jax.process_count() > 1
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-61-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-62-    def _put(arr):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:63:        # shard_leaf: single-process -> device_put (byte-unchanged); multi-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-64-        # controller -> per-process local band via make_array_from_process_local_data
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-65-        # (no all-gather, no transient global replica — issue #1100).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-66-        return shard_leaf(arr, NamedSharding(mesh, lat_spec(arr)), multiprocess=_mp)
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-87-    """Band-local Held-Suarez C-grid state, directly in the sharded layout.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-88-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-89-    The #1100 invariant for multi-process runs: **neither global builds nor
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:90:    ``device_put`` replication** — every global-shaped leaf is created with
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:91:    ``jax.make_array_from_callback``, whose callback is invoked only for the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-92-    row slices owned by THIS process's addressable devices (documented JAX
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-93-    semantics: per-addressable-shard callbacks with GLOBAL index slices).  No
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:94:    full global array is ever handed to ``device_put`` — the path measured to
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-95-    detonate under many-process packing in #1100 (its cross-process
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-96-    consistency check amplified even small replicated objects; an
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-97-    implementation behaviour we cite as measured, not as API contract).  This
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-140-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-141-    def _make(gshape, cb):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-142-        sharding = NamedSharding(mesh, P("lat", *((None,) * (len(gshape) - 1))))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:143:        return jax.make_array_from_callback(gshape, sharding, cb)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-144-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-145-    def _zeros_cb(gshape):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-146-        return lambda idx: jnp.zeros(_slice_shape(gshape, idx), dtype=_dtype)
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-184-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-185-    Multi-controller (route-B ``jax.distributed``, mesh spanning processes):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-186-    replication routes through a jit-compiled identity instead of
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:187:    ``device_put`` (see :func:`legoesm.parallel.latlon_spmd.replicate_leaf`,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-188-    the primitive shared with the ocean gather); the single-process path is
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-189-    byte-unchanged."""
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-190-    from legoesm.parallel.latlon_spmd import replicate_leaf
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-351-    polar-filter masks) over a leading band axis and lay them out on ``mesh``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-352-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-353-    ``shard_geometry=False`` (the historical layout): every stack is
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:354:    ``device_put`` REPLICATED (``P()``) — each device holds ALL bands'
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-355-    geometry and the body indexes its own band at ``axis_index``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-356-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-357-    ``shard_geometry=True`` (M2b): every stack is sharded ``P("lat", ...)``
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-387-            axis=0)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-388-    spec_of = lat_spec if shard_geometry else (lambda _arr: P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-389-    stacks = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:390:        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-391-        for name, arr in raw.items()
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-392-    }
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-393-    stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-543-    REUSES the shared primitives (band perms, the v-face round-trip, the band
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-544-    halo via the swapped backend, ``spmd_pole_end_masks``, the band slicer);
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-545-    atm-NEW is only the 6-field state/geometry walk. ``check_vma=False`` (the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:546:    band halo reads neighbour-rank data). Mirrors ``make_sharded_ocean_step``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-547-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-548-    ``physics_fn`` (optional): a STATELESS, COLUMN-LOCAL physics closure
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-549-    (``physics_fn(hs, grid, sigma_coord) -> HydrostaticTendencies``, e.g.
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1070-    inside the body from the east neighbour via the lon ring.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1071-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1072-    def _put(arr):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1073:        return jax.device_put(arr, NamedSharding(mesh, tile_spec(arr)))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1074-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1075-    n_lat, n_lon = state.T.shape[0], state.T.shape[1]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1076-    return state._replace(
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1204-            axis=0)[:, None]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1205-    spec_of = tile_spec if shard_geometry else (lambda _arr: P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1206-    stacks = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1207:        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1208-        for name, arr in raw.items()
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1209-    }
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1210-    stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}

exec
/usr/bin/bash -lc "rg -n -C 6 \"device_put|make_array_from_callback|shard_state_latlon|shard_forcing|make_sharded_ocean_step|jax\\.\" scripts/bench/bench_ocean_latlon_spmd_scaling.py; sed -n '1,360p' scripts/bench/bench_ocean_latlon_spmd_scaling.py; sed -n '1,260p' tests/parallel/test_latlon_ocean_spmd_multicontroller.py; sed -n '1,180p' tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py; sed -n '1,190p' tests/parallel/test_latlon_ocean_spmd_multiprocess.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 270ms:
2-
3-The ocean FULL-STEP twin of ``bench_atm_latlon_spmd_scaling.py`` (which this
4-mirrors flag-for-flag), closing the "no automated ocean full-step strong/weak
5-harness" gap: ``bench_ocean_mpi_scaling.py`` is the route-A (mpi4jax) phase-
6-split bench and ``bench_ocean_latlon_spmd_pcg.py`` times the barotropic PCG
7-KERNEL only — neither times the composed production step
8:(``make_sharded_ocean_step``: baroclinic + barotropic [implicit-CN
9-free-surface by default, split-explicit via ``--baro-solver``] + implicit
10-vmix + tracers) under the lat-band SPMD backend.
11-
12-  strong: fixed (n_lat, n_lon, nlev), vary n_devices -> speedup = t(1)/t(n).
13-  weak:   n_lat = nlat_per_dev * n_devices (fixed per-device rows) -> ideal flat.
14-
--
17-appends a JSON line. JAX_PLATFORMS=cpu with --xla_force_host_platform_device_count
18-gives virtual CPU devices (communication-overhead characterization, NOT a real
19-speedup); a real number needs one GPU per band. Run with JAX_ENABLE_X64=1 (the
20-ocean step's validated precision lane).
21-
22-Multi-controller (route-B, ``--multicontroller``): identical contract to the
23:atm bench — every process calls ``jax.distributed.initialize`` BEFORE any
24:other JAX use, the ("lat",) mesh is built over the GLOBAL ``jax.devices()``,
25:and the existing ``make_sharded_ocean_step`` band-ppermute halo + psum
26-reductions (incl. the barotropic ``_global_sum_pair``) run unchanged across
27-processes (NCCL on GPU / gloo on CPU). NO mpi4jax is armed in this mode (the
28-documented mixed-stack deadlock hazard).
29-
30-Launch (cluster, one process per GPU):
31-  srun -n 8 python bench_ocean_latlon_spmd_scaling.py --multicontroller \
--
52-# same pattern as bench_ocean_mpi_scaling's own cross-script imports).
53-sys.path.insert(0, str(Path(__file__).parent))
54-
55-# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
56-# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
57-# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
58:# imports JAX lazily, so this is safe before jax.distributed.initialize.
59-from metadata import (  # noqa: E402
60-    annotate_incomplete,
61-    calibrated_bound,
62-    comm_accounting,
63-    scaling_metadata,
64-    tidy_throughput_fields,
--
89-    exercises every term of the composed step — advection, Coriolis, PGF,
90-    the barotropic solve (implicit-CN by default; split-explicit when
91-    ``baro_solver="explicit_substep"``) and implicit vmix — instead of the
92-    trivial rest fixed point, mirroring the SPMD equivalence gate's IC
93-    recipe.
94-    """
95:    import jax.numpy as jnp
96-    from legoesm.grids.latlon import create_latlon_grid
97-    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
98-        LatLonCGridOceanModel,
99-    )
100-    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
101-    from legoesm.ocean.state import LatLonCGridOceanConfig
--
180-        eta=state.eta.replace(data=jnp.asarray(eta)),
181-        T=state.T.replace(data=jnp.asarray(temp)))
182-    return model, state
183-
184-
185-def _block(state):
186:    jax.block_until_ready([leaf for leaf in jax.tree.leaves(state)
187-                           if leaf is not None])
188-
189-
190-def main() -> int:
191-    p = argparse.ArgumentParser()
192-    p.add_argument("--n-lat", type=int, default=96)
--
298-                        "local per-substep clamping in this arm; A/B against "
299-                        "the default run).")
300-    p.add_argument("--wide-halo-chunk", type=int, default=0,
301-                   help="Substeps per wide exchange (0 = auto from the band "
302-                        "height).")
303-    p.add_argument("--multicontroller", action="store_true",
304:                   help="Route-B multi-controller: jax.distributed.initialize "
305-                        "per process, ('lat',) mesh over the GLOBAL device set "
306-                        "(one process per GPU / per CPU-device group). NO "
307:                        "mpi4jax. --n-devices must equal the global device "
308-                        "count.")
309-    p.add_argument("--coordinator", type=str, default=None,
310:                   help="host:port for jax.distributed when auto-detection "
311-                        "(SLURM) is unavailable; process count/id then come "
312-                        "from OMPI_COMM_WORLD_SIZE/RANK.")
313-    args = p.parse_args()
314-
315-    # Validate the block length BEFORE any model/device work.  (--warmup is
316-    # retained for CLI compat only and IGNORED by the fused-block timing —
--
326-    # as bench_ocean_mpi_scaling._ensure_precision).
327-    from legoesm.core.precision import PrecisionPolicy, set_policy
328-
329-    # Single source of truth = the LIVE jax x64 flag (an in-process caller
330-    # may have enabled it without the env var; keying on the env would
331-    # build fp32 states while every gate/metadata site keys on
332:    # jax.config — the exact mislabel this block exists to kill; codex).
333:    if jax.config.jax_enable_x64:
334-        set_policy(PrecisionPolicy.fp64())
335-    else:
336-        set_policy(PrecisionPolicy.fp32())
337-
338-    if args.multicontroller:
339-        # MUST run before any other JAX use (backend init). Shared helper:
340-        # explicit --coordinator -> OMPI/PALS launcher-env init; else
341-        # SLURM/OMPI auto-detect or PALS mpi4py bootstrap.
342-        from legoesm.parallel.early_init import init_multicontroller_distributed
343-        init_multicontroller_distributed(args.coordinator)
344-
345-    from legoesm.ocean.dynamics.sharded_ocean_step import (
346:        make_sharded_ocean_step,
347:        shard_state_latlon,
348-    )
349-
350-    nd = args.n_devices
351:    avail = len(jax.devices())
352-    if avail < nd:
353-        raise SystemExit(f"need {nd} devices, have {avail} "
354-                         f"(set --xla_force_host_platform_device_count)")
355-    if args.multicontroller and nd != avail:
356-        # A mesh over a strict subset would leave some processes' devices out
357-        # of the program (non-addressable participation hazard). Route-B uses
358-        # ALL global devices: one band per device across every process.
359-        raise SystemExit(
360-            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
361:            f"device count ({avail} across {jax.process_count()} processes).")
362-    n_lat = args.n_lat if args.mode == "strong" else args.nlat_per_dev * nd
363-    if n_lat % nd != 0:
364-        raise SystemExit(f"n_lat {n_lat} not divisible by n_devices {nd}")
365-
366-    if args.parity_gate and args.steps > SPMD_PARITY_MAX_STEPS:
367-        raise SystemExit(
--
377-    # the HOST cpu backend, not the accelerator. rest-state init runs jnp
378-    # ops at GLOBAL shape; on the default (GPU) device that materialises
379-    # ~7.4 global-field-equivalents of setup residency per device AND
380-    # compiles a global-sized init program — the 102 GB wall that killed
381-    # LL2304@64 (probe 26523157: the compiled STEP is clean; the residency
382-    # is setup-time). Host-side globals are RAM, and only the per-band
383:    # shards reach the accelerator via shard_state_latlon's device_put.
384-    # Identical values on every process (deterministic init + the step
385-    # factory's existing process-0 broadcast + content-hash guard).
386-    # nd==1 keeps the old on-device build: that lane TIMES the
387-    # single-device step, so its state belongs on the accelerator.
388-    import contextlib
389-    _build_ctx = contextlib.nullcontext()
390-    if nd > 1:
391-        try:
392:            # local_devices, NOT devices: under multicontroller jax.devices()
393-            # returns the GLOBAL list, so [0] is process 0's cpu device
394-            # — non-addressable elsewhere (probe job 26524163).
395:            _build_ctx = jax.default_device(
396:                jax.local_devices(backend="cpu")[0])
397-        except RuntimeError:
398-            # cpu backend not registered (JAX_PLATFORMS=cuda). The fix
399-            # needs JAX_PLATFORMS=cuda,cpu; fall back to the old on-device
400-            # build LOUDLY rather than crash.
401-            print("[#1370] WARNING: no cpu backend — global init will "
402-                  "materialise on the accelerator (set "
--
444-    if args.check_conservation:
445-        from bench_ocean_mpi_scaling import ocean_invariants
446-        inv_before = ocean_invariants(model, s0, n_ranks=1)
447-
448-    if nd == 1:
449-        mesh = None
450:        step = make_sharded_ocean_step(model, None)
451-        s = s0
452-    else:
453:        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
454-                                 axis_names=("lat",))
455:        step = make_sharded_ocean_step(model, mesh)
456:        s = shard_state_latlon(s0, mesh)
457-
458-    # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
459-    # blocks with sync only AROUND the block — the previous per-step
460-    # host-synced loop measured dispatch+sync latency, not fused device
461-    # throughput.  Per-step dispatch latency is still measured, SEPARATELY,
462-    # by an individually-synced probe (``step_latency_ms``); multi-controller
--
477-        sync_label="ocean_latlon_spmd_bench")
478-
479-    # Post-run ZERO-FORCING residual probe eligibility (audit item 6):
480-    # needs the gathered global final state on ONE process; multicontroller
481-    # runs skip it (zero_forcing_probe_measured=False, never faked).
482-    _probe_residual = (args.baro_solver == "implicit_cn"
483:                       and jax.process_count() == 1)
484-
485-    final_global = None
486-    if args.parity_gate or args.check_conservation or _probe_residual:
487-        from legoesm.ocean.dynamics.sharded_ocean_step import (
488-            gather_state_latlon,
489-        )
490-        final_global = (gather_state_latlon(s, mesh) if mesh is not None
491-                        else s)
492-
493-    # --- Correctness gates (before any timing is reported) -----------------
494-    if args.parity_gate or args.check_conservation:
495:        prec = "float64" if jax.config.jax_enable_x64 else "float32"
496:        rank0 = jax.process_index() == 0
497-        if args.check_conservation:
498-            from bench_ocean_mpi_scaling import (
499-                CONS_RTOL_DEFAULTS,
500-                conservation_breaches,
501-                ocean_invariants,
502-                print_conservation,
--
531-                return 5
532-
533-    # --- Solver iterations + post-run residual (audit item 6) --------------
534-    # implicit_cn dispatch (barotropic_implicit_latlon_cgrid): the SPMD /
535-    # distributed / force_pcg path runs a FIXED-iteration PCG
536-    # (barotropic_implicit_pcg_fixed_iters, static fori_loop); the plain
537:    # single-device path runs the stock adaptive jax.scipy CG whose
538-    # iteration count is not exposed (tol/maxiter recorded instead).
539-    _baro_cfg = model.config.barotropic
540-    _pcg_fixed_path = (args.baro_solver == "implicit_cn"
541-                       and (nd > 1
542-                            or bool(_baro_cfg.barotropic_implicit_force_pcg)))
543-    if _pcg_fixed_path:
--
548-    elif args.baro_solver == "implicit_cn":
549-        solver_iters = None
550-        solver_iters_mode = (
551-            "adaptive_stock_cg(maxiter="
552-            f"{int(_baro_cfg.barotropic_implicit_pcg_maxiter)},"
553-            f"tol={_baro_cfg.barotropic_implicit_pcg_tol:g}) — iteration "
554:            "count not exposed by jax.scipy CG")
555-    else:
556-        solver_iters = None
557-        solver_iters_mode = "explicit_substep (no iterative solve)"
558-
559-    # The probe below solves the free surface at the final state with ZERO
560-    # slow-forcing (F_slow defaults), which is a DIFFERENT right-hand side
--
597-                barotropic=_probe_cfg.barotropic._replace(
598-                    barotropic_implicit_force_pcg=True))
599-        _probe_out = barotropic_implicit_latlon_cgrid(
600-            final_global, args.dt, model.grid, model.z_coord, _probe_cfg,
601-            return_residual=True)
602-        zero_forcing_probe_residual = float(
603:            jax.block_until_ready(_probe_out[2]))
604-        zero_forcing_probe_measured = True
605-
606-    # --- Communication accounting (audit item 4) + calibrated bound (8) ----
607-    # Analytic INTER-DEVICE census, barotropic-solver scope ONLY (the
608-    # baroclinic 3-D pads are not counted -> bytes/comm are a LOWER census,
609-    # flagged machine-readably via halo_bytes_is_lower_bound; T_bound is a
--
621-            estimate_barotropic_halo_messages,
622-        )
623-        _halo_est = estimate_barotropic_halo_messages(
624-            model.config, model.config.barotropic.n_barotropic_substeps)
625-    else:
626-        _halo_est = None
627:    _dtype_bytes = 8 if jax.config.jax_enable_x64 else 4
628-    _n_reductions = None
629-    if nd <= 1:
630-        _msgs, _n_reductions = 0, 0
631-        _bytes_lower = False   # zero traffic is exact, not an undercount
632-        _comm_note = ("single device: no inter-device halo/reduction "
633-                      "traffic")
--
692-    )
693-
694-    rec = dict(
695-        component="ocean",
696-        mode=args.mode, n_devices=nd, n_lat=n_lat, n_lon=args.n_lon,
697-        nlev=args.nlev, steps=args.steps,
698:        platform=jax.default_backend(),
699:        n_processes=jax.process_count(),
700-        multicontroller=bool(args.multicontroller),
701-        steady_median_ms=(round(med, 4) if med is not None else None),
702-        cells=n_lat * args.n_lon * args.nlev,
703-        **timing,
704-    )
705-    # Flat aggregator-compatible identity + metric fields (see the atm
706-    # latlon twin): rows become visible to aggregate_bcw_scaling.py /
707-    # the CPU-vs-GPU plots, keyed as component="ocean" (already in rec).
708-    rec.update(
709-        grid_type="tripole" if args.tripole else "latlon",
710-        resolution=n_lat,
711-        n_levels=args.nlev,
712:        precision="float64" if jax.config.jax_enable_x64 else "float32",
713-        physics_level="none",
714:        backend=jax.default_backend(),
715-        **tidy_throughput_fields(
716-            dt_seconds=args.dt, time_per_step_ms=med,
717-            total_cells=n_lat * args.n_lon * args.nlev),
718-    )
719-    # Increment-2 accounting (audit items 4/6/8/9): flat fields so
720-    # aggregators read them without descending into metadata/extra.
--
736-    _nccl_report = nccl_transport_report()
737-    rec["metadata"] = annotate_incomplete(scaling_metadata(
738-        grid="tripole" if args.tripole else "latlon",
739-        component="ocean",
740-        resolution=f"{n_lat}x{args.n_lon}",
741-        n_levels=args.nlev,
742:        precision="float64" if jax.config.jax_enable_x64 else "float32",
743:        n_gpus=(nd if jax.default_backend() in ("gpu", "cuda", "rocm")
744-                else 0),
745-        decomposition="band" if nd > 1 else "none",
746-        # +wide_halo marks the A/B arm so aggregation never conflates it
747-        # with the per-substep-pad baseline.
748-        solver_variant=(model.config.barotropic.barotropic_solver
749-                        + ("+wide_halo" if args.wide_halo else "")),
--
752-        # rec.zero_forcing_probe_residual + extra below (codex batch4).
753-        solver_residual=None,
754-        # cells_per_rank is per PROCESS (n_ranks semantics); the per-device
755-        # share lives in extra.cells_per_device — a single-process 4-device
756-        # SPMD run has 1 rank owning ALL cells (codex finding 3).
757-        cells_per_rank=(n_lat * args.n_lon * args.nlev)
758:        // max(jax.process_count(), 1),
759-        scaling_kind=args.mode,
760-        extra={
761-            "steps": args.steps,
762-            "warmup": args.warmup,
763-            "multicontroller": bool(args.multicontroller),
764-            "fused_halo": os.environ.get(
--
783-            "solver_iters_mode": solver_iters_mode,
784-            "zero_forcing_probe_measured": zero_forcing_probe_measured,
785-        },
786-    ))
787-    # Multi-controller: every process times the same program; process 0 owns
788-    # the JSONL + stdout (others would duplicate/corrupt the append).
789:    if jax.process_index() == 0:
790-        os.makedirs(os.path.dirname(args.out), exist_ok=True)
791-        with open(args.out, "a") as f:
792-            f.write(json.dumps(rec) + "\n")
793-        print(json.dumps(rec))
794-        _fused_txt = (f"{med:.3f}ms/step" if med is not None
795-                      else "n/a (zero-length parity block)")
"""Strong/weak scaling bench for the lat-band SPMD lat-lon C-grid OCEAN step.

The ocean FULL-STEP twin of ``bench_atm_latlon_spmd_scaling.py`` (which this
mirrors flag-for-flag), closing the "no automated ocean full-step strong/weak
harness" gap: ``bench_ocean_mpi_scaling.py`` is the route-A (mpi4jax) phase-
split bench and ``bench_ocean_latlon_spmd_pcg.py`` times the barotropic PCG
KERNEL only — neither times the composed production step
(``make_sharded_ocean_step``: baroclinic + barotropic [implicit-CN
free-surface by default, split-explicit via ``--baro-solver``] + implicit
vmix + tracers) under the lat-band SPMD backend.

  strong: fixed (n_lat, n_lon, nlev), vary n_devices -> speedup = t(1)/t(n).
  weak:   n_lat = nlat_per_dev * n_devices (fixed per-device rows) -> ideal flat.

Device count is fixed at process start, so each n_devices runs as a SEPARATE
process (one sbatch step per count); this script benches ONE n_devices and
appends a JSON line. JAX_PLATFORMS=cpu with --xla_force_host_platform_device_count
gives virtual CPU devices (communication-overhead characterization, NOT a real
speedup); a real number needs one GPU per band. Run with JAX_ENABLE_X64=1 (the
ocean step's validated precision lane).

Multi-controller (route-B, ``--multicontroller``): identical contract to the
atm bench — every process calls ``jax.distributed.initialize`` BEFORE any
other JAX use, the ("lat",) mesh is built over the GLOBAL ``jax.devices()``,
and the existing ``make_sharded_ocean_step`` band-ppermute halo + psum
reductions (incl. the barotropic ``_global_sum_pair``) run unchanged across
processes (NCCL on GPU / gloo on CPU). NO mpi4jax is armed in this mode (the
documented mixed-stack deadlock hazard).

Launch (cluster, one process per GPU):
  srun -n 8 python bench_ocean_latlon_spmd_scaling.py --multicontroller \
      --n-devices 8 ...            # SLURM: coordinator auto-detected
  mpiexec -n 8 python ... --multicontroller --coordinator host0:9876
CPU smoke (single process, virtual devices):
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  JAX_ENABLE_X64=1 python scripts/bench/bench_ocean_latlon_spmd_scaling.py \
      --n-lat 48 --n-lon 96 --nlev 10 --n-devices 4 --steps 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np

# Sibling-script import (ocean_invariants / conservation helpers reuse —
# same pattern as bench_ocean_mpi_scaling's own cross-script imports).
sys.path.insert(0, str(Path(__file__).parent))

# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
# imports JAX lazily, so this is safe before jax.distributed.initialize.
from metadata import (  # noqa: E402
    annotate_incomplete,
    calibrated_bound,
    comm_accounting,
    scaling_metadata,
    tidy_throughput_fields,
    wet_cell_metrics,
)

# SPMD full-step parity tolerances — the FLOATING-POINT RE-ASSOCIATION floor
# of the sharded split-explicit barotropic (ppermute/psum reduction-order
# change over the ~30-substep loop, pole-amplified), NOT a bug margin; a real
# missing-halo regression shows up at O(1e-3+) at the band cuts.  Values
# mirror the equivalence gate (tests/parallel/test_latlon_ocean_spmd_step.py,
# 3 steps: atol 2e-4); the floor grows with steps, hence the smoke cap.
SPMD_PARITY_TOLS = {  # precision -> (rtol, atol)
    "float64": (1.0e-3, 2.0e-4),
    "float32": (1.0e-2, 2.0e-3),
}
SPMD_PARITY_MAX_STEPS = 8


def build_model_and_state(n_lat, n_lon, nlev, seed=0, *,
                          wide_halo=False, wide_halo_chunk=0,
                          tripole=False, baro_solver="implicit_cn",
                          force_pcg=False, pcg_variant="standard",
                          pcg_fixed_iters=0):
    """Ocean model + gently perturbed rest state (flat 4000 m bottom).

    The perturbation (small u/v/eta/T noise on the rest stratification)
    exercises every term of the composed step — advection, Coriolis, PGF,
    the barotropic solve (implicit-CN by default; split-explicit when
    ``baro_solver="explicit_substep"``) and implicit vmix — instead of the
    trivial rest fixed point, mirroring the SPMD equivalence gate's IC
    recipe.
    """
    import jax.numpy as jnp
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    if tripole:
        # Synthetic tripole (ORCA fold): the sharded step's fold support
        # is gated by tests/parallel/test_latlon_ocean_spmd_tripole.py;
        # the wide-halo lever refuses folds at model construction.
        from legoesm.grids.tripole import create_synthetic_tripole

        grid = create_synthetic_tripole(n_lat=n_lat, n_lon=n_lon)
    else:
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    # Wide-halo lever (A/B): one fused wide lat-halo exchange per chunk of
    # barotropic substeps instead of ~4 ppermute pads per substep.  The wide
    # path's per-substep clamp is local by contract, so pin local clamping
    # in BOTH arms for a controlled comparison.
    # Production-matching solver (scaling audit, bottleneck 4): OMIP runs
    # implicit_cn (run_omip.py full preset); the config-dataclass default
    # is explicit_substep, so it MUST be set explicitly here or the bench
    # measures a non-production step.
    flat = {"barotropic_solver": baro_solver}
    if force_pcg:
        # Solver-matched strong ladders (codex 2026-07-24 finding 2): the
        # implicit-CN dispatch runs adaptive stock CG on a SINGLE device but
        # the fixed-iteration distributed PCG under SPMD/MPI — an nd=1
        # reference leg without this flag times a DIFFERENT solver than the
        # nd>1 legs. Forces the fixed-M PCG everywhere.
        if baro_solver != "implicit_cn":
            raise SystemExit(
                "--force-pcg only affects the implicit_cn barotropic solve; "
                "drop it for explicit_substep arms.")
        flat["barotropic_implicit_force_pcg"] = True
    if pcg_fixed_iters:
        # Each PCG iteration contributes DEPENDENT reduction batches, and
        # the campaign's mechanism finding (job 26458930) is that exposed
        # dependent sync — not bytes, not schedulable overlap — is what the
        # step pays above its roofline. Iteration count is therefore the
        # most direct sync-point lever available in config.
        if baro_solver != "implicit_cn":
            raise SystemExit(
                "--pcg-fixed-iters only affects the implicit_cn fixed-M PCG.")
        flat["barotropic_implicit_pcg_fixed_iters"] = int(pcg_fixed_iters)
    if pcg_variant != "standard":
        if baro_solver != "implicit_cn":
            raise SystemExit(
                "--pcg-variant only affects the implicit_cn fixed-M PCG; "
                "drop it for explicit_substep arms.")
        flat["barotropic_implicit_pcg_variant"] = pcg_variant
    if wide_halo:
        if baro_solver != "explicit_substep":
            raise SystemExit(
                "--wide-halo is a split-explicit barotropic lever; it "
                "requires --baro-solver explicit_substep (implicit_cn has "
                "no substep halo to widen).")
        flat.update(barotropic_wide_halo=True,
                    barotropic_wide_halo_chunk=int(wide_halo_chunk),
                    barotropic_local_subcycle_clamp=True)
    model = LatLonCGridOceanModel(grid, z_coord,
                                  LatLonCGridOceanConfig.from_flat(**flat))
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    rng = np.random.default_rng(seed)
    # Project the velocity noise through the face masks: v_mask zeroes the pole
    # WALL rows, so the nd=1 and nd>1 runs time the SAME initial state (the
    # nd>1 shard drops v[n_lat] and reconstructs it as the pole-wall zero — a
    # random value there would make strong-scaling ICs differ across device
    # counts; codex).
    u_mask = np.asarray(state.u_mask.data)[..., None]
    v_mask = np.asarray(state.v_mask.data)[..., None]
    u = 0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev)) * u_mask
    v = 0.02 * rng.standard_normal((n_lat + 1, n_lon, nlev)) * v_mask
    eta = 0.005 * rng.standard_normal((n_lat, n_lon))
    temp = (5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
            + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)),
        T=state.T.replace(data=jnp.asarray(temp)))
    return model, state


def _block(state):
    jax.block_until_ready([leaf for leaf in jax.tree.leaves(state)
                           if leaf is not None])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-lat", type=int, default=96)
    p.add_argument("--n-lon", type=int, default=192)
    p.add_argument("--nlev", type=int, default=20)
    p.add_argument("--n-devices", type=int, required=True)
    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
    p.add_argument("--nlat-per-dev", type=int, default=24,
                   help="weak mode: lat rows per device")
    p.add_argument("--steps", type=int, default=12,
                   help="Steps per fused lax.scan timing block.")
    p.add_argument("--warmup", type=int, default=2,
                   help="(retained for CLI compat; fused-block timing "
                        "separates compile/probe/blocks explicitly).")
    p.add_argument("--blocks", type=int, default=2,
                   help="Timed fused blocks (per-block times expose drift).")
    p.add_argument("--probe-steps", type=int, default=3,
                   help="Individually-synced steps for the SEPARATE "
                        "dispatch-latency probe (step_latency_ms).")
    p.add_argument("--baro-solver",
                   choices=["implicit_cn", "explicit_substep"],
                   default="implicit_cn",
                   help="Barotropic solver. Default implicit_cn MATCHES "
                        "production OMIP (run_omip.py full preset); the "
                        "previous silent explicit_substep default made the "
                        "bench measure a non-production configuration "
                        "(scaling audit, bottleneck 4).")
    p.add_argument("--force-pcg", action="store_true",
                   help="Force the fixed-iteration distributed PCG for the "
                        "implicit_cn barotropic solve even on a single "
                        "device (barotropic_implicit_force_pcg=True), so an "
                        "nd=1 strong-scaling reference leg times the SAME "
                        "solver the nd>1 SPMD legs run (the dispatch "
                        "otherwise routes nd=1 to adaptive stock CG).")
    p.add_argument("--seed", type=int, default=0,
                   help="IC perturbation seed. Repeats across seeds give a "
                        "variance estimate, without which a single-run "
                        "difference between two arms cannot be called an "
                        "ordering (codex round-8).")
    p.add_argument("--pcg-fixed-iters", type=int, default=0,
                   help="Iterations of the fixed-M implicit_cn PCG "
                        "(0 = scheme default, 60). Each iteration carries "
                        "DEPENDENT reduction batches, so this is the most "
                        "direct lever on the exposed-sync cost that "
                        "dominates this step above its roofline. Lowering "
                        "it trades solver convergence for sync points — "
                        "check zero_forcing_probe_residual in the output "
                        "before believing any speedup.")
    p.add_argument("--pcg-variant", choices=["standard", "single_reduce"],
                   default="standard",
                   help="Fixed-M PCG recurrence for the implicit_cn "
                        "barotropic solve: standard = 2 dependent reduction "
                        "batches/iter; single_reduce = Chronopoulos-Gear, "
                        "ONE batched reduction/iter (halves the per-step "
                        "reduction count the census reports).")
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--single-dev-fused-ms", type=float, default=None,
                   help="fused_step_ms of the nd=1 row at the SAME per-device "
                        "size (the compute ingredient of the calibrated "
                        "T_bound, audit item 8). Omitted at nd>1 -> the bound "
                        "is emitted null + flagged incomplete (never "
                        "fabricated); nd=1 rows use their own measurement.")
    p.add_argument("--comm-latency-us", type=float, default=None,
                   help="MEASURED per-message latency [us] of THIS machine's "
                        "fabric (ping-pong microbenchmark). Default: the "
                        "MACHINE-CALIBRATED-REQUIRED placeholder in "
                        "metadata.py -> the record carries "
                        "bound_calibrated=false.")
    p.add_argument("--comm-bandwidth-gbs", type=float, default=None,
                   help="MEASURED link bandwidth [GB/s] of THIS machine's "
                        "fabric. Default: the MACHINE-CALIBRATED-REQUIRED "
                        "placeholder in metadata.py -> "
                        "bound_calibrated=false.")
    p.add_argument("--out", type=str,
                   default="results/a1/ocean_spmd_scaling.jsonl")
    p.add_argument(
        "--parity-gate", action="store_true",
        help="Correctness gate: compare the gathered sharded trajectory "
             "against the single-device trajectory at the sharded "
             "split-explicit re-association-floor tolerances (smoke windows "
             "only; the floor grows with steps).")
    p.add_argument(
        "--check-conservation", action="store_true",
        help="Gate global area/eta/heat/salt drift over the run "
             "(pre-shard global state vs gathered final state; exits "
             "nonzero on breach).")
    p.add_argument(
        "--cons-rtol", type=float, default=None,
        help="Conservation tolerance (default: 1e-9 f64 / 1e-4 f32; the "
             "raw scheme drifts ~1e-8/step — calibrate to the window).")
    p.add_argument("--tripole", action="store_true",
                   help="Synthetic tripole (ORCA-fold) lane: the sharded "
                        "step folds the north band data-dependently "
                        "(SPMD equivalence gated at 4 devices). Rows are "
                        "tagged grid=tripole. Incompatible with "
                        "--wide-halo (fold refused at construction).")
    p.add_argument("--fused-halo", action="store_true",
                   help="Opt-in SPMD halo message aggregation "
                        "(LEGOESM_LATLON_SPMD_FUSED_HALO=1): one ppermute "
                        "pair per direction per dtype group at every "
                        "pad_multi site instead of one per field — "
                        "measured 25%% fewer static collective-permutes on "
                        "this step, bit-identical results. A/B against "
                        "the default run.")
    p.add_argument("--wide-halo", action="store_true",
                   help="Opt-in wide-halo split-explicit barotropic: one "
                        "fused wide lat-halo exchange per chunk of substeps "
                        "instead of ~4 ppermute pads per substep (implies "
                        "local per-substep clamping in this arm; A/B against "
                        "the default run).")
    p.add_argument("--wide-halo-chunk", type=int, default=0,
                   help="Substeps per wide exchange (0 = auto from the band "
                        "height).")
    p.add_argument("--multicontroller", action="store_true",
                   help="Route-B multi-controller: jax.distributed.initialize "
                        "per process, ('lat',) mesh over the GLOBAL device set "
                        "(one process per GPU / per CPU-device group). NO "
                        "mpi4jax. --n-devices must equal the global device "
                        "count.")
    p.add_argument("--coordinator", type=str, default=None,
                   help="host:port for jax.distributed when auto-detection "
                        "(SLURM) is unavailable; process count/id then come "
                        "from OMPI_COMM_WORLD_SIZE/RANK.")
    args = p.parse_args()

    # Validate the block length BEFORE any model/device work.  (--warmup is
    # retained for CLI compat only and IGNORED by the fused-block timing —
    # the old `warmup < steps` check would spuriously reject valid runs,
    # e.g. a one-step parity smoke with the default --warmup=2; codex.)
    if args.steps < 1:
        raise SystemExit(f"--steps must be >= 1, got {args.steps}")

    # Align the legoESM precision POLICY with the jax x64 flag: the ocean
    # state dtype comes from get_policy().storage (default fp32), so an
    # x64-flag-only run would build f32 states and gate them against
    # f64-labeled tolerances (the tripole lane caught this; the same fix
    # as bench_ocean_mpi_scaling._ensure_precision).
    from legoesm.core.precision import PrecisionPolicy, set_policy

    # Single source of truth = the LIVE jax x64 flag (an in-process caller
    # may have enabled it without the env var; keying on the env would
    # build fp32 states while every gate/metadata site keys on
    # jax.config — the exact mislabel this block exists to kill; codex).
    if jax.config.jax_enable_x64:
        set_policy(PrecisionPolicy.fp64())
    else:
        set_policy(PrecisionPolicy.fp32())

    if args.multicontroller:
        # MUST run before any other JAX use (backend init). Shared helper:
        # explicit --coordinator -> OMPI/PALS launcher-env init; else
        # SLURM/OMPI auto-detect or PALS mpi4py bootstrap.
        from legoesm.parallel.early_init import init_multicontroller_distributed
        init_multicontroller_distributed(args.coordinator)

    from legoesm.ocean.dynamics.sharded_ocean_step import (
        make_sharded_ocean_step,
        shard_state_latlon,
    )

    nd = args.n_devices
    avail = len(jax.devices())
    if avail < nd:
        raise SystemExit(f"need {nd} devices, have {avail} "
                         f"(set --xla_force_host_platform_device_count)")
    if args.multicontroller and nd != avail:
        # A mesh over a strict subset would leave some processes' devices out
        # of the program (non-addressable participation hazard). Route-B uses
        # ALL global devices: one band per device across every process.
        raise SystemExit(
            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
"""Route-B multi-controller gate for the lat-band SPMD ocean step.

The ocean twin of ``test_atm_latlon_spmd_multicontroller.py``: N processes
federate via ``jax.distributed`` into ONE multi-controller program, the
("lat",) mesh spans the GLOBAL device set, and ``make_sharded_ocean_step`` +
the band ppermute/psum halo (incl. the barotropic ``_global_sum_pair`` psum)
run UNCHANGED — collectives cross processes via the distributed runtime
(NCCL/gloo). NO mpi4jax anywhere in this file (mixed-stack deadlock hazard),
which is also why it lives in ``tests/parallel/``, NOT ``tests/distributed/``
(whose session conftest auto-arms the mpi4jax layout).

GATED: skips unless ``LEGOESM_JAX_DISTRIBUTED_TEST=1`` — ``jax.distributed
.initialize`` must run BEFORE the first backend touch, so this file must be
launched ALONE (its own pytest process per rank), never inside a shared pytest
session. Env-fragile by nature (gloo hostname on CPU, NCCL topology on GPU) —
the gate keeps default CI green.

Run (2 processes x 2 host CPU devices = 4 bands):
  LEGOESM_JAX_DISTRIBUTED_TEST=1 JAX_PLATFORMS=cpu \
  XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_ENABLE_X64=1 \
  mpiexec -n 2 python -m pytest \
      tests/parallel/test_latlon_ocean_spmd_multicontroller.py -x -q
(SLURM: srun -n 2 ... — coordinator auto-detected, OMPI env vars unneeded.)
"""
from __future__ import annotations

import os

import pytest

if os.environ.get("LEGOESM_JAX_DISTRIBUTED_TEST") != "1":
    pytest.skip(
        "multi-controller jax.distributed test: set "
        "LEGOESM_JAX_DISTRIBUTED_TEST=1 and launch this file alone under "
        "mpiexec/srun (see module docstring)",
        allow_module_level=True,
    )

import jax  # noqa: E402  (import gated so the skip never touches the backend)

# initialize() BEFORE any backend touch — the process count/id decision comes
# from the LAUNCHER env ONLY (a jax.process_count() query here would
# instantiate the local backend client pre-federation; codex atm round-1 HIGH).
_n = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                        os.environ.get("SLURM_NTASKS", "1")))
_r = int(os.environ.get("OMPI_COMM_WORLD_RANK",
                        os.environ.get("SLURM_PROCID", "0")))
if _n > 1:
    if os.environ.get("SLURM_STEP_NODELIST"):
        jax.distributed.initialize()
    else:
        _coord = os.environ.get("LEGOESM_JAX_COORDINATOR", "127.0.0.1:29778")
        jax.distributed.initialize(
            coordinator_address=_coord, num_processes=_n, process_id=_r)
    # Verify federation AFTER init (safe to touch the backend now).
    if jax.process_count() != _n:
        pytest.skip(
            f"jax.distributed federated {jax.process_count()} processes, "
            f"launcher started {_n} — environment did not federate",
            allow_module_level=True,
        )

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: E402
    gather_state_latlon,
    make_sharded_ocean_step,
    shard_state_latlon,
)
from legoesm.ocean.state import LatLonCGridOceanConfig  # noqa: E402
from legoesm.ocean.vertical import create_ocean_z_star  # noqa: E402

# Reuse the single-process gate's perturbed-IC fixture — same conventions.
from tests.parallel.test_latlon_ocean_spmd_step import (  # noqa: E402
    _perturbed_state,
)


@pytest.fixture(autouse=True)
def _restore_halo_backend():
    yield
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("local")


def _global_mesh():
    if jax.process_count() < 2:
        pytest.skip("needs >= 2 processes (mpiexec/srun -n 2)")
    return jax.sharding.Mesh(np.array(jax.devices()), axis_names=("lat",))


def test_multicontroller_ocean_step_matches_serial():
    """The single-process SPMD equivalence gate, multi-process: N global bands
    across >= 2 processes must reproduce the per-process serial reference
    (identical host ICs on every process) at the split-explicit re-association
    floor (same atol/rtol as test_latlon_ocean_spmd_step — NOT a bug margin,
    see its tolerance note), and the gather (the multiprocess
    ``replicate_leaf`` branch, exercised FOR REAL here) must return the full
    global state on every process."""
    mesh = _global_mesh()
    n_dev = mesh.devices.size
    n_lat, n_lon, nlev = 48, 96, 10
    if n_lat % n_dev != 0:
        pytest.skip(f"n_lat {n_lat} % global devices {n_dev} != 0")
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    model = LatLonCGridOceanModel(grid, z_coord,
                                  LatLonCGridOceanConfig.from_flat())
    state0 = _perturbed_state(grid, z_coord)
    dt, n_steps = 600.0, 3

    # Serial reference: every process computes it from the identical host IC.
    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt)
    # Prime the build-once vertex-mask cache from the CONCRETE state (the
    # serial run above already did; belt-and-braces for wrapper band masks).
    model._ensure_vertex_mask(state0)

    step = make_sharded_ocean_step(model, mesh)
    ss = shard_state_latlon(state0, mesh)
    for _ in range(n_steps):
        ss = step(ss, dt)
    out = gather_state_latlon(ss, mesh)

    # Tolerance = the split-explicit FP re-association floor of the sharded
    # barotropic (see test_latlon_ocean_spmd_step's tolerance note verbatim).
    atol, rtol = 2.0e-4, 1.0e-3
    for nm in ("u", "v", "eta", "T", "S"):
        a = np.asarray(getattr(s, nm).data)
        b = np.asarray(getattr(out, nm).data)
        assert a.shape == b.shape, f"{nm} shape {b.shape} vs {a.shape}"
        np.testing.assert_allclose(
            b, a, atol=atol, rtol=rtol,
            err_msg=(
                f"multi-controller lat-band SPMD ocean ({n_dev} bands, "
                f"{jax.process_count()} processes) diverged from serial in "
                f"'{nm}' — cross-process ppermute/psum or the multi-controller "
                f"gather is wrong."))

    # Non-vacuity: the sharded split-explicit re-association must be present
    # (bands really ran) — a bit-identical result would mean the sharding
    # silently collapsed to single-device.
    u_diff = float(np.max(np.abs(
        np.asarray(out.u.data) - np.asarray(s.u.data))))
    assert u_diff > 1e-13, (
        "multi-controller ocean u is bit-identical to serial — the band "
        "decomposition did not actually run; this gate would be vacuous.")
"""SELF-SPAWNING multi-controller gate for the ocean lat-band SPMD bench.

Companion to tests/parallel/test_latlon_ocean_spmd_multicontroller.py (the
launcher-gated variant that needs an external `srun`/`mpirun` +
LEGOESM_JAX_DISTRIBUTED_TEST=1 and therefore never runs in plain pytest CI):
THIS variant spawns its own two worker processes and drives the FULL
production bench (``bench_ocean_latlon_spmd_scaling.py --multicontroller``)
end-to-end with the ported parity + conservation gates armed — pinning the
multi-controller pieces a Derecho/Levante multi-node NCCL ocean run
exercises: ``shard_state_latlon`` onto a mesh spanning non-addressable
devices, cross-process ppermute/psum inside the jitted step, the replication
gather, rank-0-gated output, and the ``sync_global_devices`` barriers.

The workers use the bench's ``--coordinator`` path with the OMPI env vars
the bench reads for rank/size; port races retry via multihost_harness.
Numerics of the sharded step itself:
tests/parallel/test_latlon_ocean_spmd_step.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_BENCH = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "bench" / "bench_ocean_latlon_spmd_scaling.py"
)
N_PROC = 2


@pytest.mark.timeout(600)
def test_ocean_spmd_two_process_selfspawn_bench_parity(tmp_path):
    from multihost_harness import run_federated

    out = tmp_path / "ocean_spmd.jsonl"
    base_env = dict(os.environ)
    base_env["JAX_PLATFORMS"] = "cpu"
    base_env["JAX_ENABLE_X64"] = "1"
    base_env.pop("XLA_FLAGS", None)  # 1 real CPU device per process

    def build_cmd(rank, port):
        return [
            sys.executable, str(_BENCH),
            "--multicontroller", "--coordinator", f"localhost:{port}",
            "--n-devices", str(N_PROC),
            "--n-lat", "16", "--n-lon", "32", "--nlev", "4",
            "--steps", "4", "--warmup", "1", "--dt", "600",
            "--parity-gate", "--check-conservation", "--cons-rtol", "1e-6",
            "--out", str(out),
        ]

    # The bench reads rank/size from OMPI env when --coordinator is given;
    # run_federated only varies argv, so vary the env via a per-rank wrapper.
    def build_cmd_with_env(rank, port):
        cmd = build_cmd(rank, port)
        # Encode env in the command via `env` so each worker gets its rank.
        return [
            "env",
            f"OMPI_COMM_WORLD_SIZE={N_PROC}",
            f"OMPI_COMM_WORLD_RANK={rank}",
        ] + cmd

    rcs, outs = run_federated(build_cmd_with_env, N_PROC, base_env,
                              timeout_s=540)
    for rank, (rc, txt) in enumerate(zip(rcs, outs)):
        assert rc == 0, (
            f"rank {rank} exited {rc}\n--- output ---\n{txt[-4000:]}"
        )

    # Rank 0 owns the JSONL; the record carries the multicontroller fields.
    assert out.exists(), outs[0][-2000:]
    rec = json.loads(out.read_text().strip().splitlines()[-1])
    assert rec["component"] == "ocean"
    assert rec["n_devices"] == N_PROC
    assert rec["n_processes"] == N_PROC
    assert rec["multicontroller"] is True
    # Parity gate ran and passed (rank-0-gated banner).
    assert "parity" in outs[0] and "MISMATCH" not in outs[0]
"""Pytest wrapper for the multi-PROCESS jax.distributed equivalence gate.

The real assertions live in the STANDALONE script
``scripts/validate/validate_latlon_ocean_spmd_multiprocess.py``: it must run
OUTSIDE pytest because the root ``tests/conftest.py`` initializes the XLA backend
(``ensure_metal_or_fallback`` -> ``jax.default_backend()``) at collection, before a
test module could call ``jax.distributed.initialize()`` (which must precede backend
init).  So this wrapper SUBPROCESS-launches the script under ``mpirun -np 2`` (fresh
processes, no conftest, bootstrap-first) and asserts on its exit code — the
standard pattern for jax.distributed correctness in CI.

The wrapper also directly exercises the script's importable pieces (the perturbed-
state builder + the single-process bootstrap no-op) so the new ``.py`` files get a
direct in-process unit test (CLAUDE.md "every new .py gets a direct test"), without
needing MPI for the bulk of the coverage.

Run::

    pytest tests/parallel/test_latlon_ocean_spmd_multiprocess.py -v
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "validate" / "validate_latlon_ocean_spmd_multiprocess.py"


def test_validation_script_present_and_compiles():
    """The standalone multi-process validation script exists and is syntactically
    valid (so the sbatch that runs it under mpirun cannot silently 404 / import-
    error)."""
    import py_compile

    assert _SCRIPT.is_file(), f"missing validation script: {_SCRIPT}"
    py_compile.compile(str(_SCRIPT), doraise=True)


def test_single_process_bootstrap_is_noop():
    """The jax.distributed bootstrap is a NO-OP for a single process (returns
    (rank=0, nproc=1)) — the default single-controller path is byte-unchanged.

    This runs IN a fresh subprocess so it does not perturb the backend state of
    the pytest process (the bootstrap is import-order sensitive)."""
    code = (
        "from legoesm.parallel.distributed import "
        "initialize_jax_distributed_multiprocess as b; "
        "r,n=b(); assert (r,n)==(0,1),(r,n); print('NOOP_OK')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert out.returncode == 0, f"stdout={out.stdout}\nstderr={out.stderr}"
    assert "NOOP_OK" in out.stdout, out.stdout


def _run_two_phase(tripole: bool, tmp_path):
    """Phase 1 (single process, 4 CPU devices): compute the serial + 4-device-SPMD
    reference, assert equal, save it.  Phase 2 (``mpirun -np 2``, 4 global devices):
    run the multi-process SPMD step and compare to the saved reference + assert the
    gather is replicated across processes.  REQUIRE a real ``[PASS]`` on both (a
    vacuous ``[SKIP]`` -- e.g. <4 devices -- is a FAILURE here; codex MED).  The
    phase-2 script LAND-reduces every rank, so its exit code is the whole-job
    verdict."""
    ref = str(tmp_path / f"ref_{'tripole' if tripole else 'latlon'}.npz")
    extra = ["--tripole"] if tripole else []

    # Phase 1: single process, 4 CPU devices (NO jax.distributed -> the serial
    # reference's eta-floor reduction is a local sum, no mpi4jax).
    env1 = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4"}
    p1 = subprocess.run(
        [sys.executable, str(_SCRIPT), "--phase", "ref", "--out", ref, *extra],
        capture_output=True, text=True, timeout=900, env=env1)
    assert p1.returncode == 0 and "[PASS ref]" in p1.stdout, (
        f"phase ref FAILED (rc={p1.returncode})\n--- stdout ---\n{p1.stdout}\n"
        f"--- stderr ---\n{p1.stderr}")

    # Phase 2: mpirun -np 2, 2 CPU devices per process -> 4 global.
    env2 = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    cmd = ["mpirun", "-np", "2", sys.executable, str(_SCRIPT),
           "--phase", "mp", "--ref", ref, *extra]
    p2 = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env2)
    assert p2.returncode == 0, (
        f"multi-process gate FAILED (rc={p2.returncode})\n"
        f"--- stdout ---\n{p2.stdout}\n--- stderr ---\n{p2.stderr}")
    assert "[PASS]" in p2.stdout, (
        "expected a real [PASS] under mpirun -np 2 (a [SKIP] means the 4-device "
        f"setup did not materialize):\n--- stdout ---\n{p2.stdout}\n"
        f"--- stderr ---\n{p2.stderr}")


@pytest.mark.skipif(shutil.which("mpirun") is None,
                    reason="needs an MPI launcher (mpirun) for the 2-process gate")
def test_multiprocess_equivalence_under_mpirun_latlon(tmp_path):
    """Regular lat-lon: the cross-process ppermute/psum band halo + barotropic
    reductions + the replicated all-gather match the serial reference.

    Skipped where mpirun is unavailable (e.g. a login node without MPI on PATH);
    the sbatch ``run_multiprocess_cpu_equiv.sbatch`` runs it on a compute node."""
    pytest.importorskip("mpi4py")
    _run_two_phase(tripole=False, tmp_path=tmp_path)


@pytest.mark.skipif(shutil.which("mpirun") is None,
                    reason="needs an MPI launcher (mpirun) for the 2-process gate")
def test_multiprocess_equivalence_under_mpirun_tripole(tmp_path):
    """TRIPOLE (active bipolar north fold): the fold band lands on the NORTH band =
    a REMOTE process, exercising the fold IN the same shard_map program as the
    cross-process collectives — the eORCA025 path the regular-grid case + the
    single-process tripole case don't jointly cover (codex MED)."""
    pytest.importorskip("mpi4py")
    _run_two_phase(tripole=True, tmp_path=tmp_path)

exec
/usr/bin/bash -lc "python - <<'PY'
import inspect
import jax
import jax._src.api as api
import jax._src.dispatch as dispatch
print('JAX', jax.__version__)
print('make_array_from_callback')
print(inspect.getsource(jax.make_array_from_callback))
print('device_put impl')
print(inspect.getsource(dispatch._device_put_sharding_impl))
PY" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 1 in 323ms:
/usr/bin/bash: cannot create temp file for here-document: Read-only file system

exec
/usr/bin/bash -lc "python -c 'import inspect, jax, jax._src.dispatch as d; print(\"JAX\",jax.__version__); print(inspect.getsource(jax.make_array_from_callback)); print(\"device_put impl\"); print(inspect.getsource(d._device_put_sharding_impl))'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 127 in 305ms:
/usr/bin/bash: python: command not found

exec
/usr/bin/bash -lc "python3 -c 'import inspect, jax, jax._src.dispatch as d; print(\"JAX\",jax.__version__); print(inspect.getsource(jax.make_array_from_callback)); print(\"device_put impl\"); print(inspect.getsource(d._device_put_sharding_impl))'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 1 in 288ms:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'jax'

exec
/usr/bin/bash -lc "rg -n -C 8 \"class LatLonCGridOceanState|LatLonCGridOceanState\\s*=|NamedTuple.*LatLonCGridOceanState|LatLonCGridOceanState\" packages/ocean/legoesm/ocean/state.py packages/ocean/legoesm/ocean -g '*.py' | head -n 1000; rg -n -C 4 \"def _build\\(|_build\\(\" scripts/bench/bench_ocean_latlon_spmd_scaling.py; sed -n '80,190p' scripts/bench/bench_ocean_latlon_spmd_scaling.py; git status --short; git diff --stat HEAD; git diff --name-only HEAD" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 1882ms:
packages/ocean/legoesm/ocean/state.py-354-    eos: str = "wright"
packages/ocean/legoesm/ocean/state.py-355-    eos_linear: object = None
packages/ocean/legoesm/ocean/state.py-356-
packages/ocean/legoesm/ocean/state.py-357-
packages/ocean/legoesm/ocean/state.py-358-# ==============================================================================
packages/ocean/legoesm/ocean/state.py-359-# Lat-Lon C-Grid FV Ocean State
packages/ocean/legoesm/ocean/state.py-360-# ==============================================================================
packages/ocean/legoesm/ocean/state.py-361-
packages/ocean/legoesm/ocean/state.py:362:class LatLonCGridOceanState(NamedTuple):
packages/ocean/legoesm/ocean/state.py-363-    """State for the lat-lon C-grid finite-volume ocean primitive equations.
packages/ocean/legoesm/ocean/state.py-364-
packages/ocean/legoesm/ocean/state.py-365-    Velocities live on cell faces (Arakawa C-grid staggering):
packages/ocean/legoesm/ocean/state.py-366-    - u at east/west faces (lon interfaces): shape (n_lat, n_lon+1, nlev)
packages/ocean/legoesm/ocean/state.py-367-    - v at north/south faces (lat interfaces): shape (n_lat+1, n_lon, nlev)
packages/ocean/legoesm/ocean/state.py-368-
packages/ocean/legoesm/ocean/state.py-369-    Scalars live at cell centers:
packages/ocean/legoesm/ocean/state.py-370-    - eta, T, S, H_bathy, land_mask: shape (n_lat, n_lon [, nlev])
--
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-6-import jax.numpy as jnp
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-7-import numpy as np
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-8-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-9-from legoesm.core.field import Field
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-10-from legoesm.core.precision import get_policy
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-11-from legoesm.grids.latlon import LatLonGrid
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-12-from legoesm.ocean.eos import scale_depth as _SCALE_DEPTH
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-13-from legoesm.ocean.vertical import OceanZStarCoordinate
packages/ocean/legoesm/ocean/init_latlon_cgrid.py:14:from legoesm.ocean.state import LatLonCGridOceanState
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-15-from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-16-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-17-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-18-def partial_periodic_seam_wall_latlon(
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-19-    grid: LatLonGrid,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-20-    open_lat_south_deg: float,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-21-    open_lat_north_deg: float,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-22-    seam_column_index: int = 0,
--
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-108-    T_water_init_C: float = 20.0,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-109-    T_deep: float = 2.0,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-110-    S_uniform: float = 35.0,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-111-    H_max: float = 5500.0,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-112-    land_lat_threshold: float = 80.0,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-113-    land_mask_override: jnp.ndarray | None = None,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-114-    H_bathy_override: jnp.ndarray | None = None,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-115-    stratification: str = "exponential",
packages/ocean/legoesm/ocean/init_latlon_cgrid.py:116:) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-117-    """Create a rest-state initial condition on a C-grid lat-lon grid.
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-118-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-119-    Temperature: exponential or linear profile (see ``stratification``).
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-120-    Salinity: uniform.
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-121-    Velocity: zero.
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-122-    Eta: zero.
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-123-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-124-    Parameters
--
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-155-          i.e. T varies linearly from ``T_water_init_C`` at the surface to
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-156-          ``T_deep`` at the deepest interface ``z = z_bottom``. This matches
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-157-          Veros ACC's ``T = (1 - z / z_bottom) * 15`` initial condition
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-158-          (``veros/setups/acc/acc.py:117``; ``T_deep=0`` there), which the
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-159-          exponential profile leaves ~1 °C too warm at the deepest cell.
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-160-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-161-    Returns
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-162-    -------
packages/ocean/legoesm/ocean/init_latlon_cgrid.py:163:    LatLonCGridOceanState
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-164-    """
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-165-    n_lat = grid.n_lat
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-166-    n_lon = grid.n_lon
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-167-    nlev = z_coord.n_levels
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-168-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-169-    # Cast bathymetry/mask inputs to the active precision policy so that a
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-170-    # caller running under x32 does not silently get x64 fields (codex
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-171-    # adversarial review iter-1, bug #6).
--
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-242-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-243-    dims_u = ("lat", "lon_u", "level")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-244-    dims_v = ("lat_v", "lon", "level")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-245-    dims_3d = ("lat", "lon", "level")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-246-    dims_2d = ("lat", "lon")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-247-    dims_u2d = ("lat", "lon_u")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-248-    dims_v2d = ("lat_v", "lon")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-249-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py:250:    return LatLonCGridOceanState(
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-251-        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-252-                staggering="edge"),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-253-        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-254-                staggering="edge"),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-255-        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-256-        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-257-        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-258-        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
--
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-268-    z_coord: OceanZStarCoordinate,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-269-    H_max: float = 5500.0,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-270-    lon_west: float = 0.0,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-271-    lon_east: float = 120.0,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-272-    lat_south: float = 15.0,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-273-    lat_north: float = 75.0,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-274-    T_uniform: float = 10.0,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-275-    S_uniform: float = 35.0,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py:276:) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-277-    """Create initial condition for a wind-driven barotropic gyre on C-grid lat-lon.
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-278-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-279-    Uniform T and S inside a rectangular basin. Purely barotropic setup.
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-280-    """
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-281-    n_lat = grid.n_lat
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-282-    n_lon = grid.n_lon
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-283-    nlev = z_coord.n_levels
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-284-    dtype = get_policy().storage
--
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-306-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-307-    dims_u = ("lat", "lon_u", "level")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-308-    dims_v = ("lat_v", "lon", "level")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-309-    dims_3d = ("lat", "lon", "level")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-310-    dims_2d = ("lat", "lon")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-311-    dims_u2d = ("lat", "lon_u")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-312-    dims_v2d = ("lat_v", "lon")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-313-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py:314:    return LatLonCGridOceanState(
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-315-        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-316-                staggering="edge"),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-317-        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-318-                staggering="edge"),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-319-        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-320-        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-321-        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-322-        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-323-        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-324-        u_mask=Field(data=u_mask, name="u_mask", dims=dims_u2d, units=""),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-325-        v_mask=Field(data=v_mask, name="v_mask", dims=dims_v2d, units=""),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-326-        w=Field(data=w_zeros, name="w", dims=dims_3d, units="m/s"),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-327-    )
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-328-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-329-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-330-def replace_land_mask(
packages/ocean/legoesm/ocean/init_latlon_cgrid.py:331:    state: LatLonCGridOceanState,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-332-    new_land_mask: jnp.ndarray,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-333-    seam_wall_rows: jnp.ndarray | None = None,
packages/ocean/legoesm/ocean/init_latlon_cgrid.py:334:) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-335-    """Replace land_mask and recompute u_mask/v_mask atomically.
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-336-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-337-    Use this instead of ``state._replace(land_mask=...)`` to ensure
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-338-    face masks stay consistent with the cell mask.  ``seam_wall_rows``
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-339-    (optional, ``(n_lat,)``, 1 = walled) closes the periodic-seam u-face
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-340-    on those rows for a partial-periodic geometry; ``None`` (default) =
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-341-    fully periodic (byte-identical).
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-342-    """
--
packages/ocean/legoesm/ocean/state.py-354-    eos: str = "wright"
packages/ocean/legoesm/ocean/state.py-355-    eos_linear: object = None
packages/ocean/legoesm/ocean/state.py-356-
packages/ocean/legoesm/ocean/state.py-357-
packages/ocean/legoesm/ocean/state.py-358-# ==============================================================================
packages/ocean/legoesm/ocean/state.py-359-# Lat-Lon C-Grid FV Ocean State
packages/ocean/legoesm/ocean/state.py-360-# ==============================================================================
packages/ocean/legoesm/ocean/state.py-361-
packages/ocean/legoesm/ocean/state.py:362:class LatLonCGridOceanState(NamedTuple):
packages/ocean/legoesm/ocean/state.py-363-    """State for the lat-lon C-grid finite-volume ocean primitive equations.
packages/ocean/legoesm/ocean/state.py-364-
packages/ocean/legoesm/ocean/state.py-365-    Velocities live on cell faces (Arakawa C-grid staggering):
packages/ocean/legoesm/ocean/state.py-366-    - u at east/west faces (lon interfaces): shape (n_lat, n_lon+1, nlev)
packages/ocean/legoesm/ocean/state.py-367-    - v at north/south faces (lat interfaces): shape (n_lat+1, n_lon, nlev)
packages/ocean/legoesm/ocean/state.py-368-
packages/ocean/legoesm/ocean/state.py-369-    Scalars live at cell centers:
packages/ocean/legoesm/ocean/state.py-370-    - eta, T, S, H_bathy, land_mask: shape (n_lat, n_lon [, nlev])
--
packages/ocean/legoesm/ocean/bathymetry.py-1468-    raise TypeError(f"Unsupported grid type: {type(grid)}")
packages/ocean/legoesm/ocean/bathymetry.py-1469-
packages/ocean/legoesm/ocean/bathymetry.py-1470-
packages/ocean/legoesm/ocean/bathymetry.py-1471-def _rest_state_latlon_cgrid(grid, z_coord, H_bathy, ocean_mask,
packages/ocean/legoesm/ocean/bathymetry.py-1472-                              T_water_init_C, T_deep, S_uniform):
packages/ocean/legoesm/ocean/bathymetry.py-1473-    """Create lat-lon C-grid ocean rest state with given bathymetry.
packages/ocean/legoesm/ocean/bathymetry.py-1474-
packages/ocean/legoesm/ocean/bathymetry.py-1475-    Delegates to ``rest_state_latlon_cgrid_ocean`` (which handles the
packages/ocean/legoesm/ocean/bathymetry.py:1476:    full LatLonCGridOceanState construction including face masks and
packages/ocean/legoesm/ocean/bathymetry.py-1477-    z* metadata) by passing the loaded bathymetry + mask via the
packages/ocean/legoesm/ocean/bathymetry.py-1478-    *_override* parameters.
packages/ocean/legoesm/ocean/bathymetry.py-1479-    """
packages/ocean/legoesm/ocean/bathymetry.py-1480-    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
packages/ocean/legoesm/ocean/bathymetry.py-1481-    return rest_state_latlon_cgrid_ocean(
packages/ocean/legoesm/ocean/bathymetry.py-1482-        grid, z_coord,
packages/ocean/legoesm/ocean/bathymetry.py-1483-        T_water_init_C=T_water_init_C,
packages/ocean/legoesm/ocean/bathymetry.py-1484-        T_deep=T_deep,
--
packages/ocean/legoesm/ocean/__init__.py-88-    "init_biogeo_state": ("legoesm.ocean.biogeochemistry", "init_biogeo_state"),
packages/ocean/legoesm/ocean/__init__.py-89-    "step_ocean_biogeochemistry": ("legoesm.ocean.biogeochemistry", "step_ocean_biogeochemistry"),
packages/ocean/legoesm/ocean/__init__.py-90-    "compute_biogeo_tendencies": ("legoesm.ocean.biogeochemistry", "compute_biogeo_tendencies"),
packages/ocean/legoesm/ocean/__init__.py-91-    "air_sea_co2_flux": ("legoesm.ocean.biogeochemistry", "air_sea_co2_flux"),
packages/ocean/legoesm/ocean/__init__.py-92-    "solve_carbonate_system": ("legoesm.ocean.biogeochemistry", "solve_carbonate_system"),
packages/ocean/legoesm/ocean/__init__.py-93-    # Lat-lon FV ocean
packages/ocean/legoesm/ocean/__init__.py-94-    # Lat-lon C-grid FV ocean
packages/ocean/legoesm/ocean/__init__.py-95-    "LatLonCGridOceanModel": ("legoesm.ocean.dynamics.ocean_model_latlon_cgrid", "LatLonCGridOceanModel"),
packages/ocean/legoesm/ocean/__init__.py:96:    "LatLonCGridOceanState": ("legoesm.ocean.state", "LatLonCGridOceanState"),
packages/ocean/legoesm/ocean/__init__.py-97-    "LatLonCGridOceanTendencies": ("legoesm.ocean.state", "LatLonCGridOceanTendencies"),
packages/ocean/legoesm/ocean/__init__.py-98-    "LatLonCGridOceanConfig": ("legoesm.ocean.state", "LatLonCGridOceanConfig"),
packages/ocean/legoesm/ocean/__init__.py-99-    "rest_state_latlon_cgrid_ocean": ("legoesm.ocean.init_latlon_cgrid", "rest_state_latlon_cgrid_ocean"),
packages/ocean/legoesm/ocean/__init__.py-100-    # NEMO eORCA tripole geometry + WOA IC loaders
packages/ocean/legoesm/ocean/__init__.py-101-    "read_mesh_mask_bathy": ("legoesm.ocean.init_tripole", "read_mesh_mask_bathy"),
packages/ocean/legoesm/ocean/__init__.py-102-    "compute_woa_3d": ("legoesm.ocean.init_tripole", "compute_woa_3d"),
packages/ocean/legoesm/ocean/__init__.py-103-    "squeeze_nemo_field_2d": ("legoesm.ocean.init_tripole", "squeeze_nemo_field_2d"),
packages/ocean/legoesm/ocean/__init__.py-104-    # Bathymetry
--
packages/ocean/legoesm/ocean/__init__.py-167-    "load_bathymetry_cubed_sphere",
packages/ocean/legoesm/ocean/__init__.py-168-    "load_bathymetry_mpas",
packages/ocean/legoesm/ocean/__init__.py-169-    "load_bathymetry_gaussian",
packages/ocean/legoesm/ocean/__init__.py-170-    "rest_state_ocean_realistic",
packages/ocean/legoesm/ocean/__init__.py-171-    "enforce_straits",
packages/ocean/legoesm/ocean/__init__.py-172-    # Lat-lon FV ocean
packages/ocean/legoesm/ocean/__init__.py-173-    # Lat-lon C-grid FV ocean
packages/ocean/legoesm/ocean/__init__.py-174-    "LatLonCGridOceanModel",
packages/ocean/legoesm/ocean/__init__.py:175:    "LatLonCGridOceanState",
packages/ocean/legoesm/ocean/__init__.py-176-    "LatLonCGridOceanTendencies",
packages/ocean/legoesm/ocean/__init__.py-177-    "LatLonCGridOceanConfig",
packages/ocean/legoesm/ocean/__init__.py-178-    "rest_state_latlon_cgrid_ocean",
packages/ocean/legoesm/ocean/__init__.py-179-    # NEMO eORCA tripole geometry + WOA IC loaders
packages/ocean/legoesm/ocean/__init__.py-180-    "read_mesh_mask_bathy",
packages/ocean/legoesm/ocean/__init__.py-181-    "compute_woa_3d",
packages/ocean/legoesm/ocean/__init__.py-182-    "squeeze_nemo_field_2d",
packages/ocean/legoesm/ocean/__init__.py-183-    # Freshwater
--
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-132-    lat-lon C-grid state (``state.T``/``state.S`` shape ``(n_lat, n_lon, nlev)``,
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-133-    ``state.land_mask`` ``(n_lat, n_lon)``) AND on an MPAS state (``(nCells,
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-134-    nlev)`` / ``(nCells,)``). The only requirement is that ``K_tidal``,
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-135-    ``h_partial`` and the state arrays share the same ``(*spatial, nlev)`` shape.
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-136-    Land cells (``land_mask=0``) are masked back to their pre-step values.
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-137-
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-138-    Parameters
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-139-    ----------
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py:140:    state : LatLonCGridOceanState or an MPAS ocean state
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-141-    K_tidal : array ``(*spatial, nlev)``
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-142-        Cell-centred tidal diffusivity [m²/s] from
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-143-        :func:`legoesm.ocean.physics.vertical_mixing.tidal.compute_tidal_diffusivity`
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-144-        (itself grid-agnostic).
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-145-    h_partial : array ``(*spatial, nlev)``
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-146-        Layer thickness [m] from
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-147-        :func:`legoesm.ocean.vertical.compute_layer_thickness`.
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py-148-    dt : float
--
packages/ocean/legoesm/ocean/conservation.py-24-import jax.numpy as jnp
packages/ocean/legoesm/ocean/conservation.py-25-
packages/ocean/legoesm/ocean/conservation.py-26-from typing import Callable, Union
packages/ocean/legoesm/ocean/conservation.py-27-
packages/ocean/legoesm/ocean/conservation.py-28-from legoesm.core.operators import is_distributed
packages/ocean/legoesm/ocean/conservation.py-29-from legoesm.core.precision import cast
packages/ocean/legoesm/ocean/conservation.py-30-from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
packages/ocean/legoesm/ocean/conservation.py-31-from legoesm.ocean.state import (
packages/ocean/legoesm/ocean/conservation.py:32:    OceanState, OceanConfig, LatLonCGridOceanState, LatLonCGridOceanConfig,
packages/ocean/legoesm/ocean/conservation.py-33-)
packages/ocean/legoesm/ocean/conservation.py-34-from legoesm.grids.cubed_sphere import CubedSphereGrid
packages/ocean/legoesm/ocean/conservation.py-35-from legoesm.grids.latlon import LatLonGrid
packages/ocean/legoesm/ocean/conservation.py-36-from legoesm.parallel.reductions import global_sum_mpi
packages/ocean/legoesm/ocean/conservation.py-37-
packages/ocean/legoesm/ocean/conservation.py-38-# Types accepted by the conservation fixer (cubed-sphere + lat-lon C-grid)
packages/ocean/legoesm/ocean/conservation.py:39:_OceanStateT = Union[OceanState, LatLonCGridOceanState]
packages/ocean/legoesm/ocean/conservation.py-40-_OceanConfigT = Union[OceanConfig, LatLonCGridOceanConfig]
packages/ocean/legoesm/ocean/conservation.py-41-_GridT = Union[CubedSphereGrid, LatLonGrid]
packages/ocean/legoesm/ocean/conservation.py-42-
packages/ocean/legoesm/ocean/conservation.py-43-
packages/ocean/legoesm/ocean/conservation.py-44-def ocean_global_sum(local_value):
packages/ocean/legoesm/ocean/conservation.py-45-    """MPI- AND SPMD-aware global sum for scalar or vector reductions.
packages/ocean/legoesm/ocean/conservation.py-46-
packages/ocean/legoesm/ocean/conservation.py-47-    NOTE: the MPI gate keys on ``is_distributed()`` (the mpi4jax/sharded flag),
--
packages/ocean/legoesm/ocean/coupler/sss_apply.py-10-with ``τ_eff`` carrying the region masks + ice gating.  The
packages/ocean/legoesm/ocean/coupler/sss_apply.py-11-helper applies the discrete step
packages/ocean/legoesm/ocean/coupler/sss_apply.py-12-
packages/ocean/legoesm/ocean/coupler/sss_apply.py-13-    S_top_new = S_top_old + dt · dS_top/dt
packages/ocean/legoesm/ocean/coupler/sss_apply.py-14-
packages/ocean/legoesm/ocean/coupler/sss_apply.py-15-clipped to the land mask so dry cells are untouched.
packages/ocean/legoesm/ocean/coupler/sss_apply.py-16-
packages/ocean/legoesm/ocean/coupler/sss_apply.py-17-Currently supports the lat-lon C-grid ocean state
packages/ocean/legoesm/ocean/coupler/sss_apply.py:18:(``LatLonCGridOceanState``).  Other grids may use the standalone
packages/ocean/legoesm/ocean/coupler/sss_apply.py-19-``compute_sss_restoring_flux`` directly and assemble the dS
packages/ocean/legoesm/ocean/coupler/sss_apply.py-20-themselves.
packages/ocean/legoesm/ocean/coupler/sss_apply.py-21-"""
packages/ocean/legoesm/ocean/coupler/sss_apply.py-22-
packages/ocean/legoesm/ocean/coupler/sss_apply.py-23-from __future__ import annotations
packages/ocean/legoesm/ocean/coupler/sss_apply.py-24-
packages/ocean/legoesm/ocean/coupler/sss_apply.py-25-import jax.numpy as jnp
packages/ocean/legoesm/ocean/coupler/sss_apply.py-26-import numpy as np
--
packages/ocean/legoesm/ocean/coupler/sss_apply.py-52-        S_top_new = S_top_old + dt · dS/dt|_restore
packages/ocean/legoesm/ocean/coupler/sss_apply.py-53-
packages/ocean/legoesm/ocean/coupler/sss_apply.py-54-    where ``dS/dt|_restore`` comes from
packages/ocean/legoesm/ocean/coupler/sss_apply.py-55-    :func:`compute_sss_restoring_flux`.  Land cells (``land_mask=0``)
packages/ocean/legoesm/ocean/coupler/sss_apply.py-56-    are left untouched.
packages/ocean/legoesm/ocean/coupler/sss_apply.py-57-
packages/ocean/legoesm/ocean/coupler/sss_apply.py-58-    Parameters
packages/ocean/legoesm/ocean/coupler/sss_apply.py-59-    ----------
packages/ocean/legoesm/ocean/coupler/sss_apply.py:60:    state : LatLonCGridOceanState
packages/ocean/legoesm/ocean/coupler/sss_apply.py-61-        Current ocean state (lat-lon C-grid).  ``state.S`` is a
packages/ocean/legoesm/ocean/coupler/sss_apply.py-62-        ``Field`` with shape ``(n_lat, n_lon, nlev)``.
packages/ocean/legoesm/ocean/coupler/sss_apply.py-63-    S_target : array ``(n_lat, n_lon)``
packages/ocean/legoesm/ocean/coupler/sss_apply.py-64-        Climatological target SSS interpolated to the grid [PSU].
packages/ocean/legoesm/ocean/coupler/sss_apply.py-65-    ice_concentration : array ``(n_lat, n_lon)`` or None
packages/ocean/legoesm/ocean/coupler/sss_apply.py-66-        Cell ice fraction in [0, 1].  When ``None``, restoring is
packages/ocean/legoesm/ocean/coupler/sss_apply.py-67-        applied everywhere without ice gating — appropriate for
packages/ocean/legoesm/ocean/coupler/sss_apply.py-68-        ocean-only experiments without a sea-ice tile.
--
packages/ocean/legoesm/ocean/coupler/sss_apply.py-74-        region-mask builder.
packages/ocean/legoesm/ocean/coupler/sss_apply.py-75-    z_coord : OceanZStarCoordinate / OceanPartialCellCoordinate
packages/ocean/legoesm/ocean/coupler/sss_apply.py-76-        Used to read ``dz_ref[0]`` for the surface-layer thickness.
packages/ocean/legoesm/ocean/coupler/sss_apply.py-77-    dt : float
packages/ocean/legoesm/ocean/coupler/sss_apply.py-78-        Time step [s].
packages/ocean/legoesm/ocean/coupler/sss_apply.py-79-
packages/ocean/legoesm/ocean/coupler/sss_apply.py-80-    Returns
packages/ocean/legoesm/ocean/coupler/sss_apply.py-81-    -------
packages/ocean/legoesm/ocean/coupler/sss_apply.py:82:    new_state : LatLonCGridOceanState
packages/ocean/legoesm/ocean/coupler/sss_apply.py-83-        Same state with the top-layer salinity updated.
packages/ocean/legoesm/ocean/coupler/sss_apply.py-84-    """
packages/ocean/legoesm/ocean/coupler/sss_apply.py-85-    if not config.enabled:
packages/ocean/legoesm/ocean/coupler/sss_apply.py-86-        return state
packages/ocean/legoesm/ocean/coupler/sss_apply.py-87-
packages/ocean/legoesm/ocean/coupler/sss_apply.py-88-    # Surface salinity from the existing state.  Slice FIRST (device-side),
packages/ocean/legoesm/ocean/coupler/sss_apply.py-89-    # THEN convert: np.asarray on the full leaf would assemble the entire
packages/ocean/legoesm/ocean/coupler/sss_apply.py-90-    # 3-D (possibly lat-band-sharded) S field on the host every step (codex
--
packages/ocean/legoesm/ocean/coupler/__init__.py-1-"""Ocean-coupler surface-flux applicators.
packages/ocean/legoesm/ocean/coupler/__init__.py-2-
packages/ocean/legoesm/ocean/coupler/__init__.py-3-The legoESM ocean dycores expose generic prognostic state containers
packages/ocean/legoesm/ocean/coupler/__init__.py:4:(``OceanState`` cube, ``LatLonCGridOceanState``, ``MPASOceanState``);
packages/ocean/legoesm/ocean/coupler/__init__.py-5-this module ports atmospheric / land flux output into those state
packages/ocean/legoesm/ocean/coupler/__init__.py-6-containers as explicit per-timestep top-layer tendencies. The Phase F
packages/ocean/legoesm/ocean/coupler/__init__.py-7-climate-scale drivers use ``omip2_applicator`` to wire JRA55-do
packages/ocean/legoesm/ocean/coupler/__init__.py-8-forcing + L&Y 2009 bulk fluxes into the ocean state.
packages/ocean/legoesm/ocean/coupler/__init__.py-9-"""
packages/ocean/legoesm/ocean/coupler/__init__.py-10-
packages/ocean/legoesm/ocean/coupler/__init__.py-11-from .omip2_applicator import (
packages/ocean/legoesm/ocean/coupler/__init__.py-12-    apply_omip2_surface_fluxes,
--
packages/ocean/legoesm/ocean/restart.py-3-Saves the full ocean prognostic state to disk as a numpy ``.npz``
packages/ocean/legoesm/ocean/restart.py-4-archive and loads it back into an empty state container of the
packages/ocean/legoesm/ocean/restart.py-5-matching grid. Round-trip preserves every prognostic field to
packages/ocean/legoesm/ocean/restart.py-6-bit-exact precision (``np.array_equal``) so a model started from a
packages/ocean/legoesm/ocean/restart.py-7-restart steps to bit-identical results vs. an uninterrupted run.
packages/ocean/legoesm/ocean/restart.py-8-
packages/ocean/legoesm/ocean/restart.py-9-Currently supports the three production grids:
packages/ocean/legoesm/ocean/restart.py-10-
packages/ocean/legoesm/ocean/restart.py:11:* lat-lon C-grid (``LatLonCGridOceanState``)
packages/ocean/legoesm/ocean/restart.py-12-* MPAS Voronoi (``MPASOceanState`` -- accessed via duck-typing on
packages/ocean/legoesm/ocean/restart.py-13-  the ``state.u.dims`` shape)
packages/ocean/legoesm/ocean/restart.py-14-* cubed-sphere (``OceanState``)
packages/ocean/legoesm/ocean/restart.py-15-
packages/ocean/legoesm/ocean/restart.py-16-The serialised payload is a dict-of-numpy-arrays keyed by field
packages/ocean/legoesm/ocean/restart.py-17-name; auxiliary metadata (model time in seconds, step index, source
packages/ocean/legoesm/ocean/restart.py-18-sha) is stored under reserved underscore keys to keep field names
packages/ocean/legoesm/ocean/restart.py-19-clean.
--
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-4-(``legoesm.ocean.bulk_flux_omip``) and applies the resulting
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-5-``(tau_x, tau_y, shflx, lhflx)`` to the ocean state's top layer as a
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-6-per-timestep forward-Euler update of u / v / T. Designed to be called
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-7-once per model step from the long-run drivers.
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-8-
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-9-State conventions handled
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-10--------------------------
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-11-
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py:12:* Lat-lon C-grid (``LatLonCGridOceanState``): ``u`` on east-west
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-13-  faces (shape ``(n_lat, n_lon+1, n_z)``), ``v`` on north-south
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-14-  faces (shape ``(n_lat+1, n_lon, n_z)``). The applicator
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-15-  interpolates cell-centred tau_x / tau_y to the matching faces
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-16-  via the C-grid 0.5*(left+right) stencil already used by the
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-17-  prescribed-forcing path.
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-18-* MPAS Voronoi: ``u`` on edges; ``tau_normal`` = tau_x*cos(angleEdge)
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-19-  + tau_y*sin(angleEdge).
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-20-* Cube C-D grid: ``u``, ``v`` collocated on cells; direct
--
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-303-
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-304-def _apply_cgrid_surface_fluxes(state, forc, *, dz_0, rho_0, c_p,
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-305-                                rho_air, sigma_sb, dt, grid=None):
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-306-    """Apply bulk fluxes to a lat-lon C-grid state from already-sampled forcing.
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-307-
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-308-    Shared by the ``latlon`` / ``latlon_regional`` path (forcing conservatively
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-309-    regridded onto the regular T grid) and the curvilinear ``tripole`` path
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-310-    (forcing nearest-neighbour sampled onto the 2-D T grid) -- both use the
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py:311:    identical ``LatLonCGridOceanState`` layout (T cell-centred; u on EW faces
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-312-    ``(n_lat, n_lon+1)``; v on NS faces ``(n_lat+1, n_lon)``). ``forc`` holds
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-313-    the seven channels as 2-D ``(n_lat, n_lon)`` arrays. Forward-Euler update of
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-314-    the top layer of T, u, v; land masking via state masks.
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-315-
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-316-    ``grid`` (optional) supplies ``cos_alpha_u`` / ``sin_alpha_u`` /
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-317-    ``cos_alpha_v`` / ``sin_alpha_v`` for the geographic->grid wind-stress
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-318-    rotation on curvilinear (tripole) geometries; absent on a regular lat-lon
packages/ocean/legoesm/ocean/coupler/omip2_applicator.py-319-    grid, where the rotation is the identity.
--
packages/ocean/legoesm/ocean/experiments/dino.py-2197-
packages/ocean/legoesm/ocean/experiments/dino.py-2198-
packages/ocean/legoesm/ocean/experiments/dino.py-2199-def dino_lat_lon_state(
packages/ocean/legoesm/ocean/experiments/dino.py-2200-    grid,
packages/ocean/legoesm/ocean/experiments/dino.py-2201-    z_coord: OceanZStarCoordinate,
packages/ocean/legoesm/ocean/experiments/dino.py-2202-    cfg: DINOConfig | None = None,
packages/ocean/legoesm/ocean/experiments/dino.py-2203-    land_mask_override=None,
packages/ocean/legoesm/ocean/experiments/dino.py-2204-):
packages/ocean/legoesm/ocean/experiments/dino.py:2205:    """Build a full ``LatLonCGridOceanState`` for DINO from rest with
packages/ocean/legoesm/ocean/experiments/dino.py-2206-    paper IC stratification + bathymetry + seam-wall land mask.
packages/ocean/legoesm/ocean/experiments/dino.py-2207-
packages/ocean/legoesm/ocean/experiments/dino.py-2208-    ``land_mask_override`` (opt-in) makes the domain i-periodic/
packages/ocean/legoesm/ocean/experiments/dino.py-2209-    re-entrant (NEMO ``ln_Iperio``) by supplying NEMO's own surface
packages/ocean/legoesm/ocean/experiments/dino.py-2210-    ``tmask`` instead of the analytic seam wall — the fix for the
packages/ocean/legoesm/ocean/experiments/dino.py-2211-    spurious equatorial land wall that trapped the deep-equatorial jet.
packages/ocean/legoesm/ocean/experiments/dino.py-2212-    ``rest_state_latlon_cgrid_ocean`` recomputes u_mask/v_mask atomically
packages/ocean/legoesm/ocean/experiments/dino.py-2213-    from it (periodic in lon, walled N/S), avoiding the stale-face-mask
packages/ocean/legoesm/ocean/experiments/dino.py-2214-    leak footgun.  Default ``None`` = the analytic seam-wall mask
packages/ocean/legoesm/ocean/experiments/dino.py-2215-    (byte-identical for the standalone bowl recipes).
packages/ocean/legoesm/ocean/experiments/dino.py-2216-
packages/ocean/legoesm/ocean/experiments/dino.py-2217-    Returns
packages/ocean/legoesm/ocean/experiments/dino.py-2218-    -------
packages/ocean/legoesm/ocean/experiments/dino.py:2219:    LatLonCGridOceanState
packages/ocean/legoesm/ocean/experiments/dino.py-2220-    """
packages/ocean/legoesm/ocean/experiments/dino.py-2221-    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
packages/ocean/legoesm/ocean/experiments/dino.py-2222-
packages/ocean/legoesm/ocean/experiments/dino.py-2223-    if cfg is None:
packages/ocean/legoesm/ocean/experiments/dino.py-2224-        cfg = DINOConfig()
packages/ocean/legoesm/ocean/experiments/dino.py-2225-
packages/ocean/legoesm/ocean/experiments/dino.py-2226-    T, S, H_bathy, land_mask = dino_lat_lon_initial_state_arrays(
packages/ocean/legoesm/ocean/experiments/dino.py-2227-        grid, z_coord, cfg, land_mask_override=land_mask_override,
--
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-67-    """Apply one step of surface T/S relaxation to a lat-lon C-grid ocean state.
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-68-
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-69-    Updates only the top model level (index 0), land-masked via
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-70-    ``state.land_mask``.  A non-positive ``tau`` disables that tracer's
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-71-    relaxation (``alpha = 0`` => unchanged).  Pure / AD-safe.
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-72-
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-73-    Parameters
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-74-    ----------
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py:75:    state : LatLonCGridOceanState
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-76-        ``state.T`` / ``state.S`` are ``Field`` s of shape
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-77-        ``(n_lat, n_lon, nlev)``; ``state.land_mask`` is ``(n_lat, n_lon)``.
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-78-    T_target, S_target : array ``(n_lat, n_lon)``
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-79-        Climatological surface target (T in the state's units [degC], S [PSU]).
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-80-    dt : float
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-81-        Coupling step [s].
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-82-    tau_T_s, tau_S_s : float
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-83-        Relaxation timescales [s] (<= 0 => that tracer is not relaxed).
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-84-
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-85-    Returns
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-86-    -------
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py:87:    new_state : LatLonCGridOceanState
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-88-    """
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-89-    mask = state.land_mask.data
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-90-    alpha_T = 0.0 if tau_T_s <= 0.0 else dt / tau_T_s
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-91-    alpha_S = 0.0 if tau_S_s <= 0.0 else dt / tau_S_s
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-92-    if alpha_T == 0.0 and alpha_S == 0.0:
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-93-        return state
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-94-    T = state.T.data
packages/ocean/legoesm/ocean/forcing/surface_relaxation.py-95-    S = state.S.data
--
packages/ocean/legoesm/ocean/physics/bbl_adv.py-318-    Operator-split with the dynamics exactly like NEMO applies trabbl within
packages/ocean/legoesm/ocean/physics/bbl_adv.py-319-    its sequential tracer trends.  Stability: the exchange is a bounded
packages/ocean/legoesm/ocean/physics/bbl_adv.py-320-    relaxation between cells; with NEMO's gamma=20 s and 1-deg cells the
packages/ocean/legoesm/ocean/physics/bbl_adv.py-321-    per-step exchange fraction ``|tr|*dt/V`` is << 1 at any ocean dt (see the
packages/ocean/legoesm/ocean/physics/bbl_adv.py-322-    unit test's magnitude check).
packages/ocean/legoesm/ocean/physics/bbl_adv.py-323-
packages/ocean/legoesm/ocean/physics/bbl_adv.py-324-    Parameters
packages/ocean/legoesm/ocean/physics/bbl_adv.py-325-    ----------
packages/ocean/legoesm/ocean/physics/bbl_adv.py:326:    state : LatLonCGridOceanState-like (T, S Fields with (..., nlev) data)
packages/ocean/legoesm/ocean/physics/bbl_adv.py-327-    geom : BBLGeometry from :func:`bbl_static_geometry` (STATIC, build once)
packages/ocean/legoesm/ocean/physics/bbl_adv.py-328-    dy_u_faces : (n_lat, n_lon-1) i-face widths [m] (NEMO e2u at the face)
packages/ocean/legoesm/ocean/physics/bbl_adv.py-329-    dx_v_faces : (n_lat-1, n_lon) j-face widths [m] (NEMO e1v)
packages/ocean/legoesm/ocean/physics/bbl_adv.py-330-    nlev : static Python int.
packages/ocean/legoesm/ocean/physics/bbl_adv.py-331-
packages/ocean/legoesm/ocean/physics/bbl_adv.py-332-    Returns the state with T, S updated.
packages/ocean/legoesm/ocean/physics/bbl_adv.py-333-    """
packages/ocean/legoesm/ocean/physics/bbl_adv.py-334-    T = jnp.asarray(state.T.data, dtype=jnp.float64)
--
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-91-    * Freshwater into ocean: ``F_FW = ρ_ice · ṁ`` [kg/m²/s] adds to
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-92-      ``η`` (``dη = F_FW · dt / ρ_0``) and dilutes top-layer S via
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-93-      virtual-salt.
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-94-    * Heat extracted from ocean: ``Q = ρ_w · c_w · γ_T · (T_a − T_b)``
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-95-      cools the top-layer T by ``dT = −Q · dt / (ρ_0 · c_p · dz_0)``.
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-96-
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-97-    Parameters
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-98-    ----------
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py:99:    state : LatLonCGridOceanState
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-100-    ice_shelf_mask : array ``(n_lat, n_lon)`` of {0, 1}
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-101-        1 = ice-shelf cavity cell.  Land + open-water cells = 0.
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-102-    ice_draft_m : array ``(n_lat, n_lon)``
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-103-        Depth of the ice-shelf base [m, positive down].  Used for the
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-104-        in-situ freezing point ``T_f(S, p)``.
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-105-    z_coord : OceanZStarCoordinate / OceanPartialCellCoordinate
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-106-    dt : float
packages/ocean/legoesm/ocean/coupler/ice_shelf_apply.py-107-        Time step [s].
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-44-    LatLonCGridGeometry,
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-45-    create_beta_plane_cgrid_geometry,
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-46-)
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-47-from legoesm.ocean.eos import LinearEOSConfig
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-48-from legoesm.ocean.fidelity.mitgcm_recipe import mitgcm_canonical_ocean_config
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-49-from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-50-from legoesm.ocean.state import (
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-51-    LatLonCGridOceanConfig,
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py:52:    LatLonCGridOceanState,
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-53-    OceanSurfaceForcing,
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-54-)
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-55-from legoesm.ocean.vertical import create_ocean_z_star
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-56-
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-57-# --- MITgcm tutorial_barotropic_gyre parameters (input/data + gendata.m). ----
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-58-NX = 62
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-59-NY = 62
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-60-DX_M = 20.0e3
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-72-
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-73-
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-74-class MitgcmGyreRecipe(NamedTuple):
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-75-    """Assembled legoESM reproduction of the MITgcm barotropic gyre."""
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-76-
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-77-    geometry: LatLonCGridGeometry
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-78-    z_coord: object
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-79-    config: LatLonCGridOceanConfig
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py:80:    state: LatLonCGridOceanState
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-81-    wind_forcing: OceanSurfaceForcing
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-82-    dt_s: float
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-83-
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-84-
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-85-def build_gyre_geometry() -> LatLonCGridGeometry:
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-86-    """Cartesian beta-plane C-grid matching MITgcm's grid + Coriolis."""
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-87-    return create_beta_plane_cgrid_geometry(
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-88-        NY, NX, dx_m=DX_M, dy_m=DY_M, f0=F0, beta=BETA,
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-198-        # MITgcm fully-backward-Euler implicit free surface (the card pins theta=1).
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-199-        # Solver is a parameter: the per-tendency oracle tier validates implicit_cn.
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-200-        barotropic_solver=barotropic_solver,
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-201-        # MITgcm abEps for this deck.
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-202-        ab2_epsilon=0.01,
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-203-    )
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-204-
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-205-
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py:206:def build_gyre_state(z_coord) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-207-    """Rest state (u=v=eta=0, uniform T/S) on the closed-box beta-plane grid."""
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-208-    geom = build_gyre_geometry()
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-209-    return rest_state_latlon_cgrid_ocean(
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-210-        geom, z_coord,
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-211-        land_mask_override=build_gyre_land_mask(),
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-212-    )
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-213-
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py-214-
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-21-                  (3rd-order DST flux limiter, multi-dim), diffKh = diffK4 =
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-22-                  diffKr = 0 (PURE advection).  Initialized AFTER a 10-year
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-23-                  spin-up (``PTRACERS_Iter0=259200``) as a single cell of
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-24-                  concentration 1 at (i=2, j=30) [1-based Fortran], i.e.
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-25-                  (y=29, x=1) [0-based], everywhere-else zero (``dye.bin``).
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-26-time              deltaT = 1200 s, 4 steps (``nTimeSteps=4``).
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-27-================  ===============================================================
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-28-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py:29:PASSIVE-TRACER CARRIER.  legoESM's ``LatLonCGridOceanState`` does not carry a
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-30-dedicated passive-tracer slot — it carries (T, S).  The dye is therefore carried
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-31-in the **salinity** field with a LINEAR EOS whose haline contraction is ZERO
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-32-(``beta_S = 0``), exactly matching MITgcm's ``sBeta = 0`` here: S is
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-33-dynamically INERT, so it is a true passive tracer.  This is the same convention
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-34-``ocean/experiments/stommel_gyre_tracer.py`` uses (a passive salinity blob) and
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-35-the same ``sBeta = 0`` MITgcm uses in this deck.
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-36-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-37-ADVECTION SCHEME.  MITgcm ``PTRACERS_advScheme=80`` = 3rd-order Direct-Space-Time
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-64-    LatLonCGridGeometry,
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-65-    create_beta_plane_cgrid_geometry,
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-66-)
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-67-from legoesm.ocean.eos import LinearEOSConfig
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-68-from legoesm.ocean.fidelity.mitgcm_recipe import mitgcm_canonical_ocean_config
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-69-from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-70-from legoesm.ocean.state import (
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-71-    LatLonCGridOceanConfig,
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py:72:    LatLonCGridOceanState,
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-73-    OceanSurfaceForcing,
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-74-)
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-75-from legoesm.ocean.vertical import create_ocean_z_star
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-76-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-77-# --- MITgcm tutorial_advection_in_gyre parameters (input/data, .ptracers). ---
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-78-NX = 60                  # delX = 60*20E3
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-79-NY = 60                  # delY = 60*20E3
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-80-DX_M = 20.0e3
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-98-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-99-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-100-class MitgcmAdvectionGyreRecipe(NamedTuple):
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-101-    """Assembled legoESM reproduction of the MITgcm advection-in-gyre case."""
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-102-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-103-    geometry: LatLonCGridGeometry
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-104-    z_coord: object
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-105-    config: LatLonCGridOceanConfig
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py:106:    state: LatLonCGridOceanState
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-107-    wind_forcing: OceanSurfaceForcing
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-108-    dt_s: float
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-109-    n_steps: int
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-110-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-111-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-112-def build_advgyre_geometry() -> LatLonCGridGeometry:
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-113-    """Cartesian beta-plane C-grid matching MITgcm's grid + Coriolis."""
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-114-    return create_beta_plane_cgrid_geometry(
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-150-    tau_y = jnp.zeros((NY, NX), dtype=tau_x.dtype)
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-151-    return OceanSurfaceForcing(tau_x=tau_x, tau_y=tau_y)
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-152-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-153-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-154-def build_dye_field() -> jnp.ndarray:
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-155-    """Dye blob: single cell of concentration 1 at (y=29, x=1), else 0.
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-156-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-157-    Shape ``(NY, NX, 1)`` to drop straight into the salinity slot of the
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py:158:    single-layer ``LatLonCGridOceanState``.
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-159-    """
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-160-    dye = np.zeros((NY, NX, 1), dtype=np.float64)
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-161-    dye[DYE_Y, DYE_X, 0] = DYE_AMPLITUDE
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-162-    return jnp.asarray(dye)
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-163-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-164-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-165-def build_advgyre_config() -> LatLonCGridOceanConfig:
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-166-    """Single-layer barotropic config + passive-dye advection.
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-193-        tracer_advection="dst3_multidim",
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-194-        # Split implicit_cn free surface from the equilibrated-gyre pickup (the
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-195-        # tracer comparison PRESCRIBES MITgcm's u,v — see module docstring).
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-196-        barotropic_solver="implicit_cn",
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-197-        ab2_epsilon=0.01,
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-198-    )
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-199-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-200-
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py:201:def build_advgyre_state(z_coord) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-202-    """Rest state with the dye blob in the (passive) salinity field."""
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-203-    geom = build_advgyre_geometry()
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-204-    state = rest_state_latlon_cgrid_ocean(
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-205-        geom, z_coord,
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-206-        land_mask_override=build_advgyre_land_mask(),
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-207-    )
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-208-    return state._replace(S=state.S.replace(data=build_dye_field()))
packages/ocean/legoesm/ocean/fidelity/mitgcm_advection_gyre_recipe.py-209-
--
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-38-    dt: float,
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-39-    rho_0: float | None = None,
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-40-) -> object:
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-41-    """Apply one timestep of Dai-Trenberth river runoff to a lat-lon
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-42-    C-grid ocean state.
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-43-
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-44-    Parameters
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-45-    ----------
packages/ocean/legoesm/ocean/coupler/runoff_apply.py:46:    state : LatLonCGridOceanState
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-47-        Must expose ``S``, ``eta``, ``land_mask`` Fields.
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-48-    R_kg_m2_s : array ``(n_lat, n_lon)``
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-49-        Runoff freshwater flux [kg/m²/s, positive INTO ocean].
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-50-        Typically the output of
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-51-        :func:`legoesm.ocean.forcing.dai_trenberth.project_runoff_to_grid`.
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-52-    z_coord : OceanZStarCoordinate / OceanPartialCellCoordinate
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-53-        Surface layer thickness read from ``z_coord.dz_ref[0]``.
packages/ocean/legoesm/ocean/coupler/runoff_apply.py-54-    dt : float
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-59-from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-60-from legoesm.ocean.physics.surface_forcing.config import (
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-61-    RestoringConfig,
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-62-    SurfaceForcingConfig,
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-63-)
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-64-from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-65-from legoesm.ocean.state import (
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-66-    LatLonCGridOceanConfig,
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py:67:    LatLonCGridOceanState,
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-68-    OceanSurfaceForcing,
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-69-)
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-70-from legoesm.ocean.vertical import create_z_star_from_thicknesses
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-71-
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-72-# --- MITgcm tutorial_baroclinic_gyre parameters (input/data + gendata.m). -----
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-73-NX_INT = 60                      # ocean interior cells (62 incl. the wall ring)
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-74-NY_INT = 60
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-75-LAT_SOUTH, LAT_NORTH = 15.0, 75.0    # interior bounds (deg); walls at 14.5 / 75.5
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-98-
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-99-
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-100-class MitgcmBaroclinicGyreRecipe(NamedTuple):
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-101-    """Assembled legoESM reproduction of the MITgcm baroclinic gyre."""
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-102-
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-103-    geometry: LatLonGrid
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-104-    z_coord: object
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-105-    config: LatLonCGridOceanConfig
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py:106:    state: LatLonCGridOceanState
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-107-    wind_forcing: OceanSurfaceForcing
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-108-    land_mask: jnp.ndarray
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-109-    dt_s: float
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-110-
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-111-
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-112-def build_baroclinic_gyre_grid() -> tuple[LatLonGrid, jnp.ndarray]:
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-113-    """Spherical 62x62 closed box (60x60 ocean + 1-cell wall ring) matching the
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-114-    MITgcm grid + spherical Coriolis ``f = 2 Omega sin(lat)``."""
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-200-            bottom_drag=BottomDragConfig(scheme="none"),
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-201-            shortwave_penetration=None,
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-202-        ),
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-203-        # MITgcm abEps=0.1 (== the card default; passed explicitly to document the map).
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-204-        ab2_epsilon=AB_EPS,
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-205-    )
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-206-
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-207-
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py:208:def build_baroclinic_gyre_state(grid, z_coord, land_mask) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-209-    """Rest state with the horizontally-uniform ``tRef`` stratification in T."""
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-210-    base = rest_state_latlon_cgrid_ocean(
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-211-        grid, z_coord, H_max=HO_M, land_mask_override=land_mask)
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-212-    nz = np.asarray(z_coord.z_full_ref).shape[0]
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-213-    t_ref = np.asarray(T_REF_DEGC)[:nz]                       # (nz,)
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-214-    n_lat, n_lon = grid.n_lat, grid.n_lon
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-215-    mask3 = np.asarray(land_mask)[:, :, None]
packages/ocean/legoesm/ocean/fidelity/mitgcm_baroclinic_gyre_recipe.py-216-    temp3d = np.broadcast_to(t_ref[None, None, :], (n_lat, n_lon, nz)) * mask3
--
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-32-from __future__ import annotations
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-33-
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-34-from typing import NamedTuple
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-35-
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-36-import jax.numpy as jnp
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-37-import numpy as np
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-38-from legoesm.ocean.fidelity.concept_registry import oracle_to_canonical
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-39-from legoesm.ocean.fidelity.oceananigans_runner import OceananigansResult
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py:40:from legoesm.ocean.state import LatLonCGridOceanState
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-41-
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-42-# Canonical state fields this bridge places. (b->T is a physics conversion done
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-43-# upstream, so it is not in this set; the bridge only places already-canonical
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-44-# state fields.)
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-45-_PLACEABLE = frozenset({"u", "v", "w", "T", "S", "eta"})
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-46-
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-47-
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-48-class OceananigansStateBridgeOutput(NamedTuple):
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-49-    """Result of :func:`oceananigans_snapshot_to_legoesm_state`."""
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-50-
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py:51:    state: LatLonCGridOceanState
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-52-    info: dict
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-53-
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-54-
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-55-def _to_latlon_lev_zrev(arr: np.ndarray) -> np.ndarray:
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-56-    """``(nz, ny, nx)`` -> ``(ny, nx, nz)`` WITH the vertical reversed;
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-57-    ``(ny, nx)`` passes through unchanged.
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-58-
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-59-    Oceananigans 3-D fields come off disk vertical-first AND bottom-up; legoESM
--
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-120-    raise ValueError(
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-121-        f"v: oracle y-face count {ny_o} is neither target_ny={target_ny} nor "
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-122-        f"target_ny-1={target_ny - 1}"
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-123-    )
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-124-
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-125-
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-126-def oceananigans_snapshot_to_legoesm_state(
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-127-    result: OceananigansResult,
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py:128:    base_state: LatLonCGridOceanState,
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-129-    *,
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-130-    time_index: int = -1,
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-131-    cyclic_x: bool | None = None,
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-132-) -> OceananigansStateBridgeOutput:
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-133-    """Initialise a legoESM lat-lon C-grid state from an Oceananigans snapshot.
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-134-
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-135-    Parameters
packages/ocean/legoesm/ocean/fidelity/oceananigans_state_bridge.py-136-    ----------
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-358-    host = np.asarray(arr)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-359-    if jax.process_count() <= 1:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-360-        return jax.device_put(jnp.asarray(host), sharding)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-361-    return jax.make_array_from_callback(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-362-        host.shape, sharding, lambda idx: host[idx])
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-363-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-364-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-365-def shard_state_latlon(state, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:366:    """Lay out a ``LatLonCGridOceanState`` for the lat-band SPMD step.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-367-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-368-    Cell / u-grid array fields (leading dim ``n_lat``) shard ``P("lat")``.  The
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-369-    staggered ``v`` / ``v_mask`` (leading dim ``n_lat+1``) are carried as
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-370-    ``v_lower = field[0:n_lat]`` (``n_lat`` rows, ``P("lat")``) — the top pole row
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-371-    ``field[n_lat]`` is a 0 wall on the regular grid and is reconstructed in-body
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-372-    by the band halo.  ``None`` fields pass through.  The inverse is
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-373-    :func:`gather_state_latlon`.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-374-
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-78-from legoesm.ocean.physics.surface_forcing.config import (
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-79-    RestoringConfig,
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-80-    SurfaceForcingConfig,
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-81-)
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-82-from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-83-from legoesm.ocean.sponge import SpongeForcing
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-84-from legoesm.ocean.state import (
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-85-    LatLonCGridOceanConfig,
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py:86:    LatLonCGridOceanState,
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-87-    OceanSurfaceForcing,
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-88-)
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-89-from legoesm.ocean.vertical import create_z_star_from_thicknesses
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-90-
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-91-# --- MITgcm tutorial_reentrant_channel parameters (input/data + gendata_50km.m).
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-92-NX = 20
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-93-NY = 40
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-94-DX_M = 50.0e3
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-130-
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-131-
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-132-class MitgcmReentrantChannelRecipe(NamedTuple):
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-133-    """Assembled legoESM reproduction of the MITgcm reentrant channel."""
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-134-
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-135-    geometry: LatLonCGridGeometry
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-136-    z_coord: object
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-137-    config: LatLonCGridOceanConfig
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py:138:    state: LatLonCGridOceanState
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-139-    wind_forcing: OceanSurfaceForcing
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-140-    sponge: SpongeForcing
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-141-    land_mask: jnp.ndarray
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-142-    dt_s: float
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-143-
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-144-
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-145-def build_reentrant_channel_geometry() -> LatLonCGridGeometry:
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-146-    """Cartesian beta-plane channel matching MITgcm's grid + Coriolis.
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-356-                restoring=build_reentrant_channel_restoring(geom)),
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-357-            bottom_drag=BottomDragConfig(scheme="none"),
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-358-            shortwave_penetration=None,
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-359-        ),
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-360-    )
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-361-
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-362-
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-363-def build_reentrant_channel_state(
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py:364:    geom, z_coord, land_mask) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-365-    """Rest state with the gendata 3-D analytic T(z) stratification (S uniform 35,
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-366-    frozen)."""
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-367-    base = rest_state_latlon_cgrid_ocean(
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-368-        geom, z_coord, H_max=HO_M, S_uniform=35.0,
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-369-        land_mask_override=land_mask)
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-370-    z_c = np.asarray(z_coord.z_full_ref)                 # negative depths (nz,)
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-371-    nz = z_c.shape[0]
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-372-    temp = reentrant_channel_initial_temperature(z_c)    # (ny, nz)
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-28-from __future__ import annotations
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-29-
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-30-from typing import NamedTuple
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-31-
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-32-import jax.numpy as jnp
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-33-import numpy as np
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-34-from legoesm.ocean.fidelity.concept_registry import oracle_to_canonical
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-35-from legoesm.ocean.fidelity.mitgcm_runner import MitgcmResult
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py:36:from legoesm.ocean.state import LatLonCGridOceanState
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-37-
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-38-# Canonical state-field names this bridge knows how to place. (Other canonical
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-39-# names returned by the registry — A_h, rho_0 … — are parameters, not state.)
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-40-_PLACEABLE = frozenset({"u", "v", "w", "T", "S", "eta"})
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-41-
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-42-
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-43-class MitgcmStateBridgeOutput(NamedTuple):
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-44-    """Result of :func:`mitgcm_snapshot_to_legoesm_state`."""
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-45-
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py:46:    state: LatLonCGridOceanState
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-47-    info: dict  # provenance / shapes / which fields were placed vs left at base
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-48-
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-49-
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-50-def _to_latlon_lev(arr: np.ndarray) -> np.ndarray:
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-51-    """``(nz, ny, nx)`` -> ``(ny, nx, nz)``; ``(ny, nx)`` passes through.
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-52-
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-53-    MITgcm 3-D fields come off disk with the vertical on the FIRST axis; legoESM
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-54-    wants it LAST. 2-D (single-level / surface) fields are already ``(ny, nx)``.
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-106-    always closed for the rectilinear ocean configs we mirror).
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-107-    """
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-108-    zero = np.zeros_like(v_cell[:1, ...])
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-109-    return np.concatenate([v_cell, zero], axis=0)
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-110-
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-111-
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-112-def mitgcm_snapshot_to_legoesm_state(
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-113-    result: MitgcmResult,
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py:114:    base_state: LatLonCGridOceanState,
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-115-    *,
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-116-    time_index: int = -1,
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-117-    cyclic_x: bool = False,
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-118-) -> MitgcmStateBridgeOutput:
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-119-    """Initialise a legoESM lat-lon C-grid state from a MITgcm snapshot.
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-120-
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-121-    Parameters
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-122-    ----------
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-123-    result : MitgcmResult
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-124-        Output of :func:`legoesm.ocean.fidelity.mitgcm_runner.load_mitgcm_reference`.
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py:125:    base_state : LatLonCGridOceanState
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-126-        Empty/rest legoESM state at the matching grid; provides grid / land-mask
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-127-        / face-mask templates and the values for any field the snapshot omits.
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-128-    time_index : int, default -1
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-129-        Which loaded iteration to extract when several were stacked (default
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-130-        last). Ignored for a single-iteration result.
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-131-    cyclic_x : bool, default False
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-132-        Whether the zonal boundary is periodic (wrap ``u``) or a closed wall
packages/ocean/legoesm/ocean/fidelity/mitgcm_state_bridge.py-133-        (zero east face). The tutorial barotropic gyre is a closed basin.
--
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-64-
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-65-from __future__ import annotations
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-66-
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-67-import jax
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-68-import jax.numpy as jnp
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-69-
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-70-from legoesm.grids.latlon import LatLonGrid
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-71-from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:72:from legoesm.ocean.state import LatLonCGridOceanState, LatLonCGridOceanConfig
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-73-from legoesm.ocean.dynamics.latlon_cgrid_operators import (
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-74-    coriolis_cgrid_energy_conserving,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-75-    fold_is_local,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-76-    north_fold_mask,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-77-    apply_north_fold,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-78-    divergence_cgrid,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-79-    fold_vface_row,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-80-    gradient_x_cgrid,
--
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1282-    _cg_area_adjoint.defvjp(_cg_fwd, _cg_bwd)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1283-
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1284-    return _cg_area_adjoint(rhs, x0, H_u, H_v, coeff,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1285-                            mask, u_mask, v_mask, inv_diag,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1286-                            jnp.asarray(tol))
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1287-
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1288-
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1289-def barotropic_implicit_latlon_cgrid(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1290:    state: LatLonCGridOceanState,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1291-    dt: float,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1292-    grid: LatLonGrid,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1293-    z_coord: OceanZStarCoordinate,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1294-    config: LatLonCGridOceanConfig,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1295-    F_slow_eta: jnp.ndarray | None = None,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1296-    F_slow_u: jnp.ndarray | None = None,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1297-    F_slow_v: jnp.ndarray | None = None,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1298-    *,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1299-    return_residual: bool = False,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1300:) -> tuple[LatLonCGridOceanState, tuple[jnp.ndarray, jnp.ndarray]]:
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1301-    """Single-step implicit free-surface solver (lat-lon C-grid).
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1302-
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1303-    Returns
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1304-    -------
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1305-    (state_new, (Hu_avg, Hv_avg))
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1306-        Same return signature as ``barotropic_substeps_latlon_cgrid``.
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1307-        When ``return_residual=True`` (static; default ``False`` keeps
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py-1308-        the hot path unchanged) the returned tuple is instead
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-46-    LatLonCGridGeometry,
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-47-    create_beta_plane_cgrid_geometry,
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-48-)
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-49-from legoesm.ocean.eos import LinearEOSConfig
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-50-from legoesm.ocean.fidelity.mitgcm_recipe import mitgcm_canonical_ocean_config
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-51-from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-52-from legoesm.ocean.state import (
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-53-    LatLonCGridOceanConfig,
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py:54:    LatLonCGridOceanState,
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-55-)
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-56-from legoesm.ocean.vertical import create_z_star_from_thicknesses
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-57-
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-58-# --- MITgcm front_relax parameters (input/data + gendata.m). -----------------
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-59-NX = 1
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-60-NY = 32
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-61-DX_M = 10.0e3
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-62-DY_M = 10.0e3
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-78-
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-79-
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-80-class MitgcmFrontRelaxRecipe(NamedTuple):
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-81-    """Assembled legoESM reproduction of the MITgcm front_relax case."""
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-82-
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-83-    geometry: LatLonCGridGeometry
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-84-    z_coord: object
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-85-    config: LatLonCGridOceanConfig
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py:86:    state: LatLonCGridOceanState
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-87-    dt_s: float
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-88-
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-89-
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-90-def build_front_relax_geometry() -> LatLonCGridGeometry:
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-91-    """Cartesian f-plane channel matching MITgcm's grid + Coriolis.
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-92-
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-93-    The front is centred at Y=0 (``sin(pi Y/Ly)``), so ``y_origin`` puts the
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-94-    domain centre at 0.  ``Ly = dy*(ny-1)`` (the wet span; the north row is a
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-200-        barotropic_solver="implicit_cn",
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-201-        ab2_epsilon=AB_EPS,
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-202-        # Biharmonic viscosity (component del4) via the card's overrides.
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-203-        B_h=b_h,
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-204-        B_h_lat_scaling=False,
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-205-    )
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-206-
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-207-
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py:208:def build_front_relax_state(geom, z_coord) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-209-    """Rest state with the analytic baroclinic front in T (+ passive S)."""
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-210-    base = rest_state_latlon_cgrid_ocean(
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-211-        geom, z_coord, land_mask_override=build_front_relax_land_mask(),
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-212-    )
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-213-    # Cell-centre y from the beta-plane origin (lat is pinned to 0 on the
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-214-    # Cartesian grid, so recover y directly: y_c = y_origin + (j+0.5) dy).
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-215-    ly = DY_M * (NY - 1)
packages/ocean/legoesm/ocean/fidelity/mitgcm_front_relax_recipe.py-216-    y_c = -0.5 * ly + (np.arange(NY) + 0.5) * DY_M
--
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-31-"""
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-32-from __future__ import annotations
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-33-
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-34-import jax.numpy as jnp
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-35-import numpy as np
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-36-from legoesm.core.field import Field
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-37-from legoesm.ocean.constants_config import VEROS_CONSTANTS_CONFIG
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-38-from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py:39:from legoesm.ocean.state import LatLonCGridOceanState
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-40-
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-41-__all__ = [
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-42-    "build_veros_global_state",
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-43-    "gm_redi_eke_isopycnal_on",
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-44-    "veros_area_t_generic",
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-45-]
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-46-
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-47-
--
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-102-    grid,
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-103-    z_coord,
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-104-    land_mask: np.ndarray,
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-105-    H_bathy: np.ndarray,
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-106-    tke_config,
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-107-    eke_config,
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-108-    T_init: np.ndarray | None = None,
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-109-    S_init: np.ndarray | None = None,
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py:110:) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/fidelity/veros_global_common.py-111-    """Initial state for a VEROS-faithful global recipe.

def build_model_and_state(n_lat, n_lon, nlev, seed=0, *,
                          wide_halo=False, wide_halo_chunk=0,
                          tripole=False, baro_solver="implicit_cn",
                          force_pcg=False, pcg_variant="standard",
                          pcg_fixed_iters=0):
    """Ocean model + gently perturbed rest state (flat 4000 m bottom).

    The perturbation (small u/v/eta/T noise on the rest stratification)
    exercises every term of the composed step — advection, Coriolis, PGF,
    the barotropic solve (implicit-CN by default; split-explicit when
    ``baro_solver="explicit_substep"``) and implicit vmix — instead of the
    trivial rest fixed point, mirroring the SPMD equivalence gate's IC
    recipe.
    """
    import jax.numpy as jnp
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    if tripole:
        # Synthetic tripole (ORCA fold): the sharded step's fold support
        # is gated by tests/parallel/test_latlon_ocean_spmd_tripole.py;
        # the wide-halo lever refuses folds at model construction.
        from legoesm.grids.tripole import create_synthetic_tripole

        grid = create_synthetic_tripole(n_lat=n_lat, n_lon=n_lon)
    else:
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    # Wide-halo lever (A/B): one fused wide lat-halo exchange per chunk of
    # barotropic substeps instead of ~4 ppermute pads per substep.  The wide
    # path's per-substep clamp is local by contract, so pin local clamping
    # in BOTH arms for a controlled comparison.
    # Production-matching solver (scaling audit, bottleneck 4): OMIP runs
    # implicit_cn (run_omip.py full preset); the config-dataclass default
    # is explicit_substep, so it MUST be set explicitly here or the bench
    # measures a non-production step.
    flat = {"barotropic_solver": baro_solver}
    if force_pcg:
        # Solver-matched strong ladders (codex 2026-07-24 finding 2): the
        # implicit-CN dispatch runs adaptive stock CG on a SINGLE device but
        # the fixed-iteration distributed PCG under SPMD/MPI — an nd=1
        # reference leg without this flag times a DIFFERENT solver than the
        # nd>1 legs. Forces the fixed-M PCG everywhere.
        if baro_solver != "implicit_cn":
            raise SystemExit(
                "--force-pcg only affects the implicit_cn barotropic solve; "
                "drop it for explicit_substep arms.")
        flat["barotropic_implicit_force_pcg"] = True
    if pcg_fixed_iters:
        # Each PCG iteration contributes DEPENDENT reduction batches, and
        # the campaign's mechanism finding (job 26458930) is that exposed
        # dependent sync — not bytes, not schedulable overlap — is what the
        # step pays above its roofline. Iteration count is therefore the
        # most direct sync-point lever available in config.
        if baro_solver != "implicit_cn":
            raise SystemExit(
                "--pcg-fixed-iters only affects the implicit_cn fixed-M PCG.")
        flat["barotropic_implicit_pcg_fixed_iters"] = int(pcg_fixed_iters)
    if pcg_variant != "standard":
        if baro_solver != "implicit_cn":
            raise SystemExit(
                "--pcg-variant only affects the implicit_cn fixed-M PCG; "
                "drop it for explicit_substep arms.")
        flat["barotropic_implicit_pcg_variant"] = pcg_variant
    if wide_halo:
        if baro_solver != "explicit_substep":
            raise SystemExit(
                "--wide-halo is a split-explicit barotropic lever; it "
                "requires --baro-solver explicit_substep (implicit_cn has "
                "no substep halo to widen).")
        flat.update(barotropic_wide_halo=True,
                    barotropic_wide_halo_chunk=int(wide_halo_chunk),
                    barotropic_local_subcycle_clamp=True)
    model = LatLonCGridOceanModel(grid, z_coord,
                                  LatLonCGridOceanConfig.from_flat(**flat))
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    rng = np.random.default_rng(seed)
    # Project the velocity noise through the face masks: v_mask zeroes the pole
    # WALL rows, so the nd=1 and nd>1 runs time the SAME initial state (the
    # nd>1 shard drops v[n_lat] and reconstructs it as the pole-wall zero — a
    # random value there would make strong-scaling ICs differ across device
    # counts; codex).
    u_mask = np.asarray(state.u_mask.data)[..., None]
    v_mask = np.asarray(state.v_mask.data)[..., None]
    u = 0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev)) * u_mask
    v = 0.02 * rng.standard_normal((n_lat + 1, n_lon, nlev)) * v_mask
    eta = 0.005 * rng.standard_normal((n_lat, n_lon))
    temp = (5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
            + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)),
        T=state.T.replace(data=jnp.asarray(temp)))
    return model, state


def _block(state):
    jax.block_until_ready([leaf for leaf in jax.tree.leaves(state)
                           if leaf is not None])


def main() -> int:
 M packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
 M scripts/plot/plot_scaling_paper_figure.py
?? .physics-validator/scaling_campaign/codex_bcast_fix_r17_2026-08-03.md
 .../legoesm/ocean/dynamics/sharded_ocean_step.py   | 37 +++++++++++++++++++---
 scripts/plot/plot_scaling_paper_figure.py          |  5 +--
 2 files changed, 35 insertions(+), 7 deletions(-)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
scripts/plot/plot_scaling_paper_figure.py

exec
/usr/bin/bash -lc "sed -n '360,455p' packages/ocean/legoesm/ocean/state.py; sed -n '235,270p' packages/ocean/legoesm/ocean/init_latlon_cgrid.py; sed -n '1,180p' packages/core/legoesm/parallel/latlon_spmd.py; sed -n '250,335p' packages/core/legoesm/parallel/latlon_spmd.py; rg -n -C 3 \"shard_forcing_stack_latlon|shard_forcing_latlon|shard_state_latlon\" packages/coupler scripts/run packages/ocean -g '*.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 403ms:
# ==============================================================================

class LatLonCGridOceanState(NamedTuple):
    """State for the lat-lon C-grid finite-volume ocean primitive equations.

    Velocities live on cell faces (Arakawa C-grid staggering):
    - u at east/west faces (lon interfaces): shape (n_lat, n_lon+1, nlev)
    - v at north/south faces (lat interfaces): shape (n_lat+1, n_lon, nlev)

    Scalars live at cell centers:
    - eta, T, S, H_bathy, land_mask: shape (n_lat, n_lon [, nlev])

    Face masks (u_mask, v_mask) are derived from the cell-center land_mask:
    a face is wet (mask=1) only if both adjacent cells are wet.

    Fields
    ------
    u : Field
        Zonal velocity at lon interfaces [m/s]. Shape (n_lat, n_lon+1, nlev).
    v : Field
        Meridional velocity at lat interfaces [m/s]. Shape (n_lat+1, n_lon, nlev).
    T : Field
        Potential temperature [degC]. Shape (n_lat, n_lon, nlev).
    S : Field
        Salinity [PSU]. Shape (n_lat, n_lon, nlev).
    eta : Field
        Sea surface height [m]. Shape (n_lat, n_lon).
    H_bathy : Field
        Bathymetry depth [m]. Static (positive downward). Shape (n_lat, n_lon).
    land_mask : Field
        Ocean mask at cell centers. Static. 1=ocean, 0=land. Shape (n_lat, n_lon).
    u_mask : Field
        Ocean mask at u-points (lon interfaces). Shape (n_lat, n_lon+1).
    v_mask : Field
        Ocean mask at v-points (lat interfaces). Shape (n_lat+1, n_lon).
    w : Field
        Vertical velocity [m/s]. Shape (n_lat, n_lon, nlev). Diagnostic field computed from flux divergence.
    T_som : Field or None
        SOM (Prather 1986) moments for temperature. Shape (n_lat, n_lon, nlev, 9).
        Order: [sx, sy, sz, sxx, syy, szz, sxy, sxz, syz].
        None when tracer_advection != "som".
    S_som : Field or None
        SOM (Prather 1986) moments for salinity. Same shape and order as T_som.
    """

    u: Field
    v: Field
    T: Field
    S: Field
    eta: Field
    H_bathy: Field
    land_mask: Field
    u_mask: Field
    v_mask: Field
    w: Field
    T_som: object = None
    S_som: object = None
    T_flux_div_prev: object = None  # Previous advection flux divergence for T (AB2 only)
    S_flux_div_prev: object = None  # Previous advection flux divergence for S (AB2 only)
    # Prognostic eddy kinetic energy [m^2/s^2], 2-D Field (n_lat, n_lon), used only
    # when the prognostic-EKE GM closure is active (config.gm_redi.eke not None).
    # Default None -> inert (no EKE): zero behaviour change for existing configs.
    eke: object = None
    # Prognostic turbulent kinetic energy [m^2/s^2] at the interior interfaces
    # (W-grid), 3-D Field (n_lat, n_lon, nlev-1), used only when the prognostic
    # TKE vertical-mixing closure is active (vertical_mixing.tke.prognostic=True).
    # Carried across model steps: each step runs ONE backward-Euler TKE solve
    # (dt = dt_mom) seeded from this field and stores the updated TKE back.
    # Default None -> inert (Mode-B diagnostic chain): zero behaviour change.
    tke: object = None
    # Prior-step ADVECTIVE TKE tendency [m^2/s^3] at the interior interfaces
    # (W-grid), 3-D Field (n_lat, n_lon, nlev-1) — the Adams-Bashforth history
    # dtke^{n-1} for the prognostic-TKE superbee advection (Veros vs.dtke,
    # tke.py:292-323), used only when vertical_mixing.tke.advection_scheme !=
    # "none" (requires prognostic=True). None on the first step => the AB2
    # increment uses a zero previous tendency, exactly like Veros's
    # zero-initialised dtke[taum1]. Default None -> inert: zero behaviour change.
    dtke: object = None
    # Carried EKE dissipation rate [m^2/s^3] at the interior interfaces (W-grid),
    # 3-D Field (n_lat, n_lon, nlev-1). Populated by the 3-D EKE step (Veros
    # eke_diss_iw, run in the GM/Redi stage) and consumed WITHIN THE SAME step by
    # the prognostic TKE source when vertical_mixing.tke.source_eke_diss=True
    # (the TKE source reads state_new.eke_diss — this step's EKE update — since
    # the implicit-vmix TKE solve runs after GM/Redi, matching Veros's same-step
    # eke->tke ordering). The CARRIED value on state is only the fallback when a
    # step produces none (e.g. the 2-D EKE path). Default None -> inert: zero
    # behaviour change.
    eke_diss: object = None
    # Prior EXPLICIT increment ΔX_expl^{n-1} for the AB2 outer integrator
    # (config.outer_integrator == "ab2"): the explicit-only forward-Euler increment
    # (advection, GM/Redi, lateral friction, Coriolis, barotropic solve, freshwater)
    # WITHOUT the implicit vertical mixing — implicit mixing is applied once, after
    # the AB2 extrapolation, and is NOT carried. Tracers store the full explicit
    # increment; u/v store the BAROCLINIC-deviation explicit increment (the barotropic
    # mode is kept from the barotropic solve, un-AB2'd). Default None -> inert
    # (forward-Euler): zero behaviour change.

    # Face masks — consult ``grid.seam_wall_rows`` for a partial-periodic
    # seam wall (NEMO DINO); None on ordinary grids → fully periodic.
    u_mask, v_mask = compute_face_masks(land_mask, grid)

    # Initialize vertical velocity with zeros (will be computed during step)
    w_zeros = jnp.zeros((n_lat, n_lon, nlev), dtype=dtype)

    dims_u = ("lat", "lon_u", "level")
    dims_v = ("lat_v", "lon", "level")
    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")
    dims_u2d = ("lat", "lon_u")
    dims_v2d = ("lat_v", "lon")

    return LatLonCGridOceanState(
        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
                staggering="edge"),
        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
                staggering="edge"),
        T=Field(data=T_3d, name="T", dims=dims_3d, units="degC"),
        S=Field(data=S_3d, name="S", dims=dims_3d, units="PSU"),
        eta=Field(data=zeros_2d, name="eta", dims=dims_2d, units="m"),
        H_bathy=Field(data=H_bathy, name="H_bathy", dims=dims_2d, units="m"),
        land_mask=Field(data=land_mask, name="land_mask", dims=dims_2d, units=""),
        u_mask=Field(data=u_mask, name="u_mask", dims=dims_u2d, units=""),
        v_mask=Field(data=v_mask, name="v_mask", dims=dims_v2d, units=""),
        w=Field(data=w_zeros, name="w", dims=dims_3d, units="m/s"),
    )


def wind_driven_gyre_latlon_cgrid(
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    H_max: float = 5500.0,
    lon_west: float = 0.0,
"""Single-controller SPMD halo for the lat-lon grid (multi-GPU, no mpi4jax).

The ocean lat-lon C-grid (and atm lat-lon) shard cleanly by LATITUDE BAND: each
device owns a contiguous lat band and the FULL longitude circle (lon is periodic
and kept local — the audit's "every rank owns all longitudes").  The halo is
therefore 1-D over the ``"lat"`` mesh axis:

  * longitude: periodic wrap — LOCAL ``jnp.pad(mode="wrap")`` (no comm), exactly
    like the serial :func:`legoesm.grids.halo_latlon.pad_halo_latlon_local`.
  * latitude interior band cuts: ``jax.lax.ppermute`` of the ``halo`` edge rows
    between adjacent bands (north neighbour's bottom rows / south neighbour's
    top rows).
  * poles: the end bands (axis_index 0 = south, N-1 = north) have no neighbour
    there, so they fold their OWN pole rows (mirror in lat + 180 deg in lon),
    selected by a ``jnp.where`` on the band index — bit-identical to the serial
    pole fold.

This is the lat-lon analogue of the cubed-sphere
``cubesphere_exchange.make_tiled_pad_body`` (which the cube SPMD uses), and the
foundation for the ocean lat-lon multi-GPU SPMD step (pure-jax ppermute over the
RTX8000 PCIe pair — no mpi4jax dependency).  Validated by BIT-IDENTITY vs the
serial local pad (``tests/parallel/test_latlon_spmd_halo.py``), the proven
methodology.  ``check_vma=False`` follows the cube SPMD halo bodies.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from legoesm.parallel.shard_map_compat import shard_map

# Reference halo depth for the topology chooser's pole-fold cost model: the
# widest production halo (the PPM halo-2 exchange).  The chooser's score is
# otherwise normalized per unit halo depth; the partner fold's h-quadratic
# E/W-extension strips are charged at this reference so the constant stays
# honest without threading the runtime halo through the chooser (see
# choose_latlon_2d_topology).
_FOLD_REF_HALO = 2


def latlon_band_perms(n_dev: int):
    """Static (src, dst) permutation pairs over the 1-D ``lat`` band axis.

    ``perm_north``: each band ``b`` receives band ``b+1``'s bottom rows as its
    NORTH ghost -> source ``b+1`` sends to ``b`` (pairs ``(s, s-1)``); the top
    band (``N-1``) is not a destination -> its north_recv is zeros, replaced by
    the north pole fold.  ``perm_south``: band ``b`` receives band ``b-1``'s top
    rows as its SOUTH ghost -> ``(s, s+1)``; the bottom band (0) gets zeros ->
    south pole fold.

    PUBLIC (issue #353 SPMD step): the ocean lat-band SPMD wrapper
    (``ocean.dynamics.sharded_ocean_step``) reuses ``perm_north`` to lift the
    staggered-v north boundary row from band ``r+1`` (the no-private-cross-import
    rule — promoted from ``_latlon_band_perms``).
    """
    perm_north = tuple((s, s - 1) for s in range(1, n_dev))   # send up->down
    perm_south = tuple((s, s + 1) for s in range(0, n_dev - 1))  # down->up
    return perm_north, perm_south


def latlon_lon_ring_perms(p_lon: int):
    """Static (src, dst) permutation pairs over the periodic ``lon`` ring axis.

    Longitude is a periodic RING (unlike the pole-terminated lat LINE), so
    every tile is both a source and a destination — the perms are full cyclic
    permutations with NO non-target zeros.

    ``perm_to_west``: source ``s`` sends to its WEST neighbour
    ``(s-1) mod p`` — the receiver ``j`` gets tile ``j+1``'s payload, i.e.
    its EAST ghost.  ``perm_to_east``: source ``s`` sends to its EAST
    neighbour ``(s+1) mod p`` — the receiver gets its WEST ghost.
    """
    perm_to_west = tuple((s, (s - 1) % p_lon) for s in range(p_lon))
    perm_to_east = tuple((s, (s + 1) % p_lon) for s in range(p_lon))
    return perm_to_west, perm_to_east


def lon_ring_ghosts_spmd(f, mesh, halo: int = 1):
    """Periodic LONGITUDE ghosts (axis 1) under the 2-D lat-lon SPMD mesh.

    The SPMD twin of the local ``jnp.pad(mode="wrap")`` lon wrap (band path)
    and of the MPI 2-D pencil's ``exchange_halo_lon``: each tile owns a lon
    SECTOR, so its east/west ghost columns are the neighbouring tiles' edge
    columns, moved by ``jax.lax.ppermute`` over the ``"lon"`` mesh axis (the
    periodic wrap IS the cyclic ring permutation —
    :func:`latlon_lon_ring_perms`).  ``p_lon == 1`` (a degenerate lon axis /
    the 1-D-band-equivalent (N, 1) mesh) is the LOCAL wrap, chosen by a
    STATIC Python branch — bit-identical to the band path's ``jnp.pad`` (no
    collective is emitted at all).

    Lon axis is axis 1; the field must be CELL-ALIGNED in lon (width
    ``n_lon_local`` — never an ``n_lon_local+1`` u-face field, whose seam
    column would double-count under the ring shift; reconstruct u-faces via
    :func:`reconstruct_uface_left` instead).  2-D and 3-D fields supported
    (trailing axes ride through).  AD-safe: ``ppermute`` is
    self-transposing.  MUST be called INSIDE a shard_map over a mesh
    carrying the ``"lon"`` axis when ``p_lon > 1``.
    """
    if halo <= 0:
        return f
    p_lon = int(mesh.shape["lon"])
    if p_lon == 1:
        pad = ((0, 0), (halo, halo)) + ((0, 0),) * (f.ndim - 2)
        return jnp.pad(f, pad, mode="wrap")
    perm_to_west, perm_to_east = latlon_lon_ring_perms(p_lon)
    # My EAST ghost = east neighbour's west edge (sources send WEST edges to
    # their west neighbour); my WEST ghost = west neighbour's east edge.
    east_ghost = jax.lax.ppermute(f[:, :halo], "lon", perm_to_west)
    west_ghost = jax.lax.ppermute(f[:, -halo:], "lon", perm_to_east)
    return jnp.concatenate([west_ghost, f, east_ghost], axis=1)


def reconstruct_vface_lower(v_lower, axis: str, perm_north):
    """Rebuild the ``n_lat+1`` staggered v-faces from the ``n_lat``-row
    ``v_lower`` representation, INSIDE a ``shard_map`` over ``axis``.

    The staggered meridional velocity ``v`` has a leading dim ``n_lat+1`` (faces
    at latitude interfaces), coprime with ``n_lat`` for ``N>1`` so it cannot be
    sharded directly; it is carried as ``v_lower = v[:n_lat]`` (``n_lat`` rows,
    divisible by ``N``). Each band's NORTH boundary face is the next band's
    ``v_lower[0]`` (= the shared global interface row), lifted down via
    ``ppermute(..., perm_north)``; the north-most band has no neighbour there and
    receives the pole-wall zero (the ppermute non-target). Pure array core (no
    Field/state coupling) shared by the ocean and atmosphere lat-band SPMD steps
    so the v-stagger numerics are written ONCE (factored from the ocean step's
    ``_reconstruct_v`` closure). AD-safe: ``ppermute`` is self-transposing.

    Parameters
    ----------
    v_lower : array ``(n_lat_band, n_lon[, nlev])``
    axis : the ``shard_map`` mesh axis name (``"lat"``).
    perm_north : the ``(src, dst)`` pairs from :func:`latlon_band_perms`.

    Returns
    -------
    array ``(n_lat_band + 1, n_lon[, nlev])`` — the band's full v-faces.
    """
    boundary = jax.lax.ppermute(v_lower[0:1], axis, perm_north)
    return jnp.concatenate([v_lower, boundary], axis=0)


def to_vface_lower(v_full):
    """Inverse of :func:`reconstruct_vface_lower`: drop the north boundary face
    (owned by the next band) to return to the ``n_lat``-row ``v_lower``.

    Round-trip identity ``to_vface_lower(reconstruct_vface_lower(v_lower)) ==
    v_lower`` holds whenever the top boundary face is the pole-wall zero (true
    after any step that zeroes v at the pole)."""
    return v_full[:-1]


def reconstruct_vface_lower_multi(v_lowers, axis: str, perm_north):
    """FUSED multi-field twin of :func:`reconstruct_vface_lower` (message
    aggregation, scaling-M4): ONE ``ppermute`` per DTYPE GROUP for the whole
    staggered field group instead of one per field.

    Each field's single boundary row (``v_lower[0:1]``) is flattened on its
    trailing axes, concatenated into one ``(1, sum_flat)`` buffer per dtype
    group, exchanged once, then split and reshaped back — value-identical to
    the per-field reconstruction (the exchange is a bit-copy; flatten/concat/
    split are layout ops), including the north band's ppermute non-target
    zeros.  Mirrors :func:`make_latlon_band_wall_multi_pad_body`'s "per dtype
    group" packing contract; gated by
    ``tests/parallel/test_latlon_spmd_fused_halo.py``.

    Parameters
    ----------
    v_lowers : sequence of arrays ``(n_lat_band, n_lon[, ...])`` — the
        ``v_lower`` carriers to reconstruct (e.g. the ocean state's ``v`` and
        ``v_mask``).  Trailing shapes may differ; dtypes group internally.
    axis : the ``shard_map`` mesh axis name (``"lat"``).
    perm_north : the ``(src, dst)`` pairs from :func:`latlon_band_perms`.

    Returns
    -------
    tuple of arrays ``(n_lat_band + 1, ...)`` — full band v-faces, input order.
    """
    fields = tuple(v_lowers)
    else:
        perm_to_west, _ = latlon_lon_ring_perms(p_lon)
        boundary = jax.lax.ppermute(u_left[:, 0:1], axis, perm_to_west)
    return jnp.concatenate([u_left, boundary], axis=1)


def to_uface_left(u_full):
    """Inverse of :func:`reconstruct_uface_left`: drop the east seam column
    (owned as the east neighbour's column 0) to return to the
    ``n_lon_local``-column ``u_left``.

    Round-trip identity holds because the dropped seam face is bit-equal to
    the east neighbour's first face (both tiles compute the shared interface
    from identical wrapped operands — the u-stagger analogue of the v
    round-trip's pole-wall-zero invariant)."""
    return u_full[:, :-1]


def replicate_leaf(arr, rep, *, multiprocess: bool):
    """Replicate one (possibly lat-sharded) leaf onto every device of ``rep``'s
    mesh — the gather primitive shared by the atm and ocean lat-band SPMD steps
    (``gather_state_atm_latlon`` / ``gather_state_latlon``).

    Single-process: plain ``jax.device_put`` (the historical path, unchanged).
    Multi-controller (``jax.process_count() > 1``, route-B ``jax.distributed``):
    a top-level ``device_put`` cannot reshard an array whose shards live on
    other processes' devices, so the replication runs as a jit-compiled
    identity with replicated ``out_shardings`` — the supported cross-process
    collective path (every process executes the same program; XLA inserts the
    all-gather). The fresh ``jax.jit`` per call recompiles per gather —
    acceptable at the segment/run output boundary where gathers happen (never
    in the step hot loop).

    Parameters
    ----------
    arr : jax.Array (any sharding on ``rep``'s mesh)
    rep : NamedSharding — the replicated ``P()`` sharding of the target mesh.
    multiprocess : pass ``jax.process_count() > 1`` (keyword-only so the
        branch is explicit at every call site).
    """
    if multiprocess:
        return jax.jit(lambda a: a, out_shardings=rep)(arr)
    return jax.device_put(arr, rep)


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
scripts/run/run_omip.py-4812-            from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip.py-4813-                gather_state_latlon,
scripts/run/run_omip.py-4814-                make_sharded_ocean_step,
scripts/run/run_omip.py:4815:                shard_forcing_stack_latlon,
scripts/run/run_omip.py:4816:                shard_state_latlon,
scripts/run/run_omip.py-4817-            )
scripts/run/run_omip.py-4818-            from legoesm.parallel.mesh import create_latlon_mesh
scripts/run/run_omip.py-4819-            # Prime the build-once vertex-mask cache from the CONCRETE
--
scripts/run/run_omip.py-4822-            _dev = create_latlon_mesh(n_devices=_nd)
scripts/run/run_omip.py-4823-            spmd_step = make_sharded_ocean_step(model, _dev.mesh)
scripts/run/run_omip.py-4824-            spmd_gather = partial(gather_state_latlon, mesh=_dev.mesh)
scripts/run/run_omip.py:4825:            state = shard_state_latlon(state, _dev.mesh)
scripts/run/run_omip.py-4826-            # Lay per-block forcing stacks out lat-band-sharded so the
scripts/run/run_omip.py-4827-            # in-scan interpolation / bulk fluxes stay shard-local (shared
scripts/run/run_omip.py:4828:            # layout helper — see shard_forcing_stack_latlon).
scripts/run/run_omip.py-4829-            spmd_shard_stack = partial(
scripts/run/run_omip.py:4830:                shard_forcing_stack_latlon, mesh=_dev.mesh)
scripts/run/run_omip.py-4831-            if jax.process_index() == 0:
scripts/run/run_omip.py-4832-                _lane = "route-B multicontroller" if _multi else "single-controller"
scripts/run/run_omip.py-4833-                print(f"  SPMD ({_lane}): lat-band sharded dynamics step over "
--
scripts/run/run_omip_core2.py-2495-    dropped), detected here by ``v`` sharing ``u``'s leading dim (the GLOBAL
scripts/run/run_omip_core2.py-2496-    layout's v always has ``n_lat+1`` rows).  The missing top row is the
scripts/run/run_omip_core2.py-2497-    pole/cap WALL — identically zero under the v-carrier contract asserted in
scripts/run/run_omip_core2.py:2498:    ``shard_state_latlon`` — appended via the SAME shared reconstruction the
scripts/run/run_omip_core2.py-2499-    full-state gather uses (``append_vface_wall_row``), so the sharded-state
scripts/run/run_omip_core2.py-2500-    read is BIT-identical to reading the gathered state WITHOUT forcing the
scripts/run/run_omip_core2.py-2501-    per-step full-state gather (gated by
--
scripts/run/run_omip_core2.py-5558-        if args.spmd_persistent_state:
scripts/run/run_omip_core2.py-5559-            # PERSISTENT lane (scaling-M2 increment 1): the state stays
scripts/run/run_omip_core2.py-5560-            # lat-band sharded ACROSS steps via the pure-dynamics inner step
scripts/run/run_omip_core2.py:5561:            # (the wrapper docstring's own guidance); shard_state_latlon /
scripts/run/run_omip_core2.py-5562-            # gather_state_latlon run only at the residency boundaries the
scripts/run/run_omip_core2.py-5563-            # helpers below manage (initial shard, snapshot/abort/final
scripts/run/run_omip_core2.py-5564-            # gathers, and the counted per-step gathers forced by host-global
--
scripts/run/run_omip_core2.py-5566-            from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip_core2.py-5567-                gather_state_latlon,
scripts/run/run_omip_core2.py-5568-                make_sharded_ocean_step,
scripts/run/run_omip_core2.py:5569:                shard_state_latlon,
scripts/run/run_omip_core2.py-5570-            )
scripts/run/run_omip_core2.py-5571-            _spmd_inner = make_sharded_ocean_step(model, _spmd_mesh)
scripts/run/run_omip_core2.py-5572-            _ocean_step = (lambda st, sf, fw, t_sec=None:
--
scripts/run/run_omip_core2.py-5574-                                       freshwater=fw))
scripts/run/run_omip_core2.py-5575-
scripts/run/run_omip_core2.py-5576-            def _pers_shard_fn(st, _mesh=_spmd_mesh):
scripts/run/run_omip_core2.py:5577:                return shard_state_latlon(st, _mesh)
scripts/run/run_omip_core2.py-5578-
scripts/run/run_omip_core2.py-5579-            def _pers_gather_fn(st, _mesh=_spmd_mesh):
scripts/run/run_omip_core2.py-5580-                return gather_state_latlon(st, _mesh)
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-362-        host.shape, sharding, lambda idx: host[idx])
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-363-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-364-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:365:def shard_state_latlon(state, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-366-    """Lay out a ``LatLonCGridOceanState`` for the lat-band SPMD step.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-367-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-368-    Cell / u-grid array fields (leading dim ``n_lat``) shard ``P("lat")``.  The
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-387-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-388-        if _np.asarray(vm.data)[-1].any():
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-389-            raise ValueError(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:390:                "shard_state_latlon: the state's TOP v-face row is LIVE "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-391-                "(v_mask[-1] has ocean faces) — the lat-band v-carrier "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-392-                "drops that row and reconstructs it as the pole/cap wall "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-393-                "zero, which would silently delete seam velocities. "
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-405-            return None
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-406-        # Drop the TOP v-row (the north pole wall, v==0 on the regular grid) so
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-407-        # the leading dim becomes n_lat (divisible by N).  ROUND-TRIP INVARIANT
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:408:        # (codex finding): shard_state_latlon -> gather_state_latlon re-appends a
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-409-        # ZERO top row, so the round-trip is a bitwise identity ONLY when the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-410-        # input's top v-row is already zero (a valid masked regular-grid state;
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-411-        # the pole-wall BC enforces v[n_lat]=0 every step).  An arbitrary nonzero
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-440-    return state._replace(**updates)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-441-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-442-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:443:def shard_forcing_latlon(forcing, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-444-    """Lay out a forcing pytree (``FreshwaterForcing`` / ``OceanSurfaceForcing``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-445-    / ``SpongeForcing`` — or any nesting of them) for the lat-band SPMD step.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-446-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-463-    return jax.tree.map(_put, forcing)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-464-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-465-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:466:def shard_forcing_stack_latlon(stack, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-467-    """Lay out a STACKED per-block forcing pytree for the lat-band SPMD
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-468-    block-scan (the ``run_omip`` JRA55 lanes; see
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-469-    ``_build_jra55_block_fn`` / ``_build_jra55_block_fn_interp``).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-470-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:471:    Unlike :func:`shard_forcing_latlon` (per-step, lat at axis 0), the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-472-    block builders stack ``N`` steps / raw records along a LEADING axis,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-473-    so the lat axis sits at position 1.  A ``(n_rec, n_lat, n_lon[, ...])``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-474-    leaf therefore shards ``P(None, "lat", ...)`` (records replicated, the
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-483-    ``(n_rec, n_meta)`` table) would be silently mis-sharded — keep such
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-484-    metadata 1-D (or replicate it explicitly) before it reaches this helper.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-485-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:486:    Keeping this next to :func:`shard_state_latlon` means the driver and
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-487-    the parity tests share ONE layout definition — the block-scan forcing
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-488-    stack must be laid out consistently with the state the sharded step
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-489-    carries, and a second copy would drift.
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-512-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-513-    This is THE reconstruction of the row the carrier drops: the regular-grid
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-514-    north pole wall / tripole cap row, identically zero under the v-carrier
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:515:    contract (:func:`shard_state_latlon` REFUSES a state whose top v-face row
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-516-    is live), so the result is bit-identical to the original staggered array.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-517-    Shared by :func:`gather_state_latlon` (full-state gather) and the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-518-    persistent-lane DEVICE-SIDE staggered reads (the OMIP driver's
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-525-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-526-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-527-def gather_state_latlon(state, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:528:    """Inverse of :func:`shard_state_latlon`: gather every leaf to a single device
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-529-    and rebuild the full ``(n_lat+1, ...)`` ``v`` / ``v_mask`` by appending the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-530-    pole-wall row (zeros) the layout dropped.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-531-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-575-    lat-band-SPMD.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-576-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-577-    The forcing channels mirror ``model.step``'s keyword surface: pass
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:578:    pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-579-    ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-580-    is a replicated scalar like ``dt``).  ``None`` forcing keeps the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-581-    dynamics-only program; each distinct None<->populated combination
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-603-    replication check) because the band halo intentionally reads neighbour-rank
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-604-    data (replication-unaware).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-605-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:606:    The state must be laid out with :func:`shard_state_latlon` (``v`` /
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-607-    ``v_mask`` carried as the ``n_lat``-row ``v_lower``).  The body reconstructs
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-608-    each band's ``nl+1`` v-faces, runs the step on the band geometry, and
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-609-    converts the result back to the ``v_lower`` representation.
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-627-    # zero row is exact — true for the cap-row convention of
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-628-    # ``create_synthetic_tripole`` and the eORCA masks (``v_mask[-1] == 0``;
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-629-    # the serial step keeps ``v[-1] == 0`` identically).  That contract is
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:630:    # asserted on the CONCRETE state in :func:`shard_state_latlon` — a live
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-631-    # (unmasked) seam v-row refuses loudly there instead of silently
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-632-    # reconstructing zeros here.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-633-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-937-    GLOBAL state — the minimal-diff driver entry point.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-938-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-939-    Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:940:    (:func:`shard_state_latlon` + :func:`shard_forcing_latlon`), runs the lat-band
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-941-    SPMD step, then gathers the state OUT (:func:`gather_state_latlon`).  This lets
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-942-    the OMIP host loop keep operating on a normal full-domain state — the per-step
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-943-    host BCs (SSS restore, prognostic ice, geothermal, BBL, nudge) see the
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-958-        # Scatter the global state to the band layout; the forcing is sharded
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-959-        # INSIDE ``inner`` (make_sharded_ocean_step lays it out), so pass it
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-960-        # through global.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:961:        ss = shard_state_latlon(state, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-962-        ss = inner(ss, dt, surface_forcing=surface_forcing,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-963-                   freshwater=freshwater)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-964-        return gather_state_latlon(ss, mesh)

exec
/usr/bin/bash -lc "sed -n '795,940p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py; sed -n '4780,4850p' scripts/run/run_omip.py; sed -n '5530,5600p' scripts/run/run_omip_core2.py; rg -n -C 5 \"spmd_shard_stack|shard_forcing_stack_latlon|spmd_step\\(\" scripts/run/run_omip.py scripts/run/run_omip_core2.py | head -n 1000; rg -n -C 4 \"shard_forcing_latlon|make_sharded_ocean_step\\(\" packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 288ms:
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
                     sponge=None, t_seconds=None):
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
            return fn(state, forcing, geom_stacks, vmask_stack,
                      jnp.asarray(dt))
        finally:
            set_spmd_mesh(_prev_mesh)
            set_halo_backend(_prev_backend, _prev_topo)

    return sharded_step


def make_sharded_ocean_step_global(model, mesh):
    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
    that takes a GLOBAL (single-device-layout) state + forcing and returns a
    GLOBAL state — the minimal-diff driver entry point.

    Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
    (:func:`shard_state_latlon` + :func:`shard_forcing_latlon`), runs the lat-band
                and jra55_state.get("_use_single_step", False)):
            raise SystemExit(
                "--enable-latlon-spmd requires the JRA55 block-scan path, "
                "but this run selected the single-step fallback "
                "(_jra55_step calls model.step directly).")
        _multi = run_config.multicontroller
        if jax.process_count() > 1 and not _multi:
            raise SystemExit(
                "--enable-latlon-spmd is single-controller only unless "
                "--multicontroller is set; multi-node ocean scaling needs "
                "the route-B lane (jax.distributed cross-process NCCL).")
        if _multi:
            # Route-B: the mesh spans ALL global devices (one band per device
            # across every process). A strict subset would leave some
            # processes' devices out of the program (non-addressable
            # participation hazard — matches the ocean bench's guard).
            _nd = len(jax.devices())
            if run_config.spmd_n_devices and run_config.spmd_n_devices != _nd:
                raise SystemExit(
                    f"--multicontroller uses ALL global devices ({_nd} across "
                    f"{jax.process_count()} processes); --spmd-n-devices "
                    f"({run_config.spmd_n_devices}) must be 0 (auto) or {_nd}.")
        else:
            _nd = run_config.spmd_n_devices or len(jax.devices())
        if _nd > 1:
            if grid.n_lat % _nd != 0:
                raise SystemExit(
                    f"--enable-latlon-spmd: n_lat ({grid.n_lat}) not "
                    f"divisible by the device count ({_nd}); pick "
                    f"--spmd-n-devices dividing n_lat.")
            from functools import partial

            from legoesm.ocean.dynamics.sharded_ocean_step import (
                gather_state_latlon,
                make_sharded_ocean_step,
                shard_forcing_stack_latlon,
                shard_state_latlon,
            )
            from legoesm.parallel.mesh import create_latlon_mesh
            # Prime the build-once vertex-mask cache from the CONCRETE
            # state so the wrapper can build per-band masks host-side.
            model.prime_step_caches(state)
            _dev = create_latlon_mesh(n_devices=_nd)
            spmd_step = make_sharded_ocean_step(model, _dev.mesh)
            spmd_gather = partial(gather_state_latlon, mesh=_dev.mesh)
            state = shard_state_latlon(state, _dev.mesh)
            # Lay per-block forcing stacks out lat-band-sharded so the
            # in-scan interpolation / bulk fluxes stay shard-local (shared
            # layout helper — see shard_forcing_stack_latlon).
            spmd_shard_stack = partial(
                shard_forcing_stack_latlon, mesh=_dev.mesh)
            if jax.process_index() == 0:
                _lane = "route-B multicontroller" if _multi else "single-controller"
                print(f"  SPMD ({_lane}): lat-band sharded dynamics step over "
                      f"{_nd} devices across {jax.process_count()} process(es) "
                      f"({jax.default_backend()}).")
        elif jax.process_index() == 0:
            print("  SPMD: single device visible — flag is a no-op.")

    # Run time loop
    state, diag, wall_time, ok, blowup_info = _run_omip_loop(
        model, state, grid_type, grid, z_coord,
        dt, n_steps, diag_every,
        label=f"{grid_type}/{resolution}",
        restoring_targets=restoring_targets,
        restoring_tau_s=restoring_tau_s,
        restoring_ramp_days=ramp_days_eff,
        jra55_state=jra55_state,
        checkpoint_days=checkpoint_days,
        checkpoint_dir=checkpoint_dir,
        max_wallclock_seconds=run_config.max_wallclock_seconds,
            if args.n_gpus > _local:
                raise SystemExit(
                    f"--n-gpus {args.n_gpus} > local device count {_local}. This "
                    f"single-controller path uses ONE process' local devices (e.g. "
                    f"a 2-GPU node sees 2). For N spanning multiple nodes, add "
                    f"--distributed and launch one process per GPU (mpirun/srun, "
                    f"ntasks=N).")
        n_lat_final = int(grid.n_lat)
        if n_lat_final % args.n_gpus != 0:
            raise SystemExit(
                f"internal: padded n_lat ({n_lat_final}) not divisible by "
                f"n_gpus ({args.n_gpus}) — the south-pad failed.")
        from legoesm.parallel.mesh import create_latlon_mesh
        from legoesm.ocean.dynamics.sharded_ocean_step import (
            make_sharded_ocean_step_global,
        )
        # Prime the build-once vertex-mask cache from the concrete state BEFORE
        # building the sharded step (the wrapper slices the primed global vmask
        # per band; an unprimed cache raises in _build_band_vertex_masks).
        model.prime_step_caches(state)
        _spmd_mesh = create_latlon_mesh(n_devices=args.n_gpus).mesh
        _tf_spmd = getattr(model.config, "tidal_forcing", None)
        if _tf_spmd is not None and _tf_spmd.enabled:
            raise SystemExit(
                "--n-gpus > 1 with tidal_forcing.enabled=True is unsupported: "
                "the lat-band sharded step does not thread t_seconds, so the "
                "equilibrium tide would be SILENTLY inert. Run the tide "
                "single-device, or disable tidal forcing for the SPMD run.")
        if args.spmd_persistent_state:
            # PERSISTENT lane (scaling-M2 increment 1): the state stays
            # lat-band sharded ACROSS steps via the pure-dynamics inner step
            # (the wrapper docstring's own guidance); shard_state_latlon /
            # gather_state_latlon run only at the residency boundaries the
            # helpers below manage (initial shard, snapshot/abort/final
            # gathers, and the counted per-step gathers forced by host-global
            # consumers).  t_sec is always None here (tide fail-fasts above).
            from legoesm.ocean.dynamics.sharded_ocean_step import (
                gather_state_latlon,
                make_sharded_ocean_step,
                shard_state_latlon,
            )
            _spmd_inner = make_sharded_ocean_step(model, _spmd_mesh)
            _ocean_step = (lambda st, sf, fw, t_sec=None:
                           _spmd_inner(st, dt, surface_forcing=sf,
                                       freshwater=fw))

            def _pers_shard_fn(st, _mesh=_spmd_mesh):
                return shard_state_latlon(st, _mesh)

            def _pers_gather_fn(st, _mesh=_spmd_mesh):
                return gather_state_latlon(st, _mesh)

            _spmd_persistent = True
            print(f"[setup] multi-GPU lat-band SPMD: {args.n_gpus} devices, "
                  f"n_lat={n_lat_final} ({n_lat_final // args.n_gpus} "
                  f"rows/band); PERSISTENT sharded state "
                  f"(--spmd-persistent-state): full-state gathers only at "
                  f"snapshot/abort/final + counted per-step forcings.")
        else:
            _spmd_step = make_sharded_ocean_step_global(model, _spmd_mesh)
            # t_sec is always None here (tide-enabled fail-fasts above).
            _ocean_step = (lambda st, sf, fw, t_sec=None:
                           _spmd_step(st, dt, surface_forcing=sf,
                                      freshwater=fw))
            print(f"[setup] multi-GPU lat-band SPMD: {args.n_gpus} devices, "
                  f"n_lat={n_lat_final} ({n_lat_final // args.n_gpus} "
                  f"rows/band); global-in/global-out wrapper (host BCs on "
                  f"gathered state).")

    # ------------------------------------------------------------------
    # --spmd-persistent-state residency helpers (scaling-M2).  The persistent
scripts/run/run_omip_core2.py-5587-                  f"snapshot/abort/final + counted per-step forcings.")
scripts/run/run_omip_core2.py-5588-        else:
scripts/run/run_omip_core2.py-5589-            _spmd_step = make_sharded_ocean_step_global(model, _spmd_mesh)
scripts/run/run_omip_core2.py-5590-            # t_sec is always None here (tide-enabled fail-fasts above).
scripts/run/run_omip_core2.py-5591-            _ocean_step = (lambda st, sf, fw, t_sec=None:
scripts/run/run_omip_core2.py:5592:                           _spmd_step(st, dt, surface_forcing=sf,
scripts/run/run_omip_core2.py-5593-                                      freshwater=fw))
scripts/run/run_omip_core2.py-5594-            print(f"[setup] multi-GPU lat-band SPMD: {args.n_gpus} devices, "
scripts/run/run_omip_core2.py-5595-                  f"n_lat={n_lat_final} ({n_lat_final // args.n_gpus} "
scripts/run/run_omip_core2.py-5596-                  f"rows/band); global-in/global-out wrapper (host BCs on "
scripts/run/run_omip_core2.py-5597-                  f"gathered state).")
--
scripts/run/run_omip.py-3158-                   max_wallclock_seconds: float = 0.0,
scripts/run/run_omip.py-3159-                   restart_buffer_seconds: float = 600.0,
scripts/run/run_omip.py-3160-                   start_step=0,
scripts/run/run_omip.py-3161-                   nudge_woa_tau=0.0, T_woa_3d=None, S_woa_3d=None,
scripts/run/run_omip.py-3162-                   snapshot_fn=None, spmd_step=None, spmd_gather=None,
scripts/run/run_omip.py:3163:                   spmd_shard_stack=None):
scripts/run/run_omip.py-3164-    """Run time loop with diagnostics.
scripts/run/run_omip.py-3165-
scripts/run/run_omip.py-3166-    Two forcing paths, mutually exclusive:
scripts/run/run_omip.py-3167-
scripts/run/run_omip.py-3168-    * **restoring** (default): plain ``model.step(state, dt)`` followed
--
scripts/run/run_omip.py-3421-                        block_start, actual, dt, jra55_state))
scripts/run/run_omip.py-3422-            else:
scripts/run/run_omip.py-3423-                atm_stack, runoff_stack = _preload_jra55_forcing_block(
scripts/run/run_omip.py-3424-                    block_start, actual, dt, jra55_state,
scripts/run/run_omip.py-3425-                )
scripts/run/run_omip.py:3426:            if spmd_shard_stack is not None:
scripts/run/run_omip.py-3427-                # Lay the per-block forcing stacks out lat-band-sharded so
scripts/run/run_omip.py-3428-                # the in-scan interpolation / bulk fluxes stay shard-local
scripts/run/run_omip.py-3429-                # (an unsharded stack commits to device 0 and serializes
scripts/run/run_omip.py-3430-                # every forcing op there).
scripts/run/run_omip.py-3431-                if use_gpu_interp:
scripts/run/run_omip.py:3432:                    raw_stack = spmd_shard_stack(raw_stack)
scripts/run/run_omip.py:3433:                    runoff_records = spmd_shard_stack(runoff_records)
scripts/run/run_omip.py-3434-                else:
scripts/run/run_omip.py:3435:                    atm_stack = spmd_shard_stack(atm_stack)
scripts/run/run_omip.py:3436:                    runoff_stack = spmd_shard_stack(runoff_stack)
scripts/run/run_omip.py-3437-            io_dt = time.time() - t_io_start
scripts/run/run_omip.py-3438-
scripts/run/run_omip.py-3439-            t_compute_start = time.time()
scripts/run/run_omip.py-3440-            if use_gpu_interp:
scripts/run/run_omip.py-3441-                bfn = _get_block_fn_interp(actual)
--
scripts/run/run_omip.py-4768-    # Single-controller only; restoring lane only (the JRA55 block
scripts/run/run_omip.py-4769-    # functions call model._step_impl directly — follow-up).  Runs AFTER
scripts/run/run_omip.py-4770-    # the restart load so a resumed state is sharded too.
scripts/run/run_omip.py-4771-    spmd_step = None
scripts/run/run_omip.py-4772-    spmd_gather = None
scripts/run/run_omip.py:4773:    spmd_shard_stack = None
scripts/run/run_omip.py-4774-    if run_config.enable_latlon_spmd:
scripts/run/run_omip.py-4775-        if grid_type != "latlon":
scripts/run/run_omip.py-4776-            raise SystemExit(
scripts/run/run_omip.py-4777-                f"--enable-latlon-spmd requires --grid latlon "
scripts/run/run_omip.py-4778-                f"(got {grid_type}).")
--
scripts/run/run_omip.py-4810-            from functools import partial
scripts/run/run_omip.py-4811-
scripts/run/run_omip.py-4812-            from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip.py-4813-                gather_state_latlon,
scripts/run/run_omip.py-4814-                make_sharded_ocean_step,
scripts/run/run_omip.py:4815:                shard_forcing_stack_latlon,
scripts/run/run_omip.py-4816-                shard_state_latlon,
scripts/run/run_omip.py-4817-            )
scripts/run/run_omip.py-4818-            from legoesm.parallel.mesh import create_latlon_mesh
scripts/run/run_omip.py-4819-            # Prime the build-once vertex-mask cache from the CONCRETE
scripts/run/run_omip.py-4820-            # state so the wrapper can build per-band masks host-side.
--
scripts/run/run_omip.py-4823-            spmd_step = make_sharded_ocean_step(model, _dev.mesh)
scripts/run/run_omip.py-4824-            spmd_gather = partial(gather_state_latlon, mesh=_dev.mesh)
scripts/run/run_omip.py-4825-            state = shard_state_latlon(state, _dev.mesh)
scripts/run/run_omip.py-4826-            # Lay per-block forcing stacks out lat-band-sharded so the
scripts/run/run_omip.py-4827-            # in-scan interpolation / bulk fluxes stay shard-local (shared
scripts/run/run_omip.py:4828:            # layout helper — see shard_forcing_stack_latlon).
scripts/run/run_omip.py:4829:            spmd_shard_stack = partial(
scripts/run/run_omip.py:4830:                shard_forcing_stack_latlon, mesh=_dev.mesh)
scripts/run/run_omip.py-4831-            if jax.process_index() == 0:
scripts/run/run_omip.py-4832-                _lane = "route-B multicontroller" if _multi else "single-controller"
scripts/run/run_omip.py-4833-                print(f"  SPMD ({_lane}): lat-band sharded dynamics step over "
scripts/run/run_omip.py-4834-                      f"{_nd} devices across {jax.process_count()} process(es) "
scripts/run/run_omip.py-4835-                      f"({jax.default_backend()}).")
--
scripts/run/run_omip.py-4856-        S_woa_3d=(S_woa * state.land_mask.data[..., jnp.newaxis]).astype(
scripts/run/run_omip.py-4857-            state.S.data.dtype) if args.nudge_woa_tau > 0 and S_woa is not None else None,
scripts/run/run_omip.py-4858-        snapshot_fn=_snapshot_fn,
scripts/run/run_omip.py-4859-        spmd_step=spmd_step,
scripts/run/run_omip.py-4860-        spmd_gather=spmd_gather,
scripts/run/run_omip.py:4861:        spmd_shard_stack=spmd_shard_stack,
scripts/run/run_omip.py-4862-    )
scripts/run/run_omip.py-4863-    if spmd_gather is not None:
scripts/run/run_omip.py-4864-        # Downstream report/plot/save paths expect the full (n_lat+1)
scripts/run/run_omip.py-4865-        # staggered v layout, not the sharded v_lower carry.  Every rank
scripts/run/run_omip.py-4866-        # dispatches this gather (it is a collective); only rank 0 writes.
439-            updates[name] = _addressable_shard_put(arr, NamedSharding(mesh, spec))
440-    return state._replace(**updates)
441-
442-
443:def shard_forcing_latlon(forcing, mesh):
444-    """Lay out a forcing pytree (``FreshwaterForcing`` / ``OceanSurfaceForcing``
445-    / ``SpongeForcing`` — or any nesting of them) for the lat-band SPMD step.
446-
447-    Every OMIP forcing field is CELL-CENTERED ``(n_lat, n_lon[, nlev])`` (the
--
467-    """Lay out a STACKED per-block forcing pytree for the lat-band SPMD
468-    block-scan (the ``run_omip`` JRA55 lanes; see
469-    ``_build_jra55_block_fn`` / ``_build_jra55_block_fn_interp``).
470-
471:    Unlike :func:`shard_forcing_latlon` (per-step, lat at axis 0), the
472-    block builders stack ``N`` steps / raw records along a LEADING axis,
473-    so the lat axis sits at position 1.  A ``(n_rec, n_lat, n_lon[, ...])``
474-    leaf therefore shards ``P(None, "lat", ...)`` (records replicated, the
475-    time index stays shard-local so the in-scan interpolation needs no
--
568-            updates[name] = _gather_arr(jnp.asarray(val))
569-    return state._replace(**updates)
570-
571-
572:def make_sharded_ocean_step(model, mesh):
573-    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
574-    sponge=None, t_seconds=None) -> state`` running ``model.step``
575-    lat-band-SPMD.
576-
577-    The forcing channels mirror ``model.step``'s keyword surface: pass
578:    pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
579-    ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
580-    is a replicated scalar like ``dt``).  ``None`` forcing keeps the
581-    dynamics-only program; each distinct None<->populated combination
582-    compiles (and caches) its own executable.
--
936-    that takes a GLOBAL (single-device-layout) state + forcing and returns a
937-    GLOBAL state — the minimal-diff driver entry point.
938-
939-    Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
940:    (:func:`shard_state_latlon` + :func:`shard_forcing_latlon`), runs the lat-band
941-    SPMD step, then gathers the state OUT (:func:`gather_state_latlon`).  This lets
942-    the OMIP host loop keep operating on a normal full-domain state — the per-step
943-    host BCs (SSS restore, prognostic ice, geothermal, BBL, nudge) see the
944-    gathered global state UNCHANGED — at the cost of a per-step gather/scatter
--
951-        return lambda state, dt, surface_forcing=None, freshwater=None: (
952-            model.step(state, dt, freshwater=freshwater,
953-                       surface_forcing=surface_forcing))
954-
955:    inner = make_sharded_ocean_step(model, mesh)
956-
957-    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
958-        # Scatter the global state to the band layout; the forcing is sharded
959-        # INSIDE ``inner`` (make_sharded_ocean_step lays it out), so pass it

exec
/usr/bin/bash -lc "sed -n '430,570p' packages/ocean/legoesm/ocean/state.py; rg -n -C 3 \"LatLonCGridOceanState\\(\" packages/ocean scripts tests -g '*.py' | head -n 1000; rg -n \"_replace\\(["'^'"\\n]*(T_flux_div_prev|S_flux_div_prev|u_explicit_prev|v_explicit_prev|eta_explicit_prev)|T_flux_div_prev=|S_flux_div_prev=|u_explicit_prev=|v_explicit_prev=|eta_explicit_prev=\" packages/ocean scripts tests -g '*.py' | head -n 1000" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 2023ms:
    # Prior-step ADVECTIVE TKE tendency [m^2/s^3] at the interior interfaces
    # (W-grid), 3-D Field (n_lat, n_lon, nlev-1) — the Adams-Bashforth history
    # dtke^{n-1} for the prognostic-TKE superbee advection (Veros vs.dtke,
    # tke.py:292-323), used only when vertical_mixing.tke.advection_scheme !=
    # "none" (requires prognostic=True). None on the first step => the AB2
    # increment uses a zero previous tendency, exactly like Veros's
    # zero-initialised dtke[taum1]. Default None -> inert: zero behaviour change.
    dtke: object = None
    # Carried EKE dissipation rate [m^2/s^3] at the interior interfaces (W-grid),
    # 3-D Field (n_lat, n_lon, nlev-1). Populated by the 3-D EKE step (Veros
    # eke_diss_iw, run in the GM/Redi stage) and consumed WITHIN THE SAME step by
    # the prognostic TKE source when vertical_mixing.tke.source_eke_diss=True
    # (the TKE source reads state_new.eke_diss — this step's EKE update — since
    # the implicit-vmix TKE solve runs after GM/Redi, matching Veros's same-step
    # eke->tke ordering). The CARRIED value on state is only the fallback when a
    # step produces none (e.g. the 2-D EKE path). Default None -> inert: zero
    # behaviour change.
    eke_diss: object = None
    # Prior EXPLICIT increment ΔX_expl^{n-1} for the AB2 outer integrator
    # (config.outer_integrator == "ab2"): the explicit-only forward-Euler increment
    # (advection, GM/Redi, lateral friction, Coriolis, barotropic solve, freshwater)
    # WITHOUT the implicit vertical mixing — implicit mixing is applied once, after
    # the AB2 extrapolation, and is NOT carried. Tracers store the full explicit
    # increment; u/v store the BAROCLINIC-deviation explicit increment (the barotropic
    # mode is kept from the barotropic solve, un-AB2'd). Default None -> inert
    # (forward-Euler): zero behaviour change.
    T_incr_prev: object = None
    S_incr_prev: object = None
    u_incr_prev: object = None
    v_incr_prev: object = None
    # NEMO Modified-Leap-Frog "before" state Nbb (t-dt) for
    # outer_integrator="leapfrog": the previous-step (Robert-Asselin-filtered)
    # now-fields, carried so the leapfrog explicit combine X(Naa)=X(Nbb)+2dt·RHS
    # and the Asselin filter X(Nnn)+=gamma·(X(Nbb)-2X(Nnn)+X(Naa)) have their
    # third time level. None on the very first step -> a forward-Euler start
    # (NEMO l_1st_euler) which then POPULATES these with the pre-step now-state.
    # Default None -> inert (forward-Euler/AB2 paths never read them): zero
    # behaviour change for existing configs.
    u_before: object = None
    v_before: object = None
    T_before: object = None
    S_before: object = None
    eta_before: object = None
    # Previous-step barotropic slow forcing (depth-mean tendency) for the
    # AB2 time-centering of F_slow (matches Oceananigans' AB2-extrapolated Gᵁ).
    # Only used when barotropic_slow_forcing_ab2=True; None otherwise (default).
    F_slow_u_prev: object = None
    F_slow_v_prev: object = None
    # Rigid-lid barotropic streamfunction state (config.barotropic_solver ==
    # "rigid_lid").  ``psi`` is the vertex-point streamfunction [m^3/s], shape
    # (n_lat+1, n_lon+1).  ``dpsi``/``dpsi_prev`` are the interior streamfunction
    # tendencies ∂ψ/∂t at the current/previous solved step (the AB2 history +
    # the leapfrog CG-guess history).  ``dpsin``/``dpsin_prev`` are the
    # per-island streamfunction-constant tendencies, shape (nisle,).  All default
    # None -> inert (free-surface path unaffected; zero behaviour change).
    psi: object = None
    dpsi: object = None
    dpsi_prev: object = None
    dpsin: object = None
    dpsin_prev: object = None
    # Cross-window barotropic AB3/AM4 substep histories for
    # barotropic_time_filter == "nemo_ab3am4" (NEMO dynspg_ts nn_bt_flt=3):
    # 6-tuple of 2-D arrays in DEVIATION form — (U_f-U_b, U_f-U_bb, V_f-V_b,
    # V_f-V_bb, eta_f-eta_b, eta_f-eta_bb), the last two substep values of
    # the previous window relative to its final value (NEMO's persistent
    # ubb_e/ub_e/vbb_e/vb_e/sshbb_e/sshb_e, written to NEMO's restart).
    # Deviation form because NEMO re-imposes the stp2d barotropic mean on the
    # 3D velocity after every stage (stprk3_stg.F90:440) so its raw histories
    # never see a window-boundary jump; legoESM's post-solve implicit vmix
    # shifts the depth mean, and raw carried values would feed that jump into
    # the AB3 extrapolation each window (see _run_substep_loop).
    # None (default) ⇒ cold start: the barotropic solver applies NEMO's
    # ll_init ramp and POPULATES this field; afterwards each window continues
    # the AB3 series across the window boundary (dynspg_ts.F90:200-226).
    # Same None-seeding pattern as the prognostic ``tke`` field. NB: cannot
    # be pre-seeded with zeros for lax.scan (zeros read as "continuation with
    # equal histories", silently skipping the cold-start ramp) — nemo_ab3am4
    # runs are step-1-eager, then scan.
    bt_hist: object = None
    # NEMO ln_bt_fw=.FALSE. CENTRED barotropic slow forcing (#1226 item 3;
    # dynspg_ts.F90:392-421): under the centred (non-forward) split-explicit
    # integration NEMO forces zu_frc/ssh_frc with the TIME-AVERAGE
    # ½(before+now) of the wind stress (utau_b+utauU) and the net freshwater
    # flux (emp_b+emp), NOT the plain now-value. These carry the PREVIOUS
    # step's forcing at the SAME representation as the ``surface_forcing``/
    # ``freshwater`` step() args (T-point, pre-rotation) so
    # ``barotropic_forcing_centred=True`` can rebuild the NEMO average
    # without re-deriving the tau sign/interp/rotation chain twice.
    # ``tau_x_prev``/``tau_y_prev`` mirror ``OceanSurfaceForcing.tau_x/tau_y``
    # (T-point, atmosphere convention [Pa]); ``freshwater_eta_prev`` is the
    # previous step's REDUCED net eta-forcing rate
    # (``freshwater_eta_tendency`` output, T-point [m/s]) rather than a full
    # carried ``FreshwaterForcing`` — only the eta/barotropic (NEMO
    # ``ssh_frc``) channel is centred; the separate virtual-salt-flux tracer
    # deposit (NEMO ``tra_sbc``) is untouched and stays at NOW. Default
    # None -> inert (barotropic_forcing_centred=False path never reads
    # these): zero behaviour change for existing configs. NEMO's nit000
    # seeding (sbcmod.F90:568-573, "before" set equal to "now" on step 1,
    # no restart) is reproduced by seeding these to the FIRST step's now
    # values rather than zero (done in ``_step_impl``/``_leapfrog_step``,
    # not here — this field only carries the state).
    tau_x_prev: object = None
    tau_y_prev: object = None
    freshwater_eta_prev: object = None


class SurfaceTracerForcing(NamedTuple):
    """Surface TRACER forcing RATE (restoring + prescribed q_net + penetrating
    shortwave) WITHHELD from the explicit ``dT_dt``/``dS_dt`` so the model step
    can apply it IMPLICITLY (weight 1.0, no AB2 extrapolation) inside the
    backward-Euler vertical-mixing solve — matching Veros's
    ``forc_temp_surface``/``forc_salt_surface`` placement
    (``veros/core/thermodynamics.py``: the surface forcing enters the implicit
    vertical-diffusion tridiagonal RHS, not the explicit AB2 tendency).

    Populated ONLY when ``LatLonCGridOceanConfig.surface_forcing_implicit`` is
    True; ``None`` otherwise (default), keeping the tendency pytree + every
    existing path bit-identical.

    The same (dT_dt, dS_dt) rate-pair container is REUSED for the implicit
    COLUMN tracer source on the SEPARATE
    ``LatLonCGridOceanTendencies.tracer_source`` slot
    (``sponge_forcing_implicit``, the Veros ``tempsalt_sources`` placement).
    The slots must stay distinct: the post-mixing TKE surface buoyancy-flux
    reconstruction column-sums THIS slot to rebuild Veros's
    ``forc_temp_surface`` and must exclude column sources (Veros keeps
    tempsalt_sources out of forc_rho_surface).

    Fields
    ------
    dT_dt : Field
        Surface temperature forcing rate [degC/s], full-column shape
        ``(n_lat, n_lon, nlev)`` — nonzero in the surface layer (index 0) for
        the restoring + non-solar q_net, plus the shortwave-penetration column.
    dS_dt : Field
        Surface salinity forcing rate [PSU/s], full-column shape; surface-layer
        restoring only (Veros ACC does not restore salinity).
    """
    dT_dt: Field
    dS_dt: Field

packages/ocean/legoesm/ocean/state.py-359-# Lat-Lon C-Grid FV Ocean State
packages/ocean/legoesm/ocean/state.py-360-# ==============================================================================
packages/ocean/legoesm/ocean/state.py-361-
packages/ocean/legoesm/ocean/state.py:362:class LatLonCGridOceanState(NamedTuple):
packages/ocean/legoesm/ocean/state.py-363-    """State for the lat-lon C-grid finite-volume ocean primitive equations.
packages/ocean/legoesm/ocean/state.py-364-
packages/ocean/legoesm/ocean/state.py-365-    Velocities live on cell faces (Arakawa C-grid staggering):
--
tests/parallel/test_latlon_mpi_tripole_serial.py-345-            T_flux_div_prev=F((n_lat, n_lon, nlev)),
tests/parallel/test_latlon_mpi_tripole_serial.py-346-            S_flux_div_prev=F((n_lat, n_lon, nlev)),
tests/parallel/test_latlon_mpi_tripole_serial.py-347-        )
tests/parallel/test_latlon_mpi_tripole_serial.py:348:    return LatLonCGridOceanState(
tests/parallel/test_latlon_mpi_tripole_serial.py-349-        u=F((n_lat, n_lon + 1, nlev)), v=F((n_lat + 1, n_lon, nlev)),
tests/parallel/test_latlon_mpi_tripole_serial.py-350-        T=F((n_lat, n_lon, nlev)), S=F((n_lat, n_lon, nlev)),
tests/parallel/test_latlon_mpi_tripole_serial.py-351-        eta=F((n_lat, n_lon)), H_bathy=F((n_lat, n_lon)),
--
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-247-    dims_u2d = ("lat", "lon_u")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-248-    dims_v2d = ("lat_v", "lon")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-249-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py:250:    return LatLonCGridOceanState(
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-251-        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-252-                staggering="edge"),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-253-        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
--
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-311-    dims_u2d = ("lat", "lon_u")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-312-    dims_v2d = ("lat_v", "lon")
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-313-
packages/ocean/legoesm/ocean/init_latlon_cgrid.py:314:    return LatLonCGridOceanState(
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-315-        u=Field(data=u_zeros, name="u", dims=dims_u, units="m/s",
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-316-                staggering="edge"),
packages/ocean/legoesm/ocean/init_latlon_cgrid.py-317-        v=Field(data=v_zeros, name="v", dims=dims_v, units="m/s",
--
scripts/run/continue_eady_uniform.py-87-        kw["T_som"] = _field(restart["T_som"], "T_som", ("lat", "lon", "z", "mom"))
scripts/run/continue_eady_uniform.py-88-    if "S_som" in restart.files:
scripts/run/continue_eady_uniform.py-89-        kw["S_som"] = _field(restart["S_som"], "S_som", ("lat", "lon", "z", "mom"))
scripts/run/continue_eady_uniform.py:90:    return LatLonCGridOceanState(**kw)
scripts/run/continue_eady_uniform.py-91-
scripts/run/continue_eady_uniform.py-92-
scripts/run/continue_eady_uniform.py-93-def _build_config_and_model(args):
--
tests/ocean/unit/test_bathymetry.py-1142-class TestLatLonCGridRestState:
tests/ocean/unit/test_bathymetry.py-1143-    """End-to-end rest_state_ocean_realistic on lat-lon C-grid."""
tests/ocean/unit/test_bathymetry.py-1144-
tests/ocean/unit/test_bathymetry.py:1145:    def test_idealized_yields_LatLonCGridOceanState(self, latlon_grid, z_coord):
tests/ocean/unit/test_bathymetry.py-1146-        from legoesm.ocean.state import LatLonCGridOceanState
tests/ocean/unit/test_bathymetry.py-1147-        cfg = BathymetryConfig(source="idealized", H_max=4000.0)
tests/ocean/unit/test_bathymetry.py-1148-        state = rest_state_ocean_realistic(latlon_grid, z_coord, cfg)
--
tests/ocean/unit/test_immersed_boundary.py-252-    nlat, nlon = land_mask_2d.shape
tests/ocean/unit/test_immersed_boundary.py-253-    um, vm = compute_face_masks(land_mask_2d)
tests/ocean/unit/test_immersed_boundary.py-254-    F = lambda d, n, dims, u="": Field(data=d, name=n, dims=dims, units=u)
tests/ocean/unit/test_immersed_boundary.py:255:    return LatLonCGridOceanState(
tests/ocean/unit/test_immersed_boundary.py-256-        u=F(jnp.zeros((nlat, nlon + 1)), "u", ("y", "x"), "m/s"),
tests/ocean/unit/test_immersed_boundary.py-257-        v=F(jnp.zeros((nlat + 1, nlon)), "v", ("y", "x"), "m/s"),
tests/ocean/unit/test_immersed_boundary.py-258-        T=F(jnp.zeros((nlat, nlon)), "T", ("y", "x"), "degC"),
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:3928:                    T_flux_div_prev=_Field(
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:3931:                    S_flux_div_prev=_Field(
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:4071:                    T_flux_div_prev=_Field(
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:4074:                    S_flux_div_prev=_Field(
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7433:                T_flux_div_prev=Field(
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7436:                S_flux_div_prev=Field(
tests/parallel/test_latlon_mpi_tripole_serial.py:345:            T_flux_div_prev=F((n_lat, n_lon, nlev)),
tests/parallel/test_latlon_mpi_tripole_serial.py:346:            S_flux_div_prev=F((n_lat, n_lon, nlev)),
tests/ocean/unit/test_prescribed_flow.py:408:        T_flux_div_prev=Field(data=poison, name="T_flux_div_prev",
tests/ocean/unit/test_prescribed_flow.py:410:        S_flux_div_prev=Field(data=-0.5 * poison, name="S_flux_div_prev",

codex
The direct `device_put` grep is clean outside the helper in this module, and the scaling bench has none. I’m checking the remaining indirect placement and whether removing JAX’s equality gate weakens the multi-rank input contract.
exec
/usr/bin/bash -lc "nl -ba packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py | sed -n '330,510p;930,970p'; nl -ba scripts/bench/bench_ocean_latlon_spmd_scaling.py | sed -n '368,460p'; rg -n -C 3 \"def shard_leaf|def replicate_leaf\" packages/core/legoesm/parallel/latlon_spmd.py; git diff --word-diff=plain HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 353ms:
   330	    n_lat = vmask.shape[0] - 1
   331	    nl = n_lat // n_dev
   332	    # v/q-row stagger: band r owns global vertex rows [r*nl : r*nl+nl+1] (the
   333	    # shared interior boundary vertex row appears in band r AND band r+1) — the
   334	    # same [s:e+1] slice slice_cgrid_geometry_to_band applies to area_q.
   335	    return [jnp.asarray(vmask[r * nl: r * nl + nl + 1]) for r in range(n_dev)]
   336	
   337	
   338	def _addressable_shard_put(arr, sharding):
   339	    """Put a host array onto a (possibly multi-process) sharding WITHOUT
   340	    jax's whole-array cross-process ``assert_equal``.
   341	
   342	    Under multicontroller, ``jax.device_put(numpy_array, sharding)`` calls
   343	    ``multihost_utils.assert_equal`` on the FULL array
   344	    (jax _src/dispatch.py::_device_put_sharding_impl) — a
   345	    ``process_allgather`` whose output is ``[n_processes, *shape]`` on one
   346	    device: at LL2304 L20 one 3-D field is 849 MB, so 96 processes fetch
   347	    81.5 GiB > an 80 GB A100 (the wall that killed oc @96/@128, jobs
   348	    26644681/26644682, AFTER the geometry-broadcast fix; @64 = 54.3 GiB
   349	    just fit, which is why smaller ladders never saw it).
   350	
   351	    ``jax.make_array_from_callback`` supplies each process's addressable
   352	    shards directly — no consistency collective. The layouts here are
   353	    band-sharded or replicated stacks of DETERMINISTICALLY-built host
   354	    values; cross-process identity of non-owned rows is not required, and
   355	    real geometry divergence is refused by the fingerprint gate at the
   356	    geometry puts.
   357	    """
   358	    host = np.asarray(arr)
   359	    if jax.process_count() <= 1:
   360	        return jax.device_put(jnp.asarray(host), sharding)
   361	    return jax.make_array_from_callback(
   362	        host.shape, sharding, lambda idx: host[idx])
   363	
   364	
   365	def shard_state_latlon(state, mesh):
   366	    """Lay out a ``LatLonCGridOceanState`` for the lat-band SPMD step.
   367	
   368	    Cell / u-grid array fields (leading dim ``n_lat``) shard ``P("lat")``.  The
   369	    staggered ``v`` / ``v_mask`` (leading dim ``n_lat+1``) are carried as
   370	    ``v_lower = field[0:n_lat]`` (``n_lat`` rows, ``P("lat")``) — the top pole row
   371	    ``field[n_lat]`` is a 0 wall on the regular grid and is reconstructed in-body
   372	    by the band halo.  ``None`` fields pass through.  The inverse is
   373	    :func:`gather_state_latlon`.
   374	
   375	    This is the layout the in-``shard_map`` body of :func:`make_sharded_ocean_step`
   376	    expects; the test uses it instead of a uniform ``tree.map(P("lat"))`` (which
   377	    fails on ``v`` because ``n_lat+1`` is not divisible by ``N``).
   378	    """
   379	    # v-carrier contract (see make_sharded_ocean_step's fold note): the TOP
   380	    # v-face row (regular pole wall OR tripole seam/cap row) must be
   381	    # wall-masked — the carrier drops it and reconstructs it as zero, which
   382	    # would silently delete a LIVE seam row.  Host-side check on the
   383	    # concrete state (this fn runs outside jit).
   384	    vm = getattr(state, "v_mask", None)
   385	    if vm is not None:
   386	        import numpy as _np
   387	
   388	        if _np.asarray(vm.data)[-1].any():
   389	            raise ValueError(
   390	                "shard_state_latlon: the state's TOP v-face row is LIVE "
   391	                "(v_mask[-1] has ocean faces) — the lat-band v-carrier "
   392	                "drops that row and reconstructs it as the pole/cap wall "
   393	                "zero, which would silently delete seam velocities. "
   394	                "Mask the cap row (the tripole cap convention) or extend "
   395	                "the carrier before sharding this state.")
   396	
   397	    def _shard_cell(field):
   398	        if field is None:
   399	            return None
   400	        sh = NamedSharding(mesh, _lat_spec(field.data))
   401	        return field.replace(data=_addressable_shard_put(field.data, sh))
   402	
   403	    def _shard_v(field):
   404	        if field is None:
   405	            return None
   406	        # Drop the TOP v-row (the north pole wall, v==0 on the regular grid) so
   407	        # the leading dim becomes n_lat (divisible by N).  ROUND-TRIP INVARIANT
   408	        # (codex finding): shard_state_latlon -> gather_state_latlon re-appends a
   409	        # ZERO top row, so the round-trip is a bitwise identity ONLY when the
   410	        # input's top v-row is already zero (a valid masked regular-grid state;
   411	        # the pole-wall BC enforces v[n_lat]=0 every step).  An arbitrary nonzero
   412	        # top row (e.g. a raw IC perturbation) is dropped -> reconstructed as 0:
   413	        # CORRECT for the dynamics (the pole wall zeros it at step 1) but not a
   414	        # bit round-trip of that one row.  NOT for a tripole north fold (raises
   415	        # in make_sharded_ocean_step) where the top row is a live fold partner.
   416	        nlat1 = field.data.shape[0]
   417	        v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
   418	        sh = NamedSharding(mesh, _lat_spec(v_lower))
   419	        return field.replace(data=_addressable_shard_put(v_lower, sh))
   420	
   421	    updates = {}
   422	    for name in _V_STAGGERED_STATE_FIELDS:
   423	        updates[name] = _shard_v(getattr(state, name))
   424	    # Every other (array) leaf: shard P("lat").  NamedTuple fields not in the
   425	    # v-staggered set and not None get the cell sharding; None stays None.
   426	    for name in state._fields:
   427	        if name in _V_STAGGERED_STATE_FIELDS:
   428	            continue
   429	        val = getattr(state, name)
   430	        if val is None:
   431	            updates[name] = None
   432	        elif hasattr(val, "data"):                 # a Field
   433	            updates[name] = _shard_cell(val)
   434	        else:
   435	            # Non-Field, non-None leaf (e.g. a raw array carry like psi).
   436	            # Shard 2-D+ on lat, replicate lower-rank — matches shard_pytree.
   437	            arr = jnp.asarray(val)
   438	            spec = _lat_spec(arr) if arr.ndim >= 1 else P()
   439	            updates[name] = _addressable_shard_put(arr, NamedSharding(mesh, spec))
   440	    return state._replace(**updates)
   441	
   442	
   443	def shard_forcing_latlon(forcing, mesh):
   444	    """Lay out a forcing pytree (``FreshwaterForcing`` / ``OceanSurfaceForcing``
   445	    / ``SpongeForcing`` — or any nesting of them) for the lat-band SPMD step.
   446	
   447	    Every OMIP forcing field is CELL-CENTERED ``(n_lat, n_lon[, nlev])`` (the
   448	    cell->face wind-stress interpolation happens INSIDE the step through the
   449	    SPMD-aware halo pads), so array leaves shard ``P("lat", None, ...)`` with
   450	    no ``v_lower`` handling; scalars replicate; ``None`` fields pass through
   451	    untouched (they vanish from the pytree structure, matching the specs the
   452	    step derives).  ``forcing=None`` returns ``None``.
   453	    """
   454	    if forcing is None or mesh is None:
   455	        return forcing
   456	
   457	    def _put(leaf):
   458	        if leaf is None:
   459	            return None
   460	        arr = jnp.asarray(leaf)
   461	        return _addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
   462	
   463	    return jax.tree.map(_put, forcing)
   464	
   465	
   466	def shard_forcing_stack_latlon(stack, mesh):
   467	    """Lay out a STACKED per-block forcing pytree for the lat-band SPMD
   468	    block-scan (the ``run_omip`` JRA55 lanes; see
   469	    ``_build_jra55_block_fn`` / ``_build_jra55_block_fn_interp``).
   470	
   471	    Unlike :func:`shard_forcing_latlon` (per-step, lat at axis 0), the
   472	    block builders stack ``N`` steps / raw records along a LEADING axis,
   473	    so the lat axis sits at position 1.  A ``(n_rec, n_lat, n_lon[, ...])``
   474	    leaf therefore shards ``P(None, "lat", ...)`` (records replicated, the
   475	    time index stays shard-local so the in-scan interpolation needs no
   476	    cross-band comm); a bare ``(n_lat, n_lon)`` leaf shards ``P("lat", None)``;
   477	    1-D metadata / scalars replicate; ``None`` and non-array leaves pass
   478	    through.  ``mesh=None`` returns ``stack`` unchanged (serial lane).
   479	
   480	    CONTRACT: rank is the ONLY signal used, so a rank-2 leaf is assumed to
   481	    be a ``(n_lat, n_lon)`` field and is lat-sharded on axis 0.  Any future
   482	    metadata that is genuinely rank-2 but NOT lat-major (e.g. a
   483	    ``(n_rec, n_meta)`` table) would be silently mis-sharded — keep such
   484	    metadata 1-D (or replicate it explicitly) before it reaches this helper.
   485	
   486	    Keeping this next to :func:`shard_state_latlon` means the driver and
   487	    the parity tests share ONE layout definition — the block-scan forcing
   488	    stack must be laid out consistently with the state the sharded step
   489	    carries, and a second copy would drift.
   490	    """
   491	    if mesh is None:
   492	        return stack
   493	
   494	    def _put(leaf):
   495	        if leaf is None or not hasattr(leaf, "ndim"):
   496	            return leaf
   497	        arr = jnp.asarray(leaf)
   498	        if arr.ndim >= 3:
   499	            spec = P(None, "lat", *((None,) * (arr.ndim - 2)))
   500	        elif arr.ndim == 2:
   501	            spec = P("lat", None)
   502	        else:
   503	            spec = P()
   504	        return _addressable_shard_put(arr, NamedSharding(mesh, spec))
   505	
   506	    return jax.tree.map(_put, stack)
   507	
   508	
   509	def append_vface_wall_row(v_lower):
   510	    """Rebuild the full ``(n_lat+1, ...)`` staggered v-array from the lat-band
   930	
   931	    return sharded_step
   932	
   933	
   934	def make_sharded_ocean_step_global(model, mesh):
   935	    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
   936	    that takes a GLOBAL (single-device-layout) state + forcing and returns a
   937	    GLOBAL state — the minimal-diff driver entry point.
   938	
   939	    Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
   940	    (:func:`shard_state_latlon` + :func:`shard_forcing_latlon`), runs the lat-band
   941	    SPMD step, then gathers the state OUT (:func:`gather_state_latlon`).  This lets
   942	    the OMIP host loop keep operating on a normal full-domain state — the per-step
   943	    host BCs (SSS restore, prognostic ice, geothermal, BBL, nudge) see the
   944	    gathered global state UNCHANGED — at the cost of a per-step gather/scatter
   945	    (acceptable for the host-coupled OMIP driver; the pure-dynamics inner loop
   946	    should use :func:`make_sharded_ocean_step` directly to stay sharded).
   947	
   948	    ``mesh is None`` ⇒ the plain single-device ``model.step`` (no scatter/gather).
   949	    """
   950	    if mesh is None:                   # single-device: plain step
   951	        return lambda state, dt, surface_forcing=None, freshwater=None: (
   952	            model.step(state, dt, freshwater=freshwater,
   953	                       surface_forcing=surface_forcing))
   954	
   955	    inner = make_sharded_ocean_step(model, mesh)
   956	
   957	    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
   958	        # Scatter the global state to the band layout; the forcing is sharded
   959	        # INSIDE ``inner`` (make_sharded_ocean_step lays it out), so pass it
   960	        # through global.
   961	        ss = shard_state_latlon(state, mesh)
   962	        ss = inner(ss, dt, surface_forcing=surface_forcing,
   963	                   freshwater=freshwater)
   964	        return gather_state_latlon(ss, mesh)
   965	
   966	    return sharded_step_global
   368	            f"--parity-gate is a smoke gate (re-association floor grows "
   369	            f"with steps); --steps {args.steps} > {SPMD_PARITY_MAX_STEPS} "
   370	            f"cap.")
   371	
   372	    if args.fused_halo:
   373	        # Trace-time switch — set BEFORE the sharded step is built/jitted.
   374	        os.environ["LEGOESM_LATLON_SPMD_FUSED_HALO"] = "1"
   375	
   376	    # #1370 fix stage (i), codex round-18: build the GLOBAL model/state on
   377	    # the HOST cpu backend, not the accelerator. rest-state init runs jnp
   378	    # ops at GLOBAL shape; on the default (GPU) device that materialises
   379	    # ~7.4 global-field-equivalents of setup residency per device AND
   380	    # compiles a global-sized init program — the 102 GB wall that killed
   381	    # LL2304@64 (probe 26523157: the compiled STEP is clean; the residency
   382	    # is setup-time). Host-side globals are RAM, and only the per-band
   383	    # shards reach the accelerator via shard_state_latlon's device_put.
   384	    # Identical values on every process (deterministic init + the step
   385	    # factory's existing process-0 broadcast + content-hash guard).
   386	    # nd==1 keeps the old on-device build: that lane TIMES the
   387	    # single-device step, so its state belongs on the accelerator.
   388	    import contextlib
   389	    _build_ctx = contextlib.nullcontext()
   390	    if nd > 1:
   391	        try:
   392	            # local_devices, NOT devices: under multicontroller jax.devices()
   393	            # returns the GLOBAL list, so [0] is process 0's cpu device
   394	            # — non-addressable elsewhere (probe job 26524163).
   395	            _build_ctx = jax.default_device(
   396	                jax.local_devices(backend="cpu")[0])
   397	        except RuntimeError:
   398	            # cpu backend not registered (JAX_PLATFORMS=cuda). The fix
   399	            # needs JAX_PLATFORMS=cuda,cpu; fall back to the old on-device
   400	            # build LOUDLY rather than crash.
   401	            print("[#1370] WARNING: no cpu backend — global init will "
   402	                  "materialise on the accelerator (set "
   403	                  "JAX_PLATFORMS=cuda,cpu to enable the host-side build)",
   404	                  flush=True)
   405	    with _build_ctx:
   406	        model, s0 = build_model_and_state(
   407	            n_lat, args.n_lon, args.nlev, seed=args.seed,
   408	            wide_halo=args.wide_halo, wide_halo_chunk=args.wide_halo_chunk,
   409	            tripole=args.tripole, baro_solver=args.baro_solver,
   410	            force_pcg=args.force_pcg, pcg_variant=args.pcg_variant,
   411	            pcg_fixed_iters=args.pcg_fixed_iters)
   412	        # Prime the build-once vertex-mask cache from the CONCRETE state so
   413	        # the wrapper can build the per-band vertex masks host-side (global
   414	        # 2-D — stays on the host under the nd>1 context).
   415	        model._ensure_vertex_mask(s0)
   416	
   417	    # Wet-cell weak metric (audit item 9): ACTIVE cell-levels from the state
   418	    # land mask (2-D column mask, z-star: a wet column is wet at all nlev
   419	    # levels — the bench_ocean_mpi_scaling wet-count convention).  Computed
   420	    # host-side from the pre-shard global state; bands are contiguous
   421	    # equal-row blocks, so the per-device split is an exact reshape.
   422	    _mask_np = np.asarray(s0.land_mask.data)
   423	    _wet_cols_per_dev = _mask_np.reshape(
   424	        nd, n_lat // nd, args.n_lon).sum(axis=(1, 2))
   425	    wet_rec = wet_cell_metrics(
   426	        wet_columns=float(_mask_np.sum()),
   427	        nlev=args.nlev,
   428	        total_cells=n_lat * args.n_lon * args.nlev,
   429	        n_devices=nd,
   430	        wet_columns_per_device=[float(w) for w in _wet_cols_per_dev],
   431	    )
   432	
   433	    # Parity reference: the plain single-device trajectory, computed BEFORE
   434	    # any sharding (deterministic identical build on every process).
   435	    serial_final = None
   436	    if args.parity_gate:
   437	        _s = s0
   438	        for _ in range(args.steps):
   439	            _s = model.step(_s, args.dt)
   440	        _block(_s)
   441	        serial_final = _s
   442	
   443	    inv_before = None
   444	    if args.check_conservation:
   445	        from bench_ocean_mpi_scaling import ocean_invariants
   446	        inv_before = ocean_invariants(model, s0, n_ranks=1)
   447	
   448	    if nd == 1:
   449	        mesh = None
   450	        step = make_sharded_ocean_step(model, None)
   451	        s = s0
   452	    else:
   453	        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
   454	                                 axis_names=("lat",))
   455	        step = make_sharded_ocean_step(model, mesh)
   456	        s = shard_state_latlon(s0, mesh)
   457	
   458	    # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
   459	    # blocks with sync only AROUND the block — the previous per-step
   460	    # host-synced loop measured dispatch+sync latency, not fused device
265-    return u_full[:, :-1]
266-
267-
268:def replicate_leaf(arr, rep, *, multiprocess: bool):
269-    """Replicate one (possibly lat-sharded) leaf onto every device of ``rep``'s
270-    mesh — the gather primitive shared by the atm and ocean lat-band SPMD steps
271-    (``gather_state_atm_latlon`` / ``gather_state_latlon``).
--
292-    return jax.device_put(arr, rep)
293-
294-
295:def shard_leaf(arr, sharding, *, multiprocess: bool):
296-    """Scatter one full-global leaf onto ``sharding``'s mesh — the SCATTER
297-    primitive symmetric to :func:`replicate_leaf`, shared by the atm and ocean
298-    lat-band SPMD steps (``shard_state_atm_latlon`` / ``shard_state_latlon``).
diff --git a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
index 406f4a76e..4991a56da 100644
--- a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
+++ b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
@@ -335,6 +335,33 @@ def _build_band_vertex_masks(model, n_dev):
    return [jnp.asarray(vmask[r * nl: r * nl + nl + 1]) for r in range(n_dev)]


{+def _addressable_shard_put(arr, sharding):+}
{+    """Put a host array onto a (possibly multi-process) sharding WITHOUT+}
{+    jax's whole-array cross-process ``assert_equal``.+}

{+    Under multicontroller, ``jax.device_put(numpy_array, sharding)`` calls+}
{+    ``multihost_utils.assert_equal`` on the FULL array+}
{+    (jax _src/dispatch.py::_device_put_sharding_impl) — a+}
{+    ``process_allgather`` whose output is ``[n_processes, *shape]`` on one+}
{+    device: at LL2304 L20 one 3-D field is 849 MB, so 96 processes fetch+}
{+    81.5 GiB > an 80 GB A100 (the wall that killed oc @96/@128, jobs+}
{+    26644681/26644682, AFTER the geometry-broadcast fix; @64 = 54.3 GiB+}
{+    just fit, which is why smaller ladders never saw it).+}

{+    ``jax.make_array_from_callback`` supplies each process's addressable+}
{+    shards directly — no consistency collective. The layouts here are+}
{+    band-sharded or replicated stacks of DETERMINISTICALLY-built host+}
{+    values; cross-process identity of non-owned rows is not required, and+}
{+    real geometry divergence is refused by the fingerprint gate at the+}
{+    geometry puts.+}
{+    """+}
{+    host = np.asarray(arr)+}
{+    if jax.process_count() <= 1:+}
{+        return jax.device_put(jnp.asarray(host), sharding)+}
{+    return jax.make_array_from_callback(+}
{+        host.shape, sharding, lambda idx: host[idx])+}


def shard_state_latlon(state, mesh):
    """Lay out a ``LatLonCGridOceanState`` for the lat-band SPMD step.

@@ -371,7 +398,7 @@ def shard_state_latlon(state, mesh):
        if field is None:
            return None
        sh = NamedSharding(mesh, _lat_spec(field.data))
        return [-field.replace(data=jax.device_put(field.data,-]{+field.replace(data=_addressable_shard_put(field.data,+} sh))

    def _shard_v(field):
        if field is None:
@@ -389,7 +416,7 @@ def shard_state_latlon(state, mesh):
        nlat1 = field.data.shape[0]
        v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
        sh = NamedSharding(mesh, _lat_spec(v_lower))
        return [-field.replace(data=jax.device_put(v_lower,-]{+field.replace(data=_addressable_shard_put(v_lower,+} sh))

    updates = {}
    for name in _V_STAGGERED_STATE_FIELDS:
@@ -409,7 +436,7 @@ def shard_state_latlon(state, mesh):
            # Shard 2-D+ on lat, replicate lower-rank — matches shard_pytree.
            arr = jnp.asarray(val)
            spec = _lat_spec(arr) if arr.ndim >= 1 else P()
            updates[name] = [-jax.device_put(arr,-]{+_addressable_shard_put(arr,+} NamedSharding(mesh, spec))
    return state._replace(**updates)


@@ -431,7 +458,7 @@ def shard_forcing_latlon(forcing, mesh):
        if leaf is None:
            return None
        arr = jnp.asarray(leaf)
        return [-jax.device_put(arr,-]{+_addressable_shard_put(arr,+} NamedSharding(mesh, _lat_spec(arr)))

    return jax.tree.map(_put, forcing)

@@ -474,7 +501,7 @@ def shard_forcing_stack_latlon(stack, mesh):
            spec = P("lat", None)
        else:
            spec = P()
        return [-jax.device_put(arr,-]{+_addressable_shard_put(arr,+} NamedSharding(mesh, spec))

    return jax.tree.map(_put, stack)


exec
/usr/bin/bash -lc "rg -n \"shard_forcing_latlon\\(\" --glob '*.py' .; rg -n \"make_sharded_ocean_step_global\\(\" --glob '*.py' .; rg -n \"jax\\.device_put\\(\" scripts/bench/bench_ocean_latlon_spmd_scaling.py packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py; rg -n \"make_array_from_callback\\(\" packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py packages/core/legoesm/parallel/latlon_spmd.py; rg -n -C 5 \"def test_.*shard|shard_forcing_stack|shard_forcing_latlon\" tests/parallel/test_latlon_ocean_spmd_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 1984ms:
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:443:def shard_forcing_latlon(forcing, mesh):
./tests/parallel/test_latlon_ocean_spmd_step.py:249:    fws = shard_forcing_latlon(fw, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py:250:    sfs = shard_forcing_latlon(sf, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py:251:    sponges = shard_forcing_latlon(sponge, dev.mesh)
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:934:def make_sharded_ocean_step_global(model, mesh):
./scripts/run/run_omip_core2.py:5589:            _spmd_step = make_sharded_ocean_step_global(model, _spmd_mesh)
./scripts/validate/validate_latlon_ocean_spmd_multiprocess.py:125:    step = make_sharded_ocean_step_global(model, dev.mesh)
./tests/parallel/test_latlon_ocean_spmd_step.py:461:    glob = make_sharded_ocean_step_global(model, dev.mesh)
./tests/parallel/test_persistent_sharded_ocean_loop.py:173:    glob = sos.make_sharded_ocean_step_global(model, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:342:    Under multicontroller, ``jax.device_put(numpy_array, sharding)`` calls
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:360:        return jax.device_put(jnp.asarray(host), sharding)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:361:    return jax.make_array_from_callback(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:712:        return jax.make_array_from_callback(
packages/core/legoesm/parallel/latlon_spmd.py:332:    return jax.make_array_from_callback(arr.shape, sharding, lambda idx: arr[idx])
tests/parallel/test_latlon_ocean_spmd_step.py-220-    """
tests/parallel/test_latlon_ocean_spmd_step.py-221-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-222-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py-223-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-224-        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py:225:        shard_forcing_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-226-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-227-    )
tests/parallel/test_latlon_ocean_spmd_step.py-228-
tests/parallel/test_latlon_ocean_spmd_step.py-229-    n_lat, n_lon, nlev = 48, 96, 10
tests/parallel/test_latlon_ocean_spmd_step.py-230-    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
--
tests/parallel/test_latlon_ocean_spmd_step.py-244-
tests/parallel/test_latlon_ocean_spmd_step.py-245-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-246-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py-247-    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-248-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:249:    fws = shard_forcing_latlon(fw, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:250:    sfs = shard_forcing_latlon(sf, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:251:    sponges = shard_forcing_latlon(sponge, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-252-
tests/parallel/test_latlon_ocean_spmd_step.py-253-    # Cache-key flip smoke: dynamics-only compile first, then the forcing
tests/parallel/test_latlon_ocean_spmd_step.py-254-    # program — the second call MUST rebuild (different forcing structure),
tests/parallel/test_latlon_ocean_spmd_step.py-255-    # not reuse the no-forcing specs.
tests/parallel/test_latlon_ocean_spmd_step.py-256-    _ = step(ss, dt)
--
tests/parallel/test_latlon_ocean_spmd_step.py-268-                                   err_msg=f"SPMD forcing {nm} mismatch")
tests/parallel/test_latlon_ocean_spmd_step.py-269-
tests/parallel/test_latlon_ocean_spmd_step.py-270-
tests/parallel/test_latlon_ocean_spmd_step.py-271-@pytest.mark.skipif(jax.device_count() < 4,
tests/parallel/test_latlon_ocean_spmd_step.py-272-                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_ocean_spmd_step.py:273:def test_shard_forcing_stack_latlon_layout():
tests/parallel/test_latlon_ocean_spmd_step.py-274-    """The block-scan stack sharder (run_omip JRA55 lanes) puts the lat axis
tests/parallel/test_latlon_ocean_spmd_step.py-275-    on the ``"lat"`` mesh axis whether the leaf is a stacked
tests/parallel/test_latlon_ocean_spmd_step.py-276-    ``(n_rec, n_lat, n_lon[, nlev])`` record (lat at axis 1) or a bare
tests/parallel/test_latlon_ocean_spmd_step.py-277-    ``(n_lat, n_lon)`` field (lat at axis 0); 1-D metadata and scalars
tests/parallel/test_latlon_ocean_spmd_step.py-278-    replicate; ``None`` / non-array leaves pass through untouched.  A drift
tests/parallel/test_latlon_ocean_spmd_step.py-279-    here silently commits the whole forcing stack to device 0 and serializes
tests/parallel/test_latlon_ocean_spmd_step.py-280-    every in-scan forcing op."""
tests/parallel/test_latlon_ocean_spmd_step.py-281-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-282-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py:283:        shard_forcing_stack_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-284-    )
tests/parallel/test_latlon_ocean_spmd_step.py-285-
tests/parallel/test_latlon_ocean_spmd_step.py-286-    def _lat_axis(arr):
tests/parallel/test_latlon_ocean_spmd_step.py-287-        spec = tuple(arr.sharding.spec)
tests/parallel/test_latlon_ocean_spmd_step.py-288-        return spec.index("lat") if "lat" in spec else None
--
tests/parallel/test_latlon_ocean_spmd_step.py-296-        "record_days": jnp.ones((n_rec,)),               # ndim 1: replicate
tests/parallel/test_latlon_ocean_spmd_step.py-297-        "scalar": jnp.asarray(3.0),                      # ndim 0: replicate
tests/parallel/test_latlon_ocean_spmd_step.py-298-        "none": None,                                    # pytree-None
tests/parallel/test_latlon_ocean_spmd_step.py-299-        "meta": "1958-01-01",                            # non-array passthrough
tests/parallel/test_latlon_ocean_spmd_step.py-300-    }
tests/parallel/test_latlon_ocean_spmd_step.py:301:    out = shard_forcing_stack_latlon(stack, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-302-
tests/parallel/test_latlon_ocean_spmd_step.py-303-    assert _lat_axis(out["rec3d"]) == 1
tests/parallel/test_latlon_ocean_spmd_step.py-304-    assert _lat_axis(out["rec4d"]) == 1
tests/parallel/test_latlon_ocean_spmd_step.py-305-    assert _lat_axis(out["field2d"]) == 0
tests/parallel/test_latlon_ocean_spmd_step.py-306-    assert _lat_axis(out["record_days"]) is None
tests/parallel/test_latlon_ocean_spmd_step.py-307-    assert _lat_axis(out["scalar"]) is None
tests/parallel/test_latlon_ocean_spmd_step.py-308-    assert out["none"] is None
tests/parallel/test_latlon_ocean_spmd_step.py-309-    assert out["meta"] == "1958-01-01"
tests/parallel/test_latlon_ocean_spmd_step.py-310-
tests/parallel/test_latlon_ocean_spmd_step.py-311-    # mesh=None is the serial-lane passthrough (identity).
tests/parallel/test_latlon_ocean_spmd_step.py:312:    assert shard_forcing_stack_latlon(stack, None) is stack
tests/parallel/test_latlon_ocean_spmd_step.py-313-
tests/parallel/test_latlon_ocean_spmd_step.py-314-
tests/parallel/test_latlon_ocean_spmd_step.py-315-@pytest.mark.skipif(jax.device_count() < 4,
tests/parallel/test_latlon_ocean_spmd_step.py-316-                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_ocean_spmd_step.py-317-def test_normalize_freshwater_net_psum_under_spmd():
--
tests/parallel/test_latlon_ocean_spmd_step.py-377-
tests/parallel/test_latlon_ocean_spmd_step.py-378-@pytest.mark.skipif(jax.device_count() < 4,
tests/parallel/test_latlon_ocean_spmd_step.py-379-                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_ocean_spmd_step.py-380-@pytest.mark.skipif(not _have_sharded_step(),
tests/parallel/test_latlon_ocean_spmd_step.py-381-                    reason="sharded_ocean_step module not present")
tests/parallel/test_latlon_ocean_spmd_step.py:382:def test_sharded_step_refuses_staggered_forcing():
tests/parallel/test_latlon_ocean_spmd_step.py-383-    """A (n_lat+1, n_lon) forcing leaf must fail LOUDLY at the wrapper, not
tests/parallel/test_latlon_ocean_spmd_step.py-384-    with an opaque shard_map divisibility error."""
tests/parallel/test_latlon_ocean_spmd_step.py-385-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-386-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py-387-        make_sharded_ocean_step,
--
tests/parallel/test_latlon_ocean_spmd_step.py-406-
tests/parallel/test_latlon_ocean_spmd_step.py-407-@pytest.mark.skipif(jax.device_count() < 4,
tests/parallel/test_latlon_ocean_spmd_step.py-408-                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_ocean_spmd_step.py-409-@pytest.mark.skipif(not _have_sharded_step(),
tests/parallel/test_latlon_ocean_spmd_step.py-410-                    reason="sharded_ocean_step module not present")
tests/parallel/test_latlon_ocean_spmd_step.py:411:def test_sharded_ocean_step_global_matches_explicit_scatter_gather():
tests/parallel/test_latlon_ocean_spmd_step.py-412-    """``make_sharded_ocean_step_global`` (global-in/global-out, the minimal
tests/parallel/test_latlon_ocean_spmd_step.py-413-    driver entry) must equal the explicit ``shard_state_latlon`` -> inner step ->
tests/parallel/test_latlon_ocean_spmd_step.py-414-    ``gather_state_latlon`` path BIT-FOR-BIT (it is literally that composition),
tests/parallel/test_latlon_ocean_spmd_step.py-415-    AND match the single-device reference to the same re-association floor.
tests/parallel/test_latlon_ocean_spmd_step.py-416-

codex
1. `HIGH` — The callback removes the only full-input equality check for state ICs and forcing, while the geometry fingerprint covers geometry only. A rank-local mismatch now silently becomes live physics (including `P()` replicated scalar/1-D metadata). Add a cheap multi-process structured byte fingerprint for each complete state/forcing input before callbacks; exact is appropriate if preserving `device_put`’s contract.

2. `HIGH` — “Single-process unchanged” is false. [`_addressable_shard_put`]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:358) calls `np.asarray` before branching, causing JAX device arrays to round-trip device→host→device even at one process. It also rejects non-fully-addressable inputs. Branch first, or reuse existing `latlon_spmd.shard_leaf`, which preserves the old single-process `device_put`.

3. `MEDIUM` — Direct grep is clean: no remaining direct `device_put` in the bench, and the module’s sole remaining call is the intended single-process helper branch. However, [`make_sharded_ocean_step_global`]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:957) claims forcing is sharded inside `inner`, but forwards global forcing directly. That remains an implicit JIT input-placement path, not this helper. Explicitly call `shard_forcing_latlon` there or correct the contract and test it under multicontroller.

4. `LOW` — Callback semantics for `P()` are correct: a scalar gets `idx=()`, a replicated 1-D leaf gets the full slice on every addressable local device. That replication is intentional device-memory duplication, not an `n_proc × full-array` all-gather. Note that state 1-D non-Field leaves are actually `P("lat")`, despite the nearby “replicate lower-rank” comment; stacked-forcing 1-D metadata is `P()`.

5. `LOW` — The docstring overclaims: inputs are often JAX device arrays, not necessarily host arrays; state/forcing are not geometry-gated; and `849 MB × 96` is 81.5 GB, not 81.5 GiB. The bench comment still says shards reach accelerators via `device_put`.

The 23 virtual-device tests do not execute the callback branch. The queued @96/@128 pair is the right OOM falsification, but add a small real-multicontroller negative test for divergent state/forcing after restoring a compact fingerprint gate.

VERDICT: FIX-FIRST
tokens used
125,669
1. `HIGH` — The callback removes the only full-input equality check for state ICs and forcing, while the geometry fingerprint covers geometry only. A rank-local mismatch now silently becomes live physics (including `P()` replicated scalar/1-D metadata). Add a cheap multi-process structured byte fingerprint for each complete state/forcing input before callbacks; exact is appropriate if preserving `device_put`’s contract.

2. `HIGH` — “Single-process unchanged” is false. [`_addressable_shard_put`]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:358) calls `np.asarray` before branching, causing JAX device arrays to round-trip device→host→device even at one process. It also rejects non-fully-addressable inputs. Branch first, or reuse existing `latlon_spmd.shard_leaf`, which preserves the old single-process `device_put`.

3. `MEDIUM` — Direct grep is clean: no remaining direct `device_put` in the bench, and the module’s sole remaining call is the intended single-process helper branch. However, [`make_sharded_ocean_step_global`]( /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:957) claims forcing is sharded inside `inner`, but forwards global forcing directly. That remains an implicit JIT input-placement path, not this helper. Explicitly call `shard_forcing_latlon` there or correct the contract and test it under multicontroller.

4. `LOW` — Callback semantics for `P()` are correct: a scalar gets `idx=()`, a replicated 1-D leaf gets the full slice on every addressable local device. That replication is intentional device-memory duplication, not an `n_proc × full-array` all-gather. Note that state 1-D non-Field leaves are actually `P("lat")`, despite the nearby “replicate lower-rank” comment; stacked-forcing 1-D metadata is `P()`.

5. `LOW` — The docstring overclaims: inputs are often JAX device arrays, not necessarily host arrays; state/forcing are not geometry-gated; and `849 MB × 96` is 81.5 GB, not 81.5 GiB. The bench comment still says shards reach accelerators via `device_put`.

The 23 virtual-device tests do not execute the callback branch. The queued @96/@128 pair is the right OOM falsification, but add a small real-multicontroller negative test for divergent state/forcing after restoring a compact fingerprint gate.

VERDICT: FIX-FIRST
