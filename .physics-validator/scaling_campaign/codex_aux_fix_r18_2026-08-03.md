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
session id: 019fc649-bc52-7db3-be77-b0213bafbb19
--------
user
Round-18: review the aux-threading fix chain (git diff HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py scripts/bench/metadata.py scripts/bench/bench_ocean_latlon_spmd_scaling.py). Discovery chain: post-assert-fix @96/@128 failed at metadata timed_scan_blocks with 'Fetching value for jax.Array that spans non-addressable devices'; 2-process CPU repro reproduced it AND showed the OLD device_put path fails identically — the multicontroller wide+explicit ocean lane has been broken since #1370-iii sharded the geometry stacks (all later failures masked by the arg wall); a patched MLIR constant handler named the captured array: a (n_dev,band,lon) P('lat') geometry stack — jit-of-jit inlining turns concrete inner-jit args into outer-trace constants.
Fix: (a) sharded_step gains aux=None param + step.aux attribute (geom_stacks, vmask_stack), inner fn uses aux when given; (b) timed_scan_blocks gains aux=None threading it through @jax.jit _scan_run(st, _aux) as an ARGUMENT, with a normalized 2-arg _advance used at compile/probe/scan sites; (c) the ocean bench passes aux=step.aux with a 2-arg lambda. 2-proc repro now emits a full receipt row (15.09 ms, multicontroller, gloo) — first working multicontroller ocean run since #1370-iii; 28 tests pass (parity+fused-halo+fingerprint+bench units).
Attack: (1) aux default None path — any caller of timed_scan_blocks broken (atm bench, mpas bench call it 1-arg)? (2) the step's aux override semantics — cache key doesn't include aux structure; can a caller pass a WRONG-shaped aux hitting a stale compiled fn? Should aux be validated (tree structure == closure stacks')? (3) probe/compile top-level _advance(state, aux) — for aux=None advance 1-arg wrapped — correct? (4) measurement contract: does threading aux change what's timed (aux transfer per call?) — aux is already on device, passing as jit arg per call = no copy, but verify no per-call host transfer; (5) sharded_step_global path unchanged (calls inner without aux at top level — fine?); (6) driver/run_omip callers of make_sharded_ocean_step — any outer-jit wrap there that now needs aux (grep). End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + numbered issues.
codex
I’ll audit the specified diff and trace every affected caller, focusing on the six attack points and the compilation/timing semantics.
exec
/usr/bin/bash -lc "git diff HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py scripts/bench/metadata.py scripts/bench/bench_ocean_latlon_spmd_scaling.py && rg -n -C 3 \"timed_scan_blocks|make_sharded_ocean_step|sharded_step_global|\\.aux\\b\" packages scripts tests 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 4024ms:
diff --git a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
index 406f4a76e..3059ddbd1 100644
--- a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
+++ b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
@@ -335,6 +335,67 @@ def _build_band_vertex_masks(model, n_dev):
     return [jnp.asarray(vmask[r * nl: r * nl + nl + 1]) for r in range(n_dev)]
 
 
+def _addressable_shard_put(arr, sharding):
+    """Put an array onto a (possibly multi-process) sharding WITHOUT jax's
+    whole-array cross-process ``assert_equal``.
+
+    Under multicontroller, ``jax.device_put(numpy_array, sharding)`` calls
+    ``multihost_utils.assert_equal`` on the FULL array
+    (jax _src/dispatch.py::_device_put_sharding_impl) — a
+    ``process_allgather`` whose output is ``[n_processes, *shape]`` on one
+    device: at LL2304 L20 one 3-D field is 849 MB, so 96 processes fetch
+    81.5 GB > an 80 GB A100 (the wall that killed oc @96/@128, jobs
+    26644681/26644682, AFTER the geometry-broadcast fix; @64 = 54.3 GB
+    just fit, which is why smaller ladders never saw it).
+
+    Single-process: the historical ``jax.device_put``, byte-unchanged and
+    with no host round-trip (codex r17 — the input may already be a jax
+    device array). Multicontroller: ``jax.make_array_from_callback``
+    supplies each process's addressable shards directly — no consistency
+    collective. The ``device_put`` bit-identity CONTRACT is preserved by
+    the callers' cheap exact-hash gate (:func:`assert_pytree_bytes_equal`)
+    instead of jax's full-array allgather.
+    """
+    if jax.process_count() <= 1:
+        return jax.device_put(arr, sharding)
+    host = np.asarray(arr)
+    return jax.make_array_from_callback(
+        host.shape, sharding, lambda idx: host[idx])
+
+
+def assert_pytree_bytes_equal(tree, what):
+    """Cheap multi-process replacement for the per-leaf ``assert_equal``
+    that :func:`_addressable_shard_put` bypasses (codex r17 HIGH-1).
+
+    Hashes every array leaf's BYTES (48-bit positional digest — the same
+    exactness as ``device_put``'s contract) into one small vector,
+    allgathers it, and refuses on any cross-process mismatch. Cost is one
+    tiny collective + a host-side hash pass, independent of process count
+    — vs jax's [n_processes, full_array] allgather.
+
+    No-op single-process. Symmetric: every process hashes the same leaves
+    in the same order, so all raise or none.
+    """
+    if jax.process_count() <= 1:
+        return
+    from jax.experimental import multihost_utils
+
+    leaves = [x for x in jax.tree_util.tree_leaves(tree)
+              if hasattr(x, "ndim")]
+    vals = np.array([_content_hash48(np.asarray(x)) for x in leaves],
+                    dtype=np.float64)
+    g = multihost_utils.process_allgather(vals)
+    if not bool(np.all(g == g[0])):
+        bad = [i for i in range(len(leaves))
+               if not bool(np.all(g[:, i] == g[0, i]))]
+        raise RuntimeError(
+            f"{what}: array leaves {bad} differ across processes (48-bit "
+            f"byte digests disagree) — the inputs each process built are "
+            f"NOT identical, which the removed jax device_put assert "
+            f"would have refused. Fix the per-process build before "
+            f"sharding.")
+
+
 def shard_state_latlon(state, mesh):
     """Lay out a ``LatLonCGridOceanState`` for the lat-band SPMD step.
 
@@ -354,6 +415,7 @@ def shard_state_latlon(state, mesh):
     # wall-masked — the carrier drops it and reconstructs it as zero, which
     # would silently delete a LIVE seam row.  Host-side check on the
     # concrete state (this fn runs outside jit).
+    assert_pytree_bytes_equal(state, "shard_state_latlon")
     vm = getattr(state, "v_mask", None)
     if vm is not None:
         import numpy as _np
@@ -371,7 +433,7 @@ def shard_state_latlon(state, mesh):
         if field is None:
             return None
         sh = NamedSharding(mesh, _lat_spec(field.data))
-        return field.replace(data=jax.device_put(field.data, sh))
+        return field.replace(data=_addressable_shard_put(field.data, sh))
 
     def _shard_v(field):
         if field is None:
@@ -389,7 +451,7 @@ def shard_state_latlon(state, mesh):
         nlat1 = field.data.shape[0]
         v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
         sh = NamedSharding(mesh, _lat_spec(v_lower))
-        return field.replace(data=jax.device_put(v_lower, sh))
+        return field.replace(data=_addressable_shard_put(v_lower, sh))
 
     updates = {}
     for name in _V_STAGGERED_STATE_FIELDS:
@@ -406,10 +468,12 @@ def shard_state_latlon(state, mesh):
             updates[name] = _shard_cell(val)
         else:
             # Non-Field, non-None leaf (e.g. a raw array carry like psi).
-            # Shard 2-D+ on lat, replicate lower-rank — matches shard_pytree.
+            # ndim>=1 lat-shards via _lat_spec (1-D included); only true
+            # scalars replicate (codex r17: the old "replicate lower-rank"
+            # wording did not match _lat_spec's behaviour).
             arr = jnp.asarray(val)
             spec = _lat_spec(arr) if arr.ndim >= 1 else P()
-            updates[name] = jax.device_put(arr, NamedSharding(mesh, spec))
+            updates[name] = _addressable_shard_put(arr, NamedSharding(mesh, spec))
     return state._replace(**updates)
 
 
@@ -426,12 +490,13 @@ def shard_forcing_latlon(forcing, mesh):
     """
     if forcing is None or mesh is None:
         return forcing
+    assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")
 
     def _put(leaf):
         if leaf is None:
             return None
         arr = jnp.asarray(leaf)
-        return jax.device_put(arr, NamedSharding(mesh, _lat_spec(arr)))
+        return _addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
 
     return jax.tree.map(_put, forcing)
 
@@ -464,6 +529,8 @@ def shard_forcing_stack_latlon(stack, mesh):
     if mesh is None:
         return stack
 
+    assert_pytree_bytes_equal(stack, "shard_forcing_stack_latlon")
+
     def _put(leaf):
         if leaf is None or not hasattr(leaf, "ndim"):
             return leaf
@@ -474,7 +541,7 @@ def shard_forcing_stack_latlon(stack, mesh):
             spec = P("lat", None)
         else:
             spec = P()
-        return jax.device_put(arr, NamedSharding(mesh, spec))
+        return _addressable_shard_put(arr, NamedSharding(mesh, spec))
 
     return jax.tree.map(_put, stack)
 
@@ -825,7 +892,18 @@ def make_sharded_ocean_step(model, mesh):
                     f"(n_lat, n_lon[, nlev]) to shard on the lat axis.")
 
     def sharded_step(state, dt, freshwater=None, surface_forcing=None,
-                     sponge=None, t_seconds=None):
+                     sponge=None, t_seconds=None, aux=None):
+        # ``aux`` (codex/2-proc repro 2026-08-03): the geometry + vmask
+        # stacks are SHARDED global arrays; when this wrapper runs INSIDE
+        # an outer trace (a bench/driver ``jit``/``scan`` over the step —
+        # jit-of-jit inlines the inner call), concrete closure arrays
+        # become OUTER-TRACE CONSTANTS and jax's MLIR constant handler
+        # tries to fetch their value — impossible for non-addressable
+        # arrays (RuntimeError: 'Fetching value ... non-addressable'; the
+        # multicontroller lane has been broken this way since the
+        # #1370-iii stack sharding). Callers that wrap the step in their
+        # own jit MUST thread ``step.aux`` through their jit boundary as
+        # an ARGUMENT and pass it back here.
         # ONE forcing operand: None fields drop out of the pytree structure,
         # so specs derived by tree.map skip them automatically and the
         # structure key below distinguishes every None<->array combination.
@@ -895,12 +973,16 @@ def make_sharded_ocean_step(model, mesh):
         _prev_mesh = get_spmd_mesh()
         activate_latlon_spmd_halo(mesh)
         try:
-            return fn(state, forcing, geom_stacks, vmask_stack,
-                      jnp.asarray(dt))
+            _geom, _vmask = aux if aux is not None else (geom_stacks,
+                                                        vmask_stack)
+            return fn(state, forcing, _geom, _vmask, jnp.asarray(dt))
         finally:
             set_spmd_mesh(_prev_mesh)
             set_halo_backend(_prev_backend, _prev_topo)
 
+    # Expose the stacks so outer-jit callers can pass them as arguments
+    # (see the ``aux`` note in the signature).
+    sharded_step.aux = (geom_stacks, vmask_stack)
     return sharded_step
 
 
@@ -928,12 +1010,15 @@ def make_sharded_ocean_step_global(model, mesh):
     inner = make_sharded_ocean_step(model, mesh)
 
     def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
-        # Scatter the global state to the band layout; the forcing is sharded
-        # INSIDE ``inner`` (make_sharded_ocean_step lays it out), so pass it
-        # through global.
+        # Scatter the global state AND forcing to the band layout explicitly
+        # (codex r17 item 3: the old comment claimed inner sharded the
+        # forcing; it forwarded it global and relied on implicit JIT input
+        # placement — which under multicontroller pays jax's whole-array
+        # device_put assert, the nd-linear wall this module removes).
         ss = shard_state_latlon(state, mesh)
-        ss = inner(ss, dt, surface_forcing=surface_forcing,
-                   freshwater=freshwater)
+        ss = inner(ss, dt,
+                   surface_forcing=shard_forcing_latlon(surface_forcing, mesh),
+                   freshwater=shard_forcing_latlon(freshwater, mesh))
         return gather_state_latlon(ss, mesh)
 
     return sharded_step_global
diff --git a/scripts/bench/bench_ocean_latlon_spmd_scaling.py b/scripts/bench/bench_ocean_latlon_spmd_scaling.py
index c7970156d..037056dbf 100644
--- a/scripts/bench/bench_ocean_latlon_spmd_scaling.py
+++ b/scripts/bench/bench_ocean_latlon_spmd_scaling.py
@@ -471,10 +471,16 @@ def main() -> int:
         _blk, _nblk, _probe = max(0, args.steps - 1), 1, 0
     else:
         _blk, _nblk, _probe = args.steps, args.blocks, args.probe_steps
+    # aux threads the sharded geometry stacks through the jit boundary as
+    # an ARGUMENT (outer-trace constants of non-addressable arrays are
+    # unfetchable — see make_sharded_ocean_step's aux note).
+    _aux = getattr(step, "aux", None)
     s, timing = timed_scan_blocks(
-        lambda st: step(st, args.dt), s,
+        (lambda st, aux: step(st, args.dt, aux=aux)) if _aux is not None
+        else (lambda st: step(st, args.dt)),
+        s,
         block_steps=_blk, n_blocks=_nblk, probe_steps=_probe,
-        sync_label="ocean_latlon_spmd_bench")
+        sync_label="ocean_latlon_spmd_bench", aux=_aux)
 
     # Post-run ZERO-FORCING residual probe eligibility (audit item 6):
     # needs the gathered global final state on ONE process; multicontroller
diff --git a/scripts/bench/metadata.py b/scripts/bench/metadata.py
index de91e4a01..f44f4f565 100644
--- a/scripts/bench/metadata.py
+++ b/scripts/bench/metadata.py
@@ -895,6 +895,7 @@ def timed_scan_blocks(
     n_blocks: int = 2,
     probe_steps: int = 3,
     sync_label: str = "timed_scan_blocks",
+    aux=None,
 ):
     """Measurement-contract timing: fused ``lax.scan`` blocks + probe latency.
 
@@ -1005,10 +1006,18 @@ def timed_scan_blocks(
             from jax.experimental import multihost_utils
             multihost_utils.sync_global_devices(f"{sync_label}_{tag}")
 
+    # Normalize to a 2-arg advance (see the aux note below): top-level
+    # calls here are outside any trace, so passing aux is just an argument.
+    if aux is None:
+        def _advance(carry, _aux):
+            return advance(carry)
+    else:
+        _advance = advance
+
     # --- 1. compile (first call of advance, separated from all timing) ---
     _fence("compile_start")
     t0 = time.perf_counter()
-    state = advance(state)
+    state = _advance(state, aux)
     _block(state)
     compile_ms = (time.perf_counter() - t0) * 1e3
 
@@ -1017,7 +1026,7 @@ def timed_scan_blocks(
     probe_ms = []
     for _ in range(probe_steps):
         t0 = time.perf_counter()
-        state = advance(state)
+        state = _advance(state, aux)
         _block(state)
         probe_ms.append((time.perf_counter() - t0) * 1e3)
     step_latency_ms = float(np.median(probe_ms)) if probe_ms else float("nan")
@@ -1026,10 +1035,16 @@ def timed_scan_blocks(
     input_dtypes = jax.tree_util.tree_map(
         lambda x: x.dtype if hasattr(x, "dtype") else None, state)
 
+    # ``aux``: extra pytree threaded through the jit boundary as an ARGUMENT
+    # (never a closure capture). Needed when ``advance`` internally calls a
+    # jit function over SHARDED-global auxiliary arrays (the ocean band
+    # geometry stacks): jit-of-jit inlines the inner call, so concrete
+    # closure arrays would become outer-trace CONSTANTS whose value the MLIR
+    # handler cannot fetch for non-addressable arrays (2026-08-03 repro).
     @jax.jit
-    def _scan_run(st):
+    def _scan_run(st, _aux):
         def _body(carry, _):
-            new = advance(carry)
+            new = _advance(carry, _aux)
             new = jax.tree_util.tree_map(
                 lambda x, d: x.astype(d)
                 if d is not None and hasattr(x, "astype") else x,
@@ -1046,7 +1061,7 @@ def timed_scan_blocks(
     # schedule — counts stay matched.
     _pre = jax.tree_util.tree_map(lambda x: x, state)
     t0 = time.perf_counter()
-    _pre_out = _scan_run(_pre)
+    _pre_out = _scan_run(_pre, aux)
     _block(_pre_out)
     scan_compile_ms = (time.perf_counter() - t0) * 1e3
     del _pre, _pre_out
@@ -1055,7 +1070,7 @@ def timed_scan_blocks(
     for b in range(n_blocks):
         _fence(f"block{b}_start")
         t0 = time.perf_counter()
-        state = _scan_run(state)
+        state = _scan_run(state, aux)
         _block(state)
         block_ms.append((time.perf_counter() - t0) * 1e3)
     _fence("blocks_end")
scripts/bench/bench_atm_latlon_spmd_scaling.py-3-
scripts/bench/bench_atm_latlon_spmd_scaling.py-4-Times the SHARDED step across an N-device ("lat",) mesh and reports per-step
scripts/bench/bench_atm_latlon_spmd_scaling.py-5-wall time + speedup vs 1 device.  The DEFAULT lane follows the M1 measurement
scripts/bench/bench_atm_latlon_spmd_scaling.py:6:contract (``metadata.timed_scan_blocks``): fused ``lax.scan`` blocks of
scripts/bench/bench_atm_latlon_spmd_scaling.py-7-``--steps`` steps with device sync only AROUND each block (``fused_step_ms``,
scripts/bench/bench_atm_latlon_spmd_scaling.py-8-slowest process across controllers) plus a SEPARATE individually-synced
scripts/bench/bench_atm_latlon_spmd_scaling.py-9-dispatch-latency probe (``step_latency_ms``) — never mixed.  The
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-175-    args = p.parse_args()
scripts/bench/bench_atm_latlon_spmd_scaling.py-176-
scripts/bench/bench_atm_latlon_spmd_scaling.py-177-    # Validate the schedule BEFORE any model/device work: a zero/negative
scripts/bench/bench_atm_latlon_spmd_scaling.py:178:    # --steps would otherwise surface only as timed_scan_blocks' None
scripts/bench/bench_atm_latlon_spmd_scaling.py-179-    # headline (default lane) or an empty timed loop (segment mode) after
scripts/bench/bench_atm_latlon_spmd_scaling.py-180-    # the expensive build (the ocean twin's guard).
scripts/bench/bench_atm_latlon_spmd_scaling.py-181-    if args.steps < 1:
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-247-    per_block_ms = None
scripts/bench/bench_atm_latlon_spmd_scaling.py-248-    completed_blocks = None
scripts/bench/bench_atm_latlon_spmd_scaling.py-249-    finite_ok = None   # default fused lane: no in-graph finite check -> null
scripts/bench/bench_atm_latlon_spmd_scaling.py:250:    timing = None   # metadata.timed_scan_blocks metrics (default lane only)
scripts/bench/bench_atm_latlon_spmd_scaling.py-251-    if seg_n > 0:
scripts/bench/bench_atm_latlon_spmd_scaling.py-252-        # Multi-controller: align every process before the timed loop so
scripts/bench/bench_atm_latlon_spmd_scaling.py-253-        # block wall times aren't skewed by startup jitter (and once after,
scripts/bench/bench_atm_latlon_spmd_scaling.py-254-        # so no process exits while peers still hold collectives in flight).
scripts/bench/bench_atm_latlon_spmd_scaling.py:255:        # (The default lane's fences live inside timed_scan_blocks.)
scripts/bench/bench_atm_latlon_spmd_scaling.py-256-        if jax.process_count() > 1:
scripts/bench/bench_atm_latlon_spmd_scaling.py-257-            from jax.experimental import multihost_utils
scripts/bench/bench_atm_latlon_spmd_scaling.py-258-            multihost_utils.sync_global_devices("atm_latlon_spmd_bench_start")
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-298-        # throughput.  Dispatch latency stays measured SEPARATELY
scripts/bench/bench_atm_latlon_spmd_scaling.py-299-        # (``step_latency_ms``); multi-controller runs record the
scripts/bench/bench_atm_latlon_spmd_scaling.py-300-        # slowest-process block time + imbalance ratio.
scripts/bench/bench_atm_latlon_spmd_scaling.py:301:        from metadata import timed_scan_blocks
scripts/bench/bench_atm_latlon_spmd_scaling.py:302:        c, timing = timed_scan_blocks(
scripts/bench/bench_atm_latlon_spmd_scaling.py-303-            lambda st: step(st, args.dt), c,
scripts/bench/bench_atm_latlon_spmd_scaling.py-304-            block_steps=args.steps, n_blocks=args.blocks,
scripts/bench/bench_atm_latlon_spmd_scaling.py-305-            probe_steps=args.probe_steps,
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-354-        halo_messages_per_step=comm_rec["halo_messages_per_step"],
scripts/bench/bench_atm_latlon_spmd_scaling.py-355-        halo_bytes_per_step=comm_rec["halo_bytes_per_step"],
scripts/bench/bench_atm_latlon_spmd_scaling.py-356-        n_reductions_per_step=_nred,
scripts/bench/bench_atm_latlon_spmd_scaling.py:357:        # rank imbalance is measured by timed_scan_blocks (default lane);
scripts/bench/bench_atm_latlon_spmd_scaling.py-358-        # the segment lane records no cross-process block gather -> null.
scripts/bench/bench_atm_latlon_spmd_scaling.py-359-        rank_imbalance=(float(timing["rank_imbalance"])
scripts/bench/bench_atm_latlon_spmd_scaling.py-360-                        if timing is not None else None),
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-390-            per_block_ms=[round(x, 2) for x in per_block_ms],
scripts/bench/bench_atm_latlon_spmd_scaling.py-391-        )
scripts/bench/bench_atm_latlon_spmd_scaling.py-392-    else:
scripts/bench/bench_atm_latlon_spmd_scaling.py:393:        # Default lane: the timed_scan_blocks metrics (fused_step_ms,
scripts/bench/bench_atm_latlon_spmd_scaling.py-394-        # step_latency_ms, block_ms, parallel_block_ms, rank_imbalance, ...
scripts/bench/bench_atm_latlon_spmd_scaling.py-395-        # — the M1 measurement contract).
scripts/bench/bench_atm_latlon_spmd_scaling.py-396-        rec.update(**timing)
--
scripts/bench/bench_ocean_mpas_scaling.py-22-n_ranks > process_count) + voronoi partition-quality metrics.
scripts/bench/bench_ocean_mpas_scaling.py-23-
scripts/bench/bench_ocean_mpas_scaling.py-24-M1 measurement contract (scaling-M3d increment-1): the headline number is a
scripts/bench/bench_ocean_mpas_scaling.py:25:fused ``lax.scan`` block (``metadata.timed_scan_blocks``; per-step
scripts/bench/bench_ocean_mpas_scaling.py-26-dispatch latency probed SEPARATELY), cross-rank MAX-reduced, and it is what
scripts/bench/bench_ocean_mpas_scaling.py-27-the aggregator-facing ``steady_median_ms`` carries (the same deliberate
scripts/bench/bench_ocean_mpas_scaling.py-28-naming as ``bench_ocean_latlon_spmd_scaling``); the host-synced gate-loop
--
scripts/bench/bench_ocean_mpas_scaling.py-67-from metadata import (  # noqa: E402
scripts/bench/bench_ocean_mpas_scaling.py-68-    annotate_incomplete,
scripts/bench/bench_ocean_mpas_scaling.py-69-    scaling_metadata,
scripts/bench/bench_ocean_mpas_scaling.py:70:    timed_scan_blocks,
scripts/bench/bench_ocean_mpas_scaling.py-71-    wet_cell_metrics,
scripts/bench/bench_ocean_mpas_scaling.py-72-)
scripts/bench/bench_ocean_mpas_scaling.py-73-
--
scripts/bench/bench_ocean_mpas_scaling.py-527-        # jit the COMPOSED step+exchange (one dispatch per step; the
scripts/bench/bench_ocean_mpas_scaling.py-528-        # sendrecv wrapper always runs traced, exactly as the atmosphere
scripts/bench/bench_ocean_mpas_scaling.py-529-        # step and the distributed PCG use it).  ``_step_impl`` avoids a
scripts/bench/bench_ocean_mpas_scaling.py:530:        # nested-JIT boundary inside this wrapper; timed_scan_blocks
scripts/bench/bench_ocean_mpas_scaling.py-531-        # re-traces the whole thing into the fused scan — same graph.
scripts/bench/bench_ocean_mpas_scaling.py-532-        @jax.jit
scripts/bench/bench_ocean_mpas_scaling.py-533-        def advance(st):
--
scripts/bench/bench_ocean_mpas_scaling.py-649-    # that block (the same slowest-rank convention as the per-step MAX).
scripts/bench/bench_ocean_mpas_scaling.py-650-    fused = None
scripts/bench/bench_ocean_mpas_scaling.py-651-    if args.block_steps > 0:
scripts/bench/bench_ocean_mpas_scaling.py:652:        state, _t = timed_scan_blocks(
scripts/bench/bench_ocean_mpas_scaling.py-653-            advance, state,
scripts/bench/bench_ocean_mpas_scaling.py-654-            block_steps=args.block_steps, n_blocks=args.blocks,
scripts/bench/bench_ocean_mpas_scaling.py-655-            probe_steps=args.probe_steps,
--
scripts/bench/metadata.py-887-    }
scripts/bench/metadata.py-888-
scripts/bench/metadata.py-889-
scripts/bench/metadata.py:890:def timed_scan_blocks(
scripts/bench/metadata.py-891-    advance,
scripts/bench/metadata.py-892-    state,
scripts/bench/metadata.py-893-    *,
scripts/bench/metadata.py-894-    block_steps: int,
scripts/bench/metadata.py-895-    n_blocks: int = 2,
scripts/bench/metadata.py-896-    probe_steps: int = 3,
scripts/bench/metadata.py:897:    sync_label: str = "timed_scan_blocks",
scripts/bench/metadata.py-898-    aux=None,
scripts/bench/metadata.py-899-):
scripts/bench/metadata.py-900-    """Measurement-contract timing: fused ``lax.scan`` blocks + probe latency.
--
scripts/bench/metadata.py-981-            np.array([block_steps, n_blocks, probe_steps], dtype=np.int64)))
scripts/bench/metadata.py-982-        if not bool((_sched == _sched[0]).all()):
scripts/bench/metadata.py-983-            raise ValueError(
scripts/bench/metadata.py:984:                "timed_scan_blocks: processes disagree on the schedule "
scripts/bench/metadata.py-985-                f"(block_steps, n_blocks, probe_steps) = {_sched.tolist()} "
scripts/bench/metadata.py-986-                "per process — a mismatched schedule deadlocks in the "
scripts/bench/metadata.py-987-                "named fences.")
--
scripts/bench/metadata.py-990-    # zero-length parity path).
scripts/bench/metadata.py-991-    if block_steps < 0:
scripts/bench/metadata.py-992-        raise ValueError(
scripts/bench/metadata.py:993:            f"timed_scan_blocks: block_steps must be >= 0, got {block_steps}")
scripts/bench/metadata.py-994-    if n_blocks < 1:
scripts/bench/metadata.py-995-        raise ValueError(
scripts/bench/metadata.py:996:            f"timed_scan_blocks: n_blocks must be >= 1, got {n_blocks}")
scripts/bench/metadata.py-997-    if probe_steps < 0:
scripts/bench/metadata.py-998-        raise ValueError(
scripts/bench/metadata.py:999:            f"timed_scan_blocks: probe_steps must be >= 0, got {probe_steps}")
scripts/bench/metadata.py-1000-
scripts/bench/metadata.py-1001-    def _block(tree):
scripts/bench/metadata.py-1002-        jax.block_until_ready(jax.tree_util.tree_leaves(tree))
--
tests/unit/test_run_omip_latlon_spmd.py-53-
tests/unit/test_run_omip_latlon_spmd.py-54-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/unit/test_run_omip_latlon_spmd.py-55-        gather_state_latlon,
tests/unit/test_run_omip_latlon_spmd.py:56:        make_sharded_ocean_step,
tests/unit/test_run_omip_latlon_spmd.py-57-        shard_state_latlon,
tests/unit/test_run_omip_latlon_spmd.py-58-    )
tests/unit/test_run_omip_latlon_spmd.py-59-    from legoesm.parallel.mesh import create_latlon_mesh
--
tests/unit/test_run_omip_latlon_spmd.py-79-
tests/unit/test_run_omip_latlon_spmd.py-80-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py-81-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py:82:    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-83-    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-84-    ckpt_dir = tmp_path / "spmd"
tests/unit/test_run_omip_latlon_spmd.py-85-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
--
tests/unit/test_run_omip_latlon_spmd.py-177-
tests/unit/test_run_omip_latlon_spmd.py-178-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/unit/test_run_omip_latlon_spmd.py-179-        gather_state_latlon,
tests/unit/test_run_omip_latlon_spmd.py:180:        make_sharded_ocean_step,
tests/unit/test_run_omip_latlon_spmd.py-181-        shard_forcing_stack_latlon,
tests/unit/test_run_omip_latlon_spmd.py-182-        shard_state_latlon,
tests/unit/test_run_omip_latlon_spmd.py-183-    )
--
tests/unit/test_run_omip_latlon_spmd.py-216-    assert ok_serial, "serial JRA55 loop reported not-ok"
tests/unit/test_run_omip_latlon_spmd.py-217-
tests/unit/test_run_omip_latlon_spmd.py-218-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py:219:    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-220-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py-221-    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-222-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
--
tests/bench/test_timed_scan_blocks.py:1:"""timed_scan_blocks + git_sha: the measurement-contract helper (M1 gap #1/#2).
tests/bench/test_timed_scan_blocks.py-2-
tests/bench/test_timed_scan_blocks.py-3-Runs on CPU with a trivial jitted step — validates the schedule accounting
tests/bench/test_timed_scan_blocks.py-4-(compile/probe/blocks executed counts, invalid schedules raise instead of
--
tests/bench/test_timed_scan_blocks.py-14-import pytest
tests/bench/test_timed_scan_blocks.py-15-
tests/bench/test_timed_scan_blocks.py-16-sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))
tests/bench/test_timed_scan_blocks.py:17:from metadata import git_sha, scaling_metadata, timed_scan_blocks  # noqa: E402
tests/bench/test_timed_scan_blocks.py-18-
tests/bench/test_timed_scan_blocks.py-19-
tests/bench/test_timed_scan_blocks.py-20-def _advance(s):
--
tests/bench/test_timed_scan_blocks.py-31-        out["n"] = s["n"] + 1.0
tests/bench/test_timed_scan_blocks.py-32-        out["a"] = s["a"] * 0.999
tests/bench/test_timed_scan_blocks.py-33-        return out
tests/bench/test_timed_scan_blocks.py:34:    s, m = timed_scan_blocks(adv, s0, block_steps=5, n_blocks=2, probe_steps=3)
tests/bench/test_timed_scan_blocks.py-35-    # 1 compile + 3 probe + 2*5 block = 14 steps
tests/bench/test_timed_scan_blocks.py-36-    assert float(s["n"]) == 14.0
tests/bench/test_timed_scan_blocks.py-37-    assert m["block_steps"] == 5 and m["n_blocks"] == 2 and m["probe_steps"] == 3
--
tests/bench/test_timed_scan_blocks.py-40-
tests/bench/test_timed_scan_blocks.py-41-def test_metrics_shape_and_separation():
tests/bench/test_timed_scan_blocks.py-42-    s0 = {"x": jnp.ones((8,))}
tests/bench/test_timed_scan_blocks.py:43:    s, m = timed_scan_blocks(_advance, s0, block_steps=4, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-44-                             probe_steps=2)
tests/bench/test_timed_scan_blocks.py-45-    for k in ("compile_ms", "scan_compile_ms", "step_latency_ms",
tests/bench/test_timed_scan_blocks.py-46-              "fused_step_ms", "parallel_block_ms", "rank_imbalance",
--
tests/bench/test_timed_scan_blocks.py-62-    run (fences/collectives matched) but a zero-step block has NO per-step
tests/bench/test_timed_scan_blocks.py-63-    time — the headline must be an honest null, not block_ms/1."""
tests/bench/test_timed_scan_blocks.py-64-    s0 = {"x": jnp.ones(())}
tests/bench/test_timed_scan_blocks.py:65:    _, m = timed_scan_blocks(_advance, s0, block_steps=0, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-66-                             probe_steps=0)
tests/bench/test_timed_scan_blocks.py-67-    assert m["fused_step_ms"] is None
tests/bench/test_timed_scan_blocks.py-68-    assert m["block_steps"] == 0 and m["n_blocks"] == 1
--
tests/bench/test_timed_scan_blocks.py-75-    must raise so the record always equals the execution."""
tests/bench/test_timed_scan_blocks.py-76-    s0 = {"x": jnp.ones(())}
tests/bench/test_timed_scan_blocks.py-77-    with pytest.raises(ValueError, match="n_blocks"):
tests/bench/test_timed_scan_blocks.py:78:        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=0,
tests/bench/test_timed_scan_blocks.py-79-                          probe_steps=1)
tests/bench/test_timed_scan_blocks.py-80-    with pytest.raises(ValueError, match="n_blocks"):
tests/bench/test_timed_scan_blocks.py:81:        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=-2,
tests/bench/test_timed_scan_blocks.py-82-                          probe_steps=1)
tests/bench/test_timed_scan_blocks.py-83-    with pytest.raises(ValueError, match="probe_steps"):
tests/bench/test_timed_scan_blocks.py:84:        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-85-                          probe_steps=-1)
tests/bench/test_timed_scan_blocks.py-86-    with pytest.raises(ValueError, match="block_steps"):
tests/bench/test_timed_scan_blocks.py:87:        timed_scan_blocks(_advance, s0, block_steps=-1, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-88-                          probe_steps=0)
tests/bench/test_timed_scan_blocks.py-89-
tests/bench/test_timed_scan_blocks.py-90-
--
tests/bench/test_timed_scan_blocks.py-95-    import numpy as np
tests/bench/test_timed_scan_blocks.py-96-    s0 = {"x": jnp.arange(6.0)}
tests/bench/test_timed_scan_blocks.py-97-    ref = np.asarray(s0["x"]).copy()
tests/bench/test_timed_scan_blocks.py:98:    s, m = timed_scan_blocks(_advance, dict(s0), block_steps=1, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-99-                             probe_steps=0)
tests/bench/test_timed_scan_blocks.py-100-    # 1 compile + 0 probe + 1x1 block = 2 steps from the ORIGINAL seed.
tests/bench/test_timed_scan_blocks.py-101-    want = ref
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
scripts/bench/bench_ocean_latlon_spmd_scaling.py-347-        shard_state_latlon,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-348-    )
scripts/bench/bench_ocean_latlon_spmd_scaling.py-349-
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
scripts/bench/bench_ocean_latlon_spmd_scaling.py-456-        s = shard_state_latlon(s0, mesh)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-457-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-458-    # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-462-    # by an individually-synced probe (``step_latency_ms``); multi-controller
scripts/bench/bench_ocean_latlon_spmd_scaling.py-463-    # runs record the SLOWEST-process block time + imbalance ratio (a
scripts/bench/bench_ocean_latlon_spmd_scaling.py-464-    # straggler band is invisible to a single process's clock).
scripts/bench/bench_ocean_latlon_spmd_scaling.py:465:    from metadata import timed_scan_blocks
scripts/bench/bench_ocean_latlon_spmd_scaling.py-466-    # Under --parity-gate the serial reference above ran EXACTLY args.steps
scripts/bench/bench_ocean_latlon_spmd_scaling.py-467-    # steps, so the SPMD arm must execute the same count: 1 compile step +
scripts/bench/bench_ocean_latlon_spmd_scaling.py-468-    # one (args.steps - 1)-long block, no probe.  Timing from a parity smoke
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-473-        _blk, _nblk, _probe = args.steps, args.blocks, args.probe_steps
scripts/bench/bench_ocean_latlon_spmd_scaling.py-474-    # aux threads the sharded geometry stacks through the jit boundary as
scripts/bench/bench_ocean_latlon_spmd_scaling.py-475-    # an ARGUMENT (outer-trace constants of non-addressable arrays are
scripts/bench/bench_ocean_latlon_spmd_scaling.py:476:    # unfetchable — see make_sharded_ocean_step's aux note).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-477-    _aux = getattr(step, "aux", None)
scripts/bench/bench_ocean_latlon_spmd_scaling.py:478:    s, timing = timed_scan_blocks(
scripts/bench/bench_ocean_latlon_spmd_scaling.py-479-        (lambda st, aux: step(st, args.dt, aux=aux)) if _aux is not None
scripts/bench/bench_ocean_latlon_spmd_scaling.py-480-        else (lambda st: step(st, args.dt)),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-481-        s,
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-323-    vmask = model._vertex_mask
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-324-    if vmask is None:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-325-        raise RuntimeError(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:326:            "make_sharded_ocean_step: the model vertex-mask cache is not "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-327-            "primed.  Call model._ensure_vertex_mask(state) (or model.step) "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-328-            "on the concrete initial state before building the sharded step.")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-329-    vmask = np.asarray(vmask)                   # (n_lat+1, n_lon+1)
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-406-    by the band halo.  ``None`` fields pass through.  The inverse is
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-407-    :func:`gather_state_latlon`.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-408-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:409:    This is the layout the in-``shard_map`` body of :func:`make_sharded_ocean_step`
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-410-    expects; the test uses it instead of a uniform ``tree.map(P("lat"))`` (which
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-411-    fails on ``v`` because ``n_lat+1`` is not divisible by ``N``).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-412-    """
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:413:    # v-carrier contract (see make_sharded_ocean_step's fold note): the TOP
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-414-    # v-face row (regular pole wall OR tripole seam/cap row) must be
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-415-    # wall-masked — the carrier drops it and reconstructs it as zero, which
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-416-    # would silently delete a LIVE seam row.  Host-side check on the
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-447-        # top row (e.g. a raw IC perturbation) is dropped -> reconstructed as 0:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-448-        # CORRECT for the dynamics (the pole wall zeros it at step 1) but not a
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-449-        # bit round-trip of that one row.  NOT for a tripole north fold (raises
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:450:        # in make_sharded_ocean_step) where the top row is a live fold partner.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-451-        nlat1 = field.data.shape[0]
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-452-        v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-453-        sh = NamedSharding(mesh, _lat_spec(v_lower))
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-609-    return state._replace(**updates)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-610-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-611-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:612:def make_sharded_ocean_step(model, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-613-    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-614-    sponge=None, t_seconds=None) -> state`` running ``model.step``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-615-    lat-band-SPMD.
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-730-            g_vals = multihost_utils.process_allgather(vals)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-731-            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-732-                raise RuntimeError(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:733:                    f"make_sharded_ocean_step: band-geometry field {name!r} "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-734-                    f"DIVERGES across processes (exact_dtype={is_exact}, "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-735-                    f"gathered={g_vals.tolist()}) — a real config/grid "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-736-                    f"inconsistency, not autotune noise; refusing to "
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-763-            _schema_fingerprint(list(array_field_names), n_dev))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-764-        if not bool(np.all(_g == _g[0])):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-765-            raise RuntimeError(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:766:                "make_sharded_ocean_step: the band-geometry SCHEMA differs "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-767-                "across processes (field list / x64 setting / device count "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-768-                f"— gathered {_g.tolist()}). Fix the per-process config "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-769-                "before sharding; the per-field checks below assume one "
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-902-        # arrays (RuntimeError: 'Fetching value ... non-addressable'; the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-903-        # multicontroller lane has been broken this way since the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-904-        # #1370-iii stack sharding). Callers that wrap the step in their
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:905:        # own jit MUST thread ``step.aux`` through their jit boundary as
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-906-        # an ARGUMENT and pass it back here.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-907-        # ONE forcing operand: None fields drop out of the pytree structure,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-908-        # so specs derived by tree.map skip them automatically and the
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-982-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-983-    # Expose the stacks so outer-jit callers can pass them as arguments
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-984-    # (see the ``aux`` note in the signature).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:985:    sharded_step.aux = (geom_stacks, vmask_stack)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-986-    return sharded_step
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-987-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-988-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:989:def make_sharded_ocean_step_global(model, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-990-    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-991-    that takes a GLOBAL (single-device-layout) state + forcing and returns a
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-992-    GLOBAL state — the minimal-diff driver entry point.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-993-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:994:    Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-995-    (:func:`shard_state_latlon` + :func:`shard_forcing_latlon`), runs the lat-band
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-996-    SPMD step, then gathers the state OUT (:func:`gather_state_latlon`).  This lets
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-997-    the OMIP host loop keep operating on a normal full-domain state — the per-step
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-998-    host BCs (SSS restore, prognostic ice, geothermal, BBL, nudge) see the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-999-    gathered global state UNCHANGED — at the cost of a per-step gather/scatter
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1000-    (acceptable for the host-coupled OMIP driver; the pure-dynamics inner loop
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:1001:    should use :func:`make_sharded_ocean_step` directly to stay sharded).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1002-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1003-    ``mesh is None`` ⇒ the plain single-device ``model.step`` (no scatter/gather).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1004-    """
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1007-            model.step(state, dt, freshwater=freshwater,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1008-                       surface_forcing=surface_forcing))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1009-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:1010:    inner = make_sharded_ocean_step(model, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1011-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:1012:    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1013-        # Scatter the global state AND forcing to the band layout explicitly
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1014-        # (codex r17 item 3: the old comment claimed inner sharded the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1015-        # forcing; it forwarded it global and relied on implicit JIT input
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1021-                   freshwater=shard_forcing_latlon(freshwater, mesh))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1022-        return gather_state_latlon(ss, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1023-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:1024:    return sharded_step_global
--
tests/ocean/unit/test_sharded_geom_fingerprint.py-1-"""Unit tests for the band-geometry cross-process fingerprint gate.
tests/ocean/unit/test_sharded_geom_fingerprint.py-2-
tests/ocean/unit/test_sharded_geom_fingerprint.py:3:The gate decides whether ``make_sharded_ocean_step`` accepts per-process
tests/ocean/unit/test_sharded_geom_fingerprint.py-4-band-geometry stacks without the (removed, nd-linear-cost) process-0
tests/ocean/unit/test_sharded_geom_fingerprint.py-5-broadcast — see the 2026-08-03 fix note at the sharded put. These tests
tests/ocean/unit/test_sharded_geom_fingerprint.py-6-pin the gate's discrimination properties single-process (the
--
tests/parallel/test_latlon_ocean_spmd_step.py-1-"""SPMD equivalence gate for the lat-lon C-grid ocean step (multi-node OMIP).
tests/parallel/test_latlon_ocean_spmd_step.py-2-
tests/parallel/test_latlon_ocean_spmd_step.py:3:This is the CORRECTNESS GATE for ``make_sharded_ocean_step`` (lat-band SPMD over
tests/parallel/test_latlon_ocean_spmd_step.py-4-the ``"lat"`` axis): N steps under ``shard_map`` on 4 (CPU) devices must match
tests/parallel/test_latlon_ocean_spmd_step.py-5-the single-device step to tolerance. It is the eORCA025 ¼° enabler (the ¼° grid
tests/parallel/test_latlon_ocean_spmd_step.py-6-OOMs on one 32 GiB GPU; lat-band sharding fits it at N>=5).
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
tests/parallel/test_latlon_ocean_spmd_step.py-101-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-102-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-103-    )
--
tests/parallel/test_latlon_ocean_spmd_step.py-127-    # fail on the n_lat+1 v rows).  The result is gathered (and the dropped pole
tests/parallel/test_latlon_ocean_spmd_step.py-128-    # row reappended) for the bit-comparison vs the single-device reference.
tests/parallel/test_latlon_ocean_spmd_step.py-129-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:130:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-131-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-132-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-133-        ss = step(ss, dt)
--
tests/parallel/test_latlon_ocean_spmd_step.py-221-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-222-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py-223-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py:224:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py-225-        shard_forcing_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-226-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-227-    )
--
tests/parallel/test_latlon_ocean_spmd_step.py-244-
tests/parallel/test_latlon_ocean_spmd_step.py-245-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-246-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:247:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-248-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-249-    fws = shard_forcing_latlon(fw, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-250-    sfs = shard_forcing_latlon(sf, dev.mesh)
--
tests/parallel/test_latlon_ocean_spmd_step.py-384-    with an opaque shard_map divisibility error."""
tests/parallel/test_latlon_ocean_spmd_step.py-385-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-386-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py:387:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py-388-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-389-    )
tests/parallel/test_latlon_ocean_spmd_step.py-390-    from legoesm.ocean.state import OceanSurfaceForcing
--
tests/parallel/test_latlon_ocean_spmd_step.py-396-    state0 = _perturbed_state(grid, z_coord)
tests/parallel/test_latlon_ocean_spmd_step.py-397-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-398-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:399:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-400-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-401-    bad = OceanSurfaceForcing(
tests/parallel/test_latlon_ocean_spmd_step.py-402-        tau_y=jnp.zeros((grid.n_lat + 1, grid.n_lon)))
--
tests/parallel/test_latlon_ocean_spmd_step.py-409-@pytest.mark.skipif(not _have_sharded_step(),
tests/parallel/test_latlon_ocean_spmd_step.py-410-                    reason="sharded_ocean_step module not present")
tests/parallel/test_latlon_ocean_spmd_step.py-411-def test_sharded_ocean_step_global_matches_explicit_scatter_gather():
tests/parallel/test_latlon_ocean_spmd_step.py:412:    """``make_sharded_ocean_step_global`` (global-in/global-out, the minimal
tests/parallel/test_latlon_ocean_spmd_step.py-413-    driver entry) must equal the explicit ``shard_state_latlon`` -> inner step ->
tests/parallel/test_latlon_ocean_spmd_step.py-414-    ``gather_state_latlon`` path BIT-FOR-BIT (it is literally that composition),
tests/parallel/test_latlon_ocean_spmd_step.py-415-    AND match the single-device reference to the same re-association floor.
--
tests/parallel/test_latlon_ocean_spmd_step.py-419-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-420-    from legoesm.grids.latlon import ensure_geometry
tests/parallel/test_latlon_ocean_spmd_step.py-421-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py:422:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py:423:        make_sharded_ocean_step_global,
tests/parallel/test_latlon_ocean_spmd_step.py-424-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-425-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-426-    )
--
tests/parallel/test_latlon_ocean_spmd_step.py-451-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py-452-
tests/parallel/test_latlon_ocean_spmd_step.py-453-    # explicit scatter -> inner sharded step -> gather
tests/parallel/test_latlon_ocean_spmd_step.py:454:    inner = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-455-    ss = shard_state_latlon(state0, dev.mesh)
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
tests/parallel/test_latlon_spmd_fused_halo.py-321-    )
tests/parallel/test_latlon_spmd_fused_halo.py-322-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_spmd_fused_halo.py-323-        gather_state_latlon,
tests/parallel/test_latlon_spmd_fused_halo.py:324:        make_sharded_ocean_step,
tests/parallel/test_latlon_spmd_fused_halo.py-325-        shard_state_latlon,
tests/parallel/test_latlon_spmd_fused_halo.py-326-    )
tests/parallel/test_latlon_spmd_fused_halo.py-327-    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
--
tests/parallel/test_latlon_spmd_fused_halo.py-346-    hlo_counts = {}
tests/parallel/test_latlon_spmd_fused_halo.py-347-    for flag in ("0", "1"):
tests/parallel/test_latlon_spmd_fused_halo.py-348-        _env_flag(monkeypatch, flag)
tests/parallel/test_latlon_spmd_fused_halo.py:349:        step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-350-        ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-351-        hlo_counts[flag] = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-352-            jax.jit(step).lower(ss, 600.0).compile().as_text())
--
tests/parallel/test_latlon_spmd_fused_halo.py-369-    # (an outer-jit artifact, not the production call pattern — the
tests/parallel/test_latlon_spmd_fused_halo.py-370-    # driver calls step() directly each step).
tests/parallel/test_latlon_spmd_fused_halo.py-371-    _env_flag(monkeypatch, "0")
tests/parallel/test_latlon_spmd_fused_halo.py:372:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-373-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-374-    n_off = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-375-        jax.jit(lambda s, d: step(s, d)).lower(ss, 600.0)
--
tests/parallel/test_persistent_sharded_ocean_loop.py-2-(scaling-M2 increment 1, ``run_omip_core2 --spmd-persistent-state``).
tests/parallel/test_persistent_sharded_ocean_loop.py-3-
tests/parallel/test_persistent_sharded_ocean_loop.py-4-The production multi-GPU OMIP host loop used to call
tests/parallel/test_persistent_sharded_ocean_loop.py:5:``make_sharded_ocean_step_global`` EVERY step — a full-state scatter
tests/parallel/test_persistent_sharded_ocean_loop.py-6-(``shard_state_latlon``) + gather (``gather_state_latlon``) per step, i.e.
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
tests/parallel/test_persistent_sharded_ocean_loop.py-202-    ss = sos.shard_state_latlon(state0, mesh)          # ONE initial shard
tests/parallel/test_persistent_sharded_ocean_loop.py-203-    for k in range(1, n_steps + 1):
tests/parallel/test_persistent_sharded_ocean_loop.py-204-        # surface-current consumer on the SHARDED state: v is the n_lat-row
--
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-13-# Runs the SINGLE-PROCESS multi-device SPMD lane of
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-14-# scripts/bench/bench_ocean_latlon_spmd_scaling.py: the FULL
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-15-# lat-lon C-grid ocean step sharded over 1/2/4 A100s via
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs:16:# make_sharded_ocean_step (shard_map ppermute band halos + psum reductions
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-17-# over NVLink/NCCL; NO mpi4jax — uses the plain `legoesm-gpu` env from
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-18-# README Step 1, no route-A overlay needed).
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs-19-#
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
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-73-    shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-74-)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-75-from legoesm.ocean.state import LatLonCGridOceanConfig  # noqa: E402
--
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-122-    # serial run above already did; belt-and-braces for wrapper band masks).
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-123-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-124-
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:125:    step = make_sharded_ocean_step(model, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-126-    ss = shard_state_latlon(state0, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-127-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-128-        ss = step(ss, dt)
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
tests/parallel/test_latlon_ocean_spmd_tripole.py-116-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_tripole.py-117-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_tripole.py-118-    )
--
tests/parallel/test_latlon_ocean_spmd_tripole.py-149-    model._ensure_vertex_mask(state0)        # prime the build-once vmask cache
tests/parallel/test_latlon_ocean_spmd_tripole.py-150-
tests/parallel/test_latlon_ocean_spmd_tripole.py-151-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_tripole.py:152:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-153-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-154-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_tripole.py-155-        ss = step(ss, dt, surface_forcing=sf)
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
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-77-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-78-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-79-    )
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-89-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-90-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-91-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:92:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-93-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-94-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-95-        ss = step(ss, dt)
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-112-def _sharded_step_hlo(cfg, n_lat=48, n_lon=96, nlev=10):
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-113-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-114-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:115:        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-116-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-117-    )
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-118-
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-122-    state0 = _perturbed_state(grid, z_coord)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-123-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-124-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:125:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-126-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-127-    lowered = jax.jit(step).lower(ss, 600.0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-128-    return lowered.compile().as_text()
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
scripts/run/run_omip_core2.py-5565-            # consumers).  t_sec is always None here (tide fail-fasts above).
scripts/run/run_omip_core2.py-5566-            from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip_core2.py-5567-                gather_state_latlon,
scripts/run/run_omip_core2.py:5568:                make_sharded_ocean_step,
scripts/run/run_omip_core2.py-5569-                shard_state_latlon,
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
scripts/run/run_omip.py-4815-                shard_forcing_stack_latlon,
scripts/run/run_omip.py-4816-                shard_state_latlon,
scripts/run/run_omip.py-4817-            )
--
scripts/run/run_omip.py-4820-            # state so the wrapper can build per-band masks host-side.
scripts/run/run_omip.py-4821-            model.prime_step_caches(state)
scripts/run/run_omip.py-4822-            _dev = create_latlon_mesh(n_devices=_nd)
scripts/run/run_omip.py:4823:            spmd_step = make_sharded_ocean_step(model, _dev.mesh)
scripts/run/run_omip.py-4824-            spmd_gather = partial(gather_state_latlon, mesh=_dev.mesh)
scripts/run/run_omip.py-4825-            state = shard_state_latlon(state, _dev.mesh)
scripts/run/run_omip.py-4826-            # Lay per-block forcing stacks out lat-band-sharded so the
--
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-2-
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-3-Exercises the builder + the nd=1 single-device main() end-to-end on a tiny
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-4-grid (JSONL record shape) and the argument guards. The multi-device timing
tests/unit/test_bench_ocean_latlon_spmd_scaling.py:5:path reuses make_sharded_ocean_step, whose correctness gate is
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-6-tests/parallel/test_latlon_ocean_spmd_step.py; the --multicontroller path is
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-7-cluster-gated (see the script docstring).
tests/unit/test_bench_ocean_latlon_spmd_scaling.py-8-"""
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-543-    REUSES the shared primitives (band perms, the v-face round-trip, the band
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-544-    halo via the swapped backend, ``spmd_pole_end_masks``, the band slicer);
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-545-    atm-NEW is only the 6-field state/geometry walk. ``check_vma=False`` (the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:546:    band halo reads neighbour-rank data). Mirrors ``make_sharded_ocean_step``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-547-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-548-    ``physics_fn`` (optional): a STATELESS, COLUMN-LOCAL physics closure
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-549-    (``physics_fn(hs, grid, sigma_coord) -> HydrostaticTendencies``, e.g.

exec
/usr/bin/bash -lc "sed -n '760,1025p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
sed -n '880,1100p' scripts/bench/metadata.py
sed -n '430,510p' scripts/bench/bench_ocean_latlon_spmd_scaling.py
rg -n -C 4 \"def timed_scan_blocks\\(|timed_scan_blocks\\(\" scripts tests packages --glob '"'!**/.git/**'"'
rg -n -C 7 \"make_sharded_ocean_step\\(|spmd_step\\(|_spmd_inner\\(|sharded_step_global\\(\" scripts packages tests --glob '"'!**/.git/**'"'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 854ms:
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
            "latency_us": lat_us,
            "bandwidth_GBs": bw_gbs,
            "halo_messages_per_step": halo_messages_per_step,
            "halo_bytes_per_step": halo_bytes_per_step,
            "n_reductions_per_step": n_reductions_per_step,
            "rank_imbalance": rank_imbalance,
        },
    }


def timed_scan_blocks(
    advance,
    state,
    *,
    block_steps: int,
    n_blocks: int = 2,
    probe_steps: int = 3,
    sync_label: str = "timed_scan_blocks",
    aux=None,
):
    """Measurement-contract timing: fused ``lax.scan`` blocks + probe latency.

    The trustworthy production-like number is a MULTI-STEP ``lax.scan`` block
    with device synchronization only AROUND the block (per-step host sync in a
    Python loop measures dispatch+sync latency, not fused device throughput —
    the audited anti-pattern in the SPMD benches).  Per-step dispatch latency
    is still physically meaningful (drivers that must step one-at-a-time pay
    it), so it is measured SEPARATELY by a short individually-synced probe and
    reported as ``step_latency_ms`` — never mixed into the fused number.

    Multi-controller runs additionally allgather EVERY process's full
    per-block vector and reduce per block: the parallel time of block ``b``
    is the SLOWEST process in that block.  A per-process median gathered
    alone would hide an alternating straggler (every rank's median can be
    fast even though every block has a slow rank — codex batch4).

    Parameters
    ----------
    advance
        ``advance(state) -> state`` — ONE production step with all static
        knobs (dt, forcing, ...) closed over.  May itself be jitted; it is
        re-traced INTO the fused scan (same graph, no double-jit penalty).
        MUST NOT donate its input buffers (``donate_argnums``): the scan
        pre-compile below runs on a SHALLOW pytree copy whose leaves ALIAS
        the live seed state — donation would invalidate the seed's buffers
        mid-benchmark.
    state
        Initial (already sharded, post-seed) model state pytree.
    block_steps
        Steps per fused ``lax.scan`` block (the amortizing window).
        ``>= 0``; ``0`` is the documented ZERO-LENGTH parity path (the
        block advances nothing and ``fused_step_ms`` is ``None`` — a
        zero-step block has no per-step time).
    n_blocks
        Timed blocks (``>= 1``); per-block times expose drift.
    probe_steps
        Individually host-synced steps (``>= 0``) for the separate
        dispatch-latency probe (each one costs a device round-trip).
    sync_label
        Base label for the multi-controller ``sync_global_devices`` fences.

    Invalid schedule values raise ``ValueError`` — never silently clamped,
    so the recorded schedule is ALWAYS the executed schedule.  Multi-process
    runs first allgather-verify the schedule tuple itself: processes that
    disagree on ``(block_steps, n_blocks, probe_steps)`` would enter
    DIFFERENT named-fence schedules and deadlock; the verification is the
    one collective every process reaches, so a mismatch raises everywhere.

    Returns
    -------
    (state, metrics) — final state (compile + probe + all blocks advanced)
    and a dict:
      ``compile_ms``           first-call cost of ``advance`` (trace+compile)
      ``scan_compile_ms``      first-call cost of the fused scan itself
      ``step_latency_ms``      median individually-synced per-step wall time
      ``block_ms``             per-block wall times, THIS process (list)
      ``parallel_block_ms``    per-block wall times of the PARALLEL step:
                               max over processes, per block (== ``block_ms``
                               for a single process)
      ``fused_step_ms``        median(parallel_block_ms)/block_steps — the
                               headline (``None`` when ``block_steps == 0``)
      ``rank_imbalance``       median over blocks of the per-block
                               max/median-across-processes ratio (>= 1.0;
                               exactly 1.0 for a single process)
      ``rank_imbalance_per_block``  the per-block ratios themselves
      ``block_steps``/``n_blocks``/``probe_steps``  the EXECUTED schedule
    """
    import time

    import jax
    import numpy as np

    multi = jax.process_count() > 1
    if multi:
        # Collectively verify the schedule BEFORE compilation or any
        # schedule-dependent fence: this allgather is the single collective
        # every process reaches first, so on a mismatch EVERY process sees
        # the same gathered table and raises together instead of hanging in
        # mismatched named fences (codex batch4 deadlock hazard).
        from jax.experimental import multihost_utils
        _sched = np.asarray(multihost_utils.process_allgather(
            np.array([block_steps, n_blocks, probe_steps], dtype=np.int64)))
        if not bool((_sched == _sched[0]).all()):
            raise ValueError(
                "timed_scan_blocks: processes disagree on the schedule "
                f"(block_steps, n_blocks, probe_steps) = {_sched.tolist()} "
                "per process — a mismatched schedule deadlocks in the "
                "named fences.")
    # No silent rewrites: invalid values raise; the recorded schedule IS
    # the executed schedule.  block_steps == 0 stays legal (the documented
    # zero-length parity path).
    if block_steps < 0:
        raise ValueError(
            f"timed_scan_blocks: block_steps must be >= 0, got {block_steps}")
    if n_blocks < 1:
        raise ValueError(
            f"timed_scan_blocks: n_blocks must be >= 1, got {n_blocks}")
    if probe_steps < 0:
        raise ValueError(
            f"timed_scan_blocks: probe_steps must be >= 0, got {probe_steps}")

    def _block(tree):
        jax.block_until_ready(jax.tree_util.tree_leaves(tree))

    def _fence(tag: str):
        if multi:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices(f"{sync_label}_{tag}")

    # Normalize to a 2-arg advance (see the aux note below): top-level
    # calls here are outside any trace, so passing aux is just an argument.
    if aux is None:
        def _advance(carry, _aux):
            return advance(carry)
    else:
        _advance = advance

    # --- 1. compile (first call of advance, separated from all timing) ---
    _fence("compile_start")
    t0 = time.perf_counter()
    state = _advance(state, aux)
    _block(state)
    compile_ms = (time.perf_counter() - t0) * 1e3

    # --- 2. dispatch-latency probe: individually synced steps, reported
    # separately (NEVER mixed into the fused number) ---
    probe_ms = []
    for _ in range(probe_steps):
        t0 = time.perf_counter()
        state = _advance(state, aux)
        _block(state)
        probe_ms.append((time.perf_counter() - t0) * 1e3)
    step_latency_ms = float(np.median(probe_ms)) if probe_ms else float("nan")

    # --- 3. fused scan block (dtype-stable carry, the OM pattern) ---
    input_dtypes = jax.tree_util.tree_map(
        lambda x: x.dtype if hasattr(x, "dtype") else None, state)

    # ``aux``: extra pytree threaded through the jit boundary as an ARGUMENT
    # (never a closure capture). Needed when ``advance`` internally calls a
    # jit function over SHARDED-global auxiliary arrays (the ocean band
    # geometry stacks): jit-of-jit inlines the inner call, so concrete
    # closure arrays would become outer-trace CONSTANTS whose value the MLIR
    # handler cannot fetch for non-addressable arrays (2026-08-03 repro).
    @jax.jit
    def _scan_run(st, _aux):
        def _body(carry, _):
            new = _advance(carry, _aux)
            new = jax.tree_util.tree_map(
                lambda x, d: x.astype(d)
                if d is not None and hasattr(x, "astype") else x,
                new, input_dtypes)
            return new, None
        return jax.lax.scan(_body, st, None, length=block_steps)[0]

    # Pre-compile the scan on a SHALLOW pytree copy: the leaves ALIAS the
    # seed state's arrays (no data copy) — sufficient AND safe because the
    # pre-call only needs matching shapes/dtypes/shardings to warm the
    # compile cache, jitted execution is pure, and ``advance`` is
    # contract-bound not to donate buffers (see the docstring).  The seed
    # value itself is untouched; every process executes the same collective
    # schedule — counts stay matched.
    _pre = jax.tree_util.tree_map(lambda x: x, state)
    t0 = time.perf_counter()
    _pre_out = _scan_run(_pre, aux)
    _block(_pre_out)
    scan_compile_ms = (time.perf_counter() - t0) * 1e3
    del _pre, _pre_out

    block_ms = []
    for b in range(n_blocks):
        _fence(f"block{b}_start")
        t0 = time.perf_counter()
        state = _scan_run(state, aux)
        _block(state)
        block_ms.append((time.perf_counter() - t0) * 1e3)
    _fence("blocks_end")

    # Slowest-rank statistics from the FULL per-block vectors (equal length
    # everywhere — the schedule was collectively verified above).  The
    # parallel time of block b is the slowest process IN that block; a
    # gather of per-process medians would let alternating stragglers make
    # every rank median look fast (codex batch4).
    my_blocks = np.asarray(block_ms, dtype=np.float64)
    if multi:
        from jax.experimental import multihost_utils
        all_blocks = np.asarray(
            multihost_utils.process_allgather(my_blocks))
    else:
        all_blocks = my_blocks[None, :]
    parallel_block_ms = np.max(all_blocks, axis=0)        # (n_blocks,)
    rank_median_ms = np.median(all_blocks, axis=0)        # (n_blocks,)
    with np.errstate(divide="ignore", invalid="ignore"):
        imb_per_block = np.where(rank_median_ms > 0.0,
                                 parallel_block_ms / rank_median_ms,
                                 np.nan)
    rank_imbalance = (float(np.nanmedian(imb_per_block))
                      if np.isfinite(imb_per_block).any() else float("nan"))
    # Headline: median over blocks of the per-block PARALLEL time, per step.
    # A zero-length (parity) block has no per-step time — honest null.
    fused_step_ms = (
        wet_columns_per_device=[float(w) for w in _wet_cols_per_dev],
    )

    # Parity reference: the plain single-device trajectory, computed BEFORE
    # any sharding (deterministic identical build on every process).
    serial_final = None
    if args.parity_gate:
        _s = s0
        for _ in range(args.steps):
            _s = model.step(_s, args.dt)
        _block(_s)
        serial_final = _s

    inv_before = None
    if args.check_conservation:
        from bench_ocean_mpi_scaling import ocean_invariants
        inv_before = ocean_invariants(model, s0, n_ranks=1)

    if nd == 1:
        mesh = None
        step = make_sharded_ocean_step(model, None)
        s = s0
    else:
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
                                 axis_names=("lat",))
        step = make_sharded_ocean_step(model, mesh)
        s = shard_state_latlon(s0, mesh)

    # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
    # blocks with sync only AROUND the block — the previous per-step
    # host-synced loop measured dispatch+sync latency, not fused device
    # throughput.  Per-step dispatch latency is still measured, SEPARATELY,
    # by an individually-synced probe (``step_latency_ms``); multi-controller
    # runs record the SLOWEST-process block time + imbalance ratio (a
    # straggler band is invisible to a single process's clock).
    from metadata import timed_scan_blocks
    # Under --parity-gate the serial reference above ran EXACTLY args.steps
    # steps, so the SPMD arm must execute the same count: 1 compile step +
    # one (args.steps - 1)-long block, no probe.  Timing from a parity smoke
    # run is not reported as a scaling number anyway (steps are capped).
    if args.parity_gate:
        _blk, _nblk, _probe = max(0, args.steps - 1), 1, 0
    else:
        _blk, _nblk, _probe = args.steps, args.blocks, args.probe_steps
    # aux threads the sharded geometry stacks through the jit boundary as
    # an ARGUMENT (outer-trace constants of non-addressable arrays are
    # unfetchable — see make_sharded_ocean_step's aux note).
    _aux = getattr(step, "aux", None)
    s, timing = timed_scan_blocks(
        (lambda st, aux: step(st, args.dt, aux=aux)) if _aux is not None
        else (lambda st: step(st, args.dt)),
        s,
        block_steps=_blk, n_blocks=_nblk, probe_steps=_probe,
        sync_label="ocean_latlon_spmd_bench", aux=_aux)

    # Post-run ZERO-FORCING residual probe eligibility (audit item 6):
    # needs the gathered global final state on ONE process; multicontroller
    # runs skip it (zero_forcing_probe_measured=False, never faked).
    _probe_residual = (args.baro_solver == "implicit_cn"
                       and jax.process_count() == 1)

    final_global = None
    if args.parity_gate or args.check_conservation or _probe_residual:
        from legoesm.ocean.dynamics.sharded_ocean_step import (
            gather_state_latlon,
        )
        final_global = (gather_state_latlon(s, mesh) if mesh is not None
                        else s)

    # --- Correctness gates (before any timing is reported) -----------------
    if args.parity_gate or args.check_conservation:
        prec = "float64" if jax.config.jax_enable_x64 else "float32"
        rank0 = jax.process_index() == 0
        if args.check_conservation:
            from bench_ocean_mpi_scaling import (
                CONS_RTOL_DEFAULTS,
                conservation_breaches,
                ocean_invariants,
                print_conservation,
            )
            inv_after = ocean_invariants(model, final_global, n_ranks=1)
scripts/bench/bench_atm_latlon_spmd_scaling.py-298-        # throughput.  Dispatch latency stays measured SEPARATELY
scripts/bench/bench_atm_latlon_spmd_scaling.py-299-        # (``step_latency_ms``); multi-controller runs record the
scripts/bench/bench_atm_latlon_spmd_scaling.py-300-        # slowest-process block time + imbalance ratio.
scripts/bench/bench_atm_latlon_spmd_scaling.py-301-        from metadata import timed_scan_blocks
scripts/bench/bench_atm_latlon_spmd_scaling.py:302:        c, timing = timed_scan_blocks(
scripts/bench/bench_atm_latlon_spmd_scaling.py-303-            lambda st: step(st, args.dt), c,
scripts/bench/bench_atm_latlon_spmd_scaling.py-304-            block_steps=args.steps, n_blocks=args.blocks,
scripts/bench/bench_atm_latlon_spmd_scaling.py-305-            probe_steps=args.probe_steps,
scripts/bench/bench_atm_latlon_spmd_scaling.py-306-            sync_label="atm_latlon_spmd_bench")
--
tests/bench/test_timed_scan_blocks.py-30-        out = dict(s)
tests/bench/test_timed_scan_blocks.py-31-        out["n"] = s["n"] + 1.0
tests/bench/test_timed_scan_blocks.py-32-        out["a"] = s["a"] * 0.999
tests/bench/test_timed_scan_blocks.py-33-        return out
tests/bench/test_timed_scan_blocks.py:34:    s, m = timed_scan_blocks(adv, s0, block_steps=5, n_blocks=2, probe_steps=3)
tests/bench/test_timed_scan_blocks.py-35-    # 1 compile + 3 probe + 2*5 block = 14 steps
tests/bench/test_timed_scan_blocks.py-36-    assert float(s["n"]) == 14.0
tests/bench/test_timed_scan_blocks.py-37-    assert m["block_steps"] == 5 and m["n_blocks"] == 2 and m["probe_steps"] == 3
tests/bench/test_timed_scan_blocks.py-38-    assert len(m["block_ms"]) == 2
tests/bench/test_timed_scan_blocks.py-39-
tests/bench/test_timed_scan_blocks.py-40-
tests/bench/test_timed_scan_blocks.py-41-def test_metrics_shape_and_separation():
tests/bench/test_timed_scan_blocks.py-42-    s0 = {"x": jnp.ones((8,))}
tests/bench/test_timed_scan_blocks.py:43:    s, m = timed_scan_blocks(_advance, s0, block_steps=4, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-44-                             probe_steps=2)
tests/bench/test_timed_scan_blocks.py-45-    for k in ("compile_ms", "scan_compile_ms", "step_latency_ms",
tests/bench/test_timed_scan_blocks.py-46-              "fused_step_ms", "parallel_block_ms", "rank_imbalance",
tests/bench/test_timed_scan_blocks.py-47-              "rank_imbalance_per_block"):
--
tests/bench/test_timed_scan_blocks.py-61-    """block_steps=0 is the documented zero-length parity path: the blocks
tests/bench/test_timed_scan_blocks.py-62-    run (fences/collectives matched) but a zero-step block has NO per-step
tests/bench/test_timed_scan_blocks.py-63-    time — the headline must be an honest null, not block_ms/1."""
tests/bench/test_timed_scan_blocks.py-64-    s0 = {"x": jnp.ones(())}
tests/bench/test_timed_scan_blocks.py:65:    _, m = timed_scan_blocks(_advance, s0, block_steps=0, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-66-                             probe_steps=0)
tests/bench/test_timed_scan_blocks.py-67-    assert m["fused_step_ms"] is None
tests/bench/test_timed_scan_blocks.py-68-    assert m["block_steps"] == 0 and m["n_blocks"] == 1
tests/bench/test_timed_scan_blocks.py-69-    assert len(m["block_ms"]) == 1
--
tests/bench/test_timed_scan_blocks.py-74-    schedule DIFFERENT from the recorded one (codex batch4): invalid values
tests/bench/test_timed_scan_blocks.py-75-    must raise so the record always equals the execution."""
tests/bench/test_timed_scan_blocks.py-76-    s0 = {"x": jnp.ones(())}
tests/bench/test_timed_scan_blocks.py-77-    with pytest.raises(ValueError, match="n_blocks"):
tests/bench/test_timed_scan_blocks.py:78:        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=0,
tests/bench/test_timed_scan_blocks.py-79-                          probe_steps=1)
tests/bench/test_timed_scan_blocks.py-80-    with pytest.raises(ValueError, match="n_blocks"):
tests/bench/test_timed_scan_blocks.py:81:        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=-2,
tests/bench/test_timed_scan_blocks.py-82-                          probe_steps=1)
tests/bench/test_timed_scan_blocks.py-83-    with pytest.raises(ValueError, match="probe_steps"):
tests/bench/test_timed_scan_blocks.py:84:        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-85-                          probe_steps=-1)
tests/bench/test_timed_scan_blocks.py-86-    with pytest.raises(ValueError, match="block_steps"):
tests/bench/test_timed_scan_blocks.py:87:        timed_scan_blocks(_advance, s0, block_steps=-1, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-88-                          probe_steps=0)
tests/bench/test_timed_scan_blocks.py-89-
tests/bench/test_timed_scan_blocks.py-90-
tests/bench/test_timed_scan_blocks.py-91-def test_seed_state_not_mutated_by_precompile():
--
tests/bench/test_timed_scan_blocks.py-94-    the timed blocks must be byte-identical to freshly advanced ones."""
tests/bench/test_timed_scan_blocks.py-95-    import numpy as np
tests/bench/test_timed_scan_blocks.py-96-    s0 = {"x": jnp.arange(6.0)}
tests/bench/test_timed_scan_blocks.py-97-    ref = np.asarray(s0["x"]).copy()
tests/bench/test_timed_scan_blocks.py:98:    s, m = timed_scan_blocks(_advance, dict(s0), block_steps=1, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-99-                             probe_steps=0)
tests/bench/test_timed_scan_blocks.py-100-    # 1 compile + 0 probe + 1x1 block = 2 steps from the ORIGINAL seed.
tests/bench/test_timed_scan_blocks.py-101-    want = ref
tests/bench/test_timed_scan_blocks.py-102-    for _ in range(2):
--
scripts/bench/bench_ocean_mpas_scaling.py-648-    # allgathered so the parallel time of block b is the SLOWEST rank in
scripts/bench/bench_ocean_mpas_scaling.py-649-    # that block (the same slowest-rank convention as the per-step MAX).
scripts/bench/bench_ocean_mpas_scaling.py-650-    fused = None
scripts/bench/bench_ocean_mpas_scaling.py-651-    if args.block_steps > 0:
scripts/bench/bench_ocean_mpas_scaling.py:652:        state, _t = timed_scan_blocks(
scripts/bench/bench_ocean_mpas_scaling.py-653-            advance, state,
scripts/bench/bench_ocean_mpas_scaling.py-654-            block_steps=args.block_steps, n_blocks=args.blocks,
scripts/bench/bench_ocean_mpas_scaling.py-655-            probe_steps=args.probe_steps,
scripts/bench/bench_ocean_mpas_scaling.py-656-            sync_label="ocean_mpas_mpi_bench")
--
scripts/bench/metadata.py-886-        },
scripts/bench/metadata.py-887-    }
scripts/bench/metadata.py-888-
scripts/bench/metadata.py-889-
scripts/bench/metadata.py:890:def timed_scan_blocks(
scripts/bench/metadata.py-891-    advance,
scripts/bench/metadata.py-892-    state,
scripts/bench/metadata.py-893-    *,
scripts/bench/metadata.py-894-    block_steps: int,
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-474-    # aux threads the sharded geometry stacks through the jit boundary as
scripts/bench/bench_ocean_latlon_spmd_scaling.py-475-    # an ARGUMENT (outer-trace constants of non-addressable arrays are
scripts/bench/bench_ocean_latlon_spmd_scaling.py-476-    # unfetchable — see make_sharded_ocean_step's aux note).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-477-    _aux = getattr(step, "aux", None)
scripts/bench/bench_ocean_latlon_spmd_scaling.py:478:    s, timing = timed_scan_blocks(
scripts/bench/bench_ocean_latlon_spmd_scaling.py-479-        (lambda st, aux: step(st, args.dt, aux=aux)) if _aux is not None
scripts/bench/bench_ocean_latlon_spmd_scaling.py-480-        else (lambda st: step(st, args.dt)),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-481-        s,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-482-        block_steps=_blk, n_blocks=_nblk, probe_steps=_probe,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-605-        elif hasattr(val, "data"):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-606-            updates[name] = val.replace(data=_gather_arr(val.data))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-607-        else:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-608-            updates[name] = _gather_arr(jnp.asarray(val))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-609-    return state._replace(**updates)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-610-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-611-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:612:def make_sharded_ocean_step(model, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-613-    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-614-    sponge=None, t_seconds=None) -> state`` running ``model.step``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-615-    lat-band-SPMD.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-616-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-617-    The forcing channels mirror ``model.step``'s keyword surface: pass
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-618-    pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-619-    ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1003-    ``mesh is None`` ⇒ the plain single-device ``model.step`` (no scatter/gather).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1004-    """
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1005-    if mesh is None:                   # single-device: plain step
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1006-        return lambda state, dt, surface_forcing=None, freshwater=None: (
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1007-            model.step(state, dt, freshwater=freshwater,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1008-                       surface_forcing=surface_forcing))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1009-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:1010:    inner = make_sharded_ocean_step(model, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1011-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:1012:    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1013-        # Scatter the global state AND forcing to the band layout explicitly
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1014-        # (codex r17 item 3: the old comment claimed inner sharded the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1015-        # forcing; it forwarded it global and relied on implicit JIT input
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1016-        # placement — which under multicontroller pays jax's whole-array
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1017-        # device_put assert, the nd-linear wall this module removes).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1018-        ss = shard_state_latlon(state, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1019-        ss = inner(ss, dt,
--
tests/unit/test_run_omip_latlon_spmd.py-75-
tests/unit/test_run_omip_latlon_spmd.py-76-    serial_final, _d, _w, ok_serial, _b = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-77-        model, state0, checkpoint_dir=tmp_path / "serial", **common)
tests/unit/test_run_omip_latlon_spmd.py-78-    assert ok_serial
tests/unit/test_run_omip_latlon_spmd.py-79-
tests/unit/test_run_omip_latlon_spmd.py-80-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py-81-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py:82:    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-83-    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-84-    ckpt_dir = tmp_path / "spmd"
tests/unit/test_run_omip_latlon_spmd.py-85-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-86-        model, ss0, checkpoint_dir=ckpt_dir,
tests/unit/test_run_omip_latlon_spmd.py-87-        spmd_step=spmd_step,
tests/unit/test_run_omip_latlon_spmd.py-88-        spmd_gather=lambda st: gather_state_latlon(st, dev.mesh),
tests/unit/test_run_omip_latlon_spmd.py-89-        **common)
--
tests/unit/test_run_omip_latlon_spmd.py-212-
tests/unit/test_run_omip_latlon_spmd.py-213-    state0 = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
tests/unit/test_run_omip_latlon_spmd.py-214-    serial_final, _d, _w, ok_serial, _b = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-215-        model, state0, jra55_state=_fresh_js(), **common)
tests/unit/test_run_omip_latlon_spmd.py-216-    assert ok_serial, "serial JRA55 loop reported not-ok"
tests/unit/test_run_omip_latlon_spmd.py-217-
tests/unit/test_run_omip_latlon_spmd.py-218-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py:219:    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-220-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py-221-    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-222-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-223-        model, ss0, jra55_state=_fresh_js(),
tests/unit/test_run_omip_latlon_spmd.py-224-        spmd_step=spmd_step,
tests/unit/test_run_omip_latlon_spmd.py-225-        spmd_gather=lambda st: gather_state_latlon(st, dev.mesh),
tests/unit/test_run_omip_latlon_spmd.py-226-        spmd_shard_stack=partial(shard_forcing_stack_latlon, mesh=dev.mesh),
--
tests/parallel/test_latlon_ocean_spmd_step.py-123-
tests/parallel/test_latlon_ocean_spmd_step.py-124-    # lat-band SPMD on 4 devices.  The state is laid out with shard_state_latlon
tests/parallel/test_latlon_ocean_spmd_step.py-125-    # (cell fields P("lat"); the staggered v / v_mask carried as v_lower, the
tests/parallel/test_latlon_ocean_spmd_step.py-126-    # n_lat-row block that DOES divide N — a uniform tree.map(P("lat")) would
tests/parallel/test_latlon_ocean_spmd_step.py-127-    # fail on the n_lat+1 v rows).  The result is gathered (and the dropped pole
tests/parallel/test_latlon_ocean_spmd_step.py-128-    # row reappended) for the bit-comparison vs the single-device reference.
tests/parallel/test_latlon_ocean_spmd_step.py-129-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:130:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-131-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-132-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-133-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_step.py-134-    ss = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-135-
tests/parallel/test_latlon_ocean_spmd_step.py-136-    # Tolerance = the FLOATING-POINT RE-ASSOCIATION floor of the sharded
tests/parallel/test_latlon_ocean_spmd_step.py-137-    # split-explicit barotropic, NOT a bug margin. A clean-input bisection proves
--
tests/parallel/test_latlon_ocean_spmd_step.py-240-    s = state0
tests/parallel/test_latlon_ocean_spmd_step.py-241-    for i in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-242-        s = model.step(s, dt, freshwater=fw, surface_forcing=sf,
tests/parallel/test_latlon_ocean_spmd_step.py-243-                       sponge=sponge, t_seconds=jnp.asarray(i * dt))
tests/parallel/test_latlon_ocean_spmd_step.py-244-
tests/parallel/test_latlon_ocean_spmd_step.py-245-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-246-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:247:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-248-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-249-    fws = shard_forcing_latlon(fw, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-250-    sfs = shard_forcing_latlon(sf, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-251-    sponges = shard_forcing_latlon(sponge, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-252-
tests/parallel/test_latlon_ocean_spmd_step.py-253-    # Cache-key flip smoke: dynamics-only compile first, then the forcing
tests/parallel/test_latlon_ocean_spmd_step.py-254-    # program — the second call MUST rebuild (different forcing structure),
--
tests/parallel/test_latlon_ocean_spmd_step.py-392-    grid = create_latlon_grid(n_lat=16, n_lon=32)
tests/parallel/test_latlon_ocean_spmd_step.py-393-    z_coord = create_ocean_z_star(n_levels=3, H_max=4000.0)
tests/parallel/test_latlon_ocean_spmd_step.py-394-    model = LatLonCGridOceanModel(grid, z_coord,
tests/parallel/test_latlon_ocean_spmd_step.py-395-                                  LatLonCGridOceanConfig.from_flat())
tests/parallel/test_latlon_ocean_spmd_step.py-396-    state0 = _perturbed_state(grid, z_coord)
tests/parallel/test_latlon_ocean_spmd_step.py-397-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-398-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:399:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-400-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-401-    bad = OceanSurfaceForcing(
tests/parallel/test_latlon_ocean_spmd_step.py-402-        tau_y=jnp.zeros((grid.n_lat + 1, grid.n_lon)))
tests/parallel/test_latlon_ocean_spmd_step.py-403-    with pytest.raises(ValueError, match="leading dim"):
tests/parallel/test_latlon_ocean_spmd_step.py-404-        step(ss, 600.0, surface_forcing=bad)
tests/parallel/test_latlon_ocean_spmd_step.py-405-
tests/parallel/test_latlon_ocean_spmd_step.py-406-
--
tests/parallel/test_latlon_ocean_spmd_step.py-447-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-448-        s = model.step(s, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_step.py-449-
tests/parallel/test_latlon_ocean_spmd_step.py-450-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-451-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py-452-
tests/parallel/test_latlon_ocean_spmd_step.py-453-    # explicit scatter -> inner sharded step -> gather
tests/parallel/test_latlon_ocean_spmd_step.py:454:    inner = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-455-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-456-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-457-        ss = inner(ss, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_step.py-458-    ss = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-459-
tests/parallel/test_latlon_ocean_spmd_step.py-460-    # global-in/global-out wrapper (scatter + gather PER STEP)
tests/parallel/test_latlon_ocean_spmd_step.py-461-    glob = make_sharded_ocean_step_global(model, dev.mesh)
--
tests/parallel/test_latlon_spmd_fused_halo.py-342-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_spmd_fused_halo.py-343-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_spmd_fused_halo.py-344-
tests/parallel/test_latlon_spmd_fused_halo.py-345-    outs = {}
tests/parallel/test_latlon_spmd_fused_halo.py-346-    hlo_counts = {}
tests/parallel/test_latlon_spmd_fused_halo.py-347-    for flag in ("0", "1"):
tests/parallel/test_latlon_spmd_fused_halo.py-348-        _env_flag(monkeypatch, flag)
tests/parallel/test_latlon_spmd_fused_halo.py:349:        step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-350-        ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-351-        hlo_counts[flag] = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-352-            jax.jit(step).lower(ss, 600.0).compile().as_text())
tests/parallel/test_latlon_spmd_fused_halo.py-353-        for _ in range(n_steps):
tests/parallel/test_latlon_spmd_fused_halo.py-354-            ss = step(ss, 600.0)
tests/parallel/test_latlon_spmd_fused_halo.py-355-        outs[flag] = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-356-
--
tests/parallel/test_latlon_spmd_fused_halo.py-365-    # switch is trace-time; the step wrapper's per-CALL cache key carries
tests/parallel/test_latlon_spmd_fused_halo.py-366-    # it — codex).  Fresh outer lambdas per lowering: jax.jit caches on
tests/parallel/test_latlon_spmd_fused_halo.py-367-    # the callable identity, so re-jitting the SAME step object would
tests/parallel/test_latlon_spmd_fused_halo.py-368-    # freeze the wrapper's python body and never re-evaluate the key
tests/parallel/test_latlon_spmd_fused_halo.py-369-    # (an outer-jit artifact, not the production call pattern — the
tests/parallel/test_latlon_spmd_fused_halo.py-370-    # driver calls step() directly each step).
tests/parallel/test_latlon_spmd_fused_halo.py-371-    _env_flag(monkeypatch, "0")
tests/parallel/test_latlon_spmd_fused_halo.py:372:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-373-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-374-    n_off = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-375-        jax.jit(lambda s, d: step(s, d)).lower(ss, 600.0)
tests/parallel/test_latlon_spmd_fused_halo.py-376-        .compile().as_text())
tests/parallel/test_latlon_spmd_fused_halo.py-377-    _env_flag(monkeypatch, "1")
tests/parallel/test_latlon_spmd_fused_halo.py-378-    n_on = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-379-        jax.jit(lambda s, d: step(s, d)).lower(ss, 600.0)
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-443-    inv_before = None
scripts/bench/bench_ocean_latlon_spmd_scaling.py-444-    if args.check_conservation:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-445-        from bench_ocean_mpi_scaling import ocean_invariants
scripts/bench/bench_ocean_latlon_spmd_scaling.py-446-        inv_before = ocean_invariants(model, s0, n_ranks=1)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-447-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-448-    if nd == 1:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-449-        mesh = None
scripts/bench/bench_ocean_latlon_spmd_scaling.py:450:        step = make_sharded_ocean_step(model, None)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-451-        s = s0
scripts/bench/bench_ocean_latlon_spmd_scaling.py-452-    else:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-453-        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-454-                                 axis_names=("lat",))
scripts/bench/bench_ocean_latlon_spmd_scaling.py:455:        step = make_sharded_ocean_step(model, mesh)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-456-        s = shard_state_latlon(s0, mesh)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-457-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-458-    # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
scripts/bench/bench_ocean_latlon_spmd_scaling.py-459-    # blocks with sync only AROUND the block — the previous per-step
scripts/bench/bench_ocean_latlon_spmd_scaling.py-460-    # host-synced loop measured dispatch+sync latency, not fused device
scripts/bench/bench_ocean_latlon_spmd_scaling.py-461-    # throughput.  Per-step dispatch latency is still measured, SEPARATELY,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-462-    # by an individually-synced probe (``step_latency_ms``); multi-controller
--
tests/parallel/test_persistent_sharded_ocean_loop.py-194-    sg = jax.block_until_ready(sg)
tests/parallel/test_persistent_sharded_ocean_loop.py-195-    old_calls = dict(calls)
tests/parallel/test_persistent_sharded_ocean_loop.py-196-    # the wrapper lane pays 2 full-state transfers PER STEP
tests/parallel/test_persistent_sharded_ocean_loop.py-197-    assert old_calls == {"shard": n_steps, "gather": n_steps}, old_calls
tests/parallel/test_persistent_sharded_ocean_loop.py-198-
tests/parallel/test_persistent_sharded_ocean_loop.py-199-    # ---------------- NEW lane: persistent sharded loop ----------------------
tests/parallel/test_persistent_sharded_ocean_loop.py-200-    calls["shard"] = calls["gather"] = 0
tests/parallel/test_persistent_sharded_ocean_loop.py:201:    inner = sos.make_sharded_ocean_step(model, mesh)
tests/parallel/test_persistent_sharded_ocean_loop.py-202-    ss = sos.shard_state_latlon(state0, mesh)          # ONE initial shard
tests/parallel/test_persistent_sharded_ocean_loop.py-203-    for k in range(1, n_steps + 1):
tests/parallel/test_persistent_sharded_ocean_loop.py-204-        # surface-current consumer on the SHARDED state: v is the n_lat-row
tests/parallel/test_persistent_sharded_ocean_loop.py-205-        # v_lower carrier at every loop top (the snapshot boundary re-shards
tests/parallel/test_persistent_sharded_ocean_loop.py-206-        # back to it); _surface_currents reconstructs the staggered top row
tests/parallel/test_persistent_sharded_ocean_loop.py-207-        # as the wall zero DEVICE-SIDE — the gather counters below prove no
tests/parallel/test_persistent_sharded_ocean_loop.py-208-        # full-state layout flip happens here.
--
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-118-    s = state0
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-119-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-120-        s = model.step(s, dt)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-121-    # Prime the build-once vertex-mask cache from the CONCRETE state (the
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-122-    # serial run above already did; belt-and-braces for wrapper band masks).
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-123-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-124-
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:125:    step = make_sharded_ocean_step(model, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-126-    ss = shard_state_latlon(state0, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-127-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-128-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-129-    out = gather_state_latlon(ss, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-130-
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-131-    # Tolerance = the split-explicit FP re-association floor of the sharded
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-132-    # barotropic (see test_latlon_ocean_spmd_step's tolerance note verbatim).
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-85-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-86-    s = state0
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-87-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-88-        s = model.step(s, dt)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-89-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-90-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-91-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:92:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-93-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-94-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-95-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-96-    ss = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-97-    return s, ss
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-98-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-99-
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-118-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-119-    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-120-    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-121-    model = LatLonCGridOceanModel(grid, z_coord, cfg)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-122-    state0 = _perturbed_state(grid, z_coord)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-123-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-124-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:125:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-126-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-127-    lowered = jax.jit(step).lower(ss, 600.0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-128-    return lowered.compile().as_text()
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-129-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-130-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-131-def _while_body_collective_count(hlo_text: str) -> int:
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-132-    """Collective-permute ops inside while-LOOP BODY computations.
--
tests/parallel/test_cube_tile_native_segment.py-194-
tests/parallel/test_cube_tile_native_segment.py-195-    # exit_ reassembles a cc HydrostaticState (segment-boundary I/O path).
tests/parallel/test_cube_tile_native_segment.py-196-    out = exit_(blk, hs)
tests/parallel/test_cube_tile_native_segment.py-197-    assert out.u.data.shape == hs.u.data.shape
tests/parallel/test_cube_tile_native_segment.py-198-    assert bool(jnp.all(jnp.isfinite(out.u.data)))
tests/parallel/test_cube_tile_native_segment.py-199-
tests/parallel/test_cube_tile_native_segment.py-200-
tests/parallel/test_cube_tile_native_segment.py:201:def test_segment_matches_face_spmd_step():
tests/parallel/test_cube_tile_native_segment.py-202-    """Cross-LANE gate: the tiled segment (6,2,2 over 24 devices) == the
tests/parallel/test_cube_tile_native_segment.py-203-    EXISTING face-SPMD production step (6-device face mesh +
tests/parallel/test_cube_tile_native_segment.py-204-    activate_spmd_halo_backend — the <=6-device production lane).
tests/parallel/test_cube_tile_native_segment.py-205-
tests/parallel/test_cube_tile_native_segment.py-206-    The task-level '(6,1,1)' tiled mesh does not exist BY DESIGN: kt=1 is
tests/parallel/test_cube_tile_native_segment.py-207-    refused (``cubesphere_exchange._build_tiled_tables`` — at one tile per
tests/parallel/test_cube_tile_native_segment.py-208-    face the tiled exchange IS the face exchange), and the driver
--
tests/parallel/test_latlon_ocean_spmd_tripole.py-145-    s = state0
tests/parallel/test_latlon_ocean_spmd_tripole.py-146-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_tripole.py-147-        s = model.step(s, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_tripole.py-148-
tests/parallel/test_latlon_ocean_spmd_tripole.py-149-    model._ensure_vertex_mask(state0)        # prime the build-once vmask cache
tests/parallel/test_latlon_ocean_spmd_tripole.py-150-
tests/parallel/test_latlon_ocean_spmd_tripole.py-151-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_tripole.py:152:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-153-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-154-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_tripole.py-155-        ss = step(ss, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_tripole.py-156-    ss = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-157-
tests/parallel/test_latlon_ocean_spmd_tripole.py-158-    # FP re-association floor (NOT a bug margin): the sharded reductions are
tests/parallel/test_latlon_ocean_spmd_tripole.py-159-    # batch_psum_spmd across bands (provably global — a missing/local reduction
--
scripts/run/run_omip_core2.py-5564-            # gathers, and the counted per-step gathers forced by host-global
scripts/run/run_omip_core2.py-5565-            # consumers).  t_sec is always None here (tide fail-fasts above).
scripts/run/run_omip_core2.py-5566-            from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip_core2.py-5567-                gather_state_latlon,
scripts/run/run_omip_core2.py-5568-                make_sharded_ocean_step,
scripts/run/run_omip_core2.py-5569-                shard_state_latlon,
scripts/run/run_omip_core2.py-5570-            )
scripts/run/run_omip_core2.py:5571:            _spmd_inner = make_sharded_ocean_step(model, _spmd_mesh)
scripts/run/run_omip_core2.py-5572-            _ocean_step = (lambda st, sf, fw, t_sec=None:
scripts/run/run_omip_core2.py:5573:                           _spmd_inner(st, dt, surface_forcing=sf,
scripts/run/run_omip_core2.py-5574-                                       freshwater=fw))
scripts/run/run_omip_core2.py-5575-
scripts/run/run_omip_core2.py-5576-            def _pers_shard_fn(st, _mesh=_spmd_mesh):
scripts/run/run_omip_core2.py-5577-                return shard_state_latlon(st, _mesh)
scripts/run/run_omip_core2.py-5578-
scripts/run/run_omip_core2.py-5579-            def _pers_gather_fn(st, _mesh=_spmd_mesh):
scripts/run/run_omip_core2.py-5580-                return gather_state_latlon(st, _mesh)
--
scripts/run/run_omip_core2.py-5585-                  f"rows/band); PERSISTENT sharded state "
scripts/run/run_omip_core2.py-5586-                  f"(--spmd-persistent-state): full-state gathers only at "
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
scripts/run/run_omip_core2.py-5598-
scripts/run/run_omip_core2.py-5599-    # ------------------------------------------------------------------
--
scripts/run/run_omip.py-4816-                shard_state_latlon,
scripts/run/run_omip.py-4817-            )
scripts/run/run_omip.py-4818-            from legoesm.parallel.mesh import create_latlon_mesh
scripts/run/run_omip.py-4819-            # Prime the build-once vertex-mask cache from the CONCRETE
scripts/run/run_omip.py-4820-            # state so the wrapper can build per-band masks host-side.
scripts/run/run_omip.py-4821-            model.prime_step_caches(state)
scripts/run/run_omip.py-4822-            _dev = create_latlon_mesh(n_devices=_nd)
scripts/run/run_omip.py:4823:            spmd_step = make_sharded_ocean_step(model, _dev.mesh)
scripts/run/run_omip.py-4824-            spmd_gather = partial(gather_state_latlon, mesh=_dev.mesh)
scripts/run/run_omip.py-4825-            state = shard_state_latlon(state, _dev.mesh)
scripts/run/run_omip.py-4826-            # Lay per-block forcing stacks out lat-band-sharded so the
scripts/run/run_omip.py-4827-            # in-scan interpolation / bulk fluxes stay shard-local (shared
scripts/run/run_omip.py-4828-            # layout helper — see shard_forcing_stack_latlon).
scripts/run/run_omip.py-4829-            spmd_shard_stack = partial(
scripts/run/run_omip.py-4830-                shard_forcing_stack_latlon, mesh=_dev.mesh)

exec
/usr/bin/bash -lc "sed -n '980,1095p' scripts/bench/metadata.py
sed -n '4700,4935p' scripts/run/run_omip.py
sed -n '5520,5635p' scripts/run/run_omip_core2.py
rg -n -C 4 \"spmd_step|_spmd_inner|_ocean_step|timed_scan_blocks|jax\\.jit|lax\\.scan\" scripts/run/run_omip.py scripts/run/run_omip_core2.py scripts/bench/bench_ocean_mpas_scaling.py scripts/bench/bench_atm_latlon_spmd_scaling.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 240ms:
        _sched = np.asarray(multihost_utils.process_allgather(
            np.array([block_steps, n_blocks, probe_steps], dtype=np.int64)))
        if not bool((_sched == _sched[0]).all()):
            raise ValueError(
                "timed_scan_blocks: processes disagree on the schedule "
                f"(block_steps, n_blocks, probe_steps) = {_sched.tolist()} "
                "per process — a mismatched schedule deadlocks in the "
                "named fences.")
    # No silent rewrites: invalid values raise; the recorded schedule IS
    # the executed schedule.  block_steps == 0 stays legal (the documented
    # zero-length parity path).
    if block_steps < 0:
        raise ValueError(
            f"timed_scan_blocks: block_steps must be >= 0, got {block_steps}")
    if n_blocks < 1:
        raise ValueError(
            f"timed_scan_blocks: n_blocks must be >= 1, got {n_blocks}")
    if probe_steps < 0:
        raise ValueError(
            f"timed_scan_blocks: probe_steps must be >= 0, got {probe_steps}")

    def _block(tree):
        jax.block_until_ready(jax.tree_util.tree_leaves(tree))

    def _fence(tag: str):
        if multi:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices(f"{sync_label}_{tag}")

    # Normalize to a 2-arg advance (see the aux note below): top-level
    # calls here are outside any trace, so passing aux is just an argument.
    if aux is None:
        def _advance(carry, _aux):
            return advance(carry)
    else:
        _advance = advance

    # --- 1. compile (first call of advance, separated from all timing) ---
    _fence("compile_start")
    t0 = time.perf_counter()
    state = _advance(state, aux)
    _block(state)
    compile_ms = (time.perf_counter() - t0) * 1e3

    # --- 2. dispatch-latency probe: individually synced steps, reported
    # separately (NEVER mixed into the fused number) ---
    probe_ms = []
    for _ in range(probe_steps):
        t0 = time.perf_counter()
        state = _advance(state, aux)
        _block(state)
        probe_ms.append((time.perf_counter() - t0) * 1e3)
    step_latency_ms = float(np.median(probe_ms)) if probe_ms else float("nan")

    # --- 3. fused scan block (dtype-stable carry, the OM pattern) ---
    input_dtypes = jax.tree_util.tree_map(
        lambda x: x.dtype if hasattr(x, "dtype") else None, state)

    # ``aux``: extra pytree threaded through the jit boundary as an ARGUMENT
    # (never a closure capture). Needed when ``advance`` internally calls a
    # jit function over SHARDED-global auxiliary arrays (the ocean band
    # geometry stacks): jit-of-jit inlines the inner call, so concrete
    # closure arrays would become outer-trace CONSTANTS whose value the MLIR
    # handler cannot fetch for non-addressable arrays (2026-08-03 repro).
    @jax.jit
    def _scan_run(st, _aux):
        def _body(carry, _):
            new = _advance(carry, _aux)
            new = jax.tree_util.tree_map(
                lambda x, d: x.astype(d)
                if d is not None and hasattr(x, "astype") else x,
                new, input_dtypes)
            return new, None
        return jax.lax.scan(_body, st, None, length=block_steps)[0]

    # Pre-compile the scan on a SHALLOW pytree copy: the leaves ALIAS the
    # seed state's arrays (no data copy) — sufficient AND safe because the
    # pre-call only needs matching shapes/dtypes/shardings to warm the
    # compile cache, jitted execution is pure, and ``advance`` is
    # contract-bound not to donate buffers (see the docstring).  The seed
    # value itself is untouched; every process executes the same collective
    # schedule — counts stay matched.
    _pre = jax.tree_util.tree_map(lambda x: x, state)
    t0 = time.perf_counter()
    _pre_out = _scan_run(_pre, aux)
    _block(_pre_out)
    scan_compile_ms = (time.perf_counter() - t0) * 1e3
    del _pre, _pre_out

    block_ms = []
    for b in range(n_blocks):
        _fence(f"block{b}_start")
        t0 = time.perf_counter()
        state = _scan_run(state, aux)
        _block(state)
        block_ms.append((time.perf_counter() - t0) * 1e3)
    _fence("blocks_end")

    # Slowest-rank statistics from the FULL per-block vectors (equal length
    # everywhere — the schedule was collectively verified above).  The
    # parallel time of block b is the slowest process IN that block; a
    # gather of per-process medians would let alternating stragglers make
    # every rank median look fast (codex batch4).
    my_blocks = np.asarray(block_ms, dtype=np.float64)
    if multi:
        from jax.experimental import multihost_utils
        all_blocks = np.asarray(
            multihost_utils.process_allgather(my_blocks))
    else:
        all_blocks = my_blocks[None, :]
    parallel_block_ms = np.max(all_blocks, axis=0)        # (n_blocks,)
    rank_median_ms = np.median(all_blocks, axis=0)        # (n_blocks,)
    with np.errstate(divide="ignore", invalid="ignore"):
        imb_per_block = np.where(rank_median_ms > 0.0,
                                 parallel_block_ms / rank_median_ms,
                                 np.nan)
        with open(config_path, "w") as f:
            json.dump(run_config_json, f, indent=2, default=str)
        print(f"  Config saved: {config_path}")
    except Exception as e:
        print(f"  Warning: could not save config: {e}")

    # Restart cadence — only wired for the JRA55 path for now (the
    # restoring path is fast enough that re-running from scratch is
    # cheaper than maintaining restarts; revisit if needed).
    checkpoint_dir = None
    checkpoint_days = None
    if (
        (jra55_state is not None and args.checkpoint_days > 0.0)
        or run_config.max_wallclock_seconds > 0.0
    ):
        checkpoint_dir = Path(args.output) / grid_type / resolution / "restarts"
        if jra55_state is not None and args.checkpoint_days > 0.0:
            checkpoint_days = float(args.checkpoint_days)
            print(
                f"  Restart cadence: every {checkpoint_days:g} simulated days "
                f"→ {checkpoint_dir}"
            )

    # Build snapshot function for auto-plotting with each restart save.
    # Only for MPAS with the tripcolor/cartopy plotter; other grids use
    # the end-of-run timeseries plot only.
    _snapshot_fn = None
    if grid_type == "mpas":
        try:
            from scripts.plot.plot_mpas_omip_snapshot import plot_snapshot as _plot_snap
            import threading
            _snap_mesh = grid
            _snap_z = z_coord
            _snap_lock = threading.Lock()

            def _snapshot_fn(restart_path):
                """Plot snapshot in a background thread so the GPU isn't blocked.

                Uses a lock to serialize matplotlib calls (not thread-safe).
                """
                def _render():
                    # The restart write is itself async (atomic tmp+rename,
                    # see _save_restart): join the in-flight writer so the
                    # npz exists before the plotter reads it.
                    _join_restart_writer()
                    with _snap_lock:
                        try:
                            _plot_snap(restart_path, _snap_mesh, _snap_z)
                        except Exception as e:
                            print(f"    Snapshot failed: {e}", flush=True)
                t = threading.Thread(target=_render, daemon=True)
                t.start()
        except ImportError:
            pass

    # Restoring ramp: cubed_sphere defaults to 14d to delay the
    # face-edge PGF instability; other grids default to 0d.  Sentinel
    # ``None`` from argparse means "use grid default" — an explicit
    # ``--restoring-ramp-days 0`` honours the user's choice.
    if args.restoring_ramp_days is not None:
        ramp_days_eff = args.restoring_ramp_days
    elif grid_type == "cubed_sphere":
        ramp_days_eff = 14.0
    else:
        ramp_days_eff = 0.0

    # --- Lat-band SPMD (--enable-latlon-spmd): wrap the dynamics step in
    # the validated multi-GPU sharded ocean step and shard the state.
    # Single-controller only; restoring lane only (the JRA55 block
    # functions call model._step_impl directly — follow-up).  Runs AFTER
    # the restart load so a resumed state is sharded too.
    spmd_step = None
    spmd_gather = None
    spmd_shard_stack = None
    if run_config.enable_latlon_spmd:
        if grid_type != "latlon":
            raise SystemExit(
                f"--enable-latlon-spmd requires --grid latlon "
                f"(got {grid_type}).")
        if (jra55_state is not None
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
        restart_buffer_seconds=run_config.restart_buffer_seconds,
        start_step=start_step,
        nudge_woa_tau=args.nudge_woa_tau,
        T_woa_3d=(T_woa * state.land_mask.data[..., jnp.newaxis]).astype(
            state.T.data.dtype) if args.nudge_woa_tau > 0 and T_woa is not None else None,
        S_woa_3d=(S_woa * state.land_mask.data[..., jnp.newaxis]).astype(
            state.S.data.dtype) if args.nudge_woa_tau > 0 and S_woa is not None else None,
        snapshot_fn=_snapshot_fn,
        spmd_step=spmd_step,
        spmd_gather=spmd_gather,
        spmd_shard_stack=spmd_shard_stack,
    )
    if spmd_gather is not None:
        # Downstream report/plot/save paths expect the full (n_lat+1)
        # staggered v layout, not the sharded v_lower carry.  Every rank
        # dispatches this gather (it is a collective); only rank 0 writes.
        state = spmd_gather(state)

    # Surface a failed FINAL async restart write while the run can still
    # report it (the writer thread swallows exceptions; _save_restart only
    # re-raises them one checkpoint later — there is no later checkpoint
    # for the last one; codex audit HIGH, 2026-07-02).  Placed AFTER the
    # spmd_gather collective above: under route-B multiproc only rank 0
    # spawns a writer thread, so a rank-0-only re-raise here must not be able
    # to skip that collective and hang the federation.
    _join_restart_writer()

    # Route-B: only rank 0 writes output files (concurrent writes to the same
    # path corrupt them); every rank still builds ``results`` so the exit code
    # is consistent across the federation.
    _io_rank = jax.process_index() == 0
    _multiproc = jax.process_count() > 1

    status = "PASS" if ok else "FAIL"
    icon = "  " if ok else "**"
    sst_str = f"SST={diag['SST'][-1]:.2f}" if diag["SST"] else ""
    # iter-97: when the run blew up, ``diag['SST'][-1]`` is the
    # last *clean* diagnostic from BEFORE the BLOWUP, which can
    # mislead the reader into thinking the run is healthy.
    # Show the BLOWUP marker explicitly.
    if blowup_info is not None:
        sst_str = f"BLOWUP at step {blowup_info['step']}"
    if _io_rank:
        print(f"\n  {icon} {status} | {grid_type}/{resolution} | "
              f"{wall_time:.1f}s | {sst_str}")

    # Save output.  Route-B: a rank-0 write failure must NOT raise past this
    # point — the loop's collectives are done, but a bare exception would give
    # rank 0 a different ALL_RESULTS / exit code than the non-root ranks (which
    # never write).  Under multiprocess, log and rebuild the results dict with
    # write=False so every rank returns the SAME record (codex r2 caveat B).
    output_dir = Path(args.output) / grid_type / resolution
    try:
        results = _save_output(
            output_dir, diag, args, grid_type, wall_time, ok,
            blowup_info=blowup_info, write=_io_rank,
        )
    except Exception as e:
        if not _multiproc:
            raise
        print(f"  WARNING: rank-0 output write failed (continuing for a "
              f"consistent federation exit code): {type(e).__name__}: {e}",
              flush=True)
        results = _save_output(
            output_dir, diag, args, grid_type, wall_time, ok,
            blowup_info=blowup_info, write=False,
        )

    # Final MLD-diagnostic snapshot (de Boyer Montegut / Treguier 2023): the
    # shared writer emits the T/S + geometry contract that
    # scripts/validate/compare_mld_dbm.py and compare_omip_nemo.py consume so
    # a finished run can be scored offline (e.g. CATKE-vs-KPP MLD).  Purely
    # additive output; a diagnostic must never abort the run.  Rank-0 only
    # (writes a file); ``state`` is already gathered/addressable on every rank.
    if _io_rank:
        try:
            from legoesm.ocean.restart import (
                save_mld_snapshot, grid_lat2d_lon2d_deg,
            )
            # Grid coords are radians; the scorers consume degrees -> convert
            # via the shared per-grid extractor (latlon/tripole/cube/mpas).
            lat2d, lon2d = grid_lat2d_lon2d_deg(grid, grid_type)
            snap = save_mld_snapshot(
                state, output_dir / "snapshot_final.npz", z_coord=z_coord,
                lat2d=lat2d, lon2d=lon2d,
                # The lat-band mesh takes the FIRST n_gpus global devices; a
                # mismatch with the launched device count silently idles ranks and
                # (worse) can place two bands on one node while another idles.
                raise SystemExit(
                    f"--distributed expects --n-gpus ({args.n_gpus}) == global "
                    f"device count ({_global}): one process per GPU, every device "
                    f"in the band mesh. Launch exactly {args.n_gpus} single-GPU "
                    f"processes (ntasks={args.n_gpus}).")
        else:
            _local = _jax.local_device_count()
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
    # lane keeps ``state`` in the lat-band SHARDED layout (v/v_mask carried as
    # the n_lat-row ``v_lower``) across steps; these two helpers flip the
    # residency at the classified boundaries and COUNT every full-state
    # transfer so the cost is visible in the run log (never silent).  With the
    # flag OFF both are exact no-ops (byte-identical default path).
    #
    # Host-op classification (scaling-M2 audit):
    # (a) sharded-safe, UNCHANGED on the persistent state: the leaf-wise host
    #     BCs (SSS restore / ice-thermo freeze relax / WOA nudge / spin-up
    #     drag) read+write single cell-centred leaves via np.asarray — an
    #     addressable sharded array assembles to the identical host values,
    #     and the drag's v touch operates on ``v_lower`` exactly (the dropped
    #     pole row is identically 0 and 0*decay == 0); the jnp per-column BCs
    #     (geothermal / ISF) are sharding-transparent under GSPMD; BBL is
    #     value-exact too, but its static lat-neighbour slice updates may make
    #     XLA insert device-side collectives / replicate T,S under eager GSPMD
    #     (correct, device-resident — NOT a host-layout flip, so it is
    #     intentionally outside the gather counters, which track full-state
    #     LAYOUT flips only); the
    #     forcing builders (compute_omip2_surface_forcing / _freshwater_)
    #     np.asarray-read state.T identically (pre-existing per-step host
    #     read, both lanes); _diag is EXACT on the sharded layout (the v top
    #     row it cannot see is identically 0 in the gathered layout too).
    # (b) global reductions: none on the host loop itself (the in-step
    #     reductions run through the SPMD-safe psum paths inside shard_map).
    # (c) host-global consumers needing the FULL (n_lat+1)-v global layout:
    #     the momentum-term debug dump (runs model internals on the host
    #     state) and the real I/O boundaries (snapshot / blowup abort /
    #     final diags+digest) — these gather via _ensure_global_state below
    #     (counted; snapshot-cadence ones re-shard lazily at the next step).
    #     Prognostic sea ice + relative winds are NOT in this class any more
    #     (scaling-M2 leftover): _surface_currents* read 2-D surface u/v
    #     slices DEVICE-SIDE and reconstruct the one dropped staggered top
    #     row as the wall zero (_surface_uv_faces -> append_vface_wall_row,
    #     exact under the v-carrier contract), so they operate on the
scripts/bench/bench_atm_latlon_spmd_scaling.py-2-atm step (make_sharded_atm_latlon_step / run_atm_latlon_spmd, the A1 work).
scripts/bench/bench_atm_latlon_spmd_scaling.py-3-
scripts/bench/bench_atm_latlon_spmd_scaling.py-4-Times the SHARDED step across an N-device ("lat",) mesh and reports per-step
scripts/bench/bench_atm_latlon_spmd_scaling.py-5-wall time + speedup vs 1 device.  The DEFAULT lane follows the M1 measurement
scripts/bench/bench_atm_latlon_spmd_scaling.py:6:contract (``metadata.timed_scan_blocks``): fused ``lax.scan`` blocks of
scripts/bench/bench_atm_latlon_spmd_scaling.py-7-``--steps`` steps with device sync only AROUND each block (``fused_step_ms``,
scripts/bench/bench_atm_latlon_spmd_scaling.py-8-slowest process across controllers) plus a SEPARATE individually-synced
scripts/bench/bench_atm_latlon_spmd_scaling.py-9-dispatch-latency probe (``step_latency_ms``) — never mixed.  The
scripts/bench/bench_atm_latlon_spmd_scaling.py-10-jit(shard_map) step is built once and cached by make_sharded_atm_latlon_step;
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-13-  strong: fixed (n_lat, n_lon, nlev), vary n_devices -> speedup = t(1)/t(n).
scripts/bench/bench_atm_latlon_spmd_scaling.py-14-  weak:   n_lat = nlat_per_dev * n_devices (fixed per-device rows) -> ideal flat.
scripts/bench/bench_atm_latlon_spmd_scaling.py-15-
scripts/bench/bench_atm_latlon_spmd_scaling.py-16-``--segment-steps N`` (M2b): times the COMPILED-SEGMENT lane instead — ONE
scripts/bench/bench_atm_latlon_spmd_scaling.py:17:jitted lax.scan of N sharded steps per block (make_sharded_atm_latlon_segment,
scripts/bench/bench_atm_latlon_spmd_scaling.py-18-band-SHARDED geometry, in-graph finite scalar), so --steps counts BLOCKS of N
scripts/bench/bench_atm_latlon_spmd_scaling.py-19-steps and the per-step numbers derive from whole-block wall times.  Unlike the
scripts/bench/bench_atm_latlon_spmd_scaling.py-20-default lane (which scans the bench-local step fn), the segment is the
scripts/bench/bench_atm_latlon_spmd_scaling.py-21-PRODUCTION artifact; the record carries ``segment_mode=true`` +
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-126-    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
scripts/bench/bench_atm_latlon_spmd_scaling.py-127-    p.add_argument("--nlat-per-dev", type=int, default=32,
scripts/bench/bench_atm_latlon_spmd_scaling.py-128-                   help="weak mode: lat rows per device")
scripts/bench/bench_atm_latlon_spmd_scaling.py-129-    p.add_argument("--steps", type=int, default=12,
scripts/bench/bench_atm_latlon_spmd_scaling.py:130:                   help="Default lane: steps per fused lax.scan timing "
scripts/bench/bench_atm_latlon_spmd_scaling.py-131-                        "block. Segment mode: number of timed BLOCKS of "
scripts/bench/bench_atm_latlon_spmd_scaling.py-132-                        "--segment-steps steps each.")
scripts/bench/bench_atm_latlon_spmd_scaling.py-133-    p.add_argument("--warmup", type=int, default=2,
scripts/bench/bench_atm_latlon_spmd_scaling.py-134-                   help="Segment mode: timed blocks dropped from steady "
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-141-    p.add_argument("--probe-steps", type=int, default=3,
scripts/bench/bench_atm_latlon_spmd_scaling.py-142-                   help="Default lane: individually-synced steps for the "
scripts/bench/bench_atm_latlon_spmd_scaling.py-143-                        "SEPARATE dispatch-latency probe (step_latency_ms).")
scripts/bench/bench_atm_latlon_spmd_scaling.py-144-    p.add_argument("--segment-steps", type=int, default=0,
scripts/bench/bench_atm_latlon_spmd_scaling.py:145:                   help="M2b: >0 compiles ONE lax.scan segment of this many "
scripts/bench/bench_atm_latlon_spmd_scaling.py-146-                        "steps (built once, reused; band-sharded geometry) "
scripts/bench/bench_atm_latlon_spmd_scaling.py-147-                        "and times BLOCKS of segment calls instead of "
scripts/bench/bench_atm_latlon_spmd_scaling.py-148-                        "per-step host dispatch. 0 = default fused lane.")
scripts/bench/bench_atm_latlon_spmd_scaling.py-149-    p.add_argument("--physics", choices=["none", "held_suarez"], default="none")
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-174-                        "from OMPI_COMM_WORLD_SIZE/RANK.")
scripts/bench/bench_atm_latlon_spmd_scaling.py-175-    args = p.parse_args()
scripts/bench/bench_atm_latlon_spmd_scaling.py-176-
scripts/bench/bench_atm_latlon_spmd_scaling.py-177-    # Validate the schedule BEFORE any model/device work: a zero/negative
scripts/bench/bench_atm_latlon_spmd_scaling.py:178:    # --steps would otherwise surface only as timed_scan_blocks' None
scripts/bench/bench_atm_latlon_spmd_scaling.py-179-    # headline (default lane) or an empty timed loop (segment mode) after
scripts/bench/bench_atm_latlon_spmd_scaling.py-180-    # the expensive build (the ocean twin's guard).
scripts/bench/bench_atm_latlon_spmd_scaling.py-181-    if args.steps < 1:
scripts/bench/bench_atm_latlon_spmd_scaling.py-182-        raise SystemExit(f"--steps must be >= 1, got {args.steps}")
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-246-
scripts/bench/bench_atm_latlon_spmd_scaling.py-247-    per_block_ms = None
scripts/bench/bench_atm_latlon_spmd_scaling.py-248-    completed_blocks = None
scripts/bench/bench_atm_latlon_spmd_scaling.py-249-    finite_ok = None   # default fused lane: no in-graph finite check -> null
scripts/bench/bench_atm_latlon_spmd_scaling.py:250:    timing = None   # metadata.timed_scan_blocks metrics (default lane only)
scripts/bench/bench_atm_latlon_spmd_scaling.py-251-    if seg_n > 0:
scripts/bench/bench_atm_latlon_spmd_scaling.py-252-        # Multi-controller: align every process before the timed loop so
scripts/bench/bench_atm_latlon_spmd_scaling.py-253-        # block wall times aren't skewed by startup jitter (and once after,
scripts/bench/bench_atm_latlon_spmd_scaling.py-254-        # so no process exits while peers still hold collectives in flight).
scripts/bench/bench_atm_latlon_spmd_scaling.py:255:        # (The default lane's fences live inside timed_scan_blocks.)
scripts/bench/bench_atm_latlon_spmd_scaling.py-256-        if jax.process_count() > 1:
scripts/bench/bench_atm_latlon_spmd_scaling.py-257-            from jax.experimental import multihost_utils
scripts/bench/bench_atm_latlon_spmd_scaling.py-258-            multihost_utils.sync_global_devices("atm_latlon_spmd_bench_start")
scripts/bench/bench_atm_latlon_spmd_scaling.py:259:        # Segment mode: each timed BLOCK is one compiled lax.scan of seg_n
scripts/bench/bench_atm_latlon_spmd_scaling.py-260-        # steps; the host sync per block is the production pattern — read the
scripts/bench/bench_atm_latlon_spmd_scaling.py-261-        # in-graph finite SCALAR, then block on the state for honest timing.
scripts/bench/bench_atm_latlon_spmd_scaling.py-262-        per_block_ms = []
scripts/bench/bench_atm_latlon_spmd_scaling.py-263-        finite_ok = True
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-291-            # this only keeps the diagnostic median well-defined).
scripts/bench/bench_atm_latlon_spmd_scaling.py-292-            steady = per_step_ms
scripts/bench/bench_atm_latlon_spmd_scaling.py-293-        med = float(np.median(steady))
scripts/bench/bench_atm_latlon_spmd_scaling.py-294-    else:
scripts/bench/bench_atm_latlon_spmd_scaling.py:295:        # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
scripts/bench/bench_atm_latlon_spmd_scaling.py-296-        # blocks with sync only AROUND the block — the previous per-step
scripts/bench/bench_atm_latlon_spmd_scaling.py-297-        # host-synced loop measured dispatch+sync latency, not fused device
scripts/bench/bench_atm_latlon_spmd_scaling.py-298-        # throughput.  Dispatch latency stays measured SEPARATELY
scripts/bench/bench_atm_latlon_spmd_scaling.py-299-        # (``step_latency_ms``); multi-controller runs record the
scripts/bench/bench_atm_latlon_spmd_scaling.py-300-        # slowest-process block time + imbalance ratio.
scripts/bench/bench_atm_latlon_spmd_scaling.py:301:        from metadata import timed_scan_blocks
scripts/bench/bench_atm_latlon_spmd_scaling.py:302:        c, timing = timed_scan_blocks(
scripts/bench/bench_atm_latlon_spmd_scaling.py-303-            lambda st: step(st, args.dt), c,
scripts/bench/bench_atm_latlon_spmd_scaling.py-304-            block_steps=args.steps, n_blocks=args.blocks,
scripts/bench/bench_atm_latlon_spmd_scaling.py-305-            probe_steps=args.probe_steps,
scripts/bench/bench_atm_latlon_spmd_scaling.py-306-            sync_label="atm_latlon_spmd_bench")
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-353-            else (args.single_dev_fused_ms if valid else None)),
scripts/bench/bench_atm_latlon_spmd_scaling.py-354-        halo_messages_per_step=comm_rec["halo_messages_per_step"],
scripts/bench/bench_atm_latlon_spmd_scaling.py-355-        halo_bytes_per_step=comm_rec["halo_bytes_per_step"],
scripts/bench/bench_atm_latlon_spmd_scaling.py-356-        n_reductions_per_step=_nred,
scripts/bench/bench_atm_latlon_spmd_scaling.py:357:        # rank imbalance is measured by timed_scan_blocks (default lane);
scripts/bench/bench_atm_latlon_spmd_scaling.py-358-        # the segment lane records no cross-process block gather -> null.
scripts/bench/bench_atm_latlon_spmd_scaling.py-359-        rank_imbalance=(float(timing["rank_imbalance"])
scripts/bench/bench_atm_latlon_spmd_scaling.py-360-                        if timing is not None else None),
scripts/bench/bench_atm_latlon_spmd_scaling.py-361-        latency_us=args.comm_latency_us,
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-389-            per_step_ms=[round(x, 2) for x in per_step_ms],
scripts/bench/bench_atm_latlon_spmd_scaling.py-390-            per_block_ms=[round(x, 2) for x in per_block_ms],
scripts/bench/bench_atm_latlon_spmd_scaling.py-391-        )
scripts/bench/bench_atm_latlon_spmd_scaling.py-392-    else:
scripts/bench/bench_atm_latlon_spmd_scaling.py:393:        # Default lane: the timed_scan_blocks metrics (fused_step_ms,
scripts/bench/bench_atm_latlon_spmd_scaling.py-394-        # step_latency_ms, block_ms, parallel_block_ms, rank_imbalance, ...
scripts/bench/bench_atm_latlon_spmd_scaling.py-395-        # — the M1 measurement contract).
scripts/bench/bench_atm_latlon_spmd_scaling.py-396-        rec.update(**timing)
scripts/bench/bench_atm_latlon_spmd_scaling.py-397-    # Flat aggregator-compatible identity + metric fields: without a
--
scripts/run/run_omip_core2.py-2503-    """
scripts/run/run_omip_core2.py-2504-    u_face = jnp.asarray(state.u.data)[..., 0]       # (n_lat, n_lon+1)
scripts/run/run_omip_core2.py-2505-    v_face = jnp.asarray(state.v.data)[..., 0]       # (n_lat[+1], n_lon)
scripts/run/run_omip_core2.py-2506-    if v_face.shape[0] == u_face.shape[0]:           # v_lower carrier layout
scripts/run/run_omip_core2.py:2507:        from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip_core2.py-2508-            append_vface_wall_row,
scripts/run/run_omip_core2.py-2509-        )
scripts/run/run_omip_core2.py-2510-        v_face = append_vface_wall_row(v_face)       # -> (n_lat+1, n_lon)
scripts/run/run_omip_core2.py-2511-    return u_face, v_face
--
scripts/run/run_omip_core2.py-3325-                   help="cubed-sphere face resolution n (C-n) for --grid cubed_sphere.")
scripts/run/run_omip_core2.py-3326-    p.add_argument("--n-gpus", type=int, default=1,
scripts/run/run_omip_core2.py-3327-                   help="Multi-GPU lat-band SPMD ocean step (latlon_bathy / tripole "
scripts/run/run_omip_core2.py-3328-                        "only): partition the ocean state by latitude band across N "
scripts/run/run_omip_core2.py:3329:                        "local devices via make_sharded_ocean_step_global. n_lat is "
scripts/run/run_omip_core2.py-3330-                        "padded with LAND rows at the SOUTH to a multiple of N (the "
scripts/run/run_omip_core2.py-3331-                        "tripole north fold stays at the north). The host post-step "
scripts/run/run_omip_core2.py-3332-                        "BCs (SSS restore / prognostic ice / geothermal / BBL / nudge) "
scripts/run/run_omip_core2.py-3333-                        "run on the gathered GLOBAL state, unchanged. Default 1 = the "
--
scripts/run/run_omip_core2.py-3347-                        "is written ONLY by process 0. Single-process (default, no "
scripts/run/run_omip_core2.py-3348-                        "--distributed) is byte-unchanged.")
scripts/run/run_omip_core2.py-3349-    p.add_argument("--spmd-persistent-state", action="store_true",
scripts/run/run_omip_core2.py-3350-                   help="With --n-gpus > 1: keep the ocean state lat-band "
scripts/run/run_omip_core2.py:3351:                        "SHARDED across steps (make_sharded_ocean_step) instead "
scripts/run/run_omip_core2.py-3352-                        "of the global-in/global-out wrapper's full-state "
scripts/run/run_omip_core2.py-3353-                        "scatter+gather EVERY step (scaling-M2). Host post-step "
scripts/run/run_omip_core2.py-3354-                        "BCs run UNCHANGED: the leaf-wise host updates (SSS "
scripts/run/run_omip_core2.py-3355-                        "restore / ice-thermo / nudge / drag) read+write "
--
scripts/run/run_omip_core2.py-3931-                        "transport is parameterized tracer ADVECTION applied "
scripts/run/run_omip_core2.py-3932-                        "outside the pinned mass-flux block, so the model "
scripts/run/run_omip_core2.py-3933-                        "constructor rejects prescribed_flow+GM/Redi.")
scripts/run/run_omip_core2.py-3934-    p.add_argument("--scan-block", type=int, default=0,
scripts/run/run_omip_core2.py:3935:                   help="Issue #354: wrap the time loop in jax.lax.scan, "
scripts/run/run_omip_core2.py-3936-                        "fusing this many steps per block (CORE-II forcing "
scripts/run/run_omip_core2.py-3937-                        "sampled on-device, no per-step host roundtrip). "
scripts/run/run_omip_core2.py-3938-                        "0 (default) = the bit-identical Python loop. "
scripts/run/run_omip_core2.py-3939-                        "Tripole only; incompatible with --nudge-woa / "
--
scripts/run/run_omip_core2.py-4333-
scripts/run/run_omip_core2.py-4334-    # --prescribed-flow gates (PRE-BUILD, on the static args): grid support +
scripts/run/run_omip_core2.py-4335-    # the --spinup-drag rejection + the --no-gm-redi requirement.  NB: no
scripts/run/run_omip_core2.py-4336-    # --scan-block gate — the lever is in-model (inside _step_impl), so the
scripts/run/run_omip_core2.py:4337:    # lax.scan block path pins correctly.
scripts/run/run_omip_core2.py-4338-    validate_prescribed_flow_args(args.prescribed_flow, args.grid,
scripts/run/run_omip_core2.py-4339-                                  args.spinup_drag_tau_days,
scripts/run/run_omip_core2.py-4340-                                  no_gm_redi=args.no_gm_redi)
scripts/run/run_omip_core2.py-4341-
--
scripts/run/run_omip_core2.py-5464-    _log_diag_csv(0, 0.0, d0, 0.0, ice=ice_state)
scripts/run/run_omip_core2.py-5465-
scripts/run/run_omip_core2.py-5466-    # ------------------------------------------------------------------
scripts/run/run_omip_core2.py-5467-    # Multi-GPU lat-band SPMD step (--n-gpus N): partition the GLOBAL ocean state
scripts/run/run_omip_core2.py:5468:    # by latitude band across N local devices.  ``_ocean_step(state, sf, fw)`` is
scripts/run/run_omip_core2.py-5469-    # the single per-step entry the host loop calls; default (N=1) is the plain
scripts/run/run_omip_core2.py-5470-    # single-device model.step (byte-identical).  The global-in/global-out wrapper
scripts/run/run_omip_core2.py-5471-    # scatters/gathers each step, so the host post-step BCs (SSS restore /
scripts/run/run_omip_core2.py-5472-    # prognostic ice / geothermal / BBL / nudge / drag) operate on the gathered
--
scripts/run/run_omip_core2.py-5481-        # through untouched).
scripts/run/run_omip_core2.py-5482-        state = model.seed_tke(state)
scripts/run/run_omip_core2.py-5483-        # MPASOceanModel.step has no t_seconds (dm2dc, its only consumer, is
scripts/run/run_omip_core2.py-5484-        # arg-gated to tripole/latlon) -- passing it TypeErrors at step 1.
scripts/run/run_omip_core2.py:5485:        _ocean_step = (lambda st, sf, fw, t_sec=None:
scripts/run/run_omip_core2.py-5486-                       model.step(st, dt, surface_forcing=sf, freshwater=fw))
scripts/run/run_omip_core2.py-5487-    else:
scripts/run/run_omip_core2.py:5488:        _ocean_step = (lambda st, sf, fw, t_sec=None:
scripts/run/run_omip_core2.py-5489-                       model.step(st, dt, surface_forcing=sf, freshwater=fw,
scripts/run/run_omip_core2.py-5490-                                  t_seconds=t_sec))
scripts/run/run_omip_core2.py-5491-    # --spmd-persistent-state lane state (scaling-M2): OFF by default so the
scripts/run/run_omip_core2.py-5492-    # residency helpers below are no-ops and the loop is byte-identical.
--
scripts/run/run_omip_core2.py-5539-            raise SystemExit(
scripts/run/run_omip_core2.py-5540-                f"internal: padded n_lat ({n_lat_final}) not divisible by "
scripts/run/run_omip_core2.py-5541-                f"n_gpus ({args.n_gpus}) — the south-pad failed.")
scripts/run/run_omip_core2.py-5542-        from legoesm.parallel.mesh import create_latlon_mesh
scripts/run/run_omip_core2.py:5543:        from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip_core2.py:5544:            make_sharded_ocean_step_global,
scripts/run/run_omip_core2.py-5545-        )
scripts/run/run_omip_core2.py-5546-        # Prime the build-once vertex-mask cache from the concrete state BEFORE
scripts/run/run_omip_core2.py-5547-        # building the sharded step (the wrapper slices the primed global vmask
scripts/run/run_omip_core2.py-5548-        # per band; an unprimed cache raises in _build_band_vertex_masks).
--
scripts/run/run_omip_core2.py-5562-            # gather_state_latlon run only at the residency boundaries the
scripts/run/run_omip_core2.py-5563-            # helpers below manage (initial shard, snapshot/abort/final
scripts/run/run_omip_core2.py-5564-            # gathers, and the counted per-step gathers forced by host-global
scripts/run/run_omip_core2.py-5565-            # consumers).  t_sec is always None here (tide fail-fasts above).
scripts/run/run_omip_core2.py:5566:            from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip_core2.py-5567-                gather_state_latlon,
scripts/run/run_omip_core2.py:5568:                make_sharded_ocean_step,
scripts/run/run_omip_core2.py-5569-                shard_state_latlon,
scripts/run/run_omip_core2.py-5570-            )
scripts/run/run_omip_core2.py:5571:            _spmd_inner = make_sharded_ocean_step(model, _spmd_mesh)
scripts/run/run_omip_core2.py:5572:            _ocean_step = (lambda st, sf, fw, t_sec=None:
scripts/run/run_omip_core2.py:5573:                           _spmd_inner(st, dt, surface_forcing=sf,
scripts/run/run_omip_core2.py-5574-                                       freshwater=fw))
scripts/run/run_omip_core2.py-5575-
scripts/run/run_omip_core2.py-5576-            def _pers_shard_fn(st, _mesh=_spmd_mesh):
scripts/run/run_omip_core2.py-5577-                return shard_state_latlon(st, _mesh)
--
scripts/run/run_omip_core2.py-5585-                  f"rows/band); PERSISTENT sharded state "
scripts/run/run_omip_core2.py-5586-                  f"(--spmd-persistent-state): full-state gathers only at "
scripts/run/run_omip_core2.py-5587-                  f"snapshot/abort/final + counted per-step forcings.")
scripts/run/run_omip_core2.py-5588-        else:
scripts/run/run_omip_core2.py:5589:            _spmd_step = make_sharded_ocean_step_global(model, _spmd_mesh)
scripts/run/run_omip_core2.py-5590-            # t_sec is always None here (tide-enabled fail-fasts above).
scripts/run/run_omip_core2.py:5591:            _ocean_step = (lambda st, sf, fw, t_sec=None:
scripts/run/run_omip_core2.py:5592:                           _spmd_step(st, dt, surface_forcing=sf,
scripts/run/run_omip_core2.py-5593-                                      freshwater=fw))
scripts/run/run_omip_core2.py-5594-            print(f"[setup] multi-GPU lat-band SPMD: {args.n_gpus} devices, "
scripts/run/run_omip_core2.py-5595-                  f"n_lat={n_lat_final} ({n_lat_final // args.n_gpus} "
scripts/run/run_omip_core2.py-5596-                  f"rows/band); global-in/global-out wrapper (host BCs on "
--
scripts/run/run_omip_core2.py-5652-
scripts/run/run_omip_core2.py-5653-    t_wall = time.time()
scripts/run/run_omip_core2.py-5654-
scripts/run/run_omip_core2.py-5655-    # ------------------------------------------------------------------
scripts/run/run_omip_core2.py:5656:    # Issue #354: optional lax.scan block-stepping (tripole; no nudge/drag).
scripts/run/run_omip_core2.py-5657-    # The CORE-II forcing is sampled on-device (no per-step host roundtrip),
scripts/run/run_omip_core2.py-5658-    # so XLA fuses each block of ``--scan-block`` steps.  Default
scripts/run/run_omip_core2.py-5659-    # (--scan-block 0) keeps the bit-identical Python loop below.
scripts/run/run_omip_core2.py-5660-    # Diagnostics / snapshots / non-finite abort run at BLOCK BOUNDARIES.
--
scripts/run/run_omip_core2.py-5662-    _tti = getattr(getattr(model, "config", None),
scripts/run/run_omip_core2.py-5663-                   "tracer_time_integrator", "euler")
scripts/run/run_omip_core2.py-5664-    if args.n_gpus > 1 and int(args.scan_block) > 0:
scripts/run/run_omip_core2.py-5665-        raise SystemExit(
scripts/run/run_omip_core2.py:5666:            "--n-gpus > 1 and --scan-block are mutually exclusive: the lax.scan "
scripts/run/run_omip_core2.py-5667-            "block path fuses single-device on-device steps (it does not use the "
scripts/run/run_omip_core2.py-5668-            "lat-band sharded step). Pick one — multi-GPU SPMD (the host Python "
scripts/run/run_omip_core2.py-5669-            "loop, --scan-block 0) OR single-device scan fusion.")
scripts/run/run_omip_core2.py-5670-    # Equilibrium tide disqualifies the scan-block path: its body steps via
--
scripts/run/run_omip_core2.py-5679-                and not _tide_enabled
scripts/run/run_omip_core2.py-5680-                and _tti != "ab2")
scripts/run/run_omip_core2.py-5681-    if int(args.scan_block) > 0 and not use_scan:
scripts/run/run_omip_core2.py-5682-        why = ("AB2 tracer time integrator (None->Field carry breaks "
scripts/run/run_omip_core2.py:5683:               "lax.scan)" if _tti == "ab2"
scripts/run/run_omip_core2.py-5684-               else "tidal_forcing enabled (the scan body does not thread the "
scripts/run/run_omip_core2.py-5685-                    "model time the tide needs)" if _tide_enabled
scripts/run/run_omip_core2.py-5686-               else "grid!=tripole or WOA-nudging / spin-up-drag / SSS-restoring "
scripts/run/run_omip_core2.py-5687-                    "enabled (those need per-step host updates)")
--
scripts/run/run_omip_core2.py-5724-        # caches from the concrete state first (vertex-mask constant).
scripts/run/run_omip_core2.py-5725-        model.prime_step_caches(state)
scripts/run/run_omip_core2.py-5726-        block_fn = build_omip2_scan_block_fn(model, dt, gshape, ramp_s=ramp_s)
scripts/run/run_omip_core2.py-5727-        bsz = int(args.scan_block)
scripts/run/run_omip_core2.py:5728:        print(f"[run] lax.scan block-stepping: block<={bsz} steps, split at "
scripts/run/run_omip_core2.py-5729-              f"diag/snapshot/year boundaries so output cadence matches the "
scripts/run/run_omip_core2.py-5730-              f"Python loop (CORE-II forcing fused on-device)", flush=True)
scripts/run/run_omip_core2.py-5731-
scripts/run/run_omip_core2.py-5732-        def _block_steps(step):
--
scripts/run/run_omip_core2.py-6154-                print(f"[fwbudget] {app_grid_type}: P={_P:+.4f} E={_E:+.4f} "
scripts/run/run_omip_core2.py-6155-                      f"R={_Rn:+.4f} ice={_Ic:+.4f} net(P-E+R+ice)="
scripts/run/run_omip_core2.py-6156-                      f"{_P - _E + _Rn + _Ic:+.4f} Sv (raw pre-normalize, "
scripts/run/run_omip_core2.py-6157-                      f"area-wtd over wet)", flush=True)
scripts/run/run_omip_core2.py:6158:            # _ocean_step = single-device model.step (default), the lat-band
scripts/run/run_omip_core2.py-6159-            # SPMD global-in/global-out step (--n-gpus > 1), or the PERSISTENT
scripts/run/run_omip_core2.py-6160-            # sharded inner step (--spmd-persistent-state); all apply the
scripts/run/run_omip_core2.py-6161-            # in-core wind-stress / heat / freshwater forcing.  Default lanes
scripts/run/run_omip_core2.py-6162-            # return a GLOBAL state, so the host post-step BCs below are
--
scripts/run/run_omip_core2.py-6165-            # the residency-helper classification).  t_seconds threads the
scripts/run/run_omip_core2.py-6166-            # equilibrium-tide model time (None when tide off; the SPMD path
scripts/run/run_omip_core2.py-6167-            # fail-fasts at setup if the tide is enabled).
scripts/run/run_omip_core2.py-6168-            state = _ensure_sharded_state(state)
scripts/run/run_omip_core2.py:6169:            state = _ocean_step(state, sf, fw, _t_sec)
scripts/run/run_omip_core2.py-6170-        if sss_restore_cfg is not None:
scripts/run/run_omip_core2.py-6171-            # NEMO-faithful ice gate (namsbc_ssr nn_sssr_ice=0: no SSS restoring
scripts/run/run_omip_core2.py-6172-            # under sea ice).  Feed the SAME prescribed siconc the albedo uses
scripts/run/run_omip_core2.py-6173-            # (``_sic``; None only if neither --ice-albedo nor a siconc field is
--
scripts/run/run_omip.py-122-    # still disables it entirely).  Carried here so --params can reach
scripts/run/run_omip.py-123-    # GMRediConfig / VisbeckConfig / TreguierConfig (#691/#724).
scripts/run/run_omip.py-124-    gm_redi: GMRediConfig = _DEFAULT_BATHY_GM_REDI
scripts/run/run_omip.py-125-    # Lat-band SPMD (single-controller multi-GPU) for the lat-lon restoring
scripts/run/run_omip.py:126:    # lane: wraps the loop's dynamics step in ``make_sharded_ocean_step``
scripts/run/run_omip.py-127-    # (the #751/#758-validated lane-D step).  0 devices = all local.
scripts/run/run_omip.py-128-    enable_latlon_spmd: bool = False
scripts/run/run_omip.py-129-    spmd_n_devices: int = 0
scripts/run/run_omip.py-130-    # Route-B multicontroller (jax.distributed cross-process NCCL): promote the
--
scripts/run/run_omip.py-706-                   ))
scripts/run/run_omip.py-707-    p.add_argument("--enable-latlon-spmd", action="store_true", default=False,
scripts/run/run_omip.py-708-                   help=(
scripts/run/run_omip.py-709-                       "Run the lat-lon lane's dynamics step lat-band-SPMD "
scripts/run/run_omip.py:710:                       "across the local devices (make_sharded_ocean_step — "
scripts/run/run_omip.py-711-                       "the validated multi-GPU ocean lane). Supports the "
scripts/run/run_omip.py-712-                       "restoring lane AND the JRA55 block-scan lanes "
scripts/run/run_omip.py-713-                       "(forcing stacks are lat-band-sharded; the in-scan "
scripts/run/run_omip.py-714-                       "bulk fluxes stay shard-local). Single-controller "
--
scripts/run/run_omip.py-803-    p.add_argument("--gpu-interp", action="store_true", default=True,
scripts/run/run_omip.py-804-                   help=(
scripts/run/run_omip.py-805-                       "Move JRA55 forcing interpolation from CPU to GPU "
scripts/run/run_omip.py-806-                       "(DEFAULT, ~38%% faster). Loads only native 3-hourly "
scripts/run/run_omip.py:807:                       "records and interpolates inside the lax.scan body, "
scripts/run/run_omip.py-808-                       "reducing host-side I/O from ~288 to ~9 calls per "
scripts/run/run_omip.py-809-                       "day-block. Use --no-gpu-interp to disable."
scripts/run/run_omip.py-810-                   ))
scripts/run/run_omip.py-811-    p.add_argument("--no-gpu-interp", action="store_false", dest="gpu_interp",
--
scripts/run/run_omip.py-1982-# The plain Python time loop calling ``model.step`` once per step is
scripts/run/run_omip.py-1983-# correct but single-core-bound: XLA can't see across the loop, so the
scripts/run/run_omip.py-1984-# Eigen / BLAS threadpools don't get a useful work item per call at 1°
scripts/run/run_omip.py-1985-# resolution. The realistic-geometry GO continuation scripts wrap N
scripts/run/run_omip.py:1986:# steps in ``jax.lax.scan`` inside a single ``@jax.jit``; that gives
scripts/run/run_omip.py-1987-# XLA one big computation graph and 5–10× speedup at 1° on multi-core
scripts/run/run_omip.py-1988-# CPU.
scripts/run/run_omip.py-1989-#
scripts/run/run_omip.py:1990:# We can't put ``load_jra55_slice`` inside ``lax.scan`` (Zarr I/O is
scripts/run/run_omip.py-1991-# not a JAX op). Instead the outer Python layer pre-loads N steps of
scripts/run/run_omip.py-1992-# forcing into stacked JAX arrays once, then calls a JIT-compiled
scripts/run/run_omip.py-1993-# block function that scans through them.
scripts/run/run_omip.py-1994-
--
scripts/run/run_omip.py-2230-    simulated day has 288 timesteps.  The old CPU path called
scripts/run/run_omip.py-2231-    ``jra55_to_atm_surface`` 288 times per day in Python, spending
scripts/run/run_omip.py-2232-    ~1.9s/day on host-side I/O.  This path loads only the ~9 native
scripts/run/run_omip.py-2233-    records that bracket the block (~0.2s/day) and defers the linear
scripts/run/run_omip.py:2234:    interpolation + solar zenith to the JIT-compiled ``lax.scan`` body
scripts/run/run_omip.py-2235-    on GPU.  Result: 38% overall speedup (9.5x I/O reduction).
scripts/run/run_omip.py-2236-
scripts/run/run_omip.py-2237-    Returns ``(raw_stack, runoff_stack, record_meta)`` where:
scripts/run/run_omip.py-2238-    - ``raw_stack``: dict of ``(n_records, n_lat, n_lon)`` arrays for
--
scripts/run/run_omip.py-2299-
scripts/run/run_omip.py-2300-    return raw_stack, runoff_stack, record_meta
scripts/run/run_omip.py-2301-
scripts/run/run_omip.py-2302-
scripts/run/run_omip.py:2303:def _build_jra55_block_fn(model, jra55_state, dt, spmd_step=None):
scripts/run/run_omip.py:2304:    """Return a JIT-compiled block function that runs N steps via lax.scan.
scripts/run/run_omip.py-2305-
scripts/run/run_omip.py-2306-    Captures everything that's static across the block (sponge, SSS
scripts/run/run_omip.py-2307-    target, freeze-cap mask, coupler config, dt) in the closure so
scripts/run/run_omip.py-2308-    the scan body has a clean ``(state, idx) → (state', None)`` signature.
--
scripts/run/run_omip.py-2361-    # Lat-band SPMD (--enable-latlon-spmd): the scan body's dynamics step
scripts/run/run_omip.py-2362-    # runs through the sharded wrapper (same forcing kwargs as _step_impl;
scripts/run/run_omip.py-2363-    # the wrapper's cache/arm-restore Python runs ONCE at block trace).
scripts/run/run_omip.py-2364-    # Sea ice is refused upstream (the ice tile is not SPMD-audited yet).
scripts/run/run_omip.py:2365:    if spmd_step is not None and enable_sea_ice:
scripts/run/run_omip.py-2366-        raise ValueError(
scripts/run/run_omip.py:2367:            "spmd_step + prognostic sea ice is unsupported "
scripts/run/run_omip.py-2368-            "(run_omip_single refuses --jra55-sea-ice with "
scripts/run/run_omip.py-2369-            "--enable-latlon-spmd).")
scripts/run/run_omip.py:2370:    _dyn_step = (spmd_step if spmd_step is not None
scripts/run/run_omip.py-2371-                 else lambda st, d, **kw: model._step_impl(st, d, **kw))
scripts/run/run_omip.py-2372-
scripts/run/run_omip.py:2373:    @jax.jit
scripts/run/run_omip.py-2374-    def block_fn(state, atm_stack, runoff_stack, block_start_step,
scripts/run/run_omip.py-2375-                 ice_state=None):
scripts/run/run_omip.py-2376-        def step_body(carry, idx):
scripts/run/run_omip.py-2377-            if enable_sea_ice:
--
scripts/run/run_omip.py-2494-            return new_state, None
scripts/run/run_omip.py-2495-
scripts/run/run_omip.py-2496-        n = atm_stack["sw_down"].shape[0]
scripts/run/run_omip.py-2497-        init = (state, ice_state) if enable_sea_ice else state
scripts/run/run_omip.py:2498:        final, _ = jax.lax.scan(
scripts/run/run_omip.py-2499-            step_body, init, jnp.arange(n, dtype=jnp.int32),
scripts/run/run_omip.py-2500-        )
scripts/run/run_omip.py-2501-        return final  # (state, ice_state) when sea-ice on, else state
scripts/run/run_omip.py-2502-
scripts/run/run_omip.py-2503-    return block_fn
scripts/run/run_omip.py-2504-
scripts/run/run_omip.py-2505-
scripts/run/run_omip.py:2506:def _build_jra55_block_fn_interp(model, jra55_state, dt, spmd_step=None):
scripts/run/run_omip.py-2507-    """JIT-compiled block function with GPU-side forcing interpolation.
scripts/run/run_omip.py-2508-
scripts/run/run_omip.py-2509-    Like ``_build_jra55_block_fn``, but instead of receiving pre-
scripts/run/run_omip.py-2510-    interpolated per-step forcing, receives the native 3-hourly records
scripts/run/run_omip.py-2511-    and computes the linear interpolation + solar zenith inside the
scripts/run/run_omip.py:2512:    ``lax.scan`` body on GPU.  This reduces host-side I/O from N calls
scripts/run/run_omip.py-2513-    to ``jra55_to_atm_surface`` (N=288 for 1 day) down to ~9 Zarr reads
scripts/run/run_omip.py-2514-    per block.
scripts/run/run_omip.py-2515-    """
scripts/run/run_omip.py-2516-    from legoesm import constants as _const
--
scripts/run/run_omip.py-2561-    _maxvel_3d = model.config.barotropic.maxvel_barotropic
scripts/run/run_omip.py-2562-    enable_maxvel = _maxvel_3d > 0.0
scripts/run/run_omip.py-2563-
scripts/run/run_omip.py-2564-    # Lat-band SPMD: see _build_jra55_block_fn.
scripts/run/run_omip.py:2565:    if spmd_step is not None and enable_sea_ice:
scripts/run/run_omip.py-2566-        raise ValueError(
scripts/run/run_omip.py:2567:            "spmd_step + prognostic sea ice is unsupported "
scripts/run/run_omip.py-2568-            "(run_omip_single refuses --jra55-sea-ice with "
scripts/run/run_omip.py-2569-            "--enable-latlon-spmd).")
scripts/run/run_omip.py:2570:    _dyn_step = (spmd_step if spmd_step is not None
scripts/run/run_omip.py-2571-                 else lambda st, d, **kw: model._step_impl(st, d, **kw))
scripts/run/run_omip.py-2572-
scripts/run/run_omip.py-2573-    lat_2d = jra55_state["lat_2d"]
scripts/run/run_omip.py-2574-    lon_2d = jra55_state["lon_2d"]
--
scripts/run/run_omip.py-2578-    cycle = bool(jra55_state.get("cycle", False))
scripts/run/run_omip.py-2579-
scripts/run/run_omip.py-2580-    def _make_block_fn(n_steps_block):
scripts/run/run_omip.py-2581-        """Create a JIT-compiled block function for a fixed block size."""
scripts/run/run_omip.py:2582:        @jax.jit
scripts/run/run_omip.py-2583-        def block_fn(state, raw_stack, runoff_records, record_days,
scripts/run/run_omip.py-2584-                     block_start_day, block_start_day_forcing,
scripts/run/run_omip.py-2585-                     ice_state=None):
scripts/run/run_omip.py-2586-            dt_days = dt / 86400.0
--
scripts/run/run_omip.py-2754-                    return (new_state, new_ice), None
scripts/run/run_omip.py-2755-                return new_state, None
scripts/run/run_omip.py-2756-
scripts/run/run_omip.py-2757-            init = (state, ice_state) if enable_sea_ice else state
scripts/run/run_omip.py:2758:            final, _ = jax.lax.scan(
scripts/run/run_omip.py-2759-                step_body, init,
scripts/run/run_omip.py-2760-                jnp.arange(n_steps_block, dtype=jnp.int32),
scripts/run/run_omip.py-2761-            )
scripts/run/run_omip.py-2762-            return final  # (state, ice_state) when sea-ice on, else state
--
scripts/run/run_omip.py-3158-                   max_wallclock_seconds: float = 0.0,
scripts/run/run_omip.py-3159-                   restart_buffer_seconds: float = 600.0,
scripts/run/run_omip.py-3160-                   start_step=0,
scripts/run/run_omip.py-3161-                   nudge_woa_tau=0.0, T_woa_3d=None, S_woa_3d=None,
scripts/run/run_omip.py:3162:                   snapshot_fn=None, spmd_step=None, spmd_gather=None,
scripts/run/run_omip.py-3163-                   spmd_shard_stack=None):
scripts/run/run_omip.py-3164-    """Run time loop with diagnostics.
scripts/run/run_omip.py-3165-
scripts/run/run_omip.py-3166-    Two forcing paths, mutually exclusive:
scripts/run/run_omip.py-3167-
scripts/run/run_omip.py-3168-    * **restoring** (default): plain ``model.step(state, dt)`` followed
scripts/run/run_omip.py-3169-      by Haney SST/SSS restoring when ``restoring_targets`` is set.
scripts/run/run_omip.py:3170:      With ``spmd_step`` set (``--enable-latlon-spmd``), the dynamics
scripts/run/run_omip.py-3171-      step runs through that lat-band-SPMD callable instead — the state
scripts/run/run_omip.py-3172-      arrives sharded and every downstream op (restoring, finite checks,
scripts/run/run_omip.py-3173-      diagnostics, restart saves) works on the sharded global arrays
scripts/run/run_omip.py-3174-      transparently under the single-controller GSPMD runtime.
--
scripts/run/run_omip.py-3200-        raise ValueError(
scripts/run/run_omip.py-3201-            "_run_omip_loop: jra55_state and restoring_targets are mutually "
scripts/run/run_omip.py-3202-            "exclusive — choose one forcing path."
scripts/run/run_omip.py-3203-        )
scripts/run/run_omip.py:3204:    if spmd_step is not None and jra55_state is not None:
scripts/run/run_omip.py:3205:        # The block-scan lanes now thread spmd_step; the two unsupported
scripts/run/run_omip.py-3206-        # JRA sub-modes still refuse loudly.
scripts/run/run_omip.py-3207-        if jra55_state.get("_use_single_step", False):
scripts/run/run_omip.py-3208-            raise ValueError(
scripts/run/run_omip.py:3209:                "spmd_step + the JRA55 single-step fallback is unsupported "
scripts/run/run_omip.py-3210-                "(_jra55_step calls model.step directly); use the "
scripts/run/run_omip.py-3211-                "block-scan path (default).")
scripts/run/run_omip.py-3212-        if jra55_state.get("enable_sea_ice", False):
scripts/run/run_omip.py-3213-            raise ValueError(
scripts/run/run_omip.py:3214:                "spmd_step + prognostic sea ice is unsupported "
scripts/run/run_omip.py-3215-                "(--jra55-sea-ice; the ice tile is not SPMD-audited).")
scripts/run/run_omip.py-3216-    if checkpoint_days is not None and checkpoint_dir is None:
scripts/run/run_omip.py-3217-        raise ValueError(
scripts/run/run_omip.py-3218-            "_run_omip_loop: checkpoint_days requires checkpoint_dir."
--
scripts/run/run_omip.py-3345-    # middle once we have 3.
scripts/run/run_omip.py-3346-    eta_history: list[np.ndarray] = []
scripts/run/run_omip.py-3347-
scripts/run/run_omip.py-3348-    # ----- JRA55-do block-scan path (multi-core friendly) ----------------
scripts/run/run_omip.py:3349:    # Wraps N ocean steps in lax.scan inside @jax.jit for ~30x GPU
scripts/run/run_omip.py-3350-    # speedup.  The scan body calls model._step_impl() (no inner JIT)
scripts/run/run_omip.py-3351-    # to avoid nested JIT boundaries that caused divergence with
scripts/run/run_omip.py-3352-    # partial-cell coordinates.
scripts/run/run_omip.py-3353-    use_scan_blocks = (jra55_state is not None
--
scripts/run/run_omip.py-3376-        # a constant (codex review MAJOR; census 8474554).
scripts/run/run_omip.py-3377-        # SPMD: the caller (run_omip_single) already primed the caches from
scripts/run/run_omip.py-3378-        # the UNSHARDED state before sharding; re-priming here would run
scripts/run/run_omip.py-3379-        # np.asarray on the sharded ``state`` — a hard non-addressable error
scripts/run/run_omip.py:3380:        # under route-B (shards span processes).  Skip it when spmd_step is set.
scripts/run/run_omip.py:3381:        if spmd_step is None:
scripts/run/run_omip.py-3382-            model.prime_step_caches(state)
scripts/run/run_omip.py-3383-        # Prognostic slab sea ice (--jra55-sea-ice): the block scan carries
scripts/run/run_omip.py-3384-        # (ocean_state, ice_state); thread the ice state across blocks.
scripts/run/run_omip.py-3385-        _ice_on = bool(jra55_state.get("enable_sea_ice", False))
scripts/run/run_omip.py-3386-        ice_state = jra55_state.get("ice_state_init") if _ice_on else None
scripts/run/run_omip.py-3387-        if use_gpu_interp:
scripts/run/run_omip.py-3388-            _get_block_fn_interp = _build_jra55_block_fn_interp(
scripts/run/run_omip.py:3389:                model, jra55_state, dt, spmd_step=spmd_step)
scripts/run/run_omip.py-3390-            print("  GPU-interp mode: forcing interpolation on GPU")
scripts/run/run_omip.py-3391-            # Pre-load the full JRA55 cache for repeat-year runs to
scripts/run/run_omip.py-3392-            # eliminate per-block Zarr I/O (~0.3s/block → ~0s/block).
scripts/run/run_omip.py-3393-            _full_cache = None
--
scripts/run/run_omip.py-3396-                    jra55_state)
scripts/run/run_omip.py-3397-                _full_cache = (_fc_all, _fc_days, _fc_len)
scripts/run/run_omip.py-3398-        else:
scripts/run/run_omip.py-3399-            block_fn = _build_jra55_block_fn(model, jra55_state, dt,
scripts/run/run_omip.py:3400:                                             spmd_step=spmd_step)
scripts/run/run_omip.py-3401-            _full_cache = None
scripts/run/run_omip.py-3402-        block_size = max(1, diag_every)
scripts/run/run_omip.py-3403-        if checkpoint_days is not None:
scripts/run/run_omip.py-3404-            steps_per_ckpt = max(1, int(round(checkpoint_days * 86400.0 / dt)))
--
scripts/run/run_omip.py-3658-    # to 0.
scripts/run/run_omip.py-3659-    ramp_days = float(restoring_ramp_days)
scripts/run/run_omip.py-3660-    restoring_ramp_steps = max(1, int(ramp_days * 86400.0 / dt)) if ramp_days > 0 else 1
scripts/run/run_omip.py-3661-
scripts/run/run_omip.py:3662:    _dyn_step = spmd_step if spmd_step is not None else model.step
scripts/run/run_omip.py-3663-
scripts/run/run_omip.py-3664-    for i in range(start_step, n_steps):
scripts/run/run_omip.py-3665-        state = _dyn_step(state, dt)
scripts/run/run_omip.py-3666-
--
scripts/run/run_omip.py-4380-        model = LatLonCGridOceanModel(
scripts/run/run_omip.py-4381-            grid, z_coord_partial, config,
scripts/run/run_omip.py-4382-            iwm_forcing=getattr(model, "_iwm_forcing", None))
scripts/run/run_omip.py-4383-        # The scan body calls model._step_impl() (no inner JIT) so
scripts/run/run_omip.py:4384:        # partial-cell + lax.scan now works correctly.
scripts/run/run_omip.py-4385-
scripts/run/run_omip.py-4386-    # Initial state: from restart, WOA, or rest.
scripts/run/run_omip.py-4387-    start_step = 0
scripts/run/run_omip.py-4388-    state = _init_rest_state(
--
scripts/run/run_omip.py-4584-            z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
scripts/run/run_omip.py-4585-        )
scripts/run/run_omip.py-4586-        # Provide the ocean mask for global freeze-cap when no sponge.
scripts/run/run_omip.py-4587-        jra55_state["_ocean_mask_2d"] = state.land_mask.data > 0.5
scripts/run/run_omip.py:4588:        # GPU-interp path (default): interpolation inside the lax.scan
scripts/run/run_omip.py-4589-        # block for all grids (MPAS regridding is handled in
scripts/run/run_omip.py-4590-        # _preload_jra55_raw_records).  --no-gpu-interp routes to the
scripts/run/run_omip.py-4591-        # CPU-interp block path (_build_jra55_block_fn).
scripts/run/run_omip.py-4592-        jra55_state["_gpu_interp"] = bool(args.gpu_interp)
--
scripts/run/run_omip.py-4767-    # the validated multi-GPU sharded ocean step and shard the state.
scripts/run/run_omip.py-4768-    # Single-controller only; restoring lane only (the JRA55 block
scripts/run/run_omip.py-4769-    # functions call model._step_impl directly — follow-up).  Runs AFTER
scripts/run/run_omip.py-4770-    # the restart load so a resumed state is sharded too.
scripts/run/run_omip.py:4771:    spmd_step = None
scripts/run/run_omip.py-4772-    spmd_gather = None
scripts/run/run_omip.py-4773-    spmd_shard_stack = None
scripts/run/run_omip.py-4774-    if run_config.enable_latlon_spmd:
scripts/run/run_omip.py-4775-        if grid_type != "latlon":
--
scripts/run/run_omip.py-4808-                    f"divisible by the device count ({_nd}); pick "
scripts/run/run_omip.py-4809-                    f"--spmd-n-devices dividing n_lat.")
scripts/run/run_omip.py-4810-            from functools import partial
scripts/run/run_omip.py-4811-
scripts/run/run_omip.py:4812:            from legoesm.ocean.dynamics.sharded_ocean_step import (
scripts/run/run_omip.py-4813-                gather_state_latlon,
scripts/run/run_omip.py:4814:                make_sharded_ocean_step,
scripts/run/run_omip.py-4815-                shard_forcing_stack_latlon,
scripts/run/run_omip.py-4816-                shard_state_latlon,
scripts/run/run_omip.py-4817-            )
scripts/run/run_omip.py-4818-            from legoesm.parallel.mesh import create_latlon_mesh
scripts/run/run_omip.py-4819-            # Prime the build-once vertex-mask cache from the CONCRETE
scripts/run/run_omip.py-4820-            # state so the wrapper can build per-band masks host-side.
scripts/run/run_omip.py-4821-            model.prime_step_caches(state)
scripts/run/run_omip.py-4822-            _dev = create_latlon_mesh(n_devices=_nd)
scripts/run/run_omip.py:4823:            spmd_step = make_sharded_ocean_step(model, _dev.mesh)
scripts/run/run_omip.py-4824-            spmd_gather = partial(gather_state_latlon, mesh=_dev.mesh)
scripts/run/run_omip.py-4825-            state = shard_state_latlon(state, _dev.mesh)
scripts/run/run_omip.py-4826-            # Lay per-block forcing stacks out lat-band-sharded so the
scripts/run/run_omip.py-4827-            # in-scan interpolation / bulk fluxes stay shard-local (shared
--
scripts/run/run_omip.py-4855-            state.T.data.dtype) if args.nudge_woa_tau > 0 and T_woa is not None else None,
scripts/run/run_omip.py-4856-        S_woa_3d=(S_woa * state.land_mask.data[..., jnp.newaxis]).astype(
scripts/run/run_omip.py-4857-            state.S.data.dtype) if args.nudge_woa_tau > 0 and S_woa is not None else None,
scripts/run/run_omip.py-4858-        snapshot_fn=_snapshot_fn,
scripts/run/run_omip.py:4859:        spmd_step=spmd_step,
scripts/run/run_omip.py-4860-        spmd_gather=spmd_gather,
scripts/run/run_omip.py-4861-        spmd_shard_stack=spmd_shard_stack,
scripts/run/run_omip.py-4862-    )
scripts/run/run_omip.py-4863-    if spmd_gather is not None:
--
scripts/bench/bench_ocean_mpas_scaling.py-21-the shared self-describing metadata (transport resolves to mpi4jax via
scripts/bench/bench_ocean_mpas_scaling.py-22-n_ranks > process_count) + voronoi partition-quality metrics.
scripts/bench/bench_ocean_mpas_scaling.py-23-
scripts/bench/bench_ocean_mpas_scaling.py-24-M1 measurement contract (scaling-M3d increment-1): the headline number is a
scripts/bench/bench_ocean_mpas_scaling.py:25:fused ``lax.scan`` block (``metadata.timed_scan_blocks``; per-step
scripts/bench/bench_ocean_mpas_scaling.py-26-dispatch latency probed SEPARATELY), cross-rank MAX-reduced, and it is what
scripts/bench/bench_ocean_mpas_scaling.py-27-the aggregator-facing ``steady_median_ms`` carries (the same deliberate
scripts/bench/bench_ocean_mpas_scaling.py-28-naming as ``bench_ocean_latlon_spmd_scaling``); the host-synced gate-loop
scripts/bench/bench_ocean_mpas_scaling.py-29-median is dispatch+sync LATENCY and is recorded only under
--
scripts/bench/bench_ocean_mpas_scaling.py-66-
scripts/bench/bench_ocean_mpas_scaling.py-67-from metadata import (  # noqa: E402
scripts/bench/bench_ocean_mpas_scaling.py-68-    annotate_incomplete,
scripts/bench/bench_ocean_mpas_scaling.py-69-    scaling_metadata,
scripts/bench/bench_ocean_mpas_scaling.py:70:    timed_scan_blocks,
scripts/bench/bench_ocean_mpas_scaling.py-71-    wet_cell_metrics,
scripts/bench/bench_ocean_mpas_scaling.py-72-)
scripts/bench/bench_ocean_mpas_scaling.py-73-
scripts/bench/bench_ocean_mpas_scaling.py-74-#: Parity tolerances (gathered MPI vs serial, f64/f32) — the re-association
--
scripts/bench/bench_ocean_mpas_scaling.py-339-                        "across the window; timing omits exchange cost; "
scripts/bench/bench_ocean_mpas_scaling.py-340-                        "the zero-forcing residual probe still refreshes "
scripts/bench/bench_ocean_mpas_scaling.py-341-                        "ONCE before probing).  See the stage audit doc.")
scripts/bench/bench_ocean_mpas_scaling.py-342-    p.add_argument("--block-steps", type=int, default=8,
scripts/bench/bench_ocean_mpas_scaling.py:343:                   help="Steps per fused lax.scan timing block (M1 "
scripts/bench/bench_ocean_mpas_scaling.py-344-                        "contract; runs AFTER the gates). 0 disables the "
scripts/bench/bench_ocean_mpas_scaling.py-345-                        "fused measurement.")
scripts/bench/bench_ocean_mpas_scaling.py-346-    p.add_argument("--blocks", type=int, default=2,
scripts/bench/bench_ocean_mpas_scaling.py-347-                   help="Number of fused timing blocks.")
--
scripts/bench/bench_ocean_mpas_scaling.py-526-
scripts/bench/bench_ocean_mpas_scaling.py-527-        # jit the COMPOSED step+exchange (one dispatch per step; the
scripts/bench/bench_ocean_mpas_scaling.py-528-        # sendrecv wrapper always runs traced, exactly as the atmosphere
scripts/bench/bench_ocean_mpas_scaling.py-529-        # step and the distributed PCG use it).  ``_step_impl`` avoids a
scripts/bench/bench_ocean_mpas_scaling.py:530:        # nested-JIT boundary inside this wrapper; timed_scan_blocks
scripts/bench/bench_ocean_mpas_scaling.py-531-        # re-traces the whole thing into the fused scan — same graph.
scripts/bench/bench_ocean_mpas_scaling.py:532:        @jax.jit
scripts/bench/bench_ocean_mpas_scaling.py-533-        def advance(st):
scripts/bench/bench_ocean_mpas_scaling.py-534-            return exchange_state_mpas_ocean(
scripts/bench/bench_ocean_mpas_scaling.py-535-                model._step_impl(st, args.dt,
scripts/bench/bench_ocean_mpas_scaling.py-536-                                 halo_refresh=_in_step_refresh),
--
scripts/bench/bench_ocean_mpas_scaling.py-638-                      "reference.", flush=True)
scripts/bench/bench_ocean_mpas_scaling.py-639-            return 5
scripts/bench/bench_ocean_mpas_scaling.py-640-
scripts/bench/bench_ocean_mpas_scaling.py-641-    # --- M1 fused-scan measurement (AFTER the gates; timing-only) ----------
scripts/bench/bench_ocean_mpas_scaling.py:642:    # The trustworthy production-like number: multi-step lax.scan blocks
scripts/bench/bench_ocean_mpas_scaling.py-643-    # with sync only AROUND the block (the per-step host-synced loop above
scripts/bench/bench_ocean_mpas_scaling.py-644-    # measures dispatch+sync latency — kept for the gates and legacy
scripts/bench/bench_ocean_mpas_scaling.py-645-    # comparability, never as the fused headline).  Route-A: mpi4jax
scripts/bench/bench_ocean_mpas_scaling.py-646-    # collectives inside the scan keep ranks in lockstep (the distributed
--
scripts/bench/bench_ocean_mpas_scaling.py-648-    # allgathered so the parallel time of block b is the SLOWEST rank in
scripts/bench/bench_ocean_mpas_scaling.py-649-    # that block (the same slowest-rank convention as the per-step MAX).
scripts/bench/bench_ocean_mpas_scaling.py-650-    fused = None
scripts/bench/bench_ocean_mpas_scaling.py-651-    if args.block_steps > 0:
scripts/bench/bench_ocean_mpas_scaling.py:652:        state, _t = timed_scan_blocks(
scripts/bench/bench_ocean_mpas_scaling.py-653-            advance, state,
scripts/bench/bench_ocean_mpas_scaling.py-654-            block_steps=args.block_steps, n_blocks=args.blocks,
scripts/bench/bench_ocean_mpas_scaling.py-655-            probe_steps=args.probe_steps,
scripts/bench/bench_ocean_mpas_scaling.py-656-            sync_label="ocean_mpas_mpi_bench")

codex
The bench path is wired correctly, and existing one-argument `timed_scan_blocks` callers remain compatible. I’m now checking the JITted OMIP block paths, where the same closure-capture failure can recur outside the benchmark.
exec
/usr/bin/bash -lc "sed -n '2280,2520p' scripts/run/run_omip.py
sed -n '2500,2790p' scripts/run/run_omip.py
sed -n '3325,3465p' scripts/run/run_omip.py
sed -n '3635,3685p' scripts/run/run_omip.py
sed -n '5555,5605p' scripts/run/run_omip_core2.py
rg -n -C 5 \"make_sharded_ocean_step_global|make_sharded_ocean_step\\(\" scripts packages tests --glob '"'!scripts/run/run_omip.py'"' --glob '"'!scripts/run/run_omip_core2.py'"'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 718ms:
        from legoesm.grids.regridding import regrid_scalar
        rw = jra55_state["regrid_weights"]
        for var in raw_stack:
            raw_stack[var] = jnp.stack([
                regrid_scalar(raw_stack[var][i], rw)
                for i in range(raw_stack[var].shape[0])
            ])

    runoff_stack = raw_stack["friver"]

    record_meta = {
        "record_days": record_days,          # (n_records,) fractional days
        "block_start_day": float(start_day),
        "block_start_day_forcing": float(start_day_f),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "cache_length_days": float(cache_length_days),
        "cycle": cycle,
    }

    return raw_stack, runoff_stack, record_meta


def _build_jra55_block_fn(model, jra55_state, dt, spmd_step=None):
    """Return a JIT-compiled block function that runs N steps via lax.scan.

    Captures everything that's static across the block (sponge, SSS
    target, freeze-cap mask, coupler config, dt) in the closure so
    the scan body has a clean ``(state, idx) → (state', None)`` signature.
    Re-using the returned function across blocks reuses the JIT cache.
    """
    from legoesm import constants as _const
    from legoesm.coupler.coupler import ocean_tile_response
    from legoesm.coupler.ocean_forcing import omip_sea_ice_surface_forcing
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing

    coupler_cfg = jra55_state["coupler_cfg"]
    co2_ppmv = float(jra55_state["co2_ppmv"])
    T_ramp_seconds = float(jra55_state.get("T_ramp_seconds", 86400.0))
    enable_ramp = T_ramp_seconds > 0
    enable_sponge = bool(jra55_state.get("enable_sponge", False))
    enable_sss = bool(jra55_state.get("enable_sss_restoring", False))
    enable_freeze = bool(jra55_state.get("enable_freeze_cap", False))
    # Prognostic slab sea ice (opt-in --jra55-sea-ice): replaces the freeze-cap
    # SST stand-in.  Static gate ⇒ ice-off blocks are bit-identical.  Setup
    # forces enable_freeze_cap=False when ice is on (no double-capping).
    enable_sea_ice = bool(jra55_state.get("enable_sea_ice", False))
    ice_cfg = jra55_state.get("ice_config")

    sponge = _build_sponge_forcing(jra55_state) if enable_sponge else None

    if enable_sss:
        sss_pv = float(jra55_state["sss_piston_velocity"])
        sss_dz = float(jra55_state["dz_top"])
        sss_alpha_static = sss_pv * dt / max(sss_dz, 1e-6)
        sss_target_static = jra55_state["sss_target_2d"]
    else:
        sss_alpha_static = 0.0
        sss_target_static = None

    if enable_freeze:
        sponge_gamma = jra55_state.get("sponge_gamma_2d", None)
        if sponge_gamma is not None and np.any(np.asarray(sponge_gamma) > 0):
            freeze_mask_static = sponge_gamma > 0.0
        else:
            # No sponge: cap globally over all ocean cells.
            # Use the land_mask from the initial state (captured below).
            freeze_mask_static = jra55_state.get("_ocean_mask_2d", None)
        T_freeze_C_static = float(jra55_state["T_freeze_ocean_C"])
    else:
        freeze_mask_static = None
        T_freeze_C_static = -1.8

    # 3D velocity clip — caps ALL velocity components (barotropic +
    # baroclinic) after each step.  The barotropic-only MAXVEL inside
    # the split-explicit solver doesn't prevent baroclinic blowup.
    _maxvel_3d = model.config.barotropic.maxvel_barotropic
    enable_maxvel = _maxvel_3d > 0.0

    # Lat-band SPMD (--enable-latlon-spmd): the scan body's dynamics step
    # runs through the sharded wrapper (same forcing kwargs as _step_impl;
    # the wrapper's cache/arm-restore Python runs ONCE at block trace).
    # Sea ice is refused upstream (the ice tile is not SPMD-audited yet).
    if spmd_step is not None and enable_sea_ice:
        raise ValueError(
            "spmd_step + prognostic sea ice is unsupported "
            "(run_omip_single refuses --jra55-sea-ice with "
            "--enable-latlon-spmd).")
    _dyn_step = (spmd_step if spmd_step is not None
                 else lambda st, d, **kw: model._step_impl(st, d, **kw))

    @jax.jit
    def block_fn(state, atm_stack, runoff_stack, block_start_step,
                 ice_state=None):
        def step_body(carry, idx):
            if enable_sea_ice:
                state_in, ice_in = carry
            else:
                state_in = carry
            atm = AtmToSurface(
                sw_down=atm_stack["sw_down"][idx],
                lw_down=atm_stack["lw_down"][idx],
                precip_total=atm_stack["precip_total"][idx],
                precip_snow=atm_stack["precip_snow"][idx],
                T_lowest=atm_stack["T_lowest"][idx],
                q_lowest=atm_stack["q_lowest"][idx],
                u_lowest=atm_stack["u_lowest"][idx],
                v_lowest=atm_stack["v_lowest"][idx],
                p_lowest=atm_stack["p_lowest"][idx],
                p_surface=atm_stack["p_surface"][idx],
                rho_lowest=atm_stack["rho_lowest"][idx],
                cos_zenith=atm_stack["cos_zenith"][idx],
                co2_ppmv=jnp.asarray(co2_ppmv, dtype=atm_stack["T_lowest"].dtype),
                has_radiation=jnp.asarray(1.0, dtype=atm_stack["T_lowest"].dtype),
                has_precipitation=jnp.asarray(1.0, dtype=atm_stack["T_lowest"].dtype),
            )

            sst_K = state_in.T.data[..., 0] + _const.T_freeze
            u_o = jnp.zeros_like(sst_K)
            v_o = jnp.zeros_like(sst_K)
            tile = ocean_tile_response(atm, sst_K, u_o, v_o, coupler_cfg)

            # Wind-stress spinup ramp (gated at compile time).
            if enable_ramp:
                abs_step = block_start_step + idx
                t_sim = abs_step.astype(jnp.float64) * dt
                ramp = jnp.minimum(1.0, t_sim / T_ramp_seconds)
                tau_x = tile.tau_x * ramp
                tau_y = tile.tau_y * ramp
            else:
                tau_x = tile.tau_x
                tau_y = tile.tau_y

            sw_net = atm.sw_down * (1.0 - tile.albedo)
            q_net = (sw_net + atm.lw_down
                     - tile.lw_up - tile.shflx - tile.lhflx)

            evap = tile.lhflx / _const.L_v
            fw = FreshwaterForcing(
                precip=atm.precip_total,
                evap=evap,
                runoff=runoff_stack[idx],
                ice_fw=jnp.zeros_like(runoff_stack[idx]),
            )
            sf = OceanSurfaceForcing(
                sw_down=atm.sw_down,
                q_net=q_net,
                tau_x=tau_x,
                tau_y=tau_y,
                freshwater=None,
            )
            # Prognostic slab sea ice: advance the ice tile and partition the
            # surface forcing (open-ocean fluxes x f_ocean=(1-A) + the ice
            # tile's basal heat / melt-freeze freshwater / brine salt / stress).
            # ocean_mask: land cells receive no ice->ocean forcing (mask-aware
            # blend contract; land_mask is scan-carry state, traced-safe).
            if enable_sea_ice:
                new_ice, fw, sf = omip_sea_ice_surface_forcing(
                    ice_state=ice_in, ice_config=ice_cfg, atm=atm,
                    ocean_sst_K=sst_K, open_ocean_sf=sf, open_ocean_fw=fw,
                    dt=dt, grid=None,
                    ocean_mask=state_in.land_mask.data,
                )
            # Ramp sponge strength alongside wind stress.
            if enable_ramp and enable_sponge:
                sponge_step = sponge._replace(gamma=sponge.gamma * ramp)
            else:
                sponge_step = sponge

            new_state = _dyn_step(
                state_in, dt,
                freshwater=fw, surface_forcing=sf, sponge=sponge_step,
            )

            # SSS restoring (gated at compile time via Python `if`).
            if enable_sss:
                S = new_state.S.data
                target = jnp.asarray(sss_target_static, dtype=S.dtype)
                alpha = jnp.asarray(sss_alpha_static, dtype=S.dtype)
                mask = jnp.asarray(new_state.land_mask.data, dtype=S.dtype)
                S_top_new = (
                    S[..., 0] - alpha * (S[..., 0] - target) * mask
                )
                new_state = new_state._replace(
                    S=new_state.S.replace(data=S.at[..., 0].set(S_top_new)),
                )

            # T_freeze cap inside sponge.
            if enable_freeze:
                T = new_state.T.data
                T_freeze_C = jnp.asarray(T_freeze_C_static, dtype=T.dtype)
                T_top = T[..., 0]
                T_top_capped = jnp.where(
                    freeze_mask_static,
                    jnp.maximum(T_top, T_freeze_C),
                    T_top,
                )
                new_state = new_state._replace(
                    T=new_state.T.replace(data=T.at[..., 0].set(T_top_capped)),
                )

            # 3D velocity clip (MOM6 MAXVEL analog for full field).
            if enable_maxvel:
                u_clipped = jnp.clip(new_state.u.data, -_maxvel_3d, _maxvel_3d)
                v_clipped = jnp.clip(new_state.v.data, -_maxvel_3d, _maxvel_3d)
                new_state = new_state._replace(
                    u=new_state.u.replace(data=u_clipped),
                    v=new_state.v.replace(data=v_clipped),
                )

            if enable_sea_ice:
                return (new_state, new_ice), None
            return new_state, None

        n = atm_stack["sw_down"].shape[0]
        init = (state, ice_state) if enable_sea_ice else state
        final, _ = jax.lax.scan(
            step_body, init, jnp.arange(n, dtype=jnp.int32),
        )
        return final  # (state, ice_state) when sea-ice on, else state

    return block_fn


def _build_jra55_block_fn_interp(model, jra55_state, dt, spmd_step=None):
    """JIT-compiled block function with GPU-side forcing interpolation.

    Like ``_build_jra55_block_fn``, but instead of receiving pre-
    interpolated per-step forcing, receives the native 3-hourly records
    and computes the linear interpolation + solar zenith inside the
    ``lax.scan`` body on GPU.  This reduces host-side I/O from N calls
    to ``jra55_to_atm_surface`` (N=288 for 1 day) down to ~9 Zarr reads
    per block.
    """
    from legoesm import constants as _const
    from legoesm.coupler.coupler import ocean_tile_response
    from legoesm.coupler.ocean_forcing import omip_sea_ice_surface_forcing
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.ocean.freshwater import FreshwaterForcing
        )
        return final  # (state, ice_state) when sea-ice on, else state

    return block_fn


def _build_jra55_block_fn_interp(model, jra55_state, dt, spmd_step=None):
    """JIT-compiled block function with GPU-side forcing interpolation.

    Like ``_build_jra55_block_fn``, but instead of receiving pre-
    interpolated per-step forcing, receives the native 3-hourly records
    and computes the linear interpolation + solar zenith inside the
    ``lax.scan`` body on GPU.  This reduces host-side I/O from N calls
    to ``jra55_to_atm_surface`` (N=288 for 1 day) down to ~9 Zarr reads
    per block.
    """
    from legoesm import constants as _const
    from legoesm.coupler.coupler import ocean_tile_response
    from legoesm.coupler.ocean_forcing import omip_sea_ice_surface_forcing
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.ocean.freshwater import FreshwaterForcing
    from legoesm.ocean.state import OceanSurfaceForcing
    from legoesm.atmosphere.physics.radiation.solar import (
        cos_zenith_angle, solar_declination,
    )
    from legoesm.forcing.jra55_do import RECORDS_PER_DAY

    coupler_cfg = jra55_state["coupler_cfg"]
    co2_ppmv = float(jra55_state["co2_ppmv"])
    T_ramp_seconds = float(jra55_state.get("T_ramp_seconds", 86400.0))
    enable_ramp = T_ramp_seconds > 0
    enable_sponge = bool(jra55_state.get("enable_sponge", False))
    enable_sss = bool(jra55_state.get("enable_sss_restoring", False))
    enable_freeze = bool(jra55_state.get("enable_freeze_cap", False))

    sponge = _build_sponge_forcing(jra55_state) if enable_sponge else None

    if enable_sss:
        sss_pv = float(jra55_state["sss_piston_velocity"])
        sss_dz = float(jra55_state["dz_top"])
        sss_alpha_static = sss_pv * dt / max(sss_dz, 1e-6)
        sss_target_static = jra55_state["sss_target_2d"]
    else:
        sss_alpha_static = 0.0
        sss_target_static = None

    if enable_freeze:
        sponge_gamma = jra55_state.get("sponge_gamma_2d", None)
        if sponge_gamma is not None and np.any(np.asarray(sponge_gamma) > 0):
            freeze_mask_static = sponge_gamma > 0.0
        else:
            freeze_mask_static = jra55_state.get("_ocean_mask_2d", None)
        T_freeze_C_static = float(jra55_state["T_freeze_ocean_C"])
    else:
        freeze_mask_static = None
        T_freeze_C_static = -1.8

    # Prognostic slab sea ice (opt-in) — see _build_jra55_block_fn.
    enable_sea_ice = bool(jra55_state.get("enable_sea_ice", False))
    ice_cfg = jra55_state.get("ice_config")

    _maxvel_3d = model.config.barotropic.maxvel_barotropic
    enable_maxvel = _maxvel_3d > 0.0

    # Lat-band SPMD: see _build_jra55_block_fn.
    if spmd_step is not None and enable_sea_ice:
        raise ValueError(
            "spmd_step + prognostic sea ice is unsupported "
            "(run_omip_single refuses --jra55-sea-ice with "
            "--enable-latlon-spmd).")
    _dyn_step = (spmd_step if spmd_step is not None
                 else lambda st, d, **kw: model._step_impl(st, d, **kw))

    lat_2d = jra55_state["lat_2d"]
    lon_2d = jra55_state["lon_2d"]
    _rpd = float(RECORDS_PER_DAY)
    # Static (compile-time) repeat-year-forcing flag.  Selects which clock
    # drives the solar-zenith insolation geometry (see the scan body).
    cycle = bool(jra55_state.get("cycle", False))

    def _make_block_fn(n_steps_block):
        """Create a JIT-compiled block function for a fixed block size."""
        @jax.jit
        def block_fn(state, raw_stack, runoff_records, record_days,
                     block_start_day, block_start_day_forcing,
                     ice_state=None):
            dt_days = dt / 86400.0

            def step_body(carry, idx):
                if enable_sea_ice:
                    state_in, ice_in = carry
                else:
                    state_in = carry
                # Two clocks (see the insolation + ramp notes below):
                # - ``day``: RAW simulation day (elapsed run time).  Always
                #   drives the spinup ramp; drives the solar-zenith clock only
                #   when NOT cycling (cycle=False ⇒ day_f == day).
                # - ``day_f``: forcing clock — the CYCLED block-start day
                #   (aligned with ``record_days``, which the preloaders
                #   unwrap across the repeat-year cache boundary).  Drives the
                #   JRA55 record interpolation, and — when cycle=True — the
                #   solar-zenith insolation clock, so the prescribed rsds and
                #   the computed zenith stay phase-locked.  Using the raw day
                #   for interpolation broke every cycle after the first:
                #   ``day - record_days[0]`` was off by k*cache_length,
                #   i_lo clipped to the last slice record, and each step
                #   read one stale record.
                day = block_start_day + idx * dt_days
                day_f = block_start_day_forcing + idx * dt_days

                # Find bracketing records: record_days is sorted,
                # find floor position relative to the first record.
                local_pos = day_f * _rpd - record_days[0] * _rpd
                i_lo = jnp.clip(
                    jnp.floor(local_pos).astype(jnp.int32),
                    0, record_days.shape[0] - 2,
                )
                i_hi = i_lo + 1
                day_lo = record_days[i_lo]
                day_hi = record_days[i_hi]
                alpha = jnp.clip(
                    jnp.where(day_hi > day_lo,
                              (day_f - day_lo) / (day_hi - day_lo), 0.0),
                    0.0, 1.0,
                )

                def _interp(arr):
                    return (1.0 - alpha) * arr[i_lo] + alpha * arr[i_hi]

                rsds = _interp(raw_stack["rsds"])
                rlds = _interp(raw_stack["rlds"])
                tas = _interp(raw_stack["tas"])
                huss = _interp(raw_stack["huss"])
                uas = _interp(raw_stack["uas"])
                vas = _interp(raw_stack["vas"])
                psl = _interp(raw_stack["psl"])
                prra = _interp(raw_stack["prra"])
                prsn = _interp(raw_stack["prsn"])
                friver = _interp(runoff_records)

                # Derived: virtual-T density + solar zenith. Canonical coefficient
                # 1/epsilon - 1 (~0.608), not the rounded 0.61 (~0.4% drift) — must
                # match jra55_to_atm_surface / _shared.virtual_temperature.
                T_v = tas * (1.0 + (1.0 / _const.epsilon - 1.0) * huss)
                rho_a = psl / (_const.R_d * T_v)
                # Insolation clock (day-of-year + diurnal hour), which sets the
                # solar zenith and hence zenith-dependent surface albedo:
                #   - cycle=True (repeat-year forcing): use the CYCLED forcing
                #     clock ``day_f`` so the solar geometry stays phase-locked
                #     to the repeated rsds/rlds records.  Using the RAW ``day``
                #     drifts the seasonal doy (and, for a non-integer cache
                #     length, the diurnal hour) whenever the cache length is
                #     not a whole multiple of 365 days (e.g. a 366-day
                #     leap-year RYF cache), biasing the surface albedo.
                #   - cycle=False: ``day_f == day`` (no wrap), so this is
                #     byte-identical to the raw-day clock — the common
                #     non-cycled path is unchanged.
                # The SPINUP RAMP (below) intentionally stays on the RAW
                # ``day``: it is a function of elapsed run time, not forcing
                # time.  ``cycle`` is a static Python bool (compile-time
                # feature gate), so this branch is resolved at trace time.
                day_insol = day_f if cycle else day
                doy = jnp.mod(day_insol, 365.0) + 1.0
                hour = jnp.mod(day_insol, 1.0) * 24.0
                cos_z = cos_zenith_angle(lat_2d, lon_2d, doy, hour)

                atm = AtmToSurface(
                    sw_down=rsds, lw_down=rlds,
                    precip_total=prra + prsn, precip_snow=prsn,
                    T_lowest=tas, q_lowest=huss,
                    u_lowest=uas, v_lowest=vas,
                    p_lowest=psl, p_surface=psl,
                    rho_lowest=rho_a, cos_zenith=cos_z,
                    co2_ppmv=jnp.asarray(co2_ppmv, dtype=tas.dtype),
                    has_radiation=jnp.asarray(1.0, dtype=tas.dtype),
                    has_precipitation=jnp.asarray(1.0, dtype=tas.dtype),
                )

                sst_K = state_in.T.data[..., 0] + _const.T_freeze
                u_o = jnp.zeros_like(sst_K)
                v_o = jnp.zeros_like(sst_K)
                tile = ocean_tile_response(atm, sst_K, u_o, v_o, coupler_cfg)

                if enable_ramp:
                    t_sim = day * 86400.0
                    ramp = jnp.minimum(1.0, t_sim / T_ramp_seconds)
                else:
                    ramp = 1.0

                sw_net = atm.sw_down * (1.0 - tile.albedo)
                q_net = (sw_net + atm.lw_down - tile.lw_up
                         - tile.shflx - tile.lhflx)

                _dtype = state_in.T.data.dtype
                E_rate = tile.lhflx / jnp.asarray(_const.L_v, dtype=_dtype)
                fw = FreshwaterForcing(
                    precip=jnp.asarray(prra + prsn, dtype=_dtype),
                    evap=jnp.asarray(E_rate, dtype=_dtype),
                    runoff=jnp.asarray(friver, dtype=_dtype),
                    ice_fw=jnp.zeros_like(sst_K, dtype=_dtype),
                )
                sf = OceanSurfaceForcing(
                    sw_down=atm.sw_down, q_net=q_net,
                    tau_x=tile.tau_x * ramp, tau_y=tile.tau_y * ramp,
                    freshwater=None,
                )

                # Prognostic slab sea ice: partition surface forcing between
                # open ocean (f_ocean=1-A) and the ice tile.  ocean_mask: land
                # cells receive no ice->ocean forcing (mask-aware blend
                # contract; land_mask is scan-carry state, traced-safe).
                if enable_sea_ice:
                    new_ice, fw, sf = omip_sea_ice_surface_forcing(
                        ice_state=ice_in, ice_config=ice_cfg, atm=atm,
                        ocean_sst_K=sst_K, open_ocean_sf=sf, open_ocean_fw=fw,
                        dt=dt, grid=None,
                        ocean_mask=state_in.land_mask.data,
                    )

                sponge_k = (sponge._replace(gamma=sponge.gamma * ramp)
                            if enable_sponge else None)
                new_state = _dyn_step(
                    state_in, dt, freshwater=fw,
                    surface_forcing=sf, sponge=sponge_k,
                )

                if enable_sss:
                    S = new_state.S.data
                    _sss_mask = new_state.land_mask.data
                    S_new = S.at[..., 0].set(
                        S[..., 0] - sss_alpha_static * (
                            S[..., 0] - sss_target_static) * _sss_mask)
                    new_state = new_state._replace(
                        S=new_state.S.replace(data=S_new))
                if enable_freeze:
                    T = new_state.T.data
                    T_top = jnp.where(
                        freeze_mask_static,
                        jnp.maximum(T[..., 0], T_freeze_C_static),
                        T[..., 0],
                    )
                    new_state = new_state._replace(
                        T=new_state.T.replace(
                            data=T.at[..., 0].set(T_top)))
                if enable_maxvel:
                    new_state = new_state._replace(
                        u=new_state.u.replace(
                            data=jnp.clip(new_state.u.data,
                                          -_maxvel_3d, _maxvel_3d)),
                        v=new_state.v.replace(
                            data=jnp.clip(new_state.v.data,
                                          -_maxvel_3d, _maxvel_3d)))

                if enable_sea_ice:
                    return (new_state, new_ice), None
                return new_state, None

            init = (state, ice_state) if enable_sea_ice else state
            final, _ = jax.lax.scan(
                step_body, init,
                jnp.arange(n_steps_block, dtype=jnp.int32),
            )
            return final  # (state, ice_state) when sea-ice on, else state

        return block_fn

    # Cache block functions by size to avoid recompilation.
    _block_fn_cache = {}

    def _get_block_fn(n):
        if n not in _block_fn_cache:
            _block_fn_cache[n] = _make_block_fn(n)
        return _block_fn_cache[n]

    return _get_block_fn


# ===========================================================================
# Diagnostics
# ===========================================================================

def _extract_scalars(state, grid_type, grid, z_coord):
    """Compute scalar diagnostics from ocean state."""
    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis_3d, sh_synthesis
        T_grid = sh_synthesis_3d(grid, state.T_hat.data).real
        S_grid = sh_synthesis_3d(grid, state.S_hat.data).real
        eta_grid = sh_synthesis(grid, state.eta_hat.data).real
        mask = np.asarray(state.land_mask_grid.data)
        sst = float(np.nanmean(np.where(mask > 0.5, np.asarray(T_grid[..., 0]), np.nan)))
        sss = float(np.nanmean(np.where(mask > 0.5, np.asarray(S_grid[..., 0]), np.nan)))
        # fname is None on route-B non-root ranks (gather ran, no write).
        if fname is not None:
            if _snapshot_fn is not None:
                try:
                    _snapshot_fn(fname)
                except Exception as e:
                    print(f"    Snapshot failed: {e}", flush=True)
            print(
                f"  Wallclock budget {max_wallclock_seconds:.0f}s nearly "
                f"reached at day {day:.2f}; restart saved: {fname.name}.",
                flush=True,
            )
        sys.exit(0)

    # ---- B2 standing-mode time diagnostic χ ----
    # χ(t) = ||η^n - ½(η^{n−1} + η^{n+1})||² / ||η^n||²  (Williams 2009).
    # Tracks the 2-Δt-block computational-mode amplitude in η: a clean
    # integration sits at χ~1e-6, a growing computational mode shows χ
    # rising exponentially 5–10 days BEFORE max|u| spikes.  Buffer
    # holds the last 3 end-of-block η snapshots; we compute χ on the
    # middle once we have 3.
    eta_history: list[np.ndarray] = []

    # ----- JRA55-do block-scan path (multi-core friendly) ----------------
    # Wraps N ocean steps in lax.scan inside @jax.jit for ~30x GPU
    # speedup.  The scan body calls model._step_impl() (no inner JIT)
    # to avoid nested JIT boundaries that caused divergence with
    # partial-cell coordinates.
    use_scan_blocks = (jra55_state is not None
                       and not jra55_state.get("_use_single_step", False))
    use_gpu_interp = (jra55_state is not None
                      and jra55_state.get("_gpu_interp", False))
    # Prognostic sea ice is wired ONLY into the block-scan path.  The
    # single-step fallback (_jra55_step) does not advance the ice tile, and
    # setup has already disabled the freeze-cap SST stand-in for --jra55-sea-ice
    # — so a fallback run would get NEITHER ice NOR the freezing-point floor.
    # Fail loudly rather than silently run a degraded polar surface closure.
    if (jra55_state is not None
            and jra55_state.get("enable_sea_ice", False)
            and not use_scan_blocks):
        raise SystemExit(
            "--jra55-sea-ice requires the block-scan path, but this run set "
            "_use_single_step (single-step fallback). The single-step path "
            "does not advance prognostic sea ice and the freeze-cap stand-in "
            "is disabled under sea ice, so polar SST would be unconstrained. "
            "Use the block-scan path (default) or drop --jra55-sea-ice."
        )
    if use_scan_blocks:
        # Scan blocks trace _step_impl directly (bypassing the public
        # step shim) — prime the build-once caches from the CONCRETE
        # initial state so the traced body captures the vertex mask as
        # a constant (codex review MAJOR; census 8474554).
        # SPMD: the caller (run_omip_single) already primed the caches from
        # the UNSHARDED state before sharding; re-priming here would run
        # np.asarray on the sharded ``state`` — a hard non-addressable error
        # under route-B (shards span processes).  Skip it when spmd_step is set.
        if spmd_step is None:
            model.prime_step_caches(state)
        # Prognostic slab sea ice (--jra55-sea-ice): the block scan carries
        # (ocean_state, ice_state); thread the ice state across blocks.
        _ice_on = bool(jra55_state.get("enable_sea_ice", False))
        ice_state = jra55_state.get("ice_state_init") if _ice_on else None
        if use_gpu_interp:
            _get_block_fn_interp = _build_jra55_block_fn_interp(
                model, jra55_state, dt, spmd_step=spmd_step)
            print("  GPU-interp mode: forcing interpolation on GPU")
            # Pre-load the full JRA55 cache for repeat-year runs to
            # eliminate per-block Zarr I/O (~0.3s/block → ~0s/block).
            _full_cache = None
            if jra55_state.get("cycle", False):
                _fc_all, _fc_days, _fc_len = _preload_jra55_full_cache(
                    jra55_state)
                _full_cache = (_fc_all, _fc_days, _fc_len)
        else:
            block_fn = _build_jra55_block_fn(model, jra55_state, dt,
                                             spmd_step=spmd_step)
            _full_cache = None
        block_size = max(1, diag_every)
        if checkpoint_days is not None:
            steps_per_ckpt = max(1, int(round(checkpoint_days * 86400.0 / dt)))
        else:
            steps_per_ckpt = None

        block_start = start_step
        while block_start < n_steps and not blown_up:
            actual = min(block_size, n_steps - block_start)
            t_io_start = time.time()

            if use_gpu_interp and _full_cache is not None:
                raw_stack, runoff_records, record_meta = (
                    _slice_preloaded_records(
                        block_start, actual, dt, jra55_state,
                        *_full_cache))
            elif use_gpu_interp:
                raw_stack, runoff_records, record_meta = (
                    _preload_jra55_raw_records(
                        block_start, actual, dt, jra55_state))
            else:
                atm_stack, runoff_stack = _preload_jra55_forcing_block(
                    block_start, actual, dt, jra55_state,
                )
            if spmd_shard_stack is not None:
                # Lay the per-block forcing stacks out lat-band-sharded so
                # the in-scan interpolation / bulk fluxes stay shard-local
                # (an unsharded stack commits to device 0 and serializes
                # every forcing op there).
                if use_gpu_interp:
                    raw_stack = spmd_shard_stack(raw_stack)
                    runoff_records = spmd_shard_stack(runoff_records)
                else:
                    atm_stack = spmd_shard_stack(atm_stack)
                    runoff_stack = spmd_shard_stack(runoff_stack)
            io_dt = time.time() - t_io_start

            t_compute_start = time.time()
            if use_gpu_interp:
                bfn = _get_block_fn_interp(actual)
                if _ice_on:
                    state, ice_state = bfn(
                        state, raw_stack, runoff_records,
                        record_meta["record_days"],
                        jnp.float64(record_meta["block_start_day"]),
                        jnp.float64(record_meta["block_start_day_forcing"]),
                        ice_state,
                    )
                else:
                    state = bfn(
                        state, raw_stack, runoff_records,
                        record_meta["record_days"],
                        jnp.float64(record_meta["block_start_day"]),
                        jnp.float64(record_meta["block_start_day_forcing"]),
                    )
            else:
                if _ice_on:
                    state, ice_state = block_fn(
                        state, atm_stack, runoff_stack,
                        jnp.int32(block_start), ice_state,
                    )
                else:
                    state = block_fn(
                        state, atm_stack, runoff_stack,
                if (steps_per_ckpt is not None and
                        (step % steps_per_ckpt == 0 or step == n_steps)):
                    fname = save_restart(state, day, step, checkpoint_dir,
                                          ice_state=ice_state,
                                          grid_type=grid_type)
                    if _snapshot_fn is not None:
                        try:
                            _snapshot_fn(fname)
                        except Exception as e:
                            print(f"    Snapshot failed: {e}", flush=True)
                    print(f"    Restart saved: {fname.name}", flush=True)
                _maybe_wallclock_exit(state, step, day)

        jax.block_until_ready(state.T.data)
        wall = time.time() - t0
        ok = not blown_up and _check_finite(state, grid_type)
        return state, diag, wall, ok, blowup_info
    # --------------------------------------------------------------------

    # Restoring ramp: scale restoring strength linearly from 0 to 1 over
    # the first ``restoring_ramp_steps`` steps.  See ``--restoring-ramp-
    # days``; cubed_sphere defaults to 14 days at the call-site to
    # delay the face-edge PGF instability onset, other grids default
    # to 0.
    ramp_days = float(restoring_ramp_days)
    restoring_ramp_steps = max(1, int(ramp_days * 86400.0 / dt)) if ramp_days > 0 else 1

    _dyn_step = spmd_step if spmd_step is not None else model.step

    for i in range(start_step, n_steps):
        state = _dyn_step(state, dt)

        # Apply SST/SSS restoring (grid-agnostic, after dynamics step)
        if restoring_targets is not None:
            T_tgt, S_tgt = restoring_targets
            ramp_scale = min(1.0, (i + 1) / restoring_ramp_steps)
            state = _apply_restoring(
                state, grid_type, grid, T_tgt, S_tgt,
                dt, restoring_tau_s, ramp_scale=ramp_scale,
            )

        step = i + 1

        if step % 100 == 0:
            if not _check_finite(state, grid_type):
                # Debug: identify what failed
                if grid_type != "spectral":
                    mask = state.land_mask.data
                    m3 = mask[:, jnp.newaxis] if grid_type == "mpas" else mask[..., jnp.newaxis]
                    T_oc = jnp.where(m3 > 0.5, state.T.data, 0.0)
                    eta_max = float(jnp.max(jnp.abs(state.eta.data)))
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
    # lane keeps ``state`` in the lat-band SHARDED layout (v/v_mask carried as
    # the n_lat-row ``v_lower``) across steps; these two helpers flip the
    # residency at the classified boundaries and COUNT every full-state
    # transfer so the cost is visible in the run log (never silent).  With the
    # flag OFF both are exact no-ops (byte-identical default path).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-607-        else:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-608-            updates[name] = _gather_arr(jnp.asarray(val))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-609-    return state._replace(**updates)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-610-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-611-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:612:def make_sharded_ocean_step(model, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-613-    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-614-    sponge=None, t_seconds=None) -> state`` running ``model.step``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-615-    lat-band-SPMD.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-616-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-617-    The forcing channels mirror ``model.step``'s keyword surface: pass
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-984-    # (see the ``aux`` note in the signature).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-985-    sharded_step.aux = (geom_stacks, vmask_stack)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-986-    return sharded_step
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-987-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-988-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:989:def make_sharded_ocean_step_global(model, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-990-    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-991-    that takes a GLOBAL (single-device-layout) state + forcing and returns a
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-992-    GLOBAL state — the minimal-diff driver entry point.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-993-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-994-    Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1005-    if mesh is None:                   # single-device: plain step
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1006-        return lambda state, dt, surface_forcing=None, freshwater=None: (
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1007-            model.step(state, dt, freshwater=freshwater,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1008-                       surface_forcing=surface_forcing))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1009-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:1010:    inner = make_sharded_ocean_step(model, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1011-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1012-    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1013-        # Scatter the global state AND forcing to the band layout explicitly
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1014-        # (codex r17 item 3: the old comment claimed inner sharded the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1015-        # forcing; it forwarded it global and relied on implicit JIT input
--
tests/parallel/test_latlon_ocean_spmd_step.py-125-    # (cell fields P("lat"); the staggered v / v_mask carried as v_lower, the
tests/parallel/test_latlon_ocean_spmd_step.py-126-    # n_lat-row block that DOES divide N — a uniform tree.map(P("lat")) would
tests/parallel/test_latlon_ocean_spmd_step.py-127-    # fail on the n_lat+1 v rows).  The result is gathered (and the dropped pole
tests/parallel/test_latlon_ocean_spmd_step.py-128-    # row reappended) for the bit-comparison vs the single-device reference.
tests/parallel/test_latlon_ocean_spmd_step.py-129-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:130:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-131-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-132-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-133-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_step.py-134-    ss = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-135-
--
tests/parallel/test_latlon_ocean_spmd_step.py-242-        s = model.step(s, dt, freshwater=fw, surface_forcing=sf,
tests/parallel/test_latlon_ocean_spmd_step.py-243-                       sponge=sponge, t_seconds=jnp.asarray(i * dt))
tests/parallel/test_latlon_ocean_spmd_step.py-244-
tests/parallel/test_latlon_ocean_spmd_step.py-245-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-246-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:247:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-248-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-249-    fws = shard_forcing_latlon(fw, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-250-    sfs = shard_forcing_latlon(sf, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-251-    sponges = shard_forcing_latlon(sponge, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-252-
--
tests/parallel/test_latlon_ocean_spmd_step.py-394-    model = LatLonCGridOceanModel(grid, z_coord,
tests/parallel/test_latlon_ocean_spmd_step.py-395-                                  LatLonCGridOceanConfig.from_flat())
tests/parallel/test_latlon_ocean_spmd_step.py-396-    state0 = _perturbed_state(grid, z_coord)
tests/parallel/test_latlon_ocean_spmd_step.py-397-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-398-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:399:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-400-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-401-    bad = OceanSurfaceForcing(
tests/parallel/test_latlon_ocean_spmd_step.py-402-        tau_y=jnp.zeros((grid.n_lat + 1, grid.n_lon)))
tests/parallel/test_latlon_ocean_spmd_step.py-403-    with pytest.raises(ValueError, match="leading dim"):
tests/parallel/test_latlon_ocean_spmd_step.py-404-        step(ss, 600.0, surface_forcing=bad)
--
tests/parallel/test_latlon_ocean_spmd_step.py-407-@pytest.mark.skipif(jax.device_count() < 4,
tests/parallel/test_latlon_ocean_spmd_step.py-408-                    reason="needs >=4 devices (XLA_FLAGS host device count)")
tests/parallel/test_latlon_ocean_spmd_step.py-409-@pytest.mark.skipif(not _have_sharded_step(),
tests/parallel/test_latlon_ocean_spmd_step.py-410-                    reason="sharded_ocean_step module not present")
tests/parallel/test_latlon_ocean_spmd_step.py-411-def test_sharded_ocean_step_global_matches_explicit_scatter_gather():
tests/parallel/test_latlon_ocean_spmd_step.py:412:    """``make_sharded_ocean_step_global`` (global-in/global-out, the minimal
tests/parallel/test_latlon_ocean_spmd_step.py-413-    driver entry) must equal the explicit ``shard_state_latlon`` -> inner step ->
tests/parallel/test_latlon_ocean_spmd_step.py-414-    ``gather_state_latlon`` path BIT-FOR-BIT (it is literally that composition),
tests/parallel/test_latlon_ocean_spmd_step.py-415-    AND match the single-device reference to the same re-association floor.
tests/parallel/test_latlon_ocean_spmd_step.py-416-
tests/parallel/test_latlon_ocean_spmd_step.py-417-    Also exercises the WITH-forcing path through the global wrapper (a smooth
tests/parallel/test_latlon_ocean_spmd_step.py-418-    cell-shaped wind-stress + heat ``OceanSurfaceForcing``)."""
tests/parallel/test_latlon_ocean_spmd_step.py-419-    from legoesm.parallel.mesh import create_latlon_mesh
tests/parallel/test_latlon_ocean_spmd_step.py-420-    from legoesm.grids.latlon import ensure_geometry
tests/parallel/test_latlon_ocean_spmd_step.py-421-    from legoesm.ocean.dynamics.sharded_ocean_step import (
tests/parallel/test_latlon_ocean_spmd_step.py-422-        make_sharded_ocean_step,
tests/parallel/test_latlon_ocean_spmd_step.py:423:        make_sharded_ocean_step_global,
tests/parallel/test_latlon_ocean_spmd_step.py-424-        shard_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-425-        gather_state_latlon,
tests/parallel/test_latlon_ocean_spmd_step.py-426-    )
tests/parallel/test_latlon_ocean_spmd_step.py-427-    from legoesm.ocean.state import OceanSurfaceForcing
tests/parallel/test_latlon_ocean_spmd_step.py-428-
--
tests/parallel/test_latlon_ocean_spmd_step.py-449-
tests/parallel/test_latlon_ocean_spmd_step.py-450-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-451-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py-452-
tests/parallel/test_latlon_ocean_spmd_step.py-453-    # explicit scatter -> inner sharded step -> gather
tests/parallel/test_latlon_ocean_spmd_step.py:454:    inner = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-455-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-456-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-457-        ss = inner(ss, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_step.py-458-    ss = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-459-
tests/parallel/test_latlon_ocean_spmd_step.py-460-    # global-in/global-out wrapper (scatter + gather PER STEP)
tests/parallel/test_latlon_ocean_spmd_step.py:461:    glob = make_sharded_ocean_step_global(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-462-    sg = state0
tests/parallel/test_latlon_ocean_spmd_step.py-463-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-464-        sg = glob(sg, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_step.py-465-
tests/parallel/test_latlon_ocean_spmd_step.py-466-    for nm in ("u", "v", "eta", "T", "S"):
--
tests/parallel/test_latlon_spmd_fused_halo.py-344-
tests/parallel/test_latlon_spmd_fused_halo.py-345-    outs = {}
tests/parallel/test_latlon_spmd_fused_halo.py-346-    hlo_counts = {}
tests/parallel/test_latlon_spmd_fused_halo.py-347-    for flag in ("0", "1"):
tests/parallel/test_latlon_spmd_fused_halo.py-348-        _env_flag(monkeypatch, flag)
tests/parallel/test_latlon_spmd_fused_halo.py:349:        step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-350-        ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-351-        hlo_counts[flag] = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-352-            jax.jit(step).lower(ss, 600.0).compile().as_text())
tests/parallel/test_latlon_spmd_fused_halo.py-353-        for _ in range(n_steps):
tests/parallel/test_latlon_spmd_fused_halo.py-354-            ss = step(ss, 600.0)
--
tests/parallel/test_latlon_spmd_fused_halo.py-367-    # the callable identity, so re-jitting the SAME step object would
tests/parallel/test_latlon_spmd_fused_halo.py-368-    # freeze the wrapper's python body and never re-evaluate the key
tests/parallel/test_latlon_spmd_fused_halo.py-369-    # (an outer-jit artifact, not the production call pattern — the
tests/parallel/test_latlon_spmd_fused_halo.py-370-    # driver calls step() directly each step).
tests/parallel/test_latlon_spmd_fused_halo.py-371-    _env_flag(monkeypatch, "0")
tests/parallel/test_latlon_spmd_fused_halo.py:372:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-373-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-374-    n_off = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-375-        jax.jit(lambda s, d: step(s, d)).lower(ss, 600.0)
tests/parallel/test_latlon_spmd_fused_halo.py-376-        .compile().as_text())
tests/parallel/test_latlon_spmd_fused_halo.py-377-    _env_flag(monkeypatch, "1")
--
tests/parallel/test_persistent_sharded_ocean_loop.py-1-"""Parity gate for the PERSISTENT lat-band-sharded OMIP ocean loop
tests/parallel/test_persistent_sharded_ocean_loop.py-2-(scaling-M2 increment 1, ``run_omip_core2 --spmd-persistent-state``).
tests/parallel/test_persistent_sharded_ocean_loop.py-3-
tests/parallel/test_persistent_sharded_ocean_loop.py-4-The production multi-GPU OMIP host loop used to call
tests/parallel/test_persistent_sharded_ocean_loop.py:5:``make_sharded_ocean_step_global`` EVERY step — a full-state scatter
tests/parallel/test_persistent_sharded_ocean_loop.py-6-(``shard_state_latlon``) + gather (``gather_state_latlon``) per step, i.e.
tests/parallel/test_persistent_sharded_ocean_loop.py-7-2 full-state transfers/step.  The persistent lane instead keeps the state
tests/parallel/test_persistent_sharded_ocean_loop.py-8-lat-band SHARDED across steps via ``make_sharded_ocean_step`` and gathers
tests/parallel/test_persistent_sharded_ocean_loop.py-9-ONLY at real output boundaries.  This gate asserts, over a 10-step loop with
tests/parallel/test_persistent_sharded_ocean_loop.py-10-production-shaped surface + freshwater forcing (``normalize_freshwater=True``
--
tests/parallel/test_persistent_sharded_ocean_loop.py-168-        return (sf._replace(tau_x=sf.tau_x + 5.0e-3 * u_sfc,
tests/parallel/test_persistent_sharded_ocean_loop.py-169-                            tau_y=sf.tau_y + 5.0e-3 * v_sfc),
tests/parallel/test_persistent_sharded_ocean_loop.py-170-                u_sfc, v_sfc)
tests/parallel/test_persistent_sharded_ocean_loop.py-171-
tests/parallel/test_persistent_sharded_ocean_loop.py-172-    # ---------------- OLD lane: per-step global-in/global-out wrapper --------
tests/parallel/test_persistent_sharded_ocean_loop.py:173:    glob = sos.make_sharded_ocean_step_global(model, mesh)
tests/parallel/test_persistent_sharded_ocean_loop.py-174-    sg = state0
tests/parallel/test_persistent_sharded_ocean_loop.py-175-    cur_probe = {}
tests/parallel/test_persistent_sharded_ocean_loop.py-176-    for k in range(1, n_steps + 1):
tests/parallel/test_persistent_sharded_ocean_loop.py-177-        # surface-current consumer on the GLOBAL state (full staggered v)
tests/parallel/test_persistent_sharded_ocean_loop.py-178-        assert sg.v.data.shape[0] == n_lat + 1
--
tests/parallel/test_persistent_sharded_ocean_loop.py-196-    # the wrapper lane pays 2 full-state transfers PER STEP
tests/parallel/test_persistent_sharded_ocean_loop.py-197-    assert old_calls == {"shard": n_steps, "gather": n_steps}, old_calls
tests/parallel/test_persistent_sharded_ocean_loop.py-198-
tests/parallel/test_persistent_sharded_ocean_loop.py-199-    # ---------------- NEW lane: persistent sharded loop ----------------------
tests/parallel/test_persistent_sharded_ocean_loop.py-200-    calls["shard"] = calls["gather"] = 0
tests/parallel/test_persistent_sharded_ocean_loop.py:201:    inner = sos.make_sharded_ocean_step(model, mesh)
tests/parallel/test_persistent_sharded_ocean_loop.py-202-    ss = sos.shard_state_latlon(state0, mesh)          # ONE initial shard
tests/parallel/test_persistent_sharded_ocean_loop.py-203-    for k in range(1, n_steps + 1):
tests/parallel/test_persistent_sharded_ocean_loop.py-204-        # surface-current consumer on the SHARDED state: v is the n_lat-row
tests/parallel/test_persistent_sharded_ocean_loop.py-205-        # v_lower carrier at every loop top (the snapshot boundary re-shards
tests/parallel/test_persistent_sharded_ocean_loop.py-206-        # back to it); _surface_currents reconstructs the staggered top row
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-445-        from bench_ocean_mpi_scaling import ocean_invariants
scripts/bench/bench_ocean_latlon_spmd_scaling.py-446-        inv_before = ocean_invariants(model, s0, n_ranks=1)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-447-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-448-    if nd == 1:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-449-        mesh = None
scripts/bench/bench_ocean_latlon_spmd_scaling.py:450:        step = make_sharded_ocean_step(model, None)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-451-        s = s0
scripts/bench/bench_ocean_latlon_spmd_scaling.py-452-    else:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-453-        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-454-                                 axis_names=("lat",))
scripts/bench/bench_ocean_latlon_spmd_scaling.py:455:        step = make_sharded_ocean_step(model, mesh)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-456-        s = shard_state_latlon(s0, mesh)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-457-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-458-    # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
scripts/bench/bench_ocean_latlon_spmd_scaling.py-459-    # blocks with sync only AROUND the block — the previous per-step
scripts/bench/bench_ocean_latlon_spmd_scaling.py-460-    # host-synced loop measured dispatch+sync latency, not fused device
--
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-120-        s = model.step(s, dt)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-121-    # Prime the build-once vertex-mask cache from the CONCRETE state (the
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-122-    # serial run above already did; belt-and-braces for wrapper band masks).
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-123-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-124-
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:125:    step = make_sharded_ocean_step(model, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-126-    ss = shard_state_latlon(state0, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-127-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-128-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-129-    out = gather_state_latlon(ss, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-130-
--
tests/unit/test_run_omip_latlon_spmd.py-77-        model, state0, checkpoint_dir=tmp_path / "serial", **common)
tests/unit/test_run_omip_latlon_spmd.py-78-    assert ok_serial
tests/unit/test_run_omip_latlon_spmd.py-79-
tests/unit/test_run_omip_latlon_spmd.py-80-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py-81-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py:82:    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-83-    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-84-    ckpt_dir = tmp_path / "spmd"
tests/unit/test_run_omip_latlon_spmd.py-85-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-86-        model, ss0, checkpoint_dir=ckpt_dir,
tests/unit/test_run_omip_latlon_spmd.py-87-        spmd_step=spmd_step,
--
tests/unit/test_run_omip_latlon_spmd.py-214-    serial_final, _d, _w, ok_serial, _b = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-215-        model, state0, jra55_state=_fresh_js(), **common)
tests/unit/test_run_omip_latlon_spmd.py-216-    assert ok_serial, "serial JRA55 loop reported not-ok"
tests/unit/test_run_omip_latlon_spmd.py-217-
tests/unit/test_run_omip_latlon_spmd.py-218-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py:219:    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-220-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py-221-    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-222-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-223-        model, ss0, jra55_state=_fresh_js(),
tests/unit/test_run_omip_latlon_spmd.py-224-        spmd_step=spmd_step,
--
tests/parallel/test_latlon_ocean_spmd_tripole.py-147-        s = model.step(s, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_tripole.py-148-
tests/parallel/test_latlon_ocean_spmd_tripole.py-149-    model._ensure_vertex_mask(state0)        # prime the build-once vmask cache
tests/parallel/test_latlon_ocean_spmd_tripole.py-150-
tests/parallel/test_latlon_ocean_spmd_tripole.py-151-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_tripole.py:152:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-153-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-154-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_tripole.py-155-        ss = step(ss, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_tripole.py-156-    ss = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-157-
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-87-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-88-        s = model.step(s, dt)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-89-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-90-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-91-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:92:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-93-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-94-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-95-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-96-    ss = gather_state_latlon(ss, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-97-    return s, ss
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-120-    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-121-    model = LatLonCGridOceanModel(grid, z_coord, cfg)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-122-    state0 = _perturbed_state(grid, z_coord)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-123-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-124-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:125:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-126-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-127-    lowered = jax.jit(step).lower(ss, 600.0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-128-    return lowered.compile().as_text()
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-129-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-130-
--
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-1-"""Multi-PROCESS jax.distributed equivalence VALIDATION for the lat-band SPMD ocean
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py:2:step — the multi-node correctness gate for ``make_sharded_ocean_step_global``.
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-3-
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-4-Runs STANDALONE (not pytest): the root ``tests/conftest.py`` initializes the XLA
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-5-backend (``ensure_metal_or_fallback`` -> ``jax.default_backend()``) at collection,
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-6-before a test module could call ``jax.distributed.initialize()`` (which must
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-7-precede backend init).  A standalone script bootstraps jax.distributed FIRST.
--
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-115-
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-116-def _spmd_step_result(model, state0, n_dev):
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
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-129-    return ss
scripts/validate/validate_latlon_ocean_spmd_multiprocess.py-130-

exec
/usr/bin/bash -lc "rg -n -C 5 \"_dyn_step\\(|spmd_step|_build_jra55_block_fn|_build_jra55_block_fn_interp\" scripts/run/run_omip.py
rg -n -C 5 \"_ocean_step\\(|_ocean_step\\s*=|jax\\.jit|lax\\.scan\" scripts/run/run_omip_core2.py
sed -n '2700,2795p' scripts/run/run_omip.py
sed -n '200,260p' tests/unit/test_run_omip_latlon_spmd.py
rg -n -C 4 \"timed_scan_blocks\" tests scripts --glob '"'!scripts/bench/metadata.py'"'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 419ms:
2298-    }
2299-
2300-    return raw_stack, runoff_stack, record_meta
2301-
2302-
2303:def _build_jra55_block_fn(model, jra55_state, dt, spmd_step=None):
2304-    """Return a JIT-compiled block function that runs N steps via lax.scan.
2305-
2306-    Captures everything that's static across the block (sponge, SSS
2307-    target, freeze-cap mask, coupler config, dt) in the closure so
2308-    the scan body has a clean ``(state, idx) → (state', None)`` signature.
--
2360-
2361-    # Lat-band SPMD (--enable-latlon-spmd): the scan body's dynamics step
2362-    # runs through the sharded wrapper (same forcing kwargs as _step_impl;
2363-    # the wrapper's cache/arm-restore Python runs ONCE at block trace).
2364-    # Sea ice is refused upstream (the ice tile is not SPMD-audited yet).
2365:    if spmd_step is not None and enable_sea_ice:
2366-        raise ValueError(
2367:            "spmd_step + prognostic sea ice is unsupported "
2368-            "(run_omip_single refuses --jra55-sea-ice with "
2369-            "--enable-latlon-spmd).")
2370:    _dyn_step = (spmd_step if spmd_step is not None
2371-                 else lambda st, d, **kw: model._step_impl(st, d, **kw))
2372-
2373-    @jax.jit
2374-    def block_fn(state, atm_stack, runoff_stack, block_start_step,
2375-                 ice_state=None):
--
2446-            if enable_ramp and enable_sponge:
2447-                sponge_step = sponge._replace(gamma=sponge.gamma * ramp)
2448-            else:
2449-                sponge_step = sponge
2450-
2451:            new_state = _dyn_step(
2452-                state_in, dt,
2453-                freshwater=fw, surface_forcing=sf, sponge=sponge_step,
2454-            )
2455-
2456-            # SSS restoring (gated at compile time via Python `if`).
--
2501-        return final  # (state, ice_state) when sea-ice on, else state
2502-
2503-    return block_fn
2504-
2505-
2506:def _build_jra55_block_fn_interp(model, jra55_state, dt, spmd_step=None):
2507-    """JIT-compiled block function with GPU-side forcing interpolation.
2508-
2509:    Like ``_build_jra55_block_fn``, but instead of receiving pre-
2510-    interpolated per-step forcing, receives the native 3-hourly records
2511-    and computes the linear interpolation + solar zenith inside the
2512-    ``lax.scan`` body on GPU.  This reduces host-side I/O from N calls
2513-    to ``jra55_to_atm_surface`` (N=288 for 1 day) down to ~9 Zarr reads
2514-    per block.
--
2552-        T_freeze_C_static = float(jra55_state["T_freeze_ocean_C"])
2553-    else:
2554-        freeze_mask_static = None
2555-        T_freeze_C_static = -1.8
2556-
2557:    # Prognostic slab sea ice (opt-in) — see _build_jra55_block_fn.
2558-    enable_sea_ice = bool(jra55_state.get("enable_sea_ice", False))
2559-    ice_cfg = jra55_state.get("ice_config")
2560-
2561-    _maxvel_3d = model.config.barotropic.maxvel_barotropic
2562-    enable_maxvel = _maxvel_3d > 0.0
2563-
2564:    # Lat-band SPMD: see _build_jra55_block_fn.
2565:    if spmd_step is not None and enable_sea_ice:
2566-        raise ValueError(
2567:            "spmd_step + prognostic sea ice is unsupported "
2568-            "(run_omip_single refuses --jra55-sea-ice with "
2569-            "--enable-latlon-spmd).")
2570:    _dyn_step = (spmd_step if spmd_step is not None
2571-                 else lambda st, d, **kw: model._step_impl(st, d, **kw))
2572-
2573-    lat_2d = jra55_state["lat_2d"]
2574-    lon_2d = jra55_state["lon_2d"]
2575-    _rpd = float(RECORDS_PER_DAY)
--
2716-                        ocean_mask=state_in.land_mask.data,
2717-                    )
2718-
2719-                sponge_k = (sponge._replace(gamma=sponge.gamma * ramp)
2720-                            if enable_sponge else None)
2721:                new_state = _dyn_step(
2722-                    state_in, dt, freshwater=fw,
2723-                    surface_forcing=sf, sponge=sponge_k,
2724-                )
2725-
2726-                if enable_sss:
--
3157-                   checkpoint_days=None, checkpoint_dir=None,
3158-                   max_wallclock_seconds: float = 0.0,
3159-                   restart_buffer_seconds: float = 600.0,
3160-                   start_step=0,
3161-                   nudge_woa_tau=0.0, T_woa_3d=None, S_woa_3d=None,
3162:                   snapshot_fn=None, spmd_step=None, spmd_gather=None,
3163-                   spmd_shard_stack=None):
3164-    """Run time loop with diagnostics.
3165-
3166-    Two forcing paths, mutually exclusive:
3167-
3168-    * **restoring** (default): plain ``model.step(state, dt)`` followed
3169-      by Haney SST/SSS restoring when ``restoring_targets`` is set.
3170:      With ``spmd_step`` set (``--enable-latlon-spmd``), the dynamics
3171-      step runs through that lat-band-SPMD callable instead — the state
3172-      arrives sharded and every downstream op (restoring, finite checks,
3173-      diagnostics, restart saves) works on the sharded global arrays
3174-      transparently under the single-controller GSPMD runtime.
3175-    * **jra55_do_tropical**: ``_jra55_step(...)`` per step using a
--
3199-    if jra55_state is not None and restoring_targets is not None:
3200-        raise ValueError(
3201-            "_run_omip_loop: jra55_state and restoring_targets are mutually "
3202-            "exclusive — choose one forcing path."
3203-        )
3204:    if spmd_step is not None and jra55_state is not None:
3205:        # The block-scan lanes now thread spmd_step; the two unsupported
3206-        # JRA sub-modes still refuse loudly.
3207-        if jra55_state.get("_use_single_step", False):
3208-            raise ValueError(
3209:                "spmd_step + the JRA55 single-step fallback is unsupported "
3210-                "(_jra55_step calls model.step directly); use the "
3211-                "block-scan path (default).")
3212-        if jra55_state.get("enable_sea_ice", False):
3213-            raise ValueError(
3214:                "spmd_step + prognostic sea ice is unsupported "
3215-                "(--jra55-sea-ice; the ice tile is not SPMD-audited).")
3216-    if checkpoint_days is not None and checkpoint_dir is None:
3217-        raise ValueError(
3218-            "_run_omip_loop: checkpoint_days requires checkpoint_dir."
3219-        )
--
3375-        # initial state so the traced body captures the vertex mask as
3376-        # a constant (codex review MAJOR; census 8474554).
3377-        # SPMD: the caller (run_omip_single) already primed the caches from
3378-        # the UNSHARDED state before sharding; re-priming here would run
3379-        # np.asarray on the sharded ``state`` — a hard non-addressable error
3380:        # under route-B (shards span processes).  Skip it when spmd_step is set.
3381:        if spmd_step is None:
3382-            model.prime_step_caches(state)
3383-        # Prognostic slab sea ice (--jra55-sea-ice): the block scan carries
3384-        # (ocean_state, ice_state); thread the ice state across blocks.
3385-        _ice_on = bool(jra55_state.get("enable_sea_ice", False))
3386-        ice_state = jra55_state.get("ice_state_init") if _ice_on else None
3387-        if use_gpu_interp:
3388:            _get_block_fn_interp = _build_jra55_block_fn_interp(
3389:                model, jra55_state, dt, spmd_step=spmd_step)
3390-            print("  GPU-interp mode: forcing interpolation on GPU")
3391-            # Pre-load the full JRA55 cache for repeat-year runs to
3392-            # eliminate per-block Zarr I/O (~0.3s/block → ~0s/block).
3393-            _full_cache = None
3394-            if jra55_state.get("cycle", False):
3395-                _fc_all, _fc_days, _fc_len = _preload_jra55_full_cache(
3396-                    jra55_state)
3397-                _full_cache = (_fc_all, _fc_days, _fc_len)
3398-        else:
3399:            block_fn = _build_jra55_block_fn(model, jra55_state, dt,
3400:                                             spmd_step=spmd_step)
3401-            _full_cache = None
3402-        block_size = max(1, diag_every)
3403-        if checkpoint_days is not None:
3404-            steps_per_ckpt = max(1, int(round(checkpoint_days * 86400.0 / dt)))
3405-        else:
--
3657-    # delay the face-edge PGF instability onset, other grids default
3658-    # to 0.
3659-    ramp_days = float(restoring_ramp_days)
3660-    restoring_ramp_steps = max(1, int(ramp_days * 86400.0 / dt)) if ramp_days > 0 else 1
3661-
3662:    _dyn_step = spmd_step if spmd_step is not None else model.step
3663-
3664-    for i in range(start_step, n_steps):
3665:        state = _dyn_step(state, dt)
3666-
3667-        # Apply SST/SSS restoring (grid-agnostic, after dynamics step)
3668-        if restoring_targets is not None:
3669-            T_tgt, S_tgt = restoring_targets
3670-            ramp_scale = min(1.0, (i + 1) / restoring_ramp_steps)
--
4586-        # Provide the ocean mask for global freeze-cap when no sponge.
4587-        jra55_state["_ocean_mask_2d"] = state.land_mask.data > 0.5
4588-        # GPU-interp path (default): interpolation inside the lax.scan
4589-        # block for all grids (MPAS regridding is handled in
4590-        # _preload_jra55_raw_records).  --no-gpu-interp routes to the
4591:        # CPU-interp block path (_build_jra55_block_fn).
4592-        jra55_state["_gpu_interp"] = bool(args.gpu_interp)
4593-        # Restart: restore the prognostic sea-ice state so a checkpointed
4594-        # --jra55-sea-ice run does NOT resume on the zero cold-start ice.  An
4595-        # old ocean-only restart (no ice_* keys) returns None -> cold start
4596-        # kept (no error).
--
4766-    # --- Lat-band SPMD (--enable-latlon-spmd): wrap the dynamics step in
4767-    # the validated multi-GPU sharded ocean step and shard the state.
4768-    # Single-controller only; restoring lane only (the JRA55 block
4769-    # functions call model._step_impl directly — follow-up).  Runs AFTER
4770-    # the restart load so a resumed state is sharded too.
4771:    spmd_step = None
4772-    spmd_gather = None
4773-    spmd_shard_stack = None
4774-    if run_config.enable_latlon_spmd:
4775-        if grid_type != "latlon":
4776-            raise SystemExit(
--
4818-            from legoesm.parallel.mesh import create_latlon_mesh
4819-            # Prime the build-once vertex-mask cache from the CONCRETE
4820-            # state so the wrapper can build per-band masks host-side.
4821-            model.prime_step_caches(state)
4822-            _dev = create_latlon_mesh(n_devices=_nd)
4823:            spmd_step = make_sharded_ocean_step(model, _dev.mesh)
4824-            spmd_gather = partial(gather_state_latlon, mesh=_dev.mesh)
4825-            state = shard_state_latlon(state, _dev.mesh)
4826-            # Lay per-block forcing stacks out lat-band-sharded so the
4827-            # in-scan interpolation / bulk fluxes stay shard-local (shared
4828-            # layout helper — see shard_forcing_stack_latlon).
--
4854-        T_woa_3d=(T_woa * state.land_mask.data[..., jnp.newaxis]).astype(
4855-            state.T.data.dtype) if args.nudge_woa_tau > 0 and T_woa is not None else None,
4856-        S_woa_3d=(S_woa * state.land_mask.data[..., jnp.newaxis]).astype(
4857-            state.S.data.dtype) if args.nudge_woa_tau > 0 and S_woa is not None else None,
4858-        snapshot_fn=_snapshot_fn,
4859:        spmd_step=spmd_step,
4860-        spmd_gather=spmd_gather,
4861-        spmd_shard_stack=spmd_shard_stack,
4862-    )
4863-    if spmd_gather is not None:
4864-        # Downstream report/plot/save paths expect the full (n_lat+1)
3930-                        "REQUIRED with --prescribed-flow: the GM bolus (skew) "
3931-                        "transport is parameterized tracer ADVECTION applied "
3932-                        "outside the pinned mass-flux block, so the model "
3933-                        "constructor rejects prescribed_flow+GM/Redi.")
3934-    p.add_argument("--scan-block", type=int, default=0,
3935:                   help="Issue #354: wrap the time loop in jax.lax.scan, "
3936-                        "fusing this many steps per block (CORE-II forcing "
3937-                        "sampled on-device, no per-step host roundtrip). "
3938-                        "0 (default) = the bit-identical Python loop. "
3939-                        "Tripole only; incompatible with --nudge-woa / "
3940-                        "--spinup-drag. Diagnostics run at block boundaries.")
--
4332-        args.ice_categories, args.ice_ridging, args.prognostic_sea_ice)
4333-
4334-    # --prescribed-flow gates (PRE-BUILD, on the static args): grid support +
4335-    # the --spinup-drag rejection + the --no-gm-redi requirement.  NB: no
4336-    # --scan-block gate — the lever is in-model (inside _step_impl), so the
4337:    # lax.scan block path pins correctly.
4338-    validate_prescribed_flow_args(args.prescribed_flow, args.grid,
4339-                                  args.spinup_drag_tau_days,
4340-                                  no_gm_redi=args.no_gm_redi)
4341-
4342-    # Precision policy. The all-fp32 policy is set when --fp32 is given; the
--
5463-
5464-    _log_diag_csv(0, 0.0, d0, 0.0, ice=ice_state)
5465-
5466-    # ------------------------------------------------------------------
5467-    # Multi-GPU lat-band SPMD step (--n-gpus N): partition the GLOBAL ocean state
5468:    # by latitude band across N local devices.  ``_ocean_step(state, sf, fw)`` is
5469-    # the single per-step entry the host loop calls; default (N=1) is the plain
5470-    # single-device model.step (byte-identical).  The global-in/global-out wrapper
5471-    # scatters/gathers each step, so the host post-step BCs (SSS restore /
5472-    # prognostic ice / geothermal / BBL / nudge / drag) operate on the gathered
5473-    # GLOBAL state UNCHANGED.  n_lat is already SPMD-divisible (build_tripole
--
5480-        # active and the carry is unseeded; a restart-loaded tke passes
5481-        # through untouched).
5482-        state = model.seed_tke(state)
5483-        # MPASOceanModel.step has no t_seconds (dm2dc, its only consumer, is
5484-        # arg-gated to tripole/latlon) -- passing it TypeErrors at step 1.
5485:        _ocean_step = (lambda st, sf, fw, t_sec=None:
5486-                       model.step(st, dt, surface_forcing=sf, freshwater=fw))
5487-    else:
5488:        _ocean_step = (lambda st, sf, fw, t_sec=None:
5489-                       model.step(st, dt, surface_forcing=sf, freshwater=fw,
5490-                                  t_seconds=t_sec))
5491-    # --spmd-persistent-state lane state (scaling-M2): OFF by default so the
5492-    # residency helpers below are no-ops and the loop is byte-identical.
5493-    _spmd_persistent = False
--
5566-            from legoesm.ocean.dynamics.sharded_ocean_step import (
5567-                gather_state_latlon,
5568-                make_sharded_ocean_step,
5569-                shard_state_latlon,
5570-            )
5571:            _spmd_inner = make_sharded_ocean_step(model, _spmd_mesh)
5572:            _ocean_step = (lambda st, sf, fw, t_sec=None:
5573-                           _spmd_inner(st, dt, surface_forcing=sf,
5574-                                       freshwater=fw))
5575-
5576-            def _pers_shard_fn(st, _mesh=_spmd_mesh):
5577-                return shard_state_latlon(st, _mesh)
--
5586-                  f"(--spmd-persistent-state): full-state gathers only at "
5587-                  f"snapshot/abort/final + counted per-step forcings.")
5588-        else:
5589-            _spmd_step = make_sharded_ocean_step_global(model, _spmd_mesh)
5590-            # t_sec is always None here (tide-enabled fail-fasts above).
5591:            _ocean_step = (lambda st, sf, fw, t_sec=None:
5592-                           _spmd_step(st, dt, surface_forcing=sf,
5593-                                      freshwater=fw))
5594-            print(f"[setup] multi-GPU lat-band SPMD: {args.n_gpus} devices, "
5595-                  f"n_lat={n_lat_final} ({n_lat_final // args.n_gpus} "
5596-                  f"rows/band); global-in/global-out wrapper (host BCs on "
--
5651-    _ledger0 = host_pull_ledger()
5652-
5653-    t_wall = time.time()
5654-
5655-    # ------------------------------------------------------------------
5656:    # Issue #354: optional lax.scan block-stepping (tripole; no nudge/drag).
5657-    # The CORE-II forcing is sampled on-device (no per-step host roundtrip),
5658-    # so XLA fuses each block of ``--scan-block`` steps.  Default
5659-    # (--scan-block 0) keeps the bit-identical Python loop below.
5660-    # Diagnostics / snapshots / non-finite abort run at BLOCK BOUNDARIES.
5661-    # ------------------------------------------------------------------
5662-    _tti = getattr(getattr(model, "config", None),
5663-                   "tracer_time_integrator", "euler")
5664-    if args.n_gpus > 1 and int(args.scan_block) > 0:
5665-        raise SystemExit(
5666:            "--n-gpus > 1 and --scan-block are mutually exclusive: the lax.scan "
5667-            "block path fuses single-device on-device steps (it does not use the "
5668-            "lat-band sharded step). Pick one — multi-GPU SPMD (the host Python "
5669-            "loop, --scan-block 0) OR single-device scan fusion.")
5670-    # Equilibrium tide disqualifies the scan-block path: its body steps via
5671-    # model._step_impl(...) with no t_seconds (bypassing step()'s eager
--
5678-                and not args.sss_restore
5679-                and not _tide_enabled
5680-                and _tti != "ab2")
5681-    if int(args.scan_block) > 0 and not use_scan:
5682-        why = ("AB2 tracer time integrator (None->Field carry breaks "
5683:               "lax.scan)" if _tti == "ab2"
5684-               else "tidal_forcing enabled (the scan body does not thread the "
5685-                    "model time the tide needs)" if _tide_enabled
5686-               else "grid!=tripole or WOA-nudging / spin-up-drag / SSS-restoring "
5687-                    "enabled (those need per-step host updates)")
5688-        print(f"[scan] --scan-block ignored: {why}.", flush=True)
--
5723-        # Scan blocks trace _step_impl directly — prime build-once
5724-        # caches from the concrete state first (vertex-mask constant).
5725-        model.prime_step_caches(state)
5726-        block_fn = build_omip2_scan_block_fn(model, dt, gshape, ramp_s=ramp_s)
5727-        bsz = int(args.scan_block)
5728:        print(f"[run] lax.scan block-stepping: block<={bsz} steps, split at "
5729-              f"diag/snapshot/year boundaries so output cadence matches the "
5730-              f"Python loop (CORE-II forcing fused on-device)", flush=True)
5731-
5732-        def _block_steps(step):
5733-            # Cap the block so it ENDS on the next diagnostic / snapshot /
--
6153-                                     _ig(fw.runoff), _ig(fw.ice_fw))
6154-                print(f"[fwbudget] {app_grid_type}: P={_P:+.4f} E={_E:+.4f} "
6155-                      f"R={_Rn:+.4f} ice={_Ic:+.4f} net(P-E+R+ice)="
6156-                      f"{_P - _E + _Rn + _Ic:+.4f} Sv (raw pre-normalize, "
6157-                      f"area-wtd over wet)", flush=True)
6158:            # _ocean_step = single-device model.step (default), the lat-band
6159-            # SPMD global-in/global-out step (--n-gpus > 1), or the PERSISTENT
6160-            # sharded inner step (--spmd-persistent-state); all apply the
6161-            # in-core wind-stress / heat / freshwater forcing.  Default lanes
6162-            # return a GLOBAL state, so the host post-step BCs below are
6163-            # unchanged; the persistent lane keeps the state SHARDED — the
6164-            # leaf-wise/jnp post-step BCs below operate on it identically (see
6165-            # the residency-helper classification).  t_seconds threads the
6166-            # equilibrium-tide model time (None when tide off; the SPMD path
6167-            # fail-fasts at setup if the tide is enabled).
6168-            state = _ensure_sharded_state(state)
6169:            state = _ocean_step(state, sf, fw, _t_sec)
6170-        if sss_restore_cfg is not None:
6171-            # NEMO-faithful ice gate (namsbc_ssr nn_sssr_ice=0: no SSS restoring
6172-            # under sea ice).  Feed the SAME prescribed siconc the albedo uses
6173-            # (``_sic``; None only if neither --ice-albedo nor a siconc field is
6174-            # available, in which case the restoring is ungated as before).
                )
                sf = OceanSurfaceForcing(
                    sw_down=atm.sw_down, q_net=q_net,
                    tau_x=tile.tau_x * ramp, tau_y=tile.tau_y * ramp,
                    freshwater=None,
                )

                # Prognostic slab sea ice: partition surface forcing between
                # open ocean (f_ocean=1-A) and the ice tile.  ocean_mask: land
                # cells receive no ice->ocean forcing (mask-aware blend
                # contract; land_mask is scan-carry state, traced-safe).
                if enable_sea_ice:
                    new_ice, fw, sf = omip_sea_ice_surface_forcing(
                        ice_state=ice_in, ice_config=ice_cfg, atm=atm,
                        ocean_sst_K=sst_K, open_ocean_sf=sf, open_ocean_fw=fw,
                        dt=dt, grid=None,
                        ocean_mask=state_in.land_mask.data,
                    )

                sponge_k = (sponge._replace(gamma=sponge.gamma * ramp)
                            if enable_sponge else None)
                new_state = _dyn_step(
                    state_in, dt, freshwater=fw,
                    surface_forcing=sf, sponge=sponge_k,
                )

                if enable_sss:
                    S = new_state.S.data
                    _sss_mask = new_state.land_mask.data
                    S_new = S.at[..., 0].set(
                        S[..., 0] - sss_alpha_static * (
                            S[..., 0] - sss_target_static) * _sss_mask)
                    new_state = new_state._replace(
                        S=new_state.S.replace(data=S_new))
                if enable_freeze:
                    T = new_state.T.data
                    T_top = jnp.where(
                        freeze_mask_static,
                        jnp.maximum(T[..., 0], T_freeze_C_static),
                        T[..., 0],
                    )
                    new_state = new_state._replace(
                        T=new_state.T.replace(
                            data=T.at[..., 0].set(T_top)))
                if enable_maxvel:
                    new_state = new_state._replace(
                        u=new_state.u.replace(
                            data=jnp.clip(new_state.u.data,
                                          -_maxvel_3d, _maxvel_3d)),
                        v=new_state.v.replace(
                            data=jnp.clip(new_state.v.data,
                                          -_maxvel_3d, _maxvel_3d)))

                if enable_sea_ice:
                    return (new_state, new_ice), None
                return new_state, None

            init = (state, ice_state) if enable_sea_ice else state
            final, _ = jax.lax.scan(
                step_body, init,
                jnp.arange(n_steps_block, dtype=jnp.int32),
            )
            return final  # (state, ice_state) when sea-ice on, else state

        return block_fn

    # Cache block functions by size to avoid recompilation.
    _block_fn_cache = {}

    def _get_block_fn(n):
        if n not in _block_fn_cache:
            _block_fn_cache[n] = _make_block_fn(n)
        return _block_fn_cache[n]

    return _get_block_fn


# ===========================================================================
# Diagnostics
# ===========================================================================

def _extract_scalars(state, grid_type, grid, z_coord):
    """Compute scalar diagnostics from ocean state."""
    if grid_type == "spectral":
        from legoesm.grids.gaussian import sh_synthesis_3d, sh_synthesis
        T_grid = sh_synthesis_3d(grid, state.T_hat.data).real
        S_grid = sh_synthesis_3d(grid, state.S_hat.data).real
        eta_grid = sh_synthesis(grid, state.eta_hat.data).real
        mask = np.asarray(state.land_mask_grid.data)
        sst = float(np.nanmean(np.where(mask > 0.5, np.asarray(T_grid[..., 0]), np.nan)))
        sss = float(np.nanmean(np.where(mask > 0.5, np.asarray(S_grid[..., 0]), np.nan)))
        ssh = float(np.nanmean(np.where(mask > 0.5, np.asarray(eta_grid), np.nan)))
        return {"SST": sst, "SSS": sss, "SSH": ssh}

    T = np.asarray(state.T.data)
    S = np.asarray(state.S.data)
        # dict per loop avoids sharing the sponge/target arrays across the
        # serial and sharded runs.
        js = run_omip._setup_jra55_forcing_state(
            args, grid, "latlon", z_coord=z_coord, T_woa=T_woa, S_woa=S_woa)
        js["_gpu_interp"] = (interp_mode == "gpu")
        return js

    common = dict(
        grid_type="latlon", grid=grid, z_coord=z_coord,
        dt=dt, n_steps=n_steps, diag_every=3,
        checkpoint_days=None, checkpoint_dir=None,
    )

    state0 = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    serial_final, _d, _w, ok_serial, _b = run_omip._run_omip_loop(
        model, state0, jra55_state=_fresh_js(), **common)
    assert ok_serial, "serial JRA55 loop reported not-ok"

    dev = create_latlon_mesh(n_devices=2)
    spmd_step = make_sharded_ocean_step(model, dev.mesh)
    model.prime_step_caches(state0)
    ss0 = shard_state_latlon(state0, dev.mesh)
    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
        model, ss0, jra55_state=_fresh_js(),
        spmd_step=spmd_step,
        spmd_gather=lambda st: gather_state_latlon(st, dev.mesh),
        spmd_shard_stack=partial(shard_forcing_stack_latlon, mesh=dev.mesh),
        **common)
    assert ok_spmd, "SPMD JRA55 loop reported not-ok"
    spmd_final = gather_state_latlon(spmd_final, dev.mesh)

    # Same re-association floor as the restoring-lane and step-level gates.
    for nm in ("u", "v", "eta", "T", "S"):
        np.testing.assert_allclose(
            np.asarray(getattr(spmd_final, nm).data),
            np.asarray(getattr(serial_final, nm).data),
            atol=2e-4, rtol=1e-3,
            err_msg=f"JRA55 {interp_mode}-interp SPMD {nm} mismatch")
    print("JRA55-PARITY-OK")


def _run_jra55_parity_subprocess(tmp_path, interp_mode: str):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    _root = str(_THIS.parents[2])
    env["PYTHONPATH"] = _root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(_THIS), "--jra55-parity-worker",
         str(tmp_path), interp_mode],
        env=env, capture_output=True, text=True, timeout=870,
    )
    assert proc.returncode == 0 and "JRA55-PARITY-OK" in proc.stdout, (
        f"rc={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
        f"stderr:\n{proc.stderr[-3000:]}")


@pytest.mark.timeout(900)
def test_jra55_block_loop_spmd_matches_serial_cpu_interp(tmp_path):
    _run_jra55_parity_subprocess(tmp_path, "cpu")
scripts/bench/bench_atm_latlon_spmd_scaling.py-2-atm step (make_sharded_atm_latlon_step / run_atm_latlon_spmd, the A1 work).
scripts/bench/bench_atm_latlon_spmd_scaling.py-3-
scripts/bench/bench_atm_latlon_spmd_scaling.py-4-Times the SHARDED step across an N-device ("lat",) mesh and reports per-step
scripts/bench/bench_atm_latlon_spmd_scaling.py-5-wall time + speedup vs 1 device.  The DEFAULT lane follows the M1 measurement
scripts/bench/bench_atm_latlon_spmd_scaling.py:6:contract (``metadata.timed_scan_blocks``): fused ``lax.scan`` blocks of
scripts/bench/bench_atm_latlon_spmd_scaling.py-7-``--steps`` steps with device sync only AROUND each block (``fused_step_ms``,
scripts/bench/bench_atm_latlon_spmd_scaling.py-8-slowest process across controllers) plus a SEPARATE individually-synced
scripts/bench/bench_atm_latlon_spmd_scaling.py-9-dispatch-latency probe (``step_latency_ms``) — never mixed.  The
scripts/bench/bench_atm_latlon_spmd_scaling.py-10-jit(shard_map) step is built once and cached by make_sharded_atm_latlon_step;
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-174-                        "from OMPI_COMM_WORLD_SIZE/RANK.")
scripts/bench/bench_atm_latlon_spmd_scaling.py-175-    args = p.parse_args()
scripts/bench/bench_atm_latlon_spmd_scaling.py-176-
scripts/bench/bench_atm_latlon_spmd_scaling.py-177-    # Validate the schedule BEFORE any model/device work: a zero/negative
scripts/bench/bench_atm_latlon_spmd_scaling.py:178:    # --steps would otherwise surface only as timed_scan_blocks' None
scripts/bench/bench_atm_latlon_spmd_scaling.py-179-    # headline (default lane) or an empty timed loop (segment mode) after
scripts/bench/bench_atm_latlon_spmd_scaling.py-180-    # the expensive build (the ocean twin's guard).
scripts/bench/bench_atm_latlon_spmd_scaling.py-181-    if args.steps < 1:
scripts/bench/bench_atm_latlon_spmd_scaling.py-182-        raise SystemExit(f"--steps must be >= 1, got {args.steps}")
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-246-
scripts/bench/bench_atm_latlon_spmd_scaling.py-247-    per_block_ms = None
scripts/bench/bench_atm_latlon_spmd_scaling.py-248-    completed_blocks = None
scripts/bench/bench_atm_latlon_spmd_scaling.py-249-    finite_ok = None   # default fused lane: no in-graph finite check -> null
scripts/bench/bench_atm_latlon_spmd_scaling.py:250:    timing = None   # metadata.timed_scan_blocks metrics (default lane only)
scripts/bench/bench_atm_latlon_spmd_scaling.py-251-    if seg_n > 0:
scripts/bench/bench_atm_latlon_spmd_scaling.py-252-        # Multi-controller: align every process before the timed loop so
scripts/bench/bench_atm_latlon_spmd_scaling.py-253-        # block wall times aren't skewed by startup jitter (and once after,
scripts/bench/bench_atm_latlon_spmd_scaling.py-254-        # so no process exits while peers still hold collectives in flight).
scripts/bench/bench_atm_latlon_spmd_scaling.py:255:        # (The default lane's fences live inside timed_scan_blocks.)
scripts/bench/bench_atm_latlon_spmd_scaling.py-256-        if jax.process_count() > 1:
scripts/bench/bench_atm_latlon_spmd_scaling.py-257-            from jax.experimental import multihost_utils
scripts/bench/bench_atm_latlon_spmd_scaling.py-258-            multihost_utils.sync_global_devices("atm_latlon_spmd_bench_start")
scripts/bench/bench_atm_latlon_spmd_scaling.py-259-        # Segment mode: each timed BLOCK is one compiled lax.scan of seg_n
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-297-        # host-synced loop measured dispatch+sync latency, not fused device
scripts/bench/bench_atm_latlon_spmd_scaling.py-298-        # throughput.  Dispatch latency stays measured SEPARATELY
scripts/bench/bench_atm_latlon_spmd_scaling.py-299-        # (``step_latency_ms``); multi-controller runs record the
scripts/bench/bench_atm_latlon_spmd_scaling.py-300-        # slowest-process block time + imbalance ratio.
scripts/bench/bench_atm_latlon_spmd_scaling.py:301:        from metadata import timed_scan_blocks
scripts/bench/bench_atm_latlon_spmd_scaling.py:302:        c, timing = timed_scan_blocks(
scripts/bench/bench_atm_latlon_spmd_scaling.py-303-            lambda st: step(st, args.dt), c,
scripts/bench/bench_atm_latlon_spmd_scaling.py-304-            block_steps=args.steps, n_blocks=args.blocks,
scripts/bench/bench_atm_latlon_spmd_scaling.py-305-            probe_steps=args.probe_steps,
scripts/bench/bench_atm_latlon_spmd_scaling.py-306-            sync_label="atm_latlon_spmd_bench")
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-353-            else (args.single_dev_fused_ms if valid else None)),
scripts/bench/bench_atm_latlon_spmd_scaling.py-354-        halo_messages_per_step=comm_rec["halo_messages_per_step"],
scripts/bench/bench_atm_latlon_spmd_scaling.py-355-        halo_bytes_per_step=comm_rec["halo_bytes_per_step"],
scripts/bench/bench_atm_latlon_spmd_scaling.py-356-        n_reductions_per_step=_nred,
scripts/bench/bench_atm_latlon_spmd_scaling.py:357:        # rank imbalance is measured by timed_scan_blocks (default lane);
scripts/bench/bench_atm_latlon_spmd_scaling.py-358-        # the segment lane records no cross-process block gather -> null.
scripts/bench/bench_atm_latlon_spmd_scaling.py-359-        rank_imbalance=(float(timing["rank_imbalance"])
scripts/bench/bench_atm_latlon_spmd_scaling.py-360-                        if timing is not None else None),
scripts/bench/bench_atm_latlon_spmd_scaling.py-361-        latency_us=args.comm_latency_us,
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-389-            per_step_ms=[round(x, 2) for x in per_step_ms],
scripts/bench/bench_atm_latlon_spmd_scaling.py-390-            per_block_ms=[round(x, 2) for x in per_block_ms],
scripts/bench/bench_atm_latlon_spmd_scaling.py-391-        )
scripts/bench/bench_atm_latlon_spmd_scaling.py-392-    else:
scripts/bench/bench_atm_latlon_spmd_scaling.py:393:        # Default lane: the timed_scan_blocks metrics (fused_step_ms,
scripts/bench/bench_atm_latlon_spmd_scaling.py-394-        # step_latency_ms, block_ms, parallel_block_ms, rank_imbalance, ...
scripts/bench/bench_atm_latlon_spmd_scaling.py-395-        # — the M1 measurement contract).
scripts/bench/bench_atm_latlon_spmd_scaling.py-396-        rec.update(**timing)
scripts/bench/bench_atm_latlon_spmd_scaling.py-397-    # Flat aggregator-compatible identity + metric fields: without a
--
tests/bench/test_timed_scan_blocks.py:1:"""timed_scan_blocks + git_sha: the measurement-contract helper (M1 gap #1/#2).
tests/bench/test_timed_scan_blocks.py-2-
tests/bench/test_timed_scan_blocks.py-3-Runs on CPU with a trivial jitted step — validates the schedule accounting
tests/bench/test_timed_scan_blocks.py-4-(compile/probe/blocks executed counts, invalid schedules raise instead of
tests/bench/test_timed_scan_blocks.py-5-being silently clamped), the separate latency-vs-fused reporting, the
--
tests/bench/test_timed_scan_blocks.py-13-import jax.numpy as jnp
tests/bench/test_timed_scan_blocks.py-14-import pytest
tests/bench/test_timed_scan_blocks.py-15-
tests/bench/test_timed_scan_blocks.py-16-sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))
tests/bench/test_timed_scan_blocks.py:17:from metadata import git_sha, scaling_metadata, timed_scan_blocks  # noqa: E402
tests/bench/test_timed_scan_blocks.py-18-
tests/bench/test_timed_scan_blocks.py-19-
tests/bench/test_timed_scan_blocks.py-20-def _advance(s):
tests/bench/test_timed_scan_blocks.py-21-    return jax.tree_util.tree_map(lambda x: x * 0.999 + 0.001, s)
--
tests/bench/test_timed_scan_blocks.py-30-        out = dict(s)
tests/bench/test_timed_scan_blocks.py-31-        out["n"] = s["n"] + 1.0
tests/bench/test_timed_scan_blocks.py-32-        out["a"] = s["a"] * 0.999
tests/bench/test_timed_scan_blocks.py-33-        return out
tests/bench/test_timed_scan_blocks.py:34:    s, m = timed_scan_blocks(adv, s0, block_steps=5, n_blocks=2, probe_steps=3)
tests/bench/test_timed_scan_blocks.py-35-    # 1 compile + 3 probe + 2*5 block = 14 steps
tests/bench/test_timed_scan_blocks.py-36-    assert float(s["n"]) == 14.0
tests/bench/test_timed_scan_blocks.py-37-    assert m["block_steps"] == 5 and m["n_blocks"] == 2 and m["probe_steps"] == 3
tests/bench/test_timed_scan_blocks.py-38-    assert len(m["block_ms"]) == 2
tests/bench/test_timed_scan_blocks.py-39-
tests/bench/test_timed_scan_blocks.py-40-
tests/bench/test_timed_scan_blocks.py-41-def test_metrics_shape_and_separation():
tests/bench/test_timed_scan_blocks.py-42-    s0 = {"x": jnp.ones((8,))}
tests/bench/test_timed_scan_blocks.py:43:    s, m = timed_scan_blocks(_advance, s0, block_steps=4, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-44-                             probe_steps=2)
tests/bench/test_timed_scan_blocks.py-45-    for k in ("compile_ms", "scan_compile_ms", "step_latency_ms",
tests/bench/test_timed_scan_blocks.py-46-              "fused_step_ms", "parallel_block_ms", "rank_imbalance",
tests/bench/test_timed_scan_blocks.py-47-              "rank_imbalance_per_block"):
--
tests/bench/test_timed_scan_blocks.py-61-    """block_steps=0 is the documented zero-length parity path: the blocks
tests/bench/test_timed_scan_blocks.py-62-    run (fences/collectives matched) but a zero-step block has NO per-step
tests/bench/test_timed_scan_blocks.py-63-    time — the headline must be an honest null, not block_ms/1."""
tests/bench/test_timed_scan_blocks.py-64-    s0 = {"x": jnp.ones(())}
tests/bench/test_timed_scan_blocks.py:65:    _, m = timed_scan_blocks(_advance, s0, block_steps=0, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-66-                             probe_steps=0)
tests/bench/test_timed_scan_blocks.py-67-    assert m["fused_step_ms"] is None
tests/bench/test_timed_scan_blocks.py-68-    assert m["block_steps"] == 0 and m["n_blocks"] == 1
tests/bench/test_timed_scan_blocks.py-69-    assert len(m["block_ms"]) == 1
--
tests/bench/test_timed_scan_blocks.py-74-    schedule DIFFERENT from the recorded one (codex batch4): invalid values
tests/bench/test_timed_scan_blocks.py-75-    must raise so the record always equals the execution."""
tests/bench/test_timed_scan_blocks.py-76-    s0 = {"x": jnp.ones(())}
tests/bench/test_timed_scan_blocks.py-77-    with pytest.raises(ValueError, match="n_blocks"):
tests/bench/test_timed_scan_blocks.py:78:        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=0,
tests/bench/test_timed_scan_blocks.py-79-                          probe_steps=1)
tests/bench/test_timed_scan_blocks.py-80-    with pytest.raises(ValueError, match="n_blocks"):
tests/bench/test_timed_scan_blocks.py:81:        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=-2,
tests/bench/test_timed_scan_blocks.py-82-                          probe_steps=1)
tests/bench/test_timed_scan_blocks.py-83-    with pytest.raises(ValueError, match="probe_steps"):
tests/bench/test_timed_scan_blocks.py:84:        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-85-                          probe_steps=-1)
tests/bench/test_timed_scan_blocks.py-86-    with pytest.raises(ValueError, match="block_steps"):
tests/bench/test_timed_scan_blocks.py:87:        timed_scan_blocks(_advance, s0, block_steps=-1, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-88-                          probe_steps=0)
tests/bench/test_timed_scan_blocks.py-89-
tests/bench/test_timed_scan_blocks.py-90-
tests/bench/test_timed_scan_blocks.py-91-def test_seed_state_not_mutated_by_precompile():
--
tests/bench/test_timed_scan_blocks.py-94-    the timed blocks must be byte-identical to freshly advanced ones."""
tests/bench/test_timed_scan_blocks.py-95-    import numpy as np
tests/bench/test_timed_scan_blocks.py-96-    s0 = {"x": jnp.arange(6.0)}
tests/bench/test_timed_scan_blocks.py-97-    ref = np.asarray(s0["x"]).copy()
tests/bench/test_timed_scan_blocks.py:98:    s, m = timed_scan_blocks(_advance, dict(s0), block_steps=1, n_blocks=1,
tests/bench/test_timed_scan_blocks.py-99-                             probe_steps=0)
tests/bench/test_timed_scan_blocks.py-100-    # 1 compile + 0 probe + 1x1 block = 2 steps from the ORIGINAL seed.
tests/bench/test_timed_scan_blocks.py-101-    want = ref
tests/bench/test_timed_scan_blocks.py-102-    for _ in range(2):
--
scripts/bench/bench_ocean_mpas_scaling.py-21-the shared self-describing metadata (transport resolves to mpi4jax via
scripts/bench/bench_ocean_mpas_scaling.py-22-n_ranks > process_count) + voronoi partition-quality metrics.
scripts/bench/bench_ocean_mpas_scaling.py-23-
scripts/bench/bench_ocean_mpas_scaling.py-24-M1 measurement contract (scaling-M3d increment-1): the headline number is a
scripts/bench/bench_ocean_mpas_scaling.py:25:fused ``lax.scan`` block (``metadata.timed_scan_blocks``; per-step
scripts/bench/bench_ocean_mpas_scaling.py-26-dispatch latency probed SEPARATELY), cross-rank MAX-reduced, and it is what
scripts/bench/bench_ocean_mpas_scaling.py-27-the aggregator-facing ``steady_median_ms`` carries (the same deliberate
scripts/bench/bench_ocean_mpas_scaling.py-28-naming as ``bench_ocean_latlon_spmd_scaling``); the host-synced gate-loop
scripts/bench/bench_ocean_mpas_scaling.py-29-median is dispatch+sync LATENCY and is recorded only under
--
scripts/bench/bench_ocean_mpas_scaling.py-66-
scripts/bench/bench_ocean_mpas_scaling.py-67-from metadata import (  # noqa: E402
scripts/bench/bench_ocean_mpas_scaling.py-68-    annotate_incomplete,
scripts/bench/bench_ocean_mpas_scaling.py-69-    scaling_metadata,
scripts/bench/bench_ocean_mpas_scaling.py:70:    timed_scan_blocks,
scripts/bench/bench_ocean_mpas_scaling.py-71-    wet_cell_metrics,
scripts/bench/bench_ocean_mpas_scaling.py-72-)
scripts/bench/bench_ocean_mpas_scaling.py-73-
scripts/bench/bench_ocean_mpas_scaling.py-74-#: Parity tolerances (gathered MPI vs serial, f64/f32) — the re-association
--
scripts/bench/bench_ocean_mpas_scaling.py-526-
scripts/bench/bench_ocean_mpas_scaling.py-527-        # jit the COMPOSED step+exchange (one dispatch per step; the
scripts/bench/bench_ocean_mpas_scaling.py-528-        # sendrecv wrapper always runs traced, exactly as the atmosphere
scripts/bench/bench_ocean_mpas_scaling.py-529-        # step and the distributed PCG use it).  ``_step_impl`` avoids a
scripts/bench/bench_ocean_mpas_scaling.py:530:        # nested-JIT boundary inside this wrapper; timed_scan_blocks
scripts/bench/bench_ocean_mpas_scaling.py-531-        # re-traces the whole thing into the fused scan — same graph.
scripts/bench/bench_ocean_mpas_scaling.py-532-        @jax.jit
scripts/bench/bench_ocean_mpas_scaling.py-533-        def advance(st):
scripts/bench/bench_ocean_mpas_scaling.py-534-            return exchange_state_mpas_ocean(
--
scripts/bench/bench_ocean_mpas_scaling.py-648-    # allgathered so the parallel time of block b is the SLOWEST rank in
scripts/bench/bench_ocean_mpas_scaling.py-649-    # that block (the same slowest-rank convention as the per-step MAX).
scripts/bench/bench_ocean_mpas_scaling.py-650-    fused = None
scripts/bench/bench_ocean_mpas_scaling.py-651-    if args.block_steps > 0:
scripts/bench/bench_ocean_mpas_scaling.py:652:        state, _t = timed_scan_blocks(
scripts/bench/bench_ocean_mpas_scaling.py-653-            advance, state,
scripts/bench/bench_ocean_mpas_scaling.py-654-            block_steps=args.block_steps, n_blocks=args.blocks,
scripts/bench/bench_ocean_mpas_scaling.py-655-            probe_steps=args.probe_steps,
scripts/bench/bench_ocean_mpas_scaling.py-656-            sync_label="ocean_mpas_mpi_bench")
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-461-    # throughput.  Per-step dispatch latency is still measured, SEPARATELY,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-462-    # by an individually-synced probe (``step_latency_ms``); multi-controller
scripts/bench/bench_ocean_latlon_spmd_scaling.py-463-    # runs record the SLOWEST-process block time + imbalance ratio (a
scripts/bench/bench_ocean_latlon_spmd_scaling.py-464-    # straggler band is invisible to a single process's clock).
scripts/bench/bench_ocean_latlon_spmd_scaling.py:465:    from metadata import timed_scan_blocks
scripts/bench/bench_ocean_latlon_spmd_scaling.py-466-    # Under --parity-gate the serial reference above ran EXACTLY args.steps
scripts/bench/bench_ocean_latlon_spmd_scaling.py-467-    # steps, so the SPMD arm must execute the same count: 1 compile step +
scripts/bench/bench_ocean_latlon_spmd_scaling.py-468-    # one (args.steps - 1)-long block, no probe.  Timing from a parity smoke
scripts/bench/bench_ocean_latlon_spmd_scaling.py-469-    # run is not reported as a scaling number anyway (steps are capped).
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-474-    # aux threads the sharded geometry stacks through the jit boundary as
scripts/bench/bench_ocean_latlon_spmd_scaling.py-475-    # an ARGUMENT (outer-trace constants of non-addressable arrays are
scripts/bench/bench_ocean_latlon_spmd_scaling.py-476-    # unfetchable — see make_sharded_ocean_step's aux note).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-477-    _aux = getattr(step, "aux", None)
scripts/bench/bench_ocean_latlon_spmd_scaling.py:478:    s, timing = timed_scan_blocks(
scripts/bench/bench_ocean_latlon_spmd_scaling.py-479-        (lambda st, aux: step(st, args.dt, aux=aux)) if _aux is not None
scripts/bench/bench_ocean_latlon_spmd_scaling.py-480-        else (lambda st: step(st, args.dt)),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-481-        s,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-482-        block_steps=_blk, n_blocks=_nblk, probe_steps=_probe,

exec
/usr/bin/bash -lc "nl -ba packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py | sed -n '880,1000p'
nl -ba scripts/bench/metadata.py | sed -n '890,1080p'
nl -ba scripts/bench/bench_ocean_latlon_spmd_scaling.py | sed -n '465,490p'
nl -ba scripts/run/run_omip.py | sed -n '2355,2460p'
nl -ba scripts/run/run_omip.py | sed -n '2558,2730p'
nl -ba scripts/run/run_omip.py | sed -n '3370,3475p'
nl -ba scripts/run/run_omip.py | sed -n '4760,4865p'
nl -ba scripts/run/run_omip_core2.py | sed -n '5555,5600p;5655,5675p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 251ms:
   880	                # (codex r1 #2).
   881	                raise ValueError(
   882	                    f"sharded ocean step: forcing leaf "
   883	                    f"{jax.tree_util.keystr(path)} is 1-D "
   884	                    f"(shape {tuple(leaf.shape)}); forcing must be "
   885	                    f"cell-centered (n_lat, n_lon[, nlev]) arrays or "
   886	                    f"scalars.")
   887	            if nd >= 2 and int(leaf.shape[0]) != n_lat_global:
   888	                raise ValueError(
   889	                    f"sharded ocean step: forcing leaf {jax.tree_util.keystr(path)} "
   890	                    f"has leading dim {leaf.shape[0]} != n_lat "
   891	                    f"({n_lat_global}); forcing must be cell-centered "
   892	                    f"(n_lat, n_lon[, nlev]) to shard on the lat axis.")
   893	
   894	    def sharded_step(state, dt, freshwater=None, surface_forcing=None,
   895	                     sponge=None, t_seconds=None, aux=None):
   896	        # ``aux`` (codex/2-proc repro 2026-08-03): the geometry + vmask
   897	        # stacks are SHARDED global arrays; when this wrapper runs INSIDE
   898	        # an outer trace (a bench/driver ``jit``/``scan`` over the step —
   899	        # jit-of-jit inlines the inner call), concrete closure arrays
   900	        # become OUTER-TRACE CONSTANTS and jax's MLIR constant handler
   901	        # tries to fetch their value — impossible for non-addressable
   902	        # arrays (RuntimeError: 'Fetching value ... non-addressable'; the
   903	        # multicontroller lane has been broken this way since the
   904	        # #1370-iii stack sharding). Callers that wrap the step in their
   905	        # own jit MUST thread ``step.aux`` through their jit boundary as
   906	        # an ARGUMENT and pass it back here.
   907	        # ONE forcing operand: None fields drop out of the pytree structure,
   908	        # so specs derived by tree.map skip them automatically and the
   909	        # structure key below distinguishes every None<->array combination.
   910	        forcing = (freshwater, surface_forcing, sponge, t_seconds)
   911	        _validate_forcing_layout((freshwater, surface_forcing, sponge))
   912	        # Cache key = the state's AND forcing's pytree STRUCTURE, plus the
   913	        # forcing leaves' RANKS: in_specs/out_specs are derived from them,
   914	        # so a later call with a different structure (an optional field
   915	        # flipping None <-> Field, a sea-ice lane populating sf.salt_flux,
   916	        # the restoring lane passing no forcing at all) OR a same-field
   917	        # rank change (SpongeForcing.gamma is legitimately 2-D horizontal
   918	        # OR 3-D full-rank — same structure, different _lat_spec; codex r1
   919	        # #1) must rebuild the shard_map rather than reuse stale specs.
   920	        forcing_ndims = tuple(
   921	            int(getattr(leaf, "ndim", np.ndim(leaf)))
   922	            for leaf in jax.tree.leaves(forcing))
   923	        # The SPMD fused-halo switch is read at TRACE time inside the pad
   924	        # dispatch — flipping LEGOESM_LATLON_SPMD_FUSED_HALO on a reused
   925	        # step object must rebuild the shard_map, not reuse a stale jaxpr
   926	        # (codex, audit item 7).
   927	        import os as _os
   928	
   929	        _fused_halo = _os.environ.get(
   930	            "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
   931	        key = (jax.tree.structure(state), jax.tree.structure(forcing),
   932	               forcing_ndims, _fused_halo)
   933	        fn = _cache.get(key)
   934	        if fn is None:
   935	            in_spec = jax.tree.map(_lat_spec, state)
   936	            # Forcing leaves are cell-centered -> plain lat-band specs;
   937	            # scalars (t_seconds) replicate, exactly like dt.
   938	            forcing_spec = jax.tree.map(_lat_spec, forcing)
   939	            # Stage (iii): the stacks are banded on their leading axis, so
   940	            # the shard_map spec matches their P("lat") placement (the body
   941	            # indexes its local (1, ...) slab at [0]).
   942	            geom_spec = jax.tree.map(lambda _x: P("lat"), geom_stacks)
   943	            vmask_spec = P("lat")
   944	            # JAX >= 0.8 top-level shard_map takes ``check_vma`` (the
   945	            # replication check); the band halo reads neighbour-rank data so
   946	            # disable it (same as the validated PCG / halo-parity shard_maps).
   947	            fn = jax.jit(shard_map(
   948	                _body,
   949	                mesh=mesh,
   950	                in_specs=(in_spec, forcing_spec, geom_spec, vmask_spec, P()),
   951	                out_specs=in_spec,
   952	                check_vma=False,
   953	            ))
   954	            _cache[key] = fn
   955	        # Arm the SPMD halo backend ONLY around the call, then RESTORE the
   956	        # previous backend (codex finding): leaving it globally armed makes a
   957	        # later serial/full-domain ocean call take SPMD-only branches
   958	        # (axis_index / ppermute / psum in pad_with_pole_bc_lat, conservation,
   959	        # eta_floor) OUTSIDE a shard_map -> crash. The FIRST call traces with
   960	        # the backend armed (baking the SPMD halo/reduction ops into the
   961	        # compiled program); later calls reuse the cached compile, and the
   962	        # arm/restore keeps any interleaved serial path untouched.
   963	        # Save+restore the FULL backend state (backend + MPI topology + SPMD
   964	        # mesh) so a prior "mpi"/"spmd" backend is restored intact: activate_*
   965	        # clears the MPI topology, and set_halo_backend("mpi") REQUIRES a
   966	        # topology (codex).
   967	        from legoesm.grids.halo import (
   968	            get_halo_backend, get_mpi_topology, get_spmd_mesh,
   969	            set_halo_backend, set_spmd_mesh,
   970	        )
   971	        _prev_backend = get_halo_backend()
   972	        _prev_topo = get_mpi_topology()
   973	        _prev_mesh = get_spmd_mesh()
   974	        activate_latlon_spmd_halo(mesh)
   975	        try:
   976	            _geom, _vmask = aux if aux is not None else (geom_stacks,
   977	                                                        vmask_stack)
   978	            return fn(state, forcing, _geom, _vmask, jnp.asarray(dt))
   979	        finally:
   980	            set_spmd_mesh(_prev_mesh)
   981	            set_halo_backend(_prev_backend, _prev_topo)
   982	
   983	    # Expose the stacks so outer-jit callers can pass them as arguments
   984	    # (see the ``aux`` note in the signature).
   985	    sharded_step.aux = (geom_stacks, vmask_stack)
   986	    return sharded_step
   987	
   988	
   989	def make_sharded_ocean_step_global(model, mesh):
   990	    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
   991	    that takes a GLOBAL (single-device-layout) state + forcing and returns a
   992	    GLOBAL state — the minimal-diff driver entry point.
   993	
   994	    Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
   995	    (:func:`shard_state_latlon` + :func:`shard_forcing_latlon`), runs the lat-band
   996	    SPMD step, then gathers the state OUT (:func:`gather_state_latlon`).  This lets
   997	    the OMIP host loop keep operating on a normal full-domain state — the per-step
   998	    host BCs (SSS restore, prognostic ice, geothermal, BBL, nudge) see the
   999	    gathered global state UNCHANGED — at the cost of a per-step gather/scatter
  1000	    (acceptable for the host-coupled OMIP driver; the pure-dynamics inner loop
   890	def timed_scan_blocks(
   891	    advance,
   892	    state,
   893	    *,
   894	    block_steps: int,
   895	    n_blocks: int = 2,
   896	    probe_steps: int = 3,
   897	    sync_label: str = "timed_scan_blocks",
   898	    aux=None,
   899	):
   900	    """Measurement-contract timing: fused ``lax.scan`` blocks + probe latency.
   901	
   902	    The trustworthy production-like number is a MULTI-STEP ``lax.scan`` block
   903	    with device synchronization only AROUND the block (per-step host sync in a
   904	    Python loop measures dispatch+sync latency, not fused device throughput —
   905	    the audited anti-pattern in the SPMD benches).  Per-step dispatch latency
   906	    is still physically meaningful (drivers that must step one-at-a-time pay
   907	    it), so it is measured SEPARATELY by a short individually-synced probe and
   908	    reported as ``step_latency_ms`` — never mixed into the fused number.
   909	
   910	    Multi-controller runs additionally allgather EVERY process's full
   911	    per-block vector and reduce per block: the parallel time of block ``b``
   912	    is the SLOWEST process in that block.  A per-process median gathered
   913	    alone would hide an alternating straggler (every rank's median can be
   914	    fast even though every block has a slow rank — codex batch4).
   915	
   916	    Parameters
   917	    ----------
   918	    advance
   919	        ``advance(state) -> state`` — ONE production step with all static
   920	        knobs (dt, forcing, ...) closed over.  May itself be jitted; it is
   921	        re-traced INTO the fused scan (same graph, no double-jit penalty).
   922	        MUST NOT donate its input buffers (``donate_argnums``): the scan
   923	        pre-compile below runs on a SHALLOW pytree copy whose leaves ALIAS
   924	        the live seed state — donation would invalidate the seed's buffers
   925	        mid-benchmark.
   926	    state
   927	        Initial (already sharded, post-seed) model state pytree.
   928	    block_steps
   929	        Steps per fused ``lax.scan`` block (the amortizing window).
   930	        ``>= 0``; ``0`` is the documented ZERO-LENGTH parity path (the
   931	        block advances nothing and ``fused_step_ms`` is ``None`` — a
   932	        zero-step block has no per-step time).
   933	    n_blocks
   934	        Timed blocks (``>= 1``); per-block times expose drift.
   935	    probe_steps
   936	        Individually host-synced steps (``>= 0``) for the separate
   937	        dispatch-latency probe (each one costs a device round-trip).
   938	    sync_label
   939	        Base label for the multi-controller ``sync_global_devices`` fences.
   940	
   941	    Invalid schedule values raise ``ValueError`` — never silently clamped,
   942	    so the recorded schedule is ALWAYS the executed schedule.  Multi-process
   943	    runs first allgather-verify the schedule tuple itself: processes that
   944	    disagree on ``(block_steps, n_blocks, probe_steps)`` would enter
   945	    DIFFERENT named-fence schedules and deadlock; the verification is the
   946	    one collective every process reaches, so a mismatch raises everywhere.
   947	
   948	    Returns
   949	    -------
   950	    (state, metrics) — final state (compile + probe + all blocks advanced)
   951	    and a dict:
   952	      ``compile_ms``           first-call cost of ``advance`` (trace+compile)
   953	      ``scan_compile_ms``      first-call cost of the fused scan itself
   954	      ``step_latency_ms``      median individually-synced per-step wall time
   955	      ``block_ms``             per-block wall times, THIS process (list)
   956	      ``parallel_block_ms``    per-block wall times of the PARALLEL step:
   957	                               max over processes, per block (== ``block_ms``
   958	                               for a single process)
   959	      ``fused_step_ms``        median(parallel_block_ms)/block_steps — the
   960	                               headline (``None`` when ``block_steps == 0``)
   961	      ``rank_imbalance``       median over blocks of the per-block
   962	                               max/median-across-processes ratio (>= 1.0;
   963	                               exactly 1.0 for a single process)
   964	      ``rank_imbalance_per_block``  the per-block ratios themselves
   965	      ``block_steps``/``n_blocks``/``probe_steps``  the EXECUTED schedule
   966	    """
   967	    import time
   968	
   969	    import jax
   970	    import numpy as np
   971	
   972	    multi = jax.process_count() > 1
   973	    if multi:
   974	        # Collectively verify the schedule BEFORE compilation or any
   975	        # schedule-dependent fence: this allgather is the single collective
   976	        # every process reaches first, so on a mismatch EVERY process sees
   977	        # the same gathered table and raises together instead of hanging in
   978	        # mismatched named fences (codex batch4 deadlock hazard).
   979	        from jax.experimental import multihost_utils
   980	        _sched = np.asarray(multihost_utils.process_allgather(
   981	            np.array([block_steps, n_blocks, probe_steps], dtype=np.int64)))
   982	        if not bool((_sched == _sched[0]).all()):
   983	            raise ValueError(
   984	                "timed_scan_blocks: processes disagree on the schedule "
   985	                f"(block_steps, n_blocks, probe_steps) = {_sched.tolist()} "
   986	                "per process — a mismatched schedule deadlocks in the "
   987	                "named fences.")
   988	    # No silent rewrites: invalid values raise; the recorded schedule IS
   989	    # the executed schedule.  block_steps == 0 stays legal (the documented
   990	    # zero-length parity path).
   991	    if block_steps < 0:
   992	        raise ValueError(
   993	            f"timed_scan_blocks: block_steps must be >= 0, got {block_steps}")
   994	    if n_blocks < 1:
   995	        raise ValueError(
   996	            f"timed_scan_blocks: n_blocks must be >= 1, got {n_blocks}")
   997	    if probe_steps < 0:
   998	        raise ValueError(
   999	            f"timed_scan_blocks: probe_steps must be >= 0, got {probe_steps}")
  1000	
  1001	    def _block(tree):
  1002	        jax.block_until_ready(jax.tree_util.tree_leaves(tree))
  1003	
  1004	    def _fence(tag: str):
  1005	        if multi:
  1006	            from jax.experimental import multihost_utils
  1007	            multihost_utils.sync_global_devices(f"{sync_label}_{tag}")
  1008	
  1009	    # Normalize to a 2-arg advance (see the aux note below): top-level
  1010	    # calls here are outside any trace, so passing aux is just an argument.
  1011	    if aux is None:
  1012	        def _advance(carry, _aux):
  1013	            return advance(carry)
  1014	    else:
  1015	        _advance = advance
  1016	
  1017	    # --- 1. compile (first call of advance, separated from all timing) ---
  1018	    _fence("compile_start")
  1019	    t0 = time.perf_counter()
  1020	    state = _advance(state, aux)
  1021	    _block(state)
  1022	    compile_ms = (time.perf_counter() - t0) * 1e3
  1023	
  1024	    # --- 2. dispatch-latency probe: individually synced steps, reported
  1025	    # separately (NEVER mixed into the fused number) ---
  1026	    probe_ms = []
  1027	    for _ in range(probe_steps):
  1028	        t0 = time.perf_counter()
  1029	        state = _advance(state, aux)
  1030	        _block(state)
  1031	        probe_ms.append((time.perf_counter() - t0) * 1e3)
  1032	    step_latency_ms = float(np.median(probe_ms)) if probe_ms else float("nan")
  1033	
  1034	    # --- 3. fused scan block (dtype-stable carry, the OM pattern) ---
  1035	    input_dtypes = jax.tree_util.tree_map(
  1036	        lambda x: x.dtype if hasattr(x, "dtype") else None, state)
  1037	
  1038	    # ``aux``: extra pytree threaded through the jit boundary as an ARGUMENT
  1039	    # (never a closure capture). Needed when ``advance`` internally calls a
  1040	    # jit function over SHARDED-global auxiliary arrays (the ocean band
  1041	    # geometry stacks): jit-of-jit inlines the inner call, so concrete
  1042	    # closure arrays would become outer-trace CONSTANTS whose value the MLIR
  1043	    # handler cannot fetch for non-addressable arrays (2026-08-03 repro).
  1044	    @jax.jit
  1045	    def _scan_run(st, _aux):
  1046	        def _body(carry, _):
  1047	            new = _advance(carry, _aux)
  1048	            new = jax.tree_util.tree_map(
  1049	                lambda x, d: x.astype(d)
  1050	                if d is not None and hasattr(x, "astype") else x,
  1051	                new, input_dtypes)
  1052	            return new, None
  1053	        return jax.lax.scan(_body, st, None, length=block_steps)[0]
  1054	
  1055	    # Pre-compile the scan on a SHALLOW pytree copy: the leaves ALIAS the
  1056	    # seed state's arrays (no data copy) — sufficient AND safe because the
  1057	    # pre-call only needs matching shapes/dtypes/shardings to warm the
  1058	    # compile cache, jitted execution is pure, and ``advance`` is
  1059	    # contract-bound not to donate buffers (see the docstring).  The seed
  1060	    # value itself is untouched; every process executes the same collective
  1061	    # schedule — counts stay matched.
  1062	    _pre = jax.tree_util.tree_map(lambda x: x, state)
  1063	    t0 = time.perf_counter()
  1064	    _pre_out = _scan_run(_pre, aux)
  1065	    _block(_pre_out)
  1066	    scan_compile_ms = (time.perf_counter() - t0) * 1e3
  1067	    del _pre, _pre_out
  1068	
  1069	    block_ms = []
  1070	    for b in range(n_blocks):
  1071	        _fence(f"block{b}_start")
  1072	        t0 = time.perf_counter()
  1073	        state = _scan_run(state, aux)
  1074	        _block(state)
  1075	        block_ms.append((time.perf_counter() - t0) * 1e3)
  1076	    _fence("blocks_end")
  1077	
  1078	    # Slowest-rank statistics from the FULL per-block vectors (equal length
  1079	    # everywhere — the schedule was collectively verified above).  The
  1080	    # parallel time of block b is the slowest process IN that block; a
   465	    from metadata import timed_scan_blocks
   466	    # Under --parity-gate the serial reference above ran EXACTLY args.steps
   467	    # steps, so the SPMD arm must execute the same count: 1 compile step +
   468	    # one (args.steps - 1)-long block, no probe.  Timing from a parity smoke
   469	    # run is not reported as a scaling number anyway (steps are capped).
   470	    if args.parity_gate:
   471	        _blk, _nblk, _probe = max(0, args.steps - 1), 1, 0
   472	    else:
   473	        _blk, _nblk, _probe = args.steps, args.blocks, args.probe_steps
   474	    # aux threads the sharded geometry stacks through the jit boundary as
   475	    # an ARGUMENT (outer-trace constants of non-addressable arrays are
   476	    # unfetchable — see make_sharded_ocean_step's aux note).
   477	    _aux = getattr(step, "aux", None)
   478	    s, timing = timed_scan_blocks(
   479	        (lambda st, aux: step(st, args.dt, aux=aux)) if _aux is not None
   480	        else (lambda st: step(st, args.dt)),
   481	        s,
   482	        block_steps=_blk, n_blocks=_nblk, probe_steps=_probe,
   483	        sync_label="ocean_latlon_spmd_bench", aux=_aux)
   484	
   485	    # Post-run ZERO-FORCING residual probe eligibility (audit item 6):
   486	    # needs the gathered global final state on ONE process; multicontroller
   487	    # runs skip it (zero_forcing_probe_measured=False, never faked).
   488	    _probe_residual = (args.baro_solver == "implicit_cn"
   489	                       and jax.process_count() == 1)
   490	
  2355	    # 3D velocity clip — caps ALL velocity components (barotropic +
  2356	    # baroclinic) after each step.  The barotropic-only MAXVEL inside
  2357	    # the split-explicit solver doesn't prevent baroclinic blowup.
  2358	    _maxvel_3d = model.config.barotropic.maxvel_barotropic
  2359	    enable_maxvel = _maxvel_3d > 0.0
  2360	
  2361	    # Lat-band SPMD (--enable-latlon-spmd): the scan body's dynamics step
  2362	    # runs through the sharded wrapper (same forcing kwargs as _step_impl;
  2363	    # the wrapper's cache/arm-restore Python runs ONCE at block trace).
  2364	    # Sea ice is refused upstream (the ice tile is not SPMD-audited yet).
  2365	    if spmd_step is not None and enable_sea_ice:
  2366	        raise ValueError(
  2367	            "spmd_step + prognostic sea ice is unsupported "
  2368	            "(run_omip_single refuses --jra55-sea-ice with "
  2369	            "--enable-latlon-spmd).")
  2370	    _dyn_step = (spmd_step if spmd_step is not None
  2371	                 else lambda st, d, **kw: model._step_impl(st, d, **kw))
  2372	
  2373	    @jax.jit
  2374	    def block_fn(state, atm_stack, runoff_stack, block_start_step,
  2375	                 ice_state=None):
  2376	        def step_body(carry, idx):
  2377	            if enable_sea_ice:
  2378	                state_in, ice_in = carry
  2379	            else:
  2380	                state_in = carry
  2381	            atm = AtmToSurface(
  2382	                sw_down=atm_stack["sw_down"][idx],
  2383	                lw_down=atm_stack["lw_down"][idx],
  2384	                precip_total=atm_stack["precip_total"][idx],
  2385	                precip_snow=atm_stack["precip_snow"][idx],
  2386	                T_lowest=atm_stack["T_lowest"][idx],
  2387	                q_lowest=atm_stack["q_lowest"][idx],
  2388	                u_lowest=atm_stack["u_lowest"][idx],
  2389	                v_lowest=atm_stack["v_lowest"][idx],
  2390	                p_lowest=atm_stack["p_lowest"][idx],
  2391	                p_surface=atm_stack["p_surface"][idx],
  2392	                rho_lowest=atm_stack["rho_lowest"][idx],
  2393	                cos_zenith=atm_stack["cos_zenith"][idx],
  2394	                co2_ppmv=jnp.asarray(co2_ppmv, dtype=atm_stack["T_lowest"].dtype),
  2395	                has_radiation=jnp.asarray(1.0, dtype=atm_stack["T_lowest"].dtype),
  2396	                has_precipitation=jnp.asarray(1.0, dtype=atm_stack["T_lowest"].dtype),
  2397	            )
  2398	
  2399	            sst_K = state_in.T.data[..., 0] + _const.T_freeze
  2400	            u_o = jnp.zeros_like(sst_K)
  2401	            v_o = jnp.zeros_like(sst_K)
  2402	            tile = ocean_tile_response(atm, sst_K, u_o, v_o, coupler_cfg)
  2403	
  2404	            # Wind-stress spinup ramp (gated at compile time).
  2405	            if enable_ramp:
  2406	                abs_step = block_start_step + idx
  2407	                t_sim = abs_step.astype(jnp.float64) * dt
  2408	                ramp = jnp.minimum(1.0, t_sim / T_ramp_seconds)
  2409	                tau_x = tile.tau_x * ramp
  2410	                tau_y = tile.tau_y * ramp
  2411	            else:
  2412	                tau_x = tile.tau_x
  2413	                tau_y = tile.tau_y
  2414	
  2415	            sw_net = atm.sw_down * (1.0 - tile.albedo)
  2416	            q_net = (sw_net + atm.lw_down
  2417	                     - tile.lw_up - tile.shflx - tile.lhflx)
  2418	
  2419	            evap = tile.lhflx / _const.L_v
  2420	            fw = FreshwaterForcing(
  2421	                precip=atm.precip_total,
  2422	                evap=evap,
  2423	                runoff=runoff_stack[idx],
  2424	                ice_fw=jnp.zeros_like(runoff_stack[idx]),
  2425	            )
  2426	            sf = OceanSurfaceForcing(
  2427	                sw_down=atm.sw_down,
  2428	                q_net=q_net,
  2429	                tau_x=tau_x,
  2430	                tau_y=tau_y,
  2431	                freshwater=None,
  2432	            )
  2433	            # Prognostic slab sea ice: advance the ice tile and partition the
  2434	            # surface forcing (open-ocean fluxes x f_ocean=(1-A) + the ice
  2435	            # tile's basal heat / melt-freeze freshwater / brine salt / stress).
  2436	            # ocean_mask: land cells receive no ice->ocean forcing (mask-aware
  2437	            # blend contract; land_mask is scan-carry state, traced-safe).
  2438	            if enable_sea_ice:
  2439	                new_ice, fw, sf = omip_sea_ice_surface_forcing(
  2440	                    ice_state=ice_in, ice_config=ice_cfg, atm=atm,
  2441	                    ocean_sst_K=sst_K, open_ocean_sf=sf, open_ocean_fw=fw,
  2442	                    dt=dt, grid=None,
  2443	                    ocean_mask=state_in.land_mask.data,
  2444	                )
  2445	            # Ramp sponge strength alongside wind stress.
  2446	            if enable_ramp and enable_sponge:
  2447	                sponge_step = sponge._replace(gamma=sponge.gamma * ramp)
  2448	            else:
  2449	                sponge_step = sponge
  2450	
  2451	            new_state = _dyn_step(
  2452	                state_in, dt,
  2453	                freshwater=fw, surface_forcing=sf, sponge=sponge_step,
  2454	            )
  2455	
  2456	            # SSS restoring (gated at compile time via Python `if`).
  2457	            if enable_sss:
  2458	                S = new_state.S.data
  2459	                target = jnp.asarray(sss_target_static, dtype=S.dtype)
  2460	                alpha = jnp.asarray(sss_alpha_static, dtype=S.dtype)
  2558	    enable_sea_ice = bool(jra55_state.get("enable_sea_ice", False))
  2559	    ice_cfg = jra55_state.get("ice_config")
  2560	
  2561	    _maxvel_3d = model.config.barotropic.maxvel_barotropic
  2562	    enable_maxvel = _maxvel_3d > 0.0
  2563	
  2564	    # Lat-band SPMD: see _build_jra55_block_fn.
  2565	    if spmd_step is not None and enable_sea_ice:
  2566	        raise ValueError(
  2567	            "spmd_step + prognostic sea ice is unsupported "
  2568	            "(run_omip_single refuses --jra55-sea-ice with "
  2569	            "--enable-latlon-spmd).")
  2570	    _dyn_step = (spmd_step if spmd_step is not None
  2571	                 else lambda st, d, **kw: model._step_impl(st, d, **kw))
  2572	
  2573	    lat_2d = jra55_state["lat_2d"]
  2574	    lon_2d = jra55_state["lon_2d"]
  2575	    _rpd = float(RECORDS_PER_DAY)
  2576	    # Static (compile-time) repeat-year-forcing flag.  Selects which clock
  2577	    # drives the solar-zenith insolation geometry (see the scan body).
  2578	    cycle = bool(jra55_state.get("cycle", False))
  2579	
  2580	    def _make_block_fn(n_steps_block):
  2581	        """Create a JIT-compiled block function for a fixed block size."""
  2582	        @jax.jit
  2583	        def block_fn(state, raw_stack, runoff_records, record_days,
  2584	                     block_start_day, block_start_day_forcing,
  2585	                     ice_state=None):
  2586	            dt_days = dt / 86400.0
  2587	
  2588	            def step_body(carry, idx):
  2589	                if enable_sea_ice:
  2590	                    state_in, ice_in = carry
  2591	                else:
  2592	                    state_in = carry
  2593	                # Two clocks (see the insolation + ramp notes below):
  2594	                # - ``day``: RAW simulation day (elapsed run time).  Always
  2595	                #   drives the spinup ramp; drives the solar-zenith clock only
  2596	                #   when NOT cycling (cycle=False ⇒ day_f == day).
  2597	                # - ``day_f``: forcing clock — the CYCLED block-start day
  2598	                #   (aligned with ``record_days``, which the preloaders
  2599	                #   unwrap across the repeat-year cache boundary).  Drives the
  2600	                #   JRA55 record interpolation, and — when cycle=True — the
  2601	                #   solar-zenith insolation clock, so the prescribed rsds and
  2602	                #   the computed zenith stay phase-locked.  Using the raw day
  2603	                #   for interpolation broke every cycle after the first:
  2604	                #   ``day - record_days[0]`` was off by k*cache_length,
  2605	                #   i_lo clipped to the last slice record, and each step
  2606	                #   read one stale record.
  2607	                day = block_start_day + idx * dt_days
  2608	                day_f = block_start_day_forcing + idx * dt_days
  2609	
  2610	                # Find bracketing records: record_days is sorted,
  2611	                # find floor position relative to the first record.
  2612	                local_pos = day_f * _rpd - record_days[0] * _rpd
  2613	                i_lo = jnp.clip(
  2614	                    jnp.floor(local_pos).astype(jnp.int32),
  2615	                    0, record_days.shape[0] - 2,
  2616	                )
  2617	                i_hi = i_lo + 1
  2618	                day_lo = record_days[i_lo]
  2619	                day_hi = record_days[i_hi]
  2620	                alpha = jnp.clip(
  2621	                    jnp.where(day_hi > day_lo,
  2622	                              (day_f - day_lo) / (day_hi - day_lo), 0.0),
  2623	                    0.0, 1.0,
  2624	                )
  2625	
  2626	                def _interp(arr):
  2627	                    return (1.0 - alpha) * arr[i_lo] + alpha * arr[i_hi]
  2628	
  2629	                rsds = _interp(raw_stack["rsds"])
  2630	                rlds = _interp(raw_stack["rlds"])
  2631	                tas = _interp(raw_stack["tas"])
  2632	                huss = _interp(raw_stack["huss"])
  2633	                uas = _interp(raw_stack["uas"])
  2634	                vas = _interp(raw_stack["vas"])
  2635	                psl = _interp(raw_stack["psl"])
  2636	                prra = _interp(raw_stack["prra"])
  2637	                prsn = _interp(raw_stack["prsn"])
  2638	                friver = _interp(runoff_records)
  2639	
  2640	                # Derived: virtual-T density + solar zenith. Canonical coefficient
  2641	                # 1/epsilon - 1 (~0.608), not the rounded 0.61 (~0.4% drift) — must
  2642	                # match jra55_to_atm_surface / _shared.virtual_temperature.
  2643	                T_v = tas * (1.0 + (1.0 / _const.epsilon - 1.0) * huss)
  2644	                rho_a = psl / (_const.R_d * T_v)
  2645	                # Insolation clock (day-of-year + diurnal hour), which sets the
  2646	                # solar zenith and hence zenith-dependent surface albedo:
  2647	                #   - cycle=True (repeat-year forcing): use the CYCLED forcing
  2648	                #     clock ``day_f`` so the solar geometry stays phase-locked
  2649	                #     to the repeated rsds/rlds records.  Using the RAW ``day``
  2650	                #     drifts the seasonal doy (and, for a non-integer cache
  2651	                #     length, the diurnal hour) whenever the cache length is
  2652	                #     not a whole multiple of 365 days (e.g. a 366-day
  2653	                #     leap-year RYF cache), biasing the surface albedo.
  2654	                #   - cycle=False: ``day_f == day`` (no wrap), so this is
  2655	                #     byte-identical to the raw-day clock — the common
  2656	                #     non-cycled path is unchanged.
  2657	                # The SPINUP RAMP (below) intentionally stays on the RAW
  2658	                # ``day``: it is a function of elapsed run time, not forcing
  2659	                # time.  ``cycle`` is a static Python bool (compile-time
  2660	                # feature gate), so this branch is resolved at trace time.
  2661	                day_insol = day_f if cycle else day
  2662	                doy = jnp.mod(day_insol, 365.0) + 1.0
  2663	                hour = jnp.mod(day_insol, 1.0) * 24.0
  2664	                cos_z = cos_zenith_angle(lat_2d, lon_2d, doy, hour)
  2665	
  2666	                atm = AtmToSurface(
  2667	                    sw_down=rsds, lw_down=rlds,
  2668	                    precip_total=prra + prsn, precip_snow=prsn,
  2669	                    T_lowest=tas, q_lowest=huss,
  2670	                    u_lowest=uas, v_lowest=vas,
  2671	                    p_lowest=psl, p_surface=psl,
  2672	                    rho_lowest=rho_a, cos_zenith=cos_z,
  2673	                    co2_ppmv=jnp.asarray(co2_ppmv, dtype=tas.dtype),
  2674	                    has_radiation=jnp.asarray(1.0, dtype=tas.dtype),
  2675	                    has_precipitation=jnp.asarray(1.0, dtype=tas.dtype),
  2676	                )
  2677	
  2678	                sst_K = state_in.T.data[..., 0] + _const.T_freeze
  2679	                u_o = jnp.zeros_like(sst_K)
  2680	                v_o = jnp.zeros_like(sst_K)
  2681	                tile = ocean_tile_response(atm, sst_K, u_o, v_o, coupler_cfg)
  2682	
  2683	                if enable_ramp:
  2684	                    t_sim = day * 86400.0
  2685	                    ramp = jnp.minimum(1.0, t_sim / T_ramp_seconds)
  2686	                else:
  2687	                    ramp = 1.0
  2688	
  2689	                sw_net = atm.sw_down * (1.0 - tile.albedo)
  2690	                q_net = (sw_net + atm.lw_down - tile.lw_up
  2691	                         - tile.shflx - tile.lhflx)
  2692	
  2693	                _dtype = state_in.T.data.dtype
  2694	                E_rate = tile.lhflx / jnp.asarray(_const.L_v, dtype=_dtype)
  2695	                fw = FreshwaterForcing(
  2696	                    precip=jnp.asarray(prra + prsn, dtype=_dtype),
  2697	                    evap=jnp.asarray(E_rate, dtype=_dtype),
  2698	                    runoff=jnp.asarray(friver, dtype=_dtype),
  2699	                    ice_fw=jnp.zeros_like(sst_K, dtype=_dtype),
  2700	                )
  2701	                sf = OceanSurfaceForcing(
  2702	                    sw_down=atm.sw_down, q_net=q_net,
  2703	                    tau_x=tile.tau_x * ramp, tau_y=tile.tau_y * ramp,
  2704	                    freshwater=None,
  2705	                )
  2706	
  2707	                # Prognostic slab sea ice: partition surface forcing between
  2708	                # open ocean (f_ocean=1-A) and the ice tile.  ocean_mask: land
  2709	                # cells receive no ice->ocean forcing (mask-aware blend
  2710	                # contract; land_mask is scan-carry state, traced-safe).
  2711	                if enable_sea_ice:
  2712	                    new_ice, fw, sf = omip_sea_ice_surface_forcing(
  2713	                        ice_state=ice_in, ice_config=ice_cfg, atm=atm,
  2714	                        ocean_sst_K=sst_K, open_ocean_sf=sf, open_ocean_fw=fw,
  2715	                        dt=dt, grid=None,
  2716	                        ocean_mask=state_in.land_mask.data,
  2717	                    )
  2718	
  2719	                sponge_k = (sponge._replace(gamma=sponge.gamma * ramp)
  2720	                            if enable_sponge else None)
  2721	                new_state = _dyn_step(
  2722	                    state_in, dt, freshwater=fw,
  2723	                    surface_forcing=sf, sponge=sponge_k,
  2724	                )
  2725	
  2726	                if enable_sss:
  2727	                    S = new_state.S.data
  2728	                    _sss_mask = new_state.land_mask.data
  2729	                    S_new = S.at[..., 0].set(
  2730	                        S[..., 0] - sss_alpha_static * (
  3370	            "Use the block-scan path (default) or drop --jra55-sea-ice."
  3371	        )
  3372	    if use_scan_blocks:
  3373	        # Scan blocks trace _step_impl directly (bypassing the public
  3374	        # step shim) — prime the build-once caches from the CONCRETE
  3375	        # initial state so the traced body captures the vertex mask as
  3376	        # a constant (codex review MAJOR; census 8474554).
  3377	        # SPMD: the caller (run_omip_single) already primed the caches from
  3378	        # the UNSHARDED state before sharding; re-priming here would run
  3379	        # np.asarray on the sharded ``state`` — a hard non-addressable error
  3380	        # under route-B (shards span processes).  Skip it when spmd_step is set.
  3381	        if spmd_step is None:
  3382	            model.prime_step_caches(state)
  3383	        # Prognostic slab sea ice (--jra55-sea-ice): the block scan carries
  3384	        # (ocean_state, ice_state); thread the ice state across blocks.
  3385	        _ice_on = bool(jra55_state.get("enable_sea_ice", False))
  3386	        ice_state = jra55_state.get("ice_state_init") if _ice_on else None
  3387	        if use_gpu_interp:
  3388	            _get_block_fn_interp = _build_jra55_block_fn_interp(
  3389	                model, jra55_state, dt, spmd_step=spmd_step)
  3390	            print("  GPU-interp mode: forcing interpolation on GPU")
  3391	            # Pre-load the full JRA55 cache for repeat-year runs to
  3392	            # eliminate per-block Zarr I/O (~0.3s/block → ~0s/block).
  3393	            _full_cache = None
  3394	            if jra55_state.get("cycle", False):
  3395	                _fc_all, _fc_days, _fc_len = _preload_jra55_full_cache(
  3396	                    jra55_state)
  3397	                _full_cache = (_fc_all, _fc_days, _fc_len)
  3398	        else:
  3399	            block_fn = _build_jra55_block_fn(model, jra55_state, dt,
  3400	                                             spmd_step=spmd_step)
  3401	            _full_cache = None
  3402	        block_size = max(1, diag_every)
  3403	        if checkpoint_days is not None:
  3404	            steps_per_ckpt = max(1, int(round(checkpoint_days * 86400.0 / dt)))
  3405	        else:
  3406	            steps_per_ckpt = None
  3407	
  3408	        block_start = start_step
  3409	        while block_start < n_steps and not blown_up:
  3410	            actual = min(block_size, n_steps - block_start)
  3411	            t_io_start = time.time()
  3412	
  3413	            if use_gpu_interp and _full_cache is not None:
  3414	                raw_stack, runoff_records, record_meta = (
  3415	                    _slice_preloaded_records(
  3416	                        block_start, actual, dt, jra55_state,
  3417	                        *_full_cache))
  3418	            elif use_gpu_interp:
  3419	                raw_stack, runoff_records, record_meta = (
  3420	                    _preload_jra55_raw_records(
  3421	                        block_start, actual, dt, jra55_state))
  3422	            else:
  3423	                atm_stack, runoff_stack = _preload_jra55_forcing_block(
  3424	                    block_start, actual, dt, jra55_state,
  3425	                )
  3426	            if spmd_shard_stack is not None:
  3427	                # Lay the per-block forcing stacks out lat-band-sharded so
  3428	                # the in-scan interpolation / bulk fluxes stay shard-local
  3429	                # (an unsharded stack commits to device 0 and serializes
  3430	                # every forcing op there).
  3431	                if use_gpu_interp:
  3432	                    raw_stack = spmd_shard_stack(raw_stack)
  3433	                    runoff_records = spmd_shard_stack(runoff_records)
  3434	                else:
  3435	                    atm_stack = spmd_shard_stack(atm_stack)
  3436	                    runoff_stack = spmd_shard_stack(runoff_stack)
  3437	            io_dt = time.time() - t_io_start
  3438	
  3439	            t_compute_start = time.time()
  3440	            if use_gpu_interp:
  3441	                bfn = _get_block_fn_interp(actual)
  3442	                if _ice_on:
  3443	                    state, ice_state = bfn(
  3444	                        state, raw_stack, runoff_records,
  3445	                        record_meta["record_days"],
  3446	                        jnp.float64(record_meta["block_start_day"]),
  3447	                        jnp.float64(record_meta["block_start_day_forcing"]),
  3448	                        ice_state,
  3449	                    )
  3450	                else:
  3451	                    state = bfn(
  3452	                        state, raw_stack, runoff_records,
  3453	                        record_meta["record_days"],
  3454	                        jnp.float64(record_meta["block_start_day"]),
  3455	                        jnp.float64(record_meta["block_start_day_forcing"]),
  3456	                    )
  3457	            else:
  3458	                if _ice_on:
  3459	                    state, ice_state = block_fn(
  3460	                        state, atm_stack, runoff_stack,
  3461	                        jnp.int32(block_start), ice_state,
  3462	                    )
  3463	                else:
  3464	                    state = block_fn(
  3465	                        state, atm_stack, runoff_stack,
  3466	                        jnp.int32(block_start),
  3467	                    )
  3468	            jax.block_until_ready(state.T.data)
  3469	            compute_dt = time.time() - t_compute_start
  3470	
  3471	            block_start += actual
  3472	            step = block_start
  3473	            day = step * dt / 86400.0
  3474	
  3475	            if not _check_finite(state, grid_type):
  4760	        ramp_days_eff = args.restoring_ramp_days
  4761	    elif grid_type == "cubed_sphere":
  4762	        ramp_days_eff = 14.0
  4763	    else:
  4764	        ramp_days_eff = 0.0
  4765	
  4766	    # --- Lat-band SPMD (--enable-latlon-spmd): wrap the dynamics step in
  4767	    # the validated multi-GPU sharded ocean step and shard the state.
  4768	    # Single-controller only; restoring lane only (the JRA55 block
  4769	    # functions call model._step_impl directly — follow-up).  Runs AFTER
  4770	    # the restart load so a resumed state is sharded too.
  4771	    spmd_step = None
  4772	    spmd_gather = None
  4773	    spmd_shard_stack = None
  4774	    if run_config.enable_latlon_spmd:
  4775	        if grid_type != "latlon":
  4776	            raise SystemExit(
  4777	                f"--enable-latlon-spmd requires --grid latlon "
  4778	                f"(got {grid_type}).")
  4779	        if (jra55_state is not None
  4780	                and jra55_state.get("_use_single_step", False)):
  4781	            raise SystemExit(
  4782	                "--enable-latlon-spmd requires the JRA55 block-scan path, "
  4783	                "but this run selected the single-step fallback "
  4784	                "(_jra55_step calls model.step directly).")
  4785	        _multi = run_config.multicontroller
  4786	        if jax.process_count() > 1 and not _multi:
  4787	            raise SystemExit(
  4788	                "--enable-latlon-spmd is single-controller only unless "
  4789	                "--multicontroller is set; multi-node ocean scaling needs "
  4790	                "the route-B lane (jax.distributed cross-process NCCL).")
  4791	        if _multi:
  4792	            # Route-B: the mesh spans ALL global devices (one band per device
  4793	            # across every process). A strict subset would leave some
  4794	            # processes' devices out of the program (non-addressable
  4795	            # participation hazard — matches the ocean bench's guard).
  4796	            _nd = len(jax.devices())
  4797	            if run_config.spmd_n_devices and run_config.spmd_n_devices != _nd:
  4798	                raise SystemExit(
  4799	                    f"--multicontroller uses ALL global devices ({_nd} across "
  4800	                    f"{jax.process_count()} processes); --spmd-n-devices "
  4801	                    f"({run_config.spmd_n_devices}) must be 0 (auto) or {_nd}.")
  4802	        else:
  4803	            _nd = run_config.spmd_n_devices or len(jax.devices())
  4804	        if _nd > 1:
  4805	            if grid.n_lat % _nd != 0:
  4806	                raise SystemExit(
  4807	                    f"--enable-latlon-spmd: n_lat ({grid.n_lat}) not "
  4808	                    f"divisible by the device count ({_nd}); pick "
  4809	                    f"--spmd-n-devices dividing n_lat.")
  4810	            from functools import partial
  4811	
  4812	            from legoesm.ocean.dynamics.sharded_ocean_step import (
  4813	                gather_state_latlon,
  4814	                make_sharded_ocean_step,
  4815	                shard_forcing_stack_latlon,
  4816	                shard_state_latlon,
  4817	            )
  4818	            from legoesm.parallel.mesh import create_latlon_mesh
  4819	            # Prime the build-once vertex-mask cache from the CONCRETE
  4820	            # state so the wrapper can build per-band masks host-side.
  4821	            model.prime_step_caches(state)
  4822	            _dev = create_latlon_mesh(n_devices=_nd)
  4823	            spmd_step = make_sharded_ocean_step(model, _dev.mesh)
  4824	            spmd_gather = partial(gather_state_latlon, mesh=_dev.mesh)
  4825	            state = shard_state_latlon(state, _dev.mesh)
  4826	            # Lay per-block forcing stacks out lat-band-sharded so the
  4827	            # in-scan interpolation / bulk fluxes stay shard-local (shared
  4828	            # layout helper — see shard_forcing_stack_latlon).
  4829	            spmd_shard_stack = partial(
  4830	                shard_forcing_stack_latlon, mesh=_dev.mesh)
  4831	            if jax.process_index() == 0:
  4832	                _lane = "route-B multicontroller" if _multi else "single-controller"
  4833	                print(f"  SPMD ({_lane}): lat-band sharded dynamics step over "
  4834	                      f"{_nd} devices across {jax.process_count()} process(es) "
  4835	                      f"({jax.default_backend()}).")
  4836	        elif jax.process_index() == 0:
  4837	            print("  SPMD: single device visible — flag is a no-op.")
  4838	
  4839	    # Run time loop
  4840	    state, diag, wall_time, ok, blowup_info = _run_omip_loop(
  4841	        model, state, grid_type, grid, z_coord,
  4842	        dt, n_steps, diag_every,
  4843	        label=f"{grid_type}/{resolution}",
  4844	        restoring_targets=restoring_targets,
  4845	        restoring_tau_s=restoring_tau_s,
  4846	        restoring_ramp_days=ramp_days_eff,
  4847	        jra55_state=jra55_state,
  4848	        checkpoint_days=checkpoint_days,
  4849	        checkpoint_dir=checkpoint_dir,
  4850	        max_wallclock_seconds=run_config.max_wallclock_seconds,
  4851	        restart_buffer_seconds=run_config.restart_buffer_seconds,
  4852	        start_step=start_step,
  4853	        nudge_woa_tau=args.nudge_woa_tau,
  4854	        T_woa_3d=(T_woa * state.land_mask.data[..., jnp.newaxis]).astype(
  4855	            state.T.data.dtype) if args.nudge_woa_tau > 0 and T_woa is not None else None,
  4856	        S_woa_3d=(S_woa * state.land_mask.data[..., jnp.newaxis]).astype(
  4857	            state.S.data.dtype) if args.nudge_woa_tau > 0 and S_woa is not None else None,
  4858	        snapshot_fn=_snapshot_fn,
  4859	        spmd_step=spmd_step,
  4860	        spmd_gather=spmd_gather,
  4861	        spmd_shard_stack=spmd_shard_stack,
  4862	    )
  4863	    if spmd_gather is not None:
  4864	        # Downstream report/plot/save paths expect the full (n_lat+1)
  4865	        # staggered v layout, not the sharded v_lower carry.  Every rank
  5555	                "the lat-band sharded step does not thread t_seconds, so the "
  5556	                "equilibrium tide would be SILENTLY inert. Run the tide "
  5557	                "single-device, or disable tidal forcing for the SPMD run.")
  5558	        if args.spmd_persistent_state:
  5559	            # PERSISTENT lane (scaling-M2 increment 1): the state stays
  5560	            # lat-band sharded ACROSS steps via the pure-dynamics inner step
  5561	            # (the wrapper docstring's own guidance); shard_state_latlon /
  5562	            # gather_state_latlon run only at the residency boundaries the
  5563	            # helpers below manage (initial shard, snapshot/abort/final
  5564	            # gathers, and the counted per-step gathers forced by host-global
  5565	            # consumers).  t_sec is always None here (tide fail-fasts above).
  5566	            from legoesm.ocean.dynamics.sharded_ocean_step import (
  5567	                gather_state_latlon,
  5568	                make_sharded_ocean_step,
  5569	                shard_state_latlon,
  5570	            )
  5571	            _spmd_inner = make_sharded_ocean_step(model, _spmd_mesh)
  5572	            _ocean_step = (lambda st, sf, fw, t_sec=None:
  5573	                           _spmd_inner(st, dt, surface_forcing=sf,
  5574	                                       freshwater=fw))
  5575	
  5576	            def _pers_shard_fn(st, _mesh=_spmd_mesh):
  5577	                return shard_state_latlon(st, _mesh)
  5578	
  5579	            def _pers_gather_fn(st, _mesh=_spmd_mesh):
  5580	                return gather_state_latlon(st, _mesh)
  5581	
  5582	            _spmd_persistent = True
  5583	            print(f"[setup] multi-GPU lat-band SPMD: {args.n_gpus} devices, "
  5584	                  f"n_lat={n_lat_final} ({n_lat_final // args.n_gpus} "
  5585	                  f"rows/band); PERSISTENT sharded state "
  5586	                  f"(--spmd-persistent-state): full-state gathers only at "
  5587	                  f"snapshot/abort/final + counted per-step forcings.")
  5588	        else:
  5589	            _spmd_step = make_sharded_ocean_step_global(model, _spmd_mesh)
  5590	            # t_sec is always None here (tide-enabled fail-fasts above).
  5591	            _ocean_step = (lambda st, sf, fw, t_sec=None:
  5592	                           _spmd_step(st, dt, surface_forcing=sf,
  5593	                                      freshwater=fw))
  5594	            print(f"[setup] multi-GPU lat-band SPMD: {args.n_gpus} devices, "
  5595	                  f"n_lat={n_lat_final} ({n_lat_final // args.n_gpus} "
  5596	                  f"rows/band); global-in/global-out wrapper (host BCs on "
  5597	                  f"gathered state).")
  5598	
  5599	    # ------------------------------------------------------------------
  5600	    # --spmd-persistent-state residency helpers (scaling-M2).  The persistent
  5655	    # ------------------------------------------------------------------
  5656	    # Issue #354: optional lax.scan block-stepping (tripole; no nudge/drag).
  5657	    # The CORE-II forcing is sampled on-device (no per-step host roundtrip),
  5658	    # so XLA fuses each block of ``--scan-block`` steps.  Default
  5659	    # (--scan-block 0) keeps the bit-identical Python loop below.
  5660	    # Diagnostics / snapshots / non-finite abort run at BLOCK BOUNDARIES.
  5661	    # ------------------------------------------------------------------
  5662	    _tti = getattr(getattr(model, "config", None),
  5663	                   "tracer_time_integrator", "euler")
  5664	    if args.n_gpus > 1 and int(args.scan_block) > 0:
  5665	        raise SystemExit(
  5666	            "--n-gpus > 1 and --scan-block are mutually exclusive: the lax.scan "
  5667	            "block path fuses single-device on-device steps (it does not use the "
  5668	            "lat-band sharded step). Pick one — multi-GPU SPMD (the host Python "
  5669	            "loop, --scan-block 0) OR single-device scan fusion.")
  5670	    # Equilibrium tide disqualifies the scan-block path: its body steps via
  5671	    # model._step_impl(...) with no t_seconds (bypassing step()'s eager
  5672	    # enabled-but-no-time guard), so an enabled tide would be SILENTLY inert
  5673	    # inside the scan.  Fall back to the host loop, which threads t.
  5674	    _tf_scan = getattr(getattr(model, "config", None), "tidal_forcing", None)
  5675	    _tide_enabled = _tf_scan is not None and _tf_scan.enabled

exec
/usr/bin/bash -lc "git status --short
git diff --name-status HEAD
nl -ba tests/bench/test_timed_scan_blocks.py | sed -n '15,115p'
nl -ba tests/unit/test_run_omip_latlon_spmd.py | sed -n '185,255p'
rg -n \"aux=|\\.aux\\b|2-arg|two-arg|multicontroller.*timed_scan|timed_scan.*multicontroller\" tests scripts/validate scripts/bench --glob '"'!**/__pycache__/**'"'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 743ms:
 M packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
 M scripts/bench/bench_ocean_latlon_spmd_scaling.py
 M scripts/bench/metadata.py
 M scripts/plot/plot_scaling_paper_figure.py
?? .physics-validator/scaling_campaign/codex_aux_fix_r18_2026-08-03.md
?? .physics-validator/scaling_campaign/codex_bcast_fix_r17_2026-08-03.md
M	packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
M	scripts/bench/bench_ocean_latlon_spmd_scaling.py
M	scripts/bench/metadata.py
M	scripts/plot/plot_scaling_paper_figure.py
    15	
    16	sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))
    17	from metadata import git_sha, scaling_metadata, timed_scan_blocks  # noqa: E402
    18	
    19	
    20	def _advance(s):
    21	    return jax.tree_util.tree_map(lambda x: x * 0.999 + 0.001, s)
    22	
    23	
    24	def test_schedule_executes_exact_step_count():
    25	    s0 = {"a": jnp.ones((4, 4)), "b": jnp.zeros((3,))}
    26	    # count steps via the value: each step is affine, invertible bookkeeping —
    27	    # use a counter leaf instead (exact).
    28	    s0["n"] = jnp.zeros(())
    29	    def adv(s):
    30	        out = dict(s)
    31	        out["n"] = s["n"] + 1.0
    32	        out["a"] = s["a"] * 0.999
    33	        return out
    34	    s, m = timed_scan_blocks(adv, s0, block_steps=5, n_blocks=2, probe_steps=3)
    35	    # 1 compile + 3 probe + 2*5 block = 14 steps
    36	    assert float(s["n"]) == 14.0
    37	    assert m["block_steps"] == 5 and m["n_blocks"] == 2 and m["probe_steps"] == 3
    38	    assert len(m["block_ms"]) == 2
    39	
    40	
    41	def test_metrics_shape_and_separation():
    42	    s0 = {"x": jnp.ones((8,))}
    43	    s, m = timed_scan_blocks(_advance, s0, block_steps=4, n_blocks=1,
    44	                             probe_steps=2)
    45	    for k in ("compile_ms", "scan_compile_ms", "step_latency_ms",
    46	              "fused_step_ms", "parallel_block_ms", "rank_imbalance",
    47	              "rank_imbalance_per_block"):
    48	        assert k in m, k
    49	    # single process: the parallel block time IS this process's block time,
    50	    # and every per-block max/median ratio is exactly 1.0
    51	    assert m["parallel_block_ms"] == m["block_ms"]
    52	    assert m["rank_imbalance"] == 1.0
    53	    assert m["rank_imbalance_per_block"] == [1.0]
    54	    # fused number derives from the block, not the probe (both fields are
    55	    # independently rounded — fused to 4 dp, block to 2 dp — so compare at
    56	    # the rounding granularity, not machine precision)
    57	    assert abs(m["fused_step_ms"] - m["parallel_block_ms"][0] / 4) < 5e-3
    58	
    59	
    60	def test_zero_block_steps_parity_path_null_headline():
    61	    """block_steps=0 is the documented zero-length parity path: the blocks
    62	    run (fences/collectives matched) but a zero-step block has NO per-step
    63	    time — the headline must be an honest null, not block_ms/1."""
    64	    s0 = {"x": jnp.ones(())}
    65	    _, m = timed_scan_blocks(_advance, s0, block_steps=0, n_blocks=1,
    66	                             probe_steps=0)
    67	    assert m["fused_step_ms"] is None
    68	    assert m["block_steps"] == 0 and m["n_blocks"] == 1
    69	    assert len(m["block_ms"]) == 1
    70	
    71	
    72	def test_invalid_schedule_raises_never_clamped():
    73	    """Silent max(1, n_blocks)/max(0, probe_steps) rewrites executed a
    74	    schedule DIFFERENT from the recorded one (codex batch4): invalid values
    75	    must raise so the record always equals the execution."""
    76	    s0 = {"x": jnp.ones(())}
    77	    with pytest.raises(ValueError, match="n_blocks"):
    78	        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=0,
    79	                          probe_steps=1)
    80	    with pytest.raises(ValueError, match="n_blocks"):
    81	        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=-2,
    82	                          probe_steps=1)
    83	    with pytest.raises(ValueError, match="probe_steps"):
    84	        timed_scan_blocks(_advance, s0, block_steps=2, n_blocks=1,
    85	                          probe_steps=-1)
    86	    with pytest.raises(ValueError, match="block_steps"):
    87	        timed_scan_blocks(_advance, s0, block_steps=-1, n_blocks=1,
    88	                          probe_steps=0)
    89	
    90	
    91	def test_seed_state_not_mutated_by_precompile():
    92	    """The scan pre-compile runs on a shallow ALIAS of the seed leaves
    93	    (documented contract: advance must not donate); the seed values used by
    94	    the timed blocks must be byte-identical to freshly advanced ones."""
    95	    import numpy as np
    96	    s0 = {"x": jnp.arange(6.0)}
    97	    ref = np.asarray(s0["x"]).copy()
    98	    s, m = timed_scan_blocks(_advance, dict(s0), block_steps=1, n_blocks=1,
    99	                             probe_steps=0)
   100	    # 1 compile + 0 probe + 1x1 block = 2 steps from the ORIGINAL seed.
   101	    want = ref
   102	    for _ in range(2):
   103	        want = want * 0.999 + 0.001
   104	    # atol at ULP scale: a pre-compile that ADVANCED the live seed (an
   105	    # extra step) would be off by ~1e-3, orders beyond this tolerance.
   106	    assert np.allclose(np.asarray(s["x"]), want, rtol=0, atol=1e-12)
   107	
   108	
   109	def test_git_sha_fail_open_and_in_metadata():
   110	    sha = git_sha()
   111	    assert isinstance(sha, str) and len(sha) >= 4  # short sha or "unknown"
   112	    md = scaling_metadata(grid="latlon", component="ocean", resolution="8x16",
   113	                          n_levels=4, precision="float64")
   114	    assert md["git_sha"] == sha
   185	
   186	    tmp_path = Path(tmp_dir)
   187	    n_lat, n_lon = 8, 16
   188	    dt, n_steps = 600.0, 6
   189	
   190	    cache = _day23._make_synthetic_cache(
   191	        tmp_path, n_lat=n_lat, n_lon=n_lon, n_records=64)
   192	    grid, z_coord, _config, model, _gt = _day23._make_tiny_latlon_setup(
   193	        n_lat=n_lat, n_lon=n_lon)
   194	    T_woa, S_woa = _day23._make_woa_like_targets(grid, nlev=4)
   195	    args = _day23._argparse_namespace(jra55_cache=str(cache))
   196	
   197	    def _fresh_js():
   198	        # _setup_jra55_forcing_state does NOT set _gpu_interp (the driver
   199	        # does, from --gpu-interp); set it here to pick the lane.  A fresh
   200	        # dict per loop avoids sharing the sponge/target arrays across the
   201	        # serial and sharded runs.
   202	        js = run_omip._setup_jra55_forcing_state(
   203	            args, grid, "latlon", z_coord=z_coord, T_woa=T_woa, S_woa=S_woa)
   204	        js["_gpu_interp"] = (interp_mode == "gpu")
   205	        return js
   206	
   207	    common = dict(
   208	        grid_type="latlon", grid=grid, z_coord=z_coord,
   209	        dt=dt, n_steps=n_steps, diag_every=3,
   210	        checkpoint_days=None, checkpoint_dir=None,
   211	    )
   212	
   213	    state0 = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
   214	    serial_final, _d, _w, ok_serial, _b = run_omip._run_omip_loop(
   215	        model, state0, jra55_state=_fresh_js(), **common)
   216	    assert ok_serial, "serial JRA55 loop reported not-ok"
   217	
   218	    dev = create_latlon_mesh(n_devices=2)
   219	    spmd_step = make_sharded_ocean_step(model, dev.mesh)
   220	    model.prime_step_caches(state0)
   221	    ss0 = shard_state_latlon(state0, dev.mesh)
   222	    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
   223	        model, ss0, jra55_state=_fresh_js(),
   224	        spmd_step=spmd_step,
   225	        spmd_gather=lambda st: gather_state_latlon(st, dev.mesh),
   226	        spmd_shard_stack=partial(shard_forcing_stack_latlon, mesh=dev.mesh),
   227	        **common)
   228	    assert ok_spmd, "SPMD JRA55 loop reported not-ok"
   229	    spmd_final = gather_state_latlon(spmd_final, dev.mesh)
   230	
   231	    # Same re-association floor as the restoring-lane and step-level gates.
   232	    for nm in ("u", "v", "eta", "T", "S"):
   233	        np.testing.assert_allclose(
   234	            np.asarray(getattr(spmd_final, nm).data),
   235	            np.asarray(getattr(serial_final, nm).data),
   236	            atol=2e-4, rtol=1e-3,
   237	            err_msg=f"JRA55 {interp_mode}-interp SPMD {nm} mismatch")
   238	    print("JRA55-PARITY-OK")
   239	
   240	
   241	def _run_jra55_parity_subprocess(tmp_path, interp_mode: str):
   242	    env = dict(os.environ)
   243	    env["JAX_PLATFORMS"] = "cpu"
   244	    env["JAX_ENABLE_X64"] = "1"
   245	    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
   246	    _root = str(_THIS.parents[2])
   247	    env["PYTHONPATH"] = _root + os.pathsep + env.get("PYTHONPATH", "")
   248	    proc = subprocess.run(
   249	        [sys.executable, str(_THIS), "--jra55-parity-worker",
   250	         str(tmp_path), interp_mode],
   251	        env=env, capture_output=True, text=True, timeout=870,
   252	    )
   253	    assert proc.returncode == 0 and "JRA55-PARITY-OK" in proc.stdout, (
   254	        f"rc={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
   255	        f"stderr:\n{proc.stderr[-3000:]}")
scripts/bench/metadata.py:898:    aux=None,
scripts/bench/metadata.py:1009:    # Normalize to a 2-arg advance (see the aux note below): top-level
scripts/bench/bench_ocean_latlon_spmd_scaling.py:479:        (lambda st, aux: step(st, args.dt, aux=aux)) if _aux is not None
scripts/bench/bench_ocean_latlon_spmd_scaling.py:483:        sync_label="ocean_latlon_spmd_bench", aux=_aux)
tests/unit/test_grid_operators_adapter.py:176:    # (divergence is 1-arg edge-normal, not 2-arg (u, v)).
tests/atmosphere/hydrostatic/unit/test_holtslag_boville_oracle.py:89:                shflx=shflx, qflx=qflx, taux=taux, tauy=tauy)
scripts/validate/ocean_fidelity/compare_legoesm_vs_veros.py:199:     ) = _veros_cell_views(result, with_salt=False, with_taux=False)
scripts/validate/ocean_fidelity/compare_legoesm_vs_veros.py:210:     ) = _veros_cell_views(result, with_salt=True, with_taux=True)
tests/unit/test_column_stepping.py:128:    # Stage 1 sees aux=0 and returns 1; later stages reuse that carry.
tests/unit/test_land_ml_checkpoint_roundtrip.py:33:        _carry_aux={}, _land_ml_state=saved,
tests/unit/test_land_ml_checkpoint_roundtrip.py:46:    dst = SimpleNamespace(_carry_aux=loaded, _land_ml_state=_state(99.0))
tests/unit/test_land_ml_checkpoint_roundtrip.py:63:        _carry_aux={}, _land_ml_state=saved,
tests/unit/test_land_ml_checkpoint_roundtrip.py:72:    dst = SimpleNamespace(_carry_aux=loaded,
tests/unit/test_land_ml_checkpoint_roundtrip.py:83:        _carry_aux={"land_ml_T_soil": np.ones(3)}, _land_ml_state=None)
tests/unit/test_land_ml_checkpoint_roundtrip.py:95:        _carry_aux={}, _land_ml_state=saved,
tests/unit/test_land_ml_checkpoint_roundtrip.py:101:    dst = SimpleNamespace(_carry_aux=dict(aux), _land_ml_state=_state(99.0))
tests/unit/test_land_ml_checkpoint_roundtrip.py:108:        _carry_aux={f"land_ml_{f}": np.ones((4, 6) if i < 3 else (4,),
tests/unit/test_land_ml_checkpoint_roundtrip.py:124:        _carry_aux=good,
tests/ocean/unit/test_isoneutral_slope_density.py:514:        (total, K33), grad = jax.value_and_grad(k33_sum, has_aux=True)(T)

exec
/usr/bin/bash -lc "rg -n -C 6 \"def _replicated_put|geom_stacks\\s*=|vmask_stack\\s*=\" packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
nl -ba scripts/run/run_omip.py | sed -n '2358,2460p'
nl -ba scripts/run/run_omip.py | sed -n '2560,2730p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 238ms:
685-    # 26524423). The leading axis has length n_dev, so P("lat") divides it
686-    # exactly; the body indexes its local slab at [0]. Values are unchanged
687-    # — same stack, different placement; the cross-process divergence
688-    # guard below runs on HOST values and is placement-blind.
689-    rep = NamedSharding(mesh, P("lat"))
690-
691:    def _replicated_put(arr, name):
692-        # (Name kept for history; since 2026-08-03 this is a SHARDED stack
693-        # put.) The band-geometry arrays are (re)computed per process and
694-        # can differ in their last ULPs (per-process XLA autotuning on
695-        # device-derived grid fields) — the fully-replicated-put era
696-        # broadcast process 0's bytes to sidestep the P() bit-identity
697-        # assert (job 26450848). With the #1370-iii P("lat") sharding each
--
766-                "make_sharded_ocean_step: the band-geometry SCHEMA differs "
767-                "across processes (field list / x64 setting / device count "
768-                f"— gathered {_g.tolist()}). Fix the per-process config "
769-                "before sharding; the per-field checks below assume one "
770-                "schema.")
771-
772:    geom_stacks = {
773-        name: _replicated_put(
774-            jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
775-                      axis=0), name)
776-        for name in array_field_names
777-    }
778:    vmask_stack = _replicated_put(
779-        jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0),
780-        "vertex_mask")
781-
782-    # Static perms for the v north-boundary-row ppermute (band r receives band
783-    # r+1's v_lower[0] = global v[e]; north band non-target receives 0).
784-    perm_north, _perm_south = latlon_band_perms(n_dev)
  2358	    _maxvel_3d = model.config.barotropic.maxvel_barotropic
  2359	    enable_maxvel = _maxvel_3d > 0.0
  2360	
  2361	    # Lat-band SPMD (--enable-latlon-spmd): the scan body's dynamics step
  2362	    # runs through the sharded wrapper (same forcing kwargs as _step_impl;
  2363	    # the wrapper's cache/arm-restore Python runs ONCE at block trace).
  2364	    # Sea ice is refused upstream (the ice tile is not SPMD-audited yet).
  2365	    if spmd_step is not None and enable_sea_ice:
  2366	        raise ValueError(
  2367	            "spmd_step + prognostic sea ice is unsupported "
  2368	            "(run_omip_single refuses --jra55-sea-ice with "
  2369	            "--enable-latlon-spmd).")
  2370	    _dyn_step = (spmd_step if spmd_step is not None
  2371	                 else lambda st, d, **kw: model._step_impl(st, d, **kw))
  2372	
  2373	    @jax.jit
  2374	    def block_fn(state, atm_stack, runoff_stack, block_start_step,
  2375	                 ice_state=None):
  2376	        def step_body(carry, idx):
  2377	            if enable_sea_ice:
  2378	                state_in, ice_in = carry
  2379	            else:
  2380	                state_in = carry
  2381	            atm = AtmToSurface(
  2382	                sw_down=atm_stack["sw_down"][idx],
  2383	                lw_down=atm_stack["lw_down"][idx],
  2384	                precip_total=atm_stack["precip_total"][idx],
  2385	                precip_snow=atm_stack["precip_snow"][idx],
  2386	                T_lowest=atm_stack["T_lowest"][idx],
  2387	                q_lowest=atm_stack["q_lowest"][idx],
  2388	                u_lowest=atm_stack["u_lowest"][idx],
  2389	                v_lowest=atm_stack["v_lowest"][idx],
  2390	                p_lowest=atm_stack["p_lowest"][idx],
  2391	                p_surface=atm_stack["p_surface"][idx],
  2392	                rho_lowest=atm_stack["rho_lowest"][idx],
  2393	                cos_zenith=atm_stack["cos_zenith"][idx],
  2394	                co2_ppmv=jnp.asarray(co2_ppmv, dtype=atm_stack["T_lowest"].dtype),
  2395	                has_radiation=jnp.asarray(1.0, dtype=atm_stack["T_lowest"].dtype),
  2396	                has_precipitation=jnp.asarray(1.0, dtype=atm_stack["T_lowest"].dtype),
  2397	            )
  2398	
  2399	            sst_K = state_in.T.data[..., 0] + _const.T_freeze
  2400	            u_o = jnp.zeros_like(sst_K)
  2401	            v_o = jnp.zeros_like(sst_K)
  2402	            tile = ocean_tile_response(atm, sst_K, u_o, v_o, coupler_cfg)
  2403	
  2404	            # Wind-stress spinup ramp (gated at compile time).
  2405	            if enable_ramp:
  2406	                abs_step = block_start_step + idx
  2407	                t_sim = abs_step.astype(jnp.float64) * dt
  2408	                ramp = jnp.minimum(1.0, t_sim / T_ramp_seconds)
  2409	                tau_x = tile.tau_x * ramp
  2410	                tau_y = tile.tau_y * ramp
  2411	            else:
  2412	                tau_x = tile.tau_x
  2413	                tau_y = tile.tau_y
  2414	
  2415	            sw_net = atm.sw_down * (1.0 - tile.albedo)
  2416	            q_net = (sw_net + atm.lw_down
  2417	                     - tile.lw_up - tile.shflx - tile.lhflx)
  2418	
  2419	            evap = tile.lhflx / _const.L_v
  2420	            fw = FreshwaterForcing(
  2421	                precip=atm.precip_total,
  2422	                evap=evap,
  2423	                runoff=runoff_stack[idx],
  2424	                ice_fw=jnp.zeros_like(runoff_stack[idx]),
  2425	            )
  2426	            sf = OceanSurfaceForcing(
  2427	                sw_down=atm.sw_down,
  2428	                q_net=q_net,
  2429	                tau_x=tau_x,
  2430	                tau_y=tau_y,
  2431	                freshwater=None,
  2432	            )
  2433	            # Prognostic slab sea ice: advance the ice tile and partition the
  2434	            # surface forcing (open-ocean fluxes x f_ocean=(1-A) + the ice
  2435	            # tile's basal heat / melt-freeze freshwater / brine salt / stress).
  2436	            # ocean_mask: land cells receive no ice->ocean forcing (mask-aware
  2437	            # blend contract; land_mask is scan-carry state, traced-safe).
  2438	            if enable_sea_ice:
  2439	                new_ice, fw, sf = omip_sea_ice_surface_forcing(
  2440	                    ice_state=ice_in, ice_config=ice_cfg, atm=atm,
  2441	                    ocean_sst_K=sst_K, open_ocean_sf=sf, open_ocean_fw=fw,
  2442	                    dt=dt, grid=None,
  2443	                    ocean_mask=state_in.land_mask.data,
  2444	                )
  2445	            # Ramp sponge strength alongside wind stress.
  2446	            if enable_ramp and enable_sponge:
  2447	                sponge_step = sponge._replace(gamma=sponge.gamma * ramp)
  2448	            else:
  2449	                sponge_step = sponge
  2450	
  2451	            new_state = _dyn_step(
  2452	                state_in, dt,
  2453	                freshwater=fw, surface_forcing=sf, sponge=sponge_step,
  2454	            )
  2455	
  2456	            # SSS restoring (gated at compile time via Python `if`).
  2457	            if enable_sss:
  2458	                S = new_state.S.data
  2459	                target = jnp.asarray(sss_target_static, dtype=S.dtype)
  2460	                alpha = jnp.asarray(sss_alpha_static, dtype=S.dtype)
  2560	
  2561	    _maxvel_3d = model.config.barotropic.maxvel_barotropic
  2562	    enable_maxvel = _maxvel_3d > 0.0
  2563	
  2564	    # Lat-band SPMD: see _build_jra55_block_fn.
  2565	    if spmd_step is not None and enable_sea_ice:
  2566	        raise ValueError(
  2567	            "spmd_step + prognostic sea ice is unsupported "
  2568	            "(run_omip_single refuses --jra55-sea-ice with "
  2569	            "--enable-latlon-spmd).")
  2570	    _dyn_step = (spmd_step if spmd_step is not None
  2571	                 else lambda st, d, **kw: model._step_impl(st, d, **kw))
  2572	
  2573	    lat_2d = jra55_state["lat_2d"]
  2574	    lon_2d = jra55_state["lon_2d"]
  2575	    _rpd = float(RECORDS_PER_DAY)
  2576	    # Static (compile-time) repeat-year-forcing flag.  Selects which clock
  2577	    # drives the solar-zenith insolation geometry (see the scan body).
  2578	    cycle = bool(jra55_state.get("cycle", False))
  2579	
  2580	    def _make_block_fn(n_steps_block):
  2581	        """Create a JIT-compiled block function for a fixed block size."""
  2582	        @jax.jit
  2583	        def block_fn(state, raw_stack, runoff_records, record_days,
  2584	                     block_start_day, block_start_day_forcing,
  2585	                     ice_state=None):
  2586	            dt_days = dt / 86400.0
  2587	
  2588	            def step_body(carry, idx):
  2589	                if enable_sea_ice:
  2590	                    state_in, ice_in = carry
  2591	                else:
  2592	                    state_in = carry
  2593	                # Two clocks (see the insolation + ramp notes below):
  2594	                # - ``day``: RAW simulation day (elapsed run time).  Always
  2595	                #   drives the spinup ramp; drives the solar-zenith clock only
  2596	                #   when NOT cycling (cycle=False ⇒ day_f == day).
  2597	                # - ``day_f``: forcing clock — the CYCLED block-start day
  2598	                #   (aligned with ``record_days``, which the preloaders
  2599	                #   unwrap across the repeat-year cache boundary).  Drives the
  2600	                #   JRA55 record interpolation, and — when cycle=True — the
  2601	                #   solar-zenith insolation clock, so the prescribed rsds and
  2602	                #   the computed zenith stay phase-locked.  Using the raw day
  2603	                #   for interpolation broke every cycle after the first:
  2604	                #   ``day - record_days[0]`` was off by k*cache_length,
  2605	                #   i_lo clipped to the last slice record, and each step
  2606	                #   read one stale record.
  2607	                day = block_start_day + idx * dt_days
  2608	                day_f = block_start_day_forcing + idx * dt_days
  2609	
  2610	                # Find bracketing records: record_days is sorted,
  2611	                # find floor position relative to the first record.
  2612	                local_pos = day_f * _rpd - record_days[0] * _rpd
  2613	                i_lo = jnp.clip(
  2614	                    jnp.floor(local_pos).astype(jnp.int32),
  2615	                    0, record_days.shape[0] - 2,
  2616	                )
  2617	                i_hi = i_lo + 1
  2618	                day_lo = record_days[i_lo]
  2619	                day_hi = record_days[i_hi]
  2620	                alpha = jnp.clip(
  2621	                    jnp.where(day_hi > day_lo,
  2622	                              (day_f - day_lo) / (day_hi - day_lo), 0.0),
  2623	                    0.0, 1.0,
  2624	                )
  2625	
  2626	                def _interp(arr):
  2627	                    return (1.0 - alpha) * arr[i_lo] + alpha * arr[i_hi]
  2628	
  2629	                rsds = _interp(raw_stack["rsds"])
  2630	                rlds = _interp(raw_stack["rlds"])
  2631	                tas = _interp(raw_stack["tas"])
  2632	                huss = _interp(raw_stack["huss"])
  2633	                uas = _interp(raw_stack["uas"])
  2634	                vas = _interp(raw_stack["vas"])
  2635	                psl = _interp(raw_stack["psl"])
  2636	                prra = _interp(raw_stack["prra"])
  2637	                prsn = _interp(raw_stack["prsn"])
  2638	                friver = _interp(runoff_records)
  2639	
  2640	                # Derived: virtual-T density + solar zenith. Canonical coefficient
  2641	                # 1/epsilon - 1 (~0.608), not the rounded 0.61 (~0.4% drift) — must
  2642	                # match jra55_to_atm_surface / _shared.virtual_temperature.
  2643	                T_v = tas * (1.0 + (1.0 / _const.epsilon - 1.0) * huss)
  2644	                rho_a = psl / (_const.R_d * T_v)
  2645	                # Insolation clock (day-of-year + diurnal hour), which sets the
  2646	                # solar zenith and hence zenith-dependent surface albedo:
  2647	                #   - cycle=True (repeat-year forcing): use the CYCLED forcing
  2648	                #     clock ``day_f`` so the solar geometry stays phase-locked
  2649	                #     to the repeated rsds/rlds records.  Using the RAW ``day``
  2650	                #     drifts the seasonal doy (and, for a non-integer cache
  2651	                #     length, the diurnal hour) whenever the cache length is
  2652	                #     not a whole multiple of 365 days (e.g. a 366-day
  2653	                #     leap-year RYF cache), biasing the surface albedo.
  2654	                #   - cycle=False: ``day_f == day`` (no wrap), so this is
  2655	                #     byte-identical to the raw-day clock — the common
  2656	                #     non-cycled path is unchanged.
  2657	                # The SPINUP RAMP (below) intentionally stays on the RAW
  2658	                # ``day``: it is a function of elapsed run time, not forcing
  2659	                # time.  ``cycle`` is a static Python bool (compile-time
  2660	                # feature gate), so this branch is resolved at trace time.
  2661	                day_insol = day_f if cycle else day
  2662	                doy = jnp.mod(day_insol, 365.0) + 1.0
  2663	                hour = jnp.mod(day_insol, 1.0) * 24.0
  2664	                cos_z = cos_zenith_angle(lat_2d, lon_2d, doy, hour)
  2665	
  2666	                atm = AtmToSurface(
  2667	                    sw_down=rsds, lw_down=rlds,
  2668	                    precip_total=prra + prsn, precip_snow=prsn,
  2669	                    T_lowest=tas, q_lowest=huss,
  2670	                    u_lowest=uas, v_lowest=vas,
  2671	                    p_lowest=psl, p_surface=psl,
  2672	                    rho_lowest=rho_a, cos_zenith=cos_z,
  2673	                    co2_ppmv=jnp.asarray(co2_ppmv, dtype=tas.dtype),
  2674	                    has_radiation=jnp.asarray(1.0, dtype=tas.dtype),
  2675	                    has_precipitation=jnp.asarray(1.0, dtype=tas.dtype),
  2676	                )
  2677	
  2678	                sst_K = state_in.T.data[..., 0] + _const.T_freeze
  2679	                u_o = jnp.zeros_like(sst_K)
  2680	                v_o = jnp.zeros_like(sst_K)
  2681	                tile = ocean_tile_response(atm, sst_K, u_o, v_o, coupler_cfg)
  2682	
  2683	                if enable_ramp:
  2684	                    t_sim = day * 86400.0
  2685	                    ramp = jnp.minimum(1.0, t_sim / T_ramp_seconds)
  2686	                else:
  2687	                    ramp = 1.0
  2688	
  2689	                sw_net = atm.sw_down * (1.0 - tile.albedo)
  2690	                q_net = (sw_net + atm.lw_down - tile.lw_up
  2691	                         - tile.shflx - tile.lhflx)
  2692	
  2693	                _dtype = state_in.T.data.dtype
  2694	                E_rate = tile.lhflx / jnp.asarray(_const.L_v, dtype=_dtype)
  2695	                fw = FreshwaterForcing(
  2696	                    precip=jnp.asarray(prra + prsn, dtype=_dtype),
  2697	                    evap=jnp.asarray(E_rate, dtype=_dtype),
  2698	                    runoff=jnp.asarray(friver, dtype=_dtype),
  2699	                    ice_fw=jnp.zeros_like(sst_K, dtype=_dtype),
  2700	                )
  2701	                sf = OceanSurfaceForcing(
  2702	                    sw_down=atm.sw_down, q_net=q_net,
  2703	                    tau_x=tile.tau_x * ramp, tau_y=tile.tau_y * ramp,
  2704	                    freshwater=None,
  2705	                )
  2706	
  2707	                # Prognostic slab sea ice: partition surface forcing between
  2708	                # open ocean (f_ocean=1-A) and the ice tile.  ocean_mask: land
  2709	                # cells receive no ice->ocean forcing (mask-aware blend
  2710	                # contract; land_mask is scan-carry state, traced-safe).
  2711	                if enable_sea_ice:
  2712	                    new_ice, fw, sf = omip_sea_ice_surface_forcing(
  2713	                        ice_state=ice_in, ice_config=ice_cfg, atm=atm,
  2714	                        ocean_sst_K=sst_K, open_ocean_sf=sf, open_ocean_fw=fw,
  2715	                        dt=dt, grid=None,
  2716	                        ocean_mask=state_in.land_mask.data,
  2717	                    )
  2718	
  2719	                sponge_k = (sponge._replace(gamma=sponge.gamma * ramp)
  2720	                            if enable_sponge else None)
  2721	                new_state = _dyn_step(
  2722	                    state_in, dt, freshwater=fw,
  2723	                    surface_forcing=sf, sponge=sponge_k,
  2724	                )
  2725	
  2726	                if enable_sss:
  2727	                    S = new_state.S.data
  2728	                    _sss_mask = new_state.land_mask.data
  2729	                    S_new = S.at[..., 0].set(
  2730	                        S[..., 0] - sss_alpha_static * (

exec
/usr/bin/bash -lc "sed -n '670,755p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 211ms:
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

exec
/usr/bin/bash -lc 'git diff --check HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py scripts/bench/metadata.py scripts/bench/bench_ocean_latlon_spmd_scaling.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 267ms:
codex
The bench fix itself is sound, but Route-B JRA55 OMIP still reintroduces the same captured-aux failure.

- Existing one-argument `timed_scan_blocks` users (atmosphere and MPAS) remain compatible: `aux=None` wraps them back to one argument.
- Compile/probe/scan consistently use the normalized two-argument form.
- With `step.aux`, `_scan_run` receives already-device, matching-sharding `jax.Array`s; this changes the executable ABI, not host/device transfer. An arbitrary host `aux` would be different, so document that requirement.
- The step cache is not stale-dangerous: `in_specs` fixes the expected aux tree and JAX recompilation keys include dynamic argument avals/shardings. A tree-structure validation would improve diagnostics, but is not required for correctness.
- `make_sharded_ocean_step_global` and `run_omip_core2` call directly from host loops; Core2 explicitly rejects its scan path with multi-GPU, so they are fine as-is.

1. **P1 — `run_omip` JRA55 block scans remain broken on multicontroller.** Both JITted block builders close over `spmd_step` and call it without `aux`: [CPU-interp path](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/run/run_omip.py:2370) → [call](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/run/run_omip.py:2451), and [GPU-interp path](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/run/run_omip.py:2570) → [call](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/run/run_omip.py:2721). Route-B is explicitly enabled when building that step, so this is reachable. Thread `spmd_step.aux` as a dynamic argument of each `block_fn`, pass it to `_dyn_step(..., aux=spmd_aux)`, and supply it at each block invocation. Capturing it in the builder closure would not fix the issue.

2. **P2 — no committed regression coverage exercises the new aux contract.** The timing-helper tests cover only one-argument advances, and the JRA SPMD parity test uses two local CPU devices, where arrays are addressable. Add a two-argument helper test and a two-process outer-JIT/scan regression, ideally covering the JRA block path.

VERDICT: FIX-FIRST
tokens used
107,897
The bench fix itself is sound, but Route-B JRA55 OMIP still reintroduces the same captured-aux failure.

- Existing one-argument `timed_scan_blocks` users (atmosphere and MPAS) remain compatible: `aux=None` wraps them back to one argument.
- Compile/probe/scan consistently use the normalized two-argument form.
- With `step.aux`, `_scan_run` receives already-device, matching-sharding `jax.Array`s; this changes the executable ABI, not host/device transfer. An arbitrary host `aux` would be different, so document that requirement.
- The step cache is not stale-dangerous: `in_specs` fixes the expected aux tree and JAX recompilation keys include dynamic argument avals/shardings. A tree-structure validation would improve diagnostics, but is not required for correctness.
- `make_sharded_ocean_step_global` and `run_omip_core2` call directly from host loops; Core2 explicitly rejects its scan path with multi-GPU, so they are fine as-is.

1. **P1 — `run_omip` JRA55 block scans remain broken on multicontroller.** Both JITted block builders close over `spmd_step` and call it without `aux`: [CPU-interp path](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/run/run_omip.py:2370) → [call](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/run/run_omip.py:2451), and [GPU-interp path](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/run/run_omip.py:2570) → [call](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/run/run_omip.py:2721). Route-B is explicitly enabled when building that step, so this is reachable. Thread `spmd_step.aux` as a dynamic argument of each `block_fn`, pass it to `_dyn_step(..., aux=spmd_aux)`, and supply it at each block invocation. Capturing it in the builder closure would not fix the issue.

2. **P2 — no committed regression coverage exercises the new aux contract.** The timing-helper tests cover only one-argument advances, and the JRA SPMD parity test uses two local CPU devices, where arrays are addressable. Add a two-argument helper test and a two-process outer-JIT/scan regression, ideally covering the JRA block path.

VERDICT: FIX-FIRST
