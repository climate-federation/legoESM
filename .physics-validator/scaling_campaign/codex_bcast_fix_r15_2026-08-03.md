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
session id: 019fc61f-20f9-7b62-81f6-b0e74ecfb70d
--------
user
Round-15 confirm on the broadcast-removal change set (git diff HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py). Your r14 items: (1) float gate strengthened to PER-BAND moments + per-band nonfinite counts via new module-level geom_band_fingerprint/band_fingerprints_agree (exact dtypes: per-band positional 48-bit digests; residual within-band moment-preserving float change documented); (2) unit test tests/ocean/unit/test_sharded_geom_fingerprint.py — 8 tests incl. band-local 1e-3 drift refused, exact permutation refused, bool two-cell flip refused, ULP drift accepted, nonfinite count mismatch refused; ran 8 passed, and tests/parallel/test_latlon_ocean_spmd_step.py + test_latlon_spmd_fused_halo.py = 23 passed total in 282s on the fixed code; (3) three stale doc spots rewritten (module Structure A, function Notes, put-site comment). Check: the rewired guard call (host.shape[0] as n_bands — is host ALWAYS the stacked array at both call sites?), raise-message locals, helper numerics (hash per band, struct concatenation order stable), test quality (any vacuous test?), leftover stale wording. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
codex
I’ll inspect the requested diff and surrounding call sites, then verify the helper/test logic for the specific failure modes you flagged.
exec
/usr/bin/bash -lc "git diff HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py && printf '\\n--- call-site context ---\\n' && rg -n -C 8 'band_fingerprints_agree|geom_band_fingerprint|_assert.*geom|geom.*fingerprint' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py && printf '\\n--- test file ---\\n' && sed -n '1,360p' tests/ocean/unit/test_sharded_geom_fingerprint.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 291ms:
diff --git a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
index 0ea2c649a..5ee227920 100644
--- a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
+++ b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
@@ -11,12 +11,13 @@ cubed-sphere ``parallel.sharded_dynamics.make_sharded_step``.
 
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
@@ -218,6 +219,69 @@ def _content_hash48(arr) -> float:
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
+      DOCUMENTED RESIDUAL: a within-band float change preserving all three
+      moments to rtol is not detected; band grids are analytic in lat/lon,
+      so any real inconsistency moves those moments.
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
@@ -501,9 +565,9 @@ def make_sharded_ocean_step(model, mesh):
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
@@ -556,13 +620,15 @@ def make_sharded_ocean_step(model, mesh):
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
@@ -581,49 +647,41 @@ def make_sharded_ocean_step(model, mesh):
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

--- call-site context ---
214-    beyond the ~10 setup-time comparisons this guard makes.
215-    """
216-    import hashlib
217-
218-    a = np.ascontiguousarray(arr)
219-    h = hashlib.blake2b(a.tobytes(), digest_size=6)
220-    return float(int.from_bytes(h.digest(), "big"))
221-
222:def geom_band_fingerprint(host, n_bands):
223-    """Low-memory cross-process fingerprint of a band-STACKED geometry field.
224-
225-    ``host`` has leading axis ``n_bands`` (the per-device band stack). The
226-    fingerprint is PER BAND (codex 2026-08-03 r14: whole-array moments let a
227-    band-local drift hide in the global sum once each process's own bytes
228-    become live computation inputs):
229-
230-    * exact dtypes (int/bool/uint — masks, index tables): one positional
--
260-                float(f64.sum()) if f64.size else 0.0,
261-                float((f64 * f64).sum()) if f64.size else 0.0,
262-                float(_np.abs(f64).max()) if f64.size else 0.0,
263-            ])
264-        vals = _np.array(per_band, dtype=_np.float64)
265-    return _np.array(struct, dtype=_np.float64), vals, is_exact
266-
267-
268:def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=1e-5):
269-    """True iff every process's fingerprint matches process 0's.
270-
271-    ``g_struct``/``g_vals`` are the ``process_allgather``-ed outputs of
272:    :func:`geom_band_fingerprint` (leading axis = process). Structure and
273-    exact-dtype digests compare EXACTLY; float moments to ``rtol``.
274-    """
275-    import numpy as _np
276-
277-    struct_ok = bool(_np.all(g_struct == g_struct[0]))
278-    if is_exact:
279-        vals_ok = bool(_np.all(g_vals == g_vals[0]))
280-    else:
--
650-            # PER-BAND fingerprints (module-level, unit-tested): exact
651-            # dtypes hash positionally per band; floats compare per-band
652-            # moments to rtol 1e-5 — bounds each band's drift instead of
653-            # letting it hide in a whole-array sum, since each process's
654-            # own bytes are now the live inputs for the bands it owns
655-            # (codex r14). A mask that genuinely differs across processes
656-            # means different wet domains = different physics: refusing is
657-            # correct, not a false alarm.
658:            struct, vals, is_exact = geom_band_fingerprint(
659-                host, host.shape[0])
660-            g_struct = multihost_utils.process_allgather(struct)
661-            g_vals = multihost_utils.process_allgather(vals)
662:            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
663-                raise RuntimeError(
664-                    f"make_sharded_ocean_step: band-geometry field {name!r} "
665-                    f"DIVERGES across processes (exact_dtype={is_exact}, "
666-                    f"gathered={g_vals.tolist()}) — a real config/grid "
667-                    f"inconsistency, not autotune noise; refusing to "
668-                    f"shard it.")
669-            # NO broadcast_one_to_all here (removed 2026-08-03): its psum
670-            # program is [n_processes, stack] in / P() fully-replicated out,

--- test file ---
"""Unit tests for the band-geometry cross-process fingerprint gate.

The gate decides whether ``make_sharded_ocean_step`` accepts per-process
band-geometry stacks without the (removed, nd-linear-cost) process-0
broadcast — see the 2026-08-03 fix note at the sharded put. These tests
pin the gate's discrimination properties single-process (the
multicontroller allgather wiring is exercised by the distributed suite).
"""
import numpy as np
import pytest

from legoesm.ocean.dynamics.sharded_ocean_step import (
    band_fingerprints_agree,
    geom_band_fingerprint,
)

N_BANDS = 4
SHAPE = (N_BANDS, 6, 8)


def _gather(*hosts):
    """Simulate process_allgather over per-process fingerprints."""
    fps = [geom_band_fingerprint(h, N_BANDS) for h in hosts]
    exacts = {fp[2] for fp in fps}
    assert len(exacts) == 1
    g_struct = np.stack([fp[0] for fp in fps])
    g_vals = np.stack([fp[1] for fp in fps])
    return g_struct, g_vals, fps[0][2]


def test_identical_float_stacks_agree():
    rng = np.random.default_rng(0)
    a = rng.normal(size=SHAPE).astype(np.float32)
    assert band_fingerprints_agree(*_gather(a, a.copy()))


def test_ulp_scale_drift_agrees():
    rng = np.random.default_rng(1)
    a = rng.normal(size=SHAPE).astype(np.float64) + 10.0
    b = a * (1.0 + 1e-9)  # autotune-class last-ULP drift
    assert band_fingerprints_agree(*_gather(a, b))


def test_band_local_drift_refused():
    # A 1e-3-relative change in ONE band must be caught even though the
    # whole-array moments would dilute it (the r14 per-band rationale).
    rng = np.random.default_rng(2)
    a = rng.normal(size=SHAPE).astype(np.float64) + 10.0
    b = a.copy()
    b[2] *= 1.0 + 1e-3
    assert not band_fingerprints_agree(*_gather(a, b))


def test_exact_dtype_permutation_refused():
    # Moment fingerprints are blind to permutations; the positional
    # per-band byte digest must not be.
    a = np.zeros(SHAPE, dtype=np.int32)
    a[1, 2, 3] = 1
    b = np.zeros_like(a)
    b[1, 3, 2] = 1  # same count, different position, same band
    assert not band_fingerprints_agree(*_gather(a, b))


def test_bool_mask_two_cell_flip_refused():
    a = np.zeros(SHAPE, dtype=bool)
    a[0, 0, 0] = True
    b = a.copy()
    b[0, 0, 0] = False
    b[0, 5, 7] = True  # true-count preserved
    assert not band_fingerprints_agree(*_gather(a, b))


def test_nonfinite_count_mismatch_refused():
    a = np.ones(SHAPE, dtype=np.float32)
    b = a.copy()
    b[3, 0, 0] = np.nan  # struct carries per-band non-finite counts
    assert not band_fingerprints_agree(*_gather(a, b))


def test_shape_mismatch_refused():
    a = np.ones(SHAPE, dtype=np.float32)
    b = np.ones((N_BANDS, 6, 9), dtype=np.float32)
    fa = geom_band_fingerprint(a, N_BANDS)
    fb = geom_band_fingerprint(b, N_BANDS)
    # Different shapes -> different struct lengths; the agree helper is
    # only called on stackable gathers, so assert the structs differ.
    assert fa[0].shape != fb[0].shape or not np.array_equal(fa[0], fb[0])


def test_wrong_leading_axis_asserts():
    with pytest.raises(AssertionError):
        geom_band_fingerprint(np.ones((3, 2)), N_BANDS)

exec
/usr/bin/bash -lc "sed -n '180,300p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py && sed -n '520,760p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py && rg -n -C 6 '_replicated_put\\(' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py && rg -n -C 5 'build_band_grids|band_grids|band.*geometry|geom_' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 247ms:
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
      DOCUMENTED RESIDUAL: a within-band float change preserving all three
      moments to rtol is not detected; band grids are analytic in lat/lon,
      so any real inconsistency moves those moments.

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

    # --- host-side band geometries + vertex masks (replicated, indexed in-body) ---
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
    # — same stack, different placement; the process-0 broadcast +
    # divergence guard below runs on HOST values and is placement-blind.
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
616-    # 26524423). The leading axis has length n_dev, so P("lat") divides it
617-    # exactly; the body indexes its local slab at [0]. Values are unchanged
618-    # — same stack, different placement; the process-0 broadcast +
619-    # divergence guard below runs on HOST values and is placement-blind.
620-    rep = NamedSharding(mesh, P("lat"))
621-
622:    def _replicated_put(arr, name):
623-        # (Name kept for history; since 2026-08-03 this is a SHARDED stack
624-        # put.) The band-geometry arrays are (re)computed per process and
625-        # can differ in their last ULPs (per-process XLA autotuning on
626-        # device-derived grid fields) — the fully-replicated-put era
627-        # broadcast process 0's bytes to sidestep the P() bit-identity
628-        # assert (job 26450848). With the #1370-iii P("lat") sharding each
--
698-                "across processes (field list / x64 setting / device count "
699-                f"— gathered {_g.tolist()}). Fix the per-process config "
700-                "before sharding; the per-field checks below assume one "
701-                "schema.")
702-
703-    geom_stacks = {
704:        name: _replicated_put(
705-            jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
706-                      axis=0), name)
707-        for name in array_field_names
708-    }
709:    vmask_stack = _replicated_put(
710-        jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0),
711-        "vertex_mask")
712-
713-    # Static perms for the v north-boundary-row ppermute (band r receives band
714-    # r+1's v_lower[0] = global v[e]; north band non-target receives 0).
715-    perm_north, _perm_south = latlon_band_perms(n_dev)
11-
12-Architecture (the two non-trivial pieces — see ``omip-multinode-spmd-scope``):
13-
14-* GRID = **band-stacked, P("lat")-SHARDED** (Structure A, sharded since
15-  #1370-iii). The ``N`` band geometries are built host-side
16:  (``build_band_grids``), their ARRAY fields are ``jnp.stack``-ed over a
17-  leading band axis, sharded ``P("lat")`` so each device holds ONLY its own
18-  band's slab, and the in-``shard_map`` body reads its local slab at
19-  ``[0]``. This AVOIDS staggered-sharding the grid itself (the v-row is
20-  ``n_lat+1``, coprime with ``n_lat`` for ``N>1``). The
21-  ``LatLonCGridGeometry`` SCALAR fields (``n_lat``, ``n_lon``, ``radius``,
22-  ``dlon``, ``dlat``, ``fold``) stay STATIC python values — they gate trace-time
23-  ``if``\ s and ``jnp.zeros((n_lat, ...))`` shape builds, so a traced ``n_lat``
24-  would break the step. Only the ``jax.Array`` fields are stacked/indexed; each
25:  band geometry is rebuilt as ``template._replace(**indexed_arrays)`` so the
26-  scalars come from a band template (all bands share ``n_lat_local = n_lat/N``).
27-
28-* STATE = cell fields shard ``P("lat")``; the STAGGERED ``v`` / ``v_mask``
29-  (shape ``(n_lat+1, n_lon, ...)``) use the **band-local v-faces** layout
30-  (Structure B). The sharded state carries ``v_lower = v[0:n_lat]`` (``n_lat``
--
87-# to the band's nl+1 faces in-body.  All OTHER array state fields are cell/u
88-# leading-dim n_lat and shard P("lat") directly.
89-_V_STAGGERED_STATE_FIELDS = ("v", "v_mask")
90-
91-
92:def _geom_array_field_names(geom):
93-    """The ``jax.Array`` field names of a ``LatLonCGridGeometry`` (everything
94-    except the static scalars + the ``fold`` descriptor).  Order-stable
95-    (the NamedTuple field order) so the host stack and the in-body index agree."""
96-    names = []
97-    for name in geom._fields:
--
118-    ``_ab2_step`` + the static-gated post-steps), the un-jitted twin of
119-    ``LatLonCGridOceanModel._step_jitted``.
120-
121-    The shard_map body must call THIS, not ``model.step`` — ``model.step`` goes
122-    through ``_step_jitted`` (``@jax.jit``), a NESTED jit that takes ``grid`` as a
123:    TRACED argument, so the band geometry's STATIC scalars (``fold.is_active``,
124-    ``fold.fold_j``, ``n_lat`` …) become tracers and the operators' trace-time
125-    ``if fold.is_active`` / ``jnp.zeros((n_lat, …))`` blow up
126-    (``TracerBoolConversionError``).  shard_map already provides the JIT boundary,
127-    so running the un-jitted body keeps ``grid`` a concrete Python-scalar-carrying
128-    pytree inside the single trace — the model docstring's own "call ``_step_impl``
--
159-    if getattr(model.config, "ew_cyclic_overlap", False):
160-        new_state = model._apply_ew_cyclic_overlap(new_state)
161-    return new_state
162-
163-
164:def build_band_grids(grid, n_devices: int):
165-    """Build the ``n_devices`` UNIFORM lat-band geometries for the SPMD step,
166-    reusing the tested MPI band slicers (no bespoke re-derivation).
167-
168-    Requires ``grid.n_lat % n_devices == 0`` so every band has the same leading
169-    shape — a ``shard_map`` body is ONE program, so the per-band grids must be
--
217-
218-    a = np.ascontiguousarray(arr)
219-    h = hashlib.blake2b(a.tobytes(), digest_size=6)
220-    return float(int.from_bytes(h.digest(), "big"))
221-
222:def geom_band_fingerprint(host, n_bands):
223:    """Low-memory cross-process fingerprint of a band-STACKED geometry field.
224-
225-    ``host`` has leading axis ``n_bands`` (the per-device band stack). The
226-    fingerprint is PER BAND (codex 2026-08-03 r14: whole-array moments let a
227-    band-local drift hide in the global sum once each process's own bytes
228-    become live computation inputs):
--
267-
268-def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=1e-5):
269-    """True iff every process's fingerprint matches process 0's.
270-
271-    ``g_struct``/``g_vals`` are the ``process_allgather``-ed outputs of
272:    :func:`geom_band_fingerprint` (leading axis = process). Structure and
273-    exact-dtype digests compare EXACTLY; float moments to ``rtol``.
274-    """
275-    import numpy as _np
276-
277-    struct_ok = bool(_np.all(g_struct == g_struct[0]))
--
300-         float(n_dev)], dtype=np.float64)
301-
302-
303-def _build_band_vertex_masks(model, n_dev):
304-    """Per-band vertex masks (n_lat/N+1, n_lon+1): SLICE the model's primed GLOBAL
305:    vertex mask ``[s : e+1]`` per band (the v/q-row stagger ``slice_cgrid_geometry_
306-    to_band`` uses for ``area_q``).
307-
308-    The slice — NOT a per-band ``compute_vertex_mask`` — is what makes the
309-    band-cut rows correct.  The global ``model._vertex_mask`` already encoded the
310-    four-cell product at EVERY vertex row including the interior cut rows (it saw
--
574-    replication check) because the band halo intentionally reads neighbour-rank
575-    data (replication-unaware).
576-
577-    The state must be laid out with :func:`shard_state_latlon` (``v`` /
578-    ``v_mask`` carried as the ``n_lat``-row ``v_lower``).  The body reconstructs
579:    each band's ``nl+1`` v-faces, runs the step on the band geometry, and
580-    converts the result back to the ``v_lower`` representation.
581-    """
582-    if mesh is None:                   # single-device: plain step
583-        return lambda state, dt, **forcing_kwargs: model.step(
584-            state, dt, **forcing_kwargs)
--
589-    # Tripole north-fold (scaling-audit item 4): SUPPORTED under the same
590-    # v-carrier contract as the regular grid.  Every fold-touching operator
591-    # is already uniform-program fold-capable (the data-dependent
592-    # ``north_fold_mask``/``apply_north_fold`` selection on
593-    # ``axis_index == N-1`` — gated by test_latlon_spmd_northfold), and
594:    # ``build_band_grids``' slicer keeps ``is_active`` rank-consistent with
595-    # the ``fold_j=-1`` sentinel off the north band.  The one structural
596-    # assumption is the v-carrier's: the TOP v-face row (the seam/cap row,
597-    # ``v[n_lat]``) must be WALL-MASKED so the in-body reconstruction's
598-    # zero row is exact — true for the cap-row convention of
599-    # ``create_synthetic_tripole`` and the eORCA masks (``v_mask[-1] == 0``;
--
601-    # asserted on the CONCRETE state in :func:`shard_state_latlon` — a live
602-    # (unmasked) seam v-row refuses loudly there instead of silently
603-    # reconstructing zeros here.
604-
605-    # --- host-side band geometries + vertex masks (replicated, indexed in-body) ---
606:    band_grids = build_band_grids(model.grid, n_dev)
607-    band_vmasks = _build_band_vertex_masks(model, n_dev)
608:    template = band_grids[0]           # static-scalar source (uniform bands)
609:    array_field_names = _geom_array_field_names(template)
610-
611-    # Stack each geometry ARRAY field over the band axis (rank 0..N-1) and
612-    # SHARD along that axis (#1370 stage (iii), codex round-18): each device
613-    # holds ONLY its own band's slab instead of the whole global stack —
614-    # this was one of the residual ~1.4-1.7 global-field-equivalents of
--
619-    # divergence guard below runs on HOST values and is placement-blind.
620-    rep = NamedSharding(mesh, P("lat"))
621-
622-    def _replicated_put(arr, name):
623-        # (Name kept for history; since 2026-08-03 this is a SHARDED stack
624:        # put.) The band-geometry arrays are (re)computed per process and
625-        # can differ in their last ULPs (per-process XLA autotuning on
626-        # device-derived grid fields) — the fully-replicated-put era
627-        # broadcast process 0's bytes to sidestep the P() bit-identity
628-        # assert (job 26450848). With the #1370-iii P("lat") sharding each
629-        # process's devices consume ONLY its own band rows, so the
--
653-            # letting it hide in a whole-array sum, since each process's
654-            # own bytes are now the live inputs for the bands it owns
655-            # (codex r14). A mask that genuinely differs across processes
656-            # means different wet domains = different physics: refusing is
657-            # correct, not a false alarm.
658:            struct, vals, is_exact = geom_band_fingerprint(
659-                host, host.shape[0])
660-            g_struct = multihost_utils.process_allgather(struct)
661-            g_vals = multihost_utils.process_allgather(vals)
662-            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
663-                raise RuntimeError(
664:                    f"make_sharded_ocean_step: band-geometry field {name!r} "
665-                    f"DIVERGES across processes (exact_dtype={is_exact}, "
666-                    f"gathered={g_vals.tolist()}) — a real config/grid "
667-                    f"inconsistency, not autotune noise; refusing to "
668-                    f"shard it.")
669-            # NO broadcast_one_to_all here (removed 2026-08-03): its psum
--
692-
693-        _g = _mhu.process_allgather(
694-            _schema_fingerprint(list(array_field_names), n_dev))
695-        if not bool(np.all(_g == _g[0])):
696-            raise RuntimeError(
697:                "make_sharded_ocean_step: the band-geometry SCHEMA differs "
698-                "across processes (field list / x64 setting / device count "
699-                f"— gathered {_g.tolist()}). Fix the per-process config "
700-                "before sharding; the per-field checks below assume one "
701-                "schema.")
702-
703:    geom_stacks = {
704-        name: _replicated_put(
705:            jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
706-                      axis=0), name)
707-        for name in array_field_names
708-    }
709-    vmask_stack = _replicated_put(
710-        jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0),
--
712-
713-    # Static perms for the v north-boundary-row ppermute (band r receives band
714-    # r+1's v_lower[0] = global v[e]; north band non-target receives 0).
715-    perm_north, _perm_south = latlon_band_perms(n_dev)
716-
717:    def _body(state_local, forcing_local, geom_stacks_local,
718-              vmask_stack_local, dt):
719-        fw_local, sf_local, sponge_local, t_s_local = forcing_local
720-        r = jax.lax.axis_index(axis)
721-
722:        # Rebuild this band's geometry. The stacks are SHARDED P("lat") on
723-        # their leading band axis, so inside shard_map each device's local
724-        # view is its own (1, ...) slab — index [0], NOT [r] (stage (iii);
725-        # [r] was the replicated-stack indexing). Static scalars
726-        # (n_lat_local, n_lon, radius, dlon, dlat, fold) come from the band
727-        # template. The fold is inactive + identical on every band here.
728:        geom_arrays = {name: geom_stacks_local[name][0]
729-                       for name in array_field_names}
730:        band_geom = template._replace(**geom_arrays)
731-        band_vmask = vmask_stack_local[0]
732-
733-        # Reconstruct the band's nl+1 v-faces from the nl-row v_lower.  band r's
734-        # north boundary row is band r+1's v_lower[0] (= global v[e]); the north
735-        # band (no r+1) receives the pole-wall 0 via the ppermute non-target.
--
857-            # scalars (t_seconds) replicate, exactly like dt.
858-            forcing_spec = jax.tree.map(_lat_spec, forcing)
859-            # Stage (iii): the stacks are banded on their leading axis, so
860-            # the shard_map spec matches their P("lat") placement (the body
861-            # indexes its local (1, ...) slab at [0]).
862:            geom_spec = jax.tree.map(lambda _x: P("lat"), geom_stacks)
863-            vmask_spec = P("lat")
864-            # JAX >= 0.8 top-level shard_map takes ``check_vma`` (the
865-            # replication check); the band halo reads neighbour-rank data so
866-            # disable it (same as the validated PCG / halo-parity shard_maps).
867-            fn = jax.jit(shard_map(
868-                _body,
869-                mesh=mesh,
870:                in_specs=(in_spec, forcing_spec, geom_spec, vmask_spec, P()),
871-                out_specs=in_spec,
872-                check_vma=False,
873-            ))
874-            _cache[key] = fn
875-        # Arm the SPMD halo backend ONLY around the call, then RESTORE the
--
891-        _prev_backend = get_halo_backend()
892-        _prev_topo = get_mpi_topology()
893-        _prev_mesh = get_spmd_mesh()
894-        activate_latlon_spmd_halo(mesh)
895-        try:
896:            return fn(state, forcing, geom_stacks, vmask_stack,
897-                      jnp.asarray(dt))
898-        finally:
899-            set_spmd_mesh(_prev_mesh)
900-            set_halo_backend(_prev_backend, _prev_topo)
901-

exec
/usr/bin/bash -lc "python - <<'PY'
import inspect
from jax.experimental import multihost_utils
print(inspect.getsourcefile(multihost_utils.process_allgather))
print(inspect.getsource(multihost_utils.process_allgather))
PY
printf '\\n--- imports / formatting / stale text ---\\n'
sed -n '1,125p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
rg -n -i -C 2 'replicat|broadcast process 0|broadcast_one_to_all|whole-array|indexed in-body' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
git diff --check HEAD -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py
git status --short" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 436ms:
/usr/bin/bash: cannot create temp file for here-document: Read-only file system

--- imports / formatting / stale text ---
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
105-
106-def _lat_spec(x):
107:    """Lat-band PartitionSpec for an array leaf: shard axis 0, replicate rest."""
108-    nd = int(getattr(x, "ndim", np.ndim(x)))
109-    if nd < 1:
110:        return P()                     # scalar -> replicated
111-    return P("lat", *((None,) * (nd - 1)))
112-
--
224-
225-    ``host`` has leading axis ``n_bands`` (the per-device band stack). The
226:    fingerprint is PER BAND (codex 2026-08-03 r14: whole-array moments let a
227-    band-local drift hide in the global sum once each process's own bytes
228-    become live computation inputs):
--
405-        else:
406-            # Non-Field, non-None leaf (e.g. a raw array carry like psi).
407:            # Shard 2-D+ on lat, replicate lower-rank — matches shard_pytree.
408-            arr = jnp.asarray(val)
409-            spec = _lat_spec(arr) if arr.ndim >= 1 else P()
--
419-    cell->face wind-stress interpolation happens INSIDE the step through the
420-    SPMD-aware halo pads), so array leaves shard ``P("lat", None, ...)`` with
421:    no ``v_lower`` handling; scalars replicate; ``None`` fields pass through
422-    untouched (they vanish from the pytree structure, matching the specs the
423-    step derives).  ``forcing=None`` returns ``None``.
--
443-    block builders stack ``N`` steps / raw records along a LEADING axis,
444-    so the lat axis sits at position 1.  A ``(n_rec, n_lat, n_lon[, ...])``
445:    leaf therefore shards ``P(None, "lat", ...)`` (records replicated, the
446-    time index stays shard-local so the in-scan interpolation needs no
447-    cross-band comm); a bare ``(n_lat, n_lon)`` leaf shards ``P("lat", None)``;
448:    1-D metadata / scalars replicate; ``None`` and non-array leaves pass
449-    through.  ``mesh=None`` returns ``stack`` unchanged (serial lane).
450-
--
453-    metadata that is genuinely rank-2 but NOT lat-major (e.g. a
454-    ``(n_rec, n_meta)`` table) would be silently mis-sharded — keep such
455:    metadata 1-D (or replicate it explicitly) before it reaches this helper.
456-
457-    Keeping this next to :func:`shard_state_latlon` means the driver and
--
506-
507-    Multi-controller (route-B ``jax.distributed``, mesh spanning processes):
508:    replication routes through a jit-compiled identity instead of
509:    ``device_put`` (see :func:`legoesm.parallel.latlon_spmd.replicate_leaf`,
510-    the primitive shared with the atm gather); the single-process path is
511-    byte-unchanged.
512-    """
513:    from legoesm.parallel.latlon_spmd import replicate_leaf
514-
515-    rep = NamedSharding(mesh, P())
--
517-
518-    def _gather_arr(a):
519:        return replicate_leaf(a, rep, multiprocess=_mp)
520-
521-    updates = {}
--
549-    pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
550-    ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
551:    is a replicated scalar like ``dt``).  ``None`` forcing keeps the
552-    dynamics-only program; each distinct None<->populated combination
553-    compiles (and caches) its own executable.
--
569-    holds only its own slab; the ``shard_map`` body reads it at ``[0]``; the
570-    geometry SCALAR fields stay static — see the module docstring).  ``dt`` is a TRACED,
571:    replicated operand and the jitted ``shard_map`` is built once and cached
572-    (a per-call rebuild re-traced the whole ocean step every call — see the
573-    ``_cache`` note below).  ``check_vma=False`` (the JAX >= 0.8
574:    replication check) because the band halo intentionally reads neighbour-rank
575:    data (replication-unaware).
576-
577-    The state must be laid out with :func:`shard_state_latlon` (``v`` /
--
603-    # reconstructing zeros here.
604-
605:    # --- host-side band geometries + vertex masks (replicated, indexed in-body) ---
606-    band_grids = build_band_grids(model.grid, n_dev)
607-    band_vmasks = _build_band_vertex_masks(model, n_dev)
--
620-    rep = NamedSharding(mesh, P("lat"))
621-
622:    def _replicated_put(arr, name):
623-        # (Name kept for history; since 2026-08-03 this is a SHARDED stack
624-        # put.) The band-geometry arrays are (re)computed per process and
625-        # can differ in their last ULPs (per-process XLA autotuning on
626:        # device-derived grid fields) — the fully-replicated-put era
627:        # broadcast process 0's bytes to sidestep the P() bit-identity
628-        # assert (job 26450848). With the #1370-iii P("lat") sharding each
629-        # process's devices consume ONLY its own band rows, so the
--
651-            # dtypes hash positionally per band; floats compare per-band
652-            # moments to rtol 1e-5 — bounds each band's drift instead of
653:            # letting it hide in a whole-array sum, since each process's
654-            # own bytes are now the live inputs for the bands it owns
655-            # (codex r14). A mask that genuinely differs across processes
--
667-                    f"inconsistency, not autotune noise; refusing to "
668-                    f"shard it.")
669:            # NO broadcast_one_to_all here (removed 2026-08-03): its psum
670:            # program is [n_processes, stack] in / P() fully-replicated out,
671-            # so its logical arg bytes are nd x the global stack — 82.4 GB
672-            # at nd=96 and 109.6 GB at nd=128 for one 3-D field, the
--
702-
703-    geom_stacks = {
704:        name: _replicated_put(
705-            jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
706-                      axis=0), name)
707-        for name in array_field_names
708-    }
709:    vmask_stack = _replicated_put(
710-        jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0),
711-        "vertex_mask")
--
723-        # their leading band axis, so inside shard_map each device's local
724-        # view is its own (1, ...) slab — index [0], NOT [r] (stage (iii);
725:        # [r] was the replicated-stack indexing). Static scalars
726-        # (n_lat_local, n_lon, radius, dlon, dlat, fold) come from the band
727-        # template. The fold is inactive + identical on every band here.
--
855-            in_spec = jax.tree.map(_lat_spec, state)
856-            # Forcing leaves are cell-centered -> plain lat-band specs;
857:            # scalars (t_seconds) replicate, exactly like dt.
858-            forcing_spec = jax.tree.map(_lat_spec, forcing)
859-            # Stage (iii): the stacks are banded on their leading axis, so
--
863-            vmask_spec = P("lat")
864-            # JAX >= 0.8 top-level shard_map takes ``check_vma`` (the
865:            # replication check); the band halo reads neighbour-rank data so
866-            # disable it (same as the validated PCG / halo-parity shard_maps).
867-            fn = jax.jit(shard_map(
 M packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
 M scripts/plot/plot_scaling_paper_figure.py
?? .physics-validator/scaling_campaign/codex_bcast_fix_r14_2026-08-03.md
?? .physics-validator/scaling_campaign/codex_bcast_fix_r15_2026-08-03.md
?? tests/ocean/unit/test_sharded_geom_fingerprint.py

exec
/usr/bin/bash -lc "python -c 'import inspect; from jax.experimental import multihost_utils; print(inspect.getsourcefile(multihost_utils.process_allgather)); print(inspect.getsource(multihost_utils.process_allgather))'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 127 in 267ms:
/usr/bin/bash: python: command not found

exec
/usr/bin/bash -lc "python3 -c 'import inspect; from jax.experimental import multihost_utils; print(inspect.getsourcefile(multihost_utils.process_allgather)); print(inspect.getsource(multihost_utils.process_allgather))'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 1 in 263ms:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'jax'

codex
Issues found:

- `test_band_local_drift_refused` does not prove the new per-band gate: a 0.1% change in one of four bands changes whole-array moments by ~2.5e-4, already above the old `rtol=1e-5` threshold. Use a smaller band-local drift with an unaffected band holding the global `absmax`, and show the old global fingerprint would accept it.
- `test_ulp_scale_drift_agrees` uses a `1e-9` relative change—millions of float64 ULPs near 10—not a ULP-scale change. Use `np.nextafter` or rename it.
- Stale wording remains at [sharded_ocean_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:605) (“replicated, indexed”) and [line 618](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:618) (“broadcast … runs”).
- The documented float residual should also acknowledge that changed nonfinite positions/types with unchanged per-band counts pass the gate.

Confirmed: both `_replicated_put` callers supply `jnp.stack(..., axis=0)` arrays, so `host.shape[0]` is the correct `n_bands`. Per-band digest/moment ordering is deterministic, and the raise message has no undefined locals (though structural failures only display `g_vals`, limiting diagnostics).

VERDICT: FIX-FIRST
tokens used
35,564
Issues found:

- `test_band_local_drift_refused` does not prove the new per-band gate: a 0.1% change in one of four bands changes whole-array moments by ~2.5e-4, already above the old `rtol=1e-5` threshold. Use a smaller band-local drift with an unaffected band holding the global `absmax`, and show the old global fingerprint would accept it.
- `test_ulp_scale_drift_agrees` uses a `1e-9` relative change—millions of float64 ULPs near 10—not a ULP-scale change. Use `np.nextafter` or rename it.
- Stale wording remains at [sharded_ocean_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:605) (“replicated, indexed”) and [line 618](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:618) (“broadcast … runs”).
- The documented float residual should also acknowledge that changed nonfinite positions/types with unchanged per-band counts pass the gate.

Confirmed: both `_replicated_put` callers supply `jnp.stack(..., axis=0)` arrays, so `host.shape[0]` is the correct `n_bands`. Per-band digest/moment ordering is deterministic, and the raise message has no undefined locals (though structural failures only display `g_vals`, limiting diagnostics).

VERDICT: FIX-FIRST
