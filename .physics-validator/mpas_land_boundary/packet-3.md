You are an independent adversarial reviewer for LegoESM (JAX-native differentiable ESM).
ROUND-3 CLOSING review of the MPAS land surface boundary + mass-weighted mean-T change.
In rounds 1-2 you found F1-F5 and F-B1..F-B7. This round verifies the fixes. FINDINGS ONLY —
do NOT modify any file. Read the actual source; cite file:line; label CONFIRMED/PLAUSIBLE +
severity. mpi4py is NOT installed in this venv, so tests/unit/test_mpas_global_diag.py is
SKIPPED here — reason about whether it would PASS at an MPI CI where mpi4py IS present.

## Round-3 deltas to verify (repo cwd; working tree)
- F-B2: static flat-topo guard in `ExperimentConfig.validate_strict`
  (packages/coupler/legoesm/driver/config.py ~1743) is now
  `... and self.topography == "flat" and not self.land_mask_path`; the runtime all-zero
  guard in `_run_mpas` (model_driver.py ~5852) stays authoritative.
- F-B3: `_is_mpas = (d.discretization == "mpas" or normalize_grid_type(g.grid_type) == "mpas")`
  (config.py ~1716). `normalize_grid_type` is a module-level function in config.py.
- F-B4: tests/unit/test_mpas_global_diag.py updated — `_make_stub(..., nlev=3)` now sets
  `drv.sigma = create_sigma_coordinate(nlev)`; expected mean via `_expected_mean_T`
  (pressure_at_half -> dp -> owned Σ(T·dp)/Σ(dp)); new halo-NaN assertions for BOTH T and p_s.
- F-B6: `Tdp_sum_l = jnp.sum(jnp.where(om_c[:,None], T_data * _dp_d, 0.0))` (model_driver.py
  ~5340) — the where() now wraps the PRODUCT so a non-finite halo p_s cannot leak via 0*NaN.
- F-B1: comments relabeled "pressure-weighted (equal cell weight, quasi-uniform SCVT)";
  areaCell weighting deferred.
- F-B5: intentionally NOT changed (spectral-shared short-run fallback, smoke-length runs).

## Specific asks (rank CONFIRMED/PLAUSIBLE)
1. Do the updated tests in tests/unit/test_mpas_global_diag.py actually PASS at MPI CI?
   Check the shape consistency between each test's `T` array and the stub's
   `sigma = create_sigma_coordinate(nlev)` (i.e. does `T_data * _dp_d` broadcast?). Note the
   stub default is nlev=3 while some tests build `T = (6, 2)`.
2. Is the F-B2 two-layer guard (static skips when land_mask_path set; runtime jnp.any is
   authoritative) correct for: flat+no-mask+knobs; flat+mask+knobs; flat+degenerate-all-ocean-
   mask+knobs?
3. Is `normalize_grid_type` in scope at the F-B3 call site, and does it cover all Voronoi
   aliases (voronoi/icosahedral/ico/mpas_voronoi)?
4. Does the F-B6 where()-wrapped product fully close the 0*NaN halo leak? Any residual path
   (dp_sum_l, finite flag) still exposed?
5. Any remaining inconsistency between the MPI method docstring (line ~5318 "MASS-WEIGHTED")
   and the relabeled body comments ("pressure-weighted, equal cell weight")?
6. Anything you would still BLOCK the 60-day A/B launch on.

## The round-3 cumulative working-tree diff (3 files)
```diff
diff --git a/packages/coupler/legoesm/driver/config.py b/packages/coupler/legoesm/driver/config.py
index f33842ec5..27f78b05e 100644
--- a/packages/coupler/legoesm/driver/config.py
+++ b/packages/coupler/legoesm/driver/config.py
@@ -1705,7 +1705,14 @@ class ExperimentConfig(NamedTuple):
         # there — accepting those flags on MPAS ran a 60-day A/B against a
         # byte-identical twin (2026-07-23).  Conversely the MPAS land boundary
         # knobs are consumed only by the MPAS lane.
-        _is_mpas = d.discretization == "mpas"
+        # Lane detection keys on BOTH fields: the _run_mpas dispatch actually
+        # keys on grid_type (any Voronoi alias), and discretization stays
+        # consistent only via run_amip's postprocessor; a mismatched pair is
+        # fail-closed at the component factory, but the guard here must not
+        # emit a wrong-lane message for grid_type-keyed configs (codex F4,
+        # alias set via normalize_grid_type per codex F-B3).
+        _is_mpas = (d.discretization == "mpas"
+                    or normalize_grid_type(g.grid_type) == "mpas")
         if _is_mpas:
             for _flag in ("slab_land_active", "land_soil_bucket",
                           "surface_tiled"):
@@ -1717,6 +1724,34 @@ class ExperimentConfig(NamedTuple):
                         "knobs instead: mpas_land_lapse_K_per_km / "
                         "mpas_land_beta."
                     )
+            # Inert-corner rejection (codex F1-F3): each knob needs the
+            # machinery it modifies to actually be on.
+            if (self.mpas_land_lapse_K_per_km > 0.0
+                    and self.radiation == "none"):
+                errors.append(
+                    "mpas_land_lapse_K_per_km adjusts the SST-forcing "
+                    "surface anchor, which is only built when radiation != "
+                    "'none' — the knob would be silently inert."
+                )
+            if self.mpas_land_beta != 1.0 and self.turbulence == "none":
+                errors.append(
+                    "mpas_land_beta throttles the turbulence surface "
+                    "humidity; turbulence='none' has no surface latent flux "
+                    "to throttle — the knob would be silently inert."
+                )
+            # flat topography yields all-zero f_land UNLESS an explicit land
+            # mask overrides it (codex F-B2); the driver's runtime all-zero
+            # guard remains authoritative for degenerate mask files.
+            if ((self.mpas_land_lapse_K_per_km > 0.0
+                 or self.mpas_land_beta != 1.0)
+                    and self.topography == "flat"
+                    and not self.land_mask_path):
+                errors.append(
+                    "mpas_land_lapse_K_per_km/mpas_land_beta need a land "
+                    "fraction, but topography='flat' (with no land-mask "
+                    "file) yields an all-zero f_land — the knobs would "
+                    "change nothing."
+                )
         else:
             if self.mpas_land_lapse_K_per_km != 0.0 or self.mpas_land_beta != 1.0:
                 errors.append(
diff --git a/packages/coupler/legoesm/driver/model_driver.py b/packages/coupler/legoesm/driver/model_driver.py
index 620cddd4e..aee160a46 100644
--- a/packages/coupler/legoesm/driver/model_driver.py
+++ b/packages/coupler/legoesm/driver/model_driver.py
@@ -5315,12 +5315,12 @@ class ModelDriver:
 
         Returns ``(mean_T, mean_ps, max_u, T_min, T_max, T_finite, cwv)`` as
         host floats / bool.  ``cwv`` is NaN when ``cwv_field`` is None.
+        ``mean_T`` is the MASS-WEIGHTED global mean (sum T*dp / sum dp).
         """
         from mpi4py import MPI as _MPI
         vl = self._voronoi_layout
         om_c = vl.owned_mask_cells          # (n_local_cells,) bool
         om_e = vl.owned_mask_edges          # (n_local_edges,) bool
-        nlev = T_data.shape[-1]
         T_owned = jnp.where(om_c[:, None], T_data, 0.0)
         ps_owned = jnp.where(om_c, p_s_data, 0.0)
         absu_owned = jnp.where(om_e[:, None], jnp.abs(u_data), 0.0)
@@ -5329,11 +5329,24 @@ class ModelDriver:
         finite_l = jnp.all(jnp.isfinite(T_owned))
         cwv_sum_l = (jnp.sum(jnp.where(om_c, cwv_field, 0.0))
                      if cwv_field is not None else jnp.asarray(0.0))
+        # Pressure-weighted mean T (sum T*dp / sum dp; equal cell weight —
+        # the quasi-uniform SCVT makes areaCell weighting a negligible
+        # refinement, and the pre-fix convention was equal-cell too), owned
+        # cells only: an unweighted level mean is coordinate-dependent
+        # (stretched hybrid grids overstate it by ~+9 K vs sigma on the same
+        # state; quantified 2026-07-23), which made hybrid-vs-sigma
+        # stability curves incomparable.  Mirrors the serial day-line
+        # diagnostic.  The where() wraps the PRODUCT so a non-finite halo
+        # p_s cannot leak NaN through 0*NaN (codex F-B6).
+        _p_half_d = self.sigma.pressure_at_half(p_s_data)
+        _dp_d = _p_half_d[..., 1:] - _p_half_d[..., :-1]
+        Tdp_sum_l = jnp.sum(jnp.where(om_c[:, None], T_data * _dp_d, 0.0))
+        dp_sum_l = jnp.sum(jnp.where(om_c[:, None], _dp_d, 0.0))
         # One device→host transfer for all local reductions.
         _loc = np.asarray(jnp.stack([
-            jnp.sum(T_owned), jnp.sum(ps_owned), jnp.max(absu_owned),
+            Tdp_sum_l, jnp.sum(ps_owned), jnp.max(absu_owned),
             T_min_l, T_max_l, finite_l.astype(T_data.dtype),
-            cwv_sum_l.astype(T_data.dtype),
+            cwv_sum_l.astype(T_data.dtype), dp_sum_l,
         ]))
         comm = _MPI.COMM_WORLD
         # THREE batched buffer allreduces instead of eight scalar pickle
@@ -5341,13 +5354,14 @@ class ModelDriver:
         # collective; at multi-node rank counts the per-diag latency is
         # 8x a single round for no reason).  The finite flag (as a float)
         # rides the MIN batch: all-ranks-finite  <=>  min(finite) == 1.
-        _sums = np.array([_loc[0], _loc[1], _loc[6]], dtype=np.float64)
+        _sums = np.array([_loc[0], _loc[1], _loc[6], _loc[7]],
+                         dtype=np.float64)
         _maxs = np.array([_loc[2], _loc[4]], dtype=np.float64)
         _mins = np.array([_loc[3], _loc[5]], dtype=np.float64)
         comm.Allreduce(_MPI.IN_PLACE, _sums, op=_MPI.SUM)
         comm.Allreduce(_MPI.IN_PLACE, _maxs, op=_MPI.MAX)
         comm.Allreduce(_MPI.IN_PLACE, _mins, op=_MPI.MIN)
-        g_sum_T, g_sum_ps, g_sum_cwv = (float(v) for v in _sums)
+        g_sum_Tdp, g_sum_ps, g_sum_cwv, g_sum_dp = (float(v) for v in _sums)
         g_max_u, g_T_max = (float(v) for v in _maxs)
         g_T_min, g_finite_min = (float(v) for v in _mins)
         g_finite = bool(g_finite_min > 0.5)
@@ -5356,7 +5370,7 @@ class ModelDriver:
             self._mpas_g_n_cells = comm.allreduce(
                 int(vl.partition.n_owned_cells), op=_MPI.SUM)
         g_n_cells = self._mpas_g_n_cells
-        mean_T = g_sum_T / (g_n_cells * nlev)
+        mean_T = g_sum_Tdp / max(g_sum_dp, 1e-30)
         mean_ps = g_sum_ps / g_n_cells
         cwv = (g_sum_cwv / g_n_cells) if cwv_field is not None else float("nan")
         return mean_T, mean_ps, g_max_u, g_T_min, g_T_max, g_finite, cwv
@@ -5835,12 +5849,14 @@ class ModelDriver:
         _f_land_cells = None
         if self._f_land is not None:
             _f_land_cells = jnp.asarray(self._f_land).reshape(-1)
-        if (_land_beta != 1.0 or _land_lapse_K_m > 0.0) \
-                and _f_land_cells is None:
+        if (_land_beta != 1.0 or _land_lapse_K_m > 0.0) and (
+                _f_land_cells is None
+                or not bool(jnp.any(_f_land_cells > 0.0))):
             raise ValueError(
                 "mpas_land_beta/mpas_land_lapse_K_per_km need a land "
-                "fraction, but none was loaded (no --topography / land "
-                "mask source) — the knobs would be silently inert."
+                "fraction, but none was loaded or it is all-zero (flat / "
+                "ocean-only topography) — the knobs would be silently "
+                "inert."
             )
         if _land_beta != 1.0 or _land_lapse_K_m > 0.0:
             logger.info(
@@ -5976,8 +5992,15 @@ class ModelDriver:
                 _ts = blend_surface_temperature(
                     jnp.asarray(_sst), jnp.asarray(_sic), _T_ice).reshape(-1)
                 if _lapse_z is not None:
+                    # Cast the storage-dtype (possibly f32) statics to the
+                    # anchor dtype so the correction is formed at anchor
+                    # precision (mirrors the turbulence-path cast).  Where
+                    # f_land and sic overlap (elevated icy coasts) the lapse
+                    # cools the blended anchor's land fraction too —
+                    # directionally harmless (see codex F5).
                     _ts = land_lapse_adjusted_surface_temperature(
-                        _ts, _f_land_cells, _lapse_z, _land_lapse_K_m)
+                        _ts, _f_land_cells.astype(_ts.dtype),
+                        _lapse_z.astype(_ts.dtype), _land_lapse_K_m)
                 return _ts
 
             # Shape guard once, up front: a non-per-cell get_sst_sic would
@@ -6543,8 +6566,19 @@ class ModelDriver:
                 else:
                     # Serial / single-rank: fuse the reductions into one
                     # device→host transfer (each ``float()`` is a GPU stall).
+                    # mean T is PRESSURE-WEIGHTED (sum T*dp / sum dp; equal
+                    # cell weight on the quasi-uniform SCVT): an unweighted
+                    # level mean is coordinate-dependent — the stretched
+                    # hybrid grid packs thin warm near-surface levels that
+                    # each get one equal vote, overstating the global mean
+                    # by +9.2 K vs sigma on the same state (quantified
+                    # 2026-07-23, hybrid-L20 day-8 ckpt), which made
+                    # hybrid-vs-sigma stability curves incomparable.
+                    _p_half_diag = self.sigma.pressure_at_half(p_s_data)
+                    _dp_diag = (_p_half_diag[..., 1:]
+                                - _p_half_diag[..., :-1])
                     _stats = jnp.stack([
-                        jnp.mean(T_data),
+                        jnp.sum(T_data * _dp_diag) / jnp.sum(_dp_diag),
                         jnp.mean(p_s_data),
                         jnp.max(jnp.abs(u_data)),
                         jnp.min(T_data),
diff --git a/tests/unit/test_mpas_global_diag.py b/tests/unit/test_mpas_global_diag.py
index 533ea5002..c9ad034e5 100644
--- a/tests/unit/test_mpas_global_diag.py
+++ b/tests/unit/test_mpas_global_diag.py
@@ -18,6 +18,7 @@ jnp = pytest.importorskip("jax.numpy")
 pytest.importorskip("mpi4py")
 
 from legoesm.driver.model_driver import ModelDriver
+from legoesm.grids.vertical import create_sigma_coordinate
 
 
 class _Partition:
@@ -32,17 +33,27 @@ class _VLayout:
         self.partition = _Partition(n_owned_cells)
 
 
-def _make_stub(n_cells=6, n_owned=4, n_edges=5, n_owned_edges=3):
+def _make_stub(n_cells=6, n_owned=4, n_edges=5, n_owned_edges=3, nlev=3):
     """A bare ModelDriver instance (no __init__) with just the attrs
-    ``_mpas_global_diag`` reads."""
+    ``_mpas_global_diag`` reads (incl. ``sigma`` for the pressure-weighted
+    mean T, 2026-07-23)."""
     drv = ModelDriver.__new__(ModelDriver)
     om_c = jnp.asarray(np.arange(n_cells) < n_owned)
     om_e = jnp.asarray(np.arange(n_edges) < n_owned_edges)
     drv._voronoi_layout = _VLayout(om_c, om_e, n_owned)
     drv._mpas_g_n_cells = None
+    drv.sigma = create_sigma_coordinate(nlev)
     return drv, np.asarray(om_c), np.asarray(om_e)
 
 
+def _expected_mean_T(drv, T, ps, om_c):
+    """Pressure-weighted owned-cell mean, mirroring the fixed convention."""
+    p_half = np.asarray(drv.sigma.pressure_at_half(jnp.asarray(ps)))
+    dp = p_half[..., 1:] - p_half[..., :-1]
+    own = om_c == 1
+    return (T[own] * dp[own]).sum() / dp[own].sum()
+
+
 def test_owned_masked_means_and_extrema_single_rank():
     drv, om_c, om_e = _make_stub()
     n_cells, nlev, n_edges = 6, 3, 5
@@ -63,7 +74,7 @@ def test_owned_masked_means_and_extrema_single_rank():
                               jnp.asarray(u), jnp.asarray(cwv)))
 
     T_own = T[om_c == 1]
-    assert mean_T == pytest.approx(T_own.sum() / (om_c.sum() * nlev))
+    assert mean_T == pytest.approx(_expected_mean_T(drv, T, ps, om_c))
     assert mean_ps == pytest.approx(ps[om_c == 1].mean())
     assert max_u == pytest.approx(np.abs(u[om_e == 1]).max())
     assert T_min == pytest.approx(T_own.min())
@@ -77,12 +88,22 @@ def test_finite_flag_owned_only():
     T = np.full((6, 2), 280.0)
     ps = np.full((6,), 1.0e5)
     u = np.zeros((5, 2))
-    # NaN in a HALO cell only: owned field is finite -> flag stays True.
+    # NaN in a HALO cell only: owned field is finite -> flag stays True,
+    # and the pressure-weighted mean must not be poisoned through the
+    # T*dp product (codex F-B6: the where() wraps the product).
     T_halo_nan = T.copy()
     T_halo_nan[int(np.argmax(om_c == 0)), 0] = np.nan
-    *_, finite, _ = drv._mpas_global_diag(
+    mean_T_h, *_, finite, _ = drv._mpas_global_diag(
         jnp.asarray(T_halo_nan), jnp.asarray(ps), jnp.asarray(u), None)
     assert finite is True
+    assert np.isfinite(mean_T_h) and mean_T_h == pytest.approx(280.0)
+    # Non-finite HALO p_s must not leak either (0*NaN trap).
+    ps_halo_nan = ps.copy()
+    ps_halo_nan[int(np.argmax(om_c == 0))] = np.nan
+    mean_T_p, *_, finite_p, _ = drv._mpas_global_diag(
+        jnp.asarray(T), jnp.asarray(ps_halo_nan), jnp.asarray(u), None)
+    assert finite_p is True
+    assert np.isfinite(mean_T_p) and mean_T_p == pytest.approx(280.0)
     # NaN in an OWNED cell -> False.
     T_owned_nan = T.copy()
     T_owned_nan[0, 0] = np.nan
```
