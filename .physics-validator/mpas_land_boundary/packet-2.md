You are an independent adversarial reviewer for LegoESM (JAX-native differentiable ESM).
This is an ADDENDUM review. In round 1 you found F1-F5 on the MPAS land surface boundary
change. This addendum (uncommitted working tree vs f0efce5a4) claims to fix F1-F4/F6/F7 AND
adds a NEW mass-weighted mean-T diagnostic. FINDINGS ONLY — do NOT modify any file. Read the
actual source with your tools; cite file:line; label CONFIRMED/PLAUSIBLE and severity.

## Part A — fixes for round-1 findings
- F1: `validate_strict` (packages/coupler/legoesm/driver/config.py ~1723) now refuses knobs with
  `topography=='flat'`; `_run_mpas` (model_driver.py ~5848) also rejects an all-zero f_land at
  runtime via `not bool(jnp.any(f_land>0))`.
- F2: `validate_strict` refuses `mpas_land_lapse_K_per_km>0` with `radiation=='none'`.
- F3: refuses `mpas_land_beta!=1` with `turbulence=='none'`. beta+radiation='none' stays ACCEPTED
  (the humidity throttle still acts on the fallback anchor `_resolve_T_sfc`).
- F4: lane guard now `_is_mpas = (d.discretization=='mpas' or g.grid_type in ('mpas','voronoi'))`.
- F6: `_compute_T_sfc` casts f_land/z to the anchor dtype before the lapse (model_driver.py ~5988).
- F7: beta docstring softened to "first-order analogue" (surface_layer.py ~175).

## Part B — NEW change: mass-weighted global mean T (diagnostic only)
The MPAS day-line/timeseries mean T was `jnp.mean(T)` — UNWEIGHTED over levels; a stretched hybrid
coordinate overstates it by ~+9 K vs sigma on the same state. Fixed to `sum(T*dp)/sum(dp)` with
`dp` from `self.sigma.pressure_at_half`, in BOTH:
- serial day-line (model_driver.py ~6572): `sum(T_data*_dp_diag)/sum(_dp_diag)`.
- MPI `_mpas_global_diag` (model_driver.py ~5307): `Tdp_sum_l`/`dp_sum_l` appended to the batched
  SUM allreduce; `_loc[7]=dp_sum`, `_sums=[_loc[0],_loc[1],_loc[6],_loc[7]]`, then
  `mean_T = g_sum_Tdp / max(g_sum_dp, 1e-30)`. The unused `nlev` local was dropped.
No feedback into stepping; blowup detection uses T_min/T_max; tests in
tests/unit/test_mpas_meanT_massweighted.py (4).

## Specific verification asks (rank CONFIRMED/PLAUSIBLE)
1. The MPI allreduce packing after appending `_loc[7]` (dp_sum): is the `_sums`/`_maxs`/`_mins`
   index map still correct? Does the unpack arity match? Is `nlev` truly unused now?
2. Mass-weighting math: is `dp = p_half[...,1:]-p_half[...,:-1]` positive (top-to-surface ordering)
   for BOTH the sigma and hybrid coordinates, so `max(g_sum_dp,1e-30)` is a safe floor and not a
   sign trap? Is serial consistent with MPI (owned-cell masking)? Is there a coordinate whose
   `pressure_at_half` is surface-first (dp<0) that would break the guard?
3. Do the new validate errors fire in the right lane branch only (inside `if _is_mpas`)? Any
   false-positive? (Hint: `_create_topography` overrides f_land from `land_mask_path` even for
   `topography=='flat'`; and `normalize_grid_type` maps voronoi/icosahedral/ico/mpas_voronoi ->
   mpas, but the F4 guard lists only 'mpas','voronoi'.)
4. Regression risk to any consumer of `mean_T` / `_ts['T_atm']` (blowup detection, CMOR feed,
   restart digest, reproduce-check) or to the `_mpas_global_diag` 7-tuple return contract.
5. Anything you would BLOCK the 60-day A/B launch on.

## The addendum unified diff (working tree vs f0efce5a4)
```diff
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py b/packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py
index e870b34c7..29942a6e7 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py
@@ -175,8 +175,12 @@ def beta_limited_surface_humidity(
         q_sfc = q_air + (1 - f_land * (1 - beta_land)) * (q_sat_sfc - q_air)
 
     so the land-fraction latent flux is ``beta_land`` times its wet-surface
-    potential — the same alpha-method form the tiled pipeline applies per
-    tile (``physics_pipeline._tiled_surface_flux``).  ``beta_land = 1``
+    potential — a first-order analogue of the tiled pipeline's per-tile
+    alpha method (``physics_pipeline._tiled_surface_flux``).  The
+    decomposition is exact only for a fixed shared transfer coefficient; a
+    stability-dependent MOST/COARE bulk scheme recomputes ``C_E`` from the
+    throttled ``q_sfc``, so this is approximate, not a true mosaic.
+    ``beta_land = 1``
     returns ``q_sat_sfc`` exactly (byte-identical wet surface);
     ``beta_land = 0`` zeroes the land-fraction humidity gradient in both
     directions (no land evaporation and no land dew — a closed surface).
diff --git a/packages/coupler/legoesm/driver/config.py b/packages/coupler/legoesm/driver/config.py
index f33842ec5..27b584ce3 100644
--- a/packages/coupler/legoesm/driver/config.py
+++ b/packages/coupler/legoesm/driver/config.py
@@ -1705,7 +1705,13 @@ class ExperimentConfig(NamedTuple):
         # there — accepting those flags on MPAS ran a 60-day A/B against a
         # byte-identical twin (2026-07-23).  Conversely the MPAS land boundary
         # knobs are consumed only by the MPAS lane.
-        _is_mpas = d.discretization == "mpas"
+        # Lane detection keys on BOTH fields: the _run_mpas dispatch actually
+        # keys on grid_type ("mpas"/"voronoi"), and discretization stays
+        # consistent only via run_amip's postprocessor; a mismatched pair is
+        # fail-closed at the component factory, but the guard here must not
+        # emit a wrong-lane message for grid_type-keyed configs (codex F4).
+        _is_mpas = (d.discretization == "mpas"
+                    or g.grid_type in ("mpas", "voronoi"))
         if _is_mpas:
             for _flag in ("slab_land_active", "land_soil_bucket",
                           "surface_tiled"):
@@ -1717,6 +1723,29 @@ class ExperimentConfig(NamedTuple):
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
+            if ((self.mpas_land_lapse_K_per_km > 0.0
+                 or self.mpas_land_beta != 1.0)
+                    and self.topography == "flat"):
+                errors.append(
+                    "mpas_land_lapse_K_per_km/mpas_land_beta need a land "
+                    "fraction, but topography='flat' yields an all-zero "
+                    "f_land — the knobs would change nothing."
+                )
         else:
             if self.mpas_land_lapse_K_per_km != 0.0 or self.mpas_land_beta != 1.0:
                 errors.append(
diff --git a/packages/coupler/legoesm/driver/model_driver.py b/packages/coupler/legoesm/driver/model_driver.py
index 620cddd4e..6ca8a0f76 100644
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
@@ -5329,11 +5329,20 @@ class ModelDriver:
         finite_l = jnp.all(jnp.isfinite(T_owned))
         cwv_sum_l = (jnp.sum(jnp.where(om_c, cwv_field, 0.0))
                      if cwv_field is not None else jnp.asarray(0.0))
+        # Mass-weighted mean T (sum T*dp / sum dp), owned cells only: an
+        # unweighted level mean is coordinate-dependent (stretched hybrid
+        # grids overstate it by ~+9 K vs sigma on the same state; quantified
+        # 2026-07-23), which made hybrid-vs-sigma stability curves
+        # incomparable.  Mirrors the serial day-line diagnostic.
+        _p_half_d = self.sigma.pressure_at_half(p_s_data)
+        _dp_d = _p_half_d[..., 1:] - _p_half_d[..., :-1]
+        Tdp_sum_l = jnp.sum(T_owned * _dp_d)
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
@@ -5341,13 +5350,14 @@ class ModelDriver:
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
@@ -5356,7 +5366,7 @@ class ModelDriver:
             self._mpas_g_n_cells = comm.allreduce(
                 int(vl.partition.n_owned_cells), op=_MPI.SUM)
         g_n_cells = self._mpas_g_n_cells
-        mean_T = g_sum_T / (g_n_cells * nlev)
+        mean_T = g_sum_Tdp / max(g_sum_dp, 1e-30)
         mean_ps = g_sum_ps / g_n_cells
         cwv = (g_sum_cwv / g_n_cells) if cwv_field is not None else float("nan")
         return mean_T, mean_ps, g_max_u, g_T_min, g_T_max, g_finite, cwv
@@ -5835,12 +5845,14 @@ class ModelDriver:
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
@@ -5976,8 +5988,15 @@ class ModelDriver:
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
@@ -6543,8 +6562,18 @@ class ModelDriver:
                 else:
                     # Serial / single-rank: fuse the reductions into one
                     # device→host transfer (each ``float()`` is a GPU stall).
+                    # mean T is MASS-WEIGHTED (sum T*dp / sum dp): an
+                    # unweighted level mean is coordinate-dependent — the
+                    # stretched hybrid grid packs thin warm near-surface
+                    # levels that each get one equal vote, overstating the
+                    # global mean by +9.2 K vs sigma on the same state
+                    # (quantified 2026-07-23, hybrid-L20 day-8 ckpt), which
+                    # made hybrid-vs-sigma stability curves incomparable.
+                    _p_half_diag = self.sigma.pressure_at_half(p_s_data)
+                    _dp_diag = (_p_half_diag[..., 1:]
+                                - _p_half_diag[..., :-1])
                     _stats = jnp.stack([
-                        jnp.mean(T_data),
+                        jnp.sum(T_data * _dp_diag) / jnp.sum(_dp_diag),
                         jnp.mean(p_s_data),
                         jnp.max(jnp.abs(u_data)),
                         jnp.min(T_data),
diff --git a/tests/unit/test_mpas_land_boundary.py b/tests/unit/test_mpas_land_boundary.py
index 3933ca896..0bdf0d8e7 100644
--- a/tests/unit/test_mpas_land_boundary.py
+++ b/tests/unit/test_mpas_land_boundary.py
@@ -51,14 +51,14 @@ def test_lapse_ocean_fraction_unchanged():
 
 
 def test_lapse_land_cooling_matches_rate():
-    # full land at 2 km with 6.5 K/km -> exactly -13 K
+    # full land at 1 km with 6.5 K/km -> exactly -6.5 K
     out = land_lapse_adjusted_surface_temperature(
-        jnp.array([300.0]), jnp.array([1.0]), jnp.array([2000.0]), 6.5e-3)
-    np.testing.assert_allclose(np.asarray(out), [300.0 - 13.0], rtol=1e-12)
+        jnp.array([300.0]), jnp.array([1.0]), jnp.array([1000.0]), 6.5e-3)
+    np.testing.assert_allclose(np.asarray(out), [300.0 - 6.5], rtol=1e-12)
     # fractional land scales linearly
     out_half = land_lapse_adjusted_surface_temperature(
-        jnp.array([300.0]), jnp.array([0.5]), jnp.array([2000.0]), 6.5e-3)
-    np.testing.assert_allclose(np.asarray(out_half), [300.0 - 6.5], rtol=1e-12)
+        jnp.array([300.0]), jnp.array([0.5]), jnp.array([1000.0]), 6.5e-3)
+    np.testing.assert_allclose(np.asarray(out_half), [300.0 - 3.25], rtol=1e-12)
 
 
 def test_lapse_below_sea_level_clipped_not_warmed():
@@ -223,12 +223,17 @@ def _mpas_cfg(**kw):
     from legoesm.driver.config import (
         DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
     )
+    # turbulence + non-flat topography on by default so the land-boundary
+    # knobs are non-inert (the inert corners are tested explicitly below).
+    kw.setdefault("turbulence", "louis")
+    kw.setdefault("topography", "gaussian")
+    kw.setdefault("radiation", "gray")
     return ExperimentConfig(
         grid=GridConfig(grid_type="mpas", resolution=1, nlev=8,
                         vertical_coord="sigma"),
         dycore=DycoreConfig(dt=600.0, discretization="mpas"),
         output=OutputConfig(diag_days=1),
-        days=1, dataset="analytical", radiation="gray",
+        days=1, dataset="analytical",
         **kw,
     )
 
@@ -270,6 +275,42 @@ def test_validate_cdgrid_refuses_mpas_knobs():
         _cdgrid_cfg(mpas_land_lapse_K_per_km=6.5).validate_strict()
 
 
+def test_validate_refuses_inert_corners():
+    """Codex F1-F3: a knob whose machinery is off must be refused, not
+    silently accepted."""
+    with pytest.raises(ValueError, match="silently inert"):
+        _mpas_cfg(mpas_land_lapse_K_per_km=6.5,
+                  radiation="none").validate_strict()
+    with pytest.raises(ValueError, match="silently inert"):
+        _mpas_cfg(mpas_land_beta=0.6, turbulence="none").validate_strict()
+    with pytest.raises(ValueError, match="all-zero"):
+        _mpas_cfg(mpas_land_beta=0.6, topography="flat").validate_strict()
+    # beta without radiation is fine (the turbulence anchor falls back but
+    # the humidity throttle still applies).
+    _mpas_cfg(mpas_land_beta=0.6, radiation="none").validate_strict()
+
+
+def test_validate_lane_guard_keys_on_grid_type_too():
+    """Codex F4: _run_mpas dispatch keys on grid_type; a grid_type='mpas'
+    config with the default discretization must still be treated as the
+    MPAS lane by the guard (pipeline land flags refused with the
+    MPAS-lane message, not the wrong-lane one)."""
+    from legoesm.driver.config import (
+        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
+    )
+    cfg = ExperimentConfig(
+        grid=GridConfig(grid_type="mpas", resolution=1, nlev=8,
+                        vertical_coord="sigma"),
+        dycore=DycoreConfig(dt=600.0),   # default discretization
+        output=OutputConfig(diag_days=1),
+        days=1, dataset="analytical", radiation="gray",
+        turbulence="louis", topography="gaussian",
+        slab_land_active=True,
+    )
+    with pytest.raises(ValueError, match="silently inert on the MPAS lane"):
+        cfg.validate_strict()
+
+
 def test_validate_bounds():
     with pytest.raises(ValueError, match="mpas_land_beta"):
         _mpas_cfg(mpas_land_beta=1.5).validate_strict()
diff --git a/tests/unit/test_run_amip_cli.py b/tests/unit/test_run_amip_cli.py
index 488b5f3e4..bfd46d757 100644
--- a/tests/unit/test_run_amip_cli.py
+++ b/tests/unit/test_run_amip_cli.py
@@ -459,7 +459,9 @@ def test_mpas_land_boundary_flags_flow_to_config():
     ]), parser))
     assert cfg.mpas_land_lapse_K_per_km == 6.5
     assert cfg.mpas_land_beta == 0.6
-    cfg.validate_strict()
+    # validate_strict is exercised in test_mpas_land_boundary (the bare CLI
+    # invocation here has topography='flat', which the inert-corner guard
+    # correctly refuses).
 def test_land_surface_scheme_validate_strict_rejects_unknown():
     """validate_strict() rejects an unknown surface scheme (dispatch hardening —
     a typo must fail early, not silently fall through in model_driver)."""
```
