# DIFF REVIEW r1: Zhang-McFarlane net rain flux takes the surface route on the spectral bridge

Worktree: /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6_mem (branch wb/cam6-baseline, base a0038f279). The CLAIM behind this diff was reviewed first
(docs/wb/reviews/2026-09-24_zm_spectral_surface_route_claim.md; verdicts: codex HOLD on guard/test/measurement points,
GLM SHIP with asks). Findings addressed in this diff:
- codex 1 / GLM 5: q_v AND q_c must be present (new raise), skip covers both the q_r slot and the q_c fold.
- codex 2: tests assert zero q_r tendency with a rain tracer present, unchanged q_c, missing-q_c refusal,
  and the Tiedtke (>=0 per-layer) routes byte-identical with and without q_r.
- codex 3 / GLM 2: REAL ZM kernel run through the spectral bridge on convecting columns (20 sigma levels,
  deep to stable): pressure-weighted column closure sum dp*(dq_v+dq_c+dq_r)=0, surface rain >= 0 everywhere and
  > 0 somewhere, convective mask non-empty, booked q_v/q_c equal the kernel's outputs exactly.
- GLM 4: gated on the trait alone (no rain_to_surface knob exists on the spectral bridge; none added).
- GLM 7: total precipitation is NOT a WB training target on this lane (grep of the training script, the
  rollout and the deck loss block: no precip term); silent-zero surface precip pre-exists for every scheme there.
- Non-vacuity: with the integration.py change reverted (control worktree at a0038f279) the 4 ZM tests FAIL
  with the old refusal and the 2 Tiedtke tests pass: 4 failed, 2 passed. With the change: 6 passed.
Out of scope, stated: no conv_precip carry published on spectral (nothing reads it there); no surface-precip
channel on SpectralHydrostaticState; bechtold/tiedtke per-layer sign histogram (GLM 3) untouched behaviour.

Review the diff at docs/wb/reviews/2026-09-24_zm_spectral_r1.diff against the code. Attack: correctness of the skip (both
branches), the guard, JIT-safety (all gates static Python), differentiability (no new non-diff op), test
vacuity, sign/units, scope words. Verdict SHIP/HOLD first, then numbered findings with file:line.

NOTE: you have NO file access. Everything you need is inlined below. codex r1 found one P1 (an existing test still pinned the old refusal); it is updated in this diff.

## The diff
```diff
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py b/packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py
index fc99ea277..9e5586835 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py
@@ -1782,15 +1782,28 @@ def _make_spectral_pe_convection(
         # (``dq_r_conv_dt is not None`` => bechtold/tiedtke precip_efficiency>0);
         # the ``conv_fn is not None`` clause short-circuits before ``conv_out``
         # is read, so a physics-free spectral step is unaffected.
-        if (conv_fn is not None and _tr.rain_is_net_flux
-                and conv_out.dq_r_conv_dt is not None):
+        # A NET precipitation-flux divergence (CAM ZM ``ntprprd``, signed per
+        # layer: production minus evaporation of rain from above) is never a
+        # tracer source -- booked per layer it would sink condensate from
+        # layers holding none.  Its column integral IS the surface rain, so it
+        # takes the surface route here exactly as on the hydrostatic bridge
+        # (``_to_sfc``) and in the unified pipeline: only dq_v and dq_c are
+        # booked and the rain leaves the column (this lane has no surface
+        # precipitation channel; every precipitating species already leaves
+        # the prescribed surface this way).  Column water then closes by the
+        # scheme's own contract, sum dp*dq_r = -sum dp*(dq_v + dq_c), which
+        # requires BOTH q_v and q_c to be carried -- without q_c the detrained
+        # condensate would vanish unrecorded.
+        _rain_to_sfc = conv_fn is not None and _tr.rain_is_net_flux
+        if (_rain_to_sfc and conv_out.dq_r_conv_dt is not None
+                and (state.tracers is None or "q_v" not in state.tracers
+                     or "q_c" not in state.tracers)):
             raise ValueError(
-                f"convection scheme '{scheme_name}' emits a signed NET rain-flux "
-                "divergence (dq_r_conv_dt) that must be column-integrated to "
-                "surface precipitation; the spectral bridge has no surface-precip "
-                "sink and booking it per layer into a tracer would create "
-                "condensate sinks in layers holding none. Use the hydrostatic "
-                "bridge or the unified PhysicsPipeline."
+                f"convection scheme '{scheme_name}' routes its net rain flux "
+                "(dq_r_conv_dt) to the surface and books vapour and detrained "
+                "condensate into q_v and q_c; the spectral state must carry "
+                "both tracers, got "
+                f"{None if state.tracers is None else sorted(state.tracers)}."
             )
         if (conv_fn is not None and state.tracers is None
                 and conv_out.dq_r_conv_dt is not None):
@@ -1831,7 +1844,7 @@ def _make_spectral_pe_convection(
             # SIGN: ``dq_r_conv_dt >= 0`` is a condensate SOURCE, same sign as
             # ``dq_c_conv_dt``.  Schemes with no rain split emit ``None`` ->
             # no-op (byte-identical).
-            _dq_r_conv = conv_out.dq_r_conv_dt
+            _dq_r_conv = None if _rain_to_sfc else conv_out.dq_r_conv_dt
             if _dq_r_conv is not None:
                 _dq_r_grid = _dq_r_conv.reshape(n_lat, n_lon, nlev)
                 if "q_r" in _state_tracers:
diff --git a/tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py b/tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py
index 7e9699af8..9b5ead666 100644
--- a/tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py
+++ b/tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py
@@ -244,24 +244,28 @@ class TestSpectralPECMT:
         assert float(jnp.max(jnp.abs(tendencies.vor_hat.data))) == 0.0
         assert float(jnp.max(jnp.abs(tendencies.div_hat.data))) == 0.0
 
-    def test_zm_refused_without_surface_precip_sink(
+    def test_zm_runs_with_vapour_and_cloud_tracers_and_refuses_without_cloud(
         self, grid, sigma_coord, rest_state,
     ):
-        """CAM6 Zhang-McFarlane emits a SIGNED net rain-flux divergence that
-        only column-integrates to surface rain; this bridge has no surface
-        precip sink, so it must refuse loudly rather than book the field per
-        layer into a tracer (codex round 1, #3)."""
+        """CAM6 Zhang-McFarlane emits a SIGNED net rain-flux divergence whose
+        column integral is the surface rain.  The bridge books only dq_v and
+        dq_c and lets the rain leave the column (the hydrostatic bridge's
+        surface route), so it needs BOTH tracers; without q_c the detrained
+        condensate would vanish unrecorded, and that is refused loudly."""
         physics_fn = make_convection_physics(
             ConvectionConfig(scheme="zhang_mcfarlane"),
             model_type="spectral_pe", dt=300.0,
         )
         zeros = jnp.zeros((grid.n_lat, grid.n_lon, sigma_coord.n_levels))
-        state = _state_with_tracers(rest_state, {
-            "q_v": Field(data=zeros, name="q_v", dims=("lat", "lon", "level"), units="kg/kg"),
-            "q_c": Field(data=zeros, name="q_c", dims=("lat", "lon", "level"), units="kg/kg"),
-        })
-        with pytest.raises(ValueError, match="NET rain-flux"):
-            physics_fn(state, grid, sigma_coord)
+        q_v = Field(data=zeros, name="q_v", dims=("lat", "lon", "level"), units="kg/kg")
+        q_c = Field(data=zeros, name="q_c", dims=("lat", "lon", "level"), units="kg/kg")
+        tendencies, _ = physics_fn(
+            _state_with_tracers(rest_state, {"q_v": q_v, "q_c": q_c}), grid, sigma_coord)
+        assert set(tendencies.tracers) == {"q_v", "q_c"}
+        for k in ("q_v", "q_c"):
+            assert bool(jnp.all(jnp.isfinite(tendencies.tracers[k].data)))
+        with pytest.raises(ValueError, match="must carry both tracers"):
+            physics_fn(_state_with_tracers(rest_state, {"q_v": q_v}), grid, sigma_coord)
 
     def test_tiedtke_with_cape_and_winds_yields_nonzero_cmt(
         self, grid, sigma_coord, rest_state,
@@ -614,8 +618,10 @@ class TestSpectralPECMT:
 @pytest.fixture(
     scope="module",
     params=[
-        # zhang_mcfarlane is absent by design: CAM6 ZM's rain is a signed net
-        # flux divergence this bridge refuses (test_zm_refused_without_surface_precip_sink).
+        # zhang_mcfarlane is absent by design: this fixture's state is q_v-only
+        # and ZM needs a q_c tracer for its detrained condensate (its signed
+        # net rain flux takes the surface route); it is pinned through the
+        # bridge in tests/unit/test_spectral_zm_rain_surface_route.py.
         ("kain_fritsch", "kain_fritsch", KainFritschConfig),
         ("emanuel", "emanuel", EmanuelConfig),
         ("tiedtke", "tiedtke", TiedtkeConfig),
diff --git a/tests/unit/test_spectral_zm_rain_surface_route.py b/tests/unit/test_spectral_zm_rain_surface_route.py
new file mode 100644
index 000000000..2e996a659
--- /dev/null
+++ b/tests/unit/test_spectral_zm_rain_surface_route.py
@@ -0,0 +1,179 @@
+"""Zhang-McFarlane's net rain flux takes the surface route on the spectral bridge.
+
+ZM emits ``dq_r_conv_dt`` as CAM's ``ntprprd``: a SIGNED per-layer net
+precipitation-flux divergence whose column integral is the surface rain.  The
+hydrostatic bridge and the unified pipeline never book it into a tracer; the
+spectral bridge used to refuse the scheme outright.  Now it books only dq_v and
+dq_c and lets the rain leave the column, as every precipitating species on
+this prescribed-surface lane already does.
+
+Every assertion in the first three tests fails with the change reverted (the
+bridge raises "signed NET rain-flux divergence" before booking anything); the
+Tiedtke test pins the >=0 per-layer rain routes byte-identical; the last test
+runs the REAL ZM kernel through the bridge on convecting columns and measures
+the column-water closure the surface route relies on.
+"""
+from __future__ import annotations
+
+import jax
+import jax.numpy as jnp
+import numpy as np
+import pytest
+
+jax.config.update("jax_enable_x64", True)
+
+from legoesm import constants  # noqa: E402
+from legoesm.atmosphere.dynamics.gcm.spectral_pe import (  # noqa: E402
+    isothermal_rest_state_spectral,
+)
+from legoesm.atmosphere.physics.convection import integration as convint  # noqa: E402
+from legoesm.atmosphere.physics.convection.config import ConvectionConfig  # noqa: E402
+from legoesm.atmosphere.physics.convection.output import ConvectionOutput  # noqa: E402
+from legoesm.grids.gaussian import create_gaussian_grid, sh_analysis_3d  # noqa: E402
+from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402
+from legoesm.thermo import saturation_mixing_ratio  # noqa: E402
+
+_N_MAX = 8
+_N_LEV = 8
+_DT = 1800.0
+_TRACER_VALUES = {"q_v": 5.0e-3, "q_c": 1.0e-4, "q_i": 2.0e-5, "q_r": 3.0e-6}
+
+
+def _setup(nlev=_N_LEV, tracers=("q_v", "q_c", "q_i"), convecting=False):
+    grid = create_gaussian_grid(_N_MAX, dealiasing="quadratic")
+    sigma = create_sigma_coordinate(nlev)
+    shp = (grid.n_lat, grid.n_lon, nlev)
+    tr = {k: jnp.full(shp, _TRACER_VALUES[k]) for k in tracers}
+    state = isothermal_rest_state_spectral(
+        grid, sigma, T_init=280.0, p_s_init=1.0e5,
+        perturbation_amplitude=0.0, tracers=tr)
+    if convecting:
+        # Warm, humid columns at low latitude index grading to cool, dry
+        # ones: deep convection, marginal, and stable columns in one state.
+        p_s = jnp.full((grid.n_lat, grid.n_lon), 1.0e5)
+        p_full = sigma.pressure_at_full(p_s)
+        z = -8000.0 * jnp.log(p_full / p_s[..., None])
+        T_sfc = jnp.linspace(303.0, 286.0, grid.n_lat)[:, None, None]
+        rh = jnp.linspace(0.85, 0.35, grid.n_lat)[:, None, None]
+        T = jnp.maximum(T_sfc - 6.5e-3 * z, 200.0)
+        state = state._replace(
+            T_hat=state.T_hat.replace(data=sh_analysis_3d(grid, T)),
+            tracers={**state.tracers, "q_v": rh * saturation_mixing_ratio(T, p_full)})
+    ncol = int(grid.n_lat) * int(grid.n_lon)
+    return grid, sigma, state, ncol
+
+
+def _patterns(ncol, nlev):
+    idx = jnp.arange(ncol * nlev, dtype=jnp.float64).reshape(ncol, nlev)
+    dq_v = -(idx + 1.0) * 1.0e-8
+    dq_c = (idx + 1.0) * 3.0e-9
+    # Signed net rain flux: production aloft, evaporation of falling rain in
+    # the lower layers (surface is index -1), column sum positive.
+    k = jnp.arange(nlev, dtype=jnp.float64)
+    dq_r = ((nlev / 2.0 - k) * 1.0e-7)[None, :] * jnp.ones((ncol, 1)) + idx * 1.0e-10
+    return dq_v, dq_c, dq_r
+
+
+def _fake_zm(ncol, nlev, dq_v, dq_c, dq_r):
+    z = jnp.zeros((ncol, nlev))
+
+    def fake(*, T, q_v, p_full, p_half, u, v, conv_prog_profile, dt, config,
+             land_frac=None, cld_frac=None, pref_edge=None):
+        return ConvectionOutput(
+            dT_dt=z, dq_v_dt=dq_v, dq_c_conv_dt=dq_c, cape=jnp.zeros((ncol,)),
+            convective_mask=jnp.ones((ncol,)), dq_r_conv_dt=dq_r), z
+    return fake
+
+
+def _run(monkeypatch, name, fake, state, grid, sigma):
+    monkeypatch.setattr(convint, "_get_convection_fn",
+                        lambda cfg: (name, fake, getattr(cfg, name)))
+    fn = convint.make_convection_physics(ConvectionConfig(scheme=name),
+                                         "spectral_pe", _DT)
+    tend, _prog = fn(state, grid, sigma)
+    return {k: np.asarray(getattr(v, "data", v)) for k, v in tend.tracers.items()}
+
+
+def test_zm_rain_takes_the_surface_route_and_only_vapour_and_cloud_are_booked(monkeypatch):
+    grid, sigma, state, ncol = _setup()
+    dq_v, dq_c, dq_r = _patterns(ncol, _N_LEV)
+    assert float(dq_r.min()) < 0.0 < float(dq_r.sum(axis=1).min())
+    tt = _run(monkeypatch, "zhang_mcfarlane", _fake_zm(ncol, _N_LEV, dq_v, dq_c, dq_r),
+              state, grid, sigma)
+    shp = (grid.n_lat, grid.n_lon, _N_LEV)
+    np.testing.assert_array_equal(tt["q_v"], np.asarray(dq_v).reshape(shp))
+    np.testing.assert_array_equal(tt["q_c"], np.asarray(dq_c).reshape(shp))
+    assert "q_r" not in tt
+    np.testing.assert_array_equal(tt["q_i"], 0.0)
+
+
+def test_zm_leaves_an_existing_rain_tracer_untouched(monkeypatch):
+    grid, sigma, state, ncol = _setup(tracers=("q_v", "q_c", "q_r"))
+    dq_v, dq_c, dq_r = _patterns(ncol, _N_LEV)
+    tt = _run(monkeypatch, "zhang_mcfarlane", _fake_zm(ncol, _N_LEV, dq_v, dq_c, dq_r),
+              state, grid, sigma)
+    shp = (grid.n_lat, grid.n_lon, _N_LEV)
+    np.testing.assert_array_equal(tt["q_r"], 0.0)
+    np.testing.assert_array_equal(tt["q_c"], np.asarray(dq_c).reshape(shp))
+
+
+def test_zm_refuses_a_state_without_a_cloud_tracer(monkeypatch):
+    grid, sigma, state, ncol = _setup(tracers=("q_v",))
+    dq_v, dq_c, dq_r = _patterns(ncol, _N_LEV)
+    with pytest.raises(ValueError, match="must carry both tracers"):
+        _run(monkeypatch, "zhang_mcfarlane", _fake_zm(ncol, _N_LEV, dq_v, dq_c, dq_r),
+             state, grid, sigma)
+
+
+@pytest.mark.parametrize("with_qr", [False, True])
+def test_tiedtke_rain_split_route_is_unchanged(monkeypatch, with_qr):
+    tracers = ("q_v", "q_c", "q_r") if with_qr else ("q_v", "q_c")
+    grid, sigma, state, ncol = _setup(tracers=tracers)
+    dq_v, dq_c, _ = _patterns(ncol, _N_LEV)
+    dq_r = jnp.abs(_patterns(ncol, _N_LEV)[2])  # a >=0 per-layer rain SOURCE
+    z = jnp.zeros((ncol, _N_LEV))
+
+    def fake(*, T, q_v, p_full, p_half, u, v, conv_prog_profile, dt, config,
+             moisture_convergence=None):
+        return ConvectionOutput(
+            dT_dt=z, dq_v_dt=dq_v, dq_c_conv_dt=dq_c, cape=jnp.zeros((ncol,)),
+            convective_mask=jnp.ones((ncol,)), dq_r_conv_dt=dq_r), z
+    tt = _run(monkeypatch, "tiedtke", fake, state, grid, sigma)
+    shp = (grid.n_lat, grid.n_lon, _N_LEV)
+    if with_qr:
+        np.testing.assert_array_equal(tt["q_r"], np.asarray(dq_r).reshape(shp))
+        np.testing.assert_array_equal(tt["q_c"], np.asarray(dq_c).reshape(shp))
+    else:
+        np.testing.assert_array_equal(tt["q_c"], np.asarray(dq_c + dq_r).reshape(shp))
+
+
+def test_real_zm_kernel_closes_column_water_on_the_spectral_bridge(monkeypatch):
+    nlev = 20
+    grid, sigma, state, ncol = _setup(nlev=nlev, tracers=("q_v", "q_c"), convecting=True)
+    real = convint.zhang_mcfarlane_convection
+    seen = {}
+
+    def spy(*a, **k):
+        out, prog = real(*a, **k)
+        seen["out"] = out
+        return out, prog
+    tt = _run(monkeypatch, "zhang_mcfarlane", spy, state, grid, sigma)
+    out = seen["out"]
+    n_active = int(jnp.sum(out.convective_mask > 0))
+    assert n_active >= 1, "no column convected: the closure test would be vacuous"
+
+    dp = np.asarray(sigma.layer_thickness_dp(jnp.full((ncol,), 1.0e5)))
+    dq_v = np.asarray(out.dq_v_dt); dq_c = np.asarray(out.dq_c_conv_dt)
+    dq_r = np.asarray(out.dq_r_conv_dt)
+    # Contract: sum dp*dq_r = -sum dp*(dq_v + dq_c) per column (pressure-weighted).
+    resid = np.sum(dp * (dq_v + dq_c + dq_r), axis=1)
+    scale = np.sum(dp * np.abs(dq_v), axis=1)
+    assert np.all(np.abs(resid) <= 1.0e-9 * scale.max() + 1.0e-12 * dp.sum(axis=1))
+    # The surface rain the route implies is non-negative in every column
+    # (the pipeline's clip-at-zero would be inactive) and positive somewhere.
+    P = np.sum(dp * dq_r, axis=1) / constants.g
+    assert P.min() >= -1.0e-15 * abs(P).max() and P.max() > 0.0
+    # And the bridge booked exactly the kernel's vapour and cloud tendencies.
+    shp = (grid.n_lat, grid.n_lon, nlev)
+    np.testing.assert_array_equal(tt["q_v"], dq_v.reshape(shp))
+    np.testing.assert_array_equal(tt["q_c"], dq_c.reshape(shp))
```

## Reference: the hydrostatic bridge's surface route (integration.py ~L805-870, unchanged)
```python
            # all-condensate-to-cloud routing.  SIGN: ``dq_r_conv_dt >= 0`` is a
            # condensate SOURCE, the SAME sign as ``dq_c_conv_dt``.  Schemes with
            # no rain split emit ``None`` -> no-op (byte-identical).
            _dq_r_conv = conv_out.dq_r_conv_dt
            _has_qr = state.tracers is not None and "q_r" in state.tracers
            # Static Python bool (closure constant): rain_to_surface routes the
            # survivor rain out of the column as surface precip below instead
            # of into q_r / q_c.
            _to_sfc = bool(getattr(convection_config, "rain_to_surface", False))
            # A NET precipitation-flux divergence (CAM ZM) is signed per layer:
            # booking it into q_r/q_c would create condensate sinks in layers
            # holding none.  Its column integral IS the surface rain, so it
            # takes the surface route regardless of ``rain_to_surface``.
            _to_sfc = _to_sfc or _tr.rain_is_net_flux
            if _dq_r_conv is not None and not _has_qr and not _to_sfc:
                dq_c_conv_dt = dq_c_conv_dt + _dq_r_conv.reshape(shape_3d)
            tracer_tends = {
                "q_v": Field(
                    data=dq_v_dt, name="dq_v_dt_conv",
                    dims=dims_3d, units="kg/kg/s",
                ),
                "q_c": Field(
                    data=dq_c_conv_dt, name="dq_c_conv_dt",
                    dims=dims_3d, units="kg/kg/s",
                ),
            }
            if _dq_r_conv is not None and _has_qr and not _to_sfc:
                tracer_tends["q_r"] = Field(
                    data=_dq_r_conv.reshape(shape_3d), name="dq_r_conv_dt",
                    dims=dims_3d, units="kg/kg/s",
                )
```
## Reference: unified pipeline (physics_pipeline.py ~L1791-1797, unchanged)
```python
            if conv_out.dq_r_conv_dt is not None:
                dq_r_dt_conv = ad.unflatten_3d(conv_out.dq_r_conv_dt)
                _dp_r = self.sigma_coord.layer_thickness_dp(p_s)
                precip_conv_rain = jnp.maximum(
                    jnp.sum(dq_r_dt_conv * _dp_r / constants.g, axis=-1),
                    0.0)  # (..., n, n) kg/m2/s in-updraft rain to the surface
                precip = precip + precip_conv_rain
```
## Reference: ZM contract (zhang_mcfarlane.py L29-33, L94-96)
The rain field ``dq_r_conv_dt`` is CAM's ``ntprprd``: the NET divergence of
the falling-precipitation flux (``prdprec - evpprec``), negative where rain
from above evaporates.  It column-integrates to the surface rain and is not
a per-layer condensate source; hosts must sum it (the bridges do, or refuse).

        "dq_r_conv_dt integrates over the column to the surface convective "
        "precipitation (kg/m^2/s = -sum dp (dq_v_dt + dq_c_conv_dt)/g)."
    ),
