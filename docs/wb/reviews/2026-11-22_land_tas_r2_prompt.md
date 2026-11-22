ADVERSARIAL REVIEW ROUND 2 (read-only; do not edit files). Worktree: /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193, branch fix/land-tas-t-ref2m (base 8750d0005d; HEAD is a WIP commit of round 1, the working tree holds round-2 fixes; the diff below is the full change vs base). CLM5: /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys.

Round 1 (codex NOT CLEAN; GLM CLEAN) findings and what was done:
1. Screen profile used the post-update zeta, not the Obukhov length the heat flux used -> monin_obukhov_stability gained `return_flux_obu=False` (default unchanged, 5-tuple); _stability_step output gained the incoming obu (last[6]); canopy_forward requests it and returns `obu`; screen_level_temperature now takes `obu` (not zeta), so temp1 is exactly the flux ch (new test: ch(obu)*ustar == 1/rah to 1e-12, with a check that zl/obu differs from the returned zeta in the unstable case).
2. Zero-filled cells could publish 0 K -> packed scatter of T2m now NaN-fills off the land columns; _land_step_none returns NaN for the T2m slot (other slots still zeros). Consumer (_tas_2m) keeps the existing land profile where T2m is non-finite. T2m is NOT halo-exchanged and NOT checkpointed (vote V3: author A, GLM A, codex B -> A).
3. Tests: oracle extended to CLM5 very-unstable heat regime (zeta -2, -20); canopy bracket now strict; gather fixture gives T2m a distinct value (f + 1.0).
Votes recorded: V1 A/A/A (publish CLM t_ref2m verbatim, no extra 2 m lapse step); V2 A/A/A (keep the existing land profile where no land T2m).

Check that the fixes are correct and complete, that nothing else changed behaviour (flux solve outputs identical: rah/raw/ustar/zeta unchanged), and the tuple/NaN handling in every producer/consumer. Report <=300 words: findings ranked (file:line), verdict CLEAN or NOT CLEAN.

DIFF vs base:
diff --git a/packages/coupler/legoesm/driver/diagnostics.py b/packages/coupler/legoesm/driver/diagnostics.py
index 8003d760c3..b06f536cc3 100644
--- a/packages/coupler/legoesm/driver/diagnostics.py
+++ b/packages/coupler/legoesm/driver/diagnostics.py
@@ -806,7 +806,8 @@ class DiagnosticCollector:
         return f_lo + alpha * (f_hi - f_lo)
 
     def _tas_2m(self, state, q_v, sst, sic, T_ice, u_low=None, v_low=None,
-                *, T_land=None, q_land=None, land_fraction=None):
+                *, T_land=None, q_land=None, land_fraction=None,
+                T2m_land=None):
         """2 m air temperature for CMIP ``tas`` from the MOST surface-layer
         similarity profile (interpolate the lowest model level down to 2 m).
         Returns the lowest-level T when the surface inputs (SST / sigma) are
@@ -821,6 +822,14 @@ class DiagnosticCollector:
         land branch uses the same MOST profile anchored on the land skin; do
         not read the result as a sub-canopy screen temperature.
 
+        ``T2m_land`` (keyword-only, optional) is the land scheme's OWN 2 m air
+        temperature (two-leaf canopy: CLM5 ``t_ref2m`` from its roughness,
+        displacement, Obukhov length and canopy-air node).  Where it is finite
+        it REPLACES the profile above on the land leg, which runs the ocean
+        bulk law (Charnock roughness, no displacement) over the land skin.
+        Non-finite entries (held columns, schemes that solve none) keep that
+        profile.
+
         ``u_low`` / ``v_low`` override the lowest-level winds (lets the MPAS
         path pass the reconstructed CELL winds from
         ``reconstruct_cell_velocity`` instead of the edge-normal ``state.u``,
@@ -901,6 +910,9 @@ class DiagnosticCollector:
         q_sfc_land = (saturation_mixing_ratio(T_land, p_s) if q_land is None
                       else q_land)
         T_2m_land = _profile_to_2m(T_land, q_sfc_land, ocean=False)
+        if T2m_land is not None:
+            T2m_land = jnp.asarray(T2m_land, dtype=T_2m_land.dtype)
+            T_2m_land = jnp.where(jnp.isfinite(T2m_land), T2m_land, T_2m_land)
         T_2m_sea = _profile_to_2m(T_sfc_sea, q_sfc_sea, ocean=True)
         f_land = jnp.clip(jnp.asarray(land_fraction, dtype=T_2m_land.dtype),
                           0.0, 1.0)
diff --git a/packages/coupler/legoesm/driver/model_driver.py b/packages/coupler/legoesm/driver/model_driver.py
index 656bda453e..b7a265ac53 100644
--- a/packages/coupler/legoesm/driver/model_driver.py
+++ b/packages/coupler/legoesm/driver/model_driver.py
@@ -297,7 +297,7 @@ def land_radiating_surface(T_sea, f_land, E_sea, T_rad_land, eps_land):
 
 # Per-cell fields of the MPAS land step tuple, after the state: T_sfc, albedo,
 # q_surface, shflx, lhflx, surface_mass_flux, stress, held, T_rad, emissivity.
-_LAND_STEP_N_CELL = 10
+_LAND_STEP_N_CELL = 11
 
 
 def gather_land_step(run, ncol, padded, proc, allgather):
@@ -10930,6 +10930,7 @@ class ModelDriver:
                 # published land tas is not the model's land at all.
                 _tas_land_T = getattr(self, "_land_T_skin_last", None)
                 _tas_land_q = getattr(self, "_land_qsfc_last", None)
+                _tas_land_T2m = getattr(self, "_land_T2m_last", None)
                 _tas_f_land = getattr(self, "_f_land", None)
                 _tas_kw = {}
                 if _tas_land_T is not None and _tas_f_land is not None:
@@ -10937,7 +10938,9 @@ class ModelDriver:
                         T_land=jnp.asarray(_tas_land_T).reshape(-1),
                         q_land=(None if _tas_land_q is None
                                 else jnp.asarray(_tas_land_q).reshape(-1)),
-                        land_fraction=jnp.asarray(_tas_f_land).reshape(-1))
+                        land_fraction=jnp.asarray(_tas_f_land).reshape(-1),
+                        T2m_land=(None if _tas_land_T2m is None
+                                  else jnp.asarray(_tas_land_T2m).reshape(-1)))
                 from legoesm.forcing.surface_utils import blend_surface_temperature
                 ts = blend_surface_temperature(_sst, _sic, _tas_ice)
                 if _tas_land_T is not None and _tas_f_land is not None:
@@ -14719,6 +14722,13 @@ class ModelDriver:
                 _emis = jnp.broadcast_to(
                     jnp.asarray(resp.emissivity, resp.T_rad.dtype),
                     resp.T_rad.shape)
+                # The scheme's own 2 m air temperature (CLM5 t_ref2m) for the
+                # CMOR ``tas`` land leg; NaN where the scheme solves none or the
+                # column was held (no solved surface layer this call).
+                _t2m = (jnp.full_like(resp.T_rad, jnp.nan)
+                        if getattr(_sfc, "T_2m", None) is None
+                        else jnp.asarray(_sfc.T_2m, resp.T_rad.dtype))
+                _t2m = jnp.where(_held_f > 0.5, jnp.nan, _t2m)
                 if _land_pack_on:
                     # Scatter the advanced columns back into the full-grid
                     # state (ocean columns keep their frozen init values,
@@ -14735,11 +14745,15 @@ class ModelDriver:
                                     resp.T_sfc, resp.albedo, resp.q_surface,
                                     resp.shflx, resp.lhflx, resp.surface_mass_flux,
                                     _taumag, _held_f, resp.T_rad, _emis))
+                            # 2 m T: NaN (= no land screen value) off the
+                            # land columns, never a valid-looking 0 K
+                            + (jnp.full((_land_ncol_full,), jnp.nan, _t2m.dtype)
+                               .at[_land_out_idx].set(_t2m.reshape(-1)),)
                             + (_n_held, _n_held_land, _n_fb_land,
                                _n_fb_rej, _n_unconv_land, _n_iter_max))
                 return (new_state, resp.T_sfc, resp.albedo, resp.q_surface,
                         resp.shflx, resp.lhflx, resp.surface_mass_flux,
-                        _taumag, _held_f, resp.T_rad, _emis,
+                        _taumag, _held_f, resp.T_rad, _emis, _t2m,
                         _n_held, _n_held_land, _n_fb_land, _n_fb_rej,
                         _n_unconv_land, _n_iter_max)
               if not _land_none:
@@ -14752,8 +14766,12 @@ class ModelDriver:
                   # the same message on every rank.
                   out = jax.eval_shape(
                       _land_step, land_state, a2s, doy, lai_doy, lai_year)
+                  # (the 2 m T, last per-cell field, is NaN = none solved)
                   return (land_state,) + tuple(
-                      jnp.zeros(o.shape, o.dtype) for o in out[1:])
+                      jnp.full(o.shape, jnp.nan, o.dtype)
+                      if i == _LAND_STEP_N_CELL - 1
+                      else jnp.zeros(o.shape, o.dtype)
+                      for i, o in enumerate(out[1:]))
               return _land_step_none
 
             _land_step_fn = _make_land_step(DT_LAND)
@@ -16307,7 +16325,7 @@ class ModelDriver:
                      _land_albedo_cells, _land_qsfc_step,
                      _land_shflx_step, _land_lhflx_step, _land_evap_step,
                      _land_taumag_step, _land_held_step,
-                     _land_Trad_cells, _land_emis_cells,
+                     _land_Trad_cells, _land_emis_cells, _land_T2m_step,
                      _land_n_held_step, _land_n_held_land_step,
                      _land_n_fb_step, _land_n_fb_rej_step,
                      _land_n_unconv_step, _land_n_iter_max) = _land_fn(
@@ -16332,6 +16350,7 @@ class ModelDriver:
                     # land and publishes a land temperature the model never had.
                     self._land_T_skin_last = _land_T_skin
                     self._land_qsfc_last = _land_qsfc_step
+                    self._land_T2m_last = _land_T2m_step
                     # Accumulate ON DEVICE and read at the same cadence the
                     # other post-step warnings use: reading it every step
                     # would stall the accelerator once per step for a number
diff --git a/packages/land/legoesm/land/canopy/solver.py b/packages/land/legoesm/land/canopy/solver.py
index 6e3d520689..119dae6c87 100644
--- a/packages/land/legoesm/land/canopy/solver.py
+++ b/packages/land/legoesm/land/canopy/solver.py
@@ -435,9 +435,10 @@ def canopy_forward(
     Ts = b.Ts_bc
     zldis = b.z0 - b.displa
 
-    ustar, rah_above, raw_above, uav, zeta = monin_obukhov_stability(
+    ustar, rah_above, raw_above, uav, zeta, obu = monin_obukhov_stability(
         b.ur, b.Ta, b.Tv_atm, Tc, b.q_atm, q_c, zldis, b.z0m,
-        n_iters=most_n_iters, zeta_cap_width=zeta_cap_width)
+        n_iters=most_n_iters, zeta_cap_width=zeta_cap_width,
+        return_flux_obu=True)
 
     Rb_Sun, Rb_Sh = compute_boundary_layer_resistance(uav, b.LAI, b.fSun, b.cv, b.d_leaf)
     rah_below, raw_below = compute_below_canopy_resistance(uav, b.CI, b.LAI)
@@ -549,7 +550,7 @@ def canopy_forward(
         G=G, Ls=Ls, Lcanopy_up=Lcanopy_up, gap_LW=gap_LW, LW_out=LW_out,
         eps_col=eps_col, LW_emit=LW_emit,
         gs_Sun=gs_Sun, gs_Sh=gs_Sh,
-        ustar=ustar, zeta=zeta,
+        ustar=ustar, zeta=zeta, obu=obu,
         rah_above=rah_above, rah_below=rah_below,
         Rb_Sun=Rb_Sun, Rb_Sh=Rb_Sh,
         ALW_Sun=ALW_Sun, ALW_Sh=ALW_Sh, ALW_Soil=ALW_Soil,
diff --git a/packages/land/legoesm/land/canopy/stability.py b/packages/land/legoesm/land/canopy/stability.py
index b5efb6366a..2db8bc0fa8 100644
--- a/packages/land/legoesm/land/canopy/stability.py
+++ b/packages/land/legoesm/land/canopy/stability.py
@@ -92,6 +92,7 @@ _ZETA_MAX_STABLE = 0.5
 # stability forms are continuous through zeta = 0 (user 2026-09-25).
 _ZETA_NEUTRAL_FLOOR = 1.0e-6
 _CONV_BDY_HEIGHT = 1000.0  # convective boundary layer height [m]
+_SCREEN_HEIGHT_M = 2.0     # CMIP tas / CLM5 t_ref2m height above d + z0h [m]
 _Z0MG_BARE = 0.01          # bare-soil momentum roughness length [m]
 
 # --- CLM5 Monin-Obukhov similarity-theory (MOST) constants ---
@@ -286,6 +287,29 @@ def _temperature_humidity_relation(zldis: jax.Array, obu: jax.Array,
     return ch
 
 
+def screen_level_temperature(theta_ref: jax.Array, T_canopy_air: jax.Array,
+                             zldis: jax.Array, obu: jax.Array,
+                             z0h: jax.Array,
+                             z_screen: float = _SCREEN_HEIGHT_M) -> jax.Array:
+    """CLM5 ``t_ref2m``: air temperature ``z_screen`` above the displacement
+    height plus ``z0h``, interpolated between the reference-height potential
+    temperature ``theta_ref`` and the canopy-air (aerodynamic surface)
+    temperature with this module's heat similarity relation at the Obukhov
+    length ``obu`` the solve's heat flux was evaluated with
+    (CanopyFluxesMod.F90:1182, FrictionVelocityMod.F90:636-652)::
+
+        t_ref2m = thm + temp1 * dth * (1/temp12m - 1/temp1),  dth = thm - taf
+
+    ``temp1`` and ``temp12m`` are evaluated at the SAME ``obu`` (as CLM does
+    within one FrictionVelocity call), so ``temp1`` is the flux's own ``ch``.  Stable side: Beljaars & Holtslag (1991),
+    as in the solve (module docstring).  Output is in the same frame as
+    ``theta_ref`` (surface-referenced potential temperature, CLM ``thm``)."""
+    temp1 = _temperature_humidity_relation(zldis, obu, z0h)
+    temp12m = _temperature_humidity_relation(z_screen + z0h, obu, z0h)
+    dth = theta_ref - T_canopy_air
+    return theta_ref + temp1 * dth * (1.0 / temp12m - 1.0 / temp1)
+
+
 def _cap_stable_zeta(zeta: jax.Array, width: float) -> jax.Array:
     """Smooth cap at ``_ZETA_MAX_STABLE`` (no kink), then the neutral floor.
 
@@ -327,7 +351,8 @@ def _stability_step(carry: jax.Array, _xs: None,
 
     carry : [z0h, obu, um]
     forcing: [Ta, Tv_atm, q_atm, q_atm, zldis, z0m, dq, dth, ur]
-    output : [ustar, tstar, qstar, thvstar, ch, zeta]
+    output : [ustar, tstar, qstar, thvstar, ch, zeta, obu]  (``obu`` = the
+             Obukhov length ``ch`` was evaluated with, i.e. the incoming one)
     """
     z0h, obu, um = carry[0], carry[1], carry[2]
     Ta, Tv_atm, q_atm, _q2, zldis, z0m, dq, dth, ur = (
@@ -360,7 +385,7 @@ def _stability_step(carry: jax.Array, _xs: None,
     obu_new  = zldis / zeta_new
 
     new_carry = jnp.array([z0h, obu_new, um_new])
-    output    = jnp.array([ustar, tstar, qstar, thvstar, ch, zeta_new])
+    output    = jnp.array([ustar, tstar, qstar, thvstar, ch, zeta_new, obu])
     return new_carry, output
 
 
@@ -376,7 +401,8 @@ def monin_obukhov_stability(
     n_iters: int = 5,
     *,
     zeta_cap_width: float,
-) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
+    return_flux_obu: bool = False,
+) -> tuple[jax.Array, ...]:
     """Above-canopy MOST iteration via jax.lax.scan.
 
     Parameters
@@ -400,6 +426,9 @@ def monin_obukhov_stability(
     raw      : aerodynamic resistance to water vapour [s/m]
     uav      : mean wind speed within canopy (used for Rb) [m/s]
     zeta     : stability parameter [-]
+    obu      : (only with ``return_flux_obu``) the Obukhov length [m] the
+               returned ``rah``/``raw`` were evaluated with (the last
+               iteration's INCOMING one; ``zeta`` is the updated stability)
     """
     # Heat roughness length.  CLM5 vegetation uses kB^-1 = 0 (z0h = z0m); the
     # earlier BESS default kB^-1 = 2 (z0h = z0m / e^2) over-suppressed heat
@@ -443,6 +472,8 @@ def monin_obukhov_stability(
     ram      = jnp.maximum(um_final / ustar**2, 1e-9)
     uav      = jnp.sqrt(um_final / ram)
 
+    if return_flux_obu:
+        return ustar, rah, raw, uav, zeta, last[6]
     return ustar, rah, raw, uav, zeta
 
 
diff --git a/packages/land/legoesm/land/surface_scheme/base.py b/packages/land/legoesm/land/surface_scheme/base.py
index 3a98280647..b7f7dcdc78 100644
--- a/packages/land/legoesm/land/surface_scheme/base.py
+++ b/packages/land/legoesm/land/surface_scheme/base.py
@@ -88,6 +88,10 @@ class SurfaceFluxOutput(NamedTuple):
     # ``T_sfc`` for the atmosphere's sensible-heat coupling.  Distinct from the
     # LW-derived radiometric ``T_surface``.  None for SimpleSEB.
     T_canopy_air: jax.Array | None = None
+    # CLM5 ``t_ref2m``: 2 m air temperature from the scheme's OWN surface layer
+    # (roughness, displacement, Obukhov length, canopy-air node) [K].  The CMIP
+    # ``tas`` land leg.  None for schemes that do not solve one.
+    T_2m: jax.Array | None = None
     gs_Sun: jax.Array | None = None        # sunlit stomatal conductance [m/s]
     gs_Sh: jax.Array | None = None         # shaded stomatal conductance [m/s]
     n_iters: jax.Array | None = None       # canopy Newton iteration count
diff --git a/packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py b/packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py
index 3aa3908df3..0986be2b26 100644
--- a/packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py
+++ b/packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py
@@ -40,7 +40,7 @@ from legoesm.land.canopy.radiative_transfer import (
 )
 from legoesm.land.canopy.sif import two_leaf_canopy_sif
 from legoesm.land.canopy.stability import (
-    compute_aerodynamics, sat_specific_humidity,
+    compute_aerodynamics, sat_specific_humidity, screen_level_temperature,
 )
 from legoesm.land.canopy.solver import (
     CanopyForcingBundle, solve_canopy_closure_diag, canopy_forward,
@@ -730,6 +730,8 @@ def compute_two_leaf_canopy_fluxes(
         Tf_Sun=Tf_Sun,
         Tf_Sh=Tf_Sh,
         T_canopy_air=Tc_cvg,
+        T_2m=screen_level_temperature(Ta, Tc_cvg, z_ref - displa,
+                                      fluxes_per_col["obu"], z0m),
         gs_Sun=gs_Sun,
         gs_Sh=gs_Sh,
         n_iters=n_iters,
diff --git a/tests/land/unit/test_land_screen_temperature.py b/tests/land/unit/test_land_screen_temperature.py
new file mode 100644
index 0000000000..1ff6a35ac4
--- /dev/null
+++ b/tests/land/unit/test_land_screen_temperature.py
@@ -0,0 +1,157 @@
+"""Land 2 m air temperature (CLM5 ``t_ref2m``) from the two-leaf canopy's own
+surface layer, and its use as the CMIP ``tas`` land leg.
+
+The profile oracle below is an INDEPENDENT scalar transcription of the
+similarity integral (Paulson 1970 unstable psi_h, Beljaars & Holtslag 1991
+stable psi_h), not a call into the code under test."""
+import math
+
+import jax
+import jax.numpy as jnp
+import numpy as np
+import pytest
+
+from legoesm import constants
+from legoesm.land.canopy.stability import screen_level_temperature
+
+pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64,
+                                reason="needs JAX_ENABLE_X64=1")
+
+K = constants.kappa_von_karman
+
+
+def _psi_h(zeta):
+    if zeta < 0.0:
+        x = (1.0 - 16.0 * zeta) ** 0.25
+        return 2.0 * math.log((1.0 + x * x) / 2.0)
+    a, b, c, d = 1.0, 0.667, 5.0, 0.35   # BH91 as published (b rounded)
+    return (-(1.0 + 2.0 * a * zeta / 3.0) ** 1.5
+            - b * (zeta - c / d) * math.exp(-d * zeta) - b * c / d + 1.0)
+
+
+ZETAT = 0.465   # CLM5 heat free-convection transition (FrictionVelocityMod)
+
+
+def _F(z, L, z0h):
+    """Integrated heat similarity function, CLM5 FrictionVelocityMod regimes."""
+    if z / L < -ZETAT:
+        return (math.log(-ZETAT * L / z0h) - _psi_h(-ZETAT) + _psi_h(z0h / L)
+                + 0.8 * (ZETAT ** (-1.0 / 3.0) - (-z / L) ** (-1.0 / 3.0)))
+    return math.log(z / z0h) - _psi_h(z / L) + _psi_h(z0h / L)
+
+
+def _oracle(theta, Tc, zl, zeta, z0h, zs=2.0):
+    L = zl / zeta
+    return Tc + (theta - Tc) * _F(zs + z0h, L, z0h) / _F(zl, L, z0h)
+
+
+def _screen(theta, Tc, zl, zeta, z0h):
+    return float(screen_level_temperature(
+        jnp.asarray(theta), jnp.asarray(Tc), jnp.asarray(zl),
+        jnp.asarray(zl / zeta), jnp.asarray(z0h)))
+
+
+@pytest.mark.parametrize("zeta", [-20.0, -2.0, -0.3, -0.05, 0.05, 0.3])
+@pytest.mark.parametrize("z0h", [0.05, 1.5])
+def test_profile_matches_independent_most(zeta, z0h):
+    theta, Tc, zl = 270.0, 262.0, 55.0
+    got = _screen(theta, Tc, zl, zeta, z0h)
+    np.testing.assert_allclose(got, _oracle(theta, Tc, zl, zeta, z0h),
+                               rtol=0.0, atol=1e-9)
+    assert min(theta, Tc) <= got <= max(theta, Tc)
+
+
+def test_neutral_limit_is_log_law():
+    theta, Tc, zl, z0h = 285.0, 289.0, 60.0, 0.8
+    got = _screen(theta, Tc, zl, 1e-6, z0h)
+    log_law = Tc + (theta - Tc) * math.log((2.0 + z0h) / z0h) / math.log(zl / z0h)
+    np.testing.assert_allclose(got, log_law, rtol=0.0, atol=1e-5)
+
+
+def test_reference_at_screen_height_returns_reference():
+    z0h = 0.3
+    got = _screen(271.0, 260.0, 2.0 + z0h, 0.2, z0h)
+    np.testing.assert_allclose(got, 271.0, rtol=0.0, atol=1e-12)
+
+
+def test_gradient_finite_and_nonzero():
+    f = lambda th, tc, zeta: screen_level_temperature(
+        th, tc, jnp.asarray(50.0), 50.0 / zeta, jnp.asarray(0.5))
+    for zeta in (-0.3, 0.3):
+        g = jax.grad(f, argnums=(0, 1, 2))(jnp.asarray(270.0), jnp.asarray(262.0),
+                                          jnp.asarray(zeta))
+        assert all(np.isfinite(float(x)) and abs(float(x)) > 1e-8 for x in g), g
+
+
+@pytest.mark.parametrize("dT", [-6.0, 4.0])
+def test_screen_profile_uses_the_flux_obukhov_length(dT):
+    """temp1 of the screen profile IS the flux solve's heat coefficient:
+    1 / (rah * ustar) at the Obukhov length the solve returns for its flux
+    (not the post-update stability; few iterations so the two differ)."""
+    from legoesm.land.canopy.stability import (
+        _temperature_humidity_relation, monin_obukhov_stability)
+    zl, z0m, Ta = 45.0, 0.9, 275.0
+    ustar, rah, _raw, _uav, zeta, obu = monin_obukhov_stability(
+        jnp.asarray(3.0), jnp.asarray(Ta), jnp.asarray(Ta), jnp.asarray(Ta + dT),
+        jnp.asarray(3e-3), jnp.asarray(3e-3), jnp.asarray(zl), jnp.asarray(z0m),
+        n_iters=2, zeta_cap_width=0.05, return_flux_obu=True)
+    ch = float(_temperature_humidity_relation(jnp.asarray(zl), obu, jnp.asarray(z0m)))
+    np.testing.assert_allclose(ch * float(ustar), 1.0 / float(rah), rtol=1e-12)
+    if dT > 0:   # unstable: the flux and the post-update stability differ
+        assert abs(float(zl / obu) - float(zeta)) > 1e-6
+
+
+def _two_leaf(sw, lw, T_soil):
+    from legoesm.core.coupling_fields import AtmToSurface
+    from legoesm.land.config import MultiLayerLandConfig
+    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
+    from legoesm.land.surface_scheme.two_leaf_canopy import (
+        compute_two_leaf_canopy_fluxes)
+    f = lambda v: jnp.full(1, v)
+    forcing = AtmToSurface(
+        sw_down=f(sw), lw_down=f(lw), precip_total=f(0.), precip_snow=f(0.),
+        T_lowest=f(270.), q_lowest=f(.002), u_lowest=f(4.), v_lowest=f(0.),
+        p_lowest=f(99300.), p_surface=f(100000.), rho_lowest=f(1.28),
+        cos_zenith=f(.7 if sw > 0 else 0.), co2_ppmv=f(412.),
+        has_radiation=f(1.), has_precipitation=f(1.), z_lowest=f(60.))
+    return compute_two_leaf_canopy_fluxes(
+        T_soil_top=f(T_soil), forcing=forcing,
+        canopy_config=TwoLeafCanopyConfig(max_iters=30),
+        land_config=MultiLayerLandConfig(), canopy_params=None,
+        w_frac_rz=f(.5), wind_speed=f(4.), wind_dir_x=jnp.ones(1),
+        wind_dir_y=jnp.zeros(1),
+        soil_thermal_fn=lambda G, dt: jnp.full(1, T_soil),
+        TgC_override=forcing.T_lowest - constants.T_freeze, dt=600.)
+
+
+@pytest.mark.parametrize("sw,lw,T_soil", [(0., 200., 262.), (700., 300., 285.)])
+def test_two_leaf_exports_screen_temperature_between_air_and_canopy(sw, lw, T_soil):
+    out = _two_leaf(sw, lw, T_soil)
+    theta = 270.0 + constants.g / constants.c_pd * 60.0
+    t2, tc = float(out.T_2m[0]), float(out.T_canopy_air[0])
+    assert np.isfinite(t2)
+    assert abs(theta - tc) > 1.0
+    assert min(theta, tc) + 1e-3 < t2 < max(theta, tc) - 1e-3, (t2, tc, theta)
+
+
+def test_tas_land_leg_takes_the_land_screen_temperature():
+    """Fails on the pre-fix code (no T2m_land): finite entries replace the
+    ocean-law land profile exactly, non-finite ones keep it."""
+    from types import SimpleNamespace as NS
+    from legoesm.driver.diagnostics import DiagnosticCollector
+    from legoesm.grids.vertical import create_sigma_coordinate
+    nlev = 20
+    coord = create_sigma_coordinate(nlev)
+    dc = DiagnosticCollector(nlev=nlev, sigma_full=coord.sigma_full,
+                             dsigma=coord.dsigma, vcoord=coord)
+    state = NS(T=NS(data=jnp.broadcast_to(jnp.linspace(250., 280., nlev), (2, nlev))),
+               u=NS(data=jnp.full((2, nlev), 5.)), v=NS(data=jnp.zeros((2, nlev))),
+               p_s=NS(data=jnp.full((2,), 1.0e5)))
+    q_v = jnp.full((2, nlev), 2e-3)
+    kw = dict(T_land=jnp.full((2,), 262.0), land_fraction=jnp.ones((2,)))
+    args = (state, q_v, jnp.full((2,), 275.0), jnp.zeros((2,)))
+    old = np.asarray(dc._tas_2m(*args, T_ice=271.35, **kw))
+    new = np.asarray(dc._tas_2m(*args, T_ice=271.35,
+                                T2m_land=jnp.asarray([266.5, jnp.nan]), **kw))
+    assert new[0] == 266.5
+    assert new[1] == old[1]
diff --git a/tests/unit/test_land_column_split.py b/tests/unit/test_land_column_split.py
index c7096ac176..b1ce1ac882 100644
--- a/tests/unit/test_land_column_split.py
+++ b/tests/unit/test_land_column_split.py
@@ -23,13 +23,13 @@ class _State(NamedTuple):
 
 def _step(state, cols):
     """Column-independent stand-in for the land step on columns ``cols``:
-    the same tuple layout (state, 10 per-cell fields, 5 counts, iter max)."""
+    the same tuple layout (state, 11 per-cell fields, 5 counts, iter max)."""
     T = state.T_soil.at[cols].set(state.T_soil[cols] * 1.01 + 0.5)
     wet = state.wet.at[cols].set(T[cols, 0] > 290.0)
     f = jnp.zeros(NCOL).at[cols].set(T[cols, 0] - 273.0)
     held = jnp.zeros(NCOL).at[cols].set((T[cols, 1] > 280.0).astype(float))
     n = jnp.sum(T[cols, 0] > 290.0).astype(jnp.int32)
-    return ((_State(T, wet),) + (f,) * 7 + (held,) + (f, f)
+    return ((_State(T, wet),) + (f,) * 7 + (held,) + (f, f, f + 1.0)
             + (n, n, n, 2 * n, n) + (jnp.max(cols).astype(jnp.int32),))
 
 
