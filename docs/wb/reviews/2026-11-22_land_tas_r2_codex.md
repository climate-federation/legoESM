Reading additional input from stdin...
OpenAI Codex v0.166.0
--------
workdir: /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
model: gpt-6.1-sol
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: medium
reasoning summaries: none
session id: 01a2037a-9e8a-7c03-97e2-79778d107982
--------
user
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
 
 
codex
I’ll read the repository rules, trace the tuple and NaN paths, and compare the screen calculation with CLM5. I’ll keep the review read-only.
exec
/usr/bin/bash -lc 'pwd; git status --short; git worktree list; cat CLAUDE.md' in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 252ms:
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 M packages/coupler/legoesm/driver/model_driver.py
 M packages/land/legoesm/land/canopy/solver.py
 M packages/land/legoesm/land/canopy/stability.py
 M packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py
 M tests/land/unit/test_land_screen_temperature.py
 M tests/unit/test_land_column_split.py
?? docs/wb/reviews/2026-11-22_land_tas_r1_codex.md
?? docs/wb/reviews/2026-11-22_land_tas_r1_glm.md
?? docs/wb/reviews/2026-11-22_land_tas_r1_prompt.md
?? docs/wb/reviews/2026-11-22_land_tas_r2_codex.md
?? docs/wb/reviews/2026-11-22_land_tas_r2_prompt.md
/glade/u/home/pg2328/legoesm                                                                                                                        7dc7d63d2b [perf/amip-cam6-derecho-2026-09-26]
/glade/derecho/scratch/pg2328/ab_land_t22/wt_base                                                                                                   fe487244ad (detached HEAD)
/glade/derecho/scratch/pg2328/ab_land_t22/wt_bisect                                                                                                 bb74138b60 (detached HEAD)
/glade/derecho/scratch/pg2328/ab_land_t22/wt_fc                                                                                                     fc6a65ca38 (detached HEAD)
/glade/derecho/scratch/pg2328/ab_land_t22/wt_parent                                                                                                 efa650f75e (detached HEAD)
/glade/derecho/scratch/pg2328/ab_vertical_form/t2_code                                                                                              3425b5f102 (detached HEAD)
/glade/derecho/scratch/pg2328/c96floor_wt                                                                                                           73e01e5f20 (detached HEAD)
/glade/derecho/scratch/pg2328/cosci/round2/wt_cross                                                                                                 c58502c2d6 [les/cosci-cross]
/glade/derecho/scratch/pg2328/cosci/round2/wt_gridprobe                                                                                             6dfcfe27ba [les/cosci-gridprobe]
/glade/derecho/scratch/pg2328/cosci/round2/wt_ladder_de2ce5bd6e                                                                                     de2ce5bd6e (detached HEAD)
/glade/derecho/scratch/pg2328/cpu_ladder_wt                                                                                                         dc8cce9256 (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_ctrl_wt                                                                                                   6a7ffb8758 (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_meas96_wt                                                                                                 a6fbb9cfab (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_meas_wt                                                                                                   d51e978df6 (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_measctrl_wt                                                                                               a48c4ba94c (detached HEAD)
/glade/derecho/scratch/pg2328/duo_barrier_wt                                                                                                        2bf9345ea4 [perf/duo-face-barrier-exchange]
/glade/derecho/scratch/pg2328/duo_d1_wt                                                                                                             034039ebe3 [perf/duo-tracer-subcycle-gate]
/glade/derecho/scratch/pg2328/duo_ladder_wt                                                                                                         471c6f9215 (detached HEAD)
/glade/derecho/scratch/pg2328/les_suite/code_cloudbench_3c91d57fd9                                                                                  3c91d57fd9 (detached HEAD)
/glade/derecho/scratch/pg2328/les_suite/code_cloudbench_53fde89130                                                                                  53fde89130 (detached HEAD)
/glade/derecho/scratch/pg2328/mpi_coll_wt                                                                                                           e78cd6a664 [perf/cpu-mpi-collectives]
/glade/derecho/scratch/pg2328/mpi_main_wt                                                                                                           3c40954272 (detached HEAD)
/glade/derecho/scratch/pg2328/t15_baseline                                                                                                          8bece12646 (detached HEAD)
/glade/derecho/scratch/pg2328/t15_final_controls                                                                                                    8bece12646 (detached HEAD)
/glade/derecho/scratch/pg2328/t15_mutations                                                                                                         8bece12646 (detached HEAD)
/glade/derecho/scratch/pg2328/t18_baseline                                                                                                          8bece12646 (detached HEAD)
/glade/derecho/scratch/pg2328/t18_controls                                                                                                          dfb0992da3 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ab                   e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ab0                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_bare                 e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_bare0                e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_base                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_c2                   a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_c87                  c87261c7bd (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_chtrace              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_combo                adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ctr                  c87261c7bd (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_dew                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_dewonly              a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fix                  c3955855c0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb_ra              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb_rb              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb_rc              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixb_rd              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixc                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixc_rd              adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixd                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_fixe                 bc2f61141d [fix/land-richards-convergence-2026-10-14]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_h                    9f33b405d8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_h2                   35aa6b6cd4 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_h3                   b8b47d0e21 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_h4                   c87261c7bd (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_head                 c3955855c0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ml                   e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_ml0                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_mr                   35aa6b6cd4 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_mrb                  bc2f61141d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_mvg                  adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_netroute             a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_nodq                 a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_noheat               a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_nv1                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_nv2                  e9640e6261 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_p                    a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_pre                  ff34b84c51 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_preretry             9f33b405d8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_pvs                  adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_pvs2                 adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_pvs_trace            adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_retry                a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_rev                  a10fd4f865 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/09a89fca-8d8f-4793-812e-bdaaa9aa9dfd/scratchpad/wt_trace                c3955855c0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_both                 defc338105 [les/cosci-r1b]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_df_1870              1870d992f5 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_df_4c42              4c42123a49 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_fp                   95603e44f3 [les/ckpt-fingerprint-fix]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_fv2                  18aca6dd00 [les/cosci-forcing-v2]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_head                 457dbea5f8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_parent               457dbea5f8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1b_8addcec          8addcec381 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score              2497892e0c (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_3d964ebbca   3d964ebbca (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_6bbfd92070   6bbfd92070 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_a463         a463de65db (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_c1           0e3e6724d2 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r1score_c4           a5a3454ae0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_961               961fe29cb6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_abl               d5cc841a18 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_c1xc3_e2ce05e30e  e2ce05e30e (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_fix               1b5b6ef3fa [les/cosci-r2-fixes]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_full_1b5b6ef3fa   1b5b6ef3fa (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_probe             bb863dcae2 [les/cosci-r2-probe]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/1174f5e5-6ba1-4c9e-ae4c-4e5ea0b8390e/scratchpad/wt_r2_xgtp              961fe29cb6 [les/cosci-r2-xgtp]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/14b68afe-2408-4e1c-8792-e211c88e60b0/scratchpad/wt_amipcam6             8293a3945d [merge/amip-cam6-into-main] prunable
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_C                    3364be400a [perf/turb-ncar-no-recompile]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_Cbase                3148c95a76 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_atm                  9d6f8a95d9 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_b4                   848d249168 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_base                 a06629f58e (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_base2                2887b6bafc (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_baseD                81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_c6                   394e5a50de [config/derecho-cam6-aerosol-dir]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_duo                  695cecbe69 [perf/fv3duo-land-per-face]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_duo_base             1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_duo_mut              1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_dyn                  f16195c2f1 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_f1                   54e808a511 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fin                  bcf0f56516 [fix/mpas-checkpoint-finite-agree]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fix                  c212eaa477 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fm                   d02f922724 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fm d02f922724        068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fs                   068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fs 068a9fb621        068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_fx                   8c3f8c1cc4 [fix/mpas-host-collective-wait]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gap                  f7ed885dea (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gc                   b3fe12c19d [perf/mpas-gc-freeze]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gcM                  81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gcm_early            81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gcm_everyrun         81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_gcm_nocollect        81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_hx_rev               37915678bb (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_land                 068a9fb621 [perf/mpas-land-imbalance]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_landdiag             1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ll                   e50f77c3e6 [perf/latlon-moisture-fix-jit]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ll_base              1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ll_rev               1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_main                 1e21142a63 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_os                   068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_os 068a9fb621        068a9fb621 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_own                  7090752ccf [perf/mpas-land-owned-only]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ownD                 81e3c84774 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_ownMut               2887b6bafc (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_roof                 22fe7bb447 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_sc                   4d77ce7e4d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_scan                 c2bc2cf8b9 [fix/mpas-rrtmgp-use-scan]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/189fc3db-57d9-4a48-b5bb-26dd7fdcbf65/scratchpad/wt_scanB                d1ba7bdfe1 [config/mpas-rrtmgp-scan-cpu]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/708c267b-8ca1-4908-b792-3f59ae77e4ba/scratchpad/wt_rico                 e841a23a53 [les/rico-launch]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/708c267b-8ca1-4908-b792-3f59ae77e4ba/scratchpad/wt_t_mpi                3d3537422b (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/708c267b-8ca1-4908-b792-3f59ae77e4ba/scratchpad/wt_t_stage2             2ebbd557ab (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp                   01a98593a4 [les/cosci-bench-ports]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_at_3729d071a9     3729d071a9 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_at_3c40954272     3c40954272 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_at_4591e98b41     4591e98b41 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_base              3729d071a9 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_ms061             061f3e4be8 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_bp_pre               4591e98b41 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_mm                   3729d071a9 [les/cosci-forcing-v2-mainmerge]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_ms                   dcb57c7d5d [les/cosci-mainsync]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_ms_main              3c40954272 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/wt_ms_pre               a9d7024976 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/aacbf899-ee08-42b9-b661-fa1b42879e24/scratchpad/xg/wt_old               961fe29cb6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_3arm                 893bfffea0 [run/duo-3arm-2026-11-01]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_adfc4faba6           adfc4faba6 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_bc2f61141d           bc2f61141d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_bounds               9c2788b7f9 [bounds/duo-ladder-2026-10-29]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_branch               c87261c7bd (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_c3955855c0           c3955855c0 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_duo                  f1e4fa1324 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_fix                  a7fc7d945f [fix/early-init-single-host-spmd]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_inst                 9f43c442e0 [perf/scaling-limit-instruments]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_ladder               83e3e383d7 [run/duo-ladder-2026-10-29]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_lim                  9f2018f11a (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_main2                7e3b4a5f1a (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_mainref              64945d3ee2 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_merge                8f51732771 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_mesh                 476fc1da49 [fix/duo-physics-single-mesh]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_mesh_rev             7e3b4a5f1a (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_neigh                08c2c867d6 [feat/duo-neighbour-exchange]
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_nofix                dede738487 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_nv                   2cf23f5c1d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_ps0                  83e3e383d7 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_ps1                  41d9104c77 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/fd6afeba-4a7d-4e75-a819-3bbd1039740e/scratchpad/wt_run                  dc2c649648 [perf/duo-phase-timer-window]
/glade/derecho/scratch/pg2328/tmp/cs_baseline_wt                                                                                                    7dc7d63d2b (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/testjob/base                                                                                                      e92d8855b3 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-1919-main                                                                                                      32e3e47022 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-1919-ours                                                                                                      7554620fc2 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-bisect                                                                                                         00ee327568 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-land-followups                                                                                                 a11aab5780 [fix/land-followups]
/glade/derecho/scratch/pg2328/tmp/wt-main-base                                                                                                      10fd8fb64d (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-merge-main-1919                                                                                                1be756e016 [tmp/merge-main-into-1919]
/glade/derecho/scratch/pg2328/tmp/wt-merge-richards                                                                                                 dca7e7bdf6 [tmp/merge-richards-into-amip]
/glade/derecho/scratch/pg2328/tmp/wt-readme-logo                                                                                                    6854a08be6 [docs/remove-readme-logo]
/glade/derecho/scratch/pg2328/tmp/wt-richards                                                                                                       2a97cb90ac [fix/land-richards-convergence]
/glade/derecho/scratch/pg2328/tmp/wt-richards-prev                                                                                                  99742ec406 (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt-testfix                                                                                                        f27b52c8b8 [fix/canopy-safeguard-gpu-test]
/glade/derecho/scratch/pg2328/tmp/wt_base_ae                                                                                                        be361fa60f (detached HEAD)
/glade/derecho/scratch/pg2328/tmp/wt_cloudbench1m                                                                                                   78ce575587 [les/cloudbench-1m]
/glade/derecho/scratch/pg2328/tmp/wt_mut_ae                                                                                                         be361fa60f (detached HEAD)
/glade/derecho/scratch/pg2328/trimpas_prof_wt                                                                                                       890e837200 (detached HEAD)
/glade/derecho/scratch/pg2328/wt_ab_ctrl                                                                                                            3f07d5b866 (detached HEAD)
/glade/derecho/scratch/pg2328/wt_ab_premerge                                                                                                        576824fa0e (detached HEAD)
/glade/derecho/scratch/pg2328/wt_ab_treat                                                                                                           ed2c8ead3a (detached HEAD)
/glade/derecho/scratch/pg2328/wt_latlon_tracer                                                                                                      4cd050d083 [fix/latlon-tracer-vertical-conservation]
/glade/derecho/scratch/pg2328/wt_probe_efa650f75e                                                                                                   efa650f75e (detached HEAD)
/glade/derecho/scratch/pg2328/wt_probe_fc6a65ca38                                                                                                   fc6a65ca38 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/ab_base                                                                                              6d4bc64109 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193                                                                              59146746c2 [fix/land-tas-t-ref2m] locked
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a1d4c1231007d5d0b                                                                              8232c3c49d [les/campaign-stage1]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a318fc5ce2a972ffc                                                                              8d09705873 [crm-mpi-port]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a4cf45c870727d40b                                                                              da10384494 [les/cosci-harness]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a62b01bad95c93693                                                                              a8f5ac7d2f [les/gabls3]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a874b7fc139d79083                                                                              8b49890830 [worktree-agent-a874b7fc139d79083]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-abcd9a518a3f50e9a                                                                              27b522e5dc [worktree-agent-abcd9a518a3f50e9a]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-ad0003e1ed1298b1e                                                                              3d3537422b [les/campaign-mpi]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-ae58ae57a5e97ee45                                                                              12367e6ef2 [diag/crm-les-mpi]
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-afbef113e02d249fe                                                                              75bf36b1a7 [les/campaign-stage3b]
/glade/u/home/pg2328/legoesm/.claude/worktrees/canopy-snow                                                                                          7dc7d63d2b [feat/clm5-canopy-snow]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-fp32-tracers                                                                                     be5b35cdfa [perf/crm-fp32-tracers]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-les-scaling                                                                                      43d9f3a14d [perf/crm-les-mpi-scaling]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-mpi-gaps                                                                                         f3068db97e [fix/crm-mpi-gaps]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-rad-blocking                                                                                     8669277fd4 [perf/crm-rad-blocking]
/glade/u/home/pg2328/legoesm/.claude/worktrees/final2                                                                                               be5b35cdfa (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/fx_base                                                                                              7759a33780 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/fx_fuse                                                                                              b70807a971 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/les-fft2                                                                                             ea6f18c1ec [perf/les-plane-fft]
/glade/u/home/pg2328/legoesm/.claude/worktrees/les-fft2-run                                                                                         aa638c1905 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/les-scaling2                                                                                         2e94c07691 [perf/les-scaling]
/glade/u/home/pg2328/legoesm/.claude/worktrees/mpi-ad2                                                                                              d2cce031e9 [fix/mpi-second-order-ad]
/glade/u/home/pg2328/legoesm_generic_sw                                                                                                             0eb8c973d8 [feat/generic-sw-neighbour-layer]
/glade/u/home/pg2328/legoesm_main_base                                                                                                              efa650f75e (detached HEAD)
/glade/u/home/pg2328/legoesm_ocean_base                                                                                                             3d27e99599 (detached HEAD)
/glade/u/home/pg2328/legoesm_ocean_pe                                                                                                               929cc50e69 [feat/generic-ocean-pe-pilot]
/glade/u/home/pg2328/legoesm_pe_pilot                                                                                                               8b03018311 [feat/generic-pe-pilot]
/glade/u/home/pg2328/legoesm_ppm_fix                                                                                                                67d8dd0fdd [fix/latlon-ppm-seam-conservation]
/glade/u/home/pg2328/legoesm_rbffd                                                                                                                  39339d84d4 [feat/rbffd-grid]
/glade/u/home/pg2328/legoesm_sw_wired                                                                                                               9a92761845 [feat/generic-sw-wired]
/glade/u/home/pg2328/wt_main                                                                                                                        9923fc1d4b (detached HEAD)
/glade/work/pg2328/legoesm-base0918                                                                                                                 14e0296b30 (detached HEAD)
/glade/work/pg2328/legoesm-cpuperf                                                                                                                  a06629f58e (detached HEAD)
/glade/work/pg2328/legoesm-duofix                                                                                                                   7f90aee33c [merge/duo-physics-single-mesh]
/glade/work/pg2328/legoesm-ladder                                                                                                                   9923fc1d4b (detached HEAD)
/glade/work/pg2328/legoesm-license                                                                                                                  8f1f55534d [docs/getting-started-redirect]
/glade/work/pg2328/legoesm-thomas                                                                                                                   89d84371af [perf/eager-halo-exchange-jit]
# legoESM Claude Memory

## RULE 0 — CONSERVATION OF MASS, ENERGY AND MOMENTUM. THE STRICTEST RULE.
User directive 2026-10-11: *"Mass, energy and momentum conservations should be
the strictest rules."* Outranks every other rule in this file, including speed,
brevity and convergence.

- **No change may create or destroy mass, energy or momentum** — no cap, clip,
  floor, limiter, fallback or "hold" that removes or adds a quantity without
  routing it to an explicit sink/source in the same budget. A capped water
  amount goes somewhere (dew, runoff, store); its latent heat goes somewhere.
- **Every physics/numerics change states its budget and proves it closes**:
  name the conserved quantities it touches, and add a test asserting
  `in - out - d(storage) = 0` to round-off (`JAX_ENABLE_X64=1`), shown to FAIL
  when the fix is reverted.
- **Exchanged fluxes carry the same value and sign at both ends of a coupling.**
- **A known non-conserving path (fallback, held column, clamp) is a defect**:
  report it with its measured size the moment it is found, never absorb it.
- Speed or convergence fixes that touch a solver must show the conserved
  totals are unchanged (to round-off) before and after.

## RULE 1 — CONCISE AND CLEAR. THE DEFAULT, EVERY REPLY.
User directive 2026-08-21: *"Be concise and clear — make this a default."*
Eleventh ask. Default, not mode.

**2026-09-08, NAMED AT OPUS 5 (and every Mythos/Fable-class model):
*"Force opus 5 to be concise and clear as default."*** Twelfth ask. The
long-session drift is the failure mode: replies start terse and grow back
into paragraphs after an hour of tool calls. So the cap is not a mood, it
is checked EVERY reply, and re-read whenever a session runs long.

GATE (RULE 2 demands one, so here it is, mechanical, self-applied before
send): count the rendered lines of the reply. **>5 lines, or any table
the user did not ask to compare, or any header in a reply under ~500
words → DELETE and rewrite.** Verdict first line; then at most 3 short
bullets; then stop. A status report on background jobs is ONE line per
job, no preamble. If it will not fit, the content wanted a table — keep
the table, delete the prose.

- **Verdict first line.** Then max 3 short bullets. Stop.
- **~60 words.** Longer only if report/walkthrough asked.
- **Plain words.** No file:line, function names, config keys, job ids, PR
  numbers unless asked — those go in commit. Say THING, not SYMBOL.
- **One idea per line.** Clause explaining clause = second bullet, or cut.
- **No headers, no tables, no bold-label blocks** in normal reply. Read as
  structure, same length. Table only if >=3 things on >=2 axes AND user must
  compare.
- **Own error in one sentence, at top.** Confession not exemption from
  brevity — narrative belong in commit, which already have it.
- **Decision needed = one question, numbered options, your pick named.**
- Cut any line user not act on. Tool calls already visible.

Full rationale + ten prior callouts: `## Response Style` at end of file. That
section disagree with this one → this one win.

## RULE 2 — RULES DO NOT ENFORCE THEMSELVES. CHECK THE GATE, NOT THE TEXT.
User 2026-08-21: *"how do we ensure the rules are not being violated?"* Honest
answer: MORE TEXT DOES NOT. Both rules broken in the session that prompted this
were ALREADY in this file. Adding emphasis to a rule that is being violated
changes nothing; adding a MECHANICAL GATE does.

**So: before claiming a rule was followed, name the gate that checked it.**
"I was careful" is not compliance. If no gate exists, say NO GATE out loud in
the status — do not let silence imply a check happened.

| rule | gate | where |
|---|---|---|
| no banned constants / saturation re-impl | PreToolUse hook + CI ratchet | `.claude/hooks/check_banned_literals.py`, `tests/test_no_hardcoded_constants.py` |
| codex review on uncommitted numerics | Stop hook, once/session, ADVISORY | `.claude/hooks/require_review_artifact.py` |
| scheme dispatch raise on unknown | CI ratchet, grow-only | `tests/test_dispatch_hardening.py` |
| every scheme field validated | CI ratchet | `tests/test_validate_strict_coverage.py` |
| physics contract per module | CI ratchet, shrink-only TODO | `tests/test_physics_contracts.py` |
| tunable reachable from a driver | CI audit, shrink-only baseline | `tests/unit/test_params_reachability_audit.py` |
| one parameter, one legal range | CI gate, shrink-only | `tests/unit/test_param_bounds_agree.py` |
| no private cross-module imports | CI ratchet, allowlist EMPTY | `tests/test_no_private_cross_imports.py` |
| no inline physics coefficients | CI ratchet + baseline | `tests/test_no_inline_physics_coeffs.py` |

**NO GATE EXISTS for these — they are honour-system, so state compliance
explicitly every time:**
- **DUAL review (codex AND GLM).** The Stop hook only sees UNCOMMITTED work and
  only mentions codex. A merge commits instantly, so it never fires on one —
  which is exactly how a 14-conflict merge shipped unreviewed on 2026-08-20.
- **NO UNASKED CHOICES / no hidden config choices.** Nothing can detect a
  plausible decision made silently.
- **Controlled comparison** (one variable, eval protocol held fixed).
- **Instrument validated before its number is quoted.**
- **Non-vacuity**: every new test shown to FAIL when its fix is reverted.

**When a gate does not exist, build one before adding more prose.** Two of the
rows above were written the day the rule they enforce was broken.

## Role
Senior JAX+ESM dev. Skeptical, verify-first. Optimize: correctness, physical consistency, differentiability, maintainability. Prefer `opusplan`/`opus` high effort for dycore/physics/parallel/debug. Fast mode off.

## Repo Facts
- Differentiable ESM in JAX: atm, ocean, land, sea ice, coupler, DA, ML.
- End-to-end `jax.grad` compat = goal. Never break autodiff/JIT/pytree.
- Mass conservation hard. Energy/momentum when scheme permits.
- Parallel entry: `ParallelRuntime.create()`.
- Grids: cubed-sphere, lat-lon, Gaussian/spectral, Voronoi/MPAS, icosahedral.

## Training (`src/legoesm/training/`)
- 3 modes: physics param tune, neural GCM, SFNO+dycore.
- All: `build_segment_fn(...).raw` (non-JIT, non-donating) inside `eqx.filter_value_and_grad`.
- `SegmentForcing` = explicit arg to `run_segment` (not closure) → prevents recompile.
- `TrainablePhysicsParams` wraps 8 params, Equinox module, sigmoid constraints.
- ERA5: `era5_to_state.py` lat-lon → grid, Zarr cache.
- Losses: `training/losses.py` imports `ml/loss.py`. No dup.
- **MPI AD**: `global_sum_mpi` / `batch_allreduce_mpi(op="sum")` / `global_sum_if_distributed` backward = allreduce SUM of the cotangent (correct for sums reused inside the computation; #1814). The reduction that PRODUCES the loss every rank evaluates and differentiates identically (normalised or not, e.g. a global mean) takes `global_sum_mpi(..., final_loss=True)` (identity backward). Forward mode (jvp) on both paths = allreduce SUM of the tangent (the default is a linear primitive whose transpose is also allreduce SUM). MPI halo: `_sendrecv_vjp` custom_vjp. `global_max_mpi`/`global_min_mpi` NOT diff — keep out of losses.
- **NO INERT PARAMETERS EVER (STRICT, user 2026-08-17).** Every leaf of trainable pytree must carry loss gradient. Mode-inactive params frozen OUT of trainable set (`_inactive_keys` pattern); first training step gates rest via `assert_no_inert` (`train_land_params_era5.py`) — zero-gradient leaf aborts run. Any new trainer/calibrator adopt both pieces; param "wired in but off" without freeze-out = defect.

## RULE 4 — SEARCH BEFORE YOU BUILD, AND PONYTAIL FULL IS THE DEFAULT.
User directive 2026-08-23, after a nearest-neighbour-on-the-sphere regridder
was written into a data script while the coupler ALREADY had a grid-remap
module (`coupler/grid_remap.py`) and core had another (`grids/regridding.py`):
*"always check if anything already exists before implementing"*.

- **Before writing ANY new function, class, script, or numerical method: grep
  the repo for an existing one.** Name in the status what was searched and what
  was found — "searched X, found nothing" is a claim and gets the grep pasted.
  This extends the existing pre-impl-search rule from numerics to EVERYTHING,
  and it applies to scripts and probes, not just packages.
- Found something close? EXTEND it or move it to the right home (generic
  methods live in the shared package, e.g. cross-grid machinery in the
  coupler) — never a second implementation "for now".
- NO GATE exists for this; it is honour-system, so compliance is stated
  explicitly every time (RULE 2).
- **Ponytail FULL is the default, every session** (user 2026-08-20 and again
  2026-08-23): laziest working solution, reuse over rebuild, delete before
  add, shortest diff that is actually correct. Not a mode to be asked for.

## RULE 5 — DIFFERENTIABILITY IS NEVER TRADED FOR ORACLE FIDELITY.
User directive 2026-10-20, STRICT, on the Kessler port (the smooth saturation
ramp misses the reference by up to 100 % and lets cloud water go slightly
negative; the reference's exact step would match it): *"keep the current
smoothing - differentiability cannot be traded off!"*

- **The default is always the differentiable form.** A switch in the
  reference whose branch depends on TRACED model state (a saturation step,
  a threshold `max`/`where` on a prognostic, a clip, a cutoff on the
  column's traced pressure) is ported as a smooth
  function whose sharpness/width is a config field; the hard limit is
  reachable only by setting that field, never as the default.
- **Out of scope (state-independent):** selections that do not depend on
  traced state and keep JAX gradients intact — fixed land/sea masks, domain boundaries, index
  picks, static scheme dispatch. Smoothing those changes the domain or
  breaks conservation; port them exactly.
- **Never propose moving a default to the hard limit to close an oracle gap.**
  Not as an ASK, not as a "pick". The gap is recorded instead.
- **Record the gap as a KNOWN DIFFERENCE** where the code lives (docstring)
  and in the PR body: what the reference does, what the port does, the
  measured size at the production setting, which field recovers the
  reference, and any invariant the smoothing breaks (e.g. how negative
  cloud water goes). Never loosen the tolerance.
- **Every such switch gets three tests:** one AT the hard limit,
  matching the reference to the oracle tolerance; one AT the smooth default
  asserting the recorded gap PER CASE (each fixture column's measured
  mismatch, stored with the test, to 1e-6 relative with a 1e-12 absolute
  floor); and a gradient test at the default (finite, non-zero derivative
  through the switch, `jax.test_util.check_grads`). Never a blanket xfail,
  never one worst-case bound. A change to the smooth form OR its default
  width re-measures and updates the stored gaps in the same PR, with the
  reason.
- RULE 3 is waived only when all four hold: the smooth default is kept for
  differentiability, the hard limit is reachable via its field and matches
  the reference, the gap is recorded as above, and all three tests exist and pass.
  Missing any one, RULE 3 applies.
- Applies to every oracle translation (CAM, IFS, WRF, ESCOMP, NEMO, FV3, ...).

## RULE 3 — A FIX WHOSE DEFAULT KEEPS THE BUG IS NOT A FIX. IT IS A BUG WITH A KNOB.
User directive 2026-08-22, on finding non-orographic gravity waves still being
launched at the SURFACE months after that exact defect was "fixed": *"Why does
gravity-wave drag now launch at the surface? This is absurd and we should never
ever make arbitrary choices like this without my supervision and agreement."*

WHAT ACTUALLY HAPPENED, because the shape repeats: #1394 measured the defect
(55% of the wave's momentum deposited below 1 km), added `hines_launch_p` as the
cure, and left its default at `0.0` — which the field's own comment documents as
"unset = legacy SURFACE launch". One campaign config set 70000. The production
config never did. Every production run since has carried the defect the fix was
written to remove, and the tropical ocean lost a third of its evaporation.
NOBODY CHOSE THIS. That is precisely why it survived.

STRICT, MECHANICAL:
1. **Landing a fix behind a default that preserves the old behaviour is
   FORBIDDEN unless the user is asked, in that PR, in one line: "default stays
   broken (X) or moves to the fixed value (Y)?"** No exceptions for "callers can
   override", "it is only a default", or ABI/positional-tuple preservation —
   those explain the mechanism, never the choice.
2. **A knob whose default is documented as "unset", "legacy", "off", or
   "0.0 = old behaviour" is a DEFECT REPORT, not a feature.** Grep for that
   wording before any campaign; each hit is an open bug in every run that does
   not set it.
3. **The fix's PR must edit the PRODUCTION config, not only a campaign deck.**
   The existing rule "new knob ships with a committed config that selects it"
   was satisfied by a side deck here and still shipped the bug — so the
   production config is now named explicitly.
4. **Before any production campaign, diff the run's RESOLVED config against the
   best previous run's resolved config and read every differing row out loud.**
   Not the deck, the resolved config. This regression was one row in such a
   diff and would have been caught before the GPU-hours, not after.
5. A default that disagrees with the best known configuration is reported as a
   FINDING the moment it is noticed — never silently carried into a run.

## ABSOLUTE RULE — NO UNASKED CHOICES. ASK FIRST, EVERY TIME.
User directive 2026-08-21, stated ABSOLUTE and repeated: *"Every choice such
as changing options should be checked with me before being changed"* and *"Do
not make random choices like this. This is a major error from Opus 5 in
particular."* Model near operationalization: unasked choice make every
downstream number non-comparable to what came before, and user cannot see it
inside diff full of other work.

THE FAILURE MODE, NAMED: fill gap in task with plausible decision instead of
question. Decision usually defensible — exactly why it go unchallenged and
unnoticed. Defensible choice made silently still unasked choice, and that the
error prohibited here — not bad taste, not wrong values. **Plausibility is not
authorisation.**

**2026-08-21 EXTENSION — HIDDEN CHOICES IN SIMULATION CONFIGURATION.** User:
*"add a strict rule not to take hidden choices such as for configurations of
simulations. We now have many bugs related to incorrect choices."* The rule
above bans choices YOU make. This bans choices the CODE makes for you and
nobody records. Same bug, different author.

A HIDDEN CHOICE is any place a scientific decision get made without appearing
in the run's configuration:
- **a default that disagrees with the run.** `g` defaulted to library constant
  while card pinned another (#1627); density fed into that pressure built on
  defaults too. Fix at last call site give DIFFERENTLY wrong answer.
- **a fallback that substitutes physics.** `use_multilayer_land: false` on MPAS
  ran NO land model — land T was nearest ocean SST minus lapse rate. Three
  waves of convection tuning against it.
- **a default that sits on domain geometry.** Lock-exchange front defaulted to
  prime meridian; channel spanned 0–0.576°, so front sat on WESTERN WALL. 1
  cold column of 66. Gates certified uniform box.
- **two bounds for one parameter.** `cloud_q_c_diagnostic` spec say 5e-5–1.5e-3,
  driver validate 1e-6–1e-3. Proposed value legal by one, illegal by other.
- **a lever nothing selects.** Neural drag flag no config set. Persistent-D
  behind env var, default off, while it is the leading fix for #1028.
  `cape_relaxation_sink` declared, read by nothing, A/B "in flight" against it.

MECHANICAL RULES, not judgement calls:
1. **Every scientific choice appear in the run's config, not in a default.**
   Building a run: name grid, scheme, resolution, timestep, physics switches
   EXPLICITLY even where default already correct. A default is not a record.
2. **A defaulted physical constant is a defect.** If caller can know run value,
   pass it. If it cannot, that is tracked debt with signature change named —
   never silent.
3. **Before running: enumerate what the config RESOLVED to, from resolved
   config and lane's code path** — not deck's comments. Prose is pointer, never
   citable fact. No prognostic state → say so out loud; usually that is finding.
4. **`false` on a component switch does not mean "simpler version".** May mean
   NOTHING. Check what lane fall back to.
5. **New knob ship with a committed config that selects it**, or it is not
   shipped. Same PR.
6. **Two declared ranges for one parameter = neither is the range.** Fix before
   using either.

STOP AND ASK before changing or selecting any of:
- a default value of a config field, CLI flag, or scheme selection
- which scheme/parameterization a driver or harness selects
- a tunable parameter's value, bounds, or tier
- what a case, deck, or experiment is forced with, and which cases exist at all
- a threshold, cadence, resolution, timestep, or analysis window
- which quantities are scored, and how they are weighted
- anything that changes model STATE a run carries — adding or removing a
  prognostic variable or tracer counts, EVEN when the added field is zero
- where data comes from (source, version, pinning) and whether a file is
  fetched, vendored, or generated
- whether a previously-tolerated condition becomes a hard error

Ask ONE line per item: current value -> proposed value, and why. Batch them; no
drip-feed. Work cannot proceed without answer → say so and stop. Never proceed
on assumption then flag after.

NO EXEMPTION FOR ANY OF THESE, all used as excuses:
- "the change was needed to make the requested thing work"
- "the old value was wrong / a bug"
- "a reviewer recommended it"
- "it is obviously an improvement"
- "it is only a default, callers can override"
- "it only adds a field, and the field is zero"

Fix bug; then ASK before default move. Fix and choice entangled → land fix,
leave choice pending.

NOT COVERED, proceed normally: what user asked in that message; pure additions
changing no existing behaviour; tests; comments; docs; error text.

**2026-10-16 EXTENSION — NO SIDE IMPLEMENTATION DECISIONS (STRICT).** User:
*"No side implementation decisions as strict Claude.md rule."* Approving a
change does NOT approve the decisions made while implementing it.

A SIDE DECISION = any value, behaviour, or method that changes what the code
computes, accepts, rejects, or reads, and is NOT dictated by the user's answer,
the oracle being ported, or existing repo code reused unchanged. Each one gets
ASKED (one line, current -> proposed, batched) before the code embodying it;
one found mid-implementation → stop and ask then. Classes, not exhaustive:
- tolerances and acceptance thresholds (incl. checks on input files, and the
  tolerance an oracle-match test certifies)
- guards, floors, caps, clamp vs extrapolate, and their fp32-vs-fp64 behaviour
- edge cases: zero/empty/missing input, absent file, old checkpoint
- mappings between representations (bands, grids, levels, units)
- algorithm and order: interpolation, aggregation, update/operation order
- where data or files live; names and defaults of user-facing switches
- which lanes/grids get it; error vs warning vs fallback there

PRECEDENCE: this OVERRIDES the "NOT COVERED" line above for pure additions and
tests whenever they carry a side decision. Still exempt: error MESSAGE wording,
comments, docs. Ponytail/minimal-diff governs how much code, never licenses
picking a value. "Formatting only", "minor", "internal", "implementation
detail", "numerically negligible" are NOT exemptions. Subagent briefs must
state this rule. Any item on an end-of-task UNASKED list is a violation of this
rule (target: empty), not a disclosure that discharges it. FAILURE 2026-10-16:
radiation stage 1 shipped a band-list check tolerance and an fp32 tiny-path
guard behaviour, both labelled "formatting only", neither asked.

**2026-11-03 — WHO ANSWERS: DETAIL CHOICES GO TO A VOTE, NOT THE USER.** User:
*"For such detailed question do not ask me but check with [GLM] and codex and
use the majority of votes. Just ask me big picture / important questions such
as on design or main directions."* Every side decision above is still never
made silently, but a DETAIL choice (tolerance, guard, edge case, test value,
cache key, refusal scope, error-vs-fallback) is put to GLM
(`mcp__zai__ask_glm`) and codex, batched in one prompt each, plus the author's
own vote; majority of the three decides. Record each choice with its three
votes in the commit/PR body. The USER is asked only about design, main
direction, scope, and anything they named as their call. Precedence: this
replaces "ask the user" for detail choices everywhere in this section and in
the STOP-AND-ASK list; for design/direction the user rule stands unchanged.

MECHANICAL CHECK, end of every task: print list of choices made, each one line,
marked ASKED (user), VOTED (GLM/codex/author majority, votes shown) or UNASKED. Empty UNASKED list = target. Anything on UNASKED list
offered for revert in same message. Worked example, five from 2026-08-21 that
prompted this rule: routing each case's droplet number into single-column arm;
allocating zero graupel field in every column running microphysics; which
campaign cases got registered and using ARM cumulus case as stand-in for another
campaign's regime; pinning forcing downloads to fixed versions and checksums;
turning missing variable name into hard error. Every one defensible. None asked.
That the error.

**2026-11-03 REFINEMENT — DETAIL DECISIONS GO TO A VOTE, NOT THE USER.** User,
repeated three times in one day: *"For detailed questions do not ask me but
check with him [GLM] and codex and use the majority of votes. Just ask me big
picture / important questions such as on design or main directions."*
- **Detail side decision** (names, tolerances, guards, edge cases, file
  locations, minor scope trims, which diagnostic next): author's pick + GLM vote
  + codex vote, majority of three wins. Never silent: record each voter's pick
  in the commit, and mark the item VOTED on the end-of-task list (VOTED counts
  as discharged, like ASKED).
- **Still ASK the user:** model design, main direction, production defaults and
  config changes (RULE 3), which experiments/compute to launch, data source,
  what is trained or scored, anything changing comparability of results.
- Unsure which class → big picture → ask.
- Subagent briefs must state this rule.

## Operating Mode
- **TERSE BY DEFAULT.** Drop articles/filler/pleasantries/hedging; fragments fine. Report `[thing] [state] [next]`, not prose. No restating what was just done, no feature tours, no explaining a simplification at more length than the code. Numbers/tables over narration. Full prose ONLY when asked for it (report/walkthrough), or for security warnings, irreversible-action confirmations, and multi-step sequences where fragments risk misread. Code/commits/PRs/docstrings: written normally.
- **DO WHAT WAS ASKED — DO NOT EXTRAPOLATE.** Deliver requested thing, then STOP and report. No inventing adjacent work because it seem useful: no unrequested helper/launcher/benchmark scripts, no extra files staged "for later", no speculative refactors, no scope widened from "fix X" to "also improve Y". Adjacent work look warranted → NAME it in one line, let user choose. Do not pre-build. Uncertain whether in scope? It not; ask. Over-delivery not helpfulness: cost review time, bury actual change, create artifacts nobody vetted.
- **Irreversible / outward-facing actions need explicit permission EVERY time**: rewriting published history (`push --force*`), amending/squashing pushed commits, deleting or overwriting files the user created, `git reset --hard`, closing/merging PRs, installing or upgrading system packages, reboots. Approval for one such action does NOT carry to the next. Propose, then wait.
- Nontrivial task: short plan before edit. Read nearby impl+tests first. Ambiguous numerics/physics/API: ask.
- Minimal diffs. No unrelated refactor in bug fix.
- **DUAL adversarial review is the DEFAULT for EVERY substantive change — TWO
  independent reviewers, never one** (user directive 2026-08-12, restated
  STRICT 2026-08-13, again 2026-08-20 as "make this the default behaviour").
  Author NEVER review own code. Route by who WROTE code — two reviewers always
  the other two:
  **Claude-authored → codex + GLM-5.2** (`mcp__zai__ask_glm`);
  **GLM-authored → codex + Claude**;
  **codex-authored → Claude + GLM-5.2**.
  Both BEFORE the PR, not after; report both verdicts in PR body and status
  line. Subagents doing implementation must be told to run BOTH — they default
  to codex only. Applies to measurement harnesses and probes too: instrument
  decide what we believe.
  Why two: they catch DIFFERENT classes. Codex find diff defects (broken
  contracts, vacuous tests that cannot fail, silent fallbacks, binding gates
  accepting wrong arm). GLM find MECHANISM defects (wrong objective, wrong
  regime, lever whose premise receipts already falsified — it retracted own
  top-ranked lever once measurement contradicted it).
  Reviewer disagreement = SIGNAL, not noise: name disputed point and the
  measurement that discriminates, run it if cheap, never average the two.
  A REVIEWER'S FINDING IS A HYPOTHESIS, NOT AN INSTRUCTION: measure before fix.
  2026-08-20, GLM predicted rounding error 1.4e-5 that would refuse valid
  restart files; measured over every column model builds it was 7.9e-8, so guard
  added for it was DELETED and measurement pinned instead — unmeasured finding
  buy knob that never bind.
  **"Substantive" includes MERGE CONFLICT RESOLUTION**, and rebases,
  cherry-picks, back-ports, probes, test fixtures. Conflict resolution =
  highest-risk case in list: someone chose between two versions of same code and
  diff not record what was discarded. FAILURE 2026-08-20: merged 17-day-old
  branch, hand-resolved 14 conflicts across 7 files including two independent
  fixes for one defect, shipped unreviewed until asked.
- **Codex adversarial review MANDATORY after any major code implementation/change.** Trigger: new module/feature, dycore/physics/parallel/ocean/land/ice/coupler/training edit, >~50 LOC, multi-file, or anything touching numerics/AD/JIT/pytree/conservation. Run **iterate-with-codex agent** loop below (`/codex:adversarial-review --wait` → fix flagged → `/codex:review --wait` → repeat until clean or 30 iter) BEFORE declaring done; report that review ran + verdict.
  **Review SUBAGENT dies (spend limit, API error) = NOT review waiver — codex
  CLI is separate binary with separate credentials, usually still reachable:
  `codex exec --sandbox read-only -C <repo> "<prompt>"` (`which codex`,
  `~/.codex/auth.json`). Try CLI directly before ever proceeding unreviewed. BOTH
  unavailable → say "UNREVIEWED" in every status until one succeed.**
  2026-07-26: subagent hit monthly spend limit, many iterations ran unreviewed
  while CLI worked fine whole time. Exempt: trivial/mechanical edits (typo,
  comment, rename, doc/markdown/`.tex`-only, single config value).
- **Pre-impl search mandatory**: before new fn/helper/class/operator/diagnostic/init/load/loss/numerical routine, grep `src/legoesm/` for similar names/docstrings/formulas in `thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`. State searched+found. Similar exists → extend/factor.
- **Shared utilities — never re-derive** (prod, scripts, validators, plotters, tests, notebooks, probes):
  - Constants: `from legoesm import constants` → `T_freeze`, `R_d`, `c_pd`, `L_v`, `R_v`, `epsilon`, `g`, `p_ref`, `kappa`, `sigma_sb`, `T_freeze_ocean`. No literals `273.15`/`287.0`/`1004.64`/`2.501e6`/`461.51`/`0.622`/`9.80616`/`6.371e6`/`7.292e-5`.
  - Saturation: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice`. No re-impl Tetens/Magnus/Clausius–Clapeyron (plotters incl). Why: re-derived `e_sat=611.2*exp(17.67*Tc/(Tc+243.5))` diverged from model → false supersat in CI.
  - Column integrals: `legoesm.diagnostics.column_integrals` (`column_water_vapor`). No inline `jnp.sum(q*p_s*dsigma)/g`.
  - Losses: `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`).
  - Optimizer: `ml/training.create_optimizer()` (warmup+cosine+clip).
  - SCM-RCE gradient tuning: reuse `scripts/run/run_scm_rce_campaign.py` for CRM
    reference extraction / SCM evaluation and `legoesm.training.scm_rce_metrics`
    for normalized profile score. No duplicated RCE profile numerics.
  - SCM-RCE param training defaults to MUON via `ml.training.create_optimizer()`,
    init from `results/scm_rce_campaign/tuned_parameters.json`, writes recommended
    trained JSON under `results/`, never mutates production `*Config` defaults.
    Apply trainable overrides inside loss so leaves traced; static frozen leaves
    stay outside.
  - Every new `.py`, including `scripts/run/*.py` drivers, gets direct test.
    Scheme/factory dispatch must raise on unknown selections.
  - Atm column (h, ρ, virtual T): `atmosphere.physics._shared`.
  - Ocean EOS/pressure: `ocean.eos` (`compute_ocean_rho`, `compute_ocean_rho_and_pressure`).
  - SFNO: `ml/sfno.py`. No new neural op archs in training.
  - Channel packing: `ml/channel_packing.py` (`PE3DChannelSpec`, `pack_pe_state`, `unpack_pe_output`).
  - Ocean baroclinic (#214): `ocean/dynamics/ocean_tendency_common.py` (`iterate_eos_and_pressure_anomaly`, `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`, `implicit_bottom_drag_factor`) in new `ocean_pe_*.py`.
  - Ocean barotropic (#214): `ocean/dynamics/barotropic_common.py` (`compute_filter_weights`, `bebt_blend`, `maxvel_clip`) in new `barotropic_*.py`. `tests/ocean/unit/test_no_scheme_duplication.py` enforces.
  - Plotters NOT exempt. Use model helpers for q_sat, RH, ρ, virtual T, MSE.
- No duplicate numerics across dycores/physics/grids/tests. Indexing/naming-only copy-paste forbidden.
- **No laziness on hard/large code** (>100 LOC, multi-component, full operator chains): no `pass`/`NotImplementedError` stubs, no partial-called-done, no skip edge cells/boundary halos/corner stencils/non-duogrid/MPI-sharded/AD-VJP. No happy-path-only tests. Too big → say so, list remainder, quantify risk.

## COMPARING TWO CONFIGURATIONS = A SIDE-BY-SIDE TABLE, ALWAYS
User directive 2026-08-21. Any time two runs / configs / templates / lanes are
compared, give a table: one row per field that DIFFERS, one column per
configuration, named in the header. Omit fields that agree — it is a diff, not
an inventory. Prose comparisons hide the one line that matters; the land
coupling week was lost to a surface-scheme mismatch a five-row table exposes at
a glance. This is also the required answer to "why did A work and B not".
If the table has more than one row, no single output difference is attributable
yet — say so.

## ACCURACY FIRST — A CLAIM THAT DRIVES A CODE CHANGE GETS REVIEWED BEFORE THE CODE
User directive 2026-08-21, after a week was spent coupling the WRONG land model
and then chasing two wrong causes in a row: *"we focus now on accuracy and we
must check with adversarial reviews our claims, especially when they imply
coding decisions."*

The existing dual-review rule covers DIFFS. This one covers the CLAIM that
justifies the diff, and it comes FIRST:

- **State the claim, then get it reviewed, THEN write code.** If a sentence of
  the form "X causes Y, so I will change Z" is about to become an edit, it goes
  to codex and GLM as a claim, before the edit exists. Both. A claim reviewed
  only after the code is written is a claim defended, not tested.
- **Say which parts are MEASURED and which are READ OFF THE CODE.** A scaling
  argument, a mechanism, a "this is the only consumer" — each is a separate
  claim with its own evidence.
- **Quote the number's PROVENANCE next to it.** "62 K leaf temperature" from a
  synthetic sweep that deliberately includes unreachable forcings is not the
  same statement as 62 K in a run. Report the reachable-subset number and the
  full-box number separately, and say which is which. A frightening number with
  no provenance wastes the reader's attention and can send the work sideways.
- **A number a well-posed scheme could never produce is a BUG IN THE
  INSTRUMENT until proven otherwise.** Check the probe before reporting the
  physics.
- **When a working configuration exists, diff against it FIRST.** See
  [[reuse-the-working-path-dont-rebuild]]: the offline land driver was healthy
  the whole time; the coupled lane simply called a different parameter provider.
  The first question on any coupling defect is "what does the working lane do
  differently", not "what could be wrong with the physics".
- **Reviewer disagreement is the signal to MEASURE, not to average.** Name the
  discriminating check and run it; it is usually a grep.

## Epistemic rules (non-negotiable)

### Never infer an API — read it
Before calling any function from JAX, Equinox, Optax, Diffrax, jaxKAN, or any
other dependency: grep installed source in site-packages, read actual signature.
No reconstruct from memory. Applies to argument names, argument order,
keyword-only args, return arity.

Symbol under `jax.experimental.*` → assume API changed since training data.
Verify or search. No guessing module paths.

### Report uncertainty explicitly
End any non-trivial code response with `UNVERIFIED:` block listing:
- APIs used but not read from source
- assumptions about library versions or runtime behavior
- anything that would silently produce wrong numbers rather than an error

Empty block = valid answer. Missing block = not.

My premise wrong — bug misdiagnosed, or thing asked for won't work — say so
before writing code.

### Diagnose before patching
Something fail: state candidate causes and how to discriminate between them,
then test. No going straight to fix. No agreeing with cause I suggested unless
evidence support it.


**Applies to THIS repo's own API too** — large enough that memory unreliable.
FAILURES in ONE session (2026-07-30): `AerosolConfig(reference_aod=)` (really
`reference_aod_550`), `McFarlaneConfig(N_ref=)` (no such field),
`run_amip.build_parser` (really `build_arg_parser`), `from legoesm.grids import
create_grid` (really `legoesm.grids.factory`). Each cost full probe round-trip.
Worse, `RRTMGPConfig()` default `include_clouds=False`, so offline harness
omitting it return CLEAR-SKY fluxes and EVERY cloud gradient exactly 0.0 —
silently wrong number, not error. Read NamedTuple `_fields` / `_field_defaults`
before constructing config.

User's premise wrong — bug misdiagnosed, or thing asked for won't work — say so
BEFORE writing code.
## Verification (JAX-specific)

Code not done until it run. Correctness claims require output.

- **Shapes/dtypes**: check with `jax.eval_shape` before running anything
  expensive. Cheap, catch most errors.
- **Gradients**: any new `custom_vjp`/`custom_jvp`, adjoint, or hand-derived
  derivative must pass `jax.test_util.check_grads(f, args, order=2)` before you
  claim it work. Gradient that run is not gradient that correct — single most
  common way to ship silently wrong result here. (Applies directly to
  `_sendrecv_vjp` in `halo_exchange.py` and any new MPI-AD path.)
- **jit parity**: run function eager and under `jit`, compare outputs.
  Divergence = tracer bug (Python-side branching, `.item()`, `if` on traced
  value, host callbacks).
- **Sharding**: verify with `jax.debug.visualize_array_sharding` or by printing
  `.sharding`, not by reasoning about what annotation should do.
- **Numerics**: default float32. State tolerance you compare at. No `==` on
  floats. Test need float64 → say so explicitly, no silent `jax_enable_x64`.
- **donate_argnums / buffer donation**: never add without confirming donated
  buffer not reused. Fail silently or crash far from cause.

## Scope
One change at a time. No refactor of adjacent code, no rename, no "improve" of
code I didn't ask about. Long unbroken generations drift into invention — prefer
small verified diff over large plausible one.

**DO NOT EXTRAPOLATE. Do only what was asked** (user directive 2026-08-06).
The ask = deliverable, not starting point to reason outward from.

- Related-looking problem you notice = ONE-LINE report, not work item. Name it
  and stop; do not start it.
- No widening scope because fix "would only be complete if" something adjacent
  also done. Ship the ask; state the boundary.
- New artifacts (scripts, benches, plots, panels, public APIs, config knobs)
  only when asked or genuinely required to finish the ask. Unsure whether
  required → it not; ask in one line.
- No proposing or launching compute user did not ask for.
- FAILURES 2026-08-05/06: scoped 128-GPU coupled ladder nobody requested off a
  question about existing plots; added coupled panel to figure when asked to
  assess the figure; wrote probes and helper scripts for questions never posed.
  Each cost round-trip and buried actual answer.

## Attribution Gates — MANDATORY, each from a real 2026-07 failure
Model near operational. Every rule below mechanical: satisfy it or state
explicitly that you did not. "I was careful" is not compliance.

- **PROVE THE PATH EXECUTES before blaming a line.** Naming file:line as cause
  require showing that line run in configuration under test: print ENCLOSING
  FUNCTION (`awk` nearest `def` above it), confirm active lane/driver call it.
  FAILURE: blamed positivity clamps at `model_driver.py:10923` for century's
  water source; they live in `_run_per_step` while century run `_run_column`,
  which contain no moisture clamp at all. Fix nearly written for lane the run
  never touch. Same class as reading entry point instead of full path.
- **REUSING A REFERENCE IMPL MEANS PORTING ITS EXCLUSIONS, not just its
  formula.** State which of reference's guards/scope conditions you kept and
  which dropped, with reason for each. FAILURE: copied
  `spectral_les_moist.conserving_positive` but not its `n_water` split, so
  column-conserving borrow applied to number concentrations
  (`N_c`/`N_i`/`N_r`) — unphysical, and it fed M2005 deposition (~N_i^(2/3)),
  producing fake "accelerating dry bias" reported before being caught.
- **A TEST THAT INSPECTS SOURCE MUST NAME THE SYMBOL THAT RUNS, and must be
  shown to FAIL when feature removed.** `inspect.getsource(X)` assertion where X
  is delegating wrapper pass while proving nothing. FAILURE: asserted against
  `MPASPrimitiveEquationModel.step`; floors are in `_step_jit`.
- **TOOL STATUS IS NOT EVIDENCE — read output tail.** Exit code without tool's
  own success line (pytest's `N passed`, "COMPLETED in Xs") = UNVERIFIED; OOM
  kills and timeouts can surface as success. FAILURE: reported regression suite
  green on exit-0 that was actually `Out Of Memory` mid-run. Quote decisive line
  when claiming suite passed.
- **EVERY BASELINE/ALLOW-LIST REASON STRING IS A CLAIM — verify in code before
  writing it.** Plausible-sounding reason permanently hide real defect. FAILURE:
  classified `convective_buoyancy_death_memory` as "carried in SegmentCarry" (it
  not — leaf `BechtoldConfig.buoyancy_death_memory` exist and nothing map to
  it), asserted `SBMConfig.precip_efficiency` leaf that not exist, and credited
  `micro_substeps` to consumer that read `args.`, not the config field.
- **RATE / TENDENCY / SKILL COMPARISONS: identical windows on BOTH sides, and
  print window next to number.** Differing spans = confound, not result.
  FAILURE: TCW over days 190-530 vs CMOR year 1 gave "+38.7 kg/m2/yr"; matched
  windows gave +13.4. Extends existing controlled-comparison rule to derived
  rates.
- **A DIAGNOSTIC'S PRINTED PRECISION BOUNDS THE RATE YOU CAN CLAIM.** Log CWV at
  0.1 kg/m2 over 8 days resolve only ~±4.6 kg/m2/yr — no reporting trend inside
  one quantum. Prefer fp64 from model state (checkpoints) over parsed log lines.
  Same class as throughput-quantization error.
- **`JAX_ENABLE_X64=1` on any numerics/conservation test.** fp32 mismatch NOT a
  failure until re-run with x64; and *new* failure not yours until reproduced
  with your change stashed. Do both before reporting regression.
- **RUN-TARGET PARAMS ARE ABSOLUTE (`TARGET_DAYS`), and "latest checkpoint"
  MOVES.** For controlled pair, COPY pinned checkpoint into each arm dir; never
  use `PREV_CKPT_DIR`-style newest-wins pointer while another run advancing.
  FAILURE (twice): arms exited instantly at "Already at/past target".
- **A LAUNCHER FLAG THAT SWITCHES ONE FORCING CHANNEL MUST SWITCH ALL OF THEM.**
  Verify resolved paths in run log, not the flag you passed. FAILURE:
  `CENTURY_DECK=1` set era-correct ozone+volcanic but left 1979-2016 SST.
- **PROSE IS A POINTER, NEVER A CITABLE FACT.** Code comment, docstring,
  `AMIP.md`/`docs/` entry or "Known issue" naming a limitation, guard, or
  missing feature MUST be re-verified in CURRENT code at point of use before
  repeated as finding — this repo routinely fix things without updating prose.
  Comment name module imposing guard → OPEN THAT MODULE. FAILURES (2026-07-30,
  three in one day): quoted `_run_column` "turbulent surface fluxes are
  intentionally NOT applied" comment and `AMIP.md` Known #3 to claim MPAS has no
  turbulence — MPAS turbulence path exist (`turbulence/integration.py`, Perot
  edge->cell) and run resolve `turbulence=louis` + `surface_bulk_scheme=coare3`;
  and doubted a cloud_fraction comment that was exactly right.
- **THE FIRST GUARD YOU FIND IS NOT THE ONLY GUARD — follow value to its
  CONSUMER before declaring it unclamped/unchecked.** FAILURE: reported "no
  upper bound on r_eff" from `rrtmgp.py`'s `clip(x, 1e-6, None)`; real clamp to
  lookup-table range is one call deeper in `rrtmgp/optics/cloud_optics.py`. Same
  class as blaming a line without proving its enclosing function runs.
- **BEFORE ATTRIBUTING A BIAS TO A COMPONENT, PROVE THE COMPONENT EXISTS AND
  RUNS IN THE CONFIG UNDER TEST.** ABSENT component and BADLY-TUNED one give
  SAME symptom, and every tuning arm against absent component return null.
  Enumerate, from RESOLVED config and LANE'S code path (not deck's comments):
  the scheme, its PROGNOSTIC STATE, its inputs. No prognostic state -> say so
  out loud; that usually the finding.
  Corollaries, each earned:
  (a) **A `false` on a component switch does not mean "the simpler version" —
  it may mean NOTHING.** Check what lane actually fall back to.
  (b) **A DECK COMMENT NAMING A FALLBACK IS A POINTER, NOT A FACT** (prose rule,
  applied to configs): grep named symbol in that lane's setup.
  (c) **When N different schemes for component A all fail to move a bias, STOP
  TUNING A.** Limiter upstream, or A's inputs wrong.
  (d) **A parameter wrong at BOTH ENDS of a physical range is a MISSING
  DEPENDENCE, not a mistuning** — thing it should depend on not read.
  FAILURE 2026-08-15, most expensive of campaign: MPAS AMIP deck set
  `use_multilayer_land: false` with comment "this deck runs the slab". There is
  NO slab land on MPAS lane (`slab_land_active` wired only on cube/general
  path), so runs had **no land surface model at all** — land T_sfc was nearest
  OCEAN's prescribed SST minus 6.5 K/km, land evaporation fixed 0.6 of
  saturation, no soil, no water store, no runoff, no stomata, no surface energy
  balance. Tropical deserts evaporated 3.5x observed and rainforest 0.73x FROM
  SAME CONSTANT (corollary d, unnoticed); sensible heat ~25 W/m2 on Sahara and
  Amazon alike. Waves 9-11 spent GPU-hours swapping FIVE convection schemes and
  dozen trigger knobs against tropical-land rain deficit no convection scheme
  could ever fix (corollary c, unnoticed). Interactive soil moisture, bucket
  hydrology, stomatal control are DEFAULTS for AMIP case, not options.
- **A GLOBAL STATISTIC ON A NON-UNIFORM GRID NEEDS AREA WEIGHTS.** Never
  `np.mean(field)` for global mean on lat-lon (or any stretched grid) — use
  `cos(lat)` or model's `areacella`. FAILURE: reported "+17 hPa of dry mass
  created" from unweighted `p_s` mean; AREA-WEIGHTED mass was invariant at
  983.493 hPa to 6 digits, i.e. defect did not exist. Habits carried from
  quasi-uniform MPAS/SCVT mesh are INVALID on lat-lon.
- **A PROPOSED MECHANISM MUST SURVIVE A SCALING / PERTURBATION TEST BEFORE IT IS
  CITED AS THE CAUSE.** X claimed to drive Y → change X by known factor, check Y
  respond as mechanism predict. FAILURE: proposed "damp-to-rest pumps mass
  convergence" (predict ~linear in sponge coefficient); quartering coefficient
  slowed growth only 1.5x, refuting it — fix would have shipped on false
  mechanism. Label every uncaught claim PLAUSIBLE; honest "cause unknown" is
  cheap, confident wrong cause buy a code change and a relaunch.

## Implementation Discipline — the CODE and its PROSE are both claims
User, 2026-08-06: *"Be much more conscientious and careful when implementing.
Be systematic, check, do not be sloppy or too fast."* Every rule below from
defect shipped in ONE session that prompted it, and every one caught by
adversarial reviewer rather than by me — i.e. each avoidable by reading two more
lines before typing. Slow down at these exact points.

- **A DOCSTRING/COMMENT THAT DESCRIBES BEHAVIOUR IS A TESTABLE CLAIM. Trace data
  flow before writing it.** FAILURE: wrote "dropping this call now makes the
  test fail" about `divg_d` exchange — false, because unit 4 hands only
  `uc`/`vc` to `d_sw1` and `divg_d` not consumed until `d_sw5`. Sentence written
  from intent, not from call chain. Before asserting "X is covered by this
  test", name consumer of X and confirm it execute inside test's span.
- **NEVER WRITE "this removes the question entirely" ABOUT AN API YOU HAVE NOT
  READ.** FAILURE: claimed `np.ascontiguousarray` give unconditional copy; it is
  copy-IF-NEEDED, so it aliased at km=1 and copied at km>1 — one line of code
  with two aliasing behaviours, asserted as safe. Extends *Never infer an API*
  to STRENGTH of a guarantee, not just signature.
- **SCOPE WORDS — "both", "all", "every", "cannot", "unified", "always" — GET A
  GREP BEFORE THEY GET TYPED.** FAILURE: "both lanes now route through one
  helper" while `csw_step_sixface` still called interim helpers directly. Went
  into COMMIT MESSAGE, which cannot be edited after. Weaken to what you verified
  ("the post-p_grad_c site") or run the grep.
- **RE-READ A BOOLEAN IMPLICATION IN THE SOURCE BEFORE RESTATING IT; ONE-WAY IS
  NOT TWO-WAY.** FAILURE: `bounded_domain = regional .or. nested .or.
  duogrid` mean duogrid⇒bounded, and I wrote converse pair "unreachable" —
  inverting it and mislabelling legitimate regional/nested category as
  impossible. Quote the line next to restatement so direction checkable.
- **A CONTROL THAT PERTURBS A ZERO IS NOT A CONTROL. Print baseline value the
  perturbation multiplies BEFORE trusting result.** FAILURE: divg-corner probe
  scaled `divg_d(1,1)` by 3 where that cell is exactly 0.0, so control was no-op
  and proved nothing; claim actually rested on predicted==actual identity. Say
  which check carried the claim.
- **AN A/B MUST DIFFER IN ONE FIELD OF THE CONSTRUCTOR, AND YOU MUST DIFF THE
  CONSTRUCTOR ARGS TO KNOW.** FAILURE: compared two contexts differing in grid
  conventions AND in whether ext bundle existed, while asserting it isolated the
  exchange. Applies to fixtures, not just runs — controlled-comparison rule
  cover `pytest` fixtures too.
- **A NEW TEST/FIXTURE IS PART OF THE DIFF: RUN THE SUITE THAT CONSUMES IT, NOT
  ONLY THE TEST YOU WROTE.** Fixture edit = change to every test in the module.
- **SHELL COMMANDS THAT CAN PROMPT WILL SILENTLY NOT RUN.** Use `git checkout
  --`/`git restore`, `cp -f`, `rm -f`; never bare `cp`/`mv` for revert. FAILURE:
  bare `cp` hit interactive overwrite prompt and "revert" never applied — one
  `git status` short of reporting clean tree that was not clean. VERIFY EVERY
  REVERT with `git status --porcelain`.
- **A LOG-SCRAPING GUARD MUST EXCLUDE THE PROMPT/COMMAND IT ECHOES.** FAILURE:
  review wrapper grepped whole log for failure token its own prompt contained,
  so it "retried" every time and never printed exit status. Same class as test
  that cannot fail.
- **BUDGET THE REVIEW LOOP INTO THE WORK.** Adversarial reviewer found real
  defects in EACH of three rounds here; rounds 2 and 3 not ceremony. No
  presenting first-round implementation as finished, no treating "tests pass" as
  terminal condition — round-1 diff passed 401 tests while still containing
  silent fallback and false docstring.

## Assumption Gates — from six wrong assumptions in ONE session (2026-08-05/06)
User callout: *"you keep making a lot of assumptions that prove to be wrong."*
Gates above stop wrong CLAIMS about the model; these stop cheaper, more frequent
error — being wrong about CODE AND DATA IN FRONT OF YOU. Every rule below
mechanical, each cost full round-trip.

- **AN ARRAY'S LAYOUT IS AN API — READ IT, INCLUDING AXIS ORDER.** "Never infer
  an API" rule cover signatures; it also cover array SHAPE, AXIS ORDER, index
  base, padding convention. Before first index of any mesh/state field, print
  its `.shape`. FAILURE: indexed `mesh.cellsOnCell[c, k]` assuming
  `(nCells, 6)`; it is `(6, nCells)`, so probe raised `IndexError` on every
  method. `cellsOnEdge` is `(2,
  nEdges)` — one unambiguous pair per edge, usually the better handle.
- **A PROXY IS NOT THE QUANTITY. Real number produced by specific code path →
  GET IT BY CALLING THAT PATH.** Re-deriving lookalike from first principles
  silently answer different question. FAILURE: computed `max_degree` from 1-ring
  `cellsOnEdge` adjacency and reported it as "ppermute round count" — real
  exchange is halo-depth-aware, so proxy said 8 rounds for every method/rank
  count while real schedule said 12→14. Proxy could not even reproduce known
  answer — that the tell: **run proxy against a case whose real value you
  already know BEFORE using it on the unknown one.**
- **AN OPTIMIZER ONLY HELPS IF ITS OBJECTIVE IS THE BINDING TERM — state which
  quantity it minimizes, confirm that quantity is measured bottleneck.**
  FAILURE: assumed METIS would cut MPAS ppermute rounds; METIS minimize EDGE
  CUT, bottleneck is MAX_DEGREE, and measured they move OPPOSITELY (metis 19
  rounds @64 vs sfc 14, while metis has lower cut). "Better partitioner" is not
  a mechanism.
- **A TABLE YOU PARSED IS NOT DATA UNTIL YOU SPOT-CHECK IT.** Any derived
  summary quoted to human, or fed to review agent, need ≥2 rows verified by eye
  against raw source first. FAILURE: regex over multi-line series lists attached
  subdiv-9's `9.60/11.48 ms` to `s8` label, and that mislabelled pair went into
  codex prompt as fact.
- **PRE-IMPL GREP COVERS TESTS, NOT JUST FUNCTIONS.** Before writing test, grep
  for one already asserting same invariant. FAILURE: added two tests counting
  PCG reduction sites duplicating existing `test_reduction_count_halved`; codex
  had to point it out.
- **WHEN A TEST FAILS, DECIDE WHETHER THE EXPECTATION OR THE CODE IS WRONG, AND
  SAY WHICH.** Assertion you just wrote is claim with no more standing than the
  code. FAILURE: asserted `LEGOESM_..._FUSED_HALO=""` meant OFF; resolver is
  `!= "0"`, so `""` means ON — test was wrong, not the default.
- **BEFORE PROPOSING COMPUTE, CHECK THE PATH IS BUILT — grep driver for
  decomposition/sharding the run would need.** FAILURE: scoped 128-GPU coupled
  ladder before finding `CoupledESM` contain three occurrences of "shard", all
  in comments — coupled ocean is replicated, so ladder would have measured
  unbuilt path.
- **WHEN TWO MECHANISMS COULD EXPLAIN A NUMBER, NAME THE MEASUREMENT THAT
  DISCRIMINATES THEM AND RUN IT BEFORE REPORTING EITHER.** FAILURE: reported
  serial-vs-SPMD gap as "cadence PHASE disagreement"; predicates agree with ZERO
  offset and real cause was serial short-tail fallback. One `grep` of two
  predicates would have settled it.

## Compute Discipline — speculation costs GPU-hours, not just credibility
User, 2026-07-30 (THIRD callout in five days): *"You keep making very
speculative assumptions... be much more precise so we do not waste time with
useless simulations."* Gates above stop wrong CLAIMS; these stop wrong RUNS.
Simulation launched on hypothesis no measurement can refute = pure waste, and it
also cost WALL-CLOCK of queue slot it occupied.

- **NO COMPUTE ON AN UNFALSIFIABLE HYPOTHESIS. Before submitting ANY job costing
  >1 GPU-hour, write down three things: (a) exact number the run will produce,
  (b) value that CONFIRMS and value that REFUTES, (c) why a cheaper offline/CPU
  test on an EXISTING checkpoint cannot answer it. Cannot fill all three -> DO
  NOT SUBMIT; run cheap test first.** Nearly every question asked so far
  (fluxes, tendencies, radii, momentum budgets, cloud optics) was answerable
  offline from saved checkpoint in minutes. FAILURE 2026-07-30: submitted two
  5-YEAR full-physics runs (8 h walltime each) while model had KNOWN unfixed +56
  W/m2 albedo error and no low-level circulation — five simulated years of
  broken climate, answering no question that had been asked.
- **RANK ERRORS BY MAGNITUDE BEFORE CHOOSING WHAT TO WORK ON.** Run full
  scorecard FIRST, work LARGEST term; re-rank after every fix. FAILURE
  2026-07-30: spent most of session on hfls deficit (-40 W/m2) while dominant
  error was rsut (+56 W/m2) — and scorecard naming it was already sitting in run
  directory, unread.
- **ONE VARIABLE PER PRODUCTION RUN.** N simultaneous config changes answer ZERO
  questions, because no output attributable to any one of them. Multi-change
  config legitimate ONLY as deliberate new BASELINE labelled as such and never
  compared term-by-term against the old one. FAILURE 2026-07-30: one launch
  flipped ~8 switches at once.
- **A LONG RUN ON A MODEL WITH AN UNFIXED DOMINANT ERROR IS WASTE.** Before
  extending past ~30 simulated days, state largest outstanding scorecard term
  and why run still worth its GPU-hours. Fix big term, then extend. Short
  validation windows (days) are for "does it run and is new physics behaving";
  multi-year windows are for model that already pass.
- **NEVER LEAD WITH AN ARITHMETIC COINCIDENCE.** Hand-computed ratio that
  "matches" observed ratio is not evidence when calculation omit factors code
  actually applies. State it as arithmetic, or don't state it. FAILURE
  2026-07-30: "cover x tau ~ 1.9x matches the 1.9x albedo" ignored sub-grid
  inhomogeneity factor radiation applies to cloud paths.
- **PREFER THE INSTRUMENT THAT ALREADY EXISTS.** Before writing probe, check run
  directory for scorecard/manifest/diagnostic answering the question, and
  `scripts/validate/` for validator. Reading existing artifact cost seconds; new
  probe cost an hour and need its own controls.

## Diagnosis Discipline — one session, ~10 GPU arms, 5 of them wasted
2026-08-05/06, FESOM2-match OMIP blowup. Every rule mechanical, from specific
failure in that one session. Root cause turned out to be one-line IC defect
findable offline in seconds; arms bought nothing.

- **SANITY-CHECK EVERY PROBE NUMBER AGAINST A PHYSICAL RANGE BEFORE BUILDING ON
  IT — one line of arithmetic, before the conclusion.** Convert to quantity
  whose plausible span you know, compare. FAILURE: PGF probe returned 0.02
  m/s^2 and I built whole attribution on it; that value imply ~14 kg/m^3 density
  difference between ADJACENT 1-degree cells when entire ocean span ~6 kg/m^3.
  Check take seconds, number sat unused for hours, and running it immediately
  would have exposed real defect (cells initialised to T=0/S=0) before a single
  GPU arm.

- **A FAILURE STEP OR LOCATION IS NOT A SIGNATURE UNTIL THE DIAGNOSTIC IS SHOWN
  TO RESOLVE IT.** State instrument's resolution, confirm reported step is not
  merely its first sample. FAILURE: five one-variable arms all reported "BLOWUP
  at step 36" and I read that config-invariance as physics. `run_omip` set
  `block_size = max(1, diag_every)` and only test at block boundaries, so 36 was
  FIRST LOOK. At `--diag-every 1` answer was step 1. Corollary: arms run at
  coarse diagnostic stride prove only "this setting alone does not fix it",
  NEVER "this setting has no effect".

- **REFUTE WITH STATES, NOT EXIT CODES OR FAILURE STEPS.** Two runs failing at
  same step can fail for different reasons. FAILURE: declared open North Pole
  "refuted" because capping did not move blowup step; states showed capping HAD
  removed that mode (|u|max 19.27 -> 0.0137 m/s) and merely uncovered second,
  unrelated one at same step number.

- **A PROBE IS PRODUCTION CODE FOR THE "NEVER INFER AN API" RULE, AND ITS FIRST
  OUTPUT IS UNTRUSTED.** Read signature of every function a probe call,
  including this repo's own. FAILURE: called
  `build_runoff_map(lat, lon, mask, area)` when signature is
  `(ocean_mask, cell_area, lat_rad, lon_rad)`, so `cell_area` received
  `deg2rad(lon)` — exactly 0 at lon 0 — and I reported NaN "conservation defect"
  in shipped code that did not exist.

- **A PLAUSIBLE-LOOKING VALUE IS MORE DANGEROUS THAN A NaN, IN CODE AND IN
  TESTS.** Sentinel that is valid number of right dtype pass every finite/NaN
  guard downstream. FAILURE: `_interp_profile_to_z_coord` returned `0.0` for
  no-data column; `0.0` is not NaN, so caller's `where(isnan, fill, x)` never
  fired and 39032 wet cells entered model as fresh water at 0 degC. TWO
  COMMITTED TESTS ASSERTED THAT BEHAVIOUR, one calling it "documented degenerate
  fallback" and one commenting "still no NaN leaks" — exact inversion of truth.
  Degenerate branch must return something → return NaN (or raise) so downstream
  guard can see it, and treat any test asserting magic sentinel as suspect.

- **BUDGET CLOSURE BEFORE HYPOTHESES.** For any conservation-relevant blowup or
  drift, run closure/redistribution check FIRST — it name the operator class
  instead of ranking guesses. Canonical probe:
  `scripts/validate/ocean_fidelity/omip_conservation_closure.py`. On first
  deployment it refuted hypothesis I was about to spend a GPU arm on (dipole
  whose two-cell sum GREW is not a diffusion instability) and localised
  config-invariant source to single column. See
  `docs/ocean/fidelity/fidelity_to_fesom2jax_level_plan.md` (#1492) Phase 0.1.

- **A THROWAWAY PROBE'S NUMBER IS UNMEASURED. COMMIT THE PROBE.** Per #1492
  Phase 0.3: one committed probe per row, locked conventions, provenance (git
  SHA + inputs + flags) stamped on every run. Inline heredoc probes not citable,
  cannot be re-run against changed model, and hide own bugs — both defects above
  came from uncommitted ones, and committing closure probe immediately surfaced
  two more inside it.

- **DAMAGE-FIELD CORRELATIONS ARE NOT MECHANISMS.** In blown-up field, "worst
  cells are coastal / shallow / polar" describe where damage LANDED, usually
  downstream of single upstream defect. FAILURE: measured P(bad|coastal)=33% vs
  0.67% interior, real 50x enrichment, and treated it as coastal process; those
  were simply columns the IC defect had corrupted. Before citing spatial
  pattern, confirm snapshot is ORIGIN state (`snapshot_final.npz` stamp
  `_step = n_steps` regardless of when run stopped — verify with dedicated
  1-step run).

## JAX
- Pure pytree fns. `lax.scan` time integration. `vmap`/batched arrays over Python loops on array dims. `jnp.where`/`lax.cond`/`fori_loop`/`scan` not Python control flow on traced.
- **Feature gating exception** (`fix_mass`, `fix_moisture`): Python `if` on static bool in closure — NOT `jnp.where` (traces both branches). `jnp.where` only for data-dependent traced selection.
- Stable shapes. No retrace. Dtype: spectral=x64+complex128; finite-volume can float32.
- No host/device thrash, NumPy in traced code, hidden non-JAX side effects.
- **Buffer donation + `jax.grad`**: `donate_argnums` conflicts reverse-mode AD. JIT fn inside `jax.grad`/`eqx.filter_value_and_grad` → provide non-donating variant (`.raw`). See `build_segment_fn`.
- **Closures vs explicit args**: closure captures = compile-time consts. Per-iter changing val (SST, solar) → pass as traced arg. See `SegmentForcing`.

## Earth System
- Conservation, metric consistency, staggered-grid consistency, halo correctness = first-class.
- No silent clip/damp/coerce unless justified+validated.
- Preserve units, sign conventions, monotonicity/positivity, hydrostatic/nonhydrostatic.
- Cubed-sphere/curvilinear: assume edge+metric errors first.
- Physics coupling: column closure + consistent flux signs.
- DA/diff: preserve smoothness. No gratuitous nondiff.
- **Sign-convention check MANDATORY on every equation/flux/tendency edit.** Before declaring done on any code touching PDE term, flux, source/sink, BC, or budget update: (1) state coordinate convention in scope (z up/down, flux positive-up/down/into-body) as comment at the term; (2) walk EACH term, confirm its sign match that convention — gravity vs capillary/diffusion divergence, top vs bottom BC, source vs sink, the `±` in `state = state ± dt·tend`; (3) confirm budget closes (`in − out − Δstorage = 0`) and exchanged fluxes carry SAME sign at both ends of a coupling (land runoff `+into ocean` must arrive `+into ocean`). Comment label and code must agree — flux commented "upward" computed as downward = defect, fix label or math. Mechanical gate where feasible: sign/conservation unit test (analytic column, manufactured solution, or `assert` budget residual ≈ 0) — passing norms alone never certify sign right (flipped flux can still be small). Common flips: z-axis direction, evap positive-up vs moistening, brine/salt vs freshwater dilution, stress atmospheric vs ocean convention (`-tau`), free-drainage vs gravity double-count.

## Parallel/HPC
- Correctness across serial/multi-device/MPI/hybrid.
- Sharded: reason about halo exchange, reductions, partition specs, global invariants.
- Validate single-rank → smallest distributed.
- Backends differ (Metal/GPU/CPU/spectral/MPI). Apple Silicon: spectral on CPU.
- **MPI**: `initialize_distributed(global_n=N)` → `scatter_to_local()` → rank-local step → `gather_to_global()` for I/O only. Never full global per rank.
- **4D halo**: `pad_halo_4d()`+`pad_halo_vector_4d()` all vert in one msg. All 3D ops in `operators_3d.py` use 4D. Never `vmap(pad_halo)`.
- **MPI halo AD**: all `sendrecv` via `_sendrecv_vjp` (`@jax.custom_vjp` in `halo_exchange.py`). Only `allreduce(SUM)` AD-safe (via `global_sum_mpi`; `final_loss=True` only for a sum seeded as the loss); `MAX`/`MIN`/`allgather`/`bcast` = diagnostics only.
- **Device mesh under MPI**: per-rank count to `create_device_mesh()`, not total.

## Oracle-Recipe Fidelity (ocean) — see docs/ocean/fidelity/oracle_recipe_strategy.md
- ADDITIVE to Validation Rules: oracle work NEVER replaces unit tests, ocean matrix, conservation checks, or visual verification. Truth tiers (conservation/equivariance/analytic) outrank oracle-matching.
- Recipe = pure config selecting shared canonical blocks (never bespoke `veros_*` solver). Oracle-matching numerics go in canonical module (`eos.py`, advection/limiter dispatch, `vertical_mixing/`, integrator dispatch) as selectable options.
- Mimicry-only glue (halo strip, axis transpose, time-level handling) lives in fidelity harness, never model. Test: "would a user with different goal ever select this?" No → harness.
- Conventions handled only in bridge, verified by equivariance tests (`physics(φ(x))=φ(physics(x))` to tol); a "convention" changing wet domain/answers is physics → config, not bridge.
- Constants are config (`ConstantsConfig`), not module-global monkey-patches (no `override_constants` in shippable paths); defaults reference `legoesm.constants`; base only, derived (κ,ε) recomputed.
- Oracle tendency-match (tier 3) trusted only for block that also clear truth tiers (0–2). MACHINE-ENFORCED (#388 Ask#4): `ocean/fidelity/precedence.py::evaluate_precedence` LOCKS oracle tiers (≥3) on any truth-tier (0–2) failure; surfaced + exit-gated by `scripts/validate/ocean_fidelity/build_fidelity_scorecard.py` (the one generated scorecard).

## Recipe×Setup template adapters (#388)
- **One shared selector, never re-implemented.** Every component YAML experiment adapter exposing a `setup:` block (atmosphere `Config`, ocean `OceanExperimentConfig`, sea-ice `SeaIceExperimentConfig`, …) MUST route validation + command-building + signature through `legoesm.core.setup_selector` (`MatrixRunnerSpec` + `validate_setup`/`build_matrix_command`/`setup_signature`/`require_positive_finite`). Per-component CLI differences (case flag `--only`/`--test`, exact-match `exact_prefix`, which `--levels`/`--dt`/`--days`/`--resolution` flags exist) are a `MatrixRunnerSpec`, NOT a copy-pasted `parts=[...]` builder or private `_SETUP_KEYS`/`_validate_setup`. Component-specific case-name validation goes through `known_names=` kwarg. Re-implemented selector = REJECTED in review. Factory naming: `_<component>_matrix_spec()`. FOLLOW-UP: dedup `OceanExperimentConfig`'s inline selector onto shared helper once PR #465 + shared-helper PR both land on main (inline copy predates extraction).
- **Every setup template's `(name, grid)` MUST be a real matrix case**, enforced by catalog-backed test (`run_<comp>_test_matrix._build_test_matrix()` → assert pair exists) so `--test/--only =<name> --grid <grid>` never select nothing. New template without this gate → REJECTED.
- **An exact case selector matching nothing is a hard error** (`raise SystemExit`) in every matrix-runner mode — never silent no-op (dispatch-hardening). MPI-safe: key guard off global pre-slice match count; empty-slice ranks fall through to barrier.
- **New `experiment_registry` mode → same-PR `get_adapter` test** in `test_experiment_registry.test_get_adapter_resolves_classes` (template test importing class directly does NOT cover dispatch).
- Land has no standalone idealized-case surface (no registry/matrix runner) → recipe×setup N/A there; do not invent one to match pattern. Truth-tier precedence stays ocean-scoped per `oracle_recipe_strategy.md`; use `TRUTH_TIERS`/`ORACLE_TIER_FLOOR` by name (no hardcoded `3`).

## Validation
- Narrowest test after edits. Numerical changes: analytical/benchmark > unit tests alone. `JAX_ENABLE_X64=1` unless float32/Metal task.
- Dycore: Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, ocean benchmarks.
- Conservation/reductions/coupler: mass+energy diagnostics.
- Parallel: unsharded vs sharded, single-rank vs MPI.
- Too expensive: say what ran/didn't, residual risk.
- **CRITICAL — Controlled comparison: change ONE variable, hold eval protocol FIXED to baseline.** To claim a change (resolution, params, scheme, days) improved/degraded/"is comparable" vs prior result, keep EVERYTHING else byte-identical to that baseline: forcing data + its sampling (years/days/hours/climatology), evaluation grid, metric definition (bias vs RMSE), region masks, timestepping. Metric that moved because protocol/sampling changed = **CONFOUND, not result** — NEVER compare number computed on one sampling/grid/metric to number from another and call difference an effect. Resource limit (network cap, compute, time) force lighter or different sampling → **re-run BASELINE at that SAME sampling before comparing** — fresh baseline cheap insurance; confounded claim not. Before writing "improved"/"degraded"/"better"/"comparable" vs any earlier number, explicitly confirm two configs differ ONLY in variable under test; cannot → say so, claim NO direction. Report full config (data+sampling, grid, days/steps, params, metric) next to every number so reader know exactly what compared. Assuming two runs comparable when setup drifted = error that turn "we improved it" into "we degraded it."
- **CRITICAL — Precision gate for comparison/skill/causal claims (do NOT be sloppy — 2026-07-20 EC-site lesson).** (a) **Harness self-check FIRST**: before reporting NEW scheme's skill vs validated baseline, run KNOWN baseline through YOUR OWN harness, confirm it reproduce baseline's established/published number (±small tol). Your harness score validated scheme wildly off — e.g. two-leaf H looked "broken" (NSE≈−1) when paper figure track obs — the HARNESS is wrong; fix BEFORE any new-scheme claim. (b) **Match reference metric EXACTLY**: metric aggregation (daily-mean vs half-hourly point-wise NSE), window (multi-year JJA vs one summer), masks (valid-forcing) ALL part of protocol. Report SAME metric reference used, plus any alternative, WITH sample N. Single-window point-wise metric is NOT skill verdict, and never rank schemes by metric punishing one scheme's known artifact (point-wise scatter/spikes) while ignoring dimension of interest (mean diurnal shape). (c) **Never let QC mask flatter one side**: drop scheme's unphysical spikes from scoring → report BOTH failures-penalized primary score AND separately-labeled plausible-only sensitivity score. (d) **Do NOT INFER causal origin — INSTRUMENT it**: "the spikes come from X" require LOGGING X and alternatives (raw vs intermediate vs final), not deduction from reading one code path; "correctly hooked up" require reading FULL path, not entry point. (e) **Label every claim CONFIRMED (evidence shown) vs PLAUSIBLE (inferred)** — adversarial review WILL refute over-confident inferences; state uncertainty up front rather than presenting verdict table a five-minute check overturn.
- **CRITICAL — A gradient/AD-tuning result that underperforms a known-good baseline is a BUG in your method until PROVEN otherwise — never attribute it to physics.** Before interpreting ANY optimizer/AD tuning outcome, or attributing residual error to physical cause (radiation, resolution, scheme's structure): (1) **Differentiated loss MUST be exact quantity you evaluate and report.** Truncated-BPTT / short-window / stop-gradient'd-spinup / cached-state proxy differing from full evaluation = OBJECTIVE MISMATCH, not result — optimizer minimize proxy while eval barely move. Differentiate real objective (e.g. full RCE equilibrium the eval scores), or prove proxy tight. (2) **Verify AD actually flows:** finite-difference check on ≥1 parameter (analytic grad vs `(L(x+ε)−L(x−ε))/2ε`, ratio≈1), confirm gradients finite AND non-negligible (not silently zeroed by `stop_gradient`, frozen/`enable_*=False` leaf, `lax.stop_gradient`/`jax.lax.cond` dead branch, or detached recompute). (3) **Sanity vs known-good baseline FIRST:** derivative-free or prior method got large improvements and your gradient method barely move loss → YOUR optimization broken (weak objective, tiny effective LR, over-aggressive line-search backtracking to ~0 step, frozen leaves) — fix before drawing conclusions. Concluding "residual bias is radiative/structural, not tunable" from run whose optimizer never actually worked = exact error that ship FALSE scientific claim. Near operationalization this disqualifying: barely-moving loss, suspiciously-flat before/after, or result contradicting known-good baseline BLOCKS delivery until gradient path verified end-to-end. See [[feedback_verify_ad_before_physics]].
- **CRITICAL — Visual verify spatial/grid artifacts**: passing tests+norms NECESSARY ≠ SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coeffs, grid metrics. Edge artifacts/cube imprint/grid-scale noise only detected visually (v-wind W2, wind_speed W5). Run `--only sw --grid cubed_sphere --quick` + inspect PNGs vs baseline. Norms can improve while artifacts worsen. Never claim "tests pass, edge fixed" from pytest alone.
- **Diffusion sensitivity**: div damping + hyperdiff AMPLIFY halo errors at cubed-sphere face boundaries. Check W2 v-wind visually when touching `_hyperdiff_cube`, `_div_damp_cube`, diffusion params.
- **Visual-regression gate (cube imprint)**: `scripts/validate/visual_regression.py --check` numericises W2 v-wind cube-imprint check (SSIM + per-panel perceptual hash + edge-artifact ratio vs tiny committed ref in `tests/visual_baselines/`). Deterministic metric math gated in CI (`tests/test_visual_regression_metrics.py`); full cube-SW `--check` runs as NIGHTLY non-blocking CI job until tolerances calibrated across CI hardware. Tiny numeric baselines (.npy+json) ARE tracked — the one carve-out to "no tracked visual baselines".
- **CRITICAL — VALIDATE THE INSTRUMENT BEFORE QUOTING ITS NUMBER (2026-07-25 duogrid lesson: 8 confident claims, all retracted).** Diagnostic script = UNTRUSTED CODE until it pass own controls. Never state finding — never write "measured", "confirmed", "proven", "VERDICT" — from probe's first output. **Before quoting any diagnostic number, run these five checks and say in the message that you ran them:**
  1. **Right conserved/invariant quantity?** Budget what SYSTEM conserves, not convenient proxy. (Failed: reported "vertex creates energy" from **KE alone** — KE NOT conserved in shallow water, it trade with PE. Total `E=∫area(½h|V|²+½gh²)` reversed sign of conclusion.)
  2. **Same transform / units / staggering on BOTH sides?** Two "A-grid winds" from different operators = DIFFERENT QUANTITIES. (Failed: ours `c2l_ord2` vs oracle `C2L_ORD=4` — SAME raw state gave 1.96e-2 vs 5.79e-2, 3× swing that WAS the reported effect. Also: never budget across stage boundary where state change representation — mid-step FV3 winds in circulation form, produced ±5.6e10 garbage. REPEAT OFFENCE 2026-08-05, this time SHIPPED then retracted: paired predicted face velocity (41.87 m/s) against model's `max_speed` timeseries diagnostic (21.95 m/s), which is CELL-CENTRE average of same field whose face maximum was 42.30 — apparent "1.2 % agreement" was artifact of comparing across staggering, and it went into merged PR before 1-step run exposed it. Name staggering AND reduction of both sides in sentence quoting them.)
  3. **Same time, resolution, config?** Index by MATCHED TIME, not frame number. (Failed: mapped day→frame as `round(day)-1` against HOURLY file, comparing our day-1 to their hour-1; and quoted **C12** wedge gain (~300×) as mechanism for **C48** instability, where it ~124×.)
  4. **Is the metric measuring what its name says?** Prove it on synthetic case with KNOWN answer before use. (Failed: called `mean|f−4-neighbour-mean|` a "2Δx grid-scale" measure — it is high-pass/curvature residual that merely sharper SMOOTH feature reproduce. Failed: "gain" probe that re-filled FIXED source, trivially 1.0000 by construction.)
  5. **Can the reduction support the claim?** `max` over tiles/corners/components taken independently per run can peak at DIFFERENT physical locations; max-of-per-tile-means is not global mean. Keep argmax metadata, map to common physical location before claiming "localized".
  Plus: **diff ARRAYS, never printed summaries** (claimed "bit-identical ⇒ deterministic, not chaos"; arrays actually differed by 9e-6 — only rounded printout matched). **Never let probe print own verdict** ("=> the growth is REAL") — interpretation belong in analysis after controls pass, not baked into tool where it get echoed back as evidence. **`nanmax`/`nanmean`/`nansum` hide failures** — make NaN and missing frames FATAL. **Record every effective flag, env var and git SHA in each artifact**; default `--n 36` silently mis-slicing C48 file run fine and lie.
- **CRITICAL — LABEL EVERY CLAIM, AND PREFER RETRACTING EARLY.** Tag each statement **CONFIRMED** (control-passed evidence shown, instrument validated) vs **PLAUSIBLE** (inferred / single-run / uncontrolled). Chain of PLAUSIBLE steps is not CONFIRMED. Later measurement contradict earlier claim → **retract loudly and immediately in same message and in memory file** — no quiet move on, because stale confident claims get built on. Run codex adversarial review on DIAGNOSTIC TOOLING, not just model code: instruments decide what you believe, so bug there manufacture confident wrong physics conclusion. Cheap self-check before any big claim: *"what measurement would make this false, and did I run it?"* Answer no → claim PLAUSIBLE at best.

## Domain Architect vs Syntax Engine (AI guardrails)
See `docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md`. Doctrine: human dictate *logic* (units, signs, conserved qty, valid scheme sets, references, acceptance criteria); AI fill *body*; every declared invariant checked **mechanically** so violations fail LOUDLY. Each gate = **tripwire, not proof**, ships synthetic-violation self-test (provably non-vacuous). NON-NEGOTIABLE harness (extend, never weaken; budgets/TODOs shrink only):
- **Spec-first physics contracts**: every physics scheme module declares `__physics_contract__` (units/signs/conserves/differentiable/reference/idealized_test); `tests/test_physics_contracts.py` partitions all `*/physics/*.py` into EXCLUDED / CONTRACT_TODO(shrink-only) / annotated — NEW physics file must ship contract or be classified. Author contract + acceptance test BEFORE body.
- **Atmospheric parameterization edits are high-risk**: run codex adversarial review, #477 truth-tier tendency validators, conservation tests, equilibrium-SCM-RCE realism harness before merge. NEW scheme must pass RCE realism gate or enter its shrink-only TODO partition with a reason.
- **CI ratchets** (AST + self-test, via shared `tests/_ratchet_audit.py`): `test_no_hardcoded_constants` (constants only from `legoesm.constants`), `test_no_saturation_reimpl` (saturation only from `thermo`), `test_dispatch_hardening` (no scheme guard silently deleted), `test_validate_strict_coverage` (no scheme field skips fail-early validation). Escape: real `# const-ok:`/`# satcurve-ok:` comment.
- **Local hooks** (`.claude/hooks/`, wired in `settings.json`): PreToolUse blocks edit adding banned constant/saturation prefactor to `.py` (fail-open); Stop reminds (once/session) to run mandatory codex review on uncommitted numerics/physics. CI remain authoritative (`Bash` heredoc bypass hook, not CI).
- **Best coding practices = use existing system**, not parallel one: ruff/mypy, import-linter (`alerting="error"`), inline-import budgets, pre-impl grep + shared utilities (no re-derivation), every new `.py` gets unit test, slopbuster sweeps, iterate-with-codex on substantial changes.

## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `.venv/bin/python -m pytest tests/`
- Sci tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
- Atm matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py`
- Ocean matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py`
- AMIP: `.venv/bin/python scripts/run/run_amip.py`
- Dycore progression: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py` (SW -> hydrostatic -> non-hydrostatic ladder; `--only sw|hydro|nh`). Old `tests/validation/run_dycore_progression_suite.py` superseded — its per-case child scripts removed.
- GPU/MPI scaling: `.venv/bin/python scripts/bench/run_levante_gpu_scaling.py --grid cubed-sphere --mode strong` (`docs/performance/REAL_HARDWARE_SCALING.md`)
- Scripts reorganized into buckets: `scripts/{run,matrix,bench,plot,validate,data,experiment,cluster}/`; debug in `scripts/tmp/`. See `scripts/README.md` + `## File Layout` below.
- MPI tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/`
- MPI diff: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_mpi_differentiability.py`

## File Layout (audit — enforce on EVERY new file; no random files)
- **New scripts go in correct `scripts/` bucket — NEVER `scripts/` root or repo root.** Buckets: `run/` (prod drivers), `matrix/` (test-matrix registries), `bench/` (perf/profiling/scaling), `plot/` (plot/replot/regen), `validate/` (non-matrix validators/verifiers/conservation checks), `data/` (download/build/prepare forcing+IC), `experiment/` (init/reproduce/templates/machine-detect/fetch), `cluster/` (SLURM `.sbatch` job wrappers, e.g. `cluster/omip_nemo/`). Pick bucket by what script DOES. New bucket need real category, not dumping ground. See `scripts/README.md`.
- **Debug / throwaway / one-off → `scripts/tmp/` ONLY** (eventually deleted): `_*`-prefixed probes, `diag_*`/`diagnose_*`, per-iteration scratch. Never at `scripts/` root. `_probe_*.py` gitignored.
- **No new files dumped at repo root.** Root keeps ONLY: `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `FEDERATION.md`, `project_status.md` (generated), `pyproject.toml`/lockfile/dotfiles. Everything else has a home.
- **No `.md` notes accumulating at repo root → `docs/`.** Dev-notes, change logs, faithfulness/audit trackers (`*_faithful.md`, `*_checks.md`, review logs) live under `docs/`. Reference by BASENAME so code comments survive the move.
- **No runtime outputs in git.** `diagnostics/`, `logs/`, `**/logs/`, `results/`, `checkpoints/`, `output/`, `*.zarr`/`*.nc`, root `*.png`/`*.pdf`/`*.svg` gitignored. Visual-regression baselines stay in LOCAL working copies, regenerated on demand — not tracked. Never `git add -f` a runtime artifact. EXCEPTION: `.physics-validator/` is tracked on purpose — review logs, probes and banked oracle results cited as evidence by docs, tests and validation scripts (user 2026-10-24); only its error captures are ignored.
- **Source stays under `packages/<pkg>/legoesm/`** (federation namespace). New subpackage → update `tests/test_federation_plan.py` same PR.
- **Do NOT "flatten" `packages/<pkg>/legoesm/<subpkg>/` → `packages/<pkg>/` — double name is LOAD-BEARING, not redundant.** Editable installs (dev/test workflow) require on-disk path == import path, and hatchling REFUSES prefix-*adding* `sources` remap in dev mode (`ValueError: Dev mode installations are unsupported when any path rewrite in the sources option changes a prefix rather than removes it`; editables#20). So `packages/atmosphere/legoesm/atmosphere/` is only layout where `import legoesm.atmosphere` resolve under `pip install -e`. Flattening to keep `legoesm.*` build wheel but break editable dev; flattening to bare `import atmosphere` mean rewriting ~23k import sites + dropping namespace. Verified 2026-07-13 — don't re-attempt. `src/legoesm/` is SANCTIONED root meta-package (the `legoesm.*` TOP-LEVEL modules: `constants`, `config`, `cli`, `registry`, `dycore_factory`, …), same load-bearing reason — NOT stray `src/` to remove. "Never add source at `src/`" mean don't create NEW `src/` trees for component code; existing `src/legoesm/` stays.
- **Tests mirror package tree under `tests/`** (`tests/<component>/<tier>/...`). Curated dycore regressions in `tests/atmosphere/dycore/regression/`. No new `test_*.py` at repo root.
- Staging: explicit pathspecs, NEVER `git add .`/`-A` — catch stray scratch + concurrent-session files.

## Bug Triage
- Instability: CFL, boundary, metric, halo, pressure-gradient, diffusion, dtype.
- Conservation drift: flux form, weights, reductions, state updates before fixers.
- Differentiability: control flow, shape changes, side effects, checkpointing, nondiff branches.
- Perf regression: retrace, host callbacks, scatters, sharding, Python loops.
- Cross-backend: dtype, x64, unsupported kernels, comm semantics.

## Hygiene/Imports/Tests (audit)
- Every new `.py` ≥1 direct unit test importing+exercising leaf module (tendencies for physics schemes — not just integration via factory). No `__init__.py` re-export / `supported_matrix.py` / factory dispatch add without same-PR test. Block PRs growing untested-LIVE count.
- New config dispatch (Literal + factory in `integration.py`): test via public config.
- **New user-tunable config field → same-PR CLI flag in every affected run script.** Rule applies to `ExperimentConfig`/`DycoreConfig`/`OutputConfig` fields in `run_amip.py`, `OceanExperimentConfig`/`KPPConfig`/`VerticalMixingConfig` in `run_omip.py`, `MultiLayerLandConfig` sub-configs in `run_lmip.py`. "User-tunable" = any float/int/str/bool field user would reasonably change for sensitivity run (excludes: internal-only flags such as `debug_precision`/`diagnostics_perf_mode`, unimplemented stubs like `carbon_cycle` in AMIP, parallel-milestone-gated fields like `distributed_mode=spmd`). Each new flag also needs: (1) wiring in `build_config_from_args` / equivalent config-builder, (2) round-trip test in corresponding `tests/unit/test_run_*_cli.py`, (3) `validate_strict` membership check if it is a scheme Literal. FLOAT tunables with `__param_spec__` need no flag: all four MIP drivers take `--config` (dests, choices-validated) + `--params` (registry-qualified `scheme_key.field`, bounds-validated; atm via flattened-scalar map). Reachability machine-audited by `tests/unit/test_params_reachability_audit.py` against shrink-only `_params_reachability_baseline.py` (new spec'd tunable unreachable in its component's drivers go red) — the #691 "CLI flag gaps" debt closure.
- Removing module: also remove `__init__.py` re-export, `supported_matrix.py` entry, dispatch, test file, `__pycache__`.
- No deprecated backward-compat wrappers — update call sites. No thin dispatch-only wrappers (`X_utils.py` re-exporting `X.py`) — inline/factor. Real branching across callers (`land/stomata_utils.py`) legit. Grid variants legit when genuinely different numerics; indexing-only copy-paste forbidden.
- **No top-level cross-package imports from `core/` to `runtime/`/`parallel/`/`driver/`/`training/`/`experiments/`.** Why: `from legoesm.runtime.backend import ...` at top of `core/precision.py` triggered `runtime/__init__.py` → `runtime.precision` → `core.precision` mid-init, breaking isolated pytest. Use function-scope deferred imports.
- **No import of private (`_`-prefixed) symbols across modules.** Promote (drop underscore + `__init__.py` re-export) or factor public wrapper. Mutable singletons (`grids.halo._halo_backend`/`_mpi_topology`, `cubesphere_exchange._spmd_mesh`): use accessors (`get_halo_backend`/`get_mpi_topology`/`get_spmd_mesh`), never import global. Audit: `grep -rE "from legoesm\.[^ ]+ import [^,]*\b_[a-z]" src/legoesm/ packages/ | grep -v " as _"` = 0 (line-grep miss multi-line/function-scope/`_UPPER` imports); CI ratchet `tests/test_no_private_cross_imports.py` (AST, incl. function-scope; dunder-exempt; shrink-only allowlist EMPTY since 2026-06-10 — keep it empty).
- **Test-only modules MUST be acknowledged.** Not wired into factory/`__init__.py`/prod driver: (a) wire same PR, (b) move to `_future/` + docstring + xfail/skip, or (c) delete.
- **Never commit `docs/references/`.** Local research PDFs/extracts. Cite by filename/DOI. Notes elsewhere (`docs/ocean/experiments/`). Staging: explicit paths, never `git add .`/`-A`.
- Slopbuster periodic: `/slopbuster audit all` or `/slopbuster review`.
- High-priority untested LIVE (touch any → add test same PR): `ocean/experiments/global_overturning.py`, `ocean/physics/{bottom_drag,convection,surface_forcing,vertical_mixing}/output.py`, `ocean/physics/convection/enhanced_diffusion.py`, `ocean/physics/vertical_mixing/{k_profiles,mpas_integration}.py`. RESOLVED 2026-05-29 (direct tests added): `coupler/surface_energy.py`, `ocean/dynamics/barotropic_common.py`, `atmosphere/dynamics/sfno_pe.py`, `atmosphere/dynamics/tracer_transport_mpas.py`; `atmosphere/physics/convection/_triggers.py` + `timestepping/tridiagonal.py` already covered.

## Constants/Params (audit)
- **All physical constants in `src/legoesm/constants.py`.** New constant (T, ρ, c, L, k, μ, EOS coeff, Schmidt#, R_earth) MUST be added BEFORE use. No constants in `config.py`/fn bodies/test fixtures/plotters/notebooks even with `# = constants.X` comment.
- **`getattr(..., "X", <literal>)` fallbacks count as hardcoded.** Use `getattr(grid, "radius", constants.R_earth)`. Same `setattr`/`dict.get`/`kwargs.get`.
- **No hardcoded physical constants in fn sigs/bodies in `src/legoesm/`.** No `def f(g=9.80616, ...)` → `g: float = constants.g` or config NamedTuple. NamedTuple defaults SHOULD reference constants. Tunable scheme params (sigmoid sharpness, τ, drag) stay in config NamedTuple.
- **No hardcoded tunable params in physics bodies.** Sigmoid sharpness, τ, Louis coeffs, KPP epsilon, emissivity, drag → scheme `*Config` NamedTuple. Exempt: safety floors (`eps=1e-30`), math constants (`0.5`, `2.0`), category lookup tables (PFT `L_v` in `surface_params.py`) with doc.
- **No `273.15` for C↔K in prod.** Use `constants.T_freeze`. `T_freeze_ocean=271.35 K` in `ocean/eos.py` = only intentional exception.
- **Tests+scripts+plotters same rule.** `from legoesm import constants`. No `9.80616`, `7.292e-5`, `6.371e6` literals.
- **Sigmoid sharpness/transition widths in JAX hot loops forbidden as magic numbers.** Inside `scan_step`/`cond`: fn kwarg with doc default OR scheme `*Config` field. Ex: `compute_moist_adiabat(lcl_sigmoid_width_pa=100.0)`, `PlumeConfig.active_sigmoid_sharpness=1e4`, DM95 `transition_width_frac=0.1` on `dm95_taper`/`dm95_taper_scalar`/`_triad_taper`.
- **Saturation re-impl in forcing modules forbidden.** Use `legoesm.thermo.saturation_mixing_ratio`/`saturation_vapor_pressure`.

### Parameter hygiene — declare per category at file top + machine-readable tunable/fixed split (gated)
Two CI tripwires enforce this (extend, never weaken; baselines shrink-only): `tests/test_no_inline_physics_coeffs.py` (+`_inline_coeff_baseline.py`) and `tests/test_param_specs.py` (+`_param_spec_baseline.py`).
- **No inline empirical coefficient in physics function body OR signature default.** Across `*/physics/*`, `land/`, `ice/`, `coupler/`: every float literal (and int `|v|>16`) inside function scope flagged. Move to (a) scheme `*Config` NamedTuple field if tunable/scheme-defining, (b) module-level `_UPPER_SNAKE` constant/table block (published fits/tables: Sutherland, Hall-Pruppacher, Morel-Berthon, Jerlov, KK2000) with provenance comment if fixed published constant, or (c) real `# coeff-ok: <reason>` (reason REQUIRED) for genuine numerics one-off. Exempt: math {0,0.25,0.5,1,2,3,4,6}, exact conversions, `|v|<=1e-6` floors, `|v|>=1e20` guards, subscript indices, Pow exponents, small ints (indices/counts).
- **Per-file layout (consistent placement):** docstring → imports → `__physics_contract__` → `__param_spec__` → fixed `_UPPER_SNAKE` constant/table blocks (under `# --- <category> (<reference>) ---` comments) → `*Config` NamedTuple(s) with fields grouped by same category comments (snake_case + unit suffixes) → functions whose bodies read ONLY `cfg.<field>` / `constants.*` / exempt math. Kwarg-default literals → reference module constant or config field (Name in signature, not literal).
- **Every physics scheme `*Config` declares `__param_spec__`** (module-level pure dict literal next to NamedTuple). Per float field: `units`, `bounds (lo,hi)`, `tunable_tier`, `transform` (sigmoid/softplus/none), `category`, `reference`, `shape` (None or dim key like `n_pft` for variable-size array params), optional `legacy_name`. NEW config module must ship spec or be classified in `PARAM_SPEC_TODO` (shrink-only). Inclusion computed: only `:float`-annotated fields spec-eligible.
- **Tunable/fixed split (the continuum), classify SAME way every time:**
  - `tunable_tier 1` (**core**) = params already trained in practice / well-posed (surface albedos, ice strength `P_star`, gray optical depths, bulk exchange `C_H`/`C_E`).
  - `tunable_tier 2` (**extended**) = clear closure knobs with literature bounds (relaxation timescales τ, entrainment/drag/autoconv rate coefficients, thresholds, emissivity, roughness).
  - `tunable_tier 0` / `excluded` (with reason string) = NOT trainable: numerics floors/caps/regularisers, smoothing widths, measurement conventions (e.g. MOST 10 m), and anything **iteration-coupled** (mEVP `alpha`/`beta`, EVP `T_evp` couple to subcycle count).
  - Collector select tiers `1..N` (`build_trainable_params(config, tier="core"/"extended"/"aggressive", include=, exclude=)`); flip a param's status with 1-line `tunable_tier` edit. See [[param-hygiene-spec-effort]].
- **Loop-iteration COUNTS are never config/trainable** → module constant (e.g. `_N_EVP_DEFAULT = 120`), not config field, not kwarg-default literal. Structurally guaranteed: ints not spec-eligible, so iteration count can never reach trainable collector. See [[loop-counts-never-trainable]].
- **A tunable closure whose default is a `constants.X` reference** (e.g. `S_ice_new = constants.S_ice_bulk_default`) is *eligible* (may be `__param_spec__` param with explicit bounds + tier) though not *required* (AST gate won't force it). Expose genuine calibratable closures; keep environmental references (ocean salinity) fixed/excluded.
- **Trained values inject via config pytree, not new signatures:** `params.to_overrides()` → `legoesm.core.param_overrides.apply_param_overrides(physics_config, overrides)` (`NamedTuple._replace`) INSIDE the loss so leaves TRACED (SegmentForcing doctrine); production keeps static Python-float leaves (constant-folded, no retrace). Register newly-specced module in `param_collector.SPEC_MODULES` (drift-tested).

## Naming
- Surface T = `T_sfc` everywhere. No new `T_surface`/`Ts`.
- Driver/config schema field names match runtime field. New tunable: same name in `driver/config.py`, scheme config NamedTuple, YAML schema (consistent `hyperdiff_coeff`).
- **Same name + different units = bug magnet (audit).** Unit hint (`_C`, `_K`, `_s`, `_days`, `_m`, `_km`): use everywhere. New C-vs-K args MUST carry `_C`/`_K`. Config fields sharing base (`tau_*`, `T_*`, `c_*`, `C_*`) MUST have consistent unit suffixes OR distinct names.
- **snake_case all NamedTuple fields**, even capitalized symbols (CAPE, CIN, MSE, TKE). `SBMConfig.CAPE_threshold` → `cape_threshold`.
- **Renaming a tunable `__param_spec__` field = rename field AND its spec key together + add `legacy_name: <old>`.** Spec key must equal float field name (`test_param_specs`), so half-rename go red; `legacy_name` document prior name for any externally-saved tuned JSON. (2026-06-22: `CarbonConfig.T_opt→T_opt_C`/`T_width→T_width_C`, `StomataConfig.T_opt_jarvis→T_opt_jarvis_C`/`T_range_jarvis→T_range_jarvis_C` — Celsius fields gaining `_C` suffix.)
- **`tau_`/`τ` prefix valid for wind/wave STRESS [Pa], not timescales only.** Ocean wind stress and GWD launch stress (`tau_0`, `tau_max`) = standard GFD τ notation — do NOT auto-rename to `stress_*`. Real anti-pattern = MIXED units under one `tau_*` family inside single config without unit suffixes; fix by suffixing timescale members (`_s`), not stresses.

### Open naming debt
- `T_sfc`(368)/`T_surface`(59)/`Ts`(~6): coupler+`land/{multilayer_land,snow_budget,stomata_utils,slab_land}.py`, `ice/sea_ice.py`, `coupler/{accumulator,lake/two_layer_lake}.py` still `T_surface`; 3 files MIX BOTH — `driver/coupled_esm_driver.py`, `ice/sea_ice.py`, `driver/earth_system_driver.py`. Unify cleanup PR.
- `nlev`(4119)/`n_levels`(199)/`nz`(33): `nlev` dominates. Cleanup PR.
- `tau_relax`: RESOLVED 2026-05-29 → `KuoConfig.tau_relax_s`[s], `PhillipsTwoLayerConfig.tau_relax_days`[days] (matches `backscatter.tau_relax_days`). Keep unit suffix on any new relaxation-timescale field.
- `C_water`/`c_water`: RESOLVED 2026-05-29 → `SoilThermalConfig.C_water_vol`[J/m³/K], `LakeConfig.c_water_mass`[J/kg/K]. Keep `_vol`/`_mass` on new heat-capacity fields.
- `n_layers` overloaded: soil=`n_soil_layers`, ML=`n_hidden_layers`, reserve `n_layers` for atm/ocean vert.

## Dispatch (audit)
- **Every `scheme="..."` factory MUST `raise ValueError` on unknown.** Silent `else: <default>` mask typos+dead branches. Historical: `cloud_fraction.compute_cloud_properties` ran sundqvist on typo; `land/carbon/carbon_cycle.py:443` zero CO2; `ocean/biogeochemistry/carbon_cycle.py:108,209` silently disabled BGC; MPAS PV typos → enstrophy in `{compressible_euler_mpas,primitive_eq_mpas,shallow_water_mpas,ocean_pe_mpas}.py`; bulk-scheme typos → constant in `coupler.py:204`, `slab_land.py:156`, `multilayer_land.py:212`, `two_layer_lake.py:66`, `bulk_formulas.py:68`; `io/restart.py:232` silently wrote npz. HARDENED 2026-05-29 (now `raise ValueError`, validated at fn entry on static config): `carbon_cycle.py:step_carbon`, `coupler.py:ocean_tile_response`, `ice/sea_ice.py:_bulk_flux_dispatch`. HARDENED later — all now raise: `slab_land.py`, `multilayer_land.py`, `coupler/lake/two_layer_lake.py`, `bulk_formulas.py` (via `core/bulk_flux.py::validate_bulk_scheme`). HARDENED 2026-06-22 (atm MPAS `pv_scheme` typo→energy + thompson `snow_scheme` typo→bulk): `compressible_euler_mpas.py`, `primitive_eq_mpas.py`, `shallow_water_mpas.py` (now `energy`/`enstrophy` else-`raise`), `microphysics/thompson.py` (fn-entry `snow_scheme` guard). These are nested scheme-Config fields, NOT `ExperimentConfig` literals → factory/fn-entry raise is the defense, not `validate_strict`.
- Dispatch in `lax.fori_loop`/`lax.cond` (`coupler/bulk_flux.py:222`): validate at fn entry on static Python val, not traced body.
- **Nested scheme-Config dispatch raises too.** A `scheme`/`pv_scheme`/`snow_scheme`-style field in leaf `*Config` NamedTuple (NOT `ExperimentConfig` literal, so `validate_strict` never see it) MUST still `raise ValueError` on unknown — validated at fn/factory entry on static config value (matches `ocean_pe_mpas` sibling). Bare `else: <default>` here silently run different physics on typo. Lock each new guard in `tests/test_dispatch_hardening.py::BASELINE_DISPATCHERS` (grow-only) so it can't be silently deleted.
- Add membership-set assertions in `ExperimentConfig.validate_strict` for new scheme literals. RESOLVED 2026-06-09: `convection`/`turbulence`/`gravity_wave_drag` now HAVE validate_strict membership checks (`config.py:430-455`). Guarded + enforced going forward by `tests/test_validate_strict_coverage.py` (every scheme-like config field must be membership-validated or in its `KNOWN_UNVALIDATED` loader-validated allow-list; removing guard or adding unvalidated scheme field → red). Companion factory-level guard: `tests/test_dispatch_hardening.py` (existing unknown-scheme `raise` may not be silently deleted; 76-entry grow-only baseline).

## Common Mistakes
**NamedTuple fields**: verify actual field. `PhysicsOutput.precip` not `precipitation`. `hasattr` guard silently degrade. Adding field to `SegmentCarry`: update every call site. `grep -rn "SegmentCarry(" --include="*.py"`.

**Land/face masks (latlon C-grid)**:
- Never `state._replace(land_mask=...)` on `LatLonCGridOceanState` without updating `u_mask`+`v_mask`. Stale face masks → mass flux through walls → silent leak.
- Preferred: `land_mask_override` to `rest_state_latlon_cgrid_ocean()` at construction.
- Post-construction: `replace_land_mask(state, new_mask)` from `init_latlon_cgrid.py` — atomic update 3 masks.
- `_assert_runtime_invariants` (gated `enable_runtime_checks`) catches inconsistencies.

**JIT/compilation**:
- Never build closures in training loops. `build_segment_fn` create new fn per call → inside `for epoch`/`_loss_fn` recompile each iter. Build once outside; pass changing vals as args.
- Helper fns inside `lax.scan` body: Python defs in `_single_step` recreated each trace. Move to module scope.
- Dead code from iteration: when refactoring, grep for vars assigned never used.

**SegmentCarry**: canonical hot-loop state. Adding field cross-cutting: NamedTuple def, `pack_carry`, `unpack_carry` docstring, per-step Python ref loop in `test_compiled_segments.py`, `test_scale_tpu_compat.py`, `test_scale_jit_health.py`, direct `SegmentCarry(...)` in validation tests. New diagnostic fields (`max_cfl`) reset to zero at segment start, not accumulated.

## Assets
Specialized agents in `.claude/agents/` for dycore, validation, differentiability, physics, land/ice, scalability.

## Response Style

### RULE -3 — BREVITY IS THE DEFAULT, LENGTH IS OPT-IN (2026-08-18, FIFTH callout)
User: *"be succinct and clear - make this a default."* Rules -2, -1 and 0 all
framed brevity as a CAP to check before sending, and I wrote to the cap every
time. Inverted:

**DEFAULT REPLY = <=3 LINES. NO HEADER. NO TABLE. NO BULLETS.**

Length is earned, not assumed. It is unlocked ONLY by an explicit ask
("report", "walkthrough", "long version", "what is left") or by numbers the
user will genuinely compare (a table of >=3 rows AND >=2 columns). Bold
headers, per-item verdicts, before/after lists and caveat lines are
REQUEST-ONLY formats.

Mechanical check: write the reply, delete everything after line 3, and send it
if it still answers. If it does not, the content wanted a table — keep the
table and delete the prose.

Background-task notifications get NO reply at all unless a result is ready, a
claim was refuted, or a decision is needed.

### RULE -2 — MECHANICAL LENGTH LIMITS (2026-08-14, THIRD callout)
User: *"be more succinct and clearer - make this a rule."* RULE -1 and RULE 0
below already said this and I still shipped multi-section replies with tables
and bold headers on routine results. Prose rules did not work, so these are
COUNTABLE. Check them before sending.

- **5 lines of prose, hard.** Not 5 sentences — 5 rendered lines.
- **ONE bold header per reply, or none.** Multiple `**Headers**` = a report,
  and reports are only for when one was asked for.
- **A table needs >=3 rows AND >=2 columns of real data.** Two numbers go in a
  sentence. A table of one comparison is decoration.
- **No line explaining method, discipline, or what a control proved.** The
  result only. "Pre-registered X, got Y" is one clause, not a paragraph.
- **No restating a caveat already in the commit.** The commit is the record.
- Status/job IDs: one line total, at the end, no formatting.

If the content genuinely needs more, say "long version?" and stop.

### RULE -1 — THE CAP APPLIES TO GOOD NEWS AND BAD NEWS ALIKE (2026-08-12)
User, again, after a session of correct-but-long replies: *"be succinct and
clear."* The rule below was being followed for status and ignored for
findings — a retraction, a root cause, or a self-caught instrument bug is
NOT a licence to write six paragraphs. Length is not proof of rigor.

- **A finding is ONE line: the corrected fact.** Not the discovery story,
  not what it means for three other workstreams, not a list of what still
  stands. "X was wrong; the real number is Y" and stop.
- **Never re-explain a thing already said once in the same reply.** If a
  number appears in the verdict, it does not reappear in the evidence.
- **Cut every sentence that exists to show diligence.** "I checked", "the
  control caught it", "this is the Nth time" — delete. The tool calls are
  visible; the user is not grading effort.
- **Detail goes in the commit, the memory file, or the peer message —
  never the reply.** That is what those artifacts are for.
- Default reply: **3-5 bullets, under ~60 words.** A long reply must be
  ASKED for ("report", "walkthrough", "what is left").
**STRICT RULE, EVERY SESSION, EVERY MODEL (Opus included): be succinct AND
clear.** Not style preference, not default that decay over long session.
Succinct = answer first, ~60 words, bullets not paragraphs (RULE 0). Clear =
plain words a colleague outside this repo can act on, no bare identifiers (RULE
-1). Both must hold; reply failing either one = failed reply.

### RULE -1 — CLARITY IS THE HARD RULE. IF THE USER CANNOT FOLLOW IT, IT FAILED.
User, 2026-08-12 (and 2026-08-07, 2026-08-11 — same complaint every time):
*"Ensure you are clearer — I have no clue what you are saying most of the
time."* Outranks brevity: short reply nobody understand worse than no reply.
Brevity already being followed when this said; defect is UNEXPLAINED INTERNAL
DETAIL, not length.

MECHANICAL TEST, apply to every sentence before sending: could a colleague who
know climate modelling but never opened this repo act on it? Need file name,
function name, job id, or flag to make sense → REWRITE IT.

- Say the THING, not the SYMBOL. "the model runs radiation 18 times more often
  than intended", not "split_rad=False makes rad_update_interval_steps inert".
- NO identifiers in a reply: no file:line, no function names, no config keys, no
  job ids, no PR numbers, unless user asked for that exact thing. They belong in
  commit message. Number user should act on is fine.
- One idea per line. Line has clause explaining clause → split it.
- State CONSEQUENCE first, cause second, stop. Not mechanism, not how it was
  found, not who found it.
- Never write sentence whose subject is piece of code. Subject = model, run,
  campaign, number, or user's decision.
- Something wrong → lead with what now false and what to do. Not narrative of
  discovery.
- Question to user = numbered options in plain words, with your pick.

FAILURE PATTERN, all three callouts: long autonomous stretches. Each status
reply drifted back into repo-internal vocabulary ("the pin", "the waiver", "the
factory", "the skeleton") meaning nothing outside this session. Re-read this
section whenever session run long, and before every status report.

### RULE 0 — HARD CAP ~60 WORDS. ANSWER FIRST. STANDING ORDER, ALL SESSIONS.
Five callouts in two days (2026-08-07/08). Not style preference — a cap. Lead
with result. 3-5 bullets. Stop. Job IDs, caveats, file:line and reasoning go in
COMMIT, never the reply. Retraction = ONE sentence plus corrected fact. Long
replies only when report explicitly requested.

User, 2026-08-07, after repeated callouts in single session ("you are too
verbose", "I have no idea what you are saying", "just laser-focused summary"):
**"Make being succinct, clear and to the point a strict rule for all future
sessions."** Outranks every other formatting instinct.

Mechanical test before sending — line fail → cut it:
- Would user act differently without this line? No → delete.
- Is it method, process, or what I tried? → delete. Tool calls visible.
- Is it caveat nobody would act on? → delete (put in commit).
- Is it re-explaining something already said once? → delete.

Shape: **verdict first**, then only evidence that change the verdict, then ONE
question with numbered options if decision needed. Default length few lines.
Long reply must earn it by being asked for (report, walkthrough, per-phase
notes).

Jargon banned unless sentence also say what it mean in plain words. Say "how
much communication the split costs", not "the ppermute round count".

### PLAIN LANGUAGE FIRST — the rule that finally worked (2026-08-07)
THIRD callout: *"I really don't understand when you talk to me. Things are
incredibly unclear."* Brevity alone did NOT fix it — earlier rules were being
followed. Real defect = **unexplained jargon** and leaving user to infer the
decision. What worked, now required shape:

- **Write for colleague, not reviewer.** Short sentences. Ordinary words.
- **NO unexplained domain jargon.** Terms like *ppermute, max_degree, halo fill,
  production-exact, VJP, chromatic bound* banned unless sentence also say what
  they mean in plain words. Prefer plain phrase outright: "how much
  communication the split costs", not "the round count of the comm graph".
- **Use short LABELLED blocks**, bolded, 1-3 lines each:
  **What I built / What broke / What I was wrong about / What's safe /
  What I need from you.** Labels do the navigating so user not.
- **End with ONE explicit question and numbered options** when decision needed.
  Say which you would pick and single deciding factor.
- **Asked to compare options → answer the comparison** — what each buy, what it
  cost, and question that decide it. No re-describing options.
- **Own errors in one plain sentence, at top.** "I was wrong earlier: X is not
  guaranteed." No narration of how discovered.
- **No tables, no file:line, no job ids in reply** unless asked. Those go in
  commit message. Reply is for the decision.

### Earlier callouts (same session) — still in force
**SECOND callout (2026-08-06), because rule below was written then ignored:
_"stop being verbose. It is really hard to understand. be more direct, to the
point, clear about issues. Bullets summarizing."_ Plus: _"aren't you using
caveman?"_ — terse mode was ACTIVE and I still wrote essays.**
- **BULLETS BY DEFAULT.** Prose paragraphs = failure mode. One line per fact.
- **Lead with the issue.** Not how it was found.
- **Delete every sentence not changing what user does next.**
- **Never re-explain caveat already stated once.**
- **No near-miss stories.** "I almost got X wrong" is not finding. State
  corrected number, move on.
- Terse mode (caveman/ponytail) active → APPLIES TO WHOLE REPLY — including
  findings, status, caveats. Length not substitute for rigor.

User callout 2026-08-06 (earlier): *"you are quite unclear... provide more succinct,
clear summary, clear choice, do not make many but targeted and verified
assumptions."* Evidence reader has to assemble into conclusion is not a report.
Structure, this order, then stop:

1. **VERDICT first, <=2 lines.** What true / what happened. Never open with
   method, caveats, or narration of what was run.
2. **THE DECISION, if any: ONE recommendation.** Name option you would take and
   why, one line. Menu of options with balanced caveats push work back onto
   user — only list alternatives when they genuinely must choose, and even then
   say which you'd pick.
3. **EVIDENCE: only what change the verdict.** Decisive number, file:line, or
   measurement. Not everything checked.

- **AT MOST ONE unverified claim per response, explicitly labelled PLAUSIBLE.**
  Everything else verified before stated. No enumerating candidate causes — pick
  the one you tested, report it. Untested hypotheses = clutter reading as
  findings.
- **Do not narrate the process.** Tool calls already visible. Report outcome,
  not itinerary.
- **Table only for >=3 things compared on >=2 axes.** Otherwise a sentence.
- **Retract in one line, move on.** No re-litigating superseded claim.
- **Caveats: only those changing what user should DO.** Limitation nobody would
  act on belong in commit message, not reply.
- Numerics change -> state effect on stability, accuracy, conservation,
  differentiability. No guesses as facts.
- **No Read images** unless user asks; report path.

**TERSE. Caveman register (user, 2026-08-06: "You speak too much... no need to waste tokens").** Fragments OK. Drop articles/filler/hedging/pleasantries. No narrating what you about to do, no restating request, no re-explaining finding already stated. Prose is for FINDINGS, not for process.
- **ALWAYS end with findings summary** — table or bullets: what measured, the number, CONFIRMED vs PLAUSIBLE, what still open. That summary = deliverable; rest is scaffolding.
- Long verbatim tool output → quote only DECISIVE line (`N passed`, failing assert, peak value).
- Commits/PRs/code comments/security warnings stay full English.

# iterate-with-codex agent
1. Implement change
2. `/codex:adversarial-review --wait`
3. Parse output
4. Fix flagged
5. `/codex:review --wait` again
6. Issues remain → 4
7. Stop when clean or after 30 iter
exec
/usr/bin/bash -lc 'cat CLAUDE.md' in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
# legoESM Claude Memory

## RULE 0 — CONSERVATION OF MASS, ENERGY AND MOMENTUM. THE STRICTEST RULE.
User directive 2026-10-11: *"Mass, energy and momentum conservations should be
the strictest rules."* Outranks every other rule in this file, including speed,
brevity and convergence.

- **No change may create or destroy mass, energy or momentum** — no cap, clip,
  floor, limiter, fallback or "hold" that removes or adds a quantity without
  routing it to an explicit sink/source in the same budget. A capped water
  amount goes somewhere (dew, runoff, store); its latent heat goes somewhere.
- **Every physics/numerics change states its budget and proves it closes**:
  name the conserved quantities it touches, and add a test asserting
  `in - out - d(storage) = 0` to round-off (`JAX_ENABLE_X64=1`), shown to FAIL
  when the fix is reverted.
- **Exchanged fluxes carry the same value and sign at both ends of a coupling.**
- **A known non-conserving path (fallback, held column, clamp) is a defect**:
  report it with its measured size the moment it is found, never absorb it.
- Speed or convergence fixes that touch a solver must show the conserved
  totals are unchanged (to round-off) before and after.

## RULE 1 — CONCISE AND CLEAR. THE DEFAULT, EVERY REPLY.
User directive 2026-08-21: *"Be concise and clear — make this a default."*
Eleventh ask. Default, not mode.

**2026-09-08, NAMED AT OPUS 5 (and every Mythos/Fable-class model):
*"Force opus 5 to be concise and clear as default."*** Twelfth ask. The
long-session drift is the failure mode: replies start terse and grow back
into paragraphs after an hour of tool calls. So the cap is not a mood, it
is checked EVERY reply, and re-read whenever a session runs long.

GATE (RULE 2 demands one, so here it is, mechanical, self-applied before
send): count the rendered lines of the reply. **>5 lines, or any table
the user did not ask to compare, or any header in a reply under ~500
words → DELETE and rewrite.** Verdict first line; then at most 3 short
bullets; then stop. A status report on background jobs is ONE line per
job, no preamble. If it will not fit, the content wanted a table — keep
the table, delete the prose.

- **Verdict first line.** Then max 3 short bullets. Stop.
- **~60 words.** Longer only if report/walkthrough asked.
- **Plain words.** No file:line, function names, config keys, job ids, PR
  numbers unless asked — those go in commit. Say THING, not SYMBOL.
- **One idea per line.** Clause explaining clause = second bullet, or cut.
- **No headers, no tables, no bold-label blocks** in normal reply. Read as
  structure, same length. Table only if >=3 things on >=2 axes AND user must
  compare.
- **Own error in one sentence, at top.** Confession not exemption from
  brevity — narrative belong in commit, which already have it.
- **Decision needed = one question, numbered options, your pick named.**
- Cut any line user not act on. Tool calls already visible.

Full rationale + ten prior callouts: `## Response Style` at end of file. That
section disagree with this one → this one win.

## RULE 2 — RULES DO NOT ENFORCE THEMSELVES. CHECK THE GATE, NOT THE TEXT.
User 2026-08-21: *"how do we ensure the rules are not being violated?"* Honest
answer: MORE TEXT DOES NOT. Both rules broken in the session that prompted this
were ALREADY in this file. Adding emphasis to a rule that is being violated
changes nothing; adding a MECHANICAL GATE does.

**So: before claiming a rule was followed, name the gate that checked it.**
"I was careful" is not compliance. If no gate exists, say NO GATE out loud in
the status — do not let silence imply a check happened.

| rule | gate | where |
|---|---|---|
| no banned constants / saturation re-impl | PreToolUse hook + CI ratchet | `.claude/hooks/check_banned_literals.py`, `tests/test_no_hardcoded_constants.py` |
| codex review on uncommitted numerics | Stop hook, once/session, ADVISORY | `.claude/hooks/require_review_artifact.py` |
| scheme dispatch raise on unknown | CI ratchet, grow-only | `tests/test_dispatch_hardening.py` |
| every scheme field validated | CI ratchet | `tests/test_validate_strict_coverage.py` |
| physics contract per module | CI ratchet, shrink-only TODO | `tests/test_physics_contracts.py` |
| tunable reachable from a driver | CI audit, shrink-only baseline | `tests/unit/test_params_reachability_audit.py` |
| one parameter, one legal range | CI gate, shrink-only | `tests/unit/test_param_bounds_agree.py` |
| no private cross-module imports | CI ratchet, allowlist EMPTY | `tests/test_no_private_cross_imports.py` |
| no inline physics coefficients | CI ratchet + baseline | `tests/test_no_inline_physics_coeffs.py` |

**NO GATE EXISTS for these — they are honour-system, so state compliance
explicitly every time:**
- **DUAL review (codex AND GLM).** The Stop hook only sees UNCOMMITTED work and
  only mentions codex. A merge commits instantly, so it never fires on one —
  which is exactly how a 14-conflict merge shipped unreviewed on 2026-08-20.
- **NO UNASKED CHOICES / no hidden config choices.** Nothing can detect a
  plausible decision made silently.
- **Controlled comparison** (one variable, eval protocol held fixed).
- **Instrument validated before its number is quoted.**
- **Non-vacuity**: every new test shown to FAIL when its fix is reverted.

**When a gate does not exist, build one before adding more prose.** Two of the
rows above were written the day the rule they enforce was broken.

## Role
Senior JAX+ESM dev. Skeptical, verify-first. Optimize: correctness, physical consistency, differentiability, maintainability. Prefer `opusplan`/`opus` high effort for dycore/physics/parallel/debug. Fast mode off.

## Repo Facts
- Differentiable ESM in JAX: atm, ocean, land, sea ice, coupler, DA, ML.
- End-to-end `jax.grad` compat = goal. Never break autodiff/JIT/pytree.
- Mass conservation hard. Energy/momentum when scheme permits.
- Parallel entry: `ParallelRuntime.create()`.
- Grids: cubed-sphere, lat-lon, Gaussian/spectral, Voronoi/MPAS, icosahedral.

## Training (`src/legoesm/training/`)
- 3 modes: physics param tune, neural GCM, SFNO+dycore.
- All: `build_segment_fn(...).raw` (non-JIT, non-donating) inside `eqx.filter_value_and_grad`.
- `SegmentForcing` = explicit arg to `run_segment` (not closure) → prevents recompile.
- `TrainablePhysicsParams` wraps 8 params, Equinox module, sigmoid constraints.
- ERA5: `era5_to_state.py` lat-lon → grid, Zarr cache.
- Losses: `training/losses.py` imports `ml/loss.py`. No dup.
- **MPI AD**: `global_sum_mpi` / `batch_allreduce_mpi(op="sum")` / `global_sum_if_distributed` backward = allreduce SUM of the cotangent (correct for sums reused inside the computation; #1814). The reduction that PRODUCES the loss every rank evaluates and differentiates identically (normalised or not, e.g. a global mean) takes `global_sum_mpi(..., final_loss=True)` (identity backward). Forward mode (jvp) on both paths = allreduce SUM of the tangent (the default is a linear primitive whose transpose is also allreduce SUM). MPI halo: `_sendrecv_vjp` custom_vjp. `global_max_mpi`/`global_min_mpi` NOT diff — keep out of losses.
- **NO INERT PARAMETERS EVER (STRICT, user 2026-08-17).** Every leaf of trainable pytree must carry loss gradient. Mode-inactive params frozen OUT of trainable set (`_inactive_keys` pattern); first training step gates rest via `assert_no_inert` (`train_land_params_era5.py`) — zero-gradient leaf aborts run. Any new trainer/calibrator adopt both pieces; param "wired in but off" without freeze-out = defect.

## RULE 4 — SEARCH BEFORE YOU BUILD, AND PONYTAIL FULL IS THE DEFAULT.
User directive 2026-08-23, after a nearest-neighbour-on-the-sphere regridder
was written into a data script while the coupler ALREADY had a grid-remap
module (`coupler/grid_remap.py`) and core had another (`grids/regridding.py`):
*"always check if anything already exists before implementing"*.

- **Before writing ANY new function, class, script, or numerical method: grep
  the repo for an existing one.** Name in the status what was searched and what
  was found — "searched X, found nothing" is a claim and gets the grep pasted.
  This extends the existing pre-impl-search rule from numerics to EVERYTHING,
  and it applies to scripts and probes, not just packages.
- Found something close? EXTEND it or move it to the right home (generic
  methods live in the shared package, e.g. cross-grid machinery in the
  coupler) — never a second implementation "for now".
- NO GATE exists for this; it is honour-system, so compliance is stated
  explicitly every time (RULE 2).
- **Ponytail FULL is the default, every session** (user 2026-08-20 and again
  2026-08-23): laziest working solution, reuse over rebuild, delete before
  add, shortest diff that is actually correct. Not a mode to be asked for.

## RULE 5 — DIFFERENTIABILITY IS NEVER TRADED FOR ORACLE FIDELITY.
User directive 2026-10-20, STRICT, on the Kessler port (the smooth saturation
ramp misses the reference by up to 100 % and lets cloud water go slightly
negative; the reference's exact step would match it): *"keep the current
smoothing - differentiability cannot be traded off!"*

- **The default is always the differentiable form.** A switch in the
  reference whose branch depends on TRACED model state (a saturation step,
  a threshold `max`/`where` on a prognostic, a clip, a cutoff on the
  column's traced pressure) is ported as a smooth
  function whose sharpness/width is a config field; the hard limit is
  reachable only by setting that field, never as the default.
- **Out of scope (state-independent):** selections that do not depend on
  traced state and keep JAX gradients intact — fixed land/sea masks, domain boundaries, index
  picks, static scheme dispatch. Smoothing those changes the domain or
  breaks conservation; port them exactly.
- **Never propose moving a default to the hard limit to close an oracle gap.**
  Not as an ASK, not as a "pick". The gap is recorded instead.
- **Record the gap as a KNOWN DIFFERENCE** where the code lives (docstring)
  and in the PR body: what the reference does, what the port does, the
  measured size at the production setting, which field recovers the
  reference, and any invariant the smoothing breaks (e.g. how negative
  cloud water goes). Never loosen the tolerance.
- **Every such switch gets three tests:** one AT the hard limit,
  matching the reference to the oracle tolerance; one AT the smooth default
  asserting the recorded gap PER CASE (each fixture column's measured
  mismatch, stored with the test, to 1e-6 relative with a 1e-12 absolute
  floor); and a gradient test at the default (finite, non-zero derivative
  through the switch, `jax.test_util.check_grads`). Never a blanket xfail,
  never one worst-case bound. A change to the smooth form OR its default
  width re-measures and updates the stored gaps in the same PR, with the
  reason.
- RULE 3 is waived only when all four hold: the smooth default is kept for
  differentiability, the hard limit is reachable via its field and matches
  the reference, the gap is recorded as above, and all three tests exist and pass.
  Missing any one, RULE 3 applies.
- Applies to every oracle translation (CAM, IFS, WRF, ESCOMP, NEMO, FV3, ...).

## RULE 3 — A FIX WHOSE DEFAULT KEEPS THE BUG IS NOT A FIX. IT IS A BUG WITH A KNOB.
User directive 2026-08-22, on finding non-orographic gravity waves still being
launched at the SURFACE months after that exact defect was "fixed": *"Why does
gravity-wave drag now launch at the surface? This is absurd and we should never
ever make arbitrary choices like this without my supervision and agreement."*

WHAT ACTUALLY HAPPENED, because the shape repeats: #1394 measured the defect
(55% of the wave's momentum deposited below 1 km), added `hines_launch_p` as the
cure, and left its default at `0.0` — which the field's own comment documents as
"unset = legacy SURFACE launch". One campaign config set 70000. The production
config never did. Every production run since has carried the defect the fix was
written to remove, and the tropical ocean lost a third of its evaporation.
NOBODY CHOSE THIS. That is precisely why it survived.

STRICT, MECHANICAL:
1. **Landing a fix behind a default that preserves the old behaviour is
   FORBIDDEN unless the user is asked, in that PR, in one line: "default stays
   broken (X) or moves to the fixed value (Y)?"** No exceptions for "callers can
   override", "it is only a default", or ABI/positional-tuple preservation —
   those explain the mechanism, never the choice.
2. **A knob whose default is documented as "unset", "legacy", "off", or
   "0.0 = old behaviour" is a DEFECT REPORT, not a feature.** Grep for that
   wording before any campaign; each hit is an open bug in every run that does
   not set it.
3. **The fix's PR must edit the PRODUCTION config, not only a campaign deck.**
   The existing rule "new knob ships with a committed config that selects it"
   was satisfied by a side deck here and still shipped the bug — so the
   production config is now named explicitly.
4. **Before any production campaign, diff the run's RESOLVED config against the
   best previous run's resolved config and read every differing row out loud.**
   Not the deck, the resolved config. This regression was one row in such a
   diff and would have been caught before the GPU-hours, not after.
5. A default that disagrees with the best known configuration is reported as a
   FINDING the moment it is noticed — never silently carried into a run.

## ABSOLUTE RULE — NO UNASKED CHOICES. ASK FIRST, EVERY TIME.
User directive 2026-08-21, stated ABSOLUTE and repeated: *"Every choice such
as changing options should be checked with me before being changed"* and *"Do
not make random choices like this. This is a major error from Opus 5 in
particular."* Model near operationalization: unasked choice make every
downstream number non-comparable to what came before, and user cannot see it
inside diff full of other work.

THE FAILURE MODE, NAMED: fill gap in task with plausible decision instead of
question. Decision usually defensible — exactly why it go unchallenged and
unnoticed. Defensible choice made silently still unasked choice, and that the
error prohibited here — not bad taste, not wrong values. **Plausibility is not
authorisation.**

**2026-08-21 EXTENSION — HIDDEN CHOICES IN SIMULATION CONFIGURATION.** User:
*"add a strict rule not to take hidden choices such as for configurations of
simulations. We now have many bugs related to incorrect choices."* The rule
above bans choices YOU make. This bans choices the CODE makes for you and
nobody records. Same bug, different author.

A HIDDEN CHOICE is any place a scientific decision get made without appearing
in the run's configuration:
- **a default that disagrees with the run.** `g` defaulted to library constant
  while card pinned another (#1627); density fed into that pressure built on
  defaults too. Fix at last call site give DIFFERENTLY wrong answer.
- **a fallback that substitutes physics.** `use_multilayer_land: false` on MPAS
  ran NO land model — land T was nearest ocean SST minus lapse rate. Three
  waves of convection tuning against it.
- **a default that sits on domain geometry.** Lock-exchange front defaulted to
  prime meridian; channel spanned 0–0.576°, so front sat on WESTERN WALL. 1
  cold column of 66. Gates certified uniform box.
- **two bounds for one parameter.** `cloud_q_c_diagnostic` spec say 5e-5–1.5e-3,
  driver validate 1e-6–1e-3. Proposed value legal by one, illegal by other.
- **a lever nothing selects.** Neural drag flag no config set. Persistent-D
  behind env var, default off, while it is the leading fix for #1028.
  `cape_relaxation_sink` declared, read by nothing, A/B "in flight" against it.

MECHANICAL RULES, not judgement calls:
1. **Every scientific choice appear in the run's config, not in a default.**
   Building a run: name grid, scheme, resolution, timestep, physics switches
   EXPLICITLY even where default already correct. A default is not a record.
2. **A defaulted physical constant is a defect.** If caller can know run value,
   pass it. If it cannot, that is tracked debt with signature change named —
   never silent.
3. **Before running: enumerate what the config RESOLVED to, from resolved
   config and lane's code path** — not deck's comments. Prose is pointer, never
   citable fact. No prognostic state → say so out loud; usually that is finding.
4. **`false` on a component switch does not mean "simpler version".** May mean
   NOTHING. Check what lane fall back to.
5. **New knob ship with a committed config that selects it**, or it is not
   shipped. Same PR.
6. **Two declared ranges for one parameter = neither is the range.** Fix before
   using either.

STOP AND ASK before changing or selecting any of:
- a default value of a config field, CLI flag, or scheme selection
- which scheme/parameterization a driver or harness selects
- a tunable parameter's value, bounds, or tier
- what a case, deck, or experiment is forced with, and which cases exist at all
- a threshold, cadence, resolution, timestep, or analysis window
- which quantities are scored, and how they are weighted
- anything that changes model STATE a run carries — adding or removing a
  prognostic variable or tracer counts, EVEN when the added field is zero
- where data comes from (source, version, pinning) and whether a file is
  fetched, vendored, or generated
- whether a previously-tolerated condition becomes a hard error

Ask ONE line per item: current value -> proposed value, and why. Batch them; no
drip-feed. Work cannot proceed without answer → say so and stop. Never proceed
on assumption then flag after.

NO EXEMPTION FOR ANY OF THESE, all used as excuses:
- "the change was needed to make the requested thing work"
- "the old value was wrong / a bug"
- "a reviewer recommended it"
- "it is obviously an improvement"
- "it is only a default, callers can override"
- "it only adds a field, and the field is zero"

Fix bug; then ASK before default move. Fix and choice entangled → land fix,
leave choice pending.

NOT COVERED, proceed normally: what user asked in that message; pure additions
changing no existing behaviour; tests; comments; docs; error text.

**2026-10-16 EXTENSION — NO SIDE IMPLEMENTATION DECISIONS (STRICT).** User:
*"No side implementation decisions as strict Claude.md rule."* Approving a
change does NOT approve the decisions made while implementing it.

A SIDE DECISION = any value, behaviour, or method that changes what the code
computes, accepts, rejects, or reads, and is NOT dictated by the user's answer,
the oracle being ported, or existing repo code reused unchanged. Each one gets
ASKED (one line, current -> proposed, batched) before the code embodying it;
one found mid-implementation → stop and ask then. Classes, not exhaustive:
- tolerances and acceptance thresholds (incl. checks on input files, and the
  tolerance an oracle-match test certifies)
- guards, floors, caps, clamp vs extrapolate, and their fp32-vs-fp64 behaviour
- edge cases: zero/empty/missing input, absent file, old checkpoint
- mappings between representations (bands, grids, levels, units)
- algorithm and order: interpolation, aggregation, update/operation order
- where data or files live; names and defaults of user-facing switches
- which lanes/grids get it; error vs warning vs fallback there

PRECEDENCE: this OVERRIDES the "NOT COVERED" line above for pure additions and
tests whenever they carry a side decision. Still exempt: error MESSAGE wording,
comments, docs. Ponytail/minimal-diff governs how much code, never licenses
picking a value. "Formatting only", "minor", "internal", "implementation
detail", "numerically negligible" are NOT exemptions. Subagent briefs must
state this rule. Any item on an end-of-task UNASKED list is a violation of this
rule (target: empty), not a disclosure that discharges it. FAILURE 2026-10-16:
radiation stage 1 shipped a band-list check tolerance and an fp32 tiny-path
guard behaviour, both labelled "formatting only", neither asked.

**2026-11-03 — WHO ANSWERS: DETAIL CHOICES GO TO A VOTE, NOT THE USER.** User:
*"For such detailed question do not ask me but check with [GLM] and codex and
use the majority of votes. Just ask me big picture / important questions such
as on design or main directions."* Every side decision above is still never
made silently, but a DETAIL choice (tolerance, guard, edge case, test value,
cache key, refusal scope, error-vs-fallback) is put to GLM
(`mcp__zai__ask_glm`) and codex, batched in one prompt each, plus the author's
own vote; majority of the three decides. Record each choice with its three
votes in the commit/PR body. The USER is asked only about design, main
direction, scope, and anything they named as their call. Precedence: this
replaces "ask the user" for detail choices everywhere in this section and in
the STOP-AND-ASK list; for design/direction the user rule stands unchanged.

MECHANICAL CHECK, end of every task: print list of choices made, each one line,
marked ASKED (user), VOTED (GLM/codex/author majority, votes shown) or UNASKED. Empty UNASKED list = target. Anything on UNASKED list
offered for revert in same message. Worked example, five from 2026-08-21 that
prompted this rule: routing each case's droplet number into single-column arm;
allocating zero graupel field in every column running microphysics; which
campaign cases got registered and using ARM cumulus case as stand-in for another
campaign's regime; pinning forcing downloads to fixed versions and checksums;
turning missing variable name into hard error. Every one defensible. None asked.
That the error.

**2026-11-03 REFINEMENT — DETAIL DECISIONS GO TO A VOTE, NOT THE USER.** User,
repeated three times in one day: *"For detailed questions do not ask me but
check with him [GLM] and codex and use the majority of votes. Just ask me big
picture / important questions such as on design or main directions."*
- **Detail side decision** (names, tolerances, guards, edge cases, file
  locations, minor scope trims, which diagnostic next): author's pick + GLM vote
  + codex vote, majority of three wins. Never silent: record each voter's pick
  in the commit, and mark the item VOTED on the end-of-task list (VOTED counts
  as discharged, like ASKED).
- **Still ASK the user:** model design, main direction, production defaults and
  config changes (RULE 3), which experiments/compute to launch, data source,
  what is trained or scored, anything changing comparability of results.
- Unsure which class → big picture → ask.
- Subagent briefs must state this rule.

## Operating Mode
- **TERSE BY DEFAULT.** Drop articles/filler/pleasantries/hedging; fragments fine. Report `[thing] [state] [next]`, not prose. No restating what was just done, no feature tours, no explaining a simplification at more length than the code. Numbers/tables over narration. Full prose ONLY when asked for it (report/walkthrough), or for security warnings, irreversible-action confirmations, and multi-step sequences where fragments risk misread. Code/commits/PRs/docstrings: written normally.
- **DO WHAT WAS ASKED — DO NOT EXTRAPOLATE.** Deliver requested thing, then STOP and report. No inventing adjacent work because it seem useful: no unrequested helper/launcher/benchmark scripts, no extra files staged "for later", no speculative refactors, no scope widened from "fix X" to "also improve Y". Adjacent work look warranted → NAME it in one line, let user choose. Do not pre-build. Uncertain whether in scope? It not; ask. Over-delivery not helpfulness: cost review time, bury actual change, create artifacts nobody vetted.
- **Irreversible / outward-facing actions need explicit permission EVERY time**: rewriting published history (`push --force*`), amending/squashing pushed commits, deleting or overwriting files the user created, `git reset --hard`, closing/merging PRs, installing or upgrading system packages, reboots. Approval for one such action does NOT carry to the next. Propose, then wait.
- Nontrivial task: short plan before edit. Read nearby impl+tests first. Ambiguous numerics/physics/API: ask.
- Minimal diffs. No unrelated refactor in bug fix.
- **DUAL adversarial review is the DEFAULT for EVERY substantive change — TWO
  independent reviewers, never one** (user directive 2026-08-12, restated
  STRICT 2026-08-13, again 2026-08-20 as "make this the default behaviour").
  Author NEVER review own code. Route by who WROTE code — two reviewers always
  the other two:
  **Claude-authored → codex + GLM-5.2** (`mcp__zai__ask_glm`);
  **GLM-authored → codex + Claude**;
  **codex-authored → Claude + GLM-5.2**.
  Both BEFORE the PR, not after; report both verdicts in PR body and status
  line. Subagents doing implementation must be told to run BOTH — they default
  to codex only. Applies to measurement harnesses and probes too: instrument
  decide what we believe.
  Why two: they catch DIFFERENT classes. Codex find diff defects (broken
  contracts, vacuous tests that cannot fail, silent fallbacks, binding gates
  accepting wrong arm). GLM find MECHANISM defects (wrong objective, wrong
  regime, lever whose premise receipts already falsified — it retracted own
  top-ranked lever once measurement contradicted it).
  Reviewer disagreement = SIGNAL, not noise: name disputed point and the
  measurement that discriminates, run it if cheap, never average the two.
  A REVIEWER'S FINDING IS A HYPOTHESIS, NOT AN INSTRUCTION: measure before fix.
  2026-08-20, GLM predicted rounding error 1.4e-5 that would refuse valid
  restart files; measured over every column model builds it was 7.9e-8, so guard
  added for it was DELETED and measurement pinned instead — unmeasured finding
  buy knob that never bind.
  **"Substantive" includes MERGE CONFLICT RESOLUTION**, and rebases,
  cherry-picks, back-ports, probes, test fixtures. Conflict resolution =
  highest-risk case in list: someone chose between two versions of same code and
  diff not record what was discarded. FAILURE 2026-08-20: merged 17-day-old
  branch, hand-resolved 14 conflicts across 7 files including two independent
  fixes for one defect, shipped unreviewed until asked.
- **Codex adversarial review MANDATORY after any major code implementation/change.** Trigger: new module/feature, dycore/physics/parallel/ocean/land/ice/coupler/training edit, >~50 LOC, multi-file, or anything touching numerics/AD/JIT/pytree/conservation. Run **iterate-with-codex agent** loop below (`/codex:adversarial-review --wait` → fix flagged → `/codex:review --wait` → repeat until clean or 30 iter) BEFORE declaring done; report that review ran + verdict.
  **Review SUBAGENT dies (spend limit, API error) = NOT review waiver — codex
  CLI is separate binary with separate credentials, usually still reachable:
  `codex exec --sandbox read-only -C <repo> "<prompt>"` (`which codex`,
  `~/.codex/auth.json`). Try CLI directly before ever proceeding unreviewed. BOTH
  unavailable → say "UNREVIEWED" in every status until one succeed.**
  2026-07-26: subagent hit monthly spend limit, many iterations ran unreviewed
  while CLI worked fine whole time. Exempt: trivial/mechanical edits (typo,
  comment, rename, doc/markdown/`.tex`-only, single config value).
- **Pre-impl search mandatory**: before new fn/helper/class/operator/diagnostic/init/load/loss/numerical routine, grep `src/legoesm/` for similar names/docstrings/formulas in `thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`. State searched+found. Similar exists → extend/factor.
- **Shared utilities — never re-derive** (prod, scripts, validators, plotters, tests, notebooks, probes):
  - Constants: `from legoesm import constants` → `T_freeze`, `R_d`, `c_pd`, `L_v`, `R_v`, `epsilon`, `g`, `p_ref`, `kappa`, `sigma_sb`, `T_freeze_ocean`. No literals `273.15`/`287.0`/`1004.64`/`2.501e6`/`461.51`/`0.622`/`9.80616`/`6.371e6`/`7.292e-5`.
  - Saturation: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice`. No re-impl Tetens/Magnus/Clausius–Clapeyron (plotters incl). Why: re-derived `e_sat=611.2*exp(17.67*Tc/(Tc+243.5))` diverged from model → false supersat in CI.
  - Column integrals: `legoesm.diagnostics.column_integrals` (`column_water_vapor`). No inline `jnp.sum(q*p_s*dsigma)/g`.
  - Losses: `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`).
  - Optimizer: `ml/training.create_optimizer()` (warmup+cosine+clip).
  - SCM-RCE gradient tuning: reuse `scripts/run/run_scm_rce_campaign.py` for CRM
    reference extraction / SCM evaluation and `legoesm.training.scm_rce_metrics`
    for normalized profile score. No duplicated RCE profile numerics.
  - SCM-RCE param training defaults to MUON via `ml.training.create_optimizer()`,
    init from `results/scm_rce_campaign/tuned_parameters.json`, writes recommended
    trained JSON under `results/`, never mutates production `*Config` defaults.
    Apply trainable overrides inside loss so leaves traced; static frozen leaves
    stay outside.
  - Every new `.py`, including `scripts/run/*.py` drivers, gets direct test.
    Scheme/factory dispatch must raise on unknown selections.
  - Atm column (h, ρ, virtual T): `atmosphere.physics._shared`.
  - Ocean EOS/pressure: `ocean.eos` (`compute_ocean_rho`, `compute_ocean_rho_and_pressure`).
  - SFNO: `ml/sfno.py`. No new neural op archs in training.
  - Channel packing: `ml/channel_packing.py` (`PE3DChannelSpec`, `pack_pe_state`, `unpack_pe_output`).
  - Ocean baroclinic (#214): `ocean/dynamics/ocean_tendency_common.py` (`iterate_eos_and_pressure_anomaly`, `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`, `implicit_bottom_drag_factor`) in new `ocean_pe_*.py`.
  - Ocean barotropic (#214): `ocean/dynamics/barotropic_common.py` (`compute_filter_weights`, `bebt_blend`, `maxvel_clip`) in new `barotropic_*.py`. `tests/ocean/unit/test_no_scheme_duplication.py` enforces.
  - Plotters NOT exempt. Use model helpers for q_sat, RH, ρ, virtual T, MSE.
- No duplicate numerics across dycores/physics/grids/tests. Indexing/naming-only copy-paste forbidden.
- **No laziness on hard/large code** (>100 LOC, multi-component, full operator chains): no `pass`/`NotImplementedError` stubs, no partial-called-done, no skip edge cells/boundary halos/corner stencils/non-duogrid/MPI-sharded/AD-VJP. No happy-path-only tests. Too big → say so, list remainder, quantify risk.

## COMPARING TWO CONFIGURATIONS = A SIDE-BY-SIDE TABLE, ALWAYS
User directive 2026-08-21. Any time two runs / configs / templates / lanes are
compared, give a table: one row per field that DIFFERS, one column per
configuration, named in the header. Omit fields that agree — it is a diff, not
an inventory. Prose comparisons hide the one line that matters; the land
coupling week was lost to a surface-scheme mismatch a five-row table exposes at
a glance. This is also the required answer to "why did A work and B not".
If the table has more than one row, no single output difference is attributable
yet — say so.

## ACCURACY FIRST — A CLAIM THAT DRIVES A CODE CHANGE GETS REVIEWED BEFORE THE CODE
User directive 2026-08-21, after a week was spent coupling the WRONG land model
and then chasing two wrong causes in a row: *"we focus now on accuracy and we
must check with adversarial reviews our claims, especially when they imply
coding decisions."*

The existing dual-review rule covers DIFFS. This one covers the CLAIM that
justifies the diff, and it comes FIRST:

- **State the claim, then get it reviewed, THEN write code.** If a sentence of
  the form "X causes Y, so I will change Z" is about to become an edit, it goes
  to codex and GLM as a claim, before the edit exists. Both. A claim reviewed
  only after the code is written is a claim defended, not tested.
- **Say which parts are MEASURED and which are READ OFF THE CODE.** A scaling
  argument, a mechanism, a "this is the only consumer" — each is a separate
  claim with its own evidence.
- **Quote the number's PROVENANCE next to it.** "62 K leaf temperature" from a
  synthetic sweep that deliberately includes unreachable forcings is not the
  same statement as 62 K in a run. Report the reachable-subset number and the
  full-box number separately, and say which is which. A frightening number with
  no provenance wastes the reader's attention and can send the work sideways.
- **A number a well-posed scheme could never produce is a BUG IN THE
  INSTRUMENT until proven otherwise.** Check the probe before reporting the
  physics.
- **When a working configuration exists, diff against it FIRST.** See
  [[reuse-the-working-path-dont-rebuild]]: the offline land driver was healthy
  the whole time; the coupled lane simply called a different parameter provider.
  The first question on any coupling defect is "what does the working lane do
  differently", not "what could be wrong with the physics".
- **Reviewer disagreement is the signal to MEASURE, not to average.** Name the
  discriminating check and run it; it is usually a grep.

## Epistemic rules (non-negotiable)

### Never infer an API — read it
Before calling any function from JAX, Equinox, Optax, Diffrax, jaxKAN, or any
other dependency: grep installed source in site-packages, read actual signature.
No reconstruct from memory. Applies to argument names, argument order,
keyword-only args, return arity.

Symbol under `jax.experimental.*` → assume API changed since training data.
Verify or search. No guessing module paths.

### Report uncertainty explicitly
End any non-trivial code response with `UNVERIFIED:` block listing:
- APIs used but not read from source
- assumptions about library versions or runtime behavior
- anything that would silently produce wrong numbers rather than an error

Empty block = valid answer. Missing block = not.

My premise wrong — bug misdiagnosed, or thing asked for won't work — say so
before writing code.

### Diagnose before patching
Something fail: state candidate causes and how to discriminate between them,
then test. No going straight to fix. No agreeing with cause I suggested unless
evidence support it.


**Applies to THIS repo's own API too** — large enough that memory unreliable.
FAILURES in ONE session (2026-07-30): `AerosolConfig(reference_aod=)` (really
`reference_aod_550`), `McFarlaneConfig(N_ref=)` (no such field),
`run_amip.build_parser` (really `build_arg_parser`), `from legoesm.grids import
create_grid` (really `legoesm.grids.factory`). Each cost full probe round-trip.
Worse, `RRTMGPConfig()` default `include_clouds=False`, so offline harness
omitting it return CLEAR-SKY fluxes and EVERY cloud gradient exactly 0.0 —
silently wrong number, not error. Read NamedTuple `_fields` / `_field_defaults`
before constructing config.

User's premise wrong — bug misdiagnosed, or thing asked for won't work — say so
BEFORE writing code.
## Verification (JAX-specific)

Code not done until it run. Correctness claims require output.

- **Shapes/dtypes**: check with `jax.eval_shape` before running anything
  expensive. Cheap, catch most errors.
- **Gradients**: any new `custom_vjp`/`custom_jvp`, adjoint, or hand-derived
  derivative must pass `jax.test_util.check_grads(f, args, order=2)` before you
  claim it work. Gradient that run is not gradient that correct — single most
  common way to ship silently wrong result here. (Applies directly to
  `_sendrecv_vjp` in `halo_exchange.py` and any new MPI-AD path.)
- **jit parity**: run function eager and under `jit`, compare outputs.
  Divergence = tracer bug (Python-side branching, `.item()`, `if` on traced
  value, host callbacks).
- **Sharding**: verify with `jax.debug.visualize_array_sharding` or by printing
  `.sharding`, not by reasoning about what annotation should do.
- **Numerics**: default float32. State tolerance you compare at. No `==` on
  floats. Test need float64 → say so explicitly, no silent `jax_enable_x64`.
- **donate_argnums / buffer donation**: never add without confirming donated
  buffer not reused. Fail silently or crash far from cause.

## Scope
One change at a time. No refactor of adjacent code, no rename, no "improve" of
code I didn't ask about. Long unbroken generations drift into invention — prefer
small verified diff over large plausible one.

**DO NOT EXTRAPOLATE. Do only what was asked** (user directive 2026-08-06).
The ask = deliverable, not starting point to reason outward from.

- Related-looking problem you notice = ONE-LINE report, not work item. Name it
  and stop; do not start it.
- No widening scope because fix "would only be complete if" something adjacent
  also done. Ship the ask; state the boundary.
- New artifacts (scripts, benches, plots, panels, public APIs, config knobs)
  only when asked or genuinely required to finish the ask. Unsure whether
  required → it not; ask in one line.
- No proposing or launching compute user did not ask for.
- FAILURES 2026-08-05/06: scoped 128-GPU coupled ladder nobody requested off a
  question about existing plots; added coupled panel to figure when asked to
  assess the figure; wrote probes and helper scripts for questions never posed.
  Each cost round-trip and buried actual answer.

## Attribution Gates — MANDATORY, each from a real 2026-07 failure
Model near operational. Every rule below mechanical: satisfy it or state
explicitly that you did not. "I was careful" is not compliance.

- **PROVE THE PATH EXECUTES before blaming a line.** Naming file:line as cause
  require showing that line run in configuration under test: print ENCLOSING
  FUNCTION (`awk` nearest `def` above it), confirm active lane/driver call it.
  FAILURE: blamed positivity clamps at `model_driver.py:10923` for century's
  water source; they live in `_run_per_step` while century run `_run_column`,
  which contain no moisture clamp at all. Fix nearly written for lane the run
  never touch. Same class as reading entry point instead of full path.
- **REUSING A REFERENCE IMPL MEANS PORTING ITS EXCLUSIONS, not just its
  formula.** State which of reference's guards/scope conditions you kept and
  which dropped, with reason for each. FAILURE: copied
  `spectral_les_moist.conserving_positive` but not its `n_water` split, so
  column-conserving borrow applied to number concentrations
  (`N_c`/`N_i`/`N_r`) — unphysical, and it fed M2005 deposition (~N_i^(2/3)),
  producing fake "accelerating dry bias" reported before being caught.
- **A TEST THAT INSPECTS SOURCE MUST NAME THE SYMBOL THAT RUNS, and must be
  shown to FAIL when feature removed.** `inspect.getsource(X)` assertion where X
  is delegating wrapper pass while proving nothing. FAILURE: asserted against
  `MPASPrimitiveEquationModel.step`; floors are in `_step_jit`.
- **TOOL STATUS IS NOT EVIDENCE — read output tail.** Exit code without tool's
  own success line (pytest's `N passed`, "COMPLETED in Xs") = UNVERIFIED; OOM
  kills and timeouts can surface as success. FAILURE: reported regression suite
  green on exit-0 that was actually `Out Of Memory` mid-run. Quote decisive line
  when claiming suite passed.
- **EVERY BASELINE/ALLOW-LIST REASON STRING IS A CLAIM — verify in code before
  writing it.** Plausible-sounding reason permanently hide real defect. FAILURE:
  classified `convective_buoyancy_death_memory` as "carried in SegmentCarry" (it
  not — leaf `BechtoldConfig.buoyancy_death_memory` exist and nothing map to
  it), asserted `SBMConfig.precip_efficiency` leaf that not exist, and credited
  `micro_substeps` to consumer that read `args.`, not the config field.
- **RATE / TENDENCY / SKILL COMPARISONS: identical windows on BOTH sides, and
  print window next to number.** Differing spans = confound, not result.
  FAILURE: TCW over days 190-530 vs CMOR year 1 gave "+38.7 kg/m2/yr"; matched
  windows gave +13.4. Extends existing controlled-comparison rule to derived
  rates.
- **A DIAGNOSTIC'S PRINTED PRECISION BOUNDS THE RATE YOU CAN CLAIM.** Log CWV at
  0.1 kg/m2 over 8 days resolve only ~±4.6 kg/m2/yr — no reporting trend inside
  one quantum. Prefer fp64 from model state (checkpoints) over parsed log lines.
  Same class as throughput-quantization error.
- **`JAX_ENABLE_X64=1` on any numerics/conservation test.** fp32 mismatch NOT a
  failure until re-run with x64; and *new* failure not yours until reproduced
  with your change stashed. Do both before reporting regression.
- **RUN-TARGET PARAMS ARE ABSOLUTE (`TARGET_DAYS`), and "latest checkpoint"
  MOVES.** For controlled pair, COPY pinned checkpoint into each arm dir; never
  use `PREV_CKPT_DIR`-style newest-wins pointer while another run advancing.
  FAILURE (twice): arms exited instantly at "Already at/past target".
- **A LAUNCHER FLAG THAT SWITCHES ONE FORCING CHANNEL MUST SWITCH ALL OF THEM.**
  Verify resolved paths in run log, not the flag you passed. FAILURE:
  `CENTURY_DECK=1` set era-correct ozone+volcanic but left 1979-2016 SST.
- **PROSE IS A POINTER, NEVER A CITABLE FACT.** Code comment, docstring,
  `AMIP.md`/`docs/` entry or "Known issue" naming a limitation, guard, or
  missing feature MUST be re-verified in CURRENT code at point of use before
  repeated as finding — this repo routinely fix things without updating prose.
  Comment name module imposing guard → OPEN THAT MODULE. FAILURES (2026-07-30,
  three in one day): quoted `_run_column` "turbulent surface fluxes are
  intentionally NOT applied" comment and `AMIP.md` Known #3 to claim MPAS has no
  turbulence — MPAS turbulence path exist (`turbulence/integration.py`, Perot
  edge->cell) and run resolve `turbulence=louis` + `surface_bulk_scheme=coare3`;
  and doubted a cloud_fraction comment that was exactly right.
- **THE FIRST GUARD YOU FIND IS NOT THE ONLY GUARD — follow value to its
  CONSUMER before declaring it unclamped/unchecked.** FAILURE: reported "no
  upper bound on r_eff" from `rrtmgp.py`'s `clip(x, 1e-6, None)`; real clamp to
  lookup-table range is one call deeper in `rrtmgp/optics/cloud_optics.py`. Same
  class as blaming a line without proving its enclosing function runs.
- **BEFORE ATTRIBUTING A BIAS TO A COMPONENT, PROVE THE COMPONENT EXISTS AND
  RUNS IN THE CONFIG UNDER TEST.** ABSENT component and BADLY-TUNED one give
  SAME symptom, and every tuning arm against absent component return null.
  Enumerate, from RESOLVED config and LANE'S code path (not deck's comments):
  the scheme, its PROGNOSTIC STATE, its inputs. No prognostic state -> say so
  out loud; that usually the finding.
  Corollaries, each earned:
  (a) **A `false` on a component switch does not mean "the simpler version" —
  it may mean NOTHING.** Check what lane actually fall back to.
  (b) **A DECK COMMENT NAMING A FALLBACK IS A POINTER, NOT A FACT** (prose rule,
  applied to configs): grep named symbol in that lane's setup.
  (c) **When N different schemes for component A all fail to move a bias, STOP
  TUNING A.** Limiter upstream, or A's inputs wrong.
  (d) **A parameter wrong at BOTH ENDS of a physical range is a MISSING
  DEPENDENCE, not a mistuning** — thing it should depend on not read.
  FAILURE 2026-08-15, most expensive of campaign: MPAS AMIP deck set
  `use_multilayer_land: false` with comment "this deck runs the slab". There is
  NO slab land on MPAS lane (`slab_land_active` wired only on cube/general
  path), so runs had **no land surface model at all** — land T_sfc was nearest
  OCEAN's prescribed SST minus 6.5 K/km, land evaporation fixed 0.6 of
  saturation, no soil, no water store, no runoff, no stomata, no surface energy
  balance. Tropical deserts evaporated 3.5x observed and rainforest 0.73x FROM
  SAME CONSTANT (corollary d, unnoticed); sensible heat ~25 W/m2 on Sahara and
  Amazon alike. Waves 9-11 spent GPU-hours swapping FIVE convection schemes and
  dozen trigger knobs against tropical-land rain deficit no convection scheme
  could ever fix (corollary c, unnoticed). Interactive soil moisture, bucket
  hydrology, stomatal control are DEFAULTS for AMIP case, not options.
- **A GLOBAL STATISTIC ON A NON-UNIFORM GRID NEEDS AREA WEIGHTS.** Never
  `np.mean(field)` for global mean on lat-lon (or any stretched grid) — use
  `cos(lat)` or model's `areacella`. FAILURE: reported "+17 hPa of dry mass
  created" from unweighted `p_s` mean; AREA-WEIGHTED mass was invariant at
  983.493 hPa to 6 digits, i.e. defect did not exist. Habits carried from
  quasi-uniform MPAS/SCVT mesh are INVALID on lat-lon.
- **A PROPOSED MECHANISM MUST SURVIVE A SCALING / PERTURBATION TEST BEFORE IT IS
  CITED AS THE CAUSE.** X claimed to drive Y → change X by known factor, check Y
  respond as mechanism predict. FAILURE: proposed "damp-to-rest pumps mass
  convergence" (predict ~linear in sponge coefficient); quartering coefficient
  slowed growth only 1.5x, refuting it — fix would have shipped on false
  mechanism. Label every uncaught claim PLAUSIBLE; honest "cause unknown" is
  cheap, confident wrong cause buy a code change and a relaunch.

## Implementation Discipline — the CODE and its PROSE are both claims
User, 2026-08-06: *"Be much more conscientious and careful when implementing.
Be systematic, check, do not be sloppy or too fast."* Every rule below from
defect shipped in ONE session that prompted it, and every one caught by
adversarial reviewer rather than by me — i.e. each avoidable by reading two more
lines before typing. Slow down at these exact points.

- **A DOCSTRING/COMMENT THAT DESCRIBES BEHAVIOUR IS A TESTABLE CLAIM. Trace data
  flow before writing it.** FAILURE: wrote "dropping this call now makes the
  test fail" about `divg_d` exchange — false, because unit 4 hands only
  `uc`/`vc` to `d_sw1` and `divg_d` not consumed until `d_sw5`. Sentence written
  from intent, not from call chain. Before asserting "X is covered by this
  test", name consumer of X and confirm it execute inside test's span.
- **NEVER WRITE "this removes the question entirely" ABOUT AN API YOU HAVE NOT
  READ.** FAILURE: claimed `np.ascontiguousarray` give unconditional copy; it is
  copy-IF-NEEDED, so it aliased at km=1 and copied at km>1 — one line of code
  with two aliasing behaviours, asserted as safe. Extends *Never infer an API*
  to STRENGTH of a guarantee, not just signature.
- **SCOPE WORDS — "both", "all", "every", "cannot", "unified", "always" — GET A
  GREP BEFORE THEY GET TYPED.** FAILURE: "both lanes now route through one
  helper" while `csw_step_sixface` still called interim helpers directly. Went
  into COMMIT MESSAGE, which cannot be edited after. Weaken to what you verified
  ("the post-p_grad_c site") or run the grep.
- **RE-READ A BOOLEAN IMPLICATION IN THE SOURCE BEFORE RESTATING IT; ONE-WAY IS
  NOT TWO-WAY.** FAILURE: `bounded_domain = regional .or. nested .or.
  duogrid` mean duogrid⇒bounded, and I wrote converse pair "unreachable" —
  inverting it and mislabelling legitimate regional/nested category as
  impossible. Quote the line next to restatement so direction checkable.
- **A CONTROL THAT PERTURBS A ZERO IS NOT A CONTROL. Print baseline value the
  perturbation multiplies BEFORE trusting result.** FAILURE: divg-corner probe
  scaled `divg_d(1,1)` by 3 where that cell is exactly 0.0, so control was no-op
  and proved nothing; claim actually rested on predicted==actual identity. Say
  which check carried the claim.
- **AN A/B MUST DIFFER IN ONE FIELD OF THE CONSTRUCTOR, AND YOU MUST DIFF THE
  CONSTRUCTOR ARGS TO KNOW.** FAILURE: compared two contexts differing in grid
  conventions AND in whether ext bundle existed, while asserting it isolated the
  exchange. Applies to fixtures, not just runs — controlled-comparison rule
  cover `pytest` fixtures too.
- **A NEW TEST/FIXTURE IS PART OF THE DIFF: RUN THE SUITE THAT CONSUMES IT, NOT
  ONLY THE TEST YOU WROTE.** Fixture edit = change to every test in the module.
- **SHELL COMMANDS THAT CAN PROMPT WILL SILENTLY NOT RUN.** Use `git checkout
  --`/`git restore`, `cp -f`, `rm -f`; never bare `cp`/`mv` for revert. FAILURE:
  bare `cp` hit interactive overwrite prompt and "revert" never applied — one
  `git status` short of reporting clean tree that was not clean. VERIFY EVERY
  REVERT with `git status --porcelain`.
- **A LOG-SCRAPING GUARD MUST EXCLUDE THE PROMPT/COMMAND IT ECHOES.** FAILURE:
  review wrapper grepped whole log for failure token its own prompt contained,
  so it "retried" every time and never printed exit status. Same class as test
  that cannot fail.
- **BUDGET THE REVIEW LOOP INTO THE WORK.** Adversarial reviewer found real
  defects in EACH of three rounds here; rounds 2 and 3 not ceremony. No
  presenting first-round implementation as finished, no treating "tests pass" as
  terminal condition — round-1 diff passed 401 tests while still containing
  silent fallback and false docstring.

## Assumption Gates — from six wrong assumptions in ONE session (2026-08-05/06)
User callout: *"you keep making a lot of assumptions that prove to be wrong."*
Gates above stop wrong CLAIMS about the model; these stop cheaper, more frequent
error — being wrong about CODE AND DATA IN FRONT OF YOU. Every rule below
mechanical, each cost full round-trip.

- **AN ARRAY'S LAYOUT IS AN API — READ IT, INCLUDING AXIS ORDER.** "Never infer
  an API" rule cover signatures; it also cover array SHAPE, AXIS ORDER, index
  base, padding convention. Before first index of any mesh/state field, print
  its `.shape`. FAILURE: indexed `mesh.cellsOnCell[c, k]` assuming
  `(nCells, 6)`; it is `(6, nCells)`, so probe raised `IndexError` on every
  method. `cellsOnEdge` is `(2,
  nEdges)` — one unambiguous pair per edge, usually the better handle.
- **A PROXY IS NOT THE QUANTITY. Real number produced by specific code path →
  GET IT BY CALLING THAT PATH.** Re-deriving lookalike from first principles
  silently answer different question. FAILURE: computed `max_degree` from 1-ring
  `cellsOnEdge` adjacency and reported it as "ppermute round count" — real
  exchange is halo-depth-aware, so proxy said 8 rounds for every method/rank
  count while real schedule said 12→14. Proxy could not even reproduce known
  answer — that the tell: **run proxy against a case whose real value you
  already know BEFORE using it on the unknown one.**
- **AN OPTIMIZER ONLY HELPS IF ITS OBJECTIVE IS THE BINDING TERM — state which
  quantity it minimizes, confirm that quantity is measured bottleneck.**
  FAILURE: assumed METIS would cut MPAS ppermute rounds; METIS minimize EDGE
  CUT, bottleneck is MAX_DEGREE, and measured they move OPPOSITELY (metis 19
  rounds @64 vs sfc 14, while metis has lower cut). "Better partitioner" is not
  a mechanism.
- **A TABLE YOU PARSED IS NOT DATA UNTIL YOU SPOT-CHECK IT.** Any derived
  summary quoted to human, or fed to review agent, need ≥2 rows verified by eye
  against raw source first. FAILURE: regex over multi-line series lists attached
  subdiv-9's `9.60/11.48 ms` to `s8` label, and that mislabelled pair went into
  codex prompt as fact.
- **PRE-IMPL GREP COVERS TESTS, NOT JUST FUNCTIONS.** Before writing test, grep
  for one already asserting same invariant. FAILURE: added two tests counting
  PCG reduction sites duplicating existing `test_reduction_count_halved`; codex
  had to point it out.
- **WHEN A TEST FAILS, DECIDE WHETHER THE EXPECTATION OR THE CODE IS WRONG, AND
  SAY WHICH.** Assertion you just wrote is claim with no more standing than the
  code. FAILURE: asserted `LEGOESM_..._FUSED_HALO=""` meant OFF; resolver is
  `!= "0"`, so `""` means ON — test was wrong, not the default.
- **BEFORE PROPOSING COMPUTE, CHECK THE PATH IS BUILT — grep driver for
  decomposition/sharding the run would need.** FAILURE: scoped 128-GPU coupled
  ladder before finding `CoupledESM` contain three occurrences of "shard", all
  in comments — coupled ocean is replicated, so ladder would have measured
  unbuilt path.
- **WHEN TWO MECHANISMS COULD EXPLAIN A NUMBER, NAME THE MEASUREMENT THAT
  DISCRIMINATES THEM AND RUN IT BEFORE REPORTING EITHER.** FAILURE: reported
  serial-vs-SPMD gap as "cadence PHASE disagreement"; predicates agree with ZERO
  offset and real cause was serial short-tail fallback. One `grep` of two
  predicates would have settled it.

## Compute Discipline — speculation costs GPU-hours, not just credibility
User, 2026-07-30 (THIRD callout in five days): *"You keep making very
speculative assumptions... be much more precise so we do not waste time with
useless simulations."* Gates above stop wrong CLAIMS; these stop wrong RUNS.
Simulation launched on hypothesis no measurement can refute = pure waste, and it
also cost WALL-CLOCK of queue slot it occupied.

- **NO COMPUTE ON AN UNFALSIFIABLE HYPOTHESIS. Before submitting ANY job costing
  >1 GPU-hour, write down three things: (a) exact number the run will produce,
  (b) value that CONFIRMS and value that REFUTES, (c) why a cheaper offline/CPU
  test on an EXISTING checkpoint cannot answer it. Cannot fill all three -> DO
  NOT SUBMIT; run cheap test first.** Nearly every question asked so far
  (fluxes, tendencies, radii, momentum budgets, cloud optics) was answerable
  offline from saved checkpoint in minutes. FAILURE 2026-07-30: submitted two
  5-YEAR full-physics runs (8 h walltime each) while model had KNOWN unfixed +56
  W/m2 albedo error and no low-level circulation — five simulated years of
  broken climate, answering no question that had been asked.
- **RANK ERRORS BY MAGNITUDE BEFORE CHOOSING WHAT TO WORK ON.** Run full
  scorecard FIRST, work LARGEST term; re-rank after every fix. FAILURE
  2026-07-30: spent most of session on hfls deficit (-40 W/m2) while dominant
  error was rsut (+56 W/m2) — and scorecard naming it was already sitting in run
  directory, unread.
- **ONE VARIABLE PER PRODUCTION RUN.** N simultaneous config changes answer ZERO
  questions, because no output attributable to any one of them. Multi-change
  config legitimate ONLY as deliberate new BASELINE labelled as such and never
  compared term-by-term against the old one. FAILURE 2026-07-30: one launch
  flipped ~8 switches at once.
- **A LONG RUN ON A MODEL WITH AN UNFIXED DOMINANT ERROR IS WASTE.** Before
  extending past ~30 simulated days, state largest outstanding scorecard term
  and why run still worth its GPU-hours. Fix big term, then extend. Short
  validation windows (days) are for "does it run and is new physics behaving";
  multi-year windows are for model that already pass.
- **NEVER LEAD WITH AN ARITHMETIC COINCIDENCE.** Hand-computed ratio that
  "matches" observed ratio is not evidence when calculation omit factors code
  actually applies. State it as arithmetic, or don't state it. FAILURE
  2026-07-30: "cover x tau ~ 1.9x matches the 1.9x albedo" ignored sub-grid
  inhomogeneity factor radiation applies to cloud paths.
- **PREFER THE INSTRUMENT THAT ALREADY EXISTS.** Before writing probe, check run
  directory for scorecard/manifest/diagnostic answering the question, and
  `scripts/validate/` for validator. Reading existing artifact cost seconds; new
  probe cost an hour and need its own controls.

## Diagnosis Discipline — one session, ~10 GPU arms, 5 of them wasted
2026-08-05/06, FESOM2-match OMIP blowup. Every rule mechanical, from specific
failure in that one session. Root cause turned out to be one-line IC defect
findable offline in seconds; arms bought nothing.

- **SANITY-CHECK EVERY PROBE NUMBER AGAINST A PHYSICAL RANGE BEFORE BUILDING ON
  IT — one line of arithmetic, before the conclusion.** Convert to quantity
  whose plausible span you know, compare. FAILURE: PGF probe returned 0.02
  m/s^2 and I built whole attribution on it; that value imply ~14 kg/m^3 density
  difference between ADJACENT 1-degree cells when entire ocean span ~6 kg/m^3.
  Check take seconds, number sat unused for hours, and running it immediately
  would have exposed real defect (cells initialised to T=0/S=0) before a single
  GPU arm.

- **A FAILURE STEP OR LOCATION IS NOT A SIGNATURE UNTIL THE DIAGNOSTIC IS SHOWN
  TO RESOLVE IT.** State instrument's resolution, confirm reported step is not
  merely its first sample. FAILURE: five one-variable arms all reported "BLOWUP
  at step 36" and I read that config-invariance as physics. `run_omip` set
  `block_size = max(1, diag_every)` and only test at block boundaries, so 36 was
  FIRST LOOK. At `--diag-every 1` answer was step 1. Corollary: arms run at
  coarse diagnostic stride prove only "this setting alone does not fix it",
  NEVER "this setting has no effect".

- **REFUTE WITH STATES, NOT EXIT CODES OR FAILURE STEPS.** Two runs failing at
  same step can fail for different reasons. FAILURE: declared open North Pole
  "refuted" because capping did not move blowup step; states showed capping HAD
  removed that mode (|u|max 19.27 -> 0.0137 m/s) and merely uncovered second,
  unrelated one at same step number.

- **A PROBE IS PRODUCTION CODE FOR THE "NEVER INFER AN API" RULE, AND ITS FIRST
  OUTPUT IS UNTRUSTED.** Read signature of every function a probe call,
  including this repo's own. FAILURE: called
  `build_runoff_map(lat, lon, mask, area)` when signature is
  `(ocean_mask, cell_area, lat_rad, lon_rad)`, so `cell_area` received
  `deg2rad(lon)` — exactly 0 at lon 0 — and I reported NaN "conservation defect"
  in shipped code that did not exist.

- **A PLAUSIBLE-LOOKING VALUE IS MORE DANGEROUS THAN A NaN, IN CODE AND IN
  TESTS.** Sentinel that is valid number of right dtype pass every finite/NaN
  guard downstream. FAILURE: `_interp_profile_to_z_coord` returned `0.0` for
  no-data column; `0.0` is not NaN, so caller's `where(isnan, fill, x)` never
  fired and 39032 wet cells entered model as fresh water at 0 degC. TWO
  COMMITTED TESTS ASSERTED THAT BEHAVIOUR, one calling it "documented degenerate
  fallback" and one commenting "still no NaN leaks" — exact inversion of truth.
  Degenerate branch must return something → return NaN (or raise) so downstream
  guard can see it, and treat any test asserting magic sentinel as suspect.

- **BUDGET CLOSURE BEFORE HYPOTHESES.** For any conservation-relevant blowup or
  drift, run closure/redistribution check FIRST — it name the operator class
  instead of ranking guesses. Canonical probe:
  `scripts/validate/ocean_fidelity/omip_conservation_closure.py`. On first
  deployment it refuted hypothesis I was about to spend a GPU arm on (dipole
  whose two-cell sum GREW is not a diffusion instability) and localised
  config-invariant source to single column. See
  `docs/ocean/fidelity/fidelity_to_fesom2jax_level_plan.md` (#1492) Phase 0.1.

- **A THROWAWAY PROBE'S NUMBER IS UNMEASURED. COMMIT THE PROBE.** Per #1492
  Phase 0.3: one committed probe per row, locked conventions, provenance (git
  SHA + inputs + flags) stamped on every run. Inline heredoc probes not citable,
  cannot be re-run against changed model, and hide own bugs — both defects above
  came from uncommitted ones, and committing closure probe immediately surfaced
  two more inside it.

- **DAMAGE-FIELD CORRELATIONS ARE NOT MECHANISMS.** In blown-up field, "worst
  cells are coastal / shallow / polar" describe where damage LANDED, usually
  downstream of single upstream defect. FAILURE: measured P(bad|coastal)=33% vs
  0.67% interior, real 50x enrichment, and treated it as coastal process; those
  were simply columns the IC defect had corrupted. Before citing spatial
  pattern, confirm snapshot is ORIGIN state (`snapshot_final.npz` stamp
  `_step = n_steps` regardless of when run stopped — verify with dedicated
  1-step run).

## JAX
- Pure pytree fns. `lax.scan` time integration. `vmap`/batched arrays over Python loops on array dims. `jnp.where`/`lax.cond`/`fori_loop`/`scan` not Python control flow on traced.
- **Feature gating exception** (`fix_mass`, `fix_moisture`): Python `if` on static bool in closure — NOT `jnp.where` (traces both branches). `jnp.where` only for data-dependent traced selection.
- Stable shapes. No retrace. Dtype: spectral=x64+complex128; finite-volume can float32.
- No host/device thrash, NumPy in traced code, hidden non-JAX side effects.
- **Buffer donation + `jax.grad`**: `donate_argnums` conflicts reverse-mode AD. JIT fn inside `jax.grad`/`eqx.filter_value_and_grad` → provide non-donating variant (`.raw`). See `build_segment_fn`.
- **Closures vs explicit args**: closure captures = compile-time consts. Per-iter changing val (SST, solar) → pass as traced arg. See `SegmentForcing`.

## Earth System
- Conservation, metric consistency, staggered-grid consistency, halo correctness = first-class.
- No silent clip/damp/coerce unless justified+validated.
- Preserve units, sign conventions, monotonicity/positivity, hydrostatic/nonhydrostatic.
- Cubed-sphere/curvilinear: assume edge+metric errors first.
- Physics coupling: column closure + consistent flux signs.
- DA/diff: preserve smoothness. No gratuitous nondiff.
- **Sign-convention check MANDATORY on every equation/flux/tendency edit.** Before declaring done on any code touching PDE term, flux, source/sink, BC, or budget update: (1) state coordinate convention in scope (z up/down, flux positive-up/down/into-body) as comment at the term; (2) walk EACH term, confirm its sign match that convention — gravity vs capillary/diffusion divergence, top vs bottom BC, source vs sink, the `±` in `state = state ± dt·tend`; (3) confirm budget closes (`in − out − Δstorage = 0`) and exchanged fluxes carry SAME sign at both ends of a coupling (land runoff `+into ocean` must arrive `+into ocean`). Comment label and code must agree — flux commented "upward" computed as downward = defect, fix label or math. Mechanical gate where feasible: sign/conservation unit test (analytic column, manufactured solution, or `assert` budget residual ≈ 0) — passing norms alone never certify sign right (flipped flux can still be small). Common flips: z-axis direction, evap positive-up vs moistening, brine/salt vs freshwater dilution, stress atmospheric vs ocean convention (`-tau`), free-drainage vs gravity double-count.

## Parallel/HPC
- Correctness across serial/multi-device/MPI/hybrid.
- Sharded: reason about halo exchange, reductions, partition specs, global invariants.
- Validate single-rank → smallest distributed.
- Backends differ (Metal/GPU/CPU/spectral/MPI). Apple Silicon: spectral on CPU.
- **MPI**: `initialize_distributed(global_n=N)` → `scatter_to_local()` → rank-local step → `gather_to_global()` for I/O only. Never full global per rank.
- **4D halo**: `pad_halo_4d()`+`pad_halo_vector_4d()` all vert in one msg. All 3D ops in `operators_3d.py` use 4D. Never `vmap(pad_halo)`.
- **MPI halo AD**: all `sendrecv` via `_sendrecv_vjp` (`@jax.custom_vjp` in `halo_exchange.py`). Only `allreduce(SUM)` AD-safe (via `global_sum_mpi`; `final_loss=True` only for a sum seeded as the loss); `MAX`/`MIN`/`allgather`/`bcast` = diagnostics only.
- **Device mesh under MPI**: per-rank count to `create_device_mesh()`, not total.

## Oracle-Recipe Fidelity (ocean) — see docs/ocean/fidelity/oracle_recipe_strategy.md
- ADDITIVE to Validation Rules: oracle work NEVER replaces unit tests, ocean matrix, conservation checks, or visual verification. Truth tiers (conservation/equivariance/analytic) outrank oracle-matching.
- Recipe = pure config selecting shared canonical blocks (never bespoke `veros_*` solver). Oracle-matching numerics go in canonical module (`eos.py`, advection/limiter dispatch, `vertical_mixing/`, integrator dispatch) as selectable options.
- Mimicry-only glue (halo strip, axis transpose, time-level handling) lives in fidelity harness, never model. Test: "would a user with different goal ever select this?" No → harness.
- Conventions handled only in bridge, verified by equivariance tests (`physics(φ(x))=φ(physics(x))` to tol); a "convention" changing wet domain/answers is physics → config, not bridge.
- Constants are config (`ConstantsConfig`), not module-global monkey-patches (no `override_constants` in shippable paths); defaults reference `legoesm.constants`; base only, derived (κ,ε) recomputed.
- Oracle tendency-match (tier 3) trusted only for block that also clear truth tiers (0–2). MACHINE-ENFORCED (#388 Ask#4): `ocean/fidelity/precedence.py::evaluate_precedence` LOCKS oracle tiers (≥3) on any truth-tier (0–2) failure; surfaced + exit-gated by `scripts/validate/ocean_fidelity/build_fidelity_scorecard.py` (the one generated scorecard).

## Recipe×Setup template adapters (#388)
- **One shared selector, never re-implemented.** Every component YAML experiment adapter exposing a `setup:` block (atmosphere `Config`, ocean `OceanExperimentConfig`, sea-ice `SeaIceExperimentConfig`, …) MUST route validation + command-building + signature through `legoesm.core.setup_selector` (`MatrixRunnerSpec` + `validate_setup`/`build_matrix_command`/`setup_signature`/`require_positive_finite`). Per-component CLI differences (case flag `--only`/`--test`, exact-match `exact_prefix`, which `--levels`/`--dt`/`--days`/`--resolution` flags exist) are a `MatrixRunnerSpec`, NOT a copy-pasted `parts=[...]` builder or private `_SETUP_KEYS`/`_validate_setup`. Component-specific case-name validation goes through `known_names=` kwarg. Re-implemented selector = REJECTED in review. Factory naming: `_<component>_matrix_spec()`. FOLLOW-UP: dedup `OceanExperimentConfig`'s inline selector onto shared helper once PR #465 + shared-helper PR both land on main (inline copy predates extraction).
- **Every setup template's `(name, grid)` MUST be a real matrix case**, enforced by catalog-backed test (`run_<comp>_test_matrix._build_test_matrix()` → assert pair exists) so `--test/--only =<name> --grid <grid>` never select nothing. New template without this gate → REJECTED.
- **An exact case selector matching nothing is a hard error** (`raise SystemExit`) in every matrix-runner mode — never silent no-op (dispatch-hardening). MPI-safe: key guard off global pre-slice match count; empty-slice ranks fall through to barrier.
- **New `experiment_registry` mode → same-PR `get_adapter` test** in `test_experiment_registry.test_get_adapter_resolves_classes` (template test importing class directly does NOT cover dispatch).
- Land has no standalone idealized-case surface (no registry/matrix runner) → recipe×setup N/A there; do not invent one to match pattern. Truth-tier precedence stays ocean-scoped per `oracle_recipe_strategy.md`; use `TRUTH_TIERS`/`ORACLE_TIER_FLOOR` by name (no hardcoded `3`).

## Validation
- Narrowest test after edits. Numerical changes: analytical/benchmark > unit tests alone. `JAX_ENABLE_X64=1` unless float32/Metal task.
- Dycore: Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, ocean benchmarks.
- Conservation/reductions/coupler: mass+energy diagnostics.
- Parallel: unsharded vs sharded, single-rank vs MPI.
- Too expensive: say what ran/didn't, residual risk.
- **CRITICAL — Controlled comparison: change ONE variable, hold eval protocol FIXED to baseline.** To claim a change (resolution, params, scheme, days) improved/degraded/"is comparable" vs prior result, keep EVERYTHING else byte-identical to that baseline: forcing data + its sampling (years/days/hours/climatology), evaluation grid, metric definition (bias vs RMSE), region masks, timestepping. Metric that moved because protocol/sampling changed = **CONFOUND, not result** — NEVER compare number computed on one sampling/grid/metric to number from another and call difference an effect. Resource limit (network cap, compute, time) force lighter or different sampling → **re-run BASELINE at that SAME sampling before comparing** — fresh baseline cheap insurance; confounded claim not. Before writing "improved"/"degraded"/"better"/"comparable" vs any earlier number, explicitly confirm two configs differ ONLY in variable under test; cannot → say so, claim NO direction. Report full config (data+sampling, grid, days/steps, params, metric) next to every number so reader know exactly what compared. Assuming two runs comparable when setup drifted = error that turn "we improved it" into "we degraded it."
- **CRITICAL — Precision gate for comparison/skill/causal claims (do NOT be sloppy — 2026-07-20 EC-site lesson).** (a) **Harness self-check FIRST**: before reporting NEW scheme's skill vs validated baseline, run KNOWN baseline through YOUR OWN harness, confirm it reproduce baseline's established/published number (±small tol). Your harness score validated scheme wildly off — e.g. two-leaf H looked "broken" (NSE≈−1) when paper figure track obs — the HARNESS is wrong; fix BEFORE any new-scheme claim. (b) **Match reference metric EXACTLY**: metric aggregation (daily-mean vs half-hourly point-wise NSE), window (multi-year JJA vs one summer), masks (valid-forcing) ALL part of protocol. Report SAME metric reference used, plus any alternative, WITH sample N. Single-window point-wise metric is NOT skill verdict, and never rank schemes by metric punishing one scheme's known artifact (point-wise scatter/spikes) while ignoring dimension of interest (mean diurnal shape). (c) **Never let QC mask flatter one side**: drop scheme's unphysical spikes from scoring → report BOTH failures-penalized primary score AND separately-labeled plausible-only sensitivity score. (d) **Do NOT INFER causal origin — INSTRUMENT it**: "the spikes come from X" require LOGGING X and alternatives (raw vs intermediate vs final), not deduction from reading one code path; "correctly hooked up" require reading FULL path, not entry point. (e) **Label every claim CONFIRMED (evidence shown) vs PLAUSIBLE (inferred)** — adversarial review WILL refute over-confident inferences; state uncertainty up front rather than presenting verdict table a five-minute check overturn.
- **CRITICAL — A gradient/AD-tuning result that underperforms a known-good baseline is a BUG in your method until PROVEN otherwise — never attribute it to physics.** Before interpreting ANY optimizer/AD tuning outcome, or attributing residual error to physical cause (radiation, resolution, scheme's structure): (1) **Differentiated loss MUST be exact quantity you evaluate and report.** Truncated-BPTT / short-window / stop-gradient'd-spinup / cached-state proxy differing from full evaluation = OBJECTIVE MISMATCH, not result — optimizer minimize proxy while eval barely move. Differentiate real objective (e.g. full RCE equilibrium the eval scores), or prove proxy tight. (2) **Verify AD actually flows:** finite-difference check on ≥1 parameter (analytic grad vs `(L(x+ε)−L(x−ε))/2ε`, ratio≈1), confirm gradients finite AND non-negligible (not silently zeroed by `stop_gradient`, frozen/`enable_*=False` leaf, `lax.stop_gradient`/`jax.lax.cond` dead branch, or detached recompute). (3) **Sanity vs known-good baseline FIRST:** derivative-free or prior method got large improvements and your gradient method barely move loss → YOUR optimization broken (weak objective, tiny effective LR, over-aggressive line-search backtracking to ~0 step, frozen leaves) — fix before drawing conclusions. Concluding "residual bias is radiative/structural, not tunable" from run whose optimizer never actually worked = exact error that ship FALSE scientific claim. Near operationalization this disqualifying: barely-moving loss, suspiciously-flat before/after, or result contradicting known-good baseline BLOCKS delivery until gradient path verified end-to-end. See [[feedback_verify_ad_before_physics]].
- **CRITICAL — Visual verify spatial/grid artifacts**: passing tests+norms NECESSARY ≠ SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coeffs, grid metrics. Edge artifacts/cube imprint/grid-scale noise only detected visually (v-wind W2, wind_speed W5). Run `--only sw --grid cubed_sphere --quick` + inspect PNGs vs baseline. Norms can improve while artifacts worsen. Never claim "tests pass, edge fixed" from pytest alone.
- **Diffusion sensitivity**: div damping + hyperdiff AMPLIFY halo errors at cubed-sphere face boundaries. Check W2 v-wind visually when touching `_hyperdiff_cube`, `_div_damp_cube`, diffusion params.
- **Visual-regression gate (cube imprint)**: `scripts/validate/visual_regression.py --check` numericises W2 v-wind cube-imprint check (SSIM + per-panel perceptual hash + edge-artifact ratio vs tiny committed ref in `tests/visual_baselines/`). Deterministic metric math gated in CI (`tests/test_visual_regression_metrics.py`); full cube-SW `--check` runs as NIGHTLY non-blocking CI job until tolerances calibrated across CI hardware. Tiny numeric baselines (.npy+json) ARE tracked — the one carve-out to "no tracked visual baselines".
- **CRITICAL — VALIDATE THE INSTRUMENT BEFORE QUOTING ITS NUMBER (2026-07-25 duogrid lesson: 8 confident claims, all retracted).** Diagnostic script = UNTRUSTED CODE until it pass own controls. Never state finding — never write "measured", "confirmed", "proven", "VERDICT" — from probe's first output. **Before quoting any diagnostic number, run these five checks and say in the message that you ran them:**
  1. **Right conserved/invariant quantity?** Budget what SYSTEM conserves, not convenient proxy. (Failed: reported "vertex creates energy" from **KE alone** — KE NOT conserved in shallow water, it trade with PE. Total `E=∫area(½h|V|²+½gh²)` reversed sign of conclusion.)
  2. **Same transform / units / staggering on BOTH sides?** Two "A-grid winds" from different operators = DIFFERENT QUANTITIES. (Failed: ours `c2l_ord2` vs oracle `C2L_ORD=4` — SAME raw state gave 1.96e-2 vs 5.79e-2, 3× swing that WAS the reported effect. Also: never budget across stage boundary where state change representation — mid-step FV3 winds in circulation form, produced ±5.6e10 garbage. REPEAT OFFENCE 2026-08-05, this time SHIPPED then retracted: paired predicted face velocity (41.87 m/s) against model's `max_speed` timeseries diagnostic (21.95 m/s), which is CELL-CENTRE average of same field whose face maximum was 42.30 — apparent "1.2 % agreement" was artifact of comparing across staggering, and it went into merged PR before 1-step run exposed it. Name staggering AND reduction of both sides in sentence quoting them.)
  3. **Same time, resolution, config?** Index by MATCHED TIME, not frame number. (Failed: mapped day→frame as `round(day)-1` against HOURLY file, comparing our day-1 to their hour-1; and quoted **C12** wedge gain (~300×) as mechanism for **C48** instability, where it ~124×.)
  4. **Is the metric measuring what its name says?** Prove it on synthetic case with KNOWN answer before use. (Failed: called `mean|f−4-neighbour-mean|` a "2Δx grid-scale" measure — it is high-pass/curvature residual that merely sharper SMOOTH feature reproduce. Failed: "gain" probe that re-filled FIXED source, trivially 1.0000 by construction.)
  5. **Can the reduction support the claim?** `max` over tiles/corners/components taken independently per run can peak at DIFFERENT physical locations; max-of-per-tile-means is not global mean. Keep argmax metadata, map to common physical location before claiming "localized".
  Plus: **diff ARRAYS, never printed summaries** (claimed "bit-identical ⇒ deterministic, not chaos"; arrays actually differed by 9e-6 — only rounded printout matched). **Never let probe print own verdict** ("=> the growth is REAL") — interpretation belong in analysis after controls pass, not baked into tool where it get echoed back as evidence. **`nanmax`/`nanmean`/`nansum` hide failures** — make NaN and missing frames FATAL. **Record every effective flag, env var and git SHA in each artifact**; default `--n 36` silently mis-slicing C48 file run fine and lie.
- **CRITICAL — LABEL EVERY CLAIM, AND PREFER RETRACTING EARLY.** Tag each statement **CONFIRMED** (control-passed evidence shown, instrument validated) vs **PLAUSIBLE** (inferred / single-run / uncontrolled). Chain of PLAUSIBLE steps is not CONFIRMED. Later measurement contradict earlier claim → **retract loudly and immediately in same message and in memory file** — no quiet move on, because stale confident claims get built on. Run codex adversarial review on DIAGNOSTIC TOOLING, not just model code: instruments decide what you believe, so bug there manufacture confident wrong physics conclusion. Cheap self-check before any big claim: *"what measurement would make this false, and did I run it?"* Answer no → claim PLAUSIBLE at best.

## Domain Architect vs Syntax Engine (AI guardrails)
See `docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md`. Doctrine: human dictate *logic* (units, signs, conserved qty, valid scheme sets, references, acceptance criteria); AI fill *body*; every declared invariant checked **mechanically** so violations fail LOUDLY. Each gate = **tripwire, not proof**, ships synthetic-violation self-test (provably non-vacuous). NON-NEGOTIABLE harness (extend, never weaken; budgets/TODOs shrink only):
- **Spec-first physics contracts**: every physics scheme module declares `__physics_contract__` (units/signs/conserves/differentiable/reference/idealized_test); `tests/test_physics_contracts.py` partitions all `*/physics/*.py` into EXCLUDED / CONTRACT_TODO(shrink-only) / annotated — NEW physics file must ship contract or be classified. Author contract + acceptance test BEFORE body.
- **Atmospheric parameterization edits are high-risk**: run codex adversarial review, #477 truth-tier tendency validators, conservation tests, equilibrium-SCM-RCE realism harness before merge. NEW scheme must pass RCE realism gate or enter its shrink-only TODO partition with a reason.
- **CI ratchets** (AST + self-test, via shared `tests/_ratchet_audit.py`): `test_no_hardcoded_constants` (constants only from `legoesm.constants`), `test_no_saturation_reimpl` (saturation only from `thermo`), `test_dispatch_hardening` (no scheme guard silently deleted), `test_validate_strict_coverage` (no scheme field skips fail-early validation). Escape: real `# const-ok:`/`# satcurve-ok:` comment.
- **Local hooks** (`.claude/hooks/`, wired in `settings.json`): PreToolUse blocks edit adding banned constant/saturation prefactor to `.py` (fail-open); Stop reminds (once/session) to run mandatory codex review on uncommitted numerics/physics. CI remain authoritative (`Bash` heredoc bypass hook, not CI).
- **Best coding practices = use existing system**, not parallel one: ruff/mypy, import-linter (`alerting="error"`), inline-import budgets, pre-impl grep + shared utilities (no re-derivation), every new `.py` gets unit test, slopbuster sweeps, iterate-with-codex on substantial changes.

## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `.venv/bin/python -m pytest tests/`
- Sci tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
- Atm matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py`
- Ocean matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py`
- AMIP: `.venv/bin/python scripts/run/run_amip.py`
- Dycore progression: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py` (SW -> hydrostatic -> non-hydrostatic ladder; `--only sw|hydro|nh`). Old `tests/validation/run_dycore_progression_suite.py` superseded — its per-case child scripts removed.
- GPU/MPI scaling: `.venv/bin/python scripts/bench/run_levante_gpu_scaling.py --grid cubed-sphere --mode strong` (`docs/performance/REAL_HARDWARE_SCALING.md`)
- Scripts reorganized into buckets: `scripts/{run,matrix,bench,plot,validate,data,experiment,cluster}/`; debug in `scripts/tmp/`. See `scripts/README.md` + `## File Layout` below.
- MPI tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/`
- MPI diff: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_mpi_differentiability.py`

## File Layout (audit — enforce on EVERY new file; no random files)
- **New scripts go in correct `scripts/` bucket — NEVER `scripts/` root or repo root.** Buckets: `run/` (prod drivers), `matrix/` (test-matrix registries), `bench/` (perf/profiling/scaling), `plot/` (plot/replot/regen), `validate/` (non-matrix validators/verifiers/conservation checks), `data/` (download/build/prepare forcing+IC), `experiment/` (init/reproduce/templates/machine-detect/fetch), `cluster/` (SLURM `.sbatch` job wrappers, e.g. `cluster/omip_nemo/`). Pick bucket by what script DOES. New bucket need real category, not dumping ground. See `scripts/README.md`.
- **Debug / throwaway / one-off → `scripts/tmp/` ONLY** (eventually deleted): `_*`-prefixed probes, `diag_*`/`diagnose_*`, per-iteration scratch. Never at `scripts/` root. `_probe_*.py` gitignored.
- **No new files dumped at repo root.** Root keeps ONLY: `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `FEDERATION.md`, `project_status.md` (generated), `pyproject.toml`/lockfile/dotfiles. Everything else has a home.
- **No `.md` notes accumulating at repo root → `docs/`.** Dev-notes, change logs, faithfulness/audit trackers (`*_faithful.md`, `*_checks.md`, review logs) live under `docs/`. Reference by BASENAME so code comments survive the move.
- **No runtime outputs in git.** `diagnostics/`, `logs/`, `**/logs/`, `results/`, `checkpoints/`, `output/`, `*.zarr`/`*.nc`, root `*.png`/`*.pdf`/`*.svg` gitignored. Visual-regression baselines stay in LOCAL working copies, regenerated on demand — not tracked. Never `git add -f` a runtime artifact. EXCEPTION: `.physics-validator/` is tracked on purpose — review logs, probes and banked oracle results cited as evidence by docs, tests and validation scripts (user 2026-10-24); only its error captures are ignored.
- **Source stays under `packages/<pkg>/legoesm/`** (federation namespace). New subpackage → update `tests/test_federation_plan.py` same PR.
- **Do NOT "flatten" `packages/<pkg>/legoesm/<subpkg>/` → `packages/<pkg>/` — double name is LOAD-BEARING, not redundant.** Editable installs (dev/test workflow) require on-disk path == import path, and hatchling REFUSES prefix-*adding* `sources` remap in dev mode (`ValueError: Dev mode installations are unsupported when any path rewrite in the sources option changes a prefix rather than removes it`; editables#20). So `packages/atmosphere/legoesm/atmosphere/` is only layout where `import legoesm.atmosphere` resolve under `pip install -e`. Flattening to keep `legoesm.*` build wheel but break editable dev; flattening to bare `import atmosphere` mean rewriting ~23k import sites + dropping namespace. Verified 2026-07-13 — don't re-attempt. `src/legoesm/` is SANCTIONED root meta-package (the `legoesm.*` TOP-LEVEL modules: `constants`, `config`, `cli`, `registry`, `dycore_factory`, …), same load-bearing reason — NOT stray `src/` to remove. "Never add source at `src/`" mean don't create NEW `src/` trees for component code; existing `src/legoesm/` stays.
- **Tests mirror package tree under `tests/`** (`tests/<component>/<tier>/...`). Curated dycore regressions in `tests/atmosphere/dycore/regression/`. No new `test_*.py` at repo root.
- Staging: explicit pathspecs, NEVER `git add .`/`-A` — catch stray scratch + concurrent-session files.

## Bug Triage
- Instability: CFL, boundary, metric, halo, pressure-gradient, diffusion, dtype.
- Conservation drift: flux form, weights, reductions, state updates before fixers.
- Differentiability: control flow, shape changes, side effects, checkpointing, nondiff branches.
- Perf regression: retrace, host callbacks, scatters, sharding, Python loops.
- Cross-backend: dtype, x64, unsupported kernels, comm semantics.

## Hygiene/Imports/Tests (audit)
- Every new `.py` ≥1 direct unit test importing+exercising leaf module (tendencies for physics schemes — not just integration via factory). No `__init__.py` re-export / `supported_matrix.py` / factory dispatch add without same-PR test. Block PRs growing untested-LIVE count.
- New config dispatch (Literal + factory in `integration.py`): test via public config.
- **New user-tunable config field → same-PR CLI flag in every affected run script.** Rule applies to `ExperimentConfig`/`DycoreConfig`/`OutputConfig` fields in `run_amip.py`, `OceanExperimentConfig`/`KPPConfig`/`VerticalMixingConfig` in `run_omip.py`, `MultiLayerLandConfig` sub-configs in `run_lmip.py`. "User-tunable" = any float/int/str/bool field user would reasonably change for sensitivity run (excludes: internal-only flags such as `debug_precision`/`diagnostics_perf_mode`, unimplemented stubs like `carbon_cycle` in AMIP, parallel-milestone-gated fields like `distributed_mode=spmd`). Each new flag also needs: (1) wiring in `build_config_from_args` / equivalent config-builder, (2) round-trip test in corresponding `tests/unit/test_run_*_cli.py`, (3) `validate_strict` membership check if it is a scheme Literal. FLOAT tunables with `__param_spec__` need no flag: all four MIP drivers take `--config` (dests, choices-validated) + `--params` (registry-qualified `scheme_key.field`, bounds-validated; atm via flattened-scalar map). Reachability machine-audited by `tests/unit/test_params_reachability_audit.py` against shrink-only `_params_reachability_baseline.py` (new spec'd tunable unreachable in its component's drivers go red) — the #691 "CLI flag gaps" debt closure.
- Removing module: also remove `__init__.py` re-export, `supported_matrix.py` entry, dispatch, test file, `__pycache__`.
- No deprecated backward-compat wrappers — update call sites. No thin dispatch-only wrappers (`X_utils.py` re-exporting `X.py`) — inline/factor. Real branching across callers (`land/stomata_utils.py`) legit. Grid variants legit when genuinely different numerics; indexing-only copy-paste forbidden.
- **No top-level cross-package imports from `core/` to `runtime/`/`parallel/`/`driver/`/`training/`/`experiments/`.** Why: `from legoesm.runtime.backend import ...` at top of `core/precision.py` triggered `runtime/__init__.py` → `runtime.precision` → `core.precision` mid-init, breaking isolated pytest. Use function-scope deferred imports.
- **No import of private (`_`-prefixed) symbols across modules.** Promote (drop underscore + `__init__.py` re-export) or factor public wrapper. Mutable singletons (`grids.halo._halo_backend`/`_mpi_topology`, `cubesphere_exchange._spmd_mesh`): use accessors (`get_halo_backend`/`get_mpi_topology`/`get_spmd_mesh`), never import global. Audit: `grep -rE "from legoesm\.[^ ]+ import [^,]*\b_[a-z]" src/legoesm/ packages/ | grep -v " as _"` = 0 (line-grep miss multi-line/function-scope/`_UPPER` imports); CI ratchet `tests/test_no_private_cross_imports.py` (AST, incl. function-scope; dunder-exempt; shrink-only allowlist EMPTY since 2026-06-10 — keep it empty).
- **Test-only modules MUST be acknowledged.** Not wired into factory/`__init__.py`/prod driver: (a) wire same PR, (b) move to `_future/` + docstring + xfail/skip, or (c) delete.
- **Never commit `docs/references/`.** Local research PDFs/extracts. Cite by filename/DOI. Notes elsewhere (`docs/ocean/experiments/`). Staging: explicit paths, never `git add .`/`-A`.
- Slopbuster periodic: `/slopbuster audit all` or `/slopbuster review`.
- High-priority untested LIVE (touch any → add test same PR): `ocean/experiments/global_overturning.py`, `ocean/physics/{bottom_drag,convection,surface_forcing,vertical_mixing}/output.py`, `ocean/physics/convection/enhanced_diffusion.py`, `ocean/physics/vertical_mixing/{k_profiles,mpas_integration}.py`. RESOLVED 2026-05-29 (direct tests added): `coupler/surface_energy.py`, `ocean/dynamics/barotropic_common.py`, `atmosphere/dynamics/sfno_pe.py`, `atmosphere/dynamics/tracer_transport_mpas.py`; `atmosphere/physics/convection/_triggers.py` + `timestepping/tridiagonal.py` already covered.

## Constants/Params (audit)
- **All physical constants in `src/legoesm/constants.py`.** New constant (T, ρ, c, L, k, μ, EOS coeff, Schmidt#, R_earth) MUST be added BEFORE use. No constants in `config.py`/fn bodies/test fixtures/plotters/notebooks even with `# = constants.X` comment.
- **`getattr(..., "X", <literal>)` fallbacks count as hardcoded.** Use `getattr(grid, "radius", constants.R_earth)`. Same `setattr`/`dict.get`/`kwargs.get`.
- **No hardcoded physical constants in fn sigs/bodies in `src/legoesm/`.** No `def f(g=9.80616, ...)` → `g: float = constants.g` or config NamedTuple. NamedTuple defaults SHOULD reference constants. Tunable scheme params (sigmoid sharpness, τ, drag) stay in config NamedTuple.
- **No hardcoded tunable params in physics bodies.** Sigmoid sharpness, τ, Louis coeffs, KPP epsilon, emissivity, drag → scheme `*Config` NamedTuple. Exempt: safety floors (`eps=1e-30`), math constants (`0.5`, `2.0`), category lookup tables (PFT `L_v` in `surface_params.py`) with doc.
- **No `273.15` for C↔K in prod.** Use `constants.T_freeze`. `T_freeze_ocean=271.35 K` in `ocean/eos.py` = only intentional exception.
- **Tests+scripts+plotters same rule.** `from legoesm import constants`. No `9.80616`, `7.292e-5`, `6.371e6` literals.
- **Sigmoid sharpness/transition widths in JAX hot loops forbidden as magic numbers.** Inside `scan_step`/`cond`: fn kwarg with doc default OR scheme `*Config` field. Ex: `compute_moist_adiabat(lcl_sigmoid_width_pa=100.0)`, `PlumeConfig.active_sigmoid_sharpness=1e4`, DM95 `transition_width_frac=0.1` on `dm95_taper`/`dm95_taper_scalar`/`_triad_taper`.
- **Saturation re-impl in forcing modules forbidden.** Use `legoesm.thermo.saturation_mixing_ratio`/`saturation_vapor_pressure`.

### Parameter hygiene — declare per category at file top + machine-readable tunable/fixed split (gated)
Two CI tripwires enforce this (extend, never weaken; baselines shrink-only): `tests/test_no_inline_physics_coeffs.py` (+`_inline_coeff_baseline.py`) and `tests/test_param_specs.py` (+`_param_spec_baseline.py`).
- **No inline empirical coefficient in physics function body OR signature default.** Across `*/physics/*`, `land/`, `ice/`, `coupler/`: every float literal (and int `|v|>16`) inside function scope flagged. Move to (a) scheme `*Config` NamedTuple field if tunable/scheme-defining, (b) module-level `_UPPER_SNAKE` constant/table block (published fits/tables: Sutherland, Hall-Pruppacher, Morel-Berthon, Jerlov, KK2000) with provenance comment if fixed published constant, or (c) real `# coeff-ok: <reason>` (reason REQUIRED) for genuine numerics one-off. Exempt: math {0,0.25,0.5,1,2,3,4,6}, exact conversions, `|v|<=1e-6` floors, `|v|>=1e20` guards, subscript indices, Pow exponents, small ints (indices/counts).
- **Per-file layout (consistent placement):** docstring → imports → `__physics_contract__` → `__param_spec__` → fixed `_UPPER_SNAKE` constant/table blocks (under `# --- <category> (<reference>) ---` comments) → `*Config` NamedTuple(s) with fields grouped by same category comments (snake_case + unit suffixes) → functions whose bodies read ONLY `cfg.<field>` / `constants.*` / exempt math. Kwarg-default literals → reference module constant or config field (Name in signature, not literal).
- **Every physics scheme `*Config` declares `__param_spec__`** (module-level pure dict literal next to NamedTuple). Per float field: `units`, `bounds (lo,hi)`, `tunable_tier`, `transform` (sigmoid/softplus/none), `category`, `reference`, `shape` (None or dim key like `n_pft` for variable-size array params), optional `legacy_name`. NEW config module must ship spec or be classified in `PARAM_SPEC_TODO` (shrink-only). Inclusion computed: only `:float`-annotated fields spec-eligible.
- **Tunable/fixed split (the continuum), classify SAME way every time:**
  - `tunable_tier 1` (**core**) = params already trained in practice / well-posed (surface albedos, ice strength `P_star`, gray optical depths, bulk exchange `C_H`/`C_E`).
  - `tunable_tier 2` (**extended**) = clear closure knobs with literature bounds (relaxation timescales τ, entrainment/drag/autoconv rate coefficients, thresholds, emissivity, roughness).
  - `tunable_tier 0` / `excluded` (with reason string) = NOT trainable: numerics floors/caps/regularisers, smoothing widths, measurement conventions (e.g. MOST 10 m), and anything **iteration-coupled** (mEVP `alpha`/`beta`, EVP `T_evp` couple to subcycle count).
  - Collector select tiers `1..N` (`build_trainable_params(config, tier="core"/"extended"/"aggressive", include=, exclude=)`); flip a param's status with 1-line `tunable_tier` edit. See [[param-hygiene-spec-effort]].
- **Loop-iteration COUNTS are never config/trainable** → module constant (e.g. `_N_EVP_DEFAULT = 120`), not config field, not kwarg-default literal. Structurally guaranteed: ints not spec-eligible, so iteration count can never reach trainable collector. See [[loop-counts-never-trainable]].
- **A tunable closure whose default is a `constants.X` reference** (e.g. `S_ice_new = constants.S_ice_bulk_default`) is *eligible* (may be `__param_spec__` param with explicit bounds + tier) though not *required* (AST gate won't force it). Expose genuine calibratable closures; keep environmental references (ocean salinity) fixed/excluded.
- **Trained values inject via config pytree, not new signatures:** `params.to_overrides()` → `legoesm.core.param_overrides.apply_param_overrides(physics_config, overrides)` (`NamedTuple._replace`) INSIDE the loss so leaves TRACED (SegmentForcing doctrine); production keeps static Python-float leaves (constant-folded, no retrace). Register newly-specced module in `param_collector.SPEC_MODULES` (drift-tested).

## Naming
- Surface T = `T_sfc` everywhere. No new `T_surface`/`Ts`.
- Driver/config schema field names match runtime field. New tunable: same name in `driver/config.py`, scheme config NamedTuple, YAML schema (consistent `hyperdiff_coeff`).
- **Same name + different units = bug magnet (audit).** Unit hint (`_C`, `_K`, `_s`, `_days`, `_m`, `_km`): use everywhere. New C-vs-K args MUST carry `_C`/`_K`. Config fields sharing base (`tau_*`, `T_*`, `c_*`, `C_*`) MUST have consistent unit suffixes OR distinct names.
- **snake_case all NamedTuple fields**, even capitalized symbols (CAPE, CIN, MSE, TKE). `SBMConfig.CAPE_threshold` → `cape_threshold`.
- **Renaming a tunable `__param_spec__` field = rename field AND its spec key together + add `legacy_name: <old>`.** Spec key must equal float field name (`test_param_specs`), so half-rename go red; `legacy_name` document prior name for any externally-saved tuned JSON. (2026-06-22: `CarbonConfig.T_opt→T_opt_C`/`T_width→T_width_C`, `StomataConfig.T_opt_jarvis→T_opt_jarvis_C`/`T_range_jarvis→T_range_jarvis_C` — Celsius fields gaining `_C` suffix.)
- **`tau_`/`τ` prefix valid for wind/wave STRESS [Pa], not timescales only.** Ocean wind stress and GWD launch stress (`tau_0`, `tau_max`) = standard GFD τ notation — do NOT auto-rename to `stress_*`. Real anti-pattern = MIXED units under one `tau_*` family inside single config without unit suffixes; fix by suffixing timescale members (`_s`), not stresses.

### Open naming debt
- `T_sfc`(368)/`T_surface`(59)/`Ts`(~6): coupler+`land/{multilayer_land,snow_budget,stomata_utils,slab_land}.py`, `ice/sea_ice.py`, `coupler/{accumulator,lake/two_layer_lake}.py` still `T_surface`; 3 files MIX BOTH — `driver/coupled_esm_driver.py`, `ice/sea_ice.py`, `driver/earth_system_driver.py`. Unify cleanup PR.
- `nlev`(4119)/`n_levels`(199)/`nz`(33): `nlev` dominates. Cleanup PR.
- `tau_relax`: RESOLVED 2026-05-29 → `KuoConfig.tau_relax_s`[s], `PhillipsTwoLayerConfig.tau_relax_days`[days] (matches `backscatter.tau_relax_days`). Keep unit suffix on any new relaxation-timescale field.
- `C_water`/`c_water`: RESOLVED 2026-05-29 → `SoilThermalConfig.C_water_vol`[J/m³/K], `LakeConfig.c_water_mass`[J/kg/K]. Keep `_vol`/`_mass` on new heat-capacity fields.
- `n_layers` overloaded: soil=`n_soil_layers`, ML=`n_hidden_layers`, reserve `n_layers` for atm/ocean vert.

## Dispatch (audit)
- **Every `scheme="..."` factory MUST `raise ValueError` on unknown.** Silent `else: <default>` mask typos+dead branches. Historical: `cloud_fraction.compute_cloud_properties` ran sundqvist on typo; `land/carbon/carbon_cycle.py:443` zero CO2; `ocean/biogeochemistry/carbon_cycle.py:108,209` silently disabled BGC; MPAS PV typos → enstrophy in `{compressible_euler_mpas,primitive_eq_mpas,shallow_water_mpas,ocean_pe_mpas}.py`; bulk-scheme typos → constant in `coupler.py:204`, `slab_land.py:156`, `multilayer_land.py:212`, `two_layer_lake.py:66`, `bulk_formulas.py:68`; `io/restart.py:232` silently wrote npz. HARDENED 2026-05-29 (now `raise ValueError`, validated at fn entry on static config): `carbon_cycle.py:step_carbon`, `coupler.py:ocean_tile_response`, `ice/sea_ice.py:_bulk_flux_dispatch`. HARDENED later — all now raise: `slab_land.py`, `multilayer_land.py`, `coupler/lake/two_layer_lake.py`, `bulk_formulas.py` (via `core/bulk_flux.py::validate_bulk_scheme`). HARDENED 2026-06-22 (atm MPAS `pv_scheme` typo→energy + thompson `snow_scheme` typo→bulk): `compressible_euler_mpas.py`, `primitive_eq_mpas.py`, `shallow_water_mpas.py` (now `energy`/`enstrophy` else-`raise`), `microphysics/thompson.py` (fn-entry `snow_scheme` guard). These are nested scheme-Config fields, NOT `ExperimentConfig` literals → factory/fn-entry raise is the defense, not `validate_strict`.
- Dispatch in `lax.fori_loop`/`lax.cond` (`coupler/bulk_flux.py:222`): validate at fn entry on static Python val, not traced body.
- **Nested scheme-Config dispatch raises too.** A `scheme`/`pv_scheme`/`snow_scheme`-style field in leaf `*Config` NamedTuple (NOT `ExperimentConfig` literal, so `validate_strict` never see it) MUST still `raise ValueError` on unknown — validated at fn/factory entry on static config value (matches `ocean_pe_mpas` sibling). Bare `else: <default>` here silently run different physics on typo. Lock each new guard in `tests/test_dispatch_hardening.py::BASELINE_DISPATCHERS` (grow-only) so it can't be silently deleted.
- Add membership-set assertions in `ExperimentConfig.validate_strict` for new scheme literals. RESOLVED 2026-06-09: `convection`/`turbulence`/`gravity_wave_drag` now HAVE validate_strict membership checks (`config.py:430-455`). Guarded + enforced going forward by `tests/test_validate_strict_coverage.py` (every scheme-like config field must be membership-validated or in its `KNOWN_UNVALIDATED` loader-validated allow-list; removing guard or adding unvalidated scheme field → red). Companion factory-level guard: `tests/test_dispatch_hardening.py` (existing unknown-scheme `raise` may not be silently deleted; 76-entry grow-only baseline).

## Common Mistakes
**NamedTuple fields**: verify actual field. `PhysicsOutput.precip` not `precipitation`. `hasattr` guard silently degrade. Adding field to `SegmentCarry`: update every call site. `grep -rn "SegmentCarry(" --include="*.py"`.

**Land/face masks (latlon C-grid)**:
- Never `state._replace(land_mask=...)` on `LatLonCGridOceanState` without updating `u_mask`+`v_mask`. Stale face masks → mass flux through walls → silent leak.
- Preferred: `land_mask_override` to `rest_state_latlon_cgrid_ocean()` at construction.
- Post-construction: `replace_land_mask(state, new_mask)` from `init_latlon_cgrid.py` — atomic update 3 masks.
- `_assert_runtime_invariants` (gated `enable_runtime_checks`) catches inconsistencies.

**JIT/compilation**:
- Never build closures in training loops. `build_segment_fn` create new fn per call → inside `for epoch`/`_loss_fn` recompile each iter. Build once outside; pass changing vals as args.
- Helper fns inside `lax.scan` body: Python defs in `_single_step` recreated each trace. Move to module scope.
- Dead code from iteration: when refactoring, grep for vars assigned never used.

**SegmentCarry**: canonical hot-loop state. Adding field cross-cutting: NamedTuple def, `pack_carry`, `unpack_carry` docstring, per-step Python ref loop in `test_compiled_segments.py`, `test_scale_tpu_compat.py`, `test_scale_jit_health.py`, direct `SegmentCarry(...)` in validation tests. New diagnostic fields (`max_cfl`) reset to zero at segment start, not accumulated.

## Assets
Specialized agents in `.claude/agents/` for dycore, validation, differentiability, physics, land/ice, scalability.

## Response Style

### RULE -3 — BREVITY IS THE DEFAULT, LENGTH IS OPT-IN (2026-08-18, FIFTH callout)
User: *"be succinct and clear - make this a default."* Rules -2, -1 and 0 all
framed brevity as a CAP to check before sending, and I wrote to the cap every
time. Inverted:

**DEFAULT REPLY = <=3 LINES. NO HEADER. NO TABLE. NO BULLETS.**

Length is earned, not assumed. It is unlocked ONLY by an explicit ask
("report", "walkthrough", "long version", "what is left") or by numbers the
user will genuinely compare (a table of >=3 rows AND >=2 columns). Bold
headers, per-item verdicts, before/after lists and caveat lines are
REQUEST-ONLY formats.

Mechanical check: write the reply, delete everything after line 3, and send it
if it still answers. If it does not, the content wanted a table — keep the
table and delete the prose.

Background-task notifications get NO reply at all unless a result is ready, a
claim was refuted, or a decision is needed.

### RULE -2 — MECHANICAL LENGTH LIMITS (2026-08-14, THIRD callout)
User: *"be more succinct and clearer - make this a rule."* RULE -1 and RULE 0
below already said this and I still shipped multi-section replies with tables
and bold headers on routine results. Prose rules did not work, so these are
COUNTABLE. Check them before sending.

- **5 lines of prose, hard.** Not 5 sentences — 5 rendered lines.
- **ONE bold header per reply, or none.** Multiple `**Headers**` = a report,
  and reports are only for when one was asked for.
- **A table needs >=3 rows AND >=2 columns of real data.** Two numbers go in a
  sentence. A table of one comparison is decoration.
- **No line explaining method, discipline, or what a control proved.** The
  result only. "Pre-registered X, got Y" is one clause, not a paragraph.
- **No restating a caveat already in the commit.** The commit is the record.
- Status/job IDs: one line total, at the end, no formatting.

If the content genuinely needs more, say "long version?" and stop.

### RULE -1 — THE CAP APPLIES TO GOOD NEWS AND BAD NEWS ALIKE (2026-08-12)
User, again, after a session of correct-but-long replies: *"be succinct and
clear."* The rule below was being followed for status and ignored for
findings — a retraction, a root cause, or a self-caught instrument bug is
NOT a licence to write six paragraphs. Length is not proof of rigor.

- **A finding is ONE line: the corrected fact.** Not the discovery story,
  not what it means for three other workstreams, not a list of what still
  stands. "X was wrong; the real number is Y" and stop.
- **Never re-explain a thing already said once in the same reply.** If a
  number appears in the verdict, it does not reappear in the evidence.
- **Cut every sentence that exists to show diligence.** "I checked", "the
  control caught it", "this is the Nth time" — delete. The tool calls are
  visible; the user is not grading effort.
- **Detail goes in the commit, the memory file, or the peer message —
  never the reply.** That is what those artifacts are for.
- Default reply: **3-5 bullets, under ~60 words.** A long reply must be
  ASKED for ("report", "walkthrough", "what is left").
**STRICT RULE, EVERY SESSION, EVERY MODEL (Opus included): be succinct AND
clear.** Not style preference, not default that decay over long session.
Succinct = answer first, ~60 words, bullets not paragraphs (RULE 0). Clear =
plain words a colleague outside this repo can act on, no bare identifiers (RULE
-1). Both must hold; reply failing either one = failed reply.

### RULE -1 — CLARITY IS THE HARD RULE. IF THE USER CANNOT FOLLOW IT, IT FAILED.
User, 2026-08-12 (and 2026-08-07, 2026-08-11 — same complaint every time):
*"Ensure you are clearer — I have no clue what you are saying most of the
time."* Outranks brevity: short reply nobody understand worse than no reply.
Brevity already being followed when this said; defect is UNEXPLAINED INTERNAL
DETAIL, not length.

MECHANICAL TEST, apply to every sentence before sending: could a colleague who
know climate modelling but never opened this repo act on it? Need file name,
function name, job id, or flag to make sense → REWRITE IT.

- Say the THING, not the SYMBOL. "the model runs radiation 18 times more often
  than intended", not "split_rad=False makes rad_update_interval_steps inert".
- NO identifiers in a reply: no file:line, no function names, no config keys, no
  job ids, no PR numbers, unless user asked for that exact thing. They belong in
  commit message. Number user should act on is fine.
- One idea per line. Line has clause explaining clause → split it.
- State CONSEQUENCE first, cause second, stop. Not mechanism, not how it was
  found, not who found it.
- Never write sentence whose subject is piece of code. Subject = model, run,
  campaign, number, or user's decision.
- Something wrong → lead with what now false and what to do. Not narrative of
  discovery.
- Question to user = numbered options in plain words, with your pick.

FAILURE PATTERN, all three callouts: long autonomous stretches. Each status
reply drifted back into repo-internal vocabulary ("the pin", "the waiver", "the
factory", "the skeleton") meaning nothing outside this session. Re-read this
section whenever session run long, and before every status report.

### RULE 0 — HARD CAP ~60 WORDS. ANSWER FIRST. STANDING ORDER, ALL SESSIONS.
Five callouts in two days (2026-08-07/08). Not style preference — a cap. Lead
with result. 3-5 bullets. Stop. Job IDs, caveats, file:line and reasoning go in
COMMIT, never the reply. Retraction = ONE sentence plus corrected fact. Long
replies only when report explicitly requested.

User, 2026-08-07, after repeated callouts in single session ("you are too
verbose", "I have no idea what you are saying", "just laser-focused summary"):
**"Make being succinct, clear and to the point a strict rule for all future
sessions."** Outranks every other formatting instinct.

Mechanical test before sending — line fail → cut it:
- Would user act differently without this line? No → delete.
- Is it method, process, or what I tried? → delete. Tool calls visible.
- Is it caveat nobody would act on? → delete (put in commit).
- Is it re-explaining something already said once? → delete.

Shape: **verdict first**, then only evidence that change the verdict, then ONE
question with numbered options if decision needed. Default length few lines.
Long reply must earn it by being asked for (report, walkthrough, per-phase
notes).

Jargon banned unless sentence also say what it mean in plain words. Say "how
much communication the split costs", not "the ppermute round count".

### PLAIN LANGUAGE FIRST — the rule that finally worked (2026-08-07)
THIRD callout: *"I really don't understand when you talk to me. Things are
incredibly unclear."* Brevity alone did NOT fix it — earlier rules were being
followed. Real defect = **unexplained jargon** and leaving user to infer the
decision. What worked, now required shape:

- **Write for colleague, not reviewer.** Short sentences. Ordinary words.
- **NO unexplained domain jargon.** Terms like *ppermute, max_degree, halo fill,
  production-exact, VJP, chromatic bound* banned unless sentence also say what
  they mean in plain words. Prefer plain phrase outright: "how much
  communication the split costs", not "the round count of the comm graph".
- **Use short LABELLED blocks**, bolded, 1-3 lines each:
  **What I built / What broke / What I was wrong about / What's safe /
  What I need from you.** Labels do the navigating so user not.
- **End with ONE explicit question and numbered options** when decision needed.
  Say which you would pick and single deciding factor.
- **Asked to compare options → answer the comparison** — what each buy, what it
  cost, and question that decide it. No re-describing options.
- **Own errors in one plain sentence, at top.** "I was wrong earlier: X is not
  guaranteed." No narration of how discovered.
- **No tables, no file:line, no job ids in reply** unless asked. Those go in
  commit message. Reply is for the decision.

### Earlier callouts (same session) — still in force
**SECOND callout (2026-08-06), because rule below was written then ignored:
_"stop being verbose. It is really hard to understand. be more direct, to the
point, clear about issues. Bullets summarizing."_ Plus: _"aren't you using
caveman?"_ — terse mode was ACTIVE and I still wrote essays.**
- **BULLETS BY DEFAULT.** Prose paragraphs = failure mode. One line per fact.
- **Lead with the issue.** Not how it was found.
- **Delete every sentence not changing what user does next.**
- **Never re-explain caveat already stated once.**
- **No near-miss stories.** "I almost got X wrong" is not finding. State
  corrected number, move on.
- Terse mode (caveman/ponytail) active → APPLIES TO WHOLE REPLY — including
  findings, status, caveats. Length not substitute for rigor.

User callout 2026-08-06 (earlier): *"you are quite unclear... provide more succinct,
clear summary, clear choice, do not make many but targeted and verified
assumptions."* Evidence reader has to assemble into conclusion is not a report.
Structure, this order, then stop:

1. **VERDICT first, <=2 lines.** What true / what happened. Never open with
   method, caveats, or narration of what was run.
2. **THE DECISION, if any: ONE recommendation.** Name option you would take and
   why, one line. Menu of options with balanced caveats push work back onto
   user — only list alternatives when they genuinely must choose, and even then
   say which you'd pick.
3. **EVIDENCE: only what change the verdict.** Decisive number, file:line, or
   measurement. Not everything checked.

- **AT MOST ONE unverified claim per response, explicitly labelled PLAUSIBLE.**
  Everything else verified before stated. No enumerating candidate causes — pick
  the one you tested, report it. Untested hypotheses = clutter reading as
  findings.
- **Do not narrate the process.** Tool calls already visible. Report outcome,
  not itinerary.
- **Table only for >=3 things compared on >=2 axes.** Otherwise a sentence.
- **Retract in one line, move on.** No re-litigating superseded claim.
- **Caveats: only those changing what user should DO.** Limitation nobody would
  act on belong in commit message, not reply.
- Numerics change -> state effect on stability, accuracy, conservation,
  differentiability. No guesses as facts.
- **No Read images** unless user asks; report path.

**TERSE. Caveman register (user, 2026-08-06: "You speak too much... no need to waste tokens").** Fragments OK. Drop articles/filler/hedging/pleasantries. No narrating what you about to do, no restating request, no re-explaining finding already stated. Prose is for FINDINGS, not for process.
- **ALWAYS end with findings summary** — table or bullets: what measured, the number, CONFIRMED vs PLAUSIBLE, what still open. That summary = deliverable; rest is scaffolding.
- Long verbatim tool output → quote only DECISIVE line (`N passed`, failing assert, peak value).
- Commits/PRs/code comments/security warnings stay full English.

# iterate-with-codex agent
1. Implement change
2. `/codex:adversarial-review --wait`
3. Parse output
4. Fix flagged
5. `/codex:review --wait` again
6. Issues remain → 4
7. Stop when clean or after 30 iter
exec
/usr/bin/bash -lc "sed -n '320,700p' CLAUDE.md" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
use the majority of votes. Just ask me big picture / important questions such
as on design or main directions."* Every side decision above is still never
made silently, but a DETAIL choice (tolerance, guard, edge case, test value,
cache key, refusal scope, error-vs-fallback) is put to GLM
(`mcp__zai__ask_glm`) and codex, batched in one prompt each, plus the author's
own vote; majority of the three decides. Record each choice with its three
votes in the commit/PR body. The USER is asked only about design, main
direction, scope, and anything they named as their call. Precedence: this
replaces "ask the user" for detail choices everywhere in this section and in
the STOP-AND-ASK list; for design/direction the user rule stands unchanged.

MECHANICAL CHECK, end of every task: print list of choices made, each one line,
marked ASKED (user), VOTED (GLM/codex/author majority, votes shown) or UNASKED. Empty UNASKED list = target. Anything on UNASKED list
offered for revert in same message. Worked example, five from 2026-08-21 that
prompted this rule: routing each case's droplet number into single-column arm;
allocating zero graupel field in every column running microphysics; which
campaign cases got registered and using ARM cumulus case as stand-in for another
campaign's regime; pinning forcing downloads to fixed versions and checksums;
turning missing variable name into hard error. Every one defensible. None asked.
That the error.

**2026-11-03 REFINEMENT — DETAIL DECISIONS GO TO A VOTE, NOT THE USER.** User,
repeated three times in one day: *"For detailed questions do not ask me but
check with him [GLM] and codex and use the majority of votes. Just ask me big
picture / important questions such as on design or main directions."*
- **Detail side decision** (names, tolerances, guards, edge cases, file
  locations, minor scope trims, which diagnostic next): author's pick + GLM vote
  + codex vote, majority of three wins. Never silent: record each voter's pick
  in the commit, and mark the item VOTED on the end-of-task list (VOTED counts
  as discharged, like ASKED).
- **Still ASK the user:** model design, main direction, production defaults and
  config changes (RULE 3), which experiments/compute to launch, data source,
  what is trained or scored, anything changing comparability of results.
- Unsure which class → big picture → ask.
- Subagent briefs must state this rule.

## Operating Mode
- **TERSE BY DEFAULT.** Drop articles/filler/pleasantries/hedging; fragments fine. Report `[thing] [state] [next]`, not prose. No restating what was just done, no feature tours, no explaining a simplification at more length than the code. Numbers/tables over narration. Full prose ONLY when asked for it (report/walkthrough), or for security warnings, irreversible-action confirmations, and multi-step sequences where fragments risk misread. Code/commits/PRs/docstrings: written normally.
- **DO WHAT WAS ASKED — DO NOT EXTRAPOLATE.** Deliver requested thing, then STOP and report. No inventing adjacent work because it seem useful: no unrequested helper/launcher/benchmark scripts, no extra files staged "for later", no speculative refactors, no scope widened from "fix X" to "also improve Y". Adjacent work look warranted → NAME it in one line, let user choose. Do not pre-build. Uncertain whether in scope? It not; ask. Over-delivery not helpfulness: cost review time, bury actual change, create artifacts nobody vetted.
- **Irreversible / outward-facing actions need explicit permission EVERY time**: rewriting published history (`push --force*`), amending/squashing pushed commits, deleting or overwriting files the user created, `git reset --hard`, closing/merging PRs, installing or upgrading system packages, reboots. Approval for one such action does NOT carry to the next. Propose, then wait.
- Nontrivial task: short plan before edit. Read nearby impl+tests first. Ambiguous numerics/physics/API: ask.
- Minimal diffs. No unrelated refactor in bug fix.
- **DUAL adversarial review is the DEFAULT for EVERY substantive change — TWO
  independent reviewers, never one** (user directive 2026-08-12, restated
  STRICT 2026-08-13, again 2026-08-20 as "make this the default behaviour").
  Author NEVER review own code. Route by who WROTE code — two reviewers always
  the other two:
  **Claude-authored → codex + GLM-5.2** (`mcp__zai__ask_glm`);
  **GLM-authored → codex + Claude**;
  **codex-authored → Claude + GLM-5.2**.
  Both BEFORE the PR, not after; report both verdicts in PR body and status
  line. Subagents doing implementation must be told to run BOTH — they default
  to codex only. Applies to measurement harnesses and probes too: instrument
  decide what we believe.
  Why two: they catch DIFFERENT classes. Codex find diff defects (broken
  contracts, vacuous tests that cannot fail, silent fallbacks, binding gates
  accepting wrong arm). GLM find MECHANISM defects (wrong objective, wrong
  regime, lever whose premise receipts already falsified — it retracted own
  top-ranked lever once measurement contradicted it).
  Reviewer disagreement = SIGNAL, not noise: name disputed point and the
  measurement that discriminates, run it if cheap, never average the two.
  A REVIEWER'S FINDING IS A HYPOTHESIS, NOT AN INSTRUCTION: measure before fix.
  2026-08-20, GLM predicted rounding error 1.4e-5 that would refuse valid
  restart files; measured over every column model builds it was 7.9e-8, so guard
  added for it was DELETED and measurement pinned instead — unmeasured finding
  buy knob that never bind.
  **"Substantive" includes MERGE CONFLICT RESOLUTION**, and rebases,
  cherry-picks, back-ports, probes, test fixtures. Conflict resolution =
  highest-risk case in list: someone chose between two versions of same code and
  diff not record what was discarded. FAILURE 2026-08-20: merged 17-day-old
  branch, hand-resolved 14 conflicts across 7 files including two independent
  fixes for one defect, shipped unreviewed until asked.
- **Codex adversarial review MANDATORY after any major code implementation/change.** Trigger: new module/feature, dycore/physics/parallel/ocean/land/ice/coupler/training edit, >~50 LOC, multi-file, or anything touching numerics/AD/JIT/pytree/conservation. Run **iterate-with-codex agent** loop below (`/codex:adversarial-review --wait` → fix flagged → `/codex:review --wait` → repeat until clean or 30 iter) BEFORE declaring done; report that review ran + verdict.
  **Review SUBAGENT dies (spend limit, API error) = NOT review waiver — codex
  CLI is separate binary with separate credentials, usually still reachable:
  `codex exec --sandbox read-only -C <repo> "<prompt>"` (`which codex`,
  `~/.codex/auth.json`). Try CLI directly before ever proceeding unreviewed. BOTH
  unavailable → say "UNREVIEWED" in every status until one succeed.**
  2026-07-26: subagent hit monthly spend limit, many iterations ran unreviewed
  while CLI worked fine whole time. Exempt: trivial/mechanical edits (typo,
  comment, rename, doc/markdown/`.tex`-only, single config value).
- **Pre-impl search mandatory**: before new fn/helper/class/operator/diagnostic/init/load/loss/numerical routine, grep `src/legoesm/` for similar names/docstrings/formulas in `thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`. State searched+found. Similar exists → extend/factor.
- **Shared utilities — never re-derive** (prod, scripts, validators, plotters, tests, notebooks, probes):
  - Constants: `from legoesm import constants` → `T_freeze`, `R_d`, `c_pd`, `L_v`, `R_v`, `epsilon`, `g`, `p_ref`, `kappa`, `sigma_sb`, `T_freeze_ocean`. No literals `273.15`/`287.0`/`1004.64`/`2.501e6`/`461.51`/`0.622`/`9.80616`/`6.371e6`/`7.292e-5`.
  - Saturation: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice`. No re-impl Tetens/Magnus/Clausius–Clapeyron (plotters incl). Why: re-derived `e_sat=611.2*exp(17.67*Tc/(Tc+243.5))` diverged from model → false supersat in CI.
  - Column integrals: `legoesm.diagnostics.column_integrals` (`column_water_vapor`). No inline `jnp.sum(q*p_s*dsigma)/g`.
  - Losses: `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`).
  - Optimizer: `ml/training.create_optimizer()` (warmup+cosine+clip).
  - SCM-RCE gradient tuning: reuse `scripts/run/run_scm_rce_campaign.py` for CRM
    reference extraction / SCM evaluation and `legoesm.training.scm_rce_metrics`
    for normalized profile score. No duplicated RCE profile numerics.
  - SCM-RCE param training defaults to MUON via `ml.training.create_optimizer()`,
    init from `results/scm_rce_campaign/tuned_parameters.json`, writes recommended
    trained JSON under `results/`, never mutates production `*Config` defaults.
    Apply trainable overrides inside loss so leaves traced; static frozen leaves
    stay outside.
  - Every new `.py`, including `scripts/run/*.py` drivers, gets direct test.
    Scheme/factory dispatch must raise on unknown selections.
  - Atm column (h, ρ, virtual T): `atmosphere.physics._shared`.
  - Ocean EOS/pressure: `ocean.eos` (`compute_ocean_rho`, `compute_ocean_rho_and_pressure`).
  - SFNO: `ml/sfno.py`. No new neural op archs in training.
  - Channel packing: `ml/channel_packing.py` (`PE3DChannelSpec`, `pack_pe_state`, `unpack_pe_output`).
  - Ocean baroclinic (#214): `ocean/dynamics/ocean_tendency_common.py` (`iterate_eos_and_pressure_anomaly`, `apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`, `implicit_bottom_drag_factor`) in new `ocean_pe_*.py`.
  - Ocean barotropic (#214): `ocean/dynamics/barotropic_common.py` (`compute_filter_weights`, `bebt_blend`, `maxvel_clip`) in new `barotropic_*.py`. `tests/ocean/unit/test_no_scheme_duplication.py` enforces.
  - Plotters NOT exempt. Use model helpers for q_sat, RH, ρ, virtual T, MSE.
- No duplicate numerics across dycores/physics/grids/tests. Indexing/naming-only copy-paste forbidden.
- **No laziness on hard/large code** (>100 LOC, multi-component, full operator chains): no `pass`/`NotImplementedError` stubs, no partial-called-done, no skip edge cells/boundary halos/corner stencils/non-duogrid/MPI-sharded/AD-VJP. No happy-path-only tests. Too big → say so, list remainder, quantify risk.

## COMPARING TWO CONFIGURATIONS = A SIDE-BY-SIDE TABLE, ALWAYS
User directive 2026-08-21. Any time two runs / configs / templates / lanes are
compared, give a table: one row per field that DIFFERS, one column per
configuration, named in the header. Omit fields that agree — it is a diff, not
an inventory. Prose comparisons hide the one line that matters; the land
coupling week was lost to a surface-scheme mismatch a five-row table exposes at
a glance. This is also the required answer to "why did A work and B not".
If the table has more than one row, no single output difference is attributable
yet — say so.

## ACCURACY FIRST — A CLAIM THAT DRIVES A CODE CHANGE GETS REVIEWED BEFORE THE CODE
User directive 2026-08-21, after a week was spent coupling the WRONG land model
and then chasing two wrong causes in a row: *"we focus now on accuracy and we
must check with adversarial reviews our claims, especially when they imply
coding decisions."*

The existing dual-review rule covers DIFFS. This one covers the CLAIM that
justifies the diff, and it comes FIRST:

- **State the claim, then get it reviewed, THEN write code.** If a sentence of
  the form "X causes Y, so I will change Z" is about to become an edit, it goes
  to codex and GLM as a claim, before the edit exists. Both. A claim reviewed
  only after the code is written is a claim defended, not tested.
- **Say which parts are MEASURED and which are READ OFF THE CODE.** A scaling
  argument, a mechanism, a "this is the only consumer" — each is a separate
  claim with its own evidence.
- **Quote the number's PROVENANCE next to it.** "62 K leaf temperature" from a
  synthetic sweep that deliberately includes unreachable forcings is not the
  same statement as 62 K in a run. Report the reachable-subset number and the
  full-box number separately, and say which is which. A frightening number with
  no provenance wastes the reader's attention and can send the work sideways.
- **A number a well-posed scheme could never produce is a BUG IN THE
  INSTRUMENT until proven otherwise.** Check the probe before reporting the
  physics.
- **When a working configuration exists, diff against it FIRST.** See
  [[reuse-the-working-path-dont-rebuild]]: the offline land driver was healthy
  the whole time; the coupled lane simply called a different parameter provider.
  The first question on any coupling defect is "what does the working lane do
  differently", not "what could be wrong with the physics".
- **Reviewer disagreement is the signal to MEASURE, not to average.** Name the
  discriminating check and run it; it is usually a grep.

## Epistemic rules (non-negotiable)

### Never infer an API — read it
Before calling any function from JAX, Equinox, Optax, Diffrax, jaxKAN, or any
other dependency: grep installed source in site-packages, read actual signature.
No reconstruct from memory. Applies to argument names, argument order,
keyword-only args, return arity.

Symbol under `jax.experimental.*` → assume API changed since training data.
Verify or search. No guessing module paths.

### Report uncertainty explicitly
End any non-trivial code response with `UNVERIFIED:` block listing:
- APIs used but not read from source
- assumptions about library versions or runtime behavior
- anything that would silently produce wrong numbers rather than an error

Empty block = valid answer. Missing block = not.

My premise wrong — bug misdiagnosed, or thing asked for won't work — say so
before writing code.

### Diagnose before patching
Something fail: state candidate causes and how to discriminate between them,
then test. No going straight to fix. No agreeing with cause I suggested unless
evidence support it.


**Applies to THIS repo's own API too** — large enough that memory unreliable.
FAILURES in ONE session (2026-07-30): `AerosolConfig(reference_aod=)` (really
`reference_aod_550`), `McFarlaneConfig(N_ref=)` (no such field),
`run_amip.build_parser` (really `build_arg_parser`), `from legoesm.grids import
create_grid` (really `legoesm.grids.factory`). Each cost full probe round-trip.
Worse, `RRTMGPConfig()` default `include_clouds=False`, so offline harness
omitting it return CLEAR-SKY fluxes and EVERY cloud gradient exactly 0.0 —
silently wrong number, not error. Read NamedTuple `_fields` / `_field_defaults`
before constructing config.

User's premise wrong — bug misdiagnosed, or thing asked for won't work — say so
BEFORE writing code.
## Verification (JAX-specific)

Code not done until it run. Correctness claims require output.

- **Shapes/dtypes**: check with `jax.eval_shape` before running anything
  expensive. Cheap, catch most errors.
- **Gradients**: any new `custom_vjp`/`custom_jvp`, adjoint, or hand-derived
  derivative must pass `jax.test_util.check_grads(f, args, order=2)` before you
  claim it work. Gradient that run is not gradient that correct — single most
  common way to ship silently wrong result here. (Applies directly to
  `_sendrecv_vjp` in `halo_exchange.py` and any new MPI-AD path.)
- **jit parity**: run function eager and under `jit`, compare outputs.
  Divergence = tracer bug (Python-side branching, `.item()`, `if` on traced
  value, host callbacks).
- **Sharding**: verify with `jax.debug.visualize_array_sharding` or by printing
  `.sharding`, not by reasoning about what annotation should do.
- **Numerics**: default float32. State tolerance you compare at. No `==` on
  floats. Test need float64 → say so explicitly, no silent `jax_enable_x64`.
- **donate_argnums / buffer donation**: never add without confirming donated
  buffer not reused. Fail silently or crash far from cause.

## Scope
One change at a time. No refactor of adjacent code, no rename, no "improve" of
code I didn't ask about. Long unbroken generations drift into invention — prefer
small verified diff over large plausible one.

**DO NOT EXTRAPOLATE. Do only what was asked** (user directive 2026-08-06).
The ask = deliverable, not starting point to reason outward from.

- Related-looking problem you notice = ONE-LINE report, not work item. Name it
  and stop; do not start it.
- No widening scope because fix "would only be complete if" something adjacent
  also done. Ship the ask; state the boundary.
- New artifacts (scripts, benches, plots, panels, public APIs, config knobs)
  only when asked or genuinely required to finish the ask. Unsure whether
  required → it not; ask in one line.
- No proposing or launching compute user did not ask for.
- FAILURES 2026-08-05/06: scoped 128-GPU coupled ladder nobody requested off a
  question about existing plots; added coupled panel to figure when asked to
  assess the figure; wrote probes and helper scripts for questions never posed.
  Each cost round-trip and buried actual answer.

## Attribution Gates — MANDATORY, each from a real 2026-07 failure
Model near operational. Every rule below mechanical: satisfy it or state
explicitly that you did not. "I was careful" is not compliance.

- **PROVE THE PATH EXECUTES before blaming a line.** Naming file:line as cause
  require showing that line run in configuration under test: print ENCLOSING
  FUNCTION (`awk` nearest `def` above it), confirm active lane/driver call it.
  FAILURE: blamed positivity clamps at `model_driver.py:10923` for century's
  water source; they live in `_run_per_step` while century run `_run_column`,
  which contain no moisture clamp at all. Fix nearly written for lane the run
  never touch. Same class as reading entry point instead of full path.
- **REUSING A REFERENCE IMPL MEANS PORTING ITS EXCLUSIONS, not just its
  formula.** State which of reference's guards/scope conditions you kept and
  which dropped, with reason for each. FAILURE: copied
  `spectral_les_moist.conserving_positive` but not its `n_water` split, so
  column-conserving borrow applied to number concentrations
  (`N_c`/`N_i`/`N_r`) — unphysical, and it fed M2005 deposition (~N_i^(2/3)),
  producing fake "accelerating dry bias" reported before being caught.
- **A TEST THAT INSPECTS SOURCE MUST NAME THE SYMBOL THAT RUNS, and must be
  shown to FAIL when feature removed.** `inspect.getsource(X)` assertion where X
  is delegating wrapper pass while proving nothing. FAILURE: asserted against
  `MPASPrimitiveEquationModel.step`; floors are in `_step_jit`.
- **TOOL STATUS IS NOT EVIDENCE — read output tail.** Exit code without tool's
  own success line (pytest's `N passed`, "COMPLETED in Xs") = UNVERIFIED; OOM
  kills and timeouts can surface as success. FAILURE: reported regression suite
  green on exit-0 that was actually `Out Of Memory` mid-run. Quote decisive line
  when claiming suite passed.
- **EVERY BASELINE/ALLOW-LIST REASON STRING IS A CLAIM — verify in code before
  writing it.** Plausible-sounding reason permanently hide real defect. FAILURE:
  classified `convective_buoyancy_death_memory` as "carried in SegmentCarry" (it
  not — leaf `BechtoldConfig.buoyancy_death_memory` exist and nothing map to
  it), asserted `SBMConfig.precip_efficiency` leaf that not exist, and credited
  `micro_substeps` to consumer that read `args.`, not the config field.
- **RATE / TENDENCY / SKILL COMPARISONS: identical windows on BOTH sides, and
  print window next to number.** Differing spans = confound, not result.
  FAILURE: TCW over days 190-530 vs CMOR year 1 gave "+38.7 kg/m2/yr"; matched
  windows gave +13.4. Extends existing controlled-comparison rule to derived
  rates.
- **A DIAGNOSTIC'S PRINTED PRECISION BOUNDS THE RATE YOU CAN CLAIM.** Log CWV at
  0.1 kg/m2 over 8 days resolve only ~±4.6 kg/m2/yr — no reporting trend inside
  one quantum. Prefer fp64 from model state (checkpoints) over parsed log lines.
  Same class as throughput-quantization error.
- **`JAX_ENABLE_X64=1` on any numerics/conservation test.** fp32 mismatch NOT a
  failure until re-run with x64; and *new* failure not yours until reproduced
  with your change stashed. Do both before reporting regression.
- **RUN-TARGET PARAMS ARE ABSOLUTE (`TARGET_DAYS`), and "latest checkpoint"
  MOVES.** For controlled pair, COPY pinned checkpoint into each arm dir; never
  use `PREV_CKPT_DIR`-style newest-wins pointer while another run advancing.
  FAILURE (twice): arms exited instantly at "Already at/past target".
- **A LAUNCHER FLAG THAT SWITCHES ONE FORCING CHANNEL MUST SWITCH ALL OF THEM.**
  Verify resolved paths in run log, not the flag you passed. FAILURE:
  `CENTURY_DECK=1` set era-correct ozone+volcanic but left 1979-2016 SST.
- **PROSE IS A POINTER, NEVER A CITABLE FACT.** Code comment, docstring,
  `AMIP.md`/`docs/` entry or "Known issue" naming a limitation, guard, or
  missing feature MUST be re-verified in CURRENT code at point of use before
  repeated as finding — this repo routinely fix things without updating prose.
  Comment name module imposing guard → OPEN THAT MODULE. FAILURES (2026-07-30,
  three in one day): quoted `_run_column` "turbulent surface fluxes are
  intentionally NOT applied" comment and `AMIP.md` Known #3 to claim MPAS has no
  turbulence — MPAS turbulence path exist (`turbulence/integration.py`, Perot
  edge->cell) and run resolve `turbulence=louis` + `surface_bulk_scheme=coare3`;
  and doubted a cloud_fraction comment that was exactly right.
- **THE FIRST GUARD YOU FIND IS NOT THE ONLY GUARD — follow value to its
  CONSUMER before declaring it unclamped/unchecked.** FAILURE: reported "no
  upper bound on r_eff" from `rrtmgp.py`'s `clip(x, 1e-6, None)`; real clamp to
  lookup-table range is one call deeper in `rrtmgp/optics/cloud_optics.py`. Same
  class as blaming a line without proving its enclosing function runs.
- **BEFORE ATTRIBUTING A BIAS TO A COMPONENT, PROVE THE COMPONENT EXISTS AND
  RUNS IN THE CONFIG UNDER TEST.** ABSENT component and BADLY-TUNED one give
  SAME symptom, and every tuning arm against absent component return null.
  Enumerate, from RESOLVED config and LANE'S code path (not deck's comments):
  the scheme, its PROGNOSTIC STATE, its inputs. No prognostic state -> say so
  out loud; that usually the finding.
  Corollaries, each earned:
  (a) **A `false` on a component switch does not mean "the simpler version" —
  it may mean NOTHING.** Check what lane actually fall back to.
  (b) **A DECK COMMENT NAMING A FALLBACK IS A POINTER, NOT A FACT** (prose rule,
  applied to configs): grep named symbol in that lane's setup.
  (c) **When N different schemes for component A all fail to move a bias, STOP
  TUNING A.** Limiter upstream, or A's inputs wrong.
  (d) **A parameter wrong at BOTH ENDS of a physical range is a MISSING
  DEPENDENCE, not a mistuning** — thing it should depend on not read.
  FAILURE 2026-08-15, most expensive of campaign: MPAS AMIP deck set
  `use_multilayer_land: false` with comment "this deck runs the slab". There is
  NO slab land on MPAS lane (`slab_land_active` wired only on cube/general
  path), so runs had **no land surface model at all** — land T_sfc was nearest
  OCEAN's prescribed SST minus 6.5 K/km, land evaporation fixed 0.6 of
  saturation, no soil, no water store, no runoff, no stomata, no surface energy
  balance. Tropical deserts evaporated 3.5x observed and rainforest 0.73x FROM
  SAME CONSTANT (corollary d, unnoticed); sensible heat ~25 W/m2 on Sahara and
  Amazon alike. Waves 9-11 spent GPU-hours swapping FIVE convection schemes and
  dozen trigger knobs against tropical-land rain deficit no convection scheme
  could ever fix (corollary c, unnoticed). Interactive soil moisture, bucket
  hydrology, stomatal control are DEFAULTS for AMIP case, not options.
- **A GLOBAL STATISTIC ON A NON-UNIFORM GRID NEEDS AREA WEIGHTS.** Never
  `np.mean(field)` for global mean on lat-lon (or any stretched grid) — use
  `cos(lat)` or model's `areacella`. FAILURE: reported "+17 hPa of dry mass
  created" from unweighted `p_s` mean; AREA-WEIGHTED mass was invariant at
  983.493 hPa to 6 digits, i.e. defect did not exist. Habits carried from
  quasi-uniform MPAS/SCVT mesh are INVALID on lat-lon.
- **A PROPOSED MECHANISM MUST SURVIVE A SCALING / PERTURBATION TEST BEFORE IT IS
  CITED AS THE CAUSE.** X claimed to drive Y → change X by known factor, check Y
  respond as mechanism predict. FAILURE: proposed "damp-to-rest pumps mass
  convergence" (predict ~linear in sponge coefficient); quartering coefficient
  slowed growth only 1.5x, refuting it — fix would have shipped on false
  mechanism. Label every uncaught claim PLAUSIBLE; honest "cause unknown" is
  cheap, confident wrong cause buy a code change and a relaunch.

## Implementation Discipline — the CODE and its PROSE are both claims
User, 2026-08-06: *"Be much more conscientious and careful when implementing.
Be systematic, check, do not be sloppy or too fast."* Every rule below from
defect shipped in ONE session that prompted it, and every one caught by
adversarial reviewer rather than by me — i.e. each avoidable by reading two more
lines before typing. Slow down at these exact points.

- **A DOCSTRING/COMMENT THAT DESCRIBES BEHAVIOUR IS A TESTABLE CLAIM. Trace data
  flow before writing it.** FAILURE: wrote "dropping this call now makes the
  test fail" about `divg_d` exchange — false, because unit 4 hands only
  `uc`/`vc` to `d_sw1` and `divg_d` not consumed until `d_sw5`. Sentence written
  from intent, not from call chain. Before asserting "X is covered by this
  test", name consumer of X and confirm it execute inside test's span.
- **NEVER WRITE "this removes the question entirely" ABOUT AN API YOU HAVE NOT
  READ.** FAILURE: claimed `np.ascontiguousarray` give unconditional copy; it is
  copy-IF-NEEDED, so it aliased at km=1 and copied at km>1 — one line of code
  with two aliasing behaviours, asserted as safe. Extends *Never infer an API*
  to STRENGTH of a guarantee, not just signature.
- **SCOPE WORDS — "both", "all", "every", "cannot", "unified", "always" — GET A
  GREP BEFORE THEY GET TYPED.** FAILURE: "both lanes now route through one
  helper" while `csw_step_sixface` still called interim helpers directly. Went
  into COMMIT MESSAGE, which cannot be edited after. Weaken to what you verified
  ("the post-p_grad_c site") or run the grep.
- **RE-READ A BOOLEAN IMPLICATION IN THE SOURCE BEFORE RESTATING IT; ONE-WAY IS
  NOT TWO-WAY.** FAILURE: `bounded_domain = regional .or. nested .or.
  duogrid` mean duogrid⇒bounded, and I wrote converse pair "unreachable" —
  inverting it and mislabelling legitimate regional/nested category as
  impossible. Quote the line next to restatement so direction checkable.
- **A CONTROL THAT PERTURBS A ZERO IS NOT A CONTROL. Print baseline value the
  perturbation multiplies BEFORE trusting result.** FAILURE: divg-corner probe
  scaled `divg_d(1,1)` by 3 where that cell is exactly 0.0, so control was no-op
  and proved nothing; claim actually rested on predicted==actual identity. Say
  which check carried the claim.
- **AN A/B MUST DIFFER IN ONE FIELD OF THE CONSTRUCTOR, AND YOU MUST DIFF THE
  CONSTRUCTOR ARGS TO KNOW.** FAILURE: compared two contexts differing in grid
  conventions AND in whether ext bundle existed, while asserting it isolated the
  exchange. Applies to fixtures, not just runs — controlled-comparison rule
  cover `pytest` fixtures too.
- **A NEW TEST/FIXTURE IS PART OF THE DIFF: RUN THE SUITE THAT CONSUMES IT, NOT
  ONLY THE TEST YOU WROTE.** Fixture edit = change to every test in the module.
- **SHELL COMMANDS THAT CAN PROMPT WILL SILENTLY NOT RUN.** Use `git checkout

exec
/usr/bin/bash -lc "sed -n '290,385p' packages/coupler/legoesm/driver/model_driver.py; rg -n 'T_2m|_land_T2m|_LAND_STEP_N_CELL|gather_land_step|SurfaceFluxOutput\\(|_stability_step\\(|monin_obukhov_stability\\(' packages tests --glob '*.py'" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 407ms:
                        for x in (T_sea, f_land, T_rad_land, eps_land))
    eps_g = (1.0 - f) * E_sea + f * e_l
    T_g = T_s * (((1.0 - f) * E_sea + f * e_l * (T_l / T_s) ** 4)
                 / eps_g) ** 0.25
    out_dt = jnp.asarray(T_sea).dtype
    return T_g.astype(out_dt), eps_g.astype(out_dt)


# Per-cell fields of the MPAS land step tuple, after the state: T_sfc, albedo,
# q_surface, shflx, lhflx, surface_mass_flux, stress, held, T_rad, emissivity.
_LAND_STEP_N_CELL = 11


def gather_land_step(run, ncol, padded, proc, allgather):
    """Run this process's share of the land step and rebuild the full output
    on every process (shares from ``split_land_columns``).  ``run()`` returns
    the land step tuple ``(state, _LAND_STEP_N_CELL per-cell fields, 5 counts,
    iter max)``;
    every per-column leaf (leading axis ``ncol``) takes each column from the
    process that solved it, counts are summed, the iteration max is maxed.

    ``allgather(tree)`` stacks a pytree across processes on a new axis 0 and
    is a collective: every process calls it twice per land call.  All local
    work that can fail (the step, the payload) runs BEFORE the first one, a
    success flag, so a failure on one process raises on every process
    instead of leaving the others blocked in the payload gather."""
    err = None
    try:
        out = run()
        mine = jnp.asarray(padded[proc])
        flat = jnp.asarray(padded.reshape(-1))
        leaves, tdef = jax.tree_util.tree_flatten(out[:_LAND_STEP_N_CELL + 1])
        # A leaf that is not per column would keep only this process's
        # share's update and silently differ across processes: refuse it.
        bad = [np.shape(x) for x in leaves
               if not (hasattr(x, "shape") and x.ndim >= 1
                       and x.shape[0] == ncol)]
        if bad:
            raise ValueError(
                f"land step output leaves {bad} are not per-column "
                f"(ncol={ncol}); the split land solve cannot gather them")
        # ONE (L, K) float64 buffer for every per-column leaf (exact for the
        # float/bool/small-int leaves the land carries).
        rows = [jnp.asarray(x)[mine].reshape(mine.shape[0], -1)
                for x in leaves]
        widths = [r.shape[1] for r in rows]
        payload = jax.block_until_ready(
            (jnp.concatenate([r.astype(jnp.float64) for r in rows], axis=1),
             jnp.stack([jnp.asarray(x, jnp.float64)
                        for x in out[_LAND_STEP_N_CELL + 1:]])))
    except Exception as exc:  # noqa: BLE001 - agreed across processes below
        err = exc
    flags = allgather(jnp.asarray(0.0 if err is None else 1.0))
    if err is not None:
        raise err
    if float(np.max(np.asarray(flags))) != 0.0:
        raise RuntimeError("split land solve failed on another process "
                           "(see its traceback); stopping in lockstep")
    g_rows, g_cnt = allgather(payload)
    g_rows = jnp.asarray(g_rows).reshape(flat.shape[0], -1)
    full, k = [], 0
    for x, w in zip(leaves, widths):
        blk = g_rows[:, k:k + w].reshape((-1,) + x.shape[1:])
        full.append(jnp.asarray(x).at[flat].set(blk.astype(x.dtype)))
        k += w
    g_cnt = jnp.asarray(g_cnt)
    counts = tuple(jnp.sum(g_cnt[:, i]).astype(
                       jnp.asarray(out[_LAND_STEP_N_CELL + 1 + i]).dtype)
                   for i in range(len(out) - _LAND_STEP_N_CELL - 2))
    return (tuple(jax.tree_util.tree_unflatten(tdef, full)) + counts
            + (jnp.max(g_cnt[:, -1]).astype(jnp.asarray(out[-1]).dtype),))


def radiation_column_mesh(step_mesh, n_devices):
    """The mesh radiation splits its columns over.  A dycore that runs on its
    own step mesh (the FV3 duo face / window layouts) gets that SAME mesh:
    JAX lowers a compiled step against one mesh only, and the physics runs
    inside the duo step's jit.  Every other lane gets a new 1-D ``'col'``
    mesh over ``n_devices`` devices, as before."""
    if step_mesh is not None:
        return step_mesh
    from legoesm.parallel.column_shard import create_column_mesh
    return create_column_mesh(n_devices)


def land_device_mesh(step_mesh, local_devices):
    """The mesh the land device split runs on: the step mesh itself when this
    process owns all of its devices (one mesh per compiled step, as for
    radiation), else a 1-D ``'land'`` mesh over this process's devices.
    KNOWN GAP: several processes each driving more than one device still
    get that per-process ``'land'`` mesh, which differs from the step mesh
    inside the duo physics jit."""
    if step_mesh is not None and len(local_devices) == step_mesh.size:
        return step_mesh
    from jax.sharding import Mesh
    return Mesh(np.array(local_devices), ("land",))
tests/land/unit/test_land_unsolved_column_guard.py:134:    out = SurfaceFluxOutput(
tests/land/unit/test_land_unsolved_column_guard.py:211:    out = SurfaceFluxOutput(
tests/land/unit/test_land_unsolved_column_guard.py:246:    out = SurfaceFluxOutput(
tests/land/unit/test_land_unsolved_column_guard.py:269:    out = SurfaceFluxOutput(
tests/land/unit/test_land_unsolved_fallback.py:87:    return SurfaceFluxOutput(
tests/land/unit/test_canopy_stability_faithful.py:270:    ustar, rah, raw, uav, zeta = monin_obukhov_stability(
tests/land/unit/test_canopy_stability_faithful.py:319:        return jnp.array([float(x) for x in monin_obukhov_stability(*args, n_iters=k, zeta_cap_width=_ZW)])
tests/land/unit/test_canopy_stability_faithful.py:388:            args = lambda ur: monin_obukhov_stability(
tests/land/unit/test_land_screen_temperature.py:94:    ustar, rah, _raw, _uav, zeta, obu = monin_obukhov_stability(
tests/land/unit/test_land_screen_temperature.py:131:    t2, tc = float(out.T_2m[0]), float(out.T_canopy_air[0])
tests/land/unit/test_clm_ml_water_inverse.py:32:    return SurfaceFluxOutput(
tests/land/unit/test_canopy_stability.py:60:    ustar, rah, raw, uav, _zeta = monin_obukhov_stability(
tests/land/unit/test_canopy_stability.py:78:    out = jax.vmap(lambda t: monin_obukhov_stability(
tests/land/unit/test_canopy_stability.py:189:        monin_obukhov_stability(
tests/land/unit/test_patch_mosaic.py:107:    return SurfaceFluxOutput(
tests/land/unit/test_canopy_convergence_reporting.py:70:    return SurfaceFluxOutput(
tests/land/unit/test_snow_latent_charged_heat.py:224:        return SurfaceFluxOutput(
tests/land/unit/test_land_audit_2026_07_09.py:55:            out = monin_obukhov_stability(
tests/unit/test_stable_stability_functions.py:566:    # This is an END-TO-END check: it confirms the returned T_2m responds to
tests/unit/test_stable_stability_functions.py:597:    with the converged scales held ~fixed (strong wind, near-neutral) T_2m
tests/unit/test_stable_stability_functions.py:598:    strictly DECREASES with beta.  (Measured: T_2m(2,5,9) = 289.9261, 289.9237,
tests/unit/test_stable_stability_functions.py:873:    stable column T_2m moves by ~15 K between the default and the SHEBA
tests/unit/test_stable_stability_functions.py:880:    was measured (pre-AeroBulk-port code, 2026-08-02) to shift T_2m by an
tests/unit/test_land_column_split.py:14:from legoesm.driver.model_driver import gather_land_step, split_land_columns
tests/unit/test_land_column_split.py:48:        gather_land_step(lambda p=p: _step(state, split[p][0]), NCOL,
tests/unit/test_land_column_split.py:57:        outs.append(gather_land_step(
tests/unit/test_land_column_split.py:93:        gather_land_step(lambda: tuple(out), NCOL, padded, 0,
tests/unit/test_land_column_split.py:107:        gather_land_step(lambda: _step(state, jnp.arange(2)), NCOL, padded,
tests/unit/test_land_column_split.py:114:        gather_land_step(boom, NCOL, padded, 1, lambda tree: np.array([0.0, 1.0]))
packages/coupler/legoesm/driver/diagnostics.py:877:                *_, T_2m = compute_sam_oceflx_fluxes(
packages/coupler/legoesm/driver/diagnostics.py:882:                return T_2m
packages/coupler/legoesm/driver/diagnostics.py:883:            *_, T_2m = compute_most_fluxes(
packages/coupler/legoesm/driver/diagnostics.py:891:            return T_2m
packages/coupler/legoesm/driver/diagnostics.py:912:        T_2m_land = _profile_to_2m(T_land, q_sfc_land, ocean=False)
packages/coupler/legoesm/driver/diagnostics.py:914:            T2m_land = jnp.asarray(T2m_land, dtype=T_2m_land.dtype)
packages/coupler/legoesm/driver/diagnostics.py:915:            T_2m_land = jnp.where(jnp.isfinite(T2m_land), T2m_land, T_2m_land)
packages/coupler/legoesm/driver/diagnostics.py:916:        T_2m_sea = _profile_to_2m(T_sfc_sea, q_sfc_sea, ocean=True)
packages/coupler/legoesm/driver/diagnostics.py:917:        f_land = jnp.clip(jnp.asarray(land_fraction, dtype=T_2m_land.dtype),
packages/coupler/legoesm/driver/diagnostics.py:919:        return f_land * T_2m_land + (1.0 - f_land) * T_2m_sea
packages/coupler/legoesm/driver/model_driver.py:300:_LAND_STEP_N_CELL = 11
packages/coupler/legoesm/driver/model_driver.py:303:def gather_land_step(run, ncol, padded, proc, allgather):
packages/coupler/legoesm/driver/model_driver.py:306:    the land step tuple ``(state, _LAND_STEP_N_CELL per-cell fields, 5 counts,
packages/coupler/legoesm/driver/model_driver.py:321:        leaves, tdef = jax.tree_util.tree_flatten(out[:_LAND_STEP_N_CELL + 1])
packages/coupler/legoesm/driver/model_driver.py:339:                        for x in out[_LAND_STEP_N_CELL + 1:]])))
packages/coupler/legoesm/driver/model_driver.py:357:                       jnp.asarray(out[_LAND_STEP_N_CELL + 1 + i]).dtype)
packages/coupler/legoesm/driver/model_driver.py:358:                   for i in range(len(out) - _LAND_STEP_N_CELL - 2))
packages/coupler/legoesm/driver/model_driver.py:10933:                _tas_land_T2m = getattr(self, "_land_T2m_last", None)
packages/coupler/legoesm/driver/model_driver.py:10942:                        T2m_land=(None if _tas_land_T2m is None
packages/coupler/legoesm/driver/model_driver.py:10943:                                  else jnp.asarray(_tas_land_T2m).reshape(-1)))
packages/coupler/legoesm/driver/model_driver.py:14729:                        if getattr(_sfc, "T_2m", None) is None
packages/coupler/legoesm/driver/model_driver.py:14730:                        else jnp.asarray(_sfc.T_2m, resp.T_rad.dtype))
packages/coupler/legoesm/driver/model_driver.py:14772:                      if i == _LAND_STEP_N_CELL - 1
packages/coupler/legoesm/driver/model_driver.py:14789:                        return gather_land_step(
packages/coupler/legoesm/driver/model_driver.py:16328:                     _land_Trad_cells, _land_emis_cells, _land_T2m_step,
packages/coupler/legoesm/driver/model_driver.py:16353:                    self._land_T2m_last = _land_T2m_step
tests/unit/test_most_convergence.py:73:    """Combined flags append T_2m then the residual (documented order)."""
tests/unit/test_most_convergence.py:77:    tau_x, tau_y, shflx, lhflx, ustar, T_2m, resid = out
tests/unit/test_most_convergence.py:78:    assert jnp.isfinite(T_2m) and jnp.isfinite(resid)
packages/land/legoesm/land/canopy/solver.py:256:    ustar, rah_above, raw_above, uav, _ = monin_obukhov_stability(
packages/land/legoesm/land/canopy/solver.py:438:    ustar, rah_above, raw_above, uav, zeta, obu = monin_obukhov_stability(
packages/land/legoesm/land/surface_scheme/base.py:16:class SurfaceFluxOutput(NamedTuple):
packages/land/legoesm/land/surface_scheme/base.py:94:    T_2m: jax.Array | None = None
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:710:    return SurfaceFluxOutput(
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:733:        T_2m=screen_level_temperature(Ta, Tc_cvg, z_ref - displa,
packages/land/legoesm/land/surface_scheme/simple_seb.py:357:    return SurfaceFluxOutput(
packages/land/legoesm/land/canopy/clm_ml_interface.py:1593:    return SurfaceFluxOutput(
packages/land/legoesm/land/canopy/stability.py:347:def _stability_step(carry: jax.Array, _xs: None,
packages/land/legoesm/land/canopy/stability.py:392:def monin_obukhov_stability(
packages/core/legoesm/core/bulk_flux.py:1590:        ONLY when ``return_convergence=True`` (appended last, after ``T_2m``
packages/core/legoesm/core/bulk_flux.py:1931:            T_2m = T_sfc - (theta_star / KAPPA) * denom_d
packages/core/legoesm/core/bulk_flux.py:1952:            T_2m = T_atm - _coare["ts"] / KAPPA * prf
packages/core/legoesm/core/bulk_flux.py:1956:        T_2m = jnp.clip(T_2m, lo, hi)
packages/core/legoesm/core/bulk_flux.py:1958:            return (tau_x, tau_y, shflx, lhflx, u_star, T_2m, most_residual
packages/core/legoesm/core/bulk_flux.py:1960:        return (tau_x, tau_y, shflx, lhflx, u_star, T_2m) + _wind_out

exec
/usr/bin/bash -lc "rg -n 't_ref2m|temp12m|temp1\\(' /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/{CanopyFluxesMod,FrictionVelocityMod}.F90; sed -n '1,150p' tests/unit/test_land_column_split.py; ls -l .venv/bin/python" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
exec
/usr/bin/bash -lc "sed -n '250,490p' packages/land/legoesm/land/canopy/stability.py; sed -n '420,755p' packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
    ustar3 = constants.kappa_von_karman * um / (
        jnp.log(zldis / z0m)
        - psi_m(zeta, _STABLE_SCHEME) + psi_m(z0m / obu, _STABLE_SCHEME))

    ustar = jnp.where(zeta < -zetam, ustar1,
            jnp.where(zeta < 0.0,    ustar2, ustar3))
    return ustar


def _temperature_humidity_relation(zldis: jax.Array, obu: jax.Array,
                                   z0h: jax.Array) -> jax.Array:
    """Compute θ* / (θ_atm - θ_sfc) (4 regimes, CLM5 / DifferBESS)."""
    zetat = _ZETAT
    zeta  = zldis / obu

    # Very unstable (valid: obu < 0, zeta < -zetat).  Floor log/cbrt args; no-ops here.
    ch1 = constants.kappa_von_karman / (
        jnp.log(jnp.maximum(-zetat * obu / z0h, _MOST_ARG_FLOOR))
        - _stability_func_heat(-zetat)
        + _stability_func_heat(z0h / obu)
        + _MOST_HEAT_CONV_COEF * (1.0 / jnp.cbrt(zetat)
                                  - 1.0 / jnp.cbrt(jnp.maximum(-zeta, zetat)))
    )
    ch2 = constants.kappa_von_karman / (
        jnp.log(zldis / z0h)
        - _stability_func_heat(zeta)
        + _stability_func_heat(z0h / obu)
    )
    # Stable (zeta >= 0): Beljaars & Holtslag (1991), all stable zeta.  The core
    # psi_h floors its stable argument at 1e-10, so the (1 + 2z/3)^1.5 term stays
    # finite for the negative arguments of the discarded (obu < 0) branch.
    ch3 = constants.kappa_von_karman / (
        jnp.log(zldis / z0h)
        - psi_h(zeta, _STABLE_SCHEME) + psi_h(z0h / obu, _STABLE_SCHEME))

    ch = jnp.where(zeta < -zetat, ch1,
         jnp.where(zeta < 0.0,    ch2, ch3))
    return ch


def screen_level_temperature(theta_ref: jax.Array, T_canopy_air: jax.Array,
                             zldis: jax.Array, obu: jax.Array,
                             z0h: jax.Array,
                             z_screen: float = _SCREEN_HEIGHT_M) -> jax.Array:
    """CLM5 ``t_ref2m``: air temperature ``z_screen`` above the displacement
    height plus ``z0h``, interpolated between the reference-height potential
    temperature ``theta_ref`` and the canopy-air (aerodynamic surface)
    temperature with this module's heat similarity relation at the Obukhov
    length ``obu`` the solve's heat flux was evaluated with
    (CanopyFluxesMod.F90:1182, FrictionVelocityMod.F90:636-652)::

        t_ref2m = thm + temp1 * dth * (1/temp12m - 1/temp1),  dth = thm - taf

    ``temp1`` and ``temp12m`` are evaluated at the SAME ``obu`` (as CLM does
    within one FrictionVelocity call), so ``temp1`` is the flux's own ``ch``.  Stable side: Beljaars & Holtslag (1991),
    as in the solve (module docstring).  Output is in the same frame as
    ``theta_ref`` (surface-referenced potential temperature, CLM ``thm``)."""
    temp1 = _temperature_humidity_relation(zldis, obu, z0h)
    temp12m = _temperature_humidity_relation(z_screen + z0h, obu, z0h)
    dth = theta_ref - T_canopy_air
    return theta_ref + temp1 * dth * (1.0 / temp12m - 1.0 / temp1)


def _cap_stable_zeta(zeta: jax.Array, width: float) -> jax.Array:
    """Smooth cap at ``_ZETA_MAX_STABLE`` (no kink), then the neutral floor.

    Floor LAST: the softplus leaks ~w*exp(-0.5/w) below the cap, which would
    push a floored 1e-6 negative (into the unstable forms).
    """
    w = width
    z = zeta - w * jax.nn.softplus((zeta - _ZETA_MAX_STABLE) / w)
    return jnp.maximum(z, _ZETA_NEUTRAL_FLOOR)


def _monin_obukhov_init(ur: jax.Array, Tv_atm: jax.Array,
                        dthv: jax.Array, zldis: jax.Array,
                        z0m: jax.Array,
                        zeta_cap_width: float) -> tuple[jax.Array, jax.Array]:
    """Initialise MOST via bulk Richardson number (Zeng et al. 1998)."""
    wc  = 0.5
    um  = jnp.where(dthv >= 0.0, jnp.maximum(ur, 0.1), jnp.sqrt(ur**2 + wc**2))  # coeff-ok: 0.1 m/s wind floor
    rib = constants.g * zldis * dthv / (Tv_atm * um**2)

    zeta = jnp.where(
        rib >= 0.0,
        rib * jnp.log(zldis / z0m) / (1.0 - _MOST_BETA_STABLE * jnp.minimum(rib, _RIB_MAX)),
        rib * jnp.log(zldis / z0m),
    )
    zeta = jnp.where(
        rib >= 0.0,
        _cap_stable_zeta(zeta, zeta_cap_width),
        jnp.clip(zeta, -100.0, -_ZETA_NEUTRAL_FLOOR),  # coeff-ok: very-unstable bound on ζ
    )
    obu = zldis / zeta
    return um, obu


def _stability_step(carry: jax.Array, _xs: None,
                    forcing: jax.Array,
                    zeta_cap_width: float) -> tuple[jax.Array, jax.Array]:
    """Single Monin-Obukhov fixed-point iteration.

    carry : [z0h, obu, um]
    forcing: [Ta, Tv_atm, q_atm, q_atm, zldis, z0m, dq, dth, ur]
    output : [ustar, tstar, qstar, thvstar, ch, zeta, obu]  (``obu`` = the
             Obukhov length ``ch`` was evaluated with, i.e. the incoming one)
    """
    z0h, obu, um = carry[0], carry[1], carry[2]
    Ta, Tv_atm, q_atm, _q2, zldis, z0m, dq, dth, ur = (
        forcing[0], forcing[1], forcing[2], forcing[3],
        forcing[4], forcing[5], forcing[6], forcing[7], forcing[8],
    )

    ustar = _friction_velocity(zldis, z0m, obu, um)
    ch    = _temperature_humidity_relation(zldis, obu, z0h)
    tstar = ch * dth
    qstar = ch * dq
    thvstar = tstar * (1.0 + _VIRT_T_COEF * q_atm) + _VIRT_T_COEF * Ta * qstar

    zeta  = zldis * constants.kappa_von_karman * constants.g * thvstar / (ustar**2 * Tv_atm)

    zeta_stable = _cap_stable_zeta(zeta, zeta_cap_width)
    um_stable   = jnp.maximum(ur, 0.1)                     # coeff-ok: 0.1 m/s wind floor
    zeta_unstable = jnp.clip(zeta, -100.0, -_ZETA_NEUTRAL_FLOOR)  # coeff-ok: very-unstable bound on ζ
    # Floor the cbrt argument to a POSITIVE value, not 0: cbrt'(0)=inf and the
    # maximum's subgradient is 0 below the clamp, so cbrt(maximum(x, 0)) gives
    # 0*inf = NaN in the reverse-mode gradient whenever x<=0 (the stable regime,
    # where this unstable-branch wc is discarded).  cbrt(1e-12)~1e-4 m/s ~ 0.
    wc_unstable = jnp.cbrt(jnp.maximum(
        -constants.g * ustar * thvstar * _CONV_BDY_HEIGHT / Tv_atm, _MOST_ARG_FLOOR))
    um_unstable = jnp.sqrt(ur**2 + wc_unstable**2)

    is_stable = zeta >= 0.0
    zeta_new = jnp.where(is_stable, zeta_stable, zeta_unstable)
    um_new   = jnp.where(is_stable, um_stable,   um_unstable)
    obu_new  = zldis / zeta_new

    new_carry = jnp.array([z0h, obu_new, um_new])
    output    = jnp.array([ustar, tstar, qstar, thvstar, ch, zeta_new, obu])
    return new_carry, output


def monin_obukhov_stability(
    ur: jax.Array,
    Ta: jax.Array,
    Tv_atm: jax.Array,
    Tc: jax.Array,
    q_atm: jax.Array,
    q_c: jax.Array,
    zldis: jax.Array,
    z0m: jax.Array,
    n_iters: int = 5,
    *,
    zeta_cap_width: float,
    return_flux_obu: bool = False,
) -> tuple[jax.Array, ...]:
    """Above-canopy MOST iteration via jax.lax.scan.

    Parameters
    ----------
    ur     : wind speed at reference height [m/s]
    Ta     : air temperature [K]
    Tv_atm : virtual potential temperature of air [K]
    Tc     : canopy air temperature [K]  (first-guess or converged)
    q_atm  : air specific humidity [kg/kg]
    q_c    : canopy air specific humidity [kg/kg]
    zldis  : reference height minus displacement height [m]
    z0m    : roughness length for momentum [m]
    n_iters: number of fixed-point iterations
    zeta_cap_width: smoothing width of the stable zeta <= 0.5 cap
                    (``CanopyConfig.zeta_cap_smoothing_width``)

    Returns
    -------
    ustar    : friction velocity [m/s]
    rah      : aerodynamic resistance to heat [s/m]
    raw      : aerodynamic resistance to water vapour [s/m]
    uav      : mean wind speed within canopy (used for Rb) [m/s]
    zeta     : stability parameter [-]
    obu      : (only with ``return_flux_obu``) the Obukhov length [m] the
               returned ``rah``/``raw`` were evaluated with (the last
               iteration's INCOMING one; ``zeta`` is the updated stability)
    """
    # Heat roughness length.  CLM5 vegetation uses kB^-1 = 0 (z0h = z0m); the
    # earlier BESS default kB^-1 = 2 (z0h = z0m / e^2) over-suppressed heat
    # exchange and warm-biased canopy/skin temperature (DifferBESS aa6e8b9).
    z0h  = z0m   # kB^-1 = 0
    dth  = Ta - Tc
    dq   = q_atm - q_c
    dthv = (Ta - Tc) * (1.0 + _VIRT_T_COEF * q_atm) + _VIRT_T_COEF * Ta * (q_atm - q_c)

    um, obu = _monin_obukhov_init(ur, Tv_atm, dthv, zldis, z0m, zeta_cap_width)

    init    = jnp.array([z0h, obu, um])
    forcing = jnp.array([Ta, Tv_atm, q_atm, q_atm, zldis, z0m, dq, dth, ur])
    step_fn = partial(_stability_step, forcing=forcing,
                      zeta_cap_width=zeta_cap_width)

    # Only the LAST iterate's outputs are used, so carry them instead of
    # stacking all n_iters: the stacked (n_iters, ...) buffer was rewritten
    # every iteration, ~half the cost of this function under the canopy
    # solve's jacfwd + vmap on CPU (production res6, bit-identical).
    if n_iters < 1:
        raise ValueError(f"monin_obukhov_stability needs n_iters >= 1, got {n_iters!r}")

    def _step_keep_last(c, _xs):
        carry, _ = c
        return step_fn(carry, None), None

    # Placeholder output with the step's own shape/dtype (scalar or batched
    # inputs alike); it is overwritten by the first iterate.
    _out = jax.eval_shape(step_fn, init, None)[1]
    (final_carry, last), _ = jax.lax.scan(
        _step_keep_last, (init, jnp.zeros(_out.shape, _out.dtype)), xs=None,
        length=n_iters)

    ustar, ch, zeta = last[0], last[4], last[5]

    rah = 1.0 / jnp.maximum(ch * ustar, 1e-9)
    raw = rah  # same for sensible heat and water vapour (neutral Prandtl)

    um_final = final_carry[2]
    ram      = jnp.maximum(um_final / ustar**2, 1e-9)
    uav      = jnp.sqrt(um_final / ram)

    if return_flux_obu:
        return ustar, rah, raw, uav, zeta, last[6]
    return ustar, rah, raw, uav, zeta


# ---------------------------------------------------------------------------
# Leaf boundary-layer resistance
# ---------------------------------------------------------------------------

@jax.jit
def compute_boundary_layer_resistance(
    uav: jax.Array,
    LAI: jax.Array,
    fSun: jax.Array,
    cv: jax.Array,
    d_leaf: jax.Array,
    # Clip strictly below 1 before arccos: d/dx arccos(x) = -1/sqrt(1-x^2)
    # diverges at x=1 (subsolar point) and clip zeroes the subgradient above 1,
    # so the reverse-mode VJP would give 0*inf = NaN in the cos_zenith cotangent.
    SZA = jnp.degrees(jnp.arccos(jnp.clip(cos_zenith, 0.0, 1.0 - 1e-7)))
    PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV = split_sw_components(
        forcing.sw_down, cos_zenith)

    sw_rt = canopy_shortwave_rt(
        PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV,
        SZA, LAI, CI, ALB_VIS, ALB_NIR,
        Vc3_leaf_stressed, Vc4_leaf_stressed, kn)

    # ---- Thermodynamic / atmosphere variables ----
    Ps    = forcing.p_surface
    q_atm = forcing.q_lowest
    rhoa  = forcing.rho_lowest
    Tv_atm = Ta * (1.0 + _VIRT_T_COEF * q_atm)
    from legoesm.thermo import latent_heat_vaporization
    lam   = latent_heat_vaporization(T_soil_top)   # the L the multilayer inverts (same T)
    Cp    = constants.c_pd
    Ca    = forcing.co2_ppmv

    # ---- Initial Newton state ----
    Ts_old   = T_soil_top
    q_s_init = sat_specific_humidity(Ts_old, Ps)
    q_c_init = 0.5 * (q_s_init + q_atm)
    chi = _CI_CA_C3 - _CI_CA_C3_MINUS_C4 * fC4
    Ci_init = Ca * chi

    cold_state = jnp.stack(
        [Ta, Ta, Ci_init, Ci_init, Ta, q_c_init], axis=-1)  # (ncol, 6)
    _Ta_retry = Ta - _RETRY_COOLING_K
    q_c_retry = jnp.minimum(q_c_init, _RETRY_RH_CAP * sat_specific_humidity(_Ta_retry, Ps))
    retry_state = jnp.stack(
        [_Ta_retry, _Ta_retry, Ci_init, Ci_init, _Ta_retry, q_c_retry], axis=-1)

    # ---- Warm start (numerical cache) ----
    # ``canopy_seed`` is the previous timestep's LAST CONVERGED solution, carried
    # by the caller.  A column whose seed is not finite (never solved yet, or the
    # previous solve failed) OR lies outside the physical box a root may occupy
    # (a cache written before that box was enforced -- restarts heal here) falls
    # back to the cold start above.  The seed is sanitised BEFORE the select:
    # ``where`` evaluates both branches, and a NaN in the discarded one would
    # poison a reverse-mode tangent.
    #
    # The solve's adjoint returns a ZERO cotangent for its seed (``solver.
    # solve_bwd``), but that does NOT make the answer seed-independent: the
    # residual has spurious roots, and which one a far-off seed reaches is the
    # defect the admissibility box exists to reject.
    if canopy_seed is None:
        initial_state = cold_state
    else:
        _seed = jnp.asarray(canopy_seed, cold_state.dtype)
        _ok = canopy_state_admissible(_seed)[:, None]
        initial_state = jnp.where(_ok, jnp.nan_to_num(_seed), cold_state)

    def _bcast(v):
        if hasattr(v, "shape") and v.shape == (ncol,):
            return v
        return jnp.broadcast_to(jnp.asarray(v), (ncol,))

    def _build_bundle(Ts_bc):
        return CanopyForcingBundle(
            LAI=LAI, SZA=SZA, La=forcing.lw_down,
            epsf=_bcast(cc.epsf if _epsf_col is None else _epsf_col),
            epss=_bcast(cc.epss if ground_emissivity is None else ground_emissivity),
            fSun=sw_rt.fSun,
            APAR_Sun=sw_rt.APAR_Sun, APAR_Sh=sw_rt.APAR_Sh,
            Vcmax25_Sun=sw_rt.Vcmax25_C3Sun, Vcmax25_Sh=sw_rt.Vcmax25_C3Sh,
            Vcmax25_C4Sun=sw_rt.Vcmax25_C4Sun, Vcmax25_C4Sh=sw_rt.Vcmax25_C4Sh,
            ASW_Sun=sw_rt.ASW_Sun, ASW_Sh=sw_rt.ASW_Sh, ASW_Soil=sw_rt.ASW_Soil,
            Ts_bc=Ts_bc,
            Ca=_bcast(Ca), Ps=Ps, Ta=Ta,
            lam=_bcast(lam), Cp=_bcast(Cp), rhoa=rhoa, Tv_atm=Tv_atm, q_atm=q_atm,
            m=m_mix, b0=b0_mix, alf=alf, TgC=TgC,
            fC4=fC4, fStress_soil=fStress_soil,
            ur=wind_speed, CI=CI, z0m=z0m, displa=displa, z0=z_ref,
            cv=_bcast(cc.cv), d_leaf=d_leaf,
            r_soil_surface=r_soil_surface,
            fwet=(jnp.zeros_like(w_frac_rz) if fwet is None
                  else jnp.broadcast_to(fwet, w_frac_rz.shape)),
            lam_soil=None if ground_latent_heat is None else _bcast(ground_latent_heat),
        )

    def _solve_one_col(x0, bun):
        # Diagnostic entry point: identical solve, plus the terminal residual /
        # damping / cap-exit flag the caller needs to tell a stalled column from
        # a nearly-solved one.  Gradients are unchanged (zero cotangent on every
        # diagnostic output).
        return solve_canopy_closure_diag(x0, bun, cc)

    def _fwd_one_col(xf, bun):
        return canopy_forward(xf, bun, cc.LE_module, cc.stomatal_model,
                               cc.le_cap_mode, cc.use_ta_for_photosynthesis,
                               cc.rh_cap_smoothing_width,
                               cc.zeta_cap_smoothing_width,
                               cc.most_n_iters)

    # ---- Outer Picard loop: canopy closure ↔ soil thermal ----
    # See canopy_land.py module notes for the stability analysis.  The
    # soil_thermal_fn callback is responsible for advancing Ts tentatively
    # given the Picard-iterate G; slab and multilayer callers supply
    # different implementations.
    n_picard = getattr(cc, "n_picard", _DEFAULT_N_PICARD)
    omega    = getattr(cc, "picard_omega", _DEFAULT_PICARD_OMEGA)
    Ts_bc_k = Ts_old

    # Each Picard pass RESEEDS from the previous pass's converged solution
    # instead of restarting cold.  Successive passes differ only in ``Ts_bc``,
    # which the relaxation moves by a fraction of a kelvin, so the previous
    # solution is a far better seed than the air temperature — same fixed point,
    # fewer iterations to reach it.  Only a CONVERGED pass earns the right to
    # seed the next one: a failed solve's last iterate is not a root, and
    # reusing it would let one bad pass cascade through the rest.
    # ``x_conv`` holds the last CONVERGED solution per column, NaN until one
    # exists; it is both the next pass's seed and (after the loop) the cache
    # handed back to the caller.
    # A column with no converged solution yet alternates its seed across passes
    # between the caller's seed and the (slightly cooled) retry state, so a seed
    # that led the solve astray gets a different retry inside the same call.  The
    # passes run over every column anyway (one batch), so this costs nothing.  A
    # failed pass still relaxes the soil boundary with its cold-state fallback
    # flux (the one the land step accepts if the column ends unsolved); freezing
    # it held desert columns at their start-of-step soil.  Acceptance stays tied
    # to the LAST pass.
    x_conv = jnp.full_like(initial_state, jnp.nan)
    for _picard_iter in range(n_picard):
        bundles_k = _build_bundle(Ts_bc_k)
        _fallback = initial_state if _picard_iter % 2 == 0 else retry_state
        _seed_k = jnp.where(jnp.all(jnp.isfinite(x_conv), axis=-1, keepdims=True),
                            jnp.nan_to_num(x_conv), _fallback)
        # One batch over every column (user choice 2026-11-09, replacing the
        # 64-column groups chosen 2026-11-03).  Once the LM step stopped forming
        # explicit Q (PR 2398) the solve is launch-bound, so one wide batch
        # beats many narrow ones: AMIP production deck, 4 A100, fp64, 1 day,
        # 0.242 s/step with groups of 64 vs 0.138 s/step with one batch.
        # Pure performance: same per-column solve, answers differ by rounding.
        # Cost: the batch iterates until the rank's slowest column stops, so a
        # column stuck at the iteration cap holds every column to the cap.
        (x_final, n_iters, converged, resid_sq, resid_rel, lam_f,
         hit_cap) = jax.vmap(_solve_one_col)(_seed_k, bundles_k)
        x_conv = jnp.where(converged[:, None], x_final, x_conv)
        # A rejected iterate may be non-finite: evaluate the fluxes of a failed
        # column at the (finite) cold state instead, so no NaN enters a
        # reverse-mode tangent.  Those fluxes are the column's FALLBACK, not a
        # root: it reports converged=False, and the multilayer land step closes
        # their energy into sensible heat and accepts the column (or reverts it
        # if non-finite or beyond the fallback guards).
        x_final = jnp.where(converged[:, None], x_final, cold_state)
        fluxes_per_col = jax.vmap(_fwd_one_col)(x_final, bundles_k)

        G_k = jnp.clip(fluxes_per_col["G"], -500.0, 700.0)  # coeff-ok: physical range clamp on ground heat flux [W m-2]
        Ts_thermal = soil_thermal_fn(G_k, dt)
        Ts_bc_k = (1.0 - omega) * Ts_bc_k + omega * Ts_thermal

    # ---- Converged state ----
    # State vector order: [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c].
    Tf_Sun = x_final[:, 0]
    Tf_Sh  = x_final[:, 1]
    Tc_cvg = x_final[:, 4]   # canopy air-space temperature (aerodynamic node)
    Ts_cvg = Ts_bc_k

    fSun    = sw_rt.fSun
    LE_Sun  = fluxes_per_col["LE_Sun"]
    LE_Sh   = fluxes_per_col["LE_Sh"]
    # Wet-leaf evaporation (interception loss) — the store-sourced share of the
    # canopy latent flux; the caller routes it to the canopy-water store.
    LE_wet_canopy_d = fluxes_per_col["LE_wet_Sun"] + fluxes_per_col["LE_wet_Sh"]
    LE_Soil = fluxes_per_col["LE_Soil"]
    H_Sun   = fluxes_per_col["H_Sun"]
    H_Sh    = fluxes_per_col["H_Sh"]
    H_Soil  = fluxes_per_col["H_Soil"]
    An_Sun  = fluxes_per_col["An_Sun"]        # NET (drives SIF + leaf coupling)
    An_Sh   = fluxes_per_col["An_Sh"]
    Agross_Sun = fluxes_per_col["Agross_Sun"]  # GROSS (carbon-model GPP)
    Agross_Sh  = fluxes_per_col["Agross_Sh"]
    G       = fluxes_per_col["G"]
    gs_Sun  = fluxes_per_col["gs_Sun"]
    gs_Sh   = fluxes_per_col["gs_Sh"]
    ustar   = fluxes_per_col["ustar"]
    Rn_Sun_d  = fluxes_per_col["Rn_Sun"]
    Rn_Sh_d   = fluxes_per_col["Rn_Sh"]
    Rn_Soil_d = fluxes_per_col["Rn_Soil"]

    # Canopy-air condensation (dew limit): its vapour never reaches the
    # atmosphere and its latent heat warms the canopy air, so the top-of-canopy
    # fluxes are the surface sums shifted by L_c*C in opposite directions.  L_c
    # is the snow-weighted ground charge, the phase the land step stores it in
    # (``SurfaceFluxOutput.condensation``).
    condensation = fluxes_per_col["condensation"]
    L_cond = _bcast(lam if ground_latent_heat is None else ground_latent_heat)
    LE_tot = LE_Sun + LE_Sh + LE_Soil - L_cond * condensation
    # Positive UPWARD: leaf/soil heat warms air; land loses H_tot in Rn-H-LE.
    H_tot  = H_Sun  + H_Sh  + H_Soil + L_cond * condensation
    # GPP is GROSS carbon uptake (BEFORE leaf dark respiration).  The carbon
    # model (carbon_cycle.step_carbon) re-charges foliar MAINTENANCE
    # respiration r_maint_fol*C_fol separately, so exporting NET An here would
    # double-count leaf respiration (once as Rd folded into An, once as
    # r_maint_fol) and bias carbon-use efficiency (NPP/GPP) low.  This matches
    # the SimpleSEB path (carbon/stomata.py: gpp = max(A_gross, 0)*_MC).  NET
    # An_Sun/An_Sh still drive stomatal coupling, the leaf energy/CO2 flux, and
    # SIF (below); only the carbon-facing GPP is gross.
    GPP    = (Agross_Sun + Agross_Sh) * _G_C_PER_UMOL_CO2    # gC m-2 s-1 (GROSS)

    # ---- Optional solar-induced fluorescence (passive TOC diagnostic) ----
    # cc.sif is a static config leaf, so this Python gate does not double-trace.
    # An_Sun/An_Sh and APAR_Sun/APAR_Sh are canopy-integrated per leaf-class
    # (per ground area), so the sunlit+shaded sum is the canopy total.
    # Gamma* MUST use the same temperature the Farquhar An used — the solver
    # takes T_phot = Ta if use_ta_for_photosynthesis else Tf (solver.py) — else
    # the je inversion is inconsistent with the assimilation it inverts.
    # NOTE: the BEPS-SIF je inversion is C3-style (uses Gamma*); for a mixed
    # canopy (fC4 > 0) it is applied to the blended C3/C4 An as a documented
    # BEPS-parity approximation (no separate C4 fluorescence path).
    if cc.sif is not None:
        T_phot_Sun = Ta if cc.use_ta_for_photosynthesis else Tf_Sun
        T_phot_Sh  = Ta if cc.use_ta_for_photosynthesis else Tf_Sh
        sif_out = two_leaf_canopy_sif(
            An_Sun, x_final[:, 2], co2_compensation_point(T_phot_Sun), sw_rt.APAR_Sun,
            An_Sh, x_final[:, 3], co2_compensation_point(T_phot_Sh), sw_rt.APAR_Sh,
            cc.sif)
    else:
        sif_out = None

    # Per-component canopy fluxes (for offline diagnostic drivers).
    LE_canopy_d = LE_Sun + LE_Sh
    H_canopy_d  = H_Sun  + H_Sh
    Rn_canopy_d = Rn_Sun_d + Rn_Sh_d

    # Internal energy-balance closure diagnostic.
    Rn_int_d       = Rn_canopy_d + Rn_Soil_d
    residual_int_d = Rn_int_d - (LE_tot + H_tot + G)

    # ---- Canopy albedo diagnosed from SW absorption ----
    sw_absorbed = sw_rt.ASW_Sun + sw_rt.ASW_Sh + sw_rt.ASW_Soil
    sw_down_safe = jnp.maximum(forcing.sw_down, 1.0)
    alpha_canopy = jnp.clip(1.0 - sw_absorbed / sw_down_safe, 0.0, 1.0)
    alpha_canopy = jnp.where(
        forcing.sw_down < 1.0,
        jnp.broadcast_to(jnp.asarray(land_config.albedo_land), T_soil_top.shape),
        alpha_canopy)

    # ---- Surface emissivity + radiometric temperature (atmosphere-equivalent) ----
    # The conservative ``canopy_longwave_rt`` exports the column LW emissivity
    # ``eps_col = 1 - R_col`` (R_col = column LW reflectance) and the emission-only
    # upward flux ``LW_emit``, chosen so the atmosphere's property-coupling LW
    # boundary ``eps_col*sigma*T_surface^4 + (1-eps_col)*La`` reproduces the
    # canopy's conservative ``LW_out`` EXACTLY (verified to ~1e-13 W/m2 across
    # LAI/emissivities).  These are the physically-correct surface radiative
    # properties: the coupler tile-blends ``eps_eff`` (the column emissivity) and
    # ``T_surface`` and threads them into RRTMGP as the dynamic surface emissivity
    # + skin temperature, so the land->atmosphere LW boundary carries no static-
    # emissivity mismatch.  The raw upward flux is still ``lw_up = LW_out`` with
    # ``lw_net = La - LW_out``.
    LW_out_col = fluxes_per_col["LW_out"]
    eps_eff    = fluxes_per_col["eps_col"]
    LW_emit    = fluxes_per_col["LW_emit"]
    T_surface = (LW_emit / jnp.maximum(eps_eff * constants.sigma_sb, 1e-12)) ** 0.25

    # SW_net and the (conservative) external LW_net = La − LW_out.
    sw_net = (1.0 - alpha_canopy) * forcing.sw_down
    lw_net = forcing.lw_down - LW_out_col
    lw_up_out = LW_out_col   # true top-of-canopy upward LW (emission + reflection)

    # External (boundary-condition) radiation balance — what a downstream
    # observer sees from forcing + canopy-mean emission T.
    Rn_ext_d       = sw_net + lw_net
    residual_ext_d = Rn_ext_d - (LE_tot + H_tot + G)

    # Safety clamp on G: extreme non-converged values corrupt T_soil.
    G_clamped = jnp.clip(G, -500.0, 700.0)  # coeff-ok: physical range clamp on ground heat flux [W m-2]

    # ---- Wind stress from MOST ustar ----
    # Sign: stress ON THE AIR (opposes the wind), the convention of the ocean
    # bulk and SimpleSEB stresses this vector is tile-blended with (#2031).
    tau_mag = rhoa * ustar**2
    tau_x = -tau_mag * wind_dir_x
    tau_y = -tau_mag * wind_dir_y

    # ---- Surface specific humidity (proxy; refined post-step by the caller) ----
    # The canopy solver tracks q_c (canopy-air humidity) as part of its
    # Newton state; here we report q_c as the surface humidity so the
    # coupler sees a consistent value when it queries the pre-Richards
    # surface state.  Post-Richards, the caller recomputes q_surface
    # from T_surface_new and the updated soil moisture.
    q_c_final = x_final[:, 5]

    # Vegetation cover fraction (for root-zone transpiration partition).
    f_veg = jnp.clip(w_frac_rz, 0.0, 1.0)

    return SurfaceFluxOutput(
        shflx=H_tot, lhflx=LE_tot,
        tau_x=tau_x, tau_y=tau_y,
        sw_net=sw_net, lw_net=lw_net, lw_up=lw_up_out,
        G_soil=G_clamped,
        T_surface=T_surface,
        q_surface=q_c_final,
        albedo=alpha_canopy,
        emissivity=jnp.broadcast_to(eps_eff, T_soil_top.shape),
        z0=z0m,
        gpp=GPP,
        sif=sif_out,
        # Warm-start cache for the NEXT call: the last CONVERGED solve per
        # column (NaN where this column has never converged, which the seed
        # consumer reads as "cold start").  A numerical cache only — see the
        # ``canopy_seed`` note above.
        canopy_x=x_conv,
        canopy_resid_sq=resid_sq,
        canopy_resid_rel=resid_rel,
        canopy_hit_cap=hit_cap,
        Tf_Sun=Tf_Sun,
        Tf_Sh=Tf_Sh,
        T_canopy_air=Tc_cvg,
        T_2m=screen_level_temperature(Ta, Tc_cvg, z_ref - displa,
                                      fluxes_per_col["obu"], z0m),
        gs_Sun=gs_Sun,
        gs_Sh=gs_Sh,
        n_iters=n_iters,
        # Whether the Newton closure actually reached a root on this column.
        # False means the fluxes above are the cold-state fallback, not a
        # solution — the caller must close their energy before spending them
        # (multilayer land does).  Carried from the LAST Picard pass.
        converged=converged,
        f_veg=f_veg,
        fSun=fSun,
        Ts_solve=Ts_cvg,
        LE_canopy=LE_canopy_d,
        LE_wet_canopy=LE_wet_canopy_d,
        LE_soil=LE_Soil,
        H_canopy=H_canopy_d,
        H_soil=H_Soil,
        Rn_canopy=Rn_canopy_d,
        Rn_soil=Rn_Soil_d,
        Rn_int=Rn_int_d,
        residual_int=residual_int_d,
        Rn_ext=Rn_ext_d,

 exited 2 in 15ms:
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:240:    real(r8) :: temp1(bounds%begp:bounds%endp)       ! relation for potential temperature profile
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:241:    real(r8) :: temp12m(bounds%begp:bounds%endp)     ! relation for potential temperature profile applied at 2-m
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:277:    real(r8) :: de2mdT                               ! derivative of 2 m height surface saturated vapor pressure on t_ref2m
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:278:    real(r8) :: qsat_ref2m                           ! 2 m height surface saturated specific humidity [kg/kg]
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:279:    real(r8) :: dqsat2mdT                            ! derivative of 2 m height surface saturated specific humidity on t_ref2m
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:388:         swbgt_ref2m            => humanindex_inst%swbgt_ref2m_patch            , & ! Output: [real(r8) (:)   ]  2 m Simplified Wetbulb Globe temperature (C)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:389:         swbgt_ref2m_r          => humanindex_inst%swbgt_ref2m_r_patch          , & ! Output: [real(r8) (:)   ]  Rural 2 m Simplified Wetbulb Globe temperature (C)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:392:         wbt_ref2m              => humanindex_inst%wbt_ref2m_patch              , & ! Output: [real(r8) (:)   ]  2 m Stull Wet Bulb temperature (C)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:393:         wbt_ref2m_r            => humanindex_inst%wbt_ref2m_r_patch            , & ! Output: [real(r8) (:)   ]  Rural 2 m Stull Wet Bulb temperature (C)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:398:         ept_ref2m              => humanindex_inst%ept_ref2m_patch              , & ! Output: [real(r8) (:)   ]  2 m height Equivalent Potential temperature (K)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:399:         ept_ref2m_r            => humanindex_inst%ept_ref2m_r_patch            , & ! Output: [real(r8) (:)   ]  Rural 2 m height Equivalent Potential temperature (K)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:453:         t_ref2m                => temperature_inst%t_ref2m_patch               , & ! Output: [real(r8) (:)   ]  2 m height surface air temperature (Kelvin)                           
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:454:         t_ref2m_r              => temperature_inst%t_ref2m_r_patch             , & ! Output: [real(r8) (:)   ]  Rural 2 m height surface air temperature (Kelvin)                     
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:763:              temp1(begp:endp), temp2(begp:endp), temp12m(begp:endp), temp22m(begp:endp), fm(begp:endp), &
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:781:            rah(p,1) = 1._r8/(temp1(p)*ustar(p))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1084:            tstar = temp1(p)*dth(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1182:         t_ref2m(p) = thm(p) + temp1(p)*dth(p)*(1._r8/temp12m(p) - 1._r8/temp1(p))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1183:         t_ref2m_r(p) = t_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1191:         call QSat(t_ref2m(p), forc_pbot(c), e_ref2m, de2mdT, qsat_ref2m, dqsat2mdT)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1192:         rh_ref2m(p) = min(100._r8, q_ref2m(p) / qsat_ref2m * 100._r8)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1197:            call KtoC(t_ref2m(p), tc_ref2m(p))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1199:            call Wet_BulbS(tc_ref2m(p),rh_ref2m(p), wbt_ref2m(p))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1202:            call swbgt(tc_ref2m(p), vap_ref2m(p), swbgt_ref2m(p))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1204:            call dis_coiS(tc_ref2m(p), rh_ref2m(p), wbt_ref2m(p), discomf_index_ref2mS(p))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1206:               call Wet_Bulb(t_ref2m(p), vap_ref2m(p), forc_pbot(c), rh_ref2m(p), q_ref2m(p), &
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1207:                             teq_ref2m(p), ept_ref2m(p), wb_ref2m(p))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1212:            wbt_ref2m_r(p)            = wbt_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1215:            swbgt_ref2m_r(p)          = swbgt_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90:1220:               ept_ref2m_r(p)            = ept_ref2m(p)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90:375:       temp1, temp2, temp12m, temp22m, fm, frictionvel_inst, landunit_index)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90:403:    real(r8) , intent(out)   :: temp12m ( lbn: )         ! relation for potential temperature profile applied at 2-m [lbn:ubn]
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90:436:    SHR_ASSERT_ALL((ubound(temp12m) == (/ubn/)), errMsg(sourcefile, __LINE__))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90:573:            temp1(n) = vkc/(log(-zetat*obu(n)/z0h(n))&
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90:578:            temp1(n) = vkc/(log(zldis(n)/z0h(n)) &
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90:582:            temp1(n) = vkc/(log(zldis(n)/z0h(n)) + 5._r8*zeta(n) - 5._r8*z0h(n)/obu(n))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90:584:            temp1(n) = vkc/(log(obu(n)/z0h(n)) + 5._r8 - 5._r8*z0h(n)/obu(n) &
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90:592:               temp2(n) = temp1(n)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90:614:               temp2(n) = temp1(n)
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90:641:            temp12m(n) = vkc/(log(-zetat*obu(n)/z0h(n))&
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90:646:            temp12m(n) = vkc/(log(zldis(n)/z0h(n)) &
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90:650:            temp12m(n) = vkc/(log(zldis(n)/z0h(n)) + 5._r8*zeta(n) - 5._r8*z0h(n)/obu(n))
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90:652:            temp12m(n) = vkc/(log(obu(n)/z0h(n)) + 5._r8 - 5._r8*z0h(n)/obu(n) &
/glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90:659:            temp22m(n) = temp12m(n)
"""Multi-process land split (FV3 duo lane): each process solves a share of the
land columns and one gather rebuilds the full land output on every process.

Simulates P processes in one: pass 1 records each process's gather payload,
pass 2 hands every process the stacked payloads, as the collective would.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.driver.model_driver import gather_land_step, split_land_columns

NCOL = 23


class _State(NamedTuple):
    T_soil: jax.Array      # (ncol, nlev) per column
    wet: jax.Array         # (ncol,) bool per column


def _step(state, cols):
    """Column-independent stand-in for the land step on columns ``cols``:
    the same tuple layout (state, 11 per-cell fields, 5 counts, iter max)."""
    T = state.T_soil.at[cols].set(state.T_soil[cols] * 1.01 + 0.5)
    wet = state.wet.at[cols].set(T[cols, 0] > 290.0)
    f = jnp.zeros(NCOL).at[cols].set(T[cols, 0] - 273.0)
    held = jnp.zeros(NCOL).at[cols].set((T[cols, 1] > 280.0).astype(float))
    n = jnp.sum(T[cols, 0] > 290.0).astype(jnp.int32)
    return ((_State(T, wet),) + (f,) * 7 + (held,) + (f, f, f + 1.0)
            + (n, n, n, 2 * n, n) + (jnp.max(cols).astype(jnp.int32),))


def _run_procs(state, land, n_proc):
    outs, payloads = [], {}
    split = [split_land_columns(land, n_proc, p) for p in range(n_proc)]
    for p in range(n_proc):           # pass 1: record payloads
        def rec(tree, p=p):
            payloads[p] = tree
            return jax.tree_util.tree_map(
                lambda x: np.stack([np.asarray(x)] * n_proc), tree)
        def rec_flag_then_payload(tree, p=p):
            if not isinstance(tree, tuple):
                return np.zeros(n_proc)
            return rec(tree)
        gather_land_step(lambda p=p: _step(state, split[p][0]), NCOL,
                         split[p][1], p, rec_flag_then_payload)
    for p in range(n_proc):           # pass 2: the real collective result
        def ag(tree):
            if not isinstance(tree, tuple):
                return np.zeros(n_proc)
            return jax.tree_util.tree_map(
                lambda *xs: np.stack([np.asarray(x) for x in xs]),
                *[payloads[q] for q in range(n_proc)])
        outs.append(gather_land_step(
            lambda p=p: _step(state, split[p][0]), NCOL, split[p][1], p, ag))
    return outs


@pytest.mark.parametrize("n_proc", [1, 3, 6])
def test_every_process_gets_the_single_process_result(n_proc):
    rng = np.random.default_rng(0)
    state = _State(jnp.asarray(rng.uniform(270, 300, (NCOL, 3))),
                   jnp.zeros(NCOL, bool))
    # 11 columns: uneven shares at 3 and 6 processes, so padding is exercised.
    land = np.array([0, 2, 3, 5, 8, 9, 10, 13, 17, 19, 22])
    ref = _step(state, jnp.asarray(land))
    for got in _run_procs(state, land, n_proc):
        for a, b in zip(jax.tree_util.tree_leaves(ref),
                        jax.tree_util.tree_leaves(got)):
            assert np.asarray(a).dtype == np.asarray(b).dtype
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_split_covers_every_column_once_and_refuses_empty_shares():
    land = np.arange(10, 21)
    shares = [split_land_columns(land, 4, p)[0] for p in range(4)]
    np.testing.assert_array_equal(np.sort(np.concatenate(shares)), land)
    _, padded = split_land_columns(land, 4, 0)
    assert padded.shape == (4, 3) and set(padded.ravel()) == set(land)
    with pytest.raises(ValueError, match="empty"):
        split_land_columns(np.arange(3), 4, 0)


def test_a_leaf_that_is_not_per_column_is_refused():
    state = _State(jnp.ones((NCOL, 3)) * 280.0, jnp.zeros(NCOL, bool))
    out = list(_step(state, jnp.arange(4)))
    out[1] = jnp.zeros(NCOL + 1)
    _, padded = split_land_columns(np.arange(4), 2, 0)
    with pytest.raises(ValueError, match="not per-column"):
        gather_land_step(lambda: tuple(out), NCOL, padded, 0,
                         lambda tree: np.zeros(2))


def test_a_failure_on_one_process_stops_every_process_before_the_payload():
    state = _State(jnp.ones((NCOL, 3)) * 280.0, jnp.zeros(NCOL, bool))
    _, padded = split_land_columns(np.arange(4), 2, 0)
    calls = []

    def ag(tree):                     # process 1 reports failure in the flag
        calls.append(isinstance(tree, tuple))
        return np.array([0.0, 1.0])

    with pytest.raises(RuntimeError, match="another process"):
        gather_land_step(lambda: _step(state, jnp.arange(2)), NCOL, padded,
                         0, ag)
    assert calls == [False], "the payload gather ran after a peer failed"

    def boom():
        raise FloatingPointError("local land failure")
    with pytest.raises(FloatingPointError):
        gather_land_step(boom, NCOL, padded, 1, lambda tree: np.array([0.0, 1.0]))
ls: cannot access '.venv/bin/python': No such file or directory

exec
/usr/bin/bash -lc "sed -n '14560,14825p' packages/coupler/legoesm/driver/model_driver.py; sed -n '16280,16445p' packages/coupler/legoesm/driver/model_driver.py" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
                # leaf when only params/lat were packed).
                _lml_cfg_p = _land_pack(_lml_cfg)
                _lml_params_p = _land_pack(_lml_params)
                _lml_lat_p = _land_pack(_lml_lat)
                _lml_carbon_p = (_land_pack(_lml_carbon)
                                 if _lml_carbon is not None else None)
                _f_land_cols_p = _f_land_cols[_land_out_idx]
                _owned_cols_p = _owned_cols[_land_out_idx]
            else:
                _lml_cfg_p = _lml_cfg
                _lml_params_p, _lml_lat_p = _lml_params, _lml_lat
                _lml_carbon_p = _lml_carbon
                _f_land_cols_p = _f_land_cols
                _owned_cols_p = _owned_cols

            # --- Land cadence: call the tile every _LAND_K host steps with
            # the forcing MEANED over the interval (a mean rate times the
            # tile's DT_LAND conserves the interval's precip mass and
            # radiant energy) and its fluxes/skin held in between — the
            # blend/consumer guards below are already None-tolerant, holding
            # is what they do before the first call today.  Time-based knob:
            # the interval, not the count, is what the canopy certifies, so
            # a dt change cannot silently stretch it.
            _LAND_K = (max(1, int(round(
                float(cfg.land_update_seconds) / DT)))
                if float(getattr(cfg, "land_update_seconds", 0.0)) > 0.0
                else 1)
            DT_LAND = _LAND_K * DT

            def _make_land_step(_dt_land):
              @jax.jit
              def _land_step(land_state, a2s, doy, lai_doy, lai_year):
                from legoesm.land.multilayer_land import (
                    step_multilayer_land_with_diagnostics)
                _state_in = (_land_pack(land_state) if _land_pack_on
                             else land_state)
                _a2s_in = _land_pack(a2s) if _land_pack_on else a2s
                if _lml_update is not None:
                    # Full-grid rebuild, then the same pack as the static
                    # params: identical column set, the packed gather only.
                    _p = mpas_land_step_params(
                        _lml_update, _lml_params,
                        land_state.theta_soil[:, 0], lai_doy, lai_year)
                    _params_in = _land_pack(_p) if _land_pack_on else _p
                else:
                    _params_in = _lml_params_p
                if _land_mesh is None:
                    new_state, resp, _carbon, _sfc = (
                        step_multilayer_land_with_diagnostics(
                            _state_in, _a2s_in, _lml_cfg_p, _lml_umin,
                            _dt_land, lat=_lml_lat_p, doy=doy,
                            land_params=_params_in,
                            carbon_state=_lml_carbon_p))
                else:
                    # per-column = the leaves the packing gathered (leading
                    # axis = the full column count before packing)
                    _full = (land_state, a2s, _lml_cfg,
                             _p if _lml_update is not None else _lml_params,
                             _lml_lat, _lml_carbon, doy)
                    new_state, resp, _carbon, _sfc = solve_land_on_devices(
                        lambda st, a2s, c, p, la, cb, d: (
                            step_multilayer_land_with_diagnostics(
                                st, a2s, c, _lml_umin, _dt_land, lat=la,
                                doy=d, land_params=p, carbon_state=cb)),
                        (_state_in, _a2s_in, _lml_cfg_p, _params_in,
                         _lml_lat_p, _lml_carbon_p, doy),
                        jax.tree_util.tree_map(
                            lambda x: (hasattr(x, "shape") and len(x.shape) >= 1
                                       and x.shape[0] == _land_ncol_full),
                            _full),
                        _land_mesh, _land_keep)
                # resp.albedo is the END-OF-STEP land albedo, already
                # snow-brightened by the tile (band_albedo / snow_albedo) and
                # dry-soil-brightened.  It used to be discarded here, so the
                # land tile's snow-albedo feedback never reached radiation.
                # resp.q_surface is the scheme's SOLVED boundary humidity;
                # resp.shflx / resp.lhflx are the fluxes its OWN energy
                # balance closed with.  The FLUXES are the coupling now: the
                # humidity handoff was measured insufficient (the canopy's
                # boundary humidity sits close to the air by construction, so
                # the atmosphere re-applying its own exchange coefficient
                # delivered ~a tenth of the solved flux -- Amazon latent heat
                # 78 W/m2 offline vs 7 coupled, land 10 K cold in 30 days).
                # A column whose surface solve was rejected is HELD: its
                # state is reverted and it conserves neither energy nor water
                # over that step. The land step is inside a compiled region
                # where a print is not available on a GPU-only runtime, so the
                # count comes out here and the loop below reports it. Without
                # that, a run whose land is quietly frozen somewhere looks
                # exactly like a healthy one.
                _n_held = (_sfc.n_held if _sfc.n_held is not None
                           else jnp.zeros((), jnp.int32))
                # The raw count spans EVERY mesh column, but ocean columns run
                # the land solve on placeholder forcing and their output is
                # discarded by the land-fraction weighting — a held ocean
                # column costs nothing physically.  Count the LAND-weighted
                # holds separately, so the log can tell a frozen continent
                # from noise on discarded columns (codex, 2026-08-23: the
                # undivided counter read as half the mesh held when the
                # physically-meaningful share was unknown).  Any land
                # fraction counts: a coastal column's land fluxes and stress
                # are blended in by its fraction, so its hold is not discarded.
                _held_mask = getattr(_sfc, "held", None)
                _n_held_land = (
                    jnp.sum((jnp.asarray(_held_mask).reshape(-1)
                             & (jnp.asarray(_f_land_cols_p) > 0.0))
                            .astype(jnp.int32))
                    if _held_mask is not None
                    else jnp.zeros((), jnp.int32))
                # the canopy Newton's iteration count (last Picard pass), MAX
                # over columns (M7 G0). The solve runs in column groups, each
                # to its own slowest column, so this bounds the step's cost
                # from below; it does not set it.
                _n_it = getattr(_sfc, "n_iters", None)
                _n_iter_max = (jnp.max(jnp.asarray(_n_it)).astype(jnp.int32)
                               if _n_it is not None
                               else jnp.zeros((), jnp.int32))
                # Unsolved canopy columns ACCEPTED with energy-closed fallback
                # fluxes, and those the fallback guards reverted instead (the
                # latter are already inside the held count). Same land mask as
                # the held count (any land fraction), so the two are comparable.
                _fb_mask = getattr(_sfc, "fallback", None)
                _n_fb_land = (
                    jnp.sum((jnp.asarray(_fb_mask).reshape(-1)
                             & (jnp.asarray(_f_land_cols_p) > 0.0))
                            .astype(jnp.int32))
                    if _fb_mask is not None
                    else jnp.zeros((), jnp.int32))
                _fb_rej_mask = getattr(_sfc, "fallback_rejected", None)
                _n_fb_rej = (
                    jnp.sum((jnp.asarray(_fb_rej_mask).reshape(-1)
                             & (jnp.asarray(_f_land_cols_p) > 0.0))
                            .astype(jnp.int32))
                    if _fb_rej_mask is not None
                    else (_sfc.n_fallback_rejected
                          if getattr(_sfc, "n_fallback_rejected", None)
                          is not None else jnp.zeros((), jnp.int32)))
                # The land's solved stress magnitude rho*u*^2 [Pa] (its canopy
                # roughness and stability) and the held mask; consumed only
                # under land_stress_from_land, where a held or non-finite
                # column keeps its last valid value (driver loop below).
                # (sqrt guarded so a zeroed column has a finite derivative.)
                _taumag = solved_stress_magnitude(_sfc, resp)
                _held_f = (jnp.asarray(_held_mask).reshape(-1)
                           .astype(_taumag.dtype)
                           if _held_mask is not None
                           else jnp.zeros_like(_taumag))
                # Owned columns with any land whose soil-water solve kept an
                # unconverged iterate (any water it left is in water_created).
                _unconv_mask = getattr(_sfc, "soil_unconverged", None)
                _n_unconv_land = (
                    jnp.sum((jnp.asarray(_unconv_mask).reshape(-1)
                             & (jnp.asarray(_f_land_cols_p) > 0.0)
                             & _owned_cols_p)
                            .astype(jnp.int32))
                    if _unconv_mask is not None
                    else jnp.zeros((), jnp.int32))
                # The land's RADIATING pair (emission-equivalent temperature
                # and column emissivity) its energy balance closed with:
                # LW_out = eps*sigma*T_rad^4 + (1 - eps)*La.  The surface
                # blend hands both to radiation (tier-5 U11-1), so the
                # atmosphere's upward LW is the one the land lost.
                _emis = jnp.broadcast_to(
                    jnp.asarray(resp.emissivity, resp.T_rad.dtype),
                    resp.T_rad.shape)
                # The scheme's own 2 m air temperature (CLM5 t_ref2m) for the
                # CMOR ``tas`` land leg; NaN where the scheme solves none or the
                # column was held (no solved surface layer this call).
                _t2m = (jnp.full_like(resp.T_rad, jnp.nan)
                        if getattr(_sfc, "T_2m", None) is None
                        else jnp.asarray(_sfc.T_2m, resp.T_rad.dtype))
                _t2m = jnp.where(_held_f > 0.5, jnp.nan, _t2m)
                if _land_pack_on:
                    # Scatter the advanced columns back into the full-grid
                    # state (ocean columns keep their frozen init values,
                    # exactly what the unpacked solve left them at after the
                    # f_land blend discarded its work) and the five consumed
                    # response fields into zero-filled cell arrays.
                    new_state = scatter_land_columns(
                        land_state, new_state,
                        _land_out_idx, _land_ncol_full)
                    return ((new_state,)
                            + tuple(scatter_cells(
                                o, _land_out_idx, _land_ncol_full)
                                for o in (
                                    resp.T_sfc, resp.albedo, resp.q_surface,
                                    resp.shflx, resp.lhflx, resp.surface_mass_flux,
                                    _taumag, _held_f, resp.T_rad, _emis))
                            # 2 m T: NaN (= no land screen value) off the
                            # land columns, never a valid-looking 0 K
                            + (jnp.full((_land_ncol_full,), jnp.nan, _t2m.dtype)
                               .at[_land_out_idx].set(_t2m.reshape(-1)),)
                            + (_n_held, _n_held_land, _n_fb_land,
                               _n_fb_rej, _n_unconv_land, _n_iter_max))
                return (new_state, resp.T_sfc, resp.albedo, resp.q_surface,
                        resp.shflx, resp.lhflx, resp.surface_mass_flux,
                        _taumag, _held_f, resp.T_rad, _emis, _t2m,
                        _n_held, _n_held_land, _n_fb_land, _n_fb_rej,
                        _n_unconv_land, _n_iter_max)
              if not _land_none:
                  return _land_step

              @jax.jit
              def _land_step_none(land_state, a2s, doy, lai_doy, lai_year):
                  # Same output structure and dtypes as the solve (traced
                  # shape-only, never run), so the land halo exchange packs
                  # the same message on every rank.
                  out = jax.eval_shape(
                      _land_step, land_state, a2s, doy, lai_doy, lai_year)
                  # (the 2 m T, last per-cell field, is NaN = none solved)
                  return (land_state,) + tuple(
                      jnp.full(o.shape, jnp.nan, o.dtype)
                      if i == _LAND_STEP_N_CELL - 1
                      else jnp.zeros(o.shape, o.dtype)
                      for i, o in enumerate(out[1:]))
              return _land_step_none

            _land_step_fn = _make_land_step(DT_LAND)
            # Bootstrap variant: one host step's forcing, tile advanced by
            # DT — used only for the first call after a (re)start so the
            # surface blend never runs land-free for a whole interval.
            _land_step_boot_fn = (
                _land_step_fn if _LAND_K == 1 else _make_land_step(DT))
            if _land_split is not None:
                from jax.experimental import multihost_utils as _mhu
                _proc = jax.process_index()

                def _with_gather(step):
                    def _step(*a):
                        return gather_land_step(
                            lambda: step(*a), _land_ncol_full, _land_split,
                            _proc, _mhu.process_allgather)
                    return _step
                _land_step_boot_fn = (
                    _with_gather(_land_step_boot_fn)
                    if _land_step_boot_fn is not _land_step_fn else None)
                _land_step_fn = _with_gather(_land_step_fn)
                if _land_step_boot_fn is None:
                    _land_step_boot_fn = _land_step_fn

            # Phase 2b (#1312): per-cell root-zone beta_soil -> the traced
            # ``forcing["beta_land"]`` the turbulence surface flux consumes.
            # Same helper (land_tile_beta_soil) and thresholds the coupled
            # pipeline's land tile applies — one formula, one place.
            _land_beta_fn = None
            if _land_beta_soil_on:
                from legoesm.land.multilayer_land import land_tile_beta_soil

                @jax.jit
                def _land_beta_fn(land_state):
                    return jnp.clip(
                        land_tile_beta_soil(
                            land_state.theta_soil, land_state.T_soil,
                            _lml_cfg, _lml_params),
                        0.0, 1.0)

                # The land's SOLVED surface humidity is handed to the
                # turbulence VERBATIM via forcing["q_sfc_land"] -- review
                # refuted the previous effective-beta round trip here (its
                # saturation anchors disagreed by 0.3-2.5 % and its clip could
                # only shrink the flux), so there is no inversion any more.

                if _land_beta != 1.0:
                    logger.info(
                        "  land_beta_soil: traced per-cell beta_soil "
                        "REPLACES the static land_beta=%.2f over land",
                    # tile consumes the interval MEAN, so the interval's
                    # precip mass and radiant energy are conserved exactly
                    # (mean rate x DT_LAND = sum of per-step rate x DT).
                    # With _LAND_K == 1 the mean is the identity and the
                    # trajectory is unchanged.  cos_zenith is accumulated
                    # SW-WEIGHTED separately (weight sw_down + 1 W/m2, so it
                    # degrades to the plain mean at night): the canopy puts
                    # cos_zenith under the direct-beam extinction, and a
                    # plain mean over an interval that straddles the
                    # terminator hands it a small-positive sun with full
                    # daytime SW (GLM review, 2026-08-26).
                    _land_a2s_sum = (
                        _a2s if _land_a2s_sum is None
                        else jax.tree_util.tree_map(
                            jnp.add, _land_a2s_sum, _a2s))
                    _land_cz_w = _a2s.sw_down + 1.0
                    if _land_a2s_n == 0:
                        _land_cz_wsum = _a2s.cos_zenith * _land_cz_w
                        _land_w_sum = _land_cz_w
                    else:
                        _land_cz_wsum = (
                            _land_cz_wsum + _a2s.cos_zenith * _land_cz_w)
                        _land_w_sum = _land_w_sum + _land_cz_w
                    _land_a2s_n += 1
                # A full interval fires the cadence step; a SINGLE sample
                # fires the bootstrap step when no held outputs exist yet
                # (run start and every chain-link restart: the held fluxes
                # are loop-locals, so without this the first _LAND_K steps
                # of a resumed link would run with NO land in the surface
                # blend at all — codex P1, 2026-08-26).
                _land_call = (_land_a2s_n >= _LAND_K
                              or (_land_T_skin is None and _land_a2s_n >= 1))
                if _land_call:
                    _a2s_mean = (
                        _land_a2s_sum if _land_a2s_n == 1
                        else jax.tree_util.tree_map(
                            lambda s: s / _land_a2s_n, _land_a2s_sum))
                    _a2s_mean = _a2s_mean._replace(
                        cos_zenith=_land_cz_wsum / _land_w_sum)
                    _land_fn = (_land_step_fn if _land_a2s_n >= _LAND_K
                                else _land_step_boot_fn)
                    _land_a2s_sum = None
                    _land_a2s_n = 0
                    _pt0 = time.perf_counter() if _PHASE_TIMERS else 0.0
                    (self._land_ml_state, _land_T_skin,
                     _land_albedo_cells, _land_qsfc_step,
                     _land_shflx_step, _land_lhflx_step, _land_evap_step,
                     _land_taumag_step, _land_held_step,
                     _land_Trad_cells, _land_emis_cells, _land_T2m_step,
                     _land_n_held_step, _land_n_held_land_step,
                     _land_n_fb_step, _land_n_fb_rej_step,
                     _land_n_unconv_step, _land_n_iter_max) = _land_fn(
                        self._land_ml_state, _a2s_mean,
                        jnp.asarray(_doy, dtype=jnp.float64),
                        # 0-based days since Jan 1 (the calibration's clock)
                        # and the cover year, same formula as the transient
                        # cover rebuild.  Traced: no retrace per step.
                        jnp.asarray(_doy - 1.0, dtype=jnp.float64),
                        jnp.asarray(_lml_year0 + _force_day / 365.0,
                                    dtype=jnp.float64))
                    if _PHASE_TIMERS:
                        jax.block_until_ready(_land_T_skin)
                        _phase_t["land"] += time.perf_counter() - _pt0
                        _phase_t["land_iter_max"] = max(
                            _phase_t["land_iter_max"], int(_land_n_iter_max))
                    # Mirror the land tile's skin and surface humidity onto
                    # the driver, the same way the ice skin above is mirrored:
                    # the CMOR ``tas`` diagnostic runs in a different method and
                    # cannot see these closure locals, and without them it
                    # anchors its profile on the neighbouring OCEAN skin over
                    # land and publishes a land temperature the model never had.
                    self._land_T_skin_last = _land_T_skin
                    self._land_qsfc_last = _land_qsfc_step
                    self._land_T2m_last = _land_T2m_step
                    # Accumulate ON DEVICE and read at the same cadence the
                    # other post-step warnings use: reading it every step
                    # would stall the accelerator once per step for a number
                    # that is almost always zero.
                    _land_n_held_accum = _land_n_held_accum + _land_n_held_step
                    _land_n_unconv_accum = _land_n_unconv_accum + _land_n_unconv_step
                    _land_n_held_land_accum = (
                        _land_n_held_land_accum + _land_n_held_land_step)
                    _land_n_held_steps_accum = (
                        _land_n_held_steps_accum
                        + (_land_n_held_step > 0).astype(jnp.int32))
                    _land_n_fb_accum = _land_n_fb_accum + _land_n_fb_step
                    _land_n_fb_rej_accum = (
                        _land_n_fb_rej_accum + _land_n_fb_rej_step)
                    # Published only under the same switch that threads f_land
                    # into the turbulence factory: without the land fraction
                    # the consumer refuses the key, and adding it mid-run
                    # would change the forcing pytree and retrace.
                    if _land_beta_soil_on:
                        _land_qsfc_cells = _land_qsfc_step
                        _land_shflx_cells = _land_shflx_step
                        _land_lhflx_cells = _land_lhflx_step
                        _land_evap_cells = _land_evap_step
                        if _land_stress_on:
                            # Last valid land stress per column: a held or
                            # non-finite solve keeps the previous value.
                            (_land_taumag_cells, _land_taumag_valid,
                             _land_taumag_fresh) = hold_last_valid_land_stress(
                                _land_taumag_cells, _land_taumag_valid,
                                _land_taumag_step, _land_held_step)
                            self._land_taumag = _land_taumag_cells
                            self._land_taumag_valid = _land_taumag_valid
                            self._land_taumag_fresh = _land_taumag_fresh
                    if _land_beta_fn is not None and _land_qsfc_cells is None:
                        # Root-zone beta only until the humidity channel is
                        # live (or when the scheme solves none).
                        _land_beta_cells = _land_beta_fn(self._land_ml_state)
                    if _is_mpas_cell_partitioned(self):
                        # A halo cell steps its OWN land column, which is
                        # never exchanged and drifts from the owner's; the
                        # turbulence's cell->edge tendency projection reads
                        # the halo cell's surface stress, so without this an
                        # owned boundary edge is forced by a land column
                        # nobody owns (measured: 1e-8 m/s restart drift on
                        # exactly the boundary edges, zero with the land
                        # stress off).  ``_land_call`` is rank-uniform
                        # (step cadence + a loop-local None test), so every
                        # rank issues this collective together.
                        (_land_T_skin, _land_albedo_cells, _land_qsfc_step,
                         _land_qsfc_cells, _land_shflx_cells,
                         _land_lhflx_cells, _land_evap_cells,
                         _land_beta_cells, _land_taumag_cells,
                         _land_taumag_valid, _land_taumag_fresh,
                         _land_Trad_cells, _land_emis_cells
                         ) = _exchange_halo_cell_fields(
                            self._voronoi_layout.halo_exchange,
                            (_land_T_skin, _land_albedo_cells,
                             _land_qsfc_step, _land_qsfc_cells,
                             _land_shflx_cells, _land_lhflx_cells,
                             _land_evap_cells, _land_beta_cells,
                             _land_taumag_cells, _land_taumag_valid,
                             _land_taumag_fresh,
                             _land_Trad_cells, _land_emis_cells))
                        self._land_T_skin_last = _land_T_skin
                        self._land_qsfc_last = _land_qsfc_step
                        if _land_taumag_cells is not None:
                            self._land_taumag = _land_taumag_cells
                            self._land_taumag_valid = _land_taumag_valid
                            self._land_taumag_fresh = _land_taumag_fresh
                # Publish the cadence state for the checkpoint writer (F31):
                # Python references only, no device work.
                self._land_window = land_window_checkpoint_payload(
                    _land_a2s_sum, _land_a2s_n,
                    _land_cz_wsum if _land_a2s_n else None,
                    _land_w_sum if _land_a2s_n else None,
                    (_land_T_skin, _land_albedo_cells,
                     getattr(self, "_land_qsfc_last", None),
                     _land_qsfc_cells, _land_shflx_cells,
                     _land_lhflx_cells, _land_evap_cells, _land_beta_cells,
                     _land_Trad_cells, _land_emis_cells))
                if (step % _HARD_SAT_LOG_CADENCE_STEPS) == 0:
                    if _ls_seed_win is not None:
                        self._flush_land_stress_counts(
                            _ls_seed_win, _ls_reused_win, step)
                        _ls_seed_win = jnp.zeros((), jnp.int32)
                        _ls_reused_win = jnp.zeros((), jnp.int32)
                    _window_cols = int(_land_n_held_accum)
                    _window_land = int(_land_n_held_land_accum)
                    _window_steps = int(_land_n_held_steps_accum)
                    _land_n_held_accum = jnp.zeros((), jnp.int32)
                    _land_n_held_land_accum = jnp.zeros((), jnp.int32)
                    _land_n_held_steps_accum = jnp.zeros((), jnp.int32)

exec
/usr/bin/bash -lc "sed -n '250,289p' packages/land/legoesm/land/canopy/stability.py; sed -n '430,482p' packages/land/legoesm/land/canopy/stability.py; rg -n 'fluxes_per_col|Tc_cvg|Ta =|Ta +|z_ref|displa' packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py; rg -n 'fallback|T_canopy_air|tree_map|_replace' packages/land/legoesm/land/multilayer_land.py" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
    ustar3 = constants.kappa_von_karman * um / (
        jnp.log(zldis / z0m)
        - psi_m(zeta, _STABLE_SCHEME) + psi_m(z0m / obu, _STABLE_SCHEME))

    ustar = jnp.where(zeta < -zetam, ustar1,
            jnp.where(zeta < 0.0,    ustar2, ustar3))
    return ustar


def _temperature_humidity_relation(zldis: jax.Array, obu: jax.Array,
                                   z0h: jax.Array) -> jax.Array:
    """Compute θ* / (θ_atm - θ_sfc) (4 regimes, CLM5 / DifferBESS)."""
    zetat = _ZETAT
    zeta  = zldis / obu

    # Very unstable (valid: obu < 0, zeta < -zetat).  Floor log/cbrt args; no-ops here.
    ch1 = constants.kappa_von_karman / (
        jnp.log(jnp.maximum(-zetat * obu / z0h, _MOST_ARG_FLOOR))
        - _stability_func_heat(-zetat)
        + _stability_func_heat(z0h / obu)
        + _MOST_HEAT_CONV_COEF * (1.0 / jnp.cbrt(zetat)
                                  - 1.0 / jnp.cbrt(jnp.maximum(-zeta, zetat)))
    )
    ch2 = constants.kappa_von_karman / (
        jnp.log(zldis / z0h)
        - _stability_func_heat(zeta)
        + _stability_func_heat(z0h / obu)
    )
    # Stable (zeta >= 0): Beljaars & Holtslag (1991), all stable zeta.  The core
    # psi_h floors its stable argument at 1e-10, so the (1 + 2z/3)^1.5 term stays
    # finite for the negative arguments of the discarded (obu < 0) branch.
    ch3 = constants.kappa_von_karman / (
        jnp.log(zldis / z0h)
        - psi_h(zeta, _STABLE_SCHEME) + psi_h(z0h / obu, _STABLE_SCHEME))

    ch = jnp.where(zeta < -zetat, ch1,
         jnp.where(zeta < 0.0,    ch2, ch3))
    return ch


               returned ``rah``/``raw`` were evaluated with (the last
               iteration's INCOMING one; ``zeta`` is the updated stability)
    """
    # Heat roughness length.  CLM5 vegetation uses kB^-1 = 0 (z0h = z0m); the
    # earlier BESS default kB^-1 = 2 (z0h = z0m / e^2) over-suppressed heat
    # exchange and warm-biased canopy/skin temperature (DifferBESS aa6e8b9).
    z0h  = z0m   # kB^-1 = 0
    dth  = Ta - Tc
    dq   = q_atm - q_c
    dthv = (Ta - Tc) * (1.0 + _VIRT_T_COEF * q_atm) + _VIRT_T_COEF * Ta * (q_atm - q_c)

    um, obu = _monin_obukhov_init(ur, Tv_atm, dthv, zldis, z0m, zeta_cap_width)

    init    = jnp.array([z0h, obu, um])
    forcing = jnp.array([Ta, Tv_atm, q_atm, q_atm, zldis, z0m, dq, dth, ur])
    step_fn = partial(_stability_step, forcing=forcing,
                      zeta_cap_width=zeta_cap_width)

    # Only the LAST iterate's outputs are used, so carry them instead of
    # stacking all n_iters: the stacked (n_iters, ...) buffer was rewritten
    # every iteration, ~half the cost of this function under the canopy
    # solve's jacfwd + vmap on CPU (production res6, bit-identical).
    if n_iters < 1:
        raise ValueError(f"monin_obukhov_stability needs n_iters >= 1, got {n_iters!r}")

    def _step_keep_last(c, _xs):
        carry, _ = c
        return step_fn(carry, None), None

    # Placeholder output with the step's own shape/dtype (scalar or batched
    # inputs alike); it is overwritten by the first iterate.
    _out = jax.eval_shape(step_fn, init, None)[1]
    (final_carry, last), _ = jax.lax.scan(
        _step_keep_last, (init, jnp.zeros(_out.shape, _out.dtype)), xs=None,
        length=n_iters)

    ustar, ch, zeta = last[0], last[4], last[5]

    rah = 1.0 / jnp.maximum(ch * ustar, 1e-9)
    raw = rah  # same for sensible heat and water vapour (neutral Prandtl)

    um_final = final_carry[2]
    ram      = jnp.maximum(um_final / ustar**2, 1e-9)
    uav      = jnp.sqrt(um_final / ram)

    if return_flux_obu:
        return ustar, rah, raw, uav, zeta, last[6]
    return ustar, rah, raw, uav, zeta


# ---------------------------------------------------------------------------
# Leaf boundary-layer resistance
# ---------------------------------------------------------------------------
82:# at Tc = Ta is supersaturated whenever the soil is warmer than the air (measured
123:_DEFAULT_RD = 0.67          # [-] displacement-height / hc ratio
224:    land_config,                      # MultiLayerLandConfig or LandConfig (for z_ref, beta_min)
249:                  only ``z_ref``, ``emissivity_land``, ``albedo_land`` are
412:    z0m, displa = compute_aerodynamics(hc, LAI, rz0m, rd)
413:    # Lift reference height above the displacement + roughness sub-layer.
414:    Ta, z_ref_base = surface_reference_state(
415:        forcing.T_lowest, land_config.z_ref, forcing.z_lowest)
416:    z_ref = jnp.maximum(z_ref_base, displa + 10.0 * z0m + 2.0)
436:    Tv_atm = Ta * (1.0 + _VIRT_T_COEF * q_atm)
451:    _Ta_retry = Ta - _RETRY_COOLING_K
496:            ur=wind_speed, CI=CI, z0m=z0m, displa=displa, z0=z_ref,
569:        fluxes_per_col = jax.vmap(_fwd_one_col)(x_final, bundles_k)
571:        G_k = jnp.clip(fluxes_per_col["G"], -500.0, 700.0)  # coeff-ok: physical range clamp on ground heat flux [W m-2]
579:    Tc_cvg = x_final[:, 4]   # canopy air-space temperature (aerodynamic node)
583:    LE_Sun  = fluxes_per_col["LE_Sun"]
584:    LE_Sh   = fluxes_per_col["LE_Sh"]
587:    LE_wet_canopy_d = fluxes_per_col["LE_wet_Sun"] + fluxes_per_col["LE_wet_Sh"]
588:    LE_Soil = fluxes_per_col["LE_Soil"]
589:    H_Sun   = fluxes_per_col["H_Sun"]
590:    H_Sh    = fluxes_per_col["H_Sh"]
591:    H_Soil  = fluxes_per_col["H_Soil"]
592:    An_Sun  = fluxes_per_col["An_Sun"]        # NET (drives SIF + leaf coupling)
593:    An_Sh   = fluxes_per_col["An_Sh"]
594:    Agross_Sun = fluxes_per_col["Agross_Sun"]  # GROSS (carbon-model GPP)
595:    Agross_Sh  = fluxes_per_col["Agross_Sh"]
596:    G       = fluxes_per_col["G"]
597:    gs_Sun  = fluxes_per_col["gs_Sun"]
598:    gs_Sh   = fluxes_per_col["gs_Sh"]
599:    ustar   = fluxes_per_col["ustar"]
600:    Rn_Sun_d  = fluxes_per_col["Rn_Sun"]
601:    Rn_Sh_d   = fluxes_per_col["Rn_Sh"]
602:    Rn_Soil_d = fluxes_per_col["Rn_Soil"]
609:    condensation = fluxes_per_col["condensation"]
629:    # takes T_phot = Ta if use_ta_for_photosynthesis else Tf (solver.py) — else
635:        T_phot_Sun = Ta if cc.use_ta_for_photosynthesis else Tf_Sun
636:        T_phot_Sh  = Ta if cc.use_ta_for_photosynthesis else Tf_Sh
674:    LW_out_col = fluxes_per_col["LW_out"]
675:    eps_eff    = fluxes_per_col["eps_col"]
676:    LW_emit    = fluxes_per_col["LW_emit"]
732:        T_canopy_air=Tc_cvg,
733:        T_2m=screen_level_temperature(Ta, Tc_cvg, z_ref - displa,
734:                                      fluxes_per_col["obu"], z0m),
82:# --- Unsolved-canopy fallback guards (containment thresholds, NOT physics) ---
84:# scheme's fallback fluxes, energy-closed into sensible heat (CLM CanopyFluxes
132:def _get(lp, name: str, fallback):
133:    """Read a per-column field from ``lp`` if present, else return ``fallback``.
135:    Uses the safe ``getattr(lp, name, fallback)`` form because ``lp`` can be
147:        return fallback
148:    v = getattr(lp, name, fallback)
149:    return fallback if v is None else v
556:    # Spatially-varying surface parameters (or config scalar fallbacks).
835:                return _snow_skin(pack._replace(T=T_s), T_g[:, 0], f_snow)
910:        # these bands) * sw_down in daylight (the RT's own low-light fallback,
926:            lp = lp._replace(ALB_VIS=_vis(lp.ALB_VIS), ALB_NIR=_nir(lp.ALB_NIR))
1016:        clmml_forcing = forcing._replace(
1080:    # A column whose canopy solve did not converge carries the scheme's fallback
1091:    _fallback_ok = (surface_out.converged is not None
1094:    if _fallback_ok:
1096:        _fallback_resid = surface_out.Rn_ext - (shflx + lhflx + G_surface)
1097:        shflx = jnp.where(_unsolved, shflx + _fallback_resid, shflx)
1225:    #     ``LE_soil is None`` fallback (whole positive latent -> L_v soil) is a
1338:        pack = pack._replace(swe_ice=pack.swe_ice.at[:, 0].set(
1552:            pack._replace(T=T_pack_solved), scc)
2004:        # CLMMLCanopyConfig populates T_canopy_air in _extract_surface_fluxes;
2006:        response_T_sfc = (surface_out.T_canopy_air
2007:                          if surface_out.T_canopy_air is not None
2054:    # carbon pools and exchanges no CO2: the fallback GPP is the cold state's,
2060:        fallback_ok=_fallback_ok)
2061:    surface_out = surface_out._replace(
2062:        held=_held_mask, n_held=_n_held, fallback=_fb_mask,
2063:        n_fallback=_n_fb, fallback_rejected=_fb_rej_mask,
2064:        n_fallback_rejected=_n_fb_rej,
2067:        surface_out = surface_out._replace(
2110:    return state._replace(
2123:                           fallback_ok=False):
2130:      scheme's fallback rather than a solution of the surface energy balance; or
2139:    A FINITE UNSOLVED column, when ``fallback_ok`` (the caller closed its energy
2142:    snow and the response advance with the fallback fluxes, which close
2144:    flux is zero (the fallback GPP is not physics), and its canopy warm-start
2148:    ``FALLBACK_MAX_FLUX_W_M2``.  Without ``fallback_ok`` every unsolved column
2173:    fallback_mask (ncol,) bool, n_fallback () int32 -- accepted unsolved columns
2211:    if fallback_ok:
2217:        fallback = candidate & ~guard_fail
2220:        fallback = jnp.zeros(ncol, dtype=bool)
2255:    # fallback cannot honour D3, so the column's state and response are reverted
2266:            rejected = rejected | fallback
2267:            fallback = jnp.zeros(ncol, dtype=bool)
2269:    bad = nonfinite | (unsolved & ~fallback)
2271:    n_fallback = jnp.sum(fallback.astype(jnp.int32))
2276:    _cache_mask = bad | fallback
2277:    held_state = new_state._replace(
2307:        fallback_val = inert.get(name, jnp.zeros_like(arr))
2308:        arr = jnp.where(bad, jnp.broadcast_to(fallback_val, arr.shape), arr)
2311:            arr = jnp.where(fallback, jnp.zeros_like(arr), arr)
2314:    held_response = response._replace(
2319:            fallback, n_fallback, rejected, n_rejected)
2480:    return state._replace(T_snow=T_snow.astype(T0.dtype))
2540:    return jax.tree_util.tree_map(
2554:    return jax.tree_util.tree_map(

exec
/usr/bin/bash -lc "sed -n '560,588p' /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90; sed -n '630,657p' /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90; sed -n '1168,1187p' /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90; command -v python; ls /glade/u/home/pg2328/legoesm/.venv/bin/python" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 exited 2 in 0ms:
            end if
            va(n) = um(n)
         end if

         ! Temperature profile

         if (present(landunit_index)) then
            zldis(n) = forc_hgt_t_patch(pfti(n))-displa(n)
         else
            zldis(n) = forc_hgt_t_patch(n)-displa(n)
         end if
         zeta(n) = zldis(n)/obu(n)
         if (zeta(n) < -zetat) then
            temp1(n) = vkc/(log(-zetat*obu(n)/z0h(n))&
                 - StabilityFunc2(-zetat) &
                 + StabilityFunc2(z0h(n)/obu(n)) &
                 + 0.8_r8*((zetat)**(-0.333_r8)-(-zeta(n))**(-0.333_r8)))
         else if (zeta(n) < 0._r8) then
            temp1(n) = vkc/(log(zldis(n)/z0h(n)) &
                 - StabilityFunc2(zeta(n)) &
                 + StabilityFunc2(z0h(n)/obu(n)))
         else if (zeta(n) <=  1._r8) then
            temp1(n) = vkc/(log(zldis(n)/z0h(n)) + 5._r8*zeta(n) - 5._r8*z0h(n)/obu(n))
         else
            temp1(n) = vkc/(log(obu(n)/z0h(n)) + 5._r8 - 5._r8*z0h(n)/obu(n) &
                 + (5._r8*log(zeta(n))+zeta(n)-1._r8))
         end if

         ! Humidity profile
                  temp2(n) = vkc/(log(obu(n)/z0q(n)) + 5._r8 - 5._r8*z0q(n)/obu(n) &
                       + (5._r8*log(zeta(n))+zeta(n)-1._r8))
               end if
            endif
         endif

         ! Temperature profile applied at 2-m

         zldis(n) = 2.0_r8 + z0h(n)
         zeta(n) = zldis(n)/obu(n)
         if (zeta(n) < -zetat) then
            temp12m(n) = vkc/(log(-zetat*obu(n)/z0h(n))&
                 - StabilityFunc2(-zetat) &
                 + StabilityFunc2(z0h(n)/obu(n)) &
                 + 0.8_r8*((zetat)**(-0.333_r8)-(-zeta(n))**(-0.333_r8)))
         else if (zeta(n) < 0._r8) then
            temp12m(n) = vkc/(log(zldis(n)/z0h(n)) &
                 - StabilityFunc2(zeta(n))  &
                 + StabilityFunc2(z0h(n)/obu(n)))
         else if (zeta(n) <=  1._r8) then
            temp12m(n) = vkc/(log(zldis(n)/z0h(n)) + 5._r8*zeta(n) - 5._r8*z0h(n)/obu(n))
         else
            temp12m(n) = vkc/(log(obu(n)/z0h(n)) + 5._r8 - 5._r8*z0h(n)/obu(n) &
                 + (5._r8*log(zeta(n))+zeta(n)-1._r8))
         end if

         ! Humidity profile applied at 2-m

         qflx_evap_soi(p) = forc_rho(c)*wtgq(p)*delq(p)

         ! compute individual latent heat fluxes
         delq_snow = wtalq(p)*qg_snow(c)-wtlq0(p)*qsatl(p)-wtaq0(p)*forc_q(c)
         qflx_ev_snow(p) = forc_rho(c)*wtgq(p)*delq_snow

         delq_soil = wtalq(p)*qg_soil(c)-wtlq0(p)*qsatl(p)-wtaq0(p)*forc_q(c)
         qflx_ev_soil(p) = forc_rho(c)*wtgq(p)*delq_soil

         delq_h2osfc = wtalq(p)*qg_h2osfc(c)-wtlq0(p)*qsatl(p)-wtaq0(p)*forc_q(c)
         qflx_ev_h2osfc(p) = forc_rho(c)*wtgq(p)*delq_h2osfc

         ! 2 m height air temperature

         t_ref2m(p) = thm(p) + temp1(p)*dth(p)*(1._r8/temp12m(p) - 1._r8/temp1(p))
         t_ref2m_r(p) = t_ref2m(p)

         ! 2 m height specific humidity

         q_ref2m(p) = forc_q(c) + temp2(p)*dqh(p)*(1._r8/temp22m(p) - 1._r8/temp2(p))
/glade/u/apps/derecho/25.10/opt/view/bin/python
ls: cannot access '/glade/u/home/pg2328/legoesm/.venv/bin/python': No such file or directory

codex
The incoming Obukhov length now feeds the screen profile without changing the flux formulas. The packed, empty-rank, and gather paths also use the new tuple slot consistently. I’m checking the fallback path and running the focused tests next.
exec
/usr/bin/bash -lc 'PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m pytest -p no:cacheprovider tests/land/unit/test_land_screen_temperature.py tests/unit/test_land_column_split.py tests/land/unit/test_canopy_stability.py tests/land/unit/test_canopy_stability_faithful.py -q' in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 exited 1 in 3ms:
/glade/u/apps/derecho/25.10/opt/view/bin/python: No module named pytest

exec
/usr/bin/bash -lc "sed -n '1980,2080p' packages/land/legoesm/land/multilayer_land.py; sed -n '2250,2320p' packages/land/legoesm/land/multilayer_land.py; sed -n '810,922p' packages/coupler/legoesm/driver/diagnostics.py; sed -n '16342,16420p' packages/coupler/legoesm/driver/model_driver.py" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
            _, gpp_override, _ = compute_effective_beta(
                T_surface_new, forcing, beta_soil_new, config, carbon_state,
                dt, land_params=lp)
        carbon_state_new, co2_flux = step_carbon(
            carbon_state, forcing.sw_down, T_surface_new, forcing.co2_ppmv,
            beta_soil_new, lat_arr, doy, forcing.precip_total, config.carbon,
            dt, gpp_override=gpp_override,
            soil_frozen_fraction=soil_frozen_fraction,
        )
    else:
        carbon_state_new = carbon_state
        co2_flux = jnp.zeros(ncol)

    # --- TileResponse ---
    # ``T_sfc`` is the AERODYNAMIC surface temperature for the atmosphere's
    # sensible-heat coupling.  For the two-leaf canopy that is the canopy
    # air-space temperature ``Tc`` (the exchange node, H_tot = rho*cp*(Tc-Ta)/Ra);
    # for SimpleSEB it is the top-soil surface temperature ``T_surface_new``.
    # Static dispatch on the (compile-time) surface-scheme type — feature gating,
    # not data-dependent selection.
    if isinstance(config.surface_scheme, (TwoLeafCanopyConfig, CLMMLCanopyConfig)):
        # For both canopy schemes the coupler must receive the canopy-consistent
        # LW_up (computed inside the canopy RT, not recomputed from post-step soil T)
        # and the aerodynamic canopy-air temperature (not the soil skin T).
        # CLMMLCanopyConfig populates T_canopy_air in _extract_surface_fluxes;
        # fall back to T_surface if absent (should not happen post-Iter 9).
        response_T_sfc = (surface_out.T_canopy_air
                          if surface_out.T_canopy_air is not None
                          else surface_out.T_surface)
        response_lw_up = surface_out.lw_up          # canopy LW_out
    else:
        response_T_sfc = T_skin_new                 # SimpleSEB: surface skin temp
        response_lw_up = lw_up_new                  # recomputed from post-step skin T
    response = TileResponse(
        T_sfc=response_T_sfc,
        # Emission-equivalent canopy temperature for the LW boundary
        # (eps_col*sigma*T_surface^4 = LW_emit); T_sfc above is the aerodynamic
        # (Tc for canopy / top-soil for SimpleSEB) temperature for sensible heat.
        # The tile blend emits with this radiometric T_rad, not the aerodynamic
        # temp, to conserve LW for vegetated cells.
        T_rad=surface_out.T_surface,
        albedo=alpha_new,
        emissivity=surface_out.emissivity,
        z0=surface_out.z0,
        q_surface=q_sfc_new,
        shflx=shflx,
        # Latent heat to the atmosphere = the evaporative/sublimation demand PLUS the
        # blowing-snow sublimation (gap 5): its L_s was charged to the surface energy
        # budget, so it must reach the atmosphere as latent heat (0 when bands off).
        lhflx=lhflx_actual + blow_subl * _L_s_T - _cond_liq * _L_v_T,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=response_lw_up,
        u_ocean_sfc=jnp.zeros(ncol),
        v_ocean_sfc=jnp.zeros(ncol),
        co2_flux=co2_flux,
        # Freshwater leaving the column to the ocean = total Richards runoff
        # (surface + subsurface) [kg/m²/s]; closes the coupler water budget.
        freshwater_flux=runoff_surface_total + richards_out.runoff_subsurface,
        # Land does not extract heat or exert stress on the ocean.
        ocean_heat_extraction=jnp.zeros(ncol),
        ocean_stress_x=jnp.zeros(ncol),
        ocean_stress_y=jnp.zeros(ncol),
        # Phase-aware moisture mass flux (up): pack sublimation (``sublim_actual``,
        # L_s) + soil / plant-water evaporation (``soil_evap``, L_v) + the blowing-
        # snow sublimated SWE that left as vapor (gap 5) — so the vapor mass
        # balances the reported latent heat across BOTH phases.
        surface_mass_flux=sublim_actual + soil_evap + blow_subl - _cond_liq,
        # Land tile does not exchange salt with the ocean directly.
        salt_flux=jnp.zeros(ncol),
    )

    # A REVERTED column is reverted atomically (soil, snow, carbon, cache).  An
    # ACCEPTED unsolved column advances its energy and water but keeps its
    # carbon pools and exchanges no CO2: the fallback GPP is the cold state's,
    # not physics, and pools + flux are held as a consistent pair.
    (new_state, response, carbon_state_new, _held_mask, _n_held,
     _fb_mask, _n_fb, _fb_rej_mask, _n_fb_rej) = _hold_unsolved_columns(
        state, new_state, response, surface_out, forcing, config, ncol,
        carbon_old=carbon_state, carbon_new=carbon_state_new,
        fallback_ok=_fallback_ok)
    surface_out = surface_out._replace(
        held=_held_mask, n_held=_n_held, fallback=_fb_mask,
        n_fallback=_n_fb, fallback_rejected=_fb_rej_mask,
        n_fallback_rejected=_n_fb_rej,
        soil_unconverged=~richards_out.converged)
    if layered:
        surface_out = surface_out._replace(
            snow_T_top_excess=snow_T_top_excess,
            snow_advected_heat=(snow_advected_heat - snow_drainage_heat
                                if snow_drain_sensible is None else
                                snow_advected_heat - snow_drainage_heat + snow_drain_sensible),
            snow_ground_heat_applied=snow_ground_heat_applied)

    return new_state, response, carbon_state_new, surface_out


def drainage_heat_to_soil(drain_sensible, runoff_surface, water_in):
    """Share [J/m2] of the pack drainage's sensible heat ``drain_sensible`` that
    stays with the surface water the soil (or its pond) accepts: the surface
    runoff ``runoff_surface`` [kg/m2/s] takes its proportional share of the
            # silent — the count below still reports the column as held.
            return new_field

    # Carbon first: an accepted unsolved column keeps its pools (D3).  If the
    # carrier changed structure this step there is nothing to revert to: the
    # fallback cannot honour D3, so the column's state and response are reverted
    # instead, but its carbon CANNOT be and stays new -- the same pre-existing
    # limitation as for any reverted column.  Unreachable today: CarbonState has
    # no optional fields and step_carbon returns the structure it was given.
    held_carbon = carbon_new
    if carbon_old is not None and carbon_new is not None:
        try:
            held_carbon = jax.tree.map(
                _revert_leaf(nonfinite | unsolved), carbon_new, carbon_old)
        except (ValueError, TypeError):
            held_carbon = carbon_new
            rejected = rejected | fallback
            fallback = jnp.zeros(ncol, dtype=bool)

    bad = nonfinite | (unsolved & ~fallback)
    n_held = jnp.sum(bad.astype(jnp.int32))
    n_fallback = jnp.sum(fallback.astype(jnp.int32))
    n_rejected = jnp.sum(rejected.astype(jnp.int32))

    # The canopy cache of an accepted unsolved column keeps its previous root
    # (its new entry is NaN: nothing converged this step).
    _cache_mask = bad | fallback
    held_state = new_state._replace(
        **{name: _hold_field(getattr(new_state, name), getattr(state, name),
                             _cache_mask if name == "canopy_x" else bad)
           for name in new_state._fields})

    # Inert-surface response for a reverted column.
    T_prev = land_skin_temperature(state, config)
    eps = jnp.full(ncol, config.emissivity_land)
    inert = dict(
        T_sfc=T_prev, T_rad=T_prev,
        albedo=jnp.full(ncol, config.albedo_land),
        emissivity=eps,
        z0=jnp.full(ncol, config.z0_land),
        # Equal to the air, so any consumer that forms a humidity GRADIENT from
        # this field gets zero.
        q_surface=jnp.asarray(forcing.q_lowest).reshape(-1),
        # Grey-body upward LW of the exported (T_rad, emissivity) pair: the
        # emission PLUS the reflected (1 - eps) * La, so the radiation's LW
        # boundary eps*sigma*T_rad^4 + (1 - eps)*La reproduces this field
        # (it omitted the reflected term: 14-20 W/m2 off, tier-5 U11-2).
        lw_up=(eps * constants.sigma_sb * T_prev ** 4
               + (1.0 - eps) * jnp.asarray(forcing.lw_down).reshape(-1)),
    )

    def _hold_response(name, leaf):
        if leaf is None or not _is_float_leaf(leaf):
            return leaf
        arr = jnp.asarray(leaf)
        if arr.ndim == 0 or arr.shape[0] != ncol:
            return leaf
        fallback_val = inert.get(name, jnp.zeros_like(arr))
        arr = jnp.where(bad, jnp.broadcast_to(fallback_val, arr.shape), arr)
        if name == "co2_flux":
            # Carbon is held on an accepted unsolved column: no exchange either.
            arr = jnp.where(fallback, jnp.zeros_like(arr), arr)
        return arr

    held_response = response._replace(
        **{name: _hold_response(name, getattr(response, name))
           for name in response._fields})

    return (held_state, held_response, held_carbon, bad, n_held,
            fallback, n_fallback, rejected, n_rejected)

                T2m_land=None):
        """2 m air temperature for CMIP ``tas`` from the MOST surface-layer
        similarity profile (interpolate the lowest model level down to 2 m).
        Returns the lowest-level T when the surface inputs (SST / sigma) are
        unavailable — e.g. a prescribed-SST run that does not pass SST here.

        ``T_land`` / ``land_fraction`` (keyword-only) make the diagnostic
        LAND-AWARE.  Without them the surface is the blended SST/sea-ice skin
        with a saturated humidity EVERYWHERE, so the published ``tas`` over
        land is the neighbouring ocean's temperature rather than the model's
        land — the reason a February Arctic land bias could not be read from
        this field.  There is no below-canopy 2 m profile in this code, so the
        land branch uses the same MOST profile anchored on the land skin; do
        not read the result as a sub-canopy screen temperature.

        ``T2m_land`` (keyword-only, optional) is the land scheme's OWN 2 m air
        temperature (two-leaf canopy: CLM5 ``t_ref2m`` from its roughness,
        displacement, Obukhov length and canopy-air node).  Where it is finite
        it REPLACES the profile above on the land leg, which runs the ocean
        bulk law (Charnock roughness, no displacement) over the land skin.
        Non-finite entries (held columns, schemes that solve none) keep that
        profile.

        ``u_low`` / ``v_low`` override the lowest-level winds (lets the MPAS
        path pass the reconstructed CELL winds from
        ``reconstruct_cell_velocity`` instead of the edge-normal ``state.u``,
        which is not cell-collocated); default reads them off ``state``."""
        T_low = state.T.data[..., -1]
        if sst is None or self.sigma_full is None:
            return T_low
        import jax.numpy as jnp
        from legoesm import constants
        from legoesm.thermo import saturation_mixing_ratio
        from legoesm.core.bulk_flux import compute_most_fluxes
        u_low = state.u.data[..., -1] if u_low is None else u_low
        v_low = state.v.data[..., -1] if v_low is None else v_low
        q_low = q_v[..., -1] if q_v is not None else jnp.zeros_like(T_low)
        p_s = state.p_s.data
        p_low = jnp.asarray(self._p_full(p_s))[..., -1]
        rho_low = p_low / (constants.R_d * T_low)
        # Similarity profile matched to the experiment's surface fluxes:
        # SAME bulk scheme (constant -> the historical coare3 stand-in, see
        # __init__) and SAME stable-branch selector (codex 2026-08-02: a
        # large_yeager or grachev/gryanik run otherwise published a
        # coare3-native-default tas).  The 2 m value is set by stability, so
        # gustiness is left scheme-native here (pre-existing choice).
        def _profile_to_2m(T_sfc, q_sfc, *, ocean: bool):
            """One surface's 2 m temperature.

            MERGE NOTE (#1773), and it is a physics decision, not a textual
            one. ``main`` added a ``large_yeager_cesm`` arm that returns CESM
            ``shr_flux_atmOcn``'s own ``tref`` instead of the MOST profile;
            this branch split the diagnostic into a LAND surface and a SEA
            surface. ``shr_flux_atmOcn`` is the atmosphere-OCEAN coupler law
            (the module's own docstring: "SAM ocean surface fluxes"), with
            Charnock roughness and a saturated surface, so handing it a land
            skin would re-create the exact artifact this branch exists to
            remove -- land ``tas`` computed as if the surface were ocean.

            So the CESM arm applies to the SEA leg ONLY; the land leg keeps
            the MOST profile on every scheme. Review (GLM) flagged the
            alternative -- dispatching for both legs -- as a third scheme
            neither side tested.
            """
            if ocean and self.tas_profile_scheme == "large_yeager_cesm":
                # CESM shr_flux_atmOcn's own ``tref`` diagnostic (2 m).
                from legoesm.core.bulk_flux import compute_sam_oceflx_fluxes
                *_, T_2m = compute_sam_oceflx_fluxes(
                    u_low, v_low, T_low, q_low, T_sfc, q_sfc, rho_low,
                    z_bot=10.0, variant="cesm", return_2m=True,
                    t_bot=T_low,
                )
                return T_2m
            *_, T_2m = compute_most_fluxes(
                u_low, v_low, T_low, q_low, T_sfc, q_sfc, rho_low,
                scheme=(self.tas_profile_scheme
                        if self.tas_profile_scheme != "large_yeager_cesm"
                        else "large_yeager"),
                return_2m=True,
                stability_scheme=self.surface_stability_scheme,
            )
            return T_2m

        T_sfc_sea = blend_surface_temperature(sst, sic, T_ice)
        q_sfc_sea = saturation_mixing_ratio(T_sfc_sea, p_s)
        if T_land is None or land_fraction is None:
            # Legacy path, bit-identical: ONE profile anchored on the
            # ocean/sea-ice skin with a saturated surface.  Over land that is
            # the neighbouring ocean's temperature and a saturated surface the
            # land does not have, so a caller that owns a land skin should pass
            # it — see the land branch below.
            return _profile_to_2m(T_sfc_sea, q_sfc_sea, ocean=True)

        # Land and sea get their OWN surface, then combine by land fraction.
        # Both branches keep the same bulk scheme and stability selector, so
        # this introduces no second convention — only the missing surface.
        # ``q_land`` is a surface mixing ratio; without one the land surface is
        # treated as saturated, which is what the legacy path assumed
        # everywhere and is an OVERESTIMATE of surface humidity over dry or
        # frozen ground.
        q_sfc_land = (saturation_mixing_ratio(T_land, p_s) if q_land is None
                      else q_land)
        T_2m_land = _profile_to_2m(T_land, q_sfc_land, ocean=False)
        if T2m_land is not None:
            T2m_land = jnp.asarray(T2m_land, dtype=T_2m_land.dtype)
            T_2m_land = jnp.where(jnp.isfinite(T2m_land), T2m_land, T_2m_land)
        T_2m_sea = _profile_to_2m(T_sfc_sea, q_sfc_sea, ocean=True)
        f_land = jnp.clip(jnp.asarray(land_fraction, dtype=T_2m_land.dtype),
                          0.0, 1.0)
        return f_land * T_2m_land + (1.0 - f_land) * T_2m_sea

    def _cloud_config_for(self, cloud_fraction):
        """The collector's cloud config, with the cloud-fraction override over
                        _phase_t["land"] += time.perf_counter() - _pt0
                        _phase_t["land_iter_max"] = max(
                            _phase_t["land_iter_max"], int(_land_n_iter_max))
                    # Mirror the land tile's skin and surface humidity onto
                    # the driver, the same way the ice skin above is mirrored:
                    # the CMOR ``tas`` diagnostic runs in a different method and
                    # cannot see these closure locals, and without them it
                    # anchors its profile on the neighbouring OCEAN skin over
                    # land and publishes a land temperature the model never had.
                    self._land_T_skin_last = _land_T_skin
                    self._land_qsfc_last = _land_qsfc_step
                    self._land_T2m_last = _land_T2m_step
                    # Accumulate ON DEVICE and read at the same cadence the
                    # other post-step warnings use: reading it every step
                    # would stall the accelerator once per step for a number
                    # that is almost always zero.
                    _land_n_held_accum = _land_n_held_accum + _land_n_held_step
                    _land_n_unconv_accum = _land_n_unconv_accum + _land_n_unconv_step
                    _land_n_held_land_accum = (
                        _land_n_held_land_accum + _land_n_held_land_step)
                    _land_n_held_steps_accum = (
                        _land_n_held_steps_accum
                        + (_land_n_held_step > 0).astype(jnp.int32))
                    _land_n_fb_accum = _land_n_fb_accum + _land_n_fb_step
                    _land_n_fb_rej_accum = (
                        _land_n_fb_rej_accum + _land_n_fb_rej_step)
                    # Published only under the same switch that threads f_land
                    # into the turbulence factory: without the land fraction
                    # the consumer refuses the key, and adding it mid-run
                    # would change the forcing pytree and retrace.
                    if _land_beta_soil_on:
                        _land_qsfc_cells = _land_qsfc_step
                        _land_shflx_cells = _land_shflx_step
                        _land_lhflx_cells = _land_lhflx_step
                        _land_evap_cells = _land_evap_step
                        if _land_stress_on:
                            # Last valid land stress per column: a held or
                            # non-finite solve keeps the previous value.
                            (_land_taumag_cells, _land_taumag_valid,
                             _land_taumag_fresh) = hold_last_valid_land_stress(
                                _land_taumag_cells, _land_taumag_valid,
                                _land_taumag_step, _land_held_step)
                            self._land_taumag = _land_taumag_cells
                            self._land_taumag_valid = _land_taumag_valid
                            self._land_taumag_fresh = _land_taumag_fresh
                    if _land_beta_fn is not None and _land_qsfc_cells is None:
                        # Root-zone beta only until the humidity channel is
                        # live (or when the scheme solves none).
                        _land_beta_cells = _land_beta_fn(self._land_ml_state)
                    if _is_mpas_cell_partitioned(self):
                        # A halo cell steps its OWN land column, which is
                        # never exchanged and drifts from the owner's; the
                        # turbulence's cell->edge tendency projection reads
                        # the halo cell's surface stress, so without this an
                        # owned boundary edge is forced by a land column
                        # nobody owns (measured: 1e-8 m/s restart drift on
                        # exactly the boundary edges, zero with the land
                        # stress off).  ``_land_call`` is rank-uniform
                        # (step cadence + a loop-local None test), so every
                        # rank issues this collective together.
                        (_land_T_skin, _land_albedo_cells, _land_qsfc_step,
                         _land_qsfc_cells, _land_shflx_cells,
                         _land_lhflx_cells, _land_evap_cells,
                         _land_beta_cells, _land_taumag_cells,
                         _land_taumag_valid, _land_taumag_fresh,
                         _land_Trad_cells, _land_emis_cells
                         ) = _exchange_halo_cell_fields(
                            self._voronoi_layout.halo_exchange,
                            (_land_T_skin, _land_albedo_cells,
                             _land_qsfc_step, _land_qsfc_cells,
                             _land_shflx_cells, _land_lhflx_cells,
                             _land_evap_cells, _land_beta_cells,
                             _land_taumag_cells, _land_taumag_valid,
                             _land_taumag_fresh,
                             _land_Trad_cells, _land_emis_cells))
                        self._land_T_skin_last = _land_T_skin
                        self._land_qsfc_last = _land_qsfc_step
                        if _land_taumag_cells is not None:
                            self._land_taumag = _land_taumag_cells

exec
/usr/bin/bash -lc "ls /glade/u/home/pg2328; rg -n 'python|pytest|venv' docs/wb/reviews/2026-11-22_land_tas_r1_prompt.md docs/wb/reviews/2026-11-22_land_tas_r1_codex.md pyproject.toml | head -40; git diff 8750d0005d --stat; sed -n '710,950p' CLAUDE.md" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 36ms:
amip_cpuprof.o7605802
amip_cpuprof.o7605848
amip_cpuprof.o7605849
amip_cpuprof.o7605880
amip_cpuprof.o7605881
amip_cpuprof.o7605933
amip_cpuprof.o7620975
as1.pbs.o8122391
base.pbs.o8121749
c1.pbs.o8122617
chacc.o7778591
d4.pbs.o8122920
d5.pbs.o8122929
e1.pbs.o8122154
e2.pbs.o8122171
e3.pbs.o8122189
e3.pbs.o8122190
e4.pbs.o8122207
e5.pbs.o8122215
e6.pbs.o8122219
e7.pbs.o8122258
ed_latlon_new_0.5.o8122159
ed_latlon_new_1.0.o8122158
ed_latlon_old_0.5.o8122157
ed_latlon_old_1.0.o8122156
ed_mpas_new_0.5.o8122164
ed_mpas_new_1.0.o8122162
ed_mpas_old_0.5.o8122161
ed_mpas_old_1.0.o8122160
fin_fix.o7745096
fin_main.o7745097
floor.pbs.o8133622
floor.pbs.o8134380
floor.pbs.o8134385
floor.pbs.o8134405
g3_nonvac2.o7865076
g3_nonvac3.o7868669
g3_nonvac4.o7874301
g3_nonvac5.o7874569
g3_nonvac6.o7875064
g3_nonvac7.o7875122
g3_nonvac8.o7875173
g3_nonvac9.o7879046
g3_nonvac.o7864921
g3_nonvac.o7864947
g3_smoke.o7864825
g3_sponge.o7875062
g3_timing.o7879806
gate.pbs.o8133614
gate.pbs.o8134379
gate.pbs.o8134384
gate.pbs.o8134387
gate.pbs.o8134404
gate.pbs.o8134436
gmpi.pbs.o8134435
h1.pbs.o8122556
h2.pbs.o8122602
h3.pbs.o8122608
intel
j_latlon_as83_0.5.o8122579
j_latlon_as83_1.0.o8122578
j_latlon_as83_logmean_0.5.o8122581
j_latlon_as83_logmean_1.0.o8122580
j_latlon_old_0.5.o8122575
j_latlon_old_1.0.o8122574
j_latlon_sb81_theta_0.5.o8122577
j_latlon_sb81_theta_1.0.o8122576
j_mpas_as83_0.5.o8122587
j_mpas_as83_1.0.o8122586
j_mpas_as83_logmean_0.5.o8122589
j_mpas_as83_logmean_1.0.o8122588
j_mpas_old_0.5.o8122583
j_mpas_old_1.0.o8122582
j_mpas_sb81_theta_0.5.o8122585
j_mpas_sb81_theta_1.0.o8122584
jw.pbs.o8121695
land_tail2.o7630178
land_tail.o7625600
legoesm
legoesm_generic_sw
legoesm_main_base
legoesm_ocean_base
legoesm_ocean_pe
legoesm_pe_pilot
legoesm_ppm_fix
legoesm_rbffd
legoesm_sw_wired
lf_probe.o7779893
lm_ab.o7653149
lm_ab.o7653150
lt6_amip_cpuprof.o7654258
most_ab.o7654631
richards_rerun.o7727579
richards_rerun.o7727583
richards_tests.o7724465
richards_tests.o7725479
richards_tests.o7725760
rich_fix.o7729441
rich_main.o7729442
rk_2node.o7681574
rk_2node.o7682894
rk_ab.o7657552
rk_ab.o7657553
rk_ab.o7658445
rk_ab.o7658446
rk_c87.o7871466
rk_c87rec.o7874118
rk_clr.o7690005
rk_cs.o7688954
rk_cs.o7688955
rk_day.o7659957
rk_day.o7659958
rk_day.o7659959
rk_day.o7659961
rk_day.o7666102
rk_day.o7666103
rk_dump.o7707036
rk_dyn.o7707006
rk_fixe2.o7862770
rk_jit.o7686679
rk_landt2.o7683206
rk_landt.o7683194
rk_landt.o7683205
rk_mem2.o7675994
rk_mem.o7675745
rk_mem.o7675748
rk_minor.o7706645
rk_mr2.o7863914
rk_mrb2.o7863925
rk_nudge.o7666889
rk_vprof.o7683907
rk_vprof.o7683914
rr_fix.o7730539
rr_iso.o7731097
rr_main.o7730541
run1.pbs.o8121665
run2.pbs.o8121677
run3.pbs.o8121685
run4.pbs.o8121706
run5.pbs.o8121734
run6.pbs.o8121762
run7.pbs.o8121923
run8.pbs.o8121997
s2_nvck2.o7858019
s2_nvck2.o7858039
s2_nvck3.o7858075
s2_nvck.o7857991
s2_nvpos.o7858633
s2_probe.o7858226
s3b_g4time.o7857968
s3_tests.o7827797
s3_tests.o7828702
tA.pbs.o8133618
tB.pbs.o8133619
tC.pbs.o8133620
tF.pbs.o8134742
tG.pbs.o8134867
tG.pbs.o8134888
tH.pbs.o8137519
tI.pbs.o8137637
tI.pbs.o8137712
tN.pbs.o8134643
tR.pbs.o8134432
tT.pbs.o8134816
vkd_tests2.o8162566
vkd_tests.o8162509
w1.pbs.o8122520
wt_main
pyproject.toml:12:requires-python = ">=3.11"
pyproject.toml:108:    "pytest>=8.0",
pyproject.toml:109:    "pytest-xdist",
pyproject.toml:110:    "pytest-timeout",
pyproject.toml:320:# subsequent module-level imports trip E402; the bootstrap makes ``python scripts/...`` work.
pyproject.toml:361:python_version = "3.11"
pyproject.toml:369:[tool.pytest.ini_options]
pyproject.toml:374:    "lint: static checks run through pytest because CI is disabled; skip in a fast inner loop with -m 'not lint'",
pyproject.toml:395:# Run: `.venv/bin/lint-imports`
docs/wb/reviews/2026-11-22_land_tas_r1_prompt.md:230:+import pytest
docs/wb/reviews/2026-11-22_land_tas_r1_prompt.md:235:+pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64,
docs/wb/reviews/2026-11-22_land_tas_r1_prompt.md:256:+@pytest.mark.parametrize("zeta", [-0.3, -0.05, 0.05, 0.3])
docs/wb/reviews/2026-11-22_land_tas_r1_prompt.md:257:+@pytest.mark.parametrize("z0h", [0.05, 1.5])
docs/wb/reviews/2026-11-22_land_tas_r1_prompt.md:317:+@pytest.mark.parametrize("sw,lw,T_soil", [(0., 200., 262.), (700., 300., 285.)])
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:243:+import pytest
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:248:+pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64,
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:269:+@pytest.mark.parametrize("zeta", [-0.3, -0.05, 0.05, 0.3])
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:270:+@pytest.mark.parametrize("z0h", [0.05, 1.5])
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:330:+@pytest.mark.parametrize("sw,lw,T_soil", [(0., 200., 262.), (700., 300., 285.)])
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:1195:  own success line (pytest's `N passed`, "COMPLETED in Xs") = UNVERIFIED; OOM
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:1317:  cover `pytest` fixtures too.
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:1551:- **CRITICAL — Visual verify spatial/grid artifacts**: passing tests+norms NECESSARY ≠ SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coeffs, grid metrics. Edge artifacts/cube imprint/grid-scale noise only detected visually (v-wind W2, wind_speed W5). Run `--only sw --grid cubed_sphere --quick` + inspect PNGs vs baseline. Norms can improve while artifacts worsen. Never claim "tests pass, edge fixed" from pytest alone.
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:1573:- Tests: `.venv/bin/python -m pytest tests/`
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:1574:- Sci tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:1575:- Atm matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py`
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:1576:- Ocean matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py`
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:1577:- AMIP: `.venv/bin/python scripts/run/run_amip.py`
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:1578:- Dycore progression: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py` (SW -> hydrostatic -> non-hydrostatic ladder; `--only sw|hydro|nh`). Old `tests/validation/run_dycore_progression_suite.py` superseded — its per-case child scripts removed.
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:1579:- GPU/MPI scaling: `.venv/bin/python scripts/bench/run_levante_gpu_scaling.py --grid cubed-sphere --mode strong` (`docs/performance/REAL_HARDWARE_SCALING.md`)
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:1581:- MPI tests: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/`
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:1582:- MPI diff: `mpirun -np 2 .venv/bin/python -m pytest tests/distributed/test_mpi_differentiability.py`
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:1608:- **No top-level cross-package imports from `core/` to `runtime/`/`parallel/`/`driver/`/`training/`/`experiments/`.** Why: `from legoesm.runtime.backend import ...` at top of `core/precision.py` triggered `runtime/__init__.py` → `runtime.precision` → `core.precision` mid-init, breaking isolated pytest. Use function-scope deferred imports.
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:2453:  own success line (pytest's `N passed`, "COMPLETED in Xs") = UNVERIFIED; OOM
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:2575:  cover `pytest` fixtures too.
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:2809:- **CRITICAL — Visual verify spatial/grid artifacts**: passing tests+norms NECESSARY ≠ SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coeffs, grid metrics. Edge artifacts/cube imprint/grid-scale noise only detected visually (v-wind W2, wind_speed W5). Run `--only sw --grid cubed_sphere --quick` + inspect PNGs vs baseline. Norms can improve while artifacts worsen. Never claim "tests pass, edge fixed" from pytest alone.
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:2831:- Tests: `.venv/bin/python -m pytest tests/`
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:2832:- Sci tests: `JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>`
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:2833:- Atm matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_atmosphere_test_matrix.py`
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:2834:- Ocean matrix: `JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py`
docs/wb/reviews/2026-11-22_land_tas_r1_codex.md:2835:- AMIP: `.venv/bin/python scripts/run/run_amip.py`
 packages/coupler/legoesm/driver/diagnostics.py     |  14 +-
 packages/coupler/legoesm/driver/model_driver.py    |  29 +++-
 packages/land/legoesm/land/canopy/solver.py        |   7 +-
 packages/land/legoesm/land/canopy/stability.py     |  37 ++++-
 packages/land/legoesm/land/surface_scheme/base.py  |   4 +
 .../legoesm/land/surface_scheme/two_leaf_canopy.py |   4 +-
 tests/land/unit/test_land_screen_temperature.py    | 157 +++++++++++++++++++++
 tests/unit/test_land_column_split.py               |   4 +-
 8 files changed, 241 insertions(+), 15 deletions(-)
  defects in EACH of three rounds here; rounds 2 and 3 not ceremony. No
  presenting first-round implementation as finished, no treating "tests pass" as
  terminal condition — round-1 diff passed 401 tests while still containing
  silent fallback and false docstring.

## Assumption Gates — from six wrong assumptions in ONE session (2026-08-05/06)
User callout: *"you keep making a lot of assumptions that prove to be wrong."*
Gates above stop wrong CLAIMS about the model; these stop cheaper, more frequent
error — being wrong about CODE AND DATA IN FRONT OF YOU. Every rule below
mechanical, each cost full round-trip.

- **AN ARRAY'S LAYOUT IS AN API — READ IT, INCLUDING AXIS ORDER.** "Never infer
  an API" rule cover signatures; it also cover array SHAPE, AXIS ORDER, index
  base, padding convention. Before first index of any mesh/state field, print
  its `.shape`. FAILURE: indexed `mesh.cellsOnCell[c, k]` assuming
  `(nCells, 6)`; it is `(6, nCells)`, so probe raised `IndexError` on every
  method. `cellsOnEdge` is `(2,
  nEdges)` — one unambiguous pair per edge, usually the better handle.
- **A PROXY IS NOT THE QUANTITY. Real number produced by specific code path →
  GET IT BY CALLING THAT PATH.** Re-deriving lookalike from first principles
  silently answer different question. FAILURE: computed `max_degree` from 1-ring
  `cellsOnEdge` adjacency and reported it as "ppermute round count" — real
  exchange is halo-depth-aware, so proxy said 8 rounds for every method/rank
  count while real schedule said 12→14. Proxy could not even reproduce known
  answer — that the tell: **run proxy against a case whose real value you
  already know BEFORE using it on the unknown one.**
- **AN OPTIMIZER ONLY HELPS IF ITS OBJECTIVE IS THE BINDING TERM — state which
  quantity it minimizes, confirm that quantity is measured bottleneck.**
  FAILURE: assumed METIS would cut MPAS ppermute rounds; METIS minimize EDGE
  CUT, bottleneck is MAX_DEGREE, and measured they move OPPOSITELY (metis 19
  rounds @64 vs sfc 14, while metis has lower cut). "Better partitioner" is not
  a mechanism.
- **A TABLE YOU PARSED IS NOT DATA UNTIL YOU SPOT-CHECK IT.** Any derived
  summary quoted to human, or fed to review agent, need ≥2 rows verified by eye
  against raw source first. FAILURE: regex over multi-line series lists attached
  subdiv-9's `9.60/11.48 ms` to `s8` label, and that mislabelled pair went into
  codex prompt as fact.
- **PRE-IMPL GREP COVERS TESTS, NOT JUST FUNCTIONS.** Before writing test, grep
  for one already asserting same invariant. FAILURE: added two tests counting
  PCG reduction sites duplicating existing `test_reduction_count_halved`; codex
  had to point it out.
- **WHEN A TEST FAILS, DECIDE WHETHER THE EXPECTATION OR THE CODE IS WRONG, AND
  SAY WHICH.** Assertion you just wrote is claim with no more standing than the
  code. FAILURE: asserted `LEGOESM_..._FUSED_HALO=""` meant OFF; resolver is
  `!= "0"`, so `""` means ON — test was wrong, not the default.
- **BEFORE PROPOSING COMPUTE, CHECK THE PATH IS BUILT — grep driver for
  decomposition/sharding the run would need.** FAILURE: scoped 128-GPU coupled
  ladder before finding `CoupledESM` contain three occurrences of "shard", all
  in comments — coupled ocean is replicated, so ladder would have measured
  unbuilt path.
- **WHEN TWO MECHANISMS COULD EXPLAIN A NUMBER, NAME THE MEASUREMENT THAT
  DISCRIMINATES THEM AND RUN IT BEFORE REPORTING EITHER.** FAILURE: reported
  serial-vs-SPMD gap as "cadence PHASE disagreement"; predicates agree with ZERO
  offset and real cause was serial short-tail fallback. One `grep` of two
  predicates would have settled it.

## Compute Discipline — speculation costs GPU-hours, not just credibility
User, 2026-07-30 (THIRD callout in five days): *"You keep making very
speculative assumptions... be much more precise so we do not waste time with
useless simulations."* Gates above stop wrong CLAIMS; these stop wrong RUNS.
Simulation launched on hypothesis no measurement can refute = pure waste, and it
also cost WALL-CLOCK of queue slot it occupied.

- **NO COMPUTE ON AN UNFALSIFIABLE HYPOTHESIS. Before submitting ANY job costing
  >1 GPU-hour, write down three things: (a) exact number the run will produce,
  (b) value that CONFIRMS and value that REFUTES, (c) why a cheaper offline/CPU
  test on an EXISTING checkpoint cannot answer it. Cannot fill all three -> DO
  NOT SUBMIT; run cheap test first.** Nearly every question asked so far
  (fluxes, tendencies, radii, momentum budgets, cloud optics) was answerable
  offline from saved checkpoint in minutes. FAILURE 2026-07-30: submitted two
  5-YEAR full-physics runs (8 h walltime each) while model had KNOWN unfixed +56
  W/m2 albedo error and no low-level circulation — five simulated years of
  broken climate, answering no question that had been asked.
- **RANK ERRORS BY MAGNITUDE BEFORE CHOOSING WHAT TO WORK ON.** Run full
  scorecard FIRST, work LARGEST term; re-rank after every fix. FAILURE
  2026-07-30: spent most of session on hfls deficit (-40 W/m2) while dominant
  error was rsut (+56 W/m2) — and scorecard naming it was already sitting in run
  directory, unread.
- **ONE VARIABLE PER PRODUCTION RUN.** N simultaneous config changes answer ZERO
  questions, because no output attributable to any one of them. Multi-change
  config legitimate ONLY as deliberate new BASELINE labelled as such and never
  compared term-by-term against the old one. FAILURE 2026-07-30: one launch
  flipped ~8 switches at once.
- **A LONG RUN ON A MODEL WITH AN UNFIXED DOMINANT ERROR IS WASTE.** Before
  extending past ~30 simulated days, state largest outstanding scorecard term
  and why run still worth its GPU-hours. Fix big term, then extend. Short
  validation windows (days) are for "does it run and is new physics behaving";
  multi-year windows are for model that already pass.
- **NEVER LEAD WITH AN ARITHMETIC COINCIDENCE.** Hand-computed ratio that
  "matches" observed ratio is not evidence when calculation omit factors code
  actually applies. State it as arithmetic, or don't state it. FAILURE
  2026-07-30: "cover x tau ~ 1.9x matches the 1.9x albedo" ignored sub-grid
  inhomogeneity factor radiation applies to cloud paths.
- **PREFER THE INSTRUMENT THAT ALREADY EXISTS.** Before writing probe, check run
  directory for scorecard/manifest/diagnostic answering the question, and
  `scripts/validate/` for validator. Reading existing artifact cost seconds; new
  probe cost an hour and need its own controls.

## Diagnosis Discipline — one session, ~10 GPU arms, 5 of them wasted
2026-08-05/06, FESOM2-match OMIP blowup. Every rule mechanical, from specific
failure in that one session. Root cause turned out to be one-line IC defect
findable offline in seconds; arms bought nothing.

- **SANITY-CHECK EVERY PROBE NUMBER AGAINST A PHYSICAL RANGE BEFORE BUILDING ON
  IT — one line of arithmetic, before the conclusion.** Convert to quantity
  whose plausible span you know, compare. FAILURE: PGF probe returned 0.02
  m/s^2 and I built whole attribution on it; that value imply ~14 kg/m^3 density
  difference between ADJACENT 1-degree cells when entire ocean span ~6 kg/m^3.
  Check take seconds, number sat unused for hours, and running it immediately
  would have exposed real defect (cells initialised to T=0/S=0) before a single
  GPU arm.

- **A FAILURE STEP OR LOCATION IS NOT A SIGNATURE UNTIL THE DIAGNOSTIC IS SHOWN
  TO RESOLVE IT.** State instrument's resolution, confirm reported step is not
  merely its first sample. FAILURE: five one-variable arms all reported "BLOWUP
  at step 36" and I read that config-invariance as physics. `run_omip` set
  `block_size = max(1, diag_every)` and only test at block boundaries, so 36 was
  FIRST LOOK. At `--diag-every 1` answer was step 1. Corollary: arms run at
  coarse diagnostic stride prove only "this setting alone does not fix it",
  NEVER "this setting has no effect".

- **REFUTE WITH STATES, NOT EXIT CODES OR FAILURE STEPS.** Two runs failing at
  same step can fail for different reasons. FAILURE: declared open North Pole
  "refuted" because capping did not move blowup step; states showed capping HAD
  removed that mode (|u|max 19.27 -> 0.0137 m/s) and merely uncovered second,
  unrelated one at same step number.

- **A PROBE IS PRODUCTION CODE FOR THE "NEVER INFER AN API" RULE, AND ITS FIRST
  OUTPUT IS UNTRUSTED.** Read signature of every function a probe call,
  including this repo's own. FAILURE: called
  `build_runoff_map(lat, lon, mask, area)` when signature is
  `(ocean_mask, cell_area, lat_rad, lon_rad)`, so `cell_area` received
  `deg2rad(lon)` — exactly 0 at lon 0 — and I reported NaN "conservation defect"
  in shipped code that did not exist.

- **A PLAUSIBLE-LOOKING VALUE IS MORE DANGEROUS THAN A NaN, IN CODE AND IN
  TESTS.** Sentinel that is valid number of right dtype pass every finite/NaN
  guard downstream. FAILURE: `_interp_profile_to_z_coord` returned `0.0` for
  no-data column; `0.0` is not NaN, so caller's `where(isnan, fill, x)` never
  fired and 39032 wet cells entered model as fresh water at 0 degC. TWO
  COMMITTED TESTS ASSERTED THAT BEHAVIOUR, one calling it "documented degenerate
  fallback" and one commenting "still no NaN leaks" — exact inversion of truth.
  Degenerate branch must return something → return NaN (or raise) so downstream
  guard can see it, and treat any test asserting magic sentinel as suspect.

- **BUDGET CLOSURE BEFORE HYPOTHESES.** For any conservation-relevant blowup or
  drift, run closure/redistribution check FIRST — it name the operator class
  instead of ranking guesses. Canonical probe:
  `scripts/validate/ocean_fidelity/omip_conservation_closure.py`. On first
  deployment it refuted hypothesis I was about to spend a GPU arm on (dipole
  whose two-cell sum GREW is not a diffusion instability) and localised
  config-invariant source to single column. See
  `docs/ocean/fidelity/fidelity_to_fesom2jax_level_plan.md` (#1492) Phase 0.1.

- **A THROWAWAY PROBE'S NUMBER IS UNMEASURED. COMMIT THE PROBE.** Per #1492
  Phase 0.3: one committed probe per row, locked conventions, provenance (git
  SHA + inputs + flags) stamped on every run. Inline heredoc probes not citable,
  cannot be re-run against changed model, and hide own bugs — both defects above
  came from uncommitted ones, and committing closure probe immediately surfaced
  two more inside it.

- **DAMAGE-FIELD CORRELATIONS ARE NOT MECHANISMS.** In blown-up field, "worst
  cells are coastal / shallow / polar" describe where damage LANDED, usually
  downstream of single upstream defect. FAILURE: measured P(bad|coastal)=33% vs
  0.67% interior, real 50x enrichment, and treated it as coastal process; those
  were simply columns the IC defect had corrupted. Before citing spatial
  pattern, confirm snapshot is ORIGIN state (`snapshot_final.npz` stamp
  `_step = n_steps` regardless of when run stopped — verify with dedicated
  1-step run).

## JAX
- Pure pytree fns. `lax.scan` time integration. `vmap`/batched arrays over Python loops on array dims. `jnp.where`/`lax.cond`/`fori_loop`/`scan` not Python control flow on traced.
- **Feature gating exception** (`fix_mass`, `fix_moisture`): Python `if` on static bool in closure — NOT `jnp.where` (traces both branches). `jnp.where` only for data-dependent traced selection.
- Stable shapes. No retrace. Dtype: spectral=x64+complex128; finite-volume can float32.
- No host/device thrash, NumPy in traced code, hidden non-JAX side effects.
- **Buffer donation + `jax.grad`**: `donate_argnums` conflicts reverse-mode AD. JIT fn inside `jax.grad`/`eqx.filter_value_and_grad` → provide non-donating variant (`.raw`). See `build_segment_fn`.
- **Closures vs explicit args**: closure captures = compile-time consts. Per-iter changing val (SST, solar) → pass as traced arg. See `SegmentForcing`.

## Earth System
- Conservation, metric consistency, staggered-grid consistency, halo correctness = first-class.
- No silent clip/damp/coerce unless justified+validated.
- Preserve units, sign conventions, monotonicity/positivity, hydrostatic/nonhydrostatic.
- Cubed-sphere/curvilinear: assume edge+metric errors first.
- Physics coupling: column closure + consistent flux signs.
- DA/diff: preserve smoothness. No gratuitous nondiff.
- **Sign-convention check MANDATORY on every equation/flux/tendency edit.** Before declaring done on any code touching PDE term, flux, source/sink, BC, or budget update: (1) state coordinate convention in scope (z up/down, flux positive-up/down/into-body) as comment at the term; (2) walk EACH term, confirm its sign match that convention — gravity vs capillary/diffusion divergence, top vs bottom BC, source vs sink, the `±` in `state = state ± dt·tend`; (3) confirm budget closes (`in − out − Δstorage = 0`) and exchanged fluxes carry SAME sign at both ends of a coupling (land runoff `+into ocean` must arrive `+into ocean`). Comment label and code must agree — flux commented "upward" computed as downward = defect, fix label or math. Mechanical gate where feasible: sign/conservation unit test (analytic column, manufactured solution, or `assert` budget residual ≈ 0) — passing norms alone never certify sign right (flipped flux can still be small). Common flips: z-axis direction, evap positive-up vs moistening, brine/salt vs freshwater dilution, stress atmospheric vs ocean convention (`-tau`), free-drainage vs gravity double-count.

## Parallel/HPC
- Correctness across serial/multi-device/MPI/hybrid.
- Sharded: reason about halo exchange, reductions, partition specs, global invariants.
- Validate single-rank → smallest distributed.
- Backends differ (Metal/GPU/CPU/spectral/MPI). Apple Silicon: spectral on CPU.
- **MPI**: `initialize_distributed(global_n=N)` → `scatter_to_local()` → rank-local step → `gather_to_global()` for I/O only. Never full global per rank.
- **4D halo**: `pad_halo_4d()`+`pad_halo_vector_4d()` all vert in one msg. All 3D ops in `operators_3d.py` use 4D. Never `vmap(pad_halo)`.
- **MPI halo AD**: all `sendrecv` via `_sendrecv_vjp` (`@jax.custom_vjp` in `halo_exchange.py`). Only `allreduce(SUM)` AD-safe (via `global_sum_mpi`; `final_loss=True` only for a sum seeded as the loss); `MAX`/`MIN`/`allgather`/`bcast` = diagnostics only.
- **Device mesh under MPI**: per-rank count to `create_device_mesh()`, not total.

## Oracle-Recipe Fidelity (ocean) — see docs/ocean/fidelity/oracle_recipe_strategy.md
- ADDITIVE to Validation Rules: oracle work NEVER replaces unit tests, ocean matrix, conservation checks, or visual verification. Truth tiers (conservation/equivariance/analytic) outrank oracle-matching.
- Recipe = pure config selecting shared canonical blocks (never bespoke `veros_*` solver). Oracle-matching numerics go in canonical module (`eos.py`, advection/limiter dispatch, `vertical_mixing/`, integrator dispatch) as selectable options.
- Mimicry-only glue (halo strip, axis transpose, time-level handling) lives in fidelity harness, never model. Test: "would a user with different goal ever select this?" No → harness.
- Conventions handled only in bridge, verified by equivariance tests (`physics(φ(x))=φ(physics(x))` to tol); a "convention" changing wet domain/answers is physics → config, not bridge.
- Constants are config (`ConstantsConfig`), not module-global monkey-patches (no `override_constants` in shippable paths); defaults reference `legoesm.constants`; base only, derived (κ,ε) recomputed.
- Oracle tendency-match (tier 3) trusted only for block that also clear truth tiers (0–2). MACHINE-ENFORCED (#388 Ask#4): `ocean/fidelity/precedence.py::evaluate_precedence` LOCKS oracle tiers (≥3) on any truth-tier (0–2) failure; surfaced + exit-gated by `scripts/validate/ocean_fidelity/build_fidelity_scorecard.py` (the one generated scorecard).

## Recipe×Setup template adapters (#388)
- **One shared selector, never re-implemented.** Every component YAML experiment adapter exposing a `setup:` block (atmosphere `Config`, ocean `OceanExperimentConfig`, sea-ice `SeaIceExperimentConfig`, …) MUST route validation + command-building + signature through `legoesm.core.setup_selector` (`MatrixRunnerSpec` + `validate_setup`/`build_matrix_command`/`setup_signature`/`require_positive_finite`). Per-component CLI differences (case flag `--only`/`--test`, exact-match `exact_prefix`, which `--levels`/`--dt`/`--days`/`--resolution` flags exist) are a `MatrixRunnerSpec`, NOT a copy-pasted `parts=[...]` builder or private `_SETUP_KEYS`/`_validate_setup`. Component-specific case-name validation goes through `known_names=` kwarg. Re-implemented selector = REJECTED in review. Factory naming: `_<component>_matrix_spec()`. FOLLOW-UP: dedup `OceanExperimentConfig`'s inline selector onto shared helper once PR #465 + shared-helper PR both land on main (inline copy predates extraction).
- **Every setup template's `(name, grid)` MUST be a real matrix case**, enforced by catalog-backed test (`run_<comp>_test_matrix._build_test_matrix()` → assert pair exists) so `--test/--only =<name> --grid <grid>` never select nothing. New template without this gate → REJECTED.
- **An exact case selector matching nothing is a hard error** (`raise SystemExit`) in every matrix-runner mode — never silent no-op (dispatch-hardening). MPI-safe: key guard off global pre-slice match count; empty-slice ranks fall through to barrier.
- **New `experiment_registry` mode → same-PR `get_adapter` test** in `test_experiment_registry.test_get_adapter_resolves_classes` (template test importing class directly does NOT cover dispatch).
- Land has no standalone idealized-case surface (no registry/matrix runner) → recipe×setup N/A there; do not invent one to match pattern. Truth-tier precedence stays ocean-scoped per `oracle_recipe_strategy.md`; use `TRUTH_TIERS`/`ORACLE_TIER_FLOOR` by name (no hardcoded `3`).

## Validation
- Narrowest test after edits. Numerical changes: analytical/benchmark > unit tests alone. `JAX_ENABLE_X64=1` unless float32/Metal task.
- Dycore: Williamson, Galewsky, Jablonowski-Williamson, DCMIP, Held-Suarez, ocean benchmarks.
- Conservation/reductions/coupler: mass+energy diagnostics.
- Parallel: unsharded vs sharded, single-rank vs MPI.
- Too expensive: say what ran/didn't, residual risk.
- **CRITICAL — Controlled comparison: change ONE variable, hold eval protocol FIXED to baseline.** To claim a change (resolution, params, scheme, days) improved/degraded/"is comparable" vs prior result, keep EVERYTHING else byte-identical to that baseline: forcing data + its sampling (years/days/hours/climatology), evaluation grid, metric definition (bias vs RMSE), region masks, timestepping. Metric that moved because protocol/sampling changed = **CONFOUND, not result** — NEVER compare number computed on one sampling/grid/metric to number from another and call difference an effect. Resource limit (network cap, compute, time) force lighter or different sampling → **re-run BASELINE at that SAME sampling before comparing** — fresh baseline cheap insurance; confounded claim not. Before writing "improved"/"degraded"/"better"/"comparable" vs any earlier number, explicitly confirm two configs differ ONLY in variable under test; cannot → say so, claim NO direction. Report full config (data+sampling, grid, days/steps, params, metric) next to every number so reader know exactly what compared. Assuming two runs comparable when setup drifted = error that turn "we improved it" into "we degraded it."
- **CRITICAL — Precision gate for comparison/skill/causal claims (do NOT be sloppy — 2026-07-20 EC-site lesson).** (a) **Harness self-check FIRST**: before reporting NEW scheme's skill vs validated baseline, run KNOWN baseline through YOUR OWN harness, confirm it reproduce baseline's established/published number (±small tol). Your harness score validated scheme wildly off — e.g. two-leaf H looked "broken" (NSE≈−1) when paper figure track obs — the HARNESS is wrong; fix BEFORE any new-scheme claim. (b) **Match reference metric EXACTLY**: metric aggregation (daily-mean vs half-hourly point-wise NSE), window (multi-year JJA vs one summer), masks (valid-forcing) ALL part of protocol. Report SAME metric reference used, plus any alternative, WITH sample N. Single-window point-wise metric is NOT skill verdict, and never rank schemes by metric punishing one scheme's known artifact (point-wise scatter/spikes) while ignoring dimension of interest (mean diurnal shape). (c) **Never let QC mask flatter one side**: drop scheme's unphysical spikes from scoring → report BOTH failures-penalized primary score AND separately-labeled plausible-only sensitivity score. (d) **Do NOT INFER causal origin — INSTRUMENT it**: "the spikes come from X" require LOGGING X and alternatives (raw vs intermediate vs final), not deduction from reading one code path; "correctly hooked up" require reading FULL path, not entry point. (e) **Label every claim CONFIRMED (evidence shown) vs PLAUSIBLE (inferred)** — adversarial review WILL refute over-confident inferences; state uncertainty up front rather than presenting verdict table a five-minute check overturn.
- **CRITICAL — A gradient/AD-tuning result that underperforms a known-good baseline is a BUG in your method until PROVEN otherwise — never attribute it to physics.** Before interpreting ANY optimizer/AD tuning outcome, or attributing residual error to physical cause (radiation, resolution, scheme's structure): (1) **Differentiated loss MUST be exact quantity you evaluate and report.** Truncated-BPTT / short-window / stop-gradient'd-spinup / cached-state proxy differing from full evaluation = OBJECTIVE MISMATCH, not result — optimizer minimize proxy while eval barely move. Differentiate real objective (e.g. full RCE equilibrium the eval scores), or prove proxy tight. (2) **Verify AD actually flows:** finite-difference check on ≥1 parameter (analytic grad vs `(L(x+ε)−L(x−ε))/2ε`, ratio≈1), confirm gradients finite AND non-negligible (not silently zeroed by `stop_gradient`, frozen/`enable_*=False` leaf, `lax.stop_gradient`/`jax.lax.cond` dead branch, or detached recompute). (3) **Sanity vs known-good baseline FIRST:** derivative-free or prior method got large improvements and your gradient method barely move loss → YOUR optimization broken (weak objective, tiny effective LR, over-aggressive line-search backtracking to ~0 step, frozen leaves) — fix before drawing conclusions. Concluding "residual bias is radiative/structural, not tunable" from run whose optimizer never actually worked = exact error that ship FALSE scientific claim. Near operationalization this disqualifying: barely-moving loss, suspiciously-flat before/after, or result contradicting known-good baseline BLOCKS delivery until gradient path verified end-to-end. See [[feedback_verify_ad_before_physics]].
- **CRITICAL — Visual verify spatial/grid artifacts**: passing tests+norms NECESSARY ≠ SUFFICIENT for cubed-sphere ops, halo exchange, diffusion coeffs, grid metrics. Edge artifacts/cube imprint/grid-scale noise only detected visually (v-wind W2, wind_speed W5). Run `--only sw --grid cubed_sphere --quick` + inspect PNGs vs baseline. Norms can improve while artifacts worsen. Never claim "tests pass, edge fixed" from pytest alone.
- **Diffusion sensitivity**: div damping + hyperdiff AMPLIFY halo errors at cubed-sphere face boundaries. Check W2 v-wind visually when touching `_hyperdiff_cube`, `_div_damp_cube`, diffusion params.
- **Visual-regression gate (cube imprint)**: `scripts/validate/visual_regression.py --check` numericises W2 v-wind cube-imprint check (SSIM + per-panel perceptual hash + edge-artifact ratio vs tiny committed ref in `tests/visual_baselines/`). Deterministic metric math gated in CI (`tests/test_visual_regression_metrics.py`); full cube-SW `--check` runs as NIGHTLY non-blocking CI job until tolerances calibrated across CI hardware. Tiny numeric baselines (.npy+json) ARE tracked — the one carve-out to "no tracked visual baselines".
- **CRITICAL — VALIDATE THE INSTRUMENT BEFORE QUOTING ITS NUMBER (2026-07-25 duogrid lesson: 8 confident claims, all retracted).** Diagnostic script = UNTRUSTED CODE until it pass own controls. Never state finding — never write "measured", "confirmed", "proven", "VERDICT" — from probe's first output. **Before quoting any diagnostic number, run these five checks and say in the message that you ran them:**
  1. **Right conserved/invariant quantity?** Budget what SYSTEM conserves, not convenient proxy. (Failed: reported "vertex creates energy" from **KE alone** — KE NOT conserved in shallow water, it trade with PE. Total `E=∫area(½h|V|²+½gh²)` reversed sign of conclusion.)
  2. **Same transform / units / staggering on BOTH sides?** Two "A-grid winds" from different operators = DIFFERENT QUANTITIES. (Failed: ours `c2l_ord2` vs oracle `C2L_ORD=4` — SAME raw state gave 1.96e-2 vs 5.79e-2, 3× swing that WAS the reported effect. Also: never budget across stage boundary where state change representation — mid-step FV3 winds in circulation form, produced ±5.6e10 garbage. REPEAT OFFENCE 2026-08-05, this time SHIPPED then retracted: paired predicted face velocity (41.87 m/s) against model's `max_speed` timeseries diagnostic (21.95 m/s), which is CELL-CENTRE average of same field whose face maximum was 42.30 — apparent "1.2 % agreement" was artifact of comparing across staggering, and it went into merged PR before 1-step run exposed it. Name staggering AND reduction of both sides in sentence quoting them.)
  3. **Same time, resolution, config?** Index by MATCHED TIME, not frame number. (Failed: mapped day→frame as `round(day)-1` against HOURLY file, comparing our day-1 to their hour-1; and quoted **C12** wedge gain (~300×) as mechanism for **C48** instability, where it ~124×.)
  4. **Is the metric measuring what its name says?** Prove it on synthetic case with KNOWN answer before use. (Failed: called `mean|f−4-neighbour-mean|` a "2Δx grid-scale" measure — it is high-pass/curvature residual that merely sharper SMOOTH feature reproduce. Failed: "gain" probe that re-filled FIXED source, trivially 1.0000 by construction.)
  5. **Can the reduction support the claim?** `max` over tiles/corners/components taken independently per run can peak at DIFFERENT physical locations; max-of-per-tile-means is not global mean. Keep argmax metadata, map to common physical location before claiming "localized".
  Plus: **diff ARRAYS, never printed summaries** (claimed "bit-identical ⇒ deterministic, not chaos"; arrays actually differed by 9e-6 — only rounded printout matched). **Never let probe print own verdict** ("=> the growth is REAL") — interpretation belong in analysis after controls pass, not baked into tool where it get echoed back as evidence. **`nanmax`/`nanmean`/`nansum` hide failures** — make NaN and missing frames FATAL. **Record every effective flag, env var and git SHA in each artifact**; default `--n 36` silently mis-slicing C48 file run fine and lie.
- **CRITICAL — LABEL EVERY CLAIM, AND PREFER RETRACTING EARLY.** Tag each statement **CONFIRMED** (control-passed evidence shown, instrument validated) vs **PLAUSIBLE** (inferred / single-run / uncontrolled). Chain of PLAUSIBLE steps is not CONFIRMED. Later measurement contradict earlier claim → **retract loudly and immediately in same message and in memory file** — no quiet move on, because stale confident claims get built on. Run codex adversarial review on DIAGNOSTIC TOOLING, not just model code: instruments decide what you believe, so bug there manufacture confident wrong physics conclusion. Cheap self-check before any big claim: *"what measurement would make this false, and did I run it?"* Answer no → claim PLAUSIBLE at best.

## Domain Architect vs Syntax Engine (AI guardrails)
See `docs/architecture/ai_guardrails/domain_architect_vs_syntax_engine.md`. Doctrine: human dictate *logic* (units, signs, conserved qty, valid scheme sets, references, acceptance criteria); AI fill *body*; every declared invariant checked **mechanically** so violations fail LOUDLY. Each gate = **tripwire, not proof**, ships synthetic-violation self-test (provably non-vacuous). NON-NEGOTIABLE harness (extend, never weaken; budgets/TODOs shrink only):
- **Spec-first physics contracts**: every physics scheme module declares `__physics_contract__` (units/signs/conserves/differentiable/reference/idealized_test); `tests/test_physics_contracts.py` partitions all `*/physics/*.py` into EXCLUDED / CONTRACT_TODO(shrink-only) / annotated — NEW physics file must ship contract or be classified. Author contract + acceptance test BEFORE body.
- **Atmospheric parameterization edits are high-risk**: run codex adversarial review, #477 truth-tier tendency validators, conservation tests, equilibrium-SCM-RCE realism harness before merge. NEW scheme must pass RCE realism gate or enter its shrink-only TODO partition with a reason.
- **CI ratchets** (AST + self-test, via shared `tests/_ratchet_audit.py`): `test_no_hardcoded_constants` (constants only from `legoesm.constants`), `test_no_saturation_reimpl` (saturation only from `thermo`), `test_dispatch_hardening` (no scheme guard silently deleted), `test_validate_strict_coverage` (no scheme field skips fail-early validation). Escape: real `# const-ok:`/`# satcurve-ok:` comment.
- **Local hooks** (`.claude/hooks/`, wired in `settings.json`): PreToolUse blocks edit adding banned constant/saturation prefactor to `.py` (fail-open); Stop reminds (once/session) to run mandatory codex review on uncommitted numerics/physics. CI remain authoritative (`Bash` heredoc bypass hook, not CI).
- **Best coding practices = use existing system**, not parallel one: ruff/mypy, import-linter (`alerting="error"`), inline-import budgets, pre-impl grep + shared utilities (no re-derivation), every new `.py` gets unit test, slopbuster sweeps, iterate-with-codex on substantial changes.


exec
/usr/bin/bash -lc "rg -n '/["'^ ]*(bin/python|activate)|conda|venv'"' .claude/settings* scripts/run/*.pbs scripts/validate/*.pbs 2>/dev/null | head -30; ls -d /glade/work/pg2328/*env* /glade/derecho/scratch/pg2328/*env* /glade/u/home/pg2328/.conda/envs/* 2>/dev/null; sed -n '1,100p' packages/land/legoesm/land/surface_scheme/base.py; sed -n '530,579p' packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py; git diff 8750d0005d -- packages/land/legoesm/land/canopy/stability.py" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
/glade/work/pg2328/conda-envs
/glade/work/pg2328/cpuperf_env.sh
/glade/work/pg2328/env_backups
/glade/work/pg2328/envs
/glade/work/pg2328/headroom-venv
/glade/work/pg2328/venvs
"""Shared output types for all surface schemes.

A surface scheme is a pure function from ``(state, forcing, config, params)``
to a ``SurfaceFluxOutput``.  Downstream land-model post-processing
(snow, Richards, soil thermal, carbon, TileResponse) consumes this
NamedTuple without knowing which scheme produced it.
"""

from __future__ import annotations

from typing import NamedTuple

import jax


class SurfaceFluxOutput(NamedTuple):
    """Output of a surface scheme (``SimpleSEB`` or ``TwoLeafCanopy``).

    All energy fluxes in [W / m^2].  ``shflx`` / ``lhflx`` follow the
    coupler sign convention (positive = INTO the atmosphere).  ``G_soil``
    is positive INTO the soil.  ``sw_net`` / ``lw_net`` are positive INTO
    the surface (absorbed).

    The output contains everything the shared post-flux pipeline in
    ``step_multilayer_land`` / ``step_land`` needs to run snow / Richards
    / soil thermal / carbon / TileResponse — no scheme-specific branches
    are required downstream.

    Fields that only apply to the two-leaf canopy scheme (``Tf_Sun``,
    ``gs_Sun``, ``n_iters``, ``f_veg_canopy``, etc.) are ``None`` for
    SimpleSEB.  Conversely, ``beta_soil_post`` is populated only by
    SimpleSEB (which uses it in the post-step ``q_surface`` recomputation)
    — for canopy the skin-layer bulk-beta concept does not apply.
    """

    # ---- Turbulent fluxes (atmosphere-surface interface) ----
    shflx: jax.Array             # sensible heat to atmosphere [W/m^2]
    lhflx: jax.Array             # latent heat to atmosphere [W/m^2] (demanded)
    tau_x: jax.Array             # zonal momentum flux [N/m^2]
    tau_y: jax.Array             # meridional momentum flux [N/m^2]

    # ---- Radiation ----
    sw_net: jax.Array            # absorbed shortwave [W/m^2]
    lw_net: jax.Array            # net longwave (down - up) [W/m^2]
    lw_up: jax.Array             # upward longwave (for coupler) [W/m^2]

    # ---- Ground heat flux ----
    G_soil: jax.Array            # positive INTO soil (top BC for solve_soil_thermal) [W/m^2]

    # ---- Surface state for coupler / radiation ----
    T_surface: jax.Array         # emission-weighted surface T [K]
    q_surface: jax.Array         # surface specific humidity [kg/kg]
    albedo: jax.Array            # broadband shortwave albedo [-]
    emissivity: jax.Array        # broadband longwave emissivity [-]
    z0: jax.Array                # roughness length [m]

    # ---- Photosynthesis (None if stomata disabled / carbon off) ----
    gpp: jax.Array | None = None  # gross primary production [gC/m^2/s]

    # ---- Solar-induced fluorescence (None unless a SIFConfig is attached) ----
    # Observed top-of-canopy SIF photon flux [umol/m^2/s] (sunlit+shaded sum for
    # the two-leaf canopy, single leaf for SimpleSEB), scaled by the escape
    # probability fesc.  Passive diagnostic — no feedback into the land state.
    sif: jax.Array | None = None
    # Last CONVERGED canopy Newton solution per column, (ncol, 6) in the solver's
    # own variable order, NaN where this column has never converged.  Purely a
    # warm-start cache for the next call (the fixed point does not depend on the
    # seed, and the solve's adjoint returns a zero cotangent for it); None for
    # every scheme that does not run the canopy closure.
    canopy_x: jax.Array | None = None
    # Canopy solver TERMINAL DIAGNOSTICS (None for every other scheme).  The
    # solver's own squared residual at exit, that value relative to the seed's
    # (the ratio the relative convergence gate tests), and 1.0 where the loop
    # left by the ITERATION CAP rather than the damping ceiling.  These exist
    # because ``converged`` alone cannot distinguish a stalled column from a
    # nearly-solved one; non-differentiable, no feedback into the land state.
    canopy_resid_sq: jax.Array | None = None
    canopy_resid_rel: jax.Array | None = None
    canopy_hit_cap: jax.Array | None = None

    # ---- Canopy-specific diagnostics (None for SimpleSEB) ----
    Tf_Sun: jax.Array | None = None        # sunlit leaf T [K]
    Tf_Sh: jax.Array | None = None         # shaded leaf T [K]
    # Canopy air-space temperature Tc [K] = conductance-weighted blend of the
    # above-canopy air, sunlit/shaded leaves, and ground.  This is the canopy's
    # AERODYNAMIC surface temperature (the exchange node for sensible heat,
    # H_tot = rho*cp*(Tc - Ta)/Ra), and is what the coupler reports as the tile
    # ``T_sfc`` for the atmosphere's sensible-heat coupling.  Distinct from the
    # LW-derived radiometric ``T_surface``.  None for SimpleSEB.
    T_canopy_air: jax.Array | None = None
    # CLM5 ``t_ref2m``: 2 m air temperature from the scheme's OWN surface layer
    # (roughness, displacement, Obukhov length, canopy-air node) [K].  The CMIP
    # ``tas`` land leg.  None for schemes that do not solve one.
    T_2m: jax.Array | None = None
    gs_Sun: jax.Array | None = None        # sunlit stomatal conductance [m/s]
    gs_Sh: jax.Array | None = None         # shaded stomatal conductance [m/s]
    n_iters: jax.Array | None = None       # canopy Newton iteration count
    # Per-column flag: did the canopy Newton closure reach a root?  ``False``
    # means the accompanying fluxes come from a stopped iterate (iteration cap
    # hit, or a singular Jacobian whose step had to be discarded) and are NOT a
    # solution is a far better seed than the air temperature — same fixed point,
    # fewer iterations to reach it.  Only a CONVERGED pass earns the right to
    # seed the next one: a failed solve's last iterate is not a root, and
    # reusing it would let one bad pass cascade through the rest.
    # ``x_conv`` holds the last CONVERGED solution per column, NaN until one
    # exists; it is both the next pass's seed and (after the loop) the cache
    # handed back to the caller.
    # A column with no converged solution yet alternates its seed across passes
    # between the caller's seed and the (slightly cooled) retry state, so a seed
    # that led the solve astray gets a different retry inside the same call.  The
    # passes run over every column anyway (one batch), so this costs nothing.  A
    # failed pass still relaxes the soil boundary with its cold-state fallback
    # flux (the one the land step accepts if the column ends unsolved); freezing
    # it held desert columns at their start-of-step soil.  Acceptance stays tied
    # to the LAST pass.
    x_conv = jnp.full_like(initial_state, jnp.nan)
    for _picard_iter in range(n_picard):
        bundles_k = _build_bundle(Ts_bc_k)
        _fallback = initial_state if _picard_iter % 2 == 0 else retry_state
        _seed_k = jnp.where(jnp.all(jnp.isfinite(x_conv), axis=-1, keepdims=True),
                            jnp.nan_to_num(x_conv), _fallback)
        # One batch over every column (user choice 2026-11-09, replacing the
        # 64-column groups chosen 2026-11-03).  Once the LM step stopped forming
        # explicit Q (PR 2398) the solve is launch-bound, so one wide batch
        # beats many narrow ones: AMIP production deck, 4 A100, fp64, 1 day,
        # 0.242 s/step with groups of 64 vs 0.138 s/step with one batch.
        # Pure performance: same per-column solve, answers differ by rounding.
        # Cost: the batch iterates until the rank's slowest column stops, so a
        # column stuck at the iteration cap holds every column to the cap.
        (x_final, n_iters, converged, resid_sq, resid_rel, lam_f,
         hit_cap) = jax.vmap(_solve_one_col)(_seed_k, bundles_k)
        x_conv = jnp.where(converged[:, None], x_final, x_conv)
        # A rejected iterate may be non-finite: evaluate the fluxes of a failed
        # column at the (finite) cold state instead, so no NaN enters a
        # reverse-mode tangent.  Those fluxes are the column's FALLBACK, not a
        # root: it reports converged=False, and the multilayer land step closes
        # their energy into sensible heat and accepts the column (or reverts it
        # if non-finite or beyond the fallback guards).
        x_final = jnp.where(converged[:, None], x_final, cold_state)
        fluxes_per_col = jax.vmap(_fwd_one_col)(x_final, bundles_k)

        G_k = jnp.clip(fluxes_per_col["G"], -500.0, 700.0)  # coeff-ok: physical range clamp on ground heat flux [W m-2]
        Ts_thermal = soil_thermal_fn(G_k, dt)
        Ts_bc_k = (1.0 - omega) * Ts_bc_k + omega * Ts_thermal

    # ---- Converged state ----
    # State vector order: [Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c].
    Tf_Sun = x_final[:, 0]
    Tf_Sh  = x_final[:, 1]
    Tc_cvg = x_final[:, 4]   # canopy air-space temperature (aerodynamic node)
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
 
 

exec
/usr/bin/bash -lc "ls /glade/work/pg2328/conda-envs /glade/work/pg2328/envs /glade/work/pg2328/venvs; sed -n '820,870p' packages/coupler/legoesm/driver/model_driver.py; rg -n 'surface_out\\[[0-9]|for .*surface_out|zip\\(.*surface_out|SurfaceFluxOutput\\(\\*|_fields|T_canopy_air' packages/land/legoesm/land/patch_mosaic.py packages/land/legoesm/land/step_land.py packages/land/legoesm/land/surface_scheme/*.py packages/coupler/legoesm/driver/model_driver.py | head -50" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
/glade/work/pg2328/conda-envs:
legoesm-gpu

/glade/work/pg2328/envs:
headroom

/glade/work/pg2328/venvs:
context-compress
    microphysics ice+snow+graupel + convective snow, whose fusion heat the
    atmosphere already booked), bounded by the total ``precip`` (review
    2026-10-10 F29; it replaces a hard 0 C re-split of the total that
    discarded the phase and left L_f unclosed at the interface).

    No precipitation published (slot 2 empty) -> zeros.  Precipitation
    published WITHOUT its frozen part (a scheme that does not publish it)
    -> ValueError: never a temperature re-split.
    """
    from legoesm.core.state import (
        MPAS_SFC_DIAG_BASE_KEYS,
        MPAS_SFC_DIAG_EXTRA_KEYS,
    )
    slot = (len(MPAS_SFC_DIAG_BASE_KEYS)
            + MPAS_SFC_DIAG_EXTRA_KEYS.index("precip_frozen"))
    frozen = sfc_diag[slot] if len(sfc_diag) > slot else None
    if frozen is not None:
        return jnp.clip(jnp.asarray(frozen.data).reshape(-1), 0.0, precip)
    if sfc_diag[2] is None:
        return jnp.zeros_like(precip)
    raise ValueError(
        "interactive land: the physics publishes surface precipitation but "
        "not its frozen part (precip_frozen), so the land cannot take the "
        "atmosphere's own rain/snow partition (F29). Use a microphysics/"
        "convection scheme that publishes it (every physical scheme does; "
        "the ML microphysics emulators and neural physics do not).")


def _sed_substeps_slot() -> int:
    """Index of ``sed_substeps_required`` in the MPAS ``_sfc_diag`` tuple.

    Derived from the SHARED slot contract (``MPAS_SFC_DIAG_EXTRA_KEYS``
    entries start at tuple slot 3), never hand-counted.
    """
    from legoesm.core.state import (
        MPAS_SFC_DIAG_BASE_KEYS,
        MPAS_SFC_DIAG_EXTRA_KEYS,
    )
    return (len(MPAS_SFC_DIAG_BASE_KEYS)
            + MPAS_SFC_DIAG_EXTRA_KEYS.index("sed_substeps_required"))


def _sed_published_count(sfc_diag, slot, counts):
    """The step's sub-step count from either publication route, or None.

    ``counts`` is the count a NON-MPAS lane returns directly (the
    finite-volume pipeline's ``PhysicsOutput.sed_substeps_required``);
    ``sfc_diag`` is the MPAS lean-loop export tuple.  Both routes exist
    because the two lanes publish diagnostics differently (codex 2026-09-22:
    reporting only inside the MPAS loop left the finite-volume lane silent).
    """
rg: packages/land/legoesm/land/patch_mosaic.py: No such file or directory (os error 2)
rg: packages/land/legoesm/land/step_land.py: No such file or directory (os error 2)
packages/land/legoesm/land/surface_scheme/simple_seb.py:32:from legoesm.core.coupling_fields import AtmToSurface
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:31:from legoesm.core.coupling_fields import AtmToSurface
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:732:        T_canopy_air=Tc_cvg,
packages/coupler/legoesm/driver/model_driver.py:222:def _exchange_halo_cell_fields(halo_ex, fields):
packages/coupler/legoesm/driver/model_driver.py:1716:    from legoesm.core.coupling_fields import AtmToSurface
packages/coupler/legoesm/driver/model_driver.py:1732:                  for f in AtmToSurface._fields
packages/coupler/legoesm/driver/model_driver.py:1734:        missing = [f for f in AtmToSurface._fields
packages/coupler/legoesm/driver/model_driver.py:1811:    carry = {f: getattr(base_params, f) for f in params._fields
packages/coupler/legoesm/driver/model_driver.py:2214:        _root_fields = {
packages/coupler/legoesm/driver/model_driver.py:2219:        if _root_fields:
packages/coupler/legoesm/driver/model_driver.py:2220:            params = params._replace(**_root_fields)
packages/coupler/legoesm/driver/model_driver.py:2224:                ", ".join(sorted(_root_fields)))
packages/coupler/legoesm/driver/model_driver.py:2282:        _canopy_fields = [f for f in ("hc", "ALB_VIS", "ALB_NIR", "rz0m")
packages/coupler/legoesm/driver/model_driver.py:2284:        if not _canopy_fields:
packages/coupler/legoesm/driver/model_driver.py:2968:        valid = set(template._fields)
packages/coupler/legoesm/driver/model_driver.py:2974:        expected = {f for f in template._fields
packages/coupler/legoesm/driver/model_driver.py:2978:        _snow_layer_fields = {"snow_ice_layers", "snow_liq_layers",
packages/coupler/legoesm/driver/model_driver.py:2984:        _seed_pack = (got != expected and (expected - got) == _snow_layer_fields
packages/coupler/legoesm/driver/model_driver.py:2987:            expected = expected - _snow_layer_fields
packages/coupler/legoesm/driver/model_driver.py:2988:        if got != expected and (expected - got) == _snow_layer_fields:
packages/coupler/legoesm/driver/model_driver.py:5582:            aerosol_layer_fields,
packages/coupler/legoesm/driver/model_driver.py:5692:        aerosol_od, _ccn_aod, self._aerosol_lw_od = aerosol_layer_fields(
packages/coupler/legoesm/driver/model_driver.py:5890:            self.diagnostics.set_fixed_fields(
packages/coupler/legoesm/driver/model_driver.py:6258:        # names an input dataset. Membership-checked against _fields (legacy
packages/coupler/legoesm/driver/model_driver.py:6261:        dataset_path_fields = (
packages/coupler/legoesm/driver/model_driver.py:6273:        config_fields = type(self._input_config)._fields
packages/coupler/legoesm/driver/model_driver.py:6276:            for name in dataset_path_fields
packages/coupler/legoesm/driver/model_driver.py:6277:            if name in config_fields and getattr(self._input_config, name)
packages/coupler/legoesm/driver/model_driver.py:8127:                    for _name in _ps_carry._fields:
packages/coupler/legoesm/driver/model_driver.py:9694:                and _k[len("physstate_"):] not in PhysicsState._fields
packages/coupler/legoesm/driver/model_driver.py:9700:                    f"schema {list(PhysicsState._fields)} — this binary is "
packages/coupler/legoesm/driver/model_driver.py:10823:                                   global_fields: bool = False) -> None:
packages/coupler/legoesm/driver/model_driver.py:10842:        (``global_fields=True``: ``kw`` then holds GLOBAL ``q_v`` / ``p_s`` /
packages/coupler/legoesm/driver/model_driver.py:10846:        if _is_mpas_cell_partitioned(self) and not global_fields:
packages/coupler/legoesm/driver/model_driver.py:10857:        if global_fields:
packages/coupler/legoesm/driver/model_driver.py:11508:                    day, diag, gathered, global_fields=True)
packages/coupler/legoesm/driver/model_driver.py:12192:    def _fv3_duo_snapshot_fields_and_blowup(self, bundle: dict):
packages/coupler/legoesm/driver/model_driver.py:12266:        fields, bad, umax = self._fv3_duo_snapshot_fields_and_blowup(bundle)
packages/coupler/legoesm/driver/model_driver.py:13355:            and "hard_saturation_adjustment" in getattr(_hsub, "_fields", ()))
packages/coupler/legoesm/driver/model_driver.py:14413:            from legoesm.core.coupling_fields import AtmToSurface
packages/coupler/legoesm/driver/model_driver.py:14905:                from legoesm.core.coupling_fields import (
packages/coupler/legoesm/driver/model_driver.py:15221:            _present_fields = {
packages/coupler/legoesm/driver/model_driver.py:15294:                _missing = [f for f in _phys_state._fields
packages/coupler/legoesm/driver/model_driver.py:15295:                            if f not in _present_fields
packages/coupler/legoesm/driver/model_driver.py:15311:                        f"present {sorted(_present_fields)}, missing "
packages/coupler/legoesm/driver/model_driver.py:15330:                if _name not in _phys_state._fields:
packages/coupler/legoesm/driver/model_driver.py:16408:                         ) = _exchange_halo_cell_fields(
packages/coupler/legoesm/driver/model_driver.py:19407:            # (``coupling_fields.require_surface_radiation_aux``); without them
packages/coupler/legoesm/driver/model_driver.py:19451:                for _nm in carry._fields
packages/coupler/legoesm/driver/model_driver.py:20860:                # packages/core/legoesm/core/coupling_fields.py:18) is a RATE

exec
/usr/bin/bash -lc "sed -n '10823,10997p' packages/coupler/legoesm/driver/model_driver.py; ls /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python /glade/work/pg2328/envs/headroom/bin/python; rg -n 'SurfaceFluxOutput\\(|tree_flatten\\(surface|surface_out\\._|surface_out\\)' packages/land/legoesm/land --glob '*.py'" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
                                   global_fields: bool = False) -> None:
        """Record the atmospheric water-budget closure E - P - dW/dt.

        The model has always carried this tracker, and this lane has never fed
        it: the tracker is updated inside ``DiagnosticCollector.collect``, which
        the MPAS run loop does not call, so every run of this campaign published
        a blank moisture residual. A ~0.4 mm/day gap between the reported global
        evaporation and rainfall therefore sat unexamined for months. An
        instrument that is not wired is not a check.

        Fed from the SAME window-mean ``precip`` and ``hfls`` the CMOR output
        publishes, which is the tracker's own stated contract: the residual then
        closes against the numbers a reader can see in ``pr`` and ``hfls``,
        rather than against a second, privately-averaged pair that could differ
        for reasons nobody could trace.

        Under a multi-rank cell partition the tracker's plain area-weighted
        mean would be rank-local and count halo cells twice, so it is fed ONLY
        on rank 0 from the owned-cell gather of the CMOR feed
        (``global_fields=True``: ``kw`` then holds GLOBAL ``q_v`` / ``p_s`` /
        ``precip`` / ``evspsbl`` and the weights are the global ``areaCell``).
        A rank-local call under the partition still returns without recording.
        """
        if _is_mpas_cell_partitioned(self) and not global_fields:
            return
        # The CMOR slot getter falls back to an INSTANTANEOUS diagnostic when a
        # slot has no accumulated samples.  Mixing a mean rainfall with an
        # instantaneous evaporation (or the reverse) manufactures an imbalance
        # out of nothing, so this closure takes the window means or nothing.
        _acc = getattr(self, "_mpas_sfc_accum", None)
        if _acc is None or not _acc.has_samples() or not _acc.is_complete():
            return
        precip = kw.get("precip")
        evspsbl = kw.get("evspsbl")
        if global_fields:
            q_v, _p_s = kw.get("q_v"), kw.get("p_s")
            area = getattr(self._grid_global, "areaCell", None)
        else:
            tracers = self.state.tracers
            q_v = (tracers["q_v"].data
                   if tracers is not None and "q_v" in tracers else None)
            _p_s = self.state.p_s.data
            area = getattr(self.grid, "areaCell", None)
        if precip is None or q_v is None or _p_s is None:
            return          # dry run, or a window whose fluxes were withheld
        if evspsbl is None:
            # Heat without water: the closure will NOT rebuild E as hfls / L_v
            # (the hidden 2-3 % fallback the water channel removed).  Loud,
            # because a silent return would read as "the budget closes".
            import warnings
            warnings.warn(
                "MPAS moisture closure skipped: the window has precip and hfls "
                "but no evspsbl (turbulence evap_sfc slot); the closure refuses "
                "to derive water from latent heat.", stacklevel=2)
            return
        _p_s = jnp.asarray(_p_s)
        _p_half = self.sigma.pressure_at_half(_p_s)
        diag.moisture_tracker.update(
            jnp.asarray(q_v), _p_s, self.sigma.dsigma,
            precip, evap=evspsbl,
            elapsed_seconds=float(day) * 86400.0,
            area_weights=(None if area is None
                          else jnp.asarray(area).reshape(-1)),
            # Hybrid coordinates make ``p_s * dsigma`` wrong for a bottom-heavy
            # tracer over terrain; the half-level difference is right for either
            # coordinate.
            dp=_p_half[..., 1:] - _p_half[..., :-1],
        )

    def _mpas_surface_temperatures(self, day, diag, u_east, v_north):
        """Shared native whole-cell tas, skin ts, prescribed sea-ice fraction
        sic and prescribed SST sst (None without a forcing) for both sample
        cadences."""
        state = self.state
        q_v = (state.tracers["q_v"].data
               if state.tracers is not None and "q_v" in state.tracers else None)
        # 2 m ``tas`` via MOST similarity when prescribed sst/sic are on
        # this path (``get_sst_sic`` set for a radiation+SST run) — matches
        # the cube-path collect() ``tas`` instead of a bare lowest-level
        # proxy.  Uses the RECONSTRUCTED cell winds (``state.u`` is
        # edge-normal on MPAS, not cell-collocated).  A failure falls back
        # to the lowest model level (logged once) so a tas-only glitch never
        # drops the whole CMOR feed.
        tas = None
        ts = None
        sic = None
        sst = None
        _get_sst_sic = getattr(self, "get_sst_sic", None)
        if _get_sst_sic is not None:
            try:
                _sst, _sic = _get_sst_sic(day)
                _sst = jnp.asarray(_sst).reshape(-1)
                _sic = jnp.asarray(_sic).reshape(-1)
                sic = _sic
                sst = _sst
                # Report tas off the SAME ice surface the radiation +
                # turbulence saw: the per-cell prognostic skin when the
                # feature is on, else the constant T_ice.  Otherwise the
                # scorecard's 2 m extrapolation uses a 271.35 K ice surface
                # while the model cooled the skin (codex-1 finding 3).
                _tas_ice = getattr(self.config, "T_ice", None)
                if (getattr(self.config, "ice_skin_prognostic", False)
                        and getattr(self, "_ice_T_skin", None) is not None):
                    _tas_ice = self._ice_T_skin
                # Give the diagnostic the LAND surface too where the
                # interactive tile has produced one, else its profile is
                # anchored on the ocean/ice skin over land as well and the
                # published land tas is not the model's land at all.
                _tas_land_T = getattr(self, "_land_T_skin_last", None)
                _tas_land_q = getattr(self, "_land_qsfc_last", None)
                _tas_land_T2m = getattr(self, "_land_T2m_last", None)
                _tas_f_land = getattr(self, "_f_land", None)
                _tas_kw = {}
                if _tas_land_T is not None and _tas_f_land is not None:
                    _tas_kw = dict(
                        T_land=jnp.asarray(_tas_land_T).reshape(-1),
                        q_land=(None if _tas_land_q is None
                                else jnp.asarray(_tas_land_q).reshape(-1)),
                        land_fraction=jnp.asarray(_tas_f_land).reshape(-1),
                        T2m_land=(None if _tas_land_T2m is None
                                  else jnp.asarray(_tas_land_T2m).reshape(-1)))
                from legoesm.forcing.surface_utils import blend_surface_temperature
                ts = blend_surface_temperature(_sst, _sic, _tas_ice)
                if _tas_land_T is not None and _tas_f_land is not None:
                    f_land = jnp.clip(jnp.asarray(_tas_f_land).reshape(-1), 0.0, 1.0)
                    ts = (f_land * jnp.asarray(_tas_land_T).reshape(-1)
                          + (1.0 - f_land) * ts)
                tas = diag._tas_2m(
                    state, q_v, _sst, _sic, _tas_ice,
                    u_low=u_east[..., -1], v_low=v_north[..., -1], **_tas_kw)
            except Exception as exc:
                if not getattr(self, "_logged_tas2m_fallback", False):
                    logger.warning(
                        "  CMOR tas: 2 m MOST calc failed (%s); using the "
                        "lowest model level as the tas proxy.", exc)
                    self._logged_tas2m_fallback = True
                tas = None
        return (state.T.data[..., -1] if tas is None else tas), ts, sic, sst

    def _mpas_cmip_native_kwargs(self, day: float, diag,
                                 u_override=None, *, state_only=False) -> dict:
        """Build the NATIVE (rank-local) cell-field kwargs for
        :meth:`DiagnosticCollector.feed_cmip_accumulators_native`.

        PURE and COLLECTIVE-FREE: reads ``self.state`` / the physics flux
        export and returns host/device arrays whose leading axis is the
        caller's cell axis (``n_local_cells`` under a Voronoi partition,
        ``nCells`` serial).  Every value may be ``None`` (the collector skips
        absent fields); ``flux_interval_days`` is the one scalar entry.

        Split out of :meth:`_feed_mpas_cmip_accumulators` so the multi-rank
        gather can reuse the SAME field construction — and so this work stays
        strictly outside any MPI collective (a rank that fails here must not
        leave its peers blocked; see :meth:`_feed_mpas_cmip_multirank`).

        ``u_override`` supplies an edge field whose HALO edges have already
        been refreshed.  It exists because this method must stay
        collective-free while the Perot reconstruction below genuinely needs
        valid halo edges: under a cell partition an OWNED cell on the cut is
        RINGED by halo edges, so a stale one corrupts that cell's ``ua``/``va``
        — silently, and only along the cut.

        The clear case is column-local physics: ``_phys_col_local``, set from
        ``physics_fn._column_local`` (today only the idealized Kessler MPAS
        forcing sets it), makes ``make_voronoi_mpi_step`` skip the pre-physics
        state exchange, and there is no post-physics one, so halo edges keep
        ``old_halo + dt*local_tendency``.  The full AMIP physics does NOT set
        that attribute and therefore takes the exchanging path.  The exchange
        is done unconditionally anyway: it costs one collective per DIAGNOSTIC
        INTERVAL and removes the reconstruction's dependence on which physics
        path ran, rather than leaving a correctness argument that has to be
        re-derived whenever that gate moves.

        ``None`` (serial / single-rank) uses ``state.u`` directly and is
        byte-identical.
/glade/work/pg2328/conda-envs/legoesm-gpu/bin/python
/glade/work/pg2328/envs/headroom/bin/python
packages/land/legoesm/land/slab_land.py:629:    # albedo and emissivity from surface_out).  T_soil_new is already
packages/land/legoesm/land/multilayer_land.py:2061:    surface_out = surface_out._replace(
packages/land/legoesm/land/multilayer_land.py:2067:        surface_out = surface_out._replace(
packages/land/legoesm/land/surface_scheme/base.py:16:class SurfaceFluxOutput(NamedTuple):
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:710:    return SurfaceFluxOutput(
packages/land/legoesm/land/surface_scheme/simple_seb.py:357:    return SurfaceFluxOutput(
packages/land/legoesm/land/canopy/clm_ml_interface.py:1593:    return SurfaceFluxOutput(

exec
/usr/bin/bash -lc 'PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -m pytest -p no:cacheprovider tests/land/unit/test_land_screen_temperature.py tests/unit/test_land_column_split.py tests/land/unit/test_canopy_stability.py tests/land/unit/test_canopy_stability_faithful.py -q' in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
exec
/usr/bin/bash -lc "sed -n '1,65p' tests/conftest.py; sed -n '1,120p' tests/land/conftest.py; sed -n '1,95p' packages/land/legoesm/land/canopy/stability.py; rg -n 'SurfaceFluxOutput|surface_flux' packages/land/legoesm/land/mosaic* tests/land/unit/test_patch_mosaic.py; git diff --check 8750d0005d" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 24ms:
"""Shared test fixtures for legoESM."""

import functools
import os
import resource
import sys
from pathlib import Path

#: Cap the CPU affinity mask BEFORE JAX is imported (below).  XLA sizes its
#: Eigen thread pool from the affinity mask, so on a many-core login node a
#: single JAX process opens far more threads than ``ulimit -u`` allows once a
#: few run at once, and the pool constructor aborts the interpreter:
#:
#:   F env.cc:93] Check failed: ret == 0 (11 vs. 0) Thread tf_XLAPjRtCpuClient
#:   creation via pthread_create() failed.
#:     @ Eigen::ThreadPoolTempl<>::ThreadPoolTempl()
#:
#: (11 = EAGAIN.)  It surfaces as "Fatal Python error: Aborted" mid-run, which
#: reads like a test failure but is the process running out of thread slots.
#:
#: Measured on a 256-core Levante login node (``ulimit -u`` = 2048), threads
#: opened by one client for a trivial jitted matmul:
#:
#:     CPUs    4     8    16    32    64   256
#:     threads 29    45    77   141   237   621
#:
#: Uncapped that is 621 threads each, so FOUR concurrent clients (2484) are the
#: first count over the 2048 ceiling; five were observed to fail 5/5.  At 32
#: CPUs a client takes ~141, so ~14 fit.  Capping costs nothing measurable --
#: these tests are XLA-compile-bound, not intra-op-parallelism-bound
#: (tests/test_corner_div_damp_nh.py: 166.9 s on 256 CPUs, 166.3 s on 8,
#: 152.7 s on 32).
#:
#: Set ``LEGOESM_TEST_CPU_CAP`` to a positive integer to force a cap, or to 0
#: (or any value <= 0) to disable and inherit the machine's full affinity mask.
_DEFAULT_TEST_CPU_CAP = 32

#: Threads one XLA CPU client opens ~= _THREADS_PER_CPU * ncpu + _THREADS_BASE,
#: fitted to the table above (slope (621-29)/(256-4) = 2.35).  Used to size the
#: cap against the ACTUAL thread ceiling rather than trusting a fixed 32, which
#: is only evidenced safe for ulimit -u = 2048.
_THREADS_PER_CPU = 2.4
_THREADS_BASE = 20.0
#: Fraction of the thread ceiling the test session may claim, leaving room for
#: the user's other processes (a login node already carries ~50 threads).
_RLIMIT_SHARE = 0.5

#: Env vars every common launcher sets in a rank's environment.  Under MPI the
#: launcher (or the batch scheduler) owns placement: each rank already gets its
#: own binding, and clamping every rank to the SAME low-numbered CPUs here would
#: pile all of them onto one core set and serialise the run.  So the cap is
#: skipped for MPI ranks -- they are separate processes with a launcher-assigned
#: mask, which is exactly the placement this function would otherwise destroy.
#: MPI does not standardise a rank env var, so detection is BEST-EFFORT: this
#: covers the launchers in use here.  For an unlisted launcher, disable the cap
#: explicitly with ``LEGOESM_TEST_CPU_CAP=0``.
_MPI_RANK_ENV = (
    "OMPI_COMM_WORLD_RANK",     # Open MPI
    "PMI_RANK",                 # MPICH / Intel MPI
    "PMIX_RANK",                # PMIx
    "SLURM_PROCID",             # srun-launched
    "MV2_COMM_WORLD_RANK",      # MVAPICH2
    "MPI_RANKID",               # IBM Platform MPI
    "LAMRANK",                  # legacy LAM/MPI
)
"""Per-test JAX cache reset for the land tests.

Every compiled XLA executable keeps its code mapped in the process, and the
land tests compile many distinct per-column graphs (the canopy Newton solve,
the soil Richards/thermal solvers, eager op-by-op calls).  One long pytest
process climbs about 1600 memory maps per canopy test until it reaches the
kernel's ``vm.max_map_count`` (65530 on Derecho); LLVM then fails
``allocateMappedMemory`` ("Cannot allocate memory", "Failed to materialize
symbols") and the whole process aborts, at ~5 GB RSS on a 230 GB node.  With
the caches cleared after each test the count stays near 2000.

Test-harness mitigation only; production steps the same solvers inside one
jitted scan and holds a bounded set of executables.
"""

import jax
import pytest


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    yield
    jax.clear_caches()
"""Monin-Obukhov similarity theory and canopy aerodynamics.

Provides:
  - Aerodynamic roughness / displacement from canopy height and LAI.
  - Above-canopy MOST stability iteration (CLM5 unstable regimes; Beljaars &
    Holtslag 1991 stable side).
  - Below-canopy resistance (clumping-weighted Cs approach).
  - Leaf boundary-layer resistance from wind speed.

All functions are pure JAX, JIT-compatible, and differentiable.
Sources:
  DifferBESS/process/stability.py (Ryu et al. / CLM5 stability functions)
  DifferBESS/process/CarbonWaterFluxes.py (aerodynamic block)

Faithfulness
------------
The above-canopy MOST core is an exact port of the CLM5 ``FrictionVelocityMod``
4-regime Monin-Obukhov similarity functions (Oleson et al. 2013 CLM5 Tech Note;
Zeng et al. 1998).  ``tests/land/unit/test_canopy_stability_faithful.py`` pins the
per-regime resistance FORMS to round-off (rel 1e-9) against an independent scalar
reimplementation of those functions, at MATCHED ``zeta`` / ``z0`` / ``obu``:

  * momentum ``ustar`` (:func:`_friction_velocity`) and heat/scalar ``ch``
    (:func:`_temperature_humidity_relation`) in the two UNSTABLE regimes —
    very-unstable ``zeta < -zetam`` (mom) / ``< -zetat`` (heat) and unstable;
  * the Paulson (1970) unstable ``psi_m``/``psi_h``
    (:func:`_stability_func_momentum`/:func:`_stability_func_heat`), the
    free-convection matches (momentum ``1.14 * ((-zeta)^1/3 - zetam^1/3)``, heat
    ``0.8 * (zetat^-1/3 - (-zeta)^-1/3)`` — CLM5's INVERSE cube-root);
  * the neutral log-law limit ``ustar -> kappa u / ln(z/z0)`` and continuity of
    the forms across the free-convection transitions (the matches are C0).

DEPARTURE from CLM5 on the STABLE side (``zeta >= 0``, user 2026-09-25): CLM5's
linear ``psi = -5 zeta`` (``0 <= zeta <= 1``) and very-stable log branch are
replaced by Beljaars & Holtslag (1991), the shared
``legoesm.core.bulk_flux.psi_m/psi_h(..., "beljaars_holtslag1991")``.  Under
the CLM5 forms a dense canopy at night could have NO root for the canopy-air
energy balance (sensible heat fell with a growing air-surface temperature
difference), so the canopy solve stalled; BH weakens that decline.  Heat flux is
not strictly monotone for tall rough canopies at low wind (a ~1 W m-2 dip just
below the ``zeta <= 0.5`` cap remains).  See
``docs/land/canopy_stable_stability_bh91.md``.

The pin is on the FORMS at fixed ``zeta``, NOT the whole solve, because the
iteration DRIVER is a departure (see below), so a converged ``zeta`` is
model-specific.

DEPARTURES from the gSAM/CESM-LSM4 sibling MOST ``transfer_coef.f90`` (a related
Businger-Dyer scheme on disk — cross-checked in the shared regimes, NOT the same
scheme; each departure is a test canary):
  * very-unstable HEAT uses CLM5's INVERSE cube-root ``zetat^-1/3 - (-zeta)^-1/3``
    whereas the LSM4 sibling uses a growing ``(-zeta)^1/3 - zetat^1/3`` — the
    heat free-convection correction genuinely differs between the two models;
  * ``kB^-1 = 0`` (``z0h = z0m``, CLM5 vegetation; DifferBESS aa6e8b9) vs the
    LSM4 ``kB^-1 approx 2`` (``z0h = 0.135 z0``, i.e. ``ln(z0/z0h) = 2.0025``,
    the LSM4 rounding of the exact ``kB^-1 = 2`` -> ``z0h = z0 e^-2``);
  * no high-wind roughness reduction ``z0 (1 + U/10)^-0.6`` and no LSM4
    post-solve flux limiters (50%-slowdown cap; the LSM4 sibling also applies an
    UNCONDITIONAL ``ustar -> sqrt(ustar^2 + 0.05^2)`` floor, absent here);
  * the Obukhov solve is a FIXED ``n_iters`` (default 5) buoyancy-flux fixed
    point (``zeta = zldis kappa g thetav*/(ustar^2 Tv)``, Zeng 1998 bulk-Ri init),
    NOT the LSM4 bulk-Richardson ``zeta = r fm^2/fh`` iterated to tolerance.

NUMERICS / AD guards (no-ops in their own regime): ``_MOST_ARG_FLOOR`` floors
the log/cbrt args of the DISCARDED where-branches so ``0*NaN`` cannot poison the
reverse-mode gradient.  The scalar oracle does NOT reproduce this floor — it
evaluates only the in-regime branch (if/elif) and so never touches the discarded
args; the floor is instead exercised by the AD test (eager ``where`` evaluates
every regime).  ``zeta`` is clamped to ``[1e-6, 0.5]`` (stable; the 0.5 cap is a
smooth min, see ``CanopyConfig.zeta_cap_smoothing_width`` — CLM5 clips hard) / ``[-100,
-1e-6]`` (unstable) each iterate (CLM5: 0.01, see ``_ZETA_NEUTRAL_FLOOR``); wind floors (0.1, 1e-3 m/s) and resistance
floors (1e-9) guard calm/degenerate columns.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial

from legoesm import constants
from legoesm.core.bulk_flux import psi_h, psi_m
from legoesm.thermo import saturation_vapor_pressure_aerk

# Module-local numerics / stability parameters (not physical constants —
# those come from ``legoesm.constants``).
_ZETA_MAX_STABLE = 0.5
# Near-neutral |zeta| floor (keeps obu = zldis/zeta finite).  CLM5 uses 0.01,
# which makes zeta, ustar and the resistances JUMP by that much at neutral
# stability; the canopy Newton solve then stalls whenever its root sits at
# neutral (sunset).  At 1e-6 the jump is below the solver tolerance and the
# stability forms are continuous through zeta = 0 (user 2026-09-25).
_ZETA_NEUTRAL_FLOOR = 1.0e-6
_CONV_BDY_HEIGHT = 1000.0  # convective boundary layer height [m]
_SCREEN_HEIGHT_M = 2.0     # CMIP tas / CLM5 t_ref2m height above d + z0h [m]
rg: packages/land/legoesm/land/mosaic*: No such file or directory (os error 2)
tests/land/unit/test_patch_mosaic.py:14:from legoesm.land.surface_scheme.base import SurfaceFluxOutput
tests/land/unit/test_patch_mosaic.py:102:    """A minimal SurfaceFluxOutput over a patch axis; None diagnostic fields."""
tests/land/unit/test_patch_mosaic.py:107:    return SurfaceFluxOutput(

 exited 2 in 11753ms:
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
configfile: pyproject.toml
plugins: zarr-3.4.0, timeout-2.4.0, jaxtyping-0.3.11, xdist-3.8.0
collected 43 items / 1 error

==================================== ERRORS ====================================
_______ ERROR collecting tests/land/unit/test_land_screen_temperature.py _______
/glade/work/pg2328/conda-envs/legoesm-gpu/lib/python3.12/site-packages/_pytest/python.py:508: in importtestmodule
    mod = import_path(
/glade/work/pg2328/conda-envs/legoesm-gpu/lib/python3.12/site-packages/_pytest/pathlib.py:596: in import_path
    importlib.import_module(module_name)
/glade/work/pg2328/conda-envs/legoesm-gpu/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
<frozen importlib._bootstrap>:1387: in _gcd_import
    ???
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1331: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:935: in _load_unlocked
    ???
/glade/work/pg2328/conda-envs/legoesm-gpu/lib/python3.12/site-packages/_pytest/assertion/rewrite.py:188: in exec_module
    exec(co, module.__dict__)
tests/land/unit/test_land_screen_temperature.py:15: in <module>
    from legoesm.land.canopy.stability import screen_level_temperature
../../../packages/land/legoesm/land/__init__.py:4: in <module>
    from legoesm.land.config import LandConfig, MultiLayerLandConfig
../../../packages/land/legoesm/land/config.py:17: in <module>
    from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig
../../../packages/land/legoesm/land/surface_scheme/__init__.py:41: in <module>
    from legoesm.land.surface_scheme.simple_seb import (
../../../packages/land/legoesm/land/surface_scheme/simple_seb.py:26: in <module>
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
../../../packages/atmosphere/legoesm/atmosphere/physics/__init__.py:23: in <module>
    from legoesm.atmosphere.physics.combined import (
E     File "/glade/u/home/pg2328/legoesm/packages/atmosphere/legoesm/atmosphere/physics/combined.py", line 805
E       cirrus_aerosol=None) -> Callable:
E                          ^
E   SyntaxError: unmatched ')'
=========================== short test summary info ============================
ERROR tests/land/unit/test_land_screen_temperature.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 5.82s ===============================

exec
/usr/bin/bash -lc "rg -n 'canopy_forward|fluxes\\.values|fluxes\\.items|fluxes_per_col\\.' packages/land tests/land --glob '*.py' | head -65; sed -n '80,130p' tests/conftest.py; rg -n 'python.*site|sys.path' tests/conftest.py; rg -n 'T_2m|fallback' docs/wb/reviews/2026-11-22_land_tas_r1_glm.md | tail -12; sed -n '350,390p' packages/land/legoesm/land/canopy/stability.py" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
tests/land/unit/test_canopy_warm_start.py:34:    canopy_forward,
tests/land/unit/test_canopy_warm_start.py:104:    fl = lambda x: canopy_forward(x, b, _CFG.LE_module, _CFG.stomatal_model,
tests/land/unit/test_canopy_rh_cap.py:90:    from legoesm.land.canopy.solver import canopy_forward
tests/land/unit/test_canopy_rh_cap.py:98:    fwd = lambda x, c: canopy_forward(x, b, c.LE_module, c.stomatal_model, c.le_cap_mode,
tests/land/unit/test_canopy_dew_limit.py:24:from legoesm.land.canopy.solver import (CanopyForcingBundle, canopy_forward,
tests/land/unit/test_canopy_dew_limit.py:44:    f = canopy_forward(x, _B, cfg.LE_module, cfg.stomatal_model, cfg.le_cap_mode,
tests/land/unit/test_canopy_solver_grad.py:142:def test_canopy_forward_reports_gross_and_net():
tests/land/unit/test_canopy_solver_grad.py:145:    ``canopy_forward`` must return BOTH the NET ``An_*`` (which drives the leaf
tests/land/unit/test_canopy_solver_grad.py:153:    from legoesm.land.canopy.solver import canopy_forward
tests/land/unit/test_canopy_solver_grad.py:159:    fluxes = canopy_forward(
tests/land/unit/test_canopy_solver_grad.py:165:        assert k in fluxes, f"canopy_forward dict missing {k}"
tests/land/unit/test_canopy_solver_grad.py:193:    makes THIS test fail (the prior canopy_forward test does not, because it
tests/land/unit/test_canopy_solver_grad.py:215:        CanopyForcingBundle, solve_canopy_closure, canopy_forward)
tests/land/unit/test_canopy_solver_grad.py:303:    d = jax.vmap(lambda x, b: canopy_forward(
packages/land/legoesm/land/canopy/solver.py:283:    # in ``canopy_forward``, so discard it here.
packages/land/legoesm/land/canopy/solver.py:413:def canopy_forward(
packages/land/legoesm/land/canopy/energy_balance.py:324:    caller (``LE_wet`` recomputed in ``canopy_forward``).
packages/land/legoesm/land/canopy/photosynthesis.py:467:        respiration is not double-counted (see ``solver.canopy_forward``).
packages/land/legoesm/land/canopy/photosynthesis.py:469:    The two big-leaf callers (``_canopy_residual``, ``canopy_forward``) unpack
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:46:    CanopyForcingBundle, solve_canopy_closure_diag, canopy_forward,
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:512:        return canopy_forward(xf, bun, cc.LE_module, cc.stomatal_model,
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(0, value)


def _xdist_worker_count() -> int:
    """How many pytest-xdist workers share this machine (1 when not under xdist)."""
    try:
        return max(1, int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")))
    except ValueError:
        return 1


def _effective_cap(n_cpus: int, soft_rlimit: int) -> int:
    """CPUs this process may use so every worker's XLA pool fits the ceiling.

    Budgets ``_RLIMIT_SHARE`` of the thread ceiling across the concurrent xdist
    workers and inverts the measured threads-per-CPU fit.  An explicit
    ``LEGOESM_TEST_CPU_CAP`` wins outright -- if a user names a number, honour
    it rather than second-guessing their machine.
    """
    requested = _requested_cap()
    if requested is not None:
        return requested
    budget = (soft_rlimit * _RLIMIT_SHARE) / _xdist_worker_count()
    allowed = int((budget - _THREADS_BASE) / _THREADS_PER_CPU)
    # Never below 4 (XLA needs a workable pool) nor above the documented default.
    return max(4, min(_DEFAULT_TEST_CPU_CAP, allowed))


def _worker_slice(cpus: list[int], cap: int) -> set[int]:
    """Pick this xdist worker's DISJOINT slice of *cpus*.

    Without this every worker inherits the controller's identical low-numbered
    mask and four XLA clients pile onto the same cores.  ``PYTEST_XDIST_WORKER``
    is ``gw0``, ``gw1``, ...; the slice wraps if the mask is too narrow to give
    everyone a private one.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    index = 0
    if worker.startswith("gw") and worker[2:].isdigit():
        index = int(worker[2:])
    start = (index * cap) % len(cpus)
    picked = [cpus[(start + i) % len(cpus)] for i in range(cap)]
    return set(picked)


def _cap_cpu_affinity_for_thread_rlimit() -> None:
4:2. `model_driver.py:14719-14752` — the non-`_land_none` return and the packed-scatter tuple both append `_t2m` after `_emis`, and the consumer at `model_driver.py:16318` unpacks it in the same slot. Consistent. **Not visible in the diff**: the `_land_none` / `_land_step_none` producer and the multi-rank halo block — if any of those still produce/slice 10 fields, the 11-slot unpack breaks at runtime. The `_held_f > 0.5 → NaN` and `_land_T2m_last` cadence (set only when a land step ran) are coherent with the V2 fallback.
6:4. Tests: oracle is a genuinely independent scalar transcription (Paulson/BH91), covers unstable/stable/neutral, monotonicity between canopy air and reference, exact replacement-and-fallback in `_tas_2m` (fails pre-fix), and gradients are finite/non-zero. Non-vacuous. Not covered: held-column NaN export and halo/restart fallbacks — worth one test.
7:5. No state/flux/coupling mutation anywhere; `T_2m` is export-only in `SurfaceFluxOutput`. Differentiability preserved (jnp.where on isfinite keeps gradients through finite entries).
14:- **V3: A** — it is a diagnostic; first-interval/halo fallback is bounded by the profile error already accepted. Add the payload field only if item 3 forces a format bump anyway.
    """Single Monin-Obukhov fixed-point iteration.

    carry : [z0h, obu, um]
    forcing: [Ta, Tv_atm, q_atm, q_atm, zldis, z0m, dq, dth, ur]
    output : [ustar, tstar, qstar, thvstar, ch, zeta, obu]  (``obu`` = the
             Obukhov length ``ch`` was evaluated with, i.e. the incoming one)
    """
    z0h, obu, um = carry[0], carry[1], carry[2]
    Ta, Tv_atm, q_atm, _q2, zldis, z0m, dq, dth, ur = (
        forcing[0], forcing[1], forcing[2], forcing[3],
        forcing[4], forcing[5], forcing[6], forcing[7], forcing[8],
    )

    ustar = _friction_velocity(zldis, z0m, obu, um)
    ch    = _temperature_humidity_relation(zldis, obu, z0h)
    tstar = ch * dth
    qstar = ch * dq
    thvstar = tstar * (1.0 + _VIRT_T_COEF * q_atm) + _VIRT_T_COEF * Ta * qstar

    zeta  = zldis * constants.kappa_von_karman * constants.g * thvstar / (ustar**2 * Tv_atm)

    zeta_stable = _cap_stable_zeta(zeta, zeta_cap_width)
    um_stable   = jnp.maximum(ur, 0.1)                     # coeff-ok: 0.1 m/s wind floor
    zeta_unstable = jnp.clip(zeta, -100.0, -_ZETA_NEUTRAL_FLOOR)  # coeff-ok: very-unstable bound on ζ
    # Floor the cbrt argument to a POSITIVE value, not 0: cbrt'(0)=inf and the
    # maximum's subgradient is 0 below the clamp, so cbrt(maximum(x, 0)) gives
    # 0*inf = NaN in the reverse-mode gradient whenever x<=0 (the stable regime,
    # where this unstable-branch wc is discarded).  cbrt(1e-12)~1e-4 m/s ~ 0.
    wc_unstable = jnp.cbrt(jnp.maximum(
        -constants.g * ustar * thvstar * _CONV_BDY_HEIGHT / Tv_atm, _MOST_ARG_FLOOR))
    um_unstable = jnp.sqrt(ur**2 + wc_unstable**2)

    is_stable = zeta >= 0.0
    zeta_new = jnp.where(is_stable, zeta_stable, zeta_unstable)
    um_new   = jnp.where(is_stable, um_stable,   um_unstable)
    obu_new  = zldis / zeta_new

    new_carry = jnp.array([z0h, obu_new, um_new])
    output    = jnp.array([ustar, tstar, qstar, thvstar, ch, zeta_new, obu])
    return new_carry, output


exec
/usr/bin/bash -lc "ls packages; ls src/legoesm; sed -n '1,90p' src/legoesm/__init__.py; rg -n 'sys.path|packages/' sitecustomize.py conftest.py pyproject.toml tests/conftest.py 2>/dev/null" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 exited 2 in 0ms:
atmosphere
core
coupler
ice
land
ml
ocean
tools
cli.py
config.py
dycore_factory.py
experiment_registry.py
scaling_preflight.py
supported_matrix.py
sed: can't read src/legoesm/__init__.py: No such file or directory
pyproject.toml:137:# packages/atmosphere/pyproject.toml.
pyproject.toml:149:# (packages/core, dist name legoesm-core) and the root ``legoesm`` is the
pyproject.toml:151:# and has no src/ level (packages/<m>/legoesm/<subpkg>); the root depends on them
pyproject.toml:155:members = ["packages/*"]
pyproject.toml:175:extend-exclude = ["packages/land/legoesm/land/canopy/clm_ml_backend"]
pyproject.toml:189:"packages/core/legoesm/core/fv3_native_sw_core.py" = ["N803", "N806", "N801"]
pyproject.toml:190:"packages/core/legoesm/core/fv3_native_d_sw.py" = ["N803", "N806", "N801"]
pyproject.toml:191:"packages/core/legoesm/core/fv3_native_duo_sw_core.py" = ["N803", "N806", "N801"]
pyproject.toml:192:"packages/core/legoesm/grids/fv3_native_gridstruct.py" = ["N801", "N803", "N806"]
pyproject.toml:196:"packages/core/legoesm/core/fv3_native_pgrad.py" = ["N803", "N806", "N801"]
pyproject.toml:208:"packages/ocean/legoesm/ocean/experiments/bench.py" = ["N803", "N806"]
pyproject.toml:223:"packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb*.py" = ["N802", "N803", "N806", "N815"]
pyproject.toml:227:"packages/atmosphere/legoesm/atmosphere/dynamics/column_les_diagnosis.py" = ["N803", "N806", "N815"]
pyproject.toml:228:"packages/atmosphere/legoesm/atmosphere/dynamics/column_les.py" = ["N803", "N806", "N815"]
pyproject.toml:234:"packages/atmosphere/legoesm/atmosphere/les_suite/registry.py" = ["N815"]
pyproject.toml:235:"packages/atmosphere/legoesm/atmosphere/les_suite/catalog.py" = ["N803"]
pyproject.toml:239:"packages/atmosphere/legoesm/atmosphere/les_suite/intercomparison.py" = ["N803", "N815"]
pyproject.toml:245:"packages/atmosphere/legoesm/atmosphere/les_suite/emit.py" = ["N803"]
pyproject.toml:252:"packages/atmosphere/legoesm/atmosphere/les_suite/scm_runner.py" = ["N803", "N806"]
pyproject.toml:255:"packages/atmosphere/legoesm/atmosphere/_future/arm_obs_score.py" = ["N806"]
pyproject.toml:260:"packages/atmosphere/legoesm/atmosphere/forcing/scm/scm_forcing_io.py" = ["N803", "N806"]
pyproject.toml:261:"packages/atmosphere/legoesm/atmosphere/forcing/scm/sccm_arm.py" = ["N803", "N806"]
pyproject.toml:267:"packages/atmosphere/legoesm/atmosphere/dynamics/column_large_scale_extract.py" = ["N803", "N806"]
pyproject.toml:272:"packages/ml/legoesm/training/compare_reanalysis.py" = ["N803", "N806", "N815"]
pyproject.toml:280:"packages/ml/legoesm/training/bias_metrics.py" = ["N815"]
pyproject.toml:319:# Run-as-a-script sys.path bootstrap (iter 321) must precede the ``scripts.*`` import, so the
pyproject.toml:322:# Same run-as-a-script sys.path bootstrap (REPO_ROOT insert precedes the ``scripts.run``
pyproject.toml:328:"packages/ml/legoesm/training/era5_to_state.py" = ["E402", "N803", "N806", "N815"]
pyproject.toml:358:"packages/coupler/legoesm/coupler/coupled_latlon_band.py" = ["N806", "N815"]
pyproject.toml:367:exclude = ['packages/land/legoesm/land/canopy/clm_ml_backend/']

exec
/usr/bin/bash -lc 'ls packages/core/legoesm/__init__.py; cat packages/core/legoesm/__init__.py' in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 exited 1 in 12ms:
ls: cannot access 'packages/core/legoesm/__init__.py': No such file or directory
cat: packages/core/legoesm/__init__.py: No such file or directory

exec
/usr/bin/bash -lc 'PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH="$PWD/src:$PWD/packages/core:$PWD/packages/atmosphere:$PWD/packages/coupler:$PWD/packages/land:$PWD/packages/ocean:$PWD/packages/ice:$PWD/packages/ml:$PWD/packages/tools" /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -m pytest -p no:cacheprovider tests/land/unit/test_land_screen_temperature.py tests/unit/test_land_column_split.py tests/land/unit/test_canopy_stability.py tests/land/unit/test_canopy_stability_faithful.py -q' in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
exec
/usr/bin/bash -lc "rg -n '"'^def solve_land_on_devices|''^def scatter_cells|''^def split_land_columns'"' packages/coupler/legoesm/driver/model_driver.py; rg -n 'SurfaceFluxOutput|surface_flux' packages/land/legoesm/land --glob '*patch*'; sed -n '1,95p' docs/wb/reviews/2026-11-22_land_tas_r1_glm.md; git status --short" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 2387ms:
254:def split_land_columns(idx, n_proc, proc):
407:def solve_land_on_devices(step, args, is_col, mesh, keep):
packages/land/legoesm/land/surface_scheme/patch_mosaic.py:6:fraction.  The patch fluxes are aggregated to one grid-mean ``SurfaceFluxOutput``
packages/land/legoesm/land/surface_scheme/patch_mosaic.py:37:from legoesm.land.surface_scheme.base import SurfaceFluxOutput
packages/land/legoesm/land/surface_scheme/patch_mosaic.py:321:) -> SurfaceFluxOutput:
packages/land/legoesm/land/surface_scheme/patch_mosaic.py:322:    """Run the ``N``-patch mosaic and return one area-weighted grid ``SurfaceFluxOutput``.
packages/land/legoesm/land/surface_scheme/patch_mosaic.py:382:def _area_weight(out: SurfaceFluxOutput, fracs: jnp.ndarray) -> SurfaceFluxOutput:
packages/land/legoesm/land/surface_scheme/patch_mosaic.py:383:    """Area-weight-reduce a patch-axis ``SurfaceFluxOutput`` to one grid column.
**Findings (ranked)**

1. `two_leaf_canopy.py:733` — `screen_level_temperature(Ta, Tc_cvg, z_ref - displa, fluxes_per_col["zeta"], z0m)`: zldis and z0m(=z0h) match CLM5 (`zldis = zrefw - displa`, CLM uses z0m for heat), and dth/thm frame matches `t_ref2m = thm + temp1*dth*(1/temp12m − 1/temp1)`. Good. **But** `obu = zldis / zeta` reconstructs the Obukhov length from the *capped/smoothed* zeta (module caps stable zeta at `_ZETA_MAX_STABLE` with a smooth cap), so in strongly stable columns the diagnostic uses a different obu than the flux solve. Prefer passing obu directly from the solve. ~sub-0.1 K effect; diagnostic only.
2. `model_driver.py:14719-14752` — the non-`_land_none` return and the packed-scatter tuple both append `_t2m` after `_emis`, and the consumer at `model_driver.py:16318` unpacks it in the same slot. Consistent. **Not visible in the diff**: the `_land_none` / `_land_step_none` producer and the multi-rank halo block — if any of those still produce/slice 10 fields, the 11-slot unpack breaks at runtime. The `_held_f > 0.5 → NaN` and `_land_T2m_last` cadence (set only when a land step ran) are coherent with the V2 fallback.
3. `_LAND_STEP_N_CELL = 11` — if the checkpoint land-window payload is sized from this constant, old restarts have a layout mismatch; needs a guard or a documented version bump (ties to V3).
4. Tests: oracle is a genuinely independent scalar transcription (Paulson/BH91), covers unstable/stable/neutral, monotonicity between canopy air and reference, exact replacement-and-fallback in `_tas_2m` (fails pre-fix), and gradients are finite/non-zero. Non-vacuous. Not covered: held-column NaN export and halo/restart fallbacks — worth one test.
5. No state/flux/coupling mutation anywhere; `T_2m` is export-only in `SurfaceFluxOutput`. Differentiability preserved (jnp.where on isfinite keeps gradients through finite entries).

**Verdict: CLEAN** (subject to items 2/3 being confirmed in the unshown `_land_none`/halo/checkpoint paths; nothing in the diff itself is wrong).

**Votes**
- **V1: A** — CLM's `thm` already carries the lapse correction (thm = th + g/cp·z_forc), so verbatim `t_ref2m` is already a temperature; an extra −0.02 K would double-correct.
- **V2: A** — keeps continuity with the pre-fix behaviour and the sea legs; a jump to lowest-level T is a larger, discontinuous change for held/halo columns.
- **V3: A** — it is a diagnostic; first-interval/halo fallback is bounded by the profile error already accepted. Add the payload field only if item 3 forces a format bump anyway.
 M packages/coupler/legoesm/driver/model_driver.py
 M packages/land/legoesm/land/canopy/solver.py
 M packages/land/legoesm/land/canopy/stability.py
 M packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py
 M tests/land/unit/test_land_screen_temperature.py
 M tests/unit/test_land_column_split.py
?? docs/wb/reviews/2026-11-22_land_tas_r1_codex.md
?? docs/wb/reviews/2026-11-22_land_tas_r1_glm.md
?? docs/wb/reviews/2026-11-22_land_tas_r1_prompt.md
?? docs/wb/reviews/2026-11-22_land_tas_r2_codex.md
?? docs/wb/reviews/2026-11-22_land_tas_r2_glm.md
?? docs/wb/reviews/2026-11-22_land_tas_r2_prompt.md

codex
The tuple and NaN paths look consistent so far. The test environment initially imported a different checkout; I’ve corrected the import paths and restarted the focused suite on this worktree.
exec
/usr/bin/bash -lc "sed -n '382,435p' packages/land/legoesm/land/surface_scheme/patch_mosaic.py; sed -n '407,480p' packages/coupler/legoesm/driver/model_driver.py; sed -n '367,391p' pyproject.toml; ps -o pid,etime,pcpu,stat,args -u pg2328 | rg 'pytest.*land_screen'" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 268ms:
def _area_weight(out: SurfaceFluxOutput, fracs: jnp.ndarray) -> SurfaceFluxOutput:
    """Area-weight-reduce a patch-axis ``SurfaceFluxOutput`` to one grid column.

    ``x_grid = sum_i frac_i * x_i`` over the leading (patch) axis, keeping the
    ``ncol=1`` dimension.  Exact for per-area fluxes; the area-mean for intensive
    state (albedo, ...).  Two fields are reduced non-linearly so the output stays
    self-consistent:

    * ``T_surface`` is the emission-equivalent (radiometric) temperature, so it is
      reconstructed to keep ``eps_grid * sigma * T_grid**4 == sum_i frac_i * eps_i *
      sigma * T_i**4`` (matching the linearly-summed ``lw_up``): with
      ``eps_grid = sum_i frac_i * eps_i``,
      ``T_grid = (sum_i frac_i * eps_i * T_i**4 / eps_grid)**0.25``.  A linear mean
      of ``T_surface`` would emit the wrong grid longwave.
    * ``n_iters`` is a solver iteration COUNT, not a flux — reduced by ``max`` (the
      worst-case patch), keeping it an integer count so a single unit patch is
      identical to the direct canopy call.
    * ``converged`` is a per-column flag — reduced by ``all``: a column is converged
      only when every patch's root solve converged (an area-weighted boolean would
      report a column with a failed patch as e.g. 0.6 "converged").

    ``None`` fields pass through.
    """
    def w(x):
        weights = fracs.reshape((-1,) + (1,) * (jnp.ndim(x) - 1))
        return jnp.sum(weights * x, axis=0, keepdims=True)
    agg = jax.tree_util.tree_map(w, out)
    if out.T_surface is not None and out.emissivity is not None:
        eps_grid = w(out.emissivity)
        t4 = w(out.emissivity * out.T_surface ** 4)
        agg = agg._replace(T_surface=(t4 / eps_grid) ** 0.25)
    if out.n_iters is not None:
        agg = agg._replace(n_iters=jnp.max(out.n_iters, axis=0, keepdims=True))
    if out.converged is not None:
        agg = agg._replace(converged=jnp.all(out.converged, axis=0, keepdims=True))
    return agg
def solve_land_on_devices(step, args, is_col, mesh, keep):
    """``step(*args)`` (the land step on packed slots, leading axis = slot)
    run one equal shard per device of ``mesh`` (split over all its axes, so
    the duo's 3-D window mesh works too) under a shard_map,
    then cut to the real slots ``keep``.  ``is_col`` is ``args``' tree of
    booleans marking the per-column array leaves (split over the devices);
    every other leaf is shared.  Every OUTPUT leaf must be per column (its
    leading axis follows the column count, traced at two counts), else
    ValueError.
    The step's whole-run
    counters would be per shard there, so they are recounted from their
    masks over the real slots.  Returns ``(state, response, carbon,
    surface)`` like the step."""
    from jax.sharding import PartitionSpec as P
    from legoesm.parallel.shard_map_compat import shard_map

    n_dev = int(np.prod(mesh.devices.shape))
    names = mesh.axis_names
    axis = names[0] if len(names) == 1 else tuple(names)
    leaves, tdef = jax.tree_util.tree_flatten(args)
    cols = jax.tree_util.tree_leaves(is_col)
    if len(cols) != len(leaves):
        raise ValueError("is_col does not match the land step arguments")
    is_arr = [hasattr(x, "shape") and hasattr(x, "dtype") for x in leaves]
    arrs = [x for x, a in zip(leaves, is_arr) if a]
    arr_col = [c for c, a in zip(cols, is_arr) if a]
    n_slot = {x.shape[0] for x, c in zip(arrs, arr_col) if c}
    if len(n_slot) != 1 or next(iter(n_slot)) % n_dev:
        raise ValueError(f"per-column land leaves have leading sizes {n_slot}"
                         f"; need one size divisible by {n_dev} devices")
    n_slot = n_slot.pop()

    def body(*a):
        it = iter(a)
        o = step(*tdef.unflatten([next(it) if t else x
                                  for x, t in zip(leaves, is_arr)]))
        return o[:3] + (o[3]._replace(n_held=None, n_fallback=None,
                                      n_fallback_rejected=None),)

    def shapes(n):
        return [jax.ShapeDtypeStruct(((n,) + tuple(x.shape[1:])) if c
                                     else x.shape, x.dtype)
                for x, c in zip(arrs, arr_col)]
    # an output is per column when its LEADING axis follows the column count
    a_out = jax.eval_shape(body, *shapes(n_slot))
    b_out = jax.eval_shape(body, *shapes(n_slot + 1))
    shared = [jax.tree_util.keystr(k) for (k, a), b in zip(
        jax.tree_util.tree_flatten_with_path(a_out)[0],
        jax.tree_util.tree_leaves(b_out))
        if not a.shape or b.shape != (a.shape[0] + 1,) + a.shape[1:]]
    if shared:
        raise ValueError(
            f"land step outputs {shared} are not per column; the device "
            "split can only return per-column outputs (and the three "
            "fallback/held counters, recounted here)")
    # check_vma off: the canopy scan carries start replicated.  Safe only
    # because every output is per column (checked above).
    out = shard_map(
        body, mesh=mesh,
        in_specs=tuple(P(axis) if c else P() for c in arr_col),
        out_specs=jax.tree_util.tree_map(lambda _: P(axis), a_out),
        check_vma=False)(*arrs)
    keep = jnp.asarray(keep)
    out = jax.tree_util.tree_map(lambda o: o[keep], out)
    state, resp, carbon, sfc = out

    def count(m):
        return None if m is None else jnp.sum(
            jnp.asarray(m).astype(jnp.int32))
    return state, resp, carbon, sfc._replace(
        n_held=count(sfc.held), n_fallback=count(sfc.fallback),
        n_fallback_rejected=count(sfc.fallback_rejected))


exclude = ['packages/land/legoesm/land/canopy/clm_ml_backend/']

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short -m 'not slow'"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "lint: static checks run through pytest because CI is disabled; skip in a fast inner loop with -m 'not lint'",
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
2008694       02:03  2.1 Sl   codex exec --dangerously-bypass-approvals-and-sandbox -C /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193 ADVERSARIAL REVIEW ROUND 2 (read-only; do not edit files). Worktree: /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193, branch fix/land-tas-t-ref2m (base 8750d0005d; HEAD is a WIP commit of round 1, the working tree holds round-2 fixes; the diff below is the full change vs base). CLM5: /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys.  Round 1 (codex NOT CLEAN; GLM CLEAN) findings and what was done: 1. Screen profile used the post-update zeta, not the Obukhov length the heat flux used -> monin_obukhov_stability gained `return_flux_obu=False` (default unchanged, 5-tuple); _stability_step output gained the incoming obu (last[6]); canopy_forward requests it and returns `obu`; screen_level_temperature now takes `obu` (not zeta), so temp1 is exactly the flux ch (new test: ch(obu)*ustar == 1/rah to 1e-12, with a check that zl/obu differs from the returned zeta in the unstable case). 2. Zero-filled cells could publish 0 K -> packed scatter of T2m now NaN-fills off the land columns; _land_step_none returns NaN for the T2m slot (other slots still zeros). Consumer (_tas_2m) keeps the existing land profile where T2m is non-finite. T2m is NOT halo-exchanged and NOT checkpointed (vote V3: author A, GLM A, codex B -> A). 3. Tests: oracle extended to CLM5 very-unstable heat regime (zeta -2, -20); canopy bracket now strict; gather fixture gives T2m a distinct value (f + 1.0). Votes recorded: V1 A/A/A (publish CLM t_ref2m verbatim, no extra 2 m lapse step); V2 A/A/A (keep the existing land profile where no land T2m).  Check that the fixes are correct and complete, that nothing else changed behaviour (flux solve outputs identical: rah/raw/ustar/zeta unchanged), and the tuple/NaN handling in every producer/consumer. Report <=300 words: findings ranked (file:line), verdict CLEAN or NOT CLEAN.  DIFF vs base: diff --git a/packages/coupler/legoesm/driver/diagnostics.py b/packages/coupler/legoesm/driver/diagnostics.py index 8003d760c3..b06f536cc3 100644 --- a/packages/coupler/legoesm/driver/diagnostics.py +++ b/packages/coupler/legoesm/driver/diagnostics.py @@ -806,7 +806,8 @@ class DiagnosticCollector:          return f_lo + alpha * (f_hi - f_lo)        def _tas_2m(self, state, q_v, sst, sic, T_ice, u_low=None, v_low=None, -                *, T_land=None, q_land=None, land_fraction=None): +                *, T_land=None, q_land=None, land_fraction=None, +                T2m_land=None):          """2 m air temperature for CMIP ``tas`` from the MOST surface-layer          similarity profile (interpolate the lowest model level down to 2 m).          Returns the lowest-level T when the surface inputs (SST / sigma) are @@ -821,6 +822,14 @@ class DiagnosticCollector:          land branch uses the same MOST profile anchored on the land skin; do          not read the result as a sub-canopy screen temperature.   +        ``T2m_land`` (keyword-only, optional) is the land scheme's OWN 2 m air +        temperature (two-leaf canopy: CLM5 ``t_ref2m`` from its roughness, +        displacement, Obukhov length and canopy-air node).  Where it is finite +        it REPLACES the profile above on the land leg, which runs the ocean +        bulk law (Charnock roughness, no displacement) over the land skin. +        Non-finite entries (held columns, schemes that solve none) keep that +        profile. +          ``u_low`` / ``v_low`` override the lowest-level winds (lets the MPAS          path pass the reconstructed CELL winds from          ``reconstruct_cell_velocity`` instead of the edge-normal ``state.u``, @@ -901,6 +910,9 @@ class DiagnosticCollector:          q_sfc_land = (saturation_mixing_ratio(T_land, p_s) if q_land is None                        else q_land)          T_2m_land = _profile_to_2m(T_land, q_sfc_land, ocean=False) +        if T2m_land is not None: +            T2m_land = jnp.asarray(T2m_land, dtype=T_2m_land.dtype) +            T_2m_land = jnp.where(jnp.isfinite(T2m_land), T2m_land, T_2m_land)          T_2m_sea = _profile_to_2m(T_sfc_sea, q_sfc_sea, ocean=True)          f_land = jnp.clip(jnp.asarray(land_fraction, dtype=T_2m_land.dtype),                            0.0, 1.0) diff --git a/packages/coupler/legoesm/driver/model_driver.py b/packages/coupler/legoesm/driver/model_driver.py index 656bda453e..b7a265ac53 100644 --- a/packages/coupler/legoesm/driver/model_driver.py +++ b/packages/coupler/legoesm/driver/model_driver.py @@ -297,7 +297,7 @@ def land_radiating_surface(T_sea, f_land, E_sea, T_rad_land, eps_land):    # Per-cell fields of the MPAS land step tuple, after the state: T_sfc, albedo,  # q_surface, shflx, lhflx, surface_mass_flux, stress, held, T_rad, emissivity. -_LAND_STEP_N_CELL = 10 +_LAND_STEP_N_CELL = 11      def gather_land_step(run, ncol, padded, proc, allgather): @@ -10930,6 +10930,7 @@ class ModelDriver:                  # published land tas is not the model's land at all.                  _tas_land_T = getattr(self, "_land_T_skin_last", None)                  _tas_land_q = getattr(self, "_land_qsfc_last", None) +                _tas_land_T2m = getattr(self, "_land_T2m_last", None)                  _tas_f_land = getattr(self, "_f_land", None)                  _tas_kw = {}                  if _tas_land_T is not None and _tas_f_land is not None: @@ -10937,7 +10938,9 @@ class ModelDriver:                          T_land=jnp.asarray(_tas_land_T).reshape(-1),                          q_land=(None if _tas_land_q is None                                  else jnp.asarray(_tas_land_q).reshape(-1)), -                        land_fraction=jnp.asarray(_tas_f_land).reshape(-1)) +                        land_fraction=jnp.asarray(_tas_f_land).reshape(-1), +                        T2m_land=(None if _tas_land_T2m is None +                                  else jnp.asarray(_tas_land_T2m).reshape(-1)))                  from legoesm.forcing.surface_utils import blend_surface_temperature                  ts = blend_surface_temperature(_sst, _sic, _tas_ice)                  if _tas_land_T is not None and _tas_f_land is not None: @@ -14719,6 +14722,13 @@ class ModelDriver:                  _emis = jnp.broadcast_to(                      jnp.asarray(resp.emissivity, resp.T_rad.dtype),                      resp.T_rad.shape) +                # The scheme's own 2 m air temperature (CLM5 t_ref2m) for the +                # CMOR ``tas`` land leg; NaN where the scheme solves none or the +                # column was held (no solved surface layer this call). +                _t2m = (jnp.full_like(resp.T_rad, jnp.nan) +                        if getattr(_sfc, "T_2m", None) is None +                        else jnp.asarray(_sfc.T_2m, resp.T_rad.dtype)) +                _t2m = jnp.where(_held_f > 0.5, jnp.nan, _t2m)                  if _land_pack_on:                      # Scatter the advanced columns back into the full-grid                      # state (ocean columns keep their frozen init values, @@ -14735,11 +14745,15 @@ class ModelDriver:                                      resp.T_sfc, resp.albedo, resp.q_surface,                                      resp.shflx, resp.lhflx, resp.surface_mass_flux,                                      _taumag, _held_f, resp.T_rad, _emis)) +                            # 2 m T: NaN (= no land screen value) off the +                            # land columns, never a valid-looking 0 K +                            + (jnp.full((_land_ncol_full,), jnp.nan, _t2m.dtype) +                               .at[_land_out_idx].set(_t2m.reshape(-1)),)                              + (_n_held, _n_held_land, _n_fb_land,                                 _n_fb_rej, _n_unconv_land, _n_iter_max))                  return (new_state, resp.T_sfc, resp.albedo, resp.q_surface,                          resp.shflx, resp.lhflx, resp.surface_mass_flux, -                        _taumag, _held_f, resp.T_rad, _emis, +                        _taumag, _held_f, resp.T_rad, _emis, _t2m,                          _n_held, _n_held_land, _n_fb_land, _n_fb_rej,                          _n_unconv_land, _n_iter_max)                if not _land_none: @@ -14752,8 +14766,12 @@ class ModelDriver:                    # the same message on every rank.                    out = jax.eval_shape(                        _land_step, land_state, a2s, doy, lai_doy, lai_year) +                  # (the 2 m T, last per-cell field, is NaN = none solved)                    return (land_state,) + tuple( -                      jnp.zeros(o.shape, o.dtype) for o in out[1:]) +                      jnp.full(o.shape, jnp.nan, o.dtype) +                      if i == _LAND_STEP_N_CELL - 1 +                      else jnp.zeros(o.shape, o.dtype) +                      for i, o in enumerate(out[1:]))                return _land_step_none                _land_step_fn = _make_land_step(DT_LAND) @@ -16307,7 +16325,7 @@ class ModelDriver:                       _land_albedo_cells, _land_qsfc_step,                       _land_shflx_step, _land_lhflx_step, _land_evap_step,                       _land_taumag_step, _land_held_step, -                     _land_Trad_cells, _land_emis_cells, +                     _land_Trad_cells, _land_emis_cells, _land_T2m_step,                       _land_n_held_step, _land_n_held_land_step,                       _land_n_fb_step, _land_n_fb_rej_step,                       _land_n_unconv_step, _land_n_iter_max) = _land_fn( @@ -16332,6 +16350,7 @@ class ModelDriver:                      # land and publishes a land temperature the model never had.                      self._land_T_skin_last = _land_T_skin                      self._land_qsfc_last = _land_qsfc_step +                    self._land_T2m_last = _land_T2m_step                      # Accumulate ON DEVICE and read at the same cadence the                      # other post-step warnings use: reading it every step                      # would stall the accelerator once per step for a number diff --git a/packages/land/legoesm/land/canopy/solver.py b/packages/land/legoesm/land/canopy/solver.py index 6e3d520689..119dae6c87 100644 --- a/packages/land/legoesm/land/canopy/solver.py +++ b/packages/land/legoesm/land/canopy/solver.py @@ -435,9 +435,10 @@ def canopy_forward(      Ts = b.Ts_bc      zldis = b.z0 - b.displa   -    ustar, rah_above, raw_above, uav, zeta = monin_obukhov_stability( +    ustar, rah_above, raw_above, uav, zeta, obu = monin_obukhov_stability(          b.ur, b.Ta, b.Tv_atm, Tc, b.q_atm, q_c, zldis, b.z0m, -        n_iters=most_n_iters, zeta_cap_width=zeta_cap_width) +        n_iters=most_n_iters, zeta_cap_width=zeta_cap_width, +        return_flux_obu=True)        Rb_Sun, Rb_Sh = compute_boundary_layer_resistance(uav, b.LAI, b.fSun, b.cv, b.d_leaf)      rah_below, raw_below = compute_below_canopy_resistance(uav, b.CI, b.LAI) @@ -549,7 +550,7 @@ def canopy_forward(          G=G, Ls=Ls, Lcanopy_up=Lcanopy_up, gap_LW=gap_LW, LW_out=LW_out,          eps_col=eps_col, LW_emit=LW_emit,          gs_Sun=gs_Sun, gs_Sh=gs_Sh, -        ustar=ustar, zeta=zeta, +        ustar=ustar, zeta=zeta, obu=obu,          rah_above=rah_above, rah_below=rah_below,          Rb_Sun=Rb_Sun, Rb_Sh=Rb_Sh,          ALW_Sun=ALW_Sun, ALW_Sh=ALW_Sh, ALW_Soil=ALW_Soil, diff --git a/packages/land/legoesm/land/canopy/stability.py b/packages/land/legoesm/land/canopy/stability.py index b5efb6366a..2db8bc0fa8 100644 --- a/packages/land/legoesm/land/canopy/stability.py +++ b/packages/land/legoesm/land/canopy/stability.py @@ -92,6 +92,7 @@ _ZETA_MAX_STABLE = 0.5  # stability forms are continuous through zeta = 0 (user 2026-09-25).  _ZETA_NEUTRAL_FLOOR = 1.0e-6  _CONV_BDY_HEIGHT = 1000.0  # convective boundary layer height [m] +_SCREEN_HEIGHT_M = 2.0     # CMIP tas / CLM5 t_ref2m height above d + z0h [m]  _Z0MG_BARE = 0.01          # bare-soil momentum roughness length [m]    # --- CLM5 Monin-Obukhov similarity-theory (MOST) constants --- @@ -286,6 +287,29 @@ def _temperature_humidity_relation(zldis: jax.Array, obu: jax.Array,      return ch     +def screen_level_temperature(theta_ref: jax.Array, T_canopy_air: jax.Array, +                             zldis: jax.Array, obu: jax.Array, +                             z0h: jax.Array, +                             z_screen: float = _SCREEN_HEIGHT_M) -> jax.Array: +    """CLM5 ``t_ref2m``: air temperature ``z_screen`` above the displacement +    height plus ``z0h``, interpolated between the reference-height potential +    temperature ``theta_ref`` and the canopy-air (aerodynamic surface) +    temperature with this module's heat similarity relation at the Obukhov +    length ``obu`` the solve's heat flux was evaluated with +    (CanopyFluxesMod.F90:1182, FrictionVelocityMod.F90:636-652):: + +        t_ref2m = thm + temp1 * dth * (1/temp12m - 1/temp1),  dth = thm - taf + +    ``temp1`` and ``temp12m`` are evaluated at the SAME ``obu`` (as CLM does +    within one FrictionVelocity call), so ``temp1`` is the flux's own ``ch``.  Stable side: Beljaars & Holtslag (1991), +    as in the solve (module docstring).  Output is in the same frame as +    ``theta_ref`` (surface-referenced potential temperature, CLM ``thm``).""" +    temp1 = _temperature_humidity_relation(zldis, obu, z0h) +    temp12m = _temperature_humidity_relation(z_screen + z0h, obu, z0h) +    dth = theta_ref - T_canopy_air +    return theta_ref + temp1 * dth * (1.0 / temp12m - 1.0 / temp1) + +  def _cap_stable_zeta(zeta: jax.Array, width: float) -> jax.Array:      """Smooth cap at ``_ZETA_MAX_STABLE`` (no kink), then the neutral floor.   @@ -327,7 +351,8 @@ def _stability_step(carry: jax.Array, _xs: None,        carry : [z0h, obu, um]      forcing: [Ta, Tv_atm, q_atm, q_atm, zldis, z0m, dq, dth, ur] -    output : [ustar, tstar, qstar, thvstar, ch, zeta] +    output : [ustar, tstar, qstar, thvstar, ch, zeta, obu]  (``obu`` = the +             Obukhov length ``ch`` was evaluated with, i.e. the incoming one)      """      z0h, obu, um = carry[0], carry[1], carry[2]      Ta, Tv_atm, q_atm, _q2, zldis, z0m, dq, dth, ur = ( @@ -360,7 +385,7 @@ def _stability_step(carry: jax.Array, _xs: None,      obu_new  = zldis / zeta_new        new_carry = jnp.array([z0h, obu_new, um_new]) -    output    = jnp.array([ustar, tstar, qstar, thvstar, ch, zeta_new]) +    output    = jnp.array([ustar, tstar, qstar, thvstar, ch, zeta_new, obu])      return new_carry, output     @@ -376,7 +401,8 @@ def monin_obukhov_stability(      n_iters: int = 5,      *,      zeta_cap_width: float, -) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]: +    return_flux_obu: bool = False, +) -> tuple[jax.Array, ...]:      """Above-canopy MOST iteration via jax.lax.scan.        Parameters @@ -400,6 +426,9 @@ def monin_obukhov_stability(      raw      : aerodynamic resistance to water vapour [s/m]      uav      : mean wind speed within canopy (used for Rb) [m/s]      zeta     : stability parameter [-] +    obu      : (only with ``return_flux_obu``) the Obukhov length [m] the +               returned ``rah``/``raw`` were evaluated with (the last +               iteration's INCOMING one; ``zeta`` is the updated stability)      """      # Heat roughness length.  CLM5 vegetation uses kB^-1 = 0 (z0h = z0m); the      # earlier BESS default kB^-1 = 2 (z0h = z0m / e^2) over-suppressed heat @@ -443,6 +472,8 @@ def monin_obukhov_stability(      ram      = jnp.maximum(um_final / ustar**2, 1e-9)      uav      = jnp.sqrt(um_final / ram)   +    if return_flux_obu: +        return ustar, rah, raw, uav, zeta, last[6]      return ustar, rah, raw, uav, zeta     diff --git a/packages/land/legoesm/land/surface_scheme/base.py b/packages/land/legoesm/land/surface_scheme/base.py index 3a98280647..b7f7dcdc78 100644 --- a/packages/land/legoesm/land/surface_scheme/base.py +++ b/packages/land/legoesm/land/surface_scheme/base.py @@ -88,6 +88,10 @@ class SurfaceFluxOutput(NamedTuple):      # ``T_sfc`` for the atmosphere's sensible-heat coupling.  Distinct from the      # LW-derived radiometric ``T_surface``.  None for SimpleSEB.      T_canopy_air: jax.Array | None = None +    # CLM5 ``t_ref2m``: 2 m air temperature from the scheme's OWN surface layer +    # (roughness, displacement, Obukhov length, canopy-air node) [K].  The CMIP +    # ``tas`` land leg.  None for schemes that do not solve one. +    T_2m: jax.Array | None = None      gs_Sun: jax.Array | None = None        # sunlit stomatal conductance [m/s]      gs_Sh: jax.Array | None = None         # shaded stomatal conductance [m/s]      n_iters: jax.Array | None = None       # canopy Newton iteration count diff --git a/packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py b/packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py index 3aa3908df3..0986be2b26 100644 --- a/packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py +++ b/packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py @@ -40,7 +40,7 @@ from legoesm.land.canopy.radiative_transfer import (  )  from legoesm.land.canopy.sif import two_leaf_canopy_sif  from legoesm.land.canopy.stability import ( -    compute_aerodynamics, sat_specific_humidity, +    compute_aerodynamics, sat_specific_humidity, screen_level_temperature,  )  from legoesm.land.canopy.solver import (      CanopyForcingBundle, solve_canopy_closure_diag, canopy_forward, @@ -730,6 +730,8 @@ def compute_two_leaf_canopy_fluxes(          Tf_Sun=Tf_Sun,          Tf_Sh=Tf_Sh,          T_canopy_air=Tc_cvg, +        T_2m=screen_level_temperature(Ta, Tc_cvg, z_ref - displa, +                                      fluxes_per_col["obu"], z0m),          gs_Sun=gs_Sun,          gs_Sh=gs_Sh,          n_iters=n_iters, diff --git a/tests/land/unit/test_land_screen_temperature.py b/tests/land/unit/test_land_screen_temperature.py new file mode 100644 index 0000000000..1ff6a35ac4 --- /dev/null +++ b/tests/land/unit/test_land_screen_temperature.py @@ -0,0 +1,157 @@ +"""Land 2 m air temperature (CLM5 ``t_ref2m``) from the two-leaf canopy's own +surface layer, and its use as the CMIP ``tas`` land leg. + +The profile oracle below is an INDEPENDENT scalar transcription of the +similarity integral (Paulson 1970 unstable psi_h, Beljaars & Holtslag 1991 +stable psi_h), not a call into the code under test.""" +import math + +import jax +import jax.numpy as jnp +import numpy as np +import pytest + +from legoesm import constants +from legoesm.land.canopy.stability import screen_level_temperature + +pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64, +                                reason="needs JAX_ENABLE_X64=1") + +K = constants.kappa_von_karman + + +def _psi_h(zeta): +    if zeta < 0.0: +        x = (1.0 - 16.0 * zeta) ** 0.25 +        return 2.0 * math.log((1.0 + x * x) / 2.0) +    a, b, c, d = 1.0, 0.667, 5.0, 0.35   # BH91 as published (b rounded) +    return (-(1.0 + 2.0 * a * zeta / 3.0) ** 1.5 +            - b * (zeta - c / d) * math.exp(-d * zeta) - b * c / d + 1.0) + + +ZETAT = 0.465   # CLM5 heat free-convection transition (FrictionVelocityMod) + + +def _F(z, L, z0h): +    """Integrated heat similarity function, CLM5 FrictionVelocityMod regimes.""" +    if z / L < -ZETAT: +        return (math.log(-ZETAT * L / z0h) - _psi_h(-ZETAT) + _psi_h(z0h / L) +                + 0.8 * (ZETAT ** (-1.0 / 3.0) - (-z / L) ** (-1.0 / 3.0))) +    return math.log(z / z0h) - _psi_h(z / L) + _psi_h(z0h / L) + + +def _oracle(theta, Tc, zl, zeta, z0h, zs=2.0): +    L = zl / zeta +    return Tc + (theta - Tc) * _F(zs + z0h, L, z0h) / _F(zl, L, z0h) + + +def _screen(theta, Tc, zl, zeta, z0h): +    return float(screen_level_temperature( +        jnp.asarray(theta), jnp.asarray(Tc), jnp.asarray(zl), +        jnp.asarray(zl / zeta), jnp.asarray(z0h))) + + +@pytest.mark.parametrize("zeta", [-20.0, -2.0, -0.3, -0.05, 0.05, 0.3]) +@pytest.mark.parametrize("z0h", [0.05, 1.5]) +def test_profile_matches_independent_most(zeta, z0h): +    theta, Tc, zl = 270.0, 262.0, 55.0 +    got = _screen(theta, Tc, zl, zeta, z0h) +    np.testing.assert_allclose(got, _oracle(theta, Tc, zl, zeta, z0h), +                               rtol=0.0, atol=1e-9) +    assert min(theta, Tc) <= got <= max(theta, Tc) + + +def test_neutral_limit_is_log_law(): +    theta, Tc, zl, z0h = 285.0, 289.0, 60.0, 0.8 +    got = _screen(theta, Tc, zl, 1e-6, z0h) +    log_law = Tc + (theta - Tc) * math.log((2.0 + z0h) / z0h) / math.log(zl / z0h) +    np.testing.assert_allclose(got, log_law, rtol=0.0, atol=1e-5) + + +def test_reference_at_screen_height_returns_reference(): +    z0h = 0.3 +    got = _screen(271.0, 260.0, 2.0 + z0h, 0.2, z0h) +    np.testing.assert_allclose(got, 271.0, rtol=0.0, atol=1e-12) + + +def test_gradient_finite_and_nonzero(): +    f = lambda th, tc, zeta: screen_level_temperature( +        th, tc, jnp.asarray(50.0), 50.0 / zeta, jnp.asarray(0.5)) +    for zeta in (-0.3, 0.3): +        g = jax.grad(f, argnums=(0, 1, 2))(jnp.asarray(270.0), jnp.asarray(262.0), +                                          jnp.asarray(zeta)) +        assert all(np.isfinite(float(x)) and abs(float(x)) > 1e-8 for x in g), g + + +@pytest.mark.parametrize("dT", [-6.0, 4.0]) +def test_screen_profile_uses_the_flux_obukhov_length(dT): +    """temp1 of the screen profile IS the flux solve's heat coefficient: +    1 / (rah * ustar) at the Obukhov length the solve returns for its flux +    (not the post-update stability; few iterations so the two differ).""" +    from legoesm.land.canopy.stability import ( +        _temperature_humidity_relation, monin_obukhov_stability) +    zl, z0m, Ta = 45.0, 0.9, 275.0 +    ustar, rah, _raw, _uav, zeta, obu = monin_obukhov_stability( +        jnp.asarray(3.0), jnp.asarray(Ta), jnp.asarray(Ta), jnp.asarray(Ta + dT), +        jnp.asarray(3e-3), jnp.asarray(3e-3), jnp.asarray(zl), jnp.asarray(z0m), +        n_iters=2, zeta_cap_width=0.05, return_flux_obu=True) +    ch = float(_temperature_humidity_relation(jnp.asarray(zl), obu, jnp.asarray(z0m))) +    np.testing.assert_allclose(ch * float(ustar), 1.0 / float(rah), rtol=1e-12) +    if dT > 0:   # unstable: the flux and the post-update stability differ +        assert abs(float(zl / obu) - float(zeta)) > 1e-6 + + +def _two_leaf(sw, lw, T_soil): +    from legoesm.core.coupling_fields import AtmToSurface +    from legoesm.land.config import MultiLayerLandConfig +    from legoesm.land.surface_scheme import TwoLeafCanopyConfig +    from legoesm.land.surface_scheme.two_leaf_canopy import ( +        compute_two_leaf_canopy_fluxes) +    f = lambda v: jnp.full(1, v) +    forcing = AtmToSurface( +        sw_down=f(sw), lw_down=f(lw), precip_total=f(0.), precip_snow=f(0.), +        T_lowest=f(270.), q_lowest=f(.002), u_lowest=f(4.), v_lowest=f(0.), +        p_lowest=f(99300.), p_surface=f(100000.), rho_lowest=f(1.28), +        cos_zenith=f(.7 if sw > 0 else 0.), co2_ppmv=f(412.), +        has_radiation=f(1.), has_precipitation=f(1.), z_lowest=f(60.)) +    return compute_two_leaf_canopy_fluxes( +        T_soil_top=f(T_soil), forcing=forcing, +        canopy_config=TwoLeafCanopyConfig(max_iters=30), +        land_config=MultiLayerLandConfig(), canopy_params=None, +        w_frac_rz=f(.5), wind_speed=f(4.), wind_dir_x=jnp.ones(1), +        wind_dir_y=jnp.zeros(1), +        soil_thermal_fn=lambda G, dt: jnp.full(1, T_soil), +        TgC_override=forcing.T_lowest - constants.T_freeze, dt=600.) + + +@pytest.mark.parametrize("sw,lw,T_soil", [(0., 200., 262.), (700., 300., 285.)]) +def test_two_leaf_exports_screen_temperature_between_air_and_canopy(sw, lw, T_soil): +    out = _two_leaf(sw, lw, T_soil) +    theta = 270.0 + constants.g / constants.c_pd * 60.0 +    t2, tc = float(out.T_2m[0]), float(out.T_canopy_air[0]) +    assert np.isfinite(t2) +    assert abs(theta - tc) > 1.0 +    assert min(theta, tc) + 1e-3 < t2 < max(theta, tc) - 1e-3, (t2, tc, theta) + + +def test_tas_land_leg_takes_the_land_screen_temperature(): +    """Fails on the pre-fix code (no T2m_land): finite entries replace the +    ocean-law land profile exactly, non-finite ones keep it.""" +    from types import SimpleNamespace as NS +    from legoesm.driver.diagnostics import DiagnosticCollector +    from legoesm.grids.vertical import create_sigma_coordinate +    nlev = 20 +    coord = create_sigma_coordinate(nlev) +    dc = DiagnosticCollector(nlev=nlev, sigma_full=coord.sigma_full, +                             dsigma=coord.dsigma, vcoord=coord) +    state = NS(T=NS(data=jnp.broadcast_to(jnp.linspace(250., 280., nlev), (2, nlev))), +               u=NS(data=jnp.full((2, nlev), 5.)), v=NS(data=jnp.zeros((2, nlev))), +               p_s=NS(data=jnp.full((2,), 1.0e5))) +    q_v = jnp.full((2, nlev), 2e-3) +    kw = dict(T_land=jnp.full((2,), 262.0), land_fraction=jnp.ones((2,))) +    args = (state, q_v, jnp.full((2,), 275.0), jnp.zeros((2,))) +    old = np.asarray(dc._tas_2m(*args, T_ice=271.35, **kw)) +    new = np.asarray(dc._tas_2m(*args, T_ice=271.35, +                                T2m_land=jnp.asarray([266.5, jnp.nan]), **kw)) +    assert new[0] == 266.5 +    assert new[1] == old[1] diff --git a/tests/unit/test_land_column_split.py b/tests/unit/test_land_column_split.py index c7096ac176..b1ce1ac882 100644 --- a/tests/unit/test_land_column_split.py +++ b/tests/unit/test_land_column_split.py @@ -23,13 +23,13 @@ class _State(NamedTuple):    def _step(state, cols):      """Column-independent stand-in for the land step on columns ``cols``: -    the same tuple layout (state, 10 per-cell fields, 5 counts, iter max).""" +    the same tuple layout (state, 11 per-cell fields, 5 counts, iter max)."""      T = state.T_soil.at[cols].set(state.T_soil[cols] * 1.01 + 0.5)      wet = state.wet.at[cols].set(T[cols, 0] > 290.0)      f = jnp.zeros(NCOL).at[cols].set(T[cols, 0] - 273.0)      held = jnp.zeros(NCOL).at[cols].set((T[cols, 1] > 280.0).astype(float))      n = jnp.sum(T[cols, 0] > 290.0).astype(jnp.int32) -    return ((_State(T, wet),) + (f,) * 7 + (held,) + (f, f) +    return ((_State(T, wet),) + (f,) * 7 + (held,) + (f, f, f + 1.0)              + (n, n, n, 2 * n, n) + (jnp.max(cols).astype(jnp.int32),))    
2011416       00:24  8.3 Ds   /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -m pytest -p no:cacheprovider tests/land/unit/test_land_screen_temperature.py tests/unit/test_land_column_split.py tests/land/unit/test_canopy_stability.py tests/land/unit/test_canopy_stability_faithful.py -q
2012163       00:00  6.6 Ss   /usr/bin/bash -c sed -n '382,435p' packages/land/legoesm/land/surface_scheme/patch_mosaic.py; sed -n '407,480p' packages/coupler/legoesm/driver/model_driver.py; sed -n '367,391p' pyproject.toml; ps -o pid,etime,pcpu,stat,args -u pg2328 | rg 'pytest.*land_screen'
2012173       00:00  0.0 S    rg pytest.*land_screen

exec
/usr/bin/bash -lc 'PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH="$PWD/src:$PWD/packages/core:$PWD/packages/atmosphere:$PWD/packages/coupler:$PWD/packages/land:$PWD/packages/ocean:$PWD/packages/ice:$PWD/packages/ml:$PWD/packages/tools" /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python - <<'"'PY'
import os
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
import subprocess
import jax
import jax.numpy as jnp
import numpy as np
from legoesm.land.canopy import stability as current
base = {'__name__': 'base_stability'}
exec(compile(subprocess.check_output(['git', 'show', '8750d0005d:packages/land/legoesm/land/canopy/stability.py'], text=True), '<base_stability>', 'exec'), base)
for dtype in (jnp.float32, jnp.float64):
    for dt in (-6., 0., 4.):
        args = tuple(jnp.asarray(x, dtype=dtype) for x in (3., 275., 275., 275.+dt, .003, .003, 45., .9))
        for n in (1, 2, 5):
            kw = dict(n_iters=n, zeta_cap_width=.05)
            ref = base['monin_obukhov_stability'](*args, **kw)
            for got in (current.monin_obukhov_stability(*args, **kw), current.monin_obukhov_stability(*args, **kw, return_flux_obu=True)[:5]):
                for a, b in zip(ref, got):
                    np.testing.assert_array_equal(a,b)
print('PASS: original five MOST outputs equal base in fp32/fp64, stable/neutral/unstable, 1/2/5 iterations, both return modes')
PY" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
exec
/usr/bin/bash -lc "nl -ba packages/land/legoesm/land/canopy/stability.py | sed -n '299,312p;379,390p;462,480p'; nl -ba packages/coupler/legoesm/driver/model_driver.py | sed -n '14723,14779p'; nl -ba packages/coupler/legoesm/driver/diagnostics.py | sed -n '910,920p'; rg -n 'land.*T2m|land.*skin_last|land.*qsfc_last' packages/coupler/legoesm/driver/model_driver.py" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
   299	    (CanopyFluxesMod.F90:1182, FrictionVelocityMod.F90:636-652)::
   300	
   301	        t_ref2m = thm + temp1 * dth * (1/temp12m - 1/temp1),  dth = thm - taf
   302	
   303	    ``temp1`` and ``temp12m`` are evaluated at the SAME ``obu`` (as CLM does
   304	    within one FrictionVelocity call), so ``temp1`` is the flux's own ``ch``.  Stable side: Beljaars & Holtslag (1991),
   305	    as in the solve (module docstring).  Output is in the same frame as
   306	    ``theta_ref`` (surface-referenced potential temperature, CLM ``thm``)."""
   307	    temp1 = _temperature_humidity_relation(zldis, obu, z0h)
   308	    temp12m = _temperature_humidity_relation(z_screen + z0h, obu, z0h)
   309	    dth = theta_ref - T_canopy_air
   310	    return theta_ref + temp1 * dth * (1.0 / temp12m - 1.0 / temp1)
   311	
   312	
   379	        -constants.g * ustar * thvstar * _CONV_BDY_HEIGHT / Tv_atm, _MOST_ARG_FLOOR))
   380	    um_unstable = jnp.sqrt(ur**2 + wc_unstable**2)
   381	
   382	    is_stable = zeta >= 0.0
   383	    zeta_new = jnp.where(is_stable, zeta_stable, zeta_unstable)
   384	    um_new   = jnp.where(is_stable, um_stable,   um_unstable)
   385	    obu_new  = zldis / zeta_new
   386	
   387	    new_carry = jnp.array([z0h, obu_new, um_new])
   388	    output    = jnp.array([ustar, tstar, qstar, thvstar, ch, zeta_new, obu])
   389	    return new_carry, output
   390	
   462	    (final_carry, last), _ = jax.lax.scan(
   463	        _step_keep_last, (init, jnp.zeros(_out.shape, _out.dtype)), xs=None,
   464	        length=n_iters)
   465	
   466	    ustar, ch, zeta = last[0], last[4], last[5]
   467	
   468	    rah = 1.0 / jnp.maximum(ch * ustar, 1e-9)
   469	    raw = rah  # same for sensible heat and water vapour (neutral Prandtl)
   470	
   471	    um_final = final_carry[2]
   472	    ram      = jnp.maximum(um_final / ustar**2, 1e-9)
   473	    uav      = jnp.sqrt(um_final / ram)
   474	
   475	    if return_flux_obu:
   476	        return ustar, rah, raw, uav, zeta, last[6]
   477	    return ustar, rah, raw, uav, zeta
   478	
   479	
   480	# ---------------------------------------------------------------------------
 14723	                    jnp.asarray(resp.emissivity, resp.T_rad.dtype),
 14724	                    resp.T_rad.shape)
 14725	                # The scheme's own 2 m air temperature (CLM5 t_ref2m) for the
 14726	                # CMOR ``tas`` land leg; NaN where the scheme solves none or the
 14727	                # column was held (no solved surface layer this call).
 14728	                _t2m = (jnp.full_like(resp.T_rad, jnp.nan)
 14729	                        if getattr(_sfc, "T_2m", None) is None
 14730	                        else jnp.asarray(_sfc.T_2m, resp.T_rad.dtype))
 14731	                _t2m = jnp.where(_held_f > 0.5, jnp.nan, _t2m)
 14732	                if _land_pack_on:
 14733	                    # Scatter the advanced columns back into the full-grid
 14734	                    # state (ocean columns keep their frozen init values,
 14735	                    # exactly what the unpacked solve left them at after the
 14736	                    # f_land blend discarded its work) and the five consumed
 14737	                    # response fields into zero-filled cell arrays.
 14738	                    new_state = scatter_land_columns(
 14739	                        land_state, new_state,
 14740	                        _land_out_idx, _land_ncol_full)
 14741	                    return ((new_state,)
 14742	                            + tuple(scatter_cells(
 14743	                                o, _land_out_idx, _land_ncol_full)
 14744	                                for o in (
 14745	                                    resp.T_sfc, resp.albedo, resp.q_surface,
 14746	                                    resp.shflx, resp.lhflx, resp.surface_mass_flux,
 14747	                                    _taumag, _held_f, resp.T_rad, _emis))
 14748	                            # 2 m T: NaN (= no land screen value) off the
 14749	                            # land columns, never a valid-looking 0 K
 14750	                            + (jnp.full((_land_ncol_full,), jnp.nan, _t2m.dtype)
 14751	                               .at[_land_out_idx].set(_t2m.reshape(-1)),)
 14752	                            + (_n_held, _n_held_land, _n_fb_land,
 14753	                               _n_fb_rej, _n_unconv_land, _n_iter_max))
 14754	                return (new_state, resp.T_sfc, resp.albedo, resp.q_surface,
 14755	                        resp.shflx, resp.lhflx, resp.surface_mass_flux,
 14756	                        _taumag, _held_f, resp.T_rad, _emis, _t2m,
 14757	                        _n_held, _n_held_land, _n_fb_land, _n_fb_rej,
 14758	                        _n_unconv_land, _n_iter_max)
 14759	              if not _land_none:
 14760	                  return _land_step
 14761	
 14762	              @jax.jit
 14763	              def _land_step_none(land_state, a2s, doy, lai_doy, lai_year):
 14764	                  # Same output structure and dtypes as the solve (traced
 14765	                  # shape-only, never run), so the land halo exchange packs
 14766	                  # the same message on every rank.
 14767	                  out = jax.eval_shape(
 14768	                      _land_step, land_state, a2s, doy, lai_doy, lai_year)
 14769	                  # (the 2 m T, last per-cell field, is NaN = none solved)
 14770	                  return (land_state,) + tuple(
 14771	                      jnp.full(o.shape, jnp.nan, o.dtype)
 14772	                      if i == _LAND_STEP_N_CELL - 1
 14773	                      else jnp.zeros(o.shape, o.dtype)
 14774	                      for i, o in enumerate(out[1:]))
 14775	              return _land_step_none
 14776	
 14777	            _land_step_fn = _make_land_step(DT_LAND)
 14778	            # Bootstrap variant: one host step's forcing, tile advanced by
 14779	            # DT — used only for the first call after a (re)start so the
   910	        q_sfc_land = (saturation_mixing_ratio(T_land, p_s) if q_land is None
   911	                      else q_land)
   912	        T_2m_land = _profile_to_2m(T_land, q_sfc_land, ocean=False)
   913	        if T2m_land is not None:
   914	            T2m_land = jnp.asarray(T2m_land, dtype=T_2m_land.dtype)
   915	            T_2m_land = jnp.where(jnp.isfinite(T2m_land), T2m_land, T_2m_land)
   916	        T_2m_sea = _profile_to_2m(T_sfc_sea, q_sfc_sea, ocean=True)
   917	        f_land = jnp.clip(jnp.asarray(land_fraction, dtype=T_2m_land.dtype),
   918	                          0.0, 1.0)
   919	        return f_land * T_2m_land + (1.0 - f_land) * T_2m_sea
   920	
10931:                _tas_land_T = getattr(self, "_land_T_skin_last", None)
10932:                _tas_land_q = getattr(self, "_land_qsfc_last", None)
10933:                _tas_land_T2m = getattr(self, "_land_T2m_last", None)
10942:                        T2m_land=(None if _tas_land_T2m is None
10943:                                  else jnp.asarray(_tas_land_T2m).reshape(-1)))
15607:                _land_T_skin = self._land_T_skin_last = _lw_T
15609:                self._land_qsfc_last = _lw_qlast
16328:                     _land_Trad_cells, _land_emis_cells, _land_T2m_step,
16351:                    self._land_T_skin_last = _land_T_skin
16352:                    self._land_qsfc_last = _land_qsfc_step
16353:                    self._land_T2m_last = _land_T2m_step
16417:                        self._land_T_skin_last = _land_T_skin
16418:                        self._land_qsfc_last = _land_qsfc_step
16430:                     getattr(self, "_land_qsfc_last", None),

 succeeded in 29824ms:
PASS: original five MOST outputs equal base in fp32/fp64, stable/neutral/unstable, 1/2/5 iterations, both return modes

exec
/usr/bin/bash -lc "ps -o pid,etime,time,pcpu,stat,comm -p 2011416; sed -n '15600,15635p' packages/coupler/legoesm/driver/model_driver.py; rg -n 'test_.*(flux|jit|grad)' tests/land/unit/test_canopy_stability.py tests/land/unit/test_canopy_stability_faithful.py" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 99ms:
    PID     ELAPSED     TIME %CPU STAT COMMAND
2011416       01:27 00:01:25 98.8 Ssl  python
        if _land_ml_on and _lw_ck:
            (_land_a2s_sum, _land_a2s_n, _land_cz_wsum, _land_w_sum,
             (_lw_T, _lw_alb, _lw_qlast, _lw_qs, _lw_sh, _lw_lh,
              _lw_ev, _lw_beta, _lw_Trad, _lw_emis)) = land_window_from_checkpoint(
                _lw_ck, _LAND_K,
                int(jnp.asarray(self.state.p_s.data).reshape(-1).shape[0]))
            if _lw_T is not None:
                _land_T_skin = self._land_T_skin_last = _lw_T
                _land_albedo_cells = _lw_alb
                self._land_qsfc_last = _lw_qlast
                # A checkpoint written before the radiating pair was
                # persisted: seed it with the held skin T and the radiation's
                # own emissivity (vote V2a, 3-0) until the first land call.
                _land_Trad_cells = _lw_T if _lw_Trad is None else _lw_Trad
                _land_emis_cells = (
                    jnp.full_like(_lw_T, _E_sfc_rad) if _lw_emis is None
                    else _lw_emis)
                if _lw_Trad is None:
                    logger.warning(
                        "  MPAS land window: the checkpoint carries no held "
                        "land radiating pair (written before it was "
                        "persisted); radiation uses the held skin T at the "
                        "radiation emissivity until the first land call.")
            # The flux channel exists only with the land beta handoff; its
            # seeds above are replaced by the held solved values when the
            # checkpoint carries them.
            if _land_qsfc_cells is not None and _lw_qs is not None:
                _land_qsfc_cells, _land_shflx_cells = _lw_qs, _lw_sh
                _land_lhflx_cells, _land_evap_cells = _lw_lh, _lw_ev
            if _land_beta_cells is not None and _lw_beta is not None:
                _land_beta_cells = _lw_beta
            elif (_land_beta_cells is not None and _lw_T is not None
                  and _land_qsfc_cells is not None):
                logger.warning(
                    "  MPAS land window: the checkpoint carries no held beta "
                    "(written before it was persisted); beta is re-derived "
tests/land/unit/test_canopy_stability_faithful.py:369:def test_ustar_grad_wrt_um_is_kappa_over_fm(name, obu):
tests/land/unit/test_canopy_stability_faithful.py:380:def test_solver_grad_finite_x64_and_float32():

codex
The original five MOST outputs match the base exactly in the fp32/fp64 checks, across stable, neutral, and unstable cases, with either return mode. The focused suite is still running the full canopy cases.
exec
/usr/bin/bash -lc "git diff 8750d0005d -- packages/land/legoesm/land/canopy/solver.py packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py; sed -n '748,770p' /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90; rg -n '"'^ *dth'"\\(p\\) =' /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 27ms:
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
      itlef = 0    
      fnorig = fn
      fporig(1:fn) = filterp(1:fn)

      ! Begin stability iteration

      call t_startf('can_iter')
      ITERATION : do while (itlef <= itmax .and. fn > 0)

         ! Determine friction velocity, and potential temperature and humidity
         ! profiles of the surface boundary layer

         call FrictionVelocity (begp, endp, fn, filterp, &
              displa(begp:endp), z0mv(begp:endp), z0hv(begp:endp), z0qv(begp:endp), &
              obu(begp:endp), itlef+1, ur(begp:endp), um(begp:endp), ustar(begp:endp), &
              temp1(begp:endp), temp2(begp:endp), temp12m(begp:endp), temp22m(begp:endp), fm(begp:endp), &
              frictionvel_inst)

         



         do f = 1, fn
715:         dth(p) = thm(p)-taf(p)
1080:            dth(p) = thm(p)-taf(p)

exec
/usr/bin/bash -lc 'ps -o pid,etime,time,pcpu,stat,comm -p 2011416; git diff --numstat 8750d0005d' in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 562ms:
    PID     ELAPSED     TIME %CPU STAT COMMAND
2011416       02:25 00:03:08  130 Ssl  python
13	1	packages/coupler/legoesm/driver/diagnostics.py
24	5	packages/coupler/legoesm/driver/model_driver.py
4	3	packages/land/legoesm/land/canopy/solver.py
34	3	packages/land/legoesm/land/canopy/stability.py
4	0	packages/land/legoesm/land/surface_scheme/base.py
3	1	packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py
157	0	tests/land/unit/test_land_screen_temperature.py
2	2	tests/unit/test_land_column_split.py

codex
No additional defects found. The focused suite has passed the profile and coefficient checks and the first full canopy case; the second canopy case is still running. MPI and restart paths are checked by source inspection only.
 succeeded in 222796ms:
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
configfile: pyproject.toml
plugins: zarr-3.4.0, timeout-2.4.0, jaxtyping-0.3.11, xdist-3.8.0
collected 63 items

tests/land/unit/test_land_screen_temperature.py ....................     [ 31%]
tests/unit/test_land_column_split.py ......                              [ 41%]
tests/land/unit/test_canopy_stability.py ...........                     [ 58%]
tests/land/unit/test_canopy_stability_faithful.py ...................... [ 93%]
....                                                                     [100%]

======================== 63 passed in 200.12s (0:03:20) ========================

codex
CLEAN — no blocking findings.

- `stability.py:388,475` correctly returns the incoming flux Obukhov length. The original five outputs—including `rah`, `raw`, `ustar`, and `zeta`—matched base exactly in fp32/fp64 across stable, neutral, and unstable cases, both return modes, and 1/2/5 iterations.
- `model_driver.py:14728,14750,14770` correctly NaN-fills missing, held, packed-out, and empty-rank T2m values. Producers, gather slicing, and unpacking consistently use 11 per-cell fields.
- `diagnostics.py:915` replaces only finite land T2m entries; non-finite entries preserve the existing profile. The screen formula matches CLM5’s construction, retaining the documented BH91 stable-side departure.

Validation: **63 passed** across screen-temperature, column-split, and canopy-stability suites. No files edited.

UNVERIFIED: live MPI and checkpoint/restart execution; those paths were reviewed by source inspection.
tokens used
117,839
CLEAN — no blocking findings.

- `stability.py:388,475` correctly returns the incoming flux Obukhov length. The original five outputs—including `rah`, `raw`, `ustar`, and `zeta`—matched base exactly in fp32/fp64 across stable, neutral, and unstable cases, both return modes, and 1/2/5 iterations.
- `model_driver.py:14728,14750,14770` correctly NaN-fills missing, held, packed-out, and empty-rank T2m values. Producers, gather slicing, and unpacking consistently use 11 per-cell fields.
- `diagnostics.py:915` replaces only finite land T2m entries; non-finite entries preserve the existing profile. The screen formula matches CLM5’s construction, retaining the documented BH91 stable-side departure.

Validation: **63 passed** across screen-temperature, column-split, and canopy-stability suites. No files edited.

UNVERIFIED: live MPI and checkpoint/restart execution; those paths were reviewed by source inspection.
