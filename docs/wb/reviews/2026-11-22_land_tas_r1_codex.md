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
session id: 01a202f8-9d13-7113-985c-df09934e9430
--------
user
ADVERSARIAL REVIEW (read-only; do not edit files). Repo worktree: /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193, branch fix/land-tas-t-ref2m (base 8750d0005d). CLM5 source: /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys.

Change: CMIP `tas` land leg on the production MPAS lane (driver/model_driver.py _mpas_surface_temperatures -> diagnostics.py _tas_2m) used the OCEAN bulk law (large_yeager MOST, Charnock roughness, no displacement) anchored on the land T_sfc (= two-leaf canopy-air temperature) from the lowest model level. Fix: the two-leaf canopy now exports CLM5 t_ref2m (CanopyFluxesMod.F90:1182, FrictionVelocityMod.F90:636-652) computed with the land's own z0h(=z0m), displacement, reference height and converged Obukhov length (stability.screen_level_temperature), carried through the land step tuple (_LAND_STEP_N_CELL 10->11, NaN where held / scheme has none) to the driver (self._land_T2m_last) and into _tas_2m(T2m_land=...), where finite entries replace the ocean-law land profile. Diagnostic only: no model state, flux or coupling value changes.

Check: (1) formula fidelity vs CLM5 (frame of theta_ref vs thm, zldis, obu consistency with the flux solve, z0h), (2) the stable side uses the model's BH91 forms (documented departure), (3) driver plumbing: tuple order at every producer/consumer incl. packed scatter path, _land_step_none, gather_land_step, the multi-rank halo block, restart; (4) tests can fail (non-vacuous), oracle independent; (5) differentiability; (6) anything that silently changes non-diagnostic behaviour.

VOTE on these detail choices (answer A or B each, one line why):
V1 frame: A = publish CLM t_ref2m verbatim (no 2 m potential-to-temperature step, CLM does not apply it); B = subtract g/c_p*2 (~0.02 K) like the existing MOST/sea legs of _tas_2m.
V2 missing value (held column, non-two-leaf scheme, before first land call): A = keep the existing ocean-law land profile for that column; B = lowest-model-level temperature.
V3 restart/halo: A = accept that T2m is not in the checkpoint land-window payload nor halo-exchanged (first land interval after restart and halo cells use the V2 fallback / rank-local value), documented; B = add it to the payload and the halo exchange now.

Report <=350 words: findings ranked (file:line), verdict CLEAN or NOT CLEAN, votes V1-V3.

DIFF:
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
index 656bda453e..37c457d518 100644
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
@@ -14734,12 +14744,13 @@ class ModelDriver:
                                 for o in (
                                     resp.T_sfc, resp.albedo, resp.q_surface,
                                     resp.shflx, resp.lhflx, resp.surface_mass_flux,
-                                    _taumag, _held_f, resp.T_rad, _emis))
+                                    _taumag, _held_f, resp.T_rad, _emis,
+                                    _t2m))
                             + (_n_held, _n_held_land, _n_fb_land,
                                _n_fb_rej, _n_unconv_land, _n_iter_max))
                 return (new_state, resp.T_sfc, resp.albedo, resp.q_surface,
                         resp.shflx, resp.lhflx, resp.surface_mass_flux,
-                        _taumag, _held_f, resp.T_rad, _emis,
+                        _taumag, _held_f, resp.T_rad, _emis, _t2m,
                         _n_held, _n_held_land, _n_fb_land, _n_fb_rej,
                         _n_unconv_land, _n_iter_max)
               if not _land_none:
@@ -16307,7 +16318,7 @@ class ModelDriver:
                      _land_albedo_cells, _land_qsfc_step,
                      _land_shflx_step, _land_lhflx_step, _land_evap_step,
                      _land_taumag_step, _land_held_step,
-                     _land_Trad_cells, _land_emis_cells,
+                     _land_Trad_cells, _land_emis_cells, _land_T2m_step,
                      _land_n_held_step, _land_n_held_land_step,
                      _land_n_fb_step, _land_n_fb_rej_step,
                      _land_n_unconv_step, _land_n_iter_max) = _land_fn(
@@ -16332,6 +16343,7 @@ class ModelDriver:
                     # land and publishes a land temperature the model never had.
                     self._land_T_skin_last = _land_T_skin
                     self._land_qsfc_last = _land_qsfc_step
+                    self._land_T2m_last = _land_T2m_step
                     # Accumulate ON DEVICE and read at the same cadence the
                     # other post-step warnings use: reading it every step
                     # would stall the accelerator once per step for a number
diff --git a/packages/land/legoesm/land/canopy/stability.py b/packages/land/legoesm/land/canopy/stability.py
index b5efb6366a..1cb2225d45 100644
--- a/packages/land/legoesm/land/canopy/stability.py
+++ b/packages/land/legoesm/land/canopy/stability.py
@@ -92,6 +92,7 @@ _ZETA_MAX_STABLE = 0.5
 # stability forms are continuous through zeta = 0 (user 2026-09-25).
 _ZETA_NEUTRAL_FLOOR = 1.0e-6
 _CONV_BDY_HEIGHT = 1000.0  # convective boundary layer height [m]
+_SCREEN_HEIGHT_M = 2.0     # CMIP tas / CLM5 t_ref2m height above d + z0h [m]
 _Z0MG_BARE = 0.01          # bare-soil momentum roughness length [m]
 
 # --- CLM5 Monin-Obukhov similarity-theory (MOST) constants ---
@@ -286,6 +287,30 @@ def _temperature_humidity_relation(zldis: jax.Array, obu: jax.Array,
     return ch
 
 
+def screen_level_temperature(theta_ref: jax.Array, T_canopy_air: jax.Array,
+                             zldis: jax.Array, zeta: jax.Array,
+                             z0h: jax.Array,
+                             z_screen: float = _SCREEN_HEIGHT_M) -> jax.Array:
+    """CLM5 ``t_ref2m``: air temperature ``z_screen`` above the displacement
+    height plus ``z0h``, interpolated between the reference-height potential
+    temperature ``theta_ref`` and the canopy-air (aerodynamic surface)
+    temperature with this module's heat similarity relation at the solve's
+    Obukhov length ``obu = zldis / zeta``
+    (CanopyFluxesMod.F90:1182, FrictionVelocityMod.F90:636-652)::
+
+        t_ref2m = thm + temp1 * dth * (1/temp12m - 1/temp1),  dth = thm - taf
+
+    ``temp1`` and ``temp12m`` are evaluated at the SAME ``obu`` (as CLM does
+    within one FrictionVelocity call).  Stable side: Beljaars & Holtslag (1991),
+    as in the solve (module docstring).  Output is in the same frame as
+    ``theta_ref`` (surface-referenced potential temperature, CLM ``thm``)."""
+    obu = zldis / zeta
+    temp1 = _temperature_humidity_relation(zldis, obu, z0h)
+    temp12m = _temperature_humidity_relation(z_screen + z0h, obu, z0h)
+    dth = theta_ref - T_canopy_air
+    return theta_ref + temp1 * dth * (1.0 / temp12m - 1.0 / temp1)
+
+
 def _cap_stable_zeta(zeta: jax.Array, width: float) -> jax.Array:
     """Smooth cap at ``_ZETA_MAX_STABLE`` (no kink), then the neutral floor.
 
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
index 3aa3908df3..876be1155b 100644
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
+                                      fluxes_per_col["zeta"], z0m),
         gs_Sun=gs_Sun,
         gs_Sh=gs_Sh,
         n_iters=n_iters,
diff --git a/tests/land/unit/test_land_screen_temperature.py b/tests/land/unit/test_land_screen_temperature.py
new file mode 100644
index 0000000000..61080a6ea2
--- /dev/null
+++ b/tests/land/unit/test_land_screen_temperature.py
@@ -0,0 +1,128 @@
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
+def _oracle(theta, Tc, zl, zeta, z0h, zs=2.0):
+    L = zl / zeta
+    F = lambda z: math.log(z / z0h) - _psi_h(z / L) + _psi_h(z0h / L)
+    return Tc + (theta - Tc) * F(zs + z0h) / F(zl)
+
+
+@pytest.mark.parametrize("zeta", [-0.3, -0.05, 0.05, 0.3])
+@pytest.mark.parametrize("z0h", [0.05, 1.5])
+def test_profile_matches_independent_most(zeta, z0h):
+    theta, Tc, zl = 270.0, 262.0, 55.0
+    got = float(screen_level_temperature(jnp.asarray(theta), jnp.asarray(Tc),
+                                         jnp.asarray(zl), jnp.asarray(zeta),
+                                         jnp.asarray(z0h)))
+    np.testing.assert_allclose(got, _oracle(theta, Tc, zl, zeta, z0h),
+                               rtol=0.0, atol=1e-9)
+    assert min(theta, Tc) <= got <= max(theta, Tc)
+
+
+def test_neutral_limit_is_log_law():
+    theta, Tc, zl, z0h = 285.0, 289.0, 60.0, 0.8
+    got = float(screen_level_temperature(jnp.asarray(theta), jnp.asarray(Tc),
+                                         jnp.asarray(zl), jnp.asarray(1e-6),
+                                         jnp.asarray(z0h)))
+    log_law = Tc + (theta - Tc) * math.log((2.0 + z0h) / z0h) / math.log(zl / z0h)
+    np.testing.assert_allclose(got, log_law, rtol=0.0, atol=1e-5)
+
+
+def test_reference_at_screen_height_returns_reference():
+    z0h = 0.3
+    got = screen_level_temperature(jnp.asarray(271.0), jnp.asarray(260.0),
+                                   jnp.asarray(2.0 + z0h), jnp.asarray(0.2),
+                                   jnp.asarray(z0h))
+    np.testing.assert_allclose(float(got), 271.0, rtol=0.0, atol=1e-12)
+
+
+def test_gradient_finite_and_nonzero():
+    f = lambda th, tc, zeta: screen_level_temperature(
+        th, tc, jnp.asarray(50.0), zeta, jnp.asarray(0.5))
+    for zeta in (-0.3, 0.3):
+        g = jax.grad(f, argnums=(0, 1, 2))(jnp.asarray(270.0), jnp.asarray(262.0),
+                                          jnp.asarray(zeta))
+        assert all(np.isfinite(float(x)) and abs(float(x)) > 1e-8 for x in g), g
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
+    assert min(theta, tc) - 1e-9 <= t2 <= max(theta, tc) + 1e-9, (t2, tc, theta)
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
index c7096ac176..b02786e446 100644
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
+    return ((_State(T, wet),) + (f,) * 7 + (held,) + (f, f, f)
             + (n, n, n, 2 * n, n) + (jnp.max(cols).astype(jnp.int32),))
 
 
codex
I’ll check the repository rules, CLM5 source, and driver paths, then run focused checks without editing files.
exec
/usr/bin/bash -lc 'pwd; git status --porcelain; git worktree list; cat CLAUDE.md' in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 149ms:
/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
?? docs/wb/reviews/2026-11-22_land_tas_r1_codex.md
?? docs/wb/reviews/2026-11-22_land_tas_r1_prompt.md
/glade/u/home/pg2328/legoesm                                                                                                                        8750d0005d [perf/amip-cam6-derecho-2026-09-26]
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
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-fp32-tracers                                                                                     be5b35cdfa [perf/crm-fp32-tracers]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-les-scaling                                                                                      43d9f3a14d [perf/crm-les-mpi-scaling]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-mpi-gaps                                                                                         f3068db97e [fix/crm-mpi-gaps]
/glade/u/home/pg2328/legoesm/.claude/worktrees/crm-rad-blocking                                                                                     8669277fd4 [perf/crm-rad-blocking]
/glade/u/home/pg2328/legoesm/.claude/worktrees/final2                                                                                               be5b35cdfa (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/fx_base                                                                                              7759a33780 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/fx_fuse                                                                                              b70807a971 (detached HEAD)
/glade/u/home/pg2328/legoesm/.claude/worktrees/les-fft2                                                                                             aa638c1905 [perf/les-plane-fft]
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
/usr/bin/bash -lc "sed -n '290,350p' packages/coupler/legoesm/driver/model_driver.py; rg -n '_land_step_none|_LAND_STEP_N_CELL|_land_T2m|_land_Trad_cells|land_window|_land_fn|SurfaceFluxOutput\\(' packages/coupler/legoesm/driver/model_driver.py packages/land tests/unit/test_land_column_split.py; sed -n '220,320p' packages/land/legoesm/land/canopy/stability.py; sed -n '1160,1195p' /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90; sed -n '590,660p' /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/FrictionVelocityMod.F90" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 40ms:
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
packages/coupler/legoesm/driver/model_driver.py:300:_LAND_STEP_N_CELL = 11
packages/coupler/legoesm/driver/model_driver.py:306:    the land step tuple ``(state, _LAND_STEP_N_CELL per-cell fields, 5 counts,
packages/coupler/legoesm/driver/model_driver.py:321:        leaves, tdef = jax.tree_util.tree_flatten(out[:_LAND_STEP_N_CELL + 1])
packages/coupler/legoesm/driver/model_driver.py:339:                        for x in out[_LAND_STEP_N_CELL + 1:]])))
packages/coupler/legoesm/driver/model_driver.py:357:                       jnp.asarray(out[_LAND_STEP_N_CELL + 1 + i]).dtype)
packages/coupler/legoesm/driver/model_driver.py:358:                   for i in range(len(out) - _LAND_STEP_N_CELL - 2))
packages/coupler/legoesm/driver/model_driver.py:1684:def land_window_checkpoint_payload(a2s_sum, n, cz_wsum, w_sum, held):
packages/coupler/legoesm/driver/model_driver.py:1707:def land_window_from_checkpoint(payload, land_k, n_cells):
packages/coupler/legoesm/driver/model_driver.py:1708:    """Inverse of :func:`land_window_checkpoint_payload`.
packages/coupler/legoesm/driver/model_driver.py:2768:    def _land_window_for_checkpoint(self):
packages/coupler/legoesm/driver/model_driver.py:2776:        win = getattr(self, "_land_window", None)
packages/coupler/legoesm/driver/model_driver.py:8225:                _lw_save = self._land_window_for_checkpoint()
packages/coupler/legoesm/driver/model_driver.py:8255:                _lw_save = self._land_window_for_checkpoint()
packages/coupler/legoesm/driver/model_driver.py:9591:            self._land_window = None
packages/coupler/legoesm/driver/model_driver.py:10933:                _tas_land_T2m = getattr(self, "_land_T2m_last", None)
packages/coupler/legoesm/driver/model_driver.py:10942:                        T2m_land=(None if _tas_land_T2m is None
packages/coupler/legoesm/driver/model_driver.py:10943:                                  else jnp.asarray(_tas_land_T2m).reshape(-1)))
packages/coupler/legoesm/driver/model_driver.py:14352:        _land_Trad_cells = None
packages/coupler/legoesm/driver/model_driver.py:14760:              def _land_step_none(land_state, a2s, doy, lai_doy, lai_year):
packages/coupler/legoesm/driver/model_driver.py:14768:              return _land_step_none
packages/coupler/legoesm/driver/model_driver.py:15559:        self._land_window = None
packages/coupler/legoesm/driver/model_driver.py:15596:              _lw_ev, _lw_beta, _lw_Trad, _lw_emis)) = land_window_from_checkpoint(
packages/coupler/legoesm/driver/model_driver.py:15606:                _land_Trad_cells = _lw_T if _lw_Trad is None else _lw_Trad
packages/coupler/legoesm/driver/model_driver.py:16055:                    if _land_rad_bc_on and _land_Trad_cells is None:
packages/coupler/legoesm/driver/model_driver.py:16063:                            _land_Trad_cells, _land_emis_cells)
packages/coupler/legoesm/driver/model_driver.py:16312:                    _land_fn = (_land_step_fn if _land_a2s_n >= _LAND_K
packages/coupler/legoesm/driver/model_driver.py:16321:                     _land_Trad_cells, _land_emis_cells, _land_T2m_step,
packages/coupler/legoesm/driver/model_driver.py:16324:                     _land_n_unconv_step, _land_n_iter_max) = _land_fn(
packages/coupler/legoesm/driver/model_driver.py:16346:                    self._land_T2m_last = _land_T2m_step
packages/coupler/legoesm/driver/model_driver.py:16400:                         _land_Trad_cells, _land_emis_cells
packages/coupler/legoesm/driver/model_driver.py:16409:                             _land_Trad_cells, _land_emis_cells))
packages/coupler/legoesm/driver/model_driver.py:16418:                self._land_window = land_window_checkpoint_payload(
packages/coupler/legoesm/driver/model_driver.py:16426:                     _land_Trad_cells, _land_emis_cells))
packages/coupler/legoesm/driver/model_driver.py:16999:                        "_land_Trad_cells", "_land_emis_cells",
packages/land/legoesm/land/surface_scheme/base.py:16:class SurfaceFluxOutput(NamedTuple):
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:710:    return SurfaceFluxOutput(
packages/land/legoesm/land/surface_scheme/simple_seb.py:357:    return SurfaceFluxOutput(
packages/land/legoesm/land/canopy/clm_ml_interface.py:1593:    return SurfaceFluxOutput(
    Valid only for ζ ≤ 0 (see ``_stability_func_momentum``); clamp for
    grad-safety, a no-op for every in-regime call.
    """
    zeta = jnp.minimum(zeta, 0.0)
    chik2 = jnp.sqrt(1.0 - _MOST_GAMMA_UNSTABLE * zeta)
    return 2.0 * jnp.log((1.0 + chik2) * 0.5)


def _friction_velocity(zldis: jax.Array, z0m: jax.Array,
                       obu: jax.Array, um: jax.Array) -> jax.Array:
    """Compute ustar using 4-regime stability functions (CLM5 / DifferBESS)."""
    zetam = _ZETAM  # momentum regime transition
    zeta = zldis / obu

    # Very unstable (valid: obu < 0, zeta < -zetam).  Floor the log/cbrt args so
    # the discarded (obu > 0) branch stays finite; both floors are no-ops here.
    ustar1 = constants.kappa_von_karman * um / (
        jnp.log(jnp.maximum(-zetam * obu / z0m, _MOST_ARG_FLOOR))
        - _stability_func_momentum(-zetam)
        + _stability_func_momentum(z0m / obu)
        + _MOST_MOM_CONV_COEF * (jnp.cbrt(jnp.maximum(-zeta, zetam)) - jnp.cbrt(zetam))
    )
    # Unstable
    ustar2 = constants.kappa_von_karman * um / (
        jnp.log(zldis / z0m)
        - _stability_func_momentum(zeta)
        + _stability_func_momentum(z0m / obu)
    )
    # Stable (zeta >= 0): Beljaars & Holtslag (1991), all stable zeta.  psi_m
    # is finite for the negative arguments of the discarded (obu < 0) branch.
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
                             zldis: jax.Array, zeta: jax.Array,
                             z0h: jax.Array,
                             z_screen: float = _SCREEN_HEIGHT_M) -> jax.Array:
    """CLM5 ``t_ref2m``: air temperature ``z_screen`` above the displacement
    height plus ``z0h``, interpolated between the reference-height potential
    temperature ``theta_ref`` and the canopy-air (aerodynamic surface)
    temperature with this module's heat similarity relation at the solve's
    Obukhov length ``obu = zldis / zeta``
    (CanopyFluxesMod.F90:1182, FrictionVelocityMod.F90:636-652)::

        t_ref2m = thm + temp1 * dth * (1/temp12m - 1/temp1),  dth = thm - taf

    ``temp1`` and ``temp12m`` are evaluated at the SAME ``obu`` (as CLM does
    within one FrictionVelocity call).  Stable side: Beljaars & Holtslag (1991),
    as in the solve (module docstring).  Output is in the same frame as
    ``theta_ref`` (surface-referenced potential temperature, CLM ``thm``)."""
    obu = zldis / zeta
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
         delt_snow = wtal(p)*t_soisno(c,snl(c)+1)-wtl0(p)*t_veg(p)-wta0(p)*thm(p)
         eflx_sh_snow(p) = cpair*forc_rho(c)*wtg(p)*delt_snow

         delt_soil  = wtal(p)*t_soisno(c,1)-wtl0(p)*t_veg(p)-wta0(p)*thm(p)
         eflx_sh_soil(p) = cpair*forc_rho(c)*wtg(p)*delt_soil

         delt_h2osfc  = wtal(p)*t_h2osfc(c)-wtl0(p)*t_veg(p)-wta0(p)*thm(p)
         eflx_sh_h2osfc(p) = cpair*forc_rho(c)*wtg(p)*delt_h2osfc
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

         ! 2 m height relative humidity

         call QSat(t_ref2m(p), forc_pbot(c), e_ref2m, de2mdT, qsat_ref2m, dqsat2mdT)
         rh_ref2m(p) = min(100._r8, q_ref2m(p) / qsat_ref2m * 100._r8)
         rh_ref2m_r(p) = rh_ref2m(p)

         ! Human Heat Stress
         if (present(landunit_index)) then
            if (forc_hgt_q_patch(pfti(n)) == forc_hgt_t_patch(pfti(n)) .and. z0q(n) == z0h(n)) then
               temp2(n) = temp1(n)
            else
               zldis(n) = forc_hgt_q_patch(pfti(n))-displa(n)
               zeta(n) = zldis(n)/obu(n)
               if (zeta(n) < -zetat) then
                  temp2(n) = vkc/(log(-zetat*obu(n)/z0q(n)) &
                       - StabilityFunc2(-zetat) &
                       + StabilityFunc2(z0q(n)/obu(n)) &
                       + 0.8_r8*((zetat)**(-0.333_r8)-(-zeta(n))**(-0.333_r8)))
               else if (zeta(n) < 0._r8) then
                  temp2(n) = vkc/(log(zldis(n)/z0q(n)) &
                       - StabilityFunc2(zeta(n)) &
                       + StabilityFunc2(z0q(n)/obu(n)))
               else if (zeta(n) <=  1._r8) then
                  temp2(n) = vkc/(log(zldis(n)/z0q(n)) + 5._r8*zeta(n)-5._r8*z0q(n)/obu(n))
               else
                  temp2(n) = vkc/(log(obu(n)/z0q(n)) + 5._r8 - 5._r8*z0q(n)/obu(n) &
                       + (5._r8*log(zeta(n))+zeta(n)-1._r8))
               end if
            end if
         else
            if (forc_hgt_q_patch(n) == forc_hgt_t_patch(n) .and. z0q(n) == z0h(n)) then
               temp2(n) = temp1(n)
            else
               zldis(n) = forc_hgt_q_patch(n)-displa(n)
               zeta(n) = zldis(n)/obu(n)
               if (zeta(n) < -zetat) then
                  temp2(n) = vkc/(log(-zetat*obu(n)/z0q(n)) &
                       - StabilityFunc2(-zetat) &
                       + StabilityFunc2(z0q(n)/obu(n)) &
                       + 0.8_r8*((zetat)**(-0.333_r8)-(-zeta(n))**(-0.333_r8)))
               else if (zeta(n) < 0._r8) then
                  temp2(n) = vkc/(log(zldis(n)/z0q(n)) &
                       - StabilityFunc2(zeta(n)) &
                       + StabilityFunc2(z0q(n)/obu(n)))
               else if (zeta(n) <=  1._r8) then
                  temp2(n) = vkc/(log(zldis(n)/z0q(n)) + 5._r8*zeta(n)-5._r8*z0q(n)/obu(n))
               else
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

         if (z0q(n) == z0h(n)) then
            temp22m(n) = temp12m(n)
         else

exec
/usr/bin/bash -lc "sed -n '14745,14785p' packages/coupler/legoesm/driver/model_driver.py; sed -n '16315,16435p' packages/coupler/legoesm/driver/model_driver.py; rg -n 'Ta =|z_ref =|displa =|z0m =|zeta|compute_aerodynamics' packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py; rg -n 'zeta|compute_aerodynamics' packages/land/legoesm/land/canopy/{solver.py,energy_balance.py,stability.py}; rg -n 'thm\\(p\\) =|dth\\(p\\) =|z0h\\(p\\) =' /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90; ls -d .venv /glade/u/home/pg2328/legoesm/.venv; git diff --stat 8750d0005d" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 106ms:
                                    resp.T_sfc, resp.albedo, resp.q_surface,
                                    resp.shflx, resp.lhflx, resp.surface_mass_flux,
                                    _taumag, _held_f, resp.T_rad, _emis,
                                    _t2m))
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
                  return (land_state,) + tuple(
                      jnp.zeros(o.shape, o.dtype) for o in out[1:])
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
43:    compute_aerodynamics, sat_specific_humidity, screen_level_temperature,
196:    return compute_aerodynamics(
412:    z0m, displa = compute_aerodynamics(hc, LAI, rz0m, rd)
416:    z_ref = jnp.maximum(z_ref_base, displa + 10.0 * z0m + 2.0)
515:                               cc.zeta_cap_smoothing_width,
734:                                      fluxes_per_col["zeta"], z0m),
packages/land/legoesm/land/canopy/stability.py:21:reimplementation of those functions, at MATCHED ``zeta`` / ``z0`` / ``obu``:
packages/land/legoesm/land/canopy/stability.py:25:    very-unstable ``zeta < -zetam`` (mom) / ``< -zetat`` (heat) and unstable;
packages/land/legoesm/land/canopy/stability.py:28:    free-convection matches (momentum ``1.14 * ((-zeta)^1/3 - zetam^1/3)``, heat
packages/land/legoesm/land/canopy/stability.py:29:    ``0.8 * (zetat^-1/3 - (-zeta)^-1/3)`` — CLM5's INVERSE cube-root);
packages/land/legoesm/land/canopy/stability.py:33:DEPARTURE from CLM5 on the STABLE side (``zeta >= 0``, user 2026-09-25): CLM5's
packages/land/legoesm/land/canopy/stability.py:34:linear ``psi = -5 zeta`` (``0 <= zeta <= 1``) and very-stable log branch are
packages/land/legoesm/land/canopy/stability.py:41:below the ``zeta <= 0.5`` cap remains).  See
packages/land/legoesm/land/canopy/stability.py:44:The pin is on the FORMS at fixed ``zeta``, NOT the whole solve, because the
packages/land/legoesm/land/canopy/stability.py:45:iteration DRIVER is a departure (see below), so a converged ``zeta`` is
packages/land/legoesm/land/canopy/stability.py:51:  * very-unstable HEAT uses CLM5's INVERSE cube-root ``zetat^-1/3 - (-zeta)^-1/3``
packages/land/legoesm/land/canopy/stability.py:52:    whereas the LSM4 sibling uses a growing ``(-zeta)^1/3 - zetat^1/3`` — the
packages/land/legoesm/land/canopy/stability.py:61:    point (``zeta = zldis kappa g thetav*/(ustar^2 Tv)``, Zeng 1998 bulk-Ri init),
packages/land/legoesm/land/canopy/stability.py:62:    NOT the LSM4 bulk-Richardson ``zeta = r fm^2/fh`` iterated to tolerance.
packages/land/legoesm/land/canopy/stability.py:69:every regime).  ``zeta`` is clamped to ``[1e-6, 0.5]`` (stable; the 0.5 cap is a
packages/land/legoesm/land/canopy/stability.py:70:smooth min, see ``CanopyConfig.zeta_cap_smoothing_width`` — CLM5 clips hard) / ``[-100,
packages/land/legoesm/land/canopy/stability.py:88:# Near-neutral |zeta| floor (keeps obu = zldis/zeta finite).  CLM5 uses 0.01,
packages/land/legoesm/land/canopy/stability.py:89:# which makes zeta, ustar and the resistances JUMP by that much at neutral
packages/land/legoesm/land/canopy/stability.py:92:# stability forms are continuous through zeta = 0 (user 2026-09-25).
packages/land/legoesm/land/canopy/stability.py:103:# Stable-side (zeta >= 0) similarity functions: Beljaars & Holtslag (1991), the
packages/land/legoesm/land/canopy/stability.py:105:# -5 zeta + very-stable log forms, see docs/land/canopy_stable_stability_bh91.md).
packages/land/legoesm/land/canopy/stability.py:162:def compute_aerodynamics(
packages/land/legoesm/land/canopy/stability.py:201:def _stability_func_momentum(zeta: jax.Array) -> jax.Array:
packages/land/legoesm/land/canopy/stability.py:208:    zeta = jnp.minimum(zeta, 0.0)
packages/land/legoesm/land/canopy/stability.py:209:    chik2 = jnp.sqrt(1.0 - _MOST_GAMMA_UNSTABLE * zeta)
packages/land/legoesm/land/canopy/stability.py:217:def _stability_func_heat(zeta: jax.Array) -> jax.Array:
packages/land/legoesm/land/canopy/stability.py:223:    zeta = jnp.minimum(zeta, 0.0)
packages/land/legoesm/land/canopy/stability.py:224:    chik2 = jnp.sqrt(1.0 - _MOST_GAMMA_UNSTABLE * zeta)
packages/land/legoesm/land/canopy/stability.py:231:    zetam = _ZETAM  # momentum regime transition
packages/land/legoesm/land/canopy/stability.py:232:    zeta = zldis / obu
packages/land/legoesm/land/canopy/stability.py:234:    # Very unstable (valid: obu < 0, zeta < -zetam).  Floor the log/cbrt args so
packages/land/legoesm/land/canopy/stability.py:237:        jnp.log(jnp.maximum(-zetam * obu / z0m, _MOST_ARG_FLOOR))
packages/land/legoesm/land/canopy/stability.py:238:        - _stability_func_momentum(-zetam)
packages/land/legoesm/land/canopy/stability.py:240:        + _MOST_MOM_CONV_COEF * (jnp.cbrt(jnp.maximum(-zeta, zetam)) - jnp.cbrt(zetam))
packages/land/legoesm/land/canopy/stability.py:245:        - _stability_func_momentum(zeta)
packages/land/legoesm/land/canopy/stability.py:248:    # Stable (zeta >= 0): Beljaars & Holtslag (1991), all stable zeta.  psi_m
packages/land/legoesm/land/canopy/stability.py:252:        - psi_m(zeta, _STABLE_SCHEME) + psi_m(z0m / obu, _STABLE_SCHEME))
packages/land/legoesm/land/canopy/stability.py:254:    ustar = jnp.where(zeta < -zetam, ustar1,
packages/land/legoesm/land/canopy/stability.py:255:            jnp.where(zeta < 0.0,    ustar2, ustar3))
packages/land/legoesm/land/canopy/stability.py:262:    zetat = _ZETAT
packages/land/legoesm/land/canopy/stability.py:263:    zeta  = zldis / obu
packages/land/legoesm/land/canopy/stability.py:265:    # Very unstable (valid: obu < 0, zeta < -zetat).  Floor log/cbrt args; no-ops here.
packages/land/legoesm/land/canopy/stability.py:267:        jnp.log(jnp.maximum(-zetat * obu / z0h, _MOST_ARG_FLOOR))
packages/land/legoesm/land/canopy/stability.py:268:        - _stability_func_heat(-zetat)
packages/land/legoesm/land/canopy/stability.py:270:        + _MOST_HEAT_CONV_COEF * (1.0 / jnp.cbrt(zetat)
packages/land/legoesm/land/canopy/stability.py:271:                                  - 1.0 / jnp.cbrt(jnp.maximum(-zeta, zetat)))
packages/land/legoesm/land/canopy/stability.py:275:        - _stability_func_heat(zeta)
packages/land/legoesm/land/canopy/stability.py:278:    # Stable (zeta >= 0): Beljaars & Holtslag (1991), all stable zeta.  The core
packages/land/legoesm/land/canopy/stability.py:283:        - psi_h(zeta, _STABLE_SCHEME) + psi_h(z0h / obu, _STABLE_SCHEME))
packages/land/legoesm/land/canopy/stability.py:285:    ch = jnp.where(zeta < -zetat, ch1,
packages/land/legoesm/land/canopy/stability.py:286:         jnp.where(zeta < 0.0,    ch2, ch3))
packages/land/legoesm/land/canopy/stability.py:291:                             zldis: jax.Array, zeta: jax.Array,
packages/land/legoesm/land/canopy/stability.py:298:    Obukhov length ``obu = zldis / zeta``
packages/land/legoesm/land/canopy/stability.py:307:    obu = zldis / zeta
packages/land/legoesm/land/canopy/stability.py:314:def _cap_stable_zeta(zeta: jax.Array, width: float) -> jax.Array:
packages/land/legoesm/land/canopy/stability.py:321:    z = zeta - w * jax.nn.softplus((zeta - _ZETA_MAX_STABLE) / w)
packages/land/legoesm/land/canopy/stability.py:328:                        zeta_cap_width: float) -> tuple[jax.Array, jax.Array]:
packages/land/legoesm/land/canopy/stability.py:334:    zeta = jnp.where(
packages/land/legoesm/land/canopy/stability.py:339:    zeta = jnp.where(
packages/land/legoesm/land/canopy/stability.py:341:        _cap_stable_zeta(zeta, zeta_cap_width),
packages/land/legoesm/land/canopy/stability.py:342:        jnp.clip(zeta, -100.0, -_ZETA_NEUTRAL_FLOOR),  # coeff-ok: very-unstable bound on ζ
packages/land/legoesm/land/canopy/stability.py:344:    obu = zldis / zeta
packages/land/legoesm/land/canopy/stability.py:350:                    zeta_cap_width: float) -> tuple[jax.Array, jax.Array]:
packages/land/legoesm/land/canopy/stability.py:355:    output : [ustar, tstar, qstar, thvstar, ch, zeta]
packages/land/legoesm/land/canopy/stability.py:369:    zeta  = zldis * constants.kappa_von_karman * constants.g * thvstar / (ustar**2 * Tv_atm)
packages/land/legoesm/land/canopy/stability.py:371:    zeta_stable = _cap_stable_zeta(zeta, zeta_cap_width)
packages/land/legoesm/land/canopy/stability.py:373:    zeta_unstable = jnp.clip(zeta, -100.0, -_ZETA_NEUTRAL_FLOOR)  # coeff-ok: very-unstable bound on ζ
packages/land/legoesm/land/canopy/stability.py:382:    is_stable = zeta >= 0.0
packages/land/legoesm/land/canopy/stability.py:383:    zeta_new = jnp.where(is_stable, zeta_stable, zeta_unstable)
packages/land/legoesm/land/canopy/stability.py:385:    obu_new  = zldis / zeta_new
packages/land/legoesm/land/canopy/stability.py:388:    output    = jnp.array([ustar, tstar, qstar, thvstar, ch, zeta_new])
packages/land/legoesm/land/canopy/stability.py:403:    zeta_cap_width: float,
packages/land/legoesm/land/canopy/stability.py:418:    zeta_cap_width: smoothing width of the stable zeta <= 0.5 cap
packages/land/legoesm/land/canopy/stability.py:419:                    (``CanopyConfig.zeta_cap_smoothing_width``)
packages/land/legoesm/land/canopy/stability.py:427:    zeta     : stability parameter [-]
packages/land/legoesm/land/canopy/stability.py:437:    um, obu = _monin_obukhov_init(ur, Tv_atm, dthv, zldis, z0m, zeta_cap_width)
packages/land/legoesm/land/canopy/stability.py:442:                      zeta_cap_width=zeta_cap_width)
packages/land/legoesm/land/canopy/stability.py:462:    ustar, ch, zeta = last[0], last[4], last[5]
packages/land/legoesm/land/canopy/stability.py:471:    return ustar, rah, raw, uav, zeta
packages/land/legoesm/land/canopy/solver.py:234:    zeta_cap_width: float,
packages/land/legoesm/land/canopy/solver.py:258:        n_iters=most_n_iters, zeta_cap_width=zeta_cap_width)
packages/land/legoesm/land/canopy/solver.py:421:    zeta_cap_width: float,
packages/land/legoesm/land/canopy/solver.py:438:    ustar, rah_above, raw_above, uav, zeta = monin_obukhov_stability(
packages/land/legoesm/land/canopy/solver.py:440:        n_iters=most_n_iters, zeta_cap_width=zeta_cap_width)
packages/land/legoesm/land/canopy/solver.py:552:        ustar=ustar, zeta=zeta,
packages/land/legoesm/land/canopy/solver.py:603:    if not 0.0 < config.zeta_cap_smoothing_width <= ZETA_CAP_WIDTH_MAX:
packages/land/legoesm/land/canopy/solver.py:604:        raise ValueError("zeta_cap_smoothing_width must be in "
packages/land/legoesm/land/canopy/solver.py:606:                         f"cap divides by it), got {config.zeta_cap_smoothing_width!r}")
packages/land/legoesm/land/canopy/solver.py:613:        zeta_cap_width=config.zeta_cap_smoothing_width,
packages/land/legoesm/land/canopy/solver.py:647:    if not 0.0 < config.zeta_cap_smoothing_width <= ZETA_CAP_WIDTH_MAX:
packages/land/legoesm/land/canopy/solver.py:648:        raise ValueError("zeta_cap_smoothing_width must be in "
packages/land/legoesm/land/canopy/solver.py:650:                         f"cap divides by it), got {config.zeta_cap_smoothing_width!r}")
packages/land/legoesm/land/canopy/solver.py:657:        zeta_cap_width=config.zeta_cap_smoothing_width,
packages/land/legoesm/land/canopy/solver.py:675:    zeta_cap_width: float,
packages/land/legoesm/land/canopy/solver.py:691:            zeta_cap_width=zeta_cap_width,
715:         dth(p) = thm(p)-taf(p)
1080:            dth(p) = thm(p)-taf(p)
ls: cannot access '.venv': No such file or directory
ls: cannot access '/glade/u/home/pg2328/legoesm/.venv': No such file or directory
 packages/coupler/legoesm/driver/diagnostics.py     |  14 ++-
 packages/coupler/legoesm/driver/model_driver.py    |  22 +++-
 packages/land/legoesm/land/canopy/stability.py     |  25 ++++
 packages/land/legoesm/land/surface_scheme/base.py  |   4 +
 .../legoesm/land/surface_scheme/two_leaf_canopy.py |   4 +-
 tests/land/unit/test_land_screen_temperature.py    | 128 +++++++++++++++++++++
 tests/unit/test_land_column_split.py               |   4 +-
 7 files changed, 192 insertions(+), 9 deletions(-)

exec
/usr/bin/bash -lc "sed -n '345,475p' packages/land/legoesm/land/canopy/stability.py; sed -n '325,430p' packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py; sed -n '480,585p' packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py; sed -n '420,455p' packages/land/legoesm/land/canopy/solver.py; sed -n '15575,15625p' packages/coupler/legoesm/driver/model_driver.py; command -v python; python -c 'import jax; print(jax.__version__)'; rg -n 'thm|z0h|FrictionVelocity' /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys/CanopyFluxesMod.F90 | head -35; rg -n 'SurfaceFluxOutput|T_canopy_air' packages --glob '*.py' | head -55" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 34ms:
    return um, obu


def _stability_step(carry: jax.Array, _xs: None,
                    forcing: jax.Array,
                    zeta_cap_width: float) -> tuple[jax.Array, jax.Array]:
    """Single Monin-Obukhov fixed-point iteration.

    carry : [z0h, obu, um]
    forcing: [Ta, Tv_atm, q_atm, q_atm, zldis, z0m, dq, dth, ur]
    output : [ustar, tstar, qstar, thvstar, ch, zeta]
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
    output    = jnp.array([ustar, tstar, qstar, thvstar, ch, zeta_new])
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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
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

    return ustar, rah, raw, uav, zeta


# ---------------------------------------------------------------------------
# Leaf boundary-layer resistance
    # (matches the ``use_prognostic_lai`` getattr gate above; the multilayer
    # path always carries the real bool).  Still a static Python bool → the
    # ``if`` is a compile-time feature gate, not a traced select.
    if soil_surface_relsat is not None and getattr(
            land_config, "soil_evap_series_resistance", False):
        # Litter cover is driven by a persistent STRUCTURAL LAI (a deciduous forest
        # floor keeps its litter through the leaf-off season); fall back to the live
        # LAI when the caller does not supply ``litter_LAI``.
        _litter_LAI = getattr(lp, "litter_LAI", None) if lp is not None else None
        if _litter_LAI is None:
            _litter_LAI = LAI
        r_soil_surface = soil_surface_evap_resistance(
            soil_surface_relsat, LAI, land_config.soil_evap_litter_resistance_s_m,
            litter_LAI=_litter_LAI)
    else:
        r_soil_surface = jnp.zeros(ncol)
    # No-PFT fallback Vcmax25: DBF-temperate (C3) / mean C4 grass+crop (C4),
    # not a flat 60/40.  A driver should pre-assign per-column Vcmax25 from
    # ``lookup_vcmax25(pft, climate)`` (canopy.config) when PFTs are known.
    Vc3_leaf   = _get(lp, "Vcmax25_C3_leaf", jnp.full(ncol, VCMAX25_C3_DEFAULT))
    Vc4_leaf   = _get(lp, "Vcmax25_C4_leaf", jnp.full(ncol, VCMAX25_C4_DEFAULT))
    m_C3       = _get(lp, "m_C3",     jnp.full(ncol, _DEFAULT_M_C3))
    m_C4       = _get(lp, "m_C4",     jnp.full(ncol, _DEFAULT_M_C4))
    b0_C3      = _get(lp, "b0_C3",    jnp.full(ncol, _DEFAULT_B0_C3))
    b0_C4      = _get(lp, "b0_C4",    jnp.full(ncol, _DEFAULT_B0_C4))
    alf        = _get(lp, "alf",      jnp.full(ncol, _DEFAULT_ALF))
    # TgC is handled below — the priority order (state EMA > lp > forcing)
    # is set after all the other per-column params are resolved.
    ALB_VIS    = _get(lp, "ALB_VIS",  jnp.full(ncol, _DEFAULT_ALB_VIS))
    ALB_NIR    = _get(lp, "ALB_NIR",  jnp.full(ncol, _DEFAULT_ALB_NIR))
    rz0m       = _get(lp, "rz0m",     jnp.full(ncol, _DEFAULT_RZ0M))
    rd         = _get(lp, "rd",       jnp.full(ncol, _DEFAULT_RD))
    # Characteristic leaf width [m]; None (field absent or unset) -> 0.025 m
    # (Schuepp 1993 midrange), matching PFT_LEAF_WIDTH's default leaf class.
    _epsf_col = _get(lp, "epsf", None)   # per-column leaf emissivity, else cc.epsf
    _d_leaf_in = _get(lp, "d_leaf", None)
    d_leaf = jnp.full(ncol, _DEFAULT_D_LEAF) if _d_leaf_in is None else _d_leaf_in
    # NB: the two-stream ``canopy_longwave_rt`` resolves longwave from the
    # SEPARATE leaf and soil emissivities (cc.epsf / cc.epss) and exports the
    # conservative column ``eps_eff`` used as the surface emissivity below.  The
    # single broadband ``CanopyLandParams.emissivity`` has no unique mapping to
    # that leaf/soil pair, so this scheme uses the config-resolved pair and does
    # NOT consume the broadband field — that field drives the simpler SEB/slab
    # path instead (``simple_seb.compute_seb_fluxes``).  (Threading an OBSERVED
    # broadband emissivity into the two-stream RT would be a deliberate mapping
    # policy — a separate, validated change.)

    # ``TgC`` priority:
    #   1. Caller-supplied ``TgC_override`` (state-carried 30-day EMA from
    #      ``advance_TgC_ema``, threaded by ``step_*_land``).
    #   2. Per-column ``CanopyLandParams.TgC`` (externally prescribed).
    #   3. Instantaneous ``forcing.T_lowest - constants.T_freeze`` (degraded fallback;
    #      not a true 30-day mean — emits sensible but biased Vcmax
    #      acclimation outside any coupled / standalone driver).
    if TgC_override is not None:
        TgC = TgC_override
    else:
        TgC = _get(lp, "TgC", forcing.T_lowest - constants.T_freeze)

    # ---- Soil moisture stress ----
    # Photosynthesis/transpiration down-regulation uses the ROOT-ZONE beta.
    # Bare-soil evaporation is governed by the fast-drying SURFACE layer, not the
    # root zone: when ``w_frac_soil_evap`` (a top-layer availability) is supplied
    # by the caller, use it for ``fStress_soil``; otherwise fall back to the
    # root-zone beta (legacy behaviour).  Using the root-zone beta for soil evap
    # over-estimated forest-floor evaporation (it stays wet while the surface
    # dries), inflating LE and starving H.
    fStress_vcmax = w_frac_rz         # Vcmax / transpiration down-regulation
    fStress_soil  = (w_frac_rz if w_frac_soil_evap is None
                     else w_frac_soil_evap)   # soil evaporation stress

    Vc3_leaf_stressed = Vc3_leaf * fStress_vcmax
    Vc4_leaf_stressed = Vc4_leaf * fStress_vcmax
    m_eff  = m_C3  * fStress_vcmax
    # The Ball-Berry INTERCEPT b0 is the cuticular / residual minimum conductance.
    # ``cc.stress_b0`` (static Python bool) controls whether soil-moisture stress
    # down-regulates it too: True = legacy (b0 also stressed); False keeps the
    # cuticular leak alive under drought (baseline dry-season transpiration; the
    # photosynthesis-linked SLOPE m is still stressed).  ``_b0_stress`` is the
    # traced factor (1) or the plain scalar 1.0 — a compile-time branch, not a
    # traced select.
    _b0_stress = fStress_vcmax if cc.stress_b0 else 1.0
    b0_eff = b0_C3 * _b0_stress
    m_mix  = (1.0 - fC4) * m_eff  + fC4 * m_C4  * fStress_vcmax
    b0_mix = (1.0 - fC4) * b0_eff + fC4 * b0_C4 * _b0_stress

    # ---- Aerodynamics ----
    z0m, displa = compute_aerodynamics(hc, LAI, rz0m, rd)
    # Lift reference height above the displacement + roughness sub-layer.
    Ta, z_ref_base = surface_reference_state(
        forcing.T_lowest, land_config.z_ref, forcing.z_lowest)
    z_ref = jnp.maximum(z_ref_base, displa + 10.0 * z0m + 2.0)

    # ---- SW decomposition ----
    cos_zenith = forcing.cos_zenith
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
    rh_cap_width: float,
    zeta_cap_width: float,
    most_n_iters: int,
) -> dict:
    """Evaluate the FULLY_COUPLED canopy state and return all fluxes.

    Used after the Newton solver has converged to extract final diagnostics.
    The returned ``G`` is the surface-energy-budget residual
    ``Rn_soil - LE_soil - H_soil``, intended to be passed as the top BC to
    ``solve_soil_thermal`` in the caller.
    """
    Tf_Sun, Tf_Sh, Ci_Sun, Ci_Sh, Tc, q_c = (
        x[0], x[1], x[2], x[3], x[4], x[5])

    b = bundle
    Ts = b.Ts_bc
    zldis = b.z0 - b.displa

    ustar, rah_above, raw_above, uav, zeta = monin_obukhov_stability(
        b.ur, b.Ta, b.Tv_atm, Tc, b.q_atm, q_c, zldis, b.z0m,
        n_iters=most_n_iters, zeta_cap_width=zeta_cap_width)

    Rb_Sun, Rb_Sh = compute_boundary_layer_resistance(uav, b.LAI, b.fSun, b.cv, b.d_leaf)
    rah_below, raw_below = compute_below_canopy_resistance(uav, b.CI, b.LAI)
    # Soil evaporation (vapour) path adds the soil-surface resistance in SERIES
    # (Sellers 1992 r_ss + Sakaguchi-Zeng 2009 litter); the sensible-heat path
    # (rah_below) is a direct skin-to-canopy-air conduction and takes NO soil-side
    # resistance.  r_soil_surface = 0 recovers the pure-aerodynamic (legacy) form.
    raw_soil_evap = raw_below + b.r_soil_surface

    lw_out  = canopy_longwave_rt(
        b.LAI, b.CI, b.SZA, Ts, Tf_Sun, Tf_Sh, b.La, b.epsf, b.epss)
    ALW_Sun, ALW_Sh, ALW_Soil = lw_out.ALW_Sun, lw_out.ALW_Sh, lw_out.ALW_Soil
    Ls = lw_out.Ls
    Lcanopy_up = lw_out.Lcanopy_up
    gap_LW = lw_out.gap_LW
        _cpl_sst_due = False
        if self._segment_callback is not None and _cpl_dt is not None:
            _k = float(_cpl_dt) / DT
            if (not np.isfinite(_k) or _k < 1 or _k != round(_k)
                    or float(_cpl_dt) % 1800.0 != 0.0):
                raise ValueError(
                    f"coupling_dt={_cpl_dt!r} s must be a positive multiple "
                    f"of 1800 s and of the atmosphere step DT={DT} s.")
            _CPL_K = int(round(_k))
            if _cpl_ck:
                _cpl_sums, _cpl_n = coupling_window_from_checkpoint(
                    _cpl_ck, _CPL_K,
                    int(jnp.asarray(self.state.p_s.data).reshape(-1).shape[0]))
            elif start_step > 0:
                logger.warning(
                    "  Coupling window: the checkpoint carries no coupling "
                    "window (written before it was persisted); the pending "
                    "steps of the previous link are lost.")
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
/glade/u/apps/derecho/25.10/opt/view/bin/python
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'jax'
177:    use FrictionVelocityMod, only : FrictionVelocity, MoninObukIni, frictionvel_parms_inst
441:         z0hv                   => frictionvel_inst%z0hv_patch                  , & ! Output: [real(r8) (:)   ]  roughness length over vegetation, sensible heat [m]                   
449:         thm                    => temperature_inst%thm_patch                   , & ! Input:  [real(r8) (:)   ]  intermediate variable (forc_t+0.0098*forc_hgt_t_patch)                  
676:         z0hv(p)   = z0mv(p)
711:         taf(p) = (t_grnd(c) + thm(p))/2._r8
715:         dth(p) = thm(p)-taf(p)
760:         call FrictionVelocity (begp, endp, fn, filterp, &
761:              displa(begp:endp), z0mv(begp:endp), z0hv(begp:endp), z0qv(begp:endp), &
993:            efsh   = dc1*(wtga*t_veg(p)-wtg0*t_grnd(c)-wta0(p)*thm(p))
1074:            taf(p) = wtg0*t_grnd(c) + wta0(p)*thm(p) + wtl0(p)*t_veg(p)
1080:            dth(p) = thm(p)-taf(p)
1154:         delt    = wtal(p)*t_grnd(c)-wtl0(p)*t_veg(p)-wta0(p)*thm(p)
1160:         delt_snow = wtal(p)*t_soisno(c,snl(c)+1)-wtl0(p)*t_veg(p)-wta0(p)*thm(p)
1163:         delt_soil  = wtal(p)*t_soisno(c,1)-wtl0(p)*t_veg(p)-wta0(p)*thm(p)
1166:         delt_h2osfc  = wtal(p)*t_h2osfc(c)-wtl0(p)*t_veg(p)-wta0(p)*thm(p)
1182:         t_ref2m(p) = thm(p) + temp1(p)*dth(p)*(1._r8/temp12m(p) - 1._r8/temp1(p))
packages/land/legoesm/land/carbon_diagnostics.py:93:        surface scheme's ``SurfaceFluxOutput.gpp`` -- two-leaf canopy, CLM-ML
packages/land/legoesm/land/slab_land.py:82:    ``SurfaceFluxOutput``.
packages/land/legoesm/land/slab_land.py:101:    """Like :func:`step_land` but also returns the ``SurfaceFluxOutput``.
packages/land/legoesm/land/slab_land.py:636:    response_T_surface     = _unflat(surface_out.T_canopy_air)
packages/land/legoesm/land/__init__.py:19:    SurfaceFluxOutput,
packages/land/legoesm/land/__init__.py:36:    "SimpleSEBConfig", "TwoLeafCanopyConfig", "SurfaceFluxOutput",
packages/land/legoesm/land/multilayer_land.py:12:``SurfaceFluxOutput``; the post-flux pipeline (snow, Richards, soil
packages/land/legoesm/land/multilayer_land.py:19:2. Surface scheme dispatch: compute ``SurfaceFluxOutput`` (``shflx``,
packages/land/legoesm/land/multilayer_land.py:312:    """Like :func:`step_multilayer_land` but also returns the ``SurfaceFluxOutput``.
packages/land/legoesm/land/multilayer_land.py:314:    The 4th element is the surface scheme's ``SurfaceFluxOutput`` with all
packages/land/legoesm/land/multilayer_land.py:412:    ``SurfaceFluxOutput`` for diagnostic inspection.
packages/land/legoesm/land/multilayer_land.py:523:    """Internal 4-tuple (new_state, TileResponse, carbon, SurfaceFluxOutput).
packages/land/legoesm/land/multilayer_land.py:1290:                "(SurfaceFluxOutput.LE_snow): call compute_simple_seb_fluxes with "
packages/land/legoesm/land/multilayer_land.py:1308:                "charged it (SurfaceFluxOutput.L_soil)")
packages/land/legoesm/land/multilayer_land.py:2004:        # CLMMLCanopyConfig populates T_canopy_air in _extract_surface_fluxes;
packages/land/legoesm/land/multilayer_land.py:2006:        response_T_sfc = (surface_out.T_canopy_air
packages/land/legoesm/land/multilayer_land.py:2007:                          if surface_out.T_canopy_air is not None
packages/land/legoesm/land/multilayer_land.py:2129:      (``SurfaceFluxOutput.converged is False``), so its fluxes are the
packages/land/legoesm/land/stomata.py:220:    # output on SurfaceFluxOutput.sif (coupled Farquhar path only — the Jarvis
packages/land/legoesm/land/surface_scheme/base.py:4:to a ``SurfaceFluxOutput``.  Downstream land-model post-processing
packages/land/legoesm/land/surface_scheme/base.py:16:class SurfaceFluxOutput(NamedTuple):
packages/land/legoesm/land/surface_scheme/base.py:90:    T_canopy_air: jax.Array | None = None
packages/land/legoesm/land/surface_scheme/patch_mosaic.py:6:fraction.  The patch fluxes are aggregated to one grid-mean ``SurfaceFluxOutput``
packages/land/legoesm/land/surface_scheme/patch_mosaic.py:37:from legoesm.land.surface_scheme.base import SurfaceFluxOutput
packages/land/legoesm/land/surface_scheme/patch_mosaic.py:321:) -> SurfaceFluxOutput:
packages/land/legoesm/land/surface_scheme/patch_mosaic.py:322:    """Run the ``N``-patch mosaic and return one area-weighted grid ``SurfaceFluxOutput``.
packages/land/legoesm/land/surface_scheme/patch_mosaic.py:382:def _area_weight(out: SurfaceFluxOutput, fracs: jnp.ndarray) -> SurfaceFluxOutput:
packages/land/legoesm/land/surface_scheme/patch_mosaic.py:383:    """Area-weight-reduce a patch-axis ``SurfaceFluxOutput`` to one grid column.
packages/land/legoesm/land/surface_scheme/__init__.py:23:All three schemes produce a common ``SurfaceFluxOutput`` that downstream
packages/land/legoesm/land/surface_scheme/__init__.py:40:from legoesm.land.surface_scheme.base import SurfaceFluxOutput
packages/land/legoesm/land/surface_scheme/__init__.py:51:    "SurfaceFluxOutput",
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:15:``compute_two_leaf_canopy_fluxes`` returns a ``SurfaceFluxOutput`` that
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:50:from legoesm.land.surface_scheme.base import SurfaceFluxOutput
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:240:) -> SurfaceFluxOutput:
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:278:                    ``SurfaceFluxOutput.L_soil``.
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:282:    SurfaceFluxOutput with canopy diagnostics (``Tf_Sun``, ``gs_Sun``,
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:608:    # (``SurfaceFluxOutput.condensation``).
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:710:    return SurfaceFluxOutput(
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:732:        T_canopy_air=Tc_cvg,
packages/land/legoesm/land/surface_scheme/simple_seb.py:36:from legoesm.land.surface_scheme.base import SurfaceFluxOutput
packages/land/legoesm/land/surface_scheme/simple_seb.py:95:) -> SurfaceFluxOutput:
packages/land/legoesm/land/surface_scheme/simple_seb.py:130:    SurfaceFluxOutput with ``gpp`` populated if ``stomata.enabled`` and
packages/land/legoesm/land/surface_scheme/simple_seb.py:357:    return SurfaceFluxOutput(
packages/land/legoesm/land/canopy/config.py:229:    # (sunlit+shaded sum) on ``SurfaceFluxOutput.sif``.  Static config leaf —
packages/land/legoesm/land/canopy/config.py:776:    # on ``SurfaceFluxOutput.sif`` — a leaf-area-weighted sum over the CLM-ML
packages/land/legoesm/land/canopy/clm_ml_interface.py:6:``SurfaceFluxOutput``.
packages/land/legoesm/land/canopy/clm_ml_interface.py:42:from legoesm.land.surface_scheme import SurfaceFluxOutput
packages/land/legoesm/land/canopy/clm_ml_interface.py:63:        "T_canopy_air": "K (PAI-weighted mean within-canopy air temperature)",
packages/land/legoesm/land/canopy/clm_ml_interface.py:1462:) -> SurfaceFluxOutput:
packages/land/legoesm/land/canopy/clm_ml_interface.py:1463:    """Map ``mlcanopy_type`` output fields to a ``SurfaceFluxOutput``.
packages/land/legoesm/land/canopy/clm_ml_interface.py:1559:    # T_canopy_air: aerodynamic exchange temperature for SH coupling to atmosphere.
packages/land/legoesm/land/canopy/clm_ml_interface.py:1564:        T_canopy_air = jnp.stack([mlcanopy.taveg_canopy[i + 1] for i in range(ncol)])
packages/land/legoesm/land/canopy/clm_ml_interface.py:1566:        T_canopy_air = jnp.where(
packages/land/legoesm/land/canopy/clm_ml_interface.py:1567:            (T_canopy_air > 1e30) | (T_canopy_air < 100.0),
packages/land/legoesm/land/canopy/clm_ml_interface.py:1568:            T_surface, T_canopy_air,

codex
The screen formula follows CLM5 and uses the documented BH91 stable forms. One concern: the exported stability parameter is one iteration newer than the heat coefficient used by the flux solve. I’m checking that mismatch and the missing-value paths.
exec
/usr/bin/bash -lc "rg -n 'venv|conda|python' .claude/settings* pyproject.toml | head -25; ls /glade/work/pg2328; sed -n '14355,14465p' packages/coupler/legoesm/driver/model_driver.py; sed -n '14610,14655p' packages/coupler/legoesm/driver/model_driver.py; sed -n '382,425p' packages/land/legoesm/land/surface_scheme/patch_mosaic.py; sed -n '830,845p' packages/coupler/legoesm/driver/diagnostics.py; rg -n 'most_n_iters' packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py packages/land/legoesm/land/canopy/config.py; sed -n '350,370p' packages/coupler/legoesm/driver/model_driver.py" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
pyproject.toml:12:requires-python = ">=3.11"
pyproject.toml:320:# subsequent module-level imports trip E402; the bootstrap makes ``python scripts/...`` work.
pyproject.toml:361:python_version = "3.11"
pyproject.toml:395:# Run: `.venv/bin/lint-imports`
.claude/settings.json:8:      "Bash(python *)",
.claude/settings.json:9:      "Bash(python3 *)",
.claude/settings.json:25:            "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/check_banned_literals.py\""
.claude/settings.json:36:            "command": "python3 \"$CLAUDE_PROJECT_DIR/.claude/hooks/require_review_artifact.py\""
amip_inputs
conda-envs
cpuperf_env.sh
env_backups
envs
headroom-venv
legoesm-base0918
legoesm-cpuperf
legoesm-duofix
legoesm-ladder
legoesm-license
legoesm-thomas
mpiwrapper
ollama
venvs
        _land_beta_cells = None        # (nCells,) traced beta of the last step
        _land_qsfc_cells = None        # (nCells,) land's solved q_sfc, last step
        _land_shflx_cells = None       # (nCells,) land's own sensible flux
        _land_lhflx_cells = None       # (nCells,) land's own latent flux
        _land_evap_cells = None        # (nCells,) land's own water flux [kg/m2/s]
        _land_taumag_cells = None      # (nCells,) land's last valid |stress| [Pa]
        _land_taumag_valid = None      # (nCells,) a land step has succeeded
        _land_taumag_fresh = None      # (nCells,) value from the LAST land step
        _land_stress_landcell = None   # (nCells,) f_land > 0
        _ls_seed_win = None            # device: seeded land column-steps, window
        _ls_reused_win = None          # device: reused land column-steps, window
        _land_z0m_cells = None         # (nCells,) static land roughness [m]
        _land_d_cells = None           # (nCells,) static displacement [m]
        from legoesm.driver.config import (
            resolve_land_stress_from_land, resolve_ocean_flux_on_ocean_surface)
        from legoesm.land.multilayer_land import solved_stress_magnitude
        from legoesm.atmosphere.physics.turbulence.surface_layer import (
            hold_last_valid_land_stress,
        )
        _land_stress_on = resolve_land_stress_from_land(cfg)
        # Validated (validate_strict) to imply the land-flux handoff, so the
        # turbulence always receives T_sfc_ocean together with shflx_land.
        _ocean_sfc_on = resolve_ocean_flux_on_ocean_surface(cfg)
        print(f"  Non-land surface fluxes evaluated on: "
              f"{'OCEAN/ICE surface' if _ocean_sfc_on else 'land-blended surface'}"
              f" (ocean_flux_on_ocean_surface={_ocean_sfc_on})")
        print(f"  Land surface stress to the boundary layer: "
              f"{'LAND MODEL' if _land_stress_on else 'atmosphere bulk law'} "
              f"(land_stress_from_land="
              f"{getattr(cfg, 'land_stress_from_land', None)!r}"
              f"{' -> auto' if getattr(cfg, 'land_stress_from_land', None) is None else ''})")
        _land_a2s_sum = None           # cadence: running forcing sum
        _land_a2s_n = 0                # cadence: steps accumulated
        if _land_ml_on:
            if not _sst_forcing:
                raise ValueError(
                    "use_multilayer_land on the MPAS lane requires the SST "
                    "surface-forcing channel (radiation != 'none' with SST "
                    "data): the land skin temperature feeds back through "
                    "forcing['T_sfc'], which only exists on that path.")
            if _f_land_cells is None:
                raise ValueError(
                    "use_multilayer_land on the MPAS lane requires a land "
                    "fraction (--topography / --land-mask-file); none was "
                    "loaded — the land tile would be silently inert.")
            # Distributed is supported (#1321).  No column scatter is needed:
            # ``_create_grid`` installs the rank-local mesh BEFORE
            # ``_create_physics`` runs, and ``_setup_multilayer_land`` sizes
            # its columns from ``self.grid.latCell``, so the soil columns are
            # built rank-local already.  (The guard that used to sit here said
            # the opposite — "the setup built them on the full mesh" — which
            # stopped being true when grid creation moved ahead of physics.)
            # What WAS missing is the downwelling-radiation slots the land
            # forcing reads; ``make_voronoi_mpi_step`` now publishes the full
            # 10-slot contract, and this asserts it rather than letting
            # ``_marshal_land_forcing`` return None and the soil silently
            # never advance.
            from legoesm.land.multilayer_land import step_multilayer_land
            from legoesm.core.coupling_fields import AtmToSurface
            from legoesm.grids.voronoi import reconstruct_cell_velocity
            # Compiled once: eagerly it ran op-by-op every land-forcing call
            # (~0.5 s/step at 16 ranks, res6).
            _recon_grid = self.grid
            _recon_cell_velocity_jit = jax.jit(
                lambda u_edge: reconstruct_cell_velocity(u_edge, _recon_grid))
            _lml_cfg = self.physics.land_ml_cfg
            _lml_params = self.physics.land_ml_params
            _lml_lat = self.physics.land_ml_lat
            _lml_umin = float(getattr(self.physics, "land_ml_u_min", 1.0))
            # PRESCRIBED leaf carbon (fixed LAI).  Without it the coupled
            # stomatal dispatch falls back to Jarvis even when the config asks
            # for Farquhar, so the baked canopy conductance would be inert on
            # exactly the lane the AMIP campaign runs.  None on every other
            # configuration, which keeps those runs byte-identical.
            _lml_carbon = getattr(self.physics, "land_ml_carbon", None)
            # Per-step canopy-parameter rebuild (seasonal LAI etc.), as the
            # offline calibration runs it; refused, not skipped, when the
            # config asks for it and the scheme has no updater.
            _lml_update = None
            if bool(getattr(cfg, "land_params_refresh", False)):
                _lml_update = getattr(self.physics, "land_ml_params_update", None)
                _factory = getattr(
                    self.physics, "land_ml_params_update_factory", None)
                if _lml_update is None and _factory is not None:
                    _lml_update = _factory()
                if _lml_update is None:
                    raise ValueError(
                        "land_params_refresh=True but the land tile has no "
                        "per-step parameter updater: it is built only for the "
                        "two-leaf canopy on per-PFT surfdata parameters "
                        f"(land_surface_scheme={cfg.land_surface_scheme!r}). Set "
                        "land_params_refresh=false to keep the start-day "
                        "parameters knowingly.")
            _lml_year0 = (float(cfg.start_year)
                          if getattr(cfg, "start_year", None) is not None
                          else 0.0)

            # Land fraction as a closure constant of the compiled land step,
            # for the land-weighted held count below.
            _f_land_cols = jnp.asarray(self._f_land).reshape(-1)

            # --- Packed land columns (2026-08-25 step-cost profile: the tile
            # was ~58% of wall time while solving EVERY column, ocean
            # included, whose output every consumer blends away by f_land).
            # Gather the f_land > 0 columns — the mask is a compile-time
            # constant, so the gather is static — solve only those, scatter
            # back.  The land STATE stays full-grid between calls, so
            # checkpoints, restarts, MPI ownership and the beta_soil reader
            # are byte-identical.  f_land > 0, not > 0.5: fractional coastal
            # cells' outputs ARE consumed.  Scatter fills are 0.0, never NaN
            # (0 * NaN would contaminate the f_land blends).  clm_ml keeps
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
packages/land/legoesm/land/canopy/config.py:271:    most_n_iters: int = 10
packages/land/legoesm/land/canopy/config.py:302:        if not (isinstance(self.most_n_iters, int) and self.most_n_iters >= 1):
packages/land/legoesm/land/canopy/config.py:303:            raise ValueError("most_n_iters must be a positive int, got "
packages/land/legoesm/land/canopy/config.py:304:                             f"{self.most_n_iters!r}")
packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py:516:                               cc.most_n_iters)
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

exec
/usr/bin/bash -lc "rg -n '_land_none|_land_pack_on|def _scatter' packages/coupler/legoesm/driver/model_driver.py; ls /glade/work/pg2328/venvs /glade/work/pg2328/conda-envs /glade/work/pg2328/legoesm-cpuperf/bin/python; sed -n '440,510p' CLAUDE.md; rg -n 'read.only|review|agent' CLAUDE.md | head -30; sed -n '10910,10990p' packages/coupler/legoesm/driver/model_driver.py; rg -n 'def surface_reference_state' packages/land" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 exited 1 in 0ms:
85:def _scatter_flat_columns(flat_arr, layout, n_tile):
106:def _scatter_1based_columns(arr, layout, n_tile):
148:def _scatter_voronoi_columns(arr, partition):
165:def _scatter_1based_voronoi_columns(arr, partition):
7506:    def _scatter_global_state_to_bands(self, state_global, tracers_global):
9396:                def _scatter(name, entity):
14522:            _land_pack_on = (
14529:            _land_none = _land_pack_idx_np.size == 0
14556:            if _land_pack_on:
14594:                _state_in = (_land_pack(land_state) if _land_pack_on
14596:                _a2s_in = _land_pack(a2s) if _land_pack_on else a2s
14603:                    _params_in = _land_pack(_p) if _land_pack_on else _p
14732:                if _land_pack_on:
14756:              if not _land_none:
14929:                _land_pack_on, int(_land_pack_idx_np.size),
ls: cannot access '/glade/work/pg2328/legoesm-cpuperf/bin/python': No such file or directory
/glade/work/pg2328/conda-envs:
legoesm-gpu

/glade/work/pg2328/venvs:
context-compress
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
69:| codex review on uncommitted numerics | Stop hook, once/session, ADVISORY | `.claude/hooks/require_review_artifact.py` |
80:- **DUAL review (codex AND GLM).** The Stop hook only sees UNCOMMITTED work and
82:  which is exactly how a 14-conflict merge shipped unreviewed on 2026-08-20.
279:- "a reviewer recommended it"
312:detail", "numerically negligible" are NOT exemptions. Subagent briefs must
354:- Subagent briefs must state this rule.
358:- **DO WHAT WAS ASKED — DO NOT EXTRAPOLATE.** Deliver requested thing, then STOP and report. No inventing adjacent work because it seem useful: no unrequested helper/launcher/benchmark scripts, no extra files staged "for later", no speculative refactors, no scope widened from "fix X" to "also improve Y". Adjacent work look warranted → NAME it in one line, let user choose. Do not pre-build. Uncertain whether in scope? It not; ask. Over-delivery not helpfulness: cost review time, bury actual change, create artifacts nobody vetted.
362:- **DUAL adversarial review is the DEFAULT for EVERY substantive change — TWO
363:  independent reviewers, never one** (user directive 2026-08-12, restated
365:  Author NEVER review own code. Route by who WROTE code — two reviewers always
371:  line. Subagents doing implementation must be told to run BOTH — they default
391:  fixes for one defect, shipped unreviewed until asked.
392:- **Codex adversarial review MANDATORY after any major code implementation/change.** Trigger: new module/feature, dycore/physics/parallel/ocean/land/ice/coupler/training edit, >~50 LOC, multi-file, or anything touching numerics/AD/JIT/pytree/conservation. Run **iterate-with-codex agent** loop below (`/codex:adversarial-review --wait` → fix flagged → `/codex:review --wait` → repeat until clean or 30 iter) BEFORE declaring done; report that review ran + verdict.
393:  **Review SUBAGENT dies (spend limit, API error) = NOT review waiver — codex
395:  `codex exec --sandbox read-only -C <repo> "<prompt>"` (`which codex`,
396:  `~/.codex/auth.json`). Try CLI directly before ever proceeding unreviewed. BOTH
398:  2026-07-26: subagent hit monthly spend limit, many iterations ran unreviewed
441:must check with adversarial reviews our claims, especially when they imply
444:The existing dual-review rule covers DIFFS. This one covers the CLAIM that
447:- **State the claim, then get it reviewed, THEN write code.** If a sentence of
449:  to codex and GLM as a claim, before the edit exists. Both. A claim reviewed
664:adversarial reviewer rather than by me — i.e. each avoidable by reading two more
706:  review wrapper grepped whole log for failure token its own prompt contained,
709:- **BUDGET THE REVIEW LOOP INTO THE WORK.** Adversarial reviewer found real
743:  summary quoted to human, or fed to review agent, need ≥2 rows verified by eye
916:- **One shared selector, never re-implemented.** Every component YAML experiment adapter exposing a `setup:` block (atmosphere `Config`, ocean `OceanExperimentConfig`, sea-ice `SeaIceExperimentConfig`, …) MUST route validation + command-building + signature through `legoesm.core.setup_selector` (`MatrixRunnerSpec` + `validate_setup`/`build_matrix_command`/`setup_signature`/`require_positive_finite`). Per-component CLI differences (case flag `--only`/`--test`, exact-match `exact_prefix`, which `--levels`/`--dt`/`--days`/`--resolution` flags exist) are a `MatrixRunnerSpec`, NOT a copy-pasted `parts=[...]` builder or private `_SETUP_KEYS`/`_validate_setup`. Component-specific case-name validation goes through `known_names=` kwarg. Re-implemented selector = REJECTED in review. Factory naming: `_<component>_matrix_spec()`. FOLLOW-UP: dedup `OceanExperimentConfig`'s inline selector onto shared helper once PR #465 + shared-helper PR both land on main (inline copy predates extraction).
929:- **CRITICAL — Precision gate for comparison/skill/causal claims (do NOT be sloppy — 2026-07-20 EC-site lesson).** (a) **Harness self-check FIRST**: before reporting NEW scheme's skill vs validated baseline, run KNOWN baseline through YOUR OWN harness, confirm it reproduce baseline's established/published number (±small tol). Your harness score validated scheme wildly off — e.g. two-leaf H looked "broken" (NSE≈−1) when paper figure track obs — the HARNESS is wrong; fix BEFORE any new-scheme claim. (b) **Match reference metric EXACTLY**: metric aggregation (daily-mean vs half-hourly point-wise NSE), window (multi-year JJA vs one summer), masks (valid-forcing) ALL part of protocol. Report SAME metric reference used, plus any alternative, WITH sample N. Single-window point-wise metric is NOT skill verdict, and never rank schemes by metric punishing one scheme's known artifact (point-wise scatter/spikes) while ignoring dimension of interest (mean diurnal shape). (c) **Never let QC mask flatter one side**: drop scheme's unphysical spikes from scoring → report BOTH failures-penalized primary score AND separately-labeled plausible-only sensitivity score. (d) **Do NOT INFER causal origin — INSTRUMENT it**: "the spikes come from X" require LOGGING X and alternatives (raw vs intermediate vs final), not deduction from reading one code path; "correctly hooked up" require reading FULL path, not entry point. (e) **Label every claim CONFIRMED (evidence shown) vs PLAUSIBLE (inferred)** — adversarial review WILL refute over-confident inferences; state uncertainty up front rather than presenting verdict table a five-minute check overturn.
941:- **CRITICAL — LABEL EVERY CLAIM, AND PREFER RETRACTING EARLY.** Tag each statement **CONFIRMED** (control-passed evidence shown, instrument validated) vs **PLAUSIBLE** (inferred / single-run / uncontrolled). Chain of PLAUSIBLE steps is not CONFIRMED. Later measurement contradict earlier claim → **retract loudly and immediately in same message and in memory file** — no quiet move on, because stale confident claims get built on. Run codex adversarial review on DIAGNOSTIC TOOLING, not just model code: instruments decide what you believe, so bug there manufacture confident wrong physics conclusion. Cheap self-check before any big claim: *"what measurement would make this false, and did I run it?"* Answer no → claim PLAUSIBLE at best.
946:- **Atmospheric parameterization edits are high-risk**: run codex adversarial review, #477 truth-tier tendency validators, conservation tests, equilibrium-SCM-RCE realism harness before merge. NEW scheme must pass RCE realism gate or enter its shrink-only TODO partition with a reason.
948:- **Local hooks** (`.claude/hooks/`, wired in `settings.json`): PreToolUse blocks edit adding banned constant/saturation prefactor to `.py` (fail-open); Stop reminds (once/session) to run mandatory codex review on uncommitted numerics/physics. CI remain authoritative (`Bash` heredoc bypass hook, not CI).
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

exec
/usr/bin/bash -lc "ls /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python; sed -n '14505,14585p' packages/coupler/legoesm/driver/model_driver.py; rg -n 'def surface_reference_state' packages; sed -n '1,60p' tests/conftest.py; sed -n '1,80p' tests/land/conftest.py" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 7ms:
/glade/work/pg2328/conda-envs/legoesm-gpu/bin/python
                         or not self._is_spmd_multiprocess())
                    and _land_pack_idx_np.size >= len(_land_devs)):
                _land_pack_idx_np, _land_keep = shard_land_columns(
                    _land_pack_idx_np, len(_land_devs))
                _land_mesh = land_device_mesh(_step_mesh, _land_devs)
            # the device fallback is per process (shares differ by one
            # column), the process fallback is the same on every process
            if _land_mesh is None and (
                    len(_land_devs) > 1
                    or (jax.process_index() == 0
                        and self._is_spmd_multiprocess()
                        and _land_split is None)):
                logger.warning(
                    "land: every device solves ALL its land columns "
                    "(replicated, no speed-up from extra devices): %s",
                    "clm_ml land is not split" if _land_clm else
                    "too few land columns to give every device a share")
            _land_pack_on = (
                not _land_clm
                and (_land_keep is not None
                     or 0 < _land_pack_idx_np.size < _f_land_cols.shape[0]))
            # An all-ocean MPI rank: nothing to solve.  Its outputs are the
            # packed path's ocean-column fills (zeros, state unchanged), so it
            # costs nothing instead of solving every ocean column.
            _land_none = _land_pack_idx_np.size == 0
            _land_ncol_full = int(_f_land_cols.shape[0])
            # Owned (non-halo) columns: under MPAS MPI local cells are owned
            # first, then halo copies, which must not be counted twice.
            _vl_land = getattr(self, "_voronoi_layout", None)
            if _vl_land is not None and (
                    _land_ncol_full != int(_vl_land.partition.n_local_cells)):
                raise ValueError(
                    f"land columns ({_land_ncol_full}) are not the rank's local "
                    f"cells ({_vl_land.partition.n_local_cells}); the owned-"
                    "column mask would be wrong")
            _owned_cols = jnp.arange(_land_ncol_full) < (
                int(_vl_land.partition.n_owned_cells) if _vl_land is not None
                else _land_ncol_full)
            _land_pack_idx = jnp.asarray(_land_pack_idx_np)
            # Split over devices: the scatter-back writes the REAL slots only
            # (a padding slot repeats a real column; its result is dropped).
            _land_out_idx = (_land_pack_idx if _land_keep is None
                             else jnp.asarray(_land_pack_idx_np[_land_keep]))

            from legoesm.land.multilayer_land import (
                gather_land_columns, scatter_land_columns, scatter_cells)

            def _land_pack(tree):
                return gather_land_columns(
                    tree, _land_pack_idx, _land_ncol_full)

            if _land_pack_on:
                # The CONFIG packs too: the coupled tile carries per-column
                # arrays inside it (spatial hydraulics theta_sat/theta_r are
                # (ncol, 1) — the accel_pack A/B crashed on exactly that
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
packages/core/legoesm/core/bulk_flux.py:184:def surface_reference_state(T, z_ref, z_low=None):
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

exec
/usr/bin/bash -lc 'PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -m pytest -p no:cacheprovider tests/land/unit/test_land_screen_temperature.py tests/unit/test_land_column_split.py -q' in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
exec
/usr/bin/bash -lc "rg -n 'tree_(map|leaves)|isfinite|surface_out|surface_candidate' packages/land/legoesm/land/multilayer_land.py | tail -65; sed -n '2110,2295p' packages/land/legoesm/land/multilayer_land.py; sed -n '184,222p' packages/core/legoesm/core/bulk_flux.py; sed -n '1,160p' tests/unit/test_land_column_split.py; sed -n '14465,14505p' packages/coupler/legoesm/driver/model_driver.py" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
928:        surface_out = compute_two_leaf_canopy_fluxes(
1019:        surface_out, canopy_state_new = compute_clm_ml_canopy_fluxes(
1042:        surface_out = compute_simple_seb_fluxes(
1068:    shflx = surface_out.shflx
1074:    lhflx = surface_out.lhflx
1075:    tau_x = surface_out.tau_x
1076:    tau_y = surface_out.tau_y
1077:    G_surface = surface_out.G_soil
1091:    _fallback_ok = (surface_out.converged is not None
1092:                    and surface_out.Rn_ext is not None
1095:        _unsolved = ~jnp.asarray(surface_out.converged).reshape(-1).astype(bool)
1096:        _fallback_resid = surface_out.Rn_ext - (shflx + lhflx + G_surface)
1132:        # surface_out.G_soil = sw_net + lw_net - shflx - lhflx; swap the radiation.
1223:    #     pack.  ``surface_out.LE_soil`` carries the ground component for BOTH
1239:    _cond = surface_out.condensation
1249:    elif surface_out.LE_soil is not None:
1250:        lhflx_ground = surface_out.LE_soil
1275:    # ground charge ``surface_out.L_soil`` = (1-w) L_leaf + w L_s(T_surface).
1287:        if scheme_is_seb and surface_out.LE_snow is None:
1293:        snow_latent = (surface_out.LE_snow if layered else
1303:        if surface_out.LE_soil is None:
1305:        elif surface_out.L_soil is None:
1310:            _E_ground = lhflx_ground / surface_out.L_soil
1428:    # the canopy energy balance, so surface_out.lhflx already reflects it;
1481:        _le_soil = (None if surface_out.LE_soil is None
1482:                    else surface_out.LE_soil - _E_snow_ground * _L_s_T)
1485:            surface_out.LE_canopy, _le_soil)
1489:            surface_out.LE_canopy, surface_out.LE_soil)
1493:    # (``surface_out.LE_wet_canopy``, the fwet share of the boosted LE — a real
1504:            surface_out.LE_wet_canopy, 0.0) / _L_v_T   # kg m-2 s-1
1539:            surface_conductance=surface_out.surface_conductance,
1546:        if surface_out.surface_conductance is not None:
1548:                surface_out.surface_conductance
1718:            surface_conductance=surface_out.surface_conductance,
1800:        # change).  ``surface_out.canopy_x`` is None for every non-canopy
1802:        canopy_x=(_match(surface_out.canopy_x, state.canopy_x)
1804:                      and surface_out.canopy_x is not None) else None),
1893:            alpha_new = surface_out.albedo
1918:    stom_ratio = surface_out.stomatal_ratio
1930:    if surface_out.q_surface is not None:
1940:        q_sfc_new = jnp.where(has_snow_new, q_sat_sfc_new, surface_out.q_surface)
1954:    if _partial_cover and surface_out.q_surface is not None:
1958:        q_sfc_new = surface_out.q_surface
1959:        if surface_out.vapour_conductance is not None:
1968:                surface_out.vapour_conductance, 1e-12)
1973:        if surface_out.gpp is not None:
1975:            gpp_override = surface_out.gpp
2006:        response_T_sfc = (surface_out.T_canopy_air
2007:                          if surface_out.T_canopy_air is not None
2008:                          else surface_out.T_surface)
2009:        response_lw_up = surface_out.lw_up          # canopy LW_out
2020:        T_rad=surface_out.T_surface,
2022:        emissivity=surface_out.emissivity,
2023:        z0=surface_out.z0,
2058:        state, new_state, response, surface_out, forcing, config, ncol,
2061:    surface_out = surface_out._replace(
2067:        surface_out = surface_out._replace(
2074:    return new_state, response, carbon_state_new, surface_out
2121:def _hold_unsolved_columns(state, new_state, response, surface_out, forcing,
2192:    def _any_bad(tree, test=lambda a: ~jnp.isfinite(a)):
2209:    unsolved = (jnp.zeros(ncol, dtype=bool) if surface_out.converged is None
2210:                else ~jnp.asarray(surface_out.converged).reshape(-1).astype(bool))
2442:        # is the honest sentinel — the canopy tests it with ``isfinite`` and
2540:    return jax.tree_util.tree_map(
2554:    return jax.tree_util.tree_map(
    return state._replace(
        snow_ice_layers=pack.swe_ice.astype(dt_),
        snow_liq_layers=pack.swe_liq.astype(dt_),
        snow_T_layers=pack.T.astype(dt_),
        snow_rho_layers=pack.density.astype(dt_))


# ---------------------------------------------------------------------------
# Per-column containment: an unsolved column must not be able to kill the model
# ---------------------------------------------------------------------------

def _hold_unsolved_columns(state, new_state, response, surface_out, forcing,
                           config, ncol, carbon_old=None, carbon_new=None,
                           fallback_ok=False):
    """Contain columns the land model failed to solve, and say so.

    Two things make a column suspect:

    * the surface scheme's iterative closure did not reach a root
      (``SurfaceFluxOutput.converged is False``), so its fluxes are the
      scheme's fallback rather than a solution of the surface energy balance; or
    * some leaf of the new state or of the tile response came back non-finite.

    A NON-FINITE column is REVERTED (held): its state (soil, snow, ponding,
    carbon, canopy cache) goes back to the start of the step and its tile
    response reports no turbulent exchange and no runoff, with a surface
    temperature equal to the (finite) previous top-soil temperature and a
    surface humidity equal to the air's.

    A FINITE UNSOLVED column, when ``fallback_ok`` (the caller closed its energy
    into sensible heat: two-leaf canopy, no elevation bands), is ACCEPTED, the
    way CLM CanopyFluxes proceeds past its iteration cap: soil heat and water,
    snow and the response advance with the fallback fluxes, which close
    ``Rn_ext = SH + LE + G`` exactly.  Its carbon pools are kept and its CO2
    flux is zero (the fallback GPP is not physics), and its canopy warm-start
    cache keeps the previous root, so the next step retries from it.  It is
    reverted instead if accepting it would move the top soil layer by more than
    ``FALLBACK_MAX_TOP_SOIL_CHANGE_K`` or report a turbulent flux beyond
    ``FALLBACK_MAX_FLUX_W_M2``.  Without ``fallback_ok`` every unsolved column
    is reverted, as before.

    Why the revert exists.  Before it, ONE column of 2562 that went non-finite
    reached the atmosphere through the land skin temperature and, via the
    dynamical core's global mass fixer, made every column of the model
    non-finite within a single step.  Why unsolved columns are no longer
    reverted: a reverted column restarts from the identical state, tends to
    fail again, and was measured frozen for 40 consecutive land steps (20 h) at
    its 342 K midday skin, with its energy discarded every step.

    The canopy warm-start cache (``canopy_x``) is NaN by design where a column
    has never converged ("cold start"), so NaN there is not a failure; Inf is.

    On the MPAS lane with ``land_beta_soil`` (production) the atmosphere
    receives the land's own sensible and latent fluxes (``shflx_land`` /
    ``lhflx_land``), so what this returns is what it exchanges with; without
    that switch it recomputes its fluxes from the returned skin state.  This
    runs inside the jitted step, where a host print is not available on a
    GPU-only runtime; the caller surfaces the counts.

    Returns
    -------
    held_state, held_response, held_carbon,
    held_mask (ncol,) bool, n_held () int32         -- reverted columns
    fallback_mask (ncol,) bool, n_fallback () int32 -- accepted unsolved columns
    rejected_mask (ncol,) bool, n_rejected () int32 -- finite unsolved columns
        the guards reverted (already inside held_mask / n_held)
    """
    def _is_float_leaf(leaf):
        # NOT ``dtype.kind in "fc"``: bfloat16 is an extension dtype whose kind
        # is "V", so a kind test silently skips it and a bfloat16 NaN would go
        # unheld (reproduced by review).  ``jnp.issubdtype(..., jnp.inexact)``
        # covers every float and complex dtype JAX supports.
        dtype = getattr(leaf, "dtype", None)
        return dtype is not None and jnp.issubdtype(dtype, jnp.inexact)

    def _col_reduce(leaf, test):
        """Per-column ``any(test)`` over one (ncol, ...) array leaf."""
        arr = jnp.asarray(leaf)
        if arr.ndim == 0 or arr.shape[0] != ncol:
            return jnp.zeros(ncol, dtype=bool)
        return jnp.any(test(arr.reshape(ncol, -1)), axis=-1)

    def _any_bad(tree, test=lambda a: ~jnp.isfinite(a)):
        # Over tree LEAVES, so a nested carrier (the CLM-ML canopy state) is
        # inspected too, not just the top-level fields.
        acc = jnp.zeros(ncol, dtype=bool)
        for leaf in jax.tree.leaves(tree):
            if _is_float_leaf(leaf):
                acc = acc | _col_reduce(leaf, test)
        return acc

    nonfinite = _any_bad(response)
    for name in new_state._fields:
        field = getattr(new_state, name)
        if name == "canopy_x":
            nonfinite = nonfinite | _any_bad(field, jnp.isinf)
        else:
            nonfinite = nonfinite | _any_bad(field)

    unsolved = (jnp.zeros(ncol, dtype=bool) if surface_out.converged is None
                else ~jnp.asarray(surface_out.converged).reshape(-1).astype(bool))
    if fallback_ok:
        dT_top = jnp.abs(new_state.T_soil[:, 0] - state.T_soil[:, 0])
        guard_fail = ((dT_top > FALLBACK_MAX_TOP_SOIL_CHANGE_K)
                      | (jnp.abs(response.shflx) > FALLBACK_MAX_FLUX_W_M2)
                      | (jnp.abs(response.lhflx) > FALLBACK_MAX_FLUX_W_M2))
        candidate = unsolved & ~nonfinite
        fallback = candidate & ~guard_fail
        rejected = candidate & guard_fail
    else:
        fallback = jnp.zeros(ncol, dtype=bool)
        rejected = jnp.zeros(ncol, dtype=bool)
    def _revert_leaf(mask):
        def _rl(new_leaf, old_leaf):
            if not _is_float_leaf(new_leaf) or old_leaf is None:
                return new_leaf
            if new_leaf.ndim == 0 or new_leaf.shape[0] != ncol:
                return new_leaf
            old_arr = jnp.asarray(old_leaf)
            if old_arr.shape != new_leaf.shape:
                return new_leaf
            m = mask.reshape((ncol,) + (1,) * (new_leaf.ndim - 1))
            return jnp.where(m, old_arr, new_leaf)
        return _rl

    def _hold_field(new_field, old_field, mask):
        # Per FIELD rather than one tree.map over the whole state: a field that
        # is None on one side and an array on the other (an optional store
        # switched on mid-run) makes the two states different pytrees, which a
        # single tree.map cannot walk.  Inside a field we DO recurse, so a
        # nested carrier (the CLM-ML canopy state) is held as well.
        if new_field is None or old_field is None:
            return new_field
        if _is_float_leaf(new_field):
            return _revert_leaf(mask)(new_field, old_field)
        try:
            return jax.tree.map(_revert_leaf(mask), new_field, old_field)
        except (ValueError, TypeError):
            # Structurally different this step (e.g. a carrier rebuilt from
            # scratch): nothing to revert to, so leave it. Loud rather than
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
def surface_reference_state(T, z_ref, z_low=None):
    """Pair model-level height with dry-adiabatic surface-referenced air T.

    Observed forcing (z_low=None) already follows its supplied reference
    convention. Model-level forcing must move BOTH height and temperature.
    """
    if z_low is None:
        return T, z_ref
    # Sensible heat is positive UPWARD: H = rho cp Ch U (T_sfc - T_air).
    # Bringing air down warms it (+g z/cp), reducing upward H; land loses H
    # through -H in its energy budget. Height simultaneously reduces exchange.
    return T + (constants.g / constants.c_pd) * z_low, z_low


def apply_gustiness(u: jax.Array, v: jax.Array, gustiness: float) -> jax.Array:
    """Effective surface wind with a sub-grid convective gustiness floor.

    ``|U|_eff = sqrt(u^2 + v^2 + u_gust^2)``.  The resolved grid-mean wind misses
    sub-grid wind variability — boundary-layer convective gustiness — which in
    calm/convective regions (the tropics, where the mean wind is light but deep
    convection drives gusts) is the dominant contributor to the air-sea latent
    and sensible flux.  Omitting it (or using a ~1 m/s numerical floor) yields
    ~1/3 of the realistic surface evaporation at light winds, starving the
    hydrological cycle (measured: coupled hfls ~35 vs ~80 W/m^2 at a 1.9 m/s
    mean surface wind).  Same algebraic form as the numerical wind floor
    ``sqrt(u^2+v^2+U_min^2)`` — a larger, physically-motivated ``u_gust``
    subsumes it.

    The floor lives INSIDE the single ``sqrt`` (not a two-stage
    ``sqrt(sqrt(u^2+v^2)^2 + g^2)``): with ``gustiness > 0`` the argument is
    strictly positive everywhere, so the gradient is finite even at exact calm
    wind ``u = v = 0`` (a bare ``sqrt(u^2+v^2)`` would give a NaN gradient
    there).  AD-safe for the differentiable coupler.

    Parameters
    ----------
    u, v : array
        Resolved grid-mean surface wind components [m/s].  A caller holding only
        the wind SPEED magnitude ``s`` passes ``apply_gustiness(s, 0.0, gust)``
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
    return ((_State(T, wet),) + (f,) * 7 + (held,) + (f, f, f)
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
            # (0 * NaN would contaminate the f_land blends).  clm_ml keeps
            # the full-grid path: its 1-based canopy arrays and GridInfo do
            # not repack.
            _land_pack_idx_np = np.nonzero(
                np.asarray(self._f_land).reshape(-1) > 0.0)[0]
            if _is_mpas_cell_partitioned(self):
                # Owned columns only.  After every land call the land halo
                # exchange below overwrites the halo rows of every consumed
                # output with the owner's, so solving the halo copies was
                # discarded work (about a third of the land solve at 8
                # nodes).  Halo rows of the land STATE stay frozen: what is
                # derived from them (soil beta) is exchanged with the outputs,
                # and checkpoints, digests and the CMOR feed take owned rows.
                _land_pack_idx_np = _land_pack_idx_np[
                    _land_pack_idx_np
                    < int(self._voronoi_layout.partition.n_owned_cells)]
            # Multi-controller SPMD (the FV3 duo lane): every process holds
            # the whole globe's land, so each solves only its share of the
            # land columns and one gather per land call rebuilds the full
            # copy.  All processes take this branch (config + process count
            # only), so the gather is reached by every one of them.
            _land_split = None
            if (self._is_spmd_multiprocess()
                    and str(getattr(cfg, "land_surface_scheme", "")) != "clm_ml"
                    and _land_pack_idx_np.size >= jax.process_count()):
                _land_pack_idx_np, _land_split = split_land_columns(
                    _land_pack_idx_np, jax.process_count(),
                    jax.process_index())
            # One process driving several devices (the duo's face-shard or
            # window layouts on one host): this process's land columns are
            # split over its devices the same way and solved under a
            # shard_map, one equal shard per device.
            _land_keep = _land_mesh = None
            _step_mesh = getattr(self.model, "step_spmd_mesh", None)
            _land_devs = ([d for d in _step_mesh.devices.flat
                           if d.process_index == jax.process_index()]
                          if _step_mesh is not None else [])
            _land_clm = str(getattr(cfg, "land_surface_scheme", "")) == "clm_ml"
            if (len(_land_devs) > 1 and not _land_clm
                    and (_land_split is not None
                         or not self._is_spmd_multiprocess())

exec
/usr/bin/bash -lc "sed -n '2190,2210p' packages/land/legoesm/land/multilayer_land.py; sed -n '184,201p' packages/core/legoesm/core/bulk_flux.py; sed -n '144,174p' packages/coupler/legoesm/driver/model_driver.py; rg -n 'def scatter_cells' packages/land/legoesm/land/multilayer_land.py" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
        return jnp.any(test(arr.reshape(ncol, -1)), axis=-1)

    def _any_bad(tree, test=lambda a: ~jnp.isfinite(a)):
        # Over tree LEAVES, so a nested carrier (the CLM-ML canopy state) is
        # inspected too, not just the top-level fields.
        acc = jnp.zeros(ncol, dtype=bool)
        for leaf in jax.tree.leaves(tree):
            if _is_float_leaf(leaf):
                acc = acc | _col_reduce(leaf, test)
        return acc

    nonfinite = _any_bad(response)
    for name in new_state._fields:
        field = getattr(new_state, name)
        if name == "canopy_x":
            nonfinite = nonfinite | _any_bad(field, jnp.isinf)
        else:
            nonfinite = nonfinite | _any_bad(field)

    unsolved = (jnp.zeros(ncol, dtype=bool) if surface_out.converged is None
                else ~jnp.asarray(surface_out.converged).reshape(-1).astype(bool))
def surface_reference_state(T, z_ref, z_low=None):
    """Pair model-level height with dry-adiabatic surface-referenced air T.

    Observed forcing (z_low=None) already follows its supplied reference
    convention. Model-level forcing must move BOTH height and temperature.
    """
    if z_low is None:
        return T, z_ref
    # Sensible heat is positive UPWARD: H = rho cp Ch U (T_sfc - T_air).
    # Bringing air down warms it (+g z/cp), reducing upward H; land loses H
    # through -H in its energy budget. Height simultaneously reduces exchange.
    return T + (constants.g / constants.c_pd) * z_low, z_low


def apply_gustiness(u: jax.Array, v: jax.Array, gustiness: float) -> jax.Array:
    """Effective surface wind with a sub-grid convective gustiness floor.

    ``|U|_eff = sqrt(u^2 + v^2 + u_gust^2)``.  The resolved grid-mean wind misses
    glob = _gather(faces, layout, root_only=root_only)    # (6, n, n, ...)
    return glob.reshape((6 * n_tile * n_tile,) + trailing)


def _scatter_voronoi_columns(arr, partition):
    """Scatter a global per-column array ``(nCells_global, ...)`` to this
    rank's LOCAL cells (owned + halo), in the same order the rank-local MPAS
    physics columns use (#1321).

    The Voronoi analogue of :func:`_scatter_flat_columns`.  Halo columns are
    included, not trimmed, so every per-column land array keeps the same
    leading length as the rank-local atmospheric state and no consumer needs a
    special case.  Integrating them is redundant but not wrong: the land step
    is column-local (no lateral soil coupling) and a halo column sees the same
    exchanged atmospheric forcing as its owner, so it tracks the owner exactly.
    ``gather_voronoi_field`` keeps only the owned prefix on the way out.
    """
    from legoesm.parallel.voronoi_partition import scatter_to_local
    return scatter_to_local(arr, partition, "cell")


def _scatter_1based_voronoi_columns(arr, partition):
    """CLM 1-based ``(nCells_global + 1, ...)`` -> ``(n_local_cells + 1, ...)``.

    Same strip / scatter / re-prepend composition as
    :func:`_scatter_1based_columns`, over the Voronoi cell partition.
    """
    body = _scatter_voronoi_columns(arr[1:], partition)
    return jnp.concatenate([arr[:1], body], axis=0)


2563:def scatter_cells(v, idx, ncol_full: int):

exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python - <<'PY'
import os
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
import jax.numpy as j
from legoesm.land.canopy.stability import monin_obukhov_stability, _temperature_humidity_relation
for n in (1,10):
    u, rah, _, _, zeta = monin_obukhov_stability(*map(j.asarray,(4.,270.,271.,262.,.002,.003,55.,1.5)),n_iters=n,zeta_cap_width=.05)
    ch_flux=1/(u*rah)
    ch_screen=_temperature_humidity_relation(j.asarray(55.),55./zeta,j.asarray(1.5))
    print(n, 'zeta',float(zeta),'ch_flux',float(ch_flux),'ch_screen',float(ch_screen),flush=True)
PY" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 4239ms:
1 zeta 0.4999999999999997 ch_flux 0.06799154330636877 ch_screen 0.06799154330636878
10 zeta 0.4999999999999999 ch_flux 0.06799154330636878 ch_screen 0.06799154330636877

 exited 2 in 34762ms:
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
configfile: pyproject.toml
plugins: zarr-3.4.0, timeout-2.4.0, jaxtyping-0.3.11, xdist-3.8.0
collected 6 items / 1 error

==================================== ERRORS ====================================
_______ ERROR collecting tests/land/unit/test_land_screen_temperature.py _______
ImportError while importing test module '/glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193/tests/land/unit/test_land_screen_temperature.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/glade/work/pg2328/conda-envs/legoesm-gpu/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/land/unit/test_land_screen_temperature.py:15: in <module>
    from legoesm.land.canopy.stability import screen_level_temperature
E   ImportError: cannot import name 'screen_level_temperature' from 'legoesm.land.canopy.stability' (/glade/u/home/pg2328/legoesm/packages/land/legoesm/land/canopy/stability.py)
=========================== short test summary info ============================
ERROR tests/land/unit/test_land_screen_temperature.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 13.61s ===============================

exec
/usr/bin/bash -lc "sed -n '2560,2578p' packages/land/legoesm/land/multilayer_land.py; nl -ba packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py | sed -n '726,739p'; nl -ba tests/land/unit/test_land_screen_temperature.py | sed -n '25,45p'; sed -n '510,550p' packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py; git status --porcelain" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 38ms:
        full_tree, packed_tree)


def scatter_cells(v, idx, ncol_full: int):
    """Packed per-column vector -> full-grid cells, zero fill (never NaN:
    the consumers multiply by f_land, and 0 * NaN would contaminate them)."""
    return jnp.zeros((ncol_full,), v.dtype).at[idx].set(v.reshape(-1))
   726	        canopy_x=x_conv,
   727	        canopy_resid_sq=resid_sq,
   728	        canopy_resid_rel=resid_rel,
   729	        canopy_hit_cap=hit_cap,
   730	        Tf_Sun=Tf_Sun,
   731	        Tf_Sh=Tf_Sh,
   732	        T_canopy_air=Tc_cvg,
   733	        T_2m=screen_level_temperature(Ta, Tc_cvg, z_ref - displa,
   734	                                      fluxes_per_col["zeta"], z0m),
   735	        gs_Sun=gs_Sun,
   736	        gs_Sh=gs_Sh,
   737	        n_iters=n_iters,
   738	        # Whether the Newton closure actually reached a root on this column.
   739	        # False means the fluxes above are the cold-state fallback, not a
    25	        x = (1.0 - 16.0 * zeta) ** 0.25
    26	        return 2.0 * math.log((1.0 + x * x) / 2.0)
    27	    a, b, c, d = 1.0, 0.667, 5.0, 0.35   # BH91 as published (b rounded)
    28	    return (-(1.0 + 2.0 * a * zeta / 3.0) ** 1.5
    29	            - b * (zeta - c / d) * math.exp(-d * zeta) - b * c / d + 1.0)
    30	
    31	
    32	def _oracle(theta, Tc, zl, zeta, z0h, zs=2.0):
    33	    L = zl / zeta
    34	    F = lambda z: math.log(z / z0h) - _psi_h(z / L) + _psi_h(z0h / L)
    35	    return Tc + (theta - Tc) * F(zs + z0h) / F(zl)
    36	
    37	
    38	@pytest.mark.parametrize("zeta", [-0.3, -0.05, 0.05, 0.3])
    39	@pytest.mark.parametrize("z0h", [0.05, 1.5])
    40	def test_profile_matches_independent_most(zeta, z0h):
    41	    theta, Tc, zl = 270.0, 262.0, 55.0
    42	    got = float(screen_level_temperature(jnp.asarray(theta), jnp.asarray(Tc),
    43	                                         jnp.asarray(zl), jnp.asarray(zeta),
    44	                                         jnp.asarray(z0h)))
    45	    np.testing.assert_allclose(got, _oracle(theta, Tc, zl, zeta, z0h),

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
?? docs/wb/reviews/2026-11-22_land_tas_r1_codex.md
?? docs/wb/reviews/2026-11-22_land_tas_r1_glm.md
?? docs/wb/reviews/2026-11-22_land_tas_r1_prompt.md

exec
/usr/bin/bash -lc "ls packages; sed -n '70,115p' tests/conftest.py; sed -n '1,55p' packages/core/legoesm/__init__.py; ls src/legoesm" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
atmosphere
core
coupler
ice
land
ml
ocean
tools
    return any(v in os.environ for v in _MPI_RANK_ENV)


def _requested_cap() -> int | None:
    """The user's explicit ``LEGOESM_TEST_CPU_CAP``, or None when unset/garbage.

    A value <= 0 disables capping entirely and is returned as 0.
    """
    raw = os.environ.get("LEGOESM_TEST_CPU_CAP")
    if raw is None:
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

sed: can't read packages/core/legoesm/__init__.py: No such file or directory
cli.py
config.py
dycore_factory.py
experiment_registry.py
scaling_preflight.py
supported_matrix.py

codex
Packed scatter and the empty-rank path fill missing T2m values with zero. That can produce 0 K on land halo cells, though CMIP output keeps only owned cells. The first test attempt imported the base checkout; I’m rerunning against this worktree.
exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python - <<'PY'
import sys, pathlib
sys.path[:0] = [str(p.resolve()) for p in pathlib.Path('packages').iterdir() if p.is_dir()]
import pytest
raise SystemExit(pytest.main(['-p','no:cacheprovider','tests/land/unit/test_land_screen_temperature.py','tests/unit/test_land_column_split.py','-q']))
PY" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python - <<'PY'
import os,sys,pathlib
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
sys.path[:0] = [str(p.resolve()) for p in pathlib.Path('packages').iterdir() if p.is_dir()]
import jax.numpy as j
from legoesm.land.canopy.stability import monin_obukhov_stability, _temperature_humidity_relation
for tc in (269.,270.,271.,280.):
 for n in (1,10):
    u, rah, _, _, zeta = monin_obukhov_stability(*map(j.asarray,(4.,270.,271.,tc,.002,.002,55.,1.5)),n_iters=n,zeta_cap_width=.05)
    ch_flux=1/(u*rah)
    ch_screen=_temperature_humidity_relation(j.asarray(55.),55./zeta,j.asarray(1.5))
    print(tc,n,float(zeta),float(ch_screen/ch_flux-1),flush=True)
PY" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 6933ms:
269.0 1 0.49942342250033217 0.00041966152865291484
269.0 10 0.49942012345169323 0.0
270.0 1 1e-06 4.440892098500626e-16
270.0 10 1e-06 4.440892098500626e-16
271.0 1 -0.4336290671531375 -0.004312333979936467
271.0 10 -0.3865118437862066 -1.20496326250219e-08
280.0 1 -4.557998242717087 0.011470502083841039
280.0 10 -1.8389718441490102 0.00020625945028673875

exec
/usr/bin/bash -lc "rg -n 'n_owned|gather_voronoi_field|_mpas_surface_temperatures' packages/coupler/legoesm/driver/model_driver.py | tail -35; nl -ba packages/land/legoesm/land/canopy/stability.py | sed -n '357,390p'; nl -ba packages/coupler/legoesm/driver/model_driver.py | sed -n '14737,14769p'; nl -ba packages/coupler/legoesm/driver/model_driver.py | sed -n '15590,15605p'" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 0ms:
5871:                _cmip_phis = gather_voronoi_field(
5873:                _cmip_fland = gather_voronoi_field(
6578:                    return a[:part.n_owned_cells]
6580:                    return a[:part.n_owned_cells + 1]
6582:                    return a[:part.n_owned_edges]
6837:                vlayout.partition.n_owned_cells,
6839:                - vlayout.partition.n_owned_cells,
8108:                from legoesm.parallel.voronoi_mpi import gather_voronoi_field
8110:                u_d = gather_voronoi_field(s.u.data, part, "edge")
8111:                T_d = gather_voronoi_field(s.T.data, part, "cell")
8112:                ps_d = gather_voronoi_field(s.p_s.data, part, "cell")
8113:                phis_d = gather_voronoi_field(s.phis.data, part, "cell")
8115:                    _k: gather_voronoi_field(s.tracers[_k].data, part, "cell")
8148:                            _flat = gather_voronoi_field(
8155:                            ps_d_carry[_name] = gather_voronoi_field(
8170:                            return gather_voronoi_field(x, _p, "cell")
8173:                                [x[:1], gather_voronoi_field(x[1:], _p, "cell")],
8196:                        _skin_save = gather_voronoi_field(
8201:                        gather_voronoi_field(jnp.asarray(_a).reshape(-1),
8219:                    return gather_voronoi_field(x, _p, "cell")
8229:                             gather_voronoi_field(jnp.asarray(_v).reshape(-1),
10610:                int(vl.partition.n_owned_cells), op=_MPI.SUM)
10892:    def _mpas_surface_temperatures(self, day, diag, u_east, v_north):
11005:        tas, ts, sic, sst = ModelDriver._mpas_surface_temperatures(
11261:        its OWNED rows (``partition.local_cells[:n_owned_cells]`` — the
11340:        n_owned = 0
11360:            n_owned = int(part.n_owned_cells)
11361:            owned_ids = np.asarray(part.local_cells[:n_owned])
11435:                owned[name] = np.asarray(val[:n_owned])
11491:        # this measurable, pack the agreed names into one (n_owned, -1) buffer
14480:                    < int(self._voronoi_layout.partition.n_owned_cells)]
14541:                int(_vl_land.partition.n_owned_cells) if _vl_land is not None
16796:                            gather_voronoi_field,
16799:                        _led_rates = np.asarray(gather_voronoi_field(
16803:                            _area_w_led = np.asarray(gather_voronoi_field(
   357	    z0h, obu, um = carry[0], carry[1], carry[2]
   358	    Ta, Tv_atm, q_atm, _q2, zldis, z0m, dq, dth, ur = (
   359	        forcing[0], forcing[1], forcing[2], forcing[3],
   360	        forcing[4], forcing[5], forcing[6], forcing[7], forcing[8],
   361	    )
   362	
   363	    ustar = _friction_velocity(zldis, z0m, obu, um)
   364	    ch    = _temperature_humidity_relation(zldis, obu, z0h)
   365	    tstar = ch * dth
   366	    qstar = ch * dq
   367	    thvstar = tstar * (1.0 + _VIRT_T_COEF * q_atm) + _VIRT_T_COEF * Ta * qstar
   368	
   369	    zeta  = zldis * constants.kappa_von_karman * constants.g * thvstar / (ustar**2 * Tv_atm)
   370	
   371	    zeta_stable = _cap_stable_zeta(zeta, zeta_cap_width)
   372	    um_stable   = jnp.maximum(ur, 0.1)                     # coeff-ok: 0.1 m/s wind floor
   373	    zeta_unstable = jnp.clip(zeta, -100.0, -_ZETA_NEUTRAL_FLOOR)  # coeff-ok: very-unstable bound on ζ
   374	    # Floor the cbrt argument to a POSITIVE value, not 0: cbrt'(0)=inf and the
   375	    # maximum's subgradient is 0 below the clamp, so cbrt(maximum(x, 0)) gives
   376	    # 0*inf = NaN in the reverse-mode gradient whenever x<=0 (the stable regime,
   377	    # where this unstable-branch wc is discarded).  cbrt(1e-12)~1e-4 m/s ~ 0.
   378	    wc_unstable = jnp.cbrt(jnp.maximum(
   379	        -constants.g * ustar * thvstar * _CONV_BDY_HEIGHT / Tv_atm, _MOST_ARG_FLOOR))
   380	    um_unstable = jnp.sqrt(ur**2 + wc_unstable**2)
   381	
   382	    is_stable = zeta >= 0.0
   383	    zeta_new = jnp.where(is_stable, zeta_stable, zeta_unstable)
   384	    um_new   = jnp.where(is_stable, um_stable,   um_unstable)
   385	    obu_new  = zldis / zeta_new
   386	
   387	    new_carry = jnp.array([z0h, obu_new, um_new])
   388	    output    = jnp.array([ustar, tstar, qstar, thvstar, ch, zeta_new])
   389	    return new_carry, output
   390	
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
 14747	                                    _taumag, _held_f, resp.T_rad, _emis,
 14748	                                    _t2m))
 14749	                            + (_n_held, _n_held_land, _n_fb_land,
 14750	                               _n_fb_rej, _n_unconv_land, _n_iter_max))
 14751	                return (new_state, resp.T_sfc, resp.albedo, resp.q_surface,
 14752	                        resp.shflx, resp.lhflx, resp.surface_mass_flux,
 14753	                        _taumag, _held_f, resp.T_rad, _emis, _t2m,
 14754	                        _n_held, _n_held_land, _n_fb_land, _n_fb_rej,
 14755	                        _n_unconv_land, _n_iter_max)
 14756	              if not _land_none:
 14757	                  return _land_step
 14758	
 14759	              @jax.jit
 14760	              def _land_step_none(land_state, a2s, doy, lai_doy, lai_year):
 14761	                  # Same output structure and dtypes as the solve (traced
 14762	                  # shape-only, never run), so the land halo exchange packs
 14763	                  # the same message on every rank.
 14764	                  out = jax.eval_shape(
 14765	                      _land_step, land_state, a2s, doy, lai_doy, lai_year)
 14766	                  return (land_state,) + tuple(
 14767	                      jnp.zeros(o.shape, o.dtype) for o in out[1:])
 14768	              return _land_step_none
 14769	
 15590	                    "  Coupling window: the checkpoint carries no coupling "
 15591	                    "window (written before it was persisted); the pending "
 15592	                    "steps of the previous link are lost.")
 15593	        if _land_ml_on and _lw_ck:
 15594	            (_land_a2s_sum, _land_a2s_n, _land_cz_wsum, _land_w_sum,
 15595	             (_lw_T, _lw_alb, _lw_qlast, _lw_qs, _lw_sh, _lw_lh,
 15596	              _lw_ev, _lw_beta, _lw_Trad, _lw_emis)) = land_window_from_checkpoint(
 15597	                _lw_ck, _LAND_K,
 15598	                int(jnp.asarray(self.state.p_s.data).reshape(-1).shape[0]))
 15599	            if _lw_T is not None:
 15600	                _land_T_skin = self._land_T_skin_last = _lw_T
 15601	                _land_albedo_cells = _lw_alb
 15602	                self._land_qsfc_last = _lw_qlast
 15603	                # A checkpoint written before the radiating pair was
 15604	                # persisted: seed it with the held skin T and the radiation's
 15605	                # own emissivity (vote V2a, 3-0) until the first land call.

exec
/usr/bin/bash -lc "rg -n 'def psi_h|0.667|beljaars' packages/core/legoesm/core/monin_obukhov.py packages/atmosphere/legoesm/atmosphere/physics/turbulence/* 2>/dev/null | head -20; git diff --check 8750d0005d" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 18ms:
packages/atmosphere/legoesm/atmosphere/physics/turbulence/mynnedmf_wrf.py:215:_SA_EXP = 0.667
packages/atmosphere/legoesm/atmosphere/physics/turbulence/config.py:288:    # "beljaars_holtslag1991" | "grachev2007_sheba" | "gryanik2020".  On

codex
The stability mismatch also occurs with the default iteration count, so the new profile does not use exactly the Obukhov length behind the flux coefficient. I found no new state or coupling dependency on T2m. The focused tests have passed the profile checks and are still running the canopy cases.
exec
/usr/bin/bash -lc "ps -u pg2328 -o pid,etime,pcpu,args | rg 'pytest|python -' | tail -10" in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 95ms:
1840808       02:41  1.6 codex exec --dangerously-bypass-approvals-and-sandbox -C /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193 ADVERSARIAL REVIEW (read-only; do not edit files). Repo worktree: /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193, branch fix/land-tas-t-ref2m (base 8750d0005d). CLM5 source: /glade/derecho/scratch/pg2328/cesm_src/CLM5/src/biogeophys.  Change: CMIP `tas` land leg on the production MPAS lane (driver/model_driver.py _mpas_surface_temperatures -> diagnostics.py _tas_2m) used the OCEAN bulk law (large_yeager MOST, Charnock roughness, no displacement) anchored on the land T_sfc (= two-leaf canopy-air temperature) from the lowest model level. Fix: the two-leaf canopy now exports CLM5 t_ref2m (CanopyFluxesMod.F90:1182, FrictionVelocityMod.F90:636-652) computed with the land's own z0h(=z0m), displacement, reference height and converged Obukhov length (stability.screen_level_temperature), carried through the land step tuple (_LAND_STEP_N_CELL 10->11, NaN where held / scheme has none) to the driver (self._land_T2m_last) and into _tas_2m(T2m_land=...), where finite entries replace the ocean-law land profile. Diagnostic only: no model state, flux or coupling value changes.  Check: (1) formula fidelity vs CLM5 (frame of theta_ref vs thm, zldis, obu consistency with the flux solve, z0h), (2) the stable side uses the model's BH91 forms (documented departure), (3) driver plumbing: tuple order at every producer/consumer incl. packed scatter path, _land_step_none, gather_land_step, the multi-rank halo block, restart; (4) tests can fail (non-vacuous), oracle independent; (5) differentiability; (6) anything that silently changes non-diagnostic behaviour.  VOTE on these detail choices (answer A or B each, one line why): V1 frame: A = publish CLM t_ref2m verbatim (no 2 m potential-to-temperature step, CLM does not apply it); B = subtract g/c_p*2 (~0.02 K) like the existing MOST/sea legs of _tas_2m. V2 missing value (held column, non-two-leaf scheme, before first land call): A = keep the existing ocean-law land profile for that column; B = lowest-model-level temperature. V3 restart/halo: A = accept that T2m is not in the checkpoint land-window payload nor halo-exchanged (first land interval after restart and halo cells use the V2 fallback / rank-local value), documented; B = add it to the payload and the halo exchange now.  Report <=350 words: findings ranked (file:line), verdict CLEAN or NOT CLEAN, votes V1-V3.  DIFF: diff --git a/packages/coupler/legoesm/driver/diagnostics.py b/packages/coupler/legoesm/driver/diagnostics.py index 8003d760c3..b06f536cc3 100644 --- a/packages/coupler/legoesm/driver/diagnostics.py +++ b/packages/coupler/legoesm/driver/diagnostics.py @@ -806,7 +806,8 @@ class DiagnosticCollector:          return f_lo + alpha * (f_hi - f_lo)        def _tas_2m(self, state, q_v, sst, sic, T_ice, u_low=None, v_low=None, -                *, T_land=None, q_land=None, land_fraction=None): +                *, T_land=None, q_land=None, land_fraction=None, +                T2m_land=None):          """2 m air temperature for CMIP ``tas`` from the MOST surface-layer          similarity profile (interpolate the lowest model level down to 2 m).          Returns the lowest-level T when the surface inputs (SST / sigma) are @@ -821,6 +822,14 @@ class DiagnosticCollector:          land branch uses the same MOST profile anchored on the land skin; do          not read the result as a sub-canopy screen temperature.   +        ``T2m_land`` (keyword-only, optional) is the land scheme's OWN 2 m air +        temperature (two-leaf canopy: CLM5 ``t_ref2m`` from its roughness, +        displacement, Obukhov length and canopy-air node).  Where it is finite +        it REPLACES the profile above on the land leg, which runs the ocean +        bulk law (Charnock roughness, no displacement) over the land skin. +        Non-finite entries (held columns, schemes that solve none) keep that +        profile. +          ``u_low`` / ``v_low`` override the lowest-level winds (lets the MPAS          path pass the reconstructed CELL winds from          ``reconstruct_cell_velocity`` instead of the edge-normal ``state.u``, @@ -901,6 +910,9 @@ class DiagnosticCollector:          q_sfc_land = (saturation_mixing_ratio(T_land, p_s) if q_land is None                        else q_land)          T_2m_land = _profile_to_2m(T_land, q_sfc_land, ocean=False) +        if T2m_land is not None: +            T2m_land = jnp.asarray(T2m_land, dtype=T_2m_land.dtype) +            T_2m_land = jnp.where(jnp.isfinite(T2m_land), T2m_land, T_2m_land)          T_2m_sea = _profile_to_2m(T_sfc_sea, q_sfc_sea, ocean=True)          f_land = jnp.clip(jnp.asarray(land_fraction, dtype=T_2m_land.dtype),                            0.0, 1.0) diff --git a/packages/coupler/legoesm/driver/model_driver.py b/packages/coupler/legoesm/driver/model_driver.py index 656bda453e..37c457d518 100644 --- a/packages/coupler/legoesm/driver/model_driver.py +++ b/packages/coupler/legoesm/driver/model_driver.py @@ -297,7 +297,7 @@ def land_radiating_surface(T_sea, f_land, E_sea, T_rad_land, eps_land):    # Per-cell fields of the MPAS land step tuple, after the state: T_sfc, albedo,  # q_surface, shflx, lhflx, surface_mass_flux, stress, held, T_rad, emissivity. -_LAND_STEP_N_CELL = 10 +_LAND_STEP_N_CELL = 11      def gather_land_step(run, ncol, padded, proc, allgather): @@ -10930,6 +10930,7 @@ class ModelDriver:                  # published land tas is not the model's land at all.                  _tas_land_T = getattr(self, "_land_T_skin_last", None)                  _tas_land_q = getattr(self, "_land_qsfc_last", None) +                _tas_land_T2m = getattr(self, "_land_T2m_last", None)                  _tas_f_land = getattr(self, "_f_land", None)                  _tas_kw = {}                  if _tas_land_T is not None and _tas_f_land is not None: @@ -10937,7 +10938,9 @@ class ModelDriver:                          T_land=jnp.asarray(_tas_land_T).reshape(-1),                          q_land=(None if _tas_land_q is None                                  else jnp.asarray(_tas_land_q).reshape(-1)), -                        land_fraction=jnp.asarray(_tas_f_land).reshape(-1)) +                        land_fraction=jnp.asarray(_tas_f_land).reshape(-1), +                        T2m_land=(None if _tas_land_T2m is None +                                  else jnp.asarray(_tas_land_T2m).reshape(-1)))                  from legoesm.forcing.surface_utils import blend_surface_temperature                  ts = blend_surface_temperature(_sst, _sic, _tas_ice)                  if _tas_land_T is not None and _tas_f_land is not None: @@ -14719,6 +14722,13 @@ class ModelDriver:                  _emis = jnp.broadcast_to(                      jnp.asarray(resp.emissivity, resp.T_rad.dtype),                      resp.T_rad.shape) +                # The scheme's own 2 m air temperature (CLM5 t_ref2m) for the +                # CMOR ``tas`` land leg; NaN where the scheme solves none or the +                # column was held (no solved surface layer this call). +                _t2m = (jnp.full_like(resp.T_rad, jnp.nan) +                        if getattr(_sfc, "T_2m", None) is None +                        else jnp.asarray(_sfc.T_2m, resp.T_rad.dtype)) +                _t2m = jnp.where(_held_f > 0.5, jnp.nan, _t2m)                  if _land_pack_on:                      # Scatter the advanced columns back into the full-grid                      # state (ocean columns keep their frozen init values, @@ -14734,12 +14744,13 @@ class ModelDriver:                                  for o in (                                      resp.T_sfc, resp.albedo, resp.q_surface,                                      resp.shflx, resp.lhflx, resp.surface_mass_flux, -                                    _taumag, _held_f, resp.T_rad, _emis)) +                                    _taumag, _held_f, resp.T_rad, _emis, +                                    _t2m))                              + (_n_held, _n_held_land, _n_fb_land,                                 _n_fb_rej, _n_unconv_land, _n_iter_max))                  return (new_state, resp.T_sfc, resp.albedo, resp.q_surface,                          resp.shflx, resp.lhflx, resp.surface_mass_flux, -                        _taumag, _held_f, resp.T_rad, _emis, +                        _taumag, _held_f, resp.T_rad, _emis, _t2m,                          _n_held, _n_held_land, _n_fb_land, _n_fb_rej,                          _n_unconv_land, _n_iter_max)                if not _land_none: @@ -16307,7 +16318,7 @@ class ModelDriver:                       _land_albedo_cells, _land_qsfc_step,                       _land_shflx_step, _land_lhflx_step, _land_evap_step,                       _land_taumag_step, _land_held_step, -                     _land_Trad_cells, _land_emis_cells, +                     _land_Trad_cells, _land_emis_cells, _land_T2m_step,                       _land_n_held_step, _land_n_held_land_step,                       _land_n_fb_step, _land_n_fb_rej_step,                       _land_n_unconv_step, _land_n_iter_max) = _land_fn( @@ -16332,6 +16343,7 @@ class ModelDriver:                      # land and publishes a land temperature the model never had.                      self._land_T_skin_last = _land_T_skin                      self._land_qsfc_last = _land_qsfc_step +                    self._land_T2m_last = _land_T2m_step                      # Accumulate ON DEVICE and read at the same cadence the                      # other post-step warnings use: reading it every step                      # would stall the accelerator once per step for a number diff --git a/packages/land/legoesm/land/canopy/stability.py b/packages/land/legoesm/land/canopy/stability.py index b5efb6366a..1cb2225d45 100644 --- a/packages/land/legoesm/land/canopy/stability.py +++ b/packages/land/legoesm/land/canopy/stability.py @@ -92,6 +92,7 @@ _ZETA_MAX_STABLE = 0.5  # stability forms are continuous through zeta = 0 (user 2026-09-25).  _ZETA_NEUTRAL_FLOOR = 1.0e-6  _CONV_BDY_HEIGHT = 1000.0  # convective boundary layer height [m] +_SCREEN_HEIGHT_M = 2.0     # CMIP tas / CLM5 t_ref2m height above d + z0h [m]  _Z0MG_BARE = 0.01          # bare-soil momentum roughness length [m]    # --- CLM5 Monin-Obukhov similarity-theory (MOST) constants --- @@ -286,6 +287,30 @@ def _temperature_humidity_relation(zldis: jax.Array, obu: jax.Array,      return ch     +def screen_level_temperature(theta_ref: jax.Array, T_canopy_air: jax.Array, +                             zldis: jax.Array, zeta: jax.Array, +                             z0h: jax.Array, +                             z_screen: float = _SCREEN_HEIGHT_M) -> jax.Array: +    """CLM5 ``t_ref2m``: air temperature ``z_screen`` above the displacement +    height plus ``z0h``, interpolated between the reference-height potential +    temperature ``theta_ref`` and the canopy-air (aerodynamic surface) +    temperature with this module's heat similarity relation at the solve's +    Obukhov length ``obu = zldis / zeta`` +    (CanopyFluxesMod.F90:1182, FrictionVelocityMod.F90:636-652):: + +        t_ref2m = thm + temp1 * dth * (1/temp12m - 1/temp1),  dth = thm - taf + +    ``temp1`` and ``temp12m`` are evaluated at the SAME ``obu`` (as CLM does +    within one FrictionVelocity call).  Stable side: Beljaars & Holtslag (1991), +    as in the solve (module docstring).  Output is in the same frame as +    ``theta_ref`` (surface-referenced potential temperature, CLM ``thm``).""" +    obu = zldis / zeta +    temp1 = _temperature_humidity_relation(zldis, obu, z0h) +    temp12m = _temperature_humidity_relation(z_screen + z0h, obu, z0h) +    dth = theta_ref - T_canopy_air +    return theta_ref + temp1 * dth * (1.0 / temp12m - 1.0 / temp1) + +  def _cap_stable_zeta(zeta: jax.Array, width: float) -> jax.Array:      """Smooth cap at ``_ZETA_MAX_STABLE`` (no kink), then the neutral floor.   diff --git a/packages/land/legoesm/land/surface_scheme/base.py b/packages/land/legoesm/land/surface_scheme/base.py index 3a98280647..b7f7dcdc78 100644 --- a/packages/land/legoesm/land/surface_scheme/base.py +++ b/packages/land/legoesm/land/surface_scheme/base.py @@ -88,6 +88,10 @@ class SurfaceFluxOutput(NamedTuple):      # ``T_sfc`` for the atmosphere's sensible-heat coupling.  Distinct from the      # LW-derived radiometric ``T_surface``.  None for SimpleSEB.      T_canopy_air: jax.Array | None = None +    # CLM5 ``t_ref2m``: 2 m air temperature from the scheme's OWN surface layer +    # (roughness, displacement, Obukhov length, canopy-air node) [K].  The CMIP +    # ``tas`` land leg.  None for schemes that do not solve one. +    T_2m: jax.Array | None = None      gs_Sun: jax.Array | None = None        # sunlit stomatal conductance [m/s]      gs_Sh: jax.Array | None = None         # shaded stomatal conductance [m/s]      n_iters: jax.Array | None = None       # canopy Newton iteration count diff --git a/packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py b/packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py index 3aa3908df3..876be1155b 100644 --- a/packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py +++ b/packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py @@ -40,7 +40,7 @@ from legoesm.land.canopy.radiative_transfer import (  )  from legoesm.land.canopy.sif import two_leaf_canopy_sif  from legoesm.land.canopy.stability import ( -    compute_aerodynamics, sat_specific_humidity, +    compute_aerodynamics, sat_specific_humidity, screen_level_temperature,  )  from legoesm.land.canopy.solver import (      CanopyForcingBundle, solve_canopy_closure_diag, canopy_forward, @@ -730,6 +730,8 @@ def compute_two_leaf_canopy_fluxes(          Tf_Sun=Tf_Sun,          Tf_Sh=Tf_Sh,          T_canopy_air=Tc_cvg, +        T_2m=screen_level_temperature(Ta, Tc_cvg, z_ref - displa, +                                      fluxes_per_col["zeta"], z0m),          gs_Sun=gs_Sun,          gs_Sh=gs_Sh,          n_iters=n_iters, diff --git a/tests/land/unit/test_land_screen_temperature.py b/tests/land/unit/test_land_screen_temperature.py new file mode 100644 index 0000000000..61080a6ea2 --- /dev/null +++ b/tests/land/unit/test_land_screen_temperature.py @@ -0,0 +1,128 @@ +"""Land 2 m air temperature (CLM5 ``t_ref2m``) from the two-leaf canopy's own +surface layer, and its use as the CMIP ``tas`` land leg. + +The profile oracle below is an INDEPENDENT scalar transcription of the +similarity integral (Paulson 1970 unstable psi_h, Beljaars & Holtslag 1991 +stable psi_h), not a call into the code under test.""" +import math + +import jax +import jax.numpy as jnp +import numpy as np +import pytest + +from legoesm import constants +from legoesm.land.canopy.stability import screen_level_temperature + +pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64, +                                reason="needs JAX_ENABLE_X64=1") + +K = constants.kappa_von_karman + + +def _psi_h(zeta): +    if zeta < 0.0: +        x = (1.0 - 16.0 * zeta) ** 0.25 +        return 2.0 * math.log((1.0 + x * x) / 2.0) +    a, b, c, d = 1.0, 0.667, 5.0, 0.35   # BH91 as published (b rounded) +    return (-(1.0 + 2.0 * a * zeta / 3.0) ** 1.5 +            - b * (zeta - c / d) * math.exp(-d * zeta) - b * c / d + 1.0) + + +def _oracle(theta, Tc, zl, zeta, z0h, zs=2.0): +    L = zl / zeta +    F = lambda z: math.log(z / z0h) - _psi_h(z / L) + _psi_h(z0h / L) +    return Tc + (theta - Tc) * F(zs + z0h) / F(zl) + + +@pytest.mark.parametrize("zeta", [-0.3, -0.05, 0.05, 0.3]) +@pytest.mark.parametrize("z0h", [0.05, 1.5]) +def test_profile_matches_independent_most(zeta, z0h): +    theta, Tc, zl = 270.0, 262.0, 55.0 +    got = float(screen_level_temperature(jnp.asarray(theta), jnp.asarray(Tc), +                                         jnp.asarray(zl), jnp.asarray(zeta), +                                         jnp.asarray(z0h))) +    np.testing.assert_allclose(got, _oracle(theta, Tc, zl, zeta, z0h), +                               rtol=0.0, atol=1e-9) +    assert min(theta, Tc) <= got <= max(theta, Tc) + + +def test_neutral_limit_is_log_law(): +    theta, Tc, zl, z0h = 285.0, 289.0, 60.0, 0.8 +    got = float(screen_level_temperature(jnp.asarray(theta), jnp.asarray(Tc), +                                         jnp.asarray(zl), jnp.asarray(1e-6), +                                         jnp.asarray(z0h))) +    log_law = Tc + (theta - Tc) * math.log((2.0 + z0h) / z0h) / math.log(zl / z0h) +    np.testing.assert_allclose(got, log_law, rtol=0.0, atol=1e-5) + + +def test_reference_at_screen_height_returns_reference(): +    z0h = 0.3 +    got = screen_level_temperature(jnp.asarray(271.0), jnp.asarray(260.0), +                                   jnp.asarray(2.0 + z0h), jnp.asarray(0.2), +                                   jnp.asarray(z0h)) +    np.testing.assert_allclose(float(got), 271.0, rtol=0.0, atol=1e-12) + + +def test_gradient_finite_and_nonzero(): +    f = lambda th, tc, zeta: screen_level_temperature( +        th, tc, jnp.asarray(50.0), zeta, jnp.asarray(0.5)) +    for zeta in (-0.3, 0.3): +        g = jax.grad(f, argnums=(0, 1, 2))(jnp.asarray(270.0), jnp.asarray(262.0), +                                          jnp.asarray(zeta)) +        assert all(np.isfinite(float(x)) and abs(float(x)) > 1e-8 for x in g), g + + +def _two_leaf(sw, lw, T_soil): +    from legoesm.core.coupling_fields import AtmToSurface +    from legoesm.land.config import MultiLayerLandConfig +    from legoesm.land.surface_scheme import TwoLeafCanopyConfig +    from legoesm.land.surface_scheme.two_leaf_canopy import ( +        compute_two_leaf_canopy_fluxes) +    f = lambda v: jnp.full(1, v) +    forcing = AtmToSurface( +        sw_down=f(sw), lw_down=f(lw), precip_total=f(0.), precip_snow=f(0.), +        T_lowest=f(270.), q_lowest=f(.002), u_lowest=f(4.), v_lowest=f(0.), +        p_lowest=f(99300.), p_surface=f(100000.), rho_lowest=f(1.28), +        cos_zenith=f(.7 if sw > 0 else 0.), co2_ppmv=f(412.), +        has_radiation=f(1.), has_precipitation=f(1.), z_lowest=f(60.)) +    return compute_two_leaf_canopy_fluxes( +        T_soil_top=f(T_soil), forcing=forcing, +        canopy_config=TwoLeafCanopyConfig(max_iters=30), +        land_config=MultiLayerLandConfig(), canopy_params=None, +        w_frac_rz=f(.5), wind_speed=f(4.), wind_dir_x=jnp.ones(1), +        wind_dir_y=jnp.zeros(1), +        soil_thermal_fn=lambda G, dt: jnp.full(1, T_soil), +        TgC_override=forcing.T_lowest - constants.T_freeze, dt=600.) + + +@pytest.mark.parametrize("sw,lw,T_soil", [(0., 200., 262.), (700., 300., 285.)]) +def test_two_leaf_exports_screen_temperature_between_air_and_canopy(sw, lw, T_soil): +    out = _two_leaf(sw, lw, T_soil) +    theta = 270.0 + constants.g / constants.c_pd * 60.0 +    t2, tc = float(out.T_2m[0]), float(out.T_canopy_air[0]) +    assert np.isfinite(t2) +    assert min(theta, tc) - 1e-9 <= t2 <= max(theta, tc) + 1e-9, (t2, tc, theta) + + +def test_tas_land_leg_takes_the_land_screen_temperature(): +    """Fails on the pre-fix code (no T2m_land): finite entries replace the +    ocean-law land profile exactly, non-finite ones keep it.""" +    from types import SimpleNamespace as NS +    from legoesm.driver.diagnostics import DiagnosticCollector +    from legoesm.grids.vertical import create_sigma_coordinate +    nlev = 20 +    coord = create_sigma_coordinate(nlev) +    dc = DiagnosticCollector(nlev=nlev, sigma_full=coord.sigma_full, +                             dsigma=coord.dsigma, vcoord=coord) +    state = NS(T=NS(data=jnp.broadcast_to(jnp.linspace(250., 280., nlev), (2, nlev))), +               u=NS(data=jnp.full((2, nlev), 5.)), v=NS(data=jnp.zeros((2, nlev))), +               p_s=NS(data=jnp.full((2,), 1.0e5))) +    q_v = jnp.full((2, nlev), 2e-3) +    kw = dict(T_land=jnp.full((2,), 262.0), land_fraction=jnp.ones((2,))) +    args = (state, q_v, jnp.full((2,), 275.0), jnp.zeros((2,))) +    old = np.asarray(dc._tas_2m(*args, T_ice=271.35, **kw)) +    new = np.asarray(dc._tas_2m(*args, T_ice=271.35, +                                T2m_land=jnp.asarray([266.5, jnp.nan]), **kw)) +    assert new[0] == 266.5 +    assert new[1] == old[1] diff --git a/tests/unit/test_land_column_split.py b/tests/unit/test_land_column_split.py index c7096ac176..b02786e446 100644 --- a/tests/unit/test_land_column_split.py +++ b/tests/unit/test_land_column_split.py @@ -23,13 +23,13 @@ class _State(NamedTuple):    def _step(state, cols):      """Column-independent stand-in for the land step on columns ``cols``: -    the same tuple layout (state, 10 per-cell fields, 5 counts, iter max).""" +    the same tuple layout (state, 11 per-cell fields, 5 counts, iter max)."""      T = state.T_soil.at[cols].set(state.T_soil[cols] * 1.01 + 0.5)      wet = state.wet.at[cols].set(T[cols, 0] > 290.0)      f = jnp.zeros(NCOL).at[cols].set(T[cols, 0] - 273.0)      held = jnp.zeros(NCOL).at[cols].set((T[cols, 1] > 280.0).astype(float))      n = jnp.sum(T[cols, 0] > 290.0).astype(jnp.int32) -    return ((_State(T, wet),) + (f,) * 7 + (held,) + (f, f) +    return ((_State(T, wet),) + (f,) * 7 + (held,) + (f, f, f)              + (n, n, n, 2 * n, n) + (jnp.max(cols).astype(jnp.int32),))    
1842901       01:06  0.0 /usr/bin/zsh -c source /glade/u/home/pg2328/.claude/shell-snapshots/snapshot-zsh-1795338233552-bpscww.sh 2>/dev/null || true && \builtin setopt NO_EXTENDED_GLOB NO_BARE_GLOB_QUAL NO_BRACE_CCL NO_MAGIC_EQUAL_SUBST NO_EQUALS 2>/dev/null || true && { \builtin unalias -- 'which'; \builtin unset -f -- 'which'; } >/dev/null 2>&1 || true && { \builtin unalias -- 'unsetenv'; \builtin unset -f -- 'unsetenv'; } >/dev/null 2>&1 || true && eval 'SP=/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/5744ad21-cfc1-41c2-8559-0d2d61db48aa/scratchpad; cat > $SP/cmp_fail.sh <<'"'"'EOF'"'"' #!/bin/bash -l module load conda >/dev/null 2>&1; conda activate /glade/work/pg2328/conda-envs/legoesm-gpu export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 T=tests/unit/test_surface_height_and_temperature_output.py for D in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193 /glade/u/home/pg2328/legoesm; do   echo "== $D"; cd $D   PYTHONPATH="$D/src:$(ls -d $D/packages/* | tr '"'"'\n'"'"' '"'"':'"'"')" python -m pytest -q -p no:cacheprovider $T -k '"'"'grid_height or mpas_loop or skin_temperature_blend or clm_warm_step'"'"' -rf 2>&1 | grep -E '"'"'^FAILED|passed|failed'"'"' | cut -c1-250 done EOF chmod +x $SP/cmp_fail.sh; timeout 5000 qcmd -A UOSC0036 -l walltime=01:20:00 --nthreads 8 --mem 120GB -- $SP/cmp_fail.sh 2>&1 | grep -v "^qsub\|Waiting\|^Job\|tty\|job control" | tail -20' && pwd -P >| /glade/derecho/scratch/pg2328/tmp/claude-02d2-cwd
1842905       01:06  0.0 /usr/bin/zsh -c source /glade/u/home/pg2328/.claude/shell-snapshots/snapshot-zsh-1795338233552-bpscww.sh 2>/dev/null || true && \builtin setopt NO_EXTENDED_GLOB NO_BARE_GLOB_QUAL NO_BRACE_CCL NO_MAGIC_EQUAL_SUBST NO_EQUALS 2>/dev/null || true && { \builtin unalias -- 'which'; \builtin unset -f -- 'which'; } >/dev/null 2>&1 || true && { \builtin unalias -- 'unsetenv'; \builtin unset -f -- 'unsetenv'; } >/dev/null 2>&1 || true && eval 'SP=/glade/derecho/scratch/pg2328/tmp/claude-24190/-glade-u-home-pg2328-legoesm/5744ad21-cfc1-41c2-8559-0d2d61db48aa/scratchpad; cat > $SP/cmp_fail.sh <<'"'"'EOF'"'"' #!/bin/bash -l module load conda >/dev/null 2>&1; conda activate /glade/work/pg2328/conda-envs/legoesm-gpu export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 T=tests/unit/test_surface_height_and_temperature_output.py for D in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193 /glade/u/home/pg2328/legoesm; do   echo "== $D"; cd $D   PYTHONPATH="$D/src:$(ls -d $D/packages/* | tr '"'"'\n'"'"' '"'"':'"'"')" python -m pytest -q -p no:cacheprovider $T -k '"'"'grid_height or mpas_loop or skin_temperature_blend or clm_warm_step'"'"' -rf 2>&1 | grep -E '"'"'^FAILED|passed|failed'"'"' | cut -c1-250 done EOF chmod +x $SP/cmp_fail.sh; timeout 5000 qcmd -A UOSC0036 -l walltime=01:20:00 --nthreads 8 --mem 120GB -- $SP/cmp_fail.sh 2>&1 | grep -v "^qsub\|Waiting\|^Job\|tty\|job control" | tail -20' && pwd -P >| /glade/derecho/scratch/pg2328/tmp/claude-02d2-cwd
1843135       00:52  0.0 /usr/bin/bash -c PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python - <<'PY' import sys, pathlib sys.path[:0] = [str(p.resolve()) for p in pathlib.Path('packages').iterdir() if p.is_dir()] import pytest raise SystemExit(pytest.main(['-p','no:cacheprovider','tests/land/unit/test_land_screen_temperature.py','tests/unit/test_land_column_split.py','-q'])) PY
1843140       00:52  156 /glade/work/pg2328/conda-envs/legoesm-gpu/bin/python -
1844141       00:00 16.6 /usr/bin/bash -c ps -u pg2328 -o pid,etime,pcpu,args | rg 'pytest|python -' | tail -10
1844147       00:00  0.0 rg pytest|python -

exec
/usr/bin/bash -lc 'git status --porcelain' in /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
 succeeded in 24ms:
?? docs/wb/reviews/2026-11-22_land_tas_r1_codex.md
?? docs/wb/reviews/2026-11-22_land_tas_r1_glm.md
?? docs/wb/reviews/2026-11-22_land_tas_r1_prompt.md

 succeeded in 115390ms:
============================= test session starts ==============================
platform linux -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
rootdir: /glade/u/home/pg2328/legoesm/.claude/worktrees/agent-a19b47c5f4c684193
configfile: pyproject.toml
plugins: zarr-3.4.0, timeout-2.4.0, jaxtyping-0.3.11, xdist-3.8.0
collected 20 items

tests/land/unit/test_land_screen_temperature.py ..............           [ 70%]
tests/unit/test_land_column_split.py ......                              [100%]

======================== 20 passed in 111.65s (0:01:51) ========================

codex
Verdict: **NOT CLEAN**.

1. **P2 — Obukhov length differs from the flux solve.** [two_leaf_canopy.py:733](packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py#L733) uses the returned `zeta_new`. However, [stability.py:364](packages/land/legoesm/land/canopy/stability.py#L364) evaluates the flux heat coefficient using the incoming `obu`, then updates stability. Reconstructing both screen coefficients from `zeta_new` therefore does not preserve the flux solve’s Obukhov length. Finite iteration does not guarantee equality. Export the length actually used, without changing the existing flux solve.

2. **P3 — Missing halo values become valid zero temperatures.** [model_driver.py:14742](packages/coupler/legoesm/driver/model_driver.py#L14742) scatters T2m with zero fill; the empty-rank producer at line 14766 also returns zeros. T2m is omitted from the halo exchange. Consequently, unsolved land halos can override the fallback with **0 K**, rather than use the documented fallback or a solved local value. Current CMIP gathering discards halo rows, limiting the consequence.

3. **P3 — Tests miss both defects.** [test_land_screen_temperature.py:38](tests/land/unit/test_land_screen_temperature.py#L38) tests prescribed stability independently, but never checks consistency with the flux coefficient. Its canopy bounds admit either endpoint and omit the very-unstable oracle branch. The split fixture repeats identical fields, masking ordering mistakes.

The CLM expression, temperature frame, displacement and roughness wiring otherwise agree with the cited source; BH91 is documented. Tuple arity/gather handling is consistent. No new state, flux or coupling dependency found. **20 focused tests passed**, including helper gradients.

V1: **A** — preserve CLM’s published `t_ref2m` convention.

V2: **A** — preserve the established fallback where the scheme provides no screen temperature.

V3: **B** — persist and exchange T2m now to preserve diagnostic continuity across restarts and ranks.

UNVERIFIED: full driver/MPI execution and full-canopy gradients.
tokens used
84,758
Verdict: **NOT CLEAN**.

1. **P2 — Obukhov length differs from the flux solve.** [two_leaf_canopy.py:733](packages/land/legoesm/land/surface_scheme/two_leaf_canopy.py#L733) uses the returned `zeta_new`. However, [stability.py:364](packages/land/legoesm/land/canopy/stability.py#L364) evaluates the flux heat coefficient using the incoming `obu`, then updates stability. Reconstructing both screen coefficients from `zeta_new` therefore does not preserve the flux solve’s Obukhov length. Finite iteration does not guarantee equality. Export the length actually used, without changing the existing flux solve.

2. **P3 — Missing halo values become valid zero temperatures.** [model_driver.py:14742](packages/coupler/legoesm/driver/model_driver.py#L14742) scatters T2m with zero fill; the empty-rank producer at line 14766 also returns zeros. T2m is omitted from the halo exchange. Consequently, unsolved land halos can override the fallback with **0 K**, rather than use the documented fallback or a solved local value. Current CMIP gathering discards halo rows, limiting the consequence.

3. **P3 — Tests miss both defects.** [test_land_screen_temperature.py:38](tests/land/unit/test_land_screen_temperature.py#L38) tests prescribed stability independently, but never checks consistency with the flux coefficient. Its canopy bounds admit either endpoint and omit the very-unstable oracle branch. The split fixture repeats identical fields, masking ordering mistakes.

The CLM expression, temperature frame, displacement and roughness wiring otherwise agree with the cited source; BH91 is documented. Tuple arity/gather handling is consistent. No new state, flux or coupling dependency found. **20 focused tests passed**, including helper gradients.

V1: **A** — preserve CLM’s published `t_ref2m` convention.

V2: **A** — preserve the established fallback where the scheme provides no screen temperature.

V3: **B** — persist and exchange T2m now to preserve diagnostic continuity across restarts and ranks.

UNVERIFIED: full driver/MPI execution and full-canopy gradients.
