You are an independent adversarial physics/JAX reviewer for LegoESM (a JAX-native,
fully-differentiable Earth System Model). Review the "MPAS land surface boundary"
change on branch `mpas-stability-campaign`, commit `f0efce5a4`. Repo root is the
current working directory. FINDINGS ONLY — do NOT modify any file.

Read the actual source with your tools; the summary below is a guide, not ground truth.
Cite `file:line`. For each finding label severity (CRITICAL/HIGH/MEDIUM/LOW/INFO) and
CONFIRMED vs PLAUSIBLE. If you find no bug in a category, say so and justify why each
candidate concern is not one.

## What the change does (all knobs default OFF, byte-identical when off)
The standalone MPAS AMIP lane (`ModelDriver._run_mpas` +
`atmosphere/physics/combined.make_physics(model_type="mpas")`) has NO land tile: the
AMIP SST loader fills land cells with nearest-ocean sea-level SST, and MPAS turbulence
uses `q_sfc = saturation_mixing_ratio(T_sfc, p_low)` everywhere, so land behaves as a
warm infinite swamp (surface too warm by lapse*z over elevated terrain; spurious latent
flux + convection + a cell-scale CWV recharge/discharge speckle).

Two first-order corrections, both default OFF:
1. `forcing/surface_utils.py::land_lapse_adjusted_surface_temperature(T_sfc, f_land,
   z_sfc, lapse_K_per_m)` = `T_sfc - f_land*lapse*max(z,0)`.
2. `turbulence/surface_layer.py::beta_limited_surface_humidity(q_sat_sfc, q_air, f_land,
   beta_land)` = `q_air + (1 - f_land*(1-beta))*(q_sat_sfc - q_air)`.
3. `turbulence/integration.py`: `make_turbulence_physics` + `_make_mpas_turbulence` take
   `f_land`/`land_beta` (closure consts); q_sfc beta-limited under a static Python gate
   `if f_land is not None and land_beta != 1.0`; non-MPAS model_types RAISE if knobs set.
4. `combined.py::make_physics(+f_land,+land_beta)` threads to the turbulence factory on
   the mpas path only; raises otherwise.
5. `driver/config.py`: `ExperimentConfig.mpas_land_lapse_K_per_km` (0=off, [0,20]) +
   `mpas_land_beta` (1=off, [0,1]); `validate_strict` refuses `slab_land_active /
   land_soil_bucket / surface_tiled` on the MPAS lane and refuses the `mpas_land_*` knobs
   on non-MPAS lanes.
6. `driver/model_driver.py::_run_mpas`: computes `_land_beta`/`_land_lapse_K_m`, raises if
   knobs set but no f_land, passes f_land/land_beta into BOTH make_physics calls (full +
   norad), and lapse-adjusts the T_sfc forcing anchor inside `_compute_T_sfc` (which feeds
   BOTH radiation and turbulence via `forcing["T_sfc"]`).
7. `run_amip.py`: `--mpas-land-lapse-k-per-km` / `--mpas-land-beta` flags; tests in
   `tests/unit/test_mpas_land_boundary.py` + CLI round-trip.

## Files to review (all under repo root)
- `packages/tools/legoesm/forcing/surface_utils.py` (lapse helper, ~line 40)
- `packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py` (beta helper, ~line 158)
- `packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py`
  (`make_turbulence_physics` ~298, `_make_mpas_turbulence` ~508, q_sfc gate ~594)
- `packages/atmosphere/legoesm/atmosphere/physics/combined.py::make_physics` (~116)
- `packages/coupler/legoesm/driver/config.py::ExperimentConfig.validate_strict` (~1708)
- `packages/coupler/legoesm/driver/model_driver.py::_run_mpas` (~5567; knob wiring ~5828,
  T_sfc anchor ~5952; forcing dispatch to full/norad ~6449); run dispatch ~5245.
- `scripts/run/run_amip.py` (flags ~1209; `_default_discretization_for_grid` ~1466;
  grid normalization ~2008)
- `tests/unit/test_mpas_land_boundary.py`

## Rank and label these specific questions
1. Sign/units in the two formulas. Dtype/promotion hazards (f_land is stored at the
   precision-policy `storage` dtype — float32 under default/mixed, float64 only under
   fp64/mixed_fp64_storage; the turbulence path casts to `q_sfc.dtype`, the lapse path in
   `_compute_T_sfc` does NOT cast). Closure consts on GPU.
2. JIT/retrace: are the f_land closures compile-time constants or per-step traffic? Does
   the static Python gate hold (no traced bool)?
3. The lapse being applied to the RADIATION anchor too (via `forcing["T_sfc"]`) — is that
   consistent/desirable? Any OTHER consumer of `forcing["T_sfc"]` that would wrongly see
   lapse-adjusted values? Sea-ice blend order: lapse is applied AFTER the sic blend
   `blend_surface_temperature(sst,sic,T_ice)=sic*T_ice+(1-sic)*sst`; over a high-elevation
   land cell whose nearest-ocean fill carries sic>0, the lapse cools a temperature that
   already includes T_ice on the f_land fraction — bug or acceptable? Can f_land and sic
   overlap?
4. Missed call sites: any other `make_physics(model_type="mpas")` or `_make_mpas_turbulence`
   caller (SCM? spmd? tests?) that would now diverge from the driver behavior.
5. `validate_strict` guard correctness: the guard keys on `d.discretization=="mpas"`, but
   the actual `_run_mpas` dispatch keys on `self.config.grid.grid_type=="mpas"`
   (model_driver ~5245). Are these ALWAYS consistent? Is there any config path that runs
   `_run_mpas` with a different discretization value (e.g. directly-constructed
   `ExperimentConfig(grid_type="mpas")` with the DycoreConfig default
   `discretization="cdgrid"`), or that runs the driver pipeline WITH
   `discretization=="mpas"` (which would now wrongly refuse slab_land_active)? Is there a
   config-level validation tying the two fields?
6. The norad variant gets the same knobs — verify no divergence between the full and norad
   physics build/forcing.
7. Restart/checkpoint: no new state, but does the T_sfc anchor change break restart
   bit-exactness mid-run when a chain relaunches with the same flags, or when flags change
   between links (chain contract)?
8. Test gaps: what CONFIRMED failure mode is untested?

## The unified diff of the change (commit f0efce5a4)
```diff
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/combined.py b/packages/atmosphere/legoesm/atmosphere/physics/combined.py
index 25655f8a2..a2fda8c4d 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/combined.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/combined.py
@@ -121,6 +121,8 @@ def make_physics(
     sfc_albedo_override=None,
     sfc_emissivity_override=None,
     need_rad: bool = True,
+    f_land=None,
+    land_beta: float = 1.0,
 ) -> Callable:
     """Create a combined physics function for a dynamical core.
 
@@ -171,6 +173,15 @@ def make_physics(
     # a trained value never touches RRTMGPConfig.sfc_* (RRTMGP's solver-cache
     # key). Only the spectral_pe combined path threads them today; reject loudly
     # elsewhere rather than silently dropping a trained surface field.
+    # MPAS land surface boundary knobs (f_land + land_beta): consumed by the
+    # MPAS turbulence factory only.  Reject loudly elsewhere — the FV/spectral
+    # pipelines have their own land tile, so accepting the args there would be
+    # a silently-inert configuration (the 2026-07-23 --slab-land-active lesson).
+    if model_type != "mpas" and (f_land is not None or land_beta != 1.0):
+        raise ValueError(
+            "f_land/land_beta are MPAS-only land surface boundary knobs "
+            f"(model_type={model_type!r} would silently ignore them)."
+        )
     if (sfc_albedo_override is not None or sfc_emissivity_override is not None) \
             and model_type != "spectral_pe":
         raise ValueError(
@@ -192,7 +203,7 @@ def make_physics(
         # MPAS (Voronoi mesh) uses the unified hydrostatic combined path.
         fn = _make_hydrostatic_combined(
             config, dt, model_type="mpas", column_mesh=column_mesh,
-            need_rad=need_rad)
+            need_rad=need_rad, f_land=f_land, land_beta=land_beta)
     else:
         raise ValueError(
             f"Unknown model_type: {model_type!r}. "
@@ -313,7 +324,9 @@ def _attach_lifecycle_hooks(physics_fn, tagged_fns):
 def _make_hydrostatic_combined(config: PhysicsConfig, dt: float,
                                model_type: str = "hydrostatic",
                                column_mesh=None,
-                               need_rad: bool = True) -> Callable:
+                               need_rad: bool = True,
+                               f_land=None,
+                               land_beta: float = 1.0) -> Callable:
     """Combined physics for any hydrostatic model (cubed-sphere, lat-lon, MPAS).
 
     Uses the unified ``HydrostaticTendencies`` with optional ``dv_dt``.
@@ -381,7 +394,8 @@ def _make_hydrostatic_combined(config: PhysicsConfig, dt: float,
         _turb_sn, _, _turb_sc = get_turbulence_fn(config.turbulence)
         _turb_field = turbulence_carry_field(_turb_sn, _turb_sc)
         tagged_fns.append((
-            make_turbulence_physics(config.turbulence, model_type, dt),
+            make_turbulence_physics(config.turbulence, model_type, dt,
+                                    f_land=f_land, land_beta=land_beta),
             True,
             _turb_field,
         ))
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py b/packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py
index e2896a3f9..b923e4cfb 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py
@@ -298,6 +298,8 @@ def make_turbulence_physics(
     turbulence_config: TurbulenceConfig,
     model_type: str = "hydrostatic",
     dt: float = 300.0,  # coeff-ok: default physics timestep [s]
+    f_land=None,
+    land_beta: float = 1.0,
 ) -> Callable:
     """Create a physics function for turbulence matching a model's signature.
 
@@ -309,12 +311,27 @@ def make_turbulence_physics(
         One of "hydrostatic", "nonhydrostatic", "spectral_pe".
     dt : float
         Model time step [s].
+    f_land : array or None
+        Static per-cell land fraction in [0, 1] for the MPAS land surface
+        boundary (baked as a closure constant).  MPAS-only: the FV /
+        spectral pipelines carry their own land tile, so passing it for any
+        other ``model_type`` raises rather than being silently inert.
+    land_beta : float
+        Land evaporation efficiency in [0, 1] applied on the ``f_land``
+        fraction of the surface humidity (MPAS-only, with ``f_land``).
+        ``1.0`` (default) keeps the saturated wet surface, byte-identical.
 
     Returns
     -------
     Callable
         Physics function with the correct signature for the model.
     """
+    if model_type != "mpas" and (f_land is not None or land_beta != 1.0):
+        raise ValueError(
+            "f_land/land_beta are the MPAS land surface boundary knobs; the "
+            f"{model_type!r} pipeline has its own land tile (they would be "
+            "silently inert here). Drop them or use model_type='mpas'."
+        )
     if model_type == "hydrostatic":
         return _make_hydrostatic_turbulence(turbulence_config, dt)
     elif model_type == "nonhydrostatic":
@@ -331,7 +348,8 @@ def make_turbulence_physics(
         # projection) is the implementation; it reads the prescribed surface
         # temperature from the per-step ``forcing["T_sfc"]`` (AMIP SST) when
         # supplied, so the surface sensible/latent fluxes are SST-driven.
-        _mpas_turb_fn = _make_mpas_turbulence(turbulence_config, dt)
+        _mpas_turb_fn = _make_mpas_turbulence(
+            turbulence_config, dt, f_land=f_land, land_beta=land_beta)
         # Forcing-aware: the combined-physics dispatcher forwards
         # ``forcing["T_sfc"]`` to fns advertising this (mirrors radiation).
         _mpas_turb_fn._wants_forcing = True
@@ -490,6 +508,8 @@ def _make_hydrostatic_turbulence(
 def _make_mpas_turbulence(
     turbulence_config: TurbulenceConfig,
     dt: float,
+    f_land=None,
+    land_beta: float = 1.0,
 ) -> Callable:
     """Create turbulence physics_fn for MPASPrimitiveEquationModel.
 
@@ -572,6 +592,20 @@ def _make_mpas_turbulence(
         else:
             T_sfc = _resolve_T_sfc(T_col, phys_state)
         q_sfc = saturation_mixing_ratio(T_sfc, p_full_col[:, -1])
+        # MPAS land surface boundary: throttle the LAND fraction's surface
+        # humidity gradient by ``land_beta`` (soil-moisture availability)
+        # instead of the saturated infinite-swamp value the nearest-ocean
+        # SST fill otherwise implies.  Static feature gate (Python ``if`` on
+        # build-time config, the JAX feature-gating exception): defaults
+        # (f_land None / beta 1) keep this branch out of the trace entirely,
+        # byte-identical to the pre-knob path.
+        if f_land is not None and land_beta != 1.0:
+            from legoesm.atmosphere.physics.turbulence.surface_layer import (
+                beta_limited_surface_humidity,
+            )
+            _f_land_col = jnp.asarray(f_land, dtype=q_sfc.dtype).reshape(nCells)
+            q_sfc = beta_limited_surface_humidity(
+                q_sfc, q_v_col[:, -1], _f_land_col, land_beta)
 
         if needs_tke:
             tke_in = _read_turb_carry(
diff --git a/packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py b/packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py
index ff1db95e8..e870b34c7 100644
--- a/packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py
+++ b/packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py
@@ -158,6 +158,48 @@ def compute_surface_fluxes(
     return tau_x, tau_y, shflx, lhflx, ustar
 
 
+def beta_limited_surface_humidity(
+    q_sat_sfc: jax.Array,
+    q_air: jax.Array,
+    f_land: jax.Array,
+    beta_land: float,
+) -> jax.Array:
+    """Soil-moisture-limited effective surface humidity for a BLENDED surface.
+
+    On a non-tiled surface the bulk latent flux is
+    ``LH ∝ (q_sfc - q_air)``.  Using the saturated ``q_sat_sfc`` everywhere
+    makes every land cell an infinite swamp (beta = 1).  This throttles the
+    LAND fraction's humidity gradient by ``beta_land`` while leaving the
+    ocean/ice fraction saturated::
+
+        q_sfc = q_air + (1 - f_land * (1 - beta_land)) * (q_sat_sfc - q_air)
+
+    so the land-fraction latent flux is ``beta_land`` times its wet-surface
+    potential — the same alpha-method form the tiled pipeline applies per
+    tile (``physics_pipeline._tiled_surface_flux``).  ``beta_land = 1``
+    returns ``q_sat_sfc`` exactly (byte-identical wet surface);
+    ``beta_land = 0`` zeroes the land-fraction humidity gradient in both
+    directions (no land evaporation and no land dew — a closed surface).
+
+    Parameters
+    ----------
+    q_sat_sfc : array
+        Saturation mixing ratio at the surface anchor [kg/kg], shape (ncol,).
+    q_air : array
+        Lowest-level vapor mixing ratio [kg/kg], shape (ncol,).
+    f_land : array
+        Land fraction in [0, 1], shape (ncol,).
+    beta_land : float
+        Land evaporation efficiency in [0, 1].
+
+    Returns
+    -------
+    q_sfc : array
+        Effective surface humidity for the bulk latent flux [kg/kg].
+    """
+    return q_air + (1.0 - f_land * (1.0 - beta_land)) * (q_sat_sfc - q_air)
+
+
 class SurfaceTileSpec(NamedTuple):
     """Per-tile, per-column surface STATE for a mosaic (tiled) surface.
 
diff --git a/packages/coupler/legoesm/driver/config.py b/packages/coupler/legoesm/driver/config.py
index 0dd40f3f3..f33842ec5 100644
--- a/packages/coupler/legoesm/driver/config.py
+++ b/packages/coupler/legoesm/driver/config.py
@@ -1064,6 +1064,21 @@ class ExperimentConfig(NamedTuple):
     sponge_coeff_per_day: float = 2.0   # Rayleigh damping rate at the model top [1/day]
     sponge_sigma_top: float = 0.15      # sponge base: sigma below which damping ramps up
 
+    # --- MPAS land surface boundary (standalone MPAS lane only) -------------
+    # The MPAS combined-physics lane has NO land tile: the AMIP loader fills
+    # land cells with the NEAREST-OCEAN SST (a sea-level temperature) and the
+    # surface humidity is saturated everywhere, so every land cell acts as a
+    # warm infinite swamp (+5-12 K vs its own air over elevated terrain).
+    # Instrumented on the 2026-07-23 pilot: tropical-land latent flux 110-240
+    # W/m2 (vs 74 ocean), SBM precip up to 19 mm/day on highlands, and a
+    # cell-scale CWV recharge/discharge speckle.  Two first-order corrections:
+    # lapse-adjust the land anchor (T_eff = T_sfc - f_land*lapse*z) and
+    # throttle the land evaporation efficiency (beta).  Defaults are OFF /
+    # byte-identical.  FV / spectral lanes have a real land tile — these
+    # knobs are refused there (validate_strict).
+    mpas_land_lapse_K_per_km: float = 0.0  # land anchor lapse [K/km]; 0=off, 6.5=ICAO std
+    mpas_land_beta: float = 1.0            # land evaporation efficiency [0-1]; 1=wet swamp
+
     # Held-Suarez forcing
     held_suarez_forcing: bool = False  # add HS Newtonian relaxation + Rayleigh drag
 
@@ -1683,6 +1698,46 @@ class ExperimentConfig(NamedTuple):
                 f"land_bucket_w_init_frac (initial fill fraction) must be in "
                 f"[0, 1]; got {self.land_bucket_w_init_frac!r}."
             )
+        # --- MPAS lane land-surface consistency (fail-fast, no silent no-ops)
+        # The standalone MPAS lane builds its physics from combined.make_physics
+        # (NOT the driver PhysicsPipeline), so the pipeline land tile
+        # (slab_land_active / land_soil_bucket / surface_tiled) never executes
+        # there — accepting those flags on MPAS ran a 60-day A/B against a
+        # byte-identical twin (2026-07-23).  Conversely the MPAS land boundary
+        # knobs are consumed only by the MPAS lane.
+        _is_mpas = d.discretization == "mpas"
+        if _is_mpas:
+            for _flag in ("slab_land_active", "land_soil_bucket",
+                          "surface_tiled"):
+                if getattr(self, _flag):
+                    errors.append(
+                        f"{_flag}=True is silently inert on the MPAS lane "
+                        "(its physics comes from combined.make_physics, not "
+                        "the driver pipeline). Use the MPAS land boundary "
+                        "knobs instead: mpas_land_lapse_K_per_km / "
+                        "mpas_land_beta."
+                    )
+        else:
+            if self.mpas_land_lapse_K_per_km != 0.0 or self.mpas_land_beta != 1.0:
+                errors.append(
+                    "mpas_land_lapse_K_per_km/mpas_land_beta are MPAS-lane "
+                    f"knobs; discretization={d.discretization!r} has its own "
+                    "land tile (slab_land_active / use_multilayer_land) and "
+                    "would silently ignore them."
+                )
+        if not (math.isfinite(self.mpas_land_lapse_K_per_km)
+                and 0.0 <= self.mpas_land_lapse_K_per_km <= 20.0):
+            errors.append(
+                f"mpas_land_lapse_K_per_km must be finite in [0, 20] "
+                f"(0=off, 6.5=ICAO standard); got "
+                f"{self.mpas_land_lapse_K_per_km!r}."
+            )
+        if not (math.isfinite(self.mpas_land_beta)
+                and 0.0 <= self.mpas_land_beta <= 1.0):
+            errors.append(
+                f"mpas_land_beta (land evaporation efficiency) must be finite "
+                f"in [0, 1]; got {self.mpas_land_beta!r}."
+            )
         # gs_max is a physical conductance [mol/m2/s]: must be finite and
         # strictly positive (nan/<=0 would zero or NaN the whole land latent
         # flux).  Upper sanity bound 2.0 is well above the StomataConfig
diff --git a/packages/coupler/legoesm/driver/model_driver.py b/packages/coupler/legoesm/driver/model_driver.py
index 9575c1c95..620cddd4e 100644
--- a/packages/coupler/legoesm/driver/model_driver.py
+++ b/packages/coupler/legoesm/driver/model_driver.py
@@ -5825,8 +5825,35 @@ class ModelDriver:
                 f"divisible by n_devices={_n_dev}; running single-device "
                 f"(throughput not scaled across GPUs)"
             )
+        # --- MPAS land surface boundary knobs (validate_strict-bounded) -----
+        # beta throttles the land-fraction surface humidity inside the MPAS
+        # turbulence factory (closure const); the lapse adjusts the T_sfc
+        # forcing anchor below.  Both default OFF => byte-identical builds.
+        _land_beta = float(getattr(cfg, "mpas_land_beta", 1.0))
+        _land_lapse_K_m = (
+            float(getattr(cfg, "mpas_land_lapse_K_per_km", 0.0)) * 1.0e-3)
+        _f_land_cells = None
+        if self._f_land is not None:
+            _f_land_cells = jnp.asarray(self._f_land).reshape(-1)
+        if (_land_beta != 1.0 or _land_lapse_K_m > 0.0) \
+                and _f_land_cells is None:
+            raise ValueError(
+                "mpas_land_beta/mpas_land_lapse_K_per_km need a land "
+                "fraction, but none was loaded (no --topography / land "
+                "mask source) — the knobs would be silently inert."
+            )
+        if _land_beta != 1.0 or _land_lapse_K_m > 0.0:
+            logger.info(
+                "  MPAS land boundary: lapse=%.2f K/km, beta=%.2f "
+                "(f_land mean=%.3f)",
+                _land_lapse_K_m * 1.0e3, _land_beta,
+                float(jnp.mean(_f_land_cells)),
+            )
         physics_fn = make_physics(phys_cfg, model_type="mpas", dt=DT,
-                                  column_mesh=_column_mesh)
+                                  column_mesh=_column_mesh,
+                                  f_land=(_f_land_cells
+                                          if _land_beta != 1.0 else None),
+                                  land_beta=_land_beta)
 
         # ---- Radiation sub-cycle (issue #316, MPAS port) ----
         # The MPAS physics_fn fuses radiation into ``model.step`` and ran the
@@ -5846,7 +5873,10 @@ class ModelDriver:
         _subcycle_rad = RAD_UPDATE_STEPS > 1 and cfg.radiation != "none"
         physics_fn_norad = (
             make_physics(phys_cfg, model_type="mpas", dt=DT,
-                         column_mesh=_column_mesh, need_rad=False)
+                         column_mesh=_column_mesh, need_rad=False,
+                         f_land=(_f_land_cells
+                                 if _land_beta != 1.0 else None),
+                         land_beta=_land_beta)
             if _subcycle_rad else None
         )
         if _subcycle_rad:
@@ -5922,17 +5952,33 @@ class ModelDriver:
         _sst_forcing = (cfg.radiation != "none" and self.get_sst_sic is not None)
         _compute_T_sfc = None
         if _sst_forcing:
-            from legoesm.forcing.surface_utils import blend_surface_temperature
+            from legoesm.forcing.surface_utils import (
+                blend_surface_temperature,
+                land_lapse_adjusted_surface_temperature,
+            )
             _T_ice = cfg.T_ice
             _ncell = int(self.state.T.data.shape[0])
+            # Land anchor lapse correction: the AMIP loader fills land cells
+            # with the NEAREST-OCEAN SST (sea-level temperature); anchoring
+            # elevated land at that value overheats its surface by lapse*z.
+            # Applied on the land fraction only; z clipped at 0 inside the
+            # helper.  Static gate (validate_strict bounds the rate).
+            _lapse_z = None
+            if _land_lapse_K_m > 0.0:
+                _lapse_z = (jnp.asarray(self.state.phis.data).reshape(-1)
+                            / constants.g)
 
             def _compute_T_sfc(day):
                 # Prescribed SST/SIC at the MPAS cell latitudes (get_sst_sic is
                 # built on grid.grid_lat = mesh.latCell for analytical/AMIP
                 # data), sea-ice-blended, as a (nCells,) surface temperature.
                 _sst, _sic = self.get_sst_sic(day)
-                return blend_surface_temperature(
+                _ts = blend_surface_temperature(
                     jnp.asarray(_sst), jnp.asarray(_sic), _T_ice).reshape(-1)
+                if _lapse_z is not None:
+                    _ts = land_lapse_adjusted_surface_temperature(
+                        _ts, _f_land_cells, _lapse_z, _land_lapse_K_m)
+                return _ts
 
             # Shape guard once, up front: a non-per-cell get_sst_sic would
             # otherwise surface as an opaque error deep inside the JIT trace.
diff --git a/packages/tools/legoesm/forcing/surface_utils.py b/packages/tools/legoesm/forcing/surface_utils.py
index 0559ab60c..fa37c5766 100644
--- a/packages/tools/legoesm/forcing/surface_utils.py
+++ b/packages/tools/legoesm/forcing/surface_utils.py
@@ -37,6 +37,49 @@ def blend_surface_temperature(
     return sic * T_ice + (1.0 - sic) * sst
 
 
+def land_lapse_adjusted_surface_temperature(
+    T_sfc: jnp.ndarray,
+    f_land: jnp.ndarray,
+    z_sfc: jnp.ndarray,
+    lapse_K_per_m: float,
+) -> jnp.ndarray:
+    """Lower the LAND fraction of a surface-temperature anchor by a lapse rate.
+
+    AMIP SST loaders fill land cells with the nearest-ocean SST (a sea-level
+    temperature).  Anchoring surface fluxes / radiation to that value at an
+    elevated cell overheats the surface by ``lapse * z`` (e.g. ~+13 K at 2 km
+    for 6.5 K/km), driving spurious surface fluxes and convection over
+    highlands.  This applies the standard-atmosphere correction on the land
+    fraction only::
+
+        T_eff = T_sfc - f_land * lapse_K_per_m * max(z_sfc, 0)
+
+    Sign convention: ``z_sfc`` positive up [m]; ``lapse_K_per_m > 0`` cools
+    with height.  The ocean fraction (``f_land = 0``) is unchanged; negative
+    elevations (below-sea-level basins) are clipped to zero rather than
+    warmed.
+
+    Parameters
+    ----------
+    T_sfc : array
+        Surface-temperature anchor [K] (ocean/ice blended, land = nearest-
+        ocean fill).
+    f_land : array
+        Land fraction in [0, 1], same shape as ``T_sfc``.
+    z_sfc : array
+        Surface elevation [m] (e.g. ``phis / g``), same shape.
+    lapse_K_per_m : float
+        Lapse rate [K/m]; 0 disables (returns ``T_sfc`` unchanged up to
+        floating-point identity).
+
+    Returns
+    -------
+    T_eff : array
+        Lapse-adjusted surface temperature [K].
+    """
+    return T_sfc - f_land * lapse_K_per_m * jnp.maximum(z_sfc, 0.0)
+
+
 def blend_surface_property(
     sic: jnp.ndarray,
     value_ice: float | jnp.ndarray,
diff --git a/scripts/run/run_amip.py b/scripts/run/run_amip.py
index b5bc9f7c7..56857a602 100644
--- a/scripts/run/run_amip.py
+++ b/scripts/run/run_amip.py
@@ -1206,6 +1206,20 @@ def build_arg_parser() -> argparse.ArgumentParser:
                         dest="sponge_sigma_top",
                         help="Sponge base: sigma below which the sin^2 damping "
                              "ramps up toward the lid (default 0.15).")
+    parser.add_argument("--mpas-land-lapse-k-per-km", type=float, default=None,
+                        dest="mpas_land_lapse_K_per_km",
+                        help="MPAS lane only: lapse-adjust the LAND fraction's "
+                             "surface-temperature anchor by this rate [K/km] "
+                             "times elevation (the AMIP loader fills land "
+                             "cells with nearest-ocean sea-level SST, which "
+                             "overheats elevated terrain). 0=off (default); "
+                             "6.5=ICAO standard atmosphere.")
+    parser.add_argument("--mpas-land-beta", type=float, default=None,
+                        dest="mpas_land_beta",
+                        help="MPAS lane only: land evaporation efficiency in "
+                             "[0, 1] throttling the land-fraction surface "
+                             "humidity gradient (1.0=saturated wet swamp, "
+                             "default; ~0.6 first-order continental mean).")
     # --cloud-conv-cloud-max closes the AMIP CLI gap for the existing
     # ExperimentConfig.cloud_conv_cloud_max field (--q-c-diagnostic / --rh-crit /
     # --subgrid-autoconv already ship from run_coupled-mirrored #647 + #613).
@@ -1663,6 +1677,13 @@ def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
         sponge_sigma_top=(args.sponge_sigma_top
                           if args.sponge_sigma_top is not None
                           else _EXPERIMENT_DEFAULTS.sponge_sigma_top),
+        mpas_land_lapse_K_per_km=(
+            args.mpas_land_lapse_K_per_km
+            if args.mpas_land_lapse_K_per_km is not None
+            else _EXPERIMENT_DEFAULTS.mpas_land_lapse_K_per_km),
+        mpas_land_beta=(args.mpas_land_beta
+                        if args.mpas_land_beta is not None
+                        else _EXPERIMENT_DEFAULTS.mpas_land_beta),
         snow_albedo_feedback=args.snow_albedo_feedback,
         cloud_conv_cloud_max=args.conv_cloud_max,
         cloud_conv_cloud_condensate=args.conv_cloud_condensate,
diff --git a/tests/unit/test_mpas_land_boundary.py b/tests/unit/test_mpas_land_boundary.py
new file mode 100644
index 000000000..3933ca896
--- /dev/null
+++ b/tests/unit/test_mpas_land_boundary.py
@@ -0,0 +1,281 @@
+"""MPAS land surface boundary knobs (speckle fix, 2026-07-23).
+
+Covers the full chain:
+  - ``land_lapse_adjusted_surface_temperature`` (forcing.surface_utils):
+    lapse math, ocean identity, below-sea-level clip.
+  - ``beta_limited_surface_humidity`` (turbulence.surface_layer): swamp
+    identity at beta=1, closed surface at beta=0 over full land, ocean
+    fraction untouched.
+  - Factory guards: ``make_turbulence_physics`` / ``make_physics`` REFUSE
+    the knobs for non-MPAS model types (no silently-inert configuration).
+  - ``_make_mpas_turbulence`` end-to-end on a small SCVT mesh: beta < 1
+    reduces the land-cell surface moistening, leaves ocean cells
+    bit-identical.
+  - ``ExperimentConfig.validate_strict``: pipeline land-tile flags are
+    refused on the MPAS lane (they never execute there); the MPAS knobs are
+    refused on non-MPAS lanes; bounds are enforced.
+"""
+
+from __future__ import annotations
+
+import jax
+import jax.numpy as jnp
+import numpy as np
+import pytest
+
+jax.config.update("jax_enable_x64", True)
+
+from legoesm.forcing.surface_utils import (
+    land_lapse_adjusted_surface_temperature,
+)
+from legoesm.atmosphere.physics.turbulence.surface_layer import (
+    beta_limited_surface_humidity,
+)
+from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
+from legoesm.atmosphere.physics.turbulence.integration import (
+    _make_mpas_turbulence,
+    make_turbulence_physics,
+)
+
+
+# ---------------------------------------------------------------------------
+# Pure helpers
+# ---------------------------------------------------------------------------
+
+def test_lapse_ocean_fraction_unchanged():
+    T = jnp.array([300.0, 290.0])
+    f_land = jnp.array([0.0, 0.0])
+    z = jnp.array([0.0, 3000.0])
+    out = land_lapse_adjusted_surface_temperature(T, f_land, z, 6.5e-3)
+    np.testing.assert_allclose(np.asarray(out), np.asarray(T))
+
+
+def test_lapse_land_cooling_matches_rate():
+    # full land at 2 km with 6.5 K/km -> exactly -13 K
+    out = land_lapse_adjusted_surface_temperature(
+        jnp.array([300.0]), jnp.array([1.0]), jnp.array([2000.0]), 6.5e-3)
+    np.testing.assert_allclose(np.asarray(out), [300.0 - 13.0], rtol=1e-12)
+    # fractional land scales linearly
+    out_half = land_lapse_adjusted_surface_temperature(
+        jnp.array([300.0]), jnp.array([0.5]), jnp.array([2000.0]), 6.5e-3)
+    np.testing.assert_allclose(np.asarray(out_half), [300.0 - 6.5], rtol=1e-12)
+
+
+def test_lapse_below_sea_level_clipped_not_warmed():
+    out = land_lapse_adjusted_surface_temperature(
+        jnp.array([300.0]), jnp.array([1.0]), jnp.array([-400.0]), 6.5e-3)
+    np.testing.assert_allclose(np.asarray(out), [300.0])
+
+
+def test_lapse_zero_rate_identity():
+    T = jnp.array([288.0, 301.5])
+    out = land_lapse_adjusted_surface_temperature(
+        T, jnp.array([1.0, 0.3]), jnp.array([500.0, 1500.0]), 0.0)
+    np.testing.assert_allclose(np.asarray(out), np.asarray(T))
+
+
+def test_beta_one_is_saturated_identity():
+    q_sat = jnp.array([0.025, 0.020])
+    q_air = jnp.array([0.010, 0.015])
+    f_land = jnp.array([1.0, 0.4])
+    out = beta_limited_surface_humidity(q_sat, q_air, f_land, 1.0)
+    np.testing.assert_allclose(np.asarray(out), np.asarray(q_sat))
+
+
+def test_beta_zero_full_land_closes_surface():
+    q_sat = jnp.array([0.025])
+    q_air = jnp.array([0.010])
+    out = beta_limited_surface_humidity(q_sat, q_air, jnp.array([1.0]), 0.0)
+    np.testing.assert_allclose(np.asarray(out), np.asarray(q_air))
+
+
+def test_beta_ocean_fraction_stays_saturated():
+    q_sat = jnp.array([0.025])
+    q_air = jnp.array([0.010])
+    out = beta_limited_surface_humidity(q_sat, q_air, jnp.array([0.0]), 0.3)
+    np.testing.assert_allclose(np.asarray(out), np.asarray(q_sat))
+
+
+def test_beta_half_land_half_beta_interpolates():
+    # q_sfc = q_air + (1 - f*(1-beta)) * (q_sat - q_air)
+    q_sat, q_air, f, beta = 0.030, 0.010, 0.5, 0.6
+    out = beta_limited_surface_humidity(
+        jnp.array([q_sat]), jnp.array([q_air]), jnp.array([f]), beta)
+    expected = q_air + (1.0 - f * (1.0 - beta)) * (q_sat - q_air)
+    np.testing.assert_allclose(np.asarray(out), [expected], rtol=1e-12)
+
+
+# ---------------------------------------------------------------------------
+# Factory guards: refuse silently-inert configurations
+# ---------------------------------------------------------------------------
+
+@pytest.mark.parametrize("model_type", ["hydrostatic", "nonhydrostatic",
+                                        "spectral_pe"])
+def test_make_turbulence_physics_refuses_knobs_off_mpas(model_type):
+    with pytest.raises(ValueError, match="MPAS"):
+        make_turbulence_physics(
+            TurbulenceConfig(scheme="louis"), model_type, 300.0,
+            f_land=jnp.zeros(4))
+    with pytest.raises(ValueError, match="MPAS"):
+        make_turbulence_physics(
+            TurbulenceConfig(scheme="louis"), model_type, 300.0,
+            land_beta=0.5)
+
+
+def test_make_physics_refuses_knobs_off_mpas():
+    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
+    cfg = PhysicsConfig()
+    with pytest.raises(ValueError, match="MPAS-only"):
+        make_physics(cfg, model_type="hydrostatic", dt=300.0,
+                     land_beta=0.5)
+
+
+# ---------------------------------------------------------------------------
+# End-to-end on a small SCVT mesh: beta throttles land moistening only
+# ---------------------------------------------------------------------------
+
+@pytest.fixture(scope="module")
+def mpas_mesh():
+    from legoesm.grids.voronoi import create_voronoi_mesh
+    return create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
+
+
+@pytest.fixture(scope="module")
+def sigma_coord():
+    from legoesm.grids.vertical import create_sigma_coordinate
+    return create_sigma_coordinate(8, sigma_top=0.05)
+
+
+@pytest.fixture(scope="module")
+def mpas_state(mpas_mesh, sigma_coord):
+    from legoesm.atmosphere.forcing.idealized.held_suarez import (
+        held_suarez_init_mpas,
+    )
+    state = held_suarez_init_mpas(
+        mpas_mesh, sigma_coord, T_init=290.0, perturbation_amplitude=0.0)
+    ncell = state.T.data.shape[0]
+    nlev = state.T.data.shape[1]
+    # Moist but subsaturated boundary layer so a warm anchor evaporates.
+    q_v = jnp.full((ncell, nlev), 0.008, dtype=state.T.data.dtype)
+    return state._replace(tracers={"q_v": q_v})
+
+
+def test_mpas_turbulence_beta_throttles_land_only(mpas_mesh, sigma_coord,
+                                                  mpas_state):
+    ncell = mpas_state.T.data.shape[0]
+    # Land on one hemisphere of cells, ocean on the other.
+    f_land = jnp.asarray(
+        (np.arange(ncell) < ncell // 2).astype(np.float64))
+    T_sfc = jnp.full((ncell,), 302.0)
+    forcing = {"T_sfc": T_sfc}
+    tc = TurbulenceConfig(scheme="louis")
+
+    fn_swamp = _make_mpas_turbulence(tc, 300.0)
+    fn_beta = _make_mpas_turbulence(tc, 300.0, f_land=f_land, land_beta=0.4)
+
+    tend_swamp = fn_swamp(mpas_state, mpas_mesh, sigma_coord, forcing=forcing)
+    tend_beta = fn_beta(mpas_state, mpas_mesh, sigma_coord, forcing=forcing)
+    if isinstance(tend_swamp, tuple):
+        tend_swamp = tend_swamp[0]
+    if isinstance(tend_beta, tuple):
+        tend_beta = tend_beta[0]
+
+    dq_swamp = np.asarray(tend_swamp.tracer_tendencies["q_v"])[..., -1]
+    dq_beta = np.asarray(tend_beta.tracer_tendencies["q_v"])[..., -1]
+    land = np.asarray(f_land) > 0.5
+    ocean = ~land
+
+    # Ocean cells: bit-identical (the knob must not touch them).
+    np.testing.assert_array_equal(dq_beta[ocean], dq_swamp[ocean])
+    # Land cells: swamp moistens the surface layer; beta=0.4 moistens LESS
+    # (strictly, since the anchor is 12 K warmer than the air => q_sat >
+    # q_air everywhere).
+    assert (dq_swamp[land] > 0.0).all(), "warm swamp must moisten land BL"
+    assert (dq_beta[land] < dq_swamp[land]).all()
+    # And the throttled flux is still non-negative (beta in (0,1) cannot
+    # reverse the gradient sign).
+    assert (dq_beta[land] >= 0.0).all()
+
+
+def test_mpas_turbulence_beta_one_bit_identical(mpas_mesh, sigma_coord,
+                                                 mpas_state):
+    """beta=1 with f_land supplied must be byte-identical to no knobs."""
+    ncell = mpas_state.T.data.shape[0]
+    f_land = jnp.asarray((np.arange(ncell) % 2).astype(np.float64))
+    forcing = {"T_sfc": jnp.full((ncell,), 300.0)}
+    tc = TurbulenceConfig(scheme="louis")
+    base = _make_mpas_turbulence(tc, 300.0)(
+        mpas_state, mpas_mesh, sigma_coord, forcing=forcing)
+    with_knob = _make_mpas_turbulence(tc, 300.0, f_land=f_land, land_beta=1.0)(
+        mpas_state, mpas_mesh, sigma_coord, forcing=forcing)
+    if isinstance(base, tuple):
+        base, with_knob = base[0], with_knob[0]
+    np.testing.assert_array_equal(
+        np.asarray(base.tracer_tendencies["q_v"]),
+        np.asarray(with_knob.tracer_tendencies["q_v"]))
+
+
+# ---------------------------------------------------------------------------
+# validate_strict lane guards
+# ---------------------------------------------------------------------------
+
+def _mpas_cfg(**kw):
+    from legoesm.driver.config import (
+        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
+    )
+    return ExperimentConfig(
+        grid=GridConfig(grid_type="mpas", resolution=1, nlev=8,
+                        vertical_coord="sigma"),
+        dycore=DycoreConfig(dt=600.0, discretization="mpas"),
+        output=OutputConfig(diag_days=1),
+        days=1, dataset="analytical", radiation="gray",
+        **kw,
+    )
+
+
+def _cdgrid_cfg(**kw):
+    from legoesm.driver.config import (
+        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
+    )
+    return ExperimentConfig(
+        grid=GridConfig(resolution=8, nlev=8),
+        dycore=DycoreConfig(dt=600.0),
+        output=OutputConfig(diag_days=1),
+        days=1, dataset="analytical", radiation="gray",
+        **kw,
+    )
+
+
+def test_validate_mpas_accepts_land_boundary_knobs():
+    _mpas_cfg(mpas_land_lapse_K_per_km=6.5, mpas_land_beta=0.6).validate_strict()
+
+
+@pytest.mark.parametrize("flag", ["slab_land_active", "land_soil_bucket",
+                                  "surface_tiled"])
+def test_validate_mpas_refuses_pipeline_land_flags(flag):
+    kw = {flag: True}
+    if flag == "land_soil_bucket":
+        kw["slab_land_active"] = True   # bucket alone already fails elsewhere
+    if flag == "surface_tiled":
+        kw["slab_land_active"] = True   # tiled alone already fails elsewhere
+        kw["turbulence"] = "louis"
+    with pytest.raises(ValueError, match="silently inert on the MPAS lane"):
+        _mpas_cfg(**kw).validate_strict()
+
+
+def test_validate_cdgrid_refuses_mpas_knobs():
+    with pytest.raises(ValueError, match="MPAS-lane"):
+        _cdgrid_cfg(mpas_land_beta=0.6).validate_strict()
+    with pytest.raises(ValueError, match="MPAS-lane"):
+        _cdgrid_cfg(mpas_land_lapse_K_per_km=6.5).validate_strict()
+
+
+def test_validate_bounds():
+    with pytest.raises(ValueError, match="mpas_land_beta"):
+        _mpas_cfg(mpas_land_beta=1.5).validate_strict()
+    with pytest.raises(ValueError, match="mpas_land_beta"):
+        _mpas_cfg(mpas_land_beta=float("nan")).validate_strict()
+    with pytest.raises(ValueError, match="mpas_land_lapse_K_per_km"):
+        _mpas_cfg(mpas_land_lapse_K_per_km=-1.0).validate_strict()
+    with pytest.raises(ValueError, match="mpas_land_lapse_K_per_km"):
+        _mpas_cfg(mpas_land_lapse_K_per_km=25.0).validate_strict()
diff --git a/tests/unit/test_run_amip_cli.py b/tests/unit/test_run_amip_cli.py
index e6c9985c1..488b5f3e4 100644
--- a/tests/unit/test_run_amip_cli.py
+++ b/tests/unit/test_run_amip_cli.py
@@ -441,6 +441,25 @@ def test_sponge_flags_flow_to_config():
     assert cfg.sponge_enabled is True
     assert cfg.sponge_coeff_per_day == 4.0
     assert cfg.sponge_sigma_top == 0.2
+
+
+def test_mpas_land_boundary_flags_flow_to_config():
+    """--mpas-land-lapse-k-per-km / --mpas-land-beta round-trip (MPAS land
+    surface boundary, 2026-07-23 speckle fix); defaults byte-identical OFF."""
+    parser = build_arg_parser()
+    cfg_default = build_config_from_args(_postprocess_args(
+        parser.parse_args(["--dataset", "analytical"]), parser))
+    assert cfg_default.mpas_land_lapse_K_per_km == 0.0
+    assert cfg_default.mpas_land_beta == 1.0
+
+    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
+        "--dataset", "analytical",
+        "--grid-type", "voronoi", "--discretization", "mpas",
+        "--mpas-land-lapse-k-per-km", "6.5", "--mpas-land-beta", "0.6",
+    ]), parser))
+    assert cfg.mpas_land_lapse_K_per_km == 6.5
+    assert cfg.mpas_land_beta == 0.6
+    cfg.validate_strict()
 def test_land_surface_scheme_validate_strict_rejects_unknown():
     """validate_strict() rejects an unknown surface scheme (dispatch hardening —
     a typo must fail early, not silently fall through in model_driver)."""
```
