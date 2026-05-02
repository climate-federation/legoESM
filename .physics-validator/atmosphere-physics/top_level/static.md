# Top-level physics — static analysis

## A. Confirmed correctness

1. **`combined.py` tracer accumulation**: per-module tracer tendencies accumulated using `dict.update / +=` keyed on tracer name. Tracer pytree structure preserved via `zero_like_tracers` (duck-types Field vs raw arrays). ✓
2. **Hydrostatic-tendencies dimensions**: `du_dt`, `dv_dt`, `dT_dt`, `dp_s_dt`, `dphis_dt` all built with the right Field metadata (units, dims). ✓
3. **`diagnose_grid_w_from_omega`**: virtual T correction uses `1/ε - 1 = 1/0.622 - 1 ≈ 0.608`. Comment matches and uses `constants.R_d / constants.R_v`. ✓
4. **Implicit-Euler-style update path**: every Bechtold/Tiedtke/ZM scheme returns its physics state update as a dict, merged into `phys_updates` correctly in `combined._make_hydrostatic_combined` (line 196-205, 220-232).

## B. Issues (RANKED)

### B1. **`compute_heights_from_sigma` integrates from surface upward then uses `jnp.pad`** [VERIFIED]
`_shared.py:49-54`: cumulative sum of dz reversed gives heights from surface upward (z[surface] = 0, z[top] = z_max). Standard hydrostatic integration. ✓

### B2. **`thermodynamics.compute_cape:298`: `R_d * sum(max(0, T_p - T_e) * dp / p)` integration form** [VERIFIED]
The CAPE integrand is `R_d (T_p - T_e) dp/p` (positive parcel buoyancy only). Sign and integration are standard. ✓

### B3. **`thermodynamics.compute_moist_adiabat:230` Trapezoidal predictor-corrector via `jax.lax.scan`** [VERIFIED]
Predictor: Euler. Corrector: trapezoidal. `T_new = jnp.clip(T_new, 100.0, 350.0)` — bounds for AD safety, but the clip removes gradient if profile passes through. Acceptable for limit cases.

### B4. **`thermodynamics.compute_moist_adiabat`** assumes saturated parcel everywhere [INFO]
`moist_adiabat_lapse_rate` uses `saturation_mixing_ratio` at every level — implicitly assumes parcel is saturated. For unsaturated initial parcels (LCL > base), the adiabat above LCL is correct, but BELOW LCL the parcel should follow a *dry* adiabat. The current implementation uses the moist lapse rate the whole way down, which underestimates the adiabat-environmental contrast in subsaturated boundary layers.

This is a long-standing simplification in convection schemes; many schemes (SBM, DCA, Kuo, ZM, KF, Emanuel, Tiedtke, Bechtold) consume the result. **CAPE values from this function are systematically biased low for unsaturated launch parcels** because the parcel curve is too steep below LCL.

### B5. **`compute_moist_adiabat` `dtype` promotion** [VERIFIED]
Lines 226-228: pins both T_base and p_rev to `jnp.result_type(T_base, p_rev)` to avoid scan carry-dtype mismatch. ✓

### B6. **`combined.py: zero_tend = HydrostaticTendencies(... dv_dt=dv_dt_zero)`** [VERIFIED]
Lines 174-191: when no physics is active, builds zero tendencies. Handles optional `dv_dt` correctly via `has_v` check.

### B7. **`physics_state.update_physics_state`** [VERIFIED]
Returns a fresh PhysicsState with each field updated only if present in `updates` dict. Safe pattern.

### B8. **`compute_moisture_convergence` uses limiter-disabled FV divergence** [INTENTIONAL]
`_shared.py:412-419`: documented decision — smooth gradient required for AD through convection trigger. The moisture convergence diagnostic is *not* an advected quantity, so disabling the limiter is justified. ✓

## C. Top issues

* **B4** (compute_moist_adiabat below-LCL assumption) — systematic CAPE bias. Documented but worth a note. Fix would require an explicit LCL diagnosis before integration, switching to dry adiabat below it. Significant refactor.
