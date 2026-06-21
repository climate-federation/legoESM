Reading prompt from stdin...
OpenAI Codex v0.125.0 (research preview)
--------
workdir: /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
model: gpt-5.5
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019de5e0-cc6c-7f91-9486-29bd043eae57
--------
user
# Codex Adversarial Review — Microphysics (Cycle 2 — Iteration 3)

You are an independent adversarial physics-parameterization reviewer for legoESM. In iteration 2 you flagged the bug "subsaturated clear air can create negative cloud water" and I have applied a fix. Please verify the fix is correct.

## The bug (you flagged it in iteration 2)

`saturation_adjustment(T, q_v, p_full, dt)` returns `condensation = sigmoid(s · excess) · excess / dt` where `excess = q_v - q_sat`. For `q_v < q_sat` (subsaturated), `condensation < 0`, but Morrison/Thompson/Kessler all add this directly to `dq_c_dt` without checking q_c availability. In a clear-air column (q_c = 0), explicit Euler then drives q_c to a negative value of order ~3e-4 kg/kg per timestep at 95% RH, dt=1200 s.

## The fix

### Part 1: `_warm_rain.saturation_adjustment` now donor-clamps the evaporation branch

```python
def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0, q_c=None):
    q_sat = saturation_mixing_ratio(T, p_full)
    excess = q_v - q_sat
    cond_frac = jax.nn.sigmoid(sharpness * excess)
    condensation = cond_frac * excess / dt
    if q_c is not None:
        # Evaporation rate (negative ``condensation``) bounded by available q_c:
        # |condensation| × dt ≤ q_c, i.e. condensation ≥ -q_c / dt.
        q_c_avail = jnp.clip(q_c, 0.0, None)
        condensation = jnp.maximum(
            condensation, -q_c_avail / jnp.maximum(dt, 1e-10),
        )
    return condensation, q_sat
```

### Part 2: Morrison/Thompson include `cond_evap_sink = max(-condensation, 0)` in the donor clamp

```python
cond_evap_sink = jnp.maximum(-condensation, 0.0)
qc_sink_total = (
    dq_c_au + dq_c_ac + bergeron + riming_i + riming_s + cond_evap_sink
)
qc_avail = jnp.clip(q_c, 0.0)
qc_scale = jnp.minimum(
    1.0,
    qc_avail / jnp.maximum(qc_sink_total * dt_safe, 1e-30),
)
# Scale all sinks (including condensation when negative):
dq_c_au = dq_c_au * qc_scale
# ... etc
condensation = jnp.where(
    condensation < 0.0, condensation * qc_scale, condensation,
)
```

### Part 3: Kessler (which inlines saturation logic) gets the same clamp inline

```python
condensation = cond_frac * excess / dt  # signed
q_c_avail = jnp.clip(q_c, 0.0, None)
condensation = jnp.maximum(
    condensation, -q_c_avail / jnp.maximum(dt, 1e-10),
)
# Kessler doesn't have a multi-process donor clamp because:
#   accretion = k_ac * q_c * q_r^0.875  (proportional to q_c)
#   autoconv = max(q_c_updated - threshold, 0) * rate  (gated)
# So q_c=0 → accretion=0, autoconv=0; only condensation needs clamping.
```

## Mass conservation check

For Morrison/Thompson, after my fix:
- `dq_v_dt = -condensation + evaporation - dq_i_dep` (condensation already scaled)
- `dq_c_dt = condensation - (sinks scaled by qc_scale)` (condensation already scaled)
- Total `(dq_v + dq_c)` from saturation branch: `-condensation_scaled + condensation_scaled = 0` ✓

For Kessler:
- `dq_v_dt = -condensation + evaporation` (condensation now donor-clamped)
- `dq_c_dt = condensation - autoconv - accretion`
- Saturation contribution: `-condensation + condensation = 0` ✓

## Test verification

```
test_subsaturated_clear_air_does_not_create_negative_qc[kessler]: PASS
test_subsaturated_clear_air_does_not_create_negative_qc[seifert_beheng]: PASS
test_subsaturated_clear_air_does_not_create_negative_qc[morrison]: PASS
test_subsaturated_clear_air_does_not_create_negative_qc[thompson]: PASS
```

All 257 physics-related tests still pass.

## Your task

1. Verify the fix is mathematically and dimensionally correct.
2. Verify mass conservation holds in both clear-air (q_c=0) and cloudy (q_c>0) columns.
3. Confirm the fix handles all four code paths (Kessler, SB, Morrison, Thompson).
4. Check for any new bugs introduced by these changes.
5. Specifically: in Morrison/Thompson, the `condensation` variable is referenced *both* in the donor clamp (where it's scaled when negative) AND in `dq_v_dt = -condensation + evaporation - dq_i_dep`. The scaling is in-place via `condensation = jnp.where(...)`, so the subsequent reference picks up the scaled value. Verify this is correct.

## Outstanding deferred findings (from iteration 2)

These were not fixed and may need deferral notes:

1. **Hines drag dimensional inconsistency** (P1, deferred): `drag = rho * (sigma_grown - sigma_new)` has units `kg/(m²·s)` not Pa. Requires re-tuning so deferred.
2. **N_i nucleation creates number without ice mass** (P2): `dN_i_nuc > 0` but `dq_i_dep ∝ q_i`, so q_i can't grow from 0.
3. **Thompson `graupel_frac` uses pre-clamp `total_riming`** (P3): mass-conservative but threshold decision is inconsistent with what the limiter returns.

Provide:
- Confirmation that the subsaturated-clear-air fix is correct (or NEW concerns).
- Status assessment for deferred items.
- Any NEW bugs you see in the post-fix code.

If you have no substantive findings, state so explicitly.

codex
I’ll review the implemented paths directly in the repo, starting with the saturation helper and the four scheme implementations, then I’ll check tests and any limiter ordering issues around the changed variables.
  → Inspect warm-rain saturation helper and scheme call sites
  • Trace Morrison/Thompson/Kessler/SB mass tendencies and limiter ordering
  • Review tests covering clear-air and cloudy donor behavior
  • Summarize findings and deferred-item status
exec
/bin/zsh -lc 'rg --files | rg "microphysics|warm_rain|physics|test"' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/ml/physics/data.py
src/legoesm/ml/physics/evaluate.py
src/legoesm/ml/physics/train.py
src/legoesm/ml/physics/model.py
src/legoesm/ml/physics/__init__.py
src/legoesm/ml/physics/io.py
src/legoesm/ml/physics/plotting.py
scripts/ml_physics_parameterization.py
scripts/run_ocean_test_matrix.py
scripts/validate_convection_physics.py
scripts/run_held_suarez_rrtmgp_allgrids_test.py
scripts/run_atmosphere_test_matrix.py
scripts/run_ocean_spectral_tests.py
scripts/run_sea_ice_test_matrix.py
FIXME_test_audit_prompt.md
src/legoesm/atmosphere/physics/ml_parameterization.py
src/legoesm/atmosphere/physics/combined.py
src/legoesm/atmosphere/physics/learned_column.py
src/legoesm/atmosphere/physics/neural_physics.py
scripts/ocean_test_matrix/xarray_output.py
scripts/ocean_test_matrix/timeloop.py
scripts/ocean_test_matrix/postprocessing.py
scripts/ocean_test_matrix/extraction.py
scripts/ocean_test_matrix/cli.py
scripts/ocean_test_matrix/experiments.py
scripts/ocean_test_matrix/setup.py
scripts/ocean_test_matrix/__init__.py
scripts/ocean_test_matrix/regridding.py
scripts/ocean_test_matrix/diagnostic_io.py
scripts/ocean_test_matrix/testcase.py
scripts/ocean_test_matrix/checkpoint.py
scripts/ocean_test_matrix/config.py
src/legoesm/atmosphere/physics/gravity_wave_drag/output.py
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py
src/legoesm/atmosphere/physics/gravity_wave_drag/rayleigh.py
src/legoesm/atmosphere/physics/gravity_wave_drag/integration.py
src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py
src/legoesm/atmosphere/physics/gravity_wave_drag/ml_emulator.py
src/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py
src/legoesm/atmosphere/physics/gravity_wave_drag/__init__.py
src/legoesm/atmosphere/physics/gravity_wave_drag/lindzen.py
src/legoesm/atmosphere/physics/gravity_wave_drag/config.py
src/legoesm/atmosphere/physics/__init__.py
src/legoesm/atmosphere/physics/microphysics/output.py
src/legoesm/atmosphere/physics/microphysics/sundqvist.py
src/legoesm/atmosphere/physics/microphysics/thompson.py
src/legoesm/atmosphere/physics/microphysics/integration.py
src/legoesm/atmosphere/physics/microphysics/ml_emulator.py
src/legoesm/atmosphere/physics/microphysics/__init__.py
src/legoesm/atmosphere/physics/microphysics/kessler.py
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py
src/legoesm/atmosphere/physics/microphysics/config.py
src/legoesm/atmosphere/physics/microphysics/morrison.py
src/legoesm/atmosphere/physics/physics_state.py
src/legoesm/atmosphere/physics/thermodynamics.py
src/legoesm/driver/physics_pipeline.py
src/legoesm/atmosphere/physics/_shared.py
docs/ocean_test_experiments_audit.md
docs/science/ml_physics_parameterization.md
src/legoesm/atmosphere/physics/turbulence/tke.py
src/legoesm/atmosphere/physics/turbulence/output.py
src/legoesm/atmosphere/physics/turbulence/surface_layer.py
src/legoesm/atmosphere/physics/turbulence/ysu.py
src/legoesm/atmosphere/physics/turbulence/clubb_lite.py
src/legoesm/atmosphere/physics/turbulence/louis.py
src/legoesm/atmosphere/physics/turbulence/integration.py
src/legoesm/atmosphere/physics/turbulence/__init__.py
src/legoesm/atmosphere/physics/turbulence/smagorinsky.py
src/legoesm/atmosphere/physics/turbulence/vertical_diffusion.py
src/legoesm/atmosphere/physics/turbulence/holtslag_boville.py
src/legoesm/atmosphere/physics/turbulence/config.py
src/legoesm/atmosphere/physics/turbulence/edmf.py
src/legoesm/atmosphere/physics/turbulence/pbl_height.py
src/legoesm/ocean/physics/combined.py
scripts/diagnostic/diag_fv3_csw_test.py
src/legoesm/atmosphere/physics/clouds/__init__.py
src/legoesm/atmosphere/physics/clouds/config.py
src/legoesm/atmosphere/physics/clouds/cloud_fraction.py
docs/ocean_test_matrix_changelog.md
src/legoesm/ocean/physics/surface_forcing/output.py
src/legoesm/ocean/physics/surface_forcing/wind_profiles.py
src/legoesm/ocean/physics/surface_forcing/integration.py
src/legoesm/ocean/physics/surface_forcing/prescribed.py
src/legoesm/ocean/physics/surface_forcing/__init__.py
src/legoesm/ocean/physics/surface_forcing/config.py
src/legoesm/ocean/physics/surface_forcing/bulk_formulas.py
src/legoesm/ocean/physics/surface_forcing/restoring.py
src/legoesm/atmosphere/physics/convection/output.py
src/legoesm/atmosphere/physics/convection/tiedtke.py
src/legoesm/atmosphere/physics/convection/sbm.py
src/legoesm/atmosphere/physics/convection/bechtold.py
src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py
src/legoesm/atmosphere/physics/convection/_plume.py
src/legoesm/atmosphere/physics/convection/kain_fritsch.py
src/legoesm/atmosphere/physics/convection/mass_flux.py
src/legoesm/atmosphere/physics/convection/dca.py
src/legoesm/atmosphere/physics/convection/integration.py
src/legoesm/atmosphere/physics/convection/__init__.py
src/legoesm/atmosphere/physics/convection/kuo.py
tests/williamson_diagnostic.py
src/legoesm/atmosphere/physics/convection/config.py
src/legoesm/atmosphere/physics/convection/emanuel.py
src/legoesm/atmosphere/physics/convection/_triggers.py
src/legoesm/atmosphere/physics/radiation/config.py
tests/debug/spectral_hs100_l10.png
tests/debug/spectral_hs100.png
tests/debug/spectral_hs100_freedrag.png
tests/debug/spectral_hs100_correct_nu.png
tests/debug/spectral_hs100_dealias.png
tests/debug/spectral_hs100_eqinit.png
tests/debug/spectral_hs100_implicit.png
tests/debug/spectral_hs100_final.png
tests/debug/spectral_hs100_strong.png
src/legoesm/ocean/physics/shortwave_penetration.py
src/legoesm/ocean/physics/mpas_physics.py
src/legoesm/ocean/physics/mixing.py
src/legoesm/atmosphere/physics/radiation/gray.py
src/legoesm/atmosphere/physics/radiation/output.py
src/legoesm/atmosphere/physics/radiation/solar.py
tests/stress/test_phase5_cmip_io.py
tests/stress/test_phase1_sea_ice.py
tests/stress/test_phase0_infrastructure.py
tests/stress/test_cmip_operationalization.py
tests/stress/test_phase2_coupled.py
tests/stress/test_phase1_ocean_slab.py
tests/stress/test_phase4_coupled_mpi.py
tests/stress/test_phase1_radiation_ghg.py
tests/stress/test_phase3_restart.py
tests/stress/__init__.py
tests/stress/test_phase4_sea_ice_mpi.py
tests/stress/test_phase7_multiyear.py
tests/stress/test_phase6_cmip_e2e.py
tests/stress/test_phase1_land_carbon.py
tests/conftest.py
src/legoesm/ocean/physics/convection/output.py
src/legoesm/ocean/physics/convection/enhanced_diffusion.py
src/legoesm/ocean/physics/convection/plume.py
src/legoesm/ocean/physics/convection/integration.py
src/legoesm/ocean/physics/convection/__init__.py
src/legoesm/ocean/physics/convection/config.py
src/legoesm/ocean/physics/__init__.py
src/legoesm/atmosphere/physics/radiation/__init__.py
src/legoesm/atmosphere/physics/radiation/rrtmgp_radiation.py
src/legoesm/atmosphere/physics/radiation/integration.py
tests/da/test_control_vector.py
tests/da/test_minimizer.py
tests/da/test_background_error.py
tests/da/test_gen_be.py
tests/da/test_incremental.py
tests/da/test_preconditioning.py
tests/da/__init__.py
tests/da/test_cycling.py
tests/da/test_observation.py
src/legoesm/ocean/physics/lateral_mixing/output.py
src/legoesm/ocean/physics/lateral_mixing/harmonic.py
src/legoesm/ocean/physics/lateral_mixing/gm_redi_mpas.py
src/legoesm/ocean/physics/lateral_mixing/biharmonic.py
src/legoesm/ocean/physics/lateral_mixing/integration.py
src/legoesm/ocean/physics/lateral_mixing/gm_redi.py
src/legoesm/ocean/physics/lateral_mixing/backscatter.py
src/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py
src/legoesm/ocean/physics/lateral_mixing/__init__.py
src/legoesm/ocean/physics/lateral_mixing/config.py
src/legoesm/ocean/physics/lateral_mixing/_gm_redi_common.py
config/williamson_test5.yaml
config/williamson_test2.yaml
src/legoesm/ocean/physics/vertical_mixing/kpp.py
src/legoesm/ocean/physics/vertical_mixing/output.py
src/legoesm/ocean/physics/vertical_mixing/constant.py
src/legoesm/ocean/physics/vertical_mixing/richardson.py
src/legoesm/ocean/physics/vertical_mixing/implicit_solver.py
src/legoesm/ocean/physics/vertical_mixing/integration.py
src/legoesm/ocean/physics/vertical_mixing/__init__.py
src/legoesm/ocean/physics/vertical_mixing/config.py
tests/validation/test_ec_eigenvalues2.py
tests/validation/test_ensemble_correctness.py
tests/validation/README_DYCORE_PROGRESSION.md
tests/validation/bench_spectral_nh.py
tests/validation/test_continuous_stability.py
tests/validation/test_corrected_eigenvalues.py
tests/validation/test_differentiability_regression.py
tests/validation/bench_spectral_sw.py
tests/validation/test_isolate_instability.py
tests/validation/test_restart_reproducibility.py
tests/validation/test_conservation_baseline.py
tests/validation/test_scaling_readiness.py
tests/validation/bench_spectral_pe.py
tests/validation/validation_differentiability_all.py
tests/validation/test_precision_amip.py
tests/validation/test_evar_instability.py
tests/validation/run_dycore_progression_suite.py
tests/validation/__init__.py
tests/validation/test_amip_validation.py
tests/validation/test_ec_eigenvalues.py
tests/test_mpas_conservation.py
tests/test_smagorinsky_biharmonic_comprehensive.py
tests/da/integration/test_ocean_4dvar.py
tests/da/integration/test_end_to_end.py
tests/da/integration/test_pe_4dvar.py
tests/da/integration/__init__.py
tests/da/test_cost_function.py
tests/parallel/test_scaling_operators.py
tests/parallel/test_cubesphere_exchange.py
tests/ocean/run_ocean_all_grids_matrix.py
src/legoesm/ocean/physics/bottom_drag/output.py
src/legoesm/ocean/physics/bottom_drag/integration.py
src/legoesm/ocean/physics/bottom_drag/__init__.py
src/legoesm/ocean/physics/bottom_drag/linear.py
src/legoesm/ocean/physics/bottom_drag/quadratic.py
src/legoesm/ocean/physics/bottom_drag/config.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/__init__.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/constants.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp_common.py
tests/integration/__init__.py
tests/land/validation/test_bulk_flux_differentiability.py
tests/land/validation/__init__.py
tests/land/validation/test_bulk_flux_all_tiles.py
tests/land/test_land_stability.py
tests/land/__init__.py
tests/core/test_vertical_remap.py
tests/core/test_weno.py
tests/atmosphere/__init__.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/config/radiative_transfer.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/config/__init__.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/interpolation.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/LICENSE
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/constants.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_cloud_optics.py
tests/ocean/validation/test_differentiability_ocean.py
tests/ocean/validation/__init__.py
tests/ocean/__init__.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_gas_optics_longwave.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/gas_optics.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/optics.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/cloud_optics.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/atmospheric_state.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_gas_optics_base.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_volume_mixing_ratio.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/data_loader_base.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/__init__.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/optics_utils.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/lookup_gas_optics_shortwave.py
tests/unit/test_land_ice_stomata.py
tests/unit/test_bechtold.py
tests/unit/test_sfno.py
tests/unit/test_land_ice_slab_land.py
tests/unit/test_hybrid_vertical.py
tests/unit/test_ensemble_diagnostics.py
tests/unit/test_land_ice_units.py
tests/unit/test_earth_system_driver.py
tests/unit/test_physics_grid_adapters.py
tests/unit/test_land_ice_sea_ice_dynamics.py
tests/unit/test_land_ice_soil_hydraulics.py
tests/unit/test_physical_balances.py
tests/unit/test_component_factory.py
tests/unit/test_rce_script.py
tests/unit/test_ensemble.py
tests/unit/test_backend_precision.py
tests/unit/test_land_ice_soil_grid.py
tests/unit/test_corrections.py
tests/unit/test_duogrid_metrics.py
tests/unit/test_scale_device_mesh.py
tests/unit/test_version_guards.py
tests/unit/test_plot_amip.py
tests/unit/test_physics_ocean.py
tests/unit/test_diff_sea_ice.py
tests/unit/test_greens_function.py
tests/unit/test_operators_fv.py
tests/unit/test_diff_atmosphere_physics.py
tests/unit/test_fc_gram.py
tests/unit/test_physics_gwd.py
tests/unit/test_land_ice_integrated.py
tests/unit/test_land_ice_snow.py
tests/unit/test_distributed_checkpoint.py
tests/unit/test_warm_rain.py
tests/unit/test_diff_atmosphere_dynamics.py
tests/unit/test_mpas_atmosphere.py
tests/unit/test_land_ice_albedo.py
tests/unit/test_hardware_runtime_config.py
tests/unit/test_tiedtke.py
tests/unit/test_compiled_segments.py
tests/unit/test_scale_global_reductions.py
tests/unit/test_precision_dtype_contracts.py
tests/unit/test_physics_convection.py
tests/unit/test_williamson2_cdgrid.py
tests/unit/test_physics_smoke.py
tests/unit/test_cmor_experiments_restart.py
tests/unit/test_convection_plume.py
tests/unit/test_installed_package_imports.py
tests/unit/test_state_checkpoint.py
tests/unit/test_grid_protocol.py
tests/unit/test_equation_fixes.py
tests/unit/test_spectral_pe_training_tracers.py
tests/unit/test_surface_exchange.py
tests/unit/test_land_ice_sea_ice_thermo.py
tests/unit/test_batch_allreduce.py
tests/unit/test_cmor_output_dtype.py
tests/unit/test_regional_voronoi.py
tests/unit/test_parallel.py
tests/unit/test_coupled_esm.py
tests/unit/test_fv3_audit_harness.py
tests/unit/test_ml_physics_workflow.py
tests/unit/test_physics_surface_models.py
tests/unit/test_weatherbench.py
tests/unit/test_diff_data_assimilation.py
tests/unit/test_halo.py
tests/unit/test_production_blockers.py
tests/unit/test_scale_tpu_compat.py
tests/unit/test_config_validation.py
tests/unit/test_timestepping.py
tests/unit/test_coupler.py
tests/unit/test_issue_fixes.py
tests/unit/test_step_cache.py
tests/unit/test_convection_triggers.py
tests/unit/test_scale_runtime.py
tests/unit/test_physics_state_migration.py
tests/unit/test_field.py
tests/unit/test_land_ice_sea_ice_transport.py
tests/unit/test_dcmip_transport.py
tests/unit/test_tracer_transport.py
tests/unit/test_diff_land.py
tests/unit/test_symmetry_invariance.py
tests/unit/test_ml_physics_parameterization.py
tests/unit/test_regridding_cubedsphere.py
tests/unit/test_physics_turbulence.py
tests/unit/test_runtime_bootstrap.py
tests/unit/test_sfno_s2s.py
tests/unit/test_operators_fc.py
tests/unit/test_diff_ocean.py
tests/unit/test_multi_gpu.py
tests/unit/test_physics_units.py
tests/unit/test_spectral_dycores_comprehensive.py
tests/unit/test_device_config.py
tests/unit/__init__.py
tests/unit/test_sharded_dynamics.py
tests/unit/test_scale_sharded_dynamics.py
tests/unit/test_operators.py
tests/unit/test_external_forcing.py
tests/unit/test_land_ice_lake.py
tests/unit/test_scale_jit_health.py
tests/unit/test_distributed_layout.py
tests/unit/test_conservation_laws.py
tests/unit/test_training_modules.py
tests/unit/test_gradient_checkpointing.py
tests/unit/test_backend_guard.py
tests/unit/test_scale_halo.py
tests/unit/test_neural_gcm_spectral.py
tests/unit/test_thermodynamics.py
tests/unit/test_conservation.py
tests/unit/test_kain_fritsch.py
tests/unit/test_diff_taylor_tests.py
tests/unit/test_scale_ensemble.py
tests/unit/test_zarr_checkpoint.py
tests/unit/test_land_params.py
tests/unit/test_run_amip_cli.py
tests/unit/test_physics_radiation.py
tests/unit/test_voronoi_precision.py
tests/unit/test_duogrid.py
tests/unit/test_operators_fc_3d.py
tests/unit/test_parallel_runtime.py
tests/unit/test_physics_microphysics.py
tests/unit/test_config_roundtrip.py
tests/unit/test_deprecation_warnings.py
tests/unit/test_scale_latlon_spectral.py
tests/unit/test_physics_combined.py
tests/unit/test_vector_calculus_identities.py
tests/unit/test_moisture_budget.py
tests/unit/test_neuralgcm_s2s.py
tests/unit/test_precision.py
tests/unit/test_async_halo.py
tests/unit/test_operators_latlon.py
tests/unit/test_land_ice_multilayer.py
tests/unit/test_emanuel.py
tests/unit/test_smooth.py
tests/unit/test_cross_discretization.py
tests/unit/test_sea_ice_dynamics.py
tests/unit/test_convergence_rates.py
tests/unit/test_land_ice_carbon.py
tests/unit/test_cfl.py
tests/unit/test_diff_coupled_system.py
tests/unit/test_scale_metal.py
tests/unit/test_diff_coupler.py
tests/unit/test_cdgrid.py
tests/unit/test_grid_dycore_fixes.py
tests/unit/test_precision_modes.py
tests/unit/test_latlon_grid.py
tests/unit/test_grid.py
tests/unit/test_cdgrid_fv3_regression.py
tests/unit/test_voronoi_trisk_weights.py
tests/unit/test_scale_mpi_layout.py
tests/unit/test_diagnostic_collector.py
tests/unit/test_learned_column.py
tests/unit/test_zhang_mcfarlane.py
tests/unit/test_scale_portability.py
tests/unit/test_driver_forcing_dispatch.py
tests/land/unit/test_multilayer_land.py
tests/land/unit/test_land_water_budget.py
tests/land/unit/test_stomata.py
tests/land/unit/test_land_audit_fixes.py
tests/land/unit/__init__.py
tests/land/unit/test_carbon_cycle.py
tests/atmosphere/shallow_water/validation/__init__.py
tests/atmosphere/shallow_water/__init__.py
tests/test_cases/baroclinic_wave.py
tests/test_cases/cosine_bell.py
tests/test_cases/dcmip_transport.py
tests/test_cases/williamson.py
tests/__init__.py
tests/ocean/distributed/test_ocean_mpi_conservation.py
tests/ocean/distributed/__init__.py
tests/test_cases/williamson_latlon.py
tests/test_cases/__init__.py
tests/sea_ice/__init__.py
tests/distributed/test_mpi_driver.py
tests/distributed/test_scale_mpi_halo.py
tests/distributed/test_voronoi_halo.py
tests/distributed/test_mpi_differentiability.py
tests/distributed/__init__.py
tests/distributed/test_mpi_bootstrap.py
tests/distributed/test_coupler_mpi.py
tests/distributed/test_halo_mpi.py
tests/distributed/conftest.py
tests/distributed/test_voronoi_mpi.py
tests/atmosphere/shallow_water/integration/test_shallow_water.py
tests/atmosphere/shallow_water/integration/test_fv_cubesphere.py
tests/atmosphere/shallow_water/integration/__init__.py
tests/atmosphere/shallow_water/integration/test_shallow_water_mpas.py
tests/atmosphere/shallow_water/integration/test_boundary_fix.py
tests/atmosphere/shallow_water/integration/test_fv_convergence.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/stretched_grid_util.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/kernel_ops.py
tests/ocean/unit/test_surface_forcing_dispatch.py
tests/ocean/unit/test_ocean_fc.py
tests/ocean/unit/test_barotropic_noise_invariant.py
tests/ocean/unit/test_gm_redi_latlon_cgrid.py
tests/ocean/unit/test_ocean_differentiability.py
tests/ocean/unit/test_ocean_diagnostics.py
tests/ocean/unit/test_sfno_ocean.py
tests/ocean/unit/test_ocean_compatibility.py
tests/ocean/unit/test_latlon_cgrid_ocean.py
tests/ocean/unit/test_weno_momentum.py
tests/ocean/unit/test_advection_weno.py
tests/ocean/unit/test_gm_redi_eady_physics.py
tests/ocean/unit/test_eady_overrides.py
tests/ocean/unit/test_barotropic_implicit_mpas.py
tests/ocean/unit/__init__.py
tests/ocean/unit/test_gm_redi_mpas.py
tests/ocean/unit/test_visbeck_gm.py
tests/ocean/unit/test_advection_som.py
tests/ocean/unit/test_cross_grid_parity.py
tests/ocean/unit/test_ocean_biogeochemistry.py
tests/ocean/unit/test_ocean_fv.py
tests/ocean/unit/test_leith.py
tests/ocean/unit/test_bottom_drag_sponge.py
tests/ocean/unit/test_shortwave_penetration.py
tests/ocean/unit/test_mpas_ocean.py
tests/ocean/unit/test_freshwater.py
tests/ocean/unit/test_mpas_physics.py
tests/ocean/unit/test_no_scheme_duplication.py
tests/ocean/unit/test_backscatter.py
tests/ocean/unit/test_eta_floor.py
tests/ocean/unit/test_implicit_vertical_solver.py
tests/ocean/unit/test_implicit_solver.py
tests/ocean/unit/test_advection_dst3.py
tests/ocean/unit/test_inertia_gravity_wave.py
tests/ocean/unit/test_bathymetry.py
tests/ocean/unit/test_biharmonic_vorticity.py
tests/ocean/unit/test_smagorinsky.py
tests/ocean/unit/test_momentum_diagnostics_closure.py
tests/ocean/unit/test_advection_fct_zalesak.py
tests/ocean/unit/test_barotropic_cgrid.py
tests/ocean/unit/test_ocean.py
tests/test_cases/dcmip2025/test_case_3.py
tests/test_cases/dcmip2025/test_case_2.py
tests/test_cases/dcmip2025/common.py
tests/test_cases/dcmip2025/__init__.py
tests/test_cases/dcmip2025/test_case_1.py
tests/sea_ice/validation/test_sea_ice_validation.py
tests/sea_ice/validation/__init__.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/optics_base.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/rrtmgp.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/utils/file_io.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/utils/__init__.py
tests/atmosphere/shallow_water/test_cases/williamson.py
tests/atmosphere/shallow_water/test_cases/__init__.py
tests/atmosphere/shallow_water/test_cases/williamson_latlon.py
tests/atmosphere/shallow_water/test_cases/williamson_mpas.py
tests/atmosphere/shallow_water/unit/__init__.py
tests/atmosphere/shallow_water/unit/test_sfno_sw.py
tests/sea_ice/unit/test_surface_albedo.py
tests/sea_ice/unit/__init__.py
tests/atmosphere/shallow_water/unit/test_spectral.py
tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/rte/two_stream.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/rte/__init__.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/rte/rte_utils.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/rte/monochromatic_two_stream.py
tests/atmosphere/nonhydrostatic/integration/test_fv_cubesphere.py
tests/atmosphere/nonhydrostatic/unit/__init__.py
tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py
tests/atmosphere/nonhydrostatic/unit/test_spectral_nh.py
tests/atmosphere/nonhydrostatic/integration/__init__.py
tests/atmosphere/hydrostatic/__init__.py
tests/atmosphere/nonhydrostatic/__init__.py
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_advection.py
tests/atmosphere/hydrostatic/unit/test_diagnose_w_from_omega.py
tests/atmosphere/hydrostatic/unit/test_topography.py
tests/atmosphere/hydrostatic/unit/test_primitive_eq_latlon_cgrid.py
tests/atmosphere/hydrostatic/unit/test_turbulence.py
tests/atmosphere/hydrostatic/unit/test_spectral_pe_convection.py
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py
tests/atmosphere/hydrostatic/unit/test_primitive_eq.py
tests/atmosphere/hydrostatic/unit/test_combined_physics.py
tests/atmosphere/hydrostatic/unit/test_convection_latlon_mpas.py
tests/atmosphere/hydrostatic/unit/test_spectral_pe_tracer_filter.py
tests/atmosphere/hydrostatic/unit/test_moisture_convergence.py
tests/atmosphere/hydrostatic/unit/test_atmosphere_invariants.py
tests/atmosphere/hydrostatic/unit/test_gravity_wave_drag.py
tests/atmosphere/hydrostatic/unit/__init__.py
tests/atmosphere/hydrostatic/unit/test_amip_forcing.py
tests/atmosphere/hydrostatic/unit/test_spectral_pe.py
tests/atmosphere/hydrostatic/unit/test_energy_budget.py
tests/atmosphere/hydrostatic/unit/test_monthly_means.py
tests/atmosphere/hydrostatic/unit/test_microphysics.py
tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py
tests/atmosphere/hydrostatic/unit/test_amip_config.py
tests/atmosphere/hydrostatic/unit/test_convection.py
tests/atmosphere/hydrostatic/unit/test_radiation.py
tests/atmosphere/hydrostatic/test_cases/dcmip_transport.py
tests/atmosphere/hydrostatic/test_cases/__init__.py
tests/atmosphere/nonhydrostatic/validation/__init__.py
tests/atmosphere/nonhydrostatic/test_cases/__init__.py
tests/atmosphere/hydrostatic/validation/test_stability_fix.py
tests/atmosphere/hydrostatic/validation/test_spectral_pe_moist_held_suarez.py
tests/atmosphere/hydrostatic/validation/__init__.py
tests/atmosphere/hydrostatic/validation/test_held_suarez_fix.py
tests/atmosphere/hydrostatic/integration/__init__.py
tests/atmosphere/hydrostatic/integration/test_amip_rrtmg.py
tests/atmosphere/hydrostatic/integration/test_amip_stability.py
tests/atmosphere/hydrostatic/integration/test_amip_smoke.py
tests/atmosphere/hydrostatic/integration/test_fv_cubesphere.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_3.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_1_mpas.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_2.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_3_mpas.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/common.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_2_mpas.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/dcmip_test_suite_dev.md
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/__init__.py
tests/atmosphere/nonhydrostatic/test_cases/dcmip2025/test_case_1.py
src/legoesm/atmosphere/physics/radiation/rrtmgp/optics/rrtmgp_data/LICENSE

exec
/bin/zsh -lc "rg -n \"def saturation_adjustment|saturation_adjustment\\(|cond_evap_sink|qc_sink_total|qc_scale|condensation = jnp.where|class .*Kessler|def .*kessler|graupel_frac|dN_i_nuc|drag = rho\" -S ." in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
./src/legoesm/atmosphere/physics/ml_parameterization.py:247:def _apply_predicted_kessler_microphysics(
./src/legoesm/atmosphere/physics/ml_parameterization.py:269:def _limit_predicted_kessler_microphysics_tendencies(
./src/legoesm/atmosphere/physics/microphysics/thompson.py:85:    condensation, q_sat = saturation_adjustment(
./src/legoesm/atmosphere/physics/microphysics/thompson.py:119:    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)
./src/legoesm/atmosphere/physics/microphysics/thompson.py:171:    graupel_frac = jax.nn.sigmoid(
./src/legoesm/atmosphere/physics/microphysics/thompson.py:174:    rime_to_graupel_from_i = config.rime_to_graupel_rate * riming_i * graupel_frac
./src/legoesm/atmosphere/physics/microphysics/thompson.py:175:    rime_to_graupel_from_s = config.rime_to_graupel_rate * riming_s * graupel_frac
./src/legoesm/atmosphere/physics/microphysics/thompson.py:186:    cond_evap_sink = jnp.maximum(-condensation, 0.0)
./src/legoesm/atmosphere/physics/microphysics/thompson.py:187:    qc_sink_total = (
./src/legoesm/atmosphere/physics/microphysics/thompson.py:188:        dq_c_au + dq_c_ac + bergeron + riming_i + riming_s + cond_evap_sink
./src/legoesm/atmosphere/physics/microphysics/thompson.py:191:    qc_scale = jnp.minimum(
./src/legoesm/atmosphere/physics/microphysics/thompson.py:193:        qc_avail / jnp.maximum(qc_sink_total * dt_safe, 1e-30),
./src/legoesm/atmosphere/physics/microphysics/thompson.py:195:    dq_c_au = dq_c_au * qc_scale
./src/legoesm/atmosphere/physics/microphysics/thompson.py:196:    dq_c_ac = dq_c_ac * qc_scale
./src/legoesm/atmosphere/physics/microphysics/thompson.py:197:    bergeron = bergeron * qc_scale
./src/legoesm/atmosphere/physics/microphysics/thompson.py:198:    riming_i = riming_i * qc_scale
./src/legoesm/atmosphere/physics/microphysics/thompson.py:199:    riming_s = riming_s * qc_scale
./src/legoesm/atmosphere/physics/microphysics/thompson.py:202:    # which has already been scaled by qc_scale above.  Re-scaling
./src/legoesm/atmosphere/physics/microphysics/thompson.py:204:    # by ``qc_scale`` once preserves both per-donor proportionality and
./src/legoesm/atmosphere/physics/microphysics/thompson.py:206:    rime_to_graupel_from_i = rime_to_graupel_from_i * qc_scale
./src/legoesm/atmosphere/physics/microphysics/thompson.py:207:    rime_to_graupel_from_s = rime_to_graupel_from_s * qc_scale
./src/legoesm/atmosphere/physics/microphysics/thompson.py:209:    dN_r_au = dN_r_au * qc_scale
./src/legoesm/atmosphere/physics/microphysics/thompson.py:211:    # factor; positive condensation is unaffected (cond_evap_sink = 0).
./src/legoesm/atmosphere/physics/microphysics/thompson.py:212:    condensation = jnp.where(
./src/legoesm/atmosphere/physics/microphysics/thompson.py:213:        condensation < 0.0, condensation * qc_scale, condensation,
./src/legoesm/atmosphere/physics/microphysics/thompson.py:272:    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
./src/legoesm/atmosphere/physics/microphysics/kessler.py:32:def kessler_microphysics(
./src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:72:    condensation, q_sat = saturation_adjustment(
./src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:52:def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0, q_c=None):
./src/legoesm/atmosphere/physics/microphysics/config.py:26:class KesslerConfig(NamedTuple):
./src/legoesm/atmosphere/physics/microphysics/morrison.py:78:    condensation, q_sat = saturation_adjustment(
./src/legoesm/atmosphere/physics/microphysics/morrison.py:98:    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)
./src/legoesm/atmosphere/physics/microphysics/morrison.py:154:    cond_evap_sink = jnp.maximum(-condensation, 0.0)
./src/legoesm/atmosphere/physics/microphysics/morrison.py:155:    qc_sink_total = dq_c_au + dq_c_ac + bergeron + riming_i + riming_s + cond_evap_sink
./src/legoesm/atmosphere/physics/microphysics/morrison.py:157:    qc_scale = jnp.minimum(
./src/legoesm/atmosphere/physics/microphysics/morrison.py:159:        qc_avail / jnp.maximum(qc_sink_total * jnp.maximum(dt, 1e-10), 1e-30),
./src/legoesm/atmosphere/physics/microphysics/morrison.py:161:    dq_c_au = dq_c_au * qc_scale
./src/legoesm/atmosphere/physics/microphysics/morrison.py:162:    dq_c_ac = dq_c_ac * qc_scale
./src/legoesm/atmosphere/physics/microphysics/morrison.py:163:    bergeron = bergeron * qc_scale
./src/legoesm/atmosphere/physics/microphysics/morrison.py:164:    riming_i = riming_i * qc_scale
./src/legoesm/atmosphere/physics/microphysics/morrison.py:165:    riming_s = riming_s * qc_scale
./src/legoesm/atmosphere/physics/microphysics/morrison.py:170:    # nothing because ``cond_evap_sink = 0``.
./src/legoesm/atmosphere/physics/microphysics/morrison.py:171:    condensation = jnp.where(
./src/legoesm/atmosphere/physics/microphysics/morrison.py:172:        condensation < 0.0, condensation * qc_scale, condensation,
./src/legoesm/atmosphere/physics/microphysics/morrison.py:175:    dN_r_au = dN_r_au * qc_scale
./src/legoesm/atmosphere/physics/microphysics/morrison.py:224:    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
./scripts/run_atmosphere_test_matrix.py:2869:def _make_kessler_nh_physics_fn(height_coord, dt_phys, tendencies_cls):
./tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:393:class TestKesslerMicrophysics:
./tests/atmosphere/nonhydrostatic/unit/test_compressible_euler.py:419:    def test_kessler_tendencies_finite(self, grid, height_coord, terrain_metric):
./tests/unit/test_physics_microphysics.py:198:def test_kessler_autoconversion_sensitivity():
./tests/unit/test_physics_microphysics.py:659:def test_kessler_grad_finite_at_zero_qr():
./tests/unit/test_physics_microphysics.py:674:    def kessler_objective(q_r):
./tests/unit/test_ml_physics_parameterization.py:61:def test_kessler_feature_vector_shape():
./tests/unit/test_ml_physics_parameterization.py:121:def test_kessler_target_vector_shape_and_unpack():
./tests/unit/test_ml_physics_parameterization.py:195:def test_kessler_checkpoint_and_stats_roundtrip():
./tests/unit/test_physics_grid_adapters.py:240:    def test_resolve_kessler(self):
./tests/unit/test_physics_grid_adapters.py:318:    def test_kessler_microphysics(self, cs_grid):
./tests/atmosphere/hydrostatic/unit/test_microphysics.py:171:class TestKessler:
./tests/atmosphere/hydrostatic/unit/test_microphysics.py:651:class TestKesslerNonhydrostatic:
./tests/atmosphere/hydrostatic/unit/test_microphysics.py:653:    def test_kessler_nonhydrostatic_tendencies(self):
./tests/atmosphere/hydrostatic/unit/test_microphysics.py:783:    def test_kessler_microphysics(self):
./tests/atmosphere/hydrostatic/unit/test_microphysics.py:802:    def test_kessler_in_column_physics(self):
./tests/atmosphere/hydrostatic/unit/test_all_physics_schemes.py:215:    def test_kessler(self):
./tests/unit/test_equation_fixes.py:185:    def test_kessler_condensation_depends_on_dt(self):
./tests/unit/test_equation_fixes.py:198:    def test_kessler_condensation_heating_scales_correctly(self):
./tests/unit/test_ml_physics_workflow.py:29:def test_kessler_workflow_uses_rrtmgp_base_config():
./tests/unit/test_warm_rain.py:41:def test_saturation_adjustment(column_state):
./tests/unit/test_warm_rain.py:43:    cond, q_sat = saturation_adjustment(T, q_v, p, dt=300.0)
./tests/unit/test_warm_rain.py:107:    _, q_sat = saturation_adjustment(T, q_v_low, p, dt=300.0)
./tests/unit/test_warm_rain.py:121:        cond, q_sat = saturation_adjustment(T_, q_v_, p_, dt=300.0)

exec
/bin/zsh -lc 'rg -n "test_subsaturated_clear_air_does_not_create_negative_qc|subsaturated|clear_air|negative_qc|negative q_c" -S tests src .' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
./NEXT_STEPS.md:95:   where `RH` is relative humidity, `q_c = q_cloud + q_ice`, `q_s` is saturation mixing ratio, and `p ≈ 0.25`, `α ≈ 100`. This gives cloud fraction that is 0 in dry subsaturated air and approaches 1 near saturation with condensate present.
./src/legoesm/atmosphere/physics/microphysics/thompson.py:84:    # subsaturated clear air.  See _warm_rain.saturation_adjustment.
./src/legoesm/atmosphere/physics/microphysics/thompson.py:184:    # condensation) in the q_c sink budget so subsaturated clear-air
./src/legoesm/atmosphere/physics/microphysics/kessler.py:79:    # against the available ``q_c`` so a subsaturated clear-air column
./src/legoesm/atmosphere/physics/microphysics/kessler.py:81:    # cycle 2: "subsaturated clear air can create negative cloud water").
./src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:70:        evaporation cannot drive ``q_c`` below zero in subsaturated
./src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:86:    A subsaturated column with ``q_c = 0`` would otherwise produce
./src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:89:    "subsaturated clear air can create negative cloud water".  The
./src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:224:    """Compute rain evaporation in subsaturated air.
./src/legoesm/atmosphere/physics/microphysics/morrison.py:77:    # subsaturated clear air.  See _warm_rain.saturation_adjustment.
./src/legoesm/atmosphere/physics/microphysics/morrison.py:147:    # ``condensation``) in the q_c sink budget — otherwise a subsaturated
./src/legoesm/atmosphere/physics/microphysics/morrison.py:150:    # (Codex audit cycle 2: "subsaturated clear air can create negative
./src/legoesm/atmosphere/physics/thermodynamics.py:226:    contrast in subsaturated boundary layers and therefore systematically
./src/legoesm/atmosphere/physics/convection/sbm.py:144:    # field is still non-negative (no negative q_c production); when
./src/legoesm/atmosphere/physics/convection/kuo.py:94:    # 2. Column moisture excess: positive part only (zero when subsaturated)
src/legoesm/atmosphere/physics/microphysics/thompson.py:84:    # subsaturated clear air.  See _warm_rain.saturation_adjustment.
src/legoesm/atmosphere/physics/microphysics/thompson.py:184:    # condensation) in the q_c sink budget so subsaturated clear-air
src/legoesm/atmosphere/physics/microphysics/kessler.py:79:    # against the available ``q_c`` so a subsaturated clear-air column
src/legoesm/atmosphere/physics/microphysics/kessler.py:81:    # cycle 2: "subsaturated clear air can create negative cloud water").
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:70:        evaporation cannot drive ``q_c`` below zero in subsaturated
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:86:    A subsaturated column with ``q_c = 0`` would otherwise produce
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:89:    "subsaturated clear air can create negative cloud water".  The
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:224:    """Compute rain evaporation in subsaturated air.
src/legoesm/atmosphere/physics/microphysics/morrison.py:77:    # subsaturated clear air.  See _warm_rain.saturation_adjustment.
src/legoesm/atmosphere/physics/microphysics/morrison.py:147:    # ``condensation``) in the q_c sink budget — otherwise a subsaturated
src/legoesm/atmosphere/physics/microphysics/morrison.py:150:    # (Codex audit cycle 2: "subsaturated clear air can create negative
src/legoesm/atmosphere/physics/thermodynamics.py:226:    contrast in subsaturated boundary layers and therefore systematically
src/legoesm/atmosphere/physics/convection/sbm.py:144:    # field is still non-negative (no negative q_c production); when
src/legoesm/atmosphere/physics/convection/kuo.py:94:    # 2. Column moisture excess: positive part only (zero when subsaturated)
tests/atmosphere/hydrostatic/unit/test_microphysics.py:115:    """Create warm, subsaturated columns with rain to isolate evaporation."""
./tests/unit/test_physics_microphysics.py:201:    # Below threshold: subsaturated, small q_c
./tests/unit/test_physics_microphysics.py:207:    # Above threshold: subsaturated, large q_c
./tests/unit/test_physics_microphysics.py:350:def test_subsaturated_clear_air_does_not_create_negative_qc(scheme, call):
./tests/unit/test_physics_microphysics.py:354:    to ``dq_c_dt`` in subsaturated clear air (``q_c = 0``) produces a
./tests/unit/test_physics_microphysics.py:386:    q_v = 0.95 * q_sat                           # subsaturated
./tests/unit/test_physics_microphysics.py:404:        "subsaturated clear-air column (q_v = 95% RH, q_c = 0).  "
./tests/unit/test_physics_microphysics.py:409:        "branch.  Audit cycle 2 Codex finding 'subsaturated clear air "
./tests/unit/test_physics_microphysics.py:700:    # Subsaturated column was set up with q_v = 0.8 * q_sat so subsaturation > 0.
./tests/unit/test_physics_microphysics.py:701:    assert float(jnp.max(g_evap_pos)) > 0.0, "rain_evaporation grad should be positive for subsaturated column"
tests/unit/test_physics_microphysics.py:201:    # Below threshold: subsaturated, small q_c
tests/unit/test_physics_microphysics.py:207:    # Above threshold: subsaturated, large q_c
tests/unit/test_physics_microphysics.py:350:def test_subsaturated_clear_air_does_not_create_negative_qc(scheme, call):
tests/unit/test_physics_microphysics.py:354:    to ``dq_c_dt`` in subsaturated clear air (``q_c = 0``) produces a
tests/unit/test_physics_microphysics.py:386:    q_v = 0.95 * q_sat                           # subsaturated
tests/unit/test_physics_microphysics.py:404:        "subsaturated clear-air column (q_v = 95% RH, q_c = 0).  "
tests/unit/test_physics_microphysics.py:409:        "branch.  Audit cycle 2 Codex finding 'subsaturated clear air "
tests/unit/test_physics_microphysics.py:700:    # Subsaturated column was set up with q_v = 0.8 * q_sat so subsaturation > 0.
tests/unit/test_physics_microphysics.py:701:    assert float(jnp.max(g_evap_pos)) > 0.0, "rain_evaporation grad should be positive for subsaturated column"
./tests/unit/test_warm_rain.py:105:    # Subsaturated vapor profile to drive evaporation.
./tests/atmosphere/hydrostatic/unit/test_microphysics.py:115:    """Create warm, subsaturated columns with rain to isolate evaporation."""
./tests/unit/test_ml_physics_parameterization.py:418:    q_v = jnp.array([[5.0e-3, 3.0e-3]])  # subsaturated to suppress autoconv
tests/unit/test_warm_rain.py:105:    # Subsaturated vapor profile to drive evaporation.
tests/unit/test_ml_physics_parameterization.py:418:    q_v = jnp.array([[5.0e-3, 3.0e-3]])  # subsaturated to suppress autoconv
tests/unit/test_physics_convection.py:773:    ``q_v = 0.7 * q_sat`` (subsaturated) which gave ``MC = 0``,
tests/unit/test_physics_convection.py:809:    # subsaturated elsewhere to make the deficit profile non-trivial.
./tests/unit/test_physics_convection.py:773:    ``q_v = 0.7 * q_sat`` (subsaturated) which gave ``MC = 0``,
./tests/unit/test_physics_convection.py:809:    # subsaturated elsewhere to make the deficit profile non-trivial.

codex
I found the changed symbols in the expected microphysics files. Next I’m reading the actual tendency construction so I can verify the algebra, not just the snippets.
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/kessler.py | sed -n '1,180p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/_warm_rain.py | sed -n '1,130p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py | sed -n '1,180p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba tests/unit/test_physics_microphysics.py | sed -n '320,430p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Kessler warm-rain microphysics column backend.
     2	
     3	A simple one-moment warm-rain scheme tracking cloud water and rain.
     4	Processes: saturation adjustment, autoconversion, accretion, evaporation,
     5	rain sedimentation, and latent heating.
     6	
     7	All operations use smooth (differentiable) approximations for
     8	compatibility with jax.grad.
     9	
    10	References
    11	----------
    12	- Kessler (1969): On the Distribution and Continuity of Water
    13	  Substance in Atmospheric Circulations.
    14	"""
    15	
    16	from __future__ import annotations
    17	
    18	import jax
    19	import jax.numpy as jnp
    20	
    21	from legoesm import constants
    22	from legoesm.thermo import saturation_mixing_ratio
    23	from legoesm.atmosphere.physics.microphysics._warm_rain import safe_pow
    24	from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
    25	from legoesm.atmosphere.physics.microphysics.output import (
    26	    HydrometeorState,
    27	    MicrophysicsOutput,
    28	    sedimentation_tendency,
    29	)
    30	
    31	
    32	def kessler_microphysics(
    33	    T: jax.Array,
    34	    q_v: jax.Array,
    35	    hydrometeors: HydrometeorState,
    36	    p_full: jax.Array,
    37	    p_half: jax.Array,
    38	    rho: jax.Array,
    39	    dz: jax.Array,
    40	    dt: float,
    41	    config: KesslerConfig = KesslerConfig(),
    42	) -> MicrophysicsOutput:
    43	    """Compute Kessler microphysics tendencies.
    44	
    45	    Parameters
    46	    ----------
    47	    T : jax.Array
    48	        Temperature [K], shape (ncol, nlev).
    49	    q_v : jax.Array
    50	        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    51	    hydrometeors : HydrometeorState
    52	        Hydrometeor state (only q_c, q_r used).
    53	    p_full : jax.Array
    54	        Pressure at full levels [Pa], shape (ncol, nlev).
    55	    p_half : jax.Array
    56	        Pressure at half levels [Pa], shape (ncol, nlev+1).
    57	    rho : jax.Array
    58	        Air density [kg/m^3], shape (ncol, nlev).
    59	    dz : jax.Array
    60	        Layer thickness [m], shape (ncol, nlev).
    61	    dt : float
    62	        Time step [s].
    63	    config : KesslerConfig
    64	
    65	    Returns
    66	    -------
    67	    MicrophysicsOutput
    68	    """
    69	    ncol, nlev = T.shape
    70	    q_c = hydrometeors.q_c
    71	    q_r = hydrometeors.q_r
    72	    sharpness = config.saturation_sharpness
    73	
    74	    # Saturation mixing ratio
    75	    q_sat = saturation_mixing_ratio(T, p_full)
    76	
    77	    # 1. Saturation adjustment — convert from increment [kg/kg] to tendency [kg/kg/s].
    78	    # The evaporation branch (negative ``condensation``) is donor-clamped
    79	    # against the available ``q_c`` so a subsaturated clear-air column
    80	    # (q_v < q_sat, q_c = 0) cannot drive ``q_c`` below zero (Codex audit
    81	    # cycle 2: "subsaturated clear air can create negative cloud water").
    82	    # Same pattern as ``_warm_rain.saturation_adjustment``.
    83	    excess = q_v - q_sat
    84	    cond_frac = jax.nn.sigmoid(sharpness * excess)
    85	    condensation = cond_frac * excess / dt  # [kg/kg/s]
    86	    q_c_avail = jnp.clip(q_c, 0.0, None)
    87	    condensation = jnp.maximum(
    88	        condensation, -q_c_avail / jnp.maximum(dt, 1e-10),
    89	    )
    90	
    91	    dq_v_sat = -condensation
    92	    dq_c_sat = condensation
    93	
    94	    # 2. Autoconversion: cloud -> rain (threshold excess)
    95	    # dq_c_sat is a tendency [kg/kg/s]; multiply by dt to get increment [kg/kg]
    96	    q_c_updated = q_c + dq_c_sat * dt
    97	    autoconv = config.autoconversion_rate * jnp.maximum(
    98	        q_c_updated - config.autoconversion_threshold, 0.0
    99	    )
   100	
   101	    # 3. Accretion: cloud collected by rain.  Fractional powers of q_r
   102	    # have unbounded derivative at q_r=0 — safe_pow handles the AD guard.
   103	    accretion = config.accretion_coeff * q_c * safe_pow(q_r, 0.875)
   104	
   105	    # 4. Evaporation of rain (q_r^0.525).
   106	    subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
   107	    evaporation = config.evaporation_coeff * subsaturation * safe_pow(q_r, 0.525)
   108	
   109	    # 5. Rain sedimentation
   110	    rho_sfc = rho[:, -1:]
   111	    V_t = config.rain_fall_speed * jnp.sqrt(
   112	        rho_sfc / jnp.clip(rho, 0.1)
   113	    )
   114	    sed_tend = sedimentation_tendency(q_r, rho, V_t, dz)
   115	
   116	    # 6. Latent heating
   117	    dT_dt = constants.L_v * (condensation - evaporation) / constants.c_pd
   118	
   119	    # Combine tracer tendencies
   120	    dq_v_dt = dq_v_sat + evaporation
   121	    dq_c_dt = dq_c_sat - autoconv - accretion
   122	    dq_r_dt = autoconv + accretion - evaporation + sed_tend
   123	
   124	    # Precipitation: surface flux
   125	    q_r_bot = jnp.clip(q_r[:, -1], 0.0)
   126	    V_t_bot = V_t[:, -1]
   127	    precipitation = q_r_bot * rho[:, -1] * V_t_bot
   128	
   129	    # Pin dtype to the input precision so we never silently promote
   130	    # the unused-tendency placeholders to f64 under x64 mode.
   131	    _dtype = T.dtype
   132	    z = jnp.zeros((ncol, nlev), dtype=_dtype)
   133	    z1 = jnp.zeros((ncol,), dtype=_dtype)
   134	    return MicrophysicsOutput(
   135	        dT_dt=dT_dt,
   136	        dq_v_dt=dq_v_dt,
   137	        dq_c_dt=dq_c_dt,
   138	        dq_r_dt=dq_r_dt,
   139	        dq_i_dt=z,
   140	        dq_s_dt=z,
   141	        dq_g_dt=z,
   142	        dN_c_dt=z,
   143	        dN_r_dt=z,
   144	        dN_i_dt=z,
   145	        precipitation=precipitation,
   146	    )

 succeeded in 0ms:
     1	"""Shared warm-rain microphysics helpers.
     2	
     3	Functions here are used by multiple microphysics backends (Seifert-Beheng,
     4	Morrison, Thompson, Kessler) to avoid duplicating identical physics code.
     5	"""
     6	
     7	from __future__ import annotations
     8	
     9	import jax
    10	import jax.numpy as jnp
    11	
    12	from legoesm import constants
    13	from legoesm.thermo import saturation_mixing_ratio
    14	
    15	
    16	def safe_pow(x, p):
    17	    """Differentiable ``x ** p`` with grad=0 wherever ``x <= 0``.
    18	
    19	    Microphysics has many Marshall-Palmer-style fractional powers of
    20	    hydrometeor mixing ratios (``q_r``, ``q_i``, ``q_s``, ``q_g``,
    21	    ``N_i``, …) with exponents in (0, 1) — typically 0.5 for fall
    22	    speeds, 1/3 for diameters, 0.525/0.875 for ventilation/accretion.
    23	    Their analytic derivative ``p * x**(p-1)`` is unbounded at ``x=0``
    24	    and complex for ``x<0``.  ``jnp.clip(x, 0.0) ** p`` therefore
    25	    returns ``inf`` (at zero) or ``nan`` (at negatives) under
    26	    ``jax.grad``, breaking AD on cold-start (no-precip) initial
    27	    conditions.
    28	
    29	    The double-where pattern below routes the AD graph through a
    30	    placeholder of 1.0 in the inactive branch so the gradient never
    31	    sees ``0**(p-1)``.
    32	
    33	    Parameters
    34	    ----------
    35	    x : array
    36	        Argument of the power.  May be zero or negative.
    37	    p : float or array
    38	        Exponent.  Intended for ``0 < p < 1`` where the bug applies;
    39	        also safe for ``p >= 1``.
    40	
    41	    Returns
    42	    -------
    43	    array
    44	        ``x ** p`` for ``x > 0``, else 0; gradient is finite
    45	        everywhere.
    46	    """
    47	    positive = x > 0.0
    48	    safe_x = jnp.where(positive, x, 1.0)
    49	    return jnp.where(positive, safe_x ** p, 0.0)
    50	
    51	
    52	def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0, q_c=None):
    53	    """Compute smooth saturation adjustment (condensation tendency).
    54	
    55	    Parameters
    56	    ----------
    57	    T : array (ncol, nlev)
    58	        Temperature [K].
    59	    q_v : array (ncol, nlev)
    60	        Water vapor mixing ratio [kg/kg].
    61	    p_full : array (ncol, nlev)
    62	        Pressure [Pa].
    63	    dt : float
    64	        Time step [s].
    65	    sharpness : float
    66	        Sigmoid sharpness for smooth condensation switch.
    67	    q_c : array or None
    68	        Cloud water mixing ratio [kg/kg].  When provided, the negative
    69	        (evaporation) branch is donor-clamped against ``q_c`` so that
    70	        evaporation cannot drive ``q_c`` below zero in subsaturated
    71	        clear air.  Without ``q_c`` the legacy signed return is
    72	        produced (callers must apply their own donor clamp).
    73	
    74	    Returns
    75	    -------
    76	    condensation : array (ncol, nlev)
    77	        Condensation tendency [kg/kg/s].  Positive = condensation;
    78	        negative = evaporation (donor-clamped against ``q_c`` when
    79	        provided).  Without ``q_c``, the legacy unclamped signed
    80	        value is returned for backward compatibility.
    81	    q_sat : array (ncol, nlev)
    82	        Saturation mixing ratio [kg/kg].
    83	
    84	    Notes
    85	    -----
    86	    A subsaturated column with ``q_c = 0`` would otherwise produce
    87	    spurious negative ``q_c`` after the explicit Euler step ``q_c_new
    88	    = q_c + condensation * dt`` — Codex audit cycle 2 finding
    89	    "subsaturated clear air can create negative cloud water".  The
    90	    ``q_c``-aware donor clamp on the evaporation branch is the
    91	    minimal fix that conserves total water in both clear and cloudy
    92	    columns.
    93	    """
    94	    q_sat = saturation_mixing_ratio(T, p_full)
    95	    excess = q_v - q_sat
    96	    cond_frac = jax.nn.sigmoid(sharpness * excess)
    97	    condensation = cond_frac * excess / dt
    98	    if q_c is not None:
    99	        # Evaporation rate (negative ``condensation``) is bounded by
   100	        # the available cloud water: |condensation| × dt ≤ q_c, i.e.
   101	        # condensation ≥ -q_c / dt.  ``maximum(condensation, -q_c/dt)``
   102	        # achieves this cleanly.  Differentiable everywhere — the
   103	        # clamp is a smooth-ish max on the evaporation magnitude.
   104	        q_c_avail = jnp.clip(q_c, 0.0, None)
   105	        condensation = jnp.maximum(condensation, -q_c_avail / jnp.maximum(dt, 1e-10))
   106	    return condensation, q_sat
   107	
   108	
   109	def effective_Nc(N_c, Nc_0):
   110	    """Use config default cloud droplet number where N_c is zero.
   111	
   112	    Parameters
   113	    ----------
   114	    N_c : array
   115	        Cloud droplet number concentration [1/kg].
   116	    Nc_0 : float
   117	        Default cloud droplet number.
   118	
   119	    Returns
   120	    -------
   121	    array : Effective N_c.
   122	    """
   123	    return jnp.where(N_c > 1.0, N_c, Nc_0 * jnp.ones_like(N_c))
   124	
   125	
   126	def autoconversion_sb(q_c, N_c_eff, rho, k_au, x_star, sharpness=50.0, gamma_norm=1.0):
   127	    """Seifert-Beheng mass-dependent autoconversion.
   128	
   129	    Parameters
   130	    ----------

 succeeded in 0ms:
     1	"""Seifert-Beheng two-moment warm-rain microphysics.
     2	
     3	A two-moment scheme tracking mass and number concentration of cloud
     4	droplets and rain drops. Processes: saturation adjustment, autoconversion
     5	(mass-dependent), accretion, self-collection, breakup, rain evaporation,
     6	and sedimentation.
     7	
     8	All operations use smooth (differentiable) approximations.
     9	
    10	References
    11	----------
    12	- Seifert, A., & Beheng, K. D. (2001). A two-moment cloud microphysics
    13	  parameterization for mixed-phase clouds. Meteorol. Atmos. Phys., 77, 127-151.
    14	"""
    15	
    16	from __future__ import annotations
    17	
    18	import jax
    19	import jax.numpy as jnp
    20	
    21	from legoesm import constants
    22	from legoesm.atmosphere.physics.microphysics.config import SeifertBehengConfig
    23	from legoesm.atmosphere.physics.microphysics.output import (
    24	    HydrometeorState,
    25	    MicrophysicsOutput,
    26	    sedimentation_tendency,
    27	)
    28	from legoesm.atmosphere.physics.microphysics._warm_rain import (
    29	    saturation_adjustment,
    30	    effective_Nc,
    31	    autoconversion_sb,
    32	    accretion,
    33	    self_collection_breakup,
    34	    rain_evaporation,
    35	    safe_pow,
    36	)
    37	
    38	
    39	def seifert_beheng_microphysics(
    40	    T: jax.Array,
    41	    q_v: jax.Array,
    42	    hydrometeors: HydrometeorState,
    43	    p_full: jax.Array,
    44	    p_half: jax.Array,
    45	    rho: jax.Array,
    46	    dz: jax.Array,
    47	    dt: float,
    48	    config: SeifertBehengConfig = SeifertBehengConfig(),
    49	) -> MicrophysicsOutput:
    50	    """Compute Seifert-Beheng two-moment warm-rain tendencies.
    51	
    52	    Parameters
    53	    ----------
    54	    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
    55	        Same interface as all microphysics backends.
    56	
    57	    Returns
    58	    -------
    59	    MicrophysicsOutput
    60	    """
    61	    ncol, nlev = T.shape
    62	    q_c = hydrometeors.q_c
    63	    q_r = hydrometeors.q_r
    64	    N_c = hydrometeors.N_c
    65	    N_r = hydrometeors.N_r
    66	    sharpness = config.saturation_sharpness
    67	
    68	    N_c_eff = effective_Nc(N_c, config.Nc_0)
    69	
    70	    # Pass ``q_c`` so the evaporation branch (negative ``condensation``) is
    71	    # donor-clamped — see _warm_rain.saturation_adjustment.
    72	    condensation, q_sat = saturation_adjustment(
    73	        T, q_v, p_full, dt, sharpness, q_c=q_c,
    74	    )
    75	
    76	    # 1. Autoconversion (mass-dependent)
    77	    dq_c_au, dN_r_au, x_c = autoconversion_sb(
    78	        q_c, N_c_eff, rho, config.k_au, config.x_star, sharpness,
    79	    )
    80	
    81	    # 2. Accretion
    82	    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac)
    83	
    84	    # 3-4. Self-collection and breakup
    85	    dN_r_sc, dN_r_br = self_collection_breakup(
    86	        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
    87	    )
    88	
    89	    # 5. Rain evaporation
    90	    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff)
    91	
    92	    # 6. Sedimentation — Marshall-Palmer fall speed (q_r * rho/rho_sfc)^b_v_r
    93	    # has fractional exponent (b_v_r=0.5); guard the AD path with safe_pow.
    94	    rho_sfc = rho[:, -1:]
    95	    V_t_r = config.a_v_r * safe_pow(
    96	        jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, 0.1), config.b_v_r,
    97	    )
    98	    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
    99	    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
   100	
   101	    # 7. Latent heating
   102	    dT_dt = constants.L_v * (condensation - evaporation) / constants.c_pd
   103	
   104	    # Combine tendencies
   105	    dq_v_dt = -condensation + evaporation
   106	    dq_c_dt = condensation - dq_c_au - dq_c_ac
   107	    dq_r_dt = dq_c_au + dq_c_ac - evaporation + sed_r
   108	    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
   109	    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
   110	
   111	    # Precipitation
   112	    q_r_bot = jnp.clip(q_r[:, -1], 0.0)
   113	    precipitation = q_r_bot * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
   114	
   115	    # Pin dtype to the input precision so we never silently promote
   116	    # the unused-species placeholders to f64 under x64 mode.
   117	    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
   118	    return MicrophysicsOutput(
   119	        dT_dt=dT_dt,
   120	        dq_v_dt=dq_v_dt,
   121	        dq_c_dt=dq_c_dt,
   122	        dq_r_dt=dq_r_dt,
   123	        dq_i_dt=z,
   124	        dq_s_dt=z,
   125	        dq_g_dt=z,
   126	        dN_c_dt=dN_c_dt,
   127	        dN_r_dt=dN_r_dt,
   128	        dN_i_dt=z,
   129	        precipitation=precipitation,
   130	    )

 succeeded in 0ms:
   320	        N_r=jnp.zeros_like(q_c),
   321	        N_i=1e4 * jnp.ones_like(q_c),
   322	    )
   323	    q_v = 0.8 * saturation_mixing_ratio(T, p_full)
   324	    dt = 1200.0
   325	    out = thompson_microphysics(
   326	        T, q_v, hydro, p_full, p_half, rho, dz, dt, ThompsonConfig(),
   327	    )
   328	    q_i_after = q_i + out.dq_i_dt * dt
   329	    # Skip the top level: with q_i uniform throughout the column the top
   330	    # level always loses sedimentation flux without compensating inflow,
   331	    # which is a sedimentation-CFL concern rather than a melt-clamp one.
   332	    # The melt-clamp bug shows up uniformly in interior levels.
   333	    min_qi_interior = float(jnp.min(q_i_after[:, 1:]))
   334	    assert min_qi_interior >= -1e-9, (
   335	        f"Thompson: interior q_i went negative ({min_qi_interior:.3e}) at "
   336	        f"T=280 K with dt=1200s — melt rate × q_i × melt_frac × dt exceeded "
   337	        "q_i without a donor clamp."
   338	    )
   339	
   340	
   341	@pytest.mark.parametrize(
   342	    "scheme,call",
   343	    [
   344	        ("kessler", kessler_microphysics),
   345	        ("seifert_beheng", seifert_beheng_microphysics),
   346	        ("morrison", morrison_microphysics),
   347	        ("thompson", thompson_microphysics),
   348	    ],
   349	)
   350	def test_subsaturated_clear_air_does_not_create_negative_qc(scheme, call):
   351	    """Audit cycle 2 (Codex): the smooth saturation adjustment
   352	    ``condensation = sigmoid(s · excess) · excess / dt`` is *signed* —
   353	    negative for ``q_v < q_sat`` (evaporation).  Adding this directly
   354	    to ``dq_c_dt`` in subsaturated clear air (``q_c = 0``) produces a
   355	    spurious negative cloud-water tendency that drives ``q_c`` below
   356	    zero in the explicit Euler step.
   357	
   358	    The fix is to donor-clamp the evaporation branch against the
   359	    available cloud water — implemented inside
   360	    ``saturation_adjustment`` (and inline for Kessler) and joined to
   361	    the per-scheme donor clamp on q_c sinks.
   362	
   363	    Test column: 95 % RH at T=280 K, q_c = 0.  In all four schemes
   364	    the buggy form produced ``q_c_after ≈ -3e-4 kg/kg`` after a
   365	    1200-s timestep.
   366	    """
   367	    if scheme in ("kessler", "seifert_beheng"):
   368	        config_cls = {
   369	            "kessler": KesslerConfig,
   370	            "seifert_beheng": SeifertBehengConfig,
   371	        }[scheme]
   372	    elif scheme == "morrison":
   373	        config_cls = MorrisonConfig
   374	    else:
   375	        config_cls = ThompsonConfig
   376	
   377	    ncol, nlev = 1, 5
   378	    T = jnp.full((ncol, nlev), 280.0)
   379	    p_full = jnp.full((ncol, nlev), 5e4)
   380	    p_half = jnp.broadcast_to(
   381	        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
   382	    )
   383	    rho = p_full / (constants.R_d * T)
   384	    dz = jnp.full((ncol, nlev), 1000.0)
   385	    q_sat = saturation_mixing_ratio(T, p_full)
   386	    q_v = 0.95 * q_sat                           # subsaturated
   387	    q_c = jnp.zeros((ncol, nlev))                # NO cloud water
   388	    hydro = HydrometeorState(
   389	        q_c=q_c, q_r=jnp.zeros_like(q_c),
   390	        q_i=jnp.zeros_like(q_c), q_s=jnp.zeros_like(q_c),
   391	        q_g=jnp.zeros_like(q_c),
   392	        N_c=1e8 * jnp.ones_like(q_c),
   393	        N_r=jnp.zeros_like(q_c),
   394	        N_i=jnp.zeros_like(q_c),
   395	    )
   396	    dt = 1200.0
   397	    out = call(T, q_v, hydro, p_full, p_half, rho, dz, dt, config_cls())
   398	    q_c_after = q_c + out.dq_c_dt * dt
   399	    min_q_c = float(jnp.min(q_c_after))
   400	    # Tolerance allows for f64 → f32 promotion noise; the buggy form
   401	    # produces q_c_after ≈ -3e-4, well above this threshold.
   402	    assert min_q_c >= -1e-9, (
   403	        f"{scheme}: q_c_after went negative ({min_q_c:.3e}) in a "
   404	        "subsaturated clear-air column (q_v = 95% RH, q_c = 0).  "
   405	        "The signed saturation adjustment ``condensation = sigmoid(s · "
   406	        "(q_v - q_sat)) · (q_v - q_sat) / dt`` is negative there, and "
   407	        "adding it to ``dq_c_dt`` over an explicit Euler step drives "
   408	        "q_c below zero without a donor clamp on the evaporation "
   409	        "branch.  Audit cycle 2 Codex finding 'subsaturated clear air "
   410	        "can create negative cloud water' has regressed."
   411	    )
   412	
   413	
   414	def test_thompson_rime_to_graupel_donor_split():
   415	    """Audit cycle 2 (Codex): the Thompson rime-to-graupel conversion
   416	    must subtract from the SOURCE species (q_i for ``riming_i``,
   417	    q_s for ``riming_s``), not split via fixed 1.0 / 0.5 / 1.5
   418	    coefficients on the total.
   419	
   420	    Pathological column: ``q_c > 0`` (cloud water source for riming),
   421	    ``q_i = 0`` (no ice to be rimed), ``q_s > 0`` (snow that gets
   422	    rimed by cloud water), ``T < T_freeze`` (active ice phase).  In
   423	    this column ``riming_i = 0`` (no q_i to rime) and ``riming_s > 0``
   424	    (q_c × q_s × f_ice).  Pre-fix:
   425	        rime_to_graupel = rate * (riming_i + riming_s) > 0
   426	        dq_i_dt -= rime_to_graupel              # full subtraction!
   427	        dq_s_dt -= 0.5 * rime_to_graupel
   428	        dq_g_dt += 1.5 * rime_to_graupel
   429	    drives ``q_i`` negative and creates 1.5× extra mass — both
   430	    conservation violations.  Post-fix the donor split scales by

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/thompson.py | sed -n '1,320p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/config.py | sed -n '1,220p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/morrison.py | sed -n '1,270p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Thompson hybrid-moment microphysics.
     2	
     3	Extends Morrison with graupel formation from intense riming and
     4	gamma distribution shape corrections for autoconversion/accretion.
     5	
     6	All operations use smooth (differentiable) approximations.
     7	
     8	References
     9	----------
    10	- Thompson, G., Field, P. R., Rasmussen, R. M., & Hall, W. D. (2008).
    11	  Explicit forecasts of winter precipitation using an improved bulk
    12	  microphysics scheme. Part II: Implementation of a new snow
    13	  parameterization. Mon. Wea. Rev., 136, 5095-5115.
    14	"""
    15	
    16	from __future__ import annotations
    17	
    18	import jax
    19	import jax.numpy as jnp
    20	
    21	from legoesm import constants
    22	from legoesm.thermo import saturation_mixing_ratio_ice as _saturation_mixing_ratio_ice
    23	from legoesm.atmosphere.physics.microphysics._warm_rain import (
    24	    saturation_adjustment,
    25	    effective_Nc,
    26	    autoconversion_sb,
    27	    accretion,
    28	    self_collection_breakup,
    29	    rain_evaporation,
    30	    safe_pow,
    31	)
    32	from legoesm.atmosphere.physics.microphysics.config import ThompsonConfig
    33	from legoesm.atmosphere.physics.microphysics.output import (
    34	    HydrometeorState,
    35	    MicrophysicsOutput,
    36	    sedimentation_tendency,
    37	)
    38	
    39	
    40	def _gamma_ratio(mu):
    41	    """Gamma(mu+4)/Gamma(mu+1) = (mu+3)(mu+2)(mu+1) for integer-like mu."""
    42	    return (mu + 3.0) * (mu + 2.0) * (mu + 1.0)
    43	
    44	
    45	def thompson_microphysics(
    46	    T: jax.Array,
    47	    q_v: jax.Array,
    48	    hydrometeors: HydrometeorState,
    49	    p_full: jax.Array,
    50	    p_half: jax.Array,
    51	    rho: jax.Array,
    52	    dz: jax.Array,
    53	    dt: float,
    54	    config: ThompsonConfig = ThompsonConfig(),
    55	) -> MicrophysicsOutput:
    56	    """Compute Thompson hybrid-moment microphysics tendencies.
    57	
    58	    Parameters
    59	    ----------
    60	    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
    61	        Same interface as all microphysics backends.
    62	
    63	    Returns
    64	    -------
    65	    MicrophysicsOutput
    66	    """
    67	    ncol, nlev = T.shape
    68	    q_c = hydrometeors.q_c
    69	    q_r = hydrometeors.q_r
    70	    q_i = hydrometeors.q_i
    71	    q_s = hydrometeors.q_s
    72	    q_g = hydrometeors.q_g
    73	    N_c = hydrometeors.N_c
    74	    N_r = hydrometeors.N_r
    75	    N_i = hydrometeors.N_i
    76	    sharpness = config.saturation_sharpness
    77	
    78	    N_c_eff = effective_Nc(N_c, config.Nc_0)
    79	
    80	    # === WARM RAIN ===
    81	    # Saturation adjustment — convert increment [kg/kg] to tendency [kg/kg/s].
    82	    # Pass ``q_c`` so the evaporation branch (negative ``condensation``) is
    83	    # donor-clamped: evaporation cannot drive ``q_c`` below zero in
    84	    # subsaturated clear air.  See _warm_rain.saturation_adjustment.
    85	    condensation, q_sat = saturation_adjustment(
    86	        T, q_v, p_full, dt, sharpness, q_c=q_c,
    87	    )
    88	
    89	    # Gamma distribution corrections
    90	    gamma_c = _gamma_ratio(config.mu_c)
    91	    gamma_r = _gamma_ratio(config.mu_r)
    92	    gamma_c_norm = gamma_c / _gamma_ratio(0.0)  # normalize to mu=0 baseline (=24)
    93	    gamma_r_norm = gamma_r / _gamma_ratio(0.0)
    94	
    95	    # Autoconversion (gamma-corrected)
    96	    dq_c_au, dN_r_au, x_c = autoconversion_sb(
    97	        q_c, N_c_eff, rho, config.k_au, config.x_star, sharpness, gamma_norm=gamma_c_norm,
    98	    )
    99	
   100	    # Accretion (gamma-corrected)
   101	    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac, gamma_norm=gamma_r_norm)
   102	
   103	    # Self-collection / breakup
   104	    dN_r_sc, dN_r_br = self_collection_breakup(
   105	        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
   106	    )
   107	
   108	    # Rain evaporation
   109	    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff)
   110	
   111	    # === ICE PHASE (Morrison processes) ===
   112	    T_freeze = constants.T_freeze
   113	    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))
   114	
   115	    # Ice nucleation
   116	    N_i_target = config.N_i0 * jnp.exp(
   117	        config.cooper_a * jnp.maximum(T_freeze - T, 0.0)
   118	    ) / jnp.clip(rho, 0.1)
   119	    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)
   120	
   121	    # Depositional growth
   122	    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
   123	    S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0
   124	    dq_i_dep = (
   125	        config.dep_coeff
   126	        * jnp.maximum(S_i, 0.0)
   127	        * jnp.clip(q_i, 0.0)
   128	        * safe_pow(N_i, 1.0 / 3.0)
   129	        * f_ice
   130	    )
   131	
   132	    # Bergeron
   133	    berg_window = (
   134	        jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
   135	        * jax.nn.sigmoid(config.melt_sharpness * (T - (config.T_center - config.T_width)))
   136	    )
   137	    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window
   138	
   139	    # Riming
   140	    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   141	    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   142	    total_riming = riming_i + riming_s
   143	
   144	    # Aggregation
   145	    aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice
   146	
   147	    # Melting (clamp to available mass so an explicit Euler step cannot
   148	    # drive q_i / q_s / q_g negative — same pattern Morrison already uses).
   149	    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
   150	    dt_safe = jnp.maximum(dt, 1e-10)
   151	    melt_ice = jnp.minimum(
   152	        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
   153	        jnp.clip(q_i, 0.0) / dt_safe,
   154	    )
   155	    melt_snow = jnp.minimum(
   156	        config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
   157	        jnp.clip(q_s, 0.0) / dt_safe,
   158	    )
   159	
   160	    # === GRAUPEL (Thompson extension) ===
   161	    # Split the rime → graupel conversion by donor: the fraction of
   162	    # rime_to_graupel that comes from q_i scales with riming_i, and
   163	    # the fraction from q_s scales with riming_s.  This avoids a
   164	    # mass-leak corner case where ``q_i = 0`` and ``riming_s > 0``:
   165	    # the previous form set ``rime_to_graupel ∝ total_riming``, then
   166	    # subtracted the FULL value from ``q_i`` (driving it negative)
   167	    # while only subtracting half from ``q_s`` and adding 150% to
   168	    # ``q_g`` — a non-conservative split that depended on the
   169	    # ad-hoc 1.0 / 0.5 / 1.5 coefficients.  (Codex audit cycle 2:
   170	    # "Thompson graupel conversion can draw from the wrong donor".)
   171	    graupel_frac = jax.nn.sigmoid(
   172	        config.graupel_sharpness * (total_riming - config.rime_to_graupel_threshold)
   173	    )
   174	    rime_to_graupel_from_i = config.rime_to_graupel_rate * riming_i * graupel_frac
   175	    rime_to_graupel_from_s = config.rime_to_graupel_rate * riming_s * graupel_frac
   176	    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
   177	    melt_graupel = jnp.minimum(
   178	        config.melt_rate * jnp.clip(q_g, 0.0) * melt_frac,
   179	        jnp.clip(q_g, 0.0) / dt_safe,
   180	    )
   181	
   182	    # === DONOR CLAMP for q_c sinks (see morrison.py for rationale) ===
   183	    # Include the evaporation branch of saturation_adjustment (negative
   184	    # condensation) in the q_c sink budget so subsaturated clear-air
   185	    # columns cannot drive q_c negative (Codex audit cycle 2).
   186	    cond_evap_sink = jnp.maximum(-condensation, 0.0)
   187	    qc_sink_total = (
   188	        dq_c_au + dq_c_ac + bergeron + riming_i + riming_s + cond_evap_sink
   189	    )
   190	    qc_avail = jnp.clip(q_c, 0.0)
   191	    qc_scale = jnp.minimum(
   192	        1.0,
   193	        qc_avail / jnp.maximum(qc_sink_total * dt_safe, 1e-30),
   194	    )
   195	    dq_c_au = dq_c_au * qc_scale
   196	    dq_c_ac = dq_c_ac * qc_scale
   197	    bergeron = bergeron * qc_scale
   198	    riming_i = riming_i * qc_scale
   199	    riming_s = riming_s * qc_scale
   200	    total_riming = riming_i + riming_s
   201	    # Each rime-to-graupel donor scales with its parent riming term —
   202	    # which has already been scaled by qc_scale above.  Re-scaling
   203	    # ``rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s``
   204	    # by ``qc_scale`` once preserves both per-donor proportionality and
   205	    # mass conservation.
   206	    rime_to_graupel_from_i = rime_to_graupel_from_i * qc_scale
   207	    rime_to_graupel_from_s = rime_to_graupel_from_s * qc_scale
   208	    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
   209	    dN_r_au = dN_r_au * qc_scale
   210	    # Scale negative-condensation (evaporation) branch by the same
   211	    # factor; positive condensation is unaffected (cond_evap_sink = 0).
   212	    condensation = jnp.where(
   213	        condensation < 0.0, condensation * qc_scale, condensation,
   214	    )
   215	
   216	    # === SEDIMENTATION ===
   217	    # Marshall-Palmer fall speeds use fractional exponents (b_v_x in
   218	    # [0.25, 0.5]); guard the AD path with safe_pow.
   219	    rho_sfc = rho[:, -1:]
   220	    rho_ratio = rho / jnp.clip(rho_sfc, 0.1)
   221	    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
   222	    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
   223	    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
   224	    V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
   225	    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
   226	    V_t_s = jnp.clip(V_t_s, 0.0, 5.0)
   227	    V_t_g = config.a_v_g * safe_pow(jnp.clip(q_g, 0.0) * rho_ratio, config.b_v_g)
   228	    V_t_g = jnp.clip(V_t_g, 0.0, 30.0)
   229	
   230	    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
   231	    sed_i = sedimentation_tendency(q_i, rho, V_t_i, dz)
   232	    sed_s = sedimentation_tendency(q_s, rho, V_t_s, dz)
   233	    sed_g = sedimentation_tendency(q_g, rho, V_t_g, dz)
   234	
   235	    # === LATENT HEATING ===
   236	    L_v = constants.L_v
   237	    L_s = constants.L_s
   238	    L_f = constants.L_f
   239	    c_pd = constants.c_pd
   240	    dT_dt = (
   241	        L_v * condensation / c_pd
   242	        - L_v * evaporation / c_pd
   243	        + L_s * dq_i_dep / c_pd
   244	        # Cloud water → ice/snow freezing releases L_f (Bergeron, riming).
   245	        # See morrison.py for the moist-enthalpy rationale; Thompson
   246	        # mirrors Morrison's ice-phase latent heating.
   247	        + L_f * (bergeron + riming_i + riming_s) / c_pd
   248	        - L_f * (melt_ice + melt_snow + melt_graupel) / c_pd
   249	    )
   250	
   251	    # === COMBINE TENDENCIES ===
   252	    # Conservation: each rime-to-graupel donor leaves its parent
   253	    # species and arrives in q_g.  The total mass moved is
   254	    # ``rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s``.
   255	    # An earlier form used 1.0 / 0.5 / 1.5 splits on ``rime_to_graupel``
   256	    # which (a) drove ``q_i`` negative when only snow was being rimed
   257	    # (``q_i = 0`` but ``riming_s > 0``), and (b) created mass
   258	    # apparently from nothing in the same regime.  See the
   259	    # ``=== GRAUPEL ===`` block above for the donor-split rationale.
   260	    dq_v_dt = -condensation + evaporation - dq_i_dep
   261	    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
   262	    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + melt_graupel + sed_r
   263	    dq_i_dt = (
   264	        dq_i_dep + bergeron + riming_i - aggregation - melt_ice
   265	        - rime_to_graupel_from_i + sed_i
   266	    )
   267	    dq_s_dt = aggregation + riming_s - melt_snow - rime_to_graupel_from_s + sed_s
   268	    dq_g_dt = rime_to_graupel - melt_graupel + sed_g
   269	
   270	    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
   271	    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
   272	    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
   273	
   274	    # Precipitation
   275	    precip_r = jnp.clip(q_r[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
   276	    precip_i = jnp.clip(q_i[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_i[:, -1], 0.0)
   277	    precip_s = jnp.clip(q_s[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_s[:, -1], 0.0)
   278	    precip_g = jnp.clip(q_g[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_g[:, -1], 0.0)
   279	    precipitation = precip_r + precip_i + precip_s + precip_g
   280	
   281	    return MicrophysicsOutput(
   282	        dT_dt=dT_dt,
   283	        dq_v_dt=dq_v_dt,
   284	        dq_c_dt=dq_c_dt,
   285	        dq_r_dt=dq_r_dt,
   286	        dq_i_dt=dq_i_dt,
   287	        dq_s_dt=dq_s_dt,
   288	        dq_g_dt=dq_g_dt,
   289	        dN_c_dt=dN_c_dt,
   290	        dN_r_dt=dN_r_dt,
   291	        dN_i_dt=dN_i_dt,
   292	        precipitation=precipitation,
   293	    )

 succeeded in 0ms:
     1	"""Configuration for atmospheric microphysics schemes.
     2	
     3	Provides configuration NamedTuples for:
     4	1. Kessler — warm-rain one-moment (refactored from physics/kessler.py)
     5	2. Sundqvist — large-scale diagnostic condensation
     6	3. Seifert-Beheng — two-moment warm rain
     7	4. Morrison — double-moment ice+liquid
     8	5. Thompson — hybrid moment with graupel
     9	6. ML Emulator — Equinox MLP surrogate
    10	7. Top-level MicrophysicsConfig that selects the active scheme.
    11	
    12	References
    13	----------
    14	- Kessler (1969): On the Distribution and Continuity of Water Substance.
    15	- Sundqvist et al. (1989): Condensation and cloud parameterization studies.
    16	- Seifert & Beheng (2001): A two-moment cloud microphysics scheme.
    17	- Morrison et al. (2005): A new double-moment microphysics scheme.
    18	- Thompson et al. (2008): Explicit forecasts of winter precipitation.
    19	"""
    20	
    21	from __future__ import annotations
    22	
    23	from typing import NamedTuple
    24	
    25	
    26	class KesslerConfig(NamedTuple):
    27	    """Configuration for Kessler warm-rain microphysics."""
    28	    autoconversion_threshold: float = 1.0e-3   # q_c threshold [kg/kg]
    29	    autoconversion_rate: float = 1.0e-3         # Rate [1/s]
    30	    accretion_coeff: float = 2.2                # Collection coefficient
    31	    evaporation_coeff: float = 1.0              # Evaporation coefficient
    32	    rain_fall_speed: float = 5.0                # Terminal velocity [m/s]
    33	    saturation_sharpness: float = 100.0         # Smooth switch sharpness
    34	
    35	
    36	class SundqvistConfig(NamedTuple):
    37	    """Configuration for Sundqvist large-scale condensation."""
    38	    RH_crit: float = 0.8              # Critical relative humidity
    39	    sigmoid_sharpness: float = 20.0   # Sharpness for smooth activation
    40	    auto_rate: float = 1e-3           # Autoconversion rate [1/s]
    41	    evap_coeff: float = 5e-4          # Sub-cloud evaporation coefficient
    42	
    43	
    44	class SeifertBehengConfig(NamedTuple):
    45	    """Configuration for Seifert-Beheng two-moment warm rain."""
    46	    k_au: float = 6e2                # Autoconversion rate [1/(kg*s)]
    47	    x_star: float = 2.6e-10          # Separation mass [kg]
    48	    Nc_0: float = 1e8                # Initial cloud droplet number [1/kg]
    49	    k_ac: float = 5.25               # Accretion rate [m^3/(kg*s)]
    50	    k_sc: float = 1e-3               # Self-collection rate [m^3/(kg*s)]
    51	    D_eq: float = 1.1e-3             # Equilibrium breakup diameter [m]
    52	    breakup_sharpness: float = 1e4   # Sigmoid sharpness for breakup
    53	    a_v_r: float = 130.0             # Rain fall speed coefficient a [m^(1-b)/s]
    54	    b_v_r: float = 0.5               # Rain fall speed exponent b
    55	    evap_coeff: float = 1.0          # Evaporation coefficient
    56	    saturation_sharpness: float = 100.0  # Sigmoid sharpness for saturation
    57	
    58	
    59	class MorrisonConfig(NamedTuple):
    60	    """Configuration for Morrison double-moment (ice+liquid)."""
    61	    # Warm rain (same as SB)
    62	    k_au: float = 6e2
    63	    x_star: float = 2.6e-10
    64	    Nc_0: float = 1e8
    65	    k_ac: float = 5.25
    66	    k_sc: float = 1e-3
    67	    D_eq: float = 1.1e-3
    68	    breakup_sharpness: float = 1e4
    69	    a_v_r: float = 130.0
    70	    b_v_r: float = 0.5
    71	    evap_coeff: float = 1.0
    72	    saturation_sharpness: float = 100.0
    73	    # Ice nucleation (Cooper 1986)
    74	    N_i0: float = 5e3               # Base ice crystal number [1/m^3]
    75	    cooper_a: float = 0.304          # Cooper exponent
    76	    cooper_T_act: float = 265.0      # Activation temperature [K]
    77	    ice_sigmoid_sharpness: float = 5.0  # Sharpness for ice-liquid partition
    78	    # Depositional growth
    79	    dep_coeff: float = 1e-3          # Deposition growth coefficient
    80	    # Bergeron
    81	    bergeron_rate: float = 1e-3      # Bergeron conversion rate [1/s]
    82	    T_center: float = 258.0          # Bergeron T window center [K]
    83	    T_width: float = 10.0            # Bergeron T window width [K]
    84	    # Riming
    85	    rime_coeff: float = 1.0          # Riming collection efficiency
    86	    # Aggregation
    87	    agg_coeff: float = 1e-3          # Ice-to-snow aggregation rate [1/s]
    88	    # Melting
    89	    melt_rate: float = 5e-3          # Melting rate [1/s]
    90	    melt_sharpness: float = 2.0      # Sigmoid sharpness near T_freeze
    91	    # Ice sedimentation
    92	    a_v_i: float = 50.0              # Ice fall speed coefficient [m^(1-b)/s]
    93	    b_v_i: float = 0.25              # Ice fall speed exponent
    94	    # Snow sedimentation
    95	    a_v_s: float = 30.0              # Snow fall speed coefficient
    96	    b_v_s: float = 0.3               # Snow fall speed exponent
    97	
    98	
    99	class ThompsonConfig(NamedTuple):
   100	    """Configuration for Thompson hybrid-moment microphysics."""
   101	    # All Morrison params
   102	    k_au: float = 6e2
   103	    x_star: float = 2.6e-10
   104	    Nc_0: float = 1e8
   105	    k_ac: float = 5.25
   106	    k_sc: float = 1e-3
   107	    D_eq: float = 1.1e-3
   108	    breakup_sharpness: float = 1e4
   109	    a_v_r: float = 130.0
   110	    b_v_r: float = 0.5
   111	    evap_coeff: float = 1.0
   112	    saturation_sharpness: float = 100.0
   113	    N_i0: float = 5e3
   114	    cooper_a: float = 0.304
   115	    cooper_T_act: float = 265.0
   116	    ice_sigmoid_sharpness: float = 5.0
   117	    dep_coeff: float = 1e-3
   118	    bergeron_rate: float = 1e-3
   119	    T_center: float = 258.0
   120	    T_width: float = 10.0
   121	    rime_coeff: float = 1.0
   122	    agg_coeff: float = 1e-3
   123	    melt_rate: float = 5e-3
   124	    melt_sharpness: float = 2.0
   125	    a_v_i: float = 50.0
   126	    b_v_i: float = 0.25
   127	    a_v_s: float = 30.0
   128	    b_v_s: float = 0.3
   129	    # Graupel
   130	    rime_to_graupel_threshold: float = 1e-4  # Riming threshold for graupel [kg/kg/s]
   131	    rime_to_graupel_rate: float = 0.5        # Fraction converted to graupel
   132	    graupel_sharpness: float = 1e4           # Sigmoid sharpness
   133	    a_v_g: float = 80.0                      # Graupel fall speed coefficient
   134	    b_v_g: float = 0.4                       # Graupel fall speed exponent
   135	    # Gamma distribution shape
   136	    mu_c: float = 3.0                        # Cloud droplet shape parameter
   137	    mu_r: float = 1.0                        # Rain drop shape parameter
   138	
   139	
   140	class MLEmulatorConfig(NamedTuple):
   141	    """Configuration for ML microphysics emulator (Equinox MLP)."""
   142	    n_input: int = 9
   143	    n_hidden: int = 128
   144	    n_layers: int = 3
   145	    n_output: int = 7
   146	    seed: int = 0
   147	    use_residual: bool = True
   148	    norm_T: float = 300.0       # Temperature scale [K] for input normalization
   149	    norm_q_factor: float = 1e3  # q_v / q_c / q_r / q_i scale
   150	    norm_rho: float = 1.2       # Air density scale [kg/m^3]
   151	    norm_dz: float = 1000.0     # Layer thickness scale [m]
   152	    norm_dt: float = 3600.0     # Time-step scale [s]
   153	
   154	
   155	class MicrophysicsConfig(NamedTuple):
   156	    """Top-level microphysics configuration.
   157	
   158	    Selects the active scheme and holds sub-configurations.
   159	
   160	    Fields
   161	    ------
   162	    scheme : str
   163	        Active scheme: "kessler", "sundqvist", "seifert_beheng",
   164	        "morrison", "thompson", "ml_emulator", or "none".
   165	    kessler : KesslerConfig
   166	    sundqvist : SundqvistConfig
   167	    seifert_beheng : SeifertBehengConfig
   168	    morrison : MorrisonConfig
   169	    thompson : ThompsonConfig
   170	    ml_emulator : MLEmulatorConfig
   171	    """
   172	    scheme: str = "none"
   173	    kessler: KesslerConfig = KesslerConfig()
   174	    sundqvist: SundqvistConfig = SundqvistConfig()
   175	    seifert_beheng: SeifertBehengConfig = SeifertBehengConfig()
   176	    morrison: MorrisonConfig = MorrisonConfig()
   177	    thompson: ThompsonConfig = ThompsonConfig()
   178	    ml_emulator: MLEmulatorConfig = MLEmulatorConfig()

 succeeded in 0ms:
     1	"""Morrison double-moment ice+liquid microphysics.
     2	
     3	Extends Seifert-Beheng warm-rain with ice-phase processes: nucleation
     4	(Cooper 1986), depositional growth, Bergeron process, riming, snow
     5	aggregation, and melting. Tracks cloud water, rain, ice, and snow.
     6	
     7	All operations use smooth (differentiable) approximations.
     8	
     9	References
    10	----------
    11	- Morrison, H., Curry, J. A., & Khvorostyanov, V. I. (2005). A new
    12	  double-moment microphysics parameterization. Part I: Description.
    13	  J. Atmos. Sci., 62, 1665-1677.
    14	"""
    15	
    16	from __future__ import annotations
    17	
    18	import jax
    19	import jax.numpy as jnp
    20	
    21	from legoesm import constants
    22	from legoesm.thermo import saturation_mixing_ratio_ice as _saturation_mixing_ratio_ice
    23	from legoesm.atmosphere.physics.microphysics._warm_rain import (
    24	    saturation_adjustment,
    25	    effective_Nc,
    26	    autoconversion_sb,
    27	    accretion,
    28	    self_collection_breakup,
    29	    rain_evaporation,
    30	    safe_pow,
    31	)
    32	from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    33	from legoesm.atmosphere.physics.microphysics.output import (
    34	    HydrometeorState,
    35	    MicrophysicsOutput,
    36	    sedimentation_tendency,
    37	)
    38	
    39	
    40	def morrison_microphysics(
    41	    T: jax.Array,
    42	    q_v: jax.Array,
    43	    hydrometeors: HydrometeorState,
    44	    p_full: jax.Array,
    45	    p_half: jax.Array,
    46	    rho: jax.Array,
    47	    dz: jax.Array,
    48	    dt: float,
    49	    config: MorrisonConfig = MorrisonConfig(),
    50	) -> MicrophysicsOutput:
    51	    """Compute Morrison double-moment microphysics tendencies.
    52	
    53	    Parameters
    54	    ----------
    55	    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
    56	        Same interface as all microphysics backends.
    57	
    58	    Returns
    59	    -------
    60	    MicrophysicsOutput
    61	    """
    62	    ncol, nlev = T.shape
    63	    q_c = hydrometeors.q_c
    64	    q_r = hydrometeors.q_r
    65	    q_i = hydrometeors.q_i
    66	    q_s = hydrometeors.q_s
    67	    N_c = hydrometeors.N_c
    68	    N_r = hydrometeors.N_r
    69	    N_i = hydrometeors.N_i
    70	    sharpness = config.saturation_sharpness
    71	
    72	    N_c_eff = effective_Nc(N_c, config.Nc_0)
    73	
    74	    # === WARM RAIN (shared Seifert-Beheng helpers) ===
    75	    # Pass ``q_c`` so the evaporation branch (negative ``condensation``)
    76	    # is donor-clamped: evaporation cannot drive ``q_c`` below zero in
    77	    # subsaturated clear air.  See _warm_rain.saturation_adjustment.
    78	    condensation, q_sat = saturation_adjustment(
    79	        T, q_v, p_full, dt, sharpness, q_c=q_c,
    80	    )
    81	    dq_c_au, dN_r_au, x_c = autoconversion_sb(
    82	        q_c, N_c_eff, rho, config.k_au, config.x_star, sharpness,
    83	    )
    84	    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac)
    85	    dN_r_sc, dN_r_br = self_collection_breakup(
    86	        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
    87	    )
    88	    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff)
    89	
    90	    # === ICE PHASE ===
    91	    T_freeze = constants.T_freeze
    92	    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))
    93	
    94	    # 1. Ice nucleation (Cooper 1986, smoothed)
    95	    N_i_target = config.N_i0 * jnp.exp(
    96	        config.cooper_a * jnp.maximum(T_freeze - T, 0.0)
    97	    ) / jnp.clip(rho, 0.1)
    98	    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)
    99	
   100	    # 2. Depositional growth
   101	    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
   102	    S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0
   103	    dq_i_dep = (
   104	        config.dep_coeff
   105	        * jnp.maximum(S_i, 0.0)
   106	        * jnp.clip(q_i, 0.0)
   107	        * safe_pow(N_i, 1.0 / 3.0)
   108	        * f_ice
   109	    )
   110	
   111	    # 3. Bergeron process: cloud water -> ice in mixed-phase zone
   112	    berg_window = (
   113	        jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
   114	        * jax.nn.sigmoid(config.melt_sharpness * (T - (config.T_center - config.T_width)))
   115	    )
   116	    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window
   117	
   118	    # 4. Riming: ice/snow collect cloud water
   119	    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   120	    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice
   121	
   122	    # 5. Snow aggregation: ice -> snow
   123	    aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice
   124	
   125	    # 6. Melting near T_freeze: ice/snow -> rain (clipped to available mass)
   126	    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
   127	    melt_ice = jnp.minimum(
   128	        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
   129	        jnp.clip(q_i, 0.0) / jnp.maximum(dt, 1e-10),
   130	    )
   131	    melt_snow = jnp.minimum(
   132	        config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
   133	        jnp.clip(q_s, 0.0) / jnp.maximum(dt, 1e-10),
   134	    )
   135	
   136	    # === DONOR CLAMP for q_c sinks ===
   137	    # Scale q_c-consuming processes (autoconversion, accretion, Bergeron,
   138	    # riming) by a common factor so the total loss per timestep does not
   139	    # exceed available q_c.  Without this clamp, default rates at dt = 1200s
   140	    # in a mixed-phase column drive q_c negative on a single explicit step
   141	    # (bergeron alone gives bergeron_rate * q_c * dt = 1.2 * q_c).  Mass is
   142	    # conserved because each process's matching source term in dq_r/dq_i/dq_s
   143	    # gets the same scale factor (the rates appear once as sinks in dq_c and
   144	    # once as sources elsewhere, so a uniform rescale preserves the budget).
   145	    #
   146	    # Include the *evaporation* branch of saturation adjustment (negative
   147	    # ``condensation``) in the q_c sink budget — otherwise a subsaturated
   148	    # clear-air column with q_c just above zero can lose more q_c to
   149	    # evaporation + accretion combined than is available, going negative
   150	    # (Codex audit cycle 2: "subsaturated clear air can create negative
   151	    # cloud water").  ``saturation_adjustment`` has already donor-clamped
   152	    # ``-condensation`` against ``q_c`` in isolation; including it here
   153	    # makes the joint budget consistent when other q_c sinks are active.
   154	    cond_evap_sink = jnp.maximum(-condensation, 0.0)
   155	    qc_sink_total = dq_c_au + dq_c_ac + bergeron + riming_i + riming_s + cond_evap_sink
   156	    qc_avail = jnp.clip(q_c, 0.0)
   157	    qc_scale = jnp.minimum(
   158	        1.0,
   159	        qc_avail / jnp.maximum(qc_sink_total * jnp.maximum(dt, 1e-10), 1e-30),
   160	    )
   161	    dq_c_au = dq_c_au * qc_scale
   162	    dq_c_ac = dq_c_ac * qc_scale
   163	    bergeron = bergeron * qc_scale
   164	    riming_i = riming_i * qc_scale
   165	    riming_s = riming_s * qc_scale
   166	    # Scale the evaporation branch by the same factor: when condensation
   167	    # is negative (evaporation), reduce its magnitude proportionally so
   168	    # q_c doesn't go negative.  When condensation is positive
   169	    # (saturation adjustment from supersaturation), the scaling does
   170	    # nothing because ``cond_evap_sink = 0``.
   171	    condensation = jnp.where(
   172	        condensation < 0.0, condensation * qc_scale, condensation,
   173	    )
   174	    # Number tendency for autoconverted droplets must scale identically.
   175	    dN_r_au = dN_r_au * qc_scale
   176	
   177	    # === SEDIMENTATION ===
   178	    # Marshall-Palmer fall speeds V_t = a_v * (q * rho / rho_sfc)^b_v use
   179	    # fractional exponents (b_v_r=0.5, b_v_i=0.25, b_v_s=0.3); guard the
   180	    # AD path with safe_pow so cold-start columns (q=0) don't NaN gradients.
   181	    rho_sfc = rho[:, -1:]
   182	    rho_ratio = rho / jnp.clip(rho_sfc, 0.1)
   183	    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
   184	    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
   185	    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
   186	    V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
   187	    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
   188	    V_t_s = jnp.clip(V_t_s, 0.0, 5.0)
   189	
   190	    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
   191	    sed_i = sedimentation_tendency(q_i, rho, V_t_i, dz)
   192	    sed_s = sedimentation_tendency(q_s, rho, V_t_s, dz)
   193	
   194	    # === LATENT HEATING ===
   195	    L_v = constants.L_v
   196	    L_s = constants.L_s
   197	    L_f = constants.L_f
   198	    c_pd = constants.c_pd
   199	
   200	    dT_dt = (
   201	        L_v * condensation / c_pd
   202	        - L_v * evaporation / c_pd
   203	        + L_s * dq_i_dep / c_pd
   204	        # Cloud water → ice/snow freezing releases latent heat of fusion
   205	        # (~333 kJ/kg).  Bergeron is liquid → ice via the WBF mechanism,
   206	        # riming is supercooled-droplet capture by ice/snow.  Both are
   207	        # phase changes that release L_f; the moist-enthalpy invariant
   208	        # ``h = c_pd T + L_v q_v - L_f q_ice`` requires this term for
   209	        # column conservation.  Magnitude estimate: ~2 K/day at default
   210	        # rates in mixed-phase clouds.
   211	        + L_f * (bergeron + riming_i + riming_s) / c_pd
   212	        - L_f * (melt_ice + melt_snow) / c_pd
   213	    )
   214	
   215	    # === COMBINE TENDENCIES ===
   216	    dq_v_dt = -condensation + evaporation - dq_i_dep
   217	    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
   218	    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + sed_r
   219	    dq_i_dt = dq_i_dep + bergeron + riming_i - aggregation - melt_ice + sed_i
   220	    dq_s_dt = aggregation + riming_s - melt_snow + sed_s
   221	
   222	    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
   223	    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
   224	    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
   225	
   226	    # Precipitation (rain + ice + snow at surface)
   227	    precip_r = jnp.clip(q_r[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
   228	    precip_i = jnp.clip(q_i[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_i[:, -1], 0.0)
   229	    precip_s = jnp.clip(q_s[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_s[:, -1], 0.0)
   230	    precipitation = precip_r + precip_i + precip_s
   231	
   232	    # Pin dtype to the input precision so we never silently promote
   233	    # the unused-species placeholders to f64 under x64 mode.
   234	    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
   235	    return MicrophysicsOutput(
   236	        dT_dt=dT_dt,
   237	        dq_v_dt=dq_v_dt,
   238	        dq_c_dt=dq_c_dt,
   239	        dq_r_dt=dq_r_dt,
   240	        dq_i_dt=dq_i_dt,
   241	        dq_s_dt=dq_s_dt,
   242	        dq_g_dt=z,
   243	        dN_c_dt=dN_c_dt,
   244	        dN_r_dt=dN_r_dt,
   245	        dN_i_dt=dN_i_dt,
   246	        precipitation=precipitation,
   247	    )

codex
The main clamp ordering is in the expected place: Morrison/Thompson scale `condensation` before `dq_v_dt`, `dq_c_dt`, and latent heating are formed, so later references do pick up the scaled value. I’m checking the shared warm-rain sinks now for any donor-budget gaps introduced by relying only on proportional source terms.
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/_warm_rain.py | sed -n '126,270p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/output.py | sed -n '1,180p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   126	def autoconversion_sb(q_c, N_c_eff, rho, k_au, x_star, sharpness=50.0, gamma_norm=1.0):
   127	    """Seifert-Beheng mass-dependent autoconversion.
   128	
   129	    Parameters
   130	    ----------
   131	    q_c : array
   132	        Cloud water mixing ratio [kg/kg].
   133	    N_c_eff : array
   134	        Effective cloud droplet number [1/kg].
   135	    rho : array
   136	        Air density [kg/m3].
   137	    k_au : float
   138	        Autoconversion rate constant.
   139	    x_star : float
   140	        Mean droplet mass threshold [kg].
   141	    sharpness : float
   142	        Sigmoid sharpness.
   143	    gamma_norm : float
   144	        Gamma distribution correction (1.0 for SB/Morrison, != 1.0 for Thompson).
   145	
   146	    Returns
   147	    -------
   148	    dq_c_au : array
   149	        Cloud water autoconversion rate [kg/kg/s].
   150	    dN_r_au : array
   151	        Rain number formation rate [1/kg/s].
   152	    x_c : array
   153	        Mean cloud droplet mass [kg].
   154	    """
   155	    q_c_pos = jnp.clip(q_c, 0.0)
   156	    x_c = q_c_pos * rho / jnp.clip(N_c_eff, 1.0)
   157	    onset = jax.nn.sigmoid(sharpness * (x_c - x_star))
   158	    dq_c_au = k_au * q_c_pos ** 2 * onset * gamma_norm * rho
   159	    dN_r_au = dq_c_au * rho / (x_star * 20.0)
   160	    return dq_c_au, dN_r_au, x_c
   161	
   162	
   163	def accretion(q_c, q_r, rho, k_ac, gamma_norm=1.0):
   164	    """Rain collecting cloud water (accretion).
   165	
   166	    Parameters
   167	    ----------
   168	    q_c, q_r : array
   169	        Cloud water and rain mixing ratios [kg/kg].
   170	    rho : array
   171	        Air density [kg/m3].
   172	    k_ac : float
   173	        Accretion rate constant.
   174	    gamma_norm : float
   175	        Gamma distribution correction.
   176	
   177	    Returns
   178	    -------
   179	    array : Accretion rate [kg/kg/s].
   180	    """
   181	    return k_ac * jnp.clip(q_c, 0.0) * jnp.clip(q_r, 0.0) * rho * gamma_norm
   182	
   183	
   184	def self_collection_breakup(N_r, q_r, rho, k_sc, breakup_sharpness, D_eq):
   185	    """Self-collection and breakup of rain drops.
   186	
   187	    Parameters
   188	    ----------
   189	    N_r : array
   190	        Rain drop number concentration [1/kg].
   191	    q_r : array
   192	        Rain mixing ratio [kg/kg].
   193	    rho : array
   194	        Air density [kg/m3].
   195	    k_sc : float
   196	        Self-collection rate constant.
   197	    breakup_sharpness : float
   198	        Sigmoid sharpness for breakup onset.
   199	    D_eq : float
   200	        Equilibrium drop diameter [m].
   201	
   202	    Returns
   203	    -------
   204	    dN_r_sc : array
   205	        Self-collection tendency [1/kg/s].
   206	    dN_r_br : array
   207	        Breakup tendency [1/kg/s].
   208	    """
   209	    dN_r_sc = -k_sc * jnp.clip(N_r, 0.0) * jnp.clip(q_r, 0.0) * rho
   210	    # Mean drop diameter D ~ (q_r * rho / N_r / (pi/6 * rho_water))^(1/3).
   211	    # Cube-root has unbounded derivative at zero — guard with safe_pow.
   212	    D_r_arg = (
   213	        jnp.clip(q_r, 0.0) * rho
   214	        / jnp.clip(N_r, 1.0)
   215	        / (jnp.pi / 6.0 * constants.rho_water)
   216	    )
   217	    D_r = safe_pow(D_r_arg, 1.0 / 3.0)
   218	    breakup_frac = jax.nn.sigmoid(breakup_sharpness * (D_r - D_eq))
   219	    dN_r_br = -dN_r_sc * breakup_frac
   220	    return dN_r_sc, dN_r_br
   221	
   222	
   223	def rain_evaporation(q_v, q_r, q_sat, evap_coeff):
   224	    """Compute rain evaporation in subsaturated air.
   225	
   226	    Parameters
   227	    ----------
   228	    q_v : array
   229	        Water vapor mixing ratio [kg/kg].
   230	    q_r : array
   231	        Rain mixing ratio [kg/kg].
   232	    q_sat : array
   233	        Saturation mixing ratio [kg/kg].
   234	    evap_coeff : float
   235	        Evaporation rate coefficient.
   236	
   237	    Returns
   238	    -------
   239	    array : Evaporation rate [kg/kg/s].
   240	    """
   241	    subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
   242	    # Marshall-Palmer ventilation factor q_r^0.525 — fractional power has
   243	    # an unbounded derivative at q_r=0; safe_pow handles the AD guard.
   244	    return evap_coeff * subsaturation * safe_pow(q_r, 0.525)

 succeeded in 0ms:
     1	"""Microphysics output containers.
     2	
     3	HydrometeorState holds the prognostic hydrometeor fields passed to backends.
     4	MicrophysicsOutput is the common interface returned by all backends.
     5	
     6	All backends accept and return the same containers so that integration
     7	code can be backend-agnostic.
     8	"""
     9	
    10	from __future__ import annotations
    11	
    12	from typing import NamedTuple
    13	
    14	import jax
    15	import jax.numpy as jnp
    16	
    17	
    18	class HydrometeorState(NamedTuple):
    19	    """Hydrometeor state for backends. All fields shape (ncol, nlev)."""
    20	    q_c: jax.Array    # cloud water [kg/kg]
    21	    q_r: jax.Array    # rain water [kg/kg]
    22	    q_i: jax.Array    # cloud ice [kg/kg]
    23	    q_s: jax.Array    # snow [kg/kg]
    24	    q_g: jax.Array    # graupel [kg/kg]
    25	    N_c: jax.Array    # cloud droplet number [1/kg]
    26	    N_r: jax.Array    # rain drop number [1/kg]
    27	    N_i: jax.Array    # ice crystal number [1/kg]
    28	
    29	
    30	class MicrophysicsOutput(NamedTuple):
    31	    """Backend-agnostic output. All (ncol, nlev) except precipitation (ncol,)."""
    32	    dT_dt: jax.Array          # latent heating [K/s]
    33	    dq_v_dt: jax.Array        # vapor tendency [kg/kg/s]
    34	    dq_c_dt: jax.Array        # cloud water tendency
    35	    dq_r_dt: jax.Array        # rain tendency
    36	    dq_i_dt: jax.Array        # ice tendency
    37	    dq_s_dt: jax.Array        # snow tendency
    38	    dq_g_dt: jax.Array        # graupel tendency
    39	    dN_c_dt: jax.Array        # cloud number tendency [1/kg/s]
    40	    dN_r_dt: jax.Array        # rain number tendency
    41	    dN_i_dt: jax.Array        # ice number tendency
    42	    precipitation: jax.Array  # surface precip [kg/m^2/s]
    43	
    44	
    45	def make_zero_hydrometeors(
    46	    ncol: int, nlev: int, dtype=None,
    47	) -> HydrometeorState:
    48	    """Create a zero-initialized HydrometeorState.
    49	
    50	    ``dtype`` defaults to the JAX default float (``float64`` under x64,
    51	    ``float32`` otherwise).  Callers integrating with the column physics
    52	    pipeline should pass the upstream state dtype explicitly so this
    53	    fallback never silently promotes a float32 column path to float64.
    54	    """
    55	    z = jnp.zeros((ncol, nlev), dtype=dtype)
    56	    return HydrometeorState(
    57	        q_c=z, q_r=z, q_i=z, q_s=z, q_g=z,
    58	        N_c=z, N_r=z, N_i=z,
    59	    )
    60	
    61	
    62	def make_zero_output(
    63	    ncol: int, nlev: int, dtype=None,
    64	) -> MicrophysicsOutput:
    65	    """Create a zero-initialized MicrophysicsOutput.
    66	
    67	    ``dtype`` is forwarded to ``jnp.zeros`` for the same reason as
    68	    ``make_zero_hydrometeors``: defaulting allows x64 mode to silently
    69	    promote the precip path.
    70	    """
    71	    z2 = jnp.zeros((ncol, nlev), dtype=dtype)
    72	    z1 = jnp.zeros((ncol,), dtype=dtype)
    73	    return MicrophysicsOutput(
    74	        dT_dt=z2, dq_v_dt=z2, dq_c_dt=z2, dq_r_dt=z2,
    75	        dq_i_dt=z2, dq_s_dt=z2, dq_g_dt=z2,
    76	        dN_c_dt=z2, dN_r_dt=z2, dN_i_dt=z2,
    77	        precipitation=z1,
    78	    )
    79	
    80	
    81	def sedimentation_tendency(
    82	    q: jax.Array,
    83	    rho: jax.Array,
    84	    V_t: jax.Array,
    85	    dz: jax.Array,
    86	) -> jax.Array:
    87	    """Compute sedimentation tendency from vertical flux divergence.
    88	
    89	    Parameters
    90	    ----------
    91	    q : jax.Array
    92	        Hydrometeor mixing ratio [kg/kg], shape (ncol, nlev).
    93	    rho : jax.Array
    94	        Air density [kg/m^3], shape (ncol, nlev).
    95	    V_t : jax.Array
    96	        Terminal velocity [m/s], shape (ncol, nlev).
    97	    dz : jax.Array
    98	        Layer thickness [m], shape (ncol, nlev).
    99	
   100	    Returns
   101	    -------
   102	    jax.Array
   103	        Sedimentation tendency [kg/kg/s], shape (ncol, nlev).
   104	    """
   105	    q_pos = jnp.clip(q, 0.0, None)
   106	    flux = V_t * q_pos * rho  # (ncol, nlev)
   107	
   108	    # Flux from above: zero at top, flux[k-1] enters level k.  Use
   109	    # ``jnp.pad`` (single Pad HLO) instead of allocating a fresh
   110	    # zero buffer + concatenate.
   111	    flux_in = jnp.pad(flux[:, :-1], ((0, 0), (1, 0)))
   112	    dz_safe = jnp.clip(dz, 1.0, None)
   113	    return (flux_in - flux) / (rho * dz_safe)

exec
/bin/zsh -lc "rg -n \"saturation_adjustment\\(\" src tests -S" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/atmosphere/physics/microphysics/thompson.py:85:    condensation, q_sat = saturation_adjustment(
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:72:    condensation, q_sat = saturation_adjustment(
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:52:def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0, q_c=None):
src/legoesm/atmosphere/physics/microphysics/morrison.py:78:    condensation, q_sat = saturation_adjustment(
tests/unit/test_warm_rain.py:41:def test_saturation_adjustment(column_state):
tests/unit/test_warm_rain.py:43:    cond, q_sat = saturation_adjustment(T, q_v, p, dt=300.0)
tests/unit/test_warm_rain.py:107:    _, q_sat = saturation_adjustment(T, q_v_low, p, dt=300.0)
tests/unit/test_warm_rain.py:121:        cond, q_sat = saturation_adjustment(T_, q_v_, p_, dt=300.0)

exec
/bin/zsh -lc "nl -ba tests/unit/test_warm_rain.py | sed -n '1,150p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/integration.py | sed -n '1,240p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Smoke tests for warm-rain microphysics helpers (`_warm_rain.py`).
     2	
     3	Run with:
     4	
     5	    JAX_ENABLE_X64=1 .venv/bin/python3.14 -m pytest \
     6	        tests/unit/test_warm_rain.py -v
     7	"""
     8	
     9	from __future__ import annotations
    10	
    11	import jax
    12	import jax.numpy as jnp
    13	import pytest
    14	
    15	from legoesm import constants
    16	from legoesm.atmosphere.physics.microphysics._warm_rain import (
    17	    saturation_adjustment,
    18	    effective_Nc,
    19	    autoconversion_sb,
    20	    accretion,
    21	    self_collection_breakup,
    22	    rain_evaporation,
    23	)
    24	
    25	
    26	@pytest.fixture
    27	def column_state():
    28	    """Build a small (ncol, nlev) column with realistic warm-cloud values."""
    29	    ncol, nlev = 4, 6
    30	    T = jnp.full((ncol, nlev), 285.0)            # mid-troposphere temperature [K]
    31	    p = jnp.full((ncol, nlev), 8.0e4)            # ~800 hPa
    32	    rho = p / (constants.R_d * T)
    33	    q_v = jnp.full((ncol, nlev), 1.5e-2)         # high vapor (likely supersat)
    34	    q_c = jnp.full((ncol, nlev), 5.0e-4)         # cloud water
    35	    q_r = jnp.full((ncol, nlev), 2.0e-4)         # rain
    36	    N_c = 1.0e8 * jnp.ones((ncol, nlev))         # cloud droplet number [1/kg]
    37	    N_r = 1.0e3 * jnp.ones((ncol, nlev))         # rain number [1/kg]
    38	    return T, p, rho, q_v, q_c, q_r, N_c, N_r
    39	
    40	
    41	def test_saturation_adjustment(column_state):
    42	    T, p, _, q_v, _, _, _, _ = column_state
    43	    cond, q_sat = saturation_adjustment(T, q_v, p, dt=300.0)
    44	    assert cond.shape == T.shape
    45	    assert q_sat.shape == T.shape
    46	    assert jnp.all(jnp.isfinite(cond))
    47	    assert jnp.all(jnp.isfinite(q_sat))
    48	    assert jnp.all(q_sat > 0.0)
    49	    # Supersaturated air → positive condensation
    50	    assert float(jnp.mean(cond)) > 0.0
    51	
    52	
    53	def test_effective_Nc(column_state):
    54	    *_, N_c, _ = column_state
    55	    Nc_eff = effective_Nc(N_c, Nc_0=5.0e7)
    56	    assert Nc_eff.shape == N_c.shape
    57	    assert jnp.all(jnp.isfinite(Nc_eff))
    58	    assert jnp.all(Nc_eff > 1.0)
    59	    # Where N_c is set (>1) the effective value passes through.
    60	    assert jnp.allclose(Nc_eff, N_c)
    61	    # Where N_c is zero, fallback kicks in.
    62	    fallback = effective_Nc(jnp.zeros_like(N_c), Nc_0=5.0e7)
    63	    assert jnp.allclose(fallback, 5.0e7)
    64	
    65	
    66	def test_autoconversion_sb(column_state):
    67	    _, _, rho, _, q_c, _, N_c, _ = column_state
    68	    dq_au, dN_au, x_c = autoconversion_sb(
    69	        q_c, N_c, rho, k_au=9.44e9, x_star=2.6e-10,
    70	    )
    71	    assert dq_au.shape == q_c.shape
    72	    assert dN_au.shape == q_c.shape
    73	    assert x_c.shape == q_c.shape
    74	    assert jnp.all(jnp.isfinite(dq_au))
    75	    assert jnp.all(jnp.isfinite(dN_au))
    76	    assert jnp.all(jnp.isfinite(x_c))
    77	    assert jnp.all(dq_au >= 0.0)
    78	    assert jnp.all(x_c > 0.0)
    79	
    80	
    81	def test_accretion(column_state):
    82	    _, _, rho, _, q_c, q_r, _, _ = column_state
    83	    rate = accretion(q_c, q_r, rho, k_ac=5.25)
    84	    assert rate.shape == q_c.shape
    85	    assert jnp.all(jnp.isfinite(rate))
    86	    assert jnp.all(rate >= 0.0)
    87	
    88	
    89	def test_self_collection_breakup(column_state):
    90	    _, _, rho, _, _, q_r, _, N_r = column_state
    91	    dN_sc, dN_br = self_collection_breakup(
    92	        N_r, q_r, rho, k_sc=7.12, breakup_sharpness=10.0, D_eq=9.0e-4,
    93	    )
    94	    assert dN_sc.shape == N_r.shape
    95	    assert dN_br.shape == N_r.shape
    96	    assert jnp.all(jnp.isfinite(dN_sc))
    97	    assert jnp.all(jnp.isfinite(dN_br))
    98	    # Self-collection always reduces number; breakup either zero or positive.
    99	    assert jnp.all(dN_sc <= 0.0)
   100	    assert jnp.all(dN_br >= 0.0)
   101	
   102	
   103	def test_rain_evaporation(column_state):
   104	    T, p, _, _, _, q_r, _, _ = column_state
   105	    # Subsaturated vapor profile to drive evaporation.
   106	    q_v_low = 1.0e-3 * jnp.ones_like(q_r)
   107	    _, q_sat = saturation_adjustment(T, q_v_low, p, dt=300.0)
   108	    evap = rain_evaporation(q_v_low, q_r, q_sat, evap_coeff=1.0e-3)
   109	    assert evap.shape == q_r.shape
   110	    assert jnp.all(jnp.isfinite(evap))
   111	    assert jnp.all(evap >= 0.0)
   112	    # With substantial subsaturation, average evaporation is strictly positive.
   113	    assert float(jnp.mean(evap)) > 0.0
   114	
   115	
   116	def test_jit_compiles(column_state):
   117	    T, p, rho, q_v, q_c, q_r, N_c, N_r = column_state
   118	
   119	    @jax.jit
   120	    def chain(T_, q_v_, q_c_, q_r_, N_c_, N_r_, rho_, p_):
   121	        cond, q_sat = saturation_adjustment(T_, q_v_, p_, dt=300.0)
   122	        Nc_eff = effective_Nc(N_c_, Nc_0=5.0e7)
   123	        dq_au, dN_au, _ = autoconversion_sb(
   124	            q_c_, Nc_eff, rho_, k_au=9.44e9, x_star=2.6e-10,
   125	        )
   126	        acc = accretion(q_c_, q_r_, rho_, k_ac=5.25)
   127	        evap = rain_evaporation(q_v_, q_r_, q_sat, evap_coeff=1.0e-3)
   128	        return cond + dq_au + dN_au + acc + evap
   129	
   130	    out = chain(T, q_v, q_c, q_r, N_c, N_r, rho, p)
   131	    assert out.shape == T.shape
   132	    assert jnp.all(jnp.isfinite(out))

 succeeded in 0ms:
     1	"""Model integration bridge for microphysics.
     2	
     3	Provides `make_microphysics_physics()`, a factory that returns a physics
     4	function matching each dynamical core's `step_with_physics` signature.
     5	
     6	Supported model types:
     7	- "hydrostatic"  : PrimitiveEquationModel (sigma coordinates)
     8	- "nonhydrostatic": CompressibleEulerModel (z* coordinates)
     9	- "spectral_pe"  : SpectralPEModel (Gaussian grid + sigma coordinates)
    10	"""
    11	
    12	from __future__ import annotations
    13	
    14	from typing import Callable
    15	
    16	import jax
    17	import jax.numpy as jnp
    18	
    19	from legoesm.core.field import Field
    20	from legoesm.core.state import (
    21	    HydrostaticState,
    22	    HydrostaticTendencies,
    23	    NonHydrostaticState,
    24	    NonHydrostaticTendencies,
    25	)
    26	from legoesm.grids.cubed_sphere import CubedSphereGrid
    27	from legoesm.grids.vertical import (
    28	    HeightCoordinate,
    29	    SigmaCoordinate,
    30	    TerrainMetric,
    31	    pressure_from_sigma,
    32	)
    33	from legoesm import constants
    34	
    35	from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    36	from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    37	from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
    38	from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
    39	from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
    40	from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
    41	from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
    42	from legoesm.atmosphere.physics.microphysics.ml_emulator import (
    43	    ml_microphysics,
    44	    MicrophysicsEmulator,
    45	)
    46	from legoesm.atmosphere.physics.thermodynamics import (
    47	    pressure_from_eos,
    48	    reconstruct_half_level_pressure_hydrostatic,
    49	    sanitize_theta_rho,
    50	)
    51	
    52	
    53	def _get_microphysics_fn(config: MicrophysicsConfig):
    54	    """Select the microphysics backend based on config.scheme.
    55	
    56	    Returns
    57	    -------
    58	    scheme_name : str
    59	    micro_fn : callable or None
    60	    scheme_config : NamedTuple or None
    61	    """
    62	    if config.scheme == "kessler":
    63	        return "kessler", kessler_microphysics, config.kessler
    64	    elif config.scheme == "sundqvist":
    65	        return "sundqvist", sundqvist_microphysics, config.sundqvist
    66	    elif config.scheme == "seifert_beheng":
    67	        return "seifert_beheng", seifert_beheng_microphysics, config.seifert_beheng
    68	    elif config.scheme == "morrison":
    69	        return "morrison", morrison_microphysics, config.morrison
    70	    elif config.scheme == "thompson":
    71	        return "thompson", thompson_microphysics, config.thompson
    72	    elif config.scheme == "ml_emulator":
    73	        return "ml_emulator", ml_microphysics, config.ml_emulator
    74	    elif config.scheme == "none":
    75	        return "none", None, None
    76	    else:
    77	        raise ValueError(f"Unknown microphysics scheme: {config.scheme!r}")
    78	
    79	
    80	from legoesm.atmosphere.physics._shared import (
    81	    compute_layer_dz as _compute_heights_from_sigma,
    82	    compute_rho as _compute_rho,
    83	)
    84	
    85	
    86	def make_microphysics_physics(
    87	    microphysics_config: MicrophysicsConfig,
    88	    model_type: str = "hydrostatic",
    89	    dt: float = 300.0,
    90	) -> Callable:
    91	    """Create a physics function for microphysics matching a model's signature.
    92	
    93	    Parameters
    94	    ----------
    95	    microphysics_config : MicrophysicsConfig
    96	        Microphysics configuration (selects scheme).
    97	    model_type : str
    98	        One of "hydrostatic", "nonhydrostatic", "spectral_pe".
    99	    dt : float
   100	        Model time step [s].
   101	
   102	    Returns
   103	    -------
   104	    Callable
   105	        Physics function with the correct signature for the model.
   106	    """
   107	    # ``model_type="mpas"`` reuses the hydrostatic factory: the
   108	    # ``_make_hydrostatic_microphysics`` bridge reshapes
   109	    # ``(*shape_2d, nlev)`` to ``(ncol, nlev)`` and never references
   110	    # grid lat/lon — works identically for cubed-sphere ``(face, n, n)``,
   111	    # lat-lon ``(n_lat, n_lon)``, and MPAS Voronoi ``(nCells,)``.
   112	    if model_type in ("hydrostatic", "mpas"):
   113	        return _make_hydrostatic_microphysics(microphysics_config, dt)
   114	    elif model_type == "nonhydrostatic":
   115	        return _make_nonhydrostatic_microphysics(microphysics_config, dt)
   116	    elif model_type == "spectral_pe":
   117	        return _make_spectral_pe_microphysics(microphysics_config, dt)
   118	    else:
   119	        raise ValueError(
   120	            f"Unknown model_type: {model_type!r}. "
   121	            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe', 'mpas'."
   122	        )
   123	
   124	
   125	# ===========================================================================
   126	# Hydrostatic PE
   127	# ===========================================================================
   128	
   129	def _make_hydrostatic_microphysics(
   130	    microphysics_config: MicrophysicsConfig,
   131	    dt: float,
   132	) -> Callable:
   133	    """Create microphysics physics_fn for PrimitiveEquationModel.
   134	
   135	    Signature: (state, grid, sigma_coord) -> HydrostaticTendencies
   136	    """
   137	    scheme_name, micro_fn, scheme_config = _get_microphysics_fn(microphysics_config)
   138	    is_ml = scheme_name == "ml_emulator"
   139	    _ml_model_cache = [None]
   140	
   141	    def physics_fn(
   142	        state: HydrostaticState,
   143	        grid,
   144	        sigma_coord: SigmaCoordinate,
   145	    ) -> HydrostaticTendencies:
   146	        T = state.T.data
   147	        p_s = state.p_s.data
   148	
   149	        nlev = sigma_coord.n_levels
   150	        shape_3d = T.shape
   151	        shape_2d = p_s.shape
   152	
   153	        # Derive Field metadata from the input state so the returned
   154	        # tendencies match the underlying grid: cubed-sphere uses
   155	        # ("face","x","y",...), lat-lon uses ("lat","lon",...), and
   156	        # MPAS uses ("nCells",...).
   157	        dims_3d = state.T.dims
   158	        dims_2d = state.p_s.dims
   159	        u_shape = state.u.data.shape
   160	        u_dims = state.u.dims
   161	        v_dims = state.v.dims if state.v is not None else None
   162	        v_shape = state.v.data.shape if state.v is not None else None
   163	
   164	        # Pin defaulted allocations to the state precision so we never
   165	        # silently flow x64 zeros into the column physics path.
   166	        _state_dtype = T.dtype
   167	
   168	        def _zero_dv_dt():
   169	            """``None`` for MPAS (no v), Field of zeros otherwise."""
   170	            if state.v is None:
   171	                return None
   172	            return Field(
   173	                data=jnp.zeros(v_shape, dtype=_state_dtype),
   174	                name="dv_dt_micro", dims=v_dims, units="m/s^2",
   175	            )
   176	
   177	        if micro_fn is None:
   178	            return HydrostaticTendencies(
   179	                du_dt=Field(
   180	                    data=jnp.zeros(u_shape, dtype=_state_dtype),
   181	                    name="du_dt_micro", dims=u_dims, units="m/s^2",
   182	                ),
   183	                dv_dt=_zero_dv_dt(),
   184	                dT_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dT_dt_micro", dims=dims_3d, units="K/s"),
   185	                dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dp_s_dt_micro", dims=dims_2d, units="Pa/s"),
   186	                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
   187	            )
   188	
   189	        # Pressure at full and half levels
   190	        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)
   191	        p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)
   192	
   193	        # Reshape to columns generically across cubed-sphere
   194	        # ``shape_2d=(6,n,n)``, lat-lon ``(n_lat,n_lon)``, and MPAS
   195	        # ``(nCells,)``.
   196	        ncol = 1
   197	        for s in shape_2d:
   198	            ncol *= int(s)
   199	        T_col = T.reshape(ncol, nlev)
   200	        p_full_col = p_full.reshape(ncol, nlev)
   201	        p_half_col = p_half.reshape(ncol, nlev + 1)
   202	        # Helper: extract a tracer from the tracer dict, returning a
   203	        # column-reshaped (ncol, nlev) array clipped to non-negative.
   204	        def _get_tracer(name):
   205	            if state.tracers is not None and name in state.tracers:
   206	                raw = state.tracers[name]
   207	                data = raw.data if hasattr(raw, "data") else raw
   208	                return jnp.maximum(data.reshape(ncol, nlev), 0.0)
   209	            return jnp.zeros((ncol, nlev), dtype=_state_dtype)
   210	
   211	        # Extract water vapor from tracers if available; else assume dry.
   212	        q_v_col = _get_tracer("q_v")
   213	
   214	        rho = _compute_rho(T_col, p_full_col)
   215	        dz = _compute_heights_from_sigma(T_col, p_half_col)
   216	
   217	        # Extract actual hydrometeor state from tracers (fall back to zero
   218	        # for any species not present in the tracer registry).
   219	        hydrometeors = HydrometeorState(
   220	            q_c=_get_tracer("q_c"),
   221	            q_r=_get_tracer("q_r"),
   222	            q_i=_get_tracer("q_i"),
   223	            q_s=_get_tracer("q_s"),
   224	            q_g=_get_tracer("q_g"),
   225	            N_c=_get_tracer("N_c"),
   226	            N_r=_get_tracer("N_r"),
   227	            N_i=_get_tracer("N_i"),
   228	        )
   229	
   230	        if is_ml:
   231	            if _ml_model_cache[0] is None:
   232	                key = jax.random.PRNGKey(scheme_config.seed)
   233	                _ml_model_cache[0] = MicrophysicsEmulator(
   234	                    scheme_config.n_input, scheme_config.n_hidden,
   235	                    scheme_config.n_layers, scheme_config.n_output, key=key,
   236	                )
   237	            micro_out = micro_fn(
   238	                T_col, q_v_col, hydrometeors,
   239	                p_full_col, p_half_col, rho, dz, dt,
   240	                scheme_config, _ml_model_cache[0],

exec
/bin/zsh -lc "nl -ba tests/unit/test_physics_microphysics.py | sed -n '1,120p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Category 4: Microphysics -- Physical Consistency.
     2	
     3	Tests total water conservation, temperature-moisture coupling (Clausius-
     4	Clapeyron), saturation adjustment, precipitation positivity, ice-phase
     5	bounds, and autoconversion threshold for all microphysics schemes.
     6	"""
     7	
     8	from __future__ import annotations
     9	
    10	import jax
    11	import jax.numpy as jnp
    12	import pytest
    13	
    14	from legoesm import constants
    15	from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
    16	from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
    17	from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
    18	from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
    19	from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
    20	from legoesm.atmosphere.physics.microphysics.config import (
    21	    KesslerConfig, SundqvistConfig, SeifertBehengConfig, MorrisonConfig,
    22	    ThompsonConfig,
    23	)
    24	from legoesm.atmosphere.physics.microphysics.output import (
    25	    HydrometeorState, make_zero_hydrometeors,
    26	)
    27	from legoesm.thermo import saturation_mixing_ratio
    28	
    29	
    30	# ---------------------------------------------------------------------------
    31	# Helpers
    32	# ---------------------------------------------------------------------------
    33	
    34	def _make_column(nlev=20, ncol=4, T_sfc=280.0, q_c_val=1e-4, supersaturated=False):
    35	    """Build a microphysics column with realistic conditions."""
    36	    p_s = 1.0e5
    37	    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    38	    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    39	    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
    40	    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))
    41	
    42	    T = T_sfc * jnp.clip(sigma_full, 0.01, None) ** 0.19
    43	    T = jnp.maximum(T, 200.0)
    44	    T = jnp.broadcast_to(T[None, :], (ncol, nlev))
    45	
    46	    rho = p_full / (constants.R_d * jnp.clip(T, 1.0, None))
    47	
    48	    dp = p_half[:, 1:] - p_half[:, :-1]
    49	    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    50	    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    51	    dz = jnp.abs(dz)
    52	
    53	    q_sat = saturation_mixing_ratio(T, p_full)
    54	    if supersaturated:
    55	        q_v = 1.2 * q_sat
    56	    else:
    57	        q_v = 0.8 * q_sat
    58	
    59	    q_c = jnp.zeros((ncol, nlev))
    60	    q_c = q_c.at[..., -5:].set(q_c_val)
    61	
    62	    hydro = HydrometeorState(
    63	        q_c=q_c,
    64	        q_r=jnp.zeros((ncol, nlev)),
    65	        q_i=jnp.zeros((ncol, nlev)),
    66	        q_s=jnp.zeros((ncol, nlev)),
    67	        q_g=jnp.zeros((ncol, nlev)),
    68	        N_c=1e8 * jnp.ones((ncol, nlev)),
    69	        N_r=jnp.zeros((ncol, nlev)),
    70	        N_i=jnp.zeros((ncol, nlev)),
    71	    )
    72	    return T, q_v, hydro, p_full, p_half, rho, dz
    73	
    74	
    75	def _call_scheme(name, T, q_v, hydro, p_full, p_half, rho, dz, dt=300.0):
    76	    """Call a microphysics scheme backend and return MicrophysicsOutput."""
    77	    if name == "kessler":
    78	        return kessler_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
    79	                                     config=KesslerConfig())
    80	    elif name == "sundqvist":
    81	        return sundqvist_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
    82	                                       config=SundqvistConfig())
    83	    elif name == "seifert_beheng":
    84	        return seifert_beheng_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
    85	                                            config=SeifertBehengConfig())
    86	    elif name == "morrison":
    87	        return morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
    88	                                      config=MorrisonConfig())
    89	    elif name == "thompson":
    90	        return thompson_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
    91	                                      config=ThompsonConfig())
    92	    else:
    93	        raise ValueError(f"Unknown scheme: {name}")
    94	
    95	
    96	ALL_SCHEMES = ["kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"]
    97	
    98	
    99	# ============================================================================
   100	# 4a  All outputs finite
   101	# ============================================================================
   102	
   103	@pytest.mark.parametrize("scheme", ALL_SCHEMES)
   104	def test_all_outputs_finite(scheme):
   105	    """All MicrophysicsOutput fields should be finite."""
   106	    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
   107	    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
   108	    for fname in out._fields:
   109	        val = getattr(out, fname)
   110	        assert jnp.all(jnp.isfinite(val)), f"{scheme}: {fname} has NaN/Inf"
   111	
   112	
   113	# ============================================================================
   114	# 4b  Temperature-moisture coupling (Clausius-Clapeyron)
   115	# ============================================================================
   116	
   117	@pytest.mark.parametrize("scheme", ALL_SCHEMES)
   118	def test_temperature_moisture_coupling(scheme):
   119	    """cp * dT_dt approx Lv * condensation_rate (first order).
   120	

exec
/bin/zsh -lc "nl -ba tests/unit/test_physics_microphysics.py | sed -n '120,260p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc "nl -ba tests/unit/test_physics_microphysics.py | sed -n '430,560p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   120	
   121	    Checks that where heating is significant, it correlates with moisture
   122	    removal (condensation heats, evaporation cools).
   123	    """
   124	    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(supersaturated=True)
   125	    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
   126	
   127	    lhs = constants.c_pd * out.dT_dt
   128	    rhs = -constants.L_v * out.dq_v_dt
   129	    scale = jnp.maximum(jnp.abs(lhs), 1e-10)
   130	    rel_err = jnp.abs(lhs - rhs) / scale
   131	    active = jnp.abs(lhs) > 1e-8
   132	    if jnp.any(active):
   133	        median_err = float(jnp.median(rel_err[active]))
   134	        assert median_err < 0.5, (
   135	            f"{scheme}: median Clausius-Clapeyron rel_err = {median_err:.3f}"
   136	        )
   137	
   138	
   139	# ============================================================================
   140	# 4c  Saturation adjustment: supersaturated -> vapor decreases
   141	# ============================================================================
   142	
   143	@pytest.mark.parametrize("scheme", ["kessler", "sundqvist"])
   144	def test_saturation_adjustment_vapor_decreases(scheme):
   145	    """Starting from supersaturated, vapor should decrease (dq_v_dt < 0)."""
   146	    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(
   147	        supersaturated=True, q_c_val=0.0
   148	    )
   149	    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
   150	
   151	    # Some levels should show condensation (dq_v_dt < 0)
   152	    min_dqv = float(jnp.min(out.dq_v_dt))
   153	    assert min_dqv < 0.0, (
   154	        f"{scheme}: no condensation in supersaturated column, min dq_v_dt = {min_dqv:.2e}"
   155	    )
   156	
   157	
   158	# ============================================================================
   159	# 4d  Precipitation non-negative
   160	# ============================================================================
   161	
   162	@pytest.mark.parametrize("scheme", ALL_SCHEMES)
   163	def test_precipitation_non_negative(scheme):
   164	    """Precipitation must be >= 0."""
   165	    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
   166	    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
   167	    min_precip = float(jnp.min(out.precipitation))
   168	    assert min_precip >= -1e-15, (
   169	        f"{scheme}: negative precipitation = {min_precip:.2e}"
   170	    )
   171	
   172	
   173	# ============================================================================
   174	# 4e  Ice-phase temperature bounds (Morrison, Thompson)
   175	# ============================================================================
   176	
   177	@pytest.mark.parametrize("scheme", ["morrison", "thompson"])
   178	def test_ice_no_formation_above_freezing(scheme):
   179	    """Above freezing (T > 273.15 K), ice formation should be negligible."""
   180	    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(T_sfc=290.0)
   181	    T_warm = jnp.maximum(T, 280.0)
   182	    out = _call_scheme(scheme, T_warm, q_v, hydro, p_full, p_half, rho, dz)
   183	
   184	    warm_mask = T_warm > constants.T_freeze
   185	    if jnp.any(warm_mask):
   186	        ice_formation = out.dq_i_dt[warm_mask]
   187	        max_ice_form = float(jnp.max(ice_formation))
   188	        # Allow small numerical noise from sigmoid tails
   189	        assert max_ice_form <= 1e-10, (
   190	            f"{scheme}: ice forms above freezing, max dq_i_dt = {max_ice_form:.2e}"
   191	        )
   192	
   193	
   194	# ============================================================================
   195	# 4g  Autoconversion threshold (Kessler)
   196	# ============================================================================
   197	
   198	def test_kessler_autoconversion_sensitivity():
   199	    """Rain production should increase when q_c exceeds autoconversion threshold."""
   200	    config = KesslerConfig()
   201	    # Below threshold: subsaturated, small q_c
   202	    T_lo, q_v_lo, hydro_lo, p_full, p_half, rho, dz = _make_column(
   203	        q_c_val=0.1 * config.autoconversion_threshold, supersaturated=False
   204	    )
   205	    out_lo = kessler_microphysics(T_lo, q_v_lo, hydro_lo, p_full, p_half, rho, dz,
   206	                                   300.0, config=config)
   207	    # Above threshold: subsaturated, large q_c
   208	    _, _, hydro_hi, _, _, _, _ = _make_column(
   209	        q_c_val=5.0 * config.autoconversion_threshold, supersaturated=False
   210	    )
   211	    out_hi = kessler_microphysics(T_lo, q_v_lo, hydro_hi, p_full, p_half, rho, dz,
   212	                                   300.0, config=config)
   213	
   214	    max_dqr_lo = float(jnp.max(out_lo.dq_r_dt))
   215	    max_dqr_hi = float(jnp.max(out_hi.dq_r_dt))
   216	    assert max_dqr_hi > max_dqr_lo, (
   217	        f"Kessler: rain production not higher above threshold: "
   218	        f"lo={max_dqr_lo:.2e}, hi={max_dqr_hi:.2e}"
   219	    )
   220	
   221	
   222	# ============================================================================
   223	# Heating rate magnitude bounds
   224	# ============================================================================
   225	
   226	@pytest.mark.parametrize("scheme", ALL_SCHEMES)
   227	def test_heating_rate_bounded(scheme):
   228	    """|dT_dt| should be bounded (< 10 K/s for strongly supersaturated)."""
   229	    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(supersaturated=True)
   230	    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
   231	    max_hr = float(jnp.max(jnp.abs(out.dT_dt)))
   232	    assert max_hr < 10.0, (
   233	        f"{scheme}: |dT_dt| = {max_hr:.2e} K/s exceeds 10 K/s"
   234	    )
   235	
   236	
   237	# ============================================================================
   238	# Donor clamps: Morrison/Thompson q_c, q_i sinks must not over-extract
   239	# ============================================================================
   240	
   241	@pytest.mark.parametrize(
   242	    "scheme,call",
   243	    [("morrison", morrison_microphysics), ("thompson", thompson_microphysics)],
   244	)
   245	def test_qc_does_not_go_negative_at_long_dt(scheme, call):
   246	    """An explicit Euler step must not drive q_c below zero in mixed-phase.
   247	
   248	    Default config rates with dt=1200s drive Bergeron+riming+evaporation
   249	    sinks past 100% of q_c per step:
   250	        bergeron_rate = 1e-3 /s -> bergeron * dt = 1.2 * q_c
   251	    The fix proportionally scales all q_c sink processes so the total
   252	    loss per step is bounded by q_c (mass conservation preserved by
   253	    scaling source terms in dq_r/dq_i/dq_s by the same factor).
   254	
   255	    Saturation is set to RH=1.0 so saturation_adjustment does not
   256	    additionally evaporate cloud water; layer thickness is large
   257	    (10 km) to keep sedimentation Courant well below 1.
   258	    """
   259	    cfg_class = MorrisonConfig if scheme == "morrison" else ThompsonConfig
   260	    ncol, nlev = 1, 5

 succeeded in 0ms:
   430	    conservation violations.  Post-fix the donor split scales by
   431	    ``riming_i / riming_s`` and only the actually-rimed species is
   432	    drained.  Total mass moved (``rime_to_graupel_from_i +
   433	    rime_to_graupel_from_s``) goes 1:1 to graupel.
   434	    """
   435	    ncol, nlev = 1, 5
   436	    T = jnp.full((ncol, nlev), 250.0)  # below freezing — ice active
   437	    p_full = jnp.full((ncol, nlev), 5e4)
   438	    p_half = jnp.broadcast_to(
   439	        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
   440	    )
   441	    rho = p_full / (constants.R_d * T)
   442	    dz = jnp.full((ncol, nlev), 10_000.0)
   443	    # The pathological mix: cloud water + snow, NO ice.
   444	    q_c = jnp.full((ncol, nlev), 5e-3)
   445	    q_i = jnp.zeros((ncol, nlev))                # zero ice — would be drained negative pre-fix
   446	    q_s = jnp.full((ncol, nlev), 1e-3)            # snow gets rimed
   447	    hydro = HydrometeorState(
   448	        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
   449	        q_g=jnp.zeros_like(q_c),
   450	        N_c=1e8 * jnp.ones_like(q_c),
   451	        N_r=jnp.zeros_like(q_c),
   452	        N_i=jnp.zeros_like(q_c),
   453	    )
   454	    q_v = 0.5 * saturation_mixing_ratio(T, p_full)
   455	    dt = 1200.0
   456	    out = thompson_microphysics(
   457	        T, q_v, hydro, p_full, p_half, rho, dz, dt, ThompsonConfig(),
   458	    )
   459	    q_i_after = q_i + out.dq_i_dt * dt
   460	    # q_i must remain non-negative (donor split: only riming_i drains q_i).
   461	    min_qi = float(jnp.min(q_i_after))
   462	    assert min_qi >= -1e-9, (
   463	        f"Thompson: q_i went negative ({min_qi:.3e}) when starting at "
   464	        "q_i=0 with q_s>0 and active riming.  Pre-fix the rime_to_graupel "
   465	        "subtracted the full conversion from q_i regardless of which "
   466	        "species was actually rimed.  Audit cycle 2 Codex finding "
   467	        "'Thompson graupel conversion can draw from the wrong donor' "
   468	        "has regressed."
   469	    )
   470	    # Mass conservation: ∑ dq_i + dq_s + dq_g (pure ice phase) should
   471	    # equal sed_i + sed_s + sed_g (sedimentation only escapes the column);
   472	    # internal phase changes cancel in the sum.  We can verify rime->graupel
   473	    # specifically by checking that any q_g gain matches a q_s loss.
   474	    dq_g_total = float(jnp.sum(out.dq_g_dt))
   475	    # Without melting (T well below freeze), all q_g must come from
   476	    # rime_to_graupel.  q_g production must be matched by q_s loss
   477	    # (donor split: only riming_s active here).
   478	    assert dq_g_total >= 0.0, (
   479	        f"Thompson: dq_g_dt total ({dq_g_total:.3e}) is negative "
   480	        "without graupel sedimentation source — graupel mass conservation "
   481	        "violation."
   482	    )
   483	
   484	
   485	# ============================================================================
   486	# Morrison/Thompson moist-enthalpy conservation (latent heat of fusion)
   487	# ============================================================================
   488	
   489	@pytest.mark.parametrize(
   490	    "scheme,call",
   491	    [("morrison", morrison_microphysics), ("thompson", thompson_microphysics)],
   492	)
   493	def test_freezing_releases_latent_heat_of_fusion(scheme, call):
   494	    """Bergeron + riming (cloud water → ice/snow) must release ``L_f``.
   495	
   496	    Moist enthalpy ``h = c_pd T + L_v q_v - L_f * q_ice`` is conserved
   497	    by phase transitions in a closed column, so per-level the residual
   498	        c_pd * dT_dt + L_v * dq_v_dt - L_f * (dq_i + dq_s + dq_g)
   499	    is the divergence of sedimentation flux (surface boundary effect),
   500	    not a phase-change heating/cooling.
   501	
   502	    Pre-fix the dT_dt assembly omitted the ``+ L_f * (bergeron +
   503	    riming_i + riming_s)`` term, so in a mixed-phase column where
   504	    ~6e-4 kg/kg of cloud water freezes per step the column residual
   505	    is ~16 K/day too cold (L_f * freezing / c_pd = 333e3 * 6.6e-4 /
   506	    1004 / 1200s = 1.83e-4 K/s = 15.8 K/day at default rates).
   507	
   508	    Post-fix the residual collapses to the small contribution from
   509	    sedimentation flux divergence at the interior of the column.
   510	    """
   511	    cfg_class = MorrisonConfig if scheme == "morrison" else ThompsonConfig
   512	    ncol, nlev = 1, 5
   513	    # Mid mixed-phase regime, sub-saturated wrt liquid (suppresses
   514	    # condensation/evaporation as the dominant balance) so the
   515	    # cloud→ice freezing branches dominate the residual.  Short dt
   516	    # keeps sedimentation flux divergence small.
   517	    T = jnp.full((ncol, nlev), 260.0)
   518	    p_full = jnp.full((ncol, nlev), 5e4)
   519	    p_half = jnp.broadcast_to(
   520	        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
   521	    )
   522	    rho = p_full / (constants.R_d * T)
   523	    dz = jnp.full((ncol, nlev), 10_000.0)
   524	    q_c = jnp.full((ncol, nlev), 1e-3)
   525	    q_i = jnp.full((ncol, nlev), 5e-4)
   526	    q_s = jnp.full((ncol, nlev), 5e-4)
   527	    hydro = HydrometeorState(
   528	        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
   529	        q_g=jnp.zeros_like(q_c),
   530	        N_c=1e8 * jnp.ones_like(q_c),
   531	        N_r=jnp.zeros_like(q_c),
   532	        N_i=1e4 * jnp.ones_like(q_c),
   533	    )
   534	    q_v = 0.85 * saturation_mixing_ratio(T, p_full)
   535	    dt = 60.0
   536	    out = call(T, q_v, hydro, p_full, p_half, rho, dz, dt, cfg_class())
   537	
   538	    # Inspect interior level (k=2) where sedimentation flux divergence
   539	    # is approximately zero (uniform q_i, q_s ⇒ flux_in ≈ flux_out).
   540	    k = 2
   541	    h_residual = float(
   542	        constants.c_pd * out.dT_dt[0, k]
   543	        + constants.L_v * out.dq_v_dt[0, k]
   544	        - constants.L_f * (out.dq_i_dt[0, k] + out.dq_s_dt[0, k] + out.dq_g_dt[0, k])
   545	    )
   546	    # Magnitude scale of L_f-driven heating (lower bound)
   547	    freezing_scale = float(
   548	        constants.L_f * jnp.abs(out.dq_i_dt[0, k] + out.dq_s_dt[0, k] + out.dq_g_dt[0, k])
   549	    )
   550	    # Pre-fix: |h_residual| ≈ L_f * freezing rate (entire term missing);
   551	    # post-fix: |h_residual| << L_f * freezing rate (just sedimentation div).
   552	    assert abs(h_residual) < 0.2 * max(freezing_scale, 1e-10), (
   553	        f"{scheme}: per-level moist-enthalpy residual at k={k} = "
   554	        f"{h_residual:.3e} W/m^3-equivalent, freezing scale = "
   555	        f"{freezing_scale:.3e}; ratio = {abs(h_residual)/max(freezing_scale,1e-30):.3f}.  "
   556	        "dT_dt is missing +L_f * (bergeron + riming_i + riming_s) / c_pd "
   557	        "for the cloud-water → ice/snow freezing branches."
   558	    )
   559	
   560	

exec
/bin/zsh -lc 'rg -n "conserv|total water|dq_v|dq_c|mass" tests/unit/test_physics_microphysics.py -S' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
3:Tests total water conservation, temperature-moisture coupling (Clausius-
128:    rhs = -constants.L_v * out.dq_v_dt
145:    """Starting from supersaturated, vapor should decrease (dq_v_dt < 0)."""
151:    # Some levels should show condensation (dq_v_dt < 0)
152:    min_dqv = float(jnp.min(out.dq_v_dt))
154:        f"{scheme}: no condensation in supersaturated column, min dq_v_dt = {min_dqv:.2e}"
252:    loss per step is bounded by q_c (mass conservation preserved by
284:    q_c_after = q_c + out.dq_c_dt * dt
300:    to available mass.
354:    to ``dq_c_dt`` in subsaturated clear air (``q_c = 0``) produces a
398:    q_c_after = q_c + out.dq_c_dt * dt
407:        "adding it to ``dq_c_dt`` over an explicit Euler step drives "
429:    drives ``q_i`` negative and creates 1.5× extra mass — both
430:    conservation violations.  Post-fix the donor split scales by
432:    drained.  Total mass moved (``rime_to_graupel_from_i +
470:    # Mass conservation: ∑ dq_i + dq_s + dq_g (pure ice phase) should
480:        "without graupel sedimentation source — graupel mass conservation "
486:# Morrison/Thompson moist-enthalpy conservation (latent heat of fusion)
496:    Moist enthalpy ``h = c_pd T + L_v q_v - L_f * q_ice`` is conserved
498:        c_pd * dT_dt + L_v * dq_v_dt - L_f * (dq_i + dq_s + dq_g)
543:        + constants.L_v * out.dq_v_dt[0, k]
568:    Column total water tendency must balance surface precipitation:
569:        int (dq_v + dq_c + dq_r) dp/g  +  precipitation  ~  0
582:        (out.dq_v_dt + out.dq_c_dt + out.dq_r_dt) * dp, axis=1,
597:    surface in one step and account for the drained mass in
602:    under-reported the actual mass leaving.  This test would catch that
603:    regression: assert q_r is zero after one step AND the drained mass
627:    # Drained mass [kg/m^2/s] must appear in precipitation.
644:        (out.dq_v_dt + out.dq_c_dt + out.dq_r_dt) * dp, axis=1,
679:        return jnp.sum(out.dq_r_dt + out.dq_v_dt + out.dT_dt)

exec
/bin/zsh -lc "nl -ba tests/unit/test_physics_microphysics.py | sed -n '260,330p;560,660p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
   260	    ncol, nlev = 1, 5
   261	    T = jnp.full((ncol, nlev), 250.0)  # mixed-phase regime
   262	    p_full = jnp.full((ncol, nlev), 5e4)
   263	    p_half = jnp.broadcast_to(
   264	        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
   265	    )
   266	    rho = p_full / (constants.R_d * T)
   267	    dz = jnp.full((ncol, nlev), 10_000.0)
   268	    q_c = jnp.full((ncol, nlev), 1e-3)
   269	    q_i = jnp.full((ncol, nlev), 1e-4)
   270	    q_s = jnp.full((ncol, nlev), 1e-4)
   271	    hydro = HydrometeorState(
   272	        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
   273	        q_g=jnp.zeros_like(q_c),
   274	        N_c=1e8 * jnp.ones_like(q_c),
   275	        N_r=jnp.zeros_like(q_c),
   276	        N_i=1e4 * jnp.ones_like(q_c),
   277	    )
   278	    # RH = 1: keep condensation ~ 0 so this test isolates the q_c sinks
   279	    # (autoconv / accretion / Bergeron / riming) from saturation-driven
   280	    # cloud evaporation.
   281	    q_v = 1.0 * saturation_mixing_ratio(T, p_full)
   282	    dt = 1200.0
   283	    out = call(T, q_v, hydro, p_full, p_half, rho, dz, dt, cfg_class())
   284	    q_c_after = q_c + out.dq_c_dt * dt
   285	    min_qc = float(jnp.min(q_c_after))
   286	    # Allow ~1e-9 floor for floating-point round-off in scaled tendencies.
   287	    assert min_qc >= -1e-9, (
   288	        f"{scheme}: q_c went negative ({min_qc:.3e}) after one explicit "
   289	        f"step at dt={dt}s — Bergeron + riming + autoconv + accretion "
   290	        "sinks combined exceeded available q_c without a donor clamp."
   291	    )
   292	
   293	
   294	def test_thompson_qi_does_not_go_negative_warm():
   295	    """Thompson melt processes must not over-extract q_i above freezing.
   296	
   297	    Morrison clamps melt_ice/melt_snow with ``min(rate*q*frac, q/dt)``;
   298	    Thompson did not, so at T=280K (above freezing) and dt=1200s with
   299	    default rates, q_i was driven below zero.  After fix, melt is clamped
   300	    to available mass.
   301	
   302	    Layer thickness is 10 km so sedimentation Courant
   303	    ``V_t * dt / dz <= 5*1200/10000 = 0.6`` stays below 1.
   304	    """
   305	    ncol, nlev = 1, 5
   306	    T = jnp.full((ncol, nlev), 280.0)  # above freezing
   307	    p_full = jnp.full((ncol, nlev), 5e4)
   308	    p_half = jnp.broadcast_to(
   309	        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
   310	    )
   311	    rho = p_full / (constants.R_d * T)
   312	    dz = jnp.full((ncol, nlev), 10_000.0)
   313	    q_c = jnp.full((ncol, nlev), 1e-3)
   314	    q_i = jnp.full((ncol, nlev), 1e-4)
   315	    q_s = jnp.full((ncol, nlev), 1e-4)
   316	    hydro = HydrometeorState(
   317	        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
   318	        q_g=jnp.zeros_like(q_c),
   319	        N_c=1e8 * jnp.ones_like(q_c),
   320	        N_r=jnp.zeros_like(q_c),
   321	        N_i=1e4 * jnp.ones_like(q_c),
   322	    )
   323	    q_v = 0.8 * saturation_mixing_ratio(T, p_full)
   324	    dt = 1200.0
   325	    out = thompson_microphysics(
   326	        T, q_v, hydro, p_full, p_half, rho, dz, dt, ThompsonConfig(),
   327	    )
   328	    q_i_after = q_i + out.dq_i_dt * dt
   329	    # Skip the top level: with q_i uniform throughout the column the top
   330	    # level always loses sedimentation flux without compensating inflow,
   560	
   561	# ============================================================================
   562	# Sundqvist column water budget (closes against surface precipitation)
   563	# ============================================================================
   564	
   565	def test_sundqvist_column_water_budget_closes():
   566	    """Sundqvist is a diagnostic scheme — rain falls instantly.
   567	
   568	    Column total water tendency must balance surface precipitation:
   569	        int (dq_v + dq_c + dq_r) dp/g  +  precipitation  ~  0
   570	    """
   571	    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(
   572	        T_sfc=290.0, q_c_val=1e-3,
   573	    )
   574	    # Saturate the column so condensation (and therefore rain) actually fires.
   575	    q_sat = saturation_mixing_ratio(T, p_full)
   576	    q_v = q_sat
   577	    out = sundqvist_microphysics(
   578	        T, q_v, hydro, p_full, p_half, rho, dz, 300.0, SundqvistConfig(),
   579	    )
   580	    dp = p_half[:, 1:] - p_half[:, :-1]
   581	    col_water_tend = jnp.sum(
   582	        (out.dq_v_dt + out.dq_c_dt + out.dq_r_dt) * dp, axis=1,
   583	    ) / constants.g
   584	    imbalance = col_water_tend + out.precipitation
   585	    # Allow numerics; pre-fix imbalance was ~+9e-3 (= +precipitation).
   586	    max_imbalance = float(jnp.max(jnp.abs(imbalance)))
   587	    max_precip = float(jnp.max(out.precipitation))
   588	    assert max_imbalance < 1e-6 * max(max_precip, 1.0), (
   589	        f"Sundqvist water budget unclosed: imbalance = {max_imbalance:.3e}, "
   590	        f"precip scale = {max_precip:.3e}"
   591	    )
   592	
   593	
   594	def test_sundqvist_drains_incoming_qr_to_precipitation():
   595	    """If q_r is non-zero on input (e.g., warm-started or after a prior
   596	    Kessler step), Sundqvist's diagnostic-rain semantics drain it to the
   597	    surface in one step and account for the drained mass in
   598	    ``precipitation``.
   599	
   600	    Previously ``dq_r_dt = 0`` left any pre-existing q_r frozen in the
   601	    column — column water grew indefinitely and surface precipitation
   602	    under-reported the actual mass leaving.  This test would catch that
   603	    regression: assert q_r is zero after one step AND the drained mass
   604	    appears in the precipitation flux.
   605	    """
   606	    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(
   607	        T_sfc=290.0, q_c_val=0.0,
   608	    )
   609	    # Sub-saturated column with NO q_c so condensation/autoconversion
   610	    # don't generate fresh rain — this isolates the q_r-drain channel.
   611	    q_v_sub = 0.5 * saturation_mixing_ratio(T, p_full)
   612	    # Inject q_r at a few mid-column levels so dq_r_drain is nontrivial.
   613	    q_r_in = jnp.zeros_like(hydro.q_r).at[..., 8:12].set(5e-4)
   614	    hydro_with_qr = hydro._replace(q_r=q_r_in)
   615	
   616	    dt = 300.0
   617	    out = sundqvist_microphysics(
   618	        T, q_v_sub, hydro_with_qr, p_full, p_half, rho, dz, dt, SundqvistConfig(),
   619	    )
   620	    q_r_after = q_r_in + out.dq_r_dt * dt
   621	    # q_r must drain to zero in one step (modulo float round-off).
   622	    max_qr_residual = float(jnp.max(jnp.abs(q_r_after)))
   623	    assert max_qr_residual < 1e-12, (
   624	        f"Sundqvist failed to drain incoming q_r: max(|q_r_after|) = "
   625	        f"{max_qr_residual:.3e}; expected ~0 (diagnostic-rain semantics)."
   626	    )
   627	    # Drained mass [kg/m^2/s] must appear in precipitation.
   628	    dp = p_half[:, 1:] - p_half[:, :-1]
   629	    expected_drain_flux = jnp.sum(q_r_in * dp, axis=1) / (constants.g * dt)
   630	    # No autoconversion source here (q_c=0, sub-saturated) so precip
   631	    # equals the drain flux to a few percent (allowing for the smooth
   632	    # condensation sigmoid producing a tiny residual).
   633	    rel_err = float(jnp.max(
   634	        jnp.abs(out.precipitation - expected_drain_flux) /
   635	        jnp.maximum(expected_drain_flux, 1e-30),
   636	    ))
   637	    assert rel_err < 0.05, (
   638	        f"Sundqvist precip = {[float(x) for x in out.precipitation]}; "
   639	        f"expected drain flux = {[float(x) for x in expected_drain_flux]} "
   640	        "(no fresh autoconversion in this dry-c column)."
   641	    )
   642	    # Column water budget must still close with q_r in the loop.
   643	    col_water_tend = jnp.sum(
   644	        (out.dq_v_dt + out.dq_c_dt + out.dq_r_dt) * dp, axis=1,
   645	    ) / constants.g
   646	    imbalance = col_water_tend + out.precipitation
   647	    max_imbalance = float(jnp.max(jnp.abs(imbalance)))
   648	    max_precip = float(jnp.max(out.precipitation))
   649	    assert max_imbalance < 1e-6 * max(max_precip, 1.0), (
   650	        f"Sundqvist q_r-drain water budget unclosed: imbalance = "
   651	        f"{max_imbalance:.3e}, precip scale = {max_precip:.3e}"
   652	    )
   653	
   654	
   655	# ============================================================================
   656	# 4l  Kessler / warm-rain accretion & evaporation: gradient finite at q_r=0
   657	# ============================================================================
   658	
   659	def test_kessler_grad_finite_at_zero_qr():
   660	    """Marshall-Palmer fractional powers q_r**0.875 (accretion) and

exec
/bin/zsh -lc 'git diff -- src/legoesm/atmosphere/physics/microphysics/_warm_rain.py src/legoesm/atmosphere/physics/microphysics/kessler.py src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py src/legoesm/atmosphere/physics/microphysics/morrison.py src/legoesm/atmosphere/physics/microphysics/thompson.py tests/unit/test_physics_microphysics.py' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-trSdYcot' (errno=Operation not permitted)
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-wx3R3f4S' (errno=Operation not permitted)
diff --git a/src/legoesm/atmosphere/physics/microphysics/_warm_rain.py b/src/legoesm/atmosphere/physics/microphysics/_warm_rain.py
index d4d19a2..986bb9f 100644
--- a/src/legoesm/atmosphere/physics/microphysics/_warm_rain.py
+++ b/src/legoesm/atmosphere/physics/microphysics/_warm_rain.py
@@ -13,7 +13,43 @@ from legoesm import constants
 from legoesm.thermo import saturation_mixing_ratio
 
 
-def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0):
+def safe_pow(x, p):
+    """Differentiable ``x ** p`` with grad=0 wherever ``x <= 0``.
+
+    Microphysics has many Marshall-Palmer-style fractional powers of
+    hydrometeor mixing ratios (``q_r``, ``q_i``, ``q_s``, ``q_g``,
+    ``N_i``, …) with exponents in (0, 1) — typically 0.5 for fall
+    speeds, 1/3 for diameters, 0.525/0.875 for ventilation/accretion.
+    Their analytic derivative ``p * x**(p-1)`` is unbounded at ``x=0``
+    and complex for ``x<0``.  ``jnp.clip(x, 0.0) ** p`` therefore
+    returns ``inf`` (at zero) or ``nan`` (at negatives) under
+    ``jax.grad``, breaking AD on cold-start (no-precip) initial
+    conditions.
+
+    The double-where pattern below routes the AD graph through a
+    placeholder of 1.0 in the inactive branch so the gradient never
+    sees ``0**(p-1)``.
+
+    Parameters
+    ----------
+    x : array
+        Argument of the power.  May be zero or negative.
+    p : float or array
+        Exponent.  Intended for ``0 < p < 1`` where the bug applies;
+        also safe for ``p >= 1``.
+
+    Returns
+    -------
+    array
+        ``x ** p`` for ``x > 0``, else 0; gradient is finite
+        everywhere.
+    """
+    positive = x > 0.0
+    safe_x = jnp.where(positive, x, 1.0)
+    return jnp.where(positive, safe_x ** p, 0.0)
+
+
+def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0, q_c=None):
     """Compute smooth saturation adjustment (condensation tendency).
 
     Parameters
@@ -28,18 +64,45 @@ def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0):
         Time step [s].
     sharpness : float
         Sigmoid sharpness for smooth condensation switch.
+    q_c : array or None
+        Cloud water mixing ratio [kg/kg].  When provided, the negative
+        (evaporation) branch is donor-clamped against ``q_c`` so that
+        evaporation cannot drive ``q_c`` below zero in subsaturated
+        clear air.  Without ``q_c`` the legacy signed return is
+        produced (callers must apply their own donor clamp).
 
     Returns
     -------
     condensation : array (ncol, nlev)
-        Condensation tendency [kg/kg/s].
+        Condensation tendency [kg/kg/s].  Positive = condensation;
+        negative = evaporation (donor-clamped against ``q_c`` when
+        provided).  Without ``q_c``, the legacy unclamped signed
+        value is returned for backward compatibility.
     q_sat : array (ncol, nlev)
         Saturation mixing ratio [kg/kg].
+
+    Notes
+    -----
+    A subsaturated column with ``q_c = 0`` would otherwise produce
+    spurious negative ``q_c`` after the explicit Euler step ``q_c_new
+    = q_c + condensation * dt`` — Codex audit cycle 2 finding
+    "subsaturated clear air can create negative cloud water".  The
+    ``q_c``-aware donor clamp on the evaporation branch is the
+    minimal fix that conserves total water in both clear and cloudy
+    columns.
     """
     q_sat = saturation_mixing_ratio(T, p_full)
     excess = q_v - q_sat
     cond_frac = jax.nn.sigmoid(sharpness * excess)
     condensation = cond_frac * excess / dt
+    if q_c is not None:
+        # Evaporation rate (negative ``condensation``) is bounded by
+        # the available cloud water: |condensation| × dt ≤ q_c, i.e.
+        # condensation ≥ -q_c / dt.  ``maximum(condensation, -q_c/dt)``
+        # achieves this cleanly.  Differentiable everywhere — the
+        # clamp is a smooth-ish max on the evaporation magnitude.
+        q_c_avail = jnp.clip(q_c, 0.0, None)
+        condensation = jnp.maximum(condensation, -q_c_avail / jnp.maximum(dt, 1e-10))
     return condensation, q_sat
 
 
@@ -144,10 +207,14 @@ def self_collection_breakup(N_r, q_r, rho, k_sc, breakup_sharpness, D_eq):
         Breakup tendency [1/kg/s].
     """
     dN_r_sc = -k_sc * jnp.clip(N_r, 0.0) * jnp.clip(q_r, 0.0) * rho
-    D_r = jnp.clip(
-        (jnp.clip(q_r, 0.0) * rho / jnp.clip(N_r, 1.0) / (jnp.pi / 6.0 * constants.rho_water)),
-        0.0,
-    ) ** (1.0 / 3.0)
+    # Mean drop diameter D ~ (q_r * rho / N_r / (pi/6 * rho_water))^(1/3).
+    # Cube-root has unbounded derivative at zero — guard with safe_pow.
+    D_r_arg = (
+        jnp.clip(q_r, 0.0) * rho
+        / jnp.clip(N_r, 1.0)
+        / (jnp.pi / 6.0 * constants.rho_water)
+    )
+    D_r = safe_pow(D_r_arg, 1.0 / 3.0)
     breakup_frac = jax.nn.sigmoid(breakup_sharpness * (D_r - D_eq))
     dN_r_br = -dN_r_sc * breakup_frac
     return dN_r_sc, dN_r_br
@@ -172,4 +239,6 @@ def rain_evaporation(q_v, q_r, q_sat, evap_coeff):
     array : Evaporation rate [kg/kg/s].
     """
     subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
-    return evap_coeff * subsaturation * jnp.clip(q_r, 0.0) ** 0.525
+    # Marshall-Palmer ventilation factor q_r^0.525 — fractional power has
+    # an unbounded derivative at q_r=0; safe_pow handles the AD guard.
+    return evap_coeff * subsaturation * safe_pow(q_r, 0.525)
diff --git a/src/legoesm/atmosphere/physics/microphysics/kessler.py b/src/legoesm/atmosphere/physics/microphysics/kessler.py
index d1f310f..0b17223 100644
--- a/src/legoesm/atmosphere/physics/microphysics/kessler.py
+++ b/src/legoesm/atmosphere/physics/microphysics/kessler.py
@@ -20,6 +20,7 @@ import jax.numpy as jnp
 
 from legoesm import constants
 from legoesm.thermo import saturation_mixing_ratio
+from legoesm.atmosphere.physics.microphysics._warm_rain import safe_pow
 from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
 from legoesm.atmosphere.physics.microphysics.output import (
     HydrometeorState,
@@ -73,10 +74,19 @@ def kessler_microphysics(
     # Saturation mixing ratio
     q_sat = saturation_mixing_ratio(T, p_full)
 
-    # 1. Saturation adjustment — convert from increment [kg/kg] to tendency [kg/kg/s]
+    # 1. Saturation adjustment — convert from increment [kg/kg] to tendency [kg/kg/s].
+    # The evaporation branch (negative ``condensation``) is donor-clamped
+    # against the available ``q_c`` so a subsaturated clear-air column
+    # (q_v < q_sat, q_c = 0) cannot drive ``q_c`` below zero (Codex audit
+    # cycle 2: "subsaturated clear air can create negative cloud water").
+    # Same pattern as ``_warm_rain.saturation_adjustment``.
     excess = q_v - q_sat
     cond_frac = jax.nn.sigmoid(sharpness * excess)
     condensation = cond_frac * excess / dt  # [kg/kg/s]
+    q_c_avail = jnp.clip(q_c, 0.0, None)
+    condensation = jnp.maximum(
+        condensation, -q_c_avail / jnp.maximum(dt, 1e-10),
+    )
 
     dq_v_sat = -condensation
     dq_c_sat = condensation
@@ -88,12 +98,13 @@ def kessler_microphysics(
         q_c_updated - config.autoconversion_threshold, 0.0
     )
 
-    # 3. Accretion: cloud collected by rain
-    accretion = config.accretion_coeff * q_c * jnp.clip(q_r, 0.0) ** 0.875
+    # 3. Accretion: cloud collected by rain.  Fractional powers of q_r
+    # have unbounded derivative at q_r=0 — safe_pow handles the AD guard.
+    accretion = config.accretion_coeff * q_c * safe_pow(q_r, 0.875)
 
-    # 4. Evaporation of rain
+    # 4. Evaporation of rain (q_r^0.525).
     subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
-    evaporation = config.evaporation_coeff * subsaturation * jnp.clip(q_r, 0.0) ** 0.525
+    evaporation = config.evaporation_coeff * subsaturation * safe_pow(q_r, 0.525)
 
     # 5. Rain sedimentation
     rho_sfc = rho[:, -1:]
diff --git a/src/legoesm/atmosphere/physics/microphysics/morrison.py b/src/legoesm/atmosphere/physics/microphysics/morrison.py
index ccc01d8..5082695 100644
--- a/src/legoesm/atmosphere/physics/microphysics/morrison.py
+++ b/src/legoesm/atmosphere/physics/microphysics/morrison.py
@@ -27,6 +27,7 @@ from legoesm.atmosphere.physics.microphysics._warm_rain import (
     accretion,
     self_collection_breakup,
     rain_evaporation,
+    safe_pow,
 )
 from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
 from legoesm.atmosphere.physics.microphysics.output import (
@@ -71,7 +72,12 @@ def morrison_microphysics(
     N_c_eff = effective_Nc(N_c, config.Nc_0)
 
     # === WARM RAIN (shared Seifert-Beheng helpers) ===
-    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)
+    # Pass ``q_c`` so the evaporation branch (negative ``condensation``)
+    # is donor-clamped: evaporation cannot drive ``q_c`` below zero in
+    # subsaturated clear air.  See _warm_rain.saturation_adjustment.
+    condensation, q_sat = saturation_adjustment(
+        T, q_v, p_full, dt, sharpness, q_c=q_c,
+    )
     dq_c_au, dN_r_au, x_c = autoconversion_sb(
         q_c, N_c_eff, rho, config.k_au, config.x_star, sharpness,
     )
@@ -98,7 +104,7 @@ def morrison_microphysics(
         config.dep_coeff
         * jnp.maximum(S_i, 0.0)
         * jnp.clip(q_i, 0.0)
-        * jnp.clip(N_i, 0.0) ** (1.0 / 3.0)
+        * safe_pow(N_i, 1.0 / 3.0)
         * f_ice
     )
 
@@ -127,13 +133,58 @@ def morrison_microphysics(
         jnp.clip(q_s, 0.0) / jnp.maximum(dt, 1e-10),
     )
 
+    # === DONOR CLAMP for q_c sinks ===
+    # Scale q_c-consuming processes (autoconversion, accretion, Bergeron,
+    # riming) by a common factor so the total loss per timestep does not
+    # exceed available q_c.  Without this clamp, default rates at dt = 1200s
+    # in a mixed-phase column drive q_c negative on a single explicit step
+    # (bergeron alone gives bergeron_rate * q_c * dt = 1.2 * q_c).  Mass is
+    # conserved because each process's matching source term in dq_r/dq_i/dq_s
+    # gets the same scale factor (the rates appear once as sinks in dq_c and
+    # once as sources elsewhere, so a uniform rescale preserves the budget).
+    #
+    # Include the *evaporation* branch of saturation adjustment (negative
+    # ``condensation``) in the q_c sink budget — otherwise a subsaturated
+    # clear-air column with q_c just above zero can lose more q_c to
+    # evaporation + accretion combined than is available, going negative
+    # (Codex audit cycle 2: "subsaturated clear air can create negative
+    # cloud water").  ``saturation_adjustment`` has already donor-clamped
+    # ``-condensation`` against ``q_c`` in isolation; including it here
+    # makes the joint budget consistent when other q_c sinks are active.
+    cond_evap_sink = jnp.maximum(-condensation, 0.0)
+    qc_sink_total = dq_c_au + dq_c_ac + bergeron + riming_i + riming_s + cond_evap_sink
+    qc_avail = jnp.clip(q_c, 0.0)
+    qc_scale = jnp.minimum(
+        1.0,
+        qc_avail / jnp.maximum(qc_sink_total * jnp.maximum(dt, 1e-10), 1e-30),
+    )
+    dq_c_au = dq_c_au * qc_scale
+    dq_c_ac = dq_c_ac * qc_scale
+    bergeron = bergeron * qc_scale
+    riming_i = riming_i * qc_scale
+    riming_s = riming_s * qc_scale
+    # Scale the evaporation branch by the same factor: when condensation
+    # is negative (evaporation), reduce its magnitude proportionally so
+    # q_c doesn't go negative.  When condensation is positive
+    # (saturation adjustment from supersaturation), the scaling does
+    # nothing because ``cond_evap_sink = 0``.
+    condensation = jnp.where(
+        condensation < 0.0, condensation * qc_scale, condensation,
+    )
+    # Number tendency for autoconverted droplets must scale identically.
+    dN_r_au = dN_r_au * qc_scale
+
     # === SEDIMENTATION ===
+    # Marshall-Palmer fall speeds V_t = a_v * (q * rho / rho_sfc)^b_v use
+    # fractional exponents (b_v_r=0.5, b_v_i=0.25, b_v_s=0.3); guard the
+    # AD path with safe_pow so cold-start columns (q=0) don't NaN gradients.
     rho_sfc = rho[:, -1:]
-    V_t_r = config.a_v_r * (jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_r
+    rho_ratio = rho / jnp.clip(rho_sfc, 0.1)
+    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
     V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
-    V_t_i = config.a_v_i * (jnp.clip(q_i, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_i
+    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
     V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
-    V_t_s = config.a_v_s * (jnp.clip(q_s, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_s
+    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
     V_t_s = jnp.clip(V_t_s, 0.0, 5.0)
 
     sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
@@ -150,6 +201,14 @@ def morrison_microphysics(
         L_v * condensation / c_pd
         - L_v * evaporation / c_pd
         + L_s * dq_i_dep / c_pd
+        # Cloud water → ice/snow freezing releases latent heat of fusion
+        # (~333 kJ/kg).  Bergeron is liquid → ice via the WBF mechanism,
+        # riming is supercooled-droplet capture by ice/snow.  Both are
+        # phase changes that release L_f; the moist-enthalpy invariant
+        # ``h = c_pd T + L_v q_v - L_f q_ice`` requires this term for
+        # column conservation.  Magnitude estimate: ~2 K/day at default
+        # rates in mixed-phase clouds.
+        + L_f * (bergeron + riming_i + riming_s) / c_pd
         - L_f * (melt_ice + melt_snow) / c_pd
     )
 
diff --git a/src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py b/src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py
index 720022e..8f02697 100644
--- a/src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py
+++ b/src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py
@@ -32,6 +32,7 @@ from legoesm.atmosphere.physics.microphysics._warm_rain import (
     accretion,
     self_collection_breakup,
     rain_evaporation,
+    safe_pow,
 )
 
 
@@ -66,7 +67,11 @@ def seifert_beheng_microphysics(
 
     N_c_eff = effective_Nc(N_c, config.Nc_0)
 
-    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)
+    # Pass ``q_c`` so the evaporation branch (negative ``condensation``) is
+    # donor-clamped — see _warm_rain.saturation_adjustment.
+    condensation, q_sat = saturation_adjustment(
+        T, q_v, p_full, dt, sharpness, q_c=q_c,
+    )
 
     # 1. Autoconversion (mass-dependent)
     dq_c_au, dN_r_au, x_c = autoconversion_sb(
@@ -84,9 +89,12 @@ def seifert_beheng_microphysics(
     # 5. Rain evaporation
     evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff)
 
-    # 6. Sedimentation
+    # 6. Sedimentation — Marshall-Palmer fall speed (q_r * rho/rho_sfc)^b_v_r
+    # has fractional exponent (b_v_r=0.5); guard the AD path with safe_pow.
     rho_sfc = rho[:, -1:]
-    V_t_r = config.a_v_r * (jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_r
+    V_t_r = config.a_v_r * safe_pow(
+        jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, 0.1), config.b_v_r,
+    )
     V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
     sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
 
diff --git a/src/legoesm/atmosphere/physics/microphysics/thompson.py b/src/legoesm/atmosphere/physics/microphysics/thompson.py
index 6b09284..edcb678 100644
--- a/src/legoesm/atmosphere/physics/microphysics/thompson.py
+++ b/src/legoesm/atmosphere/physics/microphysics/thompson.py
@@ -27,6 +27,7 @@ from legoesm.atmosphere.physics.microphysics._warm_rain import (
     accretion,
     self_collection_breakup,
     rain_evaporation,
+    safe_pow,
 )
 from legoesm.atmosphere.physics.microphysics.config import ThompsonConfig
 from legoesm.atmosphere.physics.microphysics.output import (
@@ -77,8 +78,13 @@ def thompson_microphysics(
     N_c_eff = effective_Nc(N_c, config.Nc_0)
 
     # === WARM RAIN ===
-    # Saturation adjustment — convert increment [kg/kg] to tendency [kg/kg/s]
-    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)
+    # Saturation adjustment — convert increment [kg/kg] to tendency [kg/kg/s].
+    # Pass ``q_c`` so the evaporation branch (negative ``condensation``) is
+    # donor-clamped: evaporation cannot drive ``q_c`` below zero in
+    # subsaturated clear air.  See _warm_rain.saturation_adjustment.
+    condensation, q_sat = saturation_adjustment(
+        T, q_v, p_full, dt, sharpness, q_c=q_c,
+    )
 
     # Gamma distribution corrections
     gamma_c = _gamma_ratio(config.mu_c)
@@ -119,7 +125,7 @@ def thompson_microphysics(
         config.dep_coeff
         * jnp.maximum(S_i, 0.0)
         * jnp.clip(q_i, 0.0)
-        * jnp.clip(N_i, 0.0) ** (1.0 / 3.0)
+        * safe_pow(N_i, 1.0 / 3.0)
         * f_ice
     )
 
@@ -138,27 +144,87 @@ def thompson_microphysics(
     # Aggregation
     aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice
 
-    # Melting
+    # Melting (clamp to available mass so an explicit Euler step cannot
+    # drive q_i / q_s / q_g negative — same pattern Morrison already uses).
     melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
-    melt_ice = config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac
-    melt_snow = config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac
+    dt_safe = jnp.maximum(dt, 1e-10)
+    melt_ice = jnp.minimum(
+        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
+        jnp.clip(q_i, 0.0) / dt_safe,
+    )
+    melt_snow = jnp.minimum(
+        config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
+        jnp.clip(q_s, 0.0) / dt_safe,
+    )
 
     # === GRAUPEL (Thompson extension) ===
+    # Split the rime → graupel conversion by donor: the fraction of
+    # rime_to_graupel that comes from q_i scales with riming_i, and
+    # the fraction from q_s scales with riming_s.  This avoids a
+    # mass-leak corner case where ``q_i = 0`` and ``riming_s > 0``:
+    # the previous form set ``rime_to_graupel ∝ total_riming``, then
+    # subtracted the FULL value from ``q_i`` (driving it negative)
+    # while only subtracting half from ``q_s`` and adding 150% to
+    # ``q_g`` — a non-conservative split that depended on the
+    # ad-hoc 1.0 / 0.5 / 1.5 coefficients.  (Codex audit cycle 2:
+    # "Thompson graupel conversion can draw from the wrong donor".)
     graupel_frac = jax.nn.sigmoid(
         config.graupel_sharpness * (total_riming - config.rime_to_graupel_threshold)
     )
-    rime_to_graupel = config.rime_to_graupel_rate * total_riming * graupel_frac
-    melt_graupel = config.melt_rate * jnp.clip(q_g, 0.0) * melt_frac
+    rime_to_graupel_from_i = config.rime_to_graupel_rate * riming_i * graupel_frac
+    rime_to_graupel_from_s = config.rime_to_graupel_rate * riming_s * graupel_frac
+    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
+    melt_graupel = jnp.minimum(
+        config.melt_rate * jnp.clip(q_g, 0.0) * melt_frac,
+        jnp.clip(q_g, 0.0) / dt_safe,
+    )
+
+    # === DONOR CLAMP for q_c sinks (see morrison.py for rationale) ===
+    # Include the evaporation branch of saturation_adjustment (negative
+    # condensation) in the q_c sink budget so subsaturated clear-air
+    # columns cannot drive q_c negative (Codex audit cycle 2).
+    cond_evap_sink = jnp.maximum(-condensation, 0.0)
+    qc_sink_total = (
+        dq_c_au + dq_c_ac + bergeron + riming_i + riming_s + cond_evap_sink
+    )
+    qc_avail = jnp.clip(q_c, 0.0)
+    qc_scale = jnp.minimum(
+        1.0,
+        qc_avail / jnp.maximum(qc_sink_total * dt_safe, 1e-30),
+    )
+    dq_c_au = dq_c_au * qc_scale
+    dq_c_ac = dq_c_ac * qc_scale
+    bergeron = bergeron * qc_scale
+    riming_i = riming_i * qc_scale
+    riming_s = riming_s * qc_scale
+    total_riming = riming_i + riming_s
+    # Each rime-to-graupel donor scales with its parent riming term —
+    # which has already been scaled by qc_scale above.  Re-scaling
+    # ``rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s``
+    # by ``qc_scale`` once preserves both per-donor proportionality and
+    # mass conservation.
+    rime_to_graupel_from_i = rime_to_graupel_from_i * qc_scale
+    rime_to_graupel_from_s = rime_to_graupel_from_s * qc_scale
+    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
+    dN_r_au = dN_r_au * qc_scale
+    # Scale negative-condensation (evaporation) branch by the same
+    # factor; positive condensation is unaffected (cond_evap_sink = 0).
+    condensation = jnp.where(
+        condensation < 0.0, condensation * qc_scale, condensation,
+    )
 
     # === SEDIMENTATION ===
+    # Marshall-Palmer fall speeds use fractional exponents (b_v_x in
+    # [0.25, 0.5]); guard the AD path with safe_pow.
     rho_sfc = rho[:, -1:]
-    V_t_r = config.a_v_r * (jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_r
+    rho_ratio = rho / jnp.clip(rho_sfc, 0.1)
+    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
     V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
-    V_t_i = config.a_v_i * (jnp.clip(q_i, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_i
+    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
     V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
-    V_t_s = config.a_v_s * (jnp.clip(q_s, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_s
+    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
     V_t_s = jnp.clip(V_t_s, 0.0, 5.0)
-    V_t_g = config.a_v_g * (jnp.clip(q_g, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_g
+    V_t_g = config.a_v_g * safe_pow(jnp.clip(q_g, 0.0) * rho_ratio, config.b_v_g)
     V_t_g = jnp.clip(V_t_g, 0.0, 30.0)
 
     sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
@@ -175,16 +241,31 @@ def thompson_microphysics(
         L_v * condensation / c_pd
         - L_v * evaporation / c_pd
         + L_s * dq_i_dep / c_pd
+        # Cloud water → ice/snow freezing releases L_f (Bergeron, riming).
+        # See morrison.py for the moist-enthalpy rationale; Thompson
+        # mirrors Morrison's ice-phase latent heating.
+        + L_f * (bergeron + riming_i + riming_s) / c_pd
         - L_f * (melt_ice + melt_snow + melt_graupel) / c_pd
     )
 
     # === COMBINE TENDENCIES ===
+    # Conservation: each rime-to-graupel donor leaves its parent
+    # species and arrives in q_g.  The total mass moved is
+    # ``rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s``.
+    # An earlier form used 1.0 / 0.5 / 1.5 splits on ``rime_to_graupel``
+    # which (a) drove ``q_i`` negative when only snow was being rimed
+    # (``q_i = 0`` but ``riming_s > 0``), and (b) created mass
+    # apparently from nothing in the same regime.  See the
+    # ``=== GRAUPEL ===`` block above for the donor-split rationale.
     dq_v_dt = -condensation + evaporation - dq_i_dep
     dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
     dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + melt_graupel + sed_r
-    dq_i_dt = dq_i_dep + bergeron + riming_i - aggregation - melt_ice - rime_to_graupel + sed_i
-    dq_s_dt = aggregation + riming_s - melt_snow - rime_to_graupel * 0.5 + sed_s
-    dq_g_dt = rime_to_graupel * 1.5 - melt_graupel + sed_g
+    dq_i_dt = (
+        dq_i_dep + bergeron + riming_i - aggregation - melt_ice
+        - rime_to_graupel_from_i + sed_i
+    )
+    dq_s_dt = aggregation + riming_s - melt_snow - rime_to_graupel_from_s + sed_s
+    dq_g_dt = rime_to_graupel - melt_graupel + sed_g
 
     dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
     dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
diff --git a/tests/unit/test_physics_microphysics.py b/tests/unit/test_physics_microphysics.py
index 3522cba..7d7894c 100644
--- a/tests/unit/test_physics_microphysics.py
+++ b/tests/unit/test_physics_microphysics.py
@@ -232,3 +232,470 @@ def test_heating_rate_bounded(scheme):
     assert max_hr < 10.0, (
         f"{scheme}: |dT_dt| = {max_hr:.2e} K/s exceeds 10 K/s"
     )
+
+
+# ============================================================================
+# Donor clamps: Morrison/Thompson q_c, q_i sinks must not over-extract
+# ============================================================================
+
+@pytest.mark.parametrize(
+    "scheme,call",
+    [("morrison", morrison_microphysics), ("thompson", thompson_microphysics)],
+)
+def test_qc_does_not_go_negative_at_long_dt(scheme, call):
+    """An explicit Euler step must not drive q_c below zero in mixed-phase.
+
+    Default config rates with dt=1200s drive Bergeron+riming+evaporation
+    sinks past 100% of q_c per step:
+        bergeron_rate = 1e-3 /s -> bergeron * dt = 1.2 * q_c
+    The fix proportionally scales all q_c sink processes so the total
+    loss per step is bounded by q_c (mass conservation preserved by
+    scaling source terms in dq_r/dq_i/dq_s by the same factor).
+
+    Saturation is set to RH=1.0 so saturation_adjustment does not
+    additionally evaporate cloud water; layer thickness is large
+    (10 km) to keep sedimentation Courant well below 1.
+    """
+    cfg_class = MorrisonConfig if scheme == "morrison" else ThompsonConfig
+    ncol, nlev = 1, 5
+    T = jnp.full((ncol, nlev), 250.0)  # mixed-phase regime
+    p_full = jnp.full((ncol, nlev), 5e4)
+    p_half = jnp.broadcast_to(
+        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
+    )
+    rho = p_full / (constants.R_d * T)
+    dz = jnp.full((ncol, nlev), 10_000.0)
+    q_c = jnp.full((ncol, nlev), 1e-3)
+    q_i = jnp.full((ncol, nlev), 1e-4)
+    q_s = jnp.full((ncol, nlev), 1e-4)
+    hydro = HydrometeorState(
+        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
+        q_g=jnp.zeros_like(q_c),
+        N_c=1e8 * jnp.ones_like(q_c),
+        N_r=jnp.zeros_like(q_c),
+        N_i=1e4 * jnp.ones_like(q_c),
+    )
+    # RH = 1: keep condensation ~ 0 so this test isolates the q_c sinks
+    # (autoconv / accretion / Bergeron / riming) from saturation-driven
+    # cloud evaporation.
+    q_v = 1.0 * saturation_mixing_ratio(T, p_full)
+    dt = 1200.0
+    out = call(T, q_v, hydro, p_full, p_half, rho, dz, dt, cfg_class())
+    q_c_after = q_c + out.dq_c_dt * dt
+    min_qc = float(jnp.min(q_c_after))
+    # Allow ~1e-9 floor for floating-point round-off in scaled tendencies.
+    assert min_qc >= -1e-9, (
+        f"{scheme}: q_c went negative ({min_qc:.3e}) after one explicit "
+        f"step at dt={dt}s — Bergeron + riming + autoconv + accretion "
+        "sinks combined exceeded available q_c without a donor clamp."
+    )
+
+
+def test_thompson_qi_does_not_go_negative_warm():
+    """Thompson melt processes must not over-extract q_i above freezing.
+
+    Morrison clamps melt_ice/melt_snow with ``min(rate*q*frac, q/dt)``;
+    Thompson did not, so at T=280K (above freezing) and dt=1200s with
+    default rates, q_i was driven below zero.  After fix, melt is clamped
+    to available mass.
+
+    Layer thickness is 10 km so sedimentation Courant
+    ``V_t * dt / dz <= 5*1200/10000 = 0.6`` stays below 1.
+    """
+    ncol, nlev = 1, 5
+    T = jnp.full((ncol, nlev), 280.0)  # above freezing
+    p_full = jnp.full((ncol, nlev), 5e4)
+    p_half = jnp.broadcast_to(
+        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
+    )
+    rho = p_full / (constants.R_d * T)
+    dz = jnp.full((ncol, nlev), 10_000.0)
+    q_c = jnp.full((ncol, nlev), 1e-3)
+    q_i = jnp.full((ncol, nlev), 1e-4)
+    q_s = jnp.full((ncol, nlev), 1e-4)
+    hydro = HydrometeorState(
+        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
+        q_g=jnp.zeros_like(q_c),
+        N_c=1e8 * jnp.ones_like(q_c),
+        N_r=jnp.zeros_like(q_c),
+        N_i=1e4 * jnp.ones_like(q_c),
+    )
+    q_v = 0.8 * saturation_mixing_ratio(T, p_full)
+    dt = 1200.0
+    out = thompson_microphysics(
+        T, q_v, hydro, p_full, p_half, rho, dz, dt, ThompsonConfig(),
+    )
+    q_i_after = q_i + out.dq_i_dt * dt
+    # Skip the top level: with q_i uniform throughout the column the top
+    # level always loses sedimentation flux without compensating inflow,
+    # which is a sedimentation-CFL concern rather than a melt-clamp one.
+    # The melt-clamp bug shows up uniformly in interior levels.
+    min_qi_interior = float(jnp.min(q_i_after[:, 1:]))
+    assert min_qi_interior >= -1e-9, (
+        f"Thompson: interior q_i went negative ({min_qi_interior:.3e}) at "
+        f"T=280 K with dt=1200s — melt rate × q_i × melt_frac × dt exceeded "
+        "q_i without a donor clamp."
+    )
+
+
+@pytest.mark.parametrize(
+    "scheme,call",
+    [
+        ("kessler", kessler_microphysics),
+        ("seifert_beheng", seifert_beheng_microphysics),
+        ("morrison", morrison_microphysics),
+        ("thompson", thompson_microphysics),
+    ],
+)
+def test_subsaturated_clear_air_does_not_create_negative_qc(scheme, call):
+    """Audit cycle 2 (Codex): the smooth saturation adjustment
+    ``condensation = sigmoid(s · excess) · excess / dt`` is *signed* —
+    negative for ``q_v < q_sat`` (evaporation).  Adding this directly
+    to ``dq_c_dt`` in subsaturated clear air (``q_c = 0``) produces a
+    spurious negative cloud-water tendency that drives ``q_c`` below
+    zero in the explicit Euler step.
+
+    The fix is to donor-clamp the evaporation branch against the
+    available cloud water — implemented inside
+    ``saturation_adjustment`` (and inline for Kessler) and joined to
+    the per-scheme donor clamp on q_c sinks.
+
+    Test column: 95 % RH at T=280 K, q_c = 0.  In all four schemes
+    the buggy form produced ``q_c_after ≈ -3e-4 kg/kg`` after a
+    1200-s timestep.
+    """
+    if scheme in ("kessler", "seifert_beheng"):
+        config_cls = {
+            "kessler": KesslerConfig,
+            "seifert_beheng": SeifertBehengConfig,
+        }[scheme]
+    elif scheme == "morrison":
+        config_cls = MorrisonConfig
+    else:
+        config_cls = ThompsonConfig
+
+    ncol, nlev = 1, 5
+    T = jnp.full((ncol, nlev), 280.0)
+    p_full = jnp.full((ncol, nlev), 5e4)
+    p_half = jnp.broadcast_to(
+        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
+    )
+    rho = p_full / (constants.R_d * T)
+    dz = jnp.full((ncol, nlev), 1000.0)
+    q_sat = saturation_mixing_ratio(T, p_full)
+    q_v = 0.95 * q_sat                           # subsaturated
+    q_c = jnp.zeros((ncol, nlev))                # NO cloud water
+    hydro = HydrometeorState(
+        q_c=q_c, q_r=jnp.zeros_like(q_c),
+        q_i=jnp.zeros_like(q_c), q_s=jnp.zeros_like(q_c),
+        q_g=jnp.zeros_like(q_c),
+        N_c=1e8 * jnp.ones_like(q_c),
+        N_r=jnp.zeros_like(q_c),
+        N_i=jnp.zeros_like(q_c),
+    )
+    dt = 1200.0
+    out = call(T, q_v, hydro, p_full, p_half, rho, dz, dt, config_cls())
+    q_c_after = q_c + out.dq_c_dt * dt
+    min_q_c = float(jnp.min(q_c_after))
+    # Tolerance allows for f64 → f32 promotion noise; the buggy form
+    # produces q_c_after ≈ -3e-4, well above this threshold.
+    assert min_q_c >= -1e-9, (
+        f"{scheme}: q_c_after went negative ({min_q_c:.3e}) in a "
+        "subsaturated clear-air column (q_v = 95% RH, q_c = 0).  "
+        "The signed saturation adjustment ``condensation = sigmoid(s · "
+        "(q_v - q_sat)) · (q_v - q_sat) / dt`` is negative there, and "
+        "adding it to ``dq_c_dt`` over an explicit Euler step drives "
+        "q_c below zero without a donor clamp on the evaporation "
+        "branch.  Audit cycle 2 Codex finding 'subsaturated clear air "
+        "can create negative cloud water' has regressed."
+    )
+
+
+def test_thompson_rime_to_graupel_donor_split():
+    """Audit cycle 2 (Codex): the Thompson rime-to-graupel conversion
+    must subtract from the SOURCE species (q_i for ``riming_i``,
+    q_s for ``riming_s``), not split via fixed 1.0 / 0.5 / 1.5
+    coefficients on the total.
+
+    Pathological column: ``q_c > 0`` (cloud water source for riming),
+    ``q_i = 0`` (no ice to be rimed), ``q_s > 0`` (snow that gets
+    rimed by cloud water), ``T < T_freeze`` (active ice phase).  In
+    this column ``riming_i = 0`` (no q_i to rime) and ``riming_s > 0``
+    (q_c × q_s × f_ice).  Pre-fix:
+        rime_to_graupel = rate * (riming_i + riming_s) > 0
+        dq_i_dt -= rime_to_graupel              # full subtraction!
+        dq_s_dt -= 0.5 * rime_to_graupel
+        dq_g_dt += 1.5 * rime_to_graupel
+    drives ``q_i`` negative and creates 1.5× extra mass — both
+    conservation violations.  Post-fix the donor split scales by
+    ``riming_i / riming_s`` and only the actually-rimed species is
+    drained.  Total mass moved (``rime_to_graupel_from_i +
+    rime_to_graupel_from_s``) goes 1:1 to graupel.
+    """
+    ncol, nlev = 1, 5
+    T = jnp.full((ncol, nlev), 250.0)  # below freezing — ice active
+    p_full = jnp.full((ncol, nlev), 5e4)
+    p_half = jnp.broadcast_to(
+        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
+    )
+    rho = p_full / (constants.R_d * T)
+    dz = jnp.full((ncol, nlev), 10_000.0)
+    # The pathological mix: cloud water + snow, NO ice.
+    q_c = jnp.full((ncol, nlev), 5e-3)
+    q_i = jnp.zeros((ncol, nlev))                # zero ice — would be drained negative pre-fix
+    q_s = jnp.full((ncol, nlev), 1e-3)            # snow gets rimed
+    hydro = HydrometeorState(
+        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
+        q_g=jnp.zeros_like(q_c),
+        N_c=1e8 * jnp.ones_like(q_c),
+        N_r=jnp.zeros_like(q_c),
+        N_i=jnp.zeros_like(q_c),
+    )
+    q_v = 0.5 * saturation_mixing_ratio(T, p_full)
+    dt = 1200.0
+    out = thompson_microphysics(
+        T, q_v, hydro, p_full, p_half, rho, dz, dt, ThompsonConfig(),
+    )
+    q_i_after = q_i + out.dq_i_dt * dt
+    # q_i must remain non-negative (donor split: only riming_i drains q_i).
+    min_qi = float(jnp.min(q_i_after))
+    assert min_qi >= -1e-9, (
+        f"Thompson: q_i went negative ({min_qi:.3e}) when starting at "
+        "q_i=0 with q_s>0 and active riming.  Pre-fix the rime_to_graupel "
+        "subtracted the full conversion from q_i regardless of which "
+        "species was actually rimed.  Audit cycle 2 Codex finding "
+        "'Thompson graupel conversion can draw from the wrong donor' "
+        "has regressed."
+    )
+    # Mass conservation: ∑ dq_i + dq_s + dq_g (pure ice phase) should
+    # equal sed_i + sed_s + sed_g (sedimentation only escapes the column);
+    # internal phase changes cancel in the sum.  We can verify rime->graupel
+    # specifically by checking that any q_g gain matches a q_s loss.
+    dq_g_total = float(jnp.sum(out.dq_g_dt))
+    # Without melting (T well below freeze), all q_g must come from
+    # rime_to_graupel.  q_g production must be matched by q_s loss
+    # (donor split: only riming_s active here).
+    assert dq_g_total >= 0.0, (
+        f"Thompson: dq_g_dt total ({dq_g_total:.3e}) is negative "
+        "without graupel sedimentation source — graupel mass conservation "
+        "violation."
+    )
+
+
+# ============================================================================
+# Morrison/Thompson moist-enthalpy conservation (latent heat of fusion)
+# ============================================================================
+
+@pytest.mark.parametrize(
+    "scheme,call",
+    [("morrison", morrison_microphysics), ("thompson", thompson_microphysics)],
+)
+def test_freezing_releases_latent_heat_of_fusion(scheme, call):
+    """Bergeron + riming (cloud water → ice/snow) must release ``L_f``.
+
+    Moist enthalpy ``h = c_pd T + L_v q_v - L_f * q_ice`` is conserved
+    by phase transitions in a closed column, so per-level the residual
+        c_pd * dT_dt + L_v * dq_v_dt - L_f * (dq_i + dq_s + dq_g)
+    is the divergence of sedimentation flux (surface boundary effect),
+    not a phase-change heating/cooling.
+
+    Pre-fix the dT_dt assembly omitted the ``+ L_f * (bergeron +
+    riming_i + riming_s)`` term, so in a mixed-phase column where
+    ~6e-4 kg/kg of cloud water freezes per step the column residual
+    is ~16 K/day too cold (L_f * freezing / c_pd = 333e3 * 6.6e-4 /
+    1004 / 1200s = 1.83e-4 K/s = 15.8 K/day at default rates).
+
+    Post-fix the residual collapses to the small contribution from
+    sedimentation flux divergence at the interior of the column.
+    """
+    cfg_class = MorrisonConfig if scheme == "morrison" else ThompsonConfig
+    ncol, nlev = 1, 5
+    # Mid mixed-phase regime, sub-saturated wrt liquid (suppresses
+    # condensation/evaporation as the dominant balance) so the
+    # cloud→ice freezing branches dominate the residual.  Short dt
+    # keeps sedimentation flux divergence small.
+    T = jnp.full((ncol, nlev), 260.0)
+    p_full = jnp.full((ncol, nlev), 5e4)
+    p_half = jnp.broadcast_to(
+        jnp.linspace(4e4, 6e4, nlev + 1)[None, :], (ncol, nlev + 1),
+    )
+    rho = p_full / (constants.R_d * T)
+    dz = jnp.full((ncol, nlev), 10_000.0)
+    q_c = jnp.full((ncol, nlev), 1e-3)
+    q_i = jnp.full((ncol, nlev), 5e-4)
+    q_s = jnp.full((ncol, nlev), 5e-4)
+    hydro = HydrometeorState(
+        q_c=q_c, q_r=jnp.zeros_like(q_c), q_i=q_i, q_s=q_s,
+        q_g=jnp.zeros_like(q_c),
+        N_c=1e8 * jnp.ones_like(q_c),
+        N_r=jnp.zeros_like(q_c),
+        N_i=1e4 * jnp.ones_like(q_c),
+    )
+    q_v = 0.85 * saturation_mixing_ratio(T, p_full)
+    dt = 60.0
+    out = call(T, q_v, hydro, p_full, p_half, rho, dz, dt, cfg_class())
+
+    # Inspect interior level (k=2) where sedimentation flux divergence
+    # is approximately zero (uniform q_i, q_s ⇒ flux_in ≈ flux_out).
+    k = 2
+    h_residual = float(
+        constants.c_pd * out.dT_dt[0, k]
+        + constants.L_v * out.dq_v_dt[0, k]
+        - constants.L_f * (out.dq_i_dt[0, k] + out.dq_s_dt[0, k] + out.dq_g_dt[0, k])
+    )
+    # Magnitude scale of L_f-driven heating (lower bound)
+    freezing_scale = float(
+        constants.L_f * jnp.abs(out.dq_i_dt[0, k] + out.dq_s_dt[0, k] + out.dq_g_dt[0, k])
+    )
+    # Pre-fix: |h_residual| ≈ L_f * freezing rate (entire term missing);
+    # post-fix: |h_residual| << L_f * freezing rate (just sedimentation div).
+    assert abs(h_residual) < 0.2 * max(freezing_scale, 1e-10), (
+        f"{scheme}: per-level moist-enthalpy residual at k={k} = "
+        f"{h_residual:.3e} W/m^3-equivalent, freezing scale = "
+        f"{freezing_scale:.3e}; ratio = {abs(h_residual)/max(freezing_scale,1e-30):.3f}.  "
+        "dT_dt is missing +L_f * (bergeron + riming_i + riming_s) / c_pd "
+        "for the cloud-water → ice/snow freezing branches."
+    )
+
+
+# ============================================================================
+# Sundqvist column water budget (closes against surface precipitation)
+# ============================================================================
+
+def test_sundqvist_column_water_budget_closes():
+    """Sundqvist is a diagnostic scheme — rain falls instantly.
+
+    Column total water tendency must balance surface precipitation:
+        int (dq_v + dq_c + dq_r) dp/g  +  precipitation  ~  0
+    """
+    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(
+        T_sfc=290.0, q_c_val=1e-3,
+    )
+    # Saturate the column so condensation (and therefore rain) actually fires.
+    q_sat = saturation_mixing_ratio(T, p_full)
+    q_v = q_sat
+    out = sundqvist_microphysics(
+        T, q_v, hydro, p_full, p_half, rho, dz, 300.0, SundqvistConfig(),
+    )
+    dp = p_half[:, 1:] - p_half[:, :-1]
+    col_water_tend = jnp.sum(
+        (out.dq_v_dt + out.dq_c_dt + out.dq_r_dt) * dp, axis=1,
+    ) / constants.g
+    imbalance = col_water_tend + out.precipitation
+    # Allow numerics; pre-fix imbalance was ~+9e-3 (= +precipitation).
+    max_imbalance = float(jnp.max(jnp.abs(imbalance)))
+    max_precip = float(jnp.max(out.precipitation))
+    assert max_imbalance < 1e-6 * max(max_precip, 1.0), (
+        f"Sundqvist water budget unclosed: imbalance = {max_imbalance:.3e}, "
+        f"precip scale = {max_precip:.3e}"
+    )
+
+
+def test_sundqvist_drains_incoming_qr_to_precipitation():
+    """If q_r is non-zero on input (e.g., warm-started or after a prior
+    Kessler step), Sundqvist's diagnostic-rain semantics drain it to the
+    surface in one step and account for the drained mass in
+    ``precipitation``.
+
+    Previously ``dq_r_dt = 0`` left any pre-existing q_r frozen in the
+    column — column water grew indefinitely and surface precipitation
+    under-reported the actual mass leaving.  This test would catch that
+    regression: assert q_r is zero after one step AND the drained mass
+    appears in the precipitation flux.
+    """
+    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(
+        T_sfc=290.0, q_c_val=0.0,
+    )
+    # Sub-saturated column with NO q_c so condensation/autoconversion
+    # don't generate fresh rain — this isolates the q_r-drain channel.
+    q_v_sub = 0.5 * saturation_mixing_ratio(T, p_full)
+    # Inject q_r at a few mid-column levels so dq_r_drain is nontrivial.
+    q_r_in = jnp.zeros_like(hydro.q_r).at[..., 8:12].set(5e-4)
+    hydro_with_qr = hydro._replace(q_r=q_r_in)
+
+    dt = 300.0
+    out = sundqvist_microphysics(
+        T, q_v_sub, hydro_with_qr, p_full, p_half, rho, dz, dt, SundqvistConfig(),
+    )
+    q_r_after = q_r_in + out.dq_r_dt * dt
+    # q_r must drain to zero in one step (modulo float round-off).
+    max_qr_residual = float(jnp.max(jnp.abs(q_r_after)))
+    assert max_qr_residual < 1e-12, (
+        f"Sundqvist failed to drain incoming q_r: max(|q_r_after|) = "
+        f"{max_qr_residual:.3e}; expected ~0 (diagnostic-rain semantics)."
+    )
+    # Drained mass [kg/m^2/s] must appear in precipitation.
+    dp = p_half[:, 1:] - p_half[:, :-1]
+    expected_drain_flux = jnp.sum(q_r_in * dp, axis=1) / (constants.g * dt)
+    # No autoconversion source here (q_c=0, sub-saturated) so precip
+    # equals the drain flux to a few percent (allowing for the smooth
+    # condensation sigmoid producing a tiny residual).
+    rel_err = float(jnp.max(
+        jnp.abs(out.precipitation - expected_drain_flux) /
+        jnp.maximum(expected_drain_flux, 1e-30),
+    ))
+    assert rel_err < 0.05, (
+        f"Sundqvist precip = {[float(x) for x in out.precipitation]}; "
+        f"expected drain flux = {[float(x) for x in expected_drain_flux]} "
+        "(no fresh autoconversion in this dry-c column)."
+    )
+    # Column water budget must still close with q_r in the loop.
+    col_water_tend = jnp.sum(
+        (out.dq_v_dt + out.dq_c_dt + out.dq_r_dt) * dp, axis=1,
+    ) / constants.g
+    imbalance = col_water_tend + out.precipitation
+    max_imbalance = float(jnp.max(jnp.abs(imbalance)))
+    max_precip = float(jnp.max(out.precipitation))
+    assert max_imbalance < 1e-6 * max(max_precip, 1.0), (
+        f"Sundqvist q_r-drain water budget unclosed: imbalance = "
+        f"{max_imbalance:.3e}, precip scale = {max_precip:.3e}"
+    )
+
+
+# ============================================================================
+# 4l  Kessler / warm-rain accretion & evaporation: gradient finite at q_r=0
+# ============================================================================
+
+def test_kessler_grad_finite_at_zero_qr():
+    """Marshall-Palmer fractional powers q_r**0.875 (accretion) and
+    q_r**0.525 (evaporation) have unbounded analytic derivative at q_r=0.
+
+    Without an AD guard, ``jnp.clip(q_r, 0)**0.525`` returns +inf for the
+    gradient at q_r=0 (and NaN at q_r<0).  This test imports the kessler
+    leaf and the shared ``rain_evaporation`` and asserts that the gradient
+    of a column scalar w.r.t. q_r is finite when q_r is exactly zero.
+    """
+    from legoesm.atmosphere.physics.microphysics._warm_rain import rain_evaporation
+
+    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(q_c_val=1e-4)
+    q_r_zero = jnp.zeros_like(hydro.q_r)
+    hydro_zero = hydro._replace(q_r=q_r_zero)
+
+    def kessler_objective(q_r):
+        h = hydro_zero._replace(q_r=q_r)
+        out = kessler_microphysics(
+            T, q_v, h, p_full, p_half, rho, dz, 300.0, KesslerConfig(),
+        )
+        return jnp.sum(out.dq_r_dt + out.dq_v_dt + out.dT_dt)
+
+    g_kessler = jax.grad(kessler_objective)(q_r_zero)
+    assert jnp.all(jnp.isfinite(g_kessler)), (
+        "Kessler: grad w.r.t. q_r at q_r=0 contains NaN/Inf — "
+        "fractional-power AD guard regressed."
+    )
+
+    def evap_objective(q_r):
+        q_sat = saturation_mixing_ratio(T, p_full)
+        return jnp.sum(rain_evaporation(q_v, q_r, q_sat, evap_coeff=1.0))
+
+    g_evap_zero = jax.grad(evap_objective)(q_r_zero)
+    g_evap_neg = jax.grad(evap_objective)(-1e-3 * jnp.ones_like(q_r_zero))
+    assert jnp.all(jnp.isfinite(g_evap_zero)), "rain_evaporation grad at q_r=0 not finite"
+    assert jnp.all(jnp.isfinite(g_evap_neg)), "rain_evaporation grad at q_r<0 not finite"
+
+    # Sanity: at q_r > 0 the gradient should be the analytic derivative
+    # (positive evap_coeff, positive subsaturation, p<1) ⇒ positive grad.
+    q_r_pos = 1e-4 * jnp.ones_like(q_r_zero)
+    g_evap_pos = jax.grad(evap_objective)(q_r_pos)
+    # Subsaturated column was set up with q_v = 0.8 * q_sat so subsaturation > 0.
+    assert float(jnp.max(g_evap_pos)) > 0.0, "rain_evaporation grad should be positive for subsaturated column"

exec
/bin/zsh -lc 'git status --short' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-ge2e30B9' (errno=Operation not permitted)
git: warning: confstr() failed with code 5: couldn't get path of DARWIN_USER_TEMP_DIR; using /tmp instead
git: error: couldn't create cache file '/tmp/xcrun_db-IgcLclH2' (errno=Operation not permitted)
 M src/legoesm/atmosphere/physics/convection/_plume.py
 M src/legoesm/atmosphere/physics/convection/bechtold.py
 M src/legoesm/atmosphere/physics/convection/config.py
 M src/legoesm/atmosphere/physics/convection/emanuel.py
 M src/legoesm/atmosphere/physics/convection/kain_fritsch.py
 M src/legoesm/atmosphere/physics/convection/kuo.py
 M src/legoesm/atmosphere/physics/convection/mass_flux.py
 M src/legoesm/atmosphere/physics/convection/tiedtke.py
 M src/legoesm/atmosphere/physics/convection/zhang_mcfarlane.py
 M src/legoesm/atmosphere/physics/gravity_wave_drag/config.py
 M src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py
 M src/legoesm/atmosphere/physics/gravity_wave_drag/mcfarlane.py
 M src/legoesm/atmosphere/physics/microphysics/_warm_rain.py
 M src/legoesm/atmosphere/physics/microphysics/kessler.py
 M src/legoesm/atmosphere/physics/microphysics/morrison.py
 M src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py
 M src/legoesm/atmosphere/physics/microphysics/sundqvist.py
 M src/legoesm/atmosphere/physics/microphysics/thompson.py
 M src/legoesm/atmosphere/physics/ml_parameterization.py
 M src/legoesm/atmosphere/physics/radiation/solar.py
 M src/legoesm/atmosphere/physics/thermodynamics.py
 M src/legoesm/atmosphere/physics/turbulence/config.py
 M src/legoesm/atmosphere/physics/turbulence/edmf.py
 M src/legoesm/atmosphere/physics/turbulence/holtslag_boville.py
 M src/legoesm/ml/physics/data.py
 M src/legoesm/thermo.py
 M tests/unit/test_bechtold.py
 M tests/unit/test_convection_plume.py
 M tests/unit/test_diff_atmosphere_physics.py
 M tests/unit/test_ml_physics_parameterization.py
 M tests/unit/test_physics_convection.py
 M tests/unit/test_physics_gwd.py
 M tests/unit/test_physics_microphysics.py
 M tests/unit/test_tiedtke.py
 M tests/unit/test_zhang_mcfarlane.py
?? .claude/agents/physics-validator.md
?? .physics-validator/

codex
The saturation fix itself looks dimensionally sound. I’m also checking the three deferred findings against current code so I can separate “still deferred” from anything newly caused by this patch.
exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py | sed -n '1,260p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Hines (1997) Doppler-spread gravity wave drag parameterization.
     2	
     3	Non-orographic GWD scheme based on Doppler shifting and spectral
     4	saturation of gravity waves. Uses bottom-up propagation with smooth
     5	sigmoid activation for full differentiability.
     6	
     7	References
     8	----------
     9	- Hines, C. O. (1997). Doppler-spread parameterization of gravity-wave
    10	  momentum deposition in the middle atmosphere. 1. Basic formulation.
    11	  J. Atmos. Solar-Terr. Phys., 59, 371-386.
    12	"""
    13	
    14	from __future__ import annotations
    15	
    16	import jax
    17	import jax.numpy as jnp
    18	
    19	from legoesm import constants
    20	from legoesm.atmosphere.physics.gravity_wave_drag.config import HinesConfig
    21	from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
    22	
    23	
    24	def hines_gwd(
    25	    u: jax.Array,
    26	    v: jax.Array,
    27	    T: jax.Array,
    28	    p_full: jax.Array,
    29	    p_half: jax.Array,
    30	    z_full: jax.Array,
    31	    z_half: jax.Array,
    32	    rho: jax.Array,
    33	    lat: jax.Array,
    34	    dt: float,
    35	    config: HinesConfig,
    36	) -> GWDOutput:
    37	    """Compute Hines Doppler-spread GWD tendencies.
    38	
    39	    Parameters
    40	    ----------
    41	    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt, config
    42	        Standard GWD backend signature. All column arrays (ncol, nlev).
    43	
    44	    Returns
    45	    -------
    46	    GWDOutput
    47	    """
    48	    ncol, nlev = u.shape
    49	
    50	    # Brunt-Väisälä frequency
    51	    theta = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    52	    dz_full = jnp.abs(z_full[:, :-1] - z_full[:, 1:])
    53	    dz_full = jnp.clip(dz_full, 1.0, None)
    54	    dtheta_dz = (theta[:, :-1] - theta[:, 1:]) / dz_full
    55	    theta_bar = 0.5 * (theta[:, :-1] + theta[:, 1:])
    56	    N2_half = (constants.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz
    57	    N2_half = jnp.clip(N2_half, 1e-8, None)
    58	    N_half = jnp.sqrt(N2_half)
    59	
    60	    N_full = jnp.concatenate([
    61	        N_half[:, :1],
    62	        0.5 * (N_half[:, :-1] + N_half[:, 1:]),
    63	        N_half[:, -1:],
    64	    ], axis=1)
    65	
    66	    # Wind magnitude at each level
    67	    U_mag = jnp.sqrt(u ** 2 + v ** 2 + 1e-10)
    68	
    69	    # Layer thickness
    70	    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    71	    dz = jnp.clip(dz, 1.0, None)
    72	
    73	    # Saturation amplitude per level:
    74	    # As gravity waves propagate upward, their amplitude grows with
    75	    # decreasing density (energy conservation: F ~ rho * sigma^2 = const).
    76	    # Saturation occurs when the wave-induced velocity perturbation
    77	    # reaches the critical Doppler-broadening amplitude
    78	    # ``sigma_critical = N / m_*`` (Hines 1997 eq. 9), which is constant
    79	    # per column at fixed N and m_*.  An earlier formulation divided this
    80	    # by ``rho_ratio = sqrt(rho_sfc/rho) ≥ 1`` ⇒ ``sigma_sat`` *decreased*
    81	    # with altitude, the opposite of physical expectation: amplitudes
    82	    # grow with altitude (1/sqrt(rho)) so the cap should remain at least
    83	    # constant.  The /rho_ratio factor caused premature saturation aloft
    84	    # and biased the drag deposition lower in the column (audit GWD-B2).
    85	    sigma_sat = N_full / jnp.clip(config.m_star, 1e-6, None)
    86	
    87	    # Per-level WKB growth factor for the bottom-up scan.  Going from
    88	    # level (k+1) to level k (one step upward), the amplitude grows by
    89	    # ``sqrt(rho[k+1] / rho[k])`` (energy conservation rho * sigma^2).
    90	    # The carry already contains the integrated WKB amplitude from the
    91	    # surface to level k+1, so we multiply by the *inter-level* ratio,
    92	    # not the cumulative ``sqrt(rho_sfc/rho_k)``.  Multiplying by the
    93	    # cumulative factor at every step compounds the growth and
    94	    # over-amplifies the wave by a product of cumulative ratios — a
    95	    # bug masked in operational use only because the sigma_sat cap
    96	    # truncates the runaway.
    97	    rho_ratio_step = jnp.ones_like(rho)
    98	    rho_ratio_step = rho_ratio_step.at[:, :-1].set(
    99	        jnp.sqrt(jnp.clip(
   100	            rho[:, 1:] / jnp.clip(rho[:, :-1], 0.01, None), 1.0, None,
   101	        ))
   102	    )
   103	
   104	    # Bottom-up scan: propagate sigma_gw upward from surface.
   105	    # ``rho_ratio_step[:, k]`` carries amplitude from level k+1 to level k;
   106	    # at the surface (k = nlev-1) the step factor is 1 (initial condition).
   107	    def scan_fn(carry, k_rev):
   108	        sigma_gw = carry
   109	        k = nlev - 1 - k_rev
   110	
   111	        # Amplitude growth from density decrease (single-layer step)
   112	        sigma_grown = sigma_gw * rho_ratio_step[:, k]
   113	
   114	        # Dissipation where grown amplitude exceeds saturation
   115	        f_diss = jax.nn.sigmoid(
   116	            config.doppler_sharpness * (sigma_grown - sigma_sat[:, k])
   117	        )
   118	        sigma_new = sigma_grown * (1.0 - f_diss) + sigma_sat[:, k] * f_diss
   119	
   120	        # Momentum deposited: rho * (sigma_grown - sigma_new) ~ stress gradient.
   121	        # Clamp the lower bound to zero: the smooth ``f_diss`` sigmoid does
   122	        # not vanish exactly when ``sigma_grown < sigma_sat``, so without
   123	        # the floor a small "anti-drag" leak can appear in the transition
   124	        # region (``sigma_new`` slightly larger than ``sigma_grown`` ⇒ drag
   125	        # negative ⇒ accel positive ⇒ wave accelerates the resolved flow).
   126	        # GWD on the mean flow is always a momentum sink, never a source.
   127	        drag = (sigma_grown - sigma_new) * rho[:, k]
   128	        drag = jnp.clip(drag, 0.0, config.Fmax)
   129	
   130	        return sigma_new, drag
   131	
   132	    # Pin the carry dtype so the scan body stays at the input precision
   133	    # (defaulting allows x64 to silently promote the launch wind to f64).
   134	    sigma_gw_init = jnp.full((ncol,), config.total_rms_wind, dtype=u.dtype)
   135	    _, drag_stack = jax.lax.scan(scan_fn, sigma_gw_init, jnp.arange(nlev))
   136	    drag_all = drag_stack.T[:, ::-1]  # (ncol, nlev), top-first
   137	
   138	    # Convert to acceleration
   139	    accel = -drag_all / jnp.clip(rho * dz, 1e-10, None)
   140	
   141	    cos_a = u / jnp.clip(U_mag, config.U_mag_floor, None)
   142	    sin_a = v / jnp.clip(U_mag, config.U_mag_floor, None)
   143	    du_dt = accel * cos_a
   144	    dv_dt = accel * sin_a
   145	
   146	    # Frictional heating
   147	    dT_dt = -(u * du_dt + v * dv_dt) / constants.c_pd
   148	
   149	    # Column dissipation (positive-definite: KE lost by the mean flow)
   150	    eps_gwd = -jnp.sum(rho * (u * du_dt + v * dv_dt) * dz, axis=1)
   151	
   152	    return GWDOutput(du_dt=du_dt, dv_dt=dv_dt, dT_dt=dT_dt, eps_gwd=eps_gwd)

exec
/bin/zsh -lc 'rg -n "sigma_grown|sigma_new|drag = rho|dN_i_nuc|dq_i_dep|graupel_frac|total_riming" src/legoesm/atmosphere/physics -S' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:112:        sigma_grown = sigma_gw * rho_ratio_step[:, k]
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:116:            config.doppler_sharpness * (sigma_grown - sigma_sat[:, k])
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:118:        sigma_new = sigma_grown * (1.0 - f_diss) + sigma_sat[:, k] * f_diss
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:120:        # Momentum deposited: rho * (sigma_grown - sigma_new) ~ stress gradient.
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:122:        # not vanish exactly when ``sigma_grown < sigma_sat``, so without
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:124:        # region (``sigma_new`` slightly larger than ``sigma_grown`` ⇒ drag
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:127:        drag = (sigma_grown - sigma_new) * rho[:, k]
src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:130:        return sigma_new, drag
src/legoesm/atmosphere/physics/microphysics/thompson.py:119:    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)
src/legoesm/atmosphere/physics/microphysics/thompson.py:124:    dq_i_dep = (
src/legoesm/atmosphere/physics/microphysics/thompson.py:142:    total_riming = riming_i + riming_s
src/legoesm/atmosphere/physics/microphysics/thompson.py:165:    # the previous form set ``rime_to_graupel ∝ total_riming``, then
src/legoesm/atmosphere/physics/microphysics/thompson.py:171:    graupel_frac = jax.nn.sigmoid(
src/legoesm/atmosphere/physics/microphysics/thompson.py:172:        config.graupel_sharpness * (total_riming - config.rime_to_graupel_threshold)
src/legoesm/atmosphere/physics/microphysics/thompson.py:174:    rime_to_graupel_from_i = config.rime_to_graupel_rate * riming_i * graupel_frac
src/legoesm/atmosphere/physics/microphysics/thompson.py:175:    rime_to_graupel_from_s = config.rime_to_graupel_rate * riming_s * graupel_frac
src/legoesm/atmosphere/physics/microphysics/thompson.py:200:    total_riming = riming_i + riming_s
src/legoesm/atmosphere/physics/microphysics/thompson.py:243:        + L_s * dq_i_dep / c_pd
src/legoesm/atmosphere/physics/microphysics/thompson.py:260:    dq_v_dt = -condensation + evaporation - dq_i_dep
src/legoesm/atmosphere/physics/microphysics/thompson.py:264:        dq_i_dep + bergeron + riming_i - aggregation - melt_ice
src/legoesm/atmosphere/physics/microphysics/thompson.py:272:    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)
src/legoesm/atmosphere/physics/microphysics/morrison.py:98:    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)
src/legoesm/atmosphere/physics/microphysics/morrison.py:103:    dq_i_dep = (
src/legoesm/atmosphere/physics/microphysics/morrison.py:203:        + L_s * dq_i_dep / c_pd
src/legoesm/atmosphere/physics/microphysics/morrison.py:216:    dq_v_dt = -condensation + evaporation - dq_i_dep
src/legoesm/atmosphere/physics/microphysics/morrison.py:219:    dq_i_dt = dq_i_dep + bergeron + riming_i - aggregation - melt_ice + sed_i
src/legoesm/atmosphere/physics/microphysics/morrison.py:224:    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)

  ✓ Inspect warm-rain saturation helper and scheme call sites
  ✓ Trace Morrison/Thompson/Kessler/SB mass tendencies and limiter ordering
  → Review tests covering clear-air and cloudy donor behavior
  • Summarize findings and deferred-item status
exec
/bin/zsh -lc 'rg -n "q_c_after|qc_does_not|q_c went negative|Kessler|seifert" tests/unit/test_physics_microphysics.py tests/unit/test_warm_rain.py -S' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
exec
/bin/zsh -lc 'rg -n "dq_c_dt =|condensation =|qc_scale|saturation_adjustment" src/legoesm/atmosphere/physics/microphysics -S' in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
src/legoesm/atmosphere/physics/microphysics/sundqvist.py:62:    condensation = (
src/legoesm/atmosphere/physics/microphysics/sundqvist.py:170:    dq_c_dt = rates.condensation - rates.autoconversion
src/legoesm/atmosphere/physics/microphysics/thompson.py:24:    saturation_adjustment,
src/legoesm/atmosphere/physics/microphysics/thompson.py:84:    # subsaturated clear air.  See _warm_rain.saturation_adjustment.
src/legoesm/atmosphere/physics/microphysics/thompson.py:85:    condensation, q_sat = saturation_adjustment(
src/legoesm/atmosphere/physics/microphysics/thompson.py:183:    # Include the evaporation branch of saturation_adjustment (negative
src/legoesm/atmosphere/physics/microphysics/thompson.py:191:    qc_scale = jnp.minimum(
src/legoesm/atmosphere/physics/microphysics/thompson.py:195:    dq_c_au = dq_c_au * qc_scale
src/legoesm/atmosphere/physics/microphysics/thompson.py:196:    dq_c_ac = dq_c_ac * qc_scale
src/legoesm/atmosphere/physics/microphysics/thompson.py:197:    bergeron = bergeron * qc_scale
src/legoesm/atmosphere/physics/microphysics/thompson.py:198:    riming_i = riming_i * qc_scale
src/legoesm/atmosphere/physics/microphysics/thompson.py:199:    riming_s = riming_s * qc_scale
src/legoesm/atmosphere/physics/microphysics/thompson.py:202:    # which has already been scaled by qc_scale above.  Re-scaling
src/legoesm/atmosphere/physics/microphysics/thompson.py:204:    # by ``qc_scale`` once preserves both per-donor proportionality and
src/legoesm/atmosphere/physics/microphysics/thompson.py:206:    rime_to_graupel_from_i = rime_to_graupel_from_i * qc_scale
src/legoesm/atmosphere/physics/microphysics/thompson.py:207:    rime_to_graupel_from_s = rime_to_graupel_from_s * qc_scale
src/legoesm/atmosphere/physics/microphysics/thompson.py:209:    dN_r_au = dN_r_au * qc_scale
src/legoesm/atmosphere/physics/microphysics/thompson.py:212:    condensation = jnp.where(
src/legoesm/atmosphere/physics/microphysics/thompson.py:213:        condensation < 0.0, condensation * qc_scale, condensation,
src/legoesm/atmosphere/physics/microphysics/thompson.py:261:    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:29:    saturation_adjustment,
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:71:    # donor-clamped — see _warm_rain.saturation_adjustment.
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:72:    condensation, q_sat = saturation_adjustment(
src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:106:    dq_c_dt = condensation - dq_c_au - dq_c_ac
src/legoesm/atmosphere/physics/microphysics/ml_emulator.py:90:    dq_c_dt = y[..., 2]
src/legoesm/atmosphere/physics/microphysics/ml_emulator.py:100:        dq_c_dt = dq_c_dt * 1e-6
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:52:def saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0, q_c=None):
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:97:    condensation = cond_frac * excess / dt
src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:105:        condensation = jnp.maximum(condensation, -q_c_avail / jnp.maximum(dt, 1e-10))
src/legoesm/atmosphere/physics/microphysics/kessler.py:82:    # Same pattern as ``_warm_rain.saturation_adjustment``.
src/legoesm/atmosphere/physics/microphysics/kessler.py:85:    condensation = cond_frac * excess / dt  # [kg/kg/s]
src/legoesm/atmosphere/physics/microphysics/kessler.py:87:    condensation = jnp.maximum(
src/legoesm/atmosphere/physics/microphysics/kessler.py:121:    dq_c_dt = dq_c_sat - autoconv - accretion
src/legoesm/atmosphere/physics/microphysics/morrison.py:24:    saturation_adjustment,
src/legoesm/atmosphere/physics/microphysics/morrison.py:77:    # subsaturated clear air.  See _warm_rain.saturation_adjustment.
src/legoesm/atmosphere/physics/microphysics/morrison.py:78:    condensation, q_sat = saturation_adjustment(
src/legoesm/atmosphere/physics/microphysics/morrison.py:151:    # cloud water").  ``saturation_adjustment`` has already donor-clamped
src/legoesm/atmosphere/physics/microphysics/morrison.py:157:    qc_scale = jnp.minimum(
src/legoesm/atmosphere/physics/microphysics/morrison.py:161:    dq_c_au = dq_c_au * qc_scale
src/legoesm/atmosphere/physics/microphysics/morrison.py:162:    dq_c_ac = dq_c_ac * qc_scale
src/legoesm/atmosphere/physics/microphysics/morrison.py:163:    bergeron = bergeron * qc_scale
src/legoesm/atmosphere/physics/microphysics/morrison.py:164:    riming_i = riming_i * qc_scale
src/legoesm/atmosphere/physics/microphysics/morrison.py:165:    riming_s = riming_s * qc_scale
src/legoesm/atmosphere/physics/microphysics/morrison.py:171:    condensation = jnp.where(
src/legoesm/atmosphere/physics/microphysics/morrison.py:172:        condensation < 0.0, condensation * qc_scale, condensation,
src/legoesm/atmosphere/physics/microphysics/morrison.py:175:    dN_r_au = dN_r_au * qc_scale
src/legoesm/atmosphere/physics/microphysics/morrison.py:217:    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s

 succeeded in 0ms:
tests/unit/test_physics_microphysics.py:17:from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
tests/unit/test_physics_microphysics.py:21:    KesslerConfig, SundqvistConfig, SeifertBehengConfig, MorrisonConfig,
tests/unit/test_physics_microphysics.py:79:                                     config=KesslerConfig())
tests/unit/test_physics_microphysics.py:83:    elif name == "seifert_beheng":
tests/unit/test_physics_microphysics.py:84:        return seifert_beheng_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
tests/unit/test_physics_microphysics.py:96:ALL_SCHEMES = ["kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"]
tests/unit/test_physics_microphysics.py:195:# 4g  Autoconversion threshold (Kessler)
tests/unit/test_physics_microphysics.py:200:    config = KesslerConfig()
tests/unit/test_physics_microphysics.py:217:        f"Kessler: rain production not higher above threshold: "
tests/unit/test_physics_microphysics.py:245:def test_qc_does_not_go_negative_at_long_dt(scheme, call):
tests/unit/test_physics_microphysics.py:284:    q_c_after = q_c + out.dq_c_dt * dt
tests/unit/test_physics_microphysics.py:285:    min_qc = float(jnp.min(q_c_after))
tests/unit/test_physics_microphysics.py:288:        f"{scheme}: q_c went negative ({min_qc:.3e}) after one explicit "
tests/unit/test_physics_microphysics.py:345:        ("seifert_beheng", seifert_beheng_microphysics),
tests/unit/test_physics_microphysics.py:360:    ``saturation_adjustment`` (and inline for Kessler) and joined to
tests/unit/test_physics_microphysics.py:364:    the buggy form produced ``q_c_after ≈ -3e-4 kg/kg`` after a
tests/unit/test_physics_microphysics.py:367:    if scheme in ("kessler", "seifert_beheng"):
tests/unit/test_physics_microphysics.py:369:            "kessler": KesslerConfig,
tests/unit/test_physics_microphysics.py:370:            "seifert_beheng": SeifertBehengConfig,
tests/unit/test_physics_microphysics.py:398:    q_c_after = q_c + out.dq_c_dt * dt
tests/unit/test_physics_microphysics.py:399:    min_q_c = float(jnp.min(q_c_after))
tests/unit/test_physics_microphysics.py:401:    # produces q_c_after ≈ -3e-4, well above this threshold.
tests/unit/test_physics_microphysics.py:403:        f"{scheme}: q_c_after went negative ({min_q_c:.3e}) in a "
tests/unit/test_physics_microphysics.py:596:    Kessler step), Sundqvist's diagnostic-rain semantics drain it to the
tests/unit/test_physics_microphysics.py:656:# 4l  Kessler / warm-rain accretion & evaporation: gradient finite at q_r=0
tests/unit/test_physics_microphysics.py:677:            T, q_v, h, p_full, p_half, rho, dz, 300.0, KesslerConfig(),
tests/unit/test_physics_microphysics.py:683:        "Kessler: grad w.r.t. q_r at q_r=0 contains NaN/Inf — "

exec
/bin/zsh -lc "nl -ba src/legoesm/atmosphere/physics/microphysics/sundqvist.py | sed -n '1,210p'" in /Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM
 succeeded in 0ms:
     1	"""Sundqvist large-scale diagnostic condensation scheme.
     2	
     3	A diagnostic scheme that activates condensation when relative humidity
     4	exceeds a critical threshold. Produces large-scale (non-convective)
     5	precipitation through autoconversion and sub-cloud evaporation.
     6	
     7	All operations use smooth (differentiable) approximations.
     8	
     9	References
    10	----------
    11	- Sundqvist et al. (1989): Condensation and cloud parameterization
    12	  studies with a mesoscale numerical weather prediction model.
    13	"""
    14	
    15	from __future__ import annotations
    16	
    17	from typing import NamedTuple
    18	
    19	import jax
    20	import jax.numpy as jnp
    21	
    22	from legoesm import constants
    23	from legoesm.thermo import saturation_mixing_ratio
    24	from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig
    25	from legoesm.atmosphere.physics.microphysics.output import (
    26	    HydrometeorState,
    27	    MicrophysicsOutput,
    28	)
    29	
    30	
    31	class SundqvistProcessRates(NamedTuple):
    32	    """Intermediate Sundqvist process rates used to assemble tendencies."""
    33	
    34	    condensation: jax.Array
    35	    autoconversion: jax.Array
    36	    evaporation: jax.Array
    37	    precipitation: jax.Array
    38	
    39	
    40	def diagnose_sundqvist_process_rates(
    41	    T: jax.Array,
    42	    q_v: jax.Array,
    43	    hydrometeors: HydrometeorState,
    44	    p_full: jax.Array,
    45	    p_half: jax.Array,
    46	    rho: jax.Array,
    47	    dz: jax.Array,
    48	    dt: float,
    49	    config: SundqvistConfig = SundqvistConfig(),
    50	) -> SundqvistProcessRates:
    51	    """Diagnose the Sundqvist condensation, rain conversion, and evaporation terms."""
    52	    del p_half  # Included for signature parity with ``sundqvist_microphysics``.
    53	    q_c = hydrometeors.q_c
    54	    sharpness = config.sigmoid_sharpness
    55	
    56	    # Saturation
    57	    q_sat = saturation_mixing_ratio(T, p_full)
    58	    RH = q_v / jnp.clip(q_sat, 1e-10)
    59	
    60	    # 1. Smooth condensation activation — convert increment [kg/kg] to tendency [kg/kg/s]
    61	    f = jax.nn.sigmoid(sharpness * (RH - config.RH_crit))
    62	    condensation = (
    63	        f * jnp.maximum(q_v - config.RH_crit * q_sat, 0.0) / dt
    64	    )  # [kg/kg/s]
    65	
    66	    # 2. Autoconversion
    67	    # condensation is a tendency [kg/kg/s]; multiply by dt to get increment [kg/kg]
    68	    P_auto = config.auto_rate * jnp.maximum(q_c + condensation * dt, 0.0)
    69	
    70	    # 3. Sub-cloud evaporation
    71	    evap_mask = jax.nn.sigmoid(sharpness * (config.RH_crit - RH))
    72	    P_flux_layer = P_auto * rho * dz
    73	
    74	    def scan_fn(carry, x):
    75	        P_above = carry
    76	        P_local, evap_m, rho_k, dz_k = x
    77	        P_total = P_above + P_local
    78	        evap = config.evap_coeff * evap_m * P_total / jnp.clip(rho_k * dz_k, 1.0)
    79	        evap = jnp.minimum(evap, P_total / jnp.clip(rho_k * dz_k, 1.0))
    80	        P_out = jnp.clip(P_total - evap * rho_k * dz_k, 0.0)
    81	        return P_out, evap
    82	
    83	    # Pick a working dtype that ``scan`` can carry without promotion.
    84	    # Under ``JAX_ENABLE_X64=1`` ``jnp.zeros``/``jnp.ones`` default to
    85	    # f64, so a state assembled from a mix of (f32) ``T`` and (f64)
    86	    # tracers ends up with f64 ``q_v``/``q_c``.  ``P_flux_layer``
    87	    # then inherits the f64 promotion from ``q_c + condensation * dt``,
    88	    # while a carry pinned to ``T.dtype`` (f32) would mismatch the
    89	    # f64 scan output.  Promoting to the wider of carry/input dtype
    90	    # keeps ``scan`` happy without silently downcasting precipitation
    91	    # mass.
    92	    _scan_dtype = jnp.promote_types(T.dtype, P_flux_layer.dtype)
    93	    inputs = (
    94	        jnp.moveaxis(P_flux_layer.astype(_scan_dtype), 1, 0),
    95	        jnp.moveaxis(evap_mask.astype(_scan_dtype), 1, 0),
    96	        jnp.moveaxis(rho.astype(_scan_dtype), 1, 0),
    97	        jnp.moveaxis(dz.astype(_scan_dtype), 1, 0),
    98	    )
    99	    P_init = jnp.zeros(T.shape[0], dtype=_scan_dtype)
   100	    P_final, evap_col = jax.lax.scan(scan_fn, P_init, inputs)
   101	    evaporation = jnp.moveaxis(evap_col, 0, 1)
   102	    return SundqvistProcessRates(
   103	        condensation=condensation,
   104	        autoconversion=P_auto,
   105	        evaporation=evaporation,
   106	        precipitation=P_final,
   107	    )
   108	
   109	
   110	def sundqvist_microphysics(
   111	    T: jax.Array,
   112	    q_v: jax.Array,
   113	    hydrometeors: HydrometeorState,
   114	    p_full: jax.Array,
   115	    p_half: jax.Array,
   116	    rho: jax.Array,
   117	    dz: jax.Array,
   118	    dt: float,
   119	    config: SundqvistConfig = SundqvistConfig(),
   120	) -> MicrophysicsOutput:
   121	    """Compute Sundqvist diagnostic condensation tendencies.
   122	
   123	    Parameters
   124	    ----------
   125	    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
   126	        Same interface as all microphysics backends.
   127	
   128	    Returns
   129	    -------
   130	    MicrophysicsOutput
   131	    """
   132	    ncol, nlev = T.shape
   133	    rates = diagnose_sundqvist_process_rates(
   134	        T=T,
   135	        q_v=q_v,
   136	        hydrometeors=hydrometeors,
   137	        p_full=p_full,
   138	        p_half=p_half,
   139	        rho=rho,
   140	        dz=dz,
   141	        dt=dt,
   142	        config=config,
   143	    )
   144	
   145	    # 4. Latent heating
   146	    net_cond = rates.condensation - rates.evaporation
   147	    dT_dt = constants.L_v * net_cond / constants.c_pd
   148	
   149	    # Tendencies.  Sundqvist is a *diagnostic* large-scale precipitation
   150	    # scheme: rain produced by autoconversion is treated as falling
   151	    # instantly through the column (the bottom-up scan in
   152	    # ``diagnose_sundqvist_process_rates`` accumulates the layer
   153	    # autoconversion source into a downward mass flux ``P_total`` and
   154	    # subtracts sub-cloud evaporation, so ``rates.precipitation`` is the
   155	    # surface flux).  Adding ``autoconversion - evaporation`` to ``dq_r_dt``
   156	    # would also accumulate that mass as a ``q_r`` tracer, double-counting
   157	    # it: the column would lose water to surface precipitation AND grow
   158	    # ``q_r`` per step.
   159	    #
   160	    # Diagnostic-rain semantics (full): the scheme should *own* the q_r
   161	    # tracer, not just leave it untouched.  Any q_r passed in (from a
   162	    # prior step under a prognostic scheme like Kessler, or from a
   163	    # warm-start) is treated as already-falling rain and drained to the
   164	    # surface in one step.  The drained mass is added to the surface
   165	    # precipitation flux so the column water budget closes:
   166	    #     int (dq_v + dq_c + dq_r) dp/g  =  -precipitation
   167	    # In a steady state with q_r = 0 input, ``dq_r_dt = 0`` and the
   168	    # column budget reduces to ``int (dq_v + dq_c) dp/g = -precipitation``.
   169	    dq_v_dt = -rates.condensation + rates.evaporation
   170	    dq_c_dt = rates.condensation - rates.autoconversion
   171	    dt_safe = jnp.maximum(dt, 1e-10)
   172	    q_r_in = jnp.clip(hydrometeors.q_r, 0.0, None)
   173	    dq_r_dt = -q_r_in / dt_safe
   174	    # Drained mass [kg/m^2/s] is added to the surface precipitation
   175	    # diagnostic so total column water exits the column at the correct
   176	    # rate.  ``rates.precipitation`` is the autoconversion-driven surface
   177	    # flux; ``q_r_drain_flux`` is the column-integrated drain.
   178	    dp = p_half[:, 1:] - p_half[:, :-1]
   179	    q_r_drain_flux = jnp.sum(q_r_in * dp, axis=1) / (constants.g * dt_safe)
   180	    precipitation = rates.precipitation + q_r_drain_flux
   181	
   182	    # Pin dtype to the input precision so we never silently promote
   183	    # the unused-tendency placeholders to f64 under x64 mode.
   184	    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
   185	    return MicrophysicsOutput(
   186	        dT_dt=dT_dt,
   187	        dq_v_dt=dq_v_dt,
   188	        dq_c_dt=dq_c_dt,
   189	        dq_r_dt=dq_r_dt,
   190	        dq_i_dt=z,
   191	        dq_s_dt=z,
   192	        dq_g_dt=z,
   193	        dN_c_dt=z,
   194	        dN_r_dt=z,
   195	        dN_i_dt=z,
   196	        precipitation=precipitation,
   197	    )

  ✓ Inspect warm-rain saturation helper and scheme call sites
  ✓ Trace Morrison/Thompson/Kessler/SB mass tendencies and limiter ordering
  ✓ Review tests covering clear-air and cloudy donor behavior
  → Summarize findings and deferred-item status
codex
**Findings**

One residual concern: Kessler/SB still do not have a joint `q_c` donor limiter for cloudy/rainy columns. The clear-air bug is fixed, but “accretion is proportional to `q_c`” only guarantees zero sink when `q_c = 0`; it does not bound an explicit Euler sink for `q_c > 0`. In [kessler.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/kessler.py:121>) and [seifert_beheng.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:106>), `dq_c_dt` subtracts autoconversion/accretion after the saturation clamp. Example: Kessler with `q_c=1e-3`, `q_r=1e-3`, `dt=1200 s`, saturated air gives accretion about `5.2e-6 kg/kg/s`, or `6.3e-3 kg/kg` per step, exceeding available `q_c`. This is not a total-water conservation error, because the sink appears as rain source, and it is not the subsaturated clear-air regression. It is a remaining cloudy-column positivity gap if that is in scope.

For the reported subsaturated clear-air bug: confirmed fixed. `_warm_rain.saturation_adjustment` clamps negative condensation to `>= -q_c/dt` in [_warm_rain.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:98>), so `q_c=0` forces the evaporation branch to zero. Kessler mirrors that inline; SB/Morrison/Thompson pass `q_c`.

Morrison/Thompson ordering is correct. The reassignment at [morrison.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/morrison.py:171>) and [thompson.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:212>) happens before `dq_v_dt`, `dq_c_dt`, and `dT_dt` are assembled, so `dq_v_dt = -condensation + ...` uses the scaled value. Mass conservation for the saturation branch is therefore exact: `dq_v = -C_scaled`, `dq_c = +C_scaled`.

Dimensionally, the clamp is correct: `condensation` and `q_c/dt` are both `kg/kg/s`; `qc_sink_total * dt` is `kg/kg`; `qc_avail / (...)` is dimensionless.

**Deferred Items**

Still deferred:
- Hines drag dimensional inconsistency remains at [hines.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:127>).
- `N_i` nucleation without ice mass remains in Morrison/Thompson.
- Thompson `graupel_frac` still uses pre-clamp `total_riming`; mass is conserved, but thresholding is still based on unscaled demand.

I did not rerun pytest in this read-only sandbox; this is a source-level review.
2026-05-01T23:34:19.469586Z ERROR codex_core::session: failed to record rollout items: thread 019de5e0-cc6c-7f91-9486-29bd043eae57 not found
tokens used
101,230
**Findings**

One residual concern: Kessler/SB still do not have a joint `q_c` donor limiter for cloudy/rainy columns. The clear-air bug is fixed, but “accretion is proportional to `q_c`” only guarantees zero sink when `q_c = 0`; it does not bound an explicit Euler sink for `q_c > 0`. In [kessler.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/kessler.py:121>) and [seifert_beheng.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:106>), `dq_c_dt` subtracts autoconversion/accretion after the saturation clamp. Example: Kessler with `q_c=1e-3`, `q_r=1e-3`, `dt=1200 s`, saturated air gives accretion about `5.2e-6 kg/kg/s`, or `6.3e-3 kg/kg` per step, exceeding available `q_c`. This is not a total-water conservation error, because the sink appears as rain source, and it is not the subsaturated clear-air regression. It is a remaining cloudy-column positivity gap if that is in scope.

For the reported subsaturated clear-air bug: confirmed fixed. `_warm_rain.saturation_adjustment` clamps negative condensation to `>= -q_c/dt` in [_warm_rain.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/_warm_rain.py:98>), so `q_c=0` forces the evaporation branch to zero. Kessler mirrors that inline; SB/Morrison/Thompson pass `q_c`.

Morrison/Thompson ordering is correct. The reassignment at [morrison.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/morrison.py:171>) and [thompson.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/microphysics/thompson.py:212>) happens before `dq_v_dt`, `dq_c_dt`, and `dT_dt` are assembled, so `dq_v_dt = -condensation + ...` uses the scaled value. Mass conservation for the saturation branch is therefore exact: `dq_v = -C_scaled`, `dq_c = +C_scaled`.

Dimensionally, the clamp is correct: `condensation` and `q_c/dt` are both `kg/kg/s`; `qc_sink_total * dt` is `kg/kg`; `qc_avail / (...)` is dimensionless.

**Deferred Items**

Still deferred:
- Hines drag dimensional inconsistency remains at [hines.py](</Users/pierregentine/Library/CloudStorage/GoogleDrive-pg2328@columbia.edu/My Drive/Code/legoESM/src/legoesm/atmosphere/physics/gravity_wave_drag/hines.py:127>).
- `N_i` nucleation without ice mass remains in Morrison/Thompson.
- Thompson `graupel_frac` still uses pre-clamp `total_riming`; mass is conserved, but thresholding is still based on unscaled demand.

I did not rerun pytest in this read-only sandbox; this is a source-level review.
