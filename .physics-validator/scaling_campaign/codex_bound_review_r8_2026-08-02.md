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
session id: 019fc301-389e-7522-b984-fe8dd49f7fa0
--------
user
Round-8 confirm. Your r7 items, dispositions: (1) bound table corrected to 2.863/1.894/2.999 -> 2.35/2.94/3.20 with an explicit mixed-input-artefact note + your nd=128 rerun and f64 census recorded in doc; (2) probe: compile-once ordering kept simple but count-assert added (parsed cp_ops must equal census count), per-op dtype/dims/bytes records kept, X64 mode arg added; (3) wording: 'above this MODEL', calibration-extrapolation note, pole-bias PLAUSIBLE-direction; (4) A/B: A2 trailing control added (A,B,C,A2), interpretation-limit note (null refutes this threshold/impl only), ocean-null contrast reworded to dependency-mix difference; resubmitted as job 26630576. Read the doc tail section + scripts/tmp/_probe_latlon_halo_census.py + scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch. Consistent? End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
codex
I’ll check the three requested artifacts against each listed disposition, then give a release verdict with any concrete mismatches.
exec
/usr/bin/bash -lc "pwd && rg --files -g '*probe_latlon_halo_census.py' -g 'atm_ll128_combine_ab.sbatch' -g '*.md' -g '*.txt' | sed -n '1,240p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 341ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
CHANGELOG.md
requirements_mpi.txt
CONTRIBUTING.md
CMIP.md
README.md
docs/canopy_longwave_graybody_conservation.md
docs/land_skin_t_bias_reproduce.md
docs/references.md
docs/COMPARE_REANALYSIS.md
docs/index.md
docs/ec_site_offline_run.md
docs/production_reorg.md
docs/next_implementations.md
FEDERATION.md
docs/amip_test_runs_2026-06-07.md
project_status.md
docs/performance/multinode_gpu_direct_cxi.md
docs/performance/issue_852_cube_shardmap_rootcause.md
docs/archive/rrtmgp.iter70-backup.md
docs/archive/rrtmgp.iter50-backup.md
docs/archive/rrtmgp.iter40-backup.md
docs/archive/rrtmgp.iter30-backup.md
docs/archive/CLAUDE.original.prev.md
docs/archive/rrtmgp.iter60-backup.md
docs/archive/CRM_implementation.original.md
docs/archive/rrtmgp.iter20-backup.md
docs/archive/rrtmgp.original.md
docs/archive/CLAUDE.original.md
docs/archive/rrtmgp.iter80-backup.md
docs/land_offline_moisture_scoping.md
docs/atmosphere/amip_realism_investigation.md
docs/land_high_elevation_snow_ice.md
docs/validation/cmip_readiness.md
docs/validation/cubed_sphere_sw_504_506.md
docs/validation/dycore_validation_catalog.md
docs/validation/PHYSICS_PARAMETERIZATION_TESTS.md
docs/validation/TESTING.md
config/4DVar_single/README.md
docs/physics-notes/moist_les_weno5_stabilization.md
docs/physics-notes/pseudo_incompressible_les.md
docs/physics-notes/pseudo_spectral_advection_smoothness.md
docs/physics-notes/les_crossgrid_regression_2026-07.md
docs/physics-notes/moist_les_spectral.md
docs/physics-notes/parameterization_faithfulness_audit.md
docs/performance/scaling/scaling_theoretical_limit_report_2026-06-15.md
docs/performance/scaling/SCALING_STATUS_AUDIT.md
docs/performance/scaling/bcw_scaling_status.md
docs/performance/scaling/scaling_review_2026-06-13.md
docs/performance/scaling/crm_les_scaling.md
docs/performance/scaling/c3_cube_face_scatter_driver_plan.md
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md
docs/performance/scaling/RESUME_STATE_2026-06-13.md
docs/performance/scaling/distance_to_limit_2026-06-13.md
docs/performance/scaling/crm_gpu_l2_tiling.md
docs/performance/scaling/cube_transport_tiling_design.md
docs/performance/scaling/derecho_levante_sota_review_2026-07.md
docs/performance/scaling/spmd_message_census_2026-07-08.md
docs/performance/scaling/SCALING_SUMMARY.md
docs/performance/scaling/scaling_tpu.md
docs/performance/scaling/latlon_2d_build_plan.md
docs/performance/scaling/derecho_colleague_runbook_2026-07.md
docs/performance/scaling/mpas_ocean_distributed_stage_audit.md
docs/performance/scaling/sh_gemm_design_2026-06-13.md
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md
docs/performance/scaling/cube_moist_tiled_step_design.md
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md
docs/performance/scaling/fig_scaling_caption.md
docs/performance/scaling/latlon_2d_decomposition_design.md
docs/performance/scaling/amip_mpi_scaling.md
docs/performance/scaling/d2a2c_spmd_stage_design.md
docs/performance/scaling/scaling.md
docs/performance/scaling/spectral_level_shard_cliff.md
docs/performance/scaling/cube_tiled_step_design.md
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md
docs/performance/scaling/mpas_atm_native_step_audit.md
docs/performance/scaling/d2a2c_edge_specials_design.txt
docs/performance/scaling/levante_campaign_2026-07-24.md
docs/performance/scaling/scaling_crm_gpu.md
docs/performance/scaling/scaling_levers_audit_2026-06-14.md
docs/performance/scaling/CRM_LES_SUMMARY.md
docs/performance/scaling/scaling_gpu.md
docs/performance/scaling/scaling_bottleneck_audit_2026-06-10.md
docs/performance/scaling/spectral_gpu_feasibility.md
docs/performance/scaling/literature_parallelization_2026-06.md
docs/performance/scaling/cube_production_tiling_design.md
docs/performance/scaling/barotropic_multinode_verdict_2026-06-15.md
docs/performance/scaling/scaling_levers_audit_2026-06-15.md
docs/performance/REAL_HARDWARE_SCALING.md
docs/performance/scream_parity_scope.md
docs/scaling/external_scaling_transfer_assessment.md
docs/scaling/atm_latlon_spmd_scaling.md
docs/wb/scale_training_runbook.md
docs/ocean/fidelity/phase_b1_cube_bottom_drag.md
docs/ocean/fidelity/recipe_comparison.md
docs/ocean/fidelity/phase_c_diagnostics.md
docs/ocean/fidelity/phase_e_climate_infra.md
docs/ocean/fidelity/phase_f_long_runs.md
docs/ocean/fidelity/phase_b3_cube_vs_latlon.md
docs/ocean/fidelity/oceananigans_reproduction_scoreboard.md
docs/ocean/fidelity/initial_comparison_5e23d684.md
docs/ocean/fidelity/dino_handoff_2026_07.md
docs/ocean/fidelity/oceananigans_recipe_wiring_plan.md
docs/ocean/fidelity/initial_comparison_828dd7e0.md
docs/ocean/fidelity/veros_acc_tendency_comparison.md
docs/ocean/fidelity/mitgcm_oracle_status.md
docs/ocean/fidelity/differentiable_nemo_plan.md
docs/ocean/fidelity/autonomous_execution_spec.md
docs/ocean/fidelity/initial_comparison_f84f4029.md
docs/ocean/fidelity/oracle_recipe_strategy.md
docs/ocean/fidelity/nemo_gyre_wiring_comparison.md
docs/ocean/fidelity/dino_pgf_alignment.md
docs/ocean/fidelity/bulletproof_local_runs.md
docs/ocean/fidelity/internal_tide_barotropic_coriolis_fix_plan.md
docs/ocean/fidelity/eke_len_build_spec.md
docs/ocean/fidelity/flux_form_momentum_scope.md
docs/ocean/fidelity/bulletproof_summary.md
docs/ocean/fidelity/mitgcm_gyre_energy_conservation.md
docs/ocean/fidelity/flux_form_build_progress.md
docs/ocean/fidelity/flux_form_build_spec.md
docs/ocean/fidelity/eke_scope.md
docs/ocean/fidelity/phase_g_veros_recipe_audit.md
docs/ocean/fidelity/legoesm_vs_veros_v2.md
docs/ocean/fidelity/lock_exchange_velocity_investigation.md
docs/ocean/fidelity/aerobulk_bulk_flux_oracle.md
docs/ocean/fidelity/recipe_case_board.md
docs/ocean/fidelity/dino_wiring_diagram.md
docs/ocean/fidelity/dino_session_2026_07_25.md
docs/ocean/fidelity/eke_len_build_progress.md
docs/ocean/fidelity/phase_g_recipe_fidelity_plan.md
docs/ocean/fidelity/mitgcm_unsplit_freesurface_fix.md
docs/ocean/fidelity/initial_comparison_607701ba.md
docs/ocean/fidelity/eke_build_progress.md
docs/ocean/fidelity/eke_build_spec.md
docs/ocean/fidelity/dino_setup_audit.md
docs/ocean/fidelity/nemo_gyre_fidelity_plan.md
docs/ocean/fidelity/dino_l1_exactness_audit.md
docs/ocean/fidelity/phase_a_6f0e0de1.md
docs/ocean/fidelity/autonomous_progress.md
docs/ocean/fidelity/dino_tendency_certificate.md
docs/ocean/fidelity/phase_d_benchmarks.md
docs/ocean/fidelity/dino_program_summary.md
docs/ocean/fidelity/bulletproof_run_cube_vs_latlon.md
docs/ocean/fidelity/bulletproof_run_eady_dino.md
docs/ocean/fidelity/veros_acc_final_scoreboard.md
docs/ocean/fidelity/dino_surface_forcing_alignment.md
docs/dycore/fv3_native_p4c_oracle.md
docs/specs/mc3d_oracle_fidelity.md
docs/specs/mc3d_raytracer.md
docs/land/arctic_carbon_residual_audit.md
docs/land/stageB_carbon_calibration_plan.md
docs/land/clm_ml_implementation_plan.md
docs/land/multipool_som_phenology_plan.md
docs/land/two_leaf_canopy_wet_forest_bistability.md
docs/land/carbon_equilibrium_audit.md
docs/land/lmip_s3_scope.md
docs/land/clm_ml_s3_masked_vmap_plan.md
docs/land/ec_site_bigleaf_regression.md
docs/land/lmip_biophys_runbook.md
config/machines/README.md
docs/ocean/long_runs/results_bryan_30yr_local.md
docs/ocean/long_runs/results_omip2_5deg_5yr_local.md
docs/ocean/long_runs/results_bryan_thc_skeleton.md
docs/ocean/long_runs/results_omip2_skeleton.md
docs/dev-notes/fv3_fortran_fidelity_review.md
docs/dev-notes/les_plane_turbulence_notes.md
docs/dev-notes/ocean_test_experiments_audit.md
docs/dev-notes/ocean_test_matrix_changelog.md
docs/dev-notes/GPU_SCALING_BRANCH.md
docs/dev-notes/slab_s2s_documentation.md
docs/dev-notes/ocean_boundary_conditions_analysis.md
docs/dev-notes/clubb.md
docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md
docs/dev-notes/CROSS_GRID_COMPARISON_REPORT.md
docs/dev-notes/implementation_summary.md
docs/dev-notes/ocean_grid_staggering.md
docs/dev-notes/regrid_polar_coverage_2026-07-24.md
docs/dev-notes/LATLON_CGRID_MIGRATION.md
docs/dev-notes/parameterization_checks.md
docs/dev-notes/cross_grid_comparison_plots_plan.md
docs/dev-notes/FIX_RESTART_TIME.md
docs/dev-notes/OCEAN_DEVELOPMENT_LOG.md
docs/dev-notes/ocean_faithfulness_nemo.md
docs/dev-notes/README.md
docs/architecture/DISTRIBUTED_ARCHITECTURE.md
docs/dev-notes/cubed_sphere_cgrid_ocean_plan.md
docs/dev-notes/faithful_latlon_FV.md
docs/dev-notes/clubb_port_history.md
docs/dev-notes/CRM_faithful_SAM.md
packages/ocean/README.md
docs/dev-notes/ocean_validation_improvement_plan.md
docs/dev-notes/cubed_sphere_edge_artifacts.md
docs/dev-notes/ocean_experiments_reference.md
docs/dev-notes/mpi_local_setup.md
docs/dev-notes/fv3_faithful.md
docs/dev-notes/OMIP_faithful.md
docs/dev-notes/SIMULATION_FULL_CHECK_PROGRESS.md
docs/ocean/experiments/omip_1deg_production_plan.md
docs/ocean/experiments/etopo_instability_dycore_review.md
docs/ocean/experiments/zhang2024_acc_channel_spec.md
docs/ocean/experiments/mle_mpas_port_plan.md
docs/ocean/experiments/gm_redi_latlon_cgrid_plan.md
docs/ocean/experiments/al81_corner_triad_audit.md
docs/ocean/experiments/dissipation_research_ocean.md
docs/ocean/experiments/dissipation_research_dycore.md
docs/ocean/experiments/NEXT_STEPS_advection_comparison.md
docs/ocean/experiments/pgf_production_models_research.md
docs/ocean/experiments/dino_rigid_lid_acc_analysis.md
docs/ocean/experiments/realistic_geometry_lat_lon_plan.md
docs/ocean/experiments/omip_protocol_review.md
docs/ocean/experiments/mpas_bci_damping_investigation.md
docs/ocean/experiments/tripole_grid_implementation_status.md
docs/ocean/experiments/jra55do_pipeline_plan.md
docs/ocean/experiments/tripole_grid_plan.md
docs/ocean/experiments/partial_cells_results.md
docs/ocean/experiments/mpas_vs_latlon_comparison_plan.md
docs/ocean/experiments/tropical_omip_plan.md
docs/ocean/experiments/cubed_sphere_pgf_stability.md
docs/ocean/experiments/global_overturning_plan.md
docs/ocean/experiments/realistic_geometry_phase3a_results.md
docs/ocean/experiments/pgf_test_plan.md
docs/ocean/experiments/silvestri_baroclinic_jet_plan.md
docs/ocean/experiments/advection_convergence_testing.md
docs/ocean/experiments/omip_1deg_plan.md
docs/ocean/experiments/gm_redi_mpas_plan.md
docs/ocean/experiments/dino_acc_solver_diagnostic_and_controls.md
docs/ocean/experiments/mpas_ico5_production_tuning_plan.md
docs/ocean/experiments/partial_cells_plan.md
docs/ocean/experiments/pgf_smc03_code_review.md
docs/ocean/experiments/advection_scheme_comparison.md
docs/ocean/experiments/density_jacobian_pgf_plan.md
docs/ocean/experiments/dissipation_research_synthesis.md
docs/ocean/experiments/etopo_instability_ocean_review.md
docs/ocean/experiments/density_jacobian_pgf_mpas.md
docs/ocean/experiments/silvestri2024_weno_iles_plan.md
docs/ocean/experiments/ocean_timestepper_plan.md
docs/ocean/experiments/bulk_flux_ly09_audit.md
docs/ocean/experiments/distributed_barotropic_pcg.md
docs/ocean/experiments/barotropic_gyre_design.md
docs/ocean/experiments/realistic_geometry_topology_fixes.md
docs/ocean/experiments/realistic_geometry_mpas_plan.md

exec
/usr/bin/bash -lc "rg -l -i '2\\.863|mixed-input|calibration-extrapolation|pole-bias|A2|26630576|above this MODEL' docs scripts tmp . 2>/dev/null | sed -n '1,160p'
for f in scripts/tmp/_probe_latlon_halo_census.py scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch; do
  if [ -f \""'$f" ]; then printf '"'\\n=== %s ===\\n' \""'$f"; nl -ba "$f" | sed -n '"'1,320p'; else printf 'MISSING %s\\n' \""'$f"; fi
done' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 644ms:
./CMIP.md
./tests/bench/test_scaling_metadata.py
./pyproject.toml
./docs/science/legoesm_documentation.tex
./docs/science/legoesm_documentation.html
./docs/COMPARE_REANALYSIS.md
./docs/science/legoesm_scientific_guide.tex
./uv.lock
./docs/science/specification.html
./docs/performance/scaling/bcw_scaling_status.md
./docs/science/specs/new_test_dycores.md
scripts/bench/bench_fv3_sw_fb_vs_production.py
./packages/ocean/legoesm/ocean/dynamics/latlon_cgrid_operators.py
./docs/performance/scaling/RESUME_STATE_2026-06-13.md
./packages/ocean/legoesm/ocean/dynamics/pgf_ahh08.py
./docs/performance/scaling/distance_to_limit_2026-06-13.md
./packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py
./docs/science/specs/FV3_3D.md
./docs/performance/scaling/cube_transport_tiling_design.md
./docs/science/specs/CRM_implementation.md
./docs/performance/scaling/derecho_levante_sota_review_2026-07.md
./packages/ocean/legoesm/ocean/physics/lateral_mixing/_gm_redi_common.py
./scripts/bench/bench_fv3_sw_fb_vs_production.py
./docs/performance/scaling/scaling_indicators.csv
./packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py
scripts/bench/bench_ocean_mpi_scaling.py
./packages/ocean/legoesm/ocean/dynamics/barotropic_cgrid.py
./packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py
./packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py
./docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md
./docs/performance/scaling/amip_mpi_scaling.md
./docs/performance/scaling/d2a2c_spmd_stage_design.md
./scripts/bench/bench_ocean_mpi_scaling.py
./packages/ocean/legoesm/ocean/physics/shortwave_penetration.py
./docs/performance/scaling/scaling.md
scripts/cluster/csw_oracle.sbatch
./tests/test_cases/colliding_modons.py
./docs/ocean/fidelity/oracle_recipe_strategy.md
docs/ocean/fidelity/oracle_recipe_strategy.md
./docs/performance/scaling/d2a2c_edge_specials_design.txt
./docs/ocean/fidelity/dino_pgf_alignment.md
./tests/validate/test_compare_amip_era5.py
docs/ocean/fidelity/dino_pgf_alignment.md
./docs/performance/scaling/levante_campaign_2026-07-24.md
./tests/test_cases/baroclinic_wave.py
./docs/performance/scaling/scaling_bottleneck_audit_2026-06-10.md
./docs/performance/scaling/cube_production_tiling_design.md
docs/architecture/fv3_single_implementation_program.md
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch
docs/land_canopy_amip_plan.md
./tests/test_atmosphere_cross_grid_plots.py
./packages/ocean/legoesm/ocean/physics/vertical_mixing/implicit_solver.py
./scripts/cluster/csw_oracle.sbatch
./docs/scaling/external_scaling_transfer_assessment.md
./docs/scaling/atm_latlon_spmd_scaling.md
./tests/unit/test_two_stream_faithful.py
./packages/ocean/legoesm/ocean/physics/vertical_mixing/kpp.py
./docs/architecture/fv3_single_implementation_program.md
./docs/land_canopy_amip_plan.md
./docs/ocean/fidelity/veros_acc_final_scoreboard.md
docs/ocean/fidelity/veros_acc_final_scoreboard.md
./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch
docs/validation/cubed_sphere_sw_504_506.md
docs/validation/TESTING.md
./packages/ocean/legoesm/ocean/experiments/dino.py
docs/land/multipool_som_phenology_plan.md
./docs/validation/cubed_sphere_sw_504_506.md
./docs/validation/TESTING.md
docs/land/clm_ml_s3_masked_vmap_plan.md
docs/land/lmip_biophys_runbook.md
./tests/test_fv3_d_sw5_corner_corrections.py
./docs/land/multipool_som_phenology_plan.md
./docs/land/clm_ml_s3_masked_vmap_plan.md
./docs/ocean/experiments/tripole_grid_implementation_status.md
docs/ocean/experiments/tripole_grid_implementation_status.md
./docs/land/lmip_biophys_runbook.md
./docs/ocean/experiments/jra55do_pipeline_plan.md
docs/ocean/experiments/jra55do_pipeline_plan.md
./docs/ocean/experiments/cubed_sphere_pgf_stability.md
docs/ocean/experiments/cubed_sphere_pgf_stability.md
./tests/ocean/fidelity/test_mitgcm_gyre_canonical.py
./tests/validation/scm_oracle/manifest.json
./packages/ocean/legoesm/ocean/eos.py
docs/dycore/fv3_native_p4c_oracle.md
docs/ocean/experiments/dissipation_research_synthesis.md
./docs/dycore/fv3_native_p4c_oracle.md
docs/dev-notes/ocean_faithfulness_nemo.md
./docs/ocean/experiments/dissipation_research_synthesis.md
docs/ocean/experiments/density_jacobian_pgf_mpas.md
./docs/ocean/experiments/density_jacobian_pgf_mpas.md
./docs/dev-notes/ocean_faithfulness_nemo.md
docs/dev-notes/faithful_latlon_FV.md
./docs/ocean/experiments/silvestri2024_weno_iles_plan.md
./tests/ocean/fidelity/test_oceananigans_internal_tide.py
docs/ocean/experiments/silvestri2024_weno_iles_plan.md
./docs/dev-notes/faithful_latlon_FV.md
docs/dev-notes/CRM_faithful_SAM.md
./docs/dev-notes/CRM_faithful_SAM.md
./tests/unit/test_conservative_regrid_curvilinear.py
docs/dev-notes/cubed_sphere_edge_artifacts.md
./docs/dev-notes/cubed_sphere_edge_artifacts.md
scripts/cluster/dchain_oracle.sbatch
./packages/ocean/legoesm/ocean/vertical.py
./tests/validation/bench_spectral_pe.py
docs/dev-notes/fv3_faithful.md
./docs/dev-notes/fv3_faithful.md
docs/dev-notes/OMIP_faithful.md
./docs/dev-notes/OMIP_faithful.md
scripts/cluster/d2a2c_oracle.sbatch
./docs/dev-notes/research/barotropic_noise_handling_in_production_models.md
docs/dev-notes/research/barotropic_noise_handling_in_production_models.md
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch
./docs/ocean/experiments/realistic_geometry_forcing_literature_review.md
docs/ocean/experiments/realistic_geometry_forcing_literature_review.md
./docs/dev-notes/planning/eady_eddy_resolving_ralph.md
docs/dev-notes/planning/eady_eddy_resolving_ralph.md
./tests/ocean/unit/test_fv3edge_barotropic.py
./docs/dev-notes/planning/silvestri_weno_reproduction_ralph.md
docs/dev-notes/planning/silvestri_weno_reproduction_ralph.md
./scripts/cluster/dchain_oracle.sbatch
docs/dev-notes/fv3_fortran_fidelity_review.md
./docs/dev-notes/fv3_fortran_fidelity_review.md
./tests/ocean/unit/test_bgc_carbonate_faithful.py
./scripts/cluster/d2a2c_oracle.sbatch
./docs/dev-notes/GPU_SCALING_BRANCH.md
docs/dev-notes/GPU_SCALING_BRANCH.md
./tests/core/test_streamfunction_freestream.py
./packages/land/legoesm/land/carbon/carbon_cycle.py
./tests/land/test_init_experiment.py
./packages/ocean/legoesm/ocean/config.py
scripts/matrix/scm/oracle.py
docs/dev-notes/FIX_RESTART_TIME.md
./docs/dev-notes/FIX_RESTART_TIME.md
./packages/land/legoesm/land/carbon/config.py
scripts/matrix/run_atmosphere_test_matrix.py
./tests/unit/test_interface_contract.py
./tests/land/validation/test_land_carbon_equilibrium.py
docs/COMPARE_REANALYSIS.md
scripts/run/run_correction_campaign.py
./tests/ocean/unit/test_gm_redi_latlon_cgrid.py
docs/performance/scaling/bcw_scaling_status.md
./packages/tools/legoesm/experiments/abstract.py
docs/performance/scaling/RESUME_STATE_2026-06-13.md
./packages/tools/legoesm/experiments/gradient_check.py
docs/performance/scaling/distance_to_limit_2026-06-13.md
docs/performance/scaling/cube_transport_tiling_design.md
./tests/unit/test_ccn_from_aod_faithful.py
docs/performance/scaling/derecho_levante_sota_review_2026-07.md
docs/performance/scaling/scaling_indicators.csv
./packages/core/legoesm/io/state_checkpoint.py
./tests/ocean/unit/test_nemo_bn2.py
./tests/ocean/unit/test_mpas_tke.py
./packages/core/legoesm/io/state_digest.py
./packages/land/legoesm/land/canopy/clm_ml_interface.py
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md
docs/performance/scaling/amip_mpi_scaling.md
docs/performance/scaling/d2a2c_spmd_stage_design.md
docs/performance/scaling/scaling.md
./data/les_cases/ARM9707/lsf
./packages/core/legoesm/parallel/cubesphere_exchange.py

=== scripts/tmp/_probe_latlon_halo_census.py ===
     1	"""Virtual-device HLO census of the atm lat-lon SPMD band step.
     2	
     3	Feeds the 'no analytic halo-message census for the atm latlon step yet'
     4	bound gap (audit item 4 follow-up): count collectives + estimate bytes
     5	per step from the COMPILED HLO on a forced-host-platform device mesh,
     6	using the bench's own builders (no re-derived model).
     7	
     8	CAVEAT (metadata.py:214): counts are flag- and backend-dependent — the
     9	default schedule reproduces from a CPU virtual-device compile, but
    10	GPU-only collective combining can LOWER the executed count. So counts
    11	here are census=virtual-cpu; a bound built on them is a MODEL.
    12	Structure check across nd: per-device collective count must be
    13	nd-independent for a 1-D band halo.
    14	
    15	Usage: python _probe_latlon_halo_census.py ND N_LAT [N_LON] [NLEV] [X64]
    16	(X64=1 measures the f64 census directly — codex r7: four 4-byte scalar
    17	CPs stay f32 under x64, so the x2 extrapolation is 16 B high.)
    18	"""
    19	import os
    20	import sys
    21	
    22	nd = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    23	n_lat = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    24	n_lon = int(sys.argv[3]) if len(sys.argv) > 3 else 2 * n_lat
    25	nlev = int(sys.argv[4]) if len(sys.argv) > 4 else 26
    26	if len(sys.argv) > 5 and sys.argv[5] == "1":
    27	    os.environ["JAX_ENABLE_X64"] = "1"
    28	
    29	os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
    30	                           + f" --xla_force_host_platform_device_count={nd}")
    31	os.environ["JAX_PLATFORMS"] = "cpu"
    32	
    33	HERE = os.path.dirname(os.path.abspath(__file__))
    34	sys.path.insert(0, os.path.join(HERE, "..", "bench"))
    35	
    36	import importlib
    37	import json
    38	
    39	import jax
    40	import numpy as np
    41	
    42	bench = importlib.import_module("bench_atm_latlon_spmd_scaling")
    43	from metadata import hlo_collective_census  # noqa: E402
    44	
    45	from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (  # noqa: E402
    46	    build_sharded_held_suarez_state_atm_latlon,
    47	    make_sharded_atm_latlon_step,
    48	)
    49	
    50	devs = jax.devices()
    51	assert len(devs) == nd, (len(devs), nd)
    52	mesh = jax.sharding.Mesh(np.array(devs), axis_names=("lat",))
    53	
    54	model = bench._build_model(n_lat, n_lon, nlev)
    55	c = build_sharded_held_suarez_state_atm_latlon(model.grid, model.sigma_coord, mesh)
    56	step = make_sharded_atm_latlon_step(model, mesh, physics_fn=None)
    57	
    58	census = hlo_collective_census(lambda s: step(s, 60.0), c)
    59	
    60	# Exact per-CP payload: sum collective-permute RESULT bytes from the
    61	# compiled HLO text (valid for synchronous CP: shape-preserving; the
    62	# regex matches bare CP and -start, never -done, so async pairs are not
    63	# double-counted).  Codex r7 hardening: compile ONCE, keep per-op
    64	# records, and assert the parsed count equals the census count so a
    65	# layout change fails loudly instead of silently under-summing.
    66	import re
    67	lowered = jax.jit(lambda s: step(s, 60.0)).lower(c).compile()
    68	txt = lowered.as_text()
    69	DT = {"f32": 4, "f64": 8, "bf16": 2, "f16": 2, "s32": 4, "u32": 4,
    70	      "pred": 1, "s8": 1, "u8": 1, "c64": 8, "c128": 16}
    71	cp_bytes = 0
    72	cp_ops = 0
    73	cp_records = []
    74	for m in re.finditer(r"collective-permute(?:-start)?\(", txt):
    75	    # walk back to the result shape at line start: '%name = TYPE[dims]{...} collective-permute'
    76	    line = txt[txt.rfind("\n", 0, m.start())+1:m.start()]
    77	    sm = re.search(r"(\w+)\[([0-9,]*)\]", line)
    78	    if not sm:
    79	        continue
    80	    dt, dims = sm.group(1), sm.group(2)
    81	    if dt not in DT:
    82	        continue
    83	    n = 1
    84	    for d in dims.split(","):
    85	        if d:
    86	            n *= int(d)
    87	    cp_bytes += n * DT[dt]
    88	    cp_ops += 1
    89	    cp_records.append({"dtype": dt, "dims": dims, "bytes": n * DT[dt]})
    90	if census is not None:
    91	    assert cp_ops == census["collective_permute"], (
    92	        f"parsed {cp_ops} CP shapes vs census {census['collective_permute']} "
    93	        "— HLO layout changed; per-op parse is under/over-counting")
    94	out = {
    95	    "cp_ops_with_shape": cp_ops,
    96	    "cp_result_bytes_per_dev_per_step": cp_bytes,
    97	    "cp_records": cp_records,
    98	    "nd": nd, "n_lat": n_lat, "n_lon": n_lon, "nlev": nlev,
    99	    "rows_per_dev": n_lat // nd,
   100	    "census_backend": "virtual-cpu (forced host platform)",
   101	    "census": census,
   102	}
   103	print(json.dumps(out))

=== scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch ===
     1	#!/bin/bash -l
     2	#SBATCH --job-name=ll128_comb
     3	#SBATCH --account=bb1596_gpu
     4	#SBATCH --partition=gpu
     5	#SBATCH --constraint=a100_80
     6	#SBATCH --nodes=32
     7	#SBATCH --gpus-per-node=4
     8	#SBATCH --exclusive
     9	#SBATCH --mem=0
    10	#SBATCH --time=01:30:00
    11	#SBATCH --output=ll128_comb.%j.log
    12	# CP-COMBINING A/B at LL2048@128 (the measured/bound ~2.9 gap).
    13	# The calibrated bound (census 41 CP + 1 AR/step, exact bytes 18.5 MB/dev,
    14	# IB 26.3us/23.5GB/s, nd=1 same-tile compute 1.659 ms) models 1.85-1.89 ms;
    15	# measured is 5.58 (job 26534060). Leading PLAUSIBLE mechanism: effective
    16	# per-CP overhead (launch+schedule+sync) >> raw fabric latency across 41
    17	# dependency-chained exchanges — the arXiv:2607.16100 regime. Lever:
    18	# COMBINE independent CPs into fewer, larger messages.
    19	# NOTE: the closed-levers null for these flags was the OCEAN lane
    20	# (reduction-dominated); this is the first atm-latlon test — not a rerun
    21	# of a closed null.
    22	# Falsifiability, BEFORE submit — arms byte-matched to 26534060 protocol:
    23	#   A control (default flags)      : expect ~5.6 ms
    24	#   B +cp-combine 8MB threshold    : CONFIRM lever if >=10% under A
    25	#   C +combine +pipelined-p2p      : scheduling interaction
    26	#   A2 control repeat  : drift bracket (codex r7)
    27	#   REFUTE if B,C within 2% of A/A2 -> refutes THIS threshold/impl only
    28	#   (no GPU post-pass census per arm); next = scheduling/overlap lever.
    29	set -uo pipefail
    30	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    31	export JAX_PLATFORMS=cuda,cpu
    32	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    33	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    34	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    35	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    36	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll128_comb_j${SLURM_JOB_ID}}"
    37	mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
    38	rc=0
    39	run_arm () { # tag extra_xla_flags
    40	  echo "=== LL2048@128 f32 arm=$1 XLA_EXTRA='$2' ==="
    41	  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
    42	      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    43	    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
    44	      --multicontroller --n-devices 128 --mode strong \
    45	      --n-lat 2048 --n-lon 4096 --nlev 26 --steps 12 --warmup 3 \
    46	      --out "$OUTDIR/$1.jsonl" || { echo "$1 FAILED"; rc=1; }
    47	}
    48	run_arm A_default ""
    49	run_arm B_combine "--xla_gpu_collective_permute_combine_threshold_bytes=8388608"
    50	run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"
    51	run_arm A2_default ""
    52	echo "=== RESULTS ==="
    53	for T in A_default B_combine C_combine_pipelined A2_default; do
    54	  "$PY" -c "
    55	import json,math,sys
    56	try:
    57	    d=json.loads(open('$OUTDIR/$T.jsonl').readline())
    58	    ms=d['steady_median_ms']; assert math.isfinite(ms) and ms>0
    59	except Exception as e:
    60	    print('$T: MISSING/INVALID ->', e); sys.exit(1)
    61	print(f'$T: {ms:8.3f} ms')" || { echo "$T receipt invalid"; rc=1; }
    62	done
    63	echo "DONE rc=$rc"; exit $rc

exec
/usr/bin/bash -lc "for f in docs/scaling/atm_latlon_spmd_scaling.md docs/performance/scaling/levante_campaign_2026-07-24.md docs/performance/scaling/bcw_scaling_status.md; do
  printf '\\n=== %s (tail) ===\\n' \""'$f"; tail -n 260 "$f" | nl -ba -v "$(($(wc -l < "$f")-259))";
done' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 233ms:

=== docs/scaling/atm_latlon_spmd_scaling.md (tail) ===
  -210	# Lat-band SPMD scaling — lat-lon C-grid hydrostatic atm step (A1)
  -209	
  -208	Scaling of `make_sharded_atm_latlon_step` / `run_atm_latlon_spmd` (the
  -207	single-process multi-device lat-band decomposition of the lat-lon C-grid
  -206	hydrostatic dycore). Bench: `scripts/bench/bench_atm_latlon_spmd_scaling.py`;
  -205	plot: `scripts/plot/plot_atm_latlon_spmd_scaling.py`.
  -204	
  -203	All numbers are steady-state min ms/step (compile excluded); JAX_ENABLE_X64.
  -202	
  -201	## Perf prerequisite (jit-cache)
  -200	
  -199	The bench surfaced a real perf bug: a bare `shard_map` is **not**
  -198	compilation-cached, so the un-jitted band step recompiled on **every** call —
  -197	~142 s/step at 32×64×10 nd=2 (job 8560671), and the whole equivalence gate ran
  -196	~65 min. Fixed by passing `dt` as a traced operand + wrapping the shard_map in
  -195	`jax.jit` built once (commit `fa2ce32b6`): first call compiles, the rest hit the
  -194	cache (probe 8561202: `[3079, 1.4, 1.2, 1.1, 1.1]` ms). The full 17-test SPMD
  -193	suite then runs in 129 s (was ~65 min); equivalence unchanged.
  -192	
  -191	## GPU (Ginsburg, 2 GPU/node) — job 8561259
  -190	
  -189	| mode | grid | nd=1 | nd=2 | result |
  -188	|------|------|------|------|--------|
  -187	| strong | 128×256×30 (fixed) | 6.09 ms | 5.56 ms | **1.10× speedup** |
  -186	| weak   | 64 lat/dev | 3.51 ms (64×256×30) | 5.51 ms (128×256×30) | **0.64 efficiency** |
  -185	
  -184	Modest, sub-linear: at this size the per-GPU work is small and the cross-band
  -183	`ppermute` halo (over the node's PCIe interconnect) dominates the gain. This
  -182	matches the prior Ginsburg 2-GPU practical-limit findings for the ocean / BCW
  -181	campaigns — the lat-band atm SPMD is correct and positive-scaling, but the
  -180	2-GPU PCIe fabric caps strong scaling. Larger per-device grids (more compute per
  -179	halo byte) and >2 devices are the levers for better efficiency.
  -178	
  -177	> **Update:** the multi-node `jax.distributed` path IS wired now —
  -176	> `bench_atm_latlon_spmd_scaling.py --multicontroller` (route-B, native NCCL
  -175	> ppermute, no mpi4jax). Derecho/Levante job lanes exist; production-scale
  -174	> numbers are the remaining measurement gap. See
  -173	> `docs/performance/scaling/SCALING_STATUS_AUDIT.md`.
  -172	
  -171	## CPU virtual devices (characterization, NOT speedup) — job 8561216
  -170	
  -169	| mode | grid | nd=1 | nd=2 | nd=4 |
  -168	|------|------|------|------|------|
  -167	| strong | 64×128×20 | 9.85 ms | 11.05 ms (0.89×) | 10.85 ms (0.91×) |
  -166	
  -165	`--xla_force_host_platform_device_count=N` puts all "devices" on the same
  -164	physical CPU (threads sharing cores) → no real parallelism; the ~10% slowdown is
  -163	the (small) `ppermute` halo overhead. Useful only to confirm correctness at
  -162	device scale + bound the comm cost, not as a speedup.

=== docs/performance/scaling/levante_campaign_2026-07-24.md (tail) ===
  1591	(`.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md`,
  1592	VERDICT FIX-FIRST, 12 items — the round that caught a Lloyd-mesh confound
  1593	in the first draft's weak-scaling claim).
  1594	
  1595	### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094)
  1596	
  1597	Matrix at a NOMINAL MEAN target of 5,120 cells/rank (the JSONL's
  1598	`cells_per_rank_achieved` is global floor division —
  1599	`int(mesh.nCells) // n_ranks`, bench_ocean_mpas_scaling.py:751 — NOT a
  1600	balance statement; method pinned per arm, never `auto`, which flipped
  1601	meaning when pymetis appeared in `.venv-mpi` on 2026-07-31), f64,
  1602	nlev 20, 32 ranks/node. Actual per-rank OWNED ranges
  1603	(`metadata.partition_metrics.cells_per_rank_min/max`): geometric
  1604	5,120–5,121 at BOTH scales; metis 5,100–5,145 @32 and 5,093–5,144 @128
  1605	(±0.5 %). WET load is looser still under metis — per-rank wet
  1606	cell-levels min/max: geometric@32 100,740–102,420; metis@32
  1607	96,800–102,720; metis@128 **78,000–102,880** (one rank 24 % under the
  1608	mean) — METIS balances owned cells (approximately), not wet cells, on
  1609	this bathymetry.
  1610	
  1611	| arm | config | ms/step |
  1612	|---|---|---|
  1613	| A | s7 np32 geometric, block:cyclic | 189.82 |
  1614	| B | s8 np128 geometric, block:cyclic | 308.96 |
  1615	| C | s7 np32 metis, block:cyclic | 191.99 |
  1616	| D | s8 np128 metis, block:cyclic | 333.39 |
  1617	| E | s8 np128 metis, block:block | 537.91 |
  1618	
  1619	* **This METIS configuration LOSES to geometric at s8/np128** (D/B =
  1620	  +7.9 %), despite the better offline cut (partq s8@np128: edge_cut
  1621	  1.89 % vs 2.01 %, halo mean 592 vs 629). Scale-out term (s7@32 ->
  1622	  s8@128, which crosses 1 -> 4 nodes as well as 4x ranks — NOT a pure
  1623	  rank-count isolate): geometric 1.628, metis 1.736. The offline-quality
  1624	  -> step-time inference FAILS on this lane; part of metis's loss is
  1625	  PLAUSIBLY its own wet-load imbalance (above). Scope: closes the
  1626	  "swap in METIS as-is" lever on the CPU-MPI ocean lane; does NOT rule
  1627	  out partition/mapping improvements generally (e.g. wet-cell-weighted
  1628	  METIS was NOT tested).
  1629	* **`block:cyclic` stays mandatory on packed CPU lanes** (E/D = 1.61x at
  1630	  a byte-identical partition). NOTE the second `--distribution` field is
  1631	  the INTRA-NODE (socket) distribution — both arms place ranks on nodes
  1632	  identically; the swing is socket-level. Mechanism (per-socket
  1633	  memory-bandwidth balance) PLAUSIBLE, consistent with the np16 Milan
  1634	  2.13x receipt; never instrumented with bandwidth counters.
  1635	* Caveats: timing-only receipt — no parity/conservation gate ran in
  1636	  these arms, and the CPU nodes emit `UCX WARN transports
  1637	  'cuda_copy','cuda_ipc','gdr_copy' are not available` (the _env.sh GPU
  1638	  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
  1639	  timing, but a "production config" claim would need a gated arm).
  1640	
  1641	### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)
  1642	
  1643	f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
  1644	scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
  1645	L26; steps 12 / warmup 3; physics=none dynamics-only bench. Provenance:
  1646	np64 and np128 rows record `git_sha: 7151d12a1`; the np32 row's field
  1647	reads `unknown` — same allocation, same submitted script, so the same
  1648	binary is PLAUSIBLE but that row stays non-reproduction-grade on its
  1649	own (codex r20/r21):
  1650	
  1651	| GPUs | cells/GPU | ms/step | GC/s (cell-levels) |
  1652	|---|---|---|---|
  1653	| 32 | 81.9k | 12.47 | 5.47 |
  1654	| 64 | 41.0k | 9.60 | 7.10 |
  1655	| 128 | 20.5k | 11.48 | 5.94 |
  1656	
  1657	* **np64 = 7.10 GC/s is the best MPAS-atmosphere number on this
  1658	  synthetic dynamics-only bench** (2.19x the subdiv-8 best, 3.23 GC/s at
  1659	  its np64: 655,362 natural cells x 26 lev / 5.27 ms).
  1660	* Strong 32->64 speedup 1.299 (eff 0.65); 64->128 speedup 0.836 —
  1661	  ANTI-scales at 20.5k cells/GPU. CONSISTENT WITH the ~30k floor seen on
  1662	  the other lanes (single unreplicated point on a lane with known
  1663	  count-specific codegen variation — not by itself proof).
  1664	* **RETRACTED (codex round-20): the first draft's s8->s9 "weak
  1665	  efficiency 0.55–0.74" pairs and the "~1.4x per 4x ranks GPU rank-count
  1666	  term".** Confounds: (a) every existing s8 receipt is the generator's
  1667	  default PRODUCTION Lloyd mesh, while s9 is lloyd=0 — different mesh
  1668	  family, not the same protocol; (b) two comparator points came from the
  1669	  np2-16 ladder (jobs 26454476/26454618), not the np32-128 extension
  1670	  rows; (c) the three ratios are 1.80/1.35/1.41 — not "consistent
  1671	  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
  1672	  no weak-scaling direction is claimed until it lands.
  1673	
  1674	### Next receipts submitted 2026-08-02
  1675	
  1676	1. **Ensemble receipt** (`scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch`)
  1677	   — codex lever #1: 4 concurrent 32-GPU s9 replicas on disjoint 8-node
  1678	   sets vs SAME-JOB solo controls bracketing phase B (solo before AND
  1679	   after — BRACKETED, not fully counterbalanced; a penalty's attribution
  1680	   to fabric vs placement/drift needs the per-step nodelist table +
  1681	   follow-up). steps=5000 so the stepping window
  1682	   (~60 s) dwarfs launch skew; per-arm `SLURM_STEP_NODELIST` +
  1683	   wall-clock brackets logged as overlap evidence. CONFIRM bar:
  1684	   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
  1685	   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
  1686	   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
  1687	   >10 % = a CO-EXECUTION penalty, quantified per replica — its
  1688	   attribution (fabric contention vs placement/topology vs drift) is a
  1689	   follow-up, not a conclusion of this job.
  1690	2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
  1691	   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
  1692	   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
  1693	   recomputed only from these.
  1694	
  1695	### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)
  1696	
  1697	LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
  1698	@64 row (job 26502539, f32 6.73 ms):
  1699	
  1700	| arm | ms/step | GC/s (col-levels) |
  1701	|---|---|---|
  1702	| f32 @128 (65,536 cols/GPU) | 5.5767 | **39.11** |
  1703	| f64 @128 | 9.6015 | 22.72 |
  1704	
  1705	f32 strong 64->128: 1.207x for 2x devices (eff 0.60) with the tile at
  1706	**65.5k cols/GPU — comfortably ABOVE the ~30k floor** (codex round-21
  1707	caught the first draft halving this), so the loss is NOT
  1708	floor-attributable. Mechanism OPEN — candidates (uninstrumented): 1-D
  1709	band thinning to 16 rows/rank raising halo/compute ratio, and the
  1710	16 -> 32-node NCCL topology step. 39.11 GC/s (from 5.5767 ms) is the
  1711	highest measured throughput of ANY lane in the campaign. The companion
  1712	oc128 (26534067) FAILED pre-#1370-fix with the 109.5 GB resident-args
  1713	signature; retry submitted post-fix (below).
  1714	
  1715	## Hundreds-of-devices push (user directive 2026-08-02)
  1716	
  1717	"Push the scaling to hundreds of CPUs and GPUs for lat-lon and MPAS on
  1718	GPUs." Machine ceiling: 56 nodes x 4 = 224 a100_80 GPUs; compute
  1719	partition effectively unbounded for our rank counts. Submitted set:
  1720	
  1721	| job | what | devices | why |
  1722	|---|---|---|---|
  1723	| 26628196 | s9 ensemble contention (v3; 26628021/26627810 superseded pre-start) | 128 GPU (4x32) | lever #1 receipt |
  1724	| 26628071 | oc LL2304 retry post-#1370 | 128 GPU | pre-fix failure was resident-args; predicted PASS at ~0.10 GB/dev residency |
  1725	| 26628072 | atm LL2304 @96/@192 + LL2880 @192 | 96-192 GPU | LL2048 does not divide 192; LL2880@192 = 86.4k cols/GPU ABOVE floor |
  1726	| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
  1727	| 26628074 | subdiv-10 lloyd0 prewarm | 1 CPU | unlocks MPAS 128-224 GPUs ABOVE floor (81.9k-46.8k cells/GPU) |
  1728	| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
  1729	
  1730	s10 ladder (128/192/224 GPUs) submits once 26628074's cache lands.
  1731	
  1732	### First hundreds receipt in: lat-lon CPU 2-D pencil to 512 ranks (job 26628073)
  1733	
  1734	r512 (512x1024 = 524k cols) L26 f64 moist, 32 rpn block:cyclic,
  1735	wall-pole 2-D pencil lane (labelled; NOT the pole fold):
  1736	
  1737	| ranks | cols/rank | ms/step | speedup vs np64 | eff |
  1738	|---|---|---|---|---|
  1739	| 64 | 8,192 | 297.57 | 1.00 | 1.00 |
  1740	| 128 | 4,096 | 161.03 | 1.848 | 0.92 |
  1741	| 256 | 2,048 | 72.06 | 4.129 | 1.03 |
  1742	| 512 | 1,024 | 44.78 | 6.645 | **0.83** |
  1743	
  1744	Distribution verified against the masquerade trap: result rows carry
  1745	`n_ranks: 512` (the JSON's `metadata.process_count: 1` is the jax-LOCAL
  1746	count on this mpi4jax lane, not the world size). 128->256 is
  1747	SUPERLINEAR (2.23x for 2x) — classic per-rank working-set cache
  1748	transition on Milan (mechanism PLAUSIBLE, uninstrumented). End-to-end
  1749	64->512 eff 0.83 at 1k cols/rank: TIMING-ONLY evidence that the lat-lon
  1750	CPU lane scales into the hundreds. QUALIFIER (codex r22): the run logs
  1751	an out-of-tested-range mpi4jax==0.9.0 pairing ("may fail or produce
  1752	incorrect results", parallel/reductions.py runtime check) and UCX
  1753	VM_UNMAP warnings — no parity/conservation gate ran, so this ladder is
  1754	unvalidated timing evidence until a supported-stack rerun. (Exact pencil factorisations are not recorded in the
  1755	result JSON — only `decomposition: 2d`; a follow-up could add them to
  1756	the bench metadata.)
  1757	
  1758	### s8 lloyd=0 de-confound ladder landed (job 26628076): the near-matched-tile scale-out cost persists without the Lloyd confound
  1759	
  1760	s8 np8/16/32, lloyd=0, f32, sfc + `--reorder-for 128`, steps 12 /
  1761	warmup 3 — configuration-matched to the s9 ladder (26600095); NOT fully
  1762	reproduction-grade: the s9 np32 row's git_sha reads `unknown`, and this
  1763	run's `7151d12a1-dirty` has no archived dirty-file manifest (the live
  1764	diff touched only doc+plot, which supports but cannot retrospectively
  1765	prove the bench path was untouched): **6.58 / 6.43 / 7.29 ms**.
  1766	
  1767	Weak pairs (~4x cells with 4x GPUs — global ratio 3.9994 after both
  1768	meshes pad +126 cells; tiles NEAR-matched to 0.015 %:
  1769	81,936/81,924, 40,968/40,962, 20,484/20,481), SAME lloyd-0 family:
  1770	
  1771	| cells/GPU | s8 rung | s9 rung | ratio | weak eff |
  1772	|---|---|---|---|---|
  1773	| 81.9k | np8 6.58 | np32 12.47 | 1.895 | **0.53** |
  1774	| 41.0k | np16 6.43 | np64 9.60 | 1.493 | 0.67 |
  1775	| 20.5k | np32 7.29 | np128 11.48 | 1.575 | 0.64 |
  1776	
  1777	* The falsifiability block's CONFIRM branch fires: ratios stay well
  1778	  above 1 with the known Lloyd-family mismatch REMOVED. (This does not
  1779	  prove the old confound "only" biased the size — these are
  1780	  unreplicated single runs from separate allocations, one comparator
  1781	  without row-level provenance; the confounded draft read
  1782	  1.80/1.35/1.41 vs 1.90/1.49/1.57 here, and the production-mesh s8
  1783	  np8 was 6.92 vs lloyd-0 6.58, -4.9 %, so the mesh family does shift
  1784	  absolutes.)
  1785	* Restated: at NEAR-matched per-GPU tile, ~quadrupling devices+problem
  1786	  costs 1.5-1.9x on this lane — the GPU-side analogue of the ocean CPU
  1787	  scale-out term. Weak efficiency 0.53-0.67 at 4x. Mechanism still
  1788	  UNATTRIBUTED (PLAUSIBLE candidates unchanged: inter-node neighbour
  1789	  fraction growth, collective latency vs count, sfc partition-quality
  1790	  decay with parts; the metis receipt argues against pure
  1791	  partition-cut explanations, on the CPU lane at least).
  1792	* The non-monotone tile dependence of the ratio (largest at the
  1793	  LARGEST tile, 1.90 at 81.9k) is unexplained; recorded, not theorised.
  1794	
  1795	## Distance-to-modeled-limit: atm lat-lon GPU (2026-08-02, "near theoretical limit" directive)
  1796	
  1797	Closed the bench's own honest-null bound gap (audit item 4) for the
  1798	lat-lon lane, using only repo instruments:
  1799	
  1800	* **Halo census** (new probe `scripts/tmp/_probe_latlon_halo_census.py`,
  1801	  virtual-CPU forced-host-platform lowering of the REAL
  1802	  `make_sharded_atm_latlon_step`): **41 collective-permutes + 1
  1803	  all-reduce per step**, nd-INDEPENDENT (identical at nd=8 and nd=16 —
  1804	  the 1-D band structure check). Exact CP payload from compiled-HLO
  1805	  result shapes: 4,635,408 B/dev/step at n_lon=1024 L26 f32 = 1.06x the
  1806	  single-row slab model; linear in n_lon (checked 1024 vs 2048, 0.07 %
  1807	  residual) -> **18.5 MB/dev/step at n_lon=4096 f32**. CAVEAT: CPU
  1808	  lowering; GPU-side collective combining could change the executed
  1809	  count (metadata.py:214) — the bound is a MODEL.
  1810	* **Same-tile nd=1 compute baselines** (job 26630370, roofline recipe):
  1811	  16x4096 f32 1.659 ms, 32x4096 f32 2.837, 16x4096 f64 2.973.
  1812	  Approximation, recorded: nd=1 includes pole tiles; the bias
  1813	  DIRECTION on the compute term is PLAUSIBLE-high, not proven
  1814	  (matters most for the compute-dominated f64 row).
  1815	* **Calibrated bound** (`metadata.calibrated_bound`, measured fabric
  1816	  constants: IB 26.3 us / 23.5 GB/s, NVLink 17.8 / 64.2):
  1817	
  1818	| row | measured | t_bound (IB) | measured/bound |
  1819	|---|---|---|---|
  1820	| LL2048@64 f32 | 6.732 | 2.863 | **2.35** |
  1821	| LL2048@128 f32 | 5.577 | 1.894 | **2.94** |
  1822	| LL2048@128 f64 | 9.602 | 2.999 | **3.20** |
  1823	
  1824	(Codex r7 corrected the @128 f32 row: the first draft fed the slab-byte
  1825	lower bound into a table labelled exact-bytes — 1.848/3.02 was the
  1826	mixed-input artefact; with the exact 18,541,632 B payload the bound is
  1827	1.894 ms. r7 also independently RERAN the census at nd=128 — 41 CP + 1
  1828	AR confirmed at the target device count, not just extrapolated from
  1829	8/16 — and measured the f64 census directly: 9,270,800 B at n_lon=1024,
  1830	four 4-byte scalar CPs staying f32, so the x2 extrapolation was 16 B
  1831	high.)
  1832	
  1833	* **The lat-lon GPU panel sits ~2.4-3.2x ABOVE this MODEL** (2.35-3.20)
  1834	  — the eff-0.60 strong leg is not close to the fabric+compute MODEL
  1835	  (a heuristic, not a proven floor). The 2-node/8-process IB
  1836	  calibration is extrapolated to a 32-node/128-process communicator. Leading
  1837	  PLAUSIBLE mechanism (uninstrumented): effective per-CP cost
  1838	  (launch + schedule + stream sync) well above the raw 26 us fabric
  1839	  latency across 41 dependency-chained exchanges — the arXiv:2607.16100
  1840	  small-collective regime. The model itself notes the serialized-latency
  1841	  vs overlap biases pull opposite ways; treat measured/bound as a
  1842	  consistency diagnostic, not proven headroom.
  1843	* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
  1844	  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
  1845	  same-job control + trailing A2 drift bracket. Interpretation limit:
  1846	  without a GPU post-pass CP census per arm, a null refutes THIS
  1847	  threshold/implementation, not combinable-CP count in general. The
  1848	  ocean-lane null for these flags came from a different
  1849	  implicit-PCG/dependency mix — not predictive for the atm lane either
  1850	  way.

=== docs/performance/scaling/bcw_scaling_status.md (tail) ===
   132	| moist f64 | 31.5 | 106.7 | **0.85** |
   133	| dry  f32 | 54.1 | 140.3 | 0.65 |
   134	| moist f32 | 40.3 | 116.9 | 0.73 |
   135	
   136	Weak efficiency 0.85–0.90 in f64 across 4× the nodes (np64=8 nodes → np256=32
   137	nodes) — the multi-node fix scales weakly, not just strong. f32 is lower (0.65–
   138	0.73): halving the compute makes the fixed halo/latency a larger fraction, so f32
   139	goes comm-bound sooner — the same fabric wall, reached at a smaller compute
   140	budget. This matches the SOTA picture (MPAS/MOM6 weak E falls with thinner
   141	arithmetic intensity on a latency-bound interconnect).
   142	
   143	### Status of every grid × precision toward its theoretical limit
   144	
   145	| grid | precision | multi-node | limit reached | residual |
   146	|---|---|---|---|---|
   147	| atm icosahedral | f64 & f32 | **np8..256 ✓** | yes — res-dependent strong peak (np128 @ res6), fabric comm-bound past it | none (fabric-bound) |
   148	| ocean lat-lon | f64 & f32 | **np1..128 ✓** | yes — res-dependent strong peak, same fabric wall | none (fabric-bound) |
   149	| atm lat-lon (FV) | f64 & f32 | band np..128 ✓ | yes — band fabric-optimal on Gloo; 2-D pencil loses (latency-bound) | 2-D win needs InfiniBand |
   150	| atm cubed-sphere | f64 & f32 | ≤6 faces/node | partial | >6-device sub-face tiling = future-HW project |
   151	| atm spectral | f64 | single-device | n/a | no tracer storage (moist) + no MPI path — large additions |
   152	| GPU multi-device (jax-mesh SPMD) | f32 & f64 | **WORKS intra-node** | 1.11x @ 2-GPU C192 | single-process jax device-mesh (commit 001b18bcb); PCIe-bound (no NVLink), crossover ~C192. See GPU UPDATE below |
   153	| GPU multi-device (mpi4jax / jax.distributed) | f32 & f64 | **blocked** | n/a | mpi4jax CPU-only build; jax.distributed multi-controller NCCL topology times out — infra, not a code gap |
   154	
   155	The CPU-decomposable grids (atm icosahedral, ocean lat-lon, atm lat-lon band)
   156	are characterised to their fabric limit on Ginsburg. The open items are
   157	capability gaps (spectral moist, cubed-sphere sub-face tiling) or infra blocks
   158	(CUDA-aware mpi4jax), not algorithmic scaling bugs.
   159	
   160	## Where each configuration stands
   161	
   162	| component / grid | multi-node path | weak E | strong E | peak Mc/s·dev | verdict |
   163	|---|---|---|---|---|---|
   164	| **atm icosahedral** (MPAS) | CPU-MPI + GPU MPI, full ladder | **0.80** (f32) | 0.14–0.63 | **~107** | near practical limit |
   165	| **ocean lat-lon** (C-grid) | CPU band + GPU SPMD | **0.92** | 0.20–0.92 | ~105 | near practical limit |
   166	| **atm lat-lon** (FV C-grid) | CPU band + 2-D pencil | 0.05 @128 (band) | 0.04–0.11 | ~49 | band is **fabric-optimal on Gloo**; 2-D built+validated but loses here (latency-bound, see below) |
   167	| **atm cubed-sphere** (FV3) | SPMD ≤6/node | — | — | ~46 (1 dev) | single-node only |
   168	| **atm spectral** | none | — | — | — | single-device (no MPI) |
   169	
   170	(weak E = per-rank-throughput retention, ideal 1; strong E = speedup/ideal at
   171	the largest resolution with ≥2 device points.)
   172	
   173	## Interpretation — have we reached the limit?
   174	
   175	- **atm icosahedral** and **ocean**: yes, at the practical limit for this
   176	  hardware. Weak efficiency 0.80–0.92 with per-device throughput ~105–107
   177	  Mcells/s. The residual gap to E=1 is the halo-exchange overhead, which is
   178	  **latency-bound**: `halo_exchange_voronoi._exchange_mpi` /
   179	  `exchange_halo_latlon` issue one serialized mpi4jax `sendrecv` per neighbour
   180	  over Gloo on PCIe-Gen3 (no NVLink / InfiniBand). Field-batching the halo
   181	  (one message per neighbour for all prognostic fields) is already the default;
   182	  the remaining per-neighbour serialization is a library/fabric limit, not an
   183	  algorithmic one. This matches the prior ocean-campaign "convergent practical
   184	  limit" verdict.
   185	
   186	- **atm lat-lon**: the 1D latitude-band decomposition starves at high rank
   187	  count (each rank gets few lat rows; halo perimeter dominates → weak E 0.05 at
   188	  128 ranks). The SOTA fix is a 2D pencil decomposition (MOM6/E3SM) — now
   189	  **built, validated, and measured** (below). **Verdict: on this Gloo/PCIe
   190	  fabric the band is OPTIMAL and the 2-D pencil LOSES.** The band keeps
   191	  longitude LOCAL (ZERO lon messages — only N/S, one *direction*); the 2-D
   192	  pencil adds an E/W direction. On a latency-bound fabric the 267 µs sendrecv
   193	  floor dominates the payload (even a full-lon N/S message is ~343 µs ≈ one
   194	  floor), so the minimum-message-DIRECTION decomposition wins: band (1
   195	  direction) beats 2-D (2 directions) **regardless of np or fusion**. Measured
   196	  (job 8503081, strong res=64 f64): 2-D vs band SYPD np4 3.38/3.31, np8
   197	  2.95/4.05, np16 0.47/2.22. The np16 4.7× gap is inflated by the current
   198	  UNFUSED per-operator `exchange_halo_lon` (one lon exchange per lon-padding
   199	  op); a lon-halo fusion (one exchange/step) would narrow it to ~1.5× (the
   200	  single extra E/W floor) but NOT flip the verdict — 2 directions still lose to
   201	  1 when latency ≫ bandwidth. **The 2-D pencil wins only when payload ≫ latency
   202	  (RDMA/InfiniBand, or very high resolution) — the same fabric wall the roofline
   203	  already identified.** So atm-lat-lon is fabric-bound like the others; the band
   204	  is its fabric-optimal decomposition on Ginsburg.
   205	
   206	## Levers evaluated this campaign
   207	
   208	| lever | result |
   209	|---|---|
   210	| moist on MPAS (Kessler, column-local) | shipped — moist now scales on the ico ladder |
   211	| column-local pre-physics halo skip (Kessler) | correctness-neutral; perf null at tested np (compute-bound regime); helps only deep in the latency-bound regime |
   212	| batched Voronoi halo (field-batched) | already default (prior campaign) |
   213	| METIS vs RCB partition | dead — RCB already balanced on uniform mesh (prior campaign) |
   214	| mixed precision (FP32 work) | atm ico FP32 ~107 Mc/s vs FP64 ~47; FP32 ~2× as expected |
   215	
   216	## Remaining gaps = architectural projects (not quick levers)
   217	
   218	1. **atm lat-lon 2D pencil decomposition — BUILT + VALIDATED + MEASURED
   219	   (task #14 done).** Full wall-pole 2-D C-grid step: `LatLon2DLayout`,
   220	   `pad_halo_latlon_2d`, `exchange_halo_lon`, the operator lon-op conversion
   221	   (`pad_lon_cgrid`: gradient_x / interp_uface / curl-v / mask ops +
   222	   `absolute_vorticity_coriolis`), u-face scatter/gather convention,
   223	   `make_latlon_2d_mpi_step`, and the `run_cpu_mpi_scaling --latlon-2d` harness.
   224	   Validated: dycore gate `test_latlon_2d_mpi_step.py` (2×2 == 1×4, mass
   225	   < 1e-12, decomposition-invariant) + operator equivariance np={2,3,6} + codex
   226	   (multiple rounds). **Result: the band is fabric-optimal on Gloo; the 2-D
   227	   pencil loses here** (latency-floor analysis above — 2 message directions
   228	   can't beat 1 when latency ≫ bandwidth). Remaining (fabric-gated, like the
   229	   GPU-multi infra block): (a) the 2-D win requires RDMA/InfiniBand — re-measure
   230	   there to demonstrate it; (b) a lon-halo FUSION would narrow the Gloo gap
   231	   4.7×→~1.5× but not flip it (deferred — no Gloo payoff); (c) the atmosphere's
   232	   180° pole-FOLD under a lon split still needs a lat-pencil transpose
   233	   (`pole_bc="fold"` raises; the shipped path is wall-pole only — a labeled
   234	   midlatitude throughput benchmark, NOT atm-pole-correct). Ocean (wall poles)
   235	   could use 2-D but does not need it (band weak E already 0.92).
   236	2. **cubed-sphere multi-device.** No SPMD sub-face tiling beyond 6 faces;
   237	   separate capability (see prior `omip_tiled_d2a2c_kernels` work).
   238	3. **spectral moist — DONE (capability; single-device, no scaling).** Moist
   239	   physics (q_v/q_c/q_r + Kessler warm-rain) now runs on the global spectral PE
   240	   dycore, so moisture is wired on ALL atmosphere grids (icosahedral,
   241	   cubed-sphere, lat-lon, spectral). Shipped: `make_kessler_forcing_spectral`
   242	   (commit cc85018ee) — an SH-transform bridge to the SHARED column Kessler
   243	   (inverse-SH `T_hat`→grid + `p_s=exp(synthesis(lnps_hat))`, flatten to
   244	   `(ncol,nlev)` so `pressure_from_sigma`/`compute_rho`/`compute_layer_dz` +
   245	   `kessler_microphysics` apply unchanged, forward-SH the latent-heating rate
   246	   back to `T_hat`); the `moist=` IC on `baroclinic_wave_init_spectral` + the
   247	   `--grid spectral --physics moist` harness path (commit d7b61a8a3). Bugfix in
   248	   the same commit: the timing scan passed `dt` as a jit argument → tracer, and
   249	   the spectral dycore caches integrator/filter matrices keyed on a CONCRETE dt
   250	   (`_ensure_tracer_filter`/`_ensure_si_data`) → `TracerBoolConversionError`;
   251	   `dt` is now closed over as a static float. Tests 7/7 + clean T21 e2e.
   252	
   253	   **Spectral throughput (single device, CPU, the theoretical limit):** it is
   254	   spherical-harmonic-transform bound — O(N³) Legendre transforms dominate, so
   255	   per-device throughput is ~0.2–0.6 Mcells/s (≈100× below the grid-point ico
   256	   ~40–150 Mc/s) and collapses with truncation: dry f64 T21 97 ms/step → T42
   257	   459 ms → T85 3.15 s; moist ≈2× (extra tracer transforms). There is no MPI
   258	   path, so spectral does not scale across devices here; the limit is
   259	   algorithmic (global transforms), reached.
   260	
   261	## The np16 -> np32 (2^4 -> 2^5) MPAS cliff — root cause + fix
   262	
   263	Symptom: MPAS/icosahedral CPU strong scaling DROPS across the 16->32 rank
   264	boundary on one node (I5 strong f64: np16 = 33.9 ms/step, np32 = 76.0 ms —
   265	2.24x SLOWER at 2x ranks; weak-eff cratered 0.80 -> 0.07).
   266	
   267	Ginsburg nodes = 2 sockets x 16 cores, so np16 fills exactly one socket and
   268	np32 spans both — which looks like a NUMA cross-socket cliff. But codex
   269	adversarial review + the A/B refute pure-NUMA: the **hybrid 16r x 2c config also
   270	spans both sockets yet recovers to 34 ms** (2.2x). So the binding mechanism is
   271	not cross-socket *memory*; it is **rank count** — 32 single-threaded MPI ranks
   272	on one node hammer the mpi4jax/Gloo path (267.8 us sendrecv latency floor x
   273	per-RK-stage halo x 32 ranks, plus MPI-progress starvation when every core is a
   274	rank). Fewer ranks => fewer messages => the cliff disappears.
   275	
   276	Fix (shipped): run **fewer ranks x more cores/rank** per node. Measured I5
   277	strong f64, 32 cores/node: 32r x1c = 76 ms; 16r x2c = 34 ms (2.2x); **8r x4c =
   278	28.8 ms (2.64x, optimum)**; 4r x8c = 30 ms; 2r x16c = 40 ms.  Enabled by
   279	`run_cpu_mpi_scaling._configure_jax_cpu` becoming cpus-per-task-aware (multi-
   280	threaded Eigen when SLURM_CPUS_PER_TASK>1; single-thread when =1).  Scaling is
   281	now plotted vs CORES (n_resource), so packed and hybrid compare honestly.
   282	
   283	RECOMMENDED MPAS CPU config: `--ntasks-per-node=8 --cpus-per-task=4`
   284	(`numactl --localalloc` + `--distribution=block:block` give a small extra
   285	trim; not the primary fix). Do NOT pack 32 single-thread ranks/node.
   286	
   287	## GPU: single-device real, multi-GPU MPI blocked by the env
   288	
   289	Two GPU bugs found + handled:
   290	1. The GPU ladder set `env JAX_PLATFORMS=cuda` but did NOT pass `--device gpu`,
   291	   so `_configure_jax_cpu` pinned `JAX_PLATFORMS=cpu` — the ENTIRE g1..g32 "GPU"
   292	   ladder silently ran on CPU (JSON `backend=cpu`). Fixed: `--device gpu` + a
   293	   backend assertion that SystemExits on CUDA fallback (commit c8bd94fc3), so
   294	   CPU can never be recorded as GPU again. Bogus dirs purged.
   295	2. With the fix, SINGLE GPU works (real): 1 GPU I5 = **f32 791 SYPD (197
   296	   Mc/s), f64 158 SYPD (39 Mc/s)** — 8-16x the bogus CPU-fallback numbers and
   297	   ~2x the CPU per-device throughput. But MULTI-GPU MPI (g2+) fails with
   298	   "mpi4jax GPU extensions could not be imported — rebuild mpi4jax with CUDA":
   299	   the env's mpi4jax is CPU-only, so the GPU halo exchange cannot run. The
   300	   mpi4jax GPU path is therefore BLOCKED until mpi4jax is rebuilt CUDA-aware.
   301	
   302	### UPDATE 2026-06-17 — GPU multi-device is NOT wholly blocked (jax-mesh SPMD works)
   303	
   304	The "single-process multi-GPU SPMD path (no mpi4jax)" anticipated above now
   305	RUNS (commit 001b18bcb). `--cs-spmd --device gpu` as a SINGLE process with
   306	`CUDA_VISIBLE_DEVICES=0,1` builds the global face mesh from `jax.devices()` (=2)
   307	and shards the cubed-sphere over both GPUs via the jax device-mesh + multiface
   308	ppermute halo — no mpi4jax, and `jax.distributed` is skipped for one process.
   309	Three harness fixes unblocked it: defer the `--device gpu` backend assertion past
   310	the (skipped) cs-spmd init, and stop `_configure_jax_gpu` from clobbering an
   311	explicit `CUDA_VISIBLE_DEVICES` (it had double-restricted / single-pinned).
   312	
   313	So the GPU-multi-device picture is three-way, not "blocked":
   314	- **mpi4jax-GPU halo**: blocked (CPU-only mpi4jax build) — infra.
   315	- **jax.distributed multi-controller (srun -n2) on GPU**: blocked here — the NCCL
   316	  local-topology gather times out (`GetKeyValue cuda:local_topology/cuda/1`, 2 min)
   317	  on this stack — an env/NCCL issue, not a code gap.
   318	- **single-process jax device-mesh SPMD**: WORKS.
   319	
   320	Measured single-process cubed-sphere f32, 1-GPU vs 2-GPU (strong, same face):
   321	C48 1.62→2.99 ms (222→120 Mc/s), C96 4.63→6.22 ms (310→231), **C192 19.97→17.96
   322	ms (288→320 Mc/s = 1.11x speedup)**. Crossover ~C192: below it the cross-GPU
   323	ppermute halo over the shared PCIe-Gen3 link (no NVLink on these RTX-8000 pairs)
   324	costs more than the per-GPU compute saved; at C192 compute finally dominates and
   325	2 GPUs win — modestly, the same PCIe-fabric wall the CPU side hit. Single-device
   326	GPU per-resolution throughput (the per-GPU ceiling) stands as the main GPU panel.
   327	
   328	KNOWN minor: a single-process cs-spmd run records `n_ranks=1` (the process count)
   329	rather than the jax mesh size `n_global`, so the 2-GPU point lands in the CSV
   330	mislabeled n=1 (the numbers above are read from the run logs). A label fix
   331	(record `n_global` for cs-spmd) is a small follow-up; it does not affect the
   332	multi-controller CPU cs-spmd runs (those use one process per device).
   333	
   334	**Ocean/lat-lon GPU multi-device (same route-B jax-mesh):** the latitude-band
   335	SPMD step (`bench_ocean_latlon_spmd_pcg.py`, `shard_map` over a 1-D `lat` mesh,
   336	single process drives both GPUs — no mpi4jax) also runs on 2 GPUs. Barotropic
   337	solve, LL720 (720x1440), 1- vs 2-GPU: **f64 18.90→16.35 ms = 1.16x** (eff 0.58);
   338	f32 8.59→9.79 ms = 0.88x (too light — the cross-GPU halo over PCIe outweighs the
   339	shrunk per-device compute, the same crossover as cubed-sphere f32 at C96). So
   340	GPU multi-device scaling works via the single-process jax-mesh path for BOTH
   341	decomposable grids — cubed-sphere (C192 f32 1.11x) and ocean lat-lon (LL720 f64
   342	1.16x) — modest and PCIe-bound (no NVLink), positive once the per-device problem
   343	is large enough. This is a microbench (ms/solve, not a tidy-CSV SYPD row), so the
   344	numbers live here rather than in the auto-generated figure.
   345	
   346	**GPU multi-device — final verdict.** Three paths, now fully characterised:
   347	mpi4jax-GPU halo = infra-blocked (CPU-only build); jax.distributed multi-
   348	controller = env-blocked (NCCL local-topology gather times out); single-process
   349	jax device-mesh SPMD = WORKS for both cubed-sphere and ocean, ~1.1–1.2x at 2 GPUs
   350	intra-node, PCIe-bound. The "GPU multi-device infra-blocked" verdict was wrong
   351	for the third path; bigger speedups need NVLink (or >2 GPUs/node), which this
   352	hardware does not have.
   353	
   354	## Measured roofline (the quantified limit)
   355	
   356	`scripts/bench/roofline_probe.py` on Ginsburg (job 8502024):
   357	
   358	- **GPU (RTX 8000, single device):** sustained **403.9 GB/s = 65% of the 624
   359	  GB/s peak**. The dycore per-device throughput is at a healthy fraction of the
   360	  memory-bandwidth roofline; closing the last 35% is a kernel-fusion project
   361	  (Oceananigans-style), not a parallel one.
   362	- **CPU MPI (mpi4jax sendrecv over Gloo):** **latency floor 267.8 µs**,
   363	  asymptotic **1.4 GB/s**. This is the multi-device wall: a step issues ~3
   364	  RK-stage halo exchanges, each O(neighbours) serialized `sendrecv`s; at
   365	  I5/np8 (~45 messages) the 268 µs floor alone is ~12 ms/step — exactly the
   366	  observed latency-bound collapse. Lowering it needs InfiniBand/NCCL
   367	  (CUDA-aware, RDMA), which Ginsburg's PCIe-Gen3 + Gloo stack does not provide.
   368	
   369	So the weak-scaling efficiencies (atm-ico 0.80, ocean 0.92) are at the fabric
   370	limit, and per-device throughput is at the bandwidth limit. We have reached the
   371	practical theoretical limit for this hardware on every decomposable config.
   372	
   373	## Bottom line
   374	
   375	For every grid that can decompose across nodes today (atm icosahedral, ocean
   376	lat-lon), weak/strong scaling is at the practical limit set by the Gloo/PCIe
   377	fabric and mpi4jax's serialized per-neighbour `sendrecv`; per-device throughput
   378	is at the memory-bandwidth-bound ceiling (quantified by
   379	`scripts/bench/roofline_probe.py`).
   380	
   381	The **cubed-sphere >6-device** project is now DONE as a capability: the sub-face
   382	np=6·kt² tiling is shipped + np24/54 bit-identity parity-gated for the full 3D-PE
   383	step — DRY *and* (2026-06-20) MOIST (tracers + Kessler;
   384	`cube_moist_tiled_step_design.md`). It is FUTURE-HW (np>6 anti-scales on
   385	Ginsburg's CPU shard_map / cross-node ppermute by design, so it is gated by
   386	bit-identity, not benchmarked here). With it, every atmosphere grid ×
   387	{dry,moist} × {f32,f64} × {weak,strong} is at its Ginsburg practical limit and
   388	the cube has a parity-gated path beyond 6 devices for fast-interconnect HW. The
   389	two remaining decomposition projects (atm lat-lon 2-D pencil at high np, spectral
   390	transpose) stay HW-blocked on Gloo/PCIe — the SOTA fixes (MOM6/E3SM 2-D pencil;
   391	NeuralGCM/spectral transpose) need InfiniBand/NCCL Ginsburg does not have.

exec
/usr/bin/bash -lc "printf '%s\\n' '=== metadata census implementation ==='
rg -n -A100 -B16 'def hlo_collective_census|hlo_collective_census' scripts packages tests | sed -n '1,280p'
printf '%s\\n' '=== current campaign tail, selected claim lines ==='
nl -ba docs/performance/scaling/levante_campaign_2026-07-24.md | sed -n '1790,1860p'
printf '%s\\n' '=== job id / A-B references ==='
rg -n -i '26630438|26630576|ll128_comb|collective_permute_combine|combine-8mb|A2_default' docs scripts .physics-validator 2>/dev/null | sed -n '1,260p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 953ms:
=== metadata census implementation ===
tests/bench/test_scaling_metadata.py-447-    a line that merely CONTAINS the substring "done" elsewhere — an XLA
tests/bench/test_scaling_metadata.py-448-    metadata op_name, a ``%done_*`` SSA name — must STILL be counted.  A blunt
tests/bench/test_scaling_metadata.py-449-    ``"done" not in line`` filter would false-drop these to zero."""
tests/bench/test_scaling_metadata.py-450-    hlo = "\n".join([
tests/bench/test_scaling_metadata.py-451-        '  %r = all-reduce(%p), metadata={op_name="jit(step)/done_stage/psum"}',
tests/bench/test_scaling_metadata.py-452-        '  %done_mass = f32[] collective-permute(%q)',
tests/bench/test_scaling_metadata.py-453-        '  %g = all-gather(%z), metadata={op_name="reduce_done/x"}',
tests/bench/test_scaling_metadata.py-454-    ])
tests/bench/test_scaling_metadata.py-455-    c = md.count_collectives(hlo)
tests/bench/test_scaling_metadata.py-456-    assert c["all_reduce"] == 1        # NOT dropped despite "done" in metadata
tests/bench/test_scaling_metadata.py-457-    assert c["collective_permute"] == 1  # NOT dropped despite %done_ SSA name
tests/bench/test_scaling_metadata.py-458-    assert c["all_gather"] == 1
tests/bench/test_scaling_metadata.py-459-    # canonical permute helper is fixed by the same shared counter
tests/bench/test_scaling_metadata.py-460-    assert md.count_collective_permutes(hlo) == 1
tests/bench/test_scaling_metadata.py-461-
tests/bench/test_scaling_metadata.py-462-
tests/bench/test_scaling_metadata.py:463:def test_hlo_collective_census_lowers_and_is_error_safe():
tests/bench/test_scaling_metadata.py-464-    """Best-effort full-census probe: a collective-free fn -> all-zero dict;
tests/bench/test_scaling_metadata.py-465-    an unlowerable fn -> None (never raises)."""
tests/bench/test_scaling_metadata.py-466-    import jax.numpy as jnp
tests/bench/test_scaling_metadata.py-467-    with _cpu_compile():
tests/bench/test_scaling_metadata.py:468:        census = md.hlo_collective_census(lambda x: x + 1, jnp.arange(4.0))
tests/bench/test_scaling_metadata.py-469-        assert census is not None and census["total"] == 0
tests/bench/test_scaling_metadata.py-470-
tests/bench/test_scaling_metadata.py-471-        def _boom(x):
tests/bench/test_scaling_metadata.py-472-            raise RuntimeError("unlowerable")
tests/bench/test_scaling_metadata.py:473:        assert md.hlo_collective_census(_boom, jnp.arange(4.0)) is None
--
scripts/bench/bench_mpas_spmd_scaling.py-60-from pathlib import Path
scripts/bench/bench_mpas_spmd_scaling.py-61-
scripts/bench/bench_mpas_spmd_scaling.py-62-import jax
scripts/bench/bench_mpas_spmd_scaling.py-63-import numpy as np
scripts/bench/bench_mpas_spmd_scaling.py-64-
scripts/bench/bench_mpas_spmd_scaling.py-65-# Repo root on the path for tests.test_cases.baroclinic_wave (the same
scripts/bench/bench_mpas_spmd_scaling.py-66-# baroclinic-wave IC the icosahedral lanes of run_levante_gpu_scaling use).
scripts/bench/bench_mpas_spmd_scaling.py-67-sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
scripts/bench/bench_mpas_spmd_scaling.py-68-# Bench dir for the shared metadata module (sibling-script import pattern).
scripts/bench/bench_mpas_spmd_scaling.py-69-sys.path.insert(0, str(Path(__file__).resolve().parent))
scripts/bench/bench_mpas_spmd_scaling.py-70-
scripts/bench/bench_mpas_spmd_scaling.py-71-# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
scripts/bench/bench_mpas_spmd_scaling.py-72-# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
scripts/bench/bench_mpas_spmd_scaling.py-73-# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
scripts/bench/bench_mpas_spmd_scaling.py-74-# imports JAX lazily, so this is safe before jax.distributed.initialize.
scripts/bench/bench_mpas_spmd_scaling.py-75-from metadata import (  # noqa: E402
scripts/bench/bench_mpas_spmd_scaling.py:76:    annotate_incomplete, hlo_collective_census, scaling_metadata,
scripts/bench/bench_mpas_spmd_scaling.py-77-    tidy_throughput_fields)
scripts/bench/bench_mpas_spmd_scaling.py-78-
scripts/bench/bench_mpas_spmd_scaling.py-79-# SPMD full-step parity tolerances — the FLOATING-POINT RE-ASSOCIATION floor
scripts/bench/bench_mpas_spmd_scaling.py-80-# of the sharded step (ppermute halo + mass-fix psum reduction-order change),
scripts/bench/bench_mpas_spmd_scaling.py-81-# NOT a bug margin; a real halo/partition regression shows up orders of
scripts/bench/bench_mpas_spmd_scaling.py-82-# magnitude above these.  Values extend the 1-step envelope of
scripts/bench/bench_mpas_spmd_scaling.py-83-# tests/parallel/test_voronoi_sharded_equivalence.py (u/T atol 1e-6, p_s
scripts/bench/bench_mpas_spmd_scaling.py-84-# atol 1e-1) to the smoke window; the floor grows with steps, hence the cap.
scripts/bench/bench_mpas_spmd_scaling.py-85-MPAS_PARITY_TOLS = {  # precision -> field -> (rtol, atol)
scripts/bench/bench_mpas_spmd_scaling.py-86-    "float64": {"u": (1.0e-5, 1.0e-5), "T": (1.0e-6, 1.0e-5),
scripts/bench/bench_mpas_spmd_scaling.py-87-                "p_s": (1.0e-5, 1.0)},
scripts/bench/bench_mpas_spmd_scaling.py-88-    "float32": {"u": (1.0e-3, 1.0e-3), "T": (1.0e-4, 1.0e-3),
scripts/bench/bench_mpas_spmd_scaling.py-89-                "p_s": (1.0e-3, 50.0)},
scripts/bench/bench_mpas_spmd_scaling.py-90-}
scripts/bench/bench_mpas_spmd_scaling.py-91-MPAS_PARITY_MAX_STEPS = 8
scripts/bench/bench_mpas_spmd_scaling.py-92-
scripts/bench/bench_mpas_spmd_scaling.py-93-# Conservation gate default: with fix_mass=True the step restores the global
scripts/bench/bench_mpas_spmd_scaling.py-94-# dry mass to the pre-step value each step, so the drift over a smoke window
scripts/bench/bench_mpas_spmd_scaling.py-95-# is the allreduce rounding floor, not scheme drift.
scripts/bench/bench_mpas_spmd_scaling.py-96-MASS_RTOL_DEFAULTS = {"float64": 1.0e-11, "float32": 1.0e-5}
scripts/bench/bench_mpas_spmd_scaling.py-97-
scripts/bench/bench_mpas_spmd_scaling.py-98-
scripts/bench/bench_mpas_spmd_scaling.py-99-def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
scripts/bench/bench_mpas_spmd_scaling.py-100-                          moist=False, lloyd_iterations=50):
scripts/bench/bench_mpas_spmd_scaling.py-101-    """Reordered+padded global mesh, MPAS PE model, baroclinic-wave IC.
scripts/bench/bench_mpas_spmd_scaling.py-102-
scripts/bench/bench_mpas_spmd_scaling.py-103-    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
scripts/bench/bench_mpas_spmd_scaling.py-104-    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
scripts/bench/bench_mpas_spmd_scaling.py-105-    device count of THIS run's mesh/model (the two differ for the
scripts/bench/bench_mpas_spmd_scaling.py-106-    single-device reference leg of a ladder, via ``--reorder-for``).
scripts/bench/bench_mpas_spmd_scaling.py-107-    ``moist=True`` attaches the q_v/q_c/q_r tracers (moist baroclinic
scripts/bench/bench_mpas_spmd_scaling.py-108-    wave) so the sharded step's packed tracer halo exchange + RK tracer
scripts/bench/bench_mpas_spmd_scaling.py-109-    advection sit on the timed/gated path.
scripts/bench/bench_mpas_spmd_scaling.py-110-    """
scripts/bench/bench_mpas_spmd_scaling.py-111-    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
scripts/bench/bench_mpas_spmd_scaling.py-112-        MPASPrimitiveEquationConfig,
scripts/bench/bench_mpas_spmd_scaling.py-113-        MPASPrimitiveEquationModel,
scripts/bench/bench_mpas_spmd_scaling.py-114-    )
scripts/bench/bench_mpas_spmd_scaling.py-115-    from legoesm.grids.vertical import create_sigma_coordinate
scripts/bench/bench_mpas_spmd_scaling.py-116-    from legoesm.grids.voronoi import create_voronoi_mesh
scripts/bench/bench_mpas_spmd_scaling.py-117-    from legoesm.parallel.mesh import create_voronoi_device_mesh
scripts/bench/bench_mpas_spmd_scaling.py-118-    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
scripts/bench/bench_mpas_spmd_scaling.py-119-
scripts/bench/bench_mpas_spmd_scaling.py-120-    mesh = create_voronoi_mesh(subdivision_level=subdivision,
scripts/bench/bench_mpas_spmd_scaling.py-121-                               lloyd_iterations=lloyd_iterations)
scripts/bench/bench_mpas_spmd_scaling.py-122-    mesh = reorder_voronoi_for_sharding(mesh, reorder_target, method=method)
scripts/bench/bench_mpas_spmd_scaling.py-123-    if run_nd > 1 and (mesh.nCells % run_nd or mesh.nEdges % run_nd):
scripts/bench/bench_mpas_spmd_scaling.py-124-        # Padding only guarantees divisibility for reorder_target.
scripts/bench/bench_mpas_spmd_scaling.py-125-        raise SystemExit(
scripts/bench/bench_mpas_spmd_scaling.py-126-            f"padded mesh (nCells={mesh.nCells}, nEdges={mesh.nEdges}) not "
scripts/bench/bench_mpas_spmd_scaling.py-127-            f"divisible by --n-devices {run_nd}; use a ladder where every "
scripts/bench/bench_mpas_spmd_scaling.py-128-            f"count divides --reorder-for ({reorder_target}).")
scripts/bench/bench_mpas_spmd_scaling.py-129-    sigma = create_sigma_coordinate(nlev)
scripts/bench/bench_mpas_spmd_scaling.py-130-    # Same recipe as the icosahedral lane of run_levante_gpu_scaling /
scripts/bench/bench_mpas_spmd_scaling.py-131-    # tests/parallel/test_voronoi_sharded_equivalence.py: del4 hyperdiffusion,
scripts/bench/bench_mpas_spmd_scaling.py-132-    # energy-conserving PV flux, SSP-RK3, global mass fixer.
scripts/bench/bench_mpas_spmd_scaling.py-133-    cfg = MPASPrimitiveEquationConfig(
scripts/bench/bench_mpas_spmd_scaling.py-134-        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
scripts/bench/bench_mpas_spmd_scaling.py-135-        pv_scheme="energy", time_integrator="ssp_rk3",
scripts/bench/bench_mpas_spmd_scaling.py-136-    )
scripts/bench/bench_mpas_spmd_scaling.py-137-    dev_config = create_voronoi_device_mesh(
scripts/bench/bench_mpas_spmd_scaling.py-138-        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
scripts/bench/bench_mpas_spmd_scaling.py-139-        n_devices=run_nd,
scripts/bench/bench_mpas_spmd_scaling.py-140-    )
scripts/bench/bench_mpas_spmd_scaling.py-141-    if dev_config.n_devices > 1:
scripts/bench/bench_mpas_spmd_scaling.py-142-        from legoesm.parallel.mesh import replicate_pytree
scripts/bench/bench_mpas_spmd_scaling.py-143-        mesh_model = replicate_pytree(mesh, dev_config)
scripts/bench/bench_mpas_spmd_scaling.py-144-    else:
scripts/bench/bench_mpas_spmd_scaling.py-145-        mesh_model = mesh
scripts/bench/bench_mpas_spmd_scaling.py-146-    model = MPASPrimitiveEquationModel(mesh_model, sigma, cfg)
scripts/bench/bench_mpas_spmd_scaling.py-147-    # #1100 MPAS twin: the timed path never global-builds the state.
scripts/bench/bench_mpas_spmd_scaling.py-148-    # build_sharded_baroclinic_wave_state_mpas creates every leaf via
scripts/bench/bench_mpas_spmd_scaling.py-149-    # jax.make_array_from_callback (only THIS process's shard rows are
scripts/bench/bench_mpas_spmd_scaling.py-150-    # ever materialised; value-identical (few-ULP contract, measured
scripts/bench/bench_mpas_spmd_scaling.py-151-    # exact on the pinned CPU stack) to global-build + shard_pytree —
scripts/bench/bench_mpas_spmd_scaling.py-152-    # tests/parallel/test_mpas_partitionlocal_build.py).  The GLOBAL
scripts/bench/bench_mpas_spmd_scaling.py-153-    # state is built lazily in main() only for the parity/conservation
scripts/bench/bench_mpas_spmd_scaling.py-154-    # gates (small smoke scales).  The mesh itself is still global per
scripts/bench/bench_mpas_spmd_scaling.py-155-    # process — its SFC-partition-local construction is the open
scripts/bench/bench_mpas_spmd_scaling.py-156-    # remainder of #1100.
scripts/bench/bench_mpas_spmd_scaling.py-157-    from tests.test_cases.baroclinic_wave import (
scripts/bench/bench_mpas_spmd_scaling.py-158-        build_sharded_baroclinic_wave_state_mpas,
scripts/bench/bench_mpas_spmd_scaling.py-159-    )
scripts/bench/bench_mpas_spmd_scaling.py-160-    state_sharded = build_sharded_baroclinic_wave_state_mpas(
scripts/bench/bench_mpas_spmd_scaling.py-161-        mesh, sigma, dev_config, perturbed=True, moist=moist)
scripts/bench/bench_mpas_spmd_scaling.py-162-    return mesh, model, state_sharded, dev_config
scripts/bench/bench_mpas_spmd_scaling.py-163-
scripts/bench/bench_mpas_spmd_scaling.py-164-
scripts/bench/bench_mpas_spmd_scaling.py-165-def _block(state):
scripts/bench/bench_mpas_spmd_scaling.py-166-    jax.block_until_ready([leaf for leaf in jax.tree.leaves(state)
scripts/bench/bench_mpas_spmd_scaling.py-167-                           if leaf is not None])
scripts/bench/bench_mpas_spmd_scaling.py-168-
scripts/bench/bench_mpas_spmd_scaling.py-169-
scripts/bench/bench_mpas_spmd_scaling.py-170-def _global_dry_mass(state, mesh):
scripts/bench/bench_mpas_spmd_scaling.py-171-    """sum(p_s * areaCell) on host arrays — the quantity fix_mass pins."""
scripts/bench/bench_mpas_spmd_scaling.py-172-    ps = np.asarray(state.p_s.data)
scripts/bench/bench_mpas_spmd_scaling.py-173-    area = np.asarray(mesh.areaCell)
scripts/bench/bench_mpas_spmd_scaling.py-174-    return float(np.sum(ps * area))
scripts/bench/bench_mpas_spmd_scaling.py-175-
scripts/bench/bench_mpas_spmd_scaling.py-176-
--
scripts/bench/bench_mpas_spmd_scaling.py-417-
scripts/bench/bench_mpas_spmd_scaling.py-418-    # HLO collective-permute census (#1113 ask 2): a STATIC compile property of
scripts/bench/bench_mpas_spmd_scaling.py-419-    # the sharded step — the ppermute ROUND count that decomposes multi-node
scripts/bench/bench_mpas_spmd_scaling.py-420-    # overhead (overhead ~= CPs/step * ~0.11 ms launch floor). The cube benches
scripts/bench/bench_mpas_spmd_scaling.py-421-    # record this; the MPAS row did not, forcing an out-of-band census. Counted
scripts/bench/bench_mpas_spmd_scaling.py-422-    # AFTER the timed loop so the census compile can't perturb per_step_ms[0]'s
scripts/bench/bench_mpas_spmd_scaling.py-423-    # compile timing (the executable is already cached — this re-lower/compile
scripts/bench/bench_mpas_spmd_scaling.py-424-    # is a cache hit; the count is data-independent, static in the partition).
scripts/bench/bench_mpas_spmd_scaling.py-425-    # Best-effort (None if compilation is unsupported); the serial n=1 leg has
scripts/bench/bench_mpas_spmd_scaling.py-426-    # no ppermute halo -> 0.
scripts/bench/bench_mpas_spmd_scaling.py-427-    if physics_fn is not None:
scripts/bench/bench_mpas_spmd_scaling.py-428-        _census_fn = lambda st: step(st, dt, physics_fn=physics_fn)  # noqa: E731
scripts/bench/bench_mpas_spmd_scaling.py-429-    else:
scripts/bench/bench_mpas_spmd_scaling.py-430-        _census_fn = lambda st: step(st, dt)  # noqa: E731
scripts/bench/bench_mpas_spmd_scaling.py-431-    # ONE compile → full per-family census; the CP scalar (the #1113 round-count
scripts/bench/bench_mpas_spmd_scaling.py-432-    # wall) is the collective_permute member, so no second compile for it.
scripts/bench/bench_mpas_spmd_scaling.py:433:    hlo_census = hlo_collective_census(_census_fn, s)
scripts/bench/bench_mpas_spmd_scaling.py-434-    hlo_cp = hlo_census["collective_permute"] if hlo_census else None
scripts/bench/bench_mpas_spmd_scaling.py-435-
scripts/bench/bench_mpas_spmd_scaling.py-436-    # --- Correctness gates (before any timing is reported) -----------------
scripts/bench/bench_mpas_spmd_scaling.py-437-    if args.parity_gate or args.check_conservation:
scripts/bench/bench_mpas_spmd_scaling.py-438-        final_global = (gather_voronoi_state_spmd(s, dev_config)
scripts/bench/bench_mpas_spmd_scaling.py-439-                        if dev_config.n_devices > 1 else s)
scripts/bench/bench_mpas_spmd_scaling.py-440-        prec = "float64" if jax.config.jax_enable_x64 else "float32"
scripts/bench/bench_mpas_spmd_scaling.py-441-        rank0 = jax.process_index() == 0
scripts/bench/bench_mpas_spmd_scaling.py-442-        if args.check_conservation:
scripts/bench/bench_mpas_spmd_scaling.py-443-            mass_after = _global_dry_mass(final_global, mesh)
scripts/bench/bench_mpas_spmd_scaling.py-444-            tol = (args.mass_rtol if args.mass_rtol is not None
scripts/bench/bench_mpas_spmd_scaling.py-445-                   else MASS_RTOL_DEFAULTS[prec])
scripts/bench/bench_mpas_spmd_scaling.py-446-            rel = abs(mass_after - mass_before) / abs(mass_before)
scripts/bench/bench_mpas_spmd_scaling.py-447-            if rank0:
scripts/bench/bench_mpas_spmd_scaling.py-448-                print(f"    conservation dry-mass: rel drift={rel:.3e} "
scripts/bench/bench_mpas_spmd_scaling.py-449-                      f"(tol {tol:.1e}) over {args.steps} steps", flush=True)
scripts/bench/bench_mpas_spmd_scaling.py-450-            if rel > tol:
scripts/bench/bench_mpas_spmd_scaling.py-451-                if rank0:
scripts/bench/bench_mpas_spmd_scaling.py-452-                    print("ERROR: conservation gate BREACHED.", flush=True)
scripts/bench/bench_mpas_spmd_scaling.py-453-                return 4
scripts/bench/bench_mpas_spmd_scaling.py-454-        if args.parity_gate:
scripts/bench/bench_mpas_spmd_scaling.py-455-            tols = MPAS_PARITY_TOLS[prec]
scripts/bench/bench_mpas_spmd_scaling.py-456-            ok = True
scripts/bench/bench_mpas_spmd_scaling.py-457-            checks = [
scripts/bench/bench_mpas_spmd_scaling.py-458-                (name, getattr(serial_final, name).data,
scripts/bench/bench_mpas_spmd_scaling.py-459-                 getattr(final_global, name).data, rtol, atol)
scripts/bench/bench_mpas_spmd_scaling.py-460-                for name, (rtol, atol) in tols.items()
scripts/bench/bench_mpas_spmd_scaling.py-461-            ]
scripts/bench/bench_mpas_spmd_scaling.py-462-            if serial_final.tracers is not None:
scripts/bench/bench_mpas_spmd_scaling.py-463-                # Moist run: the tracer fields ride the packed exchange +
scripts/bench/bench_mpas_spmd_scaling.py-464-                # RK advection — gate them too (q re-association floor is
scripts/bench/bench_mpas_spmd_scaling.py-465-                # far below the q_v scale; reuse the T tolerances).
scripts/bench/bench_mpas_spmd_scaling.py-466-                q_rtol, q_atol = tols["T"]
scripts/bench/bench_mpas_spmd_scaling.py-467-                if set(final_global.tracers or {}) != set(
scripts/bench/bench_mpas_spmd_scaling.py-468-                        serial_final.tracers):
scripts/bench/bench_mpas_spmd_scaling.py-469-                    if rank0:
scripts/bench/bench_mpas_spmd_scaling.py-470-                        print("ERROR: sharded run dropped tracer fields.",
scripts/bench/bench_mpas_spmd_scaling.py-471-                              flush=True)
scripts/bench/bench_mpas_spmd_scaling.py-472-                    return 5
scripts/bench/bench_mpas_spmd_scaling.py-473-                checks += [
scripts/bench/bench_mpas_spmd_scaling.py-474-                    (k, serial_final.tracers[k].data,
scripts/bench/bench_mpas_spmd_scaling.py-475-                     final_global.tracers[k].data, q_rtol, q_atol * 1e-3)
scripts/bench/bench_mpas_spmd_scaling.py-476-                    for k in sorted(serial_final.tracers)
scripts/bench/bench_mpas_spmd_scaling.py-477-                ]
scripts/bench/bench_mpas_spmd_scaling.py-478-            for name, want, got, rtol, atol in checks:
scripts/bench/bench_mpas_spmd_scaling.py-479-                want = np.asarray(want)
scripts/bench/bench_mpas_spmd_scaling.py-480-                got = np.asarray(got)
scripts/bench/bench_mpas_spmd_scaling.py-481-                field_ok = bool(np.allclose(got, want, rtol=rtol, atol=atol))
scripts/bench/bench_mpas_spmd_scaling.py-482-                ok &= field_ok
scripts/bench/bench_mpas_spmd_scaling.py-483-                if rank0:
scripts/bench/bench_mpas_spmd_scaling.py-484-                    mx = (float(np.max(np.abs(got - want)))
scripts/bench/bench_mpas_spmd_scaling.py-485-                          if want.size else 0.0)
scripts/bench/bench_mpas_spmd_scaling.py-486-                    print(f"    parity {name:>4s}: max|diff|={mx:.3e} "
scripts/bench/bench_mpas_spmd_scaling.py-487-                          f"{'OK' if field_ok else 'MISMATCH'}", flush=True)
scripts/bench/bench_mpas_spmd_scaling.py-488-            if not ok:
scripts/bench/bench_mpas_spmd_scaling.py-489-                if rank0:
scripts/bench/bench_mpas_spmd_scaling.py-490-                    print("ERROR: SPMD parity gate MISMATCH vs the "
scripts/bench/bench_mpas_spmd_scaling.py-491-                          "single-device reference.", flush=True)
scripts/bench/bench_mpas_spmd_scaling.py-492-                return 5
scripts/bench/bench_mpas_spmd_scaling.py-493-
scripts/bench/bench_mpas_spmd_scaling.py-494-    steady = per_step_ms[args.warmup:]
scripts/bench/bench_mpas_spmd_scaling.py-495-    med = float(np.median(steady))
scripts/bench/bench_mpas_spmd_scaling.py-496-    rec = dict(
scripts/bench/bench_mpas_spmd_scaling.py-497-        component="mpas_atm",
scripts/bench/bench_mpas_spmd_scaling.py-498-        subdivision=args.subdivision, n_devices=nd,
scripts/bench/bench_mpas_spmd_scaling.py-499-        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
scripts/bench/bench_mpas_spmd_scaling.py-500-        partition_method=args.partition_method, physics=args.physics,
scripts/bench/bench_mpas_spmd_scaling.py-501-        # lloyd=0 is the LABELLED synthetic scaling mesh — anti-masquerade:
scripts/bench/bench_mpas_spmd_scaling.py-502-        # a row without this field could pass as a production-SCVT receipt.
scripts/bench/bench_mpas_spmd_scaling.py-503-        lloyd_iterations=args.lloyd,
scripts/bench/bench_mpas_spmd_scaling.py-504-        # Requested vs EFFECTIVE (post-"auto") strategy — a JSONL row
scripts/bench/bench_mpas_spmd_scaling.py-505-        # saying "auto" would not reveal whether ppermute or allgather
scripts/bench/bench_mpas_spmd_scaling.py-506-        # was actually measured (codex M3c-2 MINOR).
scripts/bench/bench_mpas_spmd_scaling.py-507-        halo_strategy_requested=args.halo_strategy,
scripts/bench/bench_mpas_spmd_scaling.py-508-        halo_strategy_effective=getattr(
scripts/bench/bench_mpas_spmd_scaling.py-509-            step, "_halo_strategy_effective", "serial"),
scripts/bench/bench_mpas_spmd_scaling.py-510-        steps=args.steps, dt=dt,
scripts/bench/bench_mpas_spmd_scaling.py-511-        platform=jax.default_backend(),
scripts/bench/bench_mpas_spmd_scaling.py-512-        n_processes=jax.process_count(),
scripts/bench/bench_mpas_spmd_scaling.py-513-        multicontroller=bool(args.multicontroller),
scripts/bench/bench_mpas_spmd_scaling.py-514-        compile_ms=round(per_step_ms[0], 1),
scripts/bench/bench_mpas_spmd_scaling.py-515-        steady_median_ms=round(med, 2),
scripts/bench/bench_mpas_spmd_scaling.py-516-        steady_min_ms=round(float(np.min(steady)), 2),
scripts/bench/bench_mpas_spmd_scaling.py-517-        per_step_ms=[round(x, 1) for x in per_step_ms],
scripts/bench/bench_mpas_spmd_scaling.py-518-        cells=int(mesh.nCells) * args.nlev,
scripts/bench/bench_mpas_spmd_scaling.py-519-        # ppermute round count/step (static compile property; #1113) — the
scripts/bench/bench_mpas_spmd_scaling.py-520-        # multi-node ceiling is this count x the ~0.11 ms launch floor, so it
scripts/bench/bench_mpas_spmd_scaling.py-521-        # belongs on every row like the cube benches.
scripts/bench/bench_mpas_spmd_scaling.py-522-        hlo_collective_permutes=hlo_cp,
scripts/bench/bench_mpas_spmd_scaling.py-523-        # full per-family census (permute + all-reduce + all-gather + ...) on
scripts/bench/bench_mpas_spmd_scaling.py-524-        # the SAME compile: exposes any reduction the ico step introduces.
scripts/bench/bench_mpas_spmd_scaling.py-525-        hlo_collectives=hlo_census,
scripts/bench/bench_mpas_spmd_scaling.py-526-    )
scripts/bench/bench_mpas_spmd_scaling.py-527-    # Flat aggregator-compatible identity + metric fields (see the latlon
scripts/bench/bench_mpas_spmd_scaling.py-528-    # twin): resolution = subdivision level, matching run_cpu_mpi_scaling's
scripts/bench/bench_mpas_spmd_scaling.py-529-    # icosahedral convention so both lanes land on the same plot curves.
scripts/bench/bench_mpas_spmd_scaling.py-530-    rec.update(
scripts/bench/bench_mpas_spmd_scaling.py-531-        grid_type="icosahedral",
scripts/bench/bench_mpas_spmd_scaling.py-532-        resolution=args.subdivision,
scripts/bench/bench_mpas_spmd_scaling.py-533-        n_levels=args.nlev,
--
scripts/bench/metadata.py-189-
scripts/bench/metadata.py-190-    Superset of :func:`count_collective_permutes`: adds ``all_reduce`` (the
scripts/bench/metadata.py-191-    reduction wall that dominates ocean implicit-CN strong scaling and hides
scripts/bench/metadata.py-192-    from a permute-only count), ``all_gather``, ``all_to_all``,
scripts/bench/metadata.py-193-    ``reduce_scatter``.  Same STATIC, op-call-form discipline (see
scripts/bench/metadata.py-194-    :func:`_count_op_calls`: config-header flag names never inflate; an async
scripts/bench/metadata.py-195-    collective counts once via ``-start`` with its ``-done`` companion excluded
scripts/bench/metadata.py-196-    by the regex, NOT by a fragile line-wide substring test).  Keys are
scripts/bench/metadata.py-197-    underscore-normalized op names plus a ``total``.  Canonical census for the
scripts/bench/metadata.py-198-    scaling-diagnosis tool and the SPMD benches — no re-implementation."""
scripts/bench/metadata.py-199-    counts = {key: _count_op_calls(hlo_text, rx)
scripts/bench/metadata.py-200-              for key, rx in _COLLECTIVE_OP_RES.items()}
scripts/bench/metadata.py-201-    counts["total"] = sum(counts.values())
scripts/bench/metadata.py-202-    return counts
scripts/bench/metadata.py-203-
scripts/bench/metadata.py-204-
=== current campaign tail, selected claim lines ===
  1790	  decay with parts; the metis receipt argues against pure
  1791	  partition-cut explanations, on the CPU lane at least).
  1792	* The non-monotone tile dependence of the ratio (largest at the
  1793	  LARGEST tile, 1.90 at 81.9k) is unexplained; recorded, not theorised.
  1794	
  1795	## Distance-to-modeled-limit: atm lat-lon GPU (2026-08-02, "near theoretical limit" directive)
  1796	
  1797	Closed the bench's own honest-null bound gap (audit item 4) for the
  1798	lat-lon lane, using only repo instruments:
  1799	
  1800	* **Halo census** (new probe `scripts/tmp/_probe_latlon_halo_census.py`,
  1801	  virtual-CPU forced-host-platform lowering of the REAL
  1802	  `make_sharded_atm_latlon_step`): **41 collective-permutes + 1
  1803	  all-reduce per step**, nd-INDEPENDENT (identical at nd=8 and nd=16 —
  1804	  the 1-D band structure check). Exact CP payload from compiled-HLO
  1805	  result shapes: 4,635,408 B/dev/step at n_lon=1024 L26 f32 = 1.06x the
  1806	  single-row slab model; linear in n_lon (checked 1024 vs 2048, 0.07 %
  1807	  residual) -> **18.5 MB/dev/step at n_lon=4096 f32**. CAVEAT: CPU
  1808	  lowering; GPU-side collective combining could change the executed
  1809	  count (metadata.py:214) — the bound is a MODEL.
  1810	* **Same-tile nd=1 compute baselines** (job 26630370, roofline recipe):
  1811	  16x4096 f32 1.659 ms, 32x4096 f32 2.837, 16x4096 f64 2.973.
  1812	  Approximation, recorded: nd=1 includes pole tiles; the bias
  1813	  DIRECTION on the compute term is PLAUSIBLE-high, not proven
  1814	  (matters most for the compute-dominated f64 row).
  1815	* **Calibrated bound** (`metadata.calibrated_bound`, measured fabric
  1816	  constants: IB 26.3 us / 23.5 GB/s, NVLink 17.8 / 64.2):
  1817	
  1818	| row | measured | t_bound (IB) | measured/bound |
  1819	|---|---|---|---|
  1820	| LL2048@64 f32 | 6.732 | 2.863 | **2.35** |
  1821	| LL2048@128 f32 | 5.577 | 1.894 | **2.94** |
  1822	| LL2048@128 f64 | 9.602 | 2.999 | **3.20** |
  1823	
  1824	(Codex r7 corrected the @128 f32 row: the first draft fed the slab-byte
  1825	lower bound into a table labelled exact-bytes — 1.848/3.02 was the
  1826	mixed-input artefact; with the exact 18,541,632 B payload the bound is
  1827	1.894 ms. r7 also independently RERAN the census at nd=128 — 41 CP + 1
  1828	AR confirmed at the target device count, not just extrapolated from
  1829	8/16 — and measured the f64 census directly: 9,270,800 B at n_lon=1024,
  1830	four 4-byte scalar CPs staying f32, so the x2 extrapolation was 16 B
  1831	high.)
  1832	
  1833	* **The lat-lon GPU panel sits ~2.4-3.2x ABOVE this MODEL** (2.35-3.20)
  1834	  — the eff-0.60 strong leg is not close to the fabric+compute MODEL
  1835	  (a heuristic, not a proven floor). The 2-node/8-process IB
  1836	  calibration is extrapolated to a 32-node/128-process communicator. Leading
  1837	  PLAUSIBLE mechanism (uninstrumented): effective per-CP cost
  1838	  (launch + schedule + stream sync) well above the raw 26 us fabric
  1839	  latency across 41 dependency-chained exchanges — the arXiv:2607.16100
  1840	  small-collective regime. The model itself notes the serialized-latency
  1841	  vs overlap biases pull opposite ways; treat measured/bound as a
  1842	  consistency diagnostic, not proven headroom.
  1843	* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
  1844	  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
  1845	  same-job control + trailing A2 drift bracket. Interpretation limit:
  1846	  without a GPU post-pass CP census per arm, a null refutes THIS
  1847	  threshold/implementation, not combinable-CP count in general. The
  1848	  ocean-lane null for these flags came from a different
  1849	  implicit-PCG/dependency mix — not predictive for the atm lane either
  1850	  way.
=== job id / A-B references ===
scripts/bench/metadata.py:106:# ``xla_gpu_collective_permute_combine_threshold_bytes=`` (set by #1175's
scripts/bench/metadata.py:216:    combining / pipelined-p2p (``--xla_gpu_collective_permute_combine_*``, lane
scripts/cluster/scaling_levante/README.md:110:(`--xla_gpu_collective_permute_combine_threshold_bytes=32MiB` +
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:236:    _XLA_COMM_FLAGS="--xla_gpu_collective_permute_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:2:#SBATCH --job-name=ll128_comb
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:11:#SBATCH --output=ll128_comb.%j.log
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:36:OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll128_comb_j${SLURM_JOB_ID}}"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:49:run_arm B_combine "--xla_gpu_collective_permute_combine_threshold_bytes=8388608"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:50:run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:51:run_arm A2_default ""
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:53:for T in A_default B_combine C_combine_pipelined A2_default; do
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:97:_XLA_COMM_FLAGS="--xla_gpu_collective_permute_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_p2p=true"
docs/performance/scaling/spmd_message_census_2026-07-08.md:117:   (`--xla_gpu_collective_permute_combine_threshold_bytes`,
docs/performance/scaling/levante_campaign_2026-07-24.md:1478:`xla_gpu_collective_permute_combine_threshold_bytes` alone,
docs/performance/scaling/levante_campaign_2026-07-24.md:1843:* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
docs/performance/scaling/levante_campaign_2026-07-24.md:1844:  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:14:Round-8 confirm. Your r7 items, dispositions: (1) bound table corrected to 2.863/1.894/2.999 -> 2.35/2.94/3.20 with an explicit mixed-input-artefact note + your nd=128 rerun and f64 census recorded in doc; (2) probe: compile-once ordering kept simple but count-assert added (parsed cp_ops must equal census count), per-op dtype/dims/bytes records kept, X64 mode arg added; (3) wording: 'above this MODEL', calibration-extrapolation note, pole-bias PLAUSIBLE-direction; (4) A/B: A2 trailing control added (A,B,C,A2), interpretation-limit note (null refutes this threshold/impl only), ocean-null contrast reworded to dependency-mix difference; resubmitted as job 26630576. Read the doc tail section + scripts/tmp/_probe_latlon_halo_census.py + scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch. Consistent? End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + issues.
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:18:/usr/bin/bash -lc "pwd && rg --files -g '*probe_latlon_halo_census.py' -g 'atm_ll128_combine_ab.sbatch' -g '*.md' -g '*.txt' | sed -n '1,240p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:263:/usr/bin/bash -lc "rg -l -i '2\\.863|mixed-input|calibration-extrapolation|pole-bias|A2|26630576|above this MODEL' docs scripts tmp . 2>/dev/null | sed -n '1,160p'
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:264:for f in scripts/tmp/_probe_latlon_halo_census.py scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch; do
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:316:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:329:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:534:=== scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch ===
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:536:     2	#SBATCH --job-name=ll128_comb
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:545:    11	#SBATCH --output=ll128_comb.%j.log
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:570:    36	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll128_comb_j${SLURM_JOB_ID}}"
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:583:    49	run_arm B_combine "--xla_gpu_collective_permute_combine_threshold_bytes=8388608"
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:584:    50	run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:585:    51	run_arm A2_default ""
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:587:    53	for T in A_default B_combine C_combine_pipelined A2_default; do
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:909:  1843	* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:910:  1844	  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
.physics-validator/scaling_campaign/codex_bound_review_r8_2026-08-02.md:1186:rg -n -i '26630438|26630576|ll128_comb|collective_permute_combine|combine-8mb|A2_default' docs scripts .physics-validator 2>/dev/null | sed -n '1,260p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md:1547:1478-`xla_gpu_collective_permute_combine_threshold_bytes` alone,
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:17:3. Claims wording: '~2.4-3.2x above modeled limit', mechanism labelled PLAUSIBLE (per-CP effective overhead, arXiv:2607.16100 regime), A/B lever job 26630438 with falsifiability block (CONFIRM >=10%, REFUTE <=2%). Any overclaim/confound (e.g. f64 bytes: do 2-D f32 fields stay f32 under x64? census was f32-only), pole-tile compute bias direction, the ocean-null distinction argument.
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:22:/usr/bin/bash -lc "pwd && rg --files -g 'levante_campaign_2026-07-24.md' -g '_probe_latlon_halo_census.py' -g '26630370*' -g '26630438*' -g '*.csv' -g '*.json' | sed -n '1,240p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:56:/usr/bin/bash -lc "rg -n -C 5 \"modeled limit|calibrated_bound|26630370|26630438|distance-to|collective-permute|ocean-null|PL[A-Z]+IBLE|CONFIRM|REFUTE\" docs/performance/scaling/levante_campaign_2026-07-24.md scripts tmp . --glob '"'!*.ipynb'"' 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:326:docs/performance/scaling/levante_campaign_2026-07-24.md:1831:* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:327:docs/performance/scaling/levante_campaign_2026-07-24.md-1832-  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:373:./tests/bench/test_scaling_metadata.py-417-        "  // xla_gpu_collective_permute_combine_threshold_bytes=33554432",
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:716:scripts/bench/metadata.py-106-# ``xla_gpu_collective_permute_combine_threshold_bytes=`` (set by #1175's
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1100:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-10-#SBATCH --time=01:30:00
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1101:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-11-#SBATCH --output=ll128_comb.%j.log
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1102:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-12-# CP-COMBINING A/B at LL2048@128 (the measured/bound ~2.9 gap).
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1103:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-13-# The calibrated bound (census 41 CP + 1 AR/step, exact bytes 18.5 MB/dev,
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1104:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-14-# IB 26.3us/23.5GB/s, nd=1 same-tile compute 1.659 ms) models 1.85-1.89 ms;
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1105:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:15:# measured is 5.58 (job 26534060). Leading PLAUSIBLE mechanism: effective
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1106:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-16-# per-CP overhead (launch+schedule+sync) >> raw fabric latency across 41
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1107:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-17-# dependency-chained exchanges — the arXiv:2607.16100 regime. Lever:
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1108:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-18-# COMBINE independent CPs into fewer, larger messages.
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1109:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-19-# NOTE: the closed-levers null for these flags was the OCEAN lane
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1110:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-20-# (reduction-dominated); this is the first atm-latlon test — not a rerun
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1111:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-21-# of a closed null.
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1112:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-22-# Falsifiability, BEFORE submit — arms byte-matched to 26534060 protocol:
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1113:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-23-#   A control (default flags)      : expect ~5.6 ms
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1114:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:24:#   B +cp-combine 8MB threshold    : CONFIRM lever if >=10% under A
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1115:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-25-#   C +combine +pipelined-p2p      : scheduling interaction
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1116:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:26:#   REFUTE if B,C within 2% of A -> overhead is not combinable-CP count;
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1117:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-27-#   next hypothesis = unoverlapped serial chain (scheduling lever).
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1118:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-28-set -uo pipefail
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1119:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-29-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1120:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-30-export JAX_PLATFORMS=cuda,cpu
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1121:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-31-export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:1948:scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-97-_XLA_COMM_FLAGS="--xla_gpu_collective_permute_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_p2p=true"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:2631:./docs/performance/scaling/levante_campaign_2026-07-24.md:1831:* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:2632:./docs/performance/scaling/levante_campaign_2026-07-24.md-1832-  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:2950:./scripts/bench/metadata.py-106-# ``xla_gpu_collective_permute_combine_threshold_bytes=`` (set by #1175's
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3126:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-10-#SBATCH --time=01:30:00
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3127:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-11-#SBATCH --output=ll128_comb.%j.log
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3128:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-12-# CP-COMBINING A/B at LL2048@128 (the measured/bound ~2.9 gap).
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3129:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-13-# The calibrated bound (census 41 CP + 1 AR/step, exact bytes 18.5 MB/dev,
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3130:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-14-# IB 26.3us/23.5GB/s, nd=1 same-tile compute 1.659 ms) models 1.85-1.89 ms;
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3131:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:15:# measured is 5.58 (job 26534060). Leading PLAUSIBLE mechanism: effective
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3132:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-16-# per-CP overhead (launch+schedule+sync) >> raw fabric latency across 41
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3133:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-17-# dependency-chained exchanges — the arXiv:2607.16100 regime. Lever:
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3134:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-18-# COMBINE independent CPs into fewer, larger messages.
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3135:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-19-# NOTE: the closed-levers null for these flags was the OCEAN lane
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3136:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-20-# (reduction-dominated); this is the first atm-latlon test — not a rerun
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3137:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-21-# of a closed null.
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3138:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-22-# Falsifiability, BEFORE submit — arms byte-matched to 26534060 protocol:
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3139:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-23-#   A control (default flags)      : expect ~5.6 ms
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3140:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:24:#   B +cp-combine 8MB threshold    : CONFIRM lever if >=10% under A
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3141:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-25-#   C +combine +pipelined-p2p      : scheduling interaction
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3142:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:26:#   REFUTE if B,C within 2% of A -> overhead is not combinable-CP count;
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3143:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-27-#   next hypothesis = unoverlapped serial chain (scheduling lever).
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3144:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-28-set -uo pipefail
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3145:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-29-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3146:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-30-export JAX_PLATFORMS=cuda,cpu
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3147:./scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-31-export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3221:./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs-97-_XLA_COMM_FLAGS="--xla_gpu_collective_permute_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_p2p=true"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3646:/usr/bin/bash -lc "wc -l docs/performance/scaling/levante_campaign_2026-07-24.md && sed -n '1760,2120p' docs/performance/scaling/levante_campaign_2026-07-24.md && rg --files scripts/tmp 2>/dev/null | rg 'probe_latlon_halo_census|26630370|26630438|latlon.*(bound|census)' || true" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3720:* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3721:  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:3726:/usr/bin/bash -lc "ls -la scripts/tmp | sed -n '1,240p'; find scripts/tmp -maxdepth 1 -type f -name '*latlon*' -printf '%f\\n' 2>/dev/null; rg -n -C 8 \"calibrated_bound|def calibrated_bound|collective-permute|collective_permute|combine-8|pipelined|26630438|26630370|falsif|CONFIRM|REFUTE\" scripts/tmp packages docs/performance/scaling --glob '*.py' --glob '*.md' 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:4072:docs/performance/scaling/spmd_message_census_2026-07-08.md:117:   (`--xla_gpu_collective_permute_combine_threshold_bytes`,
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:4399:docs/performance/scaling/levante_campaign_2026-07-24.md:1478:`xla_gpu_collective_permute_combine_threshold_bytes` alone,
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:4504:docs/performance/scaling/levante_campaign_2026-07-24.md:1831:* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:4505:docs/performance/scaling/levante_campaign_2026-07-24.md:1832:  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:4818:rg -n -C 8 \"26630370|26630438|combine-8MB|combine.*8|pipelined-p2p\" scripts/tmp docs/performance/scaling --glob '*.sbatch' --glob '*.py' --glob '*.md'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:5202:docs/performance/scaling/levante_campaign_2026-07-24.md:1831:* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:5203:docs/performance/scaling/levante_campaign_2026-07-24.md:1832:  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:5229:rg -l \"26630438|combine-8MB|LL2048@128|latlon.*combine|CP-combin\" scripts scripts/tmp docs --glob '*.sbatch' --glob '*.py' --glob '*.md' | sort" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:5679:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:5685:/usr/bin/bash -lc "sed -n '1,300p' scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:5691:#SBATCH --job-name=ll128_comb
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:5700:#SBATCH --output=ll128_comb.%j.log
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:5724:OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll128_comb_j${SLURM_JOB_ID}}"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:5737:run_arm B_combine "--xla_gpu_collective_permute_combine_threshold_bytes=8388608"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:5738:run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:5893:git status --short && git diff -- docs/performance/scaling/levante_campaign_2026-07-24.md scripts/tmp/_probe_latlon_halo_census.py scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:5981:    combining / pipelined-p2p (``--xla_gpu_collective_permute_combine_*``, lane
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:6184:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:40:  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:6230:?? scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:6279:+* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:6280:+  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:6307:3. Claims wording: '~2.4-3.2x above modeled limit', mechanism labelled PLAUSIBLE (per-CP effective overhead, arXiv:2607.16100 regime), A/B lever job 26630438 with falsifiability block (CONFIRM >=10%, REFUTE <=2%). Any overclaim/confound (e.g. f64 bytes: do 2-D f32 fields stay f32 under x64? census was f32-only), pole-tile compute bias direction, the ocean-null distinction argument.
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:6312:/usr/bin/bash -lc "pwd && rg --files -g 'levante_campaign_2026-07-24.md' -g '_probe_latlon_halo_census.py' -g '26630370*' -g '26630438*' -g '*.csv' -g '*.json' | sed -n '1,240p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:6346:/usr/bin/bash -lc "rg -n -C 5 \"modeled limit|calibrated_bound|26630370|26630438|distance-to|collective-permute|ocean-null|PL[A-Z]+IBLE|CONFIRM|REFUTE\" docs/performance/scaling/levante_campaign_2026-07-24.md scripts tmp . --glob '"'!*.ipynb'"' 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:6872:        "  // xla_gpu_collective_permute_combine_threshold_bytes=33554432",
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9039:docs/performance/scaling/levante_campaign_2026-07-24.md-1831-* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9040:docs/performance/scaling/levante_campaign_2026-07-24.md:1832:  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9083:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-1-#!/bin/bash -l
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9084:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-2-#SBATCH --job-name=ll128_comb
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9085:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-3-#SBATCH --account=bb1596_gpu
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9086:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-4-#SBATCH --partition=gpu
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9087:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-5-#SBATCH --constraint=a100_80
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9088:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-6-#SBATCH --nodes=32
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9089:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-7-#SBATCH --gpus-per-node=4
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9090:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-8-#SBATCH --exclusive
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9091:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-9-#SBATCH --mem=0
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9092:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-10-#SBATCH --time=01:30:00
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9093:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-11-#SBATCH --output=ll128_comb.%j.log
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9094:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:12:# CP-COMBINING A/B at LL2048@128 (the measured/bound ~2.9 gap).
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9095:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-13-# The calibrated bound (census 41 CP + 1 AR/step, exact bytes 18.5 MB/dev,
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9096:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-14-# IB 26.3us/23.5GB/s, nd=1 same-tile compute 1.659 ms) models 1.85-1.89 ms;
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9097:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:15:# measured is 5.58 (job 26534060). Leading PLAUSIBLE mechanism: effective
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9098:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-16-# per-CP overhead (launch+schedule+sync) >> raw fabric latency across 41
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9099:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-17-# dependency-chained exchanges — the arXiv:2607.16100 regime. Lever:
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9100:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-18-# COMBINE independent CPs into fewer, larger messages.
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9101:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-19-# NOTE: the closed-levers null for these flags was the OCEAN lane
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9102:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-20-# (reduction-dominated); this is the first atm-latlon test — not a rerun
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9103:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-21-# of a closed null.
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9104:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:22:# Falsifiability, BEFORE submit — arms byte-matched to 26534060 protocol:
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9105:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-23-#   A control (default flags)      : expect ~5.6 ms
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9106:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-24-#   B +cp-combine 8MB threshold    : CONFIRM lever if >=10% under A
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9107:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-25-#   C +combine +pipelined-p2p      : scheduling interaction
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9108:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-26-#   REFUTE if B,C within 2% of A -> overhead is not combinable-CP count;
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9109:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-27-#   next hypothesis = unoverlapped serial chain (scheduling lever).
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9110:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-28-set -uo pipefail
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9111:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-29-SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9112:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-30-export JAX_PLATFORMS=cuda,cpu
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9113:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-31-export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9114:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-32-export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9115:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-33-source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9116:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-34-cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9117:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-35-OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll128_comb_j${SLURM_JOB_ID}}"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9118:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-36-mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9119:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-37-rc=0
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9120:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-38-run_arm () { # tag extra_xla_flags
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9121:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:39:  echo "=== LL2048@128 f32 arm=$1 XLA_EXTRA='$2' ==="
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9122:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-40-  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9123:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-41-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9124:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-42-    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9125:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-43-      --multicontroller --n-devices 128 --mode strong \
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9126:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-44-      --n-lat 2048 --n-lon 4096 --nlev 26 --steps 12 --warmup 3 \
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9127:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-45-      --out "$OUTDIR/$1.jsonl" || { echo "$1 FAILED"; rc=1; }
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9128:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-46-}
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9129:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-47-run_arm A_default ""
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9130:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-48-run_arm B_combine "--xla_gpu_collective_permute_combine_threshold_bytes=8388608"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9131:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-49-run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9132:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-50-echo "=== RESULTS ==="
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9133:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-51-for T in A_default B_combine C_combine_pipelined; do
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9256:scripts/tmp/ocean_overlap_test.sbatch-41-        combine)  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_collective_permute_combine_threshold_bytes=33554432" ;;
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9328:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-9-#SBATCH --mem=0
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9329:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-10-#SBATCH --time=01:30:00
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9330:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-11-#SBATCH --output=ll128_comb.%j.log
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9331:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-12-# CP-COMBINING A/B at LL2048@128 (the measured/bound ~2.9 gap).
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9332:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-13-# The calibrated bound (census 41 CP + 1 AR/step, exact bytes 18.5 MB/dev,
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9333:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:14:# IB 26.3us/23.5GB/s, nd=1 same-tile compute 1.659 ms) models 1.85-1.89 ms;
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9334:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-15-# measured is 5.58 (job 26534060). Leading PLAUSIBLE mechanism: effective
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9335:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-16-# per-CP overhead (launch+schedule+sync) >> raw fabric latency across 41
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9336:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-17-# dependency-chained exchanges — the arXiv:2607.16100 regime. Lever:
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9337:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-18-# COMBINE independent CPs into fewer, larger messages.
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9338:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-19-# NOTE: the closed-levers null for these flags was the OCEAN lane
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9762:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-37-rc=0
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9763:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-38-run_arm () { # tag extra_xla_flags
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9764:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-39-  echo "=== LL2048@128 f32 arm=$1 XLA_EXTRA='$2' ==="
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9765:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:40:  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9766:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-41-      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9767:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-42-    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9768:scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch-43-      --multicontroller --n-devices 128 --mode strong \
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9793:sed -n '1,180p' scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9988:#SBATCH --job-name=ll128_comb
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:9997:#SBATCH --output=ll128_comb.%j.log
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:10021:OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll128_comb_j${SLURM_JOB_ID}}"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:10034:run_arm B_combine "--xla_gpu_collective_permute_combine_threshold_bytes=8388608"
.physics-validator/scaling_campaign/codex_bound_review_r7_2026-08-02.md:10035:run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:463:`xla_gpu_collective_permute_combine_threshold_bytes` alone,

exec
/usr/bin/bash -lc "nl -ba scripts/bench/metadata.py | sed -n '180,270p'
printf '\\n=== Exact job/reference hits (excluding generated review transcript) ===\\n'
rg -n -i '26630438|26630576' --glob '"'!*.physics-validator/**'"' . | sed -n '1,160p'
printf '\\n=== Worktree status (requested artifacts) ===\\n'
git status --short -- docs/performance/scaling/levante_campaign_2026-07-24.md scripts/tmp/_probe_latlon_halo_census.py scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch
git diff --check -- docs/performance/scaling/levante_campaign_2026-07-24.md scripts/tmp/_probe_latlon_halo_census.py scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 613ms:
   180	    "all_reduce": _op_call_re("all-reduce"),
   181	    "all_gather": _op_call_re("all-gather"),
   182	    "all_to_all": _op_call_re("all-to-all"),
   183	    "reduce_scatter": _op_call_re("reduce-scatter"),
   184	}
   185	
   186	
   187	def count_collectives(hlo_text: str) -> dict[str, int]:
   188	    """Full per-family collective census of a lowered/compiled HLO text dump.
   189	
   190	    Superset of :func:`count_collective_permutes`: adds ``all_reduce`` (the
   191	    reduction wall that dominates ocean implicit-CN strong scaling and hides
   192	    from a permute-only count), ``all_gather``, ``all_to_all``,
   193	    ``reduce_scatter``.  Same STATIC, op-call-form discipline (see
   194	    :func:`_count_op_calls`: config-header flag names never inflate; an async
   195	    collective counts once via ``-start`` with its ``-done`` companion excluded
   196	    by the regex, NOT by a fragile line-wide substring test).  Keys are
   197	    underscore-normalized op names plus a ``total``.  Canonical census for the
   198	    scaling-diagnosis tool and the SPMD benches — no re-implementation."""
   199	    counts = {key: _count_op_calls(hlo_text, rx)
   200	              for key, rx in _COLLECTIVE_OP_RES.items()}
   201	    counts["total"] = sum(counts.values())
   202	    return counts
   203	
   204	
   205	def hlo_collective_census(fn, *args) -> dict[str, int] | None:
   206	    """Best-effort full collective census of the COMPILED HLO of ``fn(*args)``.
   207	
   208	    Superset of :func:`hlo_collective_permutes` — compiles (so combined /
   209	    pipelined collectives are reflected as executed, not as emitted) and runs
   210	    :func:`count_collectives`.  Returns ``None`` (never raises) if
   211	    lowering/compilation is unsupported OR the backend's ``as_text()`` yields
   212	    no HLO, so a probe records "unknown" not a crash.
   213	
   214	    NOTE the count is device-count- and FLAG-dependent: the DEFAULT schedule is
   215	    reproducible from a CPU virtual-device compile, but GPU-only XLA collective
   216	    combining / pipelined-p2p (``--xla_gpu_collective_permute_combine_*``, lane
   217	    T) can lower the optimized count — which is exactly why the cluster jobs run
   218	    this against the REAL on-device executable, not a CPU proxy."""
   219	    import jax
   220	    try:
   221	        text = jax.jit(fn).lower(*args).compile().as_text()
   222	        return count_collectives(text) if text else None
   223	    except Exception:
   224	        return None
   225	
   226	
   227	def _is_empty(v: Any) -> bool:
   228	    """True for a non-informative required value: ``None``, ``""``, or an EMPTY
   229	    container (e.g. ``precision_knobs={}`` — which would hide an f32/TF32
   230	    ablation).  Scalars ``0`` / ``0.0`` / ``False`` are NOT empty (a legitimate
   231	    ``n_gpus=0`` / ``gpu_direct_active=False`` must pass)."""
   232	    if v is None or v == "":
   233	        return True
   234	    if isinstance(v, (dict, list, tuple, set, frozenset)) and len(v) == 0:
   235	        return True
   236	    return False
   237	
   238	
   239	def git_sha(short: bool = True) -> str:
   240	    """Best-effort git commit SHA of THIS source tree (repro signature).
   241	
   242	    Every scaling row must be attributable to an exact code state — a
   243	    measurement without a SHA cannot be reproduced or compared across
   244	    branches (audit item 7: no harness recorded one).  A dirty working
   245	    tree is marked ``<sha>-dirty`` so the SHA never over-claims
   246	    reproducibility.  Fail-open (``"unknown"``): a missing git binary or
   247	    a non-repo run directory must never kill a benchmark.
   248	    """
   249	    import subprocess
   250	
   251	    cwd = os.path.dirname(os.path.abspath(__file__))
   252	    try:
   253	        cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
   254	        out = subprocess.run(cmd, capture_output=True, text=True,
   255	                             timeout=5, cwd=cwd)
   256	        sha = out.stdout.strip()
   257	        if out.returncode != 0 or not sha:
   258	            return "unknown"
   259	        dirty = subprocess.run(
   260	            ["git", "status", "--porcelain", "--untracked-files=no"],
   261	            capture_output=True, text=True, timeout=5, cwd=cwd)
   262	        if dirty.returncode == 0 and dirty.stdout.strip():
   263	            return sha + "-dirty"
   264	        return sha
   265	    except Exception:
   266	        return "unknown"
   267	
   268	
   269	def detect_backend() -> str:
   270	    """Live JAX backend ("cpu"/"gpu"/"tpu"), or "unknown" if JAX is absent."""

=== Exact job/reference hits (excluding generated review transcript) ===
./docs/performance/scaling/levante_campaign_2026-07-24.md:1843:* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at

=== Worktree status (requested artifacts) ===
 M docs/performance/scaling/levante_campaign_2026-07-24.md
?? scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch

codex
The substantive r7 corrections are present. I’m checking the remaining cross-artifact details now—especially the current submission receipt and whether the A/B wording precisely matches the four executed runs.
exec
/usr/bin/bash -lc "printf '%s\\n' '=== bench output / multicontroller contract ==='
rg -n -A12 -B12 'args\\.out|--out|multicontroller|process_index|json\\.dump|jsonl' scripts/bench/bench_atm_latlon_spmd_scaling.py | sed -n '1,360p'
printf '%s\\n' '=== submitted job state / receipts available locally ==='
rg --files | rg '26630(438|576)|ll128_comb|combine' | sed -n '1,160p'
printf '%s\\n' '=== numbered exact final doc block ==='
nl -ba docs/performance/scaling/levante_campaign_2026-07-24.md | sed -n '1795,1852p'
printf '%s\\n' '=== shell parse syntax ==='
bash -n scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch && echo 'bash syntax: OK'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 312ms:
=== bench output / multicontroller contract ===
27-``valid=false`` (with ``completed_blocks`` saying how far it got, a
28-``diverged: ...`` entry under ``metadata._incomplete``, and nulled
29-``sypd``/``mcells_per_s``) — a diverging trajectory is never serialized as
30-valid scaling data.  The default fused lane has no in-graph finite check, so
31-its rows carry ``finite_ok=null``.
32-
33-Device count is fixed at process start, so each n_devices runs as a SEPARATE
34-process (one sbatch step per count); this script benches ONE n_devices and
35-appends a JSON line. JAX_PLATFORMS=cpu with --xla_force_host_platform_device_count
36-gives virtual CPU devices (communication-overhead characterization, NOT a real
37-speedup); a real number needs one GPU per band.
38-
39:Multi-controller (route-B, ``--multicontroller``): the lat-lon analogue of the
40-cubed-sphere ``run_cpu_mpi_scaling --cs-spmd`` A1 path. Every process calls
41-``jax.distributed.initialize`` BEFORE any other JAX use, the ("lat",) mesh is
42-built over the GLOBAL ``jax.devices()`` (all processes), and the existing
43-``make_sharded_atm_latlon_step`` + band-ppermute halo runs unchanged — the
44-ppermute/psum collectives cross processes via the distributed runtime (NCCL on
45-GPU / gloo on CPU). NO mpi4jax is armed in this mode (mixing the mpi4jax halo
46-machinery with jax.distributed collectives in one program is the documented
47-mixed-stack deadlock hazard — see run_cpu_mpi_scaling._build_cubed_sphere_spmd).
48-This is the halo path that keeps intra-node GPU traffic on NCCL and bypasses
49-the host-staged / CXI-inject-broken cross-node GPU-direct MPI route (see
50-docs/performance/multinode_gpu_direct_cxi.md).
51-
52-Launch (cluster, one process per GPU):
53:  srun -n 8 python bench_atm_latlon_spmd_scaling.py --multicontroller \
54-      --n-devices 8 ...            # SLURM: coordinator auto-detected
55:  mpiexec -n 8 python ... --multicontroller --coordinator host0:9876
56-"""
57-from __future__ import annotations
58-
59-import argparse
60-import json
61-import os
62-import time
63-
64-import sys
65-from pathlib import Path
66-
67-import jax
--
152-                   help="fused_step_ms of the nd=1 row at the SAME per-device "
153-                        "size (compute ingredient of the calibrated T_bound, "
154-                        "audit item 8). Omitted at nd>1 -> bound emitted null "
155-                        "+ flagged incomplete; nd=1 uses its own measurement.")
156-    p.add_argument("--comm-latency-us", type=float, default=None,
157-                   help="MEASURED per-message latency [us] of THIS machine's "
158-                        "fabric. Default: MACHINE-CALIBRATED-REQUIRED "
159-                        "placeholder in metadata.py -> bound_calibrated=false.")
160-    p.add_argument("--comm-bandwidth-gbs", type=float, default=None,
161-                   help="MEASURED link bandwidth [GB/s] of THIS machine's "
162-                        "fabric. Default: MACHINE-CALIBRATED-REQUIRED "
163-                        "placeholder in metadata.py -> bound_calibrated=false.")
164:    p.add_argument("--out", type=str, default="results/a1/spmd_scaling.jsonl")
165:    p.add_argument("--multicontroller", action="store_true",
166-                   help="Route-B multi-controller: jax.distributed.initialize "
167-                        "per process, ('lat',) mesh over the GLOBAL device set "
168-                        "(one process per GPU / per CPU-device group). NO "
169-                        "mpi4jax. --n-devices must equal the global device "
170-                        "count.")
171-    p.add_argument("--coordinator", type=str, default=None,
172-                   help="host:port for jax.distributed when auto-detection "
173-                        "(SLURM) is unavailable; process count/id then come "
174-                        "from OMPI_COMM_WORLD_SIZE/RANK.")
175-    args = p.parse_args()
176-
177-    # Validate the schedule BEFORE any model/device work: a zero/negative
178-    # --steps would otherwise surface only as timed_scan_blocks' None
179-    # headline (default lane) or an empty timed loop (segment mode) after
180-    # the expensive build (the ocean twin's guard).
181-    if args.steps < 1:
182-        raise SystemExit(f"--steps must be >= 1, got {args.steps}")
183-
184:    if args.multicontroller:
185-        # MUST run before any other JAX use (backend init).  The SHARED
186-        # helper owns the launcher-env contract (SLURM/OMPI auto-detect,
187-        # PALS mpi4py bootstrap, explicit-coordinator path) AND the
188-        # hardening: post-init silent-fallback guard + NCCL net-plugin
189-        # warning — an inline init here would bypass both (codex).
190-        from legoesm.parallel.early_init import (
191:            init_multicontroller_distributed,
192-        )
193:        init_multicontroller_distributed(args.coordinator)
194-
195-    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
196-        atm_latlon_geometry_bytes,
197-        build_sharded_held_suarez_state_atm_latlon,
198-        make_sharded_atm_latlon_segment,
199-        make_sharded_atm_latlon_step)
200-    seg_n = int(args.segment_steps)
201-    if seg_n < 0:
202-        raise SystemExit(f"--segment-steps must be >= 0, got {seg_n}")
203-    physics_fn = None
204-    if args.physics == "held_suarez":
205-        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_latlon
206-        physics_fn = held_suarez_forcing_latlon
207-
208-    nd = args.n_devices
209-    avail = len(jax.devices())
210-    if avail < nd:
211-        raise SystemExit(f"need {nd} devices, have {avail} "
212-                         f"(set --xla_force_host_platform_device_count)")
213:    if args.multicontroller and nd != avail:
214-        # A mesh over a strict subset would leave some processes' devices out
215-        # of the program (non-addressable participation hazard). Route-B uses
216-        # ALL global devices: one band per device across every process.
217-        raise SystemExit(
218:            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
219-            f"device count ({avail} across {jax.process_count()} processes).")
220-    n_lat = args.n_lat if args.mode == "strong" else args.nlat_per_dev * nd
221-    if n_lat % nd != 0:
222-        raise SystemExit(f"n_lat {n_lat} not divisible by n_devices {nd}")
223-
224-    if nd == 1:
225-        model, c0 = _build(n_lat, args.n_lon, args.nlev)
226-        mesh = None
227-        c = c0
228-    else:
229-        # #1100: band-local IC construction. The nd>1 lanes never materialise
230-        # the global (n_lat, n_lon, nlev) state per process — each leaf is
--
358-        # the segment lane records no cross-process block gather -> null.
359-        rank_imbalance=(float(timing["rank_imbalance"])
360-                        if timing is not None else None),
361-        latency_us=args.comm_latency_us,
362-        bandwidth_GBs=args.comm_bandwidth_gbs,
363-    )
364-
365-    rec = dict(
366-        mode=args.mode, n_devices=nd, n_lat=n_lat, n_lon=args.n_lon,
367-        nlev=args.nlev, physics=args.physics, steps=args.steps,
368-        platform=jax.default_backend(),
369-        n_processes=jax.process_count(),
370:        multicontroller=bool(args.multicontroller),
371-        segment_mode=(seg_n > 0),
372-        segment_steps=(seg_n if seg_n > 0 else None),
373-        # Measurement validity (codex batch4): finite_ok is the ACCUMULATED
374-        # in-graph finite verdict (null in the unchecked default fused lane);
375-        # valid=false marks the row as NOT scaling data; completed_blocks
376-        # says where a diverging segment run stopped.
377-        finite_ok=finite_ok,
378-        valid=valid,
379-        completed_blocks=completed_blocks,
380-        steady_median_ms=round(med, 4),
381-        cells=n_lat * args.n_lon * args.nlev,
382-    )
--
434-                else 0),
435-        decomposition="band" if nd > 1 else "none",
436-        # cells_per_rank is per PROCESS (n_ranks semantics); the per-device
437-        # share lives in extra.cells_per_device — a single-process 4-device
438-        # SPMD run has 1 rank owning ALL cells (codex finding 3).
439-        cells_per_rank=(n_lat * args.n_lon * args.nlev)
440-        // max(jax.process_count(), 1),
441-        scaling_kind=args.mode,
442-        extra={
443-            "physics": args.physics,
444-            "steps": args.steps,
445-            "warmup": args.warmup,
446:            "multicontroller": bool(args.multicontroller),
447-            # M2b compiled-segment lane facts: a segment row is falsifiable
448-            # from the record alone (block timings + geometry residency +
449-            # the finite/validity verdict — codex batch4).
450-            "segment_mode": seg_n > 0,
451-            "segment_steps": (seg_n if seg_n > 0 else None),
452-            "geometry_bytes_per_device": geom_bytes,
453-            "finite_ok": finite_ok,
454-            "valid": valid,
455-            "completed_blocks": completed_blocks,
456-            # Route-B transport facts (socket-fallback flag): a
457-            # multi-node row without an NCCL net plugin is
458-            # falsifiable from the record alone.
459:            "nccl": (_nccl_report if args.multicontroller
460-                     else None),
461-            "cells_per_device": (n_lat // nd) * args.n_lon * args.nlev,
462-        },
463-    ))
464-    if not valid:
465-        # Divergence reason on the aggregator-facing incomplete list (the
466-        # same channel annotate_incomplete uses for missing metadata).
467-        rec["metadata"].setdefault("_incomplete", []).append(
468-            f"diverged: segment finite scalar false after block "
469-            f"{completed_blocks - 1} of {args.steps} — timings describe a "
470-            "non-finite trajectory, not valid scaling data")
471-    # Multi-controller: every process times the same program; process 0 owns
472-    # the JSONL + stdout (others would duplicate/corrupt the append).
473:    if jax.process_index() == 0:
474:        os.makedirs(os.path.dirname(args.out), exist_ok=True)
475:        with open(args.out, "a") as f:
476:            f.write(json.dumps(rec) + "\n")
477:        print(json.dumps(rec))
478-        seg_note = (f" segment[{seg_n}-step blocks]" if seg_n > 0 else "")
479-        invalid_note = ("" if valid
480-                        else " INVALID[diverged: finite_ok=false]")
481-        head = (f"[nd={nd} {args.mode} {n_lat}x{args.n_lon}x{args.nlev}"
482-                f"{seg_note}]{invalid_note} ")
483-        if seg_n > 0:
484-            print(head +
485-                  f"compile={rec['compile_ms']}ms "
486-                  f"steady_median={med:.2f}ms/step "
487-                  f"(per-step: {rec['per_step_ms']})")
488-        else:
489-            print(head +
=== submitted job state / receipts available locally ===
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch
tests/ocean/unit/test_combined_pipeline_tke_evd_gate.py
tests/ocean/unit/test_tracer_combine_conservation.py
packages/ocean/legoesm/ocean/physics/combined.py
tests/unit/test_physics_combined.py
tests/atmosphere/hydrostatic/unit/test_aerosol_ccn_combined.py
tests/atmosphere/hydrostatic/unit/test_combined_physics.py
packages/atmosphere/legoesm/atmosphere/physics/combined.py
=== numbered exact final doc block ===
  1795	## Distance-to-modeled-limit: atm lat-lon GPU (2026-08-02, "near theoretical limit" directive)
  1796	
  1797	Closed the bench's own honest-null bound gap (audit item 4) for the
  1798	lat-lon lane, using only repo instruments:
  1799	
  1800	* **Halo census** (new probe `scripts/tmp/_probe_latlon_halo_census.py`,
  1801	  virtual-CPU forced-host-platform lowering of the REAL
  1802	  `make_sharded_atm_latlon_step`): **41 collective-permutes + 1
  1803	  all-reduce per step**, nd-INDEPENDENT (identical at nd=8 and nd=16 —
  1804	  the 1-D band structure check). Exact CP payload from compiled-HLO
  1805	  result shapes: 4,635,408 B/dev/step at n_lon=1024 L26 f32 = 1.06x the
  1806	  single-row slab model; linear in n_lon (checked 1024 vs 2048, 0.07 %
  1807	  residual) -> **18.5 MB/dev/step at n_lon=4096 f32**. CAVEAT: CPU
  1808	  lowering; GPU-side collective combining could change the executed
  1809	  count (metadata.py:214) — the bound is a MODEL.
  1810	* **Same-tile nd=1 compute baselines** (job 26630370, roofline recipe):
  1811	  16x4096 f32 1.659 ms, 32x4096 f32 2.837, 16x4096 f64 2.973.
  1812	  Approximation, recorded: nd=1 includes pole tiles; the bias
  1813	  DIRECTION on the compute term is PLAUSIBLE-high, not proven
  1814	  (matters most for the compute-dominated f64 row).
  1815	* **Calibrated bound** (`metadata.calibrated_bound`, measured fabric
  1816	  constants: IB 26.3 us / 23.5 GB/s, NVLink 17.8 / 64.2):
  1817	
  1818	| row | measured | t_bound (IB) | measured/bound |
  1819	|---|---|---|---|
  1820	| LL2048@64 f32 | 6.732 | 2.863 | **2.35** |
  1821	| LL2048@128 f32 | 5.577 | 1.894 | **2.94** |
  1822	| LL2048@128 f64 | 9.602 | 2.999 | **3.20** |
  1823	
  1824	(Codex r7 corrected the @128 f32 row: the first draft fed the slab-byte
  1825	lower bound into a table labelled exact-bytes — 1.848/3.02 was the
  1826	mixed-input artefact; with the exact 18,541,632 B payload the bound is
  1827	1.894 ms. r7 also independently RERAN the census at nd=128 — 41 CP + 1
  1828	AR confirmed at the target device count, not just extrapolated from
  1829	8/16 — and measured the f64 census directly: 9,270,800 B at n_lon=1024,
  1830	four 4-byte scalar CPs staying f32, so the x2 extrapolation was 16 B
  1831	high.)
  1832	
  1833	* **The lat-lon GPU panel sits ~2.4-3.2x ABOVE this MODEL** (2.35-3.20)
  1834	  — the eff-0.60 strong leg is not close to the fabric+compute MODEL
  1835	  (a heuristic, not a proven floor). The 2-node/8-process IB
  1836	  calibration is extrapolated to a 32-node/128-process communicator. Leading
  1837	  PLAUSIBLE mechanism (uninstrumented): effective per-CP cost
  1838	  (launch + schedule + stream sync) well above the raw 26 us fabric
  1839	  latency across 41 dependency-chained exchanges — the arXiv:2607.16100
  1840	  small-collective regime. The model itself notes the serialized-latency
  1841	  vs overlap biases pull opposite ways; treat measured/bound as a
  1842	  consistency diagnostic, not proven headroom.
  1843	* **Lever test submitted (job 26630438)**: 3-arm CP-combining A/B at
  1844	  LL2048@128 (default / combine-8MB / combine+pipelined-p2p),
  1845	  same-job control + trailing A2 drift bracket. Interpretation limit:
  1846	  without a GPU post-pass CP census per arm, a null refutes THIS
  1847	  threshold/implementation, not combinable-CP count in general. The
  1848	  ocean-lane null for these flags came from a different
  1849	  implicit-PCG/dependency mix — not predictive for the atm lane either
  1850	  way.
=== shell parse syntax ===
bash syntax: OK

exec
/usr/bin/bash -lc "nl -ba scripts/bench/metadata.py | sed -n '130,190p'
printf '\\n=== Parser matching strings/search test corpus ===\\n'
rg -n -C 7 'collective-permute(?:-start)?|cp_ops_with_shape|cp_records|exact per-CP|result bytes' scripts tests docs | sed -n '1,360p'
printf '\\n=== Collectives and flag setup in environment helper ===\\n'
nl -ba scripts/cluster/scaling_levante/_env.sh | sed -n '1,240p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 394ms:
   130	    it is the round-count metric #1113 needs to decompose multi-node overhead.
   131	    Matches the op-call form only (StableHLO underscore + optimized-XLA hyphen,
   132	    async ``-start`` counted once, ``-done`` companion excluded by the regex),
   133	    so config-header flag names that merely CONTAIN "collective_permute" never
   134	    inflate the count.  Canonical for every bench that reports
   135	    ``hlo_collective_permutes`` (cube tiled, MPAS ico) — no re-implementation."""
   136	    return _count_op_calls(hlo_text, _COLLECTIVE_PERMUTE_RE)
   137	
   138	
   139	def hlo_collective_permutes(fn, *args) -> int | None:
   140	    """Best-effort: count the collective-permutes in the COMPILED HLO of
   141	    ``fn(*args)``.
   142	
   143	    Compiles (``.lower(...).compile().as_text()``), NOT bare
   144	    ``.lower().as_text()``: the census must reflect the EXECUTABLE's round
   145	    count, because XLA collective-permute combining / pipelined-p2p
   146	    (#1175, ``MPAS_CP_COMBINE``) fuses rounds during optimization — the whole
   147	    metric #1113 tracks. Pre-optimization StableHLO would overstate CPs versus
   148	    the timed executable. Matches the cube tiled bench, which compiles too.
   149	    Returns ``None`` (never raises) if lowering/compilation is unsupported OR
   150	    the backend's ``as_text()`` yields no HLO, so a timing probe can record
   151	    "unknown" rather than crash."""
   152	    import jax
   153	    try:
   154	        text = jax.jit(fn).lower(*args).compile().as_text()
   155	        return count_collective_permutes(text) if text else None
   156	    except Exception:
   157	        return None
   158	
   159	
   160	def _op_call_re(op_name: str) -> "re.Pattern[str]":
   161	    """Op-call-form matcher for a single HLO collective ``op_name``.
   162	
   163	    ``op_name`` is the hyphen spelling (``"all-reduce"``).  Matches BOTH the
   164	    optimized-XLA hyphen and StableHLO underscore spellings, an optional async
   165	    ``-start``/``_start`` suffix, and requires the ``(`` op-call form so a
   166	    config-header ``XLA_FLAGS`` echo that merely CONTAINS the op name can never
   167	    inflate the count (same guard as :data:`_COLLECTIVE_PERMUTE_RE`)."""
   168	    stem = op_name.replace("-", "[_-]")
   169	    return re.compile(stem + r"(?:[_-]start)?\(")
   170	
   171	
   172	#: Every collective OP family a scaling row can run.  ``collective-permute`` is
   173	#: the band/face halo (reuse the canonical permute regex so its count stays
   174	#: bit-identical to :func:`count_collective_permutes`); ``all-reduce`` is the
   175	#: conservation fixer AND the ocean implicit-CN PCG reduction wall (~120/step —
   176	#: the #1 ocean strong-scaling bottleneck, invisible to a permute-only census);
   177	#: the rest surface any SPMD resharding an operator introduces.
   178	_COLLECTIVE_OP_RES: dict[str, "re.Pattern[str]"] = {
   179	    "collective_permute": _COLLECTIVE_PERMUTE_RE,
   180	    "all_reduce": _op_call_re("all-reduce"),
   181	    "all_gather": _op_call_re("all-gather"),
   182	    "all_to_all": _op_call_re("all-to-all"),
   183	    "reduce_scatter": _op_call_re("reduce-scatter"),
   184	}
   185	
   186	
   187	def count_collectives(hlo_text: str) -> dict[str, int]:
   188	    """Full per-family collective census of a lowered/compiled HLO text dump.
   189	
   190	    Superset of :func:`count_collective_permutes`: adds ``all_reduce`` (the

=== Parser matching strings/search test corpus ===
tests/bench/test_scaling_metadata.py-363-
tests/bench/test_scaling_metadata.py-364-
tests/bench/test_scaling_metadata.py-365-def test_count_collective_permutes_matches_hyphen_and_underscore():
tests/bench/test_scaling_metadata.py-366-    """Canonical CP census (#1113): counts StableHLO underscore + optimized
tests/bench/test_scaling_metadata.py-367-    hyphen spellings, and drops the async ``-done`` companion so one logical
tests/bench/test_scaling_metadata.py-368-    exchange counts once."""
tests/bench/test_scaling_metadata.py-369-    hlo = "\n".join([
tests/bench/test_scaling_metadata.py:370:        "  %a = collective-permute(%x)",          # optimized HLO
tests/bench/test_scaling_metadata.py-371-        "  %b = collective_permute(%y)",          # StableHLO
tests/bench/test_scaling_metadata.py:372:        "  %c = collective-permute-done(%a)",     # async companion -> excluded
tests/bench/test_scaling_metadata.py-373-        "  %d = collective_permute_done(%b)",     # async companion -> excluded
tests/bench/test_scaling_metadata.py-374-        "  %e = all-gather(%z)",                  # different collective
tests/bench/test_scaling_metadata.py-375-    ])
tests/bench/test_scaling_metadata.py-376-    assert md.count_collective_permutes(hlo) == 2
tests/bench/test_scaling_metadata.py-377-    assert md.count_collective_permutes("no collectives here") == 0
tests/bench/test_scaling_metadata.py-378-
tests/bench/test_scaling_metadata.py-379-
--
tests/bench/test_scaling_metadata.py-390-    executable.
tests/bench/test_scaling_metadata.py-391-    """
tests/bench/test_scaling_metadata.py-392-    import jax
tests/bench/test_scaling_metadata.py-393-    return jax.default_device(jax.devices("cpu")[0])
tests/bench/test_scaling_metadata.py-394-
tests/bench/test_scaling_metadata.py-395-
tests/bench/test_scaling_metadata.py-396-def test_hlo_collective_permutes_lowers_counts_and_is_error_safe():
tests/bench/test_scaling_metadata.py:397:    """The best-effort probe lowers a fn and counts its collective-permutes: a
tests/bench/test_scaling_metadata.py-398-    fn with none -> 0; an unlowerable fn -> None (never raises). Real ppermute
tests/bench/test_scaling_metadata.py-399-    counting is exercised by the MPAS/cube bench gates and
tests/bench/test_scaling_metadata.py-400-    count_collective_permutes' synthetic HLO test above."""
tests/bench/test_scaling_metadata.py-401-    import jax.numpy as jnp
tests/bench/test_scaling_metadata.py-402-    with _cpu_compile():
tests/bench/test_scaling_metadata.py-403-        assert md.hlo_collective_permutes(lambda x: x + 1, jnp.arange(4.0)) == 0
tests/bench/test_scaling_metadata.py-404-
--
tests/bench/test_scaling_metadata.py-411-    """Full census counts every collective family with the same op-call-form
tests/bench/test_scaling_metadata.py-412-    discipline: async ``-start`` once (``-done`` excluded), StableHLO
tests/bench/test_scaling_metadata.py-413-    underscore + optimized hyphen, and a config-header flag echo that merely
tests/bench/test_scaling_metadata.py-414-    CONTAINS an op name never inflates the count."""
tests/bench/test_scaling_metadata.py-415-    hlo = "\n".join([
tests/bench/test_scaling_metadata.py-416-        # config-header echo of XLA_FLAGS -> must NOT match (no op-call paren)
tests/bench/test_scaling_metadata.py-417-        "  // xla_gpu_collective_permute_combine_threshold_bytes=33554432",
tests/bench/test_scaling_metadata.py:418:        "  %a = collective-permute(%x)",             # permute (optimized)
tests/bench/test_scaling_metadata.py-419-        "  %b = collective_permute(%y)",             # permute (StableHLO)
tests/bench/test_scaling_metadata.py:420:        "  %c = collective-permute-done(%a)",        # async companion -> drop
tests/bench/test_scaling_metadata.py-421-        "  %r1 = all-reduce(%p)",                    # reduction (hyphen)
tests/bench/test_scaling_metadata.py-422-        "  %r2 = all_reduce_start(%q)",              # async reduction -> count once
tests/bench/test_scaling_metadata.py-423-        "  %r3 = all-reduce-done(%r2)",              # async companion -> drop
tests/bench/test_scaling_metadata.py-424-        "  %g = all-gather(%z)",                     # all-gather
tests/bench/test_scaling_metadata.py-425-        "  %a2a = all-to-all(%w)",                   # all-to-all
tests/bench/test_scaling_metadata.py-426-        "  %rs = reduce-scatter(%v)",                # reduce-scatter
tests/bench/test_scaling_metadata.py-427-    ])
--
tests/bench/test_scaling_metadata.py-445-    """Regression (codex): the ``-done`` async COMPANION is excluded by the
tests/bench/test_scaling_metadata.py-446-    regex structurally (``op-done(`` never matches ``op(?:[_-]start)?\\(``), so
tests/bench/test_scaling_metadata.py-447-    a line that merely CONTAINS the substring "done" elsewhere — an XLA
tests/bench/test_scaling_metadata.py-448-    metadata op_name, a ``%done_*`` SSA name — must STILL be counted.  A blunt
tests/bench/test_scaling_metadata.py-449-    ``"done" not in line`` filter would false-drop these to zero."""
tests/bench/test_scaling_metadata.py-450-    hlo = "\n".join([
tests/bench/test_scaling_metadata.py-451-        '  %r = all-reduce(%p), metadata={op_name="jit(step)/done_stage/psum"}',
tests/bench/test_scaling_metadata.py:452:        '  %done_mass = f32[] collective-permute(%q)',
tests/bench/test_scaling_metadata.py-453-        '  %g = all-gather(%z), metadata={op_name="reduce_done/x"}',
tests/bench/test_scaling_metadata.py-454-    ])
tests/bench/test_scaling_metadata.py-455-    c = md.count_collectives(hlo)
tests/bench/test_scaling_metadata.py-456-    assert c["all_reduce"] == 1        # NOT dropped despite "done" in metadata
tests/bench/test_scaling_metadata.py-457-    assert c["collective_permute"] == 1  # NOT dropped despite %done_ SSA name
tests/bench/test_scaling_metadata.py-458-    assert c["all_gather"] == 1
tests/bench/test_scaling_metadata.py-459-    # canonical permute helper is fixed by the same shared counter
--
scripts/bench/bench_mpas_spmd_scaling.py-411-        _block(s)
scripts/bench/bench_mpas_spmd_scaling.py-412-        per_step_ms.append((time.perf_counter() - t0) * 1e3)
scripts/bench/bench_mpas_spmd_scaling.py-413-
scripts/bench/bench_mpas_spmd_scaling.py-414-    if jax.process_count() > 1:
scripts/bench/bench_mpas_spmd_scaling.py-415-        from jax.experimental import multihost_utils
scripts/bench/bench_mpas_spmd_scaling.py-416-        multihost_utils.sync_global_devices("mpas_spmd_bench_end")
scripts/bench/bench_mpas_spmd_scaling.py-417-
scripts/bench/bench_mpas_spmd_scaling.py:418:    # HLO collective-permute census (#1113 ask 2): a STATIC compile property of
scripts/bench/bench_mpas_spmd_scaling.py-419-    # the sharded step — the ppermute ROUND count that decomposes multi-node
scripts/bench/bench_mpas_spmd_scaling.py-420-    # overhead (overhead ~= CPs/step * ~0.11 ms launch floor). The cube benches
scripts/bench/bench_mpas_spmd_scaling.py-421-    # record this; the MPAS row did not, forcing an out-of-band census. Counted
scripts/bench/bench_mpas_spmd_scaling.py-422-    # AFTER the timed loop so the census compile can't perturb per_step_ms[0]'s
scripts/bench/bench_mpas_spmd_scaling.py-423-    # compile timing (the executable is already cached — this re-lower/compile
scripts/bench/bench_mpas_spmd_scaling.py-424-    # is a cache hit; the count is data-independent, static in the partition).
scripts/bench/bench_mpas_spmd_scaling.py-425-    # Best-effort (None if compilation is unsupported); the serial n=1 leg has
--
docs/performance/scaling/SCALING_STATUS_AUDIT.md-21-
docs/performance/scaling/SCALING_STATUS_AUDIT.md-22-## Atmosphere support matrix
docs/performance/scaling/SCALING_STATUS_AUDIT.md-23-
docs/performance/scaling/SCALING_STATUS_AUDIT.md-24-| grid | CPU-MPI | GPU / SPMD | true weak/strong evidence (a) | known blockers | next measurement |
docs/performance/scaling/SCALING_STATUS_AUDIT.md-25-|------|---------|------------|-------------------------------|----------------|------------------|
docs/performance/scaling/SCALING_STATUS_AUDIT.md-26-| **spectral** | (d) single-rank only (global Legendre transforms; no MPI path) | (d) N/A — both multi-device schemes measured anti-scaling (`spectral_level_shard_cliff.md`); fp64-only | none possible | O(N³) global transform | none — stays N/A unless a GPU-native SHT effort (SHTns/sphericart) is explicitly launched |
docs/performance/scaling/SCALING_STATUS_AUDIT.md-27-| **cubed-sphere** | (a) genuine ≤6-face decomposition via `run_levante_gpu_scaling.py --cs-mpi-scatter` (bit-equal 1e-15 vs serial); default (no flag) is replicated dynamics, refused for scaling claims. Gloo/TCP multinode anti-scales (fabric, not code — `multinode_clean`) | (a)≤6 devices: face-sharded SPMD (`--cs-spmd`, single-process or multi-controller NCCL); Derecho/Levante NCCL job lanes exist (`scripts/cluster/scaling_*`) | 2-GPU strong 0.73–0.83 eff (Ginsburg, AT the PCIe roofline of that host) | >6 GPUs: (c) sub-face tiled production lanes wired (`_run_tiled_cube_spmd` blocked loop, per-segment `lax.scan` since M3b inc-1 — see lever 7) but NO >6-GPU hardware receipts; envelope = default-config dycore + Kessler / operator-split unified physics | production-size Derecho/Levante multi-node NCCL runs incl. the >6-GPU tiled lanes |
docs/performance/scaling/SCALING_STATUS_AUDIT.md:28:| **icosahedral / MPAS** | (a) graph-partition domain decomposition (METIS/RCB/SFC `auto`), validated vs serial ~1e-9 | (a) multi-GPU via route-A mpi4jax (opaque to XLA overlap); (c) route-B multi-controller NCCL SPMD wired since M3c #981 (native ppermute step, `bench_mpas_spmd_scaling --multicontroller`, cluster lane E; XLA collective-permute combining defaulted for the lane, #1113) — no production-scale receipts yet | Derecho 2026-07-02: CPU-MPI to 128 ranks f32 ~75× (eff ~0.59); GPU 1→16 A100 @28 km ~6× (eff ~0.38), coarse grids flat (per-device floor, not a defect) | route-A mpi4jax leg is latency-bound, no comm/compute overlap; route-B ppermute round count (edge-coloring rounds × launch latency, #1113) | production-size lane-E runs; re-measure 8→16 GPU leg on native ppermute |
docs/performance/scaling/SCALING_STATUS_AUDIT.md-29-| **lat-lon** | (a) latitude-band decomposition (`make_latlon_mpi_step`, #641; pole_bc='wall'), validated; driven by `run_levante_gpu_scaling.py --grid latlon` | (a) lat-band SPMD `bench_atm_latlon_spmd_scaling.py`: single-process multi-device + `--multicontroller` route-B (native NCCL ppermute, no mpi4jax) | Derecho: CPU-MPI 128 ranks f32 ~30× (eff ~0.23 — 1-D band perimeter cost, as designed); GPU 16 A100 @28 km 7.5× (eff ~0.47, route-A). Ginsburg 2-GPU: strong 1.10×, weak eff 0.64 (PCIe-capped) | 1-D band decomposition perimeter at high rank counts; 2-D latlon decomposition untested on a real fabric | production-size native-NCCL multicontroller runs on Derecho/Levante (job lanes C exist) |
docs/performance/scaling/SCALING_STATUS_AUDIT.md-30-
docs/performance/scaling/SCALING_STATUS_AUDIT.md-31-## Ocean support matrix
docs/performance/scaling/SCALING_STATUS_AUDIT.md-32-
docs/performance/scaling/SCALING_STATUS_AUDIT.md-33-| grid | CPU-MPI | GPU / SPMD | true weak/strong evidence (a) | known blockers | next measurement |
docs/performance/scaling/SCALING_STATUS_AUDIT.md-34-|------|---------|------------|-------------------------------|----------------|------------------|
docs/performance/scaling/SCALING_STATUS_AUDIT.md-35-| **lat-lon C-grid** | (a) latitude-band MPI, `bench_ocean_mpi_scaling.py` (parity + conservation gates; `--wet-balance`; distributed fixed-M PCG / single_reduce / preconditioner options) | (a) full-step SPMD `bench_ocean_latlon_spmd_scaling.py` (single-process multi-device AND `--multicontroller` NCCL; parity + conservation gates); jit-once sharded step | 2-GPU full step: strong 0.92 eff at production size, weak 0.97 @ ~590k cells/rank (Ginsburg). CPU-MPI strong np16→32 eff 0.55 (implicit-CN reduction wall; split-explicit + local clamp opt-in 1.15–1.65× at ≥2 nodes) | implicit-CN allreduce wall at high ranks; land-cell load imbalance | Derecho/Levante A100 ladders (jobs exist, unrun); wide-halo split-explicit A/B at ≥16 ranks; wet-cell-balanced partitions at scale |
--
tests/bench/test_bench_cube_tiled_step_scaling.py-52-                                      "--warmup", "1"])
tests/bench/test_bench_cube_tiled_step_scaling.py-53-    with pytest.raises(SystemExit, match="need 24 devices"):
tests/bench/test_bench_cube_tiled_step_scaling.py-54-        mod.main()
tests/bench/test_bench_cube_tiled_step_scaling.py-55-
tests/bench/test_bench_cube_tiled_step_scaling.py-56-
tests/bench/test_bench_cube_tiled_step_scaling.py-57-def test_collective_census_helper():
tests/bench/test_bench_cube_tiled_step_scaling.py-58-    hlo = "\n".join([
tests/bench/test_bench_cube_tiled_step_scaling.py:59:        "%x = collective-permute(...)",
tests/bench/test_bench_cube_tiled_step_scaling.py:60:        "%y = collective-permute-start(...)",
tests/bench/test_bench_cube_tiled_step_scaling.py:61:        "%z = collective-permute-done(...)",  # not counted (done)
tests/bench/test_bench_cube_tiled_step_scaling.py-62-        "%w = add(...)",
tests/bench/test_bench_cube_tiled_step_scaling.py-63-    ])
tests/bench/test_bench_cube_tiled_step_scaling.py-64-    assert mod._count_collective_permutes(hlo) == 2
tests/bench/test_bench_cube_tiled_step_scaling.py-65-
tests/bench/test_bench_cube_tiled_step_scaling.py-66-
tests/bench/test_bench_cube_tiled_step_scaling.py-67-def test_parity_tolerances_match_adapter_gate():
tests/bench/test_bench_cube_tiled_step_scaling.py-68-    # The lane's tolerances must stay the adapter gate's f32-honest bounds
--
scripts/bench/roofline_probe.py-918-
scripts/bench/roofline_probe.py-919-    Collective census: pass the HLO counts straight from a
scripts/bench/roofline_probe.py-920-    ``run_levante_gpu_scaling.TimingResult`` (``hlo_collective_permute*``)
scripts/bench/roofline_probe.py-921-    — this reporter does NOT re-derive the census, it consumes the one the
scripts/bench/roofline_probe.py-922-    timed-executable HLO guard already produced.
scripts/bench/roofline_probe.py-923-
scripts/bench/roofline_probe.py-924-    Collective time-floor: ``collective_latency_ms`` × (number of
scripts/bench/roofline_probe.py:925:    collective ops in the step).  We count each sync collective-permute
scripts/bench/roofline_probe.py-926-    plus each async start as one round (done ops are the completion half
scripts/bench/roofline_probe.py-927-    of a start and are not double-counted).
scripts/bench/roofline_probe.py-928-
scripts/bench/roofline_probe.py-929-    PCG Amdahl: if ``pcg_iterations`` (M) is given, evaluate
scripts/bench/roofline_probe.py-930-    ``T = local_stencil + M*(2*allreduce_lat + halo + stencil)`` and report
scripts/bench/roofline_probe.py-931-    the nonlocal (communication) fraction."""
scripts/bench/roofline_probe.py-932-    total_cl = _cells_for_grid(grid_type, n_grid, n_levels)
--
scripts/bench/bench_cube_tiled_step_scaling.py-10-adapter REFUSES configs with hyperdiffusion / divergence damping / del-6 /
scripts/bench/bench_cube_tiled_step_scaling.py-11-Smagorinsky / implicit sponge / duogrid / the inner mass fixer (fail-closed
scripts/bench/bench_cube_tiled_step_scaling.py-12-envelope, ``tiled_step_adapter._refuse``).  The full production driver keeps
scripts/bench/bench_cube_tiled_step_scaling.py-13-its loud "tiled dycore unwired (P4)" warning; this lane is where >6-GPU
scripts/bench/bench_cube_tiled_step_scaling.py-14-production stepping is measured TODAY.
scripts/bench/bench_cube_tiled_step_scaling.py-15-
scripts/bench/bench_cube_tiled_step_scaling.py-16-Anti-fake-scaling guards:
scripts/bench/bench_cube_tiled_step_scaling.py:17:  * compiled-HLO census: the step must contain collective-permutes and NO
scripts/bench/bench_cube_tiled_step_scaling.py-18-    full-cube all-gather (``find_fullcube_allgathers`` — an all-gather means
scripts/bench/bench_cube_tiled_step_scaling.py-19-    replicated, not tiled, execution): the row is REFUSED otherwise;
scripts/bench/bench_cube_tiled_step_scaling.py-20-  * shared metadata v2 rows (virtual-CPU devices flagged; transport
scripts/bench/bench_cube_tiled_step_scaling.py-21-    auto-resolves) — a CPU smoke row can never masquerade as GPU scaling;
scripts/bench/bench_cube_tiled_step_scaling.py-22-  * --parity-gate: ONE tiled step vs one serial untiled step at the adapter
scripts/bench/bench_cube_tiled_step_scaling.py-23-    gate's f32-honest tolerances (the adapter is single-shot — tile-replicated
scripts/bench/bench_cube_tiled_step_scaling.py-24-    in, tile-sharded out; single-process only; smoke windows).
--
scripts/bench/bench_cube_tiled_step_scaling.py-224-    # Full per-family census (superset of the CP count): surfaces the
scripts/bench/bench_cube_tiled_step_scaling.py-225-    # conservation all-reduce and any operator-introduced resharding on the
scripts/bench/bench_cube_tiled_step_scaling.py-226-    # SAME audited executable — the message-count = latency-bound lever.
scripts/bench/bench_cube_tiled_step_scaling.py-227-    hlo_census = count_collectives(hlo)
scripts/bench/bench_cube_tiled_step_scaling.py-228-    allgathers = find_fullcube_allgathers(hlo, n=args.resolution)
scripts/bench/bench_cube_tiled_step_scaling.py-229-    if n_ppermute == 0:
scripts/bench/bench_cube_tiled_step_scaling.py-230-        raise SystemExit(
scripts/bench/bench_cube_tiled_step_scaling.py:231:            "compiled tiled step contains NO collective-permutes — the "
scripts/bench/bench_cube_tiled_step_scaling.py-232-            "halos did not tile (replicated execution); refusing to "
scripts/bench/bench_cube_tiled_step_scaling.py-233-            "record a fake scaling row.")
scripts/bench/bench_cube_tiled_step_scaling.py-234-    if allgathers:
scripts/bench/bench_cube_tiled_step_scaling.py-235-        raise SystemExit(
scripts/bench/bench_cube_tiled_step_scaling.py-236-            f"compiled tiled step contains full-cube all-gathers "
scripts/bench/bench_cube_tiled_step_scaling.py-237-            f"({allgathers[:3]}...) — replicated, not tiled, execution; "
scripts/bench/bench_cube_tiled_step_scaling.py-238-            f"refusing to record a fake scaling row.")
--
scripts/bench/bench_cube_tiled_step_scaling.py-262-
scripts/bench/bench_cube_tiled_step_scaling.py-263-    if jax.process_count() > 1:
scripts/bench/bench_cube_tiled_step_scaling.py-264-        from jax.experimental import multihost_utils
scripts/bench/bench_cube_tiled_step_scaling.py-265-
scripts/bench/bench_cube_tiled_step_scaling.py-266-        multihost_utils.sync_global_devices("cube_tiled_bench_start")
scripts/bench/bench_cube_tiled_step_scaling.py-267-
scripts/bench/bench_cube_tiled_step_scaling.py-268-    if args.closed_loop:
scripts/bench/bench_cube_tiled_step_scaling.py:269:        # #921: the closed-loop step fuses the halo collective-permutes with
scripts/bench/bench_cube_tiled_step_scaling.py-270-        # the in-stage mass-fixer psum in ONE executable; on multi-process GPU
scripts/bench/bench_cube_tiled_step_scaling.py-271-        # the NCCL comm-init of those two clique kinds can be ordered
scripts/bench/bench_cube_tiled_step_scaling.py-272-        # differently per rank and DEADLOCK.  Prime every clique in a fixed,
scripts/bench/bench_cube_tiled_step_scaling.py-273-        # rank-independent order FIRST (no-op single-process / CPU-virtual).
scripts/bench/bench_cube_tiled_step_scaling.py-274-        from legoesm.parallel.tiled_production_cdgrid import (
scripts/bench/bench_cube_tiled_step_scaling.py-275-            warmup_tiled_cube_comms,
scripts/bench/bench_cube_tiled_step_scaling.py-276-        )
--
scripts/bench/run_levante_gpu_scaling.py-324-    grid_type: str = "cubed-sphere"
scripts/bench/run_levante_gpu_scaling.py-325-    scaling_efficiency: float = 1.0
scripts/bench/run_levante_gpu_scaling.py-326-    # Collective-permute op census of the compiled TIMED executable
scripts/bench/run_levante_gpu_scaling.py-327-    # (comm-minimisation step 1: measurement infrastructure for the
scripts/bench/run_levante_gpu_scaling.py-328-    # upcoming halo-fusion work).  ``-1`` = not measured — single
scripts/bench/run_levante_gpu_scaling.py-329-    # device, MPI, non-cubed-sphere, or the HLO guard was skipped via
scripts/bench/run_levante_gpu_scaling.py-330-    # LEGOESM_SPMD_FORCE_ALLGATHER.  Sync ops lower as
scripts/bench/run_levante_gpu_scaling.py:331:    # ``collective-permute``; async pairs as ``-start``/``-done``.
scripts/bench/run_levante_gpu_scaling.py-332-    hlo_collective_permute: int = -1
scripts/bench/run_levante_gpu_scaling.py-333-    hlo_collective_permute_start: int = -1
scripts/bench/run_levante_gpu_scaling.py-334-    hlo_collective_permute_done: int = -1
scripts/bench/run_levante_gpu_scaling.py-335-
scripts/bench/run_levante_gpu_scaling.py-336-
scripts/bench/run_levante_gpu_scaling.py-337-@dataclass
scripts/bench/run_levante_gpu_scaling.py-338-class ScalingReport:
--
scripts/bench/run_levante_gpu_scaling.py-609-
scripts/bench/run_levante_gpu_scaling.py-610-
scripts/bench/run_levante_gpu_scaling.py-611-# ===========================================================================
scripts/bench/run_levante_gpu_scaling.py-612-# Timed scan runner — shared between the dry-dycore and moist-segment paths
scripts/bench/run_levante_gpu_scaling.py-613-# ===========================================================================
scripts/bench/run_levante_gpu_scaling.py-614-
scripts/bench/run_levante_gpu_scaling.py-615-def _count_collective_permute_ops(hlo_text: str) -> dict[str, int]:
scripts/bench/run_levante_gpu_scaling.py:616:    """Census of collective-permute ops in a compiled HLO module.
scripts/bench/run_levante_gpu_scaling.py-617-
scripts/bench/run_levante_gpu_scaling.py-618-    Counts opcode *applications* (``<opcode>(``) so each op is counted
scripts/bench/run_levante_gpu_scaling.py-619-    once regardless of how many times its result name appears.  Sync
scripts/bench/run_levante_gpu_scaling.py:620:    halo exchanges lower to ``collective-permute``; the async form
scripts/bench/run_levante_gpu_scaling.py:621:    lowers to ``collective-permute-start`` / ``collective-permute-done``
scripts/bench/run_levante_gpu_scaling.py-622-    pairs.  Comm-minimisation sequencing step 1: this census is the
scripts/bench/run_levante_gpu_scaling.py-623-    before/after metric for the upcoming halo-fusion work.
scripts/bench/run_levante_gpu_scaling.py-624-    """
scripts/bench/run_levante_gpu_scaling.py-625-    import re
scripts/bench/run_levante_gpu_scaling.py-626-    return {
scripts/bench/run_levante_gpu_scaling.py:627:        "collective-permute": len(
scripts/bench/run_levante_gpu_scaling.py:628:            re.findall(r"\bcollective-permute\(", hlo_text)),
scripts/bench/run_levante_gpu_scaling.py:629:        "collective-permute-start": len(
scripts/bench/run_levante_gpu_scaling.py:630:            re.findall(r"\bcollective-permute-start\(", hlo_text)),
scripts/bench/run_levante_gpu_scaling.py:631:        "collective-permute-done": len(
scripts/bench/run_levante_gpu_scaling.py:632:            re.findall(r"\bcollective-permute-done\(", hlo_text)),
scripts/bench/run_levante_gpu_scaling.py-633-    }
scripts/bench/run_levante_gpu_scaling.py-634-
scripts/bench/run_levante_gpu_scaling.py-635-
scripts/bench/run_levante_gpu_scaling.py-636-def _hlo_census_fields(hlo_counts: dict[str, int] | None) -> dict[str, int]:
scripts/bench/run_levante_gpu_scaling.py:637:    """``TimingResult`` kwargs for the collective-permute census.
scripts/bench/run_levante_gpu_scaling.py-638-
scripts/bench/run_levante_gpu_scaling.py-639-    Empty dict (→ the ``-1`` "not measured" defaults) when the HLO
scripts/bench/run_levante_gpu_scaling.py-640-    guard did not run.
scripts/bench/run_levante_gpu_scaling.py-641-    """
scripts/bench/run_levante_gpu_scaling.py-642-    if hlo_counts is None:
scripts/bench/run_levante_gpu_scaling.py-643-        return {}
scripts/bench/run_levante_gpu_scaling.py-644-    return {
scripts/bench/run_levante_gpu_scaling.py:645:        "hlo_collective_permute": hlo_counts["collective-permute"],
scripts/bench/run_levante_gpu_scaling.py:646:        "hlo_collective_permute_start": hlo_counts["collective-permute-start"],
scripts/bench/run_levante_gpu_scaling.py:647:        "hlo_collective_permute_done": hlo_counts["collective-permute-done"],
scripts/bench/run_levante_gpu_scaling.py-648-    }
scripts/bench/run_levante_gpu_scaling.py-649-
scripts/bench/run_levante_gpu_scaling.py-650-
scripts/bench/run_levante_gpu_scaling.py-651-def _build_timed_scan_runner(
scripts/bench/run_levante_gpu_scaling.py-652-    *,
scripts/bench/run_levante_gpu_scaling.py-653-    step_fn,
scripts/bench/run_levante_gpu_scaling.py-654-    state,
--
scripts/bench/run_levante_gpu_scaling.py-677-       (``create_output_shardings``) as the inner compiled step.  Only
scripts/bench/run_levante_gpu_scaling.py-678-       for single-process multi-device cubed-sphere runs; ``None``
scripts/bench/run_levante_gpu_scaling.py-679-       (plain ``jax.jit``) everywhere else — zero behavior change for
scripts/bench/run_levante_gpu_scaling.py-680-       single-GPU / MPI / non-cubed-sphere rows;
scripts/bench/run_levante_gpu_scaling.py-681-    2. sharding tripwire #1 on the post-warmup seed state;
scripts/bench/run_levante_gpu_scaling.py-682-    3. the compiled-HLO hot-path guard: zero full-cube all-gathers in
scripts/bench/run_levante_gpu_scaling.py-683-       the timed executable (LEGOESM_SPMD_FORCE_ALLGATHER=1 skips with
scripts/bench/run_levante_gpu_scaling.py:684:       a loud warning), plus the collective-permute op census (printed
scripts/bench/run_levante_gpu_scaling.py-685-       and returned for the result row metadata).  The
scripts/bench/run_levante_gpu_scaling.py-686-       ``lower().compile()`` result is reused as the timed runner, so
scripts/bench/run_levante_gpu_scaling.py-687-       the guard adds no extra compilation;
scripts/bench/run_levante_gpu_scaling.py-688-    4. precompile against leaf-cloned state (so the timed run still
scripts/bench/run_levante_gpu_scaling.py-689-       starts from the post-warmup state, and queued XLA work cannot
scripts/bench/run_levante_gpu_scaling.py-690-       overlap the timed region — block on the precompile OUTPUT);
scripts/bench/run_levante_gpu_scaling.py-691-    5. sharding tripwire #2 on the precompile output (the actual timed
--
scripts/bench/run_levante_gpu_scaling.py-799-            print(
scripts/bench/run_levante_gpu_scaling.py-800-                "    HLO guard: hot path clean — no full-cube all-gather "
scripts/bench/run_levante_gpu_scaling.py-801-                "ops in the timed executable",
scripts/bench/run_levante_gpu_scaling.py-802-                flush=True,
scripts/bench/run_levante_gpu_scaling.py-803-            )
scripts/bench/run_levante_gpu_scaling.py-804-            hlo_counts = _count_collective_permute_ops(_hlo_text)
scripts/bench/run_levante_gpu_scaling.py-805-            print(
scripts/bench/run_levante_gpu_scaling.py:806:                f"    HLO census: {hlo_counts['collective-permute']} "
scripts/bench/run_levante_gpu_scaling.py:807:                f"collective-permute, "
scripts/bench/run_levante_gpu_scaling.py:808:                f"{hlo_counts['collective-permute-start']} -start, "
scripts/bench/run_levante_gpu_scaling.py:809:                f"{hlo_counts['collective-permute-done']} -done op(s) "
scripts/bench/run_levante_gpu_scaling.py-810-                f"in the timed executable",
scripts/bench/run_levante_gpu_scaling.py-811-                flush=True,
scripts/bench/run_levante_gpu_scaling.py-812-            )
scripts/bench/run_levante_gpu_scaling.py-813-            scan_runner = _compiled_runner
scripts/bench/run_levante_gpu_scaling.py-814-
scripts/bench/run_levante_gpu_scaling.py-815-    # Pre-compile the scan runner without mutating the timed state.
scripts/bench/run_levante_gpu_scaling.py-816-    # Re-binding ``state`` to the precompile output would start the
--
scripts/bench/metadata.py-96-
scripts/bench/metadata.py-97-
scripts/bench/metadata.py-98-def _env_flag_true(name: str) -> bool:
scripts/bench/metadata.py-99-    """True iff env var ``name`` is a truthy flag ("1"/"true"/"yes"/"on")."""
scripts/bench/metadata.py-100-    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")
scripts/bench/metadata.py-101-
scripts/bench/metadata.py-102-
scripts/bench/metadata.py:103:# Match the OP-CALL form ``collective-permute(`` / ``collective_permute(`` /
scripts/bench/metadata.py-104-# ``...-start(`` (a paren directly after the op name), NOT bare substrings: the
scripts/bench/metadata.py-105-# COMPILED-HLO config header echoes XLA_FLAGS, so a flag name like
scripts/bench/metadata.py-106-# ``xla_gpu_collective_permute_combine_threshold_bytes=`` (set by #1175's
scripts/bench/metadata.py-107-# MPAS_CP_COMBINE) would false-match a plain substring scan and over-count.
scripts/bench/metadata.py-108-_COLLECTIVE_PERMUTE_RE = re.compile(r"collective[_-]permute(?:[_-]start)?\(")
scripts/bench/metadata.py-109-
scripts/bench/metadata.py-110-
--
scripts/bench/metadata.py-133-    so config-header flag names that merely CONTAIN "collective_permute" never
scripts/bench/metadata.py-134-    inflate the count.  Canonical for every bench that reports
scripts/bench/metadata.py-135-    ``hlo_collective_permutes`` (cube tiled, MPAS ico) — no re-implementation."""
scripts/bench/metadata.py-136-    return _count_op_calls(hlo_text, _COLLECTIVE_PERMUTE_RE)
scripts/bench/metadata.py-137-
scripts/bench/metadata.py-138-
scripts/bench/metadata.py-139-def hlo_collective_permutes(fn, *args) -> int | None:
scripts/bench/metadata.py:140:    """Best-effort: count the collective-permutes in the COMPILED HLO of
scripts/bench/metadata.py-141-    ``fn(*args)``.
scripts/bench/metadata.py-142-
scripts/bench/metadata.py-143-    Compiles (``.lower(...).compile().as_text()``), NOT bare
scripts/bench/metadata.py-144-    ``.lower().as_text()``: the census must reflect the EXECUTABLE's round
scripts/bench/metadata.py:145:    count, because XLA collective-permute combining / pipelined-p2p
scripts/bench/metadata.py-146-    (#1175, ``MPAS_CP_COMBINE``) fuses rounds during optimization — the whole
scripts/bench/metadata.py-147-    metric #1113 tracks. Pre-optimization StableHLO would overstate CPs versus
scripts/bench/metadata.py-148-    the timed executable. Matches the cube tiled bench, which compiles too.
scripts/bench/metadata.py-149-    Returns ``None`` (never raises) if lowering/compilation is unsupported OR
scripts/bench/metadata.py-150-    the backend's ``as_text()`` yields no HLO, so a timing probe can record
scripts/bench/metadata.py-151-    "unknown" rather than crash."""
scripts/bench/metadata.py-152-    import jax
--
scripts/bench/metadata.py-165-    ``-start``/``_start`` suffix, and requires the ``(`` op-call form so a
scripts/bench/metadata.py-166-    config-header ``XLA_FLAGS`` echo that merely CONTAINS the op name can never
scripts/bench/metadata.py-167-    inflate the count (same guard as :data:`_COLLECTIVE_PERMUTE_RE`)."""
scripts/bench/metadata.py-168-    stem = op_name.replace("-", "[_-]")
scripts/bench/metadata.py-169-    return re.compile(stem + r"(?:[_-]start)?\(")
scripts/bench/metadata.py-170-
scripts/bench/metadata.py-171-
scripts/bench/metadata.py:172:#: Every collective OP family a scaling row can run.  ``collective-permute`` is
scripts/bench/metadata.py-173-#: the band/face halo (reuse the canonical permute regex so its count stays
scripts/bench/metadata.py-174-#: bit-identical to :func:`count_collective_permutes`); ``all-reduce`` is the
scripts/bench/metadata.py-175-#: conservation fixer AND the ocean implicit-CN PCG reduction wall (~120/step —
scripts/bench/metadata.py-176-#: the #1 ocean strong-scaling bottleneck, invisible to a permute-only census);
scripts/bench/metadata.py-177-#: the rest surface any SPMD resharding an operator introduces.
scripts/bench/metadata.py-178-_COLLECTIVE_OP_RES: dict[str, "re.Pattern[str]"] = {
scripts/bench/metadata.py-179-    "collective_permute": _COLLECTIVE_PERMUTE_RE,
--
docs/performance/scaling/scaling_indicators.csv-17-2026-06-13,cabd2819,latlon2d_halo,ocean_latlon,mpi,strong,1.73,halo_2dover1d_np16,8477039,2D 4x4 59.3ms vs 1D-band 34.2ms — 2D HURTS (latency fabric); DEFER 2D step integ
docs/performance/scaling/scaling_indicators.csv-18-2026-06-13,cabd2819,latlon2d_halo,ocean_latlon,mpi,strong,1.58,halo_2dover1d_np32,8477039,2D 8x4 68.0ms vs 1D-band 43.1ms — 2D HURTS; 1D band not exhausted till np>n_lat
docs/performance/scaling/scaling_indicators.csv-19-2026-06-13,0293f865,chebyshev,ocean_latlon,mpi,precond,2.31,speedup_iters_d8,8478404,cheby deg8 35 iters to rel1e-6 vs jacobi >80 (budget-censored lower bound); coastal 48x96; cuts latency-bound global reductions >=2.3x; deg4=72iters
docs/performance/scaling/scaling_indicators.csv-20-2026-06-13,7d153934,tridiag_lapack,atm_cpu,cpu,vmix,13.1,tridiag_lapack_cpu,8478438,LAPACK gtsv vs fori-loop Thomas CPU; C96-scale 55296col x72lev 4143ms->317ms; 8.4x@20k/32 13.0x@50k/64; parity 8.9e-16; opt-in LEGOESM_TRIDIAG=lapack
docs/performance/scaling/scaling_indicators.csv-21-2026-06-14,ec1b0a36,lnps_pack,atm_cube,spmd,halo,2,collective_cut,8481289,PE ln_ps+hybrid_factor ride the {zeta,B,1/T} stage pack; -2 cross-rank collectives/RK3-substep (hybrid AMIP) on multinode-SPMD/CPU-MPI; provable-by-construction count-cut; SPMD16/16+MPI5/5 parity (bit-identical); re-audit lever
docs/performance/scaling/scaling_indicators.csv-22-2026-06-14,83a180bd,divv_pack,atm_cube,spmd,halo,3,collective_cut,8481480,div_v completes the PE stage-pack; ALL 6 cube-PE scalars {zeta,B,1/T,ln_ps,hf,div_v} in ONE collective; -3 cross-rank collectives/RK3-substep (div_damp+hybrid AMIP, non-async); SPMD16/16+MPI5/5 parity (bit-identical)
docs/performance/scaling/scaling_indicators.csv-23-2026-06-11,e209bbb0,a1_spmd,atm_cube,spmd,capability,6,tiled_np_validated,8460192,face-SPMD cube ceiling = 6 faces (1 device/face) before sub-face tiling
docs/performance/scaling/scaling_indicators.csv:24:2026-06-14,7d5c5188,tiled_np24,atm_cube,spmd,capability,24,tiled_np_validated,8482583,full fv3_sw_tendencies sub-face tiled np24=6x2x2 bit-EXACT on 2 nodes (200 cross-proc collective-permute) — np54 host-gate; np>6 future-HW (anti-scales Gloo-TCP so capability not speedup)
docs/performance/scaling/scaling_indicators.csv-25-2026-06-14,eb1fe737,tiled_3d_vort,atm_cube,spmd,capability,1,ops3d_np24,8484502,dgrid_vorticity = first fv3_hydrostatic_tendencies (3D PE) op np24-tiled 4D bit-identity (host + shard_map); grows as cc2c/geopotential/ln_ps PGF tile toward the full 3D stage
docs/performance/scaling/scaling_indicators.csv-26-2026-06-14,2fc8897c,tiled_3d_alinterp,atm_cube,spmd,capability,3,ops3d_np24,8485473,arakawa_lamb_gradient + interp_corner_to_center np24 4D stages added (now 3 shared 3D-PE cc-ops np24-tiled: +dgrid_vorticity); bit-identity host+shard_map
docs/performance/scaling/scaling_indicators.csv-27-2026-06-14,8fd4ef0d,tiled_3d_vel,atm_cube,spmd,capability,5,ops3d_np24,8485629,dgrid_to_cgrid + dgrid_to_center_vector np24 4D (both LOCAL within-face; no vector halo unlike SW) — now 5 shared 3D-PE cube ops np24-tiled
docs/performance/scaling/scaling_indicators.csv-28-2026-06-14,58475e41,tiled_3d_geo,atm_cube,spmd,capability,6,ops3d_np24,8485746,compute_geopotential (Simmons-Burridge vertical integration) np24-tiled — vertical-local per-column exact-partition; 6 shared 3D-PE ops np24-tiled
docs/performance/scaling/scaling_indicators.csv-29-2026-06-14,2c53abea,ocean_spmd_pcg,ocean_latlon,spmd,strong,0.3087,eff_2gpu_ocean_pcg_standard,8486096,FIRST ocean-GPU SPMD number — latlon 360x720 M60 f64 barotropic PCG on 2x RTX8000 PCIe (1GPU 5.06ms 2GPU 8.20ms) kernel anti-scales (psum-latency-bound — audit wall MEASURED)
docs/performance/scaling/scaling_indicators.csv-30-2026-06-14,2c53abea,ocean_spmd_pcg,ocean_latlon,spmd,strong,0.4280,eff_2gpu_ocean_pcg_single_reduce,8486096,single-reduce (1 psum/iter lever #1) cuts 2-GPU penalty to 7.59 vs 8.20ms — opt-in for the comm-bound multi-GPU regime (1-GPU slower: extra matvec)
docs/performance/scaling/scaling_indicators.csv-31-2026-06-14,246df03f,ocean_fullstep_2gpu,ocean_latlon,mpi,strong,0.659,eff_2gpu_ocean_fullstep,8486172,FULL ocean step 2xRTX8000 mpi4jax HOST-STAGED 180x360 nlev30 implicit_cn f64 (1GPU 27.0ms 2GPU 20.5ms 1.32x) — AMORTIZATION CONFIRMED barotropic-only anti-scales 0.31 to full-step 0.66 (cuda-aware MPI next lever toward 0.8-0.93)
--
docs/performance/scaling/spmd_message_census_2026-07-08.md-22-the production count). One-off probes (untracked scratch, `scripts/tmp/`);
docs/performance/scaling/spmd_message_census_2026-07-08.md-23-the method is three lines and the cube tiled bench already records it
docs/performance/scaling/spmd_message_census_2026-07-08.md-24-per-row as `hlo_collective_permutes`:
docs/performance/scaling/spmd_message_census_2026-07-08.md-25-
docs/performance/scaling/spmd_message_census_2026-07-08.md-26-```python
docs/performance/scaling/spmd_message_census_2026-07-08.md-27-# XLA_FLAGS=--xla_force_host_platform_device_count=N  JAX_PLATFORMS=cpu
docs/performance/scaling/spmd_message_census_2026-07-08.md-28-hlo = jit_step.lower(state, dt).compile().as_text()
docs/performance/scaling/spmd_message_census_2026-07-08.md:29:n_cp = hlo.count("collective-permute-start") + hlo.count("collective-permute(")
docs/performance/scaling/spmd_message_census_2026-07-08.md-30-```
docs/performance/scaling/spmd_message_census_2026-07-08.md-31-
docs/performance/scaling/spmd_message_census_2026-07-08.md-32-Counted alongside `all-gather` / `all-reduce` occurrences.

=== Collectives and flag setup in environment helper ===
     1	# Shared environment for DKRZ Levante GPU scaling jobs (sourced by the SLURM
     2	# scripts here).  SLURM/OpenMPI twin of scripts/cluster/scaling_derecho/_env.sh
     3	# (which is PBS/Cray-MPICH).
     4	# ---------------------------------------------------------------------------
     5	# EDIT the marked values (account / repo / conda env / module versions) before
     6	# the first submit.  Everything is overridable from the sbatch environment, e.g.
     7	#   sbatch --export=ALL,LEGOESM_CONDA_ENV=my-jax-env gpu_moist_scaling.slurm
     8	#
     9	# Levante GPU partition (partition `gpu`): 60 nodes, each 2x AMD EPYC 7763 +
    10	# 4x NVIDIA A100 (56 nodes 80GB, 4 nodes 40GB), InfiniBand HDR200.  MPI stack is
    11	# OpenMPI over UCX with CUDA-aware transports -- NOT Cray MPICH.
    12	# ---------------------------------------------------------------------------
    13	
    14	# --- (1) Project allocation (matches SBATCH --account in the job scripts) -----
    15	#     Levante GPU jobs bill a *_gpu sub-account (run_levante_gpu_scaling.sh uses
    16	#     bd1083_gpu, matching the SBATCH --account in the .slurm, which is the
    17	#     source of truth).  This default is only for interactive sourcing.
    18	export LEGOESM_SLURM_ACCOUNT="${LEGOESM_SLURM_ACCOUNT:-bd1083_gpu}"
    19	
    20	# --- (2) Repo location on Levante -- EDIT to where you cloned legoESM ---------
    21	REPO="${LEGOESM_REPO:-/work/bd1083/$USER/legoESM}"
    22	export REPO
    23	
    24	# --- (3) Conda env with a CUDA jaxlib AND a CUDA-aware mpi4jax (see README) ---
    25	CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
    26	
    27	# --- Federation PYTHONPATH (belt-and-braces; `pip install -e .` makes it
    28	#     redundant but harmless) ------------------------------------------------
    29	PP="$REPO/src"
    30	for p in atmosphere core coupler ice land ml ocean tools; do
    31	  PP="$PP:$REPO/packages/$p"
    32	done
    33	export PYTHONPATH="$PP:${PYTHONPATH:-}"
    34	
    35	# --- Modules + conda -- EDIT the module versions to the Levante stack you built
    36	#     mpi4py / mpi4jax against (README Step 1); pinned versions matter because
    37	#     the runtime libmpi ABI must match the build ABI ------------------------
    38	module load python3 2>/dev/null || true      # EDIT: e.g. python3/2023.01-gcc-11.2.0
    39	module load openmpi 2>/dev/null || true       # EDIT: the CUDA-aware openmpi you built against
    40	module load cuda    2>/dev/null || true       # EDIT: matching cuda toolkit
    41	if command -v conda >/dev/null 2>&1; then
    42	  conda activate "$CONDA_ENV" 2>/dev/null || true
    43	fi
    44	PY="${LEGOESM_PYTHON:-$(command -v python)}"
    45	export PY
    46	
    47	# --- JAX / runtime knobs -----------------------------------------------------
    48	export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
    49	export MPI4JAX_NO_WARN_JAX_VERSION=1
    50	export MPLBACKEND="${MPLBACKEND:-Agg}"          # headless plotting
    51	# DKRZ scratch is /scratch/<first-letter-of-user>/<user>.
    52	export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
    53	# Persistent JIT cache reuses compiles across runs; set empty to force a cold
    54	# compile (true compile_time_s).  On SCRATCH so it survives between jobs.
    55	export LEGOESM_JIT_CACHE_DIR="${LEGOESM_JIT_CACHE_DIR:-$SCRATCH/legoesm_jit_cache}"
    56	
    57	# --- OpenMPI + UCX CUDA-aware fabric (GPU route-A) ---------------------------
    58	# Route-A hands the on-device sendrecv buffer straight to MPI (the whole point:
    59	# no device->host->device staging, which would erase multi-GPU scaling).  On
    60	# Levante that path is OpenMPI-over-UCX; the pml/osc + UCX transports below turn
    61	# on GPU-direct: cuda_copy + cuda_ipc intra-node, gdr_copy over InfiniBand HDR
    62	# inter-node.  Requires a CUDA-aware mpi4jax (README) + MPI4JAX_USE_CUDA_MPI=1
    63	# (set in the job script).  UCX_MEMTYPE_CACHE=n avoids a stale device/host
    64	# memtype-cache hang that CUDA-aware sendrecv is prone to.
    65	export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
    66	export OMPI_MCA_osc="${OMPI_MCA_osc:-ucx}"
    67	export UCX_TLS="${UCX_TLS:-rc,cuda_copy,cuda_ipc,gdr_copy,sm,self}"
    68	export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
    69	export UCX_RNDV_SCHEME="${UCX_RNDV_SCHEME:-put_zcopy}"
    70	
    71	# --- NCCL over InfiniBand (route-B: jax.distributed multi-node lanes) --------
    72	# NCCL (shard_map/ppermute collectives under jax.distributed) uses its own
    73	# IB-verbs stack — independent of the UCX/MPI settings above; the two configs
    74	# coexist. Bootstrap ring runs over IPoIB: verify the interface name once with
    75	# `ip addr` on a gpu node (a wrong NCCL_SOCKET_IFNAME is the #1 cause of
    76	# multi-node NCCL bootstrap timeouts on IB clusters).
    77	export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
    78	export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
    79	# Prefix-match BOTH HCAs (mlx5_0/mlx5_1 — one per socket on Levante nodes).
    80	export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
    81	# GPUDirect RDMA when NIC and GPU share a NUMA/PCIe root.
    82	export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
    83	export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
    84	
    85	export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
    86	mkdir -p "$TMPDIR" 2>/dev/null || true

exec
/usr/bin/bash -lc "printf '%s\\n' '=== Original LL2048@128 protocol script(s) ==='
rg -l '26534060|LL2048|n-lat 2048|n-lon 4096' scripts/cluster scripts/tmp docs/performance/scaling --glob '*.sbatch' --glob '*.sh' --glob '*.md' | sort | sed -n '1,180p'
for f in "'$(rg -l '"'26534060' scripts/cluster scripts/tmp docs/performance/scaling --glob '*.sbatch' --glob '*.sh' 2>/dev/null | head -n 8); do printf '\\n=== %s ===\\n' \""'$f"; nl -ba "$f" | sed -n '"'1,180p'; done
printf '%s\\n' '=== no user XLA override facts ==='
rg -n 'XLA_FLAGS' scripts/cluster/scaling_levante | sed -n '1,300p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 466ms:
=== Original LL2048@128 protocol script(s) ===
docs/performance/scaling/levante_campaign_2026-07-24.md
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch
scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch
scripts/tmp/fig3_atm128.sbatch

=== scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch ===
     1	#!/bin/bash -l
     2	#SBATCH --job-name=atm_ll_192
     3	#SBATCH --account=bb1596_gpu
     4	#SBATCH --partition=gpu
     5	#SBATCH --constraint=a100_80
     6	#SBATCH --nodes=48
     7	#SBATCH --gpus-per-node=4
     8	#SBATCH --exclusive
     9	#SBATCH --mem=0
    10	#SBATCH --time=02:00:00
    11	#SBATCH --output=atm_ll_192.%j.log
    12	# HUNDREDS-OF-GPUS lat-lon atmosphere (user directive 2026-08-02).
    13	# LL2048 does not divide 192 ranks (2048/192 non-integer bands), so the
    14	# >128 strong pair moves to LL2304 (96 and 192 both divide 2304) and the
    15	# above-floor "hundreds" point is LL2880x5760 @192 = 86.4k cols/GPU.
    16	#   arm 1: LL2304x4608 @ 96  f32 (110.6k cols/GPU)
    17	#   arm 2: LL2304x4608 @192  f32 (55.3k cols/GPU — still ABOVE the ~30k
    18	#          floor; codex r21 caught the first draft halving these — so
    19	#          this pair tests device-count cost at healthy tiles)
    20	#   arm 3: LL2880x5760 @192  f32 (86.4k cols/GPU — the honest 192-GPU
    21	#          working point; predicted to hold ~existing GC/s levels)
    22	# Context anchors (same bench, steps 12/warmup 3): LL2048@128 f32
    23	# 5.58 ms 39.11 GC/s (job 26534060).
    24	set -uo pipefail
    25	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    26	export JAX_PLATFORMS=cuda,cpu
    27	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    28	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    29	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    30	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    31	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/atm_ll192_j${SLURM_JOB_ID}}"
    32	mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
    33	rc=0
    34	run_arm () { # nlat nlon np tag
    35	  echo "=== atm LL$1x$2 @$3 f32 ==="
    36	  JAX_ENABLE_X64=0 srun --ntasks="$3" --ntasks-per-node=4 \
    37	      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    38	    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
    39	      --multicontroller --n-devices "$3" --mode strong \
    40	      --n-lat "$1" --n-lon "$2" --nlev 26 \
    41	      --steps 12 --warmup 3 \
    42	      --out "$OUTDIR/$4.jsonl" || { echo "$4 FAILED"; rc=1; }
    43	}
    44	run_arm 2304 4608  96 LL2304_f32_np96
    45	run_arm 2304 4608 192 LL2304_f32_np192
    46	run_arm 2880 5760 192 LL2880_f32_np192
    47	echo "=== RESULTS ==="
    48	for T in LL2304_f32_np96 LL2304_f32_np192 LL2880_f32_np192; do
    49	  F="$OUTDIR/$T.jsonl"
    50	  "$PY" -c "
    51	import json,math,sys
    52	try:
    53	    d=json.loads(open('$F').readline())
    54	    ms=d['steady_median_ms']
    55	    assert math.isfinite(ms) and ms > 0
    56	except Exception as e:
    57	    print('$T: MISSING/INVALID ->', e); sys.exit(1)
    58	print(f'$T: {ms:8.2f} ms {d.get(\"mcells_per_s\",0)/1000:6.2f} GC/s')" \
    59	    || { echo "$T receipt invalid"; rc=1; }
    60	done
    61	echo "DONE rc=$rc"; exit $rc

=== scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch ===
     1	#!/bin/bash -l
     2	#SBATCH --job-name=ll128_comb
     3	#SBATCH --account=bb1596_gpu
     4	#SBATCH --partition=gpu
     5	#SBATCH --constraint=a100_80
     6	#SBATCH --nodes=32
     7	#SBATCH --gpus-per-node=4
     8	#SBATCH --exclusive
     9	#SBATCH --mem=0
    10	#SBATCH --time=01:30:00
    11	#SBATCH --output=ll128_comb.%j.log
    12	# CP-COMBINING A/B at LL2048@128 (the measured/bound ~2.9 gap).
    13	# The calibrated bound (census 41 CP + 1 AR/step, exact bytes 18.5 MB/dev,
    14	# IB 26.3us/23.5GB/s, nd=1 same-tile compute 1.659 ms) models 1.85-1.89 ms;
    15	# measured is 5.58 (job 26534060). Leading PLAUSIBLE mechanism: effective
    16	# per-CP overhead (launch+schedule+sync) >> raw fabric latency across 41
    17	# dependency-chained exchanges — the arXiv:2607.16100 regime. Lever:
    18	# COMBINE independent CPs into fewer, larger messages.
    19	# NOTE: the closed-levers null for these flags was the OCEAN lane
    20	# (reduction-dominated); this is the first atm-latlon test — not a rerun
    21	# of a closed null.
    22	# Falsifiability, BEFORE submit — arms byte-matched to 26534060 protocol:
    23	#   A control (default flags)      : expect ~5.6 ms
    24	#   B +cp-combine 8MB threshold    : CONFIRM lever if >=10% under A
    25	#   C +combine +pipelined-p2p      : scheduling interaction
    26	#   A2 control repeat  : drift bracket (codex r7)
    27	#   REFUTE if B,C within 2% of A/A2 -> refutes THIS threshold/impl only
    28	#   (no GPU post-pass census per arm); next = scheduling/overlap lever.
    29	set -uo pipefail
    30	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    31	export JAX_PLATFORMS=cuda,cpu
    32	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    33	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    34	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    35	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    36	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll128_comb_j${SLURM_JOB_ID}}"
    37	mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
    38	rc=0
    39	run_arm () { # tag extra_xla_flags
    40	  echo "=== LL2048@128 f32 arm=$1 XLA_EXTRA='$2' ==="
    41	  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
    42	      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    43	    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
    44	      --multicontroller --n-devices 128 --mode strong \
    45	      --n-lat 2048 --n-lon 4096 --nlev 26 --steps 12 --warmup 3 \
    46	      --out "$OUTDIR/$1.jsonl" || { echo "$1 FAILED"; rc=1; }
    47	}
    48	run_arm A_default ""
    49	run_arm B_combine "--xla_gpu_collective_permute_combine_threshold_bytes=8388608"
    50	run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"
    51	run_arm A2_default ""
    52	echo "=== RESULTS ==="
    53	for T in A_default B_combine C_combine_pipelined A2_default; do
    54	  "$PY" -c "
    55	import json,math,sys
    56	try:
    57	    d=json.loads(open('$OUTDIR/$T.jsonl').readline())
    58	    ms=d['steady_median_ms']; assert math.isfinite(ms) and ms>0
    59	except Exception as e:
    60	    print('$T: MISSING/INVALID ->', e); sys.exit(1)
    61	print(f'$T: {ms:8.3f} ms')" || { echo "$T receipt invalid"; rc=1; }
    62	done
    63	echo "DONE rc=$rc"; exit $rc
=== no user XLA override facts ===
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:41:  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:62:JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:66:JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:79:export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_latency_hiding_scheduler=false"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:242:              xla)   export XLA_FLAGS="${XLA_FLAGS:-} ${_XLA_COMM_FLAGS}" ;;
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:275:            [ "$ARM" = xla ] && export XLA_FLAGS="${XLA_FLAGS:-} ${_XLA_COMM_FLAGS}"

exec
/usr/bin/bash -lc "printf '%s\\n' '=== Bound-baseline job script ==='
nl -ba scripts/cluster/scaling_levante/atm_ll_bound_base.sbatch | sed -n '1,260p'
printf '%s\\n' '=== Historic fig3 atmosphere 128 script ==='
nl -ba scripts/tmp/fig3_atm128.sbatch | sed -n '1,260p'
printf '%s\\n' '=== exact xla_flag script/metadata references ==='
rg -n -i 'collective.*combine|pipelined.p2p|combine_threshold|XLA_FLAGS.*=' scripts/cluster scripts/tmp scripts/bench | sed -n '1,360p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 274ms:
=== Bound-baseline job script ===
     1	#!/bin/bash -l
     2	#SBATCH --job-name=ll_bound
     3	#SBATCH --account=bb1596_gpu
     4	#SBATCH --partition=gpu
     5	#SBATCH --constraint=a100_80
     6	#SBATCH --nodes=1
     7	#SBATCH --gpus-per-node=4
     8	#SBATCH --mem=120G
     9	#SBATCH --time=01:00:00
    10	#SBATCH --output=ll_bound.%j.log
    11	# Single-device SAME-PER-DEVICE-SIZE baselines to complete the calibrated
    12	# T_bound for the LL2048 rows (roofline recipe: the nd=1 time must be at
    13	# the PER-DEVICE tile, not the global grid — passing the global one gives
    14	# measured/bound < 1, the known tell).  Tiles: 16x4096 (the @128 band),
    15	# 32x4096 (the @64 band).  APPROXIMATION, recorded: an nd=1 run includes
    16	# the pole tiles, so its operator mix slightly OVERCOUNTS an interior
    17	# band's compute -> the bound is conservative (biased high).
    18	set -uo pipefail
    19	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    20	export JAX_PLATFORMS=cuda,cpu
    21	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    22	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    23	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    24	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    25	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/ll_bound_j${SLURM_JOB_ID}}"
    26	mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
    27	rc=0
    28	run_arm () { # nlat x64 tag
    29	  echo "=== nd=1 LL$1x4096 x64=$2 ($3) ==="
    30	  JAX_ENABLE_X64=$2 srun --ntasks=1 --gpus=1 --kill-on-bad-exit=1 \
    31	    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
    32	      --n-devices 1 --mode strong --n-lat "$1" --n-lon 4096 --nlev 26 \
    33	      --steps 12 --warmup 3 \
    34	      --out "$OUTDIR/$3.jsonl" || { echo "$3 FAILED"; rc=1; }
    35	}
    36	run_arm 16 0 tile16_f32
    37	run_arm 32 0 tile32_f32
    38	run_arm 16 1 tile16_f64
    39	echo "=== RESULTS ==="
    40	for T in tile16_f32 tile32_f32 tile16_f64; do
    41	  "$PY" -c "
    42	import json,math,sys
    43	try:
    44	    d=json.loads(open('$OUTDIR/$T.jsonl').readline())
    45	    ms=d['steady_median_ms']; assert math.isfinite(ms) and ms>0
    46	except Exception as e:
    47	    print('$T: MISSING/INVALID ->', e); sys.exit(1)
    48	print(f'$T: {ms:8.3f} ms')" || { echo "$T receipt invalid"; rc=1; }
    49	done
    50	echo "DONE rc=$rc"; exit $rc
=== Historic fig3 atmosphere 128 script ===
     1	#!/bin/bash -l
     2	#SBATCH --job-name=atm128
     3	#SBATCH --partition=gpu
     4	#SBATCH --constraint=a100_80
     5	#SBATCH --nodes=32
     6	#SBATCH --gpus-per-node=4
     7	#SBATCH --exclusive
     8	#SBATCH --mem=0
     9	#SBATCH --time=06:00:00
    10	#SBATCH --output=atm128.%j.log
    11	# Figure v3: atmosphere lat-lon at 128 GPUs (the machine's practical max
    12	# for one job: 32 of 63 nodes), both precisions, LL2048x4096 L26 =
    13	# 65.5k cols/GPU (above the ~30k floor -> should still scale).
    14	set -uo pipefail
    15	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    16	export JAX_PLATFORMS=cuda,cpu
    17	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    18	cd "$REPO"
    19	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/atm128_j${SLURM_JOB_ID}}"
    20	mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
    21	rc=0
    22	for PREC in 0 1; do
    23	  P=f32; [ "$PREC" = 1 ] && P=f64
    24	  echo "=== atm LL2048x4096 @128 $P ==="
    25	  JAX_ENABLE_X64=$PREC srun --ntasks=128 --ntasks-per-node=4 \
    26	      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    27	    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
    28	      --multicontroller --n-devices 128 --mode strong \
    29	      --n-lat 2048 --n-lon 4096 --nlev 26 \
    30	      --steps 12 --warmup 3 \
    31	      --out "$OUTDIR/LL2048_${P}_np128.jsonl" || { echo "$P FAILED"; rc=1; }
    32	done
    33	for F in "$OUTDIR"/*.jsonl; do
    34	  "$PY" -c "
    35	import json,os; d=json.loads(open('$F').readline())
    36	print(f\"{os.path.basename('$F'):24s} {d['steady_median_ms']:8.2f} ms {d.get('mcells_per_s',0)/1000:.1f} GC/s\")" 2>/dev/null
    37	done
    38	echo "DONE rc=$rc"; exit $rc
=== exact xla_flag script/metadata references ===
scripts/bench/bench_spectral_les_dd_scaling.py:19:           XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" OMP_NUM_THREADS=1
scripts/bench/probe_spectral_shard.py:7:    JAX_ENABLE_X64=1 XLA_FLAGS=--xla_force_host_platform_device_count=N \\
scripts/bench/run_cpu_mpi_scaling.py:106:        xla_flags = os.environ.get("XLA_FLAGS", "")
scripts/bench/run_cpu_mpi_scaling.py:108:            xla_flags = f"{xla_flags} --xla_cpu_multi_thread_eigen=false".strip()
scripts/bench/run_cpu_mpi_scaling.py:109:        os.environ["XLA_FLAGS"] = xla_flags
scripts/bench/bench_mpas_spmd_scaling.py:48:  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
scripts/bench/bench_ocean_gpu_scaling.py:49:    os.environ["XLA_FLAGS"] = f"{existing} {_CUDA_GRAPH_FLAG}".strip()
scripts/bench/run_levante_gpu_scaling.py:140:        xla_flags = os.environ.get("XLA_FLAGS", "")
scripts/bench/run_levante_gpu_scaling.py:145:                xla_flags = f"{xla_flags} {flag}" if xla_flags else flag
scripts/bench/run_levante_gpu_scaling.py:146:        os.environ["XLA_FLAGS"] = xla_flags
scripts/bench/metadata.py:106:# ``xla_gpu_collective_permute_combine_threshold_bytes=`` (set by #1175's
scripts/bench/metadata.py:145:    count, because XLA collective-permute combining / pipelined-p2p
scripts/bench/metadata.py:208:    Superset of :func:`hlo_collective_permutes` — compiles (so combined /
scripts/bench/metadata.py:216:    combining / pipelined-p2p (``--xla_gpu_collective_permute_combine_*``, lane
scripts/bench/metadata.py:307:    ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` (N>1) on a CPU
scripts/bench/run_scaling_diagnosis.py:126:        xla_flags = os.environ.get("XLA_FLAGS", "")
scripts/bench/run_scaling_diagnosis.py:131:                xla_flags = f"{xla_flags} {flag}" if xla_flags else flag
scripts/bench/run_scaling_diagnosis.py:132:        os.environ["XLA_FLAGS"] = xla_flags
scripts/bench/run_scaling_diagnosis.py:490:    backend-independent — GPU-only XLA collective combining / pipelined-p2p
scripts/bench/run_scaling_diagnosis.py:756:                          "with XLA_FLAGS=--xla_force_host_platform_device_"
scripts/cluster/land_carbon/build_global_carbon_ic.sbatch:32:export XLA_FLAGS="--xla_force_host_platform_device_count=1"
scripts/bench/profile_mpas_ocean.py:67:    os.environ["XLA_FLAGS"] = f"{existing} {_CUDA_GRAPH_FLAG}".strip()
scripts/bench/profile_mpas_ocean.py:144:    print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")
scripts/cluster/land_carbon/run_equilibrium.sbatch:27:export XLA_FLAGS="--xla_force_host_platform_device_count=1"
scripts/bench/bench_cube_tiled_step_scaling.py:31:    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
scripts/cluster/land_carbon/validate_global_carbon_ic.sbatch:27:export XLA_FLAGS="--xla_force_host_platform_device_count=1"
scripts/cluster/wb_forecast/run_dp_test.sbatch:16:export XLA_FLAGS="--xla_force_host_platform_device_count=2"
scripts/bench/bench_cube_shardmap_halo.py:42:    XLA_FLAGS=--xla_force_host_platform_device_count=6 \
scripts/bench/bench_cube_shardmap_halo.py:485:            "XLA_FLAGS=--xla_force_host_platform_device_count>=2 to exercise it.")
scripts/bench/bench_cube_shardmap_halo.py:508:            "XLA_FLAGS=--xla_force_host_platform_device_count / launch multi-GPU.")
scripts/bench/slurm_scaling_diagnosis.sh:107:export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
scripts/bench/bench_ocean_latlon_spmd_pcg.py:35:    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_ENABLE_X64=1 \\
scripts/cluster/omip_nemo/run_multiprocess_cpu_equiv.sbatch:54:  env JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
scripts/cluster/omip_nemo/run_multiprocess_cpu_equiv.sbatch:61:    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_ENABLE_X64=1 \
scripts/cluster/scaling_levante/README.md:110:(`--xla_gpu_collective_permute_combine_threshold_bytes=32MiB` +
scripts/cluster/scaling_levante/README.md:111:`--xla_gpu_enable_pipelined_p2p=true`), `pgle`
scripts/cluster/scaling_derecho/README.md:535:2. `xla` — CP-combine 32 MiB + pipelined p2p.  **MEASURED: −10 % — not
scripts/cluster/scaling_derecho/cube_tiled_step.pbs:59:JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
scripts/cluster/scaling_derecho/cube_tiled_step.pbs:63:JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
scripts/cluster/scaling_derecho/cube_tiled_step.pbs:92:export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_latency_hiding_scheduler=false"
scripts/bench/bench_ocean_latlon_spmd_scaling.py:35:  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:25:#                         XLA CP-combining + pipelined p2p / PGLE); outputs
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:229:# pipelined p2p), pgle (profile-guided latency estimation; recompiles after
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:236:    _XLA_COMM_FLAGS="--xla_gpu_collective_permute_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:242:              xla)   export XLA_FLAGS="${XLA_FLAGS:-} ${_XLA_COMM_FLAGS}" ;;
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:275:            [ "$ARM" = xla ] && export XLA_FLAGS="${XLA_FLAGS:-} ${_XLA_COMM_FLAGS}"
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:62:JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:66:JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:79:export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_latency_hiding_scheduler=false"
scripts/cluster/scaling_derecho/scaling_cpu.sh:90:    export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false"
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:41:#     XLA collective-permute combining + pipelined p2p / PGLE) on the latlon
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:89:# XLA collective-permute combining + pipelined p2p. #1113 found the route-B MPAS
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:97:_XLA_COMM_FLAGS="--xla_gpu_collective_permute_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:220:        # CP-combining + pipelined p2p ON by default (#1113): the ppermute
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:223:            export XLA_FLAGS="${XLA_FLAGS:-} ${_XLA_COMM_FLAGS}"
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:289:#   xla   : collective-permute combining + pipelined p2p — the cube/latlon
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:309:              xla)   export XLA_FLAGS="${XLA_FLAGS:-} ${_XLA_COMM_FLAGS}" ;;
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:329:            [ "$ARM" = xla ] && export XLA_FLAGS="${XLA_FLAGS:-} ${_XLA_COMM_FLAGS}"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:25:#   C +combine +pipelined-p2p      : scheduling interaction
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:41:  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:49:run_arm B_combine "--xla_gpu_collective_permute_combine_threshold_bytes=8388608"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:50:run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"

codex
Most r7 dispositions are correctly reflected: corrected bound arithmetic, mixed-input note, nd=128/f64 census evidence, X64 mode, per-op records/assertion, and cautious model/lever wording.

Issues to fix:

- The document still cites job **26630438**, not the resubmission **26630576**, at [levante_campaign_2026-07-24.md](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/docs/performance/scaling/levante_campaign_2026-07-24.md:1843).
- That same sentence calls it a “3-arm” test, while the script executes four runs: A, B, C, A2 ([sbatch](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:48)).
- The probe’s “compile ONCE” comment is inaccurate: `hlo_collective_census()` compiles once, then the probe compiles the same function again for payload parsing ([probe](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/tmp/_probe_latlon_halo_census.py:58), [probe](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/tmp/_probe_latlon_halo_census.py:67)). The count assertion therefore compares two artifacts, rather than parsing and censusing one compiled HLO text.

VERDICT: FIX-FIRST
tokens used
83,047
Most r7 dispositions are correctly reflected: corrected bound arithmetic, mixed-input note, nd=128/f64 census evidence, X64 mode, per-op records/assertion, and cautious model/lever wording.

Issues to fix:

- The document still cites job **26630438**, not the resubmission **26630576**, at [levante_campaign_2026-07-24.md](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/docs/performance/scaling/levante_campaign_2026-07-24.md:1843).
- That same sentence calls it a “3-arm” test, while the script executes four runs: A, B, C, A2 ([sbatch](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:48)).
- The probe’s “compile ONCE” comment is inaccurate: `hlo_collective_census()` compiles once, then the probe compiles the same function again for payload parsing ([probe](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/tmp/_probe_latlon_halo_census.py:58), [probe](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/tmp/_probe_latlon_halo_census.py:67)). The count assertion therefore compares two artifacts, rather than parsing and censusing one compiled HLO text.

VERDICT: FIX-FIRST
