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
session id: 019fc669-02ea-7be0-87ca-2da07c9e18c3
--------
user
Round-20: review the merge-port (git show e13de3afe --stat; git diff e13de3afe^1 e13de3afe -- packages/core/legoesm/parallel/geometry_consistency.py packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py). Context: origin/main gained #1362 (shared geometry_consistency module + entry gates; its broadcast_checked ends in broadcast_one_to_all and its ocean _replicated_put re-adds device_put — centralizing walls 1+2). The port: (a) NEW shared fns in geometry_consistency: band_fingerprint (per-band, ValueError on wrong leading axis, schema-gate prerequisite documented), band_fingerprints_agree (rtol defaults _FLOAT_RTOL), checked_shard_put (per-band gate + make_array_from_callback; single-process plain device_put), assert_pytree_bytes_equal (per-leaf 48-bit digests), addressable_shard_put (ungated, caller-gated inputs); wall history comment block. (b) ocean: _replicated_put now calls checked_shard_put(n_bands=n_dev) AFTER their assert_schema_agrees(arrays=...) gate; 5 state/forcing puts -> addressable_shard_put with assert_pytree_bytes_equal gates; aux threading on sharded_step + step.aux; _global shards forcing explicitly; their entry gates + schema-arrays gate kept verbatim. (c) my unit test retargeted to the shared module (ValueError test). Receipts: 181 gate/parity + 48 run_omip tests + 2-proc multicontroller repro (15.39 ms) green post-merge.
Attack: (1) does anything still call broadcast_checked with a BAND STACK (grep callers — atm lane #1362 sites!) — if the atm lane broadcasts global-sized stacks it carries wall 1 at high nd; is that a this-PR fix or a flagged follow-up? (2) checked_shard_put's n_bands prerequisite — is the ocean call ordering (schema gate before puts) actually enforced in code order? (3) duplicate/conflicting fingerprint machinery left over (their fixed-width whole-array vs my per-band — both used where?); (4) import cycles (core/parallel importing jax fine; ocean importing core ok per layering?); (5) anything from their entry gates broken by my aux param (\_agree_ocean_spmd_call signature vs new aux arg?). End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + numbered issues.
codex
I’ll inspect the merge commit and the relevant call sites/tests, including the atmospheric lane’s `broadcast_checked` usage.
exec
/usr/bin/bash -lc 'git show e13de3afe --stat && git diff e13de3afe''^1 e13de3afe -- packages/core/legoesm/parallel/geometry_consistency.py packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py && printf '"'\\n-- callers --\\n' && rg -n -C 3 'broadcast_checked\\(' packages tests && printf '\\n-- fingerprints --\\n' && rg -n -C 2 'fingerprint|checked_shard_put|addressable_shard_put|assert_pytree_bytes_equal|assert_schema_agrees|_agree_ocean_spmd_call|sharded_step\\(|\\.aux' packages/core packages/ocean packages/atmosphere tests/ocean/unit/test_sharded_geom_fingerprint.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 3989ms:
commit e13de3afe12891150344dbc2a4d873863e2deea0
Merge: 1b71f5dc3 d5fcc93f8
Author: Pierre Gentine <b381103@levante2.lvt.dkrz.de>
Date:   Mon Aug 3 08:56:08 2026 +0200

    Merge remote-tracking branch 'origin/main' into worktree-scaling-campaign
    
    # Conflicts:
    #       packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py

 .claude/ralph_dino_fidelity_to_bar_task.md         |  321 +++
 .claude/ralph_nemo_term_correlation_task.md        |  297 ++-
 .claude/skills/oracle-fidelity/SKILL.md            |  157 +-
 .github/workflows/ci.yml                           |   25 +-
 .ralph/fix_plan.md                                 |  641 ++++-
 .ralphrc                                           |   26 +-
 CLAUDE.md                                          |  192 +-
 config/3DVar_single/README.md                      |   32 +
 config/3DVar_single/experiment.yaml                |   37 +
 config/4DVar_single/README.md                      |    8 +-
 config/4DVar_single/experiment.yaml                |    2 +-
 config/4DVar_single/nmc.yaml                       |   13 +-
 config/amip/amip_production_latlon24.yaml          |   41 +-
 .../cube_faithfulness_audit_2026-07-31.md          |  264 +++
 .../scm_rce_convection_intercomparison.md          |  440 ++++
 docs/dev-notes/issues/issue_480_wall_grid_mode.md  |    2 +-
 docs/ocean/fidelity/dino_1226_state.md             |  176 ++
 docs/ocean/fidelity/dino_1226_status_board.md      |  168 ++
 .../fidelity/dino_config_reachability_audit.md     |   97 +
 .../fidelity/dino_outstanding_fidelity_debt.md     |  186 ++
 docs/ocean/fidelity/dino_step_chain_coverage.md    |  188 ++
 docs/ocean/fidelity/dino_wiring_diagram.md         |   24 +-
 docs/production_reorg.md                           |   21 +-
 .../atmosphere/dynamics/crm/moist_mass_fixer.py    |    5 +-
 .../dynamics/gcm/compressible_euler_cdgrid.py      |   74 +-
 .../atmosphere/dynamics/gcm/dcmip2025_ic.py        |   17 +
 .../atmosphere/dynamics/gcm/primitive_eq_cdgrid.py |   26 +-
 .../atmosphere/dynamics/gcm/primitive_eq_mpas.py   |   15 +-
 .../dynamics/gcm/sharded_atm_latlon_step.py        |  422 +++-
 .../legoesm/atmosphere/dynamics/gcm/spectral_nh.py |   29 +-
 .../legoesm/atmosphere/forcing/scm/sam_case_scm.py |  547 +++++
 .../legoesm/atmosphere/forcing/scm/scm.py          |    9 +-
 .../atmosphere/physics/clouds/cloud_fraction.py    |  206 +-
 .../legoesm/atmosphere/physics/clouds/config.py    |   30 +
 .../atmosphere/physics/clouds/subcolumns.py        |  256 ++
 .../atmosphere/physics/convection/bechtold.py      |   21 +-
 .../atmosphere/physics/convection/config.py        |   98 +-
 .../atmosphere/physics/convection/emanuel.py       |   16 +
 .../atmosphere/physics/convection/kain_fritsch.py  |   16 +-
 .../atmosphere/physics/convection/mass_flux.py     |   80 +-
 .../atmosphere/physics/convection/tiedtke.py       |   16 +-
 .../physics/convection/zhang_mcfarlane.py          |    9 +
 .../atmosphere/physics/gravity_wave_drag/config.py |   19 +
 .../atmosphere/physics/gravity_wave_drag/hines.py  |   64 +-
 .../atmosphere/physics/ml_parameterization.py      |    1 +
 .../atmosphere/physics/radiation/integration.py    |   42 +
 .../radiation/rrtmgp/config/radiative_transfer.py  |    8 +-
 .../core/legoesm/core/_fv3_divergence_corner.py    |  118 +-
 packages/core/legoesm/core/bulk_flux.py            |   42 +
 packages/core/legoesm/core/conservation.py         |   21 +
 packages/core/legoesm/core/fv3_sw_core.py          |   51 +-
 packages/core/legoesm/core/fv_tp_2d.py             |    4 +-
 packages/core/legoesm/core/param_overrides.py      |   40 +
 packages/core/legoesm/grids/cubed_sphere.py        |   22 +-
 packages/core/legoesm/grids/dgrid_halo.py          |  323 +++
 packages/core/legoesm/grids/halo.py                |  106 +-
 packages/core/legoesm/grids/halo_latlon.py         |    2 +-
 packages/core/legoesm/grids/latlon.py              |  191 +-
 packages/core/legoesm/grids/tripole.py             |   10 +
 packages/core/legoesm/grids/vertical.py            |   93 +
 packages/core/legoesm/grids/voronoi.py             |   38 +-
 packages/core/legoesm/io/cmor_output.py            |   46 +
 .../core/legoesm/parallel/geometry_consistency.py  |  733 ++++++
 packages/core/legoesm/parallel/latlon_mpi.py       |    2 +
 packages/core/legoesm/parallel/sharded_dynamics.py |   36 +-
 packages/core/legoesm/parallel/voronoi_mpi.py      |    4 +-
 .../core/legoesm/parallel/voronoi_partition.py     |   17 +
 packages/coupler/legoesm/driver/config.py          |   48 +
 packages/coupler/legoesm/driver/diagnostics.py     |  247 +-
 packages/coupler/legoesm/driver/model_driver.py    |   70 +-
 .../coupler/legoesm/driver/physics_pipeline.py     |   20 +-
 packages/coupler/legoesm/driver/run_config_yaml.py |    9 +-
 .../legoesm/driver/sharded_operator_split_step.py  |  210 +-
 packages/ice/legoesm/ice/sea_ice.py                |   14 +-
 .../cime_src_share_util/shr_orb_mod.py             |   70 +-
 .../clm_src_biogeophys/SurfaceAlbedoMod.py         |    6 +-
 .../multilayer_canopy/MLCanopyTurbulenceMod.py     |   15 +-
 .../multilayer_canopy/MLclm_varcon.py              |    2 +-
 .../clm_ml_backend/offline_driver/TowerDataMod.py  |   57 +-
 packages/land/legoesm/land/carbon/fast_analytic.py |    2 +-
 packages/land/legoesm/land/carbon/global_init.py   |   17 +-
 packages/ml/legoesm/da/gen_be.py                   |  470 ++--
 packages/ml/legoesm/ml/loss.py                     |   14 +-
 packages/ml/legoesm/ml/s2s/sfno_slab/evaluation.py |    8 +-
 packages/ml/legoesm/training/__init__.py           |    2 +-
 packages/ml/legoesm/training/aimip_params.py       |    8 +-
 packages/ml/legoesm/training/data_parallel.py      |  146 +-
 packages/ml/legoesm/training/feedback.py           |    4 +-
 packages/ml/legoesm/training/les_reference.py      |  318 +++
 packages/ml/legoesm/training/param_collector.py    |   25 +-
 packages/ml/legoesm/training/parameter_field.py    |    2 +-
 packages/ml/legoesm/training/scm_rce_metrics.py    |   36 +-
 packages/ml/legoesm/training/trainable_params.py   |    9 +-
 packages/ocean/legoesm/ocean/advection.py          |  102 +-
 packages/ocean/legoesm/ocean/constants_config.py   |   26 +-
 .../ocean/dynamics/barotropic_latlon_cgrid.py      |  321 ++-
 .../ocean/dynamics/latlon_cgrid_operators.py       |   72 +-
 .../ocean/dynamics/ocean_model_latlon_cgrid.py     |  230 +-
 .../ocean/dynamics/ocean_pe_latlon_cgrid.py        |  284 ++-
 .../legoesm/ocean/dynamics/sharded_ocean_step.py   |  540 +++--
 packages/ocean/legoesm/ocean/eos.py                |  125 +-
 packages/ocean/legoesm/ocean/experiments/dino.py   |  337 ++-
 .../legoesm/ocean/fidelity/box_heat_budget.py      |  304 ++-
 packages/ocean/legoesm/ocean/fidelity/nemo_io.py   |   33 +-
 .../legoesm/ocean/fidelity/nemo_state_bridge.py    |   91 +
 .../ocean/legoesm/ocean/fidelity/precision_gate.py |  148 ++
 .../ocean/legoesm/ocean/fidelity/tendency_probe.py |   20 +-
 .../ocean/legoesm/ocean/fidelity/time_levels.py    |  291 +++
 packages/ocean/legoesm/ocean/forcing/qflux.py      |   12 +-
 packages/ocean/legoesm/ocean/freshwater.py         |  164 ++
 .../ocean/physics/convection/enhanced_diffusion.py |   19 +-
 .../ocean/physics/convection/integration.py        |    1 +
 .../physics/lateral_mixing/_gm_redi_common.py      |   35 +-
 .../legoesm/ocean/physics/lateral_mixing/config.py |   17 +
 .../physics/lateral_mixing/gm_redi_latlon_cgrid.py |  727 +++++-
 .../ocean/legoesm/ocean/physics/mpas_physics.py    |    2 +
 .../legoesm/ocean/physics/shortwave_penetration.py |   53 +-
 .../ocean/physics/vertical_mixing/_shared.py       |  161 ++
 .../ocean/physics/vertical_mixing/config.py        |   60 +-
 .../ocean/physics/vertical_mixing/k_profiles.py    |   90 +-
 .../legoesm/ocean/physics/vertical_mixing/tke.py   |  288 ++-
 packages/ocean/legoesm/ocean/state.py              |  121 +-
 packages/ocean/legoesm/ocean/vertical.py           |  170 ++
 .../tools/legoesm/diagnostics/column_integrals.py  |   35 +-
 .../tools/legoesm/diagnostics/energy_budget.py     |   77 +-
 pyproject.toml                                     |   30 +-
 scripts/README.md                                  |    4 +-
 scripts/bench/bench_atm_latlon_spmd_scaling.py     |   51 +-
 scripts/bench/bench_cube_tiled_step_scaling.py     |   27 +
 scripts/bench/bench_ocean_latlon_spmd_scaling.py   |   62 +-
 scripts/bench/bench_ocean_mpi_scaling.py           |    1 +
 scripts/cluster/les_scm/codex_review.sbatch        |   88 +
 scripts/cluster/les_scm/pipeline_smoke.sbatch      |   56 +
 scripts/cluster/les_scm/production_les.sbatch      |  125 +
 scripts/cluster/les_scm/smoke_all_cases.sbatch     |  120 +
 scripts/cluster/les_scm/test_bridge.sbatch         |   51 +
 scripts/cluster/les_scm/tune_turbulence.sbatch     |   67 +
 scripts/cluster/levante/submit_fullphys.sh         |   71 +
 scripts/cluster/levante/submit_gwd_ab.sh           |   73 +
 scripts/cluster/omip_nemo/_ab_ice_melt70.sbatch    |   39 +
 scripts/cluster/omip_nemo/_oom_control_d7.sbatch   |   32 +
 scripts/cluster/omip_nemo/_oom_try_frac75.sbatch   |   32 +
 scripts/cluster/omip_nemo/_oom_try_prealloc.sbatch |   32 +
 scripts/cluster/scaling_derecho/_env.sh            |    8 +
 scripts/cluster/scaling_levante/_env.sh            |    6 +
 .../cluster/scm_rce_paper/arm_c_gradient.sbatch    |   64 +
 scripts/cluster/scm_rce_paper/arms_ab.sbatch       |   58 +
 .../scm_rce_paper/arms_ab_substep_control.sbatch   |   59 +
 .../scm_rce_paper/codex_review_focused.sbatch      |   67 +
 .../scm_rce_paper/codex_review_phase2.sbatch       |   90 +
 scripts/cluster/scm_rce_paper/crm_reference.sbatch |   84 +
 .../scm_rce_paper/isolate_nonhydro_failure.sbatch  |   20 +
 scripts/cluster/scm_rce_paper/merge_ab.sbatch      |   35 +
 scripts/cluster/scm_rce_paper/test_drivers.sbatch  |   22 +
 scripts/cluster/scm_rce_paper/test_phase2.sbatch   |   37 +
 .../cluster/scm_rce_paper/test_phase2_full.sbatch  |   34 +
 .../scm_rce_paper/test_regression_split.sbatch     |   29 +
 .../scm_rce_paper/validator_baseline.sbatch        |   23 +
 scripts/data/fit_mpas_gen_be.py                    |  173 +-
 scripts/matrix/ocean_test_matrix/postprocessing.py |    6 +-
 scripts/matrix/run_atmosphere_test_matrix.py       |   98 +-
 scripts/matrix/run_ocean_test_matrix.py            |    6 +-
 .../plot/plot_mpas_temperature_single_obs_3dvar.py |  322 +++
 scripts/plot/plot_scm_les_turbulence.py            |  100 +
 scripts/reorg/reorg_dynamics.py                    |  339 ---
 scripts/reorg/reorg_forcing.py                     |  334 ---
 scripts/run/les_record.py                          |   42 +-
 scripts/run/mpas_3dvar_single/__init__.py          |    1 +
 scripts/run/mpas_3dvar_single/run_assimilation.py  |  583 +++++
 scripts/run/run_amip.py                            |   63 +
 scripts/run/run_bomex_les.py                       |   22 +-
 scripts/run/run_dino.py                            |    5 +-
 scripts/run/run_dycoms_les.py                      |   11 +-
 scripts/run/run_omip.py                            |    7 +-
 scripts/run/run_omip_core2.py                      |   14 +-
 scripts/run/run_rico_les.py                        |   11 +-
 scripts/run/run_scm_les_turbulence_tuning.py       | 1189 ++++++++++
 scripts/run/run_scm_rce_campaign.py                |  153 +-
 .../run/run_scm_rce_convection_intercomparison.py  |  149 +-
 scripts/run/run_spectral_cbl.py                    |   10 +-
 scripts/run/run_spectral_les.py                    |   10 +-
 scripts/run/run_spectral_sbl.py                    |   10 +-
 scripts/run/train_carbon_params.py                 |   12 +-
 scripts/run/train_scm_rce_params.py                |  193 +-
 .../dino_1226/acc_acceptance_znmax.py              |   74 +
 .../dino_1226/acc_momentum_budget.py               |  991 ++++++++
 .../ocean_fidelity/dino_1226/acc_thermal_wind.py   |  440 ++++
 .../ocean_fidelity/dino_1226/atf_filter_walk.py    |  426 ++++
 .../ocean_fidelity/dino_1226/bn2_alpha_compare.py  |  216 ++
 .../dino_1226/bolus_nondivergence_ladder_ab.py     |  371 +++
 .../dino_1226/cancelling_rows_per_element.py       |  653 ++++++
 .../dino_1226/coverage_rows_measure.py             |  985 ++++++++
 .../dino_1226/deep_box_heat_budget.py              |  441 ++++
 .../ocean_fidelity/dino_1226/dyn_zad_ldf_walk.py   |  721 ++++++
 .../ocean_fidelity/dino_1226/eiv_transport_walk.py |  403 ++++
 .../dino_1226/eos_rab_bn2_per_element.py           |  630 +++++
 .../ocean_fidelity/dino_1226/fidelity_bar_gate.py  | 2471 ++++++++++++++++++++
 .../dino_1226/hpg_tendency_compare.py              |  437 ++++
 .../dino_1226/ldf_eiv_aeiu_per_element.py          |  350 +++
 .../dino_1226/ldf_slp_per_element.py               | 1500 ++++++++++++
 .../dino_1226/ldftra_ahtv_compare.py               |  113 +
 .../dino_1226/momentum_jacobian_probe.py           |  582 +++++
 .../dino_1226/n2_trigger_reconcile.py              |  703 ++++++
 .../validate/ocean_fidelity/dino_1226/run_fp64.py  |   15 +
 .../validate/ocean_fidelity/dino_1226/sh2_walk.py  |  624 +++++
 .../dino_1226/southern_vmix_profile.py             |  622 +++++
 .../ocean_fidelity/dino_1226/spg_substep_chain.py  | 1222 ++++++++++
 .../dino_1226/stpmlf_call_coverage.py              |  546 +++++
 .../dino_1226/surface_flux_divisor_probe.py        |  384 +++
 .../dino_1226/tra_sbc_tem_piece_decompose.py       |  318 +++
 .../ocean_fidelity/dino_1226/traadv_fct_probe.py   |  335 +++
 .../dino_1226/tracer_tendency_compare.py           |   34 +-
 .../dino_1226/unit_harness/binary_io.py            |   77 +
 .../unit_harness/run_dom_qco_r3c_probe.py          |  152 ++
 .../dino_1226/unit_harness/run_dyn_zad_probe.py    |  188 ++
 .../dino_1226/unit_harness/run_dynldf_lap_probe.py |  273 +++
 .../run_dynldf_lap_vertexmask_reconcile.py         |  411 ++++
 .../dino_1226/unit_harness/run_eos_rab_probe.py    |  192 ++
 .../dino_1226/unit_harness/run_zdf_sh2_probe.py    |  232 ++
 .../dino_1226/v_unification_timelevel_retest.py    |  502 ++++
 .../dino_1226/ww_inheritance_walk.py               |  587 +++++
 .../dino_1226/zad_gate_corr_ratio_1226.py          |  119 +
 .../dino_1226/zad_level29_onset_walk.py            |  572 +++++
 .../dino_1226/zad_recurrence_walk.py               |  573 +++++
 .../dino_1226/zad_vertical_metric_walk.py          |  477 ++++
 .../dino_1226/zdf_mxl_nmln_compare.py              |  354 +++
 .../dino_1226/zdftke_avm_offset_scan.py            |  218 ++
 .../ocean_fidelity/dino_1226/zdftke_chain_walk.py  |  560 +++++
 .../dino_1226/zdftke_composite_sh2_inheritance.py  |  409 ++++
 .../dino_1226/zu_frc_budget_completion.py          |  605 +++++
 .../dino_1226/zu_frc_leapfrog_residual_probe.py    |  462 ++++
 .../zu_frc_momentum_row_reconstruction.py          |  472 ++++
 .../ocean_fidelity/dino_1226/zu_frc_term_walk.py   |  519 ++++
 .../dino_1226/zu_frc_u_structure_probe.py          |  552 +++++
 .../dino_1226/zu_frc_write_ledger.py               |  516 ++++
 scripts/validate/sweep_ocean_tests.py              |  148 ++
 scripts/validate/validate_convection_physics.py    |  190 +-
 scripts/validate/validate_federation_packaging.py  |    9 +-
 src/legoesm/scaling_preflight.py                   |  194 ++
 .../test_run_scm_les_turbulence_tuning.py          |  559 +++++
 .../atmosphere/hydrostatic/unit/test_convection.py |    2 +-
 .../hydrostatic/unit/test_edmf_convection_824.py   |   23 +-
 .../hydrostatic/unit/test_sam_case_scm.py          |  373 +++
 .../test_cases/dcmip2025/test_case_2.py            |    4 +-
 .../test_cases/dcmip2025/test_case_2_mpas.py       |    6 +-
 .../test_cases/dcmip2025/test_case_3.py            |    4 +-
 .../test_cases/dcmip2025/test_case_3_mpas.py       |    8 +-
 .../unit/test_dcmip2025_rotation_consistency.py    |  204 ++
 .../unit/test_ext_vector_dgrid_halo.py             |    4 +-
 tests/da/test_gen_be.py                            |  169 +-
 tests/da/test_mpas_temperature_single_obs_3dvar.py |   26 +
 tests/distributed/test_data_parallel_mpi.py        |   11 +-
 tests/distributed/test_geometry_consistency_mp.py  |  145 ++
 tests/grids/test_dgrid_metric_pair_halo.py         |   85 +
 tests/grids/test_dgrid_sg_slot_halo.py             |  324 +++
 tests/grids/test_mercator.py                       |  195 ++
 tests/land/unit/test_d13c_forward.py               |    2 +-
 tests/land/unit/test_live_pool_forward.py          |    3 +-
 tests/land/unit/test_sif_forward.py                |    2 +-
 tests/land/unit/test_train_carbon_params.py        |    4 +-
 tests/legoesm_paths.py                             |   46 +-
 tests/ml/test_data_parallel.py                     |  218 ++
 .../fixtures/baroclinic_decomposition_golden.npz   |  Bin 174064 -> 173864 bytes
 tests/ocean/unit/test_advection_fct_zalesak.py     |  328 ++-
 tests/ocean/unit/test_al81_budget.py               |  220 ++
 tests/ocean/unit/test_baroclinic_decomposition.py  |   38 +-
 .../unit/test_barotropic_continuity_and_drag.py    |  337 ++-
 .../unit/test_barotropic_coriolis_null_mode.py     |  151 +-
 .../unit/test_barotropic_inertial_oscillation.py   |   24 +
 .../ocean/unit/test_barotropic_noise_invariant.py  |   78 +-
 tests/ocean/unit/test_box_heat_budget.py           |  274 +++
 tests/ocean/unit/test_deep_ventilation_fix.py      |   38 +-
 .../unit/test_dino_1226_unit_harness_binary_io.py  |   90 +
 ...dino_1226_unit_harness_dom_qco_dynldf_probes.py |   96 +
 tests/ocean/unit/test_dino_experiment.py           |  537 ++++-
 tests/ocean/unit/test_fidelity_bar_gate.py         |  159 ++
 tests/ocean/unit/test_fidelity_precision_gate.py   |  140 ++
 tests/ocean/unit/test_fidelity_time_levels.py      |   75 +
 tests/ocean/unit/test_geometric_eos_live_depth.py  |  237 ++
 tests/ocean/unit/test_gm_bolus_through_fct.py      |    8 +-
 tests/ocean/unit/test_gm_redi_latlon_cgrid.py      |   88 +
 tests/ocean/unit/test_hpg_tendency_compare.py      |  142 ++
 .../unit/test_joint_volume_salt_normalization.py   |  232 ++
 tests/ocean/unit/test_leapfrog_integrator.py       |   70 +
 tests/ocean/unit/test_linear_free_surface.py       |   35 +-
 .../test_momentum_diagnostics_closure_nemo_dino.py |  163 ++
 tests/ocean/unit/test_nemo_bn2.py                  |  174 ++
 tests/ocean/unit/test_nemo_io.py                   |   44 +-
 tests/ocean/unit/test_nemo_state_bridge.py         |   33 +
 tests/ocean/unit/test_nemo_zdfmxl_transcription.py |  733 ++++++
 tests/ocean/unit/test_neumann_fill_vertex_seam.py  |  101 +
 tests/ocean/unit/test_rk3_ws_and_mxl3.py           |    8 +-
 tests/ocean/unit/test_sharded_geom_fingerprint.py  |    8 +-
 tests/ocean/unit/test_shortwave_penetration.py     |   80 +
 tests/ocean/unit/test_stpmlf_call_coverage.py      |   86 +
 tests/ocean/unit/test_tendency_probe.py            |    8 +-
 tests/ocean/unit/test_tke_nemo_identity.py         |  707 +++++-
 tests/ocean/unit/test_tke_nemo_nn_mxl2.py          |  112 +
 tests/ocean/unit/test_tracer_pair_advection.py     |    4 +-
 tests/ocean/unit/test_treguier_gm.py               |  149 +-
 tests/ocean/unit/test_vertical_momentum_scheme.py  |  207 +-
 tests/ocean/unit/test_zad_bottom_face_mask.py      |  286 +++
 tests/parallel/test_build_band_grids.py            |   52 +
 tests/test_atmosphere_cross_grid_plots.py          |   91 +-
 tests/test_corner_div_damp_nh.py                   |  137 ++
 tests/test_d2a2c_ua_va_halo.py                     |  115 +
 tests/test_dispatch_hardening.py                   |   20 +
 tests/test_div_damp_adaptive.py                    |  123 +-
 tests/test_fv3_d_sw5_corner_corrections.py         |    4 +-
 tests/test_fv3_divergence_corner.py                |  319 ++-
 tests/test_pe_dycore_inline_imports.py             |   58 +-
 tests/unit/_params_reachability_baseline.py        |    8 +-
 tests/unit/test_afcrps_area_correction.py          |   68 +
 tests/unit/test_cdgrid_fv3_regression.py           |   36 +-
 tests/unit/test_clivi_radiative_ice_only.py        |  141 ++
 tests/unit/test_cloud_subcolumns.py                |  320 +++
 tests/unit/test_clubb_param_spec.py                |    2 +-
 tests/unit/test_cmor_duplicate_time_guard.py       |  131 ++
 tests/unit/test_conservative_positive_clip.py      |    4 +-
 .../test_convection_subsidence_solve_threading.py  |  487 ++++
 tests/unit/test_diagnostics_hybrid_pressure.py     |  240 ++
 .../unit/test_diff_atmosphere_convection_micro.py  | 1163 +++++++++
 tests/unit/test_diff_atmosphere_physics.py         |   66 +-
 tests/unit/test_diff_atmosphere_turb_gwd_rad.py    | 2326 ++++++++++++++++++
 tests/unit/test_diff_coupler.py                    |   26 -
 tests/unit/test_diff_jax_transforms.py             |   65 +
 tests/unit/test_diff_land_ice_schemes.py           |  825 +++++++
 tests/unit/test_diff_ml.py                         | 1981 ++++++++++++++++
 tests/unit/test_diff_ocean.py                      |  481 +++-
 tests/unit/test_diff_ocean_operators.py            | 1613 +++++++++++++
 tests/unit/test_diff_physics_params.py             |  441 +++-
 tests/unit/test_diff_sea_ice.py                    |   57 +
 tests/unit/test_diff_taylor_tests.py               |  191 +-
 tests/unit/test_diff_tools.py                      | 1477 ++++++++++++
 tests/unit/test_duogrid.py                         |    4 +-
 tests/unit/test_energy_budget_hybrid.py            |  119 +
 tests/unit/test_geometry_consistency.py            | 1439 ++++++++++++
 tests/unit/test_geometry_consistency_trace_gate.py |  105 +
 tests/unit/test_hines_launch_level.py              |  197 ++
 tests/unit/test_hybrid_level_validity.py           |  140 ++
 tests/unit/test_installed_package_imports.py       |  190 +-
 tests/unit/test_legoesm_paths_worktree.py          |   65 +
 tests/unit/test_les_record.py                      |  176 ++
 tests/unit/test_les_reference.py                   |  264 +++
 tests/unit/test_mpas_clt_feed.py                   |  297 +++
 tests/unit/test_no_module_top_jax_alloc.py         |    9 +-
 tests/unit/test_param_collector.py                 |    2 +-
 tests/unit/test_partial_coverage_optics.py         |  386 +++
 tests/unit/test_rce_cloud_radiation.py             |   24 +-
 tests/unit/test_run_amip_cli.py                    |    7 +
 tests/unit/test_scaling_preflight.py               |  152 ++
 tests/unit/test_scaling_preflight_ordering.py      |  116 +
 tests/unit/test_scm_rce_clubb_nesting.py           |    2 +-
 .../test_scm_rce_convection_intercomparison_cli.py |  240 ++
 .../unit/test_scm_rce_subsidence_solve_override.py |  123 +
 .../test_standalone_cloud_config_forwarding.py     |   88 +
 tests/unit/test_sweep_ocean_tests_runner.py        |   47 +
 tests/unit/test_tiled_kt_validation_gate.py        |   66 +
 tests/unit/test_train_scm_rce_params_cli.py        |  313 +++
 tests/unit/test_voronoi_surface_fields.py          |  148 ++
 uv.lock                                            |  173 +-
 361 files changed, 69421 insertions(+), 2954 deletions(-)
diff --git a/packages/core/legoesm/parallel/geometry_consistency.py b/packages/core/legoesm/parallel/geometry_consistency.py
new file mode 100644
index 000000000..0a131ca18
--- /dev/null
+++ b/packages/core/legoesm/parallel/geometry_consistency.py
@@ -0,0 +1,733 @@
+"""Cross-process agreement checks for per-process-recomputed SPMD geometry.
+
+Every multi-controller SPMD lane faces the same hazard: each process rebuilds
+the band/tile geometry from the same config, then hands it to a REPLICATED
+``device_put``.  A ``P()`` (fully-replicated) put ASSERTS the value is
+bit-identical on every process, and per-process XLA autotuning on
+device-derived grid fields makes the last ULPs differ at larger sizes (job
+26450848: LL576 np=4, area-scale fields differing at 1e-7 relative), which
+trips that assert.
+
+The remedy is to broadcast process 0's bytes — but broadcasting BLINDLY would
+silently paper over a REAL cross-process inconsistency (a different wet
+domain, a different field list, a mixed ``jax_enable_x64``), turning a loud
+crash into wrong physics.  So every broadcast here is GUARDED: an allgathered
+fingerprint must agree first, and a disagreement RAISES.
+
+This module is the ONE implementation of that protocol.  It was extracted
+from ``ocean.dynamics.sharded_ocean_step`` (where it was developed and
+hardened over five rounds of adversarial review) so the atmosphere lat-lon
+lane — which had the identical defect (#1362) — reuses it instead of growing
+a second, drifting copy.  Per legoESM's no-duplicated-numerics rule, new SPMD
+lanes MUST call these helpers rather than re-derive the fingerprints.
+
+Sequencing contract, in this order:
+
+1. :func:`assert_schema_agrees` ONCE, before any per-field work — a single
+   fixed-shape collective that every process reaches.  A process-dependent
+   field selection (e.g. an optional mask present on some ranks only) would
+   otherwise DESYNCHRONIZE the per-field gathers below instead of failing
+   with a clear message.
+2. :func:`broadcast_checked` per field, in an order identical on every
+   process.
+
+NO DEADLOCK RISK: every process fingerprints the same fields in the same
+order and derives its verdict from the SAME gathered array, so the refusal is
+symmetric — all raise or none.
+"""
+
+from __future__ import annotations
+
+import hashlib
+
+import jax
+import numpy as np
+
+__all__ = [
+    "content_hash48",
+    "name_digest48",
+    "schema_fingerprint",
+    "assert_schema_agrees",
+    "assert_flags_agree",
+    "broadcast_checked",
+    "coerce_count",
+    "coerce_bool",
+    "config_digest48",
+    "tree_schema_digest48",
+    "safe_repr",
+    "FLAG_ABSENT",
+    "FLAG_UNCOERCIBLE",
+    "FLAG_OUT_OF_RANGE",
+    "FLAG_NEGATIVE",
+    "FLAG_MAX_EXACT",
+    "FLAG_DIGEST_FAILED",
+]
+
+# --- entry-gate payload sentinels -------------------------------------------
+# An entry gate turns rank-local scalars (n_steps, segment_steps, grid dims)
+# into a fixed-width float payload.  Building that payload must NEVER raise:
+# a rank that dies in `int(n_steps)` while its peers block in
+# `process_allgather` is a HANG, which is strictly worse than the bug the gate
+# exists to fix (codex 2026-07-29 round-3, blocker 3).  So an unusable value is
+# mapped to a SENTINEL that travels through the collective; every rank then
+# sees it in the gathered payload and the raise that follows is symmetric.
+#
+# The sentinels are large-magnitude NEGATIVE values that NO legitimate count
+# can take.  They must also not collide with each other: ``FLAG_ABSENT`` used
+# to be ``-1.0``, so a rank passing ``segment_steps=None`` and a peer passing
+# ``-1`` produced the SAME payload entry, agreed, and then diverged downstream
+# (codex round-4, blocker 1).  Counts are validated non-negative, so every
+# sentinel is unreachable from valid data AND distinct from every other.
+FLAG_ABSENT = -6.0e15
+FLAG_UNCOERCIBLE = -8.0e15
+FLAG_OUT_OF_RANGE = -7.0e15
+FLAG_NEGATIVE = -5.0e15
+FLAG_DIGEST_FAILED = -4.0e15
+# 2**53 is the largest integer whose successor is exactly representable in
+# float64.  Above it two DIFFERENT counts alias to the same payload entry, so
+# the gate would pass a real divergence (codex round-3, minor 2).  Values past
+# the bound are refused rather than silently compared.
+FLAG_MAX_EXACT = 2.0 ** 53
+_MAX_EXACT_INT = 2 ** 53
+
+
+def safe_repr(value, limit: int = 120) -> str:
+    """``repr(value)`` that cannot raise and cannot blow up the message.
+
+    A user object whose ``__repr__`` raises would otherwise propagate out of
+    the payload build — the very pre-collective throw the gates exist to
+    remove (codex round-4, blocker 1).
+    """
+    try:
+        text = repr(value)
+    except Exception:                       # pragma: no cover - defensive
+        try:
+            text = f"<unrepresentable {type(value).__name__}>"
+        except Exception:                   # pragma: no cover - defensive
+            text = "<unrepresentable>"
+    return text if len(text) <= limit else text[:limit] + "..."
+
+
+def coerce_count(value, *, absent: float = FLAG_ABSENT):
+    """Map a rank-local COUNT to an exactly-comparable entry-gate payload float.
+
+    Returns ``(payload, problem)``.  ``problem`` is ``None`` when the value is
+    usable; otherwise it is a human-readable clause naming the offending value,
+    which the caller must raise AFTER its collective so the refusal is
+    symmetric across processes.
+
+    This function NEVER raises.  That is the whole point: it is called while
+    ASSEMBLING a collective payload, upstream of the collective itself, where a
+    raise deadlocks the peers (codex round-3, blocker 3).
+
+    STRICT by type, not by coercibility (codex round-4, blocker 1).  Only a
+    real non-negative Python/NumPy integer is accepted:
+
+    * ``3.5`` is REJECTED.  ``int(3.5) == 3`` made a rank carrying ``3.5``
+      indistinguishable from a peer carrying ``3``; the payloads agreed and
+      then ``range(3.5)`` blew up on one rank alone while its peer entered the
+      step collective.
+    * ``bool`` is REJECTED.  ``True`` is not a step count, and silently
+      encoding it as ``1`` hides a caller bug.
+    * Arrays (even size-1) are REJECTED: ``int(arr)`` succeeds for size 1 and
+      raises for size > 1, so accepting them makes the gate's behaviour depend
+      on rank-local shape.
+    * NEGATIVE integers get their OWN sentinel, so they can never collide with
+      the "absent" encoding.
+
+    ``None`` maps to ``absent`` (default :data:`FLAG_ABSENT`, itself outside
+    the valid range) so a call site that does not carry the value still emits a
+    FIXED-WIDTH payload.
+    """
+    if value is None:
+        return float(absent), None
+    # `bool` is a subclass of `int`, so it must be excluded FIRST.
+    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
+        return FLAG_UNCOERCIBLE, (
+            "must be a non-negative Python/NumPy integer (got type "
+            f"{type(value).__name__}: {safe_repr(value)}); every process must "
+            "be launched with the same value")
+    try:
+        as_int = int(value)
+    except Exception:                       # pragma: no cover - defensive
+        return FLAG_UNCOERCIBLE, (
+            f"could not be read as an integer ({safe_repr(value)})")
+    if as_int < 0:
+        return FLAG_NEGATIVE, (
+            f"must be non-negative (got {safe_repr(value)})")
+    # Range check on the INTEGER: converting to float first would already have
+    # collapsed 2**53+1 onto 2**53, so the very aliasing this guards against
+    # would be invisible to the guard.
+    if as_int > _MAX_EXACT_INT:
+        return FLAG_OUT_OF_RANGE, (
+            f"is outside the exactly-comparable range n <= 2**53 (got "
+            f"{safe_repr(value)}); beyond that bound two different counts "
+            f"alias to the same float64 payload entry and the cross-process "
+            f"agreement check would pass a real divergence")
+    return float(as_int), None
+
+
+def coerce_bool(value, *, absent: float = FLAG_ABSENT):
+    """Strict, NON-THROWING tri-state encoder for a rank-local BOOLEAN flag.
+
+    Returns ``(payload, problem)`` exactly like :func:`coerce_count`.
+    ``None`` -> ``absent`` ("not applicable at this call site").
+
+    Only a real ``bool`` / ``np.bool_`` is accepted.  ``bool(value)`` on an
+    arbitrary object RAISES for a multi-element array ("truth value of an array
+    is ambiguous") — inside a gate that is a pre-collective throw, i.e. a hang
+    (codex round-4, blocker 2).  Anything else becomes a sentinel that travels
+    through the collective and is refused symmetrically afterwards.
+    """
+    if value is None:
+        return float(absent), None
+    if isinstance(value, (bool, np.bool_)):
+        return (1.0 if value else 0.0), None
+    return FLAG_UNCOERCIBLE, (
+        f"must be a bool (got type {type(value).__name__}: "
+        f"{safe_repr(value)})")
+
+
+def _canonical_config_terms(obj, prefix: str = "", depth: int = 0,
+                            out=None, seen=None):
+    """Flatten a config object into ORDERED ``"path=value"`` strings.
+
+    Covers the STATIC scalars that select a compiled program: scheme literals,
+    integrator names, and every feature-gating bool (``fix_mass``,
+    ``fix_moisture``, ``use_polar_filter``, ...).  Arrays contribute only
+    ``dtype`` + ``shape`` — comparing their VALUES is the job of
+    :func:`broadcast_checked`, not of a cheap fixed-width entry gate.
+
+    Never raises: any unreadable field becomes a ``<unreadable>`` term, which
+    still participates in the comparison.
+    """
+    if out is None:
+        out, seen = [], set()
+    if depth > 4 or len(out) > 512:          # bounded work, bounded payload
+        return out
+    if id(obj) in seen:
+        return out
+    seen.add(id(obj))
+    fields = getattr(obj, "_fields", None)   # NamedTuple
+    if fields is None:
+        dc = getattr(obj, "__dataclass_fields__", None)
+        fields = tuple(dc) if dc else None
+    if fields is None:
+        return out
+    for name in fields:
+        try:
+            val = getattr(obj, name)
+        except Exception:                    # pragma: no cover - defensive
+            out.append(f"{prefix}{name}=<unreadable>")
+            continue
+        path = f"{prefix}{name}"
+        if val is None or isinstance(val, (bool, int, float, str, np.bool_,
+                                           np.integer, np.floating)):
+            out.append(f"{path}={safe_repr(val, 64)}")
+        elif hasattr(val, "dtype") and hasattr(val, "shape"):
+            out.append(f"{path}=array:{safe_repr(val.dtype, 32)}:"
+                       f"{safe_repr(tuple(val.shape), 64)}")
+        elif getattr(val, "_fields", None) or getattr(
+                val, "__dataclass_fields__", None):
+            _canonical_config_terms(val, path + ".", depth + 1, out, seen)
+        else:
+            out.append(f"{path}=<{type(val).__name__}>")
+    return out
+
+
+def config_digest48(obj) -> float:
+    """One fixed-width, order-sensitive digest of a config's STATIC scalars.
+
+    Why a digest instead of a hand-listed set of flags: an entry gate that
+    enumerates ``fold``/``anchor``/``polar`` by hand agrees only the fields
+    somebody remembered.  ``fix_mass`` gates a global-area psum,
+    ``outer_integrator`` selects a different program, ``fix_moisture`` adds a
+    reduction — each was MISSING from the hand-written list (codex round-4,
+    blocker 3).  Digesting every static scalar closes the class instead of the
+    three instances, and costs ONE payload entry.
+
+    Never raises; an internal failure returns :data:`FLAG_DIGEST_FAILED`,
+    which still compares equal across ranks that fail identically and unequal
+    against a rank that succeeded.
+    """
+    try:
+        return name_digest48(_canonical_config_terms(obj))
+    except Exception:                        # pragma: no cover - defensive
+        return FLAG_DIGEST_FAILED
+
+
+def tree_schema_digest48(tree) -> float:
+    """Digest of a pytree's LEAF SCHEMA: ordered path, dtype and full shape.
+
+    The gather/scatter entry points run one cross-process replication PER
+    NON-``None`` LEAF, so the NUMBER and ORDER of those collectives is
+    rank-local data: a state whose tracer dict differs across processes (extra
+    species, different insertion order, different shape) produces mismatched
+    schedules and hangs (codex round-4, blocker 5).  Folding the whole leaf
+    schema into ONE fixed-width float makes that a clean symmetric raise.
+
+    ``jax.tree_util`` key paths give a canonical, ORDER-SENSITIVE description
+    (dict keys are sorted by ``tree_flatten_with_path``, so an insertion-order
+    difference alone does not false-positive, while a KEY-SET difference does
+    move the digest).  Never raises.
+    """
+    try:
+        from jax.tree_util import tree_flatten_with_path, keystr
+        leaves, _ = tree_flatten_with_path(tree)
+        terms = []
+        for path, leaf in leaves:
+            dtype = getattr(leaf, "dtype", None)
+            shape = getattr(leaf, "shape", None)
+            terms.append(
+                f"{keystr(path)}:{safe_repr(dtype, 32)}:"
+                f"{safe_repr(tuple(shape) if shape is not None else None, 64)}")
+        return name_digest48(terms)
+    except Exception:                        # pragma: no cover - defensive
+        return FLAG_DIGEST_FAILED
+
+
+def _dtype_kind_and_ndim(a):
+    """``(kind, ndim)`` for the schema digest, tolerant of a plain scalar.
+
+    Reads ``.dtype``/``.ndim`` from METADATA when present (a jax array exposes
+    both without materialising, so no device sync).  A plain Python scalar or
+    list has neither; falling through to ``np.asarray`` there is FREE (it is
+    already host data) and, critically, keeps this function from dying with a
+    bare ``AttributeError`` BEFORE :func:`assert_schema_agrees` reaches its
+    collective — a rank-local raise ahead of a collective is a HANG, so
+    "fail explicitly" here must NOT mean "raise here" (codex round-3, minor 3).
+
+    An unsupported dtype class (object/str) is reported as ``unsupported:<k>``
+    rather than being silently bucketed with the float fields, so a
+    disagreement about it is visible in the digest and a same-on-all-ranks
+    unsupported field fails later in :func:`broadcast_checked` with its own
+    message instead of here.
+    """
+    dtype = getattr(a, "dtype", None)
+    if dtype is None:
+        host = np.asarray(a)
+        dtype, ndim = host.dtype, host.ndim
+    else:
+        ndim = int(getattr(a, "ndim", np.ndim(a)))
+    k = np.dtype(dtype).kind
+    if k in "biu":
+        kind = "exact"
+    elif k in "fc":
+        kind = "inexact"
+    else:
+        kind = f"unsupported:{k}"
+    return kind, int(ndim)
+
+
+# Every per-field collective payload is padded to these FIXED widths.  A
+# payload whose LENGTH depends on rank-local data (dtype class, ndim,
+# non-finite count) would let two processes enter `process_allgather` with
+# different shapes and DEADLOCK -- the exact failure this module exists to
+# turn into a clean symmetric raise (codex 2026-07-29, blocker 2; the flaw was
+# inherited from the pre-extraction ocean implementation, so fixing it here
+# fixes BOTH lanes).
+_STRUCT_WIDTH = 8
+_VALS_WIDTH = 3
+
+# Relative tolerance for FLOAT geometry fields. Only ULP-scale autotune drift
+# is expected there; quantize-then-assert-equal false-positived on a rounding
+# boundary (job 26453240), so compare with a tolerance instead.
+_FLOAT_RTOL = 1e-5
+
+
+def content_hash48(arr) -> float:
+    """48-bit content digest of ``arr``'s bytes, exactly representable in f64.
+
+    Used to compare EXACT-dtype arrays (masks, index tables) across
+    processes: unlike moment fingerprints, a byte digest is positional, so a
+    permutation or a two-cell flip cannot cancel. 48 bits keeps the value
+    under 2**53 so it survives the float64 ``process_allgather`` payload
+    exactly. Not cryptographic — collision-resistance at 2**-48 is far
+    beyond the ~10 setup-time comparisons this guard makes.
+    """
+    a = np.ascontiguousarray(arr)
+    h = hashlib.blake2b(a.tobytes(), digest_size=6)
+    return float(int.from_bytes(h.digest(), "big"))
+
+
+def name_digest48(names) -> float:
+    """Order-sensitive, UNAMBIGUOUS digest of a sequence of names.
+
+    Uses a NUL separator, which cannot occur in a Python identifier or any
+    legoESM field name, so ``["a,b", "c"]`` and ``["a", "b,c"]`` cannot
+    collide.  A plain ``",".join`` COULD (codex 2026-07-29, minor 5): those
+    two lists have the same length, so a count check does not separate them
+    either.
+    """
+    joined = "\x00".join(names).encode()
+    return float(int.from_bytes(
+        hashlib.blake2b(joined, digest_size=6).digest(), "big"))
+
+
+def schema_fingerprint(names, n_dev, dtype_kinds=(), ndims=()) -> np.ndarray:
+    """Fixed-shape schema digest gathered ONCE before the per-field loop.
+
+    Covers the field-name list (order-sensitive), the count, the x64 flag,
+    ``n_dev``, and -- critically -- the per-field DTYPE CLASS and NDIM.
+
+    The dtype/ndim terms are not cosmetic.  :func:`broadcast_checked` routes
+    exact dtypes to a 1-value digest and float dtypes to a 3-moment
+    fingerprint, and its struct entry depends on ndim.  If the schema gate
+    did not cover those, a field that is bool on one process and float on
+    another would PASS the gate and then deadlock inside the per-field
+    gather with mismatched payloads.  Catching it here converts that hang
+    into a clean symmetric RuntimeError (codex 2026-07-29, blocker 2).
+
+    ``dtype_kinds``/``ndims`` default to empty for callers that have not yet
+    resolved the arrays; passing them is strongly preferred.
+    """
+    return np.array(
+        [float(len(names)),
+         name_digest48(names),
+         float(bool(jax.config.jax_enable_x64)),
+         float(n_dev),
+         name_digest48([str(k) for k in dtype_kinds]),
+         name_digest48([str(int(n)) for n in ndims])],
+        dtype=np.float64)
+
+
+def in_jax_trace() -> bool:
+    """True when the caller runs inside a JAX trace (``jit``/``scan``/``vmap``).
+
+    The host-side gates below call ``multihost_utils.process_allgather``, which
+    is an EAGER utility: it ``device_put``s its payload per addressable device.
+    Under an active trace those puts are staged into the jaxpr and come back as
+    tracers, so ``make_array_from_single_device_arrays`` is handed tracers and
+    raises — every multi-process lat-lon SPMD run died this way once the step
+    was wrapped in ``lax.scan``/``jax.jit`` (#1405, follow-up to #1362).
+
+    TWO LIMITATIONS, stated because a reader will otherwise assume they are
+    covered (both raised by codex adversarial review of this change, both
+    accepted deliberately — the alternative is a lane that cannot run at all):
+
+    1. The skip is symmetric only as long as every process reaches this call
+       in the SAME transform state, which is the SPMD lockstep property the
+       gate itself exists to enforce.  If one rank called the step eagerly
+       while another traced it, the eager rank would now BLOCK in
+       ``process_allgather`` instead of its peer crashing.  That divergence is
+       already fatal today (the traced rank dies here), so this trades a
+       guaranteed crash on every multi-process traced run for a hang in an
+       already-divergent one.  It is NOT a proof of symmetry.
+    2. Coverage IS lost on a lane that is only ever traced.  The build-time
+       gates (``_agree_spmd_entry``) agree the model/mesh/config; the per-CALL
+       payload — state pytree schema, ``phys_state``/forcing presence and its
+       schema — is agreed ONLY here, and under a trace it now goes unchecked.
+
+    The trace-safe design that would fix both (stage the digest comparison as
+    a mesh collective inside the traced program instead of a host allgather)
+    needs a real multi-process rig to validate and is deliberately left as
+    follow-up rather than written blind — see #1405.
+
+    ``jax.core.trace_state_clean`` was removed from the public ``jax.core`` in
+    jax 0.7 and survives only as ``jax._src.core``, so this reads the private
+    module.  The ``except`` returns False — i.e. the gate RUNS and the traced
+    lane crashes loudly again — deliberately: for a correctness gate a loud
+    crash beats a silent skip.  ``tests/unit/test_geometry_consistency_trace_
+    gate.py`` asserts this returns True inside ``jax.jit`` AND inside
+    ``lax.scan``, so a JAX version that moves the symbol turns CI red first.
+    """
+    try:
+        from jax._src import core as _jax_core
+        return not _jax_core.trace_state_clean()
+    except Exception:  # pragma: no cover - JAX internal moved; test goes red
+        return False
+
+
+def assert_flags_agree(names, values, *, context: str) -> None:
+    """Raise unless every process agrees on a tuple of rank-local CONFIG flags.
+
+    Call this BEFORE any rank-local ``raise`` that inspects per-process
+    config.  Otherwise one process can reject its config and exit while its
+    peers proceed into a collective and block forever — a collective-ORDER
+    violation whose symptom (hang vs backend error) is backend-dependent
+    (codex 2026-07-29, blocker 1).
+
+    ``names`` and ``values`` must be STATIC tuples written at the call site,
+    so the payload length is fixed by the code path rather than by data.
+
+    No-op under a JAX trace — see :func:`in_jax_trace` (#1405).
+    """
+    if jax.process_count() <= 1 or in_jax_trace():
+        return
+    from jax.experimental import multihost_utils
+
+    payload = np.array(
+        [float(len(values)), name_digest48(names),
+         *(float(v) for v in values)], dtype=np.float64)
+    gathered = multihost_utils.process_allgather(payload)
+    if not bool(np.all(gathered == gathered[0])):
+        raise RuntimeError(
+            f"{context}: per-process CONFIG differs across processes "
+            f"(flags {list(names)} -> gathered {gathered.tolist()}). Every "
+            f"process must be built from the same config; refusing before "
+            f"any rank-local rejection so the failure is symmetric rather "
+            f"than a hang.")
+
+
+def assert_schema_agrees(names, n_dev, *, context: str, arrays=None) -> None:
+    """Raise unless every process agrees on the geometry field SCHEMA.
+
+    ``names`` must be an ORDERED sequence — the per-field
+    :func:`broadcast_checked` calls that follow are matched positionally
+    across processes, so a reordering is itself a divergence worth catching.
+
+    Pass ``arrays`` (the per-name arrays, same order) so the gate also covers
+    each field's DTYPE CLASS and NDIM.  Those decide the per-field payload
+    SHAPE in :func:`broadcast_checked`, so leaving them out lets a
+    bool-vs-float disagreement slip past this gate and deadlock in the
+    per-field gather instead of raising here.
+
+    No-op when ``jax.process_count() == 1``.
+    """
+    if jax.process_count() <= 1:
+        return
+    from jax.experimental import multihost_utils
+
+    names = list(names)
+    if arrays is None:
+        kinds, ndims = (), ()
+    else:
+        # Read dtype/ndim from array METADATA, never via np.asarray: a jax
+        # array exposes both without materialising, so forcing a host copy
+        # here would add a device sync per field AND could itself fail
+        # (transfer error / OOM) BEFORE the collective below — reintroducing
+        # the very "one rank exits while a peer blocks" hazard this gate
+        # exists to remove (codex round-2 minor). `broadcast_checked` does
+        # the single real materialisation later.
+        #
+        # `_dtype_kind_and_ndim` also survives a plain Python scalar, which a
+        # bare `a.dtype` read did not (codex round-3, minor 3): no production
+        # caller passes one today, but an AttributeError HERE would be a
+        # rank-local raise BEFORE the collective, i.e. a hang rather than a
+        # clear failure.
+        described = [_dtype_kind_and_ndim(a) for a in arrays]
+        kinds = [d[0] for d in described]
+        ndims = [d[1] for d in described]
+    gathered = multihost_utils.process_allgather(
+        schema_fingerprint(names, n_dev, kinds, ndims))
+    if not bool(np.all(gathered == gathered[0])):
+        raise RuntimeError(
+            f"{context}: the band-geometry SCHEMA differs across processes "
+            f"(field list / x64 setting / device count / per-field dtype "
+            f"class / ndim — gathered {gathered.tolist()}). Fix the "
+            f"per-process config before sharding; the per-field checks "
+            f"assume one schema.")
+
+
+def broadcast_checked(arr, name: str, *, context: str) -> np.ndarray:
+    """Verify ``arr`` agrees across processes, then broadcast process 0's bytes.
+
+    Multi-process: returns a host ``np.ndarray`` that is bit-identical on
+    every process, safe to hand to a replicated ``device_put``.
+    Single-process: returns ``arr`` ITSELF, untouched — no collectives, no
+    host round trip, no dtype/weak-type change.
+
+    The fingerprint compares structural entries exactly; value entries
+    EXACTLY for integer/bool arrays and to ``rtol=1e-5`` for float arrays.
+
+    Integer/bool arrays (masks, index tables) are exact data, not autotuned
+    arithmetic: their BYTES are fingerprinted so a positional difference is
+    caught.  Moment-only compares are blind to a permutation — a bool mask's
+    ``(sum, sumsq, absmax)`` is identical for every arrangement with the same
+    true-count (codex round-5).  A mask that genuinely differs across
+    processes means different wet domains = different physics: refusing is
+    the correct outcome, not a false alarm.
+
+    Residual, documented: a float divergence preserving sum, sum-of-squares
+    AND absmax to ``rtol`` is not detected.  Band grids are analytic in
+    lat/lon, so any real inconsistency moves those moments.
+    """
+    # EARLY return, BEFORE np.asarray: single process has nothing to compare,
+    # and converting here would force a device->host->device round trip and
+    # strip weak-type metadata on a 1-process mesh. The ocean lane already
+    # held host arrays so it was unaffected, but the atmosphere lane passes
+    # `jnp.stack` results straight in and WAS regressed by an unconditional
+    # conversion (codex 2026-07-29, major 3). Return the caller's object
+    # untouched.
+    if jax.process_count() <= 1:
+        return arr
+    from jax.experimental import multihost_utils
+
+    host = np.asarray(arr)
+    flat = host.ravel()
+    is_exact = host.dtype.kind in "biu"
+    # FIXED-WIDTH payloads (see _STRUCT_WIDTH/_VALS_WIDTH): the gathered shape
+    # must never depend on rank-local data, or two processes can enter this
+    # collective with different shapes and hang. Shape is folded in as a
+    # digest rather than splatted, so an ndim difference cannot change the
+    # length either.
+    struct = np.zeros(_STRUCT_WIDTH, dtype=np.float64)
+    struct[0] = float(host.ndim)
+    struct[1] = float(np.dtype(host.dtype).num)
+    struct[2] = float(1.0 if is_exact else 0.0)
+    struct[3] = float(host.size)
+    struct[4] = name_digest48([str(d) for d in host.shape])
+    vals = np.zeros(_VALS_WIDTH, dtype=np.float64)
+    if is_exact:
+        vals[0] = content_hash48(host)
+    else:
+        finite = flat[np.isfinite(flat)]
+        f64 = finite.astype(np.float64)
+        # Non-finite COUNT is structural: a NaN appearing on one process only
+        # must not be averaged away by the moment compare below.
+        struct[5] = float(flat.size - finite.size)
+        vals[0] = float(f64.sum()) if f64.size else 0.0
+        vals[1] = float((f64 * f64).sum()) if f64.size else 0.0
+        vals[2] = float(np.abs(f64).max()) if f64.size else 0.0
+
+    g_struct = multihost_utils.process_allgather(struct)
+    g_vals = multihost_utils.process_allgather(vals)
+    struct_ok = bool(np.all(g_struct == g_struct[0]))
+    if is_exact:
+        vals_ok = bool(np.all(g_vals == g_vals[0]))
+    else:
+        vals_ok = bool(np.allclose(g_vals, g_vals[0],
+                                   rtol=_FLOAT_RTOL, atol=0.0))
+    if not (struct_ok and vals_ok):
+        raise RuntimeError(
+            f"{context}: geometry field {name!r} DIVERGES across processes "
+            f"(struct_ok={struct_ok}, vals_ok={vals_ok}, "
+            f"exact_dtype={is_exact}, gathered={g_vals.tolist()}) — a real "
+            f"config/grid inconsistency, not autotune noise; refusing to "
+            f"broadcast process 0 over it.")
+    return np.asarray(multihost_utils.broadcast_one_to_all(host))
+
+
+# --- assert-free sharded puts + per-band gates (2026-08-03, ocean walls) ----
+# Three stacked multicontroller walls were found on the ocean lane (codex
+# r14-r19; PR #1457): (1) broadcast_one_to_all of a band stack lowers to an
+# [n_processes, stack] psum program (nd x 849 MB at LL2304 L20 — 81.5 GB at
+# 96 procs); (2) jax.device_put of a NUMPY array onto an all-process
+# sharding internally runs multihost_utils.assert_equal on the FULL array
+# ([n_proc, field] landing on ONE device: fits under an 80 GB A100 up to
+# ~64 procs, dies at 96 — jax _src/dispatch.py::_device_put_sharding_impl);
+# (3) a concrete sharded-global array captured by an OUTER trace (jit-of-
+# jit) becomes an MLIR constant whose value cannot be fetched for
+# non-addressable arrays. The helpers below remove (1) and (2) — (3) is the
+# callers' aux-threading contract, see make_sharded_ocean_step.
+
+def band_fingerprint(host, n_bands):
+    """Per-band fingerprint of a band-STACKED field (leading axis n_bands).
+
+    PREREQUISITE: ``n_bands`` (and each field's dtype class / shape) must
+    already be schema-gated across processes (:func:`assert_schema_agrees`)
+    — the payload widths depend on it, and mismatched widths would hang the
+    allgather rather than raise.
+
+    Exact dtypes (int/bool/uint): one positional 48-bit byte digest per
+    band. Floats: per-band ``[sum, sum_of_squares, absmax]`` of finite
+    entries plus per-band non-finite counts folded into ``struct``.
+    Per-band (not whole-array) because each process's OWN bytes become the
+    live inputs for the bands it owns under the assert-free put: a
+    band-local drift must not hide in a whole-array sum (codex r14).
+    DOCUMENTED RESIDUALS: a within-band float change preserving all three
+    moments to rtol, and non-finite entries changing position/kind at a
+    fixed per-band count, pass the float gate.
+    """
+    host = np.asarray(host)
+    if host.shape[0] != n_bands:
+        raise ValueError(
+            f"band_fingerprint: leading axis {host.shape[0]} != n_bands "
+            f"{n_bands}")
+    is_exact = host.dtype.kind in "biu"
+    struct = [float(host.ndim), *map(float, host.shape),
+              float(np.dtype(host.dtype).num)]
+    if is_exact:
+        vals = np.array([content_hash48(host[b]) for b in range(n_bands)],
+                        dtype=np.float64)
+    else:
+        per_band = []
+        for b in range(n_bands):
+            flat = host[b].ravel()
+            finite = flat[np.isfinite(flat)]
+            f64 = finite.astype(np.float64)
+            struct.append(float(flat.size - finite.size))
+            per_band.extend([
+                float(f64.sum()) if f64.size else 0.0,
+                float((f64 * f64).sum()) if f64.size else 0.0,
+                float(np.abs(f64).max()) if f64.size else 0.0,
+            ])
+        vals = np.array(per_band, dtype=np.float64)
+    return np.array(struct, dtype=np.float64), vals, is_exact
+
+
+def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=None):
+    """True iff every process's :func:`band_fingerprint` matches process 0's."""
+    if rtol is None:
+        rtol = _FLOAT_RTOL
+    struct_ok = bool(np.all(g_struct == g_struct[0]))
+    if is_exact:
+        vals_ok = bool(np.all(g_vals == g_vals[0]))
+    else:
+        vals_ok = bool(np.allclose(g_vals, g_vals[0], rtol=rtol, atol=0.0))
+    return struct_ok and vals_ok
+
+
+def checked_shard_put(arr, name, sharding, *, context, n_bands):
+    """Gate a band-stacked field per band, then put WITHOUT broadcast or
+    jax's whole-array device_put assert (walls 1+2 above).
+
+    Single-process: plain ``jax.device_put`` — byte-unchanged, no host
+    round trip. Multi-process: per-band fingerprint gate (symmetric raise
+    on real divergence), then ``jax.make_array_from_callback`` hands each
+    process exactly its addressable slabs. Cross-process byte-identity of
+    NON-owned bands is not required — owned bands are the only bytes that
+    reach any device, and their drift is bounded by the gate.
+    """
+    if jax.process_count() <= 1:
+        return jax.device_put(arr, sharding)
+    from jax.experimental import multihost_utils
+
+    host = np.asarray(arr)
+    struct, vals, is_exact = band_fingerprint(host, n_bands)
+    g_struct = multihost_utils.process_allgather(struct)
+    g_vals = multihost_utils.process_allgather(vals)
+    if not band_fingerprints_agree(g_struct, g_vals, is_exact):
+        raise RuntimeError(
+            f"{context}: band-stacked field {name!r} DIVERGES across "
+            f"processes (exact_dtype={is_exact}, "
+            f"gathered={g_vals.tolist()}) — a real config/grid "
+            f"inconsistency, not autotune noise; refusing to shard it.")
+    return jax.make_array_from_callback(
+        host.shape, sharding, lambda idx: host[idx])
+
+
+def assert_pytree_bytes_equal(tree, what):
+    """Cheap multi-process replacement for the per-leaf assert_equal that
+    :func:`checked_shard_put`-style puts bypass on NON-band inputs (state /
+    forcing pytrees): one 48-bit digest per array leaf, one tiny allgather,
+    symmetric raise on mismatch. No-op single-process.
+    """
+    if jax.process_count() <= 1:
+        return
+    from jax.experimental import multihost_utils
+
+    leaves = [x for x in jax.tree_util.tree_leaves(tree)
+              if hasattr(x, "ndim")]
+    vals = np.array([content_hash48(np.asarray(x)) for x in leaves],
+                    dtype=np.float64)
+    g = multihost_utils.process_allgather(vals)
+    if not bool(np.all(g == g[0])):
+        bad = [i for i in range(len(leaves))
+               if not bool(np.all(g[:, i] == g[0, i]))]
+        raise RuntimeError(
+            f"{what}: array leaves {bad} differ across processes (48-bit "
+            f"byte digests disagree) — the per-process inputs are NOT "
+            f"identical, which jax's device_put assert would have refused. "
+            f"Fix the per-process build before sharding.")
+
+
+def addressable_shard_put(arr, sharding):
+    """Ungated assert-free put (walls 1+2) for inputs whose cross-process
+    consistency the CALLER has already gated (state/forcing pytrees via
+    :func:`assert_pytree_bytes_equal`). Single-process: plain device_put."""
+    if jax.process_count() <= 1:
+        return jax.device_put(arr, sharding)
+    host = np.asarray(arr)
+    return jax.make_array_from_callback(
+        host.shape, sharding, lambda idx: host[idx])
diff --git a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
index 3059ddbd1..f9e7ecb7e 100644
--- a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
+++ b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
@@ -11,13 +11,12 @@ cubed-sphere ``parallel.sharded_dynamics.make_sharded_step``.
 
 Architecture (the two non-trivial pieces — see ``omip-multinode-spmd-scope``):
 
-* GRID = **band-stacked, P("lat")-SHARDED** (Structure A, sharded since
-  #1370-iii). The ``N`` band geometries are built host-side
-  (``build_band_grids``), their ARRAY fields are ``jnp.stack``-ed over a
-  leading band axis, sharded ``P("lat")`` so each device holds ONLY its own
-  band's slab, and the in-``shard_map`` body reads its local slab at
-  ``[0]``. This AVOIDS staggered-sharding the grid itself (the v-row is
-  ``n_lat+1``, coprime with ``n_lat`` for ``N>1``). The
+* GRID = **replicated-stacked, indexed** (Structure A). The ``N`` band
+  geometries are built host-side (``build_band_grids``), their ARRAY fields are
+  ``jnp.stack``-ed into a replicated pytree, and the in-``shard_map`` body picks
+  its own band by ``jax.lax.axis_index("lat")``. The grid is 2-D/small so
+  replication is cheap, and this AVOIDS staggered-sharding the grid (the v-row
+  is ``n_lat+1``, coprime with ``n_lat`` for ``N>1``). The
   ``LatLonCGridGeometry`` SCALAR fields (``n_lat``, ``n_lon``, ``radius``,
   ``dlon``, ``dlat``, ``fold``) stay STATIC python values — they gate trace-time
   ``if``\ s and ``jnp.zeros((n_lat, ...))`` shape builds, so a traced ``n_lat``
@@ -46,6 +45,11 @@ from __future__ import annotations
 import jax
 import jax.numpy as jnp
 import numpy as np
+from legoesm.parallel.geometry_consistency import (
+    FLAG_ABSENT, addressable_shard_put, assert_flags_agree,
+    assert_pytree_bytes_equal, assert_schema_agrees, checked_shard_put,
+    coerce_bool, coerce_count, config_digest48, name_digest48,
+    tree_schema_digest48)
 from jax.sharding import NamedSharding, PartitionSpec as P
 
 try:                                   # JAX >= 0.8 exposes shard_map at top level
@@ -203,105 +207,6 @@ def build_band_grids(grid, n_devices: int):
     ]
 
 
-def _content_hash48(arr) -> float:
-    """48-bit content digest of ``arr``'s bytes, exactly representable in f64.
-
-    Used to compare EXACT-dtype arrays (masks, index tables) across
-    processes: unlike moment fingerprints, a byte digest is positional, so a
-    permutation or a two-cell flip cannot cancel. 48 bits keeps the value
-    under 2**53 so it survives the float64 ``process_allgather`` payload
-    exactly. Not cryptographic — collision-resistance at 2**-48 is far
-    beyond the ~10 setup-time comparisons this guard makes.
-    """
-    import hashlib
-
-    a = np.ascontiguousarray(arr)
-    h = hashlib.blake2b(a.tobytes(), digest_size=6)
-    return float(int.from_bytes(h.digest(), "big"))
-
-def geom_band_fingerprint(host, n_bands):
-    """Low-memory cross-process fingerprint of a band-STACKED geometry field.
-
-    ``host`` has leading axis ``n_bands`` (the per-device band stack). The
-    fingerprint is PER BAND (codex 2026-08-03 r14: whole-array moments let a
-    band-local drift hide in the global sum once each process's own bytes
-    become live computation inputs):
-
-    * exact dtypes (int/bool/uint — masks, index tables): one positional
-      48-bit byte digest per band (a permutation or two-cell flip within a
-      band cannot cancel);
-    * float dtypes: per-band ``[sum, sum_of_squares, absmax]`` of the finite
-      entries in float64, plus per-band non-finite counts in ``struct``.
-      DOCUMENTED RESIDUALS: a within-band float change preserving all
-      three moments to rtol is not detected, and non-finite entries that
-      change POSITION or kind (nan vs inf) with an unchanged per-band
-      count also pass; band grids are analytic in lat/lon, so any real
-      inconsistency moves the moments.
-
-    Returns ``(struct, vals, is_exact)`` as float64 arrays safe for
-    ``process_allgather``.
-    """
-    import numpy as _np
-
-    host = _np.asarray(host)
-    assert host.shape[0] == n_bands, (host.shape, n_bands)
-    is_exact = host.dtype.kind in "biu"
-    struct = [float(host.ndim), *map(float, host.shape),
-              float(_np.dtype(host.dtype).num)]
-    if is_exact:
-        vals = _np.array([_content_hash48(host[b]) for b in range(n_bands)],
-                         dtype=_np.float64)
-    else:
-        per_band = []
-        for b in range(n_bands):
-            flat = host[b].ravel()
-            finite = flat[_np.isfinite(flat)]
-            f64 = finite.astype(_np.float64)
-            struct.append(float(flat.size - finite.size))
-            per_band.extend([
-                float(f64.sum()) if f64.size else 0.0,
-                float((f64 * f64).sum()) if f64.size else 0.0,
-                float(_np.abs(f64).max()) if f64.size else 0.0,
-            ])
-        vals = _np.array(per_band, dtype=_np.float64)
-    return _np.array(struct, dtype=_np.float64), vals, is_exact
-
-
-def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=1e-5):
-    """True iff every process's fingerprint matches process 0's.
-
-    ``g_struct``/``g_vals`` are the ``process_allgather``-ed outputs of
-    :func:`geom_band_fingerprint` (leading axis = process). Structure and
-    exact-dtype digests compare EXACTLY; float moments to ``rtol``.
-    """
-    import numpy as _np
-
-    struct_ok = bool(_np.all(g_struct == g_struct[0]))
-    if is_exact:
-        vals_ok = bool(_np.all(g_vals == g_vals[0]))
-    else:
-        vals_ok = bool(_np.allclose(g_vals, g_vals[0], rtol=rtol, atol=0.0))
-    return struct_ok and vals_ok
-
-
-
-def _schema_fingerprint(names, n_dev) -> np.ndarray:
-    """Fixed-shape schema digest: field-name list, count, x64 flag, n_dev.
-
-    Gathered ONCE before the per-field loop so a process-dependent field
-    selection is caught by a collective every process reaches, instead of
-    desynchronizing the per-field gathers (codex round-5 findings 3/4).
-    """
-    import hashlib
-
-    joined = ",".join(names).encode()
-    digest = float(int.from_bytes(
-        hashlib.blake2b(joined, digest_size=6).digest(), "big"))
-    return np.array(
-        [float(len(names)), digest, float(bool(jax.config.jax_enable_x64)),
-         float(n_dev)], dtype=np.float64)
-
-
 def _build_band_vertex_masks(model, n_dev):
     """Per-band vertex masks (n_lat/N+1, n_lon+1): SLICE the model's primed GLOBAL
     vertex mask ``[s : e+1]`` per band (the v/q-row stagger ``slice_cgrid_geometry_
@@ -335,65 +240,52 @@ def _build_band_vertex_masks(model, n_dev):
     return [jnp.asarray(vmask[r * nl: r * nl + nl + 1]) for r in range(n_dev)]
 
 
-def _addressable_shard_put(arr, sharding):
-    """Put an array onto a (possibly multi-process) sharding WITHOUT jax's
-    whole-array cross-process ``assert_equal``.
-
-    Under multicontroller, ``jax.device_put(numpy_array, sharding)`` calls
-    ``multihost_utils.assert_equal`` on the FULL array
-    (jax _src/dispatch.py::_device_put_sharding_impl) — a
-    ``process_allgather`` whose output is ``[n_processes, *shape]`` on one
-    device: at LL2304 L20 one 3-D field is 849 MB, so 96 processes fetch
-    81.5 GB > an 80 GB A100 (the wall that killed oc @96/@128, jobs
-    26644681/26644682, AFTER the geometry-broadcast fix; @64 = 54.3 GB
-    just fit, which is why smaller ladders never saw it).
-
-    Single-process: the historical ``jax.device_put``, byte-unchanged and
-    with no host round-trip (codex r17 — the input may already be a jax
-    device array). Multicontroller: ``jax.make_array_from_callback``
-    supplies each process's addressable shards directly — no consistency
-    collective. The ``device_put`` bit-identity CONTRACT is preserved by
-    the callers' cheap exact-hash gate (:func:`assert_pytree_bytes_equal`)
-    instead of jax's full-array allgather.
-    """
-    if jax.process_count() <= 1:
-        return jax.device_put(arr, sharding)
-    host = np.asarray(arr)
-    return jax.make_array_from_callback(
-        host.shape, sharding, lambda idx: host[idx])
-
-
-def assert_pytree_bytes_equal(tree, what):
-    """Cheap multi-process replacement for the per-leaf ``assert_equal``
-    that :func:`_addressable_shard_put` bypasses (codex r17 HIGH-1).
+# Ordered flag names for the ocean MESH+TREE gate used by the scatter/gather
+# bridges and by the returned SPMD callable. STATIC tuple: fixed width.
+_OCEAN_MESH_ENTRY_FLAGS = (
+    "has_mesh", "n_dev", "n_axes", "axis_names", "axis_sizes",
+    "has_tree", "tree_schema",
+)
 
-    Hashes every array leaf's BYTES (48-bit positional digest — the same
-    exactness as ``device_put``'s contract) into one small vector,
-    allgathers it, and refuses on any cross-process mismatch. Cost is one
-    tiny collective + a host-side hash pass, independent of process count
-    — vs jax's [n_processes, full_array] allgather.
 
-    No-op single-process. Symmetric: every process hashes the same leaves
-    in the same order, so all raise or none.
+def _ocean_mesh_axis_terms(mesh):
+    """``(axis_names, axis_sizes)`` term lists; never raises."""
+    if mesh is None:
+        return (), ()
+    try:
+        names = tuple(str(a) for a in mesh.axis_names)
+    except Exception:                       # pragma: no cover - defensive
+        return ("<unreadable>",), ("<unreadable>",)
+    try:
+        shape = dict(mesh.shape)
+        sizes = tuple(f"{n}={shape.get(n, '?')}" for n in names)
+    except Exception:                       # pragma: no cover - defensive
+        sizes = ("<unreadable>",)
+    return names, sizes
+
+
+def _agree_ocean_mesh_entry(mesh, tree=None, *, where: str) -> None:
+    """Agree the mesh AND a pytree LEAF SCHEMA before a scatter/gather.
+
+    #1362 round 4, blockers 5-6.  Both directions are collective here: the
+    gather's ``replicate_leaf`` compiles a jit identity with replicated
+    ``out_shardings``, and the scatter's DIRECT ``device_put`` of a full
+    global array onto a cross-process ``NamedSharding`` falls back to an
+    all-gather (documented on ``latlon_spmd.shard_leaf``) -- so the earlier
+    "SCATTER, therefore no collective" exemption was FALSE for this lane.
+    One collective runs PER LEAF, so the leaf schedule (optional fields,
+    dtypes, shapes) is rank-local data and is folded into one digest.
     """
-    if jax.process_count() <= 1:
-        return
-    from jax.experimental import multihost_utils
-
-    leaves = [x for x in jax.tree_util.tree_leaves(tree)
-              if hasattr(x, "ndim")]
-    vals = np.array([_content_hash48(np.asarray(x)) for x in leaves],
-                    dtype=np.float64)
-    g = multihost_utils.process_allgather(vals)
-    if not bool(np.all(g == g[0])):
-        bad = [i for i in range(len(leaves))
-               if not bool(np.all(g[:, i] == g[0, i]))]
-        raise RuntimeError(
-            f"{what}: array leaves {bad} differ across processes (48-bit "
-            f"byte digests disagree) — the inputs each process built are "
-            f"NOT identical, which the removed jax device_put assert "
-            f"would have refused. Fix the per-process build before "
-            f"sharding.")
+    names, sizes = _ocean_mesh_axis_terms(mesh)
+    assert_flags_agree(_OCEAN_MESH_ENTRY_FLAGS, (
+        float(mesh is not None),
+        float(mesh.devices.size if mesh is not None else 0),
+        float(len(names)),
+        name_digest48(names),
+        name_digest48(sizes),
+        float(tree is not None),
+        tree_schema_digest48(tree) if tree is not None else FLAG_ABSENT,
+    ), context=where)
 
 
 def shard_state_latlon(state, mesh):
@@ -410,6 +302,11 @@ def shard_state_latlon(state, mesh):
     expects; the test uses it instead of a uniform ``tree.map(P("lat"))`` (which
     fails on ``v`` because ``n_lat+1`` is not divisible by ``N``).
     """
+    # FIRST statement: a DIRECT ``device_put`` of full global arrays onto a
+    # cross-process ``NamedSharding`` is serviced by an ALL-GATHER, and one
+    # runs per leaf -- so both the mesh and the leaf SCHEDULE are rank-local
+    # inputs to a collective (#1362 round 4, blockers 5-6).
+    _agree_ocean_mesh_entry(mesh, state, where="shard_state_latlon")
     # v-carrier contract (see make_sharded_ocean_step's fold note): the TOP
     # v-face row (regular pole wall OR tripole seam/cap row) must be
     # wall-masked — the carrier drops it and reconstructs it as zero, which
@@ -433,7 +330,7 @@ def shard_state_latlon(state, mesh):
         if field is None:
             return None
         sh = NamedSharding(mesh, _lat_spec(field.data))
-        return field.replace(data=_addressable_shard_put(field.data, sh))
+        return field.replace(data=addressable_shard_put(field.data, sh))
 
     def _shard_v(field):
         if field is None:
@@ -451,7 +348,7 @@ def shard_state_latlon(state, mesh):
         nlat1 = field.data.shape[0]
         v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
         sh = NamedSharding(mesh, _lat_spec(v_lower))
-        return field.replace(data=_addressable_shard_put(v_lower, sh))
+        return field.replace(data=addressable_shard_put(v_lower, sh))
 
     updates = {}
     for name in _V_STAGGERED_STATE_FIELDS:
@@ -468,12 +365,12 @@ def shard_state_latlon(state, mesh):
             updates[name] = _shard_cell(val)
         else:
             # Non-Field, non-None leaf (e.g. a raw array carry like psi).
-            # ndim>=1 lat-shards via _lat_spec (1-D included); only true
-            # scalars replicate (codex r17: the old "replicate lower-rank"
-            # wording did not match _lat_spec's behaviour).
+            # Shard 2-D+ on lat, replicate lower-rank — matches shard_pytree.
             arr = jnp.asarray(val)
             spec = _lat_spec(arr) if arr.ndim >= 1 else P()
-            updates[name] = _addressable_shard_put(arr, NamedSharding(mesh, spec))
+            # ndim>=1 lat-shards via _lat_spec (1-D included); only true
+            # scalars replicate.
+            updates[name] = addressable_shard_put(arr, NamedSharding(mesh, spec))
     return state._replace(**updates)
 
 
@@ -488,6 +385,10 @@ def shard_forcing_latlon(forcing, mesh):
     untouched (they vanish from the pytree structure, matching the specs the
     step derives).  ``forcing=None`` returns ``None``.
     """
+    # FIRST statement: the `forcing is None or mesh is None` return below
+    # SKIPS every per-leaf put, so a rank with no forcing would leave a peer
+    # blocked in one (#1362 round 4, blocker 6).
+    _agree_ocean_mesh_entry(mesh, forcing, where="shard_forcing_latlon")
     if forcing is None or mesh is None:
         return forcing
     assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")
@@ -496,7 +397,7 @@ def shard_forcing_latlon(forcing, mesh):
         if leaf is None:
             return None
         arr = jnp.asarray(leaf)
-        return _addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
+        return addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
 
     return jax.tree.map(_put, forcing)
 
@@ -526,6 +427,10 @@ def shard_forcing_stack_latlon(stack, mesh):
     stack must be laid out consistently with the state the sharded step
     carries, and a second copy would drift.
     """
+    # FIRST statement: same rank-local early-return + per-leaf put as
+    # shard_forcing_latlon (#1362 round 4, blocker 6).
+    _agree_ocean_mesh_entry(mesh, stack,
+                            where="shard_forcing_stack_latlon")
     if mesh is None:
         return stack
 
@@ -541,7 +446,7 @@ def shard_forcing_stack_latlon(stack, mesh):
             spec = P("lat", None)
         else:
             spec = P()
-        return _addressable_shard_put(arr, NamedSharding(mesh, spec))
+        return addressable_shard_put(arr, NamedSharding(mesh, spec))
 
     return jax.tree.map(_put, stack)
 
@@ -579,6 +484,15 @@ def gather_state_latlon(state, mesh):
     the primitive shared with the atm gather); the single-process path is
     byte-unchanged.
     """
+    # FIRST statement (#1362 round 4). Two rank-local hazards live below:
+    # ``replicate_leaf`` runs a real cross-process collective (a jit identity
+    # with replicated out_shardings; XLA inserts the all-gather), and the
+    # ``if ... is None: continue`` skips mean a state whose OPTIONAL fields
+    # differ across processes performs a DIFFERENT NUMBER of those
+    # collectives -- one rank finishing while a peer still waits. So agree the
+    # mesh AND the ordered list of fields that will actually be gathered,
+    # before the first one runs.
+    _agree_ocean_mesh_entry(mesh, state, where="gather_state_latlon")
     from legoesm.parallel.latlon_spmd import replicate_leaf
 
     rep = NamedSharding(mesh, P())
@@ -609,6 +523,122 @@ def gather_state_latlon(state, mesh):
     return state._replace(**updates)
 
 
+# Ordered flag names for the ocean SPMD entry gate. STATIC tuple: the payload
+# width is fixed by this literal, never by rank-local data.
+_OCEAN_SPMD_ENTRY_FLAGS = (
+    "has_mesh", "n_dev", "n_axes", "axis_names", "axis_sizes",
+    "grid_n_lat", "grid_n_lon", "geom_schema", "fold_active",
+    "config_digest", "has_vertex_mask",
+)
+
+
+def _agree_ocean_spmd_entry(model, mesh, *, where: str) -> None:
+    """Agree every rank-local input, as the FIRST statement of a public factory.
+
+    #1362 / codex round 2, ocean twin of the atmosphere's
+    ``_agree_spmd_entry``.  This lane has the same shape of hazard: the
+    ``mesh is None`` early return, ``build_band_grids``' divisibility
+    validation, and ``_build_band_vertex_masks`` (which throws if only THIS
+    rank lacks a primed vertex-mask cache) all execute BEFORE the schema
+    collective.  Any of them lets one process raise or return while a peer
+    blocks in ``process_allgather`` -- a HANG rather than an error.
+
+    Agreeing the mesh shape, grid dimensions and fold state up front makes
+    every downstream rank-local check symmetric by construction.
+
+    ``axis_names`` carries the ORDERED axis-name digest, not just the axis
+    COUNT: the band body indexes ``mesh.axis_names[0]``, so two processes
+    whose meshes name that axis differently would psum/ppermute over
+    different axes while every count-based flag agreed (the ocean instance of
+    codex round-3 blocker 2, which was found on the atmosphere twin --
+    fixing only the lane where a defect was reported is what left five
+    unguarded paths after round 1).
+
+    Grid dimensions go through :func:`coerce_count`, which NEVER raises, and
+    the refusal is deferred until AFTER the collective; building a collective
+    payload must not be able to kill one rank while its peers block in the
+    gather (codex round-3 blocker 3, same rationale as the atm twin).
+    """
+    # Defensive attribute reads: nothing in the payload build may raise before
+    # the collective (see the atm twin for the full rule).
+    grid = getattr(model, "grid", None)
+    fold = getattr(grid, "fold", None)
+    names, sizes = _ocean_mesh_axis_terms(mesh)
+    problems = []
+
+    def _count(value, label, absent=FLAG_ABSENT):
+        payload, problem = coerce_count(value, absent=absent)
+        if problem is not None:
+            problems.append((label, problem))
+        return payload
+
+    flags = (
+        float(mesh is not None),
+        float(mesh.devices.size if mesh is not None else 0),
+        float(len(names)),
+        name_digest48(names),
+        name_digest48(sizes),
+        _count(getattr(grid, "n_lat", None), "grid.n_lat", absent=0.0),
+        _count(getattr(grid, "n_lon", None), "grid.n_lon", absent=0.0),
+        # The geometry array fields that `build_band_grids` slices and
+        # `_replicated_put` broadcasts, by dtype + shape.
+        tree_schema_digest48(grid),
+        float(bool(fold is not None and getattr(fold, "is_active", False))),
+        # ONE digest over EVERY static scalar of the ocean config instead of a
+        # hand-picked few: the step body branches on `outer_integrator`, the
+        # tracer integrator, the polar filter, the freeze floor and the EW
+        # overlap, and NONE of them were agreed (codex round-4, blocker 3).
+        # A valid/invalid or euler/ab2 split makes one rank raise during
+        # tracing while its peer compiles a different program.
+        config_digest48(getattr(model, "config", None)),
+        # `_build_band_vertex_masks` RAISES when this cache is unprimed, and
+        # it runs before the schema collective -- so its presence must be
+        # agreed first or an unprimed rank dies while its peer blocks
+        # (codex round-4, blocker 4).
+        float(getattr(model, "_vertex_mask", None) is not None),
+    )
+    assert_flags_agree(_OCEAN_SPMD_ENTRY_FLAGS, flags, context=where)
+    # AFTER the collective only: symmetric on every rank (see atm twin).
+    for label, problem in problems:
+        raise ValueError(f"{where}: {label} {problem}")
+
+
+# Ordered flag names for the PER-INVOCATION gate on the returned ocean SPMD
+# callable. STATIC tuple: fixed width, never rank-local.
+_OCEAN_CALL_ENTRY_FLAGS = (
+    "has_mesh", "n_dev", "axis_names", "axis_sizes",
+    "state_schema", "has_forcing", "forcing_schema",
+)
+
+
+def _agree_ocean_spmd_call(mesh, state, forcing, *, where: str) -> None:
+    """Agree a returned ocean SPMD callable's per-CALL inputs, FIRST statement.
+
+    #1362 round 4, blocker 7.  ``sharded_step`` runs ``_validate_forcing_layout``
+    and builds a rank-local cache key BEFORE entering its ``shard_map``: a
+    forcing layout that is invalid on one rank only makes that rank raise while
+    its peers enter the collective program -- a hang.  The state + forcing leaf
+    SCHEMA is agreed too, because ``in_specs``/``out_specs`` are derived from
+    it, so two processes with different optional fields compile different
+    programs.
+
+    Cost: one small allgather per CALL, and an exact no-op under a single
+    process.  See the atmosphere twin ``_agree_spmd_call`` for why gating only
+    on cache misses is NOT a valid optimisation.
+    """
+    names, sizes = _ocean_mesh_axis_terms(mesh)
+    assert_flags_agree(_OCEAN_CALL_ENTRY_FLAGS, (
+        float(mesh is not None),
+        float(mesh.devices.size if mesh is not None else 0),
+        name_digest48(names),
+        name_digest48(sizes),
+        tree_schema_digest48(state),
+        float(forcing is not None),
+        (tree_schema_digest48(forcing) if forcing is not None
+         else FLAG_ABSENT),
+    ), context=where)
+
+
 def make_sharded_ocean_step(model, mesh):
     """Return ``step(state, dt, freshwater=None, surface_forcing=None,
     sponge=None, t_seconds=None) -> state`` running ``model.step``
@@ -634,9 +664,9 @@ def make_sharded_ocean_step(model, mesh):
     Notes
     -----
     Build the ``N`` band geometries + band vertex masks host-side, stack their
-    ARRAY fields over a leading band axis SHARDED ``P("lat")`` (each device
-    holds only its own slab; the ``shard_map`` body reads it at ``[0]``; the
-    geometry SCALAR fields stay static — see the module docstring).  ``dt`` is a TRACED,
+    ARRAY fields into a replicated pytree, and index by
+    ``jax.lax.axis_index("lat")`` in the ``shard_map`` body (the geometry SCALAR
+    fields stay static — see the module docstring).  ``dt`` is a TRACED,
     replicated operand and the jitted ``shard_map`` is built once and cached
     (a per-call rebuild re-traced the whole ocean step every call — see the
     ``_cache`` note below).  ``check_vma=False`` (the JAX >= 0.8
@@ -648,6 +678,9 @@ def make_sharded_ocean_step(model, mesh):
     each band's ``nl+1`` v-faces, runs the step on the band geometry, and
     converts the result back to the ``v_lower`` representation.
     """
+    # FIRST statement: agree every rank-local input before ANY
+    # rank-local check can raise or return (codex round-2).
+    _agree_ocean_spmd_entry(model, mesh, where="make_sharded_ocean_step")
     if mesh is None:                   # single-device: plain step
         return lambda state, dt, **forcing_kwargs: model.step(
             state, dt, **forcing_kwargs)
@@ -671,7 +704,7 @@ def make_sharded_ocean_step(model, mesh):
     # (unmasked) seam v-row refuses loudly there instead of silently
     # reconstructing zeros here.
 
-    # --- host-side band geometries + vertex masks (band-stacked, P("lat")-sharded) ---
+    # --- host-side band geometries + vertex masks (replicated, indexed in-body) ---
     band_grids = build_band_grids(model.grid, n_dev)
     band_vmasks = _build_band_vertex_masks(model, n_dev)
     template = band_grids[0]           # static-scalar source (uniform bands)
@@ -684,100 +717,45 @@ def make_sharded_ocean_step(model, mesh):
     # per-device residency left after the host-side-build fix (probe
     # 26524423). The leading axis has length n_dev, so P("lat") divides it
     # exactly; the body indexes its local slab at [0]. Values are unchanged
-    # — same stack, different placement; the cross-process divergence
-    # guard below runs on HOST values and is placement-blind.
+    # — same stack, different placement; the process-0 broadcast +
+    # divergence guard below runs on HOST values and is placement-blind.
     rep = NamedSharding(mesh, P("lat"))
 
     def _replicated_put(arr, name):
-        # (Name kept for history; since 2026-08-03 this is a SHARDED stack
-        # put.) The band-geometry arrays are (re)computed per process and
-        # can differ in their last ULPs (per-process XLA autotuning on
-        # device-derived grid fields) — the fully-replicated-put era
-        # broadcast process 0's bytes to sidestep the P() bit-identity
-        # assert (job 26450848). With the #1370-iii P("lat") sharding each
-        # process's devices consume ONLY its own band rows, so the
-        # broadcast became both unnecessary and, at nd>=96, fatal (its
-        # psum program is nd x the stack — see the note at the put below).
-        # GUARD (codex round-3): process 0 must not silently mask REAL
-        # cross-process divergence. Compare an allgathered fingerprint:
-        # structural entries exactly; value entries EXACTLY for integer/bool
-        # arrays (masks are comparison results — bit-reproducible, and an
-        # exact compare is the only way to catch a two-cell flip that cancels
-        # in the sum, codex round-4) and to rtol 1e-5 for float arrays (only
-        # ULP autotune drift is expected there; quantize-then-assert-equal
-        # false-positived on a rounding boundary, job 26453240).
-        # NO DEADLOCK RISK: every process fingerprints the same fields in the
-        # same order and derives the verdict from the SAME gathered array, so
-        # the refusal is symmetric — all raise or none.
-        # Residual (documented): a float-geometry divergence preserving sum,
-        # sum-of-squares AND absmax to 1e-5 is not detected; band grids are
-        # analytic in lat/lon, so any real inconsistency moves those moments.
-        host = np.asarray(arr)
-        if jax.process_count() > 1:
-            from jax.experimental import multihost_utils
-
-            # PER-BAND fingerprints (module-level, unit-tested): exact
-            # dtypes hash positionally per band; floats compare per-band
-            # moments to rtol 1e-5 — bounds each band's drift instead of
-            # letting it hide in a whole-array sum, since each process's
-            # own bytes are now the live inputs for the bands it owns
-            # (codex r14). A mask that genuinely differs across processes
-            # means different wet domains = different physics: refusing is
-            # correct, not a false alarm.
-            struct, vals, is_exact = geom_band_fingerprint(
-                host, host.shape[0])
-            g_struct = multihost_utils.process_allgather(struct)
-            g_vals = multihost_utils.process_allgather(vals)
-            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
-                raise RuntimeError(
-                    f"make_sharded_ocean_step: band-geometry field {name!r} "
-                    f"DIVERGES across processes (exact_dtype={is_exact}, "
-                    f"gathered={g_vals.tolist()}) — a real config/grid "
-                    f"inconsistency, not autotune noise; refusing to "
-                    f"shard it.")
-            # NO broadcast_one_to_all here (removed 2026-08-03): its psum
-            # program is [n_processes, stack] in / P() fully-replicated out,
-            # so its logical arg bytes are nd x the global stack — 82.4 GB
-            # at nd=96 and 109.6 GB at nd=128 for one 3-D field, the
-            # near-linear-in-nd wall that killed oc @96/@128 (jobs
-            # 26642771/26636762) while @64 sat just under XLA's 63.8 GB
-            # limit.  The target sharding is P("lat"): each process's
-            # devices consume ONLY its own band rows, so cross-process
-            # byte-identity of non-owned rows is irrelevant, and REAL
-            # divergence is already refused by the fingerprint gate above.
-            # make_array_from_callback hands each process exactly its
-            # addressable slabs — the same #1100 pattern as the state
-            # build — with no global-sized collective program at all.
-        host_np = np.asarray(host)
-        return jax.make_array_from_callback(
-            host_np.shape, rep, lambda idx: host_np[idx])
-
-    if jax.process_count() > 1:
-        # Schema gate FIRST (one fixed-shape collective every process
-        # reaches): a process-dependent field list or a mixed
-        # jax_enable_x64 setting would otherwise desynchronize the
-        # per-field gathers below instead of failing with a clear message.
-        from jax.experimental import multihost_utils as _mhu
-
-        _g = _mhu.process_allgather(
-            _schema_fingerprint(list(array_field_names), n_dev))
-        if not bool(np.all(_g == _g[0])):
-            raise RuntimeError(
-                "make_sharded_ocean_step: the band-geometry SCHEMA differs "
-                "across processes (field list / x64 setting / device count "
-                f"— gathered {_g.tolist()}). Fix the per-process config "
-                "before sharding; the per-field checks below assume one "
-                "schema.")
-
-    geom_stacks = {
-        name: _replicated_put(
-            jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
-                      axis=0), name)
+        # (Name kept for history; this is a SHARDED P("lat") stack put.)
+        # checked_shard_put replaces the broadcast_checked+device_put pair:
+        # the broadcast's psum program is [n_processes, stack] (nd x 849 MB
+        # at LL2304 — the @96/@128 wall), and a numpy device_put onto an
+        # all-process sharding pays jax's whole-array assert_equal on top.
+        # The per-band gate keeps the divergence contract (n_bands is
+        # schema-gated just below, so payload widths agree). ONE shared
+        # implementation: legoesm.parallel.geometry_consistency.
+        return checked_shard_put(
+            arr, name, rep, context="make_sharded_ocean_step",
+            n_bands=n_dev)
+
+    # Schema gate FIRST (one fixed-shape collective every process reaches):
+    # a process-dependent field list or a mixed jax_enable_x64 setting would
+    # otherwise desynchronize the per-field gathers below instead of failing
+    # with a clear message.
+    # Build the raw stacks FIRST so the schema gate can also cover each
+    # field's dtype class and ndim -- those decide the per-field payload
+    # shape below, so a bool-vs-float disagreement must fail HERE rather than
+    # deadlock in the per-field gather.
+    _raw_geom = {
+        name: jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
+                        axis=0)
         for name in array_field_names
     }
-    vmask_stack = _replicated_put(
-        jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0),
-        "vertex_mask")
+    _raw_vmask = jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0)
+    _gate_names = [*array_field_names, "vertex_mask"]
+    assert_schema_agrees(
+        _gate_names, n_dev, context="make_sharded_ocean_step",
+        arrays=[*(_raw_geom[n] for n in array_field_names), _raw_vmask])
+
+    geom_stacks = {name: _replicated_put(_raw_geom[name], name)
+                   for name in array_field_names}
+    vmask_stack = _replicated_put(_raw_vmask, "vertex_mask")
 
     # Static perms for the v north-boundary-row ppermute (band r receives band
     # r+1's v_lower[0] = global v[e]; north band non-target receives 0).
@@ -893,20 +871,19 @@ def make_sharded_ocean_step(model, mesh):
 
     def sharded_step(state, dt, freshwater=None, surface_forcing=None,
                      sponge=None, t_seconds=None, aux=None):
-        # ``aux`` (codex/2-proc repro 2026-08-03): the geometry + vmask
-        # stacks are SHARDED global arrays; when this wrapper runs INSIDE
-        # an outer trace (a bench/driver ``jit``/``scan`` over the step —
-        # jit-of-jit inlines the inner call), concrete closure arrays
-        # become OUTER-TRACE CONSTANTS and jax's MLIR constant handler
-        # tries to fetch their value — impossible for non-addressable
-        # arrays (RuntimeError: 'Fetching value ... non-addressable'; the
-        # multicontroller lane has been broken this way since the
-        # #1370-iii stack sharding). Callers that wrap the step in their
-        # own jit MUST thread ``step.aux`` through their jit boundary as
-        # an ARGUMENT and pass it back here.
+        # ``aux``: the sharded geometry+vmask stacks. When this wrapper runs
+        # INSIDE an outer trace (a bench/driver jit/scan — jit-of-jit
+        # inlines the inner call), concrete closure arrays become
+        # OUTER-trace constants whose value the MLIR handler cannot fetch
+        # for non-addressable arrays (broken since #1370-iii sharded the
+        # stacks). Outer-jit callers MUST thread ``step.aux`` through their
+        # jit boundary as an ARGUMENT and pass it back here.
         # ONE forcing operand: None fields drop out of the pytree structure,
         # so specs derived by tree.map skip them automatically and the
         # structure key below distinguishes every None<->array combination.
+        _agree_ocean_spmd_call(
+            mesh, state, (freshwater, surface_forcing, sponge, t_seconds),
+            where="make_sharded_ocean_step.step")
         forcing = (freshwater, surface_forcing, sponge, t_seconds)
         _validate_forcing_layout((freshwater, surface_forcing, sponge))
         # Cache key = the state's AND forcing's pytree STRUCTURE, plus the
@@ -1002,6 +979,9 @@ def make_sharded_ocean_step_global(model, mesh):
 
     ``mesh is None`` ⇒ the plain single-device ``model.step`` (no scatter/gather).
     """
+    # FIRST statement: agree every rank-local input before ANY
+    # rank-local check can raise or return (codex round-2).
+    _agree_ocean_spmd_entry(model, mesh, where="make_sharded_ocean_step_global")
     if mesh is None:                   # single-device: plain step
         return lambda state, dt, surface_forcing=None, freshwater=None: (
             model.step(state, dt, freshwater=freshwater,
@@ -1010,11 +990,13 @@ def make_sharded_ocean_step_global(model, mesh):
     inner = make_sharded_ocean_step(model, mesh)
 
     def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
-        # Scatter the global state AND forcing to the band layout explicitly
-        # (codex r17 item 3: the old comment claimed inner sharded the
-        # forcing; it forwarded it global and relied on implicit JIT input
-        # placement — which under multicontroller pays jax's whole-array
-        # device_put assert, the nd-linear wall this module removes).
+        _agree_ocean_spmd_call(mesh, state, (surface_forcing, freshwater),
+                               where="make_sharded_ocean_step_global.step")
+        # Scatter the global state AND forcing to the band layout
+        # explicitly (the old comment claimed inner sharded the forcing;
+        # it forwarded it global and relied on implicit JIT input
+        # placement — jax's whole-array device_put assert under
+        # multicontroller, the nd-linear wall this module removes).
         ss = shard_state_latlon(state, mesh)
         ss = inner(ss, dt,
                    surface_forcing=shard_forcing_latlon(surface_forcing, mesh),
diff --git a/tests/ocean/unit/test_sharded_geom_fingerprint.py b/tests/ocean/unit/test_sharded_geom_fingerprint.py
index 2f61c0709..5b24a4b9e 100644
--- a/tests/ocean/unit/test_sharded_geom_fingerprint.py
+++ b/tests/ocean/unit/test_sharded_geom_fingerprint.py
@@ -9,9 +9,9 @@ multicontroller allgather wiring is exercised by the distributed suite).
 import numpy as np
 import pytest
 
-from legoesm.ocean.dynamics.sharded_ocean_step import (
+from legoesm.parallel.geometry_consistency import (
+    band_fingerprint as geom_band_fingerprint,
     band_fingerprints_agree,
-    geom_band_fingerprint,
 )
 
 N_BANDS = 4
@@ -99,6 +99,6 @@ def test_shape_mismatch_refused():
     assert fa[0].shape != fb[0].shape or not np.array_equal(fa[0], fb[0])
 
 
-def test_wrong_leading_axis_asserts():
-    with pytest.raises(AssertionError):
+def test_wrong_leading_axis_raises():
+    with pytest.raises(ValueError):
         geom_band_fingerprint(np.ones((3, 2)), N_BANDS)

-- callers --
packages/core/legoesm/parallel/geometry_consistency.py-519-            f"assume one schema.")
packages/core/legoesm/parallel/geometry_consistency.py-520-
packages/core/legoesm/parallel/geometry_consistency.py-521-
packages/core/legoesm/parallel/geometry_consistency.py:522:def broadcast_checked(arr, name: str, *, context: str) -> np.ndarray:
packages/core/legoesm/parallel/geometry_consistency.py-523-    """Verify ``arr`` agrees across processes, then broadcast process 0's bytes.
packages/core/legoesm/parallel/geometry_consistency.py-524-
packages/core/legoesm/parallel/geometry_consistency.py-525-    Multi-process: returns a host ``np.ndarray`` that is bit-identical on
--
packages/coupler/legoesm/driver/sharded_operator_split_step.py-480-                         arrays=[raw[n] for n in _ordered])
packages/coupler/legoesm/driver/sharded_operator_split_step.py-481-    stacks = {
packages/coupler/legoesm/driver/sharded_operator_split_step.py-482-        name: jax.device_put(
packages/coupler/legoesm/driver/sharded_operator_split_step.py:483:            jnp.asarray(broadcast_checked(
packages/coupler/legoesm/driver/sharded_operator_split_step.py-484-                raw[name], name,
packages/coupler/legoesm/driver/sharded_operator_split_step.py-485-                context="make_sharded_operator_split_step")),
packages/coupler/legoesm/driver/sharded_operator_split_step.py-486-            rep)
--
tests/distributed/test_geometry_consistency_mp.py-96-
tests/distributed/test_geometry_consistency_mp.py-97-    assert_schema_agrees(("lat", "lon"), 2, context="mp-test")
tests/distributed/test_geometry_consistency_mp.py-98-    assert_flags_agree(("alpha", "beta"), (1.0, 0.0), context="mp-test")
tests/distributed/test_geometry_consistency_mp.py:99:    out = broadcast_checked(np.arange(4, dtype=np.float64), "field", context="mp-test")
tests/distributed/test_geometry_consistency_mp.py-100-    np.testing.assert_allclose(out, np.arange(4, dtype=np.float64))
tests/distributed/test_geometry_consistency_mp.py-101-
tests/distributed/test_geometry_consistency_mp.py-102-
--
tests/distributed/test_geometry_consistency_mp.py-129-    _, _, broadcast_checked = _guards()
tests/distributed/test_geometry_consistency_mp.py-130-    arr = np.arange(4, dtype=np.float64) + (0.0 if _rank() == 0 else 1.0)
tests/distributed/test_geometry_consistency_mp.py-131-    with pytest.raises(RuntimeError, match="DIVERGES"):
tests/distributed/test_geometry_consistency_mp.py:132:        broadcast_checked(arr, "field", context="mp-test")
tests/distributed/test_geometry_consistency_mp.py-133-
tests/distributed/test_geometry_consistency_mp.py-134-
tests/distributed/test_geometry_consistency_mp.py-135-def test_guards_still_agree_after_a_divergence_was_raised():
--
tests/distributed/test_geometry_consistency_mp.py-140-    than on the passing path, this is where the ranks would desynchronise.
tests/distributed/test_geometry_consistency_mp.py-141-    """
tests/distributed/test_geometry_consistency_mp.py-142-    assert_schema_agrees, _, broadcast_checked = _guards()
tests/distributed/test_geometry_consistency_mp.py:143:    out = broadcast_checked(np.full(3, 2.5, dtype=np.float64), "after", context="mp-test")
tests/distributed/test_geometry_consistency_mp.py-144-    np.testing.assert_allclose(out, 2.5)
tests/distributed/test_geometry_consistency_mp.py-145-    assert_schema_agrees(("lat", "lon"), 2, context="mp-test")
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-489-                         context="make_sharded_atm_latlon_step",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-490-                         arrays=[raw[n] for n in ordered_names])
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-491-    raw = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:492:        name: jnp.asarray(broadcast_checked(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-493-            raw[name], name, context="make_sharded_atm_latlon_step"))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-494-        for name in ordered_names
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-495-    }
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1583-                         context="make_sharded_atm_latlon_step_2d",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1584-                         arrays=[raw[n] for n in ordered_names])
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1585-    raw = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1586:        name: jnp.asarray(broadcast_checked(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1587-            raw[name], name, context="make_sharded_atm_latlon_step_2d"))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1588-        for name in ordered_names
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1589-    }
--
tests/unit/test_geometry_consistency.py-143-
tests/unit/test_geometry_consistency.py-144-    def test_broadcast_checked_returns_the_SAME_OBJECT(self):
tests/unit/test_geometry_consistency.py-145-        a = np.linspace(0.0, 1.0, 40).reshape(5, 8)
tests/unit/test_geometry_consistency.py:146:        assert broadcast_checked(a, "area", context="test") is a
tests/unit/test_geometry_consistency.py-147-
tests/unit/test_geometry_consistency.py-148-    def test_single_process_does_NOT_convert_a_device_array_to_host(self):
tests/unit/test_geometry_consistency.py-149-        """codex 2026-07-29 (major 3): converting unconditionally forced a
--
tests/unit/test_geometry_consistency.py-152-        straight in, so it must come back as the SAME jax array."""
tests/unit/test_geometry_consistency.py-153-        jnp = pytest.importorskip("jax.numpy")
tests/unit/test_geometry_consistency.py-154-        a = jnp.arange(6, dtype=jnp.int32)
tests/unit/test_geometry_consistency.py:155:        out = broadcast_checked(a, "idx", context="test")
tests/unit/test_geometry_consistency.py-156-        assert out is a
tests/unit/test_geometry_consistency.py-157-        assert not isinstance(out, np.ndarray)
tests/unit/test_geometry_consistency.py-158-
--
tests/unit/test_geometry_consistency.py-224-    def test_exact_array_payload_is_the_byte_digest(self, multiproc):
tests/unit/test_geometry_consistency.py-225-        fake = multiproc()
tests/unit/test_geometry_consistency.py-226-        a = np.arange(24, dtype=np.int32).reshape(4, 6)
tests/unit/test_geometry_consistency.py:227:        broadcast_checked(a, "idx", context="t")
tests/unit/test_geometry_consistency.py-228-        vals = fake.seen[-1]
tests/unit/test_geometry_consistency.py-229-        assert vals[0] == content_hash48(a), (
tests/unit/test_geometry_consistency.py-230-            "exact arrays must be fingerprinted by the POSITIONAL byte "
--
tests/unit/test_geometry_consistency.py-233-    def test_float_array_payload_is_the_moment_triple(self, multiproc):
tests/unit/test_geometry_consistency.py-234-        fake = multiproc()
tests/unit/test_geometry_consistency.py-235-        a = np.linspace(-2.0, 3.0, 30).reshape(5, 6)
tests/unit/test_geometry_consistency.py:236:        broadcast_checked(a, "area", context="t")
tests/unit/test_geometry_consistency.py-237-        vals = fake.seen[-1]
tests/unit/test_geometry_consistency.py-238-        np.testing.assert_allclose(
tests/unit/test_geometry_consistency.py-239-            vals[:3], [a.sum(), (a * a).sum(), np.abs(a).max()], rtol=1e-12)
--
tests/unit/test_geometry_consistency.py-256-        multiproc(override=lambda row: (
tests/unit/test_geometry_consistency.py-257-            _make_vals_for(perm) if row.shape == (3,) else None))
tests/unit/test_geometry_consistency.py-258-        with pytest.raises(RuntimeError, match="DIVERGES"):
tests/unit/test_geometry_consistency.py:259:            broadcast_checked(a, "wet_mask", context="t")
tests/unit/test_geometry_consistency.py-260-
tests/unit/test_geometry_consistency.py-261-    def test_agreeing_processes_pass(self, multiproc):
tests/unit/test_geometry_consistency.py-262-        multiproc()
tests/unit/test_geometry_consistency.py-263-        a = np.linspace(0, 1, 12)
tests/unit/test_geometry_consistency.py:264:        out = broadcast_checked(a, "area", context="t")
tests/unit/test_geometry_consistency.py-265-        np.testing.assert_array_equal(out, a)
tests/unit/test_geometry_consistency.py-266-
tests/unit/test_geometry_consistency.py-267-    def test_float_divergence_beyond_rtol_raises(self, multiproc):
--
tests/unit/test_geometry_consistency.py-269-        struct mismatch instead."""
tests/unit/test_geometry_consistency.py-270-        multiproc(override=lambda row: row * 1.1 if row.shape == (3,) else None)
tests/unit/test_geometry_consistency.py-271-        with pytest.raises(RuntimeError, match="DIVERGES"):
tests/unit/test_geometry_consistency.py:272:            broadcast_checked(np.linspace(1, 2, 10), "area", context="t")
tests/unit/test_geometry_consistency.py-273-
tests/unit/test_geometry_consistency.py-274-    def test_float_drift_within_rtol_is_ACCEPTED(self, multiproc):
tests/unit/test_geometry_consistency.py-275-        """The whole reason for the broadcast: ULP-scale autotune drift must
tests/unit/test_geometry_consistency.py-276-        NOT raise, or every large multi-process run fails spuriously."""
tests/unit/test_geometry_consistency.py-277-        multiproc(override=lambda row:
tests/unit/test_geometry_consistency.py-278-                  row * (1.0 + 1e-9) if row.shape == (3,) else None)
tests/unit/test_geometry_consistency.py:279:        out = broadcast_checked(np.linspace(1, 2, 10), "area", context="t")
tests/unit/test_geometry_consistency.py-280-        assert out is not None
tests/unit/test_geometry_consistency.py-281-
tests/unit/test_geometry_consistency.py-282-    def test_nonfinite_count_is_structural(self, multiproc):
tests/unit/test_geometry_consistency.py-283-        """A NaN on one process only must not be averaged away."""
tests/unit/test_geometry_consistency.py-284-        fake = multiproc()
tests/unit/test_geometry_consistency.py-285-        a = np.array([1.0, 2.0, np.nan, 4.0])
tests/unit/test_geometry_consistency.py:286:        broadcast_checked(a, "area", context="t")
tests/unit/test_geometry_consistency.py-287-        struct = fake.seen[-2]
tests/unit/test_geometry_consistency.py-288-        assert struct[5] == 1.0, "non-finite count must be carried in struct"
tests/unit/test_geometry_consistency.py-289-
--
tests/unit/test_geometry_consistency.py-310-    ])
tests/unit/test_geometry_consistency.py-311-    def test_every_payload_has_identical_shape(self, multiproc, arr):
tests/unit/test_geometry_consistency.py-312-        fake = multiproc()
tests/unit/test_geometry_consistency.py:313:        broadcast_checked(arr, "f", context="t")
tests/unit/test_geometry_consistency.py-314-        struct, vals = fake.seen[-2], fake.seen[-1]
tests/unit/test_geometry_consistency.py-315-        assert struct.shape == (8,), struct.shape
tests/unit/test_geometry_consistency.py-316-        assert vals.shape == (3,), vals.shape
--
tests/unit/test_geometry_consistency.py-1436-                          for c in ast.walk(p))]
tests/unit/test_geometry_consistency.py-1437-        assert guarded, (
tests/unit/test_geometry_consistency.py-1438-            "every replicated geometry device_put must take a "
tests/unit/test_geometry_consistency.py:1439:            "broadcast_checked(...) value as its argument")

-- fingerprints --
tests/ocean/unit/test_sharded_geom_fingerprint.py:1:"""Unit tests for the band-geometry cross-process fingerprint gate.
tests/ocean/unit/test_sharded_geom_fingerprint.py-2-
tests/ocean/unit/test_sharded_geom_fingerprint.py-3-The gate decides whether ``make_sharded_ocean_step`` accepts per-process
--
tests/ocean/unit/test_sharded_geom_fingerprint.py-11-
tests/ocean/unit/test_sharded_geom_fingerprint.py-12-from legoesm.parallel.geometry_consistency import (
tests/ocean/unit/test_sharded_geom_fingerprint.py:13:    band_fingerprint as geom_band_fingerprint,
tests/ocean/unit/test_sharded_geom_fingerprint.py:14:    band_fingerprints_agree,
tests/ocean/unit/test_sharded_geom_fingerprint.py-15-)
tests/ocean/unit/test_sharded_geom_fingerprint.py-16-
--
tests/ocean/unit/test_sharded_geom_fingerprint.py-20-
tests/ocean/unit/test_sharded_geom_fingerprint.py-21-def _gather(*hosts):
tests/ocean/unit/test_sharded_geom_fingerprint.py:22:    """Simulate process_allgather over per-process fingerprints."""
tests/ocean/unit/test_sharded_geom_fingerprint.py:23:    fps = [geom_band_fingerprint(h, N_BANDS) for h in hosts]
tests/ocean/unit/test_sharded_geom_fingerprint.py-24-    exacts = {fp[2] for fp in fps}
tests/ocean/unit/test_sharded_geom_fingerprint.py-25-    assert len(exacts) == 1
--
tests/ocean/unit/test_sharded_geom_fingerprint.py-32-    rng = np.random.default_rng(0)
tests/ocean/unit/test_sharded_geom_fingerprint.py-33-    a = rng.normal(size=SHAPE).astype(np.float32)
tests/ocean/unit/test_sharded_geom_fingerprint.py:34:    assert band_fingerprints_agree(*_gather(a, a.copy()))
tests/ocean/unit/test_sharded_geom_fingerprint.py-35-
tests/ocean/unit/test_sharded_geom_fingerprint.py-36-
--
tests/ocean/unit/test_sharded_geom_fingerprint.py-39-    a = rng.normal(size=SHAPE).astype(np.float64) + 10.0
tests/ocean/unit/test_sharded_geom_fingerprint.py-40-    b = np.nextafter(a, np.inf)  # a TRUE 1-ULP elementwise drift
tests/ocean/unit/test_sharded_geom_fingerprint.py:41:    assert band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py-42-
tests/ocean/unit/test_sharded_geom_fingerprint.py-43-
--
tests/ocean/unit/test_sharded_geom_fingerprint.py-61-                       rtol=1e-5, atol=0.0)
tests/ocean/unit/test_sharded_geom_fingerprint.py-62-    # ...the per-band gate refuses it.
tests/ocean/unit/test_sharded_geom_fingerprint.py:63:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py-64-
tests/ocean/unit/test_sharded_geom_fingerprint.py-65-
tests/ocean/unit/test_sharded_geom_fingerprint.py-66-def test_exact_dtype_permutation_refused():
tests/ocean/unit/test_sharded_geom_fingerprint.py:67:    # Moment fingerprints are blind to permutations; the positional
tests/ocean/unit/test_sharded_geom_fingerprint.py-68-    # per-band byte digest must not be.
tests/ocean/unit/test_sharded_geom_fingerprint.py-69-    a = np.zeros(SHAPE, dtype=np.int32)
--
tests/ocean/unit/test_sharded_geom_fingerprint.py-71-    b = np.zeros_like(a)
tests/ocean/unit/test_sharded_geom_fingerprint.py-72-    b[1, 3, 2] = 1  # same count, different position, same band
tests/ocean/unit/test_sharded_geom_fingerprint.py:73:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py-74-
tests/ocean/unit/test_sharded_geom_fingerprint.py-75-
--
tests/ocean/unit/test_sharded_geom_fingerprint.py-80-    b[0, 0, 0] = False
tests/ocean/unit/test_sharded_geom_fingerprint.py-81-    b[0, 5, 7] = True  # true-count preserved
tests/ocean/unit/test_sharded_geom_fingerprint.py:82:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py-83-
tests/ocean/unit/test_sharded_geom_fingerprint.py-84-
--
tests/ocean/unit/test_sharded_geom_fingerprint.py-87-    b = a.copy()
tests/ocean/unit/test_sharded_geom_fingerprint.py-88-    b[3, 0, 0] = np.nan  # struct carries per-band non-finite counts
tests/ocean/unit/test_sharded_geom_fingerprint.py:89:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py-90-
tests/ocean/unit/test_sharded_geom_fingerprint.py-91-
--
tests/ocean/unit/test_sharded_geom_fingerprint.py-93-    a = np.ones(SHAPE, dtype=np.float32)
tests/ocean/unit/test_sharded_geom_fingerprint.py-94-    b = np.ones((N_BANDS, 6, 9), dtype=np.float32)
tests/ocean/unit/test_sharded_geom_fingerprint.py:95:    fa = geom_band_fingerprint(a, N_BANDS)
tests/ocean/unit/test_sharded_geom_fingerprint.py:96:    fb = geom_band_fingerprint(b, N_BANDS)
tests/ocean/unit/test_sharded_geom_fingerprint.py-97-    # Different shapes -> different struct lengths; the agree helper is
tests/ocean/unit/test_sharded_geom_fingerprint.py-98-    # only called on stackable gathers, so assert the structs differ.
--
tests/ocean/unit/test_sharded_geom_fingerprint.py-102-def test_wrong_leading_axis_raises():
tests/ocean/unit/test_sharded_geom_fingerprint.py-103-    with pytest.raises(ValueError):
tests/ocean/unit/test_sharded_geom_fingerprint.py:104:        geom_band_fingerprint(np.ones((3, 2)), N_BANDS)
--
packages/core/legoesm/parallel/sharded_dynamics.py-56-    # Compilation happens on the first call; subsequent calls reuse the
packages/core/legoesm/parallel/sharded_dynamics.py-57-    # cached executable.
packages/core/legoesm/parallel/sharded_dynamics.py:58:    sharded_step = make_sharded_step(model, config)
packages/core/legoesm/parallel/sharded_dynamics.py-59-    state = shard_state(state, config)
packages/core/legoesm/parallel/sharded_dynamics.py:60:    state = sharded_step(state, dt=600.0)   # compiles once here
packages/core/legoesm/parallel/sharded_dynamics.py:61:    state = sharded_step(state, dt=600.0)   # reuses compiled executable
packages/core/legoesm/parallel/sharded_dynamics.py-62-    full_state = gather_state(state, config)
packages/core/legoesm/parallel/sharded_dynamics.py-63-
--
packages/core/legoesm/parallel/sharded_dynamics.py-109-            "reseed every step (issue #405/#413).  Use a diagnostic "
packages/core/legoesm/parallel/sharded_dynamics.py-110-            "scheme, the ModelDriver loops / MPAS step, or "
packages/core/legoesm/parallel/sharded_dynamics.py:111:            "make_voronoi_sharded_step(return_phys_state=True), which "
packages/core/legoesm/parallel/sharded_dynamics.py-112-            "thread the carry."
packages/core/legoesm/parallel/sharded_dynamics.py-113-        )
--
packages/core/legoesm/parallel/sharded_dynamics.py-418-    callable with the same signature as a plain step function::
packages/core/legoesm/parallel/sharded_dynamics.py-419-
packages/core/legoesm/parallel/sharded_dynamics.py:420:        compiled = make_sharded_step(model, config)
packages/core/legoesm/parallel/sharded_dynamics.py-421-        new_state = compiled(state, dt=600.0)
packages/core/legoesm/parallel/sharded_dynamics.py-422-
--
packages/core/legoesm/parallel/sharded_dynamics.py-660-
packages/core/legoesm/parallel/sharded_dynamics.py-661-
packages/core/legoesm/parallel/sharded_dynamics.py:662:def make_sharded_step(
packages/core/legoesm/parallel/sharded_dynamics.py-663-    model,
packages/core/legoesm/parallel/sharded_dynamics.py-664-    config: DeviceConfig,
--
packages/core/legoesm/parallel/sharded_dynamics.py-1005-    """
packages/core/legoesm/parallel/sharded_dynamics.py-1006-    if step_fn is None:
packages/core/legoesm/parallel/sharded_dynamics.py:1007:        step_fn = make_sharded_step(model, config, halo_exchange_fn)
packages/core/legoesm/parallel/sharded_dynamics.py-1008-
packages/core/legoesm/parallel/sharded_dynamics.py-1009-    # Ensure state is sharded
--
packages/core/legoesm/parallel/sharded_dynamics.py-1950-
packages/core/legoesm/parallel/sharded_dynamics.py-1951-
packages/core/legoesm/parallel/sharded_dynamics.py:1952:def make_voronoi_sharded_step(
packages/core/legoesm/parallel/sharded_dynamics.py-1953-    model,
packages/core/legoesm/parallel/sharded_dynamics.py-1954-    dev_config: DeviceConfig,
--
packages/core/legoesm/parallel/sharded_dynamics.py-2053-            # carry contract is the model/driver's own.
packages/core/legoesm/parallel/sharded_dynamics.py-2054-            raise ValueError(
packages/core/legoesm/parallel/sharded_dynamics.py:2055:                "make_voronoi_sharded_step(return_phys_state=True) needs "
packages/core/legoesm/parallel/sharded_dynamics.py-2056-                "a multi-device config; on a single device use "
packages/core/legoesm/parallel/sharded_dynamics.py-2057-                "model.step (eager, carry stashed on the model) or the "
--
packages/core/legoesm/parallel/sharded_dynamics.py-2571-            refuse_unthreaded_stateful_physics(
packages/core/legoesm/parallel/sharded_dynamics.py-2572-                physics_fn, phys_state,
packages/core/legoesm/parallel/sharded_dynamics.py:2573:                where="make_voronoi_sharded_step(return_phys_state=True)")
packages/core/legoesm/parallel/sharded_dynamics.py-2574-        else:
packages/core/legoesm/parallel/sharded_dynamics.py-2575-            _refuse_stateful_physics_unthreaded_wrapper(physics_fn)
--
packages/core/legoesm/parallel/__init__.py-41-
packages/core/legoesm/parallel/__init__.py-42-9. **Sharded dynamics** (shard_map-based):
packages/core/legoesm/parallel/__init__.py:43:   ``make_sharded_step()`` wraps a dynamics model for SPMD execution
packages/core/legoesm/parallel/__init__.py-44-   across devices with explicit sharding constraints and optional
packages/core/legoesm/parallel/__init__.py-45-   halo exchange.  ``shard_state()`` / ``gather_state()`` move data
--
packages/core/legoesm/grids/cubed_sphere_cdgrid.py-353-        # deliberate O(dx²) approximation to FV3's spherical-excess `get_area`
packages/core/legoesm/grids/cubed_sphere_cdgrid.py-354-        # (fv_grid_utils.F90).  The two agree to ~1e-4 at C36 / ~1e-2 at C8 and
packages/core/legoesm/grids/cubed_sphere_cdgrid.py:355:        # converge as the grid refines; the entire SW-core gold-file fingerprint
packages/core/legoesm/grids/cubed_sphere_cdgrid.py-356-        # surface (cosine-bell, d_sw_native, production-tendencies, corner-
packages/core/legoesm/grids/cubed_sphere_cdgrid.py-357-        # vorticity, ...) is pinned to this chord convention.  The FV3-faithful
--
packages/core/legoesm/parallel/geometry_consistency.py-13-domain, a different field list, a mixed ``jax_enable_x64``), turning a loud
packages/core/legoesm/parallel/geometry_consistency.py-14-crash into wrong physics.  So every broadcast here is GUARDED: an allgathered
packages/core/legoesm/parallel/geometry_consistency.py:15:fingerprint must agree first, and a disagreement RAISES.
packages/core/legoesm/parallel/geometry_consistency.py-16-
packages/core/legoesm/parallel/geometry_consistency.py-17-This module is the ONE implementation of that protocol.  It was extracted
--
packages/core/legoesm/parallel/geometry_consistency.py-20-lane — which had the identical defect (#1362) — reuses it instead of growing
packages/core/legoesm/parallel/geometry_consistency.py-21-a second, drifting copy.  Per legoESM's no-duplicated-numerics rule, new SPMD
packages/core/legoesm/parallel/geometry_consistency.py:22:lanes MUST call these helpers rather than re-derive the fingerprints.
packages/core/legoesm/parallel/geometry_consistency.py-23-
packages/core/legoesm/parallel/geometry_consistency.py-24-Sequencing contract, in this order:
packages/core/legoesm/parallel/geometry_consistency.py-25-
packages/core/legoesm/parallel/geometry_consistency.py:26:1. :func:`assert_schema_agrees` ONCE, before any per-field work — a single
packages/core/legoesm/parallel/geometry_consistency.py-27-   fixed-shape collective that every process reaches.  A process-dependent
packages/core/legoesm/parallel/geometry_consistency.py-28-   field selection (e.g. an optional mask present on some ranks only) would
--
packages/core/legoesm/parallel/geometry_consistency.py-32-   process.
packages/core/legoesm/parallel/geometry_consistency.py-33-
packages/core/legoesm/parallel/geometry_consistency.py:34:NO DEADLOCK RISK: every process fingerprints the same fields in the same
packages/core/legoesm/parallel/geometry_consistency.py-35-order and derives its verdict from the SAME gathered array, so the refusal is
packages/core/legoesm/parallel/geometry_consistency.py-36-symmetric — all raise or none.
--
packages/core/legoesm/parallel/geometry_consistency.py-47-    "content_hash48",
packages/core/legoesm/parallel/geometry_consistency.py-48-    "name_digest48",
packages/core/legoesm/parallel/geometry_consistency.py:49:    "schema_fingerprint",
packages/core/legoesm/parallel/geometry_consistency.py:50:    "assert_schema_agrees",
packages/core/legoesm/parallel/geometry_consistency.py-51-    "assert_flags_agree",
packages/core/legoesm/parallel/geometry_consistency.py-52-    "broadcast_checked",
--
packages/core/legoesm/parallel/geometry_consistency.py-294-    list has neither; falling through to ``np.asarray`` there is FREE (it is
packages/core/legoesm/parallel/geometry_consistency.py-295-    already host data) and, critically, keeps this function from dying with a
packages/core/legoesm/parallel/geometry_consistency.py:296:    bare ``AttributeError`` BEFORE :func:`assert_schema_agrees` reaches its
packages/core/legoesm/parallel/geometry_consistency.py-297-    collective — a rank-local raise ahead of a collective is a HANG, so
packages/core/legoesm/parallel/geometry_consistency.py-298-    "fail explicitly" here must NOT mean "raise here" (codex round-3, minor 3).
--
packages/core/legoesm/parallel/geometry_consistency.py-340-
packages/core/legoesm/parallel/geometry_consistency.py-341-    Used to compare EXACT-dtype arrays (masks, index tables) across
packages/core/legoesm/parallel/geometry_consistency.py:342:    processes: unlike moment fingerprints, a byte digest is positional, so a
packages/core/legoesm/parallel/geometry_consistency.py-343-    permutation or a two-cell flip cannot cancel. 48 bits keeps the value
packages/core/legoesm/parallel/geometry_consistency.py-344-    under 2**53 so it survives the float64 ``process_allgather`` payload
--
packages/core/legoesm/parallel/geometry_consistency.py-365-
packages/core/legoesm/parallel/geometry_consistency.py-366-
packages/core/legoesm/parallel/geometry_consistency.py:367:def schema_fingerprint(names, n_dev, dtype_kinds=(), ndims=()) -> np.ndarray:
packages/core/legoesm/parallel/geometry_consistency.py-368-    """Fixed-shape schema digest gathered ONCE before the per-field loop.
packages/core/legoesm/parallel/geometry_consistency.py-369-
--
packages/core/legoesm/parallel/geometry_consistency.py-373-    The dtype/ndim terms are not cosmetic.  :func:`broadcast_checked` routes
packages/core/legoesm/parallel/geometry_consistency.py-374-    exact dtypes to a 1-value digest and float dtypes to a 3-moment
packages/core/legoesm/parallel/geometry_consistency.py:375:    fingerprint, and its struct entry depends on ndim.  If the schema gate
packages/core/legoesm/parallel/geometry_consistency.py-376-    did not cover those, a field that is bool on one process and float on
packages/core/legoesm/parallel/geometry_consistency.py-377-    another would PASS the gate and then deadlock inside the per-field
--
packages/core/legoesm/parallel/geometry_consistency.py-470-
packages/core/legoesm/parallel/geometry_consistency.py-471-
packages/core/legoesm/parallel/geometry_consistency.py:472:def assert_schema_agrees(names, n_dev, *, context: str, arrays=None) -> None:
packages/core/legoesm/parallel/geometry_consistency.py-473-    """Raise unless every process agrees on the geometry field SCHEMA.
packages/core/legoesm/parallel/geometry_consistency.py-474-
--
packages/core/legoesm/parallel/geometry_consistency.py-510-        ndims = [d[1] for d in described]
packages/core/legoesm/parallel/geometry_consistency.py-511-    gathered = multihost_utils.process_allgather(
packages/core/legoesm/parallel/geometry_consistency.py:512:        schema_fingerprint(names, n_dev, kinds, ndims))
packages/core/legoesm/parallel/geometry_consistency.py-513-    if not bool(np.all(gathered == gathered[0])):
packages/core/legoesm/parallel/geometry_consistency.py-514-        raise RuntimeError(
--
packages/core/legoesm/parallel/geometry_consistency.py-528-    host round trip, no dtype/weak-type change.
packages/core/legoesm/parallel/geometry_consistency.py-529-
packages/core/legoesm/parallel/geometry_consistency.py:530:    The fingerprint compares structural entries exactly; value entries
packages/core/legoesm/parallel/geometry_consistency.py-531-    EXACTLY for integer/bool arrays and to ``rtol=1e-5`` for float arrays.
packages/core/legoesm/parallel/geometry_consistency.py-532-
packages/core/legoesm/parallel/geometry_consistency.py-533-    Integer/bool arrays (masks, index tables) are exact data, not autotuned
packages/core/legoesm/parallel/geometry_consistency.py:534:    arithmetic: their BYTES are fingerprinted so a positional difference is
packages/core/legoesm/parallel/geometry_consistency.py-535-    caught.  Moment-only compares are blind to a permutation — a bool mask's
packages/core/legoesm/parallel/geometry_consistency.py-536-    ``(sum, sumsq, absmax)`` is identical for every arrangement with the same
--
packages/core/legoesm/parallel/geometry_consistency.py-612-# callers' aux-threading contract, see make_sharded_ocean_step.
packages/core/legoesm/parallel/geometry_consistency.py-613-
packages/core/legoesm/parallel/geometry_consistency.py:614:def band_fingerprint(host, n_bands):
packages/core/legoesm/parallel/geometry_consistency.py:615:    """Per-band fingerprint of a band-STACKED field (leading axis n_bands).
packages/core/legoesm/parallel/geometry_consistency.py-616-
packages/core/legoesm/parallel/geometry_consistency.py-617-    PREREQUISITE: ``n_bands`` (and each field's dtype class / shape) must
packages/core/legoesm/parallel/geometry_consistency.py:618:    already be schema-gated across processes (:func:`assert_schema_agrees`)
packages/core/legoesm/parallel/geometry_consistency.py-619-    — the payload widths depend on it, and mismatched widths would hang the
packages/core/legoesm/parallel/geometry_consistency.py-620-    allgather rather than raise.
--
packages/core/legoesm/parallel/geometry_consistency.py-633-    if host.shape[0] != n_bands:
packages/core/legoesm/parallel/geometry_consistency.py-634-        raise ValueError(
packages/core/legoesm/parallel/geometry_consistency.py:635:            f"band_fingerprint: leading axis {host.shape[0]} != n_bands "
packages/core/legoesm/parallel/geometry_consistency.py-636-            f"{n_bands}")
packages/core/legoesm/parallel/geometry_consistency.py-637-    is_exact = host.dtype.kind in "biu"
--
packages/core/legoesm/parallel/geometry_consistency.py-657-
packages/core/legoesm/parallel/geometry_consistency.py-658-
packages/core/legoesm/parallel/geometry_consistency.py:659:def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=None):
packages/core/legoesm/parallel/geometry_consistency.py:660:    """True iff every process's :func:`band_fingerprint` matches process 0's."""
packages/core/legoesm/parallel/geometry_consistency.py-661-    if rtol is None:
packages/core/legoesm/parallel/geometry_consistency.py-662-        rtol = _FLOAT_RTOL
--
packages/core/legoesm/parallel/geometry_consistency.py-669-
packages/core/legoesm/parallel/geometry_consistency.py-670-
packages/core/legoesm/parallel/geometry_consistency.py:671:def checked_shard_put(arr, name, sharding, *, context, n_bands):
packages/core/legoesm/parallel/geometry_consistency.py-672-    """Gate a band-stacked field per band, then put WITHOUT broadcast or
packages/core/legoesm/parallel/geometry_consistency.py-673-    jax's whole-array device_put assert (walls 1+2 above).
packages/core/legoesm/parallel/geometry_consistency.py-674-
packages/core/legoesm/parallel/geometry_consistency.py-675-    Single-process: plain ``jax.device_put`` — byte-unchanged, no host
packages/core/legoesm/parallel/geometry_consistency.py:676:    round trip. Multi-process: per-band fingerprint gate (symmetric raise
packages/core/legoesm/parallel/geometry_consistency.py-677-    on real divergence), then ``jax.make_array_from_callback`` hands each
packages/core/legoesm/parallel/geometry_consistency.py-678-    process exactly its addressable slabs. Cross-process byte-identity of
--
packages/core/legoesm/parallel/geometry_consistency.py-685-
packages/core/legoesm/parallel/geometry_consistency.py-686-    host = np.asarray(arr)
packages/core/legoesm/parallel/geometry_consistency.py:687:    struct, vals, is_exact = band_fingerprint(host, n_bands)
packages/core/legoesm/parallel/geometry_consistency.py-688-    g_struct = multihost_utils.process_allgather(struct)
packages/core/legoesm/parallel/geometry_consistency.py-689-    g_vals = multihost_utils.process_allgather(vals)
packages/core/legoesm/parallel/geometry_consistency.py:690:    if not band_fingerprints_agree(g_struct, g_vals, is_exact):
packages/core/legoesm/parallel/geometry_consistency.py-691-        raise RuntimeError(
packages/core/legoesm/parallel/geometry_consistency.py-692-            f"{context}: band-stacked field {name!r} DIVERGES across "
--
packages/core/legoesm/parallel/geometry_consistency.py-698-
packages/core/legoesm/parallel/geometry_consistency.py-699-
packages/core/legoesm/parallel/geometry_consistency.py:700:def assert_pytree_bytes_equal(tree, what):
packages/core/legoesm/parallel/geometry_consistency.py-701-    """Cheap multi-process replacement for the per-leaf assert_equal that
packages/core/legoesm/parallel/geometry_consistency.py:702:    :func:`checked_shard_put`-style puts bypass on NON-band inputs (state /
packages/core/legoesm/parallel/geometry_consistency.py-703-    forcing pytrees): one 48-bit digest per array leaf, one tiny allgather,
packages/core/legoesm/parallel/geometry_consistency.py-704-    symmetric raise on mismatch. No-op single-process.
--
packages/core/legoesm/parallel/geometry_consistency.py-723-
packages/core/legoesm/parallel/geometry_consistency.py-724-
packages/core/legoesm/parallel/geometry_consistency.py:725:def addressable_shard_put(arr, sharding):
packages/core/legoesm/parallel/geometry_consistency.py-726-    """Ungated assert-free put (walls 1+2) for inputs whose cross-process
packages/core/legoesm/parallel/geometry_consistency.py-727-    consistency the CALLER has already gated (state/forcing pytrees via
packages/core/legoesm/parallel/geometry_consistency.py:728:    :func:`assert_pytree_bytes_equal`). Single-process: plain device_put."""
packages/core/legoesm/parallel/geometry_consistency.py-729-    if jax.process_count() <= 1:
packages/core/legoesm/parallel/geometry_consistency.py-730-        return jax.device_put(arr, sharding)
--
packages/ocean/legoesm/ocean/experiments/recipe_map.py-19-
packages/ocean/legoesm/ocean/experiments/recipe_map.py-20-# experiment (setup) name -> the catalog recipe its dycore matches today.
packages/ocean/legoesm/ocean/experiments/recipe_map.py:21:# Derived from the empirical dycore-fingerprint clustering (see module docstring);
packages/ocean/legoesm/ocean/experiments/recipe_map.py-22-# kept honest by the verification test.
packages/ocean/legoesm/ocean/experiments/recipe_map.py-23-EXPERIMENT_RECIPES: dict[str, str] = {
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-47-import numpy as np
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-48-from legoesm.parallel.geometry_consistency import (
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:49:    FLAG_ABSENT, addressable_shard_put, assert_flags_agree,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:50:    assert_pytree_bytes_equal, assert_schema_agrees, checked_shard_put,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-51-    coerce_bool, coerce_count, config_digest48, name_digest48,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-52-    tree_schema_digest48)
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-313-    # would silently delete a LIVE seam row.  Host-side check on the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-314-    # concrete state (this fn runs outside jit).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:315:    assert_pytree_bytes_equal(state, "shard_state_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-316-    vm = getattr(state, "v_mask", None)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-317-    if vm is not None:
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-331-            return None
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-332-        sh = NamedSharding(mesh, _lat_spec(field.data))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:333:        return field.replace(data=addressable_shard_put(field.data, sh))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-334-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-335-    def _shard_v(field):
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-349-        v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-350-        sh = NamedSharding(mesh, _lat_spec(v_lower))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:351:        return field.replace(data=addressable_shard_put(v_lower, sh))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-352-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-353-    updates = {}
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-371-            # ndim>=1 lat-shards via _lat_spec (1-D included); only true
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-372-            # scalars replicate.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:373:            updates[name] = addressable_shard_put(arr, NamedSharding(mesh, spec))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-374-    return state._replace(**updates)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-375-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-392-    if forcing is None or mesh is None:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-393-        return forcing
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:394:    assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-395-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-396-    def _put(leaf):
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-398-            return None
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-399-        arr = jnp.asarray(leaf)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:400:        return addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-401-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-402-    return jax.tree.map(_put, forcing)
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-435-        return stack
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-436-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:437:    assert_pytree_bytes_equal(stack, "shard_forcing_stack_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-438-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-439-    def _put(leaf):
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-447-        else:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-448-            spec = P()
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:449:        return addressable_shard_put(arr, NamedSharding(mesh, spec))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-450-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-451-    return jax.tree.map(_put, stack)
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-612-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-613-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:614:def _agree_ocean_spmd_call(mesh, state, forcing, *, where: str) -> None:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-615-    """Agree a returned ocean SPMD callable's per-CALL inputs, FIRST statement.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-616-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-724-    def _replicated_put(arr, name):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-725-        # (Name kept for history; this is a SHARDED P("lat") stack put.)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:726:        # checked_shard_put replaces the broadcast_checked+device_put pair:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-727-        # the broadcast's psum program is [n_processes, stack] (nd x 849 MB
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-728-        # at LL2304 — the @96/@128 wall), and a numpy device_put onto an
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-731-        # schema-gated just below, so payload widths agree). ONE shared
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-732-        # implementation: legoesm.parallel.geometry_consistency.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:733:        return checked_shard_put(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-734-            arr, name, rep, context="make_sharded_ocean_step",
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-735-            n_bands=n_dev)
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-750-    _raw_vmask = jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-751-    _gate_names = [*array_field_names, "vertex_mask"]
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:752:    assert_schema_agrees(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-753-        _gate_names, n_dev, context="make_sharded_ocean_step",
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-754-        arrays=[*(_raw_geom[n] for n in array_field_names), _raw_vmask])
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-870-                    f"(n_lat, n_lon[, nlev]) to shard on the lat axis.")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-871-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:872:    def sharded_step(state, dt, freshwater=None, surface_forcing=None,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-873-                     sponge=None, t_seconds=None, aux=None):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-874-        # ``aux``: the sharded geometry+vmask stacks. When this wrapper runs
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-877-        # OUTER-trace constants whose value the MLIR handler cannot fetch
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-878-        # for non-addressable arrays (broken since #1370-iii sharded the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:879:        # stacks). Outer-jit callers MUST thread ``step.aux`` through their
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-880-        # jit boundary as an ARGUMENT and pass it back here.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-881-        # ONE forcing operand: None fields drop out of the pytree structure,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-882-        # so specs derived by tree.map skip them automatically and the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-883-        # structure key below distinguishes every None<->array combination.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:884:        _agree_ocean_spmd_call(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-885-            mesh, state, (freshwater, surface_forcing, sponge, t_seconds),
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-886-            where="make_sharded_ocean_step.step")
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-960-    # Expose the stacks so outer-jit callers can pass them as arguments
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-961-    # (see the ``aux`` note in the signature).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:962:    sharded_step.aux = (geom_stacks, vmask_stack)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-963-    return sharded_step
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-964-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-991-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-992-    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:993:        _agree_ocean_spmd_call(mesh, state, (surface_forcing, freshwater),
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-994-                               where="make_sharded_ocean_step_global.step")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-995-        # Scatter the global state AND forcing to the band layout
--
packages/ocean/legoesm/ocean/fidelity/precision_gate.py-10-with x64 enabled and still build its GRID in single precision while T/S look
packages/ocean/legoesm/ocean/fidelity/precision_gate.py-11-correct.  ``create_z_star_from_thicknesses`` did exactly that to NEMO's f64
packages/ocean/legoesm/ocean/fidelity/precision_gate.py:12:``gdept_1d`` (median |rel| 2.555e-8 = 0.21 x f32 eps, the fingerprint of
packages/ocean/legoesm/ocean/fidelity/precision_gate.py-13-single precision), which alone accounted for the residuals in ``eos_rab
packages/ocean/legoesm/ocean/fidelity/precision_gate.py-14-alpha``, ``bn2`` and 10 of ``zdf_mxl``'s MLD columns.
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-34-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-35-from legoesm.parallel.geometry_consistency import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:36:    FLAG_ABSENT, assert_flags_agree, assert_schema_agrees, broadcast_checked,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-37-    coerce_bool, coerce_count, config_digest48, name_digest48,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-38-    tree_schema_digest48)
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-486-    # would desynchronize the per-field gathers rather than fail cleanly.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-487-    ordered_names = list(raw)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:488:    assert_schema_agrees(ordered_names, n_dev,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-489-                         context="make_sharded_atm_latlon_step",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-490-                         arrays=[raw[n] for n in ordered_names])
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-980-    _cache = {}
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-981-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:982:    def sharded_step(c_state, dt, phys_state=None):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-983-        _agree_spmd_call(mesh, c_state, phys_state,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-984-                         where="make_sharded_atm_latlon_step.step")
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1578-    # per-process recompute, same replicated-device_put bit-identity assert.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1579-    # n_dev is the FULL device count here (p_lat * p_lon): the schema
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1580:    # fingerprint must describe this process's whole mesh, not one axis.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1581-    ordered_names = list(raw)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1582:    assert_schema_agrees(ordered_names, p_lat * p_lon,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1583-                         context="make_sharded_atm_latlon_step_2d",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1584-                         arrays=[raw[n] for n in ordered_names])
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1749-    _cache = {}
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1750-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1751:    def sharded_step(c_state, dt, phys_state=None):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1752-        _agree_spmd_call(mesh, c_state, phys_state,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1753-                         where="make_sharded_atm_latlon_step_2d.step")

exec
/usr/bin/bash -lc "pwd; rg --files -g 'AGENTS.md' -g 'CLAUDE.md' -g 'pyproject.toml' -g 'geometry_consistency.py' -g 'sharded_ocean_step.py' -g 'sharded_atm_latlon_step.py' -g 'test_sharded_geom_fingerprint.py' | sed -n '1,80p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 391ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
CLAUDE.md
pyproject.toml
packages/ocean/pyproject.toml
tests/ocean/unit/test_sharded_geom_fingerprint.py
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
packages/land/pyproject.toml
packages/tools/pyproject.toml
packages/ice/pyproject.toml
packages/atmosphere/pyproject.toml
packages/ml/pyproject.toml
packages/coupler/pyproject.toml
packages/core/pyproject.toml
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
packages/core/legoesm/parallel/geometry_consistency.py

exec
/usr/bin/bash -lc 'git status --short; git rev-parse --short HEAD; git diff --check e13de3afe''^1 e13de3afe -- packages/core/legoesm/parallel/geometry_consistency.py packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py; git diff --unified=80 e13de3afe''^1 e13de3afe -- packages/core/legoesm/parallel/geometry_consistency.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 572ms:
?? .physics-validator/scaling_campaign/codex_merge_r20_2026-08-03.md
e13de3afe
diff --git a/packages/core/legoesm/parallel/geometry_consistency.py b/packages/core/legoesm/parallel/geometry_consistency.py
new file mode 100644
index 000000000..0a131ca18
--- /dev/null
+++ b/packages/core/legoesm/parallel/geometry_consistency.py
@@ -0,0 +1,733 @@
+"""Cross-process agreement checks for per-process-recomputed SPMD geometry.
+
+Every multi-controller SPMD lane faces the same hazard: each process rebuilds
+the band/tile geometry from the same config, then hands it to a REPLICATED
+``device_put``.  A ``P()`` (fully-replicated) put ASSERTS the value is
+bit-identical on every process, and per-process XLA autotuning on
+device-derived grid fields makes the last ULPs differ at larger sizes (job
+26450848: LL576 np=4, area-scale fields differing at 1e-7 relative), which
+trips that assert.
+
+The remedy is to broadcast process 0's bytes — but broadcasting BLINDLY would
+silently paper over a REAL cross-process inconsistency (a different wet
+domain, a different field list, a mixed ``jax_enable_x64``), turning a loud
+crash into wrong physics.  So every broadcast here is GUARDED: an allgathered
+fingerprint must agree first, and a disagreement RAISES.
+
+This module is the ONE implementation of that protocol.  It was extracted
+from ``ocean.dynamics.sharded_ocean_step`` (where it was developed and
+hardened over five rounds of adversarial review) so the atmosphere lat-lon
+lane — which had the identical defect (#1362) — reuses it instead of growing
+a second, drifting copy.  Per legoESM's no-duplicated-numerics rule, new SPMD
+lanes MUST call these helpers rather than re-derive the fingerprints.
+
+Sequencing contract, in this order:
+
+1. :func:`assert_schema_agrees` ONCE, before any per-field work — a single
+   fixed-shape collective that every process reaches.  A process-dependent
+   field selection (e.g. an optional mask present on some ranks only) would
+   otherwise DESYNCHRONIZE the per-field gathers below instead of failing
+   with a clear message.
+2. :func:`broadcast_checked` per field, in an order identical on every
+   process.
+
+NO DEADLOCK RISK: every process fingerprints the same fields in the same
+order and derives its verdict from the SAME gathered array, so the refusal is
+symmetric — all raise or none.
+"""
+
+from __future__ import annotations
+
+import hashlib
+
+import jax
+import numpy as np
+
+__all__ = [
+    "content_hash48",
+    "name_digest48",
+    "schema_fingerprint",
+    "assert_schema_agrees",
+    "assert_flags_agree",
+    "broadcast_checked",
+    "coerce_count",
+    "coerce_bool",
+    "config_digest48",
+    "tree_schema_digest48",
+    "safe_repr",
+    "FLAG_ABSENT",
+    "FLAG_UNCOERCIBLE",
+    "FLAG_OUT_OF_RANGE",
+    "FLAG_NEGATIVE",
+    "FLAG_MAX_EXACT",
+    "FLAG_DIGEST_FAILED",
+]
+
+# --- entry-gate payload sentinels -------------------------------------------
+# An entry gate turns rank-local scalars (n_steps, segment_steps, grid dims)
+# into a fixed-width float payload.  Building that payload must NEVER raise:
+# a rank that dies in `int(n_steps)` while its peers block in
+# `process_allgather` is a HANG, which is strictly worse than the bug the gate
+# exists to fix (codex 2026-07-29 round-3, blocker 3).  So an unusable value is
+# mapped to a SENTINEL that travels through the collective; every rank then
+# sees it in the gathered payload and the raise that follows is symmetric.
+#
+# The sentinels are large-magnitude NEGATIVE values that NO legitimate count
+# can take.  They must also not collide with each other: ``FLAG_ABSENT`` used
+# to be ``-1.0``, so a rank passing ``segment_steps=None`` and a peer passing
+# ``-1`` produced the SAME payload entry, agreed, and then diverged downstream
+# (codex round-4, blocker 1).  Counts are validated non-negative, so every
+# sentinel is unreachable from valid data AND distinct from every other.
+FLAG_ABSENT = -6.0e15
+FLAG_UNCOERCIBLE = -8.0e15
+FLAG_OUT_OF_RANGE = -7.0e15
+FLAG_NEGATIVE = -5.0e15
+FLAG_DIGEST_FAILED = -4.0e15
+# 2**53 is the largest integer whose successor is exactly representable in
+# float64.  Above it two DIFFERENT counts alias to the same payload entry, so
+# the gate would pass a real divergence (codex round-3, minor 2).  Values past
+# the bound are refused rather than silently compared.
+FLAG_MAX_EXACT = 2.0 ** 53
+_MAX_EXACT_INT = 2 ** 53
+
+
+def safe_repr(value, limit: int = 120) -> str:
+    """``repr(value)`` that cannot raise and cannot blow up the message.
+
+    A user object whose ``__repr__`` raises would otherwise propagate out of
+    the payload build — the very pre-collective throw the gates exist to
+    remove (codex round-4, blocker 1).
+    """
+    try:
+        text = repr(value)
+    except Exception:                       # pragma: no cover - defensive
+        try:
+            text = f"<unrepresentable {type(value).__name__}>"
+        except Exception:                   # pragma: no cover - defensive
+            text = "<unrepresentable>"
+    return text if len(text) <= limit else text[:limit] + "..."
+
+
+def coerce_count(value, *, absent: float = FLAG_ABSENT):
+    """Map a rank-local COUNT to an exactly-comparable entry-gate payload float.
+
+    Returns ``(payload, problem)``.  ``problem`` is ``None`` when the value is
+    usable; otherwise it is a human-readable clause naming the offending value,
+    which the caller must raise AFTER its collective so the refusal is
+    symmetric across processes.
+
+    This function NEVER raises.  That is the whole point: it is called while
+    ASSEMBLING a collective payload, upstream of the collective itself, where a
+    raise deadlocks the peers (codex round-3, blocker 3).
+
+    STRICT by type, not by coercibility (codex round-4, blocker 1).  Only a
+    real non-negative Python/NumPy integer is accepted:
+
+    * ``3.5`` is REJECTED.  ``int(3.5) == 3`` made a rank carrying ``3.5``
+      indistinguishable from a peer carrying ``3``; the payloads agreed and
+      then ``range(3.5)`` blew up on one rank alone while its peer entered the
+      step collective.
+    * ``bool`` is REJECTED.  ``True`` is not a step count, and silently
+      encoding it as ``1`` hides a caller bug.
+    * Arrays (even size-1) are REJECTED: ``int(arr)`` succeeds for size 1 and
+      raises for size > 1, so accepting them makes the gate's behaviour depend
+      on rank-local shape.
+    * NEGATIVE integers get their OWN sentinel, so they can never collide with
+      the "absent" encoding.
+
+    ``None`` maps to ``absent`` (default :data:`FLAG_ABSENT`, itself outside
+    the valid range) so a call site that does not carry the value still emits a
+    FIXED-WIDTH payload.
+    """
+    if value is None:
+        return float(absent), None
+    # `bool` is a subclass of `int`, so it must be excluded FIRST.
+    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
+        return FLAG_UNCOERCIBLE, (
+            "must be a non-negative Python/NumPy integer (got type "
+            f"{type(value).__name__}: {safe_repr(value)}); every process must "
+            "be launched with the same value")
+    try:
+        as_int = int(value)
+    except Exception:                       # pragma: no cover - defensive
+        return FLAG_UNCOERCIBLE, (
+            f"could not be read as an integer ({safe_repr(value)})")
+    if as_int < 0:
+        return FLAG_NEGATIVE, (
+            f"must be non-negative (got {safe_repr(value)})")
+    # Range check on the INTEGER: converting to float first would already have
+    # collapsed 2**53+1 onto 2**53, so the very aliasing this guards against
+    # would be invisible to the guard.
+    if as_int > _MAX_EXACT_INT:
+        return FLAG_OUT_OF_RANGE, (
+            f"is outside the exactly-comparable range n <= 2**53 (got "
+            f"{safe_repr(value)}); beyond that bound two different counts "
+            f"alias to the same float64 payload entry and the cross-process "
+            f"agreement check would pass a real divergence")
+    return float(as_int), None
+
+
+def coerce_bool(value, *, absent: float = FLAG_ABSENT):
+    """Strict, NON-THROWING tri-state encoder for a rank-local BOOLEAN flag.
+
+    Returns ``(payload, problem)`` exactly like :func:`coerce_count`.
+    ``None`` -> ``absent`` ("not applicable at this call site").
+
+    Only a real ``bool`` / ``np.bool_`` is accepted.  ``bool(value)`` on an
+    arbitrary object RAISES for a multi-element array ("truth value of an array
+    is ambiguous") — inside a gate that is a pre-collective throw, i.e. a hang
+    (codex round-4, blocker 2).  Anything else becomes a sentinel that travels
+    through the collective and is refused symmetrically afterwards.
+    """
+    if value is None:
+        return float(absent), None
+    if isinstance(value, (bool, np.bool_)):
+        return (1.0 if value else 0.0), None
+    return FLAG_UNCOERCIBLE, (
+        f"must be a bool (got type {type(value).__name__}: "
+        f"{safe_repr(value)})")
+
+
+def _canonical_config_terms(obj, prefix: str = "", depth: int = 0,
+                            out=None, seen=None):
+    """Flatten a config object into ORDERED ``"path=value"`` strings.
+
+    Covers the STATIC scalars that select a compiled program: scheme literals,
+    integrator names, and every feature-gating bool (``fix_mass``,
+    ``fix_moisture``, ``use_polar_filter``, ...).  Arrays contribute only
+    ``dtype`` + ``shape`` — comparing their VALUES is the job of
+    :func:`broadcast_checked`, not of a cheap fixed-width entry gate.
+
+    Never raises: any unreadable field becomes a ``<unreadable>`` term, which
+    still participates in the comparison.
+    """
+    if out is None:
+        out, seen = [], set()
+    if depth > 4 or len(out) > 512:          # bounded work, bounded payload
+        return out
+    if id(obj) in seen:
+        return out
+    seen.add(id(obj))
+    fields = getattr(obj, "_fields", None)   # NamedTuple
+    if fields is None:
+        dc = getattr(obj, "__dataclass_fields__", None)
+        fields = tuple(dc) if dc else None
+    if fields is None:
+        return out
+    for name in fields:
+        try:
+            val = getattr(obj, name)
+        except Exception:                    # pragma: no cover - defensive
+            out.append(f"{prefix}{name}=<unreadable>")
+            continue
+        path = f"{prefix}{name}"
+        if val is None or isinstance(val, (bool, int, float, str, np.bool_,
+                                           np.integer, np.floating)):
+            out.append(f"{path}={safe_repr(val, 64)}")
+        elif hasattr(val, "dtype") and hasattr(val, "shape"):
+            out.append(f"{path}=array:{safe_repr(val.dtype, 32)}:"
+                       f"{safe_repr(tuple(val.shape), 64)}")
+        elif getattr(val, "_fields", None) or getattr(
+                val, "__dataclass_fields__", None):
+            _canonical_config_terms(val, path + ".", depth + 1, out, seen)
+        else:
+            out.append(f"{path}=<{type(val).__name__}>")
+    return out
+
+
+def config_digest48(obj) -> float:
+    """One fixed-width, order-sensitive digest of a config's STATIC scalars.
+
+    Why a digest instead of a hand-listed set of flags: an entry gate that
+    enumerates ``fold``/``anchor``/``polar`` by hand agrees only the fields
+    somebody remembered.  ``fix_mass`` gates a global-area psum,
+    ``outer_integrator`` selects a different program, ``fix_moisture`` adds a
+    reduction — each was MISSING from the hand-written list (codex round-4,
+    blocker 3).  Digesting every static scalar closes the class instead of the
+    three instances, and costs ONE payload entry.
+
+    Never raises; an internal failure returns :data:`FLAG_DIGEST_FAILED`,
+    which still compares equal across ranks that fail identically and unequal
+    against a rank that succeeded.
+    """
+    try:
+        return name_digest48(_canonical_config_terms(obj))
+    except Exception:                        # pragma: no cover - defensive
+        return FLAG_DIGEST_FAILED
+
+
+def tree_schema_digest48(tree) -> float:
+    """Digest of a pytree's LEAF SCHEMA: ordered path, dtype and full shape.
+
+    The gather/scatter entry points run one cross-process replication PER
+    NON-``None`` LEAF, so the NUMBER and ORDER of those collectives is
+    rank-local data: a state whose tracer dict differs across processes (extra
+    species, different insertion order, different shape) produces mismatched
+    schedules and hangs (codex round-4, blocker 5).  Folding the whole leaf
+    schema into ONE fixed-width float makes that a clean symmetric raise.
+
+    ``jax.tree_util`` key paths give a canonical, ORDER-SENSITIVE description
+    (dict keys are sorted by ``tree_flatten_with_path``, so an insertion-order
+    difference alone does not false-positive, while a KEY-SET difference does
+    move the digest).  Never raises.
+    """
+    try:
+        from jax.tree_util import tree_flatten_with_path, keystr
+        leaves, _ = tree_flatten_with_path(tree)
+        terms = []
+        for path, leaf in leaves:
+            dtype = getattr(leaf, "dtype", None)
+            shape = getattr(leaf, "shape", None)
+            terms.append(
+                f"{keystr(path)}:{safe_repr(dtype, 32)}:"
+                f"{safe_repr(tuple(shape) if shape is not None else None, 64)}")
+        return name_digest48(terms)
+    except Exception:                        # pragma: no cover - defensive
+        return FLAG_DIGEST_FAILED
+
+
+def _dtype_kind_and_ndim(a):
+    """``(kind, ndim)`` for the schema digest, tolerant of a plain scalar.
+
+    Reads ``.dtype``/``.ndim`` from METADATA when present (a jax array exposes
+    both without materialising, so no device sync).  A plain Python scalar or
+    list has neither; falling through to ``np.asarray`` there is FREE (it is
+    already host data) and, critically, keeps this function from dying with a
+    bare ``AttributeError`` BEFORE :func:`assert_schema_agrees` reaches its
+    collective — a rank-local raise ahead of a collective is a HANG, so
+    "fail explicitly" here must NOT mean "raise here" (codex round-3, minor 3).
+
+    An unsupported dtype class (object/str) is reported as ``unsupported:<k>``
+    rather than being silently bucketed with the float fields, so a
+    disagreement about it is visible in the digest and a same-on-all-ranks
+    unsupported field fails later in :func:`broadcast_checked` with its own
+    message instead of here.
+    """
+    dtype = getattr(a, "dtype", None)
+    if dtype is None:
+        host = np.asarray(a)
+        dtype, ndim = host.dtype, host.ndim
+    else:
+        ndim = int(getattr(a, "ndim", np.ndim(a)))
+    k = np.dtype(dtype).kind
+    if k in "biu":
+        kind = "exact"
+    elif k in "fc":
+        kind = "inexact"
+    else:
+        kind = f"unsupported:{k}"
+    return kind, int(ndim)
+
+
+# Every per-field collective payload is padded to these FIXED widths.  A
+# payload whose LENGTH depends on rank-local data (dtype class, ndim,
+# non-finite count) would let two processes enter `process_allgather` with
+# different shapes and DEADLOCK -- the exact failure this module exists to
+# turn into a clean symmetric raise (codex 2026-07-29, blocker 2; the flaw was
+# inherited from the pre-extraction ocean implementation, so fixing it here
+# fixes BOTH lanes).
+_STRUCT_WIDTH = 8
+_VALS_WIDTH = 3
+
+# Relative tolerance for FLOAT geometry fields. Only ULP-scale autotune drift
+# is expected there; quantize-then-assert-equal false-positived on a rounding
+# boundary (job 26453240), so compare with a tolerance instead.
+_FLOAT_RTOL = 1e-5
+
+
+def content_hash48(arr) -> float:
+    """48-bit content digest of ``arr``'s bytes, exactly representable in f64.
+
+    Used to compare EXACT-dtype arrays (masks, index tables) across
+    processes: unlike moment fingerprints, a byte digest is positional, so a
+    permutation or a two-cell flip cannot cancel. 48 bits keeps the value
+    under 2**53 so it survives the float64 ``process_allgather`` payload
+    exactly. Not cryptographic — collision-resistance at 2**-48 is far
+    beyond the ~10 setup-time comparisons this guard makes.
+    """
+    a = np.ascontiguousarray(arr)
+    h = hashlib.blake2b(a.tobytes(), digest_size=6)
+    return float(int.from_bytes(h.digest(), "big"))
+
+
+def name_digest48(names) -> float:
+    """Order-sensitive, UNAMBIGUOUS digest of a sequence of names.
+
+    Uses a NUL separator, which cannot occur in a Python identifier or any
+    legoESM field name, so ``["a,b", "c"]`` and ``["a", "b,c"]`` cannot
+    collide.  A plain ``",".join`` COULD (codex 2026-07-29, minor 5): those
+    two lists have the same length, so a count check does not separate them
+    either.
+    """
+    joined = "\x00".join(names).encode()
+    return float(int.from_bytes(
+        hashlib.blake2b(joined, digest_size=6).digest(), "big"))
+
+
+def schema_fingerprint(names, n_dev, dtype_kinds=(), ndims=()) -> np.ndarray:
+    """Fixed-shape schema digest gathered ONCE before the per-field loop.
+
+    Covers the field-name list (order-sensitive), the count, the x64 flag,
+    ``n_dev``, and -- critically -- the per-field DTYPE CLASS and NDIM.
+
+    The dtype/ndim terms are not cosmetic.  :func:`broadcast_checked` routes
+    exact dtypes to a 1-value digest and float dtypes to a 3-moment
+    fingerprint, and its struct entry depends on ndim.  If the schema gate
+    did not cover those, a field that is bool on one process and float on
+    another would PASS the gate and then deadlock inside the per-field
+    gather with mismatched payloads.  Catching it here converts that hang
+    into a clean symmetric RuntimeError (codex 2026-07-29, blocker 2).
+
+    ``dtype_kinds``/``ndims`` default to empty for callers that have not yet
+    resolved the arrays; passing them is strongly preferred.
+    """
+    return np.array(
+        [float(len(names)),
+         name_digest48(names),
+         float(bool(jax.config.jax_enable_x64)),
+         float(n_dev),
+         name_digest48([str(k) for k in dtype_kinds]),
+         name_digest48([str(int(n)) for n in ndims])],
+        dtype=np.float64)
+
+
+def in_jax_trace() -> bool:
+    """True when the caller runs inside a JAX trace (``jit``/``scan``/``vmap``).
+
+    The host-side gates below call ``multihost_utils.process_allgather``, which
+    is an EAGER utility: it ``device_put``s its payload per addressable device.
+    Under an active trace those puts are staged into the jaxpr and come back as
+    tracers, so ``make_array_from_single_device_arrays`` is handed tracers and
+    raises — every multi-process lat-lon SPMD run died this way once the step
+    was wrapped in ``lax.scan``/``jax.jit`` (#1405, follow-up to #1362).
+
+    TWO LIMITATIONS, stated because a reader will otherwise assume they are
+    covered (both raised by codex adversarial review of this change, both
+    accepted deliberately — the alternative is a lane that cannot run at all):
+
+    1. The skip is symmetric only as long as every process reaches this call
+       in the SAME transform state, which is the SPMD lockstep property the
+       gate itself exists to enforce.  If one rank called the step eagerly
+       while another traced it, the eager rank would now BLOCK in
+       ``process_allgather`` instead of its peer crashing.  That divergence is
+       already fatal today (the traced rank dies here), so this trades a
+       guaranteed crash on every multi-process traced run for a hang in an
+       already-divergent one.  It is NOT a proof of symmetry.
+    2. Coverage IS lost on a lane that is only ever traced.  The build-time
+       gates (``_agree_spmd_entry``) agree the model/mesh/config; the per-CALL
+       payload — state pytree schema, ``phys_state``/forcing presence and its
+       schema — is agreed ONLY here, and under a trace it now goes unchecked.
+
+    The trace-safe design that would fix both (stage the digest comparison as
+    a mesh collective inside the traced program instead of a host allgather)
+    needs a real multi-process rig to validate and is deliberately left as
+    follow-up rather than written blind — see #1405.
+
+    ``jax.core.trace_state_clean`` was removed from the public ``jax.core`` in
+    jax 0.7 and survives only as ``jax._src.core``, so this reads the private
+    module.  The ``except`` returns False — i.e. the gate RUNS and the traced
+    lane crashes loudly again — deliberately: for a correctness gate a loud
+    crash beats a silent skip.  ``tests/unit/test_geometry_consistency_trace_
+    gate.py`` asserts this returns True inside ``jax.jit`` AND inside
+    ``lax.scan``, so a JAX version that moves the symbol turns CI red first.
+    """
+    try:
+        from jax._src import core as _jax_core
+        return not _jax_core.trace_state_clean()
+    except Exception:  # pragma: no cover - JAX internal moved; test goes red
+        return False
+
+
+def assert_flags_agree(names, values, *, context: str) -> None:
+    """Raise unless every process agrees on a tuple of rank-local CONFIG flags.
+
+    Call this BEFORE any rank-local ``raise`` that inspects per-process
+    config.  Otherwise one process can reject its config and exit while its
+    peers proceed into a collective and block forever — a collective-ORDER
+    violation whose symptom (hang vs backend error) is backend-dependent
+    (codex 2026-07-29, blocker 1).
+
+    ``names`` and ``values`` must be STATIC tuples written at the call site,
+    so the payload length is fixed by the code path rather than by data.
+
+    No-op under a JAX trace — see :func:`in_jax_trace` (#1405).
+    """
+    if jax.process_count() <= 1 or in_jax_trace():
+        return
+    from jax.experimental import multihost_utils
+
+    payload = np.array(
+        [float(len(values)), name_digest48(names),
+         *(float(v) for v in values)], dtype=np.float64)
+    gathered = multihost_utils.process_allgather(payload)
+    if not bool(np.all(gathered == gathered[0])):
+        raise RuntimeError(
+            f"{context}: per-process CONFIG differs across processes "
+            f"(flags {list(names)} -> gathered {gathered.tolist()}). Every "
+            f"process must be built from the same config; refusing before "
+            f"any rank-local rejection so the failure is symmetric rather "
+            f"than a hang.")
+
+
+def assert_schema_agrees(names, n_dev, *, context: str, arrays=None) -> None:
+    """Raise unless every process agrees on the geometry field SCHEMA.
+
+    ``names`` must be an ORDERED sequence — the per-field
+    :func:`broadcast_checked` calls that follow are matched positionally
+    across processes, so a reordering is itself a divergence worth catching.
+
+    Pass ``arrays`` (the per-name arrays, same order) so the gate also covers
+    each field's DTYPE CLASS and NDIM.  Those decide the per-field payload
+    SHAPE in :func:`broadcast_checked`, so leaving them out lets a
+    bool-vs-float disagreement slip past this gate and deadlock in the
+    per-field gather instead of raising here.
+
+    No-op when ``jax.process_count() == 1``.
+    """
+    if jax.process_count() <= 1:
+        return
+    from jax.experimental import multihost_utils
+
+    names = list(names)
+    if arrays is None:
+        kinds, ndims = (), ()
+    else:
+        # Read dtype/ndim from array METADATA, never via np.asarray: a jax
+        # array exposes both without materialising, so forcing a host copy
+        # here would add a device sync per field AND could itself fail
+        # (transfer error / OOM) BEFORE the collective below — reintroducing
+        # the very "one rank exits while a peer blocks" hazard this gate
+        # exists to remove (codex round-2 minor). `broadcast_checked` does
+        # the single real materialisation later.
+        #
+        # `_dtype_kind_and_ndim` also survives a plain Python scalar, which a
+        # bare `a.dtype` read did not (codex round-3, minor 3): no production
+        # caller passes one today, but an AttributeError HERE would be a
+        # rank-local raise BEFORE the collective, i.e. a hang rather than a
+        # clear failure.
+        described = [_dtype_kind_and_ndim(a) for a in arrays]
+        kinds = [d[0] for d in described]
+        ndims = [d[1] for d in described]
+    gathered = multihost_utils.process_allgather(
+        schema_fingerprint(names, n_dev, kinds, ndims))
+    if not bool(np.all(gathered == gathered[0])):
+        raise RuntimeError(
+            f"{context}: the band-geometry SCHEMA differs across processes "
+            f"(field list / x64 setting / device count / per-field dtype "
+            f"class / ndim — gathered {gathered.tolist()}). Fix the "
+            f"per-process config before sharding; the per-field checks "
+            f"assume one schema.")
+
+
+def broadcast_checked(arr, name: str, *, context: str) -> np.ndarray:
+    """Verify ``arr`` agrees across processes, then broadcast process 0's bytes.
+
+    Multi-process: returns a host ``np.ndarray`` that is bit-identical on
+    every process, safe to hand to a replicated ``device_put``.
+    Single-process: returns ``arr`` ITSELF, untouched — no collectives, no
+    host round trip, no dtype/weak-type change.
+
+    The fingerprint compares structural entries exactly; value entries
+    EXACTLY for integer/bool arrays and to ``rtol=1e-5`` for float arrays.
+
+    Integer/bool arrays (masks, index tables) are exact data, not autotuned
+    arithmetic: their BYTES are fingerprinted so a positional difference is
+    caught.  Moment-only compares are blind to a permutation — a bool mask's
+    ``(sum, sumsq, absmax)`` is identical for every arrangement with the same
+    true-count (codex round-5).  A mask that genuinely differs across
+    processes means different wet domains = different physics: refusing is
+    the correct outcome, not a false alarm.
+
+    Residual, documented: a float divergence preserving sum, sum-of-squares
+    AND absmax to ``rtol`` is not detected.  Band grids are analytic in
+    lat/lon, so any real inconsistency moves those moments.
+    """
+    # EARLY return, BEFORE np.asarray: single process has nothing to compare,
+    # and converting here would force a device->host->device round trip and
+    # strip weak-type metadata on a 1-process mesh. The ocean lane already
+    # held host arrays so it was unaffected, but the atmosphere lane passes
+    # `jnp.stack` results straight in and WAS regressed by an unconditional
+    # conversion (codex 2026-07-29, major 3). Return the caller's object
+    # untouched.
+    if jax.process_count() <= 1:
+        return arr
+    from jax.experimental import multihost_utils
+
+    host = np.asarray(arr)
+    flat = host.ravel()
+    is_exact = host.dtype.kind in "biu"
+    # FIXED-WIDTH payloads (see _STRUCT_WIDTH/_VALS_WIDTH): the gathered shape
+    # must never depend on rank-local data, or two processes can enter this
+    # collective with different shapes and hang. Shape is folded in as a
+    # digest rather than splatted, so an ndim difference cannot change the
+    # length either.
+    struct = np.zeros(_STRUCT_WIDTH, dtype=np.float64)
+    struct[0] = float(host.ndim)
+    struct[1] = float(np.dtype(host.dtype).num)
+    struct[2] = float(1.0 if is_exact else 0.0)
+    struct[3] = float(host.size)
+    struct[4] = name_digest48([str(d) for d in host.shape])
+    vals = np.zeros(_VALS_WIDTH, dtype=np.float64)
+    if is_exact:
+        vals[0] = content_hash48(host)
+    else:
+        finite = flat[np.isfinite(flat)]
+        f64 = finite.astype(np.float64)
+        # Non-finite COUNT is structural: a NaN appearing on one process only
+        # must not be averaged away by the moment compare below.
+        struct[5] = float(flat.size - finite.size)
+        vals[0] = float(f64.sum()) if f64.size else 0.0
+        vals[1] = float((f64 * f64).sum()) if f64.size else 0.0
+        vals[2] = float(np.abs(f64).max()) if f64.size else 0.0
+
+    g_struct = multihost_utils.process_allgather(struct)
+    g_vals = multihost_utils.process_allgather(vals)
+    struct_ok = bool(np.all(g_struct == g_struct[0]))
+    if is_exact:
+        vals_ok = bool(np.all(g_vals == g_vals[0]))
+    else:
+        vals_ok = bool(np.allclose(g_vals, g_vals[0],
+                                   rtol=_FLOAT_RTOL, atol=0.0))
+    if not (struct_ok and vals_ok):
+        raise RuntimeError(
+            f"{context}: geometry field {name!r} DIVERGES across processes "
+            f"(struct_ok={struct_ok}, vals_ok={vals_ok}, "
+            f"exact_dtype={is_exact}, gathered={g_vals.tolist()}) — a real "
+            f"config/grid inconsistency, not autotune noise; refusing to "
+            f"broadcast process 0 over it.")
+    return np.asarray(multihost_utils.broadcast_one_to_all(host))
+
+
+# --- assert-free sharded puts + per-band gates (2026-08-03, ocean walls) ----
+# Three stacked multicontroller walls were found on the ocean lane (codex
+# r14-r19; PR #1457): (1) broadcast_one_to_all of a band stack lowers to an
+# [n_processes, stack] psum program (nd x 849 MB at LL2304 L20 — 81.5 GB at
+# 96 procs); (2) jax.device_put of a NUMPY array onto an all-process
+# sharding internally runs multihost_utils.assert_equal on the FULL array
+# ([n_proc, field] landing on ONE device: fits under an 80 GB A100 up to
+# ~64 procs, dies at 96 — jax _src/dispatch.py::_device_put_sharding_impl);
+# (3) a concrete sharded-global array captured by an OUTER trace (jit-of-
+# jit) becomes an MLIR constant whose value cannot be fetched for
+# non-addressable arrays. The helpers below remove (1) and (2) — (3) is the
+# callers' aux-threading contract, see make_sharded_ocean_step.
+
+def band_fingerprint(host, n_bands):
+    """Per-band fingerprint of a band-STACKED field (leading axis n_bands).
+
+    PREREQUISITE: ``n_bands`` (and each field's dtype class / shape) must
+    already be schema-gated across processes (:func:`assert_schema_agrees`)
+    — the payload widths depend on it, and mismatched widths would hang the
+    allgather rather than raise.
+
+    Exact dtypes (int/bool/uint): one positional 48-bit byte digest per
+    band. Floats: per-band ``[sum, sum_of_squares, absmax]`` of finite
+    entries plus per-band non-finite counts folded into ``struct``.
+    Per-band (not whole-array) because each process's OWN bytes become the
+    live inputs for the bands it owns under the assert-free put: a
+    band-local drift must not hide in a whole-array sum (codex r14).
+    DOCUMENTED RESIDUALS: a within-band float change preserving all three
+    moments to rtol, and non-finite entries changing position/kind at a
+    fixed per-band count, pass the float gate.
+    """
+    host = np.asarray(host)
+    if host.shape[0] != n_bands:
+        raise ValueError(
+            f"band_fingerprint: leading axis {host.shape[0]} != n_bands "
+            f"{n_bands}")
+    is_exact = host.dtype.kind in "biu"
+    struct = [float(host.ndim), *map(float, host.shape),
+              float(np.dtype(host.dtype).num)]
+    if is_exact:
+        vals = np.array([content_hash48(host[b]) for b in range(n_bands)],
+                        dtype=np.float64)
+    else:
+        per_band = []
+        for b in range(n_bands):
+            flat = host[b].ravel()
+            finite = flat[np.isfinite(flat)]
+            f64 = finite.astype(np.float64)
+            struct.append(float(flat.size - finite.size))
+            per_band.extend([
+                float(f64.sum()) if f64.size else 0.0,
+                float((f64 * f64).sum()) if f64.size else 0.0,
+                float(np.abs(f64).max()) if f64.size else 0.0,
+            ])
+        vals = np.array(per_band, dtype=np.float64)
+    return np.array(struct, dtype=np.float64), vals, is_exact
+
+
+def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=None):
+    """True iff every process's :func:`band_fingerprint` matches process 0's."""
+    if rtol is None:
+        rtol = _FLOAT_RTOL
+    struct_ok = bool(np.all(g_struct == g_struct[0]))
+    if is_exact:
+        vals_ok = bool(np.all(g_vals == g_vals[0]))
+    else:
+        vals_ok = bool(np.allclose(g_vals, g_vals[0], rtol=rtol, atol=0.0))
+    return struct_ok and vals_ok
+
+
+def checked_shard_put(arr, name, sharding, *, context, n_bands):
+    """Gate a band-stacked field per band, then put WITHOUT broadcast or
+    jax's whole-array device_put assert (walls 1+2 above).
+
+    Single-process: plain ``jax.device_put`` — byte-unchanged, no host
+    round trip. Multi-process: per-band fingerprint gate (symmetric raise
+    on real divergence), then ``jax.make_array_from_callback`` hands each
+    process exactly its addressable slabs. Cross-process byte-identity of
+    NON-owned bands is not required — owned bands are the only bytes that
+    reach any device, and their drift is bounded by the gate.
+    """
+    if jax.process_count() <= 1:
+        return jax.device_put(arr, sharding)
+    from jax.experimental import multihost_utils
+
+    host = np.asarray(arr)
+    struct, vals, is_exact = band_fingerprint(host, n_bands)
+    g_struct = multihost_utils.process_allgather(struct)
+    g_vals = multihost_utils.process_allgather(vals)
+    if not band_fingerprints_agree(g_struct, g_vals, is_exact):
+        raise RuntimeError(
+            f"{context}: band-stacked field {name!r} DIVERGES across "
+            f"processes (exact_dtype={is_exact}, "
+            f"gathered={g_vals.tolist()}) — a real config/grid "
+            f"inconsistency, not autotune noise; refusing to shard it.")
+    return jax.make_array_from_callback(
+        host.shape, sharding, lambda idx: host[idx])
+
+
+def assert_pytree_bytes_equal(tree, what):
+    """Cheap multi-process replacement for the per-leaf assert_equal that
+    :func:`checked_shard_put`-style puts bypass on NON-band inputs (state /
+    forcing pytrees): one 48-bit digest per array leaf, one tiny allgather,
+    symmetric raise on mismatch. No-op single-process.
+    """
+    if jax.process_count() <= 1:
+        return
+    from jax.experimental import multihost_utils
+
+    leaves = [x for x in jax.tree_util.tree_leaves(tree)
+              if hasattr(x, "ndim")]
+    vals = np.array([content_hash48(np.asarray(x)) for x in leaves],
+                    dtype=np.float64)
+    g = multihost_utils.process_allgather(vals)
+    if not bool(np.all(g == g[0])):
+        bad = [i for i in range(len(leaves))
+               if not bool(np.all(g[:, i] == g[0, i]))]
+        raise RuntimeError(
+            f"{what}: array leaves {bad} differ across processes (48-bit "
+            f"byte digests disagree) — the per-process inputs are NOT "
+            f"identical, which jax's device_put assert would have refused. "
+            f"Fix the per-process build before sharding.")
+
+
+def addressable_shard_put(arr, sharding):
+    """Ungated assert-free put (walls 1+2) for inputs whose cross-process
+    consistency the CALLER has already gated (state/forcing pytrees via
+    :func:`assert_pytree_bytes_equal`). Single-process: plain device_put."""
+    if jax.process_count() <= 1:
+        return jax.device_put(arr, sharding)
+    host = np.asarray(arr)
+    return jax.make_array_from_callback(
+        host.shape, sharding, lambda idx: host[idx])

exec
/usr/bin/bash -lc 'git diff --unified=60 e13de3afe''^1 e13de3afe -- packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py; git diff --unified=30 e13de3afe''^1 e13de3afe -- tests/ocean/unit/test_sharded_geom_fingerprint.py' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 378ms:
diff --git a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
index 3059ddbd1..f9e7ecb7e 100644
--- a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
+++ b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
@@ -1,108 +1,112 @@
 r"""Multi-GPU SPMD step for the lat-lon C-grid ocean (latlon / tripole).
 
 Wraps ``LatLonCGridOceanModel.step`` in ``jax.shard_map`` over a 1-D ``"lat"``
 device mesh, so the ocean state is partitioned by latitude band across devices
 (eORCA025 ¼° needs ~141 GiB — fits 32 GiB GPUs at N>=5 bands). The dynamics
 operators are element-wise / local-stencil and the in-step halo exchange routes
 through the SPMD band body once ``activate_latlon_spmd_halo(mesh)`` is armed; the
 barotropic PCG reductions already dispatch to ``batch_psum_spmd`` on ``"lat"``
 (``barotropic_common._global_dot_batch``). This is the ocean analogue of the
 cubed-sphere ``parallel.sharded_dynamics.make_sharded_step``.
 
 Architecture (the two non-trivial pieces — see ``omip-multinode-spmd-scope``):
 
-* GRID = **band-stacked, P("lat")-SHARDED** (Structure A, sharded since
-  #1370-iii). The ``N`` band geometries are built host-side
-  (``build_band_grids``), their ARRAY fields are ``jnp.stack``-ed over a
-  leading band axis, sharded ``P("lat")`` so each device holds ONLY its own
-  band's slab, and the in-``shard_map`` body reads its local slab at
-  ``[0]``. This AVOIDS staggered-sharding the grid itself (the v-row is
-  ``n_lat+1``, coprime with ``n_lat`` for ``N>1``). The
+* GRID = **replicated-stacked, indexed** (Structure A). The ``N`` band
+  geometries are built host-side (``build_band_grids``), their ARRAY fields are
+  ``jnp.stack``-ed into a replicated pytree, and the in-``shard_map`` body picks
+  its own band by ``jax.lax.axis_index("lat")``. The grid is 2-D/small so
+  replication is cheap, and this AVOIDS staggered-sharding the grid (the v-row
+  is ``n_lat+1``, coprime with ``n_lat`` for ``N>1``). The
   ``LatLonCGridGeometry`` SCALAR fields (``n_lat``, ``n_lon``, ``radius``,
   ``dlon``, ``dlat``, ``fold``) stay STATIC python values — they gate trace-time
   ``if``\ s and ``jnp.zeros((n_lat, ...))`` shape builds, so a traced ``n_lat``
   would break the step. Only the ``jax.Array`` fields are stacked/indexed; each
   band geometry is rebuilt as ``template._replace(**indexed_arrays)`` so the
   scalars come from a band template (all bands share ``n_lat_local = n_lat/N``).
 
 * STATE = cell fields shard ``P("lat")``; the STAGGERED ``v`` / ``v_mask``
   (shape ``(n_lat+1, n_lon, ...)``) use the **band-local v-faces** layout
   (Structure B). The sharded state carries ``v_lower = v[0:n_lat]`` (``n_lat``
   rows, ``P("lat")`` — the top pole row ``v[n_lat]`` is a 0 wall on the regular
   grid and is dropped). In-body, band ``r`` reconstructs its ``nl+1`` v-faces by
   ``ppermute``-ing band ``r+1``'s first ``v_lower`` row (= global ``v[e]``) up as
   its north boundary row; the north band's non-target receives the pole-wall 0.
   After ``model.step`` the result's ``nl+1`` v is converted back to the ``nl``-row
   ``v_lower`` representation (drop the boundary row, owned by band ``r+1``).
 
 Correctness is gated by ``tests/parallel/test_latlon_ocean_spmd_step.py`` (N-step
 tol-match vs the single-device step). Pure-dynamics (no host-loop coupling): the
 OMIP host post-step updates (ice / SSS-restore / geothermal) are NOT inside this
 wrapper — they need separate gather/scatter or jax-porting (see
 [[omip-multinode-spmd-scope]]).
 """
 from __future__ import annotations
 
 import jax
 import jax.numpy as jnp
 import numpy as np
+from legoesm.parallel.geometry_consistency import (
+    FLAG_ABSENT, addressable_shard_put, assert_flags_agree,
+    assert_pytree_bytes_equal, assert_schema_agrees, checked_shard_put,
+    coerce_bool, coerce_count, config_digest48, name_digest48,
+    tree_schema_digest48)
 from jax.sharding import NamedSharding, PartitionSpec as P
 
 try:                                   # JAX >= 0.8 exposes shard_map at top level
     from jax import shard_map
 except ImportError:                    # pragma: no cover - JAX < 0.8 fallback
     from jax.experimental.shard_map import shard_map
 
 from legoesm.parallel.latlon_spmd import (
     activate_latlon_spmd_halo,
     latlon_band_perms,
     reconstruct_vface_lower_multi,
 )
 
 # Per-device band grids: the SPMD wrapper REUSES the tested band slicers from
 # ``legoesm.parallel.latlon_mpi`` rather than a bespoke slice — they already
 # handle the staggered v-row (n_lat+1), the bipolar-fold localization (the
 # northernmost band keeps the active ``fold``; interior bands get
 # ``fold_j=cap_j=-1`` so the pad_ns_* operators fall through to the band halo
 # instead of folding their own last row), AND keep ``total_area`` GLOBAL (the
 # area-weighted-mean denominator must not become band-local).  Build a band
 # layout with ``make_latlon_band_layout(rank, n_dev, n_lat, n_lon, fold)`` then
 # call ``slice_cgrid_geometry_to_band(geom, layout)`` (curvilinear tripole /
 # eORCA025) or ``slice_latlon_grid_to_band(grid, layout)`` (regular lat-lon ½°).
 # latlon_mpi defers its ``from mpi4py import MPI`` to function scope, so these
 # pure slicers import without requiring mpi4py.  See omip-multinode-spmd-scope.
 
 # The LatLonCGridGeometry fields that MUST stay static python scalars (they gate
 # trace-time ``if``-s + ``jnp.zeros`` shape builds): never stacked/indexed.  The
 # ``fold`` is a FoldDescriptor whose ``is_active``/``fold_j``/``cap_j`` are python
 # ints/bools used at trace time; on a regular grid the fold is inactive and
 # identical on every band, so taking it from the band template is exact.  Every
 # OTHER LatLonCGridGeometry field is a ``jax.Array`` and IS stacked/indexed.
 _GEOM_STATIC_FIELDS = frozenset(
     {"n_lat", "n_lon", "radius", "dlon", "dlat", "fold"}
 )
 
 # State fields that are staggered on the v-grid (leading dim n_lat+1): carried in
 # the sharded state as ``v_lower = field[0:n_lat]`` (P("lat")) and reconstructed
 # to the band's nl+1 faces in-body.  All OTHER array state fields are cell/u
 # leading-dim n_lat and shard P("lat") directly.
 _V_STAGGERED_STATE_FIELDS = ("v", "v_mask")
 
 
 def _geom_array_field_names(geom):
     """The ``jax.Array`` field names of a ``LatLonCGridGeometry`` (everything
     except the static scalars + the ``fold`` descriptor).  Order-stable
     (the NamedTuple field order) so the host stack and the in-body index agree."""
     names = []
     for name in geom._fields:
         if name in _GEOM_STATIC_FIELDS:
             continue
         val = getattr(geom, name)
         if isinstance(val, (jax.Array, np.ndarray)):
             names.append(name)
     return names
 
 
 def _lat_spec(x):
     """Lat-band PartitionSpec for an array leaf: shard axis 0, replicate rest."""
     nd = int(getattr(x, "ndim", np.ndim(x)))
@@ -146,879 +150,857 @@ def _step_body(model, state, dt, *, grid, vertex_mask,
         new_state = model._ab2_step(
             state, dt, freshwater=freshwater,
             surface_forcing=surface_forcing, sponge=sponge,
             grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds)
     else:
         new_state = model._step_impl(
             state, dt, freshwater=freshwater,
             surface_forcing=surface_forcing, sponge=sponge,
             grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds)
     if model.config.polar_filter.use_polar_filter:
         new_state = model._apply_polar_filter(new_state, dt, grid=grid)
     if model.config.freeze_floor:
         new_state = model._apply_freeze_floor(new_state)
     if getattr(model.config, "ew_cyclic_overlap", False):
         new_state = model._apply_ew_cyclic_overlap(new_state)
     return new_state
 
 
 def build_band_grids(grid, n_devices: int):
     """Build the ``n_devices`` UNIFORM lat-band geometries for the SPMD step,
     reusing the tested MPI band slicers (no bespoke re-derivation).
 
     Requires ``grid.n_lat % n_devices == 0`` so every band has the same leading
     shape — a ``shard_map`` body is ONE program, so the per-band grids must be
     structurally identical (only the values + the north band's active fold
     differ). ``make_latlon_band_layout`` would otherwise hand the first
     ``n_lat % n_devices`` ranks one extra row.
 
     Returns a list of ``n_devices`` band-local ``LatLonCGridGeometry`` (rank 0 =
     south band ... rank ``n_devices-1`` = north band). The slicer keeps
     ``total_area`` GLOBAL on every band (area-weighted-mean denominator), slices
     v/q rows ``[s:e+1]`` (the shared staggered boundary face), and localizes the
     bipolar fold — active only on the north band, ``fold_j=cap_j=-1`` sentinel on
     interior bands so ``fold_is_local()`` is False there and the pad_ns_*
     operators fall through to the SPMD band halo. See omip-multinode-spmd-scope.
 
     ``latlon_mpi`` defers ``from mpi4py import MPI`` to function scope, so these
     pure slicers import without requiring mpi4py.
     """
     from legoesm.parallel.latlon_mpi import (
         make_latlon_band_layout,
         slice_cgrid_geometry_to_band,
     )
     n_lat = int(grid.n_lat)
     if n_devices < 1:
         raise ValueError(f"n_devices must be >= 1, got {n_devices}")
     if n_lat % n_devices != 0:
         raise ValueError(
             f"SPMD lat-band requires n_lat ({n_lat}) divisible by n_devices "
             f"({n_devices}) so every band is uniform (one shard_map program). "
             f"Pick n_devices among the divisors of {n_lat}.")
     fold = getattr(grid, "fold", None)
     return [
         slice_cgrid_geometry_to_band(
             grid,
             make_latlon_band_layout(r, n_devices, n_lat, int(grid.n_lon), fold))
         for r in range(n_devices)
     ]
 
 
-def _content_hash48(arr) -> float:
-    """48-bit content digest of ``arr``'s bytes, exactly representable in f64.
-
-    Used to compare EXACT-dtype arrays (masks, index tables) across
-    processes: unlike moment fingerprints, a byte digest is positional, so a
-    permutation or a two-cell flip cannot cancel. 48 bits keeps the value
-    under 2**53 so it survives the float64 ``process_allgather`` payload
-    exactly. Not cryptographic — collision-resistance at 2**-48 is far
-    beyond the ~10 setup-time comparisons this guard makes.
-    """
-    import hashlib
-
-    a = np.ascontiguousarray(arr)
-    h = hashlib.blake2b(a.tobytes(), digest_size=6)
-    return float(int.from_bytes(h.digest(), "big"))
-
-def geom_band_fingerprint(host, n_bands):
-    """Low-memory cross-process fingerprint of a band-STACKED geometry field.
-
-    ``host`` has leading axis ``n_bands`` (the per-device band stack). The
-    fingerprint is PER BAND (codex 2026-08-03 r14: whole-array moments let a
-    band-local drift hide in the global sum once each process's own bytes
-    become live computation inputs):
-
-    * exact dtypes (int/bool/uint — masks, index tables): one positional
-      48-bit byte digest per band (a permutation or two-cell flip within a
-      band cannot cancel);
-    * float dtypes: per-band ``[sum, sum_of_squares, absmax]`` of the finite
-      entries in float64, plus per-band non-finite counts in ``struct``.
-      DOCUMENTED RESIDUALS: a within-band float change preserving all
-      three moments to rtol is not detected, and non-finite entries that
-      change POSITION or kind (nan vs inf) with an unchanged per-band
-      count also pass; band grids are analytic in lat/lon, so any real
-      inconsistency moves the moments.
-
-    Returns ``(struct, vals, is_exact)`` as float64 arrays safe for
-    ``process_allgather``.
-    """
-    import numpy as _np
-
-    host = _np.asarray(host)
-    assert host.shape[0] == n_bands, (host.shape, n_bands)
-    is_exact = host.dtype.kind in "biu"
-    struct = [float(host.ndim), *map(float, host.shape),
-              float(_np.dtype(host.dtype).num)]
-    if is_exact:
-        vals = _np.array([_content_hash48(host[b]) for b in range(n_bands)],
-                         dtype=_np.float64)
-    else:
-        per_band = []
-        for b in range(n_bands):
-            flat = host[b].ravel()
-            finite = flat[_np.isfinite(flat)]
-            f64 = finite.astype(_np.float64)
-            struct.append(float(flat.size - finite.size))
-            per_band.extend([
-                float(f64.sum()) if f64.size else 0.0,
-                float((f64 * f64).sum()) if f64.size else 0.0,
-                float(_np.abs(f64).max()) if f64.size else 0.0,
-            ])
-        vals = _np.array(per_band, dtype=_np.float64)
-    return _np.array(struct, dtype=_np.float64), vals, is_exact
-
-
-def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=1e-5):
-    """True iff every process's fingerprint matches process 0's.
-
-    ``g_struct``/``g_vals`` are the ``process_allgather``-ed outputs of
-    :func:`geom_band_fingerprint` (leading axis = process). Structure and
-    exact-dtype digests compare EXACTLY; float moments to ``rtol``.
-    """
-    import numpy as _np
-
-    struct_ok = bool(_np.all(g_struct == g_struct[0]))
-    if is_exact:
-        vals_ok = bool(_np.all(g_vals == g_vals[0]))
-    else:
-        vals_ok = bool(_np.allclose(g_vals, g_vals[0], rtol=rtol, atol=0.0))
-    return struct_ok and vals_ok
-
-
-
-def _schema_fingerprint(names, n_dev) -> np.ndarray:
-    """Fixed-shape schema digest: field-name list, count, x64 flag, n_dev.
-
-    Gathered ONCE before the per-field loop so a process-dependent field
-    selection is caught by a collective every process reaches, instead of
-    desynchronizing the per-field gathers (codex round-5 findings 3/4).
-    """
-    import hashlib
-
-    joined = ",".join(names).encode()
-    digest = float(int.from_bytes(
-        hashlib.blake2b(joined, digest_size=6).digest(), "big"))
-    return np.array(
-        [float(len(names)), digest, float(bool(jax.config.jax_enable_x64)),
-         float(n_dev)], dtype=np.float64)
-
-
 def _build_band_vertex_masks(model, n_dev):
     """Per-band vertex masks (n_lat/N+1, n_lon+1): SLICE the model's primed GLOBAL
     vertex mask ``[s : e+1]`` per band (the v/q-row stagger ``slice_cgrid_geometry_
     to_band`` uses for ``area_q``).
 
     The slice — NOT a per-band ``compute_vertex_mask`` — is what makes the
     band-cut rows correct.  The global ``model._vertex_mask`` already encoded the
     four-cell product at EVERY vertex row including the interior cut rows (it saw
     the neighbour-band cells), so band ``r``'s vertex rows ``[r*nl : r*nl+nl+1]``
     are exact.  A per-band ``compute_vertex_mask`` would instead pad the band's
     cell mask with a POLE WALL at the cut (``pad_with_pole_bc_lat`` has no SPMD
     branch — it falls to the local constant pad), delivering a wrong (walled)
     north/south boundary vertex row at interior cuts — the "one row off at the
     cut" failure the ``compute_vertex_mask`` docstring warns about.
 
     Requires the model's vertex mask to be primed (``model._ensure_vertex_mask``
     on the concrete initial state; ``step`` does this on the first call).
     """
     vmask = model._vertex_mask
     if vmask is None:
         raise RuntimeError(
             "make_sharded_ocean_step: the model vertex-mask cache is not "
             "primed.  Call model._ensure_vertex_mask(state) (or model.step) "
             "on the concrete initial state before building the sharded step.")
     vmask = np.asarray(vmask)                   # (n_lat+1, n_lon+1)
     n_lat = vmask.shape[0] - 1
     nl = n_lat // n_dev
     # v/q-row stagger: band r owns global vertex rows [r*nl : r*nl+nl+1] (the
     # shared interior boundary vertex row appears in band r AND band r+1) — the
     # same [s:e+1] slice slice_cgrid_geometry_to_band applies to area_q.
     return [jnp.asarray(vmask[r * nl: r * nl + nl + 1]) for r in range(n_dev)]
 
 
-def _addressable_shard_put(arr, sharding):
-    """Put an array onto a (possibly multi-process) sharding WITHOUT jax's
-    whole-array cross-process ``assert_equal``.
-
-    Under multicontroller, ``jax.device_put(numpy_array, sharding)`` calls
-    ``multihost_utils.assert_equal`` on the FULL array
-    (jax _src/dispatch.py::_device_put_sharding_impl) — a
-    ``process_allgather`` whose output is ``[n_processes, *shape]`` on one
-    device: at LL2304 L20 one 3-D field is 849 MB, so 96 processes fetch
-    81.5 GB > an 80 GB A100 (the wall that killed oc @96/@128, jobs
-    26644681/26644682, AFTER the geometry-broadcast fix; @64 = 54.3 GB
-    just fit, which is why smaller ladders never saw it).
-
-    Single-process: the historical ``jax.device_put``, byte-unchanged and
-    with no host round-trip (codex r17 — the input may already be a jax
-    device array). Multicontroller: ``jax.make_array_from_callback``
-    supplies each process's addressable shards directly — no consistency
-    collective. The ``device_put`` bit-identity CONTRACT is preserved by
-    the callers' cheap exact-hash gate (:func:`assert_pytree_bytes_equal`)
-    instead of jax's full-array allgather.
-    """
-    if jax.process_count() <= 1:
-        return jax.device_put(arr, sharding)
-    host = np.asarray(arr)
-    return jax.make_array_from_callback(
-        host.shape, sharding, lambda idx: host[idx])
-
-
-def assert_pytree_bytes_equal(tree, what):
-    """Cheap multi-process replacement for the per-leaf ``assert_equal``
-    that :func:`_addressable_shard_put` bypasses (codex r17 HIGH-1).
+# Ordered flag names for the ocean MESH+TREE gate used by the scatter/gather
+# bridges and by the returned SPMD callable. STATIC tuple: fixed width.
+_OCEAN_MESH_ENTRY_FLAGS = (
+    "has_mesh", "n_dev", "n_axes", "axis_names", "axis_sizes",
+    "has_tree", "tree_schema",
+)
 
-    Hashes every array leaf's BYTES (48-bit positional digest — the same
-    exactness as ``device_put``'s contract) into one small vector,
-    allgathers it, and refuses on any cross-process mismatch. Cost is one
-    tiny collective + a host-side hash pass, independent of process count
-    — vs jax's [n_processes, full_array] allgather.
 
-    No-op single-process. Symmetric: every process hashes the same leaves
-    in the same order, so all raise or none.
+def _ocean_mesh_axis_terms(mesh):
+    """``(axis_names, axis_sizes)`` term lists; never raises."""
+    if mesh is None:
+        return (), ()
+    try:
+        names = tuple(str(a) for a in mesh.axis_names)
+    except Exception:                       # pragma: no cover - defensive
+        return ("<unreadable>",), ("<unreadable>",)
+    try:
+        shape = dict(mesh.shape)
+        sizes = tuple(f"{n}={shape.get(n, '?')}" for n in names)
+    except Exception:                       # pragma: no cover - defensive
+        sizes = ("<unreadable>",)
+    return names, sizes
+
+
+def _agree_ocean_mesh_entry(mesh, tree=None, *, where: str) -> None:
+    """Agree the mesh AND a pytree LEAF SCHEMA before a scatter/gather.
+
+    #1362 round 4, blockers 5-6.  Both directions are collective here: the
+    gather's ``replicate_leaf`` compiles a jit identity with replicated
+    ``out_shardings``, and the scatter's DIRECT ``device_put`` of a full
+    global array onto a cross-process ``NamedSharding`` falls back to an
+    all-gather (documented on ``latlon_spmd.shard_leaf``) -- so the earlier
+    "SCATTER, therefore no collective" exemption was FALSE for this lane.
+    One collective runs PER LEAF, so the leaf schedule (optional fields,
+    dtypes, shapes) is rank-local data and is folded into one digest.
     """
-    if jax.process_count() <= 1:
-        return
-    from jax.experimental import multihost_utils
-
-    leaves = [x for x in jax.tree_util.tree_leaves(tree)
-              if hasattr(x, "ndim")]
-    vals = np.array([_content_hash48(np.asarray(x)) for x in leaves],
-                    dtype=np.float64)
-    g = multihost_utils.process_allgather(vals)
-    if not bool(np.all(g == g[0])):
-        bad = [i for i in range(len(leaves))
-               if not bool(np.all(g[:, i] == g[0, i]))]
-        raise RuntimeError(
-            f"{what}: array leaves {bad} differ across processes (48-bit "
-            f"byte digests disagree) — the inputs each process built are "
-            f"NOT identical, which the removed jax device_put assert "
-            f"would have refused. Fix the per-process build before "
-            f"sharding.")
+    names, sizes = _ocean_mesh_axis_terms(mesh)
+    assert_flags_agree(_OCEAN_MESH_ENTRY_FLAGS, (
+        float(mesh is not None),
+        float(mesh.devices.size if mesh is not None else 0),
+        float(len(names)),
+        name_digest48(names),
+        name_digest48(sizes),
+        float(tree is not None),
+        tree_schema_digest48(tree) if tree is not None else FLAG_ABSENT,
+    ), context=where)
 
 
 def shard_state_latlon(state, mesh):
     """Lay out a ``LatLonCGridOceanState`` for the lat-band SPMD step.
 
     Cell / u-grid array fields (leading dim ``n_lat``) shard ``P("lat")``.  The
     staggered ``v`` / ``v_mask`` (leading dim ``n_lat+1``) are carried as
     ``v_lower = field[0:n_lat]`` (``n_lat`` rows, ``P("lat")``) — the top pole row
     ``field[n_lat]`` is a 0 wall on the regular grid and is reconstructed in-body
     by the band halo.  ``None`` fields pass through.  The inverse is
     :func:`gather_state_latlon`.
 
     This is the layout the in-``shard_map`` body of :func:`make_sharded_ocean_step`
     expects; the test uses it instead of a uniform ``tree.map(P("lat"))`` (which
     fails on ``v`` because ``n_lat+1`` is not divisible by ``N``).
     """
+    # FIRST statement: a DIRECT ``device_put`` of full global arrays onto a
+    # cross-process ``NamedSharding`` is serviced by an ALL-GATHER, and one
+    # runs per leaf -- so both the mesh and the leaf SCHEDULE are rank-local
+    # inputs to a collective (#1362 round 4, blockers 5-6).
+    _agree_ocean_mesh_entry(mesh, state, where="shard_state_latlon")
     # v-carrier contract (see make_sharded_ocean_step's fold note): the TOP
     # v-face row (regular pole wall OR tripole seam/cap row) must be
     # wall-masked — the carrier drops it and reconstructs it as zero, which
     # would silently delete a LIVE seam row.  Host-side check on the
     # concrete state (this fn runs outside jit).
     assert_pytree_bytes_equal(state, "shard_state_latlon")
     vm = getattr(state, "v_mask", None)
     if vm is not None:
         import numpy as _np
 
         if _np.asarray(vm.data)[-1].any():
             raise ValueError(
                 "shard_state_latlon: the state's TOP v-face row is LIVE "
                 "(v_mask[-1] has ocean faces) — the lat-band v-carrier "
                 "drops that row and reconstructs it as the pole/cap wall "
                 "zero, which would silently delete seam velocities. "
                 "Mask the cap row (the tripole cap convention) or extend "
                 "the carrier before sharding this state.")
 
     def _shard_cell(field):
         if field is None:
             return None
         sh = NamedSharding(mesh, _lat_spec(field.data))
-        return field.replace(data=_addressable_shard_put(field.data, sh))
+        return field.replace(data=addressable_shard_put(field.data, sh))
 
     def _shard_v(field):
         if field is None:
             return None
         # Drop the TOP v-row (the north pole wall, v==0 on the regular grid) so
         # the leading dim becomes n_lat (divisible by N).  ROUND-TRIP INVARIANT
         # (codex finding): shard_state_latlon -> gather_state_latlon re-appends a
         # ZERO top row, so the round-trip is a bitwise identity ONLY when the
         # input's top v-row is already zero (a valid masked regular-grid state;
         # the pole-wall BC enforces v[n_lat]=0 every step).  An arbitrary nonzero
         # top row (e.g. a raw IC perturbation) is dropped -> reconstructed as 0:
         # CORRECT for the dynamics (the pole wall zeros it at step 1) but not a
         # bit round-trip of that one row.  NOT for a tripole north fold (raises
         # in make_sharded_ocean_step) where the top row is a live fold partner.
         nlat1 = field.data.shape[0]
         v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
         sh = NamedSharding(mesh, _lat_spec(v_lower))
-        return field.replace(data=_addressable_shard_put(v_lower, sh))
+        return field.replace(data=addressable_shard_put(v_lower, sh))
 
     updates = {}
     for name in _V_STAGGERED_STATE_FIELDS:
         updates[name] = _shard_v(getattr(state, name))
     # Every other (array) leaf: shard P("lat").  NamedTuple fields not in the
     # v-staggered set and not None get the cell sharding; None stays None.
     for name in state._fields:
         if name in _V_STAGGERED_STATE_FIELDS:
             continue
         val = getattr(state, name)
         if val is None:
             updates[name] = None
         elif hasattr(val, "data"):                 # a Field
             updates[name] = _shard_cell(val)
         else:
             # Non-Field, non-None leaf (e.g. a raw array carry like psi).
-            # ndim>=1 lat-shards via _lat_spec (1-D included); only true
-            # scalars replicate (codex r17: the old "replicate lower-rank"
-            # wording did not match _lat_spec's behaviour).
+            # Shard 2-D+ on lat, replicate lower-rank — matches shard_pytree.
             arr = jnp.asarray(val)
             spec = _lat_spec(arr) if arr.ndim >= 1 else P()
-            updates[name] = _addressable_shard_put(arr, NamedSharding(mesh, spec))
+            # ndim>=1 lat-shards via _lat_spec (1-D included); only true
+            # scalars replicate.
+            updates[name] = addressable_shard_put(arr, NamedSharding(mesh, spec))
     return state._replace(**updates)
 
 
 def shard_forcing_latlon(forcing, mesh):
     """Lay out a forcing pytree (``FreshwaterForcing`` / ``OceanSurfaceForcing``
     / ``SpongeForcing`` — or any nesting of them) for the lat-band SPMD step.
 
     Every OMIP forcing field is CELL-CENTERED ``(n_lat, n_lon[, nlev])`` (the
     cell->face wind-stress interpolation happens INSIDE the step through the
     SPMD-aware halo pads), so array leaves shard ``P("lat", None, ...)`` with
     no ``v_lower`` handling; scalars replicate; ``None`` fields pass through
     untouched (they vanish from the pytree structure, matching the specs the
     step derives).  ``forcing=None`` returns ``None``.
     """
+    # FIRST statement: the `forcing is None or mesh is None` return below
+    # SKIPS every per-leaf put, so a rank with no forcing would leave a peer
+    # blocked in one (#1362 round 4, blocker 6).
+    _agree_ocean_mesh_entry(mesh, forcing, where="shard_forcing_latlon")
     if forcing is None or mesh is None:
         return forcing
     assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")
 
     def _put(leaf):
         if leaf is None:
             return None
         arr = jnp.asarray(leaf)
-        return _addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
+        return addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
 
     return jax.tree.map(_put, forcing)
 
 
 def shard_forcing_stack_latlon(stack, mesh):
     """Lay out a STACKED per-block forcing pytree for the lat-band SPMD
     block-scan (the ``run_omip`` JRA55 lanes; see
     ``_build_jra55_block_fn`` / ``_build_jra55_block_fn_interp``).
 
     Unlike :func:`shard_forcing_latlon` (per-step, lat at axis 0), the
     block builders stack ``N`` steps / raw records along a LEADING axis,
     so the lat axis sits at position 1.  A ``(n_rec, n_lat, n_lon[, ...])``
     leaf therefore shards ``P(None, "lat", ...)`` (records replicated, the
     time index stays shard-local so the in-scan interpolation needs no
     cross-band comm); a bare ``(n_lat, n_lon)`` leaf shards ``P("lat", None)``;
     1-D metadata / scalars replicate; ``None`` and non-array leaves pass
     through.  ``mesh=None`` returns ``stack`` unchanged (serial lane).
 
     CONTRACT: rank is the ONLY signal used, so a rank-2 leaf is assumed to
     be a ``(n_lat, n_lon)`` field and is lat-sharded on axis 0.  Any future
     metadata that is genuinely rank-2 but NOT lat-major (e.g. a
     ``(n_rec, n_meta)`` table) would be silently mis-sharded — keep such
     metadata 1-D (or replicate it explicitly) before it reaches this helper.
 
     Keeping this next to :func:`shard_state_latlon` means the driver and
     the parity tests share ONE layout definition — the block-scan forcing
     stack must be laid out consistently with the state the sharded step
     carries, and a second copy would drift.
     """
+    # FIRST statement: same rank-local early-return + per-leaf put as
+    # shard_forcing_latlon (#1362 round 4, blocker 6).
+    _agree_ocean_mesh_entry(mesh, stack,
+                            where="shard_forcing_stack_latlon")
     if mesh is None:
         return stack
 
     assert_pytree_bytes_equal(stack, "shard_forcing_stack_latlon")
 
     def _put(leaf):
         if leaf is None or not hasattr(leaf, "ndim"):
             return leaf
         arr = jnp.asarray(leaf)
         if arr.ndim >= 3:
             spec = P(None, "lat", *((None,) * (arr.ndim - 2)))
         elif arr.ndim == 2:
             spec = P("lat", None)
         else:
             spec = P()
-        return _addressable_shard_put(arr, NamedSharding(mesh, spec))
+        return addressable_shard_put(arr, NamedSharding(mesh, spec))
 
     return jax.tree.map(_put, stack)
 
 
 def append_vface_wall_row(v_lower):
     """Rebuild the full ``(n_lat+1, ...)`` staggered v-array from the lat-band
     carrier ``v_lower`` by appending the TOP wall row (zeros).
 
     This is THE reconstruction of the row the carrier drops: the regular-grid
     north pole wall / tripole cap row, identically zero under the v-carrier
     contract (:func:`shard_state_latlon` REFUSES a state whose top v-face row
     is live), so the result is bit-identical to the original staggered array.
     Shared by :func:`gather_state_latlon` (full-state gather) and the
     persistent-lane DEVICE-SIDE staggered reads (the OMIP driver's
     surface-current consumers, ``run_omip_core2._surface_uv_faces``) so the
     row reconstruction is written once.  Works on any trailing shape (3-D
     leaves or 2-D surface slices) and preserves sharding under GSPMD.
     """
     wall = jnp.zeros_like(v_lower[:1])
     return jnp.concatenate([v_lower, wall], axis=0)
 
 
 def gather_state_latlon(state, mesh):
     """Inverse of :func:`shard_state_latlon`: gather every leaf to a single device
     and rebuild the full ``(n_lat+1, ...)`` ``v`` / ``v_mask`` by appending the
     pole-wall row (zeros) the layout dropped.
 
     The appended row is the north pole wall (``v == 0`` there on the regular
     grid), which is exactly what the single-device rest/forced state carries, so
     the gathered state is bit-comparable to the single-device reference.
 
     Multi-controller (route-B ``jax.distributed``, mesh spanning processes):
     replication routes through a jit-compiled identity instead of
     ``device_put`` (see :func:`legoesm.parallel.latlon_spmd.replicate_leaf`,
     the primitive shared with the atm gather); the single-process path is
     byte-unchanged.
     """
+    # FIRST statement (#1362 round 4). Two rank-local hazards live below:
+    # ``replicate_leaf`` runs a real cross-process collective (a jit identity
+    # with replicated out_shardings; XLA inserts the all-gather), and the
+    # ``if ... is None: continue`` skips mean a state whose OPTIONAL fields
+    # differ across processes performs a DIFFERENT NUMBER of those
+    # collectives -- one rank finishing while a peer still waits. So agree the
+    # mesh AND the ordered list of fields that will actually be gathered,
+    # before the first one runs.
+    _agree_ocean_mesh_entry(mesh, state, where="gather_state_latlon")
     from legoesm.parallel.latlon_spmd import replicate_leaf
 
     rep = NamedSharding(mesh, P())
     _mp = jax.process_count() > 1
 
     def _gather_arr(a):
         return replicate_leaf(a, rep, multiprocess=_mp)
 
     updates = {}
     for name in _V_STAGGERED_STATE_FIELDS:
         field = getattr(state, name)
         if field is None:
             updates[name] = None
             continue
         v_lower = _gather_arr(field.data)
         # north pole-wall / cap row (the shared reconstruction helper)
         updates[name] = field.replace(data=append_vface_wall_row(v_lower))
     for name in state._fields:
         if name in _V_STAGGERED_STATE_FIELDS:
             continue
         val = getattr(state, name)
         if val is None:
             updates[name] = None
         elif hasattr(val, "data"):
             updates[name] = val.replace(data=_gather_arr(val.data))
         else:
             updates[name] = _gather_arr(jnp.asarray(val))
     return state._replace(**updates)
 
 
+# Ordered flag names for the ocean SPMD entry gate. STATIC tuple: the payload
+# width is fixed by this literal, never by rank-local data.
+_OCEAN_SPMD_ENTRY_FLAGS = (
+    "has_mesh", "n_dev", "n_axes", "axis_names", "axis_sizes",
+    "grid_n_lat", "grid_n_lon", "geom_schema", "fold_active",
+    "config_digest", "has_vertex_mask",
+)
+
+
+def _agree_ocean_spmd_entry(model, mesh, *, where: str) -> None:
+    """Agree every rank-local input, as the FIRST statement of a public factory.
+
+    #1362 / codex round 2, ocean twin of the atmosphere's
+    ``_agree_spmd_entry``.  This lane has the same shape of hazard: the
+    ``mesh is None`` early return, ``build_band_grids``' divisibility
+    validation, and ``_build_band_vertex_masks`` (which throws if only THIS
+    rank lacks a primed vertex-mask cache) all execute BEFORE the schema
+    collective.  Any of them lets one process raise or return while a peer
+    blocks in ``process_allgather`` -- a HANG rather than an error.
+
+    Agreeing the mesh shape, grid dimensions and fold state up front makes
+    every downstream rank-local check symmetric by construction.
+
+    ``axis_names`` carries the ORDERED axis-name digest, not just the axis
+    COUNT: the band body indexes ``mesh.axis_names[0]``, so two processes
+    whose meshes name that axis differently would psum/ppermute over
+    different axes while every count-based flag agreed (the ocean instance of
+    codex round-3 blocker 2, which was found on the atmosphere twin --
+    fixing only the lane where a defect was reported is what left five
+    unguarded paths after round 1).
+
+    Grid dimensions go through :func:`coerce_count`, which NEVER raises, and
+    the refusal is deferred until AFTER the collective; building a collective
+    payload must not be able to kill one rank while its peers block in the
+    gather (codex round-3 blocker 3, same rationale as the atm twin).
+    """
+    # Defensive attribute reads: nothing in the payload build may raise before
+    # the collective (see the atm twin for the full rule).
+    grid = getattr(model, "grid", None)
+    fold = getattr(grid, "fold", None)
+    names, sizes = _ocean_mesh_axis_terms(mesh)
+    problems = []
+
+    def _count(value, label, absent=FLAG_ABSENT):
+        payload, problem = coerce_count(value, absent=absent)
+        if problem is not None:
+            problems.append((label, problem))
+        return payload
+
+    flags = (
+        float(mesh is not None),
+        float(mesh.devices.size if mesh is not None else 0),
+        float(len(names)),
+        name_digest48(names),
+        name_digest48(sizes),
+        _count(getattr(grid, "n_lat", None), "grid.n_lat", absent=0.0),
+        _count(getattr(grid, "n_lon", None), "grid.n_lon", absent=0.0),
+        # The geometry array fields that `build_band_grids` slices and
+        # `_replicated_put` broadcasts, by dtype + shape.
+        tree_schema_digest48(grid),
+        float(bool(fold is not None and getattr(fold, "is_active", False))),
+        # ONE digest over EVERY static scalar of the ocean config instead of a
+        # hand-picked few: the step body branches on `outer_integrator`, the
+        # tracer integrator, the polar filter, the freeze floor and the EW
+        # overlap, and NONE of them were agreed (codex round-4, blocker 3).
+        # A valid/invalid or euler/ab2 split makes one rank raise during
+        # tracing while its peer compiles a different program.
+        config_digest48(getattr(model, "config", None)),
+        # `_build_band_vertex_masks` RAISES when this cache is unprimed, and
+        # it runs before the schema collective -- so its presence must be
+        # agreed first or an unprimed rank dies while its peer blocks
+        # (codex round-4, blocker 4).
+        float(getattr(model, "_vertex_mask", None) is not None),
+    )
+    assert_flags_agree(_OCEAN_SPMD_ENTRY_FLAGS, flags, context=where)
+    # AFTER the collective only: symmetric on every rank (see atm twin).
+    for label, problem in problems:
+        raise ValueError(f"{where}: {label} {problem}")
+
+
+# Ordered flag names for the PER-INVOCATION gate on the returned ocean SPMD
+# callable. STATIC tuple: fixed width, never rank-local.
+_OCEAN_CALL_ENTRY_FLAGS = (
+    "has_mesh", "n_dev", "axis_names", "axis_sizes",
+    "state_schema", "has_forcing", "forcing_schema",
+)
+
+
+def _agree_ocean_spmd_call(mesh, state, forcing, *, where: str) -> None:
+    """Agree a returned ocean SPMD callable's per-CALL inputs, FIRST statement.
+
+    #1362 round 4, blocker 7.  ``sharded_step`` runs ``_validate_forcing_layout``
+    and builds a rank-local cache key BEFORE entering its ``shard_map``: a
+    forcing layout that is invalid on one rank only makes that rank raise while
+    its peers enter the collective program -- a hang.  The state + forcing leaf
+    SCHEMA is agreed too, because ``in_specs``/``out_specs`` are derived from
+    it, so two processes with different optional fields compile different
+    programs.
+
+    Cost: one small allgather per CALL, and an exact no-op under a single
+    process.  See the atmosphere twin ``_agree_spmd_call`` for why gating only
+    on cache misses is NOT a valid optimisation.
+    """
+    names, sizes = _ocean_mesh_axis_terms(mesh)
+    assert_flags_agree(_OCEAN_CALL_ENTRY_FLAGS, (
+        float(mesh is not None),
+        float(mesh.devices.size if mesh is not None else 0),
+        name_digest48(names),
+        name_digest48(sizes),
+        tree_schema_digest48(state),
+        float(forcing is not None),
+        (tree_schema_digest48(forcing) if forcing is not None
+         else FLAG_ABSENT),
+    ), context=where)
+
+
 def make_sharded_ocean_step(model, mesh):
     """Return ``step(state, dt, freshwater=None, surface_forcing=None,
     sponge=None, t_seconds=None) -> state`` running ``model.step``
     lat-band-SPMD.
 
     The forcing channels mirror ``model.step``'s keyword surface: pass
     pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
     ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
     is a replicated scalar like ``dt``).  ``None`` forcing keeps the
     dynamics-only program; each distinct None<->populated combination
     compiles (and caches) its own executable.
 
     Parameters
     ----------
     model
         ``LatLonCGridOceanModel`` (latlon geometry; tripole north-fold is a
         follow-up — raises ``NotImplementedError`` if ``model.grid.fold`` is
         active).
     mesh
         A 1-D ``jax.sharding.Mesh`` with axis name ``"lat"`` (from
         ``legoesm.parallel.mesh.create_latlon_mesh(...).mesh``).
 
     Notes
     -----
     Build the ``N`` band geometries + band vertex masks host-side, stack their
-    ARRAY fields over a leading band axis SHARDED ``P("lat")`` (each device
-    holds only its own slab; the ``shard_map`` body reads it at ``[0]``; the
-    geometry SCALAR fields stay static — see the module docstring).  ``dt`` is a TRACED,
+    ARRAY fields into a replicated pytree, and index by
+    ``jax.lax.axis_index("lat")`` in the ``shard_map`` body (the geometry SCALAR
+    fields stay static — see the module docstring).  ``dt`` is a TRACED,
     replicated operand and the jitted ``shard_map`` is built once and cached
     (a per-call rebuild re-traced the whole ocean step every call — see the
     ``_cache`` note below).  ``check_vma=False`` (the JAX >= 0.8
     replication check) because the band halo intentionally reads neighbour-rank
     data (replication-unaware).
 
     The state must be laid out with :func:`shard_state_latlon` (``v`` /
     ``v_mask`` carried as the ``n_lat``-row ``v_lower``).  The body reconstructs
     each band's ``nl+1`` v-faces, runs the step on the band geometry, and
     converts the result back to the ``v_lower`` representation.
     """
+    # FIRST statement: agree every rank-local input before ANY
+    # rank-local check can raise or return (codex round-2).
+    _agree_ocean_spmd_entry(model, mesh, where="make_sharded_ocean_step")
     if mesh is None:                   # single-device: plain step
         return lambda state, dt, **forcing_kwargs: model.step(
             state, dt, **forcing_kwargs)
 
     n_dev = mesh.devices.size
     axis = mesh.axis_names[0]
 
     # Tripole north-fold (scaling-audit item 4): SUPPORTED under the same
     # v-carrier contract as the regular grid.  Every fold-touching operator
     # is already uniform-program fold-capable (the data-dependent
     # ``north_fold_mask``/``apply_north_fold`` selection on
     # ``axis_index == N-1`` — gated by test_latlon_spmd_northfold), and
     # ``build_band_grids``' slicer keeps ``is_active`` rank-consistent with
     # the ``fold_j=-1`` sentinel off the north band.  The one structural
     # assumption is the v-carrier's: the TOP v-face row (the seam/cap row,
     # ``v[n_lat]``) must be WALL-MASKED so the in-body reconstruction's
     # zero row is exact — true for the cap-row convention of
     # ``create_synthetic_tripole`` and the eORCA masks (``v_mask[-1] == 0``;
     # the serial step keeps ``v[-1] == 0`` identically).  That contract is
     # asserted on the CONCRETE state in :func:`shard_state_latlon` — a live
     # (unmasked) seam v-row refuses loudly there instead of silently
     # reconstructing zeros here.
 
-    # --- host-side band geometries + vertex masks (band-stacked, P("lat")-sharded) ---
+    # --- host-side band geometries + vertex masks (replicated, indexed in-body) ---
     band_grids = build_band_grids(model.grid, n_dev)
     band_vmasks = _build_band_vertex_masks(model, n_dev)
     template = band_grids[0]           # static-scalar source (uniform bands)
     array_field_names = _geom_array_field_names(template)
 
     # Stack each geometry ARRAY field over the band axis (rank 0..N-1) and
     # SHARD along that axis (#1370 stage (iii), codex round-18): each device
     # holds ONLY its own band's slab instead of the whole global stack —
     # this was one of the residual ~1.4-1.7 global-field-equivalents of
     # per-device residency left after the host-side-build fix (probe
     # 26524423). The leading axis has length n_dev, so P("lat") divides it
     # exactly; the body indexes its local slab at [0]. Values are unchanged
-    # — same stack, different placement; the cross-process divergence
-    # guard below runs on HOST values and is placement-blind.
+    # — same stack, different placement; the process-0 broadcast +
+    # divergence guard below runs on HOST values and is placement-blind.
     rep = NamedSharding(mesh, P("lat"))
 
     def _replicated_put(arr, name):
-        # (Name kept for history; since 2026-08-03 this is a SHARDED stack
-        # put.) The band-geometry arrays are (re)computed per process and
-        # can differ in their last ULPs (per-process XLA autotuning on
-        # device-derived grid fields) — the fully-replicated-put era
-        # broadcast process 0's bytes to sidestep the P() bit-identity
-        # assert (job 26450848). With the #1370-iii P("lat") sharding each
-        # process's devices consume ONLY its own band rows, so the
-        # broadcast became both unnecessary and, at nd>=96, fatal (its
-        # psum program is nd x the stack — see the note at the put below).
-        # GUARD (codex round-3): process 0 must not silently mask REAL
-        # cross-process divergence. Compare an allgathered fingerprint:
-        # structural entries exactly; value entries EXACTLY for integer/bool
-        # arrays (masks are comparison results — bit-reproducible, and an
-        # exact compare is the only way to catch a two-cell flip that cancels
-        # in the sum, codex round-4) and to rtol 1e-5 for float arrays (only
-        # ULP autotune drift is expected there; quantize-then-assert-equal
-        # false-positived on a rounding boundary, job 26453240).
-        # NO DEADLOCK RISK: every process fingerprints the same fields in the
-        # same order and derives the verdict from the SAME gathered array, so
-        # the refusal is symmetric — all raise or none.
-        # Residual (documented): a float-geometry divergence preserving sum,
-        # sum-of-squares AND absmax to 1e-5 is not detected; band grids are
-        # analytic in lat/lon, so any real inconsistency moves those moments.
-        host = np.asarray(arr)
-        if jax.process_count() > 1:
-            from jax.experimental import multihost_utils
-
-            # PER-BAND fingerprints (module-level, unit-tested): exact
-            # dtypes hash positionally per band; floats compare per-band
-            # moments to rtol 1e-5 — bounds each band's drift instead of
-            # letting it hide in a whole-array sum, since each process's
-            # own bytes are now the live inputs for the bands it owns
-            # (codex r14). A mask that genuinely differs across processes
-            # means different wet domains = different physics: refusing is
-            # correct, not a false alarm.
-            struct, vals, is_exact = geom_band_fingerprint(
-                host, host.shape[0])
-            g_struct = multihost_utils.process_allgather(struct)
-            g_vals = multihost_utils.process_allgather(vals)
-            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
-                raise RuntimeError(
-                    f"make_sharded_ocean_step: band-geometry field {name!r} "
-                    f"DIVERGES across processes (exact_dtype={is_exact}, "
-                    f"gathered={g_vals.tolist()}) — a real config/grid "
-                    f"inconsistency, not autotune noise; refusing to "
-                    f"shard it.")
-            # NO broadcast_one_to_all here (removed 2026-08-03): its psum
-            # program is [n_processes, stack] in / P() fully-replicated out,
-            # so its logical arg bytes are nd x the global stack — 82.4 GB
-            # at nd=96 and 109.6 GB at nd=128 for one 3-D field, the
-            # near-linear-in-nd wall that killed oc @96/@128 (jobs
-            # 26642771/26636762) while @64 sat just under XLA's 63.8 GB
-            # limit.  The target sharding is P("lat"): each process's
-            # devices consume ONLY its own band rows, so cross-process
-            # byte-identity of non-owned rows is irrelevant, and REAL
-            # divergence is already refused by the fingerprint gate above.
-            # make_array_from_callback hands each process exactly its
-            # addressable slabs — the same #1100 pattern as the state
-            # build — with no global-sized collective program at all.
-        host_np = np.asarray(host)
-        return jax.make_array_from_callback(
-            host_np.shape, rep, lambda idx: host_np[idx])
-
-    if jax.process_count() > 1:
-        # Schema gate FIRST (one fixed-shape collective every process
-        # reaches): a process-dependent field list or a mixed
-        # jax_enable_x64 setting would otherwise desynchronize the
-        # per-field gathers below instead of failing with a clear message.
-        from jax.experimental import multihost_utils as _mhu
-
-        _g = _mhu.process_allgather(
-            _schema_fingerprint(list(array_field_names), n_dev))
-        if not bool(np.all(_g == _g[0])):
-            raise RuntimeError(
-                "make_sharded_ocean_step: the band-geometry SCHEMA differs "
-                "across processes (field list / x64 setting / device count "
-                f"— gathered {_g.tolist()}). Fix the per-process config "
-                "before sharding; the per-field checks below assume one "
-                "schema.")
-
-    geom_stacks = {
-        name: _replicated_put(
-            jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
-                      axis=0), name)
+        # (Name kept for history; this is a SHARDED P("lat") stack put.)
+        # checked_shard_put replaces the broadcast_checked+device_put pair:
+        # the broadcast's psum program is [n_processes, stack] (nd x 849 MB
+        # at LL2304 — the @96/@128 wall), and a numpy device_put onto an
+        # all-process sharding pays jax's whole-array assert_equal on top.
+        # The per-band gate keeps the divergence contract (n_bands is
+        # schema-gated just below, so payload widths agree). ONE shared
+        # implementation: legoesm.parallel.geometry_consistency.
+        return checked_shard_put(
+            arr, name, rep, context="make_sharded_ocean_step",
+            n_bands=n_dev)
+
+    # Schema gate FIRST (one fixed-shape collective every process reaches):
+    # a process-dependent field list or a mixed jax_enable_x64 setting would
+    # otherwise desynchronize the per-field gathers below instead of failing
+    # with a clear message.
+    # Build the raw stacks FIRST so the schema gate can also cover each
+    # field's dtype class and ndim -- those decide the per-field payload
+    # shape below, so a bool-vs-float disagreement must fail HERE rather than
+    # deadlock in the per-field gather.
+    _raw_geom = {
+        name: jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
+                        axis=0)
         for name in array_field_names
     }
-    vmask_stack = _replicated_put(
-        jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0),
-        "vertex_mask")
+    _raw_vmask = jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0)
+    _gate_names = [*array_field_names, "vertex_mask"]
+    assert_schema_agrees(
+        _gate_names, n_dev, context="make_sharded_ocean_step",
+        arrays=[*(_raw_geom[n] for n in array_field_names), _raw_vmask])
+
+    geom_stacks = {name: _replicated_put(_raw_geom[name], name)
+                   for name in array_field_names}
+    vmask_stack = _replicated_put(_raw_vmask, "vertex_mask")
 
     # Static perms for the v north-boundary-row ppermute (band r receives band
     # r+1's v_lower[0] = global v[e]; north band non-target receives 0).
     perm_north, _perm_south = latlon_band_perms(n_dev)
 
     def _body(state_local, forcing_local, geom_stacks_local,
               vmask_stack_local, dt):
         fw_local, sf_local, sponge_local, t_s_local = forcing_local
         r = jax.lax.axis_index(axis)
 
         # Rebuild this band's geometry. The stacks are SHARDED P("lat") on
         # their leading band axis, so inside shard_map each device's local
         # view is its own (1, ...) slab — index [0], NOT [r] (stage (iii);
         # [r] was the replicated-stack indexing). Static scalars
         # (n_lat_local, n_lon, radius, dlon, dlat, fold) come from the band
         # template. The fold is inactive + identical on every band here.
         geom_arrays = {name: geom_stacks_local[name][0]
                        for name in array_field_names}
         band_geom = template._replace(**geom_arrays)
         band_vmask = vmask_stack_local[0]
 
         # Reconstruct the band's nl+1 v-faces from the nl-row v_lower.  band r's
         # north boundary row is band r+1's v_lower[0] (= global v[e]); the north
         # band (no r+1) receives the pole-wall 0 via the ppermute non-target.
         def _reconstruct_v(field):
             if field is None:
                 return None
             v_lower = field.data                      # (nl, n_lon[, nlev])
             boundary = jax.lax.ppermute(
                 v_lower[0:1], axis, perm_north)        # band r+1's first row -> 0 at top
             v_band = jnp.concatenate([v_lower, boundary], axis=0)  # nl+1 rows
             return field.replace(data=v_band)
 
         # Convert a returned nl+1 v back to the nl-row v_lower (drop the boundary
         # row, owned by band r+1).
         def _to_v_lower(field):
             if field is None:
                 return None
             return field.replace(data=field.data[:-1])
 
         # Fused v-carrier reconstruction (scaling-M4): under the SAME
         # trace-time switch as the pad aggregation, pack the boundary-row
         # ppermutes of ALL staggered carriers (v + v_mask) into one
         # collective per dtype group — value-identical (a bit-copy
         # exchange; the flag flip re-keys the sharded_step cache below so
         # a reused step object rebuilds).  Default OFF = the historical
         # per-field ppermutes, byte-identical.
         import os as _os
         _fused_v = _os.environ.get(
             "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
         present = [name for name in _V_STAGGERED_STATE_FIELDS
                    if getattr(state_local, name) is not None]
         if _fused_v and len(present) > 1:
             fulls = reconstruct_vface_lower_multi(
                 tuple(getattr(state_local, n).data for n in present),
                 axis, perm_north)
             v_updates = {n: getattr(state_local, n).replace(data=f)
                          for n, f in zip(present, fulls)}
             for name in _V_STAGGERED_STATE_FIELDS:
                 v_updates.setdefault(name, None)
         else:
             v_updates = {name: _reconstruct_v(getattr(state_local, name))
                          for name in _V_STAGGERED_STATE_FIELDS}
         state_band = state_local._replace(**v_updates)
 
         result = _step_body(model, state_band, dt,
                             grid=band_geom, vertex_mask=band_vmask,
                             freshwater=fw_local, surface_forcing=sf_local,
                             sponge=sponge_local, t_seconds=t_s_local)
 
         out_updates = {name: _to_v_lower(getattr(result, name))
                        for name in _V_STAGGERED_STATE_FIELDS}
         return result._replace(**out_updates)
 
     # Build the JITTED shard_map ONCE and cache it — the atm sibling's fix
     # (make_sharded_atm_latlon_step, probe 8561202): a bare shard_map is NOT
     # compilation-cached, so rebuilding it per call re-traced + recompiled the
     # (large, un-jitted) ocean band step EVERY call (~80 s/step at 16x32x3 nd=4
     # on CPU — host tracing, not compute; the bench's scaling number was
     # meaningless). ``dt`` is a TRACED operand (was a mutated closure cell,
     # which also forced the per-call rebuild) so a changing dt does not
     # retrigger compilation.
     _cache = {}
     n_lat_global = int(model.grid.n_lat)
 
     def _validate_forcing_layout(forcing):
         """Loudly refuse forcing leaves the lat-band shard cannot split.
 
         Every OMIP forcing field is CELL-CENTERED ``(n_lat, n_lon[, nlev])``
         — there are no v-face ``(n_lat+1, …)`` forcing arrays, so no
         ``v_lower`` handling.  A wrong-leading-dim leaf would otherwise die
         inside shard_map with an opaque divisibility error.
         """
         leaves, _ = jax.tree_util.tree_flatten_with_path(forcing)
         for path, leaf in leaves:
             nd = int(getattr(leaf, "ndim", np.ndim(leaf)))
             if nd == 1:
                 # No OMIP forcing field is 1-D; _lat_spec would shard a
                 # profile/staggered vector over "lat" silently-wrong
                 # (codex r1 #2).
                 raise ValueError(
                     f"sharded ocean step: forcing leaf "
                     f"{jax.tree_util.keystr(path)} is 1-D "
                     f"(shape {tuple(leaf.shape)}); forcing must be "
                     f"cell-centered (n_lat, n_lon[, nlev]) arrays or "
                     f"scalars.")
             if nd >= 2 and int(leaf.shape[0]) != n_lat_global:
                 raise ValueError(
                     f"sharded ocean step: forcing leaf {jax.tree_util.keystr(path)} "
                     f"has leading dim {leaf.shape[0]} != n_lat "
                     f"({n_lat_global}); forcing must be cell-centered "
                     f"(n_lat, n_lon[, nlev]) to shard on the lat axis.")
 
     def sharded_step(state, dt, freshwater=None, surface_forcing=None,
                      sponge=None, t_seconds=None, aux=None):
-        # ``aux`` (codex/2-proc repro 2026-08-03): the geometry + vmask
-        # stacks are SHARDED global arrays; when this wrapper runs INSIDE
-        # an outer trace (a bench/driver ``jit``/``scan`` over the step —
-        # jit-of-jit inlines the inner call), concrete closure arrays
-        # become OUTER-TRACE CONSTANTS and jax's MLIR constant handler
-        # tries to fetch their value — impossible for non-addressable
-        # arrays (RuntimeError: 'Fetching value ... non-addressable'; the
-        # multicontroller lane has been broken this way since the
-        # #1370-iii stack sharding). Callers that wrap the step in their
-        # own jit MUST thread ``step.aux`` through their jit boundary as
-        # an ARGUMENT and pass it back here.
+        # ``aux``: the sharded geometry+vmask stacks. When this wrapper runs
+        # INSIDE an outer trace (a bench/driver jit/scan — jit-of-jit
+        # inlines the inner call), concrete closure arrays become
+        # OUTER-trace constants whose value the MLIR handler cannot fetch
+        # for non-addressable arrays (broken since #1370-iii sharded the
+        # stacks). Outer-jit callers MUST thread ``step.aux`` through their
+        # jit boundary as an ARGUMENT and pass it back here.
         # ONE forcing operand: None fields drop out of the pytree structure,
         # so specs derived by tree.map skip them automatically and the
         # structure key below distinguishes every None<->array combination.
+        _agree_ocean_spmd_call(
+            mesh, state, (freshwater, surface_forcing, sponge, t_seconds),
+            where="make_sharded_ocean_step.step")
         forcing = (freshwater, surface_forcing, sponge, t_seconds)
         _validate_forcing_layout((freshwater, surface_forcing, sponge))
         # Cache key = the state's AND forcing's pytree STRUCTURE, plus the
         # forcing leaves' RANKS: in_specs/out_specs are derived from them,
         # so a later call with a different structure (an optional field
         # flipping None <-> Field, a sea-ice lane populating sf.salt_flux,
         # the restoring lane passing no forcing at all) OR a same-field
         # rank change (SpongeForcing.gamma is legitimately 2-D horizontal
         # OR 3-D full-rank — same structure, different _lat_spec; codex r1
         # #1) must rebuild the shard_map rather than reuse stale specs.
         forcing_ndims = tuple(
             int(getattr(leaf, "ndim", np.ndim(leaf)))
             for leaf in jax.tree.leaves(forcing))
         # The SPMD fused-halo switch is read at TRACE time inside the pad
         # dispatch — flipping LEGOESM_LATLON_SPMD_FUSED_HALO on a reused
         # step object must rebuild the shard_map, not reuse a stale jaxpr
         # (codex, audit item 7).
         import os as _os
 
         _fused_halo = _os.environ.get(
             "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
         key = (jax.tree.structure(state), jax.tree.structure(forcing),
                forcing_ndims, _fused_halo)
         fn = _cache.get(key)
         if fn is None:
             in_spec = jax.tree.map(_lat_spec, state)
             # Forcing leaves are cell-centered -> plain lat-band specs;
             # scalars (t_seconds) replicate, exactly like dt.
             forcing_spec = jax.tree.map(_lat_spec, forcing)
             # Stage (iii): the stacks are banded on their leading axis, so
             # the shard_map spec matches their P("lat") placement (the body
             # indexes its local (1, ...) slab at [0]).
             geom_spec = jax.tree.map(lambda _x: P("lat"), geom_stacks)
             vmask_spec = P("lat")
             # JAX >= 0.8 top-level shard_map takes ``check_vma`` (the
             # replication check); the band halo reads neighbour-rank data so
             # disable it (same as the validated PCG / halo-parity shard_maps).
             fn = jax.jit(shard_map(
                 _body,
                 mesh=mesh,
                 in_specs=(in_spec, forcing_spec, geom_spec, vmask_spec, P()),
                 out_specs=in_spec,
                 check_vma=False,
             ))
             _cache[key] = fn
         # Arm the SPMD halo backend ONLY around the call, then RESTORE the
         # previous backend (codex finding): leaving it globally armed makes a
         # later serial/full-domain ocean call take SPMD-only branches
         # (axis_index / ppermute / psum in pad_with_pole_bc_lat, conservation,
         # eta_floor) OUTSIDE a shard_map -> crash. The FIRST call traces with
         # the backend armed (baking the SPMD halo/reduction ops into the
         # compiled program); later calls reuse the cached compile, and the
         # arm/restore keeps any interleaved serial path untouched.
         # Save+restore the FULL backend state (backend + MPI topology + SPMD
         # mesh) so a prior "mpi"/"spmd" backend is restored intact: activate_*
         # clears the MPI topology, and set_halo_backend("mpi") REQUIRES a
         # topology (codex).
         from legoesm.grids.halo import (
             get_halo_backend, get_mpi_topology, get_spmd_mesh,
             set_halo_backend, set_spmd_mesh,
         )
         _prev_backend = get_halo_backend()
         _prev_topo = get_mpi_topology()
         _prev_mesh = get_spmd_mesh()
         activate_latlon_spmd_halo(mesh)
         try:
             _geom, _vmask = aux if aux is not None else (geom_stacks,
                                                         vmask_stack)
             return fn(state, forcing, _geom, _vmask, jnp.asarray(dt))
         finally:
             set_spmd_mesh(_prev_mesh)
             set_halo_backend(_prev_backend, _prev_topo)
 
     # Expose the stacks so outer-jit callers can pass them as arguments
     # (see the ``aux`` note in the signature).
     sharded_step.aux = (geom_stacks, vmask_stack)
     return sharded_step
 
 
 def make_sharded_ocean_step_global(model, mesh):
     """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
     that takes a GLOBAL (single-device-layout) state + forcing and returns a
     GLOBAL state — the minimal-diff driver entry point.
 
     Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
     (:func:`shard_state_latlon` + :func:`shard_forcing_latlon`), runs the lat-band
     SPMD step, then gathers the state OUT (:func:`gather_state_latlon`).  This lets
     the OMIP host loop keep operating on a normal full-domain state — the per-step
     host BCs (SSS restore, prognostic ice, geothermal, BBL, nudge) see the
     gathered global state UNCHANGED — at the cost of a per-step gather/scatter
     (acceptable for the host-coupled OMIP driver; the pure-dynamics inner loop
     should use :func:`make_sharded_ocean_step` directly to stay sharded).
 
     ``mesh is None`` ⇒ the plain single-device ``model.step`` (no scatter/gather).
     """
+    # FIRST statement: agree every rank-local input before ANY
+    # rank-local check can raise or return (codex round-2).
+    _agree_ocean_spmd_entry(model, mesh, where="make_sharded_ocean_step_global")
     if mesh is None:                   # single-device: plain step
         return lambda state, dt, surface_forcing=None, freshwater=None: (
             model.step(state, dt, freshwater=freshwater,
                        surface_forcing=surface_forcing))
 
     inner = make_sharded_ocean_step(model, mesh)
 
     def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
-        # Scatter the global state AND forcing to the band layout explicitly
-        # (codex r17 item 3: the old comment claimed inner sharded the
-        # forcing; it forwarded it global and relied on implicit JIT input
-        # placement — which under multicontroller pays jax's whole-array
-        # device_put assert, the nd-linear wall this module removes).
+        _agree_ocean_spmd_call(mesh, state, (surface_forcing, freshwater),
+                               where="make_sharded_ocean_step_global.step")
+        # Scatter the global state AND forcing to the band layout
+        # explicitly (the old comment claimed inner sharded the forcing;
+        # it forwarded it global and relied on implicit JIT input
+        # placement — jax's whole-array device_put assert under
+        # multicontroller, the nd-linear wall this module removes).
         ss = shard_state_latlon(state, mesh)
         ss = inner(ss, dt,
                    surface_forcing=shard_forcing_latlon(surface_forcing, mesh),
                    freshwater=shard_forcing_latlon(freshwater, mesh))
         return gather_state_latlon(ss, mesh)
 
     return sharded_step_global
diff --git a/tests/ocean/unit/test_sharded_geom_fingerprint.py b/tests/ocean/unit/test_sharded_geom_fingerprint.py
index 2f61c0709..5b24a4b9e 100644
--- a/tests/ocean/unit/test_sharded_geom_fingerprint.py
+++ b/tests/ocean/unit/test_sharded_geom_fingerprint.py
@@ -1,44 +1,44 @@
 """Unit tests for the band-geometry cross-process fingerprint gate.
 
 The gate decides whether ``make_sharded_ocean_step`` accepts per-process
 band-geometry stacks without the (removed, nd-linear-cost) process-0
 broadcast — see the 2026-08-03 fix note at the sharded put. These tests
 pin the gate's discrimination properties single-process (the
 multicontroller allgather wiring is exercised by the distributed suite).
 """
 import numpy as np
 import pytest
 
-from legoesm.ocean.dynamics.sharded_ocean_step import (
+from legoesm.parallel.geometry_consistency import (
+    band_fingerprint as geom_band_fingerprint,
     band_fingerprints_agree,
-    geom_band_fingerprint,
 )
 
 N_BANDS = 4
 SHAPE = (N_BANDS, 6, 8)
 
 
 def _gather(*hosts):
     """Simulate process_allgather over per-process fingerprints."""
     fps = [geom_band_fingerprint(h, N_BANDS) for h in hosts]
     exacts = {fp[2] for fp in fps}
     assert len(exacts) == 1
     g_struct = np.stack([fp[0] for fp in fps])
     g_vals = np.stack([fp[1] for fp in fps])
     return g_struct, g_vals, fps[0][2]
 
 
 def test_identical_float_stacks_agree():
     rng = np.random.default_rng(0)
     a = rng.normal(size=SHAPE).astype(np.float32)
     assert band_fingerprints_agree(*_gather(a, a.copy()))
 
 
 def test_ulp_scale_drift_agrees():
     rng = np.random.default_rng(1)
     a = rng.normal(size=SHAPE).astype(np.float64) + 10.0
     b = np.nextafter(a, np.inf)  # a TRUE 1-ULP elementwise drift
     assert band_fingerprints_agree(*_gather(a, b))
 
 
 def test_band_local_drift_refused_where_global_gate_passed():
@@ -72,33 +72,33 @@ def test_exact_dtype_permutation_refused():
     b[1, 3, 2] = 1  # same count, different position, same band
     assert not band_fingerprints_agree(*_gather(a, b))
 
 
 def test_bool_mask_two_cell_flip_refused():
     a = np.zeros(SHAPE, dtype=bool)
     a[0, 0, 0] = True
     b = a.copy()
     b[0, 0, 0] = False
     b[0, 5, 7] = True  # true-count preserved
     assert not band_fingerprints_agree(*_gather(a, b))
 
 
 def test_nonfinite_count_mismatch_refused():
     a = np.ones(SHAPE, dtype=np.float32)
     b = a.copy()
     b[3, 0, 0] = np.nan  # struct carries per-band non-finite counts
     assert not band_fingerprints_agree(*_gather(a, b))
 
 
 def test_shape_mismatch_refused():
     a = np.ones(SHAPE, dtype=np.float32)
     b = np.ones((N_BANDS, 6, 9), dtype=np.float32)
     fa = geom_band_fingerprint(a, N_BANDS)
     fb = geom_band_fingerprint(b, N_BANDS)
     # Different shapes -> different struct lengths; the agree helper is
     # only called on stackable gathers, so assert the structs differ.
     assert fa[0].shape != fb[0].shape or not np.array_equal(fa[0], fb[0])
 
 
-def test_wrong_leading_axis_asserts():
-    with pytest.raises(AssertionError):
+def test_wrong_leading_axis_raises():
+    with pytest.raises(ValueError):
         geom_band_fingerprint(np.ones((3, 2)), N_BANDS)

codex
The ocean construction does place the schema collective before any `checked_shard_put` calls. I’m now checking all remaining broadcasts and the new per-call gates for shape/signature mismatches.
exec
/usr/bin/bash -lc "rg -n -C 6 'broadcast_checked\\(' packages tests; rg -n -C 8 'broadcast_checked' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py; rg -n -C 14 'def _agree_ocean_spmd_call|_agree_ocean_spmd_call\\(' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py; rg -n -C 10 'def _agree_spmd_call|_agree_spmd_call\\(' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 449ms:
packages/core/legoesm/parallel/geometry_consistency.py-516-            f"(field list / x64 setting / device count / per-field dtype "
packages/core/legoesm/parallel/geometry_consistency.py-517-            f"class / ndim — gathered {gathered.tolist()}). Fix the "
packages/core/legoesm/parallel/geometry_consistency.py-518-            f"per-process config before sharding; the per-field checks "
packages/core/legoesm/parallel/geometry_consistency.py-519-            f"assume one schema.")
packages/core/legoesm/parallel/geometry_consistency.py-520-
packages/core/legoesm/parallel/geometry_consistency.py-521-
packages/core/legoesm/parallel/geometry_consistency.py:522:def broadcast_checked(arr, name: str, *, context: str) -> np.ndarray:
packages/core/legoesm/parallel/geometry_consistency.py-523-    """Verify ``arr`` agrees across processes, then broadcast process 0's bytes.
packages/core/legoesm/parallel/geometry_consistency.py-524-
packages/core/legoesm/parallel/geometry_consistency.py-525-    Multi-process: returns a host ``np.ndarray`` that is bit-identical on
packages/core/legoesm/parallel/geometry_consistency.py-526-    every process, safe to hand to a replicated ``device_put``.
packages/core/legoesm/parallel/geometry_consistency.py-527-    Single-process: returns ``arr`` ITSELF, untouched — no collectives, no
packages/core/legoesm/parallel/geometry_consistency.py-528-    host round trip, no dtype/weak-type change.
--
tests/distributed/test_geometry_consistency_mp.py-93-def test_agreeing_geometry_passes_on_every_rank():
tests/distributed/test_geometry_consistency_mp.py-94-    """Identical inputs: no rank raises, and no rank blocks."""
tests/distributed/test_geometry_consistency_mp.py-95-    assert_schema_agrees, assert_flags_agree, broadcast_checked = _guards()
tests/distributed/test_geometry_consistency_mp.py-96-
tests/distributed/test_geometry_consistency_mp.py-97-    assert_schema_agrees(("lat", "lon"), 2, context="mp-test")
tests/distributed/test_geometry_consistency_mp.py-98-    assert_flags_agree(("alpha", "beta"), (1.0, 0.0), context="mp-test")
tests/distributed/test_geometry_consistency_mp.py:99:    out = broadcast_checked(np.arange(4, dtype=np.float64), "field", context="mp-test")
tests/distributed/test_geometry_consistency_mp.py-100-    np.testing.assert_allclose(out, np.arange(4, dtype=np.float64))
tests/distributed/test_geometry_consistency_mp.py-101-
tests/distributed/test_geometry_consistency_mp.py-102-
tests/distributed/test_geometry_consistency_mp.py-103-def test_axis_order_divergence_raises_on_every_rank():
tests/distributed/test_geometry_consistency_mp.py-104-    """The sharp case: ``(lat,lon)`` vs ``(lon,lat)`` is invisible to a
tests/distributed/test_geometry_consistency_mp.py-105-    count-and-size payload, so this is what the name digest exists for.
--
tests/distributed/test_geometry_consistency_mp.py-126-    (different wet domain, different polar mask) into silently wrong physics,
tests/distributed/test_geometry_consistency_mp.py-127-    which is the entire reason the broadcast is guarded rather than blind.
tests/distributed/test_geometry_consistency_mp.py-128-    """
tests/distributed/test_geometry_consistency_mp.py-129-    _, _, broadcast_checked = _guards()
tests/distributed/test_geometry_consistency_mp.py-130-    arr = np.arange(4, dtype=np.float64) + (0.0 if _rank() == 0 else 1.0)
tests/distributed/test_geometry_consistency_mp.py-131-    with pytest.raises(RuntimeError, match="DIVERGES"):
tests/distributed/test_geometry_consistency_mp.py:132:        broadcast_checked(arr, "field", context="mp-test")
tests/distributed/test_geometry_consistency_mp.py-133-
tests/distributed/test_geometry_consistency_mp.py-134-
tests/distributed/test_geometry_consistency_mp.py-135-def test_guards_still_agree_after_a_divergence_was_raised():
tests/distributed/test_geometry_consistency_mp.py-136-    """Ordering regression guard: a raised divergence must leave the collective
tests/distributed/test_geometry_consistency_mp.py-137-    stream aligned, so a SUBSEQUENT agreeing call still works on every rank.
tests/distributed/test_geometry_consistency_mp.py-138-
tests/distributed/test_geometry_consistency_mp.py-139-    If a guard consumed a different number of collectives on the raising path
tests/distributed/test_geometry_consistency_mp.py-140-    than on the passing path, this is where the ranks would desynchronise.
tests/distributed/test_geometry_consistency_mp.py-141-    """
tests/distributed/test_geometry_consistency_mp.py-142-    assert_schema_agrees, _, broadcast_checked = _guards()
tests/distributed/test_geometry_consistency_mp.py:143:    out = broadcast_checked(np.full(3, 2.5, dtype=np.float64), "after", context="mp-test")
tests/distributed/test_geometry_consistency_mp.py-144-    np.testing.assert_allclose(out, 2.5)
tests/distributed/test_geometry_consistency_mp.py-145-    assert_schema_agrees(("lat", "lon"), 2, context="mp-test")
--
tests/unit/test_geometry_consistency.py-140-class TestSingleProcessIsAPassthrough:
tests/unit/test_geometry_consistency.py-141-    """With one process there is nothing to compare: no collective may run
tests/unit/test_geometry_consistency.py-142-    (they would hang), and the value must come back untouched."""
tests/unit/test_geometry_consistency.py-143-
tests/unit/test_geometry_consistency.py-144-    def test_broadcast_checked_returns_the_SAME_OBJECT(self):
tests/unit/test_geometry_consistency.py-145-        a = np.linspace(0.0, 1.0, 40).reshape(5, 8)
tests/unit/test_geometry_consistency.py:146:        assert broadcast_checked(a, "area", context="test") is a
tests/unit/test_geometry_consistency.py-147-
tests/unit/test_geometry_consistency.py-148-    def test_single_process_does_NOT_convert_a_device_array_to_host(self):
tests/unit/test_geometry_consistency.py-149-        """codex 2026-07-29 (major 3): converting unconditionally forced a
tests/unit/test_geometry_consistency.py-150-        device->host->device round trip and stripped weak-type metadata on a
tests/unit/test_geometry_consistency.py-151-        1-process mesh. The atmosphere lane passes `jnp.stack` results
tests/unit/test_geometry_consistency.py-152-        straight in, so it must come back as the SAME jax array."""
tests/unit/test_geometry_consistency.py-153-        jnp = pytest.importorskip("jax.numpy")
tests/unit/test_geometry_consistency.py-154-        a = jnp.arange(6, dtype=jnp.int32)
tests/unit/test_geometry_consistency.py:155:        out = broadcast_checked(a, "idx", context="test")
tests/unit/test_geometry_consistency.py-156-        assert out is a
tests/unit/test_geometry_consistency.py-157-        assert not isinstance(out, np.ndarray)
tests/unit/test_geometry_consistency.py-158-
tests/unit/test_geometry_consistency.py-159-    def test_assert_schema_agrees_is_a_noop(self):
tests/unit/test_geometry_consistency.py-160-        assert_schema_agrees(["a", "b"], 1, context="test") is None
tests/unit/test_geometry_consistency.py-161-
--
tests/unit/test_geometry_consistency.py-221-    emitted.
tests/unit/test_geometry_consistency.py-222-    """
tests/unit/test_geometry_consistency.py-223-
tests/unit/test_geometry_consistency.py-224-    def test_exact_array_payload_is_the_byte_digest(self, multiproc):
tests/unit/test_geometry_consistency.py-225-        fake = multiproc()
tests/unit/test_geometry_consistency.py-226-        a = np.arange(24, dtype=np.int32).reshape(4, 6)
tests/unit/test_geometry_consistency.py:227:        broadcast_checked(a, "idx", context="t")
tests/unit/test_geometry_consistency.py-228-        vals = fake.seen[-1]
tests/unit/test_geometry_consistency.py-229-        assert vals[0] == content_hash48(a), (
tests/unit/test_geometry_consistency.py-230-            "exact arrays must be fingerprinted by the POSITIONAL byte "
tests/unit/test_geometry_consistency.py-231-            "digest; a moment fingerprint here loses permutation detection")
tests/unit/test_geometry_consistency.py-232-
tests/unit/test_geometry_consistency.py-233-    def test_float_array_payload_is_the_moment_triple(self, multiproc):
tests/unit/test_geometry_consistency.py-234-        fake = multiproc()
tests/unit/test_geometry_consistency.py-235-        a = np.linspace(-2.0, 3.0, 30).reshape(5, 6)
tests/unit/test_geometry_consistency.py:236:        broadcast_checked(a, "area", context="t")
tests/unit/test_geometry_consistency.py-237-        vals = fake.seen[-1]
tests/unit/test_geometry_consistency.py-238-        np.testing.assert_allclose(
tests/unit/test_geometry_consistency.py-239-            vals[:3], [a.sum(), (a * a).sum(), np.abs(a).max()], rtol=1e-12)
tests/unit/test_geometry_consistency.py-240-
tests/unit/test_geometry_consistency.py-241-    def test_permutation_of_a_mask_is_REJECTED_end_to_end(self, multiproc):
tests/unit/test_geometry_consistency.py-242-        """The property the byte digest exists for, through the real helper.
--
tests/unit/test_geometry_consistency.py-253-        assert a.sum() == perm.sum() and a.shape == perm.shape
tests/unit/test_geometry_consistency.py-254-
tests/unit/test_geometry_consistency.py-255-        # touch ONLY the 3-wide value payload; leave the 8-wide struct alone
tests/unit/test_geometry_consistency.py-256-        multiproc(override=lambda row: (
tests/unit/test_geometry_consistency.py-257-            _make_vals_for(perm) if row.shape == (3,) else None))
tests/unit/test_geometry_consistency.py-258-        with pytest.raises(RuntimeError, match="DIVERGES"):
tests/unit/test_geometry_consistency.py:259:            broadcast_checked(a, "wet_mask", context="t")
tests/unit/test_geometry_consistency.py-260-
tests/unit/test_geometry_consistency.py-261-    def test_agreeing_processes_pass(self, multiproc):
tests/unit/test_geometry_consistency.py-262-        multiproc()
tests/unit/test_geometry_consistency.py-263-        a = np.linspace(0, 1, 12)
tests/unit/test_geometry_consistency.py:264:        out = broadcast_checked(a, "area", context="t")
tests/unit/test_geometry_consistency.py-265-        np.testing.assert_array_equal(out, a)
tests/unit/test_geometry_consistency.py-266-
tests/unit/test_geometry_consistency.py-267-    def test_float_divergence_beyond_rtol_raises(self, multiproc):
tests/unit/test_geometry_consistency.py-268-        """Only the VALUE payload is perturbed, so this cannot pass on a
tests/unit/test_geometry_consistency.py-269-        struct mismatch instead."""
tests/unit/test_geometry_consistency.py-270-        multiproc(override=lambda row: row * 1.1 if row.shape == (3,) else None)
tests/unit/test_geometry_consistency.py-271-        with pytest.raises(RuntimeError, match="DIVERGES"):
tests/unit/test_geometry_consistency.py:272:            broadcast_checked(np.linspace(1, 2, 10), "area", context="t")
tests/unit/test_geometry_consistency.py-273-
tests/unit/test_geometry_consistency.py-274-    def test_float_drift_within_rtol_is_ACCEPTED(self, multiproc):
tests/unit/test_geometry_consistency.py-275-        """The whole reason for the broadcast: ULP-scale autotune drift must
tests/unit/test_geometry_consistency.py-276-        NOT raise, or every large multi-process run fails spuriously."""
tests/unit/test_geometry_consistency.py-277-        multiproc(override=lambda row:
tests/unit/test_geometry_consistency.py-278-                  row * (1.0 + 1e-9) if row.shape == (3,) else None)
tests/unit/test_geometry_consistency.py:279:        out = broadcast_checked(np.linspace(1, 2, 10), "area", context="t")
tests/unit/test_geometry_consistency.py-280-        assert out is not None
tests/unit/test_geometry_consistency.py-281-
tests/unit/test_geometry_consistency.py-282-    def test_nonfinite_count_is_structural(self, multiproc):
tests/unit/test_geometry_consistency.py-283-        """A NaN on one process only must not be averaged away."""
tests/unit/test_geometry_consistency.py-284-        fake = multiproc()
tests/unit/test_geometry_consistency.py-285-        a = np.array([1.0, 2.0, np.nan, 4.0])
tests/unit/test_geometry_consistency.py:286:        broadcast_checked(a, "area", context="t")
tests/unit/test_geometry_consistency.py-287-        struct = fake.seen[-2]
tests/unit/test_geometry_consistency.py-288-        assert struct[5] == 1.0, "non-finite count must be carried in struct"
tests/unit/test_geometry_consistency.py-289-
tests/unit/test_geometry_consistency.py-290-
tests/unit/test_geometry_consistency.py-291-def _make_vals_for(arr):
tests/unit/test_geometry_consistency.py-292-    """The 3-wide vals payload `broadcast_checked` would emit for ``arr``."""
--
tests/unit/test_geometry_consistency.py-307-        np.array([1.0, np.nan, 3.0]),
tests/unit/test_geometry_consistency.py-308-        np.zeros((2, 3, 4)),
tests/unit/test_geometry_consistency.py-309-        np.zeros((5,)),
tests/unit/test_geometry_consistency.py-310-    ])
tests/unit/test_geometry_consistency.py-311-    def test_every_payload_has_identical_shape(self, multiproc, arr):
tests/unit/test_geometry_consistency.py-312-        fake = multiproc()
tests/unit/test_geometry_consistency.py:313:        broadcast_checked(arr, "f", context="t")
tests/unit/test_geometry_consistency.py-314-        struct, vals = fake.seen[-2], fake.seen[-1]
tests/unit/test_geometry_consistency.py-315-        assert struct.shape == (8,), struct.shape
tests/unit/test_geometry_consistency.py-316-        assert vals.shape == (3,), vals.shape
tests/unit/test_geometry_consistency.py-317-
tests/unit/test_geometry_consistency.py-318-    def test_schema_gate_covers_dtype_class_and_ndim(self, multiproc):
tests/unit/test_geometry_consistency.py-319-        """Without these terms a bool-on-one-rank / float-on-another field
--
tests/unit/test_geometry_consistency.py-1433-                   if any(isinstance(c, ast.Call)
tests/unit/test_geometry_consistency.py-1434-                          and isinstance(c.func, ast.Name)
tests/unit/test_geometry_consistency.py-1435-                          and c.func.id == "broadcast_checked"
tests/unit/test_geometry_consistency.py-1436-                          for c in ast.walk(p))]
tests/unit/test_geometry_consistency.py-1437-        assert guarded, (
tests/unit/test_geometry_consistency.py-1438-            "every replicated geometry device_put must take a "
tests/unit/test_geometry_consistency.py:1439:            "broadcast_checked(...) value as its argument")
--
packages/coupler/legoesm/driver/sharded_operator_split_step.py-477-    _ordered = list(raw)
packages/coupler/legoesm/driver/sharded_operator_split_step.py-478-    assert_schema_agrees(_ordered, n_dev,
packages/coupler/legoesm/driver/sharded_operator_split_step.py-479-                         context="make_sharded_operator_split_step",
packages/coupler/legoesm/driver/sharded_operator_split_step.py-480-                         arrays=[raw[n] for n in _ordered])
packages/coupler/legoesm/driver/sharded_operator_split_step.py-481-    stacks = {
packages/coupler/legoesm/driver/sharded_operator_split_step.py-482-        name: jax.device_put(
packages/coupler/legoesm/driver/sharded_operator_split_step.py:483:            jnp.asarray(broadcast_checked(
packages/coupler/legoesm/driver/sharded_operator_split_step.py-484-                raw[name], name,
packages/coupler/legoesm/driver/sharded_operator_split_step.py-485-                context="make_sharded_operator_split_step")),
packages/coupler/legoesm/driver/sharded_operator_split_step.py-486-            rep)
packages/coupler/legoesm/driver/sharded_operator_split_step.py-487-        for name in _ordered
packages/coupler/legoesm/driver/sharded_operator_split_step.py-488-    }
packages/coupler/legoesm/driver/sharded_operator_split_step.py-489-    _cache = {}
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-486-    # would desynchronize the per-field gathers rather than fail cleanly.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-487-    ordered_names = list(raw)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-488-    assert_schema_agrees(ordered_names, n_dev,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-489-                         context="make_sharded_atm_latlon_step",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-490-                         arrays=[raw[n] for n in ordered_names])
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-491-    raw = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:492:        name: jnp.asarray(broadcast_checked(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-493-            raw[name], name, context="make_sharded_atm_latlon_step"))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-494-        for name in ordered_names
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-495-    }
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-496-    spec_of = lat_spec if shard_geometry else (lambda _arr: P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-497-    stacks = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-498-        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1580-    # fingerprint must describe this process's whole mesh, not one axis.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1581-    ordered_names = list(raw)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1582-    assert_schema_agrees(ordered_names, p_lat * p_lon,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1583-                         context="make_sharded_atm_latlon_step_2d",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1584-                         arrays=[raw[n] for n in ordered_names])
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1585-    raw = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1586:        name: jnp.asarray(broadcast_checked(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1587-            raw[name], name, context="make_sharded_atm_latlon_step_2d"))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1588-        for name in ordered_names
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1589-    }
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1590-    spec_of = tile_spec if shard_geometry else (lambda _arr: P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1591-    stacks = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1592-        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
28-
29-from contextlib import contextmanager
30-
31-import jax
32-import jax.numpy as jnp
33-from jax.sharding import NamedSharding, PartitionSpec as P
34-
35-from legoesm.parallel.geometry_consistency import (
36:    FLAG_ABSENT, assert_flags_agree, assert_schema_agrees, broadcast_checked,
37-    coerce_bool, coerce_count, config_digest48, name_digest48,
38-    tree_schema_digest48)
39-
40-from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
41-    CGridLatLonHydrostaticState,
42-)
43-
44-
--
484-    # entries are CONDITIONAL on ``model._polar_mask``, so a per-process
485-    # difference in that one setting changes the field LIST itself, which
486-    # would desynchronize the per-field gathers rather than fail cleanly.
487-    ordered_names = list(raw)
488-    assert_schema_agrees(ordered_names, n_dev,
489-                         context="make_sharded_atm_latlon_step",
490-                         arrays=[raw[n] for n in ordered_names])
491-    raw = {
492:        name: jnp.asarray(broadcast_checked(
493-            raw[name], name, context="make_sharded_atm_latlon_step"))
494-        for name in ordered_names
495-    }
496-    spec_of = lat_spec if shard_geometry else (lambda _arr: P())
497-    stacks = {
498-        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
499-        for name, arr in raw.items()
500-    }
--
1578-    # per-process recompute, same replicated-device_put bit-identity assert.
1579-    # n_dev is the FULL device count here (p_lat * p_lon): the schema
1580-    # fingerprint must describe this process's whole mesh, not one axis.
1581-    ordered_names = list(raw)
1582-    assert_schema_agrees(ordered_names, p_lat * p_lon,
1583-                         context="make_sharded_atm_latlon_step_2d",
1584-                         arrays=[raw[n] for n in ordered_names])
1585-    raw = {
1586:        name: jnp.asarray(broadcast_checked(
1587-            raw[name], name, context="make_sharded_atm_latlon_step_2d"))
1588-        for name in ordered_names
1589-    }
1590-    spec_of = tile_spec if shard_geometry else (lambda _arr: P())
1591-    stacks = {
1592-        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
1593-        for name, arr in raw.items()
1594-    }
600-    assert_flags_agree(_OCEAN_SPMD_ENTRY_FLAGS, flags, context=where)
601-    # AFTER the collective only: symmetric on every rank (see atm twin).
602-    for label, problem in problems:
603-        raise ValueError(f"{where}: {label} {problem}")
604-
605-
606-# Ordered flag names for the PER-INVOCATION gate on the returned ocean SPMD
607-# callable. STATIC tuple: fixed width, never rank-local.
608-_OCEAN_CALL_ENTRY_FLAGS = (
609-    "has_mesh", "n_dev", "axis_names", "axis_sizes",
610-    "state_schema", "has_forcing", "forcing_schema",
611-)
612-
613-
614:def _agree_ocean_spmd_call(mesh, state, forcing, *, where: str) -> None:
615-    """Agree a returned ocean SPMD callable's per-CALL inputs, FIRST statement.
616-
617-    #1362 round 4, blocker 7.  ``sharded_step`` runs ``_validate_forcing_layout``
618-    and builds a rank-local cache key BEFORE entering its ``shard_map``: a
619-    forcing layout that is invalid on one rank only makes that rank raise while
620-    its peers enter the collective program -- a hang.  The state + forcing leaf
621-    SCHEMA is agreed too, because ``in_specs``/``out_specs`` are derived from
622-    it, so two processes with different optional fields compile different
623-    programs.
624-
625-    Cost: one small allgather per CALL, and an exact no-op under a single
626-    process.  See the atmosphere twin ``_agree_spmd_call`` for why gating only
627-    on cache misses is NOT a valid optimisation.
628-    """
--
870-                    f"(n_lat, n_lon[, nlev]) to shard on the lat axis.")
871-
872-    def sharded_step(state, dt, freshwater=None, surface_forcing=None,
873-                     sponge=None, t_seconds=None, aux=None):
874-        # ``aux``: the sharded geometry+vmask stacks. When this wrapper runs
875-        # INSIDE an outer trace (a bench/driver jit/scan — jit-of-jit
876-        # inlines the inner call), concrete closure arrays become
877-        # OUTER-trace constants whose value the MLIR handler cannot fetch
878-        # for non-addressable arrays (broken since #1370-iii sharded the
879-        # stacks). Outer-jit callers MUST thread ``step.aux`` through their
880-        # jit boundary as an ARGUMENT and pass it back here.
881-        # ONE forcing operand: None fields drop out of the pytree structure,
882-        # so specs derived by tree.map skip them automatically and the
883-        # structure key below distinguishes every None<->array combination.
884:        _agree_ocean_spmd_call(
885-            mesh, state, (freshwater, surface_forcing, sponge, t_seconds),
886-            where="make_sharded_ocean_step.step")
887-        forcing = (freshwater, surface_forcing, sponge, t_seconds)
888-        _validate_forcing_layout((freshwater, surface_forcing, sponge))
889-        # Cache key = the state's AND forcing's pytree STRUCTURE, plus the
890-        # forcing leaves' RANKS: in_specs/out_specs are derived from them,
891-        # so a later call with a different structure (an optional field
892-        # flipping None <-> Field, a sea-ice lane populating sf.salt_flux,
893-        # the restoring lane passing no forcing at all) OR a same-field
894-        # rank change (SpongeForcing.gamma is legitimately 2-D horizontal
895-        # OR 3-D full-rank — same structure, different _lat_spec; codex r1
896-        # #1) must rebuild the shard_map rather than reuse stale specs.
897-        forcing_ndims = tuple(
898-            int(getattr(leaf, "ndim", np.ndim(leaf)))
--
979-
980-    ``mesh is None`` ⇒ the plain single-device ``model.step`` (no scatter/gather).
981-    """
982-    # FIRST statement: agree every rank-local input before ANY
983-    # rank-local check can raise or return (codex round-2).
984-    _agree_ocean_spmd_entry(model, mesh, where="make_sharded_ocean_step_global")
985-    if mesh is None:                   # single-device: plain step
986-        return lambda state, dt, surface_forcing=None, freshwater=None: (
987-            model.step(state, dt, freshwater=freshwater,
988-                       surface_forcing=surface_forcing))
989-
990-    inner = make_sharded_ocean_step(model, mesh)
991-
992-    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
993:        _agree_ocean_spmd_call(mesh, state, (surface_forcing, freshwater),
994-                               where="make_sharded_ocean_step_global.step")
995-        # Scatter the global state AND forcing to the band layout
996-        # explicitly (the old comment claimed inner sharded the forcing;
997-        # it forwarded it global and relied on implicit JIT input
998-        # placement — jax's whole-array device_put assert under
999-        # multicontroller, the nd-linear wall this module removes).
1000-        ss = shard_state_latlon(state, mesh)
1001-        ss = inner(ss, dt,
1002-                   surface_forcing=shard_forcing_latlon(surface_forcing, mesh),
1003-                   freshwater=shard_forcing_latlon(freshwater, mesh))
1004-        return gather_state_latlon(ss, mesh)
1005-
1006-    return sharded_step_global
781-
782-
783-# Ordered flag names for the PER-INVOCATION gate on a returned SPMD callable.
784-# STATIC tuple: the payload width is fixed by this literal, never by data.
785-_CALL_ENTRY_FLAGS = (
786-    "has_mesh", "n_dev", "axis_names", "axis_sizes",
787-    "state_schema", "has_phys_state", "phys_state_schema",
788-)
789-
790-
791:def _agree_spmd_call(mesh, state, phys_state, *, where: str) -> None:
792-    """Agree a returned SPMD callable's per-CALL inputs, as its FIRST statement.
793-
794-    #1362 round 4, blocker 7.  The factories are gated, but the CLOSURES they
795-    return are public entry points in their own right and they perform
796-    rank-local checks BEFORE their collective program:
797-    ``refuse_unthreaded_stateful_physics`` raises when ``phys_state is None``
798-    on one rank while a peer carrying a state walks into ``shard_map``; the
799-    2-D closure raises ``NotImplementedError`` on a non-None carry; the cache
800-    key is built from a rank-local pytree structure.  Each of those is a raise
801-    ahead of a collective its peers enter -- a HANG.
--
973-    # [3079, 1.4, 1.2, 1.1, 1.1] ms — first call compiles, the rest hit the
974-    # cache. ``dt`` is a TRACED operand (not a closure constant) so a changing dt
975-    # does not retrigger compilation. The grid-tracer concern that kept
976-    # _step_cgrid_impl un-jitted does NOT bite here: band_geom's STATIC scalar
977-    # fields stay concrete (template._replace only swaps the array fields), and
978-    # the SPMD operator retrofits removed the trace-time static-bool checks on
979-    # the cut/pole branches.
980-    _cache = {}
981-
982-    def sharded_step(c_state, dt, phys_state=None):
983:        _agree_spmd_call(mesh, c_state, phys_state,
984-                         where="make_sharded_atm_latlon_step.step")
985-        refuse_unthreaded_stateful_physics(
986-            physics_fn, phys_state, where="atm lat-band SPMD step")
987-        # Cache key = (state pytree STRUCTURE, phys_state pytree structure):
988-        # in_specs/out_specs derive from both, so a structure change (optional
989-        # field None <-> Field, or the phys carry appearing/disappearing) must
990-        # rebuild the shard_map rather than reuse stale specs (codex finding,
991-        # ocean-twin parity). ``phys_state`` threading is the AIMIP-branch
992-        # feature main lacks (main rejects a non-None carry here).
993-        key = (jax.tree.structure(c_state),
--
1135-        out, n_left = unroll_to_dtype_fixed_point(
1136-            _step1, state_local, n_steps)
1137-        if n_left > 0:
1138-            out, _ = jax.lax.scan(lambda s, _x: (_step1(s), None),
1139-                                  out, xs=None, length=n_left)
1140-        return out, state_finite_scalar(out, axis=axis)
1141-
1142-    _cache = {}
1143-
1144-    def segment(c_state, dt, phys_state=None):
1145:        _agree_spmd_call(mesh, c_state, phys_state,
1146-                         where="make_sharded_atm_latlon_segment.segment")
1147-        _refuse_carry(phys_state)
1148-        # Cache key = state pytree STRUCTURE (in/out specs derive from it) —
1149-        # same doctrine as the per-step factory.
1150-        key = jax.tree.structure(c_state)
1151-        fn = _cache.get(key)
1152-        if fn is None:
1153-            in_spec = jax.tree.map(lat_spec, c_state)
1154-            fn = jax.jit(shard_map(
1155-                _seg_body, mesh=mesh,
--
1742-        model, template, array_field_names, perm_north, p_lon, physics_fn,
1743-        shard_geometry)
1744-
1745-    def _body(state_local, stacks_local, dt):
1746-        out, _ = tile_step(state_local, stacks_local, dt, None)
1747-        return out
1748-
1749-    _cache = {}
1750-
1751-    def sharded_step(c_state, dt, phys_state=None):
1752:        _agree_spmd_call(mesh, c_state, phys_state,
1753-                         where="make_sharded_atm_latlon_step_2d.step")
1754-        refuse_unthreaded_stateful_physics(
1755-            physics_fn, phys_state, where="atm lat-lon 2-D SPMD step")
1756-        if phys_state is not None:
1757-            raise NotImplementedError(
1758-                "make_sharded_atm_latlon_step_2d: a stateful PhysicsState "
1759-                "carry is not 2-D-tile-routed (its (ncol, ...) leaves "
1760-                "flatten lat-major over the GLOBAL grid — a contiguous "
1761-                "dim-0 shard is a lat band, not a 2-D tile).  Thread the "
1762-                "carry through the 1-D make_sharded_atm_latlon_step.")
--
1835-        out, n_left = unroll_to_dtype_fixed_point(
1836-            _step1, state_local, n_steps)
1837-        if n_left > 0:
1838-            out, _ = jax.lax.scan(lambda s, _x: (_step1(s), None),
1839-                                  out, xs=None, length=n_left)
1840-        return out, state_finite_scalar(out, axis=("lat", "lon"))
1841-
1842-    _cache = {}
1843-
1844-    def segment(c_state, dt, phys_state=None):
1845:        _agree_spmd_call(mesh, c_state, phys_state,
1846-                         where="make_sharded_atm_latlon_segment_2d.segment")
1847-        _refuse_carry(phys_state)
1848-        key = jax.tree.structure(c_state)
1849-        fn = _cache.get(key)
1850-        if fn is None:
1851-            in_spec = jax.tree.map(tile_spec, c_state)
1852-            fn = jax.jit(shard_map(
1853-                _seg_body, mesh=mesh,
1854-                in_specs=(in_spec, stacks_spec, P()),
1855-                out_specs=(in_spec, P()), check_vma=False))

exec
/usr/bin/bash -lc "sed -n '410,515p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py; sed -n '1500,1605p' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py; sed -n '430,505p' packages/coupler/legoesm/driver/sharded_operator_split_step.py; sed -n '614,680p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py; sed -n '680,765p' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 385ms:

    ``replicated_per_device_bytes``: the historical layout — every device
    holds the full all-band stack of every ``LatLonGrid`` array field
    (:func:`atm_grid_array_field_names`; dominated by the five
    ``(n_lat, n_lon)`` 2-D fields ``lat2d, lon2d, f, dx, area``).
    ``sharded_per_device_bytes``: the ``shard_geometry=True`` layout — each
    device holds only its own band's slice (exactly ``replicated /
    n_devices``; the stack leading dim is ``n_devices`` and bands are
    uniform).  Excludes the optional polar-filter mask stacks (two
    ``(n_lat,)``-scale vectors when the filter is on) — negligible next to
    the 2-D fields and absent in the default configs.
    """
    band_grids = build_band_grids_atm(grid, n_devices)
    template = band_grids[0]
    names = atm_grid_array_field_names(template)
    band_bytes = sum(int(jnp.asarray(getattr(template, n)).nbytes)
                     for n in names)
    total = band_bytes * n_devices          # the (n_dev, band...) stacks
    return {
        "n_geometry_fields": len(names),
        "replicated_per_device_bytes": int(total),
        "sharded_per_device_bytes": int(total // n_devices),
    }


def _build_geometry_stacks(model, mesh, n_dev: int, shard_geometry: bool):
    """Stack every band's ``LatLonGrid`` array fields (+ the optional
    polar-filter masks) over a leading band axis and lay them out on ``mesh``.

    ``shard_geometry=False`` (the historical layout): every stack is
    ``device_put`` REPLICATED (``P()``) — each device holds ALL bands'
    geometry and the body indexes its own band at ``axis_index``.

    ``shard_geometry=True`` (M2b): every stack is sharded ``P("lat", ...)``
    on the leading band axis — each device holds ONLY its own band's slice
    (leading extent 1 inside the shard_map body, static index ``[0]``).  The
    VALUES the body consumes are identical either way (the same band slice),
    so the step numerics are bit-unchanged; only the residency changes
    (per-device geometry bytes drop by ``n_dev`` —
    :func:`atm_latlon_geometry_bytes`).

    Returns ``(template, array_field_names, stacks, stacks_spec)``.
    """
    grid = model.grid
    band_grids = build_band_grids_atm(grid, n_dev)
    template = band_grids[0]
    array_field_names = atm_grid_array_field_names(template)
    raw = {
        name: jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
                        axis=0)
        for name in array_field_names
    }
    # Per-band polar-filter masks (only when the filter is on): slice the
    # global masks [s:e] (cell) / [s:e+1] (v-face stagger) and stack.  Absent
    # otherwise -> the body passes None -> _step_cgrid_impl resolves to
    # self._polar_mask (also None), no filter.
    nl = int(grid.n_lat) // n_dev
    if model._polar_mask is not None:
        raw["__polar_mask"] = jnp.stack(
            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(n_dev)],
            axis=0)
        raw["__polar_mask_v"] = jnp.stack(
            [model._polar_mask_v[r * nl:r * nl + nl + 1] for r in range(n_dev)],
            axis=0)
    # #1362: the band geometry above is RECOMPUTED per process from the same
    # config, and per-process XLA autotuning on device-derived grid fields
    # makes the last ULPs differ at larger sizes -- which trips the
    # bit-identical assert inside a replicated device_put (the ocean lane hit
    # exactly this at LL576 np=4; this lane hit it at LL768).  Verify
    # cross-process agreement, then broadcast process 0's bytes.  Guarded, not
    # blind: a REAL divergence (different polar masks = different filtering =
    # different physics) RAISES instead of being masked by process 0.
    #
    # The schema gate matters more here than in the ocean lane: the polar-mask
    # entries are CONDITIONAL on ``model._polar_mask``, so a per-process
    # difference in that one setting changes the field LIST itself, which
    # would desynchronize the per-field gathers rather than fail cleanly.
    ordered_names = list(raw)
    assert_schema_agrees(ordered_names, n_dev,
                         context="make_sharded_atm_latlon_step",
                         arrays=[raw[n] for n in ordered_names])
    raw = {
        name: jnp.asarray(broadcast_checked(
            raw[name], name, context="make_sharded_atm_latlon_step"))
        for name in ordered_names
    }
    spec_of = lat_spec if shard_geometry else (lambda _arr: P())
    stacks = {
        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
        for name, arr in raw.items()
    }
    stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}
    return template, array_field_names, stacks, stacks_spec


def _make_band_step_body(model, template, array_field_names, axis,
                         perm_north, physics_fn, shard_geometry: bool):
    """One band's un-jitted C-grid step body — shared by the per-step
    shard_map (:func:`make_sharded_atm_latlon_step`) and the compiled segment
    scan (:func:`make_sharded_atm_latlon_segment`) so the band numerics are
    written ONCE.

    Returns ``band_step(state_local, stacks_local, dt, ps_local) ->
    (state_out_local, ps_out)`` operating on the band-local ``v_lower``
    layout.  ``shard_geometry`` selects the geometry index (STATIC Python
    bool, feature-gating exception): sharded stacks arrive with a leading
    """
    from legoesm.parallel.latlon_mpi import (
        make_latlon_2d_layout, slice_latlon_grid_to_block_2d)
    n_lat, n_lon = int(grid.n_lat), int(grid.n_lon)
    if p_lat < 1 or p_lon < 1:
        raise ValueError(
            f"p_lat/p_lon must be >= 1, got ({p_lat}, {p_lon})")
    if n_lat % p_lat != 0 or n_lon % p_lon != 0:
        raise ValueError(
            f"atm 2-D SPMD tiling requires n_lat ({n_lat}) % p_lat "
            f"({p_lat}) == 0 and n_lon ({n_lon}) % p_lon ({p_lon}) == 0 so "
            f"every tile is uniform (one shard_map program).")
    nl, w = n_lat // p_lat, n_lon // p_lon
    if (p_lat > 1 and nl < 2) or (p_lon > 1 and w < 2):
        raise ValueError(
            f"atm 2-D SPMD tiling: tiles must keep >= 2 cells per split "
            f"dimension (PPM halo=2 single-hop exchange); got "
            f"{nl}x{w} tiles from ({p_lat}, {p_lon}) on {n_lat}x{n_lon}.")
    fold = getattr(grid, "fold", None)
    return [
        [
            slice_latlon_grid_to_block_2d(
                grid,
                make_latlon_2d_layout(
                    r * p_lon + c, p_lat, p_lon, n_lat, n_lon, fold),
                skip_total_area_reduce=True)
            for c in range(p_lon)
        ]
        for r in range(p_lat)
    ]


def _build_geometry_stacks_2d(model, mesh, p_lat: int, p_lon: int,
                              shard_geometry: bool):
    """Stack every tile's ``LatLonGrid`` array fields (+ the optional
    polar-filter masks at ``p_lon == 1``) over LEADING ``(p_lat, p_lon)``
    tile axes and lay them out on ``mesh`` — the 2-D twin of
    :func:`_build_geometry_stacks`.

    ``shard_geometry=True``: stacks are sharded ``P("lat", "lon", ...)`` on
    the tile axes — each device holds ONLY its own tile's slice (leading
    extents ``(1, 1)`` inside the body, static index ``[0, 0]``).
    ``shard_geometry=False``: replicated (``P()``) stacks, indexed at
    ``(axis_index("lat"), axis_index("lon"))``.  Same tile VALUES either way.

    Returns ``(template, array_field_names, stacks, stacks_spec)``.
    """
    grid = model.grid
    tile_grids = build_tile_grids_atm_2d(grid, p_lat, p_lon)
    template = tile_grids[0][0]
    array_field_names = atm_grid_array_field_names(template)
    raw = {
        name: jnp.stack([
            jnp.stack([jnp.asarray(getattr(tile_grids[r][c], name))
                       for c in range(p_lon)], axis=0)
            for r in range(p_lat)
        ], axis=0)
        for name in array_field_names
    }
    # Per-tile polar-filter masks: p_lon > 1 is refused by the factories
    # (the filter rfft's the full lon circle); at p_lon == 1 the stacks
    # mirror the band layout with a singleton lon-tile axis.
    nl = int(grid.n_lat) // p_lat
    if model._polar_mask is not None:
        if p_lon > 1:
            raise NotImplementedError(
                "atm 2-D SPMD tiling: use_polar_filter=True with p_lon > 1 "
                "is not wired — the polar filter FFTs the full longitude "
                "circle (needs a lon-gather FFT).  Use p_lon == 1 or "
                "disable the filter.")
        raw["__polar_mask"] = jnp.stack(
            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(p_lat)],
            axis=0)[:, None]
        raw["__polar_mask_v"] = jnp.stack(
            [model._polar_mask_v[r * nl:r * nl + nl + 1]
             for r in range(p_lat)],
            axis=0)[:, None]
    # #1362, 2-D twin of the guard in _build_geometry_stacks -- same
    # per-process recompute, same replicated-device_put bit-identity assert.
    # n_dev is the FULL device count here (p_lat * p_lon): the schema
    # fingerprint must describe this process's whole mesh, not one axis.
    ordered_names = list(raw)
    assert_schema_agrees(ordered_names, p_lat * p_lon,
                         context="make_sharded_atm_latlon_step_2d",
                         arrays=[raw[n] for n in ordered_names])
    raw = {
        name: jnp.asarray(broadcast_checked(
            raw[name], name, context="make_sharded_atm_latlon_step_2d"))
        for name in ordered_names
    }
    spec_of = tile_spec if shard_geometry else (lambda _arr: P())
    stacks = {
        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
        for name, arr in raw.items()
    }
    stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}
    return template, array_field_names, stacks, stacks_spec


def _make_tile_step_body_2d(model, template, array_field_names,
                            perm_north, p_lon: int, physics_fn,
                            shard_geometry: bool):
    """One tile's un-jitted C-grid step body — the 2-D twin of
    :func:`_make_band_step_body`, shared by the per-step and segment 2-D
    factories so the tile numerics are written ONCE.

            carry, T_new, u_new, v_new, p_s_new, need_rad, doy, sod,
            band_statics)
        return finalize_split_step(carry, lz, band_statics)

    if mesh is None:                       # serial / single-device
        def serial_step(carry, forcing):
            return _one_step(carry, model.grid, forcing)
        return serial_step

    # --- lat-band SPMD ---
    n_dev = mesh.devices.size
    axis = mesh.axis_names[0]
    grid = model.grid
    fold = getattr(grid, "fold", None)
    if fold is not None and bool(getattr(fold, "is_active", False)):
        raise NotImplementedError(
            "operator-split SPMD: tripole north-fold is a follow-up.")

    band_grids = build_band_grids_atm(grid, n_dev)
    template = band_grids[0]
    afn = atm_grid_array_field_names(template)
    rep = NamedSharding(mesh, P())
    raw = {
        name: jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids], 0)
        for name in afn
    }
    # Per-band polar-filter masks (only when the filter is on): slice the global
    # masks [r*nl:(r+1)*nl] (cell) / [r*nl:r*nl+nl+1] (v-face stagger) and stack,
    # exactly as make_sharded_atm_latlon_step does. Absent otherwise -> _body
    # passes None -> _step_cgrid_impl resolves self._polar_mask (also None).
    nl = int(grid.n_lat) // n_dev
    if model._polar_mask is not None:
        raw["__polar_mask"] = jnp.stack(
            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(n_dev)], 0)
        raw["__polar_mask_v"] = jnp.stack(
            [model._polar_mask_v[r * nl:r * nl + nl + 1] for r in range(n_dev)],
            0)
    # #1362 (round 4): the band geometry above is RECOMPUTED per process from
    # the same config, and per-process XLA autotuning on device-derived grid
    # fields makes the last ULPs differ at larger sizes -- which trips the
    # bit-identical assert inside the REPLICATED ``device_put`` below.  This
    # is the SAME defect fixed in the atm and ocean lanes; this lane reuses
    # ``build_band_grids_atm`` so it inherits the hazard verbatim.  Verify
    # cross-process agreement, then broadcast process 0's bytes -- GUARDED,
    # so a REAL divergence (different polar masks = different filtering =
    # different physics) RAISES instead of being papered over by process 0.
    # Single-process: both helpers short-circuit, so this is a no-op.
    _ordered = list(raw)
    assert_schema_agrees(_ordered, n_dev,
                         context="make_sharded_operator_split_step",
                         arrays=[raw[n] for n in _ordered])
    stacks = {
        name: jax.device_put(
            jnp.asarray(broadcast_checked(
                raw[name], name,
                context="make_sharded_operator_split_step")),
            rep)
        for name in _ordered
    }
    _cache = {}

    def sharded_split_step(carry, forcing):
        _agree_opsplit_mesh_entry(
            mesh, carry, forcing,
            where="make_sharded_operator_split_step.step")
        # Cache key = pytree STRUCTURE (None<->array flips) PLUS every leaf's
        # shape/dtype: the in/out_specs derive from arr.ndim + arr.shape[0], so a
        # shape/rank change under an unchanged structure would otherwise reuse a
        # stale P() vs P("lat") spec (codex).
        def _leaf_sig(pytree):
            return tuple((getattr(l, "shape", ()), getattr(l, "dtype", None))
                         for l in jax.tree.leaves(pytree))
        key = (jax.tree.structure(carry), jax.tree.structure(forcing),
               _leaf_sig(carry), _leaf_sig(forcing))
        fn = _cache.get(key)
        if fn is None:
def _agree_ocean_spmd_call(mesh, state, forcing, *, where: str) -> None:
    """Agree a returned ocean SPMD callable's per-CALL inputs, FIRST statement.

    #1362 round 4, blocker 7.  ``sharded_step`` runs ``_validate_forcing_layout``
    and builds a rank-local cache key BEFORE entering its ``shard_map``: a
    forcing layout that is invalid on one rank only makes that rank raise while
    its peers enter the collective program -- a hang.  The state + forcing leaf
    SCHEMA is agreed too, because ``in_specs``/``out_specs`` are derived from
    it, so two processes with different optional fields compile different
    programs.

    Cost: one small allgather per CALL, and an exact no-op under a single
    process.  See the atmosphere twin ``_agree_spmd_call`` for why gating only
    on cache misses is NOT a valid optimisation.
    """
    names, sizes = _ocean_mesh_axis_terms(mesh)
    assert_flags_agree(_OCEAN_CALL_ENTRY_FLAGS, (
        float(mesh is not None),
        float(mesh.devices.size if mesh is not None else 0),
        name_digest48(names),
        name_digest48(sizes),
        tree_schema_digest48(state),
        float(forcing is not None),
        (tree_schema_digest48(forcing) if forcing is not None
         else FLAG_ABSENT),
    ), context=where)


def make_sharded_ocean_step(model, mesh):
    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
    sponge=None, t_seconds=None) -> state`` running ``model.step``
    lat-band-SPMD.

    The forcing channels mirror ``model.step``'s keyword surface: pass
    pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
    ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
    is a replicated scalar like ``dt``).  ``None`` forcing keeps the
    dynamics-only program; each distinct None<->populated combination
    compiles (and caches) its own executable.

    Parameters
    ----------
    model
        ``LatLonCGridOceanModel`` (latlon geometry; tripole north-fold is a
        follow-up — raises ``NotImplementedError`` if ``model.grid.fold`` is
        active).
    mesh
        A 1-D ``jax.sharding.Mesh`` with axis name ``"lat"`` (from
        ``legoesm.parallel.mesh.create_latlon_mesh(...).mesh``).

    Notes
    -----
    Build the ``N`` band geometries + band vertex masks host-side, stack their
    ARRAY fields into a replicated pytree, and index by
    ``jax.lax.axis_index("lat")`` in the ``shard_map`` body (the geometry SCALAR
    fields stay static — see the module docstring).  ``dt`` is a TRACED,
    replicated operand and the jitted ``shard_map`` is built once and cached
    (a per-call rebuild re-traced the whole ocean step every call — see the
    ``_cache`` note below).  ``check_vma=False`` (the JAX >= 0.8
    replication check) because the band halo intentionally reads neighbour-rank
    data (replication-unaware).

    The state must be laid out with :func:`shard_state_latlon` (``v`` /
    ``v_mask`` carried as the ``n_lat``-row ``v_lower``).  The body reconstructs
    each band's ``nl+1`` v-faces, runs the step on the band geometry, and
    converts the result back to the ``v_lower`` representation.
    """
    """
    # FIRST statement: agree every rank-local input before ANY
    # rank-local check can raise or return (codex round-2).
    _agree_ocean_spmd_entry(model, mesh, where="make_sharded_ocean_step")
    if mesh is None:                   # single-device: plain step
        return lambda state, dt, **forcing_kwargs: model.step(
            state, dt, **forcing_kwargs)

    n_dev = mesh.devices.size
    axis = mesh.axis_names[0]

    # Tripole north-fold (scaling-audit item 4): SUPPORTED under the same
    # v-carrier contract as the regular grid.  Every fold-touching operator
    # is already uniform-program fold-capable (the data-dependent
    # ``north_fold_mask``/``apply_north_fold`` selection on
    # ``axis_index == N-1`` — gated by test_latlon_spmd_northfold), and
    # ``build_band_grids``' slicer keeps ``is_active`` rank-consistent with
    # the ``fold_j=-1`` sentinel off the north band.  The one structural
    # assumption is the v-carrier's: the TOP v-face row (the seam/cap row,
    # ``v[n_lat]``) must be WALL-MASKED so the in-body reconstruction's
    # zero row is exact — true for the cap-row convention of
    # ``create_synthetic_tripole`` and the eORCA masks (``v_mask[-1] == 0``;
    # the serial step keeps ``v[-1] == 0`` identically).  That contract is
    # asserted on the CONCRETE state in :func:`shard_state_latlon` — a live
    # (unmasked) seam v-row refuses loudly there instead of silently
    # reconstructing zeros here.

    # --- host-side band geometries + vertex masks (replicated, indexed in-body) ---
    band_grids = build_band_grids(model.grid, n_dev)
    band_vmasks = _build_band_vertex_masks(model, n_dev)
    template = band_grids[0]           # static-scalar source (uniform bands)
    array_field_names = _geom_array_field_names(template)

    # Stack each geometry ARRAY field over the band axis (rank 0..N-1) and
    # SHARD along that axis (#1370 stage (iii), codex round-18): each device
    # holds ONLY its own band's slab instead of the whole global stack —
    # this was one of the residual ~1.4-1.7 global-field-equivalents of
    # per-device residency left after the host-side-build fix (probe
    # 26524423). The leading axis has length n_dev, so P("lat") divides it
    # exactly; the body indexes its local slab at [0]. Values are unchanged
    # — same stack, different placement; the process-0 broadcast +
    # divergence guard below runs on HOST values and is placement-blind.
    rep = NamedSharding(mesh, P("lat"))

    def _replicated_put(arr, name):
        # (Name kept for history; this is a SHARDED P("lat") stack put.)
        # checked_shard_put replaces the broadcast_checked+device_put pair:
        # the broadcast's psum program is [n_processes, stack] (nd x 849 MB
        # at LL2304 — the @96/@128 wall), and a numpy device_put onto an
        # all-process sharding pays jax's whole-array assert_equal on top.
        # The per-band gate keeps the divergence contract (n_bands is
        # schema-gated just below, so payload widths agree). ONE shared
        # implementation: legoesm.parallel.geometry_consistency.
        return checked_shard_put(
            arr, name, rep, context="make_sharded_ocean_step",
            n_bands=n_dev)

    # Schema gate FIRST (one fixed-shape collective every process reaches):
    # a process-dependent field list or a mixed jax_enable_x64 setting would
    # otherwise desynchronize the per-field gathers below instead of failing
    # with a clear message.
    # Build the raw stacks FIRST so the schema gate can also cover each
    # field's dtype class and ndim -- those decide the per-field payload
    # shape below, so a bool-vs-float disagreement must fail HERE rather than
    # deadlock in the per-field gather.
    _raw_geom = {
        name: jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
                        axis=0)
        for name in array_field_names
    }
    _raw_vmask = jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0)
    _gate_names = [*array_field_names, "vertex_mask"]
    assert_schema_agrees(
        _gate_names, n_dev, context="make_sharded_ocean_step",
        arrays=[*(_raw_geom[n] for n in array_field_names), _raw_vmask])

    geom_stacks = {name: _replicated_put(_raw_geom[name], name)
                   for name in array_field_names}
    vmask_stack = _replicated_put(_raw_vmask, "vertex_mask")

    # Static perms for the v north-boundary-row ppermute (band r receives band
    # r+1's v_lower[0] = global v[e]; north band non-target receives 0).
    perm_north, _perm_south = latlon_band_perms(n_dev)

    def _body(state_local, forcing_local, geom_stacks_local,
              vmask_stack_local, dt):

exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 python -B -c 'import legoesm.parallel.geometry_consistency as gc; import legoesm.ocean.dynamics.sharded_ocean_step as os; print(gc.__file__); print(os.__file__)'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 127 in 316ms:
/usr/bin/bash: python: command not found

exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 python3 -B -c 'import legoesm.parallel.geometry_consistency as gc; import legoesm.ocean.dynamics.sharded_ocean_step as os; print(gc.__file__); print(os.__file__)'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 1 in 301ms:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'legoesm'

exec
/usr/bin/bash -lc "rg -n 'tool\\.pytest|pythonpath|packages/core|packages/ocean' pyproject.toml packages/core/pyproject.toml packages/ocean/pyproject.toml | sed -n '1,160p'; ls -la | sed -n '1,100p'; ls -la .venv 2>/dev/null | sed -n '1,40p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 538ms:
pyproject.toml:139:# (packages/core, dist name legoesm-core) and the root ``legoesm`` is the
pyproject.toml:179:"packages/core/legoesm/core/fv3_native_sw_core.py" = ["N803", "N806", "N801"]
pyproject.toml:180:"packages/core/legoesm/core/fv3_native_d_sw.py" = ["N803", "N806", "N801"]
pyproject.toml:181:"packages/core/legoesm/core/fv3_native_duo_sw_core.py" = ["N803", "N806", "N801"]
pyproject.toml:182:"packages/core/legoesm/grids/fv3_native_gridstruct.py" = ["N801", "N803", "N806"]
pyproject.toml:303:[tool.pytest.ini_options]
total 91284
drwxrws---+ 18 b381103 bd1083    20480 Aug  3 08:42 .
drwxrws---+  3 b381103 bd1083     4096 Jul 24 13:37 ..
drwxrws---+  6 b381103 bd1083     4096 Aug  3 08:42 .claude
-rw-rw----+  1 b381103 bd1083       88 Jul 24 13:37 .git
-rw-rw----+  1 b381103 bd1083       40 Jul 24 13:37 .gitattributes
drwxrws---+  3 b381103 bd1083     4096 Jul 24 13:37 .github
-rw-rw----+  1 b381103 bd1083     6408 Jul 24 13:37 .gitignore
drwxrws---+  2 b381103 bd1083     4096 Jul 24 13:37 .mplconfig
drwxrws---+ 20 b381103 bd1083     4096 Jul 26 13:37 .physics-validator
drwxr-xr-x+  3 b381103 bd1083     4096 Jul 24 19:08 .pytest_cache
drwxrws---+  2 b381103 bd1083     4096 Aug  3 08:42 .ralph
-rw-rw----+  1 b381103 bd1083     3397 Aug  3 08:42 .ralphrc
-rw-rw----+  1 b381103 bd1083      485 Jul 24 13:37 .slopbuster.yaml
-rw-rw----+  1 b381103 bd1083      938 Jul 24 13:37 .zenodo.json
-rw-rw----+  1 b381103 bd1083     8541 Jul 24 13:37 CHANGELOG.md
-rw-rw----+  1 b381103 bd1083     1233 Jul 24 13:37 CITATION.cff
-rw-rw----+  1 b381103 bd1083    60732 Aug  3 08:42 CLAUDE.md
-rw-rw----+  1 b381103 bd1083    31231 Jul 24 13:37 CMIP.md
-rw-rw----+  1 b381103 bd1083     1813 Jul 24 13:37 COMMERCIAL-LICENSE.md
-rw-rw----+  1 b381103 bd1083     6912 Jul 24 13:37 CONTRIBUTING.md
-rw-rw----+  1 b381103 bd1083     8023 Jul 24 13:37 FEDERATION.md
-rw-rw----+  1 b381103 bd1083     5260 Jul 24 13:37 LICENSE
-rw-rw----+  1 b381103 bd1083    36598 Jul 24 13:37 README.md
-rw-rw----+  1 b381103 bd1083     3458 Jul 24 17:37 anchor_1gpu.26450081.log
-rw-rw----+  1 b381103 bd1083   105255 Jul 30 01:56 atm128.26534060.log
-rw-rw----+  1 b381103 bd1083    69590 Jul 24 21:40 atm_ladders.26454476.log
-rw-rw----+  1 b381103 bd1083    42469 Aug  3 05:38 atm_ll_192.26628072.log
-rw-rw----+  1 b381103 bd1083     3628 Jul 30 02:46 bufdump.26549933.log
-rw-rw----+  1 b381103 bd1083     3214 Jul 30 10:52 bufdump2.26555987.log
-rw-rw----+  1 b381103 bd1083     4751 Aug  3 07:14 bufdump96.26644327.log
-rw-rw----+  1 b381103 bd1083     4939 Aug  3 07:19 bufdump96.26644374.log
-rw-rw----+  1 b381103 bd1083     4939 Aug  3 07:25 bufdump96.26644472.log
-rw-rw----+  1 b381103 bd1083     7161 Jul 25 01:27 comm_micro.26457469.log
-rw-rw----+  1 b381103 bd1083     6173 Jul 25 02:22 comm_micro.26457495.log
drwxrws---+ 13 b381103 bd1083     4096 Aug  3 08:42 config
-rw-rw----+  1 b381103 bd1083  4722040 Jul 28 06:00 cpu1024.26509869.log
-rw-rw----+  1 b381103 bd1083  6947641 Jul 28 11:02 cpu1024m.26512349.log
-rw-rw----+  1 b381103 bd1083 11157508 Jul 29 09:40 cpu_f32.26534068.log
-rw-rw----+  1 b381103 bd1083  1994919 Aug  2 13:12 cpu_ll2d.26628073.log
-rw-rw----+  1 b381103 bd1083     2614 Jul 28 09:59 cube_f64t.26512794.log
-rw-rw----+  1 b381103 bd1083     8089 Jul 24 18:57 cube_face6.26452602.log
-rw-rw----+  1 b381103 bd1083     8319 Jul 24 19:02 cube_face6.26452633.log
-rw-rw----+  1 b381103 bd1083    30613 Jul 24 19:23 cube_fair.26452894.log
-rw-rw----+  1 b381103 bd1083    34725 Jul 24 19:30 cube_fair.26452979.log
-rw-rw----+  1 b381103 bd1083    30791 Jul 24 20:48 cube_fair.26453782.log
-rw-rw----+  1 b381103 bd1083    99480 Jul 27 14:44 cube_ft.26498347.log
-rw-rw----+  1 b381103 bd1083   365399 Jul 27 14:33 cube_ft30.26497736.log
-rw-rw----+  1 b381103 bd1083     2794 Jul 24 18:51 cube_kt1.26452553.log
-rw-rw----+  1 b381103 bd1083   445198 Jul 27 14:11 cube_kt3.26497294.log
-rw-rw----+  1 b381103 bd1083  1662280 Jul 27 12:41 cube_matched.26495955.log
-rw-rw----+  1 b381103 bd1083    63313 Jul 27 20:04 cube_nsys.26504836.log
-rw-rw----+  1 b381103 bd1083    14107 Jul 28 03:03 cube_nsys_cl.26507936.log
-rw-rw----+  1 b381103 bd1083    35431 Jul 28 03:12 cube_nsys_cl2.26510470.log
-rw-rw----+  1 b381103 bd1083    40303 Jul 28 20:53 cube_skew.26526100.log
-rw-rw----+  1 b381103 bd1083   132274 Jul 24 17:06 cube_tiled_26446699.log
-rw-rw----+  1 b381103 bd1083    88895 Jul 24 17:42 cube_tiled_26450318.log
-rw-rw----+  1 b381103 bd1083    46408 Jul 24 18:28 cube_tiled_26450938.log
-rw-rw----+  1 b381103 bd1083    59485 Jul 24 19:06 cube_tiled_26452632.log
-rw-rw----+  1 b381103 bd1083    56485 Jul 24 20:19 cube_tiled_26453524.log
-rw-rw----+  1 b381103 bd1083    56636 Jul 24 20:33 cube_tiled_26453645.log
drwxrws---+  5 b381103 bd1083     4096 Jul 24 13:37 data
drwxrws---+ 20 b381103 bd1083     4096 Aug  3 08:42 docs
drwxrws---+  4 b381103 bd1083     4096 Jul 28 15:50 evaluations
-rw-rw----+  1 b381103 bd1083    52063 Jul 27 11:49 fig_cubef64.26495027.log
-rw-rw----+  1 b381103 bd1083     8504 Jul 27 13:52 fig_cubef64b.26495388.log
-rw-rw----+  1 b381103 bd1083    56870 Jul 27 11:12 fig_f64.26494902.log
-rw-rw----+  1 b381103 bd1083   619220 Jul 27 11:51 fig_icof32.26495083.log
-rw-rw----+  1 b381103 bd1083   556745 Jul 27 12:18 fig_icof64.26495437.log
-rw-rw----+  1 b381103 bd1083     2454 Jul 24 19:08 gate_selfspawn.26452426.log
-rw-rw----+  1 b381103 bd1083     2454 Jul 24 19:57 gate_selfspawn.26452895.log
-rw-rw----+  1 b381103 bd1083     2454 Jul 24 19:57 gate_selfspawn.26453280.log
-rw-rw----+  1 b381103 bd1083     2454 Jul 24 21:02 gate_selfspawn.26453906.log
-rw-rw----+  1 b381103 bd1083     2454 Jul 24 21:02 gate_selfspawn.26453981.log
-rw-rw----+  1 b381103 bd1083    11230 Jul 26 12:27 gather_ab.26479833.log
-rw-rw----+  1 b381103 bd1083     3192 Jul 26 12:29 gather_ab.26479884.log
-rw-rw----+  1 b381103 bd1083  1652997 Jul 24 14:48 legoesm_cpu_scaling.26445986.log
-rw-rw----+  1 b381103 bd1083  2772324 Jul 24 18:51 legoesm_cpu_scaling.26447093.log
-rw-rw----+  1 b381103 bd1083  2778934 Jul 24 18:51 legoesm_cpu_scaling.26447094.log
-rw-rw----+  1 b381103 bd1083   378758 Jul 24 15:38 legoesm_cpu_scaling.26448158.log
-rw-rw----+  1 b381103 bd1083  3175678 Jul 24 20:05 legoesm_cpu_scaling.26452578.log
-rw-rw----+  1 b381103 bd1083  3312416 Jul 25 00:53 legoesm_cpu_scaling.26452579.log
-rw-rw----+  1 b381103 bd1083     6392 Jul 24 15:11 legoesm_diag.26445834.log
-rw-rw----+  1 b381103 bd1083     7867 Jul 24 18:19 legoesm_diag.26447827.log
-rw-rw----+  1 b381103 bd1083     8059 Jul 24 21:36 legoesm_diag.26454084.log
-rw-rw----+  1 b381103 bd1083    94139 Jul 24 13:54 legoesm_gpu_multinode.26445837.log
-rw-rw----+  1 b381103 bd1083    49410 Jul 24 16:03 legoesm_gpu_multinode.26447612.log
-rw-rw----+  1 b381103 bd1083   248147 Jul 24 17:32 legoesm_gpu_multinode.26449146.log
-rw-rw----+  1 b381103 bd1083   616391 Jul 24 19:03 legoesm_gpu_multinode.26449147.log
-rw-rw----+  1 b381103 bd1083     6300 Jul 24 19:45 legoesm_gpu_multinode.26450847.log
-rw-rw----+  1 b381103 bd1083    90269 Jul 24 18:22 legoesm_gpu_multinode.26450848.log
-rw-rw----+  1 b381103 bd1083    48398 Jul 24 19:05 legoesm_gpu_multinode.26452743.log
-rw-rw----+  1 b381103 bd1083    74655 Jul 24 19:06 legoesm_gpu_multinode.26452744.log
-rw-rw----+  1 b381103 bd1083   142022 Jul 24 19:07 legoesm_gpu_multinode.26452745.log
-rw-rw----+  1 b381103 bd1083    48727 Jul 24 19:10 legoesm_gpu_multinode.26452804.log
-rw-rw----+  1 b381103 bd1083    74398 Jul 24 19:10 legoesm_gpu_multinode.26452805.log
-rw-rw----+  1 b381103 bd1083   141779 Jul 24 19:11 legoesm_gpu_multinode.26452806.log
-rw-rw----+  1 b381103 bd1083   243085 Jul 24 19:47 legoesm_gpu_multinode.26453240.log
-rw-rw----+  1 b381103 bd1083    73799 Jul 24 19:50 legoesm_gpu_multinode.26453279.log
-rw-rw----+  1 b381103 bd1083    48719 Jul 24 23:44 legoesm_gpu_multinode.26456334.log

exec
/usr/bin/bash -lc "nl -ba packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py | sed -n '626,742p'; nl -ba packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py | sed -n '860,970p'; git show e13de3afe"'^1:packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py | nl -ba | sed -n '"'610,750p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 352ms:
   626	    process.  See the atmosphere twin ``_agree_spmd_call`` for why gating only
   627	    on cache misses is NOT a valid optimisation.
   628	    """
   629	    names, sizes = _ocean_mesh_axis_terms(mesh)
   630	    assert_flags_agree(_OCEAN_CALL_ENTRY_FLAGS, (
   631	        float(mesh is not None),
   632	        float(mesh.devices.size if mesh is not None else 0),
   633	        name_digest48(names),
   634	        name_digest48(sizes),
   635	        tree_schema_digest48(state),
   636	        float(forcing is not None),
   637	        (tree_schema_digest48(forcing) if forcing is not None
   638	         else FLAG_ABSENT),
   639	    ), context=where)
   640	
   641	
   642	def make_sharded_ocean_step(model, mesh):
   643	    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
   644	    sponge=None, t_seconds=None) -> state`` running ``model.step``
   645	    lat-band-SPMD.
   646	
   647	    The forcing channels mirror ``model.step``'s keyword surface: pass
   648	    pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
   649	    ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
   650	    is a replicated scalar like ``dt``).  ``None`` forcing keeps the
   651	    dynamics-only program; each distinct None<->populated combination
   652	    compiles (and caches) its own executable.
   653	
   654	    Parameters
   655	    ----------
   656	    model
   657	        ``LatLonCGridOceanModel`` (latlon geometry; tripole north-fold is a
   658	        follow-up — raises ``NotImplementedError`` if ``model.grid.fold`` is
   659	        active).
   660	    mesh
   661	        A 1-D ``jax.sharding.Mesh`` with axis name ``"lat"`` (from
   662	        ``legoesm.parallel.mesh.create_latlon_mesh(...).mesh``).
   663	
   664	    Notes
   665	    -----
   666	    Build the ``N`` band geometries + band vertex masks host-side, stack their
   667	    ARRAY fields into a replicated pytree, and index by
   668	    ``jax.lax.axis_index("lat")`` in the ``shard_map`` body (the geometry SCALAR
   669	    fields stay static — see the module docstring).  ``dt`` is a TRACED,
   670	    replicated operand and the jitted ``shard_map`` is built once and cached
   671	    (a per-call rebuild re-traced the whole ocean step every call — see the
   672	    ``_cache`` note below).  ``check_vma=False`` (the JAX >= 0.8
   673	    replication check) because the band halo intentionally reads neighbour-rank
   674	    data (replication-unaware).
   675	
   676	    The state must be laid out with :func:`shard_state_latlon` (``v`` /
   677	    ``v_mask`` carried as the ``n_lat``-row ``v_lower``).  The body reconstructs
   678	    each band's ``nl+1`` v-faces, runs the step on the band geometry, and
   679	    converts the result back to the ``v_lower`` representation.
   680	    """
   681	    # FIRST statement: agree every rank-local input before ANY
   682	    # rank-local check can raise or return (codex round-2).
   683	    _agree_ocean_spmd_entry(model, mesh, where="make_sharded_ocean_step")
   684	    if mesh is None:                   # single-device: plain step
   685	        return lambda state, dt, **forcing_kwargs: model.step(
   686	            state, dt, **forcing_kwargs)
   687	
   688	    n_dev = mesh.devices.size
   689	    axis = mesh.axis_names[0]
   690	
   691	    # Tripole north-fold (scaling-audit item 4): SUPPORTED under the same
   692	    # v-carrier contract as the regular grid.  Every fold-touching operator
   693	    # is already uniform-program fold-capable (the data-dependent
   694	    # ``north_fold_mask``/``apply_north_fold`` selection on
   695	    # ``axis_index == N-1`` — gated by test_latlon_spmd_northfold), and
   696	    # ``build_band_grids``' slicer keeps ``is_active`` rank-consistent with
   697	    # the ``fold_j=-1`` sentinel off the north band.  The one structural
   698	    # assumption is the v-carrier's: the TOP v-face row (the seam/cap row,
   699	    # ``v[n_lat]``) must be WALL-MASKED so the in-body reconstruction's
   700	    # zero row is exact — true for the cap-row convention of
   701	    # ``create_synthetic_tripole`` and the eORCA masks (``v_mask[-1] == 0``;
   702	    # the serial step keeps ``v[-1] == 0`` identically).  That contract is
   703	    # asserted on the CONCRETE state in :func:`shard_state_latlon` — a live
   704	    # (unmasked) seam v-row refuses loudly there instead of silently
   705	    # reconstructing zeros here.
   706	
   707	    # --- host-side band geometries + vertex masks (replicated, indexed in-body) ---
   708	    band_grids = build_band_grids(model.grid, n_dev)
   709	    band_vmasks = _build_band_vertex_masks(model, n_dev)
   710	    template = band_grids[0]           # static-scalar source (uniform bands)
   711	    array_field_names = _geom_array_field_names(template)
   712	
   713	    # Stack each geometry ARRAY field over the band axis (rank 0..N-1) and
   714	    # SHARD along that axis (#1370 stage (iii), codex round-18): each device
   715	    # holds ONLY its own band's slab instead of the whole global stack —
   716	    # this was one of the residual ~1.4-1.7 global-field-equivalents of
   717	    # per-device residency left after the host-side-build fix (probe
   718	    # 26524423). The leading axis has length n_dev, so P("lat") divides it
   719	    # exactly; the body indexes its local slab at [0]. Values are unchanged
   720	    # — same stack, different placement; the process-0 broadcast +
   721	    # divergence guard below runs on HOST values and is placement-blind.
   722	    rep = NamedSharding(mesh, P("lat"))
   723	
   724	    def _replicated_put(arr, name):
   725	        # (Name kept for history; this is a SHARDED P("lat") stack put.)
   726	        # checked_shard_put replaces the broadcast_checked+device_put pair:
   727	        # the broadcast's psum program is [n_processes, stack] (nd x 849 MB
   728	        # at LL2304 — the @96/@128 wall), and a numpy device_put onto an
   729	        # all-process sharding pays jax's whole-array assert_equal on top.
   730	        # The per-band gate keeps the divergence contract (n_bands is
   731	        # schema-gated just below, so payload widths agree). ONE shared
   732	        # implementation: legoesm.parallel.geometry_consistency.
   733	        return checked_shard_put(
   734	            arr, name, rep, context="make_sharded_ocean_step",
   735	            n_bands=n_dev)
   736	
   737	    # Schema gate FIRST (one fixed-shape collective every process reaches):
   738	    # a process-dependent field list or a mixed jax_enable_x64 setting would
   739	    # otherwise desynchronize the per-field gathers below instead of failing
   740	    # with a clear message.
   741	    # Build the raw stacks FIRST so the schema gate can also cover each
   742	    # field's dtype class and ndim -- those decide the per-field payload
   860	                    f"sharded ocean step: forcing leaf "
   861	                    f"{jax.tree_util.keystr(path)} is 1-D "
   862	                    f"(shape {tuple(leaf.shape)}); forcing must be "
   863	                    f"cell-centered (n_lat, n_lon[, nlev]) arrays or "
   864	                    f"scalars.")
   865	            if nd >= 2 and int(leaf.shape[0]) != n_lat_global:
   866	                raise ValueError(
   867	                    f"sharded ocean step: forcing leaf {jax.tree_util.keystr(path)} "
   868	                    f"has leading dim {leaf.shape[0]} != n_lat "
   869	                    f"({n_lat_global}); forcing must be cell-centered "
   870	                    f"(n_lat, n_lon[, nlev]) to shard on the lat axis.")
   871	
   872	    def sharded_step(state, dt, freshwater=None, surface_forcing=None,
   873	                     sponge=None, t_seconds=None, aux=None):
   874	        # ``aux``: the sharded geometry+vmask stacks. When this wrapper runs
   875	        # INSIDE an outer trace (a bench/driver jit/scan — jit-of-jit
   876	        # inlines the inner call), concrete closure arrays become
   877	        # OUTER-trace constants whose value the MLIR handler cannot fetch
   878	        # for non-addressable arrays (broken since #1370-iii sharded the
   879	        # stacks). Outer-jit callers MUST thread ``step.aux`` through their
   880	        # jit boundary as an ARGUMENT and pass it back here.
   881	        # ONE forcing operand: None fields drop out of the pytree structure,
   882	        # so specs derived by tree.map skip them automatically and the
   883	        # structure key below distinguishes every None<->array combination.
   884	        _agree_ocean_spmd_call(
   885	            mesh, state, (freshwater, surface_forcing, sponge, t_seconds),
   886	            where="make_sharded_ocean_step.step")
   887	        forcing = (freshwater, surface_forcing, sponge, t_seconds)
   888	        _validate_forcing_layout((freshwater, surface_forcing, sponge))
   889	        # Cache key = the state's AND forcing's pytree STRUCTURE, plus the
   890	        # forcing leaves' RANKS: in_specs/out_specs are derived from them,
   891	        # so a later call with a different structure (an optional field
   892	        # flipping None <-> Field, a sea-ice lane populating sf.salt_flux,
   893	        # the restoring lane passing no forcing at all) OR a same-field
   894	        # rank change (SpongeForcing.gamma is legitimately 2-D horizontal
   895	        # OR 3-D full-rank — same structure, different _lat_spec; codex r1
   896	        # #1) must rebuild the shard_map rather than reuse stale specs.
   897	        forcing_ndims = tuple(
   898	            int(getattr(leaf, "ndim", np.ndim(leaf)))
   899	            for leaf in jax.tree.leaves(forcing))
   900	        # The SPMD fused-halo switch is read at TRACE time inside the pad
   901	        # dispatch — flipping LEGOESM_LATLON_SPMD_FUSED_HALO on a reused
   902	        # step object must rebuild the shard_map, not reuse a stale jaxpr
   903	        # (codex, audit item 7).
   904	        import os as _os
   905	
   906	        _fused_halo = _os.environ.get(
   907	            "LEGOESM_LATLON_SPMD_FUSED_HALO", "0") != "0"
   908	        key = (jax.tree.structure(state), jax.tree.structure(forcing),
   909	               forcing_ndims, _fused_halo)
   910	        fn = _cache.get(key)
   911	        if fn is None:
   912	            in_spec = jax.tree.map(_lat_spec, state)
   913	            # Forcing leaves are cell-centered -> plain lat-band specs;
   914	            # scalars (t_seconds) replicate, exactly like dt.
   915	            forcing_spec = jax.tree.map(_lat_spec, forcing)
   916	            # Stage (iii): the stacks are banded on their leading axis, so
   917	            # the shard_map spec matches their P("lat") placement (the body
   918	            # indexes its local (1, ...) slab at [0]).
   919	            geom_spec = jax.tree.map(lambda _x: P("lat"), geom_stacks)
   920	            vmask_spec = P("lat")
   921	            # JAX >= 0.8 top-level shard_map takes ``check_vma`` (the
   922	            # replication check); the band halo reads neighbour-rank data so
   923	            # disable it (same as the validated PCG / halo-parity shard_maps).
   924	            fn = jax.jit(shard_map(
   925	                _body,
   926	                mesh=mesh,
   927	                in_specs=(in_spec, forcing_spec, geom_spec, vmask_spec, P()),
   928	                out_specs=in_spec,
   929	                check_vma=False,
   930	            ))
   931	            _cache[key] = fn
   932	        # Arm the SPMD halo backend ONLY around the call, then RESTORE the
   933	        # previous backend (codex finding): leaving it globally armed makes a
   934	        # later serial/full-domain ocean call take SPMD-only branches
   935	        # (axis_index / ppermute / psum in pad_with_pole_bc_lat, conservation,
   936	        # eta_floor) OUTSIDE a shard_map -> crash. The FIRST call traces with
   937	        # the backend armed (baking the SPMD halo/reduction ops into the
   938	        # compiled program); later calls reuse the cached compile, and the
   939	        # arm/restore keeps any interleaved serial path untouched.
   940	        # Save+restore the FULL backend state (backend + MPI topology + SPMD
   941	        # mesh) so a prior "mpi"/"spmd" backend is restored intact: activate_*
   942	        # clears the MPI topology, and set_halo_backend("mpi") REQUIRES a
   943	        # topology (codex).
   944	        from legoesm.grids.halo import (
   945	            get_halo_backend, get_mpi_topology, get_spmd_mesh,
   946	            set_halo_backend, set_spmd_mesh,
   947	        )
   948	        _prev_backend = get_halo_backend()
   949	        _prev_topo = get_mpi_topology()
   950	        _prev_mesh = get_spmd_mesh()
   951	        activate_latlon_spmd_halo(mesh)
   952	        try:
   953	            _geom, _vmask = aux if aux is not None else (geom_stacks,
   954	                                                        vmask_stack)
   955	            return fn(state, forcing, _geom, _vmask, jnp.asarray(dt))
   956	        finally:
   957	            set_spmd_mesh(_prev_mesh)
   958	            set_halo_backend(_prev_backend, _prev_topo)
   959	
   960	    # Expose the stacks so outer-jit callers can pass them as arguments
   961	    # (see the ``aux`` note in the signature).
   962	    sharded_step.aux = (geom_stacks, vmask_stack)
   963	    return sharded_step
   964	
   965	
   966	def make_sharded_ocean_step_global(model, mesh):
   967	    """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
   968	    that takes a GLOBAL (single-device-layout) state + forcing and returns a
   969	    GLOBAL state — the minimal-diff driver entry point.
   970	
   610	
   611	
   612	def make_sharded_ocean_step(model, mesh):
   613	    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
   614	    sponge=None, t_seconds=None) -> state`` running ``model.step``
   615	    lat-band-SPMD.
   616	
   617	    The forcing channels mirror ``model.step``'s keyword surface: pass
   618	    pytrees laid out with :func:`shard_forcing_latlon` (cell-centered
   619	    ``(n_lat, n_lon[, nlev])`` leaves shard on the lat axis; ``t_seconds``
   620	    is a replicated scalar like ``dt``).  ``None`` forcing keeps the
   621	    dynamics-only program; each distinct None<->populated combination
   622	    compiles (and caches) its own executable.
   623	
   624	    Parameters
   625	    ----------
   626	    model
   627	        ``LatLonCGridOceanModel`` (latlon geometry; tripole north-fold is a
   628	        follow-up — raises ``NotImplementedError`` if ``model.grid.fold`` is
   629	        active).
   630	    mesh
   631	        A 1-D ``jax.sharding.Mesh`` with axis name ``"lat"`` (from
   632	        ``legoesm.parallel.mesh.create_latlon_mesh(...).mesh``).
   633	
   634	    Notes
   635	    -----
   636	    Build the ``N`` band geometries + band vertex masks host-side, stack their
   637	    ARRAY fields over a leading band axis SHARDED ``P("lat")`` (each device
   638	    holds only its own slab; the ``shard_map`` body reads it at ``[0]``; the
   639	    geometry SCALAR fields stay static — see the module docstring).  ``dt`` is a TRACED,
   640	    replicated operand and the jitted ``shard_map`` is built once and cached
   641	    (a per-call rebuild re-traced the whole ocean step every call — see the
   642	    ``_cache`` note below).  ``check_vma=False`` (the JAX >= 0.8
   643	    replication check) because the band halo intentionally reads neighbour-rank
   644	    data (replication-unaware).
   645	
   646	    The state must be laid out with :func:`shard_state_latlon` (``v`` /
   647	    ``v_mask`` carried as the ``n_lat``-row ``v_lower``).  The body reconstructs
   648	    each band's ``nl+1`` v-faces, runs the step on the band geometry, and
   649	    converts the result back to the ``v_lower`` representation.
   650	    """
   651	    if mesh is None:                   # single-device: plain step
   652	        return lambda state, dt, **forcing_kwargs: model.step(
   653	            state, dt, **forcing_kwargs)
   654	
   655	    n_dev = mesh.devices.size
   656	    axis = mesh.axis_names[0]
   657	
   658	    # Tripole north-fold (scaling-audit item 4): SUPPORTED under the same
   659	    # v-carrier contract as the regular grid.  Every fold-touching operator
   660	    # is already uniform-program fold-capable (the data-dependent
   661	    # ``north_fold_mask``/``apply_north_fold`` selection on
   662	    # ``axis_index == N-1`` — gated by test_latlon_spmd_northfold), and
   663	    # ``build_band_grids``' slicer keeps ``is_active`` rank-consistent with
   664	    # the ``fold_j=-1`` sentinel off the north band.  The one structural
   665	    # assumption is the v-carrier's: the TOP v-face row (the seam/cap row,
   666	    # ``v[n_lat]``) must be WALL-MASKED so the in-body reconstruction's
   667	    # zero row is exact — true for the cap-row convention of
   668	    # ``create_synthetic_tripole`` and the eORCA masks (``v_mask[-1] == 0``;
   669	    # the serial step keeps ``v[-1] == 0`` identically).  That contract is
   670	    # asserted on the CONCRETE state in :func:`shard_state_latlon` — a live
   671	    # (unmasked) seam v-row refuses loudly there instead of silently
   672	    # reconstructing zeros here.
   673	
   674	    # --- host-side band geometries + vertex masks (band-stacked, P("lat")-sharded) ---
   675	    band_grids = build_band_grids(model.grid, n_dev)
   676	    band_vmasks = _build_band_vertex_masks(model, n_dev)
   677	    template = band_grids[0]           # static-scalar source (uniform bands)
   678	    array_field_names = _geom_array_field_names(template)
   679	
   680	    # Stack each geometry ARRAY field over the band axis (rank 0..N-1) and
   681	    # SHARD along that axis (#1370 stage (iii), codex round-18): each device
   682	    # holds ONLY its own band's slab instead of the whole global stack —
   683	    # this was one of the residual ~1.4-1.7 global-field-equivalents of
   684	    # per-device residency left after the host-side-build fix (probe
   685	    # 26524423). The leading axis has length n_dev, so P("lat") divides it
   686	    # exactly; the body indexes its local slab at [0]. Values are unchanged
   687	    # — same stack, different placement; the cross-process divergence
   688	    # guard below runs on HOST values and is placement-blind.
   689	    rep = NamedSharding(mesh, P("lat"))
   690	
   691	    def _replicated_put(arr, name):
   692	        # (Name kept for history; since 2026-08-03 this is a SHARDED stack
   693	        # put.) The band-geometry arrays are (re)computed per process and
   694	        # can differ in their last ULPs (per-process XLA autotuning on
   695	        # device-derived grid fields) — the fully-replicated-put era
   696	        # broadcast process 0's bytes to sidestep the P() bit-identity
   697	        # assert (job 26450848). With the #1370-iii P("lat") sharding each
   698	        # process's devices consume ONLY its own band rows, so the
   699	        # broadcast became both unnecessary and, at nd>=96, fatal (its
   700	        # psum program is nd x the stack — see the note at the put below).
   701	        # GUARD (codex round-3): process 0 must not silently mask REAL
   702	        # cross-process divergence. Compare an allgathered fingerprint:
   703	        # structural entries exactly; value entries EXACTLY for integer/bool
   704	        # arrays (masks are comparison results — bit-reproducible, and an
   705	        # exact compare is the only way to catch a two-cell flip that cancels
   706	        # in the sum, codex round-4) and to rtol 1e-5 for float arrays (only
   707	        # ULP autotune drift is expected there; quantize-then-assert-equal
   708	        # false-positived on a rounding boundary, job 26453240).
   709	        # NO DEADLOCK RISK: every process fingerprints the same fields in the
   710	        # same order and derives the verdict from the SAME gathered array, so
   711	        # the refusal is symmetric — all raise or none.
   712	        # Residual (documented): a float-geometry divergence preserving sum,
   713	        # sum-of-squares AND absmax to 1e-5 is not detected; band grids are
   714	        # analytic in lat/lon, so any real inconsistency moves those moments.
   715	        host = np.asarray(arr)
   716	        if jax.process_count() > 1:
   717	            from jax.experimental import multihost_utils
   718	
   719	            # PER-BAND fingerprints (module-level, unit-tested): exact
   720	            # dtypes hash positionally per band; floats compare per-band
   721	            # moments to rtol 1e-5 — bounds each band's drift instead of
   722	            # letting it hide in a whole-array sum, since each process's
   723	            # own bytes are now the live inputs for the bands it owns
   724	            # (codex r14). A mask that genuinely differs across processes
   725	            # means different wet domains = different physics: refusing is
   726	            # correct, not a false alarm.
   727	            struct, vals, is_exact = geom_band_fingerprint(
   728	                host, host.shape[0])
   729	            g_struct = multihost_utils.process_allgather(struct)
   730	            g_vals = multihost_utils.process_allgather(vals)
   731	            if not band_fingerprints_agree(g_struct, g_vals, is_exact):
   732	                raise RuntimeError(
   733	                    f"make_sharded_ocean_step: band-geometry field {name!r} "
   734	                    f"DIVERGES across processes (exact_dtype={is_exact}, "
   735	                    f"gathered={g_vals.tolist()}) — a real config/grid "
   736	                    f"inconsistency, not autotune noise; refusing to "
   737	                    f"shard it.")
   738	            # NO broadcast_one_to_all here (removed 2026-08-03): its psum
   739	            # program is [n_processes, stack] in / P() fully-replicated out,
   740	            # so its logical arg bytes are nd x the global stack — 82.4 GB
   741	            # at nd=96 and 109.6 GB at nd=128 for one 3-D field, the
   742	            # near-linear-in-nd wall that killed oc @96/@128 (jobs
   743	            # 26642771/26636762) while @64 sat just under XLA's 63.8 GB
   744	            # limit.  The target sharding is P("lat"): each process's
   745	            # devices consume ONLY its own band rows, so cross-process
   746	            # byte-identity of non-owned rows is irrelevant, and REAL
   747	            # divergence is already refused by the fingerprint gate above.
   748	            # make_array_from_callback hands each process exactly its
   749	            # addressable slabs — the same #1100 pattern as the state
   750	            # build — with no global-sized collective program at all.

exec
/usr/bin/bash -lc 'git diff --stat e13de3afe''^2 e13de3afe -- packages/core/legoesm/parallel/geometry_consistency.py packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py; git diff --unified=25 e13de3afe''^2 e13de3afe -- packages/core/legoesm/parallel/geometry_consistency.py packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py tests/ocean/unit/test_sharded_geom_fingerprint.py; git log --oneline --decorate -8' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 794ms:
 .../core/legoesm/parallel/geometry_consistency.py  | 135 +++++++++++++++++++++
 .../legoesm/ocean/dynamics/sharded_ocean_step.py   |  73 ++++++-----
 tests/ocean/unit/test_sharded_geom_fingerprint.py  | 104 ++++++++++++++++
 3 files changed, 285 insertions(+), 27 deletions(-)
diff --git a/packages/core/legoesm/parallel/geometry_consistency.py b/packages/core/legoesm/parallel/geometry_consistency.py
index 7a54aed6b..0a131ca18 100644
--- a/packages/core/legoesm/parallel/geometry_consistency.py
+++ b/packages/core/legoesm/parallel/geometry_consistency.py
@@ -574,25 +574,160 @@ def broadcast_checked(arr, name: str, *, context: str) -> np.ndarray:
         finite = flat[np.isfinite(flat)]
         f64 = finite.astype(np.float64)
         # Non-finite COUNT is structural: a NaN appearing on one process only
         # must not be averaged away by the moment compare below.
         struct[5] = float(flat.size - finite.size)
         vals[0] = float(f64.sum()) if f64.size else 0.0
         vals[1] = float((f64 * f64).sum()) if f64.size else 0.0
         vals[2] = float(np.abs(f64).max()) if f64.size else 0.0
 
     g_struct = multihost_utils.process_allgather(struct)
     g_vals = multihost_utils.process_allgather(vals)
     struct_ok = bool(np.all(g_struct == g_struct[0]))
     if is_exact:
         vals_ok = bool(np.all(g_vals == g_vals[0]))
     else:
         vals_ok = bool(np.allclose(g_vals, g_vals[0],
                                    rtol=_FLOAT_RTOL, atol=0.0))
     if not (struct_ok and vals_ok):
         raise RuntimeError(
             f"{context}: geometry field {name!r} DIVERGES across processes "
             f"(struct_ok={struct_ok}, vals_ok={vals_ok}, "
             f"exact_dtype={is_exact}, gathered={g_vals.tolist()}) — a real "
             f"config/grid inconsistency, not autotune noise; refusing to "
             f"broadcast process 0 over it.")
     return np.asarray(multihost_utils.broadcast_one_to_all(host))
+
+
+# --- assert-free sharded puts + per-band gates (2026-08-03, ocean walls) ----
+# Three stacked multicontroller walls were found on the ocean lane (codex
+# r14-r19; PR #1457): (1) broadcast_one_to_all of a band stack lowers to an
+# [n_processes, stack] psum program (nd x 849 MB at LL2304 L20 — 81.5 GB at
+# 96 procs); (2) jax.device_put of a NUMPY array onto an all-process
+# sharding internally runs multihost_utils.assert_equal on the FULL array
+# ([n_proc, field] landing on ONE device: fits under an 80 GB A100 up to
+# ~64 procs, dies at 96 — jax _src/dispatch.py::_device_put_sharding_impl);
+# (3) a concrete sharded-global array captured by an OUTER trace (jit-of-
+# jit) becomes an MLIR constant whose value cannot be fetched for
+# non-addressable arrays. The helpers below remove (1) and (2) — (3) is the
+# callers' aux-threading contract, see make_sharded_ocean_step.
+
+def band_fingerprint(host, n_bands):
+    """Per-band fingerprint of a band-STACKED field (leading axis n_bands).
+
+    PREREQUISITE: ``n_bands`` (and each field's dtype class / shape) must
+    already be schema-gated across processes (:func:`assert_schema_agrees`)
+    — the payload widths depend on it, and mismatched widths would hang the
+    allgather rather than raise.
+
+    Exact dtypes (int/bool/uint): one positional 48-bit byte digest per
+    band. Floats: per-band ``[sum, sum_of_squares, absmax]`` of finite
+    entries plus per-band non-finite counts folded into ``struct``.
+    Per-band (not whole-array) because each process's OWN bytes become the
+    live inputs for the bands it owns under the assert-free put: a
+    band-local drift must not hide in a whole-array sum (codex r14).
+    DOCUMENTED RESIDUALS: a within-band float change preserving all three
+    moments to rtol, and non-finite entries changing position/kind at a
+    fixed per-band count, pass the float gate.
+    """
+    host = np.asarray(host)
+    if host.shape[0] != n_bands:
+        raise ValueError(
+            f"band_fingerprint: leading axis {host.shape[0]} != n_bands "
+            f"{n_bands}")
+    is_exact = host.dtype.kind in "biu"
+    struct = [float(host.ndim), *map(float, host.shape),
+              float(np.dtype(host.dtype).num)]
+    if is_exact:
+        vals = np.array([content_hash48(host[b]) for b in range(n_bands)],
+                        dtype=np.float64)
+    else:
+        per_band = []
+        for b in range(n_bands):
+            flat = host[b].ravel()
+            finite = flat[np.isfinite(flat)]
+            f64 = finite.astype(np.float64)
+            struct.append(float(flat.size - finite.size))
+            per_band.extend([
+                float(f64.sum()) if f64.size else 0.0,
+                float((f64 * f64).sum()) if f64.size else 0.0,
+                float(np.abs(f64).max()) if f64.size else 0.0,
+            ])
+        vals = np.array(per_band, dtype=np.float64)
+    return np.array(struct, dtype=np.float64), vals, is_exact
+
+
+def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=None):
+    """True iff every process's :func:`band_fingerprint` matches process 0's."""
+    if rtol is None:
+        rtol = _FLOAT_RTOL
+    struct_ok = bool(np.all(g_struct == g_struct[0]))
+    if is_exact:
+        vals_ok = bool(np.all(g_vals == g_vals[0]))
+    else:
+        vals_ok = bool(np.allclose(g_vals, g_vals[0], rtol=rtol, atol=0.0))
+    return struct_ok and vals_ok
+
+
+def checked_shard_put(arr, name, sharding, *, context, n_bands):
+    """Gate a band-stacked field per band, then put WITHOUT broadcast or
+    jax's whole-array device_put assert (walls 1+2 above).
+
+    Single-process: plain ``jax.device_put`` — byte-unchanged, no host
+    round trip. Multi-process: per-band fingerprint gate (symmetric raise
+    on real divergence), then ``jax.make_array_from_callback`` hands each
+    process exactly its addressable slabs. Cross-process byte-identity of
+    NON-owned bands is not required — owned bands are the only bytes that
+    reach any device, and their drift is bounded by the gate.
+    """
+    if jax.process_count() <= 1:
+        return jax.device_put(arr, sharding)
+    from jax.experimental import multihost_utils
+
+    host = np.asarray(arr)
+    struct, vals, is_exact = band_fingerprint(host, n_bands)
+    g_struct = multihost_utils.process_allgather(struct)
+    g_vals = multihost_utils.process_allgather(vals)
+    if not band_fingerprints_agree(g_struct, g_vals, is_exact):
+        raise RuntimeError(
+            f"{context}: band-stacked field {name!r} DIVERGES across "
+            f"processes (exact_dtype={is_exact}, "
+            f"gathered={g_vals.tolist()}) — a real config/grid "
+            f"inconsistency, not autotune noise; refusing to shard it.")
+    return jax.make_array_from_callback(
+        host.shape, sharding, lambda idx: host[idx])
+
+
+def assert_pytree_bytes_equal(tree, what):
+    """Cheap multi-process replacement for the per-leaf assert_equal that
+    :func:`checked_shard_put`-style puts bypass on NON-band inputs (state /
+    forcing pytrees): one 48-bit digest per array leaf, one tiny allgather,
+    symmetric raise on mismatch. No-op single-process.
+    """
+    if jax.process_count() <= 1:
+        return
+    from jax.experimental import multihost_utils
+
+    leaves = [x for x in jax.tree_util.tree_leaves(tree)
+              if hasattr(x, "ndim")]
+    vals = np.array([content_hash48(np.asarray(x)) for x in leaves],
+                    dtype=np.float64)
+    g = multihost_utils.process_allgather(vals)
+    if not bool(np.all(g == g[0])):
+        bad = [i for i in range(len(leaves))
+               if not bool(np.all(g[:, i] == g[0, i]))]
+        raise RuntimeError(
+            f"{what}: array leaves {bad} differ across processes (48-bit "
+            f"byte digests disagree) — the per-process inputs are NOT "
+            f"identical, which jax's device_put assert would have refused. "
+            f"Fix the per-process build before sharding.")
+
+
+def addressable_shard_put(arr, sharding):
+    """Ungated assert-free put (walls 1+2) for inputs whose cross-process
+    consistency the CALLER has already gated (state/forcing pytrees via
+    :func:`assert_pytree_bytes_equal`). Single-process: plain device_put."""
+    if jax.process_count() <= 1:
+        return jax.device_put(arr, sharding)
+    host = np.asarray(arr)
+    return jax.make_array_from_callback(
+        host.shape, sharding, lambda idx: host[idx])
diff --git a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
index 622386a61..f9e7ecb7e 100644
--- a/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
+++ b/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
@@ -24,51 +24,52 @@ Architecture (the two non-trivial pieces — see ``omip-multinode-spmd-scope``):
   band geometry is rebuilt as ``template._replace(**indexed_arrays)`` so the
   scalars come from a band template (all bands share ``n_lat_local = n_lat/N``).
 
 * STATE = cell fields shard ``P("lat")``; the STAGGERED ``v`` / ``v_mask``
   (shape ``(n_lat+1, n_lon, ...)``) use the **band-local v-faces** layout
   (Structure B). The sharded state carries ``v_lower = v[0:n_lat]`` (``n_lat``
   rows, ``P("lat")`` — the top pole row ``v[n_lat]`` is a 0 wall on the regular
   grid and is dropped). In-body, band ``r`` reconstructs its ``nl+1`` v-faces by
   ``ppermute``-ing band ``r+1``'s first ``v_lower`` row (= global ``v[e]``) up as
   its north boundary row; the north band's non-target receives the pole-wall 0.
   After ``model.step`` the result's ``nl+1`` v is converted back to the ``nl``-row
   ``v_lower`` representation (drop the boundary row, owned by band ``r+1``).
 
 Correctness is gated by ``tests/parallel/test_latlon_ocean_spmd_step.py`` (N-step
 tol-match vs the single-device step). Pure-dynamics (no host-loop coupling): the
 OMIP host post-step updates (ice / SSS-restore / geothermal) are NOT inside this
 wrapper — they need separate gather/scatter or jax-porting (see
 [[omip-multinode-spmd-scope]]).
 """
 from __future__ import annotations
 
 import jax
 import jax.numpy as jnp
 import numpy as np
 from legoesm.parallel.geometry_consistency import (
-    FLAG_ABSENT, assert_flags_agree, assert_schema_agrees, broadcast_checked,
+    FLAG_ABSENT, addressable_shard_put, assert_flags_agree,
+    assert_pytree_bytes_equal, assert_schema_agrees, checked_shard_put,
     coerce_bool, coerce_count, config_digest48, name_digest48,
     tree_schema_digest48)
 from jax.sharding import NamedSharding, PartitionSpec as P
 
 try:                                   # JAX >= 0.8 exposes shard_map at top level
     from jax import shard_map
 except ImportError:                    # pragma: no cover - JAX < 0.8 fallback
     from jax.experimental.shard_map import shard_map
 
 from legoesm.parallel.latlon_spmd import (
     activate_latlon_spmd_halo,
     latlon_band_perms,
     reconstruct_vface_lower_multi,
 )
 
 # Per-device band grids: the SPMD wrapper REUSES the tested band slicers from
 # ``legoesm.parallel.latlon_mpi`` rather than a bespoke slice — they already
 # handle the staggered v-row (n_lat+1), the bipolar-fold localization (the
 # northernmost band keeps the active ``fold``; interior bands get
 # ``fold_j=cap_j=-1`` so the pad_ns_* operators fall through to the band halo
 # instead of folding their own last row), AND keep ``total_area`` GLOBAL (the
 # area-weighted-mean denominator must not become band-local).  Build a band
 # layout with ``make_latlon_band_layout(rank, n_dev, n_lat, n_lon, fold)`` then
 # call ``slice_cgrid_geometry_to_band(geom, layout)`` (curvilinear tripole /
 # eORCA025) or ``slice_latlon_grid_to_band(grid, layout)`` (regular lat-lon ½°).
@@ -289,179 +290,185 @@ def _agree_ocean_mesh_entry(mesh, tree=None, *, where: str) -> None:
 
 def shard_state_latlon(state, mesh):
     """Lay out a ``LatLonCGridOceanState`` for the lat-band SPMD step.
 
     Cell / u-grid array fields (leading dim ``n_lat``) shard ``P("lat")``.  The
     staggered ``v`` / ``v_mask`` (leading dim ``n_lat+1``) are carried as
     ``v_lower = field[0:n_lat]`` (``n_lat`` rows, ``P("lat")``) — the top pole row
     ``field[n_lat]`` is a 0 wall on the regular grid and is reconstructed in-body
     by the band halo.  ``None`` fields pass through.  The inverse is
     :func:`gather_state_latlon`.
 
     This is the layout the in-``shard_map`` body of :func:`make_sharded_ocean_step`
     expects; the test uses it instead of a uniform ``tree.map(P("lat"))`` (which
     fails on ``v`` because ``n_lat+1`` is not divisible by ``N``).
     """
     # FIRST statement: a DIRECT ``device_put`` of full global arrays onto a
     # cross-process ``NamedSharding`` is serviced by an ALL-GATHER, and one
     # runs per leaf -- so both the mesh and the leaf SCHEDULE are rank-local
     # inputs to a collective (#1362 round 4, blockers 5-6).
     _agree_ocean_mesh_entry(mesh, state, where="shard_state_latlon")
     # v-carrier contract (see make_sharded_ocean_step's fold note): the TOP
     # v-face row (regular pole wall OR tripole seam/cap row) must be
     # wall-masked — the carrier drops it and reconstructs it as zero, which
     # would silently delete a LIVE seam row.  Host-side check on the
     # concrete state (this fn runs outside jit).
+    assert_pytree_bytes_equal(state, "shard_state_latlon")
     vm = getattr(state, "v_mask", None)
     if vm is not None:
         import numpy as _np
 
         if _np.asarray(vm.data)[-1].any():
             raise ValueError(
                 "shard_state_latlon: the state's TOP v-face row is LIVE "
                 "(v_mask[-1] has ocean faces) — the lat-band v-carrier "
                 "drops that row and reconstructs it as the pole/cap wall "
                 "zero, which would silently delete seam velocities. "
                 "Mask the cap row (the tripole cap convention) or extend "
                 "the carrier before sharding this state.")
 
     def _shard_cell(field):
         if field is None:
             return None
         sh = NamedSharding(mesh, _lat_spec(field.data))
-        return field.replace(data=jax.device_put(field.data, sh))
+        return field.replace(data=addressable_shard_put(field.data, sh))
 
     def _shard_v(field):
         if field is None:
             return None
         # Drop the TOP v-row (the north pole wall, v==0 on the regular grid) so
         # the leading dim becomes n_lat (divisible by N).  ROUND-TRIP INVARIANT
         # (codex finding): shard_state_latlon -> gather_state_latlon re-appends a
         # ZERO top row, so the round-trip is a bitwise identity ONLY when the
         # input's top v-row is already zero (a valid masked regular-grid state;
         # the pole-wall BC enforces v[n_lat]=0 every step).  An arbitrary nonzero
         # top row (e.g. a raw IC perturbation) is dropped -> reconstructed as 0:
         # CORRECT for the dynamics (the pole wall zeros it at step 1) but not a
         # bit round-trip of that one row.  NOT for a tripole north fold (raises
         # in make_sharded_ocean_step) where the top row is a live fold partner.
         nlat1 = field.data.shape[0]
         v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
         sh = NamedSharding(mesh, _lat_spec(v_lower))
-        return field.replace(data=jax.device_put(v_lower, sh))
+        return field.replace(data=addressable_shard_put(v_lower, sh))
 
     updates = {}
     for name in _V_STAGGERED_STATE_FIELDS:
         updates[name] = _shard_v(getattr(state, name))
     # Every other (array) leaf: shard P("lat").  NamedTuple fields not in the
     # v-staggered set and not None get the cell sharding; None stays None.
     for name in state._fields:
         if name in _V_STAGGERED_STATE_FIELDS:
             continue
         val = getattr(state, name)
         if val is None:
             updates[name] = None
         elif hasattr(val, "data"):                 # a Field
             updates[name] = _shard_cell(val)
         else:
             # Non-Field, non-None leaf (e.g. a raw array carry like psi).
             # Shard 2-D+ on lat, replicate lower-rank — matches shard_pytree.
             arr = jnp.asarray(val)
             spec = _lat_spec(arr) if arr.ndim >= 1 else P()
-            updates[name] = jax.device_put(arr, NamedSharding(mesh, spec))
+            # ndim>=1 lat-shards via _lat_spec (1-D included); only true
+            # scalars replicate.
+            updates[name] = addressable_shard_put(arr, NamedSharding(mesh, spec))
     return state._replace(**updates)
 
 
 def shard_forcing_latlon(forcing, mesh):
     """Lay out a forcing pytree (``FreshwaterForcing`` / ``OceanSurfaceForcing``
     / ``SpongeForcing`` — or any nesting of them) for the lat-band SPMD step.
 
     Every OMIP forcing field is CELL-CENTERED ``(n_lat, n_lon[, nlev])`` (the
     cell->face wind-stress interpolation happens INSIDE the step through the
     SPMD-aware halo pads), so array leaves shard ``P("lat", None, ...)`` with
     no ``v_lower`` handling; scalars replicate; ``None`` fields pass through
     untouched (they vanish from the pytree structure, matching the specs the
     step derives).  ``forcing=None`` returns ``None``.
     """
     # FIRST statement: the `forcing is None or mesh is None` return below
     # SKIPS every per-leaf put, so a rank with no forcing would leave a peer
     # blocked in one (#1362 round 4, blocker 6).
     _agree_ocean_mesh_entry(mesh, forcing, where="shard_forcing_latlon")
     if forcing is None or mesh is None:
         return forcing
+    assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")
 
     def _put(leaf):
         if leaf is None:
             return None
         arr = jnp.asarray(leaf)
-        return jax.device_put(arr, NamedSharding(mesh, _lat_spec(arr)))
+        return addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
 
     return jax.tree.map(_put, forcing)
 
 
 def shard_forcing_stack_latlon(stack, mesh):
     """Lay out a STACKED per-block forcing pytree for the lat-band SPMD
     block-scan (the ``run_omip`` JRA55 lanes; see
     ``_build_jra55_block_fn`` / ``_build_jra55_block_fn_interp``).
 
     Unlike :func:`shard_forcing_latlon` (per-step, lat at axis 0), the
     block builders stack ``N`` steps / raw records along a LEADING axis,
     so the lat axis sits at position 1.  A ``(n_rec, n_lat, n_lon[, ...])``
     leaf therefore shards ``P(None, "lat", ...)`` (records replicated, the
     time index stays shard-local so the in-scan interpolation needs no
     cross-band comm); a bare ``(n_lat, n_lon)`` leaf shards ``P("lat", None)``;
     1-D metadata / scalars replicate; ``None`` and non-array leaves pass
     through.  ``mesh=None`` returns ``stack`` unchanged (serial lane).
 
     CONTRACT: rank is the ONLY signal used, so a rank-2 leaf is assumed to
     be a ``(n_lat, n_lon)`` field and is lat-sharded on axis 0.  Any future
     metadata that is genuinely rank-2 but NOT lat-major (e.g. a
     ``(n_rec, n_meta)`` table) would be silently mis-sharded — keep such
     metadata 1-D (or replicate it explicitly) before it reaches this helper.
 
     Keeping this next to :func:`shard_state_latlon` means the driver and
     the parity tests share ONE layout definition — the block-scan forcing
     stack must be laid out consistently with the state the sharded step
     carries, and a second copy would drift.
     """
     # FIRST statement: same rank-local early-return + per-leaf put as
     # shard_forcing_latlon (#1362 round 4, blocker 6).
     _agree_ocean_mesh_entry(mesh, stack,
                             where="shard_forcing_stack_latlon")
     if mesh is None:
         return stack
 
+    assert_pytree_bytes_equal(stack, "shard_forcing_stack_latlon")
+
     def _put(leaf):
         if leaf is None or not hasattr(leaf, "ndim"):
             return leaf
         arr = jnp.asarray(leaf)
         if arr.ndim >= 3:
             spec = P(None, "lat", *((None,) * (arr.ndim - 2)))
         elif arr.ndim == 2:
             spec = P("lat", None)
         else:
             spec = P()
-        return jax.device_put(arr, NamedSharding(mesh, spec))
+        return addressable_shard_put(arr, NamedSharding(mesh, spec))
 
     return jax.tree.map(_put, stack)
 
 
 def append_vface_wall_row(v_lower):
     """Rebuild the full ``(n_lat+1, ...)`` staggered v-array from the lat-band
     carrier ``v_lower`` by appending the TOP wall row (zeros).
 
     This is THE reconstruction of the row the carrier drops: the regular-grid
     north pole wall / tripole cap row, identically zero under the v-carrier
     contract (:func:`shard_state_latlon` REFUSES a state whose top v-face row
     is live), so the result is bit-identical to the original staggered array.
     Shared by :func:`gather_state_latlon` (full-state gather) and the
     persistent-lane DEVICE-SIDE staggered reads (the OMIP driver's
     surface-current consumers, ``run_omip_core2._surface_uv_faces``) so the
     row reconstruction is written once.  Works on any trailing shape (3-D
     leaves or 2-D surface slices) and preserves sharding under GSPMD.
     """
     wall = jnp.zeros_like(v_lower[:1])
     return jnp.concatenate([v_lower, wall], axis=0)
 
 
 def gather_state_latlon(state, mesh):
     """Inverse of :func:`shard_state_latlon`: gather every leaf to a single device
     and rebuild the full ``(n_lat+1, ...)`` ``v`` / ``v_mask`` by appending the
@@ -693,63 +700,61 @@ def make_sharded_ocean_step(model, mesh):
     # zero row is exact — true for the cap-row convention of
     # ``create_synthetic_tripole`` and the eORCA masks (``v_mask[-1] == 0``;
     # the serial step keeps ``v[-1] == 0`` identically).  That contract is
     # asserted on the CONCRETE state in :func:`shard_state_latlon` — a live
     # (unmasked) seam v-row refuses loudly there instead of silently
     # reconstructing zeros here.
 
     # --- host-side band geometries + vertex masks (replicated, indexed in-body) ---
     band_grids = build_band_grids(model.grid, n_dev)
     band_vmasks = _build_band_vertex_masks(model, n_dev)
     template = band_grids[0]           # static-scalar source (uniform bands)
     array_field_names = _geom_array_field_names(template)
 
     # Stack each geometry ARRAY field over the band axis (rank 0..N-1) and
     # SHARD along that axis (#1370 stage (iii), codex round-18): each device
     # holds ONLY its own band's slab instead of the whole global stack —
     # this was one of the residual ~1.4-1.7 global-field-equivalents of
     # per-device residency left after the host-side-build fix (probe
     # 26524423). The leading axis has length n_dev, so P("lat") divides it
     # exactly; the body indexes its local slab at [0]. Values are unchanged
     # — same stack, different placement; the process-0 broadcast +
     # divergence guard below runs on HOST values and is placement-blind.
     rep = NamedSharding(mesh, P("lat"))
 
     def _replicated_put(arr, name):
-        # Multicontroller: a P() (fully-replicated) device_put ASSERTS the
-        # value is bit-identical on every process. The band-geometry arrays
-        # are (re)computed per process and can differ in their last ULPs
-        # (per-process XLA autotuning on device-derived grid fields), which
-        # trips that assert at larger sizes (job 26450848: LL576 np=4,
-        # area-scale fields differing at 1e-7 relative). broadcast_checked
-        # verifies cross-process agreement (raising on a REAL divergence
-        # rather than letting process 0 mask it) and then broadcasts process
-        # 0's bytes. Shared with the atmosphere lat-lon lane (#1362) —
-        # legoesm.parallel.geometry_consistency is the ONE implementation.
-        host = broadcast_checked(
-            arr, name, context="make_sharded_ocean_step")
-        return jax.device_put(jnp.asarray(host), rep)
+        # (Name kept for history; this is a SHARDED P("lat") stack put.)
+        # checked_shard_put replaces the broadcast_checked+device_put pair:
+        # the broadcast's psum program is [n_processes, stack] (nd x 849 MB
+        # at LL2304 — the @96/@128 wall), and a numpy device_put onto an
+        # all-process sharding pays jax's whole-array assert_equal on top.
+        # The per-band gate keeps the divergence contract (n_bands is
+        # schema-gated just below, so payload widths agree). ONE shared
+        # implementation: legoesm.parallel.geometry_consistency.
+        return checked_shard_put(
+            arr, name, rep, context="make_sharded_ocean_step",
+            n_bands=n_dev)
 
     # Schema gate FIRST (one fixed-shape collective every process reaches):
     # a process-dependent field list or a mixed jax_enable_x64 setting would
     # otherwise desynchronize the per-field gathers below instead of failing
     # with a clear message.
     # Build the raw stacks FIRST so the schema gate can also cover each
     # field's dtype class and ndim -- those decide the per-field payload
     # shape below, so a bool-vs-float disagreement must fail HERE rather than
     # deadlock in the per-field gather.
     _raw_geom = {
         name: jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
                         axis=0)
         for name in array_field_names
     }
     _raw_vmask = jnp.stack([jnp.asarray(m) for m in band_vmasks], axis=0)
     _gate_names = [*array_field_names, "vertex_mask"]
     assert_schema_agrees(
         _gate_names, n_dev, context="make_sharded_ocean_step",
         arrays=[*(_raw_geom[n] for n in array_field_names), _raw_vmask])
 
     geom_stacks = {name: _replicated_put(_raw_geom[name], name)
                    for name in array_field_names}
     vmask_stack = _replicated_put(_raw_vmask, "vertex_mask")
 
     # Static perms for the v north-boundary-row ppermute (band r receives band
@@ -843,51 +848,58 @@ def make_sharded_ocean_step(model, mesh):
         — there are no v-face ``(n_lat+1, …)`` forcing arrays, so no
         ``v_lower`` handling.  A wrong-leading-dim leaf would otherwise die
         inside shard_map with an opaque divisibility error.
         """
         leaves, _ = jax.tree_util.tree_flatten_with_path(forcing)
         for path, leaf in leaves:
             nd = int(getattr(leaf, "ndim", np.ndim(leaf)))
             if nd == 1:
                 # No OMIP forcing field is 1-D; _lat_spec would shard a
                 # profile/staggered vector over "lat" silently-wrong
                 # (codex r1 #2).
                 raise ValueError(
                     f"sharded ocean step: forcing leaf "
                     f"{jax.tree_util.keystr(path)} is 1-D "
                     f"(shape {tuple(leaf.shape)}); forcing must be "
                     f"cell-centered (n_lat, n_lon[, nlev]) arrays or "
                     f"scalars.")
             if nd >= 2 and int(leaf.shape[0]) != n_lat_global:
                 raise ValueError(
                     f"sharded ocean step: forcing leaf {jax.tree_util.keystr(path)} "
                     f"has leading dim {leaf.shape[0]} != n_lat "
                     f"({n_lat_global}); forcing must be cell-centered "
                     f"(n_lat, n_lon[, nlev]) to shard on the lat axis.")
 
     def sharded_step(state, dt, freshwater=None, surface_forcing=None,
-                     sponge=None, t_seconds=None):
+                     sponge=None, t_seconds=None, aux=None):
+        # ``aux``: the sharded geometry+vmask stacks. When this wrapper runs
+        # INSIDE an outer trace (a bench/driver jit/scan — jit-of-jit
+        # inlines the inner call), concrete closure arrays become
+        # OUTER-trace constants whose value the MLIR handler cannot fetch
+        # for non-addressable arrays (broken since #1370-iii sharded the
+        # stacks). Outer-jit callers MUST thread ``step.aux`` through their
+        # jit boundary as an ARGUMENT and pass it back here.
         # ONE forcing operand: None fields drop out of the pytree structure,
         # so specs derived by tree.map skip them automatically and the
         # structure key below distinguishes every None<->array combination.
         _agree_ocean_spmd_call(
             mesh, state, (freshwater, surface_forcing, sponge, t_seconds),
             where="make_sharded_ocean_step.step")
         forcing = (freshwater, surface_forcing, sponge, t_seconds)
         _validate_forcing_layout((freshwater, surface_forcing, sponge))
         # Cache key = the state's AND forcing's pytree STRUCTURE, plus the
         # forcing leaves' RANKS: in_specs/out_specs are derived from them,
         # so a later call with a different structure (an optional field
         # flipping None <-> Field, a sea-ice lane populating sf.salt_flux,
         # the restoring lane passing no forcing at all) OR a same-field
         # rank change (SpongeForcing.gamma is legitimately 2-D horizontal
         # OR 3-D full-rank — same structure, different _lat_spec; codex r1
         # #1) must rebuild the shard_map rather than reuse stale specs.
         forcing_ndims = tuple(
             int(getattr(leaf, "ndim", np.ndim(leaf)))
             for leaf in jax.tree.leaves(forcing))
         # The SPMD fused-halo switch is read at TRACE time inside the pad
         # dispatch — flipping LEGOESM_LATLON_SPMD_FUSED_HALO on a reused
         # step object must rebuild the shard_map, not reuse a stale jaxpr
         # (codex, audit item 7).
         import os as _os
 
@@ -916,72 +928,79 @@ def make_sharded_ocean_step(model, mesh):
                 out_specs=in_spec,
                 check_vma=False,
             ))
             _cache[key] = fn
         # Arm the SPMD halo backend ONLY around the call, then RESTORE the
         # previous backend (codex finding): leaving it globally armed makes a
         # later serial/full-domain ocean call take SPMD-only branches
         # (axis_index / ppermute / psum in pad_with_pole_bc_lat, conservation,
         # eta_floor) OUTSIDE a shard_map -> crash. The FIRST call traces with
         # the backend armed (baking the SPMD halo/reduction ops into the
         # compiled program); later calls reuse the cached compile, and the
         # arm/restore keeps any interleaved serial path untouched.
         # Save+restore the FULL backend state (backend + MPI topology + SPMD
         # mesh) so a prior "mpi"/"spmd" backend is restored intact: activate_*
         # clears the MPI topology, and set_halo_backend("mpi") REQUIRES a
         # topology (codex).
         from legoesm.grids.halo import (
             get_halo_backend, get_mpi_topology, get_spmd_mesh,
             set_halo_backend, set_spmd_mesh,
         )
         _prev_backend = get_halo_backend()
         _prev_topo = get_mpi_topology()
         _prev_mesh = get_spmd_mesh()
         activate_latlon_spmd_halo(mesh)
         try:
-            return fn(state, forcing, geom_stacks, vmask_stack,
-                      jnp.asarray(dt))
+            _geom, _vmask = aux if aux is not None else (geom_stacks,
+                                                        vmask_stack)
+            return fn(state, forcing, _geom, _vmask, jnp.asarray(dt))
         finally:
             set_spmd_mesh(_prev_mesh)
             set_halo_backend(_prev_backend, _prev_topo)
 
+    # Expose the stacks so outer-jit callers can pass them as arguments
+    # (see the ``aux`` note in the signature).
+    sharded_step.aux = (geom_stacks, vmask_stack)
     return sharded_step
 
 
 def make_sharded_ocean_step_global(model, mesh):
     """Return ``step(state_global, dt, surface_forcing=None, freshwater=None)``
     that takes a GLOBAL (single-device-layout) state + forcing and returns a
     GLOBAL state — the minimal-diff driver entry point.
 
     Wraps :func:`make_sharded_ocean_step`: shards the global state + forcing IN
     (:func:`shard_state_latlon` + :func:`shard_forcing_latlon`), runs the lat-band
     SPMD step, then gathers the state OUT (:func:`gather_state_latlon`).  This lets
     the OMIP host loop keep operating on a normal full-domain state — the per-step
     host BCs (SSS restore, prognostic ice, geothermal, BBL, nudge) see the
     gathered global state UNCHANGED — at the cost of a per-step gather/scatter
     (acceptable for the host-coupled OMIP driver; the pure-dynamics inner loop
     should use :func:`make_sharded_ocean_step` directly to stay sharded).
 
     ``mesh is None`` ⇒ the plain single-device ``model.step`` (no scatter/gather).
     """
     # FIRST statement: agree every rank-local input before ANY
     # rank-local check can raise or return (codex round-2).
     _agree_ocean_spmd_entry(model, mesh, where="make_sharded_ocean_step_global")
     if mesh is None:                   # single-device: plain step
         return lambda state, dt, surface_forcing=None, freshwater=None: (
             model.step(state, dt, freshwater=freshwater,
                        surface_forcing=surface_forcing))
 
     inner = make_sharded_ocean_step(model, mesh)
 
     def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
         _agree_ocean_spmd_call(mesh, state, (surface_forcing, freshwater),
                                where="make_sharded_ocean_step_global.step")
-        # Scatter the global state to the band layout; the forcing is sharded
-        # INSIDE ``inner`` (make_sharded_ocean_step lays it out), so pass it
-        # through global.
+        # Scatter the global state AND forcing to the band layout
+        # explicitly (the old comment claimed inner sharded the forcing;
+        # it forwarded it global and relied on implicit JIT input
+        # placement — jax's whole-array device_put assert under
+        # multicontroller, the nd-linear wall this module removes).
         ss = shard_state_latlon(state, mesh)
-        ss = inner(ss, dt, surface_forcing=surface_forcing,
-                   freshwater=freshwater)
+        ss = inner(ss, dt,
+                   surface_forcing=shard_forcing_latlon(surface_forcing, mesh),
+                   freshwater=shard_forcing_latlon(freshwater, mesh))
         return gather_state_latlon(ss, mesh)
 
     return sharded_step_global
diff --git a/tests/ocean/unit/test_sharded_geom_fingerprint.py b/tests/ocean/unit/test_sharded_geom_fingerprint.py
new file mode 100644
index 000000000..5b24a4b9e
--- /dev/null
+++ b/tests/ocean/unit/test_sharded_geom_fingerprint.py
@@ -0,0 +1,104 @@
+"""Unit tests for the band-geometry cross-process fingerprint gate.
+
+The gate decides whether ``make_sharded_ocean_step`` accepts per-process
+band-geometry stacks without the (removed, nd-linear-cost) process-0
+broadcast — see the 2026-08-03 fix note at the sharded put. These tests
+pin the gate's discrimination properties single-process (the
+multicontroller allgather wiring is exercised by the distributed suite).
+"""
+import numpy as np
+import pytest
+
+from legoesm.parallel.geometry_consistency import (
+    band_fingerprint as geom_band_fingerprint,
+    band_fingerprints_agree,
+)
+
+N_BANDS = 4
+SHAPE = (N_BANDS, 6, 8)
+
+
+def _gather(*hosts):
+    """Simulate process_allgather over per-process fingerprints."""
+    fps = [geom_band_fingerprint(h, N_BANDS) for h in hosts]
+    exacts = {fp[2] for fp in fps}
+    assert len(exacts) == 1
+    g_struct = np.stack([fp[0] for fp in fps])
+    g_vals = np.stack([fp[1] for fp in fps])
+    return g_struct, g_vals, fps[0][2]
+
+
+def test_identical_float_stacks_agree():
+    rng = np.random.default_rng(0)
+    a = rng.normal(size=SHAPE).astype(np.float32)
+    assert band_fingerprints_agree(*_gather(a, a.copy()))
+
+
+def test_ulp_scale_drift_agrees():
+    rng = np.random.default_rng(1)
+    a = rng.normal(size=SHAPE).astype(np.float64) + 10.0
+    b = np.nextafter(a, np.inf)  # a TRUE 1-ULP elementwise drift
+    assert band_fingerprints_agree(*_gather(a, b))
+
+
+def test_band_local_drift_refused_where_global_gate_passed():
+    # THE r14 discrimination case: a band-local drift SMALL enough that the
+    # old whole-array gate (sum/sumsq/absmax at rtol 1e-5) accepts it, with
+    # the global absmax held by an UNAFFECTED band — the per-band gate must
+    # still refuse.
+    rng = np.random.default_rng(2)
+    a = rng.normal(size=SHAPE).astype(np.float64) + 10.0
+    a[0, 0, 0] = 500.0          # absmax lives in band 0
+    b = a.copy()
+    b[2] *= 1.0 + 3e-5          # one band drifts; global sums move ~7e-6 rel
+
+    def _global_moments(x):
+        f = x.ravel().astype(np.float64)
+        return np.array([f.sum(), (f * f).sum(), np.abs(f).max()])
+
+    # the OLD global gate would have ACCEPTED this pair...
+    assert np.allclose(_global_moments(a), _global_moments(b),
+                       rtol=1e-5, atol=0.0)
+    # ...the per-band gate refuses it.
+    assert not band_fingerprints_agree(*_gather(a, b))
+
+
+def test_exact_dtype_permutation_refused():
+    # Moment fingerprints are blind to permutations; the positional
+    # per-band byte digest must not be.
+    a = np.zeros(SHAPE, dtype=np.int32)
+    a[1, 2, 3] = 1
+    b = np.zeros_like(a)
+    b[1, 3, 2] = 1  # same count, different position, same band
+    assert not band_fingerprints_agree(*_gather(a, b))
+
+
+def test_bool_mask_two_cell_flip_refused():
+    a = np.zeros(SHAPE, dtype=bool)
+    a[0, 0, 0] = True
+    b = a.copy()
+    b[0, 0, 0] = False
+    b[0, 5, 7] = True  # true-count preserved
+    assert not band_fingerprints_agree(*_gather(a, b))
+
+
+def test_nonfinite_count_mismatch_refused():
+    a = np.ones(SHAPE, dtype=np.float32)
+    b = a.copy()
+    b[3, 0, 0] = np.nan  # struct carries per-band non-finite counts
+    assert not band_fingerprints_agree(*_gather(a, b))
+
+
+def test_shape_mismatch_refused():
+    a = np.ones(SHAPE, dtype=np.float32)
+    b = np.ones((N_BANDS, 6, 9), dtype=np.float32)
+    fa = geom_band_fingerprint(a, N_BANDS)
+    fb = geom_band_fingerprint(b, N_BANDS)
+    # Different shapes -> different struct lengths; the agree helper is
+    # only called on stackable gathers, so assert the structs differ.
+    assert fa[0].shape != fb[0].shape or not np.array_equal(fa[0], fb[0])
+
+
+def test_wrong_leading_axis_raises():
+    with pytest.raises(ValueError):
+        geom_band_fingerprint(np.ones((3, 2)), N_BANDS)
e13de3afe (HEAD -> worktree-scaling-campaign, origin/worktree-scaling-campaign) Merge remote-tracking branch 'origin/main' into worktree-scaling-campaign
1b71f5dc3 fix(plot): repair literal-newline in oc panel note
ac4289131 docs(scaling): OCEAN AT HUNDREDS — LL2304 @96/@128 receipts (11.63/13.00 GC/s, eff 0.84)
625b50f9d fix(ocean-spmd): three stacked multicontroller walls — assert-free puts, exact-hash gate, aux threading
b8ec472cc fix(ocean-spmd): drop the geometry-stack broadcast — its psum program was nd x global
d5fcc93f8 (origin/main, origin/HEAD) Merge pull request #1456 from climate-federation/worktree-scaling-campaign
d50e4325e perf(scaling): night receipts — ensemble +0.6%, combine null, NET/IB proven, oc arg-linearity, sick-allocation forensics
cfd2cb366 Merge pull request #1454 from climate-federation/worktree-scaling-campaign

exec
/usr/bin/bash -lc "rg -n -C 4 'class LatLonCGridGeometry|LatLonCGridGeometry\\s*=|NamedTuple.*LatLonCGridGeometry|total_area' packages/ocean packages/core | sed -n '1,300p'; rg -n 'band_fingerprint|checked_shard_put|assert_pytree_bytes_equal|addressable_shard_put' tests packages --glob '*.py'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 586ms:
packages/core/legoesm/grids/tripole.py-398-    # Metric arrays [metres] — NEMO stores these as e1t, e2t etc.
packages/core/legoesm/grids/tripole.py-399-    dx_T = raw["e1t"].astype(dtype)
packages/core/legoesm/grids/tripole.py-400-    dy_T = raw["e2t"].astype(dtype)
packages/core/legoesm/grids/tripole.py-401-    area_T = dx_T * dy_T
packages/core/legoesm/grids/tripole.py:402:    total_area = jnp.sum(area_T)
packages/core/legoesm/grids/tripole.py-403-
packages/core/legoesm/grids/tripole.py-404-    # u-point metrics.  NEMO e1u/e2u have shape (n_lat, n_lon) but on
packages/core/legoesm/grids/tripole.py-405-    # a C-grid u-points have shape (n_lat, n_lon+1).  For NEMO, the
packages/core/legoesm/grids/tripole.py-406-    # u-point at index i is between T-cell (i-1) and T-cell i in the
--
packages/core/legoesm/grids/tripole.py-538-        lon_T=lon_T,
packages/core/legoesm/grids/tripole.py-539-        dx_T=dx_T,
packages/core/legoesm/grids/tripole.py-540-        dy_T=dy_T,
packages/core/legoesm/grids/tripole.py-541-        area_T=area_T,
packages/core/legoesm/grids/tripole.py:542:        total_area=total_area,
packages/core/legoesm/grids/tripole.py-543-        dx_u=dx_u,
packages/core/legoesm/grids/tripole.py-544-        dy_u=dy_u,
packages/core/legoesm/grids/tripole.py-545-        dx_v=dx_v,
packages/core/legoesm/grids/tripole.py-546-        dy_v=dy_v,
--
packages/core/legoesm/grids/tripole.py-591-    strictly increasing northward (Coriolis ``f_T = 2Ω sin(lat)`` follows); the
packages/core/legoesm/grids/tripole.py-592-    rotation angles south of the cap are regular lat-lon (cos=1, sin=0) — and the
packages/core/legoesm/grids/tripole.py-593-    edge row already is, below ``cap_j``.
packages/core/legoesm/grids/tripole.py-594-
packages/core/legoesm/grids/tripole.py:595:    ``total_area`` is PRESERVED exactly (the area-weighted-mean denominator must
packages/core/legoesm/grids/tripole.py:596:    not pick up the spurious land padding — the same GLOBAL-``total_area``
packages/core/legoesm/grids/tripole.py-597-    invariant the band slicer keeps).  ``fold_j``/``cap_j`` shift ``+n_pad``;
packages/core/legoesm/grids/tripole.py-598-    ``perm_T``/``perm_v`` (lon permutations) and the vector signs are unchanged.
packages/core/legoesm/grids/tripole.py-599-
packages/core/legoesm/grids/tripole.py-600-    ``n_pad == 0`` returns the grid unchanged.  Requires an ACTIVE fold (the
--
packages/core/legoesm/grids/tripole.py-664-        lon_T=lon_T_pad,
packages/core/legoesm/grids/tripole.py-665-        dx_T=_edge_pad(grid.dx_T),
packages/core/legoesm/grids/tripole.py-666-        dy_T=_edge_pad(grid.dy_T),
packages/core/legoesm/grids/tripole.py-667-        area_T=_edge_pad(grid.area_T),
packages/core/legoesm/grids/tripole.py:668:        total_area=grid.total_area,            # PRESERVED (no spurious land area)
packages/core/legoesm/grids/tripole.py-669-        dx_u=_edge_pad(grid.dx_u),
packages/core/legoesm/grids/tripole.py-670-        dy_u=_edge_pad(grid.dy_u),
packages/core/legoesm/grids/tripole.py-671-        dx_v=_edge_pad(grid.dx_v),
packages/core/legoesm/grids/tripole.py-672-        dy_v=_edge_pad(grid.dy_v),
--
packages/core/legoesm/grids/voronoi.py-126-    def grid_area(self) -> jnp.ndarray:
packages/core/legoesm/grids/voronoi.py-127-        return self.areaCell
packages/core/legoesm/grids/voronoi.py-128-
packages/core/legoesm/grids/voronoi.py-129-    @property
packages/core/legoesm/grids/voronoi.py:130:    def grid_total_area(self):
packages/core/legoesm/grids/voronoi.py-131-        return jnp.sum(self.areaCell)
packages/core/legoesm/grids/voronoi.py-132-
packages/core/legoesm/grids/voronoi.py-133-    @property
packages/core/legoesm/grids/voronoi.py-134-    def grid_coriolis(self) -> jnp.ndarray:
--
packages/core/legoesm/grids/voronoi.py-263-        c = vertices.mean(axis=0)
packages/core/legoesm/grids/voronoi.py-264-        return c / np.linalg.norm(c)
packages/core/legoesm/grids/voronoi.py-265-    # Decompose into triangles from v0 and compute (density-)area-weighted centroid
packages/core/legoesm/grids/voronoi.py-266-    v0 = vertices[0]
packages/core/legoesm/grids/voronoi.py:267:    total_area = 0.0
packages/core/legoesm/grids/voronoi.py-268-    centroid = np.zeros(3)
packages/core/legoesm/grids/voronoi.py-269-    for i in range(1, n - 1):
packages/core/legoesm/grids/voronoi.py-270-        v1, v2 = vertices[i], vertices[i + 1]
packages/core/legoesm/grids/voronoi.py-271-        # True spherical triangle area via spherical excess
--
packages/core/legoesm/grids/voronoi.py-277-            # density**2 therefore drives cell AREA ~ 1/density, i.e. a region with
packages/core/legoesm/grids/voronoi.py-278-            # density d gets ~d x smaller cells (the intuitive resolution control).
packages/core/legoesm/grids/voronoi.py-279-            weight = area * _density_at(tri_center, density_fn) ** 2
packages/core/legoesm/grids/voronoi.py-280-        centroid += weight * tri_center
packages/core/legoesm/grids/voronoi.py:281:        total_area += weight
packages/core/legoesm/grids/voronoi.py:282:    if total_area > 0:
packages/core/legoesm/grids/voronoi.py:283:        centroid /= total_area
packages/core/legoesm/grids/voronoi.py-284-    else:
packages/core/legoesm/grids/voronoi.py-285-        centroid = vertices.mean(axis=0)
packages/core/legoesm/grids/voronoi.py-286-    norm = np.linalg.norm(centroid)
packages/core/legoesm/grids/voronoi.py-287-    if norm > 1e-15:
--
packages/core/legoesm/grids/plane.py-104-    area_T : jax.Array
packages/core/legoesm/grids/plane.py-105-        Per-cell area at cell centres, shape ``(ny, nx)``. Uniform
packages/core/legoesm/grids/plane.py-106-        ``dx * dy`` on the plane but stored 2D for protocol
packages/core/legoesm/grids/plane.py-107-        compatibility.
packages/core/legoesm/grids/plane.py:108:    total_area : jax.Array
packages/core/legoesm/grids/plane.py-109-        Scalar sum equal to ``Lx * Ly``.
packages/core/legoesm/grids/plane.py-110-    surface_mask : jax.Array
packages/core/legoesm/grids/plane.py-111-        Shape ``(ny, nx)``. Convention: ``1.0`` = active surface cell;
packages/core/legoesm/grids/plane.py-112-        ``0.0`` = blocked. Uniformly ``1.0`` on the plane (no land).
--
packages/core/legoesm/grids/plane.py-131-    xu: jax.Array
packages/core/legoesm/grids/plane.py-132-    yv: jax.Array
packages/core/legoesm/grids/plane.py-133-    f_y: jax.Array
packages/core/legoesm/grids/plane.py-134-    area_T: jax.Array
packages/core/legoesm/grids/plane.py:135:    total_area: jax.Array
packages/core/legoesm/grids/plane.py-136-    surface_mask: jax.Array
packages/core/legoesm/grids/plane.py-137-
packages/core/legoesm/grids/plane.py-138-    # ------------------------------------------------------------------
packages/core/legoesm/grids/plane.py-139-    # GridProtocol
--
packages/core/legoesm/grids/plane.py-157-    def grid_area(self) -> jax.Array:
packages/core/legoesm/grids/plane.py-158-        return self.area_T
packages/core/legoesm/grids/plane.py-159-
packages/core/legoesm/grids/plane.py-160-    @property
packages/core/legoesm/grids/plane.py:161:    def grid_total_area(self) -> jax.Array:
packages/core/legoesm/grids/plane.py:162:        return self.total_area
packages/core/legoesm/grids/plane.py-163-
packages/core/legoesm/grids/plane.py-164-    @property
packages/core/legoesm/grids/plane.py-165-    def grid_coriolis(self) -> jax.Array:
packages/core/legoesm/grids/plane.py-166-        return self.f_y
--
packages/core/legoesm/grids/plane.py-413-        )
packages/core/legoesm/grids/plane.py-414-    f_y = f_y_1d[:, None] * jnp.ones((1, nx), dtype=dtype)
packages/core/legoesm/grids/plane.py-415-
packages/core/legoesm/grids/plane.py-416-    area_T = jnp.full((ny, nx), float(dx * dy), dtype=dtype)
packages/core/legoesm/grids/plane.py:417:    total_area = jnp.asarray(Lx * Ly, dtype=dtype)
packages/core/legoesm/grids/plane.py-418-    surface_mask = jnp.ones((ny, nx), dtype=dtype)
packages/core/legoesm/grids/plane.py-419-
packages/core/legoesm/grids/plane.py-420-    return PlaneGrid(
packages/core/legoesm/grids/plane.py-421-        nx=int(nx),
--
packages/core/legoesm/grids/plane.py-435-        xu=xu,
packages/core/legoesm/grids/plane.py-436-        yv=yv,
packages/core/legoesm/grids/plane.py-437-        f_y=f_y,
packages/core/legoesm/grids/plane.py-438-        area_T=area_T,
packages/core/legoesm/grids/plane.py:439:        total_area=total_area,
packages/core/legoesm/grids/plane.py-440-        surface_mask=surface_mask,
packages/core/legoesm/grids/plane.py-441-    )
--
packages/core/legoesm/grids/halo_latlon.py-940-    Field rules mirror the slicer (stagger-aware):
packages/core/legoesm/grids/halo_latlon.py-941-
packages/core/legoesm/grids/halo_latlon.py-942-    * T-/u-point metrics + 1-D lat arrays → cell-row widening;
packages/core/legoesm/grids/halo_latlon.py-943-    * v-/q-point metrics → v-face widening;
packages/core/legoesm/grids/halo_latlon.py:944:    * ``lon``/``dlon``/``dlat``/``radius``/``total_area`` unchanged
packages/core/legoesm/grids/halo_latlon.py:945:      (``total_area`` stays the GLOBAL denominator);
packages/core/legoesm/grids/halo_latlon.py-946-    * ``n_lat`` → ``n_lat + 2*halo`` (static Python int);
packages/core/legoesm/grids/halo_latlon.py-947-    * ``fold`` must be inactive/non-local — the wide-halo path refuses
packages/core/legoesm/grids/halo_latlon.py-948-      tripolar folds (caller-validated; this helper asserts).
packages/core/legoesm/grids/halo_latlon.py-949-
--
packages/core/legoesm/grids/latlon.py-72-    f: jax.Array                # (n_lat, n_lon) Coriolis = 2*Omega*sin(lat)
packages/core/legoesm/grids/latlon.py-73-    dx: jax.Array               # (n_lat, n_lon) distance over 2 cells in lon [m]
packages/core/legoesm/grids/latlon.py-74-    dy: jax.Array               # (n_lat,) distance over 2 cells in lat [m]
packages/core/legoesm/grids/latlon.py-75-    area: jax.Array             # (n_lat, n_lon) cell area [m^2]
packages/core/legoesm/grids/latlon.py:76:    total_area: jax.Array       # scalar sum of all areas
packages/core/legoesm/grids/latlon.py-77-    dlon: float                 # longitude spacing [rad]
packages/core/legoesm/grids/latlon.py-78-    dlat: float                 # representative latitude spacing [rad]
packages/core/legoesm/grids/latlon.py-79-    # For uniform-dlat grids this is the constant cell-row dlat. For
packages/core/legoesm/grids/latlon.py-80-    # Mercator the cell-row dlat varies with latitude (largest at the
--
packages/core/legoesm/grids/latlon.py-112-    def grid_area(self) -> jax.Array:
packages/core/legoesm/grids/latlon.py-113-        return self.area
packages/core/legoesm/grids/latlon.py-114-
packages/core/legoesm/grids/latlon.py-115-    @property
packages/core/legoesm/grids/latlon.py:116:    def grid_total_area(self):
packages/core/legoesm/grids/latlon.py:117:        return self.total_area
packages/core/legoesm/grids/latlon.py-118-
packages/core/legoesm/grids/latlon.py-119-    @property
packages/core/legoesm/grids/latlon.py-120-    def grid_coriolis(self) -> jax.Array:
packages/core/legoesm/grids/latlon.py-121-        return self.f
--
packages/core/legoesm/grids/latlon.py-313-    dy = radius * 2.0 * dlat * jnp.ones((n_lat,))
packages/core/legoesm/grids/latlon.py-314-
packages/core/legoesm/grids/latlon.py-315-    # Cell area: exact spherical cap (sums to 4πR² for global grid)
packages/core/legoesm/grids/latlon.py-316-    area = _exact_uniform_cell_area_lat(radius, lat, dlat, dlon, n_lon)
packages/core/legoesm/grids/latlon.py:317:    total_area = jnp.sum(area)
packages/core/legoesm/grids/latlon.py-318-
packages/core/legoesm/grids/latlon.py-319-    _c = lambda a: a.astype(dtype) if hasattr(a, 'astype') else a
packages/core/legoesm/grids/latlon.py-320-    return LatLonGrid(
packages/core/legoesm/grids/latlon.py-321-        n_lat=n_lat,
--
packages/core/legoesm/grids/latlon.py-332-        f=_c(f),
packages/core/legoesm/grids/latlon.py-333-        dx=_c(dx),
packages/core/legoesm/grids/latlon.py-334-        dy=_c(dy),
packages/core/legoesm/grids/latlon.py-335-        area=_c(area),
packages/core/legoesm/grids/latlon.py:336:        total_area=total_area,
packages/core/legoesm/grids/latlon.py-337-        dlon=float(dlon),
packages/core/legoesm/grids/latlon.py-338-        dlat=float(dlat),
packages/core/legoesm/grids/latlon.py-339-        omega=float(omega),
packages/core/legoesm/grids/latlon.py-340-    )
--
packages/core/legoesm/grids/latlon.py-481-    # values, but stored as an array for API uniformity with Mercator).
packages/core/legoesm/grids/latlon.py-482-    dy = radius * 2.0 * dlat * jnp.ones((ny,))
packages/core/legoesm/grids/latlon.py-483-
packages/core/legoesm/grids/latlon.py-484-    area = _exact_uniform_cell_area_lat(radius, lat, dlat, dlon, nx)
packages/core/legoesm/grids/latlon.py:485:    total_area = jnp.sum(area)
packages/core/legoesm/grids/latlon.py-486-
packages/core/legoesm/grids/latlon.py-487-    # Wall mask: walls at N/S always; E/W walls only for closed basin
packages/core/legoesm/grids/latlon.py-488-    wall_mask = _regional_wall_mask(ny, nx, periodic_x, dtype)
packages/core/legoesm/grids/latlon.py-489-
--
packages/core/legoesm/grids/latlon.py-503-        f=_c(f),
packages/core/legoesm/grids/latlon.py-504-        dx=_c(dx),
packages/core/legoesm/grids/latlon.py-505-        dy=_c(dy),
packages/core/legoesm/grids/latlon.py-506-        area=_c(area),
packages/core/legoesm/grids/latlon.py:507:        total_area=total_area,
packages/core/legoesm/grids/latlon.py-508-        dlon=float(dlon),
packages/core/legoesm/grids/latlon.py-509-        dlat=float(dlat),
packages/core/legoesm/grids/latlon.py-510-        omega=float(omega),
packages/core/legoesm/grids/latlon.py-511-    )
--
packages/core/legoesm/grids/latlon.py-741-        # spherical grid, not assuming uniform dlat.
packages/core/legoesm/grids/latlon.py-742-        sin_face = jnp.sin(lat_face)
packages/core/legoesm/grids/latlon.py-743-        area_lat = radius**2 * dlon * jnp.abs(sin_face[1:] - sin_face[:-1])  # (n_lat,)
packages/core/legoesm/grids/latlon.py-744-        area = area_lat[:, None] * jnp.ones((1, n_lon))
packages/core/legoesm/grids/latlon.py:745:    total_area = jnp.sum(area)
packages/core/legoesm/grids/latlon.py-746-
packages/core/legoesm/grids/latlon.py-747-    # Representative dlat — smallest cell-row dlat. Mercator cell-row
packages/core/legoesm/grids/latlon.py-748-    # dlat is largest at the equator (φ = 0, cos φ = 1) and smallest
packages/core/legoesm/grids/latlon.py-749-    # near the truncation latitude (cos φ → 0 ⇒ dφ/dk → 0), so the
--
packages/core/legoesm/grids/latlon.py-768-        f=_c(f),
packages/core/legoesm/grids/latlon.py-769-        dx=_c(dx),
packages/core/legoesm/grids/latlon.py-770-        dy=_c(dy),
packages/core/legoesm/grids/latlon.py-771-        area=_c(area),
packages/core/legoesm/grids/latlon.py:772:        total_area=total_area,
packages/core/legoesm/grids/latlon.py-773-        dlon=float(dlon),
packages/core/legoesm/grids/latlon.py-774-        dlat=float(dlat_repr),
packages/core/legoesm/grids/latlon.py-775-        omega=float(omega),
packages/core/legoesm/grids/latlon.py-776-    )
--
packages/core/legoesm/grids/latlon.py-976-    # Exact spherical area: R² · Δλ · |sin(φ_face[j+1]) − sin(φ_face[j])|.
packages/core/legoesm/grids/latlon.py-977-    sin_face = jnp.sin(lat_face)
packages/core/legoesm/grids/latlon.py-978-    area_lat = radius**2 * dlon * jnp.abs(sin_face[1:] - sin_face[:-1])
packages/core/legoesm/grids/latlon.py-979-    area = area_lat[:, None] * jnp.ones((1, nx))
packages/core/legoesm/grids/latlon.py:980:    total_area = jnp.sum(area)
packages/core/legoesm/grids/latlon.py-981-
packages/core/legoesm/grids/latlon.py-982-    # Representative scalar dlat = smallest (most CFL-stringent) row —
packages/core/legoesm/grids/latlon.py-983-    # the Mercator convention. Diagnostics only; operators use grid.dy.
packages/core/legoesm/grids/latlon.py-984-    dlat_repr = float(np.min(np.deg2rad(d_full)))
--
packages/core/legoesm/grids/latlon.py-1003-        f=_c(f),
packages/core/legoesm/grids/latlon.py-1004-        dx=_c(dx),
packages/core/legoesm/grids/latlon.py-1005-        dy=_c(dy),
packages/core/legoesm/grids/latlon.py-1006-        area=_c(area),
packages/core/legoesm/grids/latlon.py:1007:        total_area=total_area,
packages/core/legoesm/grids/latlon.py-1008-        dlon=float(dlon),
packages/core/legoesm/grids/latlon.py-1009-        dlat=float(dlat_repr),
packages/core/legoesm/grids/latlon.py-1010-        omega=float(omega),
packages/core/legoesm/grids/latlon.py-1011-    )
--
packages/core/legoesm/grids/latlon.py-1141-        vector_sign_v=-1.0,
packages/core/legoesm/grids/latlon.py-1142-    )
packages/core/legoesm/grids/latlon.py-1143-
packages/core/legoesm/grids/latlon.py-1144-
packages/core/legoesm/grids/latlon.py:1145:class LatLonCGridGeometry(NamedTuple):
packages/core/legoesm/grids/latlon.py-1146-    """Per-cell metric container for orthogonal curvilinear C-grids.
packages/core/legoesm/grids/latlon.py-1147-
packages/core/legoesm/grids/latlon.py-1148-    This NamedTuple stores pre-computed metric arrays at every stagger
packages/core/legoesm/grids/latlon.py-1149-    point (T, u, v, q) so that operators never need to compute
--
packages/core/legoesm/grids/latlon.py-1170-    dx_T, dy_T : jax.Array
packages/core/legoesm/grids/latlon.py-1171-        Single-cell width/height at T-points [m].
packages/core/legoesm/grids/latlon.py-1172-    area_T : jax.Array
packages/core/legoesm/grids/latlon.py-1173-        Cell area at T-points [m^2].
packages/core/legoesm/grids/latlon.py:1174:    total_area : jax.Array
packages/core/legoesm/grids/latlon.py-1175-        Scalar sum of all T-point areas.
packages/core/legoesm/grids/latlon.py-1176-
packages/core/legoesm/grids/latlon.py-1177-    dx_u, dy_u : jax.Array
packages/core/legoesm/grids/latlon.py-1178-        Zonal/meridional spacing at u-points [m].
--
packages/core/legoesm/grids/latlon.py-1228-    # T-point metrics
packages/core/legoesm/grids/latlon.py-1229-    dx_T: jax.Array    # (n_lat, n_lon) single-cell zonal width [m]
packages/core/legoesm/grids/latlon.py-1230-    dy_T: jax.Array    # (n_lat, n_lon) single-cell meridional height [m]
packages/core/legoesm/grids/latlon.py-1231-    area_T: jax.Array  # (n_lat, n_lon) cell area [m^2]
packages/core/legoesm/grids/latlon.py:1232:    total_area: jax.Array  # scalar
packages/core/legoesm/grids/latlon.py-1233-
packages/core/legoesm/grids/latlon.py-1234-    # u-point metrics (lon interfaces)
packages/core/legoesm/grids/latlon.py-1235-    dx_u: jax.Array    # (n_lat, n_lon+1) zonal spacing [m]
packages/core/legoesm/grids/latlon.py-1236-    dy_u: jax.Array    # (n_lat, n_lon+1) meridional extent [m]
--
packages/core/legoesm/grids/latlon.py-1307-    def grid_area(self) -> jax.Array:
packages/core/legoesm/grids/latlon.py-1308-        return self.area_T
packages/core/legoesm/grids/latlon.py-1309-
packages/core/legoesm/grids/latlon.py-1310-    @property
packages/core/legoesm/grids/latlon.py:1311:    def grid_total_area(self):
packages/core/legoesm/grids/latlon.py:1312:        return self.total_area
packages/core/legoesm/grids/latlon.py-1313-
packages/core/legoesm/grids/latlon.py-1314-    @property
packages/core/legoesm/grids/latlon.py-1315-    def grid_coriolis(self) -> jax.Array:
packages/core/legoesm/grids/latlon.py-1316-        return self.f_T
--
packages/core/legoesm/grids/latlon.py-1569-    # with the bridged LatLonGrid to storage-dtype roundoff (not bit-exact:
packages/core/legoesm/grids/latlon.py-1570-    # the grid casts lat_v then cos's, this cos's the faces then casts).
packages/core/legoesm/grids/latlon.py-1571-    cos_lat_v_1d = jnp.maximum(jnp.abs(jnp.cos(lat_face)), 1e-10)
packages/core/legoesm/grids/latlon.py-1572-
packages/core/legoesm/grids/latlon.py:1573:    total_area = jnp.sum(area_legacy)
packages/core/legoesm/grids/latlon.py-1574-
packages/core/legoesm/grids/latlon.py-1575-    # Cast 1D coordinates + legacy fields to storage dtype.  All
packages/core/legoesm/grids/latlon.py-1576-    # subsequent metric computations use the cast values so that
packages/core/legoesm/grids/latlon.py-1577-    # operator-inline and geometry-precomputed paths are bit-exact.
--
packages/core/legoesm/grids/latlon.py-1619-        * jnp.ones((1, n_lon + 1))
packages/core/legoesm/grids/latlon.py-1620-    )
packages/core/legoesm/grids/latlon.py-1621-
packages/core/legoesm/grids/latlon.py-1622-    if metric_convention == "nemo_isotropic":
packages/core/legoesm/grids/latlon.py:1623:        # total_area must track area_T's convention here (area_legacy /
packages/core/legoesm/grids/latlon.py:1624:        # the pre-branch `total_area` above is always the exact-convention
packages/core/legoesm/grids/latlon.py:1625:        # area). "exact" leaves the pre-branch `total_area` (summed from
packages/core/legoesm/grids/latlon.py-1626-        # area_legacy BEFORE the storage-dtype cast) untouched -- summing
packages/core/legoesm/grids/latlon.py-1627-        # the already-cast area_T here instead would silently change its
packages/core/legoesm/grids/latlon.py-1628-        # accumulation dtype/order and shift bit-identical callers by
packages/core/legoesm/grids/latlon.py-1629-        # float32 roundoff (caught by test_latlon_geometry.py::
packages/core/legoesm/grids/latlon.py-1630-        # TestLatLonGridParity::test_area_parity).
packages/core/legoesm/grids/latlon.py:1631:        total_area = jnp.sum(area_T)
packages/core/legoesm/grids/latlon.py-1632-
packages/core/legoesm/grids/latlon.py-1633-    # ------- v-point metrics (n_lat+1, n_lon) -------
packages/core/legoesm/parallel/geometry_consistency.py:614:def band_fingerprint(host, n_bands):
packages/core/legoesm/parallel/geometry_consistency.py:635:            f"band_fingerprint: leading axis {host.shape[0]} != n_bands "
packages/core/legoesm/parallel/geometry_consistency.py:659:def band_fingerprints_agree(g_struct, g_vals, is_exact, rtol=None):
packages/core/legoesm/parallel/geometry_consistency.py:660:    """True iff every process's :func:`band_fingerprint` matches process 0's."""
packages/core/legoesm/parallel/geometry_consistency.py:671:def checked_shard_put(arr, name, sharding, *, context, n_bands):
packages/core/legoesm/parallel/geometry_consistency.py:687:    struct, vals, is_exact = band_fingerprint(host, n_bands)
packages/core/legoesm/parallel/geometry_consistency.py:690:    if not band_fingerprints_agree(g_struct, g_vals, is_exact):
packages/core/legoesm/parallel/geometry_consistency.py:700:def assert_pytree_bytes_equal(tree, what):
packages/core/legoesm/parallel/geometry_consistency.py:702:    :func:`checked_shard_put`-style puts bypass on NON-band inputs (state /
packages/core/legoesm/parallel/geometry_consistency.py:725:def addressable_shard_put(arr, sharding):
packages/core/legoesm/parallel/geometry_consistency.py:728:    :func:`assert_pytree_bytes_equal`). Single-process: plain device_put."""
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:49:    FLAG_ABSENT, addressable_shard_put, assert_flags_agree,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:50:    assert_pytree_bytes_equal, assert_schema_agrees, checked_shard_put,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:315:    assert_pytree_bytes_equal(state, "shard_state_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:333:        return field.replace(data=addressable_shard_put(field.data, sh))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:351:        return field.replace(data=addressable_shard_put(v_lower, sh))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:373:            updates[name] = addressable_shard_put(arr, NamedSharding(mesh, spec))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:394:    assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:400:        return addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:437:    assert_pytree_bytes_equal(stack, "shard_forcing_stack_latlon")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:449:        return addressable_shard_put(arr, NamedSharding(mesh, spec))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:726:        # checked_shard_put replaces the broadcast_checked+device_put pair:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:733:        return checked_shard_put(
tests/ocean/unit/test_sharded_geom_fingerprint.py:13:    band_fingerprint as geom_band_fingerprint,
tests/ocean/unit/test_sharded_geom_fingerprint.py:14:    band_fingerprints_agree,
tests/ocean/unit/test_sharded_geom_fingerprint.py:23:    fps = [geom_band_fingerprint(h, N_BANDS) for h in hosts]
tests/ocean/unit/test_sharded_geom_fingerprint.py:34:    assert band_fingerprints_agree(*_gather(a, a.copy()))
tests/ocean/unit/test_sharded_geom_fingerprint.py:41:    assert band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:63:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:73:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:82:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:89:    assert not band_fingerprints_agree(*_gather(a, b))
tests/ocean/unit/test_sharded_geom_fingerprint.py:95:    fa = geom_band_fingerprint(a, N_BANDS)
tests/ocean/unit/test_sharded_geom_fingerprint.py:96:    fb = geom_band_fingerprint(b, N_BANDS)
tests/ocean/unit/test_sharded_geom_fingerprint.py:104:        geom_band_fingerprint(np.ones((3, 2)), N_BANDS)

codex
I found the expected ocean ordering and no remaining ocean-local fingerprint copy. One concern is real: the new `aux` call argument is not represented in the per-call agreement payload, so a rank-local `aux=None`/provided (or schema) divergence can bypass the gate. I’m validating its downstream failure mode and the package/import surface now.
exec
/usr/bin/bash -lc "nl -ba packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py | sed -n '210,450p'; nl -ba packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py | sed -n '530,625p'; sed -n '1,180p' packages/core/pyproject.toml; rg -n 'from legoesm\\.ocean|import legoesm\\.ocean|from legoesm\\.atmosphere|import legoesm\\.atmosphere' packages/core/legoesm; rg --files packages -g '__init__.py' | rg '/legoesm/__init__\\.py"'$|parallel/__init__'"\\.py"'$|ocean/__init__'"\\.py"'$'"'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 351ms:
   210	def _build_band_vertex_masks(model, n_dev):
   211	    """Per-band vertex masks (n_lat/N+1, n_lon+1): SLICE the model's primed GLOBAL
   212	    vertex mask ``[s : e+1]`` per band (the v/q-row stagger ``slice_cgrid_geometry_
   213	    to_band`` uses for ``area_q``).
   214	
   215	    The slice — NOT a per-band ``compute_vertex_mask`` — is what makes the
   216	    band-cut rows correct.  The global ``model._vertex_mask`` already encoded the
   217	    four-cell product at EVERY vertex row including the interior cut rows (it saw
   218	    the neighbour-band cells), so band ``r``'s vertex rows ``[r*nl : r*nl+nl+1]``
   219	    are exact.  A per-band ``compute_vertex_mask`` would instead pad the band's
   220	    cell mask with a POLE WALL at the cut (``pad_with_pole_bc_lat`` has no SPMD
   221	    branch — it falls to the local constant pad), delivering a wrong (walled)
   222	    north/south boundary vertex row at interior cuts — the "one row off at the
   223	    cut" failure the ``compute_vertex_mask`` docstring warns about.
   224	
   225	    Requires the model's vertex mask to be primed (``model._ensure_vertex_mask``
   226	    on the concrete initial state; ``step`` does this on the first call).
   227	    """
   228	    vmask = model._vertex_mask
   229	    if vmask is None:
   230	        raise RuntimeError(
   231	            "make_sharded_ocean_step: the model vertex-mask cache is not "
   232	            "primed.  Call model._ensure_vertex_mask(state) (or model.step) "
   233	            "on the concrete initial state before building the sharded step.")
   234	    vmask = np.asarray(vmask)                   # (n_lat+1, n_lon+1)
   235	    n_lat = vmask.shape[0] - 1
   236	    nl = n_lat // n_dev
   237	    # v/q-row stagger: band r owns global vertex rows [r*nl : r*nl+nl+1] (the
   238	    # shared interior boundary vertex row appears in band r AND band r+1) — the
   239	    # same [s:e+1] slice slice_cgrid_geometry_to_band applies to area_q.
   240	    return [jnp.asarray(vmask[r * nl: r * nl + nl + 1]) for r in range(n_dev)]
   241	
   242	
   243	# Ordered flag names for the ocean MESH+TREE gate used by the scatter/gather
   244	# bridges and by the returned SPMD callable. STATIC tuple: fixed width.
   245	_OCEAN_MESH_ENTRY_FLAGS = (
   246	    "has_mesh", "n_dev", "n_axes", "axis_names", "axis_sizes",
   247	    "has_tree", "tree_schema",
   248	)
   249	
   250	
   251	def _ocean_mesh_axis_terms(mesh):
   252	    """``(axis_names, axis_sizes)`` term lists; never raises."""
   253	    if mesh is None:
   254	        return (), ()
   255	    try:
   256	        names = tuple(str(a) for a in mesh.axis_names)
   257	    except Exception:                       # pragma: no cover - defensive
   258	        return ("<unreadable>",), ("<unreadable>",)
   259	    try:
   260	        shape = dict(mesh.shape)
   261	        sizes = tuple(f"{n}={shape.get(n, '?')}" for n in names)
   262	    except Exception:                       # pragma: no cover - defensive
   263	        sizes = ("<unreadable>",)
   264	    return names, sizes
   265	
   266	
   267	def _agree_ocean_mesh_entry(mesh, tree=None, *, where: str) -> None:
   268	    """Agree the mesh AND a pytree LEAF SCHEMA before a scatter/gather.
   269	
   270	    #1362 round 4, blockers 5-6.  Both directions are collective here: the
   271	    gather's ``replicate_leaf`` compiles a jit identity with replicated
   272	    ``out_shardings``, and the scatter's DIRECT ``device_put`` of a full
   273	    global array onto a cross-process ``NamedSharding`` falls back to an
   274	    all-gather (documented on ``latlon_spmd.shard_leaf``) -- so the earlier
   275	    "SCATTER, therefore no collective" exemption was FALSE for this lane.
   276	    One collective runs PER LEAF, so the leaf schedule (optional fields,
   277	    dtypes, shapes) is rank-local data and is folded into one digest.
   278	    """
   279	    names, sizes = _ocean_mesh_axis_terms(mesh)
   280	    assert_flags_agree(_OCEAN_MESH_ENTRY_FLAGS, (
   281	        float(mesh is not None),
   282	        float(mesh.devices.size if mesh is not None else 0),
   283	        float(len(names)),
   284	        name_digest48(names),
   285	        name_digest48(sizes),
   286	        float(tree is not None),
   287	        tree_schema_digest48(tree) if tree is not None else FLAG_ABSENT,
   288	    ), context=where)
   289	
   290	
   291	def shard_state_latlon(state, mesh):
   292	    """Lay out a ``LatLonCGridOceanState`` for the lat-band SPMD step.
   293	
   294	    Cell / u-grid array fields (leading dim ``n_lat``) shard ``P("lat")``.  The
   295	    staggered ``v`` / ``v_mask`` (leading dim ``n_lat+1``) are carried as
   296	    ``v_lower = field[0:n_lat]`` (``n_lat`` rows, ``P("lat")``) — the top pole row
   297	    ``field[n_lat]`` is a 0 wall on the regular grid and is reconstructed in-body
   298	    by the band halo.  ``None`` fields pass through.  The inverse is
   299	    :func:`gather_state_latlon`.
   300	
   301	    This is the layout the in-``shard_map`` body of :func:`make_sharded_ocean_step`
   302	    expects; the test uses it instead of a uniform ``tree.map(P("lat"))`` (which
   303	    fails on ``v`` because ``n_lat+1`` is not divisible by ``N``).
   304	    """
   305	    # FIRST statement: a DIRECT ``device_put`` of full global arrays onto a
   306	    # cross-process ``NamedSharding`` is serviced by an ALL-GATHER, and one
   307	    # runs per leaf -- so both the mesh and the leaf SCHEDULE are rank-local
   308	    # inputs to a collective (#1362 round 4, blockers 5-6).
   309	    _agree_ocean_mesh_entry(mesh, state, where="shard_state_latlon")
   310	    # v-carrier contract (see make_sharded_ocean_step's fold note): the TOP
   311	    # v-face row (regular pole wall OR tripole seam/cap row) must be
   312	    # wall-masked — the carrier drops it and reconstructs it as zero, which
   313	    # would silently delete a LIVE seam row.  Host-side check on the
   314	    # concrete state (this fn runs outside jit).
   315	    assert_pytree_bytes_equal(state, "shard_state_latlon")
   316	    vm = getattr(state, "v_mask", None)
   317	    if vm is not None:
   318	        import numpy as _np
   319	
   320	        if _np.asarray(vm.data)[-1].any():
   321	            raise ValueError(
   322	                "shard_state_latlon: the state's TOP v-face row is LIVE "
   323	                "(v_mask[-1] has ocean faces) — the lat-band v-carrier "
   324	                "drops that row and reconstructs it as the pole/cap wall "
   325	                "zero, which would silently delete seam velocities. "
   326	                "Mask the cap row (the tripole cap convention) or extend "
   327	                "the carrier before sharding this state.")
   328	
   329	    def _shard_cell(field):
   330	        if field is None:
   331	            return None
   332	        sh = NamedSharding(mesh, _lat_spec(field.data))
   333	        return field.replace(data=addressable_shard_put(field.data, sh))
   334	
   335	    def _shard_v(field):
   336	        if field is None:
   337	            return None
   338	        # Drop the TOP v-row (the north pole wall, v==0 on the regular grid) so
   339	        # the leading dim becomes n_lat (divisible by N).  ROUND-TRIP INVARIANT
   340	        # (codex finding): shard_state_latlon -> gather_state_latlon re-appends a
   341	        # ZERO top row, so the round-trip is a bitwise identity ONLY when the
   342	        # input's top v-row is already zero (a valid masked regular-grid state;
   343	        # the pole-wall BC enforces v[n_lat]=0 every step).  An arbitrary nonzero
   344	        # top row (e.g. a raw IC perturbation) is dropped -> reconstructed as 0:
   345	        # CORRECT for the dynamics (the pole wall zeros it at step 1) but not a
   346	        # bit round-trip of that one row.  NOT for a tripole north fold (raises
   347	        # in make_sharded_ocean_step) where the top row is a live fold partner.
   348	        nlat1 = field.data.shape[0]
   349	        v_lower = field.data[:nlat1 - 1]           # drop the top pole-wall row
   350	        sh = NamedSharding(mesh, _lat_spec(v_lower))
   351	        return field.replace(data=addressable_shard_put(v_lower, sh))
   352	
   353	    updates = {}
   354	    for name in _V_STAGGERED_STATE_FIELDS:
   355	        updates[name] = _shard_v(getattr(state, name))
   356	    # Every other (array) leaf: shard P("lat").  NamedTuple fields not in the
   357	    # v-staggered set and not None get the cell sharding; None stays None.
   358	    for name in state._fields:
   359	        if name in _V_STAGGERED_STATE_FIELDS:
   360	            continue
   361	        val = getattr(state, name)
   362	        if val is None:
   363	            updates[name] = None
   364	        elif hasattr(val, "data"):                 # a Field
   365	            updates[name] = _shard_cell(val)
   366	        else:
   367	            # Non-Field, non-None leaf (e.g. a raw array carry like psi).
   368	            # Shard 2-D+ on lat, replicate lower-rank — matches shard_pytree.
   369	            arr = jnp.asarray(val)
   370	            spec = _lat_spec(arr) if arr.ndim >= 1 else P()
   371	            # ndim>=1 lat-shards via _lat_spec (1-D included); only true
   372	            # scalars replicate.
   373	            updates[name] = addressable_shard_put(arr, NamedSharding(mesh, spec))
   374	    return state._replace(**updates)
   375	
   376	
   377	def shard_forcing_latlon(forcing, mesh):
   378	    """Lay out a forcing pytree (``FreshwaterForcing`` / ``OceanSurfaceForcing``
   379	    / ``SpongeForcing`` — or any nesting of them) for the lat-band SPMD step.
   380	
   381	    Every OMIP forcing field is CELL-CENTERED ``(n_lat, n_lon[, nlev])`` (the
   382	    cell->face wind-stress interpolation happens INSIDE the step through the
   383	    SPMD-aware halo pads), so array leaves shard ``P("lat", None, ...)`` with
   384	    no ``v_lower`` handling; scalars replicate; ``None`` fields pass through
   385	    untouched (they vanish from the pytree structure, matching the specs the
   386	    step derives).  ``forcing=None`` returns ``None``.
   387	    """
   388	    # FIRST statement: the `forcing is None or mesh is None` return below
   389	    # SKIPS every per-leaf put, so a rank with no forcing would leave a peer
   390	    # blocked in one (#1362 round 4, blocker 6).
   391	    _agree_ocean_mesh_entry(mesh, forcing, where="shard_forcing_latlon")
   392	    if forcing is None or mesh is None:
   393	        return forcing
   394	    assert_pytree_bytes_equal(forcing, "shard_forcing_latlon")
   395	
   396	    def _put(leaf):
   397	        if leaf is None:
   398	            return None
   399	        arr = jnp.asarray(leaf)
   400	        return addressable_shard_put(arr, NamedSharding(mesh, _lat_spec(arr)))
   401	
   402	    return jax.tree.map(_put, forcing)
   403	
   404	
   405	def shard_forcing_stack_latlon(stack, mesh):
   406	    """Lay out a STACKED per-block forcing pytree for the lat-band SPMD
   407	    block-scan (the ``run_omip`` JRA55 lanes; see
   408	    ``_build_jra55_block_fn`` / ``_build_jra55_block_fn_interp``).
   409	
   410	    Unlike :func:`shard_forcing_latlon` (per-step, lat at axis 0), the
   411	    block builders stack ``N`` steps / raw records along a LEADING axis,
   412	    so the lat axis sits at position 1.  A ``(n_rec, n_lat, n_lon[, ...])``
   413	    leaf therefore shards ``P(None, "lat", ...)`` (records replicated, the
   414	    time index stays shard-local so the in-scan interpolation needs no
   415	    cross-band comm); a bare ``(n_lat, n_lon)`` leaf shards ``P("lat", None)``;
   416	    1-D metadata / scalars replicate; ``None`` and non-array leaves pass
   417	    through.  ``mesh=None`` returns ``stack`` unchanged (serial lane).
   418	
   419	    CONTRACT: rank is the ONLY signal used, so a rank-2 leaf is assumed to
   420	    be a ``(n_lat, n_lon)`` field and is lat-sharded on axis 0.  Any future
   421	    metadata that is genuinely rank-2 but NOT lat-major (e.g. a
   422	    ``(n_rec, n_meta)`` table) would be silently mis-sharded — keep such
   423	    metadata 1-D (or replicate it explicitly) before it reaches this helper.
   424	
   425	    Keeping this next to :func:`shard_state_latlon` means the driver and
   426	    the parity tests share ONE layout definition — the block-scan forcing
   427	    stack must be laid out consistently with the state the sharded step
   428	    carries, and a second copy would drift.
   429	    """
   430	    # FIRST statement: same rank-local early-return + per-leaf put as
   431	    # shard_forcing_latlon (#1362 round 4, blocker 6).
   432	    _agree_ocean_mesh_entry(mesh, stack,
   433	                            where="shard_forcing_stack_latlon")
   434	    if mesh is None:
   435	        return stack
   436	
   437	    assert_pytree_bytes_equal(stack, "shard_forcing_stack_latlon")
   438	
   439	    def _put(leaf):
   440	        if leaf is None or not hasattr(leaf, "ndim"):
   441	            return leaf
   442	        arr = jnp.asarray(leaf)
   443	        if arr.ndim >= 3:
   444	            spec = P(None, "lat", *((None,) * (arr.ndim - 2)))
   445	        elif arr.ndim == 2:
   446	            spec = P("lat", None)
   447	        else:
   448	            spec = P()
   449	        return addressable_shard_put(arr, NamedSharding(mesh, spec))
   450	
   530	    "grid_n_lat", "grid_n_lon", "geom_schema", "fold_active",
   531	    "config_digest", "has_vertex_mask",
   532	)
   533	
   534	
   535	def _agree_ocean_spmd_entry(model, mesh, *, where: str) -> None:
   536	    """Agree every rank-local input, as the FIRST statement of a public factory.
   537	
   538	    #1362 / codex round 2, ocean twin of the atmosphere's
   539	    ``_agree_spmd_entry``.  This lane has the same shape of hazard: the
   540	    ``mesh is None`` early return, ``build_band_grids``' divisibility
   541	    validation, and ``_build_band_vertex_masks`` (which throws if only THIS
   542	    rank lacks a primed vertex-mask cache) all execute BEFORE the schema
   543	    collective.  Any of them lets one process raise or return while a peer
   544	    blocks in ``process_allgather`` -- a HANG rather than an error.
   545	
   546	    Agreeing the mesh shape, grid dimensions and fold state up front makes
   547	    every downstream rank-local check symmetric by construction.
   548	
   549	    ``axis_names`` carries the ORDERED axis-name digest, not just the axis
   550	    COUNT: the band body indexes ``mesh.axis_names[0]``, so two processes
   551	    whose meshes name that axis differently would psum/ppermute over
   552	    different axes while every count-based flag agreed (the ocean instance of
   553	    codex round-3 blocker 2, which was found on the atmosphere twin --
   554	    fixing only the lane where a defect was reported is what left five
   555	    unguarded paths after round 1).
   556	
   557	    Grid dimensions go through :func:`coerce_count`, which NEVER raises, and
   558	    the refusal is deferred until AFTER the collective; building a collective
   559	    payload must not be able to kill one rank while its peers block in the
   560	    gather (codex round-3 blocker 3, same rationale as the atm twin).
   561	    """
   562	    # Defensive attribute reads: nothing in the payload build may raise before
   563	    # the collective (see the atm twin for the full rule).
   564	    grid = getattr(model, "grid", None)
   565	    fold = getattr(grid, "fold", None)
   566	    names, sizes = _ocean_mesh_axis_terms(mesh)
   567	    problems = []
   568	
   569	    def _count(value, label, absent=FLAG_ABSENT):
   570	        payload, problem = coerce_count(value, absent=absent)
   571	        if problem is not None:
   572	            problems.append((label, problem))
   573	        return payload
   574	
   575	    flags = (
   576	        float(mesh is not None),
   577	        float(mesh.devices.size if mesh is not None else 0),
   578	        float(len(names)),
   579	        name_digest48(names),
   580	        name_digest48(sizes),
   581	        _count(getattr(grid, "n_lat", None), "grid.n_lat", absent=0.0),
   582	        _count(getattr(grid, "n_lon", None), "grid.n_lon", absent=0.0),
   583	        # The geometry array fields that `build_band_grids` slices and
   584	        # `_replicated_put` broadcasts, by dtype + shape.
   585	        tree_schema_digest48(grid),
   586	        float(bool(fold is not None and getattr(fold, "is_active", False))),
   587	        # ONE digest over EVERY static scalar of the ocean config instead of a
   588	        # hand-picked few: the step body branches on `outer_integrator`, the
   589	        # tracer integrator, the polar filter, the freeze floor and the EW
   590	        # overlap, and NONE of them were agreed (codex round-4, blocker 3).
   591	        # A valid/invalid or euler/ab2 split makes one rank raise during
   592	        # tracing while its peer compiles a different program.
   593	        config_digest48(getattr(model, "config", None)),
   594	        # `_build_band_vertex_masks` RAISES when this cache is unprimed, and
   595	        # it runs before the schema collective -- so its presence must be
   596	        # agreed first or an unprimed rank dies while its peer blocks
   597	        # (codex round-4, blocker 4).
   598	        float(getattr(model, "_vertex_mask", None) is not None),
   599	    )
   600	    assert_flags_agree(_OCEAN_SPMD_ENTRY_FLAGS, flags, context=where)
   601	    # AFTER the collective only: symmetric on every rank (see atm twin).
   602	    for label, problem in problems:
   603	        raise ValueError(f"{where}: {label} {problem}")
   604	
   605	
   606	# Ordered flag names for the PER-INVOCATION gate on the returned ocean SPMD
   607	# callable. STATIC tuple: fixed width, never rank-local.
   608	_OCEAN_CALL_ENTRY_FLAGS = (
   609	    "has_mesh", "n_dev", "axis_names", "axis_sizes",
   610	    "state_schema", "has_forcing", "forcing_schema",
   611	)
   612	
   613	
   614	def _agree_ocean_spmd_call(mesh, state, forcing, *, where: str) -> None:
   615	    """Agree a returned ocean SPMD callable's per-CALL inputs, FIRST statement.
   616	
   617	    #1362 round 4, blocker 7.  ``sharded_step`` runs ``_validate_forcing_layout``
   618	    and builds a rank-local cache key BEFORE entering its ``shard_map``: a
   619	    forcing layout that is invalid on one rank only makes that rank raise while
   620	    its peers enter the collective program -- a hang.  The state + forcing leaf
   621	    SCHEMA is agreed too, because ``in_specs``/``out_specs`` are derived from
   622	    it, so two processes with different optional fields compile different
   623	    programs.
   624	
   625	    Cost: one small allgather per CALL, and an exact no-op under a single
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "legoesm-core"
version = "0.1.0"
description = "legoESM core substrate: grids, operators, state, runtime, parallel, io, time integration, and component scaffolding (the shared base every Earth-system component depends on)."
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
authors = [{ name = "Pierre Gentine" }]
keywords = ["earth-system-model", "jax", "differentiable", "weather", "climate"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
]
# The substrate's hard runtime deps.  It imports NOTHING above it (import-linter
# contract "legoesm-core member imports nothing above it"), so it has no
# inter-member dependency — this is the root of the federation DAG.
dependencies = [
    "jax>=0.4.35",
    "jaxlib>=0.4.35",
    "equinox>=0.11",
    "numpy>=1.26",
    "scipy>=1.10",
    "xarray>=2024.0",
    "zarr>=2.18",
    "netCDF4>=1.6",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
mesh = ["pymetis>=1.2", "scipy>=1.10"]
# Tested jax envelope for MPI is >=0.8,<0.10 (runtime-guarded in
# parallel/reductions.py). No jax cap here — see root pyproject [mpi] comment;
# pinned working combo lives in requirements_mpi.txt.
mpi = ["mpi4py>=4.1,<5", "mpi4jax>=0.8,<0.10"]

# Namespace package: NO legoesm/__init__.py, so this wheel ships
# legoesm/{core,grids,runtime,parallel,io,timestepping,components} + the
# substrate top-level modules (constants, thermo, registry, _version) as a
# PEP-420 portion of the shared ``legoesm`` namespace.  Sibling members
# (legoesm-atmosphere, ...) contribute their own subpackages to the same
# namespace, so everything still imports as ``legoesm.<subpkg>``.
[tool.hatch.build.targets.wheel]
packages = ["legoesm"]
packages/core/legoesm/core/state.py:184:#   from legoesm.atmosphere.physics.physics_state import PhysicsState
packages/ocean/legoesm/ocean/__init__.py
packages/core/legoesm/parallel/__init__.py

exec
/usr/bin/bash -lc "rg -n -C 3 'jax\\.device_put|make_array_from_callback|broadcast_one_to_all|process_allgather' packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py packages/core/legoesm/parallel/geometry_consistency.py packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py packages/coupler/legoesm/driver/sharded_operator_split_step.py; rg -n -C 4 'aux\\s*=|\\.aux|make_sharded_ocean_step\\(' packages tests scripts --glob '*.py' | sed -n '1,500p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 674ms:
packages/coupler/legoesm/driver/sharded_operator_split_step.py-176-    vl = _carry_valid_leads(carry)
packages/coupler/legoesm/driver/sharded_operator_split_step.py-177-    return SegmentCarry(**{
packages/coupler/legoesm/driver/sharded_operator_split_step.py-178-        name: (None if (v := getattr(carry, name)) is None
packages/coupler/legoesm/driver/sharded_operator_split_step.py:179:               else jax.device_put(v, NamedSharding(
packages/coupler/legoesm/driver/sharded_operator_split_step.py-180-                   mesh, _carry_leaf_spec(name, v, n_dev, vl))))
packages/coupler/legoesm/driver/sharded_operator_split_step.py-181-        for name in SegmentCarry._fields
packages/coupler/legoesm/driver/sharded_operator_split_step.py-182-    })
--
packages/coupler/legoesm/driver/sharded_operator_split_step.py-193-    n_dev = mesh.devices.size
packages/coupler/legoesm/driver/sharded_operator_split_step.py-194-    return SegmentForcing(**{
packages/coupler/legoesm/driver/sharded_operator_split_step.py-195-        name: (None if (v := getattr(forcing, name)) is None
packages/coupler/legoesm/driver/sharded_operator_split_step.py:196:               else jax.device_put(v, NamedSharding(
packages/coupler/legoesm/driver/sharded_operator_split_step.py-197-                   mesh, _forcing_leaf_spec(name, v, n_dev))))
packages/coupler/legoesm/driver/sharded_operator_split_step.py-198-        for name in SegmentForcing._fields
packages/coupler/legoesm/driver/sharded_operator_split_step.py-199-    })
--
packages/coupler/legoesm/driver/sharded_operator_split_step.py-316-    assert_flags_agree(_OPSPLIT_SPMD_ENTRY_FLAGS, flags, context=where)
packages/coupler/legoesm/driver/sharded_operator_split_step.py-317-    # AFTER the collective only, so the refusal is symmetric on every rank
packages/coupler/legoesm/driver/sharded_operator_split_step.py-318-    # (see the atm twin's rationale: a raise while assembling the payload
packages/coupler/legoesm/driver/sharded_operator_split_step.py:319:    # kills one process while its peers block in process_allgather).
packages/coupler/legoesm/driver/sharded_operator_split_step.py-320-    for label, problem in problems:
packages/coupler/legoesm/driver/sharded_operator_split_step.py-321-        raise ValueError(f"{where}: {label} {problem}")
packages/coupler/legoesm/driver/sharded_operator_split_step.py-322-    if (getattr(model, "_polar_mask", None) is not None
--
packages/coupler/legoesm/driver/sharded_operator_split_step.py-479-                         context="make_sharded_operator_split_step",
packages/coupler/legoesm/driver/sharded_operator_split_step.py-480-                         arrays=[raw[n] for n in _ordered])
packages/coupler/legoesm/driver/sharded_operator_split_step.py-481-    stacks = {
packages/coupler/legoesm/driver/sharded_operator_split_step.py:482:        name: jax.device_put(
packages/coupler/legoesm/driver/sharded_operator_split_step.py-483-            jnp.asarray(broadcast_checked(
packages/coupler/legoesm/driver/sharded_operator_split_step.py-484-                raw[name], name,
packages/coupler/legoesm/driver/sharded_operator_split_step.py-485-                context="make_sharded_operator_split_step")),
--
packages/core/legoesm/parallel/geometry_consistency.py-67-# An entry gate turns rank-local scalars (n_steps, segment_steps, grid dims)
packages/core/legoesm/parallel/geometry_consistency.py-68-# into a fixed-width float payload.  Building that payload must NEVER raise:
packages/core/legoesm/parallel/geometry_consistency.py-69-# a rank that dies in `int(n_steps)` while its peers block in
packages/core/legoesm/parallel/geometry_consistency.py:70:# `process_allgather` is a HANG, which is strictly worse than the bug the gate
packages/core/legoesm/parallel/geometry_consistency.py-71-# exists to fix (codex 2026-07-29 round-3, blocker 3).  So an unusable value is
packages/core/legoesm/parallel/geometry_consistency.py-72-# mapped to a SENTINEL that travels through the collective; every rank then
packages/core/legoesm/parallel/geometry_consistency.py-73-# sees it in the gathered payload and the raise that follows is symmetric.
--
packages/core/legoesm/parallel/geometry_consistency.py-321-
packages/core/legoesm/parallel/geometry_consistency.py-322-# Every per-field collective payload is padded to these FIXED widths.  A
packages/core/legoesm/parallel/geometry_consistency.py-323-# payload whose LENGTH depends on rank-local data (dtype class, ndim,
packages/core/legoesm/parallel/geometry_consistency.py:324:# non-finite count) would let two processes enter `process_allgather` with
packages/core/legoesm/parallel/geometry_consistency.py-325-# different shapes and DEADLOCK -- the exact failure this module exists to
packages/core/legoesm/parallel/geometry_consistency.py-326-# turn into a clean symmetric raise (codex 2026-07-29, blocker 2; the flaw was
packages/core/legoesm/parallel/geometry_consistency.py-327-# inherited from the pre-extraction ocean implementation, so fixing it here
--
packages/core/legoesm/parallel/geometry_consistency.py-341-    Used to compare EXACT-dtype arrays (masks, index tables) across
packages/core/legoesm/parallel/geometry_consistency.py-342-    processes: unlike moment fingerprints, a byte digest is positional, so a
packages/core/legoesm/parallel/geometry_consistency.py-343-    permutation or a two-cell flip cannot cancel. 48 bits keeps the value
packages/core/legoesm/parallel/geometry_consistency.py:344:    under 2**53 so it survives the float64 ``process_allgather`` payload
packages/core/legoesm/parallel/geometry_consistency.py-345-    exactly. Not cryptographic — collision-resistance at 2**-48 is far
packages/core/legoesm/parallel/geometry_consistency.py-346-    beyond the ~10 setup-time comparisons this guard makes.
packages/core/legoesm/parallel/geometry_consistency.py-347-    """
--
packages/core/legoesm/parallel/geometry_consistency.py-394-def in_jax_trace() -> bool:
packages/core/legoesm/parallel/geometry_consistency.py-395-    """True when the caller runs inside a JAX trace (``jit``/``scan``/``vmap``).
packages/core/legoesm/parallel/geometry_consistency.py-396-
packages/core/legoesm/parallel/geometry_consistency.py:397:    The host-side gates below call ``multihost_utils.process_allgather``, which
packages/core/legoesm/parallel/geometry_consistency.py-398-    is an EAGER utility: it ``device_put``s its payload per addressable device.
packages/core/legoesm/parallel/geometry_consistency.py-399-    Under an active trace those puts are staged into the jaxpr and come back as
packages/core/legoesm/parallel/geometry_consistency.py-400-    tracers, so ``make_array_from_single_device_arrays`` is handed tracers and
--
packages/core/legoesm/parallel/geometry_consistency.py-409-       in the SAME transform state, which is the SPMD lockstep property the
packages/core/legoesm/parallel/geometry_consistency.py-410-       gate itself exists to enforce.  If one rank called the step eagerly
packages/core/legoesm/parallel/geometry_consistency.py-411-       while another traced it, the eager rank would now BLOCK in
packages/core/legoesm/parallel/geometry_consistency.py:412:       ``process_allgather`` instead of its peer crashing.  That divergence is
packages/core/legoesm/parallel/geometry_consistency.py-413-       already fatal today (the traced rank dies here), so this trades a
packages/core/legoesm/parallel/geometry_consistency.py-414-       guaranteed crash on every multi-process traced run for a hang in an
packages/core/legoesm/parallel/geometry_consistency.py-415-       already-divergent one.  It is NOT a proof of symmetry.
--
packages/core/legoesm/parallel/geometry_consistency.py-459-    payload = np.array(
packages/core/legoesm/parallel/geometry_consistency.py-460-        [float(len(values)), name_digest48(names),
packages/core/legoesm/parallel/geometry_consistency.py-461-         *(float(v) for v in values)], dtype=np.float64)
packages/core/legoesm/parallel/geometry_consistency.py:462:    gathered = multihost_utils.process_allgather(payload)
packages/core/legoesm/parallel/geometry_consistency.py-463-    if not bool(np.all(gathered == gathered[0])):
packages/core/legoesm/parallel/geometry_consistency.py-464-        raise RuntimeError(
packages/core/legoesm/parallel/geometry_consistency.py-465-            f"{context}: per-process CONFIG differs across processes "
--
packages/core/legoesm/parallel/geometry_consistency.py-508-        described = [_dtype_kind_and_ndim(a) for a in arrays]
packages/core/legoesm/parallel/geometry_consistency.py-509-        kinds = [d[0] for d in described]
packages/core/legoesm/parallel/geometry_consistency.py-510-        ndims = [d[1] for d in described]
packages/core/legoesm/parallel/geometry_consistency.py:511:    gathered = multihost_utils.process_allgather(
packages/core/legoesm/parallel/geometry_consistency.py-512-        schema_fingerprint(names, n_dev, kinds, ndims))
packages/core/legoesm/parallel/geometry_consistency.py-513-    if not bool(np.all(gathered == gathered[0])):
packages/core/legoesm/parallel/geometry_consistency.py-514-        raise RuntimeError(
--
packages/core/legoesm/parallel/geometry_consistency.py-580-        vals[1] = float((f64 * f64).sum()) if f64.size else 0.0
packages/core/legoesm/parallel/geometry_consistency.py-581-        vals[2] = float(np.abs(f64).max()) if f64.size else 0.0
packages/core/legoesm/parallel/geometry_consistency.py-582-
packages/core/legoesm/parallel/geometry_consistency.py:583:    g_struct = multihost_utils.process_allgather(struct)
packages/core/legoesm/parallel/geometry_consistency.py:584:    g_vals = multihost_utils.process_allgather(vals)
packages/core/legoesm/parallel/geometry_consistency.py-585-    struct_ok = bool(np.all(g_struct == g_struct[0]))
packages/core/legoesm/parallel/geometry_consistency.py-586-    if is_exact:
packages/core/legoesm/parallel/geometry_consistency.py-587-        vals_ok = bool(np.all(g_vals == g_vals[0]))
--
packages/core/legoesm/parallel/geometry_consistency.py-595-            f"exact_dtype={is_exact}, gathered={g_vals.tolist()}) — a real "
packages/core/legoesm/parallel/geometry_consistency.py-596-            f"config/grid inconsistency, not autotune noise; refusing to "
packages/core/legoesm/parallel/geometry_consistency.py-597-            f"broadcast process 0 over it.")
packages/core/legoesm/parallel/geometry_consistency.py:598:    return np.asarray(multihost_utils.broadcast_one_to_all(host))
packages/core/legoesm/parallel/geometry_consistency.py-599-
packages/core/legoesm/parallel/geometry_consistency.py-600-
packages/core/legoesm/parallel/geometry_consistency.py-601-# --- assert-free sharded puts + per-band gates (2026-08-03, ocean walls) ----
packages/core/legoesm/parallel/geometry_consistency.py-602-# Three stacked multicontroller walls were found on the ocean lane (codex
packages/core/legoesm/parallel/geometry_consistency.py:603:# r14-r19; PR #1457): (1) broadcast_one_to_all of a band stack lowers to an
packages/core/legoesm/parallel/geometry_consistency.py-604-# [n_processes, stack] psum program (nd x 849 MB at LL2304 L20 — 81.5 GB at
packages/core/legoesm/parallel/geometry_consistency.py:605:# 96 procs); (2) jax.device_put of a NUMPY array onto an all-process
packages/core/legoesm/parallel/geometry_consistency.py-606-# sharding internally runs multihost_utils.assert_equal on the FULL array
packages/core/legoesm/parallel/geometry_consistency.py-607-# ([n_proc, field] landing on ONE device: fits under an 80 GB A100 up to
packages/core/legoesm/parallel/geometry_consistency.py-608-# ~64 procs, dies at 96 — jax _src/dispatch.py::_device_put_sharding_impl);
--
packages/core/legoesm/parallel/geometry_consistency.py-672-    """Gate a band-stacked field per band, then put WITHOUT broadcast or
packages/core/legoesm/parallel/geometry_consistency.py-673-    jax's whole-array device_put assert (walls 1+2 above).
packages/core/legoesm/parallel/geometry_consistency.py-674-
packages/core/legoesm/parallel/geometry_consistency.py:675:    Single-process: plain ``jax.device_put`` — byte-unchanged, no host
packages/core/legoesm/parallel/geometry_consistency.py-676-    round trip. Multi-process: per-band fingerprint gate (symmetric raise
packages/core/legoesm/parallel/geometry_consistency.py:677:    on real divergence), then ``jax.make_array_from_callback`` hands each
packages/core/legoesm/parallel/geometry_consistency.py-678-    process exactly its addressable slabs. Cross-process byte-identity of
packages/core/legoesm/parallel/geometry_consistency.py-679-    NON-owned bands is not required — owned bands are the only bytes that
packages/core/legoesm/parallel/geometry_consistency.py-680-    reach any device, and their drift is bounded by the gate.
packages/core/legoesm/parallel/geometry_consistency.py-681-    """
packages/core/legoesm/parallel/geometry_consistency.py-682-    if jax.process_count() <= 1:
packages/core/legoesm/parallel/geometry_consistency.py:683:        return jax.device_put(arr, sharding)
packages/core/legoesm/parallel/geometry_consistency.py-684-    from jax.experimental import multihost_utils
packages/core/legoesm/parallel/geometry_consistency.py-685-
packages/core/legoesm/parallel/geometry_consistency.py-686-    host = np.asarray(arr)
packages/core/legoesm/parallel/geometry_consistency.py-687-    struct, vals, is_exact = band_fingerprint(host, n_bands)
packages/core/legoesm/parallel/geometry_consistency.py:688:    g_struct = multihost_utils.process_allgather(struct)
packages/core/legoesm/parallel/geometry_consistency.py:689:    g_vals = multihost_utils.process_allgather(vals)
packages/core/legoesm/parallel/geometry_consistency.py-690-    if not band_fingerprints_agree(g_struct, g_vals, is_exact):
packages/core/legoesm/parallel/geometry_consistency.py-691-        raise RuntimeError(
packages/core/legoesm/parallel/geometry_consistency.py-692-            f"{context}: band-stacked field {name!r} DIVERGES across "
packages/core/legoesm/parallel/geometry_consistency.py-693-            f"processes (exact_dtype={is_exact}, "
packages/core/legoesm/parallel/geometry_consistency.py-694-            f"gathered={g_vals.tolist()}) — a real config/grid "
packages/core/legoesm/parallel/geometry_consistency.py-695-            f"inconsistency, not autotune noise; refusing to shard it.")
packages/core/legoesm/parallel/geometry_consistency.py:696:    return jax.make_array_from_callback(
packages/core/legoesm/parallel/geometry_consistency.py-697-        host.shape, sharding, lambda idx: host[idx])
packages/core/legoesm/parallel/geometry_consistency.py-698-
packages/core/legoesm/parallel/geometry_consistency.py-699-
--
packages/core/legoesm/parallel/geometry_consistency.py-711-              if hasattr(x, "ndim")]
packages/core/legoesm/parallel/geometry_consistency.py-712-    vals = np.array([content_hash48(np.asarray(x)) for x in leaves],
packages/core/legoesm/parallel/geometry_consistency.py-713-                    dtype=np.float64)
packages/core/legoesm/parallel/geometry_consistency.py:714:    g = multihost_utils.process_allgather(vals)
packages/core/legoesm/parallel/geometry_consistency.py-715-    if not bool(np.all(g == g[0])):
packages/core/legoesm/parallel/geometry_consistency.py-716-        bad = [i for i in range(len(leaves))
packages/core/legoesm/parallel/geometry_consistency.py-717-               if not bool(np.all(g[:, i] == g[0, i]))]
--
packages/core/legoesm/parallel/geometry_consistency.py-727-    consistency the CALLER has already gated (state/forcing pytrees via
packages/core/legoesm/parallel/geometry_consistency.py-728-    :func:`assert_pytree_bytes_equal`). Single-process: plain device_put."""
packages/core/legoesm/parallel/geometry_consistency.py-729-    if jax.process_count() <= 1:
packages/core/legoesm/parallel/geometry_consistency.py:730:        return jax.device_put(arr, sharding)
packages/core/legoesm/parallel/geometry_consistency.py-731-    host = np.asarray(arr)
packages/core/legoesm/parallel/geometry_consistency.py:732:    return jax.make_array_from_callback(
packages/core/legoesm/parallel/geometry_consistency.py-733-        host.shape, sharding, lambda idx: host[idx])
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-164-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-165-    The #1100 invariant for multi-process runs: **neither global builds nor
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-166-    ``device_put`` replication** — every global-shaped leaf is created with
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:167:    ``jax.make_array_from_callback``, whose callback is invoked only for the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-168-    row slices owned by THIS process's addressable devices (documented JAX
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-169-    semantics: per-addressable-shard callbacks with GLOBAL index slices).  No
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-170-    full global array is ever handed to ``device_put`` — the path measured to
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-216-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-217-    def _make(gshape, cb):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-218-        sharding = NamedSharding(mesh, P("lat", *((None,) * (len(gshape) - 1))))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:219:        return jax.make_array_from_callback(gshape, sharding, cb)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-220-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-221-    def _zeros_cb(gshape):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-222-        return lambda idx: jnp.zeros(_slice_shape(gshape, idx), dtype=_dtype)
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-495-    }
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-496-    spec_of = lat_spec if shard_geometry else (lambda _arr: P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-497-    stacks = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:498:        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-499-        for name, arr in raw.items()
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-500-    }
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-501-    stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-622-    collective -- ``n_steps`` validation, ``_check_2d_mesh``, the ``mesh is
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-623-    None`` early return, and ``build_band_grids_atm``'s divisibility check.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-624-    Any one of them lets a process raise (or return a serial closure) while a
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:625:    peer walks into ``process_allgather`` and blocks forever. Patching them
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-626-    one at a time is whack-a-mole; the invariant has to be established ONCE,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-627-    before anything can diverge.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-628-
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1439-    _agree_mesh_entry(mesh, state, where="shard_state_atm_latlon_2d")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1440-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1441-    def _put(arr):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1442:        return jax.device_put(arr, NamedSharding(mesh, tile_spec(arr)))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1443-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1444-    n_lat, n_lon = state.T.shape[0], state.T.shape[1]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1445-    return state._replace(
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1589-    }
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1590-    spec_of = tile_spec if shard_geometry else (lambda _arr: P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1591-    stacks = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1592:        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1593-        for name, arr in raw.items()
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1594-    }
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1595-    stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-541-    validation, and ``_build_band_vertex_masks`` (which throws if only THIS
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-542-    rank lacks a primed vertex-mask cache) all execute BEFORE the schema
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-543-    collective.  Any of them lets one process raise or return while a peer
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:544:    blocks in ``process_allgather`` -- a HANG rather than an error.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-545-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-546-    Agreeing the mesh shape, grid dimensions and fold state up front makes
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-547-    every downstream rank-local check symmetric by construction.
tests/unit/test_run_omip_latlon_spmd.py-78-    assert ok_serial
tests/unit/test_run_omip_latlon_spmd.py-79-
tests/unit/test_run_omip_latlon_spmd.py-80-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py-81-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py:82:    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-83-    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-84-    ckpt_dir = tmp_path / "spmd"
tests/unit/test_run_omip_latlon_spmd.py-85-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-86-        model, ss0, checkpoint_dir=ckpt_dir,
--
tests/unit/test_run_omip_latlon_spmd.py-215-        model, state0, jra55_state=_fresh_js(), **common)
tests/unit/test_run_omip_latlon_spmd.py-216-    assert ok_serial, "serial JRA55 loop reported not-ok"
tests/unit/test_run_omip_latlon_spmd.py-217-
tests/unit/test_run_omip_latlon_spmd.py-218-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py:219:    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-220-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py-221-    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-222-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-223-        model, ss0, jra55_state=_fresh_js(),
--
tests/coupler/unit/test_coupled_lane_carry_aux.py-55-
tests/coupler/unit/test_coupled_lane_carry_aux.py-56-def test_missing_seg_precip_alone_raises():
tests/coupler/unit/test_coupled_lane_carry_aux.py-57-    """The spectral lane's signature: sw/lw stashed at model_driver.py:6511-
tests/coupler/unit/test_coupled_lane_carry_aux.py-58-    6512, seg_precip never written."""
tests/coupler/unit/test_coupled_lane_carry_aux.py:59:    aux = {"held_sw_net_sfc": 1.0, "held_lw_net_sfc": 2.0}
tests/coupler/unit/test_coupled_lane_carry_aux.py-60-    with pytest.raises(RuntimeError, match="seg_precip"):
tests/coupler/unit/test_coupled_lane_carry_aux.py-61-        require_surface_radiation_aux(
tests/coupler/unit/test_coupled_lane_carry_aux.py-62-            aux, radiation_active=True, precip_active=True, lane="spectral")
tests/coupler/unit/test_coupled_lane_carry_aux.py-63-
--
tests/coupler/unit/test_coupled_lane_carry_aux.py-65-def test_present_but_none_value_is_treated_as_missing():
tests/coupler/unit/test_coupled_lane_carry_aux.py-66-    """_run_per_step writes the key UNCONDITIONALLY with a None value when its
tests/coupler/unit/test_coupled_lane_carry_aux.py-67-    source is inactive (model_driver.py:9762).  A key-presence test would pass
tests/coupler/unit/test_coupled_lane_carry_aux.py-68-    that through and the consumer would then do `None / jnp.maximum(...)`."""
tests/coupler/unit/test_coupled_lane_carry_aux.py:69:    aux = {"held_sw_net_sfc": None, "held_lw_net_sfc": 2.0,
tests/coupler/unit/test_coupled_lane_carry_aux.py-70-           "seg_precip": 3.0}
tests/coupler/unit/test_coupled_lane_carry_aux.py-71-    with pytest.raises(RuntimeError, match="held_sw_net_sfc"):
tests/coupler/unit/test_coupled_lane_carry_aux.py-72-        require_surface_radiation_aux(
tests/coupler/unit/test_coupled_lane_carry_aux.py-73-            aux, radiation_active=True, precip_active=True, lane="per_step")
--
tests/coupler/unit/test_coupled_lane_carry_aux.py-83-
tests/coupler/unit/test_coupled_lane_carry_aux.py-84-
tests/coupler/unit/test_coupled_lane_carry_aux.py-85-def test_dry_run_with_radiation_requires_only_fluxes():
tests/coupler/unit/test_coupled_lane_carry_aux.py-86-    """radiation on + dry: fluxes demanded, precip correctly not demanded."""
tests/coupler/unit/test_coupled_lane_carry_aux.py:87:    aux = {"held_sw_net_sfc": 1.0, "held_lw_net_sfc": 2.0}
tests/coupler/unit/test_coupled_lane_carry_aux.py-88-    require_surface_radiation_aux(
tests/coupler/unit/test_coupled_lane_carry_aux.py-89-        aux, radiation_active=True, precip_active=False, lane="mpas-dry")
tests/coupler/unit/test_coupled_lane_carry_aux.py-90-
tests/coupler/unit/test_coupled_lane_carry_aux.py-91-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-638-         else FLAG_ABSENT),
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-639-    ), context=where)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-640-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-641-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:642:def make_sharded_ocean_step(model, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-643-    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-644-    sponge=None, t_seconds=None) -> state`` running ``model.step``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-645-    lat-band-SPMD.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-646-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-869-                    f"({n_lat_global}); forcing must be cell-centered "
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-870-                    f"(n_lat, n_lon[, nlev]) to shard on the lat axis.")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-871-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-872-    def sharded_step(state, dt, freshwater=None, surface_forcing=None,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:873:                     sponge=None, t_seconds=None, aux=None):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-874-        # ``aux``: the sharded geometry+vmask stacks. When this wrapper runs
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-875-        # INSIDE an outer trace (a bench/driver jit/scan — jit-of-jit
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-876-        # inlines the inner call), concrete closure arrays become
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-877-        # OUTER-trace constants whose value the MLIR handler cannot fetch
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-878-        # for non-addressable arrays (broken since #1370-iii sharded the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:879:        # stacks). Outer-jit callers MUST thread ``step.aux`` through their
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-880-        # jit boundary as an ARGUMENT and pass it back here.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-881-        # ONE forcing operand: None fields drop out of the pytree structure,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-882-        # so specs derived by tree.map skip them automatically and the
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-883-        # structure key below distinguishes every None<->array combination.
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-958-            set_halo_backend(_prev_backend, _prev_topo)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-959-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-960-    # Expose the stacks so outer-jit callers can pass them as arguments
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-961-    # (see the ``aux`` note in the signature).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:962:    sharded_step.aux = (geom_stacks, vmask_stack)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-963-    return sharded_step
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-964-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-965-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-966-def make_sharded_ocean_step_global(model, mesh):
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-986-        return lambda state, dt, surface_forcing=None, freshwater=None: (
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-987-            model.step(state, dt, freshwater=freshwater,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-988-                       surface_forcing=surface_forcing))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-989-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:990:    inner = make_sharded_ocean_step(model, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-991-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-992-    def sharded_step_global(state, dt, surface_forcing=None, freshwater=None):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-993-        _agree_ocean_spmd_call(mesh, state, (surface_forcing, freshwater),
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-994-                               where="make_sharded_ocean_step_global.step")
--
tests/unit/test_field.py-32-
tests/unit/test_field.py-33-    def test_pytree_flatten_unflatten(self):
tests/unit/test_field.py-34-        data = jnp.array([1.0, 2.0, 3.0])
tests/unit/test_field.py-35-        f = Field(data=data, name="test", dims=("x",), units="m")
tests/unit/test_field.py:36:        children, aux = f.tree_flatten()
tests/unit/test_field.py-37-        f2 = Field.tree_unflatten(aux, children)
tests/unit/test_field.py-38-        assert f2.name == "test"
tests/unit/test_field.py-39-        assert jnp.allclose(f2.data, data)
tests/unit/test_field.py-40-
--
tests/land/test_train_multilayer_land.py-211-    (_pack KeyError fallback).  The finite-mask drops every cell -> smse is exactly 0
tests/land/test_train_multilayer_land.py-212-    and the loss/gradient stay finite, so legacy inputs still train."""
tests/land/test_train_multilayer_land.py-213-    data = dict(_synthetic_data())
tests/land/test_train_multilayer_land.py-214-    data["sm"] = jnp.full_like(data["sm"], jnp.nan)
tests/land/test_train_multilayer_land.py:215:    l, aux = loss_ml(init_ext_params(), data)
tests/land/test_train_multilayer_land.py-216-    assert float(aux[4]) == 0.0 and jnp.isfinite(l)          # smse (index 4) masked to 0
tests/land/test_train_multilayer_land.py-217-    g = jax.grad(lambda q: loss_ml(q, data)[0])(init_ext_params())
tests/land/test_train_multilayer_land.py-218-    assert all(jnp.all(jnp.isfinite(v)) for v in g.values())
tests/land/test_train_multilayer_land.py-219-
--
tests/ocean/fidelity/test_veros_runner.py-351-    )
tests/ocean/fidelity/test_veros_runner.py-352-    assert result.case_name == "dino"
tests/ocean/fidelity/test_veros_runner.py-353-    temp = result.variables["temp"]
tests/ocean/fidelity/test_veros_runner.py-354-    salt = result.variables["salt"]
tests/ocean/fidelity/test_veros_runner.py:355:    taux = result.variables["surface_taux"]
tests/ocean/fidelity/test_veros_runner.py-356-    assert np.isfinite(temp).all()
tests/ocean/fidelity/test_veros_runner.py-357-    assert np.isfinite(salt).all()
tests/ocean/fidelity/test_veros_runner.py-358-    # Wind stress profile spans the paper's [-0.1, 0.2] N/m^2 range.
tests/ocean/fidelity/test_veros_runner.py-359-    assert float(taux.min()) < -0.05
--
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-625-    is the OCEAN-SIDE stress (eastward-positive), applied with a ``+`` sign in
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-626-    Veros (``du += surface_taux/(rho_0·dz)``)."""
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-627-    yt = np.asarray(lat_deg, dtype=np.float64)
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-628-    yu = yt + 0.5 * DYT_DEG                       # v-point lat, as Veros uses
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py:629:    taux = np.zeros_like(yt)
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-630-    south = yt < _ACC_WIND_LAT_S
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-631-    north = yt > _ACC_WIND_LAT_N
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py:632:    taux = np.where(
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-633-        south,
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-634-        _ACC_TAUX_AMP * np.sin(
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-635-            np.pi * (yu - _VEROS_YU_MIN) / (_ACC_WIND_LAT_S - _VEROS_YT_MIN)),
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-636-        taux)
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py:637:    taux = np.where(
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-638-        north,
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-639-        _ACC_TAUX_AMP * (1.0 - np.cos(
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-640-            2.0 * np.pi * (yu - _ACC_WIND_LAT_N) / (_VEROS_YU_MAX - _ACC_WIND_LAT_N))),
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-641-        taux)
--
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-652-    here, which after legoESM's internal flip reproduces Veros's ``+taux`` —
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-653-    i.e. westerlies (taux>0) accelerate the ACC eastward. Verified by the sign
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-654-    of the spun-up channel jet (must be eastward)."""
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-655-    lat_deg = np.degrees(np.asarray(grid.lat))    # cell-centre lat (n_lat,)
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py:656:    taux = _acc_taux_profile(lat_deg)             # ocean-side stress (n_lat,)
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-657-    n_lat, n_lon = grid.n_lat, grid.n_lon
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-658-    tau_x = jnp.asarray(
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-659-        np.broadcast_to(-taux[:, None], (n_lat, n_lon)))   # negated: see docstring
packages/ocean/legoesm/ocean/fidelity/veros_acc_recipe.py-660-    tau_y = jnp.zeros((n_lat, n_lon), dtype=tau_x.dtype)
--
packages/ocean/legoesm/ocean/fidelity/veros_configs/dino.py-358-            taux_2d = (
packages/ocean/legoesm/ocean/fidelity/veros_configs/dino.py-359-                taux_1d[npx.newaxis, :]
packages/ocean/legoesm/ocean/fidelity/veros_configs/dino.py-360-                * npx.ones((nx + 4, 1))                 # broadcast through halos
packages/ocean/legoesm/ocean/fidelity/veros_configs/dino.py-361-            )
packages/ocean/legoesm/ocean/fidelity/veros_configs/dino.py:362:            vs.surface_taux = update(
packages/ocean/legoesm/ocean/fidelity/veros_configs/dino.py-363-                vs.surface_taux, at[...],
packages/ocean/legoesm/ocean/fidelity/veros_configs/dino.py-364-                taux_2d * vs.maskU[:, :, -1],
packages/ocean/legoesm/ocean/fidelity/veros_configs/dino.py-365-            )
packages/ocean/legoesm/ocean/fidelity/veros_configs/dino.py-366-
--
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-240-
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-241-def build_reentrant_channel_wind(geom: LatLonCGridGeometry) -> OceanSurfaceForcing:
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-242-    """gendata zonal wind ``taux(Y) = 0.2 sin(pi Y/(ny-1))`` [N/m^2].
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-243-
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py:244:    gendata: ``taux = 0.2*sin(0 : pi/39 : pi)`` over the 40 y-rows. This is the
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-245-    ocean-side stress; pass the NEGATED value so legoESM's atmosphere->ocean flip
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-246-    reproduces MITgcm's ``+taux`` (westerlies accelerate the ACC eastward)."""
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-247-    yy = np.linspace(0.0, np.pi, NY)                     # 0 : pi/39 : pi
packages/ocean/legoesm/ocean/fidelity/mitgcm_reentrant_channel_recipe.py-248-    tau_ocean = TAU_MAX * np.sin(yy)                     # (ny,) ocean-side stress
--
tests/ocean/unit/test_veros_global_1deg_recipe.py-418-    pulling the ZERO (never-exchanged) Veros ghost into the last interior
tests/ocean/unit/test_veros_global_1deg_recipe.py-419-    column/row — measured on the live oracle's t=0 surface_taux/y (a
tests/ocean/unit/test_veros_global_1deg_recipe.py-420-    cyclic roll fails the harness gate by 0.2 N/m²)."""
tests/ocean/unit/test_veros_global_1deg_recipe.py-421-    rng = np.random.default_rng(3)
tests/ocean/unit/test_veros_global_1deg_recipe.py:422:    taux = rng.normal(size=(NX, NY, 12))
tests/ocean/unit/test_veros_global_1deg_recipe.py-423-    tauy = rng.normal(size=(NX, NY, 12))
tests/ocean/unit/test_veros_global_1deg_recipe.py-424-    tx, ty = veros_mit_tau_shift_1deg(taux, tauy)
tests/ocean/unit/test_veros_global_1deg_recipe.py-425-    np.testing.assert_array_equal(tx[:-1], taux[1:])
tests/ocean/unit/test_veros_global_1deg_recipe.py-426-    np.testing.assert_array_equal(tx[-1], 0.0)           # zero x ghost (!)
--
tests/ocean/unit/test_coriolis_scheme.py-405-        state, model = _basin(scheme)
tests/ocean/unit/test_coriolis_scheme.py-406-        grid = model.grid
tests/ocean/unit/test_coriolis_scheme.py-407-        # Zonal wind stress τ_x (N/m²) at cell centres (recipe convention).
tests/ocean/unit/test_coriolis_scheme.py-408-        lat = np.degrees(np.asarray(grid.lat))
tests/ocean/unit/test_coriolis_scheme.py:409:        taux = (0.1 * np.cos(np.pi * lat / 80.0))[:, None] * np.ones((1, grid.n_lon))
tests/ocean/unit/test_coriolis_scheme.py-410-        tauy = np.zeros((grid.n_lat, grid.n_lon))
tests/ocean/unit/test_coriolis_scheme.py-411-        sf = OceanSurfaceForcing(
tests/ocean/unit/test_coriolis_scheme.py-412-            tau_x=jnp.asarray(taux * np.asarray(state.land_mask.data)),
tests/ocean/unit/test_coriolis_scheme.py-413-            tau_y=jnp.asarray(tauy))
--
tests/unit/test_land_ml_checkpoint_roundtrip.py-29-
tests/unit/test_land_ml_checkpoint_roundtrip.py-30-def test_land_ml_survives_carry_aux_npz_roundtrip(tmp_path):
tests/unit/test_land_ml_checkpoint_roundtrip.py-31-    saved = _state(1.0)
tests/unit/test_land_ml_checkpoint_roundtrip.py-32-    src = SimpleNamespace(
tests/unit/test_land_ml_checkpoint_roundtrip.py:33:        _carry_aux={}, _land_ml_state=saved,
tests/unit/test_land_ml_checkpoint_roundtrip.py-34-        _double_moment_step_inputs=lambda: {},
tests/unit/test_land_ml_checkpoint_roundtrip.py-35-        config=SimpleNamespace(convection="none"),
tests/unit/test_land_ml_checkpoint_roundtrip.py-36-    )
tests/unit/test_land_ml_checkpoint_roundtrip.py-37-
tests/unit/test_land_ml_checkpoint_roundtrip.py:38:    aux = ModelDriver._checkpoint_carry_aux(src)
tests/unit/test_land_ml_checkpoint_roundtrip.py-39-    assert aux is not None and any(k.startswith("land_ml_") for k in aux)
tests/unit/test_land_ml_checkpoint_roundtrip.py-40-
tests/unit/test_land_ml_checkpoint_roundtrip.py-41-    # Actually go through the disk the restart chain uses.
tests/unit/test_land_ml_checkpoint_roundtrip.py-42-    np.savez(tmp_path / "c.npz", **aux)
tests/unit/test_land_ml_checkpoint_roundtrip.py-43-    loaded = dict(np.load(tmp_path / "c.npz"))
tests/unit/test_land_ml_checkpoint_roundtrip.py-44-
tests/unit/test_land_ml_checkpoint_roundtrip.py-45-    # Restart lands on the cold-start state; restore must overwrite it.
tests/unit/test_land_ml_checkpoint_roundtrip.py:46:    dst = SimpleNamespace(_carry_aux=loaded, _land_ml_state=_state(99.0))
tests/unit/test_land_ml_checkpoint_roundtrip.py-47-    ModelDriver._restore_land_ml_from_carry_aux(dst)
tests/unit/test_land_ml_checkpoint_roundtrip.py-48-
tests/unit/test_land_ml_checkpoint_roundtrip.py-49-    for f in MultiLayerLandState._fields:
tests/unit/test_land_ml_checkpoint_roundtrip.py-50-        np.testing.assert_allclose(
--
tests/unit/test_land_ml_checkpoint_roundtrip.py-59-    array into the npz and crash the load-side jnp.asarray — None fields must
tests/unit/test_land_ml_checkpoint_roundtrip.py-60-    be skipped on save and left untouched on restore."""
tests/unit/test_land_ml_checkpoint_roundtrip.py-61-    saved = _state(1.0)._replace(TgC=None)
tests/unit/test_land_ml_checkpoint_roundtrip.py-62-    src = SimpleNamespace(
tests/unit/test_land_ml_checkpoint_roundtrip.py:63:        _carry_aux={}, _land_ml_state=saved,
tests/unit/test_land_ml_checkpoint_roundtrip.py-64-        _double_moment_step_inputs=lambda: {},
tests/unit/test_land_ml_checkpoint_roundtrip.py-65-        config=SimpleNamespace(convection="none"),
tests/unit/test_land_ml_checkpoint_roundtrip.py-66-    )
tests/unit/test_land_ml_checkpoint_roundtrip.py:67:    aux = ModelDriver._checkpoint_carry_aux(src)
tests/unit/test_land_ml_checkpoint_roundtrip.py-68-    assert "land_ml_TgC" not in aux          # None never serialized
tests/unit/test_land_ml_checkpoint_roundtrip.py-69-    np.savez(tmp_path / "c.npz", **aux)      # must not need allow_pickle
tests/unit/test_land_ml_checkpoint_roundtrip.py-70-    loaded = dict(np.load(tmp_path / "c.npz"))
tests/unit/test_land_ml_checkpoint_roundtrip.py-71-
tests/unit/test_land_ml_checkpoint_roundtrip.py:72:    dst = SimpleNamespace(_carry_aux=loaded,
tests/unit/test_land_ml_checkpoint_roundtrip.py-73-                          _land_ml_state=_state(99.0)._replace(TgC=None))
tests/unit/test_land_ml_checkpoint_roundtrip.py-74-    ModelDriver._restore_land_ml_from_carry_aux(dst)
tests/unit/test_land_ml_checkpoint_roundtrip.py-75-    assert dst._land_ml_state.TgC is None    # stays None, not resurrected
tests/unit/test_land_ml_checkpoint_roundtrip.py-76-    np.testing.assert_allclose(dst._land_ml_state.T_soil, saved.T_soil)
--
tests/unit/test_land_ml_checkpoint_roundtrip.py-79-def test_restore_is_noop_for_slab_land():
tests/unit/test_land_ml_checkpoint_roundtrip.py-80-    # Slab run (_land_ml_state None) must ignore any stray land_ml_* keys AND
tests/unit/test_land_ml_checkpoint_roundtrip.py-81-    # strip them, so they never leak forward into the next slab checkpoint.
tests/unit/test_land_ml_checkpoint_roundtrip.py-82-    dst = SimpleNamespace(
tests/unit/test_land_ml_checkpoint_roundtrip.py:83:        _carry_aux={"land_ml_T_soil": np.ones(3)}, _land_ml_state=None)
tests/unit/test_land_ml_checkpoint_roundtrip.py-84-    ModelDriver._restore_land_ml_from_carry_aux(dst)
tests/unit/test_land_ml_checkpoint_roundtrip.py-85-    assert dst._land_ml_state is None
tests/unit/test_land_ml_checkpoint_roundtrip.py-86-    assert "land_ml_T_soil" not in dst._carry_aux   # popped, not left to re-save
tests/unit/test_land_ml_checkpoint_roundtrip.py-87-
--
tests/unit/test_land_ml_checkpoint_roundtrip.py-91-    # loud — restoring only the present fields would leave the rest silently at
tests/unit/test_land_ml_checkpoint_roundtrip.py-92-    # cold-start values (a mixed restart), exactly what #730 exists to prevent.
tests/unit/test_land_ml_checkpoint_roundtrip.py-93-    saved = _state(1.0)
tests/unit/test_land_ml_checkpoint_roundtrip.py-94-    src = SimpleNamespace(
tests/unit/test_land_ml_checkpoint_roundtrip.py:95:        _carry_aux={}, _land_ml_state=saved,
tests/unit/test_land_ml_checkpoint_roundtrip.py-96-        _double_moment_step_inputs=lambda: {},
tests/unit/test_land_ml_checkpoint_roundtrip.py-97-        config=SimpleNamespace(convection="none"),
tests/unit/test_land_ml_checkpoint_roundtrip.py-98-    )
tests/unit/test_land_ml_checkpoint_roundtrip.py:99:    aux = ModelDriver._checkpoint_carry_aux(src)
tests/unit/test_land_ml_checkpoint_roundtrip.py-100-    del aux["land_ml_snow_depth"]                    # simulate a dropped column
tests/unit/test_land_ml_checkpoint_roundtrip.py:101:    dst = SimpleNamespace(_carry_aux=dict(aux), _land_ml_state=_state(99.0))
tests/unit/test_land_ml_checkpoint_roundtrip.py-102-    with pytest.raises(ValueError, match="does not match"):
tests/unit/test_land_ml_checkpoint_roundtrip.py-103-        ModelDriver._restore_land_ml_from_carry_aux(dst)
tests/unit/test_land_ml_checkpoint_roundtrip.py-104-
tests/unit/test_land_ml_checkpoint_roundtrip.py-105-
tests/unit/test_land_ml_checkpoint_roundtrip.py-106-def test_unknown_field_raises():
tests/unit/test_land_ml_checkpoint_roundtrip.py-107-    dst = SimpleNamespace(
tests/unit/test_land_ml_checkpoint_roundtrip.py:108:        _carry_aux={f"land_ml_{f}": np.ones((4, 6) if i < 3 else (4,),
tests/unit/test_land_ml_checkpoint_roundtrip.py-109-                                            dtype=np.float32)
tests/unit/test_land_ml_checkpoint_roundtrip.py-110-                    for i, f in enumerate(MultiLayerLandState._fields[:7])}
tests/unit/test_land_ml_checkpoint_roundtrip.py-111-        | {"land_ml_bogus": np.ones(4, dtype=np.float32)},
tests/unit/test_land_ml_checkpoint_roundtrip.py-112-        _land_ml_state=_state(99.0)._replace(TgC=None, surface_water=None))
--
tests/unit/test_land_ml_checkpoint_roundtrip.py-120-    good = {f"land_ml_{f}": np.ones((4, 6) if i < 3 else (4,), dtype=np.float32)
tests/unit/test_land_ml_checkpoint_roundtrip.py-121-            for i, f in enumerate(MultiLayerLandState._fields[:7])}
tests/unit/test_land_ml_checkpoint_roundtrip.py-122-    good["land_ml_T_soil"] = np.ones((4, 8), dtype=np.float32)   # 8 != 6 layers
tests/unit/test_land_ml_checkpoint_roundtrip.py-123-    dst = SimpleNamespace(
tests/unit/test_land_ml_checkpoint_roundtrip.py:124:        _carry_aux=good,
tests/unit/test_land_ml_checkpoint_roundtrip.py-125-        _land_ml_state=_state(99.0)._replace(TgC=None, surface_water=None))
tests/unit/test_land_ml_checkpoint_roundtrip.py-126-    with pytest.raises(ValueError, match="shape"):
tests/unit/test_land_ml_checkpoint_roundtrip.py-127-        ModelDriver._restore_land_ml_from_carry_aux(dst)
tests/unit/test_land_ml_checkpoint_roundtrip.py-128-
--
packages/tools/legoesm/forcing/amip_config.py-389-    # restart non-bit-exact for the physics memory.  Everything else
packages/tools/legoesm/forcing/amip_config.py-390-    # (held radiation, conv_prog, T_land, …) keeps the legacy storage
packages/tools/legoesm/forcing/amip_config.py-391-    # cast.
packages/tools/legoesm/forcing/amip_config.py-392-    _keep_stored_dtype = ("tke", "qke", "gwd_spectrum")
packages/tools/legoesm/forcing/amip_config.py:393:    carry_aux = {}
packages/tools/legoesm/forcing/amip_config.py-394-    for key in data.files:
packages/tools/legoesm/forcing/amip_config.py-395-        if key.startswith("carry_"):
packages/tools/legoesm/forcing/amip_config.py-396-            _name = key[6:]
packages/tools/legoesm/forcing/amip_config.py-397-            if _name == "conv_prog_scheme":
--
tests/ocean/unit/test_isoneutral_slope_density.py-510-                mask=mask, rho_0=RHO_0, g=G,
tests/ocean/unit/test_isoneutral_slope_density.py-511-            )
tests/ocean/unit/test_isoneutral_slope_density.py-512-            return jnp.sum(K33), K33
tests/ocean/unit/test_isoneutral_slope_density.py-513-
tests/ocean/unit/test_isoneutral_slope_density.py:514:        (total, K33), grad = jax.value_and_grad(k33_sum, has_aux=True)(T)
tests/ocean/unit/test_isoneutral_slope_density.py-515-        assert jnp.all(jnp.isfinite(K33))
tests/ocean/unit/test_isoneutral_slope_density.py-516-        assert float(jnp.min(K33)) >= 0.0
tests/ocean/unit/test_isoneutral_slope_density.py-517-        assert float(total) > 0.0
tests/ocean/unit/test_isoneutral_slope_density.py-518-        assert jnp.all(jnp.isfinite(grad))
--
tests/unit/test_physics_state_carry.py-442-    # step (issue #405) both runs would end identical.
tests/unit/test_physics_state_carry.py-443-    driver_b = ModelDriver(cfg, output_dir=tmp_path / "b")
tests/unit/test_physics_state_carry.py-444-    driver_b.setup()
tests/unit/test_physics_state_carry.py-445-    tke_seed = jnp.asarray(tke_a)
tests/unit/test_physics_state_carry.py:446:    driver_b._carry_aux = {
tests/unit/test_physics_state_carry.py-447-        "tke": tke_seed.at[0, :].set(tke_seed[0, :] + 1e-2),
tests/unit/test_physics_state_carry.py-448-    }
tests/unit/test_physics_state_carry.py-449-    status_b = driver_b.run(compiled=compiled)
tests/unit/test_physics_state_carry.py-450-    assert status_b == "COMPLETED"
--
tests/unit/test_physics_state_carry.py-639-
tests/unit/test_physics_state_carry.py-640-    cfg = _stateful_driver_config()
tests/unit/test_physics_state_carry.py-641-    driver = ModelDriver(cfg, output_dir=tmp_path)
tests/unit/test_physics_state_carry.py-642-    driver.setup()
tests/unit/test_physics_state_carry.py:643:    driver._carry_aux = {"tke": jnp.zeros((7, 3))}   # wrong shape
tests/unit/test_physics_state_carry.py-644-    with pytest.raises(ValueError, match="405"):
tests/unit/test_physics_state_carry.py-645-        driver.run(compiled=True)
tests/unit/test_physics_state_carry.py-646-
tests/unit/test_physics_state_carry.py-647-
--
tests/unit/test_double_moment_checkpoint.py-24-    """Minimal object exposing just the attributes the DM checkpoint helpers
tests/unit/test_double_moment_checkpoint.py-25-    touch, with the helpers bound from ModelDriver."""
tests/unit/test_double_moment_checkpoint.py-26-    _DOUBLE_MOMENT_TRACERS = ModelDriver._DOUBLE_MOMENT_TRACERS
tests/unit/test_double_moment_checkpoint.py-27-    _double_moment_step_inputs = ModelDriver._double_moment_step_inputs
tests/unit/test_double_moment_checkpoint.py:28:    _checkpoint_carry_aux = ModelDriver._checkpoint_carry_aux
tests/unit/test_double_moment_checkpoint.py:29:    _restore_dm_tracers_from_carry_aux = ModelDriver._restore_dm_tracers_from_carry_aux
tests/unit/test_double_moment_checkpoint.py-30-
tests/unit/test_double_moment_checkpoint.py-31-    def __init__(self, tracers, carry_aux):
tests/unit/test_double_moment_checkpoint.py-32-        self.tracers = tracers
tests/unit/test_double_moment_checkpoint.py:33:        self._carry_aux = carry_aux
tests/unit/test_double_moment_checkpoint.py-34-
tests/unit/test_double_moment_checkpoint.py-35-
tests/unit/test_double_moment_checkpoint.py-36-_DM = ("q_i", "q_s", "q_g", "N_c", "N_r", "N_i")
tests/unit/test_double_moment_checkpoint.py-37-
--
tests/unit/test_double_moment_checkpoint.py-47-
tests/unit/test_double_moment_checkpoint.py-48-
tests/unit/test_double_moment_checkpoint.py-49-def test_checkpoint_carry_aux_includes_dm_tracers():
tests/unit/test_double_moment_checkpoint.py-50-    src = _Stub(_full_tracers(), {"held_dT_rad": jnp.zeros((4,))})
tests/unit/test_double_moment_checkpoint.py:51:    aux = src._checkpoint_carry_aux()
tests/unit/test_double_moment_checkpoint.py-52-    # held-radiation carry_aux preserved AND every DM tracer namespaced dmtr_*.
tests/unit/test_double_moment_checkpoint.py-53-    assert "held_dT_rad" in aux
tests/unit/test_double_moment_checkpoint.py-54-    for k in _DM:
tests/unit/test_double_moment_checkpoint.py-55-        assert f"dmtr_{k}" in aux, k
--
tests/unit/test_double_moment_checkpoint.py-60-def test_warm_rain_checkpoint_carry_aux_is_unchanged():
tests/unit/test_double_moment_checkpoint.py-61-    aux_in = {"held_dT_rad": jnp.zeros((4,))}
tests/unit/test_double_moment_checkpoint.py-62-    src = _Stub({"q_v": jnp.ones((4,)), "q_c": jnp.zeros((4,)),
tests/unit/test_double_moment_checkpoint.py-63-                 "q_r": jnp.zeros((4,))}, dict(aux_in))
tests/unit/test_double_moment_checkpoint.py:64:    aux = src._checkpoint_carry_aux()
tests/unit/test_double_moment_checkpoint.py-65-    assert set(aux) == set(aux_in)            # no dmtr_* added for warm-rain
tests/unit/test_double_moment_checkpoint.py-66-
tests/unit/test_double_moment_checkpoint.py-67-
tests/unit/test_double_moment_checkpoint.py-68-def test_restore_round_trip():
--
tests/ocean/unit/test_veros_global_flexible_recipe.py-412-def test_mit_tau_shift():
tests/ocean/unit/test_veros_global_flexible_recipe.py-413-    """surface_taux[i] = taux[i+1] with the cyclic wrap; surface_tauy[j] =
tests/ocean/unit/test_veros_global_flexible_recipe.py-414-    tauy[j+1] with the zero north ghost pulled into the last row."""
tests/ocean/unit/test_veros_global_flexible_recipe.py-415-    rng = np.random.default_rng(3)
tests/ocean/unit/test_veros_global_flexible_recipe.py:416:    taux = rng.normal(size=(NX, NY, 12))
tests/ocean/unit/test_veros_global_flexible_recipe.py-417-    tauy = rng.normal(size=(NX, NY, 12))
tests/ocean/unit/test_veros_global_flexible_recipe.py-418-    tx, ty = veros_mit_tau_shift(taux, tauy)
tests/ocean/unit/test_veros_global_flexible_recipe.py-419-    np.testing.assert_array_equal(tx[:-1], taux[1:])
tests/ocean/unit/test_veros_global_flexible_recipe.py-420-    np.testing.assert_array_equal(tx[-1], taux[0])       # cyclic wrap
--
scripts/bench/metadata.py-894-    block_steps: int,
scripts/bench/metadata.py-895-    n_blocks: int = 2,
scripts/bench/metadata.py-896-    probe_steps: int = 3,
scripts/bench/metadata.py-897-    sync_label: str = "timed_scan_blocks",
scripts/bench/metadata.py:898:    aux=None,
scripts/bench/metadata.py-899-):
scripts/bench/metadata.py-900-    """Measurement-contract timing: fused ``lax.scan`` blocks + probe latency.
scripts/bench/metadata.py-901-
scripts/bench/metadata.py-902-    The trustworthy production-like number is a MULTI-STEP ``lax.scan`` block
--
tests/parallel/test_latlon_ocean_spmd_step.py-126-    # n_lat-row block that DOES divide N — a uniform tree.map(P("lat")) would
tests/parallel/test_latlon_ocean_spmd_step.py-127-    # fail on the n_lat+1 v rows).  The result is gathered (and the dropped pole
tests/parallel/test_latlon_ocean_spmd_step.py-128-    # row reappended) for the bit-comparison vs the single-device reference.
tests/parallel/test_latlon_ocean_spmd_step.py-129-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:130:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-131-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-132-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-133-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_step.py-134-    ss = gather_state_latlon(ss, dev.mesh)
--
tests/parallel/test_latlon_ocean_spmd_step.py-243-                       sponge=sponge, t_seconds=jnp.asarray(i * dt))
tests/parallel/test_latlon_ocean_spmd_step.py-244-
tests/parallel/test_latlon_ocean_spmd_step.py-245-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-246-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:247:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-248-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-249-    fws = shard_forcing_latlon(fw, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-250-    sfs = shard_forcing_latlon(sf, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-251-    sponges = shard_forcing_latlon(sponge, dev.mesh)
--
tests/parallel/test_latlon_ocean_spmd_step.py-395-                                  LatLonCGridOceanConfig.from_flat())
tests/parallel/test_latlon_ocean_spmd_step.py-396-    state0 = _perturbed_state(grid, z_coord)
tests/parallel/test_latlon_ocean_spmd_step.py-397-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-398-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py:399:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-400-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-401-    bad = OceanSurfaceForcing(
tests/parallel/test_latlon_ocean_spmd_step.py-402-        tau_y=jnp.zeros((grid.n_lat + 1, grid.n_lon)))
tests/parallel/test_latlon_ocean_spmd_step.py-403-    with pytest.raises(ValueError, match="leading dim"):
--
tests/parallel/test_latlon_ocean_spmd_step.py-450-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-451-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py-452-
tests/parallel/test_latlon_ocean_spmd_step.py-453-    # explicit scatter -> inner sharded step -> gather
tests/parallel/test_latlon_ocean_spmd_step.py:454:    inner = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-455-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-456-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-457-        ss = inner(ss, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_step.py-458-    ss = gather_state_latlon(ss, dev.mesh)
--
tests/parallel/test_latlon_spmd_fused_halo.py-345-    outs = {}
tests/parallel/test_latlon_spmd_fused_halo.py-346-    hlo_counts = {}
tests/parallel/test_latlon_spmd_fused_halo.py-347-    for flag in ("0", "1"):
tests/parallel/test_latlon_spmd_fused_halo.py-348-        _env_flag(monkeypatch, flag)
tests/parallel/test_latlon_spmd_fused_halo.py:349:        step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-350-        ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-351-        hlo_counts[flag] = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-352-            jax.jit(step).lower(ss, 600.0).compile().as_text())
tests/parallel/test_latlon_spmd_fused_halo.py-353-        for _ in range(n_steps):
--
tests/parallel/test_latlon_spmd_fused_halo.py-368-    # freeze the wrapper's python body and never re-evaluate the key
tests/parallel/test_latlon_spmd_fused_halo.py-369-    # (an outer-jit artifact, not the production call pattern — the
tests/parallel/test_latlon_spmd_fused_halo.py-370-    # driver calls step() directly each step).
tests/parallel/test_latlon_spmd_fused_halo.py-371-    _env_flag(monkeypatch, "0")
tests/parallel/test_latlon_spmd_fused_halo.py:372:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-373-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-374-    n_off = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-375-        jax.jit(lambda s, d: step(s, d)).lower(ss, 600.0)
tests/parallel/test_latlon_spmd_fused_halo.py-376-        .compile().as_text())
--
tests/parallel/test_persistent_sharded_ocean_loop.py-197-    assert old_calls == {"shard": n_steps, "gather": n_steps}, old_calls
tests/parallel/test_persistent_sharded_ocean_loop.py-198-
tests/parallel/test_persistent_sharded_ocean_loop.py-199-    # ---------------- NEW lane: persistent sharded loop ----------------------
tests/parallel/test_persistent_sharded_ocean_loop.py-200-    calls["shard"] = calls["gather"] = 0
tests/parallel/test_persistent_sharded_ocean_loop.py:201:    inner = sos.make_sharded_ocean_step(model, mesh)
tests/parallel/test_persistent_sharded_ocean_loop.py-202-    ss = sos.shard_state_latlon(state0, mesh)          # ONE initial shard
tests/parallel/test_persistent_sharded_ocean_loop.py-203-    for k in range(1, n_steps + 1):
tests/parallel/test_persistent_sharded_ocean_loop.py-204-        # surface-current consumer on the SHARDED state: v is the n_lat-row
tests/parallel/test_persistent_sharded_ocean_loop.py-205-        # v_lower carrier at every loop top (the snapshot boundary re-shards
--
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-121-    # Prime the build-once vertex-mask cache from the CONCRETE state (the
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-122-    # serial run above already did; belt-and-braces for wrapper band masks).
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-123-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-124-
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:125:    step = make_sharded_ocean_step(model, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-126-    ss = shard_state_latlon(state0, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-127-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-128-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-129-    out = gather_state_latlon(ss, mesh)
--
tests/parallel/test_latlon_ocean_spmd_tripole.py-148-
tests/parallel/test_latlon_ocean_spmd_tripole.py-149-    model._ensure_vertex_mask(state0)        # prime the build-once vmask cache
tests/parallel/test_latlon_ocean_spmd_tripole.py-150-
tests/parallel/test_latlon_ocean_spmd_tripole.py-151-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_tripole.py:152:    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-153-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-154-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_tripole.py-155-        ss = step(ss, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_tripole.py-156-    ss = gather_state_latlon(ss, dev.mesh)
--
tests/unit/test_amip_audit_driver_fixes.py-136-    held_carry = {"held_dT_rad": jnp.zeros((6, 4, 4, 3))}
tests/unit/test_amip_audit_driver_fixes.py-137-
tests/unit/test_amip_audit_driver_fixes.py-138-    driver = ModelDriver(_cfg(2, "zarr"), output_dir=str(tmp_path / "z2"))
tests/unit/test_amip_audit_driver_fixes.py-139-    driver.setup()
tests/unit/test_amip_audit_driver_fixes.py:140:    driver._carry_aux = dict(held_carry)
tests/unit/test_amip_audit_driver_fixes.py-141-    with pytest.raises(ValueError, match="held radiation fluxes"):
tests/unit/test_amip_audit_driver_fixes.py-142-        driver.save_checkpoint(0, 0.0)
tests/unit/test_amip_audit_driver_fixes.py-143-
tests/unit/test_amip_audit_driver_fixes.py-144-    # rad_update_steps==1: held fields recomputed every step, guard silent.
tests/unit/test_amip_audit_driver_fixes.py-145-    driver1 = ModelDriver(_cfg(1, "zarr"), output_dir=str(tmp_path / "z1"))
tests/unit/test_amip_audit_driver_fixes.py-146-    driver1.setup()
tests/unit/test_amip_audit_driver_fixes.py:147:    driver1._carry_aux = dict(held_carry)
tests/unit/test_amip_audit_driver_fixes.py-148-    driver1.save_checkpoint(0, 0.0)   # must NOT raise on the held guard
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-506-        inv_before = ocean_invariants(model, s0, n_ranks=1)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-507-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-508-    if nd == 1:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-509-        mesh = None
scripts/bench/bench_ocean_latlon_spmd_scaling.py:510:        step = make_sharded_ocean_step(model, None)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-511-        s = s0
scripts/bench/bench_ocean_latlon_spmd_scaling.py-512-    else:
scripts/bench/bench_ocean_latlon_spmd_scaling.py-513-        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-514-                                 axis_names=("lat",))
scripts/bench/bench_ocean_latlon_spmd_scaling.py:515:        step = make_sharded_ocean_step(model, mesh)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-516-        s = shard_state_latlon(s0, mesh)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-517-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-518-    # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
scripts/bench/bench_ocean_latlon_spmd_scaling.py-519-    # blocks with sync only AROUND the block — the previous per-step
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-533-        _blk, _nblk, _probe = args.steps, args.blocks, args.probe_steps
scripts/bench/bench_ocean_latlon_spmd_scaling.py-534-    # aux threads the sharded geometry stacks through the jit boundary as
scripts/bench/bench_ocean_latlon_spmd_scaling.py-535-    # an ARGUMENT (outer-trace constants of non-addressable arrays are
scripts/bench/bench_ocean_latlon_spmd_scaling.py-536-    # unfetchable — see make_sharded_ocean_step's aux note).
scripts/bench/bench_ocean_latlon_spmd_scaling.py:537:    _aux = getattr(step, "aux", None)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-538-    s, timing = timed_scan_blocks(
scripts/bench/bench_ocean_latlon_spmd_scaling.py:539:        (lambda st, aux: step(st, args.dt, aux=aux)) if _aux is not None

exec
/usr/bin/bash -lc "command -v python3; command -v pytest; command -v uv; command -v pixi; command -v hatch; find /work/bd1083 -maxdepth 4 -type f -path '*/bin/python*' -print 2>/dev/null | sed -n '1,100p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 9909ms:
/usr/bin/python3
/home/b/b381103/.local/bin/uv
/work/bd1083/b382650/bin/python3.12
/work/bd1083/b382650/bin/python3.12-config
/work/bd1083/b309178/mambaforge/bin/python3.9-config
/work/bd1083/b309178/mambaforge/bin/python3.9
/work/bd1083/b382081/miniconda3/bin/python3.12
/work/bd1083/b382081/miniconda3/bin/python3.12-config
/work/bd1083/b309170/Miniconda3/bin/python3.12-config
/work/bd1083/b309170/Miniconda3/bin/python3.12
/work/bd1083/b309170/Miniconda3_OLD/bin/python3.8
/work/bd1083/b309170/Miniconda3_OLD/bin/python3.8-config
/work/bd1083/b383308/miniforge3/bin/python3.12-config
/work/bd1083/b383308/miniforge3/bin/python3.12

exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 /work/bd1083/b309178/mambaforge/bin/python3.9 -B -c 'import jax, numpy; print(jax.__version__); print(numpy.__version__)'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 1 in 792ms:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'jax'

exec
/usr/bin/bash -lc "PYTHONDONTWRITEBYTECODE=1 /work/bd1083/b382650/bin/python3.12 -B -c 'import jax, numpy; print(jax.__version__); print(numpy.__version__)'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 1 in 1216ms:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'jax'

exec
/usr/bin/bash -lc "rg -n -C 5 'def make_sharded_atm_latlon_step\\(|def make_sharded_atm_latlon_step_2d\\(|shard_geometry=' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py packages scripts tests | sed -n '1,500p'; rg -n -C 6 'shard_geometry' scripts/bench tests packages/coupler packages/atmosphere | sed -n '1,600p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 815ms:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-410-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-411-    ``replicated_per_device_bytes``: the historical layout — every device
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-412-    holds the full all-band stack of every ``LatLonGrid`` array field
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-413-    (:func:`atm_grid_array_field_names`; dominated by the five
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-414-    ``(n_lat, n_lon)`` 2-D fields ``lat2d, lon2d, f, dx, area``).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:415:    ``sharded_per_device_bytes``: the ``shard_geometry=True`` layout — each
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-416-    device holds only its own band's slice (exactly ``replicated /
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-417-    n_devices``; the stack leading dim is ``n_devices`` and bands are
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-418-    uniform).  Excludes the optional polar-filter mask stacks (two
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-419-    ``(n_lat,)``-scale vectors when the filter is on) — negligible next to
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-420-    the 2-D fields and absent in the default configs.
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-434-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-435-def _build_geometry_stacks(model, mesh, n_dev: int, shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-436-    """Stack every band's ``LatLonGrid`` array fields (+ the optional
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-437-    polar-filter masks) over a leading band axis and lay them out on ``mesh``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-438-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:439:    ``shard_geometry=False`` (the historical layout): every stack is
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-440-    ``device_put`` REPLICATED (``P()``) — each device holds ALL bands'
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-441-    geometry and the body indexes its own band at ``axis_index``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-442-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:443:    ``shard_geometry=True`` (M2b): every stack is sharded ``P("lat", ...)``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-444-    on the leading band axis — each device holds ONLY its own band's slice
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-445-    (leading extent 1 inside the shard_map body, static index ``[0]``).  The
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-446-    VALUES the body consumes are identical either way (the same band slice),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-447-    so the step numerics are bit-unchanged; only the residency changes
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-448-    (per-device geometry bytes drop by ``n_dev`` —
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-612-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-613-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-614-def _agree_spmd_entry(model, mesh, *, n_steps=None, segment_steps=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-615-                      compiled_segments=None, has_physics_fn=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-616-                      has_on_segment=None, has_phys_state=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:617:                      shard_geometry=None, where: str) -> None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-618-    """Agree EVERY rank-local input, as the FIRST statement of a public entry.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-619-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-620-    #1362 / codex rounds 2-3. Gating individual refusals was not enough: each
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-621-    public entry point performs several rank-local checks BEFORE reaching any
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-622-    collective -- ``n_steps`` validation, ``_check_2d_mesh``, the ``mesh is
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-849-    finally:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-850-        set_spmd_mesh(prev_mesh)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-851-        set_halo_backend(prev_backend, prev_topo)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-852-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-853-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:854:def make_sharded_atm_latlon_step(model, mesh, physics_fn=None, *,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-855-                                 shard_geometry: bool = False):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-856-    """Return ``step(c_state, dt) -> c_state`` running the C-grid hydrostatic atm
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-857-    step lat-band-SPMD over the 1-D ``"lat"`` mesh.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-858-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-859-    ``c_state`` is a ``CGridLatLonHydrostaticState`` laid out with
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-898-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-899-    # FIRST statement: agree every rank-local input before ANY
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-900-    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-901-    _agree_spmd_entry(model, mesh, n_steps=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-902-                      has_physics_fn=physics_fn is not None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:903:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-904-                      where="make_sharded_atm_latlon_step")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-905-    from legoesm.parallel.latlon_spmd import latlon_band_perms
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-906-    from legoesm.parallel.shard_map_compat import shard_map
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-907-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-908-    # Stochastic physics is SPMD-safe since increment 2: the Bechtold AR1
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1037-    ``all_finite`` is a REPLICATED traced scalar bool from
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1038-    :func:`state_finite_scalar` — the in-graph blowup guard (``psum`` of
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1039-    per-band non-finite presence over ALL state leaves).  The host reads
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1040-    this ONE scalar per segment instead of gathering the full state.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1041-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1042:    ``shard_geometry=True`` (default — a NEW API, no historical layout to
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1043-    preserve): each device holds ONLY its own band's geometry slice
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1044-    (``P("lat")`` stacks) instead of a replicated all-band copy —
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1045-    bit-identical numerics, 1/n_dev the geometry bytes
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1046-    (:func:`atm_latlon_geometry_bytes`).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1047-
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1071-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1072-    # FIRST statement: agree every rank-local input before ANY
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1073-    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1074-    _agree_spmd_entry(model, mesh, n_steps=n_steps,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1075-                      has_physics_fn=physics_fn is not None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1076:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1077-                      where="make_sharded_atm_latlon_segment")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1078-    from legoesm.parallel.latlon_spmd import latlon_band_perms
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1079-    from legoesm.parallel.shard_map_compat import shard_map
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1080-    from legoesm.timestepping.integration import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1081-        refuse_unthreaded_stateful_physics)
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1534-    """Stack every tile's ``LatLonGrid`` array fields (+ the optional
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1535-    polar-filter masks at ``p_lon == 1``) over LEADING ``(p_lat, p_lon)``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1536-    tile axes and lay them out on ``mesh`` — the 2-D twin of
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1537-    :func:`_build_geometry_stacks`.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1538-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1539:    ``shard_geometry=True``: stacks are sharded ``P("lat", "lon", ...)`` on
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1540-    the tile axes — each device holds ONLY its own tile's slice (leading
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1541-    extents ``(1, 1)`` inside the body, static index ``[0, 0]``).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1542:    ``shard_geometry=False``: replicated (``P()``) stacks, indexed at
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1543-    ``(axis_index("lat"), axis_index("lon"))``.  Same tile VALUES either way.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1544-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1545-    Returns ``(template, array_field_names, stacks, stacks_spec)``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1546-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1547-    grid = model.grid
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1679-            "make_latlon_2d_mpi_step DOES wire this via the AD-safe "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1680-            "lat-pencil transpose; the SPMD ppermute equivalent is a "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1681-            "follow-up.)  Use p_lon == 1 or disable the filter.")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1682-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1683-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1684:def make_sharded_atm_latlon_step_2d(model, mesh, physics_fn=None, *,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1685-                                    shard_geometry: bool = True):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1686-    """Return ``step(c_state, dt) -> c_state`` running the C-grid hydrostatic
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1687-    atm step 2-D-tile-SPMD over a ``("lat", "lon")`` mesh — the M3a native
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1688-    2-D tiling twin of :func:`make_sharded_atm_latlon_step`.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1689-
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1710-    with no collectives.  A stateful ``PhysicsState`` carry is REFUSED: its
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1711-    ``(ncol, ...)`` leaves flatten lat-major over the GLOBAL grid, so a
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1712-    contiguous dim-0 shard is a lat BAND's columns, not a 2-D tile's —
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1713-    thread carries through the 1-D :func:`make_sharded_atm_latlon_step`.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1714-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1715:    ``shard_geometry=True`` (default — new API, no historical layout):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1716-    per-device tile geometry slices (``P("lat", "lon")`` stacks);
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1717-    ``False`` replicates the all-tile stacks (indexed at the axis indices).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1718-    Same tile values either way (bit-identical numerics).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1719-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1720-    # FIRST statement: agree every rank-local input before ANY
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1721-    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1722-    _agree_spmd_entry(model, mesh, n_steps=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1723-                      has_physics_fn=physics_fn is not None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1724:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1725-                      where="make_sharded_atm_latlon_step_2d")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1726-    from legoesm.parallel.latlon_spmd import latlon_band_perms
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1727-    from legoesm.parallel.shard_map_compat import shard_map
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1728-    from legoesm.timestepping.integration import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1729-        refuse_unthreaded_stateful_physics)
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1788-    ``mesh=None`` -> the single-device compiled twin)."""
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1789-    # FIRST statement: agree every rank-local input before ANY
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1790-    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1791-    _agree_spmd_entry(model, mesh, n_steps=n_steps,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1792-                      has_physics_fn=physics_fn is not None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1793:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1794-                      where="make_sharded_atm_latlon_segment_2d")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1795-    from legoesm.parallel.latlon_spmd import latlon_band_perms
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1796-    from legoesm.parallel.shard_map_compat import shard_map
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1797-    from legoesm.timestepping.integration import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1798-        refuse_unthreaded_stateful_physics)
--
tests/parallel/test_atm_latlon_2d_tiling.py-349-    du_s, dv_s, dT_s, dps_s, _ = cgrid_latlon_hydrostatic_tendencies(
tests/parallel/test_atm_latlon_2d_tiling.py-350-        state, grid, sigma, cfg)
tests/parallel/test_atm_latlon_2d_tiling.py-351-    serial = tuple(np.asarray(x) for x in (du_s, dv_s, dT_s, dps_s))
tests/parallel/test_atm_latlon_2d_tiling.py-352-
tests/parallel/test_atm_latlon_2d_tiling.py-353-    template, afn, stacks, stacks_spec = _build_geometry_stacks_2d(
tests/parallel/test_atm_latlon_2d_tiling.py:354:        model, mesh, p_lat, p_lon, shard_geometry=True)
tests/parallel/test_atm_latlon_2d_tiling.py-355-    perm_north, _ = latlon_band_perms(p_lat)
tests/parallel/test_atm_latlon_2d_tiling.py-356-
tests/parallel/test_atm_latlon_2d_tiling.py-357-    def _body(sl, st):
tests/parallel/test_atm_latlon_2d_tiling.py-358-        tg = template._replace(**{n: st[n][0, 0] for n in afn})
tests/parallel/test_atm_latlon_2d_tiling.py-359-        vf = reconstruct_vface_lower(sl.v, "lat", perm_north)
--
tests/parallel/test_atm_latlon_2d_tiling.py-666-
tests/parallel/test_atm_latlon_2d_tiling.py-667-def test_2d_degenerate_41_bitmatches_1d_band():
tests/parallel/test_atm_latlon_2d_tiling.py-668-    """On a (4,1) mesh every lon-ring op takes its static local branch, so
tests/parallel/test_atm_latlon_2d_tiling.py-669-    the 2-D step must reproduce the 1-D band step BIT-FOR-BIT (same ops, same
tests/parallel/test_atm_latlon_2d_tiling.py-670-    order — pure data-movement differences only).  Both factories run with
tests/parallel/test_atm_latlon_2d_tiling.py:671:    shard_geometry=True (the geometry VALUES are identical either way, gated
tests/parallel/test_atm_latlon_2d_tiling.py-672-    by the M2b bit-match)."""
tests/parallel/test_atm_latlon_2d_tiling.py-673-    mesh1 = _mesh1d(N_DEV)
tests/parallel/test_atm_latlon_2d_tiling.py-674-    mesh2 = _mesh2d(N_DEV, 1)
tests/parallel/test_atm_latlon_2d_tiling.py-675-    model, state = _model_and_state()
tests/parallel/test_atm_latlon_2d_tiling.py-676-    n_steps = 3
tests/parallel/test_atm_latlon_2d_tiling.py-677-
tests/parallel/test_atm_latlon_2d_tiling.py:678:    step1d = make_sharded_atm_latlon_step(model, mesh1, shard_geometry=True)
tests/parallel/test_atm_latlon_2d_tiling.py-679-    s1 = shard_state_atm_latlon(state, mesh1)
tests/parallel/test_atm_latlon_2d_tiling.py-680-    for _ in range(n_steps):
tests/parallel/test_atm_latlon_2d_tiling.py-681-        s1 = step1d(s1, DT)
tests/parallel/test_atm_latlon_2d_tiling.py-682-    out1 = gather_state_atm_latlon(s1, mesh1)
tests/parallel/test_atm_latlon_2d_tiling.py-683-
tests/parallel/test_atm_latlon_2d_tiling.py-684-    step2d = make_sharded_atm_latlon_step_2d(model, mesh2,
tests/parallel/test_atm_latlon_2d_tiling.py:685:                                             shard_geometry=True)
tests/parallel/test_atm_latlon_2d_tiling.py-686-    s2 = shard_state_atm_latlon_2d(state, mesh2)
tests/parallel/test_atm_latlon_2d_tiling.py-687-    for _ in range(n_steps):
tests/parallel/test_atm_latlon_2d_tiling.py-688-        s2 = step2d(s2, DT)
tests/parallel/test_atm_latlon_2d_tiling.py-689-    out2 = gather_state_atm_latlon_2d(s2, mesh2)
tests/parallel/test_atm_latlon_2d_tiling.py-690-
--
tests/parallel/test_atm_latlon_segment.py-16-      per-band non-finite presence over ALL state leaves) agrees with a
tests/parallel/test_atm_latlon_segment.py-17-      host-side isfinite of the gathered state — on a healthy run AND under
tests/parallel/test_atm_latlon_segment.py-18-      synthetic NaN injections in T (band 0) and u-only (band 1), so the gate
tests/parallel/test_atm_latlon_segment.py-19-      is provably non-vacuous and the cross-band psum is exercised.
tests/parallel/test_atm_latlon_segment.py-20-
tests/parallel/test_atm_latlon_segment.py:21:(iii) the geometry-SHARDED step (``shard_geometry=True``: per-device band
tests/parallel/test_atm_latlon_segment.py-22-      slice, ``P("lat")`` stacks) BIT-matches the replicated-geometry step
tests/parallel/test_atm_latlon_segment.py-23-      (the historical default) — with and without the polar-filter mask
tests/parallel/test_atm_latlon_segment.py-24-      stacks — and the layouts are REALLY different on device (sharding
tests/parallel/test_atm_latlon_segment.py-25-      introspection), so the bit-match cannot pass vacuously.
tests/parallel/test_atm_latlon_segment.py-26-
--
tests/parallel/test_atm_latlon_segment.py-266-    mesh = _mesh()
tests/parallel/test_atm_latlon_segment.py-267-    model_r, c0 = _model_and_state(use_polar_filter=use_polar_filter)
tests/parallel/test_atm_latlon_segment.py-268-    model_s, _ = _model_and_state(use_polar_filter=use_polar_filter)
tests/parallel/test_atm_latlon_segment.py-269-
tests/parallel/test_atm_latlon_segment.py-270-    step_rep = make_sharded_atm_latlon_step(model_r, mesh)  # default: replicated
tests/parallel/test_atm_latlon_segment.py:271:    step_shd = make_sharded_atm_latlon_step(model_s, mesh, shard_geometry=True)
tests/parallel/test_atm_latlon_segment.py-272-
tests/parallel/test_atm_latlon_segment.py-273-    # Non-vacuity: the two layouts are REALLY different on device.
tests/parallel/test_atm_latlon_segment.py-274-    assert step_rep._geom_stacks["area"].sharding.is_fully_replicated
tests/parallel/test_atm_latlon_segment.py-275-    assert not step_shd._geom_stacks["area"].sharding.is_fully_replicated
tests/parallel/test_atm_latlon_segment.py-276-    if use_polar_filter:
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-410-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-411-    ``replicated_per_device_bytes``: the historical layout — every device
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-412-    holds the full all-band stack of every ``LatLonGrid`` array field
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-413-    (:func:`atm_grid_array_field_names`; dominated by the five
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-414-    ``(n_lat, n_lon)`` 2-D fields ``lat2d, lon2d, f, dx, area``).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:415:    ``sharded_per_device_bytes``: the ``shard_geometry=True`` layout — each
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-416-    device holds only its own band's slice (exactly ``replicated /
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-417-    n_devices``; the stack leading dim is ``n_devices`` and bands are
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-418-    uniform).  Excludes the optional polar-filter mask stacks (two
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-419-    ``(n_lat,)``-scale vectors when the filter is on) — negligible next to
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-420-    the 2-D fields and absent in the default configs.
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-434-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-435-def _build_geometry_stacks(model, mesh, n_dev: int, shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-436-    """Stack every band's ``LatLonGrid`` array fields (+ the optional
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-437-    polar-filter masks) over a leading band axis and lay them out on ``mesh``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-438-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:439:    ``shard_geometry=False`` (the historical layout): every stack is
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-440-    ``device_put`` REPLICATED (``P()``) — each device holds ALL bands'
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-441-    geometry and the body indexes its own band at ``axis_index``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-442-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:443:    ``shard_geometry=True`` (M2b): every stack is sharded ``P("lat", ...)``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-444-    on the leading band axis — each device holds ONLY its own band's slice
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-445-    (leading extent 1 inside the shard_map body, static index ``[0]``).  The
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-446-    VALUES the body consumes are identical either way (the same band slice),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-447-    so the step numerics are bit-unchanged; only the residency changes
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-448-    (per-device geometry bytes drop by ``n_dev`` —
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-612-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-613-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-614-def _agree_spmd_entry(model, mesh, *, n_steps=None, segment_steps=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-615-                      compiled_segments=None, has_physics_fn=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-616-                      has_on_segment=None, has_phys_state=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:617:                      shard_geometry=None, where: str) -> None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-618-    """Agree EVERY rank-local input, as the FIRST statement of a public entry.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-619-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-620-    #1362 / codex rounds 2-3. Gating individual refusals was not enough: each
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-621-    public entry point performs several rank-local checks BEFORE reaching any
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-622-    collective -- ``n_steps`` validation, ``_check_2d_mesh``, the ``mesh is
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-849-    finally:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-850-        set_spmd_mesh(prev_mesh)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-851-        set_halo_backend(prev_backend, prev_topo)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-852-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-853-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:854:def make_sharded_atm_latlon_step(model, mesh, physics_fn=None, *,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-855-                                 shard_geometry: bool = False):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-856-    """Return ``step(c_state, dt) -> c_state`` running the C-grid hydrostatic atm
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-857-    step lat-band-SPMD over the 1-D ``"lat"`` mesh.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-858-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-859-    ``c_state`` is a ``CGridLatLonHydrostaticState`` laid out with
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-898-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-899-    # FIRST statement: agree every rank-local input before ANY
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-900-    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-901-    _agree_spmd_entry(model, mesh, n_steps=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-902-                      has_physics_fn=physics_fn is not None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:903:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-904-                      where="make_sharded_atm_latlon_step")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-905-    from legoesm.parallel.latlon_spmd import latlon_band_perms
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-906-    from legoesm.parallel.shard_map_compat import shard_map
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-907-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-908-    # Stochastic physics is SPMD-safe since increment 2: the Bechtold AR1
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1037-    ``all_finite`` is a REPLICATED traced scalar bool from
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1038-    :func:`state_finite_scalar` — the in-graph blowup guard (``psum`` of
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1039-    per-band non-finite presence over ALL state leaves).  The host reads
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1040-    this ONE scalar per segment instead of gathering the full state.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1041-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1042:    ``shard_geometry=True`` (default — a NEW API, no historical layout to
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1043-    preserve): each device holds ONLY its own band's geometry slice
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1044-    (``P("lat")`` stacks) instead of a replicated all-band copy —
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1045-    bit-identical numerics, 1/n_dev the geometry bytes
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1046-    (:func:`atm_latlon_geometry_bytes`).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1047-
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1071-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1072-    # FIRST statement: agree every rank-local input before ANY
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1073-    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1074-    _agree_spmd_entry(model, mesh, n_steps=n_steps,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1075-                      has_physics_fn=physics_fn is not None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1076:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1077-                      where="make_sharded_atm_latlon_segment")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1078-    from legoesm.parallel.latlon_spmd import latlon_band_perms
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1079-    from legoesm.parallel.shard_map_compat import shard_map
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1080-    from legoesm.timestepping.integration import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1081-        refuse_unthreaded_stateful_physics)
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1534-    """Stack every tile's ``LatLonGrid`` array fields (+ the optional
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1535-    polar-filter masks at ``p_lon == 1``) over LEADING ``(p_lat, p_lon)``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1536-    tile axes and lay them out on ``mesh`` — the 2-D twin of
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1537-    :func:`_build_geometry_stacks`.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1538-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1539:    ``shard_geometry=True``: stacks are sharded ``P("lat", "lon", ...)`` on
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1540-    the tile axes — each device holds ONLY its own tile's slice (leading
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1541-    extents ``(1, 1)`` inside the body, static index ``[0, 0]``).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1542:    ``shard_geometry=False``: replicated (``P()``) stacks, indexed at
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1543-    ``(axis_index("lat"), axis_index("lon"))``.  Same tile VALUES either way.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1544-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1545-    Returns ``(template, array_field_names, stacks, stacks_spec)``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1546-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1547-    grid = model.grid
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1679-            "make_latlon_2d_mpi_step DOES wire this via the AD-safe "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1680-            "lat-pencil transpose; the SPMD ppermute equivalent is a "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1681-            "follow-up.)  Use p_lon == 1 or disable the filter.")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1682-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1683-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1684:def make_sharded_atm_latlon_step_2d(model, mesh, physics_fn=None, *,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1685-                                    shard_geometry: bool = True):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1686-    """Return ``step(c_state, dt) -> c_state`` running the C-grid hydrostatic
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1687-    atm step 2-D-tile-SPMD over a ``("lat", "lon")`` mesh — the M3a native
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1688-    2-D tiling twin of :func:`make_sharded_atm_latlon_step`.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1689-
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1710-    with no collectives.  A stateful ``PhysicsState`` carry is REFUSED: its
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1711-    ``(ncol, ...)`` leaves flatten lat-major over the GLOBAL grid, so a
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1712-    contiguous dim-0 shard is a lat BAND's columns, not a 2-D tile's —
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1713-    thread carries through the 1-D :func:`make_sharded_atm_latlon_step`.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1714-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1715:    ``shard_geometry=True`` (default — new API, no historical layout):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1716-    per-device tile geometry slices (``P("lat", "lon")`` stacks);
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1717-    ``False`` replicates the all-tile stacks (indexed at the axis indices).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1718-    Same tile values either way (bit-identical numerics).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1719-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1720-    # FIRST statement: agree every rank-local input before ANY
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1721-    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1722-    _agree_spmd_entry(model, mesh, n_steps=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1723-                      has_physics_fn=physics_fn is not None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1724:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1725-                      where="make_sharded_atm_latlon_step_2d")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1726-    from legoesm.parallel.latlon_spmd import latlon_band_perms
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1727-    from legoesm.parallel.shard_map_compat import shard_map
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1728-    from legoesm.timestepping.integration import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1729-        refuse_unthreaded_stateful_physics)
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1788-    ``mesh=None`` -> the single-device compiled twin)."""
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1789-    # FIRST statement: agree every rank-local input before ANY
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1790-    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1791-    _agree_spmd_entry(model, mesh, n_steps=n_steps,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1792-                      has_physics_fn=physics_fn is not None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1793:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1794-                      where="make_sharded_atm_latlon_segment_2d")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1795-    from legoesm.parallel.latlon_spmd import latlon_band_perms
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1796-    from legoesm.parallel.shard_map_compat import shard_map
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1797-    from legoesm.timestepping.integration import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1798-        refuse_unthreaded_stateful_physics)
tests/parallel/test_atm_latlon_segment.py-15-(ii)  the segment's IN-GRAPH finite scalar (``state_finite_scalar``: psum of
tests/parallel/test_atm_latlon_segment.py-16-      per-band non-finite presence over ALL state leaves) agrees with a
tests/parallel/test_atm_latlon_segment.py-17-      host-side isfinite of the gathered state — on a healthy run AND under
tests/parallel/test_atm_latlon_segment.py-18-      synthetic NaN injections in T (band 0) and u-only (band 1), so the gate
tests/parallel/test_atm_latlon_segment.py-19-      is provably non-vacuous and the cross-band psum is exercised.
tests/parallel/test_atm_latlon_segment.py-20-
tests/parallel/test_atm_latlon_segment.py:21:(iii) the geometry-SHARDED step (``shard_geometry=True``: per-device band
tests/parallel/test_atm_latlon_segment.py-22-      slice, ``P("lat")`` stacks) BIT-matches the replicated-geometry step
tests/parallel/test_atm_latlon_segment.py-23-      (the historical default) — with and without the polar-filter mask
tests/parallel/test_atm_latlon_segment.py-24-      stacks — and the layouts are REALLY different on device (sharding
tests/parallel/test_atm_latlon_segment.py-25-      introspection), so the bit-match cannot pass vacuously.
tests/parallel/test_atm_latlon_segment.py-26-
tests/parallel/test_atm_latlon_segment.py-27-Plus the production-lane wiring gate: ``run_atm_latlon_spmd(...,
--
tests/parallel/test_atm_latlon_segment.py-265-def test_geometry_sharded_step_bitmatches_replicated(use_polar_filter):
tests/parallel/test_atm_latlon_segment.py-266-    mesh = _mesh()
tests/parallel/test_atm_latlon_segment.py-267-    model_r, c0 = _model_and_state(use_polar_filter=use_polar_filter)
tests/parallel/test_atm_latlon_segment.py-268-    model_s, _ = _model_and_state(use_polar_filter=use_polar_filter)
tests/parallel/test_atm_latlon_segment.py-269-
tests/parallel/test_atm_latlon_segment.py-270-    step_rep = make_sharded_atm_latlon_step(model_r, mesh)  # default: replicated
tests/parallel/test_atm_latlon_segment.py:271:    step_shd = make_sharded_atm_latlon_step(model_s, mesh, shard_geometry=True)
tests/parallel/test_atm_latlon_segment.py-272-
tests/parallel/test_atm_latlon_segment.py-273-    # Non-vacuity: the two layouts are REALLY different on device.
tests/parallel/test_atm_latlon_segment.py-274-    assert step_rep._geom_stacks["area"].sharding.is_fully_replicated
tests/parallel/test_atm_latlon_segment.py-275-    assert not step_shd._geom_stacks["area"].sharding.is_fully_replicated
tests/parallel/test_atm_latlon_segment.py-276-    if use_polar_filter:
tests/parallel/test_atm_latlon_segment.py-277-        assert "__polar_mask" in step_shd._geom_stacks
--
tests/parallel/test_atm_latlon_2d_tiling.py-348-
tests/parallel/test_atm_latlon_2d_tiling.py-349-    du_s, dv_s, dT_s, dps_s, _ = cgrid_latlon_hydrostatic_tendencies(
tests/parallel/test_atm_latlon_2d_tiling.py-350-        state, grid, sigma, cfg)
tests/parallel/test_atm_latlon_2d_tiling.py-351-    serial = tuple(np.asarray(x) for x in (du_s, dv_s, dT_s, dps_s))
tests/parallel/test_atm_latlon_2d_tiling.py-352-
tests/parallel/test_atm_latlon_2d_tiling.py-353-    template, afn, stacks, stacks_spec = _build_geometry_stacks_2d(
tests/parallel/test_atm_latlon_2d_tiling.py:354:        model, mesh, p_lat, p_lon, shard_geometry=True)
tests/parallel/test_atm_latlon_2d_tiling.py-355-    perm_north, _ = latlon_band_perms(p_lat)
tests/parallel/test_atm_latlon_2d_tiling.py-356-
tests/parallel/test_atm_latlon_2d_tiling.py-357-    def _body(sl, st):
tests/parallel/test_atm_latlon_2d_tiling.py-358-        tg = template._replace(**{n: st[n][0, 0] for n in afn})
tests/parallel/test_atm_latlon_2d_tiling.py-359-        vf = reconstruct_vface_lower(sl.v, "lat", perm_north)
tests/parallel/test_atm_latlon_2d_tiling.py-360-        uf = reconstruct_uface_left(sl.u, "lon", p_lon)
--
tests/parallel/test_atm_latlon_2d_tiling.py-665-# ==============================================================================
tests/parallel/test_atm_latlon_2d_tiling.py-666-
tests/parallel/test_atm_latlon_2d_tiling.py-667-def test_2d_degenerate_41_bitmatches_1d_band():
tests/parallel/test_atm_latlon_2d_tiling.py-668-    """On a (4,1) mesh every lon-ring op takes its static local branch, so
tests/parallel/test_atm_latlon_2d_tiling.py-669-    the 2-D step must reproduce the 1-D band step BIT-FOR-BIT (same ops, same
tests/parallel/test_atm_latlon_2d_tiling.py-670-    order — pure data-movement differences only).  Both factories run with
tests/parallel/test_atm_latlon_2d_tiling.py:671:    shard_geometry=True (the geometry VALUES are identical either way, gated
tests/parallel/test_atm_latlon_2d_tiling.py-672-    by the M2b bit-match)."""
tests/parallel/test_atm_latlon_2d_tiling.py-673-    mesh1 = _mesh1d(N_DEV)
tests/parallel/test_atm_latlon_2d_tiling.py-674-    mesh2 = _mesh2d(N_DEV, 1)
tests/parallel/test_atm_latlon_2d_tiling.py-675-    model, state = _model_and_state()
tests/parallel/test_atm_latlon_2d_tiling.py-676-    n_steps = 3
tests/parallel/test_atm_latlon_2d_tiling.py-677-
tests/parallel/test_atm_latlon_2d_tiling.py:678:    step1d = make_sharded_atm_latlon_step(model, mesh1, shard_geometry=True)
tests/parallel/test_atm_latlon_2d_tiling.py-679-    s1 = shard_state_atm_latlon(state, mesh1)
tests/parallel/test_atm_latlon_2d_tiling.py-680-    for _ in range(n_steps):
tests/parallel/test_atm_latlon_2d_tiling.py-681-        s1 = step1d(s1, DT)
tests/parallel/test_atm_latlon_2d_tiling.py-682-    out1 = gather_state_atm_latlon(s1, mesh1)
tests/parallel/test_atm_latlon_2d_tiling.py-683-
tests/parallel/test_atm_latlon_2d_tiling.py-684-    step2d = make_sharded_atm_latlon_step_2d(model, mesh2,
tests/parallel/test_atm_latlon_2d_tiling.py:685:                                             shard_geometry=True)
tests/parallel/test_atm_latlon_2d_tiling.py-686-    s2 = shard_state_atm_latlon_2d(state, mesh2)
tests/parallel/test_atm_latlon_2d_tiling.py-687-    for _ in range(n_steps):
tests/parallel/test_atm_latlon_2d_tiling.py-688-        s2 = step2d(s2, DT)
tests/parallel/test_atm_latlon_2d_tiling.py-689-    out2 = gather_state_atm_latlon_2d(s2, mesh2)
tests/parallel/test_atm_latlon_2d_tiling.py-690-
tests/parallel/test_atm_latlon_2d_tiling.py-691-    for f in ("u", "v", "T", "p_s"):
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-409-    the ACTUAL band-grid array shapes (no fabricated numbers).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-410-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-411-    ``replicated_per_device_bytes``: the historical layout — every device
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-412-    holds the full all-band stack of every ``LatLonGrid`` array field
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-413-    (:func:`atm_grid_array_field_names`; dominated by the five
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-414-    ``(n_lat, n_lon)`` 2-D fields ``lat2d, lon2d, f, dx, area``).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:415:    ``sharded_per_device_bytes``: the ``shard_geometry=True`` layout — each
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-416-    device holds only its own band's slice (exactly ``replicated /
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-417-    n_devices``; the stack leading dim is ``n_devices`` and bands are
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-418-    uniform).  Excludes the optional polar-filter mask stacks (two
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-419-    ``(n_lat,)``-scale vectors when the filter is on) — negligible next to
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-420-    the 2-D fields and absent in the default configs.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-421-    """
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-429-        "n_geometry_fields": len(names),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-430-        "replicated_per_device_bytes": int(total),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-431-        "sharded_per_device_bytes": int(total // n_devices),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-432-    }
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-433-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-434-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:435:def _build_geometry_stacks(model, mesh, n_dev: int, shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-436-    """Stack every band's ``LatLonGrid`` array fields (+ the optional
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-437-    polar-filter masks) over a leading band axis and lay them out on ``mesh``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-438-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:439:    ``shard_geometry=False`` (the historical layout): every stack is
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-440-    ``device_put`` REPLICATED (``P()``) — each device holds ALL bands'
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-441-    geometry and the body indexes its own band at ``axis_index``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-442-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:443:    ``shard_geometry=True`` (M2b): every stack is sharded ``P("lat", ...)``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-444-    on the leading band axis — each device holds ONLY its own band's slice
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-445-    (leading extent 1 inside the shard_map body, static index ``[0]``).  The
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-446-    VALUES the body consumes are identical either way (the same band slice),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-447-    so the step numerics are bit-unchanged; only the residency changes
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-448-    (per-device geometry bytes drop by ``n_dev`` —
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-449-    :func:`atm_latlon_geometry_bytes`).
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-490-                         arrays=[raw[n] for n in ordered_names])
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-491-    raw = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-492-        name: jnp.asarray(broadcast_checked(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-493-            raw[name], name, context="make_sharded_atm_latlon_step"))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-494-        for name in ordered_names
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-495-    }
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:496:    spec_of = lat_spec if shard_geometry else (lambda _arr: P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-497-    stacks = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-498-        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-499-        for name, arr in raw.items()
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-500-    }
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-501-    stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-502-    return template, array_field_names, stacks, stacks_spec
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-503-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-504-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-505-def _make_band_step_body(model, template, array_field_names, axis,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:506:                         perm_north, physics_fn, shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-507-    """One band's un-jitted C-grid step body — shared by the per-step
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-508-    shard_map (:func:`make_sharded_atm_latlon_step`) and the compiled segment
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-509-    scan (:func:`make_sharded_atm_latlon_segment`) so the band numerics are
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-510-    written ONCE.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-511-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-512-    Returns ``band_step(state_local, stacks_local, dt, ps_local) ->
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-513-    (state_out_local, ps_out)`` operating on the band-local ``v_lower``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:514:    layout.  ``shard_geometry`` selects the geometry index (STATIC Python
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-515-    bool, feature-gating exception): sharded stacks arrive with a leading
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-516-    extent of 1 (static index 0); replicated stacks carry all bands
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-517-    (dynamic index at ``axis_index``).  Same band values either way.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-518-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-519-    from legoesm.parallel.latlon_spmd import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-520-        reconstruct_vface_lower, to_vface_lower, spmd_pole_end_masks)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-521-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-522-    def band_step(state_local, stacks_local, dt, ps_local):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:523:        gi = 0 if shard_geometry else jax.lax.axis_index(axis)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-524-        band_geom = template._replace(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-525-            **{name: stacks_local[name][gi] for name in array_field_names})
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-526-        pmask = (stacks_local["__polar_mask"][gi]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-527-                 if "__polar_mask" in stacks_local else None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-528-        pmaskv = (stacks_local["__polar_mask_v"][gi]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-529-                  if "__polar_mask_v" in stacks_local else None)
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-604-_SPMD_ENTRY_FLAGS = (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-605-    "has_mesh", "n_dev", "n_axes", "axis_names", "axis_sizes", "p_lat",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-606-    "p_lon", "grid_n_lat", "grid_n_lon", "grid_schema",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-607-    "fold_active", "config_digest",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-608-    "has_polar_mask", "has_polar_mask_v",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-609-    "n_steps", "segment_steps", "compiled_segments",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:610:    "has_physics_fn", "has_on_segment", "has_phys_state", "shard_geometry",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-611-)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-612-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-613-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-614-def _agree_spmd_entry(model, mesh, *, n_steps=None, segment_steps=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-615-                      compiled_segments=None, has_physics_fn=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-616-                      has_on_segment=None, has_phys_state=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:617:                      shard_geometry=None, where: str) -> None:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-618-    """Agree EVERY rank-local input, as the FIRST statement of a public entry.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-619-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-620-    #1362 / codex rounds 2-3. Gating individual refusals was not enough: each
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-621-    public entry point performs several rank-local checks BEFORE reaching any
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-622-    collective -- ``n_steps`` validation, ``_check_2d_mesh``, the ``mesh is
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-623-    None`` early return, and ``build_band_grids_atm``'s divisibility check.
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-653-      lane. Two processes on different lanes run different programs.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-654-    * ``has_physics_fn`` / ``has_phys_state`` -- change the traced program and
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-655-      the carry contract.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-656-    * ``has_on_segment`` -- the callback triggers a per-segment
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-657-      ``gather_atm_latlon_to_hydrostatic``, itself a cross-process
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-658-      replication. A callback on some ranks only = unmatched gathers.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:659:    * ``shard_geometry`` -- changes the geometry stacks' ``PartitionSpec``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-660-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-661-    ``None`` for any of these encodes "not applicable at this call site" and
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-662-    maps to a fixed sentinel, so the payload WIDTH is set by
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-663-    ``_SPMD_ENTRY_FLAGS`` alone and never by rank-local data.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-664-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-665-    # Attribute reads are ALL defensive. The rule this enforces: any value
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-724-        _count(n_steps, "n_steps"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-725-        _count(segment_steps, "segment_steps"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-726-        _flag(compiled_segments, "compiled_segments"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-727-        _flag(has_physics_fn, "has_physics_fn"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-728-        _flag(has_on_segment, "has_on_segment"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-729-        _flag(has_phys_state, "has_phys_state"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:730:        _flag(shard_geometry, "shard_geometry"),
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-731-    )
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-732-    assert_flags_agree(_SPMD_ENTRY_FLAGS, flags, context=where)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-733-    # ONLY NOW may a bad value raise. Every process has entered AND LEFT the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-734-    # same collective above, and every process sees the same sentinel in its
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-735-    # own payload, so this refusal is symmetric by construction -- unlike the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-736-    # `int(n_steps)` that used to sit inside the payload build and could kill
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-849-    finally:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-850-        set_spmd_mesh(prev_mesh)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-851-        set_halo_backend(prev_backend, prev_topo)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-852-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-853-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-854-def make_sharded_atm_latlon_step(model, mesh, physics_fn=None, *,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:855:                                 shard_geometry: bool = False):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-856-    """Return ``step(c_state, dt) -> c_state`` running the C-grid hydrostatic atm
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-857-    step lat-band-SPMD over the 1-D ``"lat"`` mesh.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-858-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-859-    ``c_state`` is a ``CGridLatLonHydrostaticState`` laid out with
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-860-    :func:`shard_state_atm_latlon` (``v`` carried as the ``n_lat``-row
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-861-    ``v_lower``). The body reconstructs each band's ``nl+1`` v-faces, runs the
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-885-    ``prng_key (2,)`` and other non-column leaves replicate.  STOCHASTIC
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-886-    schemes are decomposition-INVARIANT since increment 2: the Bechtold AR1
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-887-    innovation folds the per-step sub-key with each column's GLOBAL id
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-888-    (``PhysicsState.col_index``, band-split with the carry), so a physical
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-889-    column draws the same variate under any decomposition.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-890-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:891:    ``shard_geometry`` (M2b, default ``False`` = historical layout,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-892-    byte-identical): ``True`` lays the band-geometry stacks out SHARDED
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-893-    ``P("lat")`` on the band axis — each device holds only its own band's
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-894-    geometry slice instead of a replicated all-band copy (1/n_dev the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-895-    bytes, :func:`atm_latlon_geometry_bytes`).  The body consumes the SAME
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-896-    band values either way, so the step is bit-identical (gated by
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-897-    ``tests/parallel/test_atm_latlon_segment.py``).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-898-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-899-    # FIRST statement: agree every rank-local input before ANY
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-900-    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-901-    _agree_spmd_entry(model, mesh, n_steps=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-902-                      has_physics_fn=physics_fn is not None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:903:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-904-                      where="make_sharded_atm_latlon_step")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-905-    from legoesm.parallel.latlon_spmd import latlon_band_perms
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-906-    from legoesm.parallel.shard_map_compat import shard_map
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-907-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-908-    # Stochastic physics is SPMD-safe since increment 2: the Bechtold AR1
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-909-    # innovation folds the per-step sub-key with each column's GLOBAL id
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-934-    axis = mesh.axis_names[0]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-935-    grid = model.grid
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-936-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-937-    _refuse_unsupported_spmd_config(model)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-938-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-939-    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:940:        model, mesh, n_dev, shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-941-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-942-    perm_north, _perm_south = latlon_band_perms(n_dev)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-943-    band_step = _make_band_step_body(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-944-        model, template, array_field_names, axis, perm_north, physics_fn,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:945:        shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-946-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-947-    def _body(state_local, stacks_local, dt):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-948-        out, _ = band_step(state_local, stacks_local, dt, None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-949-        return out
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-950-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-951-    def _body_with_carry(state_local, stacks_local, dt, ps_local):
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1016-    sharded_step._geom_stacks = stacks   # test/introspection only
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1017-    return sharded_step
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1018-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1019-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1020-def make_sharded_atm_latlon_segment(model, mesh, n_steps: int,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1021-                                    physics_fn=None, *,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1022:                                    shard_geometry: bool = True):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1023-    """Return ``segment(c_state, dt) -> (c_state, all_finite)`` advancing
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1024-    ``n_steps`` C-grid steps in ONE compiled program — a ``lax.scan`` of the
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1025-    band step inside a single jitted ``shard_map``, built once and reused
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1026-    (the M2b lever: "compile atmosphere lat-lon segments instead of
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1027-    launching one step at a time").
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1028-
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1036-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1037-    ``all_finite`` is a REPLICATED traced scalar bool from
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1038-    :func:`state_finite_scalar` — the in-graph blowup guard (``psum`` of
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1039-    per-band non-finite presence over ALL state leaves).  The host reads
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1040-    this ONE scalar per segment instead of gathering the full state.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1041-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1042:    ``shard_geometry=True`` (default — a NEW API, no historical layout to
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1043-    preserve): each device holds ONLY its own band's geometry slice
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1044-    (``P("lat")`` stacks) instead of a replicated all-band copy —
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1045-    bit-identical numerics, 1/n_dev the geometry bytes
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1046-    (:func:`atm_latlon_geometry_bytes`).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1047-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1048-    STATELESS physics only (``None`` / Held-Suarez / column-local closures,
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1070-    per-step lane — never a silent precision change.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1071-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1072-    # FIRST statement: agree every rank-local input before ANY
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1073-    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1074-    _agree_spmd_entry(model, mesh, n_steps=n_steps,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1075-                      has_physics_fn=physics_fn is not None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1076:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1077-                      where="make_sharded_atm_latlon_segment")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1078-    from legoesm.parallel.latlon_spmd import latlon_band_perms
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1079-    from legoesm.parallel.shard_map_compat import shard_map
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1080-    from legoesm.timestepping.integration import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1081-        refuse_unthreaded_stateful_physics)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1082-
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1118-    n_dev = mesh.devices.size
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1119-    axis = mesh.axis_names[0]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1120-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1121-    _refuse_unsupported_spmd_config(model)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1122-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1123-    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1124:        model, mesh, n_dev, shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1125-    perm_north, _perm_south = latlon_band_perms(n_dev)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1126-    band_step = _make_band_step_body(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1127-        model, template, array_field_names, axis, perm_north, physics_fn,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1128:        shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1129-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1130-    def _seg_body(state_local, stacks_local, dt):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1131-        def _step1(s):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1132-            out, _ps = band_step(s, stacks_local, dt, None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1133-            return out
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1134-        # Unroll to the scan-carry dtype fixed point (helper docstring).
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1527-        ]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1528-        for r in range(p_lat)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1529-    ]
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1530-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1531-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1532-def _build_geometry_stacks_2d(model, mesh, p_lat: int, p_lon: int,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1533:                              shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1534-    """Stack every tile's ``LatLonGrid`` array fields (+ the optional
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1535-    polar-filter masks at ``p_lon == 1``) over LEADING ``(p_lat, p_lon)``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1536-    tile axes and lay them out on ``mesh`` — the 2-D twin of
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1537-    :func:`_build_geometry_stacks`.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1538-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1539:    ``shard_geometry=True``: stacks are sharded ``P("lat", "lon", ...)`` on
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1540-    the tile axes — each device holds ONLY its own tile's slice (leading
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1541-    extents ``(1, 1)`` inside the body, static index ``[0, 0]``).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1542:    ``shard_geometry=False``: replicated (``P()``) stacks, indexed at
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1543-    ``(axis_index("lat"), axis_index("lon"))``.  Same tile VALUES either way.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1544-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1545-    Returns ``(template, array_field_names, stacks, stacks_spec)``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1546-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1547-    grid = model.grid
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1548-    tile_grids = build_tile_grids_atm_2d(grid, p_lat, p_lon)
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1584-                         arrays=[raw[n] for n in ordered_names])
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1585-    raw = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1586-        name: jnp.asarray(broadcast_checked(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1587-            raw[name], name, context="make_sharded_atm_latlon_step_2d"))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1588-        for name in ordered_names
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1589-    }
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1590:    spec_of = tile_spec if shard_geometry else (lambda _arr: P())
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1591-    stacks = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1592-        name: jax.device_put(arr, NamedSharding(mesh, spec_of(arr)))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1593-        for name, arr in raw.items()
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1594-    }
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1595-    stacks_spec = {name: spec_of(arr) for name, arr in raw.items()}
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1596-    return template, array_field_names, stacks, stacks_spec
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1597-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1598-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1599-def _make_tile_step_body_2d(model, template, array_field_names,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1600-                            perm_north, p_lon: int, physics_fn,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1601:                            shard_geometry: bool):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1602-    """One tile's un-jitted C-grid step body — the 2-D twin of
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1603-    :func:`_make_band_step_body`, shared by the per-step and segment 2-D
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1604-    factories so the tile numerics are written ONCE.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1605-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1606-    Returns ``tile_step(state_local, stacks_local, dt, ps_local) ->
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1607-    (state_out_local, ps_out)`` operating on the tile-local
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1611-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1612-    from legoesm.parallel.latlon_spmd import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1613-        reconstruct_uface_left, reconstruct_vface_lower,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1614-        spmd_pole_end_masks, to_uface_left, to_vface_lower)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1615-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1616-    def tile_step(state_local, stacks_local, dt, ps_local):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1617:        if shard_geometry:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1618-            gi, gj = 0, 0
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1619-        else:
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1620-            gi = jax.lax.axis_index("lat")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1621-            gj = jax.lax.axis_index("lon")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1622-        tile_geom = template._replace(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1623-            **{name: stacks_local[name][gi, gj]
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1679-            "make_latlon_2d_mpi_step DOES wire this via the AD-safe "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1680-            "lat-pencil transpose; the SPMD ppermute equivalent is a "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1681-            "follow-up.)  Use p_lon == 1 or disable the filter.")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1682-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1683-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1684-def make_sharded_atm_latlon_step_2d(model, mesh, physics_fn=None, *,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1685:                                    shard_geometry: bool = True):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1686-    """Return ``step(c_state, dt) -> c_state`` running the C-grid hydrostatic
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1687-    atm step 2-D-tile-SPMD over a ``("lat", "lon")`` mesh — the M3a native
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1688-    2-D tiling twin of :func:`make_sharded_atm_latlon_step`.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1689-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1690-    ``c_state`` is a ``CGridLatLonHydrostaticState`` laid out with
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1691-    :func:`shard_state_atm_latlon_2d` (``v`` as ``v_lower``, ``u`` as
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1709-    evaluated per RK stage on the TILE geometry — decomposition-invariant
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1710-    with no collectives.  A stateful ``PhysicsState`` carry is REFUSED: its
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1711-    ``(ncol, ...)`` leaves flatten lat-major over the GLOBAL grid, so a
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1712-    contiguous dim-0 shard is a lat BAND's columns, not a 2-D tile's —
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1713-    thread carries through the 1-D :func:`make_sharded_atm_latlon_step`.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1714-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1715:    ``shard_geometry=True`` (default — new API, no historical layout):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1716-    per-device tile geometry slices (``P("lat", "lon")`` stacks);
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1717-    ``False`` replicates the all-tile stacks (indexed at the axis indices).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1718-    Same tile values either way (bit-identical numerics).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1719-    """
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1720-    # FIRST statement: agree every rank-local input before ANY
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1721-    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1722-    _agree_spmd_entry(model, mesh, n_steps=None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1723-                      has_physics_fn=physics_fn is not None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1724:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1725-                      where="make_sharded_atm_latlon_step_2d")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1726-    from legoesm.parallel.latlon_spmd import latlon_band_perms
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1727-    from legoesm.parallel.shard_map_compat import shard_map
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1728-    from legoesm.timestepping.integration import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1729-        refuse_unthreaded_stateful_physics)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1730-
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1733-                                            physics_fn=physics_fn)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1734-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1735-    p_lat, p_lon = _check_2d_mesh(mesh)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1736-    _refuse_unsupported_spmd_config_2d(model, p_lon)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1737-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1738-    template, array_field_names, stacks, stacks_spec = (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1739:        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1740-    perm_north, _perm_south = latlon_band_perms(p_lat)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1741-    tile_step = _make_tile_step_body_2d(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1742-        model, template, array_field_names, perm_north, p_lon, physics_fn,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1743:        shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1744-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1745-    def _body(state_local, stacks_local, dt):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1746-        out, _ = tile_step(state_local, stacks_local, dt, None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1747-        return out
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1748-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1749-    _cache = {}
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1774-    sharded_step._geom_stacks = stacks   # test/introspection only
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1775-    return sharded_step
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1776-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1777-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1778-def make_sharded_atm_latlon_segment_2d(model, mesh, n_steps: int,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1779-                                       physics_fn=None, *,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1780:                                       shard_geometry: bool = True):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1781-    """Return ``segment(c_state, dt) -> (c_state, all_finite)`` advancing
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1782-    ``n_steps`` C-grid steps in ONE compiled ``lax.scan`` over the 2-D
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1783-    ``("lat", "lon")`` tile mesh — the M3a twin of
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1784-    :func:`make_sharded_atm_latlon_segment` (same contract: static
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1785-    ``n_steps``, replicated in-graph finite scalar psum'd over BOTH mesh
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1786-    axes, leading steps unrolled to the scan-carry dtype fixed point via
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1787-    :func:`unroll_to_dtype_fixed_point`, STATELESS physics only,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1788-    ``mesh=None`` -> the single-device compiled twin)."""
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1789-    # FIRST statement: agree every rank-local input before ANY
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1790-    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1791-    _agree_spmd_entry(model, mesh, n_steps=n_steps,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1792-                      has_physics_fn=physics_fn is not None,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1793:                      shard_geometry=shard_geometry,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1794-                      where="make_sharded_atm_latlon_segment_2d")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1795-    from legoesm.parallel.latlon_spmd import latlon_band_perms
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1796-    from legoesm.parallel.shard_map_compat import shard_map
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1797-    from legoesm.timestepping.integration import (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1798-        refuse_unthreaded_stateful_physics)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1799-
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1815-            raise NotImplementedError(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1816-                "make_sharded_atm_latlon_segment_2d: a stateful PhysicsState "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1817-                "carry is not 2-D-tile-routed — use the 1-D per-step "
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1818-                "make_sharded_atm_latlon_step(phys_state=...) path.")
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1819-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1820-    template, array_field_names, stacks, stacks_spec = (
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1821:        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1822-    perm_north, _perm_south = latlon_band_perms(p_lat)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1823-    tile_step = _make_tile_step_body_2d(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1824-        model, template, array_field_names, perm_north, p_lon, physics_fn,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1825:        shard_geometry)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1826-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1827-    def _seg_body(state_local, stacks_local, dt):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1828-        def _step1(s):
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1829-            out, _ps = tile_step(s, stacks_local, dt, None)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1830-            return out
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1831-        # Unroll to the scan-carry dtype fixed point (helper docstring).
--
tests/unit/test_geometry_consistency.py-1202-
tests/unit/test_geometry_consistency.py-1203-    @pytest.mark.parametrize("kwargs", [
tests/unit/test_geometry_consistency.py-1204-        {},
tests/unit/test_geometry_consistency.py-1205-        {"n_steps": 3},
tests/unit/test_geometry_consistency.py-1206-        {"n_steps": 3, "segment_steps": 1, "compiled_segments": True,
tests/unit/test_geometry_consistency.py-1207-         "has_physics_fn": True, "has_on_segment": False,
tests/unit/test_geometry_consistency.py:1208:         "has_phys_state": None, "shard_geometry": True},
tests/unit/test_geometry_consistency.py-1209-    ])
tests/unit/test_geometry_consistency.py-1210-    def test_payload_width_is_INDEPENDENT_of_which_flags_are_supplied(
tests/unit/test_geometry_consistency.py-1211-            self, capture_flags, kwargs):
tests/unit/test_geometry_consistency.py-1212-        """A width that varies per call site is a deadlock: two processes
tests/unit/test_geometry_consistency.py-1213-        entering the same allgather with different shapes."""
tests/unit/test_geometry_consistency.py-1214-        m = _atm_module()

exec
/usr/bin/bash -lc "sed -n '1,150p' packages/ocean/pyproject.toml; rg -n -C 4 'def _build_geometry_stacks|_build_geometry_stacks\\(' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py; rg -n -C 3 'broadcast_checked\\(' packages --glob '*.py'; rg -n -C 2 'from legoesm\\.parallel\\.geometry_consistency import \\*|import \\*' packages tests --glob '*.py' | sed -n '1,240p'; python3 -B -c 'from pathlib import Path; [compile(p.read_text(), str(p), \"exec\") for p in map(Path, [\"packages/core/legoesm/parallel/geometry_consistency.py\", \"packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py\", \"tests/ocean/unit/test_sharded_geom_fingerprint.py\"])]; print(\"syntax-ok\")'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 1 in 464ms:
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "legoesm-ocean"
version = "0.1.0"
description = "legoESM ocean component: MPAS/Voronoi + lat-lon C-grid primitive-equation and barotropic dynamics, EOS, vertical mixing, GM/Redi, biogeochemistry."
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
authors = [{ name = "Pierre Gentine" }]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Atmospheric Science",
]
# Depends ONLY on the substrate (legoesm-core) — NOT on the other Earth-system
# components, NOR on the coupler/driver/training above it.  Both invariants are
# machine-enforced by the import-linter contracts ("Earth-system components are
# independent of each other" + "Components must not import the coupler, driver,
# or training"), which is what lets `pip install legoesm-ocean` run standalone.
dependencies = [
    "legoesm-core~=0.1.0",
    "jax>=0.4.35",
    "jaxlib>=0.4.35",
    "numpy>=1.26",
    "equinox>=0.11",
]

# Optional heavy features that reach UP into the orchestration cluster.  The BASE
# ocean imports only legoesm-core (verified: `import legoesm.ocean` does not pull
# ml/tools), so the component stays standalone; these extras declare the install
# path for the specific modules that do:
#   * ml    — the SFNO neural ocean dycore (ocean.dynamics.sfno_ocean imports
#             legoesm.ml.{sfno,normalization,conservation}) at top level;
#   * tools — the Stommel-gyre tracer experiment's drift diagnostic
#             (ocean.experiments.stommel_gyre_tracer -> legoesm.diagnostics, deferred).
# These component->higher edges are baselined in import-linter contract #3.
[project.optional-dependencies]
ml = ["legoesm-ml~=0.1.0"]
tools = ["legoesm-tools~=0.1.0"]

[tool.uv.sources]
legoesm-core = { workspace = true }
legoesm-ml = { workspace = true }
legoesm-tools = { workspace = true }

# PEP-420 namespace package: no legoesm/__init__.py; this wheel ships only
# legoesm/ocean into the shared `legoesm` namespace, so it imports as legoesm.ocean.
[tool.hatch.build.targets.wheel]
packages = ["legoesm"]
431-        "sharded_per_device_bytes": int(total // n_devices),
432-    }
433-
434-
435:def _build_geometry_stacks(model, mesh, n_dev: int, shard_geometry: bool):
436-    """Stack every band's ``LatLonGrid`` array fields (+ the optional
437-    polar-filter masks) over a leading band axis and lay them out on ``mesh``.
438-
439-    ``shard_geometry=False`` (the historical layout): every stack is
--
935-    grid = model.grid
936-
937-    _refuse_unsupported_spmd_config(model)
938-
939:    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
940-        model, mesh, n_dev, shard_geometry)
941-
942-    perm_north, _perm_south = latlon_band_perms(n_dev)
943-    band_step = _make_band_step_body(
--
1119-    axis = mesh.axis_names[0]
1120-
1121-    _refuse_unsupported_spmd_config(model)
1122-
1123:    template, array_field_names, stacks, stacks_spec = _build_geometry_stacks(
1124-        model, mesh, n_dev, shard_geometry)
1125-    perm_north, _perm_south = latlon_band_perms(n_dev)
1126-    band_step = _make_band_step_body(
1127-        model, template, array_field_names, axis, perm_north, physics_fn,
--
1528-        for r in range(p_lat)
1529-    ]
1530-
1531-
1532:def _build_geometry_stacks_2d(model, mesh, p_lat: int, p_lon: int,
1533-                              shard_geometry: bool):
1534-    """Stack every tile's ``LatLonGrid`` array fields (+ the optional
1535-    polar-filter masks at ``p_lon == 1``) over LEADING ``(p_lat, p_lon)``
1536-    tile axes and lay them out on ``mesh`` — the 2-D twin of
packages/coupler/legoesm/driver/sharded_operator_split_step.py-480-                         arrays=[raw[n] for n in _ordered])
packages/coupler/legoesm/driver/sharded_operator_split_step.py-481-    stacks = {
packages/coupler/legoesm/driver/sharded_operator_split_step.py-482-        name: jax.device_put(
packages/coupler/legoesm/driver/sharded_operator_split_step.py:483:            jnp.asarray(broadcast_checked(
packages/coupler/legoesm/driver/sharded_operator_split_step.py-484-                raw[name], name,
packages/coupler/legoesm/driver/sharded_operator_split_step.py-485-                context="make_sharded_operator_split_step")),
packages/coupler/legoesm/driver/sharded_operator_split_step.py-486-            rep)
--
packages/core/legoesm/parallel/geometry_consistency.py-519-            f"assume one schema.")
packages/core/legoesm/parallel/geometry_consistency.py-520-
packages/core/legoesm/parallel/geometry_consistency.py-521-
packages/core/legoesm/parallel/geometry_consistency.py:522:def broadcast_checked(arr, name: str, *, context: str) -> np.ndarray:
packages/core/legoesm/parallel/geometry_consistency.py-523-    """Verify ``arr`` agrees across processes, then broadcast process 0's bytes.
packages/core/legoesm/parallel/geometry_consistency.py-524-
packages/core/legoesm/parallel/geometry_consistency.py-525-    Multi-process: returns a host ``np.ndarray`` that is bit-identical on
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-489-                         context="make_sharded_atm_latlon_step",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-490-                         arrays=[raw[n] for n in ordered_names])
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-491-    raw = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:492:        name: jnp.asarray(broadcast_checked(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-493-            raw[name], name, context="make_sharded_atm_latlon_step"))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-494-        for name in ordered_names
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-495-    }
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1583-                         context="make_sharded_atm_latlon_step_2d",
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1584-                         arrays=[raw[n] for n in ordered_names])
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1585-    raw = {
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1586:        name: jnp.asarray(broadcast_checked(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1587-            raw[name], name, context="make_sharded_atm_latlon_step_2d"))
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1588-        for name in ordered_names
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py-1589-    }
tests/test_cases/dcmip2025/test_case_1.py-1-"""Compatibility wrapper for relocated DCMIP-2025 test case 1."""
tests/test_cases/dcmip2025/test_case_1.py-2-
tests/test_cases/dcmip2025/test_case_1.py:3:from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_1 import *  # noqa: F401,F403
--
tests/test_cases/dcmip2025/common.py-1-"""Compatibility wrapper for relocated DCMIP-2025 shared utilities."""
tests/test_cases/dcmip2025/common.py-2-
tests/test_cases/dcmip2025/common.py:3:from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.common import *  # noqa: F401,F403
--
tests/test_cases/dcmip2025/test_case_2.py-1-"""Compatibility wrapper for relocated DCMIP-2025 test case 2."""
tests/test_cases/dcmip2025/test_case_2.py-2-
tests/test_cases/dcmip2025/test_case_2.py:3:from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_2 import *  # noqa: F401,F403
--
tests/test_cases/dcmip2025/test_case_3.py-1-"""Compatibility wrapper for relocated DCMIP-2025 test case 3."""
tests/test_cases/dcmip2025/test_case_3.py-2-
tests/test_cases/dcmip2025/test_case_3.py:3:from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_3 import *  # noqa: F401,F403
--
tests/test_cases/williamson.py-1-"""Compatibility wrapper for relocated Williamson shallow-water test cases."""
tests/test_cases/williamson.py-2-
tests/test_cases/williamson.py:3:from tests.atmosphere.shallow_water.test_cases.williamson import *  # noqa: F401,F403
--
tests/test_cases/dcmip_transport.py-1-"""Compatibility wrapper for relocated DCMIP-2012 transport test cases."""
tests/test_cases/dcmip_transport.py-2-
tests/test_cases/dcmip_transport.py:3:from tests.atmosphere.hydrostatic.test_cases.dcmip_transport import *  # noqa: F401,F403
tests/test_cases/dcmip_transport.py-4-from tests.atmosphere.hydrostatic.test_cases.dcmip_transport import (  # noqa: F401
tests/test_cases/dcmip_transport.py-5-    _height_from_sigma,
--
tests/test_cases/williamson_latlon.py-1-"""Compatibility wrapper for relocated lat-lon Williamson test cases."""
tests/test_cases/williamson_latlon.py-2-
tests/test_cases/williamson_latlon.py:3:from tests.atmosphere.shallow_water.test_cases.williamson_latlon import *  # noqa: F401,F403
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "<string>", line 1, in <listcomp>
  File "packages/core/legoesm/parallel/geometry_consistency.py", line 39
    from __future__ import annotations
    ^
SyntaxError: future feature annotations is not defined

exec
/usr/bin/bash -lc "python3 --version; /work/bd1083/b382650/bin/python3.12 --version; /work/bd1083/b382650/bin/python3.12 -B -c 'from pathlib import Path; ps=[Path(\"packages/core/legoesm/parallel/geometry_consistency.py\"), Path(\"packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py\"), Path(\"tests/ocean/unit/test_sharded_geom_fingerprint.py\")]; [compile(p.read_text(), str(p), \"exec\") for p in ps]; print(\"syntax-ok\")'; nl -ba packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py | sed -n '1020,1090p'; nl -ba packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py | sed -n '1780,1840p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 1967ms:
Python 3.6.8
Python 3.12.9
syntax-ok
  1020	def make_sharded_atm_latlon_segment(model, mesh, n_steps: int,
  1021	                                    physics_fn=None, *,
  1022	                                    shard_geometry: bool = True):
  1023	    """Return ``segment(c_state, dt) -> (c_state, all_finite)`` advancing
  1024	    ``n_steps`` C-grid steps in ONE compiled program — a ``lax.scan`` of the
  1025	    band step inside a single jitted ``shard_map``, built once and reused
  1026	    (the M2b lever: "compile atmosphere lat-lon segments instead of
  1027	    launching one step at a time").
  1028	
  1029	    Contrast with driving :func:`make_sharded_atm_latlon_step` in a Python
  1030	    loop: ONE host dispatch (+ halo-backend arm/restore + cache-key hash) per
  1031	    SEGMENT instead of per STEP, and no per-step host round-trip between
  1032	    device launches.  The scanned band body is the SAME
  1033	    ``_make_band_step_body`` the per-step path runs, so the trajectory
  1034	    matches the sequential sharded steps to compilation-order roundoff
  1035	    (gated at 1e-12 by ``tests/parallel/test_atm_latlon_segment.py``).
  1036	
  1037	    ``all_finite`` is a REPLICATED traced scalar bool from
  1038	    :func:`state_finite_scalar` — the in-graph blowup guard (``psum`` of
  1039	    per-band non-finite presence over ALL state leaves).  The host reads
  1040	    this ONE scalar per segment instead of gathering the full state.
  1041	
  1042	    ``shard_geometry=True`` (default — a NEW API, no historical layout to
  1043	    preserve): each device holds ONLY its own band's geometry slice
  1044	    (``P("lat")`` stacks) instead of a replicated all-band copy —
  1045	    bit-identical numerics, 1/n_dev the geometry bytes
  1046	    (:func:`atm_latlon_geometry_bytes`).
  1047	
  1048	    STATELESS physics only (``None`` / Held-Suarez / column-local closures,
  1049	    the production ``run_atm_latlon_spmd`` envelope): a stateful
  1050	    ``PhysicsState`` carry is refused loudly — thread it through the
  1051	    per-step :func:`make_sharded_atm_latlon_step` until the segment lane
  1052	    routes the carry through the scan.
  1053	
  1054	    ``mesh=None``: the single-device twin — ``jit(lax.scan)`` over the serial
  1055	    C-grid step with the model's own geometry, same ``(state, all_finite)``
  1056	    contract.
  1057	
  1058	    ``n_steps`` is STATIC (the compiled scan length): one compiled program
  1059	    per distinct segment length (``run_atm_latlon_spmd`` caches per length —
  1060	    at most two: the regular segment and the final remainder).
  1061	
  1062	    Carry dtype: leading steps are UNROLLED outside the ``lax.scan`` until
  1063	    the state's dtype signature is a fixed point of the step
  1064	    (:func:`unroll_to_dtype_fixed_point` — ``jax.eval_shape`` probe, zero
  1065	    FLOPs, trace-time constant).  A mixed-precision IC promotes over the
  1066	    first stepS (``p_s`` first, ``u/v/T`` next via the promoted ``p_s`` —
  1067	    observed jobs 8916406/8916740) exactly as the per-step Python loop
  1068	    absorbs silently; an already-stable state unrolls NOTHING and scans all
  1069	    ``n_steps``.  Zero extra casts, zero numerical difference vs the
  1070	    per-step lane — never a silent precision change.
  1071	    """
  1072	    # FIRST statement: agree every rank-local input before ANY
  1073	    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
  1074	    _agree_spmd_entry(model, mesh, n_steps=n_steps,
  1075	                      has_physics_fn=physics_fn is not None,
  1076	                      shard_geometry=shard_geometry,
  1077	                      where="make_sharded_atm_latlon_segment")
  1078	    from legoesm.parallel.latlon_spmd import latlon_band_perms
  1079	    from legoesm.parallel.shard_map_compat import shard_map
  1080	    from legoesm.timestepping.integration import (
  1081	        refuse_unthreaded_stateful_physics)
  1082	
  1083	    if int(n_steps) < 1:
  1084	        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
  1085	    n_steps = int(n_steps)
  1086	
  1087	    def _refuse_carry(phys_state):
  1088	        refuse_unthreaded_stateful_physics(
  1089	            physics_fn, phys_state, where="atm lat-lon compiled segment")
  1090	        if phys_state is not None:
  1780	                                       shard_geometry: bool = True):
  1781	    """Return ``segment(c_state, dt) -> (c_state, all_finite)`` advancing
  1782	    ``n_steps`` C-grid steps in ONE compiled ``lax.scan`` over the 2-D
  1783	    ``("lat", "lon")`` tile mesh — the M3a twin of
  1784	    :func:`make_sharded_atm_latlon_segment` (same contract: static
  1785	    ``n_steps``, replicated in-graph finite scalar psum'd over BOTH mesh
  1786	    axes, leading steps unrolled to the scan-carry dtype fixed point via
  1787	    :func:`unroll_to_dtype_fixed_point`, STATELESS physics only,
  1788	    ``mesh=None`` -> the single-device compiled twin)."""
  1789	    # FIRST statement: agree every rank-local input before ANY
  1790	    # rank-local check can raise or return (codex round-2 blocker 1/2/3).
  1791	    _agree_spmd_entry(model, mesh, n_steps=n_steps,
  1792	                      has_physics_fn=physics_fn is not None,
  1793	                      shard_geometry=shard_geometry,
  1794	                      where="make_sharded_atm_latlon_segment_2d")
  1795	    from legoesm.parallel.latlon_spmd import latlon_band_perms
  1796	    from legoesm.parallel.shard_map_compat import shard_map
  1797	    from legoesm.timestepping.integration import (
  1798	        refuse_unthreaded_stateful_physics)
  1799	
  1800	    if int(n_steps) < 1:
  1801	        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
  1802	    n_steps = int(n_steps)
  1803	
  1804	    if mesh is None:                       # single-device compiled segment
  1805	        return make_sharded_atm_latlon_segment(model, None, n_steps,
  1806	                                               physics_fn=physics_fn)
  1807	
  1808	    p_lat, p_lon = _check_2d_mesh(mesh)
  1809	    _refuse_unsupported_spmd_config_2d(model, p_lon)
  1810	
  1811	    def _refuse_carry(phys_state):
  1812	        refuse_unthreaded_stateful_physics(
  1813	            physics_fn, phys_state, where="atm lat-lon 2-D compiled segment")
  1814	        if phys_state is not None:
  1815	            raise NotImplementedError(
  1816	                "make_sharded_atm_latlon_segment_2d: a stateful PhysicsState "
  1817	                "carry is not 2-D-tile-routed — use the 1-D per-step "
  1818	                "make_sharded_atm_latlon_step(phys_state=...) path.")
  1819	
  1820	    template, array_field_names, stacks, stacks_spec = (
  1821	        _build_geometry_stacks_2d(model, mesh, p_lat, p_lon, shard_geometry))
  1822	    perm_north, _perm_south = latlon_band_perms(p_lat)
  1823	    tile_step = _make_tile_step_body_2d(
  1824	        model, template, array_field_names, perm_north, p_lon, physics_fn,
  1825	        shard_geometry)
  1826	
  1827	    def _seg_body(state_local, stacks_local, dt):
  1828	        def _step1(s):
  1829	            out, _ps = tile_step(s, stacks_local, dt, None)
  1830	            return out
  1831	        # Unroll to the scan-carry dtype fixed point (helper docstring).
  1832	        # Public name (the _-prefixed original was promoted; the 2-D path
  1833	        # kept the stale private reference — NameError on first segment
  1834	        # trace, caught by test_2d_segment_matches_sequential_and_serial).
  1835	        out, n_left = unroll_to_dtype_fixed_point(
  1836	            _step1, state_local, n_steps)
  1837	        if n_left > 0:
  1838	            out, _ = jax.lax.scan(lambda s, _x: (_step1(s), None),
  1839	                                  out, xs=None, length=n_left)
  1840	        return out, state_finite_scalar(out, axis=("lat", "lon"))

exec
/usr/bin/bash -lc "rg -n -C 5 'class (FreshwaterForcing|OceanSurfaceForcing|SpongeForcing)|FreshwaterForcing\\s*=|OceanSurfaceForcing\\s*=|SpongeForcing\\s*=' packages/ocean/legoesm/ocean; rg -n -C 3 'shard_forcing_latlon\\(' packages tests scripts --glob '*.py'; rg -n -C 3 'shard_state_latlon\\(' packages tests scripts --glob '*.py' | sed -n '1,500p'; rg -n -C 3 't_seconds=' packages/ocean tests/ocean scripts/run scripts/bench --glob '*.py' | sed -n '1,450p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 1197ms:
packages/ocean/legoesm/ocean/sponge.py-20-
packages/ocean/legoesm/ocean/sponge.py-21-import jax.numpy as jnp
packages/ocean/legoesm/ocean/sponge.py-22-import numpy as np
packages/ocean/legoesm/ocean/sponge.py-23-
packages/ocean/legoesm/ocean/sponge.py-24-
packages/ocean/legoesm/ocean/sponge.py:25:class SpongeForcing(NamedTuple):
packages/ocean/legoesm/ocean/sponge.py-26-    """Sponge layer relaxation fields.
packages/ocean/legoesm/ocean/sponge.py-27-
packages/ocean/legoesm/ocean/sponge.py-28-    Parameters
packages/ocean/legoesm/ocean/sponge.py-29-    ----------
packages/ocean/legoesm/ocean/sponge.py-30-    gamma : array
--
packages/ocean/legoesm/ocean/freshwater.py-22-from typing import NamedTuple
packages/ocean/legoesm/ocean/freshwater.py-23-
packages/ocean/legoesm/ocean/freshwater.py-24-import jax.numpy as jnp
packages/ocean/legoesm/ocean/freshwater.py-25-
packages/ocean/legoesm/ocean/freshwater.py-26-
packages/ocean/legoesm/ocean/freshwater.py:27:class FreshwaterForcing(NamedTuple):
packages/ocean/legoesm/ocean/freshwater.py-28-    """Freshwater fluxes applied to the ocean surface.
packages/ocean/legoesm/ocean/freshwater.py-29-
packages/ocean/legoesm/ocean/freshwater.py-30-    All fields have shape (nCells,) and units kg/m²/s.
packages/ocean/legoesm/ocean/freshwater.py-31-    Positive = freshwater into ocean, except evaporation which is
packages/ocean/legoesm/ocean/freshwater.py-32-    positive upward (i.e., freshwater leaving ocean).
--
packages/ocean/legoesm/ocean/state.py-78-    dland_mask_dt: Field
packages/ocean/legoesm/ocean/state.py-79-    K_v: object = None   # tracer diffusivity at interfaces [m²/s]
packages/ocean/legoesm/ocean/state.py-80-    A_v: object = None   # momentum viscosity at interfaces [m²/s]
packages/ocean/legoesm/ocean/state.py-81-
packages/ocean/legoesm/ocean/state.py-82-
packages/ocean/legoesm/ocean/state.py:83:class OceanSurfaceForcing(NamedTuple):
packages/ocean/legoesm/ocean/state.py-84-    """External atmospheric/surface forcing data for ocean physics.
packages/ocean/legoesm/ocean/state.py-85-
packages/ocean/legoesm/ocean/state.py-86-    Carries coupler-provided fields into the ocean physics pipeline.
packages/ocean/legoesm/ocean/state.py-87-    All fields are optional (None means not available).  Shape of 2D
packages/ocean/legoesm/ocean/state.py-88-    fields matches the horizontal grid; 3D fields add a level axis.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-374-    return state._replace(**updates)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-375-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-376-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:377:def shard_forcing_latlon(forcing, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-378-    """Lay out a forcing pytree (``FreshwaterForcing`` / ``OceanSurfaceForcing``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-379-    / ``SpongeForcing`` — or any nesting of them) for the lat-band SPMD step.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-380-
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-999-        # multicontroller, the nd-linear wall this module removes).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1000-        ss = shard_state_latlon(state, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1001-        ss = inner(ss, dt,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:1002:                   surface_forcing=shard_forcing_latlon(surface_forcing, mesh),
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:1003:                   freshwater=shard_forcing_latlon(freshwater, mesh))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1004-        return gather_state_latlon(ss, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1005-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1006-    return sharded_step_global
--
tests/parallel/test_latlon_ocean_spmd_step.py-246-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py-247-    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-248-    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:249:    fws = shard_forcing_latlon(fw, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:250:    sfs = shard_forcing_latlon(sf, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:251:    sponges = shard_forcing_latlon(sponge, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-252-
tests/parallel/test_latlon_ocean_spmd_step.py-253-    # Cache-key flip smoke: dynamics-only compile first, then the forcing
tests/parallel/test_latlon_ocean_spmd_step.py-254-    # program — the second call MUST rebuild (different forcing structure),
tests/unit/test_run_omip_latlon_spmd.py-80-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py-81-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py-82-    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py:83:    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-84-    ckpt_dir = tmp_path / "spmd"
tests/unit/test_run_omip_latlon_spmd.py-85-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-86-        model, ss0, checkpoint_dir=ckpt_dir,
--
tests/unit/test_run_omip_latlon_spmd.py-218-    dev = create_latlon_mesh(n_devices=2)
tests/unit/test_run_omip_latlon_spmd.py-219-    spmd_step = make_sharded_ocean_step(model, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-220-    model.prime_step_caches(state0)
tests/unit/test_run_omip_latlon_spmd.py:221:    ss0 = shard_state_latlon(state0, dev.mesh)
tests/unit/test_run_omip_latlon_spmd.py-222-    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
tests/unit/test_run_omip_latlon_spmd.py-223-        model, ss0, jra55_state=_fresh_js(),
tests/unit/test_run_omip_latlon_spmd.py-224-        spmd_step=spmd_step,
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-513-        mesh = jax.sharding.Mesh(np.array(jax.devices()[:nd]),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-514-                                 axis_names=("lat",))
scripts/bench/bench_ocean_latlon_spmd_scaling.py-515-        step = make_sharded_ocean_step(model, mesh)
scripts/bench/bench_ocean_latlon_spmd_scaling.py:516:        s = shard_state_latlon(s0, mesh)
scripts/bench/bench_ocean_latlon_spmd_scaling.py-517-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-518-    # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
scripts/bench/bench_ocean_latlon_spmd_scaling.py-519-    # blocks with sync only AROUND the block — the previous per-step
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-288-    ), context=where)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-289-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-290-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:291:def shard_state_latlon(state, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-292-    """Lay out a ``LatLonCGridOceanState`` for the lat-band SPMD step.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-293-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-294-    Cell / u-grid array fields (leading dim ``n_lat``) shard ``P("lat")``.  The
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-997-        # it forwarded it global and relied on implicit JIT input
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-998-        # placement — jax's whole-array device_put assert under
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-999-        # multicontroller, the nd-linear wall this module removes).
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:1000:        ss = shard_state_latlon(state, mesh)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1001-        ss = inner(ss, dt,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1002-                   surface_forcing=shard_forcing_latlon(surface_forcing, mesh),
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-1003-                   freshwater=shard_forcing_latlon(freshwater, mesh))
--
scripts/run/run_omip_core2.py-5584-                                       freshwater=fw))
scripts/run/run_omip_core2.py-5585-
scripts/run/run_omip_core2.py-5586-            def _pers_shard_fn(st, _mesh=_spmd_mesh):
scripts/run/run_omip_core2.py:5587:                return shard_state_latlon(st, _mesh)
scripts/run/run_omip_core2.py-5588-
scripts/run/run_omip_core2.py-5589-            def _pers_gather_fn(st, _mesh=_spmd_mesh):
scripts/run/run_omip_core2.py-5590-                return gather_state_latlon(st, _mesh)
--
tests/parallel/test_latlon_ocean_spmd_step.py-128-    # row reappended) for the bit-comparison vs the single-device reference.
tests/parallel/test_latlon_ocean_spmd_step.py-129-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py-130-    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:131:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-132-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-133-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_step.py-134-    ss = gather_state_latlon(ss, dev.mesh)
--
tests/parallel/test_latlon_ocean_spmd_step.py-245-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-246-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py-247-    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:248:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-249-    fws = shard_forcing_latlon(fw, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-250-    sfs = shard_forcing_latlon(sf, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-251-    sponges = shard_forcing_latlon(sponge, dev.mesh)
--
tests/parallel/test_latlon_ocean_spmd_step.py-397-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_step.py-398-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_step.py-399-    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:400:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-401-    bad = OceanSurfaceForcing(
tests/parallel/test_latlon_ocean_spmd_step.py-402-        tau_y=jnp.zeros((grid.n_lat + 1, grid.n_lon)))
tests/parallel/test_latlon_ocean_spmd_step.py-403-    with pytest.raises(ValueError, match="leading dim"):
--
tests/parallel/test_latlon_ocean_spmd_step.py-452-
tests/parallel/test_latlon_ocean_spmd_step.py-453-    # explicit scatter -> inner sharded step -> gather
tests/parallel/test_latlon_ocean_spmd_step.py-454-    inner = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py:455:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-456-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_step.py-457-        ss = inner(ss, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_step.py-458-    ss = gather_state_latlon(ss, dev.mesh)
--
tests/parallel/test_latlon_spmd_fused_halo.py-347-    for flag in ("0", "1"):
tests/parallel/test_latlon_spmd_fused_halo.py-348-        _env_flag(monkeypatch, flag)
tests/parallel/test_latlon_spmd_fused_halo.py-349-        step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py:350:        ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-351-        hlo_counts[flag] = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-352-            jax.jit(step).lower(ss, 600.0).compile().as_text())
tests/parallel/test_latlon_spmd_fused_halo.py-353-        for _ in range(n_steps):
--
tests/parallel/test_latlon_spmd_fused_halo.py-370-    # driver calls step() directly each step).
tests/parallel/test_latlon_spmd_fused_halo.py-371-    _env_flag(monkeypatch, "0")
tests/parallel/test_latlon_spmd_fused_halo.py-372-    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py:373:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_spmd_fused_halo.py-374-    n_off = _count_ppermutes(
tests/parallel/test_latlon_spmd_fused_halo.py-375-        jax.jit(lambda s, d: step(s, d)).lower(ss, 600.0)
tests/parallel/test_latlon_spmd_fused_halo.py-376-        .compile().as_text())
--
tests/parallel/test_persistent_sharded_ocean_loop.py-199-    # ---------------- NEW lane: persistent sharded loop ----------------------
tests/parallel/test_persistent_sharded_ocean_loop.py-200-    calls["shard"] = calls["gather"] = 0
tests/parallel/test_persistent_sharded_ocean_loop.py-201-    inner = sos.make_sharded_ocean_step(model, mesh)
tests/parallel/test_persistent_sharded_ocean_loop.py:202:    ss = sos.shard_state_latlon(state0, mesh)          # ONE initial shard
tests/parallel/test_persistent_sharded_ocean_loop.py-203-    for k in range(1, n_steps + 1):
tests/parallel/test_persistent_sharded_ocean_loop.py-204-        # surface-current consumer on the SHARDED state: v is the n_lat-row
tests/parallel/test_persistent_sharded_ocean_loop.py-205-        # v_lower carrier at every loop top (the snapshot boundary re-shards
--
tests/parallel/test_persistent_sharded_ocean_loop.py-233-            ss = sos.gather_state_latlon(ss, mesh)
tests/parallel/test_persistent_sharded_ocean_loop.py-234-            v_out = np.asarray(ss.v.data)
tests/parallel/test_persistent_sharded_ocean_loop.py-235-            assert v_out.shape[0] == n_lat + 1
tests/parallel/test_persistent_sharded_ocean_loop.py:236:            ss = sos.shard_state_latlon(ss, mesh)      # lazy re-shard
tests/parallel/test_persistent_sharded_ocean_loop.py-237-        if k == ice_bc_step:
tests/parallel/test_persistent_sharded_ocean_loop.py-238-            # ice-thermo-style BC on the SHARDED state (2-D slice pull +
tests/parallel/test_persistent_sharded_ocean_loop.py-239-            # device-side .at[..., 0].set scatter — the production pattern).
--
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-123-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-124-
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-125-    step = make_sharded_ocean_step(model, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py:126:    ss = shard_state_latlon(state0, mesh)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-127-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-128-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_multicontroller.py-129-    out = gather_state_latlon(ss, mesh)
--
scripts/run/run_omip.py-4844-            _dev = create_latlon_mesh(n_devices=_nd)
scripts/run/run_omip.py-4845-            spmd_step = make_sharded_ocean_step(model, _dev.mesh)
scripts/run/run_omip.py-4846-            spmd_gather = partial(gather_state_latlon, mesh=_dev.mesh)
scripts/run/run_omip.py:4847:            state = shard_state_latlon(state, _dev.mesh)
scripts/run/run_omip.py-4848-            # Lay per-block forcing stacks out lat-band-sharded so the
scripts/run/run_omip.py-4849-            # in-scan interpolation / bulk fluxes stay shard-local (shared
scripts/run/run_omip.py-4850-            # layout helper — see shard_forcing_stack_latlon).
--
tests/parallel/test_latlon_ocean_spmd_tripole.py-150-
tests/parallel/test_latlon_ocean_spmd_tripole.py-151-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_tripole.py-152-    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py:153:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_tripole.py-154-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_tripole.py-155-        ss = step(ss, dt, surface_forcing=sf)
tests/parallel/test_latlon_ocean_spmd_tripole.py-156-    ss = gather_state_latlon(ss, dev.mesh)
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-90-
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-91-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-92-    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:93:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-94-    for _ in range(n_steps):
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-95-        ss = step(ss, dt)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-96-    ss = gather_state_latlon(ss, dev.mesh)
--
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-123-    model._ensure_vertex_mask(state0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-124-    dev = create_latlon_mesh(n_devices=4)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-125-    step = make_sharded_ocean_step(model, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:126:    ss = shard_state_latlon(state0, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-127-    lowered = jax.jit(step).lower(ss, 600.0)
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-128-    return lowered.compile().as_text()
tests/parallel/test_latlon_ocean_spmd_wide_halo.py-129-
scripts/bench/bench_ocean_gpu_scaling.py-202-        # Encode solver tag in physics_level (otherwise unused for ocean)
scripts/bench/bench_ocean_gpu_scaling.py-203-        # so CSVs from different solvers stay distinguishable when pooled.
scripts/bench/bench_ocean_gpu_scaling.py-204-        physics_level=f"baro={solver_tag}",
scripts/bench/bench_ocean_gpu_scaling.py:205:        dt_seconds=DT_BAROCLINIC, n_warmup=N_WARMUP, n_timing=N_TIMING,
scripts/bench/bench_ocean_gpu_scaling.py-206-        compile_time_s=compile_s, warmup_time_s=warmup_s,
scripts/bench/bench_ocean_gpu_scaling.py-207-        timing_time_s=timing_s, time_per_step_ms=ms, sypd=sypd,
scripts/bench/bench_ocean_gpu_scaling.py-208-        total_cells=total, cells_per_gpu=total,
--
scripts/bench/run_cpu_mpi_scaling.py-1293-        mode=mode,
scripts/bench/run_cpu_mpi_scaling.py-1294-        grid_type=grid_type,
scripts/bench/run_cpu_mpi_scaling.py-1295-        physics_level=physics_level,
scripts/bench/run_cpu_mpi_scaling.py:1296:        dt_seconds=dt_used,
scripts/bench/run_cpu_mpi_scaling.py-1297-        n_warmup=n_warmup,
scripts/bench/run_cpu_mpi_scaling.py-1298-        n_timing=n_timing,
scripts/bench/run_cpu_mpi_scaling.py-1299-        compile_time_s=compile_time,
--
scripts/bench/bench_ocean_mpi_scaling.py-2577-        precision=args.precision,
scripts/bench/bench_ocean_mpi_scaling.py-2578-        mode=mode_label,
scripts/bench/bench_ocean_mpi_scaling.py-2579-        physics_level=f"baro={args.baro_solver}",
scripts/bench/bench_ocean_mpi_scaling.py:2580:        dt_seconds=args.dt,
scripts/bench/bench_ocean_mpi_scaling.py-2581-        n_warmup=args.n_warmup,
scripts/bench/bench_ocean_mpi_scaling.py-2582-        n_timing=args.n_timing,
scripts/bench/bench_ocean_mpi_scaling.py-2583-        compile_time_s=compile_s,
--
scripts/bench/bench_atm_latlon_spmd_scaling.py-452-        physics_level=args.physics,
scripts/bench/bench_atm_latlon_spmd_scaling.py-453-        backend=jax.default_backend(),
scripts/bench/bench_atm_latlon_spmd_scaling.py-454-        **tidy_throughput_fields(
scripts/bench/bench_atm_latlon_spmd_scaling.py:455:            dt_seconds=args.dt, time_per_step_ms=med,
scripts/bench/bench_atm_latlon_spmd_scaling.py-456-            total_cells=n_lat * args.n_lon * args.nlev),
scripts/bench/bench_atm_latlon_spmd_scaling.py-457-    )
scripts/bench/bench_atm_latlon_spmd_scaling.py-458-    # Increment-2 accounting fields (audit items 4/8), flat for aggregators.
--
scripts/bench/bench_mpas_spmd_scaling.py-536-        physics_level=args.physics,
scripts/bench/bench_mpas_spmd_scaling.py-537-        backend=jax.default_backend(),
scripts/bench/bench_mpas_spmd_scaling.py-538-        **tidy_throughput_fields(
scripts/bench/bench_mpas_spmd_scaling.py:539:            dt_seconds=dt, time_per_step_ms=med,
scripts/bench/bench_mpas_spmd_scaling.py-540-            total_cells=int(mesh.nCells) * args.nlev),
scripts/bench/bench_mpas_spmd_scaling.py-541-    )
scripts/bench/bench_mpas_spmd_scaling.py-542-    rec["metadata"] = annotate_incomplete(scaling_metadata(
--
scripts/bench/bench_crm_gpu_scaling.py-222-        precision=prec, mode="crm_plane_strong",
scripts/bench/bench_crm_gpu_scaling.py-223-        physics_level=(f"f-plane_dx{int(dx)}m_L{nlev}_"
scripts/bench/bench_crm_gpu_scaling.py-224-                       f"nsub{n_acoustic_substeps}_{acoustic_tag}_dt{dt}"),
scripts/bench/bench_crm_gpu_scaling.py:225:        dt_seconds=dt, n_warmup=N_WARMUP, n_timing=N_TIMING,
scripts/bench/bench_crm_gpu_scaling.py-226-        compile_time_s=compile_s, warmup_time_s=warmup_s,
scripts/bench/bench_crm_gpu_scaling.py-227-        timing_time_s=timing_s, time_per_step_ms=ms, sypd=sypd,
scripts/bench/bench_crm_gpu_scaling.py-228-        total_cells=total, cells_per_gpu=total,
--
scripts/bench/bench_cube_tiled_step_scaling.py-379-        physics_level="none",
scripts/bench/bench_cube_tiled_step_scaling.py-380-        backend=jax.default_backend(),
scripts/bench/bench_cube_tiled_step_scaling.py-381-        **tidy_throughput_fields(
scripts/bench/bench_cube_tiled_step_scaling.py:382:            dt_seconds=args.dt, time_per_step_ms=med,
scripts/bench/bench_cube_tiled_step_scaling.py-383-            total_cells=total_cells),
scripts/bench/bench_cube_tiled_step_scaling.py-384-    )
scripts/bench/bench_cube_tiled_step_scaling.py-385-    from legoesm.parallel.early_init import nccl_transport_report
--
scripts/bench/run_levante_gpu_scaling.py-1367-        precision=precision,
scripts/bench/run_levante_gpu_scaling.py-1368-        mode=mode,
scripts/bench/run_levante_gpu_scaling.py-1369-        physics_level=physics_level,
scripts/bench/run_levante_gpu_scaling.py:1370:        dt_seconds=dt_used,
scripts/bench/run_levante_gpu_scaling.py-1371-        n_warmup=n_warmup,
scripts/bench/run_levante_gpu_scaling.py-1372-        n_timing=n_timing,
scripts/bench/run_levante_gpu_scaling.py-1373-        compile_time_s=compile_time,
--
scripts/bench/run_levante_gpu_scaling.py-1980-        precision=precision,
scripts/bench/run_levante_gpu_scaling.py-1981-        mode=mode,
scripts/bench/run_levante_gpu_scaling.py-1982-        physics_level=physics_level,
scripts/bench/run_levante_gpu_scaling.py:1983:        dt_seconds=dt,
scripts/bench/run_levante_gpu_scaling.py-1984-        n_warmup=n_warmup,
scripts/bench/run_levante_gpu_scaling.py-1985-        n_timing=n_timing,
scripts/bench/run_levante_gpu_scaling.py-1986-        compile_time_s=compile_time,
--
scripts/run/run_dino.py-712-            # NEMO time convention: step k (0-based) ends at t=(k+1)*dt —
scripts/run/run_dino.py-713-            # drives the seasonal forcing phases when forcing_annual_cycle.
scripts/run/run_dino.py-714-            state = apply_forcing(state, forcing, z, cfg, dt,
scripts/run/run_dino.py:715:                                  t_seconds=(k + 1) * dt)
scripts/run/run_dino.py-716-
scripts/run/run_dino.py-717-        state = model.step(
scripts/run/run_dino.py-718-            state, dt=dt,
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-779-        physics_level="none",
scripts/bench/bench_ocean_latlon_spmd_scaling.py-780-        backend=jax.default_backend(),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-781-        **tidy_throughput_fields(
scripts/bench/bench_ocean_latlon_spmd_scaling.py:782:            dt_seconds=args.dt, time_per_step_ms=med,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-783-            total_cells=n_lat * args.n_lon * args.nlev),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-784-    )
scripts/bench/bench_ocean_latlon_spmd_scaling.py-785-    # Increment-2 accounting (audit items 4/6/8/9): flat fields so
--
scripts/run/run_omip_core2.py-5404-                grid=str(args.grid),
scripts/run/run_omip_core2.py-5405-                mesh=str(args.mesh),
scripts/run/run_omip_core2.py-5406-                nlev=int(args.nlev),
scripts/run/run_omip_core2.py:5407:                dt_seconds=float(dt),
scripts/run/run_omip_core2.py-5408-                total_days=float(total_days),
scripts/run/run_omip_core2.py-5409-                output_path=str(args.output),
scripts/run/run_omip_core2.py-5410-                forcing="core2_nyf",
--
scripts/run/run_omip_core2.py-5497-    else:
scripts/run/run_omip_core2.py-5498-        _ocean_step = (lambda st, sf, fw, t_sec=None:
scripts/run/run_omip_core2.py-5499-                       model.step(st, dt, surface_forcing=sf, freshwater=fw,
scripts/run/run_omip_core2.py:5500:                                  t_seconds=t_sec))
scripts/run/run_omip_core2.py-5501-    # --spmd-persistent-state lane state (scaling-M2): OFF by default so the
scripts/run/run_omip_core2.py-5502-    # residency helpers below are no-ops and the loop is byte-identical.
scripts/run/run_omip_core2.py-5503-    _spmd_persistent = False
--
scripts/run/run_omip_core2.py-6096-                    wind_current_feedback_vfac=_wind_vfac)
scripts/run/run_omip_core2.py-6097-                sf = sf._replace(freshwater=net_freshwater_flux(fw))
scripts/run/run_omip_core2.py-6098-            state = model.step(state, dt, surface_forcing=sf,
scripts/run/run_omip_core2.py:6099:                               t_seconds=_t_sec)
scripts/run/run_omip_core2.py-6100-        else:
scripts/run/run_omip_core2.py-6101-            fw = None
scripts/run/run_omip_core2.py-6102-            # Build the freshwater struct if EITHER the atmospheric P-E/runoff is
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-2648-    def _step_impl(self, state: LatLonCGridOceanState, dt: float,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-2649-                   freshwater=None, surface_forcing=None,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-2650-                   sponge=None, *, _apply_implicit_vmix: bool = True,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2651:                   grid=None, vertex_mask=None, t_seconds=None,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-2652-                   _ab2_scope_override: str | None = None,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-2653-                   _barotropic_substep_scale: int = 1,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-2654-                   _barotropic_before_state=None,
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-3367-                F_slow_u=F_slow_u,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-3368-                F_slow_v=F_slow_v,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-3369-                add_barotropic_coriolis=_add_bt_cor,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:3370:                t_seconds=t_seconds,  # traced model time for the equilibrium tide
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-3371-                **_baro_seed,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-3372-            )
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-3373-
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-5953-    def step(self, state: LatLonCGridOceanState, dt: float,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-5954-             freshwater=None, surface_forcing=None,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-5955-             sponge=None, *, grid=None,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:5956:             vertex_mask=None, t_seconds=None) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-5957-        """Advance one time step using split-explicit stepping.
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-5958-
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-5959-        Eager Python shim over the JIT-compiled ``_step_jitted``: fills
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6018-                "tidal_forcing.enabled=True but step() was called without "
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6019-                "t_seconds — the equilibrium tide needs the elapsed model "
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6020-                "time and would otherwise be SILENTLY inert.  Pass "
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6021:                "t_seconds=<elapsed seconds device scalar> to step()/"
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6022-                "step_checked(), or drive the run via integrate()/"
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6023-                "integrate_scan() which thread it automatically."
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6024-            )
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6025-        return self._step_jitted(
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6026-            state, dt, freshwater, surface_forcing, sponge,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6027:            grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds)
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6028-
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6029-    @partial(jax.jit, static_argnums=(0,))
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6030-    def _step_jitted(self, state: LatLonCGridOceanState, dt: float,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6031-                     freshwater=None, surface_forcing=None,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6032-                     sponge=None, *, grid=None,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6033:                     vertex_mask=None, t_seconds=None) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6034-        """JIT body of :meth:`step` (split out so the vertex-mask cache
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6035-        fill runs eagerly — see the ``step`` docstring).
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6036-
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6075-            new_state = self._ab2_step(
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6076-                state, dt, freshwater=freshwater,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6077-                surface_forcing=surface_forcing, sponge=sponge,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6078:                grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds)
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6079-        elif _oi == "leapfrog":
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6080-            new_state = self._leapfrog_step(
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6081-                state, dt, freshwater=freshwater,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6082-                surface_forcing=surface_forcing, sponge=sponge,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6083:                grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds)
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6084-        else:
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6085-            new_state = self._step_impl(state, dt, freshwater=freshwater,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6086-                                        surface_forcing=surface_forcing,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6087-                                        sponge=sponge,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6088-                                        grid=grid, vertex_mask=vertex_mask,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6089:                                        t_seconds=t_seconds)
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6090-        # Feature-gated on a STATIC config bool (CLAUDE.md feature-gating
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6091-        # exception): a Python ``if`` selects the branch at trace time, so
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6092-        # the freeze-floor clamp is only traced when enabled — no jnp.where
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6382-    def _ab2_step(self, state: LatLonCGridOceanState, dt: float,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6383-                  freshwater=None, surface_forcing=None, sponge=None,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6384-                  *, grid=None, vertex_mask=None,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6385:                  t_seconds=None) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6386-        """Adams-Bashforth-2 outer integrator (Veros's faithful scheme).
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6387-
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6388-        Veros AB2-extrapolates only the EXPLICIT tendency and applies implicit
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6445-            state, dt, freshwater=freshwater,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6446-            surface_forcing=surface_forcing, sponge=sponge,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6447-            _apply_implicit_vmix=False, grid=_grid, vertex_mask=vertex_mask,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6448:            t_seconds=t_seconds)
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6449-        _tke_prog = self._tke_prognostic_active()
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6450-        _tke_old = (state.tke.data if (_tke_prog and state.tke is not None)
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6451-                    else None)
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6725-    def _leapfrog_step(self, state: LatLonCGridOceanState, dt: float,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6726-                       freshwater=None, surface_forcing=None, sponge=None,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6727-                       *, grid=None, vertex_mask=None,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6728:                       t_seconds=None) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6729-        """NEMO Modified-Leap-Frog step (``stp_MLF``, ``stpmlf.F90``, key_qco DINO).
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6730-
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6731-        Three time levels — before ``Nbb`` (t-dt, carried on the state's
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6880-            naa = self._step_impl(
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6881-                _entry, dt, freshwater=freshwater,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6882-                surface_forcing=surface_forcing, sponge=sponge, grid=_grid,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6883:                vertex_mask=vertex_mask, t_seconds=t_seconds)
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6884-            naa = naa._replace(
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6885-                u_before=state.u, v_before=state.v, T_before=state.T,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6886-                S_before=state.S, eta_before=state.eta,
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6943-                     tke_source, _diss_incr_nn, tracer_source) = self._step_impl(
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6944-            state, rdt, freshwater=freshwater, surface_forcing=surface_forcing,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6945-            sponge=sponge, _apply_implicit_vmix=False, grid=_grid,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6946:            vertex_mask=vertex_mask, t_seconds=t_seconds,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6947-            _ab2_scope_override="advective",
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6948-            _barotropic_substep_scale=_baro_scale,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6949-            _barotropic_before_state=(
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6975-            _kbb6) = self._step_impl(
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6976-            nbb, rdt, freshwater=freshwater, surface_forcing=surface_forcing,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6977-            sponge=sponge, _apply_implicit_vmix=False, grid=_grid,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6978:            vertex_mask=vertex_mask, t_seconds=t_seconds,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6979-            _ab2_scope_override="advective",
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6980-            _barotropic_substep_scale=_baro_scale)
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-6981-
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7343-        surface_forcing=None,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7344-        sponge=None,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7345-        *,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7346:        t_seconds=None,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7347-    ) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7348-        """Advance one timestep with host-side runtime validation."""
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7349-        if not self._cfl_checked:
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7351-            self._cfl_checked = True
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7352-        state_new = self.step(state, dt, freshwater=freshwater,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7353-                              surface_forcing=surface_forcing,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7354:                              sponge=sponge, t_seconds=t_seconds)
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7355-        if self.config.runtime_checks.enable_runtime_checks:
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7356-            self._assert_runtime_invariants(state_new)
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7357-        return state_new
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7495-        for i in range(n_steps):
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7496-            if _tide_on:
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7497-                state = step_fn(state, dt,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7498:                                t_seconds=jnp.asarray(t0_seconds + i * dt))
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7499-            else:
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7500-                state = step_fn(state, dt)
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7501-            if (i + 1) % save_every == 0:
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7812-        # discovery follows the tide-on trace and so the step()'s eager
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7813-        # enabled-but-no-time guard does not trip inside the probe.
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7814-        state = (self.seed_scan_carry(state, dt,
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7815:                                      t_seconds=jnp.asarray(t0_seconds))
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7816-                 if _tide_on else self.seed_scan_carry(state, dt))
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7817-
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7818-        if _tide_on:
--
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7823-            times = t0_seconds + dt * jnp.arange(n_steps)
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7824-
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7825-            def scan_fn(state, t):
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7826:                new_state = self.step(state, dt, t_seconds=t)
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7827-                return new_state, new_state
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7828-
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py-7829-            final_state, trajectory = jax.lax.scan(scan_fn, state, xs=times)
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-117-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-118-def _step_body(model, state, dt, *, grid, vertex_mask,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-119-               freshwater=None, surface_forcing=None, sponge=None,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:120:               t_seconds=None):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-121-    """Run ONE ocean step via the model's NON-jitted body (``_step_impl`` /
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-122-    ``_ab2_step`` + the static-gated post-steps), the un-jitted twin of
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-123-    ``LatLonCGridOceanModel._step_jitted``.
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-150-        new_state = model._ab2_step(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-151-            state, dt, freshwater=freshwater,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-152-            surface_forcing=surface_forcing, sponge=sponge,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:153:            grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-154-    else:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-155-        new_state = model._step_impl(
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-156-            state, dt, freshwater=freshwater,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-157-            surface_forcing=surface_forcing, sponge=sponge,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:158:            grid=grid, vertex_mask=vertex_mask, t_seconds=t_seconds)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-159-    if model.config.polar_filter.use_polar_filter:
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-160-        new_state = model._apply_polar_filter(new_state, dt, grid=grid)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-161-    if model.config.freeze_floor:
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-641-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-642-def make_sharded_ocean_step(model, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-643-    """Return ``step(state, dt, freshwater=None, surface_forcing=None,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:644:    sponge=None, t_seconds=None) -> state`` running ``model.step``
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-645-    lat-band-SPMD.
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-646-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-647-    The forcing channels mirror ``model.step``'s keyword surface: pass
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-824-        result = _step_body(model, state_band, dt,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-825-                            grid=band_geom, vertex_mask=band_vmask,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-826-                            freshwater=fw_local, surface_forcing=sf_local,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:827:                            sponge=sponge_local, t_seconds=t_s_local)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-828-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-829-        out_updates = {name: _to_v_lower(getattr(result, name))
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-830-                       for name in _V_STAGGERED_STATE_FIELDS}
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-870-                    f"(n_lat, n_lon[, nlev]) to shard on the lat axis.")
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-871-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-872-    def sharded_step(state, dt, freshwater=None, surface_forcing=None,
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:873:                     sponge=None, t_seconds=None, aux=None):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-874-        # ``aux``: the sharded geometry+vmask stacks. When this wrapper runs
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-875-        # INSIDE an outer trace (a bench/driver jit/scan — jit-of-jit
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-876-        # inlines the inner call), concrete closure arrays become
--
packages/ocean/legoesm/ocean/experiments/dino.py-3379-
packages/ocean/legoesm/ocean/experiments/dino.py-3380-
packages/ocean/legoesm/ocean/experiments/dino.py-3381-def apply_dino_lat_lon_surface_forcing(state, forcing, z_coord, cfg, dt,
packages/ocean/legoesm/ocean/experiments/dino.py:3382:                                        t_seconds=None):
packages/ocean/legoesm/ocean/experiments/dino.py-3383-    """Apply DINO surface forcing on the lat-lon Mercator grid.
packages/ocean/legoesm/ocean/experiments/dino.py-3384-
packages/ocean/legoesm/ocean/experiments/dino.py-3385-    Components (paper eqs 7-10):
--
packages/ocean/legoesm/ocean/experiments/dino.py-3583-
packages/ocean/legoesm/ocean/experiments/dino.py-3584-
packages/ocean/legoesm/ocean/experiments/dino.py-3585-def apply_dino_mpas_surface_forcing(state, forcing, z_coord, cfg, dt,
packages/ocean/legoesm/ocean/experiments/dino.py:3586:                                    t_seconds=None):
packages/ocean/legoesm/ocean/experiments/dino.py-3587-    """Apply DINO surface forcing on the MPAS regional mesh.
packages/ocean/legoesm/ocean/experiments/dino.py-3588-
packages/ocean/legoesm/ocean/experiments/dino.py-3589-    Same physics as lat-lon (paper eqs 7-10) with edge-projected wind
--
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py-1089-    F_slow_u=None,
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py-1090-    F_slow_v=None,
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py-1091-    add_barotropic_coriolis: bool = True,
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:1092:    t_seconds=None,
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py-1093-    eta_init=None,
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py-1094-    u_init=None,
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py-1095-    v_init=None,
--
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py-1490-    F_slow_u=None,
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py-1491-    F_slow_v=None,
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py-1492-    add_barotropic_coriolis: bool = True,
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:1493:    t_seconds=None,
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py-1494-) -> LatLonCGridOceanState:
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py-1495-    """Wide-halo twin of :func:`barotropic_substeps_latlon_cgrid`.
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py-1496-
--
tests/ocean/unit/test_run_manifest_ocean.py-98-def _run_record(**controls) -> OceanRunRecord:
tests/ocean/unit/test_run_manifest_ocean.py-99-    base = dict(
tests/ocean/unit/test_run_manifest_ocean.py-100-        runtime_config=_latlon_cfg(), grid="tripole", mesh="mesh.nc",
tests/ocean/unit/test_run_manifest_ocean.py:101:        nlev=30, dt_seconds=1800.0, total_days=365.0,
tests/ocean/unit/test_run_manifest_ocean.py-102-        output_path="output/omip", forcing="core2_nyf",
tests/ocean/unit/test_run_manifest_ocean.py-103-    )
tests/ocean/unit/test_run_manifest_ocean.py-104-    base.update(controls)
--
tests/ocean/unit/test_run_manifest_ocean.py-123-        _run_record(), "ocean"
tests/ocean/unit/test_run_manifest_ocean.py-124-    )
tests/ocean/unit/test_run_manifest_ocean.py-125-    for changed in (
tests/ocean/unit/test_run_manifest_ocean.py:126:        _run_record(dt_seconds=3600.0),
tests/ocean/unit/test_run_manifest_ocean.py-127-        _run_record(total_days=730.0),
tests/ocean/unit/test_run_manifest_ocean.py-128-        _run_record(grid="latlon_bathy"),
tests/ocean/unit/test_run_manifest_ocean.py-129-        _run_record(output_path="output/other"),
--
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-839-    A runnable step is therefore::
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-840-
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-841-        state = model.step(
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py:842:            apply_nemo_gyre_surface_forcing(state, z_coord, dt, t_seconds=t),
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py:843:            dt, surface_forcing=nemo_gyre_wind_forcing(n_lat, n_lon, t_seconds=t))
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-844-    """
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-845-    import jax.numpy as jnp
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-846-    from legoesm.core.field import Field
--
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-992-        self.grid_lat = grid_lat_rad
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-993-
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-994-
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py:995:def apply_nemo_gyre_surface_forcing(state, z_coord, dt, *, t_seconds=0.0):
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-996-    """Apply the GYRE Haney NON-solar restoring POST-step (harness route).
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-997-
packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py-998-    NEMO thermal surface forcing, reproduced with the shared canonical kernels:
--
tests/ocean/unit/test_dino_experiment.py-1274-            apply_dino_lat_lon_surface_forcing(st, frc, z, cfg, 2700.0)
tests/ocean/unit/test_dino_experiment.py-1275-        day = 86400.0
tests/ocean/unit/test_dino_experiment.py-1276-        s_jun = apply_dino_lat_lon_surface_forcing(
tests/ocean/unit/test_dino_experiment.py:1277:            st, frc, z, cfg, 2700.0, t_seconds=171.0 * day)
tests/ocean/unit/test_dino_experiment.py-1278-        s_dec = apply_dino_lat_lon_surface_forcing(
tests/ocean/unit/test_dino_experiment.py:1279:            st, frc, z, cfg, 2700.0, t_seconds=351.0 * day)
tests/ocean/unit/test_dino_experiment.py-1280-        assert float(jnp.max(jnp.abs(s_jun.T.data - s_dec.T.data))) > 0.0
tests/ocean/unit/test_dino_experiment.py-1281-        # flag OFF: t_seconds ignored -> bit-identical to the legacy call
tests/ocean/unit/test_dino_experiment.py-1282-        cfg0 = dataclasses.replace(cfg, forcing_annual_cycle=False)
tests/ocean/unit/test_dino_experiment.py-1283-        a = apply_dino_lat_lon_surface_forcing(
tests/ocean/unit/test_dino_experiment.py:1284:            st, frc, z, cfg0, 2700.0, t_seconds=171.0 * day)
tests/ocean/unit/test_dino_experiment.py-1285-        b = apply_dino_lat_lon_surface_forcing(st, frc, z, cfg0, 2700.0)
tests/ocean/unit/test_dino_experiment.py-1286-        np.testing.assert_array_equal(np.asarray(a.T.data),
tests/ocean/unit/test_dino_experiment.py-1287-                                      np.asarray(b.T.data))
--
tests/ocean/unit/test_box_heat_budget.py-124-    st = state
tests/ocean/unit/test_box_heat_budget.py-125-    t = 0.0
tests/ocean/unit/test_box_heat_budget.py-126-    if acc is not None:
tests/ocean/unit/test_box_heat_budget.py:127:        acc.sample(st, dt_step=sample_every * DT, t_seconds=t)
tests/ocean/unit/test_box_heat_budget.py-128-    for k in range(n_steps):
tests/ocean/unit/test_box_heat_budget.py-129-        st = apply_dino_lat_lon_surface_forcing(
tests/ocean/unit/test_box_heat_budget.py:130:            st, forcing, model.z_coord, dino_cfg, DT, t_seconds=t + DT,
tests/ocean/unit/test_box_heat_budget.py-131-        )
tests/ocean/unit/test_box_heat_budget.py:132:        st = model.step(st, DT, surface_forcing=sf, t_seconds=t)
tests/ocean/unit/test_box_heat_budget.py-133-        t += DT
tests/ocean/unit/test_box_heat_budget.py-134-        if acc is not None and (k + 1) % sample_every == 0:
tests/ocean/unit/test_box_heat_budget.py:135:            acc.sample(st, dt_step=sample_every * DT, t_seconds=t)
tests/ocean/unit/test_box_heat_budget.py-136-    return st, acc, t
tests/ocean/unit/test_box_heat_budget.py-137-
tests/ocean/unit/test_box_heat_budget.py-138-
--
tests/ocean/unit/test_box_heat_budget.py-636-    so accidentally passing the wrong one is not a no-op."""
tests/ocean/unit/test_box_heat_budget.py-637-    forcing = dino_lat_lon_surface_forcing_arrays(grid, dino_cfg)
tests/ocean/unit/test_box_heat_budget.py-638-    terms_model_dt = compute_box_heat_dT_terms(
tests/ocean/unit/test_box_heat_budget.py:639:        state, grid, z_coord, config, dino_cfg, forcing, DT, t_seconds=0.0,
tests/ocean/unit/test_box_heat_budget.py-640-    )
tests/ocean/unit/test_box_heat_budget.py-641-    terms_sampling_dt = compute_box_heat_dT_terms(
tests/ocean/unit/test_box_heat_budget.py:642:        state, grid, z_coord, config, dino_cfg, forcing, DT * 32, t_seconds=0.0,
tests/ocean/unit/test_box_heat_budget.py-643-    )
tests/ocean/unit/test_box_heat_budget.py-644-    mask = np.asarray(state.land_mask.data) > 0.5
tests/ocean/unit/test_box_heat_budget.py-645-    forcing_model = np.asarray(terms_model_dt["forcing"])[mask]
--
tests/ocean/unit/test_tidal_forcing.py-338-
tests/ocean/unit/test_tidal_forcing.py-339-
tests/ocean/unit/test_tidal_forcing.py-340-def test_apply_no_time_is_byte_identical_noop():
tests/ocean/unit/test_tidal_forcing.py:341:    """t_seconds=None cannot evaluate an equilibrium tide; the wrapper returns
tests/ocean/unit/test_tidal_forcing.py-342-    the inputs unchanged rather than silently substituting a frozen t=0 tide.
tests/ocean/unit/test_tidal_forcing.py-343-    This matches the inline call sites' `and t_seconds is not None` guard.
tests/ocean/unit/test_tidal_forcing.py-344-    """
--
tests/ocean/unit/test_tidal_forcing.py-495-    with pytest.raises(ValueError, match="t_seconds"):
tests/ocean/unit/test_tidal_forcing.py-496-        model.step(state, _WIRE_DT)
tests/ocean/unit/test_tidal_forcing.py-497-    # With the time supplied the same call steps fine.
tests/ocean/unit/test_tidal_forcing.py:498:    out = model.step(state, _WIRE_DT, t_seconds=jnp.asarray(0.0))
tests/ocean/unit/test_tidal_forcing.py-499-    assert jnp.all(jnp.isfinite(out.eta.data))
--
tests/ocean/unit/test_nemo_recipe.py-350-    for i in range(3):
tests/ocean/unit/test_nemo_recipe.py-351-        t = i * _NEMO_GYRE_DT_S
tests/ocean/unit/test_nemo_recipe.py-352-        st = apply_nemo_gyre_surface_forcing(
tests/ocean/unit/test_nemo_recipe.py:353:            st, recipe.z_coord, _NEMO_GYRE_DT_S, t_seconds=t)
tests/ocean/unit/test_nemo_recipe.py-354-        st = model.step(
tests/ocean/unit/test_nemo_recipe.py-355-            st, dt=_NEMO_GYRE_DT_S,
tests/ocean/unit/test_nemo_recipe.py:356:            surface_forcing=nemo_gyre_wind_forcing(n_lat, n_lon, t_seconds=t))
tests/ocean/unit/test_nemo_recipe.py-357-    for f in (st.T.data, st.S.data, st.u.data, st.v.data, st.eta.data):
tests/ocean/unit/test_nemo_recipe.py-358-        assert bool(jnp.all(jnp.isfinite(f)))
tests/ocean/unit/test_nemo_recipe.py-359-    # gently-forced GYRE spin-up stays laminar (no barotropic blow-up).
--
tests/ocean/unit/test_nemo_recipe.py-366-    for i in range(3):
tests/ocean/unit/test_nemo_recipe.py-367-        t = i * _NEMO_GYRE_DT_S
tests/ocean/unit/test_nemo_recipe.py-368-        st_nw = apply_nemo_gyre_surface_forcing(
tests/ocean/unit/test_nemo_recipe.py:369:            st_nw, recipe.z_coord, _NEMO_GYRE_DT_S, t_seconds=t)
tests/ocean/unit/test_nemo_recipe.py-370-        st_nw = model.step(st_nw, dt=_NEMO_GYRE_DT_S)
tests/ocean/unit/test_nemo_recipe.py-371-    assert float(jnp.max(jnp.abs(st.u.data - st_nw.u.data))) > 1e-4
tests/ocean/unit/test_nemo_recipe.py-372-
--
tests/ocean/unit/test_nemo_recipe.py-389-    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, recipe.model_config)
tests/ocean/unit/test_nemo_recipe.py-390-    st = recipe.initial_state

exec
/usr/bin/bash -lc "sed -n '20,115p' packages/ocean/legoesm/ocean/freshwater.py; sed -n '20,100p' packages/ocean/legoesm/ocean/sponge.py; sed -n '70,145p' packages/ocean/legoesm/ocean/state.py; rg -n -C 3 'shard_forcing_stack_latlon\\(' scripts/run/run_omip.py scripts/run/run_omip_core2.py packages tests --glob '*.py' | sed -n '1,360p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 658ms:
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class FreshwaterForcing(NamedTuple):
    """Freshwater fluxes applied to the ocean surface.

    All fields have shape (nCells,) and units kg/m²/s.
    Positive = freshwater into ocean, except evaporation which is
    positive upward (i.e., freshwater leaving ocean).

    Fields
    ------
    precip : jax.Array
        Precipitation rate [kg/m²/s].
    evap : jax.Array
        Evaporation rate [kg/m²/s], positive upward.
    runoff : jax.Array
        Land runoff rate [kg/m²/s].
    ice_fw : jax.Array
        Ice melt/freeze freshwater [kg/m²/s], positive = melt.
    restoring : jax.Array
        OMIP-2 SSS-restoring virtual freshwater flux [kg/m²/s,
        positive INTO ocean].  Computed from
        :func:`legoesm.ocean.forcing.sss_restoring.compute_sss_restoring_flux`.
        Zero by default for backward compatibility with the
        legacy 4-component constructor.
    """
    precip: jnp.ndarray
    evap: jnp.ndarray
    runoff: jnp.ndarray
    ice_fw: jnp.ndarray
    restoring: jnp.ndarray = None  # type: ignore[assignment]


def zero_freshwater(nCells: int) -> FreshwaterForcing:
    """Create zero freshwater forcing.

    Parameters
    ----------
    nCells : int
        Number of Voronoi cells.

    Returns
    -------
    FreshwaterForcing
    """
    # Init helper: keep at the JAX default float dtype.  Callers running
    # under a non-default precision policy can ``cast_pytree`` the
    # result to match their state.
    z = jnp.zeros(nCells)
    return FreshwaterForcing(precip=z, evap=z, runoff=z, ice_fw=z, restoring=z)


def net_freshwater_flux(fw: FreshwaterForcing) -> jnp.ndarray:
    """Compute net freshwater flux into ocean [kg/m²/s].

    F_fw = P - E + R + M + R_restore

    where P=precip, E=evaporation (positive up), R=runoff,
    M=ice melt, R_restore=SSS-restoring virtual FW flux (zero
    when ``restoring`` field is None or absent — legacy
    callers built without the SSS-restoring extension).

    Parameters
    ----------
    fw : FreshwaterForcing

    Returns
    -------
    jax.Array, shape (nCells,)
        Net freshwater flux [kg/m²/s], positive into ocean.
    """
    base = fw.precip - fw.evap + fw.runoff + fw.ice_fw
    # ``restoring is None`` is a Python (trace-time) check — safe
    # under JIT because the field is structural metadata.
    if fw.restoring is None:
        return base
    return base + fw.restoring


def freshwater_eta_tendency(fw: FreshwaterForcing, rho_0: float) -> jnp.ndarray:
    """Compute free-surface tendency from freshwater flux.

    deta/dt = F_fw / rho_0

    Parameters
    ----------
    fw : FreshwaterForcing
    rho_0 : float
        Reference seawater density [kg/m3].

    Returns

import jax.numpy as jnp
import numpy as np


class SpongeForcing(NamedTuple):
    """Sponge layer relaxation fields.

    Parameters
    ----------
    gamma : array
        Relaxation rate [1/s].  Either HORIZONTAL — shape
        ``(n_lat, n_lon)`` for lat-lon or ``(nCells,)`` for MPAS,
        broadcast over the vertical — or FULL-RANK per-cell — shape
        ``(n_lat, n_lon, nlev)`` / ``(nCells, nlev)`` for z-varying /
        partial-column restoring zones (e.g. the Veros north_atlantic
        ``rest_tscl(x, y, z)`` field).  Zero in the interior, ramping
        to ``1/tau`` near boundaries.  A full-rank gamma supports
        TRACER relaxation only (``u_ref``/``v_ref`` must be None: the
        momentum sponge interpolates gamma to velocity faces, which is
        defined for the horizontal form only).
    T_ref : array
        Reference temperature, same shape as the model T field.
    S_ref : array
        Reference salinity, same shape as the model S field.
    u_ref : array or None
        Reference u velocity (optional).
    v_ref : array or None
        Reference v velocity (optional, not used for MPAS edge velocity).
    """
    gamma: jnp.ndarray
    T_ref: jnp.ndarray
    S_ref: jnp.ndarray
    u_ref: jnp.ndarray | None = None
    v_ref: jnp.ndarray | None = None


def compute_sponge_gamma(
    lat_deg: np.ndarray,
    lat_south: float,
    lat_north: float,
    width_deg: float = 2.0,
    timescale_days: float = 1.0,
) -> np.ndarray:
    """Quadratic sponge ramp as a function of latitude — the shared kernel.

    Ramp from 0 in the interior to ``1/tau`` at the south/north walls, with
    south taking precedence in the (degenerate) overlap. Works on any-shaped
    ``lat_deg`` array; the grid-specific wrappers below supply the latitudes.

    Parameters
    ----------
    lat_deg : ndarray
        Latitudes [degrees], any shape.
    lat_south, lat_north : float
        Domain boundaries [degrees].
    width_deg : float
        Sponge zone width [degrees].
    timescale_days : float
        Relaxation e-folding timescale [days].

    Returns
    -------
    gamma : ndarray, same shape as ``lat_deg``
        Relaxation coefficient [1/s].
    """
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    tau = timescale_days * 86400.0
    dist_south = lat_deg - lat_south
    dist_north = lat_north - lat_deg
    # Nested where reproduces the original ``if south elif north`` precedence.
    return np.where(
        dist_south < width_deg,
        (1.0 - dist_south / width_deg) ** 2 / tau,
        np.where(
            dist_north < width_deg,
            (1.0 - dist_north / width_deg) ** 2 / tau,
            0.0,
        ),
    )

    ``(..., nlev-1)`` at interior interfaces, or None.
    """
    du_dt: Field
    dv_dt: Field
    dT_dt: Field
    dS_dt: Field
    deta_dt: Field
    dH_bathy_dt: Field
    dland_mask_dt: Field
    K_v: object = None   # tracer diffusivity at interfaces [m²/s]
    A_v: object = None   # momentum viscosity at interfaces [m²/s]


class OceanSurfaceForcing(NamedTuple):
    """External atmospheric/surface forcing data for ocean physics.

    Carries coupler-provided fields into the ocean physics pipeline.
    All fields are optional (None means not available).  Shape of 2D
    fields matches the horizontal grid; 3D fields add a level axis.

    Fields
    ------
    sw_down : array or None
        Downwelling shortwave at sea surface [W/m²].  Needed for
        subsurface SW penetration heating.
    q_net : array or None
        Net surface heat flux (positive into ocean) [W/m²].
    tau_x, tau_y : array or None
        Surface wind stress components [Pa].
    freshwater : array or None
        Net freshwater flux into ocean (P - E + R + M) [kg/m²/s].
    salt_flux : array or None
        REAL salt-mass flux into the ocean [kg(salt)/m²/s, positive = salt INTO
        ocean], e.g. sea-ice brine rejection on freeze.  Applied to the top
        layer salinity as dS/dt = salt_flux*1e3/(rho_0*dz_0); distinct from the
        ``freshwater`` (virtual-salt dilution) channel.
    chl : array or None
        Surface chlorophyll [mg/m³] for the RGB shortwave-penetration scheme
        (``ShortwavePenetrationConfig.scheme == "rgb_chl"``).  2D horizontal
        field; ``None`` when the two-band Jerlov scheme is in use.
    q_prescribed : array or None
        Prescribed part of the surface heat flux [W/m², positive into ocean]
        consumed ONLY by the ``"flux_feedback"`` surface-forcing scheme
        (Veros global_4deg ``qnet``).  Kept separate from ``q_net`` so the
        feedback scheme owns the TOTAL heat in one place (ice mask) and so
        the prescribed-channel ``c_sw`` seam is not double-counted — leave
        ``q_net=None`` when using ``flux_feedback``.
    q_feedback : array or None
        Linear SST-feedback (piston) coefficient [W/m²/K, ≥ 0 damps] for the
        ``"flux_feedback"`` scheme (Veros ``qnec``).  Heat contribution is
        ``q_feedback · (T_feedback_target − T_surf)``.
    T_feedback_target : array or None
        Target SST [°C] for the ``q_feedback`` term (Veros ``sst_clim``,
        monthly-interpolated by the driver/harness).
    S_restore_target : array or None
        Target SSS [PSU] for the ``flux_feedback`` scheme's surface-salinity
        restoring (Veros ``sss_clim``).
    q_solar : array or None
        Penetrative SOLAR component of the surface heat flux [W/m², positive
        into ocean] consumed ONLY by the ``"flux_feedback"`` scheme when
        ``FluxFeedbackConfig.penetrative_shortwave=True`` (Veros
        global_flexible / global_1deg ``qsol``).  Deposited through the water
        column with the shared two-band Jerlov profile
        (``shortwave_penetration_tendency``, water type from
        ``FluxFeedbackConfig.shortwave_water_type``; type "I" = exactly the
        Veros literals R=0.58, ζ1=0.35 m, ζ2=23.0 m).

        HEAT-OWNERSHIP CONTRACT (no double counting): when ``q_solar`` is
        provided, ``q_prescribed`` must carry the NON-SOLAR remainder only
        (harness: ``q_prescribed = qnet_total − qsol``).  legoESM deposits
        100% of ``q_solar`` in the column with the I(0)=1 surface convention;
        Veros instead keeps the solar-inclusive total in ``qnet`` and applies
        a zero-column-sum redistribution built with pen(0)=0 — the two are
        algebraically identical cell by cell (top cell receives
        ``qnet_total − qsol·I(z₁)`` either way, deeper cells receive the same
        interface-flux differences, and both catch the residual light in the
tests/parallel/test_latlon_ocean_spmd_step.py-298-        "none": None,                                    # pytree-None
tests/parallel/test_latlon_ocean_spmd_step.py-299-        "meta": "1958-01-01",                            # non-array passthrough
tests/parallel/test_latlon_ocean_spmd_step.py-300-    }
tests/parallel/test_latlon_ocean_spmd_step.py:301:    out = shard_forcing_stack_latlon(stack, dev.mesh)
tests/parallel/test_latlon_ocean_spmd_step.py-302-
tests/parallel/test_latlon_ocean_spmd_step.py-303-    assert _lat_axis(out["rec3d"]) == 1
tests/parallel/test_latlon_ocean_spmd_step.py-304-    assert _lat_axis(out["rec4d"]) == 1
--
tests/parallel/test_latlon_ocean_spmd_step.py-309-    assert out["meta"] == "1958-01-01"
tests/parallel/test_latlon_ocean_spmd_step.py-310-
tests/parallel/test_latlon_ocean_spmd_step.py-311-    # mesh=None is the serial-lane passthrough (identity).
tests/parallel/test_latlon_ocean_spmd_step.py:312:    assert shard_forcing_stack_latlon(stack, None) is stack
tests/parallel/test_latlon_ocean_spmd_step.py-313-
tests/parallel/test_latlon_ocean_spmd_step.py-314-
tests/parallel/test_latlon_ocean_spmd_step.py-315-@pytest.mark.skipif(jax.device_count() < 4,
--
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-402-    return jax.tree.map(_put, forcing)
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-403-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-404-
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:405:def shard_forcing_stack_latlon(stack, mesh):
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-406-    """Lay out a STACKED per-block forcing pytree for the lat-band SPMD
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-407-    block-scan (the ``run_omip`` JRA55 lanes; see
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py-408-    ``_build_jra55_block_fn`` / ``_build_jra55_block_fn_interp``).

exec
/usr/bin/bash -lc "rg -n -C 8 '"'^class Field|register_pytree|tree_flatten|tree_unflatten'"' packages/core/legoesm/core/field.py packages/core/legoesm/core | sed -n '1,320p'; sed -n '1,190p' packages/core/legoesm/core/field.py; rg -n -C 4 '"'^class LatLonCGridOceanState'"' packages/ocean/legoesm/ocean/state.py; sed -n '1,95p' packages/ocean/legoesm/ocean/state.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 275ms:
packages/core/legoesm/core/field.py-10-from typing import Any
packages/core/legoesm/core/field.py-11-
packages/core/legoesm/core/field.py-12-import jax
packages/core/legoesm/core/field.py-13-import jax.numpy as jnp
packages/core/legoesm/core/field.py-14-
packages/core/legoesm/core/field.py-15-from legoesm.core.precision import get_policy
packages/core/legoesm/core/field.py-16-
packages/core/legoesm/core/field.py-17-
packages/core/legoesm/core/field.py:18:class Field:
packages/core/legoesm/core/field.py-19-    """A coordinate-aware JAX array.
packages/core/legoesm/core/field.py-20-
packages/core/legoesm/core/field.py-21-    Fields carry metadata alongside their numerical data, enabling:
packages/core/legoesm/core/field.py-22-    - Automatic dimension tracking through operations
packages/core/legoesm/core/field.py-23-    - Unit-aware diagnostics and I/O
packages/core/legoesm/core/field.py-24-    - Staggering information for C-grid operators
packages/core/legoesm/core/field.py-25-
packages/core/legoesm/core/field.py-26-    As a JAX pytree, Field works seamlessly with all JAX transformations:
--
packages/core/legoesm/core/field.py-60-        object.__setattr__(self, "long_name", long_name)
packages/core/legoesm/core/field.py-61-        object.__setattr__(self, "staggering", staggering)
packages/core/legoesm/core/field.py-62-
packages/core/legoesm/core/field.py-63-    def __setattr__(self, key: str, value: Any) -> None:
packages/core/legoesm/core/field.py-64-        raise AttributeError("Field is immutable. Use .replace() to create a new Field.")
packages/core/legoesm/core/field.py-65-
packages/core/legoesm/core/field.py-66-    # ---- JAX pytree registration ----
packages/core/legoesm/core/field.py-67-
packages/core/legoesm/core/field.py:68:    def tree_flatten(self):
packages/core/legoesm/core/field.py-69-        """Flatten for JAX: data is the dynamic leaf, metadata is static.
packages/core/legoesm/core/field.py-70-
packages/core/legoesm/core/field.py-71-        All metadata is part of aux_data (static). For tree_map to work
packages/core/legoesm/core/field.py-72-        between two pytrees, their aux_data must match exactly. Ensure that
packages/core/legoesm/core/field.py-73-        tendencies returned by physics modules use state.field.replace(data=...)
packages/core/legoesm/core/field.py-74-        to preserve metadata compatibility.
packages/core/legoesm/core/field.py-75-        """
packages/core/legoesm/core/field.py-76-        children = (self.data,)
packages/core/legoesm/core/field.py-77-        aux_data = (self.name, self.dims, self.units, self.long_name, self.staggering)
packages/core/legoesm/core/field.py-78-        return children, aux_data
packages/core/legoesm/core/field.py-79-
packages/core/legoesm/core/field.py-80-    @classmethod
packages/core/legoesm/core/field.py:81:    def tree_unflatten(cls, aux_data, children):
packages/core/legoesm/core/field.py-82-        """Reconstruct Field from flattened representation."""
packages/core/legoesm/core/field.py-83-        (data,) = children
packages/core/legoesm/core/field.py-84-        name, dims, units, long_name, staggering = aux_data
packages/core/legoesm/core/field.py-85-        return cls(data=data, name=name, dims=dims, units=units,
packages/core/legoesm/core/field.py-86-                   long_name=long_name, staggering=staggering)
packages/core/legoesm/core/field.py-87-
packages/core/legoesm/core/field.py-88-    # ---- Convenience methods ----
packages/core/legoesm/core/field.py-89-
--
packages/core/legoesm/core/field.py-158-        return (
packages/core/legoesm/core/field.py-159-            f"Field(name='{self.name}', shape={self.shape}, "
packages/core/legoesm/core/field.py-160-            f"dims={self.dims}, units='{self.units}', "
packages/core/legoesm/core/field.py-161-            f"staggering='{self.staggering}')"
packages/core/legoesm/core/field.py-162-        )
packages/core/legoesm/core/field.py-163-
packages/core/legoesm/core/field.py-164-
packages/core/legoesm/core/field.py-165-# Register Field as a JAX pytree
packages/core/legoesm/core/field.py:166:jax.tree_util.register_pytree_node(
packages/core/legoesm/core/field.py-167-    Field,
packages/core/legoesm/core/field.py:168:    lambda f: f.tree_flatten(),
packages/core/legoesm/core/field.py:169:    lambda aux, children: Field.tree_unflatten(aux, children),
packages/core/legoesm/core/field.py-170-)
packages/core/legoesm/core/field.py-171-
packages/core/legoesm/core/field.py-172-
packages/core/legoesm/core/field.py-173-def zeros_field(
packages/core/legoesm/core/field.py-174-    shape: tuple[int, ...],
packages/core/legoesm/core/field.py-175-    name: str = "",
packages/core/legoesm/core/field.py-176-    dims: tuple[str, ...] = (),
packages/core/legoesm/core/field.py-177-    units: str = "",
--
packages/core/legoesm/core/field.py-10-from typing import Any
packages/core/legoesm/core/field.py-11-
packages/core/legoesm/core/field.py-12-import jax
packages/core/legoesm/core/field.py-13-import jax.numpy as jnp
packages/core/legoesm/core/field.py-14-
packages/core/legoesm/core/field.py-15-from legoesm.core.precision import get_policy
packages/core/legoesm/core/field.py-16-
packages/core/legoesm/core/field.py-17-
packages/core/legoesm/core/field.py:18:class Field:
packages/core/legoesm/core/field.py-19-    """A coordinate-aware JAX array.
packages/core/legoesm/core/field.py-20-
packages/core/legoesm/core/field.py-21-    Fields carry metadata alongside their numerical data, enabling:
packages/core/legoesm/core/field.py-22-    - Automatic dimension tracking through operations
packages/core/legoesm/core/field.py-23-    - Unit-aware diagnostics and I/O
packages/core/legoesm/core/field.py-24-    - Staggering information for C-grid operators
packages/core/legoesm/core/field.py-25-
packages/core/legoesm/core/field.py-26-    As a JAX pytree, Field works seamlessly with all JAX transformations:
--
packages/core/legoesm/core/field.py-60-        object.__setattr__(self, "long_name", long_name)
packages/core/legoesm/core/field.py-61-        object.__setattr__(self, "staggering", staggering)
packages/core/legoesm/core/field.py-62-
packages/core/legoesm/core/field.py-63-    def __setattr__(self, key: str, value: Any) -> None:
packages/core/legoesm/core/field.py-64-        raise AttributeError("Field is immutable. Use .replace() to create a new Field.")
packages/core/legoesm/core/field.py-65-
packages/core/legoesm/core/field.py-66-    # ---- JAX pytree registration ----
packages/core/legoesm/core/field.py-67-
packages/core/legoesm/core/field.py:68:    def tree_flatten(self):
packages/core/legoesm/core/field.py-69-        """Flatten for JAX: data is the dynamic leaf, metadata is static.
packages/core/legoesm/core/field.py-70-
packages/core/legoesm/core/field.py-71-        All metadata is part of aux_data (static). For tree_map to work
packages/core/legoesm/core/field.py-72-        between two pytrees, their aux_data must match exactly. Ensure that
packages/core/legoesm/core/field.py-73-        tendencies returned by physics modules use state.field.replace(data=...)
packages/core/legoesm/core/field.py-74-        to preserve metadata compatibility.
packages/core/legoesm/core/field.py-75-        """
packages/core/legoesm/core/field.py-76-        children = (self.data,)
packages/core/legoesm/core/field.py-77-        aux_data = (self.name, self.dims, self.units, self.long_name, self.staggering)
packages/core/legoesm/core/field.py-78-        return children, aux_data
packages/core/legoesm/core/field.py-79-
packages/core/legoesm/core/field.py-80-    @classmethod
packages/core/legoesm/core/field.py:81:    def tree_unflatten(cls, aux_data, children):
packages/core/legoesm/core/field.py-82-        """Reconstruct Field from flattened representation."""
packages/core/legoesm/core/field.py-83-        (data,) = children
packages/core/legoesm/core/field.py-84-        name, dims, units, long_name, staggering = aux_data
packages/core/legoesm/core/field.py-85-        return cls(data=data, name=name, dims=dims, units=units,
packages/core/legoesm/core/field.py-86-                   long_name=long_name, staggering=staggering)
packages/core/legoesm/core/field.py-87-
packages/core/legoesm/core/field.py-88-    # ---- Convenience methods ----
packages/core/legoesm/core/field.py-89-
--
packages/core/legoesm/core/field.py-158-        return (
packages/core/legoesm/core/field.py-159-            f"Field(name='{self.name}', shape={self.shape}, "
packages/core/legoesm/core/field.py-160-            f"dims={self.dims}, units='{self.units}', "
packages/core/legoesm/core/field.py-161-            f"staggering='{self.staggering}')"
packages/core/legoesm/core/field.py-162-        )
packages/core/legoesm/core/field.py-163-
packages/core/legoesm/core/field.py-164-
packages/core/legoesm/core/field.py-165-# Register Field as a JAX pytree
packages/core/legoesm/core/field.py:166:jax.tree_util.register_pytree_node(
packages/core/legoesm/core/field.py-167-    Field,
packages/core/legoesm/core/field.py:168:    lambda f: f.tree_flatten(),
packages/core/legoesm/core/field.py:169:    lambda aux, children: Field.tree_unflatten(aux, children),
packages/core/legoesm/core/field.py-170-)
packages/core/legoesm/core/field.py-171-
packages/core/legoesm/core/field.py-172-
packages/core/legoesm/core/field.py-173-def zeros_field(
packages/core/legoesm/core/field.py-174-    shape: tuple[int, ...],
packages/core/legoesm/core/field.py-175-    name: str = "",
packages/core/legoesm/core/field.py-176-    dims: tuple[str, ...] = (),
packages/core/legoesm/core/field.py-177-    units: str = "",
"""Field: A coordinate-aware array registered as a JAX pytree.

The Field is the fundamental data container in legoESM. It wraps a JAX array
with metadata (name, dimensions, units, staggering) and is registered as a JAX
pytree so that jit, grad, vmap, and scan all work transparently.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from legoesm.core.precision import get_policy


class Field:
    """A coordinate-aware JAX array.

    Fields carry metadata alongside their numerical data, enabling:
    - Automatic dimension tracking through operations
    - Unit-aware diagnostics and I/O
    - Staggering information for C-grid operators

    As a JAX pytree, Field works seamlessly with all JAX transformations:
    jit, grad, vmap, scan, checkpoint, etc.

    Parameters
    ----------
    data : jax.Array
        The numerical data.
    name : str
        Variable name (e.g., "potential_temperature").
    dims : tuple of str
        Dimension names (e.g., ("face", "x", "y")).
    units : str
        Physical units (e.g., "K", "m/s").
    long_name : str, optional
        Human-readable description.
    staggering : str, optional
        Grid staggering: "cell", "edge", or "vertex". Default "cell".
    """

    __slots__ = ("data", "name", "dims", "units", "long_name", "staggering")

    def __init__(
        self,
        data: jax.Array,
        name: str = "",
        dims: tuple[str, ...] = (),
        units: str = "",
        long_name: str = "",
        staggering: str = "cell",
    ):
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "long_name", long_name)
        object.__setattr__(self, "staggering", staggering)

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("Field is immutable. Use .replace() to create a new Field.")

    # ---- JAX pytree registration ----

    def tree_flatten(self):
        """Flatten for JAX: data is the dynamic leaf, metadata is static.

        All metadata is part of aux_data (static). For tree_map to work
        between two pytrees, their aux_data must match exactly. Ensure that
        tendencies returned by physics modules use state.field.replace(data=...)
        to preserve metadata compatibility.
        """
        children = (self.data,)
        aux_data = (self.name, self.dims, self.units, self.long_name, self.staggering)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct Field from flattened representation."""
        (data,) = children
        name, dims, units, long_name, staggering = aux_data
        return cls(data=data, name=name, dims=dims, units=units,
                   long_name=long_name, staggering=staggering)

    # ---- Convenience methods ----

    def replace(self, **kwargs) -> Field:
        """Create a new Field with some attributes replaced."""
        return Field(
            data=kwargs.get("data", self.data),
            name=kwargs.get("name", self.name),
            dims=kwargs.get("dims", self.dims),
            units=kwargs.get("units", self.units),
            long_name=kwargs.get("long_name", self.long_name),
            staggering=kwargs.get("staggering", self.staggering),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def astype(self, dtype) -> Field:
        """Cast data to a different dtype."""
        return self.replace(data=self.data.astype(dtype))

    # ---- Arithmetic (operate on data, preserve metadata) ----

    def __add__(self, other):
        if isinstance(other, Field):
            return self.replace(data=self.data + other.data)
        return self.replace(data=self.data + other)

    def __radd__(self, other):
        return self.replace(data=other + self.data)

    def __sub__(self, other):
        if isinstance(other, Field):
            return self.replace(data=self.data - other.data)
        return self.replace(data=self.data - other)

    def __rsub__(self, other):
        return self.replace(data=other - self.data)

    def __mul__(self, other):
        if isinstance(other, Field):
            return self.replace(data=self.data * other.data)
        return self.replace(data=self.data * other)

    def __rmul__(self, other):
        return self.replace(data=other * self.data)

    def __truediv__(self, other):
        if isinstance(other, Field):
            return self.replace(data=self.data / other.data)
        return self.replace(data=self.data / other)

    def __rtruediv__(self, other):
        return self.replace(data=other / self.data)

    def __neg__(self):
        return self.replace(data=-self.data)

    def __pow__(self, other):
        return self.replace(data=self.data ** other)

    def __repr__(self) -> str:
        return (
            f"Field(name='{self.name}', shape={self.shape}, "
            f"dims={self.dims}, units='{self.units}', "
            f"staggering='{self.staggering}')"
        )


# Register Field as a JAX pytree
jax.tree_util.register_pytree_node(
    Field,
    lambda f: f.tree_flatten(),
    lambda aux, children: Field.tree_unflatten(aux, children),
)


def zeros_field(
    shape: tuple[int, ...],
    name: str = "",
    dims: tuple[str, ...] = (),
    units: str = "",
    long_name: str = "",
    staggering: str = "cell",
    dtype=None,
) -> Field:
    """Create a Field filled with zeros.

    If *dtype* is ``None``, defaults to the active precision policy's
    storage dtype (``get_policy().storage``).
    """
    if dtype is None:
        dtype = get_policy().storage
    return Field(
        data=jnp.zeros(shape, dtype=dtype),
358-# ==============================================================================
359-# Lat-Lon C-Grid FV Ocean State
360-# ==============================================================================
361-
362:class LatLonCGridOceanState(NamedTuple):
363-    """State for the lat-lon C-grid finite-volume ocean primitive equations.
364-
365-    Velocities live on cell faces (Arakawa C-grid staggering):
366-    - u at east/west faces (lon interfaces): shape (n_lat, n_lon+1, nlev)
"""Ocean state containers.

All states are registered as JAX pytrees via NamedTuple + Field,
consistent with ShallowWaterState, HydrostaticState, and
NonHydrostaticState in core/state.py.
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants
from legoesm.core.field import Field
from legoesm.ocean.constants_config import ConstantsConfig
from legoesm.ocean.eos import FreezingPointConfig

# Seawater freezing point in degC (model T is in degC), captured at MODULE scope
# where ``constants`` is the module.  Inside ``LatLonCGridOceanConfig`` the field
# ``constants: ConstantsConfig`` (defined mid-class) shadows the module name, so a
# field default cannot evaluate ``constants.T_freeze_ocean`` directly -- reference
# this module-level value instead.  = 271.35 - 273.15 = -1.8 C (no literal).
_T_FREEZE_OCEAN_C: float = constants.T_freeze_ocean - constants.T_freeze


# ==============================================================================
# FV Ocean State (cubed-sphere)
# ==============================================================================

class OceanState(NamedTuple):
    """State for the ocean primitive equations on the cubed-sphere.

    3D fields: shape (6, n, n, nlev).
    2D fields: shape (6, n, n).

    Fields
    ------
    u : Field
        Zonal velocity [m/s]. Prognostic. Shape (6, n, n, nlev).
    v : Field
        Meridional velocity [m/s]. Prognostic. Shape (6, n, n, nlev).
    T : Field
        Potential temperature [degC]. Prognostic. Shape (6, n, n, nlev).
    S : Field
        Salinity [PSU]. Prognostic. Shape (6, n, n, nlev).
    eta : Field
        Sea surface height [m]. Prognostic. Shape (6, n, n).
    H_bathy : Field
        Bathymetry depth [m]. Static (positive downward). Shape (6, n, n).
    land_mask : Field
        Ocean mask. Static. 1=ocean, 0=land. Shape (6, n, n).
    """
    u: Field
    v: Field
    T: Field
    S: Field
    eta: Field
    H_bathy: Field
    land_mask: Field


class OceanTendencies(NamedTuple):
    """Tendencies for the ocean primitive equations.

    Same structure as OceanState. Static fields (H_bathy, land_mask)
    have zero tendencies, matching the phis pattern in the atmosphere.

    K_v / A_v are optional interface-level diffusivity / viscosity
    profiles populated by the physics function when
    ``implicit_vertical_mixing`` is enabled.  Shape
    ``(..., nlev-1)`` at interior interfaces, or None.
    """
    du_dt: Field
    dv_dt: Field
    dT_dt: Field
    dS_dt: Field
    deta_dt: Field
    dH_bathy_dt: Field
    dland_mask_dt: Field
    K_v: object = None   # tracer diffusivity at interfaces [m²/s]
    A_v: object = None   # momentum viscosity at interfaces [m²/s]


class OceanSurfaceForcing(NamedTuple):
    """External atmospheric/surface forcing data for ocean physics.

    Carries coupler-provided fields into the ocean physics pipeline.
    All fields are optional (None means not available).  Shape of 2D
    fields matches the horizontal grid; 3D fields add a level axis.

    Fields
    ------
    sw_down : array or None
        Downwelling shortwave at sea surface [W/m²].  Needed for
        subsurface SW penetration heating.
    q_net : array or None

codex
1. `assert_pytree_bytes_equal` skips Python scalar leaves, but the scatter paths explicitly replicate scalars via `jnp.asarray` + `addressable_shard_put`. Rank-divergent scalar inputs therefore bypass the replacement for `device_put`’s equality assert. Include numeric scalars in the digest gate. [geometry_consistency.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/geometry_consistency.py:710) [sharded_ocean_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:396)

2. `aux` is a new per-call input but is absent from `_agree_ocean_spmd_call`; a rank-local `None`/provided or schema mismatch can get past the entry gate before the JIT call. Add `has_aux` and `tree_schema_digest48(aux)` to the fixed payload and pass `aux` to the gate. [sharded_ocean_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:608) [sharded_ocean_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:872)

3. Atmosphere still sends band/tile stacks through `broadcast_checked`, which ends in `broadcast_one_to_all`; both the 1-D segment default and 2-D default use sharded geometry. This preserves wall 1 for high process counts. It is outside the ocean file port, but cannot be described as centrally eliminating wall 1. Migrate the sharded-geometry paths to an appropriate checked callback put, or explicitly track it as a scaling follow-up. [sharded_atm_latlon_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:492) [sharded_atm_latlon_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1586)

4. Minor API/doc gaps: `band_fingerprint` raises `IndexError`, not the promised `ValueError`, for a 0-D input; and the four new shared helpers are missing from `__all__`. [geometry_consistency.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/geometry_consistency.py:23) [geometry_consistency.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/geometry_consistency.py:633)

The ocean schema gate is correctly before every `checked_shard_put`; it gates `n_dev`, dtype class, and rank before the per-field collectives. No ocean-local duplicate fingerprint code remains: whole-array fixed-width fingerprints serve broadcast callers (atmosphere/coupler), while per-band fingerprints serve the assert-free ocean put. The import direction is valid: core imports only JAX/NumPy, and ocean depends on core.

Static diff check and Python 3.12 syntax parsing passed.

VERDICT: FIX-FIRST
tokens used
183,212
1. `assert_pytree_bytes_equal` skips Python scalar leaves, but the scatter paths explicitly replicate scalars via `jnp.asarray` + `addressable_shard_put`. Rank-divergent scalar inputs therefore bypass the replacement for `device_put`’s equality assert. Include numeric scalars in the digest gate. [geometry_consistency.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/geometry_consistency.py:710) [sharded_ocean_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:396)

2. `aux` is a new per-call input but is absent from `_agree_ocean_spmd_call`; a rank-local `None`/provided or schema mismatch can get past the entry gate before the JIT call. Add `has_aux` and `tree_schema_digest48(aux)` to the fixed payload and pass `aux` to the gate. [sharded_ocean_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:608) [sharded_ocean_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:872)

3. Atmosphere still sends band/tile stacks through `broadcast_checked`, which ends in `broadcast_one_to_all`; both the 1-D segment default and 2-D default use sharded geometry. This preserves wall 1 for high process counts. It is outside the ocean file port, but cannot be described as centrally eliminating wall 1. Migrate the sharded-geometry paths to an appropriate checked callback put, or explicitly track it as a scaling follow-up. [sharded_atm_latlon_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:492) [sharded_atm_latlon_step.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1586)

4. Minor API/doc gaps: `band_fingerprint` raises `IndexError`, not the promised `ValueError`, for a 0-D input; and the four new shared helpers are missing from `__all__`. [geometry_consistency.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/geometry_consistency.py:23) [geometry_consistency.py](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/geometry_consistency.py:633)

The ocean schema gate is correctly before every `checked_shard_put`; it gates `n_dev`, dtype class, and rank before the per-field collectives. No ocean-local duplicate fingerprint code remains: whole-array fixed-width fingerprints serve broadcast callers (atmosphere/coupler), while per-band fingerprints serve the assert-free ocean put. The import direction is valid: core imports only JAX/NumPy, and ocean depends on core.

Static diff check and Python 3.12 syntax parsing passed.

VERDICT: FIX-FIRST
