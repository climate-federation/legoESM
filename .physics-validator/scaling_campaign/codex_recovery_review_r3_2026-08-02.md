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
session id: 019fc231-764f-7401-b26a-8b4208500c5b
--------
user
Round-3 verify-fixes review. Your round-2 (.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md) gave FIX-FIRST with 6 items. Dispositions claimed:
1. metis doc now: nominal mean target, cells_per_rank_achieved = floor division (bench:751 cited), owned ranges geometric 5120-5121 both scales / metis 5100-5145 @32, 5093-5144 @128 (verified from metadata.partition_metrics), wet ranges kept.
2. LL2048@128 = 65,536 cols/GPU, above floor, floor attribution REMOVED, mechanism OPEN with candidates labelled uninstrumented; times to 4 digits (5.5767/9.6015).
3. atm_latlon_hundreds.sbatch comments corrected (110.6k/55.3k, both above floor).
4. ensemble v3 (resubmitted as job 26628196): rank-0 per-step nodelist echo via bash -c wrapper with ARM_TAG env, sacct table now rc-gated, INTERPRETATION LIMIT paragraph added (screens penalty, attribution needs follow-up, pre+post brackets not full counterbalance).
5. All three scripts' RESULTS loops now isfinite+positive-gated with rc=1 on missing/invalid.
6. s9 np32 SHA: np64/np128 rows record git_sha 7151d12a1 (same job); doc marks np32 row non-reproduction-grade on its own, same-binary PLAUSIBLE.
Also NEW since round-2: cpu_ll2d job 26628073 receipt added to doc+figure: r512 2-D pencil f64 moist np64/128/256/512 = 297.57/161.03/72.06/44.78 ms, eff column 1.00/0.92/1.03/0.83, superlinear 128->256 labelled PLAUSIBLE cache transition; masquerade check documented (n_ranks:512 in rows; metadata.process_count:1 explained as jax-local on mpi4jax lane); pencil factorisation claim REMOVED as unrecorded. Figure panel d retitled 'ico + lat-lon 2-D' with the new series.
Check: doc wording at the cited spots, the bash -c wrapper correctness (ARM_TAG export through srun, PROCID guard, exec "$0" "$@" semantics), the eff arithmetic for cpu_ll2d (297.57*64 vs others), and whether any round-2 item remains unaddressed. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + numbered issues.
codex
I’ll audit the cited recovery document, source scripts, and recorded artifacts against each claimed disposition, including recalculating the new CPU scaling efficiencies and tracing the wrapper’s execution semantics.
exec
/usr/bin/bash -lc "pwd && rg --files -g 'AGENTS.md' -g 'codex_recovery_review_r2_2026-08-02.md' -g '*.sbatch' -g '*scaling*' -g '*.json' | sed -n '1,240p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 305ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
legoesm_cpu_scaling.26448158.log
.zenodo.json
legoesm_cpu_scaling.26447094.log
legoesm_cpu_scaling.26452579.log
legoesm_cpu_scaling.26445986.log
legoesm_cpu_scaling.26447093.log
legoesm_gpu_scaling.26445836.log
docs/performance/scaling/scaling_theoretical_limit_report_2026-06-15.md
docs/performance/scaling/bcw_scaling_status.md
docs/performance/scaling/scaling_review_2026-06-13.md
docs/performance/scaling/crm_les_scaling.md
docs/performance/scaling/scaling_indicators.csv
docs/performance/scaling/scaling_tpu.md
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md
docs/performance/scaling/fig_scaling_caption.md
docs/performance/scaling/amip_mpi_scaling.md
docs/performance/scaling/scaling.md
docs/performance/scaling/scaling_crm_gpu.md
docs/performance/scaling/scaling_levers_audit_2026-06-14.md
docs/performance/scaling/scaling_gpu.md
docs/performance/scaling/scaling_bottleneck_audit_2026-06-10.md
docs/performance/scaling/scaling_levers_audit_2026-06-15.md
scripts/bench/bench_mpas_spmd_scaling.py
scripts/bench/bench_atm_latlon_spmd_scaling.py
scripts/bench/analyze_gpu_scaling.py
scripts/bench/aggregate_cube_shardmap_scaling.py
scripts/bench/bench_dd_scaling.py
scripts/bench/bench_crm_gpu_scaling.py
scripts/bench/bench_ocean_mpas_scaling.py
scripts/bench/aggregate_bcw_scaling.py
scripts/bench/aggregate_scaling_results.py
scripts/bench/bench_halo_ops_scaling.py
scripts/bench/bench_cube_tiled_step_scaling.py
scripts/bench/validate_scaling.sh
scripts/bench/run_levante_gpu_scaling.py
scripts/bench/run_cpu_mpi_scaling.sh
scripts/bench/bench_spectral_les_dd_scaling.py
scripts/bench/run_cpu_mpi_scaling.py
scripts/bench/run_levante_gpu_scaling.sh
scripts/bench/run_scaling_iter222_weak.sh
scripts/bench/bench_ocean_mpi_scaling.py
scripts/bench/run_cpu_mpi_scaling_local.sh
scripts/bench/run_strong_scaling_sweep.sh
scripts/bench/bench_ocean_gpu_scaling.py
scripts/bench/bench_plane_crm_dd_scaling.py
scripts/bench/run_scaling_diagnosis.py
scripts/bench/bcw_scaling_ledger.py
scripts/bench/run_dd_scaling_sweep.sh
scripts/bench/analyze_scaling_results.py
scripts/bench/scaling_summary.py
scripts/bench/bench_coupled_latlon_scaling.py
scripts/bench/slurm_scaling_diagnosis.sh
scripts/bench/bench_ocean_latlon_spmd_scaling.py
scripts/bench/bench_mpi_scaling.py
scripts/bench/jit_profile/run_jit_scaling_sweep.sbatch
scripts/bench/jit_profile/run_jit_compile_profile.sbatch
scripts/bench/jit_profile/diagnose_jit_cache_miss.sbatch
scripts/plot/plot_scaling_dashboard.py
scripts/plot/plot_scaling_paper_figure.py
scripts/plot/plot_gpu_scaling.py
scripts/plot/plot_scaling_laws.py
scripts/plot/plot_scaling_vs_clima.py
scripts/plot/plot_cpu_gpu_scaling_summary.py
scripts/plot/plot_cpu_vs_gpu_scaling.py
scripts/plot/plot_scaling_indicators.py
scripts/plot/plot_levante_gpu_scaling_comparison.py
scripts/plot/plot_scaling_family.py
scripts/plot/plot_scaling_efficiency.py
scripts/plot/plot_iter_scalings.py
scripts/plot/plot_atm_latlon_spmd_scaling.py
scripts/plot/plot_strong_scaling_by_resolution.py
scripts/plot/plot_scaling.py
scripts/plot/plot_bcw_scaling.py
scripts/cluster/reconcile_csw.sbatch
scripts/cluster/csw_oracle.sbatch
docs/scaling/external_scaling_transfer_assessment.md
docs/scaling/atm_latlon_spmd_scaling.md
scripts/cluster/scaling_ginsburg/aimip_amip_finetune.sbatch
scripts/cluster/scaling_ginsburg/aimip_fleet_plot.sbatch
scripts/cluster/scaling_ginsburg/aimip_latlon_sfno_smoke.sbatch
scripts/cluster/scaling_ginsburg/aimip_ace2loss.sbatch
scripts/cluster/scaling_ginsburg/aimip_amip_inference.sbatch
tests/bench/test_scaling_metadata.py
tests/bench/test_scaling_moist_tier.py
tests/bench/test_aggregate_bcw_scaling.py
tests/bench/test_aggregate_cube_shardmap_scaling.py
tests/bench/test_scaling_diagnosis_device_gate.py
tests/bench/test_bench_ocean_mpas_scaling.py
tests/bench/test_bench_cube_tiled_step_scaling.py
tests/bench/test_bcw_scaling_ledger.py
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch
scripts/cluster/scaling_levante/cube_tiled_step.sbatch
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch
scripts/cluster/scaling_levante/diagnosis.sbatch
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch
scripts/cluster/scaling_levante/gpu_scaling.sbatch
scripts/cluster/scaling_levante/prewarm_s10.sbatch
scripts/cluster/scaling_levante/cpu_scaling.sbatch
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm
scripts/data/setup_legoesm_mpi_venv.sbatch
scripts/cluster/fv3_native/modon_case8_c24.sbatch
scripts/cluster/fv3_native/w2_d4bg_sweep.sbatch
scripts/cluster/fv3_native/w2_bounded_smoke.sbatch
scripts/cluster/fv3_native/sw_full_battery.sbatch
scripts/cluster/fv3_native/modon_oracle_recipe.sbatch
scripts/cluster/fv3_native/dsw4_duo_oracle.sbatch
scripts/cluster/fv3_native/extproj_oracle.sbatch
scripts/cluster/fv3_native/dsw3_duo_oracle.sbatch
scripts/cluster/fv3_native/dsw5_duo_oracle.sbatch
scripts/cluster/fetch_duogrid_zenodo.sbatch
scripts/cluster/fv3_native/modon_corner_damp.sbatch
scripts/cluster/scaling_derecho/scaling_cpu.sh
scripts/cluster/fv3_native/k2e_auth_reverify.sbatch
scripts/cluster/scaling_derecho/cube_scaling_cpu_routeb.sh
scripts/cluster/fv3_native/dsw6_duo_oracle.sbatch
scripts/cluster/scaling_derecho/cube_scaling_gpu.sh
scripts/cluster/fv3_native/dsw2_duo_oracle.sbatch
scripts/cluster/fv3_native/w2_duo_nord_ab.sbatch
scripts/cluster/fv3_native/modon_fb_gpu.sbatch
scripts/cluster/fv3_native/w2_bounded_c48_gate.sbatch
scripts/cluster/fv3_native/hs_battery.sbatch
scripts/cluster/fv3_native/duo_stepper_w2.sbatch
scripts/cluster/fv3_native/sw_imprint_battery.sbatch
scripts/cluster/fv3_native/w2_bounded_c24_gate.sbatch
scripts/cluster/fv3_native/modon_case8_c48.sbatch
scripts/cluster/fv3_native/w2_corner_damp.sbatch
scripts/cluster/fv3_native/cornerlag_oracle.sbatch
scripts/cluster/fv3_native/modon_bounded_c24.sbatch
scripts/cluster/fv3_native/dsw1_duo_oracle.sbatch
scripts/cluster/fv3_native/modon_sharpness_ab.sbatch
scripts/cluster/fv3_native/ext_bundle_gate.sbatch
scripts/cluster/fv3_native/modon_dampv_sweep.sbatch
scripts/cluster/fv3_native/modon_sharpness_ab2.sbatch
scripts/cluster/scaling_derecho/finalize_scaling.sh
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs
scripts/cluster/scaling_derecho/cube_scaling_cpu.sh
scripts/cluster/scaling_derecho/ocean_cpu_scaling.pbs
scripts/cluster/scaling_derecho/submit_scaling.sh
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs
scripts/cluster/scaling_derecho/scaling_gpu.sh
packages/core/legoesm/parallel/scaling_diagnostics.py
scripts/cluster/carbon_calibration/train_carbon_params_sif.sbatch
scripts/cluster/carbon_calibration/train_carbon_params.sbatch
scripts/cluster/carbon_calibration/retest_fast.sbatch
scripts/cluster/carbon_calibration/validate_fast_analytic.sbatch
scripts/cluster/carbon_calibration/test_chunked_grad.sbatch
scripts/cluster/carbon_calibration/bisect_chunk.sbatch
scripts/cluster/carbon_calibration/train_carbon_exact_chunked.sbatch
scripts/cluster/carbon_calibration/validate_carbon_trainer.sbatch
scripts/cluster/carbon_calibration/build_sif_observations.sbatch
scripts/cluster/carbon_calibration/train_carbon_params_fast.sbatch
scripts/cluster/compare_reanalysis/preflight_ck_sensitivity.sbatch
scripts/cluster/compare_reanalysis/run_correction_campaign.sbatch
scripts/cluster/compare_reanalysis/preflight_osse.sbatch
scripts/cluster/wb_forecast/campaign_smoke.sbatch
scripts/cluster/wb_forecast/smoke_stage1.sbatch
scripts/cluster/wb_forecast/classical_sweep.sbatch
scripts/cluster/wb_forecast/train_sfno_full_scale.sbatch
scripts/cluster/wb_forecast/wb2_eval.sbatch
scripts/cluster/wb_forecast/train_aimip_neural.sbatch
scripts/cluster/wb_forecast/run_pytest.sbatch
scripts/cluster/wb_forecast/train_sfno_full_scale_ginsburg.sbatch
scripts/cluster/wb_forecast/check_config.sbatch
scripts/cluster/wb_forecast/run_dp_test.sbatch
tests/plot/test_plot_strong_scaling_by_resolution.py
tests/plot/test_cpu_gpu_scaling_summary.py
tests/plot/test_scaling_efficiency_plot.py
tests/plot/test_plot_bcw_scaling.py
tests/plot/test_plot_scaling_family.py
tests/plot/test_plot_cpu_vs_gpu_scaling.py
tests/plot/test_plot_levante_gpu_scaling_comparison.py
tests/ocean/fidelity/fixtures/tier8_global_realistic.json
tests/ocean/fidelity/fixtures/tier5_baroclinic_instability.json
tests/ocean/fidelity/fixtures/tier0_invariants.json
tests/ocean/fidelity/fixtures/tier6_channel_circulation.json
tests/ocean/fidelity/fixtures/tier1_linear_waves.json
tests/ocean/fidelity/fixtures/tier7_dino.json
tests/ocean/fidelity/fixtures/tier4_wind_driven_gyres.json
tests/ocean/fidelity/fixtures/tier3_process_benchmarks.json
tests/ocean/fidelity/fixtures/tier2_geostrophic_thermalwind.json
scripts/cluster/levante/amip_mpas_gpu_chain.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_rrtmgp_L10_cache.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_rrtmgp.sbatch
scripts/cluster/cmip6_coupled/run_unfused_test.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_rrtmgp_L10.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_gray_fullphys.sbatch
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_rrtmgp_validate.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_coupled.sbatch
scripts/cluster/omip_nemo/run_test_prescribed_flow.sbatch
scripts/cluster/omip_nemo/run_trp2_ship1.sbatch
scripts/cluster/omip_nemo/run_ref5yr.sbatch
scripts/cluster/omip_nemo/run_trp4_vfsfix_probe.sbatch
scripts/cluster/omip_nemo/run_ll_ship2.sbatch
scripts/cluster/omip_nemo/validate_changes.sbatch
scripts/cluster/omip_nemo/rerun_mpas7_kpp_r2.sbatch
scripts/cluster/omip_nemo/run_faithful_ll2_base_ab.sbatch
scripts/cluster/omip_nemo/diag_static2.sbatch
scripts/cluster/omip_nemo/run_mpas9_parity.sbatch
scripts/cluster/omip_nemo/run_core2_woasmoke.sbatch
scripts/cluster/omip_nemo/run_core2_woasmoke_dt150.sbatch
scripts/cluster/omip_nemo/run_faithful_trp2_p46.sbatch
scripts/cluster/omip_nemo/build_nemo.sbatch
scripts/cluster/omip_nemo/run_dino_r1_ablate.sbatch
scripts/cluster/omip_nemo/run_smoke.sbatch
scripts/cluster/omip_nemo/probe_compute.sbatch
scripts/cluster/omip_nemo/diag_woa_static.sbatch
scripts/cluster/omip_nemo/dl_mesh.sbatch
scripts/cluster/omip_nemo/run_scm_twins.sbatch
scripts/cluster/omip_nemo/run_faithful_p45_smoke.sbatch
scripts/cluster/omip_nemo/run_trp2_ship1_burst.sbatch
scripts/cluster/omip_nemo/run_trp5_eice90.sbatch
scripts/cluster/omip_nemo/test_compare.sbatch
scripts/cluster/omip_nemo/_ab_gm_treguier.sbatch
scripts/cluster/omip_nemo/run_dino_r1_zco.sbatch
scripts/cluster/omip_nemo/run_trp4a_eice_probe.sbatch
scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch
scripts/cluster/omip_nemo/run_ll8_kppeice.sbatch
scripts/cluster/omip_nemo/run_dino_r1_tke.sbatch
scripts/cluster/omip_nemo/stage_woa.sbatch
scripts/cluster/omip_nemo/run_trp2_iceinit.sbatch
scripts/cluster/omip_nemo/run_dino_r1_exact.sbatch
scripts/cluster/omip_nemo/run_faithful_ll2_p45.sbatch
scripts/cluster/omip_nemo/run_ico7_dm30.sbatch
scripts/cluster/omip_nemo/_ab_tke_surface_bc.sbatch
scripts/cluster/omip_nemo/run_trp2_tke_iceinit.sbatch
scripts/cluster/omip_nemo/rerun_ll3_ri015.sbatch
scripts/cluster/omip_nemo/_ab_tke_mxl3.sbatch
scripts/cluster/omip_nemo/_diag_kappa_gm.sbatch
scripts/cluster/omip_nemo/run_multiprocess_cpu_equiv.sbatch
scripts/cluster/omip_nemo/run_eorca025_4gpu_multinode.sbatch
scripts/cluster/omip_nemo/run_ll_ship1.sbatch
scripts/cluster/omip_nemo/run_core2_tripole.sbatch
scripts/cluster/omip_nemo/run_core2_nudgesmoke.sbatch
scripts/cluster/omip_nemo/run_ref10yr.sbatch
scripts/cluster/omip_nemo/compare_day90.sbatch
scripts/cluster/omip_nemo/run_mpas14_tke.sbatch
scripts/cluster/omip_nemo/rerun_dino_l2_r2.sbatch

exec
/usr/bin/bash -lc "rg --files -uu | rg '("'^|/)AGENTS'"\\.md"'$|'"\\.physics-validator/scaling_campaign|26628073|26628196|recovery_review'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 273ms:
cpu_ll2d.26628073.log
.physics-validator/scaling_campaign/review10_batch.md
.physics-validator/scaling_campaign/codex_scaleout_improvements_2026-07-31.md
.physics-validator/scaling_campaign/review8_final.md
.physics-validator/scaling_campaign/review12_gated_barrier.md
.physics-validator/scaling_campaign/review2.md
.physics-validator/scaling_campaign/review6_final.md
.physics-validator/scaling_campaign/review16_cube_halo.md
.physics-validator/scaling_campaign/review14_scaleout.md
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md
.physics-validator/scaling_campaign/review11_np4_strategy.md
.physics-validator/scaling_campaign/iter5_ocean.patch
.physics-validator/scaling_campaign/review19_meshcap.md
.physics-validator/scaling_campaign/review5.md
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md
.physics-validator/scaling_campaign/iter2_diff.patch
.physics-validator/scaling_campaign/review15_plateaus.md
.physics-validator/scaling_campaign/review9_gate_reframe.md
.physics-validator/scaling_campaign/review1.md
.physics-validator/scaling_campaign/review17_globalmem.md
.physics-validator/scaling_campaign/review3.md
.physics-validator/scaling_campaign/evidence_for_strategy.md
.physics-validator/scaling_campaign/review18_memfix.md
.physics-validator/scaling_campaign/review13_precision.md
.physics-validator/scaling_campaign/iter1_diff.patch
.physics-validator/scaling_campaign/review7_mechanism.md
.physics-validator/scaling_campaign/iter3_diff.patch
.physics-validator/scaling_campaign/iter4_code.patch

exec
/usr/bin/bash -lc "git status --short && sed -n '1,260p' .physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md && sed -n '1,320p' .physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 429ms:
 M docs/performance/scaling/levante_campaign_2026-07-24.md
 M scripts/plot/plot_scaling_paper_figure.py
?? .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md
?? .physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md
?? .physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md
?? scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch
?? scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch
?? scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
?? scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch
?? scripts/cluster/scaling_levante/prewarm_s10.sbatch
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
session id: 019fc226-2ada-7320-922e-df183bca9f5a
--------
user
Round-2 re-review after your FIX-FIRST (transcript: .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md). Changes made since:
1. docs/performance/scaling/levante_campaign_2026-07-24.md — Phase-3 section rewritten: owned-vs-wet cells distinction (metis wet min/max 78000-102880 at np128), METIS claim scoped to 'this configuration on this lane', scale-out-not-rank-count wording, block:cyclic socket-distribution correction, UCX warn caveat, s9 executed padded cells 2621568, peak scoped to synthetic physics=none bench, floor 'consistent with' not confirmed, weak-pair claim RETRACTED loudly with the three confounds named. New sections: recovered atm128 receipt (LL2048@128 f32 5.58ms/39.11 GC/s vs @64 6.7324 job 26502539 same protocol, eff 0.60), hundreds-push job table.
2. scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch — v2: per-pid wait status collection, solo_pre AND solo_post brackets, steps 5000/warmup 100 (~60s stepping window vs launch skew), per-arm epoch echoes + end-of-job sacct step nodelist/Start/End table as overlap evidence, receipts-validity gate over all 6 files, confirm bar reworded to guaranteed >=3.64x solo.
3. scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch — s8 lloyd0 np8/16/32 matched to s9 protocol (steps 12/warmup 3, sfc, reorder-for 128) with falsifiability block.
4. scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch (LL2304@96/192 + LL2880@192 f32; divisibility rationale), atm_latlon2d_cpu_hundreds.sbatch (r512 2-D pencil np64-512 f64 moist, wall-pole labelled), prewarm_s10.sbatch (shared partition).
5. scripts/plot/plot_scaling_paper_figure.py — LL2048 series (64,6.73)/(128,5.58) + f64@128 scatter; mesh-family note on MPAS panel.
Verify each of your 12 items is addressed or explicitly deferred; check new scripts for bash/srun bugs (rc propagation, step resource flags, divisibility 2304/96, 2304/192, 2880/192, r512 pencil factorability at np512); check the remaining claims' arithmetic. End with 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + numbered issues.
codex
I’ll trace the prior 12 findings against the revised artifacts, then independently audit the new job scripts and the remaining numerical claims.
exec
/usr/bin/bash -lc "pwd && rg --files -g 'AGENTS.md' -g 'codex_recovery_review_2026-08-02.md' -g 'levante_campaign_2026-07-24.md' -g 'mpas_s9_ensemble.sbatch' -g 'mpas_s8_lloyd0_ladder.sbatch' -g 'atm_latlon_hundreds.sbatch' -g 'atm_latlon2d_cpu_hundreds.sbatch' -g 'prewarm_s10.sbatch' -g 'plot_scaling_paper_figure.py' -g 'README*' -g 'pyproject.toml' -g 'pytest.ini' -g 'tox.ini' -g 'setup.cfg'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 377ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
README.md
pyproject.toml
packages/ocean/README.md
packages/ocean/pyproject.toml
packages/core/README.md
packages/core/pyproject.toml
packages/atmosphere/pyproject.toml
packages/coupler/README.md
packages/coupler/pyproject.toml
packages/ice/README.md
packages/ice/pyproject.toml
packages/atmosphere/README.md
packages/tools/README.md
packages/tools/pyproject.toml
docs/dev-notes/README.md
scripts/cluster/scaling_levante/README.md
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch
scripts/cluster/scaling_levante/prewarm_s10.sbatch
packages/ml/README.md
packages/ml/pyproject.toml
scripts/cluster/aimip_scale/README.md
docs/performance/scaling/levante_campaign_2026-07-24.md
packages/land/README.md
packages/land/pyproject.toml
scripts/cluster/scaling_derecho/README.md
scripts/cluster/derecho_lmip/README.md
scripts/README.md
config/4DVar_single/README.md
scripts/plot/plot_scaling_paper_figure.py
scripts/experiment/dino/README.md
config/templates/README.md
tests/validation/README_DYCORE_PROGRESSION.md
scripts/run/mpas_realistic_geometry/README.md
config/machines/README.md
scripts/validate/ocean_fidelity/dino_1226/README.md
config/aimip/wbcompare/classical_sweep/README.md

exec
/usr/bin/bash -lc "sed -n '1,260p' .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md && printf '\\n--- DOCUMENT ---\\n' && sed -n '1,420p' docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 246ms:
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
session id: 019fc214-68e7-7b63-b44a-89f0f7351ad0
--------
user
Adversarial review of a scaling-campaign iteration. Attack arithmetic, logic, confounds, and overclaim. Context: docs/performance/scaling/levante_campaign_2026-07-24.md, new final section 'Phase-3 receipts recovered after the 2026-07-31 session drop (2026-08-02)', plus the edit to scripts/plot/plot_scaling_paper_figure.py (git diff HEAD -- scripts/plot/plot_scaling_paper_figure.py) and the new scripts/tmp/mpas_s9_ensemble.sbatch.

Claims to attack:
1. METIS A/B (job 26600094, all arms 5120 cells/rank, f64, 32 ranks/node): A s7np32 geometric 189.82ms, B s8np128 geometric 308.96, C s7np32 metis 191.99, D s8np128 metis 333.39, E s8np128 metis block:block 537.91. My claims: rank-count term geometric 1.628 vs metis 1.736 -> partition-quality lever DEAD/refuted; block:cyclic mandatory (E/D=1.61); lever#2 closed refuted on this lane.
2. s9 GPU ladder (job 26600095, f32 sfc lloyd0, 2621442 cells L26): np32 12.47ms/5.47GC/s, np64 9.60/7.10, np128 11.48/5.94. Claims: new MPAS peak 7.10 GC/s = 2.2x s8 best 3.23 GC/s (s8 655362 cells, np64 5.27ms, 26 levels -> check GC/s arithmetic); 64->128 anti-scales at 20.5k cells/GPU consistent with ~30k floor; weak pairs s8->s9 at matched cells/GPU: 6.92->12.47 (0.55), 7.10->9.60 (0.74), 8.13->11.48 (0.71) -> ~1.4x matched-tile cost per 4x ranks = GPU rank-count term.
3. Ensemble job design (mpas_s9_ensemble.sbatch): 4 concurrent 32-GPU srun steps in one 32-node allocation, per-step SLURM_STEP_NODELIST coordinator autodetect, shared jobid-derived port claimed safe because hosts differ; solo control in-job; CONFIRM bar max(replica)<=1.10x solo. Attack: step isolation, GPU binding, srun flag errors, cache effects, whether solo-then-concurrent ordering biases, whether 12 steps is enough discrimination.
Check every ratio and GC/s conversion. Flag any claim that should be labelled PLAUSIBLE rather than CONFIRMED, any protocol mismatch making a comparison confounded, and any sbatch bug. End with 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' plus a numbered issue list.
codex
I’ll audit the receipts and scheduler script against the stated claims, recompute every metric, and look for comparison and isolation confounds.
exec
/usr/bin/bash -lc "pwd && rg --files -g 'levante_campaign_2026-07-24.md' -g 'plot_scaling_paper_figure.py' -g 'mpas_s9_ensemble.sbatch' -g 'AGENTS.md' -g 'CLAUDE.md' | sort && git status --short && git diff --check HEAD -- scripts/plot/plot_scaling_paper_figure.py && git diff HEAD -- scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 676ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
CLAUDE.md
docs/performance/scaling/levante_campaign_2026-07-24.md
scripts/plot/plot_scaling_paper_figure.py
 M docs/performance/scaling/levante_campaign_2026-07-24.md
 M scripts/plot/plot_scaling_paper_figure.py
?? .physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md
diff --git a/scripts/plot/plot_scaling_paper_figure.py b/scripts/plot/plot_scaling_paper_figure.py
index 93056ac19..a67ca53a8 100644
--- a/scripts/plot/plot_scaling_paper_figure.py
+++ b/scripts/plot/plot_scaling_paper_figure.py
@@ -33,7 +33,8 @@ from matplotlib.lines import Line2D
 SOURCES = {
     "atm_latlon": "26450848/26453240/26449147 (f32), 26494902 (f64)",
     "atm_cube": "26452894/26453782",
-    "atm_mpas": "26454476/26454618/26486288/26493638/26493734",
+    "atm_mpas": "26454476/26454618/26486288/26493638/26493734, "
+                "s8 np32-128 26549646/26538474, s9 26600095",
     "atm_ico_cpu": "26495083 (f32), 26495437 (f64) — both block:cyclic",
     "oc_latlon": "26460444-501/26460365/26493592",
     "oc_tripole": "26493837/26493648",
@@ -56,10 +57,13 @@ PANELS = [
         note="f64 pending",
     ),
     dict(
-        key="atm_mpas", title="MPAS icosahedral", sub="L8 28 km L26 · A100 NCCL",
-        series=[("float32", [(2, 19.90), (4, 14.12), (8, 6.92), (16, 7.10)]),
-                ("float64", [(2, 38.34), (4, 20.09), (8, 18.98)])],
-        note="incl. fusion fix",
+        key="atm_mpas", title="MPAS icosahedral", sub="subdiv-8/9 L26 · A100 NCCL",
+        series=[("float32 (subdiv-8)", [(2, 19.90), (4, 14.12), (8, 6.92),
+                                        (16, 7.10), (32, 8.13), (64, 5.27),
+                                        (128, 6.47)]),
+                ("float32 (subdiv-9)", [(32, 12.47), (64, 9.60), (128, 11.48)]),
+                ("float64 (subdiv-8)", [(2, 38.34), (4, 20.09), (8, 18.98)])],
+        note="s9 weak eff 0.55–0.74\nvs s8 at matched tile",
     ),
     dict(
         key="atm_ico_cpu", title="icosahedral", sub="subdiv-7 L26 · Milan CPU–MPI",
@@ -102,11 +106,13 @@ COLORS = {"float32": "#0072B2", "float64": "#D55E00",
           "float64 (packed)": "#E69F00",
           "mixed (f64 store)": "#009E73",
           "float32 (C768)": "#0072B2", "float32 (C384)": "#56B4E9",
+          "float32 (subdiv-8)": "#0072B2", "float32 (subdiv-9)": "#56B4E9",
           "float64 (subdiv-7)": "#D55E00", "float64 (subdiv-8)": "#E69F00"}
 MARKERS = {"float32": "o", "float64": "s", "mixed (f64 store)": "D",
            "f32 · LL1536/2048 @64": "*",
            "float64 (packed)": "s",
            "float32 (C768)": "o", "float32 (C384)": "^",
+           "float32 (subdiv-8)": "o", "float32 (subdiv-9)": "^",
            "float64 (subdiv-7)": "s", "float64 (subdiv-8)": "v"}
 
 

exec
/usr/bin/bash -lc "sed -n '1,240p' CLAUDE.md && printf '\\n--- campaign tail ---\\n' && tail -n 320 docs/performance/scaling/levante_campaign_2026-07-24.md && printf '\\n--- ensemble candidates ---\\n' && (rg --files --hidden --no-ignore | rg '("'^|/)mpas_s9_ensemble'"\\.sbatch"'$|mpas.*ensemble.*'"\\.sbatch"'$'"' || true) && printf '\\n--- plot remainder ---\\n' && sed -n '1,280p' scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 256ms:
# legoESM Claude Memory

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
- **MPI AD**: `global_sum_mpi` (allreduce SUM) full VJP. MPI halo: `_sendrecv_vjp` custom_vjp. `fix_mass`/`zero_mean_tendency` flow grads via global reductions. `global_max_mpi`/`global_min_mpi` NOT diff — keep out of losses.

## Operating Mode
- Nontrivial task: short plan before edit. Read nearby impl+tests first. Ambiguous numerics/physics/API: ask.
- Minimal diffs. No unrelated refactor in bug fix.
- **Codex adversarial review MANDATORY after any major code implementation/change.** Trigger: new module/feature, dycore/physics/parallel/ocean/land/ice/coupler/training edit, >~50 LOC, multi-file, or anything touching numerics/AD/JIT/pytree/conservation. Run the **iterate-with-codex agent** loop below (`/codex:adversarial-review --wait` → fix flagged → `/codex:review --wait` → repeat until clean or 30 iter) BEFORE declaring done; report that review ran + verdict.
  **If the review SUBAGENT dies (spend limit, API error), that is NOT a review
  waiver — the codex CLI is a separate binary with separate credentials and is
  usually still reachable: `codex exec --sandbox read-only -C <repo> "<prompt>"`
  (`which codex`, `~/.codex/auth.json`). Try the CLI directly before ever
  proceeding unreviewed, and if BOTH are unavailable say "UNREVIEWED" in every
  status until one succeeds.** 2026-07-26: a subagent hit a monthly spend limit
  and many iterations ran unreviewed while the CLI worked fine the whole time. Exempt: trivial/mechanical edits (typo, comment, rename, doc/markdown/`.tex`-only, single config value).
- **Pre-impl search mandatory**: before new fn/helper/class/operator/diagnostic/init/load/loss/numerical routine, grep `src/legoesm/` for similar names/docstrings/formulas in `thermo.py`, `constants.py`, `eos.py`, `ml/loss.py`, `diagnostics/`, `core/`, `atmosphere/physics/_shared.py`. State searched+found. Similar exists → extend/factor.
- **Shared utilities — never re-derive** (prod, scripts, validators, plotters, tests, notebooks, probes):
  - Constants: `from legoesm import constants` → `T_freeze`, `R_d`, `c_pd`, `L_v`, `R_v`, `epsilon`, `g`, `p_ref`, `kappa`, `sigma_sb`, `T_freeze_ocean`. No literals `273.15`/`287.0`/`1004.64`/`2.501e6`/`461.51`/`0.622`/`9.80616`/`6.371e6`/`7.292e-5`.
  - Saturation: `from legoesm.thermo import saturation_vapor_pressure, saturation_mixing_ratio, saturation_mixing_ratio_ice`. No re-impl Tetens/Magnus/Clausius–Clapeyron (plotters incl). Why: re-derived `e_sat=611.2*exp(17.67*Tc/(Tc+243.5))` diverged from model → false supersat in CI.
  - Column integrals: `legoesm.diagnostics.column_integrals` (`column_water_vapor`). No inline `jnp.sum(q*p_s*dsigma)/g`.
  - Losses: `ml/loss.py` (`area_weighted_mse`, `spectral_loss`, `per_variable_mse`).
  - Optimizer: `ml/training.create_optimizer()` (warmup+cosine+clip).
  - SCM-RCE gradient tuning: reuse `scripts/run/run_scm_rce_campaign.py` for CRM
    reference extraction / SCM evaluation and `legoesm.training.scm_rce_metrics`
    for the normalized profile score. No duplicated RCE profile numerics.
  - SCM-RCE param training defaults to MUON via `ml.training.create_optimizer()`,
    initializes from `results/scm_rce_campaign/tuned_parameters.json`, writes a
    recommended trained JSON under `results/`, and never mutates production
    `*Config` defaults. Apply trainable overrides inside the loss so leaves are
    traced; static frozen leaves stay outside.
  - Every new `.py`, including `scripts/run/*.py` drivers, gets a direct test.
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

## Attribution Gates — MANDATORY, each from a real 2026-07 failure
Model is near operational. Every rule below is mechanical: satisfy it or state
explicitly that you did not. "I was careful" is not compliance.

- **PROVE THE PATH EXECUTES before blaming a line.** Naming a file:line as the
  cause requires showing that line runs in the configuration under test: print
  the ENCLOSING FUNCTION (`awk` the nearest `def` above it) and confirm the
  active lane/driver calls it. FAILURE: blamed the positivity clamps at
  `model_driver.py:10923` for the century's water source; they live in
  `_run_per_step` while the century runs `_run_mpas`, which contains no
  moisture clamp at all. A fix was nearly written for a lane the run never
  touches. Same class as reading an entry point instead of the full path.
- **REUSING A REFERENCE IMPL MEANS PORTING ITS EXCLUSIONS, not just its
  formula.** State which of the reference's guards/scope conditions you kept
  and which you dropped, with a reason for each. FAILURE: copied
  `spectral_les_moist.conserving_positive` but not its `n_water` split, so the
  column-conserving borrow was applied to number concentrations
  (`N_c`/`N_i`/`N_r`) — unphysical, and it fed M2005 deposition (~N_i^(2/3)),
  producing a fake "accelerating dry bias" that was reported before being
  caught.
- **A TEST THAT INSPECTS SOURCE MUST NAME THE SYMBOL THAT RUNS, and must be
  shown to FAIL when the feature is removed.** An `inspect.getsource(X)`
  assertion where X is a delegating wrapper passes while proving nothing.
  FAILURE: asserted against `MPASPrimitiveEquationModel.step`; the floors are
  in `_step_jit`.
- **TOOL STATUS IS NOT EVIDENCE — read the output tail.** An exit code without
  the tool's own success line (pytest's `N passed`, "COMPLETED in Xs") is
  UNVERIFIED; OOM kills and timeouts can surface as success. FAILURE: reported
  a regression suite green on exit-0 that was actually `Out Of Memory` mid-run.
  Quote the decisive line when claiming a suite passed.
- **EVERY BASELINE/ALLOW-LIST REASON STRING IS A CLAIM — verify it in code
  before writing it.** A plausible-sounding reason permanently hides a real
  defect. FAILURE: classified `convective_buoyancy_death_memory` as "carried in
  SegmentCarry" (it is not — the leaf `BechtoldConfig.buoyancy_death_memory`
  exists and nothing maps to it), asserted an `SBMConfig.precip_efficiency`
  leaf that does not exist, and credited `micro_substeps` to a consumer that
  reads `args.`, not the config field.
- **RATE / TENDENCY / SKILL COMPARISONS: identical windows on BOTH sides, and
  print the window next to the number.** Differing spans is a confound, not a
  result. FAILURE: TCW over days 190-530 vs CMOR year 1 gave "+38.7 kg/m2/yr";
  matched windows gave +13.4. Extends the existing controlled-comparison rule
  to derived rates.
- **A DIAGNOSTIC'S PRINTED PRECISION BOUNDS THE RATE YOU CAN CLAIM.** Log CWV
  at 0.1 kg/m2 over 8 days resolves only ~±4.6 kg/m2/yr — do not report a
  trend inside one quantum. Prefer fp64 from model state (checkpoints) over
  parsed log lines. Same class as the throughput-quantization error.
- **`JAX_ENABLE_X64=1` on any numerics/conservation test.** An fp32 mismatch is
  NOT a failure until re-run with x64; and a *new* failure is not yours until
  reproduced with your change stashed. Do both before reporting a regression.
- **RUN-TARGET PARAMS ARE ABSOLUTE (`TARGET_DAYS`), and "latest checkpoint"
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
session id: 019fc231-764f-7401-b26a-8b4208500c5b
--------
user
Round-3 verify-fixes review. Your round-2 (.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md) gave FIX-FIRST with 6 items. Dispositions claimed:
1. metis doc now: nominal mean target, cells_per_rank_achieved = floor division (bench:751 cited), owned ranges geometric 5120-5121 both scales / metis 5100-5145 @32, 5093-5144 @128 (verified from metadata.partition_metrics), wet ranges kept.
2. LL2048@128 = 65,536 cols/GPU, above floor, floor attribution REMOVED, mechanism OPEN with candidates labelled uninstrumented; times to 4 digits (5.5767/9.6015).
3. atm_latlon_hundreds.sbatch comments corrected (110.6k/55.3k, both above floor).
4. ensemble v3 (resubmitted as job 26628196): rank-0 per-step nodelist echo via bash -c wrapper with ARM_TAG env, sacct table now rc-gated, INTERPRETATION LIMIT paragraph added (screens penalty, attribution needs follow-up, pre+post brackets not full counterbalance).
5. All three scripts' RESULTS loops now isfinite+positive-gated with rc=1 on missing/invalid.
6. s9 np32 SHA: np64/np128 rows record git_sha 7151d12a1 (same job); doc marks np32 row non-reproduction-grade on its own, same-binary PLAUSIBLE.
Also NEW since round-2: cpu_ll2d job 26628073 receipt added to doc+figure: r512 2-D pencil f64 moist np64/128/256/512 = 297.57/161.03/72.06/44.78 ms, eff column 1.00/0.92/1.03/0.83, superlinear 128->256 labelled PLAUSIBLE cache transition; masquerade check documented (n_ranks:512 in rows; metadata.process_count:1 explained as jax-local on mpi4jax lane); pencil factorisation claim REMOVED as unrecorded. Figure panel d retitled 'ico + lat-lon 2-D' with the new series.
Check: doc wording at the cited spots, the bash -c wrapper correctness (ARM_TAG export through srun, PROCID guard, exec "$0" "$@" semantics), the eff arithmetic for cpu_ll2d (297.57*64 vs others), and whether any round-2 item remains unaddressed. End 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' + numbered issues.
codex
I’ll audit the cited recovery document, source scripts, and recorded artifacts against each claimed disposition, including recalculating the new CPU scaling efficiencies and tracing the wrapper’s execution semantics.
exec
/usr/bin/bash -lc "pwd && rg --files -g 'AGENTS.md' -g 'codex_recovery_review_r2_2026-08-02.md' -g '*.sbatch' -g '*scaling*' -g '*.json' | sed -n '1,240p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 305ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
legoesm_cpu_scaling.26448158.log
.zenodo.json
legoesm_cpu_scaling.26447094.log
legoesm_cpu_scaling.26452579.log
legoesm_cpu_scaling.26445986.log
legoesm_cpu_scaling.26447093.log
legoesm_gpu_scaling.26445836.log
docs/performance/scaling/scaling_theoretical_limit_report_2026-06-15.md
docs/performance/scaling/bcw_scaling_status.md
docs/performance/scaling/scaling_review_2026-06-13.md
docs/performance/scaling/crm_les_scaling.md
docs/performance/scaling/scaling_indicators.csv
docs/performance/scaling/scaling_tpu.md
docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md
docs/performance/scaling/fig_scaling_caption.md
docs/performance/scaling/amip_mpi_scaling.md
docs/performance/scaling/scaling.md
docs/performance/scaling/scaling_crm_gpu.md
docs/performance/scaling/scaling_levers_audit_2026-06-14.md
docs/performance/scaling/scaling_gpu.md
docs/performance/scaling/scaling_bottleneck_audit_2026-06-10.md
docs/performance/scaling/scaling_levers_audit_2026-06-15.md
scripts/bench/bench_mpas_spmd_scaling.py
scripts/bench/bench_atm_latlon_spmd_scaling.py
scripts/bench/analyze_gpu_scaling.py
scripts/bench/aggregate_cube_shardmap_scaling.py
scripts/bench/bench_dd_scaling.py
scripts/bench/bench_crm_gpu_scaling.py
scripts/bench/bench_ocean_mpas_scaling.py
scripts/bench/aggregate_bcw_scaling.py
scripts/bench/aggregate_scaling_results.py
scripts/bench/bench_halo_ops_scaling.py
scripts/bench/bench_cube_tiled_step_scaling.py
scripts/bench/validate_scaling.sh
scripts/bench/run_levante_gpu_scaling.py
scripts/bench/run_cpu_mpi_scaling.sh
scripts/bench/bench_spectral_les_dd_scaling.py
scripts/bench/run_cpu_mpi_scaling.py
scripts/bench/run_levante_gpu_scaling.sh
scripts/bench/run_scaling_iter222_weak.sh
scripts/bench/bench_ocean_mpi_scaling.py
scripts/bench/run_cpu_mpi_scaling_local.sh
scripts/bench/run_strong_scaling_sweep.sh
scripts/bench/bench_ocean_gpu_scaling.py
scripts/bench/bench_plane_crm_dd_scaling.py
scripts/bench/run_scaling_diagnosis.py
scripts/bench/bcw_scaling_ledger.py
scripts/bench/run_dd_scaling_sweep.sh
scripts/bench/analyze_scaling_results.py
scripts/bench/scaling_summary.py
scripts/bench/bench_coupled_latlon_scaling.py
scripts/bench/slurm_scaling_diagnosis.sh
scripts/bench/bench_ocean_latlon_spmd_scaling.py
scripts/bench/bench_mpi_scaling.py
scripts/bench/jit_profile/run_jit_scaling_sweep.sbatch
scripts/bench/jit_profile/run_jit_compile_profile.sbatch
scripts/bench/jit_profile/diagnose_jit_cache_miss.sbatch
scripts/plot/plot_scaling_dashboard.py
scripts/plot/plot_scaling_paper_figure.py
scripts/plot/plot_gpu_scaling.py
scripts/plot/plot_scaling_laws.py
scripts/plot/plot_scaling_vs_clima.py
scripts/plot/plot_cpu_gpu_scaling_summary.py
scripts/plot/plot_cpu_vs_gpu_scaling.py
scripts/plot/plot_scaling_indicators.py
scripts/plot/plot_levante_gpu_scaling_comparison.py
scripts/plot/plot_scaling_family.py
scripts/plot/plot_scaling_efficiency.py
scripts/plot/plot_iter_scalings.py
scripts/plot/plot_atm_latlon_spmd_scaling.py
scripts/plot/plot_strong_scaling_by_resolution.py
scripts/plot/plot_scaling.py
scripts/plot/plot_bcw_scaling.py
scripts/cluster/reconcile_csw.sbatch
scripts/cluster/csw_oracle.sbatch
docs/scaling/external_scaling_transfer_assessment.md
docs/scaling/atm_latlon_spmd_scaling.md
scripts/cluster/scaling_ginsburg/aimip_amip_finetune.sbatch
scripts/cluster/scaling_ginsburg/aimip_fleet_plot.sbatch
scripts/cluster/scaling_ginsburg/aimip_latlon_sfno_smoke.sbatch
scripts/cluster/scaling_ginsburg/aimip_ace2loss.sbatch
scripts/cluster/scaling_ginsburg/aimip_amip_inference.sbatch
tests/bench/test_scaling_metadata.py
tests/bench/test_scaling_moist_tier.py
tests/bench/test_aggregate_bcw_scaling.py
tests/bench/test_aggregate_cube_shardmap_scaling.py
tests/bench/test_scaling_diagnosis_device_gate.py
tests/bench/test_bench_ocean_mpas_scaling.py
tests/bench/test_bench_cube_tiled_step_scaling.py
tests/bench/test_bcw_scaling_ledger.py
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch
scripts/cluster/scaling_levante/cube_tiled_step.sbatch
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch
scripts/cluster/scaling_levante/diagnosis.sbatch
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch
scripts/cluster/scaling_levante/gpu_scaling.sbatch
scripts/cluster/scaling_levante/prewarm_s10.sbatch
scripts/cluster/scaling_levante/cpu_scaling.sbatch
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm
scripts/data/setup_legoesm_mpi_venv.sbatch
scripts/cluster/fv3_native/modon_case8_c24.sbatch
scripts/cluster/fv3_native/w2_d4bg_sweep.sbatch
scripts/cluster/fv3_native/w2_bounded_smoke.sbatch
scripts/cluster/fv3_native/sw_full_battery.sbatch
scripts/cluster/fv3_native/modon_oracle_recipe.sbatch
scripts/cluster/fv3_native/dsw4_duo_oracle.sbatch
scripts/cluster/fv3_native/extproj_oracle.sbatch
scripts/cluster/fv3_native/dsw3_duo_oracle.sbatch
scripts/cluster/fv3_native/dsw5_duo_oracle.sbatch
scripts/cluster/fetch_duogrid_zenodo.sbatch
scripts/cluster/fv3_native/modon_corner_damp.sbatch
scripts/cluster/scaling_derecho/scaling_cpu.sh
scripts/cluster/fv3_native/k2e_auth_reverify.sbatch
scripts/cluster/scaling_derecho/cube_scaling_cpu_routeb.sh
scripts/cluster/fv3_native/dsw6_duo_oracle.sbatch
scripts/cluster/scaling_derecho/cube_scaling_gpu.sh
scripts/cluster/fv3_native/dsw2_duo_oracle.sbatch
scripts/cluster/fv3_native/w2_duo_nord_ab.sbatch
scripts/cluster/fv3_native/modon_fb_gpu.sbatch
scripts/cluster/fv3_native/w2_bounded_c48_gate.sbatch
scripts/cluster/fv3_native/hs_battery.sbatch
scripts/cluster/fv3_native/duo_stepper_w2.sbatch
scripts/cluster/fv3_native/sw_imprint_battery.sbatch
scripts/cluster/fv3_native/w2_bounded_c24_gate.sbatch
scripts/cluster/fv3_native/modon_case8_c48.sbatch
scripts/cluster/fv3_native/w2_corner_damp.sbatch
scripts/cluster/fv3_native/cornerlag_oracle.sbatch
scripts/cluster/fv3_native/modon_bounded_c24.sbatch
scripts/cluster/fv3_native/dsw1_duo_oracle.sbatch
scripts/cluster/fv3_native/modon_sharpness_ab.sbatch
scripts/cluster/fv3_native/ext_bundle_gate.sbatch
scripts/cluster/fv3_native/modon_dampv_sweep.sbatch
scripts/cluster/fv3_native/modon_sharpness_ab2.sbatch
scripts/cluster/scaling_derecho/finalize_scaling.sh
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs
scripts/cluster/scaling_derecho/cube_scaling_cpu.sh
scripts/cluster/scaling_derecho/ocean_cpu_scaling.pbs
scripts/cluster/scaling_derecho/submit_scaling.sh
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs
scripts/cluster/scaling_derecho/scaling_gpu.sh
packages/core/legoesm/parallel/scaling_diagnostics.py
scripts/cluster/carbon_calibration/train_carbon_params_sif.sbatch
scripts/cluster/carbon_calibration/train_carbon_params.sbatch
scripts/cluster/carbon_calibration/retest_fast.sbatch
scripts/cluster/carbon_calibration/validate_fast_analytic.sbatch
scripts/cluster/carbon_calibration/test_chunked_grad.sbatch
scripts/cluster/carbon_calibration/bisect_chunk.sbatch
scripts/cluster/carbon_calibration/train_carbon_exact_chunked.sbatch
scripts/cluster/carbon_calibration/validate_carbon_trainer.sbatch
scripts/cluster/carbon_calibration/build_sif_observations.sbatch
scripts/cluster/carbon_calibration/train_carbon_params_fast.sbatch
scripts/cluster/compare_reanalysis/preflight_ck_sensitivity.sbatch
scripts/cluster/compare_reanalysis/run_correction_campaign.sbatch
scripts/cluster/compare_reanalysis/preflight_osse.sbatch
scripts/cluster/wb_forecast/campaign_smoke.sbatch
scripts/cluster/wb_forecast/smoke_stage1.sbatch
scripts/cluster/wb_forecast/classical_sweep.sbatch
scripts/cluster/wb_forecast/train_sfno_full_scale.sbatch
scripts/cluster/wb_forecast/wb2_eval.sbatch
scripts/cluster/wb_forecast/train_aimip_neural.sbatch
scripts/cluster/wb_forecast/run_pytest.sbatch
scripts/cluster/wb_forecast/train_sfno_full_scale_ginsburg.sbatch
scripts/cluster/wb_forecast/check_config.sbatch
scripts/cluster/wb_forecast/run_dp_test.sbatch
tests/plot/test_plot_strong_scaling_by_resolution.py
tests/plot/test_cpu_gpu_scaling_summary.py
tests/plot/test_scaling_efficiency_plot.py
tests/plot/test_plot_bcw_scaling.py
tests/plot/test_plot_scaling_family.py
tests/plot/test_plot_cpu_vs_gpu_scaling.py
tests/plot/test_plot_levante_gpu_scaling_comparison.py
tests/ocean/fidelity/fixtures/tier8_global_realistic.json
tests/ocean/fidelity/fixtures/tier5_baroclinic_instability.json
tests/ocean/fidelity/fixtures/tier0_invariants.json
tests/ocean/fidelity/fixtures/tier6_channel_circulation.json
tests/ocean/fidelity/fixtures/tier1_linear_waves.json
tests/ocean/fidelity/fixtures/tier7_dino.json
tests/ocean/fidelity/fixtures/tier4_wind_driven_gyres.json
tests/ocean/fidelity/fixtures/tier3_process_benchmarks.json
tests/ocean/fidelity/fixtures/tier2_geostrophic_thermalwind.json
scripts/cluster/levante/amip_mpas_gpu_chain.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_rrtmgp_L10_cache.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_rrtmgp.sbatch
scripts/cluster/cmip6_coupled/run_unfused_test.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_rrtmgp_L10.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_gray_fullphys.sbatch
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_rrtmgp_validate.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_coupled.sbatch
scripts/cluster/omip_nemo/run_test_prescribed_flow.sbatch
scripts/cluster/omip_nemo/run_trp2_ship1.sbatch
scripts/cluster/omip_nemo/run_ref5yr.sbatch
scripts/cluster/omip_nemo/run_trp4_vfsfix_probe.sbatch
scripts/cluster/omip_nemo/run_ll_ship2.sbatch
scripts/cluster/omip_nemo/validate_changes.sbatch
scripts/cluster/omip_nemo/rerun_mpas7_kpp_r2.sbatch
scripts/cluster/omip_nemo/run_faithful_ll2_base_ab.sbatch
scripts/cluster/omip_nemo/diag_static2.sbatch
scripts/cluster/omip_nemo/run_mpas9_parity.sbatch
scripts/cluster/omip_nemo/run_core2_woasmoke.sbatch
scripts/cluster/omip_nemo/run_core2_woasmoke_dt150.sbatch
scripts/cluster/omip_nemo/run_faithful_trp2_p46.sbatch
scripts/cluster/omip_nemo/build_nemo.sbatch
scripts/cluster/omip_nemo/run_dino_r1_ablate.sbatch
scripts/cluster/omip_nemo/run_smoke.sbatch
scripts/cluster/omip_nemo/probe_compute.sbatch
scripts/cluster/omip_nemo/diag_woa_static.sbatch
scripts/cluster/omip_nemo/dl_mesh.sbatch
scripts/cluster/omip_nemo/run_scm_twins.sbatch
scripts/cluster/omip_nemo/run_faithful_p45_smoke.sbatch
scripts/cluster/omip_nemo/run_trp2_ship1_burst.sbatch
scripts/cluster/omip_nemo/run_trp5_eice90.sbatch
scripts/cluster/omip_nemo/test_compare.sbatch
scripts/cluster/omip_nemo/_ab_gm_treguier.sbatch
scripts/cluster/omip_nemo/run_dino_r1_zco.sbatch
scripts/cluster/omip_nemo/run_trp4a_eice_probe.sbatch
scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch
scripts/cluster/omip_nemo/run_ll8_kppeice.sbatch
scripts/cluster/omip_nemo/run_dino_r1_tke.sbatch
scripts/cluster/omip_nemo/stage_woa.sbatch
scripts/cluster/omip_nemo/run_trp2_iceinit.sbatch
scripts/cluster/omip_nemo/run_dino_r1_exact.sbatch
scripts/cluster/omip_nemo/run_faithful_ll2_p45.sbatch
scripts/cluster/omip_nemo/run_ico7_dm30.sbatch
scripts/cluster/omip_nemo/_ab_tke_surface_bc.sbatch
scripts/cluster/omip_nemo/run_trp2_tke_iceinit.sbatch
scripts/cluster/omip_nemo/rerun_ll3_ri015.sbatch
scripts/cluster/omip_nemo/_ab_tke_mxl3.sbatch
scripts/cluster/omip_nemo/_diag_kappa_gm.sbatch
scripts/cluster/omip_nemo/run_multiprocess_cpu_equiv.sbatch
scripts/cluster/omip_nemo/run_eorca025_4gpu_multinode.sbatch
scripts/cluster/omip_nemo/run_ll_ship1.sbatch
scripts/cluster/omip_nemo/run_core2_tripole.sbatch
scripts/cluster/omip_nemo/run_core2_nudgesmoke.sbatch
scripts/cluster/omip_nemo/run_ref10yr.sbatch
scripts/cluster/omip_nemo/compare_day90.sbatch
scripts/cluster/omip_nemo/run_mpas14_tke.sbatch
scripts/cluster/omip_nemo/rerun_dino_l2_r2.sbatch

exec
/usr/bin/bash -lc "rg --files -uu | rg '("'^|/)AGENTS'"\\.md"'$|'"\\.physics-validator/scaling_campaign|26628073|26628196|recovery_review'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 273ms:
cpu_ll2d.26628073.log
.physics-validator/scaling_campaign/review10_batch.md
.physics-validator/scaling_campaign/codex_scaleout_improvements_2026-07-31.md
.physics-validator/scaling_campaign/review8_final.md
.physics-validator/scaling_campaign/review12_gated_barrier.md
.physics-validator/scaling_campaign/review2.md
.physics-validator/scaling_campaign/review6_final.md
.physics-validator/scaling_campaign/review16_cube_halo.md
.physics-validator/scaling_campaign/review14_scaleout.md
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md
.physics-validator/scaling_campaign/review11_np4_strategy.md
.physics-validator/scaling_campaign/iter5_ocean.patch
.physics-validator/scaling_campaign/review19_meshcap.md
.physics-validator/scaling_campaign/review5.md
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md
.physics-validator/scaling_campaign/iter2_diff.patch
.physics-validator/scaling_campaign/review15_plateaus.md
.physics-validator/scaling_campaign/review9_gate_reframe.md
.physics-validator/scaling_campaign/review1.md
.physics-validator/scaling_campaign/review17_globalmem.md
.physics-validator/scaling_campaign/review3.md
.physics-validator/scaling_campaign/evidence_for_strategy.md
.physics-validator/scaling_campaign/review18_memfix.md
.physics-validator/scaling_campaign/review13_precision.md
.physics-validator/scaling_campaign/iter1_diff.patch
.physics-validator/scaling_campaign/review7_mechanism.md
.physics-validator/scaling_campaign/iter3_diff.patch
.physics-validator/scaling_campaign/iter4_code.patch

exec
/usr/bin/bash -lc "git status --short && sed -n '1,260p' .physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md && sed -n '1,320p' .physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign

exec
/usr/bin/bash -lc "rg -n -i -C 3 'VERDICT|FIX-FIRST|"'^##|''^#|''^[0-9]+'"\\.' .physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md | tail -n 180
printf '\\n--- R3 current content ---\\n'
tail -n 140 .physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md
printf '\\n--- targeted document anchors ---\\n'
rg -n -i -C 3 'metis|cells_per_rank_achieved|nominal mean|LL2048|65,?536|floor|26628196|INTERPRETATION LIMIT|26628073|cpu_ll2d|2-D|cache transition|masquerade|7151d12a1|non-reproduction|reproduction-grade' docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 374ms:
1013-13004:12. Ensemble arithmetic needs tighter wording: if solo reproduces 5.47 GC/s, ideal aggregate is 21.88 GC/s. The 1.10 threshold guarantees only 19.89 GC/s = 3.636× the 32-GPU solo rate, or 3.35× the observed 128-GPU single trajectory—not 21.9 GC/s.
1014-13005-
1015:13006:VERDICT: FIX-FIRST
1016-
1017-exec
1018-/usr/bin/bash -lc "rg -n -i -C 5 'Phase-3|recovered|s9|METIS|weak-pair|weak pair|2621568|2621568|LL2048|hundreds|2304|2880|512|floor|UCX|confirm|3\\.64|3\\.636|solo|ensemble|peak' docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
--
1534-1331-  path does not exist by design) and my CUDA_VISIBLE_DEVICES invocation
1535-1332-  never created ranks. The flat curves were the SAME single-device run
1536-1333:  repeated, not a latency floor and not a defect — codex round-13's
1537:1334-  "hypothesis, not verdict" was righter than it knew. What survives:
1538-1335-  single-device timings (s6 ~7 ms f32/f64, s7 ~21.9 ms f32).
1539-1336-
1540-1337-  **THE REAL LADDER (job 26494036, CPU-MPI f64, np1-16, block:cyclic,
--
1574-1589-Both jobs the dropped session left behind COMPLETED; neither had been
1575-1590-analysed. First read-out below, CORRECTED per codex round-20
1576-1591-(`.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md`,
1577:1592-VERDICT FIX-FIRST, 12 items — the round that caught a Lloyd-mesh confound
1578-1593-in the first draft's weak-scaling claim).
1579-1594-
1580-1595:### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094)
--
1692-exec
1693-/usr/bin/bash -lc "sed -n '1587,1735p' docs/performance/scaling/levante_campaign_2026-07-24.md && printf '\\n--- ensemble ---\\n' && nl -ba scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch && printf '\\n--- s8 ladder ---\\n' && nl -ba scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch && printf '\\n--- latlon hundreds ---\\n' && nl -ba scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch && printf '\\n--- latlon2d hundreds ---\\n' && nl -ba scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch && printf '\\n--- prewarm ---\\n' && nl -ba scripts/cluster/scaling_levante/prewarm_s10.sbatch && printf '\\n--- plot ---\\n' && nl -ba scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
1694- succeeded in 374ms:
1695:## Phase-3 receipts recovered after the 2026-07-31 session drop (2026-08-02)
1696-
1697-Both jobs the dropped session left behind COMPLETED; neither had been
1698-analysed. First read-out below, CORRECTED per codex round-20
1699-(`.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md`,
1700:VERDICT FIX-FIRST, 12 items — the round that caught a Lloyd-mesh confound
1701-in the first draft's weak-scaling claim).
1702-
1703:### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094)
1704-
1705-Matrix at a matched OWNED-cell target (`cells_per_rank_achieved` = 5,120
1706-exactly in every arm; method pinned per arm, never `auto`, which flipped
--
1740-  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
1741-  timing, but a "production config" claim would need a gated arm).
1742-
1743:### 2. subdiv-9 payoff ladder (atm MPAS ico GPU, job 26600095)
1744-
1745-f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
1746-scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
--
1770-  ~1.4x". A matched s8 lloyd=0 np8/16/32 rerun is submitted (see below);
1771-  no weak-scaling direction is claimed until it lands.
1772-
1773:### Next receipts submitted 2026-08-02
1774-
1775:1. **Ensemble receipt** (`scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch`)
1776-   — codex lever #1: 4 concurrent 32-GPU s9 replicas on disjoint 8-node
1777-   sets vs SAME-JOB solo controls bracketing phase B (solo before AND
1778-   after, ordering counterbalanced). steps=5000 so the stepping window
--
1782-   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
1783-   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
1784-   >10 % = the fabric-contention term, quantified per replica.
1785:2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
1786-   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
1787-   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
1788-   recomputed only from these.
1789-
1790:### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)
1791-
1792-LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
1793-@64 row (job 26502539, f32 6.73 ms):
--
1798-| f64 @128 | 9.60 | 22.72 |
1799-
1800-f32 strong 64->128: 1.21x for 2x devices (eff 0.60) with the tile at
1801:32.7k cols/GPU — at/near the floor, consistent with the tile law.
1802:39.1 GC/s is the highest measured throughput of ANY lane in the
1803-campaign. The companion oc128 (26534067) FAILED pre-#1370-fix with the
1804:109.5 GB resident-args signature; retry submitted post-fix (below).
1805-
1806:## Hundreds-of-devices push (user directive 2026-08-02)
1807-
1808-"Push the scaling to hundreds of CPUs and GPUs for lat-lon and MPAS on
1809-GPUs." Machine ceiling: 56 nodes x 4 = 224 a100_80 GPUs; compute
--
2477-+Both jobs the dropped session left behind COMPLETED; neither had been
2478-+analysed. First read-out below, CORRECTED per codex round-20
2479-+(`.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md`,
2480:+VERDICT FIX-FIRST, 12 items — the round that caught a Lloyd-mesh confound
2481-+in the first draft's weak-scaling claim).
2482-+
2483-+### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094)
--
4409-   371	        segment_mode=(seg_n > 0),
4410-   372	        segment_steps=(seg_n if seg_n > 0 else None),
4411-   373	        # Measurement validity (codex batch4): finite_ok is the ACCUMULATED
4412:   374	        # in-graph finite verdict (null in the unchecked default fused lane);
4413-   375	        # valid=false marks the row as NOT scaling data; completed_blocks
4414-   376	        # says where a diverging segment run stopped.
4415-   377	        finite_ok=finite_ok,
--
5325-    )
5326-
5327-
5328:# ===========================================================================
5329:# CLI
5330:# ===========================================================================
5331-
5332-def build_parser() -> argparse.ArgumentParser:
5333-    p = argparse.ArgumentParser(
--
6308-17:1. METIS A/B (job 26600094, all arms 5120 cells/rank, f64, 32 ranks/node): A s7np32 geometric 189.82ms, B s8np128 geometric 308.96, C s7np32 metis 191.99, D s8np128 metis 333.39, E s8np128 metis block:block 537.91. My claims: rank-count term geometric 1.628 vs metis 1.736 -> partition-quality lever DEAD/refuted; block:cyclic mandatory (E/D=1.61); lever#2 closed refuted on this lane.
6309-18-2. s9 GPU ladder (job 26600095, f32 sfc lloyd0, 2621442 cells L26): np32 12.47ms/5.47GC/s, np64 9.60/7.10, np128 11.48/5.94. Claims: new MPAS peak 7.10 GC/s = 2.2x s8 best 3.23 GC/s (s8 655362 cells, np64 5.27ms, 26 levels -> check GC/s arithmetic); 64->128 anti-scales at 20.5k cells/GPU consistent with ~30k floor; weak pairs s8->s9 at matched cells/GPU: 6.92->12.47 (0.55), 7.10->9.60 (0.74), 8.13->11.48 (0.71) -> ~1.4x matched-tile cost per 4x ranks = GPU rank-count term.
6310-19-3. Ensemble job design (mpas_s9_ensemble.sbatch): 4 concurrent 32-GPU srun steps in one 32-node allocation, per-step SLURM_STEP_NODELIST coordinator autodetect, shared jobid-derived port claimed safe because hosts differ; solo control in-job; CONFIRM bar max(replica)<=1.10x solo. Attack: step isolation, GPU binding, srun flag errors, cache effects, whether solo-then-concurrent ordering biases, whether 12 steps is enough discrimination.
6311:20-Check every ratio and GC/s conversion. Flag any claim that should be labelled PLAUSIBLE rather than CONFIRMED, any protocol mismatch making a comparison confounded, and any sbatch bug. End with 'VERDICT: SHIP' or 'VERDICT: FIX-FIRST' plus a numbered issue list.
6312-21-codex
6313---
6314-228-## Oracle-Recipe Fidelity (ocean) — see docs/ocean/fidelity/oracle_recipe_strategy.md
--
6350-509:   tracks wet cells (i.e. compacted), and the expensive-half measurement
6351-510:   below shows compaction itself does not pay at real wet fractions.
6352-511-
6353:512-   VERDICT on the audit's item 4 as a whole: BOTH halves measured, BOTH
6354-513-   negative on this codebase — the "~2x on ~40%-land grids" projection is
6355-514-   refuted twice over (gather penalty eats the compaction saving at 0.71
6356-515:   wet; wet-balanced bands worsen dense-compute balance). The item is
--
6371-530:   interface, sees real wet topology (not banded), richer stencils, and
6372-531-   communication, none of which the microbench prices. What survives
6373-532-   regardless: the audit's 2x assumed zero indirection cost and is
6374:533:   refuted; at ~0.71 wet the measured stencil member is a net LOSS. VERDICT: do not build compaction for the global latlon ocean;
6375-534:   revisit only for a configuration that is genuinely <~55% wet.
6376-535-5. **2-D lat-lon decomposition** at >=64 ranks: the 1-D band's perimeter
6377-536-   ceiling is now measured (0.12-0.16 at np64 spread, vs ico's 0.52), which
--
6718-8555-docs/performance/scaling/levante_campaign_2026-07-24.md-1344-  lane (bigger mesh holds efficiency deeper), and the np8->16 flattening
6719-8556-docs/performance/scaling/levante_campaign_2026-07-24.md-1345-  sits exactly where per-rank cells fall to 2.5k (s6) vs 10k (s7).
6720-8557---
6721:8558:docs/performance/scaling/levante_campaign_2026-07-24.md-1548-   refuted; at ~0.71 wet the measured stencil member is a net LOSS. VERDICT: do not build compaction for the global latlon ocean;
6722-8559:docs/performance/scaling/levante_campaign_2026-07-24.md-1549-   revisit only for a configuration that is genuinely <~55% wet.
6723-8560-docs/performance/scaling/levante_campaign_2026-07-24.md-1550-5. **2-D lat-lon decomposition** at >=64 ranks: the 1-D band's perimeter
6724-8561-docs/performance/scaling/levante_campaign_2026-07-24.md-1551-   ceiling is now measured (0.12-0.16 at np64 spread, vs ico's 0.52), which
--
8268-   446	    _INITIALIZED = True
8269-
8270-codex
8271:VERDICT: FIX-FIRST
8272-
8273-Prior 12 disposition: 1 open; 2–3 fixed; 4 explicitly deferred; 5–6 fixed; 7 partially deferred (SHA still absent); 8 fixed; 9 fixed for exit-status/readability; 10 partial; 11 open; 12 fixed.
8274-
8275:1. The METIS section still falsely equates `cells_per_rank_achieved=5120` with equal owned-cell load. That field is integer floor division, not partition balance ([source](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_ocean_mpas_scaling.py:751)). The recovered receipt’s actual owned ranges remain geometric 5120–5121; METIS 5100–5145 @32 and 5093–5144 @128. Revise [the claim](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/docs/performance/scaling/levante_campaign_2026-07-24.md:1597) to “nominal mean target,” alongside both owned and wet ranges.
8276-
8277:2. LL2048@128 has 65,536 columns/GPU, not 32.7k: `2048×4096/128`. Thus its 0.603 efficiency is correct, but attributing it to being at/near the 30k floor is not ([campaign](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/docs/performance/scaling/levante_campaign_2026-07-24.md:1692)). The displayed 5.58 ms converts to 39.09 GC/s; 39.11 is only consistent if the underlying time was more precise than the table.
8278-
8279:3. The LL2304 planned-arm tile counts are both 2× too small: @96 is 110.6k and @192 is 55.3k columns/GPU, not 55.3k and 27.6k. LL2880@192 = 86.4k is correct. Therefore the supposed LL2304@192 floor control is not below-floor ([script](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:12)).
8280-
8281:4. The ensemble is improved, but it does not counterbalance ordering: it always runs `solo_pre → all replicas → solo_post` ([script](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:70)). Nor does it enforce overlap: `sacct` is best-effort (`|| true`), and no per-step node list/coordinator/port is actually printed. It can screen a 10% co-execution penalty, but cannot confirm that any penalty is specifically fabric contention rather than placement/topology/drift.
8282-
8283:5. “Receipt validity” presently means JSON parsing plus presence of `steady_median_ms`; it accepts `NaN` and does not validate finite state, expected topology, or metadata. This matters for the 5,000-step run: the MPAS bench records timings but has no finite-state gate ([bench](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_mpas_spmd_scaling.py:402)). The two hundreds scripts also print missing/unparseable result files without converting that into nonzero `rc` ([GPU](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:47), [CPU](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:36)).
8284-
8285:6. The np32 s9 SHA is visibly deferred, not recovered. Keep the result explicitly non-reproduction-grade until the job log or executed revision is recovered ([campaign](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/docs/performance/scaling/levante_campaign_2026-07-24.md:1637)).
8286-
8287-The other arithmetic checks pass: METIS ratios, s9 GC/s and speedups, 3.64× ensemble threshold, and s10 tile counts. Slurm sizing is valid: the ensemble’s four 8-node full-GPU `--exact` steps fit disjointly; 2304 divides 96/192, 2880 divides 192; and r512@512 factors as 16×32 ranks, giving 32×32 columns/rank. Shell syntax and diff whitespace are clean.
8288-tokens used
8289-187,126
8290:VERDICT: FIX-FIRST
8291-
8292-Prior 12 disposition: 1 open; 2–3 fixed; 4 explicitly deferred; 5–6 fixed; 7 partially deferred (SHA still absent); 8 fixed; 9 fixed for exit-status/readability; 10 partial; 11 open; 12 fixed.
8293-
8294:1. The METIS section still falsely equates `cells_per_rank_achieved=5120` with equal owned-cell load. That field is integer floor division, not partition balance ([source](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_ocean_mpas_scaling.py:751)). The recovered receipt’s actual owned ranges remain geometric 5120–5121; METIS 5100–5145 @32 and 5093–5144 @128. Revise [the claim](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/docs/performance/scaling/levante_campaign_2026-07-24.md:1597) to “nominal mean target,” alongside both owned and wet ranges.
8295-
8296:2. LL2048@128 has 65,536 columns/GPU, not 32.7k: `2048×4096/128`. Thus its 0.603 efficiency is correct, but attributing it to being at/near the 30k floor is not ([campaign](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/docs/performance/scaling/levante_campaign_2026-07-24.md:1692)). The displayed 5.58 ms converts to 39.09 GC/s; 39.11 is only consistent if the underlying time was more precise than the table.
8297-
8298:3. The LL2304 planned-arm tile counts are both 2× too small: @96 is 110.6k and @192 is 55.3k columns/GPU, not 55.3k and 27.6k. LL2880@192 = 86.4k is correct. Therefore the supposed LL2304@192 floor control is not below-floor ([script](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:12)).
8299-
8300:4. The ensemble is improved, but it does not counterbalance ordering: it always runs `solo_pre → all replicas → solo_post` ([script](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:70)). Nor does it enforce overlap: `sacct` is best-effort (`|| true`), and no per-step node list/coordinator/port is actually printed. It can screen a 10% co-execution penalty, but cannot confirm that any penalty is specifically fabric contention rather than placement/topology/drift.
8301-
8302:5. “Receipt validity” presently means JSON parsing plus presence of `steady_median_ms`; it accepts `NaN` and does not validate finite state, expected topology, or metadata. This matters for the 5,000-step run: the MPAS bench records timings but has no finite-state gate ([bench](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/bench_mpas_spmd_scaling.py:402)). The two hundreds scripts also print missing/unparseable result files without converting that into nonzero `rc` ([GPU](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:47), [CPU](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:36)).
8303-
8304:6. The np32 s9 SHA is visibly deferred, not recovered. Keep the result explicitly non-reproduction-grade until the job log or executed revision is recovered ([campaign](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/docs/performance/scaling/levante_campaign_2026-07-24.md:1637)).
8305-
8306-The other arithmetic checks pass: METIS ratios, s9 GC/s and speedups, 3.64× ensemble threshold, and s10 tile counts. Slurm sizing is valid: the ensemble’s four 8-node full-GPU `--exact` steps fit disjointly; 2304 divides 96/192, 2880 divides 192; and r512@512 factors as 16×32 ranks, giving 32×32 columns/rank. Shell syntax and diff whitespace are clean.

--- R3 current content ---
scripts/cluster/carbon_calibration/train_carbon_params_sif.sbatch
scripts/cluster/carbon_calibration/train_carbon_params.sbatch
scripts/cluster/carbon_calibration/retest_fast.sbatch
scripts/cluster/carbon_calibration/validate_fast_analytic.sbatch
scripts/cluster/carbon_calibration/test_chunked_grad.sbatch
scripts/cluster/carbon_calibration/bisect_chunk.sbatch
scripts/cluster/carbon_calibration/train_carbon_exact_chunked.sbatch
scripts/cluster/carbon_calibration/validate_carbon_trainer.sbatch
scripts/cluster/carbon_calibration/build_sif_observations.sbatch
scripts/cluster/carbon_calibration/train_carbon_params_fast.sbatch
scripts/cluster/compare_reanalysis/preflight_ck_sensitivity.sbatch
scripts/cluster/compare_reanalysis/run_correction_campaign.sbatch
scripts/cluster/compare_reanalysis/preflight_osse.sbatch
scripts/cluster/wb_forecast/campaign_smoke.sbatch
scripts/cluster/wb_forecast/smoke_stage1.sbatch
scripts/cluster/wb_forecast/classical_sweep.sbatch
scripts/cluster/wb_forecast/train_sfno_full_scale.sbatch
scripts/cluster/wb_forecast/wb2_eval.sbatch
scripts/cluster/wb_forecast/train_aimip_neural.sbatch
scripts/cluster/wb_forecast/run_pytest.sbatch
scripts/cluster/wb_forecast/train_sfno_full_scale_ginsburg.sbatch
scripts/cluster/wb_forecast/check_config.sbatch
scripts/cluster/wb_forecast/run_dp_test.sbatch
tests/plot/test_plot_strong_scaling_by_resolution.py
tests/plot/test_cpu_gpu_scaling_summary.py
tests/plot/test_scaling_efficiency_plot.py
tests/plot/test_plot_bcw_scaling.py
tests/plot/test_plot_scaling_family.py
tests/plot/test_plot_cpu_vs_gpu_scaling.py
tests/plot/test_plot_levante_gpu_scaling_comparison.py
tests/ocean/fidelity/fixtures/tier8_global_realistic.json
tests/ocean/fidelity/fixtures/tier5_baroclinic_instability.json
tests/ocean/fidelity/fixtures/tier0_invariants.json
tests/ocean/fidelity/fixtures/tier6_channel_circulation.json
tests/ocean/fidelity/fixtures/tier1_linear_waves.json
tests/ocean/fidelity/fixtures/tier7_dino.json
tests/ocean/fidelity/fixtures/tier4_wind_driven_gyres.json
tests/ocean/fidelity/fixtures/tier3_process_benchmarks.json
tests/ocean/fidelity/fixtures/tier2_geostrophic_thermalwind.json
scripts/cluster/levante/amip_mpas_gpu_chain.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_rrtmgp_L10_cache.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_rrtmgp.sbatch
scripts/cluster/cmip6_coupled/run_unfused_test.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_rrtmgp_L10.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_gray_fullphys.sbatch
scripts/cluster/cmip6_coupled/coupling_fix_gates.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_rrtmgp_validate.sbatch
scripts/cluster/cmip6_coupled/run_coarse5deg_coupled.sbatch
scripts/cluster/omip_nemo/run_test_prescribed_flow.sbatch
scripts/cluster/omip_nemo/run_trp2_ship1.sbatch
scripts/cluster/omip_nemo/run_ref5yr.sbatch
scripts/cluster/omip_nemo/run_trp4_vfsfix_probe.sbatch
scripts/cluster/omip_nemo/run_ll_ship2.sbatch
scripts/cluster/omip_nemo/validate_changes.sbatch
scripts/cluster/omip_nemo/rerun_mpas7_kpp_r2.sbatch
scripts/cluster/omip_nemo/run_faithful_ll2_base_ab.sbatch
scripts/cluster/omip_nemo/diag_static2.sbatch
scripts/cluster/omip_nemo/run_mpas9_parity.sbatch
scripts/cluster/omip_nemo/run_core2_woasmoke.sbatch
scripts/cluster/omip_nemo/run_core2_woasmoke_dt150.sbatch
scripts/cluster/omip_nemo/run_faithful_trp2_p46.sbatch
scripts/cluster/omip_nemo/build_nemo.sbatch
scripts/cluster/omip_nemo/run_dino_r1_ablate.sbatch
scripts/cluster/omip_nemo/run_smoke.sbatch
scripts/cluster/omip_nemo/probe_compute.sbatch
scripts/cluster/omip_nemo/diag_woa_static.sbatch
scripts/cluster/omip_nemo/dl_mesh.sbatch
scripts/cluster/omip_nemo/run_scm_twins.sbatch
scripts/cluster/omip_nemo/run_faithful_p45_smoke.sbatch
scripts/cluster/omip_nemo/run_trp2_ship1_burst.sbatch
scripts/cluster/omip_nemo/run_trp5_eice90.sbatch
scripts/cluster/omip_nemo/test_compare.sbatch
scripts/cluster/omip_nemo/_ab_gm_treguier.sbatch
scripts/cluster/omip_nemo/run_dino_r1_zco.sbatch
scripts/cluster/omip_nemo/run_trp4a_eice_probe.sbatch
scripts/cluster/omip_nemo/run_dino_p45_drag.sbatch
scripts/cluster/omip_nemo/run_ll8_kppeice.sbatch
scripts/cluster/omip_nemo/run_dino_r1_tke.sbatch
scripts/cluster/omip_nemo/stage_woa.sbatch
scripts/cluster/omip_nemo/run_trp2_iceinit.sbatch
scripts/cluster/omip_nemo/run_dino_r1_exact.sbatch
scripts/cluster/omip_nemo/run_faithful_ll2_p45.sbatch
scripts/cluster/omip_nemo/run_ico7_dm30.sbatch
scripts/cluster/omip_nemo/_ab_tke_surface_bc.sbatch
scripts/cluster/omip_nemo/run_trp2_tke_iceinit.sbatch
scripts/cluster/omip_nemo/rerun_ll3_ri015.sbatch
scripts/cluster/omip_nemo/_ab_tke_mxl3.sbatch
scripts/cluster/omip_nemo/_diag_kappa_gm.sbatch
scripts/cluster/omip_nemo/run_multiprocess_cpu_equiv.sbatch
scripts/cluster/omip_nemo/run_eorca025_4gpu_multinode.sbatch
scripts/cluster/omip_nemo/run_ll_ship1.sbatch
scripts/cluster/omip_nemo/run_core2_tripole.sbatch
scripts/cluster/omip_nemo/run_core2_nudgesmoke.sbatch
scripts/cluster/omip_nemo/run_ref10yr.sbatch
scripts/cluster/omip_nemo/compare_day90.sbatch
scripts/cluster/omip_nemo/run_mpas14_tke.sbatch
scripts/cluster/omip_nemo/rerun_dino_l2_r2.sbatch

exec
/usr/bin/bash -lc "rg --files -uu | rg '("'^|/)AGENTS'"\\.md"'$|'"\\.physics-validator/scaling_campaign|26628073|26628196|recovery_review'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 273ms:
cpu_ll2d.26628073.log
.physics-validator/scaling_campaign/review10_batch.md
.physics-validator/scaling_campaign/codex_scaleout_improvements_2026-07-31.md
.physics-validator/scaling_campaign/review8_final.md
.physics-validator/scaling_campaign/review12_gated_barrier.md
.physics-validator/scaling_campaign/review2.md
.physics-validator/scaling_campaign/review6_final.md
.physics-validator/scaling_campaign/review16_cube_halo.md
.physics-validator/scaling_campaign/review14_scaleout.md
.physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md
.physics-validator/scaling_campaign/review11_np4_strategy.md
.physics-validator/scaling_campaign/iter5_ocean.patch
.physics-validator/scaling_campaign/review19_meshcap.md
.physics-validator/scaling_campaign/review5.md
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md
.physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md
.physics-validator/scaling_campaign/iter2_diff.patch
.physics-validator/scaling_campaign/review15_plateaus.md
.physics-validator/scaling_campaign/review9_gate_reframe.md
.physics-validator/scaling_campaign/review1.md
.physics-validator/scaling_campaign/review17_globalmem.md
.physics-validator/scaling_campaign/review3.md
.physics-validator/scaling_campaign/evidence_for_strategy.md
.physics-validator/scaling_campaign/review18_memfix.md
.physics-validator/scaling_campaign/review13_precision.md
.physics-validator/scaling_campaign/iter1_diff.patch
.physics-validator/scaling_campaign/review7_mechanism.md
.physics-validator/scaling_campaign/iter3_diff.patch
.physics-validator/scaling_campaign/iter4_code.patch

exec
/usr/bin/bash -lc "git status --short && sed -n '1,260p' .physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md && sed -n '1,320p' .physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign

exec
/usr/bin/bash -lc "rg -n -i -C 3 'VERDICT|FIX-FIRST|"'^##|''^#|''^[0-9]+'"\\.' .physics-validator/scaling_campaign/codex_recovery_review_r2_2026-08-02.md | tail -n 180
printf '\\n--- R3 current content ---\\n'
tail -n 140 .physics-validator/scaling_campaign/codex_recovery_review_r3_2026-08-02.md
printf '\\n--- targeted document anchors ---\\n'
rg -n -i -C 3 'metis|cells_per_rank_achieved|nominal mean|LL2048|65,?536|floor|26628196|INTERPRETATION LIMIT|26628073|cpu_ll2d|2-D|cache transition|masquerade|7151d12a1|non-reproduction|reproduction-grade' docs/performance/scaling/levante_campaign_2026-07-24.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign

--- targeted document anchors ---
22-| Atm MPAS ico L8 (28 km) L26 | 6→16 A100 | 8.66→7.08 ms/step; np16 = 2.41 GC/s — 1.6x the Derecho 16-A100 aggregate reported in `derecho_levante_sota_review_2026-07.md` SS3b (route-A, eff ~0.38 @16); CROSS-MACHINE, different stack/date - indicative, not a controlled A/B | 26453240/26449147 |
23-| **Atm cube C768/L60 (same-path cs-spmd)** | 6→24 A100 | 58.35→14.09 ms/step = **4.14× = eff 1.04 (at ideal)**, 15.1 GC/s (629 Mc/s/GPU) | 26453782 |
24-| Atm cube C384/L60 (same-path cs-spmd) | 6→24 A100 | 15.44→8.81 ms/step = 1.75× (eff 0.44), 6.0 GC/s | 26452894 |
25:| Atm cube C192/L60 (same-path) | 6→24 | 6.20→6.80 ms — ANTI-scales (eff 0.23): 9.2k cols/GPU is below the ~30k-column floor | 26452979 |
26-| Ocean lat-lon LL576×1152 L20, production implicit | 4→8→16 | 15.6→17.2→17.4 ms — anti-scales across nodes | 26452743-45 |
27-| Ocean same, improved (wide-halo + vmix-f32) | 4→8→16 | 12.8→11.1→8.6 ms — monotone, **2.01× at 16 GPUs** | 26452804-06, 26453279 |
28-| Cube tiled >6-GPU lane (its own bench) | 24 A100, C384/L60 closed loop | 9.01 ms/step, 5.9 GC/s, 18.2 SYPD — first >6-GPU production-lane receipts | 26450938/26452632 |
29-
30-Single-node GPU (job 26445836): cube C192 gray_sbm strong eff 0.84–1.07
31:(1→3 A100); C48/C96 latency-floored. Ocean single-node solver ladder: below.
32-
33-**THE cube strong-scaling result** — read 1.04 as "at ideal", NOT "better
34-than ideal": efficiency slightly above 1 is expected when the BASE leg is
--
95-26454476 + 26454618): 19.90 / 17.09 / 6.92 / 7.10 ms at np 2/4/8/16.
96-Taking np2 as the base (it has the BEST per-device throughput, 430
97-Mc/s/GPU): 2->8 = 2.88x = eff 0.72, 2->16 = 2.80x = eff 0.35 (small-tile
98:floor).
99-
100-OPEN ANOMALY, characterised not explained: per-GPU throughput dips at
101-np=4 (247 Mc/s/GPU vs 430 at np2 and 306 at np8), so np4 is barely faster
--
115-rises with count; a per-message-size effect is not excluded). (Fusion count 136/173/240,
116-  bitcasts 526/582/694 — the np8 program is finer-grained.)
117-- PARTITION METHOD REFUTED (job 26455829): the dip is method-independent —
118:  np4/np8 = 17.22/7.00 ms (sfc), 17.20/6.97 (metis), 19.35/6.26
119-  (geometric). Every method shows the same 2.5-3.1x jump.
120-- XLA CODEGEN ENV KNOBS REFUTED (job 26455948): np4 is 17.12 ms base,
121-  17.06 autotune-level-4, 17.10 latency-hiding-off, 17.01
--
208-device count (jobs 26456334 / 26457693 / 26456337 vs 26452804-06).
209-
210-So the ocean shows the SAME tile-size dependence the cube does: the 2.01x
211:multinode improvement measured at LL576 was partly a floor effect, and at
212-a production tile the identical code scales substantially better (0.37 ->
213-0.63). Per-device throughput also rises (259 -> 305 Mc/s/GPU at np4).
214-Config is byte-identical between the two rows; only the grid changes.
--
223-## Weak scaling at production per-device size (job 26453523)
224-
225-The earlier weak ladders used a 64-row base (0.17M cells/GPU — under the
226:latency floor). Re-run at PRODUCTION size (288 rows × 1152 lon × L20 =
227-6.6M cells/GPU, 1→4 A100, conservation gated): production implicit
228-1.00/0.72/0.70, improved wide-halo+vmix-f32 1.00/0.86/0.85.
229-PRECISION MATCHED (self-audit correction): BOTH ladders compared here are
--
231-the earlier ladder's f32 rows (eff 0.26/0.25 at nd 2/4), not its f64 rows (both from job 26445836).
232-An earlier revision of this file mislabelled the production-tile run f64
233-and cited the f64 small-base numbers; the direction and size of the effect
234:are unchanged, but the comparison is only valid precision-matched. So the earlier weak ladder measured a below-floor tile rather than a code
235-limit, and the same
236-config that fixes strong scaling also carries weak (+0.15 at nd4). Ideal is
237-flat; the improved arm holds 22.5→22.9 ms while production drifts
--
326-
327-Against the 4-node SPREAD lat-lon ladder (job 26452578) at high rank
328-counts the contrast is large: ico holds 0.52 at np64 where lat-lon r128/r256
329:is at 0.12/0.16. That matches the documented expectation that a 2-D cell
330-partition beats a 1-D latitude band on perimeter/area. CAVEAT: ico ran
331-PACKED on one node and lat-lon SPREAD over four, so this compares
332-decomposition AND placement together, not decomposition alone.
--
342-(17.82 us, 64.22 GB/s) and the per-tile single-device compute term — the
343-ocean LL576 f64 implicit ladder finally has a real roofline:
344-
345:| nd | measured | calibrated bound | measured/bound | at % of floor |
346-|---|---|---|---|---|
347-| 2 | 42.71 ms | 34.77 ms | 1.228 | 81 % |
348-| 4 | 23.53 ms | 16.82 ms | 1.398 | 72 % |
--
422-now measured to be null here.
423-
424-Honest answer to "how far from the theoretical limit are we": 72-81 % of a
425:now-calibrated floor, with the shortfall attributable to neither bandwidth
426-nor byte volume.
427-
428-## The mechanism's prediction, TESTED — and the lever it exposes (job 26459382)
--
669-   sees bit-identical inputs" is too strong; differences are XLA
670-   re-association at the serial tier and larger at the distributed tiers.
671-   ONE REAL BEHAVIOURAL DELTA to disclose: wide-halo requires LOCAL
672:   subcycle clamping, and with an active `eta_floor` the clamp/
673-   redistribution schedule differs from the standard path — a reviewer
674-   should check that config interaction, not filter stability in general.
675-
--
717-
718-Both meshes still scale at 2 600 cells/rank; subdiv-7 keeps gaining down
719-to 600 and only ANTI-SCALES at 300 (56 -> 67 ms). **The turnover tracks
720:work per rank, not rank count** — the tile-floor hypothesis, confirmed
721-on a lane where the two can be separated. Practical consequence: rank
722-counts beyond the campaign's old 64 ceiling keep paying as long as
723-resolution rises with them; subdiv-8 reaches 356 ms at 256 ranks, a
--
746-
747-The lat-lon band decomposition has no such ceiling; its matched pair
748-runs at 64 GPUs (job 26497323): LL720@16 and LL1440@64 both hold 64.8k
749:columns/GPU, with LL720@64 (16.2k) as the sub-floor control.
750-
751:**Cube tile-floor arm, measured (job 26497294):** C768 L60 from 24 to 54
752-GPUs = 19.33 -> 13.93 ms, **1.39x at 2.25x devices, efficiency 0.62** —
753-and the tile only falls to 65.5k cols/GPU, still well ABOVE the ~30k
754:floor. So unlike the CPU lane, the cube's loss here is NOT explained by
755:the tile floor alone; there is real device-count cost to quantify.
756-(Note the same-job C768@24 anchor reads 19.33 ms where the campaign's
757-figure carries 14.09 ms for C768@24 — different lane/protocol between
758-those jobs, so only the within-job 24-vs-54 contrast is used.)
--
786-FALL with tile size while collective latency could RISE with rank count;
787-the 2.25x fixed-tile contrast supports only "little growth over the
788-tested range", not universality. That single number explains
789:the cube's efficiency 0.62, the empirical tile floor, and the plateau in
790-the figure.
791-
792-**THE SAME STRUCTURE ON LAT-LON.** Two fixed-device (16 GPU) points —
--
809-| arm | devices | cells | ms/step |
810-|---|---|---|---|
811-| LL1024x2048 | 16 | 4.2 M | 5.73 |
812:| **LL2048x4096** | **64** | **218.1 M** | **6.73** |
813-
814-4x the devices carrying 4x the problem costs **+17 %** — weak-scaling
815-efficiency **0.85**, sustaining **32.4 GCells/s (506 Mcells/s/GPU) on
--
820-by the per-step fixed cost above.
821-
822-Scale-out receipts on this lane: LL1536x3072 at 64 GPUs = 4.97 ms (job
823:26498266) and the LL2048 point above.
824-
825-**WHAT THE FIXED TERM IS — ATTRIBUTED (nsys job 26504836): the halo
826-exchange, scaling with tile PERIMETER.** Profiling both tiles at the
--
964-102 GB/device, C1152 97-106 GB.
965-
966-**CONSEQUENCE, and it is the campaign's sharpest practical finding:**
967:raising resolution is the ONLY measured cure for the tile-floor plateau,
968-and it is currently unavailable on both transports — capped at subdiv-8
969-on CPU by the mesh generator, and by per-device global allocation on GPU.
970-So the useful rank/device ceilings measured here are NOT hardware limits:
--
1004-  cube-edge interpolation, and the tiled transport supports only halo
1005-  1/2. That is a new algorithm, not a port.
1006-
1007:**THE BOUND (one step, 86 SendRecv, latency floor 17.8 us measured):**
1008-
1009-| class | n | total |
1010-|---|---|---|
--
1077-## The MPAS mesh cap — lifted (subdiv-9 unblocked for 128 GPUs)
1078-
1079-The generator's hard subdiv-8 cap was the CPU-side resolution blocker
1080:and made 128-GPU MPAS floor-starved by construction (subdiv-8 at np128 =
1081-5.1k cells/GPU). Chain shipped 2026-07-30 (codex round-19 design,
1082-commit 70f3ce636):
1083-
--
1104-  np32 is WORSE than np16 while np64 is the minimum, and f64 shows the
1105-  same pattern (np32 14.26 vs np64 8.96). This is the np4-dip signature
1106-  at another count — count-specific codegen/fusion behaviour layered on
1107:  the tile floor (the fusion pathology on this lane is already proven
1108-  shape-dependent). Recorded as observed; not chased further at subdiv-8
1109:  since the mesh is below the floor at all these counts anyway.
1110-* Payoff ladder submitted (job 26549775): subdiv-9 at 32/64/128 GPUs =
1111-  81.9k/41.0k/20.5k cells/GPU — the first MPAS many-GPU ladder whose
1112:  lower rungs sit ABOVE the ~30k floor.
1113-
1114-OPERATIONAL NOTE: a Lustre incident mid-implementation left the module
1115-with an undefined constant on disk for ~12 h; two queued jobs (mpas32
--
1128-| LL2304x4608 L20 | 64 | 165.9k | **OOM — 102 GB/device** |
1129-
1130-The strong arm reaches 64 GPUs (19.83 -> 11.19 ms, 1.77x for 4x devices,
1131:eff 0.44 — the tile falls to 41.5k, near the ~30k floor, so this is the
1132:floor behaving exactly as the atmosphere's does).
1133-
1134-**The fixed-tile arm could not run, and WHY it could not is the finding.**
1135-LL2304 at 64 GPUs asked for **102.04 GB per device**. The per-device
--
1180-arrays that stay alive after sharding** — the globally-built initial
1181-state, the vertex-mask cache primed FROM the global state, and the
1182-replicated geometry stacks. The cube's level-independent wall fits: its
1183:setup residency is mesh tables + 2-D geometry.
1184-
1185-**FIX STAGE (i) SHIPPED AND MEASURED** (commits e1b502000/e5a541c65 +
1186-probe 26524423): building the global model/state under
--
1209-WHY THIS IS THE CAMPAIGN'S MOST IMPORTANT BLOCKER: the measured cure for
1210-every plateau is a LARGER TILE, i.e. raising resolution as devices are
1211-added. This defect makes that impossible — adding GPUs cannot buy
1212:resolution — so every GPU lane is pinned at the tile floor. Filed as
1213-**#1370** with the arithmetic; distinct from #1360 (the cube's kt
1214-validation ceiling), and validating kt=4 alone would NOT unblock C1152.
1215-
--
1259-  ranks but **1.97x** at 128. So the two effects compound, and the
1260-  default `auto` is doing well to land near geometric.
1261-* Practical: pin `--partition-method geometric` (or auto) on this lane;
1262:  sfc is actively harmful at scale. pymetis is absent from `.venv-mpi`,
1263:  so the low-cut METIS arm codex wanted is still unmeasured.
1264-
1265-**512-RANK LADDER (job 26508063):** s7 63.83 ms, s8 194.80 ms.
1266-s8 keeps gaining 256->512 (254.41 -> 194.80 = **1.31x**, at 1 280
1267-cells/rank) while s7 goes flat (65.71 -> 63.83 = 1.03x, at 320
1268:cells/rank — below the 300-600 floor). Same floor as the atmosphere,
1269-reached at a different rank count because the mesh differs. So this lane
1270-DOES scale to 512 ranks when the mesh is large enough; it just pays the
1271-rank-count term on the way.
--
1325-  26493648 retained above for the f64 points only.)
1326-- **MPAS-ocean Voronoi: RETRACTION — my "ladders" (subdiv 6 AND the
1327-  subdiv-7 discriminator, jobs 26493648/26493837) were INVALID.** The
1328:  bench's own metadata says it: `n_ranks: 1, cells_per_rank_achieved:
1329-  163842` — every arm ran ONE rank on the FULL mesh, because this bench
1330-  decomposes by MPI RANK (its docstring states the SPMD multi-device
1331-  path does not exist by design) and my CUDA_VISIBLE_DEVICES invocation
1332-  never created ranks. The flat curves were the SAME single-device run
1333:  repeated, not a latency floor and not a defect — codex round-13's
1334-  "hypothesis, not verdict" was righter than it knew. What survives:
1335-  single-device timings (s6 ~7 ms f32/f64, s7 ~21.9 ms f32).
1336-
--
1489-   has tolerance-parity coverage at all three transport tiers (serial
1490-   1e-12, MPI 1e-10, SPMD 2e-4/1e-3) and explicit_substep is an
1491-   established scheme (the model default; benches and OMIP explicitly
1492:   choose implicit_cn). Remaining: the eta_floor x local-clamp config
1493-   interaction, the OMIP case at production dt under forcing, and a
1494-   science sign-off on the f32 vmix solve. Worth 2.01x multinode.
1495-2. **MPAS ico np4 per-device dip** — five hypotheses refuted by
--
1547-   regardless: the audit's 2x assumed zero indirection cost and is
1548-   refuted; at ~0.71 wet the measured stencil member is a net LOSS. VERDICT: do not build compaction for the global latlon ocean;
1549-   revisit only for a configuration that is genuinely <~55% wet.
1550:5. **2-D lat-lon decomposition** at >=64 ranks: the 1-D band's perimeter
1551-   ceiling is now measured (0.12-0.16 at np64 spread, vs ico's 0.52), which
1552-   quantifies the prize.
1553-6. ~~Milan np16 anomaly~~ **RESOLVED 2026-07-26 (job 26479904): rank
--
1564-   FIX regardless of mechanism: `--distribution=block:cyclic
1565-   --cpu-bind=cores` on packed CPU lanes.
1566-
1567:7. **1-D bands vs 2-D pencils at np64 (job 26479904): the pencil path
1568-   is 1.37x faster** (latlon r256 moist f64, same dt: 253.93 -> 185.03
1569-   ms/step) — BUT this is NOT a pure decomposition A/B (codex round-10):
1570:   `--latlon-2d` selects the wall-pole 2-D path while the band path
1571-   keeps the atmospheric pole-fold, so boundary semantics change along
1572-   with the decomposition. Report as a regular-vs-wall-pole path
1573-   throughput result; attributing the 1.37x to decomposition alone would
--
1592-VERDICT FIX-FIRST, 12 items — the round that caught a Lloyd-mesh confound
1593-in the first draft's weak-scaling claim).
1594-
1595:### 1. METIS / placement A-B (ocean MPAS CPU, job 26600094)
1596-
1597:Matrix at a NOMINAL MEAN target of 5,120 cells/rank (the JSONL's
1598:`cells_per_rank_achieved` is global floor division —
1599-`int(mesh.nCells) // n_ranks`, bench_ocean_mpas_scaling.py:751 — NOT a
1600-balance statement; method pinned per arm, never `auto`, which flipped
1601:meaning when pymetis appeared in `.venv-mpi` on 2026-07-31), f64,
1602-nlev 20, 32 ranks/node. Actual per-rank OWNED ranges
1603-(`metadata.partition_metrics.cells_per_rank_min/max`): geometric
1604:5,120–5,121 at BOTH scales; metis 5,100–5,145 @32 and 5,093–5,144 @128
1605:(±0.5 %). WET load is looser still under metis — per-rank wet
1606:cell-levels min/max: geometric@32 100,740–102,420; metis@32
1607:96,800–102,720; metis@128 **78,000–102,880** (one rank 24 % under the
1608:mean) — METIS balances owned cells (approximately), not wet cells, on
1609-this bathymetry.
1610-
1611-| arm | config | ms/step |
1612-|---|---|---|
1613-| A | s7 np32 geometric, block:cyclic | 189.82 |
1614-| B | s8 np128 geometric, block:cyclic | 308.96 |
1615:| C | s7 np32 metis, block:cyclic | 191.99 |
1616:| D | s8 np128 metis, block:cyclic | 333.39 |
1617:| E | s8 np128 metis, block:block | 537.91 |
1618-
1619:* **This METIS configuration LOSES to geometric at s8/np128** (D/B =
1620-  +7.9 %), despite the better offline cut (partq s8@np128: edge_cut
1621-  1.89 % vs 2.01 %, halo mean 592 vs 629). Scale-out term (s7@32 ->
1622-  s8@128, which crosses 1 -> 4 nodes as well as 4x ranks — NOT a pure
1623:  rank-count isolate): geometric 1.628, metis 1.736. The offline-quality
1624:  -> step-time inference FAILS on this lane; part of metis's loss is
1625-  PLAUSIBLY its own wet-load imbalance (above). Scope: closes the
1626:  "swap in METIS as-is" lever on the CPU-MPI ocean lane; does NOT rule
1627-  out partition/mapping improvements generally (e.g. wet-cell-weighted
1628:  METIS was NOT tested).
1629-* **`block:cyclic` stays mandatory on packed CPU lanes** (E/D = 1.61x at
1630-  a byte-identical partition). NOTE the second `--distribution` field is
1631-  the INTRA-NODE (socket) distribution — both arms place ranks on nodes
--
1643-f32, sfc partition (padded-128 reorder), lloyd=0 LABELLED SYNTHETIC
1644-scaling mesh, executed padded n_cells = 2,621,568 (natural 2,621,442),
1645-L26; steps 12 / warmup 3; physics=none dynamics-only bench. Provenance:
1646:np64 and np128 rows record `git_sha: 7151d12a1`; the np32 row's field
1647-reads `unknown` — same allocation, same submitted script, so the same
1648:binary is PLAUSIBLE but that row stays non-reproduction-grade on its
1649-own (codex r20/r21):
1650-
1651-| GPUs | cells/GPU | ms/step | GC/s (cell-levels) |
--
1658-  synthetic dynamics-only bench** (2.19x the subdiv-8 best, 3.23 GC/s at
1659-  its np64: 655,362 natural cells x 26 lev / 5.27 ms).
1660-* Strong 32->64 speedup 1.299 (eff 0.65); 64->128 speedup 0.836 —
1661:  ANTI-scales at 20.5k cells/GPU. CONSISTENT WITH the ~30k floor seen on
1662-  the other lanes (single unreplicated point on a lane with known
1663-  count-specific codegen variation — not by itself proof).
1664-* **RETRACTED (codex round-20): the first draft's s8->s9 "weak
--
1690-
1691-### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)
1692-
1693:LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
1694-@64 row (job 26502539, f32 6.73 ms):
1695-
1696-| arm | ms/step | GC/s (col-levels) |
1697-|---|---|---|
1698:| f32 @128 (65,536 cols/GPU) | 5.5767 | **39.11** |
1699-| f64 @128 | 9.6015 | 22.72 |
1700-
1701-f32 strong 64->128: 1.207x for 2x devices (eff 0.60) with the tile at
1702:**65.5k cols/GPU — comfortably ABOVE the ~30k floor** (codex round-21
1703-caught the first draft halving this), so the loss is NOT
1704:floor-attributable. Mechanism OPEN — candidates (uninstrumented): 1-D
1705-band thinning to 16 rows/rank raising halo/compute ratio, and the
1706-16 -> 32-node NCCL topology step. 39.11 GC/s (from 5.5767 ms) is the
1707-highest measured throughput of ANY lane in the campaign. The companion
--
1718-|---|---|---|---|
1719-| 26628021 | s9 ensemble contention | 128 GPU (4x32) | lever #1 receipt |
1720-| 26628071 | oc LL2304 retry post-#1370 | 128 GPU | pre-fix failure was resident-args; predicted PASS at ~0.10 GB/dev residency |
1721:| 26628072 | atm LL2304 @96/@192 + LL2880 @192 | 96-192 GPU | LL2048 does not divide 192; LL2880@192 = 86.4k cols/GPU ABOVE floor |
1722:| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
1723:| 26628074 | subdiv-10 lloyd0 prewarm | 1 CPU | unlocks MPAS 128-224 GPUs ABOVE floor (81.9k-46.8k cells/GPU) |
1724-| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
1725-
1726-s10 ladder (128/192/224 GPUs) submits once 26628074's cache lands.
1727-
1728:### First hundreds receipt in: lat-lon CPU 2-D pencil to 512 ranks (job 26628073)
1729-
1730-r512 (512x1024 = 524k cols) L26 f64 moist, 32 rpn block:cyclic,
1731:wall-pole 2-D pencil lane (labelled; NOT the pole fold):
1732-
1733-| ranks | cols/rank | ms/step | speedup vs np64 | eff |
1734-|---|---|---|---|---|
--
1737-| 256 | 2,048 | 72.06 | 4.129 | 1.03 |
1738-| 512 | 1,024 | 44.78 | 6.645 | **0.83** |
1739-
1740:Distribution verified against the masquerade trap: result rows carry
1741-`n_ranks: 512` (the JSON's `metadata.process_count: 1` is the jax-LOCAL
1742-count on this mpi4jax lane, not the world size). 128->256 is
1743-SUPERLINEAR (2.23x for 2x) — classic per-rank working-set cache

exec
/usr/bin/bash -lc "nl -ba docs/performance/scaling/levante_campaign_2026-07-24.md | sed -n '1590,1775p'
printf '\\n--- ensemble ---\\n'
nl -ba scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch
printf '\\n--- atm hundreds ---\\n'
nl -ba scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch
printf '\\n--- cpu 2d hundreds ---\\n'
nl -ba scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch
printf '\\n--- cpu receipt ---\\n'
sed -n '1,260p' cpu_ll2d.26628073.log
printf '\\n--- plot head/panels ---\\n'
nl -ba scripts/plot/plot_scaling_paper_figure.py | sed -n '1,220p'
printf '\\n--- benchmark's cells-per-rank calculation ---\\n'
nl -ba scripts/bench/bench_ocean_mpas_scaling.py | sed -n '728,766p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 exited 1 in 308ms:
  1590	analysed. First read-out below, CORRECTED per codex round-20
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
  1679	   after, ordering counterbalanced). steps=5000 so the stepping window
  1680	   (~60 s) dwarfs launch skew; per-arm `SLURM_STEP_NODELIST` +
  1681	   wall-clock brackets logged as overlap evidence. CONFIRM bar:
  1682	   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
  1683	   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
  1684	   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
  1685	   >10 % = the fabric-contention term, quantified per replica.
  1686	2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
  1687	   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
  1688	   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
  1689	   recomputed only from these.
  1690	
  1691	### 3. Recovered phase-2 receipt: lat-lon atmosphere at 128 GPUs (job 26534060, ran 2026-07-30, unanalysed until now)
  1692	
  1693	LL2048x4096 L26, same bench + protocol (steps 12 / warmup 3) as the
  1694	@64 row (job 26502539, f32 6.73 ms):
  1695	
  1696	| arm | ms/step | GC/s (col-levels) |
  1697	|---|---|---|
  1698	| f32 @128 (65,536 cols/GPU) | 5.5767 | **39.11** |
  1699	| f64 @128 | 9.6015 | 22.72 |
  1700	
  1701	f32 strong 64->128: 1.207x for 2x devices (eff 0.60) with the tile at
  1702	**65.5k cols/GPU — comfortably ABOVE the ~30k floor** (codex round-21
  1703	caught the first draft halving this), so the loss is NOT
  1704	floor-attributable. Mechanism OPEN — candidates (uninstrumented): 1-D
  1705	band thinning to 16 rows/rank raising halo/compute ratio, and the
  1706	16 -> 32-node NCCL topology step. 39.11 GC/s (from 5.5767 ms) is the
  1707	highest measured throughput of ANY lane in the campaign. The companion
  1708	oc128 (26534067) FAILED pre-#1370-fix with the 109.5 GB resident-args
  1709	signature; retry submitted post-fix (below).
  1710	
  1711	## Hundreds-of-devices push (user directive 2026-08-02)
  1712	
  1713	"Push the scaling to hundreds of CPUs and GPUs for lat-lon and MPAS on
  1714	GPUs." Machine ceiling: 56 nodes x 4 = 224 a100_80 GPUs; compute
  1715	partition effectively unbounded for our rank counts. Submitted set:
  1716	
  1717	| job | what | devices | why |
  1718	|---|---|---|---|
  1719	| 26628021 | s9 ensemble contention | 128 GPU (4x32) | lever #1 receipt |
  1720	| 26628071 | oc LL2304 retry post-#1370 | 128 GPU | pre-fix failure was resident-args; predicted PASS at ~0.10 GB/dev residency |
  1721	| 26628072 | atm LL2304 @96/@192 + LL2880 @192 | 96-192 GPU | LL2048 does not divide 192; LL2880@192 = 86.4k cols/GPU ABOVE floor |
  1722	| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
  1723	| 26628074 | subdiv-10 lloyd0 prewarm | 1 CPU | unlocks MPAS 128-224 GPUs ABOVE floor (81.9k-46.8k cells/GPU) |
  1724	| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
  1725	
  1726	s10 ladder (128/192/224 GPUs) submits once 26628074's cache lands.
  1727	
  1728	### First hundreds receipt in: lat-lon CPU 2-D pencil to 512 ranks (job 26628073)
  1729	
  1730	r512 (512x1024 = 524k cols) L26 f64 moist, 32 rpn block:cyclic,
  1731	wall-pole 2-D pencil lane (labelled; NOT the pole fold):
  1732	
  1733	| ranks | cols/rank | ms/step | speedup vs np64 | eff |
  1734	|---|---|---|---|---|
  1735	| 64 | 8,192 | 297.57 | 1.00 | 1.00 |
  1736	| 128 | 4,096 | 161.03 | 1.848 | 0.92 |
  1737	| 256 | 2,048 | 72.06 | 4.129 | 1.03 |
  1738	| 512 | 1,024 | 44.78 | 6.645 | **0.83** |
  1739	
  1740	Distribution verified against the masquerade trap: result rows carry
  1741	`n_ranks: 512` (the JSON's `metadata.process_count: 1` is the jax-LOCAL
  1742	count on this mpi4jax lane, not the world size). 128->256 is
  1743	SUPERLINEAR (2.23x for 2x) — classic per-rank working-set cache
  1744	transition on Milan (mechanism PLAUSIBLE, uninstrumented). End-to-end
  1745	64->512 eff 0.83 at 1k cols/rank: the lat-lon CPU lane scales into the
  1746	hundreds cleanly. (Exact pencil factorisations are not recorded in the
  1747	result JSON — only `decomposition: 2d`; a follow-up could add them to
  1748	the bench metadata.)

--- ensemble ---
     1	#!/bin/bash -l
     2	#SBATCH --job-name=mpas_s9_ens
     3	#SBATCH --account=bb1596_gpu
     4	#SBATCH --partition=gpu
     5	#SBATCH --constraint=a100_80
     6	#SBATCH --nodes=32
     7	#SBATCH --gpus-per-node=4
     8	#SBATCH --exclusive
     9	#SBATCH --mem=0
    10	#SBATCH --time=01:30:00
    11	#SBATCH --output=mpas_s9_ens.%j.log
    12	# ENSEMBLE-PARALLELISM RECEIPT (codex 2026-07-31 consult lever #1; design
    13	# fixed per codex round-20 items 9-12).  The tile floor caps ONE
    14	# trajectory's strong scaling; past it the honest use of 128 GPUs is K
    15	# independent replicas at the per-trajectory sweet spot.  This job
    16	# measures the only thing that can refute that: FABRIC CONTENTION between
    17	# replicas sharing the IB tree.
    18	#
    19	#   solo_pre  : ONE 32-GPU s9 run, other 24 nodes idle
    20	#   phase B   : FOUR concurrent 32-GPU s9 runs on disjoint 8-node sets
    21	#               (SLURM_STEP_NODELIST is per-step -> per-replica
    22	#               coordinator autodetect; jobid-derived port shared but
    23	#               hosts differ)
    24	#   solo_post : solo again AFTER phase B (brackets ordering/thermal
    25	#               drift; contrast uses mean of the two solos)
    26	#
    27	# Falsifiability, written BEFORE submit:
    28	#   numbers : 2 solo + 4 replica steady_median_ms
    29	#   CONFIRM : max(replica) <= 1.10 x mean(solo) -> guaranteed aggregate
    30	#             >= 4/1.10 = 3.64x the 32-GPU solo rate (~19.9 GC/s if solo
    31	#             reproduces 5.47) = ~3.3x the observed 128-GPU
    32	#             single-trajectory rate.  NOT "4x": 1.10 is the bar, the
    33	#             margin below it is the measured contention.
    34	#   REFUTE  : any replica > 1.10x solo -> contention term, quantified
    35	#             per replica.
    36	# Protocol: config identical to job 26600095 np32 rung (sfc, lloyd 0,
    37	# f32, padded-128 reorder) EXCEPT steps 5000 / warmup 100 so the stepping
    38	# window (~60 s at 12.5 ms/step) dwarfs launch skew between replicas --
    39	# overlap is EVIDENCED, not assumed, by the per-step Start/End + NodeList
    40	# table sacct prints at the end.  Absolute ms/step is therefore only
    41	# compared WITHIN this job (solo vs replicas), never against the
    42	# steps-12 ladder rows.
    43	# INTERPRETATION LIMIT (codex r21 item 4): this design SCREENS for a
    44	# co-execution penalty vs the solo brackets; if a penalty appears, its
    45	# attribution (shared IB fabric vs node/topology placement vs drift)
    46	# needs the per-step nodelist table + follow-up, and the solo/replica
    47	# ordering is bracketed (pre+post), not fully counterbalanced.
    48	set -uo pipefail
    49	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    50	export JAX_PLATFORMS=cuda,cpu
    51	export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
    52	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    53	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    54	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    55	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    56	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_s9_ens_j${SLURM_JOB_ID}}"
    57	mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
    58	rc=0
    59	
    60	run_arm () { # tag  (one 32-GPU replica on 8 disjoint nodes)
    61	  echo "[$1] launch epoch=$(date +%s.%N)"
    62	  # rank 0 prints its step's nodelist BEFORE exec'ing the bench — the
    63	  # per-replica disjointness + coordinator-host evidence codex asked for
    64	  # ("$0" inside bash -c is $PY, passed as the first post-script arg).
    65	  JAX_ENABLE_X64=0 srun --nodes=8 --ntasks=32 --ntasks-per-node=4 \
    66	      --gpus-per-node=4 --gpu-bind=none --exact --kill-on-bad-exit=1 \
    67	      --job-name="arm_$1" \
    68	    bash -c '[ "${SLURM_PROCID:-1}" = 0 ] && echo "[step $ARM_TAG] nodelist=$SLURM_STEP_NODELIST"; exec "$0" "$@"' \
    69	    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    70	      --multicontroller --n-devices 32 \
    71	      --subdivision 9 --nlev 26 --steps 5000 --warmup 100 --lloyd 0 \
    72	      --partition-method sfc --reorder-for 128 \
    73	      --out "$OUTDIR/$1.jsonl"
    74	  s=$?
    75	  echo "[$1] exit=$s epoch=$(date +%s.%N)"
    76	  return $s
    77	}
    78	
    79	echo "=== solo_pre: 1x32 GPUs, 24 nodes idle ==="
    80	ARM_TAG=solo_pre; export ARM_TAG
    81	run_arm solo_pre || { echo "solo_pre FAILED"; rc=1; }
    82	
    83	echo "=== phase B: 4 concurrent 32-GPU replicas ==="
    84	pids=(); tags=(rep0 rep1 rep2 rep3)
    85	for T in "${tags[@]}"; do ARM_TAG=$T run_arm "$T" & pids+=($!); done
    86	for i in "${!pids[@]}"; do
    87	  wait "${pids[$i]}" || { echo "${tags[$i]} FAILED"; rc=1; }
    88	done
    89	
    90	echo "=== solo_post: 1x32 GPUs, 24 nodes idle ==="
    91	ARM_TAG=solo_post; export ARM_TAG
    92	run_arm solo_post || { echo "solo_post FAILED"; rc=1; }
    93	
    94	echo "=== RESULTS ==="
    95	for T in solo_pre rep0 rep1 rep2 rep3 solo_post; do
    96	  F="$OUTDIR/$T.jsonl"
    97	  "$PY" -c "
    98	import json,math,sys
    99	try:
   100	    d=json.loads(open('$F').readline())
   101	    ms=d['steady_median_ms']
   102	    assert math.isfinite(ms) and ms > 0, f'non-finite {ms}'
   103	except Exception as e:
   104	    print('$T: MISSING/INVALID ->', e); sys.exit(1)
   105	print(f'$T: {ms:8.2f} ms  {d.get(\"mcells_per_s\",0)/1000:.2f} GC/s')" \
   106	    || { echo "$T receipt invalid"; rc=1; }
   107	done
   108	
   109	echo "=== overlap evidence: per-step nodelist + wall window ==="
   110	sacct -j "$SLURM_JOB_ID" \
   111	  --format=JobID%18,JobName%12,NodeList%45,Start,End,State -P \
   112	  || { echo "sacct overlap table UNAVAILABLE"; rc=1; }
   113	echo "DONE rc=$rc"; exit $rc

--- atm hundreds ---
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

--- cpu 2d hundreds ---
     1	#!/bin/bash -l
     2	#SBATCH --job-name=cpu_ll2d
     3	#SBATCH --account=bb1596
     4	#SBATCH --partition=compute
     5	#SBATCH --nodes=16
     6	#SBATCH --exclusive
     7	#SBATCH --time=04:00:00
     8	#SBATCH --output=cpu_ll2d.%j.log
     9	# HUNDREDS-OF-CPUS lat-lon atmosphere (user directive 2026-08-02): the
    10	# 2-D pencil path (wall poles, labelled throughput lane — NOT the pole
    11	# fold; pole-matched A/B remains open) at r512 (512x1024 = 524k cols) so
    12	# np512 still holds ~1k cols/rank.  Self-contained strong ladder
    13	# np64->512, f64 moist, 32 rpn, block:cyclic per the placement receipt.
    14	set -uo pipefail
    15	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    16	export JAX_PLATFORMS=cpu
    17	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    18	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv-mpi/bin/python}"
    19	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    20	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    21	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/cpu_ll2d_j${SLURM_JOB_ID}}"
    22	mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
    23	rc=0
    24	for NP in 64 128 256 512; do
    25	  NODES=$(( NP / 32 )); [ "$NODES" -lt 1 ] && NODES=1
    26	  echo "--- atm latlon 2-D pencil r512 f64 np=$NP ---"
    27	  JAX_ENABLE_X64=1 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=32 \
    28	      --distribution=block:cyclic --cpu-bind=cores --kill-on-bad-exit=1 \
    29	    "$PY" scripts/bench/run_cpu_mpi_scaling.py \
    30	      --grid latlon --latlon-2d --mode single --physics moist \
    31	      --precision float64 --n-levels 26 --resolution 512 \
    32	      --n-warmup 2 --n-timing 6 \
    33	      --output-dir "$OUTDIR/np${NP}" < /dev/null || { echo "np$NP FAILED"; rc=1; }
    34	done
    35	echo "=== RESULTS ==="
    36	for NP in 64 128 256 512; do
    37	  F=$(find "$OUTDIR/np${NP}" -name "*.json" 2>/dev/null | head -1)
    38	  "$PY" -c "
    39	import json,math,sys
    40	try:
    41	    d=json.load(open('$F' or '/nonexistent'))
    42	    r=d['results'][-1] if isinstance(d.get('results'),list) else d
    43	    ms=r.get('ms_per_step') or r.get('time_per_step_ms')
    44	    assert ms is not None and math.isfinite(ms) and ms > 0
    45	except Exception as e:
    46	    print('np$NP: MISSING/INVALID ->', e); sys.exit(1)
    47	print(f'np$NP: {ms:9.2f} ms')" || { echo "np$NP receipt invalid"; rc=1; }
    48	done
    49	echo "DONE rc=$rc"; exit $rc

--- cpu receipt ---
outdir=/scratch/b/b381103/legoesm_scaling/cpu_ll2d_j26628073
--- atm latlon 2-D pencil r512 f64 np=64 ---
[l30517.lvt.dkrz.de:782347] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325538] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782354] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325539] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782345] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782362] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782340] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325541] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325543] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325542] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782350] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782341] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782355] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782353] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782344] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325544] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325533] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325540] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325525] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782363] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325546] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325547] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782366] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325545] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325535] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782337] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782338] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782348] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782358] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782360] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325548] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782361] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782365] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782367] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325536] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782339] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782342] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325537] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782359] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325526] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325518] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782357] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325530] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325529] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782336] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782346] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325527] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325534] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325549] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325520] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325522] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325521] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325532] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782356] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325519] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325523] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325528] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782351] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782352] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782349] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782364] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325524] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325531] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782343] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
========================================================================
  legoESM CPU MPI Scaling Benchmark
========================================================================
  Grid:      latlon
  Physics:   moist
  Precision: float64
  Ranks:     64
  Resolution:512
  Levels:    26
  Mode:      single
========================================================================
  [float64] LL512/L26 on 64 rank(s) | dt=0s | cells=13,631,488 | cells/rank=212,992 | physics=moist
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()

--- plot head/panels ---
     1	"""Publication figure: strong scaling per grid, per precision, vs ideal.
     2	
     3	One panel per (component, grid). Each panel plots measured ms/step against
     4	device count on log-log axes, one line per precision, with a DASHED IDEAL
     5	line anchored at each series' own base point (t_base * n_base / n).
     6	
     7	Every number is a measured Levante receipt; the SOURCES table below carries
     8	the SLURM job id for each series so a reader can trace any point. Panels
     9	with only one precision measured say so in-panel rather than leaving the
    10	reader to guess — no interpolation, no fabricated series.
    11	
    12	Anchoring note: ideal lines are anchored at each series' FIRST measured
    13	point. Where that base leg is cache-disadvantaged (a single device holding
    14	the whole problem), the measured curve can sit BELOW ideal; that is a
    15	property of the base leg, not superlinear parallelism, and is flagged in
    16	the caption rather than hidden by re-anchoring.
    17	
    18	Usage
    19	-----
    20	    python scripts/plot/plot_scaling_paper_figure.py --out fig_scaling.pdf
    21	"""
    22	from __future__ import annotations
    23	
    24	import argparse
    25	
    26	import matplotlib
    27	matplotlib.use("Agg")
    28	import matplotlib.pyplot as plt
    29	from matplotlib.lines import Line2D
    30	
    31	# --- Measured data -------------------------------------------------------
    32	# (devices, ms/step). Job ids are the provenance for each series.
    33	SOURCES = {
    34	    "atm_latlon": "26450848/26453240/26449147 (f32), 26494902 (f64), "
    35	                  "LL2048@64 26502539, LL2048@128 26534060",
    36	    "atm_cube": "26452894/26453782",
    37	    "atm_mpas": "26454476/26454618/26486288/26493638/26493734, "
    38	                "s8 np32-128 26549646/26538474, s9 26600095",
    39	    "atm_ico_cpu": "26495083 (f32), 26495437 (f64) — both block:cyclic; "
    40	                   "lat-lon 2-D r512 26628073",
    41	    "oc_latlon": "26460444-501/26460365/26493592",
    42	    "oc_tripole": "26493837/26493648",
    43	    "oc_mpas": "26494036 (f64), 26494908 (f32)",
    44	}
    45	
    46	PANELS = [
    47	    dict(
    48	        key="atm_latlon", title="lat–lon", sub="720×1440 L26 · A100 NCCL",
    49	        series=[("float32", [(4, 7.72), (8, 5.40), (16, 3.54)]),
    50	                ("float64", [(4, 16.11), (8, 11.34), (16, 5.62)]),
    51	                ("float32 (LL2048)", [(64, 6.73), (128, 5.58)]),
    52	                ],
    53	        scatter=[("LL1536 @64", 64, 4.97), ("LL2048 f64 @128", 128, 9.60)],
    54	        note="LL2048@128 = 39.1 GC/s",
    55	    ),
    56	    dict(
    57	        key="atm_cube", title="cubed-sphere", sub="C384/C768 L60 · A100 NCCL",
    58	        series=[("float32 (C768)", [(6, 58.35), (24, 14.09)]),
    59	                ("float32 (C384)", [(6, 15.44), (24, 8.81)])],
    60	        note="f64 pending",
    61	    ),
    62	    dict(
    63	        key="atm_mpas", title="MPAS icosahedral", sub="subdiv-8/9 L26 · A100 NCCL",
    64	        series=[("float32 (subdiv-8)", [(2, 19.90), (4, 14.12), (8, 6.92),
    65	                                        (16, 7.10), (32, 8.13), (64, 5.27),
    66	                                        (128, 6.47)]),
    67	                ("float32 (subdiv-9)", [(32, 12.47), (64, 9.60), (128, 11.48)]),
    68	                ("float64 (subdiv-8)", [(2, 38.34), (4, 20.09), (8, 18.98)])],
    69	        note="s8 production mesh;\ns9 lloyd-0 synthetic",
    70	    ),
    71	    dict(
    72	        key="atm_ico_cpu", title="ico + lat-lon 2-D", sub="subdiv-7 / r512 L26 · Milan CPU–MPI",
    73	        series=[("float32", [(1, 7399.72), (2, 3250.27), (4, 1673.07),
    74	                             (8, 827.06), (16, 444.58), (32, 230.35),
    75	                             (64, 131.58)]),
    76	                ("float64", [(1, 9987.81), (2, 4727.46), (4, 2349.06),
    77	                             (8, 1161.98), (16, 624.06), (32, 350.89),
    78	                             (64, 220.57), (128, 92.4), (256, 56.2),
    79	                             (512, 66.7)]),
    80	                ("f64 lat-lon 2-D (r512)", [(64, 297.57), (128, 161.03),
    81	                                            (256, 72.06), (512, 44.78)])],
    82	        note="ico to 1024; lat-lon 2-D\neff 0.83 @512",
    83	    ),
    84	    dict(
    85	        key="oc_latlon", title="lat–lon", sub="576×1152 L20 · A100 NCCL",
    86	        series=[("float32", [(1, 36.45), (2, 22.48), (4, 12.84), (8, 11.04), (16, 8.71)]),
    87	                ("float64", [(1, 64.81), (2, 41.63), (4, 22.09)]),
    88	                ("mixed (f64 store)", [(1, 52.80), (4, 19.13)])],
    89	        note="best arm shown",
    90	    ),
    91	    dict(
    92	        key="oc_tripole", title="tripole (ORCA fold)", sub="576×1152 L20 · A100 NCCL",
    93	        series=[("float32", [(1, 35.52), (2, 25.03), (4, 15.91)]),
    94	                ("float64", [(1, 63.84), (2, 43.36), (4, 24.12)])],
    95	        note="fold +1.2–3.7 %",
    96	    ),
    97	    dict(
    98	        key="oc_mpas", title="MPAS Voronoi", sub="subdiv-7/8 · Milan CPU–MPI, 32 rpn",
    99	        series=[("float64 (subdiv-7)", [(32, 190.22), (64, 147.65),
   100	                                        (128, 102.93), (256, 65.71),
   101	                                        (512, 63.83)]),
   102	                ("float64 (subdiv-8)", [(32, 861.25), (64, 494.89),
   103	                                        (128, 309.05), (256, 254.41),
   104	                                        (512, 194.80)])],
   105	        note="32 ranks/node fixed",
   106	    ),
   107	]
   108	
   109	COLORS = {"float32": "#0072B2", "float64": "#D55E00",
   110	          "f32 · LL1536/2048 @64": "#009E73",
   111	          "float64 (packed)": "#E69F00",
   112	          "mixed (f64 store)": "#009E73",
   113	          "float32 (C768)": "#0072B2", "float32 (C384)": "#56B4E9",
   114	          "float32 (LL2048)": "#009E73",
   115	          "f64 lat-lon 2-D (r512)": "#CC79A7",
   116	          "float32 (subdiv-8)": "#0072B2", "float32 (subdiv-9)": "#56B4E9",
   117	          "float64 (subdiv-7)": "#D55E00", "float64 (subdiv-8)": "#E69F00"}
   118	MARKERS = {"float32": "o", "float64": "s", "mixed (f64 store)": "D",
   119	           "f32 · LL1536/2048 @64": "*",
   120	           "float64 (packed)": "s",
   121	           "float32 (C768)": "o", "float32 (C384)": "^",
   122	           "float32 (LL2048)": "^",
   123	           "f64 lat-lon 2-D (r512)": "D",
   124	           "float32 (subdiv-8)": "o", "float32 (subdiv-9)": "^",
   125	           "float64 (subdiv-7)": "s", "float64 (subdiv-8)": "v"}
   126	
   127	
   128	def _style():
   129	    plt.rcParams.update({
   130	        "font.family": "sans-serif",
   131	        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
   132	        "font.size": 7,
   133	        "axes.labelsize": 7.5,
   134	        "axes.titlesize": 8,
   135	        "xtick.labelsize": 6.5,
   136	        "ytick.labelsize": 6.5,
   137	        "legend.fontsize": 6,
   138	        "axes.linewidth": 0.6,
   139	        "xtick.major.width": 0.6,
   140	        "ytick.major.width": 0.6,
   141	        "xtick.minor.width": 0.4,
   142	        "ytick.minor.width": 0.4,
   143	        "lines.linewidth": 1.1,
   144	        "lines.markersize": 3.4,
   145	        "figure.dpi": 300,
   146	        "savefig.dpi": 300,
   147	        "pdf.fonttype": 42,   # editable text in the PDF (journal requirement)
   148	        "ps.fonttype": 42,
   149	    })
   150	
   151	
   152	def main() -> int:
   153	    ap = argparse.ArgumentParser(description=__doc__)
   154	    ap.add_argument("--out", default="fig_scaling.pdf")
   155	    ap.add_argument("--png", default=None, help="also write a PNG here")
   156	    args = ap.parse_args()
   157	
   158	    _style()
   159	    ncol, nrow = 4, 2
   160	    # Nature double-column ~180 mm
   161	    fig, axs = plt.subplots(nrow, ncol, figsize=(180 / 25.4, 100 / 25.4))
   162	    axs = axs.ravel()
   163	
   164	    for i, spec in enumerate(PANELS):
   165	        ax = axs[i]
   166	        for label, pts in spec["series"]:
   167	            xs = [p[0] for p in pts]
   168	            ys = [p[1] for p in pts]
   169	            c = COLORS.get(label, "#444444")
   170	            ax.plot(xs, ys, MARKERS.get(label, "o") + "-", color=c,
   171	                    label=label, markerfacecolor="white",
   172	                    markeredgewidth=0.9, clip_on=False, zorder=3)
   173	            # ideal anchored at this series' own base point
   174	            n0, t0 = xs[0], ys[0]
   175	            ideal = [t0 * n0 / n for n in xs]
   176	            ax.plot(xs, ideal, "--", color=c, lw=0.7, alpha=0.55, zorder=2)
   177	
   178	        for lab, x, y in spec.get("scatter", []):
   179	            ax.plot([x], [y], "*", color="#009E73", markersize=7,
   180	                    markeredgewidth=0.8, clip_on=False, zorder=4)
   181	            ax.annotate(lab, (x, y), fontsize=5.2, color="#009E73",
   182	                        textcoords="offset points", xytext=(-4, 5), ha="right")
   183	
   184	        ax.set_xscale("log", base=2)
   185	        ax.set_yscale("log")
   186	        allx = sorted({p[0] for _, pts in spec["series"] for p in pts}
   187	                      | {x for _, x, _ in spec.get("scatter", [])})
   188	        # thin crowded tick sets to powers spanning the range
   189	        if len(allx) > 6:
   190	            allx = [x for i, x in enumerate(allx) if i % 3 == 0 or x == allx[-1]]
   191	        ax.set_xticks(allx)
   192	        ax.set_xticklabels([str(x) for x in allx])
   193	        ax.minorticks_off()
   194	        ax.set_title(spec["title"], pad=9, loc="left", fontweight="bold")
   195	        ax.text(0, 1.015, spec["sub"], transform=ax.transAxes, fontsize=5.5,
   196	                color="#555555", va="bottom")
   197	        if spec["note"]:
   198	            ax.text(0.98, 0.98, spec["note"], transform=ax.transAxes,
   199	                    fontsize=5.2, color="#888888", ha="right", va="top",
   200	                    style="italic")
   201	        ax.tick_params(direction="out", length=2.5)
   202	        for s in ("top", "right"):
   203	            ax.spines[s].set_visible(False)
   204	        ax.legend(frameon=False, loc="lower left", handlelength=1.6,
   205	                  borderpad=0.2, labelspacing=0.25)
   206	        if i % ncol == 0:
   207	            ax.set_ylabel("time per step (ms)")
   208	        if i >= ncol:
   209	            ax.set_xlabel("devices (GPUs or MPI ranks)")
   210	        # panel letter
   211	        ax.text(-0.30, 1.22, chr(ord("a") + i), transform=ax.transAxes,
   212	                fontsize=9, fontweight="bold", va="top")
   213	
   214	    # last cell: legend/provenance instead of an empty frame
   215	    ax = axs[len(PANELS)]
   216	    ax.axis("off")
   217	    handles = [
   218	        Line2D([], [], color="#0072B2", marker="o", markerfacecolor="white",
   219	               markeredgewidth=0.9, label="float32"),
   220	        Line2D([], [], color="#D55E00", marker="s", markerfacecolor="white",
/usr/bin/bash: -c: line 12: unexpected EOF while looking for matching `''
/usr/bin/bash: -c: line 13: syntax error: unexpected end of file

exec
/usr/bin/bash -lc "printf '%s\\n' '--- references to current ensemble job/claims ---'
rg -n -i -C 2 '26628196|26628021|ensemble|INTERPRETATION LIMIT|counterbalanced|fabric-contention|fabric contention|co-execution' docs/performance/scaling/levante_campaign_2026-07-24.md scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch .physics-validator/scaling_campaign -g '*.md' -g '*.sbatch' | tail -n 260
printf '%s\\n' '--- artifact inventory ---'
rg --files -uu | rg '26628196|26628021|mpas_s9_ens|26628073|cpu_ll2d'
printf '%s\\n' '--- full cpu script, pared receipt fields ---'
nl -ba scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch
rg -n 'RESULT|n_ranks|process_count|steady_median_ms|mcells_per_s|git_sha|decomposition|ERROR|Traceback|COMPLETED|DONE|mpi4jax' cpu_ll2d.26628073.log | tail -n 160
printf '%s\\n' '--- JSONL-like lines ---'
rg '"'^'"\\{' cpu_ll2d.26628073.log | tail -n 20
printf '%s\\n' '--- current script syntax ---'
bash -n scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch && echo 'bash -n: PASS'
printf '%s\\n' '--- relevant bench lines ---'
nl -ba scripts/bench/bench_ocean_mpas_scaling.py | sed -n '740,758p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 515ms:
--- references to current ensemble job/claims ---
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1031:  1652	### Next (submitted 2026-08-02): ensemble receipt, job 26627810
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1032-  1653	
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1033-  1654	Codex lever #1 — 4 concurrent 32-GPU s9 replicas vs a same-job solo
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1034:  1655	control (`scripts/tmp/mpas_s9_ensemble.sbatch`). CONFIRM bar: max
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1035-  1656	replica <= 1.10x solo => aggregate >= 3.6x/4x past the floor; REFUTE:
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1036-  1657	the contention term, quantified. This is the only remaining ranked
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1078-results/amip/C48_L26_1d_20260724_135800/run_manifest.json-109-      "enable_tiled_dycore": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1079-results/amip/C48_L26_1d_20260724_135800/run_manifest.json-110-      "energy_consistent_moisture_clip": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1080:results/amip/C48_L26_1d_20260724_135800/run_manifest.json:111:      "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1081-results/amip/C48_L26_1d_20260724_135800/run_manifest.json-112-      "era5_land_ic_path": "",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1082-results/amip/C48_L26_1d_20260724_135800/run_manifest.json-113-      "experiment": "",
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1121-results/amip/C48_L26_1d_20260724_135800/experiment_config.json-275-  "distributed": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1122-results/amip/C48_L26_1d_20260724_135800/experiment_config.json-276-  "distributed_mode": "mpi",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1123:results/amip/C48_L26_1d_20260724_135800/experiment_config.json:277:  "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1124-results/amip/C48_L26_1d_20260724_135800/experiment_config.json-278-  "n_devices": 3,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1125-results/amip/C48_L26_1d_20260724_135800/experiment_config.json-279-  "shard_radiation_columns": false,
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1129-results/amip/C48_L26_1d_20260724_135623/run_manifest.json-109-      "enable_tiled_dycore": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1130-results/amip/C48_L26_1d_20260724_135623/run_manifest.json-110-      "energy_consistent_moisture_clip": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1131:results/amip/C48_L26_1d_20260724_135623/run_manifest.json:111:      "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1132-results/amip/C48_L26_1d_20260724_135623/run_manifest.json-112-      "era5_land_ic_path": "",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1133-results/amip/C48_L26_1d_20260724_135623/run_manifest.json-113-      "experiment": "",
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1145-results/amip/C96_L26_1d_20260724_135008/run_manifest.json-109-      "enable_tiled_dycore": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1146-results/amip/C96_L26_1d_20260724_135008/run_manifest.json-110-      "energy_consistent_moisture_clip": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1147:results/amip/C96_L26_1d_20260724_135008/run_manifest.json:111:      "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1148-results/amip/C96_L26_1d_20260724_135008/run_manifest.json-112-      "era5_land_ic_path": "",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1149-results/amip/C96_L26_1d_20260724_135008/run_manifest.json-113-      "experiment": "",
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1236-results/amip/C48_L26_1d_20260724_135623/experiment_config.json-275-  "distributed": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1237-results/amip/C48_L26_1d_20260724_135623/experiment_config.json-276-  "distributed_mode": "mpi",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1238:results/amip/C48_L26_1d_20260724_135623/experiment_config.json:277:  "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1239-results/amip/C48_L26_1d_20260724_135623/experiment_config.json-278-  "n_devices": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1240-results/amip/C48_L26_1d_20260724_135623/experiment_config.json-279-  "shard_radiation_columns": false,
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1244-scripts/validate/validate_sdm_vs_pysdm.py-5-canonical Shima-2009 Golovin box (exponential spectrum, N0 = 2^23 m^-3,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1245-scripts/validate/validate_sdm_vs_pysdm.py-6-mean volume 1.19e5 µm³, b = 1500/s, dt = 1 s) with independent Monte-Carlo
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1246:scripts/validate/validate_sdm_vs_pysdm.py:7:machinery; their ensemble statistics must agree with each other and with the
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1247-scripts/validate/validate_sdm_vs_pysdm.py-8-exact Golovin/Scott analytic — a far stronger consistency check than analytic
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1248-scripts/validate/validate_sdm_vs_pysdm.py-9-moments alone.
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1252-scripts/validate/validate_sdm_vs_pysdm.py-49-N_SD = 2 ** 15
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1253-scripts/validate/validate_sdm_vs_pysdm.py-50-V_CELL = 1.0                        # per-m^3 box on our side (scale-invariant)
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1254:scripts/validate/validate_sdm_vs_pysdm.py:51:ENSEMBLE = 8                        # average our MC over a few seeds
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1255-scripts/validate/validate_sdm_vs_pysdm.py-52-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1256-scripts/validate/validate_sdm_vs_pysdm.py-53-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1257:scripts/validate/validate_sdm_vs_pysdm.py:54:def run_legoesm_ensemble():
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1258-scripts/validate/validate_sdm_vs_pysdm.py-55-    cfg = SDMConfig(collision_kernel="golovin", golovin_b=B_GOLOVIN)
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1259-scripts/validate/validate_sdm_vs_pysdm.py-56-    x0 = _RHO_W * XBAR_M3           # mean droplet MASS [kg]
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1260-scripts/validate/validate_sdm_vs_pysdm.py-57-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1261-scripts/validate/validate_sdm_vs_pysdm.py-58-    results = {t: [] for t in SNAPS}
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1262:scripts/validate/validate_sdm_vs_pysdm.py:59:    for seed in range(ENSEMBLE):
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1263-scripts/validate/validate_sdm_vs_pysdm.py-60-        k_init, k_run = random.split(random.PRNGKey(seed))
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1264-scripts/validate/validate_sdm_vs_pysdm.py-61-        droplets = exponential_water_droplets(k_init, N_SD, N0_PER_M3 * V_CELL, x0)
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1268-scripts/validate/validate_sdm_vs_pysdm.py-104-            f"ragged reference arrays at t={t}"
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1269-scripts/validate/validate_sdm_vs_pysdm.py-105-        assert np.all(np.isfinite(ref[f"radius_{t}"])), f"non-finite radii at t={t}"
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1270:scripts/validate/validate_sdm_vs_pysdm.py:106:    ours = run_legoesm_ensemble()
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1271-scripts/validate/validate_sdm_vs_pysdm.py-107-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1272-scripts/validate/validate_sdm_vs_pysdm.py-108-    lnr_edges = np.linspace(np.log(5e-6), np.log(5e-3), 50)
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1276-results/amip/C96_L26_1d_20260724_135942/run_manifest.json-109-      "enable_tiled_dycore": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1277-results/amip/C96_L26_1d_20260724_135942/run_manifest.json-110-      "energy_consistent_moisture_clip": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1278:results/amip/C96_L26_1d_20260724_135942/run_manifest.json:111:      "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1279-results/amip/C96_L26_1d_20260724_135942/run_manifest.json-112-      "era5_land_ic_path": "",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1280-results/amip/C96_L26_1d_20260724_135942/run_manifest.json-113-      "experiment": "",
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1292-results/amip/C192_L26_1d_20260724_140350/run_manifest.json-109-      "enable_tiled_dycore": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1293-results/amip/C192_L26_1d_20260724_140350/run_manifest.json-110-      "energy_consistent_moisture_clip": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1294:results/amip/C192_L26_1d_20260724_140350/run_manifest.json:111:      "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1295-results/amip/C192_L26_1d_20260724_140350/run_manifest.json-112-      "era5_land_ic_path": "",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1296-results/amip/C192_L26_1d_20260724_140350/run_manifest.json-113-      "experiment": "",
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1391-results/amip/C96_L26_1d_20260724_140034/run_manifest.json-109-      "enable_tiled_dycore": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1392-results/amip/C96_L26_1d_20260724_140034/run_manifest.json-110-      "energy_consistent_moisture_clip": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1393:results/amip/C96_L26_1d_20260724_140034/run_manifest.json:111:      "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1394-results/amip/C96_L26_1d_20260724_140034/run_manifest.json-112-      "era5_land_ic_path": "",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1395-results/amip/C96_L26_1d_20260724_140034/run_manifest.json-113-      "experiment": "",
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1486-results/amip/C96_L26_1d_20260724_140034/experiment_config.json-275-  "distributed": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1487-results/amip/C96_L26_1d_20260724_140034/experiment_config.json-276-  "distributed_mode": "mpi",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1488:results/amip/C96_L26_1d_20260724_140034/experiment_config.json:277:  "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1489-results/amip/C96_L26_1d_20260724_140034/experiment_config.json-278-  "n_devices": 3,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1490-results/amip/C96_L26_1d_20260724_140034/experiment_config.json-279-  "shard_radiation_columns": false,
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1532-docs/ocean/fidelity/phase_e_climate_infra.md-85-Phase F (OMIP-2 + Bryan THC long runs) can now drive a multi-decade
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1533-docs/ocean/fidelity/phase_e_climate_infra.md-86-forced ocean integration and diff the output against the published
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1534:docs/ocean/fidelity/phase_e_climate_infra.md:87:ensembles via the climate-diagnostics package.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1535---
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1536-results/amip/C192_L26_1d_20260724_140350/experiment_config.json-193-  "T_ice": 271.35,
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1545-results/amip/C192_L26_1d_20260724_140350/experiment_config.json-275-  "distributed": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1546-results/amip/C192_L26_1d_20260724_140350/experiment_config.json-276-  "distributed_mode": "mpi",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1547:results/amip/C192_L26_1d_20260724_140350/experiment_config.json:277:  "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1548-results/amip/C192_L26_1d_20260724_140350/experiment_config.json-278-  "n_devices": 3,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1549-results/amip/C192_L26_1d_20260724_140350/experiment_config.json-279-  "shard_radiation_columns": false,
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1561-results/amip/C96_L26_1d_20260724_135008/experiment_config.json-275-  "distributed": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1562-results/amip/C96_L26_1d_20260724_135008/experiment_config.json-276-  "distributed_mode": "mpi",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1563:results/amip/C96_L26_1d_20260724_135008/experiment_config.json:277:  "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1564-results/amip/C96_L26_1d_20260724_135008/experiment_config.json-278-  "n_devices": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1565-results/amip/C96_L26_1d_20260724_135008/experiment_config.json-279-  "shard_radiation_columns": false,
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1577-results/amip/C96_L26_1d_20260724_135942/experiment_config.json-275-  "distributed": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1578-results/amip/C96_L26_1d_20260724_135942/experiment_config.json-276-  "distributed_mode": "mpi",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1579:results/amip/C96_L26_1d_20260724_135942/experiment_config.json:277:  "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1580-results/amip/C96_L26_1d_20260724_135942/experiment_config.json-278-  "n_devices": 2,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1581-results/amip/C96_L26_1d_20260724_135942/experiment_config.json-279-  "shard_radiation_columns": false,
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1593-results/amip/C192_L26_1d_20260724_140132/run_manifest.json-109-      "enable_tiled_dycore": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1594-results/amip/C192_L26_1d_20260724_140132/run_manifest.json-110-      "energy_consistent_moisture_clip": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1595:results/amip/C192_L26_1d_20260724_140132/run_manifest.json:111:      "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1596-results/amip/C192_L26_1d_20260724_140132/run_manifest.json-112-      "era5_land_ic_path": "",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1597-results/amip/C192_L26_1d_20260724_140132/run_manifest.json-113-      "experiment": "",
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1733-results/amip/C96_L26_1d_20260724_135851/run_manifest.json-109-      "enable_tiled_dycore": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1734-results/amip/C96_L26_1d_20260724_135851/run_manifest.json-110-      "energy_consistent_moisture_clip": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1735:results/amip/C96_L26_1d_20260724_135851/run_manifest.json:111:      "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1736-results/amip/C96_L26_1d_20260724_135851/run_manifest.json-112-      "era5_land_ic_path": "",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1737-results/amip/C96_L26_1d_20260724_135851/run_manifest.json-113-      "experiment": "",
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1778-results/amip/C42_L26_1d_20260724_134326/run_manifest.json-109-      "enable_tiled_dycore": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1779-results/amip/C42_L26_1d_20260724_134326/run_manifest.json-110-      "energy_consistent_moisture_clip": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1780:results/amip/C42_L26_1d_20260724_134326/run_manifest.json:111:      "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1781-results/amip/C42_L26_1d_20260724_134326/run_manifest.json-112-      "era5_land_ic_path": "",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1782-results/amip/C42_L26_1d_20260724_134326/run_manifest.json-113-      "experiment": "",
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1802-results/amip/C192_L26_1d_20260724_140132/experiment_config.json-275-  "distributed": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1803-results/amip/C192_L26_1d_20260724_140132/experiment_config.json-276-  "distributed_mode": "mpi",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1804:results/amip/C192_L26_1d_20260724_140132/experiment_config.json:277:  "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1805-results/amip/C192_L26_1d_20260724_140132/experiment_config.json-278-  "n_devices": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1806-results/amip/C192_L26_1d_20260724_140132/experiment_config.json-279-  "shard_radiation_columns": false,
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1810-results/amip/C42_L26_1d_20260724_134618/run_manifest.json-109-      "enable_tiled_dycore": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1811-results/amip/C42_L26_1d_20260724_134618/run_manifest.json-110-      "energy_consistent_moisture_clip": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1812:results/amip/C42_L26_1d_20260724_134618/run_manifest.json:111:      "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1813-results/amip/C42_L26_1d_20260724_134618/run_manifest.json-112-      "era5_land_ic_path": "",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1814-results/amip/C42_L26_1d_20260724_134618/run_manifest.json-113-      "experiment": "",
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1869-results/amip/C96_L26_1d_20260724_135851/experiment_config.json-275-  "distributed": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1870-results/amip/C96_L26_1d_20260724_135851/experiment_config.json-276-  "distributed_mode": "mpi",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1871:results/amip/C96_L26_1d_20260724_135851/experiment_config.json:277:  "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1872-results/amip/C96_L26_1d_20260724_135851/experiment_config.json-278-  "n_devices": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1873-results/amip/C96_L26_1d_20260724_135851/experiment_config.json-279-  "shard_radiation_columns": false,
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1897-results/amip/C24_L26_1d_20260724_134422/run_manifest.json-109-      "enable_tiled_dycore": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1898-results/amip/C24_L26_1d_20260724_134422/run_manifest.json-110-      "energy_consistent_moisture_clip": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1899:results/amip/C24_L26_1d_20260724_134422/run_manifest.json:111:      "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1900-results/amip/C24_L26_1d_20260724_134422/run_manifest.json-112-      "era5_land_ic_path": "",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1901-results/amip/C24_L26_1d_20260724_134422/run_manifest.json-113-      "experiment": "",
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1913-results/amip/C48_L26_1d_20260724_134725/run_manifest.json-109-      "enable_tiled_dycore": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1914-results/amip/C48_L26_1d_20260724_134725/run_manifest.json-110-      "energy_consistent_moisture_clip": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1915:results/amip/C48_L26_1d_20260724_134725/run_manifest.json:111:      "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1916-results/amip/C48_L26_1d_20260724_134725/run_manifest.json-112-      "era5_land_ic_path": "",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1917-results/amip/C48_L26_1d_20260724_134725/run_manifest.json-113-      "experiment": "",
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1937-results/amip/C42_L26_1d_20260724_134618/experiment_config.json-275-  "distributed": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1938-results/amip/C42_L26_1d_20260724_134618/experiment_config.json-276-  "distributed_mode": "mpi",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1939:results/amip/C42_L26_1d_20260724_134618/experiment_config.json:277:  "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1940-results/amip/C42_L26_1d_20260724_134618/experiment_config.json-278-  "n_devices": 3,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1941-results/amip/C42_L26_1d_20260724_134618/experiment_config.json-279-  "shard_radiation_columns": false,
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1953-results/amip/C42_L26_1d_20260724_134326/experiment_config.json-275-  "distributed": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1954-results/amip/C42_L26_1d_20260724_134326/experiment_config.json-276-  "distributed_mode": "mpi",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1955:results/amip/C42_L26_1d_20260724_134326/experiment_config.json:277:  "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1956-results/amip/C42_L26_1d_20260724_134326/experiment_config.json-278-  "n_devices": 3,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1957-results/amip/C42_L26_1d_20260724_134326/experiment_config.json-279-  "shard_radiation_columns": false,
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1969-results/amip/C48_L26_1d_20260724_134725/experiment_config.json-275-  "distributed": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1970-results/amip/C48_L26_1d_20260724_134725/experiment_config.json-276-  "distributed_mode": "mpi",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1971:results/amip/C48_L26_1d_20260724_134725/experiment_config.json:277:  "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1972-results/amip/C48_L26_1d_20260724_134725/experiment_config.json-278-  "n_devices": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1973-results/amip/C48_L26_1d_20260724_134725/experiment_config.json-279-  "shard_radiation_columns": false,
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1985-results/amip/C24_L26_1d_20260724_134422/experiment_config.json-275-  "distributed": false,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1986-results/amip/C24_L26_1d_20260724_134422/experiment_config.json-276-  "distributed_mode": "mpi",
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:1987:results/amip/C24_L26_1d_20260724_134422/experiment_config.json:277:  "ensemble_size": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1988-results/amip/C24_L26_1d_20260724_134422/experiment_config.json-278-  "n_devices": 1,
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-1989-results/amip/C24_L26_1d_20260724_134422/experiment_config.json-279-  "shard_radiation_columns": false,
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-2040-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-2041-codex
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:2042:The headline arithmetic is mostly close, but I’ve already found a material experiment-design flaw: background `srun` steps are not explicitly assigned disjoint nodes. I’m tracing the multicontroller bootstrap and benchmark output semantics now, since that determines whether the ensemble receipt measures four replicas or accidental GPU/node overlap.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-2043-exec
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-2044-/usr/bin/bash -lc "printf '%s\\n' '--- multicontroller and timing implementation ---'; nl -ba scripts/bench/bench_mpas_spmd_scaling.py | sed -n '90,420p'; printf '%s\\n' '--- launch / launcher helpers ---'; rg -n -C 5 'multicontroller|SLURM_STEP_NODELIST|coordinator|port|n-devices|steady_median_ms|mcells_per_s|warmup|per_step' scripts/bench/bench_mpas_spmd_scaling.py scripts/cluster/scaling_levante/_env.sh src/legoesm | head -n 1400; printf '%s\\n' '--- near campaign s8 records ---'; rg -n -C 8 '26549646|26538474|6\\.92|8\\.13|5\\.27|6\\.47|3\\.23|subdiv-8.*np64|np64.*subdiv-8' docs/performance/scaling scripts/tmp scripts/cluster 2>/dev/null" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-3647-docs/performance/scaling/levante_campaign_2026-07-24.md:1650:  26549646/26538474) measured with the same bench and protocol.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-3648-docs/performance/scaling/levante_campaign_2026-07-24.md-1651-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:3649:docs/performance/scaling/levante_campaign_2026-07-24.md-1652-### Next (submitted 2026-08-02): ensemble receipt, job 26627810
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-3650-docs/performance/scaling/levante_campaign_2026-07-24.md-1653-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-3651-docs/performance/scaling/levante_campaign_2026-07-24.md-1654-Codex lever #1 — 4 concurrent 32-GPU s9 replicas vs a same-job solo
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:3652:docs/performance/scaling/levante_campaign_2026-07-24.md-1655-control (`scripts/tmp/mpas_s9_ensemble.sbatch`). CONFIRM bar: max
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-3653-docs/performance/scaling/levante_campaign_2026-07-24.md-1656-replica <= 1.10x solo => aggregate >= 3.6x/4x past the floor; REFUTE:
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-3654-docs/performance/scaling/levante_campaign_2026-07-24.md-1657-the contention term, quantified. This is the only remaining ranked
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-6081-    print(type(e).__name__, e)
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-6082-PY
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6083:printf '%s\\n' '--- shell static checks ---'; command -v shellcheck >/dev/null && shellcheck -x scripts/tmp/mpas_s9_ensemble.sbatch || true; bash -n scripts/tmp/mpas_s9_ensemble.sbatch; printf '%s\\n' '--- git ignores / tracking ---'; git check-ignore -v scripts/tmp/mpas_s9_ensemble.sbatch || true; git ls-files --error-unmatch scripts/tmp/mpas_s9_ensemble.sbatch 2>/dev/null || true" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-6084- succeeded in 373ms:
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-6085---- jax distributed/slurm source ---
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-6088---- shell static checks ---
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-6089---- git ignores / tracking ---
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:6090:.gitignore:120:scripts/tmp/	scripts/tmp/mpas_s9_ensemble.sbatch
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-6091-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-6092-exec
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-7554---
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-7555-docs/performance/scaling/levante_campaign_2026-07-24.md-1654-Codex lever #1 — 4 concurrent 32-GPU s9 replicas vs a same-job solo
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:7556:docs/performance/scaling/levante_campaign_2026-07-24.md-1655-control (`scripts/tmp/mpas_s9_ensemble.sbatch`). CONFIRM bar: max
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-7557-docs/performance/scaling/levante_campaign_2026-07-24.md-1656-replica <= 1.10x solo => aggregate >= 3.6x/4x past the floor; REFUTE:
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-7558-docs/performance/scaling/levante_campaign_2026-07-24.md-1657-the contention term, quantified. This is the only remaining ranked
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12775-docs/performance/scaling/levante_campaign_2026-07-24.md:1650:  26549646/26538474) measured with the same bench and protocol.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12776-docs/performance/scaling/levante_campaign_2026-07-24.md-1651-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:12777:docs/performance/scaling/levante_campaign_2026-07-24.md-1652-### Next (submitted 2026-08-02): ensemble receipt, job 26627810
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12778-docs/performance/scaling/levante_campaign_2026-07-24.md-1653-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12779-docs/performance/scaling/levante_campaign_2026-07-24.md-1654-Codex lever #1 — 4 concurrent 32-GPU s9 replicas vs a same-job solo
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:12780:docs/performance/scaling/levante_campaign_2026-07-24.md-1655-control (`scripts/tmp/mpas_s9_ensemble.sbatch`). CONFIRM bar: max
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12781-docs/performance/scaling/levante_campaign_2026-07-24.md-1656-replica <= 1.10x solo => aggregate >= 3.6x/4x past the floor; REFUTE:
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12782-docs/performance/scaling/levante_campaign_2026-07-24.md-1657-the contention term, quantified. This is the only remaining ranked
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12953-8. “Floor confirmed” should be **PLAUSIBLE/consistent with the floor**, not CONFIRMED: one below-floor point anti-scales, with no replication and known shape/codegen variation.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12954-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:12955:9. The ensemble script masks failures: each background `run_arm` changes `rc` only in its subshell, and `srun ... || { rc=1; }` returns success. Missing/corrupt JSON is silently skipped, so the job can print `DONE rc=0`. [The script](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/tmp/mpas_s9_ensemble.sbatch:44) must preserve each background PID/status and require all five valid receipts. It is also ignored by Git, so it is not reproducibly reviewable.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12956-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12957-10. GPU binding is otherwise sound: `--gpu-bind=none` leaves all GPUs visible and JAX’s Slurm auto-detection selects by local rank. Full-GPU `--exact` requests should force disjoint eight-node steps. But log each `SLURM_STEP_NODELIST` and coordinator host; that is the evidence the shared JAX job-ID port is safe.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12958-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:12959:11. The ensemble does not synchronize measurement windows, pair each replica topology with a solo control, or counterbalance solo-first ordering. It can measure different node/fabric groups or non-overlapping windows rather than contention. Twelve steps give nine dependent steady samples per launch, not enough independent evidence for a 10% confirmation bar.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12960-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:12961:12. Ensemble arithmetic needs tighter wording: if solo reproduces 5.47 GC/s, ideal aggregate is 21.88 GC/s. The 1.10 threshold guarantees only 19.89 GC/s = 3.636× the 32-GPU solo rate, or 3.35× the observed 128-GPU single trajectory—not 21.9 GC/s.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12962-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12963-VERDICT: FIX-FIRST
--
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12996-8. “Floor confirmed” should be **PLAUSIBLE/consistent with the floor**, not CONFIRMED: one below-floor point anti-scales, with no replication and known shape/codegen variation.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12997-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:12998:9. The ensemble script masks failures: each background `run_arm` changes `rc` only in its subshell, and `srun ... || { rc=1; }` returns success. Missing/corrupt JSON is silently skipped, so the job can print `DONE rc=0`. [The script](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/scripts/tmp/mpas_s9_ensemble.sbatch:44) must preserve each background PID/status and require all five valid receipts. It is also ignored by Git, so it is not reproducibly reviewable.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-12999-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-13000-10. GPU binding is otherwise sound: `--gpu-bind=none` leaves all GPUs visible and JAX’s Slurm auto-detection selects by local rank. Full-GPU `--exact` requests should force disjoint eight-node steps. But log each `SLURM_STEP_NODELIST` and coordinator host; that is the evidence the shared JAX job-ID port is safe.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-13001-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:13002:11. The ensemble does not synchronize measurement windows, pair each replica topology with a solo control, or counterbalance solo-first ordering. It can measure different node/fabric groups or non-overlapping windows rather than contention. Twelve steps give nine dependent steady samples per launch, not enough independent evidence for a 10% confirmation bar.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-13003-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md:13004:12. Ensemble arithmetic needs tighter wording: if solo reproduces 5.47 GC/s, ideal aggregate is 21.88 GC/s. The 1.10 threshold guarantees only 19.89 GC/s = 3.636× the 32-GPU solo rate, or 3.35× the observed 128-GPU single trajectory—not 21.9 GC/s.
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-13005-
.physics-validator/scaling_campaign/codex_recovery_review_2026-08-02.md-13006-VERDICT: FIX-FIRST
--
.physics-validator/scaling_campaign/codex_scaleout_improvements_2026-07-31.md-4404-| Rank | Improvement, mechanism, likely payoff | Cheapest confirm/refute | Applies to | Differentiability / conservation |
.physics-validator/scaling_campaign/codex_scaleout_improvements_2026-07-31.md-4405-|---|---|---|---|---|
.physics-validator/scaling_campaign/codex_scaleout_improvements_2026-07-31.md:4406:| 1 | **Use replica/ensemble parallelism once the spatial floor is reached.** Keep each member at a viable spatial tile; assign extra GPUs/nodes to independent members, parameter perturbations, or adjoint batches. This is the only credible way to regain near-linear *useful throughput* past the tile floor; it does **not** speed one forecast. | Run 1, 4, and 16 independent production-tile members concurrently; report aggregate SYPD, not single-member step time. Require ≥85–90% aggregate efficiency before adopting. | All grids, both lanes; especially spectral, whose honest multi-GPU route is replicas. | Safe: `vmap`/independent JITs preserve gradients and conservation per member. |
.physics-validator/scaling_campaign/codex_scaleout_improvements_2026-07-31.md-4407-| 2 | **MPAS hierarchical low-cut partition + hardware-aware rank mapping.** Install/use METIS, partition first into node-sized groups, then ranks/GPUs within each node; optimize weighted inter-node cut, not merely balance. This directly attacks the MPAS ocean 1.65× rank-count penalty. The absolute ceiling is a 39% time cut; demand a ≥10% full-step win to justify it. | At 32/128 ranks and fixed 5120 cells/rank, compare current/geometric/METIS partitions using: (a) edge cut and distinct remote-peer count, (b) halo-only 1,000-iteration timing, then (c) full in-step-halo-correct ocean step. Also permute rank-to-node assignment without changing the partition—this separates partition quality from placement. | MPAS/Voronoi atmosphere and ocean; CPU MPI and GPU SPMD. | Reordering only; expect roundoff changes in reductions, not conservation loss. Retain serial/owned-cell parity and gradient gates. |
.physics-validator/scaling_campaign/codex_scaleout_improvements_2026-07-31.md-4408-| 3 | **Topology-preserving placement for every finite-volume grid.** Map adjacent tiles to the same 4-GPU NVLink node first; minimize IB-cut edges, then respect CPU NUMA/HCA affinity. With 17.8 µs NVLink versus 26.3 µs IB, cube’s 44% halo phase has a realistic ~5–10% total-step opportunity; ~14% is a generous upper bound. | Change only `CUDA_VISIBLE_DEVICES`/Slurm rank ordering and the global device ordering. Hold HLO, tile shape, and partition fixed; time the halo-only path and full step. Reject if the best mapping is <3%. | Cube, lat-lon, MPAS; GPU first, CPU MPI as a separate NUMA/HCA placement sweep. | No numerical risk. |
--
.physics-validator/scaling_campaign/codex_scaleout_improvements_2026-07-31.md-4424-| Rank | Improvement, mechanism, likely payoff | Cheapest confirm/refute | Applies to | Differentiability / conservation |
.physics-validator/scaling_campaign/codex_scaleout_improvements_2026-07-31.md-4425-|---|---|---|---|---|
.physics-validator/scaling_campaign/codex_scaleout_improvements_2026-07-31.md:4426:| 1 | **Use replica/ensemble parallelism once the spatial floor is reached.** Keep each member at a viable spatial tile; assign extra GPUs/nodes to independent members, parameter perturbations, or adjoint batches. This is the only credible way to regain near-linear *useful throughput* past the tile floor; it does **not** speed one forecast. | Run 1, 4, and 16 independent production-tile members concurrently; report aggregate SYPD, not single-member step time. Require ≥85–90% aggregate efficiency before adopting. | All grids, both lanes; especially spectral, whose honest multi-GPU route is replicas. | Safe: `vmap`/independent JITs preserve gradients and conservation per member. |
.physics-validator/scaling_campaign/codex_scaleout_improvements_2026-07-31.md-4427-| 2 | **MPAS hierarchical low-cut partition + hardware-aware rank mapping.** Install/use METIS, partition first into node-sized groups, then ranks/GPUs within each node; optimize weighted inter-node cut, not merely balance. This directly attacks the MPAS ocean 1.65× rank-count penalty. The absolute ceiling is a 39% time cut; demand a ≥10% full-step win to justify it. | At 32/128 ranks and fixed 5120 cells/rank, compare current/geometric/METIS partitions using: (a) edge cut and distinct remote-peer count, (b) halo-only 1,000-iteration timing, then (c) full in-step-halo-correct ocean step. Also permute rank-to-node assignment without changing the partition—this separates partition quality from placement. | MPAS/Voronoi atmosphere and ocean; CPU MPI and GPU SPMD. | Reordering only; expect roundoff changes in reductions, not conservation loss. Retain serial/owned-cell parity and gradient gates. |
.physics-validator/scaling_campaign/codex_scaleout_improvements_2026-07-31.md-4428-| 3 | **Topology-preserving placement for every finite-volume grid.** Map adjacent tiles to the same 4-GPU NVLink node first; minimize IB-cut edges, then respect CPU NUMA/HCA affinity. With 17.8 µs NVLink versus 26.3 µs IB, cube’s 44% halo phase has a realistic ~5–10% total-step opportunity; ~14% is a generous upper bound. | Change only `CUDA_VISIBLE_DEVICES`/Slurm rank ordering and the global device ordering. Hold HLO, tile shape, and partition fixed; time the halo-only path and full step. Reject if the best mapping is <3%. | Cube, lat-lon, MPAS; GPU first, CPU MPI as a separate NUMA/HCA placement sweep. | No numerical risk. |
--- artifact inventory ---
cpu_ll2d.26628073.log
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch
scripts/tmp/mpas_s9_ensemble.sbatch
--- full cpu script, pared receipt fields ---
     1	#!/bin/bash -l
     2	#SBATCH --job-name=cpu_ll2d
     3	#SBATCH --account=bb1596
     4	#SBATCH --partition=compute
     5	#SBATCH --nodes=16
     6	#SBATCH --exclusive
     7	#SBATCH --time=04:00:00
     8	#SBATCH --output=cpu_ll2d.%j.log
     9	# HUNDREDS-OF-CPUS lat-lon atmosphere (user directive 2026-08-02): the
    10	# 2-D pencil path (wall poles, labelled throughput lane — NOT the pole
    11	# fold; pole-matched A/B remains open) at r512 (512x1024 = 524k cols) so
    12	# np512 still holds ~1k cols/rank.  Self-contained strong ladder
    13	# np64->512, f64 moist, 32 rpn, block:cyclic per the placement receipt.
    14	set -uo pipefail
    15	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    16	export JAX_PLATFORMS=cpu
    17	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    18	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv-mpi/bin/python}"
    19	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    20	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    21	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/cpu_ll2d_j${SLURM_JOB_ID}}"
    22	mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
    23	rc=0
    24	for NP in 64 128 256 512; do
    25	  NODES=$(( NP / 32 )); [ "$NODES" -lt 1 ] && NODES=1
    26	  echo "--- atm latlon 2-D pencil r512 f64 np=$NP ---"
    27	  JAX_ENABLE_X64=1 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=32 \
    28	      --distribution=block:cyclic --cpu-bind=cores --kill-on-bad-exit=1 \
    29	    "$PY" scripts/bench/run_cpu_mpi_scaling.py \
    30	      --grid latlon --latlon-2d --mode single --physics moist \
    31	      --precision float64 --n-levels 26 --resolution 512 \
    32	      --n-warmup 2 --n-timing 6 \
    33	      --output-dir "$OUTDIR/np${NP}" < /dev/null || { echo "np$NP FAILED"; rc=1; }
    34	done
    35	echo "=== RESULTS ==="
    36	for NP in 64 128 256 512; do
    37	  F=$(find "$OUTDIR/np${NP}" -name "*.json" 2>/dev/null | head -1)
    38	  "$PY" -c "
    39	import json,math,sys
    40	try:
    41	    d=json.load(open('$F' or '/nonexistent'))
    42	    r=d['results'][-1] if isinstance(d.get('results'),list) else d
    43	    ms=r.get('ms_per_step') or r.get('time_per_step_ms')
    44	    assert ms is not None and math.isfinite(ms) and ms > 0
    45	except Exception as e:
    46	    print('np$NP: MISSING/INVALID ->', e); sys.exit(1)
    47	print(f'np$NP: {ms:9.2f} ms')" || { echo "np$NP receipt invalid"; rc=1; }
    48	done
    49	echo "DONE rc=$rc"; exit $rc
5156:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5157:  mpi4jax, MPI = require_mpi_stack()
5158:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5159:  mpi4jax, MPI = require_mpi_stack()
5160:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5161:  mpi4jax, MPI = require_mpi_stack()
5162:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5163:  mpi4jax, MPI = require_mpi_stack()
5164:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5165:  mpi4jax, MPI = require_mpi_stack()
5166:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5167:  mpi4jax, MPI = require_mpi_stack()
5168:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5169:  mpi4jax, MPI = require_mpi_stack()
5170:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5171:  mpi4jax, MPI = require_mpi_stack()
5172:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5173:  mpi4jax, MPI = require_mpi_stack()
5174:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5175:  mpi4jax, MPI = require_mpi_stack()
5176:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5177:  mpi4jax, MPI = require_mpi_stack()
5178:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5179:  mpi4jax, MPI = require_mpi_stack()
5180:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5181:  mpi4jax, MPI = require_mpi_stack()
5182:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5183:  mpi4jax, MPI = require_mpi_stack()
5184:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5185:  mpi4jax, MPI = require_mpi_stack()
5186:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5187:  mpi4jax, MPI = require_mpi_stack()
5188:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5189:  mpi4jax, MPI = require_mpi_stack()
5190:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5191:  mpi4jax, MPI = require_mpi_stack()
5192:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5193:  mpi4jax, MPI = require_mpi_stack()
5194:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5195:  mpi4jax, MPI = require_mpi_stack()
5196:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5197:  mpi4jax, MPI = require_mpi_stack()
5198:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5199:  mpi4jax, MPI = require_mpi_stack()
5200:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5201:  mpi4jax, MPI = require_mpi_stack()
5202:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5203:  mpi4jax, MPI = require_mpi_stack()
5204:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5205:  mpi4jax, MPI = require_mpi_stack()
5206:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5207:  mpi4jax, MPI = require_mpi_stack()
5208:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5209:  mpi4jax, MPI = require_mpi_stack()
5210:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5211:  mpi4jax, MPI = require_mpi_stack()
5212:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5213:  mpi4jax, MPI = require_mpi_stack()
5214:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5215:  mpi4jax, MPI = require_mpi_stack()
5216:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5217:  mpi4jax, MPI = require_mpi_stack()
5218:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5219:  mpi4jax, MPI = require_mpi_stack()
5220:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5221:  mpi4jax, MPI = require_mpi_stack()
5222:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5223:  mpi4jax, MPI = require_mpi_stack()
5224:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5225:  mpi4jax, MPI = require_mpi_stack()
5226:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5227:  mpi4jax, MPI = require_mpi_stack()
5228:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5229:  mpi4jax, MPI = require_mpi_stack()
5230:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5231:  mpi4jax, MPI = require_mpi_stack()
5232:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5233:  mpi4jax, MPI = require_mpi_stack()
5234:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5235:  mpi4jax, MPI = require_mpi_stack()
5236:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5237:  mpi4jax, MPI = require_mpi_stack()
5238:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5239:  mpi4jax, MPI = require_mpi_stack()
5240:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5241:  mpi4jax, MPI = require_mpi_stack()
5242:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5243:  mpi4jax, MPI = require_mpi_stack()
5244:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5245:  mpi4jax, MPI = require_mpi_stack()
5246:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5247:  mpi4jax, MPI = require_mpi_stack()
5248:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5249:  mpi4jax, MPI = require_mpi_stack()
5250:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5251:  mpi4jax, MPI = require_mpi_stack()
5252:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5253:  mpi4jax, MPI = require_mpi_stack()
5254:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5255:  mpi4jax, MPI = require_mpi_stack()
5256:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5257:  mpi4jax, MPI = require_mpi_stack()
5258:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5259:  mpi4jax, MPI = require_mpi_stack()
5260:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5261:  mpi4jax, MPI = require_mpi_stack()
5262:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5263:  mpi4jax, MPI = require_mpi_stack()
5264:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5265:  mpi4jax, MPI = require_mpi_stack()
5266:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5267:  mpi4jax, MPI = require_mpi_stack()
5268:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5269:  mpi4jax, MPI = require_mpi_stack()
5270:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5271:  mpi4jax, MPI = require_mpi_stack()
5272:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5273:  mpi4jax, MPI = require_mpi_stack()
5274:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5275:  mpi4jax, MPI = require_mpi_stack()
5276:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5277:  mpi4jax, MPI = require_mpi_stack()
5278:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5279:  mpi4jax, MPI = require_mpi_stack()
5280:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5281:  mpi4jax, MPI = require_mpi_stack()
5282:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5283:  mpi4jax, MPI = require_mpi_stack()
5284:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5285:  mpi4jax, MPI = require_mpi_stack()
5286:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5287:  mpi4jax, MPI = require_mpi_stack()
5288:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5289:  mpi4jax, MPI = require_mpi_stack()
5290:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5291:  mpi4jax, MPI = require_mpi_stack()
5292:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5293:  mpi4jax, MPI = require_mpi_stack()
5294:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5295:  mpi4jax, MPI = require_mpi_stack()
5296:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5297:  mpi4jax, MPI = require_mpi_stack()
5298:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5299:  mpi4jax, MPI = require_mpi_stack()
5300:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5301:  mpi4jax, MPI = require_mpi_stack()
5302:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5303:  mpi4jax, MPI = require_mpi_stack()
5304:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5305:  mpi4jax, MPI = require_mpi_stack()
5306:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5307:  mpi4jax, MPI = require_mpi_stack()
5308:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5309:  mpi4jax, MPI = require_mpi_stack()
5310:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5311:  mpi4jax, MPI = require_mpi_stack()
5312:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
5313:  mpi4jax, MPI = require_mpi_stack()
5830:=== RESULTS ===
5835:DONE rc=0
--- JSONL-like lines ---
--- current script syntax ---
bash -n: PASS
--- relevant bench lines ---
   740	        platform=jax.default_backend(),
   741	        compile_ms=round(per_step_ms[0], 1),
   742	        steady_median_ms=(round(fused_step_ms, 4)
   743	                          if fused_step_ms is not None else None),
   744	        step_latency_gate_loop_ms=round(gate_loop_med, 2),
   745	        step_latency_gate_loop_min_ms=round(float(np.min(steady)), 2),
   746	        per_step_ms=[round(x, 1) for x in per_step_ms],
   747	        cells=int(mesh.nCells) * args.nlev,
   748	        # HORIZONTAL cells/rank — the same unit as --cells-per-rank, so a
   749	        # weak-mode row is comparable to its target (codex: the 3-D count
   750	        # made rows look nlev-x larger).
   751	        cells_per_rank_achieved=int(mesh.nCells) // n_ranks,
   752	        # Fix 4 honesty flag: at np=1 the parity reference pre-runs the
   753	        # SAME shape before the timed loop, so per_step_ms[0] may not
   754	        # contain the real JIT compile.
   755	        compile_prewarmed_by_parity_ref=bool(
   756	            args.parity_gate and n_ranks == 1),
   757	        # --- M1 lane fields (scaling-M3d increment-1) ---
   758	        barotropic_solver=args.barotropic_solver,

exec
/usr/bin/bash -lc "printf '%s\\n' '--- cpu script ---'
sed -n '20,100p' scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch
printf '%s\\n' '--- cpu receipt end ---'
sed -n '5790,5845p' cpu_ll2d.26628073.log
printf '%s\\n' '--- receipt warnings/count ---'
rg -c 'Detected versions outside legoESM' cpu_ll2d.26628073.log; rg -n -m 2 -C 2 'Detected versions outside legoESM' cpu_ll2d.26628073.log
printf '%s\\n' '--- lines with result values ---'
rg -n '297\\.57|161\\.03|72\\.06|44\\.78|n_ranks|process_count|decomposition' cpu_ll2d.26628073.log | tail -n 80
printf '%s\\n' '--- ensemble current doc only ---'
rg -n -i -C 3 'ensemble|counterbalanced|fabric-contention|co-execution|26628196' docs/performance/scaling/levante_campaign_2026-07-24.md
printf '%s\\n' '--- all matching artifact names only ---'
rg --files -uu | rg '(26628196|26628021|mpas_s9_ens|cpu_ll2d|26628073)'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 437ms:
--- cpu script ---
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/cpu_ll2d_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
rc=0
for NP in 64 128 256 512; do
  NODES=$(( NP / 32 )); [ "$NODES" -lt 1 ] && NODES=1
  echo "--- atm latlon 2-D pencil r512 f64 np=$NP ---"
  JAX_ENABLE_X64=1 srun --nodes="$NODES" --ntasks="$NP" --ntasks-per-node=32 \
      --distribution=block:cyclic --cpu-bind=cores --kill-on-bad-exit=1 \
    "$PY" scripts/bench/run_cpu_mpi_scaling.py \
      --grid latlon --latlon-2d --mode single --physics moist \
      --precision float64 --n-levels 26 --resolution 512 \
      --n-warmup 2 --n-timing 6 \
      --output-dir "$OUTDIR/np${NP}" < /dev/null || { echo "np$NP FAILED"; rc=1; }
done
echo "=== RESULTS ==="
for NP in 64 128 256 512; do
  F=$(find "$OUTDIR/np${NP}" -name "*.json" 2>/dev/null | head -1)
  "$PY" -c "
import json,math,sys
try:
    d=json.load(open('$F' or '/nonexistent'))
    r=d['results'][-1] if isinstance(d.get('results'),list) else d
    ms=r.get('ms_per_step') or r.get('time_per_step_ms')
    assert ms is not None and math.isfinite(ms) and ms > 0
except Exception as e:
    print('np$NP: MISSING/INVALID ->', e); sys.exit(1)
print(f'np$NP: {ms:9.2f} ms')" || { echo "np$NP receipt invalid"; rc=1; }
done
echo "DONE rc=$rc"; exit $rc
--- cpu receipt end ---
[1785669029.448027] [l30529:240645:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.377179] [l30525:450231:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.455262] [l30518:384674:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669030.329959] [l30543:524802:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.376110] [l30525:450220:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669030.311478] [l30541:338913:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669030.303211] [l30541:338899:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.473466] [l30534:376832:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.488325] [l30538:415181:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.420698] [l30524:108211:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.498592] [l30538:415188:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.497169] [l30538:415209:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.505023] [l30538:415195:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.409475] [l30517:783146:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669030.355635] [l30543:524817:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.411921] [l30517:783166:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.450507] [l30529:240650:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.521569] [l30531:281680:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.376014] [l30516:326369:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.369624] [l30516:326388:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.449847] [l30527:167417:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.386486] [l30525:450240:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.390177] [l30525:450222:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.363628] [l30516:326371:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.450751] [l30529:240636:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.498502] [l30538:415210:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669030.301858] [l30541:338906:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.396733] [l30524:108216:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.459099] [l30526:300908:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.434060] [l30527:167427:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.433343] [l30530:217135:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.458966] [l30526:300917:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669030.318048] [l30543:524819:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.524379] [l30531:281700:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.429398] [l30530:217112:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.413795] [l30517:783151:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
[1785669029.433460] [l30530:217116:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
          latlon |        moist |   512 ranks | 512   | L26 | dt=     0 |     44.78 ms/step | SYPD=   0.014 |     304.4 Mcells/s
  Result: /scratch/b/b381103/legoesm_scaling/cpu_ll2d_j26628073/np512/latlon_moist_single/latlon_2d_moist_single_r512_n512_float64.json
[1785669029.363467] [l30516:326365:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
=== RESULTS ===
np64:    297.57 ms
np128:    161.03 ms
np256:     72.06 ms
np512:     44.78 ms
DONE rc=0

********************************************************************************
*                                                                              *
*  This is the automated job summary provided by DKRZ.                         *
*  If you encounter problems, need assistance or have any suggestion, please   *
*  write an email to                                                           *
*                                                                              *
*  --  support@dkrz.de --                                                      *
*                                                                              *
*                       We hope you enjoyed the DKRZ supercomputer LEVANTE ... *
--- receipt warnings/count ---
1920
65-[l30516.lvt.dkrz.de:325531] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
66-[l30517.lvt.dkrz.de:782343] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
67:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
68-  mpi4jax, MPI = require_mpi_stack()
69:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
70-  mpi4jax, MPI = require_mpi_stack()
71:/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
--- lines with result values ---
336:    Timing: 297.57 ms/step | SYPD=0.002 | 45.8 Mcells/s
400:          latlon |        moist |    64 ranks | 512   | L26 | dt=     0 |    297.57 ms/step | SYPD=   0.002 |      45.8 Mcells/s
1057:    Timing: 161.03 ms/step | SYPD=0.004 | 84.7 Mcells/s
1185:          latlon |        moist |   128 ranks | 512   | L26 | dt=     0 |    161.03 ms/step | SYPD=   0.004 |      84.7 Mcells/s
2482:    Timing: 72.06 ms/step | SYPD=0.009 | 189.2 Mcells/s
2738:          latlon |        moist |   256 ranks | 512   | L26 | dt=     0 |     72.06 ms/step | SYPD=   0.009 |     189.2 Mcells/s
5315:    Timing: 44.78 ms/step | SYPD=0.014 | 304.4 Mcells/s
5827:          latlon |        moist |   512 ranks | 512   | L26 | dt=     0 |     44.78 ms/step | SYPD=   0.014 |     304.4 Mcells/s
5831:np64:    297.57 ms
5832:np128:    161.03 ms
5833:np256:     72.06 ms
5834:np512:     44.78 ms
--- ensemble current doc only ---
1673-
1674-### Next receipts submitted 2026-08-02
1675-
1676:1. **Ensemble receipt** (`scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch`)
1677-   — codex lever #1: 4 concurrent 32-GPU s9 replicas on disjoint 8-node
1678-   sets vs SAME-JOB solo controls bracketing phase B (solo before AND
1679:   after, ordering counterbalanced). steps=5000 so the stepping window
1680-   (~60 s) dwarfs launch skew; per-arm `SLURM_STEP_NODELIST` +
1681-   wall-clock brackets logged as overlap evidence. CONFIRM bar:
1682-   max(replica) <= 1.10x mean(solo) => guaranteed aggregate >= 3.64x the
1683-   32-GPU solo rate (>= 19.9 GC/s if solo reproduces 5.47) = ~3.3x the
1684-   observed 128-GPU single-trajectory rate. REFUTE: replica slowdown
1685:   >10 % = the fabric-contention term, quantified per replica.
1686-2. **s8 lloyd=0 matched rerun** — de-confounds the weak pair: np8/16/32
1687-   (81.9k/41.0k/20.5k cells/GPU) on the SAME lloyd=0 family, same sfc +
1688-   `--reorder-for 128`, same steps/warmup as the s9 ladder. Weak pairs
--
1716-
1717-| job | what | devices | why |
1718-|---|---|---|---|
1719:| 26628021 | s9 ensemble contention | 128 GPU (4x32) | lever #1 receipt |
1720-| 26628071 | oc LL2304 retry post-#1370 | 128 GPU | pre-fix failure was resident-args; predicted PASS at ~0.10 GB/dev residency |
1721-| 26628072 | atm LL2304 @96/@192 + LL2880 @192 | 96-192 GPU | LL2048 does not divide 192; LL2880@192 = 86.4k cols/GPU ABOVE floor |
1722-| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
--- all matching artifact names only ---
cpu_ll2d.26628073.log
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch
scripts/tmp/mpas_s9_ensemble.sbatch

codex
The first six corrections are present in the reviewed sources. I found two remaining evidence-quality checks to resolve before I can give the release verdict: the campaign prose still has an obsolete ensemble-attribution sentence, and the new CPU log reports MPI/UCX compatibility warnings that may affect how strongly that receipt can be stated.
exec
/usr/bin/bash -lc "printf '%s\\n' '--- CPU driver’s distributed metadata ---'
rg -n -C 4 'process_count|n_ranks|decomposition|world.*size|MPI.*size|mpi4jax|metadata' scripts/bench/run_cpu_mpi_scaling.py | sed -n '1,280p'
printf '%s\\n' '--- MPI compatibility gate ---'
nl -ba packages/core/legoesm/parallel/reductions.py | sed -n '380,440p'
printf '%s\\n' '--- CPU log scheduler and version context ---'
sed -n '1,120p' cpu_ll2d.26628073.log
printf '%s\\n' '--- step resource semantics available locally ---'
(srun --help 2>&1 | rg -n -i -C 2 'exact|exclusive|gpus-per-node|overlap' | head -n 180) || true
printf '%s\\n' '--- existing multistep patterns ---'
rg -n -C 2 -- '--exact|--exclusive' scripts/cluster/scaling_levante -g '*.sbatch' | head -n 260" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 572ms:
--- CPU driver’s distributed metadata ---
7-MPI-scalable grids (multi-rank weak/strong scaling, genuinely
8-domain-decomposed at the dycore level):
9-  icosahedral   -- MPAS Voronoi TRiSK PE dycore (cell partition)
10-  latlon        -- Lat-lon C-grid FV PE dycore (latitude-band
11:                   decomposition via ``make_latlon_mpi_step``;
12-                   needs >=2 lat rows per rank for the halo=2
13-                   PPM/biharmonic exchanges)
14-
15-Single-rank only (listed but their MPI paths are not domain-decomposed
--
61-from datetime import datetime, timezone
62-from pathlib import Path
63-from typing import Any
64-
65:# Shared, self-describing scaling metadata (roadmap item 9): merged into every
66-# result JSON so a host-staged / f32 / replicated run is falsifiable from the
67:# record.  ``metadata.py`` imports JAX only lazily, so importing it here does
68-# NOT trigger early JAX init before ``_configure_jax_cpu``.
69-_BENCH_DIR = Path(__file__).resolve().parent
70-if str(_BENCH_DIR) not in sys.path:
71-    sys.path.insert(0, str(_BENCH_DIR))
72:from metadata import annotate_incomplete, scaling_metadata  # noqa: E402
73-
74-# NOTE: do NOT import ``legoesm.constants`` at module load — it eagerly
75-# imports ``jax.numpy``, which initialises JAX before ``_configure_jax_cpu``
76-# has a chance to set ``JAX_ENABLE_X64`` / ``JAX_PLATFORMS`` / thread flags.
--
111-
112-def _configure_jax_gpu(precision: str) -> None:
113-    """Pin THIS MPI rank to one local GPU and run JAX on cuda (route-A).
114-
115:    Single-node multi-GPU via mpi4jax (the SAME mpi4jax halo machinery as the
116-    CPU path — make_latlon_mpi_step / cube — just on cuda devices over the
117-    PCIe pair).  Must run BEFORE any JAX import.  Local rank from the launcher
118-    env (OpenMPI / SLURM).  Mirrors the ocean harness ``_configure_jax_gpu``
119-    (bench_ocean_mpi_scaling.py) so the atm lat-lon dycore gets a 2-GPU
120-    number via the proven overlay-venv route-A (cuda jax + CUDA-built
121:    mpi4jax)."""
122-    # RESPECT an EXPLICIT CUDA_VISIBLE_DEVICES (set by the launcher's per-task
123-    # binding, or deliberately e.g. "0,1" for a single-process MULTI-GPU SPMD
124-    # run): re-deriving it from the local rank would either DOUBLE-restrict a
125-    # per-task binding (hiding the bound GPU -> "no supported devices for CUDA")
--
150-    # Two ranks share the node; do not let the allocator grab the whole GPU.
151-    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
152-
153-
154:def _launcher_world_size() -> int:
155-    """World size the MPI/SLURM/PALS launcher env reports (1 = no launcher).
156-
157-    GLOBAL sizes only (codex 2026-07-24 round-3): PALS_LOCAL_SIZE is
158:    PER-NODE — using it as world size would accept a one-node partial
159-    federation as complete. PALS jobs expose no global size env here, so
160-    they fall through to 1 and rely on the mpi4py path. Prefer the STEP
161-    task count over the allocation's SLURM_NTASKS so an `srun -n1` inside
162-    a larger allocation is not mistaken for the allocation-wide count.
--
175-    Size vars alone under-detect Cray PALS (PALS_LOCAL_SIZE is per-node;
176-    a job can expose only per-rank ids) — so the PRESENCE of a per-rank id
177-    counts as launcher evidence too (codex round-2).
178-    """
179:    if _launcher_world_size() > 1:
180-        return True
181-    return any(
182-        v in os.environ
183-        for v in ("PALS_RANKID", "PMI_RANK", "OMPI_COMM_WORLD_RANK")
184-    )
185-
186-
187-def _init_mpi() -> tuple[int, int]:
188:    """Initialize MPI and return (rank, n_ranks)."""
189-    try:
190-        from mpi4py import MPI
191-        comm = MPI.COMM_WORLD
192-        return comm.Get_rank(), comm.Get_size()
--
194-        # RuntimeError: mpi4py installed but no loadable libmpi (common in
195-        # a GPU-only venv). A single-process run must not require MPI —
196-        # BUT under a real MPI launcher a broken mpi4py must fail LOUDLY
197-        # here, or every rank silently runs duplicated serial work
198:        # reporting n_ranks=1 (codex finding).
199-        if _under_mpi_launcher():
200-            raise RuntimeError(
201:                f"MPI launcher detected (world size "
202:                f"{_launcher_world_size()}) but mpi4py is unusable: {e}"
203-            ) from e
204-        return 0, 1
205-
206-
--
209-# ===========================================================================
210-
211-@dataclass
212-class TimingResult:
213:    n_ranks: int
214-    resolution: int
215-    n_levels: int
216-    precision: str
217-    mode: str
--
228-    total_cells: int
229-    cells_per_rank: int
230-    mcells_per_s: float
231-    scaling_efficiency: float = 1.0
232:    # Lat-lon decomposition: "band" (1-D latitude band, default) or "2d"
233-    # (proc_lat x proc_lon pencil).  Lets the collector/plotter separate the
234-    # 2-D-pencil curve from the 1-D band laggard.  N/A for other grids ("band"
235-    # is a harmless default they never key on).
236:    decomposition: str = "band"
237-    # MPAS/Voronoi partition-quality telemetry (roadmap #6): edge-cut /
238-    # owned-halo-ratio / cells-per-rank min-max / message count.  Populated only
239-    # for the multi-rank icosahedral path; None otherwise.
240-    partition_metrics: dict | None = None
--
327-STRONG_RES_ICO = [4, 5, 6, 7, 8]  # levels 4-8 (L8 = 655,362 cells, ~25 km)
328-STRONG_RES_SP = [21, 42]
329-
330-
331:def _weak_resolution_cs(n_ranks: int, base_n: int = WEAK_BASE_CS) -> int:
332:    n_raw = base_n * math.sqrt(n_ranks)
333-    return max(4, 2 * round(n_raw / 2))
334-
335-
336:def _weak_resolution_ll(n_ranks: int, base_n: int = WEAK_BASE_LL) -> int:
337:    """Weak-scaling n_lat for the lat-lon band decomposition.
338-
339-    Constant *cells per rank* (the icosahedral analog): total cells
340-    scale as ``n_lat * n_lon = 2 * n_lat**2``, so ``n_lat ~
341:    sqrt(n_ranks)`` keeps cells/rank fixed.  Constraints layered on
342-    top:
343-
344:    * divisible by ``n_ranks`` (uniform bands → clean cells/rank),
345-    * at least 2 lat rows per rank — ``pad_halo_latlon_mpi`` raises
346-      when ``halo(=2 for PPM/biharmonic) > n_lat_local``, so a band
347-      must never be thinner than the deepest operator halo.
348-
--
355-    the same actual-cells normalization the icosahedral weak path
356-    needs for its discrete 4x subdivision-level jumps (see
357-    ``run_levante_gpu_scaling.run_weak_scaling``).
358-    """
359:    n_raw = base_n * math.sqrt(n_ranks)
360-    n_rounded = max(8, 2 * round(n_raw / 2))
361:    while n_rounded % n_ranks != 0 or n_rounded < 2 * n_ranks:
362-        n_rounded += 2
363-    return n_rounded
364-
365-
366:def _weak_resolution_ico(n_ranks: int, base_level: int = WEAK_BASE_ICO) -> int:
367-    base_cells = 10 * 4 ** base_level + 2
368-    best_level = base_level
369-    best_ratio = float("inf")
370-    for lev in range(base_level, 9):
371-        cells = 10 * 4 ** lev + 2
372:        cells_per_rank = cells / n_ranks
373-        ratio = max(cells_per_rank / base_cells, base_cells / cells_per_rank)
374-        if ratio < best_ratio or (ratio == best_ratio and cells_per_rank >= base_cells):
375-            best_ratio = ratio
376-            best_level = lev
--
452-    grid_type: str,
453-    resolution: int,
454-    nlev: int,
455-    rank: int,
456:    n_ranks: int,
457-    precision: str,
458-    physics_level: str,
459-    dt: float | None = None,
460-    cs_spmd: bool = False,
--
487-            out = _build_cubed_sphere_spmd(
488-                resolution, nlev, dt, dtype, physics_level, _cast)
489-        else:
490-            out = _build_cubedsphere(resolution, nlev, sigma, dt, dtype, rank,
491:                                     n_ranks, physics_level, _cast)
492-    elif grid_type == "latlon":
493:        out = _build_latlon(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
494-                            physics_level, _cast, latlon_2d=latlon_2d)
495-    elif grid_type == "icosahedral":
496-        out = _build_icosahedral(resolution, nlev, sigma, dt, dtype, rank,
497:                                 n_ranks, physics_level, _cast)
498-    elif grid_type == "spectral":
499-        out = _build_spectral(resolution, nlev, sigma, dt, dtype,
500-                              physics_level, _cast)
501-    else:
--
505-    # metrics; the other builders return a 5-tuple, padded with None here.
506-    return out if len(out) == 6 else (*out, None)
507-
508-
509:def _build_cubedsphere(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
510-                        physics_level, cast_fn):
511-    import jax
512-    import jax.numpy as jnp
513-
--
543-    # requires the full shape; each rank steps all faces and MPI halo
544-    # exchange keeps owned faces correct.  The FV3 CD-grid dycore advects
545-    # q_v/q_c/q_r in advective form (consistent with T); its tracer halo
546-    # rides the same auto-dispatched pad_halo_4d (mpi/spmd/local).
547:    if n_ranks > 1:
548-        from legoesm.parallel.distributed import initialize_distributed
549-        initialize_distributed(global_n=resolution, grid_type="cubed_sphere")
550-
551-    if _moist:
--
560-        _phys = physics_fn
561-        step_fn = lambda state, dt: model.step(state, dt, physics_fn=_phys)
562-    else:
563-        step_fn = model.step
564:    cells_per_rank = total_cells // max(1, n_ranks)
565-    return step_fn, state, dt, total_cells, cells_per_rank
566-
567-
568-def _build_cubed_sphere_spmd(resolution, nlev, dt, dtype, physics_level,
569-                             cast_fn):
570:    """A1 path: TRUE cubed-sphere decomposition via jax.distributed.
571-
572-    Multi-controller SPMD: the global face mesh is built from
573-    ``jax.devices()`` (THE multi-controller fix — ``jax.local_devices``
574-    would give each process a private 1-device mesh), the existing
--
578-    SPMD-global by construction (parity receipt 6.7e-10 @5 steps, job
579-    8462928).  ``jax.distributed.initialize()`` must already have run
580-    (``main`` does it for ``--cs-spmd`` BEFORE any other JAX use).
581-
582:    mpi4jax is NEVER armed in this mode: the replicated cubed-sphere
583:    path's mpi4jax halo machinery and jax.distributed collectives in
584-    one program is the documented mixed-stack deadlock hazard.
585-    """
586-    import jax
587-
--
804-    cells_per_rank = total_cells // n_global
805-    return step_fn, state, dt, total_cells, cells_per_rank
806-
807-
808:def _factor_2d_latlon(n_ranks, n_lat, n_lon, min_lat=2, min_lon=2):
809:    """Factor ``n_ranks`` into ``(proc_lat, proc_lon)`` for the 2-D pencil,
810-    MINIMISING the per-rank halo perimeter ``n_lat/proc_lat + n_lon/proc_lon``
811-    (the whole point of 2-D vs the 1-D band).
812-
813-    Constraints: each block keeps ``>= min_lat`` latitude rows (the halo=2
--
818-    Returns the min-perimeter pair; ties broken toward the more balanced
819-    block (smaller ``|n_lat/pl - n_lon/pc|``).
820-    """
821-    best = None  # (perimeter, imbalance, proc_lat, proc_lon)
822:    for pl in range(1, n_ranks + 1):
823:        if n_ranks % pl:
824-            continue
825:        pc = n_ranks // pl
826-        blat, blon = n_lat // pl, n_lon // pc
827-        if blat < min_lat or blon < min_lon:
828-            continue
829-        perim = n_lat / pl + n_lon / pc
--
832-        if best is None or key < best[0]:
833-            best = (key, pl, pc)
834-    if best is None:
835-        raise ValueError(
836:            f"_factor_2d_latlon: no 2-D factorisation of n_ranks={n_ranks} "
837-            f"keeps >= {min_lat} lat rows AND >= {min_lon} lon cols per block "
838-            f"for n_lat={n_lat}, n_lon={n_lon}.  Reduce ranks or raise "
839-            f"resolution."
840-        )
841-    return best[1], best[2]
842-
843-
844:def _build_latlon(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
845-                   physics_level, cast_fn, latlon_2d=False):
846-    import jax
847-    import jax.numpy as jnp
848-
--
890-        physics_fn = make_kessler_forcing_latlon(dt)
891-    else:
892-        physics_fn = _build_physics_fn(physics_level, "latlon")
893-
894:    if n_ranks > 1 and latlon_2d:
--- MPI compatibility gate ---
   380	    import mpi4jax
   381	    from mpi4py import MPI
   382	    _validate_mpi_runtime_versions(
   383	        jax.__version__,
   384	        mpi4jax.__version__,
   385	        strict=_env_flag_true("LEGOESM_MPI_STRICT_COMPAT"),
   386	    )
   387	    check_mpi4jax_transport(mpi4jax)
   388	    return mpi4jax, MPI
   389	
   390	
   391	def mpi4jax_array_result(result):
   392	    """Return the array payload from mpi4jax return values.
   393	
   394	    mpi4jax<0.8 commonly returned ``(array, token)`` while mpi4jax>=0.8
   395	    returns the array directly with automatic token management.
   396	    """
   397	    if isinstance(result, tuple):
   398	        if not result:
   399	            raise RuntimeError("mpi4jax operation returned an empty tuple.")
   400	        return result[0]
   401	    return result
   402	
   403	
   404	def global_sum_mpi(local_value: jax.Array, comm=None) -> jax.Array:
   405	    """Compute a global sum across all MPI ranks.
   406	
   407	    **Differentiable**: uses ``allreduce(SUM)`` which has full JVP and
   408	    VJP support in mpi4jax.  Safe to use inside ``jax.grad``.
   409	
   410	    Parameters
   411	    ----------
   412	    local_value : jax.Array
   413	        Scalar (or array) local partial sum.
   414	    comm : mpi4py communicator, optional
   415	        Communicator to reduce over. Defaults to ``MPI.COMM_WORLD``. Callers on a
   416	        SUB-communicator (e.g. a plane-LES layout whose distributed FFT uses
   417	        ``layout.comm``) MUST pass that same communicator — otherwise the reduction
   418	        spans the wrong rank set and can deadlock or mix unrelated ranks.
   419	    """
   420	    mpi4jax, MPI = require_mpi_stack()
   421	    if comm is None:
   422	        comm = MPI.COMM_WORLD
   423	
   424	    with mpi_timer("global_sum_mpi"):
   425	        global_val = mpi4jax_array_result(
   426	            mpi4jax.allreduce(local_value, op=MPI.SUM, comm=comm),
   427	        )
   428	    return global_val
   429	
   430	
   431	def is_multi_process() -> bool:
   432	    """Whether a local partial sum must be MANUALLY all-reduced via mpi4jax.
   433	
   434	    True exactly when an mpi4jax-style decomposition is armed — the MPI halo
   435	    backend (``is_distributed()``) or a Voronoi partition layout — i.e. when
   436	    each rank holds a LOCAL shard as a plain per-process array.  Canonical
   437	    home (#177) for the predicate the ocean conservation fixers and the
   438	    eta-floor mass redistribution previously each re-implemented identically.
   439	
   440	    ``jax.process_count() > 1`` (a jax.distributed multi-controller run) is
--- CPU log scheduler and version context ---
outdir=/scratch/b/b381103/legoesm_scaling/cpu_ll2d_j26628073
--- atm latlon 2-D pencil r512 f64 np=64 ---
[l30517.lvt.dkrz.de:782347] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325538] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782354] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325539] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782345] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782362] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782340] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325541] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325543] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325542] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782350] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782341] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782355] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782353] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782344] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325544] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325533] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325540] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325525] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782363] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325546] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325547] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782366] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325545] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325535] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782337] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782338] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782348] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782358] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782360] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325548] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782361] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782365] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782367] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325536] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782339] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782342] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325537] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782359] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325526] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325518] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782357] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325530] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325529] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782336] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782346] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325527] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325534] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325549] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325520] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325522] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325521] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325532] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782356] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325519] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325523] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325528] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782351] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782352] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782349] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782364] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325524] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30516.lvt.dkrz.de:325531] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
[l30517.lvt.dkrz.de:782343] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:420: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
  mpi4jax, MPI = require_mpi_stack()
--- step resource semantics available locally ---
61-  -o, --output=out            location of stdout redirection
62-  -O, --overcommit            overcommit resources
63:      --overlap               Allow other steps to overlap this step
64-      --het-group=value       hetjob component allocation(s) in which to launch
65-                              application
--
117-
118-Consumable resources related options:
119:      --exact                 use only the resources requested for the step
120-                              (by default, all non-gres resources on each node
121-                              in the allocation will be used in the step)
122:      --exclusive[=user]      for job allocation, this allocates nodes in
123:                              in exclusive mode
124:                              for job steps, this is equivalent to --exact
125:      --exclusive[=mcs]       allocate nodes in exclusive mode when
126-                              cpu consumable resource is enabled
127:                              and mcs plugin is enabled (--exact implied)
128-                              or don't share CPUs for job steps
129-      --mem-per-cpu=MB        maximum amount of real memory per allocated
--
158-      --gpu-bind=...          task to gpu binding options
159-      --gpu-freq=...          frequency and voltage of GPUs
160:      --gpus-per-node=n       number of GPUs required per allocated node
161-      --gpus-per-socket=n     number of GPUs required per allocated socket
162-      --gpus-per-task=n       number of GPUs required per spawned task
--- existing multistep patterns ---
rg: -g: No such file or directory (os error 2)
rg: *.sbatch: No such file or directory (os error 2)
scripts/cluster/scaling_levante/diagnosis.sbatch-6-#SBATCH --ntasks-per-node=1
scripts/cluster/scaling_levante/diagnosis.sbatch-7-#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/diagnosis.sbatch:8:#SBATCH --exclusive
scripts/cluster/scaling_levante/diagnosis.sbatch-9-#SBATCH --time=01:30:00
scripts/cluster/scaling_levante/diagnosis.sbatch-10-#SBATCH --output=legoesm_diag.%j.log
--
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-6-#SBATCH --ntasks-per-node=4
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-7-#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:8:#SBATCH --exclusive
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-9-#SBATCH --time=02:00:00
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch-10-#SBATCH --output=legoesm_gpu_multinode.%j.log
--
scripts/cluster/scaling_levante/cpu_scaling.sbatch-4-#SBATCH --nodes=1
scripts/cluster/scaling_levante/cpu_scaling.sbatch-5-#SBATCH --ntasks-per-node=64
scripts/cluster/scaling_levante/cpu_scaling.sbatch:6:#SBATCH --exclusive
scripts/cluster/scaling_levante/cpu_scaling.sbatch-7-#SBATCH --time=04:00:00
scripts/cluster/scaling_levante/cpu_scaling.sbatch-8-#SBATCH --output=legoesm_cpu_scaling.%j.log
--
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-6-#SBATCH --nodes=48
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-7-#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch:8:#SBATCH --exclusive
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-9-#SBATCH --mem=0
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch-10-#SBATCH --time=02:00:00
--
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch-4-#SBATCH --partition=compute
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch-5-#SBATCH --nodes=16
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch:6:#SBATCH --exclusive
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch-7-#SBATCH --time=04:00:00
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch-8-#SBATCH --output=cpu_ll2d.%j.log
--
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-6-#SBATCH --nodes=8
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-7-#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch:8:#SBATCH --exclusive
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-9-#SBATCH --mem=0
scripts/cluster/scaling_levante/mpas_s8_lloyd0_ladder.sbatch-10-#SBATCH --time=01:30:00
--
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm-7-#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm-8-#SBATCH --cpus-per-task=16
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm:9:#SBATCH --exclusive
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm-10-#SBATCH --time=02:00:00
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm-11-#SBATCH --output=%x_%j.log
--
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-6-#SBATCH --nodes=32
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-7-#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:8:#SBATCH --exclusive
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-9-#SBATCH --mem=0
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-10-#SBATCH --time=01:30:00
--
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-64-  # ("$0" inside bash -c is $PY, passed as the first post-script arg).
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-65-  JAX_ENABLE_X64=0 srun --nodes=8 --ntasks=32 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:66:      --gpus-per-node=4 --gpu-bind=none --exact --kill-on-bad-exit=1 \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-67-      --job-name="arm_$1" \
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch-68-    bash -c '[ "${SLURM_PROCID:-1}" = 0 ] && echo "[step $ARM_TAG] nodelist=$SLURM_STEP_NODELIST"; exec "$0" "$@"' \
--
scripts/cluster/scaling_levante/gpu_scaling.sbatch-6-#SBATCH --ntasks-per-node=1
scripts/cluster/scaling_levante/gpu_scaling.sbatch-7-#SBATCH --gpus-per-node=4
scripts/cluster/scaling_levante/gpu_scaling.sbatch:8:#SBATCH --exclusive
scripts/cluster/scaling_levante/gpu_scaling.sbatch-9-#SBATCH --time=02:00:00
scripts/cluster/scaling_levante/gpu_scaling.sbatch-10-#SBATCH --output=legoesm_gpu_scaling.%j.log

exec
/usr/bin/bash -lc "printf '%s\\n' '--- run driver: timing record construction ---'
rg -n -C 5 'TimingResult\\(|n_ranks=n_ranks|scaling_metadata\\(|process_count' scripts/bench/run_cpu_mpi_scaling.py | tail -n 260
printf '%s\\n' '--- generic metadata semantics ---'
rg -n -C 5 'process_count|def scaling_metadata' scripts/bench/metadata.py | sed -n '1,260p'
printf '%s\\n' '--- exact arithmetic ---'
node - <<'NODE'
const t0=297.57;
for (const [n,t] of [[64,297.57],[128,161.03],[256,72.06],[512,44.78]]) {
  const speedup=t0/t;
  const eff=(t0*64)/(t*n);
  console.log(n, {speedup, eff, numerator:t0*64, denominator:t*n});
}
console.log('ll gc/s', 2048*4096*26/5.5767e-3/1e9, 2048*4096*26/9.6015e-3/1e9)
NODE
printf '%s\\n' '--- r512 factor selection test / source ---'
nl -ba scripts/bench/run_cpu_mpi_scaling.py | sed -n '808,850p;884,940p'
printf '%s\\n' '--- current diff and whitespace ---'
git diff --check HEAD -- docs/performance/scaling/levante_campaign_2026-07-24.md scripts/plot/plot_scaling_paper_figure.py scripts/cluster/scaling_levante/*.sbatch
git diff --stat HEAD -- docs/performance/scaling/levante_campaign_2026-07-24.md scripts/plot/plot_scaling_paper_figure.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 485ms:
--- run driver: timing record construction ---
1159-     part_metrics) = _build_amip_step(
1160-        grid_type=grid_type,
1161-        resolution=resolution,
1162-        nlev=nlev,
1163-        rank=rank,
1164:        n_ranks=n_ranks,
1165-        precision=precision,
1166-        physics_level=physics_level,
1167-        dt=dt,
1168-        cs_spmd=cs_spmd,
1169-        latlon_2d=latlon_2d,
--
1283-    # Record the device count so a single-process 2-GPU run lands as n=2 (not
1284-    # n=1) in the CSV + JSON filename. cells_per_rank already used n_global on
1285-    # this path; the multi-controller cs-spmd path has n_ranks == device_count
1286-    # (one process per device), so this is a no-op there.
1287-    _record_ndev = jax.device_count() if cs_spmd else n_ranks
1288:    return TimingResult(
1289-        n_ranks=_record_ndev,
1290-        resolution=resolution,
1291-        n_levels=nlev,
1292-        precision=precision,
1293-        mode=mode,
--
1385-
1386-    ``cs_spmd`` marks the single-controller cubed-sphere SPMD path, where
1387-    ``result.n_ranks`` was rewritten to ``jax.device_count()`` for the
1388-    filename / plot axis.  The metadata ``n_ranks`` (= process count) must NOT
1389-    use that rewritten value, so it is left to ``scaling_metadata`` to default
1390:    to ``jax.process_count()`` (1 for single-controller, N for multi-controller
1391-    SPMD) while the device count lives in ``n_gpus`` / ``device_count``.
1392-    """
1393-    output_dir.mkdir(parents=True, exist_ok=True)
1394-    # Tag a non-default (2-D) decomposition into the filename so a 2-D-pencil
1395-    # run never overwrites the band run at the same grid/res/np (the payload
--
1409-        import jax
1410-        payload["backend"] = jax.default_backend()
1411-    except Exception:
1412-        payload["backend"] = ""
1413-
1414:    def _live_process_count() -> int:
1415-        try:
1416-            import jax
1417-
1418:            return int(jax.process_count())
1419-        except Exception:
1420-            return 1
1421-    # Record the hybrid layout so scaling can be plotted vs CORES, not ranks:
1422-    # a hybrid 8r x 4c run and a packed 32r x 1c run both report n_ranks but use
1423-    # 32 vs 128 cores. cpus_per_task * n_ranks = the true resource count.
--
1439-    # comparable and a host-staged or f32 run is falsifiable from the record.
1440-    # cs-spmd: result.n_ranks is the DEVICE count (rewritten upstream); leave
1441-    # metadata n_ranks to auto process-count.  Non-cs-spmd (mpi4jax): jax is
1442-    # unaware of the MPI world, so the real MPI rank count must be passed.
1443-    _md_n_ranks = None if cs_spmd else result.n_ranks
1444:    payload["metadata"] = annotate_incomplete(scaling_metadata(
1445-        grid=result.grid_type,
1446-        component="atmosphere",
1447-        resolution=result.resolution,
1448-        n_levels=result.n_levels,
1449-        precision=result.precision,
1450-        n_ranks=_md_n_ranks,
1451-        # Non-cs-spmd multi-rank = route-A mpi4jax halos; pin the transport
1452:        # so a run that also initialized jax.distributed (process_count ==
1453-        # world size) cannot auto-resolve to nccl/gloo (codex finding 1).
1454-        # cs-spmd (route-B) keeps auto-resolution.
1455-        transport=("mpi4jax" if (not cs_spmd and result.n_ranks > 1)
1456-                   else None),
1457-        n_gpus=(result.n_ranks
1458-                if payload["backend"] in ("gpu", "cuda", "rocm") else 0),
1459-        decomposition=result.decomposition,
1460-        # cells_per_rank is per PROCESS (n_ranks semantics).  cs-spmd:
1461-        # result.cells_per_rank is per global DEVICE (total // n_global),
1462:        # while metadata n_ranks defaults to jax.process_count() — divide
1463-        # the total by the live process count instead and keep the
1464-        # per-device share in extra (codex finding 3, cs-spmd leg).
1465:        cells_per_rank=(result.total_cells // max(_live_process_count(), 1)
1466-                        if cs_spmd else result.cells_per_rank),
1467-        scaling_kind=os.environ.get("LEGOESM_SCALING_KIND") or None,
1468-        partition_metrics=_part_metrics,
1469-        extra={
1470-            "physics_level": result.physics_level,
--
1679-        # BEFORE any JAX use) and mpi4jax is never armed — rank identity
1680-        # comes from the runtime, so a CUDA venv without a loadable libmpi
1681-        # is VALID here (job 26449146: the mpi4py loud-guard killed the
1682-        # NCCL cube lane that needs no MPI at all). n_ranks keeps the
1683-        # LAUNCHER world size so the partial-federation gate below still
1684:        # compares jax.process_count() against what was launched.
1685-        import jax as _jax
1686-        _lw = _launcher_world_size()
1687-        rank = _jax.process_index()
1688:        n_ranks = _lw if _lw > 1 else _jax.process_count()
1689-    else:
1690-        rank, n_ranks = _init_mpi()
1691-    is_rank0 = (rank == 0)
1692-
1693-    # cs-spmd consistency gate: every launched process must have joined
1694-    # ONE multi-controller program.  A mismatch means initialize() was
1695-    # skipped (unknown launcher) or partially failed — measuring would
1696-    # produce N independent serial runs labelled n_ranks=N.
1697-    if args.cs_spmd:
1698-        import jax as _jax
1699:        if n_ranks > 1 and _jax.process_count() != n_ranks:
1700-            if is_rank0:
1701-                print(
1702-                    f"ERROR: --cs-spmd launched with {n_ranks} MPI "
1703:                    f"processes but jax.process_count()="
1704:                    f"{_jax.process_count()} — jax.distributed did not "
1705-                    "federate them (unsupported launcher?).  Refusing "
1706-                    "to record replicated-serial numbers.",
1707-                    flush=True,
1708-                )
1709-            return 3
--
1805-
1806-    result = run_single_benchmark(
1807-        grid_type=grid_type,
1808-        resolution=resolution,
1809-        nlev=nlev,
1810:        n_ranks=n_ranks,
1811-        rank=rank,
1812-        precision=precision,
1813-        mode=mode,
1814-        physics_level=physics_level,
1815-        n_warmup=args.n_warmup,
--- generic metadata semantics ---
73-    "backend",
74-    "decomposition",
75-    "n_ranks",
76-    "n_gpus",
77-    "device_count",
78:    "process_count",
79-    "gpu_direct_active",
80-    "host_staged_halo",
81-    "transport",
82-    "virtual_cpu_devices",
83-    "launcher",
--
341-
342-def resolve_transport(
343-    transport: str | None,
344-    *,
345-    n_ranks: int,
346:    process_count: int,
347-    device_count: int,
348-    backend: str | None = None,
349-) -> str:
350-    """Resolve (and validate) the halo/collective transport for the record.
351-
352-    Explicit ``transport`` wins (validated against :data:`TRANSPORTS`).
353-    Auto-resolution when ``None``:
354-
355:    - ``n_ranks > process_count``: a route-A MPI world JAX cannot see (each
356-      rank is a separate single-process JAX) → ``"mpi4jax"``.
357:    - ``process_count > 1``: route-B multi-controller ``jax.distributed`` →
358-      ``"nccl"`` on a GPU backend, ``"gloo"`` on CPU (JAX's CPU collective
359-      transport).
360-    - ``device_count > 1``: single-process SPMD → ``"xla-local"``.
361-    - else ``"none"``.
362-    """
--
366-                f"unknown transport {transport!r}; expected one of "
367-                f"{TRANSPORTS} (a scaling row's transport must be a known, "
368-                "comparable fabric)."
369-            )
370-        return transport
371:    if n_ranks > process_count:
372-        return "mpi4jax"
373:    if process_count > 1:
374-        b = (backend or detect_backend()).lower()
375-        return "nccl" if b in _GPU_BACKENDS else "gloo"
376-    if device_count > 1:
377-        return "xla-local"
378-    return "none"
--
408-        "gpu_direct_active": bool(on_gpu and mpi4jax_halo and device_direct),
409-        "host_staged_halo": bool(on_gpu and mpi4jax_halo and not device_direct),
410-    }
411-
412-
413:def scaling_metadata(
414-    *,
415-    grid: str,
416-    component: str,
417-    resolution: Any,
418-    n_levels: int,
--
437-    ----------
438-    grid, component, resolution, n_levels, precision, decomposition
439-        Scientific descriptors of the case (caller-supplied).
440-    n_ranks, n_gpus, devices_per_rank, cells_per_rank
441-        Parallel layout.  ``devices_per_rank`` defaults to
442:        ``jax.device_count() // jax.process_count()`` when omitted.
443-    solver_variant, solver_residual, conservation_drift
444-        Solver identity + convergence/conservation evidence.  The roadmap
445-        forbids claiming a solver speedup without recording a residual /
446-        conservation drift, so these belong in the record.
447-    scaling_kind
--
461-    extra
462-        Any other component-specific descriptors.
463-    """
464-    backend = detect_backend()
465-    device_count = _jax_count("device_count", -1)
466:    process_count = _jax_count("process_count", 1)
467-    # Resolve the transport BEFORE gpu_direct_mode: host_staged_halo /
468-    # gpu_direct_active are mpi4jax-halo semantics and must not fire on a
469-    # route-B NCCL / xla-local / serial row (codex finding 2).
470-    # ``n_ranks`` = number of MPI processes.  Default to the auto-detected
471-    # process count so a single-process SPMD run (1 process, N GPUs) records
472-    # ``n_ranks=1`` (truthful) while the device parallelism lives in
473-    # ``n_gpus`` / ``device_count``.  Route-A drivers (1 GPU/rank) pass the
474-    # real rank count explicitly.
475-    if n_ranks is None:
476:        n_ranks = process_count
477:    if devices_per_rank is None and device_count > 0 and process_count > 0:
478:        devices_per_rank = device_count // process_count
479-    resolved_transport = resolve_transport(
480-        transport,
481-        n_ranks=int(n_ranks),
482:        process_count=process_count,
483-        device_count=device_count,
484-        backend=backend,
485-    )
486-
487-    md: dict[str, Any] = {
--
500-        "scaling_kind": scaling_kind,
501-        # --- parallel layout ---
502-        "n_ranks": n_ranks,
503-        "n_gpus": n_gpus,
504-        "device_count": device_count,
505:        "process_count": process_count,
506-        "devices_per_rank": devices_per_rank,
507-        "cells_per_rank": cells_per_rank,
508-        # --- runtime facts (auto-detected) ---
509-        "backend": backend,
510-        "precision_knobs": precision_knobs(),
--
966-    import time
967-
968-    import jax
969-    import numpy as np
970-
971:    multi = jax.process_count() > 1
972-    if multi:
973-        # Collectively verify the schedule BEFORE compilation or any
974-        # schedule-dependent fence: this allgather is the single collective
975-        # every process reaches first, so on a mismatch EVERY process sees
976-        # the same gathered table and raises together instead of hanging in
--- exact arithmetic ---
/usr/bin/bash: line 5: cannot create temp file for here-document: Read-only file system
--- r512 factor selection test / source ---
   808	def _factor_2d_latlon(n_ranks, n_lat, n_lon, min_lat=2, min_lon=2):
   809	    """Factor ``n_ranks`` into ``(proc_lat, proc_lon)`` for the 2-D pencil,
   810	    MINIMISING the per-rank halo perimeter ``n_lat/proc_lat + n_lon/proc_lon``
   811	    (the whole point of 2-D vs the 1-D band).
   812	
   813	    Constraints: each block keeps ``>= min_lat`` latitude rows (the halo=2
   814	    PPM/biharmonic exchange) and ``>= min_lon`` longitude columns.  Raises if
   815	    no valid factorisation exists (e.g. too many ranks for the resolution),
   816	    rather than silently building a degenerate block.
   817	
   818	    Returns the min-perimeter pair; ties broken toward the more balanced
   819	    block (smaller ``|n_lat/pl - n_lon/pc|``).
   820	    """
   821	    best = None  # (perimeter, imbalance, proc_lat, proc_lon)
   822	    for pl in range(1, n_ranks + 1):
   823	        if n_ranks % pl:
   824	            continue
   825	        pc = n_ranks // pl
   826	        blat, blon = n_lat // pl, n_lon // pc
   827	        if blat < min_lat or blon < min_lon:
   828	            continue
   829	        perim = n_lat / pl + n_lon / pc
   830	        imbal = abs(n_lat / pl - n_lon / pc)
   831	        key = (perim, imbal)
   832	        if best is None or key < best[0]:
   833	            best = (key, pl, pc)
   834	    if best is None:
   835	        raise ValueError(
   836	            f"_factor_2d_latlon: no 2-D factorisation of n_ranks={n_ranks} "
   837	            f"keeps >= {min_lat} lat rows AND >= {min_lon} lon cols per block "
   838	            f"for n_lat={n_lat}, n_lon={n_lon}.  Reduce ranks or raise "
   839	            f"resolution."
   840	        )
   841	    return best[1], best[2]
   842	
   843	
   844	def _build_latlon(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
   845	                   physics_level, cast_fn, latlon_2d=False):
   846	    import jax
   847	    import jax.numpy as jnp
   848	
   849	    from legoesm.grids.latlon import create_latlon_grid
   850	    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
   884	    if _moist:
   885	        # Kessler warm-rain bound to this step's dt (the physics_fn convention
   886	        # passes no timestep).  Column-local, so it adds NO horizontal halo
   887	        # coupling beyond the dycore's existing mass-consistent tracer
   888	        # exchange — moist scales on the same ladder as dry.
   889	        from legoesm.atmosphere.forcing.idealized.kessler_forcing import make_kessler_forcing_latlon
   890	        physics_fn = make_kessler_forcing_latlon(dt)
   891	    else:
   892	        physics_fn = _build_physics_fn(physics_level, "latlon")
   893	
   894	    if n_ranks > 1 and latlon_2d:
   895	        # 2-D pencil (proc_lat x proc_lon) decomposition — the SOTA fix for
   896	        # the 1-D band's high-rank halo-perimeter starvation.  Wall poles
   897	        # (regular grid; use_polar_filter already False above).  Same
   898	        # build shape as the band: global cell-centred -> global C-grid
   899	        # (serial) -> 2-D layout -> rank-local block model -> scatter ->
   900	        # make_latlon_2d_mpi_step (which arms the MPI halo backend + sets the
   901	        # rank-aware pole_v_bc + allreduced total_area itself).
   902	        from legoesm.parallel.latlon_mpi import (
   903	            make_latlon_2d_layout,
   904	            make_latlon_2d_mpi_step,
   905	            scatter_state_latlon_2d,
   906	            slice_latlon_grid_to_block_2d,
   907	        )
   908	        proc_lat, proc_lon = _factor_2d_latlon(n_ranks, n_lat, n_lon)
   909	        cgrid_global = hydrostatic_to_cgrid(state, grid)
   910	        layout2d = make_latlon_2d_layout(
   911	            rank, proc_lat, proc_lon, n_lat, n_lon,
   912	        )
   913	        block_grid = slice_latlon_grid_to_block_2d(grid, layout2d)
   914	        local_model = CGridLatLonPrimitiveEquationModel(
   915	            block_grid, sigma, config, dt=dt,
   916	        )
   917	        state = scatter_state_latlon_2d(cgrid_global, layout2d)
   918	        step_fn = make_latlon_2d_mpi_step(
   919	            local_model, layout2d, physics_fn=physics_fn,
   920	        )
   921	        cells_per_rank = layout2d.n_lat_local * layout2d.n_lon_local * nlev
   922	        return step_fn, state, dt, total_cells, cells_per_rank
   923	
   924	    if n_ranks > 1:
   925	        # Latitude-band MPI (mirrors the multi-rank icosahedral path):
   926	        # 1. convert the global state to raw C-grid arrays *before*
   927	        #    arming the MPI halo backend (the conversion's pole pads
   928	        #    must run on the global array with the local backend),
   929	        # 2. arm the band layout + MPI halo backend,
   930	        # 3. slice the global grid to this rank's band, build the
   931	        #    rank-local model on it,
   932	        # 4. scatter the global state to the band,
   933	        # 5. wrap the step;  ``make_latlon_mpi_step`` forwards
   934	        #    ``physics_fn`` per RK stage exactly like the serial
   935	        #    ``model.step(state, dt, physics_fn=...)`` path
   936	        #    (Held-Suarez is column-local, so it adds no halo
   937	        #    coupling beyond the dycore's own exchanges).
   938	        from legoesm.parallel.distributed import initialize_distributed_latlon
   939	        from legoesm.parallel.latlon_mpi import (
   940	            make_latlon_mpi_step,
--- current diff and whitespace ---
 .../scaling/levante_campaign_2026-07-24.md         | 163 +++++++++++++++++++++
 scripts/plot/plot_scaling_paper_figure.py          |  39 +++--
 2 files changed, 190 insertions(+), 12 deletions(-)

exec
/usr/bin/bash -lc "printf '%s\\n' '--- shell scope/exec simulation ---'
run_arm() { bash -c '[ \""'$SIM_PROCID" = 0 ] && echo "[step $ARM_TAG] nodelist=$SIM_NODELIST"; exec "$0" "$@"'"' /bin/printf 'exec argv: %s|%s\\n'; }
ARM_TAG=solo_pre; export ARM_TAG; SIM_PROCID=0 SIM_NODELIST=n01,n02 run_arm alpha
for tag in rep0 rep1 rep2 rep3; do ARM_TAG=\""'$tag" SIM_PROCID=0 SIM_NODELIST="${tag}nodes" run_arm "$tag" & pids+="$! "; done
for pid in $pids; do wait "$pid"; done
printf '"'%s\\n' '--- exact scaling arithmetic ---'
node -e 'const t0=297.57; for (const [n,t] of [[64,297.57],[128,161.03],[256,72.06],[512,44.78]]) { const s=t0/t, e=(t0*64)/(t*n); console.log("'`${n}: numerator=${t0*64}, denominator=${t*n}, speedup=${s.toFixed(6)}, eff=${e.toFixed(6)}`) }; console.log(`LL f32 GC/s=${(2048*4096*26/0.0055767/1e9).toFixed(6)}`); console.log(`LL f64 GC/s=${(2048*4096*26/0.0096015/1e9).toFixed(6)}`)'"'
printf '%s\\n' '--- no hidden updated ensemble receipt/doc artifact ---'
rg --files -uu | rg -i '26628196|mpas_s9_ens.*\\.(log|json|jsonl)|s9_ensemble.*\\.(log|json|jsonl)' || true
printf '%s\\n' '--- documentation claims of new CPU receipt’s validity/correctness ---'
rg -n -i -C 2 '26628073|lat-lon CPU lane scales|cleanly|MPI|mpi4jax|UCX|parity|conservation|finite' docs/performance/scaling/levante_campaign_2026-07-24.md | tail -n 240
printf '%s\\n' '--- finite / parity in CPU driver ---'
rg -n -i -C 2 'finite|parity|conservation|valid' scripts/bench/run_cpu_mpi_scaling.py | tail -n 260" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 434ms:
--- shell scope/exec simulation ---
[step solo_pre] nodelist=n01,n02
exec argv: |
[step rep2] nodelist=rep2nodes
[step rep1] nodelist=rep1nodes
[step rep0] nodelist=rep0nodes
exec argv: |
exec argv: |
exec argv: |
[step rep3] nodelist=rep3nodes
exec argv: |
--- exact scaling arithmetic ---
64: numerator=19044.48, denominator=19044.48, speedup=1.000000, eff=1.000000
128: numerator=19044.48, denominator=20611.84, speedup=1.847917, eff=0.923958
256: numerator=19044.48, denominator=18447.36, speedup=4.129475, eff=1.032369
512: numerator=19044.48, denominator=22927.36, speedup=6.645154, eff=0.830644
LL f32 GC/s=39.109833
LL f64 GC/s=22.715597
--- no hidden updated ensemble receipt/doc artifact ---
--- documentation claims of new CPU receipt’s validity/correctness ---
--
285-np128 ocean + weak ladders and the rank-failure attribution remain open.
286-
287:## Atm ICOSAHEDRAL CPU-MPI (job 26452579, 1 node, f64 moist, L26)
288-
289-The last measurement gap, and the healthiest strong-scaling curve in the
--
314-on coarse grids (0.15 at subdiv4, 0.30 at subdiv5) and holds on fine ones
315-(0.52-0.57 at subdiv6-7). Same pattern as the cube (0.23 -> 1.04) and the ocean (0.37 -> 0.63), now
316:on a third lane and a different transport (CPU-MPI, not NCCL). It is the
317-campaign's most reproducible ASSOCIATION — but changing C-resolution, LL
318-size or ico subdivision also changes the global problem, so tile size is
--
334-The ico WEAK ladder from the same job is non-monotone (1.00 / 0.47 / 0.81 /
335-0.48 / 0.46 / 0.22 / 0.45 at np 1..64) — the per-rank problem size is not
336:held constant cleanly across that sweep's subdivision steps, so no weak
337-claim is made from it.
338-
--
430-If exposed dependent sync is the cost, PCG iteration count is the most
431-direct lever on it (each iteration carries dependent reduction batches).
432:`--pcg-fixed-iters` was wired onto the SPMD bench for this (the MPI twin
433-already had it) and swept at LL576 f64:
434-
--
538-half. The second half is the same quantity wide-halo removes for the
539-explicit solver — which is why wide-halo was the larger win in the earlier
540:arms, and it now has a mechanistic reason rather than just an empirical
541-ranking.
542-
543-## MATCHED CONFIG HEAD-TO-HEAD — the production recommendation (job 26460365)
544-
545:All three arms in ONE job on ONE node, back to back, conservation-gated
546-(LL576x1152 L20 f64):
547-
--
610-This does NOT clear that gate — a gate needs the filter analysis plus an
611-ocean-science sign-off — but it supplies the first thing a reviewer would
612:ask for: 600 steps at nd4, f32, conservation-gated, both arms same job.
613-
614-| arm | eta drift [m] | heat_rel | salt_rel |
--
637-stability gate — that needs the filter analysis under stale halos and a
638-science sign-off — but "no observed failure" has become "marginally better
639:conservation with a measured variance estimate".
640-
641-Per-step time is flat 100 -> 600 steps (12.81 -> 12.74 ms). Both arms' heat
--
657-
658-1. **wide-halo is the SAME explicit-substep operators with
659:   tolerance-parity coverage at every transport tier** (codex round-9
660-   wording), not an untested scheme variant. Coverage:
661:   serial `tests/ocean/unit/test_barotropic_wide_halo.py` — parity atol
662-   1e-12 f64 across four configs (div-damp, power-law filter, multi-chunk),
663:   volume-drift parity 1e-15, NaN-sentinel stencil-reach pin; re-run
664:   2026-07-26: 11 passed, 1 skipped (mpi4jax-gated dispatch test).
665:   Distributed: `tests/ocean/distributed/test_ocean_mpi_wide_halo_parity.py`
666:   (MPI gathered-vs-serial, 1e-10, incl. rank-cut/v-face/chunk cases) and
667-   `tests/parallel/test_latlon_ocean_spmd_wide_halo.py` (4-device SPMD,
668:   2e-4/1e-3). These are TOLERANCE parity, not bit identity — "the filter
669-   sees bit-identical inputs" is too strong; differences are XLA
670-   re-association at the serial tier and larger at the distributed tiers.
671-   ONE REAL BEHAVIOURAL DELTA to disclose: wide-halo requires LOCAL
672:   subcycle clamping, and with an active `eta_floor` the clamp/
673-   redistribution schedule differs from the standard path — a reviewer
674-   should check that config interaction, not filter stability in general.
--
690-depth on that config). The remaining sign-off is experiment-level: run the
691-OMIP case with explicit_substep+wide at production dt and confirm the
692:3-seed conservation result (heat 7.1 SE, salt 15.2 SE lower than
693-implicit_cn at 600 steps unforced) holds under forcing. That is a science
694-review of ONE config field on ONE experiment, not a scheme-stability
--
703-count. Only the second contrast can show a genuine comm/N effect.
704-
705:**CPU-MPI lane, first verdict (job 26495929, ico f64, block:cyclic
706-throughout, 4 nodes, single runs).** Aligning both meshes by CELLS PER
707-RANK rather than rank count:
--
787-the 2.25x fixed-tile contrast supports only "little growth over the
788-tested range", not universality. That single number explains
789:the cube's efficiency 0.62, the empirical tile floor, and the plateau in
790-the figure.
791-
--
924-tiled lane (the CORRECT >6-GPU vehicle) failed to complete even C384 L60
925-in f64 within a 3 h wall (job 26512794), where the same arm in f32 runs
926:in ~9 ms/step. No output, no error — it never finished compiling. That
927-is consistent with #1370: if setup allocates global-sized buffers, f64
928:doubles them, and the compile/allocation path degrades accordingly. So
929-the cube's f64 GPU column stays EMPTY in the figure, and it is a
930-capability gap rather than a measurement I skipped.
--
1086-  designated single builder (opt-in env + per-key O_EXCL lockfile with
1087-  stale takeover — the opt-in alone would be a thundering herd across an
1088:  MPI launch). >10 stays hard-refused. Six policy tests + prewarm CLI
1089-  (`scripts/data/prewarm_voronoi_mesh.py`) with its direct test.
1090-* **lloyd=0 admitted as a LABELLED synthetic scaling mesh** after the
--
1169-| signal | @16 | @32 | reading |
1170-|---|---|---|---|
1171:| compiled entry args | 0.06 GB | 0.06 GB | step is CLEAN |
1172:| compiled temps | 0.22 | 0.21 | not remat pressure |
1173-| state leaves (per-device) | 0.070 sharded / 0 replicated | same | sharding correct |
1174-| **bytes_in_use** | **1.58 GB** | **3.11 GB** | **tracks GLOBAL size** |
--
1187-`jax.default_device(local cpu)` at nd>1 drops per-device residency
1188-**1.58 -> 0.30 GB (@16) and 3.11 -> 0.71 GB (@32) — a 5x reduction** —
1189:with the compiled step unchanged and parity at 1e-10. Getting there
1190-burned four probe attempts on real multicontroller facts, each recorded:
1191:lower/compile is COLLECTIVE (rank-0-only deadlocks the shutdown
1192-barrier); `jax.devices()` is the GLOBAL list under jax.distributed (use
1193-`local_devices`); `JAX_PLATFORMS=cuda` unregisters the cpu backend; and
--
1260-  default `auto` is doing well to land near geometric.
1261-* Practical: pin `--partition-method geometric` (or auto) on this lane;
1262:  sfc is actively harmful at scale. pymetis is absent from `.venv-mpi`,
1263-  so the low-cut METIS arm codex wanted is still unmeasured.
1264-
--
1284-internals: `LEGOESM_VMIX_F32_SOLVE=1 LEGOESM_BAROCLINIC_F32=1`) — was the
1285-untested corner of the decision table. Now measured** (job 26493592,
1286:LL576 L20, same-job arms, conservation-gated 1e-5):
1287-
1288-| arm | plain f64 nd1/nd4 | MIXED nd1/nd4 | nd4 mixed gain |
--
1328-  bench's own metadata says it: `n_ranks: 1, cells_per_rank_achieved:
1329-  163842` — every arm ran ONE rank on the FULL mesh, because this bench
1330:  decomposes by MPI RANK (its docstring states the SPMD multi-device
1331-  path does not exist by design) and my CUDA_VISIBLE_DEVICES invocation
1332-  never created ranks. The flat curves were the SAME single-device run
--
1335-  single-device timings (s6 ~7 ms f32/f64, s7 ~21.9 ms f32).
1336-
1337:  **THE REAL LADDER (job 26494036, CPU-MPI f64, np1-16, block:cyclic,
1338-  ranks=N verified in metadata): MPAS-ocean SCALES.** s6 (41k cells):
1339-  814.46 / 316.71 / 159.27 / 92.85 / 90.51 ms; s7 (164k cells): 3944.43
--
1427-| 26460447 | ocean LL576 np16 single_reduce | 50 min | 67 s (job 26460876) |
1428-
1429:Each time the log stops during tracing/compile with no error. Twice I
1430:suspected a real compile-time defect (a chunk-heuristic cliff, then a
1431-16-device solver problem) and twice the retry refuted it. TREAT A SINGLE
1432-MULTI-NODE HANG AS TRANSIENT until a second occurrence with the same
--
1444-   visible (`--gpu-bind=none`) and let jax bind LOCALID-th (Derecho/PALS
1445-   keeps its shim + `local_device_ids=[0]`; conventions must not mix).
1446:4. cs-spmd rank identity from the federated jax runtime (mpi4py loud-guard
1447:   killed the NCCL cube lane that needs no MPI); launcher world size now
1448-   global-only (PALS_LOCAL_SIZE dropped), step size preferred.
1449-5. `make_sharded_ocean_step` replicated-geometry `device_put` asserts
--
1455-   guard (last: jobs 26453906/26453981), plus live np4/8/16 multinode
1456-   (jobs 26452743-45, 26453279).
1457:6. Multicontroller host materialization in the tiled bench finiteness gate
1458-   → on-device global reduce.
1459-7. OUTDIR same-second stamp collision → job-ID suffix everywhere.
1460:8. `setup_mpi_venv.sh` could not resolve uv-workspace members with plain
1461:   pip → pins first + `install_federation.py --all`; mpi4jax source-built
1462:   with the system toolchain (GLIBCXX mismatch with gcc-11-built OpenMPI
1463-   module).
1464-9. Diagnosis tool halo/overlap phases timed UN-JITTED eager pads
--
1479-`--xla_gpu_enable_pipelined_p2p` alone, `LEGOESM_BAROCLINIC_F32` (~+0.5%, within run-to-run spread; job 26451282).
1480-PGLE arm invalid as measured here (the 33-step window catches its
1481:profile+recompile). The +8.5% figure is from the DERECHO lane-T campaign
1482-(see the SOTA review), not reproduced on Levante — rerun long-window if
1483-revisited.
--
1487-1. **OMIP config sign-off for explicit_substep+wide** (formerly "wide-halo
1488-   stability gates" — reframed 2026-07-26 after codex round-9). Wide-halo
1489:   has tolerance-parity coverage at all three transport tiers (serial
1490:   1e-12, MPI 1e-10, SPMD 2e-4/1e-3) and explicit_substep is an
1491-   established scheme (the model default; benches and OMIP explicitly
1492-   choose implicit_cn). Remaining: the eta_floor x local-clamp config
--
1504-
1505-   *Cheap half — wet-BALANCED bands* (no indirection, uneven band heights
1506:   equalizing OCEAN cells per rank): already implemented on the MPI lane
1507:   (`bench_ocean_mpi_scaling.py --wet-balance`, ETOPO continents); A/B at
1508-   np16/np32 running (job 26480448, r128/r256 at CFL-scaled dt 100/50s).
1509-   OPERATIONAL TRAIL kept for honesty: four earlier submissions failed —
1510:   wrong venv (26479815), then non-finite at dt 600 and 300 (26479904/
1511-   26480203/26480298), briefly mis-read as a lane defect until the
1512-   bench's own WARNING surfaced: realistic coastlines are documented to
--
1575-   ocean lane pencil refusal stands (`test_2d_pencil_layout_refused` —
1576-   wide-halo is 1-D-only by design).
1577:7. Route-A CUDA-aware mpi4jax lane (`gpu_moist_scaling.slurm`) — only if a
1578-   route-A-vs-B A/B is ever wanted; route-B beat every route-A reference
1579-   available here.
--
1581-DONE during the campaign (were open at the start): C768 same-path ladder
1582-(eff 1.04) and tiled closed loop (14.3 GCells/s); atm lat-lon weak at a
1583:production tile; ocean weak at a production tile; the ico CPU-MPI ladder;
1584-the CPU spread ladder; and the diagnosis tool's halo + overlap phases,
1585-which were found broken and fixed with a contract test.
--
1599-`int(mesh.nCells) // n_ranks`, bench_ocean_mpas_scaling.py:751 — NOT a
1600-balance statement; method pinned per arm, never `auto`, which flipped
1601:meaning when pymetis appeared in `.venv-mpi` on 2026-07-31), f64,
1602-nlev 20, 32 ranks/node. Actual per-rank OWNED ranges
1603-(`metadata.partition_metrics.cells_per_rank_min/max`): geometric
--
1624-  -> step-time inference FAILS on this lane; part of metis's loss is
1625-  PLAUSIBLY its own wet-load imbalance (above). Scope: closes the
1626:  "swap in METIS as-is" lever on the CPU-MPI ocean lane; does NOT rule
1627-  out partition/mapping improvements generally (e.g. wet-cell-weighted
1628-  METIS was NOT tested).
--
1633-  memory-bandwidth balance) PLAUSIBLE, consistent with the np16 Milan
1634-  2.13x receipt; never instrumented with bandwidth counters.
1635:* Caveats: timing-only receipt — no parity/conservation gate ran in
1636:  these arms, and the CPU nodes emit `UCX WARN transports
1637-  'cuda_copy','cuda_ipc','gdr_copy' are not available` (the _env.sh GPU
1638:  UCX_TLS list on a CPU node; UCX falls back to rc/sm — cosmetic for
1639-  timing, but a "production config" claim would need a gated arm).
1640-
--
1720-| 26628071 | oc LL2304 retry post-#1370 | 128 GPU | pre-fix failure was resident-args; predicted PASS at ~0.10 GB/dev residency |
1721-| 26628072 | atm LL2304 @96/@192 + LL2880 @192 | 96-192 GPU | LL2048 does not divide 192; LL2880@192 = 86.4k cols/GPU ABOVE floor |
1722:| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
1723-| 26628074 | subdiv-10 lloyd0 prewarm | 1 CPU | unlocks MPAS 128-224 GPUs ABOVE floor (81.9k-46.8k cells/GPU) |
1724-| 26628076 | s8 lloyd0 np8/16/32 | 32 GPU | weak-pair de-confound (codex r20 item 5) |
--
1726-s10 ladder (128/192/224 GPUs) submits once 26628074's cache lands.
1727-
1728:### First hundreds receipt in: lat-lon CPU 2-D pencil to 512 ranks (job 26628073)
1729-
1730-r512 (512x1024 = 524k cols) L26 f64 moist, 32 rpn block:cyclic,
--
1740-Distribution verified against the masquerade trap: result rows carry
1741-`n_ranks: 512` (the JSON's `metadata.process_count: 1` is the jax-LOCAL
1742:count on this mpi4jax lane, not the world size). 128->256 is
1743-SUPERLINEAR (2.23x for 2x) — classic per-rank working-set cache
1744-transition on Milan (mechanism PLAUSIBLE, uninstrumented). End-to-end
1745:64->512 eff 0.83 at 1k cols/rank: the lat-lon CPU lane scales into the
1746:hundreds cleanly. (Exact pencil factorisations are not recorded in the
1747-result JSON — only `decomposition: 2d`; a follow-up could add them to
1748-the bench metadata.)
--- finite / parity in CPU driver ---
291-
292-
293:def _validate_physics(grid_type: str, physics_level: str) -> None:
294-    supported = _SUPPORTED_PHYSICS.get(grid_type, set())
295-    if physics_level not in supported:
--
378-
379-
380:def _valid_rank_counts(max_ranks: int, grid_type: str) -> list[int]:
381-    if grid_type == "spectral":
382-        return [1]
--
526-        hyperdiff_coeff=0.0,
527-        hyperdiff_ps_coeff=0.0,
528:        use_conservation_fixer=True,
529-        fix_mass=True,
530-        anchor_mass_to_initial=True,
--
574-    would give each process a private 1-device mesh), the existing
575-    ``make_sharded_step`` + multiface-ppermute halo runs unchanged, and
576:    the state is sharded across the global device set.  Conservation
577-    (``fix_mass``) reduces on global-sharded arrays at the jnp level —
578:    SPMD-global by construction (parity receipt 6.7e-10 @5 steps, job
579-    8462928).  ``jax.distributed.initialize()`` must already have run
580-    (``main`` does it for ``--cs-spmd`` BEFORE any other JAX use).
--
615-            f"--cs-spmd needs a device count dividing 6 or 6*kt^2 "
616-            f"(sub-face tiling), got {n_global} (srun -n "
617:            f"1|2|3|6|24|54|...).  np=24 parity receipt: 4.4e-10 "
618-            f"@5 steps, job 8465445."
619-        )
--
623-    # indexing + staggered leaves cannot shard over tile axes — its own
624-    # scope note), so >6 devices dispatch to the BLOCKED persistent tiled
625:    # step (tiled_production_cdgrid; np24/np54 parity-gated) instead of
626-    # silently building a broken/replicated program.
627-    if n_global > 6:
--
639-    # IDENTICAL config to the serial cubed-sphere baseline above —
640-    # speedup comparisons are meaningless across different dynamics
641:    # settings.  The conservation fixers reduce via jnp-level global
642-    # sums on global-sharded arrays (SPMD-global psum by construction;
643:    # conservation-reduction audit + parity receipt).
644-    config = CDGridPrimitiveEquationConfig(
645-        hyperdiff_coeff=0.0,
646-        hyperdiff_ps_coeff=0.0,
647:        use_conservation_fixer=True,
648-        fix_mass=True,
649-        anchor_mass_to_initial=True,
--
679-    ``s = step_fn(s, dt)`` feedback with NO per-step gather); the serial
680-    post-step telescoping dry-mass fixer runs IN-STAGE, matching the np<=6
681:    lane's conservation config (whose anchor path threads the per-call
682-    pre-step mass under the outer jit and telescopes identically).  Same
683-    baroclinic-wave IC + config as the np<=6 cs-spmd lane so the ladder is
684:    one controlled comparison.  Parity gates:
685-    tests/parallel/test_tiled_blocked_loop.py (np24, vs serial model.step).
686-    """
--
720-    # This literal is inside the tiled envelope by construction (every damp
721-    # at its 0 default, hyperdiff pinned 0, end-step fixer ON so the serial
722:    # per-stage zero-mean is skipped); the blocked factory validates its
723-    # own grid/coord args below.
724-    config = CDGridPrimitiveEquationConfig(
725-        hyperdiff_coeff=0.0,
726-        hyperdiff_ps_coeff=0.0,
727:        use_conservation_fixer=True,
728-        fix_mass=True,
729-        anchor_mass_to_initial=True,
--
813-    Constraints: each block keeps ``>= min_lat`` latitude rows (the halo=2
814-    PPM/biharmonic exchange) and ``>= min_lon`` longitude columns.  Raises if
815:    no valid factorisation exists (e.g. too many ranks for the resolution),
816-    rather than silently building a degenerate block.
817-
--
1021-    # Mass fixer adds one global allreduce per step (267.8 us latency floor on
1022-    # Ginsburg/Gloo). LEGOESM_NO_MASS_FIX=1 disables it for a scaling ABLATION
1023:    # that isolates the dynamics+halo cost from the conservation allreduce
1024:    # (codex MPI-improve #3). Production keeps it ON (conservation).
1025-    _fix_mass = os.environ.get("LEGOESM_NO_MASS_FIX") != "1"
1026-    config = MPASPrimitiveEquationConfig(
--
1151-) -> TimingResult:
1152-    """Run a single benchmark case and return timing."""
1153:    _validate_physics(grid_type, physics_level)
1154-
1155-    import jax
--
1335-    """Generate all benchmark cases for a sweep."""
1336-    cases = []
1337:    rank_counts = _valid_rank_counts(max_ranks, grid_type)
1338-
1339-    # icosahedral MPI multi-rank now applies ``physics_fn`` via the
--
1432-    payload["cpus_per_task"] = _cpt
1433-    payload["n_cores"] = result.n_ranks * _cpt
1434:    # Record conservation mode so a LEGOESM_NO_MASS_FIX ablation never dedups
1435-    # with / is mislabeled as a production (mass-conserving) run (codex audit).
1436-    payload["fix_mass"] = os.environ.get("LEGOESM_NO_MASS_FIX") != "1"
--
1538-             "Uses jax.distributed ONLY — the mpi4jax halo backend is "
1539-             "never armed in this mode (mixed stacks deadlock).  "
1540:             "Parity receipt: scripts/tmp/_probe_spmd_cube_parity.py "
1541-             "(shard-local vs serial = 6.7e-10 @5 steps, job 8462928).",
1542-    )
--
1548-             "(regular grid; use_polar_filter off) — a labeled throughput "
1549-             "benchmark, NOT the atmosphere's 180-deg pole fold.  Targets the "
1550:             "band's high-rank starvation (weak-E ~0.05).  Validated by "
1551-             "tests/distributed/test_latlon_2d_mpi_step.py (mass<1e-12 + "
1552-             "2x2==1x4).  --grid latlon only.",
--
1591-    # --- Sweep mode: just print cases and exit ---
1592-    if args.sweep:
1593:        # Iter 41: validate the grid+physics combo *before* the
1594-        # sweep generator runs, so unsupported tiers (e.g. moist
1595-        # physics on cubed-sphere/lat-lon CPU MPI, see iter 40)
--
1597-        # producing a JSON-lines list that subsequently fails at
1598-        # runtime under the SLURM array.
1599:        _validate_physics(args.grid, args.physics)
1600-        cases = generate_sweep_cases(
1601-            args.grid, args.mode if args.mode != "single" else "both",
--
1679-        # BEFORE any JAX use) and mpi4jax is never armed — rank identity
1680-        # comes from the runtime, so a CUDA venv without a loadable libmpi
1681:        # is VALID here (job 26449146: the mpi4py loud-guard killed the
1682-        # NCCL cube lane that needs no MPI at all). n_ranks keeps the
1683-        # LAUNCHER world size so the partial-federation gate below still
--
1750-        return 1
1751-    # Iter 14 follow-up: defensive runtime guard for cubed-sphere MPI.
1752:    # The iter 13 ``_valid_rank_counts`` guard prevents the sweep from
1753-    # *generating* multi-rank cases, but a user could still pass an
1754-    # explicit ``--case`` with ``n_ranks>1`` or invoke the script under
--
1777-    # The >=2-lat-rows-per-rank guard is for the 1-D BAND (all ranks split
1778-    # lat).  The 2-D pencil splits lat over proc_lat (< n_ranks), so its own
1779:    # _factor_2d_latlon validates the per-block rows/cols — skip the band
1780-    # guard for --latlon-2d.
1781-    if (grid_type == "latlon" and not args.latlon_2d

exec
/usr/bin/bash -lc "printf '%s\\n' '--- recovery raw-artifact availability ---'
rg --files -uu | rg '(26600094|26600095|26534060|26502539|26628072|26628076)' || true
printf '%s\\n' '--- all results validity loops ---'
for f in scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch; do
  echo \""'$f"; rg -n -C 5 '"'RESULTS|isfinite|receipt invalid|DONE rc' \""'$f";
done
printf '"'%s\\n' '--- figure executable smoke (write only to process/null?) ---'
\""'${PYTHON:-python}" -m py_compile scripts/plot/plot_scaling_paper_figure.py && echo '"'py_compile: PASS'
printf '%s\\n' '--- documented exact job id status ---'
rg -n -i '26628021|26628196|26628073' docs/performance/scaling/levante_campaign_2026-07-24.md scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 368ms:
--- recovery raw-artifact availability ---
mpas_s8_l0.26628076.log
mpasoc_metis.26600094.log
mpas_s9.26600095.log
ll_2048.26502539.log
atm128.26534060.log
--- all results validity loops ---
scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch
89-
90-echo "=== solo_post: 1x32 GPUs, 24 nodes idle ==="
91-ARM_TAG=solo_post; export ARM_TAG
92-run_arm solo_post || { echo "solo_post FAILED"; rc=1; }
93-
94:echo "=== RESULTS ==="
95-for T in solo_pre rep0 rep1 rep2 rep3 solo_post; do
96-  F="$OUTDIR/$T.jsonl"
97-  "$PY" -c "
98-import json,math,sys
99-try:
100-    d=json.loads(open('$F').readline())
101-    ms=d['steady_median_ms']
102:    assert math.isfinite(ms) and ms > 0, f'non-finite {ms}'
103-except Exception as e:
104-    print('$T: MISSING/INVALID ->', e); sys.exit(1)
105-print(f'$T: {ms:8.2f} ms  {d.get(\"mcells_per_s\",0)/1000:.2f} GC/s')" \
106:    || { echo "$T receipt invalid"; rc=1; }
107-done
108-
109-echo "=== overlap evidence: per-step nodelist + wall window ==="
110-sacct -j "$SLURM_JOB_ID" \
111-  --format=JobID%18,JobName%12,NodeList%45,Start,End,State -P \
112-  || { echo "sacct overlap table UNAVAILABLE"; rc=1; }
113:echo "DONE rc=$rc"; exit $rc
scripts/cluster/scaling_levante/atm_latlon_hundreds.sbatch
42-      --out "$OUTDIR/$4.jsonl" || { echo "$4 FAILED"; rc=1; }
43-}
44-run_arm 2304 4608  96 LL2304_f32_np96
45-run_arm 2304 4608 192 LL2304_f32_np192
46-run_arm 2880 5760 192 LL2880_f32_np192
47:echo "=== RESULTS ==="
48-for T in LL2304_f32_np96 LL2304_f32_np192 LL2880_f32_np192; do
49-  F="$OUTDIR/$T.jsonl"
50-  "$PY" -c "
51-import json,math,sys
52-try:
53-    d=json.loads(open('$F').readline())
54-    ms=d['steady_median_ms']
55:    assert math.isfinite(ms) and ms > 0
56-except Exception as e:
57-    print('$T: MISSING/INVALID ->', e); sys.exit(1)
58-print(f'$T: {ms:8.2f} ms {d.get(\"mcells_per_s\",0)/1000:6.2f} GC/s')" \
59:    || { echo "$T receipt invalid"; rc=1; }
60-done
61:echo "DONE rc=$rc"; exit $rc
scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch
30-      --grid latlon --latlon-2d --mode single --physics moist \
31-      --precision float64 --n-levels 26 --resolution 512 \
32-      --n-warmup 2 --n-timing 6 \
33-      --output-dir "$OUTDIR/np${NP}" < /dev/null || { echo "np$NP FAILED"; rc=1; }
34-done
35:echo "=== RESULTS ==="
36-for NP in 64 128 256 512; do
37-  F=$(find "$OUTDIR/np${NP}" -name "*.json" 2>/dev/null | head -1)
38-  "$PY" -c "
39-import json,math,sys
40-try:
41-    d=json.load(open('$F' or '/nonexistent'))
42-    r=d['results'][-1] if isinstance(d.get('results'),list) else d
43-    ms=r.get('ms_per_step') or r.get('time_per_step_ms')
44:    assert ms is not None and math.isfinite(ms) and ms > 0
45-except Exception as e:
46-    print('np$NP: MISSING/INVALID ->', e); sys.exit(1)
47:print(f'np$NP: {ms:9.2f} ms')" || { echo "np$NP receipt invalid"; rc=1; }
48-done
49:echo "DONE rc=$rc"; exit $rc
--- figure executable smoke (write only to process/null?) ---
/usr/bin/bash: line 7: python: command not found
--- documented exact job id status ---
docs/performance/scaling/levante_campaign_2026-07-24.md:1719:| 26628021 | s9 ensemble contention | 128 GPU (4x32) | lever #1 receipt |
docs/performance/scaling/levante_campaign_2026-07-24.md:1722:| 26628073 | atm lat-lon 2-D pencil r512 np64-512 | 512 CPU ranks | hundreds-of-CPUs lat-lon (wall-pole lane, labelled) |
docs/performance/scaling/levante_campaign_2026-07-24.md:1728:### First hundreds receipt in: lat-lon CPU 2-D pencil to 512 ranks (job 26628073)

exec
/usr/bin/bash -lc "printf '%s\\n' '--- raw METIS receipt fields ---'
rg -n -i -C 2 'cells_per_rank_achieved|cells_per_rank_min|cells_per_rank_max|wet.*min|wet.*max|partition_metrics|steady_median_ms|189\\.82|308\\.96|191\\.99|333\\.39|537\\.91' mpasoc_metis.26600094.log | tail -n 260
printf '%s\\n' '--- raw LL2048 receipt fields ---'
rg -n -i -C 2 'steady_median_ms|5\\.5767|9\\.6015|mcells_per_s|git_sha|n_ranks|65' atm128.26534060.log | tail -n 160
printf '%s\\n' '--- raw s9 SHA fields ---'
rg -n -i -C 2 'git_sha|7151d12a1|unknown|steady_median_ms|12\\.47|9\\.60|11\\.48' mpas_s9.26600095.log | tail -n 160
printf '%s\\n' '--- raw CPU job state / failure surface ---'
rg -n -i -C 1 'slurmstepd: error|error:|traceback|exception|non-finite|FAILED|CANCELLED|OUT OF MEMORY|TIME LIMIT|killed' cpu_ll2d.26628073.log | tail -n 120 || true
printf '%s\\n' '--- raw CPU result JSON excerpts (quoted) ---'
rg -n -i -C 3 'Result: .*latlon_2d.*\\.json|\"n_ranks\"|\"process_count\"|\"decomposition\"' cpu_ll2d.26628073.log | tail -n 100" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 280ms:
--- raw METIS receipt fields ---
448-[1785518297.249801] [l20119:1439095:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
449-[1785518297.254104] [l20119:1439079:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
450:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 7, "n_ranks": 32, "n_cells": 163842, "n_edges": 491520, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12255.1, "steady_median_ms": 189.8206, "step_latency_gate_loop_ms": 187.13, "step_latency_gate_loop_min_ms": 185.51, "per_step_ms": [12255.1, 188.4, 187.4, 187.6, 187.4, 186.3, 186.9, 186.0, 188.5, 186.6, 185.5, 190.5], "cells": 3276840, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 183.7, "scan_compile_ms": 13431.3, "step_latency_ms": 187.99, "block_ms": [1516.95, 1516.98], "parallel_block_ms": [1518.81, 1518.32], "fused_step_ms": 189.8206, "rank_imbalance": 1.0011, "rank_imbalance_per_block": [1.0012, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 3264640, "wet_cell_levels_per_device": 102020.0, "wet_fraction": 0.9962769009167368, "wet_equals_total": false, "wet_cell_levels_per_device_min": 100740, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:19:09.693551+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L7", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102401, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 628, "owned_halo_ratio": 0.12265625, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 628, "owned_send_cells": 626, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 20654, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
451:[mpas-ocean np=32 L7 nCells=163842 nlev=20 solver=explicit_substep halo=in_step] compile=12255.1ms fused=189.8206ms/step (probe_latency=187.99ms) gate_loop_latency=187.13ms/step
452-[1785518297.289277] [l20119:1439070:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
453---- s8 np=128 partition=geometric dist=block:cyclic ---
--
2147-[1785518353.342035] [l20119:1440310:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2148-[1785518353.342454] [l20119:1440297:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2149:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "geometric", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12628.0, "steady_median_ms": 308.9581, "step_latency_gate_loop_ms": 311.3, "step_latency_gate_loop_min_ms": 308.75, "per_step_ms": [12628.0, 312.0, 311.7, 310.9, 309.9, 310.0, 312.0, 311.9, 312.1, 308.7, 309.7, 311.7], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 308.8, "scan_compile_ms": 14719.4, "step_latency_ms": 311.988, "block_ms": [2471.77, 2467.17], "parallel_block_ms": [2472.65, 2470.68], "fused_step_ms": 308.9581, "rank_imbalance": 1.0008, "rank_imbalance_per_block": [1.0008, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96040, "wet_cell_levels_per_device_max": 102420}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:20:07.513962+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5120, "n_halo_cells": 658, "owned_halo_ratio": 0.128515625, "n_neighbor_ranks": 7, "messages_per_exchange": 7, "halo_recv_cells": 658, "owned_send_cells": 661, "cells_per_rank_min": 5120, "cells_per_rank_max": 5121, "edge_cut_total": 80510, "max_neighbor_ranks": 8}, "extra": {"partition_method": "geometric", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
2150-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=12628.0ms fused=308.9581ms/step (probe_latency=311.988ms) gate_loop_latency=311.30ms/step
2151-[1785518353.372656] [l20119:1440282:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
2685-[1785518411.080442] [l20119:1441486:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2686-[1785518411.064649] [l20119:1441490:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2687:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 7, "n_ranks": 32, "n_cells": 163842, "n_edges": 491520, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 12579.2, "steady_median_ms": 191.9856, "step_latency_gate_loop_ms": 190.29, "step_latency_gate_loop_min_ms": 188.92, "per_step_ms": [12579.2, 193.6, 191.0, 190.2, 188.9, 191.2, 189.7, 189.8, 188.9, 192.5, 191.4, 190.4], "cells": 3276840, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 186.8, "scan_compile_ms": 14192.4, "step_latency_ms": 188.993, "block_ms": [1534.07, 1532.56], "parallel_block_ms": [1535.85, 1535.92], "fused_step_ms": 191.9856, "rank_imbalance": 1.0012, "rank_imbalance_per_block": [1.0012, 1.0011], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 3264640, "wet_cell_levels_per_device": 102020.0, "wet_fraction": 0.9962769009167368, "wet_equals_total": false, "wet_cell_levels_per_device_min": 96800, "wet_cell_levels_per_device_max": 102720}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:20:51.591156+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L7", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 32, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102401, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5133, "n_halo_cells": 549, "owned_halo_ratio": 0.10695499707773232, "n_neighbor_ranks": 6, "messages_per_exchange": 6, "halo_recv_cells": 549, "owned_send_cells": 556, "cells_per_rank_min": 5100, "cells_per_rank_max": 5145, "edge_cut_total": 18784, "max_neighbor_ranks": 7}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
2688-[mpas-ocean np=32 L7 nCells=163842 nlev=20 solver=explicit_substep halo=in_step] compile=12579.2ms fused=191.9856ms/step (probe_latency=188.993ms) gate_loop_latency=190.29ms/step
2689-[1785518411.064180] [l20119:1441477:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
4385-[1785518454.766060] [l20119:1442707:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
4386-[1785518454.774753] [l20119:1442679:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
4387:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 13554.5, "steady_median_ms": 333.39, "step_latency_gate_loop_ms": 348.11, "step_latency_gate_loop_min_ms": 345.44, "per_step_ms": [13554.5, 350.4, 352.4, 348.9, 348.3, 349.4, 348.0, 347.5, 347.1, 348.5, 345.4, 347.6], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 343.0, "scan_compile_ms": 15677.5, "step_latency_ms": 343.721, "block_ms": [2663.79, 2664.88], "parallel_block_ms": [2666.74, 2667.5], "fused_step_ms": 333.39, "rank_imbalance": 1.0007, "rank_imbalance_per_block": [1.0007, 1.0008], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:21:47.977228+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
4388:[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=13554.5ms fused=333.39ms/step (probe_latency=343.721ms) gate_loop_latency=348.11ms/step
4389-[1785518454.797213] [l20119:1442677:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
4390-[1785518454.772658] [l20119:1442708:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
6253-[1785518521.996105] [l20130:52583:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6254-[1785518521.884325] [l20119:1443901:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6255:{"component": "ocean", "grid": "voronoi", "mode": "strong", "subdivision": 8, "n_ranks": 128, "n_cells": 655362, "n_edges": 1966080, "nlev": 20, "partition_method": "metis", "steps": 12, "dt": 60.0, "platform": "cpu", "compile_ms": 14329.9, "steady_median_ms": 537.905, "step_latency_gate_loop_ms": 557.24, "step_latency_gate_loop_min_ms": 553.39, "per_step_ms": [14329.9, 558.5, 561.4, 554.0, 558.4, 555.1, 560.6, 569.8, 559.2, 554.6, 556.1, 553.4], "cells": 13107240, "cells_per_rank_achieved": 5120, "compile_prewarmed_by_parity_ref": false, "barotropic_solver": "explicit_substep", "halo_refresh": "in_step", "stage_halo_correct": true, "stage_halo_note": "per-step packed entry refresh + in-step stage-frontier refreshes (make_mpas_ocean_halo_refresh) \u2014 stage-correct; see docs/performance/scaling/mpas_ocean_distributed_stage_audit.md", "fused": {"compile_ms": 547.5, "scan_compile_ms": 17807.2, "step_latency_ms": 547.997, "block_ms": [4295.08, 4293.81], "parallel_block_ms": [4303.79, 4302.69], "fused_step_ms": 537.905, "rank_imbalance": 1.0021, "rank_imbalance_per_block": [1.002, 1.0023], "block_steps": 8, "n_blocks": 2, "probe_steps": 3}, "wet_cell": {"wet_cell_levels": 13057600, "wet_cell_levels_per_device": 102012.5, "wet_fraction": 0.9962127801123654, "wet_equals_total": false, "wet_cell_levels_per_device_min": 78000, "wet_cell_levels_per_device_max": 102880}, "solver_iters": null, "solver_iters_mode": "explicit_substep (no iterative solve)", "zero_forcing_probe_residual": null, "zero_forcing_probe_measured": false, "residual_reason": "explicit_substep barotropic has no iterative solve \u2014 no solver residual exists to measure", "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-31T17:23:14.881483+00:00", "grid": "voronoi", "component": "ocean", "resolution": "L8", "n_levels": 20, "precision": "float64", "decomposition": "cell_partition", "solver_variant": "mpas_ocean_explicit_substep", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 0, "device_count": 1, "process_count": 1, "devices_per_rank": 1, "cells_per_rank": 102400, "backend": "cpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "mpi4jax", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l20119.lvt.dkrz.de", "slurm_job_id": "26600094", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": false, "gpu_direct_active": false, "host_staged_halo": false, "partition_metrics": {"n_owned_cells": 5113, "n_halo_cells": 562, "owned_halo_ratio": 0.10991590064541365, "n_neighbor_ranks": 4, "messages_per_exchange": 4, "halo_recv_cells": 562, "owned_send_cells": 553, "cells_per_rank_min": 5093, "cells_per_rank_max": 5144, "edge_cut_total": 75748, "max_neighbor_ranks": 8}, "extra": {"partition_method": "metis", "steps": 12, "warmup": 2, "parity_gate": false, "check_conservation": false, "halo_refresh": "in_step", "barotropic_solver": "explicit_substep", "block_steps": 8, "blocks": 2, "probe_steps": 3}}}
6256-[mpas-ocean np=128 L8 nCells=655362 nlev=20 solver=explicit_substep halo=in_step] compile=14329.9ms fused=537.905ms/step (probe_latency=547.997ms) gate_loop_latency=557.24ms/step
6257-[1785518521.623596] [l20119:1443880:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
--
6288-[1785518521.792657] [l20119:1443909:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
6289-=== RESULTS (A-D all hold 5120 cells/rank) ===
6290:A_s7np32_geometric:   189.82 ms  method=geometric ranks=32
6291:B_s8np128_geometric:   308.96 ms  method=geometric ranks=128
6292:C_s7np32_metis:   191.99 ms  method=metis ranks=32
6293:D_s8np128_metis:   333.39 ms  method=metis ranks=128
6294-E_s8np128_metis_blockblock:   537.90 ms  method=metis ranks=128
6295-DONE rc=0
--- raw LL2048 receipt fields ---
1:outdir=/scratch/b/b381103/legoesm_scaling/atm128_j26534060
2-=== atm LL2048x4096 @128 f32 ===
3-[early_init] WARNING: multi-node launch with no NCCL net plugin visible (libnccl-net*/NCCL_NET_PLUGIN): cross-node collectives will likely run on TCP SOCKETS (correct but slow — the documented Derecho socket-bound shape). Verify with NCCL_DEBUG=INFO; build/load the aws-ofi-nccl plugin for fabric speed.
4:{"mode": "strong", "n_devices": 128, "n_lat": 2048, "n_lon": 4096, "nlev": 26, "physics": "none", "steps": 12, "platform": "gpu", "n_processes": 128, "multicontroller": true, "segment_mode": false, "segment_steps": null, "finite_ok": null, "valid": true, "completed_blocks": null, "steady_median_ms": 5.5767, "cells": 218103808, "compile_ms": 7327.1, "scan_compile_ms": 8880.2, "step_latency_ms": 6.791, "block_ms": [66.6, 66.6], "parallel_block_ms": [66.89, 66.95], "fused_step_ms": 5.5767, "rank_imbalance": 1.0036, "rank_imbalance_per_block": [1.0034, 1.0038], "block_steps": 12, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 2048, "n_levels": 26, "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 60.0, "time_per_step_ms": 5.5767, "total_cells": 218103808, "sypd": 29.456676390683754, "mcells_per_s": 39109.83341402622, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "no analytic halo-message census for the atm latlon step yet (audit item 4 follow-up) \u2014 comm fields null, not fabricated", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0036}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-29T23:55:01.559216+00:00", "grid": "latlon", "component": "atmosphere", "resolution": "2048x4096", "n_levels": 26, "precision": "float32", "decomposition": "band", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 1703936, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26534060", "git_sha": "70f3ce636", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"physics": "none", "steps": 12, "warmup": 3, "multicontroller": true, "segment_mode": false, "segment_steps": null, "geometry_bytes_per_device": {"n_geometry_fields": 13, "replicated_per_device_bytes": 169920000, "sharded_per_device_bytes": 1327500}, "finite_ok": null, "valid": true, "completed_blocks": null, "nccl": {"nccl_net_plugin": null, "nccl_net": null, "nccl_ib_disable": "0", "nccl_ib_hca": "mlx5", "nccl_socket_ifname": "ib0", "nccl_debug": null, "n_nodes_declared": 32, "missing_net_plugin_multi_node": true}, "cells_per_device": 1703936}}}
5-[nd=128 strong 2048x4096x26] compile=7327.1ms fused=5.577ms/step latency=6.791ms/step imbalance=1.0036 blocks=[66.6, 66.6]
6-W0730 01:55:03.445913 2170983 pjrt_client.cc:1604] WatchTasksAsync failed for task 2: CANCELLED: CANCELLED
--
22-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
23-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
24:W0730 01:55:03.446233 1145651 pjrt_client.cc:1604] WatchTasksAsync failed for task 24: CANCELLED: CANCELLED
25-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
26-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
--
40-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
41-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
42:W0730 01:55:03.446239 1145650 pjrt_client.cc:1604] WatchTasksAsync failed for task 27: CANCELLED: CANCELLED
43-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
44-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
--
49-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
50-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
51:W0730 01:55:03.446257 1145652 pjrt_client.cc:1604] WatchTasksAsync failed for task 26: CANCELLED: CANCELLED
52-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
53-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
--
79-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
80-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
81:W0730 01:55:03.446656  335154 pjrt_client.cc:1604] WatchTasksAsync failed for task 103: CANCELLED: CANCELLED
82-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
83-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
--
277-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
278-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
279:W0730 01:55:03.449086 1018198 pjrt_client.cc:1604] WatchTasksAsync failed for task 65: CANCELLED: CANCELLED
280-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
281-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
--
289-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
290-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
291:W0730 01:55:03.449037 1050650 pjrt_client.cc:1604] WatchTasksAsync failed for task 60: CANCELLED: CANCELLED
292-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
293-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
--
301-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
302-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
303:W0730 01:55:03.449046 1050655 pjrt_client.cc:1604] WatchTasksAsync failed for task 61: CANCELLED: CANCELLED
304-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
305-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
--
310-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
311-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
312:W0730 01:55:03.449044 1050654 pjrt_client.cc:1604] WatchTasksAsync failed for task 62: CANCELLED: CANCELLED
313-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
314-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
--
319-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
320-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
321:W0730 01:55:03.449043 1050651 pjrt_client.cc:1604] WatchTasksAsync failed for task 63: CANCELLED: CANCELLED
322-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
323-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
--
390-=== atm LL2048x4096 @128 f64 ===
391-[early_init] WARNING: multi-node launch with no NCCL net plugin visible (libnccl-net*/NCCL_NET_PLUGIN): cross-node collectives will likely run on TCP SOCKETS (correct but slow — the documented Derecho socket-bound shape). Verify with NCCL_DEBUG=INFO; build/load the aws-ofi-nccl plugin for fabric speed.
392:{"mode": "strong", "n_devices": 128, "n_lat": 2048, "n_lon": 4096, "nlev": 26, "physics": "none", "steps": 12, "platform": "gpu", "n_processes": 128, "multicontroller": true, "segment_mode": false, "segment_steps": null, "finite_ok": null, "valid": true, "completed_blocks": null, "steady_median_ms": 9.6015, "cells": 218103808, "compile_ms": 7124.4, "scan_compile_ms": 9596.2, "step_latency_ms": 8258.994, "block_ms": [115.17, 115.1], "parallel_block_ms": [115.24, 115.19], "fused_step_ms": 9.6015, "rank_imbalance": 1.0004, "rank_imbalance_per_block": [1.0003, 1.0005], "block_steps": 12, "n_blocks": 2, "probe_steps": 3, "grid_type": "latlon", "resolution": 2048, "n_levels": 26, "precision": "float64", "physics_level": "none", "backend": "gpu", "dt_seconds": 60.0, "time_per_step_ms": 9.6015, "total_cells": 218103808, "sypd": 17.108894154863933, "mcells_per_s": 22715.59735458001, "full_state_gathers_per_step": 0, "halo_bytes_is_lower_bound": null, "halo_messages_per_step": null, "halo_bytes_per_message": null, "halo_bytes_per_step": null, "comm_scope_note": "no analytic halo-message census for the atm latlon step yet (audit item 4 follow-up) \u2014 comm fields null, not fabricated", "t_bound_ms": null, "measured_over_bound": null, "bound_calibrated": false, "bound_incomplete_reason": ["single_device_fused_step_ms", "halo_messages_per_step", "halo_bytes_per_step", "n_reductions_per_step"], "bound_ingredients": {"compute_ms": null, "comm_ms": null, "reduction_ms": null, "imbalance_ms": null, "launch_host_ms": 0.0, "latency_us": 25.0, "bandwidth_GBs": 10.0, "halo_messages_per_step": null, "halo_bytes_per_step": null, "n_reductions_per_step": null, "rank_imbalance": 1.0004}, "metadata": {"schema_version": 2, "timestamp_utc": "2026-07-29T23:56:25.473612+00:00", "grid": "latlon", "component": "atmosphere", "resolution": "2048x4096", "n_levels": 26, "precision": "float64", "decomposition": "band", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 1703936, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "1", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26534060", "git_sha": "70f3ce636", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"physics": "none", "steps": 12, "warmup": 3, "multicontroller": true, "segment_mode": false, "segment_steps": null, "geometry_bytes_per_device": {"n_geometry_fields": 13, "replicated_per_device_bytes": 169920000, "sharded_per_device_bytes": 1327500}, "finite_ok": null, "valid": true, "completed_blocks": null, "nccl": {"nccl_net_plugin": null, "nccl_net": null, "nccl_ib_disable": "0", "nccl_ib_hca": "mlx5", "nccl_socket_ifname": "ib0", "nccl_debug": null, "n_nodes_declared": 32, "missing_net_plugin_multi_node": true}, "cells_per_device": 1703936}}}
393-[nd=128 strong 2048x4096x26] compile=7124.4ms fused=9.601ms/step latency=8258.994ms/step imbalance=1.0004 blocks=[115.17, 115.1]
394:W0730 01:56:25.825652 2171839 pjrt_client.cc:1604] WatchTasksAsync failed for task 1: CANCELLED: CANCELLED
395-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
396-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
--
443-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
444-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
445:W0730 01:56:25.826165  358967 pjrt_client.cc:1604] WatchTasksAsync failed for task 17: CANCELLED: CANCELLED
446-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
447-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
--
467-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
468-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
469:W0730 01:56:25.826582 1669104 pjrt_client.cc:1604] WatchTasksAsync failed for task 59: CANCELLED: CANCELLED
470-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
471-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
472:W0730 01:56:25.826542  402280 pjrt_client.cc:1604] WatchTasksAsync failed for task 88: CANCELLED: CANCELLED
473-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
474-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
475:W0730 01:56:25.826531  402308 pjrt_client.cc:1604] WatchTasksAsync failed for task 89: CANCELLED: CANCELLED
476-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
477-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
--
479-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
480-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
481:W0730 01:56:25.826520  402278 pjrt_client.cc:1604] WatchTasksAsync failed for task 90: CANCELLED: CANCELLED
482-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
483-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
--
485-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
486-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
487:W0730 01:56:25.826533  402294 pjrt_client.cc:1604] WatchTasksAsync failed for task 91: CANCELLED: CANCELLED
488-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
489-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
490:W0730 01:56:25.826479 2361650 pjrt_client.cc:1604] WatchTasksAsync failed for task 14: CANCELLED: CANCELLED
491-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
492-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
493:W0730 01:56:25.826538 1051436 pjrt_client.cc:1604] WatchTasksAsync failed for task 61: CANCELLED: CANCELLED
494-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
495-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
496:W0730 01:56:25.826579  649075 pjrt_client.cc:1604] WatchTasksAsync failed for task 123: CANCELLED: CANCELLED
497-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
498-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
499:W0730 01:56:25.826599 1840426 pjrt_client.cc:1604] WatchTasksAsync failed for task 21: CANCELLED: CANCELLED
500-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
501-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
502:W0730 01:56:25.826471 2361651 pjrt_client.cc:1604] WatchTasksAsync failed for task 15: CANCELLED: CANCELLED
503-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
504-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
505:W0730 01:56:25.826546 1051440 pjrt_client.cc:1604] WatchTasksAsync failed for task 63: CANCELLED: CANCELLED
506-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
507-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
--
617-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
618-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
619:W0730 01:56:25.827365  906280 pjrt_client.cc:1604] WatchTasksAsync failed for task 36: CANCELLED: CANCELLED
620-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
621-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
--
638-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
639-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
640:W0730 01:56:25.827365 2113937 pjrt_client.cc:1604] WatchTasksAsync failed for task 5: CANCELLED: CANCELLED
641-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
642-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
--
668-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
669-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
670:W0730 01:56:25.827489 1018990 pjrt_client.cc:1604] WatchTasksAsync failed for task 65: CANCELLED: CANCELLED
671-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
672-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
--
680-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
681-:UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1} [type.googleapis.com/tensorflow.DerivedStatus='']
682:W0730 01:56:25.827565  335939 pjrt_client.cc:1604] WatchTasksAsync failed for task 103: CANCELLED: CANCELLED
683-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
684-:UNKNOWN:Error received from peer  {grpc_status:1, grpc_message:"CANCELLED"} [type.googleapis.com/tensorflow.DerivedStatus='']
--
790-*                       We hope you enjoyed the DKRZ supercomputer LEVANTE ... *
791-*
792:* JobID            : 26534060
793-* JobName          : atm128                                            
794-* Account          : bb1596_gpu
--- raw s9 SHA fields ---
927-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/numpy/array_methods.py:125: UserWarning: Explicitly requested dtype float64 requested in astype is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
928-  return lax_numpy.astype(self, dtype, copy=copy, device=device)
929:{"component": "mpas_atm", "subdivision": 9, "n_devices": 128, "n_cells": 2621568, "n_edges": 7864320, "nlev": 26, "partition_method": "sfc", "physics": "none", "lloyd_iterations": 0, "halo_strategy_requested": "auto", "halo_strategy_effective": "ppermute", "steps": 12, "dt": 30.0, "platform": "gpu", "n_processes": 128, "multicontroller": true, "compile_ms": 7532.3, "steady_median_ms": 11.48, "steady_min_ms": 11.37, "per_step_ms": [7532.3, 15.4, 12.4, 11.7, 11.5, 11.6, 11.4, 11.4, 11.5, 11.4, 11.5, 11.4], "cells": 68160768, "hlo_collective_permutes": null, "hlo_collectives": null, "grid_type": "icosahedral", "resolution": 9, "n_levels": 26, "mode": "strong", "precision": "float32", "physics_level": "none", "backend": "gpu", "dt_seconds": 30.0, "time_per_step_ms": 11.482901012641378, "total_cells": 68160768, "sypd": 7.152854798934616, "mcells_per_s": 5935.8491312398055, "metadata": {"schema_version": 2, "timestamp_utc": "2026-08-01T03:45:45.637722+00:00", "grid": "icosahedral", "component": "atmosphere", "resolution": "L9", "n_levels": 26, "precision": "float32", "decomposition": "cell_partition", "solver_variant": "n/a", "solver_residual": null, "conservation_drift": null, "scaling_kind": "strong", "n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 532506, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "transport": "nccl", "virtual_cpu_devices": false, "launcher": "slurm", "hostname": "l50000.lvt.dkrz.de", "slurm_job_id": "26600095", "git_sha": "7151d12a1", "gpu_direct_requested": false, "mpi4jax_cuda_support": null, "gpu_direct_active": false, "host_staged_halo": false, "extra": {"partition_method": "sfc", "physics": "none", "steps": 12, "multicontroller": true, "cells_per_device": 532506}}}
930:[mpas nd=128 L9 nCells=2621568 nlev=26] compile=7532.3ms steady_median=11.48ms/step (per-step: [7532.3, 15.4, 12.4, 11.7, 11.5, 11.6, 11.4, 11.4, 11.5, 11.4, 11.5, 11.4])
931:W0801 05:45:51.458955  432147 pjrt_client.cc:1604] WatchTasksAsync failed for task 75: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
932-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
933::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
934:W0801 05:45:51.459394  335558 pjrt_client.cc:1604] WatchTasksAsync failed for task 98: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
935-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
936::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
937:W0801 05:45:51.463212  189987 pjrt_client.cc:1604] WatchTasksAsync failed for task 120: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
938-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
939::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
940:W0801 05:45:51.465135  363030 pjrt_client.cc:1604] WatchTasksAsync failed for task 80: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
941-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
942::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
943:W0801 05:45:51.467905  233374 pjrt_client.cc:1604] WatchTasksAsync failed for task 126: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
944-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
945::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
946:W0801 05:45:51.467389  310755 pjrt_client.cc:1604] WatchTasksAsync failed for task 9: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
947-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
948::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
949:W0801 05:45:51.467647  234848 pjrt_client.cc:1604] WatchTasksAsync failed for task 103: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
950-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
951::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
952:W0801 05:45:51.469115  360945 pjrt_client.cc:1604] WatchTasksAsync failed for task 17: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
953-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
954::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
955:W0801 05:45:51.469791  258426 pjrt_client.cc:1604] WatchTasksAsync failed for task 64: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
956-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
957::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
958:W0801 05:45:51.472912  328767 pjrt_client.cc:1604] WatchTasksAsync failed for task 115: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
959-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
960::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
961:W0801 05:45:51.473454  250909 pjrt_client.cc:1604] WatchTasksAsync failed for task 58: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
962-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
963::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
964:W0801 05:45:51.476029  360949 pjrt_client.cc:1604] WatchTasksAsync failed for task 18: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
965-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
966::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
967:W0801 05:45:51.478010  233031 pjrt_client.cc:1604] WatchTasksAsync failed for task 108: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
968-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
969::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
970:W0801 05:45:51.478637  266226 pjrt_client.cc:1604] WatchTasksAsync failed for task 4: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
971-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
972::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
973:W0801 05:45:51.480620  189984 pjrt_client.cc:1604] WatchTasksAsync failed for task 123: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
974-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
975::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
976:W0801 05:45:51.486953  553829 pjrt_client.cc:1604] WatchTasksAsync failed for task 87: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
977-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
978::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
979:W0801 05:45:51.493879  739060 pjrt_client.cc:1604] WatchTasksAsync failed for task 1: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
980-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
981::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
982:W0801 05:45:51.498316  234850 pjrt_client.cc:1604] WatchTasksAsync failed for task 101: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
983-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
984::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
985:W0801 05:45:51.502108  231734 pjrt_client.cc:1604] WatchTasksAsync failed for task 95: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
986-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
987::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
988:W0801 05:45:51.508279  221320 pjrt_client.cc:1604] WatchTasksAsync failed for task 68: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
989-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
990::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
991:W0801 05:45:51.509660  468601 pjrt_client.cc:1604] WatchTasksAsync failed for task 78: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
992-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
993::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
994:W0801 05:45:51.513005  553826 pjrt_client.cc:1604] WatchTasksAsync failed for task 84: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
995-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
996::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
997:W0801 05:45:51.513900  328768 pjrt_client.cc:1604] WatchTasksAsync failed for task 112: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
998-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
999::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
1000:W0801 05:45:51.517278  258430 pjrt_client.cc:1604] WatchTasksAsync failed for task 67: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1001-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1002::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
1003:W0801 05:45:51.518335  468605 pjrt_client.cc:1604] WatchTasksAsync failed for task 79: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1004-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1005::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
1006:W0801 05:45:51.518437  270364 pjrt_client.cc:1604] WatchTasksAsync failed for task 47: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1007-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1008::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
1009:W0801 05:45:51.525090  363029 pjrt_client.cc:1604] WatchTasksAsync failed for task 82: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1010-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1011::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
1012:W0801 05:45:51.526087  233380 pjrt_client.cc:1604] WatchTasksAsync failed for task 127: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1013-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1014::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
1015:W0801 05:45:51.525716  243369 pjrt_client.cc:1604] WatchTasksAsync failed for task 51: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1016-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1017::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
1018:W0801 05:45:51.530836  335557 pjrt_client.cc:1604] WatchTasksAsync failed for task 97: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1019-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1020::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
1021:W0801 05:45:51.533411  263353 pjrt_client.cc:1604] WatchTasksAsync failed for task 14: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1022-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1023::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
1024:W0801 05:45:51.537465  270360 pjrt_client.cc:1604] WatchTasksAsync failed for task 44: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1025-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1026::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
1027:W0801 05:45:51.538029 2630926 pjrt_client.cc:1604] WatchTasksAsync failed for task 117: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1028-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1029::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
1030:W0801 05:45:51.538125  212267 pjrt_client.cc:1604] WatchTasksAsync failed for task 29: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1031-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1032::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
1033:W0801 05:45:51.540896  739059 pjrt_client.cc:1604] WatchTasksAsync failed for task 3: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1034-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1035::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
1036:W0801 05:45:51.540156  257521 pjrt_client.cc:1604] WatchTasksAsync failed for task 33: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1037-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1038::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
1039:W0801 05:45:51.548102  263851 pjrt_client.cc:1604] WatchTasksAsync failed for task 53: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1040-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1041::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
1042:W0801 05:45:51.555265  310749 pjrt_client.cc:1604] WatchTasksAsync failed for task 11: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1043-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1044::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
1045:W0801 05:45:51.555652  243374 pjrt_client.cc:1604] WatchTasksAsync failed for task 49: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1046-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1047::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
1048:W0801 05:45:51.556661  220379 pjrt_client.cc:1604] WatchTasksAsync failed for task 62: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1049-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1050::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
1051:W0801 05:45:51.563204  221325 pjrt_client.cc:1604] WatchTasksAsync failed for task 70: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1052-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1053::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
1054:W0801 05:45:51.586277  189983 pjrt_client.cc:1604] WatchTasksAsync failed for task 121: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1055-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1056::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
1057:W0801 05:45:51.586963  263848 pjrt_client.cc:1604] WatchTasksAsync failed for task 54: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1058-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1059::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
1060:W0801 05:45:51.613302  231736 pjrt_client.cc:1604] WatchTasksAsync failed for task 94: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1061-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1062::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
1063:W0801 05:45:51.621208  263853 pjrt_client.cc:1604] WatchTasksAsync failed for task 55: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1064-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1065::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
1066:W0801 05:45:51.625574  292228 pjrt_client.cc:1604] WatchTasksAsync failed for task 23: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1067-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1068::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
1069:W0801 05:45:51.696430  220378 pjrt_client.cc:1604] WatchTasksAsync failed for task 60: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1070-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1071::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
1072:W0801 05:45:51.719510  263847 pjrt_client.cc:1604] WatchTasksAsync failed for task 52: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1073-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1074::UNKNOWN:Error received from peer  {grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused", grpc_status:14}
1075:W0801 05:45:51.727143  292233 pjrt_client.cc:1604] WatchTasksAsync failed for task 20: UNAVAILABLE: failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused
1076-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1077::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"failed to connect to all addresses; last error: UNKNOWN: ipv4:10.5.0.1:62111: Failed to connect to remote host: Connection refused"}
1078-W0801 05:45:56.452862  739064 pjrt_client.cc:1604] WatchTasksAsync failed for task 0: UNAVAILABLE: Cancelling all calls
1079-Additional GRPC error information from remote target coordination_service while calling /xla.coordination.CoordinationService/WatchTasks:
1080::UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"Cancelling all calls"}
1081-=== RESULTS (cells/GPU: 81.9k / 41.0k / 20.5k) ===
1082:np32:    12.47 ms  5.47 GC/s
1083:np64:     9.60 ms  7.10 GC/s
1084:np128:    11.48 ms  5.94 GC/s
1085-DONE rc=0
1086-
--- raw CPU job state / failure surface ---
--- raw CPU result JSON excerpts (quoted) ---
398-[1785668570.922438] [l30516:325532:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
399-[1785668570.895224] [l30516:325545:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
400-          latlon |        moist |    64 ranks | 512   | L26 | dt=     0 |    297.57 ms/step | SYPD=   0.002 |      45.8 Mcells/s
401:  Result: /scratch/b/b381103/legoesm_scaling/cpu_ll2d_j26628073/np64/latlon_moist_single/latlon_2d_moist_single_r512_n64_float64.json
402-[1785668570.897366] [l30516:325518:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
403---- atm latlon 2-D pencil r512 f64 np=128 ---
404-[l30523.lvt.dkrz.de:374384] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
--
1183-[1785668746.185832] [l30518:383898:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
1184-[1785668746.136911] [l30524:107429:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
1185-          latlon |        moist |   128 ranks | 512   | L26 | dt=     0 |    161.03 ms/step | SYPD=   0.004 |      84.7 Mcells/s
1186:  Result: /scratch/b/b381103/legoesm_scaling/cpu_ll2d_j26628073/np128/latlon_moist_single/latlon_2d_moist_single_r512_n128_float64.json
1187-[1785668746.194533] [l30518:383876:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
1188---- atm latlon 2-D pencil r512 f64 np=256 ---
1189-[l30528.lvt.dkrz.de:481958] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
--
2736-[1785668889.206648] [l30538:414331:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2737-[1785668889.291672] [l30531:280899:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2738-          latlon |        moist |   256 ranks | 512   | L26 | dt=     0 |     72.06 ms/step | SYPD=   0.009 |     189.2 Mcells/s
2739:  Result: /scratch/b/b381103/legoesm_scaling/cpu_ll2d_j26628073/np256/latlon_moist_single/latlon_2d_moist_single_r512_n256_float64.json
2740-[1785668889.253984] [l30526:299988:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2741---- atm latlon 2-D pencil r512 f64 np=512 ---
2742-[l30543.lvt.dkrz.de:524801] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
--
5825-[1785669029.413795] [l30517:783151:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
5826-[1785669029.433460] [l30530:217116:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
5827-          latlon |        moist |   512 ranks | 512   | L26 | dt=     0 |     44.78 ms/step | SYPD=   0.014 |     304.4 Mcells/s
5828:  Result: /scratch/b/b381103/legoesm_scaling/cpu_ll2d_j26628073/np512/latlon_moist_single/latlon_2d_moist_single_r512_n512_float64.json
5829-[1785669029.363467] [l30516:326365:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
5830-=== RESULTS ===
5831-np64:    297.57 ms

exec
/usr/bin/bash -lc "printf '%s\\n' '--- exact s9 JSON provenance extracts ---'
rg -o '\"n_ranks\": ?(32|64|128)["'^'"\\n]{0,260}|\"git_sha\": ?\"["'^"]+"|"steady_median_ms": ?[0-9.]+'"' mpas_s9.26600095.log | tail -n 80
printf '%s\\n' '--- raw LL JSON with full precision ---'
rg -o '\"steady_median_ms\": ?[0-9.]+|\"mcells_per_s\": ?[0-9.]+|\"precision\": ?\"["'^"]+"|"n_ranks": ?[0-9]+'"' atm128.26534060.log | tail -n 80
printf '%s\\n' '--- CPU log MPI status context concise ---'
rg -n -m 3 'Detected versions outside legoESM|VM_UNMAP' cpu_ll2d.26628073.log
printf '%s\\n' '--- declarations about MPI strict defaults ---'
rg -n -C 2 'LEGOESM_MPI_STRICT_COMPAT' scripts/cluster/scaling_levante/_env.sh scripts/cluster/scaling_levante/atm_latlon2d_cpu_hundreds.sbatch scripts/bench/run_cpu_mpi_scaling.py packages/core/legoesm/parallel/reductions.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 341ms:
--- exact s9 JSON provenance extracts ---
"steady_median_ms": 12.47
"n_ranks": 32, "n_gpus": 32, "device_count": 32, "process_count": 32, "devices_per_rank": 1, "cells_per_rank": 2130024, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "
"git_sha": "unknown"
"steady_median_ms": 9.6
"n_ranks": 64, "n_gpus": 64, "device_count": 64, "process_count": 64, "devices_per_rank": 1, "cells_per_rank": 1065012, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"}, "
"git_sha": "7151d12a1"
"steady_median_ms": 11.48
"n_ranks": 128, "n_gpus": 128, "device_count": 128, "process_count": 128, "devices_per_rank": 1, "cells_per_rank": 532506, "backend": "gpu", "precision_knobs": {"JAX_ENABLE_X64": "0", "LEGOESM_VMIX_F32_SOLVE": "0", "LEGOESM_BAROCLINIC_F32": "0", "LEGOESM_ENABLE_TF32": "0"},
"git_sha": "7151d12a1"
--- raw LL JSON with full precision ---
"steady_median_ms": 5.5767
"precision": "float32"
"mcells_per_s": 39109.83341402622
"precision": "float32"
"n_ranks": 128
"steady_median_ms": 9.6015
"precision": "float64"
"mcells_per_s": 22715.59735458001
"precision": "float64"
"n_ranks": 128
--- CPU log MPI status context concise ---
3:[l30517.lvt.dkrz.de:782347] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
4:[l30516.lvt.dkrz.de:325538] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
5:[l30517.lvt.dkrz.de:782354] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
--- declarations about MPI strict defaults ---
packages/core/legoesm/parallel/reductions.py-229-        + ". MPI execution may fail or produce incorrect results. "
packages/core/legoesm/parallel/reductions.py-230-        + _MPI4JAX_FFI_MIGRATION_NOTE + " "
packages/core/legoesm/parallel/reductions.py:231:        "Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or "
packages/core/legoesm/parallel/reductions.py-232-        + fix + "."
packages/core/legoesm/parallel/reductions.py-233-    )
--
packages/core/legoesm/parallel/reductions.py-383-        jax.__version__,
packages/core/legoesm/parallel/reductions.py-384-        mpi4jax.__version__,
packages/core/legoesm/parallel/reductions.py:385:        strict=_env_flag_true("LEGOESM_MPI_STRICT_COMPAT"),
packages/core/legoesm/parallel/reductions.py-386-    )
packages/core/legoesm/parallel/reductions.py-387-    check_mpi4jax_transport(mpi4jax)

exec
/usr/bin/bash -lc "printf '%s\\n' '--- CPU dt argument and mode handling ---'
rg -n -C 4 -- '--dt|args\\.dt|dt=' scripts/bench/run_cpu_mpi_scaling.py | tail -n 280
printf '%s\\n' '--- main run / output path ---'
nl -ba scripts/bench/run_cpu_mpi_scaling.py | sed -n '1120,1325p;1520,1605p;1760,1840p'
printf '%s\\n' '--- timing result formatter ---'
rg -n -C 5 'Timing:|ms/step|SYPD' scripts/bench/run_cpu_mpi_scaling.py | tail -n 200
printf '%s\\n' '--- 2D step’s dt caveat in source ---'
rg -n -C 3 'dt.*0|dt=0|dt is|dt or' scripts/bench/run_cpu_mpi_scaling.py | head -n 180" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 317ms:
--- CPU dt argument and mode handling ---
753-        column_physics_fn = make_kessler_column_physics_fn(sigma, dt)
754-
755-    tiled = make_tiled_fv3_hydrostatic_step_blocked_2d(
756-        mesh, cdgrid, sigma, resolution, kt, nlev,
757:        p_floor=float(config.p_floor), dt=float(dt),
758-        sponge_sigma=float(config.sponge_sigma),
759-        sponge_tau_sec=float(config.sponge_tau_sec),
760-        column_physics_fn=column_physics_fn,
761-        fix_mass=True)
--
788-        # passes the dt this builder returned — refuse anything else
789-        # rather than silently integrating with the wrong step.
790-        if float(dt_arg) != _dt_built:
791-            raise ValueError(
792:                f"tiled cube step compiled for dt={_dt_built}; got "
793-                f"{dt_arg}.")
794-        if _moist:
795-            u, v, T, ps, q = tiled_jit(s["u_d"], s["v_d"], s["T"],
796-                                       s["p_s"], s["phis"], s["q_pack"])
--
911-            rank, proc_lat, proc_lon, n_lat, n_lon,
912-        )
913-        block_grid = slice_latlon_grid_to_block_2d(grid, layout2d)
914-        local_model = CGridLatLonPrimitiveEquationModel(
915:            block_grid, sigma, config, dt=dt,
916-        )
917-        state = scatter_state_latlon_2d(cgrid_global, layout2d)
918-        step_fn = make_latlon_2d_mpi_step(
919-            local_model, layout2d, physics_fn=physics_fn,
--
960-        # (3) rank-local band model (the wrapper re-instantiates it
961-        # with rank-aware pole_v_bc + allreduced total_area itself).
962-        band_grid = slice_latlon_grid_to_band(grid, layout)
963-        local_model = CGridLatLonPrimitiveEquationModel(
964:            band_grid, sigma, config, dt=dt,
965-        )
966-
967-        # (4) + (5)
968-        state = scatter_state_latlon(cgrid_global, layout)
--
977-        # is the bottleneck-rank load — the right weak-scaling
978-        # normalizer for the rank-0-written result JSON.
979-        cells_per_rank = layout.n_lat_local * n_lon * nlev
980-    else:
981:        model = CGridLatLonPrimitiveEquationModel(grid, sigma, config, dt=dt)
982-        if physics_fn is not None:
983-            _phys = physics_fn
984-            step_fn = lambda state, dt: model.step(state, dt, physics_fn=_phys)
985-        else:
--
1163-        rank=rank,
1164-        n_ranks=n_ranks,
1165-        precision=precision,
1166-        physics_level=physics_level,
1167:        dt=dt,
1168-        cs_spmd=cs_spmd,
1169-        latlon_2d=latlon_2d,
1170-    )
1171-    # "2d" only for a genuine multi-rank lat-lon pencil; everything else
--
1185-
1186-    if rank == 0:
1187-        print(
1188-            f"  [{precision}] {res_label}/L{nlev} on {n_ranks} rank(s) | "
1189:            f"dt={dt_used:.0f}s | cells={total_cells:,} | "
1190-            f"cells/rank={cells_per_rank:,} | physics={physics_level}",
1191-            flush=True,
1192-        )
1193-
--
1484-    """Print a one-line summary."""
1485-    print(
1486-        f"  {result.grid_type:>14s} | {result.physics_level:>12s} | "
1487-        f"{result.n_ranks:>5d} ranks | {result.resolution:<5d} | "
1488:        f"L{result.n_levels:>2d} | dt={result.dt_seconds:>6.0f} | "
1489-        f"{result.time_per_step_ms:>9.2f} ms/step | "
1490-        f"SYPD={result.sypd:>8.3f} | {result.mcells_per_s:>9.1f} Mcells/s"
1491-    )
1492-
--- main run / output path ---
  1120	        physics_fn = make_kessler_forcing_spectral(dt)
  1121	    else:
  1122	        physics_fn = _build_physics_fn(physics_level, "spectral")
  1123	    if physics_fn is not None:
  1124	        _phys = physics_fn
  1125	        step_fn = lambda state, dt: model.step(state, dt, physics_fn=_phys)
  1126	    else:
  1127	        step_fn = model.step
  1128	
  1129	    return step_fn, state, dt, total_cells, total_cells
  1130	
  1131	
  1132	# ===========================================================================
  1133	# Core benchmark runner
  1134	# ===========================================================================
  1135	
  1136	def run_single_benchmark(
  1137	    *,
  1138	    grid_type: str,
  1139	    resolution: int,
  1140	    nlev: int,
  1141	    n_ranks: int,
  1142	    rank: int,
  1143	    precision: str,
  1144	    mode: str,
  1145	    physics_level: str,
  1146	    n_warmup: int,
  1147	    n_timing: int,
  1148	    dt: float | None = None,
  1149	    cs_spmd: bool = False,
  1150	    latlon_2d: bool = False,
  1151	) -> TimingResult:
  1152	    """Run a single benchmark case and return timing."""
  1153	    _validate_physics(grid_type, physics_level)
  1154	
  1155	    import jax
  1156	    import jax.numpy as jnp
  1157	
  1158	    (step_fn, state, dt_used, total_cells, cells_per_rank,
  1159	     part_metrics) = _build_amip_step(
  1160	        grid_type=grid_type,
  1161	        resolution=resolution,
  1162	        nlev=nlev,
  1163	        rank=rank,
  1164	        n_ranks=n_ranks,
  1165	        precision=precision,
  1166	        physics_level=physics_level,
  1167	        dt=dt,
  1168	        cs_spmd=cs_spmd,
  1169	        latlon_2d=latlon_2d,
  1170	    )
  1171	    # "2d" only for a genuine multi-rank lat-lon pencil; everything else
  1172	    # (band, single-rank, other grids) is the default "band".
  1173	    decomposition = "2d" if (
  1174	        latlon_2d and grid_type == "latlon" and n_ranks > 1
  1175	    ) else "band"
  1176	
  1177	    if grid_type == "spectral":
  1178	        res_label = f"T{resolution}"
  1179	    elif grid_type == "icosahedral":
  1180	        res_label = f"I{resolution}"
  1181	    elif grid_type == "latlon":
  1182	        res_label = f"LL{resolution}"
  1183	    else:
  1184	        res_label = f"C{resolution}"
  1185	
  1186	    if rank == 0:
  1187	        print(
  1188	            f"  [{precision}] {res_label}/L{nlev} on {n_ranks} rank(s) | "
  1189	            f"dt={dt_used:.0f}s | cells={total_cells:,} | "
  1190	            f"cells/rank={cells_per_rank:,} | physics={physics_level}",
  1191	            flush=True,
  1192	        )
  1193	
  1194	    # --- JIT compilation ---
  1195	    t_compile_start = time.perf_counter()
  1196	    state = step_fn(state, dt_used)
  1197	    jax.block_until_ready(jax.tree.leaves(state))
  1198	    compile_time = time.perf_counter() - t_compile_start
  1199	    if rank == 0:
  1200	        print(f"    JIT compile: {compile_time:.2f}s", flush=True)
  1201	
  1202	    # --- Warmup ---
  1203	    t_warmup_start = time.perf_counter()
  1204	    for _ in range(n_warmup):
  1205	        state = step_fn(state, dt_used)
  1206	    jax.block_until_ready(jax.tree.leaves(state))
  1207	    warmup_time = time.perf_counter() - t_warmup_start
  1208	
  1209	    # --- Timed steps via lax.scan ---
  1210	    input_dtypes = jax.tree.map(
  1211	        lambda x: x.dtype if hasattr(x, "dtype") else None, state)
  1212	
  1213	    @jax.jit
  1214	    def _scan_run(st):
  1215	        # ``dt_used`` is closed over as a STATIC Python float, NOT passed as a
  1216	        # jit argument.  As an argument it becomes a tracer, and the spectral PE
  1217	        # dycore caches its integrator/filter matrices keyed on a CONCRETE dt
  1218	        # (``_ensure_tracer_filter`` / ``_ensure_si_data`` compare ``self._..._dt
  1219	        # == dt``) -> a traced dt raised TracerBoolConversionError on the moist
  1220	        # spectral path.  dt is constant per benchmark, so closing over it is
  1221	        # correct and lets every dycore (including spectral) trace cleanly; the
  1222	        # other grids are unaffected (dt was only used in jnp ops).
  1223	        def _body(carry, _):
  1224	            new = step_fn(carry, dt_used)
  1225	            new = jax.tree.map(
  1226	                lambda x, d: x.astype(d)
  1227	                if d is not None and hasattr(x, "astype") else x,
  1228	                new, input_dtypes,
  1229	            )
  1230	            return new, None
  1231	        return jax.lax.scan(_body, st, None, length=n_timing)[0]
  1232	
  1233	    # Pre-compile scan without mutating the state used for timing.
  1234	    # Previously we rebound ``state`` to the precompile output, so the
  1235	    # timed run started from state already advanced by ``n_timing``
  1236	    # steps and the benchmark was biased.  Use a leaf-cloned input so
  1237	    # XLA still warms compile + caches against identical layout but
  1238	    # ``state`` keeps its original (post-warmup) trajectory.
  1239	    _precompile_state = jax.tree.map(lambda x: x, state)
  1240	    _precompile_out = _scan_run(_precompile_state)
  1241	    jax.block_until_ready(jax.tree.leaves(_precompile_out))
  1242	
  1243	    # MPI barrier before timing
  1244	    try:
  1245	        from mpi4py import MPI as _MPI
  1246	        if _MPI.COMM_WORLD.Get_size() > 1:
  1247	            jax.block_until_ready(jax.tree.leaves(state))
  1248	            _MPI.COMM_WORLD.Barrier()
  1249	    except (ImportError, RuntimeError):
  1250	        # broken/absent mpi4py must not break a single-process run
  1251	        pass
  1252	
  1253	    t0 = time.perf_counter()
  1254	    state = _scan_run(state)
  1255	    jax.block_until_ready(jax.tree.leaves(state))
  1256	
  1257	    # MPI barrier after timing
  1258	    try:
  1259	        from mpi4py import MPI as _MPI
  1260	        if _MPI.COMM_WORLD.Get_size() > 1:
  1261	            _MPI.COMM_WORLD.Barrier()
  1262	    except (ImportError, RuntimeError):
  1263	        # broken/absent mpi4py must not break a single-process run
  1264	        pass
  1265	
  1266	    t1 = time.perf_counter()
  1267	
  1268	    timing_time = t1 - t0
  1269	    time_per_step = timing_time / n_timing
  1270	    time_per_step_ms = time_per_step * 1000.0
  1271	    sypd = (dt_used / time_per_step) / (365.25 * 86400) * 86400.0
  1272	    mcells_per_s = (total_cells / time_per_step) / 1e6
  1273	
  1274	    if rank == 0:
  1275	        print(
  1276	            f"    Timing: {time_per_step_ms:.2f} ms/step | "
  1277	            f"SYPD={sypd:.3f} | {mcells_per_s:.1f} Mcells/s",
  1278	            flush=True,
  1279	        )
  1280	
  1281	    # cs-spmd shards over the jax DEVICE mesh (n_global = jax.device_count()),
  1282	    # but n_ranks is the PROCESS count (1 for a single-process multi-GPU run).
  1283	    # Record the device count so a single-process 2-GPU run lands as n=2 (not
  1284	    # n=1) in the CSV + JSON filename. cells_per_rank already used n_global on
  1285	    # this path; the multi-controller cs-spmd path has n_ranks == device_count
  1286	    # (one process per device), so this is a no-op there.
  1287	    _record_ndev = jax.device_count() if cs_spmd else n_ranks
  1288	    return TimingResult(
  1289	        n_ranks=_record_ndev,
  1290	        resolution=resolution,
  1291	        n_levels=nlev,
  1292	        precision=precision,
  1293	        mode=mode,
  1294	        grid_type=grid_type,
  1295	        physics_level=physics_level,
  1296	        dt_seconds=dt_used,
  1297	        n_warmup=n_warmup,
  1298	        n_timing=n_timing,
  1299	        compile_time_s=compile_time,
  1300	        warmup_time_s=warmup_time,
  1301	        timing_time_s=timing_time,
  1302	        time_per_step_ms=time_per_step_ms,
  1303	        sypd=sypd,
  1304	        total_cells=total_cells,
  1305	        cells_per_rank=cells_per_rank,
  1306	        mcells_per_s=mcells_per_s,
  1307	        decomposition=decomposition,
  1308	        partition_metrics=part_metrics,
  1309	    )
  1310	
  1311	
  1312	# ===========================================================================
  1313	# Sweep mode: generate case list
  1314	# ===========================================================================
  1315	
  1316	@dataclass
  1317	class CaseSpec:
  1318	    grid: str
  1319	    resolution: int
  1320	    n_ranks: int
  1321	    mode: str
  1322	    physics: str
  1323	    nlev: int
  1324	    precision: str
  1325	
  1520	        "--precision", choices=["float32", "float64"], default="float64",
  1521	        help="Floating-point precision.",
  1522	    )
  1523	    p.add_argument(
  1524	        "--device", choices=["cpu", "gpu"], default="cpu",
  1525	        help="cpu (default, CPU-MPI) or gpu (route-A: pin each rank to one "
  1526	             "local GPU, run the SAME mpi4jax dycore on the PCIe pair). "
  1527	             "Needs the overlay venv (cuda jax + CUDA-built mpi4jax).",
  1528	    )
  1529	    p.add_argument("--n-levels", type=int, default=26)
  1530	    p.add_argument("--n-warmup", type=int, default=5)
  1531	    p.add_argument("--n-timing", type=int, default=50)
  1532	    p.add_argument(
  1533	        "--cs-spmd", action="store_true",
  1534	        help="Cubed-sphere TRUE domain decomposition via jax.distributed "
  1535	             "multi-controller SPMD (global face mesh + multiface "
  1536	             "ppermute; the A1 path).  Replaces the replicated-dynamics "
  1537	             "refusal: launch with srun -n {2,3,6} (must divide 6).  "
  1538	             "Uses jax.distributed ONLY — the mpi4jax halo backend is "
  1539	             "never armed in this mode (mixed stacks deadlock).  "
  1540	             "Parity receipt: scripts/tmp/_probe_spmd_cube_parity.py "
  1541	             "(shard-local vs serial = 6.7e-10 @5 steps, job 8462928).",
  1542	    )
  1543	    p.add_argument(
  1544	        "--latlon-2d", action="store_true",
  1545	        help="Lat-lon C-grid 2-D pencil decomposition (proc_lat x proc_lon "
  1546	             "factored from the rank count to minimise the per-rank halo "
  1547	             "perimeter) instead of the 1-D latitude band.  WALL POLES only "
  1548	             "(regular grid; use_polar_filter off) — a labeled throughput "
  1549	             "benchmark, NOT the atmosphere's 180-deg pole fold.  Targets the "
  1550	             "band's high-rank starvation (weak-E ~0.05).  Validated by "
  1551	             "tests/distributed/test_latlon_2d_mpi_step.py (mass<1e-12 + "
  1552	             "2x2==1x4).  --grid latlon only.",
  1553	    )
  1554	    p.add_argument(
  1555	        "--output-dir", type=str, default="results/cpu_scaling",
  1556	        help="Output directory for results.",
  1557	    )
  1558	    p.add_argument(
  1559	        "--sweep", action="store_true",
  1560	        help="Generate case list (JSON lines) and exit. No execution.",
  1561	    )
  1562	    p.add_argument(
  1563	        "--max-ranks", type=int, default=64,
  1564	        help="Maximum rank count for sweep mode.",
  1565	    )
  1566	    p.add_argument(
  1567	        "--case", type=str, default=None,
  1568	        help="JSON case spec (from sweep output).",
  1569	    )
  1570	    return p
  1571	
  1572	
  1573	def main() -> int:
  1574	    args = build_parser().parse_args()
  1575	
  1576	    # --latlon-2d does not propagate through the sweep/--case round-trip yet
  1577	    # (CaseSpec carries no decomposition field), so a ``--sweep --latlon-2d``
  1578	    # would silently emit band cases and the launcher would run the BAND path.
  1579	    # Fail loud — use a direct ``--resolution N --latlon-2d`` invocation (the
  1580	    # measurement sbatch does) until the sweep threads the flag (codex).
  1581	    if args.sweep and args.latlon_2d:
  1582	        print(
  1583	            "ERROR: --latlon-2d is not threaded through --sweep yet (CaseSpec "
  1584	            "has no decomposition field), so the swept cases would silently "
  1585	            "run the 1-D band.  Use a direct '--grid latlon --resolution N "
  1586	            "--latlon-2d' run (one per rank count) instead.",
  1587	            flush=True,
  1588	        )
  1589	        return 2
  1590	
  1591	    # --- Sweep mode: just print cases and exit ---
  1592	    if args.sweep:
  1593	        # Iter 41: validate the grid+physics combo *before* the
  1594	        # sweep generator runs, so unsupported tiers (e.g. moist
  1595	        # physics on cubed-sphere/lat-lon CPU MPI, see iter 40)
  1596	        # error out immediately with a clear message instead of
  1597	        # producing a JSON-lines list that subsequently fails at
  1598	        # runtime under the SLURM array.
  1599	        _validate_physics(args.grid, args.physics)
  1600	        cases = generate_sweep_cases(
  1601	            args.grid, args.mode if args.mode != "single" else "both",
  1602	            args.physics, args.max_ranks,
  1603	            args.n_levels, args.precision,
  1604	        )
  1605	        for c in cases:
  1760	            print(
  1761	                "ERROR: cubed-sphere MPI is currently replicated-"
  1762	                "dynamics-only (every rank holds full state).  Use 1 "
  1763	                "rank or --grid icosahedral for genuine MPI scaling.",
  1764	                flush=True,
  1765	            )
  1766	        return 2
  1767	    # Lat-lon band MPI is real (latitude-band decomposition), but a
  1768	    # band must hold at least 2 lat rows for the halo=2 PPM /
  1769	    # biharmonic exchange (``pad_halo_latlon_mpi`` raises when
  1770	    # ``halo > n_lat_local``).  Refuse undersized configurations
  1771	    # up-front with a clear message instead of a mid-build traceback.
  1772	    if args.latlon_2d and grid_type != "latlon":
  1773	        if is_rank0:
  1774	            print("ERROR: --latlon-2d applies only to --grid latlon.",
  1775	                  flush=True)
  1776	        return 2
  1777	    # The >=2-lat-rows-per-rank guard is for the 1-D BAND (all ranks split
  1778	    # lat).  The 2-D pencil splits lat over proc_lat (< n_ranks), so its own
  1779	    # _factor_2d_latlon validates the per-block rows/cols — skip the band
  1780	    # guard for --latlon-2d.
  1781	    if (grid_type == "latlon" and not args.latlon_2d
  1782	            and n_ranks > 1 and resolution // n_ranks < 2):
  1783	        if is_rank0:
  1784	            print(
  1785	                f"ERROR: lat-lon band MPI needs >=2 lat rows per rank "
  1786	                f"(halo=2 exchange); resolution={resolution} on "
  1787	                f"{n_ranks} ranks gives {resolution // n_ranks} "
  1788	                f"rows/rank.  Increase --resolution or reduce ranks.",
  1789	                flush=True,
  1790	            )
  1791	        return 2
  1792	
  1793	    if is_rank0:
  1794	        print("=" * 72)
  1795	        print("  legoESM CPU MPI Scaling Benchmark")
  1796	        print("=" * 72)
  1797	        print(f"  Grid:      {grid_type}")
  1798	        print(f"  Physics:   {physics_level}")
  1799	        print(f"  Precision: {precision}")
  1800	        print(f"  Ranks:     {n_ranks}")
  1801	        print(f"  Resolution:{resolution}")
  1802	        print(f"  Levels:    {nlev}")
  1803	        print(f"  Mode:      {mode}")
  1804	        print("=" * 72)
  1805	
  1806	    result = run_single_benchmark(
  1807	        grid_type=grid_type,
  1808	        resolution=resolution,
  1809	        nlev=nlev,
  1810	        n_ranks=n_ranks,
  1811	        rank=rank,
  1812	        precision=precision,
  1813	        mode=mode,
  1814	        physics_level=physics_level,
  1815	        n_warmup=args.n_warmup,
  1816	        n_timing=args.n_timing,
  1817	        cs_spmd=bool(args.cs_spmd),
  1818	        latlon_2d=bool(args.latlon_2d),
  1819	    )
  1820	
  1821	    if is_rank0:
  1822	        print_summary(result)
  1823	        output_dir = Path(args.output_dir) / f"{grid_type}_{physics_level}_{mode}"
  1824	        write_result_json(result, output_dir, cs_spmd=bool(args.cs_spmd))
  1825	
  1826	    return 0
  1827	
  1828	
  1829	if __name__ == "__main__":
  1830	    raise SystemExit(main())
--- timing result formatter ---
1-#!/usr/bin/env python
2-"""CPU MPI scaling benchmark for AMIP-like runs.
3-
4:Measures wall-clock time per step and SYPD across varying MPI rank counts
5-and resolutions for the supported grid+physics combinations.
6-
7-MPI-scalable grids (multi-rank weak/strong scaling, genuinely
8-domain-decomposed at the dycore level):
9-  icosahedral   -- MPAS Voronoi TRiSK PE dycore (cell partition)
--
1271-    sypd = (dt_used / time_per_step) / (365.25 * 86400) * 86400.0
1272-    mcells_per_s = (total_cells / time_per_step) / 1e6
1273-
1274-    if rank == 0:
1275-        print(
1276:            f"    Timing: {time_per_step_ms:.2f} ms/step | "
1277:            f"SYPD={sypd:.3f} | {mcells_per_s:.1f} Mcells/s",
1278-            flush=True,
1279-        )
1280-
1281-    # cs-spmd shards over the jax DEVICE mesh (n_global = jax.device_count()),
1282-    # but n_ranks is the PROCESS count (1 for a single-process multi-GPU run).
--
1484-    """Print a one-line summary."""
1485-    print(
1486-        f"  {result.grid_type:>14s} | {result.physics_level:>12s} | "
1487-        f"{result.n_ranks:>5d} ranks | {result.resolution:<5d} | "
1488-        f"L{result.n_levels:>2d} | dt={result.dt_seconds:>6.0f} | "
1489:        f"{result.time_per_step_ms:>9.2f} ms/step | "
1490:        f"SYPD={result.sypd:>8.3f} | {result.mcells_per_s:>9.1f} Mcells/s"
1491-    )
1492-
1493-
1494-# ===========================================================================
1495-# CLI
--- 2D step’s dt caveat in source ---
432-    dt = cfl * dx_min / (u_max + c_grav)
433-    # Round down to a "nice" value; no floor — lat-lon pole cells can
434-    # require sub-second timesteps at very high resolution.
435:    if dt >= 30.0:
436:        return 30.0 * int(dt / 30.0)
437:    elif dt >= 5.0:
438:        return 5.0 * int(dt / 5.0)
439:    elif dt >= 1.0:
440-        return float(int(dt))
441-    else:
442-        # Truncate down to 2 decimal places (never round up past CFL)
443:        return math.floor(dt * 100) / 100
444-
445-
446-# ===========================================================================
--
474-    from legoesm.grids.vertical import create_sigma_coordinate
475-    sigma = create_sigma_coordinate(nlev)
476-
477:    if dt is None:
478-        dt = _auto_dt(resolution, grid_type)
479-
480-    def _cast(x):
--
784-    _dt_built = float(dt)
785-
786-    def step_fn(s, dt_arg):
787:        # dt is compiled into the tiled stage (static); the harness always
788-        # passes the dt this builder returned — refuse anything else
789-        # rather than silently integrating with the wrong step.
790-        if float(dt_arg) != _dt_built:
--
863-    # every rank derives the identical dt — only the boundary ranks
864-    # own the actual pole rows under band MPI).
865-    dx_pole = pole_cell_dx(grid)
866:    dt = min(dt, cfl_max_dt(dx_pole, 300.0, cfl_number=0.8, ndim=1))
867-
868-    config = CGridLatLonPrimitiveEquationConfig(
869-        A_h=0.0,
--
1186-    if rank == 0:
1187-        print(
1188-            f"  [{precision}] {res_label}/L{nlev} on {n_ranks} rank(s) | "
1189:            f"dt={dt_used:.0f}s | cells={total_cells:,} | "
1190-            f"cells/rank={cells_per_rank:,} | physics={physics_level}",
1191-            flush=True,
1192-        )
--
1217-        # dycore caches its integrator/filter matrices keyed on a CONCRETE dt
1218-        # (``_ensure_tracer_filter`` / ``_ensure_si_data`` compare ``self._..._dt
1219-        # == dt``) -> a traced dt raised TracerBoolConversionError on the moist
1220:        # spectral path.  dt is constant per benchmark, so closing over it is
1221-        # correct and lets every dycore (including spectral) trace cleanly; the
1222-        # other grids are unaffected (dt was only used in jnp ops).
1223-        def _body(carry, _):
--
1268-    timing_time = t1 - t0
1269-    time_per_step = timing_time / n_timing
1270-    time_per_step_ms = time_per_step * 1000.0
1271:    sypd = (dt_used / time_per_step) / (365.25 * 86400) * 86400.0
1272-    mcells_per_s = (total_cells / time_per_step) / 1e6
1273-
1274-    if rank == 0:
--
1485-    print(
1486-        f"  {result.grid_type:>14s} | {result.physics_level:>12s} | "
1487-        f"{result.n_ranks:>5d} ranks | {result.resolution:<5d} | "
1488:        f"L{result.n_levels:>2d} | dt={result.dt_seconds:>6.0f} | "
1489-        f"{result.time_per_step_ms:>9.2f} ms/step | "
1490-        f"SYPD={result.sypd:>8.3f} | {result.mcells_per_s:>9.1f} Mcells/s"
1491-    )

exec
/usr/bin/bash -lc "printf '%s\\n' '--- dt and wall-pole construction (full relevant range) ---'
nl -ba scripts/bench/run_cpu_mpi_scaling.py | sed -n '405,490p;844,885p'
printf '%s\\n' '--- cfl_max_dt definition and pole metric ---'
rg -n -C 6 'def cfl_max_dt|def pole_cell_dx|cfl_max_dt\\(' scripts/bench/run_cpu_mpi_scaling.py packages -g '*.py' | head -n 240
printf '%s\\n' '--- dt=0 receipt evidence ---'
rg -n -C 4 '\\[float64\\] LL512/L26 on (64|128|256|512) rank\\(s\\) \\| dt=0s|latlon \\|        moist \\|.*dt=     0' cpu_ll2d.26628073.log
printf '%s\\n' '--- actual CGrid step zero-dt behavior ---'
rg -n -C 4 'def step\\(|dt.*==|dt \\*|dt\\)' packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py packages -g '*latlon*cgrid*.py' | head -n 260" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 361ms:
--- dt and wall-pole construction (full relevant range) ---
   405	# CFL-safe timestep
   406	# ===========================================================================
   407	
   408	def _auto_dt(n_grid: int, grid_type: str) -> float:
   409	    from legoesm import constants  # lazy: see top-of-file note on JAX init order
   410	
   411	    R = constants.R_earth
   412	    u_max = 60.0
   413	    c_grav = 300.0
   414	    cfl = 0.7
   415	
   416	    if grid_type == "spectral":
   417	        n_lon = 2 * (n_grid + 1)
   418	        dx_min = math.pi * R / n_lon
   419	    elif grid_type == "icosahedral":
   420	        n_cells = 10 * 4 ** n_grid + 2
   421	        dx_avg = R * math.sqrt(4.0 * math.pi / n_cells)
   422	        dx_min = 0.9 * dx_avg
   423	    elif grid_type == "latlon":
   424	        # Pole-cell dx is the limiting spacing on lat-lon grids
   425	        n_lat = n_grid
   426	        dlat = math.pi / n_lat
   427	        dlon = 2.0 * math.pi / (2 * n_lat)
   428	        dx_min = R * dlon * math.cos(math.pi / 2.0 - dlat / 2.0)
   429	    else:  # cubed-sphere
   430	        dx_min = (math.pi / 2) * R / (n_grid * math.sqrt(3))
   431	
   432	    dt = cfl * dx_min / (u_max + c_grav)
   433	    # Round down to a "nice" value; no floor — lat-lon pole cells can
   434	    # require sub-second timesteps at very high resolution.
   435	    if dt >= 30.0:
   436	        return 30.0 * int(dt / 30.0)
   437	    elif dt >= 5.0:
   438	        return 5.0 * int(dt / 5.0)
   439	    elif dt >= 1.0:
   440	        return float(int(dt))
   441	    else:
   442	        # Truncate down to 2 decimal places (never round up past CFL)
   443	        return math.floor(dt * 100) / 100
   444	
   445	
   446	# ===========================================================================
   447	# Build benchmark step functions
   448	# ===========================================================================
   449	
   450	def _build_amip_step(
   451	    *,
   452	    grid_type: str,
   453	    resolution: int,
   454	    nlev: int,
   455	    rank: int,
   456	    n_ranks: int,
   457	    precision: str,
   458	    physics_level: str,
   459	    dt: float | None = None,
   460	    cs_spmd: bool = False,
   461	    latlon_2d: bool = False,
   462	):
   463	    """Build a step function + initial state for one benchmark case.
   464	
   465	    Returns (step_fn, state, dt, total_cells, cells_per_rank).
   466	    """
   467	    import jax
   468	    import jax.numpy as jnp
   469	
   470	    if precision == "float64":
   471	        jax.config.update("jax_enable_x64", True)
   472	    dtype = jnp.float64 if precision == "float64" else jnp.float32
   473	
   474	    from legoesm.grids.vertical import create_sigma_coordinate
   475	    sigma = create_sigma_coordinate(nlev)
   476	
   477	    if dt is None:
   478	        dt = _auto_dt(resolution, grid_type)
   479	
   480	    def _cast(x):
   481	        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
   482	            return x.astype(dtype)
   483	        return x
   484	
   485	    if grid_type == "cubed-sphere":
   486	        if cs_spmd:
   487	            out = _build_cubed_sphere_spmd(
   488	                resolution, nlev, dt, dtype, physics_level, _cast)
   489	        else:
   490	            out = _build_cubedsphere(resolution, nlev, sigma, dt, dtype, rank,
   844	def _build_latlon(resolution, nlev, sigma, dt, dtype, rank, n_ranks,
   845	                   physics_level, cast_fn, latlon_2d=False):
   846	    import jax
   847	    import jax.numpy as jnp
   848	
   849	    from legoesm.grids.latlon import create_latlon_grid
   850	    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
   851	        CGridLatLonPrimitiveEquationModel,
   852	        CGridLatLonPrimitiveEquationConfig,
   853	        hydrostatic_to_cgrid,
   854	    )
   855	
   856	    from legoesm.core.cfl import pole_cell_dx, cfl_max_dt
   857	
   858	    n_lat = resolution
   859	    n_lon = 2 * resolution
   860	    grid = create_latlon_grid(n_lat, n_lon)
   861	
   862	    # Clamp dt to pole-cell CFL limit (computed on the GLOBAL grid so
   863	    # every rank derives the identical dt — only the boundary ranks
   864	    # own the actual pole rows under band MPI).
   865	    dx_pole = pole_cell_dx(grid)
   866	    dt = min(dt, cfl_max_dt(dx_pole, 300.0, cfl_number=0.8, ndim=1))
   867	
   868	    config = CGridLatLonPrimitiveEquationConfig(
   869	        A_h=0.0,
   870	        fix_mass=True,
   871	        use_polar_filter=False,
   872	    )
   873	
   874	    # Baroclinic wave init for lat-lon (cell-centred HydrostaticState).
   875	    from tests.test_cases.baroclinic_wave import (
   876	        baroclinic_wave_init_latlon,
   877	    )
   878	    _moist = physics_level == "moist"
   879	    state = baroclinic_wave_init_latlon(grid, sigma, perturbed=True, moist=_moist)
   880	    state = jax.tree.map(cast_fn, state)
   881	
   882	    total_cells = n_lat * n_lon * nlev
   883	
   884	    if _moist:
   885	        # Kessler warm-rain bound to this step's dt (the physics_fn convention
--- cfl_max_dt definition and pole metric ---
scripts/bench/run_cpu_mpi_scaling.py-860-    grid = create_latlon_grid(n_lat, n_lon)
scripts/bench/run_cpu_mpi_scaling.py-861-
scripts/bench/run_cpu_mpi_scaling.py-862-    # Clamp dt to pole-cell CFL limit (computed on the GLOBAL grid so
scripts/bench/run_cpu_mpi_scaling.py-863-    # every rank derives the identical dt — only the boundary ranks
scripts/bench/run_cpu_mpi_scaling.py-864-    # own the actual pole rows under band MPI).
scripts/bench/run_cpu_mpi_scaling.py-865-    dx_pole = pole_cell_dx(grid)
scripts/bench/run_cpu_mpi_scaling.py:866:    dt = min(dt, cfl_max_dt(dx_pole, 300.0, cfl_number=0.8, ndim=1))
scripts/bench/run_cpu_mpi_scaling.py-867-
scripts/bench/run_cpu_mpi_scaling.py-868-    config = CGridLatLonPrimitiveEquationConfig(
scripts/bench/run_cpu_mpi_scaling.py-869-        A_h=0.0,
scripts/bench/run_cpu_mpi_scaling.py-870-        fix_mass=True,
scripts/bench/run_cpu_mpi_scaling.py-871-        use_polar_filter=False,
scripts/bench/run_cpu_mpi_scaling.py-872-    )
--
packages/coupler/legoesm/driver/component_factory.py-679-        # at 1° feasible within a chained 72-h SLURM budget.
packages/coupler/legoesm/driver/component_factory.py-680-        from legoesm.core.cfl import (
packages/coupler/legoesm/driver/component_factory.py-681-            pole_cell_dx, cfl_max_dt, max_laplacian_viscosity,
packages/coupler/legoesm/driver/component_factory.py-682-        )
packages/coupler/legoesm/driver/component_factory.py-683-        dx_pole = pole_cell_dx(grid)
packages/coupler/legoesm/driver/component_factory.py-684-        c_grav = 300.0  # gravity wave speed [m/s]
packages/coupler/legoesm/driver/component_factory.py:685:        dt_max_advective_pole = cfl_max_dt(
packages/coupler/legoesm/driver/component_factory.py-686-            dx_pole, c_grav, cfl_number=0.8, ndim=1,
packages/coupler/legoesm/driver/component_factory.py-687-        )
packages/coupler/legoesm/driver/component_factory.py-688-        _effective_dt = dc.dt
packages/coupler/legoesm/driver/component_factory.py-689-
packages/coupler/legoesm/driver/component_factory.py-690-        if dc.use_polar_filter:
packages/coupler/legoesm/driver/component_factory.py-691-            # Filter on → equatorial CFL is the effective limit.
packages/coupler/legoesm/driver/component_factory.py-692-            # dx_equator = R * dlon = circumference / n_lon.
packages/coupler/legoesm/driver/component_factory.py-693-            import math as _math
packages/coupler/legoesm/driver/component_factory.py-694-            dx_equator = float(2.0 * _math.pi * grid.radius / grid.n_lon)
packages/coupler/legoesm/driver/component_factory.py:695:            dt_max_advective = cfl_max_dt(
packages/coupler/legoesm/driver/component_factory.py-696-                dx_equator, c_grav, cfl_number=0.8, ndim=1,
packages/coupler/legoesm/driver/component_factory.py-697-            )
packages/coupler/legoesm/driver/component_factory.py-698-            dx_for_diffusion = dx_equator
packages/coupler/legoesm/driver/component_factory.py-699-            _clamp_dx_label = "equatorial"
packages/coupler/legoesm/driver/component_factory.py-700-        else:
packages/coupler/legoesm/driver/component_factory.py-701-            dt_max_advective = dt_max_advective_pole
--
packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py-502-                # Apply basic pole-cell CFL safety.  The driver factory
packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py-503-                # does a more thorough job (diffusion coefficients, etc.),
packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py-504-                # but create_model() callers get at least the dt guard.
packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py-505-                from legoesm.core.cfl import pole_cell_dx, cfl_max_dt
packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py-506-                dx_pole = pole_cell_dx(_grid)
packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py-507-                _dt = kwargs.get("dt", 600.0)
packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py:508:                _dt_clamped = min(_dt, cfl_max_dt(dx_pole, 300.0,
packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py-509-                                                   cfl_number=0.8, ndim=1))
packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py-510-                if _dt_clamped < _dt:
packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py-511-                    _warnings.warn(
packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py-512-                        f"create_model: dt={_dt:.0f}s exceeds pole-cell "
packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py-513-                        f"CFL limit ({_dt_clamped:.0f}s); clamping.",
packages/atmosphere/legoesm/atmosphere/dynamics/__init__.py-514-                        stacklevel=2,
--
packages/core/legoesm/core/cfl.py-103-    # Average cell spacing from sphere area / n_cells
packages/core/legoesm/core/cfl.py-104-    dx_avg = np.sqrt(4 * np.pi * radius**2 / n_cells)
packages/core/legoesm/core/cfl.py-105-    # Minimum spacing is ~0.85 of the average for quasi-uniform meshes
packages/core/legoesm/core/cfl.py-106-    return float(0.85 * dx_avg)
packages/core/legoesm/core/cfl.py-107-
packages/core/legoesm/core/cfl.py-108-
packages/core/legoesm/core/cfl.py:109:def cfl_max_dt(
packages/core/legoesm/core/cfl.py-110-    dx_min: float,
packages/core/legoesm/core/cfl.py-111-    wave_speed: float,
packages/core/legoesm/core/cfl.py-112-    cfl_number: float = 0.8,
packages/core/legoesm/core/cfl.py-113-    ndim: int = 2,
packages/core/legoesm/core/cfl.py-114-) -> float:
packages/core/legoesm/core/cfl.py-115-    """Compute maximum stable time step from CFL condition.
--
packages/core/legoesm/core/cfl.py-172-    """
packages/core/legoesm/core/cfl.py-173-    stability_constants = {2: 0.5, 4: 0.125, 6: 1.0 / 48.0}
packages/core/legoesm/core/cfl.py-174-    C_n = stability_constants.get(order, 1.0 / (2.0 ** order))
packages/core/legoesm/core/cfl.py-175-    return safety * C_n * dx_min ** order / dt
packages/core/legoesm/core/cfl.py-176-
packages/core/legoesm/core/cfl.py-177-
packages/core/legoesm/core/cfl.py:178:def pole_cell_dx(grid) -> float:
packages/core/legoesm/core/cfl.py-179-    """Return the zonal grid spacing at the polar-most cell [m].
packages/core/legoesm/core/cfl.py-180-
packages/core/legoesm/core/cfl.py-181-    Parameters
packages/core/legoesm/core/cfl.py-182-    ----------
packages/core/legoesm/core/cfl.py-183-    grid : LatLonGrid
packages/core/legoesm/core/cfl.py-184-        Must have ``radius``, ``dlon``, ``dlat`` attributes.
--
packages/core/legoesm/core/cfl.py-341-            c_total = 50.0 + 340.0  # jet + external gravity wave
packages/core/legoesm/core/cfl.py-342-        elif model_type == "compressible":
packages/core/legoesm/core/cfl.py-343-            c_total = 50.0 + 340.0  # jet + sound speed
packages/core/legoesm/core/cfl.py-344-        else:
packages/core/legoesm/core/cfl.py-345-            c_total = 50.0 + 300.0
packages/core/legoesm/core/cfl.py-346-
packages/core/legoesm/core/cfl.py:347:    dt_max = cfl_max_dt(dx_min, c_total, cfl_number, ndim=2)
packages/core/legoesm/core/cfl.py-348-
packages/core/legoesm/core/cfl.py-349-    if verbose:
packages/core/legoesm/core/cfl.py-350-        cfl_actual = c_total * dt / dx_min * np.sqrt(2)
packages/core/legoesm/core/cfl.py-351-        logger.info(f"  CFL check: dx_min={dx_min/1000:.0f} km, "
packages/core/legoesm/core/cfl.py-352-              f"c_max={c_total:.0f} m/s, "
packages/core/legoesm/core/cfl.py-353-              f"CFL(dt={dt:.0f}s)={cfl_actual:.2f}, "
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-835-        # silently using ``_max_dt`` on direct-construction paths
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-836-        # where ``component_factory`` did not set ``effective_dt``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-837-        self.dt = float(dt)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-838-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-839-        # Pole-cell CFL limit: dx_pole is the smallest cell on the grid.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-840-        dx_pole = pole_cell_dx(grid)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:841:        self._max_dt = cfl_max_dt(dx_pole, 300.0, cfl_number=0.8, ndim=1)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-842-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-843-        # Precompute polar filter masks: one for cell-centered fields
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-844-        # (dT, dps, du after lon-trim, every tracer) and one for v-face
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-845-        # fields (dv).  The v-face mask is built against
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-846-        # ``grid.cos_lat_v`` + the half-cell-offset lat-interface
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-847-        # coordinates so the wavenumber cutoff matches the actual
--- dt=0 receipt evidence ---
202-  Resolution:512
203-  Levels:    26
204-  Mode:      single
205-========================================================================
206:  [float64] LL512/L26 on 64 rank(s) | dt=0s | cells=13,631,488 | cells/rank=212,992 | physics=moist
207-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
208-  mpi4jax, MPI = require_mpi_stack()
209-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
210-  mpi4jax, MPI = require_mpi_stack()
--
396-[1785668570.891283] [l30516:325531:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
397-[1785668570.887663] [l30516:325547:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
398-[1785668570.922438] [l30516:325532:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
399-[1785668570.895224] [l30516:325545:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
400:          latlon |        moist |    64 ranks | 512   | L26 | dt=     0 |    297.57 ms/step | SYPD=   0.002 |      45.8 Mcells/s
401-  Result: /scratch/b/b381103/legoesm_scaling/cpu_ll2d_j26628073/np64/latlon_moist_single/latlon_2d_moist_single_r512_n64_float64.json
402-[1785668570.897366] [l30516:325518:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
403---- atm latlon 2-D pencil r512 f64 np=128 ---
404-[l30523.lvt.dkrz.de:374384] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
--
795-  Resolution:512
796-  Levels:    26
797-  Mode:      single
798-========================================================================
799:  [float64] LL512/L26 on 128 rank(s) | dt=0s | cells=13,631,488 | cells/rank=106,496 | physics=moist
800-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
801-  mpi4jax, MPI = require_mpi_stack()
802-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
803-  mpi4jax, MPI = require_mpi_stack()
--
1181-[1785668746.100706] [l30525:449272:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
1182-[1785668746.168837] [l30523:374372:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
1183-[1785668746.185832] [l30518:383898:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
1184-[1785668746.136911] [l30524:107429:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
1185:          latlon |        moist |   128 ranks | 512   | L26 | dt=     0 |    161.03 ms/step | SYPD=   0.004 |      84.7 Mcells/s
1186-  Result: /scratch/b/b381103/legoesm_scaling/cpu_ll2d_j26628073/np128/latlon_moist_single/latlon_2d_moist_single_r512_n128_float64.json
1187-[1785668746.194533] [l30518:383876:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
1188---- atm latlon 2-D pencil r512 f64 np=256 ---
1189-[l30528.lvt.dkrz.de:481958] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
--
1964-  Resolution:512
1965-  Levels:    26
1966-  Mode:      single
1967-========================================================================
1968:  [float64] LL512/L26 on 256 rank(s) | dt=0s | cells=13,631,488 | cells/rank=53,248 | physics=moist
1969-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
1970-  mpi4jax, MPI = require_mpi_stack()
1971-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
1972-  mpi4jax, MPI = require_mpi_stack()
--
2734-[1785668889.253603] [l30534:376039:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2735-[1785668889.210382] [l30538:414315:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2736-[1785668889.206648] [l30538:414331:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2737-[1785668889.291672] [l30531:280899:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2738:          latlon |        moist |   256 ranks | 512   | L26 | dt=     0 |     72.06 ms/step | SYPD=   0.009 |     189.2 Mcells/s
2739-  Result: /scratch/b/b381103/legoesm_scaling/cpu_ll2d_j26628073/np256/latlon_moist_single/latlon_2d_moist_single_r512_n256_float64.json
2740-[1785668889.253984] [l30526:299988:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
2741---- atm latlon 2-D pencil r512 f64 np=512 ---
2742-[l30543.lvt.dkrz.de:524801] common_ucx.c:362  Warning: UCX is unable to handle VM_UNMAP event. This may cause performance degradation or data corruption. Pls try adding --mca opal_common_ucx_opal_mem_hooks 1 to mpirun/oshrun command line to resolve this issue.
--
4285-  Resolution:512
4286-  Levels:    26
4287-  Mode:      single
4288-========================================================================
4289:  [float64] LL512/L26 on 512 rank(s) | dt=0s | cells=13,631,488 | cells/rank=26,624 | physics=moist
4290-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
4291-  mpi4jax, MPI = require_mpi_stack()
4292-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign/packages/core/legoesm/parallel/reductions.py:647: RuntimeWarning: Detected versions outside legoESM's tested MPI range: mpi4jax==0.9.0 (tested >=0.8.0, <0.9.0). MPI execution may fail or produce incorrect results. legoESM supports two compatible generations paired TOGETHER: legacy (jax 0.8-0.9 + mpi4jax 0.8) and FFI (jax 0.10 + mpi4jax 0.9); a cross pairing is incompatible (mpi4jax 0.8's custom-call API was removed in jax 0.10, and the mpi4jax 0.9 FFI API needs jax >= 0.10). Set LEGOESM_MPI_STRICT_COMPAT=1 to turn this into a hard error, or for this jax (<0.10) install the legacy mpi4jax line: pip install 'mpi4jax>=0.8,<0.9'.
4293-  mpi4jax, MPI = require_mpi_stack()
--
5823-[1785669029.524379] [l30531:281700:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
5824-[1785669029.429398] [l30530:217112:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
5825-[1785669029.413795] [l30517:783151:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
5826-[1785669029.433460] [l30530:217116:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
5827:          latlon |        moist |   512 ranks | 512   | L26 | dt=     0 |     44.78 ms/step | SYPD=   0.014 |     304.4 Mcells/s
5828-  Result: /scratch/b/b381103/legoesm_scaling/cpu_ll2d_j26628073/np512/latlon_moist_single/latlon_2d_moist_single_r512_n512_float64.json
5829-[1785669029.363467] [l30516:326365:0]     ucp_context.c:969  UCX  WARN  transports 'cuda_copy','cuda_ipc','gdr_copy' are not available, please use one or more of: cma, dc, dc_mlx5, dc_x, ib, knem, mm, posix, rc, rc_mlx5, rc_v, rc_verbs, rc_x, self, shm, sm, sysv, tcp, ud, ud_mlx5, ud_v, ud_verbs, ud_x
5830-=== RESULTS ===
5831-np64:    297.57 ms
--- actual CGrid step zero-dt behavior ---
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-833-        # recover it without falling back to the pole-cell ``_max_dt``
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-834-        # — Codex review Stage 3-E round 3 BLOCK caught the fallback
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-835-        # silently using ``_max_dt`` on direct-construction paths
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-836-        # where ``component_factory`` did not set ``effective_dt``.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:837:        self.dt = float(dt)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-838-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-839-        # Pole-cell CFL limit: dx_pole is the smallest cell on the grid.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-840-        dx_pole = pole_cell_dx(grid)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-841-        self._max_dt = cfl_max_dt(dx_pole, 300.0, cfl_number=0.8, ndim=1)
--
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-1279-            state = state._replace(p_s=p_s_post)
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-1280-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-1281-        return state
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-1282-
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1283:    def step(
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-1284-        self,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-1285-        state,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-1286-        dt: float,
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py-1287-        target_mass: jax.Array | None = None,
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1170-        # (no singular pole -- it uses the north fold).
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1171-        _cosl = jnp.maximum(jnp.cos(lat2d), 1.0e-12)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1172-        _dx = grid.radius * _dlon * _cosl                   # zonal cell width [m]
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1173-        _dy = grid.radius * float(grid.dlat)                 # meridional cell [m]
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:1174:        cap_strict = safety / (dt * (_dx ** -2 + _dy ** -2))
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1175-        _polar = jnp.abs(lat2d) > jnp.deg2rad(89.5)          # singular-pole rows
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1176-        cap_h = jnp.where(_polar, jnp.minimum(cap_h, cap_strict), cap_h)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1177-    # Vertex ceiling: pad to (n_lat+1, n_lon+1) -- edge in lat, periodic in lon.
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1178-    cap_q = jnp.pad(cap_h, ((0, 1), (0, 0)), mode="edge")    # (n_lat+1, n_lon)
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1737-
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1738-def _bc_tracer_tendencies(T, S, config, grid, mask, J, z_coord):
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1739-    """Stage 9: horizontal (Laplacian / biharmonic) + explicit vertical tracer
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1740-    diffusion for T and S. Pure verbatim extraction (Q8). Returns ``(dT_dt,
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:1741:    dS_dt)`` (advection + physics are added by the caller / step function)."""
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1742-    # --- 9. Tracer tendencies (diffusion + physics only) ---
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1743-    # Horizontal AND vertical tracer advection are handled in the step()
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1744-    # function using barotropic-averaged transport (Hallberg 1997, #102).
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-1745-    # Vertical velocity w is diagnosed from the barotropic-averaged
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2106-    """Stage 7c: WENO divergence (D-term) momentum dissipation (Silvestri et al.
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2107-    2024 Eqs. 31-32). Active only for weno5/weno7 momentum advection with
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2108-    config.weno_d_term; otherwise the diagnostics are zero. Pure verbatim
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2109-    extraction (Q8). Returns ``(du_dt, dv_dt, diag_Dterm_u, diag_Dterm_v)``."""
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2110:    diag_Dterm_u = jnp.zeros_like(du_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2111:    diag_Dterm_v = jnp.zeros_like(dv_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2112-    # --- 7c. Divergence flux (D term, Silvestri et al. 2024 Eqs. 31-32) ---
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2113-    # The two components of ∇·u are treated asymmetrically:
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2114-    #
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2115-    #   {D}_at_u = {δ_i U; δ_i U}_i  +  ⟨δ_j V⟩_i
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2193-    interior cuts). Serial/MPI use the static ``lat_ends_are_poles()``; the SPMD
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2194-    ``shard_map`` selects the edge bands data-dependently via
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2195-    ``spmd_pole_end_masks()`` (same pattern as ``neumann_fill_cgrid``).
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2196-
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2197:    Returns ``(du_dt, dv_dt)``.
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2198-    """
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2199-    if rate_s <= 0.0 or _mom_adv not in ("weno5", "weno7", "weno9"):
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2200-        return du_dt, dv_dt
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2201-    nw = _WALL_FILTER_ROWS
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2279-    # ``w``).  The explicit flux-form tendency is STILL computed and
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2280-    # stored in ``diag_vertadv_{u,v}`` as the start-of-step estimate of
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2281-    # the (otherwise implicit) vertical-advection term, so the momentum
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2282-    # budget reports the term being stabilised rather than a silent zero
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2283:    # (it matches the O(dt) start-of-step semantics already documented on
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2284-    # ``tendencies_with_diagnostics``).
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2285-    #
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2286-    # Closure contract:
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2287:    #   - flag OFF: ``vertadv`` is in ``du_dt``     -> Σ(terms) == du_dt.
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2288-    #   - flag ON : ``vertadv`` is a diagnostic only -> Σ(terms) ==
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2289-    #     du_dt + diag_vertadv (the vertadv slice is the start-of-step
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2290-    #     estimate of the step-level implicit operator, excluded from the
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2291-    #     slow forcing on purpose).  ``test_momentum_diagnostics_closure``
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2295-    # Default the diagnostics to zero so the (Q8-decomposed) helper's
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2296-    # ``(du_dt, dv_dt, diag_vertadv_u, diag_vertadv_v)`` return contract
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2297-    # holds even on the flag-on / not-diagnosing fast path where the
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2298-    # explicit tendency is never built.
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2299:    diag_vertadv_u = jnp.zeros_like(du_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2300:    diag_vertadv_v = jnp.zeros_like(dv_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2301-    # Compute the explicit flux-form vertadv tendency when it is either
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2302-    # (a) part of the slow forcing (flag off), or (b) needed for the
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2303-    # momentum budget as the start-of-step estimate (flag on AND
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2304-    # diagnosing).  When the flag is on and we are not diagnosing, skip it
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2396-    slope-foot enhancement) plus the meridional-only Laplacian viscosity. Kept
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2397-    together because the meridional block reuses the slope-foot helper/fields
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2398-    defined in stage 10. Pure verbatim extraction (Q8). Returns the momentum
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2399-    accumulators plus the per-term viscosity diagnostics."""
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2400:    diag_Ah_lap_u = jnp.zeros_like(du_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2401:    diag_Ah_lap_v = jnp.zeros_like(dv_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2402-    # Cell-centre A_h coefficient field [m²/s] used to build the FAITHFUL
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2403-    # positive-definite K_diss_h EKE source (Veros analogue) — captured ONLY when
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2404-    # the prognostic-EKE ``source_kdiss_h`` + ``kdiss_h_flux_form`` options are on
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2405-    # (the ACC recipe).  It is the SAME ``A_h × scale`` (cos-power / eq-boost /
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2413-        and getattr(_eke_for_kdiss, "source_kdiss_h", False)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2414-        and getattr(_eke_for_kdiss, "kdiss_h_flux_form", False))
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2415-    _ah_scale_center = None   # (n_lat,) latitudinal A_h×scale at cell centres
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2416-    _slope_E_center = None    # 3-D slope-foot enhancement at cell centres (or None)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2417:    diag_Bh_bilap_u = jnp.zeros_like(du_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2418:    diag_Bh_bilap_v = jnp.zeros_like(dv_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2419:    diag_Cs_smag_u = jnp.zeros_like(du_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2420:    diag_Cs_smag_v = jnp.zeros_like(dv_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2421:    diag_Cl_leith_u = jnp.zeros_like(du_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2422:    diag_Cl_leith_v = jnp.zeros_like(dv_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2423-    # --- 10. Mixing (viscosity on perturbation velocity) ---
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2424-    # Uses the proper vector Laplacian grad(div) - k×grad(curl) directly
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2425-    # on face velocities, avoiding the lossy cell-center detour.
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2426-    # See issue #105 for details.
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2547-            # cell-centre ahmt for the K_diss_h coefficient field. APPROXIMATION:
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2548-            # ``vector_laplacian_dissipation_cgrid`` credits ahmt·(div²+ζ²) whereas
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2549-            # this operator dissipates ahmt·div² + ahmf·ζ² (distinct T/F coeffs). On
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2550-            # the Mercator grid ahmf≈ahmt (adjacent rows, e1≈e2) so the ζ²-source is
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2551:            # mis-scaled by <~1% at high lat — diagnostic only (never du_dt/dv_dt),
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2552-            # and dormant unless kdiss_h_flux_form is on (NOT the DINO card).
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2553-            _ah_scale_center = _ahmt
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2554-        du_dt = du_dt + diag_Ah_lap_u
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-2555-        dv_dt = dv_dt + diag_Ah_lap_v
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3034-    partial-cell seafloor / deepest level. Pure verbatim extraction (Q8) +
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3035-    the NEMO branch. ``h_k`` is the cell-centre actual layer thickness
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3036-    (NEMO e3t) required by the NEMO schemes. Returns ``(du_dt, dv_dt,
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3037-    diag_botdrag_u, diag_botdrag_v)``."""
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3038:    diag_botdrag_u = jnp.zeros_like(du_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3039:    diag_botdrag_v = jnp.zeros_like(dv_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3040-    _scheme = validate_bottom_drag_scheme(
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3041-        str(getattr(config.bottom_drag, "bottom_drag_scheme", "legacy")))
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3042-    _nemo_drag = _scheme != "legacy"
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3043-    u_bg = float(getattr(config.bottom_drag, "bottom_drag_bg_velocity", 0.0))
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3194-def _bc_explicit_vertical_viscosity(du_dt, dv_dt, u_prime, v_prime, u, J, z_coord, config, grid):
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3195-    """Explicit background vertical viscosity A_v on the perturbation velocity
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3196-    (skipped when implicit_vertical_mixing is enabled). Pure verbatim extraction
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3197-    (Q8). Returns ``(du_dt, dv_dt, diag_Av_vert_u, diag_Av_vert_v)``."""
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3198:    diag_Av_vert_u = jnp.zeros_like(du_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3199:    diag_Av_vert_v = jnp.zeros_like(dv_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3200-    # Skip the explicit background vertical viscosity when the host
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3201-    # dynamics requested an implicit (backward-Euler) vertical solve —
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3202-    # the LatLonCGridOceanConfig.A_v floor is folded into the implicit
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3203-    # K profile downstream and applied unconditionally-stable.  The
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3233-    """Stage 10b: the physics-pipeline tendencies (KPP/TKE vertical mixing,
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3234-    convection, etc.) applied via a cell-centre proxy state, interpolated to
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3235-    faces. Returns the captured K_v/A_v profiles for the implicit solve. Pure
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3236-    verbatim extraction (Q8)."""
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3237:    diag_phys_u = jnp.zeros_like(du_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3238:    diag_phys_v = jnp.zeros_like(dv_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3239-    # --- 10b. Physics tendencies (surface forcing, bottom drag, etc.) ---
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3240-    # The physics pipeline expects cell-center u/v shapes (shared with
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3241-    # A-grid and cubed-sphere).  Create a cell-center proxy state so
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3242-    # the physics functions produce (n_lat, n_lon, nlev) output, then
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3320-
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3321-    Returns ``(du_dt, dv_dt, dT_dt, dS_dt, dT_surf)`` where ``dT_surf`` is a
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3322-    full-column zero array unless ``route_heat_to_implicit`` AND a q_net forcing
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3323-    were both present."""
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3324:    dT_surf = jnp.zeros_like(dT_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3325-    # --- 10b'. External surface forcing (e.g. from JRA55 bulk fluxes) ---
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3326-    # When the caller passes an OceanSurfaceForcing carrying tau_x /
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3327-    # tau_y / q_net / sw_down, apply them here.  Mirrors
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3328-    # mpas_physics.py:183-245 so MPAS and the lat-lon C-grid behave
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3353-            # (nemo_stage_mean_imposition = the stprk3_stg:440 zub step).
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3354-            tau_i_u, tau_j_v, dz_0_u, dz_0_v = surface_stress_faces(
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3355-                surface_forcing, u.dtype, z_coord, J, grid)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3356-            rho_0_dt = jnp.asarray(rho_0, dtype=u.dtype)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3357:            inv_rho_dz_u = 1.0 / (rho_0_dt * jnp.maximum(dz_0_u, 1e-10))
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3358:            inv_rho_dz_v = 1.0 / (rho_0_dt * jnp.maximum(dz_0_v, 1e-10))
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3359-
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3360-            du_dt = du_dt.at[..., 0].add(tau_i_u * inv_rho_dz_u)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3361-            dv_dt = dv_dt.at[..., 0].add(tau_j_v * inv_rho_dz_v)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3362-
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3459-    the weight-1.0 implicit-vmix seam instead (Veros ``tempsalt_sources``
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3460-    placement).  The momentum sponge (``u_ref``/``v_ref``) is ALWAYS explicit
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3461-    (Veros has no momentum sponge; tempsalt_sources is tracer-only).
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3462-    """
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3463:    diag_sponge_u = jnp.zeros_like(du_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3464:    diag_sponge_v = jnp.zeros_like(dv_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3465-    # --- 10c. Sponge layer relaxation ---
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3466-    # Cast sponge arrays to state dtype to prevent float64 promotion when
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3467-    # the precision policy stores state in float32 (crashes barotropic scan).
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3468-    if sponge is not None:
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3483-                dT_dt, dS_dt, T, S, sponge, mask=None, expand_gamma_axis=-1,
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3484-            )
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3485-        _dt = T.dtype
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3486-        if sponge.u_ref is not None:
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3487:            gamma_u = interp_cell_to_uface(sponge.gamma.astype(_dt))[..., jnp.newaxis]
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3488:            diag_sponge_u = gamma_u * (sponge.u_ref.astype(_dt) - u)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3489-            du_dt = du_dt + diag_sponge_u
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3490-        if sponge.v_ref is not None:
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3491:            gamma_v = interp_to_v_points(sponge.gamma.astype(_dt), grid=grid)[..., jnp.newaxis]
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3492:            diag_sponge_v = gamma_v * (sponge.v_ref.astype(_dt) - v)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3493-            dv_dt = dv_dt + diag_sponge_v
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3494-    return du_dt, dv_dt, dT_dt, dS_dt, diag_sponge_u, diag_sponge_v
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3495-
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3496-
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3773-    # Coriolis is NOT included in the returned momentum tendencies.
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3774-    # It is applied as a forward-backward (Matsuno) step in the step
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3775-    # function (ocean_model_latlon_cgrid.py), which is unconditionally
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3776-    # stable for inertial oscillations.  Forward Euler Coriolis amplifies
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3777:    # by sqrt(1 + (f*dt)^2) per step and blows up within ~1 day at
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3778-    # high latitudes.
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3779-
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3780-    # --- Stages 6 / 6-7 / 6b: KE gradient + pressure gradient + Adcroft
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3781-    # partial-cell PGF correction. ---
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3806-    du_dt = KE_PGF_u
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3807-    dv_dt = KE_PGF_v
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3808-    # Diagnostics scaffolding: zero arrays for terms that may be
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3809-    # inactive in this config; overwritten below where active.
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3810:    _diag_zero_u = jnp.zeros_like(du_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3811:    _diag_zero_v = jnp.zeros_like(dv_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3812-    diag_vortcor_u = _diag_zero_u
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3813-    diag_vortcor_v = _diag_zero_v
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3814-    diag_Dterm_u = _diag_zero_u    # WENO momentum-advection D-term;
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3815-    diag_Dterm_v = _diag_zero_v    # zero unless WENO + weno_d_term active.
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3917-        du_dt, dv_dt, u, v, u_mask_3d, v_mask_3d,
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3918-        config.wall_grid_filter_rate_s, _mom_adv,
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3919-    )
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3920-
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3921:    # --- Stage 9: tracer diffusion tendencies (dT_dt, dS_dt). ---
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3922-    if momentum_only:
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3923-        # RK3 momentum sub-stages freeze T/S, so this tracer-diffusion
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3924-        # tendency is recomputed identically and DISCARDED by the caller
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3925:        # (_mom_pert reads only du_dt/dv_dt).  Skip it => bit-identical
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3926:        # momentum (the momentum stages never read dT_dt/dS_dt), saving the
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3927-        # laplacian/biharmonic + vertical-diffusion compute AND its lat-halos
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3928-        # per RK sub-stage (codex halo-hunt #3).  The physics/forcing/sponge
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3929-        # blocks below still run (they also produce momentum tendencies); their
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3930-        # tracer additions land on this zero and are discarded by the caller.
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3947-        # tendency; the advective tracer update (model step) then runs on a
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3948-        # dT_dt that carries only the NON-dissipative sources added below
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3949-        # (physics / surface forcing / sponge). The lateral diffusion is
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3950-        # re-applied at weight 1.0 in the model step.
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3951:        dT_dt = jnp.zeros_like(dT_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3952:        dS_dt = jnp.zeros_like(dS_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3953-    else:
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3954-        dT_diss_lat = None
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3955-        dS_diss_lat = None
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3956-
--
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3958-    if skip_lateral_viscosity:
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3959-        # NEMO WS-RK3 stage 2 (stprk3_stg:318-334): eos+hpg+vor+adv ONLY —
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3960-        # dyn_ldf is applied at stages 1 and 3, not 2. Static bool; the zero
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3961-        # diagnostics keep the per-step closure identity.
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3962:        _zu = jnp.zeros_like(du_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3963:        _zv = jnp.zeros_like(dv_dt)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3964-        (diag_Ah_lap_u, diag_Ah_lap_v, diag_Bh_bilap_u, diag_Bh_bilap_v,
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3965-         diag_Cs_smag_u, diag_Cs_smag_v, diag_Cl_leith_u, diag_Cl_leith_v) = (
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3966-            _zu, _zv, _zu, _zv, _zu, _zv, _zu, _zv)
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py-3967-        kdiss_h_cell = None
--

codex
The implementation fixes hold: METIS/load wording, LL2048 arithmetic, hundreds-GPU tile comments, finite-result gates, and SHA qualification are correct. The wrapper is also sound: `ARM_TAG` reaches `srun`, only rank 0 emits the step nodelist, and `exec "$0" "$@"` runs Python with the benchmark script and original arguments.

CPU 2-D efficiency arithmetic is correct: 0.92396, 1.03237, and 0.83064 round to 0.92/1.03/0.83. The r512 2-D factorization is valid (16×32); the generic metadata explanation for `process_count: 1` is correct.

VERDICT: FIX-FIRST

1. The campaign document still contradicts the v3 interpretation limit and cites the obsolete ensemble job. It says “ordering counterbalanced” and treats a >10% penalty as the fabric-contention term, while the script correctly says pre/post bracketing is not counterbalancing and attribution needs follow-up. It also still records job `26628021`, with no reference to `26628196`. Correct [the campaign section](docs/performance/scaling/levante_campaign_2026-07-24.md:1676) and [job table](docs/performance/scaling/levante_campaign_2026-07-24.md:1719) to match [the script’s limitation](scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:43).

2. The new CPU receipt is not qualified for a known unsupported MPI runtime. Its log reports both an out-of-tested-range `mpi4jax==0.9.0` combination that “may fail or produce incorrect results” and UCX VM_UNMAP warnings that may cause performance degradation or data corruption ([log](cpu_ll2d.26628073.log:3), [runtime check](packages/core/legoesm/parallel/reductions.py:229)). Re-run with a compatible MPI/JAX stack and corrected UCX setup, or label the CPU ladder as unvalidated timing-only evidence rather than saying it “scales cleanly.”
tokens used
192,135
The implementation fixes hold: METIS/load wording, LL2048 arithmetic, hundreds-GPU tile comments, finite-result gates, and SHA qualification are correct. The wrapper is also sound: `ARM_TAG` reaches `srun`, only rank 0 emits the step nodelist, and `exec "$0" "$@"` runs Python with the benchmark script and original arguments.

CPU 2-D efficiency arithmetic is correct: 0.92396, 1.03237, and 0.83064 round to 0.92/1.03/0.83. The r512 2-D factorization is valid (16×32); the generic metadata explanation for `process_count: 1` is correct.

VERDICT: FIX-FIRST

1. The campaign document still contradicts the v3 interpretation limit and cites the obsolete ensemble job. It says “ordering counterbalanced” and treats a >10% penalty as the fabric-contention term, while the script correctly says pre/post bracketing is not counterbalancing and attribution needs follow-up. It also still records job `26628021`, with no reference to `26628196`. Correct [the campaign section](docs/performance/scaling/levante_campaign_2026-07-24.md:1676) and [job table](docs/performance/scaling/levante_campaign_2026-07-24.md:1719) to match [the script’s limitation](scripts/cluster/scaling_levante/mpas_s9_ensemble.sbatch:43).

2. The new CPU receipt is not qualified for a known unsupported MPI runtime. Its log reports both an out-of-tested-range `mpi4jax==0.9.0` combination that “may fail or produce incorrect results” and UCX VM_UNMAP warnings that may cause performance degradation or data corruption ([log](cpu_ll2d.26628073.log:3), [runtime check](packages/core/legoesm/parallel/reductions.py:229)). Re-run with a compatible MPI/JAX stack and corrected UCX setup, or label the CPU ladder as unvalidated timing-only evidence rather than saying it “scales cleanly.”
