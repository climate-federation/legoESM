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
session id: 019fb911-4a0d-78e2-ac02-aff596ca8b85
--------
user
You are advising a JAX Earth-system model (legoESM) parallel-scaling campaign on
DKRZ Levante (A100 80GB x4/node, HDR200 IB; CPU nodes 2x AMD EPYC 7763).

QUESTION: what concrete engineering improvements would raise weak- and
strong-scaling efficiency on BOTH lanes (MPI multi-rank CPU, and multi-GPU
SPMD/jax.jit sharding) for ALL our grids: cubed-sphere, lat-lon (atm + ocean
C-grid), icosahedral/MPAS-Voronoi (atm + ocean), spectral?

Everything below is MEASURED on this codebase. Do NOT re-propose anything
already refuted; treat these as constraints and find what is LEFT.

MEASURED STATE:
- All three GPU lanes are TILE-LIMITED, not device-count-limited. Fixed-tile
  contrasts: cube 1.06x cost at 2.25x devices, lat-lon 1.17x at 4x, ocean
  0.99x at 2x. Rule of thumb floor ~30k columns/GPU (CPU-MPI floor 300-600
  cells/rank).
- Ocean implicit-CN barotropic PCG: 95.7 us/iteration is SYNC (measured 111 us
  @nd1 vs 123.8 @nd4, compute 28.1 us), x60 iters = 65% of the 8.87 ms
  distributed overhead. Splits ~50/50 global reduction vs matvec halo.
  single_reduce PCG variant and wide-halo (120->4 barotropic messages/step)
  both shipped and help; wide-halo is 1-D-decomposition-only by design.
- REFUTED already: byte-volume as the bound (comm is latency-dominated:
  measured NVLink 17.8 us / 64.2 GB/s, IB 26.3 us / 23.5 GB/s = 94% of line
  rate); latency-hiding/pipelined-p2p/CP-combining (null); wet-cell
  compaction (gather penalty 1.40-1.77x, real ocean 0.71 wet = a loss);
  wet-balanced decomposition (25-45% SLOWER, dense arrays track total rows);
  halo packing on cube (bounded at 3.6% before implementing).
- Cube fixed cost attributed by nsys marker-kernel isolation: halo SendRecv
  6.65 ms = 44%.
- Ocean MPAS-Voronoi CPU lane has a GENUINE rank-count term the atmosphere
  lane does not: at IDENTICAL 5120 cells/rank, 4x ranks costs 1.65x at the
  BEST partition (geometric); sfc partition degrades WITH rank count
  (1.22x @32 -> 1.97x @128). Load imbalance within 6%, so it is partition
  QUALITY / comm topology, not balance. pymetis absent so METIS low-cut arm
  unmeasured.
- Count-specific FUSION pathology on MPAS SPMD: np4-f32 and np8-f64 are
  anomalously slow; a statically-gated optimization_barrier on tendency
  outputs recovers 11-17%. The pathology is shape x dtype dependent, so the
  gate is a hardcoded signature list -- unsatisfying.
- SETUP-TIME global-residency defect (#1370) fixed: per-device residency was
  tracking GLOBAL problem size (102 GB wall at LL2304@64); host-side global
  build + sharded band-geometry stacks cut it 16-31x to a size-independent
  0.10 GB.
- Mixed precision (f64 storage + f32 vertical-mixing/baroclinic internals)
  buys +14-15% on every ocean solver arm.
- Reach so far: 1024 CPU MPI ranks (atm ico), 512 ranks (ocean MPAS), 128
  GPUs (MPAS ico), 64 GPUs (ocean lat-lon LL1152).
- Atm SPECTRAL lane has NO distributed scaling receipts at all (designed gap).
- Constraints: XLA/JAX only, no custom CUDA kernels; A100 (no NVLS multicast,
  no Hopper); collectives are NCCL-backed so NCCL_PROTO env A/B is reachable;
  end-to-end jax.grad differentiability must be preserved; MPI halo exchange
  goes through a custom_vjp sendrecv.

Give a RANKED list. For each item: (1) the specific mechanism you expect to
pay and roughly how much, (2) the CHEAPEST measurement that would confirm or
REFUTE it before we build it, (3) which grid/lane it applies to, (4) whether
it risks differentiability or conservation. Prioritise things that attack the
TILE-SIZE floor and the latency-bound sync, since those are the two measured
walls. Be adversarial about our framing if you think the campaign is chasing
the wrong bottleneck.
codex
I’ll treat the existing measurements as hard exclusions and rank only mechanisms that can change the tile floor, eliminate synchronization, or fix topology/partition quality. I’ll also inspect the implementation to keep the probes directly actionable.
exec
/usr/bin/bash -lc "pwd && rg --files -g '"'!*'"\\.egg-info*' | head -160" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 450ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
project_status.md
LICENSE
CONTRIBUTING.md
CITATION.cff
uv.lock
CMIP.md
CHANGELOG.md
requirements_mpi.txt
FEDERATION.md
pyproject.toml
CLAUDE.md
COMMERCIAL-LICENSE.md
README.md
data/forcing_amip/aerosol_amip_clim.nc
config/4DVar_single/README.md
data/forcing_amip/sst_sic_amip_1979-1980.nc
data/forcing_amip/ozone_amip_clim.nc
data/forcing_amip/ghg_amip_1979-1980.nc
data/forcing_amip/solar_amip_1979-1980.nc
data/forcing_amip/volcanic_amip_1979-1980.nc
config/4DVar_single/experiment.yaml
config/4DVar_single/nmc.yaml
evaluations/visualize.py
evaluations/__init__.py
evaluations/wb_regrid.py
evaluations/wb_forecast.py
evaluations/headline_diagnostics.py
evaluations/baselines.py
evaluations/wb_era5_cases.py
evaluations/metrics.py
evaluations/aimip_metrics.py
evaluations/amip_validation.py
packages/core/README.md
packages/core/pyproject.toml
scripts/bench/bench_barotropic_mcut.py
scripts/bench/probe_w5_metrics.py
scripts/bench/__init__.py
scripts/bench/bench_mpas_spmd_scaling.py
scripts/bench/probe_spectral_shard.py
scripts/bench/sweep_Ah_global_wind.py
scripts/bench/bench_atm_latlon_spmd_scaling.py
scripts/bench/analyze_gpu_scaling.py
scripts/bench/aggregate_cube_shardmap_scaling.py
scripts/bench/bench_dd_scaling.py
scripts/bench/bench_crm_gpu_scaling.py
scripts/bench/bench_ocean_mpas_scaling.py
scripts/bench/bench_plane_dycore.py
scripts/bench/aggregate_bcw_scaling.py
scripts/bench/cfl_convergence_plane.py
scripts/bench/aggregate_scaling_results.py
scripts/bench/bench_halo_exchange.py
scripts/bench/bench_latlon_2d_halo.py
scripts/bench/bench_fv3_sw_fb_vs_production.py
scripts/bench/bench_halo_ops_scaling.py
scripts/bench/roofline_probe.py
scripts/bench/bench_cube_tiled_step_scaling.py
scripts/bench/validate_scaling.sh
scripts/bench/run_levante_gpu_scaling.py
scripts/bench/metadata.py
scripts/bench/bench_voronoi_partition_methods.py
scripts/bench/profile_mpas_ocean.py
scripts/bench/run_cpu_mpi_scaling.sh
scripts/bench/bench_les_plane.py
scripts/bench/profile_crm_step.py
scripts/bench/bench_spectral_les_dd_scaling.py
scripts/bench/run_cpu_mpi_scaling.py
scripts/bench/run_levante_gpu_scaling.sh
scripts/bench/bench_gather_vs_slice_stencil.py
scripts/bench/run_scaling_iter222_weak.sh
scripts/bench/bench_ocean_mpi_scaling.py
scripts/bench/bench_ocean_latlon_spmd_pcg.py
scripts/bench/run_cpu_mpi_scaling_local.sh
scripts/bench/run_strong_scaling_sweep.sh
scripts/bench/bench_ocean_gpu_scaling.py
packages/ocean/README.md
scripts/bench/bench_plane_crm_dd_scaling.py
packages/ocean/pyproject.toml
scripts/bench/run_scaling_diagnosis.py
scripts/bench/sweep_Ah_long.py
scripts/bench/bcw_scaling_ledger.py
scripts/bench/run_dd_scaling_sweep.sh
scripts/bench/bench_cube_shardmap_halo.py
scripts/bench/analyze_scaling_results.py
scripts/bench/scaling_summary.py
scripts/bench/bench_barotropic_precond_convergence.py
scripts/bench/aggregate_rcemip100.py
scripts/bench/parse_strong_sweep.py
scripts/bench/bench_coupled_latlon_scaling.py
scripts/bench/slurm_scaling_diagnosis.sh
scripts/bench/bench_spectral_transform_micro.py
scripts/bench/profile_cs_dycore.py
scripts/bench/bench_ppermute_microbench.py
scripts/bench/bench_mc3d_raytracer.py
scripts/bench/bench_ocean_latlon_spmd_scaling.py
scripts/bench/bench_mpi_scaling.py
tests/bench/test_scaling_metadata.py
tests/bench/test_scaling_moist_tier.py
tests/bench/test_aggregate_bcw_scaling.py
tests/bench/test_m1_accounting.py
tests/bench/test_ppermute_microbench.py
tests/bench/test_gather_vs_slice_stencil.py
tests/bench/test_aggregate_cube_shardmap_scaling.py
tests/bench/test_scaling_diagnosis_device_gate.py
tests/bench/test_bench_cube_shardmap_halo.py
tests/bench/test_timed_scan_blocks.py
tests/bench/test_halo_profiler_contract.py
tests/bench/test_bench_ocean_mpas_scaling.py
tests/bench/test_bench_voronoi_partition_methods.py
tests/bench/test_bench_cube_tiled_step_scaling.py
tests/bench/test_bcw_scaling_ledger.py
tests/bench/test_bench_spectral_transform_micro.py
tests/bench/test_latlon_mpi_wiring.py
tests/bench/test_factor_2d_latlon.py
tests/bench/test_bench_mpas_spmd_gates.py
tests/bench/test_bench_ocean_latlon_spmd_gates.py
packages/coupler/README.md
packages/coupler/pyproject.toml
data/bathymetry/etopo_1deg.nc
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

exec
/usr/bin/bash -lc 'rg -n -i "scal|tile.limit|pcg|single.reduce|wide.halo|barrier|metis|spectral|latency|fusion" project_status.md docs scripts/bench packages tests | head -500' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 713ms:
tests/bench/test_bench_spectral_transform_micro.py:1:"""Direct tests for the spectral-transform microbenchmark (audit item 9)."""
tests/bench/test_bench_spectral_transform_micro.py:12:          / "scripts" / "bench" / "bench_spectral_transform_micro.py")
tests/bench/test_bench_spectral_transform_micro.py:18:def test_cost_model_scaling():
tests/bench/test_bench_spectral_transform_micro.py:58:        # round-off scale, not the truncation scale.
tests/bench/test_bench_spectral_transform_micro.py:62:    assert md["grid"] == "spectral"
tests/bench/test_bench_spectral_transform_micro.py:63:    assert md["scaling_kind"] == "throughput"
tests/test_fv3_d_sw5_corner_corrections.py:154:    deltas (scaled by da_min_c * rarea_c) and zero everywhere else.
tests/test_fv3_d_sw5_corner_corrections.py:299:    # scale (~1e-16); allow a tiny absolute tolerance.  Magnitudes
tests/test_fv3_d_sw5_corner_corrections.py:551:# B-grid ghost exchange `ext_scalar(divgd, dg, bd, domain, 1, 1)`
tests/test_fv3_d_sw5_corner_corrections.py:552:# (fv_duogrid.F90::ext_scalar_3d: mpp NORTH+EAST corner halo update of the
tests/test_fv3_d_sw5_corner_corrections.py:558:def test_pad_corner_scalar_cross_face_geometry():
tests/test_fv3_d_sw5_corner_corrections.py:559:    """Each halo point of the padded corner scalar must be the neighbour's
tests/test_fv3_d_sw5_corner_corrections.py:564:    from legoesm.core.fv3_sw_core import _pad_corner_scalar_cross_face
tests/test_fv3_d_sw5_corner_corrections.py:573:    pts_pad = np.stack([np.asarray(_pad_corner_scalar_cross_face(a, n))
tests/test_fv3_d_sw5_corner_corrections.py:602:def test_pad_corner_scalar_cross_face_ghost_is_attenuated_neighbour_row():
tests/test_fv3_d_sw5_corner_corrections.py:604:    inside its shared edge (Fortran ext_scalar exchanges divgd AFTER
tests/test_fv3_d_sw5_corner_corrections.py:613:    from legoesm.core.fv3_sw_core import _pad_corner_scalar_cross_face
tests/test_fv3_d_sw5_corner_corrections.py:623:    padded = _pad_corner_scalar_cross_face(att, n)
tests/test_fv3_d_sw5_corner_corrections.py:650:    ext_scalar-faithful attenuated ghost destabilises the FB chain at
tests/bench/test_scaling_metadata.py:1:"""Direct tests for the shared scaling-benchmark metadata helper.
tests/bench/test_scaling_metadata.py:3:Locks the roadmap item-9 contract: every scaling record is self-describing,
tests/bench/test_scaling_metadata.py:31:    return md.scaling_metadata(**kw)
tests/bench/test_scaling_metadata.py:40:    assert md.validate_scaling_metadata(rec) == []
tests/bench/test_scaling_metadata.py:47:        solver_variant="single_reduce_pcg",
tests/bench/test_scaling_metadata.py:51:        scaling_kind="strong",
tests/bench/test_scaling_metadata.py:57:        "solver_residual", "cells_per_rank", "scaling_kind",
tests/bench/test_scaling_metadata.py:62:    assert rec["solver_variant"] == "single_reduce_pcg"
tests/bench/test_scaling_metadata.py:64:    assert rec["scaling_kind"] == "strong"
tests/bench/test_scaling_metadata.py:90:    # halo (a scaling bug that must be visible in the record, not hidden).
tests/bench/test_scaling_metadata.py:120:        md.validate_scaling_metadata(rec)
tests/bench/test_scaling_metadata.py:122:    assert md.validate_scaling_metadata(rec, strict=False) == ["grid"]
tests/bench/test_scaling_metadata.py:128:        md.validate_scaling_metadata(rec)
tests/bench/test_scaling_metadata.py:139:        assert k in md.validate_scaling_metadata(rec, strict=False), (
tests/bench/test_scaling_metadata.py:142:            md.validate_scaling_metadata(rec)
tests/bench/test_scaling_metadata.py:150:    assert "precision_knobs" in md.validate_scaling_metadata(rec, strict=False)
tests/bench/test_scaling_metadata.py:152:        md.validate_scaling_metadata(rec)
tests/bench/test_scaling_metadata.py:155:def test_validate_allows_scalar_zero_and_false():
tests/bench/test_scaling_metadata.py:161:    assert md.validate_scaling_metadata(rec) == []
tests/bench/test_scaling_metadata.py:167:    problems = md.validate_scaling_metadata(rec, strict=False)
tests/bench/test_scaling_metadata.py:174:    rec = md.scaling_metadata(
tests/bench/test_scaling_metadata.py:199:        extra={"note": "metis"},
tests/bench/test_scaling_metadata.py:202:    assert rec["extra"]["note"] == "metis"
tests/bench/test_scaling_metadata.py:207:    assert md.validate_scaling_metadata(rec) == []  # "none" != empty
tests/bench/test_scaling_metadata.py:221:# schema v2: transport / virtual_cpu_devices / launcher (anti-fake-scaling)
tests/bench/test_scaling_metadata.py:229:    assert md.validate_scaling_metadata(rec) == []
tests/bench/test_scaling_metadata.py:312:        assert k in md.validate_scaling_metadata(rec, strict=False)
tests/bench/test_scaling_metadata.py:314:            md.validate_scaling_metadata(rec)
tests/bench/test_scaling_metadata.py:334:    # Parity with the canonical run_cpu_mpi_scaling.py computation so SPMD
tests/bench/test_scaling_metadata.py:435:    # permute family stays bit-identical to the canonical scalar helper
tests/bench/test_ppermute_microbench.py:1:"""Contract tests for the ppermute latency/bandwidth microbenchmark.
tests/bench/test_ppermute_microbench.py:11:from bench_ppermute_microbench import fit_latency_bandwidth  # noqa: E402
tests/bench/test_ppermute_microbench.py:14:def test_fit_recovers_known_latency_and_bandwidth():
tests/bench/test_ppermute_microbench.py:19:    fit_lat, fit_bw = fit_latency_bandwidth(sizes, times)
tests/bench/test_ppermute_microbench.py:24:def test_fit_is_not_fooled_by_a_pure_latency_curve():
tests/bench/test_ppermute_microbench.py:25:    """A flat curve (all latency) must yield an enormous, not negative, BW."""
tests/bench/test_ppermute_microbench.py:28:    _, fit_bw = fit_latency_bandwidth(sizes, times)
tests/bench/test_ppermute_microbench.py:33:    """A latency/bandwidth fit from one device would be meaningless."""
tests/timestepping/test_ssp_rk54_scan_equivalence.py:226:def test_scalar_python_tendency_leaf():
tests/timestepping/test_ssp_rk54_scan_equivalence.py:227:    """A weakly-typed Python-scalar tendency leaf is accepted, matching inline.
tests/timestepping/test_ssp_rk54_scan_equivalence.py:229:    The inline form and the carry ``.set`` both coerce Python scalars, so the
tests/timestepping/test_ssp_rk54_scan_equivalence.py:242:    assert np.isfinite(float(out_scan["y"][0])), "scalar-tendency scan non-finite"
tests/timestepping/test_ssp_rk54_scan_equivalence.py:246:        err_msg="scalar-tendency scan diverged from inline")
tests/timestepping/test_ssp_rk54_scan_equivalence.py:268:    def loss(scale, stepper):
tests/timestepping/test_ssp_rk54_scan_equivalence.py:269:        s = {"u": scale * jnp.ones((4, 6)),
tests/timestepping/test_ssp_rk54_scan_equivalence.py:270:             "T": 300.0 + scale * jnp.ones((4, 6))}
tests/bench/test_scaling_diagnosis_device_gate.py:1:"""Direct tests for the ``--expect-devices`` anti-fake-scaling gate.
tests/bench/test_scaling_diagnosis_device_gate.py:3:A scaling/census row is only meaningful if the run actually spanned the device
tests/bench/test_scaling_diagnosis_device_gate.py:19:        / "scripts" / "bench" / "run_scaling_diagnosis.py")
tests/bench/test_scaling_diagnosis_device_gate.py:20:_spec = importlib.util.spec_from_file_location("run_scaling_diagnosis", _MOD)
scripts/bench/bench_barotropic_mcut.py:1:"""Distributed barotropic-PCG iteration-count (M) cut: jacobi vs banded multigrid.
scripts/bench/bench_barotropic_mcut.py:5:per-step GLOBAL-reduction count is ``2*M`` — the multinode weak-scaling
scripts/bench/bench_barotropic_mcut.py:6:reduction-latency wall.  The banded geometric-multigrid preconditioner
scripts/bench/bench_barotropic_mcut.py:14:it a weak-scaling win, not just a per-device one).
scripts/bench/bench_barotropic_mcut.py:20:Writes ``docs/performance/scaling/barotropic_mcut.csv`` on rank 0 for the plotter.
scripts/bench/bench_barotropic_mcut.py:66:    p.add_argument("--out", default="docs/performance/scaling/barotropic_mcut.csv")
scripts/bench/bench_barotropic_mcut.py:68:    # reduction-latency win shows in ms/solve (Gloo allreduce dominates at scale).
scripts/bench/bench_barotropic_mcut.py:120:            pcg_variant="standard", dot_weight=w)
scripts/bench/bench_barotropic_mcut.py:133:    # At MULTI-NODE the Gloo allreduce latency dominates the barotropic (phase
scripts/bench/bench_barotropic_mcut.py:135:    # — single_reduce (1 allreduce/iter = M reductions, same jacobi per-iter)
scripts/bench/bench_barotropic_mcut.py:139:    # single_reduce = M.
scripts/bench/bench_barotropic_mcut.py:147:                pcg_variant=variant, dot_weight=w)[0])
scripts/bench/bench_barotropic_mcut.py:149:            comm.Barrier()
scripts/bench/bench_barotropic_mcut.py:153:            comm.Barrier()
scripts/bench/bench_barotropic_mcut.py:161:             "single_reduce", args.jacobi_m),
docs/ocean/fidelity/phase_b1_cube_bottom_drag.md:49:  * Bump `barotropic_diffusion_alpha` from 0.05 to 0.3.
tests/bench/test_latlon_mpi_wiring.py:1:"""Guard the lat-lon MPI wiring in ``run_levante_gpu_scaling.py`` (issue #641).
tests/bench/test_latlon_mpi_wiring.py:30:    / "scripts" / "bench" / "run_levante_gpu_scaling.py"
tests/bench/test_latlon_mpi_wiring.py:36:        "run_levante_gpu_scaling", _HARNESS,
tests/bench/test_latlon_mpi_wiring.py:75:    (Multi-rank lat-lon scaling goes through MPI, where ``fixed_gpu_count``
tests/test_smagorinsky_biharmonic_comprehensive.py:166:        scale_u = jnp.maximum(jnp.max(jnp.abs(grad_u_jax[:, 1:n_lon])), 1e-30)
tests/test_smagorinsky_biharmonic_comprehensive.py:167:        rel_u = float(jnp.max(diff_u_int) / scale_u)
tests/test_smagorinsky_biharmonic_comprehensive.py:171:        scale_v = jnp.maximum(jnp.max(jnp.abs(grad_v_jax)), 1e-30)
tests/test_smagorinsky_biharmonic_comprehensive.py:172:        rel_v = float(jnp.max(diff_v) / scale_v)
tests/test_smagorinsky_biharmonic_comprehensive.py:324:        scale_u = jnp.maximum(jnp.max(jnp.abs(tend_u_st[:, 1:n_lon])), 1e-30)
tests/test_smagorinsky_biharmonic_comprehensive.py:325:        scale_v = jnp.maximum(jnp.max(jnp.abs(tend_v_st)), 1e-30)
tests/test_smagorinsky_biharmonic_comprehensive.py:327:        rel_u = float(jnp.max(diff_u) / scale_u)
tests/test_smagorinsky_biharmonic_comprehensive.py:328:        rel_v = float(jnp.max(diff_v) / scale_v)
tests/bench/test_scaling_moist_tier.py:1:"""Direct tests for the grid-agnostic 'moist' (Kessler) scaling tier.
tests/bench/test_scaling_moist_tier.py:3:These cover the wiring added so the GPU scaling driver can run a moist
tests/bench/test_scaling_moist_tier.py:5:path — so a fair cross-grid many-GPU scaling comparison uses the same column
tests/bench/test_scaling_moist_tier.py:12:scaling driver); it cannot run inside a single-process pytest.
tests/bench/test_scaling_moist_tier.py:26:    / "scripts" / "bench" / "run_levante_gpu_scaling.py"
tests/bench/test_scaling_moist_tier.py:32:        "run_levante_gpu_scaling", _SCRIPT
tests/bench/test_scaling_moist_tier.py:37:    sys.modules["run_levante_gpu_scaling"] = mod
tests/bench/test_scaling_moist_tier.py:44:_ALL_GRIDS = ("spectral", "latlon", "icosahedral", "cubed-sphere")
tests/bench/test_scaling_moist_tier.py:74:    rl._validate_physics("spectral", "moist")
tests/bench/test_scaling_moist_tier.py:82:        rl._validate_physics("spectral", "rrtmg_full")
tests/bench/test_aggregate_bcw_scaling.py:1:"""Direct tests for the baroclinic-wave scaling collector (pure stdlib path)."""
tests/bench/test_aggregate_bcw_scaling.py:14:_AGG = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "aggregate_bcw_scaling.py"
tests/bench/test_aggregate_bcw_scaling.py:15:_spec = importlib.util.spec_from_file_location("aggregate_bcw_scaling", _AGG)
tests/bench/test_aggregate_bcw_scaling.py:31:        "scaling_efficiency": 1.0, "compile_time_s": 2.0,
tests/bench/test_aggregate_bcw_scaling.py:85:    d = tmp_path / "scaling_cpu_ocean"
tests/bench/test_aggregate_bcw_scaling.py:104:    # run_levante_gpu_scaling.py writes a nested report with mode 'strong'
tests/bench/test_aggregate_bcw_scaling.py:107:    d = tmp_path / "scaling" / "20260624T2252Z"
tests/bench/test_aggregate_bcw_scaling.py:117:    _write(d, "strong_scaling.json", payload)
tests/bench/test_aggregate_bcw_scaling.py:127:    a = tmp_path / "bcw_scaling" / "dry_icosahedral_cpu_np16_1"
tests/bench/test_aggregate_bcw_scaling.py:128:    o = tmp_path / "scaling_cpu_ocean"
tests/bench/test_aggregate_bcw_scaling.py:163:    d = tmp_path / "scaling" / "mystery"
tests/bench/test_aggregate_bcw_scaling.py:170:    _write(d, "strong_scaling.json", payload)
tests/bench/test_aggregate_bcw_scaling.py:177:    run_cpu_mpi_scaling.py --cs-spmd (grid_type='cubed-sphere', device
tests/bench/test_aggregate_bcw_scaling.py:254:# SPMD bench-lane JSONL ingestion (bench_*_spmd_scaling append-per-line recs)
tests/bench/test_aggregate_bcw_scaling.py:258:    """A bench_atm_latlon_spmd_scaling-shaped flat record (one JSONL line)."""
tests/bench/test_aggregate_bcw_scaling.py:280:    _write_jsonl(tmp_path / "latlon_gpu_1", "spmd_scaling.jsonl", [
tests/bench/test_aggregate_bcw_scaling.py:310:    # hardware scaling; the aggregator must never chart it as a CPU curve.
tests/bench/test_aggregate_bcw_scaling.py:311:    _write_jsonl(tmp_path / "latlon_cpu_1", "spmd_scaling.jsonl", [
tests/grids/test_fv3_native_metrics_phase2.py:50:def _rot_scalar(field: np.ndarray, k: int) -> np.ndarray:
tests/grids/test_fv3_native_metrics_phase2.py:58:        _rot_scalar(oracle_field[_GNOMONIC_ED_FACE_PERM[F]],
tests/bench/test_bench_cube_shardmap_halo.py:79:    assert result["scaling"][0]["n_devices"] == 1
tests/bench/test_bench_cube_shardmap_halo.py:80:    assert result["scaling"][0]["ms_per_step"] > 0.0
tests/bench/test_bench_cube_shardmap_halo.py:89:    exercised neither SPMD, AD, nor scaling."""
tests/bench/test_bench_cube_shardmap_halo.py:101:# --- Hermetic 6-emulated-device subprocess (real SPMD + AD + scaling) -------
tests/bench/test_bench_cube_shardmap_halo.py:131:    counts = [r["n_devices"] for r in result["scaling"]]
tests/bench/test_bench_cube_shardmap_halo.py:145:    for r in result["scaling"]:
tests/bench/test_bench_cube_shardmap_halo.py:186:    import scripts.bench.aggregate_cube_shardmap_scaling as agg
tests/bench/test_bench_cube_shardmap_halo.py:197:    # Non-decisive: CPU efficiency anti-scales, so gate only correctness of the
tests/bench/test_bench_cube_shardmap_halo.py:210:    # (CPU anti-scales) + require the exact counts present -> PASS.
tests/bench/test_factor_2d_latlon.py:1:"""Direct tests for ``run_cpu_mpi_scaling._factor_2d_latlon`` — the
tests/bench/test_factor_2d_latlon.py:22:         / "scripts" / "bench" / "run_cpu_mpi_scaling.py")
tests/bench/test_factor_2d_latlon.py:23:_spec = importlib.util.spec_from_file_location("_run_cpu_mpi_scaling", _PATH)
tests/bench/test_factor_2d_latlon.py:25:sys.modules["_run_cpu_mpi_scaling"] = _mod
scripts/bench/bench_mpas_spmd_scaling.py:1:"""Strong scaling bench for the device-sharded icosahedral/MPAS (TRiSK)
scripts/bench/bench_mpas_spmd_scaling.py:5:The Voronoi twin of ``bench_atm_latlon_spmd_scaling.py`` (mirrored
scripts/bench/bench_mpas_spmd_scaling.py:7:``reorder_voronoi_for_sharding`` (METIS/RCB/Hilbert-SFC cell partition, ghost-
scripts/bench/bench_mpas_spmd_scaling.py:13:  (Weak scaling rides the subdivision ladder: one level = 4x the cells, so
scripts/bench/bench_mpas_spmd_scaling.py:21:scalability: at ``--nlev 26`` the per-cell compute grows ~3.25x, the flat wall
scripts/bench/bench_mpas_spmd_scaling.py:40:``--partition-method auto`` to METIS and another to RCB — would silently
scripts/bench/bench_mpas_spmd_scaling.py:44:  srun -n 6 python bench_mpas_spmd_scaling.py --multicontroller \
scripts/bench/bench_mpas_spmd_scaling.py:49:  JAX_ENABLE_X64=1 python scripts/bench/bench_mpas_spmd_scaling.py \
scripts/bench/bench_mpas_spmd_scaling.py:66:# baroclinic-wave IC the icosahedral lanes of run_levante_gpu_scaling use).
scripts/bench/bench_mpas_spmd_scaling.py:71:# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
scripts/bench/bench_mpas_spmd_scaling.py:76:    annotate_incomplete, hlo_collective_census, scaling_metadata,
scripts/bench/bench_mpas_spmd_scaling.py:104:    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
scripts/bench/bench_mpas_spmd_scaling.py:130:    # Same recipe as the icosahedral lane of run_levante_gpu_scaling /
scripts/bench/bench_mpas_spmd_scaling.py:131:    # tests/parallel/test_voronoi_sharded_equivalence.py: del4 hyperdiffusion,
scripts/bench/bench_mpas_spmd_scaling.py:154:    # gates (small smoke scales).  The mesh itself is still global per
scripts/bench/bench_mpas_spmd_scaling.py:185:                        "production SCVT; 0 = labelled synthetic scaling "
scripts/bench/bench_mpas_spmd_scaling.py:186:                        "mesh (scaling receipts only, never physics — "
scripts/bench/bench_mpas_spmd_scaling.py:195:                   choices=["auto", "geometric", "metis", "sfc"],
scripts/bench/bench_mpas_spmd_scaling.py:220:                   default="results/a1/mpas_spmd_scaling.jsonl")
scripts/bench/bench_mpas_spmd_scaling.py:324:        # --partition-method auto to METIS, another to RCB) cannot slip
scripts/bench/bench_mpas_spmd_scaling.py:360:    # initial state — build it lazily here, at their smoke scales only
scripts/bench/bench_mpas_spmd_scaling.py:431:    # ONE compile → full per-family census; the CP scalar (the #1113 round-count
scripts/bench/bench_mpas_spmd_scaling.py:465:                # far below the q_v scale; reuse the T tolerances).
scripts/bench/bench_mpas_spmd_scaling.py:501:        # lloyd=0 is the LABELLED synthetic scaling mesh — anti-masquerade:
scripts/bench/bench_mpas_spmd_scaling.py:528:    # twin): resolution = subdivision level, matching run_cpu_mpi_scaling's
scripts/bench/bench_mpas_spmd_scaling.py:542:    rec["metadata"] = annotate_incomplete(scaling_metadata(
scripts/bench/bench_mpas_spmd_scaling.py:556:        scaling_kind="strong",  # this bench fixes the mesh and sweeps devices
scripts/bench/bench_mpas_spmd_scaling.py:581:                  "hardware scaling — do not report it as a speedup.")
tests/bench/test_aggregate_cube_shardmap_scaling.py:1:"""Direct tests for the cross-node cube shard_map scaling aggregator (pure Python,
tests/bench/test_aggregate_cube_shardmap_scaling.py:12:import scripts.bench.aggregate_cube_shardmap_scaling as agg
tests/bench/test_aggregate_cube_shardmap_scaling.py:122:    _pt(tmp_path, 1, 10.0, None)             # nothing to scale
tests/bench/test_aggregate_cube_shardmap_scaling.py:172:    (d / "results.json").write_text(json.dumps({"mode": "sweep", "scaling": []}))
tests/bench/test_aggregate_cube_shardmap_scaling.py:246:    assert (out / "scaling.csv").exists()
tests/bench/test_aggregate_cube_shardmap_scaling.py:249:    rows = (out / "scaling.csv").read_text().strip().splitlines()
tests/timestepping/test_leapfrog_ab2.py:66:# 2. Linear test: scalar exponential decay
tests/timestepping/test_ssp_rk3_scan_bit_equivalence.py:60:    # Compute a scalar coupling factor from p_s + first T column.
tests/timestepping/test_ssp_rk3_scan_bit_equivalence.py:144:    a float64 scalar; multiplying with a float32 state leaf upcasts the
tests/bench/test_timed_scan_blocks.py:5:being silently clamped), the separate latency-vs-fused reporting, the
tests/bench/test_timed_scan_blocks.py:17:from metadata import git_sha, scaling_metadata, timed_scan_blocks  # noqa: E402
tests/bench/test_timed_scan_blocks.py:45:    for k in ("compile_ms", "scan_compile_ms", "step_latency_ms",
tests/bench/test_timed_scan_blocks.py:104:    # atol at ULP scale: a pre-compile that ADVANCED the live seed (an
tests/bench/test_timed_scan_blocks.py:112:    md = scaling_metadata(grid="latlon", component="ocean", resolution="8x16",
scripts/bench/probe_spectral_shard.py:1:"""Probe whether wrapping the spectral PE step in a jit-with-sharding
scripts/bench/probe_spectral_shard.py:2:on emulated multi-CPU devices actually scales.  Bypasses
scripts/bench/probe_spectral_shard.py:8:        PYTHONPATH=. .venv/bin/python scripts/probe_spectral_shard.py
scripts/bench/probe_spectral_shard.py:23:                        help="spectral truncation T<n>")
scripts/bench/probe_spectral_shard.py:43:    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
scripts/bench/probe_spectral_shard.py:44:        SpectralPrimitiveEquationModel, SpectralPEConfig,
scripts/bench/probe_spectral_shard.py:45:        isothermal_rest_state_spectral, spectral_pe_tendencies,
scripts/bench/probe_spectral_shard.py:52:    state = isothermal_rest_state_spectral(grid, sigma, perturbation_amplitude=0.1)
scripts/bench/probe_spectral_shard.py:53:    model = SpectralPrimitiveEquationModel(grid, sigma, SpectralPEConfig())
scripts/bench/probe_spectral_shard.py:63:        return spectral_pe_tendencies(
scripts/bench/probe_spectral_shard.py:89:        # The level-shard path wraps a single ``_do_step`` (no scan fusion), so
docs/ocean/fidelity/phase_e_climate_infra.md:1:# Phase E — Climate-scale forcing infrastructure
tests/bench/test_halo_profiler_contract.py:1:"""Contract tests for `profile_halo_exchange` (scaling diagnosis phase 4).
tests/bench/test_halo_profiler_contract.py:13:from legoesm.parallel.scaling_diagnostics import profile_halo_exchange
tests/bench/test_halo_profiler_contract.py:15:LABELS = ("scalar_3d", "scalar_4d", "vector_4d")
tests/bench/test_halo_profiler_contract.py:65:    from legoesm.parallel.scaling_diagnostics import estimate_overlap_potential
tests/bench/test_bench_mpas_spmd_gates.py:1:"""Guard the parity/conservation gates of ``bench_mpas_spmd_scaling``.
tests/bench/test_bench_mpas_spmd_gates.py:32:    / "scripts" / "bench" / "bench_mpas_spmd_scaling.py"
tests/bench/test_bench_mpas_spmd_gates.py:38:        "bench_mpas_spmd_scaling", _BENCH,
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:2:"""Side-by-side NH dycore comparison: spectral vs lat-lon C-grid.
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:4:Runs the spectral-NH (T21/L10) and lat-lon-C-grid-NH (32x64/L10)
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:39:# Spectral imports
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:49:from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:50:    SpectralNHConfig,
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:51:    SpectralCompressibleEulerModel,
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:52:    nh_rest_state_spectral,
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:99:    print("  NH Dycore Comparison: Spectral vs lat-lon C-grid")
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:111:    # Spectral side: T21 / L10
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:114:    print("  Spectral dycore: T21 / L10 / 1 hour")
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:123:    # Hyperdiff coefficient sized off the truncation; mirror bench_spectral_nh.
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:127:    cfg_sp = SpectralNHConfig(
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:134:    model_sp = SpectralCompressibleEulerModel(
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:139:    # Build IC: rest state + theta' on the spectral grid.
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:140:    state_sp = nh_rest_state_spectral(grid_sp, hcoord_sp)
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:152:    # We leave rho' = 0 spectrally: the spectral dycore was tuned that
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:159:    print(f"  Spectral JIT compiled in {time.time()-t0:.1f}s")
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:196:    print(f"  Spectral done in {time.time()-t_wall:.1f}s")
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:198:        f"  Spectral final: max|w|={sp_max_w[-1]:.4f}, "
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:202:    # Spectral snapshot at t = 60 min for the side-by-side profile plot.
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:221:    # Horizontal Laplacian hyperdiffusion on the C-grid -- mirror what
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:222:    # the spectral side gets from its bilaplacian truncation control.
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:223:    # Scale chosen so the dissipation timescale at the grid wavelength
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:224:    # matches the spectral side at T21 (~ 1.5e6 m^2/s when read in the
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:236:        hyperdiff_scalar=1.0e6,
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:322:    axes[0].plot(sp_times_a, sp_max_w_a, "b-",  lw=1.7, label="spectral T21")
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:368:    axes[0].set_title(f"spectral T21 (t = {snap_t_min} min)")
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:404:    axes[0].set_title(f"spectral T21 (t = {snap_t_min} min)")
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:444:    summary.append("NH dycore comparison summary (spectral vs lat-lon C-grid)")
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:450:        f"  Spectral grid: T{n_max} ({grid_sp.n_lat} x {grid_sp.n_lon})"
tests/validation/bench_nh_spectral_vs_cgrid_compare.py:457:        f"  Spectral final  max|w|={sp_max_w_a[-1]:.4f}  "
tests/ice/unit/test_ice_ridging_faithful.py:8:    independent scalar reimplementation;
tests/ice/unit/test_ice_ridging_faithful.py:61:_O_H_STAR = 100.0         # maximum ridge thickness scale [m]
tests/ice/unit/test_ice_ridging_faithful.py:397:    def _snow_flux(closing_scalar, cast):
tests/ice/unit/test_ice_ridging_faithful.py:399:                            closing_scalar.reshape((1,)), n_cat, 3600.0,
docs/ocean/fidelity/phase_f_long_runs.md:1:# Phase F -- Climate-scale ocean run drivers
docs/ocean/fidelity/phase_f_long_runs.md:55:  fields via a forward-Euler ``rho_0 * c_p * dz_0`` rescaling
tests/bench/test_m1_accounting.py:21:    DEFAULT_COMM_LATENCY_US,
tests/bench/test_m1_accounting.py:45:        latency_us=10.0,
tests/bench/test_m1_accounting.py:71:        latency_us=10.0,          # comm = 10 ms > compute = 1 ms
tests/bench/test_m1_accounting.py:92:    assert out["bound_ingredients"]["latency_us"] == DEFAULT_COMM_LATENCY_US
tests/bench/test_m1_accounting.py:100:    placeholder latency/bandwidth would fabricate the number — codex
tests/bench/test_m1_accounting.py:109:        # latency_us / bandwidth_GBs left to the placeholders
tests/bench/test_m1_accounting.py:115:    assert any("latency_us" in r for r in reasons)
tests/bench/test_m1_accounting.py:166:    """Passing only ONE of latency/bandwidth still uses a placeholder."""
tests/bench/test_m1_accounting.py:171:        latency_us=3.0,           # bandwidth left to the placeholder
tests/bench/test_m1_accounting.py:178:        calibrated_bound(bandwidth_GBs=0.0, latency_us=1.0)
tests/bench/test_m1_accounting.py:180:        calibrated_bound(bandwidth_GBs=1.0, latency_us=-1.0)
packages/ocean/legoesm/ocean/diagnostics_climate.py:1:"""Climate-scale ocean diagnostics for multi-decade evaluation.
packages/ocean/legoesm/ocean/diagnostics_climate.py:110:    Reduces the (n_lat, n_lon) barotropic streamfunction to a scalar
tests/grids/test_fv3_native_duo_phase4c.py:180:    divergence (exact scalar equality, not tolerance) — codex p4c r2 P2."""
tests/grids/test_fv3_native_duo_phase4c.py:229:    # each at an interior partner index; EXACT 1/4 (scalar equality)
packages/ocean/legoesm/ocean/__init__.py:13:Spectral (Gaussian)            ``SpectralOceanModel``
packages/ocean/legoesm/ocean/__init__.py:34:    SpectralOceanState,
packages/ocean/legoesm/ocean/__init__.py:35:    SpectralOceanConfig,
packages/ocean/legoesm/ocean/__init__.py:57:# Lazy imports for heavier/less-common subsystems (spectral, MPAS,
packages/ocean/legoesm/ocean/__init__.py:62:    # Spectral ocean
packages/ocean/legoesm/ocean/__init__.py:63:    "SpectralOceanModel": ("legoesm.ocean.dynamics.spectral_ocean_pe", "SpectralOceanModel"),
packages/ocean/legoesm/ocean/__init__.py:64:    "rest_state_spectral_ocean": ("legoesm.ocean.dynamics.spectral_ocean_pe", "rest_state_spectral_ocean"),
packages/ocean/legoesm/ocean/__init__.py:120:    "SpectralOceanModel",
packages/ocean/legoesm/ocean/__init__.py:126:    "SpectralOceanState",
packages/ocean/legoesm/ocean/__init__.py:127:    "SpectralOceanConfig",
packages/ocean/legoesm/ocean/__init__.py:135:    "rest_state_spectral_ocean",
packages/ocean/legoesm/ocean/__init__.py:201:    # General lazy imports (spectral, MPAS, lat-lon, biogeo, bathymetry, freshwater)
tests/bench/test_bench_ocean_latlon_spmd_gates.py:1:"""Guard the parity/conservation gates of ``bench_ocean_latlon_spmd_scaling``.
tests/bench/test_bench_ocean_latlon_spmd_gates.py:4:``bench_ocean_mpi_scaling``) are the fail-fast correctness armor the
tests/bench/test_bench_ocean_latlon_spmd_gates.py:31:    / "scripts" / "bench" / "bench_ocean_latlon_spmd_scaling.py"
tests/bench/test_bench_ocean_latlon_spmd_gates.py:37:        "bench_ocean_latlon_spmd_scaling", _BENCH,
tests/bench/test_bench_ocean_latlon_spmd_gates.py:89:    # item 6: default implicit_cn at nd=2 runs the fixed-iteration PCG.
tests/bench/test_bench_ocean_latlon_spmd_gates.py:95:    assert rec["solver_iters_mode"].startswith("fixed_pcg")
tests/bench/test_bench_ocean_latlon_spmd_gates.py:121:    assert any("latency_us" in r for r in rec["bound_incomplete_reason"])
tests/validation/run_dycore_progression_suite.py:5:1) Shallow-water (spectral lat-lon, FV lat-lon, FV cube-sphere)
tests/validation/run_dycore_progression_suite.py:6:2) Hydrostatic core (FV cube, FV lat-lon, spectral)
tests/validation/run_dycore_progression_suite.py:7:3) Non-hydrostatic core (FV cube, spectral, harder FV cube cases)
tests/validation/run_dycore_progression_suite.py:47:    "01_latlon_spectral_williamson2": "01_sw_spectral_latlon",
tests/validation/run_dycore_progression_suite.py:54:    SuiteCase(1, "01_sw_spectral_latlon", "SW Spectral Williamson-2 (lat-lon)", "shallow", "01_latlon_spectral_williamson2"),
tests/validation/run_dycore_progression_suite.py:59:    SuiteCase(6, "06_hydro_spectral_held_suarez", "Hydro Spectral Held-Suarez (gaussian lat-lon)", "atmosphere", "hydro_spec_hs"),
tests/validation/run_dycore_progression_suite.py:61:    SuiteCase(8, "08_hydro_spectral_baroclinic", "Hydro Spectral Baroclinic Wave (gaussian lat-lon)", "atmosphere", "hydro_spec_bw"),
tests/validation/run_dycore_progression_suite.py:63:    SuiteCase(10, "10_nh_spectral_tc1", "NH Spectral DCMIP TC1 (gaussian lat-lon)", "atmosphere", "nh_spec_tc1"),
tests/validation/run_dycore_progression_suite.py:264:    parser.add_argument("--spectral-truncation", type=int, default=42)
tests/validation/run_dycore_progression_suite.py:342:            "--spectral-truncation",
tests/validation/run_dycore_progression_suite.py:343:            str(args.spectral_truncation),
tests/validation/run_dycore_progression_suite.py:410:            "--spectral-truncation",
tests/validation/run_dycore_progression_suite.py:411:            str(args.spectral_truncation),
tests/validation/run_dycore_progression_suite.py:467:            "spectral_truncation": args.spectral_truncation,
tests/test_matrix_nh_cube_parity_ast_guard.py:12:runner had remained on the legacy scalar-halo path, which produced
tests/test_matrix_nh_cube_parity_ast_guard.py:13:cube |w|_max 22x worse than ico/spectral on TC1, 13x on TC2, and
tests/test_matrix_nh_cube_parity_ast_guard.py:414:    alongside the pre-existing ``icosahedral`` and ``spectral``.
tests/test_matrix_nh_cube_parity_ast_guard.py:427:    for grid in ("cubed_sphere", "latlon", "icosahedral", "spectral"):
tests/test_matrix_nh_cube_parity_ast_guard.py:544:    # optional #753 ``scaling_exponent=`` kwarg).
tests/test_matrix_nh_cube_parity_ast_guard.py:570:    # that the matrix reads them via ``str(MODON_HYPERDIFF_{FACTOR,SCALING})``
tests/test_matrix_nh_cube_parity_ast_guard.py:579:        r'LEGOESM_SW_MODON_HYPERDIFF_SCALING"\s*,\s*str\(MODON_HYPERDIFF_SCALING\)',
tests/test_matrix_nh_cube_parity_ast_guard.py:581:        "LEGOESM_SW_MODON_HYPERDIFF_SCALING default must be str(MODON_HYPERDIFF_"
tests/test_matrix_nh_cube_parity_ast_guard.py:582:        "SCALING) (the shared #800 source of truth), not a re-hardcoded literal."
tests/test_matrix_nh_cube_parity_ast_guard.py:599:        MODON_HYPERDIFF_SCALING,
tests/test_matrix_nh_cube_parity_ast_guard.py:603:    # (ref/n)^2 law (scaling 2, the #753 item-1 default: C96 erupts under ^4 but
tests/test_matrix_nh_cube_parity_ast_guard.py:607:    assert MODON_HYPERDIFF_SCALING == 2
tests/test_matrix_nh_cube_parity_ast_guard.py:642:    spectral filter at ``(5, 6)``, line 4068 NH ``(11, 12)``) can't
tests/test_matrix_nh_cube_parity_ast_guard.py:649:    # kwarg).  This rules out the spectral filter conditional + the
tests/test_matrix_nh_cube_parity_ast_guard.py:1132:    rescale-positives to ``_mass_target_iter61``) at the tail of its
tests/test_matrix_nh_cube_parity_ast_guard.py:1139:    CB fixer (transport_step's clip+rescale logic) and applied it to
tests/test_matrix_nh_cube_parity_ast_guard.py:1144:    A refactor that drops the fixer or breaks the rescale call would
tests/test_matrix_nh_cube_parity_ast_guard.py:1148:    ``mass_target / max(mass_pos, 1.0)`` rescale pattern.
tests/test_matrix_nh_cube_parity_ast_guard.py:1155:        "fixer (clip negatives, rescale positives) on the latlon CB "
tests/test_matrix_nh_cube_parity_ast_guard.py:1163:        "iter-61 regression: latlon CB rescale pattern "
tests/test_matrix_nh_cube_parity_ast_guard.py:1165:        "The rescale is what brings 12-day drift from 5e-4 to 2e-8."
tests/test_matrix_nh_cube_parity_ast_guard.py:1220:    NOT be switched to the multiplicative-rescale-to-initial-mass
tests/test_matrix_nh_cube_parity_ast_guard.py:1227:    hexagons; cell-area ratio ~83 %); multiplicative rescale of
tests/test_matrix_nh_cube_parity_ast_guard.py:1245:        "RESULT showed multiplicative-rescale on the heterogeneous "
tests/test_matrix_nh_cube_parity_ast_guard.py:1253:    # ``h_pos * scale`` pattern cube/latlon use.
tests/test_matrix_nh_cube_parity_ast_guard.py:1276:        "multiplicative-rescale constant.  This was REVERTED "
tests/ice/unit/test_ice_shortwave_faithful.py:306:    # CANARY: thin-ice albedo scales as sqrt(h) (the surrogate ramp), so 4x
packages/ocean/legoesm/ocean/rpe.py:60:    state : OceanState (cubed-sphere / lat-lon / MPAS / spectral)
packages/ocean/legoesm/ocean/rpe.py:110:        # lat-lon C-grid / regional / channel / spectral all expose
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:78:    spectral_land_lat_threshold: float = 90.0  # No land for spectral grid
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:98:    tau_relax_days: float = 15.0       # Temperature relaxation timescale [days]
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:99:    drag_timescale: float = 25.0       # Momentum damping timescale [days]
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:114:        Grid type: "cubed_sphere", "latlon", "mpas", or "spectral"
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:130:    - Spectral grid: uses vorticity/divergence formulation
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:162:    elif grid_type == "spectral":
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:163:        from legoesm.ocean.dynamics.spectral_ocean_pe import rest_state_spectral_ocean
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:164:        state = rest_state_spectral_ocean(
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:167:            land_lat_threshold=config.spectral_land_lat_threshold
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:181:    if grid_type == "spectral":
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:182:        return _add_phillips_perturbation_spectral(state, grid, z_coord, config)
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:187:def _add_phillips_perturbation_spectral(state, grid, z_coord, config):
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:188:    """Add Phillips perturbation for spectral grid (vorticity/divergence)."""
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:399:            "drag_timescale_days": config.drag_timescale,
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:419:        "spectral_land_lat_threshold": config.spectral_land_lat_threshold,
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:562:def get_scalar_units() -> Dict[str, str]:
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:563:    """Get units for scalar diagnostic quantities.
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:568:        Mapping from scalar field names to units
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:590:    "get_scalar_units": get_scalar_units,
packages/ocean/legoesm/ocean/experiments/phillips_two_layer.py:602:        "spectral": True
tests/validation/bench_spectral_nh.py:2:"""Benchmark: Spectral Non-Hydrostatic — Rest-State + DCMIP Gravity Waves.
tests/validation/bench_spectral_nh.py:10:Outputs (in results/atmosphere/nonhydrostatic/spectral_nh/):
tests/validation/bench_spectral_nh.py:28:# Ensure project root is on sys.path so spectral_nh can import test_cases
tests/validation/bench_spectral_nh.py:50:from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
tests/validation/bench_spectral_nh.py:51:    SpectralNHConfig,
tests/validation/bench_spectral_nh.py:52:    SpectralCompressibleEulerModel,
tests/validation/bench_spectral_nh.py:53:    nh_rest_state_spectral,
tests/validation/bench_spectral_nh.py:54:    dcmip25_tc1_init_spectral,
tests/validation/bench_spectral_nh.py:60:OUT_DIR = os.path.join(PROJECT_DIR, "results", "atmosphere", "nonhydrostatic", "spectral_nh")
tests/validation/bench_spectral_nh.py:71:    """Extract grid-point fields from spectral NH state."""
tests/validation/bench_spectral_nh.py:91:    print("  Spectral Non-Hydrostatic Benchmark")
tests/validation/bench_spectral_nh.py:105:    config_rest = SpectralNHConfig(
tests/validation/bench_spectral_nh.py:111:    model_rest = SpectralCompressibleEulerModel(
tests/validation/bench_spectral_nh.py:115:    state_rest = nh_rest_state_spectral(grid21, hcoord10)
tests/validation/bench_spectral_nh.py:161:        state0, hcoord_tc1, terrain_tc1 = dcmip25_tc1_init_spectral(
tests/validation/bench_spectral_nh.py:171:        state0 = nh_rest_state_spectral(grid_tc1, hcoord_tc1)
tests/validation/bench_spectral_nh.py:189:    config_tc1 = SpectralNHConfig(
tests/validation/bench_spectral_nh.py:196:    model_tc1 = SpectralCompressibleEulerModel(
tests/validation/bench_spectral_nh.py:392:    summary.append("Spectral Non-Hydrostatic Benchmark Summary")
tests/test_bcw_benchmark_scan_steps.py:36:    iter-212 honest measurements: spectral=24 (+29..45 %),
tests/test_bcw_benchmark_scan_steps.py:43:        "spectral":     24,
tests/test_bcw_benchmark_scan_steps.py:115:    """Spectral and cubed-sphere with scan>1 must resolve to the user
tests/test_bcw_benchmark_scan_steps.py:121:        assert bcw.resolve_scan_steps("spectral", 12, verbose=False) == 12
tests/bench/test_bench_ocean_mpas_scaling.py:1:"""Direct tests for the MPAS/Voronoi OCEAN scaling lane (audit item 6).
tests/bench/test_bench_ocean_mpas_scaling.py:23:          / "scripts" / "bench" / "bench_ocean_mpas_scaling.py")
tests/bench/test_bench_ocean_mpas_scaling.py:189:    # Shared self-describing record (schema v2) — the anti-fake-scaling
tests/bench/test_bench_ocean_mpas_scaling.py:197:    # --- M1 lane fields (scaling-M3d increment-1) ---
tests/bench/test_bench_ocean_mpas_scaling.py:202:    # Fused-scan M1 contract: headline + separate dispatch-latency probe.
tests/bench/test_bench_ocean_mpas_scaling.py:206:    assert fused["step_latency_ms"] > 0
tests/bench/test_bench_ocean_mpas_scaling.py:210:    # loop is demoted to explicitly-named latency keys and the legacy
tests/bench/test_bench_ocean_mpas_scaling.py:213:    assert rec["step_latency_gate_loop_ms"] > 0
tests/bench/test_bench_ocean_mpas_scaling.py:214:    assert rec["step_latency_gate_loop_min_ms"] > 0
tests/bench/test_bench_ocean_mpas_scaling.py:245:    # synced gate-loop latency (codex finding 1).
tests/bench/test_bench_ocean_mpas_scaling.py:248:    assert rec["step_latency_gate_loop_ms"] > 0
tests/test_fv3_del6_vt_flux.py:88:def test_del6_vt_flux_damp_scales(cdgrid):
tests/test_fv3_del6_vt_flux.py:108:        # linear in damp (since fx2 scales as damp).  So ratio stays 2.
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py:3:Operates as a SEPARATE diffusion step layered on top of the dycore's
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py:5:constant κ) keep their own κ_v + diffusion solver while the tidal
packages/ocean/legoesm/ocean/coupler/tidal_mixing_apply.py:129:    GRID-AGNOSTIC. The implicit vertical diffusion is a per-column Thomas solve
tests/ice/unit/test_ice_rheology_faithful.py:12:scalar oracle:
tests/diff/test_kuo.py:40:def _scalar_loss(T, qv, pf, ph, pq):
tests/diff/test_kuo.py:60:    loss = lambda x: _scalar_loss(T, qv, pf, ph, x)
tests/diff/test_kuo.py:101:    loss = lambda x: _scalar_loss(x, qv, pf, ph, pq)
tests/diff/test_kuo.py:126:        lambda T_, q_, p_: _scalar_loss(T_, q_, pf, ph, p_),
docs/ocean/fidelity/oceananigans_reproduction_scoreboard.md:96:(blow-up ~day 90) is a SEPARATE eddy-scale instability, not the barotropic 2Δx
docs/ocean/fidelity/oceananigans_reproduction_scoreboard.md:123:solver is NOT the §5 blow-up lever.** The §5 residual is the eddy/wall-scale
docs/ocean/fidelity/oceananigans_reproduction_scoreboard.md:136:CASE 3's §5 wall blow-up are the SAME residual (2Δx grid-scale under-dissipation).
docs/ocean/fidelity/oceananigans_reproduction_scoreboard.md:139:2Δx grid-scale under-dissipation is REAL and SHARED — bickley's enstrophy excess
docs/ocean/fidelity/oceananigans_reproduction_scoreboard.md:178:   blow-up WORSE (day-18 max|u| 8.0→11.1), the unambiguous signature of a grid-scale
docs/ocean/fidelity/oceananigans_reproduction_scoreboard.md:222:   momentum under-dissipating grid-scale enstrophy vs Oceananigans' `WENOVectorInvariant`** —
tests/validation/test_gate_plane_smoke.py:3:Assembles SAM's GATE_IDEAL case (sounding + large-scale forcing + SST) on the
scripts/bench/bench_atm_latlon_spmd_scaling.py:1:"""Strong/weak scaling bench for the lat-band SPMD lat-lon C-grid hydrostatic
scripts/bench/bench_atm_latlon_spmd_scaling.py:9:dispatch-latency probe (``step_latency_ms``) — never mixed.  The
scripts/bench/bench_atm_latlon_spmd_scaling.py:18:band-SHARDED geometry, in-graph finite scalar), so --steps counts BLOCKS of N
scripts/bench/bench_atm_latlon_spmd_scaling.py:25:Receipt honesty: a segment whose in-graph finite scalar reports a non-finite
scripts/bench/bench_atm_latlon_spmd_scaling.py:30:valid scaling data.  The default fused lane has no in-graph finite check, so
scripts/bench/bench_atm_latlon_spmd_scaling.py:40:cubed-sphere ``run_cpu_mpi_scaling --cs-spmd`` A1 path. Every process calls
scripts/bench/bench_atm_latlon_spmd_scaling.py:47:mixed-stack deadlock hazard — see run_cpu_mpi_scaling._build_cubed_sphere_spmd).
scripts/bench/bench_atm_latlon_spmd_scaling.py:53:  srun -n 8 python bench_atm_latlon_spmd_scaling.py --multicontroller \
scripts/bench/bench_atm_latlon_spmd_scaling.py:75:# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
scripts/bench/bench_atm_latlon_spmd_scaling.py:83:    scaling_metadata,
scripts/bench/bench_atm_latlon_spmd_scaling.py:143:                        "SEPARATE dispatch-latency probe (step_latency_ms).")
scripts/bench/bench_atm_latlon_spmd_scaling.py:156:    p.add_argument("--comm-latency-us", type=float, default=None,
scripts/bench/bench_atm_latlon_spmd_scaling.py:157:                   help="MEASURED per-message latency [us] of THIS machine's "
scripts/bench/bench_atm_latlon_spmd_scaling.py:164:    p.add_argument("--out", type=str, default="results/a1/spmd_scaling.jsonl")
scripts/bench/bench_atm_latlon_spmd_scaling.py:261:        # in-graph finite SCALAR, then block on the state for honest timing.
scripts/bench/bench_atm_latlon_spmd_scaling.py:273:                # diverging trajectory serialize as valid scaling data, and
scripts/bench/bench_atm_latlon_spmd_scaling.py:277:                print(f"[warn] segment finite scalar FALSE after block {i} "
scripts/bench/bench_atm_latlon_spmd_scaling.py:295:        # Measurement contract (scaling audit gaps #1/#2): fused ``lax.scan``
scripts/bench/bench_atm_latlon_spmd_scaling.py:297:        # host-synced loop measured dispatch+sync latency, not fused device
scripts/bench/bench_atm_latlon_spmd_scaling.py:298:        # throughput.  Dispatch latency stays measured SEPARATELY
scripts/bench/bench_atm_latlon_spmd_scaling.py:299:        # (``step_latency_ms``); multi-controller runs record the
scripts/bench/bench_atm_latlon_spmd_scaling.py:361:        latency_us=args.comm_latency_us,
scripts/bench/bench_atm_latlon_spmd_scaling.py:375:        # valid=false marks the row as NOT scaling data; completed_blocks
scripts/bench/bench_atm_latlon_spmd_scaling.py:394:        # step_latency_ms, block_ms, parallel_block_ms, rank_imbalance, ...
scripts/bench/bench_atm_latlon_spmd_scaling.py:399:    # aggregate_bcw_scaling.py → empty SYPD panels in the CPU-vs-GPU plots.
scripts/bench/bench_atm_latlon_spmd_scaling.py:421:                   "steady_min_ms", "fused_step_ms", "step_latency_ms",
scripts/bench/bench_atm_latlon_spmd_scaling.py:427:    rec["metadata"] = annotate_incomplete(scaling_metadata(
scripts/bench/bench_atm_latlon_spmd_scaling.py:441:        scaling_kind=args.mode,
scripts/bench/bench_atm_latlon_spmd_scaling.py:468:            f"diverged: segment finite scalar false after block "
scripts/bench/bench_atm_latlon_spmd_scaling.py:470:            "non-finite trajectory, not valid scaling data")
scripts/bench/bench_atm_latlon_spmd_scaling.py:491:                  f"latency={rec['step_latency_ms']}ms/step "
scripts/bench/bench_atm_latlon_spmd_scaling.py:497:                  "hardware scaling — do not report it as a speedup.")
tests/bench/test_bench_voronoi_partition_methods.py:54:    # geometric/sfc are dependency-free; metis truthfully reports.
tests/bench/test_bench_voronoi_partition_methods.py:58:        import pymetis  # noqa: F401
tests/bench/test_bench_voronoi_partition_methods.py:60:        assert mod.method_available("metis") is True
tests/bench/test_bench_voronoi_partition_methods.py:62:        assert mod.method_available("metis") is False
tests/bench/test_bench_voronoi_partition_methods.py:75:    assert payload["auto_resolves_to"] in ("metis", "geometric")
packages/ocean/legoesm/ocean/advection.py:158:    # ONLY (direction-asymmetric diffusion).
packages/ocean/legoesm/ocean/advection.py:275:    # over-diffusive for v<0 ONLY (direction-asymmetric diffusion).
packages/ocean/legoesm/ocean/advection.py:842:    using a Zalesak limiter that adds maximum anti-diffusion without
packages/ocean/legoesm/ocean/advection.py:1311:    scale T noise to leak through under strong-frontal forcing
packages/ocean/legoesm/ocean/advection.py:1525:#   vertical diffusion already use: an independent M-level W-column with
packages/ocean/legoesm/ocean/advection.py:1581:    ``u_cfl = |vel|*dt_tracer/dx`` is the CFL-dependent anti-diffusion weight.
packages/ocean/legoesm/ocean/advection.py:1690:    ``dt_tracer`` enters ONLY the uCFL anti-diffusion weight (Veros uses
packages/ocean/legoesm/ocean/experiments/dino.py:7:horizontal scales. *Geosci. Model Dev.*, 18, 8091-8107.
packages/ocean/legoesm/ocean/experiments/dino.py:138:    # module converts these to timescales internally per layer thickness.
packages/ocean/legoesm/ocean/experiments/dino.py:226:    # Vertical mixing — KPP + enhanced-diffusion convection.
packages/ocean/legoesm/ocean/experiments/dino.py:232:    K_conv: float = 100.0          # enhanced-diffusion convective K [m²/s] (rn_evd)
packages/ocean/legoesm/ocean/experiments/dino.py:233:    # NEMO nn_evdm=1 (DINO namelist): the enhanced vertical diffusion applies
packages/ocean/legoesm/ocean/experiments/dino.py:301:    # Convective adjustment (enhanced vertical diffusion) trigger fidelity.
packages/ocean/legoesm/ocean/experiments/dino.py:325:    # ``EnhancedDiffusionConfig.two_level_trigger`` (elementwise max of the
packages/ocean/legoesm/ocean/experiments/dino.py:366:    # TKE self-diffusion Schmidt coefficient (alpha_tke). Gaspar/Veros use 30;
packages/ocean/legoesm/ocean/experiments/dino.py:369:    tke_alpha: float | None = None              # 1.0 = NEMO en self-diffusion
packages/ocean/legoesm/ocean/experiments/dino.py:432:    # GM/Redi mesoscale eddy parameterization. Adaptive κ via Visbeck 1997
packages/ocean/legoesm/ocean/experiments/dino.py:434:    # nn_aei_ijk_t=21 oracle scaling — supersedes the 2026-05-14 decision
packages/ocean/legoesm/ocean/experiments/dino.py:438:    # Adaptive-κ_GM scaling: "visbeck" (Visbeck 1997, the historical legoESM
packages/ocean/legoesm/ocean/experiments/dino.py:440:    # ACTUAL DINO+ORCA1 oracle scaling; cap aei0 = 0.5·rn_Ue·rn_Le
packages/ocean/legoesm/ocean/experiments/dino.py:454:    # iso-neutral Laplacian with kappa = ½·U_d·Δ(φ) row-scaled, slope cap
packages/ocean/legoesm/ocean/experiments/dino.py:484:    U_M: float = 0.27              # viscous velocity scale [m/s] (rn_Uv)
packages/ocean/legoesm/ocean/experiments/dino.py:491:    A_h_floor: float = 1000.0      # min effective A_h [m²/s] after cos(lat) scaling
packages/ocean/legoesm/ocean/experiments/dino.py:494:    # Tracer iso-neutral diffusion velocity scale (R1 only; eq below
packages/ocean/legoesm/ocean/experiments/dino.py:496:    # reference but do not apply a separate harmonic tracer diffusion.
packages/ocean/legoesm/ocean/experiments/dino.py:497:    U_T: float = 0.027             # tracer diffusivity velocity scale [m/s] (rn_Ut)
packages/ocean/legoesm/ocean/experiments/dino.py:607:    # LatLonCGridOceanConfig.barotropic.barotropic_diffusion_alpha (#1226),
packages/ocean/legoesm/ocean/experiments/dino.py:608:    # threaded 1:1 via from_flat/BarotropicConfig. NEMO has no eta-diffusion
packages/ocean/legoesm/ocean/experiments/dino.py:612:    barotropic_diffusion_alpha: float = 0.01
packages/ocean/legoesm/ocean/experiments/dino.py:625:    # applies a single A_h·cos(φ) scalar OUTSIDE the vector Laplacian;
packages/ocean/legoesm/ocean/experiments/dino.py:685:    # barotropic blowup (dino_l2_bisect o_ctl: basin-scale off-equatorial eta
packages/ocean/legoesm/ocean/experiments/dino.py:729:      ln_traldf_iso Redi-only Laplacian (kappa = ½·U_d·Δ(φ) row-scaled,
packages/ocean/legoesm/ocean/experiments/dino.py:747:    lateral diffusion, and the MLF leapfrog integrator.
packages/ocean/legoesm/ocean/experiments/dino.py:894:        "tke_alpha": 1.0,                        # en self-diffusion 0.5·(avm+avm) (zdftke:406)
packages/ocean/legoesm/ocean/experiments/dino.py:949:        # -- Tracer lateral diffusion (namtra_ldf: ln_traldf_iso + ln_traldf_msc,
packages/ocean/legoesm/ocean/experiments/dino.py:954:        #    has NO separate geopotential background diffusion: its ONLY lateral
packages/ocean/legoesm/ocean/experiments/dino.py:959:        #    smoothness deficit (SSH small-scale 0.41x @y1). --
packages/ocean/legoesm/ocean/experiments/dino.py:962:        # slope discretizations (PSD violated -> local antidiffusion). Fixed by
packages/ocean/legoesm/ocean/experiments/dino.py:1014:        # masked by the default 0.01 eta-diffusion damping — it corrupts
packages/ocean/legoesm/ocean/experiments/dino.py:1015:        # oracle tendency comparisons). dynspg_ts.F90 has no eta-diffusion
packages/ocean/legoesm/ocean/experiments/dino.py:1017:        "barotropic_diffusion_alpha": 0.0,
packages/ocean/legoesm/ocean/experiments/dino.py:1036:        # scale factors into ffu/ffv EXACTLY (dynspg_ts.F90:1349-1379) — the
packages/ocean/legoesm/ocean/experiments/dino.py:1281:    `depth_bot` may be a scalar or an array (e.g., the underlying
packages/ocean/legoesm/ocean/experiments/dino.py:1336:    # Length scales (degrees) — note the Mercator correction on the
packages/ocean/legoesm/ocean/experiments/dino.py:1337:    # meridional scale matches the Zenodo code, NOT the paper text.
packages/ocean/legoesm/ocean/experiments/dino.py:1586:# (timescale instead of heat-flux coefficient; no Q_sr split).
packages/ocean/legoesm/ocean/experiments/dino.py:1765:        the NEMO-fidelity caller passes these three scalars (computed once
packages/ocean/legoesm/ocean/experiments/dino.py:2033:# free in legoESM: setting LatLonCGridOceanConfig.A_h_lat_scaling=True
packages/ocean/legoesm/ocean/experiments/dino.py:2261:    depth-dependent Bryan-Lewis background switched OFF (``bg_diff_scale=0``) to
packages/ocean/legoesm/ocean/experiments/dino.py:2263:    Convective adjustment itself is handled by the enhanced-diffusion convection
packages/ocean/legoesm/ocean/experiments/dino.py:2291:            bg_diff_scale=0.0,
packages/ocean/legoesm/ocean/experiments/dino.py:2326:            # NEMO en self-diffusion uses 0.5·(avm[k+1]+avm[k]) (alpha_tke=1),
packages/ocean/legoesm/ocean/experiments/dino.py:2412:    A_h is a SCALAR; the model multiplies it by ``cos(lat)`` per row
packages/ocean/legoesm/ocean/experiments/dino.py:2413:    when ``A_h_lat_scaling=True``. With the Mercator grid this exactly
packages/ocean/legoesm/ocean/experiments/dino.py:2464:        # here, row-scaled by cos φ via kappa_redi_lat_scaling (the same
packages/ocean/legoesm/ocean/experiments/dino.py:2465:        # Mercator scaling A_h uses).  EIV stays a SEPARATE switch
packages/ocean/legoesm/ocean/experiments/dino.py:2476:        # via kappa_redi_lat_scaling.  With use_gm_redi=True this is the
packages/ocean/legoesm/ocean/experiments/dino.py:2486:            kappa_redi_lat_scaling=True,
packages/ocean/legoesm/ocean/experiments/dino.py:2500:            # kappa-scaled local tracer runaway (stable at kappa=200, T>38C
packages/ocean/legoesm/ocean/experiments/dino.py:2545:            # "treguier" (the NEMO nn_aei_ijk_t=21 oracle scaling, cap
packages/ocean/legoesm/ocean/experiments/dino.py:2566:            EnhancedDiffusionConfig, OceanConvectionConfig,
packages/ocean/legoesm/ocean/experiments/dino.py:2583:                scheme="enhanced_diffusion",
packages/ocean/legoesm/ocean/experiments/dino.py:2590:                enhanced_diffusion=EnhancedDiffusionConfig(
packages/ocean/legoesm/ocean/experiments/dino.py:2673:        A_h_lat_scaling=True,         # cos(lat) per-row scaling — Phase 1B
packages/ocean/legoesm/ocean/experiments/dino.py:2675:        # div/curl (ignores A_h_lat_scaling / eq-boost / floor, which stay OFF on
packages/ocean/legoesm/ocean/experiments/dino.py:2677:        # scalar path byte-identical for every other recipe.
packages/ocean/legoesm/ocean/experiments/dino.py:2680:             else K_h_base),   # iso-neutral replaces iso-level diffusion
packages/ocean/legoesm/ocean/experiments/dino.py:2696:        barotropic_diffusion_alpha=cfg.barotropic_diffusion_alpha,
packages/ocean/legoesm/ocean/experiments/dino.py:2795:        Used to derive a representative cell-resolution scalar for
packages/ocean/legoesm/ocean/experiments/dino.py:2800:        approximation is within ~10% of the proper per-cell scaling.
packages/ocean/legoesm/ocean/experiments/dino.py:2803:        If True, include KPP + GM/Redi + enhanced-diffusion convection
packages/ocean/legoesm/ocean/experiments/dino.py:2878:        EnhancedDiffusionConfig, OceanConvectionConfig,
packages/ocean/legoesm/ocean/experiments/dino.py:2915:            scheme="enhanced_diffusion",
packages/ocean/legoesm/ocean/experiments/dino.py:2916:            enhanced_diffusion=EnhancedDiffusionConfig(
packages/ocean/legoesm/ocean/experiments/dino.py:2942:# Convenience: surface-layer restoring timescales derived from heat-flux
packages/ocean/legoesm/ocean/experiments/dino.py:2944:def restoring_timescale_T_days(cfg: DINOConfig) -> float:
packages/ocean/legoesm/ocean/experiments/dino.py:2950:def restoring_timescale_S_days(cfg: DINOConfig) -> float:

exec
/usr/bin/bash -lc "rg --files docs results scripts packages | rg 'scal|performance|spectral|mpas|barotropic|halo|shard' | head -300" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 348ms:
scripts/bench/bench_barotropic_mcut.py
scripts/bench/bench_mpas_spmd_scaling.py
scripts/bench/probe_spectral_shard.py
scripts/bench/bench_atm_latlon_spmd_scaling.py
scripts/bench/analyze_gpu_scaling.py
scripts/bench/aggregate_cube_shardmap_scaling.py
scripts/bench/bench_dd_scaling.py
scripts/bench/bench_crm_gpu_scaling.py
scripts/bench/bench_ocean_mpas_scaling.py
scripts/bench/aggregate_bcw_scaling.py
scripts/bench/aggregate_scaling_results.py
scripts/bench/bench_halo_exchange.py
scripts/bench/bench_latlon_2d_halo.py
scripts/bench/bench_halo_ops_scaling.py
scripts/bench/bench_cube_tiled_step_scaling.py
scripts/bench/validate_scaling.sh
scripts/bench/run_levante_gpu_scaling.py
scripts/bench/profile_mpas_ocean.py
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
scripts/bench/bench_cube_shardmap_halo.py
scripts/bench/analyze_scaling_results.py
scripts/bench/scaling_summary.py
scripts/bench/bench_barotropic_precond_convergence.py
scripts/bench/bench_coupled_latlon_scaling.py
scripts/bench/slurm_scaling_diagnosis.sh
scripts/bench/bench_spectral_transform_micro.py
scripts/bench/bench_ocean_latlon_spmd_scaling.py
scripts/bench/bench_mpi_scaling.py
scripts/validate/validate_tpu_emulation_sharding.py
scripts/validate/validate_bl_new_vs_spectral.py
docs/ocean/fidelity/internal_tide_barotropic_coriolis_fix_plan.md
scripts/experiment/dino/prev_mpas_90d.yaml
scripts/experiment/dino/matched_mpas_90d.yaml
scripts/data/generate_mpas_nmc.py
scripts/validate/validate_bomex_new_vs_spectral.py
packages/ocean/legoesm/ocean/dynamics/spectral_ocean_pe.py
packages/ocean/legoesm/ocean/dynamics/barotropic.py
packages/ocean/legoesm/ocean/dynamics/ocean_pe_mpas.py
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py
packages/ocean/legoesm/ocean/dynamics/barotropic_mpas.py
packages/ocean/legoesm/ocean/dynamics/mpas_fill.py
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py
packages/ocean/legoesm/ocean/dynamics/advection_mpas.py
packages/ocean/legoesm/ocean/dynamics/barotropic_cgrid.py
packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py
packages/ocean/legoesm/ocean/dynamics/mpas_partial_cell_helpers.py
packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py
scripts/validate/eval_barotropic_noise_invariants.py
scripts/validate/compare_scaling_runs.py
packages/coupler/legoesm/coupler/mpas_adapter.py
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py
packages/ocean/legoesm/ocean/mpas_config.py
packages/ocean/legoesm/ocean/init_mpas.py
scripts/data/generate_mitgcm_barotropic_gyre_reference.py
scripts/validate/fv3_native/vertex_gridscale.py
scripts/data/generate_oceananigans_barotropic_gyre_reference.jl
scripts/data/fit_mpas_gen_be.py
packages/core/legoesm/parallel/async_halo.py
packages/core/legoesm/parallel/column_shard.py
packages/core/legoesm/parallel/shard_map_compat.py
packages/core/legoesm/parallel/sharded_dynamics.py
packages/core/legoesm/parallel/scaling_diagnostics.py
packages/core/legoesm/parallel/halo_exchange_voronoi.py
packages/core/legoesm/parallel/halo_exchange.py
docs/ocean/experiments/mle_mpas_port_plan.md
scripts/plot/plot_scaling_dashboard.py
scripts/plot/regen_mpas_snapshots.py
scripts/plot/plot_scaling_paper_figure.py
docs/ocean/experiments/mpas_bci_damping_investigation.md
docs/ocean/experiments/mpas_vs_latlon_comparison_plan.md
scripts/plot/plot_ec_site_performance.py
scripts/plot/plot_barotropic_noise_comparison.py
scripts/plot/plot_mpas_omip_snapshot.py
scripts/plot/plot_barotropic_mcut.py
scripts/plot/plot_gpu_scaling.py
scripts/plot/plot_barotropic_mcut_weakscale.py
scripts/plot/plot_mpas_single_point_t500.py
scripts/plot/plot_scaling_laws.py
scripts/plot/plot_amip_mpas_checkpoint.py
scripts/plot/plot_scaling_vs_clima.py
docs/ocean/experiments/gm_redi_mpas_plan.md
scripts/plot/plot_cpu_gpu_scaling_summary.py
docs/ocean/experiments/mpas_ico5_production_tuning_plan.md
scripts/plot/regen_mpas_cross_sections.py
scripts/plot/plot_cpu_vs_gpu_scaling.py
scripts/plot/plot_scaling_indicators.py
docs/ocean/experiments/density_jacobian_pgf_mpas.md
scripts/plot/plot_levante_gpu_scaling_comparison.py
docs/ocean/experiments/distributed_barotropic_pcg.md
docs/ocean/experiments/barotropic_gyre_design.md
docs/ocean/experiments/realistic_geometry_mpas_plan.md
docs/ocean/experiments/mpas_omip_jra55_plan.md
docs/ocean/experiments/mpas_etopo_progression_findings.md
docs/ocean/experiments/mpas_performance_plan.md
scripts/cluster/scaling_ginsburg/_env.sh
scripts/plot/plot_scaling_family.py
scripts/cluster/scaling_ginsburg/submit.sh
scripts/cluster/scaling_ginsburg/aimip_amip_finetune.sbatch
scripts/cluster/scaling_ginsburg/aimip_fleet_plot.sbatch
scripts/cluster/scaling_ginsburg/aimip_latlon_sfno_smoke.sbatch
scripts/plot/plot_scaling_efficiency.py
scripts/cluster/scaling_ginsburg/aimip_ace2loss.sbatch
scripts/cluster/scaling_ginsburg/aimip_amip_inference.sbatch
scripts/validate/validate_clm_ml_coupled_scale.py
scripts/plot/plot_iter_scalings.py
scripts/plot/plot_atm_latlon_spmd_scaling.py
scripts/plot/plot_strong_scaling_by_resolution.py
scripts/plot/plot_scaling.py
scripts/plot/plot_spectral_level_shard.py
scripts/plot/plot_bcw_scaling.py
scripts/cluster/scaling_levante/README.md
scripts/cluster/scaling_levante/_env.sh
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch
scripts/cluster/scaling_levante/cube_tiled_step.sbatch
scripts/cluster/scaling_levante/diagnosis.sbatch
scripts/cluster/scaling_levante/gpu_scaling.sbatch
scripts/cluster/scaling_levante/cpu_scaling.sbatch
scripts/cluster/scaling_levante/gpu_moist_scaling.slurm
packages/coupler/legoesm/driver/sharded_operator_split_step.py
packages/ocean/legoesm/ocean/fidelity/mitgcm_barotropic_gyre_recipe.py
packages/core/legoesm/grids/halo.py
packages/core/legoesm/grids/fv3_native_halos.py
packages/core/legoesm/grids/dgrid_halo.py
packages/core/legoesm/grids/halo_latlon.py
packages/core/legoesm/core/spectral_plane_ops.py
packages/ocean/legoesm/ocean/simple_ocean_mpas.py
packages/ocean/legoesm/ocean/conservation_mpas.py
scripts/validate/ocean_fidelity/compare_oceananigans_barotropic_gyre.py
scripts/cluster/scaling_derecho/scaling_cpu.sh
scripts/cluster/scaling_derecho/cube_scaling_cpu_routeb.sh
scripts/cluster/scaling_derecho/cube_scaling_gpu.sh
scripts/cluster/scaling_derecho/blocker1_cube_shardmap_xnode.pbs
packages/ocean/legoesm/ocean/experiments/global_barotropic_wind.py
scripts/cluster/scaling_derecho/finalize_scaling.sh
scripts/cluster/scaling_derecho/README.md
scripts/cluster/scaling_derecho/_env.sh
scripts/cluster/scaling_derecho/mc_nccl_probe.py
scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs
scripts/cluster/scaling_derecho/cube_scaling_cpu.sh
scripts/cluster/scaling_derecho/ocean_cpu_scaling.pbs
scripts/cluster/scaling_derecho/submit_scaling.sh
packages/ocean/legoesm/ocean/experiments/barotropic_wave.py
scripts/cluster/scaling_derecho/diagnosis.pbs
scripts/cluster/scaling_derecho/mc_nccl_canary.sh
scripts/cluster/scaling_derecho/blocker1_cube_shardmap.pbs
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs
scripts/cluster/scaling_derecho/cube_tiled_step.pbs
scripts/cluster/scaling_derecho/scaling_gpu.sh
scripts/cluster/scaling_derecho/build_nccl_ofi.sh
packages/ocean/legoesm/ocean/physics/mpas_physics.py
scripts/cluster/wb_forecast/train_sfno_full_scale.sbatch
scripts/cluster/wb_forecast/train_sfno_full_scale_ginsburg.sbatch
packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_mpas.py
packages/ocean/legoesm/ocean/physics/lateral_mixing/mle_mpas.py
scripts/cluster/derecho/train_sfno_full_scale.pbs
scripts/cluster/aimip_scale/README.md
scripts/cluster/aimip_scale/train_aimip_derecho.pbs
scripts/cluster/aimip_scale/train_aimip_levante.slurm
docs/superpowers/plans/2026-07-05-weatherbench-scale-training.md
packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py
packages/ml/legoesm/ml/spectral_conv.py
scripts/cluster/levante/amip_mpas_gpu_chain.sbatch
scripts/cluster/omip_nemo/run_mpas14_tke.sbatch
docs/superpowers/specs/2026-07-05-weatherbench-scale-training-design.md
scripts/run/run_spectral_cbl.py
scripts/run/run_ocean_spectral_tests.py
scripts/run/run_spectral_sbl.py
scripts/cluster/omip_nemo/rerun_mpas7_kpp_r2.sbatch
scripts/cluster/omip_nemo/run_mpas9_parity.sbatch
scripts/run/mpas_4dvar_single/__init__.py
scripts/run/mpas_4dvar_single/run_tlm.py
scripts/run/mpas_4dvar_single/common.py
scripts/run/mpas_4dvar_single/run_assimilation.py
scripts/run/run_dry_held_suarez_mpas.py
scripts/run/mpas_realistic_geometry/run_mpas_seamount_rest.py
scripts/run/mpas_realistic_geometry/README.md
scripts/run/mpas_realistic_geometry/diagnose_vertex_thickness_hybrid.py
scripts/run/mpas_realistic_geometry/run_mpas_etopo_spinup.py
scripts/run/train_neural_gcm_spectral.py
scripts/run/run_w2_mpas_convergence.py
scripts/run/global_overturning/run_global_overturning_mpas_50yr_implicit.py
scripts/run/run_spectral_les.py
scripts/run/run_dry_held_suarez_spectral.py
scripts/run/train_weatherbench_scale.py
scripts/run/global_overturning/run_comparison_mpas.py
scripts/run/global_overturning/compare_moc_and_w_mpas_vs_latlon.py
scripts/run/global_overturning/plot_mpas_50yr_implicit_all.py
scripts/run/global_overturning/run_global_overturning_mpas_etopo.py
scripts/run/global_overturning/run_global_overturning_mpas_baseline.py
scripts/run/global_overturning/diagnose_mpas_baro_noise.py
scripts/run/global_overturning/plot_mpas_baseline_snapshot.py
results/dd_scaling.txt
results/dd_scaling.png
docs/wb/scale_training_runbook.md
results/a1/cube_tiled_scaling_levante_j26453524_parity.jsonl
results/a1/cube_tiled_scaling_levante_j26453645_parity.jsonl
results/a1/cube_tiled_scaling_levante_j26453645_loop_parity.jsonl
results/a1/cube_tiled_scaling_levante_j26453524_loop_parity.jsonl
results/a1/cube_tiled_scaling_levante_j26453645.jsonl
results/a1/cube_tiled_scaling_levante_j26453524.jsonl
packages/ml/legoesm/training/scale_build.py
packages/ml/legoesm/training/neural_gcm_spectral.py
docs/physics-notes/pseudo_spectral_advection_smoothness.md
docs/physics-notes/moist_les_spectral.md
docs/dev-notes/mpas_seed_ps_reduction_nan_2026-07-23.md
docs/dev-notes/issues/mpas_run_loop_stateful_physics_carry.md
docs/dev-notes/issues/barotropic_mode_noise.md
docs/dev-notes/issues/clubb_parcel_lscale_vs_shared_mixing_length.md
docs/dev-notes/research/barotropic_noise_handling_in_production_models.md
docs/performance/multinode_gpu_direct_cxi.md
docs/performance/issue_852_cube_shardmap_rootcause.md
docs/performance/REAL_HARDWARE_SCALING.md
docs/performance/scream_parity_scope.md
docs/performance/scaling/scaling_theoretical_limit_report_2026-06-15.md
docs/performance/scaling/SCALING_STATUS_AUDIT.md
docs/performance/scaling/bcw_scaling_status.md
docs/performance/scaling/scaling_review_2026-06-13.md
docs/performance/scaling/barotropic_mcut.csv
docs/performance/scaling/crm_les_scaling.md
docs/performance/scaling/barotropic_mcut_np4.csv
docs/performance/scaling/c3_cube_face_scatter_driver_plan.md
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md
docs/performance/scaling/RESUME_STATE_2026-06-13.md
docs/performance/scaling/distance_to_limit_2026-06-13.md
docs/performance/scaling/barotropic_mcut_np2.csv
docs/performance/scaling/crm_gpu_l2_tiling.md
docs/performance/scaling/cube_transport_tiling_design.md
docs/performance/scaling/derecho_levante_sota_review_2026-07.md
docs/performance/scaling/scaling_indicators.csv
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
docs/performance/scaling/voronoi_batched_ab.png
docs/performance/scaling/spectral_gpu_feasibility.md
packages/atmosphere/legoesm/atmosphere/dynamics/les/spectral_plane.py
docs/performance/scaling/literature_parallelization_2026-06.md
docs/performance/scaling/cube_production_tiling_design.md
docs/performance/scaling/barotropic_multinode_verdict_2026-06-15.md
docs/performance/scaling/scaling_levers_audit_2026-06-15.md
packages/atmosphere/legoesm/atmosphere/dynamics/les/spectral_les_plane.py
packages/atmosphere/legoesm/atmosphere/dynamics/les/compressible_euler_plane_halo.py
packages/atmosphere/legoesm/atmosphere/dynamics/les/plane_operators_halo.py
packages/atmosphere/legoesm/atmosphere/dynamics/les/spectral_les_moist.py
packages/atmosphere/legoesm/atmosphere/forcing/column_large_scale_extract.py
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_mpas.py
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_nh.py
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_mpas.py
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_pe.py
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/compressible_euler_mpas.py
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/spectral_sw.py
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_spectral.py
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tracer_transport_mpas.py
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py
docs/scaling/atm_latlon_spmd_scaling.md
docs/scaling/external_scaling_transfer_assessment.md
packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/prognostic_spectral.py
packages/atmosphere/legoesm/atmosphere/forcing/plane_large_scale_forcing.py
packages/atmosphere/legoesm/atmosphere/physics/turbulence/override_sharding.py
packages/atmosphere/legoesm/atmosphere/forcing/idealized/large_scale_forcing.py

exec
/usr/bin/bash -lc "sed -n '1,260p' docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md && sed -n '1,260p' docs/performance/scaling/levante_campaign_2026-07-24.md && sed -n '1,260p' docs/performance/scaling/spectral_gpu_feasibility.md && sed -n '1,260p' docs/performance/scaling/mpas_ocean_distributed_stage_audit.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 346ms:
# Weak/strong scaling bottleneck diagnosis — atm + ocean, MPI + GPU (2026-07-20)

Scope: consolidate the measured weak/strong scaling bottlenecks for atmosphere
and ocean on both transports (route-A mpi4jax, route-B NCCL/SPMD), pin the
LATENCY-bound signal with fresh numbers, and close the two instrument gaps that
kept that signal off the actual Derecho/Levante runs. Read the campaign context
first: `derecho_levante_sota_review_2026-07.md` (measured baselines + SOTA),
`spmd_message_census_2026-07-08.md` (the message-count analysis this extends),
`SCALING_STATUS_AUDIT.md` (support matrix).

Every claim below is labelled **CONFIRMED** (measured / read from code) or
**PLAUSIBLE** (inferred, needs a machine receipt).

---

## 1. The bottleneck table (what limits each axis, and why)

| Component × axis | Limiter | Evidence | Tier |
|---|---|---|---|
| Atm cube strong (≤6 GPU) | **collective-permute COUNT** grows with shard count while per-msg NCCL p2p latency (~30–80 µs) doesn't amortise on small tiles → anti-scales | census: 12 CP @2dev, 41 @6dev (C24/L8, below); f64≡f32 curves ⇒ latency-bound | CONFIRMED |
| Atm latlon strong | 4→8 GPU node-crossing plateau on route-A (mpi4jax sendrecv OPAQUE to XLA latency-hiding scheduler ⇒ no comm/compute overlap) | Derecho 78 km throughput 600→480→recover@16; route-B multicontroller SHIPPED to remove it | CONFIRMED |
| Atm ico strong | best-scaling grid (low perimeter/area cell partition); multihost SPMD still open | Derecho 28 km ico eff ~0.38 @16 A100, still rising | CONFIRMED |
| Atm/all coarse | per-device saturation floor (<~30k cols/GPU flat/anti) — NOT a defect | 111 km latlon/ico FLAT 1→16 A100 | CONFIRMED |
| Ocean strong (CPU-MPI) | **implicit-CN PCG reduction wall**: default `pcg_variant="standard"` = 2 *sequentially-dependent* all-reduces/iter × `fixed_iters=60` ⇒ ~120 latency-serialized all-reduces/step | `barotropic_common.py` `_fixed_iteration_pcg` (p·Ap @:619 → dependent r·z @:626); np16→32 eff 0.55 | CONFIRMED |
| Ocean strong (mitigation) | `single_reduce` (Chronopoulos–Gear) → 1 all-reduce/iter; split-explicit + `barotropic_local_subcycle_clamp` → 3 all-reduce/step (reduction-free subcycle) | both exist, both NON-default; beats standard at ≥2 nodes 1.15–1.65× | CONFIRMED |
| Ocean weak | ~1.0 eff (as SOTA); the strong ceiling is naive land imbalance, not comm | 0.97 weak @590k cells/rank | CONFIRMED |
| Spectral | single-device by design (both multi-device schemes measured anti-scaling) | prior campaign | CONFIRMED |

Two structural hot-loop items found in the code map (both PLAUSIBLE as
scaling costs at high step counts, neither a hot-loop collective):
- **Per-step host dispatch** in the operator-split SPMD driver
  (`model_driver.py:7783` Python `for` over `seg_steps`) — a compiled
  `lax.scan`-per-segment path exists (M3b) but the operator-split loop dispatches
  per step; host-dispatch overhead scales with step count.
- **Blocking all-gather + full replicated allocation once per sim-day**
  (`model_driver.py:7798`) for the NaN-guard/callback — not per-step, but a
  cross-process barrier that materialises the global state each segment.

## 2. The LATENCY-bound signal, pinned (fresh, this session)

The 2026-07-08 census established: f64 and f32 GPU strong-scaling curves
coincide ⇒ the multi-GPU leg is **latency-bound, not bandwidth-bound**, so the
lever is the per-step *message count*, not the byte volume. The count is a
STATIC compile property — a CPU virtual-device compile yields the same count the
GPU executes.

Fresh census (sharded cube full PE step, `run_scaling_diagnosis.py --mode
census`, C24/L8, CPU virtual devices):

| devices | collective-permute / step | all-reduce / step |
|---|---|---|
| 2 | 12 | 1 |
| 6 | 41 | 1 |

Readings (CONFIRMED): the single all-reduce is the conservation fixer (fine).
The permute count scales ~linearly with the shard count (edge-coloring: each
face has 4 neighbours ⇒ 4 ppermute rounds × the exchange points) — this IS the
cube anti-scaling. (The absolute counts are lower than the 15/46 in the
2026-07-08 note — halo packing improved since — which is exactly why the count
now belongs on every row as a REGRESSION-tracked metric, not a scratch probe.)

## 3. Instrument gaps closed this session

The diagnosis was well-characterised in the docs, but two gaps kept the signal
off the real machine runs:

1. **The message census counted only collective-permutes.** The ocean PCG
   all-reduce wall — the #1 ocean strong-scaling limiter — was invisible to a
   permute-only census. Fixed: canonical `metadata.count_collectives()` /
   `hlo_collective_census()` count **all** families (permute + all-reduce +
   all-gather + all-to-all + reduce-scatter) with the same op-call-form
   discipline (config-header flag echoes never inflate; async `-start` counted
   once). The permute family stays bit-identical to `count_collective_permutes`.
2. **The bottleneck tool ran on NO Derecho/Levante job.** The cluster campaign
   invoked only the throughput benches (SYPD + permute census); no per-phase
   halo/reduction/roofline/overlap breakdown was ever captured on the target
   machines. Fixed:
   - `run_scaling_diagnosis.py` gained a `census` mode + a census phase in
     `full`/`quick` (recorded in `summary.json`) — the count is now first-class.
   - `scripts/cluster/scaling_derecho/diagnosis.pbs` +
     `scripts/cluster/scaling_levante/diagnosis.sbatch` run the tool across the
     cube face-shard `1 2 3` ladder (must divide 6; >6 uses the tiled bench), so
     the message-count-vs-shard curve AND the full per-phase diagnosis land on
     the real hardware.
   - The SPMD throughput benches the multinode jobs already run
     (`bench_cube_tiled_step_scaling`, `bench_mpas_spmd_scaling`) now record the
     full `hlo_collectives` dict alongside the legacy scalar.

## 4. What to run on Derecho / Levante now

1. **Diagnosis lane** (new): `qsub scripts/cluster/scaling_derecho/diagnosis.pbs`
   / `sbatch scripts/cluster/scaling_levante/diagnosis.sbatch` at production
   size (C96/L40, ≥30k cols/GPU). Yields the census ladder + per-phase halo BW,
   reduction latency, roofline, overlap — the WHERE, not just the SYPD.
2. **Ocean reduction-wall A/B** (the highest-value code lever, already
   selectable): re-run `bench_ocean_mpi_scaling.py` / the SPMD ocean bench with
   `--pcg-variant single_reduce` and with split-explicit +
   `barotropic_local_subcycle_clamp` at ≥16 ranks — the calculus flips toward
   the reduction-free path where per-message latency dominates.
3. **Lane-T GPU-runtime A/B** (already wired, `RUN_TUNE=1`): PGLE was the
   measured winner (+8.5 %); the XLA collective-permute-combine + pipelined-p2p
   arms still need SPLITTING to attribute (the combined arm hurt −10 %).
4. Per §1, cube >6 GPU needs the sub-face tiled production step (separate
   project); the count floor is otherwise reached in code.

## 5. Verification

- `count_collectives` / `hlo_collective_census`:
  `tests/bench/test_scaling_metadata.py` (all-family synthetic HLO + error-safe
  probe).
- `run_scaling_diagnosis.py --mode census` verified locally (2/6 virtual
  devices, table §2); the census phase records into `summary.json`.
- cube/mpas bench census gates green (`test_bench_cube_tiled_step_scaling`,
  `test_bench_mpas_spmd_gates`).
- `diagnosis.pbs` / `.sbatch`: `bash -n` clean; no hardware receipt yet
  (submit-ready, PLAUSIBLE until a real run lands).
# Levante weak/strong scaling campaign — 2026-07-24 (first hardware receipts + fixes)

One-day campaign turning the authored-but-never-run Levante job set
(`scripts/cluster/scaling_levante/`) into measured curves for every grid,
fixing what broke, and moving the worst axis (ocean strong scaling) to a
measured 2× improvement. All receipts on the post-merge tree `d3ec1ccce`+
(campaign branch `worktree-scaling-campaign`); job IDs cited throughout are
Levante SLURM jobs from 2026-07-24. Codex adversarial review: 4 rounds
(transcripts under `.physics-validator/scaling_campaign/`); every
measurement claim below carries the round-3 corrections.

Machines: Levante `gpu` partition (4× A100-80 SXM NVLink/node, IB HDR200),
`compute` (2× AMD Milan 7763). All GPU multinode = route-B
(`jax.distributed` + NCCL over IB verbs — `NET/IB mlx5` confirmed in-log;
route-A CUDA-aware mpi4jax not exercised on Levante).

## Headline results (strong scaling, f32 unless noted)

| Axis | Ladder | Result | Job(s) |
|---|---|---|---|
| Atm lat-lon LL720×1440 L26 | 4→8→16 A100 (1→4 nodes) | 7.72→5.40→3.54 ms/step, monotone; np16 = 7.6 GC/s (477 Mc/s/GPU sustained) | 26450848/26453240/26449147 |
| Atm MPAS ico L8 (28 km) L26 | 6→16 A100 | 8.66→7.08 ms/step; np16 = 2.41 GC/s — 1.6x the Derecho 16-A100 aggregate reported in `derecho_levante_sota_review_2026-07.md` SS3b (route-A, eff ~0.38 @16); CROSS-MACHINE, different stack/date - indicative, not a controlled A/B | 26453240/26449147 |
| **Atm cube C768/L60 (same-path cs-spmd)** | 6→24 A100 | 58.35→14.09 ms/step = **4.14× = eff 1.04 (at ideal)**, 15.1 GC/s (629 Mc/s/GPU) | 26453782 |
| Atm cube C384/L60 (same-path cs-spmd) | 6→24 A100 | 15.44→8.81 ms/step = 1.75× (eff 0.44), 6.0 GC/s | 26452894 |
| Atm cube C192/L60 (same-path) | 6→24 | 6.20→6.80 ms — ANTI-scales (eff 0.23): 9.2k cols/GPU is below the ~30k-column floor | 26452979 |
| Ocean lat-lon LL576×1152 L20, production implicit | 4→8→16 | 15.6→17.2→17.4 ms — anti-scales across nodes | 26452743-45 |
| Ocean same, improved (wide-halo + vmix-f32) | 4→8→16 | 12.8→11.1→8.6 ms — monotone, **2.01× at 16 GPUs** | 26452804-06, 26453279 |
| Cube tiled >6-GPU lane (its own bench) | 24 A100, C384/L60 closed loop | 9.01 ms/step, 5.9 GC/s, 18.2 SYPD — first >6-GPU production-lane receipts | 26450938/26452632 |

Single-node GPU (job 26445836): cube C192 gray_sbm strong eff 0.84–1.07
(1→3 A100); C48/C96 latency-floored. Ocean single-node solver ladder: below.

**THE cube strong-scaling result** — read 1.04 as "at ideal", NOT "better
than ideal": efficiency slightly above 1 is expected when the BASE leg is
per-device disadvantaged (np6 holds 4x the working set per GPU of np24, so
part of the 4.14x is cache/occupancy recovery rather than parallel
efficiency). The claim is that comm does not degrade this ladder, not that
parallelism is free. (same code path, harness, config, IC;
only the tile size varies — jobs 26452979/26452894/26453782): 6→24 GPU
speedup 0.91× / 1.75× / 4.14× at 9.2k / 36.9k / 147k columns per GPU. The
"poor cube strong scaling" of the earlier receipts TRACKS TILE SIZE:
holding code path, harness, config and IC fixed and varying only the tile,
efficiency goes 0.23 -> 0.44 -> 1.04, so tile size is SUFFICIENT to recover
ideal scaling at 24 A100 across 6 nodes. (That shows comm does not degrade
the ladder at production tiles; it does not prove comm costs nothing at
small tiles — that needs a per-phase profile.) Production rule confirmed:
keep >~30k columns/GPU.

Tiled-lane size sweep at fixed 24 GPUs (own bench, closed loop, CFL-scaled
dt; jobs 26450938/26452632/26453645): 1.87 / 5.89 / 14.34 GCells/s at
C192/C384/C768 = 78 / 246 / 597 Mc/s per GPU — per-device throughput still
climbing at 8.85M cells/GPU, so an A100 is not saturated even there.

## Ocean strong-scaling: bottleneck → fix (the campaign's improvement arc)

Dose-response on the implicit-CN barotropic reduction count
(LL384×768 L20 f64, solver-matched via `--force-pcg`, 4×A100 NVLink,
jobs 26447957/26449622):

| arm | reductions/step | nd1/nd2/nd4 ms | eff4 |
|---|---|---|---|
| implicit fixed-PCG standard | 123 | 29.3/21.1/15.0 | 0.49 |
| implicit PCG single_reduce | 63 | 29.6/19.8/14.0 | 0.53 |
| explicit + wide-halo | 0 (solver) | 33.4/19.1/11.5 | 0.73 |

Monotone count→efficiency mapping at every size (LL192/384/768); at LL768
all arms converge to 0.71–0.77 (tiles amortize latency). Fused-halo == plain implicit (a NULL result: pad aggregation does not
move this step, consistent with reductions being the larger cost - the
arms differ in solver internals too, so this is consistency, not proof). NCCL_PROTO
default ≡ LL, LL128 harmful (job 26449812) — protocol lever closed.
`LEGOESM_VMIX_F32_SOLVE=1` (f32 tridiagonal vmix inside the f64 step):
icn +9.1%, wide +11.7% at nd4, conservation-gated at `--cons-rtol 1e-5`
on every arm (job 26452547). Combined best config (wide + vmix-f32):
10.32 ms at LL384 nd4 = 1.45× the production config, and the multinode
2.01× above. CLAIM SCOPE (codex round-3): the count→time slope is an
*effective time per eliminated reduction-batch in these executables*
(~14–18 µs/batch), NOT a measured allreduce latency — `single_reduce`
changes solver work/fusion too; a dependency-matched microbenchmark would
be needed for a latency claim. Both winning options are existing config
selections (`barotropic_solver="explicit_substep"` + wide-halo flags,
`LEGOESM_VMIX_F32_SOLVE`); production defaults unchanged — promotion needs
the wide-halo stability gates (`SCALING_STATUS_AUDIT` item 3) and a
science sign-off on the mixed-precision vmix.

## Atm ladders added late in the campaign

**Lat-lon WEAK at a production tile** (45 rows x 1440 lon x L26 = 64.8k
columns/GPU, f32, job 26454476): efficiency 1.00 / 0.45 / 0.41 / 0.45 /
0.44 at 1/2/4/8/16 GPUs — the cost is paid ONCE on the first cross-device
step and then FLAT to 16 GPUs across two node crossings (1.09 -> 7.66
GCells/s aggregate). Weak scaling on this grid is a fixed entry toll, not
a compounding one.

**MPAS icosahedral L8 (28 km) STRONG, identical padded mesh** (jobs
26454476 + 26454618): 19.90 / 17.09 / 6.92 / 7.10 ms at np 2/4/8/16.
Taking np2 as the base (it has the BEST per-device throughput, 430
Mc/s/GPU): 2->8 = 2.88x = eff 0.72, 2->16 = 2.80x = eff 0.35 (small-tile
floor).

OPEN ANOMALY, characterised not explained: per-GPU throughput dips at
np=4 (247 Mc/s/GPU vs 430 at np2 and 306 at np8), so np4 is barely faster
than np2 while np8 is 2.5x faster than np4. Evidence gathered:
- REPRODUCIBLE: two repeats per arm agree within 1 % (19.92/19.87,
  17.14/17.04, 6.87/6.97).
- PLACEMENT REFUTED: np4 packed on one node (17.09 ms) == np4 spread over
  two nodes (17.12 ms), so node crossing is irrelevant.
- HALO VOLUME REFUTED: ghost-cell census on the padded mesh gives
  1540/1587/1400/1136 ghost cells per device at np 2/4/8/16 — flat to
  falling, and under 3 % of owned cells at every count.
- COLLECTIVE COUNT REFUTED (HLO census, ico L7, CPU virtual devices —
  device count is a compile-time property so the HLO matches what the GPUs
  execute): collective-permutes per step are 3 / 9 / 21 at np 2/4/8, i.e.
  np8 issues 2.3x MORE collectives than np4 and still runs 2.5x faster.
  Collective COUNT therefore cannot explain the np4 dip (this assumes cost
rises with count; a per-message-size effect is not excluded). (Fusion count 136/173/240,
  bitcasts 526/582/694 — the np8 program is finer-grained.)
- PARTITION METHOD REFUTED (job 26455829): the dip is method-independent —
  np4/np8 = 17.22/7.00 ms (sfc), 17.20/6.97 (metis), 19.35/6.26
  (geometric). Every method shows the same 2.5-3.1x jump.
- XLA CODEGEN ENV KNOBS REFUTED (job 26455948): np4 is 17.12 ms base,
  17.06 autotune-level-4, 17.10 latency-hiding-off, 17.01
  command-buffers-off — every arm within 1 %, none recovers np4.
  (The multi-output-fusion arm errored on an unsupported flag name and is
  not counted.)
VERDICT: five hypotheses refuted by measurement (placement, halo volume,
collective count, partition method, codegen env knobs). The cheap levers known to this campaign are exhausted; the remaining suspect — per-device kernel efficiency
for this shape — needs a GPU op-level profile (nsys / XLA op profile of
np4 vs np8), which is a separate instrumented project, not another timing
run. Per-GPU throughput across the ladder is non-monotone in tile size
(430 / 249 / 305 / 150 Mc/s/GPU at 327688 / 163844 / 81922 / 40961
cells/device), which is itself the clue to hand the profiler.
PRACTICAL GUIDANCE MEANWHILE: run this grid at np>=8, where per-device
throughput is 304-340 Mc/s/GPU vs 220-248 at np4.

RESOLVED 2026-07-26 (nsys job 26479922 + HLO dump 26480096 + sqlite
timeline): the dip is an XLA CODEGEN pathology, localized to named
kernels. CONFIRMED: (1) the dip reproduces under nsys with matched
protocol (L8, padded-16 mesh: 21.01/17.78/7.17 ms at np2/4/8 vs campaign
19.90/17.09/6.92 — ~5% profiler overhead); (2) at np4 ONLY, giant
serialized "loop fusion" kernels appear — loop_add_fusion_1/2 at 3.6 ms
per launch (vs ~3 us for ordinary elementwise kernels) plus a THIRD
once-per-step group (the unsuffixed loop_add_fusion: 12 of its 44
instances are >1 ms at ~3.3 ms, the rest are the ordinary us-scale adds)
— and the sqlite timeline places all three groups' big instances at the
17.8 ms step cadence (stddev 78 us: deterministic compute, not comm
wait): 3.6 + 3.6 + 3.3 ~= 10.5-10.9 ms/step = the np4 excess; (3) in the optimized
step HLO these are mega-fusions ON THE HALO PATH: `%loop_add_fusion =
f32[491520,26]` (edge-tendency add chain, 22 operands incl. an
input_scatter_fusion) and `%loop_add_fusion.4 = f32[163844,26]` (cell
array), with the shard_map halo-pack concatenates taking the same adds +
parameter lists as operands. PLAUSIBLE (inferred from kInput fusion
semantics + operand lists, not separately timed): the emitter RECOMPUTES
the expensive scatter+add chain inside each consumer fusion, which is why
the cost multiplies. WHY np4: fusion cost-model decisions depend on the
shard shape; at np2/np8 the mega-fusion is not built. This also explains
why the earlier env-knob sweep missed it — autotune/latency-hiding flags
do not change fusion-pass decisions. Fusion-pass flag A/B at np4 ran
(job 26480162): flag route CLOSED — three of four candidate fusion flags
no longer exist in this XLA (upstream removals), the fourth is null, and
the GPU plugin does not list its flags via --help.

FIX ATTEMPTS, both measured (base 19.90 / 17.09 / 6.92 ms at np2/4/8):

| barrier placement | np2 | np4 | np8 | verdict |
|---|---|---|---|---|
| tendency INPUT side (job 26480261) | 21.40 | 16.58 | 7.01 | null at np4, -7.5% np2 — REVERTED |
| tendency OUTPUT side (job 26480310) | 22.03 | **14.09** | 7.04 | **-17.6% time np4** (1.21x), +10.7% time np2 — REVERTED |

(Single runs per arm; the campaign's np4 repeat spread (+-0.3%) supports
an informal ~+-1 pp error on these percentages, not a formal CI.)

The HLO frame table pinpointed the fusion: the 3.6 ms kernels resolve to
`pytree_ops.py:10` (`pytree_axpy.<locals>.<lambda>`) — the RK stage
combine mega-fused with the tendency graph's tail. An output-side
optimization_barrier recovers 3 ms of the ~10.9 at np4 but costs np2
10.7% (it also blocks fusion that HELPS there), so neither barrier ships
unconditionally. Parity + conservation smoke passed on both attempts.

STATUS: **FIXED, SHIPPED GATED** (codex rounds 11-12: strategy consult
BEFORE implementing, then post-review). `_FUSION_BARRIER_WORKLOADS` in
`sharded_dynamics.py` applies the tendency-output optimization_barrier
only at the measured workload signature (n_dev, edge rows, cell rows,
nlev) = (4, 1_966_080, 655_376, 26) — every operand trace-time static.
Verification ladder (job 26486288 vs same-day dead-gate 26486123):
np2 19.86 (campaign base 19.90 — at baseline), **np4 14.12 = -20.6%
same-day / -17.4% vs campaign base**, np8 6.99 (base 6.92). Parity +
conservation smoke green; 23 SPMD parity tests pass. Two instructive
misfires on the way, both caught by measurement: the first gate keyed
per-shard rows (never fired — the trace-time array is the GLOBAL view),
and edge-rows-only was over-broad (L8 edges are unpadded and divisible
several ways — codex round-12). The residual np4 gap to ideal (~14.1 vs
~9.9 from np2/2) is the un-barriered remainder of the fusion; further
recovery needs the integrator-level restructure (codex round-11 ranked
it last on blast radius) or an upstream XLA fix — both remain
follow-ups.

## Ocean strong scaling vs TILE SIZE (jobs 26456334/37 vs 26452804-06)

The same improved config (wide-halo + vmix-f32, multicontroller NCCL/IB,
f32 L20) run at two tile sizes, 4 -> 16 GPUs:

| grid | cells/GPU @16 | np4 / np8 / np16 ms | eff @8 | eff @16 | aggregate @16 |
|---|---|---|---|---|---|
| LL576x1152 (13.3M) | 0.83M | 12.81 / 11.05 / 8.63 | 0.58 | 0.37 | 1.53 GCells/s |
| LL1152x2304 (53.1M) | 3.3M | 43.47 / 27.79 / 17.24 | **0.78** | **0.63** | **3.08 GCells/s** |

Both ladders are monotone; the bigger tile is uniformly better at every
device count (jobs 26456334 / 26457693 / 26456337 vs 26452804-06).

So the ocean shows the SAME tile-size dependence the cube does: the 2.01x
multinode improvement measured at LL576 was partly a floor effect, and at
a production tile the identical code scales substantially better (0.37 ->
0.63). Per-device throughput also rises (259 -> 305 Mc/s/GPU at np4).
Config is byte-identical between the two rows; only the grid changes.

REFUTED EN ROUTE: the np8 leg timed out twice (>90 min still tracing) while
np4 — a LARGER per-device tile — finished in ~25 min, which looked like a
compile-time cliff at that device count. It is not: the third attempt ran
the identical configuration in **99 seconds** with a 21.6 s compile (job
26457693). The earlier hangs were transient/environmental, not
reproducible, and no compile-time defect is claimed.

## Weak scaling at production per-device size (job 26453523)

The earlier weak ladders used a 64-row base (0.17M cells/GPU — under the
latency floor). Re-run at PRODUCTION size (288 rows × 1152 lon × L20 =
6.6M cells/GPU, 1→4 A100, conservation gated): production implicit
1.00/0.72/0.70, improved wide-halo+vmix-f32 1.00/0.86/0.85.
PRECISION MATCHED (self-audit correction): BOTH ladders compared here are
**float32** — the production-tile run is f32, so it is compared against
the earlier ladder's f32 rows (eff 0.26/0.25 at nd 2/4), not its f64 rows (both from job 26445836).
An earlier revision of this file mislabelled the production-tile run f64
and cited the f64 small-base numbers; the direction and size of the effect
are unchanged, but the comparison is only valid precision-matched. So the earlier weak ladder measured a below-floor tile rather than a code
limit, and the same
config that fixes strong scaling also carries weak (+0.15 at nd4). Ideal is
flat; the improved arm holds 22.5→22.9 ms while production drifts
17.8→25.3 ms.

## "It used to be faster / did we regress?" — resolved, no regression

- Cross-machine anchor (matched bench/config/grid/physics/precision,
  job 26450081): Levante single A100-80 latlon-moist r720 f32 =
  **418.5 Mc/s** vs Derecho single-A100 ≈370. SCOPE: this establishes NO
  LARGE REGRESSION, not a precise machine ranking — the two campaigns
  differ in machine (A100-80 SXM vs A100-40), jax/tree version and date,
  so the ~13% gap is not attributable to any single factor.
- Cube "404 vs 141 Mc/s": the 404 is the single-GPU RTX-5090 Held-Suarez
  row in `SCALING_SUMMARY.md` SS1; ours is gray+SBM on A100 (job 26445836).
  That file's own tier table prices gray+SBM ~3x Held-Suarez, so ~135 is
  the expected equivalent vs 141 measured. PLAUSIBLE reconciliation from
  two published tables, NOT a matched A/B (GPU, physics and date differ).
- Ocean absolutes (A100 f64 165-201 Mc/s, f32 352, job 26445836; 5090 f64
  152 / f32 400 from `SCALING_SUMMARY.md` SS1) are of the same order -
  again a cross-machine sanity check, not a controlled comparison.
- Ginsburg "0.92 eff @2 GPU" reconciled: the old bench silently defaulted
  to `explicit_substep`; our explicit/wide arm reproduces that class
  (0.88 @2, LL384) — the production implicit config was never measured
  there. So the gap is explained by the solver the old bench selected;
  labelling it 'protocol, not regression' is an inference from that
# Spectral GPU-native transform feasibility — go/no-go (2026-07-08)

Scaling-audit item 9: assess GPU-native transform options for the global
spectral dycores (`spectral_pe/sw/nh` on the Gaussian grid) — batched GEMM,
SHTns/sphericart, cuFFT/cuBLAS layout changes, or CPU-only status quo.
**Assessment + microbenchmarks only; no dycore rewrite.**

## What the transform is today

`packages/core/legoesm/grids/gaussian.py`: FFT in longitude
(`jnp.fft.rfft/irfft`) + the Legendre leg as a **dense complex GEMM**
(`_analysis/_synthesis_legendre_gemm`, einsum over the packed
`(n_lat, n_sh, nlev)` tensor), f64+complex128 hard-gated (the grid builder
raises without x64). A bf16-GEMM ablation lane already exists
(`LEGOESM_SH_GEMM_BF16`, `sh_gemm_design_2026-06-13.md`). So the "batched
GEMM on GPU" option is **not a rewrite — it is the current code running on
a CUDA backend**; the question is only whether the hardware's fp64 GEMM
rate makes it worthwhile.

## Microbenchmark (this audit)

`scripts/bench/bench_spectral_transform_micro.py` — isolated
analysis+synthesis round trip; the Legendre path is EXPLICIT per row
(`legendre_path`: the legacy gather/segment-sum default vs the opt-in
`LEGOESM_SH_GEMM=1` batched-GEMM), the timed input is band-limited first
(synthesis∘analysis is a projection, so the recorded error is the
transform's own, not truncation), and the recorded rate is an
*equivalent*-GEMM number (model FLOPs / time). Measured on the laptop
**CPU** lane (f64, nlev=30, medians of 20 — these rows are explicitly NOT
the A100/H100 answer; the backend is on every row):

| T | grid | n_sh | legacy round trip | **GEMM round trip** | GEMM share of model FLOPs | AI |
|---|------|------|------------------:|--------------------:|--------------------------:|---:|
| T42 | 64×128 | 946 | 5.34 ms | 3.04 ms | 46% | 10.0 F/B |
| T85 | 130×260 | 3,741 | 33.5 ms | 20.5 ms | 59% | 12.1 F/B |
| T170 | 256×512 | 14,706 | 145.6 ms | **43.2 ms** | 72% | 13.4 F/B |

(FLOP model: the Legendre matrices are REAL against complex fields — a
real×complex MAC is 4 real FLOPs; the first draft double-counted at 8.)

Reading: (a) the opt-in GEMM path is **3.4× faster than the legacy path
at T170 on CPU alone** (identical band-limited round-trip error,
4.2e-12) — this audit's own adversarial review caught the first draft
measuring the legacy path while labeling it GEMM; (b) the transform is
GEMM-dominated (72% of model FLOPs at T170, share growing with T) with
AI 10–13 F/B — compute-bound on fp64-weak parts, near the roofline knee
on fp64-strong parts; (c) the FFT leg shrinks with T —
**cuFFT/layout work is not the lever**.

## The consumer-vs-datacenter fp64 split (revises the prior reading)

The audit's "spectral GPU-hostile" evidence (`SCALING_SUMMARY.md`: T42
13 Mc/s fp64, CPU→GPU only 10.9×) was measured on an **RTX 5090 laptop
part — 1:64 fp64:fp32**. That measurement says consumer GPUs are bad at
fp64 GEMM, not that the transform is GPU-hostile:

- A100 fp64: 9.7 TF/s (19.5 TF/s tensor-core DGEMM). Against the measured
  T170 GEMM-path round trip (0.90 GF model / 43.2 ms ≈ 21 GF/s equivalent),
  the GEMM-roofline headroom is **roughly two orders of magnitude** — a
  ROUGH UPPER BOUND, not a prediction: the model FLOPs are the triangular
  count, the FFT leg and launch overheads are outside the GEMM roofline,
  and realized DGEMM fractions vary. The point stands at any realistic
  discount; only the measurement settles it.
- The GEMM path lowers to cuBLAS on CUDA; the measurement costs one
  command on a Derecho/Levante GPU node (``--sh-gemm both`` records the
  legacy column too, same rows schema):

```
JAX_ENABLE_X64=1 python scripts/bench/bench_spectral_transform_micro.py \
    --truncations 42,85,170,341 --nlev 60 --sh-gemm both \
    --out results/spectral_micro_a100.json
```

## Options assessed

| option | verdict | why |
|---|---|---|
| **Batched GEMM on datacenter GPU** (existing code path) | **GO — measure first** | Transform is GEMM-bound (72% at T170, AI 13); A100-class fp64 GEMM headroom is orders of magnitude; zero integration work. Run the one-command microbench above before ANY further investment. |
| **SHTns (GPU build)** | **NO-GO now** | External C dependency via FFI custom-call: breaks `jax.grad` without a hand-written custom_vjp pair, adds a build dependency to every cluster, and its GPU path targets the same GEMM/FFT arithmetic the einsum already reaches through cuBLAS. Revisit only if the GEMM microbench shows the einsum path leaving >2–3× on the table. |
| **sphericart** | **NO-GO** | Real-SH evaluation for point clouds (derivatives of Y_lm at scattered points) — not a Gauss–Legendre grid transform; wrong tool. |
| **cuFFT/cuBLAS layout surgery** | **NO-GO** | FFT share falls with T (28% at T170); the packed `(n_lat, n_sh, nlev)` GEMM layout is already the batched-GEMM-friendly one (`_maybe_chunk_trailing` handles memory). |
| **CPU-only status quo** | **Default pending measurement** | Correct, validated, and the SI solve keeps the dycore single-device anyway (below). |

## Multi-device: stays N/A (unchanged)

The audit matrix verdict stands — **no-go for multi-device spectral
investment**:

- Both sharding schemes measured ANTI-scaling
  (`spectral_level_shard_cliff.md`: T85 0.74× at 4 devices) — the
  semi-implicit solve couples all levels (all-gather per solve), and the
  transform itself is replicated under level sharding.
- The 1:64-fp64 consumer measurement does not change the collective-
  coupling math; a fast single-GPU transform makes the *single-device*
  dycore faster, not the sharded one.

## Go/no-go summary

1. **GO (cheap, first):** run the microbench on one A100/H100 node
   (command above). If the round trip lands within ~5× of the GEMM
   roofline, the existing einsum path IS the GPU-native transform and the
   spectral dycore becomes a **single-GPU** citizen on fp64-strong parts —
   no library integration, no rewrite. Wire `--truncations 341` to probe
   the production-relevant end.
2. **NO-GO:** SHTns/sphericart integration, cuFFT layout work, and any
   multi-device spectral scheme — each is dominated by either the existing
   GEMM path or the measured SI-coupling wall.
3. The dycore-side fp64 constraint is physics
   (`core/precision.py::_ATMOSPHERE_OVERRIDES`), not transform-imposed;
   the bf16-GEMM ablation lane remains the sanctioned precision
   experiment.
# MPAS-ocean distributed step — stage-correctness audit (2026-07-11)

Scaling-M3d increment-1 deliverable. Reconciles two claims:

* `SCALING_STATUS_AUDIT.md` (ocean matrix, MPAS/Voronoi row, 2026-07-07):
  "`voronoi_mpi` step exists (`make_voronoi_mpi_step`); MPI conservation
  tested; NO scaling bench lane drives it".
* User directive: "a complete stage-correct distributed MPAS-ocean step does
  not yet exist".

**Verdict: the user directive is correct; the status row was stale/conflated
on three counts.**

1. `make_voronoi_mpi_step` (`packages/core/legoesm/parallel/voronoi_mpi.py:618`)
   is the **atmosphere** MPAS step (`MPASHydrostaticState`; per-RK-stage
   packed halo refreshes + owned-cell mass fixer). It cannot step
   `MPASOceanState`. The OCEAN has no equivalent wrapper.
2. A bench lane DOES exist since audit item 6 landed:
   `scripts/bench/bench_ocean_mpas_scaling.py` steps `MPASOceanModel` on
   `layout.local_mesh` multi-rank with parity + conservation gates and
   partition metrics — but the step it drives is not stage-correct (below),
   and the lane lacked the M1 measurement contract (fused-scan
   `timed_scan_blocks`, solver-residual reporting, wet-cell metrics) until
   this increment.
3. "MPI conservation tested" for the OCEAN reduces to: the np=2
   scatter/gather roundtrip (`tests/ocean/distributed/test_mpas_ocean_scatter.py`),
   the distributed-PCG solver parity
   (`tests/ocean/distributed/test_barotropic_pcg_mpas_mpi.py` — solver-level,
   its own docstring: "the full ocean-state scatter is a separate follow-on"),
   and the bench lane's own `--check-conservation` gate.
   `tests/ocean/distributed/test_ocean_mpi_conservation.py` is CUBED-SPHERE,
   not MPAS.

## Spec audited against

"Packed stage-level cell/edge/vertex halo refreshes and owned-only
reductions." For every stage of `MPASOceanModel._step_impl`
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py:350`) under an
armed `VoronoiPartitionLayout` (rank-local mesh, `halo_depth=2`):
which halo exchanges happen, are they packed or per-field, and are global
reductions owned-only?

Key mechanics that bound correctness:

* `build_local_mesh` (`packages/core/legoesm/parallel/voronoi_partition.py:781`)
  remaps out-of-partition connectivity to `-1`; TRiSK operators mask those
  entries. So a local stencil op produces MASKED-WRONG (deterministic, not
  garbage) values on the outermost halo ring; each additional stencil hop
  propagates the wrongness one ring inward. With `halo_depth=2`, owned cells
  stay exactly correct through **2 stencil hops after the last halo
  refresh** — beyond that, owned cells adjacent to the partition boundary
  silently diverge from serial.
* The only packed full-state exchange machinery in the tree is the
  atmosphere's `_exchange_mpas_state`
  (`voronoi_mpi.py:778` batched / `voronoi_mpi.py:840` legacy per-entity).
  Until this increment the ocean had NO state exchange helper at all; the
  only ocean halo traffic was the per-field cell exchange inside the
  distributed PCG matvec. This increment adds the packed ocean twin
  `exchange_state_mpas_ocean` (`voronoi_mpi.py`, next to
  `scatter_state_mpas_ocean`) — one batched union-neighbor message per
  neighbor per dtype group for u (edge) + T, S, eta, w (cell).
* Vertex quantities (PV, curl) are recomputed locally from edge `u` each
  evaluation (`curl_vertex_3d`); no vertex halo exchange exists anywhere
  (atmosphere included) and none is needed PROVIDED edge halos are fresh and
  the vertex consumer sits within the hop budget — vertex values on the
  outer ring are masked-wrong and count as one hop.

## Stage table — `MPASOceanModel._step_impl` under a Voronoi partition

Hops = horizontal stencil depth consumed by the stage (rings of halo
correctness eaten since the last refresh). Verdicts assume fresh halos at
step ENTRY (which nothing guaranteed before this increment: the lane now
refreshes per step; within-step staleness remains).

| # | stage | file:line | stencil hops | halo refresh in stage | global reductions | verdict |
|---|-------|-----------|--------------|----------------------|-------------------|---------|
| 0 | scatter / initial fill (`scatter_state_mpas_ocean`) | `voronoi_mpi.py:300` | — | initial (slice = fresh) | — | CORRECT; np=2 roundtrip-tested |
| 1 | baroclinic tendencies (`mpas_ocean_baroclinic_tendencies`) | `ocean_pe_mpas.py:92` (called `ocean_model_mpas.py:401`) | up to 4 (B_h del4 `ocean_pe_mpas.py:585`; K_bih grad→div→grad→div `:846-854`; PV/curl/tangential 1–2; fills 1) | NONE | none on default path; `normalize_freshwater` rank-local mean is REFUSED multi-rank at the source (`ocean_pe_mpas.py:947-968`) | NOT stage-correct: consumes >2 hops with no refresh; owned boundary cells diverge |
| 2 | tracer forward-Euler + land fill | `ocean_model_mpas.py:406-415` | 1 (`fill_land_cells_mpas`) | NONE | — | eats 1 more hop of stage-1 output |
| 2a | implicit vertical tracer diffusion + KPP/TKE/conv K-profiles | `ocean_model_mpas.py:432-521` | ≥1 — KPP and TKE profile entry both call `_reconstruct_mpas_cell_fields` (`mpas_integration.py:214`), whose `reconstruct_cell_velocity` (`mpas_integration.py:246`; KPP consumer `:304`, TKE `:707`) is a horizontal TRiSK/Perot edge→cell stencil on edge `u`; only the vertical solves/EOS are column/cell-local | NONE | — | NOT zero-hop: the shear-driver reconstruction reads edge-`u` halos (stale-able); eats 1 more hop |
| 2b | GM/Redi + MLE bolus tendencies | `ocean_model_mpas.py:529-553` | 1–2 | NONE | — | further hop consumption |
| 3 | momentum Euler + implicit vertical viscosity | `ocean_model_mpas.py:560-641` | 1 (`min_cell_to_edge`) | NONE | — | |
| 3b | forward-backward Coriolis on u' | `ocean_model_mpas.py:59-136` (called `:647`) | 2 (`tangential_velocity_3d` ×2, `edgesOnEdge`) | NONE | — | |
| 4 | barotropic EXPLICIT substeps (`barotropic_substeps_mpas`) | `barotropic_mpas.py:48-382` | 2–6 PER SUBSTEP (div 1 + grad 1 + tangential 1–2 + optional div-damp/visc 2 each) × `n_barotropic_substeps` (default 30) | NONE inside the `lax.scan` | eta-floor clamp owned-masked + forced-global (`barotropic_mpas.py:122-126,272,357` via `eta_floor.py:53`) | reductions owned-only CORRECT; stencils NOT stage-correct multi-rank (≫2 hops); **no multi-rank refusal on this branch** (unlike implicit_cn) |
| 4' | barotropic IMPLICIT CN (`barotropic_implicit_mpas`) | `barotropic_implicit_mpas.py:376-794` | predictor ~3 before the solve; PCG matvec halo-composed per iteration | per-field CELL exchange inside every `A_op` (`:609-610`); solution halo refreshed post-solve (`:634`) | entry refusal for layout-less multi-rank (`:437-444`); owned-masked area-weighted dots (`:615-616`); owned-masked mass projection, ONE batched allreduce (`:665-684`); owned-masked clamp (`:695-699`); owned-masked residual (`:709-719`) | solver INTERNALLY stage-correct (np=2 parity-tested); its predictor/RHS consume stale input halos like every other stage |
| 5–6 | layer thickness + `reconcile_3d_velocity` | `ocean_model_mpas.py:753-786`, `barotropic_mpas.py:385-415` | 1 | NONE | — | |
| 7 | transport-consistency correction `delta_u` | `ocean_model_mpas.py:799-807` | 0 (edge-local) | NONE | — | |
| 8 | w diagnosis (`divergence_cell_3d`) | `ocean_model_mpas.py:819` | 1 | NONE | — | |
| 9 | flux-form tracer advection (upwind/TVD) | `ocean_model_mpas.py:838-890` | 1 (upwind) / 2 (TVD upup) | NONE | — | |
| 10 | conservation fixer | `ocean_model_mpas.py:914-972` + `conservation_mpas.py:59-97` | 0 | — | owned mask pulled from the matching layout (`ocean_model_mpas.py:923-941`); expected-forcing sums `global_sum_mpi` (`:960-966`); fixer core via `global_sum_if_distributed` | owned-only CORRECT |
| 11 | freeze floor | `ocean_model_mpas.py:974-1002` | 0 | — | — | correct |
| — | `check_barotropic_cfl` / `_assert_runtime_invariants` | `ocean_model_mpas.py:297-332,1048-1107` | — | — | rank-LOCAL host reductions | diagnostics-only (warnings/checks may differ per rank); acceptable, documented |

## Findings

1. **Owned-only reductions: PASS.** Every collective on the step path is
   owned-cell-masked (eta-floor clamp, implicit-PCG dots/projection/residual,
   conservation fixer) or refused multi-rank at the source
   (`normalize_freshwater`). No double-count bug found — nothing to fix here.
2. **Stage-level halo refreshes: FAIL (structural).** No stage of the ocean
   step refreshes state halos; the only in-step exchange is the per-field
   cell exchange inside the implicit-PCG matvec. The cumulative hop count of
   one step (≥11 even without the explicit substep loop — stage 2a's KPP/TKE
   cell-velocity reconstruction included; hundreds with it)
   vastly exceeds `halo_depth=2`, so a multi-rank step silently diverges from
   serial near partition boundaries at a rate bounded by field evolution per
   step (slow flows ⇒ small parity error — why the smoke-window parity gate
   measured ~9e-9 at L2×2 steps; that gate bounds staleness, it does not
   prove stage correctness).
3. **Explicit-substep branch has no multi-rank guard** while `implicit_cn`
   refuses a layout-less multi-rank launch
   (`ocean_model_mpas.py:727-739`, `barotropic_implicit_mpas.py:437-444`).
   The bench lane is the intended multi-rank driver of that branch and now
   labels every row with `halo_refresh` + `stage_halo_correct=false` so a row
   cannot masquerade as a stage-correct scaling claim.

## Fixed in this increment (small, gated)

* Packed full-state ocean halo refresh helper `exchange_state_mpas_ocean`
  (`voronoi_mpi.py`, batched union-neighbor exchange; schema tripwire matching
  the scatter/gather pair; identity at np=1). Unit-tested single-process:
  np=1 identity PLUS a mocked-exchange sentinel test
  (`tests/parallel/test_voronoi_mpi_ocean_exchange.py`) asserting all five
  prognostic fields (u | T, S, eta, w) route through the packed call and land
  in the right state slots — the np=1 identity path alone cannot see an
  omitted/mis-wired field. Real multi-rank transport rides the generic
  batched-halo distributed tests; the np≥2 bench-lane smokes drive this
  helper on-cluster (not a CI gate).
* Bench lane (`scripts/bench/bench_ocean_mpas_scaling.py`) extended to the M1
  measurement contract: fused-scan `timed_scan_blocks` timing (cross-rank
  MAX-reduced via the unit-tested `reduce_block_times`), recorded as the
  aggregator-facing `steady_median_ms` — the host-synced gate-loop median is
  demoted to `step_latency_gate_loop_ms` (dispatch+sync latency, never the
  headline); `wet_cell_metrics`; solver-iteration + zero-forcing
  Helmholtz residual probe (`barotropic_implicit_mpas(..., return_residual=True)`
  outside the timed loop — under `--halo-refresh none` one packed
  `exchange_state_mpas_ocean` refresh precedes the probe so it measures a
  cleanly-assembled system, not rotten halos); `--barotropic-solver`
  selection; and a per-step packed halo refresh
  (`--halo-refresh auto|per_step|none`, default auto ⇒ per_step at np>1) so
  multi-rank rows pay representative exchange cost and the step INPUT is
  fresh each step. Rows are self-describing: `stage_halo_correct=false`
  until the per-stage refreshes land, and `stage_halo_note` states the
  ACTUAL refresh mode (per-step refresh vs across-step halo rot for `none`).

## IMPLEMENTED 2026-07-17 — in-step stage-frontier refreshes (codex scaling lever 1)

The "Deferred" plan below is now BUILT: `MPASOceanModel.step(halo_refresh=
make_mpas_ocean_halo_refresh(layout))` threads a packed refresh object
(`voronoi_mpi.MPASOceanHaloRefresh`: `edges`/`cells`/`both`, each ONE batched
union-neighbor message per dtype group, AD-safe custom_vjp sendrecv,
scan-safe) through every audited frontier:

| site | where | fields | why |
|------|-------|--------|-----|
| T1 | `ocean_pe_mpas` viscosity (+ `mid_refresh` on `vector_laplacian_del4_3d` / `smagorinsky_biharmonic_3d` / `leith_biharmonic_3d`) | intermediate del2 (edge) | two-pass biharmonics = 4 hops > halo_depth |
| T2 | `ocean_pe_mpas` K_bih | inner Laplacian (cell) | bilaplacian outer pair needs a fresh ring |
| T3 | `ocean_pe_mpas` K_zeta_bih (`biharmonic_vorticity_del4_3d` `mid_refresh`) | vertex Laplacian of ζ (VERTEX channel, per-field `VoronoiHaloExchange`) | curl→vertex-Laplacian→tangential-gradient = 3 hops; the NEMO-match recipe runs `K_zeta_bih=1e14` (codex r1 #2) |
| R1 | `_step_impl` post-tracer-fill | T, S | updated-tracer ring carries neighbor-rank tendencies; KPP/GM/MLE consume 1-2 hops |
| R2 | `_step_impl` pre-Coriolis | u_star | FB-Coriolis = 2 tangential hops on UPDATED u |
| B0 | `barotropic_substeps_mpas` pre-scan | u_bar, F_slow_u + F_slow_eta | scan-constant rings refreshed once |
| B1 | substep entry (in-scan) | u_bar, eta (one packed msg) | each substep consumes 2-6 hops |
| B2 | substep PGF (in-scan) | eta_pgf | continuity already ate the 2-ring budget before fill+grad |
| — | substep optional blocks | u_bar_next / eta_next | div-damp (3 hops), baro-visc (2), eta-diffusion (3) |
| I0 | `barotropic_implicit_mpas` predictor | u_bar_old, F_slow_u | Heun tangential hops on the depth-mean of post-Coriolis u |
| I1 | implicit RHS | grad_eta_old | fill+grad+div = 3 chained hops from eta |
| R3 | `_step_impl` post-reconcile | u_3d_new, Hu_avg + T, S, eta_new (one packed msg) | w-diagnosis + TVD advection + delta_u ring |

`None` (the serial default) keeps every consumer byte-identical (static
Python branches).  Gates (`tests/ocean/distributed/test_mpas_ocean_stage_halo.py`):
serial identity-refresh bit-parity with every refresh-bearing branch enabled
(both solvers), `jax.grad` parity, and the np=2 owned-cell parity vs serial
at near round-off WITH the non-vacuity tripwire (the entry-refresh-only run
must be measurably worse — a refactor that silently no-ops the refreshes goes
red).  The bench lane's `--halo-refresh in_step` (auto's multi-rank choice)
arms it and rows then earn `stage_halo_correct=true`.  Remaining exclusions:
`normalize_freshwater` stays refused multi-rank (owned-mask plumbing, below)
and the production OMIP driver still runs MPAS single-rank (`--n-gpus`
rejects mpas) — the bench lane is the multi-rank driver.

## Deferred (structural, precisely scoped) — original plan, now superseded above

Threading per-stage refreshes into `_step_impl` is the real distributed-step
milestone, NOT a small fix:

* stage 1 entry exchange (u,T,S,eta packed — now available as
  `exchange_state_mpas_ocean`), plus `halo_depth ≥ 4` or operator-splitting
  of the del4/K_bih stencils;
* the per-stage hop budget MUST count stage 2a: the KPP/TKE K-profiles are
  not column-local — `_reconstruct_mpas_cell_fields` spends 1 horizontal hop
  (edge-`u` → cell velocity, TRiSK/Perot) before any vertical solve, so the
  refresh (or wide-halo) plan between the stage-1 entry exchange and the
  barotropic section covers stages 2+2a+2b+3+3b ≈ 7 hops, not 6;
* per-substep (or wide-halo, `halo_depth ≥ 2·n_substeps`-style) exchange of
  `(eta, u_bar)` inside `barotropic_substeps_mpas`'s scan — the same
  trade-off the lat-lon wide-halo split-explicit lane measures
  (`SCALING_STATUS_AUDIT.md` improvement candidate 3);
* post-barotropic and pre-tracer-advection refreshes of `(u_3d, eta)`;
* thread `owned_mask` into the freshwater normalization means to retire the
  multi-rank refusal (`ocean_pe_mpas.py:947`; helper support already exists —
  `freshwater.py:181-261`).

Until then, multi-rank MPAS-ocean full-step rows are throughput/parity-bounded
evidence (tag (c) of `SCALING_STATUS_AUDIT.md`), not stage-correct scaling
claims — enforced by the row metadata.

exec
/usr/bin/bash -lc "sed -n '1,260p' scripts/bench/bench_barotropic_mcut.py && sed -n '1,280p' scripts/bench/bench_barotropic_precond_convergence.py && rg -n \"precond|multigrid|chebyshev|pipelined|s.step|fixed_iteration_pcg|single_reduce|barotropic_solver\" packages/ocean/legoesm/ocean/dynamics packages/core/legoesm | head -400" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 619ms:
"""Distributed barotropic-PCG iteration-count (M) cut: jacobi vs banded multigrid.

The barotropic implicit solve is a fixed-M preconditioned-CG; each outer
iteration issues a batched ``allreduce(SUM)`` for its two dot products, so the
per-step GLOBAL-reduction count is ``2*M`` — the multinode weak-scaling
reduction-latency wall.  The banded geometric-multigrid preconditioner
(``_make_multigrid_preconditioner_banded``) has NO reduction inside its V-cycle
(only neighbour halos), so if it cuts M from ~60 (jacobi) to ~4-8 it cuts the
per-step reduction count ~120 -> ~16.

This sweeps M for each preconditioner under MPI and records the relative
residual, finding the M each needs to reach a target — the iteration-count cut.
The V-cycle is reduction-free, so the cut holds DISTRIBUTED (this is what makes
it a weak-scaling win, not just a per-device one).

Run::

    mpirun -np 2 python scripts/bench/bench_barotropic_mcut.py --n-lat 96 --n-lon 192

Writes ``docs/performance/scaling/barotropic_mcut.csv`` on rank 0 for the plotter.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from mpi4py import MPI

from legoesm.grids.halo import set_halo_backend
from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
from legoesm.parallel.latlon_mpi import (
    make_latlon_band_layout, slice_latlon_grid_to_band,
)
from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
    _make_helmholtz, _helmholtz_inv_diag, _faces_from_cell_depth,
    _select_preconditioner,
)
from legoesm.ocean.dynamics.barotropic_common import solve_helmholtz_implicit


def _coastal_mask(n_lat, n_lon):
    m = np.ones((n_lat, n_lon))
    m[0, :] = 0.0
    m[-1, :] = 0.0
    m[:, n_lon // 8: n_lon // 8 + 4] = 0.0           # meridional coast
    m[2 * n_lat // 5: 2 * n_lat // 5 + 6,
      4 * n_lon // 9: 4 * n_lon // 9 + 15] = 0.0      # interior basin
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-lat", type=int, default=96)
    p.add_argument("--n-lon", type=int, default=192)
    p.add_argument("--coeff", type=float, default=5.0e7)
    p.add_argument("--m-sweep", type=int, nargs="+",
                   default=[2, 4, 6, 8, 12, 20, 40, 60])
    p.add_argument("--target", type=float, default=1e-6)
    p.add_argument("--out", default="docs/performance/scaling/barotropic_mcut.csv")
    # Wall-time mode (multi-node): time jacobi-M vs multigrid-M solves so the
    # reduction-latency win shows in ms/solve (Gloo allreduce dominates at scale).
    p.add_argument("--time-solves", type=int, default=0,
                   help="if >0, time N jitted solves per preconditioner")
    p.add_argument("--jacobi-m", type=int, default=60)
    p.add_argument("--mg-m", type=int, default=12)
    p.add_argument("--cheby-m", type=int, default=20)
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    rank, n_ranks = comm.Get_rank(), comm.Get_size()
    n_lat, n_lon = args.n_lat, args.n_lon
    if n_lat % n_ranks != 0:
        if rank == 0:
            print(f"SKIP: n_lat {n_lat} not divisible by n_ranks {n_ranks}")
        return

    coeff = jnp.asarray(args.coeff)
    grid_raw = create_latlon_grid(n_lat, n_lon)
    mask = jnp.asarray(_coastal_mask(n_lat, n_lon))
    rng = np.random.default_rng(13)
    H_cell = jnp.asarray(1000.0 + 500.0 * rng.random((n_lat, n_lon))) * mask
    rhs_g = jnp.asarray(rng.standard_normal((n_lat, n_lon))) * mask

    layout = make_latlon_band_layout(rank, n_ranks, n_lat, n_lon)
    s, e = layout.lat_start, layout.lat_end
    set_halo_backend("mpi", layout)

    grid_l = ensure_geometry(slice_latlon_grid_to_band(grid_raw, layout))
    Hc_l, m_l = H_cell[s:e], mask[s:e]
    rhs_l = rhs_g[s:e]
    H_u, H_v, u_mask, v_mask = _faces_from_cell_depth(Hc_l, m_l,
                                                      layout.n_lat_local, n_lon)
    A_op = _make_helmholtz(H_u, H_v, coeff, grid_l, m_l, u_mask, v_mask)
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid_l, m_l)
    w = grid_l.area * m_l
    x0 = jnp.zeros_like(rhs_l)

    precs = {
        "jacobi": _select_preconditioner(
            "jacobi", inv_diag, H_u, H_v, coeff, grid_l, m_l, A_op=A_op),
        "chebyshev": _select_preconditioner(
            "chebyshev", inv_diag, H_u, H_v, coeff, grid_l, m_l,
            A_op=A_op, cheby_degree=4),
        "multigrid": _select_preconditioner(
            "multigrid", inv_diag, H_u, H_v, coeff, grid_l, m_l,
            A_op=A_op, H_cell=Hc_l, layout=layout),
    }

    def resid(M_inv, M):
        _, diag = solve_helmholtz_implicit(
            A_op, rhs_l, M_inv, x0, distributed=True, fixed_iters=M,
            residual_tol=1e-30, stock_cg_tol=1e-12, stock_cg_maxiter=200,
            pcg_variant="standard", dot_weight=w)
        return float(diag.rel_residual)

    rows = []
    for name, M_inv in precs.items():
        for M in args.m_sweep:
            r = resid(M_inv, M)
            rows.append((name, M, r))
            if rank == 0:
                print(f"  {name:10s} M={M:3d}  rel_residual={r:.3e}  "
                      f"reductions/step={2 * M}")

    # Wall-time: jitted solve ms for the candidate reduction-cutters vs jacobi.
    # At MULTI-NODE the Gloo allreduce latency dominates the barotropic (phase
    # split np32: barotropic 24%, 120-allreduce floor 21%).  The CHEAP cutters
    # — single_reduce (1 allreduce/iter = M reductions, same jacobi per-iter)
    # and chebyshev (reduction-free degree-4 inner polynomial, fewer outer M) —
    # cut reductions WITHOUT the expensive V-cycle compute that made the banded
    # MG wall-time-negative (job 8488551).  reductions/step: standard = 2*M,
    # single_reduce = M.
    if args.time_solves > 0:
        import time as _time

        def _timed(M_inv, M, variant):
            f = jax.jit(lambda b: solve_helmholtz_implicit(
                A_op, b, M_inv, x0, distributed=True, fixed_iters=M,
                residual_tol=1e-30, stock_cg_tol=1e-12, stock_cg_maxiter=200,
                pcg_variant=variant, dot_weight=w)[0])
            f(rhs_l).block_until_ready()          # compile + warmup
            comm.Barrier()
            t0 = _time.perf_counter()
            for _ in range(args.time_solves):
                f(rhs_l).block_until_ready()
            comm.Barrier()
            return (_time.perf_counter() - t0) / args.time_solves * 1000.0

        # (label, M_inv, M, variant, reductions/step)
        configs = [
            ("jacobi_std", precs["jacobi"], args.jacobi_m, "standard",
             2 * args.jacobi_m),
            ("jacobi_singlereduce", precs["jacobi"], args.jacobi_m,
             "single_reduce", args.jacobi_m),
            ("chebyshev", precs["chebyshev"], args.cheby_m, "standard",
             2 * args.cheby_m),
            ("multigrid", precs["multigrid"], args.mg_m, "standard",
             2 * args.mg_m),
        ]
        timed = [(lbl, _timed(mi, M, var), red, M)
                 for (lbl, mi, M, var, red) in configs]
        if rank == 0:
            base = timed[0][1]   # jacobi_std
            print(f"\n=== WALL-TIME np{n_ranks} ({n_lat}x{n_lon}) ===")
            for lbl, t, red, M in timed:
                print(f"  {lbl:20s} M={M:3d}: {t:8.3f} ms/solve "
                      f"({red} reductions/step)  speedup={base / t:.2f}x")
            wt_out = args.out.replace(".csv", "_walltime.csv")
            os.makedirs(os.path.dirname(wt_out), exist_ok=True)
            with open(wt_out, "w") as f:
                f.write("config,M,ms_per_solve,reductions_per_step,speedup,"
                        "n_ranks,n_lat,n_lon\n")
                for lbl, t, red, M in timed:
                    f.write(f"{lbl},{M},{t:.4f},{red},{base / t:.4f},"
                            f"{n_ranks},{n_lat},{n_lon}\n")
            print(f"wrote {wt_out}")

    if rank == 0:
        def m_to_target(name):
            ms = [M for (nm, M, r) in rows if nm == name and r <= args.target]
            return min(ms) if ms else None
        mj = m_to_target("jacobi")
        mg = m_to_target("multigrid")
        print(f"\n=== M to reach rel_residual<={args.target:.0e} "
              f"(grid {n_lat}x{n_lon}, np{n_ranks}) ===")
        print(f"  jacobi:    M={mj}  reductions/step={2 * mj if mj else 'NA'}")
        print(f"  multigrid: M={mg}  reductions/step={2 * mg if mg else 'NA'}")
        if mj and mg:
            print(f"  M-cut: {mj}/{mg} = {mj / mg:.1f}x  "
                  f"reduction-count cut {2 * mj} -> {2 * mg}")
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write("preconditioner,M,rel_residual,reductions_per_step,n_ranks,n_lat,n_lon\n")
            for (nm, M, r) in rows:
                f.write(f"{nm},{M},{r:.6e},{2 * M},{n_ranks},{n_lat},{n_lon}\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Barotropic-PCG preconditioner convergence: M-to-tol + distributed cost.

The DECISIVE measurement for the last open scaling lever (codex remaining-
headroom audit 2026-06-14: CPU-multinode ocean Chebyshev M-cut). The fixed-M
distributed PCG's weak-scaling wall is its GLOBAL allreduce latency = 2*M
reductions/step. Chebyshev preconditioning cuts the OUTER iteration count M
(lower residual per iter, NO per-iter reduction) at the cost of `degree`
extra LOCAL matvec-halos per iter. On a latency-bound multinode regime
(allreduce >> halo) trading reductions for halos WINS — IF Chebyshev cuts M
enough.

This sweeps rel_residual vs M for jacobi and chebyshev(deg 2/4/8) on a
realistic stiff (pole+coastal, variable-depth) lat-lon Helmholtz (single
device, no MPI — the *algorithm* is identical with/without ranks), and
reports, per preconditioner, the M to reach a tolerance and the resulting
DISTRIBUTED cost:
  * reductions = 2*M (standard PCG; the multinode latency wall)
  * halos      = M*(1 + degree)  (cheaper neighbor exchange)
If chebyshev's M-to-tol gives 2*M_cheb << 2*M_jac, the multinode A/B is
worth a 2-node allocation; if not, the lever is dead (CPU-multinode ocean
weak is at its practical limit too).

Usage (compute node): JAX_ENABLE_X64=1 python scripts/bench/\
bench_barotropic_precond_convergence.py [--n-lat 90 --n-lon 180]
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-lat", type=int, default=90)
    ap.add_argument("--n-lon", type=int, default=180)
    ap.add_argument("--coeff", type=float, default=5.0e7)
    ap.add_argument("--degrees", default="2,4,8")
    ap.add_argument("--m-list", default="4,8,12,16,20,24,32,40,48,60")
    ap.add_argument("--tols", default="1e-4,1e-6,1e-8")
    args = ap.parse_args()

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm.grids.latlon import create_latlon_grid, ensure_geometry
    from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
        _make_helmholtz, _helmholtz_inv_diag, _make_chebyshev_preconditioner,
    )
    from legoesm.ocean.dynamics.barotropic_common import solve_helmholtz_implicit

    n_lat, n_lon = args.n_lat, args.n_lon
    grid = ensure_geometry(create_latlon_grid(n_lat, n_lon))
    rng = np.random.default_rng(13)
    # Stiff, realistic: variable depth + pole rows + meridional coast + basin.
    H = 1000.0 + 500.0 * rng.random((n_lat, n_lon))
    mask = np.ones((n_lat, n_lon))
    mask[0, :] = 0.0
    mask[-1, :] = 0.0
    mask[:, n_lon // 8:n_lon // 8 + 4] = 0.0
    mask[n_lat // 3:n_lat // 3 + 3, n_lon // 2:n_lon // 2 + 10] = 0.0
    H_u = np.zeros((n_lat, n_lon + 1))
    H_u[:, 1:-1] = 0.5 * (H[:, 1:] + H[:, :-1])
    H_u[:, 0] = H_u[:, -1] = 0.5 * (H[:, 0] + H[:, -1])
    u_wet = np.zeros_like(H_u)
    u_wet[:, 1:-1] = mask[:, 1:] * mask[:, :-1]
    u_wet[:, 0] = u_wet[:, -1] = mask[:, 0] * mask[:, -1]
    H_u = jnp.asarray(H_u * u_wet)
    H_v = np.zeros((n_lat + 1, n_lon))
    H_v[1:-1, :] = 0.5 * (H[1:, :] + H[:-1, :])
    v_wet = np.zeros_like(H_v)
    v_wet[1:-1, :] = mask[1:, :] * mask[:-1, :]
    H_v = jnp.asarray(H_v * v_wet)
    mask = jnp.asarray(mask)
    u_wet = jnp.asarray(u_wet)
    v_wet = jnp.asarray(v_wet)
    coeff = jnp.asarray(args.coeff)

    A_op = _make_helmholtz(H_u, H_v, coeff, grid, mask, u_wet, v_wet)
    inv_diag = _helmholtz_inv_diag(H_u, H_v, coeff, grid, mask)
    w = grid.area * mask
    rhs = jnp.asarray(rng.standard_normal((n_lat, n_lon))) * mask
    x0 = jnp.zeros_like(rhs)

    def jacobi(r):
        return r * inv_diag.astype(r.dtype)

    preconds = [("jacobi", 0, jacobi)]
    for d in (int(x) for x in args.degrees.split(",") if x.strip()):
        preconds.append((f"cheby{d}", d,
                         _make_chebyshev_preconditioner(A_op, inv_diag, mask, d)))

    m_list = [int(x) for x in args.m_list.split(",") if x.strip()]
    tols = [float(x) for x in args.tols.split(",") if x.strip()]

    print(f"[conv] grid={n_lat}x{n_lon} coeff={args.coeff:g} "
          f"wet={float(jnp.sum(mask)):.0f}/{n_lat*n_lon}", flush=True)
    # residual(M) table
    resid = {}   # (name) -> {M: rel_res}
    for (name, deg, M_inv) in preconds:
        resid[name] = {}
        for M in m_list:
            _, diag = solve_helmholtz_implicit(
                A_op, rhs, M_inv, x0, distributed=True, fixed_iters=M,
                residual_tol=1e-30, stock_cg_tol=1e-12, stock_cg_maxiter=200,
                pcg_variant="standard", dot_weight=w)
            resid[name][M] = float(diag.rel_residual)
        row = "  ".join(f"M{M}={resid[name][M]:.2e}" for M in m_list)
        print(f"[conv] {name:8s} deg={deg}: {row}", flush=True)

    # M-to-tol + distributed cost (reductions=2M; halos=M*(1+deg)).
    def m_to_tol(name, tol):
        for M in m_list:
            if resid[name][M] <= tol:
                return M
        return None

    print("\n[cost] M-to-tol + distributed cost (reductions=2M, halos=M*(1+deg)):",
          flush=True)
    for tol in tols:
        print(f"  tol={tol:g}:", flush=True)
        jac_M = m_to_tol("jacobi", tol)
        for (name, deg, _) in preconds:
            M = m_to_tol(name, tol)
            if M is None:
                print(f"    {name:8s}: not reached within M<={m_list[-1]}",
                      flush=True)
                continue
            red, hal = 2 * M, M * (1 + deg)
            tag = ""
            if name != "jacobi" and jac_M is not None:
                tag = (f"  reductions x{(2*jac_M)/red:.2f} vs jacobi "
                       f"(halos x{hal/(jac_M*1):.2f})")
            print(f"    {name:8s}: M={M:3d}  reductions={red:3d}  "
                  f"halos={hal:3d}{tag}", flush=True)
    print("CONV_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
packages/ocean/legoesm/ocean/dynamics/barotropic.py:200:    _M = "barotropic_solver"
packages/core/legoesm/core/physics_output.py:18:    """Output from a single physics step.
packages/core/legoesm/components/protocol.py:94:            f"dycore.step cannot be called as step(state, dt) "
packages/core/legoesm/runtime/precision.py:104:    # ``apply_precision("fp32")`` would leave barotropic_solver / EOS / PGF /
packages/ocean/legoesm/ocean/dynamics/ocean_pe_mpas.py:432:    # the partial-cell precondition):
packages/ocean/legoesm/ocean/dynamics/ocean_pe_mpas.py:965:        # ocean_model_mpas.step) -- without it an unbalanced ∮(P-E+R) drifts the
packages/ocean/legoesm/ocean/dynamics/ocean_pe_mpas.py:970:        # normalization (ocean_model_mpas.step) are rank-local area-means with NO
packages/core/legoesm/components/surface_component.py:1:"""Wrap a live surface model's step as a ``StepComponent`` brick (Stage B).
packages/core/legoesm/runtime/config.py:152:        # initialize() in a 1-process step and hang waiting for N-way
packages/core/legoesm/core/conservation.py:1084:    Unlike ``fix_mass_hydrostatic`` which anchors to the previous step,
packages/core/legoesm/timestepping/leapfrog_ab2.py:106:        ``(state^{n-1}, state^n)`` from the previous step.
packages/core/legoesm/timestepping/leapfrog_ab2.py:144:    previous step, cached to avoid recomputation. ``curr`` is the
packages/core/legoesm/parallel/latlon_mpi.py:276:        and the banded-multigrid preconditioner needs even-aligned
packages/core/legoesm/parallel/latlon_mpi.py:1517:    same bytes every step (census job 8459289: 14 metric pads/step, two
packages/core/legoesm/parallel/latlon_mpi.py:1519:    sendrecv pairs/step at M=60).  This helper performs the exchange
packages/core/legoesm/parallel/latlon_mpi.py:2427:        # ALL rows, with NO MPI collective — so the banded-MG preconditioner
packages/core/legoesm/parallel/latlon_mpi.py:2868:        # closure-captured so its identity is stable across steps) is
packages/core/legoesm/parallel/latlon_mpi.py:2973:    # Validate the EQUAL-split precondition of the polar-filter lon-gather
packages/core/legoesm/parallel/latlon_mpi.py:3059:        # for this step regardless of build/call interleaving AND does not
packages/core/legoesm/grids/conservative_regrid.py:193:    precondition itself: a caller that wrap-pads its source longitude with
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:199:# and barotropic_solver="rigid_lid" (rejected otherwise).
packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:2264:    # Full flux-form momentum update (Level 2) still requires step-
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:366:# Jacobi-preconditioned conjugate-gradient solve converges quickly.
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:376:# exactly ``max_iter`` iterations).  Standard preconditioned CG needs two
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:384:# resolution) reductions.  A one-reduction pipelined CG
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:505:def _fixed_iteration_pcg(
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:514:    """Jacobi-preconditioned CG run for EXACTLY ``max_iter`` iterations.
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:516:    Standard preconditioned conjugate gradient (Shewchuk 1994, Alg. B3)
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:638:def _fixed_iteration_pcg_single_reduce(
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:650:    :func:`_fixed_iteration_pcg`, but restructured so each iteration
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:658:    Recurrences (preconditioned CG-CG form; ``z = M⁻¹ r``, ``w = A z``):
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:681:    ``single_reduce`` without it rather than falling back to a wrong
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:696:    contract as :func:`_fixed_iteration_pcg`.
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:877:        Helmholtz operator and Jacobi preconditioner.  ``A_op`` owns the
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:917:    #   "single_reduce" — 1 batched reduction/iter (Chronopoulos-Gear);
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:922:        # halo-carrying partitioned meshes — see _fixed_iteration_pcg.
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:923:        eta_new, rr = _fixed_iteration_pcg(
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:927:    elif pcg_variant == "single_reduce":
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:933:                "solve_helmholtz_implicit: pcg_variant='single_reduce' "
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:937:        eta_new, rr = _fixed_iteration_pcg_single_reduce(
packages/ocean/legoesm/ocean/dynamics/barotropic_common.py:944:            f"{pcg_variant!r}; expected 'standard' or 'single_reduce'."
packages/core/legoesm/core/_future/vertical_remap.py:5:production code path.  Wire it into the PE model's step() method
packages/core/legoesm/parallel/scaling_diagnostics.py:54:    """Accumulates wall-clock time per named phase across steps.
packages/core/legoesm/parallel/scaling_diagnostics.py:149:    """Track JAX device memory usage across steps."""
packages/core/legoesm/parallel/scaling_diagnostics.py:861:            f"component of this step; refusing to report an overlap "
packages/core/legoesm/core/precision.py:223:        Module name (e.g., "pressure_gradient", "barotropic_solver").
packages/core/legoesm/core/precision.py:533:    "barotropic_solver": {"compute": jnp.float64, "control": jnp.float64},
packages/core/legoesm/parallel/halo_exchange.py:53:   exchanged in sequence (not pipelined).  Each sendrecv blocks until
packages/ocean/legoesm/ocean/dynamics/ocean_tendency_common.py:414:    the previous step ``F^{n−1}``.  Open-coded identically across the
packages/ocean/legoesm/ocean/dynamics/ocean_tendency_common.py:429:        Tendency at the previous step ``F^{n−1}``.
packages/core/legoesm/parallel/halo_exchange_voronoi.py:34:   The historical ~18x CPU regression (435.9 vs 24.1 ms/step, I5 np8 f32,
packages/core/legoesm/parallel/halo_exchange_voronoi.py:167:    once removes both costs.  Measured comm overhead was 22 ms/step at
packages/core/legoesm/parallel/tiled_production_cdgrid.py:1736:        # Mirrors serial fv3_hydrostatic_tendencies step 13 exactly:
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2243:# STAY tile-sharded across steps: this section adds
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2264:# Cross-step self-consistency of the duplicated shared faces follows from
packages/core/legoesm/parallel/column_shard.py:26:using.  Only the per-column physics step is wrapped — radiation
packages/core/legoesm/parallel/column_shard.py:108:    per-column physics step run on arbitrary ``(ncol, n_devices)``
packages/core/legoesm/parallel/sharded_dynamics.py:358:# Executable cache — compile once, reuse across steps
packages/core/legoesm/parallel/sharded_dynamics.py:533:        """Execute one sharded dynamics step.
packages/core/legoesm/parallel/sharded_dynamics.py:662:    """Wrap a dynamics model's step function for multi-device execution.
packages/core/legoesm/parallel/sharded_dynamics.py:689:        after each dynamics step.  If ``None``, the model's built-in
packages/core/legoesm/parallel/sharded_dynamics.py:799:    """Execute one dynamics step with explicit halo exchange.
packages/core/legoesm/parallel/sharded_dynamics.py:807:    1. Run the dynamics step on each device's partition.
packages/core/legoesm/parallel/sharded_dynamics.py:894:        This ensures boundary values are fresh after a dynamics step.
packages/core/legoesm/parallel/sharded_dynamics.py:1798:# RK pytree_axpy (pytree_ops.py), ~10.9 ms/step in total (nsys 26479922,
packages/core/legoesm/parallel/sharded_dynamics.py:2009:        STRUCTURE must stay stable across steps.  On a single-device
packages/ocean/legoesm/ocean/dynamics/pgf_ahh08.py:60:the rest-state PGF residual under "centered" is ~1.3e-4 m/s/step on
packages/ocean/legoesm/ocean/dynamics/barotropic_cgrid.py:167:    _M = "barotropic_solver"
packages/ocean/legoesm/ocean/dynamics/barotropic_cgrid.py:413:    _M = "barotropic_solver"
packages/ocean/legoesm/ocean/dynamics/barotropic_cgrid.py:545:    _M = "barotropic_solver"
packages/core/legoesm/timestepping/integration.py:46:    carries state across steps; calling a per-step ``step`` /
packages/core/legoesm/timestepping/integration.py:80:    If the model also defines step_with_physics(state, dt, physics_fn),
packages/core/legoesm/timestepping/integration.py:180:    # previous step, preventing slow drift).  Subclasses initialise
packages/ocean/legoesm/ocean/dynamics/ocean_model.py:131:        # SW model's step is jitted on a static ``self``).  See
packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:731:    # (large, un-jitted) ocean band step EVERY call (~80 s/step at 16x32x3 nd=4
packages/core/legoesm/timestepping/semi_implicit.py:686:        State at time n-1 (already filtered from previous step).
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1224:        if self.config.barotropic.barotropic_solver == "rigid_lid":
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1356:            _bsolver = config.barotropic.barotropic_solver
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1360:                    f"barotropic_solver={_bsolver!r}: the equilibrium-tide body "
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1362:                    f"Use barotropic_solver='explicit_substep' (the default).")
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1404:                and config.barotropic.barotropic_solver != "explicit_substep"):
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1407:                "barotropic_solver='explicit_substep' (the wide-halo path "
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1409:                f"{config.barotropic.barotropic_solver!r}",
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1647:        if config.barotropic.barotropic_solver not in _valid_solvers:
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1649:                f"barotropic_solver must be one of {_valid_solvers}, "
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1650:                f"got {config.barotropic.barotropic_solver!r}")
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1651:        if config.barotropic.barotropic_solver == "implicit_unsplit":
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1691:                    'barotropic_solver="implicit_unsplit" does not yet support: '
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1696:                    'barotropic_solver="implicit_cn", or extend _unsplit_ab2_step.')
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1703:                and config.barotropic.barotropic_solver != "explicit_substep"):
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1715:                'barotropic_solver="explicit_substep" (NEMO stpmlf order) it '
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1773:            # barotropic_solver="explicit_substep", _bc_bottom_drag is ALSO
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1782:            if (config.barotropic.barotropic_solver == "explicit_substep"
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1787:                    'barotropic_solver="explicit_substep" requires '
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1796:                    "composition), use a different barotropic_solver, or "
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1837:            if config.barotropic.barotropic_solver != "explicit_substep":
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1841:                    "under barotropic_solver="
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1842:                    f"{config.barotropic.barotropic_solver!r} it would be a "
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1845:                    'barotropic_solver="explicit_substep" or leave the flag '
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1854:                and config.barotropic.barotropic_solver != "explicit_substep"):
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1858:                f"has NO effect under barotropic_solver="
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1859:                f"{config.barotropic.barotropic_solver!r}: the time filter is consumed "
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1861:                'barotropic_solver="explicit_substep" to apply it, or leave the filter '
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1970:        if config.dt_mom_ratio != 1.0 and config.barotropic.barotropic_solver != "rigid_lid":
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1973:                "requires barotropic_solver='rigid_lid': under the rigid lid the "
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1977:                "O((dt_tracer-dt_mom)·∂h/∂t) tracer mass. Got barotropic_solver="
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1978:                f"{config.barotropic.barotropic_solver!r}.")
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2088:            if config.barotropic.barotropic_solver != "explicit_substep":
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2092:                    f"{config.barotropic.barotropic_solver!r}.")
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2231:            if config.barotropic.barotropic_solver not in (
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2236:                    'barotropic_solver in ("rigid_lid","implicit_cn",'
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2250:                    f"barotropic_solver={config.barotropic.barotropic_solver!r}.")
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2329:                and config.barotropic.barotropic_solver == "implicit_unsplit"):
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2331:                'prescribed_flow is not supported with barotropic_solver='
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2696:        # Validation requires barotropic_solver="rigid_lid" when ratio != 1.0 (fixed
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:2706:        # ONLY on eta/T/S, which are FROZEN across this step's stage-1 tendency
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:3131:        if self.config.barotropic.barotropic_solver == "rigid_lid":
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:3151:        elif self.config.barotropic.barotropic_solver == "implicit_cn":
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:4257:                # update + K_diss_bot from this step's tendency), run ONE
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:4733:          already carries this step's ``eke_diss`` (matching Veros's same-step
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:4738:        - ``K_diss_bot``: this step's bottom-drag KE extraction, surfaced as
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:5163:        its step (Veros tke.py:137 ``dt_tke = dt_mom``). When ``return_tke`` is
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:5170:        the Veros step order): the fallback K-profile computation returns the
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:5346:        # used by every other operator in this step).
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:5566:        # and under barotropic_solver="explicit_substep" it additionally
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:5872:        if self.config.barotropic.barotropic_solver == "rigid_lid":
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:5930:        if self.config.barotropic.barotropic_solver == "implicit_unsplit":
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:5939:                    "barotropic_solver='implicit_unsplit' requires "
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6112:        # the held values step after step); the pinned flow needs no polar
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6299:        fixed — under ``barotropic_solver="rigid_lid"`` (Veros's streamfunction rigid
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6465:        # assembled by _step_impl from this step's EKE update + K_diss_bot);
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6695:             term is UNCONDITIONALLY UNSTABLE. This step now drives ``_step_impl``
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7091:        Opt-in via ``barotropic_solver='implicit_unsplit'`` (requires
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7100:                'barotropic_solver="implicit_unsplit" does not yet support the '
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7102:                'flux); use barotropic_solver="implicit_cn".')
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7422:        # the scan traces step() with tracers (codex round-2 MINOR).
packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7625:        if self.config.barotropic.barotropic_solver == "rigid_lid":
packages/core/legoesm/parallel/async_halo.py:862:    """Execute a dynamics step with split interior/boundary computation
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:5:``barotropic_solver = "rigid_lid"`` option, for apples-to-apples fidelity with
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:138:            "barotropic_solver='rigid_lid' is single-rank only: the global "
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:145:            "barotropic_solver='implicit_cn' (distributed fixed-iteration PCG) "
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:168:    inv_diag : (n_lat+1, n_lon+1) Jacobi preconditioner 1/diag(S) for the
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:177:        pinned rows with the Laplacian rows, so the Jacobi preconditioner is also
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:269:    """Solve L(dψ) = rhs on the wet interior (ψ=0 on land), via preconditioned CG.
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:274:    a Jacobi preconditioner (``rl_data.inv_diag = 1/diag(S)``) is used instead of
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:327:    the SAME preconditioned CG on the seam-reduced operator, and zero-pad the
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:352:    def precond(r):
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:355:    # Seam-REDUCED operator/preconditioner for the adjoint solve: S_r = P·S·E
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:361:    def precond_reduced(r):
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:372:            M=precond,
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:391:            tol=tol, atol=0.0, maxiter=maxiter, M=precond_reduced,
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:408:    """1/diag(S) on solve vertices (1 on pinned), the CG Jacobi preconditioner.
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:415:    preconditioner, so the slight periodic-seam approximation when ``n_lon`` is
packages/ocean/legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py:590:    # model's step / rigid-lid-data-build entry points; this in-body call covers
packages/core/legoesm/parallel/voronoi_mpi.py:93:#     (job 8488023).  The historical 18x regression (435.9 vs 24.1 ms/step,
packages/core/legoesm/parallel/voronoi_mpi.py:739:       through this step propagates halo cotangents back to the owning
packages/core/legoesm/parallel/voronoi_mpi.py:1081:            # producer's 8-slot contract (primitive_eq_mpas.step) EXACTLY so the
packages/core/legoesm/parallel/voronoi_mpi.py:1211:        # operator-split physics carry (TKE / convection state) across steps.
packages/core/legoesm/parallel/voronoi_mpi.py:1229:            # ``primitive_eq_mpas.step()`` (a held-radiation step refreshes precip
packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py:161:        if self.config.barotropic_solver not in _valid_solvers:
packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py:163:                f"barotropic_solver must be one of {_valid_solvers}, "
packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py:164:                f"got {self.config.barotropic_solver!r}"
packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py:176:                and self.config.barotropic_solver != "explicit_substep"):
packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py:180:                f"NO effect under barotropic_solver={self.config.barotropic_solver!r}"
packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py:182:                'barotropic substep. Use barotropic_solver="explicit_substep" to '
packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py:590:        # the physics-stepped tracer, before advection).  Mirrors the
packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py:770:        if config.barotropic_solver == "implicit_cn":
packages/ocean/legoesm/ocean/dynamics/ocean_model_mpas.py:804:                    "barotropic_solver='implicit_cn' under multi-rank MPAS "
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:269:    ``-coeff·zonal_W``.  Serves the ZONAL-LINE preconditioner;
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:274:    consistency test (``test_zonal_line_preconditioner.py``): these
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:364:    """Inverse diagonal of the Helmholtz operator (Jacobi preconditioner).
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:392:        # Tripolar: use full per-face 2D metrics so the preconditioner
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:395:        # an unrepresentative preconditioner that makes PCG diverge.
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:452:def _make_diag_preconditioner(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:459:    """Return ``M(r) = diag⁻¹ · r`` Jacobi preconditioner for the Helmholtz."""
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:468:def _make_zonal_line_preconditioner(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:475:    """Zonal-line preconditioner: exact periodic-tridiagonal row solves.
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:486:      preconditioner it adds ZERO collectives to the PCG iteration
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:487:      (the EVP-block-preconditioner principle, CESM POP GMD 9:4209:
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:496:      requirement the single_reduce (Chronopoulos–Gear) recurrences
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:497:      impose on the preconditioner (the CG-CG inner-product lesson).
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:498:      Verified numerically in ``test_zonal_line_preconditioner.py``.
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:508:            "zonal_line preconditioner needs n_lon >= 3 (periodic "
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:510:            "preconditioner='jacobi' on degenerate-longitude grids."
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:539:def _make_chebyshev_preconditioner(A_op, inv_diag, mask, degree: int):
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:540:    """Degree-``degree`` Chebyshev-polynomial preconditioner:
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:549:    halo) is the right direction.  Cheaper than multigrid (~80 LOC, no
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:551:    smoother MPAS-O/CESM use inside multigrid anyway.
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:563:    itself W-self-adjoint, so M⁻¹ composes with the single_reduce
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:565:    ``test_chebyshev_preconditioner.py``.  AD-safe: pure matvec + scalar
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:567:    gradient; the eigenvalue bound is stop_gradient'd, as a preconditioner
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:576:            f"chebyshev preconditioner needs degree >= 1, got {degree}.")
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:590:    # preconditioner tuning parameter, not part of the converged answer).
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:599:                "chebyshev preconditioner: halo backend is 'spmd' but no SPMD "
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:605:    # stop_gradient: the preconditioner's spectral window is a tuning
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:652:# Geometric multigrid preconditioner (anisotropic: zonal-line smoother)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:660:# reduction-latency wall (~120 -> ~8 global allreduces/step under MPI).  The
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:748:    """2x-coarsened ``LatLonBandLayout`` for the banded multigrid, or ``None``
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:792:    (zonal-line-smoother-only) preconditioner from L0 alone.  For EQUAL bands
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:811:def _make_multigrid_preconditioner(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:823:    """Anisotropic geometric-multigrid V-cycle preconditioner (POC 8487762).
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:828:    (:func:`_make_zonal_line_preconditioner`).  The returned ``M_inv`` runs one
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:860:            "multigrid preconditioner: the geometric MG transfers are not yet "
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:864:            "'chebyshev' under MPI; 'multigrid' is single-rank only for now.")
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:867:            "multigrid preconditioner: coarse-grid construction "
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:870:            "or 'chebyshev' on tripolar grids.")
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:880:        smoother = _make_zonal_line_preconditioner(H_u, H_v, coeff, g, m)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:894:    return _run_vcycle_preconditioner(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:899:def _run_vcycle_preconditioner(levels, mask, *, pre, post, coarse_sweeps,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:939:def _make_multigrid_preconditioner_banded(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:954:    The halo-aware sibling of :func:`_make_multigrid_preconditioner`: builds the
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:980:            "multigrid_banded: coarse-grid construction does not handle the "
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:981:            "tripolar north-fold; use 'zonal_line' or 'chebyshev'.")
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:998:                "multigrid_banded: requires EQUAL bands (the uniform even "
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1027:        smoother = _make_zonal_line_preconditioner(H_u, H_v, coeff, g, m)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1035:    return _run_vcycle_preconditioner(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1040:def _select_preconditioner(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1053:    """Dispatch the implicit-CN PCG preconditioner by config name.
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1059:    :func:`_make_zonal_line_preconditioner`).
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1060:    "chebyshev" — degree-``cheby_degree`` Chebyshev polynomial of the
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1062:    reduction; see :func:`_make_chebyshev_preconditioner`).
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1063:    "multigrid" — anisotropic geometric V-cycle with a zonal-line smoother
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1066:    :func:`_make_multigrid_preconditioner` serially and the BANDED
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1067:    :func:`_make_multigrid_preconditioner_banded` under MPI (``layout`` = the
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1077:        return _make_zonal_line_preconditioner(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1080:    if name == "chebyshev":
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1083:                "barotropic_implicit_latlon_cgrid: chebyshev preconditioner "
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1085:        return _make_chebyshev_preconditioner(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1088:    if name == "multigrid":
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1091:                "barotropic_implicit_latlon_cgrid: multigrid preconditioner "
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1099:                    "barotropic_implicit_latlon_cgrid: 'multigrid' under MPI "
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1103:            return _make_multigrid_preconditioner_banded(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1105:        return _make_multigrid_preconditioner(H_cell, coeff, grid, mask)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1108:        f"barotropic_implicit_preconditioner {name!r}; expected "
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1109:        "'jacobi', 'zonal_line', 'chebyshev', or 'multigrid'."
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1131:    call on ``A = _make_helmholtz(...)`` with the Jacobi preconditioner
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1161:    norms), reusing the SAME preconditioned CG (A is what CG likes; only
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1171:    the converged solution — IFT).  ``inv_diag`` (preconditioner) gets a
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1278:            jnp.zeros_like(inv_diag_in),  # preconditioner: zero (IFT)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1469:    # Jacobi preconditioner diagonal + Helmholtz operator.  ``A_op``
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1487:    # Cell-centred water-column depth for the multigrid coarsening (the same
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1489:    # for the other preconditioners (they ignore H_cell).
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1491:    # Band layout for the distributed 'multigrid' path (None serially / for the
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1492:    # other preconditioners, which ignore it).  get_mpi_topology() returns the
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1499:    M_inv = _select_preconditioner(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1500:        str(getattr(config.barotropic, "barotropic_implicit_preconditioner",
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1504:        cheby_degree=int(config.barotropic.barotropic_chebyshev_degree),
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1515:    #   Forward = the stock preconditioned ``jax.scipy`` CG VERBATIM
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1566:        # non-default preconditioner cannot take effect here.  Refuse
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1570:        _precond_req = str(getattr(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1571:            config.barotropic, "barotropic_implicit_preconditioner", "jacobi"))
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1572:        if _precond_req != "jacobi":
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1575:                f"barotropic_implicit_preconditioner={_precond_req!r} "
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1581:                "preconditioner='jacobi'."
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1610:            # W-inner-product weight for the single_reduce recurrences
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:56:PCG converges rapidly with a Jacobi preconditioner.
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:140:    ``c² = g·H_min`` is the correct gravity-wave speed across step
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:209:    """Inverse diagonal of the Voronoi Helmholtz (Jacobi preconditioner).
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:235:def _make_diag_preconditioner(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:242:    """Return ``M(r) = diag⁻¹ · r`` Jacobi preconditioner (see
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:366:            jnp.zeros_like(inv_diag_in),  # preconditioner: zero (IFT)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:440:            "MPAS barotropic_solver='implicit_cn' under MPI requires the "
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:566:    # forward = the stock preconditioned CG VERBATIM (``jax.custom_vjp``
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:587:        # by the single_reduce recurrences, harmless for standard).
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:638:        # Forward = stock preconditioned CG, bit-identical; reverse mode

exec
/usr/bin/bash -lc "sed -n '1,300p' docs/performance/scaling/scaling_levers_audit_2026-06-15.md && sed -n '1,300p' docs/performance/scaling/distance_to_limit_2026-06-13.md && sed -n '1,300p' docs/performance/scaling/literature_scan_2026-06-13_new_levers.md && sed -n '1,280p' docs/performance/scaling/spectral_level_shard_cliff.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 365ms:
# Scaling missed-opportunity audit — 2026-06-15 (codex, read-only)

Re-rank after the **banded distributed multigrid barotropic preconditioner**
(task #26) shipped + was measured (jacobi M=60 = 120 reductions/step and
unconverged vs banded-MG M=12 = 24 reductions/step to 1e-6; V-cycle adds zero
global reductions; banded==serial under mpirun -np 2/4). Supersedes the ranked
list in `scaling_levers_audit_2026-06-14.md`.

Hardware: Ginsburg — CPU-MPI nodes (≤16 ranks/node policy, Gloo/TCP, no IB) +
2× RTX8000 (PCIe, no NVLink).

## Highest ROI next
1. **MPAS batched-halo repair** (ocean+atm) — **DONE 2026-06-15.** The "18×
   regression" was I5/f32-stale; current code is FASTER batched at every size
   (I4/I5 f32 -12%/-5% job 8488057; I6 f64 np8/np16 -4.6%/-3.5% job 8488023).
   Flipped `_USE_BATCHED_HALO` default 0→1 (batched now production default,
   legacy opt-OUT). Gate 8488077 OK + codex-clean. ~4-12% on every MPAS/voronoi
   run.
2. **MPAS/Voronoi indirect-gather fusion** (ocean+atm GPU) — **MEASURED DEAD on
   this HW 2026-06-15** (job 8488104). The go/no-go was a command-buffer A/B at
   np1 (isolates per-device kernels): aggressive `FUSION,CUSTOM_CALL,CUBLAS,
   CUDNN,COLLECTIVES` == default `FUSION,CUSTOM_CALL,COLLECTIVES` to noise (f32
   11.03 vs 11.07 ms/step) ⇒ atm-ico is NOT dispatch-bound (already
   well-fused; 96.5 Mcells/s ≈ per-device limit). Fusing the gathers (Pallas/
   restructure) would not cut a dispatch wall that isn't there; the gathers are
   bandwidth-bound, where fewer kernels don't reduce traffic. Skip on Ginsburg
   (would need a much faster GPU to expose dispatch headroom).

## Ranked remaining
3. METIS/default partition audit for MPAS — **MEASURED 2026-06-15 (job 8488136),
   small opt-in win at scale.** icosahedral I6 f64 RCB-vs-METIS: np8 neutral
   (66.68 vs 67.32 ms/step, noise), np16 **METIS +1.9%** (59.93 vs 58.82) — the
   edge-cut-min win GROWS with rank (boundary/interior ratio), so >np16 likely
   approaches the +5-20% range. RCB is already well-balanced on the near-uniform
   icosahedral mesh, so the low-rank win is ~0; the gain is high-rank only. Kept
   **UPDATE 2026-06-21 — flipped to a capability-gated `method="auto"` default**
   (branch `perf/voronoi-graph-partition-sfc`): `auto` selects METIS when
   `pymetis` is importable, else falls back to geometric RCB. The win is now
   realized automatically wherever pymetis is present (the `[mesh]` extra) with NO
   hard dependency and byte-identical behavior where it is absent
   (dev/CI/shared venv), so the earlier "not worth a hard dep" objection no longer
   applies. Also added a dependency-free `method="sfc"` (Hilbert space-filling
   curve) for locality-preserving contiguous partitions, and
   `reorder_voronoi_for_sharding` now Hilbert-orders cells within each shard.
   Still install `[mesh]` for large multi-node MPAS where the high-rank edge-cut
   win (>np16, trending +5–20%) matters.
4. Root-only/gathered checkpoint + diagnostics writers — `scripts/run/run_omip.py`
   / matrix I/O; big wall-clock only at high output cadence, ~0 step-kernel gain.
5. Tripole wiring (lat-lon C-grid) — capability (eORCA scaling), not speedup.

## MG agglomeration — NOT worth at ≤8 ranks
One allreduce/V-cycle ⇒ break-even ≈ `2*(M_band - M_agglom) > M_agglom`. Our np4
`M=20` vs agglomerated `~12` is only marginal before coarse-solve overhead; np2
worse. Do it ONLY for many-rank runs where the even-alignment depth cap drives
clear M-growth.

## Missed lever — CUDA-graph ATM icosahedral: MEASURED, ~DEAD
- **CUDA-graph A/B for ATM icosahedral only** (env/bench, not model code).
  Codex audit 2026-06-15 named this as the ONE untried measurable step-kernel
  lever. Measured:
  - job 8488104 (A40, I6 np1): default-cmdbuf (FUSION,CUSTOM_CALL,COLLECTIVES)
    vs +CUBLAS,CUDNN = **noise** — f32 11.07→11.03 ms/step, f64 20.50→20.42.
    Adding CUBLAS/CUDNN command buffers does nothing.
  - job 8490224 (RTX8000, I6 np1): the missing **OFF arm** (command buffers
    fully DISABLED, `--xla_gpu_enable_command_buffer=`) vs default vs aggressive.
    **f32 RESULT: OFF=11.25, default=11.24, aggressive=11.23 ms/step — ALL EQUAL
    (noise).** Command buffers give ZERO benefit ⇒ zero dispatch headroom ⇒
    MPAS-atm GPU step is bandwidth/compute-bound, CUDA-graph lever **DEAD** on
    this HW, at-limit confirmed (f64 arms confirmatory). (Unlike MPAS-ocean fp32
    where CUDA graphs gave 2.4–2.9×; cubed-sphere measured negative earlier.)

## Codex at-limit verdict (2026-06-15)
Focused codex audit ("name a MEASURABLE Ginsburg lever NOT yet tried"): **mostly
at practical limit.** Only one honest untried step-kernel A/B (CUDA-graph ATM,
above — now being closed); the rest are workflow/I/O (root-only checkpoint = 0%
step ms) or narrow A/B extensions (MPAS METIS at np32, expected 2–5%, "not a new
lever"). Everything else = harvested, measured-dead, or PCIe/Gloo/TCP-blocked.

## HW-blocked — stop pursuing on Ginsburg
- Cube np>6 (Gloo/TCP + PCIe anti-scale) — future-HW capability only.
- 2-D lat-lon decomposition speedups (Gloo latency + pole/full-lon transpose).
- CUDA-aware MPI rebuild (stack-blocked; 2-GPU full step already near-ideal).
- Comm/compute overlap via nonblocking MPI (blocking mpi4jax/XLA schedule).
- Spectral multi-GPU global transform on PCIe (all-to-all dominated; GEMM is the
  right local lever).
# Distance-to-limit assessment — legoESM scaling campaign (2026-06-13)

Honest per-axis "how close to the theoretical limit" for weak/strong ×
MPI/GPU × atm/ocean, anchored to measured rooflines (roofline_probe.py,
cross-node collective clock 8473330) and phase splits. Every number has a
job receipt; "AT LIMIT" means the gap is hardware or algorithm-fundamental,
not engineering debt.

## Atmosphere

| Axis | State | Limit verdict |
|---|---|---|
| Cubed-sphere per-device | 7.2e-9 s/cell/step (JAX-Fluids-class) | **AT JAX-peer limit** — per-device XLA-fusion closed; CS dycore not launch-bound (command-buffer A/B null) |
| Cubed-sphere 2-GPU strong | eff 0.73 (f64) | **AT PCIe-link roofline** — ppermute 70-245× < HBM (measured); headroom needs NVLink/IB (absent on Ginsburg) |
| Cubed-sphere multinode SPMD | C96 np6 2.42× (1 proc/node) | bandwidth-class, scales with size; PRODUCTION path shipped (`distributed_mode='spmd'`); halo-count lever (F1 packing shipped) |
| Lat-lon CPU-MPI | comm 15→6 exch/step | 1-D band; 2-D decomposition MEASURED FABRIC-BLOCKED (job 8477039): balanced 2-D 1.58-1.73× SLOWER than 1-D band at np16/32 (latency-bound — 2-D adds the E/W collective direction; size cuts don't help; 1-D band not exhausted until np>n_lat=256). Foundation done + AD-safe (task #7); future-NVLink/IB / np≫n_lat capability — **NOT a near-term lever** |
| Icosahedral/MPAS 2-GPU | eff 0.88 | at link limit; distributed PCG shipped |
| Spectral | single-device | **by design** (global SH transform = cuBLAS-bound matmul); multi-rank N/A documented |

## Ocean (lat-lon C-grid, production tile rows/rank=48, phase-split job 8475898)

Production-tile phase split (np4 LL192): implicit_vmix 44.9%, baroclinic
16.7%, barotropic 10.1%, tracer 8.4%, residual 16.9%.

| Phase | %step | Rank-behavior | Limit verdict |
|---|---|---|---|
| implicit_vmix | 38-61% | shards ~perfectly (0 comm np1) | per-device THROUGHPUT cost, NOT a scaling limiter; f32 storage = 1.7× (opt-in; f64 = scientific choice); easy compute wins shipped (T+S shared-factor +15%); remaining = memory-traffic refactor (fuse K-profile into Thomas) — task #12, bounded by being shard-friendly |
| baroclinic | 16-21% | 27 halos = #1 rank-GROWING | **THE weak-scaling lever** — halo fusion (vertex-mask hoist shipped 39→37; neumann_fill_cgrid 12-exch cluster = next, same static-mask-hoist pattern) |
| barotropic | 9-10% | 120 reductions | reduction wall NEUTRALIZED (zonal_line / single_reduce, opt-in; regime-crossover job 8475875: +20-26% small tiles, neutral production tile because the phase is only ~6-10%) — **ceiling bounded by phase fraction** |
| tracer_advection | 8% | 4 halos | minor; tracer-pair restructure measured non-win |

Weak np16→32 production tile: ×1.6-1.8 (cross-node Gloo-TCP latency-bound;
≤8 ranks/node policy — DRAM contention above that, job 8474286).

## Where the real remaining headroom is (ranked)

1. **Ocean baroclinic halo fusion** (weak-scaling): 27 halos → fewer via the
   static-mask-sequence hoist (vertex-mask done; neumann_fill_cgrid 12-exch
   next). Measurable at production tile; bounded by baroclinic's 16-21%.
2. **Ocean vmix memory-traffic** (throughput, not scaling): 38-61% of step,
   shards fine; fuse K-profile build into the solve, cut intermediates.
3. **Cube tiled np>6 stage** (task #3): the one lever that unlocks genuinely
   NEW scaling (cube atm beyond 6 faces / 6 ranks). d2a2c operator ported
   (interior + all edge helpers committed); remaining = edge assembly →
   transport → shard_map stage → np24 bench. DELIBERATE multi-week grind,
   no perf payoff until the whole stage assembles; gate+codex each step.
4. **Cube cross-stage halo fusion** (task #11): 125 python pads → 54 wire
   ppermutes already done; remaining cuts are CROSS-STAGE (independent
   same-shape pads at different pipeline points), needing per-RK-stage
   dataflow restructuring — deliberate, risky to numerics order. Count-cut
   = the fabric-rewarded direction, but bounded by the cross-node 54-halo
   wall fraction.
5. **Multigrid/RAS barotropic** (task #14, lit-derived): resolution-
   independent iteration count (60→~6-10). HIGH-rank-count lever — barotropic
   is only ~7% of the prod step (1.9× was a 16k-core regime); bounded
   near-term ROI; from-scratch multi-day/risky. Decision, not autostart.
6. **Multi-node GPU SPMD / 2-D decomposition**: architecture/foundation
   proven but interconnect-capped on Ginsburg PCIe (ppermute < SYS/PCIe <
   network); 2-D measured fabric-blocked. **Defer to NVLink/IB hardware** —
   not engineering gaps here.

## Bottom line (updated 2026-06-13: ALL bounded levers harvested/blocked)

The high-ROI axes are AT their limits: atm per-device (JAX-peer), atm GPU
strong (PCIe roofline), ocean barotropic reductions (neutralized), ocean
vmix (memory-bandwidth floor). The last "open" structural lever — the
lat-lon 2-D decomposition — is now MEASURED fabric-blocked at ≤32 ranks
(this session). **We are demonstrably close to the practical limits on
Ginsburg's hardware/scale.** Every remaining lever is either (a) a deep
multi-week refactor whose payoff is bounded by a phase's step-fraction
(cube cross-stage fusion, baroclinic neumann fusion) OR unlocks new scale
only via a deliberate operator grind (cube tiled np>6 stage), or (b)
hardware-blocked (no NVLink/IB → multinode GPU + 2-D decomposition) or
high-rank-only (multigrid barotropic). None is a quick autonomous-loop
win; each is an explicit decision (deliberate session or hardware upgrade).
The verify-first method this campaign paid off repeatedly — it killed
fake speedups, the single-reduce overclaim, and (this session) the 2-D
decomposition before a multi-week step build.

---

## 2026-06-13 PM UPDATE — after +16 codex-clean units (review levers + cube tiling)

The levers this doc listed as "remaining/next" are now SHIPPED (PR #435),
which TIGHTENS every at-limit verdict (the engineering debt is paid; the
remaining gaps are hardware/algorithm-fundamental or future-hardware).

**Shipped this campaign extension (all gate+codex-verified):**
- Ocean barotropic weak-scaling: **Chebyshev preconditioner** (0293f865) —
  ≥2.3× fewer PCG global-reduction rounds (deg8: 35 iters to 1e-6 vs jacobi
  >80). The iteration-COUNT axis (the open one) now addressed via the cheap
  reduction-free mechanism. Opt-in.
- Ocean per-device vmix: **LAPACK gtsv FFI** (7d153934) — 8.4–13.1× CPU
  vmix-solve vs the fori-loop legacy (C96-scale 4143→317ms). Opt-in
  `LEGOESM_TRIDIAG=lapack`.
- Ocean per-device baroclinic: **GM/Redi density+Jacobian hoist** (fb0116d6)
  — removes the redundant 2-iter EOS coupling/step in the implicit_K33 path.
- Atm per-device spectral: **batched-GEMM SH transform** (74f02cb4, opt-in)
  — moves the Legendre step onto tensor-core dot_general (the one
  compute-bound atm grid). Future-hardware (fp64 tensor cores; Turing/CPU
  neutral) — correctness-validated here.
- Correctness: **TF32-on-Turing gate** (cbd2796c) — only enable TF32 where
  the hardware supports it (compute capability >= 8.0).
- Tooling: **ledger-header bugfix** (c6a1f76b) — the scaling-indicator plot
  was silently dropping all rows.
- Atm cube np>6 (the one structural lever breaking the 6-rank ceiling):
  **the ENTIRE FV3 transport+KE stack now tiles to sub-faces** — d2a2c
  D→A→C (prior) + PPM transport (U3b 2-D sweeps) + real-Courant composition
  both sweeps + np24 shard_map wiring (U3d) + B-grid corner Courant (U3c) +
  corner geo-conversion (U3e) + the full d_sw3 _bgrid_ke_transport
  (U3f capstone, bit-exact 1e-10 vs global).

**Per-grid theoretical-limit verdict (updated):**
- **Atm cube per-device / 2-GPU strong / spectral / latlon CPU-MPI**: AT
  LIMIT (unchanged — hardware/algorithm-fundamental on Ginsburg). SH→GEMM
  moves the spectral per-device roofline on Ampere+ (future).
- **Atm cube np>6**: the transport+KE STACK is tiled (correctness-complete);
  the np24 PERF payoff is NOT Ginsburg-benchable (no 24 real devices; CPU-
  virtual oversubscribes) — it is a multi-GPU-node capability. Remaining to
  a full RK tendency: d_sw1 mass transport, d_sw5 div-damp + vorticity
  transport, d_sw6 KE-grad + wind-update + vort-damp (each tileable via the
  same approach-C; DELIBERATE-deep, not loop-micro-turns; no near-term SYPD).
- **Ocean barotropic / vmix / baroclinic / tracer**: AT LIMIT — every named
  lever shipped; the residual gaps are phase-fraction-bounded or
  hardware-bandwidth floors.

**BOTTOM LINE: on Ginsburg hardware, legoESM is AT the practical theoretical
limit on every MEASURABLE weak/strong × atm/ocean axis** — all implementable
levers are shipped. The only remaining structural lever (cube np>6) has its
transport+KE stack tiled and is gated on real multi-GPU-node hardware to
bench; the rest (full RK-stage assembly, bf16 transport, SH→GEMM perf) is
future-hardware or deliberate-deep capability work, not near-term Ginsburg
SYPD.
# 2025-2026 literature scan — NEW levers beyond the shipped set (2026-06-13)

Loop "when not making progress, look at recent literature" pass, AFTER the
+17-unit campaign extension (review levers + cube transport+KE tiling). Goal:
find scaling levers NOT already shipped + transferable to Ginsburg
(RTX8000 Turing sm_75, 2-GPU PCIe no-NVLink, CPU <=32 ranks Gloo-TCP).

## Already done (agent flagged, but we ALREADY have them) — do NOT re-do
- **XLA latency-hiding scheduler** (#3): `xla_gpu_enable_latency_hiding_
  scheduler=true` is ALREADY in runtime/backend.py NVIDIA_GPU_XLA_FLAGS
  (line 192) + `xla_gpu_enable_highest_priority_async_stream=true`. The
  agent's "OFF by default" is the XLA default; our backend overrides it.
  Explains the campaign note "XLA already overlaps GPU collectives".
- **XLA command-buffer / CUDA-graph capture** (#4): ALREADY set
  `xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,COLLECTIVES` (line 199)
  — CUSTOM_CALL is included, so the LAPACK gtsv FFI IS in the captured set
  (no graph fragmentation from it). (Worth a one-off Nsight trace to CONFIRM
  the FV3 step is one graph, but the flag coverage is already correct.)
- **Pipelined/communication-avoiding Krylov** (#5): equivalent SHIPPED
  (Chronopoulos-Gear single_reduce + Chebyshev reduction-free precond). The
  2025 POP/CG-variant papers beat a naive PCG baseline we're already past.
- **Ocean f32 path** (part of #2): ALREADY clean + measured 1.72x LL128
  (campaign L2); production runs f64 by SCIENTIFIC choice, not a missing
  lever. Tensor-core/TF32/bf16: future-hardware (Turing has none; gated).
- **Distributed SHT / JAX-Fluids 2.0 SPMD / Shardy**: same SPMD class we
  shipped, or NVLink/IB-gated, or a partitioner migration (not a new lever).

## GENUINELY NEW headroom (not shipped) — both are substantial numerics
### #1 Split-explicit wide-halo barotropic → ZERO global reductions
- Silvestri 2025 (JAMES 2024MS004465) / Oceananigans (arXiv:2502.14148):
  explicit barotropic subcycling with halo width = #subcycles → ONE halo
  exchange/baroclinic step, NO allreduce; barotropic <10% of step; 75 SYPD
  1/4° 16 A100. Attacks the ocean weak-scaling wall (allreduce latency) at
  the ROOT — strictly beats Chebyshev/single_reduce (which only cheapen the
  reductions). MORE favorable on Gloo-TCP (latency is exactly what it kills).
- **CAVEAT (banked memory conflict)**: my campaign memory notes "MPAS-O
  EXPLICITLY REJECTED wider halos to batch substeps (conditionally-stable →
  only a few stale substeps)". Oceananigans makes it work with a specific
  stable subcycle + averaging; whether OUR barotropic (explicit_substep /
  implicit_cn free surface) is stable under wide-halo batching of MANY
  substeps is UNVERIFIED. Needs a stability + dispersion + conservation
  eval BEFORE committing. = a new SELECTABLE barotropic scheme
  (bebt_subcycle_widehalo in barotropic_common), not a rewrite; numerics
  change → mandatory conservation validation + codex.

### #2 Precision-partition FV3 (fp32 advection/flux, fp64 PGF + gravity)
- GRIST (GMD 17,6301, 2024): 24-44% (44% tracer transport), Hygon/Sunway
  FPUs — "memory wall, not tensor cores". neXtSIM-DG (GMD 18,3017, 2025):
  fp32 ~80% on A100, negligible 48h drift; TF32 = the UNSAFE one.
- TURING-FAVORABLE: RTX8000 fp64=1/32 fp32 → up to ~32x arithmetic swing
  on FLOP-bound stencils (vs ~2x on A100) + halves HBM/PCIe/Gloo bytes
  (helps walls #1,#3,#4). The NEW part vs the existing global-f32 path is
  the PRECISION PARTITION (fp64 ONLY on PGF/gravity/δπ; fp32 elsewhere) =
  a TYPED dycore (explicit .astype at precision-sensitive ops), not a global
  flag. Conservation-critical (keep flux-divergence sum in fp64). Agent's
  recommended contained first slice: TRACER transport (44%, lowest risk).

## Recommendation / next
- NOT at the absolute limit: #1 + #2 are real new headroom (ocean
  weak-scaling at-root; atm/ocean per-device on Turing's 1/32-fp64).
- Both are conservation-critical multi-step numerics projects (validation +
  codex mandatory). #2-tracer is the more-contained first slice; #1 is the
  higher-ROI-but-needs-stability-eval (and must resolve the MPAS-O caveat).
- The XLA scheduling levers (#3/#4) are already optimal in backend.py.

## VERIFY-FIRST VERDICT (2026-06-13, same day — both levers DEAD-END at our scale)
Verified the two "new" levers against the actual codebase + measured ledger
BEFORE building. Both collapse at production/Ginsburg scale:

- **#2 precision-partition FV3 — MOOT.** `fv3_sw_core.py` has ZERO `astype`
  calls: the cube SW dycore is dtype-POLYMORPHIC (inherits the input array
  dtype) and ALREADY runs float32 (line 1636 carries an explicit "float32
  overflow guard"). The lever's premise (fp32 flux off a fp64 baseline) does
  not apply — flux is already fp32; there is nothing below fp32 to drop to.
  The GRIST 44% was off a fp64 baseline. Ocean f32 path also already exists.
- **#1 split-explicit wide-halo barotropic — ALREADY REALIZED + the novel
  part is net-negative.** `explicit_substep` ALREADY EXISTS
  (`barotropic_latlon_cgrid.py`, `barotropic.py`, `barotropic_mpas.py`) and
  its substep body ALREADY avoids allreduces — it uses point-to-point
  `pad_ns_*` halos (lines 318, 365), NOT global reductions. The PCG-allreduce
  wall is `implicit_cn`-ONLY; the bench already runs `explicit_substep` as the
  default lane. The ONLY genuinely-new part — wide-halo BATCHING (one
  exchange/baroclinic step via an `n_substeps`-wide halo) — is:
  (a) documented net-negative on a latency fabric (`crm_gpu_l2_tiling.md`:
      "needs an `n_substeps`-wide halo (→ net-negative)"), and
  (b) bounded by a tiny budget anyway: `scaling_indicators.csv` row
      `8475875` measured the lat-lon barotropic at **~7% of the step** on the
      production tile (rows/rank=48) — eliminating ALL of it caps at ~7%, and
      this is exactly why the Chebyshev precond was NEUTRAL at production tile.
  The one regime where barotropic dominates (MPAS-O, 67% of step,
  `scaling_gpu.md:626`) is being handled by the implicit_cn MPAS work in the
  CONCURRENT session (`barotropic_implicit_mpas.py`) — out of scope here.

**Conclusion:** Ginsburg is at the practical theoretical limit on every
*measurable* axis; the literature's "new" levers are already shipped, moot, or
net-negative on this hardware. The remaining genuine in-domain engineering is
NOT a Ginsburg-benchable number but a CAPABILITY: completing the cube >6-device
sub-face tiling (the deferred d_sw1/d_sw5/d_sw6 RK-stage ops, task #3) so the
cubed-sphere grid can use >6 devices AT ALL on future fast-interconnect HW
(TPU pods, NVLink nodes). Resuming that, gated by the proven U3 bit-identity
methodology.
# Spectral grid — level-shard scaling cliff (theoretical limit)

**Verdict: the spherical-harmonic spectral dycore does NOT scale across devices
under level-axis sharding — it is single-device by design.** This is the spectral
grid's theoretical limit on the only horizontal-decomposition-free sharding
scheme the codebase exposes, and it is the reason `create_level_mesh` reports
`_valid_gpu_counts -> [1]` and `DeviceConfig.is_distributed = False` for spectral.
Measured here to replace the asserted "by-design cliff" with data.

## Measurement (job 8520392, CPU, x64, nlev=24, 120 timed steps)

`scripts/bench/probe_spectral_shard.py` times `SpectralPrimitiveEquationModel`
SSP-RK3 steps with the level axis (`P(None, "level")`) sharded across `N`
emulated CPU devices, vs the single-device `none` baseline.

| truncation | none N=1 | level N=1 | level N=2 | level N=4 | speedup @ N=4 |
|------------|---------:|----------:|----------:|----------:|--------------:|
| **T42**    |   5.0413 |    4.8553 |    5.0324 |    5.3475 | 1.06× (flat)  |
| **T85**    |   0.7369 |    0.7175 |    0.5764 |    0.5484 | **0.74× (anti-scaling)** |

(steps/s; higher is better.) Adding devices never helps: T42 is flat within
run-to-run noise (~±6 %), and the larger T85 problem actively **slows down** as
devices are added.

## Why it cannot scale (mechanism)

The spectral state is `(n_sh, nlev)`. Two facts make level sharding a dead end:

1. **The semi-implicit solve couples all vertical levels.** Each zonal wavenumber
   carries a dense `(nlev, nlev)` implicit operator (gravity-wave / hydrostatic
   coupling). Sharding levels across devices forces an **all-gather of the level
   axis** before every SI solve — pure communication with no parallel speedup,
   because the solve itself cannot be decomposed across the sharded axis.
2. **The SH transforms emit no per-shard collectives.** `sh_analysis_3d` /
   `sh_synthesis_3d` batch/chunk the level axis for *memory* only, so the
   transform is replicated work per shard, not parallel work.

The anti-scaling worsens with resolution because the all-gather volume grows with
`n_sh`: T85 has ~4× the wavenumbers of T42 (3741 vs 946), so the level
all-gather is ~4× larger while the SI solve stays un-parallelized — hence T85's
clean 0.74× cliff vs T42's flat curve.

## What scaling spectral would actually require

The standard scalable spectral-transform parallelization is the **transpose
method** (e.g. IFS): domain-decompose grid space by latitude for the FFT/physics,
then **all-to-all transpose** to decompose by wavenumber for the Legendre
transform and SI solve. That avoids the all-gather but replaces it with an
all-to-all every transform — the most communication-intensive collective. On this
cluster's Gloo/TCP fabric (no NVLink/IB GPU-direct) that is comm-bound and
anti-scales, consistent with the campaign's multi-node verdict
(`docs/performance/scaling/distance_to_limit_2026-06-13.md`). It is therefore out of scope as
a Ginsburg throughput win; the single-device verdict stands as the practical
limit here.

## Precision note

Spectral is **float64-only by physics**: the transforms use `complex128` FFTs and
the precision policy forces `spectral_transform` to f64 storage/compute/accumulate
(`core/precision.py::_ATMOSPHERE_OVERRIDES`). There is no f32 spectral path, so
the "f32 vs f64" scaling axis is N/A for this grid — unlike the finite-volume
cube / lat-lon / icosahedral grids, which carry both precisions.

## Reproduce

```
sbatch scripts/cluster/scaling_ginsburg/spectral_cliff_probe.sbatch   # -> CSV
sbatch scripts/cluster/scaling_ginsburg/spectral_cliff_plot.sbatch <probe_job_id>
# -> docs/performance/scaling/spectral_level_shard_cliff.png
```

Plotter: `scripts/plot/plot_spectral_level_shard.py` (tested,
`tests/plot/test_spectral_level_shard_plot.py`). Correctness of the level-shard
*numerics* (independent of throughput) is gated by
`tests/parallel/test_spectral_level_shard.py` (single-device == level-sharded to
1e-12 at T21/T42, 2 and 4 devices).

web search: 
web search: site:docs.nvidia.com/deeplearning/nccl/user-guide/docs environment variables NCCL_ALGO NCCL_NCHANNELS NCCL_MIN_NCHANNELS latency small messages ...
exec
/usr/bin/bash -lc "sed -n '1,320p' docs/performance/scaling/cube_tiled_step_design.md && sed -n '1,280p' docs/performance/scaling/cube_transport_tiling_design.md && sed -n '1,300p' docs/performance/scaling/cube_production_tiling_design.md && sed -n '1,260p' docs/performance/scaling/cube_moist_tiled_step_design.md && sed -n '1,320p' docs/performance/scaling/mpas_atm_native_step_audit.md" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 501ms:
# Tiled cube hydrostatic STEP — design (next track after the tendency capstone)

**Status (2026-06-17):** the full `fv3_hydrostatic_tendencies` TENDENCY is now
tiled on np=6·kt² (`make_tiled_fv3_hydrostatic_tendencies_stage_2d`, commit
4ac0cb3ee, gate 8512958 5/5). This doc scopes the remaining track: wrapping that
tiled tendency in the full SSP-RK3 **time step** (the production `_step_fv3`).
**Future-HW only** — np>6 anti-scales on Ginsburg's Gloo/TCP fabric; gate by
BIT-IDENTITY vs the global `_step_fv3` base cut, never a wall-clock number.

> **UPDATE (2026-06-18) — BUILT.** This track is complete:
> `make_tiled_fv3_hydrostatic_step_stage_2d` (no-gather 3-stage SSP-RK3, commit
> 308202ff3), `make_tiled_zero_mean_tendency_stage_2d` + the delta-first
> `make_tiled_fix_ps_mass_stage_2d` (post-RK3 conservation psums, 798e174f5 +
> b66f82bb7) are all in `parallel/tiled_production_cdgrid.py`, bit-identity- /
> conservation-gated (np24/np54), cavecrew-clean and codex-reviewed (3 findings
> fixed: mesh/kt + nl≥2 guards 0f50c4afe, f32 delta-first fixer b66f82bb7).
>
> **UPDATE (2026-07-09) — WIRED (the production assembly).** The fast-
> interconnect hardware arrived (Derecho A100 + NCCL/Slingshot route-B), so
> the deferred hookup shipped, closing the single-shot gap (the step stages
> consume face-replicated state and emit tile-sharded state — feeding back
> required a full-cube gather per step):
>
> * `make_tiled_fv3_hydrostatic_step_blocked_2d` (tiled_production_cdgrid):
>   BLOCKED persistent layout, input layout == output layout, so
>   `s = step(s)` closes the loop with no per-step gather; optional
>   in-stage telescoping `fix_ps_mass` (shared `_tile_fix_ps_mass_delta`
>   psum) matching the serial `use_conservation_fixer+fix_mass` branch;
>   optional moist `column_physics_fn` (same contract as the moist stage).
>   BIT-IDENTICAL to the gated single-shot stage
>   (`test_tiled_blocked_loop.py::test_blocked_step_bit_identical_to_
>   shipped_stage`, Held-Suarez state, exact zeros).
> * `make_tiled_cc_loop` (tiled_step_adapter): `enter/step/exit_` over the
>   blocked layout (`expand_corners_to_blocks` at entry, adapter dedup at
>   exit — both one-time); production conservation config INSIDE its
>   envelope (unlike the single-shot adapter).
> * Wiring: `run_cpu_mpi_scaling --cs-spmd` at 6·kt² devices dispatches to
>   the blocked loop (kt≥2 previously fell into the non-tile-aware generic
>   `make_sharded_step`); `bench_cube_tiled_step_scaling --closed-loop` is
>   the honest feedback-timed lane (the prior lane could only time repeated
>   single shots on the pristine input); `cube_tiled_step.pbs/.sbatch` run
>   both parity + timed arms.
> * Gates: `tests/parallel/test_tiled_blocked_loop.py` — np24 multi-step
>   parity vs serial `model.step` (production conservation config, mass
>   conserved to 1e-12, duplicated shared faces bit-identical after N
>   steps), expand/dedup round-trip, envelope refusals.
> * Known pre-existing class (NOT introduced by the loop; bounded by the
>   shipped adapter gate): on production-magnitude states the tiled step
>   differs from serial by an O(1e-6 abs) face-corner wind term (level-
>   decaying, step-constant) — invisible on the gentle-random stage gates,
>   bounded at TILED_PARITY_ATOL in the adapter gates.
>
> **UPDATE (2026-07-09b) — the two remaining hookups shipped:**
>
> * **Kessler bridge**: `make_kessler_column_physics_fn`
>   (kessler_forcing.py) — the per-tile column contract over the SAME
>   shared `kessler_column_tendencies` core the face-sharded
>   `make_kessler_forcing_cube` lane runs (one controlled comparison
>   across the device ladder).  `make_tiled_cc_loop` gained the moist
>   mode (q_pack pack/step/unpack, exact-{q_v,q_c,q_r} refusals);
>   `run_cpu_mpi_scaling --cs-spmd --physics moist` now dispatches at
>   6·kt² devices.  Gate: `test_adapter_moist_loop_kessler_matches_serial`
>   (vs serial `model.step(physics_fn=make_kessler_forcing_cube)`).
> * **ModelDriver hookup**: `run()` dispatches cube runs with a sub-face
>   device tiling to `_run_tiled_cube_spmd` — a dedicated segment loop
>   over the blocked tiled loop (the `_run_compiled_latlon_spmd`
>   precedent; state tile-sharded across steps, per-SEGMENT gather for
>   the coupler callback + blowup guard).  Envelope: dynamics-only or
>   Kessler-microphysics-only (`_tiled_cube_column_physics_fn`); the
>   unified physics pipeline, Held-Suarez-on-cube, and the
>   diagnostics/checkpoint writers refuse loudly.  Gates:
>   `tests/parallel/test_tiled_cube_spmd_driver.py`.
>
> **UPDATE (2026-07-09c) — writers + unified-physics tiling core:**
>
> * **Writers**: `_run_tiled_cube_spmd` now runs the lightweight
>   `timeseries.npz` diagnostics (`diag_days`) and the checkpoint hook
>   (`checkpoint_days`, run()'s `_checkpoint_callback`-or-
>   `save_checkpoint` contract) — segment length = gcd of the active
>   cadences with the 1-day coupling cadence, so writers only ever see
>   the gathered cc state.
> * **Unified-pipeline physics tiling
>   (`driver/tiled_operator_split_step.py`)** — the tiled-cube twin of
>   the lat-band `sharded_operator_split_step`: ONE shard_map runs the
>   serial `_single_step` composition (cc→D dynamics RK3 → target-mass
>   fixer → `step_unified` column physics → Euler write-back →
>   saturation/moisture-fixer/Rayleigh tail → carry pack) per tile.
>   The **tile-aware SegmentCarry/PhysicsState shard**
>   (`shard_tiled_split_carry` + `pack_carry_tiled`/`unpack_carry_tiled`)
>   reshapes the FLATTENED per-column leaves (conv_prog, held radiation,
>   accumulators) grid-shaped — the row-major ncol order is not
>   tile-contiguous, so grid-shaped is the only shardable layout — and
>   the conservation reductions gained a tiled-mesh psum branch
>   (`conservation._spmd_lat_psum_or_none`), so the moisture/mass fixers
>   reduce correctly inside the tiled shard_map.  np24 gate
>   (`test_tiled_operator_split_step.py`): 2-step parity vs the serial
>   composition with a shape-agnostic column mock (the lat-band lane's
>   "2b" staging) — dynamics fields in the documented corner class,
>   physics carry/held/accumulators at 1e-12, global moisture conserved
>   to 1e-9.  Envelope refusals: `qv_smooth_coeff != 0` (full-cube ∇⁴
>   halo), `owned_mask` (MPI-replicated semantics).
>
> **UPDATE (2026-07-09d) — the REAL unified pipeline runs tiled.**  The
> driver statics build shipped: `build_tile_step_unified` constructs the
> PhysicsPipeline at TILE ncol (`_TilePhysicsGrid` shape-only view —
> `make_adapter` bakes ncol/shape_2d from it; land-active pipelines and
> horizontal-operator convection (w-grid / moisture-convergence traits)
> refuse loudly).  `make_tiled_operator_split_step` now takes the
> per-segment `forcing` as a per-call traced argument (SegmentForcing
> doctrine) with flat per-column forcing packed per call, validates the
> model's dynamics envelope (`validate_tiled_envelope` — driver
> `hyperdiff_scale`>0 etc. refuse), rebuilds transient GHG per step
> (`ghg_keys`), and flatten/re-grid-brackets the pipeline's per-column-
> NATIVE carry fields (`conv_prog`/`tke`/`qke`/`gwd_spectrum`, shape-
> validated at (ncol,...)) around the physics call.  The driver
> dispatches: `_tiled_cube_unified_active` (unified schemes or HS, minus
> Kessler-alone which keeps the simple blocked lane) routes
> `_run_tiled_cube_spmd` into `_run_operator_split_tiled_cube` — the
> full `_run_operator_split_spmd` mirror (tile-ncol `step_unified`,
> carry seed/shard/thread, per-segment forcing resample, gather for
> callback + blowup only).  END-TO-END GATE
> (`test_operator_split_tiled_cube_driver_parity.py`): full
> `ModelDriver.setup()+run()` with REAL gray radiation + prognostic-TKE
> turbulence, serial `_run_compiled` vs the np24 tiled lane — parity in
> the documented corner/abs classes, q_v 1e-8; hyperdiff refusal;
> dispatch predicate units.
>
> Remaining (refused loudly where reachable): multicontroller
> (cross-process) tiled operator-split; land-active / multilayer-land
> tiling; resolved-wind moisture advection under tiles; the writers in
> the operator-split lane (the simple tiled lane has them); w-grid /
> moisture-convergence convection schemes per tile.

## The layout problem (the crux)

`ssp_rk3_step(state, tendency_fn, dt)` (timestepping/ssp_rk3.py) is generic: it
calls `tendency_fn(state) -> tend` (SAME pytree layout as `state`) then does
pointwise `axpy`/`linear_combination`. The state is FACE-REPLICATED
(`P("face",None,None,None)`); the tiled capstone OUTPUTS are TILE-SHARDED, and
the D-grid winds are CORNER-STAGGERED with a duplicated shared tile face
(gathered `(6, kt·(nl+1), kt·(nl+1), nlev) != (6, n+1, n+1, nlev)`). So the
tendency layout ≠ the state layout — the RK3 pointwise combine cannot align them
directly.

## Two approaches

1. **Gather-RK3 (capability demo, LOW value):** `tendency_fn` runs the tiled
   capstone then DEDUPs the corner-staggered tiles back to face-replicated
   `(6,n+1,n+1,nlev)` (lower-owns-shared, like the gate's corner compare) before
   returning. `ssp_rk3_step` then runs unchanged on face-replicated arrays. Proves
   the step is bit-identical with a tiled tendency, BUT the per-stage gather is
   O(global) comm → negates the tiling benefit. NOT a scaling win — skip unless a
   pure capability checkpoint is wanted.

2. **One-shard_map-3-stage (the REAL no-gather win):** put the entire RK3 (3
   stages + the 2 SSP combines) inside ONE `shard_map`. Body per device tile:
   `tend = <capstone body>(tile_state); k1 = state + dt·tend; ...` ×3. State stays
   TILE-LOCAL across stages (no gather); halos exchanged 3× (once per stage's
   tendency). ~3× the capstone body. **Requires:** the corner-staggered
   duplicated shared-face tendency to be BIT-IDENTICAL between adjacent tiles so
   the duplicated state stays self-consistent across stages (verify first — the
   momentum gate's lower-owns-shared check is necessary but the cross-stage
   accumulation must also stay consistent). Gate: bit-identity of the stepped
   state vs the global `_step_fv3` base cut.

## Increment plan

1. **Tiled global-reduction primitive (`psum`-in-shard_map):** the first tiled
   GLOBAL reduction (all tendency stages so far are halo-only). Validate via the
   `zero_mean_tendency` variant (the capstone's documented `_apply_zero_mean_
   per_stage=True` follow-up) — an area-weighted `psum` over (face, tile_i,
   tile_j) of dp_s/dt before omega. Needed downstream by the mass-fixer.
2. **Shared-face tendency self-consistency check** (gate): adjacent tiles' shared
   corner-face `du_d_dt`/`dv_d_dt` are bit-identical — the precondition for the
   no-gather state to stay consistent across RK3 stages.
3. **One-shard_map 3-stage SSP-RK3** wrapping the capstone body (base cut: no
   post-step). Gate vs global `_step_fv3` base cut, sigma+hybrid, kt2/kt3.
4. **Post-step ops** (each needs a tiled global reduction / the psum primitive):
   `fix_ps_mass` (global mass), `damp_v` (post-step vorticity damp), the
   `sponge_implicit` multiplicative damp (pointwise, easy). Add incrementally,
   gated.

## Notes
- Reuse the capstone's halos verbatim (scalar `{B,zeta,1/T,ln_ps,hf,T}` + vector
  `vert_adv_uv` lift); the 3-stage body re-exchanges them per stage.
- The cc reduction-bearing tendencies (dT, dp_s) match to ~5e-8 under XLA fusion
  (diag 8512922/8512950 — reordering, not a bug); the stepped state will inherit
  similar O(1e-8) reordering on the cc fields — gate the step at a diag-justified
  tol, keep corner winds at 1e-10.
- All bit-identity, no wall-clock; np>6 is future-HW (TPU pod / NVLink), not
  Ginsburg-benchable.
# Cube tiled np>6 stage — PPM transport tiling design (2026-06-13)

STEP 2 of the full-tendency tiling grind (task #3; d2a2c DONE). The next
operator after d2a2c in `fv3_csw_tendencies` is the B-grid KE transport
(`_bgrid_ke_transport`, fv3_sw_core.py:2545), which is two 1-D PPM hord=9
sweeps (`_ppm_transport_1d`, :2278).

## The halo requirement (carefully derived — NOT a uniform h=4 exchange)

Production `_bgrid_ke_transport` provides the cross-face halo at depth
**h_dg=2** (`_pad_halo_dgrid_for_ppm(..., halo=2)`, :2576) and calls
`_ppm_transport_1d(external_halo=2)`. PPM then pads INTERNALLY to its
stencil halo **h3=4** with `mode='edge'` for the gap (:2300-2307). So
even globally the 2 outermost cells at a FACE boundary are edge-replicated.

PPM flux-at-interface stencil reach (sw_core.F90 jord=9, verified from the
code): the flux at interface j needs the monotone slope `dm[j±1]` (→
`vp[j-2..j+2]`) and the pmp/lac limiter `dq[j-2..j+1]`, i.e. ~3-4 cells
each side. h3=4 is the FV3 margin.

Per-tile consequence (tile owns face cells `[t·n_loc:(t+1)·n_loc]` on the
sweep axis):
* **Interior tile cuts** (neighbour tile is SAME FACE): the global field
  is contiguous-real there, so the tile needs the neighbour's **4 real
  cells** (depth-4) — a PLAIN copy, NO rotation, NO interp. depth-2 +
  edge-pad would put edge-replicated values where the global has real
  cells ⇒ flux mismatch at the tile-boundary interfaces. **depth-4
  interior halo is REQUIRED.**
* **Face-edge cuts** (neighbour is another FACE): only depth-2 cross-face
  data exists in production (rotated/interp'd) + PPM edge-pads 2 more —
  exactly the EXISTING d2a2c depth-2 cross-face machinery. Reuse it.
* **Corners**: a 1-D sweep reads only along its axis ⇒ **NO corner cells**.
  The hard 4×4 cube-vertex cascade is NOT needed for transport.

⇒ STEP 1 is a depth-4 SAME-FACE interior-cut strip exchange (plain
ppermute copy between same-face neighbour tiles) + the existing depth-2
cross-face path at face edges + PPM's own edge-pad. This is far more
tractable than extending `_build_tiled_pad` to a full h=4 (corner-cascade)
exchange.

## Corrections from codex design review (2026-06-13) — apply BEFORE coding

1. **Depth is 3-sufficient, 4-safe.** The active flux at an interior tile
   boundary reaches neighbour cell `N+2` (negative/upwind branch) and `-3`
   (positive branch at a left cut), so **depth-3 real same-face halo is
   sufficient**; depth-2 is INSUFFICIENT. Use depth-4 (safe, matches the
   `h3=4` storage convention; the 4th cell is non-load-bearing).
2. **rdelta (metric) ALSO needs an interior halo — MISSED in v1.**
   `_ppm_transport_1d` edge-pads `rd=rdelta` (`fv3_sw_core.py:2519`) and
   uses the UPWIND value for the CFL in BOTH flux branches
   (`rd_pad[:nn+1]` / `rd_pad[1:nn+2]`, :2520-2521, used :2530-2534). At an
   interior tile cut, edge-padding `rd` is WRONG for the branch whose
   upwind cell is in the neighbour tile ⇒ **interior cuts need a depth-1
   real `rdelta` halo along the sweep axis** (or a tile-specialised CFL
   that supplies the global upwind `rdelta` at the boundary interface). At
   true face edges, global edge-pads `rd` — MATCH that (no cross-face rd).
3. **Face-edge field halo must be BIT-IDENTICAL to `_pad_halo_dgrid_for_ppm`**
   (the depth-2 DGRID duogrid cross-face halo, `fv3_sw_core.py:40`), NOT a
   generic "d2a2c-ish" halo — the D-grid wind cross-face halo has its own
   rotation/interp; reproduce that exact function tile-locally, then PPM
   edge-pads to 4.
4. **Corners-free is `_bgrid_ke_transport`-ONLY (scope).** The two PPM
   calls are independent 1-D sweeps (axis=2 ytp_v, axis=1 xtp_u) — no
   diagonal read. But the LATER tendency transports do NOT generalise:
   `transport_step` (mass) and `fv_tp_2d` (vorticity, Lin-Rood 2-D with
   cross-sweep halo exchanges + del-n damping, `fv_tp_2d.py:922-1067`) ARE
   2-D and WILL need corners — a separate (harder) sub-build later.

Corrected STEP-1 halo spec: interior same-face FIELD depth-4 (≥3) real +
RDELTA depth-1 real; face-edge FIELD = `_pad_halo_dgrid_for_ppm` depth-2 +
PPM edge-pad, rdelta edge-pad (match global); NO corners (this op only).

## APPROACH C + boundary-fix-OFF (2026-06-13 — the implementation approach)

Two findings collapse the implementation to the proven d2a2c approach-C
pattern (compute halo GLOBALLY in the GSPMD view, strided-slice per tile),
NOT a new ppermute exchange:

1. **`_pad_halo_dgrid_for_ppm` is a FULL-FACE op** (built on `ext_vector_dgrid`
   — D→A avg + duogrid cross-face rotation, `fv3_sw_core.py:40-84`), NOT a
   strip exchange. Reproducing it tile-locally is wrong-headed. Instead:
   compute it GLOBALLY (cheap, once per face), pre-pad the field to the PPM
   storage halo h3=4 exactly as `_ppm_transport_1d` does internally
   (`_pad_halo_dgrid_for_ppm(halo=2)` then `jnp.pad(mode='edge', gap=2)`),
   then each tile takes a strided window `[t·n_loc : t·n_loc + n_loc + 2·h3]`
   (= `tiled_padded_block`, the d2a2c approach-C helper). Interior cuts get
   real depth-4 from the contiguous global interior; face edges get the
   ext_vector depth-2 + edge-pad — bit-faithful to global by construction.
   Same for the courant `c` (interface-located, tile-sliceable) and `rdelta`
   (global edge-pad depth-1 then slice — satisfies the rd-halo requirement).
2. **Boundary-fix is OFF for THIS transport.** `_bgrid_ke_transport` calls
   `_ppm_transport_1d` WITHOUT `apply_d_sw3_boundary_fix` (default False;
   "iter-967 NEGATIVE: d_sw3 boundary fix conflicts with the iter-945
   halo"). So the SOUTH/NORTH boundary-fix branch (`fv3_sw_core.py:2449-2516`
   — the one-sided edge-PPM specials + pert_ppm) is DEAD here ⇒ the
   reconstruction is UNIFORM PPM, NO edge specials. Unlike d2a2c (which had
   intricate per-side edge specials), tiled transport is uniform — the only
   tile-position-dependence is the halo slice, which approach-C handles.

IMPLEMENTATION UNITS (gate + codex each):
* U1: refactor `_ppm_transport_1d` → extract `_ppm_flux_core(vp_h3, c, rd_pad)`
  taking a PRE-PADDED field (h3=4) + courant + pre-padded rd, with the public
  `_ppm_transport_1d` calling it (BIT-IDENTICAL global path — sbatch parity
  gate). This removes the internal edge-pad so a tile can supply real halo.
* U2: `transport_tile` = global pre-pad (U1 inputs) → `tiled_padded_block`
  slice per tile → `_ppm_flux_core` → tile interface fluxes. Parity: tiled
  == global `_bgrid_ke_transport` PPM output, all tile positions, kt=3.
* U3: wire into the tiled stage next to d2a2c; np-parity.

This is deliberate multi-hour work (a delicate dycore refactor kept
bit-identical); do U1→U2→U3 as separate gated+codex commits.

## U3 design (shard_map stage) — mirror make_tiled_d2a2c_stage (2026-06-13)

U1 (rd_prepadded, 65fa25fe) + U2 (synthetic approach-C parity, 6779dcbf) +
U2b (production cross-face/staggered i-sweep parity, 2150ce17) PROVE the
per-tile PPM compute. U3 wires it into a `shard_map(face, tile_i, tile_j)`
stage, mirroring `make_tiled_d2a2c_stage` (packages/core/legoesm/parallel/
tiled_d2a2c.py) — the established APPROACH-C-under-shard_map pattern:
* inputs FACE-REPLICATED `P("face", None, None)`: the GLOBAL h3=4 pre-padded
  transport field (computed outside shard_map in the GSPMD view, e.g.
  `_pad_halo_dgrid_for_ppm` then `jnp.pad` to h3), the courant, and the
  depth-1 pre-padded rd;
* inside `_stage`: `ti/tj = axis_index`; each device `lax.dynamic_slice`s
  its tile's sweep window (i-window `[ti*nl : ti*nl+nl+2*h3]`, j-block) from
  the replicated padded face — NO halo ppermute (the padded face already
  carries the h3 halo, the d2a2c-stage insight), then
  `_ppm_transport_1d(external_halo=4, rd_prepadded=True)` per tile;
* output tile-sharded `P("face","tile_i","tile_j")`; reassembly dedups the
  shared boundary interface (lower tile owns it), == global PPM sweep.

CARE-POINT (regression risk, do carefully + gated): the STAGGERED j-axis
slice (u_d j=n+1). Start with an i-ONLY mesh `(6, kt, 1)` (full j, no
staggered-j tiling → np=6·kt, kt=4 gives np24) to prove the shard_map
i-sweep first; then the full `(6, kt, kt)` with the staggered-j block slice
(match `tiled_face_block`'s nl-vs-nl+1 convention) = U3b. Then ytp_v
(axis=2 j-sweep, symmetric). Parity: in-shard_map vs global
`_ppm_transport_1d` (pattern: tests/parallel/test_tiled_d2a2c_ua_va.py,
XLA_FLAGS=--xla_force_host_platform_device_count). THEN real Courant
(ub/vb from uc/vc) + B-grid corner sync = full `_bgrid_ke_transport` tiled.

## Plan (gate + codex EACH piece — user directive "be very careful")

1. **depth-4 interior-cut strip exchange** (`make_tiled_ppm_halo` or a
   `_build_tiled_pad` halo=4 EDGE-ONLY path): per sweep axis, each tile
   ppermutes its 4 boundary cells to the same-face neighbour tile; at
   face-edge cuts fall back to depth-2 cross-face (existing) + edge-pad.
   Parity vs the GLOBAL `_pad_halo_dgrid_for_ppm(halo=2)` →
   `_ppm_transport_1d` intermediate, sliced to the tile, on INTERIOR
   tiles (kt=3) and FACE-EDGE tiles. f64 atol/rtol 1e-12.
2. **tile-local transport** (`_ppm_transport_1d_tile` or feed the depth-4
   tile into the existing `_ppm_transport_1d` with the right
   external_halo): the per-tile transported field == global transport
   sliced to the tile, kt=3 all tile positions.
3. Wire into the tiled stage alongside d2a2c.
4. Bernoulli/gradient + vorticity/PV operators (next ops).
5. Full RK tendency shard_map stage + np24 parity (vs serial) + bench
   (target ~15-30ms vs GSPMD-auto 2253ms C96).

No measurable payoff until the full stage assembles — deliberate
multi-session grind. Each piece: extract → host+in-shard_map parity gate
→ codex adversarial review → commit.

## U3d — real-Courant composition of _bgrid_ke_transport (2026-06-13)

Status after U3b (full 2-D `(6,kt,kt)` sweep tiling, both i/j) + U3c
(`bgrid_corner_courant_local`, corner Courant tiled): the remaining
transport pieces are the **real-Courant composition** + the **BGRID_NE
corner sync**. The CRUX (verified this session) is the STAGGERED cross-axis
mismatch — do NOT assume the U3b synthetic matched shapes:

- ytp_v (axis=2) call: `_ppm_transport_1d(v_d_jhalo, vb, rdy, axis=2,
  external_halo=h_dg=2)`. Shapes: `v_d_jhalo (6, n, n+1+2h)` [field, cell on
  axis=1, j-halo'd corner on axis=2], `vb (6, n+1, n+1)` [CORNER Courant],
  `rdy (6, n+1, n)`. Inside, swapaxes(1,2) makes the sweep axis=1; the
  field's CROSS axis is `n` (i-cells) but `vb`'s cross axis is `n+1`
  (i-corners) — **off by the staggering**. So a tile's `vb` cross slice
  (n+1 corners → nl+1) is ONE LONGER than the field's cross slice (n cells →
  nl). The U3b `transport_jsweep_tile_2d` cross slice (`[a_i:a_i+nl]`) is
  CORRECT for the field but the courant needs its own `[a_i:a_i+nl+1]` (or
  the relevant n-of-(n+1) sub-slice that `_ppm_transport_1d` actually
  indexes — TRACE the courant cross-indexing in `_ppm_transport_1d` before
  slicing; it likely uses only `n` of the `n+1` corner courant rows).
- xtp_u (axis=1) is the symmetric stagger (ub corner cross-axis n+1 vs u_d
  cell n).

PLAN (each: extract → host + in-shard_map parity → codex → commit):
1. TRACE `_ppm_transport_1d`'s courant cross-axis usage (which `n` of the
   `n+1` corner-courant rows it reads) → derive the exact courant tile slice.
   Add a `transport_sweep_tile_2d`/`_jsweep` variant (or a `courant_cross`
   arg) that slices the courant cross-axis by `nl+1` (corner) while the field
   cross by `nl` (cell). Parity: real cdgrid + real vb/ub (from U3c on the
   real cross-face-halo'd uc/vc) → tiled transported_y/x == global
   `_bgrid_ke_transport` (BEFORE Step-5), on INTERIOR (kt=3) + face-edge.
2. CROSS-FACE HALO (approach-C, GLOBAL pre-pad → slice; NOT tiled): the
   stage takes the globally-prepadded `uc_pad/vc_pad`
   (`_pad_halo_uc_vc_new_via_old_delta`) + `u_d_ihalo/v_d_jhalo`
   (`_pad_halo_dgrid_for_ppm`) as FACE-REPLICATED inputs; the tile slices
   them. The corner metrics (`cosa_corner/rsin2_corner`, `(6,n+1,n+1)`) slice
   via the mesh.py STAGGERED `tiled_face_block` (nl+1 ownership).
3. BGRID_NE corner sync (Step 5, the c2l z-matrix corner rotation) — the
   ONLY genuinely cross-TILE-coupled piece (a corner vector avg through the
   geographic frame); needs a tiled corner exchange (mirror the d2a2c
   corner rounds). Highest-risk piece; gate hardest.
4. Assemble `bgrid_ke_transport_tiled_stage` (corner Courant → sweeps →
   corner sync) under one `shard_map(face,tile_i,tile_j)`; np24 parity vs
   serial.
5. Bernoulli/gradient + vorticity/PV → full RK tendency stage → np24 bench.

GATE NOTE (banked): the bit-identity guards for any `_bgrid_ke_transport`
refactor are conftest-`slow`-marked (run with `-m "slow or not slow"`) AND
hit the LLVM section-memory OOM at 32G → use **64G + `XLA_FLAGS=
--xla_cpu_multi_thread_eigen=false` + `TF_NUM_INTEROP_THREADS=1`**.

DELIBERATE multi-session work — NOT loop micro-turns. No measurable SYPD
payoff until the full RK stage assembles, AND the np24 payoff is NOT
Ginsburg-benchable (no 24 real devices; CPU-virtual oversubscribes) — it is
a multi-GPU-node (A100/H100) capability validated for CORRECTNESS here.

## U3d shapes — RESOLVED by runtime probe (job 8480355, C18 duogrid, n=18)

`scripts/tmp/_probe_bgrid_shapes.py`. n+1=19, h_dg=2 → n+2h=22.
- `u_d (6,n,n+1)=(6,18,19)`, `v_d (6,n+1,n)=(6,19,18)`
- `uc (6,n+1,n)` → `uc_pad (6,n+1,n+2)`; `vc (6,n,n+1)` → `vc_pad (6,n+2,n+1)`
- `vb=ub=(6,n+1,n+1)=(6,19,19)` (corner)
- `u_d_ihalo (6,n+2h,n+1)=(6,22,19)`; `v_d_jhalo (6,n+1,n+2h)=(6,19,22)`
- `dy_edge_x=(6,n+1,n)`; `dx_edge_y=(6,n,n+1)`

ytp_v `_ppm_transport_1d(v_d_jhalo, vb, rdy=1/dy_edge_x, axis=2, ext=h_dg=2)`:
**ALL share CROSS axis=1 = n+1 (corners)** — `v_d_jhalo[:,n+1,·]`,
`vb[:,n+1,·]`, `rdy[:,n+1,·]`. Sweep axis=2: field n+2h, vb n+1 (interfaces),
rdy n (cells). So the U3b `transport_jsweep_tile_2d` composes with the REAL
arrays IF the **cross slice = nl+1 (corner ownership, shared corner), not
nl** — NO field-vs-courant cross mismatch (my earlier worry was wrong; the
field's cross is n+1, not n). xtp_u is the symmetric transpose (cross
axis=2 = n+1).

⇒ U3d impl (next): a corner-cross sweep tile = U3b j/i-sweep with cross
slice nl+1 + the U2b global-pre-pad-to-h3 sweep handling, driven by the REAL
vb/ub (from U3c on real cross-face-halo'd uc/vc) + real v_d_jhalo/u_d_ihalo.
Reassemble: cross (corner) lower-tile-owns-shared nl+1→n+1; sweep
lower-tile-owns-shared nl+1→n+1. Parity vs global transported_y/x (pre
Step-5) on kt=3 interior + face-edge. Then Step-5 BGRID_NE corner sync.

## U3f — tiled BGRID_NE corner SCALAR sync: KEY SIMPLIFICATION (2026-06-13)

After U3e (pointwise local↔geo factored), the only remaining corner-sync
piece is the cross-tile `synchronize_corner_scalar` (halo.py:2678). Reading
it (verify-first) gives a MAJOR simplification for the tiled stage:

**synchronize_corner_scalar modifies ONLY the FACE-BOUNDARY corners**
(Pass 1: face EDGES i=0/n, j=0/n via CONNECTIVITY pairwise avg; Pass 2: the
8 cube VERTICES, 3-face mean). The face-INTERIOR corners are UNTOUCHED.

⇒ The tiling introduces interior cuts, but the sync does NOT average at
interior corners — so the tiled stage needs **NO interior-cut corner sync**.
Each interior corner is a single global value; approach-C (face-replicated /
global-pre-synced input, per-tile slice) hands every sharing tile the SAME
value automatically. This KILLS the hardest-feared part (the same-face
interior-cut all-reduce / d2a2c diagonal+sliver rounds are NOT needed for the
scalar sync).

**Remaining tiled work = ONLY the cross-FACE part at FACE-BOUNDARY tiles:**
- Pass 1 cross-face edge avg: only tiles with ti∈{0,kt-1} or tj∈{0,kt-1}
  hold a face edge; exchange+avg that edge with the CONNECTIVITY neighbour
  face's edge tile. O(n) per face edge.
- Pass 2 vertex 3-face mean: only the 4 corner tiles (ti,tj ∈ {0,kt-1}²) per
  face hold a cube vertex; 3-face mean across the corner tiles meeting there.
- _tiled_corner_modes mode 0 (true cube vertex) already enumerates the
  vertex-tile set; modes 2/3 (slivers) and 1 (diagonal) are NOT needed
  (those are interior/halo-fill, irrelevant to the boundary-only scalar sync).

**Cleanest approach-C implementation:** run synchronize_corner_scalar
GLOBALLY on the (face-replicated) corner field BEFORE the per-tile slice —
it is a cheap O(n) edge/vertex op, NOT per-cell. The full-field is already
face-replicated in the stage inputs (P("face",None,None)), so the global
sync needs NO collective beyond what the face-replication already provides;
the tiled stage then slices the synced corners. I.e. the corner SCALAR sync
can live in the GLOBAL pre-step (like the cross-face halo pre-pad), NOT
inside the per-tile shard body — NO new tiled cross-face exchange required.
# Cube >6-device sub-face tiling — PORT TO THE PRODUCTION DYCORE (task (a))

**Goal.** The U3/U4 sub-face tiling (d2a2c, PPM transport, B-grid KE, d_sw1 ut/vt)
proved approach-C on the EXPERIMENTAL FB chain (`_d_sw_native`). Production cube
uses a DIFFERENT op set — `fv3_sw_tendencies` (operators_cdgrid.py, SW) and
`fv3_hydrostatic_tendencies` (primitive_eq_cdgrid.py, 3D AMIP). To let the
PRODUCTION cube scale past 6 devices (np = 6·kt·kt) on future fast-interconnect
HW (TPU pods / NVLink), tile the production ops with the SAME approach-C.

**Not Ginsburg-benchable** (Ginsburg = ≤2 GPU/node, no IB/NVLink; np>6 = CPU
shard_map which anti-scales on Gloo-TCP, or cross-node ppermute 70-245× < HBM).
So every increment is gated by **bit-identity** (tiled reassembly == global), the
proven U3 methodology — correctness, not a wall-clock win.

## Approach-C recap (reuse the U3/U4 machinery)
1. cheap GLOBAL face-replicated pre-pad (cross-face halo) — already
   `packed_pad_halo_4d` / `pad_halo_*` (face-replicated stage inputs).
2. per-tile `lax.dynamic_slice` of the tile window from the padded global field.
3. device-uniform local body (the op's stencil) — NO in-stage ppermute.
4. reassemble (lower tile owns the shared staggered face) == the global op.
5. shard_map stage on a `(6, kt, kt)` mesh (`tiled_d2a2c.make_tiled_*` pattern).

## Production `fv3_sw_tendencies` ops (operators_cdgrid.py:1490) — tile order
Each is a local stencil after a 1-cell (or h) halo → tileable. Increment order
= simplest-first, each its own kernel + bit-identity test + shard_map stage:

- **P-i  `dgrid_vorticity`** (corner winds → cell-centre ζ): local curl stencil.
  Smallest, no metric subtlety beyond rarea. FIRST increment (establish the
  production-tiling test harness mirroring tests/parallel/test_tiled_*).
- **P-ii `interp_corner_to_center` / `interp_center_to_corner`** (4-cell avg):
  identical box-stencil to U4a dsw1 ut/vt — reuse the dsw1_ut_vt_tile_2d shape
  logic. The Bernoulli-gradient corner winds (sec e) + dB cc (sec g) use these.
- **P-iii `arakawa_lamb_gradient`** (B, ln_ps): the corner AL gradient — already
  `padded=`-aware (the PE coalescing reuses it). Tile the corner stencil.
- **P-iv `cgrid_divergence` / `cgrid_mass_flux_divergence`** (mass tendency +
  div damp): C-grid flux divergence — local; tile the flux + divergence.
- **P-v  `fv3_d2cc` / `fv3_cc2c`** (D→cc→C velocity): vector interp (needs the
  vector halo rotation at cube edges — like the d2a2c vector case). HARDEST
  (vector); do after the scalar ops.
- **P-vi  Bernoulli `KE = 0.5*(u_cc²+v_cc²)` + `B = KE+g(h+h_s)`**: pointwise
  (trivial; rides whatever cc winds the tiles hold).

Then assemble the full `fv3_sw_tendencies` shard_map stage → bit-identity vs
global at C18/C24 (kt=2,3) → the SW production np24 capability.

## 3D follow-on (`fv3_hydrostatic_tendencies`)
Same ops + the vertical (nlev) trailing axis (all kernels already 4D-native via
the `(F,n,n,nlev)` shape). The PE stage-pack coalescing (ln_ps/hf/div_v,
shipped) already tiles the HALO; the per-op stencils are the remaining tiling.

## Gate strategy (per increment)
- Host-body bit-identity (kt=3 → interior+edge+corner tiles), `rtol=0,atol=1e-12`
  (portable; exact is non-portable per the U4a FMA finding).
- `(6,2,2)` shard_map np24 (skip < 24 host devices), same reassembly.
- Duplicate-shared-face parity (validates lower-owns trim).
- codex adversarial review each increment; smoke = the bit-identity gate.

## PRE-EXISTING tiling (Jun-11, via parallel.mesh staggered_tile_block + *_local cores) — DO NOT RE-DUP
A prior session already tiled several cube ops (host-body + shard_map) using
`staggered_tile_block`/`tiled_face_block` (parallel.mesh) + leading-axis-agnostic
`*_local` cores, with tests `test_tiled_{cgrid_divergence,cgrid_gradient,
d2a2c_ua_va,cdgrid_field_classification,staggered_layout}`:
- **`cgrid_divergence`** (`cgrid_divergence_local`) — DONE, and it IS the
  fv3_sw_tendencies mass/div-damp divergence → reuse it (do NOT re-tile; P-iv
  cgrid_divergence was attempted then reverted as a dup 2026-06-14).
- `cgrid_gradient_2d` (`cgrid_gradient_2d_local`) — DONE, but a DIFFERENT op
  (NOT arakawa_lamb_gradient; not on the fv3_sw_tendencies path).
- d2a2c / staggered-layout / field-classification — infra + experimental-d2a2c.
The two slicing styles coexist: Jun-11 `staggered_tile_block` is STATIC-index
(host-body); my `tiled_production_cdgrid` uses `dynamic_slice_in_dim` (works in
shard_map too). Both call the shared `*_local`/`*_core` numerics (no dup of
numerics).

## Status (corrected after the Jun-11 reconciliation)
- **DONE (my P-series, tiled_production_cdgrid.py, fv3_sw_tendencies-relevant):**
  P-i `dgrid_vorticity` (f0246a46), P-ii box interps (79243312), P-iii
  `arakawa_lamb_gradient` (1ab7edae). All host-body 3D+4D + np24 shard_map,
  bit-identity-gated, codex-clean.
- **DONE (Jun-11):** `cgrid_divergence` — reuse `cgrid_divergence_local`.
- **P-v `fv3_d2cc` + `fv3_cc2c` SHIPPED (406ddc6a)** — velocity transforms
  (fv3_d2cc local 2-pt avg; fv3_cc2c VECTOR via the pad_halo_vector pre-pad +
  the extracted fv3_cc2c_core; 3D-only, no 4D caller). host-body + np24, codex-clean.
- **MOMENTUM tendency (du_d_dt/dv_d_dt) now FULLY tileable** — every op it needs
  is done: P-v (cc/C winds) -> Bernoulli (pointwise) -> P-iii (A-L grad) + P-ii
  (corner winds + interp_corner_to_center) -> P-i (dgrid_vorticity) ->
  cgrid_divergence (Jun-11, div damp).
- **ASSEMBLY: tiled `fv3_sw_tendencies` MOMENTUM stage SHIPPED** —
  `make_tiled_fv3_sw_momentum_stage_2d` (tiled_production_cdgrid.py). The
  genuine np24 momentum unlock: chains fv3_d2cc -> Bernoulli (inline pointwise)
  -> [B SCALAR in-stage halo] -> A-L grad -> [u_cc/v_cc VECTOR in-stage halo]
  -> corner winds -> dgrid_vorticity -> interp_corner_to_center -> du_cc/dv_cc
  -> [VECTOR in-stage halo] -> D-grid project. The INTERMEDIATE halos (B; cc
  winds; cc tendencies — no global pre-pad) are exchanged IN-STAGE via the
  EXISTING `make_tiled_pad_body` (scalar) / `make_tiled_pad_vector_body`
  (vector) — the foundational sub-face halo (incl. the mode-1 diagonal corner
  ppermute) was already built (U3). Base case only: div_damp=0, hyperdiff=0,
  boundary_fix=False, fortran_*=False, non-duogrid. Bit-identity vs global
  `fv3_sw_tendencies` (du_d_dt, dv_d_dt) at kt=2 (np24) + kt=3 (np54),
  rel<1e-10; codex-clean (7/7 vectors). Gate: `test_tiled_fv3_sw_momentum.py`.
- **MASS-PPM `cgrid_mass_flux_divergence` (dh_dt) SHIPPED** —
  `make_tiled_cgrid_mass_divergence_stage_2d` (tiled_production_cdgrid.py). The
  design-doc HARDEST op, tiled via the U3 DEEP-GLOBAL-PRE-PAD pattern (NOT an
  in-stage halo): `h` is a STAGE INPUT, pre-padded one ring deeper than the
  production halo=2 (`h_deep` = `_pad_halo_auto_h2` + 1 edge ring, n+6) and the
  deep window sliced per tile -> the per-tile PPM reconstruction is LOCAL. The
  cc winds u_c/v_c are staggered stage inputs (NO halo — divergence reads only a
  cell's own bounding faces). Depth insight: PPM reconstruction of the
  tile-boundary cell (local -1) reads cells [-3..1], so a tile needs `halo_in=3`
  (one ring deeper than the global's halo=2); the deep pad's outer ring is
  edge-extended, so a FACE-edge tile reproduces the global's internal
  `mode='edge'` ghost. Achieved by `_cgrid_ppm_fluxes_core(h_pad, u_c, v_c, dy,
  dx, n_local, *, halo_in)` extracted from `_cgrid_ppm_fluxes_2d_no_sync` — ONE
  core, `halo_in=2` drives the global op (the inline 2D path + the 4D no_sync
  wrapper both now call it, dedup), `halo_in=3` drives the tile; the face
  indices `q_R[halo_in-1:...]`/`q_L[halo_in:...]` collapse to the exact historical
  `[1:n+2]`/`[2:n+3]` at halo_in=2. Base case: non-duogrid (no
  `synchronize_cgrid_fluxes`), `apply_fortran_xppm_boundary=False` (the
  `n_interior` face-edge override stays a later increment). Bit-identity vs
  global at kt=2 (np24) + kt=3 (np54), rel<1e-10; codex-clean (8/8 vectors).
  Refactor proven bit-identical: clean-HEAD vs refactor isolated SW integration
  runs give IDENTICAL results to 16 digits (test_fv_cubesphere 7/7,
  test_shallow_water 11/11; test_boundary_fix mass-conservation is a PRE-EXISTING
  2.51e-6-vs-1e-6 miss, identical on clean HEAD). Gate:
  `test_tiled_mass_divergence.py`.
- **FULL `fv3_sw_tendencies` np24 STAGE SHIPPED (CAPSTONE)** —
  `make_tiled_fv3_sw_tendencies_stage_2d` (tiled_production_cdgrid.py) ->
  `stage(h,u_d,v_d,h_s) -> (dh_dt, du_d_dt, dv_d_dt)`. Composes the momentum
  assembly + mass divergence, with the cc-wind VECTOR halo computed ONCE and
  SHARED: one `vector_body(u_cc,v_cc)` feeds BOTH `fv3_cc2c_core` (-> u_c/v_c
  for the mass PPM) AND the momentum corner winds — so the full stage carries
  the SAME halo budget as momentum alone (1 scalar B + 2 vector + the global
  deep-h pre-pad). h for Bernoulli is taken from the deep-window interior
  (`hw[:, 3:-3, 3:-3]`). Bit-identity of ALL THREE tendencies vs global at kt=2
  (np24) + kt=3 (np54), rel<1e-10; codex-clean (8/8 composition vectors). Gate:
  `test_tiled_fv3_sw_full.py` (job 8482569: TILED_FULL_GATE_OK). The production
  cube SW dycore tendency now runs on np=6*kt^2 devices.
- **MULTI-NODE VALIDATION** — `scripts/validate/validate_tiled_fv3_sw_multinode.py`
  + `scripts/cluster/scaling_ginsburg/tiled_fv3_sw_multinode.sbatch`: runs the
  full stage under REAL multi-controller `jax.distributed` across 2 nodes (np24,
  in-stage ppermutes -> cross-NODE collective-permute); each process
  self-validates its local tile vs serial (rel<1e-9). Correctness, not a bench.
- **PRODUCTION ASSEMBLY WIRED (2026-07-09)** — the blocked persistent step
  (`make_tiled_fv3_hydrostatic_step_blocked_2d`, input layout == output
  layout, in-stage telescoping mass fixer, optional moist column physics)
  + `make_tiled_cc_loop` (adapter enter/step/exit_) + the
  `run_cpu_mpi_scaling --cs-spmd` 6·kt² dispatch and
  `bench_cube_tiled_step_scaling --closed-loop` lane.  Bitwise-identical to
  the gated step stages (dry + moist gates in
  `tests/parallel/test_tiled_blocked_loop.py`).  See
  `cube_tiled_step_design.md` (2026-07-09 update) for the full contract.
- **REMAINING (later increments, NOT base-case-blocking):**
  - Optional momentum terms (div damp / hyperdiff / boundary smoothing /
    Fortran corner specials) — each rides the shipped per-op kernels + one more
    in-stage halo; their own increments.
  - The Fortran xppm boundary overrides (`n_interior` keyed to GLOBAL face
    index) + duogrid `synchronize_cgrid_fluxes` for the mass tile.
  - Kessler column bridge for the moist blocked loop in the drivers +
    `ModelDriver` (full-model coupling) hookup.
- LESSON: a prior session built tiling infra; ALWAYS grep `tests/parallel/
  test_tiled_*` + `parallel.mesh` before tiling a cube op.
# Cube-MOIST tiled np>6 step — design (the moist analogue of the dry tiled step)

**Status (2026-06-19):** KICKOFF. The DRY tiled 3D-PE SSP-RK3 step
(`make_tiled_fv3_hydrostatic_step_stage_2d`) is built + np24/54 bit-identity
parity-gated (`test_tiled_fv3_hydrostatic_step.py`, kt=2 np24 PASS confirmed job
8532478). Task #31 made cube-MOIST scale multi-device on the REPLICATED cs-spmd
path (`make_sharded_step` + `model.step_with_physics`), but that path is capped at
**np≤6** (one face per device; no sub-face decomposition). To reach the cube's
theoretical limit for MOIST (np=6·kt², kt≥2 → 24, 54, …) the moist column physics
+ tracer transport must ride the SAME sub-face tiling the dry step already uses.
Future-HW (np>6 anti-scales on Ginsburg CPU); every increment gated by BIT-IDENTITY
(tiled reassembly == global serial-moist), the proven U3 methodology.

## What the dry tiled step already gives us (reuse, do NOT re-derive)

- `make_tiled_fv3_hydrostatic_step_stage_2d(mesh, cdgrid, coord, n, kt)` →
  `step(u_d, v_d, T, p_s, phis) -> (u_d, v_d, T, p_s)` — the dry SSP-RK3 step on a
  `(6, kt, kt)` mesh, no gather.
- `make_tiled_fv3_hydrostatic_thermo_stage_2d` — the per-tile dT/dt **scalar**
  advection (horizontal AL/PPM flux + vertical). T is a passive scalar on the
  C-grid winds; **a tracer q advects by the IDENTICAL stencil** (same u_c/v_c,
  same vertical mass-flux `_tracer_vert_fn`). So the moist tendency is the thermo
  kernel applied to each of q_v/q_c/q_r.
- The tiled vertical transport path already handles "T/tracers/momentum" vertical
  advection (tiled_production_cdgrid.py:505).
- The sub-face halo (`make_tiled_pad_body` scalar / `make_tiled_pad_vector_body`
  vector, incl. the diagonal-corner ppermute) is bit-exact to serial
  `pad_halo`/`pad_halo_vector` — the tracer scalar halo reuses it unchanged.
- Serial reference: `fv3_hydrostatic_tendencies` already advects tracers via the
  shared `advective_tracer_tendency` (primitive_eq_cdgrid.py:898) and adds
  `physics_tendency_cc.tracer_tendencies` (Kessler). `make_kessler_forcing_cube`
  is the column-local (no-halo) physics.

## Increments (each: bit-identity np24 vs serial-moist, 24/54 faked CPU devices)

1. **Tiled single-tracer tendency.** Generalize the thermo tile body to advect a
   passed scalar `q` (NOT just T): `make_tiled_fv3_hydrostatic_tracer_stage_2d` or
   a `tracers=` arg on the thermo stage. Parity: tiled dq/dt == serial
   `advective_tracer_tendency` for one tracer at kt=2. (The scalar halo + vertical
   fn already exist — this is threading, not new numerics.)
2. **Tracer pack (q_v/q_c/q_r).** Extend to the dict/packed tracers the FV3 state
   carries (`FV3HydrostaticState.tracers`); one tiled scalar-advection per tracer
   (or batched on a trailing tracer axis like the serial `_q_packed`). Parity vs
   serial packed tracer tendency.
   **DONE (commits 155ff92a5 increment-1 + 30fbb3cd3 increment-2; gates 8532495 /
   8532505 PASS).** Single-tracer + packed q_v/q_c/q_r tiled advection in
   `tiled_production_cdgrid.py`; np24 bit-identity vs serial
   `advective_tracer_tendency`; thermo refactored onto the shared helper
   (bit-identical, no regression).
3. **Kessler per-tile (combined moist tracer tendency).** Add the warm-rain tracer
   tendencies to the increment-2 advection.  ARCHITECTURE FINDING (scout
   2026-06-20): kessler is pointwise (no halo, `mesh` unused), BUT the public
   state-based closures (`make_kessler_forcing_cube`/`_gridspace`) flatten with
   `reshape(-1, nlev)` over the horizontal axes — that **breaks under tile
   sharding** (a reshape across a sharded axis). So the COLUMN core must run
   per-tile INSIDE the shard_map (each tile reshapes only its OWN local columns).
   That core is `_kessler_column_tendencies` (atmosphere, `_`-private); core
   cannot import atmosphere. CLEAN FIX (dependency injection, keeps core
   physics-agnostic): (a) PROMOTE `_kessler_column_tendencies` ->
   `kessler_column_tendencies` (public, + update its 3 internal callers); (b) core
   `make_tiled_fv3_moist_tracer_tendency_stage_2d(mesh, cdgrid, n, kt, nlev, *,
   column_tracer_physics_fn)` = increment-2 advection + an INJECTED per-tile
   `column_tracer_physics_fn(T_t, p_s_t, q_v_t, q_c_t, q_r_t) -> (dq_v, dq_c,
   dq_r)` (built by the caller/test from `kessler_column_tendencies` with
   sigma_coord/dt/cfg closed over).  Serial reference (parity): the dynamics add
   is `_dtracers[k] += physics_tendency_cc.tracer_tendencies[k]`
   (primitive_eq_cdgrid.py:1195) so total = advection + kessler tracer rates.
   sigma-only (kessler raises on hybrid).  Gate: tiled combined dq_pack == serial
   (advective_tracer_tendency + kessler) at np24 bit-identity.
   **DONE (commit 7297b2237, gate 8532506 PASS).** `kessler_column_tendencies`
   promoted public; `make_tiled_fv3_moist_tracer_tendency_stage_2d` (advection +
   injected per-tile column physics); np24 bit-identity vs serial advection +
   kessler.
4. **Full moist tiled step. DONE + FULLY GATED (code 9ac698e42 codex-CLEAN; test
   fixes 9425e84bd; np24 full gate 8532552 = 16 passed; np54 kt=3 gate 8532570 =
   12 passed — moist_step[3] + tracer[3]/pack[3]/tendency[3], the strip-ppermute
   DIRECTION cases kt=2 cannot catch, all bit-identity).** `make_tiled_fv3_hydrostatic
   _moist_step_stage_2d` → `step(u_d, v_d, T, p_s, phis, q_pack) -> (..., q_pack)`
   + post-step tracer floor. Parity vs the serial base-cut moist RK3: u_d/v_d/T/p_s
   rel<1e-10, q_pack rel 3.6e-10 (Kessler nonlinear + 3-stage RK3 FMA reorder,
   bit-identity class). LESSON — all 3 initial gate failures were TEST-REFERENCE
   bugs, NOT tiled code: (i) the reference passed `dt_actual` → the iter-189
   corner-div adaptive cap (a shape error) the tiled base cut omits; (ii) D-grid
   CORNER fields u_d/v_d need a block-wise `_rel_corner` compare (the tiled output
   is kt blocks of nl+1 with the staggered shared edge duplicated, NOT n+1);
   (iii) q_pack rides nonlinear Kessler → looser (1e-8) FMA tol than the linear
   dynamics. The tiled code (the shared tracer-aware `_build_hydro_tile_tendency
   _fns` + `_ssp_rk3_tile_step`) was correct from the first build.

   **np24/54 bit-identity gate vs the serial moist step (the original plan):**
   ARCHITECTURE FINDING (scout 2026-06-20): NOT "compose increment 3 into the dry
   RK3" — the tracer VERTICAL advection is coupled to the dynamics' vertical
   velocity (`sigma_dot`/`mass_flux`), which is computed INSIDE the shared
   `_build_hydro_tile_tendency_fns::_tile_tendency` (the `_vadv_drive`/`_vadv`
   pair, also used for T's `vert_adv_T`).  And the serial moist step recomputes
   physics PER-RK-STAGE (`_step_fv3.tendency_fn` calls `physics_fn` each stage,
   feeding `physics_tendency_cc` into `fv3_hydrostatic_tendencies`; dT_dt already
   includes the kessler dT at primitive_eq_cdgrid.py:1188).  So increment 4 =
   EXTEND `_build_hydro_tile_tendency_fns` to be tracer-aware: when a `q_pack`
   (+ injected `column_tracer_physics_fn`) is given, additionally compute
   `dq_pack = -(u . grad q)` (the shared `_tiled_scalar_horiz_advect`, packed)
   + `_vadv(q, _vadv_drive)` (vertical, the SAME driver as T) + kessler tracer
   rates, and add the kessler dT to dT_dt; **dry path (`tracers=None`) returns
   the 4-tuple bit-identically (gated dry step/tendency UNCHANGED)**.  Then a
   `make_tiled_fv3_hydrostatic_moist_step_stage_2d` threads the 5-tuple
   (u_d,v_d,T,p_s,q_pack) through the SAME SSP-RK3 combine (generic pytree axpy
   so the dry step's explicit RK3 is not duplicated) + a post-step
   `jnp.maximum(q,0)` floor (serial floor primitive_eq_cdgrid.py:1758-1765).
   Reference: the serial cube `model.step_with_physics(state, dt, kessler)` base
   cut.  sigma-only (kessler).  DEEP + must preserve the dry gate — build as its
   own iteration.

## Non-goals / caveats

- NOT Ginsburg-benchable (np>6 anti-scales on CPU; the value is the CAPABILITY +
  the parity receipt, not a speed number — do NOT sell a microbench as a win).
- Numerics ONLY from the shared `*_core` ops + the existing tiled kernels; the moist
  path adds NO new dynamics numerics (Kessler is the only physics, already shared).
- Base cut matches the dry step (div_damp=0, hyperdiff=0, non-duogrid); optional
  terms are later increments, not base-case-blocking.
# MPAS-atmosphere NATIVE (ppermute SPMD) step — production-support audit (2026-07-13)

Scaling-M3c increment-1 deliverable. User directive audited: "Native
MPAS-atmosphere ppermute code exists but production uses blocking mpi4jax
and the native path lacks full production state/physics support."

**Verdict: correct on both counts (pre-increment).** The native path is
`make_voronoi_sharded_step` (`packages/core/legoesm/parallel/sharded_dynamics.py`,
shard_map + `jax.lax.ppermute`, single- and multi-controller); production
(ModelDriver, `packages/coupler/legoesm/driver/model_driver.py:5217`) drives
route-A `make_voronoi_mpi_step` (`packages/core/legoesm/parallel/voronoi_mpi.py:618`,
mpi4jax sendrecv — latency-bound, no comm/compute overlap). Route-A had full
production support; the native path had a dry-dynamics subset.

## Gap table (pre-increment state, line numbers at main 04265c79e)

Native = `make_voronoi_sharded_step` (sharded_dynamics.py:1721);
production reference = serial `MPASPrimitiveEquationModel._step_jit`
(`packages/atmosphere/legoesm/atmosphere/dynamics/primitive_eq_mpas.py:725`)
and route-A `make_voronoi_mpi_step` (voronoi_mpi.py:618).

| # | Item | Native (pre) | Production reference | Increment-1 |
|---|------|--------------|----------------------|-------------|
| 1 | Tracers in halo exchange | absent — cell pack = T/p_s/phis only (sharded_dynamics.py:1922-1927) | route-A batched union-neighbor exchange incl. all tracers (voronoi_mpi.py:778-832) | **CLOSED** — tracers ride the packed cell buffer in canonical sorted wire order |
| 2 | Tracer advection in RK | dropped (1984-2006; RK advances u/T/p_s only, 2099-2115) | pytree RK carries tracer advection tendencies (primitive_eq_mpas.py:764-787; voronoi_mpi.py:889-913) | **CLOSED** — tendency-shaped state pytree incl. tracers through `dispatch_integrator` |
| 3 | Integrator | hard-coded SSP-RK3 (2099-2115); config default `ssp_rk54_scan` silently overridden | `dispatch_integrator(cfg.time_integrator)` (primitive_eq_mpas.py:785; voronoi_mpi.py:935) | **CLOSED** — same dispatch as serial |
| 4 | Forcing | not threaded; physics called with `forcing=None` (2133) | traced jit arg, SegmentForcing doctrine (primitive_eq_mpas.py:731,750-751; voronoi_mpi.py:916-927) | **CLOSED** — traced jit arg; steady-state no-retrace gated |
| 5 | phys_state carry | refused loudly (2134-2146, 2192) | threaded + returned, `return_phys_state=True` (voronoi_mpi.py:945-959,1012-1027) | **CLOSED** — `return_phys_state=True` factory kwarg mirrors route-A; refusals kept loud |
| 6 | Physics tracer tendencies | not applied (2147-2149) | applied (voronoi_mpi.py:973-982) | **CLOSED** |
| 7 | Tracer non-negativity floor | missing (2152-2153 T only) | present (voronoi_mpi.py:990-994) | **CLOSED** |
| 8 | compute/storage precision cast | missing | both ends (voronoi_mpi.py:934,1004; serial 760,848) | **CLOSED** — `cast_pytree` mirror; f64 mass-budget accumulator added |
| 9 | Local-only metadata | ALL n_dev local meshes + ppermute index arrays REPLICATED per device (1858-1891); global `areaCell` replicated closure (2075) | route-A rank holds only `layout.local_mesh` (voronoi_mpi.py:721) | **CLOSED for dynamics** — stacked local meshes, halo schedule, and areaCell are `P("device")`-sharded jit ARGUMENTS (multi-controller-safe where sharded closures raise); each device holds only its slice. REMAINDER: the operator-split physics term (below) |
| 10 | Schema tripwire | none — a new HydrostaticState field silently unexchanged | ocean twin `_expected` set (voronoi_mpi.py:379-390) | **CLOSED** — `check_voronoi_spmd_state_schema` + synthetic-violation self-test |

## What landed (increment 1)

* `make_voronoi_sharded_step` rewritten (same public factory; signature
  gains `return_phys_state: bool = False`):
  full state through the packed ppermute/allgather exchange (one flat
  payload per neighbor round carrying u + T + p_s + phis + all tracers —
  the SPMD mirror of M3d's `exchange_state_mpas_ocean` batched pattern),
  `dispatch_integrator` dynamics with tracer advection, operator-split
  physics with traced `forcing` + prognostic `phys_state` carry, tracer
  floors, compute/storage casts, fp64 fused mass fix, local-only
  dynamics metadata via sharded jit args.
* `_ppermute_halo_fill` factored module-level so the exchange is
  independently testable (five-field sentinel routing gate).
* Bench `bench_mpas_spmd_scaling.py`: `--physics kessler` (moist BCW
  tracers on the timed/gated path; tracer parity gate).
* Gates: `tests/parallel/test_mpas_atm_native_step.py` (parity vs serial
  for moist kessler × {ppermute, allgather}, traced-forcing no-retrace,
  stateful-carry threading, integrator dispatch, sentinel routing,
  10-step scan dtype fixed point, refusal contracts, route-A np=1
  cross-check) + a second (moist) np=2 multicontroller selfspawn test.

## Honest remainders (NOT closed in increment 1)

1. **Physics-term metadata is not local-only.** The operator-split
   physics runs OUTSIDE shard_map on the GSPMD-sharded global state with
   the replicated global-mesh closure (`sharded_dynamics.py`, physics
   block in `_build_step`). Column-local AMIP physics reads only 1-D
   cell fields (latCell/lonCell/areaCell — O(nCells) scalars, not the
   2-D connectivity), so the memory cost is small, but a strict
   per-process-local physics would need the physics evaluated inside
   shard_map on the owned shard with per-device mesh columns. Deferred.
2. **Single-device `return_phys_state=True` is refused** (ValueError):
   `model.step` stashes the carry eagerly on the model and cannot honor
   the `(state, carry)` return contract under an outer trace (gh-417).
   Serial carry threading stays the model/driver's own contract.
3. **`anchor_mass_to_initial` is not implemented** on the native path
   (mass fix is the per-step telescoping old-vs-new correction) — same
   as route-A `make_voronoi_mpi_step`. With the fixer applied every
   step the trajectories agree to the reduction-order floor; the anchor
   only suppresses the fp random walk of the pinned value.
4. **ModelDriver still selects route-A** for MPAS MPI production runs.
   Swapping the driver to the native path (and the NCCL 8→16 GPU
   re-measurement from the audit) is the next increment.

## Parity envelope (measured, x64, subdiv 3, nlev 4, 2 devices)

Documented tolerance vs serial (floating-point re-association floor of
the sharded step — halo-cell operator coverage + mass-fix allreduce
ordering; same envelope as `test_voronoi_sharded_equivalence.py`):
u/T atol 1e-6, p_s atol 1e-1, tracers atol 1e-9. The np=2
multicontroller moist bench gate passes with the same
`MPAS_PARITY_TOLS` (T-tier tolerances for tracers, atol scaled 1e-3).

Measured (job 8970923: moist BCW + kessler, sfc partition, 4 steps,
dt=2400 s, x64, 2 virtual CPU devices, `--parity-gate
--check-conservation`):

```
conservation dry-mass: rel drift = 0.000e+00  (tol 1e-11)
parity   u: max|diff| = 1.631e-09
parity   T: max|diff| = 2.853e-09
parity p_s: max|diff| = 3.485e-07
parity q_v: max|diff| = 6.508e-14   (q_c, q_r exactly 0)
per-step ms: [1359 (compile), 1097 (output-layout promotion compile),
              3.7, 3.5 (steady)]
```

The second compile is the one-time input→output sharding-layout
promotion also pinned by the no-retrace gate in
`test_mpas_atm_native_step.py` (steady-state steps with changing
forcing values add ZERO traces).

## Codex adversarial review (gpt-5.6-sol high, job 8970917)

3 MAJOR + 2 MINOR, all addressed same-increment:

1. MAJOR (fp32 carry promotion): the fp64 mass correction promoted an
   fp32 `p_s` under x64 (the downcast-skipping storage cast cannot undo
   it → `lax.scan` carry-dtype break). FIX: apply the correction in
   fp64, then `.astype` back to the pre-fix carry dtype (bit-identical
   whenever compute is fp64); gated by
   `test_fp32_state_dtype_fixed_point_under_x64` (all-fp32 state+mesh
   under x64, every output leaf stays float32).  NB the serial
   `_fix_mass_mpas_hydro` deliberately leaves the promoted add
   (iter-11) — parity in that corner differs only by the correction-add
   rounding.
2. MAJOR (areaCell resharding): a caller handing a REPLICATED mesh
   (bench `replicate_pytree`) left `_area_for_mass` replicated under
   multi-controller (`multiprocess_safe_device_put` passes
   non-fully-addressable arrays through). FIX: host `np.asarray` copy
   before placement — always fully addressable, P("device") guaranteed.
3. MAJOR (cross-process ppermute untested): auto strategy picks
   allgather at the selfspawn gate size. FIX: bench `--halo-strategy`
   flag (recorded in the JSONL row); the moist selfspawn np=2 test
   FORCES ppermute and asserts the recorded strategy.
4. MINOR (sentinel bypassed production pack): factored
   `_pack_cell_state`/`_unpack_cell_state` as the single wire-layout
   source used by the kernel AND the sentinel test; test now checks
   NAMED fields through the production helpers, nonzero phis, reversed
   tracer insertion order vs sorted wire order, dry zero-width block,
   and a 3-device multi-round schedule.
5. MINOR (weak physics-application gate): added
   `test_physics_tracer_tendencies_partial_key` — a one-tracer toy
   physics moves exactly `q_c` by dt·rate (three orders above the
   tracer tolerance) while `q_v`/`q_r` track the dynamics-only
   trajectory; the Kessler advection non-vacuity threshold raised
   above the tracer parity atol.

Round 2 (job 8971595): ALL five round-1 findings verified CLOSED, "no
new production correctness defect".  Remaining items: (a) MAJOR-scoped
restatement of remainder #1 below (the operator-split physics closure
retains the full replicated global mesh — connectivity included — not
just the 1-D fields it reads; increment 2 moves column physics inside
shard_map with a narrowed sharded physics-mesh structure); (b) MINOR:
JSONL rows now record ``halo_strategy_requested`` AND
``halo_strategy_effective`` (the factory carries the post-"auto"
resolution on the returned callable, kessler ``_bound_dt`` pattern);
(c) NIT: the Kessler advection non-vacuity check now asserts
``not allclose`` against the FULL parity envelope.

Pre-existing (NOT this increment; flagged for a separate fix PR): the
`test_no_private_cross_imports` ratchet fails on MAIN content —
`tiled_production_cdgrid.py:2442` imports `_get_tiled_tables` /
`_tiled_diag_perms` / `_tiled_guard_perms` from `cubesphere_exchange`
(commit b52aa1220, PR #971); `test_no_module_top_jax_alloc` fails on
MAIN (`sdm.init._SQRT2`, `sdm.kernels._HALL_*`,
`training.neural_gcm_spectral._SFNO_FORCING_INPUT_SCALE`).  Verified by
running the ratchet scanners against pristine-main file content.

exec
/usr/bin/bash -lc "find docs/performance/scaling results -type f \\( -name '*mcut*' -o -name '*precond*' -o -name '*scaling*csv' \\) -maxdepth 4 -print | head -100 && sed -n '1,120p' docs/performance/scaling/barotropic_mcut.csv && if [ -f docs/performance/scaling/barotropic_mcut_walltime.csv ]; then sed -n '1,120p' docs/performance/scaling/barotropic_mcut_walltime.csv; fi && rg -n -i \"chebyshev.*(wall|negative|speed|m=|reduction)|mcut|multigrid.*wall|standalone|reduction.free\" docs/performance/scaling docs/performance | head -300" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 572ms:
docs/performance/scaling/scaling_indicators.csv
docs/performance/scaling/barotropic_mcut_np2.csv
docs/performance/scaling/barotropic_mcut_np4.csv
docs/performance/scaling/barotropic_mcut.csv
preconditioner,M,rel_residual,reductions_per_step,n_ranks,n_lat,n_lon
jacobi,2,2.910939e-01,4,2,96,192
jacobi,4,1.327163e-01,8,2,96,192
jacobi,6,7.333619e-02,12,2,96,192
jacobi,8,4.753396e-02,16,2,96,192
jacobi,12,2.719353e-02,24,2,96,192
jacobi,20,1.505235e-02,40,2,96,192
jacobi,40,6.117104e-03,80,2,96,192
jacobi,60,2.016396e-03,120,2,96,192
multigrid,2,2.009663e-02,4,2,96,192
multigrid,4,9.981725e-04,8,2,96,192
multigrid,6,6.285739e-05,12,2,96,192
multigrid,8,3.848826e-06,16,2,96,192
multigrid,12,9.244435e-09,24,2,96,192
multigrid,20,7.987835e-14,40,2,96,192
multigrid,40,1.410610e-16,80,2,96,192
multigrid,60,1.410610e-16,120,2,96,192
docs/performance/scream_parity_scope.md:154:to the 2D `laplacian_compact` (4 halos → 1); fold the standalone div-damp
docs/performance/scaling/scaling_crm_gpu.md:388:**Standalone timing (n_cols=16384, n_sys=29, fp32):**
docs/performance/scaling/scaling_crm_gpu.md:657:Standalone cuSPARSE timing (n_cols=16384, nlev=30, fp32):
docs/performance/scaling/scaling_theoretical_limit_report_2026-06-15.md:10:> 1. **Split-explicit barotropic + reduction-free local eta-floor clamp**
docs/performance/scaling/scaling_theoretical_limit_report_2026-06-15.md:100:| Banded multigrid barotropic (wall-time) | converges (M12→1e-6) but **6–9× SLOWER** — zonal-line V-cycle compute ≫ jacobi diagonal; reductions weren't the bottleneck | 8488551 |
docs/performance/scaling/scaling_theoretical_limit_report_2026-06-15.md:10:> 1. **Split-explicit barotropic + reduction-free local eta-floor clamp**
docs/performance/scaling/scaling_theoretical_limit_report_2026-06-15.md:100:| Banded multigrid barotropic (wall-time) | converges (M12→1e-6) but **6–9× SLOWER** — zonal-line V-cycle compute ≫ jacobi diagonal; reductions weren't the bottleneck | 8488551 |
docs/performance/scaling/derecho_levante_sota_review_2026-07.md:82:- Barotropic: split-explicit (halo-only) + reduction-free local clamp opt-in ≡
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md:20:  (Chronopoulos-Gear single_reduce + Chebyshev reduction-free precond). The
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md:53:### 1. JAX geometric/algebraic-multigrid (or RAS) preconditioner for the implicit barotropic solve — Wall A/B. HIGHEST ROI.
docs/performance/scaling/scaling_indicators.csv:19:2026-06-13,0293f865,chebyshev,ocean_latlon,mpi,precond,2.31,speedup_iters_d8,8478404,cheby deg8 35 iters to rel1e-6 vs jacobi >80 (budget-censored lower bound); coastal 48x96; cuts latency-bound global reductions >=2.3x; deg4=72iters
docs/performance/scaling/scaling_indicators.csv:41:2026-06-15,2786274b,baro_multigrid,ocean_latlon,mpi,reductions,24,baro_reductions_per_step,8487973,Banded distributed MULTIGRID barotropic reaches 1e-6 at M=12 np2 = 24 reductions/step (M=20=40 np4); V-cycle adds ZERO global reductions (banded==serial mpirun -np 2/4) — 5x-and-CONVERGING vs jacobi 120-and-stuck; reduction-latency-wall lever task #26
docs/performance/scaling/scaling_indicators.csv:45:2026-06-15,c2b1a0e0,baro_chebyshev_RETRACTED,ocean_latlon,mpi,speedup,0.0,baro_walltime_speedup,8489419,RETRACTED — chebyshev deg-4 DIVERGES at LL192 (residual M20=0.97 M60=0.25; job 8489419). The earlier '1.23x' (8488773) timed it WITHOUT checking residual = a fast NON-SOLVE. degree-4 cannot span the polar-anisotropic condition number. NOT a win. Lesson: always check residual before claiming a solver speedup. See docs/scaling/barotropic_multinode_verdict_2026-06-15.md
docs/performance/scaling/scaling_indicators.csv:46:2026-06-15,c2b1a0e0,baro_verdict_mn,ocean_latlon,mpi,reductions,1,baro_multinode_verdict,8489419,VERDICT (LL192 np32 residual sweep): ONLY the banded MG truly converges (M20 2e-8) — jacobi crawls (M60 6.6e-3) chebyshev diverges. MG wall-time-negative only b/c jacobi 'wins' by NOT converging. No cheap wall-time fix for the reduction wall at production accuracy; jacobi near-optimal; single_reduce ~1.05x clean (half reductions same conv); MG=accuracy-only lever
docs/performance/scaling/scaling_indicators.csv:63:2026-06-15,205780b0,baro_solver_h2h,ocean_latlon,mpi,strong,1.23,explicit_clamp_over_implicit_np32,8498971,DEFINITIVE same-job head-to-head LL192 f64 strong: implicit_cn vs explicit_substep+local-clamp. np8 133.94 vs 143.91 (implicit 1.07x - 1 node subcycle-compute-bound) np16 77.44 vs 67.44 (EXPLICIT 1.15x) np32 64.73 vs 52.52ms (EXPLICIT 1.23x). CROSSOVER ~np16: split-explicit+local-clamp = better multi-node ocean solver at >=2 nodes, win GROWS with rank (reduction-free subcycle). Strong eff np32 explicit 0.69 vs implicit 0.52. MOM6/MPAS-O SOTA design confirmed on Ginsburg
docs/performance/scaling/scaling_indicators.csv:71:2026-06-17,2d9ed627c,tiled_3d_c2d_vector,atm_cube,spmd,capability,12,ops3d_np24,8512822,Tiled center_to_dgrid_vector lift (cc wind vector -> D-grid corners) = the section-12c vert_adv_uv_d momentum contribution the standalone momentum stage omits (added unconditionally at primitive_eq_cdgrid.py:976-977). NEW tiled op = halo=1 ROTATING vector pad (ndim=4 make_tiled_pad_vector_body extended 1369e503b; ndim=3 SW path byte-identical) + 4-pt corner avg -> ops3d 11->12. Bit-identity np24/np54 corner overlap + host-body 3 passed sbatch 8512822. Prereq for the full hydrostatic capstone (compose mom+lift+continuity+thermo+vertical). codex-clean Jun 18 (3 findings fixed: mesh/kt+nl>=2 guards 0f50c4afe, f32 delta-first fix_ps_mass b66f82bb7)
docs/performance/scaling/literature_scan_2026-06-13_new_levers.md:20:  (Chronopoulos-Gear single_reduce + Chebyshev reduction-free precond). The
docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md:53:### 1. JAX geometric/algebraic-multigrid (or RAS) preconditioner for the implicit barotropic solve — Wall A/B. HIGHEST ROI.
docs/performance/scaling/scaling_crm_gpu.md:388:**Standalone timing (n_cols=16384, n_sys=29, fp32):**
docs/performance/scaling/scaling_crm_gpu.md:657:Standalone cuSPARSE timing (n_cols=16384, nlev=30, fp32):
docs/performance/scaling/derecho_levante_sota_review_2026-07.md:82:- Barotropic: split-explicit (halo-only) + reduction-free local clamp opt-in ≡
docs/performance/scaling/scaling_indicators.csv:19:2026-06-13,0293f865,chebyshev,ocean_latlon,mpi,precond,2.31,speedup_iters_d8,8478404,cheby deg8 35 iters to rel1e-6 vs jacobi >80 (budget-censored lower bound); coastal 48x96; cuts latency-bound global reductions >=2.3x; deg4=72iters
docs/performance/scaling/scaling_indicators.csv:41:2026-06-15,2786274b,baro_multigrid,ocean_latlon,mpi,reductions,24,baro_reductions_per_step,8487973,Banded distributed MULTIGRID barotropic reaches 1e-6 at M=12 np2 = 24 reductions/step (M=20=40 np4); V-cycle adds ZERO global reductions (banded==serial mpirun -np 2/4) — 5x-and-CONVERGING vs jacobi 120-and-stuck; reduction-latency-wall lever task #26
docs/performance/scaling/scaling_indicators.csv:45:2026-06-15,c2b1a0e0,baro_chebyshev_RETRACTED,ocean_latlon,mpi,speedup,0.0,baro_walltime_speedup,8489419,RETRACTED — chebyshev deg-4 DIVERGES at LL192 (residual M20=0.97 M60=0.25; job 8489419). The earlier '1.23x' (8488773) timed it WITHOUT checking residual = a fast NON-SOLVE. degree-4 cannot span the polar-anisotropic condition number. NOT a win. Lesson: always check residual before claiming a solver speedup. See docs/scaling/barotropic_multinode_verdict_2026-06-15.md
docs/performance/scaling/scaling_indicators.csv:46:2026-06-15,c2b1a0e0,baro_verdict_mn,ocean_latlon,mpi,reductions,1,baro_multinode_verdict,8489419,VERDICT (LL192 np32 residual sweep): ONLY the banded MG truly converges (M20 2e-8) — jacobi crawls (M60 6.6e-3) chebyshev diverges. MG wall-time-negative only b/c jacobi 'wins' by NOT converging. No cheap wall-time fix for the reduction wall at production accuracy; jacobi near-optimal; single_reduce ~1.05x clean (half reductions same conv); MG=accuracy-only lever
docs/performance/scaling/scaling_indicators.csv:63:2026-06-15,205780b0,baro_solver_h2h,ocean_latlon,mpi,strong,1.23,explicit_clamp_over_implicit_np32,8498971,DEFINITIVE same-job head-to-head LL192 f64 strong: implicit_cn vs explicit_substep+local-clamp. np8 133.94 vs 143.91 (implicit 1.07x - 1 node subcycle-compute-bound) np16 77.44 vs 67.44 (EXPLICIT 1.15x) np32 64.73 vs 52.52ms (EXPLICIT 1.23x). CROSSOVER ~np16: split-explicit+local-clamp = better multi-node ocean solver at >=2 nodes, win GROWS with rank (reduction-free subcycle). Strong eff np32 explicit 0.69 vs implicit 0.52. MOM6/MPAS-O SOTA design confirmed on Ginsburg
docs/performance/scaling/scaling_indicators.csv:71:2026-06-17,2d9ed627c,tiled_3d_c2d_vector,atm_cube,spmd,capability,12,ops3d_np24,8512822,Tiled center_to_dgrid_vector lift (cc wind vector -> D-grid corners) = the section-12c vert_adv_uv_d momentum contribution the standalone momentum stage omits (added unconditionally at primitive_eq_cdgrid.py:976-977). NEW tiled op = halo=1 ROTATING vector pad (ndim=4 make_tiled_pad_vector_body extended 1369e503b; ndim=3 SW path byte-identical) + 4-pt corner avg -> ops3d 11->12. Bit-identity np24/np54 corner overlap + host-body 3 passed sbatch 8512822. Prereq for the full hydrostatic capstone (compose mom+lift+continuity+thermo+vertical). codex-clean Jun 18 (3 findings fixed: mesh/kt+nl>=2 guards 0f50c4afe, f32 delta-first fix_ps_mass b66f82bb7)
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:25:| Ocean strong (mitigation) | `single_reduce` (Chronopoulos–Gear) → 1 all-reduce/iter; split-explicit + `barotropic_local_subcycle_clamp` → 3 all-reduce/step (reduction-free subcycle) | both exist, both NON-default; beats standard at ≥2 nodes 1.15–1.65× | CONFIRMED |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:99:   the reduction-free path where per-message latency dominates.
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:25:| Ocean strong (mitigation) | `single_reduce` (Chronopoulos–Gear) → 1 all-reduce/iter; split-explicit + `barotropic_local_subcycle_clamp` → 3 all-reduce/step (reduction-free subcycle) | both exist, both NON-default; beats standard at ≥2 nodes 1.15–1.65× | CONFIRMED |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:99:   the reduction-free path where per-message latency dominates.
docs/performance/scaling/distance_to_limit_2026-06-13.md:92:  reduction-free mechanism. Opt-in.
docs/performance/scaling/distance_to_limit_2026-06-13.md:92:  reduction-free mechanism. Opt-in.

exec
/usr/bin/bash -lc "sed -n '480,760p' packages/ocean/legoesm/ocean/dynamics/barotropic_common.py && sed -n '1530,1640p' packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py && sed -n '560,740p' packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py && rg -n \"fixed_iters|initial_guess|x0|eta_guess|eta_old|solve_helmholtz_implicit\" packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py | head -180" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign
 succeeded in 293ms:
                "is set; arm it via activate_latlon_spmd_halo(mesh).")
        # Route to psum ONLY for the lat-band ocean SPMD mesh, keyed on the
        # ``"lat"`` axis BY NAME (activate_latlon_spmd_halo guarantees it).
        # The cube atm SPMD backend ALSO sets backend=="spmd" but with a
        # ``("face", ...)`` mesh; in a coupled run that mesh could be armed
        # while this ocean barotropic PCG runs, and psum'ing over a
        # non-lat (or replicated) axis would multiply the dots by the
        # device count or crash (codex HIGH).  When the armed SPMD mesh is
        # not the lat-band one, fall through to the MPI/local logic below
        # (ocean fields are never cube-sharded, so the local/allreduce sum
        # is the correct reduction there).
        if "lat" in tuple(mesh.axis_names):
            from legoesm.parallel.reductions import batch_psum_spmd
            return batch_psum_spmd(local, "lat")
    # Function-scope import: ``reductions`` pulls in mpi4jax lazily and
    # ``core.operators`` (cross-package), so keep it out of module top.
    from legoesm.parallel.reductions import (
        batch_allreduce_mpi,
        is_multi_process,
    )
    if not is_multi_process():
        return local
    return batch_allreduce_mpi(local, op="sum")


def _fixed_iteration_pcg(
    A_op: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    M_inv: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    *,
    max_iter: int,
    dot_weight: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Jacobi-preconditioned CG run for EXACTLY ``max_iter`` iterations.

    Standard preconditioned conjugate gradient (Shewchuk 1994, Alg. B3)
    with a STATIC iteration count: no residual-dependent ``while_loop``,
    so all MPI ranks execute the identical collective schedule.

    Reductions per iteration (FIXED, resolution-independent):
      * ``p·Ap`` (for α) — one batched reduction; then
      * ``r·z`` AND ``r·r`` (for β and the residual monitor) — folded
        into ONE batched reduction, since both need the post-α ``r``.
    => two batched ``allreduce(SUM)`` per iteration (the residual costs
    no extra message).  Under a single process :func:`_global_dot_batch`
    is a no-op reduction, so the *algorithm* is identical with/without
    MPI (this is what makes the single-rank equivalence test runnable in
    one process).

    Grid-agnostic: ``A_op`` and ``M_inv`` are opaque callables operating
    on whatever array shape the caller uses ((n_lat, n_lon) for lat-lon,
    (nCells,) for MPAS).  ``A_op`` owns the halo exchange; ``M_inv`` is
    communication-free (diagonal Jacobi).

    Safety floors (``1e-30``) guard the CG scalar divisions on the very
    first land-only or zero-rhs columns; they are math safeguards, not
    tunable parameters (CLAUDE.md "safety floors exempt").

    Returns
    -------
    (x, rr) : the solution after ``max_iter`` iterations and the final
        GLOBAL ``r·r`` (squared residual norm).  ``r·r`` is folded into
        the per-iteration ``r·z`` reduction (no extra message), so it is
        a free self-monitoring signal — ``solve_helmholtz_implicit``
        consumes this ``rr`` directly for the relative-residual
        diagnostic (no second ``A_op``).
    """
    # ``dot_weight`` (optional): weighted/owned-masked dots for every CG
    # scalar — REQUIRED on partitioned unstructured meshes whose local
    # arrays carry HALO entries (an unweighted local sum double-counts
    # them in the allreduce; pass owned_mask·area).  ``None`` keeps the
    # historical Euclidean dots bit-exactly (the lat-lon band path,
    # whose rows partition without overlap).
    if dot_weight is None:
        def _dotw(a):
            return a
    else:
        def _dotw(a):
            return a * dot_weight

    r0 = b - A_op(x0)
    z0 = M_inv(r0)
    # Initial r·z and r·r (one batched reduction).
    rz0, rr0 = _global_dot_batch([(_dotw(r0), z0), (_dotw(r0), r0)])

    # FREEZE threshold for the CG scalar divisions, RELATIVE to the
    # initial r·z (all of rz/pAp live in the same units as rz0).  Once
    # the residual hits the working-precision floor the unrolled CG keeps
    # running its remaining static iterations on a ~converged state;
    # there ``rz``/``pAp`` are at round-off noise and ``rz_new/rz`` is a
    # noisy ``0/0`` whose REVERSE-MODE value blows up to NaN over many
    # iterations.  ``rz`` is a residual-SQUARED quantity, so its noise
    # floor is ~``rz0 * eps^2`` (the residual relative floor is ~eps).
    # Freezing at ``rz0 * eps^2`` therefore engages exactly at the
    # round-off plateau (f32: ~rz0*1e-14, matching the observed noise;
    # f64: ~rz0*5e-32, far below the 1e-10 convergence target so f64
    # accuracy is unaffected) and zeroes the already-converged update
    # with a finite gradient.  (A fixed ``1e-30`` floor never engages in
    # f32; a ``rz0*eps`` floor freezes f64 prematurely.)
    #
    # ACCEPTED NONDIFFERENTIABLE POINT: if ``x0`` ALREADY solves
    # ``A x0 = b`` exactly (``r0 = 0`` => ``rz0 = 0`` => ``den_floor =
    # tiny``), every ``_safe_div`` freezes and the loop returns ``x0`` with
    # ZERO gradient sensitivity through the solve — whereas the true
    # implicit derivative at ``rhs = A x0`` is nonzero.  For the lat-lon
    # caller ``rhs - A·eta_old = dt·(F_slow_eta·mask - div_HU_pred)``, so
    # ``r0 = 0`` is reachable at an EXACT rest state with no slow forcing
    # and zero predicted transport divergence (or exact cancellation) —
    # rare but not impossible.  At such a point the solve is already at
    # its answer and the zeroed gradient is a measure-zero degeneracy, not
    # a bias on a generic trajectory; a forward run is unaffected (the
    # output ``x0`` is correct).  Not worth a special-cased zero-residual
    # adjoint; documented so a caller differentiating through a perfectly
    # quiescent step knows the gradient there is degenerate.
    finfo = jnp.finfo(b.dtype)
    rz0_mag = jnp.abs(rz0)
    den_floor = jnp.maximum(rz0_mag * finfo.eps * finfo.eps, finfo.tiny)

    def _safe_div(num, den):
        # GRADIENT-safe guarded division: num/den when |den|>den_floor,
        # else 0.  The "double where" — dividing by
        # ``where(cond, den, 1)`` (NOT the floor) — keeps BOTH the forward
        # AND the reverse-mode value finite (dividing by the floor would
        # make the dead branch's gradient ``-num/floor^2`` overflow).
        cond = jnp.abs(den) > den_floor
        safe_den = jnp.where(cond, den, jnp.ones_like(den))
        return jnp.where(cond, num / safe_den, jnp.zeros_like(num))

    class _CGState(NamedTuple):
        x: jnp.ndarray
        r: jnp.ndarray
        p: jnp.ndarray
        rz: jnp.ndarray
        rr: jnp.ndarray

    def body(_i: int, st: _CGState) -> _CGState:
        Ap = A_op(st.p)
        # Reduction 1/iter: p·Ap (needed for α).
        (pAp,) = _global_dot_batch([(_dotw(st.p), Ap)])
        alpha = _safe_div(st.rz, pAp)
        x_new = st.x + alpha * st.p
        r_new = st.r - alpha * Ap
        z_new = M_inv(r_new)
        # Reduction 2/iter: r·z (for β) AND r·r (residual monitor),
        # batched into one message.
        rz_new, rr_new = _global_dot_batch(
            [(_dotw(r_new), z_new), (_dotw(r_new), r_new)],
        )
        beta = _safe_div(rz_new, st.rz)
        p_new = z_new + beta * st.p
        return _CGState(x=x_new, r=r_new, p=p_new, rz=rz_new, rr=rr_new)

    init = _CGState(x=x0, r=r0, p=z0, rz=rz0, rr=rr0)
    final = jax.lax.fori_loop(0, int(max_iter), body, init)
    return final.x, final.rr


def _fixed_iteration_pcg_single_reduce(
    A_op: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    M_inv: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    *,
    max_iter: int,
    dot_weight: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-reduction fixed-M PCG (Chronopoulos & Gear 1989 recurrences).

    Mathematically equivalent (in exact arithmetic) to
    :func:`_fixed_iteration_pcg`, but restructured so each iteration
    issues ONE batched ``allreduce`` (3 scalars: ``r·z``, ``w·z``,
    ``r·r``) instead of two sequentially-dependent reductions — the
    multi-node weak-scaling lever (probe 8460255: np32 weak growth
    ×7.9 is allreduce-latency-driven; this halves the per-step
    reduction count from ``2M+1`` to ``M+1``).  Costs one extra carried
    vector (``s = A p``) and one extra axpy per iteration.

    Recurrences (preconditioned CG-CG form; ``z = M⁻¹ r``, ``w = A z``):

        β_j = ρ_j / ρ_{j-1}              ρ_j = r_j·z_j
        t_j = μ_j − β_j² · t_{j-1}       μ_j = w_j·z_j   (t_j ≡ p_j·A p_j)
        α_j = ρ_j / t_j

    (β SQUARED: ``p_j·Ap_j = z_j·w_j + 2β (z_j·A p_{j-1}) + β² t_{j-1}``
    and ``z_j·A p_{j-1} = −ρ_j/α_{j-1} = −β t_{j-1} · ρ_j/ρ_{j-1}·...``
    collapses the cross term to ``−2β² t_{j-1}`` — a β¹ recurrence
    diverges, caught by the equivalence gate in job 8460547.)

    INNER PRODUCT (codex 2026-06-11 CRITICAL): every CG scalar here is
    the AREA-WEIGHTED dot ``⟨a,b⟩_W = Σ a·W·b`` with ``W =
    dot_weight`` (masked cell area).  The recurrence's cross-term
    identity ``⟨Az, p⟩ = ⟨z, Ap⟩`` requires A self-adjoint in the dot
    being used — the FV Helmholtz carries a ``1/area`` divergence
    factor and is self-adjoint ONLY in the W-inner product (``Aᵀ = W A
    W⁻¹``, pinned by the unit suite); the diagonal Jacobi ``M⁻¹`` is
    self-adjoint in W too.  Euclidean dots (the standard body's
    choice) are safe THERE because standard PCG measures ``p·Ap``
    directly each iteration instead of reconstructing it; here a
    Euclidean reconstruction silently biases α on any varying-area
    grid.  ``dot_weight`` is therefore REQUIRED — the dispatch refuses
    ``single_reduce`` without it rather than falling back to a wrong
    inner product.
        p_j = z_j + β_j p_{j-1}
        s_j = w_j + β_j s_{j-1}          (≡ A p_j — no second matvec)
        x_{j+1} = x_j + α_j p_j
        r_{j+1} = r_j − α_j s_j

    Carrying ``t = p·Ap`` directly avoids dividing by a possibly-frozen
    ``α`` and inherits the SAME ``_safe_div`` round-off-plateau freeze
    (and its documented zero-residual gradient degeneracy) as the
    standard body.  In floating point the iterates differ from standard
    PCG at round-off order; the solver-tolerance gates (not bit-exact
    ones) apply — see ``PARITY_TOLS_PCG_MPI``'s rationale.

    Returns ``(x, rr)`` with ``rr`` = final global ``r·r``, same
    contract as :func:`_fixed_iteration_pcg`.
    """
    W = dot_weight
    r0 = b - A_op(x0)
    z0 = M_inv(r0)
    w0 = A_op(z0)
    # ONE batched init reduction: ρ0, μ0 in the W-inner product, plus
    # the residual monitor — ALSO W-weighted: on halo-carrying
    # partitioned meshes (MPAS) an unweighted local r·r double-counts
    # halo entries in the allreduce; on lat-lon this makes the
    # diagnostic the area-weighted norm (documented, conservative).
    rho0, mu0, rr0 = _global_dot_batch(
        [(r0 * W, z0), (w0 * W, z0), (r0 * W, r0)],
    )

    finfo = jnp.finfo(b.dtype)
    rho0_mag = jnp.abs(rho0)
    den_floor = jnp.maximum(rho0_mag * finfo.eps * finfo.eps, finfo.tiny)

    def _safe_div(num, den):
        cond = jnp.abs(den) > den_floor
        safe_den = jnp.where(cond, den, jnp.ones_like(den))
        return jnp.where(cond, num / safe_den, jnp.zeros_like(num))

    class _CGSRState(NamedTuple):
        x: jnp.ndarray
        r: jnp.ndarray
        p: jnp.ndarray
        s: jnp.ndarray      # A p, maintained by recurrence
        rho: jnp.ndarray    # r·z
        t: jnp.ndarray      # p·A p, maintained by recurrence
        alpha: jnp.ndarray
        rr: jnp.ndarray

    def body(_i: int, st: _CGSRState) -> _CGSRState:
        # Apply the PREVIOUS iteration's α (deferred so the new scalars
        # of this iteration come from one reduction below).
        x_new = st.x + st.alpha * st.p
        r_new = st.r - st.alpha * st.s
        z_new = M_inv(r_new)
        w_new = A_op(z_new)
        # The single batched reduction of the iteration (W-dots for
        # the CG scalars AND the r·r monitor — see the init comment).
        rho_new, mu_new, rr_new = _global_dot_batch(
            [(r_new * W, z_new), (w_new * W, z_new), (r_new * W, r_new)],
        )
        beta = _safe_div(rho_new, st.rho)
        t_new = mu_new - beta * beta * st.t
        alpha_new = _safe_div(rho_new, t_new)
        p_new = z_new + beta * st.p
        s_new = w_new + beta * st.s
        return _CGSRState(
            x=x_new, r=r_new, p=p_new, s=s_new,
            rho=rho_new, t=t_new, alpha=alpha_new, rr=rr_new,
        )

    t0 = mu0
    alpha0 = _safe_div(rho0, t0)
    init = _CGSRState(
        x=x0, r=r0, p=z0, s=w0, rho=rho0, t=t0, alpha=alpha0, rr=rr0,
    )
    # Each body call applies one α-update then prepares the next α —
    # ``max_iter`` calls ⇒ exactly ``max_iter`` x/r updates and
    # ``max_iter + 1`` reductions total (incl. init), vs ``2·max_iter
    # + 1`` for the standard body.  The final iteration's prepared
    #   halo ``_sendrecv_vjp`` custom_vjp has no transpose rule — see
    #   barotropic_common's module note.)
    from legoesm.core.operators import is_distributed as _is_distributed
    from legoesm.ocean.dynamics.barotropic_common import (
        HelmholtzSolveDiagnostics,
        global_rel_residual as _rel_resid,
        precision_aware_rel_tol,
        solve_helmholtz_implicit,
    )
    _area_eta = grid.area.astype(eta_dtype)
    # Floor the relative-residual tolerance to what the working dtype can
    # reach (f64: 1e-10 default unchanged; f32: raised above ~1.2e-4 since a
    # 1e-10 rel-residual is unreachable below f32 machine epsilon).  Keeps the
    # converged diagnostic meaningful and the stock-CG while_loop terminating.
    _residual_tol = precision_aware_rel_tol(
        config.barotropic.barotropic_implicit_pcg_residual_tol, eta_dtype,
    )
    # ``force_pcg`` selects the fixed-M PCG body even single-rank
    # (solver-matched parity references + the faster-single-rank
    # option, job 8458701); its global dots reduce locally when not
    # multi-process, so the flag is safe pre-arming.
    # The lat-band SPMD backend (single-controller shard_map) is not
    # multi-PROCESS, so _is_distributed() is False — but the stock-CG branch
    # (solve_helmholtz_freesurface -> jax.scipy.sparse.linalg.cg) runs its
    # A_op halo ppermute + CG reductions inside a DATA-DEPENDENT while_loop,
    # whose collectives have no static schedule under shard_map (SIGABRT).
    # Route SPMD to the FIXED-iteration distributed PCG (static scan schedule,
    # SPMD-routed reductions — the same path MPI uses), exactly like the
    # explicit_substep barotropic loop that is already SPMD-validated.
    from legoesm.grids.halo import get_halo_backend as _get_halo_backend
    _spmd_armed = _get_halo_backend() == "spmd"
    _use_pcg = (_is_distributed() or _spmd_armed
                or bool(config.barotropic.barotropic_implicit_force_pcg))
    if not _use_pcg:
        # The stock-CG branch solves with its INTERNAL Jacobi (the
        # custom-VJP solver owns inv_diag for its exact adjoint) — a
        # non-default preconditioner cannot take effect here.  Refuse
        # loudly instead of silently running Jacobi (codex review MAJOR,
        # 2026-06-12): the fixed-M PCG honors it — set
        # barotropic_implicit_force_pcg=True for single-rank runs.
        _precond_req = str(getattr(
            config.barotropic, "barotropic_implicit_preconditioner", "jacobi"))
        if _precond_req != "jacobi":
            raise ValueError(
                "barotropic_implicit_latlon_cgrid: "
                f"barotropic_implicit_preconditioner={_precond_req!r} "
                "only applies to the fixed-M PCG path, but this "
                "single-rank run dispatches the stock-CG solver "
                "(internal Jacobi).  Set "
                "barotropic_implicit_force_pcg=True (PCG is also the "
                "faster single-rank solver, job 8458701) or use "
                "preconditioner='jacobi'."
            )
        # f32: a 1e-10 rel-tol is unreachable, so stock CG would run to
        # maxiter every step — floor it to the f32-reachable value (f64
        # unchanged).
        pcg_tol = precision_aware_rel_tol(
            config.barotropic.barotropic_implicit_pcg_tol, eta_dtype,
        )
        eta_new = solve_helmholtz_freesurface(
            rhs, eta_old, H_u_old, H_v_old, coeff, mask, u_mask, v_mask,
            inv_diag, grid, tol=pcg_tol,
            maxiter=int(config.barotropic.barotropic_implicit_pcg_maxiter),
        )
        # Same diagnostic contract as the solve_helmholtz_implicit stock
        # branch (global rel-residual + converged flag), so the
        # runtime-check consumer below is branch-uniform.
        _rel = _rel_resid(A_op, eta_new, rhs)
        _solve_diag = HelmholtzSolveDiagnostics(
            rel_residual=_rel, converged=_rel <= _residual_tol,
        )
    else:
        eta_new, _solve_diag = solve_helmholtz_implicit(
            A_op, rhs, M_inv, eta_old,
            distributed=True,
            fixed_iters=int(config.barotropic.barotropic_implicit_pcg_fixed_iters),
            residual_tol=_residual_tol,
            stock_cg_tol=config.barotropic.barotropic_implicit_pcg_tol,
            stock_cg_maxiter=int(config.barotropic.barotropic_implicit_pcg_maxiter),
            pcg_variant=str(config.barotropic.barotropic_implicit_pcg_variant),
            # W-inner-product weight for the single_reduce recurrences
            # (masked cell area — the dot in which this FV Helmholtz is
            # self-adjoint; ignored by the standard body).
            dot_weight=_area_eta * mask,
        )
    # Debug-gated consumer for the solve diagnostic (codex 2026-06-11
    # MAJOR: ``_solve_diag`` was computed and silently discarded on the
    # production path — an under-converged fixed-M solve would advance
    # the model with a bad eta).  Prints ONLY on non-convergence; the
    # production-default path (enable_runtime_checks=False) is
    # unchanged.  ``return_residual=True`` callers keep the loud
    # outside-JIT handling.
    if bool(config.runtime_checks.enable_runtime_checks):
        jax.lax.cond(
            _solve_diag.converged,
            lambda _r: None,
            lambda _r: jax.debug.print(
                "WARNING barotropic_implicit_latlon_cgrid: fixed-M PCG "
                "NOT converged (rel_residual={r:.3e} > tol) — raise "
                "barotropic_implicit_pcg_fixed_iters or check "
                "conditioning.", r=_r,
            ),
            _solve_diag.rel_residual,
        )
    eta_new = eta_new * mask

    # Global mass conservation correction.  The CG solve minimizes the
    # L2 residual but does not guarantee that sum(r * area) = 0 — the
    # area-weighted integral of the residual can have a small nonzero
    # bias that accumulates over 10⁵-10⁶ steps.  Project out the global
    # mean drift so that sum(eta_new * area) = sum(rhs * area) exactly.
    #     composed into every ``A_op`` application and owned-cell-masked
    #     area-weighted dots — the two pieces (halo-in-matvec,
    #     owned-cell reductions) a multi-rank iterated solve needs.
    #
    # MERGE COMPOSITION (PR #394 × MPI-scaling refactor): the single-rank
    # solve routes through :func:`solve_helmholtz_freesurface_mpas` —
    # forward = the stock preconditioned CG VERBATIM (``jax.custom_vjp``
    # inlines the primal, bit-identical), reverse mode = the TRUE
    # area-weighted transpose (stock cg's symmetry-reusing VJP biased the
    # free-surface gradients; commit-1e370679 sibling).  ``A_op`` is
    # still built here for the post-solve residual diagnostic; the
    # custom-VJP solver rebuilds the identical operator internally so its
    # adjoint can take exact parameter cotangents.
    A_op = _make_helmholtz(H_e_old, coeff, mesh, mask, edge_mask)
    # Jacobi diagonal (fed to the custom-VJP solver as an explicit
    # argument — closure-captured tracers fail at scan lowering).
    inv_diag = _helmholtz_inv_diag_mpas(H_e_old, coeff, mesh, mask, edge_mask)

    if _vlayout is not None:
        # ---- Distributed fixed-M PCG (shared solver) ----------------
        # The local TRiSK A_op is correct on OWNED cells provided its
        # input carries fresh ghost values — compose one cell-halo
        # exchange per application (one message round per PCG
        # iteration, static collective schedule).  Dots are owned-
        # masked AND area-weighted: owned-masking removes the halo
        # double-count in the allreduce; the area weight is the inner
        # product in which this FV Helmholtz is self-adjoint (required
        # by the single_reduce recurrences, harmless for standard).
        # AD: this path is differentiated straight THROUGH the unrolled
        # fixed-M loop (sendrecv-VJP + allreduce-SUM dots), so the
        # weighted-transpose concern the single-rank custom-VJP solver
        # addresses does not arise here — the unrolled adjoint is exact
        # by construction.
        from legoesm.parallel.halo_exchange_voronoi import (
            VoronoiHaloExchange,
        )
        from legoesm.ocean.dynamics.barotropic_common import (
            precision_aware_rel_tol,
            solve_helmholtz_implicit,
        )
        _exchanger = VoronoiHaloExchange(_vlayout.partition, backend="mpi")

        def A_op_dist(eta_in: jnp.ndarray) -> jnp.ndarray:
            return A_op(_exchanger.exchange_cell_field(eta_in))

        def _M_inv_dist(r: jnp.ndarray) -> jnp.ndarray:
            return r * inv_diag

        _owned = _vlayout.owned_mask_cells.astype(eta_dtype)
        _w_dots = _owned * mesh.areaCell.astype(eta_dtype) * mask
        eta_new, _solve_diag = solve_helmholtz_implicit(
            A_op_dist, rhs, _M_inv_dist, eta_old,
            distributed=True,
            fixed_iters=int(config.barotropic_implicit_pcg_fixed_iters),
            # f32-safe acceptance tolerance (f64 unchanged); the fixed-iter
            # PCG runs a static count, so this only floors the converged
            # diagnostic.
            residual_tol=precision_aware_rel_tol(
                config.barotropic_implicit_pcg_residual_tol, eta_dtype,
            ),
            stock_cg_tol=config.barotropic_implicit_pcg_tol,
            stock_cg_maxiter=int(config.barotropic_implicit_pcg_maxiter),
            pcg_variant=str(config.barotropic_implicit_pcg_variant),
            dot_weight=_w_dots,
        )
        # Refresh the halo ring of the solution before downstream
        # stencils consume it.
        eta_new = _exchanger.exchange_cell_field(eta_new) * mask
    else:
        # f32: floor the 1e-10 rel-tol to the f32-reachable value so stock CG
        # stops at convergence rather than maxiter (f64 unchanged).
        from legoesm.ocean.dynamics.barotropic_common import (
            precision_aware_rel_tol as _precision_aware_rel_tol,
        )
        pcg_tol = _precision_aware_rel_tol(
            config.barotropic_implicit_pcg_tol, eta_dtype,
        )
        pcg_maxiter = int(config.barotropic_implicit_pcg_maxiter)
        # Forward = stock preconditioned CG, bit-identical; reverse mode
        # uses the TRUE (area-weighted) transpose — stock cg's
        # symmetry-reusing VJP biased free-surface gradients
        # (commit-1e370679 sibling).
        eta_new = solve_helmholtz_freesurface_mpas(
            rhs, eta_old, H_e_old, coeff, mask, edge_mask, inv_diag, mesh,
            tol=pcg_tol, maxiter=pcg_maxiter,
        )
        eta_new = eta_new * mask
    # Mass projection: area-weighted sums.  Single-rank: plain local
    # sums (exact).  Distributed: OWNED-masked partial sums + ONE
    # batched global allreduce (a bare allreduce of unmasked local sums
    # would double-count halo cells).  Accumulate in
    # ``ocean_diagnostics`` precision so the huge-area ``target -
    # actual`` cancellation survives in f32.
    from legoesm.core.precision import cast as _cast
    _M = "ocean_diagnostics"
    _area_cell = mesh.areaCell.astype(eta_dtype)
    _area_acc = _cast(_area_cell, _M, "accumulate")
    _wa = _area_acc * _cast(mask, _M, "accumulate")
    if _vlayout is not None:
        _owned_acc = _cast(_owned, _M, "accumulate")
        _wa = _wa * _owned_acc
        _area_proj = _area_acc * _owned_acc
    else:
        _area_proj = _area_acc
    _ocean_area_l = jnp.sum(_wa)
    _target_mass_l = jnp.sum(_cast(rhs, _M, "accumulate") * _area_proj)
    _actual_mass_l = jnp.sum(_cast(eta_new, _M, "accumulate") * _area_proj)
    if _vlayout is not None:
        from legoesm.parallel.reductions import batch_allreduce_mpi
        _ocean_area, _target_mass, _actual_mass = batch_allreduce_mpi(
            [_ocean_area_l, _target_mass_l, _actual_mass_l],
        )
    else:
        _ocean_area = _ocean_area_l
        _target_mass = _target_mass_l
        _actual_mass = _actual_mass_l
    _correction = (_target_mass - _actual_mass) / jnp.maximum(_ocean_area, 1e-30)
    eta_new = (eta_new + _correction.astype(eta_dtype) * mask) * mask
    # Residual diagnostic (uniform return shape with the lat-lon solver's
    # ``return_residual``), computed AFTER the floor clamp below so it
    # reflects the ACTUAL returned eta.  Single-rank: plain ``jnp.sum``
    # (exact for one rank).  Distributed: owned-masked sums + ONE batched
    # allreduce — never the lat-lon helper's ``_global_dot_batch``, whose
    # bare reduction would double-count Voronoi halo cells.

    # Mass-conserving floor clamp (safety net for extreme transients;
    # in normal operation this is a no-op since the PCG converges to
    # well-resolved η).
    if _vlayout is not None:
        eta_new = _clamp_redistribute(
            eta_new, eta_floor, mask, mesh.areaCell,
            owned_weight=_owned, force_global=True,
        )
    else:
        eta_new = _clamp_redistribute(
            eta_new, eta_floor, mask, mesh.areaCell,
        )

    # Residual diagnostic of the FINAL eta (post projection + clamp).
    # Rank-local (single-rank path); ``stop_gradient`` keeps it out of
    # reverse mode.
    _res_vec = rhs - A_op(eta_new)
    if _vlayout is not None:
        # Owned-masked global residual (eta_new's halo ring was
        # refreshed above, so the local A_op is exact on owned cells;
        # halo rows are excluded from the sums and the two squared
        # norms ride one batched allreduce).
        _rr_l = jnp.sum(_owned * _res_vec**2)
        _bb_l = jnp.sum(_owned * rhs**2)
        from legoesm.parallel.reductions import (
            batch_allreduce_mpi as _bar,
        )
        _rr, _bb = _bar([_rr_l, _bb_l])
    else:
        _rr = jnp.sum(_res_vec**2)
        _bb = jnp.sum(rhs**2)
    _solve_diag_rel = jax.lax.stop_gradient(
        jnp.sqrt(_rr / jnp.maximum(_bb, jnp.asarray(1.0e-30, dtype=eta_dtype)))
    )

    # ----- Step 5: corrector for u_bar using new η gradient delta ------
    eta_filled_new = fill_land_cells_mpas(eta_new, mask, c1, c2)
    grad_eta_new = gradient_edge(eta_filled_new, mesh).astype(eta_dtype)
    delta_grad = grad_eta_new - grad_eta_old

    u_bar_new = (u_pred - theta_pgf * dt_t * g * delta_grad) * edge_mask

    # Barotropic-mode lateral viscosity on u_bar — damps modes that
    # have ∇·(H·u_bar)≈0 (so the Helmholtz solve doesn't see them) and
    # f·v_t cancellations near step edges (so the predictor-corrector
    # doesn't damp them either).  On flat bottom the implicit Helmholtz
    # is sufficient (project_mpas_barotropic_noise.md, 5-yr σ plateau);
    # on partial-cell ETOPO the topographic step edges energize a
    # rotational u_bar null mode that grows e-folding ~5 days
    # (project_mpas_etopo_instability.md).  Mirrors the explicit-substep
    # path (barotropic_mpas.py:272) and the lat-lon Follow-up C
    # recommendation (docs/dev-notes/issues/barotropic_mode_noise.md §"Residual").
    A_baro_visc = jnp.asarray(config.barotropic_u_viscosity, dtype=eta_dtype)
    # Per-edge equatorial-boost factor — SAME Gaussian mechanism as the 3D A_h
    # path (ocean_pe_mpas), guarded identically on the STATIC config float so
    # boost<=0 keeps _lat_factor==1 (no boost) and never evaluates the Gaussian
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:254:    x0: jnp.ndarray,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:291:    - ``x0`` and ``inv_diag`` get exact zero cotangents (IFT; stock-cg
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:309:    def _forward_cg(rhs_in, x0_in, H_e_in, coeff_in,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:318:            A_op, rhs_in, x0=x0_in, tol=tol_in, maxiter=maxiter, M=M_inv,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:323:    def _cg_area_adjoint(rhs_in, x0_in, H_e_in, coeff_in,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:325:        return _forward_cg(rhs_in, x0_in, H_e_in, coeff_in,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:328:    def _cg_fwd(rhs_in, x0_in, H_e_in, coeff_in,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:330:        eta_sol = _forward_cg(rhs_in, x0_in, H_e_in, coeff_in,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:348:            A_op, eta_bar_m / w_rel, x0=jnp.zeros_like(eta_bar_m),
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:363:            jnp.zeros_like(lam),          # x0: exact zero (IFT, stock-cg)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:372:    return _cg_area_adjoint(rhs, x0, H_e, coeff, mask, edge_mask,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:449:    eta_old = state.eta.data
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:452:    eta_dtype = eta_old.dtype
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:462:    eta_old = jnp.maximum(eta_old, eta_floor) * mask
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:465:        F_slow_eta = jnp.zeros_like(eta_old)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:475:        eta_old, H_bathy, z_coord,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:487:        H_total_old = jnp.maximum(eta_old + H_bathy, min_water_col)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:502:    eta_filled_old = fill_land_cells_mpas(eta_old, mask, c1, c2)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:503:    grad_eta_old = gradient_edge(eta_filled_old, mesh).astype(eta_dtype)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:508:        u_bar_old + dt_t * (-g * grad_eta_old + f_e * v_t_old + F_slow_u)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:514:            -g * grad_eta_old
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:538:    grad_eta_for_div = grad_eta_old
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:541:    flux_eta_old = H_e_old * grad_eta_for_div * edge_mask
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:542:    div_grad_eta_old = divergence_cell(flux_eta_old, mesh) * mask
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:543:    div_grad_eta_old = div_grad_eta_old.astype(eta_dtype)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:546:        eta_old
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:548:        - coeff * div_grad_eta_old
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:598:            solve_helmholtz_implicit,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:610:        eta_new, _solve_diag = solve_helmholtz_implicit(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:611:            A_op_dist, rhs, _M_inv_dist, eta_old,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:613:            fixed_iters=int(config.barotropic_implicit_pcg_fixed_iters),
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:643:            rhs, eta_old, H_e_old, coeff, mask, edge_mask, inv_diag, mesh,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:723:    delta_grad = grad_eta_new - grad_eta_old
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_mpas.py:784:    # Use H_e_old consistently so that div(Hu_avg) = (eta_old - eta_new)/dt
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1115:    x0: jnp.ndarray,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1169:    θ ∈ {H_u, H_v, coeff, mask, u_mask, v_mask}.  ``x0`` keeps the exact
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1179:    static).  Every traced array (rhs, x0, H, coeff, masks, inv_diag) AND
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1198:    def _forward_cg(rhs_in, x0_in, H_u_in, H_v_in, coeff_in,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1209:            A_op, rhs_in, x0=x0_in, tol=tol_in, maxiter=maxiter, M=M_inv,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1214:    def _cg_area_adjoint(rhs_in, x0_in, H_u_in, H_v_in, coeff_in,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1216:        return _forward_cg(rhs_in, x0_in, H_u_in, H_v_in, coeff_in,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1220:    def _cg_fwd(rhs_in, x0_in, H_u_in, H_v_in, coeff_in,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1222:        eta_sol = _forward_cg(rhs_in, x0_in, H_u_in, H_v_in, coeff_in,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1242:        # are x0 pass-through with no dependence on rhs/θ.
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1256:            A_op, eta_bar_m / w_rel, x0=jnp.zeros_like(eta_bar_m),
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1275:            jnp.zeros_like(lam),          # x0: exact zero (IFT, stock-cg)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1284:    return _cg_area_adjoint(rhs, x0, H_u, H_v, coeff,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1327:    eta_old = state.eta.data
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1331:    eta_dtype = eta_old.dtype
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1348:        F_slow_eta = jnp.zeros_like(eta_old)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1352:        F_slow_u = jnp.zeros((eta_old.shape[0], eta_old.shape[1] + 1),
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1357:        F_slow_v = jnp.zeros((eta_old.shape[0] + 1, eta_old.shape[1]),
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1363:    eta_old = jnp.maximum(eta_old, eta_floor) * mask
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1367:        eta_old, H_bathy, z_coord,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1374:    # ----- Step 2: face total depth from eta_old -------------------------
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1388:    grad_x_eta_old = gradient_x_cgrid(eta_old, grid).astype(eta_dtype)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1389:    grad_y_eta_old = gradient_y_cgrid(eta_old, grid).astype(eta_dtype)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1400:            U_old + dt_t * (-g * grad_x_eta_old + cor_u.astype(eta_dtype) + F_slow_u)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1406:            V_old + dt_t * (-g * grad_y_eta_old + cor_v.astype(eta_dtype) + F_slow_v)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1424:            U_old + dt_t * (-g * grad_x_eta_old + _cori_fac * f_u * V_at_u + F_slow_u)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1437:            V_old + dt_t * (-g * grad_y_eta_old - _cori_fac * f_v * U_pred_at_v + F_slow_v)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1455:    # ∇·(H · ∇η_old) for the eta_old gradient term in the RHS
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1456:    flux_eta_old_x = H_u_old * grad_x_eta_old * u_mask
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1457:    flux_eta_old_y = H_v_old * grad_y_eta_old * v_mask
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1458:    div_grad_eta_old = divergence_cgrid(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1459:        flux_eta_old_x, flux_eta_old_y, grid, u_mask=u_mask, v_mask=v_mask,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1463:        eta_old
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1465:        - coeff * div_grad_eta_old
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1511:    # (``barotropic_common.solve_helmholtz_implicit``).  Static Python
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1517:    #   ``solve_helmholtz_implicit`` stock branch this routes around);
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1537:        solve_helmholtz_implicit,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1590:            rhs, eta_old, H_u_old, H_v_old, coeff, mask, u_mask, v_mask,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1594:        # Same diagnostic contract as the solve_helmholtz_implicit stock
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1602:        eta_new, _solve_diag = solve_helmholtz_implicit(
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1603:            A_op, rhs, M_inv, eta_old,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1605:            fixed_iters=int(config.barotropic.barotropic_implicit_pcg_fixed_iters),
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1629:                "barotropic_implicit_pcg_fixed_iters or check "
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1715:    delta_grad_x = grad_x_eta_new - grad_x_eta_old
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1716:    delta_grad_y = grad_y_eta_new - grad_y_eta_old
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1731:    # Use H_old consistently so that div(Hu_avg) = (eta_old - eta_new)/dt
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1795:    eta_old: jnp.ndarray,
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1828:    edt = eta_old.dtype
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1839:    rhs = f64((eta_old - dt * div) * mask)
packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1842:        rhs, f64(eta_old), hu_tot, hv_tot, coeff, f64(mask), f64(u_mask), f64(v_mask),

codex
The remaining credible space is narrower than the raw halo percentage suggests: packing, overlap, and PCG variants are already exhausted. The only plausible large wins are changing the communication schedule or partition/topology—not another bandwidth or XLA-flag pass.
codex
The campaign should stop treating sub-30k columns/GPU (or 300–600 cells/CPU rank) as an engineering defect. For one trajectory, that is the measured strong-scaling limit. Below it, only reducing synchronization *epochs* or changing the workload can help.

| Rank | Improvement, mechanism, likely payoff | Cheapest confirm/refute | Applies to | Differentiability / conservation |
|---|---|---|---|---|
| 1 | **Use replica/ensemble parallelism once the spatial floor is reached.** Keep each member at a viable spatial tile; assign extra GPUs/nodes to independent members, parameter perturbations, or adjoint batches. This is the only credible way to regain near-linear *useful throughput* past the tile floor; it does **not** speed one forecast. | Run 1, 4, and 16 independent production-tile members concurrently; report aggregate SYPD, not single-member step time. Require ≥85–90% aggregate efficiency before adopting. | All grids, both lanes; especially spectral, whose honest multi-GPU route is replicas. | Safe: `vmap`/independent JITs preserve gradients and conservation per member. |
| 2 | **MPAS hierarchical low-cut partition + hardware-aware rank mapping.** Install/use METIS, partition first into node-sized groups, then ranks/GPUs within each node; optimize weighted inter-node cut, not merely balance. This directly attacks the MPAS ocean 1.65× rank-count penalty. The absolute ceiling is a 39% time cut; demand a ≥10% full-step win to justify it. | At 32/128 ranks and fixed 5120 cells/rank, compare current/geometric/METIS partitions using: (a) edge cut and distinct remote-peer count, (b) halo-only 1,000-iteration timing, then (c) full in-step-halo-correct ocean step. Also permute rank-to-node assignment without changing the partition—this separates partition quality from placement. | MPAS/Voronoi atmosphere and ocean; CPU MPI and GPU SPMD. | Reordering only; expect roundoff changes in reductions, not conservation loss. Retain serial/owned-cell parity and gradient gates. |
| 3 | **Topology-preserving placement for every finite-volume grid.** Map adjacent tiles to the same 4-GPU NVLink node first; minimize IB-cut edges, then respect CPU NUMA/HCA affinity. With 17.8 µs NVLink versus 26.3 µs IB, cube’s 44% halo phase has a realistic ~5–10% total-step opportunity; ~14% is a generous upper bound. | Change only `CUDA_VISIBLE_DEVICES`/Slurm rank ordering and the global device ordering. Hold HLO, tile shape, and partition fixed; time the halo-only path and full step. Reject if the best mapping is <3%. | Cube, lat-lon, MPAS; GPU first, CPU MPI as a separate NUMA/HCA placement sweep. | No numerical risk. |
| 4 | **Communication-avoiding temporal blocking for explicit local operators.** Exchange a \(B\)-deep halo, advance \(B=2\) initially inside a `lax.scan`, retain only the valid core, then exchange again. This is not rejected “latency hiding”: it removes dependent halo epochs. On cube, \(B=2\) has a 22% whole-step theoretical ceiling from the measured 6.65 ms halo phase; expect materially less after extra work/ghosts. | First measure halo time and memory for depths 1, 2, 4. Proceed only if depth-2 is far cheaper than two depth-1 exchanges and ghost overhead is <20%. Then prototype one conservative tracer/stencil block with forward, VJP, and two-block serial equivalence. | Explicit cube, lat-lon atmosphere, MPAS atmosphere; possibly explicit ocean baroclinic pieces. Not implicit-CN PCG. Lat-lon barotropic wide-halo is already done; do not redo it. Spectral: N/A. | High implementation risk. Exactness requires a correct causal-width proof; preserve canonical interface flux ownership and test block-boundary conservation plus `jax.grad`. |
| 5 | **Remove halo epochs by liveness/dataflow, not packing.** Build an RK-stage dependency graph, reuse already-fresh primitive halos across consumers, and reconstruct derived boundary quantities locally rather than exchanging them. This is distinct from the refuted packing/CP-combining work: the target is fewer `collective-permute`/sendrecv rounds. A 10% reduction in cube halo epochs is roughly a 4% whole-step win. | Produce an HLO/JAXPR “halo epoch” ledger by field and RK stage. Select one provably redundant refresh; implement only that removal and require both a lower collective count and ≥2% full-step improvement. | Cube first; then lat-lon. MPAS only where the stage-halo correctness proof says a refresh is not required. CPU MPI and GPU SPMD. | Medium/high risk: stale halos silently corrupt boundary cells. Require multi-step serial parity, conservation, and VJP tests. |
| 6 | **Replace hardcoded fusion barriers with an offline shape/dtype autotuner.** Generate a few semantically identical cut points around RK axpy/tendency outputs, benchmark after compilation, and cache the winner by `(grid, dtype, mesh, local shapes, XLA build)`. This generalizes the known 11–17% MPAS recoveries without guessing signatures. | For each production shape, compile baseline plus 2–3 barrier placements; use 20 steady steps and retain only winners above noise. HLO screening should flag giant multi-consumer fusions before timing. | GPU SPMD for cube, lat-lon, MPAS, and possibly spectral transforms; CPU only if profiling shows a similar pathology. | Safe in principle: `optimization_barrier` is semantically identity. Still gate forward parity, `jax.grad`, and conservation. |
| 7 | **CPU rank coarsening with deliberate thread/NUMA ownership.** Use fewer MPI ranks per node, larger tiles per rank, and give each rank a pinned core/NUMA allocation for XLA CPU work. It reduces endpoint count and moves ranks above the 300–600-cell floor. Plausible gain: 10–30% in the bad strong-scaling regime, not a universal win. | Matrix sweep: 2/4/8/16 ranks per node × pinned cores per rank, with fixed global problem and no oversubscription. Record rank-max step time, halo/reduction time, and DRAM bandwidth. | CPU MPI finite-volume grids; spectral CPU local compute may benefit, but has no distributed spectral solve yet. | Safe; no algorithm change. |
| 8 | **NCCL latency tuning beyond the already-closed `NCCL_PROTO` arm.** Separately test all-reduce algorithm and P2P settings: `NCCL_ALGO=Tree,Ring`, P2P low-latency threshold/buffer options, and a conservative CTA/channel sweep—only for options supported by Levante’s installed NCCL. This is a low-cost chance of a few percent, not a primary strategy. | Benchmark exact production-size scalar/batched all-reduces and halo payloads separately, then validate the winning setting in the full ocean and cube steps. Stop if <3%. | GPU finite-volume lanes; strongest relevance to PCG reductions and small halo messages. | No numerical risk. NCCL exposes algorithm, P2P, and CTA controls separately from protocol selection in its [environment-variable guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html). |
| 9 | **Compile the real operator-split segment as `lax.scan`; keep gathers only at true output/checkpoint boundaries.** This removes per-step host dispatch and prevents periodic full-state materialization from contaminating end-to-end scaling. It will not change the device halo wall; expect 0–10% only if a trace shows host gaps. | Trace a production segment and measure host-idle gaps. A/B the current Python step loop against a scan with identical forcing/carry/output cadence. | Both lanes and all grids with the operator-split driver. | `lax.scan` is differentiable. Preserve forcing semantics and do not defer conservation correction unless science explicitly accepts it. |
| 10 | **Spectral: run a Levante feasibility gate, not a distributed implementation campaign.** The only plausible single-trajectory design is a pencil/transpose spectral transform: latitude-sharded grid-space physics/FFT, distributed Legendre GEMM and SI solve, with all-to-all transposes. It may pay only at very high truncation; level sharding remains a dead end. | On A100/IB, microbenchmark realistic T170/T341/T682 tensor transposes plus local f64 GEMMs. Build only if the model predicts ≥1.3× at the intended device count. Otherwise use rank-1 replicas from item 1. | Spectral GPU SPMD, eventually CPU MPI; no benefit for existing level sharding. | All-JAX transpose/GEMM is differentiable; transform reordering needs round-trip and adjoint tolerances. JAX `ppermute` itself requires unique source/destination pairs, so edge-colored neighbor rounds cannot be “compressed for free.” [JAX docs](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ppermute.html) |

The key negative conclusion for implicit ocean is important: there is no remaining credible PCG micro-optimization. Do not revisit bandwidth reduction, pipelining/overlap, wet compaction/balancing, wide-halo lat-lon barotropic work, Chebyshev, or distributed V-cycle multigrid without new solver evidence. The measured wall is serialized synchronization; the already-shipped single-reduce and split-explicit options are the available solver choices.

If I had one immediate Levante allocation, I would run: (1) MPAS METIS + rank-placement separation, (2) topology-only placement sweep for cube/lat-lon/MPAS, and (3) depth-1/2/4 halo microbenchmarks to decide whether temporal blocking is worth building.
tokens used
125,152
The campaign should stop treating sub-30k columns/GPU (or 300–600 cells/CPU rank) as an engineering defect. For one trajectory, that is the measured strong-scaling limit. Below it, only reducing synchronization *epochs* or changing the workload can help.

| Rank | Improvement, mechanism, likely payoff | Cheapest confirm/refute | Applies to | Differentiability / conservation |
|---|---|---|---|---|
| 1 | **Use replica/ensemble parallelism once the spatial floor is reached.** Keep each member at a viable spatial tile; assign extra GPUs/nodes to independent members, parameter perturbations, or adjoint batches. This is the only credible way to regain near-linear *useful throughput* past the tile floor; it does **not** speed one forecast. | Run 1, 4, and 16 independent production-tile members concurrently; report aggregate SYPD, not single-member step time. Require ≥85–90% aggregate efficiency before adopting. | All grids, both lanes; especially spectral, whose honest multi-GPU route is replicas. | Safe: `vmap`/independent JITs preserve gradients and conservation per member. |
| 2 | **MPAS hierarchical low-cut partition + hardware-aware rank mapping.** Install/use METIS, partition first into node-sized groups, then ranks/GPUs within each node; optimize weighted inter-node cut, not merely balance. This directly attacks the MPAS ocean 1.65× rank-count penalty. The absolute ceiling is a 39% time cut; demand a ≥10% full-step win to justify it. | At 32/128 ranks and fixed 5120 cells/rank, compare current/geometric/METIS partitions using: (a) edge cut and distinct remote-peer count, (b) halo-only 1,000-iteration timing, then (c) full in-step-halo-correct ocean step. Also permute rank-to-node assignment without changing the partition—this separates partition quality from placement. | MPAS/Voronoi atmosphere and ocean; CPU MPI and GPU SPMD. | Reordering only; expect roundoff changes in reductions, not conservation loss. Retain serial/owned-cell parity and gradient gates. |
| 3 | **Topology-preserving placement for every finite-volume grid.** Map adjacent tiles to the same 4-GPU NVLink node first; minimize IB-cut edges, then respect CPU NUMA/HCA affinity. With 17.8 µs NVLink versus 26.3 µs IB, cube’s 44% halo phase has a realistic ~5–10% total-step opportunity; ~14% is a generous upper bound. | Change only `CUDA_VISIBLE_DEVICES`/Slurm rank ordering and the global device ordering. Hold HLO, tile shape, and partition fixed; time the halo-only path and full step. Reject if the best mapping is <3%. | Cube, lat-lon, MPAS; GPU first, CPU MPI as a separate NUMA/HCA placement sweep. | No numerical risk. |
| 4 | **Communication-avoiding temporal blocking for explicit local operators.** Exchange a \(B\)-deep halo, advance \(B=2\) initially inside a `lax.scan`, retain only the valid core, then exchange again. This is not rejected “latency hiding”: it removes dependent halo epochs. On cube, \(B=2\) has a 22% whole-step theoretical ceiling from the measured 6.65 ms halo phase; expect materially less after extra work/ghosts. | First measure halo time and memory for depths 1, 2, 4. Proceed only if depth-2 is far cheaper than two depth-1 exchanges and ghost overhead is <20%. Then prototype one conservative tracer/stencil block with forward, VJP, and two-block serial equivalence. | Explicit cube, lat-lon atmosphere, MPAS atmosphere; possibly explicit ocean baroclinic pieces. Not implicit-CN PCG. Lat-lon barotropic wide-halo is already done; do not redo it. Spectral: N/A. | High implementation risk. Exactness requires a correct causal-width proof; preserve canonical interface flux ownership and test block-boundary conservation plus `jax.grad`. |
| 5 | **Remove halo epochs by liveness/dataflow, not packing.** Build an RK-stage dependency graph, reuse already-fresh primitive halos across consumers, and reconstruct derived boundary quantities locally rather than exchanging them. This is distinct from the refuted packing/CP-combining work: the target is fewer `collective-permute`/sendrecv rounds. A 10% reduction in cube halo epochs is roughly a 4% whole-step win. | Produce an HLO/JAXPR “halo epoch” ledger by field and RK stage. Select one provably redundant refresh; implement only that removal and require both a lower collective count and ≥2% full-step improvement. | Cube first; then lat-lon. MPAS only where the stage-halo correctness proof says a refresh is not required. CPU MPI and GPU SPMD. | Medium/high risk: stale halos silently corrupt boundary cells. Require multi-step serial parity, conservation, and VJP tests. |
| 6 | **Replace hardcoded fusion barriers with an offline shape/dtype autotuner.** Generate a few semantically identical cut points around RK axpy/tendency outputs, benchmark after compilation, and cache the winner by `(grid, dtype, mesh, local shapes, XLA build)`. This generalizes the known 11–17% MPAS recoveries without guessing signatures. | For each production shape, compile baseline plus 2–3 barrier placements; use 20 steady steps and retain only winners above noise. HLO screening should flag giant multi-consumer fusions before timing. | GPU SPMD for cube, lat-lon, MPAS, and possibly spectral transforms; CPU only if profiling shows a similar pathology. | Safe in principle: `optimization_barrier` is semantically identity. Still gate forward parity, `jax.grad`, and conservation. |
| 7 | **CPU rank coarsening with deliberate thread/NUMA ownership.** Use fewer MPI ranks per node, larger tiles per rank, and give each rank a pinned core/NUMA allocation for XLA CPU work. It reduces endpoint count and moves ranks above the 300–600-cell floor. Plausible gain: 10–30% in the bad strong-scaling regime, not a universal win. | Matrix sweep: 2/4/8/16 ranks per node × pinned cores per rank, with fixed global problem and no oversubscription. Record rank-max step time, halo/reduction time, and DRAM bandwidth. | CPU MPI finite-volume grids; spectral CPU local compute may benefit, but has no distributed spectral solve yet. | Safe; no algorithm change. |
| 8 | **NCCL latency tuning beyond the already-closed `NCCL_PROTO` arm.** Separately test all-reduce algorithm and P2P settings: `NCCL_ALGO=Tree,Ring`, P2P low-latency threshold/buffer options, and a conservative CTA/channel sweep—only for options supported by Levante’s installed NCCL. This is a low-cost chance of a few percent, not a primary strategy. | Benchmark exact production-size scalar/batched all-reduces and halo payloads separately, then validate the winning setting in the full ocean and cube steps. Stop if <3%. | GPU finite-volume lanes; strongest relevance to PCG reductions and small halo messages. | No numerical risk. NCCL exposes algorithm, P2P, and CTA controls separately from protocol selection in its [environment-variable guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html). |
| 9 | **Compile the real operator-split segment as `lax.scan`; keep gathers only at true output/checkpoint boundaries.** This removes per-step host dispatch and prevents periodic full-state materialization from contaminating end-to-end scaling. It will not change the device halo wall; expect 0–10% only if a trace shows host gaps. | Trace a production segment and measure host-idle gaps. A/B the current Python step loop against a scan with identical forcing/carry/output cadence. | Both lanes and all grids with the operator-split driver. | `lax.scan` is differentiable. Preserve forcing semantics and do not defer conservation correction unless science explicitly accepts it. |
| 10 | **Spectral: run a Levante feasibility gate, not a distributed implementation campaign.** The only plausible single-trajectory design is a pencil/transpose spectral transform: latitude-sharded grid-space physics/FFT, distributed Legendre GEMM and SI solve, with all-to-all transposes. It may pay only at very high truncation; level sharding remains a dead end. | On A100/IB, microbenchmark realistic T170/T341/T682 tensor transposes plus local f64 GEMMs. Build only if the model predicts ≥1.3× at the intended device count. Otherwise use rank-1 replicas from item 1. | Spectral GPU SPMD, eventually CPU MPI; no benefit for existing level sharding. | All-JAX transpose/GEMM is differentiable; transform reordering needs round-trip and adjoint tolerances. JAX `ppermute` itself requires unique source/destination pairs, so edge-colored neighbor rounds cannot be “compressed for free.” [JAX docs](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ppermute.html) |

The key negative conclusion for implicit ocean is important: there is no remaining credible PCG micro-optimization. Do not revisit bandwidth reduction, pipelining/overlap, wet compaction/balancing, wide-halo lat-lon barotropic work, Chebyshev, or distributed V-cycle multigrid without new solver evidence. The measured wall is serialized synchronization; the already-shipped single-reduce and split-explicit options are the available solver choices.

If I had one immediate Levante allocation, I would run: (1) MPAS METIS + rank-placement separation, (2) topology-only placement sweep for cube/lat-lon/MPAS, and (3) depth-1/2/4 halo microbenchmarks to decide whether temporal blocking is worth building.
