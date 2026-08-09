Reading additional input from stdin...
OpenAI Codex v0.146.1
--------
workdir: /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
model: gpt-5.6-terra
provider: openai
approval: never
sandbox: read-only
reasoning effort: xhigh
reasoning summaries: none
session id: 019fdc62-876d-7f30-9f5f-94d954763fac
--------
user
ADVERSARIAL REVIEW of a DIAGNOSTIC that will decide a 7-14 day engineering call. Refute, do not summarize. End SHIP or NOT READY.

NEW FILES (untracked): scripts/bench/analyze_nsys_rank_skew.py, tests/bench/test_analyze_nsys_rank_skew.py. Also uncommitted: scripts/cluster/scaling_levante/{_nsys_rank_wrapper.sh,mpas_nsys_skew_capture.sbatch}.

THE DECISION IT FEEDS. MPAS GPU spends 5.94 of 9.72 ms/step in ncclDevKernel_SendRecv, median 216 us vs ~30 us wire. If that time is PAYLOAD, cutting round count saves only ~30 us x rounds removed and a round-depth partitioner is near its floor (not worth 7-14 days). If it is WAIT (blocked on a slower peer), the lever is imbalance instead. A prior single-rank analysis could not decide this and I RETRACTED a claim that it could.

THE METHOD. An NCCL SendRecv kernel starts when its rank arrives and ends when the transfer completes. So the same collective seen from both partners separates the cases: equal durations => transfer; one long one short => the long one idled, and |difference| is the wait.

MATCHING — attack this hardest. I REJECTED interval-overlap matching (GLM-5.2 proposed it) on the grounds that all 64 ranks issue their round-r collective at about the same time, so rank 0 overlaps a NON-partner as readily as its partner; overlap is necessary, never sufficient. Instead I match via the SCHEDULE: spmd_schedule_cost(..., round_profile_for_device=d) gives the rounds d participates in IN ISSUE ORDER, so if A meets B at global round r, fill k pairs A[k*len_A + idx_A] with B[k*len_B + idx_B]. Checks: symmetry (B must name A at the same round), equal fill counts across ranks, whole-fill divisibility.
 (a) Is that pairing actually correct? Specifically: does the RUNTIME issue a device's collectives in the same order as round_profile lists them, and does it really skip rounds where the device has no partner? Read _ppermute_halo_fill and the lowering. If XLA emits a collective for EVERY round on every device, len(profile) is wrong and every pairing shears.
 (b) Was rejecting overlap-matching correct, or did I discard a usable method?
 (c) Is 'kernel start == rank arrival' true for NCCL SendRecv, or does the kernel launch earlier/later than arrival in a way that breaks the inference? This is the load-bearing physical claim.
 (d) Could BOTH ranks be waiting on a third party, making |difference| ~0 and mimicking payload-bound? If so the method has a false-PAYLOAD mode and I need to say so.

ALSO: cross-node pairs are refused via the capture's _rank_nodes.tsv (CLOCK_MONOTONIC is per-machine). Is same-node comparability actually sound for kernel timestamps from two independent nsys processes? Any remaining vacuity in the 12 tests? Anything in the sbatch/wrapper that could produce a capture that looks fine and is not?

Cite file:line.
codex
I’ll inspect the diagnostic and the relevant lowering/runtime code, then try to falsify its assumptions against the tests and capture plumbing.
exec
/usr/bin/bash -lc 'pwd && git status --short && rg -n "_ppermute_halo_fill|spmd_schedule_cost|round_profile_for_device|SendRecv|ppermute" -S .' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 5648ms:
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 M scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh
?? .physics-validator/flag_reachability_audit/
?? .physics-validator/homogeneous_ice_nucleation/
?? .physics-validator/scaling_campaign/codex_review_rank_skew_2026-08-07.md
?? config/aimip/clubb_bechtold/
?? config/aimip/scale/suite_sfno_full_longtrain.yaml
?? config/aimip/scale/suite_sfno_full_maxdata.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed0.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed1.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed2.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed3.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed4.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed5.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed6.yaml
?? config/aimip/scale/suite_sfno_full_mx4_seed7.yaml
?? config/aimip/scale/suite_sfno_full_ood_finetune.yaml
?? config/aimip/scale/suite_sfno_full_res_t106.yaml
?? config/aimip/scale/suite_sfno_full_res_t21.yaml
?? config/aimip/scale/suite_sfno_full_res_t42.yaml
?? config/aimip/scale/suite_sfno_full_res_t63.yaml
?? config/aimip/scale/suite_sfno_full_t63_ace2.yaml
?? config/aimip/scale/suite_sfno_full_t63_allyears_ace2.yaml
?? config/aimip/scale/suite_sfno_full_t63_allyears_gc.yaml
?? config/aimip/scale/suite_sfno_full_ucast_dropout.yaml
?? config/aimip/scale/suite_sfno_full_ucast_mae.yaml
?? config/aimip/scale/suite_sfno_full_ucast_muon.yaml
?? config/aimip/scale/suite_sfno_full_ucast_muondrop.yaml
?? config/aimip/scale/suite_sfno_full_ucast_muonlr.yaml
?? config/wb/loss_presets/ucast.yaml
?? docs/dev-notes/ace2_learning_strategy_gap.md
?? docs/dev-notes/ucast_learning_strategy_gap.md
?? drift.txt
?? nat.txt
?? scripts/bench/analyze_nsys_rank_skew.py
?? scripts/cluster/unified_training/bench_wb2_levante.sbatch
?? scripts/cluster/unified_training/score_ucast_arms.sbatch
?? scripts/cluster/unified_training/score_ucast_probabilistic.sbatch
?? tests/bench/test_analyze_nsys_rank_skew.py
?? tests/distributed/test_sfno_full_dp_mpi.py
?? tests/unit/test_ace2_sfno_suites.py
?? tests/unit/test_aimip_eval_train_gate.py
?? tests/unit/test_aimip_generic_scheme_params.py
?? tests/unit/test_apply_params_to_pipeline.py
?? tests/unit/test_ensemble_crps.py
?? tests/unit/test_loss_level_weighting.py
?? tests/unit/test_multiseed_ensemble_cli.py
?? tests/unit/test_muon_partitioned_groups.py
?? tests/unit/test_persistence_normalized_loss.py
?? tests/unit/test_sfno_full_data_parallel.py
?? tests/unit/test_sfno_full_pmap_devices.py
?? tests/unit/test_sfno_pe_rollout_stability.py
?? tests/unit/test_ucast_learning_strategy.py
?? tests/unit/test_wb_climatology_reference.py
?? tests/unit/test_wb_scorecard_sota_units.py
./tests/bench/test_scaling_metadata.py:398:    fn with none -> 0; an unlowerable fn -> None (never raises). Real ppermute
./scripts/bench/bench_cube_tiled_step_scaling.py:222:    # (the tiled stage manages its own ppermutes through the mesh).
./scripts/bench/bench_cube_tiled_step_scaling.py:250:    n_ppermute = _count_collective_permutes(hlo)
./scripts/bench/bench_cube_tiled_step_scaling.py:256:    if n_ppermute == 0:
./scripts/bench/bench_cube_tiled_step_scaling.py:371:        hlo_collective_permutes=n_ppermute,
./scripts/bench/bench_cube_tiled_step_scaling.py:409:            "hlo_collective_permutes": n_ppermute,
./scripts/bench/bench_cube_tiled_step_scaling.py:431:              f"ppermutes={n_ppermute}")
./scripts/bench/analyze_nsys_halo_rounds.py:4:The MPAS GPU lane spends most of its step inside ``ncclDevKernel_SendRecv``
./scripts/bench/analyze_nsys_halo_rounds.py:24:   production schedule says that round moves (``spmd_schedule_cost(...,
./scripts/bench/analyze_nsys_halo_rounds.py:25:   round_profile_for_device=)``), as a Pearson r.
./scripts/bench/analyze_nsys_halo_rounds.py:53:touch it.  At s9/np64 rank 0 shows 8 ``SendRecv`` per halo fill while the
./scripts/bench/analyze_nsys_halo_rounds.py:55:CONTAINING THAT DEVICE.  ``round_profile_for_device`` returns exactly that
./scripts/bench/analyze_nsys_halo_rounds.py:86:_SENDRECV_LIKE = "ncclDevKernel_SendRecv%"
./scripts/bench/analyze_nsys_halo_rounds.py:183:    Read off the production exchange (``_ppermute_halo_fill``): it sends
./scripts/bench/analyze_nsys_halo_rounds.py:187:    the round maximum and ppermute moves that whole buffer, so a pair's own
./scripts/bench/analyze_nsys_halo_rounds.py:263:    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
./scripts/bench/analyze_nsys_halo_rounds.py:269:    cost = spmd_schedule_cost(mesh, args.n_devices,
./scripts/bench/analyze_nsys_halo_rounds.py:272:                              round_profile_for_device=args.device)
./scripts/bench/bench_mpas_spmd_scaling.py:3:ppermute halo).
./scripts/bench/bench_mpas_spmd_scaling.py:10:only the partition-boundary halo per stage via ``jax.lax.ppermute``.
./scripts/bench/bench_mpas_spmd_scaling.py:17:nlev caveat (#1113): the multi-node ceiling at THIN nlev is the ppermute ROUND
./scripts/bench/bench_mpas_spmd_scaling.py:34:``jax.devices()``, and the existing ``make_voronoi_sharded_step`` ppermute
./scripts/bench/bench_mpas_spmd_scaling.py:80:# of the sharded step (ppermute halo + mass-fix psum reduction-order change),
./scripts/bench/bench_mpas_spmd_scaling.py:206:                   choices=["auto", "ppermute", "allgather"],
./scripts/bench/bench_mpas_spmd_scaling.py:210:                        "cell threshold — force 'ppermute' to exercise "
./scripts/bench/bench_mpas_spmd_scaling.py:322:        # connectivity the ppermute schedule + TRiSK stencils read; a
./scripts/bench/bench_mpas_spmd_scaling.py:419:    # the sharded step — the ppermute ROUND count that decomposes multi-node
./scripts/bench/bench_mpas_spmd_scaling.py:426:    # no ppermute halo -> 0.
./scripts/bench/bench_mpas_spmd_scaling.py:505:        # saying "auto" would not reveal whether ppermute or allgather
./scripts/bench/bench_mpas_spmd_scaling.py:519:        # ppermute round count/step (static compile property; #1113) — the
./scripts/bench/run_levante_gpu_scaling.py:1686:            # Tripwire (codex ppermute-multiface review): the row below is
./scripts/bench/metadata.py:128:    The ppermute halo kernel's signature and a STATIC compile property (the
./tests/bench/test_ppermute_microbench.py:1:"""Contract tests for the ppermute latency/bandwidth microbenchmark.
./tests/bench/test_ppermute_microbench.py:11:from bench_ppermute_microbench import fit_latency_bandwidth  # noqa: E402
./tests/bench/test_ppermute_microbench.py:39:              / "scripts" / "bench" / "bench_ppermute_microbench.py")
./scripts/bench/bench_atm_latlon_spmd_scaling.py:43:``make_sharded_atm_latlon_step`` + band-ppermute halo runs unchanged — the
./scripts/bench/bench_atm_latlon_spmd_scaling.py:44:ppermute/psum collectives cross processes via the distributed runtime (NCCL on
./scripts/bench/bench_cube_shardmap_halo.py:7:(``jax.lax.ppermute`` inside ``shard_map``) scales, unlike the ``mpi4jax``
./scripts/bench/bench_cube_shardmap_halo.py:20:* **AD** — a nonuniform-cotangent VJP through the SPMD ``ppermute`` halo matches
./scripts/bench/bench_cube_shardmap_halo.py:22:  contain a ``collective_permute`` — the ppermute kernel's signature (the
./scripts/bench/bench_cube_shardmap_halo.py:23:  all_gather diagnostic kernel must NOT satisfy this) — proving the ppermute path
./scripts/bench/bench_cube_shardmap_halo.py:90:    ppermute/all_gather kernel-selection flag through the public API (importing
./scripts/bench/bench_cube_shardmap_halo.py:91:    the private ``_use_ppermute`` across modules is disallowed by repo policy), so
./scripts/bench/bench_cube_shardmap_halo.py:92:    reactivation resets it to the default ppermute selection and warns.
./scripts/bench/bench_cube_shardmap_halo.py:103:            "restoring a prior SPMD halo backend: the private ppermute/all_gather "
./scripts/bench/bench_cube_shardmap_halo.py:104:            "selection is reset to the default ppermute (not snapshottable via the "
./scripts/bench/bench_cube_shardmap_halo.py:133:def _hlo_has_ppermute(fn, *args) -> bool | None:
./scripts/bench/bench_cube_shardmap_halo.py:135:    ``collective_permute`` — the signature of the *ppermute* cube-halo kernel?
./scripts/bench/bench_cube_shardmap_halo.py:139:    bandwidth-optimal ppermute path; the all_gather diagnostic kernel replicates
./scripts/bench/bench_cube_shardmap_halo.py:154:                       "NOT the ppermute collective this experiment measures.")
./scripts/bench/bench_cube_shardmap_halo.py:240:    SPMD (``ppermute``/``shard_map``) backend over a face-sharded mesh. For SPMD
./scripts/bench/bench_cube_shardmap_halo.py:303:        hlo_collective = _hlo_has_ppermute(lambda st: model.step(st, dt), s0)
./scripts/bench/bench_cube_shardmap_halo.py:346:# AD gate: VJP through the SPMD ppermute halo vs single-device reference
./scripts/bench/bench_cube_shardmap_halo.py:355:    (the ppermute kernel's signature), proving the ppermute path ran rather than a
./scripts/bench/bench_cube_shardmap_halo.py:383:        # --- SPMD VJP (ppermute) ---
./scripts/bench/bench_cube_shardmap_halo.py:391:            has_collective = _hlo_has_ppermute(_f, x_sh)
./scripts/bench/bench_cube_shardmap_halo.py:474:            # Require the ppermute collective to be PROVEN present (True). None
./scripts/bench/bench_cube_shardmap_halo.py:476:            # have exercised the cross-device ppermute VJP, not a local fallback.
./scripts/bench/bench_cube_shardmap_halo.py:492:    # --- Timed-path ppermute gate -----------------------------------------
./scripts/bench/bench_cube_shardmap_halo.py:493:    # The AD gate proves ``explicit_pad_halo`` lowered ppermute, but the
./scripts/bench/bench_cube_shardmap_halo.py:495:    # at every >1-device count actually used the ppermute collective (else we may
./scripts/bench/bench_cube_shardmap_halo.py:498:    timed_ppermute_pass = (
./scripts/bench/bench_cube_shardmap_halo.py:513:             "efficiency": efficiency_pass, "timed_ppermute": timed_ppermute_pass}
./scripts/bench/bench_cube_shardmap_halo.py:518:        enforced += ["efficiency", "timed_ppermute"]
./scripts/bench/bench_cube_shardmap_halo.py:632:    logger.info("POINT n_devices=%d (procs=%d) ms/step=%.3f hlo_ppermute=%s",
./scripts/bench/aggregate_cube_shardmap_scaling.py:13:* **ppermute proof** — every ``n_devices > 1`` point lowered a
./scripts/bench/aggregate_cube_shardmap_scaling.py:15:  executable is the bandwidth-optimal ppermute path, not a replicated all_gather
./scripts/bench/aggregate_cube_shardmap_scaling.py:157:    # --- ppermute proof: every >1-device point lowered collective_permute ---
./scripts/bench/aggregate_cube_shardmap_scaling.py:159:    ppermute_pass = all(r["spmd_hlo_collective"] is True for r in multi) if multi else True
./scripts/bench/aggregate_cube_shardmap_scaling.py:197:             "ppermute": ppermute_pass, "efficiency": efficiency_pass}
./scripts/bench/aggregate_cube_shardmap_scaling.py:202:        enforced.append("ppermute")
./scripts/bench/aggregate_cube_shardmap_scaling.py:248:             "| n_devices | ms/step | speedup | efficiency | ppermute | procs |",
./tests/bench/test_aggregate_cube_shardmap_scaling.py:2:no JAX). Covers every gate branch: baseline present/missing, ppermute
./tests/bench/test_aggregate_cube_shardmap_scaling.py:45:    assert r["gates"]["ppermute"] is True
./tests/bench/test_aggregate_cube_shardmap_scaling.py:61:# --- ppermute proof gate ----------------------------------------------------
./tests/bench/test_aggregate_cube_shardmap_scaling.py:62:def test_ppermute_false_fails(tmp_path):
./tests/bench/test_aggregate_cube_shardmap_scaling.py:65:    _pt(tmp_path, 3, 5.0, False)   # a >1 point that did NOT lower ppermute
./tests/bench/test_aggregate_cube_shardmap_scaling.py:67:    assert r["gates"]["ppermute"] is False
./tests/bench/test_aggregate_cube_shardmap_scaling.py:71:def test_ppermute_none_fails(tmp_path):
./tests/bench/test_aggregate_cube_shardmap_scaling.py:75:    assert r["gates"]["ppermute"] is False
./tests/bench/test_mpas_schedule_cost_scan_sbatch.py:317:            f"(spmd_schedule_cost docstring): {sorted(parsed.items())}")
./scripts/bench/bench_voronoi_partition_methods.py:12:   the expensive layer): ``n_rounds`` — the number of SEQUENTIAL ppermute
./scripts/bench/bench_voronoi_partition_methods.py:19:   ``spmd_schedule_cost`` (which calls the production builders), never a
./scripts/bench/bench_voronoi_partition_methods.py:25:   graph (one colour = one ppermute round; ``_build_ppermute_schedule``
./scripts/bench/bench_voronoi_partition_methods.py:42:   UNDIRECTED edge colouring of THIS graph.  ``_build_ppermute_schedule``
./scripts/bench/bench_voronoi_partition_methods.py:44:   halo dependency and then emits BOTH ppermute directions, even where one
./scripts/bench/bench_voronoi_partition_methods.py:57:   cells/device threshold) there is no ppermute schedule in production and
./scripts/bench/bench_voronoi_partition_methods.py:263:    :func:`legoesm.parallel.sharded_dynamics.spmd_schedule_cost` — it reorders
./scripts/bench/bench_voronoi_partition_methods.py:295:    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
./scripts/bench/bench_voronoi_partition_methods.py:298:    cost = spmd_schedule_cost(mesh, n_ranks, method=method)
./scripts/bench/bench_voronoi_partition_methods.py:321:        # "allgather" => production runs no ppermute schedule here, so the
./scripts/bench/bench_voronoi_partition_methods.py:339:                   help="Also score the SPMD ppermute halo-schedule depth "
./scripts/bench/bench_voronoi_partition_methods.py:447:                        "allgather here, no ppermute schedule]"
./tests/bench/test_bench_cube_shardmap_halo.py:6:subprocess run that exercises the SPMD ppermute halo path + correctness + a
./tests/bench/test_bench_cube_shardmap_halo.py:108:    way to exercise the ppermute halo path.
./tests/bench/test_bench_cube_shardmap_halo.py:143:        "SPMD AD path did not lower a collective — ppermute VJP not exercised")
./tests/bench/test_bench_cube_shardmap_halo.py:176:    # The timed model.step lowered the ppermute collective.
./tests/bench/test_bench_cube_shardmap_halo.py:198:    # curve/ppermute/baseline, not the efficiency threshold.
./tests/bench/test_bench_cube_shardmap_halo.py:204:    assert verdict["gates"]["ppermute"] is True
./scripts/bench/analyze_nsys_rank_skew.py:5:``ncclDevKernel_SendRecv`` (s9/np64), median 216 us against a ~30 us wire
./scripts/bench/analyze_nsys_rank_skew.py:15:THE DISCRIMINATOR.  An NCCL SendRecv kernel starts when ITS rank arrives at
./scripts/bench/analyze_nsys_rank_skew.py:30:exact.  ``spmd_schedule_cost(..., round_profile_for_device=d)`` gives the
./scripts/bench/analyze_nsys_rank_skew.py:75:_SENDRECV_LIKE = "ncclDevKernel_SendRecv%"
./scripts/bench/analyze_nsys_rank_skew.py:231:    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
./scripts/bench/analyze_nsys_rank_skew.py:238:        cost = spmd_schedule_cost(mesh, args.n_devices,
./scripts/bench/analyze_nsys_rank_skew.py:241:                                  round_profile_for_device=rank)
./scripts/bench/run_cpu_mpi_scaling.py:575:    ``make_sharded_step`` + multiface-ppermute halo runs unchanged, and
./scripts/bench/run_cpu_mpi_scaling.py:658:        # exchange (which rides the SAME multiface-ppermute halo as T under the
./scripts/bench/run_cpu_mpi_scaling.py:1536:             "ppermute; the A1 path).  Replaces the replicated-dynamics "
./tests/bench/test_bench_voronoi_partition_methods.py:203:        "cells_per_device": 99_999, "production_strategy": "ppermute",
./tests/bench/test_bench_voronoi_partition_methods.py:207:    monkeypatch.setattr(sd, "spmd_schedule_cost", lambda *a, **k: stub)
./tests/bench/test_bench_voronoi_partition_methods.py:246:    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
./tests/bench/test_bench_voronoi_partition_methods.py:249:    ref = spmd_schedule_cost(mesh, 2, method="sfc")
./scripts/bench/bench_gather_vs_slice_stencil.py:65:    # same trick as bench_ppermute_microbench.
./scripts/bench/bench_ppermute_microbench.py:1:"""Measure this machine's ppermute latency + bandwidth for roofline lines.
./scripts/bench/bench_ppermute_microbench.py:10:collective the sharded steps use (``jax.lax.ppermute`` on a ring, inside a
./scripts/bench/bench_ppermute_microbench.py:27:    python scripts/bench/bench_ppermute_microbench.py --n-devices 4
./scripts/bench/bench_ppermute_microbench.py:31:        python scripts/bench/bench_ppermute_microbench.py \
./scripts/bench/bench_ppermute_microbench.py:56:    """jit'd program doing n_reps back-to-back ring ppermutes on device."""
./scripts/bench/bench_ppermute_microbench.py:63:                return jax.lax.ppermute(v, axis_name=AXIS, perm=perm)
./scripts/bench/bench_ppermute_microbench.py:94:    """Per-ppermute time with HOST DISPATCH SUBTRACTED.
./scripts/bench/bench_ppermute_microbench.py:138:                   help="Back-to-back ppermutes inside ONE jit call; the "
./scripts/bench/bench_ppermute_microbench.py:153:            f"ppermute needs >=2 devices; got {n_dev}. A latency/bandwidth "
./scripts/bench/bench_ppermute_microbench.py:189:        "collective": "ppermute_ring",
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:25:and the existing ``make_sharded_ocean_step`` band-ppermute halo + psum
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:82:# of the sharded split-explicit barotropic (ppermute/psum reduction-order
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:129:    # barotropic substeps instead of ~4 ppermute pads per substep.  The wide
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:310:                        "(LEGOESM_LATLON_SPMD_FUSED_HALO=1): one ppermute "
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:321:                        "ppermutes. This is the A/B CONTROL arm: the "
./scripts/bench/bench_ocean_latlon_spmd_scaling.py:327:                        "instead of ~4 ppermute pads per substep (implies "
./tests/bench/test_analyze_nsys_rank_skew.py:147:        ("ncclDevKernel_SendRecv(x)", [(100, 300), (500, 900)]),
./scripts/bench/bench_ocean_latlon_spmd_pcg.py:10:``lax.ppermute`` + pole fold; CG dots via ``jax.lax.psum``).  Pure jax, NO
./tests/bench/test_bench_mpas_spmd_gates.py:155:    # #1113 ask 2: the ppermute round count is now recorded per row.
./scripts/bench/roofline_probe.py:9:(``legoesm.parallel`` ppermute / sendrecv / batch_allreduce) and times
./scripts/bench/roofline_probe.py:22:   (``measure_ppermute_collective`` + ``measure_sendrecv_collective``):
./scripts/bench/roofline_probe.py:24:     (a) GPU/SPMD ``jax.lax.ppermute`` PRIMITIVE inside a ``shard_map``
./scripts/bench/roofline_probe.py:26:         cubed-sphere halo (``cubesphere_exchange._make_exchange_ppermute``
./scripts/bench/roofline_probe.py:109:GPU (1 or 2 devices; ppermute needs >=2, gracefully skipped on 1)::
./scripts/bench/roofline_probe.py:260:    have data-dependent outputs (triad, ppermute, sendrecv), so XLA
./scripts/bench/roofline_probe.py:455:    transport: str           # "ppermute_spmd" | "mpi4jax_sendrecv".
./scripts/bench/roofline_probe.py:467:def measure_ppermute_collective(
./scripts/bench/roofline_probe.py:475:    """Time the ``jax.lax.ppermute`` PRIMITIVE inside a ``shard_map`` over
./scripts/bench/roofline_probe.py:477:    devices are visible (single-GPU / single CPU process) — ppermute
./scripts/bench/roofline_probe.py:480:    Scope (codex MAJOR — do NOT over-claim): this measures the *ppermute
./scripts/bench/roofline_probe.py:483:    (``cubesphere_exchange._make_exchange_ppermute`` uses ``lax.ppermute``
./scripts/bench/roofline_probe.py:497:    DEPENDS on the ppermute result, so XLA cannot DCE the collective.
./scripts/bench/roofline_probe.py:510:        transport="ppermute_spmd",
./scripts/bench/roofline_probe.py:529:                got = jax.lax.ppermute(local, "face", perm)
./scripts/bench/roofline_probe.py:537:        # whose value DEPENDS on the ppermute, so the collective cannot be
./scripts/bench/roofline_probe.py:560:    """Tiny adapter so ``measure_ppermute_collective`` reads cleanly:
./scripts/bench/roofline_probe.py:562:    ``check_vma=False`` (matching the model's ppermute kernel, which also
./scripts/bench/roofline_probe.py:1285:        ppm = measure_ppermute_collective(
./scripts/bench/roofline_probe.py:1291:                print(f"[2a] ppermute SPMD: latency floor "
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:32:  ``ppermute``-ing band ``r+1``'s first ``v_lower`` row (= global ``v[e]``) up as
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:562:    whose meshes name that axis differently would psum/ppermute over
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:777:    # Static perms for the v north-boundary-row ppermute (band r receives band
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:799:        # band (no r+1) receives the pole-wall 0 via the ppermute non-target.
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:804:            boundary = jax.lax.ppermute(
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:818:        # ppermutes of ALL staggered carriers (v + v_mask) into one
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:825:        # per-field ppermutes (byte-identical, just more collectives).
./packages/ocean/legoesm/ocean/dynamics/sharded_ocean_step.py:976:        # (axis_index / ppermute / psum in pad_with_pole_bc_lat, conservation,
./packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:708:    ppermute / local pole-fold) instead of being zeroed, so the meridional
./packages/ocean/legoesm/ocean/dynamics/barotropic_implicit_latlon_cgrid.py:1554:    # A_op halo ppermute + CG reductions inside a DATA-DEPENDENT while_loop,
./scripts/cluster/scaling_levante/README.md:95:ppermute halos), 6 tasks = 2 nodes x 3 GPUs (`nCells = 10*4^L + 2` splits
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:21:# (s8@16) to 4.47x (s9@64), and one halo fill costs 12-14 SEQUENTIAL ppermute
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:29:# (one colour = one ppermute round) and max_degree is that graph's maximum
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:44:# (b) CONFIRMS a cheap fix: gap >= 2 at any production (ppermute) row
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:51:#     ppermute directions per pair even when one send map is empty, so a
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:59:#     (cells/device below the threshold), so production runs no ppermute
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:64:#     ppermute (L6 = 40,962 cells; padded, np8 gives 40,968/8 = 5,121 and
./scripts/cluster/scaling_levante/mpas_schedule_cost_scan.sbatch:70:# ARM 1 IS AN INSTRUMENT CHECK, NOT A RESULT.  spmd_schedule_cost's
./scripts/cluster/scaling_levante/_env.sh:102:# NCCL (shard_map/ppermute collectives under jax.distributed) uses its own
./scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh:36:    #            ncclDevKernel_SendRecv by name).  Asking for nccl here would
./scripts/cluster/scaling_levante/mpas_nsys_skew_capture.sbatch:17:# ncclDevKernel_SendRecv, median 216 us against a ~30 us wire latency.  The
./scripts/cluster/scaling_levante/mpas_nsys_skew_capture.sbatch:23:# THE DISCRIMINATOR.  An NCCL SendRecv kernel starts when ITS rank arrives and
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:20:#                         gap-#0 lane (native ppermute halos replace the
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:27:#                         (cell-partition reorder + ppermute halo)
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:159:# Gap-#0 lane: native ppermute band halos over NCCL replace the route-A
./scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:194:# cell-partition reorder + ppermute halo (make_voronoi_sharded_step) over
./scripts/cluster/scaling_levante/cube_tiled_step.sbatch:75:# ("face","tile_i","tile_j")) on top of the halo ppermute cliques. Under XLA
./scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch:26:#   * SendRecv is 5.94 of 9.72 ms = 61 % of the step, and rank 0 is not even
./scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch:81:from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
./scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch:87:    "WHERE s.value LIKE 'ncclDevKernel_SendRecv%'").fetchone()[0]
./scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch:89:print(f"[attr] trace holds {n_calls} SendRecv calls", flush=True)
./scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch:94:    cost = spmd_schedule_cost(mesh, 64, method="sfc", reorder_target=target,
./scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch:95:                              round_profile_for_device=0)
./packages/core/legoesm/parallel/distributed.py:63:    reductions run through pure-JAX ``ppermute``/``psum`` inside ``shard_map``,
./packages/core/legoesm/parallel/distributed.py:97:      pure-JAX ppermute/psum); a genuinely single-process run (no MPI/PMI/SLURM
./packages/core/legoesm/parallel/cubesphere_exchange.py:7:* **ppermute multiface** (DEFAULT for every face-sharded device count
./packages/core/legoesm/parallel/cubesphere_exchange.py:12:  device-pair schedule of ``jax.lax.ppermute`` rounds, one
./packages/core/legoesm/parallel/cubesphere_exchange.py:16:* **ppermute one-face** (halo=1, exactly 6 devices): the original
./packages/core/legoesm/parallel/cubesphere_exchange.py:98:# ppermute round tables
./packages/core/legoesm/parallel/cubesphere_exchange.py:101:# into 4 perfect matchings so that each round is one ppermute call
./packages/core/legoesm/parallel/cubesphere_exchange.py:104:def _build_ppermute_tables():
./packages/core/legoesm/parallel/cubesphere_exchange.py:105:    """Compute the static ppermute schedule from CONNECTIVITY.
./packages/core/legoesm/parallel/cubesphere_exchange.py:152:    # call ``_PPERMUTE_* = _build_ppermute_tables()`` does NOT
./packages/core/legoesm/parallel/cubesphere_exchange.py:157:    # the ``_make_exchange_ppermute._exchange`` closure.
./packages/core/legoesm/parallel/cubesphere_exchange.py:167:    _build_ppermute_tables()
./packages/core/legoesm/parallel/cubesphere_exchange.py:261:# Reachable only with the module ppermute flag off (force_allgather /
./packages/core/legoesm/parallel/cubesphere_exchange.py:265:# 1.00 at 2 devices) — production routing uses the ppermute kernels.
./packages/core/legoesm/parallel/cubesphere_exchange.py:291:    # detailed rationale in `_make_exchange_ppermute` below.
./packages/core/legoesm/parallel/cubesphere_exchange.py:318:        # any divisor of 6.  The ppermute multi-face refit has since
./packages/core/legoesm/parallel/cubesphere_exchange.py:385:# ppermute kernel)
./packages/core/legoesm/parallel/cubesphere_exchange.py:414:    # shard_map body. See _make_exchange_ppermute rationale.
./packages/core/legoesm/parallel/cubesphere_exchange.py:510:# Backend B: ppermute  (bandwidth-optimal for high resolution)
./packages/core/legoesm/parallel/cubesphere_exchange.py:513:def _make_exchange_ppermute(mesh, ndim, with_offsets=False):
./packages/core/legoesm/parallel/cubesphere_exchange.py:514:    """Build a shard_map exchange using 4 rounds of ppermute.
./packages/core/legoesm/parallel/cubesphere_exchange.py:541:    # `_make_exchange_ppermute` is called from
./packages/core/legoesm/parallel/cubesphere_exchange.py:544:    ppermute_send_j = jnp.asarray(_PPERMUTE_SEND)
./packages/core/legoesm/parallel/cubesphere_exchange.py:545:    ppermute_recv_j = jnp.asarray(_PPERMUTE_RECV)
./packages/core/legoesm/parallel/cubesphere_exchange.py:546:    ppermute_rev_j = jnp.asarray(_PPERMUTE_REV)
./packages/core/legoesm/parallel/cubesphere_exchange.py:579:            send_edge = ppermute_send_j[r, my_idx]   # traced int
./packages/core/legoesm/parallel/cubesphere_exchange.py:581:            received = jax.lax.ppermute(
./packages/core/legoesm/parallel/cubesphere_exchange.py:584:            recv_edge = ppermute_recv_j[r, my_idx]
./packages/core/legoesm/parallel/cubesphere_exchange.py:585:            rev = ppermute_rev_j[r, my_idx]
./packages/core/legoesm/parallel/cubesphere_exchange.py:616:# Backend C: multi-face ppermute (k = 6 / n_devices faces per shard)
./packages/core/legoesm/parallel/cubesphere_exchange.py:619:# Generalizes the one-face ppermute kernel to n_devices ∈ {1, 2, 3, 6}
./packages/core/legoesm/parallel/cubesphere_exchange.py:625:#   * cross-shard face edges → static schedule of ppermute rounds.
./packages/core/legoesm/parallel/cubesphere_exchange.py:646:    """Static per-layout tables for the multi-face ppermute exchange.
./packages/core/legoesm/parallel/cubesphere_exchange.py:654:        ``perms[r]`` is the (src_dev, dst_dev) pair list for ppermute
./packages/core/legoesm/parallel/cubesphere_exchange.py:663:        as a placeholder; they are provably overwritten by a ppermute
./packages/core/legoesm/parallel/cubesphere_exchange.py:667:        — applied receiver-side to BOTH local and ppermute strips
./packages/core/legoesm/parallel/cubesphere_exchange.py:696:    round repeats a src or a dst (each round is a valid ppermute
./packages/core/legoesm/parallel/cubesphere_exchange.py:787:                f"multiface ppermute schedule coloring failed for "
./packages/core/legoesm/parallel/cubesphere_exchange.py:864:# the ppermute kernel consumes it.  Edge conventions are EXACTLY the
./packages/core/legoesm/parallel/cubesphere_exchange.py:941:    """Static schedule for the tiled ppermute exchange (1 tile/device).
./packages/core/legoesm/parallel/cubesphere_exchange.py:1142:                     diagonal interior cell -> diagonal-tile ppermute
./packages/core/legoesm/parallel/cubesphere_exchange.py:1267:    ``lax.ppermute`` over (face, tile_i, tile_j) — valid inside ANY
./packages/core/legoesm/parallel/cubesphere_exchange.py:1275:    Schedule: 4 strip ppermute rounds (every tile edge is remote) +
./packages/core/legoesm/parallel/cubesphere_exchange.py:1291:            f"tiled ppermute exchange supports halo 1/2, got {halo}.")
./packages/core/legoesm/parallel/cubesphere_exchange.py:1347:            received = jax.lax.ppermute(send_buf, AXES, tables.perms[r])
./packages/core/legoesm/parallel/cubesphere_exchange.py:1373:            lo_from_jbwd = jax.lax.ppermute(
./packages/core/legoesm/parallel/cubesphere_exchange.py:1375:            hi_from_jfwd = jax.lax.ppermute(
./packages/core/legoesm/parallel/cubesphere_exchange.py:1377:            lo_from_ibwd = jax.lax.ppermute(
./packages/core/legoesm/parallel/cubesphere_exchange.py:1379:            hi_from_ifwd = jax.lax.ppermute(
./packages/core/legoesm/parallel/cubesphere_exchange.py:1431:        # ppermute non-targets receive zeros, masked by the mode.
./packages/core/legoesm/parallel/cubesphere_exchange.py:1450:            jax.lax.ppermute(diag_send[c], AXES, diag_perms[c])
./packages/core/legoesm/parallel/cubesphere_exchange.py:1453:        sl_jf = jax.lax.ppermute(
./packages/core/legoesm/parallel/cubesphere_exchange.py:1455:        sl_jb = jax.lax.ppermute(
./packages/core/legoesm/parallel/cubesphere_exchange.py:1457:        sl_if = jax.lax.ppermute(
./packages/core/legoesm/parallel/cubesphere_exchange.py:1459:        sl_ib = jax.lax.ppermute(
./packages/core/legoesm/parallel/cubesphere_exchange.py:1520:def _make_exchange_ppermute_tiled(mesh, ndim, halo=1, with_offsets=False):
./packages/core/legoesm/parallel/cubesphere_exchange.py:1521:    """Tiled (6*kt^2-device) ppermute shard_map exchange — halo 1/2.
./packages/core/legoesm/parallel/cubesphere_exchange.py:1551:    block via ``lax.ppermute`` directly.  A nested ``shard_map`` (what
./packages/core/legoesm/parallel/cubesphere_exchange.py:1555:    :func:`_make_exchange_ppermute_tiled` body (same code).
./packages/core/legoesm/parallel/cubesphere_exchange.py:1613:def _make_exchange_ppermute_multiface(mesh, ndim, halo=1, with_offsets=False):
./packages/core/legoesm/parallel/cubesphere_exchange.py:1614:    """Build the multi-face ppermute shard_map exchange.
./packages/core/legoesm/parallel/cubesphere_exchange.py:1626:        raise ValueError(f"multiface ppermute supports halo 1/2, got {halo}")
./packages/core/legoesm/parallel/cubesphere_exchange.py:1631:            f"multiface ppermute exchange requires a face-axis mesh whose "
./packages/core/legoesm/parallel/cubesphere_exchange.py:1699:        # ppermute rounds provably overwrite — coverage asserted at
./packages/core/legoesm/parallel/cubesphere_exchange.py:1705:        # ---- cross-shard ppermute rounds ----
./packages/core/legoesm/parallel/cubesphere_exchange.py:1709:            received = jax.lax.ppermute(send_buf, "face", perms[r])
./packages/core/legoesm/parallel/cubesphere_exchange.py:1807:    extent exactly ``n_faces`` — none exist on the ppermute hot path.
./packages/core/legoesm/parallel/cubesphere_exchange.py:1848:            f"ppermute multiface exchange (collective-permute only).  "
./packages/core/legoesm/parallel/cubesphere_exchange.py:1861:def _select_variant(use_ppermute, halo, n_devices):
./packages/core/legoesm/parallel/cubesphere_exchange.py:1864:    * ``use_ppermute=False`` → the all_gather kernels (explicit
./packages/core/legoesm/parallel/cubesphere_exchange.py:1866:    * ``use_ppermute=True, halo=1, n_devices=6`` → the original
./packages/core/legoesm/parallel/cubesphere_exchange.py:1867:      validated one-face ppermute kernel (lowest-risk path for the
./packages/core/legoesm/parallel/cubesphere_exchange.py:1870:    * every other ppermute combination (halo=2 at any count; halo=1 at
./packages/core/legoesm/parallel/cubesphere_exchange.py:1871:      1/2/3 devices) → the multiface ppermute kernel.
./packages/core/legoesm/parallel/cubesphere_exchange.py:1877:        # the use_ppermute flag (the activation pre-warm loop also
./packages/core/legoesm/parallel/cubesphere_exchange.py:1879:        return "ppermute_tiled"
./packages/core/legoesm/parallel/cubesphere_exchange.py:1880:    if not use_ppermute:
./packages/core/legoesm/parallel/cubesphere_exchange.py:1883:        return "ppermute_oneface"
./packages/core/legoesm/parallel/cubesphere_exchange.py:1884:    return "ppermute_multiface"
./packages/core/legoesm/parallel/cubesphere_exchange.py:1887:def _get_exchange(mesh, ndim, use_ppermute, halo=1, with_offsets=False):
./packages/core/legoesm/parallel/cubesphere_exchange.py:1896:    use_ppermute : bool
./packages/core/legoesm/parallel/cubesphere_exchange.py:1898:        :func:`activate_spmd_halo_backend`), route to a ppermute kernel
./packages/core/legoesm/parallel/cubesphere_exchange.py:1915:    ``use_ppermute``-boolean key could not distinguish the one-face
./packages/core/legoesm/parallel/cubesphere_exchange.py:1931:    variant = _select_variant(use_ppermute, halo, n_devices)
./packages/core/legoesm/parallel/cubesphere_exchange.py:1935:        if variant == "ppermute_tiled":
./packages/core/legoesm/parallel/cubesphere_exchange.py:1936:            _cache[key] = _make_exchange_ppermute_tiled(
./packages/core/legoesm/parallel/cubesphere_exchange.py:1947:        elif variant == "ppermute_oneface":
./packages/core/legoesm/parallel/cubesphere_exchange.py:1948:            _cache[key] = _make_exchange_ppermute(
./packages/core/legoesm/parallel/cubesphere_exchange.py:1951:        else:  # ppermute_multiface
./packages/core/legoesm/parallel/cubesphere_exchange.py:1952:            _cache[key] = _make_exchange_ppermute_multiface(
./packages/core/legoesm/parallel/cubesphere_exchange.py:1958:# Module-level flag: use ppermute by default?
./packages/core/legoesm/parallel/cubesphere_exchange.py:1959:_use_ppermute: bool = False
./packages/core/legoesm/parallel/cubesphere_exchange.py:1975:    """Decide whether to use ppermute (True) or all_gather (False).
./packages/core/legoesm/parallel/cubesphere_exchange.py:1977:    Always returns True (ppermute) unless the explicit all_gather
./packages/core/legoesm/parallel/cubesphere_exchange.py:1986:    data-volume model can see.  ppermute is therefore the only
./packages/core/legoesm/parallel/cubesphere_exchange.py:1998:        True if ppermute is selected, False only under the explicit
./packages/core/legoesm/parallel/cubesphere_exchange.py:2005:def set_ppermute_default(enabled: bool) -> None:
./packages/core/legoesm/parallel/cubesphere_exchange.py:2011:        ``True`` → ppermute kernels (the production default set by
./packages/core/legoesm/parallel/cubesphere_exchange.py:2021:    global _use_ppermute
./packages/core/legoesm/parallel/cubesphere_exchange.py:2022:    _use_ppermute = enabled
./packages/core/legoesm/parallel/cubesphere_exchange.py:2028:    Halo=1 and halo=2 both honour the module ppermute flag (the
./packages/core/legoesm/parallel/cubesphere_exchange.py:2030:    ppermute routes to the multiface kernel (one-face kernel at halo=1
./packages/core/legoesm/parallel/cubesphere_exchange.py:2041:            return _get_exchange(mesh, 3, _use_ppermute, halo=2)(data)
./packages/core/legoesm/parallel/cubesphere_exchange.py:2043:            mesh, 3, _use_ppermute, halo=2, with_offsets=True,
./packages/core/legoesm/parallel/cubesphere_exchange.py:2049:        return _get_exchange(mesh, 3, _use_ppermute, halo=1)(data)
./packages/core/legoesm/parallel/cubesphere_exchange.py:2051:        mesh, 3, _use_ppermute, halo=1, with_offsets=True,
./packages/core/legoesm/parallel/cubesphere_exchange.py:2074:            return _get_exchange(mesh, 4, _use_ppermute, halo=2)(data)
./packages/core/legoesm/parallel/cubesphere_exchange.py:2076:            mesh, 4, _use_ppermute, halo=2, with_offsets=True,
./packages/core/legoesm/parallel/cubesphere_exchange.py:2082:        return _get_exchange(mesh, 4, _use_ppermute, halo=1)(data)
./packages/core/legoesm/parallel/cubesphere_exchange.py:2084:        mesh, 4, _use_ppermute, halo=1, with_offsets=True,
./packages/core/legoesm/parallel/cubesphere_exchange.py:2183:    :func:`explicit_pad_halo_4d`: the ppermute kernels under the module
./packages/core/legoesm/parallel/cubesphere_exchange.py:2248:    The face-axis ppermute/all_gather kernels block-partition the
./packages/core/legoesm/parallel/cubesphere_exchange.py:2271:    The exchange kernel is **ppermute** for every face-sharded device
./packages/core/legoesm/parallel/cubesphere_exchange.py:2293:        environment override, otherwise ppermute.
./packages/core/legoesm/parallel/cubesphere_exchange.py:2307:    global _spmd_mesh, _use_ppermute
./packages/core/legoesm/parallel/cubesphere_exchange.py:2372:    _use_ppermute = use_pp
./packages/core/legoesm/parallel/cubesphere_exchange.py:2373:    backend_name = "ppermute" if use_pp else "all_gather(DIAGNOSTIC)"
./packages/core/legoesm/parallel/cubesphere_exchange.py:2390:    # relevant (ndim, halo, with_offsets, use_ppermute) combinations
./packages/core/legoesm/parallel/cubesphere_exchange.py:2394:    # module flag at halo=2 since the multiface ppermute kernel landed,
./packages/core/legoesm/parallel/cubesphere_exchange.py:2395:    # so a post-activation ``set_ppermute_default`` flip must still be
./packages/core/legoesm/parallel/voronoi_partition.py:1014:    """Concrete ownership for the SPMD/ppermute path (``auto`` -> ``sfc``).
./packages/core/legoesm/parallel/voronoi_partition.py:1058:    # policy.  This is the SPMD/ppermute path, where the cost that binds at
./packages/core/legoesm/parallel/voronoi_partition.py:1082:    # exchange is not this ppermute schedule and no census was run for them.
./packages/core/legoesm/parallel/async_halo.py:90:       (ppermute multiface exchange; all_gather only as the explicit
./packages/core/legoesm/parallel/async_halo.py:95:    ``jax.lax.ppermute`` for device-to-device communication instead of MPI.
./packages/core/legoesm/parallel/async_halo.py:106:        the experimental ``ppermute`` path for device-to-device halos;
./packages/core/legoesm/parallel/async_halo.py:116:    The ppermute implementation is experimental and:
./packages/core/legoesm/parallel/async_halo.py:119:    2. Applies ``jax.lax.ppermute`` calls for each edge/halo strip.
./packages/core/legoesm/parallel/async_halo.py:128:        # ppermute path only works for face-only sharding (6 faces, no tiles).
./packages/core/legoesm/parallel/async_halo.py:133:                "ppermute halo exchange does not support sub-face tiling "
./packages/core/legoesm/parallel/async_halo.py:141:            "jax_native_halo_exchange: the ppermute-based code path is "
./packages/core/legoesm/parallel/async_halo.py:147:        return _ppermute_halo_exchange(data, grid, mesh)
./packages/core/legoesm/parallel/async_halo.py:153:def _ppermute_halo_exchange(data, grid, mesh):
./packages/core/legoesm/parallel/async_halo.py:154:    """Implement halo exchange via jax.lax.ppermute (experimental).
./packages/core/legoesm/parallel/async_halo.py:214:        strips_permuted = jax.lax.ppermute(
./docs/architecture/DISTRIBUTED_ARCHITECTURE.md:17:| `spmd` | Single-node multi-GPU | `shard_map` + `ppermute` / `psum` | Yes (via `shard_map` VJP) |
./packages/core/legoesm/parallel/sharded_dynamics.py:14:   ``jax.lax.ppermute``-like collective inside shard_map.
./packages/core/legoesm/parallel/sharded_dynamics.py:701:        ``activate_spmd_halo_backend``.  The old ppermute-vs-all_gather
./packages/core/legoesm/parallel/sharded_dynamics.py:702:        volume auto-selection is RETIRED: ppermute is always selected;
./packages/core/legoesm/parallel/sharded_dynamics.py:717:    then routes through shard_map ppermute kernels (multiface; one-face
./packages/core/legoesm/parallel/sharded_dynamics.py:740:    # ppermute-multiface refit then made ppermute the DEFAULT exchange
./packages/core/legoesm/parallel/sharded_dynamics.py:748:    # 6*kt^2 sub-face tiling: the tiled ppermute EXCHANGE is serial-
./packages/core/legoesm/parallel/sharded_dynamics.py:1244:#: consumed by both the production step factory and ``spmd_schedule_cost``:
./packages/core/legoesm/parallel/sharded_dynamics.py:1250:def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
./packages/core/legoesm/parallel/sharded_dynamics.py:1252:                       ppermute_cells_per_device_threshold=2_000,
./packages/core/legoesm/parallel/sharded_dynamics.py:1253:                       round_profile_for_device=None):
./packages/core/legoesm/parallel/sharded_dynamics.py:1256:    Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
./packages/core/legoesm/parallel/sharded_dynamics.py:1264:    :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
./packages/core/legoesm/parallel/sharded_dynamics.py:1275:      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
./packages/core/legoesm/parallel/sharded_dynamics.py:1279:    * It scores the ppermute strategy.  Production auto-selects ALLGATHER when
./packages/core/legoesm/parallel/sharded_dynamics.py:1280:      cells/device is below ``ppermute_cells_per_device_threshold``, in which
./packages/core/legoesm/parallel/sharded_dynamics.py:1281:      case there is no ppermute schedule and this number is counterfactual --
./packages/core/legoesm/parallel/sharded_dynamics.py:1311:    ppermute_cells_per_device_threshold : int
./packages/core/legoesm/parallel/sharded_dynamics.py:1314:    round_profile_for_device : int | None
./packages/core/legoesm/parallel/sharded_dynamics.py:1321:        that touch it -- at s9/np64 rank 0 shows 8 ``SendRecv`` per halo fill
./packages/core/legoesm/parallel/sharded_dynamics.py:1330:        ``(n_dev, max_c)`` and ``ppermute`` moves the whole padded buffer, so
./packages/core/legoesm/parallel/sharded_dynamics.py:1342:        ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
./packages/core/legoesm/parallel/sharded_dynamics.py:1356:            f"spmd_schedule_cost: n_dev must be an integer >= 1, got {n_dev!r}")
./packages/core/legoesm/parallel/sharded_dynamics.py:1362:                "spmd_schedule_cost: reorder_target is meaningless with "
./packages/core/legoesm/parallel/sharded_dynamics.py:1382:            f"spmd_schedule_cost: prepared mesh has nCells={n_cells}, "
./packages/core/legoesm/parallel/sharded_dynamics.py:1394:    sched = _build_ppermute_schedule(
./packages/core/legoesm/parallel/sharded_dynamics.py:1417:            ("allgather" if cells_per < ppermute_cells_per_device_threshold
./packages/core/legoesm/parallel/sharded_dynamics.py:1418:             else "ppermute")),
./packages/core/legoesm/parallel/sharded_dynamics.py:1423:            {} if round_profile_for_device is None else
./packages/core/legoesm/parallel/sharded_dynamics.py:1424:            {"round_profile": _round_profile(sched, round_profile_for_device,
./packages/core/legoesm/parallel/sharded_dynamics.py:1433:    See ``spmd_schedule_cost``'s ``round_profile_for_device``.  Only rounds
./packages/core/legoesm/parallel/sharded_dynamics.py:1441:            f"round_profile_for_device must be an integer in [0, {n_dev}), "
./packages/core/legoesm/parallel/sharded_dynamics.py:1445:    for r, perm in enumerate(sched["ppermute_perms"]):
./packages/core/legoesm/parallel/sharded_dynamics.py:1451:        # arrays are (n_dev, max_c) and ppermute moves the padded buffer, so
./packages/core/legoesm/parallel/sharded_dynamics.py:1490:        Per-device partition descriptors (for ppermute schedule building).
./packages/core/legoesm/parallel/sharded_dynamics.py:1744:    ``2*max_degree - 1`` colors — each color is one ppermute ROUND, and the
./packages/core/legoesm/parallel/sharded_dynamics.py:1769:# schedule (the coloring must agree across ranks or the ppermute pattern
./packages/core/legoesm/parallel/sharded_dynamics.py:1777:    using the FEWEST colors (= ppermute rounds) across several deterministic
./packages/core/legoesm/parallel/sharded_dynamics.py:1823:def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
./packages/core/legoesm/parallel/sharded_dynamics.py:1825:    """Build a ppermute-based halo exchange schedule.
./packages/core/legoesm/parallel/sharded_dynamics.py:1828:    this schedule uses ``jax.lax.ppermute`` to exchange only halo data
./packages/core/legoesm/parallel/sharded_dynamics.py:1830:    so that each round of ppermute moves data between non-conflicting
./packages/core/legoesm/parallel/sharded_dynamics.py:1845:        ppermute_perms, send_cell_idx, recv_cell_pos,
./packages/core/legoesm/parallel/sharded_dynamics.py:1891:            'ppermute_perms': [],
./packages/core/legoesm/parallel/sharded_dynamics.py:1901:    # 3. Edge-color the graph: each color = one bidirectional ppermute
./packages/core/legoesm/parallel/sharded_dynamics.py:1926:        "improper ppermute edge coloring — two same-round exchanges "
./packages/core/legoesm/parallel/sharded_dynamics.py:1961:    # 5. Assemble per-round ppermute patterns and index arrays
./packages/core/legoesm/parallel/sharded_dynamics.py:1963:    ppermute_perms_out: list[list[tuple[int, int]]] = []
./packages/core/legoesm/parallel/sharded_dynamics.py:1984:        # Bidirectional ppermute pattern
./packages/core/legoesm/parallel/sharded_dynamics.py:1992:        ppermute_perms_out.append(perm)
./packages/core/legoesm/parallel/sharded_dynamics.py:2032:        'ppermute_perms': ppermute_perms_out,
./packages/core/legoesm/parallel/sharded_dynamics.py:2127:def _ppermute_halo_fill(cell_pack, u_shard, halo_sl, ppermute_perms,
./packages/core/legoesm/parallel/sharded_dynamics.py:2129:    """Fill (owned + halo) local buffers from owned shards via ppermute.
./packages/core/legoesm/parallel/sharded_dynamics.py:2135:    posts ONE flat ppermute carrying BOTH entity classes for ALL packed
./packages/core/legoesm/parallel/sharded_dynamics.py:2144:    schedule.  ``ppermute_perms`` is the static per-round permutation.
./packages/core/legoesm/parallel/sharded_dynamics.py:2162:        recv_packed = jax.lax.ppermute(
./packages/core/legoesm/parallel/sharded_dynamics.py:2163:            send_packed, "device", perm=ppermute_perms[r])
./packages/core/legoesm/parallel/sharded_dynamics.py:2178:    ppermute_cells_per_device_threshold: int = 2_000,
./packages/core/legoesm/parallel/sharded_dynamics.py:2194:       flat ppermute payload per neighbor round.
./packages/core/legoesm/parallel/sharded_dynamics.py:2202:    leading device axis), the ppermute schedule index arrays, and the
./packages/core/legoesm/parallel/sharded_dynamics.py:2223:        ``"auto"`` (default) selects ``"ppermute"`` for large grids and
./packages/core/legoesm/parallel/sharded_dynamics.py:2225:        *ppermute_cells_per_device_threshold*.
./packages/core/legoesm/parallel/sharded_dynamics.py:2226:        ``"ppermute"`` forces neighbor-only exchange via
./packages/core/legoesm/parallel/sharded_dynamics.py:2227:        ``jax.lax.ppermute`` — O(halo) communication.
./packages/core/legoesm/parallel/sharded_dynamics.py:2230:    ppermute_cells_per_device_threshold : int
./packages/core/legoesm/parallel/sharded_dynamics.py:2231:        When ``halo_strategy="auto"``, use ppermute only if each device
./packages/core/legoesm/parallel/sharded_dynamics.py:2233:        per-round packing/scatter overhead of ppermute exceeds the
./packages/core/legoesm/parallel/sharded_dynamics.py:2324:        if cells_per < ppermute_cells_per_device_threshold:
./packages/core/legoesm/parallel/sharded_dynamics.py:2328:                "threshold=%d — ppermute packing overhead would dominate.",
./packages/core/legoesm/parallel/sharded_dynamics.py:2329:                cells_per, ppermute_cells_per_device_threshold,
./packages/core/legoesm/parallel/sharded_dynamics.py:2332:            halo_strategy = "ppermute"
./packages/core/legoesm/parallel/sharded_dynamics.py:2334:                "Auto-selected ppermute strategy: cells_per_device=%d >= "
./packages/core/legoesm/parallel/sharded_dynamics.py:2336:                cells_per, ppermute_cells_per_device_threshold,
./packages/core/legoesm/parallel/sharded_dynamics.py:2356:        partitions_out,   # list[VoronoiPartition] (for ppermute schedule)
./packages/core/legoesm/parallel/sharded_dynamics.py:2384:    # Strategy dispatch: ppermute (O(halo)) vs allgather (O(N))
./packages/core/legoesm/parallel/sharded_dynamics.py:2387:    use_ppermute = halo_strategy == "ppermute"
./packages/core/legoesm/parallel/sharded_dynamics.py:2389:    if use_ppermute:
./packages/core/legoesm/parallel/sharded_dynamics.py:2390:        # Build ppermute schedule: neighbor-only halo exchange
./packages/core/legoesm/parallel/sharded_dynamics.py:2392:        pp_sched = _build_ppermute_schedule(
./packages/core/legoesm/parallel/sharded_dynamics.py:2397:        ppermute_perms = pp_sched['ppermute_perms']
./packages/core/legoesm/parallel/sharded_dynamics.py:2426:            "  ppermute schedule: %d rounds, max halo cells/edges per round: %s / %s",
./packages/core/legoesm/parallel/sharded_dynamics.py:2432:            "  comm volume per stage: ppermute ~%.1f KB vs allgather ~%.1f KB (%.1fx reduction)",
./packages/core/legoesm/parallel/sharded_dynamics.py:2437:        logger.info("  ppermute schedule built in %.3fs", time.time() - t1)
./packages/core/legoesm/parallel/sharded_dynamics.py:2475:            if use_ppermute:
./packages/core/legoesm/parallel/sharded_dynamics.py:2476:                cell_local, u_local = _ppermute_halo_fill(
./packages/core/legoesm/parallel/sharded_dynamics.py:2477:                    cell_pack, u_shard, halo_sl, ppermute_perms,
./packages/core/legoesm/parallel/tiled_transport.py:5:body with NO in-stage halo ppermute):
./packages/core/legoesm/parallel/tiled_transport.py:18:the halo — NO in-stage halo ppermute, the same insight as the d2a2c stage).
./scripts/cluster/scaling_derecho/README.md:513:determinism) + `make_voronoi_sharded_step` ppermute halos. 6 processes
./tests/distributed/test_voronoi_halo.py:402:# ppermute schedule (for multi-GPU halo exchange)
./tests/distributed/test_voronoi_halo.py:406:    """Tests for the ppermute-based halo exchange schedule builder."""
./tests/distributed/test_voronoi_halo.py:410:        """Every halo cell is covered by exactly one ppermute round."""
./tests/distributed/test_voronoi_halo.py:416:            _build_ppermute_schedule,
./tests/distributed/test_voronoi_halo.py:428:        sched = _build_ppermute_schedule(
./tests/distributed/test_voronoi_halo.py:450:                f"not covered by ppermute schedule"
./tests/distributed/test_voronoi_halo.py:454:    def test_ppermute_halo_matches_simulated_exchange(self, mesh, n_ranks):
./tests/distributed/test_voronoi_halo.py:455:        """ppermute schedule produces same halo values as simulated
./tests/distributed/test_voronoi_halo.py:462:            _build_ppermute_schedule,
./tests/distributed/test_voronoi_halo.py:473:        sched = _build_ppermute_schedule(
./tests/distributed/test_voronoi_halo.py:488:            # Simulate ppermute rounds
./tests/distributed/test_voronoi_halo.py:490:                perm = sched['ppermute_perms'][r]
./tests/distributed/test_voronoi_halo.py:523:        """Each ppermute round has no device appearing as sender twice."""
./tests/distributed/test_voronoi_halo.py:529:            _build_ppermute_schedule,
./tests/distributed/test_voronoi_halo.py:539:        sched = _build_ppermute_schedule(
./tests/distributed/test_voronoi_halo.py:545:            senders = [s for s, _ in sched['ppermute_perms'][r]]
./tests/distributed/test_voronoi_halo.py:547:                f"Round {r}: duplicate senders in ppermute perm"
./packages/core/legoesm/parallel/latlon_spmd.py:10:  * latitude interior band cuts: ``jax.lax.ppermute`` of the ``halo`` edge rows
./packages/core/legoesm/parallel/latlon_spmd.py:20:foundation for the ocean lat-lon multi-GPU SPMD step (pure-jax ppermute over the
./packages/core/legoesm/parallel/latlon_spmd.py:86:    columns, moved by ``jax.lax.ppermute`` over the ``"lon"`` mesh axis (the
./packages/core/legoesm/parallel/latlon_spmd.py:97:    (trailing axes ride through).  AD-safe: ``ppermute`` is
./packages/core/legoesm/parallel/latlon_spmd.py:110:    east_ghost = jax.lax.ppermute(f[:, :halo], "lon", perm_to_west)
./packages/core/legoesm/parallel/latlon_spmd.py:111:    west_ghost = jax.lax.ppermute(f[:, -halo:], "lon", perm_to_east)
./packages/core/legoesm/parallel/latlon_spmd.py:124:    ``ppermute(..., perm_north)``; the north-most band has no neighbour there and
./packages/core/legoesm/parallel/latlon_spmd.py:125:    receives the pole-wall zero (the ppermute non-target). Pure array core (no
./packages/core/legoesm/parallel/latlon_spmd.py:128:    ``_reconstruct_v`` closure). AD-safe: ``ppermute`` is self-transposing.
./packages/core/legoesm/parallel/latlon_spmd.py:140:    boundary = jax.lax.ppermute(v_lower[0:1], axis, perm_north)
./packages/core/legoesm/parallel/latlon_spmd.py:156:    aggregation, scaling-M4): ONE ``ppermute`` per DTYPE GROUP for the whole
./packages/core/legoesm/parallel/latlon_spmd.py:163:    split are layout ops), including the north band's ppermute non-target
./packages/core/legoesm/parallel/latlon_spmd.py:201:        # ONE ppermute for the whole dtype group (north band receives 0).
./packages/core/legoesm/parallel/latlon_spmd.py:202:        recv = jax.lax.ppermute(buf, axis, perm_north)
./packages/core/legoesm/parallel/latlon_spmd.py:231:    periodic so EVERY tile is a ppermute target — the wrap pair
./packages/core/legoesm/parallel/latlon_spmd.py:252:        boundary = jax.lax.ppermute(u_left[:, 0:1], axis, perm_to_west)
./packages/core/legoesm/parallel/latlon_spmd.py:347:    """Tile's padded 180-deg pole-fold window via ONE antipodal ``ppermute``
./packages/core/legoesm/parallel/latlon_spmd.py:372:    AD-safe: ``ppermute`` is self-transposing; the window ``take`` is a
./packages/core/legoesm/parallel/latlon_spmd.py:387:    # 1. E/W-extend the edge rows by 2h (one lon ring ppermute pair).
./packages/core/legoesm/parallel/latlon_spmd.py:389:    # 2. ONE antipodal ppermute over the lon ring (shift by p_lon/2 is a
./packages/core/legoesm/parallel/latlon_spmd.py:392:    recv = jax.lax.ppermute(ext, "lon", perm_anti)
./packages/core/legoesm/parallel/latlon_spmd.py:445:    ppermute (interior) + pole fold (ends).  ``negate=True`` folds with a sign
./packages/core/legoesm/parallel/latlon_spmd.py:462:        # 2. latitude band ppermute of the edge rows (axis 0 = south->north,
./packages/core/legoesm/parallel/latlon_spmd.py:466:        north_recv = jax.lax.ppermute(south_edge, axis, perm_north)  # b's N ghost = b+1's south edge
./packages/core/legoesm/parallel/latlon_spmd.py:467:        south_recv = jax.lax.ppermute(north_edge, axis, perm_south)  # b's S ghost = b-1's north edge
./packages/core/legoesm/parallel/latlon_spmd.py:469:        # 3. pole fold at the end bands (ppermute non-targets receive zeros).
./packages/core/legoesm/parallel/latlon_spmd.py:491:    * latitude interior cuts: ``ppermute`` of the ``halo`` edge rows over
./packages/core/legoesm/parallel/latlon_spmd.py:494:      ring ``ppermute`` over ``"lon"`` (``p_lon == 1``: the LOCAL wrap, a
./packages/core/legoesm/parallel/latlon_spmd.py:506:      EVEN ``p_lon`` (with ``w >= 2h``): ONE antipodal ``ppermute`` of the
./packages/core/legoesm/parallel/latlon_spmd.py:522:    AD-safe: ``ppermute`` is self-transposing, ``all_gather`` has a defined
./packages/core/legoesm/parallel/latlon_spmd.py:533:        antipodal ppermute / all_gather — static branch on ``p_lon``)."""
./packages/core/legoesm/parallel/latlon_spmd.py:541:            # 180-deg partner-tile ppermute (the documented follow-up,
./packages/core/legoesm/parallel/latlon_spmd.py:554:        # 1. latitude ppermute of the (pre-lon-pad) edge rows over "lat".
./packages/core/legoesm/parallel/latlon_spmd.py:556:        # no lat neighbour exists, so skip the (empty-perm) ppermute
./packages/core/legoesm/parallel/latlon_spmd.py:562:            north_recv = jax.lax.ppermute(tile[:halo], "lat", perm_north)
./packages/core/legoesm/parallel/latlon_spmd.py:563:            south_recv = jax.lax.ppermute(tile[-halo:], "lat", perm_south)
./packages/core/legoesm/parallel/latlon_spmd.py:570:        # 3. pole fold at the physical pole tiles (ppermute non-targets
./packages/core/legoesm/parallel/latlon_spmd.py:593:    ppermute of the edge rows at INTERIOR cuts (so the cut ghost row is the
./packages/core/legoesm/parallel/latlon_spmd.py:607:    axis — the SAME ppermute, keyed on ``mesh.shape["lat"]`` (identical to
./packages/core/legoesm/parallel/latlon_spmd.py:619:        # lat-band ppermute of the edge rows (axis 0 = south->north; row 0 is the
./packages/core/legoesm/parallel/latlon_spmd.py:624:        north_recv = jax.lax.ppermute(south_edge, axis, perm_north)  # b's N ghost = b+1's south edge
./packages/core/legoesm/parallel/latlon_spmd.py:625:        south_recv = jax.lax.ppermute(north_edge, axis, perm_south)  # b's S ghost = b-1's north edge
./packages/core/legoesm/parallel/latlon_spmd.py:627:        # CONSTANT wall pad at the physical pole end bands (ppermute non-targets
./packages/core/legoesm/parallel/latlon_spmd.py:646:    One ``ppermute`` pair per DIRECTION for the whole field GROUP instead of
./packages/core/legoesm/parallel/latlon_spmd.py:659:    (and one ppermute pair) per dtype group, matching the MPI fused path's
./packages/core/legoesm/parallel/latlon_spmd.py:708:            # ONE ppermute pair for the whole dtype group.
./packages/core/legoesm/parallel/latlon_spmd.py:709:            north_recv = jax.lax.ppermute(south_buf, axis, perm_north)
./packages/core/legoesm/parallel/latlon_spmd.py:710:            south_recv = jax.lax.ppermute(north_buf, axis, perm_south)
./packages/core/legoesm/parallel/latlon_spmd.py:816:      face from the neighbour band's edge row (``ppermute`` via the armed spmd
./packages/core/legoesm/parallel/latlon_spmd.py:870:    COMMUNICATION VOLUME of the actual pad program (ppermute perimeter PLUS
./packages/core/legoesm/parallel/latlon_spmd.py:879:        exchange, moves edge blocks of that depth in ONE ppermute hop).
./packages/core/legoesm/parallel/latlon_spmd.py:885:      * lat cut (``p_lat > 1``): the N/S ppermute pair moves ``2h*w`` cells
./packages/core/legoesm/parallel/latlon_spmd.py:887:      * lon cut (``p_lon > 1``): the E/W ring ppermute pair moves ``~2h*nl``
./packages/core/legoesm/parallel/latlon_spmd.py:939:        # ppermute pair (w) + E/W ring pair (nl) + the pole-fold term —
./packages/core/legoesm/parallel/latlon_spmd.py:940:        # even p_lon (w >= 2*_FOLD_REF_HALO): partner-ppermute fold
./packages/core/legoesm/parallel/tiled_d2a2c.py:9:adjacent strips are applied IN-STAGE: four 1-cell ``lax.ppermute`` halos on
./packages/core/legoesm/parallel/tiled_d2a2c.py:98:        ut_hi = jax.lax.ppermute(ut_t[:, :, 0], "tile_j", perm_hi)
./packages/core/legoesm/parallel/tiled_d2a2c.py:99:        ut_lo = jax.lax.ppermute(ut_t[:, :, nl - 1], "tile_j", perm_lo)
./packages/core/legoesm/parallel/tiled_d2a2c.py:100:        vt_hi = jax.lax.ppermute(vt_t[:, 0, :], "tile_i", perm_hi)
./packages/core/legoesm/parallel/tiled_d2a2c.py:101:        vt_lo = jax.lax.ppermute(vt_t[:, nl - 1, :], "tile_i", perm_lo)
./scripts/cluster/scaling_derecho/mc_nccl_probe.py:13:  4. ppermute ring shift correct                            (halo path)
./scripts/cluster/scaling_derecho/mc_nccl_probe.py:14:  5. ppermute latency loop -> min/median ms — the number that decides whether
./scripts/cluster/scaling_derecho/mc_nccl_probe.py:16:     sendrecv floor and the ~0.11 ms ppermute floor measured intra-node in
./scripts/cluster/scaling_derecho/mc_nccl_probe.py:45:                   help="stage-5 ppermute latency-loop iterations")
./scripts/cluster/scaling_derecho/mc_nccl_probe.py:92:        y = jax.lax.ppermute(xl, "lat", perm)             # halo path
./scripts/cluster/scaling_derecho/mc_nccl_probe.py:111:    print(f"[rank {r}] stage4 ppermute ring OK={ok_ring}", flush=True)
./scripts/cluster/scaling_derecho/mc_nccl_probe.py:115:    # Stage 5: ppermute latency (small message — the halo latency regime).
./scripts/cluster/scaling_derecho/mc_nccl_probe.py:118:    g = jax.jit(shard_map(lambda a: jax.lax.ppermute(a, "lat", perm),
./scripts/cluster/scaling_derecho/mc_nccl_probe.py:127:    print(f"[rank {r}] stage5 ppermute latency ms: "
./scripts/cluster/scaling_derecho/ocean_gpu_scaling.pbs:16:# make_sharded_ocean_step (shard_map ppermute band halos + psum reductions
./packages/core/legoesm/parallel/halo_exchange.py:62:  - For pure multi-GPU (no MPI), ``jax.lax.ppermute`` is the preferred
./packages/core/legoesm/parallel/tiled_production_cdgrid.py:9:bit-exactly.  NOT Ginsburg-benchable (np>6 = CPU shard_map / cross-node ppermute
./packages/core/legoesm/parallel/tiled_production_cdgrid.py:2452:    the halo collective-permutes (``jax.lax.ppermute`` over the
./packages/core/legoesm/parallel/tiled_production_cdgrid.py:2470:    1. the halo ``ppermute`` cliques — every table the step's pad body issues
./packages/core/legoesm/parallel/tiled_production_cdgrid.py:2511:    # Every ppermute the blocked step's halo pad body issues, in the SAME table
./packages/core/legoesm/parallel/tiled_production_cdgrid.py:2530:            acc = acc + jax.lax.ppermute(v, AXES, perm)
./packages/core/legoesm/parallel/tiled_production_cdgrid.py:3005:# ``make_tiled_pad_vector_body`` vector) — the SAME ppermute-over-(face,tile_i,
./packages/core/legoesm/parallel/tiled_production_cdgrid.py:3184:# -> the per-tile PPM reconstruction is LOCAL (NO in-stage ppermute).  The cc
./scripts/cluster/scaling_derecho/mc_nccl_canary.sh:14:# process per GPU, ppermute/psum over NCCL) initialize and communicate ACROSS
./scripts/cluster/scaling_derecho/mc_nccl_canary.sh:26:#   A  mc_nccl_probe.py       — pure-JAX init + psum + ppermute ring + latency
./scripts/cluster/scaling_derecho/mc_nccl_canary.sh:87:# --- Stage A: pure-JAX probe (init + psum + ppermute + latency) -------------
./tests/parallel/test_spmd_schedule_cost.py:1:"""Direct tests for ``sharded_dynamics.spmd_schedule_cost``.
./tests/parallel/test_spmd_schedule_cost.py:22:    c = sd.spmd_schedule_cost(mesh, 4)
./tests/parallel/test_spmd_schedule_cost.py:40:    c = sd.spmd_schedule_cost(mesh, 8)
./tests/parallel/test_spmd_schedule_cost.py:46:    assert sd.spmd_schedule_cost(mesh, 1)["n_rounds"] == 0
./tests/parallel/test_spmd_schedule_cost.py:48:        sd.spmd_schedule_cost(mesh, 0)
./tests/parallel/test_spmd_schedule_cost.py:51:def test_flags_allgather_when_production_would_not_use_ppermute(mesh):
./tests/parallel/test_spmd_schedule_cost.py:53:    ppermute round count is then counterfactual and must say so."""
./tests/parallel/test_spmd_schedule_cost.py:54:    c = sd.spmd_schedule_cost(mesh, 8,
./tests/parallel/test_spmd_schedule_cost.py:55:                              ppermute_cells_per_device_threshold=10**9)
./tests/parallel/test_spmd_schedule_cost.py:57:    big = sd.spmd_schedule_cost(mesh, 8,
./tests/parallel/test_spmd_schedule_cost.py:58:                                ppermute_cells_per_device_threshold=1)
./tests/parallel/test_spmd_schedule_cost.py:59:    assert big["production_strategy"] == "ppermute"
./tests/parallel/test_spmd_schedule_cost.py:67:    pre = sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
./tests/parallel/test_spmd_schedule_cost.py:68:    raw = sd.spmd_schedule_cost(mesh, 8, method="sfc")
./tests/parallel/test_spmd_schedule_cost.py:77:        sd.spmd_schedule_cost(prepared, 8, already_reordered=True,
./tests/parallel/test_spmd_schedule_cost.py:85:    same = sd.spmd_schedule_cost(mesh, 4, method="sfc")
./tests/parallel/test_spmd_schedule_cost.py:86:    split_for_16 = sd.spmd_schedule_cost(mesh, 4, method="sfc",
./tests/parallel/test_spmd_schedule_cost.py:97:    a = sd.spmd_schedule_cost(mesh, 8, method="sfc")
./tests/parallel/test_spmd_schedule_cost.py:98:    b = sd.spmd_schedule_cost(mesh, 8, method="sfc")
./tests/parallel/test_spmd_schedule_cost.py:115:    r = {m: sd.spmd_schedule_cost(big, 16, method=m)["n_rounds"]
./tests/parallel/test_spmd_schedule_cost.py:125:        sd.spmd_schedule_cost(mesh, 4, method="not_a_method")
./tests/parallel/test_spmd_schedule_cost.py:138:    sd.spmd_schedule_cost(prepared, 8, already_reordered=True)
./tests/parallel/test_spmd_schedule_cost.py:147:        sd.spmd_schedule_cost(mesh, 4, method="sfc", reorder_target=3)
./tests/parallel/test_spmd_schedule_cost.py:152:        sd.spmd_schedule_cost(mesh, 3.9)
./tests/parallel/test_spmd_schedule_cost.py:160:    assert (inspect.signature(sd.spmd_schedule_cost)
./tests/parallel/test_spmd_schedule_cost.py:169:    assert sd.spmd_schedule_cost(mesh, 1)["production_strategy"] is None
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:1:"""#921 regression: multi-process MIXED-CLIQUE (psum + ppermute) rendezvous gate.
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:5:tiled step is halo-only — cliques built solely from ``jax.lax.ppermute``.  The
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:10:``jax.lax.ppermute(send_buf, ("face","tile_i","tile_j"), perm)``
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:17:ppermute rounds + 1 global psum — federated across TWO OS processes launched by
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:44:# exact axis names the fix_mass psum and the halo ppermute both reduce over.
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:55:    """A cyclic-shift ppermute perm: device ``i`` sends to ``(i+stride) % n``.
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:129:        """One tiled step: 4 halo ppermute rounds + 1 global mass-fixer psum.
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:133:        four edge exchanges are distinct ``ppermute`` cliques; the ``psum`` is
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:139:        top = jax.lax.ppermute(t[:, :, :, 0, :], AXES, _shift_perm(24, 1))
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:140:        bot = jax.lax.ppermute(t[:, :, :, -1, :], AXES, _shift_perm(24, -1 % 24))
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:141:        left = jax.lax.ppermute(t[:, :, :, :, 0], AXES, _shift_perm(24, 6))
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:142:        right = jax.lax.ppermute(t[:, :, :, :, -1], AXES, _shift_perm(24, -6 % 24))
./tests/parallel/test_tiled_mixed_clique_multicontroller_selfspawn.py:189:    """#921: the closed-loop psum+ppermute mix survives 2-process rendezvous."""
./scripts/cluster/scaling_derecho/blocker1_cube_shardmap.pbs:12:# cubed-sphere ppermute halo hold strong-scaling efficiency on REAL A100 NVLink?
./scripts/cluster/scaling_derecho/blocker1_cube_shardmap.pbs:17:# measures ppermute scaling 1->3 on NVLink and proves correctness + reverse-mode
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:90:# (lane E) throughput wall is the ppermute ROUND COUNT (9->24->30 as the SFC
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:109:    # send/recv (= ppermute halo) throughput knob on multi-NIC nodes.
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:166:# — native ppermute band halos (route-B) replace the route-A mpi4jax leg
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:210:# — cell-partition reorder + ppermute halo (make_voronoi_sharded_step) over
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:220:        # CP-combining + pipelined p2p ON by default (#1113): the ppermute
./scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:342:    # Ocean arms: base vs fused — the flag's original 25%-fewer-ppermutes
./scripts/cluster/scaling_derecho/cube_tiled_step.pbs:80:# collective-permutes (cubesphere_exchange ppermute over the SAME axes). Under
./tests/parallel/test_latlon_ocean_spmd_step.py:10:ppermute reconstruction), the replicated-stacked grid indexed by
./tests/parallel/test_latlon_ocean_spmd_step.py:141:    # the COMPOSED step (u ~1.5e-5, eta ~2.5e-5, v ~5.3e-6) is the ppermute/psum
./scripts/cluster/scaling_derecho/blocker1_cube_shardmap_xnode.pbs:12:# cubed-sphere ppermute halo hold strong-scaling efficiency PAST the np6 cliff?
./scripts/cluster/scaling_derecho/blocker1_cube_shardmap_xnode.pbs:22:#     (efficiency at largest >= PASS_EFF, ppermute proven on every N>1 point).
./scripts/cluster/scaling_derecho/blocker1_cube_shardmap_xnode.pbs:69:# NCCL-over-Slingshot (jax.distributed transport for the ppermute halo).
./tests/parallel/test_ppermute_edge_coloring.py:1:"""Edge-coloring correctness + round-count for the route-B MPAS ppermute
./tests/parallel/test_ppermute_edge_coloring.py:2:schedule (:func:`legoesm.parallel.sharded_dynamics._build_ppermute_schedule`).
./tests/parallel/test_ppermute_edge_coloring.py:4:Each color is one bidirectional ppermute ROUND and the route-B lane is
./tests/parallel/test_ppermute_edge_coloring.py:99:    ppermute loop directly)."""
./tests/parallel/test_ppermute_edge_coloring.py:104:        _build_ppermute_schedule,
./tests/parallel/test_ppermute_edge_coloring.py:119:    sched = _build_ppermute_schedule(
./tests/parallel/test_ppermute_edge_coloring.py:129:    # ppermute perms are proper: no device appears twice as a source or
./tests/parallel/test_ppermute_edge_coloring.py:131:    for perm in sched["ppermute_perms"]:
./tests/parallel/test_latlon_spmd_fused_halo.py:17:   reconstruction (v + v_mask in ONE ppermute per dtype group inside the
./tests/parallel/test_latlon_spmd_fused_halo.py:133:def _count_ppermutes(hlo_text: str) -> int:
./tests/parallel/test_latlon_spmd_fused_halo.py:139:    """5 same-dtype fields: per-field = 10 ppermutes (2/field), fused = 2."""
./tests/parallel/test_latlon_spmd_fused_halo.py:157:    n_fused = _count_ppermutes(
./tests/parallel/test_latlon_spmd_fused_halo.py:159:    n_per = _count_ppermutes(
./tests/parallel/test_latlon_spmd_fused_halo.py:172:# the sharded ocean step's v + v_mask staggered carriers ride ONE ppermute
./tests/parallel/test_latlon_spmd_fused_halo.py:189:    north band's ppermute non-target zero row."""
./tests/parallel/test_latlon_spmd_fused_halo.py:205:    # including the north band's ppermute non-target zero row.
./tests/parallel/test_latlon_spmd_fused_halo.py:225:        # north band's boundary row is the ppermute non-target zero
./tests/parallel/test_latlon_spmd_fused_halo.py:253:    n_fused = _count_ppermutes(fused_fn.lower(v, vm).compile().as_text())
./tests/parallel/test_latlon_spmd_fused_halo.py:254:    n_per = _count_ppermutes(per_fn.lower(v, vm).compile().as_text())
./tests/parallel/test_latlon_spmd_fused_halo.py:351:        hlo_counts[flag] = _count_ppermutes(
./tests/parallel/test_latlon_spmd_fused_halo.py:374:    n_off = _count_ppermutes(
./tests/parallel/test_latlon_spmd_fused_halo.py:378:    n_on = _count_ppermutes(
./tests/parallel/test_latlon_spmd_fused_halo.py:470:    slice/concat/ppermute, so a wrong transpose is not expected -- but "not
./tests/unit/test_no_module_top_jax_alloc.py:324:    iter-94d motivation: ``_build_ppermute_tables()`` returned
./tests/unit/test_no_module_top_jax_alloc.py:326:    ``_PPERMUTE_SEND, ... = _build_ppermute_tables()``. The AST
./tests/unit/test_no_module_top_jax_alloc.py:328:    ``_build_ppermute_tables()`` (a plain function call, not a
./tests/parallel/test_ppermute_multiface.py:1:"""Multi-face-per-shard ppermute SPMD halo: bit-identity, AD, scan, HLO guard.
./tests/parallel/test_ppermute_multiface.py:3:The multiface ppermute exchange (``cubesphere_exchange._make_exchange_
./tests/parallel/test_ppermute_multiface.py:4:ppermute_multiface``) replaces the all_gather SPMD halo that provably
./tests/parallel/test_ppermute_multiface.py:8:shard-locally, cross-shard edges ride a static device-pair ppermute schedule.
./tests/parallel/test_ppermute_multiface.py:11:``scripts/tmp/_codex_ppermute_multiface_out.txt``):
./tests/parallel/test_ppermute_multiface.py:18:* ``grad(scan(step))`` AD smoke vs the serial gradient (ppermute transpose
./tests/parallel/test_ppermute_multiface.py:27:        JAX_ENABLE_X64=1 python -m pytest tests/parallel/test_ppermute_multiface.py
./tests/parallel/test_ppermute_multiface.py:76:    (same hardening as test_ppermute_halo_exchange.py: restore
./tests/parallel/test_ppermute_multiface.py:77:    _use_ppermute, the backend string, both _spmd_mesh refs and the mpi
./tests/parallel/test_ppermute_multiface.py:79:    snap = (cx._use_ppermute, cx._spmd_mesh, halo_mod._halo_backend,
./tests/parallel/test_ppermute_multiface.py:83:    (cx._use_ppermute, cx._spmd_mesh, halo_mod._halo_backend,
./tests/parallel/test_ppermute_multiface.py:123:    fill or a ppermute slot.  (Coverage violations raise inside the
./tests/parallel/test_ppermute_multiface.py:165:    BOTH peers (6 directed pairs on 3 devices), so one ppermute round
./tests/parallel/test_ppermute_multiface.py:188:    cx.set_ppermute_default(True)
./tests/parallel/test_ppermute_multiface.py:206:    cx.set_ppermute_default(True)
./tests/parallel/test_ppermute_multiface.py:221:    cx.set_ppermute_default(True)
./tests/parallel/test_ppermute_multiface.py:237:    cx.set_ppermute_default(True)
./tests/parallel/test_ppermute_multiface.py:266:    cx.set_ppermute_default(True)
./tests/parallel/test_ppermute_multiface.py:276:    fields into ONE halo=2 multiface ppermute collective."""
./tests/parallel/test_ppermute_multiface.py:299:def test_activation_defaults_to_ppermute(n_devices):
./tests/parallel/test_ppermute_multiface.py:300:    """The n_devices==6 forcing is gone: activation selects ppermute for
./tests/parallel/test_ppermute_multiface.py:304:        assert cx._use_ppermute is True
./tests/parallel/test_ppermute_multiface.py:316:        assert cx._use_ppermute is False
./tests/parallel/test_ppermute_multiface.py:323:        assert cx._use_ppermute is False
./tests/parallel/test_ppermute_multiface.py:337:    cx.set_ppermute_default(True)
./tests/parallel/test_ppermute_multiface.py:342:        assert "collective-permute" in txt, f"halo={halo}: ppermute missing"
./tests/parallel/test_ppermute_multiface.py:345:    assert cx._select_variant(True, 1, 6) == "ppermute_oneface"
./tests/parallel/test_ppermute_multiface.py:346:    assert cx._select_variant(True, 2, 6) == "ppermute_multiface"
./tests/parallel/test_ppermute_multiface.py:347:    assert cx._select_variant(True, 1, 2) == "ppermute_multiface"
./tests/parallel/test_ppermute_multiface.py:364:    assert {k[2] for k in keys} == {"ppermute_multiface", "allgather"}
./tests/parallel/test_ppermute_multiface.py:385:    """grad(scan(step)) through the multiface ppermute exchange:
./tests/parallel/test_ppermute_multiface.py:386:    ppermute's transpose rule (inverse permutation) under shard_map
./tests/parallel/test_ppermute_multiface.py:441:    compiled scan hot path under the ppermute backend must contain ZERO
./tests/parallel/test_ppermute_multiface.py:518:    cx.set_ppermute_default(True)
./tests/parallel/test_ppermute_multiface.py:526:    assert "ppermute_oneface" in variants
./tests/parallel/test_ppermute_multiface.py:527:    assert "ppermute_multiface" in variants
./tests/parallel/test_ppermute_multiface.py:531:    """n_devices=1: zero ppermute rounds; the kernel is a pure shard-
./tests/parallel/test_ppermute_multiface.py:537:    cx.set_ppermute_default(True)
./tests/parallel/test_ppermute_multiface.py:550:    cx.set_ppermute_default(False)
./tests/parallel/test_tiled_fv3_hydrostatic_momentum.py:28:measurement (np24 on Ginsburg = CPU shard_map / cross-node ppermute, both
./tests/parallel/test_tiled_fv3_hydrostatic_momentum.py:33:the strip-ppermute DIRECTION (at kt=2 ``perm_hi == perm_lo``) and catches
./tests/parallel/test_tiled_fv3_hydrostatic_momentum.py:183:    # FMA-robust relative tolerance: the in-stage ppermute receives reorder the
./tests/parallel/test_tiled_fv3_hydrostatic_momentum.py:194:    the in-stage ppermute halo (which the np24 lane exercises)."""
./tests/parallel/test_tiled_fv3_hydrostatic_momentum.py:205:    # proves the ppermute halo equals these pads).
./tests/parallel/test_latlon_ocean_spmd_multicontroller.py:6:the band ppermute/psum halo (incl. the barotropic ``_global_sum_pair`` psum)
./tests/parallel/test_latlon_ocean_spmd_multicontroller.py:143:                f"'{nm}' — cross-process ppermute/psum or the multi-controller "
./tests/parallel/test_tiled_fv3_hydrostatic_tracer.py:13:Gated by BIT-IDENTITY (FMA-robust relative tol; the in-stage ppermute reorders
./tests/parallel/test_voronoi_sharded_equivalence.py:5:``shard_map`` kernel that does ppermute (or all_gather) halo exchange
./tests/parallel/test_voronoi_sharded_equivalence.py:10:allreduce merge, and the underlying ppermute/allgather kernels.
./tests/parallel/test_latlon_spmd_halo.py:3:The foundation for the ocean lat-lon multi-GPU SPMD step: lat-band ppermute +
./tests/parallel/test_latlon_spmd_halo.py:8:is 2 GPUs but the shard_map/ppermute logic is device-agnostic.
./tests/parallel/test_latlon_spmd_halo.py:120:# physical poles, neighbour-band ppermute at interior cuts) and the post-stencil
./tests/parallel/test_latlon_spmd_halo.py:131:    interior cuts read the neighbour band's edge row (ppermute), pole end bands
./tests/parallel/test_latlon_vface_reconstruct.py:5:boundary face = the next band's ``v_lower[0]``, lifted via ppermute; the
./tests/parallel/test_tiled_center_to_dgrid_vector.py:17:the strip-ppermute direction + nl-dependent slice bugs.
./tests/parallel/test_tiled_center_to_dgrid_vector.py:109:    slice arithmetic from the in-stage ppermute halo (which the np24 lane
./tests/parallel/test_latlon_ocean_spmd_multicontroller_selfspawn.py:11:devices, cross-process ppermute/psum inside the jitted step, the replication
./tests/parallel/test_tiled_fv3_hydrostatic_thermo.py:26:Gated by BIT-IDENTITY (FMA-robust relative tolerance; the in-stage ppermute
./tests/parallel/test_tiled_fv3_hydrostatic_thermo.py:29:cross-node ppermute, both anti-scale here); the future-HW win is the capability.
./tests/parallel/test_tiled_fv3_hydrostatic_thermo.py:33:the strip-ppermute DIRECTION (at kt=2 ``perm_hi == perm_lo``) and nl-dependent
./tests/parallel/test_tiled_fv3_hydrostatic_thermo.py:163:    ppermute halo (which the np24 lane exercises)."""
./tests/parallel/test_tiled_fv3_sw_momentum.py:18:cross-node ppermute, both anti-scale here); the future-HW win is the
./tests/parallel/test_tiled_fv3_sw_momentum.py:23:the strip-ppermute DIRECTION (at kt=2 ``perm_hi == perm_lo``), and nl=8 != 12
./tests/parallel/test_tiled_fv3_sw_momentum.py:113:    # contiguous arithmetic into per-tile ppermute receives (3 in-stage halos:
./tests/parallel/test_mpas_atm_native_step.py:1:"""Production-parity gates for the NATIVE (ppermute SPMD, non-mpi4jax)
./tests/parallel/test_mpas_atm_native_step.py:7:ppermute halo exchange + RK advection, traced per-step ``forcing``,
./tests/parallel/test_mpas_atm_native_step.py:229:    @pytest.mark.parametrize("halo_strategy", ["ppermute", "allgather"])
./tests/parallel/test_mpas_atm_native_step.py:235:        leaving the production ppermute rounds untested)."""
./tests/parallel/test_mpas_atm_native_step.py:285:            mesh, model, 2, halo_strategy="ppermute")
./tests/parallel/test_mpas_atm_native_step.py:336:            mesh32, model32, 2, halo_strategy="ppermute")
./tests/parallel/test_mpas_atm_native_step.py:373:            mesh, model, 2, halo_strategy="ppermute")
./tests/parallel/test_mpas_atm_native_step.py:447:            halo_strategy="ppermute")
./tests/parallel/test_mpas_atm_native_step.py:531:        global_id`` through the PRODUCTION pack helper, the ppermute
./tests/parallel/test_mpas_atm_native_step.py:546:            _build_ppermute_schedule,
./tests/parallel/test_mpas_atm_native_step.py:549:            _ppermute_halo_fill,
./tests/parallel/test_mpas_atm_native_step.py:565:        sched = _build_ppermute_schedule(
./tests/parallel/test_mpas_atm_native_step.py:610:        perms = sched["ppermute_perms"]
./tests/parallel/test_mpas_atm_native_step.py:613:            return _ppermute_halo_fill(
./tests/parallel/test_mpas_atm_native_step.py:676:            mesh, model, 2, halo_strategy="ppermute")
./tests/parallel/test_mpas_atm_native_step.py:679:            halo_strategy="ppermute")
./tests/parallel/test_tiled_d2a2c_ua_va.py:947:    halos — the host-slice mirror of the stage's 4 ppermutes) makes the
./tests/parallel/test_tiled_d2a2c_ua_va.py:1124:    the ppermute halo).  Approach C: the global fields are face-REPLICATED;
./tests/parallel/test_tiled_d2a2c_ua_va.py:1127:    NO dynamic wind halo (the strip ppermute is the only exchange).  kt=3 is
./tests/parallel/test_tiled_d2a2c_ua_va.py:1128:    the lane that pins the ppermute DIRECTION: at kt=2 perm_hi == perm_lo
./tests/parallel/test_tiled_d2a2c_ua_va.py:1227:    # PRODUCTION lane (apply_strips=True, the default): the 4-ppermute
./tests/parallel/test_latlon_spmd_pcg.py:24:ppermute + pole fold) makes the cross-band reduction load-bearing — a
./tests/parallel/test_latlon_spmd_pcg.py:28:the production target is 2 GPUs but the shard_map/ppermute/psum logic is
./tests/parallel/test_atm_latlon_spmd_multicontroller.py:6:``make_sharded_atm_latlon_step`` + the band ppermute/psum halo run UNCHANGED —
./tests/parallel/test_atm_latlon_spmd_multicontroller.py:132:                f"'{field}' — cross-process ppermute/psum or the "
./tests/parallel/test_tiled_exchange_tables.py:5:under the 6*kt^2-device tiled ppermute halo (task: break the 6-device
./tests/parallel/test_tiled_exchange_tables.py:154:    assert n_rounds == 4, f"expected 4 ppermute rounds, got {n_rounds}"
./tests/parallel/test_warmup_tiled_cube_comms.py:58:    # perm table or axis name were wrong the ppermute/psum would raise here.
./tests/parallel/test_warmup_tiled_cube_comms.py:86:            acc = acc + jax.lax.ppermute(v, AXES, perm)
./tests/parallel/test_latlon_ocean_spmd_multiprocess.py:105:    """Regular lat-lon: the cross-process ppermute/psum band halo + barotropic
./tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:11:cross-process partition checksum), cross-process ppermute halo rounds and
./tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:89:    boundary.  ``--halo-strategy ppermute`` is FORCED (auto would pick
./tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:91:    the cross-process ppermute rounds and their P("device")-sharded
./tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:111:            "--halo-strategy", "ppermute",
./tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:133:    assert rec["halo_strategy_requested"] == "ppermute"
./tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py:134:    assert rec["halo_strategy_effective"] == "ppermute"
./tests/parallel/test_atm_latlon_spmd_multicontroller_selfspawn.py:18:spanning NON-addressable devices, cross-process ppermute/psum inside the
./tests/parallel/test_tiled_fv3_hydrostatic_tendencies.py:23:elementwise).  FMA-robust relative tolerance (the in-stage ppermute reorders the
./tests/parallel/test_tiled_cgrid_gradient.py:111:    # orders ppermute receives differently from the global pad, so the
./tests/parallel/test_ppermute_halo_exchange.py:1:"""End-to-end SPMD cubed-sphere halo exchange (ppermute + all_gather) vs serial.
./tests/parallel/test_ppermute_halo_exchange.py:3:The ppermute halo backend (``cubesphere_exchange``, 4 rounds of
./tests/parallel/test_ppermute_halo_exchange.py:4:``jax.lax.ppermute`` inside ``shard_map``) is the JAX-0.10-native, mpi4jax-FREE
./tests/parallel/test_ppermute_halo_exchange.py:7:that the ppermute exchange BIT-MATCHES the serial/local halo (and so does
./tests/parallel/test_ppermute_halo_exchange.py:13:        JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/parallel/test_ppermute_halo_exchange.py
./tests/parallel/test_ppermute_halo_exchange.py:51:    globals are test-fragile — restore _use_ppermute, the backend string, both
./tests/parallel/test_ppermute_halo_exchange.py:53:    snap = (cx._use_ppermute, cx._spmd_mesh, halo_mod._halo_backend,
./tests/parallel/test_ppermute_halo_exchange.py:57:    (cx._use_ppermute, cx._spmd_mesh, halo_mod._halo_backend,
./tests/parallel/test_ppermute_halo_exchange.py:68:@pytest.mark.parametrize("use_ppermute", [True, False])
./tests/parallel/test_ppermute_halo_exchange.py:69:def test_spmd_scalar_halo_matches_serial(use_ppermute):
./tests/parallel/test_ppermute_halo_exchange.py:73:    cx.set_ppermute_default(use_ppermute)
./tests/parallel/test_ppermute_halo_exchange.py:79:@pytest.mark.parametrize("use_ppermute", [True, False])
./tests/parallel/test_ppermute_halo_exchange.py:80:def test_spmd_4d_halo_matches_serial(use_ppermute):
./tests/parallel/test_ppermute_halo_exchange.py:84:    cx.set_ppermute_default(use_ppermute)
./tests/parallel/test_ppermute_halo_exchange.py:91:    """The HLO must actually contain collective-permute for ppermute and
./tests/parallel/test_ppermute_halo_exchange.py:97:    cx.set_ppermute_default(True)
./tests/parallel/test_ppermute_halo_exchange.py:102:    cx.set_ppermute_default(False)
./tests/parallel/test_ppermute_halo_exchange.py:109:    """Vector ppermute path (explicit_pad_halo_vector_4d) — identity rotation
./tests/parallel/test_ppermute_halo_exchange.py:118:    cx.set_ppermute_default(True)
./tests/parallel/test_ppermute_halo_exchange.py:125:def test_halo2_routes_ppermute_multiface_and_matches():
./tests/parallel/test_ppermute_halo_exchange.py:126:    """halo=2 with the ppermute default ON routes to the multiface
./tests/parallel/test_ppermute_halo_exchange.py:127:    ppermute kernel (k=1 at 6 devices) — it must bit-match serial AND
./tests/parallel/test_ppermute_halo_exchange.py:130:    allgather_h2; see tests/parallel/test_ppermute_multiface.py for
./tests/parallel/test_ppermute_halo_exchange.py:136:    cx.set_ppermute_default(True)
./tests/parallel/test_ppermute_halo_exchange.py:146:def test_ppermute_schedule_covers_24_adjacencies_once():
./tests/parallel/test_ppermute_halo_exchange.py:147:    """Invariant the ppermute correctness relies on: the 4 rounds cover all 24
./tests/parallel/test_ppermute_halo_exchange.py:176:    cx.set_ppermute_default(False)
./tests/parallel/test_tiled_fv3_sw_full.py:108:    safety of the in-stage halo ppermutes (codex audit MED).  This drives
./tests/parallel/test_tiled_fv3_sw_full.py:110:    PPM + the *_core ops) — the ppermute VJP is a ppermute with the inverse
./tests/parallel/test_cubesphere_exchange_tables.py:18:  back to itself (this is what enables the ppermute schedule
./tests/parallel/test_cubesphere_exchange_tables.py:19:  generation in ``_build_ppermute_tables``).
./tests/parallel/test_cubesphere_exchange_tables.py:205:    This is the property ``_build_ppermute_tables`` relies on when
./tests/parallel/test_cubesphere_exchange_tables.py:207:    ppermute schedule would be ill-defined and the halo would not
./tests/parallel/test_atm_latlon_2d_tiling.py:6:adds a native 2-D tiling: periodic longitude as a cyclic ring ppermute,
./tests/parallel/test_atm_latlon_2d_tiling.py:8:EXACT serial 180-deg pole fold under a lon split (antipodal partner ppermute
./tests/parallel/test_atm_latlon_2d_tiling.py:45:(6)  CHOOSER — modeled pad communication volume (ppermute perimeter PLUS the
./tests/parallel/test_atm_latlon_2d_tiling.py:226:    the antipodal partner ppermute — collective-permute in HLO) while
./tests/parallel/test_atm_latlon_2d_tiling.py:238:        "(partner ppermute lever)")
./tests/parallel/test_atm_latlon_spmd_step.py:33:the production target is multi-GPU/TPU but the shard_map/ppermute/psum logic is
./tests/parallel/test_tiled_pad_body.py:9:1. wrapped ``_make_exchange_ppermute_tiled`` still equals the serial
./tests/parallel/test_tiled_pad_body.py:36:    _make_exchange_ppermute_tiled,
./tests/parallel/test_tiled_pad_body.py:63:    fn = _make_exchange_ppermute_tiled(mesh, ndim=3, halo=1,
./tests/parallel/test_tiled_pad_body.py:85:    wrapped = _make_exchange_ppermute_tiled(
./tests/parallel/test_tiled_mass_divergence.py:8:ghost); the per-tile reconstruction is LOCAL (no in-stage ppermute).  The cc
./tests/parallel/test_sharded_step_sharding_tripwire.py:36:``tests/parallel/test_ppermute_halo_exchange.py``.
./tests/parallel/test_sharded_step_sharding_tripwire.py:42:# (same pattern as tests/parallel/test_ppermute_halo_exchange.py).
./tests/parallel/test_cubesphere_exchange.py:3:Validates that both the all_gather and ppermute backends produce
./tests/parallel/test_cubesphere_exchange.py:74:# ppermute backend
./tests/parallel/test_cubesphere_exchange.py:81:            set_ppermute_default, explicit_pad_halo, _cache,
./tests/parallel/test_cubesphere_exchange.py:84:        set_ppermute_default(True)
./tests/parallel/test_cubesphere_exchange.py:92:                                       err_msg="ppermute 3D differs from local")
./tests/parallel/test_cubesphere_exchange.py:94:            set_ppermute_default(False)
./tests/parallel/test_cubesphere_exchange.py:102:            set_ppermute_default, explicit_pad_halo_4d, _cache,
./tests/parallel/test_cubesphere_exchange.py:105:        set_ppermute_default(True)
./tests/parallel/test_cubesphere_exchange.py:112:                                       err_msg="ppermute 4D differs from local")
./tests/parallel/test_cubesphere_exchange.py:114:            set_ppermute_default(False)
./tests/parallel/test_cubesphere_exchange.py:434:    def test_halo1_3d_with_offsets_ppermute(self, mesh_6):
./tests/parallel/test_cubesphere_exchange.py:435:        """ppermute backend with offsets matches local pad_halo
./tests/parallel/test_cubesphere_exchange.py:437:        ppermute kernel so the bandwidth-optimal high-resolution path
./tests/parallel/test_cubesphere_exchange.py:442:            explicit_pad_halo, set_ppermute_default,
./tests/parallel/test_cubesphere_exchange.py:450:        set_ppermute_default(True)
./tests/parallel/test_cubesphere_exchange.py:457:            set_ppermute_default(False)
./tests/parallel/test_cubesphere_exchange.py:499:    ``_make_exchange_allgather_h2``, ``_make_exchange_ppermute``) call
./tests/unit/test_production_blockers.py:7:4. ppermute halo safety for sub-face tiling
./tests/unit/test_production_blockers.py:344:# 4. ppermute halo safety for sub-face tiling
./tests/unit/test_production_blockers.py:348:    """ppermute path must fall back when tiling != (1,1)."""
./tests/unit/test_production_blockers.py:350:    def test_ppermute_rejects_tiled_mesh(self):
./tests/unit/test_production_blockers.py:351:        """When active config has tiling != (1,1), ppermute must fall back."""
./tests/unit/test_production_blockers.py:375:    def test_ppermute_allows_face_only_mesh(self):
./tests/unit/test_production_blockers.py:376:        """When tiling == (1,1), ppermute path is allowed (with experimental warning)."""
./tests/unit/test_production_blockers.py:393:             patch("legoesm.parallel.async_halo._ppermute_halo_exchange", return_value=data) as mock_pp:
./tests/unit/test_production_blockers.py:398:    def test_ppermute_no_mesh_uses_local_pad(self):
./tests/unit/test_production_blockers.py:399:        """mesh=None must always use local pad_halo, no ppermute."""
./scripts/cluster/omip_nemo/run_multiprocess_cpu_equiv.sbatch:21:# 4-GPU smoke.  No mpi4jax needed (the SPMD halo is pure-JAX ppermute/psum).
./scripts/cluster/omip_nemo/run_eorca025_4gpu_multinode.sbatch:52:# per-process GPU from SLURM_LOCALID; the SPMD halo is pure-JAX ppermute/psum
./docs/performance/multinode_gpu_direct_cxi.md:31:  (M3c native-ppermute step, `bench_mpas_spmd_scaling`) route-B lanes
./docs/performance/issue_852_cube_shardmap_rootcause.md:7:defect**. The ppermute cube halo is **bit-exact**. The decomposition dependence
./docs/performance/issue_852_cube_shardmap_rootcause.md:18:| `explicit_pad_halo` forward + VJP (bare ppermute halo), n=2 & n=3, C24–C192 | float32 | **Δ = 0.0** (bit-exact) |
./docs/performance/scaling/scaling_theoretical_limit_report_2026-06-15.md:60:| GPU | Quadro RTX 8000 / A40, in **PCIe pairs** | no NVLink/no P2P → cross-GPU `ppermute` 70–245× slower than HBM (107 µs latency floor, 1.7 GB/s @256 KB, 5.8 GB/s asymptotic, jobs 8458801/2) |
./docs/performance/scaling/scaling_theoretical_limit_report_2026-06-15.md:76:| Multiface ppermute halo (fix fake 2-GPU) | cube 2-GPU | eff 0.46→0.64 (f32 C192), **0.73 f64** (PCIe roofline) | 8457520 |
./docs/performance/scaling/SCALING_STATUS_AUDIT.md:28:| **icosahedral / MPAS** | (a) graph-partition domain decomposition (METIS/RCB/SFC `auto`), validated vs serial ~1e-9 | (a) multi-GPU via route-A mpi4jax (opaque to XLA overlap); (c) route-B multi-controller NCCL SPMD wired since M3c #981 (native ppermute step, `bench_mpas_spmd_scaling --multicontroller`, cluster lane E; XLA collective-permute combining defaulted for the lane, #1113) — no production-scale receipts yet | Derecho 2026-07-02: CPU-MPI to 128 ranks f32 ~75× (eff ~0.59); GPU 1→16 A100 @28 km ~6× (eff ~0.38), coarse grids flat (per-device floor, not a defect) | route-A mpi4jax leg is latency-bound, no comm/compute overlap; route-B ppermute round count (edge-coloring rounds × launch latency, #1113) | production-size lane-E runs; re-measure 8→16 GPU leg on native ppermute |
./docs/performance/scaling/SCALING_STATUS_AUDIT.md:29:| **lat-lon** | (a) latitude-band decomposition (`make_latlon_mpi_step`, #641; pole_bc='wall'), validated; driven by `run_levante_gpu_scaling.py --grid latlon` | (a) lat-band SPMD `bench_atm_latlon_spmd_scaling.py`: single-process multi-device + `--multicontroller` route-B (native NCCL ppermute, no mpi4jax) | Derecho: CPU-MPI 128 ranks f32 ~30× (eff ~0.23 — 1-D band perimeter cost, as designed); GPU 16 A100 @28 km 7.5× (eff ~0.47, route-A). Ginsburg 2-GPU: strong 1.10×, weak eff 0.64 (PCIe-capped) | 1-D band decomposition perimeter at high rank counts; 2-D latlon decomposition untested on a real fabric | production-size native-NCCL multicontroller runs on Derecho/Levante (job lanes C exist) |
./docs/performance/scaling/SCALING_STATUS_AUDIT.md:97:   route-B MPAS ppermute rounds are now minimized by a multi-start edge
./docs/performance/scaling/SCALING_STATUS_AUDIT.md:101:   `tests/parallel/test_ppermute_edge_coloring.py`); the data-parallel
./docs/performance/scaling/SCALING_STATUS_AUDIT.md:161:  ppermute): `xla_gpu_enable_latency_hiding_scheduler=true` is ON by default
./docs/performance/scaling/SCALING_STATUS_AUDIT.md:162:  (`runtime/backend.py`), so ppermute collectives are overlapped by XLA in
./docs/performance/scaling/bcw_scaling_status.md:37:(`model.step_with_physics`) under the jax.distributed multiface-ppermute halo. The
./docs/performance/scaling/bcw_scaling_status.md:308:ppermute halo — no mpi4jax, and `jax.distributed` is skipped for one process.
./docs/performance/scaling/bcw_scaling_status.md:323:ppermute halo over the shared PCIe-Gen3 link (no NVLink on these RTX-8000 pairs)
./docs/performance/scaling/bcw_scaling_status.md:385:Ginsburg's CPU shard_map / cross-node ppermute by design, so it is gated by
./docs/performance/scaling/scaling_review_2026-06-13.md:54:   with ppermute halos + merge small collectives. A/B-gated (low-impact on
./docs/performance/scaling/d2a2c_spmd_stage_design.md:24:  `make_tiled_pad_vector_body(...)`, `_make_exchange_ppermute_tiled(...)`,
./docs/performance/scaling/d2a2c_spmd_stage_design.md:26:  perfect-matching peel gives the 4-round ppermute schedule.
./docs/performance/scaling/d2a2c_spmd_stage_design.md:63:4. **Strip ppermute (the only dynamic exchange).** The vt i-strip (i=0/n-1)
./docs/performance/scaling/d2a2c_spmd_stage_design.md:67:   `lax.ppermute` on the `tile_i`/`tile_j` mesh axes (a 1-wide analogue of the
./docs/performance/scaling/d2a2c_spmd_stage_design.md:85:  (the ppermute is `lax.ppermute`, AD-safe).
./docs/performance/scaling/d2a2c_spmd_stage_design.md:97:- The strip ppermute schedule at corners (a corner defers BOTH strips, and the
./docs/dev-notes/mpi_local_setup.md:75:(`jax.lax.ppermute` over a device mesh — already noted in
./packages/core/legoesm/core/fv3_sw_core.py:1460:    stage applies them via the same-face-neighbour ppermute / global
./packages/core/legoesm/core/fv3_sw_core.py:1767:    boundary, so the stage can feed periodic-``ppermute`` halos
./docs/dev-notes/fv3_faithful.md:127:   (serial/MPI + the explicit SPMD all_gather/ppermute backends; restructure 2026-06-02 added the single-field
./docs/performance/scaling/RESUME_STATE_2026-06-13.md:19:| adaca24f | P4 tiled-d2a2c in-stage strip ppermute (np>6 sub-face exchange) |
./docs/performance/scaling/distance_to_limit_2026-06-13.md:14:| Cubed-sphere 2-GPU strong | eff 0.73 (f64) | **AT PCIe-link roofline** — ppermute 70-245× < HBM (measured); headroom needs NVLink/IB (absent on Ginsburg) |
./docs/performance/scaling/distance_to_limit_2026-06-13.md:48:   ppermutes already done; remaining cuts are CROSS-STAGE (independent
./docs/performance/scaling/distance_to_limit_2026-06-13.md:58:   proven but interconnect-capped on Ginsburg PCIe (ppermute < SYS/PCIe <
./docs/performance/scaling/cube_transport_tiling_design.md:36:ppermute copy between same-face neighbour tiles) + the existing depth-2
./docs/performance/scaling/cube_transport_tiling_design.md:77:NOT a new ppermute exchange:
./docs/performance/scaling/cube_transport_tiling_design.md:126:  the replicated padded face — NO halo ppermute (the padded face already
./docs/performance/scaling/cube_transport_tiling_design.md:146:   ppermutes its 4 boundary cells to the same-face neighbour tile; at
./docs/performance/scaling/derecho_levante_sota_review_2026-07.md:75:- shard_map+ppermute = JAX-Fluids-validated architecture (0.95 weak eff @512 A100).
./docs/performance/scaling/derecho_levante_sota_review_2026-07.md:180:   the ceiling the native-NCCL multi-controller path removes: ppermute
./docs/performance/scaling/derecho_levante_sota_review_2026-07.md:200:     1-proc-per-GPU launches into one JAX program (native ppermute band
./docs/performance/scaling/derecho_levante_sota_review_2026-07.md:245:7. **Pipelined-p2p XLA experiment** for scan-loop ppermutes
./docs/performance/scaling/scaling_indicators.csv:11:2026-06-10,multiface,ppermute_fix,atm_cube,gpu,strong,0.64,eff_2gpu_f32,8457520,C192 f32 post multiface-ppermute
./docs/performance/scaling/scaling_indicators.csv:12:2026-06-10,multiface,ppermute_fix,atm_cube,gpu,strong,0.73,eff_2gpu_f64,8457520,C192 f64 (PCIe roofline)
./docs/performance/scaling/scaling_indicators.csv:80:2026-06-18,8698eefdd,multinode_cube_included,atm_cube,cpu,both,0,allgrids_mn_complete,8522132,Cube cs-spmd multi-node added to the clean plot (fix: --cs-spmd is dry-dycore-only so physics=none). f64 strong n1=16.0 best -> n2=5.9 -> n3=9.4 -> n6=11.9 = ANTI-scaling (cross-node ppermute latency-bound on Gloo/PCIe; cube sub-face tiling is future-HW capability per the campaign verdict NOT a Gloo speedup). ALL MPI grids (cube/latlon/ico atm + ocean) x f32/f64 x weak/strong now have clean controlled multi-node curves; spectral single-device by design. docs/scaling/multinode_clean/throughput_by_grid.png
./docs/performance/scaling/scaling_indicators.csv:81:2026-06-18,8523465,scaling_efficiency_plot,multi,cpu,both,0,clean_efficiency_view,8523462,Fixed the messy weak/strong plots PROPERLY (user asked twice): root cause = problem sizes too small (strong auto-res=64 + weak base LL64/CS24/ICO4 -> halo-latency-bound dips) + raw-throughput presentation (crossing lines). Fix: re-ran STRONG at LARGE compute-bound res (latlon256/ico6/cube-C96-cs-spmd/ocean-LL288, jobs 8523462 f64 + 8523463 f32) + new parallel-efficiency plotter scripts/plot/plot_scaling_efficiency.py (E(N)=mc(N)*N0/(N*mc(N0)) vs nodes + ideal=1 line, normalized). CLEAN now: ocean ~0.85-1.0 (weak near-flat), ico ~0.7-0.75, latlon ~0.5@8nodes (halo-bound but monotonic, dips gone), cube ~0.25 (Gloo cross-node ppermute latency = future-HW, honest). docs/scaling/multinode_clean/scaling_efficiency.png
./docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md:64:### 2. Field-coalesced halo exchange — one `ppermute` over STACKED fields — Walls A & C. CHEAP, HIGH ROI on PCIe/Gloo.
./docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md:68:`ppermute` → unstack. On PCIe/no-IB where per-message LATENCY dominates,
./docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md:70:(cube-path halo fusion, cut 54 ppermutes/step).
./docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md:139:  compiled `ppermute` schedule across `lax.scan`. Borrow only the
./docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md:140:  *principle* (emit halo-independent interior compute between a `ppermute`
./docs/performance/scaling/literature_neuralgcm_veros_mpas_2026-06.md:144:  `ppermute` with MXU matmul), but the win NEEDS a fabric that overlaps
./docs/performance/scaling/spmd_message_census_2026-07-08.md:14:Since the transport is already overlappable (route-B ppermute, not route-A
./docs/performance/scaling/spmd_message_census_2026-07-08.md:45:devices = ~11.5 exchange points × 4 ppermute rounds (the edge-coloring
./docs/performance/scaling/spmd_message_census_2026-07-08.md:47:pairwise ppermute).
./docs/performance/scaling/spmd_message_census_2026-07-08.md:72:25%-fewer-ppermutes receipt) also covers the atm step via
./docs/performance/scaling/mpas_atm_native_step_audit.md:1:# MPAS-atmosphere NATIVE (ppermute SPMD) step — production-support audit (2026-07-13)
./docs/performance/scaling/mpas_atm_native_step_audit.md:4:MPAS-atmosphere ppermute code exists but production uses blocking mpi4jax
./docs/performance/scaling/mpas_atm_native_step_audit.md:9:shard_map + `jax.lax.ppermute`, single- and multi-controller); production
./docs/performance/scaling/mpas_atm_native_step_audit.md:32:| 9 | Local-only metadata | ALL n_dev local meshes + ppermute index arrays REPLICATED per device (1858-1891); global `areaCell` replicated closure (2075) | route-A rank holds only `layout.local_mesh` (voronoi_mpi.py:721) | **CLOSED for dynamics** — stacked local meshes, halo schedule, and areaCell are `P("device")`-sharded jit ARGUMENTS (multi-controller-safe where sharded closures raise); each device holds only its slice. REMAINDER: the operator-split physics term (below) |
./docs/performance/scaling/mpas_atm_native_step_audit.md:39:  full state through the packed ppermute/allgather exchange (one flat
./docs/performance/scaling/mpas_atm_native_step_audit.md:46:* `_ppermute_halo_fill` factored module-level so the exchange is
./docs/performance/scaling/mpas_atm_native_step_audit.md:51:  for moist kessler × {ppermute, allgather}, traced-forcing no-retrace,
./docs/performance/scaling/mpas_atm_native_step_audit.md:126:3. MAJOR (cross-process ppermute untested): auto strategy picks
./docs/performance/scaling/mpas_atm_native_step_audit.md:129:   FORCES ppermute and asserts the recorded strategy.
./packages/core/legoesm/core/conservation.py:326:    wrongly attributed there to the ppermute halo — the halo is bit-exact; this
./docs/performance/scaling/levante_campaign_2026-07-24.md:371:v-row-reconstruction machinery present, ppermutes self-to-self so no real
./docs/performance/scaling/levante_campaign_2026-07-24.md:861:| `ncclDevKernel_SendRecv` (halo) | 311.8 ms / 1440 | 186.7 / 1116 | ~120 launches |
./docs/performance/scaling/levante_campaign_2026-07-24.md:895:| **of which NCCL SendRecv (halo)** | **6.65 (44 %)** |
./docs/performance/scaling/levante_campaign_2026-07-24.md:990:* My "88 SendRecv / 4 rounds = 22 exchanges" was wrong. The kt=2 tile pad
./docs/performance/scaling/levante_campaign_2026-07-24.md:999:  optimized HLO ops and 88 traced SendRecv. So field packing's ceiling
./docs/performance/scaling/levante_campaign_2026-07-24.md:1007:**THE BOUND (one step, 86 SendRecv, latency floor 17.8 us measured):**
./docs/performance/scaling/levante_campaign_2026-07-24.md:1030:**SKEW LOCATED (4-rank profile, job 26526100).** Aligning SendRecv
./docs/performance/scaling/levante_campaign_2026-07-24.md:1391:`scripts/bench/bench_ppermute_microbench.py` (new) times the SAME collective
./docs/performance/scaling/levante_campaign_2026-07-24.md:1392:the sharded steps use — `lax.ppermute` on a ring inside `shard_map` — over a
./docs/performance/scaling/levante_campaign_2026-07-24.md:1499:   `bench_ppermute_microbench.py` measured the fabric constants (NVLink
./docs/performance/scaling/levante_campaign_2026-07-24.md:1911:  exchange is a hand-rolled edge-coloured ppermute schedule, so an
./docs/performance/scaling/levante_campaign_2026-07-24.md:2148:  measures ppermute rings with dispatch subtracted — not production
./docs/performance/scaling/levante_campaign_2026-07-24.md:2242:ppermute halo, the fixed replicate/put path) and wire the OCEAN MPAS
./docs/performance/scaling/levante_campaign_2026-07-24.md:2285:scheduling-bound — explains the overlap null) and NCCL SendRecv medians
./docs/performance/scaling/levante_campaign_2026-07-24.md:2432:4.47x at s9@64), and one halo fill costs 12-14 SEQUENTIAL ppermute rounds.
./docs/performance/scaling/levante_campaign_2026-07-24.md:2435:(12 fewer SendRecv calls/step x 218 us) and a hardware-only floor of
./docs/performance/scaling/levante_campaign_2026-07-24.md:2440:colourer's own lower bound. `_build_ppermute_schedule` produces a proper
./docs/performance/scaling/levante_campaign_2026-07-24.md:2455:`spmd_schedule_cost` rather than its own 1-ring proxy):
./docs/performance/scaling/levante_campaign_2026-07-24.md:2460:| L6 lloyd=0 | 8 | 7 | 7 | 6 | 0 | ppermute (5,121 cells/dev) |
./docs/performance/scaling/levante_campaign_2026-07-24.md:2461:| L6 lloyd=0 | 16 | 13 | 10 | 10 | 0 | ppermute (2,561 cells/dev) |
./docs/performance/scaling/levante_campaign_2026-07-24.md:2462:| L8 lloyd=0 | 64 | 16 | - | - | 0 | ppermute |
./docs/performance/scaling/levante_campaign_2026-07-24.md:2468:  no ppermute schedule there. Only L6@16 upward are real. L6@8's
./docs/performance/scaling/levante_campaign_2026-07-24.md:2471:  nothing more. `_build_ppermute_schedule` enters a device pair into
./docs/performance/scaling/levante_campaign_2026-07-24.md:2473:  BOTH ppermute directions, even where one send map is empty — so a
./docs/performance/scaling/levante_campaign_2026-07-24.md:2485:`spmd_schedule_cost`'s docstring (s8 sfc 12/14, metis 13/19, geometric
./docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:57:face has 4 neighbours ⇒ 4 ppermute rounds × the exchange points) — this IS the
./docs/performance/scaling/cube_moist_tiled_step_design.md:27:  vector, incl. the diagonal-corner ppermute) is bit-exact to serial
./docs/performance/scaling/cube_moist_tiled_step_design.md:76:   12 passed — moist_step[3] + tracer[3]/pack[3]/tendency[3], the strip-ppermute
./docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md:125:atm GPU: ppermute pad fusion intentionally NOT pursued — micro-bench
./docs/performance/scaling/ginsburg_mpi_gpu_scaling_plan.md:219:  applied IN-stage via 4 one-cell `lax.ppermute` halos feeding the new
./tests/unit/test_driver_spmd_halo_lifecycle.py:5:(``ppermute`` / ``all_gather``) so the compiled segment kernel can lower
./tests/unit/test_driver_spmd_halo_lifecycle.py:208:        # ppermute kernel assumes one face per shard, so activation
./docs/performance/scaling/latlon_2d_decomposition_design.md:229:n_lat`.  The **single-controller SPMD leg** (`shard_map` + `ppermute` over
./docs/performance/scaling/latlon_2d_decomposition_design.md:235:* **Periodic longitude = cyclic ring ppermute** (`latlon_lon_ring_perms`,
./docs/performance/scaling/latlon_2d_decomposition_design.md:244:  Follow-up optimisation: a 180°-partner ppermute pair (even `p_lon`)
./docs/performance/scaling/latlon_2d_decomposition_design.md:263:  partner-ppermute fold lands and removes the gather term.  The 1-D band
./docs/performance/scaling/latlon_2d_decomposition_design.md:270:on the MPI leg); GPU A/B of the fold all_gather vs partner-ppermute; wiring
./docs/performance/scaling/scaling_levers_audit_2026-06-14.md:14:   cleanest comm (N/S ppermute halos + psum barotropic reductions). Generic lat
./docs/dev-notes/GPU_SCALING_BRANCH.md:41:| Cubed-sphere SPMD halo=1 with offsets at high-res  |  all_gather|    ppermute (~50× less interconnect) |
./docs/dev-notes/GPU_SCALING_BRANCH.md:227:- **Multi-face ppermute**: the all_gather-based SPMD halo kernels
./docs/dev-notes/GPU_SCALING_BRANCH.md:229:  (`n_faces_per_shard ∈ {1, 2, 3, 6}`, iter-49); the ppermute kernel
./docs/dev-notes/GPU_SCALING_BRANCH.md:233:  multiple of the ppermute payload up to the 6-device limit — so this
./docs/dev-notes/GPU_SCALING_BRANCH.md:314:a7f4de67 cubesphere_exchange: ppermute kernel applies interp_offsets too
./docs/performance/scaling/scaling_bottleneck_audit_2026-06-10.md:5:nodes. Fixes already landed this campaign: multiface ppermute halo (killed
./docs/performance/scaling/scaling_bottleneck_audit_2026-06-10.md:37:+ ppermute halo + JAX-native reductions across nodes), **NOT** mixing mpi4jax
./docs/performance/scaling/literature_parallelization_2026-06.md:48:5. **Our shard_map+ppermute architecture is the literature's winning
./docs/performance/scaling/literature_parallelization_2026-06.md:50:   pmap+ppermute *to preserve differentiability*, and weak-scales ω>0.95 to
./docs/performance/scaling/literature_parallelization_2026-06.md:55:   halos in ordered directional ppermutes (diagonals relay through faces —
./docs/performance/scaling/literature_parallelization_2026-06.md:67:   ppermute+partial-compute "collective matmul" removed ~77% of the
./docs/performance/scaling/literature_parallelization_2026-06.md:86:    ppermute; a transport swap must never mix with jax.distributed in one
./docs/performance/scaling/cube_production_tiling_design.md:11:shard_map which anti-scales on Gloo-TCP, or cross-node ppermute 70-245× < HBM).
./docs/performance/scaling/cube_production_tiling_design.md:19:3. device-uniform local body (the op's stencil) — NO in-stage ppermute.
./docs/performance/scaling/cube_production_tiling_design.md:96:  ppermute) was already built (U3). Base case only: div_damp=0, hyperdiff=0,
./docs/performance/scaling/cube_production_tiling_design.md:140:  in-stage ppermutes -> cross-NODE collective-permute); each process
./docs/performance/scream_parity_scope.md:69:- **(B) GSPMD `ppermute` over NCCL/ICI** — cube-SPMD + lat-lon-band-SPMD, wired
./docs/performance/scream_parity_scope.md:77:- Cube-panel cross-node `ppermute` schedule **anti-scales on TCP** (107→206 ms/step
./docs/performance/scream_parity_scope.md:207:- AD: nonuniform-cotangent VJP through the ppermute halo matches the
./docs/performance/scream_parity_scope.md:215:proven ppermute collective.
./docs/performance/scream_parity_scope.md:228:  over-ranks timing, ppermute proven in the timed HLO, degrade-guarded.
./docs/performance/scream_parity_scope.md:230:  speed-up/efficiency curve and gates: baseline-present, ppermute-on-every-N>1,
./tests/unit/test_sharded_dynamics.py:549:    reported in #852, wrongly attributed to the ppermute halo — the halo is
./scripts/validate/validate_latlon_ocean_spmd_multiprocess.py:219:    # process boundary via ppermute/psum (the "spmd" backend armed per-call inside
./docs/scaling/external_scaling_transfer_assessment.md:9:The large transferable value is **TPU + future-hardware AGILITY**, not current-silicon speed. Every external system that reaches TPU does it the same way legoESM's `shard_map` path already does: lower the whole program to XLA and express collectives as `shard_map` + `ppermute`/`psum`/`all_to_all`. Oceananigans-via-Reactant (JAMES 2026, doi 10.1029/2025MS005615) and jaxDecomp's pure-JAX backend are the two proofs. **legoESM already sits on that substrate** — but carries two comm stacks: `mpi4jax` (multi-controller, **TPU dead-end**, pinned jax<0.10) and `shard_map`+`ppermute`/`psum` (single-controller, **the only TPU-capable path**; already shipped for cube, lat-band ocean, tripole north-fold). The transfer is convergence onto the XLA-collective stack + making it reachable beyond intra-node — not new numerics.
./docs/scaling/external_scaling_transfer_assessment.md:19:| JAX-Fluids 2.0 | structured block, `pmap`+`ppermute` | multi-GPU XLA | **first-class (same code)** | **halo = `lax.ppermute` → differentiability FREE**; face→edge→vertex corner fill; one `jax.distributed.initialize()` = CPU/GPU/TPU-pod, no MPI |
./docs/scaling/external_scaling_transfer_assessment.md:43:- **SFC (Hilbert) cell ordering** · **batched multi-field halo** (shipped: ocean latlon strong np8 0.35→0.55; MPAS union-neighbor default) · **single AD-safe primitive doctrine** (shipped + test-gated: `get_sendrecv_vjp`, `ppermute`/`psum`/`allreduce(SUM)`, `test_mpi_differentiability.py`).
./docs/scaling/atm_latlon_spmd_scaling.md:28:`ppermute` halo (over the node's PCIe interconnect) dominates the gain. This
./docs/scaling/atm_latlon_spmd_scaling.md:36:> ppermute, no mpi4jax). Derecho/Levante job lanes exist; production-scale
./docs/scaling/atm_latlon_spmd_scaling.md:48:the (small) `ppermute` halo overhead. Useful only to confirm correctness at
./scripts/validate/validate_tiled_fv3_sw_multinode.py:5:``jax.distributed`` across N nodes — so the in-stage halo ``lax.ppermute``
./scripts/validate/validate_tiled_fv3_sw_multinode.py:52:    # ppermutes -> cross-process collective-permute).  A single-process run with
./tests/unit/test_mc_nccl_probe.py:4:devices: every stage (device visibility, psum, ppermute ring, latency loop)
./tests/unit/test_mc_nccl_probe.py:33:    assert "stage4 ppermute ring OK=True" in out
./CLAUDE.md:294:  "ppermute round count" — the real exchange is halo-depth-aware, so the
./CLAUDE.md:301:  bottleneck.** FAILURE: assumed METIS would cut MPAS ppermute rounds; METIS
./CLAUDE.md:696:- **NO unexplained domain jargon.** Terms like *ppermute, max_degree,
./packages/core/legoesm/grids/halo_latlon.py:145:    """Route a lat-lon halo through the band SPMD ppermute body if the ``spmd``
./packages/core/legoesm/grids/halo_latlon.py:153:    ppermute + periodic lon ring ppermute + the exact serial 180-deg pole
./packages/core/legoesm/grids/halo_latlon.py:633:    # must read the NEIGHBOUR band's edge row (ppermute), not a constant wall;
./packages/core/legoesm/grids/halo_latlon.py:770:    # ppermute pair per direction per dtype group instead of one per
./packages/core/legoesm/grids/halo_latlon.py:786:    # ppermutes (byte-identical, just more collectives).
./packages/core/legoesm/grids/operators_latlon_cgrid.py:283:      ring ``ppermute`` over the ``"lon"`` mesh axis
./packages/core/legoesm/grids/operators_latlon_cgrid.py:293:    sendrecv VJP; ``ppermute`` is self-transposing).  ``f`` may be 2-D
./packages/core/legoesm/grids/operators_latlon_cgrid.py:493:        # sendrecv / SPMD lat-band ppermute).  A pole-touching end gets a
./packages/core/legoesm/grids/halo.py:501:    routes to the matching shard_map ppermute body (cube: face/tile axes;
./packages/core/legoesm/grids/halo.py:521:    ``jnp.pad`` — re-dispatching them to MPI sendrecv / SPMD ppermute would
./scripts/validate/validate_driver_cs_spmd_parity.py:17:PARITY CONTRACT (matches the bench --cs-spmd receipts): the ppermute SPMD
./scripts/run/run_omip_core2.py:4502:    # lat-band SPMD halo is pure-JAX ppermute/psum (no mpi4jax), so the bootstrap
./packages/core/legoesm/runtime/backend.py:116:    # Async overlap for all collectives (all-reduce, ppermute-based halo
./packages/coupler/legoesm/driver/model_driver.py:3985:            # ``pad_halo_4d`` lowers to ``ppermute`` / ``all_gather``
./packages/coupler/legoesm/driver/model_driver.py:3995:            # because ppermute / all_gather kernels assume divisors of
./packages/coupler/legoesm/driver/model_driver.py:4040:        * face-only sharding (``tiling == (1, 1)``) — the ppermute
./packages/coupler/legoesm/driver/model_driver.py:4054:            # scaling regime).  The tiled ppermute EXCHANGE layer is
./packages/coupler/legoesm/driver/model_driver.py:4077:                        "own in-stage halos (the ppermute SPMD backend "
./packages/coupler/legoesm/driver/model_driver.py:4094:                "whose in-stage ppermute pads are mesh-bound and do not "
./packages/coupler/legoesm/driver/model_driver.py:4175:        # ppermute/all_gather auto-selection is RETIRED (the all_gather
./packages/coupler/legoesm/driver/model_driver.py:4177:        # ppermute is the only auto-selectable exchange, all_gather is
./packages/coupler/legoesm/driver/sharded_operator_split_step.py:19:  ppermute). All fire correctly inside the armed ``shard_map``.
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/primitive_eq_latlon_cgrid.py:1080:                # neighbour's first FILTERED column (an MPI/ppermute ring
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:5:device mesh so the dycore runs multi-GPU/TPU with pure ``ppermute``/``psum``
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:547:        # row via ppermute), run the un-jitted band step, convert v back.
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1400:# line (ppermute at cuts, the serial 180-deg fold at the pole tiles), lon
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1401:# becomes a periodic ring (cyclic ppermute — the wrap IS the roll
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1423:# advantage until the partner-ppermute fold lands; the 2-D lane is for the
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1515:    edge blocks of that depth in one ppermute hop).
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1634:    (lat ppermute) AND ``w+1`` u-faces (lon ring ppermute), run the un-jitted
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1655:        # lat ppermute) and w+1 u-faces (periodic seam column via the lon
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1705:            "lat-pencil transpose; the SPMD ppermute equivalent is a "
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1718:    the lat ppermute, u via the periodic lon ring), runs the UN-jitted
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/sharded_atm_latlon_step.py:1722:    ppermute + lon ring + the EXACT serial 180-deg pole fold via a lon-ring
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:194:        # ppermute at cuts; the constant ghost at a pole-touching end never
./packages/atmosphere/legoesm/atmosphere/dynamics/gcm/shallow_water_latlon_cgrid.py:259:        # lat-band ppermute; u-face fields with the n_lon+1 wrap column are

exec
/usr/bin/bash -lc "nl -ba scripts/bench/analyze_nsys_rank_skew.py | sed -n '1,420p'
nl -ba tests/bench/test_analyze_nsys_rank_skew.py | sed -n '1,460p'
nl -ba scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh | sed -n '1,280p'
nl -ba scripts/cluster/scaling_levante/mpas_nsys_skew_capture.sbatch | sed -n '1,360p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 172ms:
     1	#!/usr/bin/env python
     2	"""Payload or wait? Compare ONE halo collective from BOTH of its partners.
     3	
     4	THE QUESTION.  The MPAS GPU lane spends 5.94 of 9.72 ms per step inside
     5	``ncclDevKernel_SendRecv`` (s9/np64), median 216 us against a ~30 us wire
     6	latency.  Two mechanisms explain that and they imply opposite engineering:
     7	cutting the ROUND COUNT pays well if the time is idle waiting, and pays only
     8	the per-round latency if the time is real transfer.
     9	
    10	WHY A SINGLE-RANK TRACE CANNOT ANSWER IT.  Grouped by schedule slot, those
    11	durations reproduce to CV <= 1.5 % across steps.  That refutes RANDOM jitter
    12	-- and nothing else.  A consistently late peer is exactly as reproducible.
    13	Wait is an inter-rank quantity; one timeline cannot see it.
    14	
    15	THE DISCRIMINATOR.  An NCCL SendRecv kernel starts when ITS rank arrives at
    16	the collective and ends when the transfer completes.  So for ONE collective
    17	seen from BOTH partners:
    18	
    19	  * both durations ~equal  -> the time is transfer.  PAYLOAD-bound.
    20	  * one long, one short    -> the long one arrived early and idled.  WAIT-
    21	                              bound, and (long - short) is the skew.
    22	
    23	HOW COLLECTIVES ARE MATCHED, and why not by overlap.  The obvious method --
    24	pair kernels whose [start, end] intervals overlap -- does not work here and
    25	its own control proves it: all 64 ranks issue their round-r collective at
    26	about the same time, so rank 0's kernel overlaps a NON-partner's just as
    27	readily as its true partner's.  Overlap is necessary, never sufficient.
    28	
    29	This uses the SCHEDULE as ground truth instead, which makes the pairing
    30	exact.  ``spmd_schedule_cost(..., round_profile_for_device=d)`` gives the
    31	rounds device ``d`` participates in, IN THE ORDER IT ISSUES THEM.  If A's
    32	profile shows it exchanges with B at global round r, and B's profile shows
    33	the mirror-image entry at the same r, then A's k-th halo fill issues that
    34	collective at position ``k*len(A_profile) + index_of_r_in_A`` and B's at
    35	``k*len(B_profile) + index_of_r_in_B``.  Same collective, by construction.
    36	
    37	Three things are then checked rather than assumed:
    38	  * SYMMETRY -- B's profile must name A as its partner at the same round.
    39	    A one-sided entry means the schedule was misread.
    40	  * FILL COUNT -- both ranks must have executed the same number of halo
    41	    fills (``n_calls / len(profile)``).  Ranks step in lockstep, so a
    42	    mismatch means the traces are not of the same run or one is truncated.
    43	  * DIVISIBILITY -- each rank's call count must be a whole number of fills.
    44	
    45	NON-PARTNER CONTROL.  Passing a pair that does not exchange (e.g. rank 0 and
    46	a same-node rank that is not its neighbour) finds no shared round and is
    47	reported as such.  That is the correct outcome and it is what shows the
    48	matching is schedule-driven rather than coincidence -- an overlap-based
    49	method would happily have "matched" that pair.
    50	
    51	SAME NODE ONLY.  Timestamps come from independent nsys processes and share a
    52	timebase only within one machine (``CLOCK_MONOTONIC`` is per-machine and
    53	undisciplined across nodes; inter-node drift dwarfs a 216 us signal).  The
    54	capture co-locates the profiled ranks and records the rank->node map; this
    55	script requires that map and REFUSES a cross-node pair.
    56	
    57	No verdict string is printed.  The interpretation belongs in the analysis
    58	once the controls pass, not baked into the tool where it gets echoed back as
    59	evidence.
    60	"""
    61	from __future__ import annotations
    62	
    63	import argparse
    64	import json
    65	import os
    66	import sqlite3
    67	import statistics as st
    68	import sys
    69	from pathlib import Path
    70	
    71	sys.path.insert(0, str(Path(__file__).resolve().parent))
    72	
    73	from metadata import scaling_metadata  # noqa: E402
    74	
    75	_SENDRECV_LIKE = "ncclDevKernel_SendRecv%"
    76	
    77	
    78	def load_kernels(sqlite_path: Path, kernel_like: str = _SENDRECV_LIKE):
    79	    """``(start_ns, end_ns)`` per collective, in issue order.
    80	
    81	    Raises on an EMPTY result: a name that matches nothing would otherwise
    82	    make every downstream count zero and read as "no skew".
    83	    """
    84	    if not sqlite_path.is_file():
    85	        raise SystemExit(f"missing sqlite: {sqlite_path}")
    86	    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    87	    try:
    88	        rows = con.execute(
    89	            "SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    90	            "JOIN StringIds s ON k.demangledName = s.id "
    91	            "WHERE s.value LIKE ? ORDER BY k.start", (kernel_like,)).fetchall()
    92	    finally:
    93	        con.close()
    94	    if not rows:
    95	        raise SystemExit(
    96	            f"{sqlite_path.name}: no kernel matches {kernel_like!r}. The NCCL "
    97	            f"kernel may be named differently in this build — check with "
    98	            f"SELECT DISTINCT value FROM StringIds WHERE value LIKE '%nccl%'.")
    99	    out = []
   100	    for start, end in rows:
   101	        if start is None or end is None or end < start:
   102	            raise SystemExit(
   103	                f"{sqlite_path.name}: bad kernel row start={start} end={end}")
   104	        out.append((int(start), int(end)))
   105	    return out
   106	
   107	
   108	def read_rank_nodes(path: Path):
   109	    """Parse the capture's ``_rank_nodes.tsv`` into ``{rank: node}``."""
   110	    if not path.is_file():
   111	        raise SystemExit(
   112	            f"missing {path}. It is written by _nsys_rank_wrapper.sh and is "
   113	            f"what proves the profiled ranks shared a node; without it a "
   114	            f"cross-node pair cannot be refused and its timestamps would be "
   115	            f"incomparable.")
   116	    nodes = {}
   117	    for line in path.read_text().splitlines():
   118	        fields = dict(f.split("=", 1) for f in line.strip().split("\t") if "=" in f)
   119	        if "rank" in fields and "node" in fields:
   120	            nodes[int(fields["rank"])] = fields["node"]
   121	    if not nodes:
   122	        raise SystemExit(f"{path}: no rank=/node= records parsed")
   123	    return nodes
   124	
   125	
   126	def shared_round(profile_a, profile_b, rank_a: int, rank_b: int):
   127	    """Global round where A and B exchange, plus each one's issue index.
   128	
   129	    Returns ``(round, index_in_a, index_in_b)`` or ``None`` when the two do
   130	    not exchange at all (the non-partner control).  Enforces SYMMETRY: B must
   131	    name A at the same round, otherwise the schedule has been misread and any
   132	    pairing built on it would be wrong.
   133	    """
   134	    a_hits = [(i, e) for i, e in enumerate(profile_a) if e["partner"] == rank_b]
   135	    if not a_hits:
   136	        return None
   137	    if len(a_hits) > 1:
   138	        raise SystemExit(
   139	            f"rank {rank_a} exchanges with rank {rank_b} in {len(a_hits)} "
   140	            f"rounds; this pairing is not unique and the script would have to "
   141	            f"guess which collective is which.")
   142	    idx_a, entry = a_hits[0]
   143	    b_hits = [i for i, e in enumerate(profile_b)
   144	              if e["partner"] == rank_a and e["round"] == entry["round"]]
   145	    if not b_hits:
   146	        raise SystemExit(
   147	            f"asymmetric schedule: rank {rank_a} names {rank_b} at round "
   148	            f"{entry['round']}, but {rank_b}'s profile has no mirror entry. "
   149	            f"The schedule has been misread; refusing to pair.")
   150	    return entry["round"], idx_a, b_hits[0]
   151	
   152	
   153	def compare(kernels_a, kernels_b, len_a, len_b, idx_a, idx_b, drop_fills):
   154	    """One record per halo fill, comparing the SAME collective on both ranks."""
   155	    if len(kernels_a) % len_a or len(kernels_b) % len_b:
   156	        raise SystemExit(
   157	            f"call counts are not whole fills: rank A has {len(kernels_a)} "
   158	            f"calls over {len_a} rounds/fill, rank B {len(kernels_b)} over "
   159	            f"{len_b}. The traces and the schedule disagree about the run.")
   160	    fills_a, fills_b = len(kernels_a) // len_a, len(kernels_b) // len_b
   161	    if fills_a != fills_b:
   162	        raise SystemExit(
   163	            f"rank A executed {fills_a} halo fills, rank B {fills_b}. Ranks "
   164	            f"step in lockstep, so these are not the same run (or one trace "
   165	            f"is truncated).")
   166	    recs = []
   167	    for k in range(drop_fills, fills_a):
   168	        sa, ea = kernels_a[k * len_a + idx_a]
   169	        sb, eb = kernels_b[k * len_b + idx_b]
   170	        dur_a, dur_b = ea - sa, eb - sb
   171	        recs.append({
   172	            "fill": k,
   173	            "a_dur_us": dur_a / 1e3,
   174	            "b_dur_us": dur_b / 1e3,
   175	            # >0: A started later, i.e. B was waiting for A.
   176	            "start_delta_us": (sa - sb) / 1e3,
   177	            # The idle time the EARLIER arriver spent waiting, which is the
   178	            # quantity a round-count cut would actually remove.
   179	            "wait_estimate_us": abs(dur_a - dur_b) / 1e3,
   180	            "ratio_long_short": (max(dur_a, dur_b) / min(dur_a, dur_b)
   181	                                 if min(dur_a, dur_b) > 0 else None),
   182	        })
   183	    return recs
   184	
   185	
   186	def main() -> int:
   187	    p = argparse.ArgumentParser(
   188	        description=__doc__,
   189	        formatter_class=argparse.RawDescriptionHelpFormatter)
   190	    p.add_argument("--sqlite-dir", required=True, type=Path,
   191	                   help="Capture dir holding rank_N.sqlite + _rank_nodes.tsv")
   192	    p.add_argument("--pairs", required=True,
   193	                   help="Comma-separated rank pairs, e.g. 0:1,0:2,0:3 "
   194	                        "(include a non-partner as the control).")
   195	    p.add_argument("--subdivision", type=int, required=True)
   196	    p.add_argument("--n-devices", type=int, required=True)
   197	    p.add_argument("--partition-method", default="sfc")
   198	    p.add_argument("--lloyd", type=int, default=0)
   199	    p.add_argument("--reorder-target", type=int, default=None,
   200	                   help="The run's --reorder-for, when it differs from the "
   201	                        "device count. A different ownership is a different "
   202	                        "schedule, so getting this wrong misaligns everything.")
   203	    p.add_argument("--drop-fills", type=int, default=3,
   204	                   help="Leading halo fills to discard as warmup/compile.")
   205	    p.add_argument("--out", default="results/a1/nsys_rank_skew.json")
   206	    args = p.parse_args()
   207	
   208	    pairs = []
   209	    for item in (s.strip() for s in args.pairs.split(",")):
   210	        if not item:
   211	            continue
   212	        a, _, b = item.partition(":")
   213	        if not b:
   214	            raise SystemExit(f"bad pair {item!r}; expected 'A:B'")
   215	        pairs.append((int(a), int(b)))
   216	    if not pairs:
   217	        raise SystemExit("--pairs selected nothing")
   218	
   219	    nodes = read_rank_nodes(args.sqlite_dir / "_rank_nodes.tsv")
   220	    for a, b in pairs:
   221	        for r in (a, b):
   222	            if r not in nodes:
   223	                raise SystemExit(f"rank {r} absent from the rank->node map")
   224	        if nodes[a] != nodes[b]:
   225	            raise SystemExit(
   226	                f"pair {a}:{b} spans nodes ({nodes[a]} vs {nodes[b]}). "
   227	                f"Timestamps from different machines share no timebase, so "
   228	                f"any skew computed from them would be drift, not physics.")
   229	
   230	    from legoesm.grids.voronoi import create_voronoi_mesh
   231	    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
   232	
   233	    print(f"[mesh] L{args.subdivision} lloyd={args.lloyd} …", flush=True)
   234	    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
   235	                               lloyd_iterations=args.lloyd)
   236	    profiles, kernels = {}, {}
   237	    for rank in sorted({r for pr in pairs for r in pr}):
   238	        cost = spmd_schedule_cost(mesh, args.n_devices,
   239	                                  method=args.partition_method,
   240	                                  reorder_target=args.reorder_target,
   241	                                  round_profile_for_device=rank)
   242	        profiles[rank] = cost["round_profile"]
   243	        kernels[rank] = load_kernels(args.sqlite_dir / f"rank_{rank}.sqlite")
   244	        print(f"[rank {rank}] node={nodes[rank]} "
   245	              f"{len(kernels[rank])} collectives, "
   246	              f"{len(profiles[rank])} rounds/fill", flush=True)
   247	
   248	    out_pairs = {}
   249	    for a, b in pairs:
   250	        key = f"{a}:{b}"
   251	        shared = shared_round(profiles[a], profiles[b], a, b)
   252	        if shared is None:
   253	            print(f"\n[{key}] NOT PARTNERS — no shared round. This is the "
   254	                  f"control: a schedule-driven pairing finds nothing here, "
   255	                  f"where an overlap-based one would have matched.", flush=True)
   256	            out_pairs[key] = {"partners": False}
   257	            continue
   258	        rnd, idx_a, idx_b = shared
   259	        recs = compare(kernels[a], kernels[b], len(profiles[a]),
   260	                       len(profiles[b]), idx_a, idx_b, args.drop_fills)
   261	        if not recs:
   262	            raise SystemExit(
   263	                f"{key}: no fills left after dropping {args.drop_fills}")
   264	        med_a = st.median(r["a_dur_us"] for r in recs)
   265	        med_b = st.median(r["b_dur_us"] for r in recs)
   266	        med_wait = st.median(r["wait_estimate_us"] for r in recs)
   267	        med_delta = st.median(r["start_delta_us"] for r in recs)
   268	        payload = profiles[a][idx_a]
   269	        out_pairs[key] = {
   270	            "partners": True, "round": rnd, "n_fills": len(recs),
   271	            "halo_cells": payload["halo_cells"],
   272	            "halo_edges": payload["halo_edges"],
   273	            "median_a_us": round(med_a, 2), "median_b_us": round(med_b, 2),
   274	            "median_wait_estimate_us": round(med_wait, 2),
   275	            "median_start_delta_us": round(med_delta, 2),
   276	            "wait_fraction_of_longer": round(
   277	                med_wait / max(med_a, med_b), 4) if max(med_a, med_b) else None,
   278	            "fills": recs,
   279	        }
   280	        print(f"\n[{key}] round {rnd}: cells={payload['halo_cells']} "
   281	              f"edges={payload['halo_edges']}, {len(recs)} fills")
   282	        print(f"   rank {a} median {med_a:8.1f} us")
   283	        print(f"   rank {b} median {med_b:8.1f} us")
   284	        print(f"   |difference| {med_wait:8.1f} us  "
   285	              f"= {100 * med_wait / max(med_a, med_b):.1f}% of the longer")
   286	        print(f"   start delta  {med_delta:+8.1f} us "
   287	              f"({'A later' if med_delta > 0 else 'B later'})")
   288	
   289	    payload_out = {
   290	        "pairs": out_pairs,
   291	        "rank_nodes": {str(k): v for k, v in nodes.items()},
   292	        "metadata": scaling_metadata(
   293	            grid="voronoi", component="atmosphere",
   294	            resolution=f"L{args.subdivision}", n_levels=0, precision="n/a",
   295	            decomposition="cell_partition", solver_variant="n/a",
   296	            scaling_kind="halo-skew-attribution", transport="nccl",
   297	            extra={"n_devices": args.n_devices,
   298	                   "partition_method": args.partition_method,
   299	                   "lloyd_iterations": args.lloyd,
   300	                   "reorder_target": args.reorder_target,
   301	                   "drop_fills": args.drop_fills,
   302	                   "sqlite_dir": str(args.sqlite_dir)}),
   303	    }
   304	    outdir = os.path.dirname(args.out)
   305	    if outdir:
   306	        os.makedirs(outdir, exist_ok=True)
   307	    with open(args.out, "w") as f:
   308	        json.dump(payload_out, f, indent=2)
   309	    print(f"\nJSON: {args.out}")
   310	    return 0
   311	
   312	
   313	if __name__ == "__main__":
   314	    raise SystemExit(main())
     1	"""Direct tests for the multi-rank halo-skew analyzer.
     2	
     3	The load-bearing logic is the PAIRING: which collective on rank A is the same
     4	collective as which on rank B.  Get that wrong and every duration comparison
     5	is between unrelated kernels — which is exactly how the previous probe in this
     6	family produced a confident wrong number.  So these tests hammer the pairing
     7	and its refusals, not the arithmetic.
     8	"""
     9	from __future__ import annotations
    10	
    11	import importlib.util
    12	import sqlite3
    13	import sys
    14	from pathlib import Path
    15	
    16	import pytest
    17	
    18	_SCRIPT = (Path(__file__).resolve().parents[2]
    19	           / "scripts" / "bench" / "analyze_nsys_rank_skew.py")
    20	_spec = importlib.util.spec_from_file_location("nsys_rank_skew", _SCRIPT)
    21	mod = importlib.util.module_from_spec(_spec)
    22	_spec.loader.exec_module(mod)
    23	
    24	
    25	def _profile(*entries):
    26	    """entries: (round, partner) -> the round_profile shape."""
    27	    return [{"round": r, "partner": p, "halo_cells": 10, "halo_edges": 20}
    28	            for r, p in entries]
    29	
    30	
    31	# --- shared_round: the pairing and its refusals ---------------------------
    32	
    33	
    34	def test_shared_round_finds_the_mirrored_entry_and_both_indices():
    35	    # A issues rounds [0(->1), 3(->2)]; B=1 issues [0(->0), 5(->2)].
    36	    a = _profile((0, 1), (3, 2))
    37	    b = _profile((0, 0), (5, 2))
    38	    assert mod.shared_round(a, b, 0, 1) == (0, 0, 0)
    39	
    40	    # Same, but the shared round sits at DIFFERENT issue positions on each
    41	    # rank — the case a naive "k-th call on both" pairing gets wrong.
    42	    a2 = _profile((7, 4), (9, 1))
    43	    b2 = _profile((2, 8), (5, 6), (9, 0))
    44	    assert mod.shared_round(a2, b2, 0, 1) == (9, 1, 2)
    45	
    46	
    47	def test_shared_round_returns_none_for_non_partners():
    48	    """The control. An overlap-based matcher would pair these anyway."""
    49	    a = _profile((0, 1), (3, 2))
    50	    b = _profile((4, 5), (6, 7))
    51	    assert mod.shared_round(a, b, 0, 3) is None
    52	
    53	
    54	def test_shared_round_refuses_a_one_sided_schedule():
    55	    """A names B, B does not name A: the schedule was misread."""
    56	    a = _profile((0, 1))
    57	    b = _profile((0, 9))          # B thinks it talks to 9 at round 0
    58	    with pytest.raises(SystemExit, match="asymmetric schedule"):
    59	        mod.shared_round(a, b, 0, 1)
    60	
    61	
    62	def test_shared_round_refuses_an_ambiguous_pairing():
    63	    """Two rounds with the same partner: which collective is which?"""
    64	    a = _profile((0, 1), (4, 1))
    65	    b = _profile((0, 0), (4, 0))
    66	    with pytest.raises(SystemExit, match="not unique"):
    67	        mod.shared_round(a, b, 0, 1)
    68	
    69	
    70	# --- compare: indexing and the consistency refusals -----------------------
    71	
    72	
    73	def _k(*start_duration_pairs):
    74	    """Kernels as (start_ns, duration_ns) -> the (start, end) rows."""
    75	    return [(start, start + dur) for start, dur in start_duration_pairs]
    76	
    77	
    78	def test_compare_indexes_the_same_collective_on_both_ranks():
    79	    """Ranks have DIFFERENT rounds-per-fill, so the stride differs.
    80	
    81	    A: 2 rounds/fill, shared round at index 1.
    82	    B: 3 rounds/fill, shared round at index 0.
    83	    Fill k therefore sits at A[2k+1] and B[3k+0]; pairing by raw call index
    84	    would compare A[1] with B[1], a different collective entirely.
    85	    """
    86	    # 3 fills. The collectives of interest get distinctive durations so a
    87	    # mis-stride shows up as the WRONG duration, not merely a wrong index.
    88	    a = _k((0, 1000), (10_000, 5000),       # fill 0: idx1 -> 5000 ns
    89	           (20_000, 1000), (30_000, 6000),   # fill 1: idx1 -> 6000 ns
    90	           (40_000, 1000), (50_000, 7000))   # fill 2: idx1 -> 7000 ns
    91	    b = _k((5_000, 2000), (6_000, 100), (7_000, 100),
    92	           (25_000, 2500), (26_000, 100), (27_000, 100),
    93	           (45_000, 3000), (46_000, 100), (47_000, 100))
    94	    recs = mod.compare(a, b, 2, 3, 1, 0, drop_fills=0)
    95	    assert [r["fill"] for r in recs] == [0, 1, 2]
    96	    assert [r["a_dur_us"] for r in recs] == [5.0, 6.0, 7.0]
    97	    assert [r["b_dur_us"] for r in recs] == [2.0, 2.5, 3.0]
    98	    # wait estimate = |difference| between the two ranks' time in the SAME
    99	    # collective.
   100	    assert [r["wait_estimate_us"] for r in recs] == [3.0, 3.5, 4.0]
   101	
   102	
   103	def test_compare_drops_warmup_fills():
   104	    a = _k((0, 1000), (10_000, 1000), (20_000, 1000), (30_000, 1000))
   105	    b = _k((0, 1000), (10_000, 1000), (20_000, 1000), (30_000, 1000))
   106	    recs = mod.compare(a, b, 1, 1, 0, 0, drop_fills=2)
   107	    assert [r["fill"] for r in recs] == [2, 3]
   108	
   109	
   110	def test_compare_refuses_partial_fills():
   111	    """A call count that is not a whole number of fills means the trace and
   112	    the schedule disagree; pairing would silently shear."""
   113	    a = _k((0, 1), (10, 1), (20, 1))       # 3 calls, 2 rounds/fill
   114	    b = _k((0, 1), (10, 1), (20, 1), (30, 1))
   115	    with pytest.raises(SystemExit, match="not whole fills"):
   116	        mod.compare(a, b, 2, 2, 0, 0, drop_fills=0)
   117	
   118	
   119	def test_compare_refuses_unequal_fill_counts():
   120	    """Ranks step in lockstep; different fill counts means different runs."""
   121	    a = _k((0, 1), (10, 1), (20, 1), (30, 1))   # 2 rounds/fill -> 2 fills
   122	    b = _k((0, 1), (10, 1), (20, 1))            # 1 round/fill  -> 3 fills
   123	    with pytest.raises(SystemExit, match="lockstep|halo fills"):
   124	        mod.compare(a, b, 2, 1, 0, 0, drop_fills=0)
   125	
   126	
   127	# --- trace loading --------------------------------------------------------
   128	
   129	
   130	def _make_sqlite(path, names_and_rows):
   131	    con = sqlite3.connect(str(path))
   132	    con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
   133	    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
   134	                "(start INTEGER, end INTEGER, demangledName INTEGER)")
   135	    for sid, (name, rows) in enumerate(names_and_rows):
   136	        con.execute("INSERT INTO StringIds VALUES (?,?)", (sid, name))
   137	        for s, e in rows:
   138	            con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?)",
   139	                        (s, e, sid))
   140	    con.commit()
   141	    con.close()
   142	
   143	
   144	def test_load_kernels_reads_only_the_collective_and_keeps_order(tmp_path):
   145	    db = tmp_path / "rank_0.sqlite"
   146	    _make_sqlite(db, [
   147	        ("ncclDevKernel_SendRecv(x)", [(100, 300), (500, 900)]),
   148	        ("loop_multiply_fusion", [(0, 50)]),          # must be ignored
   149	        ("ncclDevKernel_AllGather_RING_LL(y)", [(0, 99999)]),  # ignored
   150	    ])
   151	    assert mod.load_kernels(db) == [(100, 300), (500, 900)]
   152	
   153	
   154	def test_load_kernels_raises_when_the_name_matches_nothing(tmp_path):
   155	    """A renamed NCCL kernel would otherwise yield zero rows everywhere and
   156	    read as 'no skew' rather than 'no data'."""
   157	    db = tmp_path / "rank_0.sqlite"
   158	    _make_sqlite(db, [("some_other_kernel", [(0, 10)])])
   159	    with pytest.raises(SystemExit, match="no kernel matches"):
   160	        mod.load_kernels(db)
   161	
   162	
   163	# --- the cross-node refusal ----------------------------------------------
   164	
   165	
   166	def test_rank_node_map_parsed_and_cross_node_pair_refused(tmp_path,
   167	                                                          monkeypatch):
   168	    (tmp_path / "_rank_nodes.tsv").write_text(
   169	        "rank=0\tnode=l50018\tpid=1\n"
   170	        "rank=1\tnode=l50018\tpid=2\n"
   171	        "rank=9\tnode=l50033\tpid=3\n")
   172	    nodes = mod.read_rank_nodes(tmp_path / "_rank_nodes.tsv")
   173	    assert nodes == {0: "l50018", 1: "l50018", 9: "l50033"}
   174	
   175	    # 0 and 9 are on different machines: their CLOCK_MONOTONIC values share
   176	    # no origin, so a "skew" computed from them would be clock drift.
   177	    monkeypatch.setattr(sys, "argv", [
   178	        "skew", "--sqlite-dir", str(tmp_path), "--pairs", "0:9",
   179	        "--subdivision", "2", "--n-devices", "8"])
   180	    with pytest.raises(SystemExit, match="spans nodes"):
   181	        mod.main()
   182	
   183	
   184	def test_missing_rank_node_map_is_refused(tmp_path, monkeypatch):
   185	    monkeypatch.setattr(sys, "argv", [
   186	        "skew", "--sqlite-dir", str(tmp_path), "--pairs", "0:1",
   187	        "--subdivision", "2", "--n-devices", "8"])
   188	    with pytest.raises(SystemExit, match="_rank_nodes.tsv"):
   189	        mod.main()
     1	#!/usr/bin/env bash
     2	# Per-rank srun wrapper: run the target under `nsys profile` for the ranks in
     3	# $PROFILE_RANKS, plain $PY_BIN for every other rank.
     4	#
     5	# Profiling all 64 ranks is not an option (~400 MB each), and profiling an
     6	# arbitrary subset answers nothing: skew is measured BETWEEN PARTNERS, so the
     7	# selected ranks must actually exchange with each other AND sit on one node
     8	# (see the launcher for why cross-node timestamps are not comparable).
     9	set -uo pipefail
    10	
    11	: "${NSYS_BIN:?}"
    12	: "${NSYS_OUTDIR:?}"
    13	: "${PROFILE_RANKS:?}"
    14	: "${PY_BIN:?}"
    15	: "${SLURM_PROCID:?}"
    16	
    17	# Rank -> node map, so co-location of the profiled ranks is a RECORD and not
    18	# an assumption about how Slurm laid the job out.
    19	printf 'rank=%s\tnode=%s\tpid=%s\n' "$SLURM_PROCID" "$(hostname -s)" "$$" \
    20	    >> "${NSYS_OUTDIR}/_rank_nodes.tsv"
    21	
    22	in_list() {
    23	    local r="$1" x
    24	    local IFS=','
    25	    for x in $PROFILE_RANKS; do [ "$x" = "$r" ] && return 0; done
    26	    return 1
    27	}
    28	
    29	if in_list "$SLURM_PROCID"; then
    30	    out="${NSYS_OUTDIR}/rank_${SLURM_PROCID}.nsys-rep"
    31	    # Flags VERIFIED against nsys 2023.2.3 on Levante:
    32	    #   --trace: valid tokens are cuda,nvtx,cublas,cusparse,mpi,oshmem,ucx,
    33	    #            osrt,cudnn,opengl,... There is NO 'nccl' token in this
    34	    #            version, so NCCL kernels are captured as ordinary CUDA
    35	    #            kernels (which is all the analysis needs -- it matches
    36	    #            ncclDevKernel_SendRecv by name).  Asking for nccl here would
    37	    #            abort the run.
    38	    #   --export=sqlite: exists as a profile sub-option in this version.
    39	    #   osrt is deliberately OMITTED: it adds thread-level overhead, and
    40	    #            perturbing the very timing under test is the one thing this
    41	    #            capture cannot afford.
    42	    # NO --delay/--duration: this run is ~7.3 s of compile plus 12 steps of
    43	    # ~10 ms, so any delay long enough to skip compile also skips the entire
    44	    # steady state.  Capture everything; the analysis drops warmup cycles.
    45	    # NO `--` separator before the application: nsys 2023.2.3 parses a bare
    46	    # `--` as an ambiguous long-option abbreviation and aborts with "option
    47	    # is ambiguous and matches ..." (verified — it killed job 26771842's four
    48	    # profiled ranks). The application simply follows the flags.
    49	    # `env -u QUADD_INJECTION_PROXY` is REQUIRED, not hygiene. nsys sets that
    50	    # variable for its injection library, and JAX treats any *_PROXY env var
    51	    # as distributed-coordinator proxy configuration -- it prints "JAX
    52	    # detected proxy variable(s) ... may cause a hang of
    53	    # distributed.initialize" and then does exactly that. Only the PROFILED
    54	    # ranks get the variable, so they diverge from the other 60 and the whole
    55	    # job deadlocks in init (observed: job 26771961 hung with precisely four
    56	    # such warnings, one per profiled rank, and produced nothing).
    57	    # Unsetting it here is safe: nsys's injection is already active via
    58	    # LD_PRELOAD by the time this exec runs, so the variable has done its job.
    59	    exec "$NSYS_BIN" profile \
    60	        --trace=cuda,nvtx \
    61	        --output="$out" \
    62	        --force-overwrite=true \
    63	        --export=sqlite \
    64	        env -u QUADD_INJECTION_PROXY "$PY_BIN" "$@"
    65	else
    66	    exec "$PY_BIN" "$@"
    67	fi
     1	#!/bin/bash -l
     2	#SBATCH --job-name=mpas_skew
     3	#SBATCH --account=bb1596_gpu
     4	#SBATCH --partition=gpu
     5	#SBATCH --constraint=a100_80
     6	#SBATCH --nodes=16
     7	#SBATCH --ntasks-per-node=4
     8	#SBATCH --gpus-per-node=4
     9	#SBATCH --cpus-per-task=8
    10	#SBATCH --mem=120G
    11	#SBATCH --time=01:00:00
    12	#SBATCH --output=mpas_nsys_skew.%j.log
    13	# ===========================================================================
    14	# MULTI-RANK capture: is the MPAS halo collective PAYLOAD or WAIT?
    15	#
    16	# THE QUESTION. s9/np64 spends 5.94 of 9.72 ms per step inside
    17	# ncclDevKernel_SendRecv, median 216 us against a ~30 us wire latency.  The
    18	# single-rank capture (job 26680051) CANNOT decide why: it showed the
    19	# durations are reproducible per schedule slot (most positions CV <= 1.5 %),
    20	# which refutes RANDOM jitter but not a CONSISTENTLY late peer.  Wait is an
    21	# inter-rank quantity; one rank's timeline cannot see it.
    22	#
    23	# THE DISCRIMINATOR.  An NCCL SendRecv kernel starts when ITS rank arrives and
    24	# ends when the transfer completes.  So for ONE collective seen from BOTH
    25	# partners:
    26	#   * both durations ~equal  -> the time is transfer. PAYLOAD-bound. Cutting
    27	#     round count then saves only ~30 us x rounds removed; the bytes still
    28	#     move.  That is the LOW end of a round-depth partitioner's value.
    29	#   * one long, one short    -> the long one arrived early and idled. WAIT-
    30	#     bound, and the fix is imbalance, not round count.
    31	# The gap between those two readings is the entire uncertainty in whether the
    32	# 7-14 day partitioner is worth building.
    33	#
    34	# WHY THESE RANKS.  Timestamps from different nsys processes share a timebase
    35	# only on ONE node (CLOCK_MONOTONIC is per-machine and undisciplined across
    36	# nodes; drift there is tens of ms against a 216 us signal).  So the profiled
    37	# ranks must be partners AND co-located.  Both hold here, and that is
    38	# measured, not assumed: scoring the traced configuration (s9, np64, sfc,
    39	# reorder_target=128 -- see below) gives rank 0 eight rounds with partners
    40	# 1, 2, 12, 13, 49, 50, 61, 63.  Ranks 1 and 2 are on rank 0's node under
    41	# block:block, and rounds 0 and 1 (partners 1 and 2) carry the LARGEST
    42	# payloads of the eight (4809/12217 and 4626/11865 cells/edges), so they are
    43	# also the most informative.  Rank 3 is captured as a same-node NON-partner
    44	# reference: it shares the node's PCIe/NVLink and CPU but exchanges nothing
    45	# with rank 0, which separates "this node is slow" from "this pair is skewed".
    46	#
    47	# WHY reorder-for 128.  The traced run's own launcher was a throwaway and is
    48	# gone, so this was recovered by measurement: with reorder_target=64 the
    49	# schedule gives rank 0 seven rounds, and no multiple of 7 divides the 288
    50	# collectives in that trace; with 128 it gives eight, and 8 x 3 fills = 24
    51	# divides 288 exactly (12 steps).  Matching it is what makes the new capture
    52	# comparable to the old one.
    53	#
    54	# NO --delay/--duration.  The run is ~7.3 s of compile plus 12 steps of
    55	# ~10 ms, so any delay long enough to skip the compile also skips the whole
    56	# steady state.  Capture everything; the analysis drops warmup cycles.
    57	#
    58	# COST: 16 nodes x 1 h wall cap, but the run itself is seconds -- the job is
    59	# dominated by launch and by writing four ~400 MB profiles.
    60	#
    61	# SUBMIT (from the repo root):
    62	#   sbatch scripts/cluster/scaling_levante/mpas_nsys_skew_capture.sbatch
    63	# ===========================================================================
    64	set -uo pipefail
    65	SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    66	export JAX_PLATFORMS=cuda,cpu
    67	export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
    68	export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
    69	export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
    70	source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
    71	cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
    72	
    73	# Absolute path, VERIFIED present (2023.2.3). No module load needed, and
    74	# pinning the version keeps the trace schema stable for the analysis.
    75	NSYS_BIN="${NSYS_BIN:-/sw/spack-levante/cuda-12.2.0-2ttufp/nsight-systems-2023.2.3/target-linux-x64/nsys}"
    76	[ -x "$NSYS_BIN" ] || { echo "nsys not executable: $NSYS_BIN"; exit 2; }
    77	
    78	# rank 0 + partners 1,2 + same-node non-partner reference 3.
    79	PROFILE_RANKS="${PROFILE_RANKS:-0,1,2,3}"
    80	OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_skew_j${SLURM_JOB_ID}}"
    81	mkdir -p "$OUTDIR"
    82	echo "outdir=$OUTDIR  nsys=$NSYS_BIN  ranks=$PROFILE_RANKS"
    83	"$NSYS_BIN" --version
    84	
    85	WRAPPER="${SUBMIT_DIR}/scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh"
    86	[ -x "$WRAPPER" ] || { echo "wrapper missing/not executable: $WRAPPER"; exit 2; }
    87	
    88	export NSYS_BIN NSYS_OUTDIR="$OUTDIR" PROFILE_RANKS PY_BIN="$PY"
    89	
    90	# block:block is what puts ranks 0-3 on the first node; the wrapper records
    91	# the rank->node map so that placement is verified from the receipt rather
    92	# than trusted.
    93	JAX_ENABLE_X64=0 srun --ntasks=64 --distribution=block:block \
    94	  --cpu-bind=cores --kill-on-bad-exit=1 \
    95	  "$WRAPPER" scripts/bench/bench_mpas_spmd_scaling.py \
    96	    --n-devices 64 --subdivision 9 --nlev 26 --steps 12 --warmup 3 \
    97	    --lloyd 0 --partition-method sfc --reorder-for 128 --multicontroller \
    98	    --out "$OUTDIR/s9np64.jsonl"
    99	RC=$?
   100	echo "[skew] srun exit=$RC"
   101	
   102	# An exit code is not evidence a profile exists. Require one sqlite per
   103	# requested rank, or fail: a job that reports COMPLETED with no usable
   104	# capture is worse than one that reports failure.
   105	missing=""
   106	IFS=',' read -ra WANT <<< "$PROFILE_RANKS"
   107	for r in "${WANT[@]}"; do
   108	  if [ ! -s "$OUTDIR/rank_${r}.sqlite" ]; then
   109	    # --export=sqlite should have produced it; retry explicitly before giving up.
   110	    if [ -s "$OUTDIR/rank_${r}.nsys-rep" ]; then
   111	      echo "[skew] rank $r: exporting sqlite explicitly"
   112	      "$NSYS_BIN" export --type=sqlite --force-overwrite=true \
   113	          --output "$OUTDIR/rank_${r}.sqlite" "$OUTDIR/rank_${r}.nsys-rep" \
   114	          || echo "[skew] rank $r: export FAILED"
   115	    fi
   116	  fi
   117	  [ -s "$OUTDIR/rank_${r}.sqlite" ] || missing="$missing $r"
   118	done
   119	
   120	echo "[skew] rank->node map:"; cat "$OUTDIR/_rank_nodes.tsv" 2>/dev/null | sort -t= -k2 | head -8
   121	
   122	if [ -n "$missing" ]; then
   123	  echo "[skew] MISSING sqlite for ranks:$missing"
   124	  echo "SKEW_CAPTURE_INCOMPLETE"
   125	  exit 4
   126	fi
   127	if [ "$RC" -ne 0 ]; then
   128	  echo "SKEW_CAPTURE_RUN_FAILED"
   129	  exit "$RC"
   130	fi
   131	echo "SKEW_CAPTURE_DONE"

exec
/usr/bin/bash -lc "rg -n -C 8 \"def _ppermute_halo_fill|def spmd_schedule_cost|round_profile|_build_ppermute_schedule|ppermute_schedule|collective_permute|lax\\.ppermute\" packages/core/legoesm -S" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 124ms:
packages/core/legoesm/parallel/halo_exchange.py-54-   both send and receive complete.
packages/core/legoesm/parallel/halo_exchange.py-55-
packages/core/legoesm/parallel/halo_exchange.py-56-Near-term improvement path:
packages/core/legoesm/parallel/halo_exchange.py-57-  - Batch edge packing into a single contiguous buffer per neighbor
packages/core/legoesm/parallel/halo_exchange.py-58-    rank and do one sendrecv per neighbor (reduces from 4 to ≤4
packages/core/legoesm/parallel/halo_exchange.py-59-    messages per rank, with larger messages for better bandwidth).
packages/core/legoesm/parallel/halo_exchange.py-60-  - When ``mpi4jax`` gains ``Isend``/``Irecv``, convert to non-blocking
packages/core/legoesm/parallel/halo_exchange.py-61-    with ``Waitall`` after all sends/receives are posted.
packages/core/legoesm/parallel/halo_exchange.py:62:  - For pure multi-GPU (no MPI), ``jax.lax.ppermute`` is the preferred
packages/core/legoesm/parallel/halo_exchange.py-63-    path and integrates with XLA's SPMD partitioner.
packages/core/legoesm/parallel/halo_exchange.py-64-"""
packages/core/legoesm/parallel/halo_exchange.py-65-
packages/core/legoesm/parallel/halo_exchange.py-66-from __future__ import annotations
packages/core/legoesm/parallel/halo_exchange.py-67-
packages/core/legoesm/parallel/halo_exchange.py-68-import jax
packages/core/legoesm/parallel/halo_exchange.py-69-import jax.numpy as jnp
packages/core/legoesm/parallel/halo_exchange.py-70-
--
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2444-def warmup_tiled_cube_comms(mesh, kt: int, *, force: bool = False) -> bool:
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2445-    """Deterministically prime EVERY NCCL communicator the closed-loop tiled
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2446-    cube step uses, BEFORE the first real step, so multi-process communicator
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2447-    init cannot deadlock (issue #921).
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2448-
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2449-    The closed-loop blocked step
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2450-    (:func:`make_tiled_fv3_hydrostatic_step_blocked_2d` with ``fix_mass=True``,
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2451-    and the operator-split twin) issues, inside ONE compiled executable, BOTH
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2452:    the halo collective-permutes (``jax.lax.ppermute`` over the
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2453-    ``(face, tile_i, tile_j)`` mesh axes — the tiled halo cliques) AND the
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2454-    mass-fixer GLOBAL reduction (``jax.lax.psum`` over the SAME axes —
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2455-    :func:`_tile_fix_ps_mass_delta` / :func:`_tile_fix_ps_mass_target`).  NCCL
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2456-    communicator init is itself a collective over the clique; under the XLA/GPU
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2457-    defaults (latency-hiding scheduler + async collectives +
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2458-    ``nccl_comm_splitting``) the two clique KINDS can be scheduled for init in a
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2459-    DIFFERENT relative order on different ranks — rank A blocks initialising the
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2460-    reduction clique while rank B blocks initialising a permute clique — a
--
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2522-    dummy = jax.device_put(jnp.zeros((6, kt, kt), dtype=jnp.float32), sh)
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2523-
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2524-    @partial(shard_map, mesh=mesh, in_specs=P(*AXES), out_specs=P(*AXES),
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2525-             check_vma=False)
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2526-    def _halo_warm(x):
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2527-        v = x.reshape((1,))
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2528-        acc = v
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2529-        for perm in perms:
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2530:            acc = acc + jax.lax.ppermute(v, AXES, perm)
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2531-        return acc.reshape((1, 1, 1))
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2532-
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2533-    @partial(shard_map, mesh=mesh, in_specs=P(*AXES), out_specs=P(*AXES),
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2534-             check_vma=False)
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2535-    def _reduce_warm(x):
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2536-        v = x.reshape((1,))
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2537-        return (v + jax.lax.psum(v, axis_name=AXES)).reshape((1, 1, 1))
packages/core/legoesm/parallel/tiled_production_cdgrid.py-2538-
--
packages/core/legoesm/parallel/tiled_d2a2c.py-1-"""Sub-face tiled FV3 d2a2c SPMD stage (P4 phase-1b).
packages/core/legoesm/parallel/tiled_d2a2c.py-2-
packages/core/legoesm/parallel/tiled_d2a2c.py-3-Approach C: the cheap D→A step + global fields (:func:`d2a2c_global_fields`)
packages/core/legoesm/parallel/tiled_d2a2c.py-4-run in the global (face-replicated) view; the A→C compute is sharded over a
packages/core/legoesm/parallel/tiled_d2a2c.py-5-``(6, kt, kt)`` device mesh — each device ``lax.dynamic_slice``s its tile's
packages/core/legoesm/parallel/tiled_d2a2c.py-6-blocks from the replicated fields and runs the single device-uniform
packages/core/legoesm/parallel/tiled_d2a2c.py-7-:func:`d2a2c_tile_unified` (flag/mask-driven, codex-clean), so there is NO
packages/core/legoesm/parallel/tiled_d2a2c.py-8-dynamic wind-halo exchange.  With ``apply_strips=True`` (default) the two
packages/core/legoesm/parallel/tiled_d2a2c.py:9:adjacent strips are applied IN-STAGE: four 1-cell ``lax.ppermute`` halos on
packages/core/legoesm/parallel/tiled_d2a2c.py-10-the same-face tile axes feed :func:`d2a2c_tile_strips`, making the
packages/core/legoesm/parallel/tiled_d2a2c.py-11-reassembled output equal ``d2a2c_vect`` with NO global post-pass.  With
packages/core/legoesm/parallel/tiled_d2a2c.py-12-``apply_strips=False`` the strips are deferred and the caller applies
packages/core/legoesm/parallel/tiled_d2a2c.py-13-:func:`d2a2c_adjacent_strips` on the reassembled global staggered fields
packages/core/legoesm/parallel/tiled_d2a2c.py-14-(correctness proven bit-exact vs ``d2a2c_vect`` in
packages/core/legoesm/parallel/tiled_d2a2c.py-15-``tests/parallel/test_tiled_d2a2c_ua_va.py``).  Design:
packages/core/legoesm/parallel/tiled_d2a2c.py-16-``docs/performance/scaling/d2a2c_spmd_stage_design.md``.
packages/core/legoesm/parallel/tiled_d2a2c.py-17-"""
--
packages/core/legoesm/parallel/tiled_d2a2c.py-90-            a, b, n, npt, ti == 0, ti == kt - 1, tj == 0, tj == kt - 1)
packages/core/legoesm/parallel/tiled_d2a2c.py-91-        if not apply_strips:
packages/core/legoesm/parallel/tiled_d2a2c.py-92-            return out
packages/core/legoesm/parallel/tiled_d2a2c.py-93-        ua_t, va_t, uc_t, vc_t, ut_t, vt_t = out
packages/core/legoesm/parallel/tiled_d2a2c.py-94-        # hi halo: every source sends its LOW edge (col/row 0) to t-1, so
packages/core/legoesm/parallel/tiled_d2a2c.py-95-        # receiver r holds tile r+1's cell b+nl / a+nl.  lo halo: sends
packages/core/legoesm/parallel/tiled_d2a2c.py-96-        # its HIGH edge (col/row nl-1) to t+1 -> receiver holds cell
packages/core/legoesm/parallel/tiled_d2a2c.py-97-        # b-1 / a-1.
packages/core/legoesm/parallel/tiled_d2a2c.py:98:        ut_hi = jax.lax.ppermute(ut_t[:, :, 0], "tile_j", perm_hi)
packages/core/legoesm/parallel/tiled_d2a2c.py:99:        ut_lo = jax.lax.ppermute(ut_t[:, :, nl - 1], "tile_j", perm_lo)
packages/core/legoesm/parallel/tiled_d2a2c.py:100:        vt_hi = jax.lax.ppermute(vt_t[:, 0, :], "tile_i", perm_hi)
packages/core/legoesm/parallel/tiled_d2a2c.py:101:        vt_lo = jax.lax.ppermute(vt_t[:, nl - 1, :], "tile_i", perm_lo)
packages/core/legoesm/parallel/tiled_d2a2c.py-102-        ut_t, vt_t = d2a2c_tile_strips(
packages/core/legoesm/parallel/tiled_d2a2c.py-103-            uc_t, vc_t, ut_t, vt_t, ut_lo, ut_hi, vt_lo, vt_hi,
packages/core/legoesm/parallel/tiled_d2a2c.py-104-            cu_blk, cv_blk, a, b, n, nl,
packages/core/legoesm/parallel/tiled_d2a2c.py-105-            ti == 0, ti == kt - 1, tj == 0, tj == kt - 1)
packages/core/legoesm/parallel/tiled_d2a2c.py-106-        return ua_t, va_t, uc_t, vc_t, ut_t, vt_t
packages/core/legoesm/parallel/tiled_d2a2c.py-107-
packages/core/legoesm/parallel/tiled_d2a2c.py-108-    def stage(u_d, v_d, fields):
packages/core/legoesm/parallel/tiled_d2a2c.py-109-        # Wide low-padded covariant blocks: the kernel's -1 cell in-bounds
--
packages/core/legoesm/parallel/cubesphere_exchange.py-4-under face-axis sharding with **explicit collective operations**
packages/core/legoesm/parallel/cubesphere_exchange.py-5-inside ``shard_map``.  Three collective kernels are provided:
packages/core/legoesm/parallel/cubesphere_exchange.py-6-
packages/core/legoesm/parallel/cubesphere_exchange.py-7-* **ppermute multiface** (DEFAULT for every face-sharded device count
packages/core/legoesm/parallel/cubesphere_exchange.py-8-  ``n_devices ∈ {1, 2, 3, 6}``, halo 1 and 2): each shard owns
packages/core/legoesm/parallel/cubesphere_exchange.py-9-  ``k = 6 / n_devices`` contiguous faces.  Face edges *within* a shard
packages/core/legoesm/parallel/cubesphere_exchange.py-10-  are filled shard-locally (same rotation + corner conventions as the
packages/core/legoesm/parallel/cubesphere_exchange.py-11-  serial path); edges *crossing* shard boundaries ride a static
packages/core/legoesm/parallel/cubesphere_exchange.py:12:  device-pair schedule of ``jax.lax.ppermute`` rounds, one
packages/core/legoesm/parallel/cubesphere_exchange.py-13-  concatenated strip buffer per (src, dst) device pair per round.
packages/core/legoesm/parallel/cubesphere_exchange.py-14-  No value is ever materialized with full face extent on any device.
packages/core/legoesm/parallel/cubesphere_exchange.py-15-
packages/core/legoesm/parallel/cubesphere_exchange.py-16-* **ppermute one-face** (halo=1, exactly 6 devices): the original
packages/core/legoesm/parallel/cubesphere_exchange.py-17-  validated 4-round perfect-matching kernel — kept as the lowest-risk
packages/core/legoesm/parallel/cubesphere_exchange.py-18-  path for the 1-face-per-device layout.
packages/core/legoesm/parallel/cubesphere_exchange.py-19-
packages/core/legoesm/parallel/cubesphere_exchange.py-20-* **all_gather** (EXPLICIT DIAGNOSTIC OPT-IN ONLY — ``force_allgather``
--
packages/core/legoesm/parallel/cubesphere_exchange.py-573-            padded = jnp.pad(my_face, ((1, 1), (1, 1), (0, 0)))
packages/core/legoesm/parallel/cubesphere_exchange.py-574-        halo_strips = [None, None, None, None]
packages/core/legoesm/parallel/cubesphere_exchange.py-575-        if with_offsets:
packages/core/legoesm/parallel/cubesphere_exchange.py-576-            from legoesm.grids.halo import interp_strip
packages/core/legoesm/parallel/cubesphere_exchange.py-577-
packages/core/legoesm/parallel/cubesphere_exchange.py-578-        for r in range(4):
packages/core/legoesm/parallel/cubesphere_exchange.py-579-            send_edge = ppermute_send_j[r, my_idx]   # traced int
packages/core/legoesm/parallel/cubesphere_exchange.py-580-            to_send = my_strips[send_edge]           # (n,) or (n, C)
packages/core/legoesm/parallel/cubesphere_exchange.py:581:            received = jax.lax.ppermute(
packages/core/legoesm/parallel/cubesphere_exchange.py-582-                to_send, "face", _PPERMUTE_PERMS[r],
packages/core/legoesm/parallel/cubesphere_exchange.py-583-            )
packages/core/legoesm/parallel/cubesphere_exchange.py-584-            recv_edge = ppermute_recv_j[r, my_idx]
packages/core/legoesm/parallel/cubesphere_exchange.py-585-            rev = ppermute_rev_j[r, my_idx]
packages/core/legoesm/parallel/cubesphere_exchange.py-586-            received = jnp.where(rev, received[::-1], received)
packages/core/legoesm/parallel/cubesphere_exchange.py-587-            if with_offsets:
packages/core/legoesm/parallel/cubesphere_exchange.py-588-                # offsets[my_idx, recv_edge] selects the right per-edge
packages/core/legoesm/parallel/cubesphere_exchange.py-589-                # offset for whichever halo slot this round fills.  The
--
packages/core/legoesm/parallel/cubesphere_exchange.py-1259-    ).astype(seg_padded.dtype)
packages/core/legoesm/parallel/cubesphere_exchange.py-1260-
packages/core/legoesm/parallel/cubesphere_exchange.py-1261-
packages/core/legoesm/parallel/cubesphere_exchange.py-1262-def _build_tiled_pad(mesh, ndim, halo=1, with_offsets=False):
packages/core/legoesm/parallel/cubesphere_exchange.py-1263-    """Shared tiled-pad builder — validate + static tables + the
packages/core/legoesm/parallel/cubesphere_exchange.py-1264-    body-side pad function.  Returns ``(_pad_body, in_sp, in_sp_data)``.
packages/core/legoesm/parallel/cubesphere_exchange.py-1265-
packages/core/legoesm/parallel/cubesphere_exchange.py-1266-    ``_pad_body(tile, offsets)`` pads a SINGLE device's local tile via
packages/core/legoesm/parallel/cubesphere_exchange.py:1267:    ``lax.ppermute`` over (face, tile_i, tile_j) — valid inside ANY
packages/core/legoesm/parallel/cubesphere_exchange.py-1268-    shard_map over those axes, so both the wrapped exchange and the
packages/core/legoesm/parallel/cubesphere_exchange.py-1269-    unwrapped tiled tendency stage share this ONE body (zero
packages/core/legoesm/parallel/cubesphere_exchange.py-1270-    duplication / parity drift).
packages/core/legoesm/parallel/cubesphere_exchange.py-1271-
packages/core/legoesm/parallel/cubesphere_exchange.py-1272-    Original tiled exchange semantics — halo 1/2.
packages/core/legoesm/parallel/cubesphere_exchange.py-1273-
packages/core/legoesm/parallel/cubesphere_exchange.py-1274-    One tile per device on mesh axes ("face", "tile_i", "tile_j").
packages/core/legoesm/parallel/cubesphere_exchange.py-1275-    Schedule: 4 strip ppermute rounds (every tile edge is remote) +
--
packages/core/legoesm/parallel/cubesphere_exchange.py-1339-            ], axis=0)
packages/core/legoesm/parallel/cubesphere_exchange.py-1340-
packages/core/legoesm/parallel/cubesphere_exchange.py-1341-        # Strip rounds: every (device, edge) slot is provably covered
packages/core/legoesm/parallel/cubesphere_exchange.py-1342-        # (table build asserts), so zeros init is dead weight only
packages/core/legoesm/parallel/cubesphere_exchange.py-1343-        # until overwritten.
packages/core/legoesm/parallel/cubesphere_exchange.py-1344-        halo_strips = jnp.zeros_like(strips)
packages/core/legoesm/parallel/cubesphere_exchange.py-1345-        for r in range(n_rounds):
packages/core/legoesm/parallel/cubesphere_exchange.py-1346-            send_buf = strips[send_edge_j[r, my_id]]
packages/core/legoesm/parallel/cubesphere_exchange.py:1347:            received = jax.lax.ppermute(send_buf, AXES, tables.perms[r])
packages/core/legoesm/parallel/cubesphere_exchange.py-1348-            tgt = recv_tgt_j[r, my_id][None]
packages/core/legoesm/parallel/cubesphere_exchange.py-1349-            halo_strips = halo_strips.at[tgt].set(
packages/core/legoesm/parallel/cubesphere_exchange.py-1350-                received[None], mode="drop")
packages/core/legoesm/parallel/cubesphere_exchange.py-1351-
packages/core/legoesm/parallel/cubesphere_exchange.py-1352-        # Receiver-side reversal (global strip orientation).  Strip
packages/core/legoesm/parallel/cubesphere_exchange.py-1353-        # axis: 1 for halo=1 payloads (4, n_loc, ...), 2 for halo=2
packages/core/legoesm/parallel/cubesphere_exchange.py-1354-        # (4, 2, n_loc, ...).
packages/core/legoesm/parallel/cubesphere_exchange.py-1355-        strip_axis = 1 if halo == 1 else 2
--
packages/core/legoesm/parallel/cubesphere_exchange.py-1365-            sx = strip_axis
packages/core/legoesm/parallel/cubesphere_exchange.py-1366-            we = halo_strips[0:2]
packages/core/legoesm/parallel/cubesphere_exchange.py-1367-            sn = halo_strips[2:4]
packages/core/legoesm/parallel/cubesphere_exchange.py-1368-
packages/core/legoesm/parallel/cubesphere_exchange.py-1369-            def _take(arr, sl):
packages/core/legoesm/parallel/cubesphere_exchange.py-1370-                idx = (slice(None),) * sx + (sl,)
packages/core/legoesm/parallel/cubesphere_exchange.py-1371-                return arr[idx]
packages/core/legoesm/parallel/cubesphere_exchange.py-1372-
packages/core/legoesm/parallel/cubesphere_exchange.py:1373:            lo_from_jbwd = jax.lax.ppermute(
packages/core/legoesm/parallel/cubesphere_exchange.py-1374-                _take(we, slice(-g, None)), AXES, j_fwd)
packages/core/legoesm/parallel/cubesphere_exchange.py:1375:            hi_from_jfwd = jax.lax.ppermute(
packages/core/legoesm/parallel/cubesphere_exchange.py-1376-                _take(we, slice(None, g)), AXES, j_bwd)
packages/core/legoesm/parallel/cubesphere_exchange.py:1377:            lo_from_ibwd = jax.lax.ppermute(
packages/core/legoesm/parallel/cubesphere_exchange.py-1378-                _take(sn, slice(-g, None)), AXES, i_fwd)
packages/core/legoesm/parallel/cubesphere_exchange.py:1379:            hi_from_ifwd = jax.lax.ppermute(
packages/core/legoesm/parallel/cubesphere_exchange.py-1380-                _take(sn, slice(None, g)), AXES, i_bwd)
packages/core/legoesm/parallel/cubesphere_exchange.py-1381-            we_pad = jnp.concatenate(
packages/core/legoesm/parallel/cubesphere_exchange.py-1382-                [lo_from_jbwd, we, hi_from_jfwd], axis=sx)
packages/core/legoesm/parallel/cubesphere_exchange.py-1383-            sn_pad = jnp.concatenate(
packages/core/legoesm/parallel/cubesphere_exchange.py-1384-                [lo_from_ibwd, sn, hi_from_ifwd], axis=sx)
packages/core/legoesm/parallel/cubesphere_exchange.py-1385-            guarded = jnp.concatenate([we_pad, sn_pad], axis=0)
packages/core/legoesm/parallel/cubesphere_exchange.py-1386-
packages/core/legoesm/parallel/cubesphere_exchange.py-1387-            offs_dev = jax.lax.dynamic_slice_in_dim(
--
packages/core/legoesm/parallel/cubesphere_exchange.py-1442-            # 2x2 interior corner blocks, rows/cols in face order.
packages/core/legoesm/parallel/cubesphere_exchange.py-1443-            diag_send = (tile[-2:, -2:], tile[-2:, :2],
packages/core/legoesm/parallel/cubesphere_exchange.py-1444-                         tile[:2, -2:], tile[:2, :2])
packages/core/legoesm/parallel/cubesphere_exchange.py-1445-
packages/core/legoesm/parallel/cubesphere_exchange.py-1446-            def _ends(e, lo):
packages/core/legoesm/parallel/cubesphere_exchange.py-1447-                s = final[e]              # (2, n_loc[, C]) = (depth, j)
packages/core/legoesm/parallel/cubesphere_exchange.py-1448-                return s[:, :h] if lo else s[:, -h:]
packages/core/legoesm/parallel/cubesphere_exchange.py-1449-        diag_recv = [
packages/core/legoesm/parallel/cubesphere_exchange.py:1450:            jax.lax.ppermute(diag_send[c], AXES, diag_perms[c])
packages/core/legoesm/parallel/cubesphere_exchange.py-1451-            for c in range(4)
packages/core/legoesm/parallel/cubesphere_exchange.py-1452-        ]
packages/core/legoesm/parallel/cubesphere_exchange.py:1453:        sl_jf = jax.lax.ppermute(
packages/core/legoesm/parallel/cubesphere_exchange.py-1454-            jnp.stack([_ends(0, False), _ends(1, False)]), AXES, j_fwd)
packages/core/legoesm/parallel/cubesphere_exchange.py:1455:        sl_jb = jax.lax.ppermute(
packages/core/legoesm/parallel/cubesphere_exchange.py-1456-            jnp.stack([_ends(0, True), _ends(1, True)]), AXES, j_bwd)
packages/core/legoesm/parallel/cubesphere_exchange.py:1457:        sl_if = jax.lax.ppermute(
packages/core/legoesm/parallel/cubesphere_exchange.py-1458-            jnp.stack([_ends(2, False), _ends(3, False)]), AXES, i_fwd)
packages/core/legoesm/parallel/cubesphere_exchange.py:1459:        sl_ib = jax.lax.ppermute(
packages/core/legoesm/parallel/cubesphere_exchange.py-1460-            jnp.stack([_ends(2, True), _ends(3, True)]), AXES, i_bwd)
packages/core/legoesm/parallel/cubesphere_exchange.py-1461-        # Per corner (lo,lo),(lo,hi),(hi,lo),(hi,hi): the W/E-row
packages/core/legoesm/parallel/cubesphere_exchange.py-1462-        # sliver candidate and the S/N-col sliver candidate.
packages/core/legoesm/parallel/cubesphere_exchange.py-1463-        sliv_we = (sl_jf[0], sl_jb[0], sl_jf[1], sl_jb[1])
packages/core/legoesm/parallel/cubesphere_exchange.py-1464-        sliv_sn = (sl_if[0], sl_if[1], sl_ib[0], sl_ib[1])
packages/core/legoesm/parallel/cubesphere_exchange.py-1465-        my_mode = corner_mode_j[my_id]
packages/core/legoesm/parallel/cubesphere_exchange.py-1466-
packages/core/legoesm/parallel/cubesphere_exchange.py-1467-        if halo == 1:
--
packages/core/legoesm/parallel/cubesphere_exchange.py-1543-
packages/core/legoesm/parallel/cubesphere_exchange.py-1544-def make_tiled_pad_body(mesh, ndim, halo=1, with_offsets=False):
packages/core/legoesm/parallel/cubesphere_exchange.py-1545-    """Unwrapped tiled pad body for use INSIDE an outer shard_map over
packages/core/legoesm/parallel/cubesphere_exchange.py-1546-    the same (face, tile_i, tile_j) axes — the tiled FV3 tendency
packages/core/legoesm/parallel/cubesphere_exchange.py-1547-    stage.
packages/core/legoesm/parallel/cubesphere_exchange.py-1548-
packages/core/legoesm/parallel/cubesphere_exchange.py-1549-    Takes one device's local ``(n_loc, n_loc[, C])`` tile plus optional
packages/core/legoesm/parallel/cubesphere_exchange.py-1550-    replicated ``offsets`` and returns the ``(n_loc+2h, ...)`` padded
packages/core/legoesm/parallel/cubesphere_exchange.py:1551:    block via ``lax.ppermute`` directly.  A nested ``shard_map`` (what
packages/core/legoesm/parallel/cubesphere_exchange.py-1552-    the operators' ``explicit_pad_halo`` SPMD branch does) is illegal
packages/core/legoesm/parallel/cubesphere_exchange.py-1553-    inside an outer ``shard_map``, so the stage routes its metric/state
packages/core/legoesm/parallel/cubesphere_exchange.py-1554-    pads through THIS body instead.  Bit-identical to the wrapped
packages/core/legoesm/parallel/cubesphere_exchange.py-1555-    :func:`_make_exchange_ppermute_tiled` body (same code).
packages/core/legoesm/parallel/cubesphere_exchange.py-1556-    """
packages/core/legoesm/parallel/cubesphere_exchange.py-1557-    _pad_body, _in_sp, _out_sp = _build_tiled_pad(
packages/core/legoesm/parallel/cubesphere_exchange.py-1558-        mesh, ndim, halo=halo, with_offsets=with_offsets)
packages/core/legoesm/parallel/cubesphere_exchange.py-1559-    return _pad_body
--
packages/core/legoesm/parallel/cubesphere_exchange.py-1701-        my_loc_lf = loc_lf_j[my_idx].reshape(-1)   # (4k,) traced
packages/core/legoesm/parallel/cubesphere_exchange.py-1702-        my_loc_le = loc_le_j[my_idx].reshape(-1)
packages/core/legoesm/parallel/cubesphere_exchange.py-1703-        halo_flat = my_strips[my_loc_lf, my_loc_le]  # (4k, [2,] n[, C])
packages/core/legoesm/parallel/cubesphere_exchange.py-1704-
packages/core/legoesm/parallel/cubesphere_exchange.py-1705-        # ---- cross-shard ppermute rounds ----
packages/core/legoesm/parallel/cubesphere_exchange.py-1706-        for r in range(n_rounds):
packages/core/legoesm/parallel/cubesphere_exchange.py-1707-            send_buf = my_strips[send_lf_j[r, my_idx],
packages/core/legoesm/parallel/cubesphere_exchange.py-1708-                                 send_le_j[r, my_idx]]  # (max_slots, ...)
packages/core/legoesm/parallel/cubesphere_exchange.py:1709:            received = jax.lax.ppermute(send_buf, "face", perms[r])
packages/core/legoesm/parallel/cubesphere_exchange.py-1710-            tgt = recv_tgt_j[r, my_idx]  # (max_slots,) — 4k sentinel drops
packages/core/legoesm/parallel/cubesphere_exchange.py-1711-            halo_flat = halo_flat.at[tgt].set(received, mode="drop")
packages/core/legoesm/parallel/cubesphere_exchange.py-1712-
packages/core/legoesm/parallel/cubesphere_exchange.py-1713-        halo_buf = halo_flat.reshape((k, 4) + halo_flat.shape[1:])
packages/core/legoesm/parallel/cubesphere_exchange.py-1714-
packages/core/legoesm/parallel/cubesphere_exchange.py-1715-        # ---- receiver-side reversal along the spatial strip axis ----
packages/core/legoesm/parallel/cubesphere_exchange.py-1716-        # (senders transmit unreversed source-edge strips; reversal is a
packages/core/legoesm/parallel/cubesphere_exchange.py-1717-        # property of the receiving (face, edge), exactly as the serial
--
packages/core/legoesm/parallel/latlon_spmd.py-2-
packages/core/legoesm/parallel/latlon_spmd.py-3-The ocean lat-lon C-grid (and atm lat-lon) shard cleanly by LATITUDE BAND: each
packages/core/legoesm/parallel/latlon_spmd.py-4-device owns a contiguous lat band and the FULL longitude circle (lon is periodic
packages/core/legoesm/parallel/latlon_spmd.py-5-and kept local — the audit's "every rank owns all longitudes").  The halo is
packages/core/legoesm/parallel/latlon_spmd.py-6-therefore 1-D over the ``"lat"`` mesh axis:
packages/core/legoesm/parallel/latlon_spmd.py-7-
packages/core/legoesm/parallel/latlon_spmd.py-8-  * longitude: periodic wrap — LOCAL ``jnp.pad(mode="wrap")`` (no comm), exactly
packages/core/legoesm/parallel/latlon_spmd.py-9-    like the serial :func:`legoesm.grids.halo_latlon.pad_halo_latlon_local`.
packages/core/legoesm/parallel/latlon_spmd.py:10:  * latitude interior band cuts: ``jax.lax.ppermute`` of the ``halo`` edge rows
packages/core/legoesm/parallel/latlon_spmd.py-11-    between adjacent bands (north neighbour's bottom rows / south neighbour's
packages/core/legoesm/parallel/latlon_spmd.py-12-    top rows).
packages/core/legoesm/parallel/latlon_spmd.py-13-  * poles: the end bands (axis_index 0 = south, N-1 = north) have no neighbour
packages/core/legoesm/parallel/latlon_spmd.py-14-    there, so they fold their OWN pole rows (mirror in lat + 180 deg in lon),
packages/core/legoesm/parallel/latlon_spmd.py-15-    selected by a ``jnp.where`` on the band index — bit-identical to the serial
packages/core/legoesm/parallel/latlon_spmd.py-16-    pole fold.
packages/core/legoesm/parallel/latlon_spmd.py-17-
packages/core/legoesm/parallel/latlon_spmd.py-18-This is the lat-lon analogue of the cubed-sphere
--
packages/core/legoesm/parallel/latlon_spmd.py-78-
packages/core/legoesm/parallel/latlon_spmd.py-79-
packages/core/legoesm/parallel/latlon_spmd.py-80-def lon_ring_ghosts_spmd(f, mesh, halo: int = 1):
packages/core/legoesm/parallel/latlon_spmd.py-81-    """Periodic LONGITUDE ghosts (axis 1) under the 2-D lat-lon SPMD mesh.
packages/core/legoesm/parallel/latlon_spmd.py-82-
packages/core/legoesm/parallel/latlon_spmd.py-83-    The SPMD twin of the local ``jnp.pad(mode="wrap")`` lon wrap (band path)
packages/core/legoesm/parallel/latlon_spmd.py-84-    and of the MPI 2-D pencil's ``exchange_halo_lon``: each tile owns a lon
packages/core/legoesm/parallel/latlon_spmd.py-85-    SECTOR, so its east/west ghost columns are the neighbouring tiles' edge
packages/core/legoesm/parallel/latlon_spmd.py:86:    columns, moved by ``jax.lax.ppermute`` over the ``"lon"`` mesh axis (the
packages/core/legoesm/parallel/latlon_spmd.py-87-    periodic wrap IS the cyclic ring permutation —
packages/core/legoesm/parallel/latlon_spmd.py-88-    :func:`latlon_lon_ring_perms`).  ``p_lon == 1`` (a degenerate lon axis /
packages/core/legoesm/parallel/latlon_spmd.py-89-    the 1-D-band-equivalent (N, 1) mesh) is the LOCAL wrap, chosen by a
packages/core/legoesm/parallel/latlon_spmd.py-90-    STATIC Python branch — bit-identical to the band path's ``jnp.pad`` (no
packages/core/legoesm/parallel/latlon_spmd.py-91-    collective is emitted at all).
packages/core/legoesm/parallel/latlon_spmd.py-92-
packages/core/legoesm/parallel/latlon_spmd.py-93-    Lon axis is axis 1; the field must be CELL-ALIGNED in lon (width
packages/core/legoesm/parallel/latlon_spmd.py-94-    ``n_lon_local`` — never an ``n_lon_local+1`` u-face field, whose seam
--
packages/core/legoesm/parallel/latlon_spmd.py-102-        return f
packages/core/legoesm/parallel/latlon_spmd.py-103-    p_lon = int(mesh.shape["lon"])
packages/core/legoesm/parallel/latlon_spmd.py-104-    if p_lon == 1:
packages/core/legoesm/parallel/latlon_spmd.py-105-        pad = ((0, 0), (halo, halo)) + ((0, 0),) * (f.ndim - 2)
packages/core/legoesm/parallel/latlon_spmd.py-106-        return jnp.pad(f, pad, mode="wrap")
packages/core/legoesm/parallel/latlon_spmd.py-107-    perm_to_west, perm_to_east = latlon_lon_ring_perms(p_lon)
packages/core/legoesm/parallel/latlon_spmd.py-108-    # My EAST ghost = east neighbour's west edge (sources send WEST edges to
packages/core/legoesm/parallel/latlon_spmd.py-109-    # their west neighbour); my WEST ghost = west neighbour's east edge.
packages/core/legoesm/parallel/latlon_spmd.py:110:    east_ghost = jax.lax.ppermute(f[:, :halo], "lon", perm_to_west)
packages/core/legoesm/parallel/latlon_spmd.py:111:    west_ghost = jax.lax.ppermute(f[:, -halo:], "lon", perm_to_east)
packages/core/legoesm/parallel/latlon_spmd.py-112-    return jnp.concatenate([west_ghost, f, east_ghost], axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-113-
packages/core/legoesm/parallel/latlon_spmd.py-114-
packages/core/legoesm/parallel/latlon_spmd.py-115-def reconstruct_vface_lower(v_lower, axis: str, perm_north):
packages/core/legoesm/parallel/latlon_spmd.py-116-    """Rebuild the ``n_lat+1`` staggered v-faces from the ``n_lat``-row
packages/core/legoesm/parallel/latlon_spmd.py-117-    ``v_lower`` representation, INSIDE a ``shard_map`` over ``axis``.
packages/core/legoesm/parallel/latlon_spmd.py-118-
packages/core/legoesm/parallel/latlon_spmd.py-119-    The staggered meridional velocity ``v`` has a leading dim ``n_lat+1`` (faces
--
packages/core/legoesm/parallel/latlon_spmd.py-132-    v_lower : array ``(n_lat_band, n_lon[, nlev])``
packages/core/legoesm/parallel/latlon_spmd.py-133-    axis : the ``shard_map`` mesh axis name (``"lat"``).
packages/core/legoesm/parallel/latlon_spmd.py-134-    perm_north : the ``(src, dst)`` pairs from :func:`latlon_band_perms`.
packages/core/legoesm/parallel/latlon_spmd.py-135-
packages/core/legoesm/parallel/latlon_spmd.py-136-    Returns
packages/core/legoesm/parallel/latlon_spmd.py-137-    -------
packages/core/legoesm/parallel/latlon_spmd.py-138-    array ``(n_lat_band + 1, n_lon[, nlev])`` — the band's full v-faces.
packages/core/legoesm/parallel/latlon_spmd.py-139-    """
packages/core/legoesm/parallel/latlon_spmd.py:140:    boundary = jax.lax.ppermute(v_lower[0:1], axis, perm_north)
packages/core/legoesm/parallel/latlon_spmd.py-141-    return jnp.concatenate([v_lower, boundary], axis=0)
packages/core/legoesm/parallel/latlon_spmd.py-142-
packages/core/legoesm/parallel/latlon_spmd.py-143-
packages/core/legoesm/parallel/latlon_spmd.py-144-def to_vface_lower(v_full):
packages/core/legoesm/parallel/latlon_spmd.py-145-    """Inverse of :func:`reconstruct_vface_lower`: drop the north boundary face
packages/core/legoesm/parallel/latlon_spmd.py-146-    (owned by the next band) to return to the ``n_lat``-row ``v_lower``.
packages/core/legoesm/parallel/latlon_spmd.py-147-
packages/core/legoesm/parallel/latlon_spmd.py-148-    Round-trip identity ``to_vface_lower(reconstruct_vface_lower(v_lower)) ==
--
packages/core/legoesm/parallel/latlon_spmd.py-194-            row = fields[i][0:1]
packages/core/legoesm/parallel/latlon_spmd.py-195-            w = 1
packages/core/legoesm/parallel/latlon_spmd.py-196-            for s in row.shape[1:]:
packages/core/legoesm/parallel/latlon_spmd.py-197-                w *= int(s)
packages/core/legoesm/parallel/latlon_spmd.py-198-            widths.append(w)
packages/core/legoesm/parallel/latlon_spmd.py-199-            flats.append(row.reshape(1, w))
packages/core/legoesm/parallel/latlon_spmd.py-200-        buf = jnp.concatenate(flats, axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-201-        # ONE ppermute for the whole dtype group (north band receives 0).
packages/core/legoesm/parallel/latlon_spmd.py:202:        recv = jax.lax.ppermute(buf, axis, perm_north)
packages/core/legoesm/parallel/latlon_spmd.py-203-        off = 0
packages/core/legoesm/parallel/latlon_spmd.py-204-        for k, i in enumerate(idxs):
packages/core/legoesm/parallel/latlon_spmd.py-205-            w = widths[k]
packages/core/legoesm/parallel/latlon_spmd.py-206-            tail = fields[i].shape[1:]
packages/core/legoesm/parallel/latlon_spmd.py-207-            boundaries[i] = recv[:, off:off + w].reshape((1,) + tail)
packages/core/legoesm/parallel/latlon_spmd.py-208-            off += w
packages/core/legoesm/parallel/latlon_spmd.py-209-
packages/core/legoesm/parallel/latlon_spmd.py-210-    return tuple(
--
packages/core/legoesm/parallel/latlon_spmd.py-244-    Returns
packages/core/legoesm/parallel/latlon_spmd.py-245-    -------
packages/core/legoesm/parallel/latlon_spmd.py-246-    array ``(n_lat_local, n_lon_local + 1[, nlev])`` — the tile's u-faces.
packages/core/legoesm/parallel/latlon_spmd.py-247-    """
packages/core/legoesm/parallel/latlon_spmd.py-248-    if p_lon == 1:
packages/core/legoesm/parallel/latlon_spmd.py-249-        boundary = u_left[:, 0:1]
packages/core/legoesm/parallel/latlon_spmd.py-250-    else:
packages/core/legoesm/parallel/latlon_spmd.py-251-        perm_to_west, _ = latlon_lon_ring_perms(p_lon)
packages/core/legoesm/parallel/latlon_spmd.py:252:        boundary = jax.lax.ppermute(u_left[:, 0:1], axis, perm_to_west)
packages/core/legoesm/parallel/latlon_spmd.py-253-    return jnp.concatenate([u_left, boundary], axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-254-
packages/core/legoesm/parallel/latlon_spmd.py-255-
packages/core/legoesm/parallel/latlon_spmd.py-256-def to_uface_left(u_full):
packages/core/legoesm/parallel/latlon_spmd.py-257-    """Inverse of :func:`reconstruct_uface_left`: drop the east seam column
packages/core/legoesm/parallel/latlon_spmd.py-258-    (owned as the east neighbour's column 0) to return to the
packages/core/legoesm/parallel/latlon_spmd.py-259-    ``n_lon_local``-column ``u_left``.
packages/core/legoesm/parallel/latlon_spmd.py-260-
--
packages/core/legoesm/parallel/latlon_spmd.py-384-            f"(the E/W extension strips would exceed the neighbour tile); "
packages/core/legoesm/parallel/latlon_spmd.py-385-            f"use the all_gather fold for this degenerate tiling")
packages/core/legoesm/parallel/latlon_spmd.py-386-    W = w * p_lon
packages/core/legoesm/parallel/latlon_spmd.py-387-    # 1. E/W-extend the edge rows by 2h (one lon ring ppermute pair).
packages/core/legoesm/parallel/latlon_spmd.py-388-    ext = lon_ring_ghosts_spmd(edge, mesh, halo=2 * h)  # (halo, w+4h[, lev])
packages/core/legoesm/parallel/latlon_spmd.py-389-    # 2. ONE antipodal ppermute over the lon ring (shift by p_lon/2 is a
packages/core/legoesm/parallel/latlon_spmd.py-390-    # bijection, and c' != c for every even p_lon >= 2).
packages/core/legoesm/parallel/latlon_spmd.py-391-    perm_anti = [(s, (s + p_lon // 2) % p_lon) for s in range(p_lon)]
packages/core/legoesm/parallel/latlon_spmd.py:392:    recv = jax.lax.ppermute(ext, "lon", perm_anti)
packages/core/legoesm/parallel/latlon_spmd.py-393-    # 3. lat-mirror + sign (the _pole_fold row flip), then the window
packages/core/legoesm/parallel/latlon_spmd.py-394-    # column map (derivation above; seam-straddling windows mix branches).
packages/core/legoesm/parallel/latlon_spmd.py-395-    sign = -1.0 if negate else 1.0
packages/core/legoesm/parallel/latlon_spmd.py-396-    recv = sign * recv[::-1]
packages/core/legoesm/parallel/latlon_spmd.py-397-    t = jnp.arange(w + 2 * h)
packages/core/legoesm/parallel/latlon_spmd.py-398-    j = lon_index * w + t                       # global padded column
packages/core/legoesm/parallel/latlon_spmd.py-399-    r = jnp.where(j < W // 2 + h, t + 2 * h, t)
packages/core/legoesm/parallel/latlon_spmd.py-400-    return jnp.take(recv, r, axis=1)
--
packages/core/legoesm/parallel/latlon_spmd.py-458-        # 1. longitude periodic wrap (LOCAL — full lon circle per band).
packages/core/legoesm/parallel/latlon_spmd.py-459-        pad_lon = ((0, 0),) + ((halo, halo),) + ((0, 0),) * (tile.ndim - 2)
packages/core/legoesm/parallel/latlon_spmd.py-460-        data_lon = jnp.pad(tile, pad_lon, mode="wrap")  # (nl, n_lon+2h[, lev])
packages/core/legoesm/parallel/latlon_spmd.py-461-
packages/core/legoesm/parallel/latlon_spmd.py-462-        # 2. latitude band ppermute of the edge rows (axis 0 = south->north,
packages/core/legoesm/parallel/latlon_spmd.py-463-        # so row 0 is the SOUTH edge, row -1 the NORTH edge).
packages/core/legoesm/parallel/latlon_spmd.py-464-        south_edge = data_lon[:halo]   # my south rows -> band below (b-1)'s N ghost
packages/core/legoesm/parallel/latlon_spmd.py-465-        north_edge = data_lon[-halo:]  # my north rows -> band above (b+1)'s S ghost
packages/core/legoesm/parallel/latlon_spmd.py:466:        north_recv = jax.lax.ppermute(south_edge, axis, perm_north)  # b's N ghost = b+1's south edge
packages/core/legoesm/parallel/latlon_spmd.py:467:        south_recv = jax.lax.ppermute(north_edge, axis, perm_south)  # b's S ghost = b-1's north edge
packages/core/legoesm/parallel/latlon_spmd.py-468-
packages/core/legoesm/parallel/latlon_spmd.py-469-        # 3. pole fold at the end bands (ppermute non-targets receive zeros).
packages/core/legoesm/parallel/latlon_spmd.py-470-        b = jax.lax.axis_index(axis)
packages/core/legoesm/parallel/latlon_spmd.py-471-        south_ghost = jnp.where(b == 0,
packages/core/legoesm/parallel/latlon_spmd.py-472-                                _pole_fold(data_lon[:halo], negate), south_recv)
packages/core/legoesm/parallel/latlon_spmd.py-473-        north_ghost = jnp.where(b == n_dev - 1,
packages/core/legoesm/parallel/latlon_spmd.py-474-                                _pole_fold(data_lon[-halo:], negate), north_recv)
packages/core/legoesm/parallel/latlon_spmd.py-475-        return jnp.concatenate([south_ghost, data_lon, north_ghost], axis=0)
--
packages/core/legoesm/parallel/latlon_spmd.py-554-        # 1. latitude ppermute of the (pre-lon-pad) edge rows over "lat".
packages/core/legoesm/parallel/latlon_spmd.py-555-        # p_lat == 1 (a pure lon split): both lat ends are physical poles —
packages/core/legoesm/parallel/latlon_spmd.py-556-        # no lat neighbour exists, so skip the (empty-perm) ppermute
packages/core/legoesm/parallel/latlon_spmd.py-557-        # statically; the zero rows are overwritten by the folds below.
packages/core/legoesm/parallel/latlon_spmd.py-558-        if p_lat == 1:
packages/core/legoesm/parallel/latlon_spmd.py-559-            north_recv = jnp.zeros_like(tile[:halo])
packages/core/legoesm/parallel/latlon_spmd.py-560-            south_recv = jnp.zeros_like(tile[-halo:])
packages/core/legoesm/parallel/latlon_spmd.py-561-        else:
packages/core/legoesm/parallel/latlon_spmd.py:562:            north_recv = jax.lax.ppermute(tile[:halo], "lat", perm_north)
packages/core/legoesm/parallel/latlon_spmd.py:563:            south_recv = jax.lax.ppermute(tile[-halo:], "lat", perm_south)
packages/core/legoesm/parallel/latlon_spmd.py-564-        ext = jnp.concatenate([south_recv, tile, north_recv], axis=0)
packages/core/legoesm/parallel/latlon_spmd.py-565-
packages/core/legoesm/parallel/latlon_spmd.py-566-        # 2. longitude ring ghosts on the lat-EXTENDED block (fills corners
packages/core/legoesm/parallel/latlon_spmd.py-567-        # from the E/W neighbour's lat-ghost rows == the diagonal tile).
packages/core/legoesm/parallel/latlon_spmd.py-568-        ext = lon_ring_ghosts_spmd(ext, mesh, halo=halo)
packages/core/legoesm/parallel/latlon_spmd.py-569-
packages/core/legoesm/parallel/latlon_spmd.py-570-        # 3. pole fold at the physical pole tiles (ppermute non-targets
packages/core/legoesm/parallel/latlon_spmd.py-571-        # received zeros in step 1; overwritten here on the pole rows).
--
packages/core/legoesm/parallel/latlon_spmd.py-616-    perm_north, perm_south = latlon_band_perms(n_dev)
packages/core/legoesm/parallel/latlon_spmd.py-617-
packages/core/legoesm/parallel/latlon_spmd.py-618-    def body(tile):
packages/core/legoesm/parallel/latlon_spmd.py-619-        # lat-band ppermute of the edge rows (axis 0 = south->north; row 0 is the
packages/core/legoesm/parallel/latlon_spmd.py-620-        # SOUTH edge, row -1 the NORTH edge).  No lon pad (pad_with_pole_bc_lat
packages/core/legoesm/parallel/latlon_spmd.py-621-        # leaves lon untouched).
packages/core/legoesm/parallel/latlon_spmd.py-622-        south_edge = tile[:halo]       # my south rows -> band below (b-1)'s N ghost
packages/core/legoesm/parallel/latlon_spmd.py-623-        north_edge = tile[-halo:]      # my north rows -> band above (b+1)'s S ghost
packages/core/legoesm/parallel/latlon_spmd.py:624:        north_recv = jax.lax.ppermute(south_edge, axis, perm_north)  # b's N ghost = b+1's south edge
packages/core/legoesm/parallel/latlon_spmd.py:625:        south_recv = jax.lax.ppermute(north_edge, axis, perm_south)  # b's S ghost = b-1's north edge
packages/core/legoesm/parallel/latlon_spmd.py-626-
packages/core/legoesm/parallel/latlon_spmd.py-627-        # CONSTANT wall pad at the physical pole end bands (ppermute non-targets
packages/core/legoesm/parallel/latlon_spmd.py-628-        # receive zeros from these rows, but the where below overrides them with
packages/core/legoesm/parallel/latlon_spmd.py-629-        # the wall constant of the right shape).
packages/core/legoesm/parallel/latlon_spmd.py-630-        b = jax.lax.axis_index(axis)
packages/core/legoesm/parallel/latlon_spmd.py-631-        south_wall = jnp.full_like(south_recv, south_value)
packages/core/legoesm/parallel/latlon_spmd.py-632-        north_wall = jnp.full_like(north_recv, north_value)
packages/core/legoesm/parallel/latlon_spmd.py-633-        south_ghost = jnp.where(b == 0, south_wall, south_recv)
--
packages/core/legoesm/parallel/latlon_spmd.py-701-                for s in edge_shape[1:]:
packages/core/legoesm/parallel/latlon_spmd.py-702-                    w *= int(s)
packages/core/legoesm/parallel/latlon_spmd.py-703-                widths.append(w)
packages/core/legoesm/parallel/latlon_spmd.py-704-                flats.append((fields[i][:halo].reshape(halo, w),
packages/core/legoesm/parallel/latlon_spmd.py-705-                              fields[i][-halo:].reshape(halo, w)))
packages/core/legoesm/parallel/latlon_spmd.py-706-            south_buf = jnp.concatenate([s for s, _ in flats], axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-707-            north_buf = jnp.concatenate([n for _, n in flats], axis=1)
packages/core/legoesm/parallel/latlon_spmd.py-708-            # ONE ppermute pair for the whole dtype group.
packages/core/legoesm/parallel/latlon_spmd.py:709:            north_recv = jax.lax.ppermute(south_buf, axis, perm_north)
packages/core/legoesm/parallel/latlon_spmd.py:710:            south_recv = jax.lax.ppermute(north_buf, axis, perm_south)
packages/core/legoesm/parallel/latlon_spmd.py-711-            off = 0
packages/core/legoesm/parallel/latlon_spmd.py-712-            for k, i in enumerate(idxs):
packages/core/legoesm/parallel/latlon_spmd.py-713-                w = widths[k]
packages/core/legoesm/parallel/latlon_spmd.py-714-                tail = fields[i].shape[1:]
packages/core/legoesm/parallel/latlon_spmd.py-715-                sr = south_recv[:, off:off + w].reshape((halo,) + tail)
packages/core/legoesm/parallel/latlon_spmd.py-716-                nr = north_recv[:, off:off + w].reshape((halo,) + tail)
packages/core/legoesm/parallel/latlon_spmd.py-717-                s_wall = jnp.full_like(sr, south_values[i])
packages/core/legoesm/parallel/latlon_spmd.py-718-                n_wall = jnp.full_like(nr, north_values[i])
--
packages/core/legoesm/parallel/async_halo.py-87-       This is a **legacy** entry point.  For production multi-device
packages/core/legoesm/parallel/async_halo.py-88-       halo exchange, use the SPMD backend activated via
packages/core/legoesm/parallel/async_halo.py-89-       ``activate_spmd_halo_backend()`` in ``cubesphere_exchange.py``
packages/core/legoesm/parallel/async_halo.py-90-       (ppermute multiface exchange; all_gather only as the explicit
packages/core/legoesm/parallel/async_halo.py-91-       ``LEGOESM_SPMD_FORCE_ALLGATHER=1`` diagnostic — it replicates
packages/core/legoesm/parallel/async_halo.py-92-       compute, HLO probe job 8456476).
packages/core/legoesm/parallel/async_halo.py-93-
packages/core/legoesm/parallel/async_halo.py-94-    When running on multi-GPU/TPU with JAX sharding, this uses
packages/core/legoesm/parallel/async_halo.py:95:    ``jax.lax.ppermute`` for device-to-device communication instead of MPI.
packages/core/legoesm/parallel/async_halo.py-96-    Falls back to the local halo pad when mesh is None.
packages/core/legoesm/parallel/async_halo.py-97-
packages/core/legoesm/parallel/async_halo.py-98-    Parameters
packages/core/legoesm/parallel/async_halo.py-99-    ----------
packages/core/legoesm/parallel/async_halo.py-100-    data : jax.Array
packages/core/legoesm/parallel/async_halo.py-101-        Field to exchange halos for, shape (6, n, n) or (6, n, n, nlev).
packages/core/legoesm/parallel/async_halo.py-102-    grid : CubedSphereGrid
packages/core/legoesm/parallel/async_halo.py-103-        Grid with halo metadata and connectivity information.
--
packages/core/legoesm/parallel/async_halo.py-111-    jax.Array
packages/core/legoesm/parallel/async_halo.py-112-        Halo-padded field with same shape as input but with halo regions filled.
packages/core/legoesm/parallel/async_halo.py-113-
packages/core/legoesm/parallel/async_halo.py-114-    Notes
packages/core/legoesm/parallel/async_halo.py-115-    -----
packages/core/legoesm/parallel/async_halo.py-116-    The ppermute implementation is experimental and:
packages/core/legoesm/parallel/async_halo.py-117-
packages/core/legoesm/parallel/async_halo.py-118-    1. Uses the grid's face connectivity to define permutation patterns.
packages/core/legoesm/parallel/async_halo.py:119:    2. Applies ``jax.lax.ppermute`` calls for each edge/halo strip.
packages/core/legoesm/parallel/async_halo.py-120-    3. Avoids MPI entirely, keeping computation on-device.
packages/core/legoesm/parallel/async_halo.py-121-    4. Integrates with JAX's collective operations for potential
packages/core/legoesm/parallel/async_halo.py-122-       compiler-level optimizations.
packages/core/legoesm/parallel/async_halo.py-123-
packages/core/legoesm/parallel/async_halo.py-124-    This path has not been validated beyond single-node multi-device setups.
packages/core/legoesm/parallel/async_halo.py-125-    """
packages/core/legoesm/parallel/async_halo.py-126-    if mesh is not None:
packages/core/legoesm/parallel/async_halo.py-127-        import warnings
--
packages/core/legoesm/parallel/async_halo.py-146-        )
packages/core/legoesm/parallel/async_halo.py-147-        return _ppermute_halo_exchange(data, grid, mesh)
packages/core/legoesm/parallel/async_halo.py-148-    # Fall back to the standard local halo pad (no MPI needed for
packages/core/legoesm/parallel/async_halo.py-149-    # single-node multi-GPU when XLA handles data movement via sharding).
packages/core/legoesm/parallel/async_halo.py-150-    return pad_halo(data)
packages/core/legoesm/parallel/async_halo.py-151-
packages/core/legoesm/parallel/async_halo.py-152-
packages/core/legoesm/parallel/async_halo.py-153-def _ppermute_halo_exchange(data, grid, mesh):
packages/core/legoesm/parallel/async_halo.py:154:    """Implement halo exchange via jax.lax.ppermute (experimental).
packages/core/legoesm/parallel/async_halo.py-155-
packages/core/legoesm/parallel/async_halo.py-156-    .. warning::
packages/core/legoesm/parallel/async_halo.py-157-
packages/core/legoesm/parallel/async_halo.py-158-       This function is experimental and has not been validated beyond
packages/core/legoesm/parallel/async_halo.py-159-       basic single-node multi-device setups.  It may produce incorrect
packages/core/legoesm/parallel/async_halo.py-160-       halo data for complex connectivity patterns or multi-node runs.
packages/core/legoesm/parallel/async_halo.py-161-
packages/core/legoesm/parallel/async_halo.py-162-    Uses the cubed-sphere CONNECTIVITY table to build permutation
--
packages/core/legoesm/parallel/async_halo.py-206-            continue
packages/core/legoesm/parallel/async_halo.py-207-
packages/core/legoesm/parallel/async_halo.py-208-        # Extract edge strips for all faces along this edge.
packages/core/legoesm/parallel/async_halo.py-209-        strips = jax.vmap(lambda f: extract_edge_strip(data, f, edge))(
packages/core/legoesm/parallel/async_halo.py-210-            jnp.arange(6)
packages/core/legoesm/parallel/async_halo.py-211-        )  # (6, n)
packages/core/legoesm/parallel/async_halo.py-212-
packages/core/legoesm/parallel/async_halo.py-213-        # Permute strips between devices.
packages/core/legoesm/parallel/async_halo.py:214:        strips_permuted = jax.lax.ppermute(
packages/core/legoesm/parallel/async_halo.py-215-            strips, axis_name="face", perm=perm,
packages/core/legoesm/parallel/async_halo.py-216-        )
packages/core/legoesm/parallel/async_halo.py-217-
packages/core/legoesm/parallel/async_halo.py-218-        # Place received strips into halo positions.
packages/core/legoesm/parallel/async_halo.py-219-        for src_face in range(6):
packages/core/legoesm/parallel/async_halo.py-220-            nbr_face, nbr_edge, is_reversed = CONNECTIVITY[src_face][edge]
packages/core/legoesm/parallel/async_halo.py-221-            strip = strips_permuted[nbr_face]
packages/core/legoesm/parallel/async_halo.py-222-            if is_reversed:
--
packages/core/legoesm/parallel/sharded_dynamics.py-6-Design overview
packages/core/legoesm/parallel/sharded_dynamics.py-7----------------
packages/core/legoesm/parallel/sharded_dynamics.py-8-The cubed-sphere has 6 faces, each carrying an (n, n) or (n, n, nlev)
packages/core/legoesm/parallel/sharded_dynamics.py-9-grid.  On a multi-device system the natural decomposition is:
packages/core/legoesm/parallel/sharded_dynamics.py-10-
packages/core/legoesm/parallel/sharded_dynamics.py-11-1. **Face sharding** (<=6 devices): assign one or more faces per device.
packages/core/legoesm/parallel/sharded_dynamics.py-12-   Each device computes the dynamics for its assigned faces, and halo
packages/core/legoesm/parallel/sharded_dynamics.py-13-   exchange (ghost-zone fill from neighbor faces) happens as a
packages/core/legoesm/parallel/sharded_dynamics.py:14:   ``jax.lax.ppermute``-like collective inside shard_map.
packages/core/legoesm/parallel/sharded_dynamics.py-15-
packages/core/legoesm/parallel/sharded_dynamics.py-16-2. **Sub-face tiling** (>6 devices): each face is further split into
packages/core/legoesm/parallel/sharded_dynamics.py-17-   a (tx x ty) tile grid, giving up to 6*tx*ty devices.  Halo exchange
packages/core/legoesm/parallel/sharded_dynamics.py-18-   happens both between tiles on the same face and across face boundaries.
packages/core/legoesm/parallel/sharded_dynamics.py-19-
packages/core/legoesm/parallel/sharded_dynamics.py-20-Both modes are handled transparently by the functions in this module.
packages/core/legoesm/parallel/sharded_dynamics.py-21-
packages/core/legoesm/parallel/sharded_dynamics.py-22-Compilation strategy
--
packages/core/legoesm/parallel/sharded_dynamics.py-1242-
packages/core/legoesm/parallel/sharded_dynamics.py-1243-#: Halo depth the SPMD Voronoi partition infra is built at.  ONE definition
packages/core/legoesm/parallel/sharded_dynamics.py-1244-#: consumed by both the production step factory and ``spmd_schedule_cost``:
packages/core/legoesm/parallel/sharded_dynamics.py-1245-#: a score computed at a different depth describes a different comm graph, and
packages/core/legoesm/parallel/sharded_dynamics.py-1246-#: two independently hardcoded 3s let production drift unnoticed.
packages/core/legoesm/parallel/sharded_dynamics.py-1247-SPMD_HALO_DEPTH = 3
packages/core/legoesm/parallel/sharded_dynamics.py-1248-
packages/core/legoesm/parallel/sharded_dynamics.py-1249-
packages/core/legoesm/parallel/sharded_dynamics.py:1250:def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
packages/core/legoesm/parallel/sharded_dynamics.py-1251-                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
packages/core/legoesm/parallel/sharded_dynamics.py-1252-                       ppermute_cells_per_device_threshold=2_000,
packages/core/legoesm/parallel/sharded_dynamics.py:1253:                       round_profile_for_device=None):
packages/core/legoesm/parallel/sharded_dynamics.py-1254-    """How much halo communication one ownership choice costs, computed offline.
packages/core/legoesm/parallel/sharded_dynamics.py-1255-
packages/core/legoesm/parallel/sharded_dynamics.py-1256-    Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
packages/core/legoesm/parallel/sharded_dynamics.py-1257-    ROUNDS one halo exchange needs -- the sequential collective launches that
packages/core/legoesm/parallel/sharded_dynamics.py-1258-    dominate MPAS strong scaling above ~64 devices.  Runs on a laptop: no GPU,
packages/core/legoesm/parallel/sharded_dynamics.py-1259-    no MPI, no benchmark job, so a split can be compared before it costs an
packages/core/legoesm/parallel/sharded_dynamics.py-1260-    allocation.
packages/core/legoesm/parallel/sharded_dynamics.py-1261-
packages/core/legoesm/parallel/sharded_dynamics.py-1262-    It calls the SAME builders production calls
packages/core/legoesm/parallel/sharded_dynamics.py-1263-    (:func:`_build_voronoi_partition_infra` then
packages/core/legoesm/parallel/sharded_dynamics.py:1264:    :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
packages/core/legoesm/parallel/sharded_dynamics.py-1265-    different question: a 1-ring ``cellsOnEdge`` adjacency graph reports 8
packages/core/legoesm/parallel/sharded_dynamics.py-1266-    rounds where the real depth-3-plus-closure graph reports 12-14.
packages/core/legoesm/parallel/sharded_dynamics.py-1267-
packages/core/legoesm/parallel/sharded_dynamics.py-1268-    WHAT THE NUMBER IS NOT
packages/core/legoesm/parallel/sharded_dynamics.py-1269-    ----------------------
packages/core/legoesm/parallel/sharded_dynamics.py-1270-    * ``n_rounds`` is per HALO FILL, not per model step.  A step costs
packages/core/legoesm/parallel/sharded_dynamics.py-1271-      ``n_rounds`` x (tendency evaluations per step), which depends on the
packages/core/legoesm/parallel/sharded_dynamics.py-1272-      configured integrator -- SSP-RK3 evaluates 3 times, but the MPAS default
packages/core/legoesm/parallel/sharded_dynamics.py-1273-      is ``ssp_rk54_scan``.  Multiply with the integrator you actually run.
packages/core/legoesm/parallel/sharded_dynamics.py-1274-    * ``n_rounds`` is NOT proven equal to the comm graph's ``max_degree``.
packages/core/legoesm/parallel/sharded_dynamics.py:1275:      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
packages/core/legoesm/parallel/sharded_dynamics.py-1276-      keeps the best; equality is MEASURED (compare the returned
packages/core/legoesm/parallel/sharded_dynamics.py-1277-      ``max_degree``), never assumed.  Do not claim "the colouring is already
packages/core/legoesm/parallel/sharded_dynamics.py-1278-      optimal so only ownership can help" from this function.
packages/core/legoesm/parallel/sharded_dynamics.py-1279-    * It scores the ppermute strategy.  Production auto-selects ALLGATHER when
packages/core/legoesm/parallel/sharded_dynamics.py-1280-      cells/device is below ``ppermute_cells_per_device_threshold``, in which
packages/core/legoesm/parallel/sharded_dynamics.py-1281-      case there is no ppermute schedule and this number is counterfactual --
packages/core/legoesm/parallel/sharded_dynamics.py-1282-      see the returned ``production_strategy``.
packages/core/legoesm/parallel/sharded_dynamics.py-1283-
--
packages/core/legoesm/parallel/sharded_dynamics.py-1306-    reorder_target : int | None
packages/core/legoesm/parallel/sharded_dynamics.py-1307-        Device count the reorder targets, when it differs from *n_dev*.
packages/core/legoesm/parallel/sharded_dynamics.py-1308-    already_reordered : bool
packages/core/legoesm/parallel/sharded_dynamics.py-1309-    halo_depth : int
packages/core/legoesm/parallel/sharded_dynamics.py-1310-        Must match production (3) or the graph is a different graph.
packages/core/legoesm/parallel/sharded_dynamics.py-1311-    ppermute_cells_per_device_threshold : int
packages/core/legoesm/parallel/sharded_dynamics.py-1312-        Mirror of the production auto-select threshold, only used to report
packages/core/legoesm/parallel/sharded_dynamics.py-1313-        ``production_strategy``.
packages/core/legoesm/parallel/sharded_dynamics.py:1314:    round_profile_for_device : int | None
packages/core/legoesm/parallel/sharded_dynamics.py:1315:        When set to a device id, also return ``round_profile``: the per-round
packages/core/legoesm/parallel/sharded_dynamics.py-1316-        payload THAT DEVICE exchanges, in schedule order, restricted to the
packages/core/legoesm/parallel/sharded_dynamics.py-1317-        rounds it actually participates in.
packages/core/legoesm/parallel/sharded_dynamics.py-1318-
packages/core/legoesm/parallel/sharded_dynamics.py-1319-        This exists to make a profiler trace interpretable.  An ``nsys``
packages/core/legoesm/parallel/sharded_dynamics.py-1320-        capture is per RANK, and a rank appears only in the colour classes
packages/core/legoesm/parallel/sharded_dynamics.py-1321-        that touch it -- at s9/np64 rank 0 shows 8 ``SendRecv`` per halo fill
packages/core/legoesm/parallel/sharded_dynamics.py-1322-        while the graph's ``max_degree`` is 11 -- so the k-th observed
packages/core/legoesm/parallel/sharded_dynamics.py-1323-        collective is the k-th round CONTAINING THAT DEVICE, not the k-th
--
packages/core/legoesm/parallel/sharded_dynamics.py-1386-            f"a device count that does not divide it silently mis-slices the "
packages/core/legoesm/parallel/sharded_dynamics.py-1387-            f"owned blocks. Score at a device count that divides the prepared "
packages/core/legoesm/parallel/sharded_dynamics.py-1388-            f"mesh.")
packages/core/legoesm/parallel/sharded_dynamics.py-1389-    (
packages/core/legoesm/parallel/sharded_dynamics.py-1390-        _stacked, _gc, _ge, _noc, _noe, max_lc, max_le, partitions, cell_owner,
packages/core/legoesm/parallel/sharded_dynamics.py-1391-    ) = _build_voronoi_partition_infra(prepared, n_dev, halo_depth=halo_depth)
packages/core/legoesm/parallel/sharded_dynamics.py-1392-    cells_per = n_cells // n_dev
packages/core/legoesm/parallel/sharded_dynamics.py-1393-    edges_per = n_edges // n_dev
packages/core/legoesm/parallel/sharded_dynamics.py:1394:    sched = _build_ppermute_schedule(
packages/core/legoesm/parallel/sharded_dynamics.py-1395-        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
packages/core/legoesm/parallel/sharded_dynamics.py-1396-    )
packages/core/legoesm/parallel/sharded_dynamics.py-1397-    return {
packages/core/legoesm/parallel/sharded_dynamics.py-1398-        "method": method,
packages/core/legoesm/parallel/sharded_dynamics.py-1399-        "resolved_method": resolved,
packages/core/legoesm/parallel/sharded_dynamics.py-1400-        "n_dev": n_dev,
packages/core/legoesm/parallel/sharded_dynamics.py-1401-        # Unknown for a pre-reordered mesh: the ownership is baked in and the
packages/core/legoesm/parallel/sharded_dynamics.py-1402-        # target that produced it is not recoverable from the mesh. Reporting
--
packages/core/legoesm/parallel/sharded_dynamics.py-1415-        "production_strategy": (
packages/core/legoesm/parallel/sharded_dynamics.py-1416-            None if n_dev == 1 else
packages/core/legoesm/parallel/sharded_dynamics.py-1417-            ("allgather" if cells_per < ppermute_cells_per_device_threshold
packages/core/legoesm/parallel/sharded_dynamics.py-1418-             else "ppermute")),
packages/core/legoesm/parallel/sharded_dynamics.py-1419-        "cells_per_device": cells_per,
packages/core/legoesm/parallel/sharded_dynamics.py-1420-        "max_local_cells": int(max_lc),
packages/core/legoesm/parallel/sharded_dynamics.py-1421-        "max_local_edges": int(max_le),
packages/core/legoesm/parallel/sharded_dynamics.py-1422-        **(
packages/core/legoesm/parallel/sharded_dynamics.py:1423:            {} if round_profile_for_device is None else
packages/core/legoesm/parallel/sharded_dynamics.py:1424:            {"round_profile": _round_profile(sched, round_profile_for_device,
packages/core/legoesm/parallel/sharded_dynamics.py-1425-                                             n_dev)}
packages/core/legoesm/parallel/sharded_dynamics.py-1426-        ),
packages/core/legoesm/parallel/sharded_dynamics.py-1427-    }
packages/core/legoesm/parallel/sharded_dynamics.py-1428-
packages/core/legoesm/parallel/sharded_dynamics.py-1429-
packages/core/legoesm/parallel/sharded_dynamics.py:1430:def _round_profile(sched, device, n_dev):
packages/core/legoesm/parallel/sharded_dynamics.py-1431-    """Per-round payload for ONE device, in the order it observes them.
packages/core/legoesm/parallel/sharded_dynamics.py-1432-
packages/core/legoesm/parallel/sharded_dynamics.py:1433:    See ``spmd_schedule_cost``'s ``round_profile_for_device``.  Only rounds
packages/core/legoesm/parallel/sharded_dynamics.py-1434-    whose colour class touches *device* are returned, because those are the
packages/core/legoesm/parallel/sharded_dynamics.py-1435-    only ones on which it issues a collective.
packages/core/legoesm/parallel/sharded_dynamics.py-1436-    """
packages/core/legoesm/parallel/sharded_dynamics.py-1437-    # Strict, like the n_dev check: int() would coerce 0.9 to 0 and silently
packages/core/legoesm/parallel/sharded_dynamics.py-1438-    # profile a different device than the caller named.
packages/core/legoesm/parallel/sharded_dynamics.py-1439-    if int(device) != device or not 0 <= device < n_dev:
packages/core/legoesm/parallel/sharded_dynamics.py-1440-        raise ValueError(
packages/core/legoesm/parallel/sharded_dynamics.py:1441:            f"round_profile_for_device must be an integer in [0, {n_dev}), "
packages/core/legoesm/parallel/sharded_dynamics.py-1442-            f"got {device!r}")
packages/core/legoesm/parallel/sharded_dynamics.py-1443-    device = int(device)
packages/core/legoesm/parallel/sharded_dynamics.py-1444-    out = []
packages/core/legoesm/parallel/sharded_dynamics.py-1445-    for r, perm in enumerate(sched["ppermute_perms"]):
packages/core/legoesm/parallel/sharded_dynamics.py-1446-        partner = next((dst for src, dst in perm if src == device), None)
packages/core/legoesm/parallel/sharded_dynamics.py-1447-        if partner is None:
packages/core/legoesm/parallel/sharded_dynamics.py-1448-            continue
packages/core/legoesm/parallel/sharded_dynamics.py-1449-        # halo_cells_per_round is the round's PADDED extent, and that is the
--
packages/core/legoesm/parallel/sharded_dynamics.py-1815-        rounds = max(ec.values(), default=-1) + 1
packages/core/legoesm/parallel/sharded_dynamics.py-1816-        if best_rounds is None or rounds < best_rounds:
packages/core/legoesm/parallel/sharded_dynamics.py-1817-            best_rounds, best_colors = rounds, ec
packages/core/legoesm/parallel/sharded_dynamics.py-1818-            if best_rounds <= max_degree:
packages/core/legoesm/parallel/sharded_dynamics.py-1819-                break            # hit the chromatic-index floor — optimal
packages/core/legoesm/parallel/sharded_dynamics.py-1820-    return best_colors, max_degree
packages/core/legoesm/parallel/sharded_dynamics.py-1821-
packages/core/legoesm/parallel/sharded_dynamics.py-1822-
packages/core/legoesm/parallel/sharded_dynamics.py:1823:def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
packages/core/legoesm/parallel/sharded_dynamics.py-1824-                             edges_per, max_lc, max_le):
packages/core/legoesm/parallel/sharded_dynamics.py-1825-    """Build a ppermute-based halo exchange schedule.
packages/core/legoesm/parallel/sharded_dynamics.py-1826-
packages/core/legoesm/parallel/sharded_dynamics.py-1827-    Instead of all-gathering the full state (O(N) communication),
packages/core/legoesm/parallel/sharded_dynamics.py:1828:    this schedule uses ``jax.lax.ppermute`` to exchange only halo data
packages/core/legoesm/parallel/sharded_dynamics.py-1829-    between neighboring devices.  The communication graph is edge-colored
packages/core/legoesm/parallel/sharded_dynamics.py-1830-    so that each round of ppermute moves data between non-conflicting
packages/core/legoesm/parallel/sharded_dynamics.py-1831-    pairs simultaneously.
packages/core/legoesm/parallel/sharded_dynamics.py-1832-
packages/core/legoesm/parallel/sharded_dynamics.py-1833-    Parameters
packages/core/legoesm/parallel/sharded_dynamics.py-1834-    ----------
packages/core/legoesm/parallel/sharded_dynamics.py-1835-    partitions : list[VoronoiPartition]
packages/core/legoesm/parallel/sharded_dynamics.py-1836-    cell_owner : np.ndarray, (nCells,)
--
packages/core/legoesm/parallel/sharded_dynamics.py-2119-
packages/core/legoesm/parallel/sharded_dynamics.py-2120-
packages/core/legoesm/parallel/sharded_dynamics.py-2121-def _unpack_cell_state(cell_buf, nlev):
packages/core/legoesm/parallel/sharded_dynamics.py-2122-    """Inverse of :func:`_pack_cell_state`: ``(T, p_s, phis, q_flat)``."""
packages/core/legoesm/parallel/sharded_dynamics.py-2123-    return (cell_buf[:, :nlev], cell_buf[:, nlev],
packages/core/legoesm/parallel/sharded_dynamics.py-2124-            cell_buf[:, nlev + 1], cell_buf[:, nlev + 2:])
packages/core/legoesm/parallel/sharded_dynamics.py-2125-
packages/core/legoesm/parallel/sharded_dynamics.py-2126-
packages/core/legoesm/parallel/sharded_dynamics.py:2127:def _ppermute_halo_fill(cell_pack, u_shard, halo_sl, ppermute_perms,
packages/core/legoesm/parallel/sharded_dynamics.py-2128-                        max_lc, max_le):
packages/core/legoesm/parallel/sharded_dynamics.py-2129-    """Fill (owned + halo) local buffers from owned shards via ppermute.
packages/core/legoesm/parallel/sharded_dynamics.py-2130-
packages/core/legoesm/parallel/sharded_dynamics.py-2131-    Runs INSIDE ``shard_map``.  ``cell_pack`` ``(cells_per, W)`` is the
packages/core/legoesm/parallel/sharded_dynamics.py-2132-    packed owned-cell buffer — ALL cell-centred prognostics (T | p_s |
packages/core/legoesm/parallel/sharded_dynamics.py-2133-    phis | tracers) concatenated on the trailing axis; ``u_shard``
packages/core/legoesm/parallel/sharded_dynamics.py-2134-    ``(edges_per, nlev)`` the owned-edge buffer.  Each edge-colored round
packages/core/legoesm/parallel/sharded_dynamics.py-2135-    posts ONE flat ppermute carrying BOTH entity classes for ALL packed
--
packages/core/legoesm/parallel/sharded_dynamics.py-2154-    cell_local = jnp.pad(cell_pack, ((0, max_lc + 1 - cells_per), (0, 0)))
packages/core/legoesm/parallel/sharded_dynamics.py-2155-    u_local = jnp.pad(u_shard, ((0, max_le + 1 - edges_per), (0, 0)))
packages/core/legoesm/parallel/sharded_dynamics.py-2156-
packages/core/legoesm/parallel/sharded_dynamics.py-2157-    for r, (sc, rc, se, re) in enumerate(halo_sl):
packages/core/legoesm/parallel/sharded_dynamics.py-2158-        send_c = cell_pack[sc[0]]             # (hc_r, W)
packages/core/legoesm/parallel/sharded_dynamics.py-2159-        send_e = u_shard[se[0]]               # (he_r, nlev)
packages/core/legoesm/parallel/sharded_dynamics.py-2160-        send_c_flat = send_c.ravel()
packages/core/legoesm/parallel/sharded_dynamics.py-2161-        send_packed = jnp.concatenate([send_c_flat, send_e.ravel()])
packages/core/legoesm/parallel/sharded_dynamics.py:2162:        recv_packed = jax.lax.ppermute(
packages/core/legoesm/parallel/sharded_dynamics.py-2163-            send_packed, "device", perm=ppermute_perms[r])
packages/core/legoesm/parallel/sharded_dynamics.py-2164-        split_at = send_c_flat.shape[0]       # static
packages/core/legoesm/parallel/sharded_dynamics.py-2165-        recv_c = recv_packed[:split_at].reshape(send_c.shape)
packages/core/legoesm/parallel/sharded_dynamics.py-2166-        recv_e = recv_packed[split_at:].reshape(send_e.shape)
packages/core/legoesm/parallel/sharded_dynamics.py-2167-        cell_local = cell_local.at[rc[0]].set(recv_c)
packages/core/legoesm/parallel/sharded_dynamics.py-2168-        u_local = u_local.at[re[0]].set(recv_e)
packages/core/legoesm/parallel/sharded_dynamics.py-2169-
packages/core/legoesm/parallel/sharded_dynamics.py-2170-    return cell_local[:max_lc], u_local[:max_le]
--
packages/core/legoesm/parallel/sharded_dynamics.py-2219-        ``.config``.
packages/core/legoesm/parallel/sharded_dynamics.py-2220-    dev_config : DeviceConfig
packages/core/legoesm/parallel/sharded_dynamics.py-2221-        From :func:`~legoesm.parallel.mesh.create_voronoi_device_mesh`.
packages/core/legoesm/parallel/sharded_dynamics.py-2222-    halo_strategy : str
packages/core/legoesm/parallel/sharded_dynamics.py-2223-        ``"auto"`` (default) selects ``"ppermute"`` for large grids and
packages/core/legoesm/parallel/sharded_dynamics.py-2224-        ``"allgather"`` for small ones based on
packages/core/legoesm/parallel/sharded_dynamics.py-2225-        *ppermute_cells_per_device_threshold*.
packages/core/legoesm/parallel/sharded_dynamics.py-2226-        ``"ppermute"`` forces neighbor-only exchange via
packages/core/legoesm/parallel/sharded_dynamics.py:2227:        ``jax.lax.ppermute`` — O(halo) communication.
packages/core/legoesm/parallel/sharded_dynamics.py-2228-        ``"allgather"`` forces the full-state all-gather —
packages/core/legoesm/parallel/sharded_dynamics.py-2229-        O(N) communication.
packages/core/legoesm/parallel/sharded_dynamics.py-2230-    ppermute_cells_per_device_threshold : int
packages/core/legoesm/parallel/sharded_dynamics.py-2231-        When ``halo_strategy="auto"``, use ppermute only if each device
packages/core/legoesm/parallel/sharded_dynamics.py-2232-        owns at least this many cells.  Below this threshold the
packages/core/legoesm/parallel/sharded_dynamics.py-2233-        per-round packing/scatter overhead of ppermute exceeds the
packages/core/legoesm/parallel/sharded_dynamics.py-2234-        communication savings over allgather.  Default: 2 000.
packages/core/legoesm/parallel/sharded_dynamics.py-2235-        (Lowered from 25 000 to avoid the O(N) allgather bottleneck
--
packages/core/legoesm/parallel/sharded_dynamics.py-2384-    # Strategy dispatch: ppermute (O(halo)) vs allgather (O(N))
packages/core/legoesm/parallel/sharded_dynamics.py-2385-    # ------------------------------------------------------------------
packages/core/legoesm/parallel/sharded_dynamics.py-2386-
packages/core/legoesm/parallel/sharded_dynamics.py-2387-    use_ppermute = halo_strategy == "ppermute"
packages/core/legoesm/parallel/sharded_dynamics.py-2388-
packages/core/legoesm/parallel/sharded_dynamics.py-2389-    if use_ppermute:
packages/core/legoesm/parallel/sharded_dynamics.py-2390-        # Build ppermute schedule: neighbor-only halo exchange
packages/core/legoesm/parallel/sharded_dynamics.py-2391-        t1 = time.time()
packages/core/legoesm/parallel/sharded_dynamics.py:2392:        pp_sched = _build_ppermute_schedule(
packages/core/legoesm/parallel/sharded_dynamics.py-2393-            partitions_out, cell_owner_out, n_dev,
packages/core/legoesm/parallel/sharded_dynamics.py-2394-            cells_per, edges_per, max_lc, max_le,
packages/core/legoesm/parallel/sharded_dynamics.py-2395-        )
packages/core/legoesm/parallel/sharded_dynamics.py-2396-        n_rounds = pp_sched['n_rounds']
packages/core/legoesm/parallel/sharded_dynamics.py-2397-        ppermute_perms = pp_sched['ppermute_perms']
packages/core/legoesm/parallel/sharded_dynamics.py-2398-
packages/core/legoesm/parallel/sharded_dynamics.py-2399-        # LOCAL-ONLY metadata: shard the per-round index arrays on the
packages/core/legoesm/parallel/sharded_dynamics.py-2400-        # leading device axis (each device holds only its own schedule

exec
/usr/bin/bash -lc "nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1420,1475p'
nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1820,1995p'
nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '2120,2180p'
nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '2380,2485p'
rg -n -C 5 \"_ppermute_halo_fill\\(\" packages/core/legoesm/parallel/sharded_dynamics.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 182ms:
  1420	        "max_local_cells": int(max_lc),
  1421	        "max_local_edges": int(max_le),
  1422	        **(
  1423	            {} if round_profile_for_device is None else
  1424	            {"round_profile": _round_profile(sched, round_profile_for_device,
  1425	                                             n_dev)}
  1426	        ),
  1427	    }
  1428	
  1429	
  1430	def _round_profile(sched, device, n_dev):
  1431	    """Per-round payload for ONE device, in the order it observes them.
  1432	
  1433	    See ``spmd_schedule_cost``'s ``round_profile_for_device``.  Only rounds
  1434	    whose colour class touches *device* are returned, because those are the
  1435	    only ones on which it issues a collective.
  1436	    """
  1437	    # Strict, like the n_dev check: int() would coerce 0.9 to 0 and silently
  1438	    # profile a different device than the caller named.
  1439	    if int(device) != device or not 0 <= device < n_dev:
  1440	        raise ValueError(
  1441	            f"round_profile_for_device must be an integer in [0, {n_dev}), "
  1442	            f"got {device!r}")
  1443	    device = int(device)
  1444	    out = []
  1445	    for r, perm in enumerate(sched["ppermute_perms"]):
  1446	        partner = next((dst for src, dst in perm if src == device), None)
  1447	        if partner is None:
  1448	            continue
  1449	        # halo_cells_per_round is the round's PADDED extent, and that is the
  1450	        # right payload measure rather than a per-pair count: the index
  1451	        # arrays are (n_dev, max_c) and ppermute moves the padded buffer, so
  1452	        # every pair in the round puts max_c entities on the wire.  (The
  1453	        # per-pair send maps cannot be recovered from those arrays anyway --
  1454	        # they are zero-padded and 0 is a valid index.)
  1455	        out.append({
  1456	            "round": r,
  1457	            "partner": int(partner),
  1458	            "halo_cells": int(sched["halo_cells_per_round"][r]),
  1459	            "halo_edges": int(sched["halo_edges_per_round"][r]),
  1460	        })
  1461	    return out
  1462	
  1463	
  1464	def _build_voronoi_partition_infra(global_mesh, n_dev, halo_depth=2):
  1465	    """Pre-compute per-device local meshes and gather/scatter indices.
  1466	
  1467	    After ``reorder_voronoi_for_sharding`` the global mesh has *both*
  1468	    cells and edges ordered by contiguous device blocks.  We compute
  1469	    partitions whose owned-entity boundaries exactly match the shard
  1470	    boundaries (``cells_per = nCells // n_dev``, ``edges_per = nEdges //
  1471	    n_dev``), then build local meshes with remapped connectivity.
  1472	
  1473	    Using ``partition_voronoi_mesh`` directly is unsuitable because it
  1474	    derives edge ownership from cell ownership, producing an uneven edge
  1475	    split that mismatches the even shard split.  Instead we construct the
  1820	    return best_colors, max_degree
  1821	
  1822	
  1823	def _build_ppermute_schedule(partitions, cell_owner, n_dev, cells_per,
  1824	                             edges_per, max_lc, max_le):
  1825	    """Build a ppermute-based halo exchange schedule.
  1826	
  1827	    Instead of all-gathering the full state (O(N) communication),
  1828	    this schedule uses ``jax.lax.ppermute`` to exchange only halo data
  1829	    between neighboring devices.  The communication graph is edge-colored
  1830	    so that each round of ppermute moves data between non-conflicting
  1831	    pairs simultaneously.
  1832	
  1833	    Parameters
  1834	    ----------
  1835	    partitions : list[VoronoiPartition]
  1836	    cell_owner : np.ndarray, (nCells,)
  1837	    n_dev, cells_per, edges_per : int
  1838	    max_lc, max_le : int
  1839	        Maximum local cell/edge counts (owned + halo) across devices.
  1840	
  1841	    Returns
  1842	    -------
  1843	    dict with keys:
  1844	        n_rounds, n_rounds_greedy, max_degree, coloring_method,
  1845	        ppermute_perms, send_cell_idx, recv_cell_pos,
  1846	        send_edge_idx, recv_edge_pos, halo_cells_per_round,
  1847	        halo_edges_per_round.
  1848	    """
  1849	    from collections import defaultdict
  1850	
  1851	    import numpy as np
  1852	
  1853	    # ------------------------------------------------------------------
  1854	    # 1. For each device pair, find which cells/edges cross the boundary
  1855	    # ------------------------------------------------------------------
  1856	    # halo_cells_from[d][d'] = global indices of d's halo cells owned by d'
  1857	    halo_cells_from: dict[int, dict[int, list[int]]] = defaultdict(
  1858	        lambda: defaultdict(list))
  1859	    halo_edges_from: dict[int, dict[int, list[int]]] = defaultdict(
  1860	        lambda: defaultdict(list))
  1861	
  1862	    for d, part in enumerate(partitions):
  1863	        for h_idx in range(part.n_owned_cells, part.n_local_cells):
  1864	            g = int(part.local_cells[h_idx])
  1865	            owner = int(cell_owner[g])
  1866	            halo_cells_from[d][owner].append(g)
  1867	
  1868	        for h_idx in range(part.n_owned_edges, part.n_local_edges):
  1869	            g = int(part.local_edges[h_idx])
  1870	            owner = min(g // edges_per, n_dev - 1)
  1871	            halo_edges_from[d][owner].append(g)
  1872	
  1873	    # ------------------------------------------------------------------
  1874	    # 2. Build undirected communication graph
  1875	    # ------------------------------------------------------------------
  1876	    comm_pairs: set[tuple[int, int]] = set()
  1877	    for d in range(n_dev):
  1878	        for d_prime in halo_cells_from[d]:
  1879	            if d != d_prime:
  1880	                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
  1881	        for d_prime in halo_edges_from[d]:
  1882	            if d != d_prime:
  1883	                comm_pairs.add((min(d, d_prime), max(d, d_prime)))
  1884	
  1885	    if not comm_pairs:
  1886	        return {
  1887	            'n_rounds': 0,
  1888	            'n_rounds_greedy': 0,
  1889	            'max_degree': 0,
  1890	            'coloring_method': 'none',
  1891	            'ppermute_perms': [],
  1892	            'send_cell_idx': [],
  1893	            'recv_cell_pos': [],
  1894	            'send_edge_idx': [],
  1895	            'recv_edge_pos': [],
  1896	            'halo_cells_per_round': [],
  1897	            'halo_edges_per_round': [],
  1898	        }
  1899	
  1900	    # ------------------------------------------------------------------
  1901	    # 3. Edge-color the graph: each color = one bidirectional ppermute
  1902	    #    ROUND, and the route-B lane is round-latency-bound (#1113), so
  1903	    #    fewer colors = directly less wall-clock. First-fit greedy is
  1904	    #    order-sensitive; the multi-start coloring reaches the
  1905	    #    chromatic-index floor (= max_degree) on every probed MPAS config
  1906	    #    where the legacy sorted greedy overshoots (up to 3 rounds at 16
  1907	    #    devices). It can never regress: the legacy sorted order is one of
  1908	    #    its candidates and it takes the min. Both are verified proper.
  1909	    # ------------------------------------------------------------------
  1910	    greedy_colors = _greedy_edge_coloring(comm_pairs)
  1911	    n_rounds_greedy = max(greedy_colors.values()) + 1
  1912	    multi_colors, max_degree = _multi_ordering_edge_coloring(comm_pairs)
  1913	    n_rounds_multi = max(multi_colors.values()) + 1
  1914	    # Adopt the multi-start coloring ONLY when it STRICTLY reduces rounds;
  1915	    # on a tie keep the exact legacy sorted-greedy coloring so the produced
  1916	    # schedule is byte-identical to before wherever there is no round win
  1917	    # (the win only appears at high device counts — >=16 on the probed
  1918	    # MPAS meshes). Both colorings are proper.
  1919	    if n_rounds_multi < n_rounds_greedy:
  1920	        edge_colors, n_rounds, coloring_method = (
  1921	            multi_colors, n_rounds_multi, "multi_greedy")
  1922	    else:
  1923	        edge_colors, n_rounds, coloring_method = (
  1924	            greedy_colors, n_rounds_greedy, "greedy")
  1925	    assert _check_proper_edge_coloring(edge_colors, comm_pairs), (
  1926	        "improper ppermute edge coloring — two same-round exchanges "
  1927	        "would collide at a device")
  1928	    rounds: dict[int, list[tuple[int, int]]] = defaultdict(list)
  1929	    for (u, v), color in edge_colors.items():
  1930	        rounds[color].append((u, v))
  1931	
  1932	    # ------------------------------------------------------------------
  1933	    # 4. Build directed send/recv maps for each device pair
  1934	    # ------------------------------------------------------------------
  1935	    # cell_send_map[(src, dst)] = list of owned-local indices in src to send
  1936	    # cell_recv_map[(dst, src)] = list of local positions in dst to place data
  1937	    cell_send_map: dict[tuple[int, int], list[int]] = {}
  1938	    cell_recv_map: dict[tuple[int, int], list[int]] = {}
  1939	    edge_send_map: dict[tuple[int, int], list[int]] = {}
  1940	    edge_recv_map: dict[tuple[int, int], list[int]] = {}
  1941	
  1942	    for d in range(n_dev):
  1943	        for d_prime, cells_g in halo_cells_from[d].items():
  1944	            if d_prime == d:
  1945	                continue
  1946	            # d_prime sends its owned cells that d needs as halo
  1947	            cell_send_map[(d_prime, d)] = [
  1948	                g - d_prime * cells_per for g in cells_g]
  1949	            cell_recv_map[(d, d_prime)] = [
  1950	                int(partitions[d].cell_g2l[g]) for g in cells_g]
  1951	
  1952	        for d_prime, edges_g in halo_edges_from[d].items():
  1953	            if d_prime == d:
  1954	                continue
  1955	            edge_send_map[(d_prime, d)] = [
  1956	                g - d_prime * edges_per for g in edges_g]
  1957	            edge_recv_map[(d, d_prime)] = [
  1958	                int(partitions[d].edge_g2l[g]) for g in edges_g]
  1959	
  1960	    # ------------------------------------------------------------------
  1961	    # 5. Assemble per-round ppermute patterns and index arrays
  1962	    # ------------------------------------------------------------------
  1963	    ppermute_perms_out: list[list[tuple[int, int]]] = []
  1964	    send_cell_idx_out: list[jnp.ndarray] = []
  1965	    recv_cell_pos_out: list[jnp.ndarray] = []
  1966	    send_edge_idx_out: list[jnp.ndarray] = []
  1967	    recv_edge_pos_out: list[jnp.ndarray] = []
  1968	    halo_cells_per_round: list[int] = []
  1969	    halo_edges_per_round: list[int] = []
  1970	
  1971	    for r in range(n_rounds):
  1972	        # Max halo size across all pairs in this round
  1973	        max_c = 0
  1974	        max_e = 0
  1975	        for u, v in rounds[r]:
  1976	            for src, dst in [(u, v), (v, u)]:
  1977	                max_c = max(max_c, len(cell_send_map.get((src, dst), [])))
  1978	                max_e = max(max_e, len(edge_send_map.get((src, dst), [])))
  1979	        max_c = max(max_c, 1)  # at least 1 for array shape
  1980	        max_e = max(max_e, 1)
  1981	        halo_cells_per_round.append(max_c)
  1982	        halo_edges_per_round.append(max_e)
  1983	
  1984	        # Bidirectional ppermute pattern
  1985	        perm: list[tuple[int, int]] = []
  1986	        partner: dict[int, int] = {}
  1987	        for u, v in rounds[r]:
  1988	            perm.append((u, v))
  1989	            perm.append((v, u))
  1990	            partner[u] = v
  1991	            partner[v] = u
  1992	        ppermute_perms_out.append(perm)
  1993	
  1994	        # Per-device index arrays (padded with safe defaults)
  1995	        sc = np.zeros((n_dev, max_c), dtype=np.int64)
  2120	
  2121	def _unpack_cell_state(cell_buf, nlev):
  2122	    """Inverse of :func:`_pack_cell_state`: ``(T, p_s, phis, q_flat)``."""
  2123	    return (cell_buf[:, :nlev], cell_buf[:, nlev],
  2124	            cell_buf[:, nlev + 1], cell_buf[:, nlev + 2:])
  2125	
  2126	
  2127	def _ppermute_halo_fill(cell_pack, u_shard, halo_sl, ppermute_perms,
  2128	                        max_lc, max_le):
  2129	    """Fill (owned + halo) local buffers from owned shards via ppermute.
  2130	
  2131	    Runs INSIDE ``shard_map``.  ``cell_pack`` ``(cells_per, W)`` is the
  2132	    packed owned-cell buffer — ALL cell-centred prognostics (T | p_s |
  2133	    phis | tracers) concatenated on the trailing axis; ``u_shard``
  2134	    ``(edges_per, nlev)`` the owned-edge buffer.  Each edge-colored round
  2135	    posts ONE flat ppermute carrying BOTH entity classes for ALL packed
  2136	    fields — the SPMD mirror of route-A's batched union-neighbor exchange
  2137	    (one message per neighbor per dtype group; the compute-precision cast
  2138	    upstream guarantees a single dtype group here).
  2139	
  2140	    ``halo_sl`` is a tuple of per-round ``(send_cell_idx, recv_cell_pos,
  2141	    send_edge_idx, recv_edge_pos)`` tuples whose arrays are ALREADY
  2142	    device-local ``(1, n_round)`` shard_map arguments (``P("device")``
  2143	    specs) — per-rank LOCAL metadata; no device materializes the global
  2144	    schedule.  ``ppermute_perms`` is the static per-round permutation.
  2145	
  2146	    Returns ``(cell_local, u_local)`` of shapes ``(max_lc, W)`` /
  2147	    ``(max_le, nlev)``; ghost tail rows stay zero.
  2148	    """
  2149	    cells_per = cell_pack.shape[0]
  2150	    edges_per = u_shard.shape[0]
  2151	    # +1 garbage slot for padded scatter targets (trimmed at the end):
  2152	    # schedule rows are padded to the round's max halo count, and padding
  2153	    # entries target position max_lc / max_le.
  2154	    cell_local = jnp.pad(cell_pack, ((0, max_lc + 1 - cells_per), (0, 0)))
  2155	    u_local = jnp.pad(u_shard, ((0, max_le + 1 - edges_per), (0, 0)))
  2156	
  2157	    for r, (sc, rc, se, re) in enumerate(halo_sl):
  2158	        send_c = cell_pack[sc[0]]             # (hc_r, W)
  2159	        send_e = u_shard[se[0]]               # (he_r, nlev)
  2160	        send_c_flat = send_c.ravel()
  2161	        send_packed = jnp.concatenate([send_c_flat, send_e.ravel()])
  2162	        recv_packed = jax.lax.ppermute(
  2163	            send_packed, "device", perm=ppermute_perms[r])
  2164	        split_at = send_c_flat.shape[0]       # static
  2165	        recv_c = recv_packed[:split_at].reshape(send_c.shape)
  2166	        recv_e = recv_packed[split_at:].reshape(send_e.shape)
  2167	        cell_local = cell_local.at[rc[0]].set(recv_c)
  2168	        u_local = u_local.at[re[0]].set(recv_e)
  2169	
  2170	    return cell_local[:max_lc], u_local[:max_le]
  2171	
  2172	
  2173	def make_voronoi_sharded_step(
  2174	    model,
  2175	    dev_config: DeviceConfig,
  2176	    *,
  2177	    halo_strategy: str = "auto",
  2178	    ppermute_cells_per_device_threshold: int = 2_000,
  2179	    return_phys_state: bool = False,
  2180	):
  2380	
  2381	    nlev = model.sigma_coord.n_levels
  2382	
  2383	    # ------------------------------------------------------------------
  2384	    # Strategy dispatch: ppermute (O(halo)) vs allgather (O(N))
  2385	    # ------------------------------------------------------------------
  2386	
  2387	    use_ppermute = halo_strategy == "ppermute"
  2388	
  2389	    if use_ppermute:
  2390	        # Build ppermute schedule: neighbor-only halo exchange
  2391	        t1 = time.time()
  2392	        pp_sched = _build_ppermute_schedule(
  2393	            partitions_out, cell_owner_out, n_dev,
  2394	            cells_per, edges_per, max_lc, max_le,
  2395	        )
  2396	        n_rounds = pp_sched['n_rounds']
  2397	        ppermute_perms = pp_sched['ppermute_perms']
  2398	
  2399	        # LOCAL-ONLY metadata: shard the per-round index arrays on the
  2400	        # leading device axis (each device holds only its own schedule
  2401	        # rows) and thread them as shard_map ARGUMENTS — see the stacked
  2402	        # meshes above for why args, not closures.
  2403	        halo_args = tuple(
  2404	            (
  2405	                multiprocess_safe_device_put(
  2406	                    pp_sched['send_cell_idx'][r], dev_sharding),
  2407	                multiprocess_safe_device_put(
  2408	                    pp_sched['recv_cell_pos'][r], dev_sharding),
  2409	                multiprocess_safe_device_put(
  2410	                    pp_sched['send_edge_idx'][r], dev_sharding),
  2411	                multiprocess_safe_device_put(
  2412	                    pp_sched['recv_edge_pos'][r], dev_sharding),
  2413	            )
  2414	            for r in range(n_rounds)
  2415	        )
  2416	
  2417	        # Log halo exchange statistics (dry-state estimate: tracers add
  2418	        # nlev*n_tracers further cell channels to both strategies).
  2419	        total_pp_bytes = sum(
  2420	            hc * (nlev + 2) + he * nlev
  2421	            for hc, he in zip(pp_sched['halo_cells_per_round'],
  2422	                              pp_sched['halo_edges_per_round'])
  2423	        ) * 4  # float32
  2424	        ag_bytes = (nCells * (nlev + 2) + nEdges * nlev) * 4
  2425	        logger.info(
  2426	            "  ppermute schedule: %d rounds, max halo cells/edges per round: %s / %s",
  2427	            n_rounds,
  2428	            pp_sched['halo_cells_per_round'],
  2429	            pp_sched['halo_edges_per_round'],
  2430	        )
  2431	        logger.info(
  2432	            "  comm volume per stage: ppermute ~%.1f KB vs allgather ~%.1f KB (%.1fx reduction)",
  2433	            total_pp_bytes / 1024,
  2434	            ag_bytes / 1024,
  2435	            ag_bytes / max(total_pp_bytes, 1),
  2436	        )
  2437	        logger.info("  ppermute schedule built in %.3fs", time.time() - t1)
  2438	
  2439	    else:
  2440	        # ---- Legacy all-gather strategy: local gather indices ----
  2441	        halo_args = (
  2442	            multiprocess_safe_device_put(gather_cells, dev_sharding),
  2443	            multiprocess_safe_device_put(gather_edges, dev_sharding),
  2444	        )
  2445	
  2446	    # ------------------------------------------------------------------
  2447	    # shard_map kernel: packed full-state halo fill → local tendency
  2448	    # ------------------------------------------------------------------
  2449	    # The kernel is built per canonical tracer-key tuple (the keys are
  2450	    # part of the traced program: cell-pack width and the tracer dict
  2451	    # rebuilt on the local mesh).  Memoized so a stable state structure
  2452	    # reuses one shard_map object → one jit executable (no retrace).
  2453	
  2454	    mesh_in_specs = jax.tree.map(lambda _: P("device"), stacked_meshes)
  2455	    halo_in_specs = jax.tree.map(lambda _: P("device"), halo_args)
  2456	
  2457	    def _make_local_tendency(tkeys: tuple):
  2458	
  2459	        def _local_tendency(u_shard, T_shard, ps_shard, phis_shard,
  2460	                            q_shard, dt_val, mesh_sl, halo_sl):
  2461	            """Inside shard_map: full-state halo fill → local tendency.
  2462	
  2463	            ``q_shard`` is the tracer block ``(cells_per, nlev * n_q)``
  2464	            — tracers concatenated on the trailing axis in the canonical
  2465	            sorted-key WIRE order (width 0 for a dry run).  ``mesh_sl``
  2466	            / ``halo_sl`` are this device's P("device") slices of the
  2467	            stacked local meshes and the halo schedule (leading axis 1).
  2468	            """
  2469	            # Pack ALL cell-centred prognostics into a single buffer
  2470	            # (cells_per, nlev + 2 + nlev*n_q) via the shared wire-layout
  2471	            # helper (also driven directly by the sentinel routing test).
  2472	            cell_pack = _pack_cell_state(T_shard, ps_shard, phis_shard,
  2473	                                         q_shard)
  2474	
  2475	            if use_ppermute:
  2476	                cell_local, u_local = _ppermute_halo_fill(
  2477	                    cell_pack, u_shard, halo_sl, ppermute_perms,
  2478	                    max_lc, max_le,
  2479	                )
  2480	            else:
  2481	                cell_full = jax.lax.all_gather(
  2482	                    cell_pack, "device", axis=0, tiled=True)
  2483	                u_full = jax.lax.all_gather(
  2484	                    u_shard, "device", axis=0, tiled=True)
  2485	                gc, ge = halo_sl
2122-    """Inverse of :func:`_pack_cell_state`: ``(T, p_s, phis, q_flat)``."""
2123-    return (cell_buf[:, :nlev], cell_buf[:, nlev],
2124-            cell_buf[:, nlev + 1], cell_buf[:, nlev + 2:])
2125-
2126-
2127:def _ppermute_halo_fill(cell_pack, u_shard, halo_sl, ppermute_perms,
2128-                        max_lc, max_le):
2129-    """Fill (owned + halo) local buffers from owned shards via ppermute.
2130-
2131-    Runs INSIDE ``shard_map``.  ``cell_pack`` ``(cells_per, W)`` is the
2132-    packed owned-cell buffer — ALL cell-centred prognostics (T | p_s |
--
2471-            # helper (also driven directly by the sentinel routing test).
2472-            cell_pack = _pack_cell_state(T_shard, ps_shard, phis_shard,
2473-                                         q_shard)
2474-
2475-            if use_ppermute:
2476:                cell_local, u_local = _ppermute_halo_fill(
2477-                    cell_pack, u_shard, halo_sl, ppermute_perms,
2478-                    max_lc, max_le,
2479-                )
2480-            else:
2481-                cell_full = jax.lax.all_gather(

exec
/usr/bin/bash -lc "rg -n \"SPMD_HALO_DEPTH|def make_voronoi_sharded_step|halo_depth=\" packages/core/legoesm/parallel/sharded_dynamics.py tests -S | head -200
rg --files | rg -i \"(nsys|rank_.*sqlite|\\.nsys-rep|mpas.*hlo|hlo.*mpas)\" | head -200
rg -n -C 6 \"round_profile\" tests packages scripts | head -400" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 504ms:
packages/core/legoesm/parallel/sharded_dynamics.py:1247:SPMD_HALO_DEPTH = 3
packages/core/legoesm/parallel/sharded_dynamics.py:1251:                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
packages/core/legoesm/parallel/sharded_dynamics.py:1391:    ) = _build_voronoi_partition_infra(prepared, n_dev, halo_depth=halo_depth)
packages/core/legoesm/parallel/sharded_dynamics.py:1464:def _build_voronoi_partition_infra(global_mesh, n_dev, halo_depth=2):
packages/core/legoesm/parallel/sharded_dynamics.py:2173:def make_voronoi_sharded_step(
packages/core/legoesm/parallel/sharded_dynamics.py:2344:        "(nCells=%d, nEdges=%d, halo_depth=3, strategy=%s) ...",
packages/core/legoesm/parallel/sharded_dynamics.py:2359:                                       halo_depth=SPMD_HALO_DEPTH)
tests/bench/test_bench_voronoi_partition_methods.py:111:    q = mod.partition_quality(mesh, owner, 2, halo_depth=1)
tests/bench/test_bench_voronoi_partition_methods.py:116:    # halo_depth=1: each rank's halo = the 2 cells of the other rank that
tests/bench/test_bench_voronoi_partition_methods.py:141:    q = mod.partition_quality(mesh, owner, 4, halo_depth=2)
tests/bench/test_bench_voronoi_partition_methods.py:145:            mesh, 4, r, method="geometric", halo_depth=2, cell_owner=owner)
tests/bench/test_bench_voronoi_partition_methods.py:236:    from legoesm.parallel.sharded_dynamics import SPMD_HALO_DEPTH
tests/bench/test_bench_voronoi_partition_methods.py:239:    assert sc["schedule_halo_depth"] == SPMD_HALO_DEPTH
tests/distributed/test_voronoi_halo.py:426:         ) = _build_voronoi_partition_infra(reordered, n_ranks, halo_depth=2)
tests/distributed/test_voronoi_halo.py:471:         ) = _build_voronoi_partition_infra(reordered, n_ranks, halo_depth=2)
tests/distributed/test_voronoi_halo.py:537:         ) = _build_voronoi_partition_infra(reordered, n_ranks, halo_depth=2)
tests/parallel/test_spmd_schedule_cost.py:159:    assert sd.SPMD_HALO_DEPTH == 3
tests/parallel/test_spmd_schedule_cost.py:161:            .parameters["halo_depth"].default == sd.SPMD_HALO_DEPTH)
tests/parallel/test_spmd_schedule_cost.py:163:    assert "halo_depth=SPMD_HALO_DEPTH" in prod, (
tests/parallel/test_ppermute_edge_coloring.py:118:     cell_owner) = _build_voronoi_partition_infra(mesh, n_dev, halo_depth=3)
tests/parallel/test_voronoi_schedule_symmetry.py:36:                                      halo_depth=2)
tests/ocean/distributed/test_mpas_ocean_stage_halo.py:4:proved the multi-rank step consumes more stencil hops than ``halo_depth=2``
tests/parallel/test_mpas_atm_native_step.py:564:            mesh, n_dev, halo_depth=3)
tests/bench/test_analyze_nsys_rank_skew.py
scripts/bench/analyze_nsys_rank_skew.py
scripts/bench/analyze_nsys_halo_rounds.py
scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh
scripts/cluster/scaling_levante/mpas_nsys_skew_capture.sbatch
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch
scripts/bench/analyze_nsys_rank_skew.py-24-pair kernels whose [start, end] intervals overlap -- does not work here and
scripts/bench/analyze_nsys_rank_skew.py-25-its own control proves it: all 64 ranks issue their round-r collective at
scripts/bench/analyze_nsys_rank_skew.py-26-about the same time, so rank 0's kernel overlaps a NON-partner's just as
scripts/bench/analyze_nsys_rank_skew.py-27-readily as its true partner's.  Overlap is necessary, never sufficient.
scripts/bench/analyze_nsys_rank_skew.py-28-
scripts/bench/analyze_nsys_rank_skew.py-29-This uses the SCHEDULE as ground truth instead, which makes the pairing
scripts/bench/analyze_nsys_rank_skew.py:30:exact.  ``spmd_schedule_cost(..., round_profile_for_device=d)`` gives the
scripts/bench/analyze_nsys_rank_skew.py-31-rounds device ``d`` participates in, IN THE ORDER IT ISSUES THEM.  If A's
scripts/bench/analyze_nsys_rank_skew.py-32-profile shows it exchanges with B at global round r, and B's profile shows
scripts/bench/analyze_nsys_rank_skew.py-33-the mirror-image entry at the same r, then A's k-th halo fill issues that
scripts/bench/analyze_nsys_rank_skew.py-34-collective at position ``k*len(A_profile) + index_of_r_in_A`` and B's at
scripts/bench/analyze_nsys_rank_skew.py-35-``k*len(B_profile) + index_of_r_in_B``.  Same collective, by construction.
scripts/bench/analyze_nsys_rank_skew.py-36-
--
scripts/bench/analyze_nsys_rank_skew.py-235-                               lloyd_iterations=args.lloyd)
scripts/bench/analyze_nsys_rank_skew.py-236-    profiles, kernels = {}, {}
scripts/bench/analyze_nsys_rank_skew.py-237-    for rank in sorted({r for pr in pairs for r in pr}):
scripts/bench/analyze_nsys_rank_skew.py-238-        cost = spmd_schedule_cost(mesh, args.n_devices,
scripts/bench/analyze_nsys_rank_skew.py-239-                                  method=args.partition_method,
scripts/bench/analyze_nsys_rank_skew.py-240-                                  reorder_target=args.reorder_target,
scripts/bench/analyze_nsys_rank_skew.py:241:                                  round_profile_for_device=rank)
scripts/bench/analyze_nsys_rank_skew.py:242:        profiles[rank] = cost["round_profile"]
scripts/bench/analyze_nsys_rank_skew.py-243-        kernels[rank] = load_kernels(args.sqlite_dir / f"rank_{rank}.sqlite")
scripts/bench/analyze_nsys_rank_skew.py-244-        print(f"[rank {rank}] node={nodes[rank]} "
scripts/bench/analyze_nsys_rank_skew.py-245-              f"{len(kernels[rank])} collectives, "
scripts/bench/analyze_nsys_rank_skew.py-246-              f"{len(profiles[rank])} rounds/fill", flush=True)
scripts/bench/analyze_nsys_rank_skew.py-247-
scripts/bench/analyze_nsys_rank_skew.py-248-    out_pairs = {}
--
scripts/bench/analyze_nsys_halo_rounds.py-19-It reports two things from an EXISTING trace, with no GPU job:
scripts/bench/analyze_nsys_halo_rounds.py-20-
scripts/bench/analyze_nsys_halo_rounds.py-21-1. **Reproducibility.** Durations grouped by position in the repeating
scripts/bench/analyze_nsys_halo_rounds.py-22-   schedule, as within-position CV against the overall CV.
scripts/bench/analyze_nsys_halo_rounds.py-23-2. **Payload correlation.** Each position paired with the payload the
scripts/bench/analyze_nsys_halo_rounds.py-24-   production schedule says that round moves (``spmd_schedule_cost(...,
scripts/bench/analyze_nsys_halo_rounds.py:25:   round_profile_for_device=)``), as a Pearson r.
scripts/bench/analyze_nsys_halo_rounds.py-26-
scripts/bench/analyze_nsys_halo_rounds.py-27-Neither SETTLES payload-vs-wait, and an earlier version of this docstring
scripts/bench/analyze_nsys_halo_rounds.py-28-wrongly claimed both did:
scripts/bench/analyze_nsys_halo_rounds.py-29-
scripts/bench/analyze_nsys_halo_rounds.py-30-* Low within-position CV refutes only RANDOM jitter.  A consistently late
scripts/bench/analyze_nsys_halo_rounds.py-31-  peer, repeatable stream serialisation, or a fixed topology path produces
--
scripts/bench/analyze_nsys_halo_rounds.py-49-not to award a mechanism.
scripts/bench/analyze_nsys_halo_rounds.py-50-
scripts/bench/analyze_nsys_halo_rounds.py-51-ALIGNMENT, the thing that silently corrupts this comparison: an nsys capture
scripts/bench/analyze_nsys_halo_rounds.py-52-is per RANK, and a rank issues a collective only in the colour classes that
scripts/bench/analyze_nsys_halo_rounds.py-53-touch it.  At s9/np64 rank 0 shows 8 ``SendRecv`` per halo fill while the
scripts/bench/analyze_nsys_halo_rounds.py-54-graph's ``max_degree`` is 11.  So the k-th observed call is the k-th round
scripts/bench/analyze_nsys_halo_rounds.py:55:CONTAINING THAT DEVICE.  ``round_profile_for_device`` returns exactly that
scripts/bench/analyze_nsys_halo_rounds.py-56-subset, in order.  The check that this is the RIGHT subset is that the
scripts/bench/analyze_nsys_halo_rounds.py-57-detected period divides the trace length; when it does not, the schedule and
scripts/bench/analyze_nsys_halo_rounds.py-58-the trace describe different runs and the script REFUSES rather than pairing
scripts/bench/analyze_nsys_halo_rounds.py-59-rounds with unrelated collectives.
scripts/bench/analyze_nsys_halo_rounds.py-60-
scripts/bench/analyze_nsys_halo_rounds.py-61-Run (no GPU needed; the mesh build dominates at subdiv>=9):
--
scripts/bench/analyze_nsys_halo_rounds.py-266-          flush=True)
scripts/bench/analyze_nsys_halo_rounds.py-267-    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
scripts/bench/analyze_nsys_halo_rounds.py-268-                               lloyd_iterations=args.lloyd)
scripts/bench/analyze_nsys_halo_rounds.py-269-    cost = spmd_schedule_cost(mesh, args.n_devices,
scripts/bench/analyze_nsys_halo_rounds.py-270-                              method=args.partition_method,
scripts/bench/analyze_nsys_halo_rounds.py-271-                              reorder_target=args.reorder_target,
scripts/bench/analyze_nsys_halo_rounds.py:272:                              round_profile_for_device=args.device)
scripts/bench/analyze_nsys_halo_rounds.py:273:    profile = cost["round_profile"]
scripts/bench/analyze_nsys_halo_rounds.py-274-    print(f"[sched] n_rounds={cost['n_rounds']} "
scripts/bench/analyze_nsys_halo_rounds.py-275-          f"max_degree={cost['max_degree']} "
scripts/bench/analyze_nsys_halo_rounds.py-276-          f"device {args.device} participates in {len(profile)}", flush=True)
scripts/bench/analyze_nsys_halo_rounds.py-277-    if not profile:
scripts/bench/analyze_nsys_halo_rounds.py-278-        raise SystemExit(
scripts/bench/analyze_nsys_halo_rounds.py-279-            f"device {args.device} participates in no round; nothing to pair.")
--
tests/bench/test_analyze_nsys_rank_skew.py-20-_spec = importlib.util.spec_from_file_location("nsys_rank_skew", _SCRIPT)
tests/bench/test_analyze_nsys_rank_skew.py-21-mod = importlib.util.module_from_spec(_spec)
tests/bench/test_analyze_nsys_rank_skew.py-22-_spec.loader.exec_module(mod)
tests/bench/test_analyze_nsys_rank_skew.py-23-
tests/bench/test_analyze_nsys_rank_skew.py-24-
tests/bench/test_analyze_nsys_rank_skew.py-25-def _profile(*entries):
tests/bench/test_analyze_nsys_rank_skew.py:26:    """entries: (round, partner) -> the round_profile shape."""
tests/bench/test_analyze_nsys_rank_skew.py-27-    return [{"round": r, "partner": p, "halo_cells": 10, "halo_edges": 20}
tests/bench/test_analyze_nsys_rank_skew.py-28-            for r, p in entries]
tests/bench/test_analyze_nsys_rank_skew.py-29-
tests/bench/test_analyze_nsys_rank_skew.py-30-
tests/bench/test_analyze_nsys_rank_skew.py-31-# --- shared_round: the pairing and its refusals ---------------------------
tests/bench/test_analyze_nsys_rank_skew.py-32-
--
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch-89-print(f"[attr] trace holds {n_calls} SendRecv calls", flush=True)
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch-90-
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch-91-mesh = create_voronoi_mesh(subdivision_level=9, lloyd_iterations=0)
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch-92-found = {}
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch-93-for target in (64, 128):
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch-94-    cost = spmd_schedule_cost(mesh, 64, method="sfc", reorder_target=target,
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch:95:                              round_profile_for_device=0)
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch:96:    dev_rounds = len(cost["round_profile"])
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch-97-    # A candidate is viable only if some whole number of halo fills per step
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch-98-    # makes the device's round count divide the trace exactly.
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch-99-    fills = [k for k in range(1, 9) if n_calls % (dev_rounds * k) == 0]
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch-100-    print(f"[attr] reorder_target={target}: n_rounds={cost['n_rounds']} "
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch-101-          f"max_degree={cost['max_degree']} device0_rounds={dev_rounds} "
scripts/cluster/scaling_levante/mpas_nsys_round_attribution.sbatch-102-          f"viable_fills_per_step={fills}", flush=True)
--
packages/core/legoesm/parallel/sharded_dynamics.py-1247-SPMD_HALO_DEPTH = 3
packages/core/legoesm/parallel/sharded_dynamics.py-1248-
packages/core/legoesm/parallel/sharded_dynamics.py-1249-
packages/core/legoesm/parallel/sharded_dynamics.py-1250-def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
packages/core/legoesm/parallel/sharded_dynamics.py-1251-                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
packages/core/legoesm/parallel/sharded_dynamics.py-1252-                       ppermute_cells_per_device_threshold=2_000,
packages/core/legoesm/parallel/sharded_dynamics.py:1253:                       round_profile_for_device=None):
packages/core/legoesm/parallel/sharded_dynamics.py-1254-    """How much halo communication one ownership choice costs, computed offline.
packages/core/legoesm/parallel/sharded_dynamics.py-1255-
packages/core/legoesm/parallel/sharded_dynamics.py-1256-    Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
packages/core/legoesm/parallel/sharded_dynamics.py-1257-    ROUNDS one halo exchange needs -- the sequential collective launches that
packages/core/legoesm/parallel/sharded_dynamics.py-1258-    dominate MPAS strong scaling above ~64 devices.  Runs on a laptop: no GPU,
packages/core/legoesm/parallel/sharded_dynamics.py-1259-    no MPI, no benchmark job, so a split can be compared before it costs an
--
packages/core/legoesm/parallel/sharded_dynamics.py-1308-    already_reordered : bool
packages/core/legoesm/parallel/sharded_dynamics.py-1309-    halo_depth : int
packages/core/legoesm/parallel/sharded_dynamics.py-1310-        Must match production (3) or the graph is a different graph.
packages/core/legoesm/parallel/sharded_dynamics.py-1311-    ppermute_cells_per_device_threshold : int
packages/core/legoesm/parallel/sharded_dynamics.py-1312-        Mirror of the production auto-select threshold, only used to report
packages/core/legoesm/parallel/sharded_dynamics.py-1313-        ``production_strategy``.
packages/core/legoesm/parallel/sharded_dynamics.py:1314:    round_profile_for_device : int | None
packages/core/legoesm/parallel/sharded_dynamics.py:1315:        When set to a device id, also return ``round_profile``: the per-round
packages/core/legoesm/parallel/sharded_dynamics.py-1316-        payload THAT DEVICE exchanges, in schedule order, restricted to the
packages/core/legoesm/parallel/sharded_dynamics.py-1317-        rounds it actually participates in.
packages/core/legoesm/parallel/sharded_dynamics.py-1318-
packages/core/legoesm/parallel/sharded_dynamics.py-1319-        This exists to make a profiler trace interpretable.  An ``nsys``
packages/core/legoesm/parallel/sharded_dynamics.py-1320-        capture is per RANK, and a rank appears only in the colour classes
packages/core/legoesm/parallel/sharded_dynamics.py-1321-        that touch it -- at s9/np64 rank 0 shows 8 ``SendRecv`` per halo fill
--
packages/core/legoesm/parallel/sharded_dynamics.py-1417-            ("allgather" if cells_per < ppermute_cells_per_device_threshold
packages/core/legoesm/parallel/sharded_dynamics.py-1418-             else "ppermute")),
packages/core/legoesm/parallel/sharded_dynamics.py-1419-        "cells_per_device": cells_per,
packages/core/legoesm/parallel/sharded_dynamics.py-1420-        "max_local_cells": int(max_lc),
packages/core/legoesm/parallel/sharded_dynamics.py-1421-        "max_local_edges": int(max_le),
packages/core/legoesm/parallel/sharded_dynamics.py-1422-        **(
packages/core/legoesm/parallel/sharded_dynamics.py:1423:            {} if round_profile_for_device is None else
packages/core/legoesm/parallel/sharded_dynamics.py:1424:            {"round_profile": _round_profile(sched, round_profile_for_device,
packages/core/legoesm/parallel/sharded_dynamics.py-1425-                                             n_dev)}
packages/core/legoesm/parallel/sharded_dynamics.py-1426-        ),
packages/core/legoesm/parallel/sharded_dynamics.py-1427-    }
packages/core/legoesm/parallel/sharded_dynamics.py-1428-
packages/core/legoesm/parallel/sharded_dynamics.py-1429-
packages/core/legoesm/parallel/sharded_dynamics.py:1430:def _round_profile(sched, device, n_dev):
packages/core/legoesm/parallel/sharded_dynamics.py-1431-    """Per-round payload for ONE device, in the order it observes them.
packages/core/legoesm/parallel/sharded_dynamics.py-1432-
packages/core/legoesm/parallel/sharded_dynamics.py:1433:    See ``spmd_schedule_cost``'s ``round_profile_for_device``.  Only rounds
packages/core/legoesm/parallel/sharded_dynamics.py-1434-    whose colour class touches *device* are returned, because those are the
packages/core/legoesm/parallel/sharded_dynamics.py-1435-    only ones on which it issues a collective.
packages/core/legoesm/parallel/sharded_dynamics.py-1436-    """
packages/core/legoesm/parallel/sharded_dynamics.py-1437-    # Strict, like the n_dev check: int() would coerce 0.9 to 0 and silently
packages/core/legoesm/parallel/sharded_dynamics.py-1438-    # profile a different device than the caller named.
packages/core/legoesm/parallel/sharded_dynamics.py-1439-    if int(device) != device or not 0 <= device < n_dev:
packages/core/legoesm/parallel/sharded_dynamics.py-1440-        raise ValueError(
packages/core/legoesm/parallel/sharded_dynamics.py:1441:            f"round_profile_for_device must be an integer in [0, {n_dev}), "
packages/core/legoesm/parallel/sharded_dynamics.py-1442-            f"got {device!r}")
packages/core/legoesm/parallel/sharded_dynamics.py-1443-    device = int(device)
packages/core/legoesm/parallel/sharded_dynamics.py-1444-    out = []
packages/core/legoesm/parallel/sharded_dynamics.py-1445-    for r, perm in enumerate(sched["ppermute_perms"]):
packages/core/legoesm/parallel/sharded_dynamics.py-1446-        partner = next((dst for src, dst in perm if src == device), None)
packages/core/legoesm/parallel/sharded_dynamics.py-1447-        if partner is None:

codex
The source loop is already a concern: it calls `lax.ppermute` for every global colour on every rank, so I’m checking whether the compiled GPU path elides non-participating ranks before accepting the profile-stride premise.
exec
/usr/bin/bash -lc "PYTHONPATH=packages/core XLA_FLAGS='--xla_force_host_platform_device_count=4' .venv/bin/python - <<'PY'
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
mesh=Mesh(np.array(jax.devices()),('device',))
@jax.jit
def f(x):
  @shard_map(mesh=mesh,in_specs=P('device'),out_specs=P('device'),check_vma=False)
  def g(y):
    z=jax.lax.ppermute(y,'device',[(0,1),(1,0)])
    return z
  return g(x)
x=jnp.arange(4)
print(f.lower(x).compiler_ir(dialect='hlo').as_hlo_text())
PY" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 1 in 132ms:
/usr/bin/bash: cannot create temp file for here-document: Read-only file system

exec
/usr/bin/bash -lc "PYTHONPATH=packages/core XLA_FLAGS='--xla_force_host_platform_device_count=4' .venv/bin/python -c \"import jax, jax.numpy as jnp, numpy as np; from jax.experimental.shard_map import shard_map; from jax.sharding import Mesh, PartitionSpec as P; mesh=Mesh(np.array(jax.devices()),('device',)); f=jax.jit(lambda x: shard_map(lambda y: jax.lax.ppermute(y,'device',[(0,1),(1,0)]),mesh=mesh,in_specs=P('device'),out_specs=P('device'),check_vma=False)(x)); print(f.lower(jnp.arange(4)).compiler_ir(dialect='hlo').as_hlo_text())\"" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 exited 1 in 7960ms:
<string>:1: DeprecationWarning: jax.experimental.shard_map is deprecated in v0.8.0. Used jax.shard_map instead.
Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
Traceback (most recent call last):
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax/_src/xla_bridge.py", line 487, in discover_pjrt_plugins
    plugin_module.initialize()
    ~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax_plugins/xla_cuda12/__init__.py", line 370, in initialize
    _check_cuda_versions(raise_on_first_error = True)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/lib/python3.14/site-packages/jax_plugins/xla_cuda12/__init__.py", line 274, in _check_cuda_versions
    for d in range(cuda_versions.cuda_device_count())
                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
RuntimeError: jaxlib/cuda/versions_helpers.cc:135: operation cuInit(0) failed: Unknown CUDA error 303; cuGetErrorName failed. This probably means that JAX was unable to load the CUDA libraries.
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import jax, jax.numpy as jnp, numpy as np; from jax.experimental.shard_map import shard_map; from jax.sharding import Mesh, PartitionSpec as P; mesh=Mesh(np.array(jax.devices()),('device',)); f=jax.jit(lambda x: shard_map(lambda y: jax.lax.ppermute(y,'device',[(0,1),(1,0)]),mesh=mesh,in_specs=P('device'),out_specs=P('device'),check_vma=False)(x)); print(f.lower(jnp.arange(4)).compiler_ir(dialect='hlo').as_hlo_text())
                                                                                                                                                                                                                                                                                                                                                                        ~~~~~~~^^^^^^^^^^^^^^^
  File "<string>", line 1, in <lambda>
    import jax, jax.numpy as jnp, numpy as np; from jax.experimental.shard_map import shard_map; from jax.sharding import Mesh, PartitionSpec as P; mesh=Mesh(np.array(jax.devices()),('device',)); f=jax.jit(lambda x: shard_map(lambda y: jax.lax.ppermute(y,'device',[(0,1),(1,0)]),mesh=mesh,in_specs=P('device'),out_specs=P('device'),check_vma=False)(x)); print(f.lower(jnp.arange(4)).compiler_ir(dialect='hlo').as_hlo_text())
                                                                                                                                                                                                                        ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: shard_map() got an unexpected keyword argument 'check_vma'. Did you mean 'check_rep'?
--------------------
For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

exec
/usr/bin/bash -lc "PYTHONPATH=packages/core JAX_PLATFORMS=cpu XLA_FLAGS='--xla_force_host_platform_device_count=4' .venv/bin/python -c \"import jax, jax.numpy as jnp, numpy as np; from jax.experimental.shard_map import shard_map; from jax.sharding import Mesh, PartitionSpec as P; mesh=Mesh(np.array(jax.devices()),('device',)); f=jax.jit(lambda x: shard_map(lambda y: jax.lax.ppermute(y,'device',[(0,1),(1,0)]),mesh=mesh,in_specs=P('device'),out_specs=P('device'),check_rep=False)(x)); print(f.lower(jnp.arange(4)).compiler_ir(dialect='hlo').as_hlo_text())\"" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 2135ms:
<string>:1: DeprecationWarning: jax.experimental.shard_map is deprecated in v0.8.0. Used jax.shard_map instead.
HloModule jit__lambda, entry_computation_layout={(s32[4]{0})->s32[4]{0}}, frontend_attributes={xla.sdy.meshes={mesh = #sdy.mesh<["device"=4]>}}

xla.sdy.manual_computation_body.1 {
  shard_map.2 = s32[1]{0} parameter(0)
  ROOT ppermute.1 = s32[1]{0} collective-permute(shard_map.2), channel_id=1, source_target_pairs={{0,1},{1,0}}
}

ENTRY main.2 {
  x.1 = s32[4]{0} parameter(0)
  shard_map.5 = s32[1]{0} custom-call(x.1), custom_call_target="xla.sdy.GlobalToLocalShape", custom_call_has_side_effect=true, frontend_attributes={xla.sdy.in_shardings="#sdy.sharding_per_value<[<@mesh, [{\"device\"}]>]>",xla.sdy.manual_axes="#sdy<manual_axes{\"device\"}>"}
  shard_map.6 = s32[1]{0} call(shard_map.5), to_apply=xla.sdy.manual_computation_body.1, frontend_attributes={inlineable="false"}
  ROOT shard_map.7 = s32[4]{0} custom-call(shard_map.6), custom_call_target="xla.sdy.LocalToGlobalShape", custom_call_has_side_effect=true, frontend_attributes={xla.sdy.manual_axes="#sdy<manual_axes{\"device\"}>",xla.sdy.out_shardings="#sdy.sharding_per_value<[<@mesh, [{\"device\"}]>]>"}
}



exec
/usr/bin/bash -lc 'rg -n -i "collective.permute|CollectivePermute|sendrecv" .venv/lib/python3.14/site-packages/jax* .venv/lib/python3.14/site-packages/nvidia 2>/dev/null | head -300
rg -n "source_target_pairs|collective-permute" tests packages scripts docs -S | head -300' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 431ms:
tests/bench/test_scaling_metadata.py:370:        "  %a = collective-permute(%x)",          # optimized HLO
tests/bench/test_scaling_metadata.py:372:        "  %c = collective-permute-done(%a)",     # async companion -> excluded
tests/bench/test_scaling_metadata.py:397:    """The best-effort probe lowers a fn and counts its collective-permutes: a
tests/bench/test_scaling_metadata.py:418:        "  %a = collective-permute(%x)",             # permute (optimized)
tests/bench/test_scaling_metadata.py:420:        "  %c = collective-permute-done(%a)",        # async companion -> drop
tests/bench/test_scaling_metadata.py:452:        '  %done_mass = f32[] collective-permute(%q)',
scripts/bench/bench_mpas_spmd_scaling.py:418:    # HLO collective-permute census (#1113 ask 2): a STATIC compile property of
docs/performance/scaling/SCALING_STATUS_AUDIT.md:28:| **icosahedral / MPAS** | (a) graph-partition domain decomposition (METIS/RCB/SFC `auto`), validated vs serial ~1e-9 | (a) multi-GPU via route-A mpi4jax (opaque to XLA overlap); (c) route-B multi-controller NCCL SPMD wired since M3c #981 (native ppermute step, `bench_mpas_spmd_scaling --multicontroller`, cluster lane E; XLA collective-permute combining defaulted for the lane, #1113) — no production-scale receipts yet | Derecho 2026-07-02: CPU-MPI to 128 ranks f32 ~75× (eff ~0.59); GPU 1→16 A100 @28 km ~6× (eff ~0.38), coarse grids flat (per-device floor, not a defect) | route-A mpi4jax leg is latency-bound, no comm/compute overlap; route-B ppermute round count (edge-coloring rounds × launch latency, #1113) | production-size lane-E runs; re-measure 8→16 GPU leg on native ppermute |
tests/bench/test_bench_cube_tiled_step_scaling.py:59:        "%x = collective-permute(...)",
tests/bench/test_bench_cube_tiled_step_scaling.py:60:        "%y = collective-permute-start(...)",
tests/bench/test_bench_cube_tiled_step_scaling.py:61:        "%z = collective-permute-done(...)",  # not counted (done)
docs/performance/scaling/levante_campaign_2026-07-24.md:112:  execute): collective-permutes per step are 3 / 9 / 21 at np 2/4/8, i.e.
docs/performance/scaling/levante_campaign_2026-07-24.md:360:collective-permutes vs the censused 2.25 MB — a 7.2x undercount**. But
docs/performance/scaling/levante_campaign_2026-07-24.md:1802:  `make_sharded_atm_latlon_step`): **41 collective-permutes + 1
docs/performance/scaling/levante_campaign_2026-07-24.md:2347:  128-GPU collective trace: **206 -> 133 collective-permutes/step**
docs/performance/scaling/scaling_indicators.csv:24:2026-06-14,7d5c5188,tiled_np24,atm_cube,spmd,capability,24,tiled_np_validated,8482583,full fv3_sw_tendencies sub-face tiled np24=6x2x2 bit-EXACT on 2 nodes (200 cross-proc collective-permute) — np54 host-gate; np>6 future-HW (anti-scales Gloo-TCP so capability not speedup)
docs/performance/scaling/spmd_message_census_2026-07-08.md:29:n_cp = hlo.count("collective-permute-start") + hlo.count("collective-permute(")
docs/performance/scaling/spmd_message_census_2026-07-08.md:38:| devices | collective-permutes / step | all-reduce / step |
docs/performance/scaling/spmd_message_census_2026-07-08.md:66:| arm | collective-permutes / step |
scripts/bench/roofline_probe.py:925:    collective ops in the step).  We count each sync collective-permute
scripts/bench/bench_ocean_latlon_spmd_scaling.py:313:                        "measured 25%% fewer static collective-permutes on "
scripts/bench/bench_cube_tiled_step_scaling.py:17:  * compiled-HLO census: the step must contain collective-permutes and NO
scripts/bench/bench_cube_tiled_step_scaling.py:258:            "compiled tiled step contains NO collective-permutes — the "
scripts/bench/bench_cube_tiled_step_scaling.py:296:        # #921: the closed-loop step fuses the halo collective-permutes with
scripts/bench/run_levante_gpu_scaling.py:326:    # Collective-permute op census of the compiled TIMED executable
scripts/bench/run_levante_gpu_scaling.py:331:    # ``collective-permute``; async pairs as ``-start``/``-done``.
scripts/bench/run_levante_gpu_scaling.py:616:    """Census of collective-permute ops in a compiled HLO module.
scripts/bench/run_levante_gpu_scaling.py:620:    halo exchanges lower to ``collective-permute``; the async form
scripts/bench/run_levante_gpu_scaling.py:621:    lowers to ``collective-permute-start`` / ``collective-permute-done``
scripts/bench/run_levante_gpu_scaling.py:627:        "collective-permute": len(
scripts/bench/run_levante_gpu_scaling.py:628:            re.findall(r"\bcollective-permute\(", hlo_text)),
scripts/bench/run_levante_gpu_scaling.py:629:        "collective-permute-start": len(
scripts/bench/run_levante_gpu_scaling.py:630:            re.findall(r"\bcollective-permute-start\(", hlo_text)),
scripts/bench/run_levante_gpu_scaling.py:631:        "collective-permute-done": len(
scripts/bench/run_levante_gpu_scaling.py:632:            re.findall(r"\bcollective-permute-done\(", hlo_text)),
scripts/bench/run_levante_gpu_scaling.py:637:    """``TimingResult`` kwargs for the collective-permute census.
scripts/bench/run_levante_gpu_scaling.py:645:        "hlo_collective_permute": hlo_counts["collective-permute"],
scripts/bench/run_levante_gpu_scaling.py:646:        "hlo_collective_permute_start": hlo_counts["collective-permute-start"],
scripts/bench/run_levante_gpu_scaling.py:647:        "hlo_collective_permute_done": hlo_counts["collective-permute-done"],
scripts/bench/run_levante_gpu_scaling.py:684:       a loud warning), plus the collective-permute op census (printed
scripts/bench/run_levante_gpu_scaling.py:806:                f"    HLO census: {hlo_counts['collective-permute']} "
scripts/bench/run_levante_gpu_scaling.py:807:                f"collective-permute, "
scripts/bench/run_levante_gpu_scaling.py:808:                f"{hlo_counts['collective-permute-start']} -start, "
scripts/bench/run_levante_gpu_scaling.py:809:                f"{hlo_counts['collective-permute-done']} -done op(s) "
scripts/bench/metadata.py:103:# Match the OP-CALL form ``collective-permute(`` / ``collective_permute(`` /
scripts/bench/metadata.py:140:    """Best-effort: count the collective-permutes in the COMPILED HLO of
scripts/bench/metadata.py:145:    count, because XLA collective-permute combining / pipelined-p2p
scripts/bench/metadata.py:172:#: Every collective OP family a scaling row can run.  ``collective-permute`` is
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:20:| Atm cube strong (≤6 GPU) | **collective-permute COUNT** grows with shard count while per-msg NCCL p2p latency (~30–80 µs) doesn't amortise on small tiles → anti-scales | census: 12 CP @2dev, 41 @6dev (C24/L8, below); f64≡f32 curves ⇒ latency-bound | CONFIRMED |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:50:| devices | collective-permute / step | all-reduce / step |
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:67:1. **The message census counted only collective-permutes.** The ocean PCG
docs/performance/scaling/bottleneck_diagnosis_2026-07-20.md:101:   measured winner (+8.5 %); the XLA collective-permute-combine + pipelined-p2p
docs/performance/scaling/cube_production_tiling_design.md:140:  in-stage ppermutes -> cross-NODE collective-permute); each process
scripts/bench/run_scaling_diagnosis.py:751:                      f"{c['collective_permute']} collective-permute, "
scripts/bench/bench_ocean_latlon_spmd_pcg.py:16:ideal 1.0.  On a PCIe pair (no NVLink) the psum collective-permute is the
scripts/cluster/scaling_derecho/README.md:488:| `diagnosis.pbs` | BOTTLENECK diagnosis via `scripts/bench/run_scaling_diagnosis.py` — per-phase halo bandwidth, reduction latency, roofline, compute/comm overlap, and the **static collective census** (collective-permute + all-reduce + all-gather per step) that the throughput jobs above do NOT capture. Climbs the cube face-shard `1 2 3` ladder (must divide 6) so the message-count-vs-shard curve — the LATENCY-bound anti-scaling signal — is recorded. `qsub -v MODE=census` for counts only. |
scripts/cluster/scaling_derecho/README.md:525:(census: cube 46 collective-permutes/step at 6 devices with field packing
scripts/cluster/scaling_levante/README.md:77:| `diagnosis.sbatch` | BOTTLENECK diagnosis via `scripts/bench/run_scaling_diagnosis.py` — per-phase halo bandwidth, reduction latency, roofline, compute/comm overlap, and the **static collective census** (collective-permute + all-reduce + all-gather per step) the throughput jobs do NOT capture. Climbs the cube face-shard `1 2 3` ladder so the message-count-vs-shard curve — the LATENCY-bound anti-scaling signal — is recorded. `sbatch --export=ALL,MODE=census` for counts only. Derecho twin: `scaling_derecho/diagnosis.pbs`. |
scripts/cluster/scaling_derecho/diagnosis.pbs:15:# STATIC collective census: collective-permute + all-reduce + all-gather per
scripts/cluster/scaling_derecho/diagnosis.pbs:26:#   * Cube face-shard anti-scales at small tiles because the collective-permute
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:41:#     XLA collective-permute combining + pipelined p2p / PGLE) on the latlon
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:89:# XLA collective-permute combining + pipelined p2p. #1113 found the route-B MPAS
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:93:# sizes). These flags fuse adjacent collective-permutes and pipeline the p2p,
scripts/cluster/scaling_derecho/gpu_multinode_scaling.pbs:289:#   xla   : collective-permute combining + pipelined p2p — the cube/latlon
scripts/cluster/scaling_derecho/cube_tiled_step.pbs:19:#   full-cube all-gather (replicated execution) or zero collective-permutes.
scripts/cluster/scaling_derecho/cube_tiled_step.pbs:80:# collective-permutes (cubesphere_exchange ppermute over the SAME axes). Under
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:228:# trace receipt atm 41->29 CPs/step), xla (collective-permute combining +
scripts/cluster/scaling_levante/diagnosis.sbatch:17:# STATIC collective census: collective-permute + all-reduce + all-gather per
packages/core/legoesm/parallel/cubesphere_exchange.py:1848:            f"ppermute multiface exchange (collective-permute only).  "
packages/core/legoesm/parallel/voronoi_partition.py:1059:    # high device counts is the number of collective-permute ROUNDS -- equal
packages/core/legoesm/parallel/voronoi_partition.py:1074:    # step, auto->metis would cost +15 collective-permutes/step at 128 on
packages/core/legoesm/parallel/tiled_production_cdgrid.py:2452:    the halo collective-permutes (``jax.lax.ppermute`` over the
packages/core/legoesm/parallel/sharded_dynamics.py:735:    # collective-permute rounds, producing much better XLA communication
packages/core/legoesm/grids/halo_latlon.py:773:    # 41 -> 29 collective-permutes/step at IDENTICAL bytes.
packages/atmosphere/legoesm/atmosphere/dynamics/gcm/tiled_step_adapter.py:444:    body's in-stage tile-halo ``collective-permute``s and, with
tests/parallel/test_latlon_spmd_fused_halo.py:11:3. The point, mechanically: compiled ``collective-permute`` count drops
tests/parallel/test_latlon_spmd_fused_halo.py:19:   ``reconstruct_vface_lower`` + the compiled collective-permute drop.
tests/parallel/test_latlon_spmd_fused_halo.py:135:               if "collective-permute" in line and "done" not in line)
tests/parallel/test_latlon_spmd_fused_halo.py:231:    collective-permutes, fused = 1."""
tests/parallel/test_latlon_spmd_fused_halo.py:316:    collective-permutes with fusion on (the aggregation actually fires on
tests/parallel/test_ppermute_multiface.py:331:    """The compiled multiface h1+h2 programs contain collective-permute
tests/parallel/test_ppermute_multiface.py:342:        assert "collective-permute" in txt, f"halo={halo}: ppermute missing"
tests/parallel/test_ppermute_multiface.py:444:    collective-permute."""
tests/parallel/test_ppermute_multiface.py:454:    assert "collective-permute" in txt
tests/parallel/test_ppermute_multiface.py:496:    # Benign: collective-permute is the expected op.
tests/parallel/test_ppermute_multiface.py:497:    assert not flag("%cp = f32[24,8]{1,0} collective-permute(f32[24,8] %s)")
tests/parallel/test_ppermute_multiface.py:546:    assert "collective-permute" not in txt
tests/parallel/test_cube_tile_native_segment.py:18:  executable; ``collective-permute`` (tile halos) present; collective
tests/parallel/test_cube_tile_native_segment.py:319:    * ``collective-permute`` present (the in-stage tile halos really run),
tests/parallel/test_cube_tile_native_segment.py:346:    assert "collective-permute" in hlo5, "tile halos missing from segment"
tests/parallel/test_cube_tile_native_segment.py:349:    for op in ("collective-permute", "all-reduce", "all-gather",
tests/parallel/test_segment_sharding_device_config.py:448:            "%collective-permute.5 = f32[3,24,24]{2,1,0}"
tests/parallel/test_segment_sharding_device_config.py:449:            " collective-permute(f32[3,24,24]{2,1,0} %p),"
tests/parallel/test_segment_sharding_device_config.py:450:            " source_target_pairs={{0,1},{1,0}}",
tests/parallel/test_segment_sharding_device_config.py:451:            "%add.1 = f32[] add(f32[] %collective-permute.5, f32[] %c)",
tests/parallel/test_segment_sharding_device_config.py:454:            "%cps = (f32[3],f32[3]) collective-permute-start(f32[3] %q)",
tests/parallel/test_segment_sharding_device_config.py:455:            "%cpd = f32[3] collective-permute-done((f32[3],f32[3]) %cps)",
tests/parallel/test_segment_sharding_device_config.py:456:            "%cps2 = (f32[3],f32[3]) collective-permute-start(f32[3] %r)",
tests/parallel/test_segment_sharding_device_config.py:457:            "%cpd2 = f32[3] collective-permute-done((f32[3],f32[3]) %cps2)",
tests/parallel/test_segment_sharding_device_config.py:460:        assert counts["collective-permute"] == 1
tests/parallel/test_segment_sharding_device_config.py:461:        assert counts["collective-permute-start"] == 2
tests/parallel/test_segment_sharding_device_config.py:462:        assert counts["collective-permute-done"] == 2
tests/parallel/test_segment_sharding_device_config.py:473:            "collective-permute": 0,
tests/parallel/test_segment_sharding_device_config.py:474:            "collective-permute-start": 0,
tests/parallel/test_segment_sharding_device_config.py:475:            "collective-permute-done": 0,
tests/parallel/test_warmup_tiled_cube_comms.py:4:collective-permute cliques AND the mass-fixer ``psum`` all-reduce clique over
tests/parallel/test_warmup_tiled_cube_comms.py:70:    """The halo executable issues ONLY collective-permutes; the reduce
tests/parallel/test_warmup_tiled_cube_comms.py:100:                   if "collective-permute" in ln and "done" not in ln)
tests/parallel/test_ppermute_halo_exchange.py:91:    """The HLO must actually contain collective-permute for ppermute and
tests/parallel/test_ppermute_halo_exchange.py:100:    assert "collective-permute" in hlo_pp and "all-gather" not in hlo_pp
tests/parallel/test_ppermute_halo_exchange.py:128:    actually emit collective-permute, not the retired all_gather
tests/parallel/test_ppermute_halo_exchange.py:143:    assert "collective-permute" in hlo and "all-gather" not in hlo
packages/coupler/legoesm/driver/model_driver.py:9524:        # collective-permutes + the target-mass / moisture-fixer psums) in a
packages/coupler/legoesm/driver/model_driver.py:9786:        # collective-permutes + the in-stage mass-fixer psum) in a fixed,
tests/parallel/test_atm_latlon_2d_tiling.py:226:    the antipodal partner ppermute — collective-permute in HLO) while
tests/parallel/test_atm_latlon_2d_tiling.py:239:    assert "collective-permute" in hlo
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:11:step's WHILE-BODY ``collective-permute`` count must DROP vs the standard
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:132:    """Collective-permute ops inside while-LOOP BODY computations.
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:142:    to the closing ``}``), counts collective-permutes per block, and sums
tests/parallel/test_latlon_ocean_spmd_wide_halo.py:159:            elif "collective-permute" in line and "done" not in line:
tests/unit/test_voronoi_partition_method.py:249:    high device counts is the collective-permute ROUND count (= max degree of
tests/unit/test_voronoi_partition_method.py:281:        f"use SFC — METIS costs +15 collective-permutes/step at 128 devices "
scripts/validate/validate_tiled_fv3_sw_multinode.py:8:collective-permute, not the single-process ``XLA_FLAGS=--xla_force_host_platform_
scripts/validate/validate_tiled_fv3_sw_multinode.py:52:    # ppermutes -> cross-process collective-permute).  A single-process run with
scripts/validate/validate_tiled_fv3_sw_multinode.py:133:            cp = len(re.findall(r"\bcollective-permute\b", hlo))
scripts/validate/validate_tiled_fv3_sw_multinode.py:134:            log(f"HLO: collective-permute={cp} (cross-process halo exchange)")

exec
/usr/bin/bash -lc "nl -ba scripts/cluster/scaling_levante/_env.sh | sed -n '1,360p'
nl -ba scripts/bench/bench_mpas_spmd_scaling.py | sed -n '1,560p'
rg -n -C 5 \"xla_gpu_enable_pipelined_p2p|collective_permute|async_collective|async.*collective|NCCL|XLA_FLAGS\" scripts/cluster/scaling_levante packages/core scripts/bench -S | head -700" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 248ms:
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
    21	# The default is a GUESS at a per-user clone path. When it is wrong the job
    22	# does not fail here — it fails ~60 lines later with a bare
    23	# "cd: <path>: No such file or directory" plus "_chain_body.sh: No such file",
    24	# 7 seconds in, which reads like a broken launcher rather than an unset
    25	# variable (three U-Cast arms lost this way, 2026-08-01). Say it plainly.
    26	REPO="${LEGOESM_REPO:-/work/bd1083/$USER/legoESM}"
    27	if [ ! -d "$REPO" ]; then
    28	  echo "[_env.sh] REPO='$REPO' does not exist." >&2
    29	  if [ -z "${LEGOESM_REPO:-}" ]; then
    30	    echo "[_env.sh] LEGOESM_REPO is unset, so this is the per-user DEFAULT" >&2
    31	    echo "[_env.sh] guess, not a configured path. Submit with" >&2
    32	    echo "[_env.sh]   sbatch --export=ALL,LEGOESM_REPO=\$PWD,... " >&2
    33	    echo "[_env.sh] (--export=ALL alone does NOT carry it if your shell" >&2
    34	    echo "[_env.sh]  never exported it)." >&2
    35	  fi
    36	  exit 1
    37	fi
    38	export REPO
    39	
    40	# --- (3) Conda env with a CUDA jaxlib AND a CUDA-aware mpi4jax (see README) ---
    41	CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
    42	
    43	# --- Federation PYTHONPATH (belt-and-braces; `pip install -e .` makes it
    44	#     redundant but harmless) ------------------------------------------------
    45	PP="$REPO/src"
    46	for p in atmosphere core coupler ice land ml ocean tools; do
    47	  PP="$PP:$REPO/packages/$p"
    48	done
    49	export PYTHONPATH="$PP:${PYTHONPATH:-}"
    50	
    51	# --- Modules + conda -- EDIT the module versions to the Levante stack you built
    52	#     mpi4py / mpi4jax against (README Step 1); pinned versions matter because
    53	#     the runtime libmpi ABI must match the build ABI ------------------------
    54	module load python3 2>/dev/null || true      # EDIT: e.g. python3/2023.01-gcc-11.2.0
    55	module load openmpi 2>/dev/null || true       # EDIT: the CUDA-aware openmpi you built against
    56	module load cuda    2>/dev/null || true       # EDIT: matching cuda toolkit
    57	if command -v conda >/dev/null 2>&1; then
    58	  conda activate "$CONDA_ENV" 2>/dev/null || true
    59	fi
    60	# Prefer the repo's own uv venv when it exists — that is the interpreter every
    61	# dev/test workflow uses, and the bare `python` on a Levante compute node has
    62	# no jax (three U-Cast arms died at `import jax` inside 7 s, 2026-08-02).
    63	if [ -z "${LEGOESM_PYTHON:-}" ] && [ -x "$REPO/.venv/bin/python" ]; then
    64	  LEGOESM_PYTHON="$REPO/.venv/bin/python"
    65	fi
    66	PY="${LEGOESM_PYTHON:-$(command -v python)}"
    67	export PY
    68	# Fail at source time, not 4 GPU-hours in: the launcher's first real work is
    69	# `$PY scripts/run/run_aimip.py`, which imports jax immediately.
    70	if ! "$PY" -c "import jax" >/dev/null 2>&1; then
    71	  echo "[_env.sh] PY='$PY' cannot import jax." >&2
    72	  echo "[_env.sh] Set LEGOESM_PYTHON=<repo>/.venv/bin/python (uv venv) or" >&2
    73	  echo "[_env.sh] LEGOESM_CONDA_ENV=<env with a CUDA jaxlib>." >&2
    74	  exit 1
    75	fi
    76	
    77	# --- JAX / runtime knobs -----------------------------------------------------
    78	export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
    79	export MPI4JAX_NO_WARN_JAX_VERSION=1
    80	export MPLBACKEND="${MPLBACKEND:-Agg}"          # headless plotting
    81	# DKRZ scratch is /scratch/<first-letter-of-user>/<user>.
    82	export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
    83	# Persistent JIT cache reuses compiles across runs; set empty to force a cold
    84	# compile (true compile_time_s).  On SCRATCH so it survives between jobs.
    85	export LEGOESM_JIT_CACHE_DIR="${LEGOESM_JIT_CACHE_DIR:-$SCRATCH/legoesm_jit_cache}"
    86	
    87	# --- OpenMPI + UCX CUDA-aware fabric (GPU route-A) ---------------------------
    88	# Route-A hands the on-device sendrecv buffer straight to MPI (the whole point:
    89	# no device->host->device staging, which would erase multi-GPU scaling).  On
    90	# Levante that path is OpenMPI-over-UCX; the pml/osc + UCX transports below turn
    91	# on GPU-direct: cuda_copy + cuda_ipc intra-node, gdr_copy over InfiniBand HDR
    92	# inter-node.  Requires a CUDA-aware mpi4jax (README) + MPI4JAX_USE_CUDA_MPI=1
    93	# (set in the job script).  UCX_MEMTYPE_CACHE=n avoids a stale device/host
    94	# memtype-cache hang that CUDA-aware sendrecv is prone to.
    95	export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"
    96	export OMPI_MCA_osc="${OMPI_MCA_osc:-ucx}"
    97	export UCX_TLS="${UCX_TLS:-rc,cuda_copy,cuda_ipc,gdr_copy,sm,self}"
    98	export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
    99	export UCX_RNDV_SCHEME="${UCX_RNDV_SCHEME:-put_zcopy}"
   100	
   101	# --- NCCL over InfiniBand (route-B: jax.distributed multi-node lanes) --------
   102	# NCCL (shard_map/ppermute collectives under jax.distributed) uses its own
   103	# IB-verbs stack — independent of the UCX/MPI settings above; the two configs
   104	# coexist. Bootstrap ring runs over IPoIB: verify the interface name once with
   105	# `ip addr` on a gpu node (a wrong NCCL_SOCKET_IFNAME is the #1 cause of
   106	# multi-node NCCL bootstrap timeouts on IB clusters).
   107	export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
   108	export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
   109	# Prefix-match BOTH HCAs (mlx5_0/mlx5_1 — one per socket on Levante nodes).
   110	export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
   111	# GPUDirect RDMA when NIC and GPU share a NUMA/PCIe root.
   112	export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
   113	export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
   114	
   115	# --- XLA overlap defaults for the lat-lon SPMD lanes (2026-08-04) --------
   116	# Latency-hiding scheduler + pipelined p2p: -8.4% at LL2048@64 (job
   117	# 26677602) and -8.3% at @128 (26677668), A/A2 drift 0.3-0.4% — twice-
   118	# reproduced, parity suites green with flags on. MPAS lane: null (0.0%,
   119	# 26677669 — its edge-coloured schedule does not benefit; harmless).
   120	# Below the pre-registered 10% bar AND other lanes are unvalidated
   121	# (cube_tiled_step.sbatch force-disables latency hiding for a known
   122	# comm-init sensitivity; MPAS is null) — so this is strictly OPT-IN
   123	# (codex r23): set LEGOESM_XLA_OVERLAP=1 in validated lat-lon
   124	# launchers; never a shared default, and A/B control arms must keep
   125	# REPLACING XLA_FLAGS, not appending.
   126	if [ "${LEGOESM_XLA_OVERLAP:-0}" = 1 ]; then
   127	  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_p2p=true"
   128	fi
   129	
   130	export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
   131	mkdir -p "$TMPDIR" 2>/dev/null || true
   132	
   133	# #1361 memory preflight: target device whose HBM the benches gate against
   134	# (`--device-hbm`). Set in the SHARED env so the gate is on for every launcher
   135	# that sources this file — codex found the Derecho-only export left every
   136	# Levante bench ungated. Levante's GPU jobs request `--constraint=a100_80`.
   137	export LEGOESM_DEVICE_HBM="${LEGOESM_DEVICE_HBM:-a100-80}"
     1	"""Strong scaling bench for the device-sharded icosahedral/MPAS (TRiSK)
     2	hydrostatic atm step (``make_voronoi_sharded_step`` — cell-partition reorder +
     3	ppermute halo).
     4	
     5	The Voronoi twin of ``bench_atm_latlon_spmd_scaling.py`` (mirrored
     6	flag-for-flag where the grids allow): the global mesh is REORDERED with
     7	``reorder_voronoi_for_sharding`` (METIS/RCB/Hilbert-SFC cell partition, ghost-
     8	padded to an even device split) so each device's contiguous ``P("device")``
     9	shard is a spatially compact cell cluster, then the SSP-RK3 step exchanges
    10	only the partition-boundary halo per stage via ``jax.lax.ppermute``.
    11	
    12	  strong: fixed subdivision level, vary n_devices -> speedup = t(1)/t(n).
    13	  (Weak scaling rides the subdivision ladder: one level = 4x the cells, so
    14	  L at 4*n_dev matches L-1 at n_dev per-device load — there is no per-device
    15	  row knob like the lat-lon benches' --nlat-per-dev.)
    16	
    17	nlev caveat (#1113): the multi-node ceiling at THIN nlev is the ppermute ROUND
    18	count (``hlo_collective_permutes``, recorded per row) x the ~0.11 ms launch
    19	floor, NOT bandwidth — a fixed per-step overhead. At the ``--nlev 8`` default it
    20	dominates (~1.34 Gcells/s wall from N=4), so the default UNDERSTATES production
    21	scalability: at ``--nlev 26`` the per-cell compute grows ~3.25x, the flat wall
    22	dissolves (~2.07+ Gcells/s, ~1.9x higher at 16 GPUs), and a size-dependent term
    23	enters. Report the production curve at production thickness; nlev=8 is the
    24	overhead-mechanism receipt, not the campaign number.
    25	
    26	Device count is fixed at process start, so each n_devices runs as a SEPARATE
    27	process; this script benches ONE n_devices and appends a JSON line.
    28	JAX_PLATFORMS=cpu with --xla_force_host_platform_device_count gives virtual
    29	CPU devices (communication-overhead characterization, NOT a real speedup).
    30	
    31	Multi-controller (route-B, ``--multicontroller``): identical contract to the
    32	lat-lon benches — every process calls ``jax.distributed.initialize`` BEFORE
    33	any other JAX use, the ("device",) mesh is built over the GLOBAL
    34	``jax.devices()``, and the existing ``make_voronoi_sharded_step`` ppermute
    35	halo + the mass-fix psum run unchanged across processes (NCCL on GPU / gloo
    36	on CPU). NO mpi4jax is armed in this mode (the documented mixed-stack
    37	deadlock hazard). Every process computes the SAME reorder host-side; under
    38	``--multicontroller`` the partition checksum is asserted equal across
    39	processes (a rank-divergent partition — e.g. one rank resolving
    40	``--partition-method auto`` to METIS and another to RCB — would silently
    41	corrupt the halo schedule).
    42	
    43	Launch (cluster, one process per GPU):
    44	  srun -n 6 python bench_mpas_spmd_scaling.py --multicontroller \
    45	      --n-devices 6 ...            # SLURM: coordinator auto-detected
    46	  mpiexec -n 6 python ... --multicontroller --coordinator host0:9876
    47	CPU smoke (single process, virtual devices):
    48	  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
    49	  JAX_ENABLE_X64=1 python scripts/bench/bench_mpas_spmd_scaling.py \
    50	      --subdivision 3 --nlev 4 --n-devices 2 --steps 4 --parity-gate
    51	"""
    52	from __future__ import annotations
    53	
    54	import argparse
    55	import json
    56	import os
    57	import sys
    58	import time
    59	import zlib
    60	from pathlib import Path
    61	
    62	import jax
    63	import numpy as np
    64	
    65	# Repo root on the path for tests.test_cases.baroclinic_wave (the same
    66	# baroclinic-wave IC the icosahedral lanes of run_levante_gpu_scaling use).
    67	sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    68	# Bench dir for the shared metadata module (sibling-script import pattern).
    69	sys.path.insert(0, str(Path(__file__).resolve().parent))
    70	
    71	# Shared self-describing scaling metadata (anti-fake-scaling audit): merged
    72	# under rec["metadata"] so a virtual-CPU-device proxy, a gloo/TCP fabric run,
    73	# or an f32 ablation is falsifiable from the JSONL row alone.  metadata.py
    74	# imports JAX lazily, so this is safe before jax.distributed.initialize.
    75	from metadata import (  # noqa: E402
    76	    annotate_incomplete, hlo_collective_census, scaling_metadata,
    77	    tidy_throughput_fields)
    78	
    79	# SPMD full-step parity tolerances — the FLOATING-POINT RE-ASSOCIATION floor
    80	# of the sharded step (ppermute halo + mass-fix psum reduction-order change),
    81	# NOT a bug margin; a real halo/partition regression shows up orders of
    82	# magnitude above these.  Values extend the 1-step envelope of
    83	# tests/parallel/test_voronoi_sharded_equivalence.py (u/T atol 1e-6, p_s
    84	# atol 1e-1) to the smoke window; the floor grows with steps, hence the cap.
    85	MPAS_PARITY_TOLS = {  # precision -> field -> (rtol, atol)
    86	    "float64": {"u": (1.0e-5, 1.0e-5), "T": (1.0e-6, 1.0e-5),
    87	                "p_s": (1.0e-5, 1.0)},
    88	    "float32": {"u": (1.0e-3, 1.0e-3), "T": (1.0e-4, 1.0e-3),
    89	                "p_s": (1.0e-3, 50.0)},
    90	}
    91	MPAS_PARITY_MAX_STEPS = 8
    92	
    93	# Conservation gate default: with fix_mass=True the step restores the global
    94	# dry mass to the pre-step value each step, so the drift over a smoke window
    95	# is the allreduce rounding floor, not scheme drift.
    96	MASS_RTOL_DEFAULTS = {"float64": 1.0e-11, "float32": 1.0e-5}
    97	
    98	
    99	def build_model_and_state(subdivision, nlev, reorder_target, run_nd, method,
   100	                          moist=False, lloyd_iterations=50):
   101	    """Reordered+padded global mesh, MPAS PE model, baroclinic-wave IC.
   102	
   103	    ``reorder_target`` sets the PARTITION (and ghost padding) so every run
   104	    of a strong-scaling ladder times the IDENTICAL mesh; ``run_nd`` is the
   105	    device count of THIS run's mesh/model (the two differ for the
   106	    single-device reference leg of a ladder, via ``--reorder-for``).
   107	    ``moist=True`` attaches the q_v/q_c/q_r tracers (moist baroclinic
   108	    wave) so the sharded step's packed tracer halo exchange + RK tracer
   109	    advection sit on the timed/gated path.
   110	    """
   111	    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
   112	        MPASPrimitiveEquationConfig,
   113	        MPASPrimitiveEquationModel,
   114	    )
   115	    from legoesm.grids.vertical import create_sigma_coordinate
   116	    from legoesm.grids.voronoi import create_voronoi_mesh
   117	    from legoesm.parallel.mesh import create_voronoi_device_mesh
   118	    from legoesm.parallel.voronoi_partition import reorder_voronoi_for_sharding
   119	
   120	    mesh = create_voronoi_mesh(subdivision_level=subdivision,
   121	                               lloyd_iterations=lloyd_iterations)
   122	    mesh = reorder_voronoi_for_sharding(mesh, reorder_target, method=method)
   123	    if run_nd > 1 and (mesh.nCells % run_nd or mesh.nEdges % run_nd):
   124	        # Padding only guarantees divisibility for reorder_target.
   125	        raise SystemExit(
   126	            f"padded mesh (nCells={mesh.nCells}, nEdges={mesh.nEdges}) not "
   127	            f"divisible by --n-devices {run_nd}; use a ladder where every "
   128	            f"count divides --reorder-for ({reorder_target}).")
   129	    sigma = create_sigma_coordinate(nlev)
   130	    # Same recipe as the icosahedral lane of run_levante_gpu_scaling /
   131	    # tests/parallel/test_voronoi_sharded_equivalence.py: del4 hyperdiffusion,
   132	    # energy-conserving PV flux, SSP-RK3, global mass fixer.
   133	    cfg = MPASPrimitiveEquationConfig(
   134	        nu_del4=1e16, nu_del4_ps=1e16, fix_mass=True,
   135	        pv_scheme="energy", time_integrator="ssp_rk3",
   136	    )
   137	    dev_config = create_voronoi_device_mesh(
   138	        nCells=mesh.nCells, nEdges=mesh.nEdges, nVertices=mesh.nVertices,
   139	        n_devices=run_nd,
   140	    )
   141	    if dev_config.n_devices > 1:
   142	        from legoesm.parallel.mesh import replicate_pytree
   143	        mesh_model = replicate_pytree(mesh, dev_config)
   144	    else:
   145	        mesh_model = mesh
   146	    model = MPASPrimitiveEquationModel(mesh_model, sigma, cfg)
   147	    # #1100 MPAS twin: the timed path never global-builds the state.
   148	    # build_sharded_baroclinic_wave_state_mpas creates every leaf via
   149	    # jax.make_array_from_callback (only THIS process's shard rows are
   150	    # ever materialised; value-identical (few-ULP contract, measured
   151	    # exact on the pinned CPU stack) to global-build + shard_pytree —
   152	    # tests/parallel/test_mpas_partitionlocal_build.py).  The GLOBAL
   153	    # state is built lazily in main() only for the parity/conservation
   154	    # gates (small smoke scales).  The mesh itself is still global per
   155	    # process — its SFC-partition-local construction is the open
   156	    # remainder of #1100.
   157	    from tests.test_cases.baroclinic_wave import (
   158	        build_sharded_baroclinic_wave_state_mpas,
   159	    )
   160	    state_sharded = build_sharded_baroclinic_wave_state_mpas(
   161	        mesh, sigma, dev_config, perturbed=True, moist=moist)
   162	    return mesh, model, state_sharded, dev_config
   163	
   164	
   165	def _block(state):
   166	    jax.block_until_ready([leaf for leaf in jax.tree.leaves(state)
   167	                           if leaf is not None])
   168	
   169	
   170	def _global_dry_mass(state, mesh):
   171	    """sum(p_s * areaCell) on host arrays — the quantity fix_mass pins."""
   172	    ps = np.asarray(state.p_s.data)
   173	    area = np.asarray(mesh.areaCell)
   174	    return float(np.sum(ps * area))
   175	
   176	
   177	def main() -> int:
   178	    p = argparse.ArgumentParser()
   179	    p.add_argument("--subdivision", type=int, default=5,
   180	                   help="icosahedral subdivision level L "
   181	                        "(nCells = 10*4^L + 2 before ghost padding)")
   182	    p.add_argument("--nlev", type=int, default=8)
   183	    p.add_argument("--lloyd", type=int, default=50,
   184	                   help="Lloyd relaxation iterations for the mesh. 50 = "
   185	                        "production SCVT; 0 = labelled synthetic scaling "
   186	                        "mesh (scaling receipts only, never physics — "
   187	                        "must match the prewarmed cache key at subdiv>=9).")
   188	    p.add_argument("--n-devices", type=int, required=True)
   189	    p.add_argument("--reorder-for", type=int, default=None,
   190	                   help="partition/reorder the mesh for THIS device count "
   191	                        "(default: --n-devices). Pin it to the ladder's "
   192	                        "max so single-device reference runs time the "
   193	                        "identical reordered mesh.")
   194	    p.add_argument("--partition-method",
   195	                   choices=["auto", "geometric", "metis", "sfc"],
   196	                   default="auto")
   197	    p.add_argument("--physics", choices=["none", "held_suarez", "kessler"],
   198	                   default="none",
   199	                   help="Operator-split physics on the timed path. "
   200	                        "'kessler' also attaches the q_v/q_c/q_r moist-"
   201	                        "baroclinic-wave tracers (packed tracer halo "
   202	                        "exchange + RK tracer advection on the gated "
   203	                        "path) and extends the parity gate to the "
   204	                        "tracer fields.")
   205	    p.add_argument("--halo-strategy",
   206	                   choices=["auto", "ppermute", "allgather"],
   207	                   default="auto",
   208	                   help="Halo strategy for make_voronoi_sharded_step. "
   209	                        "'auto' picks allgather below the per-device "
   210	                        "cell threshold — force 'ppermute' to exercise "
   211	                        "the neighbor-round schedule on small gate "
   212	                        "meshes (the multicontroller selfspawn tests "
   213	                        "do).  Recorded in the JSONL row.")
   214	    p.add_argument("--steps", type=int, default=12)
   215	    p.add_argument("--warmup", type=int, default=2)
   216	    p.add_argument("--dt", type=float, default=None,
   217	                   help="timestep [s]; default auto: 600 * 4**(4-L) "
   218	                        "(CFL: dx halves per level), min 30 s.")
   219	    p.add_argument("--out", type=str,
   220	                   default="results/a1/mpas_spmd_scaling.jsonl")
   221	    p.add_argument(
   222	        "--parity-gate", action="store_true",
   223	        help="Correctness gate: compare the gathered sharded trajectory "
   224	             "against the single-device model.step trajectory on the SAME "
   225	             "reordered mesh (smoke windows only; the re-association floor "
   226	             "grows with steps).")
   227	    p.add_argument(
   228	        "--check-conservation", action="store_true",
   229	        help="Gate global dry-mass drift sum(p_s*areaCell) over the run "
   230	             "(pre-shard state vs gathered final state; exits nonzero on "
   231	             "breach).")
   232	    p.add_argument("--mass-rtol", type=float, default=None,
   233	                   help="Conservation tolerance (default: 1e-11 f64 / "
   234	                        "1e-5 f32 — fix_mass pins the mass each step).")
   235	    p.add_argument("--multicontroller", action="store_true",
   236	                   help="Route-B multi-controller: jax.distributed.initialize "
   237	                        "per process, ('device',) mesh over the GLOBAL device "
   238	                        "set (one process per GPU / per CPU-device group). NO "
   239	                        "mpi4jax. --n-devices must equal the global device "
   240	                        "count.")
   241	    p.add_argument("--coordinator", type=str, default=None,
   242	                   help="host:port for jax.distributed when auto-detection "
   243	                        "(SLURM) is unavailable; process count/id then come "
   244	                        "from OMPI_COMM_WORLD_SIZE/RANK.")
   245	    args = p.parse_args()
   246	
   247	    # Validate the timing window BEFORE any model/device work (codex, ocean
   248	    # twin): an empty steady slice would only fail after the expensive run.
   249	    if args.steps < 1:
   250	        raise SystemExit(f"--steps must be >= 1, got {args.steps}")
   251	    if not (0 <= args.warmup < args.steps):
   252	        raise SystemExit(
   253	            f"--warmup must satisfy 0 <= warmup < steps "
   254	            f"(got warmup={args.warmup}, steps={args.steps})")
   255	    if args.parity_gate and args.steps > MPAS_PARITY_MAX_STEPS:
   256	        raise SystemExit(
   257	            f"--parity-gate is a smoke gate (re-association floor grows "
   258	            f"with steps); --steps {args.steps} > {MPAS_PARITY_MAX_STEPS} "
   259	            f"cap.")
   260	
   261	    if args.multicontroller:
   262	        # MUST run before any other JAX use (backend init). SLURM auto-detects;
   263	        # mpiexec needs the explicit coordinator + launcher env vars (OpenMPI
   264	        # OMPI_*, or Cray PALS PMI_* on Derecho).
   265	        if args.coordinator is not None:
   266	            n_procs = int(os.environ.get(
   267	                "OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "0")))
   268	            proc_id = int(os.environ.get(
   269	                "OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "-1")))
   270	            if n_procs < 1 or proc_id < 0:
   271	                raise SystemExit(
   272	                    "--coordinator given but no launcher rank env found "
   273	                    "(OMPI_COMM_WORLD_SIZE/RANK or PMI_SIZE/PMI_RANK).")
   274	            jax.distributed.initialize(
   275	                coordinator_address=args.coordinator,
   276	                num_processes=n_procs, process_id=proc_id)
   277	        else:
   278	            # Environment-routed: SLURM/OMPI -> bare auto-detect; PALS/PMI
   279	            # (Derecho mpiexec) -> mpi4py bootstrap. Real init failures
   280	            # re-raise loudly.
   281	            from legoesm.parallel.early_init import (
   282	                init_jax_distributed_with_fallback,
   283	            )
   284	            init_jax_distributed_with_fallback()
   285	
   286	    from legoesm.parallel.sharded_dynamics import (
   287	        gather_voronoi_state_spmd,
   288	        make_voronoi_sharded_step,
   289	    )
   290	
   291	    nd = args.n_devices
   292	    avail = len(jax.devices())
   293	    if avail < nd:
   294	        raise SystemExit(f"need {nd} devices, have {avail} "
   295	                         f"(set --xla_force_host_platform_device_count)")
   296	    if args.multicontroller and nd != avail:
   297	        # A mesh over a strict subset would leave some processes' devices out
   298	        # of the program (non-addressable participation hazard). Route-B uses
   299	        # ALL global devices: one shard per device across every process.
   300	        raise SystemExit(
   301	            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
   302	            f"device count ({avail} across {jax.process_count()} processes).")
   303	
   304	    dt = args.dt
   305	    if dt is None:
   306	        dt = max(600.0 * 4.0 ** (4 - args.subdivision), 30.0)
   307	
   308	    reorder_for = args.reorder_for if args.reorder_for is not None else nd
   309	    if reorder_for < nd:
   310	        raise SystemExit(
   311	            f"--reorder-for ({reorder_for}) must be >= --n-devices ({nd}): "
   312	            f"the ghost padding only guarantees divisibility for the "
   313	            f"partition target.")
   314	    mesh, model, s0, dev_config = build_model_and_state(
   315	        args.subdivision, args.nlev, reorder_for, nd, args.partition_method,
   316	        moist=(args.physics == "kessler"), lloyd_iterations=args.lloyd)
   317	
   318	    if args.multicontroller:
   319	        # Every process computed the reorder independently — assert the
   320	        # partitions agree before any collective uses the halo schedule.
   321	        # The checksum covers the entity ORDER (coordinates) and the
   322	        # connectivity the ppermute schedule + TRiSK stencils read; a
   323	        # rank-divergent partition (e.g. one rank resolving
   324	        # --partition-method auto to METIS, another to RCB) cannot slip
   325	        # through on cell positions alone.
   326	        from jax.experimental import multihost_utils
   327	        crc = 0
   328	        for arr, dtype in (
   329	            (mesh.latCell, np.float64), (mesh.latEdge, np.float64),
   330	            (mesh.cellsOnEdge, np.int64), (mesh.edgesOnCell, np.int64),
   331	            (mesh.cellsOnCell, np.int64), (mesh.areaCell, np.float64),
   332	        ):
   333	            crc = zlib.crc32(np.ascontiguousarray(
   334	                np.asarray(arr, dtype=dtype)).tobytes(), crc)
   335	        crc = zlib.crc32(
   336	            np.asarray([mesh.nCells, mesh.nEdges, mesh.nVertices],
   337	                       dtype=np.int64).tobytes(), crc)
   338	        multihost_utils.assert_equal(
   339	            np.uint32(crc),
   340	            fail_message="partition/reorder checksum differs across "
   341	                         "processes (rank-divergent --partition-method "
   342	                         "resolution?)")
   343	
   344	    physics_fn = None
   345	    if args.physics == "held_suarez":
   346	        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_mpas
   347	        physics_fn = held_suarez_forcing_mpas
   348	    elif args.physics == "kessler":
   349	        # Warm-rain microphysics over the moist BCW tracers.  Kessler's
   350	        # saturation adjustment is a rate over the dt bound HERE, so it
   351	        # must match the stepping dt (make_kessler_forcing_mpas contract).
   352	        from legoesm.atmosphere.forcing.idealized.kessler_forcing import (
   353	            make_kessler_forcing_mpas,
   354	        )
   355	        physics_fn = make_kessler_forcing_mpas(dt)
   356	
   357	    # #1100: s0 from build_model_and_state is ALREADY partition-local-sharded
   358	    # when n_devices > 1 (no per-process global build on the timed path).
   359	    # The parity/conservation gates are the ONLY consumers of a global
   360	    # initial state — build it lazily here, at their smoke scales only
   361	    # (deterministic identical build on every process; bit-identical to the
   362	    # sharded s0 per tests/parallel/test_mpas_partitionlocal_build.py).
   363	    s0_global = None
   364	    if args.parity_gate or args.check_conservation:
   365	        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
   366	        s0_global = (s0 if dev_config.n_devices <= 1
   367	                     else baroclinic_wave_init_mpas(
   368	                         mesh, model.sigma_coord, perturbed=True,
   369	                         moist=(args.physics == "kessler")))
   370	
   371	    # Parity reference: the plain single-device trajectory on the SAME
   372	    # reordered mesh.  model.step's signature is call-compatible.
   373	    serial_final = None
   374	    if args.parity_gate:
   375	        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
   376	            MPASPrimitiveEquationModel,
   377	        )
   378	        ref_model = (model if dev_config.n_devices <= 1
   379	                     else MPASPrimitiveEquationModel(
   380	                         mesh, model.sigma_coord, model.config))
   381	        _s = s0_global
   382	        for _ in range(args.steps):
   383	            _s = ref_model.step(_s, dt, physics_fn=physics_fn)
   384	        _block(_s)
   385	        serial_final = _s
   386	
   387	    mass_before = None
   388	    if args.check_conservation:
   389	        mass_before = _global_dry_mass(s0_global, mesh)
   390	
   391	    step = make_voronoi_sharded_step(
   392	        model, dev_config, halo_strategy=args.halo_strategy)
   393	    # Already in the sharded layout (partition-local build) for nd > 1;
   394	    # single-device s0 is the plain global state.
   395	    s = s0
   396	
   397	    # Multi-controller: align every process around the timed loop.
   398	    if jax.process_count() > 1:
   399	        from jax.experimental import multihost_utils
   400	        multihost_utils.sync_global_devices("mpas_spmd_bench_start")
   401	
   402	    # Per-step timing: step 0 includes compile; record each step so re-trace
   403	    # (every step slow) is visible vs steady-state (steps 1.. fast).
   404	    per_step_ms = []
   405	    for _ in range(args.steps):
   406	        t0 = time.perf_counter()
   407	        if physics_fn is not None:
   408	            s = step(s, dt, physics_fn=physics_fn)
   409	        else:
   410	            s = step(s, dt)
   411	        _block(s)
   412	        per_step_ms.append((time.perf_counter() - t0) * 1e3)
   413	
   414	    if jax.process_count() > 1:
   415	        from jax.experimental import multihost_utils
   416	        multihost_utils.sync_global_devices("mpas_spmd_bench_end")
   417	
   418	    # HLO collective-permute census (#1113 ask 2): a STATIC compile property of
   419	    # the sharded step — the ppermute ROUND count that decomposes multi-node
   420	    # overhead (overhead ~= CPs/step * ~0.11 ms launch floor). The cube benches
   421	    # record this; the MPAS row did not, forcing an out-of-band census. Counted
   422	    # AFTER the timed loop so the census compile can't perturb per_step_ms[0]'s
   423	    # compile timing (the executable is already cached — this re-lower/compile
   424	    # is a cache hit; the count is data-independent, static in the partition).
   425	    # Best-effort (None if compilation is unsupported); the serial n=1 leg has
   426	    # no ppermute halo -> 0.
   427	    if physics_fn is not None:
   428	        _census_fn = lambda st: step(st, dt, physics_fn=physics_fn)  # noqa: E731
   429	    else:
   430	        _census_fn = lambda st: step(st, dt)  # noqa: E731
   431	    # ONE compile → full per-family census; the CP scalar (the #1113 round-count
   432	    # wall) is the collective_permute member, so no second compile for it.
   433	    hlo_census = hlo_collective_census(_census_fn, s)
   434	    hlo_cp = hlo_census["collective_permute"] if hlo_census else None
   435	
   436	    # --- Correctness gates (before any timing is reported) -----------------
   437	    if args.parity_gate or args.check_conservation:
   438	        final_global = (gather_voronoi_state_spmd(s, dev_config)
   439	                        if dev_config.n_devices > 1 else s)
   440	        prec = "float64" if jax.config.jax_enable_x64 else "float32"
   441	        rank0 = jax.process_index() == 0
   442	        if args.check_conservation:
   443	            mass_after = _global_dry_mass(final_global, mesh)
   444	            tol = (args.mass_rtol if args.mass_rtol is not None
   445	                   else MASS_RTOL_DEFAULTS[prec])
   446	            rel = abs(mass_after - mass_before) / abs(mass_before)
   447	            if rank0:
   448	                print(f"    conservation dry-mass: rel drift={rel:.3e} "
   449	                      f"(tol {tol:.1e}) over {args.steps} steps", flush=True)
   450	            if rel > tol:
   451	                if rank0:
   452	                    print("ERROR: conservation gate BREACHED.", flush=True)
   453	                return 4
   454	        if args.parity_gate:
   455	            tols = MPAS_PARITY_TOLS[prec]
   456	            ok = True
   457	            checks = [
   458	                (name, getattr(serial_final, name).data,
   459	                 getattr(final_global, name).data, rtol, atol)
   460	                for name, (rtol, atol) in tols.items()
   461	            ]
   462	            if serial_final.tracers is not None:
   463	                # Moist run: the tracer fields ride the packed exchange +
   464	                # RK advection — gate them too (q re-association floor is
   465	                # far below the q_v scale; reuse the T tolerances).
   466	                q_rtol, q_atol = tols["T"]
   467	                if set(final_global.tracers or {}) != set(
   468	                        serial_final.tracers):
   469	                    if rank0:
   470	                        print("ERROR: sharded run dropped tracer fields.",
   471	                              flush=True)
   472	                    return 5
   473	                checks += [
   474	                    (k, serial_final.tracers[k].data,
   475	                     final_global.tracers[k].data, q_rtol, q_atol * 1e-3)
   476	                    for k in sorted(serial_final.tracers)
   477	                ]
   478	            for name, want, got, rtol, atol in checks:
   479	                want = np.asarray(want)
   480	                got = np.asarray(got)
   481	                field_ok = bool(np.allclose(got, want, rtol=rtol, atol=atol))
   482	                ok &= field_ok
   483	                if rank0:
   484	                    mx = (float(np.max(np.abs(got - want)))
   485	                          if want.size else 0.0)
   486	                    print(f"    parity {name:>4s}: max|diff|={mx:.3e} "
   487	                          f"{'OK' if field_ok else 'MISMATCH'}", flush=True)
   488	            if not ok:
   489	                if rank0:
   490	                    print("ERROR: SPMD parity gate MISMATCH vs the "
   491	                          "single-device reference.", flush=True)
   492	                return 5
   493	
   494	    steady = per_step_ms[args.warmup:]
   495	    med = float(np.median(steady))
   496	    rec = dict(
   497	        component="mpas_atm",
   498	        subdivision=args.subdivision, n_devices=nd,
   499	        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
   500	        partition_method=args.partition_method, physics=args.physics,
   501	        # lloyd=0 is the LABELLED synthetic scaling mesh — anti-masquerade:
   502	        # a row without this field could pass as a production-SCVT receipt.
   503	        lloyd_iterations=args.lloyd,
   504	        # Requested vs EFFECTIVE (post-"auto") strategy — a JSONL row
   505	        # saying "auto" would not reveal whether ppermute or allgather
   506	        # was actually measured (codex M3c-2 MINOR).
   507	        halo_strategy_requested=args.halo_strategy,
   508	        halo_strategy_effective=getattr(
   509	            step, "_halo_strategy_effective", "serial"),
   510	        steps=args.steps, dt=dt,
   511	        platform=jax.default_backend(),
   512	        n_processes=jax.process_count(),
   513	        multicontroller=bool(args.multicontroller),
   514	        compile_ms=round(per_step_ms[0], 1),
   515	        steady_median_ms=round(med, 2),
   516	        steady_min_ms=round(float(np.min(steady)), 2),
   517	        per_step_ms=[round(x, 1) for x in per_step_ms],
   518	        cells=int(mesh.nCells) * args.nlev,
   519	        # ppermute round count/step (static compile property; #1113) — the
   520	        # multi-node ceiling is this count x the ~0.11 ms launch floor, so it
   521	        # belongs on every row like the cube benches.
   522	        hlo_collective_permutes=hlo_cp,
   523	        # full per-family census (permute + all-reduce + all-gather + ...) on
   524	        # the SAME compile: exposes any reduction the ico step introduces.
   525	        hlo_collectives=hlo_census,
   526	    )
   527	    # Flat aggregator-compatible identity + metric fields (see the latlon
   528	    # twin): resolution = subdivision level, matching run_cpu_mpi_scaling's
   529	    # icosahedral convention so both lanes land on the same plot curves.
   530	    rec.update(
   531	        grid_type="icosahedral",
   532	        resolution=args.subdivision,
   533	        n_levels=args.nlev,
   534	        mode="strong",  # this bench fixes the mesh and sweeps devices
   535	        precision="float64" if jax.config.jax_enable_x64 else "float32",
   536	        physics_level=args.physics,
   537	        backend=jax.default_backend(),
   538	        **tidy_throughput_fields(
   539	            dt_seconds=dt, time_per_step_ms=med,
   540	            total_cells=int(mesh.nCells) * args.nlev),
   541	    )
   542	    rec["metadata"] = annotate_incomplete(scaling_metadata(
   543	        grid="icosahedral",
   544	        component="atmosphere",
   545	        resolution=f"L{args.subdivision}",
   546	        n_levels=args.nlev,
   547	        precision="float64" if jax.config.jax_enable_x64 else "float32",
   548	        n_gpus=(nd if jax.default_backend() in ("gpu", "cuda", "rocm")
   549	                else 0),
   550	        decomposition="cell_partition" if nd > 1 else "none",
   551	        # cells_per_rank is per PROCESS (n_ranks semantics); the per-device
   552	        # share lives in extra.cells_per_device — a single-process 4-device
   553	        # SPMD run has 1 rank owning ALL cells (codex finding 3).
   554	        cells_per_rank=int(mesh.nCells) * args.nlev
   555	        // max(jax.process_count(), 1),
   556	        scaling_kind="strong",  # this bench fixes the mesh and sweeps devices
   557	        extra={
   558	            "partition_method": args.partition_method,
   559	            "physics": args.physics,
   560	            "steps": args.steps,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-22-Multi-controller (route-B, ``--multicontroller``): identical contract to the
scripts/bench/bench_ocean_latlon_spmd_scaling.py-23-atm bench — every process calls ``jax.distributed.initialize`` BEFORE any
scripts/bench/bench_ocean_latlon_spmd_scaling.py-24-other JAX use, the ("lat",) mesh is built over the GLOBAL ``jax.devices()``,
scripts/bench/bench_ocean_latlon_spmd_scaling.py-25-and the existing ``make_sharded_ocean_step`` band-ppermute halo + psum
scripts/bench/bench_ocean_latlon_spmd_scaling.py-26-reductions (incl. the barotropic ``_global_sum_pair``) run unchanged across
scripts/bench/bench_ocean_latlon_spmd_scaling.py:27:processes (NCCL on GPU / gloo on CPU). NO mpi4jax is armed in this mode (the
scripts/bench/bench_ocean_latlon_spmd_scaling.py-28-documented mixed-stack deadlock hazard).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-29-
scripts/bench/bench_ocean_latlon_spmd_scaling.py-30-Launch (cluster, one process per GPU):
scripts/bench/bench_ocean_latlon_spmd_scaling.py-31-  srun -n 8 python bench_ocean_latlon_spmd_scaling.py --multicontroller \
scripts/bench/bench_ocean_latlon_spmd_scaling.py-32-      --n-devices 8 ...            # SLURM: coordinator auto-detected
scripts/bench/bench_ocean_latlon_spmd_scaling.py-33-  mpiexec -n 8 python ... --multicontroller --coordinator host0:9876
scripts/bench/bench_ocean_latlon_spmd_scaling.py-34-CPU smoke (single process, virtual devices):
scripts/bench/bench_ocean_latlon_spmd_scaling.py:35:  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
scripts/bench/bench_ocean_latlon_spmd_scaling.py-36-  JAX_ENABLE_X64=1 python scripts/bench/bench_ocean_latlon_spmd_scaling.py \
scripts/bench/bench_ocean_latlon_spmd_scaling.py-37-      --n-lat 48 --n-lon 96 --nlev 10 --n-devices 4 --steps 4
scripts/bench/bench_ocean_latlon_spmd_scaling.py-38-"""
scripts/bench/bench_ocean_latlon_spmd_scaling.py-39-from __future__ import annotations
scripts/bench/bench_ocean_latlon_spmd_scaling.py-40-
--
scripts/bench/bench_ocean_latlon_spmd_scaling.py-847-            # recorded fused_halo=false — a falsely-labelled baseline row
scripts/bench/bench_ocean_latlon_spmd_scaling.py-848-            # (codex review).
scripts/bench/bench_ocean_latlon_spmd_scaling.py-849-            "fused_halo": os.environ.get(
scripts/bench/bench_ocean_latlon_spmd_scaling.py-850-                "LEGOESM_LATLON_SPMD_FUSED_HALO", "1") != "0",
scripts/bench/bench_ocean_latlon_spmd_scaling.py-851-            # Route-B transport facts (socket-fallback flag): a
scripts/bench/bench_ocean_latlon_spmd_scaling.py:852:            # multi-node row without an NCCL net plugin is
scripts/bench/bench_ocean_latlon_spmd_scaling.py-853-            # falsifiable from the record alone.
scripts/bench/bench_ocean_latlon_spmd_scaling.py-854-            "nccl": (_nccl_report if args.multicontroller
scripts/bench/bench_ocean_latlon_spmd_scaling.py-855-                     else None),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-856-            "parity_gate": bool(args.parity_gate),
scripts/bench/bench_ocean_latlon_spmd_scaling.py-857-            "check_conservation": bool(args.check_conservation),
--
scripts/cluster/scaling_levante/README.md-63-thing to validate (intra-node cuda_ipc is well-trodden; inter-node gdr_copy less
scripts/cluster/scaling_levante/README.md-64-so).
scripts/cluster/scaling_levante/README.md-65-
scripts/cluster/scaling_levante/README.md-66----
scripts/cluster/scaling_levante/README.md-67-
scripts/cluster/scaling_levante/README.md:68:## Route-B (jax.distributed + NCCL) + ocean + CPU additions (2026-07)
scripts/cluster/scaling_levante/README.md-69-
scripts/cluster/scaling_levante/README.md-70-Beyond the route-A `gpu_moist_scaling.slurm` above:
scripts/cluster/scaling_levante/README.md-71-
scripts/cluster/scaling_levante/README.md-72-| File | What |
scripts/cluster/scaling_levante/README.md-73-|---|---|
scripts/cluster/scaling_levante/README.md-74-| `gpu_scaling.sbatch` | ATM (cube AMIP physics, single-process multi-GPU) + OCEAN (`bench_ocean_latlon_spmd_scaling.py`, 1/2/4 A100 with parity+conservation smoke) on one GPU node. Plain GPU env — no mpi4jax. |
scripts/cluster/scaling_levante/README.md:75:| `gpu_multinode_scaling.sbatch` | MULTI-NODE route-B lanes over NCCL/IB (SLURM auto-detected `jax.distributed`): A = cube `--cs-spmd` (6 GPU / 2 nodes), C = atm lat-lon `--multicontroller` (8 GPU), D = ocean `--multicontroller` (8 GPU). |
scripts/cluster/scaling_levante/README.md-76-| `cpu_scaling.sbatch` | ATM + OCEAN CPU-MPI rank ladders on `compute` nodes (`legoesm-mpi` env), with fail-fast smokes. |
scripts/cluster/scaling_levante/README.md-77-| `diagnosis.sbatch` | BOTTLENECK diagnosis via `scripts/bench/run_scaling_diagnosis.py` — per-phase halo bandwidth, reduction latency, roofline, compute/comm overlap, and the **static collective census** (collective-permute + all-reduce + all-gather per step) the throughput jobs do NOT capture. Climbs the cube face-shard `1 2 3` ladder so the message-count-vs-shard curve — the LATENCY-bound anti-scaling signal — is recorded. `sbatch --export=ALL,MODE=census` for counts only. Derecho twin: `scaling_derecho/diagnosis.pbs`. |
scripts/cluster/scaling_levante/README.md-78-
scripts/cluster/scaling_levante/README.md:79:`_env.sh` now also carries the NCCL-over-IB defaults for the route-B lanes
scripts/cluster/scaling_levante/README.md:80:(`NCCL_IB_HCA=mlx5`, `NCCL_SOCKET_IFNAME=ib0`, `NCCL_NET_GDR_LEVEL=PHB`,
scripts/cluster/scaling_levante/README.md:81:`NCCL_CROSS_NIC=1`) — independent of, and coexisting with, the UCX/MPI
scripts/cluster/scaling_levante/README.md:82:settings used by route-A. First multi-node run: `NCCL_DEBUG=INFO` must show
scripts/cluster/scaling_levante/README.md:83:`NET/IB` (not `NET/Socket`); a wrong `NCCL_SOCKET_IFNAME` is the #1 cause of
scripts/cluster/scaling_levante/README.md-84-IB-cluster bootstrap timeouts (verify with `ip addr` on a gpu node).
scripts/cluster/scaling_levante/README.md-85-
scripts/cluster/scaling_levante/README.md-86-Account: export `SBATCH_ACCOUNT=<project>` (or pass `sbatch -A <project>`) —
scripts/cluster/scaling_levante/README.md-87-SLURM directives cannot expand env vars.
scripts/cluster/scaling_levante/README.md-88-
scripts/cluster/scaling_levante/README.md-89-
scripts/cluster/scaling_levante/README.md-90-## 2026-07 lane E: icosahedral/MPAS multicontroller
scripts/cluster/scaling_levante/README.md-91-
scripts/cluster/scaling_levante/README.md-92-`gpu_multinode_scaling.sbatch` gained lane E (`RUN_MPAS=1`, default on):
scripts/cluster/scaling_levante/README.md:93:icosahedral MPAS PE over `jax.distributed` + NCCL via
scripts/cluster/scaling_levante/README.md-94-`scripts/bench/bench_mpas_spmd_scaling.py` (cell-partition reorder +
scripts/cluster/scaling_levante/README.md-95-ppermute halos), 6 tasks = 2 nodes x 3 GPUs (`nCells = 10*4^L + 2` splits
scripts/cluster/scaling_levante/README.md-96-evenly for 1/2/3/6). Subdiv-4 parity+conservation smoke gates the timed
scripts/cluster/scaling_levante/README.md-97-`ICO_LEVEL` (default L7) case. Federation gate:
scripts/cluster/scaling_levante/README.md-98-`tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py`.
--
scripts/cluster/scaling_levante/README.md-105-Derecho README's lane-T section). Arms on the atm latlon (np=8), ocean
scripts/cluster/scaling_levante/README.md-106-(np=8) and cube cs-spmd (np=6, `base`/`xla` only) lanes, all within ONE
scripts/cluster/scaling_levante/README.md-107-allocation against a fresh `base` control arm:
scripts/cluster/scaling_levante/README.md-108-`fused` (`LEGOESM_LATLON_SPMD_FUSED_HALO=1`, bit-identical packing,
scripts/cluster/scaling_levante/README.md-109-trace receipt 41→29 CPs/step), `xla`
scripts/cluster/scaling_levante/README.md:110:(`--xla_gpu_collective_permute_combine_threshold_bytes=32MiB` +
scripts/cluster/scaling_levante/README.md:111:`--xla_gpu_enable_pipelined_p2p=true`), `pgle`
scripts/cluster/scaling_levante/README.md-112-(`JAX_ENABLE_PGLE=true`, per-run opt-in — recompiles after profiling).
scripts/cluster/scaling_levante/README.md-113-
scripts/cluster/scaling_levante/README.md-114-```bash
scripts/cluster/scaling_levante/README.md:115:sbatch --export=ALL,RUN_TUNE=1,RUN_NCCL=0,RUN_LATLON=0,RUN_OCEAN=0,RUN_MPAS=0 \
scripts/cluster/scaling_levante/README.md-116-    scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch
scripts/cluster/scaling_levante/README.md-117-```
scripts/cluster/scaling_levante/README.md-118-
scripts/cluster/scaling_levante/README.md-119-Outputs under `$OUTDIR/_ab_tuning/` (aggregator-skipped by design).
--
scripts/bench/roofline_probe.py-870-    achieved_gbps_upper: float    # most assumed bytes (upper-bytes).
scripts/bench/roofline_probe.py-871-    sustained_gbps: float | None
scripts/bench/roofline_probe.py-872-    pct_sustained_central: float | None
scripts/bench/roofline_probe.py-873-    pct_sustained_range: tuple[float | None, float | None]
scripts/bench/roofline_probe.py-874-    # Collective census + floor.
scripts/bench/roofline_probe.py:875:    hlo_collective_permute: int
scripts/bench/roofline_probe.py:876:    hlo_collective_permute_start: int
scripts/bench/roofline_probe.py:877:    hlo_collective_permute_done: int
scripts/bench/roofline_probe.py-878-    collective_time_floor_ms: float | None
scripts/bench/roofline_probe.py-879-    collective_floor_note: str
scripts/bench/roofline_probe.py-880-    # PCG Amdahl (None for non-PCG steps).
scripts/bench/roofline_probe.py-881-    pcg_amdahl: dict | None = None
scripts/bench/roofline_probe.py-882-
--
scripts/bench/roofline_probe.py-900-    n_levels: int,
scripts/bench/roofline_probe.py-901-    n_gpus: int,
scripts/bench/roofline_probe.py-902-    time_per_step_ms: float,
scripts/bench/roofline_probe.py-903-    bytes_model: BytesModel,
scripts/bench/roofline_probe.py-904-    sustained_gbps: float | None,
scripts/bench/roofline_probe.py:905:    hlo_collective_permute: int = -1,
scripts/bench/roofline_probe.py:906:    hlo_collective_permute_start: int = -1,
scripts/bench/roofline_probe.py:907:    hlo_collective_permute_done: int = -1,
scripts/bench/roofline_probe.py-908-    collective_latency_ms: float | None = None,
scripts/bench/roofline_probe.py-909-    allreduce_latency_ms: float | None = None,
scripts/bench/roofline_probe.py-910-    pcg_iterations: int | None = None,
scripts/bench/roofline_probe.py-911-    pcg_local_stencil_ms: float | None = None,
scripts/bench/roofline_probe.py-912-    pcg_halo_ms: float | None = None,
--
scripts/bench/roofline_probe.py-915-
scripts/bench/roofline_probe.py-916-    ``bytes_model`` carries the documented bytes/cell-level ESTIMATE; the
scripts/bench/roofline_probe.py-917-    achieved-GB/s output is therefore a BRACKET (central + lo/hi).
scripts/bench/roofline_probe.py-918-
scripts/bench/roofline_probe.py-919-    Collective census: pass the HLO counts straight from a
scripts/bench/roofline_probe.py:920:    ``run_levante_gpu_scaling.TimingResult`` (``hlo_collective_permute*``)
scripts/bench/roofline_probe.py-921-    — this reporter does NOT re-derive the census, it consumes the one the
scripts/bench/roofline_probe.py-922-    timed-executable HLO guard already produced.
scripts/bench/roofline_probe.py-923-
scripts/bench/roofline_probe.py-924-    Collective time-floor: ``collective_latency_ms`` × (number of
scripts/bench/roofline_probe.py-925-    collective ops in the step).  We count each sync collective-permute
--
scripts/bench/roofline_probe.py-955-        return 100.0 * g / sustained_gbps if sustained_gbps else None
scripts/bench/roofline_probe.py-956-
scripts/bench/roofline_probe.py-957-    # Collective time-floor.
scripts/bench/roofline_probe.py-958-    floor_ms = None
scripts/bench/roofline_probe.py-959-    floor_note = "no collective-latency input provided"
scripts/bench/roofline_probe.py:960:    if collective_latency_ms is not None and hlo_collective_permute >= 0:
scripts/bench/roofline_probe.py:961:        n_rounds = max(hlo_collective_permute, 0) + max(
scripts/bench/roofline_probe.py:962:            hlo_collective_permute_start, 0)
scripts/bench/roofline_probe.py-963-        floor_ms = n_rounds * collective_latency_ms
scripts/bench/roofline_probe.py-964-        floor_note = (
scripts/bench/roofline_probe.py-965-            f"{n_rounds} collective round(s) "
scripts/bench/roofline_probe.py:966:            f"(sync={max(hlo_collective_permute,0)} + "
scripts/bench/roofline_probe.py:967:            f"async-start={max(hlo_collective_permute_start,0)}) × "
scripts/bench/roofline_probe.py-968-            f"{collective_latency_ms:.4f} ms latency floor"
scripts/bench/roofline_probe.py-969-        )
scripts/bench/roofline_probe.py-970-
scripts/bench/roofline_probe.py-971-    pcg = None
scripts/bench/roofline_probe.py-972-    if pcg_iterations is not None:
--
scripts/bench/roofline_probe.py-1008-        achieved_gbps_lower=g_lower,
scripts/bench/roofline_probe.py-1009-        achieved_gbps_upper=g_upper,
scripts/bench/roofline_probe.py-1010-        sustained_gbps=sustained_gbps,
scripts/bench/roofline_probe.py-1011-        pct_sustained_central=_pct(g_central),
scripts/bench/roofline_probe.py-1012-        pct_sustained_range=(_pct(g_lower), _pct(g_upper)),
scripts/bench/roofline_probe.py:1013:        hlo_collective_permute=hlo_collective_permute,
scripts/bench/roofline_probe.py:1014:        hlo_collective_permute_start=hlo_collective_permute_start,
scripts/bench/roofline_probe.py:1015:        hlo_collective_permute_done=hlo_collective_permute_done,
scripts/bench/roofline_probe.py-1016-        collective_time_floor_ms=floor_ms,
scripts/bench/roofline_probe.py-1017-        collective_floor_note=floor_note,
scripts/bench/roofline_probe.py-1018-        pcg_amdahl=pcg,
scripts/bench/roofline_probe.py-1019-    )
scripts/bench/roofline_probe.py-1020-
--
scripts/bench/profile_mpas_ocean.py-59-            return True
scripts/bench/profile_mpas_ocean.py-60-    return False
scripts/bench/profile_mpas_ocean.py-61-
scripts/bench/profile_mpas_ocean.py-62-
scripts/bench/profile_mpas_ocean.py-63-def _enable_cuda_graphs():
scripts/bench/profile_mpas_ocean.py:64:    existing = os.environ.get("XLA_FLAGS", "")
scripts/bench/profile_mpas_ocean.py-65-    if _has_xla_token(existing, "--xla_gpu_enable_command_buffer"):
scripts/bench/profile_mpas_ocean.py-66-        return
scripts/bench/profile_mpas_ocean.py:67:    os.environ["XLA_FLAGS"] = f"{existing} {_CUDA_GRAPH_FLAG}".strip()
scripts/bench/profile_mpas_ocean.py-68-
scripts/bench/profile_mpas_ocean.py-69-
scripts/bench/profile_mpas_ocean.py-70-_enable_cuda_graphs()
scripts/bench/profile_mpas_ocean.py-71-
scripts/bench/profile_mpas_ocean.py-72-import jax
--
scripts/bench/profile_mpas_ocean.py-139-    print(
scripts/bench/profile_mpas_ocean.py-140-        f"profile_mpas_ocean: I{subdivision_level}/L{n_levels}, "
scripts/bench/profile_mpas_ocean.py-141-        f"precision={precision}, dt={dt}s"
scripts/bench/profile_mpas_ocean.py-142-    )
scripts/bench/profile_mpas_ocean.py-143-    print(f"device: {jax.devices()}  jax={jax.__version__}")
scripts/bench/profile_mpas_ocean.py:144:    print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}")
scripts/bench/profile_mpas_ocean.py-145-    print("=" * 78)
scripts/bench/profile_mpas_ocean.py-146-
scripts/bench/profile_mpas_ocean.py-147-    # --- Build mesh + state + model ----------------------------------------
scripts/bench/profile_mpas_ocean.py-148-    t0 = time.perf_counter()
scripts/bench/profile_mpas_ocean.py-149-    mesh = create_voronoi_mesh(
--
scripts/bench/bench_cube_tiled_step_scaling.py-26-Launch:
scripts/bench/bench_cube_tiled_step_scaling.py-27-  CPU smoke (24 virtual devices; NOTE the tiled-stage compile at 24
scripts/bench/bench_cube_tiled_step_scaling.py-28-  virtual devices is CLUSTER-scale — the pre-existing adapter parity gate
scripts/bench/bench_cube_tiled_step_scaling.py-29-  itself runs ~10 min on a laptop CPU, so budget accordingly or run the
scripts/bench/bench_cube_tiled_step_scaling.py-30-  smoke on a cluster CPU node):
scripts/bench/bench_cube_tiled_step_scaling.py:31:    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
scripts/bench/bench_cube_tiled_step_scaling.py-32-    python scripts/bench/bench_cube_tiled_step_scaling.py --kt 2 \
scripts/bench/bench_cube_tiled_step_scaling.py-33-        --resolution 48 --nlev 30 --steps 6 --parity-gate
scripts/bench/bench_cube_tiled_step_scaling.py-34-  Derecho (PBS/PALS, 24 A100 = 6 nodes x 4):
scripts/bench/bench_cube_tiled_step_scaling.py-35-    #PBS -l select=6:ncpus=64:mpiprocs=4:ngpus=4
scripts/bench/bench_cube_tiled_step_scaling.py-36-    mpiexec -n 24 python scripts/bench/bench_cube_tiled_step_scaling.py \
--
scripts/bench/bench_cube_tiled_step_scaling.py-56-import numpy as np
scripts/bench/bench_cube_tiled_step_scaling.py-57-
scripts/bench/bench_cube_tiled_step_scaling.py-58-sys.path.insert(0, str(Path(__file__).resolve().parent))
scripts/bench/bench_cube_tiled_step_scaling.py-59-
scripts/bench/bench_cube_tiled_step_scaling.py-60-from metadata import (  # noqa: E402
scripts/bench/bench_cube_tiled_step_scaling.py:61:    annotate_incomplete, count_collective_permutes, count_collectives,
scripts/bench/bench_cube_tiled_step_scaling.py-62-    scaling_metadata, tidy_throughput_fields)
scripts/bench/bench_cube_tiled_step_scaling.py-63-
scripts/bench/bench_cube_tiled_step_scaling.py-64-#: Parity tolerances vs the serial untiled step — the adapter gate's
scripts/bench/bench_cube_tiled_step_scaling.py-65-#: f32-honest bounds (exact f32 ulps of the field scales; a real stage
scripts/bench/bench_cube_tiled_step_scaling.py-66-#: regression is 2e-5-abs class).  These bound the SINGLE tiled-vs-serial
--
scripts/bench/bench_cube_tiled_step_scaling.py-68-#: stay the adapter gate's own tolerances so a silent loosening here can't let
scripts/bench/bench_cube_tiled_step_scaling.py-69-#: a stage regression pass the lane.
scripts/bench/bench_cube_tiled_step_scaling.py-70-TILED_PARITY_ATOL = {"u": 2e-5, "v": 2e-5, "T": 1e-4, "p_s": 0.06}
scripts/bench/bench_cube_tiled_step_scaling.py-71-
scripts/bench/bench_cube_tiled_step_scaling.py-72-
scripts/bench/bench_cube_tiled_step_scaling.py:73:# Shared canonical CP census (metadata.count_collective_permutes); kept as a
scripts/bench/bench_cube_tiled_step_scaling.py-74-# module-level name for the existing test + call site.
scripts/bench/bench_cube_tiled_step_scaling.py:75:_count_collective_permutes = count_collective_permutes
scripts/bench/bench_cube_tiled_step_scaling.py-76-
scripts/bench/bench_cube_tiled_step_scaling.py-77-
scripts/bench/bench_cube_tiled_step_scaling.py-78-def main() -> int:
scripts/bench/bench_cube_tiled_step_scaling.py-79-    p = argparse.ArgumentParser(
scripts/bench/bench_cube_tiled_step_scaling.py-80-        description=__doc__,
--
scripts/bench/bench_cube_tiled_step_scaling.py-245-    lowered = tiled_step.lower(_lower_arg)
scripts/bench/bench_cube_tiled_step_scaling.py-246-    _t_compile0 = time.perf_counter()
scripts/bench/bench_cube_tiled_step_scaling.py-247-    compiled = lowered.compile()
scripts/bench/bench_cube_tiled_step_scaling.py-248-    compile_ms = (time.perf_counter() - _t_compile0) * 1e3
scripts/bench/bench_cube_tiled_step_scaling.py-249-    hlo = compiled.as_text()
scripts/bench/bench_cube_tiled_step_scaling.py:250:    n_ppermute = _count_collective_permutes(hlo)
scripts/bench/bench_cube_tiled_step_scaling.py-251-    # Full per-family census (superset of the CP count): surfaces the
scripts/bench/bench_cube_tiled_step_scaling.py-252-    # conservation all-reduce and any operator-introduced resharding on the
scripts/bench/bench_cube_tiled_step_scaling.py-253-    # SAME audited executable — the message-count = latency-bound lever.
scripts/bench/bench_cube_tiled_step_scaling.py-254-    hlo_census = count_collectives(hlo)
scripts/bench/bench_cube_tiled_step_scaling.py-255-    allgathers = find_fullcube_allgathers(hlo, n=args.resolution)
--
scripts/bench/bench_cube_tiled_step_scaling.py-293-        multihost_utils.sync_global_devices("cube_tiled_bench_start")
scripts/bench/bench_cube_tiled_step_scaling.py-294-
scripts/bench/bench_cube_tiled_step_scaling.py-295-    if args.closed_loop:
scripts/bench/bench_cube_tiled_step_scaling.py-296-        # #921: the closed-loop step fuses the halo collective-permutes with
scripts/bench/bench_cube_tiled_step_scaling.py-297-        # the in-stage mass-fixer psum in ONE executable; on multi-process GPU
scripts/bench/bench_cube_tiled_step_scaling.py:298:        # the NCCL comm-init of those two clique kinds can be ordered
scripts/bench/bench_cube_tiled_step_scaling.py-299-        # differently per rank and DEADLOCK.  Prime every clique in a fixed,
scripts/bench/bench_cube_tiled_step_scaling.py-300-        # rank-independent order FIRST (no-op single-process / CPU-virtual).
scripts/bench/bench_cube_tiled_step_scaling.py-301-        from legoesm.parallel.tiled_production_cdgrid import (
scripts/bench/bench_cube_tiled_step_scaling.py-302-            warmup_tiled_cube_comms,
scripts/bench/bench_cube_tiled_step_scaling.py-303-        )
--
scripts/bench/bench_cube_tiled_step_scaling.py-366-        first_timed_step_ms=round(per_step_ms[0], 1),
scripts/bench/bench_cube_tiled_step_scaling.py-367-        steady_median_ms=round(med, 2),
scripts/bench/bench_cube_tiled_step_scaling.py-368-        steady_min_ms=round(float(np.min(steady)), 2),
scripts/bench/bench_cube_tiled_step_scaling.py-369-        per_step_ms=[round(x, 1) for x in per_step_ms],
scripts/bench/bench_cube_tiled_step_scaling.py-370-        cells=total_cells,
scripts/bench/bench_cube_tiled_step_scaling.py:371:        hlo_collective_permutes=n_ppermute,
scripts/bench/bench_cube_tiled_step_scaling.py-372-    )
scripts/bench/bench_cube_tiled_step_scaling.py-373-    # Flat aggregator-compatible identity + metric fields (see the latlon
scripts/bench/bench_cube_tiled_step_scaling.py-374-    # twin).  grid_type (not just rec["grid"]) is the aggregator's key.
scripts/bench/bench_cube_tiled_step_scaling.py-375-    rec.update(
scripts/bench/bench_cube_tiled_step_scaling.py-376-        grid_type="cubed-sphere",
--
scripts/bench/bench_cube_tiled_step_scaling.py-397-                        else "tiled_cc_base_cut"),
scripts/bench/bench_cube_tiled_step_scaling.py-398-        cells_per_rank=total_cells // max(int(jax.process_count()), 1),
scripts/bench/bench_cube_tiled_step_scaling.py-399-        scaling_kind="strong",
scripts/bench/bench_cube_tiled_step_scaling.py-400-        extra={
scripts/bench/bench_cube_tiled_step_scaling.py-401-            # Route-B transport facts (the PBS wrapper's contract): a
scripts/bench/bench_cube_tiled_step_scaling.py:402:            # multi-node row without an NCCL net plugin is falsifiable.
scripts/bench/bench_cube_tiled_step_scaling.py-403-            "nccl": (_nccl_report if args.multicontroller else None),
scripts/bench/bench_cube_tiled_step_scaling.py-404-            "kt": args.kt,
scripts/bench/bench_cube_tiled_step_scaling.py-405-            "steps": args.steps,
scripts/bench/bench_cube_tiled_step_scaling.py-406-            "warmup": args.warmup,
scripts/bench/bench_cube_tiled_step_scaling.py-407-            "multicontroller": bool(args.multicontroller),
scripts/bench/bench_cube_tiled_step_scaling.py-408-            "cells_per_device": total_cells // n_devices,
scripts/bench/bench_cube_tiled_step_scaling.py:409:            "hlo_collective_permutes": n_ppermute,
scripts/bench/bench_cube_tiled_step_scaling.py-410-            "hlo_collectives": hlo_census,
scripts/bench/bench_cube_tiled_step_scaling.py-411-            "closed_loop": bool(args.closed_loop),
scripts/bench/bench_cube_tiled_step_scaling.py-412-            "envelope": (
scripts/bench/bench_cube_tiled_step_scaling.py-413-                "blocked closed loop + in-stage telescoping mass fixer "
scripts/bench/bench_cube_tiled_step_scaling.py-414-                "(production conservation config; adapter-refused knobs "
--
scripts/bench/bench_mpas_spmd_scaling.py-13-  (Weak scaling rides the subdivision ladder: one level = 4x the cells, so
scripts/bench/bench_mpas_spmd_scaling.py-14-  L at 4*n_dev matches L-1 at n_dev per-device load — there is no per-device
scripts/bench/bench_mpas_spmd_scaling.py-15-  row knob like the lat-lon benches' --nlat-per-dev.)
scripts/bench/bench_mpas_spmd_scaling.py-16-
scripts/bench/bench_mpas_spmd_scaling.py-17-nlev caveat (#1113): the multi-node ceiling at THIN nlev is the ppermute ROUND
scripts/bench/bench_mpas_spmd_scaling.py:18:count (``hlo_collective_permutes``, recorded per row) x the ~0.11 ms launch
scripts/bench/bench_mpas_spmd_scaling.py-19-floor, NOT bandwidth — a fixed per-step overhead. At the ``--nlev 8`` default it
scripts/bench/bench_mpas_spmd_scaling.py-20-dominates (~1.34 Gcells/s wall from N=4), so the default UNDERSTATES production
scripts/bench/bench_mpas_spmd_scaling.py-21-scalability: at ``--nlev 26`` the per-cell compute grows ~3.25x, the flat wall
scripts/bench/bench_mpas_spmd_scaling.py-22-dissolves (~2.07+ Gcells/s, ~1.9x higher at 16 GPUs), and a size-dependent term
scripts/bench/bench_mpas_spmd_scaling.py-23-enters. Report the production curve at production thickness; nlev=8 is the
--
scripts/bench/bench_mpas_spmd_scaling.py-30-
scripts/bench/bench_mpas_spmd_scaling.py-31-Multi-controller (route-B, ``--multicontroller``): identical contract to the
scripts/bench/bench_mpas_spmd_scaling.py-32-lat-lon benches — every process calls ``jax.distributed.initialize`` BEFORE
scripts/bench/bench_mpas_spmd_scaling.py-33-any other JAX use, the ("device",) mesh is built over the GLOBAL
scripts/bench/bench_mpas_spmd_scaling.py-34-``jax.devices()``, and the existing ``make_voronoi_sharded_step`` ppermute
scripts/bench/bench_mpas_spmd_scaling.py:35:halo + the mass-fix psum run unchanged across processes (NCCL on GPU / gloo
scripts/bench/bench_mpas_spmd_scaling.py-36-on CPU). NO mpi4jax is armed in this mode (the documented mixed-stack
scripts/bench/bench_mpas_spmd_scaling.py-37-deadlock hazard). Every process computes the SAME reorder host-side; under
scripts/bench/bench_mpas_spmd_scaling.py-38-``--multicontroller`` the partition checksum is asserted equal across
scripts/bench/bench_mpas_spmd_scaling.py-39-processes (a rank-divergent partition — e.g. one rank resolving
scripts/bench/bench_mpas_spmd_scaling.py-40-``--partition-method auto`` to METIS and another to RCB — would silently
--
scripts/bench/bench_mpas_spmd_scaling.py-43-Launch (cluster, one process per GPU):
scripts/bench/bench_mpas_spmd_scaling.py-44-  srun -n 6 python bench_mpas_spmd_scaling.py --multicontroller \
scripts/bench/bench_mpas_spmd_scaling.py-45-      --n-devices 6 ...            # SLURM: coordinator auto-detected
scripts/bench/bench_mpas_spmd_scaling.py-46-  mpiexec -n 6 python ... --multicontroller --coordinator host0:9876
scripts/bench/bench_mpas_spmd_scaling.py-47-CPU smoke (single process, virtual devices):
scripts/bench/bench_mpas_spmd_scaling.py:48:  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
scripts/bench/bench_mpas_spmd_scaling.py-49-  JAX_ENABLE_X64=1 python scripts/bench/bench_mpas_spmd_scaling.py \
scripts/bench/bench_mpas_spmd_scaling.py-50-      --subdivision 3 --nlev 4 --n-devices 2 --steps 4 --parity-gate
scripts/bench/bench_mpas_spmd_scaling.py-51-"""
scripts/bench/bench_mpas_spmd_scaling.py-52-from __future__ import annotations
scripts/bench/bench_mpas_spmd_scaling.py-53-
--
scripts/bench/bench_mpas_spmd_scaling.py-427-    if physics_fn is not None:
scripts/bench/bench_mpas_spmd_scaling.py-428-        _census_fn = lambda st: step(st, dt, physics_fn=physics_fn)  # noqa: E731
scripts/bench/bench_mpas_spmd_scaling.py-429-    else:
scripts/bench/bench_mpas_spmd_scaling.py-430-        _census_fn = lambda st: step(st, dt)  # noqa: E731
scripts/bench/bench_mpas_spmd_scaling.py-431-    # ONE compile → full per-family census; the CP scalar (the #1113 round-count
scripts/bench/bench_mpas_spmd_scaling.py:432:    # wall) is the collective_permute member, so no second compile for it.
scripts/bench/bench_mpas_spmd_scaling.py-433-    hlo_census = hlo_collective_census(_census_fn, s)
scripts/bench/bench_mpas_spmd_scaling.py:434:    hlo_cp = hlo_census["collective_permute"] if hlo_census else None
scripts/bench/bench_mpas_spmd_scaling.py-435-
scripts/bench/bench_mpas_spmd_scaling.py-436-    # --- Correctness gates (before any timing is reported) -----------------
scripts/bench/bench_mpas_spmd_scaling.py-437-    if args.parity_gate or args.check_conservation:
scripts/bench/bench_mpas_spmd_scaling.py-438-        final_global = (gather_voronoi_state_spmd(s, dev_config)
scripts/bench/bench_mpas_spmd_scaling.py-439-                        if dev_config.n_devices > 1 else s)
--
scripts/bench/bench_mpas_spmd_scaling.py-517-        per_step_ms=[round(x, 1) for x in per_step_ms],
scripts/bench/bench_mpas_spmd_scaling.py-518-        cells=int(mesh.nCells) * args.nlev,
scripts/bench/bench_mpas_spmd_scaling.py-519-        # ppermute round count/step (static compile property; #1113) — the
scripts/bench/bench_mpas_spmd_scaling.py-520-        # multi-node ceiling is this count x the ~0.11 ms launch floor, so it
scripts/bench/bench_mpas_spmd_scaling.py-521-        # belongs on every row like the cube benches.
scripts/bench/bench_mpas_spmd_scaling.py:522:        hlo_collective_permutes=hlo_cp,
scripts/bench/bench_mpas_spmd_scaling.py-523-        # full per-family census (permute + all-reduce + all-gather + ...) on
scripts/bench/bench_mpas_spmd_scaling.py-524-        # the SAME compile: exposes any reduction the ico step introduces.
scripts/bench/bench_mpas_spmd_scaling.py-525-        hlo_collectives=hlo_census,
scripts/bench/bench_mpas_spmd_scaling.py-526-    )
scripts/bench/bench_mpas_spmd_scaling.py-527-    # Flat aggregator-compatible identity + metric fields (see the latlon
--
scripts/bench/bench_cube_shardmap_halo.py-1-#!/usr/bin/env python
scripts/bench/bench_cube_shardmap_halo.py-2-"""Blocker-1 decisive experiment: does the GSPMD ``shard_map`` cube-panel halo
scripts/bench/bench_cube_shardmap_halo.py-3-hold strong-scaling efficiency past the ``mpi4jax`` np6 anti-scale cliff?
scripts/bench/bench_cube_shardmap_halo.py-4-
scripts/bench/bench_cube_shardmap_halo.py-5-Context (``docs/performance/scream_parity_scope.md`` §5). The single question that
scripts/bench/bench_cube_shardmap_halo.py:6:gates SCREAM-parity feasibility is whether the native-NCCL GSPMD cube halo
scripts/bench/bench_cube_shardmap_halo.py-7-(``jax.lax.ppermute`` inside ``shard_map``) scales, unlike the ``mpi4jax``
scripts/bench/bench_cube_shardmap_halo.py-8-point-to-point path which is blocking, host-staged, and measured to *anti-scale*
scripts/bench/bench_cube_shardmap_halo.py-9-across cube faces on a TCP fabric (107→206 ms/step np1→6,
scripts/bench/bench_cube_shardmap_halo.py-10-``coupler/.../compiled_segments.py:484``).
scripts/bench/bench_cube_shardmap_halo.py-11-
--
scripts/bench/bench_cube_shardmap_halo.py-17-  step (the halo must move exactly the right data). The pytree structure, leaf
scripts/bench/bench_cube_shardmap_halo.py-18-  count, shape and dtype are all asserted equal first, so the check cannot pass
scripts/bench/bench_cube_shardmap_halo.py-19-  vacuously.
scripts/bench/bench_cube_shardmap_halo.py-20-* **AD** — a nonuniform-cotangent VJP through the SPMD ``ppermute`` halo matches
scripts/bench/bench_cube_shardmap_halo.py-21-  the single-device reference VJP, AND the lowered SPMD function is asserted to
scripts/bench/bench_cube_shardmap_halo.py:22:  contain a ``collective_permute`` — the ppermute kernel's signature (the
scripts/bench/bench_cube_shardmap_halo.py-23-  all_gather diagnostic kernel must NOT satisfy this) — proving the ppermute path
scripts/bench/bench_cube_shardmap_halo.py:24:  ran, not a silent local fallback. This is the ``collective_permute`` VJP
scripts/bench/bench_cube_shardmap_halo.py-25-  correctness that Blocker 1 depends on (CLAUDE.md MPI-AD doctrine).
scripts/bench/bench_cube_shardmap_halo.py-26-* **Strong scaling** — steady-state ms/step at fixed global cube size, sweeping
scripts/bench/bench_cube_shardmap_halo.py-27-  device count (1 → 2 → 3 → 6, and ``6·kt²`` tiled if enough devices), reported as
scripts/bench/bench_cube_shardmap_halo.py-28-  speed-up and parallel efficiency vs the single-device baseline.
scripts/bench/bench_cube_shardmap_halo.py-29-
scripts/bench/bench_cube_shardmap_halo.py-30-Pass condition (Blocker 1 clears its gate): the GSPMD cube halo holds
scripts/bench/bench_cube_shardmap_halo.py-31-**≥ ``--pass-efficiency`` strong-scaling efficiency at the largest device count**.
scripts/bench/bench_cube_shardmap_halo.py-32-On a local macOS/CPU box (Metal is unusable — see memory) this validates
scripts/bench/bench_cube_shardmap_halo.py-33-correctness + AD + the ``shard_map`` plumbing on emulated devices and yields a CPU
scripts/bench/bench_cube_shardmap_halo.py:34:scaling proxy; the real GPU numbers come from the Derecho 2-node NCCL job. The
scripts/bench/bench_cube_shardmap_halo.py-35-``--gate-efficiency`` flag makes this a *decisive* run: the efficiency threshold
scripts/bench/bench_cube_shardmap_halo.py-36-becomes a hard exit gate AND at least one device count > 1 is required (a
scripts/bench/bench_cube_shardmap_halo.py-37-single-device run can never be decisive). Left off, the harness is a
scripts/bench/bench_cube_shardmap_halo.py-38-correctness/AD smoke.
scripts/bench/bench_cube_shardmap_halo.py-39-
scripts/bench/bench_cube_shardmap_halo.py-40-Local (emulated 6 devices)::
scripts/bench/bench_cube_shardmap_halo.py-41-
scripts/bench/bench_cube_shardmap_halo.py:42:    XLA_FLAGS=--xla_force_host_platform_device_count=6 \
scripts/bench/bench_cube_shardmap_halo.py-43-      .venv/bin/python scripts/bench/bench_cube_shardmap_halo.py \
scripts/bench/bench_cube_shardmap_halo.py-44-      --device-counts 1,2,3,6 --n-grid 24 --output-dir results/blocker1
scripts/bench/bench_cube_shardmap_halo.py-45-
scripts/bench/bench_cube_shardmap_halo.py:46:Real hardware (2-node NCCL, via scripts/cluster/scaling_derecho/): add
scripts/bench/bench_cube_shardmap_halo.py-47-``--gate-efficiency --pass-efficiency 0.6`` and launch under ``jax.distributed``.
scripts/bench/bench_cube_shardmap_halo.py-48-"""
scripts/bench/bench_cube_shardmap_halo.py-49-
scripts/bench/bench_cube_shardmap_halo.py-50-from __future__ import annotations
scripts/bench/bench_cube_shardmap_halo.py-51-
--
scripts/bench/bench_cube_shardmap_halo.py-130-_ALLGATHER_RE = re.compile(r"all[_-]gather")
scripts/bench/bench_cube_shardmap_halo.py-131-
scripts/bench/bench_cube_shardmap_halo.py-132-
scripts/bench/bench_cube_shardmap_halo.py-133-def _hlo_has_ppermute(fn, *args) -> bool | None:
scripts/bench/bench_cube_shardmap_halo.py-134-    """Best-effort: does the lowered HLO of ``fn(*args)`` contain a
scripts/bench/bench_cube_shardmap_halo.py:135:    ``collective_permute`` — the signature of the *ppermute* cube-halo kernel?
scripts/bench/bench_cube_shardmap_halo.py-136-
scripts/bench/bench_cube_shardmap_halo.py-137-    Deliberately matches ONLY ``collective[_-]permute`` (StableHLO underscore +
scripts/bench/bench_cube_shardmap_halo.py-138-    optimized-XLA hyphen), NOT ``all_gather``: this experiment is about the
scripts/bench/bench_cube_shardmap_halo.py-139-    bandwidth-optimal ppermute path; the all_gather diagnostic kernel replicates
scripts/bench/bench_cube_shardmap_halo.py-140-    all compute and must never satisfy the proof. If ``all_gather`` is found
--
scripts/bench/bench_cube_shardmap_halo.py-349-    """VJP of the cube halo with a NONUNIFORM cotangent; SPMD must match local.
scripts/bench/bench_cube_shardmap_halo.py-350-
scripts/bench/bench_cube_shardmap_halo.py-351-    A uniform loss like ``sum(pad(x)**2)`` is too symmetric — a wrong permutation
scripts/bench/bench_cube_shardmap_halo.py-352-    with equal copy-multiplicity yields the same gradient. Here the cotangent
scripts/bench/bench_cube_shardmap_halo.py-353-    ``w`` is unique per output cell, so any mis-routed strip changes the adjoint.
scripts/bench/bench_cube_shardmap_halo.py:354:    The lowered SPMD function is also asserted to contain a ``collective_permute``
scripts/bench/bench_cube_shardmap_halo.py-355-    (the ppermute kernel's signature), proving the ppermute path ran rather than a
scripts/bench/bench_cube_shardmap_halo.py-356-    silent local fallback that would trivially match. The whole routine snapshots
scripts/bench/bench_cube_shardmap_halo.py-357-    and restores the caller's prior halo backend so neither the forced-local
scripts/bench/bench_cube_shardmap_halo.py-358-    reference nor the SPMD activation leaks.
scripts/bench/bench_cube_shardmap_halo.py-359-    """
--
scripts/bench/bench_cube_shardmap_halo.py-480-            ad_pass = False
scripts/bench/bench_cube_shardmap_halo.py-481-            logger.error("AD gate raised (Blocker-1 AD risk): %r", exc)
scripts/bench/bench_cube_shardmap_halo.py-482-    else:
scripts/bench/bench_cube_shardmap_halo.py-483-        logger.warning(
scripts/bench/bench_cube_shardmap_halo.py-484-            "AD gate SKIPPED — only 1 device available; run with "
scripts/bench/bench_cube_shardmap_halo.py:485:            "XLA_FLAGS=--xla_force_host_platform_device_count>=2 to exercise it.")
scripts/bench/bench_cube_shardmap_halo.py-486-
scripts/bench/bench_cube_shardmap_halo.py-487-    # --- Efficiency gate (hard only with --gate-efficiency) ---------------
scripts/bench/bench_cube_shardmap_halo.py-488-    eff_at_largest = largest_row["efficiency"]
scripts/bench/bench_cube_shardmap_halo.py-489-    efficiency_pass = (not decisive) or (
scripts/bench/bench_cube_shardmap_halo.py-490-        multi_device_ran and eff_at_largest >= args.pass_efficiency)
--
scripts/bench/bench_cube_shardmap_halo.py-503-    reasons = []
scripts/bench/bench_cube_shardmap_halo.py-504-    if decisive and not multi_device_ran:
scripts/bench/bench_cube_shardmap_halo.py-505-        reasons.append(
scripts/bench/bench_cube_shardmap_halo.py-506-            "decisive run (--gate-efficiency) requires >=1 device count > 1; "
scripts/bench/bench_cube_shardmap_halo.py-507-            f"only single-device ran (available={available}). Set "
scripts/bench/bench_cube_shardmap_halo.py:508:            "XLA_FLAGS=--xla_force_host_platform_device_count / launch multi-GPU.")
scripts/bench/bench_cube_shardmap_halo.py-509-    if decisive and multi_device_ran and not ad_ran:
scripts/bench/bench_cube_shardmap_halo.py-510-        reasons.append("decisive run requires the AD gate to execute")
scripts/bench/bench_cube_shardmap_halo.py-511-
scripts/bench/bench_cube_shardmap_halo.py-512-    gates = {"correctness": correctness_pass, "ad": ad_pass,
scripts/bench/bench_cube_shardmap_halo.py-513-             "efficiency": efficiency_pass, "timed_ppermute": timed_ppermute_pass}
--
scripts/bench/analyze_scaling_results.py-439-            "potential_speedup_pct": 10,
scripts/bench/analyze_scaling_results.py-440-            "actions": [
scripts/bench/analyze_scaling_results.py-441-                "Check message sizes: very small messages have high per-message overhead",
scripts/bench/analyze_scaling_results.py-442-                "Pack multiple edges into single buffer per neighbor rank",
scripts/bench/analyze_scaling_results.py-443-                "Verify GPU direct RDMA is active (not routing through host memory)",
scripts/bench/analyze_scaling_results.py:444:                "Check NCCL vs UCX transport selection",
scripts/bench/analyze_scaling_results.py-445-            ],
scripts/bench/analyze_scaling_results.py-446-        })
scripts/bench/analyze_scaling_results.py-447-
scripts/bench/analyze_scaling_results.py-448-    # Compilation cost
scripts/bench/analyze_scaling_results.py-449-    comp = analyses.get("compilation", {})
--
scripts/bench/slurm_scaling_diagnosis.sh-102-# don't have to edit this script — the documented override now
scripts/bench/slurm_scaling_diagnosis.sh-103-# actually works.
scripts/bench/slurm_scaling_diagnosis.sh-104-export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
scripts/bench/slurm_scaling_diagnosis.sh-105-export XLA_PYTHON_CLIENT_PREALLOCATE="false"
scripts/bench/slurm_scaling_diagnosis.sh-106-export XLA_PYTHON_CLIENT_MEM_FRACTION="0.90"
scripts/bench/slurm_scaling_diagnosis.sh:107:export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"
scripts/bench/slurm_scaling_diagnosis.sh-108-
scripts/bench/slurm_scaling_diagnosis.sh-109-if [ "$PRECISION" = "float64" ]; then
scripts/bench/slurm_scaling_diagnosis.sh-110-    export JAX_ENABLE_X64=1
scripts/bench/slurm_scaling_diagnosis.sh-111-fi
scripts/bench/slurm_scaling_diagnosis.sh-112-
--
scripts/bench/slurm_scaling_diagnosis.sh-118-# ``_configure_env`` (which reads ``SLURM_LOCALID`` per-rank).
scripts/bench/slurm_scaling_diagnosis.sh-119-# Exporting ``CUDA_VISIBLE_DEVICES`` here at wrapper scope binds
scripts/bench/slurm_scaling_diagnosis.sh-120-# *every* rank to the launcher's local id (typically 0), pinning all
scripts/bench/slurm_scaling_diagnosis.sh-121-# ranks to GPU 0 and serialising the run.
scripts/bench/slurm_scaling_diagnosis.sh-122-
scripts/bench/slurm_scaling_diagnosis.sh:123:# NCCL tuning for Levante
scripts/bench/slurm_scaling_diagnosis.sh:124:export NCCL_DEBUG=WARN
scripts/bench/slurm_scaling_diagnosis.sh:125:export NCCL_IB_DISABLE=0
scripts/bench/slurm_scaling_diagnosis.sh:126:export NCCL_NET_GDR_LEVEL=5
scripts/bench/slurm_scaling_diagnosis.sh-127-
scripts/bench/slurm_scaling_diagnosis.sh-128-# ---------------------------------------------------------------------------
scripts/bench/slurm_scaling_diagnosis.sh-129-# Build command
scripts/bench/slurm_scaling_diagnosis.sh-130-# ---------------------------------------------------------------------------
scripts/bench/slurm_scaling_diagnosis.sh-131-DIAG_CMD="python scripts/bench/run_scaling_diagnosis.py \
--
scripts/bench/bench_spectral_les_dd_scaling.py-14-
scripts/bench/bench_spectral_les_dd_scaling.py-15-    export PATH="$HOME/.local/mpich/bin:$PATH" \
scripts/bench/bench_spectral_les_dd_scaling.py-16-           LD_LIBRARY_PATH="$HOME/.local/mpich/lib:$LD_LIBRARY_PATH" \
scripts/bench/bench_spectral_les_dd_scaling.py-17-           PYTHONPATH="$PWD/packages/core:$PWD/packages/atmosphere:$PWD" \
scripts/bench/bench_spectral_les_dd_scaling.py-18-           JAX_PLATFORMS=cpu MPI4JAX_NO_WARN_JAX_VERSION=1 \
scripts/bench/bench_spectral_les_dd_scaling.py:19:           XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" OMP_NUM_THREADS=1
scripts/bench/bench_spectral_les_dd_scaling.py-20-    mpirun -np 4 .venv-mpi/bin/python scripts/bench/bench_spectral_les_dd_scaling.py \
scripts/bench/bench_spectral_les_dd_scaling.py-21-        --mode strong --precision float64 --nx 64 --ny 64 --nz 32 --steps 30
scripts/bench/bench_spectral_les_dd_scaling.py-22-
scripts/bench/bench_spectral_les_dd_scaling.py-23-CSV columns: mode,precision,n_ranks,ny_global,nx,nz,ny_local,wall_s,ms_per_step.
scripts/bench/bench_spectral_les_dd_scaling.py-24-"""
--
scripts/cluster/scaling_levante/_env.sh-96-export OMPI_MCA_osc="${OMPI_MCA_osc:-ucx}"
scripts/cluster/scaling_levante/_env.sh-97-export UCX_TLS="${UCX_TLS:-rc,cuda_copy,cuda_ipc,gdr_copy,sm,self}"
scripts/cluster/scaling_levante/_env.sh-98-export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"
scripts/cluster/scaling_levante/_env.sh-99-export UCX_RNDV_SCHEME="${UCX_RNDV_SCHEME:-put_zcopy}"
scripts/cluster/scaling_levante/_env.sh-100-
scripts/cluster/scaling_levante/_env.sh:101:# --- NCCL over InfiniBand (route-B: jax.distributed multi-node lanes) --------
scripts/cluster/scaling_levante/_env.sh:102:# NCCL (shard_map/ppermute collectives under jax.distributed) uses its own
scripts/cluster/scaling_levante/_env.sh-103-# IB-verbs stack — independent of the UCX/MPI settings above; the two configs
scripts/cluster/scaling_levante/_env.sh-104-# coexist. Bootstrap ring runs over IPoIB: verify the interface name once with
scripts/cluster/scaling_levante/_env.sh:105:# `ip addr` on a gpu node (a wrong NCCL_SOCKET_IFNAME is the #1 cause of
scripts/cluster/scaling_levante/_env.sh:106:# multi-node NCCL bootstrap timeouts on IB clusters).
scripts/cluster/scaling_levante/_env.sh:107:export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
scripts/cluster/scaling_levante/_env.sh:108:export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
scripts/cluster/scaling_levante/_env.sh-109-# Prefix-match BOTH HCAs (mlx5_0/mlx5_1 — one per socket on Levante nodes).
scripts/cluster/scaling_levante/_env.sh:110:export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
scripts/cluster/scaling_levante/_env.sh-111-# GPUDirect RDMA when NIC and GPU share a NUMA/PCIe root.
scripts/cluster/scaling_levante/_env.sh:112:export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
scripts/cluster/scaling_levante/_env.sh:113:export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
scripts/cluster/scaling_levante/_env.sh-114-
scripts/cluster/scaling_levante/_env.sh-115-# --- XLA overlap defaults for the lat-lon SPMD lanes (2026-08-04) --------
scripts/cluster/scaling_levante/_env.sh-116-# Latency-hiding scheduler + pipelined p2p: -8.4% at LL2048@64 (job
scripts/cluster/scaling_levante/_env.sh-117-# 26677602) and -8.3% at @128 (26677668), A/A2 drift 0.3-0.4% — twice-
scripts/cluster/scaling_levante/_env.sh-118-# reproduced, parity suites green with flags on. MPAS lane: null (0.0%,
--
scripts/cluster/scaling_levante/_env.sh-120-# Below the pre-registered 10% bar AND other lanes are unvalidated
scripts/cluster/scaling_levante/_env.sh-121-# (cube_tiled_step.sbatch force-disables latency hiding for a known
scripts/cluster/scaling_levante/_env.sh-122-# comm-init sensitivity; MPAS is null) — so this is strictly OPT-IN
scripts/cluster/scaling_levante/_env.sh-123-# (codex r23): set LEGOESM_XLA_OVERLAP=1 in validated lat-lon
scripts/cluster/scaling_levante/_env.sh-124-# launchers; never a shared default, and A/B control arms must keep
scripts/cluster/scaling_levante/_env.sh:125:# REPLACING XLA_FLAGS, not appending.
scripts/cluster/scaling_levante/_env.sh-126-if [ "${LEGOESM_XLA_OVERLAP:-0}" = 1 ]; then
scripts/cluster/scaling_levante/_env.sh:127:  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_levante/_env.sh-128-fi
scripts/cluster/scaling_levante/_env.sh-129-
scripts/cluster/scaling_levante/_env.sh-130-export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
scripts/cluster/scaling_levante/_env.sh-131-mkdir -p "$TMPDIR" 2>/dev/null || true
scripts/cluster/scaling_levante/_env.sh-132-
--
scripts/bench/probe_spectral_shard.py-2-on emulated multi-CPU devices actually scales.  Bypasses
scripts/bench/probe_spectral_shard.py-3-``model.step`` (which has ``dt`` as a static arg and conflicts with
scripts/bench/probe_spectral_shard.py-4-outer JIT) by jitting ``model._do_step`` directly with primed caches.
scripts/bench/probe_spectral_shard.py-5-
scripts/bench/probe_spectral_shard.py-6-Use:
scripts/bench/probe_spectral_shard.py:7:    JAX_ENABLE_X64=1 XLA_FLAGS=--xla_force_host_platform_device_count=N \\
scripts/bench/probe_spectral_shard.py-8-        PYTHONPATH=. .venv/bin/python scripts/probe_spectral_shard.py
scripts/bench/probe_spectral_shard.py-9-"""
scripts/bench/probe_spectral_shard.py-10-
scripts/bench/probe_spectral_shard.py-11-from __future__ import annotations
scripts/bench/probe_spectral_shard.py-12-
--
scripts/bench/probe_spectral_shard.py-26-    parser.add_argument("--n-steps", type=int, default=100)
scripts/bench/probe_spectral_shard.py-27-    parser.add_argument(
scripts/bench/probe_spectral_shard.py-28-        "--shard-axis", choices=["none", "level"], default="level",
scripts/bench/probe_spectral_shard.py-29-        help="Sharding axis: none = single-device baseline, level = "
scripts/bench/probe_spectral_shard.py-30-             "shard along the vertical level axis using the JAX device "
scripts/bench/probe_spectral_shard.py:31:             "mesh exposed via XLA_FLAGS.",
scripts/bench/probe_spectral_shard.py-32-    )
scripts/bench/probe_spectral_shard.py-33-    parser.add_argument(
scripts/bench/probe_spectral_shard.py-34-        "--scan-steps", type=int, default=1,
scripts/bench/probe_spectral_shard.py-35-        help="If >1, fuse this many ``_do_step`` calls inside a single "
scripts/bench/probe_spectral_shard.py-36-             "``jax.lax.scan`` body (combines the iter-204 scan-steps "
--
scripts/bench/bench_ocean_latlon_spmd_pcg.py-30-Usage (compute node only; never the login node):
scripts/bench/bench_ocean_latlon_spmd_pcg.py-31-  GPU (route-B, the real target):
scripts/bench/bench_ocean_latlon_spmd_pcg.py-32-    JAX_ENABLE_X64=1 python scripts/bench/bench_ocean_latlon_spmd_pcg.py \\
scripts/bench/bench_ocean_latlon_spmd_pcg.py-33-        --device gpu --device-counts 1,2 --n-lat 360 --n-lon 720
scripts/bench/bench_ocean_latlon_spmd_pcg.py-34-  CPU smoke (virtual devices):
scripts/bench/bench_ocean_latlon_spmd_pcg.py:35:    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_ENABLE_X64=1 \\
scripts/bench/bench_ocean_latlon_spmd_pcg.py-36-      python scripts/bench/bench_ocean_latlon_spmd_pcg.py \\
scripts/bench/bench_ocean_latlon_spmd_pcg.py-37-        --device cpu --device-counts 1,2,4 --n-lat 64 --n-lon 128
scripts/bench/bench_ocean_latlon_spmd_pcg.py-38-"""
scripts/bench/bench_ocean_latlon_spmd_pcg.py-39-from __future__ import annotations
scripts/bench/bench_ocean_latlon_spmd_pcg.py-40-
--
scripts/bench/bench_ocean_latlon_spmd_pcg.py-58-        # Two devices share the node; do not let the allocator grab it all.
scripts/bench/bench_ocean_latlon_spmd_pcg.py-59-        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
scripts/bench/bench_ocean_latlon_spmd_pcg.py-60-    else:
scripts/bench/bench_ocean_latlon_spmd_pcg.py-61-        os.environ["JAX_PLATFORMS"] = "cpu"
scripts/bench/bench_ocean_latlon_spmd_pcg.py-62-        os.environ.setdefault(
scripts/bench/bench_ocean_latlon_spmd_pcg.py:63:            "XLA_FLAGS",
scripts/bench/bench_ocean_latlon_spmd_pcg.py-64-            f"--xla_force_host_platform_device_count={n_cpu_devices}",
scripts/bench/bench_ocean_latlon_spmd_pcg.py-65-        )
scripts/bench/bench_ocean_latlon_spmd_pcg.py-66-
scripts/bench/bench_ocean_latlon_spmd_pcg.py-67-
scripts/bench/bench_ocean_latlon_spmd_pcg.py-68-def _build_ops(coeff: float):
--
scripts/bench/metadata.py-29-from datetime import datetime, timezone
scripts/bench/metadata.py-30-from typing import Any
scripts/bench/metadata.py-31-
scripts/bench/metadata.py-32-#: Bump when the record shape changes so aggregators can branch on it.
scripts/bench/metadata.py-33-#: v2: + ``transport`` / ``virtual_cpu_devices`` / ``launcher`` (anti-fake-
scripts/bench/metadata.py:34:#: scaling audit) — a route-A mpi4jax row, a route-B NCCL row, a gloo/TCP
scripts/bench/metadata.py-35-#: fabric row, and a CPU-virtual-device infra proxy are now distinguishable
scripts/bench/metadata.py-36-#: from the record alone.
scripts/bench/metadata.py-37-METADATA_SCHEMA_VERSION = 2
scripts/bench/metadata.py-38-
scripts/bench/metadata.py-39-#: Halo/collective transports a scaling row may run on.  ``xla-local`` =
scripts/bench/metadata.py-40-#: single-process multi-device SPMD (intra-process XLA collectives);
scripts/bench/metadata.py-41-#: ``none`` = serial.  gloo/TCP multi-node is KNOWN anti-scaling fabric
scripts/bench/metadata.py-42-#: (docs/performance/scaling/spectral_level_shard_cliff.md) — recording it
scripts/bench/metadata.py:43:#: verbatim is what keeps such a row from masquerading as an NCCL result.
scripts/bench/metadata.py-44-TRANSPORTS: tuple[str, ...] = ("mpi4jax", "nccl", "gloo", "xla-local", "none")
scripts/bench/metadata.py-45-
scripts/bench/metadata.py-46-#: Env knobs that change a run's numerics / comparability.  Recorded VERBATIM so
scripts/bench/metadata.py-47-#: an f32 (or TF32) ablation is never silently compared to an f64 baseline.
scripts/bench/metadata.py-48-PRECISION_ENV_KNOBS: tuple[str, ...] = (
--
scripts/bench/metadata.py-98-def _env_flag_true(name: str) -> bool:
scripts/bench/metadata.py-99-    """True iff env var ``name`` is a truthy flag ("1"/"true"/"yes"/"on")."""
scripts/bench/metadata.py-100-    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")
scripts/bench/metadata.py-101-
scripts/bench/metadata.py-102-
scripts/bench/metadata.py:103:# Match the OP-CALL form ``collective-permute(`` / ``collective_permute(`` /
scripts/bench/metadata.py-104-# ``...-start(`` (a paren directly after the op name), NOT bare substrings: the
scripts/bench/metadata.py:105:# COMPILED-HLO config header echoes XLA_FLAGS, so a flag name like
scripts/bench/metadata.py:106:# ``xla_gpu_collective_permute_combine_threshold_bytes=`` (set by #1175's
scripts/bench/metadata.py-107-# MPAS_CP_COMBINE) would false-match a plain substring scan and over-count.
scripts/bench/metadata.py-108-_COLLECTIVE_PERMUTE_RE = re.compile(r"collective[_-]permute(?:[_-]start)?\(")
scripts/bench/metadata.py-109-
scripts/bench/metadata.py-110-
scripts/bench/metadata.py-111-def _count_op_calls(hlo_text: str, rx: "re.Pattern[str]") -> int:
--
scripts/bench/metadata.py-116-    the op name is ``-``/``_``, not ``(`` or a ``start`` suffix) — so no
scripts/bench/metadata.py-117-    substring ``"done" not in line`` filter is used: that filter would
scripts/bench/metadata.py-118-    false-drop a legitimate collective whose line merely CONTAINS "done"
scripts/bench/metadata.py-119-    elsewhere (an XLA ``metadata={op_name="…/done_stage/…"}`` tag, a
scripts/bench/metadata.py-120-    ``%done_mass`` SSA name).  The ``\\(`` op-call anchor still keeps a
scripts/bench/metadata.py:121:    config-header ``XLA_FLAGS`` echo (a flag name, no paren) from inflating."""
scripts/bench/metadata.py-122-    return sum(1 for line in hlo_text.splitlines() if rx.search(line))
scripts/bench/metadata.py-123-
scripts/bench/metadata.py-124-
scripts/bench/metadata.py:125:def count_collective_permutes(hlo_text: str) -> int:
scripts/bench/metadata.py:126:    """Count ``collective_permute`` OPS in a lowered/compiled HLO text dump.
scripts/bench/metadata.py-127-
scripts/bench/metadata.py-128-    The ppermute halo kernel's signature and a STATIC compile property (the
scripts/bench/metadata.py-129-    count is fixed by the partition/edge-coloring schedule, not the data), so
scripts/bench/metadata.py-130-    it is the round-count metric #1113 needs to decompose multi-node overhead.
scripts/bench/metadata.py-131-    Matches the op-call form only (StableHLO underscore + optimized-XLA hyphen,
scripts/bench/metadata.py-132-    async ``-start`` counted once, ``-done`` companion excluded by the regex),
scripts/bench/metadata.py:133:    so config-header flag names that merely CONTAIN "collective_permute" never
scripts/bench/metadata.py-134-    inflate the count.  Canonical for every bench that reports
scripts/bench/metadata.py:135:    ``hlo_collective_permutes`` (cube tiled, MPAS ico) — no re-implementation."""
scripts/bench/metadata.py-136-    return _count_op_calls(hlo_text, _COLLECTIVE_PERMUTE_RE)
scripts/bench/metadata.py-137-
scripts/bench/metadata.py-138-
scripts/bench/metadata.py:139:def hlo_collective_permutes(fn, *args) -> int | None:
scripts/bench/metadata.py-140-    """Best-effort: count the collective-permutes in the COMPILED HLO of
scripts/bench/metadata.py-141-    ``fn(*args)``.
scripts/bench/metadata.py-142-
scripts/bench/metadata.py-143-    Compiles (``.lower(...).compile().as_text()``), NOT bare
scripts/bench/metadata.py-144-    ``.lower().as_text()``: the census must reflect the EXECUTABLE's round
--
scripts/bench/metadata.py-150-    the backend's ``as_text()`` yields no HLO, so a timing probe can record
scripts/bench/metadata.py-151-    "unknown" rather than crash."""
scripts/bench/metadata.py-152-    import jax
scripts/bench/metadata.py-153-    try:
scripts/bench/metadata.py-154-        text = jax.jit(fn).lower(*args).compile().as_text()
scripts/bench/metadata.py:155:        return count_collective_permutes(text) if text else None
scripts/bench/metadata.py-156-    except Exception:
scripts/bench/metadata.py-157-        return None
scripts/bench/metadata.py-158-
scripts/bench/metadata.py-159-
scripts/bench/metadata.py-160-def _op_call_re(op_name: str) -> "re.Pattern[str]":
scripts/bench/metadata.py-161-    """Op-call-form matcher for a single HLO collective ``op_name``.
scripts/bench/metadata.py-162-
scripts/bench/metadata.py-163-    ``op_name`` is the hyphen spelling (``"all-reduce"``).  Matches BOTH the
scripts/bench/metadata.py-164-    optimized-XLA hyphen and StableHLO underscore spellings, an optional async
scripts/bench/metadata.py-165-    ``-start``/``_start`` suffix, and requires the ``(`` op-call form so a
scripts/bench/metadata.py:166:    config-header ``XLA_FLAGS`` echo that merely CONTAINS the op name can never
scripts/bench/metadata.py-167-    inflate the count (same guard as :data:`_COLLECTIVE_PERMUTE_RE`)."""
scripts/bench/metadata.py-168-    stem = op_name.replace("-", "[_-]")
scripts/bench/metadata.py-169-    return re.compile(stem + r"(?:[_-]start)?\(")
scripts/bench/metadata.py-170-
scripts/bench/metadata.py-171-
scripts/bench/metadata.py-172-#: Every collective OP family a scaling row can run.  ``collective-permute`` is
scripts/bench/metadata.py-173-#: the band/face halo (reuse the canonical permute regex so its count stays
scripts/bench/metadata.py:174:#: bit-identical to :func:`count_collective_permutes`); ``all-reduce`` is the
scripts/bench/metadata.py-175-#: conservation fixer AND the ocean implicit-CN PCG reduction wall (~120/step —
scripts/bench/metadata.py-176-#: the #1 ocean strong-scaling bottleneck, invisible to a permute-only census);
scripts/bench/metadata.py-177-#: the rest surface any SPMD resharding an operator introduces.
scripts/bench/metadata.py-178-_COLLECTIVE_OP_RES: dict[str, "re.Pattern[str]"] = {
scripts/bench/metadata.py:179:    "collective_permute": _COLLECTIVE_PERMUTE_RE,
scripts/bench/metadata.py-180-    "all_reduce": _op_call_re("all-reduce"),
scripts/bench/metadata.py-181-    "all_gather": _op_call_re("all-gather"),
scripts/bench/metadata.py-182-    "all_to_all": _op_call_re("all-to-all"),
scripts/bench/metadata.py-183-    "reduce_scatter": _op_call_re("reduce-scatter"),
scripts/bench/metadata.py-184-}
scripts/bench/metadata.py-185-
scripts/bench/metadata.py-186-
scripts/bench/metadata.py-187-def count_collectives(hlo_text: str) -> dict[str, int]:
scripts/bench/metadata.py-188-    """Full per-family collective census of a lowered/compiled HLO text dump.
scripts/bench/metadata.py-189-
scripts/bench/metadata.py:190:    Superset of :func:`count_collective_permutes`: adds ``all_reduce`` (the
scripts/bench/metadata.py-191-    reduction wall that dominates ocean implicit-CN strong scaling and hides
scripts/bench/metadata.py-192-    from a permute-only count), ``all_gather``, ``all_to_all``,
scripts/bench/metadata.py-193-    ``reduce_scatter``.  Same STATIC, op-call-form discipline (see
scripts/bench/metadata.py-194-    :func:`_count_op_calls`: config-header flag names never inflate; an async
scripts/bench/metadata.py-195-    collective counts once via ``-start`` with its ``-done`` companion excluded
--
scripts/bench/metadata.py-203-
scripts/bench/metadata.py-204-
scripts/bench/metadata.py-205-def hlo_collective_census(fn, *args) -> dict[str, int] | None:
scripts/bench/metadata.py-206-    """Best-effort full collective census of the COMPILED HLO of ``fn(*args)``.
scripts/bench/metadata.py-207-
scripts/bench/metadata.py:208:    Superset of :func:`hlo_collective_permutes` — compiles (so combined /
scripts/bench/metadata.py-209-    pipelined collectives are reflected as executed, not as emitted) and runs
scripts/bench/metadata.py-210-    :func:`count_collectives`.  Returns ``None`` (never raises) if
scripts/bench/metadata.py-211-    lowering/compilation is unsupported OR the backend's ``as_text()`` yields
scripts/bench/metadata.py-212-    no HLO, so a probe records "unknown" not a crash.
scripts/bench/metadata.py-213-
scripts/bench/metadata.py-214-    NOTE the count is device-count- and FLAG-dependent: the DEFAULT schedule is
scripts/bench/metadata.py-215-    reproducible from a CPU virtual-device compile, but GPU-only XLA collective
scripts/bench/metadata.py:216:    combining / pipelined-p2p (``--xla_gpu_collective_permute_combine_*``, lane

exec
/usr/bin/bash -lc "nl -ba scripts/bench/bench_mpas_spmd_scaling.py | sed -n '177,520p'
nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '2480,2670p'
rg -n \"MPAS_CP_COMBINE|collective_permute_combine|pipelined_p2p|latency_hiding|XLA_FLAGS\" scripts/cluster/scaling_levante -S" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 147ms:
   177	def main() -> int:
   178	    p = argparse.ArgumentParser()
   179	    p.add_argument("--subdivision", type=int, default=5,
   180	                   help="icosahedral subdivision level L "
   181	                        "(nCells = 10*4^L + 2 before ghost padding)")
   182	    p.add_argument("--nlev", type=int, default=8)
   183	    p.add_argument("--lloyd", type=int, default=50,
   184	                   help="Lloyd relaxation iterations for the mesh. 50 = "
   185	                        "production SCVT; 0 = labelled synthetic scaling "
   186	                        "mesh (scaling receipts only, never physics — "
   187	                        "must match the prewarmed cache key at subdiv>=9).")
   188	    p.add_argument("--n-devices", type=int, required=True)
   189	    p.add_argument("--reorder-for", type=int, default=None,
   190	                   help="partition/reorder the mesh for THIS device count "
   191	                        "(default: --n-devices). Pin it to the ladder's "
   192	                        "max so single-device reference runs time the "
   193	                        "identical reordered mesh.")
   194	    p.add_argument("--partition-method",
   195	                   choices=["auto", "geometric", "metis", "sfc"],
   196	                   default="auto")
   197	    p.add_argument("--physics", choices=["none", "held_suarez", "kessler"],
   198	                   default="none",
   199	                   help="Operator-split physics on the timed path. "
   200	                        "'kessler' also attaches the q_v/q_c/q_r moist-"
   201	                        "baroclinic-wave tracers (packed tracer halo "
   202	                        "exchange + RK tracer advection on the gated "
   203	                        "path) and extends the parity gate to the "
   204	                        "tracer fields.")
   205	    p.add_argument("--halo-strategy",
   206	                   choices=["auto", "ppermute", "allgather"],
   207	                   default="auto",
   208	                   help="Halo strategy for make_voronoi_sharded_step. "
   209	                        "'auto' picks allgather below the per-device "
   210	                        "cell threshold — force 'ppermute' to exercise "
   211	                        "the neighbor-round schedule on small gate "
   212	                        "meshes (the multicontroller selfspawn tests "
   213	                        "do).  Recorded in the JSONL row.")
   214	    p.add_argument("--steps", type=int, default=12)
   215	    p.add_argument("--warmup", type=int, default=2)
   216	    p.add_argument("--dt", type=float, default=None,
   217	                   help="timestep [s]; default auto: 600 * 4**(4-L) "
   218	                        "(CFL: dx halves per level), min 30 s.")
   219	    p.add_argument("--out", type=str,
   220	                   default="results/a1/mpas_spmd_scaling.jsonl")
   221	    p.add_argument(
   222	        "--parity-gate", action="store_true",
   223	        help="Correctness gate: compare the gathered sharded trajectory "
   224	             "against the single-device model.step trajectory on the SAME "
   225	             "reordered mesh (smoke windows only; the re-association floor "
   226	             "grows with steps).")
   227	    p.add_argument(
   228	        "--check-conservation", action="store_true",
   229	        help="Gate global dry-mass drift sum(p_s*areaCell) over the run "
   230	             "(pre-shard state vs gathered final state; exits nonzero on "
   231	             "breach).")
   232	    p.add_argument("--mass-rtol", type=float, default=None,
   233	                   help="Conservation tolerance (default: 1e-11 f64 / "
   234	                        "1e-5 f32 — fix_mass pins the mass each step).")
   235	    p.add_argument("--multicontroller", action="store_true",
   236	                   help="Route-B multi-controller: jax.distributed.initialize "
   237	                        "per process, ('device',) mesh over the GLOBAL device "
   238	                        "set (one process per GPU / per CPU-device group). NO "
   239	                        "mpi4jax. --n-devices must equal the global device "
   240	                        "count.")
   241	    p.add_argument("--coordinator", type=str, default=None,
   242	                   help="host:port for jax.distributed when auto-detection "
   243	                        "(SLURM) is unavailable; process count/id then come "
   244	                        "from OMPI_COMM_WORLD_SIZE/RANK.")
   245	    args = p.parse_args()
   246	
   247	    # Validate the timing window BEFORE any model/device work (codex, ocean
   248	    # twin): an empty steady slice would only fail after the expensive run.
   249	    if args.steps < 1:
   250	        raise SystemExit(f"--steps must be >= 1, got {args.steps}")
   251	    if not (0 <= args.warmup < args.steps):
   252	        raise SystemExit(
   253	            f"--warmup must satisfy 0 <= warmup < steps "
   254	            f"(got warmup={args.warmup}, steps={args.steps})")
   255	    if args.parity_gate and args.steps > MPAS_PARITY_MAX_STEPS:
   256	        raise SystemExit(
   257	            f"--parity-gate is a smoke gate (re-association floor grows "
   258	            f"with steps); --steps {args.steps} > {MPAS_PARITY_MAX_STEPS} "
   259	            f"cap.")
   260	
   261	    if args.multicontroller:
   262	        # MUST run before any other JAX use (backend init). SLURM auto-detects;
   263	        # mpiexec needs the explicit coordinator + launcher env vars (OpenMPI
   264	        # OMPI_*, or Cray PALS PMI_* on Derecho).
   265	        if args.coordinator is not None:
   266	            n_procs = int(os.environ.get(
   267	                "OMPI_COMM_WORLD_SIZE", os.environ.get("PMI_SIZE", "0")))
   268	            proc_id = int(os.environ.get(
   269	                "OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "-1")))
   270	            if n_procs < 1 or proc_id < 0:
   271	                raise SystemExit(
   272	                    "--coordinator given but no launcher rank env found "
   273	                    "(OMPI_COMM_WORLD_SIZE/RANK or PMI_SIZE/PMI_RANK).")
   274	            jax.distributed.initialize(
   275	                coordinator_address=args.coordinator,
   276	                num_processes=n_procs, process_id=proc_id)
   277	        else:
   278	            # Environment-routed: SLURM/OMPI -> bare auto-detect; PALS/PMI
   279	            # (Derecho mpiexec) -> mpi4py bootstrap. Real init failures
   280	            # re-raise loudly.
   281	            from legoesm.parallel.early_init import (
   282	                init_jax_distributed_with_fallback,
   283	            )
   284	            init_jax_distributed_with_fallback()
   285	
   286	    from legoesm.parallel.sharded_dynamics import (
   287	        gather_voronoi_state_spmd,
   288	        make_voronoi_sharded_step,
   289	    )
   290	
   291	    nd = args.n_devices
   292	    avail = len(jax.devices())
   293	    if avail < nd:
   294	        raise SystemExit(f"need {nd} devices, have {avail} "
   295	                         f"(set --xla_force_host_platform_device_count)")
   296	    if args.multicontroller and nd != avail:
   297	        # A mesh over a strict subset would leave some processes' devices out
   298	        # of the program (non-addressable participation hazard). Route-B uses
   299	        # ALL global devices: one shard per device across every process.
   300	        raise SystemExit(
   301	            f"--multicontroller: --n-devices ({nd}) must equal the GLOBAL "
   302	            f"device count ({avail} across {jax.process_count()} processes).")
   303	
   304	    dt = args.dt
   305	    if dt is None:
   306	        dt = max(600.0 * 4.0 ** (4 - args.subdivision), 30.0)
   307	
   308	    reorder_for = args.reorder_for if args.reorder_for is not None else nd
   309	    if reorder_for < nd:
   310	        raise SystemExit(
   311	            f"--reorder-for ({reorder_for}) must be >= --n-devices ({nd}): "
   312	            f"the ghost padding only guarantees divisibility for the "
   313	            f"partition target.")
   314	    mesh, model, s0, dev_config = build_model_and_state(
   315	        args.subdivision, args.nlev, reorder_for, nd, args.partition_method,
   316	        moist=(args.physics == "kessler"), lloyd_iterations=args.lloyd)
   317	
   318	    if args.multicontroller:
   319	        # Every process computed the reorder independently — assert the
   320	        # partitions agree before any collective uses the halo schedule.
   321	        # The checksum covers the entity ORDER (coordinates) and the
   322	        # connectivity the ppermute schedule + TRiSK stencils read; a
   323	        # rank-divergent partition (e.g. one rank resolving
   324	        # --partition-method auto to METIS, another to RCB) cannot slip
   325	        # through on cell positions alone.
   326	        from jax.experimental import multihost_utils
   327	        crc = 0
   328	        for arr, dtype in (
   329	            (mesh.latCell, np.float64), (mesh.latEdge, np.float64),
   330	            (mesh.cellsOnEdge, np.int64), (mesh.edgesOnCell, np.int64),
   331	            (mesh.cellsOnCell, np.int64), (mesh.areaCell, np.float64),
   332	        ):
   333	            crc = zlib.crc32(np.ascontiguousarray(
   334	                np.asarray(arr, dtype=dtype)).tobytes(), crc)
   335	        crc = zlib.crc32(
   336	            np.asarray([mesh.nCells, mesh.nEdges, mesh.nVertices],
   337	                       dtype=np.int64).tobytes(), crc)
   338	        multihost_utils.assert_equal(
   339	            np.uint32(crc),
   340	            fail_message="partition/reorder checksum differs across "
   341	                         "processes (rank-divergent --partition-method "
   342	                         "resolution?)")
   343	
   344	    physics_fn = None
   345	    if args.physics == "held_suarez":
   346	        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_mpas
   347	        physics_fn = held_suarez_forcing_mpas
   348	    elif args.physics == "kessler":
   349	        # Warm-rain microphysics over the moist BCW tracers.  Kessler's
   350	        # saturation adjustment is a rate over the dt bound HERE, so it
   351	        # must match the stepping dt (make_kessler_forcing_mpas contract).
   352	        from legoesm.atmosphere.forcing.idealized.kessler_forcing import (
   353	            make_kessler_forcing_mpas,
   354	        )
   355	        physics_fn = make_kessler_forcing_mpas(dt)
   356	
   357	    # #1100: s0 from build_model_and_state is ALREADY partition-local-sharded
   358	    # when n_devices > 1 (no per-process global build on the timed path).
   359	    # The parity/conservation gates are the ONLY consumers of a global
   360	    # initial state — build it lazily here, at their smoke scales only
   361	    # (deterministic identical build on every process; bit-identical to the
   362	    # sharded s0 per tests/parallel/test_mpas_partitionlocal_build.py).
   363	    s0_global = None
   364	    if args.parity_gate or args.check_conservation:
   365	        from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas
   366	        s0_global = (s0 if dev_config.n_devices <= 1
   367	                     else baroclinic_wave_init_mpas(
   368	                         mesh, model.sigma_coord, perturbed=True,
   369	                         moist=(args.physics == "kessler")))
   370	
   371	    # Parity reference: the plain single-device trajectory on the SAME
   372	    # reordered mesh.  model.step's signature is call-compatible.
   373	    serial_final = None
   374	    if args.parity_gate:
   375	        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
   376	            MPASPrimitiveEquationModel,
   377	        )
   378	        ref_model = (model if dev_config.n_devices <= 1
   379	                     else MPASPrimitiveEquationModel(
   380	                         mesh, model.sigma_coord, model.config))
   381	        _s = s0_global
   382	        for _ in range(args.steps):
   383	            _s = ref_model.step(_s, dt, physics_fn=physics_fn)
   384	        _block(_s)
   385	        serial_final = _s
   386	
   387	    mass_before = None
   388	    if args.check_conservation:
   389	        mass_before = _global_dry_mass(s0_global, mesh)
   390	
   391	    step = make_voronoi_sharded_step(
   392	        model, dev_config, halo_strategy=args.halo_strategy)
   393	    # Already in the sharded layout (partition-local build) for nd > 1;
   394	    # single-device s0 is the plain global state.
   395	    s = s0
   396	
   397	    # Multi-controller: align every process around the timed loop.
   398	    if jax.process_count() > 1:
   399	        from jax.experimental import multihost_utils
   400	        multihost_utils.sync_global_devices("mpas_spmd_bench_start")
   401	
   402	    # Per-step timing: step 0 includes compile; record each step so re-trace
   403	    # (every step slow) is visible vs steady-state (steps 1.. fast).
   404	    per_step_ms = []
   405	    for _ in range(args.steps):
   406	        t0 = time.perf_counter()
   407	        if physics_fn is not None:
   408	            s = step(s, dt, physics_fn=physics_fn)
   409	        else:
   410	            s = step(s, dt)
   411	        _block(s)
   412	        per_step_ms.append((time.perf_counter() - t0) * 1e3)
   413	
   414	    if jax.process_count() > 1:
   415	        from jax.experimental import multihost_utils
   416	        multihost_utils.sync_global_devices("mpas_spmd_bench_end")
   417	
   418	    # HLO collective-permute census (#1113 ask 2): a STATIC compile property of
   419	    # the sharded step — the ppermute ROUND count that decomposes multi-node
   420	    # overhead (overhead ~= CPs/step * ~0.11 ms launch floor). The cube benches
   421	    # record this; the MPAS row did not, forcing an out-of-band census. Counted
   422	    # AFTER the timed loop so the census compile can't perturb per_step_ms[0]'s
   423	    # compile timing (the executable is already cached — this re-lower/compile
   424	    # is a cache hit; the count is data-independent, static in the partition).
   425	    # Best-effort (None if compilation is unsupported); the serial n=1 leg has
   426	    # no ppermute halo -> 0.
   427	    if physics_fn is not None:
   428	        _census_fn = lambda st: step(st, dt, physics_fn=physics_fn)  # noqa: E731
   429	    else:
   430	        _census_fn = lambda st: step(st, dt)  # noqa: E731
   431	    # ONE compile → full per-family census; the CP scalar (the #1113 round-count
   432	    # wall) is the collective_permute member, so no second compile for it.
   433	    hlo_census = hlo_collective_census(_census_fn, s)
   434	    hlo_cp = hlo_census["collective_permute"] if hlo_census else None
   435	
   436	    # --- Correctness gates (before any timing is reported) -----------------
   437	    if args.parity_gate or args.check_conservation:
   438	        final_global = (gather_voronoi_state_spmd(s, dev_config)
   439	                        if dev_config.n_devices > 1 else s)
   440	        prec = "float64" if jax.config.jax_enable_x64 else "float32"
   441	        rank0 = jax.process_index() == 0
   442	        if args.check_conservation:
   443	            mass_after = _global_dry_mass(final_global, mesh)
   444	            tol = (args.mass_rtol if args.mass_rtol is not None
   445	                   else MASS_RTOL_DEFAULTS[prec])
   446	            rel = abs(mass_after - mass_before) / abs(mass_before)
   447	            if rank0:
   448	                print(f"    conservation dry-mass: rel drift={rel:.3e} "
   449	                      f"(tol {tol:.1e}) over {args.steps} steps", flush=True)
   450	            if rel > tol:
   451	                if rank0:
   452	                    print("ERROR: conservation gate BREACHED.", flush=True)
   453	                return 4
   454	        if args.parity_gate:
   455	            tols = MPAS_PARITY_TOLS[prec]
   456	            ok = True
   457	            checks = [
   458	                (name, getattr(serial_final, name).data,
   459	                 getattr(final_global, name).data, rtol, atol)
   460	                for name, (rtol, atol) in tols.items()
   461	            ]
   462	            if serial_final.tracers is not None:
   463	                # Moist run: the tracer fields ride the packed exchange +
   464	                # RK advection — gate them too (q re-association floor is
   465	                # far below the q_v scale; reuse the T tolerances).
   466	                q_rtol, q_atol = tols["T"]
   467	                if set(final_global.tracers or {}) != set(
   468	                        serial_final.tracers):
   469	                    if rank0:
   470	                        print("ERROR: sharded run dropped tracer fields.",
   471	                              flush=True)
   472	                    return 5
   473	                checks += [
   474	                    (k, serial_final.tracers[k].data,
   475	                     final_global.tracers[k].data, q_rtol, q_atol * 1e-3)
   476	                    for k in sorted(serial_final.tracers)
   477	                ]
   478	            for name, want, got, rtol, atol in checks:
   479	                want = np.asarray(want)
   480	                got = np.asarray(got)
   481	                field_ok = bool(np.allclose(got, want, rtol=rtol, atol=atol))
   482	                ok &= field_ok
   483	                if rank0:
   484	                    mx = (float(np.max(np.abs(got - want)))
   485	                          if want.size else 0.0)
   486	                    print(f"    parity {name:>4s}: max|diff|={mx:.3e} "
   487	                          f"{'OK' if field_ok else 'MISMATCH'}", flush=True)
   488	            if not ok:
   489	                if rank0:
   490	                    print("ERROR: SPMD parity gate MISMATCH vs the "
   491	                          "single-device reference.", flush=True)
   492	                return 5
   493	
   494	    steady = per_step_ms[args.warmup:]
   495	    med = float(np.median(steady))
   496	    rec = dict(
   497	        component="mpas_atm",
   498	        subdivision=args.subdivision, n_devices=nd,
   499	        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
   500	        partition_method=args.partition_method, physics=args.physics,
   501	        # lloyd=0 is the LABELLED synthetic scaling mesh — anti-masquerade:
   502	        # a row without this field could pass as a production-SCVT receipt.
   503	        lloyd_iterations=args.lloyd,
   504	        # Requested vs EFFECTIVE (post-"auto") strategy — a JSONL row
   505	        # saying "auto" would not reveal whether ppermute or allgather
   506	        # was actually measured (codex M3c-2 MINOR).
   507	        halo_strategy_requested=args.halo_strategy,
   508	        halo_strategy_effective=getattr(
   509	            step, "_halo_strategy_effective", "serial"),
   510	        steps=args.steps, dt=dt,
   511	        platform=jax.default_backend(),
   512	        n_processes=jax.process_count(),
   513	        multicontroller=bool(args.multicontroller),
   514	        compile_ms=round(per_step_ms[0], 1),
   515	        steady_median_ms=round(med, 2),
   516	        steady_min_ms=round(float(np.min(steady)), 2),
   517	        per_step_ms=[round(x, 1) for x in per_step_ms],
   518	        cells=int(mesh.nCells) * args.nlev,
   519	        # ppermute round count/step (static compile property; #1113) — the
   520	        # multi-node ceiling is this count x the ~0.11 ms launch floor, so it
  2480	            else:
  2481	                cell_full = jax.lax.all_gather(
  2482	                    cell_pack, "device", axis=0, tiled=True)
  2483	                u_full = jax.lax.all_gather(
  2484	                    u_shard, "device", axis=0, tiled=True)
  2485	                gc, ge = halo_sl
  2486	                cell_local = cell_full[gc[0]]
  2487	                u_local = u_full[ge[0]]
  2488	
  2489	            # Unpack cell fields (inverse of the shared pack helper)
  2490	            T_local, ps_local, phis_local, q_local = _unpack_cell_state(
  2491	                cell_local, nlev)
  2492	
  2493	            # This device's local mesh (leading axis is the length-1
  2494	            # device slice of the stacked meshes).
  2495	            my_mesh = jax.tree.map(lambda x: x[0], mesh_sl)
  2496	
  2497	            tracers_local = None
  2498	            if tkeys:
  2499	                tracers_local = {
  2500	                    k: Field(data=q_local[:, i * nlev:(i + 1) * nlev],
  2501	                             name=k, dims=("nCells", "nlev"),
  2502	                             units="kg/kg", staggering="cell")
  2503	                    for i, k in enumerate(tkeys)
  2504	                }
  2505	
  2506	            # Build local state and compute tendency
  2507	            local_state = MPASHydrostaticState(
  2508	                u=Field(data=u_local, name="u",
  2509	                        dims=("nEdges", "nlev"), units="m/s",
  2510	                        long_name="normal velocity", staggering="edge"),
  2511	                T=Field(data=T_local, name="T",
  2512	                        dims=("nCells", "nlev"), units="K",
  2513	                        long_name="temperature", staggering="cell"),
  2514	                p_s=Field(data=ps_local, name="p_s",
  2515	                          dims=("nCells",), units="Pa",
  2516	                          long_name="surface pressure", staggering="cell"),
  2517	                phis=Field(data=phis_local, name="phis",
  2518	                           dims=("nCells",), units="m^2/s^2",
  2519	                           long_name="surface geopotential",
  2520	                           staggering="cell"),
  2521	                tracers=tracers_local,
  2522	            )
  2523	            tend = mpas_hydrostatic_tendencies(
  2524	                local_state, my_mesh, sigma, cfg, dt=dt_val,
  2525	            )
  2526	
  2527	            # Owned shards only.  Tracer ADVECTION tendencies ride back
  2528	            # in the same packed wire order (mpas_hydrostatic_tendencies
  2529	            # always returns tracer_tendencies for a tracered state).
  2530	            if tkeys:
  2531	                dq_owned = jnp.concatenate(
  2532	                    [tend.tracer_tendencies[k].data for k in tkeys],
  2533	                    axis=-1)[:cells_per]
  2534	            else:
  2535	                dq_owned = jnp.zeros((cells_per, 0), dtype=T_shard.dtype)
  2536	            return (tend.du_dt.data[:edges_per],
  2537	                    tend.dT_dt.data[:cells_per],
  2538	                    tend.dp_s_dt.data[:cells_per],
  2539	                    dq_owned)
  2540	
  2541	        return _local_tendency
  2542	
  2543	    _shard_tendency_cache: dict = {}
  2544	
  2545	    def _get_shard_tendency(tkeys: tuple):
  2546	        fn = _shard_tendency_cache.get(tkeys)
  2547	        if fn is None:
  2548	            fn = shard_map(
  2549	                _make_local_tendency(tkeys),
  2550	                mesh=jax_mesh,
  2551	                in_specs=(P("device"), P("device"), P("device"),
  2552	                          P("device"), P("device"), P(),
  2553	                          mesh_in_specs, halo_in_specs),
  2554	                out_specs=(P("device"), P("device"), P("device"),
  2555	                           P("device")),
  2556	                check_vma=False,
  2557	            )
  2558	            _shard_tendency_cache[tkeys] = fn
  2559	        return fn
  2560	
  2561	    # ------------------------------------------------------------------
  2562	    # Pre-compute mass conservation constants (avoid per-step allreduce)
  2563	    # ------------------------------------------------------------------
  2564	    if cfg.fix_mass:
  2565	        # areaCell rides as a P("device")-sharded jit ARGUMENT aligned
  2566	        # with the p_s cell shards (elementwise product stays local;
  2567	        # GSPMD emits one allreduce for the sum) — local-only, and
  2568	        # multi-controller-safe because it is an argument, not a closure
  2569	        # constant.  Take a HOST copy first (codex M3c-1 MAJOR): the
  2570	        # caller's mesh may arrive REPLICATED (bench replicate_pytree),
  2571	        # and multiprocess_safe_device_put passes non-fully-addressable
  2572	        # arrays through UNCHANGED — a replicated leaf would silently
  2573	        # stay replicated under multi-controller.  A host array is
  2574	        # always fully addressable, so the P("device") shard is
  2575	        # guaranteed on both controllers.  total_area is a host float.
  2576	        _area_for_mass = multiprocess_safe_device_put(
  2577	            np.asarray(global_mesh.areaCell), dev_sharding)
  2578	        # fp64 area sum to match the fp64 mass-budget accumulator below
  2579	        # (mirrors make_voronoi_mpi_step; identical under x64).
  2580	        _total_area = float(jnp.sum(
  2581	            global_mesh.areaCell.astype(jnp.float64)))
  2582	    else:
  2583	        _area_for_mass = jnp.zeros((0,))  # unused placeholder arg
  2584	        _total_area = 1.0
  2585	
  2586	    # ------------------------------------------------------------------
  2587	    # JIT-compiled step: dispatch_integrator dynamics (tracer advection
  2588	    # included) → operator-split physics → floors → global mass fix.
  2589	    # Mirrors the serial MPASPrimitiveEquationModel._step_jit ordering.
  2590	    # ------------------------------------------------------------------
  2591	
  2592	    def _build_step(physics_fn=None):
  2593	        """Build one jitted step executable.
  2594	
  2595	        ``physics_fn`` is captured in the closure — JAX cannot trace a
  2596	        Python callable as an array argument (same convention as
  2597	        ``CompiledShardedStep._compile`` / ``_SingleDeviceStep``).
  2598	        ``physics_fn=None`` produces exactly the dynamics-only graph.
  2599	        ``forcing`` / ``phys_state`` are traced jit arguments (NOT
  2600	        static) so per-step values do not retrace (SegmentForcing
  2601	        doctrine); the local-mesh / halo-schedule / area constants are
  2602	        threaded as sharded arguments (see the factory docstring).
  2603	        """
  2604	        _phys = physics_fn
  2605	
  2606	        @jax.jit
  2607	        def _step(state, dt, forcing, phys_state,
  2608	                  mesh_arg, halo_arg, area_arg):
  2609	            # Canonical tracer wire order — static at trace time (part
  2610	            # of the state's pytree structure).
  2611	            tkeys = (tuple(sorted(state.tracers))
  2612	                     if state.tracers is not None else ())
  2613	
  2614	            # Precision parity with the serial ``_step_jit``: integrate
  2615	            # in compute precision, store in storage precision.
  2616	            state = cast_pytree(state, None, "compute")
  2617	            shard_tendency = _get_shard_tendency(tkeys)
  2618	
  2619	            def _pack_tracers(s):
  2620	                if not tkeys:
  2621	                    return jnp.zeros(
  2622	                        s.T.data.shape[:-1] + (0,), dtype=s.T.data.dtype)
  2623	                return jnp.concatenate(
  2624	                    [s.tracers[k].data for k in tkeys], axis=-1)
  2625	
  2626	            def dyn_tendency_fn(s):
  2627	                """Dynamics tendencies as a state-shaped pytree (tracer
  2628	                ADVECTION rides under the same tracer keys) so the pytree
  2629	                RK integrator advances moisture mass-consistently with
  2630	                u/T/p_s — mirroring the serial ``dyn_tendency_fn``."""
  2631	                du, dT, dps, dq = shard_tendency(
  2632	                    s.u.data, s.T.data, s.p_s.data, s.phis.data,
  2633	                    _pack_tracers(s), dt, mesh_arg, halo_arg,
  2634	                )
  2635	                # Static workload-gated fusion barrier (codex rounds
  2636	                # 11-12; see _FUSION_BARRIER_WORKLOADS). Every gate
  2637	                # operand is trace-time static (closure int + aval
  2638	                # shapes) — no retrace; the barrier is an identity for
  2639	                # numerics and AD.
  2640	                _sig = (n_dev, s.u.data.shape[0],
  2641	                        s.T.data.shape[0], s.T.data.shape[1],
  2642	                        str(s.u.data.dtype))
  2643	                if _sig in _FUSION_BARRIER_WORKLOADS:
  2644	                    du, dT, dps, dq = jax.lax.optimization_barrier(
  2645	                        (du, dT, dps, dq))
  2646	                tr_tend = None
  2647	                if s.tracers is not None:
  2648	                    tr_tend = {
  2649	                        k: s.tracers[k].replace(
  2650	                            data=dq[..., i * nlev:(i + 1) * nlev])
  2651	                        for i, k in enumerate(tkeys)
  2652	                    }
  2653	                return MPASHydrostaticState(
  2654	                    u=s.u.replace(data=du),
  2655	                    T=s.T.replace(data=dT),
  2656	                    p_s=s.p_s.replace(data=dps),
  2657	                    phis=s.phis.replace(
  2658	                        data=jnp.zeros_like(s.phis.data)),
  2659	                    v=s.v,
  2660	                    tracers=tr_tend,
  2661	                )
  2662	
  2663	            # --- 1. Dynamics: RK-integrate (physics OFF; tracer
  2664	            #        advection stays in the dynamics).  Honors
  2665	            #        config.time_integrator via the SAME dispatch the
  2666	            #        serial step uses (the previous hard-coded SSP-RK3
  2667	            #        silently overrode e.g. the ssp_rk54_scan default).
  2668	            state_new = dispatch_integrator(
  2669	                state, dyn_tendency_fn, dt, cfg.time_integrator,
  2670	            )
scripts/cluster/scaling_levante/README.md:110:(`--xla_gpu_collective_permute_combine_threshold_bytes=32MiB` +
scripts/cluster/scaling_levante/README.md:111:`--xla_gpu_enable_pipelined_p2p=true`), `pgle`
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:47:  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=128 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:55:run_arm B_combine "--xla_gpu_collective_permute_combine_threshold_bytes=8388608"
scripts/cluster/scaling_levante/atm_ll128_combine_ab.sbatch:56:run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:62:JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:66:JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=24 \
scripts/cluster/scaling_levante/cube_tiled_step.sbatch:79:export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_latency_hiding_scheduler=false"
scripts/cluster/scaling_levante/_env.sh:125:# REPLACING XLA_FLAGS, not appending.
scripts/cluster/scaling_levante/_env.sh:127:  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:236:    _XLA_COMM_FLAGS="--xla_gpu_collective_permute_combine_threshold_bytes=33554432 --xla_gpu_enable_pipelined_p2p=true"
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:242:              xla)   export XLA_FLAGS="${XLA_FLAGS:-} ${_XLA_COMM_FLAGS}" ;;
scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch:275:            [ "$ARM" = xla ] && export XLA_FLAGS="${XLA_FLAGS:-} ${_XLA_COMM_FLAGS}"
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch:47:  XLA_FLAGS="${2}" JAX_ENABLE_X64=0 srun --ntasks=64 --ntasks-per-node=4 \
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch:55:run_arm B_combine "--xla_gpu_collective_permute_combine_threshold_bytes=8388608"
scripts/cluster/scaling_levante/atm_ll64_combine_ab.sbatch:56:run_arm C_combine_pipelined "--xla_gpu_collective_permute_combine_threshold_bytes=8388608 --xla_gpu_enable_pipelined_p2p=true"

web search: 
web search: site:docs.nvidia.com nsight systems sqlite CUPTI_ACTIVITY_KIND_KERNEL start end timestamp clock ...
exec
/usr/bin/bash -lc "nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '1238,1428p'
nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '2275,2390p'
nl -ba packages/core/legoesm/parallel/sharded_dynamics.py | sed -n '2670,2795p'
nl -ba scripts/bench/metadata.py | sed -n '124,225p'
nl -ba scripts/cluster/scaling_levante/README.md | sed -n '85,128p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 193ms:
  1238	        if new_cells.size == 0:
  1239	            return
  1240	        halo_cells_set.update(new_cells.tolist())
  1241	
  1242	
  1243	#: Halo depth the SPMD Voronoi partition infra is built at.  ONE definition
  1244	#: consumed by both the production step factory and ``spmd_schedule_cost``:
  1245	#: a score computed at a different depth describes a different comm graph, and
  1246	#: two independently hardcoded 3s let production drift unnoticed.
  1247	SPMD_HALO_DEPTH = 3
  1248	
  1249	
  1250	def spmd_schedule_cost(mesh, n_dev, *, method="auto", reorder_target=None,
  1251	                       already_reordered=False, halo_depth=SPMD_HALO_DEPTH,
  1252	                       ppermute_cells_per_device_threshold=2_000,
  1253	                       round_profile_for_device=None):
  1254	    """How much halo communication one ownership choice costs, computed offline.
  1255	
  1256	    Scores a Voronoi ownership (mesh split) by the number of ``ppermute``
  1257	    ROUNDS one halo exchange needs -- the sequential collective launches that
  1258	    dominate MPAS strong scaling above ~64 devices.  Runs on a laptop: no GPU,
  1259	    no MPI, no benchmark job, so a split can be compared before it costs an
  1260	    allocation.
  1261	
  1262	    It calls the SAME builders production calls
  1263	    (:func:`_build_voronoi_partition_infra` then
  1264	    :func:`_build_ppermute_schedule`).  A re-derived lookalike answers a
  1265	    different question: a 1-ring ``cellsOnEdge`` adjacency graph reports 8
  1266	    rounds where the real depth-3-plus-closure graph reports 12-14.
  1267	
  1268	    WHAT THE NUMBER IS NOT
  1269	    ----------------------
  1270	    * ``n_rounds`` is per HALO FILL, not per model step.  A step costs
  1271	      ``n_rounds`` x (tendency evaluations per step), which depends on the
  1272	      configured integrator -- SSP-RK3 evaluates 3 times, but the MPAS default
  1273	      is ``ssp_rk54_scan``.  Multiply with the integrator you actually run.
  1274	    * ``n_rounds`` is NOT proven equal to the comm graph's ``max_degree``.
  1275	      ``_build_ppermute_schedule`` tries a finite set of greedy orders and
  1276	      keeps the best; equality is MEASURED (compare the returned
  1277	      ``max_degree``), never assumed.  Do not claim "the colouring is already
  1278	      optimal so only ownership can help" from this function.
  1279	    * It scores the ppermute strategy.  Production auto-selects ALLGATHER when
  1280	      cells/device is below ``ppermute_cells_per_device_threshold``, in which
  1281	      case there is no ppermute schedule and this number is counterfactual --
  1282	      see the returned ``production_strategy``.
  1283	
  1284	    MESH STATE -- the one thing that silently invalidates the score
  1285	    --------------------------------------------------------------
  1286	    Production does NOT reorder inside ``make_voronoi_sharded_step``; it
  1287	    consumes an already-reordered ``model.mesh``.  The scaling bench reorders
  1288	    ONCE for a ``reorder_target`` device count and then runs at a possibly
  1289	    DIFFERENT device count.  So pass what you actually have:
  1290	
  1291	    * raw mesh, scoring a run at ``n_dev``: defaults are right.
  1292	    * raw mesh, but the run reorders for a different target: pass
  1293	      ``reorder_target=<that target>``; the split is built for the target and
  1294	      scored at ``n_dev``.
  1295	    * already-reordered mesh (what production holds): pass
  1296	      ``already_reordered=True``; ``method`` is then ignored and reported as
  1297	      ``"pre-reordered"``, because the ownership is already baked in.
  1298	
  1299	    Parameters
  1300	    ----------
  1301	    mesh : VoronoiMesh
  1302	    n_dev : int
  1303	        Device count the run uses.  Must be >= 1.
  1304	    method : str
  1305	        Ownership for the reorder; ignored when *already_reordered*.
  1306	    reorder_target : int | None
  1307	        Device count the reorder targets, when it differs from *n_dev*.
  1308	    already_reordered : bool
  1309	    halo_depth : int
  1310	        Must match production (3) or the graph is a different graph.
  1311	    ppermute_cells_per_device_threshold : int
  1312	        Mirror of the production auto-select threshold, only used to report
  1313	        ``production_strategy``.
  1314	    round_profile_for_device : int | None
  1315	        When set to a device id, also return ``round_profile``: the per-round
  1316	        payload THAT DEVICE exchanges, in schedule order, restricted to the
  1317	        rounds it actually participates in.
  1318	
  1319	        This exists to make a profiler trace interpretable.  An ``nsys``
  1320	        capture is per RANK, and a rank appears only in the colour classes
  1321	        that touch it -- at s9/np64 rank 0 shows 8 ``SendRecv`` per halo fill
  1322	        while the graph's ``max_degree`` is 11 -- so the k-th observed
  1323	        collective is the k-th round CONTAINING THAT DEVICE, not the k-th
  1324	        round.  Pairing measured durations against all rounds would silently
  1325	        misalign them.
  1326	
  1327	        Each entry gives ``round`` (index in the full schedule), ``partner``
  1328	        and ``halo_cells``/``halo_edges`` -- the schedule's per-round PADDED
  1329	        extents, which are what goes on the wire: the index arrays are
  1330	        ``(n_dev, max_c)`` and ``ppermute`` moves the whole padded buffer, so
  1331	        a pair's own send count does not set its cost.  Sizes are ENTITY
  1332	        COUNTS, not bytes; converting needs the packed cell width from
  1333	        :func:`_pack_cell_state` (``nlev*(1+n_tracers)+2``) for cells and
  1334	        ``nlev`` for edges.
  1335	
  1336	    Returns
  1337	    -------
  1338	    dict
  1339	        ``n_rounds`` (the cost), ``max_degree`` (the lower bound to compare
  1340	        it against), ``n_rounds_greedy``, ``coloring_method``,
  1341	        ``resolved_method`` (concrete, never ``"auto"``),
  1342	        ``production_strategy`` (``"ppermute"`` or ``"allgather"``),
  1343	        ``max_local_cells``, ``max_local_edges``, and the echoed inputs.
  1344	
  1345	    Reference census on the unrelaxed mesh, which any change here must still
  1346	    reproduce: subdiv-8 sfc 12/14 rounds at 64/128 devices, metis 13/19,
  1347	    geometric 16/21; subdiv-9 sfc 11/13, metis 14/18, geometric 14/18.
  1348	    """
  1349	    from legoesm.parallel.voronoi_partition import (
  1350	        reorder_voronoi_for_sharding, resolve_sharding_partition_method,
  1351	    )
  1352	
  1353	    if int(n_dev) != n_dev or int(n_dev) < 1:
  1354	        # int() would silently truncate 3.9 -> 3 and score the wrong split.
  1355	        raise ValueError(
  1356	            f"spmd_schedule_cost: n_dev must be an integer >= 1, got {n_dev!r}")
  1357	    n_dev = int(n_dev)
  1358	
  1359	    if already_reordered:
  1360	        if reorder_target is not None:
  1361	            raise ValueError(
  1362	                "spmd_schedule_cost: reorder_target is meaningless with "
  1363	                "already_reordered=True — the ownership is already baked into "
  1364	                "the mesh.")
  1365	        prepared, resolved = mesh, "pre-reordered"
  1366	    else:
  1367	        target = n_dev if reorder_target is None else int(reorder_target)
  1368	        prepared = reorder_voronoi_for_sharding(mesh, target, method=method)
  1369	        # Report the CONCRETE ownership: "auto" hides which partitioner ran.
  1370	        # Uses the SAME resolver the reorder used, so the label cannot drift
  1371	        # from the policy.
  1372	        resolved = resolve_sharding_partition_method(method)
  1373	
  1374	    # The builder assigns residual entities to the LAST owner but excludes them
  1375	    # from every owned contiguous block, so schedule send indices can exceed a
  1376	    # device's shard length -- a number that looks fine and is not.  Reachable
  1377	    # via reorder_target: a mesh padded for 3 devices is not divisible by 4.
  1378	    # The scaling bench rejects that pairing; so does this.
  1379	    n_cells, n_edges = int(prepared.nCells), int(prepared.nEdges)
  1380	    if n_cells % n_dev or n_edges % n_dev:
  1381	        raise ValueError(
  1382	            f"spmd_schedule_cost: prepared mesh has nCells={n_cells}, "
  1383	            f"nEdges={n_edges}, neither divisible by n_dev={n_dev}. The mesh "
  1384	            f"is padded for its reorder target"
  1385	            f"{'' if already_reordered else f' ({target})'}, so scoring it at "
  1386	            f"a device count that does not divide it silently mis-slices the "
  1387	            f"owned blocks. Score at a device count that divides the prepared "
  1388	            f"mesh.")
  1389	    (
  1390	        _stacked, _gc, _ge, _noc, _noe, max_lc, max_le, partitions, cell_owner,
  1391	    ) = _build_voronoi_partition_infra(prepared, n_dev, halo_depth=halo_depth)
  1392	    cells_per = n_cells // n_dev
  1393	    edges_per = n_edges // n_dev
  1394	    sched = _build_ppermute_schedule(
  1395	        partitions, cell_owner, n_dev, cells_per, edges_per, max_lc, max_le,
  1396	    )
  1397	    return {
  1398	        "method": method,
  1399	        "resolved_method": resolved,
  1400	        "n_dev": n_dev,
  1401	        # Unknown for a pre-reordered mesh: the ownership is baked in and the
  1402	        # target that produced it is not recoverable from the mesh. Reporting
  1403	        # n_dev there would assert something we did not verify.
  1404	        "reorder_target": (None if already_reordered else
  1405	                           (n_dev if reorder_target is None
  1406	                            else int(reorder_target))),
  1407	        "already_reordered": bool(already_reordered),
  1408	        "halo_depth": halo_depth,
  1409	        "n_rounds": int(sched["n_rounds"]),
  1410	        "n_rounds_greedy": int(sched["n_rounds_greedy"]),
  1411	        "max_degree": int(sched.get("max_degree", -1)),
  1412	        "coloring_method": sched["coloring_method"],
  1413	        # Production returns before selecting a strategy at n_dev==1, and a
  1414	        # caller may force halo_strategy; this reports what AUTO would pick.
  1415	        "production_strategy": (
  1416	            None if n_dev == 1 else
  1417	            ("allgather" if cells_per < ppermute_cells_per_device_threshold
  1418	             else "ppermute")),
  1419	        "cells_per_device": cells_per,
  1420	        "max_local_cells": int(max_lc),
  1421	        "max_local_edges": int(max_le),
  1422	        **(
  1423	            {} if round_profile_for_device is None else
  1424	            {"round_profile": _round_profile(sched, round_profile_for_device,
  1425	                                             n_dev)}
  1426	        ),
  1427	    }
  1428	
  2275	            raise ValueError(
  2276	                "make_voronoi_sharded_step(return_phys_state=True) needs "
  2277	                "a multi-device config; on a single device use "
  2278	                "model.step (eager, carry stashed on the model) or the "
  2279	                "ModelDriver loop, which threads the carry."
  2280	            )
  2281	        return model.step
  2282	
  2283	    from legoesm.core.precision import cast_pytree
  2284	    from legoesm.core.state import MPASHydrostaticState
  2285	    from legoesm.parallel.mesh import multiprocess_safe_device_put
  2286	    from legoesm.parallel.shard_map_compat import shard_map
  2287	    from legoesm.timestepping.dispatch import dispatch_integrator
  2288	    from legoesm.timestepping.integration import (
  2289	        refuse_unthreaded_stateful_physics,
  2290	    )
  2291	
  2292	    n_dev = dev_config.n_devices
  2293	    voronoi_dims = dev_config.voronoi_dims
  2294	    if voronoi_dims is None:
  2295	        raise ValueError("dev_config.voronoi_dims must be set for Voronoi grids")
  2296	    nCells, nEdges, _nVerts = voronoi_dims
  2297	    jax_mesh = dev_config.mesh
  2298	
  2299	    cells_per = nCells // n_dev
  2300	    edges_per = nEdges // n_dev
  2301	
  2302	    global_mesh = model.mesh
  2303	    sigma = model.sigma_coord
  2304	    cfg = model.config
  2305	
  2306	    # The MPAS RHS comes from the passed-in model instance, not an atmosphere
  2307	    # import — this substrate ``parallel`` module must not depend UP on the
  2308	    # atmosphere component (federation: legoesm-core stays standalone-installable).
  2309	    # The free function is needed (not ``.tendencies``) because each device runs
  2310	    # it on its own rank-local, traced mesh.
  2311	    mpas_hydrostatic_tendencies = getattr(model, "sharded_tendency_fn", None)
  2312	    if mpas_hydrostatic_tendencies is None:
  2313	        raise TypeError(
  2314	            f"{type(model).__name__} does not expose a 'sharded_tendency_fn' "
  2315	            f"staticmethod; the multi-device Voronoi sharder needs the free "
  2316	            f"tendency RHS (state, mesh, sigma_coord, config, *, dt=...) to run "
  2317	            f"on a rank-local mesh without importing the dycore's component."
  2318	        )
  2319	
  2320	    # ------------------------------------------------------------------
  2321	    # Auto-select halo strategy based on grid size per device
  2322	    # ------------------------------------------------------------------
  2323	    if halo_strategy == "auto":
  2324	        if cells_per < ppermute_cells_per_device_threshold:
  2325	            halo_strategy = "allgather"
  2326	            logger.info(
  2327	                "Auto-selected allgather strategy: cells_per_device=%d < "
  2328	                "threshold=%d — ppermute packing overhead would dominate.",
  2329	                cells_per, ppermute_cells_per_device_threshold,
  2330	            )
  2331	        else:
  2332	            halo_strategy = "ppermute"
  2333	            logger.info(
  2334	                "Auto-selected ppermute strategy: cells_per_device=%d >= "
  2335	                "threshold=%d.",
  2336	                cells_per, ppermute_cells_per_device_threshold,
  2337	            )
  2338	
  2339	    # ------------------------------------------------------------------
  2340	    # Setup: build per-device local meshes and gather indices
  2341	    # ------------------------------------------------------------------
  2342	    logger.info(
  2343	        "Building halo-partitioned infrastructure for %d device(s) "
  2344	        "(nCells=%d, nEdges=%d, halo_depth=3, strategy=%s) ...",
  2345	        n_dev, nCells, nEdges, halo_strategy,
  2346	    )
  2347	    t0 = time.time()
  2348	    (
  2349	        stacked_meshes,   # VoronoiMesh pytree with (n_dev, max_l*) leaves
  2350	        gather_cells,     # (n_dev, max_lc)
  2351	        gather_edges,     # (n_dev, max_le)
  2352	        _n_owned_cells,
  2353	        _n_owned_edges,
  2354	        max_lc,
  2355	        max_le,
  2356	        partitions_out,   # list[VoronoiPartition] (for ppermute schedule)
  2357	        cell_owner_out,   # np.ndarray (nCells,) cell ownership
  2358	    ) = _build_voronoi_partition_infra(global_mesh, n_dev,
  2359	                                       halo_depth=SPMD_HALO_DEPTH)
  2360	    logger.info(
  2361	        "  partition setup done in %.2fs  "
  2362	        "(max_local_cells=%d, max_local_edges=%d, cells_per=%d, edges_per=%d)",
  2363	        time.time() - t0, max_lc, max_le, cells_per, edges_per,
  2364	    )
  2365	
  2366	    # LOCAL-ONLY metadata: shard the stacked local meshes on the leading
  2367	    # device axis — device i holds ONLY its own local mesh (leaf slice
  2368	    # [i]), never the other devices' connectivity.  The mesh rides into
  2369	    # the jitted step as an ARGUMENT with P("device") shard_map in_specs
  2370	    # (multi-controller-safe: a sharded jit ARG is legal where a sharded
  2371	    # CLOSURE constant raises at trace time under jax.distributed;
  2372	    # ``multiprocess_safe_device_put`` builds the global array from each
  2373	    # process's local copy).  All leaves are arrays after the jnp.stack
  2374	    # in _build_voronoi_partition_infra (ints become (n_dev,) arrays).
  2375	    dev_sharding = dev_config.face_sharding  # P("device") on axis 0
  2376	    stacked_meshes = jax.tree.map(
  2377	        lambda x: multiprocess_safe_device_put(x, dev_sharding),
  2378	        stacked_meshes,
  2379	    )
  2380	
  2381	    nlev = model.sigma_coord.n_levels
  2382	
  2383	    # ------------------------------------------------------------------
  2384	    # Strategy dispatch: ppermute (O(halo)) vs allgather (O(N))
  2385	    # ------------------------------------------------------------------
  2386	
  2387	    use_ppermute = halo_strategy == "ppermute"
  2388	
  2389	    if use_ppermute:
  2390	        # Build ppermute schedule: neighbor-only halo exchange
  2670	            )
  2671	
  2672	            # --- 2. Operator-split physics (mirrors _step_jit): evaluate
  2673	            #     ONCE on the post-dynamics state and apply forward over
  2674	            #     dt, BEFORE the floors and the mass fix.  Column physics
  2675	            #     is cell/edge-local, so it runs on the sharded global
  2676	            #     arrays OUTSIDE the shard_map kernel — GSPMD partitions
  2677	            #     the pointwise work per device with no halo traffic (the
  2678	            #     global-mesh closure contributes only replicated 1-D
  2679	            #     cell fields such as latCell). ---
  2680	            phys_state_out = phys_state
  2681	            if _phys is not None:
  2682	                _pr = _phys(state_new, global_mesh, sigma,
  2683	                            phys_state=phys_state, forcing=forcing)
  2684	                # NB ``type(...) is tuple`` (not isinstance): tendencies
  2685	                # are themselves a NamedTuple — mirror the serial guard.
  2686	                if type(_pr) is tuple:
  2687	                    if not return_phys_state:
  2688	                        # Stateful-physics convention with no carry
  2689	                        # channel armed: silently dropping the carry
  2690	                        # would corrupt stateful schemes (TKE etc.).
  2691	                        # Trace-time Python check → loud failure.
  2692	                        raise TypeError(
  2693	                            "make_voronoi_sharded_step: physics_fn "
  2694	                            "returned a (tendencies, phys_state) tuple, "
  2695	                            "but the step was built with "
  2696	                            "return_phys_state=False (no carry channel). "
  2697	                            "Rebuild with return_phys_state=True and "
  2698	                            "thread the returned carry, or use a "
  2699	                            "stateless physics_fn returning bare "
  2700	                            "tendencies (e.g. held_suarez_forcing_mpas)."
  2701	                        )
  2702	                    _pt, phys_state_out = _pr[0], _pr[1]
  2703	                else:
  2704	                    _pt = _pr
  2705	                state_new = MPASHydrostaticState(
  2706	                    u=state_new.u.replace(
  2707	                        data=state_new.u.data + dt * _pt.du_dt.data),
  2708	                    T=state_new.T.replace(
  2709	                        data=state_new.T.data + dt * _pt.dT_dt.data),
  2710	                    p_s=state_new.p_s.replace(
  2711	                        data=state_new.p_s.data + dt * _pt.dp_s_dt.data),
  2712	                    phis=state_new.phis,
  2713	                    v=state_new.v,
  2714	                    tracers=state_new.tracers,
  2715	                )
  2716	                if (state_new.tracers is not None
  2717	                        and _pt.tracer_tendencies is not None):
  2718	                    state_new = state_new._replace(tracers={
  2719	                        k: (state_new.tracers[k].replace(
  2720	                                data=state_new.tracers[k].data
  2721	                                + dt * _pt.tracer_tendencies[k].data)
  2722	                            if k in _pt.tracer_tendencies
  2723	                            else state_new.tracers[k])
  2724	                        for k in state_new.tracers
  2725	                    })
  2726	
  2727	            # --- 3. Floors (mirrors _step_jit): temperature and tracer
  2728	            #        non-negativity (advection is not positive-definite;
  2729	            #        clamp before tracers feed saturation). ---
  2730	            if cfg.T_min > 0:
  2731	                state_new = state_new._replace(
  2732	                    T=state_new.T.replace(
  2733	                        data=jnp.maximum(state_new.T.data, cfg.T_min)))
  2734	            if state_new.tracers is not None:
  2735	                state_new = state_new._replace(tracers={
  2736	                    k: f.replace(data=jnp.maximum(f.data, 0.0))
  2737	                    for k, f in state_new.tracers.items()
  2738	                })
  2739	
  2740	            # --- 4. Global mass fixer ---
  2741	            if cfg.fix_mass:
  2742	                # Promote to the fp64 budget accumulator (mirrors the
  2743	                # serial fixer — plain fp32 reductions over 1e4-1e5 cells
  2744	                # leak N·eps noise), and compute both masses inside a
  2745	                # single fused reduction: stacking the two weighted ps
  2746	                # fields lets XLA emit ONE allreduce instead of two
  2747	                # sequentially-dependent ones.
  2748	                acc = jnp.float64
  2749	                area_acc = area_arg.astype(acc)
  2750	                ps_pair = jnp.stack([
  2751	                    state.p_s.data.astype(acc),
  2752	                    state_new.p_s.data.astype(acc),
  2753	                ], axis=0) * area_acc[None]
  2754	                masses = jnp.sum(ps_pair, axis=1)  # shape (2,)
  2755	                correction = (masses[0] - masses[1]) / _total_area
  2756	                # Apply in fp64, then return p_s to its pre-fix carry
  2757	                # dtype (codex M3c-1 MAJOR): under an fp32 compute state
  2758	                # with x64 enabled the fp64 correction would otherwise
  2759	                # promote the carry — the downcast-skipping storage cast
  2760	                # below cannot undo it, breaking the lax.scan carry-dtype
  2761	                # contract and re-tracing host loops.  No-op (bit
  2762	                # identical) whenever the compute state is already fp64.
  2763	                # NB the serial _fix_mass_mpas_hydro deliberately leaves
  2764	                # the promoted add (iter-11); parity in that corner mode
  2765	                # differs only by the rounding of the correction add.
  2766	                _ps = state_new.p_s.data
  2767	                state_new = state_new._replace(
  2768	                    p_s=state_new.p_s.replace(
  2769	                        data=(_ps + correction).astype(_ps.dtype)))
  2770	
  2771	            return cast_pytree(state_new, None, "storage"), phys_state_out
  2772	
  2773	        return _step
  2774	
  2775	    # Executable cache keyed by physics_fn identity (same convention as
  2776	    # ``_make_cache_key``: distinct callables ⇒ distinct XLA programs; a
  2777	    # stable callable ⇒ exactly one compile).  Each cached executable's
  2778	    # closure holds a strong reference to its physics_fn, so an id()
  2779	    # cannot be recycled while its cache entry is alive.
  2780	    _step_cache: dict = {}
  2781	
  2782	    def _voronoi_step(state, dt, physics_fn=None, forcing=None,
  2783	                      phys_state=None):
  2784	        """Sharded Voronoi step.  ``physics_fn`` is closure-captured into
  2785	        the jitted executable (selected by object identity) — it is never
  2786	        passed to ``jax.jit`` as a traced argument.  ``forcing`` /
  2787	        ``phys_state`` ARE traced jit arguments."""
  2788	        check_voronoi_spmd_state_schema(state)
  2789	        # Issue #405/#413: never silently run stateful physics without
  2790	        # its carry.  The predicates also see a partial/__wrapped__
  2791	        # wrapper that hides the tag.
  2792	        if return_phys_state:
  2793	            refuse_unthreaded_stateful_physics(
  2794	                physics_fn, phys_state,
  2795	                where="make_voronoi_sharded_step(return_phys_state=True)")
   124	
   125	def count_collective_permutes(hlo_text: str) -> int:
   126	    """Count ``collective_permute`` OPS in a lowered/compiled HLO text dump.
   127	
   128	    The ppermute halo kernel's signature and a STATIC compile property (the
   129	    count is fixed by the partition/edge-coloring schedule, not the data), so
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
    85	
    86	Account: export `SBATCH_ACCOUNT=<project>` (or pass `sbatch -A <project>`) —
    87	SLURM directives cannot expand env vars.
    88	
    89	
    90	## 2026-07 lane E: icosahedral/MPAS multicontroller
    91	
    92	`gpu_multinode_scaling.sbatch` gained lane E (`RUN_MPAS=1`, default on):
    93	icosahedral MPAS PE over `jax.distributed` + NCCL via
    94	`scripts/bench/bench_mpas_spmd_scaling.py` (cell-partition reorder +
    95	ppermute halos), 6 tasks = 2 nodes x 3 GPUs (`nCells = 10*4^L + 2` splits
    96	evenly for 1/2/3/6). Subdiv-4 parity+conservation smoke gates the timed
    97	`ICO_LEVEL` (default L7) case. Federation gate:
    98	`tests/parallel/test_mpas_spmd_multicontroller_selfspawn.py`.
    99	
   100	
   101	## 2026-07 lane T: comm-tuning A/B ladder (`RUN_TUNE=1`)
   102	
   103	Derecho-twin of the tuning ladder (full rationale + census numbers:
   104	`docs/performance/scaling/spmd_message_census_2026-07-08.md` and the
   105	Derecho README's lane-T section). Arms on the atm latlon (np=8), ocean
   106	(np=8) and cube cs-spmd (np=6, `base`/`xla` only) lanes, all within ONE
   107	allocation against a fresh `base` control arm:
   108	`fused` (`LEGOESM_LATLON_SPMD_FUSED_HALO=1`, bit-identical packing,
   109	trace receipt 41→29 CPs/step), `xla`
   110	(`--xla_gpu_collective_permute_combine_threshold_bytes=32MiB` +
   111	`--xla_gpu_enable_pipelined_p2p=true`), `pgle`
   112	(`JAX_ENABLE_PGLE=true`, per-run opt-in — recompiles after profiling).
   113	
   114	```bash
   115	sbatch --export=ALL,RUN_TUNE=1,RUN_NCCL=0,RUN_LATLON=0,RUN_OCEAN=0,RUN_MPAS=0 \
   116	    scripts/cluster/scaling_levante/gpu_multinode_scaling.sbatch
   117	```
   118	
   119	Outputs under `$OUTDIR/_ab_tuning/` (aggregator-skipped by design).

exec
/usr/bin/bash -lc 'git diff -- scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh scripts/cluster/scaling_levante/mpas_nsys_skew_capture.sbatch; git diff --no-index /dev/null scripts/bench/analyze_nsys_rank_skew.py || true; git diff --no-index /dev/null tests/bench/test_analyze_nsys_rank_skew.py || true' in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 707ms:
diff --git a/scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh b/scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh
index d9549cb85..1008f7ae9 100755
--- a/scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh
+++ b/scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh
@@ -42,13 +42,26 @@ if in_list "$SLURM_PROCID"; then
     # NO --delay/--duration: this run is ~7.3 s of compile plus 12 steps of
     # ~10 ms, so any delay long enough to skip compile also skips the entire
     # steady state.  Capture everything; the analysis drops warmup cycles.
+    # NO `--` separator before the application: nsys 2023.2.3 parses a bare
+    # `--` as an ambiguous long-option abbreviation and aborts with "option
+    # is ambiguous and matches ..." (verified — it killed job 26771842's four
+    # profiled ranks). The application simply follows the flags.
+    # `env -u QUADD_INJECTION_PROXY` is REQUIRED, not hygiene. nsys sets that
+    # variable for its injection library, and JAX treats any *_PROXY env var
+    # as distributed-coordinator proxy configuration -- it prints "JAX
+    # detected proxy variable(s) ... may cause a hang of
+    # distributed.initialize" and then does exactly that. Only the PROFILED
+    # ranks get the variable, so they diverge from the other 60 and the whole
+    # job deadlocks in init (observed: job 26771961 hung with precisely four
+    # such warnings, one per profiled rank, and produced nothing).
+    # Unsetting it here is safe: nsys's injection is already active via
+    # LD_PRELOAD by the time this exec runs, so the variable has done its job.
     exec "$NSYS_BIN" profile \
         --trace=cuda,nvtx \
         --output="$out" \
         --force-overwrite=true \
         --export=sqlite \
-        -- \
-        "$PY_BIN" "$@"
+        env -u QUADD_INJECTION_PROXY "$PY_BIN" "$@"
 else
     exec "$PY_BIN" "$@"
 fi
diff --git a/scripts/bench/analyze_nsys_rank_skew.py b/scripts/bench/analyze_nsys_rank_skew.py
new file mode 100644
index 000000000..a6e8202b1
--- /dev/null
+++ b/scripts/bench/analyze_nsys_rank_skew.py
@@ -0,0 +1,314 @@
+#!/usr/bin/env python
+"""Payload or wait? Compare ONE halo collective from BOTH of its partners.
+
+THE QUESTION.  The MPAS GPU lane spends 5.94 of 9.72 ms per step inside
+``ncclDevKernel_SendRecv`` (s9/np64), median 216 us against a ~30 us wire
+latency.  Two mechanisms explain that and they imply opposite engineering:
+cutting the ROUND COUNT pays well if the time is idle waiting, and pays only
+the per-round latency if the time is real transfer.
+
+WHY A SINGLE-RANK TRACE CANNOT ANSWER IT.  Grouped by schedule slot, those
+durations reproduce to CV <= 1.5 % across steps.  That refutes RANDOM jitter
+-- and nothing else.  A consistently late peer is exactly as reproducible.
+Wait is an inter-rank quantity; one timeline cannot see it.
+
+THE DISCRIMINATOR.  An NCCL SendRecv kernel starts when ITS rank arrives at
+the collective and ends when the transfer completes.  So for ONE collective
+seen from BOTH partners:
+
+  * both durations ~equal  -> the time is transfer.  PAYLOAD-bound.
+  * one long, one short    -> the long one arrived early and idled.  WAIT-
+                              bound, and (long - short) is the skew.
+
+HOW COLLECTIVES ARE MATCHED, and why not by overlap.  The obvious method --
+pair kernels whose [start, end] intervals overlap -- does not work here and
+its own control proves it: all 64 ranks issue their round-r collective at
+about the same time, so rank 0's kernel overlaps a NON-partner's just as
+readily as its true partner's.  Overlap is necessary, never sufficient.
+
+This uses the SCHEDULE as ground truth instead, which makes the pairing
+exact.  ``spmd_schedule_cost(..., round_profile_for_device=d)`` gives the
+rounds device ``d`` participates in, IN THE ORDER IT ISSUES THEM.  If A's
+profile shows it exchanges with B at global round r, and B's profile shows
+the mirror-image entry at the same r, then A's k-th halo fill issues that
+collective at position ``k*len(A_profile) + index_of_r_in_A`` and B's at
+``k*len(B_profile) + index_of_r_in_B``.  Same collective, by construction.
+
+Three things are then checked rather than assumed:
+  * SYMMETRY -- B's profile must name A as its partner at the same round.
+    A one-sided entry means the schedule was misread.
+  * FILL COUNT -- both ranks must have executed the same number of halo
+    fills (``n_calls / len(profile)``).  Ranks step in lockstep, so a
+    mismatch means the traces are not of the same run or one is truncated.
+  * DIVISIBILITY -- each rank's call count must be a whole number of fills.
+
+NON-PARTNER CONTROL.  Passing a pair that does not exchange (e.g. rank 0 and
+a same-node rank that is not its neighbour) finds no shared round and is
+reported as such.  That is the correct outcome and it is what shows the
+matching is schedule-driven rather than coincidence -- an overlap-based
+method would happily have "matched" that pair.
+
+SAME NODE ONLY.  Timestamps come from independent nsys processes and share a
+timebase only within one machine (``CLOCK_MONOTONIC`` is per-machine and
+undisciplined across nodes; inter-node drift dwarfs a 216 us signal).  The
+capture co-locates the profiled ranks and records the rank->node map; this
+script requires that map and REFUSES a cross-node pair.
+
+No verdict string is printed.  The interpretation belongs in the analysis
+once the controls pass, not baked into the tool where it gets echoed back as
+evidence.
+"""
+from __future__ import annotations
+
+import argparse
+import json
+import os
+import sqlite3
+import statistics as st
+import sys
+from pathlib import Path
+
+sys.path.insert(0, str(Path(__file__).resolve().parent))
+
+from metadata import scaling_metadata  # noqa: E402
+
+_SENDRECV_LIKE = "ncclDevKernel_SendRecv%"
+
+
+def load_kernels(sqlite_path: Path, kernel_like: str = _SENDRECV_LIKE):
+    """``(start_ns, end_ns)`` per collective, in issue order.
+
+    Raises on an EMPTY result: a name that matches nothing would otherwise
+    make every downstream count zero and read as "no skew".
+    """
+    if not sqlite_path.is_file():
+        raise SystemExit(f"missing sqlite: {sqlite_path}")
+    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
+    try:
+        rows = con.execute(
+            "SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
+            "JOIN StringIds s ON k.demangledName = s.id "
+            "WHERE s.value LIKE ? ORDER BY k.start", (kernel_like,)).fetchall()
+    finally:
+        con.close()
+    if not rows:
+        raise SystemExit(
+            f"{sqlite_path.name}: no kernel matches {kernel_like!r}. The NCCL "
+            f"kernel may be named differently in this build — check with "
+            f"SELECT DISTINCT value FROM StringIds WHERE value LIKE '%nccl%'.")
+    out = []
+    for start, end in rows:
+        if start is None or end is None or end < start:
+            raise SystemExit(
+                f"{sqlite_path.name}: bad kernel row start={start} end={end}")
+        out.append((int(start), int(end)))
+    return out
+
+
+def read_rank_nodes(path: Path):
+    """Parse the capture's ``_rank_nodes.tsv`` into ``{rank: node}``."""
+    if not path.is_file():
+        raise SystemExit(
+            f"missing {path}. It is written by _nsys_rank_wrapper.sh and is "
+            f"what proves the profiled ranks shared a node; without it a "
+            f"cross-node pair cannot be refused and its timestamps would be "
+            f"incomparable.")
+    nodes = {}
+    for line in path.read_text().splitlines():
+        fields = dict(f.split("=", 1) for f in line.strip().split("\t") if "=" in f)
+        if "rank" in fields and "node" in fields:
+            nodes[int(fields["rank"])] = fields["node"]
+    if not nodes:
+        raise SystemExit(f"{path}: no rank=/node= records parsed")
+    return nodes
+
+
+def shared_round(profile_a, profile_b, rank_a: int, rank_b: int):
+    """Global round where A and B exchange, plus each one's issue index.
+
+    Returns ``(round, index_in_a, index_in_b)`` or ``None`` when the two do
+    not exchange at all (the non-partner control).  Enforces SYMMETRY: B must
+    name A at the same round, otherwise the schedule has been misread and any
+    pairing built on it would be wrong.
+    """
+    a_hits = [(i, e) for i, e in enumerate(profile_a) if e["partner"] == rank_b]
+    if not a_hits:
+        return None
+    if len(a_hits) > 1:
+        raise SystemExit(
+            f"rank {rank_a} exchanges with rank {rank_b} in {len(a_hits)} "
+            f"rounds; this pairing is not unique and the script would have to "
+            f"guess which collective is which.")
+    idx_a, entry = a_hits[0]
+    b_hits = [i for i, e in enumerate(profile_b)
+              if e["partner"] == rank_a and e["round"] == entry["round"]]
+    if not b_hits:
+        raise SystemExit(
+            f"asymmetric schedule: rank {rank_a} names {rank_b} at round "
+            f"{entry['round']}, but {rank_b}'s profile has no mirror entry. "
+            f"The schedule has been misread; refusing to pair.")
+    return entry["round"], idx_a, b_hits[0]
+
+
+def compare(kernels_a, kernels_b, len_a, len_b, idx_a, idx_b, drop_fills):
+    """One record per halo fill, comparing the SAME collective on both ranks."""
+    if len(kernels_a) % len_a or len(kernels_b) % len_b:
+        raise SystemExit(
+            f"call counts are not whole fills: rank A has {len(kernels_a)} "
+            f"calls over {len_a} rounds/fill, rank B {len(kernels_b)} over "
+            f"{len_b}. The traces and the schedule disagree about the run.")
+    fills_a, fills_b = len(kernels_a) // len_a, len(kernels_b) // len_b
+    if fills_a != fills_b:
+        raise SystemExit(
+            f"rank A executed {fills_a} halo fills, rank B {fills_b}. Ranks "
+            f"step in lockstep, so these are not the same run (or one trace "
+            f"is truncated).")
+    recs = []
+    for k in range(drop_fills, fills_a):
+        sa, ea = kernels_a[k * len_a + idx_a]
+        sb, eb = kernels_b[k * len_b + idx_b]
+        dur_a, dur_b = ea - sa, eb - sb
+        recs.append({
+            "fill": k,
+            "a_dur_us": dur_a / 1e3,
+            "b_dur_us": dur_b / 1e3,
+            # >0: A started later, i.e. B was waiting for A.
+            "start_delta_us": (sa - sb) / 1e3,
+            # The idle time the EARLIER arriver spent waiting, which is the
+            # quantity a round-count cut would actually remove.
+            "wait_estimate_us": abs(dur_a - dur_b) / 1e3,
+            "ratio_long_short": (max(dur_a, dur_b) / min(dur_a, dur_b)
+                                 if min(dur_a, dur_b) > 0 else None),
+        })
+    return recs
+
+
+def main() -> int:
+    p = argparse.ArgumentParser(
+        description=__doc__,
+        formatter_class=argparse.RawDescriptionHelpFormatter)
+    p.add_argument("--sqlite-dir", required=True, type=Path,
+                   help="Capture dir holding rank_N.sqlite + _rank_nodes.tsv")
+    p.add_argument("--pairs", required=True,
+                   help="Comma-separated rank pairs, e.g. 0:1,0:2,0:3 "
+                        "(include a non-partner as the control).")
+    p.add_argument("--subdivision", type=int, required=True)
+    p.add_argument("--n-devices", type=int, required=True)
+    p.add_argument("--partition-method", default="sfc")
+    p.add_argument("--lloyd", type=int, default=0)
+    p.add_argument("--reorder-target", type=int, default=None,
+                   help="The run's --reorder-for, when it differs from the "
+                        "device count. A different ownership is a different "
+                        "schedule, so getting this wrong misaligns everything.")
+    p.add_argument("--drop-fills", type=int, default=3,
+                   help="Leading halo fills to discard as warmup/compile.")
+    p.add_argument("--out", default="results/a1/nsys_rank_skew.json")
+    args = p.parse_args()
+
+    pairs = []
+    for item in (s.strip() for s in args.pairs.split(",")):
+        if not item:
+            continue
+        a, _, b = item.partition(":")
+        if not b:
+            raise SystemExit(f"bad pair {item!r}; expected 'A:B'")
+        pairs.append((int(a), int(b)))
+    if not pairs:
+        raise SystemExit("--pairs selected nothing")
+
+    nodes = read_rank_nodes(args.sqlite_dir / "_rank_nodes.tsv")
+    for a, b in pairs:
+        for r in (a, b):
+            if r not in nodes:
+                raise SystemExit(f"rank {r} absent from the rank->node map")
+        if nodes[a] != nodes[b]:
+            raise SystemExit(
+                f"pair {a}:{b} spans nodes ({nodes[a]} vs {nodes[b]}). "
+                f"Timestamps from different machines share no timebase, so "
+                f"any skew computed from them would be drift, not physics.")
+
+    from legoesm.grids.voronoi import create_voronoi_mesh
+    from legoesm.parallel.sharded_dynamics import spmd_schedule_cost
+
+    print(f"[mesh] L{args.subdivision} lloyd={args.lloyd} …", flush=True)
+    mesh = create_voronoi_mesh(subdivision_level=args.subdivision,
+                               lloyd_iterations=args.lloyd)
+    profiles, kernels = {}, {}
+    for rank in sorted({r for pr in pairs for r in pr}):
+        cost = spmd_schedule_cost(mesh, args.n_devices,
+                                  method=args.partition_method,
+                                  reorder_target=args.reorder_target,
+                                  round_profile_for_device=rank)
+        profiles[rank] = cost["round_profile"]
+        kernels[rank] = load_kernels(args.sqlite_dir / f"rank_{rank}.sqlite")
+        print(f"[rank {rank}] node={nodes[rank]} "
+              f"{len(kernels[rank])} collectives, "
+              f"{len(profiles[rank])} rounds/fill", flush=True)
+
+    out_pairs = {}
+    for a, b in pairs:
+        key = f"{a}:{b}"
+        shared = shared_round(profiles[a], profiles[b], a, b)
+        if shared is None:
+            print(f"\n[{key}] NOT PARTNERS — no shared round. This is the "
+                  f"control: a schedule-driven pairing finds nothing here, "
+                  f"where an overlap-based one would have matched.", flush=True)
+            out_pairs[key] = {"partners": False}
+            continue
+        rnd, idx_a, idx_b = shared
+        recs = compare(kernels[a], kernels[b], len(profiles[a]),
+                       len(profiles[b]), idx_a, idx_b, args.drop_fills)
+        if not recs:
+            raise SystemExit(
+                f"{key}: no fills left after dropping {args.drop_fills}")
+        med_a = st.median(r["a_dur_us"] for r in recs)
+        med_b = st.median(r["b_dur_us"] for r in recs)
+        med_wait = st.median(r["wait_estimate_us"] for r in recs)
+        med_delta = st.median(r["start_delta_us"] for r in recs)
+        payload = profiles[a][idx_a]
+        out_pairs[key] = {
+            "partners": True, "round": rnd, "n_fills": len(recs),
+            "halo_cells": payload["halo_cells"],
+            "halo_edges": payload["halo_edges"],
+            "median_a_us": round(med_a, 2), "median_b_us": round(med_b, 2),
+            "median_wait_estimate_us": round(med_wait, 2),
+            "median_start_delta_us": round(med_delta, 2),
+            "wait_fraction_of_longer": round(
+                med_wait / max(med_a, med_b), 4) if max(med_a, med_b) else None,
+            "fills": recs,
+        }
+        print(f"\n[{key}] round {rnd}: cells={payload['halo_cells']} "
+              f"edges={payload['halo_edges']}, {len(recs)} fills")
+        print(f"   rank {a} median {med_a:8.1f} us")
+        print(f"   rank {b} median {med_b:8.1f} us")
+        print(f"   |difference| {med_wait:8.1f} us  "
+              f"= {100 * med_wait / max(med_a, med_b):.1f}% of the longer")
+        print(f"   start delta  {med_delta:+8.1f} us "
+              f"({'A later' if med_delta > 0 else 'B later'})")
+
+    payload_out = {
+        "pairs": out_pairs,
+        "rank_nodes": {str(k): v for k, v in nodes.items()},
+        "metadata": scaling_metadata(
+            grid="voronoi", component="atmosphere",
+            resolution=f"L{args.subdivision}", n_levels=0, precision="n/a",
+            decomposition="cell_partition", solver_variant="n/a",
+            scaling_kind="halo-skew-attribution", transport="nccl",
+            extra={"n_devices": args.n_devices,
+                   "partition_method": args.partition_method,
+                   "lloyd_iterations": args.lloyd,
+                   "reorder_target": args.reorder_target,
+                   "drop_fills": args.drop_fills,
+                   "sqlite_dir": str(args.sqlite_dir)}),
+    }
+    outdir = os.path.dirname(args.out)
+    if outdir:
+        os.makedirs(outdir, exist_ok=True)
+    with open(args.out, "w") as f:
+        json.dump(payload_out, f, indent=2)
+    print(f"\nJSON: {args.out}")
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
diff --git a/tests/bench/test_analyze_nsys_rank_skew.py b/tests/bench/test_analyze_nsys_rank_skew.py
new file mode 100644
index 000000000..1d5698138
--- /dev/null
+++ b/tests/bench/test_analyze_nsys_rank_skew.py
@@ -0,0 +1,189 @@
+"""Direct tests for the multi-rank halo-skew analyzer.
+
+The load-bearing logic is the PAIRING: which collective on rank A is the same
+collective as which on rank B.  Get that wrong and every duration comparison
+is between unrelated kernels — which is exactly how the previous probe in this
+family produced a confident wrong number.  So these tests hammer the pairing
+and its refusals, not the arithmetic.
+"""
+from __future__ import annotations
+
+import importlib.util
+import sqlite3
+import sys
+from pathlib import Path
+
+import pytest
+
+_SCRIPT = (Path(__file__).resolve().parents[2]
+           / "scripts" / "bench" / "analyze_nsys_rank_skew.py")
+_spec = importlib.util.spec_from_file_location("nsys_rank_skew", _SCRIPT)
+mod = importlib.util.module_from_spec(_spec)
+_spec.loader.exec_module(mod)
+
+
+def _profile(*entries):
+    """entries: (round, partner) -> the round_profile shape."""
+    return [{"round": r, "partner": p, "halo_cells": 10, "halo_edges": 20}
+            for r, p in entries]
+
+
+# --- shared_round: the pairing and its refusals ---------------------------
+
+
+def test_shared_round_finds_the_mirrored_entry_and_both_indices():
+    # A issues rounds [0(->1), 3(->2)]; B=1 issues [0(->0), 5(->2)].
+    a = _profile((0, 1), (3, 2))
+    b = _profile((0, 0), (5, 2))
+    assert mod.shared_round(a, b, 0, 1) == (0, 0, 0)
+
+    # Same, but the shared round sits at DIFFERENT issue positions on each
+    # rank — the case a naive "k-th call on both" pairing gets wrong.
+    a2 = _profile((7, 4), (9, 1))
+    b2 = _profile((2, 8), (5, 6), (9, 0))
+    assert mod.shared_round(a2, b2, 0, 1) == (9, 1, 2)
+
+
+def test_shared_round_returns_none_for_non_partners():
+    """The control. An overlap-based matcher would pair these anyway."""
+    a = _profile((0, 1), (3, 2))
+    b = _profile((4, 5), (6, 7))
+    assert mod.shared_round(a, b, 0, 3) is None
+
+
+def test_shared_round_refuses_a_one_sided_schedule():
+    """A names B, B does not name A: the schedule was misread."""
+    a = _profile((0, 1))
+    b = _profile((0, 9))          # B thinks it talks to 9 at round 0
+    with pytest.raises(SystemExit, match="asymmetric schedule"):
+        mod.shared_round(a, b, 0, 1)
+
+
+def test_shared_round_refuses_an_ambiguous_pairing():
+    """Two rounds with the same partner: which collective is which?"""
+    a = _profile((0, 1), (4, 1))
+    b = _profile((0, 0), (4, 0))
+    with pytest.raises(SystemExit, match="not unique"):
+        mod.shared_round(a, b, 0, 1)
+
+
+# --- compare: indexing and the consistency refusals -----------------------
+
+
+def _k(*start_duration_pairs):
+    """Kernels as (start_ns, duration_ns) -> the (start, end) rows."""
+    return [(start, start + dur) for start, dur in start_duration_pairs]
+
+
+def test_compare_indexes_the_same_collective_on_both_ranks():
+    """Ranks have DIFFERENT rounds-per-fill, so the stride differs.
+
+    A: 2 rounds/fill, shared round at index 1.
+    B: 3 rounds/fill, shared round at index 0.
+    Fill k therefore sits at A[2k+1] and B[3k+0]; pairing by raw call index
+    would compare A[1] with B[1], a different collective entirely.
+    """
+    # 3 fills. The collectives of interest get distinctive durations so a
+    # mis-stride shows up as the WRONG duration, not merely a wrong index.
+    a = _k((0, 1000), (10_000, 5000),       # fill 0: idx1 -> 5000 ns
+           (20_000, 1000), (30_000, 6000),   # fill 1: idx1 -> 6000 ns
+           (40_000, 1000), (50_000, 7000))   # fill 2: idx1 -> 7000 ns
+    b = _k((5_000, 2000), (6_000, 100), (7_000, 100),
+           (25_000, 2500), (26_000, 100), (27_000, 100),
+           (45_000, 3000), (46_000, 100), (47_000, 100))
+    recs = mod.compare(a, b, 2, 3, 1, 0, drop_fills=0)
+    assert [r["fill"] for r in recs] == [0, 1, 2]
+    assert [r["a_dur_us"] for r in recs] == [5.0, 6.0, 7.0]
+    assert [r["b_dur_us"] for r in recs] == [2.0, 2.5, 3.0]
+    # wait estimate = |difference| between the two ranks' time in the SAME
+    # collective.
+    assert [r["wait_estimate_us"] for r in recs] == [3.0, 3.5, 4.0]
+
+
+def test_compare_drops_warmup_fills():
+    a = _k((0, 1000), (10_000, 1000), (20_000, 1000), (30_000, 1000))
+    b = _k((0, 1000), (10_000, 1000), (20_000, 1000), (30_000, 1000))
+    recs = mod.compare(a, b, 1, 1, 0, 0, drop_fills=2)
+    assert [r["fill"] for r in recs] == [2, 3]
+
+
+def test_compare_refuses_partial_fills():
+    """A call count that is not a whole number of fills means the trace and
+    the schedule disagree; pairing would silently shear."""
+    a = _k((0, 1), (10, 1), (20, 1))       # 3 calls, 2 rounds/fill
+    b = _k((0, 1), (10, 1), (20, 1), (30, 1))
+    with pytest.raises(SystemExit, match="not whole fills"):
+        mod.compare(a, b, 2, 2, 0, 0, drop_fills=0)
+
+
+def test_compare_refuses_unequal_fill_counts():
+    """Ranks step in lockstep; different fill counts means different runs."""
+    a = _k((0, 1), (10, 1), (20, 1), (30, 1))   # 2 rounds/fill -> 2 fills
+    b = _k((0, 1), (10, 1), (20, 1))            # 1 round/fill  -> 3 fills
+    with pytest.raises(SystemExit, match="lockstep|halo fills"):
+        mod.compare(a, b, 2, 1, 0, 0, drop_fills=0)
+
+
+# --- trace loading --------------------------------------------------------
+
+
+def _make_sqlite(path, names_and_rows):
+    con = sqlite3.connect(str(path))
+    con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
+    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
+                "(start INTEGER, end INTEGER, demangledName INTEGER)")
+    for sid, (name, rows) in enumerate(names_and_rows):
+        con.execute("INSERT INTO StringIds VALUES (?,?)", (sid, name))
+        for s, e in rows:
+            con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?)",
+                        (s, e, sid))
+    con.commit()
+    con.close()
+
+
+def test_load_kernels_reads_only_the_collective_and_keeps_order(tmp_path):
+    db = tmp_path / "rank_0.sqlite"
+    _make_sqlite(db, [
+        ("ncclDevKernel_SendRecv(x)", [(100, 300), (500, 900)]),
+        ("loop_multiply_fusion", [(0, 50)]),          # must be ignored
+        ("ncclDevKernel_AllGather_RING_LL(y)", [(0, 99999)]),  # ignored
+    ])
+    assert mod.load_kernels(db) == [(100, 300), (500, 900)]
+
+
+def test_load_kernels_raises_when_the_name_matches_nothing(tmp_path):
+    """A renamed NCCL kernel would otherwise yield zero rows everywhere and
+    read as 'no skew' rather than 'no data'."""
+    db = tmp_path / "rank_0.sqlite"
+    _make_sqlite(db, [("some_other_kernel", [(0, 10)])])
+    with pytest.raises(SystemExit, match="no kernel matches"):
+        mod.load_kernels(db)
+
+
+# --- the cross-node refusal ----------------------------------------------
+
+
+def test_rank_node_map_parsed_and_cross_node_pair_refused(tmp_path,
+                                                          monkeypatch):
+    (tmp_path / "_rank_nodes.tsv").write_text(
+        "rank=0\tnode=l50018\tpid=1\n"
+        "rank=1\tnode=l50018\tpid=2\n"
+        "rank=9\tnode=l50033\tpid=3\n")
+    nodes = mod.read_rank_nodes(tmp_path / "_rank_nodes.tsv")
+    assert nodes == {0: "l50018", 1: "l50018", 9: "l50033"}
+
+    # 0 and 9 are on different machines: their CLOCK_MONOTONIC values share
+    # no origin, so a "skew" computed from them would be clock drift.
+    monkeypatch.setattr(sys, "argv", [
+        "skew", "--sqlite-dir", str(tmp_path), "--pairs", "0:9",
+        "--subdivision", "2", "--n-devices", "8"])
+    with pytest.raises(SystemExit, match="spans nodes"):
+        mod.main()
+
+
+def test_missing_rank_node_map_is_refused(tmp_path, monkeypatch):
+    monkeypatch.setattr(sys, "argv", [
+        "skew", "--sqlite-dir", str(tmp_path), "--pairs", "0:1",
+        "--subdivision", "2", "--n-devices", "8"])
+    with pytest.raises(SystemExit, match="_rank_nodes.tsv"):
+        mod.main()

web search: 
web search: site:docs.nvidia.com/cupti "global timer" "timestamp" GPUs ...
web search: 
web search: site:github.com/openxla/xla nccl collective-permute SendRecv implementation GPU ...
web search: 
web search: 
exec
/usr/bin/bash -lc "nl -ba scripts/bench/metadata.py | sed -n '1,125p'
nl -ba scripts/bench/metadata.py | sed -n '300,430p'
rg -n \"git|XLA_FLAGS|NCCL|JAX\" scripts/bench/metadata.py" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 230ms:
     1	"""Shared, self-describing metadata for scaling-benchmark outputs (roadmap item 9).
     2	
     3	SINGLE source of the metadata block that every scaling JSON carries, so that
     4	each bench driver stops rolling its own record.  The goal is that a scaling row
     5	is comparable and *falsifiable* from the record ALONE — a CPU fallback, a
     6	host-staged halo, a replicated (non-SPMD) run, or an f32 ablation can no longer
     7	masquerade as a valid GPU-direct f64 scaling point.
     8	
     9	Every record answers the roadmap's item-9 questions:
    10	  grid, component, resolution, levels; precision + precision knobs; backend,
    11	  rank count, GPU count, devices per rank; decomposition type; MPI / GPU-direct
    12	  settings; solver variant + residual; cells per rank/GPU.
    13	
    14	Scientific descriptors (grid, resolution, solver, ...) are passed by the caller;
    15	runtime facts (backend, device / process count, precision knobs, GPU-direct
    16	mode) are auto-detected from the LIVE process so they cannot be mislabeled.
    17	
    18	Consumers: ``run_levante_gpu_scaling.py``, ``run_cpu_mpi_scaling.py``,
    19	``bench_atm_latlon_spmd_scaling.py``, ``bench_ocean_latlon_spmd_scaling.py``,
    20	``bench_mpas_spmd_scaling.py``, ``bench_ocean_mpi_scaling.py``,
    21	``bench_ocean_gpu_scaling.py`` (and any future bench driver) merge
    22	``scaling_metadata(...)`` under the ``"metadata"`` key of their JSON payload.
    23	Aggregators read ``payload["metadata"]``.
    24	"""
    25	from __future__ import annotations
    26	
    27	import os
    28	import re
    29	from datetime import datetime, timezone
    30	from typing import Any
    31	
    32	#: Bump when the record shape changes so aggregators can branch on it.
    33	#: v2: + ``transport`` / ``virtual_cpu_devices`` / ``launcher`` (anti-fake-
    34	#: scaling audit) — a route-A mpi4jax row, a route-B NCCL row, a gloo/TCP
    35	#: fabric row, and a CPU-virtual-device infra proxy are now distinguishable
    36	#: from the record alone.
    37	METADATA_SCHEMA_VERSION = 2
    38	
    39	#: Halo/collective transports a scaling row may run on.  ``xla-local`` =
    40	#: single-process multi-device SPMD (intra-process XLA collectives);
    41	#: ``none`` = serial.  gloo/TCP multi-node is KNOWN anti-scaling fabric
    42	#: (docs/performance/scaling/spectral_level_shard_cliff.md) — recording it
    43	#: verbatim is what keeps such a row from masquerading as an NCCL result.
    44	TRANSPORTS: tuple[str, ...] = ("mpi4jax", "nccl", "gloo", "xla-local", "none")
    45	
    46	#: Env knobs that change a run's numerics / comparability.  Recorded VERBATIM so
    47	#: an f32 (or TF32) ablation is never silently compared to an f64 baseline.
    48	PRECISION_ENV_KNOBS: tuple[str, ...] = (
    49	    "JAX_ENABLE_X64",
    50	    "LEGOESM_VMIX_F32_SOLVE",
    51	    "LEGOESM_BAROCLINIC_F32",
    52	    "LEGOESM_ENABLE_TF32",
    53	)
    54	
    55	#: The env toggle requesting CUDA-aware (device-direct) mpi4jax halos.
    56	GPU_DIRECT_ENV = "MPI4JAX_USE_CUDA_MPI"
    57	
    58	#: Backends on which the GPU-direct toggle is meaningful.
    59	_GPU_BACKENDS = ("gpu", "cuda", "rocm")
    60	
    61	#: Keys a self-describing scaling record MUST carry with a NON-EMPTY value
    62	#: (fail-fast hygiene).  ``0`` / ``False`` / a populated ``precision_knobs``
    63	#: dict are legal values; only ``None`` / ``""`` fail.  These are the fields
    64	#: without which a row cannot be compared or a fake-scaling run detected.
    65	REQUIRED_KEYS: tuple[str, ...] = (
    66	    "schema_version",
    67	    "grid",
    68	    "component",
    69	    "resolution",
    70	    "n_levels",
    71	    "precision",
    72	    "precision_knobs",
    73	    "backend",
    74	    "decomposition",
    75	    "n_ranks",
    76	    "n_gpus",
    77	    "device_count",
    78	    "process_count",
    79	    "gpu_direct_active",
    80	    "host_staged_halo",
    81	    "transport",
    82	    "virtual_cpu_devices",
    83	    "launcher",
    84	)
    85	
    86	#: Keys that MUST be PRESENT (the roadmap requires the field) but whose value
    87	#: may legitimately be ``None`` — an atmosphere run has no barotropic-solver
    88	#: residual; a single-device run has no partition metrics or cells-per-rank.
    89	PRESENT_KEYS: tuple[str, ...] = (
    90	    "devices_per_rank",
    91	    "cells_per_rank",
    92	    "solver_variant",
    93	    "solver_residual",
    94	    "scaling_kind",
    95	)
    96	
    97	
    98	def _env_flag_true(name: str) -> bool:
    99	    """True iff env var ``name`` is a truthy flag ("1"/"true"/"yes"/"on")."""
   100	    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")
   101	
   102	
   103	# Match the OP-CALL form ``collective-permute(`` / ``collective_permute(`` /
   104	# ``...-start(`` (a paren directly after the op name), NOT bare substrings: the
   105	# COMPILED-HLO config header echoes XLA_FLAGS, so a flag name like
   106	# ``xla_gpu_collective_permute_combine_threshold_bytes=`` (set by #1175's
   107	# MPAS_CP_COMBINE) would false-match a plain substring scan and over-count.
   108	_COLLECTIVE_PERMUTE_RE = re.compile(r"collective[_-]permute(?:[_-]start)?\(")
   109	
   110	
   111	def _count_op_calls(hlo_text: str, rx: "re.Pattern[str]") -> int:
   112	    """Count lines of ``hlo_text`` matching the op-call regex ``rx``.
   113	
   114	    The ``-done``/``_done`` async companion is excluded STRUCTURALLY by the
   115	    regex — ``op(?:[_-]start)?\\(`` cannot match ``op-done(`` (the char after
   116	    the op name is ``-``/``_``, not ``(`` or a ``start`` suffix) — so no
   117	    substring ``"done" not in line`` filter is used: that filter would
   118	    false-drop a legitimate collective whose line merely CONTAINS "done"
   119	    elsewhere (an XLA ``metadata={op_name="…/done_stage/…"}`` tag, a
   120	    ``%done_mass`` SSA name).  The ``\\(`` op-call anchor still keeps a
   121	    config-header ``XLA_FLAGS`` echo (a flag name, no paren) from inflating."""
   122	    return sum(1 for line in hlo_text.splitlines() if rx.search(line))
   123	
   124	
   125	def count_collective_permutes(hlo_text: str) -> int:
   300	    """Snapshot of every precision-affecting env knob (default ``"0"`` = off)."""
   301	    return {k: os.environ.get(k, "0") for k in PRECISION_ENV_KNOBS}
   302	
   303	
   304	def detect_virtual_cpu_devices(backend: str | None = None) -> bool:
   305	    """True when the run's parallelism comes from FORCED virtual CPU devices.
   306	
   307	    ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` (N>1) on a CPU
   308	    backend is the infra-validation mode: it characterizes communication
   309	    overhead and correctness, NEVER hardware scaling.  A row with this flag
   310	    True must not be reported as a device-count speedup (roadmap: "do not
   311	    report CPU virtual devices as speedup").
   312	    """
   313	    b = (backend or detect_backend()).lower()
   314	    if b != "cpu":
   315	        return False
   316	    m = re.search(
   317	        r"xla_force_host_platform_device_count\s*=\s*(\d+)",
   318	        os.environ.get("XLA_FLAGS", ""),
   319	    )
   320	    return bool(m and int(m.group(1)) > 1)
   321	
   322	
   323	def detect_launcher() -> str:
   324	    """Job launcher this process runs under, from launcher-specific env.
   325	
   326	    ``slurm`` | ``pbs-pals`` (Cray PALS, e.g. Derecho ``mpiexec``) | ``pbs`` |
   327	    ``openmpi`` (bare ``mpirun``) | ``none`` (local shell).  Order matters:
   328	    PALS jobs also carry ``PBS_JOBID``, and SLURM steps may export OMPI vars.
   329	    """
   330	    env = os.environ
   331	    if "SLURM_JOB_ID" in env:
   332	        return "slurm"
   333	    if "PALS_RANKID" in env or "PALS_NODEID" in env:
   334	        return "pbs-pals"
   335	    if "PBS_JOBID" in env:
   336	        return "pbs"
   337	    if "OMPI_COMM_WORLD_SIZE" in env or "PMI_SIZE" in env:
   338	        return "openmpi"
   339	    return "none"
   340	
   341	
   342	def resolve_transport(
   343	    transport: str | None,
   344	    *,
   345	    n_ranks: int,
   346	    process_count: int,
   347	    device_count: int,
   348	    backend: str | None = None,
   349	) -> str:
   350	    """Resolve (and validate) the halo/collective transport for the record.
   351	
   352	    Explicit ``transport`` wins (validated against :data:`TRANSPORTS`).
   353	    Auto-resolution when ``None``:
   354	
   355	    - ``n_ranks > process_count``: a route-A MPI world JAX cannot see (each
   356	      rank is a separate single-process JAX) → ``"mpi4jax"``.
   357	    - ``process_count > 1``: route-B multi-controller ``jax.distributed`` →
   358	      ``"nccl"`` on a GPU backend, ``"gloo"`` on CPU (JAX's CPU collective
   359	      transport).
   360	    - ``device_count > 1``: single-process SPMD → ``"xla-local"``.
   361	    - else ``"none"``.
   362	    """
   363	    if transport is not None:
   364	        if transport not in TRANSPORTS:
   365	            raise ValueError(
   366	                f"unknown transport {transport!r}; expected one of "
   367	                f"{TRANSPORTS} (a scaling row's transport must be a known, "
   368	                "comparable fabric)."
   369	            )
   370	        return transport
   371	    if n_ranks > process_count:
   372	        return "mpi4jax"
   373	    if process_count > 1:
   374	        b = (backend or detect_backend()).lower()
   375	        return "nccl" if b in _GPU_BACKENDS else "gloo"
   376	    if device_count > 1:
   377	        return "xla-local"
   378	    return "none"
   379	
   380	
   381	def gpu_direct_mode(
   382	    backend: str | None = None, transport: str = "mpi4jax"
   383	) -> dict[str, Any]:
   384	    """Device-direct MPI configuration for the record.
   385	
   386	    ``gpu_direct_active`` is ``True`` ONLY when a GPU backend is live, the
   387	    ``MPI4JAX_USE_CUDA_MPI`` toggle is set, AND mpi4jax reports CUDA support —
   388	    the exact conjunction that distinguishes a real GPU-direct halo from a
   389	    silently host-staged one.  ``host_staged_halo`` flags the dangerous case: a
   390	    GPU run whose halos are NOT device-direct (a scaling bug hiding as a valid
   391	    row).  On CPU/TPU the toggle is moot, so both are ``False``.
   392	
   393	    Both flags describe the **mpi4jax halo path only**: on any other
   394	    ``transport`` (route-B NCCL, gloo, intra-process ``xla-local``, serial)
   395	    there IS no mpi4jax halo to host-stage, so a GPU row must not be flagged
   396	    ``host_staged_halo`` (codex: an NCCL row would otherwise look like a
   397	    broken MPI row).  The raw request/support facts stay recorded verbatim.
   398	    """
   399	    b = (backend or detect_backend()).lower()
   400	    on_gpu = b in _GPU_BACKENDS
   401	    requested = _env_flag_true(GPU_DIRECT_ENV)
   402	    cuda = mpi4jax_cuda_support()
   403	    device_direct = bool(requested and cuda is True)
   404	    mpi4jax_halo = transport == "mpi4jax"
   405	    return {
   406	        "gpu_direct_requested": requested,
   407	        "mpi4jax_cuda_support": cuda,
   408	        "gpu_direct_active": bool(on_gpu and mpi4jax_halo and device_direct),
   409	        "host_staged_halo": bool(on_gpu and mpi4jax_halo and not device_direct),
   410	    }
   411	
   412	
   413	def scaling_metadata(
   414	    *,
   415	    grid: str,
   416	    component: str,
   417	    resolution: Any,
   418	    n_levels: int,
   419	    precision: str,
   420	    n_ranks: int | None = None,
   421	    n_gpus: int = 0,
   422	    devices_per_rank: int | None = None,
   423	    decomposition: str = "none",
   424	    solver_variant: str = "n/a",
   425	    solver_residual: float | None = None,
   426	    conservation_drift: float | None = None,
   427	    cells_per_rank: int | None = None,
   428	    scaling_kind: str | None = None,
   429	    transport: str | None = None,
   430	    partition_metrics: dict[str, Any] | None = None,
34:#: scaling audit) — a route-A mpi4jax row, a route-B NCCL row, a gloo/TCP
43:#: verbatim is what keeps such a row from masquerading as an NCCL result.
49:    "JAX_ENABLE_X64",
56:GPU_DIRECT_ENV = "MPI4JAX_USE_CUDA_MPI"
87:#: may legitimately be ``None`` — an atmosphere run has no barotropic-solver
105:# COMPILED-HLO config header echoes XLA_FLAGS, so a flag name like
118:    false-drop a legitimate collective whose line merely CONTAINS "done"
121:    config-header ``XLA_FLAGS`` echo (a flag name, no paren) from inflating."""
166:    config-header ``XLA_FLAGS`` echo that merely CONTAINS the op name can never
230:    ablation).  Scalars ``0`` / ``0.0`` / ``False`` are NOT empty (a legitimate
239:def git_sha(short: bool = True) -> str:
240:    """Best-effort git commit SHA of THIS source tree (repro signature).
246:    reproducibility.  Fail-open (``"unknown"``): a missing git binary or
253:        cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
260:            ["git", "status", "--porcelain", "--untracked-files=no"],
270:    """Live JAX backend ("cpu"/"gpu"/"tpu"), or "unknown" if JAX is absent."""
307:    ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` (N>1) on a CPU
318:        os.environ.get("XLA_FLAGS", ""),
355:    - ``n_ranks > process_count``: a route-A MPI world JAX cannot see (each
356:      rank is a separate single-process JAX) → ``"mpi4jax"``.
358:      ``"nccl"`` on a GPU backend, ``"gloo"`` on CPU (JAX's CPU collective
387:    ``MPI4JAX_USE_CUDA_MPI`` toggle is set, AND mpi4jax reports CUDA support —
394:    ``transport`` (route-B NCCL, gloo, intra-process ``xla-local``, serial)
396:    ``host_staged_halo`` (codex: an NCCL row would otherwise look like a
469:    # route-B NCCL / xla-local / serial row (codex finding 2).
517:        "git_sha": git_sha(),

exec
/usr/bin/bash -lc "nl -ba scripts/bench/metadata.py | sed -n '230,270p'
nl -ba scripts/bench/metadata.py | sed -n '430,535p'
nl -ba scripts/bench/bench_mpas_spmd_scaling.py | sed -n '494,560p'" in /work/bd1083/b309178/diffESM/legoesm_pg/legoESM
 succeeded in 288ms:
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
   430	    partition_metrics: dict[str, Any] | None = None,
   431	    timestamp_utc: str | None = None,
   432	    extra: dict[str, Any] | None = None,
   433	) -> dict[str, Any]:
   434	    """Return the complete self-describing metadata block for one scaling row.
   435	
   436	    Parameters
   437	    ----------
   438	    grid, component, resolution, n_levels, precision, decomposition
   439	        Scientific descriptors of the case (caller-supplied).
   440	    n_ranks, n_gpus, devices_per_rank, cells_per_rank
   441	        Parallel layout.  ``devices_per_rank`` defaults to
   442	        ``jax.device_count() // jax.process_count()`` when omitted.
   443	    solver_variant, solver_residual, conservation_drift
   444	        Solver identity + convergence/conservation evidence.  The roadmap
   445	        forbids claiming a solver speedup without recording a residual /
   446	        conservation drift, so these belong in the record.
   447	    scaling_kind
   448	        ``"weak"`` | ``"strong"`` | ``"throughput"`` | ``None`` — keeps
   449	        weak/strong normalization (and single-device throughput-vs-size
   450	        sweeps, which are NOT device-count scaling) from being conflated
   451	        downstream.
   452	    transport
   453	        Halo/collective fabric (see :data:`TRANSPORTS`).  ``None`` →
   454	        auto-resolved by :func:`resolve_transport`; route-A mpi4jax drivers
   455	        that pass ``n_ranks`` explicitly resolve correctly, route-B /
   456	        single-process SPMD auto-detect from the live process.
   457	    partition_metrics
   458	        MPAS / Voronoi partition-quality numbers when available
   459	        (``edge_cut``, ``owned_halo_ratio``, ``cells_per_rank_min/max``,
   460	        ``message_count``).
   461	    extra
   462	        Any other component-specific descriptors.
   463	    """
   464	    backend = detect_backend()
   465	    device_count = _jax_count("device_count", -1)
   466	    process_count = _jax_count("process_count", 1)
   467	    # Resolve the transport BEFORE gpu_direct_mode: host_staged_halo /
   468	    # gpu_direct_active are mpi4jax-halo semantics and must not fire on a
   469	    # route-B NCCL / xla-local / serial row (codex finding 2).
   470	    # ``n_ranks`` = number of MPI processes.  Default to the auto-detected
   471	    # process count so a single-process SPMD run (1 process, N GPUs) records
   472	    # ``n_ranks=1`` (truthful) while the device parallelism lives in
   473	    # ``n_gpus`` / ``device_count``.  Route-A drivers (1 GPU/rank) pass the
   474	    # real rank count explicitly.
   475	    if n_ranks is None:
   476	        n_ranks = process_count
   477	    if devices_per_rank is None and device_count > 0 and process_count > 0:
   478	        devices_per_rank = device_count // process_count
   479	    resolved_transport = resolve_transport(
   480	        transport,
   481	        n_ranks=int(n_ranks),
   482	        process_count=process_count,
   483	        device_count=device_count,
   484	        backend=backend,
   485	    )
   486	
   487	    md: dict[str, Any] = {
   488	        "schema_version": METADATA_SCHEMA_VERSION,
   489	        "timestamp_utc": timestamp_utc or datetime.now(timezone.utc).isoformat(),
   490	        # --- scientific descriptors ---
   491	        "grid": grid,
   492	        "component": component,
   493	        "resolution": resolution,
   494	        "n_levels": n_levels,
   495	        "precision": precision,
   496	        "decomposition": decomposition,
   497	        "solver_variant": solver_variant,
   498	        "solver_residual": solver_residual,
   499	        "conservation_drift": conservation_drift,
   500	        "scaling_kind": scaling_kind,
   501	        # --- parallel layout ---
   502	        "n_ranks": n_ranks,
   503	        "n_gpus": n_gpus,
   504	        "device_count": device_count,
   505	        "process_count": process_count,
   506	        "devices_per_rank": devices_per_rank,
   507	        "cells_per_rank": cells_per_rank,
   508	        # --- runtime facts (auto-detected) ---
   509	        "backend": backend,
   510	        "precision_knobs": precision_knobs(),
   511	        "transport": resolved_transport,
   512	        "virtual_cpu_devices": detect_virtual_cpu_devices(backend),
   513	        "launcher": detect_launcher(),
   514	        "hostname": os.environ.get("HOSTNAME")
   515	        or os.environ.get("SLURMD_NODENAME", ""),
   516	        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
   517	        "git_sha": git_sha(),
   518	    }
   519	    md.update(gpu_direct_mode(backend, transport=resolved_transport))
   520	    if partition_metrics:
   521	        md["partition_metrics"] = partition_metrics
   522	    if extra:
   523	        md["extra"] = extra
   524	    return md
   525	
   526	
   527	def validate_scaling_metadata(
   528	    md: dict[str, Any], *, strict: bool = True
   529	) -> list[str]:
   530	    """Return the list of comparability problems (empty = self-describing).
   531	
   532	    A REQUIRED key that is ``None``/``""`` and a PRESENT key that is absent
   533	    both count.  Raises ``ValueError`` when ``strict`` and any problem exists —
   534	    benchmark hygiene fail-fast, so a record that cannot be compared is never
   535	    silently written.  (``decomposition == "none"`` and ``solver_residual is
   494	    steady = per_step_ms[args.warmup:]
   495	    med = float(np.median(steady))
   496	    rec = dict(
   497	        component="mpas_atm",
   498	        subdivision=args.subdivision, n_devices=nd,
   499	        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
   500	        partition_method=args.partition_method, physics=args.physics,
   501	        # lloyd=0 is the LABELLED synthetic scaling mesh — anti-masquerade:
   502	        # a row without this field could pass as a production-SCVT receipt.
   503	        lloyd_iterations=args.lloyd,
   504	        # Requested vs EFFECTIVE (post-"auto") strategy — a JSONL row
   505	        # saying "auto" would not reveal whether ppermute or allgather
   506	        # was actually measured (codex M3c-2 MINOR).
   507	        halo_strategy_requested=args.halo_strategy,
   508	        halo_strategy_effective=getattr(
   509	            step, "_halo_strategy_effective", "serial"),
   510	        steps=args.steps, dt=dt,
   511	        platform=jax.default_backend(),
   512	        n_processes=jax.process_count(),
   513	        multicontroller=bool(args.multicontroller),
   514	        compile_ms=round(per_step_ms[0], 1),
   515	        steady_median_ms=round(med, 2),
   516	        steady_min_ms=round(float(np.min(steady)), 2),
   517	        per_step_ms=[round(x, 1) for x in per_step_ms],
   518	        cells=int(mesh.nCells) * args.nlev,
   519	        # ppermute round count/step (static compile property; #1113) — the
   520	        # multi-node ceiling is this count x the ~0.11 ms launch floor, so it
   521	        # belongs on every row like the cube benches.
   522	        hlo_collective_permutes=hlo_cp,
   523	        # full per-family census (permute + all-reduce + all-gather + ...) on
   524	        # the SAME compile: exposes any reduction the ico step introduces.
   525	        hlo_collectives=hlo_census,
   526	    )
   527	    # Flat aggregator-compatible identity + metric fields (see the latlon
   528	    # twin): resolution = subdivision level, matching run_cpu_mpi_scaling's
   529	    # icosahedral convention so both lanes land on the same plot curves.
   530	    rec.update(
   531	        grid_type="icosahedral",
   532	        resolution=args.subdivision,
   533	        n_levels=args.nlev,
   534	        mode="strong",  # this bench fixes the mesh and sweeps devices
   535	        precision="float64" if jax.config.jax_enable_x64 else "float32",
   536	        physics_level=args.physics,
   537	        backend=jax.default_backend(),
   538	        **tidy_throughput_fields(
   539	            dt_seconds=dt, time_per_step_ms=med,
   540	            total_cells=int(mesh.nCells) * args.nlev),
   541	    )
   542	    rec["metadata"] = annotate_incomplete(scaling_metadata(
   543	        grid="icosahedral",
   544	        component="atmosphere",
   545	        resolution=f"L{args.subdivision}",
   546	        n_levels=args.nlev,
   547	        precision="float64" if jax.config.jax_enable_x64 else "float32",
   548	        n_gpus=(nd if jax.default_backend() in ("gpu", "cuda", "rocm")
   549	                else 0),
   550	        decomposition="cell_partition" if nd > 1 else "none",
   551	        # cells_per_rank is per PROCESS (n_ranks semantics); the per-device
   552	        # share lives in extra.cells_per_device — a single-process 4-device
   553	        # SPMD run has 1 rank owning ALL cells (codex finding 3).
   554	        cells_per_rank=int(mesh.nCells) * args.nlev
   555	        // max(jax.process_count(), 1),
   556	        scaling_kind="strong",  # this bench fixes the mesh and sweeps devices
   557	        extra={
   558	            "partition_method": args.partition_method,
   559	            "physics": args.physics,
   560	            "steps": args.steps,

codex
## Verdict

The diagnostic cannot make this decision as written. Its central `|duration_A - duration_B| == wait` claim is unproven and has a false-PAYLOAD mode.

- The inference is algebraically incomplete. `D_A-D_B = (E_A-E_B) - (S_A-S_B)`. The tool computes only `|D_A-D_B|` as “wait” and records an unvalidated start delta; it never checks that the paired kernels end together. Without `E_A≈E_B`, duration difference is not arrival skew. See [analyzer:169](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/analyze_nsys_rank_skew.py:169) and [analyzer:175](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/analyze_nsys_rank_skew.py:175).

- Same-node raw SQLite timestamps are not comparable across these independent Nsight sessions. The wrapper starts four separate `nsys profile` sessions ([wrapper:29](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh:29)); node co-location only proves topology. Nsight’s own guidance says report timestamps are relative to each report’s `t=0`, and recommends `nsys export --ts-normalize=true` or a shift for multi-report alignment. [NVIDIA explanation](https://forums.developer.nvidia.com/t/combine-several-files-from-nsight-systems-into-one/252665) Thus `median_start_delta_us` is currently garbage, even for ranks 0 and 1. Cross-node refusal is conservative and sensible, but insufficient.

- “Kernel start == rank arrival” is too strong. CUPTI defines these as kernel *execution* start/end; queued and submitted are separate timestamps. It does not establish NCCL logical-collective arrival or prove that all time after kernel start is peer-arrival wait. [CUPTI Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html) An asymmetric result would be evidence of asymmetric kernel residency, not a quantified peer-delay result, unless you first demonstrate the expected signature with a controlled one-rank delay.

- Equal durations emphatically do not prove payload. Both endpoints can be held by common-mode network/proxy/stream congestion, or both can be delayed by another rank/operation. This is especially material because the Python program executes every global-round `lax.ppermute` in the loop on every device ([sharded_dynamics:2157](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:2157)); “not a pair member” is not a source-level skip. Equal durations therefore have a false-PAYLOAD mode. At best, this method can detect pairwise asymmetry; it cannot exclude shared waiting.

On matching:

- The static schedule is aligned at the source level: both offline scoring and production call the same schedule builder ([sharded_dynamics:1391](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1391), [sharded_dynamics:2392](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:2392)), and `_round_profile` lists only source/target members ([sharded_dynamics:1445](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1445)). But that does **not** prove the CUPTI kernel sequence skips non-members or preserves this order. The analyzer orders by GPU start time, not launch/issue order ([analyzer:89](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/analyze_nsys_rank_skew.py:89)). A wrong stride can still pass divisibility and equal-fill checks.

- This risk is not theoretical: the repository explicitly says the compiled executable may combine or pipeline collective permutes ([metadata:143](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/metadata.py:143), [metadata:214](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/metadata.py:214)). The capture does not pin `XLA_FLAGS`; inherited `LEGOESM_XLA_OVERLAP=1` enables pipelined P2P ([env:126](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/_env.sh:126)). This can produce a visually plausible but sheared capture.

- Rejecting **pure** interval-overlap matching was correct: overlap cannot identify a partner in a synchronized many-rank phase. But discarding it entirely was wrong. After timestamps are normalized, overlap/start-end geometry should be a mandatory consistency check on the schedule-selected pair, and a non-partner should be tested through the actual overlap matcher. The current “control” only observes that the schedule has no edge ([analyzer:251](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/analyze_nsys_rank_skew.py:251)); it does not show an overlap algorithm would falsely match anything.

The 12 tests do not cover these failure modes. They validate synthetic profile arithmetic ([test:25](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_analyze_nsys_rank_skew.py:25)) and synthetic timestamp lists ([test:73](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_analyze_nsys_rank_skew.py:73)), not the compiled GPU executable, rank-local no-op behavior, timestamp normalization, end alignment, asynchronous reorder, or a deliberately delayed partner. The non-partner test has no intervals at all ([test:47](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_analyze_nsys_rank_skew.py:47)).

Capture weaknesses:

- Profiling only selected ranks changes exactly the arrival behavior being diagnosed; no unprofiled timing/control is required. The job can complete with four valid SQLite files yet measure profiler-induced skew ([sbatch:93](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_nsys_skew_capture.sbatch:93)).
- `_rank_nodes.tsv` is appended concurrently by all 64 ranks ([wrapper:19](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh:19)); the analyzer accepts a partial/duplicate map so long as selected rank keys parse ([analyzer:117](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/analyze_nsys_rank_skew.py:117)).
- The analyzer regenerates the schedule later but never checks it against the capture’s JSONL receipt/git state/effective halo strategy. A schedule/code drift can still divide cleanly.

Minimum bar before using this for the call: normalize/calibrate timestamps; pin and record XLA flags and compiled-HLO kernel mapping; require normalized end alignment; add a controlled partner-delay experiment; and downgrade equal-duration output to “no pairwise asymmetry observed,” never “payload-bound.”

NOT READY
tokens used
175,775
## Verdict

The diagnostic cannot make this decision as written. Its central `|duration_A - duration_B| == wait` claim is unproven and has a false-PAYLOAD mode.

- The inference is algebraically incomplete. `D_A-D_B = (E_A-E_B) - (S_A-S_B)`. The tool computes only `|D_A-D_B|` as “wait” and records an unvalidated start delta; it never checks that the paired kernels end together. Without `E_A≈E_B`, duration difference is not arrival skew. See [analyzer:169](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/analyze_nsys_rank_skew.py:169) and [analyzer:175](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/analyze_nsys_rank_skew.py:175).

- Same-node raw SQLite timestamps are not comparable across these independent Nsight sessions. The wrapper starts four separate `nsys profile` sessions ([wrapper:29](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh:29)); node co-location only proves topology. Nsight’s own guidance says report timestamps are relative to each report’s `t=0`, and recommends `nsys export --ts-normalize=true` or a shift for multi-report alignment. [NVIDIA explanation](https://forums.developer.nvidia.com/t/combine-several-files-from-nsight-systems-into-one/252665) Thus `median_start_delta_us` is currently garbage, even for ranks 0 and 1. Cross-node refusal is conservative and sensible, but insufficient.

- “Kernel start == rank arrival” is too strong. CUPTI defines these as kernel *execution* start/end; queued and submitted are separate timestamps. It does not establish NCCL logical-collective arrival or prove that all time after kernel start is peer-arrival wait. [CUPTI Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html) An asymmetric result would be evidence of asymmetric kernel residency, not a quantified peer-delay result, unless you first demonstrate the expected signature with a controlled one-rank delay.

- Equal durations emphatically do not prove payload. Both endpoints can be held by common-mode network/proxy/stream congestion, or both can be delayed by another rank/operation. This is especially material because the Python program executes every global-round `lax.ppermute` in the loop on every device ([sharded_dynamics:2157](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:2157)); “not a pair member” is not a source-level skip. Equal durations therefore have a false-PAYLOAD mode. At best, this method can detect pairwise asymmetry; it cannot exclude shared waiting.

On matching:

- The static schedule is aligned at the source level: both offline scoring and production call the same schedule builder ([sharded_dynamics:1391](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1391), [sharded_dynamics:2392](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:2392)), and `_round_profile` lists only source/target members ([sharded_dynamics:1445](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/packages/core/legoesm/parallel/sharded_dynamics.py:1445)). But that does **not** prove the CUPTI kernel sequence skips non-members or preserves this order. The analyzer orders by GPU start time, not launch/issue order ([analyzer:89](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/analyze_nsys_rank_skew.py:89)). A wrong stride can still pass divisibility and equal-fill checks.

- This risk is not theoretical: the repository explicitly says the compiled executable may combine or pipeline collective permutes ([metadata:143](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/metadata.py:143), [metadata:214](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/metadata.py:214)). The capture does not pin `XLA_FLAGS`; inherited `LEGOESM_XLA_OVERLAP=1` enables pipelined P2P ([env:126](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/_env.sh:126)). This can produce a visually plausible but sheared capture.

- Rejecting **pure** interval-overlap matching was correct: overlap cannot identify a partner in a synchronized many-rank phase. But discarding it entirely was wrong. After timestamps are normalized, overlap/start-end geometry should be a mandatory consistency check on the schedule-selected pair, and a non-partner should be tested through the actual overlap matcher. The current “control” only observes that the schedule has no edge ([analyzer:251](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/analyze_nsys_rank_skew.py:251)); it does not show an overlap algorithm would falsely match anything.

The 12 tests do not cover these failure modes. They validate synthetic profile arithmetic ([test:25](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_analyze_nsys_rank_skew.py:25)) and synthetic timestamp lists ([test:73](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_analyze_nsys_rank_skew.py:73)), not the compiled GPU executable, rank-local no-op behavior, timestamp normalization, end alignment, asynchronous reorder, or a deliberately delayed partner. The non-partner test has no intervals at all ([test:47](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/tests/bench/test_analyze_nsys_rank_skew.py:47)).

Capture weaknesses:

- Profiling only selected ranks changes exactly the arrival behavior being diagnosed; no unprofiled timing/control is required. The job can complete with four valid SQLite files yet measure profiler-induced skew ([sbatch:93](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/mpas_nsys_skew_capture.sbatch:93)).
- `_rank_nodes.tsv` is appended concurrently by all 64 ranks ([wrapper:19](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/cluster/scaling_levante/_nsys_rank_wrapper.sh:19)); the analyzer accepts a partial/duplicate map so long as selected rank keys parse ([analyzer:117](/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/scripts/bench/analyze_nsys_rank_skew.py:117)).
- The analyzer regenerates the schedule later but never checks it against the capture’s JSONL receipt/git state/effective halo strategy. A schedule/code drift can still divide cleanly.

Minimum bar before using this for the call: normalize/calibrate timestamps; pin and record XLA flags and compiled-HLO kernel mapping; require normalized end alignment; add a controlled partner-delay experiment; and downgrade equal-duration output to “no pairwise asymmetry observed,” never “payload-bound.”

NOT READY
